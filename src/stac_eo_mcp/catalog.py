"""STAC catalog access: clients, search, item lookup and Sentinel-2 band mapping.

Supported catalogs:
- Microsoft Planetary Computer (assets are signed with planetary-computer)
- Element84 Earth Search (public, no signing)
"""

import difflib
import logging
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import planetary_computer
import pystac
from pystac_client import Client
from pystac_client.exceptions import APIError
from shapely.geometry import mapping, shape

from stac_eo_mcp.config import CatalogName
from stac_eo_mcp.errors import CatalogError, DataUnavailableError, InvalidInputError, NoScenesError
from stac_eo_mcp.indices import S2_BASELINE_OFFSET_DN, S2_SCALE
from stac_eo_mcp.models import (
    AssetInfo,
    BBox,
    CollectionSummary,
    DateRange,
    SceneDetails,
    SceneSummary,
)

log = logging.getLogger(__name__)

# Earth Search items trigger a harmless pystac storage-extension migration warning.
warnings.filterwarnings("ignore", message="Could not parse bucket/account from href")

# Canonical band name -> asset key, per catalog.
CANONICAL_BANDS = ("blue", "green", "red", "nir", "swir16", "swir22", "scl")


@dataclass(frozen=True)
class CatalogConfig:
    name: CatalogName
    url: str
    title: str
    s2_collection: str
    band_assets: dict[str, str]
    preview_asset: str
    sign: bool = False


CATALOGS: dict[CatalogName, CatalogConfig] = {
    CatalogName.PLANETARY_COMPUTER: CatalogConfig(
        name=CatalogName.PLANETARY_COMPUTER,
        url="https://planetarycomputer.microsoft.com/api/stac/v1",
        title="Microsoft Planetary Computer",
        s2_collection="sentinel-2-l2a",
        band_assets={
            "blue": "B02",
            "green": "B03",
            "red": "B04",
            "nir": "B08",
            "swir16": "B11",
            "swir22": "B12",
            "scl": "SCL",
        },
        preview_asset="rendered_preview",
        sign=True,
    ),
    CatalogName.EARTH_SEARCH: CatalogConfig(
        name=CatalogName.EARTH_SEARCH,
        url="https://earth-search.aws.element84.com/v1",
        title="Element84 Earth Search",
        s2_collection="sentinel-2-c1-l2a",
        band_assets={b: b for b in CANONICAL_BANDS},
        preview_asset="thumbnail",
    ),
}

# Short names an agent is likely to use for Sentinel-2 L2A.
S2_ALIASES = {"sentinel-2", "sentinel2", "s2", "sentinel-2-l2a", "s2-l2a", "sentinel-2-c1-l2a"}

# Item properties worth showing an LLM (the rest is noise).
DETAIL_PROPERTIES = (
    "datetime",
    "platform",
    "constellation",
    "eo:cloud_cover",
    "s2:mgrs_tile",
    "mgrs:utm_zone",
    "proj:code",
    "proj:epsg",
    "s2:processing_baseline",
    "s2:nodata_pixel_percentage",
    "s2:vegetation_percentage",
    "s2:water_percentage",
    "s2:snow_ice_percentage",
    "s2:high_proba_clouds_percentage",
    "s2:cloud_shadow_percentage",
    "sat:relative_orbit",
    "sat:orbit_state",
    "view:sun_elevation",
)


def get_config(catalog: CatalogName | str) -> CatalogConfig:
    try:
        return CATALOGS[CatalogName(catalog)]
    except ValueError:
        options = ", ".join(c.value for c in CatalogName)
        raise InvalidInputError(
            f"Unknown catalog '{catalog}'.", hint=f"Use one of: {options}."
        ) from None


@lru_cache
def get_client(catalog: CatalogName) -> Client:
    cfg = get_config(catalog)
    modifier = planetary_computer.sign_inplace if cfg.sign else None
    try:
        log.info("Opening STAC catalog %s", cfg.url)
        return Client.open(cfg.url, modifier=modifier)
    except Exception as exc:
        raise CatalogError(
            f"Could not reach {cfg.title} ({cfg.url}): {exc}",
            hint="Check the network connection, or try the other catalog.",
        ) from exc


# --- Collections ----------------------------------------------------------


@lru_cache
def _collections(catalog: CatalogName) -> tuple[CollectionSummary, ...]:
    client = get_client(catalog)
    try:
        cols = list(client.get_collections())
    except APIError as exc:
        raise CatalogError(f"Listing collections failed: {exc}") from exc
    return tuple(
        CollectionSummary(id=c.id, title=c.title, description=_first_sentence(c.description))
        for c in sorted(cols, key=lambda c: c.id)
    )


def list_collections(catalog: CatalogName) -> list[CollectionSummary]:
    return list(_collections(catalog))


def resolve_collection(catalog: CatalogName, collection: str) -> str:
    """Map S2 aliases to the catalog's collection and check that it exists."""
    cfg = get_config(catalog)
    collection = collection.strip()
    if collection.lower() in S2_ALIASES:
        return cfg.s2_collection
    ids = [c.id for c in _collections(catalog)]
    if collection in ids:
        return collection
    close = difflib.get_close_matches(collection, ids, n=3, cutoff=0.5)
    hint = f"Did you mean: {', '.join(close)}?" if close else "Call list_collections to see ids."
    raise InvalidInputError(f"Collection '{collection}' is not in {cfg.title}.", hint=hint)


# --- Search and lookup ----------------------------------------------------


def search_items(
    catalog: CatalogName,
    collection: str,
    bbox: BBox,
    date_range: DateRange,
    max_cloud_cover: float | None = None,
    limit: int = 10,
) -> list[pystac.Item]:
    """Search scenes, newest first. Raises NoScenesError if none match."""
    collection = resolve_collection(catalog, collection)
    params: dict[str, Any] = {
        "collections": [collection],
        "bbox": bbox.as_list(),
        "datetime": date_range.to_stac(),
        "max_items": limit,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }
    if max_cloud_cover is not None:
        params["query"] = {"eo:cloud_cover": {"lte": max_cloud_cover}}
    try:
        items = list(get_client(catalog).search(**params).items())
    except APIError as exc:
        raise CatalogError(f"Search failed on {get_config(catalog).title}: {exc}") from exc
    if not items:
        raise _no_scenes(collection, date_range, max_cloud_cover)
    items.sort(key=lambda i: i.datetime or 0, reverse=True)
    return items


def _no_scenes(collection: str, dr: DateRange, max_cc: float | None) -> NoScenesError:
    if max_cc is not None:
        msg = (
            f"No {collection} scenes with cloud cover <= {max_cc:g}% "
            f"between {dr.start} and {dr.end} for this bbox."
        )
        hint = "Raise max_cloud_cover or widen the date range."
    else:
        msg = f"No {collection} scenes between {dr.start} and {dr.end} for this bbox."
        hint = "Widen the date range, or check the collection covers this area and period."
    return NoScenesError(msg, hint=hint)


def get_item(catalog: CatalogName, collection: str, item_id: str) -> pystac.Item:
    collection = resolve_collection(catalog, collection)
    try:
        items = list(
            get_client(catalog).search(collections=[collection], ids=[item_id], max_items=1).items()
        )
    except APIError as exc:
        raise CatalogError(f"Looking up '{item_id}' failed: {exc}") from exc
    if not items:
        raise NoScenesError(
            f"Scene '{item_id}' was not found in {collection} on {get_config(catalog).title}.",
            hint="Use an id returned by search_scenes, with the same catalog.",
        )
    return items[0]


# --- Summaries ------------------------------------------------------------


def summarize_item(item: pystac.Item, catalog: CatalogName) -> SceneSummary:
    cfg = get_config(catalog)
    preview = item.assets.get(cfg.preview_asset)
    return SceneSummary(
        id=item.id,
        datetime=_iso(item),
        cloud_cover=_round(item.properties.get("eo:cloud_cover")),
        bbox=[round(v, 5) for v in item.bbox] if item.bbox else None,
        footprint=simplify_geometry(item.geometry),
        preview_url=preview.href if preview else None,
    )


def describe_item(item: pystac.Item) -> SceneDetails:
    props = {k: item.properties[k] for k in DETAIL_PROPERTIES if item.properties.get(k) is not None}
    props.pop("datetime", None)
    assets = [
        AssetInfo(
            key=key,
            title=a.title,
            media_type=a.media_type,
            roles=a.roles or [],
            gsd=a.extra_fields.get("gsd"),
        )
        for key, a in sorted(item.assets.items())
    ]
    return SceneDetails(
        id=item.id,
        collection=item.collection_id or "",
        datetime=_iso(item),
        bbox=[round(v, 5) for v in item.bbox] if item.bbox else None,
        properties=props,
        assets=assets,
    )


def simplify_geometry(geom: dict[str, Any] | None, tolerance: float = 0.001) -> dict | None:
    """Simplify (~100 m) and round a GeoJSON geometry to keep responses small."""
    if not geom:
        return None
    simple = shape(geom).simplify(tolerance, preserve_topology=True)
    return _round_coords(mapping(simple))


def _round_coords(obj: Any, nd: int = 4) -> Any:
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, list | tuple):
        return [_round_coords(o, nd) for o in obj]
    if isinstance(obj, dict):
        return {k: _round_coords(v, nd) for k, v in obj.items()}
    return obj


# --- Sentinel-2 bands -----------------------------------------------------


@dataclass(frozen=True)
class BandScaling:
    scale: float
    offset_dn: float  # added to DN before scaling


def band_asset_keys(catalog: CatalogName, item: pystac.Item, bands: list[str]) -> dict[str, str]:
    """Canonical band -> asset key, checking that the item has each asset."""
    cfg = get_config(catalog)
    keys = {b: cfg.band_assets[b] for b in bands}
    missing = [f"{b} ({k})" for b, k in keys.items() if k not in item.assets]
    if missing:
        raise DataUnavailableError(
            f"Scene {item.id} lacks assets {missing}.",
            hint=f"Only Sentinel-2 L2A scenes ({cfg.s2_collection}) are supported for indices.",
        )
    return keys


def band_scaling(item: pystac.Item, asset_key: str) -> BandScaling:
    """DN -> reflectance parameters for a Sentinel-2 L2A band.

    Prefer raster:bands scale/offset from the asset (Earth Search). Otherwise
    fall back to the processing baseline (Planetary Computer): baseline 04.00
    and later store DN with a +1000 offset.
    """
    raster_bands = item.assets[asset_key].extra_fields.get("raster:bands") or []
    if raster_bands and "scale" in raster_bands[0]:
        scale = float(raster_bands[0]["scale"])
        offset = float(raster_bands[0].get("offset", 0.0))
        return BandScaling(scale=scale, offset_dn=offset / scale)
    baseline = item.properties.get("s2:processing_baseline")
    offset_dn = S2_BASELINE_OFFSET_DN if _baseline_ge(baseline, 4.0) else 0.0
    return BandScaling(scale=S2_SCALE, offset_dn=offset_dn)


def _baseline_ge(baseline: Any, threshold: float) -> bool:
    try:
        return float(baseline) >= threshold
    except (TypeError, ValueError):
        # Unknown baseline: assume modern data, which is the common case since 2022.
        return True


# --- Helpers --------------------------------------------------------------


def _iso(item: pystac.Item) -> str | None:
    return item.datetime.isoformat().replace("+00:00", "Z") if item.datetime else None


def _round(v: Any, nd: int = 2) -> float | None:
    return round(float(v), nd) if v is not None else None


def _first_sentence(text: str | None, max_len: int = 200) -> str | None:
    if not text:
        return None
    text = " ".join(text.split())
    cut = text.find(". ")
    if 0 < cut < max_len:
        return text[: cut + 1]
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"
