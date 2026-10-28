"""Tool workflows: scene -> window read -> reflectance -> index -> stats/files."""

import logging

import pystac
import xarray as xr
from shapely.geometry import box, shape

from stac_eo_mcp import catalog, raster
from stac_eo_mcp.config import CatalogName, Settings
from stac_eo_mcp.errors import InvalidInputError
from stac_eo_mcp.indices import INDEX_DEFINITIONS, compute_index, compute_stats, to_reflectance
from stac_eo_mcp.models import BBox, IndexResult, SaveFormat, SpectralIndex

log = logging.getLogger(__name__)

RESOLUTION_M = 10.0
LOW_VALID_PCT = 20.0


def check_overlap(item: pystac.Item, bbox: BBox) -> None:
    if item.geometry and not shape(item.geometry).intersects(box(*bbox.as_list())):
        raise InvalidInputError(
            f"The bbox does not overlap scene {item.id}.",
            hint="Use search_scenes with this bbox to find scenes that cover it.",
        )


def index_from_dataset(
    ds: xr.Dataset, item: pystac.Item, asset_keys: dict[str, str], index: SpectralIndex
) -> xr.DataArray:
    """Turn a raw-DN window into a masked index DataArray on the same grid."""
    definition = INDEX_DEFINITIONS[index]
    bands = {}
    for name in definition.bands:
        scaling = catalog.band_scaling(item, asset_keys[name])
        bands[name] = to_reflectance(ds[name].values, scaling.offset_dn, scaling.scale)
    scl = ds["scl"].values if "scl" in ds else None
    values = compute_index(index, bands, scl=scl)
    template = ds[definition.a]
    return xr.DataArray(values, coords=template.coords, dims=template.dims, name=index.value)


def compute_index_for_item(
    settings: Settings,
    catalog_name: CatalogName,
    item_id: str,
    index: SpectralIndex,
    bbox: BBox,
    save: SaveFormat = SaveFormat.NONE,
) -> IndexResult:
    bbox.check_area(settings.max_area_km2)
    cfg = catalog.get_config(catalog_name)
    item = catalog.get_item(catalog_name, cfg.s2_collection, item_id)
    check_overlap(item, bbox)

    needed = [*INDEX_DEFINITIONS[index].bands, "scl"]
    keys = catalog.band_asset_keys(catalog_name, item, needed)
    ds = raster.load_window(item, keys, bbox, resolution=RESOLUTION_M, sign=cfg.sign)
    index_da = index_from_dataset(ds, item, keys, index)
    stats = compute_stats(index_da.values)

    outputs: dict[str, str] = {}
    if save is not SaveFormat.NONE and stats.valid_pixels:
        stem = settings.output_dir / raster.output_stem(item.id, index, bbox)
        if save in (SaveFormat.PNG, SaveFormat.BOTH):
            outputs["png"] = str(raster.save_png(index_da.values, index, stem.with_suffix(".png")))
        if save in (SaveFormat.GEOTIFF, SaveFormat.BOTH):
            outputs["geotiff"] = str(raster.save_geotiff(index_da, stem.with_suffix(".tif")))

    return IndexResult(
        item_id=item.id,
        index=index,
        datetime=catalog.summarize_item(item, catalog_name).datetime,
        bbox=bbox.as_list(),
        stats=stats,
        resolution_m=RESOLUTION_M,
        outputs=outputs,
        note=_quality_note(stats.valid_pixel_pct),
    )


def _quality_note(valid_pct: float) -> str | None:
    if valid_pct == 0:
        return (
            "All pixels are masked (cloud, shadow or no data) over this bbox, so no statistics "
            "and no files. Try another scene from search_scenes with a lower max_cloud_cover."
        )
    if valid_pct < LOW_VALID_PCT:
        return (
            f"Only {valid_pct:.1f}% of pixels are clear; the statistics may not be "
            "representative. Consider another scene."
        )
    return None
