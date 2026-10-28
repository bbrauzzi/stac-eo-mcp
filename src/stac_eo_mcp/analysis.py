"""Tool workflows: scene -> window read -> reflectance -> index -> stats/files."""

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pystac
import xarray as xr
from shapely.geometry import box, shape

from stac_eo_mcp import catalog, raster
from stac_eo_mcp.config import CatalogName, Settings
from stac_eo_mcp.errors import InvalidInputError, StacMcpError
from stac_eo_mcp.indices import INDEX_DEFINITIONS, compute_index, compute_stats, to_reflectance
from stac_eo_mcp.models import (
    BBox,
    DateRange,
    IndexResult,
    IndexStats,
    SaveFormat,
    SkippedScene,
    SpectralIndex,
    TimeSeriesPoint,
    TimeSeriesResult,
    TimeSeriesSummary,
)

log = logging.getLogger(__name__)

RESOLUTION_M = 10.0
LOW_VALID_PCT = 20.0
# Metadata-only search pool for time series; pixels are read for max_scenes at most.
SEARCH_POOL = 500
READ_WORKERS = 4


# --- Single scene ---------------------------------------------------------


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


def index_on_item(
    catalog_name: CatalogName, item: pystac.Item, index: SpectralIndex, bbox: BBox
) -> tuple[xr.DataArray, IndexStats]:
    """Read the bbox window of one scene and compute the masked index."""
    check_overlap(item, bbox)
    keys = catalog.band_asset_keys(catalog_name, item, [*INDEX_DEFINITIONS[index].bands, "scl"])
    sign = catalog.get_config(catalog_name).sign
    ds = raster.load_window(item, keys, bbox, resolution=RESOLUTION_M, sign=sign)
    index_da = index_from_dataset(ds, item, keys, index)
    return index_da, compute_stats(index_da.values)


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
    index_da, stats = index_on_item(catalog_name, item, index, bbox)

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


# --- Time series ----------------------------------------------------------


def _day(item: pystac.Item) -> str:
    return item.datetime.date().isoformat() if item.datetime else "unknown"


def one_per_day(items: list[pystac.Item], bbox: BBox) -> list[pystac.Item]:
    """Keep one scene per acquisition day, oldest first.

    Areas on a tile overlap are covered by two items per pass. Pick the one
    whose footprint covers most of the bbox, then the lower cloud cover.
    """
    target = box(*bbox.as_list())
    by_day: dict[str, list[pystac.Item]] = defaultdict(list)
    for item in items:
        by_day[_day(item)].append(item)

    def rank(item: pystac.Item) -> tuple[float, float]:
        covered = shape(item.geometry).intersection(target).area if item.geometry else 0.0
        cloud = item.properties.get("eo:cloud_cover", 100.0)
        return (-round(covered / target.area, 3), cloud)

    return [min(group, key=rank) for _, group in sorted(by_day.items())]


def sample_evenly(items: list[pystac.Item], n: int) -> list[pystac.Item]:
    """Pick n items spread across the list, always keeping first and last."""
    if len(items) <= n:
        return items
    idx = np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))
    return [items[i] for i in idx]


def time_series(
    settings: Settings,
    catalog_name: CatalogName,
    index: SpectralIndex,
    bbox: BBox,
    date_range: DateRange,
    max_cloud_cover: float | None = None,
) -> TimeSeriesResult:
    bbox.check_area(settings.max_area_km2)
    cfg = catalog.get_config(catalog_name)
    found = catalog.search_items(
        catalog_name, cfg.s2_collection, bbox, date_range, max_cloud_cover, limit=SEARCH_POOL
    )
    daily = one_per_day(found, bbox)
    selected = sample_evenly(daily, settings.max_scenes)
    log.info("time_series: %d scenes, %d dates, reading %d", len(found), len(daily), len(selected))

    def analyse(item: pystac.Item) -> TimeSeriesPoint | SkippedScene:
        try:
            _, stats = index_on_item(catalog_name, item, index, bbox)
        except StacMcpError as exc:
            return SkippedScene(date=_day(item), item_id=item.id, reason=exc.message)
        except Exception as exc:  # one bad scene must not sink the series
            log.warning("time_series: %s failed: %s", item.id, exc)
            return SkippedScene(date=_day(item), item_id=item.id, reason=f"read error: {exc}")
        if stats.mean is None or stats.valid_pixel_pct < LOW_VALID_PCT:
            return SkippedScene(
                date=_day(item),
                item_id=item.id,
                reason=f"only {stats.valid_pixel_pct:.1f}% clear pixels over the bbox",
            )
        return TimeSeriesPoint(
            date=_day(item),
            item_id=item.id,
            mean=stats.mean,
            std=stats.std,
            valid_pixel_pct=stats.valid_pixel_pct,
        )

    with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
        results = list(pool.map(analyse, selected))

    points = [r for r in results if isinstance(r, TimeSeriesPoint)]
    skipped = [r for r in results if isinstance(r, SkippedScene)]
    return TimeSeriesResult(
        index=index,
        bbox=bbox.as_list(),
        date_range=f"{date_range.start}/{date_range.end}",
        scenes_found=len(found),
        dates_analysed=len(selected),
        points=points,
        skipped=skipped,
        summary=_summarize(points),
        note=_series_note(len(daily), len(selected), points, settings.max_scenes),
    )


def _summarize(points: list[TimeSeriesPoint]) -> TimeSeriesSummary | None:
    if not points:
        return None
    return TimeSeriesSummary(
        mean_of_means=round(float(np.mean([p.mean for p in points])), 4),
        min=min(points, key=lambda p: p.mean),
        max=max(points, key=lambda p: p.mean),
        change=round(points[-1].mean - points[0].mean, 4),
    )


def _series_note(
    n_dates: int, n_selected: int, points: list[TimeSeriesPoint], max_scenes: int
) -> str | None:
    notes = []
    if n_selected < n_dates:
        notes.append(
            f"{n_dates} acquisition dates matched; {n_selected} were sampled evenly "
            f"(limit {max_scenes}). Narrow the date range for every date."
        )
    if not points:
        notes.append(
            "No date had enough clear pixels. Raise max_cloud_cover, widen the date range "
            "or move the bbox."
        )
    elif len(points) < 3:
        notes.append("Fewer than 3 usable dates: treat any trend with caution.")
    return " ".join(notes) or None
