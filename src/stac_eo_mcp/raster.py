"""Windowed raster reads with odc-stac and PNG/GeoTIFF export."""

import hashlib
import logging
from functools import cache
from pathlib import Path

import matplotlib as mpl
import numpy as np
import odc.stac
import planetary_computer
import pystac
import xarray as xr

from stac_eo_mcp.errors import CatalogError, DataUnavailableError
from stac_eo_mcp.models import BBox, SpectralIndex

log = logging.getLogger(__name__)

mpl.use("Agg")  # headless, never open a window
import matplotlib.pyplot as plt  # noqa: E402

COLORMAPS = {
    SpectralIndex.NDVI: "RdYlGn",
    SpectralIndex.NDWI: "RdBu",
    SpectralIndex.NBR: "RdYlGn",
}


@cache
def _configure_gdal() -> None:
    # Sensible GDAL settings for reading COGs over HTTP (fewer requests).
    odc.stac.configure_rio(cloud_defaults=True)


def load_window(
    item: pystac.Item,
    asset_keys: dict[str, str],
    bbox: BBox,
    resolution: float = 10.0,
    sign: bool = False,
) -> xr.Dataset:
    """Read only the bbox window of the given assets, on one 10 m UTM grid.

    Returns a Dataset with canonical band names as variables (raw DN, y/x dims).
    Nearest resampling keeps SCL classes and raw DN intact (B12/SCL are 20 m).
    """
    _configure_gdal()
    if sign:
        # Re-sign so a long session never hits an expired SAS token.
        item = planetary_computer.sign(item)
    crs = item.properties.get("proj:code") or (
        f"EPSG:{item.properties['proj:epsg']}" if "proj:epsg" in item.properties else "utm"
    )
    try:
        ds = odc.stac.load(
            [item],
            bands=list(asset_keys.values()),
            bbox=tuple(bbox.as_list()),
            crs=crs,
            resolution=resolution,
            resampling="nearest",
            # Sentinel-2 L2A is uint16 DN (SCL fits too); fill value 0 = no data.
            # Without this, catalogs lacking raster:bands load as float32.
            dtype="uint16",
            groupby="id",
        )
    except Exception as exc:
        raise CatalogError(
            f"Reading pixels for {item.id} failed: {exc}",
            hint="Retry; if it persists, try another scene or catalog.",
        ) from exc
    ds = ds.isel(time=0, drop=True) if "time" in ds.dims else ds
    ds = ds.rename({v: k for k, v in asset_keys.items()})
    if ds.sizes.get("x", 0) == 0 or ds.sizes.get("y", 0) == 0:
        raise DataUnavailableError(
            f"The bbox produced an empty window on {item.id}.",
            hint="Use a bbox at least a few pixels (10 m) wide.",
        )
    log.info("Loaded %s %s window %sx%s", item.id, list(asset_keys), ds.sizes["x"], ds.sizes["y"])
    return ds


def output_stem(item_id: str, index: SpectralIndex, bbox: BBox) -> str:
    digest = hashlib.sha1(",".join(f"{v:.5f}" for v in bbox.as_list()).encode()).hexdigest()[:8]
    return f"{item_id}_{index.value}_{digest}"


def save_png(values: np.ndarray, index: SpectralIndex, path: Path) -> Path:
    """Colour-mapped PNG in [-1, 1]; masked pixels are transparent."""
    cmap = plt.get_cmap(COLORMAPS[index]).with_extremes(bad=(0, 0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.ma.masked_invalid(values), cmap=cmap, vmin=-1, vmax=1)
    return path


def save_geotiff(index_da: xr.DataArray, path: Path) -> Path:
    """Float32 GeoTIFF with NaN as no-data, keeping the source CRS/transform."""
    import rioxarray  # noqa: F401  (registers the .rio accessor)

    path.parent.mkdir(parents=True, exist_ok=True)
    da = index_da.astype("float32")
    da = da.rio.write_nodata(np.nan, encoded=False)
    da.rio.to_raster(path, compress="deflate", tiled=True)
    return path
