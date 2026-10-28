"""Spectral indices, Sentinel-2 cloud masking and summary statistics.

Pure numpy: callers pass band arrays already read from the raster, so this
module is fully testable with synthetic data.

Band names here are canonical (Earth Search style: red, nir, green, swir22, scl).
The catalog module maps them to each catalog's asset keys.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from stac_eo_mcp.errors import DataUnavailableError
from stac_eo_mcp.models import IndexStats, SpectralIndex

# Sentinel-2 L2A reflectance = (DN + offset) / 10000. DN 0 is no-data.
S2_SCALE = 1e-4
S2_NODATA = 0
# Processing baseline 04.00 (from 25 Jan 2022) adds an offset of -1000 DN.
S2_BASELINE_OFFSET_DN = -1000

# Scene Classification Layer (SCL) classes.
SCL_CLASSES = {
    0: "no data",
    1: "saturated or defective",
    2: "dark area pixels",
    3: "cloud shadows",
    4: "vegetation",
    5: "not vegetated",
    6: "water",
    7: "unclassified",
    8: "cloud medium probability",
    9: "cloud high probability",
    10: "thin cirrus",
    11: "snow or ice",
}
# Kept pixels: vegetation, not vegetated, water, unclassified, snow/ice.
DEFAULT_VALID_SCL = frozenset({4, 5, 6, 7, 11})


@dataclass(frozen=True)
class IndexDefinition:
    index: SpectralIndex
    name: str
    formula: str
    s2_formula: str
    a: str  # (a - b) / (a + b)
    b: str
    use: str

    @property
    def bands(self) -> tuple[str, str]:
        return (self.a, self.b)


INDEX_DEFINITIONS: dict[SpectralIndex, IndexDefinition] = {
    SpectralIndex.NDVI: IndexDefinition(
        SpectralIndex.NDVI,
        "Normalized Difference Vegetation Index",
        "(NIR - Red) / (NIR + Red)",
        "(B08 - B04) / (B08 + B04)",
        "nir",
        "red",
        "Vegetation vigour. Bare soil ~0.1-0.2, healthy vegetation > 0.6, water < 0.",
    ),
    SpectralIndex.NDWI: IndexDefinition(
        SpectralIndex.NDWI,
        "Normalized Difference Water Index (McFeeters 1996)",
        "(Green - NIR) / (Green + NIR)",
        "(B03 - B08) / (B03 + B08)",
        "green",
        "nir",
        "Open water. Water > 0, vegetation and soil < 0.",
    ),
    SpectralIndex.NBR: IndexDefinition(
        SpectralIndex.NBR,
        "Normalized Burn Ratio",
        "(NIR - SWIR2) / (NIR + SWIR2)",
        "(B08 - B12) / (B08 + B12)",
        "nir",
        "swir22",
        "Burn severity. Compare pre/post fire: dNBR = NBR_pre - NBR_post (> 0.27 moderate-high).",
    ),
}


def to_reflectance(
    dn: np.ndarray, offset_dn: float = 0.0, scale: float = S2_SCALE, nodata: int = S2_NODATA
) -> np.ndarray:
    """Convert digital numbers to surface reflectance; no-data becomes NaN."""
    dn = np.asarray(dn)
    refl = (dn.astype(np.float32) + np.float32(offset_dn)) * np.float32(scale)
    refl[dn == nodata] = np.nan
    return refl


def scl_mask(scl: np.ndarray, valid_classes: Iterable[int] = DEFAULT_VALID_SCL) -> np.ndarray:
    """True where the SCL class is in valid_classes (clear pixel)."""
    return np.isin(np.asarray(scl), list(valid_classes))


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), NaN where undefined, clipped to [-1, 1]."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / denom
    out[~np.isfinite(out) | (denom == 0)] = np.nan
    # Slightly negative reflectances (dark water after offset) can push
    # the ratio outside the theoretical range.
    return np.clip(out, -1.0, 1.0)


def compute_index(
    index: SpectralIndex,
    bands: Mapping[str, np.ndarray],
    scl: np.ndarray | None = None,
    valid_classes: Iterable[int] = DEFAULT_VALID_SCL,
) -> np.ndarray:
    """Compute an index from reflectance bands, masking with SCL if given.

    `bands` must hold reflectance (not DN) for the index's two bands.
    Returns float32 with NaN for masked or undefined pixels.
    """
    definition = INDEX_DEFINITIONS[index]
    missing = [b for b in definition.bands if b not in bands]
    if missing:
        raise DataUnavailableError(
            f"Bands {missing} are required for {index.value.upper()} but were not loaded.",
            hint="Check the scene's assets with get_scene_details.",
        )
    a, b = bands[definition.a], bands[definition.b]
    if np.shape(a) != np.shape(b):
        raise ValueError(f"band shapes differ: {np.shape(a)} vs {np.shape(b)}")
    out = normalized_difference(a, b)
    if scl is not None:
        if np.shape(scl) != out.shape:
            raise ValueError(f"SCL shape {np.shape(scl)} differs from bands {out.shape}")
        out[~scl_mask(scl, valid_classes)] = np.nan
    return out


def compute_stats(values: np.ndarray, decimals: int = 4) -> IndexStats:
    """Summary statistics over finite pixels."""
    values = np.asarray(values)
    total = int(values.size)
    finite = values[np.isfinite(values)]
    valid = int(finite.size)
    pct = round(100.0 * valid / total, 2) if total else 0.0
    if valid == 0:
        return IndexStats(
            min=None, max=None, mean=None, std=None,
            valid_pixel_pct=pct, valid_pixels=0, total_pixels=total,
        )  # fmt: skip

    def r(x: np.floating) -> float:
        return round(float(x), decimals)

    return IndexStats(
        min=r(finite.min()),
        max=r(finite.max()),
        mean=r(finite.mean()),
        std=r(finite.std()),
        valid_pixel_pct=pct,
        valid_pixels=valid,
        total_pixels=total,
    )


def describe_indices() -> list[dict[str, object]]:
    """Index catalogue for the MCP resource."""
    return [
        {
            "index": d.index.value,
            "name": d.name,
            "formula": d.formula,
            "sentinel2_formula": d.s2_formula,
            "bands": list(d.bands),
            "range": [-1, 1],
            "interpretation": d.use,
        }
        for d in INDEX_DEFINITIONS.values()
    ]
