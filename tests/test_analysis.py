import numpy as np
import pytest
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from PIL import Image

from stac_eo_mcp import analysis, catalog, raster
from stac_eo_mcp.config import CatalogName, Settings
from stac_eo_mcp.errors import AreaTooLargeError, InvalidInputError
from stac_eo_mcp.models import BBox, SaveFormat, SpectralIndex

from .conftest import ROME_BBOX, make_item

PC = CatalogName.PLANETARY_COMPUTER


def synthetic_window(nir_dn=4000, red_dn=1500, cloud_rows=0, shape=(20, 30)) -> xr.Dataset:
    ny, nx = shape
    x = 290_000 + 10 * np.arange(nx) + 5
    y = 4_650_000 - 10 * np.arange(ny) - 5
    scl = np.full(shape, 4, dtype=np.uint8)
    scl[:cloud_rows] = 9
    ds = xr.Dataset(
        {
            "nir": (("y", "x"), np.full(shape, nir_dn, dtype=np.uint16)),
            "red": (("y", "x"), np.full(shape, red_dn, dtype=np.uint16)),
            "scl": (("y", "x"), scl),
        },
        coords={"x": x, "y": y},
    )
    return ds.rio.write_crs("EPSG:32633")


@pytest.fixture
def settings(tmp_path):
    return Settings(output_dir=tmp_path / "out", max_area_km2=200)


@pytest.fixture
def patched(monkeypatch, pc_item):
    """get_item returns pc_item, load_window returns a synthetic window."""
    state = {"ds": synthetic_window()}
    monkeypatch.setattr(catalog, "get_item", lambda *_a, **_k: pc_item)
    monkeypatch.setattr(raster, "load_window", lambda *_a, **_k: state["ds"])
    return state


def test_index_from_dataset_applies_offset(pc_item):
    keys = catalog.band_asset_keys(PC, pc_item, ["nir", "red", "scl"])
    da = analysis.index_from_dataset(synthetic_window(), pc_item, keys, SpectralIndex.NDVI)
    # Baseline 05.11: (3000 - 500) / (3000 + 500)
    np.testing.assert_allclose(da.values, 2500 / 3500, rtol=1e-5)
    assert da.rio.crs.to_epsg() == 32633


def test_compute_index_end_to_end(patched, settings):
    patched["ds"] = synthetic_window(cloud_rows=5)
    res = analysis.compute_index_for_item(
        settings, PC, "S2A_TEST", SpectralIndex.NDVI, BBox.parse(ROME_BBOX), SaveFormat.BOTH
    )
    assert res.stats.valid_pixel_pct == 75.0
    assert res.stats.mean == pytest.approx(0.7143, abs=1e-4)
    assert res.note is None
    assert res.datetime == "2025-07-10T10:00:00Z"

    png = Image.open(res.outputs["png"])
    assert png.size == (30, 20)
    assert png.getpixel((0, 0))[3] == 0  # cloudy row is transparent

    with rasterio.open(res.outputs["geotiff"]) as src:
        assert src.crs.to_epsg() == 32633
        assert (src.width, src.height) == (30, 20)
        assert src.res == (10.0, 10.0)
        data = src.read(1)
        assert np.isnan(data[0, 0])
        assert data[-1, -1] == pytest.approx(0.7143, abs=1e-4)


def test_all_cloudy_returns_note_and_no_files(patched, settings):
    patched["ds"] = synthetic_window(cloud_rows=20)
    res = analysis.compute_index_for_item(
        settings, PC, "S2A_TEST", SpectralIndex.NDVI, BBox.parse(ROME_BBOX), SaveFormat.PNG
    )
    assert res.stats.valid_pixels == 0
    assert res.outputs == {}
    assert "All pixels are masked" in res.note


def test_low_valid_note(patched, settings):
    patched["ds"] = synthetic_window(cloud_rows=18)
    res = analysis.compute_index_for_item(
        settings, PC, "S2A_TEST", SpectralIndex.NDVI, BBox.parse(ROME_BBOX)
    )
    assert "Only 10.0% of pixels are clear" in res.note


def test_area_limit_checked_before_network(settings, monkeypatch):
    monkeypatch.setattr(catalog, "get_item", lambda *_a, **_k: pytest.fail("network called"))
    with pytest.raises(AreaTooLargeError):
        analysis.compute_index_for_item(
            settings, PC, "x", SpectralIndex.NDVI, BBox.parse([12, 41, 13, 42])
        )


def test_bbox_outside_scene(settings, monkeypatch):
    monkeypatch.setattr(catalog, "get_item", lambda *_a, **_k: make_item(assets={}))
    with pytest.raises(InvalidInputError, match="does not overlap"):
        analysis.compute_index_for_item(
            settings, PC, "x", SpectralIndex.NDVI, BBox.parse([0.0, 0.0, 0.05, 0.05])
        )


def test_output_stem_is_stable():
    bbox = BBox.parse(ROME_BBOX)
    assert raster.output_stem("S2A", SpectralIndex.NBR, bbox) == raster.output_stem(
        "S2A", SpectralIndex.NBR, BBox.parse(list(ROME_BBOX))
    )
    assert raster.output_stem("S2A", SpectralIndex.NBR, bbox).startswith("S2A_nbr_")
