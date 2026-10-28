"""Live end-to-end index computation on a real Sentinel-2 scene over Rome."""

import pytest
import rasterio

from stac_eo_mcp import analysis, catalog
from stac_eo_mcp.config import CatalogName, Settings
from stac_eo_mcp.models import BBox, DateRange, SaveFormat, SpectralIndex

pytestmark = pytest.mark.slow

# Villa Borghese / Villa Ada area: mixed park and city, ~2.5 km x 2.2 km.
PARKS = BBox.parse([12.470, 41.905, 12.500, 41.925])


@pytest.mark.parametrize("cat", list(CatalogName))
def test_ndvi_on_real_scene(cat, tmp_path):
    items = catalog.search_items(
        cat, "sentinel-2", PARKS, DateRange.parse("2025-06-01/2025-07-31"), 5, limit=1
    )
    settings = Settings(output_dir=tmp_path)
    res = analysis.compute_index_for_item(
        settings, cat, items[0].id, SpectralIndex.NDVI, PARKS, SaveFormat.BOTH
    )
    s = res.stats
    assert s.valid_pixel_pct > 80
    assert 0.1 < s.mean < 0.7  # mixed urban/park in summer
    assert s.max > 0.6  # parks are green
    with rasterio.open(res.outputs["geotiff"]) as src:
        assert src.res == (10.0, 10.0)
        assert 200 < src.width < 300  # ~2.5 km at 10 m


def test_time_series_spring_greening():
    from stac_eo_mcp.analysis import time_series

    res = time_series(
        Settings(max_scenes=8),
        CatalogName.EARTH_SEARCH,
        SpectralIndex.NDVI,
        PARKS,
        DateRange.parse("2025-01-01/2025-06-30"),
        max_cloud_cover=30,
    )
    assert len(res.points) >= 3
    dates = [p.date for p in res.points]
    assert dates == sorted(dates)
    assert len(set(dates)) == len(dates)  # one point per day despite tile overlap
    assert all(-1 <= p.mean <= 1 for p in res.points)
