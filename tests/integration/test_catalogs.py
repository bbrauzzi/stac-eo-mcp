"""Live catalog tests. Run with: uv run pytest -m slow"""

import time

import pytest

from stac_eo_mcp import catalog
from stac_eo_mcp.config import CatalogName
from stac_eo_mcp.errors import NoScenesError
from stac_eo_mcp.models import BBox, DateRange

pytestmark = pytest.mark.slow

ROME = BBox.parse([12.40, 41.85, 12.55, 41.95])
SUMMER = DateRange.parse("2025-06-01/2025-07-15")


@pytest.mark.parametrize("cat", list(CatalogName))
def test_list_collections_includes_s2(cat):
    catalog._collections.cache_clear()
    start = time.monotonic()
    ids = {c.id for c in catalog.list_collections(cat)}
    # Signing every collection made this take ~60 s on Planetary Computer.
    assert time.monotonic() - start < 15
    assert catalog.get_config(cat).s2_collection in ids


@pytest.mark.parametrize("cat", list(CatalogName))
def test_search_and_details(cat):
    items = catalog.search_items(cat, "sentinel-2", ROME, SUMMER, max_cloud_cover=20, limit=3)
    assert 1 <= len(items) <= 3
    assert all(i.properties["eo:cloud_cover"] <= 20 for i in items)

    summary = catalog.summarize_item(items[0], cat)
    assert summary.preview_url
    assert summary.footprint

    item = catalog.get_item(cat, "sentinel-2", items[0].id)
    details = catalog.describe_item(item)
    assert details.id == items[0].id
    keys = catalog.band_asset_keys(cat, item, ["red", "nir", "green", "swir22", "scl"])
    assert catalog.band_scaling(item, keys["red"]).offset_dn == pytest.approx(-1000)


def test_pc_assets_are_signed():
    items = catalog.search_items(
        CatalogName.PLANETARY_COMPUTER, "sentinel-2", ROME, SUMMER, max_cloud_cover=20, limit=1
    )
    assert "sig=" in items[0].assets["B04"].href


def test_no_scenes_message():
    with pytest.raises(NoScenesError, match="cloud cover <= 0%"):
        catalog.search_items(
            CatalogName.EARTH_SEARCH,
            "sentinel-2",
            ROME,
            DateRange.parse("2014-01-01/2014-01-31"),  # before Sentinel-2
            max_cloud_cover=0,
        )
