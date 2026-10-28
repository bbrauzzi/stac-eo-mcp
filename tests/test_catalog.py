import pytest

from stac_eo_mcp import catalog
from stac_eo_mcp.config import CatalogName
from stac_eo_mcp.errors import DataUnavailableError, InvalidInputError, NoScenesError
from stac_eo_mcp.models import BBox, CollectionSummary, DateRange

from .conftest import ROME_BBOX, make_item

PC = CatalogName.PLANETARY_COMPUTER
ES = CatalogName.EARTH_SEARCH


@pytest.fixture
def fake_collections(monkeypatch):
    cols = (
        CollectionSummary(id="landsat-c2-l2"),
        CollectionSummary(id="sentinel-1-grd"),
        CollectionSummary(id="sentinel-2-l2a"),
    )
    monkeypatch.setattr(catalog, "_collections", lambda _c: cols)


class FakeSearch:
    def __init__(self, items):
        self._items = items

    def items(self):
        return iter(self._items)


class FakeClient:
    def __init__(self, items):
        self.items = items
        self.calls = []

    def search(self, **params):
        self.calls.append(params)
        return FakeSearch(self.items)


@pytest.fixture
def fake_client(monkeypatch, fake_collections):
    client = FakeClient([])
    monkeypatch.setattr(catalog, "get_client", lambda _c: client)
    return client


def test_get_config_unknown():
    with pytest.raises(InvalidInputError, match="planetary-computer, earth-search"):
        catalog.get_config("nasa")


class TestResolveCollection:
    @pytest.mark.parametrize("alias", ["sentinel-2", "S2", "sentinel-2-l2a"])
    def test_s2_aliases(self, alias):
        assert catalog.resolve_collection(PC, alias) == "sentinel-2-l2a"
        assert catalog.resolve_collection(ES, alias) == "sentinel-2-c1-l2a"

    def test_existing(self, fake_collections):
        assert catalog.resolve_collection(PC, "landsat-c2-l2") == "landsat-c2-l2"

    def test_unknown_suggests(self, fake_collections):
        with pytest.raises(InvalidInputError, match="Did you mean: landsat-c2-l2"):
            catalog.resolve_collection(PC, "landsat-c2")


class TestSearch:
    def test_builds_query_and_sorts_newest_first(self, fake_client):
        from datetime import UTC, datetime

        old = make_item("old", dt=datetime(2025, 6, 1, tzinfo=UTC))
        new = make_item("new", dt=datetime(2025, 7, 1, tzinfo=UTC))
        fake_client.items = [old, new]
        items = catalog.search_items(
            PC, "sentinel-2", BBox.parse(ROME_BBOX), DateRange.parse("2025-06-01/2025-07-31"), 20, 5
        )
        assert [i.id for i in items] == ["new", "old"]
        call = fake_client.calls[0]
        assert call["collections"] == ["sentinel-2-l2a"]
        assert call["query"] == {"eo:cloud_cover": {"lte": 20}}
        assert call["max_items"] == 5
        assert call["datetime"] == "2025-06-01T00:00:00Z/2025-07-31T23:59:59Z"

    def test_no_scenes_with_cloud_filter(self, fake_client):
        with pytest.raises(NoScenesError) as exc:
            catalog.search_items(
                PC,
                "sentinel-2",
                BBox.parse(ROME_BBOX),
                DateRange.parse("2025-06-01/2025-06-05"),
                20,
            )
        assert "cloud cover <= 20%" in str(exc.value)
        assert "Raise max_cloud_cover or widen the date range" in str(exc.value)

    def test_no_scenes_without_filter(self, fake_client):
        with pytest.raises(NoScenesError, match="Widen the date range"):
            catalog.search_items(
                PC, "sentinel-2", BBox.parse(ROME_BBOX), DateRange.parse("2025-06-01/2025-06-05")
            )

    def test_get_item_not_found(self, fake_client):
        with pytest.raises(NoScenesError, match="was not found"):
            catalog.get_item(PC, "sentinel-2", "missing")


class TestSummaries:
    def test_summarize_pc(self, pc_item):
        s = catalog.summarize_item(pc_item, PC)
        assert s.id == "S2A_TEST"
        assert s.datetime == "2025-07-10T10:00:00Z"
        assert s.cloud_cover == 9.71
        assert s.preview_url.endswith("rendered_preview.tif")
        assert s.footprint["type"] == "Polygon"

    def test_summarize_es_uses_thumbnail(self, es_item):
        assert catalog.summarize_item(es_item, ES).preview_url.endswith("thumbnail.tif")

    def test_describe(self, pc_item):
        d = catalog.describe_item(pc_item)
        assert d.properties == {
            "platform": "sentinel-2a",
            "s2:processing_baseline": "05.11",
            "eo:cloud_cover": 9.7071,
        }
        keys = {a.key: a for a in d.assets}
        assert keys["B04"].gsd == 10
        assert keys["rendered_preview"].roles == ["overview"]

    def test_simplify_geometry_rounds_and_drops_points(self):
        coords = [[12.0 + i * 1e-5, 41.0] for i in range(100)] + [[12.1, 41.1], [12.0, 41.0]]
        geom = catalog.simplify_geometry({"type": "Polygon", "coordinates": [coords]})
        assert len(geom["coordinates"][0]) < 10
        assert all(len(str(c).split(".")[-1]) <= 4 for pt in geom["coordinates"][0] for c in pt)

    def test_first_sentence(self):
        assert catalog._first_sentence("Short one. And more.") == "Short one."
        assert catalog._first_sentence("x" * 300).endswith("…")
        assert catalog._first_sentence(None) is None


class TestBands:
    def test_asset_keys_pc(self, pc_item):
        assert catalog.band_asset_keys(PC, pc_item, ["nir", "red", "scl"]) == {
            "nir": "B08",
            "red": "B04",
            "scl": "SCL",
        }

    def test_asset_keys_missing(self, es_item):
        with pytest.raises(DataUnavailableError, match="swir22"):
            catalog.band_asset_keys(ES, es_item, ["nir", "swir22"])

    def test_scaling_from_raster_bands(self, es_item):
        s = catalog.band_scaling(es_item, "red")
        assert s.scale == 0.0001
        assert s.offset_dn == pytest.approx(-1000)

    @pytest.mark.parametrize(
        ("baseline", "offset"), [("05.11", -1000), ("04.00", -1000), ("03.01", 0), (None, -1000)]
    )
    def test_scaling_from_baseline(self, baseline, offset):
        item = make_item(baseline=baseline, assets={"B04": {}})
        assert catalog.band_scaling(item, "B04").offset_dn == offset
