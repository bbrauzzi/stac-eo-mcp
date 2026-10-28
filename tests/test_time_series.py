from datetime import UTC, datetime, timedelta

import pytest

from stac_eo_mcp import analysis, catalog
from stac_eo_mcp.config import CatalogName, Settings
from stac_eo_mcp.errors import CatalogError, DataUnavailableError
from stac_eo_mcp.indices import compute_stats
from stac_eo_mcp.models import BBox, DateRange, SpectralIndex

from .conftest import ROME_BBOX, make_item

PC = CatalogName.PLANETARY_COMPUTER
BBOX = BBox.parse(ROME_BBOX)
START = datetime(2025, 4, 1, 10, tzinfo=UTC)


def poly(w, s, e, n):
    return {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}


def day_item(day: int, suffix: str = "A", cloud: float = 5.0, geometry=None):
    item = make_item(f"S2_{day:03d}_{suffix}", dt=START + timedelta(days=day), cloud=cloud)
    if geometry:
        item.geometry = geometry
    return item


class TestOnePerDay:
    def test_prefers_full_coverage_then_low_cloud(self):
        w, s, e, n = ROME_BBOX
        half = poly(w, s, (w + e) / 2, n)
        items = [
            day_item(0, "half", cloud=1, geometry=half),
            day_item(0, "full", cloud=30),
            day_item(5, "cloudy", cloud=40),
            day_item(5, "clear", cloud=2),
        ]
        picked = analysis.one_per_day(items, BBOX)
        assert [i.id for i in picked] == ["S2_000_full", "S2_005_clear"]

    def test_sorted_oldest_first(self):
        picked = analysis.one_per_day([day_item(10), day_item(0), day_item(5)], BBOX)
        assert [i.id for i in picked] == ["S2_000_A", "S2_005_A", "S2_010_A"]


def test_sample_evenly_keeps_ends():
    items = [day_item(d) for d in range(100)]
    picked = analysis.sample_evenly(items, 5)
    assert [i.id for i in picked] == ["S2_000_A", "S2_025_A", "S2_050_A", "S2_074_A", "S2_099_A"]
    assert analysis.sample_evenly(items[:3], 5) == items[:3]


@pytest.fixture
def series_env(monkeypatch):
    """Fake search results and per-scene stats keyed by item id."""
    env = {"items": [], "stats": {}, "calls": []}

    def fake_search(*args, **kwargs):
        env["calls"].append((args, kwargs))
        return list(env["items"])

    def fake_index(_cat, item, _index, _bbox):
        outcome = env["stats"][item.id]
        if isinstance(outcome, Exception):
            raise outcome
        return None, outcome

    monkeypatch.setattr(catalog, "search_items", fake_search)
    monkeypatch.setattr(analysis, "index_on_item", fake_index)
    return env


def stats(mean, valid_pct=100.0):
    import numpy as np

    n = 100
    valid = int(valid_pct)
    arr = np.full(n, np.nan)
    arr[:valid] = mean
    return compute_stats(arr)


def run(settings=None, **kw):
    return analysis.time_series(
        settings or Settings(max_area_km2=200),
        PC,
        SpectralIndex.NDVI,
        BBOX,
        DateRange.parse("2025-04-01/2025-09-30"),
        kw.get("max_cloud_cover", 30),
    )


def test_series_points_skips_and_summary(series_env):
    series_env["items"] = [day_item(d) for d in (0, 10, 20, 30, 40)]
    series_env["stats"] = {
        "S2_000_A": stats(0.30),
        "S2_010_A": stats(0.50),
        "S2_020_A": stats(0.80, valid_pct=10),  # too cloudy
        "S2_030_A": DataUnavailableError("Scene lacks assets"),
        "S2_040_A": stats(0.70),
    }
    res = run()
    assert [p.date for p in res.points] == ["2025-04-01", "2025-04-11", "2025-05-11"]
    assert [p.mean for p in res.points] == [0.3, 0.5, 0.7]
    reasons = {s.item_id: s.reason for s in res.skipped}
    assert reasons["S2_020_A"] == "only 10.0% clear pixels over the bbox"
    assert reasons["S2_030_A"] == "Scene lacks assets"
    assert res.summary.change == pytest.approx(0.4)
    assert res.summary.max.date == "2025-05-11"
    assert res.summary.mean_of_means == 0.5
    assert res.note is None
    assert res.scenes_found == 5
    assert res.dates_analysed == 5


def test_unexpected_error_does_not_sink_series(series_env):
    series_env["items"] = [day_item(0), day_item(5), day_item(9)]
    series_env["stats"] = {
        "S2_000_A": stats(0.3),
        "S2_005_A": RuntimeError("HTTP 503"),
        "S2_009_A": stats(0.4),
    }
    res = run()
    assert len(res.points) == 2
    assert res.skipped[0].reason == "read error: HTTP 503"
    assert "Fewer than 3 usable dates" in res.note


def test_samples_when_over_limit(series_env):
    series_env["items"] = [day_item(d) for d in range(30)]
    series_env["stats"] = {i.id: stats(0.5) for i in series_env["items"]}
    res = run(Settings(max_scenes=6, max_area_km2=200))
    assert res.dates_analysed == 6
    assert res.points[0].date == "2025-04-01"
    assert res.points[-1].date == "2025-04-30"
    assert "30 acquisition dates matched; 6 were sampled evenly" in res.note


def test_all_cloudy(series_env):
    series_env["items"] = [day_item(0), day_item(5)]
    series_env["stats"] = {"S2_000_A": stats(0.5, 0), "S2_005_A": stats(0.5, 5)}
    res = run()
    assert res.points == []
    assert res.summary is None
    assert "No date had enough clear pixels" in res.note


def test_search_params(series_env):
    series_env["items"] = [day_item(0)]
    series_env["stats"] = {"S2_000_A": stats(0.5)}
    run()
    args, kwargs = series_env["calls"][0]
    assert args[1] == "sentinel-2-l2a"
    assert args[4] == 30
    assert kwargs["limit"] == analysis.SEARCH_POOL


def test_search_errors_propagate(series_env, monkeypatch):
    def boom(*_a, **_k):
        raise CatalogError("down")

    monkeypatch.setattr(catalog, "search_items", boom)
    with pytest.raises(CatalogError):
        run()
