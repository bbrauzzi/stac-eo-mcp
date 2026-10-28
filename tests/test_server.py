"""MCP-level tests: an in-process client talks to the real server object."""

import json

import pytest
from mcp import Client

from stac_eo_mcp import analysis, server
from stac_eo_mcp import catalog as stac_catalog
from stac_eo_mcp.config import CatalogName, Settings
from stac_eo_mcp.errors import NoScenesError
from stac_eo_mcp.models import (
    CollectionSummary,
    IndexResult,
    IndexStats,
    SpectralIndex,
    TimeSeriesResult,
)

pytestmark = pytest.mark.anyio

ROME = [12.47, 41.905, 12.5, 41.925]


@pytest.fixture
async def client(monkeypatch, tmp_path):
    settings = Settings(output_dir=tmp_path, catalog=CatalogName.EARTH_SEARCH)
    monkeypatch.setattr(server, "get_settings", lambda: settings)
    async with Client(server.mcp) as c:
        yield c


def text(result) -> str:
    return result.content[0].text


async def test_lists_tools_resources_prompts(client):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {
        "list_collections",
        "search_scenes",
        "get_scene_details",
        "compute_index",
        "time_series",
    }
    assert tools["search_scenes"].input_schema["properties"]["bbox"]["minItems"] == 4
    assert tools["compute_index"].input_schema["properties"]["index"]["enum"] == [
        "ndvi",
        "ndwi",
        "nbr",
    ]
    assert tools["time_series"].annotations.read_only_hint is True
    assert tools["compute_index"].annotations.read_only_hint is False
    assert all(t.output_schema for t in tools.values())

    uris = {str(r.uri) for r in (await client.list_resources()).resources}
    assert uris == {"stac://indices", "stac://catalogs"}
    prompts = (await client.list_prompts()).prompts
    assert [p.name for p in prompts] == ["analyze_vegetation"]


async def test_list_collections_uses_default_catalog(client, monkeypatch):
    seen = []

    def fake(cat):
        seen.append(cat)
        return [CollectionSummary(id="sentinel-2-c1-l2a", title="Sentinel-2")]

    monkeypatch.setattr(stac_catalog, "list_collections", fake)
    res = await client.call_tool("list_collections", {})
    assert not res.is_error
    assert seen == [CatalogName.EARTH_SEARCH]
    assert "sentinel-2-c1-l2a" in text(res)


async def test_search_scenes(client, monkeypatch, es_item):
    calls = []

    def fake_search(cat, coll, bbox, dr, cc, limit):
        calls.append((cat, coll, bbox.as_list(), dr.to_stac(), cc, limit))
        return [es_item]

    monkeypatch.setattr(stac_catalog, "resolve_collection", lambda _c, _n: "sentinel-2-c1-l2a")
    monkeypatch.setattr(stac_catalog, "search_items", fake_search)
    res = await client.call_tool(
        "search_scenes",
        {
            "collection": "sentinel-2",
            "bbox": ROME,
            "date_range": "2025-06-01/2025-06-30",
            "catalog": "earth-search",
        },
    )
    assert not res.is_error, text(res)
    data = res.structured_content
    assert data["returned"] == 1
    assert data["scenes"][0]["id"] == es_item.id
    assert calls[0][4:] == (20.0, 10)  # default cloud cover and limit


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        ({"bbox": [10, 41, 9, 42]}, "west (10.0) must be less than east"),
        ({"date_range": "2025-06-30/2025-06-01"}, "is after end"),
        ({"max_cloud_cover": 150}, "outside [0, 100]"),
        ({"limit": 500}, "limit=500 is outside [1, 50]"),
    ],
)
async def test_search_validation_errors_reach_agent(client, monkeypatch, args, fragment):
    monkeypatch.setattr(stac_catalog, "resolve_collection", lambda _c, _n: "x")
    base = {"collection": "sentinel-2", "bbox": ROME, "date_range": "2025-06-01/2025-06-30"}
    res = await client.call_tool("search_scenes", base | args)
    assert res.is_error
    assert fragment in text(res)
    assert "Hint:" in text(res)


async def test_schema_rejects_bad_bbox_length(client):
    res = await client.call_tool(
        "search_scenes",
        {"collection": "s2", "bbox": [1, 2, 3], "date_range": "2025-01-01/2025-01-02"},
    )
    assert res.is_error


async def test_no_scenes_error_has_hint(client, monkeypatch):
    def none(*_a, **_k):
        raise NoScenesError("No scenes with cloud cover <= 5%.", hint="Raise max_cloud_cover.")

    monkeypatch.setattr(stac_catalog, "resolve_collection", lambda _c, _n: "x")
    monkeypatch.setattr(stac_catalog, "search_items", none)
    res = await client.call_tool(
        "search_scenes",
        {
            "collection": "s2",
            "bbox": ROME,
            "date_range": "2025-06-01/2025-06-30",
            "max_cloud_cover": 5,
        },
    )
    assert res.is_error
    assert "No scenes with cloud cover <= 5%. Hint: Raise max_cloud_cover." in text(res)


async def test_unexpected_error_is_explained(client, monkeypatch):
    def boom(_cat):
        raise ConnectionError("connection reset")

    monkeypatch.setattr(stac_catalog, "list_collections", boom)
    res = await client.call_tool("list_collections", {})
    assert res.is_error
    assert "Unexpected ConnectionError: connection reset" in text(res)


async def test_get_scene_details(client, monkeypatch, es_item):
    monkeypatch.setattr(stac_catalog, "get_item", lambda *_a: es_item)
    res = await client.call_tool(
        "get_scene_details", {"collection": "sentinel-2", "item_id": es_item.id}
    )
    assert not res.is_error
    assert {a["key"] for a in res.structured_content["assets"]} >= {"red", "nir", "scl"}


async def test_compute_index(client, monkeypatch):
    captured = {}

    def fake(settings, cat, item_id, index, bbox, save):
        captured.update(cat=cat, index=index, save=save, out=settings.output_dir)
        stats = IndexStats(
            min=0.1, max=0.9, mean=0.5, std=0.2, valid_pixel_pct=90, valid_pixels=9, total_pixels=10
        )
        return IndexResult(
            item_id=item_id, index=index, datetime=None, bbox=bbox.as_list(), stats=stats
        )

    monkeypatch.setattr(analysis, "compute_index_for_item", fake)
    res = await client.call_tool(
        "compute_index", {"item_id": "S2X", "index": "ndvi", "bbox": ROME, "save": "png"}
    )
    assert not res.is_error, text(res)
    assert res.structured_content["stats"]["mean"] == 0.5
    assert captured["index"] is SpectralIndex.NDVI
    assert captured["save"].value == "png"


async def test_compute_index_area_limit(client):
    res = await client.call_tool(
        "compute_index", {"item_id": "S2X", "index": "ndvi", "bbox": [12, 41, 13, 42]}
    )
    assert res.is_error
    assert "above the 100 km² limit" in text(res)


async def test_compute_index_rejects_unknown_index(client):
    res = await client.call_tool("compute_index", {"item_id": "S2X", "index": "evi", "bbox": ROME})
    assert res.is_error


async def test_time_series(client, monkeypatch):
    def fake(settings, cat, index, bbox, dr, cc):
        return TimeSeriesResult(
            index=index,
            bbox=bbox.as_list(),
            date_range=f"{dr.start}/{dr.end}",
            scenes_found=0,
            dates_analysed=0,
            points=[],
            note=f"cc={cc}",
        )

    monkeypatch.setattr(analysis, "time_series", fake)
    res = await client.call_tool(
        "time_series", {"index": "nbr", "bbox": ROME, "date_range": "2025-01-01/2025-03-01"}
    )
    assert not res.is_error, text(res)
    assert res.structured_content["note"] == "cc=30.0"


async def test_indices_resource(client):
    res = await client.read_resource("stac://indices")
    data = json.loads(res.contents[0].text)
    assert {i["index"] for i in data["indices"]} == {"ndvi", "ndwi", "nbr"}
    assert "9" in data["cloud_mask"]["masked_classes"]


async def test_catalogs_resource(client):
    data = json.loads((await client.read_resource("stac://catalogs")).contents[0].text)
    assert data["default_catalog"] == "earth-search"
    assert data["limits"]["max_area_km2"] == 100


async def test_prompt(client):
    res = await client.get_prompt("analyze_vegetation", {"bbox": "12.47,41.905,12.5,41.925"})
    msg = res.messages[0].content.text
    assert 'index="ndvi"' in msg
    assert "bbox=[12.47,41.905,12.5,41.925]" in msg
