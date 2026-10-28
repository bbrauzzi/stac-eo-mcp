"""Launch the real console script over stdio, as an MCP host would."""

import os
import shutil

import pytest
from mcp import Client, StdioServerParameters

pytestmark = [pytest.mark.slow, pytest.mark.anyio]

PARKS = [12.47, 41.905, 12.5, 41.925]


async def test_stdio_search_and_compute(tmp_path):
    exe = shutil.which("stac-eo-mcp")
    assert exe, "console script not installed (run uv sync)"
    params = StdioServerParameters(
        command=exe,
        env={
            **os.environ,
            "STAC_EO_MCP_OUTPUT_DIR": str(tmp_path),
            "STAC_EO_MCP_LOG_LEVEL": "DEBUG",
        },
    )
    async with Client(params) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        assert "compute_index" in tools

        found = await client.call_tool(
            "search_scenes",
            {
                "collection": "sentinel-2",
                "bbox": PARKS,
                "date_range": "2025-07-25/2025-08-05",
                "max_cloud_cover": 10,
                "limit": 3,
            },
        )
        assert not found.is_error, found.content[0].text
        scene = found.structured_content["scenes"][0]["id"]

        res = await client.call_tool(
            "compute_index", {"item_id": scene, "index": "ndvi", "bbox": PARKS, "save": "png"}
        )
        assert not res.is_error, res.content[0].text
        assert res.structured_content["stats"]["valid_pixel_pct"] > 50
        assert os.path.exists(res.structured_content["outputs"]["png"])
