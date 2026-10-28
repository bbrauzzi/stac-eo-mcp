"""MCP server: tools, resources and prompts over stdio.

Blocking work (HTTP, raster reads) runs in worker threads so the event loop
stays responsive. Errors become ToolError so the agent reads the message and
the hint; stdout is reserved for the protocol and logs go to stderr.
"""

import json
import logging
import sys
from collections.abc import Callable
from datetime import date, timedelta
from functools import partial
from typing import Annotated, Literal

import anyio.to_thread
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from stac_eo_mcp import __version__, analysis
from stac_eo_mcp import catalog as stac_catalog
from stac_eo_mcp.config import CatalogName, get_settings
from stac_eo_mcp.errors import StacMcpError
from stac_eo_mcp.indices import DEFAULT_VALID_SCL, SCL_CLASSES, describe_indices
from stac_eo_mcp.models import (
    BBox,
    CollectionSummary,
    DateRange,
    IndexResult,
    SceneDetails,
    SearchResult,
    TimeSeriesResult,
    check_cloud_cover,
    check_limit,
    parse_index,
    parse_save,
)

log = logging.getLogger("stac_eo_mcp")

INSTRUCTIONS = """\
Search Earth observation imagery in STAC catalogs and compute Sentinel-2 indices.

Typical flow:
1. search_scenes(collection="sentinel-2-l2a", bbox, date_range, max_cloud_cover) to find scenes.
2. compute_index(item_id, index, bbox) for statistics on one scene (optionally save PNG/GeoTIFF).
3. time_series(index, bbox, date_range) for the index trend over time.

bbox is [west, south, east, north] in WGS84 degrees; date_range is 'YYYY-MM-DD/YYYY-MM-DD'.
Pixel reads are limited to small areas (see stac://catalogs). Indices use the SCL layer to
mask clouds, so check valid_pixel_pct: tile-level cloud cover can differ from your bbox.
Read stac://indices for formulas and how to interpret values.
"""

mcp = MCPServer(
    name="stac-eo-mcp",
    title="STAC Earth Observation",
    instructions=INSTRUCTIONS,
    version=__version__,
)

# --- Argument types (descriptions end up in the tool JSON schema) ----------

CatalogArg = Annotated[
    Literal["planetary-computer", "earth-search"] | None,
    Field(description="STAC catalog. Defaults to the server's configured catalog."),
]
BBoxArg = Annotated[
    list[float],
    Field(
        min_length=4,
        max_length=4,
        description="[west, south, east, north] in WGS84 degrees, e.g. [12.47, 41.9, 12.5, 41.92]",
    ),
]
DateRangeArg = Annotated[
    str, Field(description="ISO interval 'YYYY-MM-DD/YYYY-MM-DD', e.g. '2025-04-01/2025-09-30'")
]
CloudArg = Annotated[
    float | None,
    Field(description="Maximum scene cloud cover in percent (0-100). Omit for no filter."),
]
IndexArg = Annotated[
    Literal["ndvi", "ndwi", "nbr"],
    Field(description="Spectral index: ndvi (vegetation), ndwi (water), nbr (burn)."),
]

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=True, idempotentHint=True)


def _catalog(value: str | None) -> CatalogName:
    return stac_catalog.get_config(value or get_settings().catalog).name


async def _run[T](fn: Callable[[], T]) -> T:
    """Run blocking work in a thread and turn failures into agent-readable errors."""
    try:
        return await anyio.to_thread.run_sync(fn)
    except StacMcpError as exc:
        log.debug("Tool error: %s", exc)  # the SDK logs it at INFO
        raise ToolError(str(exc)) from exc
    except Exception as exc:
        log.exception("Unexpected error")
        raise ToolError(
            f"Unexpected {type(exc).__name__}: {exc}. Hint: retry once; if it persists, "
            "try the other catalog or a smaller request."
        ) from exc


# --- Tools ----------------------------------------------------------------


@mcp.tool(annotations=READ_ONLY)
async def list_collections(catalog: CatalogArg = None) -> list[CollectionSummary]:
    """List the collections available in a STAC catalog, with a one-sentence description.

    Use the returned ids as `collection` in search_scenes and get_scene_details.
    """
    cat = _catalog(catalog)
    return await _run(partial(stac_catalog.list_collections, cat))


@mcp.tool(annotations=READ_ONLY)
async def search_scenes(
    collection: Annotated[
        str, Field(description="Collection id, e.g. 'sentinel-2-l2a' (alias 'sentinel-2' works)")
    ],
    bbox: BBoxArg,
    date_range: DateRangeArg,
    max_cloud_cover: CloudArg = 20,
    limit: Annotated[int, Field(description="Maximum scenes to return (1-50)")] = 10,
    catalog: CatalogArg = None,
) -> SearchResult:
    """Search scenes intersecting a bbox in a date range, newest first.

    Returns id, date, cloud cover, a simplified footprint and a preview image link.
    Cloud cover is for the whole tile, not the bbox.
    """
    settings = get_settings()
    cat = _catalog(catalog)

    def work() -> SearchResult:
        box = BBox.parse(bbox)
        dr = DateRange.parse(date_range)
        cc = check_cloud_cover(max_cloud_cover)
        n = check_limit(limit, settings.max_search_limit)
        coll = stac_catalog.resolve_collection(cat, collection)
        items = stac_catalog.search_items(cat, coll, box, dr, cc, limit=n)
        return SearchResult(
            catalog=cat.value,
            collection=coll,
            returned=len(items),
            scenes=[stac_catalog.summarize_item(i, cat) for i in items],
        )

    return await _run(work)


@mcp.tool(annotations=READ_ONLY)
async def get_scene_details(
    collection: Annotated[str, Field(description="Collection id the scene belongs to")],
    item_id: Annotated[str, Field(description="Scene id from search_scenes")],
    catalog: CatalogArg = None,
) -> SceneDetails:
    """Key metadata and the list of assets (bands, previews, metadata files) of one scene."""
    cat = _catalog(catalog)
    return await _run(
        lambda: stac_catalog.describe_item(stac_catalog.get_item(cat, collection, item_id))
    )


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=True
    )
)
async def compute_index(
    item_id: Annotated[str, Field(description="Sentinel-2 L2A scene id from search_scenes")],
    index: IndexArg,
    bbox: BBoxArg,
    save: Annotated[
        Literal["none", "png", "geotiff", "both"],
        Field(description="Also write the index map to the output folder"),
    ] = "none",
    catalog: CatalogArg = None,
) -> IndexResult:
    """Compute NDVI, NDWI or NBR on one Sentinel-2 L2A scene, reading only the bbox window.

    Clouds, shadows and no-data are masked with the SCL layer. Returns min, max, mean, std
    and the share of valid pixels, plus file paths when `save` is set. No raster data is
    returned. Area is limited (default 100 km²).
    """
    settings = get_settings()
    cat = _catalog(catalog)
    return await _run(
        lambda: analysis.compute_index_for_item(
            settings, cat, item_id, parse_index(index), BBox.parse(bbox), parse_save(save)
        )
    )


@mcp.tool(annotations=READ_ONLY)
async def time_series(
    index: IndexArg,
    bbox: BBoxArg,
    date_range: DateRangeArg,
    max_cloud_cover: CloudArg = 30,
    catalog: CatalogArg = None,
) -> TimeSeriesResult:
    """Mean of an index over a bbox for each Sentinel-2 acquisition date in a range.

    One scene per day, cloudy dates skipped with a reason, sampled evenly when there are
    more dates than the server limit (default 24). Includes a summary (mean, min/max date,
    change from first to last). Can take 10-60 s for long ranges.
    """
    settings = get_settings()
    cat = _catalog(catalog)

    def work() -> TimeSeriesResult:
        return analysis.time_series(
            settings,
            cat,
            parse_index(index),
            BBox.parse(bbox),
            DateRange.parse(date_range),
            check_cloud_cover(max_cloud_cover),
        )

    return await _run(work)


# --- Resources ------------------------------------------------------------


@mcp.resource(
    "stac://indices",
    name="indices",
    title="Supported spectral indices",
    description="Formulas, Sentinel-2 bands, interpretation and the SCL cloud mask.",
    mime_type="application/json",
)
def indices_resource() -> str:
    return json.dumps(
        {
            "indices": describe_indices(),
            "cloud_mask": {
                "layer": "Sentinel-2 Scene Classification (SCL)",
                "kept_classes": {c: SCL_CLASSES[c] for c in sorted(DEFAULT_VALID_SCL)},
                "masked_classes": {
                    c: n for c, n in SCL_CLASSES.items() if c not in DEFAULT_VALID_SCL
                },
            },
            "reflectance": "(DN + offset) / 10000, offset -1000 for processing baseline >= 04.00",
            "resolution_m": analysis.RESOLUTION_M,
        },
        indent=2,
    )


@mcp.resource(
    "stac://catalogs",
    name="catalogs",
    title="Catalogs and limits",
    description="Supported STAC catalogs, the default one and the server's request limits.",
    mime_type="application/json",
)
def catalogs_resource() -> str:
    s = get_settings()
    return json.dumps(
        {
            "default_catalog": s.catalog.value,
            "catalogs": [
                {
                    "name": c.name.value,
                    "title": c.title,
                    "url": c.url,
                    "sentinel2_collection": c.s2_collection,
                }
                for c in stac_catalog.CATALOGS.values()
            ],
            "limits": {
                "max_area_km2": s.max_area_km2,
                "max_time_series_scenes": s.max_scenes,
                "max_search_limit": s.max_search_limit,
            },
            "output_dir": str(s.output_dir),
        },
        indent=2,
    )


# --- Prompts --------------------------------------------------------------


@mcp.prompt(
    title="Analyse vegetation",
    description="Vegetation health of an area over the last months (NDVI time series).",
)
def analyze_vegetation(
    bbox: Annotated[str, Field(description="west,south,east,north in WGS84 degrees")],
    months: Annotated[str, Field(description="How many months back from today")] = "6",
) -> str:
    end = date.today()
    start = end - timedelta(days=round(30.44 * int(months)))
    return f"""\
Analyse the vegetation of the area with bbox [{bbox}] from {start} to {end}.

1. Call time_series with index="ndvi", bbox=[{bbox}], date_range="{start}/{end}".
2. Describe the trend: when NDVI peaked and dipped, and the overall change.
   Healthy vegetation is above ~0.6, sparse vegetation 0.2-0.5, bare soil or urban below 0.2.
3. Pick the clearest recent scene (highest valid_pixel_pct) and call compute_index with
   save="png" so I can look at the spatial pattern; report the file path.
4. Mention skipped dates and anything that limits confidence (few clear dates, low valid pixels).
Keep the answer short: a few sentences and a small table of dates and NDVI means.
"""


# --- Entry point ----------------------------------------------------------


def main() -> None:
    settings = get_settings()
    # stdout carries the MCP protocol: every log line must go to stderr.
    logging.basicConfig(
        stream=sys.stderr,
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,  # MCPServer installs its own handler at import time
    )
    log.info(
        "stac-eo-mcp %s starting (catalog=%s, output_dir=%s)",
        __version__,
        settings.catalog.value,
        settings.output_dir,
    )
    mcp.run("stdio")
