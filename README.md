# stac-eo-mcp

[![CI](https://github.com/bbrauzzi/stac-eo-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/bbrauzzi/stac-eo-mcp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/stac-eo-mcp)](https://pypi.org/project/stac-eo-mcp/)
[![Python](https://img.shields.io/pypi/pyversions/stac-eo-mcp)](https://pypi.org/project/stac-eo-mcp/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

An [MCP](https://modelcontextprotocol.io) server that lets AI agents (Claude Desktop, Claude Code,
Cursor) **search satellite imagery in STAC catalogs and measure vegetation, water and burn
indices** on Sentinel-2 scenes, without downloading whole images.

Ask *"How green was Villa Borghese in Rome during 2025?"* and the agent finds the scenes, reads only
the pixels inside your area, masks clouds and answers with numbers, a map and a trend.

<p align="center">
  <img src="https://raw.githubusercontent.com/bbrauzzi/stac-eo-mcp/main/docs/ndvi_map_rome.png" width="48%" alt="NDVI map of Villa Borghese, Rome, 30 July 2025">
  <img src="https://raw.githubusercontent.com/bbrauzzi/stac-eo-mcp/main/docs/ndvi_timeseries_rome.png" width="50%" alt="NDVI time series over Villa Borghese, 2025">
</p>
<p align="center"><sub>Produced by <code>compute_index</code> and <code>time_series</code> on real data.
White pixels are clouds and shadows masked with the Sentinel-2 SCL layer.</sub></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/bbrauzzi/stac-eo-mcp/main/docs/demo.gif" width="100%"
       alt="Demo: install with claude mcp add, ask Claude Code about Villa Borghese, get NDVI numbers and a map">
</p>
<p align="center"><sub>Install in Claude Code, ask a question, get numbers and a map
(a real session, replayed faster).</sub></p>

## Features

- **Two public catalogs**: [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com)
  (assets signed automatically) and [Element84 Earth Search](https://earth-search.aws.element84.com/v1).
  No account or API key needed.
- **Windowed reads**: only the bbox is read from Cloud-Optimized GeoTIFFs with
  [odc-stac](https://github.com/opendatacube/odc-stac), at 10 m in the scene's UTM grid.
- **Correct reflectance**: applies the Sentinel-2 processing baseline 04.00 offset (−1000 DN),
  from asset metadata when present, otherwise from the baseline. Both catalogs give identical
  numbers for the same scene.
- **Cloud masking with SCL**, reported as `valid_pixel_pct`. Tile cloud cover often differs from
  your area (a "9.7% cloudy" tile can be 100% cloudy over your bbox), so this matters.
- **LLM-friendly output**: statistics, dates and file paths only; never raster arrays. Footprints
  are simplified. Errors say what went wrong *and what to try next*.
- **Safe by default**: area and scene-count limits stop accidental large downloads.

## Tools

| Tool | What it does |
|---|---|
| `list_collections(catalog?)` | Collections in a catalog, with one-sentence descriptions |
| `search_scenes(collection, bbox, date_range, max_cloud_cover?, limit?, catalog?)` | Scenes newest first: id, date, cloud cover, footprint, preview link |
| `get_scene_details(collection, item_id, catalog?)` | Key metadata and the list of assets |
| `compute_index(item_id, index, bbox, save?, catalog?)` | NDVI / NDWI / NBR on one scene: min, max, mean, std, % valid pixels; optional PNG / GeoTIFF |
| `time_series(index, bbox, date_range, max_cloud_cover?, catalog?)` | Mean index per acquisition date, cloudy dates skipped with a reason, plus a trend summary |

**Resources**: `stac://indices` (formulas, bands, interpretation, cloud mask classes) and
`stac://catalogs` (catalogs, defaults and limits).
**Prompt**: `analyze_vegetation(bbox, months=6)`, a guided vegetation analysis.

Conventions: `bbox` is `[west, south, east, north]` in WGS84 degrees; `date_range` is
`"YYYY-MM-DD/YYYY-MM-DD"`.

## Install

Requires [uv](https://docs.astral.sh/uv/). The server runs locally over stdio; nothing to deploy.

```bash
uvx stac-eo-mcp   # runs the server (it waits for an MCP client on stdin)
```

### Claude Desktop

Add to `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "stac-eo-mcp": {
      "command": "uvx",
      "args": ["stac-eo-mcp"],
      "env": {
        "STAC_EO_MCP_OUTPUT_DIR": "/Users/you/stac-eo-mcp-output"
      }
    }
  }
}
```

If Claude Desktop cannot find `uvx`, use its full path (`which uvx`).

### Claude Code

```bash
claude mcp add stac-eo-mcp -- uvx stac-eo-mcp
# with options:
claude mcp add stac-eo-mcp -e STAC_EO_MCP_CATALOG=earth-search -- uvx stac-eo-mcp
```

### Cursor

`.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "stac-eo-mcp": { "command": "uvx", "args": ["stac-eo-mcp"] }
  }
}
```

## Configuration

All optional, via environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `STAC_EO_MCP_CATALOG` | `planetary-computer` | Default catalog (`planetary-computer` or `earth-search`) |
| `STAC_EO_MCP_OUTPUT_DIR` | `~/stac-eo-mcp-output` | Where PNG / GeoTIFF files are written |
| `STAC_EO_MCP_MAX_AREA_KM2` | `100` | Largest bbox for pixel reads |
| `STAC_EO_MCP_MAX_SCENES` | `24` | Most dates read by `time_series` (more are sampled evenly) |
| `STAC_EO_MCP_MAX_SEARCH_LIMIT` | `50` | Most scenes returned by `search_scenes` |
| `STAC_EO_MCP_LOG_LEVEL` | `INFO` | Log level; logs go to stderr only |

## Example conversation

> **You:** How did vegetation in Villa Borghese, Rome (bbox 12.47, 41.905, 12.50, 41.925) change
> during 2025?
>
> **Claude** *calls `time_series(index="ndvi", bbox=[12.47, 41.905, 12.5, 41.925], date_range="2025-01-01/2025-12-31")`*
>
> **Claude:** I analysed 24 of the 51 Sentinel-2 dates in 2025 (sampled evenly); 22 had enough
> clear pixels. Mean NDVI stays between 0.28 and 0.39, typical of a park mixed with city blocks:
>
> | Date | Mean NDVI |
> |---|---|
> | 2025-01-04 | 0.37 |
> | 2025-05-01 | 0.35 |
> | 2025-07-03 | **0.28** (minimum) |
> | 2025-09-28 | 0.35 |
> | 2025-11-05 | **0.39** (maximum) |
>
> The summer dip and autumn recovery is the usual Mediterranean pattern: dry-season stress in
> July, regrowth after the autumn rains. 10 July and 16 October were skipped (fully cloudy over
> the park even though the tiles reported under 30% cloud).
>
> **You:** Show me the map for the end of July.
>
> **Claude** *calls `search_scenes(...)`, then `compute_index(item_id="S2A_MSIL2A_20250730T100051_R122_T33TTG_20250730T153414", index="ndvi", bbox=[...], save="png")`*
>
> **Claude:** 30 July is 91% clear, mean NDVI 0.32 (max 0.90 on the park's tree cover, negative on
> the Tiber). Map saved to `~/stac-eo-mcp-output/S2A_MSIL2A_20250730T…_ndvi_c75d9e23.png`.

Other questions that work well:

- *"Find Sentinel-2 scenes under 10% cloud over Lake Trasimeno in August 2025 and compute NDWI on the clearest one."*
- *"Compare NBR before and after a wildfire"*: two `compute_index` calls; dNBR = NBR<sub>pre</sub> − NBR<sub>post</sub>.
- *"Which collections does Earth Search have?"*

## How it works

```
agent ──MCP/stdio──▶ server.py ──▶ analysis.py ──▶ catalog.py  (pystac-client: search, signing, band names)
                                         │
                                         ├──────▶ raster.py   (odc-stac window read, PNG/GeoTIFF)
                                         └──────▶ indices.py  (numpy: reflectance, SCL mask, stats)
```

- `models.py` validates inputs (WGS84 bbox, ISO dates, limits) and defines compact outputs.
- `indices.py` is pure numpy and fully unit-tested on synthetic arrays.
- Blocking I/O runs in worker threads; `time_series` reads up to 4 scenes in parallel.
- Where two Sentinel-2 tiles overlap, `time_series` keeps one scene per day (best bbox coverage,
  then lowest cloud cover).

## Development

```bash
git clone https://github.com/bbrauzzi/stac-eo-mcp && cd stac-eo-mcp
uv sync
uv run pytest            # fast tests (no network)
uv run pytest -m slow    # live tests against both catalogs (~1-2 min)
uv run ruff check . && uv run ruff format --check .
npx @modelcontextprotocol/inspector uv run stac-eo-mcp   # try it in MCP Inspector
uv run python scripts/make_readme_figures.py             # regenerate the README figures
```

## Limitations

- Indices are Sentinel-2 L2A only; search and details work on any collection.
- Bounding boxes crossing the antimeridian are not supported.
- The mask keeps snow/ice pixels (SCL class 11); exclude snowy periods for vegetation work.

## License

MIT. Sentinel-2 data: Copernicus Sentinel data, processed by ESA.
