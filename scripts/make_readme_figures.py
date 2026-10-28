"""Regenerate the README figures from live data.

    uv run python scripts/make_readme_figures.py

Writes docs/ndvi_map_rome.png and docs/ndvi_timeseries_rome.png.
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from stac_eo_mcp import analysis
from stac_eo_mcp import catalog as stac_catalog
from stac_eo_mcp.config import CatalogName, Settings
from stac_eo_mcp.models import BBox, DateRange, SpectralIndex

DOCS = Path(__file__).resolve().parent.parent / "docs"
BBOX = BBox.parse([12.470, 41.905, 12.500, 41.925])  # Villa Borghese, Rome
CAT = CatalogName.PLANETARY_COMPUTER

SURFACE, INK, MUTED, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6"


def style(ax, fig):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)


def timeseries_figure() -> None:
    res = analysis.time_series(
        Settings(), CAT, SpectralIndex.NDVI, BBOX, DateRange.parse("2025-01-01/2025-12-31"), 30
    )
    dates = [p.date for p in res.points]
    means = [p.mean for p in res.points]
    import datetime as dt

    x = [dt.date.fromisoformat(d) for d in dates]
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=150)
    style(ax, fig)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.plot(x, means, color=SERIES, linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5)  # fmt: skip
    for p in (res.summary.min, res.summary.max):
        d = dt.date.fromisoformat(p.date)
        ax.annotate(f"{p.mean:.2f}", (d, p.mean), textcoords="offset points",
                    xytext=(0, 9 if p is res.summary.max else -15), ha="center",
                    fontsize=9, color=INK)  # fmt: skip
    ax.set_ylim(0.2, 0.45)
    ax.set_ylabel("Mean NDVI", color=INK, fontsize=9)
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b"))
    ax.set_title("NDVI over Villa Borghese, Rome, 2025 (one point per clear Sentinel-2 date)",
                 loc="left", fontsize=10, color=INK)  # fmt: skip
    fig.tight_layout()
    fig.savefig(DOCS / "ndvi_timeseries_rome.png", facecolor=SURFACE)
    print("time series:", len(res.points), "points")


def map_figure() -> None:
    cfg = stac_catalog.get_config(CAT)
    items = stac_catalog.search_items(
        CAT, cfg.s2_collection, BBOX, DateRange.parse("2025-07-30/2025-07-30"), 20
    )
    item = analysis.one_per_day(items, BBOX)[0]
    da, stats = analysis.index_on_item(CAT, item, SpectralIndex.NDVI, BBOX)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    style(ax, fig)
    cmap = plt.get_cmap("RdYlGn").with_extremes(bad=SURFACE)
    im = ax.imshow(da.values, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.set_title(f"NDVI, 30 Jul 2025, mean {stats.mean:.2f}, {stats.valid_pixel_pct:.0f}% clear",
                 loc="left", fontsize=10, color=INK)  # fmt: skip
    fig.tight_layout()
    fig.savefig(DOCS / "ndvi_map_rome.png", facecolor=SURFACE)
    print("map:", item.id)


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    map_figure()
    timeseries_figure()
