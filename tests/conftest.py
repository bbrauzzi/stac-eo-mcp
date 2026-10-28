from datetime import UTC, datetime

import pystac
import pytest

ROME_BBOX = [12.40, 41.85, 12.55, 41.95]


def make_item(
    item_id: str = "S2A_TEST",
    dt: datetime = datetime(2025, 7, 10, 10, 0, tzinfo=UTC),
    cloud: float | None = 9.7071,
    baseline: str | None = "05.11",
    assets: dict[str, dict] | None = None,
    collection: str = "sentinel-2-l2a",
) -> pystac.Item:
    w, s, e, n = ROME_BBOX
    geom = {
        "type": "Polygon",
        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
    }
    props = {"platform": "sentinel-2a", "s2:processing_baseline": baseline, "eo:cloud_cover": cloud}
    item = pystac.Item(
        item_id, geom, ROME_BBOX, dt, {k: v for k, v in props.items() if v is not None},
        collection=collection,
    )  # fmt: skip
    for key, extra in (assets or {}).items():
        item.add_asset(
            key,
            pystac.Asset(
                href=f"https://example.com/{key}.tif",
                title=extra.pop("title", key),
                media_type=extra.pop("type", pystac.MediaType.COG),
                roles=extra.pop("roles", ["data"]),
                extra_fields=extra,
            ),
        )
    return item


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def pc_item() -> pystac.Item:
    return make_item(
        assets={
            "B03": {"gsd": 10},
            "B04": {"gsd": 10},
            "B08": {"gsd": 10},
            "B12": {"gsd": 20},
            "SCL": {"gsd": 20},
            "rendered_preview": {"type": "image/png", "roles": ["overview"]},
        }
    )


@pytest.fixture
def es_item() -> pystac.Item:
    rb = [{"nodata": 0, "data_type": "uint16", "scale": 0.0001, "offset": -0.1}]
    return make_item(
        item_id="S2A_T33TTG_20250710T100622_L2A",
        collection="sentinel-2-c1-l2a",
        assets={
            "red": {"gsd": 10, "raster:bands": rb},
            "nir": {"gsd": 10, "raster:bands": rb},
            "scl": {"gsd": 20, "raster:bands": [{"nodata": 0, "data_type": "uint8"}]},
            "thumbnail": {"type": "image/jpeg", "roles": ["thumbnail"]},
        },
    )
