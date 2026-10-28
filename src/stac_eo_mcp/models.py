"""Input validation and compact output models for the MCP tools.

Inputs are parsed leniently (lists, tuples or strings) but validated strictly,
and every failure raises InvalidInputError with a hint the agent can act on.
"""

import math
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from stac_eo_mcp.errors import AreaTooLargeError, InvalidInputError

EARTH_RADIUS_KM = 6371.0088


class SpectralIndex(StrEnum):
    NDVI = "ndvi"
    NDWI = "ndwi"
    NBR = "nbr"


def parse_index(value: str) -> SpectralIndex:
    try:
        return SpectralIndex(value.strip().lower())
    except ValueError:
        options = ", ".join(i.value for i in SpectralIndex)
        raise InvalidInputError(
            f"Unknown index '{value}'.", hint=f"Use one of: {options}."
        ) from None


# --- Bounding box ---------------------------------------------------------


class BBox(BaseModel):
    """Bounding box in WGS84 degrees: west, south, east, north."""

    west: float
    south: float
    east: float
    north: float

    @model_validator(mode="after")
    def _check(self) -> Self:
        for name in ("west", "east"):
            v = getattr(self, name)
            if not -180 <= v <= 180:
                raise ValueError(f"{name}={v} is outside [-180, 180]")
        for name in ("south", "north"):
            v = getattr(self, name)
            if not -90 <= v <= 90:
                raise ValueError(f"{name}={v} is outside [-90, 90]")
        if self.west >= self.east:
            raise ValueError(
                f"west ({self.west}) must be less than east ({self.east}); "
                "boxes crossing the antimeridian are not supported"
            )
        if self.south >= self.north:
            raise ValueError(f"south ({self.south}) must be less than north ({self.north})")
        return self

    @classmethod
    def parse(cls, value: Any) -> "BBox":
        """Accept [w, s, e, n], (w, s, e, n), 'w,s,e,n' or a BBox."""
        if isinstance(value, BBox):
            return value
        if isinstance(value, str):
            parts: list[Any] = [p for p in value.replace(" ", ",").split(",") if p]
        elif isinstance(value, list | tuple):
            parts = list(value)
        else:
            raise _bbox_error(f"expected a list of 4 numbers, got {type(value).__name__}")
        if len(parts) != 4:
            raise _bbox_error(f"expected 4 numbers, got {len(parts)}")
        try:
            w, s, e, n = (float(p) for p in parts)
        except (TypeError, ValueError):
            raise _bbox_error(f"values must be numbers, got {parts}") from None
        if any(math.isnan(x) or math.isinf(x) for x in (w, s, e, n)):
            raise _bbox_error("values must be finite numbers")
        try:
            return cls(west=w, south=s, east=e, north=n)
        except ValueError as exc:
            msg = exc.errors()[0]["msg"] if hasattr(exc, "errors") else str(exc)
            raise _bbox_error(msg.removeprefix("Value error, ")) from None

    def as_list(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]

    def area_km2(self) -> float:
        """Area on a spherical Earth, exact for a lat/lon rectangle."""
        dlon = math.radians(self.east - self.west)
        dsin = math.sin(math.radians(self.north)) - math.sin(math.radians(self.south))
        return EARTH_RADIUS_KM**2 * dlon * abs(dsin)

    def check_area(self, max_km2: float) -> None:
        area = self.area_km2()
        if area > max_km2:
            raise AreaTooLargeError(
                f"The bbox covers {area:,.0f} km², above the {max_km2:,.0f} km² limit.",
                hint=(
                    "Use a smaller bbox (about 0.09° x 0.09° is ~100 km² at mid latitudes) "
                    "or raise STAC_EO_MCP_MAX_AREA_KM2."
                ),
            )


def _bbox_error(detail: str) -> InvalidInputError:
    return InvalidInputError(
        f"Invalid bbox: {detail}.",
        hint="Pass [west, south, east, north] in WGS84 degrees, e.g. [12.40, 41.85, 12.55, 41.95].",
    )


# --- Date range -----------------------------------------------------------


class DateRange(BaseModel):
    start: date
    end: date

    @classmethod
    def parse(cls, value: Any, *, today: date | None = None) -> "DateRange":
        """Accept 'YYYY-MM-DD/YYYY-MM-DD', [start, end] or a DateRange."""
        if isinstance(value, DateRange):
            return value
        if isinstance(value, str):
            parts: list[Any] = [p.strip() for p in value.split("/")]
        elif isinstance(value, list | tuple):
            parts = list(value)
        else:
            raise _date_error(f"expected a string or list, got {type(value).__name__}")
        if len(parts) != 2:
            raise _date_error(f"expected a start and an end, got {len(parts)} value(s)")
        start, end = (_parse_date(p) for p in parts)
        if start > end:
            raise _date_error(f"start {start} is after end {end}")
        today = today or date.today()
        if start > today:
            raise _date_error(f"start {start} is in the future (today is {today})")
        return cls(start=start, end=end)

    def to_stac(self) -> str:
        """STAC datetime interval covering both days fully."""
        return f"{self.start.isoformat()}T00:00:00Z/{self.end.isoformat()}T23:59:59Z"

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


def _parse_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            return date.fromisoformat(text)
        except ValueError:
            try:
                return datetime.fromisoformat(text).date()
            except ValueError:
                pass
    raise _date_error(f"'{value}' is not an ISO date")


def _date_error(detail: str) -> InvalidInputError:
    return InvalidInputError(
        f"Invalid date_range: {detail}.",
        hint="Use 'YYYY-MM-DD/YYYY-MM-DD', e.g. '2025-04-01/2025-09-30'.",
    )


# --- Scalar checks --------------------------------------------------------


def check_cloud_cover(value: float | None) -> float | None:
    if value is None:
        return None
    if not 0 <= value <= 100:
        raise InvalidInputError(
            f"max_cloud_cover={value} is outside [0, 100].",
            hint="It is a percentage; 20 is a sensible default.",
        )
    return float(value)


def check_limit(value: int, maximum: int) -> int:
    if not 1 <= value <= maximum:
        raise InvalidInputError(
            f"limit={value} is outside [1, {maximum}].",
            hint="Narrow the date range or bbox instead of asking for more scenes.",
        )
    return value


# --- Outputs (compact, LLM friendly) --------------------------------------


class CollectionSummary(BaseModel):
    id: str
    title: str | None = None
    description: str | None = None


class SceneSummary(BaseModel):
    id: str
    datetime: str | None
    cloud_cover: float | None = Field(None, description="Percent, from eo:cloud_cover")
    bbox: list[float] | None = None
    footprint: dict[str, Any] | None = Field(None, description="Simplified GeoJSON geometry")
    preview_url: str | None = None


class SearchResult(BaseModel):
    catalog: str
    collection: str
    returned: int
    scenes: list[SceneSummary]


class AssetInfo(BaseModel):
    key: str
    title: str | None = None
    media_type: str | None = None
    roles: list[str] = []
    gsd: float | None = Field(None, description="Ground sample distance in metres")


class SceneDetails(BaseModel):
    id: str
    collection: str
    datetime: str | None
    bbox: list[float] | None = None
    properties: dict[str, Any]
    assets: list[AssetInfo]


class IndexStats(BaseModel):
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    valid_pixel_pct: float = Field(description="Share of pixels kept after cloud/no-data masking")
    valid_pixels: int
    total_pixels: int


class IndexResult(BaseModel):
    item_id: str
    index: SpectralIndex
    datetime: str | None
    bbox: list[float]
    stats: IndexStats
    outputs: dict[str, str] = Field(default_factory=dict, description="format -> file path")


class TimeSeriesPoint(BaseModel):
    date: str
    item_id: str
    mean: float | None
    valid_pixel_pct: float


class SkippedScene(BaseModel):
    item_id: str
    reason: str


class TimeSeriesResult(BaseModel):
    index: SpectralIndex
    bbox: list[float]
    date_range: str
    scenes_found: int
    points: list[TimeSeriesPoint]
    skipped: list[SkippedScene] = []
