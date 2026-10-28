from datetime import date

import pytest

from stac_eo_mcp.errors import AreaTooLargeError, InvalidInputError
from stac_eo_mcp.models import (
    BBox,
    DateRange,
    SpectralIndex,
    check_cloud_cover,
    check_limit,
    parse_index,
)

ROME = [12.40, 41.85, 12.55, 41.95]


class TestBBox:
    @pytest.mark.parametrize(
        "value", [ROME, tuple(ROME), "12.40,41.85,12.55,41.95", "12.4 41.85 12.55 41.95"]
    )
    def test_accepts_common_shapes(self, value):
        assert BBox.parse(value).as_list() == ROME

    @pytest.mark.parametrize(
        ("value", "fragment"),
        [
            ([1, 2, 3], "expected 4 numbers"),
            ([0, 0, "a", 1], "must be numbers"),
            ([0, 0, float("nan"), 1], "finite"),
            ([-200, 0, 1, 1], "outside [-180, 180]"),
            ([0, -95, 1, 1], "outside [-90, 90]"),
            ([10, 0, 5, 1], "west (10.0) must be less than east"),
            ([0, 5, 1, 1], "south (5.0) must be less than north"),
            (42, "expected a list"),
        ],
    )
    def test_rejects_invalid(self, value, fragment):
        with pytest.raises(InvalidInputError) as exc:
            BBox.parse(value)
        assert fragment in str(exc.value)
        assert "Hint:" in str(exc.value)

    def test_area_one_degree_at_equator(self):
        # 1° x 1° at the equator is about 111.2 km x 110.6 km.
        assert BBox.parse([0, 0, 1, 1]).area_km2() == pytest.approx(12_364, rel=1e-3)

    def test_area_shrinks_with_latitude(self):
        assert BBox.parse([0, 60, 1, 61]).area_km2() < BBox.parse([0, 0, 1, 1]).area_km2() / 1.9

    def test_check_area(self):
        small = BBox.parse(ROME)  # ~ 12.4 km x 11.1 km
        small.check_area(200)
        with pytest.raises(AreaTooLargeError, match="above the 100 km² limit"):
            small.check_area(100)


class TestDateRange:
    TODAY = date(2026, 10, 28)

    @pytest.mark.parametrize(
        "value",
        [
            "2025-04-01/2025-09-30",
            ["2025-04-01", "2025-09-30"],
            ("2025-04-01T10:00:00Z", "2025-09-30T00:00:00+00:00"),
            [date(2025, 4, 1), date(2025, 9, 30)],
        ],
    )
    def test_accepts_common_shapes(self, value):
        dr = DateRange.parse(value, today=self.TODAY)
        assert (dr.start, dr.end) == (date(2025, 4, 1), date(2025, 9, 30))

    def test_to_stac_covers_full_days(self):
        dr = DateRange.parse("2025-04-01/2025-04-01", today=self.TODAY)
        assert dr.to_stac() == "2025-04-01T00:00:00Z/2025-04-01T23:59:59Z"
        assert dr.days == 1

    @pytest.mark.parametrize(
        ("value", "fragment"),
        [
            ("2025-04-01", "expected a start and an end"),
            ("2025-13-01/2025-12-01", "not an ISO date"),
            ("2025-09-30/2025-04-01", "is after end"),
            ("2027-01-01/2027-02-01", "in the future"),
            (123, "expected a string or list"),
        ],
    )
    def test_rejects_invalid(self, value, fragment):
        with pytest.raises(InvalidInputError) as exc:
            DateRange.parse(value, today=self.TODAY)
        assert fragment in str(exc.value)
        assert "YYYY-MM-DD/YYYY-MM-DD" in str(exc.value)


def test_parse_index():
    assert parse_index(" NDVI ") is SpectralIndex.NDVI
    with pytest.raises(InvalidInputError, match="ndvi, ndwi, nbr"):
        parse_index("evi")


def test_cloud_cover_and_limit():
    assert check_cloud_cover(None) is None
    assert check_cloud_cover(20) == 20.0
    with pytest.raises(InvalidInputError):
        check_cloud_cover(120)
    assert check_limit(10, 50) == 10
    with pytest.raises(InvalidInputError, match=r"\[1, 50\]"):
        check_limit(51, 50)
