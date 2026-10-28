import numpy as np
import pytest

from stac_eo_mcp.errors import DataUnavailableError
from stac_eo_mcp.indices import (
    DEFAULT_VALID_SCL,
    INDEX_DEFINITIONS,
    S2_BASELINE_OFFSET_DN,
    compute_index,
    compute_stats,
    describe_indices,
    normalized_difference,
    scl_mask,
    to_reflectance,
)
from stac_eo_mcp.models import SpectralIndex


def full(value, shape=(4, 4)):
    return np.full(shape, value, dtype=np.float32)


class TestReflectance:
    def test_scale_and_nodata(self):
        dn = np.array([[0, 1000], [5000, 10000]], dtype=np.uint16)
        refl = to_reflectance(dn)
        assert np.isnan(refl[0, 0])
        np.testing.assert_allclose(refl[~np.isnan(refl)], [0.1, 0.5, 1.0], rtol=1e-6)

    def test_baseline_offset(self):
        # Baseline >= 04.00: DN 2000 is reflectance 0.1, not 0.2.
        refl = to_reflectance(np.array([2000], dtype=np.uint16), offset_dn=S2_BASELINE_OFFSET_DN)
        assert refl[0] == pytest.approx(0.1)

    def test_offset_changes_ndvi(self):
        # Ignoring the offset biases NDVI low: this is why we correct it.
        nir_dn, red_dn = np.array([4000]), np.array([1500])
        raw = normalized_difference(to_reflectance(nir_dn), to_reflectance(red_dn))
        fixed = normalized_difference(
            to_reflectance(nir_dn, offset_dn=-1000), to_reflectance(red_dn, offset_dn=-1000)
        )
        assert raw[0] == pytest.approx(2500 / 5500, rel=1e-5)
        assert fixed[0] == pytest.approx(2500 / 3500, rel=1e-5)


class TestNormalizedDifference:
    def test_known_value(self):
        out = normalized_difference(full(0.5), full(0.1))
        np.testing.assert_allclose(out, 0.4 / 0.6, rtol=1e-6)

    def test_zero_denominator_is_nan(self):
        assert np.isnan(normalized_difference(np.array([0.0]), np.array([0.0]))[0])

    def test_nan_propagates(self):
        assert np.isnan(normalized_difference(np.array([np.nan]), np.array([0.1]))[0])

    def test_clipped_to_unit_range(self):
        # Negative reflectance after offset can push the ratio past 1.
        out = normalized_difference(np.array([0.2]), np.array([-0.05]))
        assert out[0] == 1.0


class TestComputeIndex:
    @pytest.mark.parametrize(
        ("index", "bands", "expected"),
        [
            (SpectralIndex.NDVI, {"nir": 0.4, "red": 0.1}, 0.6),
            (SpectralIndex.NDWI, {"green": 0.1, "nir": 0.3}, -0.5),
            (SpectralIndex.NBR, {"nir": 0.3, "swir22": 0.1}, 0.5),
        ],
    )
    def test_formulas(self, index, bands, expected):
        out = compute_index(index, {k: full(v) for k, v in bands.items()})
        np.testing.assert_allclose(out, expected, rtol=1e-6)

    def test_scl_masks_clouds_and_shadows(self):
        scl = np.array([[4, 9], [3, 6]], dtype=np.uint8)  # veg, cloud, shadow, water
        out = compute_index(
            SpectralIndex.NDVI, {"nir": full(0.4, (2, 2)), "red": full(0.1, (2, 2))}, scl=scl
        )
        assert np.isfinite(out[0, 0])
        assert np.isfinite(out[1, 1])
        assert np.isnan(out[0, 1])
        assert np.isnan(out[1, 0])

    def test_custom_valid_classes(self):
        scl = np.array([[4, 11]], dtype=np.uint8)
        out = compute_index(
            SpectralIndex.NDVI,
            {"nir": full(0.4, (1, 2)), "red": full(0.1, (1, 2))},
            scl=scl,
            valid_classes={4},
        )
        assert np.isfinite(out[0, 0])
        assert np.isnan(out[0, 1])

    def test_missing_band(self):
        with pytest.raises(DataUnavailableError, match="swir22"):
            compute_index(SpectralIndex.NBR, {"nir": full(0.3)})

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="SCL shape"):
            compute_index(
                SpectralIndex.NDVI, {"nir": full(0.4), "red": full(0.1)}, scl=np.zeros((2, 2))
            )


class TestStats:
    def test_basic(self):
        values = np.array([[0.2, 0.4], [np.nan, 0.6]], dtype=np.float32)
        s = compute_stats(values)
        assert (s.min, s.max, s.mean) == (0.2, 0.6, 0.4)
        assert s.std == pytest.approx(0.1633, abs=1e-4)
        assert (s.valid_pixels, s.total_pixels, s.valid_pixel_pct) == (3, 4, 75.0)

    def test_all_masked(self):
        s = compute_stats(np.full((3, 3), np.nan))
        assert s.mean is None
        assert s.min is None
        assert s.valid_pixels == 0
        assert s.valid_pixel_pct == 0.0

    def test_empty(self):
        assert compute_stats(np.array([])).total_pixels == 0

    def test_end_to_end_with_clouds(self):
        rng = np.random.default_rng(0)
        nir_dn = rng.integers(3000, 5000, (50, 50)).astype(np.uint16)
        red_dn = rng.integers(1100, 1500, (50, 50)).astype(np.uint16)
        scl = np.full((50, 50), 4, dtype=np.uint8)
        scl[:10] = 9  # 20% cloud
        bands = {
            "nir": to_reflectance(nir_dn, offset_dn=-1000),
            "red": to_reflectance(red_dn, offset_dn=-1000),
        }
        s = compute_stats(compute_index(SpectralIndex.NDVI, bands, scl=scl))
        assert s.valid_pixel_pct == 80.0
        assert 0.6 < s.mean < 1.0


def test_scl_mask_default():
    scl = np.arange(12)
    assert set(np.flatnonzero(scl_mask(scl))) == set(DEFAULT_VALID_SCL)


def test_describe_indices_covers_all():
    described = describe_indices()
    assert {d["index"] for d in described} == {i.value for i in SpectralIndex}
    assert all(
        set(d["bands"]) == set(INDEX_DEFINITIONS[SpectralIndex(d["index"])].bands)
        for d in described
    )
