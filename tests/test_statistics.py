"""Tests for statistical analysis and power analysis functions."""
import numpy as np
import pytest

from src.statistics import bonferroni_correction, bootstrap_slope_ci, power_analysis_slope


class TestBootstrapSlopeCI:
    def test_recovers_known_negative_slope(self):
        rng = np.random.default_rng(42)
        x = np.log2([1, 2, 3, 4, 6, 8, 12])
        y = 10.0 - 1.0 * x + rng.normal(0, 0.1, len(x))
        result = bootstrap_slope_ci(x, y, n_bootstrap=2000, seed=0)
        assert -1.5 < result["slope"] < -0.5, f"Slope {result['slope']:.3f} too far from -1"

    def test_ci_contains_true_slope(self):
        rng = np.random.default_rng(99)
        x = np.log2([1, 2, 3, 4, 6, 8, 12])
        y = 8.0 - 0.8 * x + rng.normal(0, 0.05, len(x))
        result = bootstrap_slope_ci(x, y, n_bootstrap=3000, seed=1)
        assert result["ci_lower"] <= -0.8 <= result["ci_upper"]

    def test_ci_contains_zero_for_flat_data(self):
        x = np.log2([1, 2, 3, 4, 6, 8, 12])
        y = np.ones_like(x) * 8.0
        result = bootstrap_slope_ci(x, y, n_bootstrap=1000, seed=2)
        assert result["ci_lower"] <= 0 <= result["ci_upper"]

    def test_ordering_ci_lower_lt_slope_lt_ci_upper(self):
        rng = np.random.default_rng(5)
        x = np.arange(7, dtype=float)
        y = -0.5 * x + rng.normal(0, 0.2, 7)
        result = bootstrap_slope_ci(x, y, n_bootstrap=500, seed=3)
        assert result["ci_lower"] <= result["slope"] <= result["ci_upper"]

    def test_required_output_keys(self):
        x = np.array([0.0, 1, 2, 3])
        y = np.array([8.0, 7, 6, 5])
        result = bootstrap_slope_ci(x, y, n_bootstrap=100)
        required = {"slope", "ci_lower", "ci_upper", "ci_level", "p_value",
                    "reject_null_slope_zero", "n_bootstrap"}
        assert required.issubset(result.keys())

    def test_p_value_in_unit_interval(self):
        x = np.log2([1, 2, 4, 8])
        y = np.array([10.0, 9, 8, 7])
        result = bootstrap_slope_ci(x, y, n_bootstrap=200)
        assert 0 <= result["p_value"] <= 1


class TestPowerAnalysis:
    def test_output_structure(self):
        result = power_analysis_slope()
        required = {"target_slope", "null_slope", "n_group_sizes", "n_seeds",
                    "power", "sufficient_power", "recommendation"}
        assert required.issubset(result.keys())

    def test_power_in_unit_interval(self):
        result = power_analysis_slope(target_slope=-1.0, n_seeds=5)
        assert 0 <= result["power"] <= 1

    def test_more_seeds_increases_power(self):
        r5  = power_analysis_slope(n_seeds=5)
        r10 = power_analysis_slope(n_seeds=10)
        assert r10["power"] >= r5["power"]

    def test_larger_effect_increases_power(self):
        r_small = power_analysis_slope(target_slope=-0.3)
        r_large = power_analysis_slope(target_slope=-1.0)
        assert r_large["power"] >= r_small["power"]

    def test_sufficient_power_flag(self):
        result = power_analysis_slope(target_slope=-1.0, n_seeds=5, n_group_sizes=7)
        assert isinstance(result["sufficient_power"], bool)


class TestBonferroniCorrection:
    def test_corrects_alpha(self):
        p_vals = {"H1": 0.01, "H2": 0.03, "H3": 0.10}
        result = bonferroni_correction(p_vals, alpha=0.05)
        corrected_alpha = 0.05 / 3
        for k, v in result.items():
            assert abs(v["corrected_alpha"] - corrected_alpha) < 1e-10

    def test_significance_flags(self):
        p_vals = {"small": 0.001, "large": 0.5}
        result = bonferroni_correction(p_vals, alpha=0.05)
        corrected = 0.05 / 2
        assert result["small"]["significant"] == (0.001 < corrected)
        assert result["large"]["significant"] == (0.5 < corrected)
