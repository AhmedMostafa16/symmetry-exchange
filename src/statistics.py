"""
Statistical design for the symmetry exchange rate experiment.

All methods are pre-registered before running experiments.
The bootstrap slope CI is the primary inferential tool; parametric tests
are secondary cross-checks only.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy import stats


# ─── Power analysis ───────────────────────────────────────────────────────────


def power_analysis_slope(
    target_slope: float = -1.0,
    null_slope: float = 0.0,
    n_group_sizes: int = 7,
    n_seeds: int = 5,
    alpha: float = 0.05,
    assumed_sigma_logN: float = 0.5,
) -> dict:
    """
    Analytical power analysis for OLS slope on log₂(N_target) ~ log₂(|G|).

    Pre-registered: requires power ≥ 0.80 before the experiment runs.
    If insufficient, the recommendation field specifies the minimum n_seeds needed.

    x-grid is fixed to log₂({1,2,3,4,6,8,12}).
    N_target per group size is the median over n_seeds — effective σ divides by √n_seeds.
    """
    x = np.log2([1, 2, 3, 4, 6, 8, 12])
    sigma_x = np.std(x)

    effective_sigma = assumed_sigma_logN / np.sqrt(n_seeds)
    se_slope = effective_sigma / (sigma_x * np.sqrt(n_group_sizes))

    t_stat = abs(target_slope - null_slope) / (se_slope + 1e-15)
    df = n_group_sizes - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df=df)

    power = float(
        1
        - stats.t.cdf(t_crit - t_stat, df=df)
        + stats.t.cdf(-t_crit - t_stat, df=df)
    )

    needed_seeds = int(np.ceil(n_seeds * (0.80 / max(power, 1e-6)) ** 2))
    recommendation = "proceed" if power >= 0.80 else f"increase seeds to {needed_seeds}"

    return {
        "target_slope": target_slope,
        "null_slope": null_slope,
        "n_group_sizes": n_group_sizes,
        "n_seeds": n_seeds,
        "sigma_x": float(sigma_x),
        "se_slope": float(se_slope),
        "power": power,
        "sufficient_power": power >= 0.80,
        "recommendation": recommendation,
    }


# ─── Bootstrap CI on OLS slope ────────────────────────────────────────────────


def bootstrap_slope_ci(
    log_group_sizes: np.ndarray,
    log_n_target: np.ndarray,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 999,
) -> dict:
    """
    Non-parametric bootstrap 95% CI on the OLS slope of log(N_target) ~ log(|G|).

    Resamples (x_i, y_i) pairs with replacement.
    This is the primary inferential estimator in the pre-registered analysis.

    p-value: two-sided proportion of bootstrap slopes on the wrong side of 0.
    """
    rng = np.random.default_rng(seed)
    x, y = np.asarray(log_group_sizes, dtype=float), np.asarray(log_n_target, dtype=float)
    n = len(x)

    def _ols_slope(xi: np.ndarray, yi: np.ndarray) -> float:
        xc = xi - xi.mean()
        denom = (xc ** 2).sum()
        if denom < 1e-15:
            return float("nan")
        return float((xc * yi).sum() / denom)

    observed_slope = _ols_slope(x, y)

    bs_slopes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bs_slopes[i] = _ols_slope(x[idx], y[idx])

    # Filter degenerate bootstrap samples (zero x-variance resamples → NaN slope)
    valid = bs_slopes[np.isfinite(bs_slopes)]
    if len(valid) == 0:
        return {
            "slope": observed_slope, "ci_lower": float("nan"),
            "ci_upper": float("nan"), "ci_level": ci_level,
            "p_value": float("nan"), "reject_null_slope_zero": False,
            "n_bootstrap": n_bootstrap, "n_valid_bootstrap": 0,
        }

    alpha = 1 - ci_level
    ci_lower = float(np.percentile(valid, 100 * alpha / 2))
    ci_upper = float(np.percentile(valid, 100 * (1 - alpha / 2)))

    p_value = float(2 * min(
        (valid >= 0).mean(),
        (valid <= 0).mean(),
    ))

    return {
        "slope": observed_slope,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": ci_level,
        "p_value": p_value,
        "reject_null_slope_zero": not (ci_lower <= 0 <= ci_upper),
        "n_bootstrap": n_bootstrap,
    }


# ─── Multiple testing correction ─────────────────────────────────────────────


def bonferroni_correction(p_values: dict[str, float], alpha: float = 0.05) -> dict:
    """
    Bonferroni correction for the three pre-registered comparisons:
      H1: slope(equivariant) < 0
      H2: slope(equivariant) < slope(wrong_group)
      H3: slope(equivariant) < slope(augmented) [informative, not primary]
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    return {
        k: {
            "p": v,
            "corrected_alpha": corrected_alpha,
            "significant": v < corrected_alpha,
        }
        for k, v in p_values.items()
    }


# ─── Pre-registration guard ───────────────────────────────────────────────────


def compute_analysis_hash(config: dict) -> str:
    """
    SHA-256 of the pre-registered config (sorted keys, JSON-serialised).
    Compare at analysis time to the hash logged at experiment start.
    A mismatch signals unregistered deviation — flag in the paper.
    """
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
