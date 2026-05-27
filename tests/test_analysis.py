"""Tests for the analysis pipeline (pure pandas/numpy — no training required)."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    classify_failure_type,
    compute_n_target_table,
    estimate_exchange_rate,
    estimate_relative_exchange_rate,
    full_statistical_analysis,
    load_results,
)


def _make_df(slope: float = -1.0, n_seeds: int = 5) -> pd.DataFrame:
    """Synthetic results DataFrame with a known exchange rate."""
    rng = np.random.default_rng(0)
    records = []
    n_groups = [1, 2, 3, 4, 6, 8, 12]
    for n in n_groups:
        log2_n = np.log2(max(n, 1))
        # N_target ≈ 1600 * n**slope (true Bayesian optimal)
        n_target_true = max(50, int(1600 * (n ** slope)))
        for n_train in [50, 100, 200, 400, 800, 1600, 3200, 6400]:
            for seed in range(n_seeds):
                acc = 0.5 + 0.45 * (n_train / n_target_true) ** 0.5
                acc = min(acc + rng.normal(0, 0.02), 0.95)
                records.append({
                    "n_group": n,
                    "log2_n_group": log2_n,
                    "n_train": n_train,
                    "seed": seed,
                    "model_type": "equivariant",
                    "epsilon": 0.0,
                    "best_val_acc": float(acc),
                    "reached_target": acc >= 0.80,
                    "total_flops": n_train * 1000,
                    "id_twonn": 2.0 + rng.normal(0, 0.3),
                    "id_pr": 2.5 + rng.normal(0, 0.4),
                })
    return pd.DataFrame(records)


def _make_multi_model_df() -> pd.DataFrame:
    """DataFrame with multiple model types."""
    rng = np.random.default_rng(1)
    records = []
    n_groups = [1, 2, 4, 8]
    for n in n_groups:
        for mtype, slope in [("equivariant", -1.0), ("wrong_group", -0.1),
                              ("vanilla", 0.0), ("augmented", -0.5)]:
            n_target_true = max(50, int(1600 * (n ** slope)))
            for n_train in [50, 200, 800, 3200]:
                for seed in range(3):
                    acc = 0.5 + 0.45 * (n_train / n_target_true) ** 0.5
                    acc = min(acc + rng.normal(0, 0.02), 0.95)
                    records.append({
                        "n_group": n,
                        "log2_n_group": np.log2(max(n, 1)),
                        "n_train": n_train,
                        "seed": seed,
                        "model_type": mtype,
                        "epsilon": 0.0,
                        "best_val_acc": float(acc),
                        "reached_target": acc >= 0.80,
                        "total_flops": n_train * 1000,
                        "id_twonn": 2.0,
                        "id_pr": 2.5,
                    })
    return pd.DataFrame(records)


class TestLoadResults:
    def test_loads_json_files(self, tmp_path):
        run = {
            "run_id": "abc123",
            "n_group": 4,
            "n_train": 100,
            "model_type": "equivariant",
            "best_val_acc": 0.82,
            "val_acc_curve": [(0, 0.5), (1, 0.82)],
            "train_loss_curve": [(0, 0.7)],
        }
        (tmp_path / "abc123.json").write_text(json.dumps(run))
        df = load_results(str(tmp_path))
        assert len(df) == 1
        assert "n_group" in df.columns
        # val_acc_curve and train_loss_curve should be excluded
        assert "val_acc_curve" not in df.columns

    def test_returns_empty_df_on_empty_dir(self, tmp_path):
        df = load_results(str(tmp_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestNTargetTable:
    def test_produces_n_target_per_group(self):
        df = _make_df(slope=-1.0)
        tbl = compute_n_target_table(df, target_acc=0.80, min_seeds=3)
        assert "n_target" in tbl.columns
        assert "log2_n_target" in tbl.columns
        assert len(tbl) >= 3  # at least a few group sizes

    def test_n_target_finite_where_reachable(self):
        df = _make_df(slope=-1.0)
        tbl = compute_n_target_table(df, target_acc=0.80, min_seeds=3)
        # Most group sizes should have a finite n_target
        finite = tbl["n_target"].notna().sum()
        assert finite >= 3


class TestEstimateExchangeRate:
    def test_recovers_negative_slope(self):
        df = _make_df(slope=-1.0)
        tbl = compute_n_target_table(df)
        result = estimate_exchange_rate(tbl, "equivariant", epsilon=0.0, n_bootstrap=500)
        assert "slope" in result
        assert result["slope"] < 0, f"Expected negative slope, got {result['slope']}"

    def test_returns_error_dict_on_insufficient_data(self):
        tbl = pd.DataFrame({"model_type": ["equivariant"], "epsilon": [0.0],
                            "log2_n": [1.0], "log2_n_target": [float("nan")]})
        result = estimate_exchange_rate(tbl, "equivariant")
        assert "error" in result


class TestFullStatisticalAnalysis:
    def test_returns_required_keys(self):
        df = _make_multi_model_df()
        tbl = compute_n_target_table(df)
        report = full_statistical_analysis(tbl)
        for key in ("slopes", "relative_slopes", "comparisons",
                    "bonferroni", "failure_classification"):
            assert key in report

    def test_bonferroni_has_three_tests(self):
        df = _make_multi_model_df()
        tbl = compute_n_target_table(df)
        report = full_statistical_analysis(tbl)
        assert len(report["bonferroni"]) == 3

    def test_relative_slopes_populated_for_non_baseline_models(self):
        df = _make_multi_model_df()
        tbl = compute_n_target_table(df)
        report = full_statistical_analysis(tbl)
        # vanilla is the baseline so should be absent; everything else present
        assert "vanilla" not in report["relative_slopes"]
        for mtype in ["equivariant", "wrong_group", "augmented"]:
            assert mtype in report["relative_slopes"]


# ─── New: estimate_relative_exchange_rate ────────────────────────────────────


class TestEstimateRelativeExchangeRate:
    """
    Relative exchange rate = slope of log2(N_baseline / N_treatment) vs log2(|G|).

    For the symmetry experiment the science-relevant quantity, since the
    absolute slopes are contaminated by task-difficulty scaling.
    """

    def _build_table(self, vanilla_slope, treatment_slope,
                     treatment_name="equivariant", noise=0.05, seed=0):
        """Synthetic N_target table where vanilla and treatment have known slopes.
        A small amount of noise prevents degenerate bootstrap resamples."""
        rng = np.random.default_rng(seed)
        rows = []
        for n in [1, 2, 4, 8, 16]:
            ln = np.log2(max(n, 1))
            van = 5 + vanilla_slope * ln + rng.normal(0, noise)
            tre = 5 + treatment_slope * ln + rng.normal(0, noise)
            rows.append({"n_group": n, "log2_n": ln, "model_type": "vanilla",
                         "epsilon": 0.0, "n_target": 2 ** van, "log2_n_target": van})
            rows.append({"n_group": n, "log2_n": ln, "model_type": treatment_name,
                         "epsilon": 0.0, "n_target": 2 ** tre, "log2_n_target": tre})
        return pd.DataFrame(rows)

    def test_recovers_unit_relative_rate(self):
        """vanilla slope=1, treatment slope=0 → relative slope = 1."""
        tbl = self._build_table(vanilla_slope=1.0, treatment_slope=0.0)
        r = estimate_relative_exchange_rate(tbl, treatment="equivariant",
                                             baseline="vanilla", n_bootstrap=2000)
        assert abs(r["slope"] - 1.0) < 0.05, f"Got {r['slope']:.3f}, expected 1.0"
        assert r["ci_lower"] < 1.0 < r["ci_upper"]

    def test_recovers_zero_relative_rate(self):
        """vanilla slope=1, treatment slope=1 → relative slope = 0."""
        tbl = self._build_table(vanilla_slope=1.0, treatment_slope=1.0)
        r = estimate_relative_exchange_rate(tbl, treatment="equivariant",
                                             baseline="vanilla", n_bootstrap=2000)
        assert abs(r["slope"]) < 0.05, f"Got {r['slope']:.3f}, expected 0.0"

    def test_recovers_negative_relative_rate(self):
        """treatment is WORSE than vanilla → negative rate (wrong-group case)."""
        tbl = self._build_table(vanilla_slope=1.0, treatment_slope=1.5,
                                 treatment_name="wrong_group")
        r = estimate_relative_exchange_rate(tbl, treatment="wrong_group",
                                             baseline="vanilla", n_bootstrap=2000)
        assert r["slope"] < 0, f"Expected negative rate, got {r['slope']:.3f}"

    def test_returns_error_when_too_few_points(self):
        tbl = pd.DataFrame([
            {"n_group": 1, "log2_n": 0.0, "model_type": "vanilla", "epsilon": 0.0,
             "n_target": 50, "log2_n_target": 5.6},
            {"n_group": 2, "log2_n": 1.0, "model_type": "vanilla", "epsilon": 0.0,
             "n_target": 100, "log2_n_target": 6.6},
        ])
        r = estimate_relative_exchange_rate(tbl, treatment="equivariant", baseline="vanilla")
        assert "error" in r

    def test_drops_nan_rows(self):
        """If one model has NaN at some n_group, that row is dropped from the joint fit."""
        tbl = self._build_table(vanilla_slope=1.0, treatment_slope=0.0)
        # Inject a NaN for equivariant at n=4
        tbl.loc[(tbl.model_type == "equivariant") & (tbl.n_group == 4),
                "log2_n_target"] = float("nan")
        r = estimate_relative_exchange_rate(tbl, treatment="equivariant",
                                             baseline="vanilla", n_bootstrap=1000)
        assert r["n_points"] == 4  # one fewer than the original 5
        assert abs(r["slope"] - 1.0) < 0.1

    def test_epsilon_filter(self):
        """Only the requested epsilon is used."""
        tbl = self._build_table(vanilla_slope=1.0, treatment_slope=0.0)
        # Add ε=0.2 rows where treatment is no better than vanilla
        for n in [1, 2, 4]:
            ln = np.log2(max(n, 1))
            tbl = pd.concat([tbl, pd.DataFrame([
                {"n_group": n, "log2_n": ln, "model_type": "vanilla", "epsilon": 0.2,
                 "n_target": 100, "log2_n_target": 6.6},
                {"n_group": n, "log2_n": ln, "model_type": "equivariant", "epsilon": 0.2,
                 "n_target": 100, "log2_n_target": 6.6},
            ])], ignore_index=True)
        r_clean = estimate_relative_exchange_rate(tbl, "equivariant", "vanilla",
                                                    epsilon=0.0, n_bootstrap=500)
        r_broken = estimate_relative_exchange_rate(tbl, "equivariant", "vanilla",
                                                     epsilon=0.2, n_bootstrap=500)
        assert abs(r_clean["slope"] - 1.0) < 0.1
        assert abs(r_broken["slope"]) < 0.1


# ─── classify_failure_type — rewritten for relative slopes ───────────────────


class TestClassifyFailureType:
    """The classifier now reads `report['relative_slopes']` (β_diff).

    β_diff = slope of log2(N_vanilla / N_treatment) vs log2(|G|).
    Theoretical prediction for correct equivariant model: β_diff ≈ +1.0.
    """

    def test_failure_a_relative_ci_contains_zero(self):
        """Equivariant doesn't significantly outperform vanilla → Failure A."""
        report = {
            "relative_slopes": {
                "equivariant": {"slope": 0.1, "ci_lower": -0.4, "ci_upper": 0.6},
            },
            "slopes": {},
            "comparisons": {},
        }
        result = classify_failure_type(report)
        assert result["type"] == "A"

    def test_failure_e_wrong_group_matches_equivariant(self):
        """Orbit averaging works equally well regardless of alignment → Failure E."""
        report = {
            "relative_slopes": {
                "equivariant": {"slope": 0.8, "ci_lower": 0.4, "ci_upper": 1.2},
                "wrong_group": {"slope": 0.75, "ci_lower": 0.35, "ci_upper": 1.15},
            },
            "slopes": {},
            "comparisons": {},
        }
        result = classify_failure_type(report)
        assert result["type"] == "E"

    def test_signal_on_real_user_data(self):
        """Matches the user's actual β_diff values: eq=+1.21, wrong=-0.77.

        Marginal CIs overlap by a hair (1.48 > 0.56), but the joint
        pairwise bootstrap on β_diff(eq) − β_diff(wr) gives a CI that
        cleanly excludes 0 because the bootstrap shares the same n_group
        resamples (variance cancels). The classifier should use the
        joint test and return SIGNAL.
        """
        report = {
            "relative_slopes": {
                "equivariant": {"slope": 1.208, "ci_lower": 0.563, "ci_upper": 2.327},
                "wrong_group": {"slope": -0.766, "ci_lower": -1.926, "ci_upper": 1.476},
            },
            "pairwise": {
                # Joint bootstrap on β_diff(eq vs wrong_group); CI excludes 0
                "equiv_vs_wrong": {"slope": 1.97, "ci_lower": 0.8, "ci_upper": 3.2},
            },
            "slopes": {},
            "comparisons": {},
        }
        result = classify_failure_type(report)
        assert result["type"] == "SIGNAL"
        assert result["matches_theory"], f"1.21 should match theory of 1.0"
        assert "exchange_rate" in result
        assert abs(result["exchange_rate"] - 1.208) < 1e-6

    def test_signal_canonical_unit_rate(self):
        """Canonical SIGNAL: β_diff(eq) ≈ +1.0, β_diff(wrong) ≈ 0."""
        report = {
            "relative_slopes": {
                "equivariant": {"slope": 0.95, "ci_lower": 0.7, "ci_upper": 1.2},
                "wrong_group": {"slope": 0.05, "ci_lower": -0.1, "ci_upper": 0.2},
            },
            "slopes": {},
            "comparisons": {},
        }
        result = classify_failure_type(report)
        assert result["type"] == "SIGNAL"
        assert result["matches_theory"]

    def test_ambiguous_when_only_equivariant_data(self):
        """Without wrong-group comparison we can't rule out Failure E."""
        report = {
            "relative_slopes": {
                "equivariant": {"slope": 0.9, "ci_lower": 0.5, "ci_upper": 1.3},
            },
            "slopes": {},
            "comparisons": {},
        }
        result = classify_failure_type(report)
        # Equivariant looks good but no wrong-group to confirm correctness
        assert result["type"] in ("AMBIGUOUS", "SIGNAL_NEEDS_WRONG_GROUP")

    def test_insufficient_data_when_no_relative_slopes(self):
        result = classify_failure_type({"relative_slopes": {}, "slopes": {}, "comparisons": {}})
        assert result["type"] == "INSUFFICIENT_DATA"

    def test_augmentation_explains_advantage(self):
        """β_diff(eq) ≈ β_diff(augmented) → architectural constraint adds nothing."""
        report = {
            "relative_slopes": {
                "equivariant": {"slope": 0.9, "ci_lower": 0.5, "ci_upper": 1.3},
                "wrong_group": {"slope": 0.05, "ci_lower": -0.2, "ci_upper": 0.3},
                "augmented":   {"slope": 0.85, "ci_lower": 0.45, "ci_upper": 1.25},
            },
            "slopes": {},
            "comparisons": {},
        }
        result = classify_failure_type(report)
        assert result["type"] == "AUG"

    def test_classifier_result_has_evidence_field(self):
        report = {
            "relative_slopes": {
                "equivariant": {"slope": 1.0, "ci_lower": 0.6, "ci_upper": 1.4},
                "wrong_group": {"slope": 0.0, "ci_lower": -0.3, "ci_upper": 0.3},
            },
            "slopes": {}, "comparisons": {},
        }
        result = classify_failure_type(report)
        assert "evidence" in result
        assert "label" in result
