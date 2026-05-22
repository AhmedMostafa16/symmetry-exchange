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
        for key in ("slopes", "comparisons", "bonferroni", "failure_classification"):
            assert key in report

    def test_bonferroni_has_three_tests(self):
        df = _make_multi_model_df()
        tbl = compute_n_target_table(df)
        report = full_statistical_analysis(tbl)
        assert len(report["bonferroni"]) == 3


class TestClassifyFailureType:
    def test_failure_a_ci_contains_zero(self):
        report = {
            "slopes": {"equivariant": {"slope": -0.1, "ci_lower": -0.3, "ci_upper": 0.1}},
            "comparisons": {},
        }
        result = classify_failure_type(report)
        assert result["type"] == "A"

    def test_failure_e_regularisation_collapse(self):
        report = {
            "slopes": {
                "equivariant": {"slope": -0.5, "ci_lower": -0.9, "ci_upper": -0.1},
                "wrong_group": {"slope": -0.45, "ci_lower": -0.8, "ci_upper": -0.1},
            },
            "comparisons": {},
        }
        result = classify_failure_type(report)
        assert result["type"] == "E"

    def test_signal_genuine_advantage(self):
        report = {
            "slopes": {
                "equivariant": {"slope": -0.95, "ci_lower": -1.2, "ci_upper": -0.7},
                "wrong_group": {"slope": -0.05, "ci_lower": -0.2, "ci_upper": 0.1},
            },
            "comparisons": {
                "equiv_vs_wrong": {"slope_difference": -0.9, "ci_overlap": False,
                                   "h2_supported": True}
            },
        }
        result = classify_failure_type(report)
        assert result["type"] == "SIGNAL"
        assert result["matches_theory"]  # |slope| ≈ 1.0

    def test_ambiguous_returns_dict(self):
        report = {
            "slopes": {"equivariant": {"slope": -0.3, "ci_lower": -0.6, "ci_upper": -0.05}},
            "comparisons": {},
        }
        result = classify_failure_type(report)
        assert "type" in result
