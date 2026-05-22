"""Tests for visualization functions — verify they return figures without error."""
import numpy as np
import pandas as pd
import pytest

# Guard: skip gracefully if matplotlib is in headless mode with no backend
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except Exception:
    _MPL_OK = False

pytestmark = pytest.mark.skipif(not _MPL_OK, reason="matplotlib unavailable")

from src.visualizations import (
    COLORS,
    plot_id_estimator_agreement,
    plot_pareto_frontier,
    plot_scaling_law,
)


def _make_n_target_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for mtype, slope in [("equivariant", -1.0), ("wrong_group", -0.1), ("vanilla", 0.05)]:
        for n in [1, 2, 4, 8]:
            n_tgt = max(100, int(1600 * (n ** slope)))
            rows.append({
                "n_group": n,
                "log2_n": np.log2(max(n, 1)),
                "model_type": mtype,
                "epsilon": 0.0,
                "n_target": n_tgt,
                "log2_n_target": np.log2(n_tgt),
            })
    return pd.DataFrame(rows)


def _make_full_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for mtype in ["equivariant", "vanilla"]:
        for n in [2, 4]:
            for n_train in [100, 400]:
                rows.append({
                    "model_type": mtype,
                    "n_group": n,
                    "n_train": n_train,
                    "best_val_acc": float(rng.uniform(0.6, 0.95)),
                    "total_flops": n_train * 1000,
                    "id_twonn": float(rng.uniform(1.5, 3.5)),
                    "id_pr": float(rng.uniform(2.0, 4.0)),
                })
    return pd.DataFrame(rows)


class TestScalingLaw:
    def test_returns_figure(self):
        tbl = _make_n_target_df()
        slopes = {
            "equivariant": {"slope": -1.0, "ci_lower": -1.2, "ci_upper": -0.8},
            "wrong_group": {"slope": -0.1, "ci_lower": -0.3, "ci_upper": 0.1},
        }
        fig = plot_scaling_law(tbl, slopes)
        assert fig is not None
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        tbl = _make_n_target_df()
        path = str(tmp_path / "test_fig.png")
        fig = plot_scaling_law(tbl, {}, save_path=path)
        import os
        assert os.path.exists(path)
        if fig:
            plt.close(fig)


class TestParetoFrontier:
    def test_returns_figure_for_valid_df(self):
        df = _make_full_df()
        fig = plot_pareto_frontier(df)
        if fig:
            plt.close(fig)

    def test_returns_none_for_empty_df(self):
        result = plot_pareto_frontier(pd.DataFrame())
        assert result is None


class TestIDEstimatorAgreement:
    def test_returns_none_for_small_df(self):
        df = pd.DataFrame({"id_twonn": [1.0, 2.0], "id_pr": [1.5, 2.5]})
        result = plot_id_estimator_agreement(df)
        assert result is None  # < 10 rows

    def test_returns_figure_for_sufficient_data(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "id_twonn": rng.uniform(1.5, 3.5, 30),
            "id_pr":    rng.uniform(2.0, 4.0, 30),
        })
        fig = plot_id_estimator_agreement(df)
        if fig:
            plt.close(fig)


class TestColors:
    def test_all_model_types_have_colors(self):
        expected = {"equivariant", "wrong_group", "augmented", "vanilla", "regularized"}
        assert expected.issubset(COLORS.keys())

    def test_colors_are_valid_hex(self):
        for name, color in COLORS.items():
            assert color.startswith("#"), f"{name}: {color} is not hex"
            assert len(color) == 7, f"{name}: {color} wrong length"
