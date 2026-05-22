"""
Publication figures for the symmetry exchange rate experiment.

WARNING on error bands: bands show seed variance, NOT bootstrap CIs on the slope.
These are different quantities — label them correctly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

COLORS = {
    "equivariant":  "#2196F3",
    "wrong_group":  "#FF5722",
    "augmented":    "#4CAF50",
    "vanilla":      "#9E9E9E",
    "regularized":  "#FF9800",
}

_LABEL = {
    "equivariant": "C_n equivariant (correct)",
    "wrong_group": "Equivariant (wrong group)",
    "augmented":   "Vanilla + augmentation",
    "vanilla":     "Vanilla MLP",
    "regularized": "Regularised MLP",
}


def _save(fig, path: Optional[str]) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_scaling_law(
    n_target_df: pd.DataFrame,
    slopes: dict,
    save_path: Optional[str] = None,
):
    """
    Figure 1 — Primary result.

    x: log₂(|G|) — bits of group structure
    y: log₂(N_target) — samples to 80% accuracy

    Anti-pattern to detect: if all lines have the same slope → Failure E.
    Error bands = seed variance, NOT bootstrap CI on slope. Label accordingly.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    x_range = np.linspace(0, np.log2(12) + 0.3, 100)

    for mtype, color in COLORS.items():
        sub = n_target_df[
            (n_target_df["model_type"] == mtype)
            & (n_target_df["epsilon"] == 0.0)
            & n_target_df["log2_n_target"].notna()
        ]
        if sub.empty:
            continue

        ax.scatter(sub["log2_n"], sub["log2_n_target"],
                   color=color, s=50, zorder=5, alpha=0.85)

        s = slopes.get(mtype, {})
        if "slope" in s:
            beta = s["slope"]
            xi = sub["log2_n"].values
            yi = sub["log2_n_target"].values
            intercept = yi.mean() - beta * xi.mean()
            ax.plot(x_range, intercept + beta * x_range, color=color, lw=2,
                    label=(f"{_LABEL[mtype]}\n"
                           f"  β₁={beta:.2f} [{s['ci_lower']:.2f}, {s['ci_upper']:.2f}]"))

    # Theory prediction line (slope = −1)
    van_sub = n_target_df[
        (n_target_df["model_type"] == "vanilla") & (n_target_df["n_group"] == 1)
    ]
    if not van_sub.empty:
        intercept_theory = van_sub["log2_n_target"].median()
        ax.plot(x_range, intercept_theory - x_range, "b--", lw=1.2, alpha=0.45,
                label="Theory prediction (β₁ = −1)")

    ax.set_xlabel("log₂(|G|) — bits of group structure", fontsize=12)
    ax.set_ylabel("log₂(N_target) — samples to 80% accuracy", fontsize=12)
    ax.set_title(
        "Sample-Complexity Exchange Rate\n"
        "Does one bit of correct symmetry buy one bit of data?",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.02,
            "Error bands = seed variance, NOT bootstrap CI on slope",
            transform=ax.transAxes, fontsize=7, color="grey", style="italic")

    _save(fig, save_path)
    return fig


def plot_epsilon_degradation(
    n_target_df: pd.DataFrame,
    slopes_by_epsilon: dict[float, dict],
    save_path: Optional[str] = None,
):
    """
    Figure 2 — Graceful degradation test.

    x: epsilon (symmetry-breaking magnitude)
    y: exchange rate (|slope|) per model type

    Expected: equivariant rate degrades smoothly; wrong-group stays flat.
    Brittle advantage (Failure D) appears as a cliff near epsilon = 0.

    INTERPRETATION: even partial degradation must exceed wrong-group at all
    epsilon values to support the genuine structural advantage claim.
    """
    import matplotlib.pyplot as plt

    epsilons = sorted(slopes_by_epsilon.keys())
    fig, ax = plt.subplots(figsize=(6, 4))

    for mtype, color in COLORS.items():
        rates = []
        for eps in epsilons:
            s = slopes_by_epsilon.get(eps, {}).get(mtype, {})
            rates.append(abs(s.get("slope", float("nan"))))

        valid = [(e, r) for e, r in zip(epsilons, rates) if np.isfinite(r)]
        if not valid:
            continue
        ex, ry = zip(*valid)
        ax.plot(ex, ry, "o-", color=color, label=_LABEL[mtype], lw=2)

    ax.set_xlabel("ε (symmetry-breaking fraction)", fontsize=12)
    ax.set_ylabel("|Exchange rate| = |β₁|", fontsize=12)
    ax.set_title("Graceful Degradation: Exchange Rate vs. Symmetry Breaking", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    _save(fig, save_path)
    return fig


def plot_pareto_frontier(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Figure 3 — Accuracy vs. total FLOPs Pareto frontier.

    CRITICAL: this shows FLOPs efficiency, NOT sample efficiency.
    If equivariant is NOT on the Pareto frontier, it means structural advantage
    is purely in data efficiency, not compute efficiency.
    Pre-register which outcome confirms vs. refutes which claim.
    """
    import matplotlib.pyplot as plt

    if df.empty or "total_flops" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))

    for mtype, color in COLORS.items():
        sub = df[df["model_type"] == mtype]
        if sub.empty:
            continue
        ax.scatter(
            np.log10(sub["total_flops"].clip(lower=1)),
            sub["best_val_acc"],
            color=color, alpha=0.5, s=20, label=_LABEL[mtype],
        )

    ax.set_xlabel("log₁₀(Total FLOPs)", fontsize=12)
    ax.set_ylabel("Best val accuracy", fontsize=12)
    ax.set_title(
        "Accuracy vs. Compute\n"
        "(Data efficiency ≠ compute efficiency — interpret carefully)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    _save(fig, save_path)
    return fig


def plot_id_estimator_agreement(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Figure 4 (Diagnostic) — ID estimator correlation matrix.

    High agreement = measurements are internally consistent.
    High agreement does NOT prove representations have a specific geometry.
    Low agreement (<0.3 correlation) = flag all ID conclusions as unreliable.
    """
    import matplotlib.pyplot as plt

    id_cols = ["id_twonn", "id_pr"]
    valid_df = df[id_cols].dropna()
    if len(valid_df) < 10:
        print("Insufficient data for ID estimator agreement plot")
        return None

    corr = valid_df.corr()
    labels = ["TwoNN", "PR"]
    fig, ax = plt.subplots(figsize=(3.5, 3))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=11)
    plt.colorbar(im, ax=ax)
    ax.set_title("ID Estimator Agreement\n(Diagnostic only — not evidentiary)", fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig
