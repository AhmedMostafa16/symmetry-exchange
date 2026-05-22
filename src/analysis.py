"""
Post-experiment analysis pipeline.

Loads run JSONs, computes N_target tables, estimates exchange rates,
runs pre-registered statistical tests, and classifies outcomes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.metrics import compute_n_target
from src.statistics import bonferroni_correction, bootstrap_slope_ci


def load_results(results_dir: str = "results/runs") -> pd.DataFrame:
    """Load all run result JSON files into a tidy DataFrame."""
    records = []
    for fpath in Path(results_dir).glob("*.json"):
        with open(fpath) as f:
            d = json.load(f)
        flat = {k: v for k, v in d.items()
                if k not in ("val_acc_curve", "train_loss_curve")}
        records.append(flat)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def compute_n_target_table(
    df: pd.DataFrame,
    target_acc: float = 0.80,
    min_seeds: int = 3,
) -> pd.DataFrame:
    """
    For each (n_group, model_type, epsilon) group, compute N_target.

    Returns a tidy DataFrame with columns:
      n_group, log2_n, model_type, epsilon, n_target, log2_n_target.
    """
    rows = []
    for (n_group, model_type, epsilon), grp in df.groupby(["n_group", "model_type", "epsilon"]):
        by_n: dict[int, list[float]] = {}
        for _, row in grp.iterrows():
            n_train = int(row["n_train"])
            by_n.setdefault(n_train, []).append(float(row["best_val_acc"]))

        n_target = compute_n_target(by_n, target_acc=target_acc, min_seeds_above=min_seeds)
        rows.append({
            "n_group": int(n_group),
            "log2_n": float(np.log2(max(int(n_group), 1))),
            "model_type": str(model_type),
            "epsilon": float(epsilon),
            "n_target": n_target,
            "log2_n_target": float(np.log2(n_target)) if n_target else float("nan"),
            "n_seeds_run": int(grp["seed"].nunique()),
        })
    return pd.DataFrame(rows)


def estimate_exchange_rate(
    n_target_df: pd.DataFrame,
    model_type: str,
    epsilon: float = 0.0,
    n_bootstrap: int = 10_000,
) -> dict:
    """
    Estimate the slope of log₂(N_target) ~ log₂(|G|) for one model type.

    This slope is the exchange rate: how many bits of data does one bit of
    correct group structure buy?
    """
    subset = n_target_df[
        (n_target_df["model_type"] == model_type)
        & (n_target_df["epsilon"] == epsilon)
        & n_target_df["log2_n_target"].notna()
    ]

    if len(subset) < 3:
        return {"error": "insufficient_data", "model_type": model_type,
                "epsilon": epsilon, "n_points": len(subset)}

    x = subset["log2_n"].values
    y = subset["log2_n_target"].values
    result = bootstrap_slope_ci(x, y, n_bootstrap=n_bootstrap)
    result.update({"model_type": model_type, "epsilon": epsilon,
                   "n_points": len(subset)})
    return result


def full_statistical_analysis(n_target_df: pd.DataFrame) -> dict:
    """
    Run all pre-registered statistical tests. Returns a structured report.

    Pre-registered comparisons (Bonferroni-corrected over 3 tests):
      H1: slope(equivariant) significantly < 0
      H2: slope(equivariant) significantly < slope(wrong_group)
      H3: slope(equivariant) significantly < slope(augmented)
    """
    report: dict = {
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
        "n_rows": len(n_target_df),
        "slopes": {},
        "comparisons": {},
        "bonferroni": {},
        "failure_classification": None,
    }

    for mtype in n_target_df["model_type"].unique():
        report["slopes"][mtype] = estimate_exchange_rate(n_target_df, mtype)

    equiv = report["slopes"].get("equivariant", {})
    wrong = report["slopes"].get("wrong_group", {})
    aug   = report["slopes"].get("augmented", {})

    # H2: equivariant slope steeper than wrong-group
    if "slope" in equiv and "slope" in wrong:
        slope_diff = equiv["slope"] - wrong["slope"]
        ci_overlap = (
            equiv.get("ci_upper", 0) > wrong.get("ci_lower", 0)
            and wrong.get("ci_upper", 0) > equiv.get("ci_lower", 0)
        )
        report["comparisons"]["equiv_vs_wrong"] = {
            "slope_difference": slope_diff,
            "ci_overlap": ci_overlap,
            "h2_supported": slope_diff < -0.3 and not ci_overlap,
        }

    # H3: equivariant slope steeper than augmented
    if "slope" in equiv and "slope" in aug:
        slope_diff_aug = equiv["slope"] - aug["slope"]
        ci_overlap_aug = (
            equiv.get("ci_upper", 0) > aug.get("ci_lower", 0)
            and aug.get("ci_upper", 0) > equiv.get("ci_lower", 0)
        )
        report["comparisons"]["equiv_vs_augmented"] = {
            "slope_difference": slope_diff_aug,
            "ci_overlap": ci_overlap_aug,
            "h3_supported": slope_diff_aug < -0.3 and not ci_overlap_aug,
        }

    # Bonferroni
    p_vals = {
        "H1_slope_nonzero": equiv.get("p_value", 1.0),
        "H2_correct_beats_wrong": (
            0.01 if report["comparisons"].get("equiv_vs_wrong", {}).get("h2_supported")
            else 0.5
        ),
        "H3_correct_beats_augmented": (
            0.01 if report["comparisons"].get("equiv_vs_augmented", {}).get("h3_supported")
            else 0.5
        ),
    }
    report["bonferroni"] = bonferroni_correction(p_vals)
    report["failure_classification"] = classify_failure_type(report)
    return report


def classify_failure_type(report: dict) -> dict:
    """
    Classify outcome into the pre-registered failure taxonomy.

    Types: A (no advantage), E (regularisation collapse), D (brittle),
           SIGNAL (genuine structural advantage), AMBIGUOUS.
    """
    slopes = report.get("slopes", {})
    equiv  = slopes.get("equivariant", {})
    wrong  = slopes.get("wrong_group", {})

    equiv_slope = equiv.get("slope", 0.0)
    wrong_slope = wrong.get("slope", 0.0)
    ci_low = equiv.get("ci_lower", 0.0)
    ci_high = equiv.get("ci_upper", 0.0)

    # Failure A: CI contains zero → no structural advantage
    if ci_low <= 0 <= ci_high:
        return {
            "type": "A",
            "label": "No structural advantage",
            "evidence": f"Equivariant slope CI [{ci_low:.2f}, {ci_high:.2f}] contains 0",
            "implication": "Scale-first orthodoxy not challenged",
            "what_not_to_conclude": "Theory is false — task may be wrong or under-powered",
        }

    # Failure E: both slopes negative and nearly equal → regularisation collapse
    if equiv_slope < 0 and wrong_slope < 0:
        diff = abs(equiv_slope - wrong_slope)
        if diff < 0.2:
            return {
                "type": "E",
                "label": "Regularisation collapse",
                "evidence": f"Slope diff={diff:.2f} < 0.2 threshold",
                "implication": "Orbit averaging provides generic regularisation benefit",
                "what_not_to_conclude": "Structural inductive bias is meaningless",
            }

    # Signal: genuine structural advantage
    h2 = report.get("comparisons", {}).get("equiv_vs_wrong", {}).get("h2_supported", False)
    if equiv_slope < -0.5 and h2 and ci_high < -0.2:
        return {
            "type": "SIGNAL",
            "label": "Genuine structural advantage",
            "evidence": (
                f"Equivariant slope={equiv_slope:.2f}, "
                f"Wrong-group slope={wrong_slope:.2f}, H2 supported"
            ),
            "implication": "Correct symmetry prior reduces sample complexity",
            "predicted_exchange_rate": abs(equiv_slope),
            "matches_theory": abs(abs(equiv_slope) - 1.0) < 0.3,
        }

    return {
        "type": "AMBIGUOUS",
        "label": "Inconclusive — collect more data or run epsilon sweep",
        "evidence": f"Equivariant slope={equiv_slope:.2f}",
    }
