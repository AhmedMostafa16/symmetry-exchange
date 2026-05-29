"""
Post-experiment analysis pipeline.

Loads run JSONs, computes N_target tables, estimates absolute and *relative*
exchange rates, runs the pre-specified statistical tests, and classifies outcomes.

**Why relative slopes are primary.** The absolute slope of
log₂(N_target) ~ log₂(|G|) for any single model is contaminated by
task-difficulty scaling: the petal task gets harder as n grows for every
model, so every slope is positive. The science-relevant quantity is the
*difference of slopes* — equivalently, the slope of log₂(N_baseline /
N_treatment) ~ log₂(|G|). That isolates the symmetry benefit from the shared
task-difficulty scaling and matches the theoretical prediction (≈ +1.0 for a
correctly-aligned C_n equivariant model).
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
        # Skip writer artefacts
        if fpath.name in {"heartbeat.json"}:
            continue
        try:
            with open(fpath) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if "run_id" not in d:
            continue
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


# ─── Absolute exchange rate (per model) ──────────────────────────────────────


def estimate_exchange_rate(
    n_target_df: pd.DataFrame,
    model_type: str,
    epsilon: float = 0.0,
    n_bootstrap: int = 10_000,
) -> dict:
    """
    Slope of log₂(N_target) ~ log₂(|G|) for one model type, with bootstrap CI.

    For interpretation, prefer `estimate_relative_exchange_rate` — absolute
    slopes are positive for every model due to task-difficulty scaling.
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


# ─── Relative exchange rate (treatment vs baseline) — PRIMARY METRIC ─────────


def estimate_relative_exchange_rate(
    n_target_df: pd.DataFrame,
    treatment: str,
    baseline: str = "vanilla",
    epsilon: float = 0.0,
    n_bootstrap: int = 10_000,
) -> dict:
    """
    Slope of log₂(N_baseline / N_treatment) ~ log₂(|G|), with bootstrap CI.

    This is the science-relevant exchange rate: how many bits of data does
    one bit of `treatment`'s inductive bias save *relative to `baseline`*?
    Theoretical prediction for a correctly-aligned C_n equivariant treatment
    against a vanilla MLP baseline: **≈ +1.0**.

    Rows where either model is missing N_target at a given n_group are
    dropped from the joint fit.
    """
    subset = n_target_df[
        (n_target_df["epsilon"] == epsilon)
        & (n_target_df["model_type"].isin([baseline, treatment]))
        & n_target_df["log2_n_target"].notna()
    ]
    pivoted = subset.pivot_table(
        index="n_group", columns="model_type", values="log2_n_target"
    )
    if baseline not in pivoted.columns or treatment not in pivoted.columns:
        return {
            "error": "missing_model",
            "treatment": treatment, "baseline": baseline,
            "epsilon": epsilon, "n_points": 0,
        }
    common = pivoted.dropna(subset=[baseline, treatment])

    if len(common) < 3:
        return {
            "error": "insufficient_data",
            "treatment": treatment, "baseline": baseline,
            "epsilon": epsilon, "n_points": int(len(common)),
        }

    x = np.log2(common.index.astype(float).values)
    y = common[baseline].values - common[treatment].values   # log₂(N_base / N_treat)
    result = bootstrap_slope_ci(x, y, n_bootstrap=n_bootstrap)
    result.update({
        "treatment": treatment, "baseline": baseline,
        "epsilon": epsilon, "n_points": int(len(common)),
    })
    return result


# ─── Full report ─────────────────────────────────────────────────────────────


def full_statistical_analysis(
    n_target_df: pd.DataFrame,
    baseline: str = "vanilla",
    epsilon: float = 0.0,
    n_bootstrap: int = 10_000,
) -> dict:
    """
    Run all pre-specified statistical tests. Returns a structured report.

    Primary inference uses `relative_slopes` (β_diff vs baseline) and
    `pairwise` (β_diff of one treatment vs another, jointly bootstrapped
    so the CI on the difference is tighter than naively combining the
    marginal CIs).

    `slopes` (per-model absolute slopes) is retained for reference and
    cross-checking but is not the basis of the classification.
    """
    report: dict = {
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
        "n_rows": len(n_target_df),
        "baseline": baseline,
        "epsilon": epsilon,
        "slopes": {},
        "relative_slopes": {},
        "pairwise": {},
        "comparisons": {},
        "bonferroni": {},
        "failure_classification": None,
    }

    if n_target_df.empty:
        report["failure_classification"] = {
            "type": "INSUFFICIENT_DATA",
            "label": "Empty N_target table",
        }
        return report

    mtypes = list(n_target_df["model_type"].unique())

    # Absolute slopes (per model)
    for mtype in mtypes:
        report["slopes"][mtype] = estimate_exchange_rate(
            n_target_df, mtype, epsilon=epsilon, n_bootstrap=n_bootstrap,
        )

    # Relative slopes (PRIMARY)
    for mtype in mtypes:
        if mtype == baseline:
            continue
        report["relative_slopes"][mtype] = estimate_relative_exchange_rate(
            n_target_df, treatment=mtype, baseline=baseline,
            epsilon=epsilon, n_bootstrap=n_bootstrap,
        )

    # Pairwise joint-difference CIs (for the H2 / H3 significance tests).
    # Each entry is slope of log₂(N_b / N_a) vs log₂(|G|) — equivalent to
    # the difference β_diff(a vs baseline) − β_diff(b vs baseline), but
    # bootstrapped jointly so the CI on the difference is tighter.
    # CI lower > 0 means treatment `a` significantly beats treatment `b`.
    pairwise_targets = [
        ("equivariant", "wrong_group", "equiv_vs_wrong"),
        ("equivariant", "augmented",   "equiv_vs_augmented"),
        ("equivariant", "vanilla",     "equiv_vs_vanilla"),
        ("equivariant", "regularized", "equiv_vs_regularized"),
    ]
    for a, b, label in pairwise_targets:
        if a in mtypes and b in mtypes:
            report["pairwise"][label] = estimate_relative_exchange_rate(
                n_target_df, treatment=a, baseline=b,
                epsilon=epsilon, n_bootstrap=n_bootstrap,
            )

    # Comparison summaries (retained for the paper's Bonferroni table)
    eq_rel = report["relative_slopes"].get("equivariant", {})
    wr_rel = report["relative_slopes"].get("wrong_group", {})
    au_rel = report["relative_slopes"].get("augmented", {})

    def _cmp(a: dict, b: dict) -> dict | None:
        if "slope" not in a or "slope" not in b:
            return None
        diff = a["slope"] - b["slope"]
        ci_overlap = (
            a.get("ci_upper", 0) > b.get("ci_lower", 0)
            and b.get("ci_upper", 0) > a.get("ci_lower", 0)
        )
        return {"slope_difference": diff, "ci_overlap": ci_overlap}

    cmp_eq_wr = _cmp(eq_rel, wr_rel)
    if cmp_eq_wr is not None:
        cmp_eq_wr["h2_supported"] = (
            cmp_eq_wr["slope_difference"] > 0.3 and not cmp_eq_wr["ci_overlap"]
        )
        report["comparisons"]["equiv_vs_wrong"] = cmp_eq_wr

    cmp_eq_au = _cmp(eq_rel, au_rel)
    if cmp_eq_au is not None:
        cmp_eq_au["h3_supported"] = (
            cmp_eq_au["slope_difference"] > 0.3 and not cmp_eq_au["ci_overlap"]
        )
        report["comparisons"]["equiv_vs_augmented"] = cmp_eq_au

    # Bonferroni: three pre-specified tests
    p_vals = {
        "H1_eq_beats_vanilla": eq_rel.get("p_value", 1.0),
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


# ─── Failure classifier (rewritten — uses relative slopes) ───────────────────


# Tunable thresholds. The exchange-rate sign convention here is:
#   β_diff > 0  →  treatment needs FEWER samples than baseline (good)
#   β_diff < 0  →  treatment needs MORE samples than baseline (active harm)
_SIGNIFICANT_RATE = 0.3       # |β_diff| < this → can't distinguish from baseline
_THEORY_MATCH_TOLERANCE = 0.3  # |β_diff − 1| < this → matches theoretical 1.0
_CLOSE_TO_OTHER = 0.3         # treatments within this AND overlapping CIs → "same"


def classify_failure_type(report: dict) -> dict:
    """
    Classify outcome using the *relative* exchange rates.

    Reads ``report['relative_slopes'][treatment]``. Each entry must have
    ``slope``, ``ci_lower``, ``ci_upper`` keys. The classifier returns the
    first matching label among:

    * ``INSUFFICIENT_DATA`` — no relative-slope data available
    * ``A`` — equivariant β_diff CI contains 0 (no significant advantage)
    * ``A_NEGATIVE`` — equivariant β_diff < 0 with CI excluding 0
    * ``E`` — equivariant β_diff ≈ wrong_group β_diff (regularisation collapse)
    * ``AUG`` — equivariant β_diff ≈ augmented β_diff (augmentation explains it)
    * ``SIGNAL`` — equivariant beats vanilla AND wrong_group, with margin
    * ``SIGNAL_NEEDS_WRONG_GROUP`` — equivariant beats vanilla but no
      wrong-group comparison available
    * ``AMBIGUOUS`` — none of the above conditions match cleanly
    """
    rel = report.get("relative_slopes", {})
    eq = rel.get("equivariant", {})

    if "slope" not in eq:
        return {
            "type": "INSUFFICIENT_DATA",
            "label": "No relative-slope data for equivariant model",
            "evidence": f"relative_slopes keys: {list(rel.keys())}",
        }

    eq_slope = eq["slope"]
    eq_lo, eq_hi = eq["ci_lower"], eq["ci_upper"]

    # ── Failure A: CI contains 0 → no significant advantage ──────────────────
    if eq_lo <= 0 <= eq_hi:
        return {
            "type": "A",
            "label": "No structural advantage",
            "evidence": (
                f"β_diff(equivariant) = {eq_slope:+.2f}, "
                f"CI [{eq_lo:+.2f}, {eq_hi:+.2f}] contains 0"
            ),
            "implication": "Equivariant model does not significantly outperform vanilla.",
            "what_not_to_conclude":
                "Theory is false — the test may be under-powered or the task wrong.",
        }

    # ── Failure A_NEGATIVE: equivariant is actively worse than vanilla ───────
    if eq_hi < 0:
        return {
            "type": "A_NEGATIVE",
            "label": "Equivariant is worse than vanilla",
            "evidence": (
                f"β_diff(equivariant) = {eq_slope:+.2f}, "
                f"CI [{eq_lo:+.2f}, {eq_hi:+.2f}] entirely below 0"
            ),
            "implication": "The 'correct' inductive bias hurts on this task.",
            "what_not_to_conclude":
                "The architecture is broken — check equivariance unit tests first.",
        }

    # Helper: prefer joint pairwise CI when available, fall back to marginal CIs.
    # The pairwise difference's bootstrap CI is tighter than the union of
    # marginal CIs because resampled differences cancel out shared variance.
    pairwise = report.get("pairwise", {})

    def _significantly_beats(a_name: str, pairwise_key: str,
                              marginal: dict) -> Optional[bool]:
        """True if equivariant significantly beats `a_name`, False if not, None if no data."""
        pw = pairwise.get(pairwise_key, {})
        if "slope" in pw and "ci_lower" in pw:
            # CI lower > 0 ⇒ equivariant beats a_name at the requested level
            return pw["ci_lower"] > 0
        if "slope" not in marginal:
            return None
        # Fallback to marginal CI comparison (looser, more conservative)
        cis_overlap = (eq_hi > marginal["ci_lower"]) and (marginal["ci_upper"] > eq_lo)
        return (eq_slope - marginal["slope"]) > _SIGNIFICANT_RATE and not cis_overlap

    def _indistinguishable(a_name: str, pairwise_key: str,
                            marginal: dict) -> Optional[bool]:
        """True if equivariant is NOT distinguishable from `a_name`."""
        pw = pairwise.get(pairwise_key, {})
        if "slope" in pw and "ci_lower" in pw:
            # CI of the difference contains 0 ⇒ indistinguishable
            return pw["ci_lower"] <= 0 <= pw["ci_upper"]
        if "slope" not in marginal:
            return None
        cis_overlap = (eq_hi > marginal["ci_lower"]) and (marginal["ci_upper"] > eq_lo)
        return abs(eq_slope - marginal["slope"]) < _CLOSE_TO_OTHER and cis_overlap

    # ── Failure E: wrong-group helps as much as correct ──────────────────────
    wr = rel.get("wrong_group", {})
    if _indistinguishable("wrong_group", "equiv_vs_wrong", wr):
        wr_slope = wr["slope"]
        return {
            "type": "E",
            "label": "Regularisation collapse — wrong structure helps as much as correct",
            "evidence": (
                f"β_diff(equivariant) = {eq_slope:+.2f} ≈ "
                f"β_diff(wrong_group) = {wr_slope:+.2f}; "
                f"pairwise CI on the difference contains 0"
                if "equiv_vs_wrong" in pairwise else
                f"β_diff(equivariant) = {eq_slope:+.2f} ≈ "
                f"β_diff(wrong_group) = {wr_slope:+.2f} (marginal CIs overlap)"
            ),
            "implication":
                "The advantage is from orbit averaging in general, not correct alignment.",
            "what_not_to_conclude":
                "Structural inductive bias is meaningless — design a stronger wrong-group control.",
        }

    # ── Failure AUG: augmentation alone matches architecture ─────────────────
    au = rel.get("augmented", {})
    if _indistinguishable("augmented", "equiv_vs_augmented", au):
        au_slope = au["slope"]
        return {
            "type": "AUG",
            "label": "Augmentation explains the advantage — no architectural benefit",
            "evidence": (
                f"β_diff(equivariant) = {eq_slope:+.2f} ≈ "
                f"β_diff(augmented) = {au_slope:+.2f}"
            ),
            "implication":
                "Data augmentation alone reproduces the equivariant model's gain.",
            "what_not_to_conclude":
                "Architectural equivariance is useless — adjusted-N comparison may still favor it.",
        }

    # ── SIGNAL: significantly > 0 AND significantly > wrong_group ────────────
    if "slope" in wr:
        h2_ok = _significantly_beats("wrong_group", "equiv_vs_wrong", wr)
        if h2_ok:
            wr_slope = wr["slope"]
            wr_lo, wr_hi = wr["ci_lower"], wr["ci_upper"]
            pw = pairwise.get("equiv_vs_wrong", {})
            test_label = (
                f"pairwise CI on β_diff(eq) − β_diff(wr) = "
                f"[{pw['ci_lower']:+.2f}, {pw['ci_upper']:+.2f}], excludes 0"
                if "slope" in pw else
                "marginal CIs non-overlapping and gap > 0.3"
            )
            return {
                "type": "SIGNAL",
                "label": "Genuine structural advantage",
                "evidence": (
                    f"β_diff(equivariant) = {eq_slope:+.2f} CI [{eq_lo:+.2f}, {eq_hi:+.2f}]; "
                    f"β_diff(wrong_group) = {wr_slope:+.2f} CI [{wr_lo:+.2f}, {wr_hi:+.2f}]; "
                    f"{test_label}"
                ),
                "implication":
                    "Correctly-aligned symmetry prior produces a real reduction in sample complexity.",
                "exchange_rate": eq_slope,
                "matches_theory": abs(eq_slope - 1.0) < _THEORY_MATCH_TOLERANCE,
                "h2_supported": True,
            }
        # Equivariant beats vanilla but wrong-group test inconclusive
        wr_slope = wr["slope"]
        wr_lo, wr_hi = wr["ci_lower"], wr["ci_upper"]
        cis_overlap = (eq_hi > wr_lo) and (wr_hi > eq_lo)
        return {
            "type": "AMBIGUOUS",
            "label": "Equivariant beats vanilla but wrong-group test inconclusive",
            "evidence": (
                f"β_diff(equivariant) = {eq_slope:+.2f} CI [{eq_lo:+.2f}, {eq_hi:+.2f}]; "
                f"β_diff(wrong_group) = {wr_slope:+.2f} CI [{wr_lo:+.2f}, {wr_hi:+.2f}]; "
                f"slope difference = {eq_slope - wr_slope:+.2f}, CIs overlap = {cis_overlap}"
            ),
            "implication":
                "Need more data or a tighter wrong-group control to distinguish from Failure E.",
        }

    # No wrong-group data available
    return {
        "type": "SIGNAL_NEEDS_WRONG_GROUP",
        "label": "Equivariant beats vanilla; wrong-group control missing",
        "evidence": (
            f"β_diff(equivariant) = {eq_slope:+.2f} CI [{eq_lo:+.2f}, {eq_hi:+.2f}]; "
            "no wrong_group entry in relative_slopes"
        ),
        "implication":
            "Promising but cannot rule out Failure E without a wrong-group baseline.",
        "exchange_rate": eq_slope,
        "matches_theory": abs(eq_slope - 1.0) < _THEORY_MATCH_TOLERANCE,
    }
