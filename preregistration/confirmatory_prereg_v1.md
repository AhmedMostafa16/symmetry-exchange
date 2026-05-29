# Pre-Registration: Confirmatory Symmetry Exchange Rate Experiment

## Version 1.0

---

### Deposit fields (fill before running any training)

| Field                              | Value                                                                                                                        |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Title**                          | Confirmatory measurement of the architectural symmetry / training-data exchange rate via the relative-slope estimator β_diff |
| **Depositor**                      | [your identifier]                                                                                                            |
| **Deposit date**                   | [fill on deposit — YYYY-MM-DD]                                                                                               |
| **Deposit platform**               | [OSF / arXiv / public git tag]                                                                                               |
| **Deposit URL / tag**              | [fill after deposit]                                                                                                         |
| **Config hash (confirmatory)**     | `5bcdf06f89f70de4d56b5d4be0514211f9ecb42ab2dc5cc9bff171cfe43acc88`                                                           |
| **Motivating study (exploratory)** | Git commit range `4104785`–`0a990ae`, config hash `1bd8889315ec0f15640eb35854279e87dbcec1f57f7ea0cbb4a9fdd8a9f4a00e`         |

This document must be deposited in an externally timestamped registry **before any training run** for the new seeds. The exploratory study (seeds 0–4) is cited above for motivation only; none of its data are reused.

---

## 1. Plain-language summary

We measure whether a C_n-equivariant neural network requires approximately |G| = n times fewer training samples than a parameter-matched unconstrained network to reach the same validation accuracy on a C_n-symmetric classification task. The theoretical prediction is a factor-of-n reduction in sample complexity, corresponding to a relative exchange rate β_diff = +1.0 (one bit of data per bit of correctly-aligned group structure).

A prior exploratory study (seeds 0–4; ~5 600 training runs) found β_diff = +1.28 [95% CI: +0.92, +2.05] using a relative-slope estimator that was adopted **post-hoc** after Phase 1 was analysed. The present study pre-specifies the relative-slope estimator β_diff as the **primary outcome before any data are collected**, making the result genuinely confirmatory.

---

## 2. Motivating exploratory evidence

The exploratory study is the direct predecessor of this study. Its key findings — which motivate but do NOT count as evidence for this pre-registered study — are:

- β_diff(equivariant vs vanilla) = +1.28 [+0.92, +2.05] at ε = 0; the CI contains the theoretical prediction of +1.0.
- The rate is robust to training-label noise: 88–97% of the ε=0 value at ε ∈ {0.1, 0.2, 0.3}.
- The augmented baseline (orbit augmentation without equivariant pooling) fails to converge at n ≥ 3 within the sample budget, suggesting architecture and augmentation are not equivalent.
- The wrong-group control (misaligned orbit pooling) is **worse** than vanilla, indicating that orbit-averaging is harmful when misaligned with the task symmetry.
- The exploratory primary estimator was the absolute OLS slope β₁ of log₂(N_target) ~ log₂(|G|), predicted to be ≈ −1.0. All models showed positive absolute slopes because task difficulty grows with n. The relative-slope estimator β_diff — which cancels this shared scaling — was adopted post-hoc and gives the cleaner pre-specified test.

**The confirmatory study's single purpose is to replicate these findings with β_diff pre-specified, on new independent seeds, in a registry-timestamped document deposited before data collection.**

---

## 3. Experimental design (identical to exploratory study except seeds)

### 3.1 Task

C_n petal binary classification on 2D annular inputs (r ∈ [0.1, 1.0]):

```
y = 1[cos(n·θ) > 0]  with label noise ε (probability of flip)
```

The task's true symmetry is D_n; the equivariant model exploits the rotational subgroup C_n only (conservative).

### 3.2 Model families

| Family        | Description                                                                          |
| ------------- | ------------------------------------------------------------------------------------ |
| `equivariant` | C_n MLP via regular-representation orbit pooling (correct group)                     |
| `wrong_group` | Same orbit-pooling architecture with period 0.7× (misaligned group)                  |
| `augmented`   | Vanilla MLP with orbit augmentation at training time (identical data to equivariant) |
| `vanilla`     | Parameter-matched unconstrained MLP (the primary baseline)                           |
| `regularized` | Vanilla MLP with L2 weight decay λ = 1×10⁻³                                          |

### 3.3 Sweep grid

| Hyperparameter  | Values                                                                     |
| --------------- | -------------------------------------------------------------------------- |
| n (= \|G\|)     | {1, 2, 3, 4, 6, 8, 12}                                                     |
| N_train         | {50, 100, 200, 400, 800, 1 600, 3 200, 6 400}                              |
| ε (label noise) | Phase 1: {0.0}; Phase 2: {0.1, 0.2, 0.3}                                   |
| Seeds           | **{100, 101, 102, 103, 104}** (new; disjoint from exploratory seeds {0–4}) |

### 3.4 Architecture and training

Identical to the exploratory study:

- hidden_dim = 32, n_hidden = 2 (2-hidden-layer MLP)
- lr = 1×10⁻³ (Adam), batch_size = 64
- max_epochs = 500, patience = 30
- val fraction = 20% of N_train (stratified split)

### 3.5 Primary outcome: N_target

N_target(model, n, ε) = min {N ∈ N_train_grid : ≥ 3 of 5 seeds reach val_acc ≥ 0.80 by end of training}.

If no N in the grid satisfies this, N_target = ∞ (model failed to reach target within sample budget).

---

## 4. Hypotheses

All hypotheses are stated in terms of the relative exchange rate β_diff defined in §5. They are listed in priority order; H1 is the primary. Bonferroni correction is applied across H1, H2, H3 (α_corrected = 0.05/3 ≈ 0.0167 per test). H4 is a robustness check with a pre-specified threshold, reported descriptively.

### H1 (Primary — relative exchange rate)

The relative exchange rate between the equivariant and vanilla models is positive, consistent with the theoretical prediction of +1.0, and the 95% CI excludes 0.

**Operationalisation:**

- β_diff(equivariant vs vanilla) is estimated by OLS on the log₂-log₂ regression (§5.1).
- 95% CI is computed by joint pairwise bootstrap (§5.2).
- **Pre-specified minimum effect**: β_diff > 0 with CI excluding 0 (one-sided; direction predicted).
- **Pre-specified theory-consistency criterion**: 95% CI contains +1.0 AND |β_diff − 1.0| < 0.5.
- **Success**: both the significance criterion and the theory-consistency criterion are met.

### H2 (Wrong-group control — alignment specificity)

The equivariant model has a strictly higher relative exchange rate than the wrong-group model.

**Operationalisation:**

- Δβ_diff = β_diff(equivariant vs vanilla) − β_diff(wrong_group vs vanilla) > 0 at ε = 0.
- 95% CI on Δβ_diff (joint pairwise bootstrap, §5.2) excludes 0.
- **Success**: CI on Δβ_diff excludes 0 on the positive side.

### H3 (Augmented baseline — architecture vs augmentation)

The orbit-augmented vanilla model fails to reach N_target ≤ 6 400 at n ≥ 4, while the equivariant model succeeds at n ≥ 4.

**Operationalisation:**

- At ε = 0, for every n ∈ {4, 6, 8, 12}: N_target(augmented, n, 0) = ∞ (undefined in grid).
- Simultaneously: N_target(equivariant, n, 0) < ∞ for at least 3 values of n ∈ {4, 6, 8, 12}.
- **Success**: both sub-conditions hold.

### H4 (Graceful degradation — robustness to label noise)

The relative exchange rate at ε = 0.2 exceeds 50% of the rate at ε = 0.

**Operationalisation:**

- β_diff(ε=0.2) / β_diff(ε=0) > 0.50.
- Reported descriptively; no Bonferroni adjustment (separate outcome, different phase).

---

## 5. Analysis plan

### 5.1 Primary estimator: β_diff

Let log₂ N_v(n) = log₂ N_target(vanilla, n, ε) and log₂ N_e(n) = log₂ N_target(equivariant, n, ε). Define the pairwise log-ratio:

```
r(n) = log₂ N_v(n) − log₂ N_e(n)   [log₂(N_vanilla / N_equivariant)]
```

β_diff is the OLS slope of r(n) ~ log₂(n) across n ∈ {1, 2, 3, 4, 6, 8, 12} where both models have finite N_target.

- If N_target = ∞ for either model at a given n, that n is excluded from the regression.
- If fewer than 4 finite pairs remain, β_diff is reported as undefined.

### 5.2 Confidence intervals: joint pairwise bootstrap

CI is computed by the joint pairwise bootstrap over n_group resamples:

1. Resample (with replacement) from the set of n values {n : both models have finite N_target}.
2. On each resample, recompute β_diff via OLS on the resampled pairs.
3. Repeat 10 000 times. Discard iterations where the OLS fit has fewer than 3 finite pairs.
4. 95% CI = [2.5th, 97.5th] percentile of the bootstrap distribution of β_diff.

This is the same estimator and CI procedure as the exploratory study (implemented in `src/analysis.py::estimate_relative_exchange_rate`). The code is frozen at the same commit as the exploratory study.

### 5.3 Pairwise difference CI for H2

Δβ_diff = β_diff(equivariant vs vanilla) − β_diff(wrong_group vs vanilla) is computed jointly in each bootstrap iteration (using the same n-group resample for both models), giving a paired CI on the difference. This is tighter than comparing two marginal CIs.

### 5.4 Decision procedure

The following decision rules are applied in order after Phase 1 (ε = 0) is complete. Results are inspected **only after all Phase 1 training runs are finished** (no early stopping, no peeking per-n).

| Outcome                                        | Classification               | Action                                                      |
| ---------------------------------------------- | ---------------------------- | ----------------------------------------------------------- |
| H1: CI excludes 0, contains +1.0               | **SIGNAL** (replicated)      | Report and proceed to Phase 2                               |
| H1: CI excludes 0, does NOT contain +1.0       | **SIGNAL_SHIFTED**           | Report and proceed to Phase 2; note discrepancy with theory |
| H1: CI includes 0, β_diff point estimate > 0.3 | **AMBIGUOUS**                | Report; do not make strong confirmatory claim               |
| H1: CI includes 0, β_diff ≤ 0.3                | **FAILURE_A** (no advantage) | Report; confirmatory hypothesis rejected                    |
| H2 (or H3) failed                              | **FAILURE_E/AUG**            | Report; partial replication                                 |

Phase 2 is run regardless of Phase 1 outcome (no adaptive stopping). The graceful-degradation check H4 is computed from Phase 2 results.

### 5.5 Failure taxonomy: complete specification

| Code           | Condition                                     | Interpretation                                                            |
| -------------- | --------------------------------------------- | ------------------------------------------------------------------------- |
| SIGNAL         | H1 CI > 0, contains 1.0; H2 CI > 0; H3 met    | Full replication of exploratory result                                    |
| SIGNAL_SHIFTED | H1 CI > 0 but excludes 1.0                    | Equivariant advantage confirmed, rate differs from prediction             |
| AMBIGUOUS      | H1 CI includes 0                              | Inconclusive; insufficient power or effect absent                         |
| FAILURE_A      | H1 CI includes 0, β_diff ≤ 0.3                | No equivariant advantage; exploratory result not replicated               |
| FAILURE_E      | H2 not met (wrong-group β_diff ≥ equivariant) | Alignment specificity absent; regularization or capacity explanation live |
| FAILURE_AUG    | H3 not met (augmented converges at n ≥ 4)     | Architecture and augmentation are equivalent at this scale                |

---

## 6. Exclusion criteria

The following runs are **excluded from the primary analysis but retained and reported in the supplement**:

1. Training loss is NaN at any epoch during training.
2. Validation accuracy remains ≤ 0.53 across all N_train values for all 5 seeds (chance performance at any n; flagged as data or implementation error).
3. Config hash of the run does not match the config hash recorded in this document.
4. GPU out-of-memory error causing the run to abort before completing all N_train values.

Exclusions are determined by the criterion only; no inspection of the slope value is permitted before applying exclusion criteria.

---

## 7. Stopping rules

- **No early stopping of the sweep.** All (n, N_train, seed, model_type) combinations in the grid are run to completion before any analysis.
- **Phase 2 runs unconditionally** after Phase 1 is complete (the Phase 1 result does not influence whether Phase 2 is run).
- The experiment terminates after Phase 2 results are analysed and reported, regardless of outcome.
- GPU OOM or hardware failure may cause early termination; in that case the partial results are reported as-is, with the completeness fraction noted.

---

## 8. What is NOT pre-specified (to prevent post-hoc inflation)

The following quantities are collected and reported as exploratory/descriptive only. They are NOT used to support or undermine the pre-specified hypotheses:

- **Absolute slopes** (β₁ of log₂ N_target ~ log₂ n per model): reported as supplementary table, not as test statistics.
- **Intrinsic dimensionality estimates** (TwoNN, MLE, PR dimension of penultimate-layer activations): reported as supplementary observation; flagged as unreliable in the exploratory study.
- **Decision boundary qualitative figures**: illustrative only.
- **Any analysis of the relationship between β_diff and ε beyond the H4 threshold test** (e.g., fitting a degradation curve).
- **Any sub-group analysis by seed** (e.g., identifying outlier seeds and excluding them post-hoc).

Any such analysis conducted after data collection is exploratory; it may generate hypotheses for a future study but does NOT provide confirmatory evidence for H1–H4.

---

## 9. Scope and limitations acknowledged before data collection

1. **2D synthetic task.** The C_n petal task has exact C_n symmetry by construction. Results do not directly establish the exchange rate for approximate symmetries (vision, language, molecular data) or for tasks in more than 2 dimensions.

2. **C_n subgroup only.** The task's true symmetry group is D_n (rotation + reflection). The equivariant model exploits only C_n. The measured β_diff is a conservative lower bound on what a D_n-equivariant model would achieve.

3. **FLOP-neutral by design.** The equivariant model performs n forward passes per input. The sample-efficiency gain factor of n exactly cancels the per-sample FLOP cost. This experiment measures sample efficiency, not compute efficiency.

4. **ε breaks training labels, not architecture.** The graceful-degradation sweep corrupts training labels; it does not test architectural mismatch (applying a C_k model to a C_n task, k ≠ n).

5. **5 seeds.** Power is modest. If the exploratory estimate β_diff ≈ 1.28 is accurate, 5 seeds and 7 group orders provide adequate power to reject β_diff = 0; if the true rate is closer to 0.5, the study may be underpowered.

---

## 10. Configuration specification

The confirmatory experiment uses the same software codebase and configuration as the exploratory study, with one change: seeds = [100, 101, 102, 103, 104].

### Confirmatory config (compute hash from this before depositing)

```python
CONFIRMATORY_CONFIG = {
    "task": "cn_petal_classification",
    "n_groups": [1, 2, 3, 4, 6, 8, 12],
    "n_train_grid": [50, 100, 200, 400, 800, 1600, 3200, 6400],
    "epsilon_values": [0.0, 0.1, 0.2, 0.3],
    "n_seeds": 5,
    "seeds": [100, 101, 102, 103, 104],          # NEW — disjoint from exploratory [0..4]
    "model_types": ["equivariant", "wrong_group", "augmented", "vanilla", "regularized"],
    "hidden_dim": 32,
    "n_hidden": 2,
    "lambda_l2_regularized": 1e-3,
    "target_acc": 0.80,
    "max_epochs": 500,
    "patience": 30,
    "batch_size": 64,
    "lr": 1e-3,
    "primary_metric": "n_target_median3of5",
    "primary_estimator": "beta_diff_equivariant_vs_vanilla",  # NOW PRE-SPECIFIED
    "primary_comparison": "relative_slope_equivariant_vs_vanilla",
    "secondary_comparison": "relative_slope_equivariant_vs_wrong_group",
    "alpha": 0.05,
    "bonferroni_n": 3,
    "n_bootstrap": 10_000,
}
```

### How to compute and verify the config hash

```bash
uv run python -c "
from src.statistics import compute_analysis_hash

CONFIRMATORY_CONFIG = {
    'task': 'cn_petal_classification',
    'n_groups': [1, 2, 3, 4, 6, 8, 12],
    'n_train_grid': [50, 100, 200, 400, 800, 1600, 3200, 6400],
    'epsilon_values': [0.0, 0.1, 0.2, 0.3],
    'n_seeds': 5,
    'seeds': [100, 101, 102, 103, 104],
    'model_types': ['equivariant', 'wrong_group', 'augmented', 'vanilla', 'regularized'],
    'hidden_dim': 32,
    'n_hidden': 2,
    'lambda_l2_regularized': 1e-3,
    'target_acc': 0.80,
    'max_epochs': 500,
    'patience': 30,
    'batch_size': 64,
    'lr': 1e-3,
    'primary_metric': 'n_target_median3of5',
    'primary_estimator': 'beta_diff_equivariant_vs_vanilla',
    'primary_comparison': 'relative_slope_equivariant_vs_vanilla',
    'secondary_comparison': 'relative_slope_equivariant_vs_wrong_group',
    'alpha': 0.05,
    'bonferroni_n': 3,
    'n_bootstrap': 10_000,
}
print(compute_analysis_hash(CONFIRMATORY_CONFIG))
"
```

Paste the output into the **Config hash (confirmatory)** field in the deposit header above and into the deposit platform before running any training.

### How to verify the timestamp is genuinely pre-data

After running the experiment, verifiable evidence that this document predates training:

1. **OSF deposit**: the OSF registration URL contains a creation timestamp.
2. **Git tag**: `git log <tag> --format="%H %ai"` shows the tag creation time; the earliest training result file has a filesystem timestamp from `results/runs/`.
3. **arXiv pre-print**: submission timestamp visible in the arXiv record.

The config hash in the deposit must match the hash computed from the config above. A mismatch indicates the config was changed after deposit.

---

## 11. Relationship to the exploratory study

| Property            | Exploratory study                                     | This confirmatory study         |
| ------------------- | ----------------------------------------------------- | ------------------------------- |
| Seeds               | {0, 1, 2, 3, 4}                                       | {100, 101, 102, 103, 104}       |
| Primary estimator   | Absolute slope β₁ (pre-specified) → β_diff (post-hoc) | **β_diff (pre-specified)**      |
| Deposit before data | No                                                    | **Yes (this document)**         |
| Role                | Hypothesis generation; motivates confirmatory study   | Tests pre-specified predictions |
| β_diff result       | +1.28 [+0.92, +2.05]                                  | Unknown until data collected    |

The exploratory study's result is the basis for H1's directional prediction. It is **not evidence for H1**; it is the prior that makes H1 scientifically motivated. The confirmatory study is the test.

---

## 12. How to run the confirmatory experiment

The codebase at `experiment_runner.py` hardcodes seeds [0,1,2,3,4] in `PREREGISTERED_CONFIG`. Before running, create a thin override script or patch the runner to use the confirmatory config:

```python
# confirmatory_runner.py — run AFTER depositing this document
from experiment_runner import *  # re-use all helpers

CONFIRMATORY_CONFIG = { ... }   # paste from §10
# Override the module-level config for this run
import experiment_runner
experiment_runner.PREREGISTERED_CONFIG = CONFIRMATORY_CONFIG
experiment_runner.PREREGISTERED_HASH = compute_analysis_hash(CONFIRMATORY_CONFIG)

if __name__ == "__main__":
    # Call the same phase runners with the same CLI arguments
    import sys
    sys.argv = sys.argv  # pass through; use same --phase/--device/--trainer flags
    exec(open("experiment_runner.py").read())  # or refactor to call main() directly
```

Or add `--seeds 100 101 102 103 104` as a CLI argument to `experiment_runner.py` before running. Verify the printed config hash matches `5bcdf06f89f70de4d56b5d4be0514211f9ecb42ab2dc5cc9bff171cfe43acc88` at startup before any training begins.

The analysis (`src/analysis.py::full_statistical_analysis`) works unchanged because it reads results from the `results/` directory and does not hard-code seeds.

---

## 13. Signature block

```
Researcher: Ahmed M. Adly
Date deposited: 2026-05-29
Platform: public git tag (https://github.com/AhmedMostafa16/symmetry-exchange)
Deposit URL / git tag:  https://github.com/AhmedMostafa16/symmetry-exchange/tree/confirmatory-prereg-v1
Config hash (confirmatory):
Exploratory study hash (reference):
Software commit at deposit: _______________  (git rev-parse HEAD)
```
