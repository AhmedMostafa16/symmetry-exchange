# Pre-Registration: Symmetry Exchange Rate Experiment

**Pre-registration ID:** [git commit hash — fill before first training run]
**Date:** [fill before first training run]
**Researcher:** [your identifier]

---

## 1. Hypotheses (State before seeing any training results)

**H1 (Primary):** For a C_n petal classification task, a C_n-equivariant model
achieves target accuracy T=0.80 with N_target samples, where
log₂(N_target) = β₀ + β₁ · log₂(n) and β₁ is significantly negative.

**Predicted value:** β₁ ≈ −1.0
**Null:** β₁ = 0
**Minimum effect size to claim support:** β₁ < −0.4 with 95% CI excluding 0.

**H2 (Correctness):** β₁(equivariant, correct) < β₁(wrong_group) by at least 0.3,
with non-overlapping 95% bootstrap CIs.

**H3 (FLOP exchange rate):** The FLOP-normalised exchange rate will be near 0
(structure converts data to compute at roughly 1:1 ratio). Structure buys data
cheapness, not compute cheapness.

---

## 2. Analysis Plan (State before seeing results)

**Primary estimator:** OLS slope of log₂(N_target) vs log₂(|G|) for equivariant model
**CI method:** 10,000-sample non-parametric bootstrap, 95% CIs
**Significance:** α = 0.05, Bonferroni-corrected for 3 pre-specified tests
**N_target definition:** min N such that 3/5 seeds reach ≥ 0.80 val accuracy

---

## 3. Pre-registered Comparisons (Bonferroni-corrected, α/3 = 0.0167)

| Test | H0 | Direction |
|------|-----|-----------|
| H1   | β₁(equivariant) = 0 | CI excludes 0, β₁ < 0 |
| H2   | β₁(equivariant) = β₁(wrong_group) | Equivariant slope steeper by >0.3, non-overlapping CI |
| H3   | β₁(equivariant) = β₁(augmented) | Equivariant slope steeper (informative, not primary) |

---

## 4. Exclusion Criteria (Specify before running)

Runs are **excluded and flagged** (never deleted) if:

- Training loss is NaN at any epoch
- Val accuracy remains below 0.53 at all N_train values for all seeds
- GPU OOM error during training
- Config hash mismatch with pre-registered config

**Excluded runs are retained and reported in the supplement.**

---

## 5. Stopping Rules

- **Phase 1** (epsilon=0): Run all seeds before inspecting results
- **Stop after Phase 1** if 95% CI on equivariant slope contains 0 AND point
  estimate > −0.3 (no practically significant trend) → classify as Failure A
- **Proceed to Phase 2** (epsilon sweep) otherwise

---

## 6. Pre-registered Graceful Degradation Prediction

We expect exchange rate at ε=0.2 to be > 50% of the exchange rate at ε=0.0.
If degradation is sharper → classify as Failure D (brittle advantage).

---

## 7. Failure Taxonomy (State before seeing results)

| Outcome | Classification | Action |
|---------|---------------|--------|
| CI on β₁ contains 0 | **A** — No advantage | Report and stop |
| β₁ < 0 but β₁(eq) − β₁(wrong) < 0.2 | **E** — Regularisation collapse | Report, redesign wrong-group control |
| β₁ < 0, H2 supported, but cliff at ε > 0 | **D** — Brittle | Report |
| β₁ ∈ [−1.3, −0.7], H2 supported, smooth degradation | **SIGNAL** | Proceed to replication |

---

## 8. What This Experiment Cannot Conclude

- Even if slope ≈ −1, this does not prove "intelligence emerges from invariant compression"
- This experiment tests one specific family of tasks and groups;
  generalisation to other domains is **NOT** established
- A confirmed exchange rate in samples does **NOT** imply an exchange rate in FLOPs
- ID measurements are observational only; they do not establish causal relevance
- The petal task has **exact** C_n symmetry; real-world tasks rarely do

---

## 9. Four Tests This Experiment Must Pass to Be Meaningful

1. **Augmented control:** slope(equivariant) significantly steeper than slope(augmented-adjusted).
   If not → orbit augmentation explains the effect, not architectural bias.

2. **Wrong-group control:** slope(equivariant) significantly steeper than slope(wrong_group).
   If not → any orbit-size-n constraint helps equally.

3. **Graceful degradation:** exchange rate at ε=0.2 > 50% of ε=0.0 value.
   If not → result is a Platonic toy effect (Failure D).

4. **Large-vanilla control:** slope(equivariant) significantly steeper than slope(vanilla)
   with matched capacity (vanilla_large). If not → capacity amplification drives the effect.

---

## Config Hash: [RUN `python -c "from experiment_runner import PREREGISTERED_HASH; print(PREREGISTERED_HASH)"` AND FILL IN]
