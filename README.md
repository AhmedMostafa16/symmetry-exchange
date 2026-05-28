# Symmetry Exchange Rate Experiment

Measures whether a C_n equivariant model trades **one bit of correct group structure for
one bit of data** when learning a C_n petal classification task.

Pre-registered: see [`preregistration/preregistration.md`](preregistration/preregistration.md)
and the config hash in [`preregistration/config_hash.txt`](preregistration/config_hash.txt).

## Setup

```bash
uv sync --all-extras            # install runtime + dev deps
uv run pytest                   # 116 tests, 86% coverage
```

## Run the experiment

The runner has three trainer backends:

| `--trainer` | Description | Phase 1 wall-clock (Kaggle T4) |
|---|---|---|
| `original` | Reference implementation (DataLoader, per-batch transfers, full ID metrics) | ~5–6 h |
| `fast`     | Pre-loaded GPU data, manual batching, bf16 autocast, val every 5 epochs | ~1 h |
| `ensemble` | Same as `fast` plus vmap-batches all 5 seeds in one parallel pass (default) | ~15–30 min |

```bash
# Pilot smoke test (2 groups × 2 N_train × 2 seeds, ~2 min on GPU)
uv run python experiment_runner.py --pilot --device cuda

# Phase 1 full sweep (ε=0)
uv run python experiment_runner.py --phase 1 --device cuda

# Phase 2 ε sweep (after Phase 1 review)
uv run python experiment_runner.py --phase 2 --device cuda
```

CLI flags:

- `--trainer {original,fast,ensemble}` — default `ensemble`
- `--patience N` — early-stop patience for fast/ensemble (default 10)
- `--val-every K` — validate every K epochs (default 5)
- `--no-amp` — disable bf16 autocast (forces fp32)
- `--results-dir PATH` — where to write run JSONs and done markers

Resume safety: each completed `(n_group, N_train, ε, model_type)` cell writes a
`done_*.flag` marker. Re-running the command skips finished cells.

## Benchmark

```bash
uv run python scripts/benchmark.py --device cuda
```

Reports per-trainer wall time on a small grid (100 runs) and extrapolates to full Phase 1.

## Project structure

```
src/
├── models.py            5 model families: equivariant, wrong_group, augmented, vanilla, regularized
├── data_generator.py    C_n petal binary classification + adversarial checks
├── trainer.py           Reference single-model trainer (slow but verified)
├── fast_trainer.py      Optimised single-model trainer (~5× faster)
├── ensemble_trainer.py  Vmap-batched ensemble trainer (~15-25× faster)
├── metrics.py           N_target, TwoNN/PR/MLE intrinsic dim, orbit consistency
├── statistics.py        Bootstrap slope CI, power analysis, Bonferroni
├── analysis.py          N_target table, exchange rate estimation, failure taxonomy
├── visualizations.py    4 publication figures
└── resource_tracker.py  RunResult schema, JSON persistence

tests/                   116 tests, 86% coverage
preregistration/         Pre-registration doc + config hash
results/runs/            One JSON per training run (never delete)
scripts/benchmark.py     Trainer comparison
experiment_runner.py     Main sweep entry point
```

## Determinism

- All trainers seed `torch`, `torch.cuda`, and `numpy` RNGs.
- `fast_trainer` and `ensemble_trainer` set `cudnn.benchmark=True` (irrelevant for MLPs).
- bfloat16 autocast is deterministic within a single GPU type but may differ across GPU
  generations. Disable with `--no-amp` for strict cross-hardware determinism.
- Model initialisation is seeded *before* `get_model_suite()` in the runner; this was a
  fix relative to the first Phase 1 run and is more faithful to the pre-registration.
