"""
Benchmark trainer variants on a representative slice of Phase 1.

Usage:
    uv run python scripts/benchmark.py --device cpu          # quick check on CPU
    uv run python scripts/benchmark.py --device cuda         # real measurement
    uv run python scripts/benchmark.py --device cuda --trainers original fast ensemble

Reports wall-clock time per trainer, per-cell average, speedup ratios,
and extrapolated Phase 1 cost.
"""
from __future__ import annotations

import argparse
import time
from typing import Callable

import numpy as np
import torch

from src.data_generator import generate_petal_dataset
from src.models import get_model_suite


# Grid: 2 group sizes × 2 N_train × 5 seeds × 5 model types = 100 runs (20 cells).
# Picks small-N (fast convergence) and larger-N (full max_epochs) regimes.
BENCH_GROUPS  = [2, 8]
BENCH_N_TRAIN = [100, 1600]
BENCH_SEEDS   = [0, 1, 2, 3, 4]
BENCH_MTYPES  = ["equivariant", "wrong_group", "augmented", "vanilla", "regularized"]

FULL_PHASE_1_CELLS = 7 * 8 * 5 * 5   # 1400


def _build_datasets(n_group: int, n_train: int, seeds: list[int]):
    trains, vals, hashes = [], [], []
    for s in seeds:
        tr, va, _, meta = generate_petal_dataset(
            n_petals=n_group, N_train=n_train, epsilon=0.0, seed=s,
        )
        trains.append(tr); vals.append(va); hashes.append(meta["dataset_hash"])
    return trains, vals, hashes


def bench_original(device: str, max_epochs: int) -> float:
    from src.trainer import train_one_run
    start = time.perf_counter()
    for n_group in BENCH_GROUPS:
        for n_train in BENCH_N_TRAIN:
            trains, vals, _ = _build_datasets(n_group, n_train, BENCH_SEEDS)
            for s_idx, seed in enumerate(BENCH_SEEDS):
                torch.manual_seed(seed); np.random.seed(seed)
                models = get_model_suite(n_group, 32, 2)
                for mtype, model in models.items():
                    lam = 1e-3 if mtype == "regularized" else 0.0
                    train_one_run(
                        model=model, model_type=mtype,
                        train_dataset=trains[s_idx], val_dataset=vals[s_idx],
                        n_group=n_group, n_train=n_train, epsilon=0.0, seed=seed,
                        lambda_l2=lam, max_epochs=max_epochs, patience=30,
                        batch_size=64, lr=1e-3, device=device,
                    )
    return time.perf_counter() - start


def bench_fast(device: str, max_epochs: int, use_amp: bool) -> float:
    from src.fast_trainer import train_one_run_fast
    start = time.perf_counter()
    for n_group in BENCH_GROUPS:
        for n_train in BENCH_N_TRAIN:
            trains, vals, _ = _build_datasets(n_group, n_train, BENCH_SEEDS)
            for s_idx, seed in enumerate(BENCH_SEEDS):
                torch.manual_seed(seed); np.random.seed(seed)
                models = get_model_suite(n_group, 32, 2)
                for mtype, model in models.items():
                    lam = 1e-3 if mtype == "regularized" else 0.0
                    train_one_run_fast(
                        model=model, model_type=mtype,
                        train_dataset=trains[s_idx], val_dataset=vals[s_idx],
                        n_group=n_group, n_train=n_train, epsilon=0.0, seed=seed,
                        lambda_l2=lam, max_epochs=max_epochs, patience=10,
                        batch_size=64, lr=1e-3, device=device,
                        val_every=5, use_amp=use_amp,
                    )
    return time.perf_counter() - start


def bench_ensemble(device: str, max_epochs: int, use_amp: bool) -> float:
    from src.ensemble_trainer import train_seeds_in_parallel
    from experiment_runner import _model_factory
    start = time.perf_counter()
    for n_group in BENCH_GROUPS:
        for n_train in BENCH_N_TRAIN:
            trains, vals, hashes = _build_datasets(n_group, n_train, BENCH_SEEDS)
            for mtype in BENCH_MTYPES:
                lam = 1e-3 if mtype == "regularized" else 0.0
                factory = _model_factory(mtype, n_group, 32, 2)
                train_seeds_in_parallel(
                    model_factory=factory, model_type=mtype,
                    train_datasets=trains, val_datasets=vals,
                    seeds=BENCH_SEEDS, n_group=n_group, n_train=n_train, epsilon=0.0,
                    lambda_l2=lam, max_epochs=max_epochs, patience=10,
                    batch_size=64, lr=1e-3, device=device,
                    val_every=5, use_amp=use_amp,
                    dataset_hashes=hashes,
                )
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-epochs", type=int, default=100,
                        help="Cap epochs; benchmark is about overhead, not full convergence")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--trainers", nargs="+", default=["original", "fast", "ensemble"],
                        choices=["original", "fast", "ensemble"])
    args = parser.parse_args()
    use_amp = not args.no_amp

    # Warm up (compile kernels, JIT init)
    print("Warming up…")
    if "original" in args.trainers:
        from src.trainer import train_one_run
        tr, va, _, _ = generate_petal_dataset(n_petals=2, N_train=50, seed=0)
        m = get_model_suite(2, 32, 2)["equivariant"]
        train_one_run(model=m, model_type="equivariant", train_dataset=tr, val_dataset=va,
                      n_group=2, n_train=50, epsilon=0.0, seed=0,
                      max_epochs=5, patience=30, batch_size=64, lr=1e-3, device=args.device)

    bench_runs = len(BENCH_GROUPS) * len(BENCH_N_TRAIN) * len(BENCH_SEEDS) * len(BENCH_MTYPES)
    bench_cells = len(BENCH_GROUPS) * len(BENCH_N_TRAIN) * len(BENCH_MTYPES)
    print(f"\nBenchmark grid: {bench_runs} runs ({bench_cells} ensemble cells)")
    print(f"Full Phase 1:   {FULL_PHASE_1_CELLS} runs ({FULL_PHASE_1_CELLS // 5} ensemble cells)")
    print(f"Scale factor:   {FULL_PHASE_1_CELLS / bench_runs:.0f}×")
    print(f"Device: {args.device}  AMP: {use_amp}\n")

    results = {}
    if "original" in args.trainers:
        print("Running ORIGINAL trainer…")
        results["original"] = bench_original(args.device, args.max_epochs)
        print(f"  → {results['original']:.1f}s\n")
    if "fast" in args.trainers:
        print("Running FAST trainer…")
        results["fast"] = bench_fast(args.device, args.max_epochs, use_amp)
        print(f"  → {results['fast']:.1f}s\n")
    if "ensemble" in args.trainers:
        print("Running ENSEMBLE trainer…")
        results["ensemble"] = bench_ensemble(args.device, args.max_epochs, use_amp)
        print(f"  → {results['ensemble']:.1f}s\n")

    # Report
    print("─" * 70)
    print(f"{'Trainer':<12} {'Wall time':>12} {'Per run':>12} {'Speedup':>10} {'Phase 1 est':>14}")
    print("─" * 70)
    baseline = results.get("original", min(results.values()))
    for trainer in ("original", "fast", "ensemble"):
        if trainer not in results:
            continue
        t = results[trainer]
        per_run = t / bench_runs * 1000  # ms
        speedup = baseline / t
        phase_1_est = t * (FULL_PHASE_1_CELLS / bench_runs)
        phase_1_min = phase_1_est / 60
        print(f"{trainer:<12} {t:>9.1f}s   {per_run:>9.1f}ms   {speedup:>7.2f}×   "
              f"{phase_1_min:>10.1f} min")
    print("─" * 70)

    if "ensemble" in results:
        ens_min = results["ensemble"] * (FULL_PHASE_1_CELLS / bench_runs) / 60
        target_min = 60
        if ens_min <= target_min:
            print(f"\n✓ Ensemble projected Phase 1 = {ens_min:.1f} min  ≤  {target_min} min target")
        else:
            print(f"\n✗ Ensemble projected Phase 1 = {ens_min:.1f} min  >  {target_min} min target")


if __name__ == "__main__":
    main()
