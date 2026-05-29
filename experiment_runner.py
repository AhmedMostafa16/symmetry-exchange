"""
Main experiment runner for the symmetry exchange rate experiment.

PREREGISTERED_CONFIG holds the pre-specified experimental configuration; its
SHA-256 hash provides reproducibility / tamper-evidence (it confirms the released
config matches the data) but NOT pre-registration: the config was never deposited
in an external, timestamped registry before data collection. This study is
exploratory. (The PREREGISTERED_* names are retained for code continuity.)

Trainers (--trainer):
    original  : src/trainer.train_one_run (reference; slowest)
    fast      : src/fast_trainer.train_one_run_fast (~5× faster)
    ensemble  : src/ensemble_trainer.train_seeds_in_parallel (~15-25× faster)
                Batches all 5 seeds into a single vmapped training step.

Usage:
    uv run python experiment_runner.py --phase 1 --device cuda                 # ensemble (default)
    uv run python experiment_runner.py --phase 1 --device cuda --trainer fast  # per-seed
    uv run python experiment_runner.py --pilot --device cpu --trainer ensemble # smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.data_generator import generate_petal_dataset, run_dataset_adversarial_checks
from src.metrics import calibrate_id_estimators
from src.models import get_model_suite
from src.statistics import compute_analysis_hash

if TYPE_CHECKING:
    from src.result_writer import ResultWriter


# ─── Pre-registered config ────────────────────────────────────────────────────

PREREGISTERED_CONFIG = {
    "task": "cn_petal_classification",
    "n_groups": [1, 2, 3, 4, 6, 8, 12],
    "n_train_grid": [50, 100, 200, 400, 800, 1600, 3200, 6400],
    "epsilon_values": [0.0, 0.1, 0.2, 0.3],
    "n_seeds": 5,
    "seeds": [0, 1, 2, 3, 4],
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
    "primary_comparison": "slope_equivariant_vs_null",
    "secondary_comparison": "slope_equivariant_vs_wrong_group",
    "alpha": 0.05,
    "bonferroni_n": 3,
    "n_bootstrap": 10_000,
    "kill_if_equivariant_slope_ci_contains_zero": True,
    "kill_if_wrong_group_slope_within_0.2_of_equivariant": True,
}

PREREGISTERED_HASH = compute_analysis_hash(PREREGISTERED_CONFIG)


# ─── Done markers (resumability) ─────────────────────────────────────────────


def _done_marker_seq(results_dir: Path, n: int, N: int, seed: int, mtype: str) -> Path:
    return results_dir / f"done_{n}_{N}_{seed}_{mtype}.flag"


def _done_marker_ens(results_dir: Path, n: int, N: int, epsilon: float, mtype: str) -> Path:
    eps_str = f"{epsilon:.2f}".replace(".", "p")
    return results_dir / f"done_ens_{n}_{N}_eps{eps_str}_{mtype}.flag"


# ─── Model factory for ensemble training ─────────────────────────────────────


def _model_factory(model_type: str, n_group: int, hidden_dim: int, n_hidden: int):
    """Closure that returns a fresh model of `model_type`.
    Called inside torch.manual_seed(seed) context for per-seed init."""
    def factory():
        return get_model_suite(n_group, hidden_dim, n_hidden)[model_type]
    return factory


# ─── Phase runners ───────────────────────────────────────────────────────────


def run_phase_ensemble(
    phase: int,
    device: str,
    pilot: bool,
    results_dir: Path,
    fast_patience: int = 10,
    val_every: int = 5,
    use_amp: bool = True,
    epsilons_override: list[float] | None = None,
    writer: "ResultWriter | None" = None,
) -> None:
    """Ensemble trainer: batches all seeds per (epsilon, n_group, N, model_type) cell."""
    from src.ensemble_trainer import train_seeds_in_parallel
    from src.result_writer import ResultWriter

    if writer is None:
        writer = ResultWriter(results_dir=str(results_dir),
                              experiment_name=f"symmetry_exchange_phase{phase}")
    writer.cleanup_temp_files()

    cfg = PREREGISTERED_CONFIG
    n_groups = [2, 4] if pilot else cfg["n_groups"]
    n_trains = [100, 400] if pilot else cfg["n_train_grid"]
    seeds    = [0, 1] if pilot else cfg["seeds"]
    if epsilons_override is not None:
        epsilons = epsilons_override
    else:
        epsilons = [0.0] if phase == 1 else cfg["epsilon_values"]
    mtypes   = cfg["model_types"]

    n_cells = len(epsilons) * len(n_groups) * len(n_trains) * len(mtypes)
    cell_idx = 0
    completed_cells = 0
    failed_cells = 0
    import time as _time

    writer.log_event("phase_started", phase=phase, n_cells_total=n_cells,
                     trainer="ensemble", pilot=pilot,
                     n_groups=n_groups, n_trains=n_trains,
                     seeds=seeds, epsilons=epsilons)

    phase_start = _time.perf_counter()

    for epsilon in epsilons:
        for n_group in n_groups:
            for n_train in n_trains:
                # Build per-seed datasets (cheap, ~ms each)
                trains, vals, hashes = [], [], []
                for s in seeds:
                    tr, va, _, meta = generate_petal_dataset(
                        n_petals=n_group, N_train=n_train, epsilon=epsilon, seed=s,
                    )
                    trains.append(tr); vals.append(va); hashes.append(meta["dataset_hash"])

                if epsilon == 0.0:
                    checks = run_dataset_adversarial_checks(
                        trains[0], vals[0], n_petals=n_group, epsilon=epsilon, verbose=False,
                    )
                    if not checks["all_checks_passed"]:
                        msg = f"Dataset checks FAILED for n={n_group}. Skipping."
                        print(f"  ✗ {msg}")
                        writer.log_event("dataset_check_failed", n_group=n_group,
                                         n_train=n_train, epsilon=epsilon, details=checks)
                        continue

                for mtype in mtypes:
                    cell_idx += 1
                    cell_key = f"n{n_group}_N{n_train}_eps{epsilon:.2f}_{mtype}"
                    marker = _done_marker_ens(results_dir, n_group, n_train, epsilon, mtype)
                    if marker.exists():
                        continue

                    lambda_l2 = cfg["lambda_l2_regularized"] if mtype == "regularized" else 0.0
                    factory = _model_factory(mtype, n_group, cfg["hidden_dim"], cfg["n_hidden"])

                    writer.log_event("cell_started", cell=cell_key, cell_idx=cell_idx,
                                     n_cells=n_cells, n_seeds=len(seeds))
                    writer.heartbeat({
                        "cell": cell_key, "cell_idx": cell_idx, "n_cells": n_cells,
                        "completed": completed_cells, "failed": failed_cells,
                        "phase": phase,
                    })

                    t0 = _time.perf_counter()
                    try:
                        results = train_seeds_in_parallel(
                            model_factory=factory,
                            model_type=mtype,
                            train_datasets=trains,
                            val_datasets=vals,
                            seeds=seeds,
                            n_group=n_group,
                            n_train=n_train,
                            epsilon=epsilon,
                            lambda_l2=lambda_l2,
                            target_acc=cfg["target_acc"],
                            max_epochs=cfg["max_epochs"],
                            patience=fast_patience,
                            batch_size=cfg["batch_size"],
                            lr=cfg["lr"],
                            device=device,
                            val_every=val_every,
                            use_amp=use_amp,
                            config_hash=PREREGISTERED_HASH,
                            dataset_hashes=hashes,
                            result_writer=writer,
                            cell_key=cell_key,
                            save_partial_on_error=True,
                        )
                    except Exception as e:
                        failed_cells += 1
                        print(f"  ✗ Cell {cell_key} FAILED: {e}")
                        writer.log_event("cell_error", cell=cell_key, error=str(e),
                                         error_type=type(e).__name__)
                        continue
                    cell_time = _time.perf_counter() - t0

                    # Save per-seed results atomically — write each one before
                    # touching the done marker
                    accs = []
                    saved_run_ids = []
                    for r in results:
                        writer.save_run(r)
                        accs.append(r["best_val_acc"])
                        saved_run_ids.append(r["run_id"])
                    # Only mark cell done if all results saved successfully
                    marker.touch()
                    completed_cells += 1

                    n_above = sum(a >= cfg["target_acc"] for a in accs)
                    status = "✓" if n_above >= max(3, len(seeds) // 2) else " "

                    writer.log_event("cell_complete", cell=cell_key,
                                     wall_time_s=cell_time, best_val_accs=accs,
                                     n_above_target=n_above, run_ids=saved_run_ids)

                    print(
                        f"  [{status}] [{cell_idx}/{n_cells}] eps={epsilon:.1f} "
                        f"n={n_group:2d} N={n_train:5d} {mtype:12s}  "
                        f"accs={[f'{a:.2f}' for a in accs]}  "
                        f"({cell_time:.1f}s)"
                    )

    phase_wall = _time.perf_counter() - phase_start
    writer.log_event("phase_complete", phase=phase,
                     completed_cells=completed_cells, failed_cells=failed_cells,
                     n_cells_total=n_cells, wall_time_s=phase_wall)
    print(f"\nPhase {phase} (ensemble) complete: {completed_cells} cells, "
          f"{failed_cells} failed, {phase_wall:.1f}s")


def run_phase_sequential(
    phase: int,
    device: str,
    pilot: bool,
    results_dir: Path,
    trainer: str = "fast",
    fast_patience: int = 10,
    val_every: int = 5,
    use_amp: bool = True,
    epsilons_override: list[float] | None = None,
    writer: "ResultWriter | None" = None,
) -> None:
    """Sequential trainer: one run per (cell, seed). Use trainer='fast' or 'original'."""
    from src.result_writer import ResultWriter

    if trainer == "fast":
        from src.fast_trainer import train_one_run_fast as train_fn
    else:
        from src.trainer import train_one_run as train_fn

    if writer is None:
        writer = ResultWriter(results_dir=str(results_dir),
                              experiment_name=f"symmetry_exchange_phase{phase}")
    writer.cleanup_temp_files()

    cfg = PREREGISTERED_CONFIG
    n_groups = [2, 4] if pilot else cfg["n_groups"]
    n_trains = [100, 400] if pilot else cfg["n_train_grid"]
    seeds    = [0, 1] if pilot else cfg["seeds"]
    if epsilons_override is not None:
        epsilons = epsilons_override
    else:
        epsilons = [0.0] if phase == 1 else cfg["epsilon_values"]

    writer.log_event("phase_started", phase=phase, trainer=trainer, pilot=pilot,
                     epsilons=epsilons, seeds=seeds)

    for epsilon in epsilons:
        for n_group in n_groups:
            for n_train in n_trains:
                for seed in seeds:
                    tr, va, _, ds_meta = generate_petal_dataset(
                        n_petals=n_group, N_train=n_train, epsilon=epsilon, seed=seed,
                    )

                    if seed == 0 and epsilon == 0.0:
                        checks = run_dataset_adversarial_checks(
                            tr, va, n_petals=n_group, epsilon=epsilon, verbose=False,
                        )
                        if not checks["all_checks_passed"]:
                            print(f"  ✗ Dataset checks FAILED for n={n_group}. Skipping.")
                            continue

                    # Seed model init deterministically before constructing models
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    models = get_model_suite(n_group, cfg["hidden_dim"], cfg["n_hidden"])

                    for mtype, model in models.items():
                        marker = _done_marker_seq(results_dir, n_group, n_train, seed, mtype)
                        if marker.exists():
                            continue

                        lambda_l2 = cfg["lambda_l2_regularized"] if mtype == "regularized" else 0.0
                        cell_key = f"n{n_group}_N{n_train}_eps{epsilon:.2f}_s{seed}_{mtype}"

                        kwargs = dict(
                            model=model, model_type=mtype,
                            train_dataset=tr, val_dataset=va,
                            n_group=n_group, n_train=n_train, epsilon=epsilon, seed=seed,
                            lambda_l2=lambda_l2,
                            target_acc=cfg["target_acc"],
                            max_epochs=cfg["max_epochs"],
                            batch_size=cfg["batch_size"],
                            lr=cfg["lr"],
                            device=device,
                            config_hash=PREREGISTERED_HASH,
                            dataset_hash=ds_meta["dataset_hash"],
                        )
                        if trainer == "fast":
                            kwargs.update(patience=fast_patience, val_every=val_every,
                                          use_amp=use_amp,
                                          result_writer=writer, cell_key=cell_key)
                        else:
                            kwargs.update(patience=cfg["patience"],
                                          checkpoint_dir="results/checkpoints")

                        writer.log_event("cell_started", cell=cell_key)

                        try:
                            result = train_fn(**kwargs)
                        except Exception as e:
                            writer.log_event("cell_error", cell=cell_key,
                                             error=str(e), error_type=type(e).__name__)
                            print(f"  ✗ {cell_key} FAILED: {e}")
                            continue

                        # Atomic save via writer (replaces direct file write)
                        writer.save_run(result)
                        marker.touch()

                        writer.log_event("cell_complete", cell=cell_key,
                                         best_val_acc=result["best_val_acc"],
                                         reached_target=result["reached_target"])

                        status = "✓" if result["reached_target"] else " "
                        print(
                            f"  [{status}] eps={epsilon:.1f} n={n_group:2d} N={n_train:5d} "
                            f"seed={seed} {mtype:12s}  val={result['best_val_acc']:.3f}"
                        )

    writer.log_event("phase_complete", phase=phase)
    print(f"\nPhase {phase} ({trainer}) complete.")


# ─── Entry point ─────────────────────────────────────────────────────────────


def run_phase(
    phase: int,
    device: str,
    pilot: bool = False,
    trainer: str = "ensemble",
    results_dir: Path = Path("results/runs"),
    fast_patience: int = 10,
    val_every: int = 5,
    use_amp: bool = True,
    epsilons_override: list[float] | None = None,
) -> None:
    from src.result_writer import ResultWriter

    results_dir.mkdir(parents=True, exist_ok=True)
    Path("results/checkpoints").mkdir(parents=True, exist_ok=True)

    print(f"\n── Phase {phase} {'(pilot)' if pilot else ''}  trainer={trainer}  device={device} ──")
    print(f"Config hash: {PREREGISTERED_HASH[:16]}")
    if epsilons_override is not None:
        print(f"Epsilons (overridden): {epsilons_override}")

    # One ResultWriter for the whole phase; passed down to sub-runners.
    # Cleans up orphaned .tmp files from previous crashed runs at startup.
    writer = ResultWriter(
        results_dir=str(results_dir),
        experiment_name=f"symmetry_exchange_phase{phase}",
    )
    n_cleaned = writer.cleanup_temp_files()
    if n_cleaned:
        print(f"  Cleaned up {n_cleaned} orphaned .tmp file(s) from prior runs")
    print(f"  Progress log: {writer.progress_log_path}")
    print(f"  Heartbeat:    {writer.heartbeat_path}")

    print("\n── Calibrating ID estimators ──")
    cal = calibrate_id_estimators()
    if not cal.get("twonn_reliable"):
        print("WARNING: TwoNN unreliable on calibration data. Flag all ID results.")

    if trainer == "ensemble":
        run_phase_ensemble(phase, device, pilot, results_dir,
                            fast_patience=fast_patience, val_every=val_every,
                            use_amp=use_amp, epsilons_override=epsilons_override,
                            writer=writer)
    else:
        run_phase_sequential(phase, device, pilot, results_dir,
                              trainer=trainer, fast_patience=fast_patience,
                              val_every=val_every, use_amp=use_amp,
                              epsilons_override=epsilons_override,
                              writer=writer)

    # Final summary
    summary = writer.summarise()
    print(f"\n=== Phase {phase} summary ===")
    print(f"  Runs saved:      {summary['n_runs_saved']}")
    print(f"  Events logged:   {summary['n_events_logged']}")

    if phase == 1 and epsilons_override is None:
        print("Run analysis before proceeding to phase 2.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Symmetry exchange rate experiment")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--pilot", action="store_true",
                        help="Smoke test: 2 groups × 2 N_train × 2 seeds only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trainer", default="ensemble",
                        choices=["original", "fast", "ensemble"])
    parser.add_argument("--patience", type=int, default=10,
                        help="Early-stop patience (for fast/ensemble trainers)")
    parser.add_argument("--val-every", type=int, default=5,
                        help="Validate every K epochs (for fast/ensemble)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable bfloat16 autocast on GPU")
    parser.add_argument("--results-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--epsilons", nargs="+", type=float, default=None,
                        help="Override epsilon values (e.g. --epsilons 0.1 0.2 0.3 to skip eps=0)")
    args = parser.parse_args()

    run_phase(
        phase=args.phase, device=args.device, pilot=args.pilot, trainer=args.trainer,
        results_dir=args.results_dir,
        fast_patience=args.patience, val_every=args.val_every,
        use_amp=not args.no_amp,
        epsilons_override=args.epsilons,
    )


if __name__ == "__main__":
    main()
