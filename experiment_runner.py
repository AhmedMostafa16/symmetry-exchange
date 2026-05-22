"""
Main experiment runner for the symmetry exchange rate experiment.

Pre-register PREREGISTERED_CONFIG by committing this file (or its hash) to git
BEFORE running any training. Deviations from the pre-registered config must
be reported as exploratory analyses, not confirmatory tests.

Usage:
    uv run python experiment_runner.py --phase 1         # epsilon=0 sweep
    uv run python experiment_runner.py --phase 2         # epsilon sweep
    uv run python experiment_runner.py --pilot           # quick smoke test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_generator import generate_petal_dataset, run_dataset_adversarial_checks
from src.metrics import calibrate_id_estimators, estimate_flops_per_forward
from src.models import get_model_suite
from src.statistics import compute_analysis_hash
from src.trainer import train_one_run


# ─── Pre-registered config ────────────────────────────────────────────────────
# COMMIT THIS FILE BEFORE RUNNING ANY TRAINING.
# The hash printed at startup must match the one in preregistration/config_hash.txt.

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


def _done_marker(results_dir: Path, n: int, N: int, seed: int, mtype: str) -> Path:
    return results_dir / f"done_{n}_{N}_{seed}_{mtype}.flag"


def run_phase(
    phase: int,
    device: str,
    pilot: bool = False,
    results_dir: Path = Path("results/runs"),
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    Path("results/checkpoints").mkdir(parents=True, exist_ok=True)

    cfg = PREREGISTERED_CONFIG

    # Narrow grid for pilot smoke test
    n_groups   = [2, 4] if pilot else cfg["n_groups"]
    n_trains   = [100, 400] if pilot else cfg["n_train_grid"]
    seeds      = [0, 1] if pilot else cfg["seeds"]
    epsilons   = [0.0] if phase == 1 else cfg["epsilon_values"]

    print(f"\n── Phase {phase} {'(pilot)' if pilot else ''} ─────────────────")
    print(f"Config hash: {PREREGISTERED_HASH[:16]}")

    # Step 0: Calibrate ID estimators
    print("\n── Calibrating ID estimators ──")
    cal = calibrate_id_estimators()
    if not cal.get("twonn_reliable"):
        print("WARNING: TwoNN unreliable on calibration data. Flag all ID results.")

    for epsilon in epsilons:
        for n_group in n_groups:
            for n_train in n_trains:
                for seed in seeds:
                    # Dataset (same across model types)
                    train_ds, val_ds, _, ds_meta = generate_petal_dataset(
                        n_petals=n_group, N_train=n_train, epsilon=epsilon, seed=seed
                    )

                    if seed == 0 and epsilon == 0.0:
                        checks = run_dataset_adversarial_checks(
                            train_ds, val_ds, n_petals=n_group,
                            epsilon=epsilon, verbose=False,
                        )
                        if not checks["all_checks_passed"]:
                            print(f"  ✗ Dataset checks FAILED for n={n_group}. Skipping.")
                            continue

                    models = get_model_suite(n_group, cfg["hidden_dim"], cfg["n_hidden"])

                    for mtype, model in models.items():
                        marker = _done_marker(results_dir, n_group, n_train, seed, mtype)
                        if marker.exists():
                            continue

                        lambda_l2 = (
                            cfg["lambda_l2_regularized"] if mtype == "regularized" else 0.0
                        )

                        result = train_one_run(
                            model=model,
                            model_type=mtype,
                            train_dataset=train_ds,
                            val_dataset=val_ds,
                            n_group=n_group,
                            n_train=n_train,
                            epsilon=epsilon,
                            seed=seed,
                            lambda_l2=lambda_l2,
                            target_acc=cfg["target_acc"],
                            max_epochs=cfg["max_epochs"],
                            patience=cfg["patience"],
                            batch_size=cfg["batch_size"],
                            lr=cfg["lr"],
                            device=device,
                            checkpoint_dir="results/checkpoints",
                            config_hash=PREREGISTERED_HASH,
                            dataset_hash=ds_meta["dataset_hash"],
                        )

                        rpath = results_dir / f"{result['run_id']}.json"
                        with open(rpath, "w") as f:
                            json.dump(result, f, indent=2)

                        marker.touch()
                        status = "✓" if result["reached_target"] else " "
                        print(
                            f"  [{status}] n={n_group:2d} N={n_train:5d} "
                            f"seed={seed} {mtype:12s}  "
                            f"val={result['best_val_acc']:.3f}"
                        )

    print(f"\nPhase {phase} complete.")
    if phase == 1:
        print("Run analysis (notebook 04) before proceeding to phase 2.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Symmetry exchange rate experiment")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--pilot", action="store_true",
                        help="Smoke test: 2 groups × 2 N_train × 2 seeds only")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_phase(phase=args.phase, device=args.device, pilot=args.pilot)


if __name__ == "__main__":
    main()
