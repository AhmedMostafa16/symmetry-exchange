"""Tests for src/ensemble_trainer.py — vmap-batched seeds."""
import pytest
import torch

from src.data_generator import generate_petal_dataset
from src.ensemble_trainer import train_seeds_in_parallel
from src.models import AugmentedVanillaMLP, CnEquivariantMLP, VanillaMLP


def _build_datasets(seeds, n_petals=4, N=80):
    trains, vals, hashes = [], [], []
    for s in seeds:
        tr, va, _, meta = generate_petal_dataset(n_petals=n_petals, N_train=N, N_val=200, seed=s)
        trains.append(tr); vals.append(va); hashes.append(meta["dataset_hash"])
    return trains, vals, hashes


_FAST = dict(
    max_epochs=15, patience=999, batch_size=16, lr=1e-3,
    device="cpu", val_every=5, use_amp=False,
)


class TestEnsembleSmoke:
    def test_returns_per_seed_results(self):
        seeds = [0, 1, 2, 3, 4]
        trains, vals, hashes = _build_datasets(seeds)
        results = train_seeds_in_parallel(
            model_factory=lambda: CnEquivariantMLP(n=4, hidden_dim=8),
            model_type="equivariant",
            train_datasets=trains, val_datasets=vals,
            seeds=seeds, n_group=4, n_train=80, epsilon=0.0,
            dataset_hashes=hashes,
            **_FAST,
        )
        assert len(results) == len(seeds)
        assert {r["seed"] for r in results} == set(seeds)
        for r in results:
            assert 0 <= r["best_val_acc"] <= 1
            assert "ensemble_wall_clock_seconds" in r

    def test_vanilla_model(self):
        seeds = [0, 1]
        trains, vals, hashes = _build_datasets(seeds)
        results = train_seeds_in_parallel(
            model_factory=lambda: VanillaMLP(hidden_dim=8),
            model_type="vanilla",
            train_datasets=trains, val_datasets=vals,
            seeds=seeds, n_group=4, n_train=80, epsilon=0.0,
            dataset_hashes=hashes,
            **_FAST,
        )
        assert len(results) == 2

    def test_augmented_model(self):
        seeds = [0, 1]
        trains, vals, hashes = _build_datasets(seeds)
        results = train_seeds_in_parallel(
            model_factory=lambda: AugmentedVanillaMLP(n=4, hidden_dim=8),
            model_type="augmented",
            train_datasets=trains, val_datasets=vals,
            seeds=seeds, n_group=4, n_train=80, epsilon=0.0,
            dataset_hashes=hashes,
            **_FAST,
        )
        assert len(results) == 2
        # Augmented model should produce sensible accuracies
        for r in results:
            assert 0 <= r["best_val_acc"] <= 1


class TestEnsembleEquivalence:
    """Ensemble per-seed N_target should match sequential fast_trainer per-seed."""

    def test_per_seed_accuracy_matches_sequential(self):
        from src.fast_trainer import train_one_run_fast
        seeds = [0, 1, 2]
        trains, vals, hashes = _build_datasets(seeds, n_petals=2, N=200)

        # Sequential
        seq_results = []
        for s_idx, s in enumerate(seeds):
            torch.manual_seed(s)
            model = CnEquivariantMLP(n=2, hidden_dim=8)
            seq_results.append(train_one_run_fast(
                model=model, model_type="equivariant",
                train_dataset=trains[s_idx], val_dataset=vals[s_idx],
                n_group=2, n_train=200, epsilon=0.0, seed=s,
                max_epochs=20, patience=999, batch_size=32, lr=1e-3,
                device="cpu", val_every=5, use_amp=False,
            ))

        # Ensemble
        ens_results = train_seeds_in_parallel(
            model_factory=lambda: CnEquivariantMLP(n=2, hidden_dim=8),
            model_type="equivariant",
            train_datasets=trains, val_datasets=vals,
            seeds=seeds, n_group=2, n_train=200, epsilon=0.0,
            max_epochs=20, patience=999, batch_size=32, lr=1e-3,
            device="cpu", val_every=5, use_amp=False,
            dataset_hashes=hashes,
        )

        # Should be roughly equivalent; allow small numerical drift from vmap
        for seq, ens in zip(seq_results, sorted(ens_results, key=lambda r: r["seed"])):
            diff = abs(seq["best_val_acc"] - ens["best_val_acc"])
            assert diff < 0.1, (
                f"seed {seq['seed']}: seq={seq['best_val_acc']:.3f}, "
                f"ens={ens['best_val_acc']:.3f}, diff={diff:.3f}"
            )


class TestEnsembleDeterminism:
    def test_same_seed_same_result(self):
        seeds = [0, 1, 2]
        trains, vals, hashes = _build_datasets(seeds)

        results_a = train_seeds_in_parallel(
            model_factory=lambda: CnEquivariantMLP(n=4, hidden_dim=8),
            model_type="equivariant",
            train_datasets=trains, val_datasets=vals,
            seeds=seeds, n_group=4, n_train=80, epsilon=0.0,
            dataset_hashes=hashes,
            **_FAST,
        )
        results_b = train_seeds_in_parallel(
            model_factory=lambda: CnEquivariantMLP(n=4, hidden_dim=8),
            model_type="equivariant",
            train_datasets=trains, val_datasets=vals,
            seeds=seeds, n_group=4, n_train=80, epsilon=0.0,
            dataset_hashes=hashes,
            **_FAST,
        )
        for ra, rb in zip(results_a, results_b):
            assert ra["best_val_acc"] == pytest.approx(rb["best_val_acc"], abs=1e-6)
