"""Tests for src/fast_trainer.py — determinism, equivalence, and smoke runs."""
import pytest
import torch

from src.data_generator import generate_petal_dataset
from src.fast_trainer import train_one_run_fast
from src.models import AugmentedVanillaMLP, CnEquivariantMLP, VanillaMLP

_FAST = dict(
    max_epochs=10, patience=999, batch_size=16, lr=1e-3,
    device="cpu", val_every=5, use_amp=False,
)


def _tiny(n_petals=4, N=40):
    return generate_petal_dataset(n_petals=n_petals, N_train=N, N_val=200, seed=0)


class TestSmoke:
    def test_returns_required_keys(self):
        tr, va, _, _ = _tiny()
        model = CnEquivariantMLP(n=4, hidden_dim=4)
        result = train_one_run_fast(
            model=model, model_type="equivariant",
            train_dataset=tr, val_dataset=va,
            n_group=4, n_train=40, epsilon=0.0, seed=0, **_FAST,
        )
        required = {"run_id", "best_val_acc", "reached_target", "total_epochs",
                    "wall_clock_seconds", "val_acc_curve", "train_loss_curve",
                    "anomaly_flag", "n_parameters"}
        assert required.issubset(result)

    def test_augmented_path(self):
        tr, va, _, _ = _tiny()
        model = AugmentedVanillaMLP(n=4, hidden_dim=4)
        result = train_one_run_fast(
            model=model, model_type="augmented",
            train_dataset=tr, val_dataset=va,
            n_group=4, n_train=40, epsilon=0.0, seed=0, **_FAST,
        )
        assert 0 <= result["best_val_acc"] <= 1

    def test_l2_regularization_accepted(self):
        tr, va, _, _ = _tiny()
        model = VanillaMLP(hidden_dim=4)
        result = train_one_run_fast(
            model=model, model_type="regularized",
            train_dataset=tr, val_dataset=va,
            n_group=4, n_train=40, epsilon=0.0, seed=1,
            lambda_l2=1e-3, **_FAST,
        )
        assert "best_val_acc" in result


class TestDeterminism:
    """Same seed → identical results across two independent runs."""

    def test_equivariant_deterministic(self):
        tr, va, _, _ = _tiny()
        results = []
        for _ in range(2):
            torch.manual_seed(0)
            model = CnEquivariantMLP(n=4, hidden_dim=8)
            results.append(train_one_run_fast(
                model=model, model_type="equivariant",
                train_dataset=tr, val_dataset=va,
                n_group=4, n_train=40, epsilon=0.0, seed=42, **_FAST,
            ))
        assert results[0]["best_val_acc"] == pytest.approx(results[1]["best_val_acc"], abs=1e-6)
        assert results[0]["train_loss_curve"] == results[1]["train_loss_curve"]

    def test_vanilla_deterministic(self):
        tr, va, _, _ = _tiny()
        results = []
        for _ in range(2):
            torch.manual_seed(0)
            model = VanillaMLP(hidden_dim=8)
            results.append(train_one_run_fast(
                model=model, model_type="vanilla",
                train_dataset=tr, val_dataset=va,
                n_group=4, n_train=40, epsilon=0.0, seed=7, **_FAST,
            ))
        assert results[0]["val_acc_curve"] == results[1]["val_acc_curve"]


class TestEarlyStopping:
    def test_stops_when_target_reached_with_patience(self):
        # Use a permissive target so model converges quickly
        tr, va, _, _ = _tiny(n_petals=2, N=200)
        model = CnEquivariantMLP(n=2, hidden_dim=16)
        result = train_one_run_fast(
            model=model, model_type="equivariant",
            train_dataset=tr, val_dataset=va,
            n_group=2, n_train=200, epsilon=0.0, seed=0,
            target_acc=0.55,  # easy target
            max_epochs=200, patience=3, batch_size=64,
            lr=1e-2, device="cpu", val_every=2, use_amp=False,
        )
        # Should stop well before max_epochs since target is easy
        assert result["total_epochs"] < 200


class TestEquivalenceWithOriginal:
    """Fast and original should reach the same target (within grid-step
    tolerance); exact loss values may differ due to val frequency."""

    def test_similar_final_accuracy(self):
        from src.trainer import train_one_run
        tr, va, _, _ = _tiny(n_petals=2, N=200)

        torch.manual_seed(0)
        m1 = CnEquivariantMLP(n=2, hidden_dim=16)
        r_orig = train_one_run(
            model=m1, model_type="equivariant",
            train_dataset=tr, val_dataset=va,
            n_group=2, n_train=200, epsilon=0.0, seed=0,
            max_epochs=30, patience=999, batch_size=32, lr=1e-3, device="cpu",
        )

        torch.manual_seed(0)
        m2 = CnEquivariantMLP(n=2, hidden_dim=16)
        r_fast = train_one_run_fast(
            model=m2, model_type="equivariant",
            train_dataset=tr, val_dataset=va,
            n_group=2, n_train=200, epsilon=0.0, seed=0,
            max_epochs=30, patience=999, batch_size=32, lr=1e-3,
            device="cpu", val_every=1, use_amp=False,
        )

        # Different shuffle orders (DataLoader vs torch.randperm) and BCE
        # reduction order give different gradient paths. Both should converge
        # to comparable accuracy on this easy task — within 0.15 is plenty
        # for scientific equivalence (target_acc=0.80 grid step is much coarser).
        assert abs(r_orig["best_val_acc"] - r_fast["best_val_acc"]) < 0.15, (
            f"orig={r_orig['best_val_acc']:.3f}, fast={r_fast['best_val_acc']:.3f}"
        )
        assert r_orig["best_val_acc"] > 0.55, "Original failed to learn easy task"
        assert r_fast["best_val_acc"] > 0.55, "Fast failed to learn easy task"
