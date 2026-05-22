"""Smoke tests for the training loop — fast config (epochs=5, N=40, h=4)."""
import pytest
import torch

from src.data_generator import generate_petal_dataset
from src.models import CnEquivariantMLP, VanillaMLP, AugmentedVanillaMLP
from src.trainer import train_one_run, _extract_penultimate

_FAST = dict(max_epochs=5, patience=3, batch_size=16, lr=1e-3, device="cpu")


def _tiny_datasets(n_petals=4, N=40):
    return generate_petal_dataset(n_petals=n_petals, N_train=N, N_val=200, seed=0)


class TestTrainOneRun:
    def test_returns_required_keys(self):
        train_ds, val_ds, _, _ = _tiny_datasets()
        model = CnEquivariantMLP(n=4, hidden_dim=4, n_hidden=2)
        result = train_one_run(
            model=model, model_type="equivariant",
            train_dataset=train_ds, val_dataset=val_ds,
            n_group=4, n_train=40, epsilon=0.0, seed=0,
            **_FAST,
        )
        required = {"run_id", "best_val_acc", "reached_target", "total_epochs",
                    "wall_clock_seconds", "val_acc_curve", "train_loss_curve",
                    "anomaly_flag", "n_parameters", "orbit_consistency"}
        assert required.issubset(result.keys())

    def test_best_val_acc_in_unit_interval(self):
        train_ds, val_ds, _, _ = _tiny_datasets()
        model = VanillaMLP(hidden_dim=4)
        result = train_one_run(
            model=model, model_type="vanilla",
            train_dataset=train_ds, val_dataset=val_ds,
            n_group=4, n_train=40, epsilon=0.0, seed=1,
            **_FAST,
        )
        assert 0.0 <= result["best_val_acc"] <= 1.0

    def test_val_acc_curve_has_right_length(self):
        train_ds, val_ds, _, _ = _tiny_datasets()
        model = VanillaMLP(hidden_dim=4)
        result = train_one_run(
            model=model, model_type="vanilla",
            train_dataset=train_ds, val_dataset=val_ds,
            n_group=4, n_train=40, epsilon=0.0, seed=2,
            **_FAST,
        )
        assert len(result["val_acc_curve"]) <= _FAST["max_epochs"]
        assert len(result["val_acc_curve"]) > 0

    def test_augmented_model_trains(self):
        train_ds, val_ds, _, _ = _tiny_datasets()
        model = AugmentedVanillaMLP(n=4, hidden_dim=4)
        result = train_one_run(
            model=model, model_type="augmented",
            train_dataset=train_ds, val_dataset=val_ds,
            n_group=4, n_train=40, epsilon=0.0, seed=3,
            **_FAST,
        )
        assert result["best_val_acc"] > 0

    def test_l2_regularization_accepted(self):
        train_ds, val_ds, _, _ = _tiny_datasets()
        model = VanillaMLP(hidden_dim=4)
        result = train_one_run(
            model=model, model_type="regularized",
            train_dataset=train_ds, val_dataset=val_ds,
            n_group=4, n_train=40, epsilon=0.0, seed=4,
            lambda_l2=1e-3,
            **_FAST,
        )
        assert "best_val_acc" in result

    def test_no_nan_loss_for_normal_run(self):
        # A 5-epoch/40-sample smoke test may trigger stuck_at_chance (correct behaviour).
        # Only assert NaN loss — the real anomaly that indicates numerical failure.
        train_ds, val_ds, _, _ = _tiny_datasets()
        model = CnEquivariantMLP(n=4, hidden_dim=4)
        result = train_one_run(
            model=model, model_type="equivariant",
            train_dataset=train_ds, val_dataset=val_ds,
            n_group=4, n_train=40, epsilon=0.0, seed=5,
            **_FAST,
        )
        assert result["anomaly_reason"] != "loss_nan", "Training produced NaN loss"
        assert isinstance(result["anomaly_flag"], bool)


class TestExtractPenultimate:
    def test_returns_tensor(self):
        model = VanillaMLP(hidden_dim=8)
        x = torch.randn(20, 2)
        act = _extract_penultimate(model, x, "cpu")
        assert isinstance(act, torch.Tensor)
        assert act.shape[0] == 20

    def test_no_relu_returns_x(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 1))
        x = torch.randn(10, 2)
        act = _extract_penultimate(model, x, "cpu")
        assert act.shape == x.shape
