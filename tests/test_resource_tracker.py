"""Tests for RunResult schema and ResourceTracker file operations."""
import json
import time
from pathlib import Path

import pytest

from src.resource_tracker import ResourceTracker, RunResult


def _make_run_result(**overrides) -> RunResult:
    defaults = dict(
        run_id="test123",
        experiment_name="test",
        timestamp="2026-01-01T00:00:00",
        config_hash="abc",
        n_group=4,
        log2_n_group=2.0,
        n_train=100,
        epsilon=0.0,
        seed=0,
        model_type="equivariant",
        lambda_l2=0.0,
        best_val_acc=0.85,
        reached_target=True,
        epochs_to_target=42,
        total_epochs=72,
        wall_clock_seconds=5.2,
        gpu_energy_joules=None,
        flops_per_forward=1000,
        total_flops=72_000,
        peak_memory_mb=None,
        n_parameters=1185,
        n_effective_params=1185,
        id_twonn=2.1,
        id_pr=2.8,
        id_mle_k10=2.3,
        id_agreement_flag=True,
        orbit_consistency=0.001,
        val_acc_curve=[(0, 0.5), (42, 0.85)],
        train_loss_curve=[(0, 0.7)],
        dataset_hash="deadbeef",
        converged=True,
        anomaly_flag=False,
        anomaly_reason=None,
    )
    defaults.update(overrides)
    return RunResult(**defaults)


class TestRunResult:
    def test_instantiation(self):
        r = _make_run_result()
        assert r.run_id == "test123"
        assert r.reached_target is True

    def test_optional_fields_accept_none(self):
        r = _make_run_result(gpu_energy_joules=None, epochs_to_target=None,
                             anomaly_reason=None)
        assert r.gpu_energy_joules is None
        assert r.epochs_to_target is None


class TestResourceTrackerIO:
    def test_save_and_load(self, tmp_path):
        tracker = ResourceTracker(results_dir=str(tmp_path))
        result = _make_run_result()
        fpath = tracker.save(result)
        assert fpath.exists()

        loaded = tracker.load_all()
        assert len(loaded) == 1
        assert loaded[0]["run_id"] == "test123"

    def test_save_produces_valid_json(self, tmp_path):
        tracker = ResourceTracker(results_dir=str(tmp_path))
        result = _make_run_result()
        fpath = tracker.save(result)
        with open(fpath) as f:
            data = json.load(f)
        assert data["best_val_acc"] == pytest.approx(0.85)

    def test_load_all_empty(self, tmp_path):
        tracker = ResourceTracker(results_dir=str(tmp_path))
        assert tracker.load_all() == []

    def test_creates_results_dir(self, tmp_path):
        new_dir = tmp_path / "nested" / "runs"
        ResourceTracker(results_dir=str(new_dir))
        assert new_dir.exists()

    def test_start_stop_returns_timing(self):
        tracker = ResourceTracker()
        tracker.start()
        time.sleep(0.05)
        info = tracker.stop()
        assert "wall_clock_seconds" in info
        assert info["wall_clock_seconds"] >= 0.04
        assert "gpu_energy_joules" in info  # may be None if no GPU
