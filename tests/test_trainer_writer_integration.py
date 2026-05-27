"""
Tests that ResultWriter is correctly invoked by the trainers and that
crash recovery works end-to-end.
"""
import json
from pathlib import Path

import pytest
import torch

from src.data_generator import generate_petal_dataset
from src.ensemble_trainer import train_seeds_in_parallel
from src.fast_trainer import train_one_run_fast
from src.models import CnEquivariantMLP, VanillaMLP
from src.result_writer import ResultWriter


def _tiny():
    return generate_petal_dataset(n_petals=2, N_train=80, N_val=200, seed=0)


# ─── fast_trainer + writer ────────────────────────────────────────────────────


class TestFastTrainerWriter:
    def test_runs_without_writer(self):
        tr, va, _, _ = _tiny()
        model = CnEquivariantMLP(n=2, hidden_dim=4)
        result = train_one_run_fast(
            model=model, model_type="equivariant",
            train_dataset=tr, val_dataset=va,
            n_group=2, n_train=80, epsilon=0.0, seed=0,
            max_epochs=10, patience=999, batch_size=16, lr=1e-3,
            device="cpu", val_every=5, use_amp=False,
        )
        assert "best_val_acc" in result

    def test_writes_heartbeat_when_writer_provided(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        tr, va, _, _ = _tiny()
        model = CnEquivariantMLP(n=2, hidden_dim=4)
        train_one_run_fast(
            model=model, model_type="equivariant",
            train_dataset=tr, val_dataset=va,
            n_group=2, n_train=80, epsilon=0.0, seed=0,
            max_epochs=20, patience=999, batch_size=16, lr=1e-3,
            device="cpu", val_every=5, use_amp=False,
            result_writer=rw, cell_key="test_cell",
            heartbeat_every_s=0.001,  # heartbeat every iteration
        )
        # Heartbeat should have been written at least once
        assert rw.heartbeat_path.exists()
        hb = rw.read_heartbeat()
        assert hb["cell"] == "test_cell"
        assert hb["trainer"] == "fast"


# ─── ensemble_trainer + writer ───────────────────────────────────────────────


def _build_seed_datasets(seeds, n_petals=2, N=80):
    trains, vals, hashes = [], [], []
    for s in seeds:
        tr, va, _, meta = generate_petal_dataset(n_petals=n_petals, N_train=N, N_val=200, seed=s)
        trains.append(tr); vals.append(va); hashes.append(meta["dataset_hash"])
    return trains, vals, hashes


class TestEnsembleTrainerWriter:
    def test_runs_without_writer(self):
        seeds = [0, 1]
        trains, vals, hashes = _build_seed_datasets(seeds)
        results = train_seeds_in_parallel(
            model_factory=lambda: CnEquivariantMLP(n=2, hidden_dim=4),
            model_type="equivariant",
            train_datasets=trains, val_datasets=vals,
            seeds=seeds, n_group=2, n_train=80, epsilon=0.0,
            max_epochs=10, patience=999, batch_size=16, lr=1e-3,
            device="cpu", val_every=5, use_amp=False,
            dataset_hashes=hashes,
        )
        assert len(results) == 2

    def test_heartbeat_with_cell_key_format(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        seeds = [0, 1]
        trains, vals, hashes = _build_seed_datasets(seeds)
        train_seeds_in_parallel(
            model_factory=lambda: CnEquivariantMLP(n=2, hidden_dim=4),
            model_type="equivariant",
            train_datasets=trains, val_datasets=vals,
            seeds=seeds, n_group=2, n_train=80, epsilon=0.0,
            max_epochs=20, patience=999, batch_size=16, lr=1e-3,
            device="cpu", val_every=5, use_amp=False,
            dataset_hashes=hashes,
            result_writer=rw, cell_key="myCell",
            heartbeat_every_s=0.001,
        )
        hb = rw.read_heartbeat()
        assert hb["cell"] == "myCell"
        assert hb["trainer"] == "ensemble"
        assert hb["n_seeds"] == 2

    def test_partial_results_on_runtime_error(self, tmp_path, monkeypatch):
        """If RuntimeError fires mid-training, partial results are returned with anomaly_flag."""
        rw = ResultWriter(results_dir=str(tmp_path))
        seeds = [0, 1]
        trains, vals, hashes = _build_seed_datasets(seeds)

        # Force a RuntimeError after a few epochs
        call_count = {"n": 0}
        from src import ensemble_trainer as et
        original_einsum = torch.einsum

        def boom_einsum(*a, **k):
            call_count["n"] += 1
            if call_count["n"] > 5:
                raise RuntimeError("simulated CUDA error")
            return original_einsum(*a, **k)

        # Inject only for augmented path; otherwise call original
        # Actually we test on equivariant — patch directly in vmapped call
        # Simpler: patch torch.gather
        original_gather = torch.gather
        def boom_gather(*a, **k):
            call_count["n"] += 1
            if call_count["n"] > 5:
                raise RuntimeError("simulated CUDA error")
            return original_gather(*a, **k)
        monkeypatch.setattr(torch, "gather", boom_gather)

        results = train_seeds_in_parallel(
            model_factory=lambda: CnEquivariantMLP(n=2, hidden_dim=4),
            model_type="equivariant",
            train_datasets=trains, val_datasets=vals,
            seeds=seeds, n_group=2, n_train=80, epsilon=0.0,
            max_epochs=50, patience=999, batch_size=16, lr=1e-3,
            device="cpu", val_every=5, use_amp=False,
            dataset_hashes=hashes,
            result_writer=rw, cell_key="boomCell",
            save_partial_on_error=True,
        )
        # Should return partial results, not raise
        assert len(results) == 2
        for r in results:
            assert r["anomaly_flag"] is True
            assert "interrupted" in r["anomaly_reason"]

        # The progress log should record the interruption
        events = rw.read_progress_log()
        interrupt_events = [e for e in events if e["event"] == "cell_interrupted"]
        assert len(interrupt_events) == 1
        assert interrupt_events[0]["cell"] == "boomCell"

    def test_keyboard_interrupt_always_propagates(self, tmp_path, monkeypatch):
        """KeyboardInterrupt must propagate even with save_partial_on_error=True."""
        rw = ResultWriter(results_dir=str(tmp_path))
        seeds = [0, 1]
        trains, vals, hashes = _build_seed_datasets(seeds)

        call_count = {"n": 0}
        original_gather = torch.gather
        def boom_gather(*a, **k):
            call_count["n"] += 1
            if call_count["n"] > 5:
                raise KeyboardInterrupt("user Ctrl-C")
            return original_gather(*a, **k)
        monkeypatch.setattr(torch, "gather", boom_gather)

        with pytest.raises(KeyboardInterrupt):
            train_seeds_in_parallel(
                model_factory=lambda: CnEquivariantMLP(n=2, hidden_dim=4),
                model_type="equivariant",
                train_datasets=trains, val_datasets=vals,
                seeds=seeds, n_group=2, n_train=80, epsilon=0.0,
                max_epochs=50, patience=999, batch_size=16, lr=1e-3,
                device="cpu", val_every=5, use_amp=False,
                dataset_hashes=hashes,
                result_writer=rw, cell_key="ctrlcCell",
                save_partial_on_error=True,  # still propagates KeyboardInterrupt
            )

        # Even though we raised, the interrupt should have been logged
        events = rw.read_progress_log()
        interrupt_events = [e for e in events if e["event"] == "cell_interrupted"]
        assert len(interrupt_events) == 1
        assert "KeyboardInterrupt" in interrupt_events[0]["reason"]

    def test_save_partial_on_error_false_propagates(self, tmp_path, monkeypatch):
        """When save_partial_on_error=False, the exception propagates."""
        rw = ResultWriter(results_dir=str(tmp_path))
        seeds = [0, 1]
        trains, vals, hashes = _build_seed_datasets(seeds)

        call_count = {"n": 0}
        original_gather = torch.gather
        def boom_gather(*a, **k):
            call_count["n"] += 1
            if call_count["n"] > 5:
                raise RuntimeError("simulated")
            return original_gather(*a, **k)
        monkeypatch.setattr(torch, "gather", boom_gather)

        with pytest.raises(RuntimeError):
            train_seeds_in_parallel(
                model_factory=lambda: CnEquivariantMLP(n=2, hidden_dim=4),
                model_type="equivariant",
                train_datasets=trains, val_datasets=vals,
                seeds=seeds, n_group=2, n_train=80, epsilon=0.0,
                max_epochs=50, patience=999, batch_size=16, lr=1e-3,
                device="cpu", val_every=5, use_amp=False,
                dataset_hashes=hashes,
                result_writer=rw, cell_key="boomCell2",
                save_partial_on_error=False,
            )
