"""
Tests for src/result_writer.py — atomic writes, JSONL progress log,
heartbeat, mid-training checkpoints.

The contract:
    - A crash mid-write must never produce a half-written JSON.
    - The progress log is append-only and remains parseable line-by-line.
    - Orphaned .tmp files from prior crashes get cleaned up on init.
    - Heartbeat reflects the latest call.
    - Checkpoints round-trip exactly.
"""
import json
import os
import time
from pathlib import Path

import pytest
import torch

from src.result_writer import ResultWriter, _atomic_write_json


# ─── _atomic_write_json ──────────────────────────────────────────────────────


class TestAtomicWrite:
    def test_writes_complete_json(self, tmp_path):
        target = tmp_path / "x.json"
        _atomic_write_json(target, {"a": 1, "b": [2, 3]})
        loaded = json.loads(target.read_text())
        assert loaded == {"a": 1, "b": [2, 3]}

    def test_no_tmp_file_left_after_success(self, tmp_path):
        target = tmp_path / "x.json"
        _atomic_write_json(target, {"a": 1})
        tmps = list(tmp_path.glob("*.tmp"))
        assert tmps == [], f"Orphaned tmp files: {tmps}"

    def test_overwrites_existing_file(self, tmp_path):
        target = tmp_path / "x.json"
        target.write_text("old content")
        _atomic_write_json(target, {"new": True})
        assert json.loads(target.read_text()) == {"new": True}

    def test_never_produces_half_written_file(self, tmp_path, monkeypatch):
        """If json.dump raises mid-write, the target must not exist (only .tmp may)."""
        target = tmp_path / "x.json"

        # Patch json.dump to raise after writing some bytes
        original_dump = json.dump
        def bad_dump(obj, fp, **kw):
            fp.write('{"partial":')
            raise RuntimeError("simulated crash")
        monkeypatch.setattr(json, "dump", bad_dump)

        with pytest.raises(RuntimeError):
            _atomic_write_json(target, {"x": 1})

        # Target itself must not exist (rename was never called)
        assert not target.exists(), "Half-written file leaked to target path"


# ─── ResultWriter basics ─────────────────────────────────────────────────────


class TestResultWriterInit:
    def test_creates_results_dir(self, tmp_path):
        path = tmp_path / "new" / "deep" / "dir"
        ResultWriter(results_dir=str(path))
        assert path.exists()

    def test_creates_checkpoint_dir(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        assert (tmp_path / "_checkpoints").exists()

    def test_cleanup_removes_orphaned_tmp_files(self, tmp_path):
        # Pre-existing .tmp files from a crashed run
        (tmp_path / "abc123.json.tmp").write_text('{"partial":')
        (tmp_path / "def456.json.tmp").write_text('{"also partial')
        ResultWriter(results_dir=str(tmp_path)).cleanup_temp_files()
        assert not list(tmp_path.glob("*.tmp"))


# ─── save_run ────────────────────────────────────────────────────────────────


class TestSaveRun:
    def test_writes_run_id_named_file(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        rw.save_run({"run_id": "abc123", "best_val_acc": 0.85})
        target = tmp_path / "abc123.json"
        assert target.exists()
        assert json.loads(target.read_text())["best_val_acc"] == 0.85

    def test_save_is_atomic_no_tmp_leak(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        rw.save_run({"run_id": "abc", "best_val_acc": 0.7})
        assert not list(tmp_path.glob("*.tmp"))


# ─── log_event (JSONL progress log) ──────────────────────────────────────────


class TestProgressLog:
    def test_appends_one_line_per_event(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        rw.log_event("started", phase=1)
        rw.log_event("epoch_complete", epoch=10, val_acc=0.81)
        rw.log_event("completed", phase=1)

        lines = (tmp_path / "progress.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3

    def test_each_line_is_valid_json(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        for i in range(5):
            rw.log_event("test", iteration=i, value=i * 2.5)

        for line in (tmp_path / "progress.jsonl").read_text().strip().split("\n"):
            obj = json.loads(line)
            assert "timestamp" in obj
            assert "event" in obj

    def test_event_contains_timestamp(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        rw.log_event("test", x=1)
        line = (tmp_path / "progress.jsonl").read_text().strip()
        obj = json.loads(line)
        assert "timestamp" in obj
        # ISO 8601 format check
        assert "T" in obj["timestamp"]

    def test_extra_fields_preserved(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        rw.log_event("cell_complete", cell="n=4_eps=0.0", n_seeds=5,
                     wall_time_s=12.3, val_accs=[0.81, 0.82, 0.79, 0.83, 0.80])
        obj = json.loads((tmp_path / "progress.jsonl").read_text().strip())
        assert obj["cell"] == "n=4_eps=0.0"
        assert obj["val_accs"] == [0.81, 0.82, 0.79, 0.83, 0.80]


# ─── heartbeat ───────────────────────────────────────────────────────────────


class TestHeartbeat:
    def test_writes_heartbeat_file(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        rw.heartbeat({"current_cell": "n=4", "epoch": 50})
        assert (tmp_path / "heartbeat.json").exists()

    def test_heartbeat_overwrites_previous(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        rw.heartbeat({"epoch": 10})
        rw.heartbeat({"epoch": 20})
        obj = json.loads((tmp_path / "heartbeat.json").read_text())
        assert obj["epoch"] == 20

    def test_heartbeat_includes_pid_and_timestamp(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        rw.heartbeat({"x": 1})
        obj = json.loads((tmp_path / "heartbeat.json").read_text())
        assert obj["pid"] == os.getpid()
        assert "timestamp" in obj


# ─── checkpoints ─────────────────────────────────────────────────────────────


class TestCheckpoints:
    def test_save_and_load_round_trip(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        state = {"epoch": 100, "best_val_acc": 0.85,
                 "model_state": torch.zeros(3, 4)}
        rw.save_checkpoint("cell_xyz", state)
        loaded = rw.load_checkpoint("cell_xyz")
        assert loaded is not None
        assert loaded["epoch"] == 100
        assert torch.equal(loaded["model_state"], state["model_state"])

    def test_load_missing_returns_none(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        assert rw.load_checkpoint("does_not_exist") is None

    def test_clear_removes_checkpoint(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        rw.save_checkpoint("c", {"x": 1})
        assert rw.load_checkpoint("c") is not None
        rw.clear_checkpoint("c")
        assert rw.load_checkpoint("c") is None

    def test_clear_nonexistent_is_noop(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        rw.clear_checkpoint("never_existed")  # must not raise


# ─── get_completed_runs ──────────────────────────────────────────────────────


class TestCompletedRuns:
    def test_returns_run_ids_in_dir(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        rw.save_run({"run_id": "abc", "val": 0.8})
        rw.save_run({"run_id": "def", "val": 0.85})
        assert rw.get_completed_runs() == {"abc", "def"}

    def test_ignores_underscore_prefixed_files(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        rw.save_run({"run_id": "abc", "val": 0.8})
        # heartbeat and progress log shouldn't count
        rw.heartbeat({"x": 1})
        rw.log_event("e")
        completed = rw.get_completed_runs()
        assert "abc" in completed
        assert "heartbeat" not in completed
        assert "progress" not in completed

    def test_returns_empty_for_fresh_dir(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        assert rw.get_completed_runs() == set()


# ─── concurrent-safety (single writer, monotonic events) ─────────────────────


class TestProgressLogOrdering:
    def test_many_events_preserve_order(self, tmp_path):
        rw = ResultWriter(results_dir=str(tmp_path))
        for i in range(50):
            rw.log_event("tick", n=i)

        lines = (tmp_path / "progress.jsonl").read_text().strip().split("\n")
        n_seq = [json.loads(l)["n"] for l in lines]
        assert n_seq == list(range(50))
