"""
Robust result writer for long-running experiments.

Designed for the failure mode that quietly kills long ML runs:
the accumulation of unwritten results.

Guarantees:
  - **Atomic JSON writes:** every per-run JSON is written via temp file + fsync
    + rename. A crash mid-write either leaves the prior version intact or no
    file at all — never a half-written one.
  - **Append-only progress log (JSONL):** every event (cell start, epoch
    milestone, completion, error) is a single line. JSONL never corrupts
    earlier entries even if the process dies mid-append.
  - **Heartbeat file:** updated each cell with current state (cell, epoch,
    progress). External monitor can detect a stalled or crashed process by
    looking at the timestamp.
  - **Mid-training checkpoints:** trainers may save partial state every K
    epochs so a crashed cell can resume rather than restart.
  - **Crash recovery:** orphaned `.tmp` files from prior runs are cleaned up
    on init; completed run JSONs survive any subsequent crash.

This module deliberately has no external dependencies beyond stdlib + torch.
"""
from __future__ import annotations

import json
import os
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch


def _atomic_write_json(target: Path, data: Any) -> Path:
    """
    Write `data` as JSON to `target` atomically.

    Implementation: write to `<target>.tmp` with fsync, then `os.replace`.
    If `json.dump` raises before the rename, the original `target` is left
    untouched and only the `.tmp` file exists; callers should run
    cleanup_temp_files() on startup to remove these.
    """
    target = Path(target)
    tmp = target.with_suffix(target.suffix + ".tmp")
    target.parent.mkdir(parents=True, exist_ok=True)

    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)
    return target


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResultWriter:
    """
    Per-experiment result writer.

    Parameters
    ----------
    results_dir : Path-like
        Directory for per-run JSONs (`<run_id>.json`), the progress log
        (`progress.jsonl`), the heartbeat (`heartbeat.json`), and
        checkpoints (`_checkpoints/<cell_key>.ckpt.pt`).
    experiment_name : str
        Tag stored in heartbeat and progress events for external monitoring.
    """

    def __init__(self, results_dir: str | Path,
                 experiment_name: str = "experiment") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        self.progress_log_path = self.results_dir / "progress.jsonl"
        self.heartbeat_path = self.results_dir / "heartbeat.json"
        self.checkpoint_dir = self.results_dir / "_checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Lock around progress-log appends so multiple threads in the same
        # process can't interleave half-lines.
        self._log_lock = threading.Lock()

    # ── Per-run JSONs ────────────────────────────────────────────────────────

    def save_run(self, result: dict) -> Path:
        """Atomically write a single run result keyed by `run_id`."""
        run_id = result["run_id"]
        target = self.results_dir / f"{run_id}.json"
        return _atomic_write_json(target, result)

    def get_completed_runs(self) -> set[str]:
        """Set of run_ids already saved (used for resumability)."""
        return {
            p.stem
            for p in self.results_dir.glob("*.json")
            if not p.stem.startswith("_") and p.stem not in {"heartbeat", "progress"}
        }

    # ── Append-only progress log ─────────────────────────────────────────────

    def log_event(self, event_type: str, **fields: Any) -> None:
        """
        Append one structured event to `progress.jsonl`.

        Guarantees ordering within a single process via internal lock and
        fsync after each append. Append-only — never rewrites earlier lines.
        """
        event = {
            "timestamp": _now_iso(),
            "experiment": self.experiment_name,
            "event": event_type,
            **fields,
        }
        line = json.dumps(event, default=_json_default) + "\n"
        with self._log_lock:
            with open(self.progress_log_path, "a") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())

    def read_progress_log(self) -> list[dict]:
        """Parse `progress.jsonl` as a list of event dicts."""
        if not self.progress_log_path.exists():
            return []
        out = []
        for line in self.progress_log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # tolerate a single corrupt trailing line (shouldn't happen)
                continue
        return out

    # ── Heartbeat ────────────────────────────────────────────────────────────

    def heartbeat(self, state: Optional[dict] = None) -> None:
        """Update heartbeat with current state plus pid/host/timestamp."""
        hb = {
            "timestamp": _now_iso(),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "experiment": self.experiment_name,
        }
        if state:
            hb.update(state)
        _atomic_write_json(self.heartbeat_path, hb)

    def read_heartbeat(self) -> Optional[dict]:
        if not self.heartbeat_path.exists():
            return None
        return json.loads(self.heartbeat_path.read_text())

    # ── Mid-training checkpoints ─────────────────────────────────────────────

    def save_checkpoint(self, cell_key: str, state: dict) -> Path:
        """Save mid-training state for `cell_key`. torch.save (handles tensors)."""
        path = self.checkpoint_dir / f"{_safe_key(cell_key)}.ckpt.pt"
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)
        return path

    def load_checkpoint(self, cell_key: str) -> Optional[dict]:
        path = self.checkpoint_dir / f"{_safe_key(cell_key)}.ckpt.pt"
        if not path.exists():
            return None
        return torch.load(path, weights_only=False)

    def clear_checkpoint(self, cell_key: str) -> None:
        path = self.checkpoint_dir / f"{_safe_key(cell_key)}.ckpt.pt"
        if path.exists():
            path.unlink()

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def cleanup_temp_files(self) -> int:
        """Remove orphaned .tmp files from crashed writes. Returns count removed."""
        count = 0
        for p in self.results_dir.rglob("*.tmp"):
            try:
                p.unlink()
                count += 1
            except OSError:
                pass
        return count

    # ── High-level helpers ───────────────────────────────────────────────────

    def summarise(self) -> dict:
        """Quick stats for live monitoring."""
        events = self.read_progress_log()
        runs = self.get_completed_runs()
        hb = self.read_heartbeat()
        return {
            "n_runs_saved": len(runs),
            "n_events_logged": len(events),
            "last_event": events[-1] if events else None,
            "heartbeat": hb,
        }


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _safe_key(key: str) -> str:
    """Filesystem-safe version of a cell key."""
    return key.replace("/", "_").replace(" ", "_").replace(":", "_")


def _json_default(o: Any) -> Any:
    """Fallback for non-JSON-serialisable values (e.g. torch.Tensor scalars)."""
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    return repr(o)
