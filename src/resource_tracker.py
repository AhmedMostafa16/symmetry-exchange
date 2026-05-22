"""Resource tracking and run result schema."""
from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RunResult:
    """Complete result schema — one instance per training run. Never delete run files."""

    # Identity
    run_id: str
    experiment_name: str
    timestamp: str
    config_hash: str

    # Experimental conditions
    n_group: int
    log2_n_group: float
    n_train: int
    epsilon: float
    seed: int
    model_type: str
    lambda_l2: float

    # Primary outcome
    best_val_acc: float
    reached_target: bool
    epochs_to_target: Optional[int]

    # Resources
    total_epochs: int
    wall_clock_seconds: float
    gpu_energy_joules: Optional[float]
    flops_per_forward: int
    total_flops: int
    peak_memory_mb: Optional[float]

    # Model details
    n_parameters: int
    n_effective_params: int

    # Representation metrics
    id_twonn: Optional[float]
    id_pr: Optional[float]
    id_mle_k10: Optional[float]
    id_agreement_flag: Optional[bool]
    orbit_consistency: Optional[float]

    # Convergence curves
    val_acc_curve: list = field(default_factory=list)
    train_loss_curve: list = field(default_factory=list)

    # Dataset
    dataset_hash: str = ""

    # Flags
    converged: bool = False
    anomaly_flag: bool = False
    anomaly_reason: Optional[str] = None


class ResourceTracker:
    """Tracks wall-clock time and GPU energy for a training run."""

    def __init__(self, results_dir: str = "results/runs") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._start_time: Optional[float] = None
        self._start_power: Optional[float] = None

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._start_power = self._read_gpu_power()

    def stop(self) -> dict:
        elapsed = time.perf_counter() - (self._start_time or time.perf_counter())
        end_power = self._read_gpu_power()
        energy_j: Optional[float] = None
        if self._start_power is not None and end_power is not None:
            avg_power = (self._start_power + end_power) / 2
            energy_j = avg_power * elapsed
        return {"wall_clock_seconds": elapsed, "gpu_energy_joules": energy_j}

    def save(self, result: RunResult) -> Path:
        fpath = self.results_dir / f"{result.run_id}.json"
        with open(fpath, "w") as f:
            json.dump(asdict(result), f, indent=2)
        return fpath

    def load_all(self) -> list[dict]:
        results = []
        for fpath in self.results_dir.glob("*.json"):
            with open(fpath) as f:
                results.append(json.load(f))
        return results

    @staticmethod
    def _read_gpu_power() -> Optional[float]:
        """Instantaneous GPU power draw in watts, or None if unavailable."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            return float(out.stdout.strip())
        except Exception:
            return None
