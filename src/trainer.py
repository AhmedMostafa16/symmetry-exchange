"""Training loop for the symmetry exchange rate experiment."""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import estimate_flops_per_forward, id_estimator_agreement, orbit_consistency_score


def train_one_run(
    model: nn.Module,
    model_type: str,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    n_group: int,
    n_train: int,
    epsilon: float,
    seed: int,
    lambda_l2: float = 0.0,
    target_acc: float = 0.80,
    max_epochs: int = 500,
    patience: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    checkpoint_dir: Optional[str] = None,
    config_hash: str = "",
    dataset_hash: str = "",
) -> dict:
    """
    Train one model configuration. Returns a result dict compatible with RunResult.

    Determinism: seeds all RNG sources before any random operation.
    Early stopping: triggered when val_acc ≥ target_acc for `patience` consecutive epochs.
    Checkpointing: every 50 epochs if checkpoint_dir is provided.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run_id = str(uuid.uuid4())[:12]
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_l2)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    val_acc_curve: list[tuple[int, float]] = []
    train_loss_curve: list[tuple[int, float]] = []
    best_val_acc = 0.0
    epochs_to_target: Optional[int] = None
    consecutive_above = 0
    flops_per_forward = estimate_flops_per_forward(model)

    start_time = time.perf_counter()

    for epoch in range(max_epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            if model_type == "augmented":
                logits = model.augmented_forward(x_b)  # type: ignore[attr-defined]
            else:
                logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        train_loss_curve.append((epoch, epoch_loss / max(len(train_loader), 1)))

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                preds = (torch.sigmoid(model(x_b)) > 0.5).float()
                correct += (preds == y_b).sum().item()
                total += len(y_b)

        val_acc = correct / total
        val_acc_curve.append((epoch, val_acc))
        best_val_acc = max(best_val_acc, val_acc)

        if val_acc >= target_acc:
            if epochs_to_target is None:
                epochs_to_target = epoch
            consecutive_above += 1
        else:
            consecutive_above = 0

        if consecutive_above >= patience:
            break

        if checkpoint_dir and epoch % 50 == 0:
            p = Path(checkpoint_dir) / f"{run_id}_ep{epoch}.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": val_acc, "run_id": run_id}, p)

    wall_clock = time.perf_counter() - start_time
    total_epochs = len(val_acc_curve)

    # ── Representation metrics ─────────────────────────────────────────────────
    model.eval()
    x_val_sample = val_dataset.tensors[0][:500].to(device)
    activations = _extract_penultimate(model, x_val_sample, device)

    id_metrics = id_estimator_agreement(activations.cpu().numpy())
    orbit_cons = orbit_consistency_score(model, x_val_sample[:200], n_group, device)

    # ── Anomaly detection ─────────────────────────────────────────────────────
    anomaly_flag = False
    anomaly_reason: Optional[str] = None
    last_loss = train_loss_curve[-1][1] if train_loss_curve else 0.0
    if np.isnan(last_loss):
        anomaly_flag, anomaly_reason = True, "loss_nan"
    elif best_val_acc < 0.52 and total_epochs == max_epochs:
        anomaly_flag, anomaly_reason = True, "stuck_at_chance"

    return {
        "run_id": run_id,
        "n_group": n_group,
        "log2_n_group": float(np.log2(max(n_group, 1))),
        "n_train": n_train,
        "epsilon": epsilon,
        "seed": seed,
        "model_type": model_type,
        "lambda_l2": lambda_l2,
        "best_val_acc": best_val_acc,
        "reached_target": best_val_acc >= target_acc,
        "epochs_to_target": epochs_to_target,
        "total_epochs": total_epochs,
        "wall_clock_seconds": wall_clock,
        "total_flops": flops_per_forward * n_train * total_epochs,
        "flops_per_forward": flops_per_forward,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "id_twonn": id_metrics.get("twonn"),
        "id_pr": id_metrics.get("pr"),
        "id_mle_k10": id_metrics.get("mle_k10"),
        "id_agreement_flag": id_metrics.get("agreement_flag"),
        "orbit_consistency": orbit_cons,
        "val_acc_curve": val_acc_curve,
        "train_loss_curve": train_loss_curve,
        "anomaly_flag": anomaly_flag,
        "anomaly_reason": anomaly_reason,
        "config_hash": config_hash,
        "dataset_hash": dataset_hash,
    }


def _extract_penultimate(model: nn.Module, x: torch.Tensor, device: str) -> torch.Tensor:
    """Extract activations from the last ReLU layer via a forward hook."""
    activations: dict[str, torch.Tensor] = {}

    last_relu: Optional[nn.Module] = None
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            last_relu = m

    if last_relu is None:
        return x.to(device)

    handle = last_relu.register_forward_hook(
        lambda _m, _inp, out: activations.update({"act": out.detach()})
    )
    model.eval()
    with torch.no_grad():
        model(x.to(device))
    handle.remove()
    return activations.get("act", x)
