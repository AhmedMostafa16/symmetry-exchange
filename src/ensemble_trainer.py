"""
Vmap-batched ensemble trainer: trains all N_seeds models for one
(model_type, n_group, n_train, epsilon) cell in a single parallel pass.

Strategy:
  - Build N_seeds independent models, each with its own seed-dependent init.
  - Stack their parameters via torch.func.stack_module_state.
  - Vmap forward/loss/grad over the seed dimension.
  - One Adam optimizer over stacked Parameters; manual .grad assignment per step.
  - Per-seed dataset (different label-noise realisations) stacked into one tensor.

Why this is faster than running 5 seeds sequentially:
  - Single kernel launch per layer instead of 5 (huge on tiny models).
  - GPU utilisation goes up because each "batch" is now N_seeds×B.
  - One epoch loop instead of 5; one validation pass instead of 5.

Trade-off:
  - All seeds run for the same number of epochs (until ALL hit patience).
  - For homogeneous convergence times this is near-optimal; if one seed
    converges much slower than the others, parallel time = slowest seed time.

Determinism: each seed's RNG is independent (own Generator), so per-seed
weights and per-seed shuffles match what train_one_run_fast would produce
when given the same seed.
"""
from __future__ import annotations

import time
import uuid
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, stack_module_state, vmap

from src.metrics import estimate_flops_per_forward


def train_seeds_in_parallel(
    model_factory: Callable[[], nn.Module],
    model_type: str,
    train_datasets: list,   # list of N_seeds TensorDatasets
    val_datasets: list,     # list of N_seeds TensorDatasets
    seeds: list[int],
    n_group: int,
    n_train: int,
    epsilon: float,
    lambda_l2: float = 0.0,
    target_acc: float = 0.80,
    max_epochs: int = 500,
    patience: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda",
    val_every: int = 5,
    use_amp: bool = True,
    config_hash: str = "",
    dataset_hashes: Optional[list[str]] = None,
) -> list[dict]:
    """
    Train N_seeds copies of model_factory() simultaneously via vmap.
    Returns a list of per-seed result dicts (one per seed).
    """
    assert len(train_datasets) == len(seeds) == len(val_datasets), \
        "Need one dataset and val_dataset per seed"
    n_seeds = len(seeds)
    dataset_hashes = dataset_hashes or [""] * n_seeds

    # ── Build N_seeds models, each with own seed-dependent init ──────────────
    models = []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        models.append(model_factory().to(device))

    base_model = model_factory().to(device)  # template for functional_call

    # Stack params and buffers across the seed dimension
    params, buffers = stack_module_state(models)
    # Make params leaf tensors that an optimizer can drive
    params = {k: nn.Parameter(v.detach().clone()) for k, v in params.items()}

    # ── Stack data ──────────────────────────────────────────────────────────
    x_train = torch.stack([ds.tensors[0].to(device) for ds in train_datasets])
    y_train = torch.stack([ds.tensors[1].to(device) for ds in train_datasets])
    x_val   = torch.stack([ds.tensors[0].to(device) for ds in val_datasets])
    y_val   = torch.stack([ds.tensors[1].to(device) for ds in val_datasets])

    # AMP
    use_amp = bool(use_amp and device.startswith("cuda")
                   and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    amp_ctx = (lambda: torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)) \
              if use_amp else (lambda: _NullCtx())

    # ── Build vmapped functions ──────────────────────────────────────────────
    # functional_call always invokes module.forward(); for AugmentedVanillaMLP
    # we replicate augmented_forward by hand inside loss_one using the
    # rotation_matrices buffer. At validation time we want the plain forward
    # (no augmentation), so forward_one calls module.forward() directly.
    is_augmented = model_type == "augmented"

    def forward_one(params, buffers, x):
        """Standard forward — used for validation."""
        return functional_call(base_model, (params, buffers), (x,))

    def loss_one(params, buffers, x, y):
        if is_augmented:
            rot = buffers["rotation_matrices"]    # (n, 2, 2)
            n = rot.shape[0]
            B = x.shape[0]
            x_rot = torch.einsum("bi,nij->bnj", x, rot).reshape(B * n, 2)
            logits = functional_call(base_model, (params, buffers), (x_rot,))
            logits = logits.reshape(B, n).mean(dim=1, keepdim=True)
        else:
            logits = functional_call(base_model, (params, buffers), (x,))
        return F.binary_cross_entropy_with_logits(logits, y)

    # vmap over seed dimension; params/buffers/x/y all stacked at dim 0
    vmapped_loss = vmap(loss_one, in_dims=(0, 0, 0, 0))
    vmapped_forward = vmap(forward_one, in_dims=(0, 0, 0))

    optimizer = torch.optim.Adam(list(params.values()), lr=lr,
                                  weight_decay=lambda_l2)

    # ── Per-seed bookkeeping ─────────────────────────────────────────────────
    val_acc_curves: list[list[tuple[int, float]]] = [[] for _ in range(n_seeds)]
    train_loss_curves: list[list[tuple[int, float]]] = [[] for _ in range(n_seeds)]
    best_val_accs = torch.zeros(n_seeds, device=device)
    epochs_to_target: list[Optional[int]] = [None] * n_seeds
    consecutive_above = torch.zeros(n_seeds, device=device, dtype=torch.int32)

    flops_per_forward = estimate_flops_per_forward(base_model)
    N = x_train.size(1)

    # Per-seed shuffle generator (CPU because torch.randperm with device gens
    # is awkward to stack; we generate CPU perms then move).
    cpu_gens = [torch.Generator().manual_seed(s) for s in seeds]

    start = time.perf_counter()

    for epoch in range(max_epochs):
        # Per-seed shuffle stacked into (n_seeds, N)
        perm = torch.stack([torch.randperm(N, generator=g) for g in cpu_gens]).to(device)

        epoch_loss = torch.zeros(n_seeds, device=device)
        n_batches = 0

        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            idx = perm[:, i:end]                                # (n_seeds, b)
            # Gather per-seed batches
            b = end - i
            idx_exp_x = idx.unsqueeze(-1).expand(-1, -1, 2)     # (n_seeds, b, 2)
            idx_exp_y = idx.unsqueeze(-1)                       # (n_seeds, b, 1)
            x_b = torch.gather(x_train, 1, idx_exp_x)
            y_b = torch.gather(y_train, 1, idx_exp_y)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                losses = vmapped_loss(params, buffers, x_b, y_b)  # (n_seeds,)
                total_loss = losses.sum()
            total_loss.backward()
            optimizer.step()

            epoch_loss += losses.detach()
            n_batches += 1

        # Log per-seed train loss (one sync per epoch)
        epoch_loss_np = (epoch_loss / max(n_batches, 1)).cpu().numpy()
        for s_idx in range(n_seeds):
            train_loss_curves[s_idx].append((epoch, float(epoch_loss_np[s_idx])))

        # ── Validation ───────────────────────────────────────────────────────
        should_val = (epoch % val_every == 0) or (epoch == max_epochs - 1)
        if should_val:
            with torch.no_grad(), amp_ctx():
                val_logits = vmapped_forward(params, buffers, x_val)  # (n_seeds, N_val, 1)
                preds = (val_logits > 0).float()
                val_accs = (preds == y_val).float().mean(dim=(1, 2))  # (n_seeds,)

            best_val_accs = torch.maximum(best_val_accs, val_accs)
            above = val_accs >= target_acc
            consecutive_above = torch.where(above, consecutive_above + 1,
                                             torch.zeros_like(consecutive_above))

            val_accs_np = val_accs.cpu().numpy()
            for s_idx in range(n_seeds):
                val_acc_curves[s_idx].append((epoch, float(val_accs_np[s_idx])))
                if val_accs_np[s_idx] >= target_acc and epochs_to_target[s_idx] is None:
                    epochs_to_target[s_idx] = epoch

            if torch.all(consecutive_above >= patience):
                break

    wall = time.perf_counter() - start
    total_epochs = len(train_loss_curves[0])
    best_val_accs_np = best_val_accs.cpu().numpy()

    # ── Construct per-seed result dicts ──────────────────────────────────────
    results = []
    for s_idx, seed in enumerate(seeds):
        last_loss = train_loss_curves[s_idx][-1][1] if train_loss_curves[s_idx] else float("nan")
        anomaly_flag = False
        anomaly_reason: Optional[str] = None
        if np.isnan(last_loss) or np.isinf(last_loss):
            anomaly_flag, anomaly_reason = True, "loss_nan"
        elif best_val_accs_np[s_idx] < 0.52 and total_epochs == max_epochs:
            anomaly_flag, anomaly_reason = True, "stuck_at_chance"

        results.append({
            "run_id": str(uuid.uuid4())[:12],
            "n_group": n_group,
            "log2_n_group": float(np.log2(max(n_group, 1))),
            "n_train": n_train,
            "epsilon": epsilon,
            "seed": seed,
            "model_type": model_type,
            "lambda_l2": lambda_l2,
            "best_val_acc": float(best_val_accs_np[s_idx]),
            "reached_target": bool(best_val_accs_np[s_idx] >= target_acc),
            "epochs_to_target": epochs_to_target[s_idx],
            "total_epochs": total_epochs,
            "wall_clock_seconds": wall / n_seeds,  # amortised
            "ensemble_wall_clock_seconds": wall,    # actual wall time of ensemble
            "total_flops": flops_per_forward * n_train * total_epochs,
            "flops_per_forward": flops_per_forward,
            "n_parameters": sum(p.numel() for p in base_model.parameters()),
            "id_twonn": None,
            "id_pr": None,
            "id_mle_k10": None,
            "id_agreement_flag": None,
            "orbit_consistency": None,
            "val_acc_curve": val_acc_curves[s_idx],
            "train_loss_curve": train_loss_curves[s_idx],
            "anomaly_flag": anomaly_flag,
            "anomaly_reason": anomaly_reason,
            "config_hash": config_hash,
            "dataset_hash": dataset_hashes[s_idx],
        })
    return results


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *args): return False
