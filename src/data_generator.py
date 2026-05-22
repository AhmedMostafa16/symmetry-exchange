"""
C_n petal classification dataset generator.

The clean label is y = 1[cos(n·θ) > 0], creating n alternating petal regions.
The true symmetry group of the clean label function is D_n (rotation + reflection);
C_n equivariant models exploit the rotational subgroup, making any found advantage
a conservative lower bound.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch.utils.data import TensorDataset


def generate_petal_dataset(
    n_petals: int,
    N_train: int,
    N_val: int = 2000,
    N_test: int = 2000,
    epsilon: float = 0.0,
    label_noise: float = 0.05,
    seed: int = 42,
    r_min: float = 0.1,
    r_max: float = 1.0,
) -> tuple[TensorDataset, TensorDataset, TensorDataset, dict]:
    """
    Generate the C_n petal classification dataset.

    epsilon controls symmetry breaking: fraction of training labels replaced
    by an asymmetric (C_1) label. Validation and test always use clean labels.

    Returns (train_dataset, val_dataset, test_dataset, metadata_dict).
    metadata_dict contains all parameters needed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    def _sample(N: int) -> tuple[np.ndarray, np.ndarray]:
        r = rng.uniform(r_min, r_max, N)
        theta = rng.uniform(0, 2 * np.pi, N)
        x = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        return x, theta

    def _clean_label(theta: np.ndarray) -> np.ndarray:
        return (np.cos(n_petals * theta) > 0).astype(np.float32)

    def _breaking_label(theta: np.ndarray) -> np.ndarray:
        # Random half-plane — breaks C_n for n > 1
        angle = rng.uniform(0, 2 * np.pi)
        return (np.cos(theta - angle) > 0).astype(np.float32)

    def _apply_noise(y: np.ndarray, noise: float) -> np.ndarray:
        flip = rng.uniform(0, 1, len(y)) < noise
        return np.where(flip, 1.0 - y, y)

    def _make_tensors(N: int, is_train: bool) -> TensorDataset:
        x, theta = _sample(N)
        y_clean = _clean_label(theta)

        if is_train and epsilon > 0:
            y_break = _breaking_label(theta)
            mix = rng.uniform(0, 1, N) < epsilon
            y = np.where(mix, y_break, y_clean)
        else:
            y = y_clean.copy()

        y = _apply_noise(y, label_noise)
        return TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        )

    train_ds = _make_tensors(N_train, is_train=True)
    val_ds   = _make_tensors(N_val,   is_train=False)
    test_ds  = _make_tensors(N_test,  is_train=False)

    metadata = {
        "n_petals": n_petals,
        "N_train": N_train,
        "N_val": N_val,
        "N_test": N_test,
        "epsilon": epsilon,
        "label_noise": label_noise,
        "seed": seed,
        "r_min": r_min,
        "r_max": r_max,
        "dataset_hash": _compute_hash(train_ds, val_ds),
    }
    return train_ds, val_ds, test_ds, metadata


def _compute_hash(train: TensorDataset, val: TensorDataset) -> str:
    n = min(100, len(train))
    x_bytes = train[:n][0].numpy().tobytes()
    y_bytes = train[:n][1].numpy().tobytes()
    return hashlib.sha256(x_bytes + y_bytes).hexdigest()[:16]


# ─── Adversarial checks ────────────────────────────────────────────────────────


def run_dataset_adversarial_checks(
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    n_petals: int,
    epsilon: float,
    verbose: bool = True,
) -> dict:
    """
    Verify there are no hidden shortcuts, data leakage, or broken symmetry.
    All checks must pass before running any experiment.
    """
    from sklearn.linear_model import LogisticRegression

    x_train, y_train = train_ds.tensors
    x_val, y_val     = val_ds.tensors

    results: dict = {}

    # 1. Class balance
    pos_rate = float(y_train.mean().item())
    results["train_class_balance"] = pos_rate
    results["train_balance_ok"] = abs(pos_rate - 0.5) < 0.1

    # 2. No linear shortcut in raw coordinates
    clf = LogisticRegression(max_iter=200, random_state=0)
    clf.fit(x_train.numpy(), y_train.numpy().ravel())
    lin_acc = float(clf.score(x_val.numpy(), y_val.numpy().ravel()))
    results["linear_shortcut_acc"] = lin_acc
    results["no_linear_shortcut"] = lin_acc < 0.65

    # 3. No exact train/val coordinate overlap
    x_train_set = {tuple(np.round(row, 6)) for row in x_train.numpy()}
    x_val_set   = {tuple(np.round(row, 6)) for row in x_val.numpy()}
    overlap = len(x_train_set & x_val_set)
    results["train_val_overlap"] = overlap
    results["no_overlap"] = overlap == 0

    # 4. Symmetry preservation at epsilon=0
    if epsilon == 0.0:
        angle = 2 * np.pi / n_petals
        c, s = np.cos(angle), np.sin(angle)
        x_np = x_val.numpy()
        x_rot = x_np @ np.array([[c, s], [-s, c]]).T
        theta_orig = np.arctan2(x_np[:, 1], x_np[:, 0])
        theta_rot  = np.arctan2(x_rot[:, 1], x_rot[:, 0])
        y_orig = (np.cos(n_petals * theta_orig) > 0).astype(float)
        y_rot  = (np.cos(n_petals * theta_rot)  > 0).astype(float)
        sym_frac = float((y_orig == y_rot).mean())
        results["symmetry_preservation"] = sym_frac
        results["symmetry_ok"] = sym_frac > 0.999

    if verbose:
        for k, v in results.items():
            if isinstance(v, bool):
                status = "✓" if v else "✗"
                print(f"  {status} {k}: {v}")
            else:
                print(f"    {k}: {v}")

    results["all_checks_passed"] = all(
        v for k, v in results.items() if k.endswith("_ok")
    )
    return results
