"""
Measurement theory for the symmetry exchange rate experiment.

Primary metric: N_target (minimum training samples to reach T=0.80 accuracy).
Secondary metrics: intrinsic dimensionality of learned representations,
orbit consistency score, and FLOP accounting.

ID estimation uses scikit-dimension (TwoNN, MLE) rather than hand-rolled code.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ─── Primary metric: sample complexity ───────────────────────────────────────


def compute_n_target(
    val_acc_by_n: dict[int, list[float]],
    target_acc: float = 0.80,
    min_seeds_above: int = 3,
) -> Optional[int]:
    """
    N_target = min { N ∈ N_grid :
        |{s : ValAcc(N, s) >= target}| >= min_seeds_above }

    Returns None if the target is not reached at any N in the grid.

    Uses 3-of-5 seed majority rather than strict mean to reduce outlier sensitivity.
    """
    for N in sorted(val_acc_by_n):
        accs = np.asarray(val_acc_by_n[N])
        if (accs >= target_acc).sum() >= min_seeds_above:
            return N
    return None


# ─── Intrinsic dimensionality estimators ─────────────────────────────────────


def intrinsic_dim_twonn(activations: np.ndarray, fraction: float = 0.9) -> float:
    """
    TwoNN estimator (Facco et al. 2017) via scikit-dimension.

    Measures the ratio of 2nd-NN to 1st-NN distances to estimate local ID.
    The fraction argument follows the Facco et al. procedure for truncating
    the empirical CDF to avoid boundary effects.

    Known failure modes:
    - Overestimates ID when noise inflates local distances
    - Unreliable when N < 10 * exp(ID)
    """
    try:
        from skdim.id import TwoNN
    except ImportError:
        return _twonn_fallback(activations, fraction)

    N = len(activations)
    if N < 20:
        return float("nan")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = TwoNN()
            est.fit(activations)
        return float(est.dimension_)
    except Exception:
        return float("nan")


def _twonn_fallback(activations: np.ndarray, fraction: float) -> float:
    """Manual TwoNN when scikit-dimension is unavailable."""
    from sklearn.neighbors import NearestNeighbors

    N = len(activations)
    if N < 20:
        return float("nan")

    nbrs = NearestNeighbors(n_neighbors=3).fit(activations)
    dists, _ = nbrs.kneighbors(activations)
    r1, r2 = dists[:, 1], dists[:, 2]
    valid = r1 > 1e-10
    mu = r2[valid] / r1[valid]
    n_keep = int(len(mu) * fraction)
    mu_trunc = np.sort(mu)[:n_keep]
    return float(1.0 / np.mean(np.log(mu_trunc)))


def intrinsic_dim_participation_ratio(activations: np.ndarray) -> float:
    """
    Participation Ratio: PR = (Σλ_i)² / Σλ_i²  where λ_i are covariance eigenvalues.

    Measures effective number of dimensions with non-negligible variance.
    Range: [1, D]. Measures only linear dimensionality — use alongside TwoNN
    as a cross-check. Always center and standardize before calling.

    Known failure modes:
    - Misses nonlinear manifolds
    - Inflated by outliers
    """
    centered = activations - activations.mean(axis=0)
    cov = centered.T @ centered / len(activations)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return float("nan")
    return float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum())


def intrinsic_dim_mle_levina(
    activations: np.ndarray, k_values: list[int] | None = None
) -> dict[int, float]:
    """
    Levina-Bickel MLE estimator via scikit-dimension.

    Always returns the full k-sweep so k-sensitivity is visible.
    Never report a single k — the sweep characterises reliability.

    Known failure modes:
    - Strong k-dependence: report full sweep, not a single value
    - Biased downward for small k, upward for large k
    """
    if k_values is None:
        k_values = [5, 10, 20]

    try:
        from skdim.id import MLE
    except ImportError:
        return _mle_fallback(activations, k_values)

    results = {}
    for k in k_values:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est = MLE(k=k)
                est.fit(activations)
            results[k] = float(est.dimension_)
        except Exception:
            results[k] = float("nan")
    return results


def _mle_fallback(activations: np.ndarray, k_values: list[int]) -> dict[int, float]:
    """Manual Levina-Bickel MLE when scikit-dimension is unavailable."""
    from sklearn.neighbors import NearestNeighbors

    results = {}
    for k in k_values:
        try:
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(activations)
            dists, _ = nbrs.kneighbors(activations)
            log_d = np.log(dists[:, 1:k + 1])
            log_rk = log_d[:, -1:]
            log_rj = log_d[:, :-1]
            ratios = log_rk - log_rj
            id_per_point = (k - 1) / (ratios.sum(axis=1) + 1e-10)
            results[k] = float(np.median(id_per_point))
        except Exception:
            results[k] = float("nan")
    return results


def id_estimator_agreement(activations: np.ndarray) -> dict:
    """
    Run all three estimators and measure agreement.

    agreement_flag = True if all estimates are within 3× of each other.
    When agreement_flag is False, report ID as unreliable — do not treat
    any single estimator's value as primary evidence.
    """
    twonn  = intrinsic_dim_twonn(activations)
    pr     = intrinsic_dim_participation_ratio(activations)
    mle_d  = intrinsic_dim_mle_levina(activations, [10])
    mle10  = mle_d.get(10, float("nan"))

    estimates = [e for e in [twonn, pr, mle10] if np.isfinite(e)]

    if len(estimates) < 2:
        return {
            "twonn": twonn, "pr": pr, "mle_k10": mle10,
            "max_disagreement": float("nan"), "agreement_flag": False,
        }

    max_ratio = max(estimates) / (min(estimates) + 1e-10)
    return {
        "twonn": twonn,
        "pr": pr,
        "mle_k10": mle10,
        "max_disagreement": float(max_ratio),
        "agreement_flag": max_ratio < 3.0,
    }


def calibrate_id_estimators(
    known_dims: list[int] | None = None,
    ambient_dim: int = 32,
    N: int = 500,
    noise_std: float = 0.05,
    seed: int = 0,
) -> dict:
    """
    MUST be run before any ID measurements in the experiment.

    Embeds a known d-dim sphere in ambient_dim dimensions with Gaussian noise,
    then checks whether each estimator recovers d within 50% relative error.
    Defines the operating regime of each estimator for this configuration.
    """
    if known_dims is None:
        known_dims = [1, 2, 3, 5, 8]

    rng = np.random.default_rng(seed)
    calibration: dict = {}

    for d in known_dims:
        z = rng.standard_normal((N, d + 1))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        embedded = np.zeros((N, ambient_dim))
        embedded[:, : d + 1] = z
        Q, _ = np.linalg.qr(rng.standard_normal((ambient_dim, ambient_dim)))
        noisy = (embedded @ Q.T + rng.normal(0, noise_std, (N, ambient_dim))).astype(np.float32)

        twonn_est = intrinsic_dim_twonn(noisy)
        pr_est    = intrinsic_dim_participation_ratio(noisy)

        twonn_err = abs(twonn_est - d) / d if d > 0 else twonn_est
        pr_err    = abs(pr_est - d)    / d if d > 0 else pr_est

        calibration[d] = {
            "true_dim": d,
            "twonn": twonn_est,
            "pr": pr_est,
            "twonn_error": twonn_err,
            "pr_error": pr_err,
        }
        print(f"  d={d}: TwoNN={twonn_est:.2f}, PR={pr_est:.2f} (true={d})")

    twonn_errors = [v["twonn_error"] for v in calibration.values() if isinstance(v, dict)]
    calibration["twonn_reliable"] = float(np.mean(twonn_errors)) < 0.5
    return calibration


# ─── Orbit consistency score ─────────────────────────────────────────────────


def orbit_consistency_score(
    model: nn.Module,
    x: torch.Tensor,
    n: int,
    device: str = "cpu",
) -> float:
    """
    Measures output variation under C_n rotations.

    For a perfectly C_n invariant model: score ≈ 0.
    For a model with no symmetry: score = normalised orbit std.

    This is observational only — it shows whether the model has learned
    invariance, but does NOT establish causal relevance of the geometry.
    """
    angles = torch.tensor([2 * np.pi * k / n for k in range(n)], dtype=torch.float32)
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    rot = torch.stack(
        [torch.stack([cos_a, -sin_a], dim=1), torch.stack([sin_a, cos_a], dim=1)],
        dim=2,
    ).to(device)

    x = x.to(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        B = x.shape[0]
        x_rot   = torch.einsum("bi,nij->bnj", x, rot)
        x_flat  = x_rot.reshape(B * n, 2)
        logits  = model(x_flat).reshape(B, n)
        orbit_std  = logits.std(dim=1)
        global_std = logits.reshape(-1).std() + 1e-8
        score = (orbit_std / global_std).mean().item()

    return float(score)


# ─── FLOP accounting ─────────────────────────────────────────────────────────


def estimate_flops_per_forward(model: nn.Module) -> int:
    """
    Estimate FLOPs for one forward pass (multiply-add = 2 ops per weight).

    For equivariant models the orbit factor (model.n) is included, since
    the shared MLP is applied n times per input sample.
    """
    flops = sum(2 * m.in_features * m.out_features for m in model.modules()
                if isinstance(m, nn.Linear))
    if hasattr(model, "n"):
        flops *= model.n
    return flops
