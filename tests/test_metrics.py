"""Tests for sample-complexity and representation metrics."""
import numpy as np
import pytest
import torch

from src.metrics import (
    compute_n_target,
    estimate_flops_per_forward,
    id_estimator_agreement,
    intrinsic_dim_participation_ratio,
    orbit_consistency_score,
)
from src.models import CnEquivariantMLP, VanillaMLP


class TestNTarget:
    def test_finds_minimum_n(self):
        acc_by_n = {
            100: [0.70, 0.71, 0.69, 0.72, 0.68],
            200: [0.78, 0.79, 0.77, 0.80, 0.79],
            400: [0.82, 0.83, 0.81, 0.84, 0.80],
        }
        assert compute_n_target(acc_by_n, target_acc=0.80, min_seeds_above=3) == 400

    def test_returns_smallest_satisfying_n(self):
        acc_by_n = {
            50:  [0.85, 0.86, 0.84, 0.85, 0.83],
            100: [0.90, 0.91, 0.90, 0.92, 0.91],
        }
        assert compute_n_target(acc_by_n, 0.80, 3) == 50

    def test_returns_none_when_never_reached(self):
        acc_by_n = {100: [0.70, 0.71], 200: [0.75, 0.76]}
        assert compute_n_target(acc_by_n, 0.80, 3) is None

    def test_min_seeds_threshold_exact(self):
        acc_by_n = {
            100: [0.81, 0.82, 0.79, 0.78, 0.77],  # 2/5 above 0.80
            200: [0.82, 0.83, 0.81, 0.80, 0.79],  # 4/5 above 0.80 (>=0.80)
        }
        # min_seeds_above=3: need at least 3 seeds to reach target
        assert compute_n_target(acc_by_n, 0.80, 3) == 200
        # min_seeds_above=2: 100 satisfies
        assert compute_n_target(acc_by_n, 0.80, 2) == 100

    def test_iterates_n_in_sorted_order(self):
        """N_target must be the *smallest* N even if dict is unordered."""
        acc_by_n = {
            800: [0.85, 0.86, 0.87],
            200: [0.82, 0.81, 0.83],
            50:  [0.60, 0.61, 0.62],
        }
        assert compute_n_target(acc_by_n, 0.80, 3) == 200


class TestIDEstimators:
    @pytest.fixture
    def sphere_activations(self):
        """Known 2-dim sphere embedded in 32-dim ambient space."""
        rng = np.random.default_rng(0)
        N, d_true, D = 400, 2, 32
        z = rng.standard_normal((N, d_true + 1))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        embedded = np.zeros((N, D))
        embedded[:, : d_true + 1] = z
        Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
        return (embedded @ Q.T + rng.normal(0, 0.02, (N, D))).astype(np.float32)

    def test_pr_on_known_dim(self, sphere_activations):
        pr = intrinsic_dim_participation_ratio(sphere_activations)
        # PR should be closer to 2-3 than to 32 for a 2-sphere
        assert 1.0 < pr < 15.0

    def test_agreement_output_keys(self, sphere_activations):
        result = id_estimator_agreement(sphere_activations)
        for key in ("twonn", "pr", "mle_k10", "max_disagreement", "agreement_flag"):
            assert key in result, f"Missing key: {key}"

    def test_agreement_values_are_finite_or_nan(self, sphere_activations):
        result = id_estimator_agreement(sphere_activations)
        for k in ("twonn", "pr", "mle_k10"):
            v = result[k]
            assert v is None or np.isfinite(v) or np.isnan(v)


class TestOrbitConsistency:
    def test_equivariant_has_low_consistency_score(self):
        """A perfectly invariant model produces identical outputs → std = 0 → score ≈ 0."""
        torch.manual_seed(0)
        model = CnEquivariantMLP(n=4, hidden_dim=16)
        model.eval()
        x = torch.randn(50, 2)
        score = orbit_consistency_score(model, x, n=4)
        assert score < 0.01, f"Expected near-zero for equivariant model, got {score:.4f}"

    def test_vanilla_has_nonzero_consistency_score(self):
        """A vanilla MLP has no symmetry → orbit variance > 0."""
        torch.manual_seed(7)
        model = VanillaMLP(hidden_dim=32)
        model.eval()
        x = torch.randn(50, 2)
        score = orbit_consistency_score(model, x, n=4)
        assert score > 0.0


class TestFLOPs:
    def test_equivariant_flops_scale_with_n(self):
        from src.models import CnEquivariantMLP
        m4 = CnEquivariantMLP(n=4, hidden_dim=32)
        m8 = CnEquivariantMLP(n=8, hidden_dim=32)
        f4 = estimate_flops_per_forward(m4)
        f8 = estimate_flops_per_forward(m8)
        assert f8 == pytest.approx(2 * f4), "FLOPs should double when n doubles"
