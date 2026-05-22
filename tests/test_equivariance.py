"""
Equivariance property tests — MUST pass before any experiment runs.
If any test here fails, experimental results are void.
"""
import numpy as np
import pytest
import torch

from src.models import (
    AugmentedVanillaMLP,
    CnEquivariantMLP,
    VanillaMLP,
    WrongGroupEquivariantMLP,
    get_model_suite,
)


def _rotation_matrix(theta: float) -> torch.Tensor:
    c, s = np.cos(theta), np.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)


class TestCnInvariance:
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 6, 8, 12])
    def test_generator_rotation_leaves_output_unchanged(self, n):
        torch.manual_seed(0)
        model = CnEquivariantMLP(n=n, hidden_dim=32, n_hidden=2)
        model.eval()
        x = torch.randn(100, 2)
        out_orig = model(x)

        R = _rotation_matrix(2 * np.pi / n)
        out_rot = model(x @ R.T)

        max_diff = (out_orig - out_rot).abs().max().item()
        assert max_diff < 1e-4, f"C_{n} generator invariance failed: diff={max_diff:.2e}"

    @pytest.mark.parametrize("n", [2, 4, 6, 8])
    def test_all_group_elements_leave_output_unchanged(self, n):
        torch.manual_seed(1)
        model = CnEquivariantMLP(n=n, hidden_dim=16)
        model.eval()
        x = torch.randn(50, 2)
        out_orig = model(x)

        for k in range(n):
            R = _rotation_matrix(2 * np.pi * k / n)
            out_rot = model(x @ R.T)
            max_diff = (out_orig - out_rot).abs().max().item()
            assert max_diff < 1e-4, f"C_{n} element k={k} failed: diff={max_diff:.2e}"

    def test_output_shape(self):
        model = CnEquivariantMLP(n=4, hidden_dim=32)
        x = torch.randn(16, 2)
        assert model(x).shape == (16, 1)

    def test_rotation_offset_preserves_invariance(self):
        """
        A rotation_offset shifts all orbit angles by a constant, creating a
        coset of C_n. Rotation by 2π/n still permutes the orbit elements,
        so the model REMAINS C_n invariant. The offset affects representation
        quality (misalignment with task), not the structural invariance property.
        Use WrongGroupEquivariantMLP (different fundamental period) to break invariance.
        """
        torch.manual_seed(2)
        model = CnEquivariantMLP(n=4, hidden_dim=32, rotation_offset=0.3)
        model.eval()
        x = torch.randn(50, 2)
        R = _rotation_matrix(2 * np.pi / 4)
        max_diff = (model(x) - model(x @ R.T)).abs().max().item()
        assert max_diff < 1e-4, (
            f"rotation_offset=0.3 creates a coset — should still be C_4 invariant, "
            f"got diff={max_diff:.2e}"
        )

    def test_n1_is_trivially_invariant(self):
        """n=1 has only the identity: any initialisation must be invariant."""
        torch.manual_seed(3)
        model = CnEquivariantMLP(n=1, hidden_dim=32)
        model.eval()
        x = torch.randn(20, 2)
        out = model(x)
        # Rotating by 2π/1 = 2π is the identity
        R = _rotation_matrix(2 * np.pi)
        out_rot = model(x @ R.T)
        assert (out - out_rot).abs().max().item() < 1e-4


class TestWrongGroupNonInvariance:
    @pytest.mark.parametrize("n", [3, 4, 6, 8])
    def test_wrong_group_is_not_cn_invariant(self, n):
        """WrongGroupEquivariantMLP must NOT be invariant under the correct C_n action."""
        torch.manual_seed(4)
        model = WrongGroupEquivariantMLP(n=n, hidden_dim=32)
        model.eval()
        x = torch.randn(100, 2)
        R = _rotation_matrix(2 * np.pi / n)
        max_diff = (model(x) - model(x @ R.T)).abs().max().item()
        assert max_diff > 0.01, (
            f"WrongGroup n={n} is accidentally C_n invariant (diff={max_diff:.4f}). "
            "Control is invalid."
        )


class TestParameterCounting:
    def test_equivariant_param_count_is_shared_mlp_only(self):
        model = CnEquivariantMLP(n=8, hidden_dim=32, n_hidden=2)
        n_params = model.count_parameters()
        # Shared MLP: L(2→32) + L(32→32) + L(32→1) = (2*32+32)+(32*32+32)+(32+1) = 96+1056+33 = 1185
        assert 100 < n_params < 2000

    def test_vanilla_has_at_least_as_many_params_as_equivariant(self):
        suite = get_model_suite(n_group=4, hidden_dim=32, n_hidden=2)
        equiv_p = suite["equivariant"].count_parameters()
        vanilla_p = suite["vanilla"].count_parameters()
        assert vanilla_p >= equiv_p, (
            f"Vanilla ({vanilla_p}) must have ≥ params than equivariant ({equiv_p})"
        )

    def test_all_model_types_present(self):
        suite = get_model_suite(n_group=4)
        expected = {"equivariant", "wrong_group", "augmented", "vanilla", "regularized"}
        assert set(suite.keys()) == expected

    def test_augmented_has_same_n(self):
        suite = get_model_suite(n_group=6, hidden_dim=16)
        assert suite["augmented"].n == 6


class TestAugmentedForward:
    def test_augmented_forward_shape(self):
        model = AugmentedVanillaMLP(n=4, hidden_dim=16)
        x = torch.randn(8, 2)
        out = model.augmented_forward(x)
        assert out.shape == (8, 1)

    def test_standard_forward_shape(self):
        model = AugmentedVanillaMLP(n=4, hidden_dim=16)
        x = torch.randn(8, 2)
        assert model(x).shape == (8, 1)
