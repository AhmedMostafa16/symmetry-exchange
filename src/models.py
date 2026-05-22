"""
Neural network architectures for the symmetry exchange rate experiment.

Five model families compare how correct inductive bias vs. augmentation
vs. regularization vs. orbit size affects sample complexity.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _build_mlp(in_dim: int, hidden_dim: int, n_hidden: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, 1))
    return nn.Sequential(*layers)


def _rotation_buffers(n: int, offset: float = 0.0) -> torch.Tensor:
    """Return (n, 2, 2) rotation matrices for angles k*2π/n + offset."""
    angles = torch.tensor(
        [2 * np.pi * k / n + offset for k in range(n)], dtype=torch.float32
    )
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    return torch.stack(
        [torch.stack([cos_a, -sin_a], dim=1), torch.stack([sin_a, cos_a], dim=1)],
        dim=2,
    )  # (n, 2, 2)


# ─── Correct equivariant model ────────────────────────────────────────────────


class CnEquivariantMLP(nn.Module):
    """
    C_n-invariant network via the regular representation.

    Generates all n rotated copies of the input, applies a shared MLP to each,
    then averages the scalar outputs (G-invariant pooling). Provably invariant
    under rotation by 2π/n by construction.

    Parameter budget: only the shared MLP is counted — orbit copies share weights.
    """

    def __init__(
        self,
        n: int,
        hidden_dim: int = 32,
        n_hidden: int = 2,
        rotation_offset: float = 0.0,
    ) -> None:
        """
        rotation_offset shifts all orbit angles by a global constant, producing a
        *coset* of C_n. The model remains C_n invariant regardless of offset value,
        because rotation by 2π/n still permutes the (shifted) orbit elements.
        Use WrongGroupEquivariantMLP to actually break invariance.
        """
        super().__init__()
        self.n = n
        self.shared_mlp = _build_mlp(2, hidden_dim, n_hidden)
        self.register_buffer("rotation_matrices", _rotation_buffers(n, rotation_offset))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2) → logits: (B, 1)"""
        B = x.shape[0]
        x_rot = torch.einsum("bi,nij->bnj", x, self.rotation_matrices)  # (B, n, 2)
        out = self.shared_mlp(x_rot.reshape(B * self.n, 2)).reshape(B, self.n)
        return out.mean(dim=1, keepdim=True)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.shared_mlp.parameters())


# ─── Wrong-group control ──────────────────────────────────────────────────────


class WrongGroupEquivariantMLP(CnEquivariantMLP):
    """
    Same orbit-pooling architecture as CnEquivariantMLP, but rotation angles
    are deliberately misaligned using an irrational multiple (0.7) of the
    correct fundamental period.

    This control isolates "correct alignment" from "orbit averaging of any kind."
    If this model matches the correct equivariant model's sample-complexity slope,
    the advantage is from regularization via orbit averaging, not structural prior.
    """

    MISALIGNMENT_FACTOR = 0.7

    def __init__(self, n: int, hidden_dim: int = 32, n_hidden: int = 2) -> None:
        super().__init__(n=n, hidden_dim=hidden_dim, n_hidden=n_hidden)
        # Override buffer with misaligned rotation matrices
        wrong_angles = torch.tensor(
            [2 * np.pi * self.MISALIGNMENT_FACTOR * k / n for k in range(n)],
            dtype=torch.float32,
        )
        cos_a, sin_a = torch.cos(wrong_angles), torch.sin(wrong_angles)
        wrong_rot = torch.stack(
            [torch.stack([cos_a, -sin_a], dim=1), torch.stack([sin_a, cos_a], dim=1)],
            dim=2,
        )
        # re-register to properly replace the buffer
        self.register_buffer("rotation_matrices", wrong_rot)


# ─── Vanilla baseline ─────────────────────────────────────────────────────────


class VanillaMLP(nn.Module):
    """Standard unconstrained MLP. Parameter count is matched to be ≥ equivariant's shared MLP."""

    def __init__(self, hidden_dim: int = 32, n_hidden: int = 2) -> None:
        super().__init__()
        self.net = _build_mlp(2, hidden_dim, n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── Augmented vanilla ────────────────────────────────────────────────────────


class AugmentedVanillaMLP(VanillaMLP):
    """
    VanillaMLP with online C_n augmentation during training.

    Separates "information injection via augmentation" from
    "architectural inductive bias via weight sharing." During training,
    `augmented_forward` averages logits over all n orbit copies.
    At evaluation time, standard `forward` is used (no augmentation).

    Fair adjusted-N comparison: multiply nominal N by n to account for the
    n× more samples processed per step.
    """

    def __init__(self, n: int, hidden_dim: int = 32, n_hidden: int = 2) -> None:
        super().__init__(hidden_dim=hidden_dim, n_hidden=n_hidden)
        self.n = n
        self.register_buffer("rotation_matrices", _rotation_buffers(n))

    def augmented_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training forward: average logits over all n orbit copies."""
        B = x.shape[0]
        x_rot = torch.einsum("bi,nij->bnj", x, self.rotation_matrices)
        out = self.net(x_rot.reshape(B * self.n, 2)).reshape(B, self.n)
        return out.mean(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Regularized baseline ─────────────────────────────────────────────────────


class RegularizedMLP(VanillaMLP):
    """
    VanillaMLP with L2 weight decay (passed as `weight_decay` to Adam).
    Tests whether DOF reduction alone explains any equivariant advantage (Failure E).
    Lambda is set via a brief pilot search and passed to the optimizer at train time.
    """


# ─── Parameter matching and model suite ──────────────────────────────────────


def _vanilla_param_count(hidden_dim: int, n_hidden: int) -> int:
    """Parameter count for VanillaMLP(hidden_dim, n_hidden)."""
    count = 3 * hidden_dim  # Linear(2, h): weights + bias
    for _ in range(n_hidden - 1):
        count += hidden_dim * hidden_dim + hidden_dim
    count += hidden_dim + 1  # output layer
    return count


def make_matched_vanilla_mlp(equivariant_param_count: int, n_hidden: int = 2) -> VanillaMLP:
    """
    Smallest VanillaMLP with parameter count ≥ equivariant_param_count.
    Vanilla having ≥ params makes any equivariant advantage a conservative lower bound.
    """
    for h in range(4, 1024):
        if _vanilla_param_count(h, n_hidden) >= equivariant_param_count:
            return VanillaMLP(hidden_dim=h, n_hidden=n_hidden)
    raise ValueError(f"Cannot match {equivariant_param_count} params within hidden_dim=1024")


def get_model_suite(
    n_group: int,
    hidden_dim: int = 32,
    n_hidden: int = 2,
) -> dict[str, nn.Module]:
    """
    Return all five model variants for a given group size.

    All use the same hidden_dim for the MLP component.
    Vanilla and regularized have ≥ params than the equivariant shared MLP.
    """
    equiv = CnEquivariantMLP(n=n_group, hidden_dim=hidden_dim, n_hidden=n_hidden)
    vanilla = make_matched_vanilla_mlp(equiv.count_parameters(), n_hidden=n_hidden)

    return {
        "equivariant": equiv,
        "wrong_group": WrongGroupEquivariantMLP(n=n_group, hidden_dim=hidden_dim, n_hidden=n_hidden),
        "augmented": AugmentedVanillaMLP(n=n_group, hidden_dim=hidden_dim, n_hidden=n_hidden),
        "vanilla": vanilla,
        "regularized": RegularizedMLP(
            hidden_dim=vanilla.net[0].out_features,  # type: ignore[attr-defined]
            n_hidden=n_hidden,
        ),
    }
