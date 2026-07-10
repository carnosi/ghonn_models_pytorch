"""Vectorized Gated Higher-Order Neural Unit."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .honu import HONU


class GHONU(nn.Module):
    """Gated Higher-Order Neural Unit.

    GHONU multiplies a predictor HONU by a second HONU used as a learned gate.
    It remains useful as a standalone unit and supports vectorized outputs for
    inputs shaped ``(..., in_features)``. Setting ``gate_order=0`` omits the
    gate module and makes this unit behave exactly like its predictor HONU.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        *,
        predictor_order: int = 1,
        gate_order: int = 1,
        predictor_activation: str = "identity",
        gate_activation: str = "sigmoid",
        bias: bool = True,
        weight_init_mode: str = "random",
        weight_divisor: float = 100.0,
        monomial_chunk_size: int | None = None,
    ) -> None:
        """Initialize a gated polynomial neural unit.

        Args:
            in_features: Number of input features.
            out_features: Number of independent predictor and gate outputs.
            predictor_order: Polynomial order of the predictor.
            gate_order: Polynomial order of the gate, or ``0`` to disable it.
            predictor_activation: Activation applied to predictor outputs.
            gate_activation: Activation applied to gate outputs.
            bias: Include a constant feature in both polynomial bases.
            weight_init_mode: Weight initialization mode passed to both HONUs.
            weight_divisor: Random initialization divisor passed to both HONUs.
            monomial_chunk_size: Optional monomial chunk size passed to both HONUs.
        """
        super().__init__()
        if gate_order < 0:
            raise ValueError("gate_order must be non-negative.")
        self.in_features = in_features
        self.out_features = out_features
        self.predictor_order = predictor_order
        self.gate_order = gate_order
        self.predictor = HONU(
            in_features,
            predictor_order,
            out_features,
            bias=bias,
            activation=predictor_activation,
            weight_init_mode=weight_init_mode,
            weight_divisor=weight_divisor,
            monomial_chunk_size=monomial_chunk_size,
        )
        self.gate = (
            HONU(
                in_features,
                gate_order,
                out_features,
                bias=bias,
                activation=gate_activation,
                weight_init_mode=weight_init_mode,
                weight_divisor=weight_divisor,
                monomial_chunk_size=monomial_chunk_size,
            )
            if gate_order > 0
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return the predictor output, optionally modulated by the gate."""
        predictor = self.predictor(x)
        return predictor if self.gate is None else predictor * self.gate(x)

    def forward_with_elements(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return output, predictor output, and gate output separately."""
        predictor = self.predictor(x)
        gate = torch.ones_like(predictor) if self.gate is None else self.gate(x)
        return predictor * gate, predictor, gate

    def extra_repr(self) -> str:
        """Describe the unit in the standard ``nn.Module`` representation."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"predictor_order={self.predictor_order}, gate_order={self.gate_order})"
        )
