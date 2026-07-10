"""Convolutional GHONN temporal-window model."""

from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor, nn

from .banks import GhonuBank


class ConvGhonn(nn.Module):
    """Apply GHONU polynomial filters to causal temporal windows."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        lookback: int = 3,
        padding_mode: str = "replicate",
        predictor_orders: int | tuple[int, ...] = (1,),
        gate_orders: int | tuple[int, ...] = (1,),
        predictor_activations: str | tuple[str, ...] = ("identity",),
        gate_activations: str | tuple[str, ...] = ("sigmoid",),
        bias: bool = True,
        weight_init_mode: str = "random",
        weight_divisor: float = 100.0,
        monomial_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive.")
        if lookback < 0:
            raise ValueError("lookback must be non-negative.")
        self.in_features = in_features
        self.out_features = out_features
        self.lookback = lookback
        self.padding_mode = padding_mode
        self.bank = GhonuBank(
            in_features * (lookback + 1),
            out_features,
            predictor_orders=predictor_orders,
            gate_orders=gate_orders,
            predictor_activations=predictor_activations,
            gate_activations=gate_activations,
            bias=bias,
            weight_init_mode=weight_init_mode,
            weight_divisor=weight_divisor,
            monomial_chunk_size=monomial_chunk_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return one GHONU output for every input timestep."""
        if x.ndim != 3 or x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected input shape (batch, length, {self.in_features}), got {tuple(x.shape)}."
            )
        if self.lookback == 0:
            return self.bank(x)
        if self.padding_mode == "zeros":
            x = functional.pad(x, (0, 0, self.lookback, 0), mode="constant", value=0)
        else:
            x = functional.pad(x, (0, 0, self.lookback, 0), mode=self.padding_mode)
        windows = x.unfold(1, self.lookback + 1, 1)
        windows = windows.permute(0, 1, 3, 2).reshape(
            x.size(0), -1, self.in_features * (self.lookback + 1)
        )
        return self.bank(windows)
