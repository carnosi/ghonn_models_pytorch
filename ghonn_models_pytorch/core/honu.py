"""Vectorized Higher-Order Neural Unit."""

from __future__ import annotations

import math
from itertools import combinations_with_replacement
import torch
from torch import Tensor, nn
import torch.nn.functional as functional


class HONU(nn.Module):
    """Higher-Order Neural Unit for learned polynomial regression.

    HONU evaluates every monomial of ``order`` over the input features, including
    a constant feature when ``bias=True``, and projects those monomials through
    learned weights. The unit supports both scalar use and vectorized outputs:
    inputs shaped ``(..., in_features)`` produce outputs shaped
    ``(..., out_features)``.
    """

    def __init__(
        self,
        in_features: int,
        order: int,
        out_features: int = 1,
        *,
        bias: bool = True,
        activation: str = "identity",
        weight_init_mode: str = "random",
        weight_divisor: float = 100.0,
        monomial_chunk_size: int | None = None,
    ) -> None:
        """Initialize a vectorized polynomial neural unit.

        Args:
            in_features: Number of input features.
            order: Polynomial order. Must be positive.
            out_features: Number of independent polynomial outputs.
            bias: Include a constant feature in the polynomial basis.
            activation: Name of an activation in ``torch.nn.functional``.
            weight_init_mode: One of ``random``, ``zeros``, ``ones``, ``xavier``,
                ``kaiming_normal``, or ``kaiming_uniform``.
            weight_divisor: Divisor used by ``random`` initialization.
            monomial_chunk_size: Optional maximum number of monomials processed
                at once to reduce peak memory use.
        """
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive integers.")
        if order <= 0:
            raise ValueError("order must be a positive integer.")
        if weight_divisor <= 0:
            raise ValueError("weight_divisor must be greater than zero.")
        if monomial_chunk_size is not None and monomial_chunk_size <= 0:
            raise ValueError("monomial_chunk_size must be positive or None.")

        self.in_features = in_features
        self.order = order
        self.out_features = out_features
        self.bias = bias
        self.activation = activation.lower()
        self.weight_init_mode = weight_init_mode
        self.weight_divisor = float(weight_divisor)
        self.monomial_chunk_size = monomial_chunk_size

        combinations = torch.tensor(
            list(
                combinations_with_replacement(
                    range(in_features + int(bias)),
                    order,
                )
            ),
            dtype=torch.long,
        )
        self.register_buffer("comb_idx", combinations)
        self.num_combinations = combinations.size(0)
        self.weight = nn.Parameter(torch.empty(self.num_combinations, out_features))
        self._validate_activation()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the polynomial weights using the configured initializer."""
        if self.weight_init_mode == "random":
            nn.init.uniform_(self.weight, 0, 1)
            with torch.no_grad():
                self.weight.div_(self.weight_divisor)
        elif self.weight_init_mode == "zeros":
            nn.init.zeros_(self.weight)
        elif self.weight_init_mode == "ones":
            nn.init.ones_(self.weight)
        elif self.weight_init_mode == "xavier":
            limit = math.sqrt(6 / (self.in_features + self.num_combinations))
            nn.init.uniform_(self.weight, -limit, limit)
        elif self.weight_init_mode == "kaiming_normal":
            nn.init.normal_(self.weight, std=math.sqrt(2 / self.in_features))
        elif self.weight_init_mode == "kaiming_uniform":
            limit = math.sqrt(6 / self.in_features)
            nn.init.uniform_(self.weight, -limit, limit)
        else:
            raise ValueError(f"Unknown weight initialization mode: {self.weight_init_mode}")

    def _validate_activation(self) -> None:
        """Raise an error when the configured activation is unavailable."""
        if self.activation not in {"identity", "linear"} and not hasattr(
            functional, self.activation
        ):
            raise ValueError(f"Unknown activation function: {self.activation}")

    def _polynomial_features(self, x: Tensor) -> Tensor:
        """Build the monomial feature matrix for ``x``."""
        if self.bias:
            x = torch.cat((torch.ones_like(x[..., :1]), x), dim=-1)
        flat_x = x.reshape(-1, x.size(-1))
        chunks = (
            ((0, self.num_combinations),)
            if self.monomial_chunk_size is None
            else (
                (start, min(start + self.monomial_chunk_size, self.num_combinations))
                for start in range(0, self.num_combinations, self.monomial_chunk_size)
            )
        )
        features = []
        for start, end in chunks:
            indices = self.comb_idx[start:end]
            monomials = flat_x[:, indices[:, 0]].clone()
            for index in indices[:, 1:].unbind(dim=1):
                monomials.mul_(flat_x[:, index])
            features.append(monomials)
        return torch.cat(features, dim=-1).reshape(*x.shape[:-1], self.num_combinations)

    def forward(self, x: Tensor) -> Tensor:
        """Return polynomial outputs with shape ``(*x.shape[:-1], out_features)``."""
        if x.ndim < 2:
            raise ValueError(f"HONU expects at least 2 dimensions, got {tuple(x.shape)}.")
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Input shape mismatch: expected {self.in_features} features, got {x.size(-1)}."
            )
        features = self._polynomial_features(x)
        output = features @ self.weight
        if self.activation not in {"identity", "linear"}:
            output = getattr(functional, self.activation)(output)
        return output

    def extra_repr(self) -> str:
        """Describe the unit in the standard ``nn.Module`` representation."""
        return (
            f"in_features={self.in_features}, order={self.order}, "
            f"out_features={self.out_features}, bias={self.bias}, "
            f"activation={self.activation!r}"
        )
