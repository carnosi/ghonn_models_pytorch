"""Reusable grouped GHONU banks."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import Tensor, nn

from .ghonu import GHONU
from .honu import HONU


def _as_tuple(value: int | str | tuple, name: str) -> tuple:
    values = (
        value
        if isinstance(value, tuple)
        else (value,)
        if isinstance(value, (int, str))
        else tuple(value)
    )
    if not values:
        raise ValueError(f"{name} must not be empty.")
    return values


def _group_indices(
    out_features: int,
    *configs: int | str | tuple[int | str, ...],
) -> OrderedDict[tuple, list[int]]:
    normalized = tuple(_as_tuple(config, "configuration") for config in configs)
    groups: OrderedDict[tuple, list[int]] = OrderedDict()
    for index in range(out_features):
        config = tuple(values[index % len(values)] for values in normalized)
        groups.setdefault(config, []).append(index)
    return groups


class HonuBank(nn.Module):
    """Group vectorized HONUs by their polynomial order and activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        orders: int | tuple[int, ...] = (1,),
        activations: str | tuple[str, ...] = ("identity",),
        bias: bool = True,
        weight_init_mode: str = "random",
        weight_divisor: float = 100.0,
        poly_dropout: float = 0.0,
        monomial_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive.")
        groups = _group_indices(out_features, orders, activations)
        self.in_features = in_features
        self.out_features = out_features
        self.groups = nn.ModuleList()
        self.indices: list[list[int]] = []
        for (order, activation), indices in groups.items():
            self.groups.append(
                HONU(
                    in_features,
                    order,
                    len(indices),
                    bias=bias,
                    activation=activation,
                    weight_init_mode=weight_init_mode,
                    weight_divisor=weight_divisor,
                    poly_dropout=poly_dropout,
                    monomial_chunk_size=monomial_chunk_size,
                )
            )
            self.indices.append(indices)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        flat_x = x.reshape(-1, shape[-1])
        output = None
        for indices, group in zip(self.indices, self.groups, strict=True):
            group_output = group(flat_x)
            if output is None:
                output = group_output.new_empty(flat_x.size(0), self.out_features)
            output[:, indices] = group_output.to(output)
        if output is None:
            msg = "HonuBank requires at least one group."
            raise RuntimeError(msg)
        return output.reshape(*shape[:-1], self.out_features)


class GhonuBank(nn.Module):
    """Group vectorized GHONUs by their polynomial and activation settings."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        predictor_orders: int | tuple[int, ...] = (1,),
        gate_orders: int | tuple[int, ...] = (1,),
        predictor_activations: str | tuple[str, ...] = ("identity",),
        gate_activations: str | tuple[str, ...] = ("sigmoid",),
        force_gate_output_one: bool = False,
        bias: bool = True,
        weight_init_mode: str = "random",
        weight_divisor: float = 100.0,
        poly_dropout: float = 0.0,
        monomial_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive.")
        configs = (
            _as_tuple(predictor_orders, "predictor_orders"),
            _as_tuple(gate_orders, "gate_orders"),
            _as_tuple(predictor_activations, "predictor_activations"),
            _as_tuple(gate_activations, "gate_activations"),
        )
        groups = _group_indices(out_features, *configs)

        self.in_features = in_features
        self.out_features = out_features
        self.groups = nn.ModuleList()
        self.indices: list[list[int]] = []
        for config, indices in groups.items():
            predictor_order, gate_order, predictor_activation, gate_activation = config
            self.groups.append(
                GHONU(
                    in_features,
                    len(indices),
                    predictor_order=predictor_order,
                    gate_order=gate_order,
                    predictor_activation=predictor_activation,
                    gate_activation=gate_activation,
                    force_gate_output_one=force_gate_output_one,
                    bias=bias,
                    weight_init_mode=weight_init_mode,
                    weight_divisor=weight_divisor,
                    poly_dropout=poly_dropout,
                    monomial_chunk_size=monomial_chunk_size,
                )
            )
            self.indices.append(indices)

    def forward(self, x: Tensor) -> Tensor:
        output, _, _ = self.forward_with_elements(x)
        return output

    def forward_with_elements(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return bank output plus predictor and gate outputs."""
        shape = x.shape
        flat_x = x.reshape(-1, shape[-1])
        outputs = predictors = gates = None
        for indices, group in zip(self.indices, self.groups, strict=True):
            group_output, group_predictor, group_gate = group.forward_with_elements(flat_x)
            if outputs is None:
                outputs = group_output.new_empty(flat_x.size(0), self.out_features)
                predictors = group_predictor.new_empty(flat_x.size(0), self.out_features)
                gates = group_gate.new_empty(flat_x.size(0), self.out_features)
            outputs[:, indices] = group_output.to(outputs)
            predictors[:, indices] = group_predictor.to(predictors)
            gates[:, indices] = group_gate.to(gates)
        if outputs is None or predictors is None or gates is None:
            msg = "GhonuBank requires at least one group."
            raise RuntimeError(msg)
        new_shape = (*shape[:-1], self.out_features)
        return (
            outputs.reshape(new_shape),
            predictors.reshape(new_shape),
            gates.reshape(new_shape),
        )
