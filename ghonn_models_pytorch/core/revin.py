"""Reversible instance normalization for time-series tensors.

@inproceedings{kim2021reversible,
  title     = {Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift},
  author    = {Kim, Taesung and
               Kim, Jinhee and
               Tae, Yunwon and
               Park, Cheonbok and
               Choi, Jang-Ho and
               Choo, Jaegul},
  booktitle = {International Conference on Learning Representations},
  year      = {2021},
  url       = {https://openreview.net/forum?id=cGDAkQo1C0p}
}
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RevIN(nn.Module):
    """Normalize each sample across its temporal dimensions and reverse it later."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        *,
        affine: bool = True,
        use_squared_eps_in_denorm: bool = True,
        denorm_eps_override: float | None = None,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.use_squared_eps_in_denorm = use_squared_eps_in_denorm
        self.denorm_eps_override = denorm_eps_override
        self.mean: Tensor | None = None
        self.stdev: Tensor | None = None
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: Tensor, mode: str) -> Tensor:
        """Normalize or denormalize ``x`` using the most recent statistics."""
        if x.ndim < 3 or x.size(-1) != self.num_features:
            raise ValueError(
                f"Expected (..., {self.num_features}) input with at least 3 dimensions, "
                f"got {tuple(x.shape)}."
            )
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise ValueError("mode must be 'norm' or 'denorm'.")

    def _get_statistics(self, x: Tensor) -> None:
        dims = tuple(range(1, x.ndim - 1))
        self.mean = x.mean(dim=dims, keepdim=True).detach()
        self.stdev = (x.var(dim=dims, keepdim=True, unbiased=False) + self.eps).sqrt().detach()

    def _normalize(self, x: Tensor) -> Tensor:
        normalized = (x - self.mean) / self.stdev
        if self.affine:
            normalized = normalized * self.affine_weight + self.affine_bias
        return normalized

    def _denormalize(self, x: Tensor) -> Tensor:
        if self.mean is None or self.stdev is None:
            raise RuntimeError("RevIN must normalize input before denormalizing output.")
        if self.affine:
            if self.denorm_eps_override is not None:
                denorm_eps = self.denorm_eps_override
            elif self.use_squared_eps_in_denorm:
                denorm_eps = self.eps * self.eps
            else:
                denorm_eps = self.eps
            x = (x - self.affine_bias) / (self.affine_weight + denorm_eps)
        return x * self.stdev + self.mean
