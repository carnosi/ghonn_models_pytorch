"""Three-expert SA-GHONN forecasting model."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from .banks import GhonuBank
from .conv_ghonn_1d import ConvGhonn1D
from .convghonn import ConvGhonn
from .revin import RevIN


class _MovingAverageDecomposition(nn.Module):
    def __init__(self, trend_kernel_size: int, seasonal_kernel_size: int) -> None:
        super().__init__()
        if min(trend_kernel_size, seasonal_kernel_size) <= 0:
            msg = "Moving-average kernel sizes must be positive."
            raise ValueError(msg)
        self.trend_kernel_size = trend_kernel_size
        self.seasonal_kernel_size = seasonal_kernel_size

    @staticmethod
    def _average(x: Tensor, kernel_size: int) -> Tensor:
        if kernel_size == 1:
            return x
        left, right = (kernel_size - 1) // 2, kernel_size // 2
        return functional.avg_pool1d(
            functional.pad(x.transpose(1, 2), (left, right), mode="replicate"),
            kernel_size,
            stride=1,
        ).transpose(1, 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        trend = self._average(x, self.trend_kernel_size)
        detrended = x - trend
        seasonal = self._average(detrended, self.seasonal_kernel_size)
        return trend, seasonal, detrended - seasonal


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.encoding[:, : x.size(1)]


class _AttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.attention_norm(x + attended)
        return self.ffn_norm(x + self.ffn(x))


class _TemporalConvProjection(nn.Module):
    def __init__(
        self, input_length: int, in_features: int, out_features: int, *, bias: bool, **kwargs: Any
    ) -> None:
        super().__init__()
        self.conv = ConvGhonn1D(input_length, in_features, bias=bias, **kwargs)
        self.projection = nn.Linear(self.conv.out_features[1], out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(self.conv(x.transpose(1, 2)))


class SAGHONN(nn.Module):
    """SA-GHONN with the ETTh1 96/192/336/720 benchmark defaults."""

    def __init__(  # noqa: PLR0913
        self,
        input_length: int,
        in_features: int,
        out_features: int,
        *,
        d_model: int = 96,
        n_heads: int = 8,
        n_layers: int = 1,
        ffn_dim: int | None = 192,
        dropout: float = 0.05,
        head_dropout: float = 0.1,
        predicted_feature_indices: tuple[int, ...] = (0,),
        temporal_lookback: int = 3,
        temporal_padding_mode: str = "replicate",
        poly_dropout: float = 0.04,
        predictor_orders: int | tuple[int, ...] = (2, 1),
        gate_orders: int | tuple[int, ...] = (2, 1, 3),
        predictor_activations: str | tuple[str, ...] = ("tanh", "relu"),
        gate_activations: str | tuple[str, ...] = ("tanh", "sigmoid"),
        branch_types: tuple[str, str, str] = ("order1", "conv_ghonu", "ghonu"),
        conv_kernel_size: int = 9,
        conv_n_kernels: int = 16,
        trend_kernel_size: int = 96,
        seasonal_kernel_size: int = 3,
        decoder_hidden_dim: int | None = 576,
        force_gate_output_one: bool = False,
        bias: bool = True,
        weight_init_mode: str = "xavier",
        weight_divisor: float = 100.0,
        monomial_chunk_size: int | None = 512,
        use_rev_in: bool = True,
        rev_in_use_squared_eps_in_denorm: bool = True,
        rev_in_denorm_eps_override: float | None = None,
    ) -> None:
        super().__init__()
        ffn_dim = d_model * 4 if ffn_dim is None else ffn_dim
        decoder_hidden_dim = (
            max(d_model * 2, ffn_dim) if decoder_hidden_dim is None else decoder_hidden_dim
        )
        if (
            min(
                input_length,
                in_features,
                out_features,
                d_model,
                n_heads,
                n_layers,
                ffn_dim,
                decoder_hidden_dim,
            )
            <= 0
        ):
            msg = "Model dimensions must be positive."
            raise ValueError(msg)
        if d_model % n_heads:
            msg = "d_model must be divisible by n_heads."
            raise ValueError(msg)
        if len(branch_types) != 3:
            msg = "branch_types must specify trend, seasonal, and residual branches."
            raise ValueError(msg)
        if not predicted_feature_indices or any(
            index < 0 or index >= in_features for index in predicted_feature_indices
        ):
            msg = "predicted_feature_indices must reference input features."
            raise ValueError(msg)

        self.input_length = input_length
        self.in_features = in_features
        self.out_features = out_features
        self.predicted_feature_indices = predicted_feature_indices
        self.n_predicted_features = len(predicted_feature_indices)
        self.branch_types = branch_types
        self.rev_in = (
            RevIN(
                in_features,
                eps=1e-6,
                use_squared_eps_in_denorm=rev_in_use_squared_eps_in_denorm,
                denorm_eps_override=rev_in_denorm_eps_override,
            )
            if use_rev_in
            else None
        )
        self.decomposition = _MovingAverageDecomposition(trend_kernel_size, seasonal_kernel_size)
        common = {
            "predictor_orders": predictor_orders,
            "gate_orders": gate_orders,
            "predictor_activations": predictor_activations,
            "gate_activations": gate_activations,
            "force_gate_output_one": force_gate_output_one,
            "weight_init_mode": weight_init_mode,
            "weight_divisor": weight_divisor,
            "poly_dropout": poly_dropout,
            "monomial_chunk_size": monomial_chunk_size,
        }

        def build_expert(branch_type: str) -> nn.Module:
            if branch_type in {"conv_ghonu", "conv-ghonu", "conv"}:
                conv_common = {**common, "poly_dropout": 0.0}
                return _TemporalConvProjection(
                    input_length,
                    in_features,
                    d_model,
                    kernel_size=conv_kernel_size,
                    n_kernels=conv_n_kernels,
                    padding_mode=temporal_padding_mode,
                    bias=bias,
                    **conv_common,
                )
            if branch_type in {"order1", "honu1", "first_order"}:
                args = {
                    **common,
                    "predictor_orders": (1,),
                    "gate_orders": (1,),
                    "predictor_activations": ("identity",),
                    "gate_activations": ("sigmoid",),
                }
            elif branch_type == "ghonu":
                args = common
            else:
                msg = "branch_types must use 'order1', 'conv_ghonu', or 'ghonu'."
                raise ValueError(msg)
            if temporal_lookback:
                return ConvGhonn(
                    in_features,
                    d_model,
                    lookback=temporal_lookback,
                    padding_mode=temporal_padding_mode,
                    bias=bias,
                    **args,
                )
            return GhonuBank(in_features, d_model, bias=bias, **args)

        self.trend_expert, self.seasonal_expert, self.residual_expert = (
            build_expert(branch_type) for branch_type in branch_types
        )
        self.fusion_norm = nn.LayerNorm(d_model)
        self.positional_encoding = _PositionalEncoding(d_model, input_length * 3)
        self.dropout = nn.Dropout(dropout)
        self.attention_layers = nn.ModuleList(
            [_AttentionBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)]
        )
        self.decode_dropout = nn.Dropout(head_dropout)
        self.decode_norm = nn.LayerNorm(d_model * 8)
        self.head = nn.Sequential(
            nn.Linear(d_model * 8, decoder_hidden_dim, bias=bias),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(decoder_hidden_dim, out_features * self.n_predicted_features, bias=bias),
        )

    @staticmethod
    def _summary(tokens: Tensor) -> Tensor:
        return torch.cat((tokens.mean(dim=1), tokens[:, -1]), dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.input_length, self.in_features):
            msg = (
                f"Expected input shape (batch, {self.input_length}, {self.in_features}), "
                f"got {tuple(x.shape)}."
            )
            raise ValueError(msg)
        if self.rev_in is not None:
            x = self.rev_in(x, "norm")
        decoder_input = x
        trend, seasonal, residual = self.decomposition(x)
        tokens = torch.cat(
            (
                self.trend_expert(trend),
                self.seasonal_expert(seasonal),
                self.residual_expert(residual),
            ),
            dim=1,
        )
        tokens = self.dropout(self.positional_encoding(self.fusion_norm(tokens)))
        for layer in self.attention_layers:
            tokens = layer(tokens)
        trend_tokens, seasonal_tokens, residual_tokens = tokens.split(self.input_length, dim=1)
        representation = self.decode_norm(
            self.decode_dropout(
                torch.cat(
                    (
                        self._summary(trend_tokens),
                        self._summary(seasonal_tokens),
                        self._summary(residual_tokens),
                        self._summary(tokens),
                    ),
                    dim=-1,
                )
            )
        )
        output = self.head(representation).reshape(
            x.size(0), self.out_features, self.n_predicted_features
        )
        output = output + decoder_input[:, -1, list(self.predicted_feature_indices)].unsqueeze(1)
        if self.rev_in is None:
            return output
        full = output.new_zeros(x.size(0), self.out_features, self.in_features)
        full[:, :, list(self.predicted_feature_indices)] = output
        return self.rev_in(full, "denorm")[:, :, self.predicted_feature_indices]
