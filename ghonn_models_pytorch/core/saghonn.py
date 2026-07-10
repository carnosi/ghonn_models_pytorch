"""Self-attention Gated Higher-Order Neural Network forecasting model."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .banks import GhonuBank
from .convghonn import ConvGhonn
from .revin import RevIN


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.encoding[:, : x.size(1)]


class SAGHONN(nn.Module):
    """Self-attention Gated Higher-Order Neural Network forecaster."""

    def __init__(
        self,
        input_length: int,
        in_features: int,
        out_features: int,
        *,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        predicted_feature_indices: tuple[int, ...] = (0,),
        temporal_lookback: int = 0,
        temporal_padding_mode: str = "replicate",
        aux_all_features: bool = False,
        head_type: str = "flat",
        head_hidden: int = 256,
        head_dropout: float = 0.1,
        predictor_orders: int | tuple[int, ...] = (1, 2),
        gate_orders: int | tuple[int, ...] = (1, 2),
        predictor_activations: str | tuple[str, ...] = ("identity", "tanh"),
        gate_activations: str | tuple[str, ...] = ("sigmoid", "tanh"),
        bias: bool = True,
        weight_init_mode: str = "xavier",
        weight_divisor: float = 100.0,
        monomial_chunk_size: int | None = None,
        use_rev_in: bool = True,
    ) -> None:
        super().__init__()
        if min(input_length, in_features, out_features, d_model, n_heads, n_layers) <= 0:
            raise ValueError("Model dimensions must be positive.")
        if temporal_lookback < 0:
            raise ValueError("temporal_lookback must be non-negative.")
        if head_type not in {"flat", "pool", "query"}:
            raise ValueError("head_type must be 'flat', 'pool', or 'query'.")
        if len(set(predicted_feature_indices)) != len(predicted_feature_indices):
            raise ValueError("predicted_feature_indices must not contain duplicates.")
        if not predicted_feature_indices or any(
            index < 0 or index >= in_features for index in predicted_feature_indices
        ):
            raise ValueError("predicted_feature_indices must reference input features.")

        self.input_length = input_length
        self.in_features = in_features
        self.out_features = out_features
        self.predicted_feature_indices = predicted_feature_indices
        self.use_rev_in = use_rev_in
        self.head_type = head_type
        n_predicted = len(predicted_feature_indices)
        self.rev_in = RevIN(in_features) if use_rev_in else None
        bank_args = {
            "out_features": d_model,
            "predictor_orders": predictor_orders,
            "gate_orders": gate_orders,
            "predictor_activations": predictor_activations,
            "gate_activations": gate_activations,
            "bias": bias,
            "weight_init_mode": weight_init_mode,
            "weight_divisor": weight_divisor,
            "monomial_chunk_size": monomial_chunk_size,
        }
        if temporal_lookback:
            self.embedding = ConvGhonn(
                in_features,
                lookback=temporal_lookback,
                padding_mode=temporal_padding_mode,
                **bank_args,
            )
        else:
            self.embedding = GhonuBank(in_features, **bank_args)

        self.positional_encoding = _PositionalEncoding(d_model, input_length)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        if head_type == "flat":
            head_input = input_length * d_model
            self.head = nn.Linear(head_input, out_features * n_predicted, bias=bias)
            self.aux_head = (
                nn.Linear(head_input, out_features * in_features, bias=bias)
                if aux_all_features
                else None
            )
        elif head_type == "pool":
            self.head = self._make_mlp(d_model, head_hidden, out_features * n_predicted, bias, head_dropout)
            self.aux_head = (
                self._make_mlp(d_model, head_hidden, out_features * in_features, bias, head_dropout)
                if aux_all_features
                else None
            )
        else:
            self.query_embed = nn.Parameter(torch.randn(out_features, d_model) * 0.02)
            self.cross_attention = nn.MultiheadAttention(
                d_model, n_heads, dropout=head_dropout, batch_first=True
            )
            self.query_norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, n_predicted, bias=bias)
            self.aux_head = nn.Linear(d_model, in_features, bias=bias) if aux_all_features else None

    @staticmethod
    def _make_mlp(
        input_size: int, hidden_size: int, output_size: int, bias: bool, dropout: float
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size, bias=bias),
        )

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[1:] != (self.input_length, self.in_features):
            raise ValueError(
                f"Expected input shape (batch, {self.input_length}, {self.in_features}), "
                f"got {tuple(x.shape)}."
            )
        if self.rev_in is not None:
            x = self.rev_in(x, "norm")
        encoded = self.encoder(self.dropout(self.positional_encoding(self.embedding(x))))

        if self.head_type == "flat":
            representation = encoded.flatten(1)
            output = self.head(representation).reshape(
                x.size(0), self.out_features, -1
            )
            auxiliary = (
                self.aux_head(representation).reshape(x.size(0), self.out_features, self.in_features)
                if self.aux_head is not None
                else None
            )
        elif self.head_type == "pool":
            representation = encoded.mean(dim=1)
            output = self.head(representation).reshape(x.size(0), self.out_features, -1)
            auxiliary = (
                self.aux_head(representation).reshape(x.size(0), self.out_features, self.in_features)
                if self.aux_head is not None
                else None
            )
        else:
            queries = self.query_embed.unsqueeze(0).expand(x.size(0), -1, -1)
            attended, _ = self.cross_attention(queries, encoded, encoded, need_weights=False)
            attended = self.query_norm(attended + queries)
            output = self.head(attended)
            auxiliary = self.aux_head(attended) if self.aux_head is not None else None

        if self.rev_in is not None:
            output = self._denormalize_selected(output)
            if auxiliary is not None:
                auxiliary = self.rev_in(auxiliary, "denorm")
        return (output, auxiliary) if auxiliary is not None else output

    def _denormalize_selected(self, output: Tensor) -> Tensor:
        full = output.new_zeros(output.size(0), self.out_features, self.in_features)
        full[:, :, list(self.predicted_feature_indices)] = output
        full = self.rev_in(full, "denorm")
        return full[:, :, list(self.predicted_feature_indices)]
