"""Independent univariate Conv-GHONU filters."""

from __future__ import annotations

from torch import Tensor, nn

from .ghonu import GHONU


class ConvGhonn1D(nn.Module):
    """Apply ``n_kernels`` GHONU filters independently to each input feature."""

    def __init__(
        self,
        input_length: int,
        in_features: int,
        *,
        kernel_size: int = 25,
        n_kernels: int = 16,
        padding_mode: str = "replicate",
        predictor_orders: int | tuple[int, ...] = (1, 2),
        gate_orders: int | tuple[int, ...] = (1, 2, 3),
        predictor_activations: str | tuple[str, ...] = ("identity", "tanh"),
        gate_activations: str | tuple[str, ...] = ("sigmoid", "tanh"),
        force_gate_output_one: bool = False,
        bias: bool = True,
        weight_init_mode: str = "xavier",
        weight_divisor: float = 100.0,
        poly_dropout: float = 0.0,
        monomial_chunk_size: int | None = 512,
    ) -> None:
        """Initialize independent per-feature polynomial convolution filters."""
        super().__init__()
        if min(input_length, in_features, kernel_size, n_kernels) <= 0:
            msg = "Convolution dimensions must be positive."
            raise ValueError(msg)
        if padding_mode not in {"zeros", "reflect", "replicate", "circular", "valid"}:
            msg = f"Unsupported padding mode: {padding_mode}"
            raise ValueError(msg)

        def cycle(value: int | str | tuple, name: str) -> tuple:
            values = tuple(value) if isinstance(value, (list, tuple)) else (value,)
            if not values:
                msg = f"{name} must not be empty."
                raise ValueError(msg)
            return tuple(values[index % len(values)] for index in range(n_kernels))

        self.input_length = input_length
        self.in_features = in_features
        self.kernel_size = kernel_size
        self.n_kernels = n_kernels
        self.padding_mode = padding_mode
        self.predictor_orders = cycle(predictor_orders, "predictor_orders")
        self.gate_orders = cycle(gate_orders, "gate_orders")
        self.predictor_activations = cycle(predictor_activations, "predictor_activations")
        self.gate_activations = cycle(gate_activations, "gate_activations")
        padding = 0 if padding_mode == "valid" else kernel_size - 1
        self.left_padding = padding // 2
        self.right_padding = padding - self.left_padding
        output_length = input_length + padding - kernel_size + 1
        self.out_features = (output_length, in_features * n_kernels)
        self.filters = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        GHONU(
                            kernel_size,
                            predictor_order=self.predictor_orders[kernel],
                            gate_order=self.gate_orders[kernel],
                            predictor_activation=self.predictor_activations[kernel],
                            gate_activation=self.gate_activations[kernel],
                            force_gate_output_one=force_gate_output_one,
                            bias=bias,
                            weight_init_mode=weight_init_mode,
                            weight_divisor=weight_divisor,
                            poly_dropout=poly_dropout,
                            monomial_chunk_size=monomial_chunk_size,
                        )
                        for kernel in range(n_kernels)
                    ]
                )
                for _ in range(in_features)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return ``(batch, output_length, in_features * n_kernels)``."""
        if x.ndim != 3 or x.shape[1:] != (self.in_features, self.input_length):
            msg = (
                f"Expected input shape (batch, {self.in_features}, {self.input_length}), "
                f"got {tuple(x.shape)}."
            )
            raise ValueError(msg)
        if self.padding_mode == "zeros":
            x = nn.functional.pad(x, (self.left_padding, self.right_padding))
        elif self.padding_mode != "valid":
            x = nn.functional.pad(
                x, (self.left_padding, self.right_padding), mode=self.padding_mode
            )
        batch_size = x.size(0)
        output_length, output_channels = self.out_features
        output = None
        for feature, filters in enumerate(self.filters):
            windows = x[:, feature].unfold(1, self.kernel_size, 1).reshape(-1, self.kernel_size)
            for kernel, ghonu in enumerate(filters):
                filter_output = ghonu(windows).reshape(batch_size, output_length)
                if output is None:
                    output = filter_output.new_empty(batch_size, output_length, output_channels)
                output[:, :, feature * self.n_kernels + kernel] = filter_output.to(output)
        if output is None:
            msg = "ConvGhonn1D requires at least one filter."
            raise RuntimeError(msg)
        return output
