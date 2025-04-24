"""Module for the HONN model."""

from itertools import cycle, islice
from warnings import warn

import torch
from torch import Tensor, nn

from .honu import HONU

__version__ = "0.0.1"


class HONN(nn.Module):
    """Model for network out of HONU."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_size: int,
        polynomial_orders: list[int],
        *,
        output_type: str = "sum",
        **kwargs,
    ) -> None:
        """Initialize the HONN model.

        Args:
            in_features (int): Number of input features for the model.
            out_features (int): Number of output features for the model.
            layer_size (int): Number of neurons in each layer.
            polynomial_orders (list[int]): List specifying the polynomial order for each layer.
            output_type (str): Type of output. Can be "sum", "linear" or "raw.
                    - "sum": Sum the outputs of all units (default).
                    - "linear": Apply a linear transformation to the concatenated outputs.
                    - "raw": Return the raw outputs of all units without any transformation.
            **kwargs: Additional keyword arguments passed to the HONUs.
        """
        super().__init__()
        # Main model parameters
        self.in_features = in_features
        self.out_features = out_features
        self.layer_size = layer_size
        self.polynomial_orders = self._assign_polynomial_orders(polynomial_orders)

        # Optional params
        self.output_type = output_type

        # Initialize honu neurons
        self.honu = nn.ModuleList(
            [HONU(in_features, order, **kwargs) for order in self.polynomial_orders]
        )
        # Initialize output
        self.head = self._get_head()

    def __repr__(self) -> str:
        """Return a string representation of the HONN model."""
        cls = self.__class__.__name__
        # describe head
        if self.output_type == "sum":
            head_desc = "SummedHonuOutputs"
        elif self.output_type == "linear":
            head_desc = repr(self.head)
        else:  # raw
            head_desc = "RawHonuOutputs"
        lines = [
            f"{cls}(",
            f"  in_features={self.in_features},",
            f"  out_features={self.out_features},",
            f"  layer_size={self.layer_size},",
            f"  polynomial_orders={self.polynomial_orders},",
            f"  output_type='{self.output_type}',",
            f"  head={head_desc},",
            "  honu=[",
        ]
        for idx, layer in enumerate(self.honu):
            lines.append(f"    [{idx}]={layer!r},")
        lines += [
            "  ]",
            ")",
        ]
        return "\n".join(lines)

    def _assign_polynomial_orders(self, polynomial_orders: list[int]) -> list[int]:
        """Inflate or truncate the provided polynomial orders list so its length == layer_size."""
        n = len(polynomial_orders)
        if n == self.layer_size:
            return polynomial_orders.copy()
        if n < self.layer_size:
            # cycle through the original entries until we reach layer_size
            return list(islice(cycle(polynomial_orders), self.layer_size))
        # if too many orders, just truncate
        warn(
            f"Too many polynomial orders ({n}). Truncating to {self.layer_size} orders.",
            stacklevel=2,
        )
        return polynomial_orders[: self.layer_size]

    def _get_head(self) -> callable:
        """Get the output head based on the specified output type."""
        if self.output_type == "sum":
            return lambda x: x.sum(dim=-1)
        if self.output_type == "linear":
            return nn.Linear(self.layer_size, self.out_features)
        if self.output_type == "raw":
            return lambda x: x

        msg = f"Invalid output type: {self.output_type}"
        raise ValueError(msg)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the HONN model.

        Args:
            x: Input tensor of shape (batch_size, in_features).

        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        output = torch.stack([self.honu[i](x) for i in range(self.layer_size)], dim=-1)

        # Apply the output head
        if self.output_type == "linear":
            output = self.head(output.view(x.size(0), -1))
        elif self.output_type == "raw":
            output = output.view(x.size(0), -1)
        else:
            output = self.head(output)

        return output


if __name__ == "__main__":
    from pathlib import Path

    filename = Path(__file__).name
    MSG = f"The {filename} is not meant to be run as a script."
    raise OSError(MSG)
