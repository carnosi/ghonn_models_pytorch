"""Module for the HONU model."""

import math
from itertools import combinations_with_replacement

import torch
from torch import Tensor, nn

__version__ = "0.0.1"


class HONU(nn.Module):
    """HONU model for polynomial regression."""

    def __init__(self, input_lenght: int, order: int, *, weight_divisor: int = 100, bias=True) -> None:
        """Initialize the HONU model.

        Args:
        input_lenght: Length of the input for which the required number of weights is calculated.
        order: Order of the HONU model, by default 1 = linear
        weight_divisor: Divisor for the randomly initialized weights, by default 100
        bias: Whether to include a bias term in the model, by default True
        """
        super().__init__()
        self._order = order
        self._input_length = input_lenght
        self._weight_divisor = weight_divisor
        self._bias = bias

        # Initialize weights as trainable parameters
        self._weights = nn.Parameter(self._initialize_weights())
        self._num_combinations = self._weights.size(0)

        # Get all combinations of indices for the polynomial features
        comb_idx = self._get_combinations()
        self.register_buffer("comb_idx", comb_idx)

    def _initialize_weights(self) -> Tensor:
        """Initialize weights for the HONU model.

        This method initializes the weights for the model based on the input length,
        polynomial order, and whether a bias term is included. The number of weights
        is calculated using the formula for combinations with repetition:
            Combinations with repetition = ((n + r - 1)! / (r! * (n - 1)!))
        where:
            - n is the number of states, calculated as the input length plus 1 if a bias is included.
            - r is the polynomial order of the neuron.

        Returns:
            Array of initialized weights.
        """
        # Calculate the number of weights needed based on the order and input length
        n_weights = self._input_length + 1 if self._bias else self._input_length
        num_weights = int(
            math.factorial(n_weights + self._order - 1)
            / (math.factorial(self._order) * math.factorial(n_weights - 1))
        )
        # Initialize weights randomly and scale them
        weights = torch.rand(num_weights) / self._weight_divisor
        return weights

    def _get_combinations(self) -> Tensor:
        """Precompute and return all index combinations for the given input length and order.

        This method generates combinations with replacement of indices based on the input
        length and polynomial order. If bias is included, an additional feature is accounted
        for in the combinations. The resulting combinations are stored as a tensor.

        Returns:
            Tensor: A tensor containing all index combinations with shape
            (num_combinations, order).
        """
        # Precompute all index combinations once and store as buffer
        n_feat = self._input_length + (1 if self._bias else 0)
        comb_idx = torch.tensor(
            list(combinations_with_replacement(range(n_feat), self._order)), dtype=torch.long
        )  # shape: (num_combinations, order)
        return comb_idx

    def _get_colx(self, x: Tensor) -> Tensor:
        """Compute polynomial feature map using precomputed index combinations.

        For each sample in the batch, generates all degree-`order` monomials
        (with replacement) of the input features. If `bias=True`, a constant
        term is prepended before forming combinations.

        Args:
            x (Tensor[B, input_length]):
                Input batch of shape (batch_size, input_length).

        Returns:
            Tensor[B, num_combinations]:
                Tensor of shape (batch_size, num_combinations) where each column
                is the product of one combination of input features.
        """
        # Add bias column if needed
        if self._bias:
            ones = torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype)
            x = torch.cat([ones, x], dim=1)  # now x.shape = [B, n_feat]

        # x_exp: [B, num_combinations, n_feat]
        x_exp = x.unsqueeze(1).expand(-1, self.comb_idx.size(0), -1)

        # idx:   [B, num_combinations, order]
        idx = self.comb_idx.unsqueeze(0).expand(x.size(0), -1, -1)

        # selected: [B, num_combinations, order]
        selected = x_exp.gather(2, idx)

        # colx: [B, num_combinations]
        colx = selected.prod(dim=2)

        return colx

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the HONU model.

        Args:
            x: Input tensor of shape (batch_size, input_length).

        Returns:
            Tensor: Output tensor of shape (batch_size, 1).
        """
        colx = self._get_colx(x)
        # Compute the colX tensor for the input
        output = torch.matmul(colx, self._weights.view(-1, 1))
        return output


if __name__ == "__main__":
    from pathlib import Path

    filename = Path(__file__).name
    MSG = f"The {filename} is not meant to be run as a script."
    raise OSError(MSG)
