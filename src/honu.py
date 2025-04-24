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

        # Initialize weights and bias as trainable parameters
        self._weights = nn.Parameter(self._initialize_weights())
        if self._bias:
            self._bias_param = nn.Parameter(torch.rand(1) / self._weight_divisor)

        self._num_combinations = self._weights.size(0)

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

    def _get_colx(self, x: Tensor) -> Tensor:
        """Compute the colX tensor for a batch of input tensors.

        Computes the colX tensor for a batch of input tensors `x` based on a specified polynomial order.
        This method generates a tensor where each element corresponds to the product of
        combinations (with replacement) of elements from the input tensor `x`. The number
        of combinations is determined by the polynomial order `r`.

        Args:
            x: Input tensor of shape (batch_size, input_length) for which the colX tensor is computed.

        Returns:
            Tensor: A tensor of shape (batch_size, num_combinations) containing the computed colX values.
        """
        batch_size, _ = x.size()

        # Initialize a tensor to store the results for the entire batch
        colx = torch.zeros(batch_size, self._num_combinations, device=x.device)

        # Generate combinations with replacement and compute the product for each sample in the batch
        for i in range(batch_size):
            for pos, com in enumerate(combinations_with_replacement(x[i].tolist(), self._order)):
                colx[i, pos] = torch.prod(torch.tensor(com, device=x.device))

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
        if self._bias:
            output += self._bias_param
        return output


if __name__ == "__main__":
    from pathlib import Path

    filename = Path(__file__).name
    MSG = f"The {filename} is not meant to be run as a script."
    raise OSError(MSG)
