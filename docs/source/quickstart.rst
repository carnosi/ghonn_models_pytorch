.. _quickstart:

Quick Start 🚀
================
1. Create polynomial neurons
    Polynomial neurons, such as HONU and GHONU, are implemented as PyTorch nn.Module classes. You can easily create them as follows:

    .. code-block:: python

        import ghonn_models_pytorch as gmp

        kwargs = {
            "weight_divisor": 100,  # Divides weights to help with numerical stability
            "bias": True            # Whether to use a bias term in the model
        }

        # Create a Higher Order Neural Unit (HONU) with 3 inputs and degree 2
        honu_neuron = gmp.HONU(
            in_features=3,          # Number of input features
            order=2,                # Degree of the polynomial
            activation="identity",  # Activation function
            **kwargs
        )

        # Create a Gater Higher Order Neural Unit (GHONU) with 3 inputs, predictor order 2 and gate order 3
        ghonu_neuron = gmp.GHONU(
            in_features=3,                      # Number of input features
            predictor_order=2,                  # Degree of the predictor polynomial
            gate_order=3,                       # Degree of the gate polynomial
            predictor_activation="identity",    # Predictor activation function
            gate_activation="sigmoid",          # Gate activation function
            **kwargs
        )

2. Create a network layer
    You can combine polynomial neurons into a network layer, which is also a PyTorch nn.Module. Creating a network layer is straight forward:

    .. code-block:: python

        import ghonn_models_pytorch as gmp

        kwargs = {
            "weight_divisor": 100,
            "bias": True
        }

        # Create single HONU based layer - HONN with 4 neurons of different orders and activation functions.
        honn_layer = gmp.HONN(
            input_shape=3,                          # Number of input features
            output_shape=2,                         # Number of output features
            layer_size=4,                           # Number of neurons in the layer
            orders=(2, 3)                           # Degree of the polynomials in the layer. If shorter than layer size it works as rolling buffer
            activations=("identity", "sigmoid"),    # Activation functions for the neurons in the layer. If shorter work like a rolling buffer
            output_type="linear",                   # Output type of the layer. Can be "linear" or "sum" or "raw"
            **kwargs
        )

        # Create single GHONU based layer - GHONN with 4 neurons of different orders and activation functions.
        ghonn_layer = gmp.GHONN(
            input_shape=3,                              # Number of input features
            output_shape=2,                             # Number of output features
            layer_size=4,                               # Number of neurons in the layer
            predictor_orders=(2, 3),                    # Degree of the predictor polynomials in the layer. If shorter than layer size it works as rolling buffer
            gate_orders=(3, 5),                         # Degree of the gate polynomials in the layer. If shorter than layer size it works as rolling buffer
            predictor_activations="identity",           # Activation functions for the predictor neurons in the layer. If shorter work like a rolling buffer
            gate_activations=("identity", "sigmoid"),   # Activation functions for the gate neurons in the layer. If shorter work like a rolling buffer
            output_type="linear",                       # Output type of the layer. Can be "linear" or "sum" or "raw"
            **kwargs
        )

3. 🎉 Congratulations!
    You've successfully created a polynomial neuron and a network layer!
    You can now integrate them into your PyTorch models or even train single polynomial units or layers directly.
    Sometimes a single neuron or layer might be sufficient for your task.

    .. code-block:: python

        for i in range(0, data.size(0), batch_size):
            # Get the batch
            batch = data[i:i+batch_size]
            # Forward pass
            output = honn_layer(batch)
            # Compute loss
            loss = criterion(output, target)
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()

For more information you can check the following examples:
- TBD