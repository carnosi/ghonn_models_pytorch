.. _insight:

Insights 💡
===========

1. GHONU and GHONN training

   The GHONU model may profit from having different initial learning rates
   for the predictor and gate weights. This can be achieved by passing the
   params to desired optimizer with various learning rates. Both GHONU and
   GHONN provide named parameters for easy access.

   Where in the case of GHONU:

   .. code-block:: python

      predictor_params = list(ghonu_neuron.predictor.parameters())
      gate_params = (list(ghonu_neuron.gate.parameters())
                     if ghonu_neuron.gate is not None else [])

      # Create optimizer with different learning rates for predictor and gate
      optimizer = torch.optim.Adam([
          {"params": predictor_params, "lr": 1e-3},
          {"params": gate_params, "lr": 1e-4}
      ])

   For a GHONN layer, include the linear output head as well. This example
   assumes ``import torch`` and the ``ghonn_layer`` from :doc:`quickstart`:

   .. code-block:: python

      gate_params = list(ghonn_layer.gates.parameters())
      gate_ids = {id(parameter) for parameter in gate_params}
      other_params = [parameter for parameter in ghonn_layer.parameters()
                      if id(parameter) not in gate_ids]
      optimizer = torch.optim.Adam([
          {"params": other_params, "lr": 1e-3},
          {"params": gate_params, "lr": 1e-4},
      ])

   ``predictors`` and ``gates`` expose configuration groups, not one module
   per output neuron. ``gates`` includes only active gates. A GHONU's
   ``gate`` is ``None`` when ``gate_order=0`` or ``force_gate_output_one=True``.
   Separate learning rates are a tuning option, not a guaranteed improvement.
