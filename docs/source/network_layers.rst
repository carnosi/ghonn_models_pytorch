.. _layers:

Network Layers 🧩
==================
Single layers of :doc:`neurons` (HONU or GHONU) with configurable output
heads. Layers can be used standalone, stacked, or combined into the
:doc:`full models <full_models>`.

.. contents::
    :local:

.. _honn:

HONN
~~~~
.. autoclass:: ghonn_models_pytorch.core.HONN

.. _ghonn:

GHONN
~~~~~
.. autoclass:: ghonn_models_pytorch.core.GHONN

.. _convghonn:

Convolutional GHONN
~~~~~~~~~~~~~~~~~~~
A causal, sliding-window layer that applies a :class:`~ghonn_models_pytorch.core.GhonuBank`
over temporal windows of the input. It can be used standalone or as the
trend and residual experts inside :doc:`SA-GHONN <full_models>` when ``temporal_lookback``
is greater than zero.

.. autoclass:: ghonn_models_pytorch.core.ConvGhonn

``ConvGhonn`` takes ``(batch, length, in_features)`` and returns
``(batch, length, out_features)``. Each window contains the current step
and ``lookback`` past steps. Default replicate padding and zero padding
preserve causality; reflect/circular padding can use later observations
at the left boundary.

Independent per-feature convolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``ConvGhonn1D`` takes ``(batch, in_features, input_length)`` and returns
``(batch, output_length, in_features * n_kernels)``. It uses independent
filters per feature and centered padding, so it is not a causal layer.
With ``padding_mode="valid"``, the output length is
``input_length - kernel_size + 1``; other supported modes preserve length.
SA-GHONN uses this layer in its default seasonal expert.

.. autoclass:: ghonn_models_pytorch.core.ConvGhonn1D

Output heads
~~~~~~~~~~~~
For HONN and GHONN inputs shaped ``(..., in_features)`` (at least 2D):

* ``linear`` returns ``(..., out_features)``.
* ``raw`` returns ``(..., layer_size)`` and requires ``out_features=layer_size``.
* ``sum`` returns ``(...)`` with the final neuron dimension removed and
  requires ``out_features=1``. For a batch input, this is ``(batch,)``,
  not ``(batch, 1)``.
