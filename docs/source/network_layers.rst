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
temporal embedding inside :doc:`SA-GHONN <full_models>` when ``temporal_lookback``
is greater than zero.

.. autoclass:: ghonn_models_pytorch.core.ConvGhonn