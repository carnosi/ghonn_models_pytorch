.. _installation:

Installation 🛠️
================

**PyPI version:**

.. code-block:: bash

    pip install ghonn-models-pytorch

**The latest version from GitHub:**

.. code-block:: bash

    pip install git+https://github.com/carnosi/ghonn_models_pytorch

The current source requires Python 3.12 or newer and PyTorch ``~=2.13``.
See ``pyproject.toml`` for the complete dependency constraints. The published
PyPI release can differ from the repository's current version.

From a local checkout, install the extras for notebooks, documentation, and
development, then build the docs or run tests:

.. code-block:: bash

    python -m pip install -e ".[examples,docs,dev]"
    python -m sphinx -b html docs/source docs/build/html
    python -m pytest tests/
