# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).parent.parent.parent).resolve()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "GHONN Models"
copyright = "{}, Ondrej Budik".format(datetime.datetime.now().year)
author = "Ondrej Budik"


def get_version():
    """Get the version of the package."""
    from ghonn_models_pytorch import __version__

    return __version__


version = get_version()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "autodocsumm",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"

html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"

html_static_path = ["_static"]

# -- Extensions configuration -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

autodoc_inherit_docstrings = False
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_numpy_docstring = False

autodoc_mock_imports = [
    "torch",
    "numpy",
    # "ghonn_models_pytorch",
]

autoclass_content = "both"
autodoc_typehints = "description"

# --- Work around to make autoclass signatures not (*args, **kwargs) ----------


class FakeSignature:
    def __getattribute__(self, *args):
        raise ValueError


def f(app, obj, bound_method):
    if "__new__" in obj.__name__:
        obj.__signature__ = FakeSignature()


def setup(app):
    app.connect("autodoc-before-process-signature", f)


# Custom configuration --------------------------------------------------------

autodoc_member_order = "bysource"
