import os
import sys

# import tlcpack_sphinx_addon
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

sys.path.insert(0, os.path.abspath("../python"))
os.environ["BUILD_DOC"] = "1"
autodoc_mock_imports = ["torch"]

project = "FlashInfer"
author = "FlashInfer Contributors"
copyright = "2023-2024, {}".format(author)

version = "0.1.5"
release = "0.1.5"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_tabs.tabs",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
]

autodoc_default_flags = ["members"]
autosummary_generate = True

source_suffix = [".rst"]

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

html_theme = "furo"  # "sphinx_rtd_theme"

templates_path = []

html_static_path = []

html_theme_options = {
    "logo_only": True,
}

html_static_path = ["_static"]
html_theme_options = {
    "light_logo": "FlashInfer-white-background.png",
    "dark_logo": "FlashInfer-black-background.png",
}
