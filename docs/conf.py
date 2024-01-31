import os
import sys

import tlcpack_sphinx_addon
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

sys.path.insert(0, os.path.abspath("../python"))
os.environ["BUILD_DOC"] = "1"

import flashinfer

project = 'FlashInfer'
author = "FlashInfer Contributors"
footer_copyright = '2023-2024, {}'.format(author)

version = "0.0.1"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_tabs.tabs",
    "sphinx.ext.autodoc",
    # "sphinx_toolbox.collapse",
    # "sphinxcontrib.httpdomain",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    # "sphinx_reredirects",
]

autodoc_mock_imports = ["torch"]

source_suffix = [".rst"]

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

# The theme is set by the make target
import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

templates_path = []

html_static_path = []

footer_note = " "

html_theme_options = {
    "logo_only": True,
}

header_links = [
    ("Home", "https://flashinfer.ai"),
    ("Github", "https://github.com/flashinfer-ai/flashinfer"),
    ("Discussions", "https://github.com/orgs/flashinfer-ai/discussions"),
]

# html_logo = "https://raw.githubusercontent.com/flashinfer-ai/web-data/main/logo/FlashInfer-white-background.png"
# html_logo = "FlashInfer-white-background.png"

html_context = {
    "footer_copyright": footer_copyright,
    "footer_note": footer_note,
    "header_links": header_links,
    "display_github": True,
    "github_user": "flashinfer-ai",
    "github_repo": "flashinfer",
    "github_version": "main/docs/",
    "theme_vcs_pageview_mode": "edit",
    # "header_logo": "",
    # "header_logo_link": "https://raw.githubusercontent.com/flashinfer-ai/web-data/main/logo/FlashInfer-black-background.png",
    # "version_selecter": "",
}

# add additional overrides
# templates_path += [tlcpack_sphinx_addon.get_templates_path()]
# html_static_path += [tlcpack_sphinx_addon.get_static_path()]

