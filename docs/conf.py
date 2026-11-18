from typing import Any, List

import flashinfer  # noqa: F401

# import tlcpack_sphinx_addon
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# FlashInfer is installed via pip before building docs
autodoc_mock_imports = [
    "torch",
    "triton",
    "flashinfer._build_meta",
    "cuda",
    "quack",
    "numpy",
    "einops",
    "mpi4py",
]

project = "FlashInfer"
author = "FlashInfer Contributors"
copyright = f"2023-2026, {author}"

version = flashinfer.__version__
release = flashinfer.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_tabs.tabs",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
]

autodoc_default_flags = ["members"]
autosummary_generate = True

source_suffix = [".rst"]

language = "en"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "tutorials/jax_tvm_ffi/README.rst",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

sphinx_gallery_conf = {
    "examples_dirs": "tutorials/jax_tvm_ffi",
    "gallery_dirs": "tutorials/generated/jax_tvm_ffi",
    "filename_pattern": r".*\.py",
    "plot_gallery": "False",
    "download_all_examples": False,
}

# A list of ignored prefixes for module index sorting.
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

html_theme = "furo"  # "sphinx_rtd_theme"

templates_path: List[Any] = ["_templates"]

html_static_path = ["_static"]

html_theme_options = {
    "logo_only": True,
    "light_logo": "FlashInfer-white-background.png",
    "dark_logo": "FlashInfer-black-background.png",
}


def _limit_api_sidebar_depth(app, pagename, templatename, context, doctree):
    from bs4 import BeautifulSoup
    from furo.navigation import get_navigation_tree

    # Keep tutorial subpages visible while removing generated symbols below API topics.
    tree = BeautifulSoup(
        context["toctree"](
            collapse=False, titles_only=True, maxdepth=2, includehidden=True
        ),
        "html.parser",
    )
    for caption in tree.select("p.caption"):
        if not caption.get_text(strip=True).startswith("API:"):
            continue
        group = caption.find_next_sibling("ul")
        if group is not None:
            for topic in group.find_all("li", recursive=False):
                children = topic.find("ul", recursive=False)
                if children is not None:
                    children.decompose()
    context["furo_navigation_tree"] = get_navigation_tree(str(tree))


def setup(app):
    app.connect("html-page-context", _limit_api_sidebar_depth, priority=800)
