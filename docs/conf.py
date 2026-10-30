from typing import Any, List

import flashinfer  # noqa: F401
from docutils import nodes

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


def _unwrap_nested_docfield_blocks(_app: Any, doctree: nodes.document) -> None:
    """Keep lists and paragraphs out of Sphinx's doc-field paragraphs."""
    block_types = (nodes.paragraph, nodes.bullet_list, nodes.enumerated_list)

    for paragraph in list(doctree.findall(nodes.paragraph)):
        for _first_block, child in enumerate(paragraph.children):
            if isinstance(child, block_types):
                break
        else:
            continue

        ancestor = paragraph.parent
        while ancestor is not None and not isinstance(ancestor, nodes.field_body):
            ancestor = ancestor.parent
        if ancestor is None:
            continue

        prefix = paragraph.children[:_first_block]
        replacement = paragraph.children[_first_block:]
        if isinstance(replacement[0], nodes.paragraph):
            for node in reversed(prefix):
                replacement[0].insert(0, node)
        elif prefix:
            replacement.insert(0, nodes.paragraph("", "", *prefix))
        paragraph.replace_self(replacement)


def setup(app: Any) -> None:
    app.connect("doctree-read", _unwrap_nested_docfield_blocks)
