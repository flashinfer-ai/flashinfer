"""Unified paged-prefill — experimental implementation package.

The public entry points live in core (``flashinfer.prefill``,
marked ``@flashinfer_experimental_api``); everything backend-facing lives
here so the feature can be removed or graduated as one directory, per
``flashinfer/experimental/README.md``.

Layout (mirrors ``flashinfer/mla/_batch_mla/``):

    _contracts.py    PlanMetadata, Resolution, loud-error helpers
    _planning.py     structural/value validation, canonical -> derived forms
    _selection.py    tensor-free resolve_paged_attention() (level-1 selection)
    _controller.py   plan()/run() lifecycle, transactional publication
    _backends/       one module per backend + declarative capabilities

Reject-or-correct is the property the whole package is built around: any
call either raises a clean, actionable error or returns results matching an
independent reference. ``tests/experimental/test_paged_attention_fuzzer.py``
enforces it with randomized valid and corrupted inputs.
"""

from ._backends import CAPABILITIES, MIN_DENSE_PAGE_SIZE, PagedAttentionCapabilities
from ._contracts import PagedAttentionMetadata, PlanMetadata, Resolution
from ._controller import PagedAttentionController
from ._planning import Derived, derive
from ._selection import HEURISTIC_ORDER, resolve_paged_attention

__all__ = [
    "CAPABILITIES",
    "Derived",
    "HEURISTIC_ORDER",
    "MIN_DENSE_PAGE_SIZE",
    "PagedAttentionCapabilities",
    "PagedAttentionController",
    "PagedAttentionMetadata",
    "PlanMetadata",
    "Resolution",
    "derive",
    "resolve_paged_attention",
]
