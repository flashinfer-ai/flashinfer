"""QSA adapter for the selected-position paged-forward specialization."""

from ..paged._selected_forward import (
    clear_caches,
    launch_selected_paged_gqa_direct,
    launch_sparse_gqa_merge,
    launch_sparse_gqa_split,
    precompile_sparse_gqa_split,
)
from ._sparse_gqa_cute_config import is_candidate as is_supported

__all__ = [
    "clear_caches",
    "is_supported",
    "launch_selected_paged_gqa_direct",
    "launch_sparse_gqa_merge",
    "launch_sparse_gqa_split",
    "precompile_sparse_gqa_split",
]
