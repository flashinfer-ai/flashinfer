"""Prepared public surface for compressed DSV4 sparse MLA."""

from __future__ import annotations

from b12x._lib.gating import default_is_supported
from b12x.preparation import Plan

from .._shared.mla.api import clear_mla_caches as clear_caches
from .._shared.mla.compressed_config import (
    compressed_sparse_mla_split_chunks_for_contract as split_chunks_for_contract,
)
from . import META
from ._preparation import (
    bind,
    invocation_from_descriptors,
    invocation_from_tensors,
    plan,
    run,
)
from .cache_writer import (
    CacheWriterQuery,
    page_nbytes,
    plan as plan_cache_writer,
    write_cache,
)
from ._scratch import (
    B12XCompressedSparseMLABinding as Binding,
    B12XCompressedSparseMLAScratch as Scratch,
    B12XCompressedSparseMLAScratchCaps as Caps,
)
from ._tuning import SparseMlaConfig, SparseMlaQuery


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl and Triton available."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "CacheWriterQuery",
    "Binding",
    "Caps",
    "Plan",
    "Scratch",
    "SparseMlaConfig",
    "SparseMlaQuery",
    "bind",
    "page_nbytes",
    "plan_cache_writer",
    "clear_caches",
    "invocation_from_descriptors",
    "invocation_from_tensors",
    "is_supported",
    "write_cache",
    "plan",
    "run",
    "split_chunks_for_contract",
]
