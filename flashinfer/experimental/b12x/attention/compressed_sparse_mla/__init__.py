"""Prepared compressed sparse MLA for DeepSeek V4/V4.1 on SM12x.

``plan(Caps(...), invocation=invocation_from_descriptors(...))`` declares an
immutable compressed-cache ABI. A preparation session materializes it; only the
prepared ``Plan`` can be bound and run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="compressed_sparse_mla",
    group="attention",
    api_style="prepared",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "Scratch",
        "SparseMlaConfig",
        "SparseMlaQuery",
        "CacheWriterQuery",
        "invocation_from_tensors",
        "invocation_from_descriptors",
        "plan_cache_writer",
        "write_cache",
        "page_nbytes",
        "plan",
        "bind",
        "run",
        "split_chunks_for_contract",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "fp8_e4m3"),
    recipes=("dsv4", "dsv41"),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=(
            "b12x/attention/mla/compressed_api.py",
            "b12x/integration/compressed_scratch.py",
        ),
    ),
    test_path="tests/experimental/b12x/attention/test_compressed_sparse_mla.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        Caps,
        Plan,
        Scratch,
        SparseMlaConfig,
        SparseMlaQuery,
        CacheWriterQuery,
        bind,
        invocation_from_descriptors,
        clear_caches,
        invocation_from_tensors,
        is_supported,
        plan,
        run,
        page_nbytes,
        plan_cache_writer,
        write_cache,
        split_chunks_for_contract,
    )

install_lazy_api(globals(), META)
