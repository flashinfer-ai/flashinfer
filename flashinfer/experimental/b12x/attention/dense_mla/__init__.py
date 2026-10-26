"""Paged dense Multi-head Latent Attention on SM12x.

The operator consumes absorbed queries and combined compressed-cache records
directly. Logical Q/K width, logical value width, and physical-record width
are planned independently. Supported logical widths are ``(576, 512)`` and
``(1088, 1024)``.

* query: ``[total_q, local_heads, logical_qk_width]``;
* cache: ``[pages, page_size, physical_record_width]`` (or a singleton-head
  rank-4 view), where the logical key prefix is read directly;
* value: the logical value prefix of each record.

``window_size=None`` attends to the full causal context. A positive
``window_size`` restricts query position ``p`` to the inclusive interval
``[max(0, p - window_size + 1), p]``. The attention scale is caller-provided.
BF16 and E4M3 records are supported. For E4M3 cache with BF16 query input,
``q_dtype=torch.bfloat16`` plans fixed query-quantization storage in the same
caller-owned scratch buffer.

Planned lifecycle: ``plan(Caps(...))`` -> ``PreparationSession.prepare`` ->
``bind(plan, ...)`` -> ``run(plan=..., binding=...)``. Scratch is
caller-owned; preparation resolves and primes every native entry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="dense_mla",
    group="attention",
    api_style="planned",
    entry_points=(
        "Caps",
        "Budget",
        "DenseMlaConfig",
        "DenseMlaQuery",
        "Plan",
        "Binding",
        "Scratch",
        "plan",
        "invocation_from_descriptors",
        "invocation_from_tensors",
        "bind",
        "run",
        "reference",
        "infer_mode",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "fp8_e4m3"),
    recipes=("strided_paged_dense_mla",),
    requires=(),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="36cade0b",
        paths=(
            "b12x/attention/paged/",
            "b12x/attention/_shared/mla/",
            "b12x/attention/dsa_indexer/",
        ),
    ),
    test_path="tests/experimental/b12x/attention/test_dense_mla.py",
    since="0.8.0",
    notes=(
        "Physical page and record addressing is 64-bit. Optional BF16-to-E4M3 "
        "query conversion uses planned caller-owned storage."
    ),
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        Binding,
        Budget,
        Caps,
        DenseMlaConfig,
        DenseMlaQuery,
        Plan,
        Scratch,
        bind,
        clear_caches,
        infer_mode,
        invocation_from_descriptors,
        invocation_from_tensors,
        is_supported,
        plan,
        reference,
        run,
    )

install_lazy_api(globals(), META)
