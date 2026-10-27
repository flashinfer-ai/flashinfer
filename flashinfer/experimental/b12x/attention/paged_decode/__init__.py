"""Warp-specialized split-KV paged decode/verify attention for SM12x.

GQA decode and speculative-verify attention over paged K/V: one CTA streams a
request's K/V once for all of its query rows (rows = query tokens x GQA group,
up to 128), with a TMA producer warp and consumer warps split into row and key
groups.  Covers asymmetric Q/K and V head dims (192/128) and 128/128, BF16 or
FP8-E4M3 KV (widened exactly to BF16 in registers), attention sinks,
sliding windows, causal or non-causal masking, and ragged batches (any mix of
per-request query lengths up to ``max_q_per_req``).  Split-KV partitions are
derived on device from ``cache_seqlens`` and partials are FP32, so one captured
CUDA graph serves every context length with a single rounding of the output.

Planned lifecycle: ``plan(Caps(...))`` declares the layer family,
``PreparationSession`` compiles it, ``bind(plan, scratch=..., ...)`` maps the
caller's tensors and scratch, and ``run(binding)`` launches.

Example:
    from b12x.attention import paged_decode

    declaration = paged_decode.plan(paged_decode.Caps(
        device="cuda", num_q_heads=16, num_kv_heads=1, head_dim_qk=192,
        head_dim_vo=128, page_size=64, max_batch=32, max_q_per_req=8))
    session.prepare((declaration.request(name=..., prepare_call=...),))
    binding = paged_decode.bind(declaration, scratch=scratch, q=q, k_cache=k,
                                v_cache=v, output=out, page_table=pt,
                                cache_seqlens=lens, cu_seqlens_q=cu_q)
    paged_decode.run(binding)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="paged_decode",
    group="attention",
    api_style="planned",
    entry_points=(
        "Binding",
        "Caps",
        "PagedDecodeConfig",
        "PagedDecodeQuery",
        "Plan",
        "bind",
        "clear_caches",
        "is_supported",
        "plan",
        "run",
        "supports",
        "write_kv",
    ),
    dtypes=("bf16", "fp8_e4m3"),
    provenance=Provenance(
        repo="https://github.com/local-inference-lab/b12x",
        commit="8a99d639",
        paths=("b12x/attention/paged_decode/",),
    ),
    test_path="tests/experimental/b12x/attention/test_paged_decode.py",
    since="1.3.0",
    notes=(
        "Built from b12x.attention.paged's TMA and MMA primitives; tested against "
        "paged.reference for 192/128 GQA16/GQA8, windows, sinks, FP8 KV, "
        "non-causal windows, ragged verify and CUDA-graph replay."
    ),
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        Caps,
        PagedDecodeConfig,
        PagedDecodeQuery,
        Plan,
        bind,
        clear_caches,
        is_supported,
        plan,
        run,
        supports,
        write_kv,
    )

install_lazy_api(globals(), META)
