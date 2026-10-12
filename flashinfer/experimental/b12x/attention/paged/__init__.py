"""Paged-KV self-attention for SM12x (decode + extend/prefill).

FA2-style paged attention: BF16/FP16 queries, BF16/FP16/FP8-e4m3 KV cache
(FP8 KV needs BF16 queries + k/v descales), any head_dim multiple of 16,
attention sinks, sliding window, and an MSA block-sparse variant driven by
``q2k_indices``. The planner owns tile/split-KV/chunk policy; integrations
supply tensors, shape metadata, and capacity caps (``Budget``). Decode
supports CUDA-graph replay with all metadata rebuilt on-device.

Planned lifecycle: ``plan(caps, invocation=metadata)`` declares a configuration
without device work. ``PreparationSession`` materializes it, then ``bind(plan, ...)``
maps caller-owned scratch and ``run(binding=..., plan=...)`` launches.
Decode replay metadata is owned by the prepared plan.

Without invocation metadata, ``plan(caps)`` returns a heuristic scratch plan
with ``layout``, ``bind``, and replay-state preparation methods. Call
``compile(binding=...)`` and warm execution before CUDA graph capture.

Example:
    from b12x.attention import paged

    caps = paged.Caps(mode="decode", dtype=torch.bfloat16, ...)
    declaration = paged.plan(caps, invocation=paged.invocation_from_tensors(caps, ...))
    session.prepare((declaration.request(...),))
    binding = paged.bind(declaration, scratch=scratch, q=q, k_cache=k, v_cache=v,
                         output=out, page_table=pt, cache_seqlens=lens,
                         cu_seqlens_q=cu_q)
    out, lse = paged.run(binding=binding, plan=declaration)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="paged",
    group="attention",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "Workspace",
        "Budget",
        "DecodeGraphCapacity",
        "GqaConfig",
        "GqaQuery",
        "ExtendGraphCapacity",
        "VerifyGraphCapacity",
        "DecodeGraphScratchEnvelope",
        "decode_graph_capacity",
        "extend_graph_capacity",
        "verify_graph_capacity",
        "decode_graph_scratch_envelope",
        "plan",
        "bind",
        "compile",
        "invocation_from_descriptors",
        "invocation_from_tensors",
        "run",
        "infer_mode",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "fp16", "fp8_e4m3"),
    recipes=("dense", "msa_block_sparse"),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=(
            "b12x/attention/paged/",
            "b12x/integration/paged_attention_scratch.py",
        ),
    ),
    test_path="tests/experimental/b12x/attention/test_paged.py",
    since="0.7.0",
    notes=(
        "Decode CUDA-graph replay uses fixed preplanned capacity and rebuilds "
        "live schedules on-device; in-tree tests cover split, direct, MSA, "
        "multi-query-tile, shared-scratch, and high-page-id replay."
    ),
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        Budget,
        Caps,
        DecodeGraphCapacity,
        GqaConfig,
        GqaQuery,
        ExtendGraphCapacity,
        VerifyGraphCapacity,
        DecodeGraphScratchEnvelope,
        Plan,
        Workspace,
        bind,
        compile,
        clear_caches,
        decode_graph_capacity,
        extend_graph_capacity,
        verify_graph_capacity,
        decode_graph_scratch_envelope,
        infer_mode,
        invocation_from_descriptors,
        invocation_from_tensors,
        is_supported,
        plan,
        run,
    )

install_lazy_api(globals(), META)
