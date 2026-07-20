# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Paged-KV self-attention for SM12x (decode + extend/prefill).

FA2-style paged attention: BF16/FP16 queries, BF16/FP16/FP8-e4m3 KV cache
(FP8 KV needs BF16 queries + k/v descales), any head_dim multiple of 16,
attention sinks, sliding window, and an MSA block-sparse variant driven by
``q2k_indices``. The planner owns tile/split-KV/chunk policy; integrations
supply tensors, shape metadata, and capacity caps (``Budget``). Decode
supports CUDA-graph replay with all metadata rebuilt on-device.

Planned lifecycle: ``plan(Caps(...))`` sizes the caller-owned scratch;
``bind`` maps allocation-free views; ``run`` launches (capture safe).
``Workspace`` is the preplanned arena alternative for workspace-style
serving.

Example:
    from flashinfer.experimental.sm12x.attention import paged

    plan    = paged.plan(paged.Caps(mode="decode", dtype=torch.bfloat16, ...))
    spec    = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = paged.bind(plan, scratch=scratch, q=q, k_cache=k, v_cache=v,
                         output=out, page_table=pt, cache_seqlens=lens,
                         cu_seqlens_q=cu_q)
    out, lse = paged.run(binding=binding)
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
        "plan",
        "bind",
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
    test_path="tests/experimental/attention/test_paged.py",
    since="0.7.0",
    notes=(
        "Decode CUDA-graph replay is implemented but currently untested "
        "in-tree (b12x's replay tests are stale against the current planner); "
        "graph coverage comes from compressed_mla until they are refreshed."
    ),
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        Budget,
        Caps,
        Plan,
        Workspace,
        bind,
        clear_caches,
        infer_mode,
        is_supported,
        plan,
        run,
    )

install_lazy_api(globals(), META)
