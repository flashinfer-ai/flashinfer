# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA decode/extend for SM12x (DeepSeek-V3.2 DSV4, GLM NSA).

Multi-head latent attention over top-k-selected KV tokens from a paged
cache (DSV4: head_dim 576 = 512 nope + 64 rope, v_head_dim 512), FP8-e4m3
or BF16 compute, split-KV decode with on-device merge; a single-pass decode
path is selected automatically on SM121. Selection indices typically come
from ``attention.nsa_indexer``.

Planned lifecycle: ``plan(Caps(...))`` -> ``bind`` (views only) ->
``run_decode`` / ``run_extend`` (capture safe).

Example:
    from flashinfer.experimental.sm12x.attention import sparse_mla

    plan    = sparse_mla.plan(sparse_mla.Caps(device="cuda", num_q_heads=16,
                                              max_q_rows=64, max_width=2048))
    spec    = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = sparse_mla.bind(plan, scratch=scratch, q=q,
                              selected_indices=topk_idx,
                              cache_seqlens_int32=lens,
                              nsa_cache_seqlens_int32=active)
    out = sparse_mla.run_decode(binding=binding, kv_cache=kv, sm_scale=scale)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="sparse_mla",
    group="attention",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "Scratch",
        "DecodeMetadata",
        "ExtendMetadata",
        "plan",
        "bind",
        "run_decode",
        "run_extend",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "fp8_e4m3"),
    recipes=("dsv4", "glm_nsa"),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=(
            "b12x/integration/sparse_mla_scratch.py",
            "b12x/attention/mla/",
        ),
    ),
    test_path="tests/experimental/attention/test_sparse_mla.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        Caps,
        DecodeMetadata,
        ExtendMetadata,
        Plan,
        Scratch,
        bind,
        clear_caches,
        is_supported,
        plan,
        run_decode,
        run_extend,
    )

install_lazy_api(globals(), META)
