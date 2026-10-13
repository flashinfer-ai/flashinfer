"""Sparse MLA decode/extend for SM12x (DeepSeek-V3.2 DSV4, GLM NSA/Next).

Multi-head latent attention over top-k-selected KV tokens from a paged
cache (DSV4: head_dim 512 = 448 nope + 64 rope, v_head_dim 512), FP8-e4m3
or BF16 compute, split-KV decode with on-device merge; a single-pass decode
path is selected automatically on SM121. Selection indices typically come
from ``attention.dsa_indexer``.

Planned lifecycle: declare with ``plan(Caps(...))``, prepare it through a
``PreparationSession``, then bind and run the prepared ``Plan``.
Dynamic sequence lengths, page IDs and selected token indices remain binding
inputs; no runtime route selection or compilation is permitted.

GLM Next uses an explicit recipe identity because its absorbed 512-wide query
collides with DSV4 by shape. Plan with ``model_type=ModelType.GLM_NEXT`` and
use a 528-byte cache record: 512 E4M3 latent bytes followed by four FP32
group-128 scales, with no RoPE suffix. Its physical attention scale remains
``256**-0.5``.
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
        "ModelType",
        "SparseMlaConfig",
        "SparseMlaQuery",
        "Binding",
        "Scratch",
        "DecodeMetadata",
        "ExtendMetadata",
        "plan",
        "bind",
        "run",
        "plan_cache_writer",
        "plan_pooled_selection",
        "concat_and_cache_glm_next_mla",
        "concat_and_cache_glm_next_mla_fp8",
        "concat_and_cache_glm_next_mla_nvfp4",
        "expand_pooled_topk_to_physical_slots",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "fp8_e4m3"),
    recipes=("dsv4", "glm_nsa", "glm_next"),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=(
            "b12x/integration/sparse_mla_scratch.py",
            "b12x/attention/mla/",
        ),
    ),
    test_path="tests/experimental/b12x/attention/test_sparse_mla.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        Caps,
        DecodeMetadata,
        ExtendMetadata,
        ModelType,
        Scratch,
        SparseMlaConfig,
        SparseMlaQuery,
        bind,
        clear_caches,
        concat_and_cache_glm_next_mla,
        concat_and_cache_glm_next_mla_fp8,
        concat_and_cache_glm_next_mla_nvfp4,
        expand_pooled_topk_to_physical_slots,
        is_supported,
        plan,
        plan_cache_writer,
        plan_pooled_selection,
        run,
    )

install_lazy_api(globals(), META)
