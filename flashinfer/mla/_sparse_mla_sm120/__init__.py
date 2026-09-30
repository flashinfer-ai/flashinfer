# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""SM120 sparse MLA APIs and established downstream compatibility exports."""

from ._api import (  # noqa: F401
    SparseMLASm120DecodeConfig,
    SparseMLASm120Wrapper,
    _SparseMLAPagedAttentionRunner,
    _KV_CACHE_FORMATS,
    _bytes_per_token_for_model_type,
    _check_last_dim,
    _decode_dispatch_error_message,
    _decode_dsv4_num_splits,
    _decode_scratch_views,
    _expected_d_v,
    _inline_cache_block_contiguous,
    _packed_kv_page_block_size,
    _require_d_v,
    _sparse_mla_sm120_paged_attention,
    dsv41_fp4_quantize_append_sparse_mla_cache,
    dsv41_fp4_quantize_pack_sparse_mla_cache,
    get_sparse_mla_sm120_module,
    sparse_mla_sm120_decode_dsv3_2,
    sparse_mla_sm120_decode_dsv4,
    supported_sparse_mla_sm120_configs,
)
from ._calibration import (  # noqa: F401
    SparseMLASm120CalibrationReport,
    calibrate_sparse_mla_sm120,
)
from ._execution import (  # noqa: F401
    KV_SCALE_FORMATS as _KV_SCALE_FORMATS,
    get_sparse_mla_sm120_module as _get_sparse_mla_sm120_decode_module,
    normalize_kv_scale_format as _normalize_kv_scale_format,
    resolve_model_type as _resolve_model_type,
)
from ._policy import (  # noqa: F401
    KernelVariant,
    _BI,
    _DECODE_DOTS3_SWA_DISPATCH,
    _DECODE_DOTS3_SWA_TOPK,
    _DECODE_DSV3_2_DISPATCH,
    _DECODE_DSV3_2_TOPKS,
    _DECODE_DSV4_DISPATCH,
    _DECODE_DSV4_TOPKS,
    _DECODE_DSV4_1_DISPATCH,
    _DECODE_DSV4_1_TOPK,
    _DECODE_GLM53_NOPE_DISPATCH,
    _DECODE_GLM53_NOPE_TOPK,
    _DECODE_MAX_TOKENS,
    _D_V,
    _D_V_BY_MODEL_TYPE,
    _MODEL_TYPE_DOTS3_SWA,
    _MODEL_TYPE_DSV3_2,
    _MODEL_TYPE_DSV4,
    _MODEL_TYPE_DSV4_1,
    _MODEL_TYPE_GLM53_NOPE,
    _MODEL_TYPE_GLM_NSA,
    _MODEL_TYPE_TO_FAMILY,
    _decode_chunk_width,
    _decode_scratch_heads,
    _normalize_prefill_impl,
    _resolve_cpb,
    plan,
)
