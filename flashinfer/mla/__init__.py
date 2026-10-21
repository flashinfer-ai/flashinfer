# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ._core import *  # noqa: F401,F403

_PRIMS_TS_LAZY_EXPORTS = frozenset(
    {
        "get_prims_ts_batch_mla_decode_workspace_size",
    }
)

_SPARSE_MLA_SM120_LAZY_EXPORTS = frozenset(
    {
        "SparseMLASm120CalibrationReport",
        "SparseMLASm120DecodeConfig",
        "SparseMLASm120Wrapper",
        "calibrate_sparse_mla_sm120",
        "supported_sparse_mla_sm120_configs",
        "dsv41_fp4_quantize_append_sparse_mla_cache",
        "dsv41_fp4_quantize_pack_sparse_mla_cache",
        "dsv41_fp8_quantize_append_sparse_mla_cache",
        "dsv41_fp8_quantize_pack_sparse_mla_cache",
    }
)

_SPARSE_MLA_NVFP4_SM120_LAZY_EXPORTS = frozenset(
    {
        "nvfp4_quantize_append_sparse_mla_cache",
        "nvfp4_quantize_pack_sparse_mla_cache",
    }
)

_CAKE_SPARSE_MLA_SM120_NVFP4_LAZY_EXPORTS = frozenset(
    {
        "cake_sparse_mla_sm120_dsv4_nvfp4_decode",
        "cake_sparse_mla_sm120_dsv4_nvfp4_format_info",
        "cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks",
        "cake_sparse_mla_sm120_dsv4_nvfp4_plan_head_tiles",
        "cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill",
        "cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill_grid_head_blocks_first",
        "cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill_q_evict_first",
        "cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill_stages",
        "cake_sparse_mla_sm120_dsv4_nvfp4_plan_splits",
        "cake_sparse_mla_sm120_dsv4_nvfp4_prefill",
        "cake_sparse_mla_sm120_dsv4_nvfp4_prefill_head_tiles",
        "cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes",
        "cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel",
        "cake_sparse_mla_sm120_dsv4_nvfp4_supported_heads",
    }
)

_CAKE_DSV4_NVFP4_ROPE_INSERT_LAZY_EXPORTS = frozenset(
    {
        "cake_dsv4_nvfp4_kv_rope_quantize_insert",
        "cake_dsv4_nvfp4_rope_insert_format_info",
        "cake_dsv4_nvfp4_rope_quantize_insert",
    }
)

_CAKE_SPARSE_MLA_SM120_DSV41_MIXED_LAZY_EXPORTS = frozenset(
    {
        "cake_sparse_mla_sm120_dsv41_mixed_decode",
        "cake_sparse_mla_sm120_dsv41_mixed_format_info",
        "cake_sparse_mla_sm120_dsv41_mixed_num_chunks",
        "cake_sparse_mla_sm120_dsv41_mixed_plan_head_tiles",
        "cake_sparse_mla_sm120_dsv41_mixed_plan_splits",
        "cake_sparse_mla_sm120_dsv41_mixed_plan_variant",
        "cake_sparse_mla_sm120_dsv41_mixed_scratch_bytes",
        "cake_sparse_mla_sm120_dsv41_mixed_supported_heads",
    }
)

_CAKE_DSV4_LAZY_EXPORTS = frozenset(
    {
        "cake_dsv4_workspace_layout",
        "cake_dsv4_workspace_requirement",
        "cake_dsv4_workspace_reset",
        "get_cake_dsv4_workspace_bytes",
        "resolve_cake_dsv4_sparse_metadata",
    }
)

_CAKE_MLA_NVFP4_PAGED_DECODE_LAZY_EXPORTS = frozenset(
    {
        "CakeMlaNvfp4PagedDecode",
        "CakeMlaNvfp4QueryQuantize",
    }
)

_CAKE_KIMI_K3_MLA_LAZY_EXPORTS = frozenset(
    {
        "KimiK3MlaFp8PagedAttention",
        "run_cake_kimi_k3_mla_fp8_paged_attention",
    }
)


def __getattr__(name: str):
    """Resolve lazily-exported MLA APIs without loading their runtime at import."""

    if name in _PRIMS_TS_LAZY_EXPORTS:
        from ..attention.prims_ts import mla_decode as prims_ts_mla_decode

        value = getattr(prims_ts_mla_decode, name)
        globals()[name] = value
        return value
    if name in _SPARSE_MLA_SM120_LAZY_EXPORTS:
        from . import _sparse_mla_sm120

        value = getattr(_sparse_mla_sm120, name)
        globals()[name] = value
        return value
    if name in _SPARSE_MLA_NVFP4_SM120_LAZY_EXPORTS:
        from ._sparse_mla_sm120 import _dsv4_nvfp4

        value = getattr(_dsv4_nvfp4, name)
        globals()[name] = value
        return value
    if name in _CAKE_SPARSE_MLA_SM120_NVFP4_LAZY_EXPORTS:
        from ._sparse_mla_sm120 import _cake_dsv4_nvfp4

        value = getattr(_cake_dsv4_nvfp4, name)
        globals()[name] = value
        return value
    if name in _CAKE_DSV4_NVFP4_ROPE_INSERT_LAZY_EXPORTS:
        from ._sparse_mla_sm120 import _cake_dsv4_nvfp4_rope_insert

        value = getattr(_cake_dsv4_nvfp4_rope_insert, name)
        globals()[name] = value
        return value
    if name in _CAKE_SPARSE_MLA_SM120_DSV41_MIXED_LAZY_EXPORTS:
        from ._sparse_mla_sm120 import cake_dsv41_mixed

        value = getattr(cake_dsv41_mixed, name)
        globals()[name] = value
        return value
    if name in _CAKE_DSV4_LAZY_EXPORTS:
        from . import cake_dsv4

        value = getattr(cake_dsv4, name)
        globals()[name] = value
        return value
    if name in _CAKE_MLA_NVFP4_PAGED_DECODE_LAZY_EXPORTS:
        from ..experimental.cake_mla_nvfp4_paged_decode import cake_backend

        value = getattr(cake_backend, name)
        globals()[name] = value
        return value
    if name in _CAKE_KIMI_K3_MLA_LAZY_EXPORTS:
        from . import cake_kimi_k3_mla

        value = getattr(cake_kimi_k3_mla, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Include lazily-exported names alongside the module globals."""
    return sorted(
        set(globals())
        | _PRIMS_TS_LAZY_EXPORTS
        | _SPARSE_MLA_SM120_LAZY_EXPORTS
        | _SPARSE_MLA_NVFP4_SM120_LAZY_EXPORTS
        | _CAKE_SPARSE_MLA_SM120_NVFP4_LAZY_EXPORTS
        | _CAKE_DSV4_NVFP4_ROPE_INSERT_LAZY_EXPORTS
        | _CAKE_SPARSE_MLA_SM120_DSV41_MIXED_LAZY_EXPORTS
        | _CAKE_DSV4_LAZY_EXPORTS
        | _CAKE_KIMI_K3_MLA_LAZY_EXPORTS
        | _CAKE_MLA_NVFP4_PAGED_DECODE_LAZY_EXPORTS
    )
