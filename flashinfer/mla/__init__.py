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

from ._core import (
    BatchMLAPagedAttentionWrapper,
    MLAAutoSelectionTrace,
    MLAHeadDimensions,
    MLAKVCache,
    MLALayerDimensions,
    MLAPlanMetadata,
    MLAQuery,
    batch_mla_paged_attention,
    deepseek_mla_dimensions,
    nope_mla_dimensions,
    smaller_mla_dimensions,
    supported_mla_head_dimensions,
    supported_mla_layer_dimensions,
    trtllm_batch_decode_sparse_mla_dsv4,
    trtllm_batch_decode_with_kv_cache_mla,
    xqa_batch_decode_with_kv_cache_mla,
)

__all__ = (
    "MLAHeadDimensions",
    "deepseek_mla_dimensions",
    "nope_mla_dimensions",
    "smaller_mla_dimensions",
    "supported_mla_head_dimensions",
    "MLALayerDimensions",
    "supported_mla_layer_dimensions",
    "BatchMLAPagedAttentionWrapper",
    "MLAQuery",
    "MLAKVCache",
    "MLAPlanMetadata",
    "MLAAutoSelectionTrace",
    "batch_mla_paged_attention",
    "trtllm_batch_decode_with_kv_cache_mla",
    "xqa_batch_decode_with_kv_cache_mla",
    "trtllm_batch_decode_sparse_mla_dsv4",
    "get_prims_ts_batch_decode_mla_workspace_size",
    "prims_ts_batch_decode_with_kv_cache_mla",
)


_PRIMS_TS_LAZY_EXPORTS = frozenset(
    {
        "get_prims_ts_batch_decode_mla_workspace_size",
        "prims_ts_batch_decode_with_kv_cache_mla",
    }
)


def __getattr__(name: str):
    """Resolve PrimTS MLA APIs without loading their runtime at import."""

    if name in _PRIMS_TS_LAZY_EXPORTS:
        from ..attention.prims_ts import mla_decode as prims_ts_mla_decode

        value = getattr(prims_ts_mla_decode, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _PRIMS_TS_LAZY_EXPORTS)
