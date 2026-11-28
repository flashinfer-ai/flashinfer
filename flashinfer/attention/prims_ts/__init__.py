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

"""Experimental task-scheduled attention entry points."""

from .variants.block_sparse import (
    BlockSparsePagedTSWrapper,
    BlockSparseTSWrapper,
    block_sparse_attention,
    block_sparse_attention_with_paged_kv_cache,
)
from .variants.decode import (
    BatchDecodePagedTSWrapper,
    PrimsTSBatchDecodePlan,
    batch_decode_with_paged_kv_cache,
    get_prims_ts_batch_decode_workspace_size,
    make_q_token_kv_block_sparse_qo_indptr,
    prepare_prims_ts_batch_decode_with_kv_cache,
    suggest_q_token_kv_block_sparse_group_size,
    validate_q_token_kv_block_sparse_group_size,
)
from .variants.context import (
    BatchPrefillPagedTSWrapper,
    BatchPrefillTSWrapper,
    batch_prefill,
    batch_prefill_with_paged_kv_cache,
)
from .variants.mla_decode import (
    BatchSparseMLADecodePagedTSWrapper,
    batch_sparse_mla_decode_with_paged_kv_cache,
    get_prims_ts_sparse_mla_decode_workspace_size,
    BatchMLADecodePagedTSWrapper,
    MLAAttentionTuning,
    batch_mla_decode_with_paged_kv_cache,
    get_prims_ts_batch_mla_decode_workspace_size,
)
from .variants.q_token_kv_block_sparse_metadata import (
    QTokenKvBlockSparsePagedTSWrapper,
    get_q_token_kv_block_sparse_workspace_size,
    q_token_kv_block_sparse_attention_with_paged_kv_cache,
)
from .variants.sage import SageAttentionConfig, SageAttentionParams
from .variants.vc_attention import VCAttentionConfig, VCAttentionParams, vc_qk8_dtype
from .variants.vc_attention.quantization import VCAttentionPreprocessor

__all__ = [
    "BatchSparseMLADecodePagedTSWrapper",
    "batch_sparse_mla_decode_with_paged_kv_cache",
    "get_prims_ts_sparse_mla_decode_workspace_size",
    "BlockSparseTSWrapper",
    "SageAttentionConfig",
    "SageAttentionParams",
    "VCAttentionConfig",
    "VCAttentionParams",
    "VCAttentionPreprocessor",
    "vc_qk8_dtype",
    "BlockSparsePagedTSWrapper",
    "block_sparse_attention",
    "block_sparse_attention_with_paged_kv_cache",
    "BatchPrefillTSWrapper",
    "BatchPrefillPagedTSWrapper",
    "batch_prefill",
    "batch_prefill_with_paged_kv_cache",
    "BatchDecodePagedTSWrapper",
    "PrimsTSBatchDecodePlan",
    "batch_decode_with_paged_kv_cache",
    "get_prims_ts_batch_decode_workspace_size",
    "make_q_token_kv_block_sparse_qo_indptr",
    "prepare_prims_ts_batch_decode_with_kv_cache",
    "suggest_q_token_kv_block_sparse_group_size",
    "validate_q_token_kv_block_sparse_group_size",
    "QTokenKvBlockSparsePagedTSWrapper",
    "get_q_token_kv_block_sparse_workspace_size",
    "q_token_kv_block_sparse_attention_with_paged_kv_cache",
    "BatchMLADecodePagedTSWrapper",
    "MLAAttentionTuning",
    "batch_mla_decode_with_paged_kv_cache",
    "get_prims_ts_batch_mla_decode_workspace_size",
]

# The feature packages live in ``variants/``. Keep their public module paths
# (``flashinfer.attention.prims_ts.decode`` and so on) importable as before.
# The loop looks the packages up by name so kernel cache keys that depend on
# this initializer do not pull in every feature.
import sys

from .variants import (  # noqa: F401  (public module attributes)
    block_sparse,
    context,
    decode,
    mla_decode,
    q_token_kv_block_sparse_metadata,
    sage,
    vc_attention,
)

for name in (
    "block_sparse",
    "context",
    "decode",
    "mla_decode",
    "q_token_kv_block_sparse_metadata",
    "sage",
    "vc_attention",
    "vc_attention.quantization",
):
    sys.modules.setdefault(
        f"{__name__}.{name}", sys.modules[f"{__name__}.variants.{name}"]
    )
del name, sys
