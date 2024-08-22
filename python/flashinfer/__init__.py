"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from .cascade import (
    MultiLevelCascadeAttentionWrapper,
    BatchDecodeWithSharedPrefixPagedKVCacheWrapper,
    BatchPrefillWithSharedPrefixPagedKVCacheWrapper,
    merge_state,
    merge_state_in_place,
    merge_states,
)
from .decode import (
    BatchDecodeWithPagedKVCacheWrapper,
    CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    single_decode_with_kv_cache,
)
from .activation import gelu_tanh_and_mul, silu_and_mul
from .group_gemm import SegmentGEMMWrapper
from .norm import fused_add_rmsnorm, rmsnorm
from .page import append_paged_kv_cache
from .prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    single_prefill_with_kv_cache,
    single_prefill_with_kv_cache_return_lse,
)
from .quantization import packbits, segment_packbits
from .rope import (
    apply_llama31_rope,
    apply_llama31_rope_inplace,
    apply_rope,
    apply_rope_inplace,
)
from .sampling import (
    chain_speculative_sampling,
    sampling_from_probs,
    top_k_renorm_prob,
    top_k_mask_logits,
    top_k_sampling_from_probs,
    top_k_top_p_sampling_from_probs,
    top_k_top_p_sampling_from_logits,
    top_p_renorm_prob,
    top_p_sampling_from_probs,
    min_p_sampling_from_probs,
)
from .sparse import BlockSparseAttentionWrapper

try:
    from ._build_meta import __version__
except ImportError:
    with open("version.txt") as f:
        __version__ = f.read().strip()
