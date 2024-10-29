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

from .activation import (
    gelu_and_mul as gelu_and_mul,
    gelu_tanh_and_mul as gelu_tanh_and_mul,
    silu_and_mul as silu_and_mul,
)

from .cascade import (
    BatchDecodeWithSharedPrefixPagedKVCacheWrapper as BatchDecodeWithSharedPrefixPagedKVCacheWrapper,
    BatchPrefillWithSharedPrefixPagedKVCacheWrapper as BatchPrefillWithSharedPrefixPagedKVCacheWrapper,
    MultiLevelCascadeAttentionWrapper as MultiLevelCascadeAttentionWrapper,
    merge_state as merge_state,
    merge_state_in_place as merge_state_in_place,
    merge_states as merge_states,
)
from .decode import (
    BatchDecodeWithPagedKVCacheWrapper as BatchDecodeWithPagedKVCacheWrapper,
    CUDAGraphBatchDecodeWithPagedKVCacheWrapper as CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    BatchDecodeMlaWithPagedKVCacheWrapper as BatchDecodeMlaWithPagedKVCacheWrapper,
    single_decode_with_kv_cache as single_decode_with_kv_cache,
)
from .gemm import (
    SegmentGEMMWrapper as SegmentGEMMWrapper,
    bmm_fp8 as bmm_fp8,
)
from .norm import (
    fused_add_rmsnorm as fused_add_rmsnorm,
    gemma_fused_add_rmsnorm as gemma_fused_add_rmsnorm,
    gemma_rmsnorm as gemma_rmsnorm,
    rmsnorm as rmsnorm,
)
from .page import (
    append_paged_kv_cache as append_paged_kv_cache,
)
from .prefill import (
    BatchPrefillWithPagedKVCacheWrapper as BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper as BatchPrefillWithRaggedKVCacheWrapper,
    single_prefill_with_kv_cache as single_prefill_with_kv_cache,
    single_prefill_with_kv_cache_return_lse as single_prefill_with_kv_cache_return_lse,
)
from .quantization import (
    packbits as packbits,
    segment_packbits as segment_packbits,
)
from .rope import (
    apply_llama31_rope as apply_llama31_rope,
    apply_llama31_rope_inplace as apply_llama31_rope_inplace,
    apply_rope as apply_rope,
    apply_rope_inplace as apply_rope_inplace,
    apply_rope_pos_ids as apply_rope_pos_ids,
    apply_rope_pos_ids_inplace as apply_rope_pos_ids_inplace,
)
from .sampling import (
    chain_speculative_sampling as chain_speculative_sampling,
    min_p_sampling_from_probs as min_p_sampling_from_probs,
    sampling_from_probs as sampling_from_probs,
    top_k_mask_logits as top_k_mask_logits,
    top_k_renorm_probs as top_k_renorm_probs,
    top_k_sampling_from_probs as top_k_sampling_from_probs,
    top_k_top_p_sampling_from_logits as top_k_top_p_sampling_from_logits,
    top_k_top_p_sampling_from_probs as top_k_top_p_sampling_from_probs,
    top_p_renorm_probs as top_p_renorm_probs,
    top_p_sampling_from_probs as top_p_sampling_from_probs,
)
from .sparse import (
    BlockSparseAttentionWrapper as BlockSparseAttentionWrapper,
)

try:
    from ._build_meta import __version__ as __version__
except ImportError:
    with open("version.txt") as f:
        __version__ = f.read().strip()

try:
    import aot_config as aot_config  # type: ignore[import]
except ImportError:
    aot_config = None
