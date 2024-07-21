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

from .decode import (
    single_decode_with_kv_cache,
    BatchDecodeWithPagedKVCacheWrapper,
    CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
)
from .prefill import (
    single_prefill_with_kv_cache,
    single_prefill_with_kv_cache_return_lse,
    BatchPrefillWithRaggedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from .sparse import BlockSparseAttentionWrapper
from .cascade import (
    merge_state,
    merge_state_in_place,
    merge_states,
    BatchDecodeWithSharedPrefixPagedKVCacheWrapper,
    BatchPrefillWithSharedPrefixPagedKVCacheWrapper,
)
from .page import append_paged_kv_cache
from .sampling import (
    sampling_from_probs,
    top_p_sampling_from_probs,
    top_k_sampling_from_probs,
    top_k_top_p_sampling_from_probs,
    top_p_renorm_prob,
    top_k_renorm_prob,
    chain_speculative_sampling,
)
from .norm import rmsnorm
from .group_gemm import SegmentGEMMWrapper
from .quantization import packbits, segment_packbits

try:
    from ._build_meta import __version__
except ImportError:
    with open("version.txt") as f:
        __version__ = f.read().strip()
