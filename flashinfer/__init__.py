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

import importlib.util

from .version import __version__ as __version__
from .version import __git_version__ as __git_version__


# JIT compilation support
from . import jit as jit

# ============================================================================
# Activation functions
# ============================================================================
try:
    from .activation import gelu_and_mul as gelu_and_mul
    from .activation import gelu_tanh_and_mul as gelu_tanh_and_mul
    from .activation import silu_and_mul as silu_and_mul
    from .activation import (
        silu_and_mul_scaled_nvfp4_experts_quantize as silu_and_mul_scaled_nvfp4_experts_quantize,
    )
except ImportError:
    pass

# ============================================================================
# Attention modules
# ============================================================================
try:
    from .attention import BatchAttention as BatchAttention
    from .attention import (
        BatchAttentionWithAttentionSinkWrapper as BatchAttentionWithAttentionSinkWrapper,
    )
except ImportError:
    pass

# ============================================================================
# Autotuner
# ============================================================================
try:
    from .autotuner import autotune as autotune
except ImportError:
    pass

# ============================================================================
# Cascade attention
# ============================================================================
try:
    from .cascade import (
        BatchDecodeWithSharedPrefixPagedKVCacheWrapper as BatchDecodeWithSharedPrefixPagedKVCacheWrapper,
    )
    from .cascade import (
        BatchPrefillWithSharedPrefixPagedKVCacheWrapper as BatchPrefillWithSharedPrefixPagedKVCacheWrapper,
    )
    from .cascade import (
        MultiLevelCascadeAttentionWrapper as MultiLevelCascadeAttentionWrapper,
    )
    from .cascade import merge_state as merge_state
    from .cascade import merge_state_in_place as merge_state_in_place
    from .cascade import merge_states as merge_states
except ImportError:
    pass

# ============================================================================
# Decode operations
# ============================================================================
try:
    from .decode import (
        BatchDecodeMlaWithPagedKVCacheWrapper as BatchDecodeMlaWithPagedKVCacheWrapper,
    )
    from .decode import (
        BatchDecodeWithPagedKVCacheWrapper as BatchDecodeWithPagedKVCacheWrapper,
    )
    from .decode import (
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper as CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )
    from .decode import cudnn_batch_decode_with_kv_cache as cudnn_batch_decode_with_kv_cache
    from .decode import fast_decode_plan as fast_decode_plan
    from .decode import single_decode_with_kv_cache as single_decode_with_kv_cache
except ImportError:
    pass

# ============================================================================
# FP4 quantization
# ============================================================================
try:
    from .fp4_quantization import (
        SfLayout,
        block_scale_interleave,
        nvfp4_block_scale_interleave,
        e2m1_and_ufp8sf_scale_to_float,
        fp4_quantize,
        mxfp4_dequantize_host,
        mxfp4_dequantize,
        mxfp4_quantize,
        nvfp4_quantize,
        nvfp4_batched_quantize,
        shuffle_matrix_a,
        shuffle_matrix_sf_a,
        scaled_fp4_grouped_quantize,
    )
except ImportError:
    pass

# ============================================================================
# FP8 quantization
# ============================================================================
try:
    from .fp8_quantization import mxfp8_dequantize_host, mxfp8_quantize
except ImportError:
    pass

# ============================================================================
# Fused mixture-of-experts (MoE)
# ============================================================================
try:
    from .fused_moe import (
        RoutingMethodType,
        GatedActType,
        cutlass_fused_moe,
        reorder_rows_for_gated_act_gemm,
        trtllm_fp4_block_scale_moe,
        trtllm_fp4_block_scale_routed_moe,
        trtllm_fp8_block_scale_moe,
        trtllm_fp8_per_tensor_scale_moe,
    )
except ImportError:
    pass

# ============================================================================
# GEMM operations
# ============================================================================
try:
    from .gemm import SegmentGEMMWrapper as SegmentGEMMWrapper
    from .gemm import bmm_fp8 as bmm_fp8
    from .gemm import mm_fp4 as mm_fp4
    from .gemm import mm_fp8 as mm_fp8
    from .gemm import tgv_gemm_sm100 as tgv_gemm_sm100
except ImportError:
    pass

# ============================================================================
# Multi-latent attention (MLA)
# ============================================================================
try:
    from .mla import BatchMLAPagedAttentionWrapper as BatchMLAPagedAttentionWrapper
except ImportError:
    pass

# ============================================================================
# Normalization operations
# ============================================================================
try:
    from .norm import fused_add_rmsnorm as fused_add_rmsnorm
    from .norm import layernorm as layernorm
    from .norm import gemma_fused_add_rmsnorm as gemma_fused_add_rmsnorm
    from .norm import gemma_rmsnorm as gemma_rmsnorm
    from .norm import rmsnorm as rmsnorm
except ImportError:
    pass

# ============================================================================
# Paged KV cache operations
# ============================================================================
try:
    from .page import append_paged_kv_cache as append_paged_kv_cache
    from .page import append_paged_mla_kv_cache as append_paged_mla_kv_cache
    from .page import get_batch_indices_positions as get_batch_indices_positions
    from .page import get_seq_lens as get_seq_lens
except ImportError:
    pass

# ============================================================================
# POD (Persistent Output Decoding)
# ============================================================================
try:
    from .pod import PODWithPagedKVCacheWrapper as PODWithPagedKVCacheWrapper
except ImportError:
    pass


# ============================================================================
# Prefill operations
# ============================================================================
try:
    from .prefill import (
        BatchPrefillWithPagedKVCacheWrapper as BatchPrefillWithPagedKVCacheWrapper,
    )
    from .prefill import (
        BatchPrefillWithRaggedKVCacheWrapper as BatchPrefillWithRaggedKVCacheWrapper,
    )
    from .prefill import single_prefill_with_kv_cache as single_prefill_with_kv_cache
    from .prefill import (
        single_prefill_with_kv_cache_return_lse as single_prefill_with_kv_cache_return_lse,
    )
except ImportError:
    pass

# ============================================================================
# Quantization utilities
# ============================================================================
try:
    from .quantization import packbits as packbits
    from .quantization import segment_packbits as segment_packbits
except ImportError:
    pass

# ============================================================================
# RoPE (Rotary Position Embedding)
# ============================================================================
try:
    from .rope import apply_llama31_rope as apply_llama31_rope
    from .rope import apply_llama31_rope_inplace as apply_llama31_rope_inplace
    from .rope import apply_llama31_rope_pos_ids as apply_llama31_rope_pos_ids
    from .rope import (
        apply_llama31_rope_pos_ids_inplace as apply_llama31_rope_pos_ids_inplace,
    )
    from .rope import apply_rope as apply_rope
    from .rope import apply_rope_inplace as apply_rope_inplace
    from .rope import apply_rope_pos_ids as apply_rope_pos_ids
    from .rope import apply_rope_pos_ids_inplace as apply_rope_pos_ids_inplace
    from .rope import apply_rope_with_cos_sin_cache as apply_rope_with_cos_sin_cache
    from .rope import (
        apply_rope_with_cos_sin_cache_inplace as apply_rope_with_cos_sin_cache_inplace,
    )
except ImportError:
    pass

# ============================================================================
# Sampling operations
# ============================================================================
try:
    from .sampling import chain_speculative_sampling as chain_speculative_sampling
    from .sampling import min_p_sampling_from_probs as min_p_sampling_from_probs
    from .sampling import sampling_from_logits as sampling_from_logits
    from .sampling import sampling_from_probs as sampling_from_probs
    from .sampling import softmax as softmax
    from .sampling import top_k_mask_logits as top_k_mask_logits
    from .sampling import top_k_renorm_probs as top_k_renorm_probs
    from .sampling import top_k_sampling_from_probs as top_k_sampling_from_probs
    from .sampling import (
        top_k_top_p_sampling_from_logits as top_k_top_p_sampling_from_logits,
    )
    from .sampling import top_k_top_p_sampling_from_probs as top_k_top_p_sampling_from_probs
    from .sampling import top_p_renorm_probs as top_p_renorm_probs
    from .sampling import top_p_sampling_from_probs as top_p_sampling_from_probs
except ImportError:
    pass

# ============================================================================
# Sparse attention
# ============================================================================
try:
    from .sparse import BlockSparseAttentionWrapper as BlockSparseAttentionWrapper
    from .sparse import (
        VariableBlockSparseAttentionWrapper as VariableBlockSparseAttentionWrapper,
    )
except ImportError:
    pass

# ============================================================================
# TRT-LLM low-latency GEMM
# ============================================================================
try:
    from .trtllm_low_latency_gemm import (
        prepare_low_latency_gemm_weights as prepare_low_latency_gemm_weights,
    )
except ImportError:
    pass

# ============================================================================
# Utilities
# ============================================================================
try:
    from .utils import next_positive_power_of_2 as next_positive_power_of_2
except ImportError:
    pass

# ============================================================================
# XQA (Cross-Query Attention)
# ============================================================================
try:
    from .xqa import xqa as xqa
    from .xqa import xqa_mla as xqa_mla
except ImportError:
    pass
