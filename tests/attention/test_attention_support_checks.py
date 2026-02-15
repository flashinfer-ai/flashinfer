"""
Test file for Attention API support checks.

This file serves as a TODO list for support check implementations.
APIs with @pytest.mark.xfail need support checks to be implemented.
"""

import pytest

from flashinfer.decode import (
    BatchDecodeWithPagedKVCacheWrapper,
    CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    single_decode_with_kv_cache,
    trtllm_batch_decode_with_kv_cache,
    xqa_batch_decode_with_kv_cache,
)
from flashinfer.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    fmha_v2_prefill_deepseek,
    single_prefill_with_kv_cache,
    trtllm_batch_context_with_kv_cache,
    trtllm_ragged_attention_deepseek,
)
from flashinfer.cascade import (
    BatchDecodeWithSharedPrefixPagedKVCacheWrapper,
    BatchPrefillWithSharedPrefixPagedKVCacheWrapper,
    MultiLevelCascadeAttentionWrapper,
    merge_state,
    merge_state_in_place,
    merge_states,
)
from flashinfer.mla import (
    BatchMLAPagedAttentionWrapper,
    trtllm_batch_decode_with_kv_cache_mla,
    xqa_batch_decode_with_kv_cache_mla,
)
from flashinfer.sparse import (
    BlockSparseAttentionWrapper,
)
from flashinfer.xqa import xqa, xqa_mla
from flashinfer.page import (
    append_paged_kv_cache,
    append_paged_mla_kv_cache,
    get_batch_indices_positions,
)
from flashinfer.rope import (
    apply_llama31_rope,
    apply_llama31_rope_inplace,
    apply_llama31_rope_pos_ids,
    apply_llama31_rope_pos_ids_inplace,
    apply_rope,
    apply_rope_inplace,
    apply_rope_pos_ids,
    apply_rope_pos_ids_inplace,
    apply_rope_with_cos_sin_cache,
    apply_rope_with_cos_sin_cache_inplace,
)
from flashinfer.cudnn.decode import cudnn_batch_decode_with_kv_cache
from flashinfer.cudnn.prefill import cudnn_batch_prefill_with_kv_cache
from flashinfer.pod import PODWithPagedKVCacheWrapper, BatchPODWithPagedKVCacheWrapper


# Decode APIs
@pytest.mark.xfail(
    reason="TODO: Support checks for single_decode_with_kv_cache are not implemented"
)
def test_single_decode_with_kv_cache_support_checks():
    assert hasattr(single_decode_with_kv_cache, "is_compute_capability_supported")
    assert hasattr(single_decode_with_kv_cache, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for BatchDecodeWithPagedKVCacheWrapper are not implemented"
)
def test_batch_decode_with_paged_kv_cache_wrapper_support_checks():
    assert hasattr(
        BatchDecodeWithPagedKVCacheWrapper.run, "is_compute_capability_supported"
    )
    assert hasattr(BatchDecodeWithPagedKVCacheWrapper.run, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for CUDAGraphBatchDecodeWithPagedKVCacheWrapper are not implemented"
)
def test_cuda_graph_batch_decode_wrapper_support_checks():
    assert hasattr(
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper.run,
        "is_compute_capability_supported",
    )
    assert hasattr(
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper.run, "is_backend_supported"
    )


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_batch_decode_with_kv_cache are not implemented"
)
def test_trtllm_batch_decode_with_kv_cache_support_checks():
    assert hasattr(trtllm_batch_decode_with_kv_cache, "is_compute_capability_supported")
    assert hasattr(trtllm_batch_decode_with_kv_cache, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for xqa_batch_decode_with_kv_cache are not implemented"
)
def test_xqa_batch_decode_with_kv_cache_support_checks():
    assert hasattr(xqa_batch_decode_with_kv_cache, "is_compute_capability_supported")
    assert hasattr(xqa_batch_decode_with_kv_cache, "is_backend_supported")


# Prefill APIs
@pytest.mark.xfail(
    reason="TODO: Support checks for single_prefill_with_kv_cache are not implemented"
)
def test_single_prefill_with_kv_cache_support_checks():
    assert hasattr(single_prefill_with_kv_cache, "is_compute_capability_supported")
    assert hasattr(single_prefill_with_kv_cache, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for BatchPrefillWithPagedKVCacheWrapper are not implemented"
)
def test_batch_prefill_with_paged_kv_cache_wrapper_support_checks():
    assert hasattr(
        BatchPrefillWithPagedKVCacheWrapper.run, "is_compute_capability_supported"
    )
    assert hasattr(BatchPrefillWithPagedKVCacheWrapper.run, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for BatchPrefillWithRaggedKVCacheWrapper are not implemented"
)
def test_batch_prefill_with_ragged_kv_cache_wrapper_support_checks():
    assert hasattr(
        BatchPrefillWithRaggedKVCacheWrapper.run, "is_compute_capability_supported"
    )
    assert hasattr(BatchPrefillWithRaggedKVCacheWrapper.run, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_ragged_attention_deepseek are not implemented"
)
def test_trtllm_ragged_attention_deepseek_support_checks():
    assert hasattr(trtllm_ragged_attention_deepseek, "is_compute_capability_supported")
    assert hasattr(trtllm_ragged_attention_deepseek, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_batch_context_with_kv_cache are not implemented"
)
def test_trtllm_batch_context_with_kv_cache_support_checks():
    assert hasattr(
        trtllm_batch_context_with_kv_cache, "is_compute_capability_supported"
    )
    assert hasattr(trtllm_batch_context_with_kv_cache, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for fmha_v2_prefill_deepseek are not implemented"
)
def test_fmha_v2_prefill_deepseek_support_checks():
    assert hasattr(fmha_v2_prefill_deepseek, "is_compute_capability_supported")
    assert hasattr(fmha_v2_prefill_deepseek, "is_backend_supported")


# Cascade APIs
@pytest.mark.xfail(reason="TODO: Support checks for merge_state are not implemented")
def test_merge_state_support_checks():
    assert hasattr(merge_state, "is_compute_capability_supported")
    assert hasattr(merge_state, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for merge_state_in_place are not implemented"
)
def test_merge_state_in_place_support_checks():
    assert hasattr(merge_state_in_place, "is_compute_capability_supported")
    assert hasattr(merge_state_in_place, "is_backend_supported")


@pytest.mark.xfail(reason="TODO: Support checks for merge_states are not implemented")
def test_merge_states_support_checks():
    assert hasattr(merge_states, "is_compute_capability_supported")
    assert hasattr(merge_states, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for MultiLevelCascadeAttentionWrapper are not implemented"
)
def test_multi_level_cascade_wrapper_support_checks():
    assert hasattr(
        MultiLevelCascadeAttentionWrapper.run, "is_compute_capability_supported"
    )
    assert hasattr(MultiLevelCascadeAttentionWrapper.run, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for BatchDecodeWithSharedPrefixPagedKVCacheWrapper are not implemented"
)
def test_batch_decode_shared_prefix_wrapper_support_checks():
    assert hasattr(
        BatchDecodeWithSharedPrefixPagedKVCacheWrapper.forward,
        "is_compute_capability_supported",
    )
    assert hasattr(
        BatchDecodeWithSharedPrefixPagedKVCacheWrapper.forward, "is_backend_supported"
    )


@pytest.mark.xfail(
    reason="TODO: Support checks for BatchPrefillWithSharedPrefixPagedKVCacheWrapper are not implemented"
)
def test_batch_prefill_shared_prefix_wrapper_support_checks():
    assert hasattr(
        BatchPrefillWithSharedPrefixPagedKVCacheWrapper.forward,
        "is_compute_capability_supported",
    )
    assert hasattr(
        BatchPrefillWithSharedPrefixPagedKVCacheWrapper.forward, "is_backend_supported"
    )


# MLA APIs
@pytest.mark.xfail(
    reason="TODO: Support checks for BatchMLAPagedAttentionWrapper are not implemented"
)
def test_batch_decode_mla_wrapper_support_checks():
    assert hasattr(BatchMLAPagedAttentionWrapper.run, "is_compute_capability_supported")
    assert hasattr(BatchMLAPagedAttentionWrapper.run, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for trtllm_batch_decode_with_kv_cache_mla are not implemented"
)
def test_trtllm_batch_decode_mla_support_checks():
    assert hasattr(
        trtllm_batch_decode_with_kv_cache_mla, "is_compute_capability_supported"
    )
    assert hasattr(trtllm_batch_decode_with_kv_cache_mla, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for xqa_batch_decode_with_kv_cache_mla are not implemented"
)
def test_xqa_batch_decode_mla_support_checks():
    assert hasattr(
        xqa_batch_decode_with_kv_cache_mla, "is_compute_capability_supported"
    )
    assert hasattr(xqa_batch_decode_with_kv_cache_mla, "is_backend_supported")


# Sparse APIs
@pytest.mark.xfail(
    reason="TODO: Support checks for BlockSparseAttentionWrapper are not implemented"
)
def test_block_sparse_attention_wrapper_support_checks():
    assert hasattr(BlockSparseAttentionWrapper.run, "is_compute_capability_supported")
    assert hasattr(BlockSparseAttentionWrapper.run, "is_backend_supported")


# XQA APIs
@pytest.mark.xfail(reason="TODO: Support checks for xqa are not implemented")
def test_xqa_support_checks():
    assert hasattr(xqa, "is_compute_capability_supported")
    assert hasattr(xqa, "is_backend_supported")


@pytest.mark.xfail(reason="TODO: Support checks for xqa_mla are not implemented")
def test_xqa_mla_support_checks():
    assert hasattr(xqa_mla, "is_compute_capability_supported")
    assert hasattr(xqa_mla, "is_backend_supported")


# Page APIs
@pytest.mark.xfail(
    reason="TODO: Support checks for get_batch_indices_positions are not implemented"
)
def test_get_batch_indices_positions_support_checks():
    assert hasattr(get_batch_indices_positions, "is_compute_capability_supported")
    assert hasattr(get_batch_indices_positions, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for append_paged_mla_kv_cache are not implemented"
)
def test_append_paged_mla_kv_cache_support_checks():
    assert hasattr(append_paged_mla_kv_cache, "is_compute_capability_supported")
    assert hasattr(append_paged_mla_kv_cache, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for append_paged_kv_cache are not implemented"
)
def test_append_paged_kv_cache_support_checks():
    assert hasattr(append_paged_kv_cache, "is_compute_capability_supported")
    assert hasattr(append_paged_kv_cache, "is_backend_supported")


# RoPE APIs
@pytest.mark.xfail(
    reason="TODO: Support checks for apply_rope_inplace are not implemented"
)
def test_apply_rope_inplace_support_checks():
    assert hasattr(apply_rope_inplace, "is_compute_capability_supported")
    assert hasattr(apply_rope_inplace, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for apply_rope_pos_ids_inplace are not implemented"
)
def test_apply_rope_pos_ids_inplace_support_checks():
    assert hasattr(apply_rope_pos_ids_inplace, "is_compute_capability_supported")
    assert hasattr(apply_rope_pos_ids_inplace, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for apply_llama31_rope_inplace are not implemented"
)
def test_apply_llama31_rope_inplace_support_checks():
    assert hasattr(apply_llama31_rope_inplace, "is_compute_capability_supported")
    assert hasattr(apply_llama31_rope_inplace, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for apply_llama31_rope_pos_ids_inplace are not implemented"
)
def test_apply_llama31_rope_pos_ids_inplace_support_checks():
    assert hasattr(
        apply_llama31_rope_pos_ids_inplace, "is_compute_capability_supported"
    )
    assert hasattr(apply_llama31_rope_pos_ids_inplace, "is_backend_supported")


@pytest.mark.xfail(reason="TODO: Support checks for apply_rope are not implemented")
def test_apply_rope_support_checks():
    assert hasattr(apply_rope, "is_compute_capability_supported")
    assert hasattr(apply_rope, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for apply_rope_pos_ids are not implemented"
)
def test_apply_rope_pos_ids_support_checks():
    assert hasattr(apply_rope_pos_ids, "is_compute_capability_supported")
    assert hasattr(apply_rope_pos_ids, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for apply_llama31_rope are not implemented"
)
def test_apply_llama31_rope_support_checks():
    assert hasattr(apply_llama31_rope, "is_compute_capability_supported")
    assert hasattr(apply_llama31_rope, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for apply_llama31_rope_pos_ids are not implemented"
)
def test_apply_llama31_rope_pos_ids_support_checks():
    assert hasattr(apply_llama31_rope_pos_ids, "is_compute_capability_supported")
    assert hasattr(apply_llama31_rope_pos_ids, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for apply_rope_with_cos_sin_cache are not implemented"
)
def test_apply_rope_with_cos_sin_cache_support_checks():
    assert hasattr(apply_rope_with_cos_sin_cache, "is_compute_capability_supported")
    assert hasattr(apply_rope_with_cos_sin_cache, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for apply_rope_with_cos_sin_cache_inplace are not implemented"
)
def test_apply_rope_with_cos_sin_cache_inplace_support_checks():
    assert hasattr(
        apply_rope_with_cos_sin_cache_inplace, "is_compute_capability_supported"
    )
    assert hasattr(apply_rope_with_cos_sin_cache_inplace, "is_backend_supported")


# cuDNN APIs
@pytest.mark.xfail(
    reason="TODO: Support checks for cudnn_batch_decode_with_kv_cache are not implemented"
)
def test_cudnn_batch_decode_support_checks():
    assert hasattr(cudnn_batch_decode_with_kv_cache, "is_compute_capability_supported")
    assert hasattr(cudnn_batch_decode_with_kv_cache, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for cudnn_batch_prefill_with_kv_cache are not implemented"
)
def test_cudnn_batch_prefill_support_checks():
    assert hasattr(cudnn_batch_prefill_with_kv_cache, "is_compute_capability_supported")
    assert hasattr(cudnn_batch_prefill_with_kv_cache, "is_backend_supported")


# POD APIs
@pytest.mark.xfail(
    reason="TODO: Support checks for BatchPODWithPagedKVCacheWrapper are not implemented"
)
def test_pod_prefill_wrapper_support_checks():
    assert hasattr(
        BatchPODWithPagedKVCacheWrapper.run, "is_compute_capability_supported"
    )
    assert hasattr(BatchPODWithPagedKVCacheWrapper.run, "is_backend_supported")


@pytest.mark.xfail(
    reason="TODO: Support checks for PODWithPagedKVCacheWrapper are not implemented"
)
def test_pod_decode_wrapper_support_checks():
    assert hasattr(PODWithPagedKVCacheWrapper.run, "is_compute_capability_supported")
    assert hasattr(PODWithPagedKVCacheWrapper.run, "is_backend_supported")
