// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../kernels/dsv4_nvfp4/decode.cuh"
#include "../kernels/dsv4_nvfp4/grouped_attention.cuh"
#include "merge.cuh"
#include "attention_plan.h"
#include "attention_params.cuh"

namespace flashinfer::sparse_mla_sm120::nvfp4 {

template <int NUM_HEADS, int TOPK, int PAGE_SIZE, bool DUAL_CACHE>
cudaError_t launch_decode(const Dsv4Nvfp4AttentionParams& params, const execution::ExecutionPlan& plan,
                          cudaStream_t stream) {
  const auto& [q, cache, indices, mid_out, mid_lse, output, out_lse, topk_length, attn_sink,
               extra_cache, extra_indices, extra_topk_length, extra_topk, extra_page_size,
               extra_page_stride_bytes, num_tokens, sm_scale, page_stride_bytes] = params;
  constexpr bool CAN_GROUP_HEADS = NUM_HEADS >= STREAMING_HEADS_PER_CTA;
  constexpr int GROUPED_H_BLOCKS =
      (NUM_HEADS + STREAMING_HEADS_PER_CTA - 1) / STREAMING_HEADS_PER_CTA;
  constexpr int UNGROUPED_H_BLOCKS = (NUM_HEADS + HPB - 1) / HPB;
  const bool use_grouped = plan.implementation == execution::Implementation::Dsv4Nvfp4GroupedDecode;
  const int chunks_per_block = plan.cpb;
  const int active_splits = plan.active_splits;
  const bool stage1_only = plan.merge == execution::Merge::Stage1;
  const bool write_direct = plan.merge == execution::Merge::Direct;
  if constexpr (CAN_GROUP_HEADS) {
    if (use_grouped) {
      constexpr size_t DYN_SMEM_BYTES = StreamingNVFP4Smem::SIZE;
      auto kernel = sparse_mla_streaming_dsv4_nvfp4_kernel<NUM_HEADS, TOPK, PAGE_SIZE, DUAL_CACHE>;
      auto status =
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM_BYTES);
      if (status != cudaSuccess) return status;
      kernel<<<dim3(num_tokens, GROUPED_H_BLOCKS, active_splits), dim3(STREAMING_BLOCK_THREADS),
               DYN_SMEM_BYTES, stream>>>(
          q, cache, indices, output, out_lse, mid_out, mid_lse, attn_sink, topk_length, extra_cache,
          extra_indices, extra_topk_length, extra_topk, extra_page_size, extra_page_stride_bytes,
          num_tokens, active_splits, chunks_per_block, sm_scale, page_stride_bytes, write_direct);
    }
  }
  if (!use_grouped) {
    constexpr size_t DYN_SMEM_BYTES = DecodeNVFP4Smem<ModelType::DSV4>::SIZE;
    auto kernel = sparse_mla_decode_dsv4_nvfp4_kernel<ModelType::DSV4, NUM_HEADS, TOPK, PAGE_SIZE,
                                                      DUAL_CACHE>;
    auto status =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM_BYTES);
    if (status != cudaSuccess) return status;
    kernel<<<dim3(num_tokens, UNGROUPED_H_BLOCKS, active_splits), dim3(DECODE_BLOCK_THREADS),
             DYN_SMEM_BYTES, stream>>>(
        q, cache, indices, mid_out, mid_lse, output, out_lse, attn_sink, topk_length, extra_cache,
        extra_indices, extra_topk_length, extra_topk, extra_page_size, extra_page_stride_bytes,
        num_tokens, active_splits, chunks_per_block, sm_scale, page_stride_bytes, write_direct);
  }
  auto status = cudaGetLastError();
  if (status != cudaSuccess || stage1_only || write_direct) return status;
  if (plan.merge == execution::Merge::Merge2) {
    constexpr int MERGE_H_BLOCKS = (NUM_HEADS + HPB - 1) / HPB;
    auto kernel = sparse_mla_decode_dsv4_nvfp4_merge2_kernel<NUM_HEADS>;
    kernel<<<dim3(num_tokens, MERGE_H_BLOCKS), dim3(DECODE_MERGE2_THREADS), 0, stream>>>(
        mid_out, mid_lse, output, out_lse, attn_sink, num_tokens);
    return cudaGetLastError();
  }
  constexpr int MERGE_THREADS = 64;
  constexpr int DIMS_PER_THREAD = 512 / MERGE_THREADS;
  auto kernel = sparse_mla_decode_dsv4_merge_kernel<NUM_HEADS, 512, MERGE_THREADS, DIMS_PER_THREAD>;
  const size_t merge_smem = static_cast<size_t>(active_splits) * sizeof(float);
  kernel<<<dim3(num_tokens, NUM_HEADS), dim3(MERGE_THREADS), merge_smem, stream>>>(
      mid_out, mid_lse, output, out_lse, attn_sink, num_tokens, active_splits, NUM_HEADS, NUM_HEADS,
      NUM_HEADS);
  return cudaGetLastError();
}

template <int NUM_HEADS, int TOPK, int PAGE_SIZE, bool DUAL_CACHE>
cudaError_t launch_prefill(const Dsv4Nvfp4AttentionParams& params, const execution::ExecutionPlan& plan,
                           cudaStream_t stream) {
  const auto& [q, cache, indices, mid_out, mid_lse, output, out_lse, topk_length, attn_sink,
               extra_cache, extra_indices, extra_topk_length, extra_topk, extra_page_size,
               extra_page_stride_bytes, num_tokens, sm_scale, page_stride_bytes] = params;
  const int HEAD_BLOCKS = plan.head_blocks;
  constexpr size_t DYN_SMEM_BYTES = StreamingNVFP4Smem::SIZE;
  auto kernel = sparse_mla_streaming_dsv4_nvfp4_kernel<NUM_HEADS, TOPK, PAGE_SIZE, DUAL_CACHE>;
  auto status =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM_BYTES);
  if (status != cudaSuccess) return status;
  kernel<<<dim3(num_tokens, HEAD_BLOCKS), dim3(STREAMING_BLOCK_THREADS), DYN_SMEM_BYTES, stream>>>(
      q, cache, indices, output, out_lse, nullptr, nullptr, attn_sink, topk_length, extra_cache,
      extra_indices, extra_topk_length, extra_topk, extra_page_size, extra_page_stride_bytes, num_tokens,
      plan.scratch_split_stride, plan.cpb, sm_scale, page_stride_bytes,
      plan.merge == execution::Merge::Direct);
  return cudaGetLastError();
}

}  // namespace flashinfer::sparse_mla_sm120::nvfp4
