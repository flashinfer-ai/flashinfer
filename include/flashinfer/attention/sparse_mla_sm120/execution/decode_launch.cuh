// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "../kernels/dsv41_bf16/decode.cuh"
#include "../kernels/dsv41_fp8/decode_schedule.cuh"
#include "../kernels/fp8_decode/decode.cuh"
#include "attention_params.cuh"
#include "attention_plan.h"
#include "launch_validation.cuh"
#include "merge.cuh"

namespace flashinfer::sparse_mla_sm120::execution {

template <ModelType MT, int NUM_HEADS, int PAGE_BLOCK_SIZE>
cudaError_t launch_decode(const AttentionParams& p, const ExecutionPlan& plan,
                          cudaStream_t stream) {
  using KV = KVCacheTraits<MT>;
  const int h_blocks = plan.head_blocks;
  const int q_heads = (NUM_HEADS == 0) ? p.num_heads : NUM_HEADS;
  const int smem_bytes = plan.shared_bytes;
  const int block_threads = plan.block_threads;
  auto kernel = sparse_mla_decode_dsv4_kernel<MT, NUM_HEADS, PAGE_BLOCK_SIZE>;
  if constexpr (MT == ModelType::DSV4_1) {
    if (plan.implementation == Implementation::MixedCache) {
      kernel = sparse_mla_decode_dsv4_kernel<MT, NUM_HEADS, PAGE_BLOCK_SIZE,
                                             Dsv41MixedCacheDecodeSchedule>;
      cudaError_t result = validate_mixed_smem(kernel, smem_bytes);
      if (result != cudaSuccess) return result;
    }
    if (plan.implementation == Implementation::FullBF16) {
      kernel = plan.metadata.extra_fp4 ? sparse_mla_decode_dsv41_bf16_kernel<NUM_HEADS, true>
                                       : sparse_mla_decode_dsv41_bf16_kernel<NUM_HEADS, false>;
    }
  }
  cudaError_t result =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  if (result != cudaSuccess) return result;
  const int cpb = plan.cpb;
  kernel<<<dim3(p.num_tokens, h_blocks, p.allocated_splits), dim3(block_threads), smem_bytes,
           stream>>>(p.q, p.kv, p.indices, p.mid_out, p.mid_lse, p.topk_length, p.extra_kv,
                     p.extra_indices, p.extra_topk_length, p.extra_topk, p.extra_page_size,
                     p.extra_page_stride_bytes, p.num_tokens, q_heads, p.topk, p.allocated_splits,
                     cpb, p.sm_scale, p.page_stride_bytes, p.indices_stride_elems,
                     p.extra_indices_stride_elems, p.page_size);
  result = cudaGetLastError();
  if (result != cudaSuccess) return result;
  constexpr int MERGE_THREADS = 64;
  auto merge = sparse_mla_decode_dsv4_merge_kernel<NUM_HEADS, KV::D_V, MERGE_THREADS,
                                                   KV::D_V / MERGE_THREADS>;
  merge<<<dim3(p.num_tokens, q_heads), dim3(MERGE_THREADS),
          size_t(p.allocated_splits) * sizeof(float), stream>>>(
      p.mid_out, p.mid_lse, p.output, p.out_lse, p.attn_sink, p.num_tokens, p.allocated_splits,
      q_heads, h_blocks * HPB, p.out_lse_stride_elems);
  return cudaGetLastError();
}

}  // namespace flashinfer::sparse_mla_sm120::execution
