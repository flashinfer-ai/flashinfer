// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda_runtime.h>
#include <flashinfer/attention/sparse_mla_sm120/execution/attention_plan.h>
#include <flashinfer/attention/sparse_mla_sm120/execution/prefill_result.h>

#include <flashinfer/attention/sparse_mla_sm120/execution/attention_params.cuh>

namespace flashinfer::sparse_mla_sm120 {
PrefillLaunchResult dispatch_prefill(const execution::AttentionParams& params,
                                     const execution::ExecutionPlan& plan, cudaStream_t stream);
cudaError_t dispatch_dsv32(const execution::AttentionParams& params,
                           const execution::ExecutionPlan& plan, cudaStream_t stream);
cudaError_t dispatch_decode(const execution::AttentionParams& params,
                            const execution::ExecutionPlan& plan, cudaStream_t stream);
}  // namespace flashinfer::sparse_mla_sm120

namespace flashinfer::sparse_mla_sm120::nvfp4 {

#define SPARSE_MLA_DSV4_NVFP4_INSTANCES(F) \
  F(16, 128)                               \
  F(16, 512) F(32, 128) F(32, 512) F(64, 128) F(64, 512) F(128, 128) F(128, 512)

inline bool has_instance(int heads, int topk) {
#define MATCH(H, K) \
  if (heads == H && topk == K) return true;
  SPARSE_MLA_DSV4_NVFP4_INSTANCES(MATCH)
#undef MATCH
  return false;
}

cudaError_t dispatch_attention(const Dsv4Nvfp4AttentionParams& params,
                               const execution::ExecutionPlan& plan, cudaStream_t stream);

}  // namespace flashinfer::sparse_mla_sm120::nvfp4
