// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <flashinfer/attention/sparse_mla_sm120/execution/decode_launch.cuh>

#include "../tvm_ffi_utils.h"
#include "attention_dispatch.h"

namespace flashinfer::sparse_mla_sm120 {

cudaError_t dispatch_decode(const execution::AttentionParams& params,
                            const execution::ExecutionPlan& plan, cudaStream_t stream) {
  const auto mt = static_cast<ModelType>(plan.metadata.model);
#define MODEL(M)                                                                                \
  if (mt == ModelType::M)                                                                       \
    return execution::visit_decode_heads<ModelType::M>(plan.specialized_heads, [&](auto head) { \
      return execution::launch_decode<ModelType::M, decltype(head)::value,                      \
                                      execution::FixedPageSize>(params, plan, stream);          \
    });
  MODEL(DSV4)
  MODEL(DOTS3_SWA) MODEL(DSV4_1)
#undef MODEL
      return cudaErrorInvalidValue;
}

}  // namespace flashinfer::sparse_mla_sm120
