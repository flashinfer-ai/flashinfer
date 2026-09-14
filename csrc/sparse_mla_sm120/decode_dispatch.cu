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
  if (mt == ModelType::DSV4)
    // DSV4 instantiates its main page sizes (SPARSE_MLA_DSV4_MAIN_PAGES) as
    // separate kernels so the page divisor stays a compile-time constant.
    return execution::visit_decode_heads<ModelType::DSV4>(
        plan.specialized_heads, [&](auto head) -> cudaError_t {
          constexpr int H = decltype(head)::value;
          if (params.page_size == 32)
            return execution::launch_decode<ModelType::DSV4, H, 32>(params, plan, stream);
          if (params.page_size == execution::FixedPageSize)
            return execution::launch_decode<ModelType::DSV4, H, execution::FixedPageSize>(
                params, plan, stream);
          return cudaErrorInvalidValue;
        });
  MODEL(DOTS3_SWA)
  MODEL(DSV4_1)
#undef MODEL
  return cudaErrorInvalidValue;
}

}  // namespace flashinfer::sparse_mla_sm120
