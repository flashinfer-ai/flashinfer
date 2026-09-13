// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "attention_dispatch.h"
#include "attention_validation.h"

using tvm::ffi::Optional;

namespace flashinfer::sparse_mla_sm120 {
namespace execution {
int64_t resolve_format(int64_t query_dim, ffi::String scale);
}

void SparseMlaSm120PagedAttention(TensorView q, TensorView kv_cache, TensorView indices,
                                  TensorView output, TensorView out_lse, double sm_scale,
                                  int64_t model_type, int64_t variant,
                                  Optional<TensorView> topk_length, Optional<TensorView> attn_sink,
                                  Optional<TensorView> extra_kv_cache,
                                  Optional<TensorView> extra_indices,
                                  Optional<TensorView> extra_topk_length, bool extra_fp4) {
  CHECK_DIM(3, q);
  if (model_type == -1) model_type = execution::resolve_format(q.size(2), "auto");
  TVM_FFI_ICHECK(variant >= int(PrefillVariant::SG) && variant <= int(PrefillVariant::SWAPAB))
      << "variant must be a PrefillVariant";
  auto metadata = execution::unpack_metadata(inspect_attention_metadata<false>(
      q, kv_cache, indices, output, topk_length, attn_sink, extra_kv_cache, extra_indices,
      extra_topk_length, out_lse, {}, {}, model_type, extra_fp4, 0));
  metadata.variant = variant;
  TVM_FFI_ICHECK(out_lse.size(0) == metadata.tokens && out_lse.size(1) == metadata.heads)
      << "out_lse shape mismatch";
  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  int sm_count = 0, max_shared = 0;
  TVM_FFI_ICHECK_EQ(
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, q.device().device_id),
      cudaSuccess);
  TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                           q.device().device_id),
                    cudaSuccess);
  const auto plan = execution::resolve_attention(metadata, execution::NumericRoute::QkBF16PvFP8, 1,
                                                 {sm_count, size_t(max_shared)});
  execution::AttentionParams params{};
  params.q = static_cast<const __nv_bfloat16*>(q.data_ptr());
  params.kv = static_cast<const uint8_t*>(kv_cache.data_ptr());
  params.indices = static_cast<const int32_t*>(indices.data_ptr());
  params.output = static_cast<__nv_bfloat16*>(output.data_ptr());
  params.out_lse = static_cast<float*>(out_lse.data_ptr());
  params.extra_kv = extra_kv_cache.has_value()
                        ? static_cast<const uint8_t*>(extra_kv_cache.value().data_ptr())
                        : nullptr;
  params.extra_indices = extra_indices.has_value()
                             ? static_cast<const int32_t*>(extra_indices.value().data_ptr())
                             : nullptr;
  params.topk_length =
      topk_length.has_value() ? static_cast<const int*>(topk_length.value().data_ptr()) : nullptr;
  params.extra_topk_length = extra_topk_length.has_value()
                                 ? static_cast<const int*>(extra_topk_length.value().data_ptr())
                                 : nullptr;
  params.attn_sink =
      attn_sink.has_value() ? static_cast<const float*>(attn_sink.value().data_ptr()) : nullptr;
  params.sm_scale = sm_scale;
  check_attention_alignment(q,
                            plan.numeric == execution::NumericRoute::QkBF16PvFP8            ? 2
                            : plan.implementation == execution::Implementation::SwapAB ? 8
                                                                                       : 16,
                            "q");
  check_attention_alignment(output, 16, "output");
  const auto result = dispatch_prefill(params, plan, get_stream(q.device()));
  TVM_FFI_ICHECK(result.supported) << "unsupported sparse-MLA prefill";
  TVM_FFI_ICHECK_EQ(result.error, cudaSuccess)
      << "sparse-MLA prefill: " << cudaGetErrorString(result.error);
}

}  // namespace flashinfer::sparse_mla_sm120
