/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>

#include <flashinfer/attention/sparse_mla_sm120/streaming_dsv4_nvfp4_kernel.cuh>

#include "sparse_mla_sm120_nvfp4_common.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace flashinfer::sparse_mla_sm120::nvfp4 {

#define NVFP4_PREFILL_CUDA_CHECK(call)                                    \
  do {                                                                    \
    const cudaError_t status = (call);                                    \
    TVM_FFI_ICHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status); \
  } while (0)

namespace {

template <int NUM_HEADS, int TOPK, int PAGE_SIZE, bool DUAL_CACHE>
void launch_prefill(const bf16* q, const uint8_t* cache, const int32_t* indices, bf16* output,
                    float* out_lse, const int* topk_length, const float* attn_sink,
                    const uint8_t* extra_cache, const int32_t* extra_indices,
                    const int* extra_topk_length, int extra_topk, int extra_page_size,
                    size_t extra_page_stride, int num_tokens, float sm_scale, size_t page_stride,
                    cudaStream_t stream) {
  constexpr int HEADS_PER_CTA =
      NUM_HEADS < STREAMING_HEADS_PER_CTA ? NUM_HEADS : STREAMING_HEADS_PER_CTA;
  constexpr int HEAD_BLOCKS = NUM_HEADS / HEADS_PER_CTA;
  constexpr size_t DYN_SMEM_BYTES = StreamingNVFP4Smem::SIZE;
  auto kernel = sparse_mla_streaming_dsv4_nvfp4_kernel<NUM_HEADS, TOPK, PAGE_SIZE, DUAL_CACHE>;
  NVFP4_PREFILL_CUDA_CHECK(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM_BYTES));

  kernel<<<dim3(num_tokens, HEAD_BLOCKS), dim3(STREAMING_BLOCK_THREADS), DYN_SMEM_BYTES, stream>>>(
      q, cache, indices, output, out_lse, nullptr, nullptr, attn_sink, topk_length, extra_cache,
      extra_indices, extra_topk_length, extra_topk, extra_page_size, extra_page_stride, num_tokens,
      /*num_splits=*/1,
      /*chunks_per_block=*/0, sm_scale, page_stride, /*write_direct=*/true);
  NVFP4_PREFILL_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void SparseMlaSm120NVFP4Prefill(TensorView q, TensorView kv_cache, TensorView indices,
                                TensorView output, TensorView out_lse, double sm_scale,
                                Optional<TensorView> topk_length, Optional<TensorView> attn_sink,
                                Optional<TensorView> extra_kv_cache,
                                Optional<TensorView> extra_indices,
                                Optional<TensorView> extra_topk_length) {
  CHECK_INPUT_AND_TYPE(q, dl_bfloat16);
  CHECK_CUDA(kv_cache);
  CHECK_INPUT_TYPE(kv_cache, dl_uint8);
  CHECK_INPUT_AND_TYPE(indices, dl_int32);
  CHECK_INPUT_AND_TYPE(output, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(out_lse, dl_float32);
  TVM_FFI_ICHECK_EQ(q.ndim(), 3);
  TVM_FFI_ICHECK_EQ(q.size(2), 512);
  TVM_FFI_ICHECK_EQ(indices.ndim(), 2);
  TVM_FFI_ICHECK(indices.IsContiguous());
  TVM_FFI_ICHECK(output.IsContiguous());
  TVM_FFI_ICHECK(out_lse.IsContiguous());

  CHECK_DEVICE(q, kv_cache);
  CHECK_DEVICE(q, indices);
  CHECK_DEVICE(q, output);
  CHECK_DEVICE(q, out_lse);

  const int num_tokens = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int topk = static_cast<int>(indices.size(1));
  TVM_FFI_ICHECK_GT(topk, 0);
  const bool has_extra = extra_kv_cache.has_value();
  TVM_FFI_ICHECK_EQ(has_extra, extra_indices.has_value())
      << "extra_kv_cache and extra_indices must be provided together";
  TVM_FFI_ICHECK(!extra_topk_length.has_value() || has_extra)
      << "extra_topk_length requires an extra cache";

  int extra_topk = 0;
  PagedLayout extra_layout{0, 0, 0};
  const uint8_t* extra_cache_ptr = nullptr;
  const int32_t* extra_indices_ptr = nullptr;
  const int* extra_topk_length_ptr = nullptr;
  if (has_extra) {
    const auto& extra_cache = extra_kv_cache.value();
    const auto& extra_idx = extra_indices.value();
    CHECK_CUDA(extra_cache);
    CHECK_INPUT_TYPE(extra_cache, dl_uint8);
    CHECK_INPUT_AND_TYPE(extra_idx, dl_int32);
    CHECK_DEVICE(q, extra_cache);
    CHECK_DEVICE(q, extra_idx);
    TVM_FFI_ICHECK_EQ(extra_idx.ndim(), 2);
    TVM_FFI_ICHECK(extra_idx.IsContiguous());
    TVM_FFI_ICHECK_EQ(extra_idx.size(0), num_tokens);
    extra_topk = static_cast<int>(extra_idx.size(1));
    TVM_FFI_ICHECK_GT(extra_topk, 0);
    extra_layout = parse_nvfp4_paged_layout(extra_cache);
    TVM_FFI_ICHECK(extra_layout.page_size == 2 || extra_layout.page_size == 64)
        << "NVFP4 extra cache page_size must be 2 or 64";
    extra_cache_ptr = static_cast<const uint8_t*>(extra_cache.data_ptr());
    extra_indices_ptr = static_cast<const int32_t*>(extra_idx.data_ptr());
    if (extra_topk_length.has_value()) {
      const auto& length = extra_topk_length.value();
      CHECK_INPUT_AND_TYPE(length, dl_int32);
      CHECK_DEVICE(q, length);
      TVM_FFI_ICHECK_EQ(length.ndim(), 1);
      TVM_FFI_ICHECK_EQ(length.size(0), num_tokens);
      extra_topk_length_ptr = static_cast<const int*>(length.data_ptr());
    }
  }
  TVM_FFI_ICHECK_EQ(indices.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(output.ndim(), 3);
  TVM_FFI_ICHECK_EQ(output.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(output.size(1), num_heads);
  TVM_FFI_ICHECK_EQ(output.size(2), 512);
  TVM_FFI_ICHECK_EQ(out_lse.ndim(), 2);
  TVM_FFI_ICHECK_EQ(out_lse.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(out_lse.size(1), num_heads);

  if (topk_length.has_value()) {
    const auto& length = topk_length.value();
    CHECK_INPUT_AND_TYPE(length, dl_int32);
    CHECK_DEVICE(q, length);
    TVM_FFI_ICHECK_EQ(length.ndim(), 1);
    TVM_FFI_ICHECK_EQ(length.size(0), num_tokens);
  }
  if (attn_sink.has_value()) {
    const auto& sink = attn_sink.value();
    CHECK_INPUT_AND_TYPE(sink, dl_float32);
    CHECK_DEVICE(q, sink);
    TVM_FFI_ICHECK_EQ(sink.ndim(), 1);
    TVM_FFI_ICHECK_EQ(sink.size(0), num_heads);
  }

  const PagedLayout layout = parse_nvfp4_paged_layout(kv_cache);
  TVM_FFI_ICHECK_EQ(layout.page_size, 64) << "initial NVFP4 prefill supports page_size=64";
  const int* topk_length_ptr =
      topk_length.has_value() ? static_cast<const int*>(topk_length.value().data_ptr()) : nullptr;
  const float* attn_sink_ptr =
      attn_sink.has_value() ? static_cast<const float*>(attn_sink.value().data_ptr()) : nullptr;

  if (num_tokens == 0) return;

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
#define DISPATCH_NVFP4_PREFILL(H, K)                                                              \
  if (num_heads == (H) && topk == (K)) {                                                          \
    if (has_extra) {                                                                              \
      launch_prefill<(H), (K), 64, true>(                                                         \
          static_cast<const bf16*>(q.data_ptr()),                                                 \
          static_cast<const uint8_t*>(kv_cache.data_ptr()),                                       \
          static_cast<const int32_t*>(indices.data_ptr()), static_cast<bf16*>(output.data_ptr()), \
          static_cast<float*>(out_lse.data_ptr()), topk_length_ptr, attn_sink_ptr,                \
          extra_cache_ptr, extra_indices_ptr, extra_topk_length_ptr, extra_topk,                  \
          extra_layout.page_size, extra_layout.page_stride_bytes, num_tokens,                     \
          static_cast<float>(sm_scale), layout.page_stride_bytes, stream);                        \
    } else {                                                                                      \
      launch_prefill<(H), (K), 64, false>(                                                        \
          static_cast<const bf16*>(q.data_ptr()),                                                 \
          static_cast<const uint8_t*>(kv_cache.data_ptr()),                                       \
          static_cast<const int32_t*>(indices.data_ptr()), static_cast<bf16*>(output.data_ptr()), \
          static_cast<float*>(out_lse.data_ptr()), topk_length_ptr, attn_sink_ptr, nullptr,       \
          nullptr, nullptr, 0, 0, 0, num_tokens, static_cast<float>(sm_scale),                    \
          layout.page_stride_bytes, stream);                                                      \
    }                                                                                             \
    return;                                                                                       \
  }
  DISPATCH_NVFP4_PREFILL(16, 128)
  DISPATCH_NVFP4_PREFILL(16, 512)
  DISPATCH_NVFP4_PREFILL(32, 128)
  DISPATCH_NVFP4_PREFILL(32, 512)
  DISPATCH_NVFP4_PREFILL(64, 128)
  DISPATCH_NVFP4_PREFILL(64, 512)
  DISPATCH_NVFP4_PREFILL(128, 128)
  DISPATCH_NVFP4_PREFILL(128, 512)
#undef DISPATCH_NVFP4_PREFILL
  TVM_FFI_ICHECK(false) << "unsupported initial NVFP4 prefill shape: heads=" << num_heads
                        << ", topk=" << topk;
}

}  // namespace flashinfer::sparse_mla_sm120::nvfp4

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_nvfp4_prefill,
                              flashinfer::sparse_mla_sm120::nvfp4::SparseMlaSm120NVFP4Prefill);

#undef NVFP4_PREFILL_CUDA_CHECK
