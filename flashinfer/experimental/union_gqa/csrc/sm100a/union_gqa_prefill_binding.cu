/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cmath>
#include <cstdint>
#include <limits>

#include "tvm_ffi_utils.h"
#include "union_gqa_prefill_api.cuh"

namespace flashinfer::union_gqa {

namespace {

/** @brief Require compute capability 10.0 and report CUDA attribute-query failures. */
void CheckSM100(int device_id) {
  int major = 0;
  int minor = 0;
  cudaError_t status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaDeviceGetAttribute(major) failed: " << cudaGetErrorString(status);
  status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaDeviceGetAttribute(minor) failed: " << cudaGetErrorString(status);
  TVM_FFI_CHECK(major == 10 && minor == 0, RuntimeError)
      << "union_gqa_prefill requires compute capability 10.0, got " << major << "." << minor;
}

/** @brief Require a tensor to share Q's device type and device index. */
void CheckSameDevice(const TensorView& tensor, const TensorView& q, const char* name) {
  TVM_FFI_CHECK(tensor.device().device_type == q.device().device_type &&
                    tensor.device().device_id == q.device().device_id,
                ValueError)
      << name << " must be on cuda:" << q.device().device_id;
}

/** @brief Require the 16-byte base-pointer alignment used by TMA transfers. */
void CheckTmaAlignment(const TensorView& tensor, const char* name) {
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0, ValueError)
      << name << " must be 16-byte aligned";
}

/** @brief Return the storage size in bytes for a contiguous tensor view. */
uint64_t TensorBytes(const TensorView& tensor) {
  uint64_t elements = 1;
  for (int i = 0; i < tensor.ndim(); ++i) {
    elements *= static_cast<uint64_t>(tensor.size(i));
  }
  return elements * tensor.dtype().bits * tensor.dtype().lanes / 8;
}

/** @brief Test whether two contiguous tensor views have overlapping storage ranges. */
bool Overlaps(const TensorView& lhs, const TensorView& rhs) {
  const auto lhs_start = reinterpret_cast<uintptr_t>(lhs.data_ptr());
  const auto rhs_start = reinterpret_cast<uintptr_t>(rhs.data_ptr());
  return lhs_start <= rhs_start ? rhs_start - lhs_start < TensorBytes(lhs)
                                : lhs_start - rhs_start < TensorBytes(rhs);
}

}  // namespace

/**
 * @brief Validate Union-GQA tensor metadata and enqueue prefill on Q's current stream.
 *
 * Inputs must use contiguous storage, BF16 Q/K/V and output, int32 block indices and
 * cumulative sequence lengths, a head dimension of 128, and 16 query heads per KV head.
 * The output must not overlap any input. Tensor values are not inspected on the host.
 */
void Run(TensorView q, TensorView k, TensorView v, TensorView block_indices,
         TensorView cu_seqlens_q, TensorView out, double softmax_scale, int64_t max_seqlen_q) {
  CHECK_DIM(3, q);
  CHECK_DIM(3, k);
  CHECK_DIM(3, v);
  CHECK_DIM(3, block_indices);
  CHECK_DIM(1, cu_seqlens_q);
  CHECK_DIM(3, out);

  CHECK_INPUT_AND_TYPE(q, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(k, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(v, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(block_indices, dl_int32);
  CHECK_INPUT_AND_TYPE(cu_seqlens_q, dl_int32);
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  CheckSM100(q.device().device_id);
  CheckSameDevice(k, q, "k");
  CheckSameDevice(v, q, "v");
  CheckSameDevice(block_indices, q, "block_indices");
  CheckSameDevice(cu_seqlens_q, q, "cu_seqlens_q");
  CheckSameDevice(out, q, "out");

  const int64_t total_q = q.size(0);
  const int64_t num_qo_heads = q.size(1);
  const int64_t num_kv_heads = k.size(1);
  const int64_t topk_blocks = block_indices.size(2);
  const int64_t batch_size = cu_seqlens_q.size(0) - 1;

  TVM_FFI_CHECK(total_q > 0 && total_q <= std::numeric_limits<int>::max(), ValueError)
      << "total_q must fit int32 and be positive, got " << total_q;
  TVM_FFI_CHECK(num_kv_heads > 0 && num_qo_heads == 16 * num_kv_heads, ValueError)
      << "union_gqa_prefill requires num_qo_heads / num_kv_heads == 16, got " << num_qo_heads
      << " / " << num_kv_heads;
  TVM_FFI_CHECK(num_qo_heads <= std::numeric_limits<int>::max() / 128 &&
                    num_kv_heads <= std::numeric_limits<int>::max() / 128,
                ValueError)
      << "head counts are too large for int32 tensor strides";
  TVM_FFI_CHECK(q.size(2) == 128 && k.size(2) == 128 && v.size(2) == 128, ValueError)
      << "q/k/v head dimensions must all be 128";
  TVM_FFI_CHECK(k.size(0) == total_q && v.size(0) == total_q && v.size(1) == num_kv_heads,
                ValueError)
      << "flat self-attention requires k/v shapes (total_q, num_kv_heads, 128)";
  TVM_FFI_CHECK(out.size(0) == total_q && out.size(1) == num_qo_heads && out.size(2) == 128,
                ValueError)
      << "out must have shape (total_q, num_qo_heads, 128)";
  TVM_FFI_CHECK(block_indices.size(0) == total_q && block_indices.size(1) == num_kv_heads,
                ValueError)
      << "block_indices must have shape (total_q, num_kv_heads, topk_blocks)";
  TVM_FFI_CHECK(topk_blocks > 0 && topk_blocks <= 16, ValueError)
      << "topk_blocks must be in [1, 16], got " << topk_blocks;
  TVM_FFI_CHECK(batch_size > 0 && batch_size <= std::numeric_limits<int>::max(), ValueError)
      << "cu_seqlens_q must contain at least two entries";
  TVM_FFI_CHECK(max_seqlen_q > 0 && max_seqlen_q <= total_q &&
                    max_seqlen_q <= std::numeric_limits<int>::max(),
                ValueError)
      << "max_seqlen_q must be in [1, total_q], got " << max_seqlen_q;
  const int64_t max_q_tiles = (max_seqlen_q + 15) / 16;
  TVM_FFI_CHECK(max_q_tiles <= std::numeric_limits<int>::max() / num_kv_heads / batch_size,
                ValueError)
      << "union_gqa_prefill work count exceeds int32";
  const float kernel_scale = static_cast<float>(softmax_scale);
  TVM_FFI_CHECK(kernel_scale >= std::numeric_limits<float>::min() &&
                    std::isfinite(kernel_scale * 1.4426950408889634f),
                ValueError)
      << "softmax_scale is outside the supported float32 range";
  TVM_FFI_CHECK(!Overlaps(out, q) && !Overlaps(out, k) && !Overlaps(out, v) &&
                    !Overlaps(out, block_indices) && !Overlaps(out, cu_seqlens_q),
                ValueError)
      << "out must not overlap any input tensor";
  CheckTmaAlignment(q, "q");
  CheckTmaAlignment(k, "k");
  CheckTmaAlignment(v, "v");
  CheckTmaAlignment(out, "out");

  cudaError_t status = LaunchUnionGQAPrefill(
      q.data_ptr(), k.data_ptr(), v.data_ptr(), static_cast<int32_t*>(block_indices.data_ptr()),
      static_cast<int32_t*>(cu_seqlens_q.data_ptr()), out.data_ptr(), static_cast<int>(total_q),
      static_cast<int>(num_qo_heads), static_cast<int>(num_kv_heads), static_cast<int>(topk_blocks),
      static_cast<int>(batch_size), static_cast<int>(max_seqlen_q), kernel_scale,
      get_current_stream());
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "union_gqa_prefill launch failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer::union_gqa

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::union_gqa::Run);
