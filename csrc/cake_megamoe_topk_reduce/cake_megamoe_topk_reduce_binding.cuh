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

#pragma once

#ifndef CAKE_MEGAMOE_TOPK_REDUCE_BODY_FILE
#error "CAKE_MEGAMOE_TOPK_REDUCE_BODY_FILE must name the frozen generated body"
#endif
#ifndef CAKE_MEGAMOE_TOPK_REDUCE_KERNEL
#error "CAKE_MEGAMOE_TOPK_REDUCE_KERNEL must name the frozen kernel symbol"
#endif
#ifndef CAKE_MEGAMOE_TOPK_REDUCE_THREADS
#error "CAKE_MEGAMOE_TOPK_REDUCE_THREADS must describe the frozen thread count"
#endif
#ifndef CAKE_MEGAMOE_TOPK_REDUCE_SMEM_BYTES
#error "CAKE_MEGAMOE_TOPK_REDUCE_SMEM_BYTES must describe dynamic shared memory"
#endif

// The frozen body is a self-contained CUDA translation-unit fragment.  Keep
// its fixed-width types intact: rewriting names such as uint32_t here would
// make the generated vector-load code refer to undefined aliases.
#include CAKE_MEGAMOE_TOPK_REDUCE_BODY_FILE

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <utility>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace cake_megamoe_topk_reduce {

constexpr int32_t kTopK = 6;
constexpr int32_t kHiddenSize = 4096;
constexpr int32_t kGridCTAsPerToken = 4;
constexpr int32_t kRequiredAlignmentBytes = 128;

static_assert(CAKE_MEGAMOE_TOPK_REDUCE_THREADS == 128,
              "the frozen MegaMoE TopK reducer requires 128 threads");
static_assert(CAKE_MEGAMOE_TOPK_REDUCE_SMEM_BYTES == 0,
              "the frozen MegaMoE TopK reducer uses no dynamic shared memory");

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

inline void CheckTarget(int32_t device_id) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
  TVM_FFI_ICHECK(major == 10 && minor == 0)
      << "the frozen MegaMoE TopK reducer requires exact compute capability 10.0, got " << major
      << "." << minor;
}

inline std::pair<uintptr_t, uintptr_t> TensorByteRange(const TensorView& tensor,
                                                       const char* name) {
  const DLDataType dtype = tensor.dtype();
  const uint64_t bits = static_cast<uint64_t>(dtype.bits) * dtype.lanes;
  TVM_FFI_ICHECK(bits > 0 && bits % 8 == 0) << name << " has a non-byte dtype";
  const uint64_t bytes_per_element = bits / 8;
  const uint64_t elements = static_cast<uint64_t>(tensor.numel());
  TVM_FFI_ICHECK(elements <= std::numeric_limits<uint64_t>::max() / bytes_per_element)
      << name << " byte range overflows uint64";
  const uint64_t bytes = elements * bytes_per_element;
  const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  TVM_FFI_ICHECK(bytes <= std::numeric_limits<uintptr_t>::max() - begin)
      << name << " byte range overflows uintptr_t";
  return {begin, begin + static_cast<uintptr_t>(bytes)};
}

inline void CheckNoOverlap(const TensorView& lhs, const char* lhs_name, const TensorView& rhs,
                           const char* rhs_name) {
  const auto lhs_range = TensorByteRange(lhs, lhs_name);
  const auto rhs_range = TensorByteRange(rhs, rhs_name);
  TVM_FFI_ICHECK(lhs_range.first >= rhs_range.second || rhs_range.first >= lhs_range.second)
      << lhs_name << " must not overlap " << rhs_name
      << ": the frozen kernel uses __restrict__ pointers";
}

void Run(TensorView partials, TensorView out, int64_t num_tokens, int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  CHECK_CUDA(partials);
  const int32_t device_id = partials.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckTarget(device_id);

  CHECK_CUDA(out);
  CHECK_DEVICE(partials, out);
  CHECK_INPUT_TYPE(partials, dl_bfloat16);
  CHECK_INPUT_TYPE(out, dl_bfloat16);

  TVM_FFI_ICHECK(partials.ndim() == 3)
      << "partials must have shape [capacity, 6, 4096]";
  const int64_t capacity = partials.ndim() == 3 ? partials.size(0) : -1;
  TVM_FFI_ICHECK((capacity == 256 || capacity == 4096) && partials.size(1) == kTopK &&
                 partials.size(2) == kHiddenSize)
      << "partials must have shape [256 or 4096, 6, 4096]";
  CHECK_CONTIGUOUS(partials);

  TVM_FFI_ICHECK(out.ndim() == 2 && out.size(0) == capacity && out.size(1) == kHiddenSize)
      << "out must have shape [capacity, 4096] matching partials";
  CHECK_CONTIGUOUS(out);

  TVM_FFI_ICHECK(num_tokens >= 0 && num_tokens <= capacity)
      << "num_tokens must be in [0, capacity], got " << num_tokens;
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(partials.data_ptr()) % kRequiredAlignmentBytes == 0)
      << "partials must be 128-byte aligned";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % kRequiredAlignmentBytes == 0)
      << "out must be 128-byte aligned";
  CheckNoOverlap(partials, "partials", out, "out");

  if (num_tokens == 0) {
    return;
  }

  const dim3 grid(static_cast<uint32_t>(kGridCTAsPerToken * num_tokens), 1, 1);
  const dim3 block(CAKE_MEGAMOE_TOPK_REDUCE_THREADS, 1, 1);
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  CAKE_MEGAMOE_TOPK_REDUCE_KERNEL<<<grid, block, CAKE_MEGAMOE_TOPK_REDUCE_SMEM_BYTES, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(partials.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  CheckCuda(cudaGetLastError(), "frozen MegaMoE TopK-reduce launch");
}

}  // namespace cake_megamoe_topk_reduce
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::cake_megamoe_topk_reduce::Run);
