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

// The generated body is a self-contained CUDA translation-unit fragment shared
// by every supported architecture; each module compiles it with its own
// -gencode and its own CAKE_MEGAMOE_TOPK_REDUCE_ARCH, which picks the token
// tiers below.  Keep its fixed-width types intact: rewriting names such as
// uint32_t here would make the generated vector-load code refer to undefined
// aliases.  The include stays ahead of the toolkit headers.
// clang-format off
#include "cake_megamoe_topk_reduce_kernels.cu"
// clang-format on

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
constexpr int32_t kRequiredAlignmentBytes = 128;

// The generated source holds the reducer in four CTA shapes.  Every thread runs
// the same program (six loads, FP32 adds in increasing k, one RN rounding), so
// all shapes write bit-identical rows; they differ in threads per CTA, BF16
// values per thread and the L2 prefetch size of the loads.
enum class CtaShape { k512x2, k512x4, k512x4Pf256, k1024x4Pf256 };

// The fastest shape depends on the live token count and the architecture: the
// module launches the first tier whose max_tokens covers num_tokens.
struct TokenTier {
  int64_t max_tokens;
  CtaShape shape;
};

constexpr int64_t kNoTokenLimit = std::numeric_limits<int64_t>::max();

#if CAKE_MEGAMOE_TOPK_REDUCE_ARCH == 100
constexpr TokenTier kTokenTiers[] = {
    {48, CtaShape::k512x2},
    {96, CtaShape::k512x4},
    {2048, CtaShape::k512x4Pf256},
    {kNoTokenLimit, CtaShape::k1024x4Pf256},
};
#elif CAKE_MEGAMOE_TOPK_REDUCE_ARCH == 103
constexpr TokenTier kTokenTiers[] = {
    {48, CtaShape::k512x2},        {96, CtaShape::k512x4},
    {192, CtaShape::k512x4Pf256},  {256, CtaShape::k1024x4Pf256},
    {1024, CtaShape::k512x4Pf256}, {kNoTokenLimit, CtaShape::k1024x4Pf256},
};
#else
#error "CAKE_MEGAMOE_TOPK_REDUCE_ARCH must be 100 (sm_100a) or 103 (sm_103a)"
#endif

inline CtaShape SelectShape(int64_t num_tokens) {
  for (const TokenTier& tier : kTokenTiers) {
    if (num_tokens <= tier.max_tokens) {
      return tier.shape;
    }
  }
  return CtaShape::k1024x4Pf256;
}

// A CTA of `threads` threads with `values` BF16 outputs each covers
// threads * values of a token's kHiddenSize outputs.
inline dim3 TokenGrid(int64_t num_tokens, int32_t threads, int32_t values) {
  return dim3(static_cast<uint32_t>(kHiddenSize / (threads * values) * num_tokens), 1, 1);
}

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

inline std::pair<uintptr_t, uintptr_t> TensorByteRange(const TensorView& tensor, const char* name) {
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
      << lhs_name << " must not overlap " << rhs_name << ": the kernel uses __restrict__ pointers";
}

// partials: [capacity, 6, 4096] BF16, out: [capacity, 4096] BF16.  The kernel
// reads and writes only the first num_tokens rows; the workspace capacity is a
// host-side property and is not restricted beyond num_tokens <= capacity.
void Run(TensorView partials, TensorView out, int64_t num_tokens, int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  CHECK_CUDA(partials);
  ffi::CUDADeviceGuard device_guard(partials.device().device_id);

  CHECK_CUDA(out);
  CHECK_DEVICE(partials, out);
  CHECK_INPUT_TYPE(partials, dl_bfloat16);
  CHECK_INPUT_TYPE(out, dl_bfloat16);

  TVM_FFI_ICHECK(partials.ndim() == 3) << "partials must have shape [capacity, 6, 4096]";
  const int64_t capacity = partials.size(0);
  TVM_FFI_ICHECK(partials.size(1) == kTopK && partials.size(2) == kHiddenSize)
      << "partials must have shape [capacity, 6, 4096]";
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

  auto* partials_ptr = reinterpret_cast<__nv_bfloat16*>(partials.data_ptr());
  auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  switch (SelectShape(num_tokens)) {
    case CtaShape::k512x2:
      kernel_flashinfer_sm100_megamoe_workspace_topk_reduce_tiered_t512x2<<<
          TokenGrid(num_tokens, 512, 2), 512, 0, stream>>>(partials_ptr, out_ptr);
      break;
    case CtaShape::k512x4:
      kernel_flashinfer_sm100_megamoe_workspace_topk_reduce_tiered_t512x4<<<
          TokenGrid(num_tokens, 512, 4), 512, 0, stream>>>(partials_ptr, out_ptr);
      break;
    case CtaShape::k512x4Pf256:
      kernel_flashinfer_sm100_megamoe_workspace_topk_reduce_tiered_t512x4_pf256<<<
          TokenGrid(num_tokens, 512, 4), 512, 0, stream>>>(partials_ptr, out_ptr);
      break;
    case CtaShape::k1024x4Pf256:
      kernel_flashinfer_sm100_megamoe_workspace_topk_reduce_tiered_t1024x4_pf256<<<
          TokenGrid(num_tokens, 1024, 4), 1024, 0, stream>>>(partials_ptr, out_ptr);
      break;
  }
  CheckCuda(cudaGetLastError(), "MegaMoE TopK-reduce launch");
}

}  // namespace cake_megamoe_topk_reduce
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::cake_megamoe_topk_reduce::Run);
