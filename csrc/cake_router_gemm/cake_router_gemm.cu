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
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#include "tvm_ffi_utils.h"

// The 32 source-built programs of ``csrc/cake_router_gemm`` (16 token counts x 2 hidden
// sizes), one translation unit each; this binding declares and launches them.
#define CAKE_ROUTER_GEMM_FOR_EACH_PROGRAM(X) \
  X(m1_k6144)                                \
  X(m2_k6144)                                \
  X(m3_k6144)                                \
  X(m4_k6144)                                \
  X(m5_k6144)                                \
  X(m6_k6144)                                \
  X(m7_k6144)                                \
  X(m8_k6144)                                \
  X(m9_k6144)                                \
  X(m10_k6144)                               \
  X(m11_k6144)                               \
  X(m12_k6144)                               \
  X(m13_k6144)                               \
  X(m14_k6144)                               \
  X(m15_k6144)                               \
  X(m16_k6144)                               \
  X(m1_k7168)                                \
  X(m2_k7168)                                \
  X(m3_k7168)                                \
  X(m4_k7168)                                \
  X(m5_k7168)                                \
  X(m6_k7168)                                \
  X(m7_k7168)                                \
  X(m8_k7168)                                \
  X(m9_k7168)                                \
  X(m10_k7168)                               \
  X(m11_k7168)                               \
  X(m12_k7168)                               \
  X(m13_k7168)                               \
  X(m14_k7168)                               \
  X(m15_k7168)                               \
  X(m16_k7168)

#define CAKE_ROUTER_GEMM_DECLARE(suffix)                                                   \
  extern "C" __global__ void kernel_cake_blackwell_router_gemm_##suffix(                   \
      __nv_bfloat16* mat_a, __nv_bfloat16* mat_b, float* out_f32, __nv_bfloat16* out_bf16, \
      int num_experts, int out_is_bf16);
CAKE_ROUTER_GEMM_FOR_EACH_PROGRAM(CAKE_ROUTER_GEMM_DECLARE)
#undef CAKE_ROUTER_GEMM_DECLARE

namespace flashinfer::cake_router_gemm {

using tvm::ffi::TensorView;

using KernelFn = void (*)(__nv_bfloat16*, __nv_bfloat16*, float*, __nv_bfloat16*, int, int);

// The program for (num_tokens, hidden_dim); the table is indexed by num_tokens - 1.
KernelFn kernel_for(int64_t num_tokens, int64_t hidden_dim) {
#define CAKE_ROUTER_GEMM_ENTRY(suffix) kernel_cake_blackwell_router_gemm_##suffix,
  static const KernelFn kernels[32] = {CAKE_ROUTER_GEMM_FOR_EACH_PROGRAM(CAKE_ROUTER_GEMM_ENTRY)};
#undef CAKE_ROUTER_GEMM_ENTRY
  return kernels[(hidden_dim == 7168 ? 16 : 0) + (num_tokens - 1)];
}

// out = mat_a @ mat_b for mat_a [num_tokens, hidden_dim] (BF16, row-major), mat_b
// [hidden_dim, num_experts] (BF16, the column-major view of a row-major [num_experts,
// hidden_dim] weight) and out [num_tokens, num_experts] (FP32 or BF16, row-major).
void run(TensorView mat_a, TensorView mat_b, TensorView out, bool launch_with_pdl) {
  CHECK_CUDA(mat_a);
  CHECK_CUDA(mat_b);
  CHECK_CUDA(out);
  CHECK_DEVICE(mat_b, mat_a);
  CHECK_DEVICE(out, mat_a);
  CHECK_DIM(2, mat_a);
  CHECK_DIM(2, mat_b);
  CHECK_DIM(2, out);
  CHECK_INPUT_TYPE(mat_a, dl_bfloat16);
  CHECK_INPUT_TYPE(mat_b, dl_bfloat16);
  const bool out_is_bf16 = out.dtype() == dl_bfloat16;
  TVM_FFI_CHECK(out_is_bf16 || out.dtype() == dl_float32, TypeError)
      << "out must be a float32 or bfloat16 tensor";

  const int64_t num_tokens = mat_a.size(0);
  const int64_t hidden_dim = mat_a.size(1);
  const int64_t num_experts = mat_b.size(1);
  TVM_FFI_CHECK(num_tokens >= 1 && num_tokens <= 16, ValueError)
      << "num_tokens (mat_a.size(0)) must be in [1, 16], got " << num_tokens;
  TVM_FFI_CHECK(hidden_dim == 6144 || hidden_dim == 7168, ValueError)
      << "hidden_dim (mat_a.size(1)) must be 6144 or 7168, got " << hidden_dim;
  TVM_FFI_CHECK(mat_b.size(0) == hidden_dim, ValueError)
      << "mat_b.size(0) must equal hidden_dim " << hidden_dim << ", got " << mat_b.size(0);
  TVM_FFI_CHECK(num_experts >= 1 && num_experts <= std::numeric_limits<int32_t>::max(), ValueError)
      << "num_experts (mat_b.size(1)) must be in [1, 2^31 - 1], got " << num_experts;
  TVM_FFI_CHECK(out.size(0) == num_tokens && out.size(1) == num_experts, ValueError)
      << "out must have shape [" << num_tokens << ", " << num_experts << "], got [" << out.size(0)
      << ", " << out.size(1) << "]";

  ffi::CUDADeviceGuard device_guard(mat_a.device().device_id);
  const cudaStream_t stream = get_stream(mat_a.device());

  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(static_cast<unsigned>(num_experts), 1, 1);
  config.blockDim = dim3(128, 1, 1);
  config.dynamicSmemBytes = static_cast<size_t>(16 * num_tokens);
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = launch_with_pdl ? 1 : 0;

  KernelFn kernel = kernel_for(num_tokens, hidden_dim);
  __nv_bfloat16* a_ptr = static_cast<__nv_bfloat16*>(mat_a.data_ptr());
  __nv_bfloat16* b_ptr = static_cast<__nv_bfloat16*>(mat_b.data_ptr());
  float* out_f32 = static_cast<float*>(out.data_ptr());
  __nv_bfloat16* out_bf16 = static_cast<__nv_bfloat16*>(out.data_ptr());
  int experts = static_cast<int>(num_experts);
  int output_selector = out_is_bf16 ? 1 : 0;
  cudaError_t status = cudaLaunchKernelEx(&config, kernel, a_ptr, b_ptr, out_f32, out_bf16, experts,
                                          output_selector);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cake_router_gemm launch failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer::cake_router_gemm

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::cake_router_gemm::run);
