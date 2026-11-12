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

// The 32 programs of ``csrc/cake_router_gemm`` (16 token counts x 2 hidden sizes), one translation
// unit each, with the launch shape each was built for: threads per CTA, experts per CTA, the CTAs
// of a cluster that split the hidden dimension, and dynamic shared memory bytes.
#define CAKE_ROUTER_GEMM_FOR_EACH_PROGRAM(X) \
  X(m1_k6144, 384, 1, 1, 128)                \
  X(m2_k6144, 384, 1, 1, 128)                \
  X(m3_k6144, 384, 1, 1, 256)                \
  X(m4_k6144, 384, 1, 1, 256)                \
  X(m5_k6144, 384, 1, 1, 256)                \
  X(m6_k6144, 384, 1, 1, 384)                \
  X(m7_k6144, 384, 1, 1, 384)                \
  X(m8_k6144, 256, 2, 1, 512)                \
  X(m9_k6144, 256, 2, 1, 640)                \
  X(m10_k6144, 384, 2, 1, 1024)              \
  X(m11_k6144, 384, 2, 1, 1152)              \
  X(m12_k6144, 384, 2, 1, 1152)              \
  X(m13_k6144, 384, 2, 1, 1280)              \
  X(m14_k6144, 384, 2, 1, 1408)              \
  X(m15_k6144, 384, 2, 1, 1536)              \
  X(m16_k6144, 384, 2, 1, 1536)              \
  X(m1_k7168, 224, 1, 1, 128)                \
  X(m2_k7168, 448, 1, 1, 128)                \
  X(m3_k7168, 448, 1, 1, 256)                \
  X(m4_k7168, 448, 1, 1, 256)                \
  X(m5_k7168, 448, 1, 1, 384)                \
  X(m6_k7168, 448, 1, 1, 384)                \
  X(m7_k7168, 448, 2, 1, 896)                \
  X(m8_k7168, 448, 2, 1, 896)                \
  X(m9_k7168, 128, 4, 7, 768)                \
  X(m10_k7168, 128, 4, 7, 896)               \
  X(m11_k7168, 128, 4, 7, 1024)              \
  X(m12_k7168, 128, 4, 7, 1024)              \
  X(m13_k7168, 128, 4, 7, 1152)              \
  X(m14_k7168, 128, 4, 7, 1152)              \
  X(m15_k7168, 128, 4, 7, 1280)              \
  X(m16_k7168, 128, 4, 7, 1408)

// The narrow programs: (suffix, num_tokens, hidden_dim, threads, experts per CTA, k_split, smem
// bytes).
#define CAKE_ROUTER_GEMM_FOR_EACH_NARROW_PROGRAM(X) \
  X(m7_k7168_narrow, 7, 7168, 448, 1, 1, 512)       \
  X(m8_k7168_narrow, 8, 7168, 448, 1, 1, 512)       \
  X(m9_k7168_narrow, 9, 7168, 448, 1, 1, 512)       \
  X(m10_k7168_narrow, 10, 7168, 448, 1, 1, 640)     \
  X(m11_k7168_narrow, 11, 7168, 448, 1, 1, 640)     \
  X(m12_k7168_narrow, 12, 7168, 448, 1, 1, 768)     \
  X(m13_k7168_narrow, 13, 7168, 448, 1, 1, 768)     \
  X(m14_k7168_narrow, 14, 7168, 448, 1, 1, 896)     \
  X(m15_k7168_narrow, 15, 7168, 448, 1, 1, 896)     \
  X(m16_k7168_narrow, 16, 7168, 448, 1, 1, 896)

#define CAKE_ROUTER_GEMM_DECLARE(suffix, ...)                                              \
  extern "C" __global__ void kernel_cake_blackwell_router_gemm_##suffix(                   \
      __nv_bfloat16* mat_a, __nv_bfloat16* mat_b, float* out_f32, __nv_bfloat16* out_bf16, \
      int num_experts, int out_is_bf16);
CAKE_ROUTER_GEMM_FOR_EACH_PROGRAM(CAKE_ROUTER_GEMM_DECLARE)
CAKE_ROUTER_GEMM_FOR_EACH_NARROW_PROGRAM(CAKE_ROUTER_GEMM_DECLARE)
#undef CAKE_ROUTER_GEMM_DECLARE

namespace flashinfer::cake_router_gemm {

using tvm::ffi::TensorView;

using KernelFn = void (*)(__nv_bfloat16*, __nv_bfloat16*, float*, __nv_bfloat16*, int, int);

struct Program {
  KernelFn kernel;
  int threads;
  int experts_per_cta;
  int k_split;
  int smem_bytes;
};

// ceil(num_experts / experts_per_cta) expert groups, each split over k_split CTAs of one cluster.
int64_t num_ctas(const Program& program, int64_t num_experts) {
  return (num_experts + program.experts_per_cta - 1) / program.experts_per_cta * program.k_split;
}

// The program for (num_tokens, hidden_dim); the table is indexed by num_tokens - 1.
const Program& program_for(int64_t num_tokens, int64_t hidden_dim) {
#define CAKE_ROUTER_GEMM_ENTRY(suffix, threads, experts_per_cta, k_split, smem_bytes) \
  {kernel_cake_blackwell_router_gemm_##suffix, threads, experts_per_cta, k_split, smem_bytes},
  static const Program programs[32] = {CAKE_ROUTER_GEMM_FOR_EACH_PROGRAM(CAKE_ROUTER_GEMM_ENTRY)};
#undef CAKE_ROUTER_GEMM_ENTRY
  return programs[(hidden_dim == 7168 ? 16 : 0) + (num_tokens - 1)];
}

// The narrow program of (num_tokens, hidden_dim), or nullptr: one CTA per expert, faster than the
// table's program when its grid fits one CTA per SM.
const Program* narrow_program_for(int64_t num_tokens, int64_t hidden_dim) {
  struct Entry {
    int64_t num_tokens;
    int64_t hidden_dim;
    Program program;
  };
#define CAKE_ROUTER_GEMM_ENTRY(suffix, m, k, threads, experts_per_cta, k_split, smem_bytes) \
  {m,                                                                                       \
   k,                                                                                       \
   {kernel_cake_blackwell_router_gemm_##suffix, threads, experts_per_cta, k_split, smem_bytes}},
  static const Entry entries[] = {CAKE_ROUTER_GEMM_FOR_EACH_NARROW_PROGRAM(CAKE_ROUTER_GEMM_ENTRY)};
#undef CAKE_ROUTER_GEMM_ENTRY
  for (const Entry& entry : entries) {
    if (entry.num_tokens == num_tokens && entry.hidden_dim == hidden_dim) return &entry.program;
  }
  return nullptr;
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
  // The kernel addresses mat_a[m * hidden_dim + k], mat_b[e * hidden_dim + k] and
  // out[m * num_experts + e]: every operand must be dense, not a padded view.
  TVM_FFI_CHECK(mat_a.stride(1) == 1 && (num_tokens == 1 || mat_a.stride(0) == hidden_dim),
                ValueError)
      << "mat_a must be a dense row-major [num_tokens, hidden_dim] tensor, got strides ("
      << mat_a.stride(0) << ", " << mat_a.stride(1) << ")";
  TVM_FFI_CHECK(mat_b.stride(0) == 1 && (num_experts == 1 || mat_b.stride(1) == hidden_dim),
                ValueError)
      << "mat_b must be a dense column-major [hidden_dim, num_experts] tensor, got strides ("
      << mat_b.stride(0) << ", " << mat_b.stride(1) << ")";
  TVM_FFI_CHECK(out.stride(1) == 1 && (num_tokens == 1 || out.stride(0) == num_experts), ValueError)
      << "out must be a dense row-major [num_tokens, num_experts] tensor, got strides ("
      << out.stride(0) << ", " << out.stride(1) << ")";

  ffi::CUDADeviceGuard device_guard(mat_a.device().device_id);
  const cudaStream_t stream = get_stream(mat_a.device());

  const Program* chosen = &program_for(num_tokens, hidden_dim);
  if (const Program* narrow = narrow_program_for(num_tokens, hidden_dim)) {
    int num_sms = 0;
    TVM_FFI_ICHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount,
                                          mat_a.device().device_id) == cudaSuccess)
        << "cake_router_gemm: cannot read the SM count";
    if (num_ctas(*narrow, num_experts) <= num_sms) chosen = narrow;
  }
  const Program& program = *chosen;
  const int64_t grid = num_ctas(program, num_experts);
  TVM_FFI_CHECK(grid <= std::numeric_limits<int32_t>::max(), ValueError)
      << "num_experts " << num_experts << " needs " << grid << " CTAs, over the grid limit";

  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(static_cast<unsigned>(grid), 1, 1);
  config.blockDim = dim3(static_cast<unsigned>(program.threads), 1, 1);
  config.dynamicSmemBytes = static_cast<size_t>(program.smem_bytes);
  config.stream = stream;
  cudaLaunchAttribute attrs[2];
  int num_attrs = 0;
  if (program.k_split > 1) {
    attrs[num_attrs].id = cudaLaunchAttributeClusterDimension;
    attrs[num_attrs].val.clusterDim.x = static_cast<unsigned>(program.k_split);
    attrs[num_attrs].val.clusterDim.y = 1;
    attrs[num_attrs].val.clusterDim.z = 1;
    ++num_attrs;
  }
  if (launch_with_pdl) {
    attrs[num_attrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[num_attrs].val.programmaticStreamSerializationAllowed = 1;
    ++num_attrs;
  }
  config.attrs = attrs;
  config.numAttrs = num_attrs;

  __nv_bfloat16* a_ptr = static_cast<__nv_bfloat16*>(mat_a.data_ptr());
  __nv_bfloat16* b_ptr = static_cast<__nv_bfloat16*>(mat_b.data_ptr());
  float* out_f32 = static_cast<float*>(out.data_ptr());
  __nv_bfloat16* out_bf16 = static_cast<__nv_bfloat16*>(out.data_ptr());
  int experts = static_cast<int>(num_experts);
  int output_selector = out_is_bf16 ? 1 : 0;
  cudaError_t status = cudaLaunchKernelEx(&config, program.kernel, a_ptr, b_ptr, out_f32, out_bf16,
                                          experts, output_selector);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cake_router_gemm launch failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer::cake_router_gemm

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::cake_router_gemm::run);
