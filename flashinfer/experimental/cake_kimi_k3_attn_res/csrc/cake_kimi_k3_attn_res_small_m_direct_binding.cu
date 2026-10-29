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

// Shared tvm-ffi launch binding of the small-M direct programs (one token per 256-thread CTA; 7 tensors,
// the two snapshot-bank strides, eps, output_norm_eps, M).  A hand fold of the exporter's per-program
// bindings, pending exporter regeneration: -DCAKE_PROGRAM_KERNEL (the kernel symbol), -DCAKE_BIND_SMEM,
// -DCAKE_BIND_THREADS and -DCAKE_BIND_KM (the last snapshot row the program touches, K - 1 + write; -1 for
// the K = 0 dense program, which reads no row) select the program; the preprocessed text equals the
// exporter's binding apart from the file / line of the host checks and the spelling of the stringised
// precondition as adjacent string literals.
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "cake_ffi_checks.h"

#include <cstdint>

#ifndef CAKE_PROGRAM_KERNEL
#error "CAKE_PROGRAM_KERNEL (the kernel symbol the entry's device source defines) must be defined on the compile line"
#endif
#ifndef CAKE_BIND_SMEM
#error "CAKE_BIND_SMEM (dynamic shared memory bytes of the launch) must be defined on the compile line"
#endif
#ifndef CAKE_BIND_THREADS
#error "CAKE_BIND_THREADS (threads per CTA) must be defined on the compile line"
#endif
#ifndef CAKE_BIND_KM
#error "CAKE_BIND_KM (the last snapshot row the program touches: K - 1 + write; -1 when no row is read) must be defined on the compile line"
#endif
// Stringise / paste the expansion of a macro argument (the kernel name string, the binding namespace and
// the integer-literal suffix of a stringised precondition take the value the exporter's binding spelled).
#define CAKE_STRINGIZE_(x) #x
#define CAKE_STRINGIZE(x) CAKE_STRINGIZE_(x)
#define CAKE_PASTE_(a, b) a##b
#define CAKE_PASTE(a, b) CAKE_PASTE_(a, b)
#define CAKE_BINDING_NS CAKE_PASTE(cake_binding_, CAKE_PROGRAM_KERNEL)
// Expand the selectors of a check condition before TVM_FFI_CHECK stringises it, so the
// message spells the literal the exporter's binding carried.
#define CAKE_CHECK_EXPANDED(cond, ErrorKind) TVM_FFI_CHECK(cond, ErrorKind)

extern "C" __global__ void CAKE_PROGRAM_KERNEL(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M);

namespace CAKE_BINDING_NS {

using tvm::ffi::TensorView;

constexpr const char* KERNEL_NAME = CAKE_STRINGIZE(CAKE_PROGRAM_KERNEL);

void CheckCuda(cudaError_t status, const char* what) {
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << KERNEL_NAME << ": " << what << " failed: " << cudaGetErrorString(status);
}

void* DataPtr(const TensorView& t) { return static_cast<char*>(t.data_ptr()) + t.byte_offset(); }

#if CAKE_BIND_KM >= 1
[[noreturn]] void NoInt64(int64_t a, const char* op, int64_t b) {
  TVM_FFI_THROW(ValueError) << KERNEL_NAME << ": " << a << " " << op << " " << b
                            << " has no int64 value in a host check";
}

int64_t CheckedAdd(int64_t a, int64_t b) {
  int64_t sum;
  if (__builtin_add_overflow(a, b, &sum)) NoInt64(a, "+", b);
  return sum;
}

#if CAKE_BIND_KM >= 2
int64_t CheckedMul(int64_t a, int64_t b) {
  int64_t product;
  if (__builtin_mul_overflow(a, b, &product)) NoInt64(a, "*", b);
  return product;
}

#endif
#endif
void Run(TensorView a_out, TensorView a_prefix, TensorView a_delta, TensorView a_blocks, TensorView a_norm_weight, TensorView a_qk_weight, TensorView a_output_norm_weight, int64_t a_blocks_m_stride, int64_t a_blocks_k_stride, double a_eps, double a_output_norm_eps, int64_t a_M, int64_t a_enable_pdl, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = a_out.device();
  tvm::ffi::CUDADeviceGuard device_guard(dev.device_id);
  check_cuda_tensor(a_out, "out");
  check_dtype(a_out, DLDataType{kDLBfloat, 16, 1}, "out");
  check_cuda_tensor(a_prefix, "prefix");
  check_dtype(a_prefix, DLDataType{kDLBfloat, 16, 1}, "prefix");
  check_cuda_tensor(a_delta, "delta");
  check_dtype(a_delta, DLDataType{kDLBfloat, 16, 1}, "delta");
  check_cuda_tensor(a_blocks, "blocks");
  check_dtype(a_blocks, DLDataType{kDLBfloat, 16, 1}, "blocks");
  check_cuda_tensor(a_norm_weight, "norm_weight");
  check_dtype(a_norm_weight, DLDataType{kDLBfloat, 16, 1}, "norm_weight");
  check_cuda_tensor(a_qk_weight, "qk_weight");
  check_dtype(a_qk_weight, DLDataType{kDLBfloat, 16, 1}, "qk_weight");
  check_cuda_tensor(a_output_norm_weight, "output_norm_weight");
  check_dtype(a_output_norm_weight, DLDataType{kDLBfloat, 16, 1}, "output_norm_weight");
  TVM_FFI_CHECK(a_blocks_m_stride >= 0LL && a_blocks_m_stride <= 9223372036854775807LL, ValueError)
      << KERNEL_NAME << ": 'blocks_m_stride' = " << a_blocks_m_stride
      << " is outside the u64 range [0, 9223372036854775807]";
  TVM_FFI_CHECK(a_blocks_k_stride >= 0LL && a_blocks_k_stride <= 9223372036854775807LL, ValueError)
      << KERNEL_NAME << ": 'blocks_k_stride' = " << a_blocks_k_stride
      << " is outside the u64 range [0, 9223372036854775807]";
  TVM_FFI_CHECK(a_M >= -2147483648LL && a_M <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'M' = " << a_M
      << " is outside the i32 range [-2147483648, 2147483647]";
  check_contiguous(a_out, "out");
  check_contiguous(a_prefix, "prefix");
  check_same_device(a_prefix, a_out, "prefix", "out");
  check_contiguous(a_delta, "delta");
  check_same_device(a_delta, a_out, "delta", "out");
  check_contiguous(a_blocks, "blocks");
  check_same_device(a_blocks, a_out, "blocks", "out");
  check_contiguous(a_norm_weight, "norm_weight");
  check_same_device(a_norm_weight, a_out, "norm_weight", "out");
  check_contiguous(a_qk_weight, "qk_weight");
  check_same_device(a_qk_weight, a_out, "qk_weight", "out");
  check_contiguous(a_output_norm_weight, "output_norm_weight");
  check_same_device(a_output_norm_weight, a_out, "output_norm_weight", "out");
  TVM_FFI_CHECK((a_blocks_k_stride >= 7168LL), ValueError)
      << KERNEL_NAME << ": requires (blocks_k_stride >= 7168)";
#if CAKE_BIND_KM >= 2
  CAKE_CHECK_EXPANDED((a_blocks_m_stride >= CheckedAdd(CheckedMul(CAKE_PASTE(CAKE_BIND_KM, LL), a_blocks_k_stride), 7168LL)), ValueError)
      << KERNEL_NAME << ": requires (blocks_m_stride >= ((" CAKE_STRINGIZE(CAKE_BIND_KM) " * blocks_k_stride) + 7168))";
#elif CAKE_BIND_KM == 1
  TVM_FFI_CHECK((a_blocks_m_stride >= CheckedAdd(a_blocks_k_stride, 7168LL)), ValueError)
      << KERNEL_NAME << ": requires (blocks_m_stride >= (blocks_k_stride + 7168))";
#elif CAKE_BIND_KM == 0
  TVM_FFI_CHECK((a_blocks_m_stride >= 7168LL), ValueError)
      << KERNEL_NAME << ": requires (blocks_m_stride >= 7168)";
#endif
  TVM_FFI_CHECK(a_enable_pdl == 0LL || a_enable_pdl == 1LL, ValueError)
      << KERNEL_NAME << ": 'enable_pdl' = " << a_enable_pdl << " is not 0 or 1";
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << KERNEL_NAME << ": the grid (" << grid_x << ", " << grid_y << ", " << grid_z << ") must be positive";
  TVM_FFI_CHECK(grid_x <= 2147483647 && grid_y <= 65535 && grid_z <= 65535, ValueError)
      << KERNEL_NAME << ": the grid (" << grid_x << ", " << grid_y << ", " << grid_z << ") exceeds CUDA's limits";
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  void* v_out = DataPtr(a_out);
  void* v_prefix = DataPtr(a_prefix);
  void* v_delta = DataPtr(a_delta);
  void* v_blocks = DataPtr(a_blocks);
  void* v_norm_weight = DataPtr(a_norm_weight);
  void* v_qk_weight = DataPtr(a_qk_weight);
  void* v_output_norm_weight = DataPtr(a_output_norm_weight);
  uint64_t v_blocks_m_stride = static_cast<uint64_t>(a_blocks_m_stride);
  uint64_t v_blocks_k_stride = static_cast<uint64_t>(a_blocks_k_stride);
  float v_eps = static_cast<float>(a_eps);
  float v_output_norm_eps = static_cast<float>(a_output_norm_eps);
  int32_t v_M = static_cast<int32_t>(a_M);
  void* kernel_args[] = {&v_out, &v_prefix, &v_delta, &v_blocks, &v_norm_weight, &v_qk_weight, &v_output_norm_weight, &v_blocks_m_stride, &v_blocks_k_stride, &v_eps, &v_output_norm_eps, &v_M};
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y), static_cast<unsigned>(grid_z));
  config.blockDim = dim3(CAKE_BIND_THREADS, 1, 1);
  config.dynamicSmemBytes = CAKE_BIND_SMEM;
  config.stream = stream;
  cudaLaunchAttribute attrs[1] = {};
  int n = 0;
  if (a_enable_pdl) {
    attrs[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[n].val.programmaticStreamSerializationAllowed = 1;
    ++n;
  }
  config.attrs = attrs;
  config.numAttrs = n;
  CheckCuda(cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(CAKE_PROGRAM_KERNEL), kernel_args),
            "cudaLaunchKernelExC");
}

}  // namespace CAKE_BINDING_NS

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, CAKE_BINDING_NS::Run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(build_cuda_version, cake_ffi_build_cuda_version);
