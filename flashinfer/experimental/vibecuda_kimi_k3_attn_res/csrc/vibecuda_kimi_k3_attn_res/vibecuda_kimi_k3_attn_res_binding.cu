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

// TVM-FFI launcher for the VibeCUDA Kimi-K3 attention-residual kernel.
// The per-architecture payload translation unit
// (vibecuda_kimi_k3_attn_res_kernels_{sm100,sm103}.cu, same JIT module)
// provides the route tables, the frozen production dispatch ladder and the
// launch entry points declared below.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

int vibecuda_k3_init_host();
int vibecuda_k3_num_sms();
int vibecuda_k3_route_id(long long nvb, long long T);
int vibecuda_k3_run(const void* prefix, const void* bank, const void* cw, const void* ow, void* out,
                    long long T, long long nvb, long long stride_bm, double eps,
                    long long enable_pdl, void* stream);

namespace vibecuda_kimi_k3_attn_res_host_shim {

using tvm::ffi::TensorView;

void Run(TensorView arg_out, TensorView arg_prefix, TensorView arg_bank, TensorView arg_cw,
         TensorView arg_ow, int64_t arg_num_blocks, int64_t arg_stride_bm, double arg_eps,
         int64_t enable_pdl) {
  DLDevice dev = arg_out.device();
  tvm::ffi::CUDADeviceGuard device_guard(dev.device_id);
  check_cuda_tensor(arg_out, "out");
  check_dtype(arg_out, DLDataType{kDLBfloat, 16, 1}, "out");
  check_contiguous(arg_out, "out");
  check_cuda_tensor(arg_prefix, "prefix");
  check_dtype(arg_prefix, DLDataType{kDLBfloat, 16, 1}, "prefix");
  check_contiguous(arg_prefix, "prefix");
  check_cuda_tensor(arg_bank, "bank");
  check_dtype(arg_bank, DLDataType{kDLBfloat, 16, 1}, "bank");
  check_cuda_tensor(arg_cw, "cw");
  check_dtype(arg_cw, DLDataType{kDLBfloat, 16, 1}, "cw");
  check_contiguous(arg_cw, "cw");
  check_cuda_tensor(arg_ow, "ow");
  check_dtype(arg_ow, DLDataType{kDLBfloat, 16, 1}, "ow");
  check_contiguous(arg_ow, "ow");
  check_same_device(arg_prefix, arg_out, "prefix", "out");
  check_same_device(arg_bank, arg_out, "bank", "out");
  check_same_device(arg_cw, arg_out, "cw", "out");
  check_same_device(arg_ow, arg_out, "ow", "ow");
  // Dense rows with an arbitrary row stride (row-padded bank layouts are
  // supported by the kernel through stride_bm); the element and block
  // strides must stay packed.
  TVM_FFI_CHECK(arg_bank.stride(1) == 7168 && arg_bank.stride(2) == 1, ValueError)
      << "bank must be dense in its last two dims, got strides (" << arg_bank.stride(0) << ", "
      << arg_bank.stride(1) << ", " << arg_bank.stride(2) << ")";
  TVM_FFI_CHECK(arg_stride_bm >= 0, ValueError)
      << "bank row stride must be non-negative, got " << arg_stride_bm;
  const int64_t M = arg_prefix.size(0);
  TVM_FFI_CHECK(M > 0 && M <= 2147483647LL, ValueError) << "token count out of range: " << M;

  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  const int rc = vibecuda_k3_run(arg_prefix.data_ptr(), arg_bank.data_ptr(), arg_cw.data_ptr(),
                                 arg_ow.data_ptr(), arg_out.data_ptr(), M, arg_num_blocks,
                                 arg_stride_bm, arg_eps, enable_pdl, stream);
  TVM_FFI_CHECK(rc == 0, RuntimeError)
      << "vibecuda_k3_run failed for M=" << M << " num_blocks=" << arg_num_blocks
      << (rc == -1001 ? std::string(" (uncovered dispatch route)")
                      : std::string(" (") + cudaGetErrorString((cudaError_t)rc) + ")");
}

void Warmup() {
  const int rc = vibecuda_k3_init_host();
  TVM_FFI_CHECK(rc == 0, RuntimeError) << "vibecuda_k3 route-table initialization failed";
}

int64_t RouteId(int64_t num_blocks, int64_t M) {
  const int id = vibecuda_k3_route_id(num_blocks, M);
  TVM_FFI_CHECK(id >= 0, ValueError)
      << "no registered VibeCUDA route for num_blocks=" << num_blocks << " M=" << M;
  return id;
}

int64_t NumSms() { return vibecuda_k3_num_sms(); }

}  // namespace vibecuda_kimi_k3_attn_res_host_shim

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, vibecuda_kimi_k3_attn_res_host_shim::Run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(warmup, vibecuda_kimi_k3_attn_res_host_shim::Warmup);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(route_id, vibecuda_kimi_k3_attn_res_host_shim::RouteId);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(num_sms, vibecuda_kimi_k3_attn_res_host_shim::NumSms);
