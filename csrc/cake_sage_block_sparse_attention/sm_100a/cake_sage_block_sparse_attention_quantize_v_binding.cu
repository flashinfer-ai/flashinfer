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

// Exporter output, hand-folded here pending exporter support: the argument checks of Run
// use the shared helpers of tvm_ffi_utils.h instead of per-file copies.
// tvm-ffi direct-source launcher for Cake kernel 'kernel_cake_sage_block_sparse_attention_43e2bcadf25e569b50fb'.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

#include <cstdint>

extern "C" __global__ void kernel_cake_sage_block_sparse_attention_43e2bcadf25e569b50fb(__nv_bfloat16* __restrict__ v_in, uint8_t* __restrict__ v_out, float* __restrict__ v_amax, float* __restrict__ v_scale, int seqlen_k, int num_heads, int chunks_per_bh);

namespace cake_host_shim_d7f30ac2239a74bc {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

class ScopedCudaDevice {
 public:
  explicit ScopedCudaDevice(int device_id) {
    cudaError_t error = cudaGetDevice(&previous_device_);
    TVM_FFI_CHECK(error == cudaSuccess, RuntimeError)
        << "cudaGetDevice failed before host-shim launch: cudaError="
        << static_cast<int>(error);
    if (previous_device_ != device_id) {
      error = cudaSetDevice(device_id);
      TVM_FFI_CHECK(error == cudaSuccess, RuntimeError)
          << "cudaSetDevice failed before host-shim launch for cuda:"
          << device_id << ": cudaError=" << static_cast<int>(error);
      restore_ = true;
    }
  }

  ScopedCudaDevice(const ScopedCudaDevice&) = delete;
  ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

  ~ScopedCudaDevice() noexcept {
    if (restore_) {
      (void)cudaSetDevice(previous_device_);
    }
  }

 private:
  int previous_device_ = -1;
  bool restore_ = false;
};

void Run(TensorView arg_v_in, TensorView arg_v_out, TensorView arg_v_amax, TensorView arg_v_scale, int64_t arg_seqlen_k, int64_t arg_num_heads, int64_t arg_chunks_per_bh, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = arg_v_in.device();
  ScopedCudaDevice device_guard(dev.device_id);
  check_cuda_tensor(arg_v_in, "v_in");
  check_dtype(arg_v_in, dl_bfloat16, "v_in");
  check_contiguous(arg_v_in, "v_in");
  check_cuda_tensor(arg_v_out, "v_out");
  check_dtype(arg_v_out, dl_float8_e4m3fn, "v_out");
  check_contiguous(arg_v_out, "v_out");
  check_cuda_tensor(arg_v_amax, "v_amax");
  check_dtype(arg_v_amax, dl_float32, "v_amax");
  check_contiguous(arg_v_amax, "v_amax");
  check_cuda_tensor(arg_v_scale, "v_scale");
  check_dtype(arg_v_scale, dl_float32, "v_scale");
  check_contiguous(arg_v_scale, "v_scale");
  TVM_FFI_CHECK(arg_seqlen_k >= -2147483648LL && arg_seqlen_k <= 2147483647LL, ValueError)
      << "scalar 'seqlen_k' value " << arg_seqlen_k
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_heads >= -2147483648LL && arg_num_heads <= 2147483647LL, ValueError)
      << "scalar 'num_heads' value " << arg_num_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_chunks_per_bh >= -2147483648LL && arg_chunks_per_bh <= 2147483647LL, ValueError)
      << "scalar 'chunks_per_bh' value " << arg_chunks_per_bh
      << " is outside i32 range [-2147483648, 2147483647]";
  check_same_device(arg_v_out, arg_v_in, "v_out", "v_in");
  check_same_device(arg_v_amax, arg_v_in, "v_amax", "v_in");
  check_same_device(arg_v_scale, arg_v_in, "v_scale", "v_in");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";

  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  __nv_bfloat16* p_v_in = static_cast<__nv_bfloat16*>(arg_v_in.data_ptr());
  uint8_t* p_v_out = static_cast<uint8_t*>(arg_v_out.data_ptr());
  float* p_v_amax = static_cast<float*>(arg_v_amax.data_ptr());
  float* p_v_scale = static_cast<float*>(arg_v_scale.data_ptr());
  int32_t v_seqlen_k = (int32_t)arg_seqlen_k;
  int32_t v_num_heads = (int32_t)arg_num_heads;
  int32_t v_chunks_per_bh = (int32_t)arg_chunks_per_bh;
  void* kargs[] = {&p_v_in, &p_v_out, &p_v_amax, &p_v_scale, &v_seqlen_k, &v_num_heads, &v_chunks_per_bh};

  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(256u, 1u, 1u);

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 0u;
  config.stream = stream;
  cudaError_t launch_status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(kernel_cake_sage_block_sparse_attention_43e2bcadf25e569b50fb), kargs);
  TVM_FFI_CHECK(launch_status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC for kernel_cake_sage_block_sparse_attention_43e2bcadf25e569b50fb failed: "
      << cudaGetErrorString(launch_status);
}

}  // namespace cake_host_shim_d7f30ac2239a74bc

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, cake_host_shim_d7f30ac2239a74bc::Run);
