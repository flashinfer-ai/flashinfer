/*
 * Copyright (c) 2024 by FlashInfer team.
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

#include <driver_types.h>

#include <flashinfer/gemm/bmm_fp8.cuh>
#include <unordered_map>

#include "tvm_ffi_utils.h"

namespace {

// Handles are deliberately never destroyed. A thread_local destructor on the
// main thread may run after CUDA teardown has started, making cublasLtDestroy
// unsafe. This follows PyTorch's handle-pool convention.
struct ThreadLocalCublasLtHandles {
  std::unordered_map<int, cublasLtHandle_t> handles;
};

cublasLtHandle_t get_cublaslt_handle(int device_id) {
  static thread_local ThreadLocalCublasLtHandles cache;

  if (auto it = cache.handles.find(device_id); it != cache.handles.end()) {
    return it->second;
  }

  ffi::CUDADeviceGuard device_guard(device_id);
  cublasLtHandle_t handle = nullptr;
  FLASHINFER_CUBLAS_CHECK(cublasLtCreate(&handle));
  cache.handles.emplace(device_id, handle);
  return handle;
}

}  // namespace

void bmm_fp8(TensorView A, TensorView B, TensorView D, TensorView A_scale, TensorView B_scale,
             TensorView workspace_buffer) {
  CHECK_CUDA(A);
  CHECK_CUDA(B);
  CHECK_CUDA(D);
  CHECK_DIM(3, A);
  CHECK_DIM(3, B);
  CHECK_DIM(3, D);
  TVM_FFI_ICHECK(A.size(0) == B.size(0) && A.size(0) == D.size(0)) << "Batch sizes must match";
  TVM_FFI_ICHECK(A.size(2) == B.size(1)) << "Incompatible matrix sizes";
  TVM_FFI_ICHECK(A.size(1) == D.size(1) && B.size(2) == D.size(2))
      << "Result tensor has incorrect shape";

  // PyTorch is row major by default. cuBLASLt is column major by default.
  // We need row major D as expected.
  // A ^ T * B = D, so D ^ T = B ^ T * A
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(B.dtype(), b_type, [&] {
    return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(A.dtype(), a_type, [&] {
      return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(D.dtype(), d_type, [&] {
        auto batch_size = A.size(0);
        auto m = A.size(1);
        auto k = A.size(2);
        auto n = B.size(2);

        ffi::CUDADeviceGuard device_guard(A.device().device_id);
        auto stream = get_stream(A.device());
        auto lt_handle = get_cublaslt_handle(A.device().device_id);

        auto status = flashinfer::bmm_fp8::bmm_fp8_internal_cublaslt(
            workspace_buffer.data_ptr(),
            workspace_buffer.numel() * get_element_size(workspace_buffer),
            static_cast<b_type*>(B.data_ptr()), static_cast<a_type*>(A.data_ptr()),
            static_cast<d_type*>(D.data_ptr()), batch_size, n, m, k,
            static_cast<float*>(B_scale.data_ptr()), static_cast<float*>(A_scale.data_ptr()),
            lt_handle, stream);
        TVM_FFI_ICHECK(status == CUBLAS_STATUS_SUCCESS)
            << "bmm_fp8_internal_cublaslt failed: " << cublasGetStatusString(status);
        return true;
      });
    });
  });
}

// Serialize the heuristic algorithms for the problem with A's batch and K, B's N, but
// algo_m rows, into a CPU uint8 tensor. Returns the number of algorithms written.
int64_t bmm_fp8_get_algos(TensorView A, TensorView B, TensorView D, TensorView A_scale,
                          TensorView B_scale, TensorView workspace_buffer, TensorView algo_buffer,
                          int64_t algo_m) {
  CHECK_CUDA(A);
  CHECK_CUDA(B);
  CHECK_CUDA(D);
  CHECK_DIM(3, A);
  CHECK_DIM(3, B);
  CHECK_DIM(3, D);
  CHECK_CPU(algo_buffer);
  CHECK_CONTIGUOUS(algo_buffer);
  TVM_FFI_ICHECK(A.size(0) == B.size(0) && A.size(0) == D.size(0)) << "Batch sizes must match";
  TVM_FFI_ICHECK(A.size(2) == B.size(1)) << "Incompatible matrix sizes";
  TVM_FFI_ICHECK(A.size(1) == D.size(1) && B.size(2) == D.size(2))
      << "Result tensor has incorrect shape";
  TVM_FFI_ICHECK_GT(algo_m, 0) << "algo_m must be positive";

  int64_t result = 0;
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(B.dtype(), b_type, [&] {
    return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(A.dtype(), a_type, [&] {
      return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(D.dtype(), d_type, [&] {
        auto batch_size = A.size(0);
        auto m = algo_m;
        auto k = A.size(2);
        auto n = B.size(2);

        ffi::CUDADeviceGuard device_guard(A.device().device_id);
        auto lt_handle = get_cublaslt_handle(A.device().device_id);

        int max_algos = static_cast<int>(algo_buffer.numel() * get_element_size(algo_buffer) /
                                         flashinfer::bmm_fp8::kAlgoBytes);
        result = flashinfer::bmm_fp8::get_fp8_algorithms<b_type, a_type, d_type>(
            batch_size, n, m, k, static_cast<float*>(B_scale.data_ptr()),
            static_cast<float*>(A_scale.data_ptr()),
            workspace_buffer.numel() * get_element_size(workspace_buffer), lt_handle,
            algo_buffer.data_ptr(), max_algos);
        return true;
      });
    });
  });
  return static_cast<int64_t>(result);
}

// Run the BMM with algo_desc, a CPU uint8 tensor of kAlgoBytes holding a serialized
// cublasLtMatmulAlgo_t, or with the heuristic default when algo_desc is None or does not
// apply to this problem. Returns 1 if algo_desc ran, 0 if the heuristic default ran.
int64_t bmm_fp8_run_with_descriptor(TensorView A, TensorView B, TensorView D, TensorView A_scale,
                                    TensorView B_scale, TensorView workspace_buffer,
                                    ffi::Optional<TensorView> algo_desc) {
  CHECK_CUDA(A);
  CHECK_CUDA(B);
  CHECK_CUDA(D);
  CHECK_DIM(3, A);
  CHECK_DIM(3, B);
  CHECK_DIM(3, D);
  TVM_FFI_ICHECK(A.size(0) == B.size(0) && A.size(0) == D.size(0)) << "Batch sizes must match";
  TVM_FFI_ICHECK(A.size(2) == B.size(1)) << "Incompatible matrix sizes";
  TVM_FFI_ICHECK(A.size(1) == D.size(1) && B.size(2) == D.size(2))
      << "Result tensor has incorrect shape";

  const void* algo_desc_ptr = nullptr;
  if (algo_desc.has_value()) {
    const auto& desc = algo_desc.value();
    CHECK_CPU(desc);
    CHECK_CONTIGUOUS(desc);
    TVM_FFI_ICHECK_EQ(desc.numel() * get_element_size(desc),
                      static_cast<int64_t>(flashinfer::bmm_fp8::kAlgoBytes))
        << "algo_desc must hold one serialized cublasLtMatmulAlgo_t";
    algo_desc_ptr = desc.data_ptr();
  }

  bool used_descriptor = false;
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(B.dtype(), b_type, [&] {
    return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(A.dtype(), a_type, [&] {
      return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(D.dtype(), d_type, [&] {
        auto batch_size = A.size(0);
        auto m = A.size(1);
        auto k = A.size(2);
        auto n = B.size(2);

        ffi::CUDADeviceGuard device_guard(A.device().device_id);
        auto stream = get_stream(A.device());
        auto lt_handle = get_cublaslt_handle(A.device().device_id);

        auto status = flashinfer::bmm_fp8::bmm_fp8_run_with_descriptor<b_type, a_type, d_type>(
            workspace_buffer.data_ptr(),
            workspace_buffer.numel() * get_element_size(workspace_buffer),
            static_cast<b_type*>(B.data_ptr()), static_cast<a_type*>(A.data_ptr()),
            static_cast<d_type*>(D.data_ptr()), batch_size, n, m, k,
            static_cast<float*>(B_scale.data_ptr()), static_cast<float*>(A_scale.data_ptr()),
            lt_handle, stream, A.device().device_id, algo_desc_ptr, &used_descriptor);
        TVM_FFI_ICHECK(status == CUBLAS_STATUS_SUCCESS)
            << "bmm_fp8_run_with_descriptor failed: " << cublasGetStatusString(status);
        return true;
      });
    });
  });
  return used_descriptor ? 1 : 0;
}
