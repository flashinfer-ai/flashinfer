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

#include "tvm_ffi_utils.h"

void bmm_fp8(TensorView A, TensorView B, TensorView D, TensorView A_scale, TensorView B_scale,
             TensorView workspace_buffer, int64_t cublas_handle) {
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

        auto lt_handle = reinterpret_cast<cublasLtHandle_t>(cublas_handle);
        ffi::CUDADeviceGuard device_guard(A.device().device_id);
        auto stream = get_stream(A.device());

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

int64_t bmm_fp8_get_algos(TensorView A, TensorView B, TensorView D, TensorView A_scale,
                          TensorView B_scale, TensorView workspace_buffer, int64_t cublas_handle,
                          TensorView algo_buffer) {
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

  int64_t result = 0;
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(B.dtype(), b_type, [&] {
    return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(A.dtype(), a_type, [&] {
      return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(D.dtype(), d_type, [&] {
        auto batch_size = A.size(0);
        auto m = A.size(1);
        auto k = A.size(2);
        auto n = B.size(2);

        auto lt_handle = reinterpret_cast<cublasLtHandle_t>(cublas_handle);
        ffi::CUDADeviceGuard device_guard(A.device().device_id);

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

void bmm_fp8_run_with_algo(TensorView A, TensorView B, TensorView D, TensorView A_scale,
                           TensorView B_scale, TensorView workspace_buffer, int64_t cublas_handle,
                           TensorView algo_buffer, int64_t algo_idx) {
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

  int64_t max_algos =
      algo_buffer.numel() * get_element_size(algo_buffer) / flashinfer::bmm_fp8::kAlgoBytes;
  TVM_FFI_ICHECK(algo_idx >= 0 && algo_idx < max_algos)
      << "algo_idx " << algo_idx << " out of range [0, " << max_algos << ")";

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(B.dtype(), b_type, [&] {
    return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(A.dtype(), a_type, [&] {
      return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(D.dtype(), d_type, [&] {
        auto batch_size = A.size(0);
        auto m = A.size(1);
        auto k = A.size(2);
        auto n = B.size(2);

        auto lt_handle = reinterpret_cast<cublasLtHandle_t>(cublas_handle);
        ffi::CUDADeviceGuard device_guard(A.device().device_id);
        auto stream = get_stream(A.device());

        auto status = flashinfer::bmm_fp8::bmm_fp8_run_with_algo<b_type, a_type, d_type>(
            workspace_buffer.data_ptr(),
            workspace_buffer.numel() * get_element_size(workspace_buffer),
            static_cast<b_type*>(B.data_ptr()), static_cast<a_type*>(A.data_ptr()),
            static_cast<d_type*>(D.data_ptr()), batch_size, n, m, k,
            static_cast<float*>(B_scale.data_ptr()), static_cast<float*>(A_scale.data_ptr()),
            lt_handle, stream, algo_buffer.data_ptr(), static_cast<int>(algo_idx));
        TVM_FFI_ICHECK(status == CUBLAS_STATUS_SUCCESS)
            << "bmm_fp8_run_with_algo failed: " << cublasGetStatusString(status);
        return true;
      });
    });
  });
}
