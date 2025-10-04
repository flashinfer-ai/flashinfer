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
  TVM_FFI_ICHECK(A->shape[0] == B->shape[0] && A->shape[0] == D->shape[0])
      << "Batch sizes must match";
  TVM_FFI_ICHECK(A->shape[2] == B->shape[1]) << "Incompatible matrix sizes";
  TVM_FFI_ICHECK(A->shape[1] == D->shape[1] && B->shape[2] == D->shape[2])
      << "Result tensor has incorrect shape";

  // PyTorch is row major by default. cuBLASLt is column major by default.
  // We need row major D as expected.
  // A ^ T * B = D, so D ^ T = B ^ T * A
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(B->dtype, b_type, [&] {
    return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP8(A->dtype, a_type, [&] {
      return DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(D->dtype, d_type, [&] {
        auto batch_size = A->shape[0];
        auto m = A->shape[1];
        auto k = A->shape[2];
        auto n = B->shape[2];

        auto lt_handle = reinterpret_cast<cublasLtHandle_t>(cublas_handle);
        cudaSetDevice(A->device.device_id);
        auto stream = get_stream(A->device);

        auto status = flashinfer::bmm_fp8::bmm_fp8_internal_cublaslt(
            workspace_buffer->data, workspace_buffer.numel(), static_cast<b_type*>(B->data),
            static_cast<a_type*>(A->data), static_cast<d_type*>(D->data), batch_size, n, m, k,
            static_cast<float*>(B_scale->data), static_cast<float*>(A_scale->data), lt_handle,
            stream);
        TVM_FFI_ICHECK(status == CUBLAS_STATUS_SUCCESS)
            << "bmm_fp8_internal_cublaslt failed: " << cublasGetStatusString(status);
        return true;
      });
    });
  });
}
