/*
 * Copyright (c) 2025 by FlashInfer team.
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
#include <driver_types.h>

#include <flashinfer/gemm/mm_bf16_cublaslt.cuh>

#include "tvm_ffi_utils.h"

namespace {

cudaDataType_t get_d_type(DLDataType dtype) {
  switch (encode_dlpack_dtype(dtype)) {
    case bfloat16_code:
      return CUDA_R_16BF;
    case float16_code:
      return CUDA_R_16F;
    case float32_code:
      return CUDA_R_32F;
    default:
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "out_dtype must be one of bf16/fp16/fp32.";
      return CUDA_R_16BF;
  }
}

}  // namespace

// mat1: (m, k) bf16 contiguous row-major
// mat2: (n, k) bf16 contiguous row-major (Python passes b.transpose(-2,-1))
// out:  (m, n) bf16/fp16/fp32 contiguous row-major
void mm_bf16_cublaslt(TensorView mat1, TensorView mat2, TensorView out,
                      TensorView workspace_buffer, int64_t cublas_handle, int64_t tactic) {
  CHECK_CUDA(mat1);
  CHECK_CUDA(mat2);
  CHECK_CUDA(out);
  CHECK_INPUT_AND_TYPE(mat1, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(mat2, dl_bfloat16);
  CHECK_DIM(2, mat1);
  CHECK_DIM(2, mat2);
  CHECK_DIM(2, out);

  int64_t m = mat1.size(0);
  int64_t k = mat1.size(1);
  int64_t n = mat2.size(0);

  TVM_FFI_ICHECK_EQ(mat2.size(1), k)
      << "mat2 K dimension mismatch: expected " << k << ", got " << mat2.size(1);
  TVM_FFI_ICHECK_EQ(out.size(0), m) << "out M dimension mismatch";
  TVM_FFI_ICHECK_EQ(out.size(1), n) << "out N dimension mismatch";

  auto lt_handle = reinterpret_cast<cublasLtHandle_t>(cublas_handle);
  ffi::CUDADeviceGuard device_guard(mat1.device().device_id);
  auto stream = get_stream(mat1.device());
  cudaDataType_t d_type = get_d_type(out.dtype());

  auto status = flashinfer::mm_bf16_cublaslt::run(
      static_cast<__nv_bfloat16*>(mat1.data_ptr()), static_cast<__nv_bfloat16*>(mat2.data_ptr()),
      out.data_ptr(), static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), d_type,
      workspace_buffer.data_ptr(), workspace_buffer.numel(), lt_handle, stream,
      static_cast<int>(tactic));
  TVM_FFI_ICHECK(status == CUBLAS_STATUS_SUCCESS)
      << "mm_bf16_cublaslt failed: " << cublasGetStatusString(status);
}

int64_t mm_bf16_cublaslt_tactic_num(TensorView mat1, TensorView mat2, TensorView out,
                                    TensorView workspace_buffer, int64_t cublas_handle) {
  int64_t m = mat1.size(0);
  int64_t k = mat1.size(1);
  int64_t n = mat2.size(0);
  cudaDataType_t d_type = get_d_type(out.dtype());

  auto lt_handle = reinterpret_cast<cublasLtHandle_t>(cublas_handle);
  return static_cast<int64_t>(flashinfer::mm_bf16_cublaslt::get_algorithm_count(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), d_type,
      workspace_buffer.numel(), lt_handle));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mm_bf16_cublaslt, mm_bf16_cublaslt);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mm_bf16_cublaslt_tactic_num, mm_bf16_cublaslt_tactic_num);
