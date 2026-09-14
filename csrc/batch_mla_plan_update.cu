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

#include <cuda_runtime.h>

#include <algorithm>

#include "batch_mla_plan_update.cuh"

namespace {

constexpr int kPlanUpdateThreads = 256;
constexpr int kPlanUpdateBlocks = 128;

void CheckPlanUpdateTensor(TensorView tensor, DLDataType dtype, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
  TVM_FFI_ICHECK_EQ(tensor.ndim(), 1) << name << " must be a 1D tensor";
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dtype) << name << " has an invalid dtype";
}

void CheckSameDevice(TensorView reference, TensorView tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, reference.device().device_type)
      << name << " must be on the live workspace device";
  TVM_FFI_ICHECK_EQ(tensor.device().device_id, reference.device().device_id)
      << name << " must be on the live workspace device";
}

void CheckEqualCapacity(TensorView live, TensorView shadow, const char* name) {
  TVM_FFI_ICHECK_EQ(shadow.numel(), live.numel())
      << name << " shadow capacity must match the live tensor";
}

__global__ void CommitBatchMLACudaGraphPlanUpdateKernel(
    uint8_t* live_int_workspace, int32_t* live_qo_indptr, int32_t* live_kv_indptr,
    int32_t* live_kv_indices, int32_t* live_kv_len_arr, const uint8_t* candidate_int_workspace,
    const int32_t* candidate_qo_indptr, const int32_t* candidate_kv_indptr,
    const int32_t* source_kv_indices, const int32_t* candidate_kv_len_arr,
    int64_t staged_int_workspace_bytes, int64_t qo_indptr_elements, int64_t kv_indptr_elements,
    int64_t kv_len_arr_elements, int64_t live_kv_indices_length, int64_t work_elements) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < work_elements; index += stride) {
    if (index < staged_int_workspace_bytes) {
      live_int_workspace[index] = candidate_int_workspace[index];
    }
    if (index < qo_indptr_elements) {
      live_qo_indptr[index] = candidate_qo_indptr[index];
    }
    if (index < kv_indptr_elements) {
      live_kv_indptr[index] = candidate_kv_indptr[index];
    }
    if (index < live_kv_indices_length) {
      live_kv_indices[index] = source_kv_indices[index];
    }
    if (index < kv_len_arr_elements) {
      live_kv_len_arr[index] = candidate_kv_len_arr[index];
    }
  }
}

}  // namespace

void CommitBatchMLACudaGraphPlanUpdate(
    TensorView live_int_workspace, TensorView live_qo_indptr, TensorView live_kv_indptr,
    TensorView live_kv_indices, TensorView live_kv_len_arr, TensorView candidate_int_workspace,
    TensorView candidate_qo_indptr, TensorView candidate_kv_indptr, TensorView source_kv_indices,
    TensorView candidate_kv_len_arr, int64_t staged_int_workspace_bytes,
    int64_t live_kv_indices_length) {
  CheckPlanUpdateTensor(live_int_workspace, dl_uint8, "live_int_workspace");
  CheckPlanUpdateTensor(live_qo_indptr, dl_int32, "live_qo_indptr");
  CheckPlanUpdateTensor(live_kv_indptr, dl_int32, "live_kv_indptr");
  CheckPlanUpdateTensor(live_kv_indices, dl_int32, "live_kv_indices");
  CheckPlanUpdateTensor(live_kv_len_arr, dl_int32, "live_kv_len_arr");
  CheckPlanUpdateTensor(candidate_int_workspace, dl_uint8, "candidate_int_workspace");
  CheckPlanUpdateTensor(candidate_qo_indptr, dl_int32, "candidate_qo_indptr");
  CheckPlanUpdateTensor(candidate_kv_indptr, dl_int32, "candidate_kv_indptr");
  CheckPlanUpdateTensor(source_kv_indices, dl_int32, "source_kv_indices");
  CheckPlanUpdateTensor(candidate_kv_len_arr, dl_int32, "candidate_kv_len_arr");

  CheckSameDevice(live_int_workspace, live_qo_indptr, "live_qo_indptr");
  CheckSameDevice(live_int_workspace, live_kv_indptr, "live_kv_indptr");
  CheckSameDevice(live_int_workspace, live_kv_indices, "live_kv_indices");
  CheckSameDevice(live_int_workspace, live_kv_len_arr, "live_kv_len_arr");
  CheckSameDevice(live_int_workspace, candidate_int_workspace, "candidate_int_workspace");
  CheckSameDevice(live_int_workspace, candidate_qo_indptr, "candidate_qo_indptr");
  CheckSameDevice(live_int_workspace, candidate_kv_indptr, "candidate_kv_indptr");
  CheckSameDevice(live_int_workspace, source_kv_indices, "source_kv_indices");
  CheckSameDevice(live_int_workspace, candidate_kv_len_arr, "candidate_kv_len_arr");

  CheckEqualCapacity(live_qo_indptr, candidate_qo_indptr, "qo_indptr");
  CheckEqualCapacity(live_kv_indptr, candidate_kv_indptr, "kv_indptr");
  CheckEqualCapacity(live_kv_len_arr, candidate_kv_len_arr, "kv_len_arr");
  TVM_FFI_ICHECK_GE(staged_int_workspace_bytes, 0)
      << "staged_int_workspace_bytes must be nonnegative";
  TVM_FFI_ICHECK_LE(staged_int_workspace_bytes, live_int_workspace.numel())
      << "staged_int_workspace_bytes exceeds the live workspace capacity";
  TVM_FFI_ICHECK_LE(staged_int_workspace_bytes, candidate_int_workspace.numel())
      << "staged_int_workspace_bytes exceeds the candidate workspace capacity";
  TVM_FFI_ICHECK_GE(live_kv_indices_length, 0) << "live_kv_indices_length must be nonnegative";
  TVM_FFI_ICHECK_LE(live_kv_indices_length, live_kv_indices.numel())
      << "live_kv_indices_length exceeds the live index capacity";
  TVM_FFI_ICHECK_LE(live_kv_indices_length, source_kv_indices.numel())
      << "live_kv_indices_length exceeds the source index capacity";

  const uintptr_t live_indices_begin = reinterpret_cast<uintptr_t>(live_kv_indices.data_ptr());
  const uintptr_t live_indices_end = live_indices_begin + live_kv_indices.numel() * sizeof(int32_t);
  const uintptr_t source_indices_begin = reinterpret_cast<uintptr_t>(source_kv_indices.data_ptr());
  const uintptr_t source_indices_end =
      source_indices_begin + source_kv_indices.numel() * sizeof(int32_t);
  TVM_FFI_ICHECK(source_indices_end <= live_indices_begin ||
                 live_indices_end <= source_indices_begin)
      << "source_kv_indices must not overlap live_kv_indices";

  ffi::CUDADeviceGuard device_guard(live_int_workspace.device().device_id);
  const cudaStream_t stream = get_stream(live_int_workspace.device());
  const int64_t work_elements =
      std::max(staged_int_workspace_bytes,
               std::max(live_qo_indptr.numel(),
                        std::max(live_kv_indptr.numel(),
                                 std::max(live_kv_len_arr.numel(), live_kv_indices_length))));
  CommitBatchMLACudaGraphPlanUpdateKernel<<<kPlanUpdateBlocks, kPlanUpdateThreads, 0, stream>>>(
      static_cast<uint8_t*>(live_int_workspace.data_ptr()),
      static_cast<int32_t*>(live_qo_indptr.data_ptr()),
      static_cast<int32_t*>(live_kv_indptr.data_ptr()),
      static_cast<int32_t*>(live_kv_indices.data_ptr()),
      static_cast<int32_t*>(live_kv_len_arr.data_ptr()),
      static_cast<const uint8_t*>(candidate_int_workspace.data_ptr()),
      static_cast<const int32_t*>(candidate_qo_indptr.data_ptr()),
      static_cast<const int32_t*>(candidate_kv_indptr.data_ptr()),
      static_cast<const int32_t*>(source_kv_indices.data_ptr()),
      static_cast<const int32_t*>(candidate_kv_len_arr.data_ptr()), staged_int_workspace_bytes,
      live_qo_indptr.numel(), live_kv_indptr.numel(), live_kv_len_arr.numel(),
      live_kv_indices_length, work_elements);
  const cudaError_t error = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(error, cudaSuccess)
      << "CommitBatchMLACudaGraphPlanUpdate launch failed: " << cudaGetErrorString(error);
}
