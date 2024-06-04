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
#ifndef FLASHINFER_GROUP_GEMM_HANDLER_CUH_
#define FLASHINFER_GROUP_GEMM_HANDLER_CUH_

#include "../allocator.h"
#include "../utils.cuh"
#include "group_gemm_cutlass.cuh"
#include "group_gemm_lora.cuh"
#include "group_gemv.cuh"

namespace flashinfer {

namespace group_gemm {

enum class GroupGEMMKernelConfig {
  kGeneral,  // large d_in, d_out
  kShrink,   // large d_in, small d_out
  kExpand,   // small d_in, large d_out
};

class CutlassSegmentGEMMHandler {
 public:
  cutlass::gemm::GemmCoord* GetProblemSizes() const { return problem_sizes_; }

  template <typename DType>
  DType** GetXPtr() {
    return static_cast<DType**>(x_data_);
  }

  template <typename DType>
  DType** GetWPtr() {
    return static_cast<DType**>(w_data_);
  }

  template <typename DType>
  DType** GetYPtr() {
    return static_cast<DType**>(y_data_);
  }

  int64_t* GetLdX() const { return static_cast<int64_t*>(ld_x_); }

  int64_t* GetLdY() const { return static_cast<int64_t*>(ld_y_); }

  int64_t* GetLdW() const { return static_cast<int64_t*>(ld_w_); }

  int64_t GetBatchSize() const { return batch_size_; }

  bool IsWeightColumnMajor() const { return w_column_major_; }

  template <typename DType>
  cudaError_t RegisterProblem(void* buffer, size_t workspace_size_in_bytes, int64_t* xy_indptr_d,
                              int64_t* w_indices_d, size_t batch_size, size_t d_in, size_t d_out,
                              bool weight_column_major) {
    problem_registered_ = true;
    batch_size_ = batch_size;
    w_column_major_ = weight_column_major;

    AlignedAllocator allocator(buffer_, workspace_size_in_bytes);
    problem_sizes_ = allocator.aligned_alloc<cutlass::gemm::GemmCoord>(batch_size, 16);
    x_data_ = allocator.aligned_alloc<DType**>(batch_size, 16);
    w_data_ = allocator.aligned_alloc<DType**>(batch_size, 16);
    y_data_ = allocator.aligned_alloc<DType**>(batch_size, 16);
    ld_x_ = allocator.aligned_alloc<int64_t>(batch_size, 16);
    ld_w_ = allocator.aligned_alloc<int64_t>(batch_size, 16);
    ld_y_ = allocator.aligned_alloc<int64_t>(batch_size, 16);

    auto compute_args_kernel = compute_cutlass_group_gemm_args<DType>;

    void* args[] = {(void*)&problem_sizes_, (void*)&x_data_,     (void*)&w_data_,
                    (void*)&y_data_,        (void*)&ld_x_,       (void*)&ld_w_,
                    (void*)&ld_y_,          (void*)&xy_indptr_d, (void*)&w_indices_d,
                    (void*)&d_in,           (void*)&d_out,       (void*)&w_column_major_};

    FLASHINFER_CUDA_CALL(
        cudaLaunchKernel((void*)compute_args_kernel, batch_size, 1, args, 0, stream_));

    return cudaSuccess;
  }

  cudaStream_t GetCUDAStream() const { return stream_; }

  void SetCUDAStream(cudaStream_t stream) { stream_ = stream; }

  bool IsProblemRegistered() const { return problem_registered_; }

  CutlassSegmentGEMMHandler() {}

  ~CutlassSegmentGEMMHandler() {}

 private:
  bool problem_registered_;
  void* buffer_;
  cudaStream_t stream_;
  bool w_column_major_;
  size_t batch_size_;
  cutlass::gemm::GemmCoord* problem_sizes_;
  void* x_data_;
  void* w_data_;
  void* y_data_;
  void* ld_x_;
  void* ld_w_;
  void* ld_y_;
};

}  // namespace group_gemm

}  // namespace flashinfer

#endif  // FLASHINFER_GROUP_GEMM_HANDLER_CUH_
