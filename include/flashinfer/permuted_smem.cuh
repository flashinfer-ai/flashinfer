/*
 * Copyright (c) 2023 by FlashInfer team.
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
#ifndef FLASHINFER_PERMUTED_SMEM_CUH_
#define FLASHINFER_PERMUTED_SMEM_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuda/pipeline>

#include "cp_async.cuh"
#include "mma.cuh"

namespace flashinfer {

// Each cell is 4 bytes.
using cell_t = uint4;

template <typename T>
constexpr __host__ __device__ __forceinline__ uint32_t cell_capacity() {
  return sizeof(cell_t) / sizeof(T);
}

struct smem_t {
  cell_t* base;
  uint32_t offset;
  __device__ __forceinline__ smem_t() : base(nullptr) {}
  template <typename T>
  __device__ __forceinline__ smem_t(T* base) : base((cell_t*)base) {}

  template <uint32_t stride>
  static __device__ __forceinline__ uint32_t get_permuted_offset(uint32_t i,
                                                                 uint32_t j) {
    return (i / 2) * stride * 2 + (j / 4) * 8 + (i % 2) * 4 +
           ((j % 4) ^ ((i / 2) % 4));
  }
  __device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t* R) {
    cell_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4(R, smem_ptr);
  }
  __device__ __forceinline__ void stmatrix_m8n8x4(uint32_t* R) {
    cell_t* smem_ptr = base + offset;
    mma::stmatrix_m8n8x4(R, smem_ptr);
  }
  __device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t* R) {
    cell_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4_trans(R, smem_ptr);
  }
  template <bool fill_zero, typename T>
  __device__ __forceinline__ void load_128b_async(const T* gptr,
                                                  bool predicate) {
    cell_t* smem_ptr = base + offset;
    cp_async::pred_load_128b<true, fill_zero>(
        smem_ptr, reinterpret_cast<const cell_t*>(gptr), predicate);
  }
  template <bool fill_zero, typename T>
  __device__ __forceinline__ void load_128b_async(const T* gptr) {
    cell_t* smem_ptr = base + offset;
    cp_async::load_128b<true, fill_zero>(smem_ptr,
                                         reinterpret_cast<const cell_t*>(gptr));
  }
  template <typename T>
  __device__ __forceinline__ void store_128b(T* gptr) {
    *reinterpret_cast<cell_t*>(gptr) = *(base + offset);
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_PERMUTED_SMEM_CUH_
