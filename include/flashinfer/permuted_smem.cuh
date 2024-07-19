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

// Use 128bit as the granularity to fetch/store data per thread to maximize memory bandwidth
using b128_t = uint4;

/*!
 * \brief Compute the number of elements that can be stored in a b128_t.
 * \tparam T The data type of the elements.
 */
template <typename T>
constexpr __host__ __device__ __forceinline__ uint32_t num_elems_per_128b() {
  return sizeof(b128_t) / sizeof(T);
}

/*!
 * \brief The shared memory wrapper.
 */
struct smem_t {
  // The base pointer.
  b128_t* base;
  __device__ __forceinline__ smem_t() : base(nullptr) {}
  template <typename T>
  __device__ __forceinline__ smem_t(T* base) : base((b128_t*)base) {}

  /*!
   * \brief Compute the element offset given coordinates in a permuted shared memory.
   * \tparam stride The stride (in terms of b128_t's) in the permuted shared memory.
   * \param i The row index.
   * \param j The column index.
   */
  template <uint32_t stride>
  static __device__ __forceinline__ uint32_t get_permuted_offset(uint32_t i, uint32_t j) {
    return i * stride + (j ^ (i % 8));
  }

  template <uint32_t step_size>
  static __device__ __forceinline__ uint32_t advance_offset_by_column(uint32_t offset,
                                                                      uint32_t step_idx) {
    static_assert(step_size == 1 || step_size == 2 || step_size == 4 || step_size % 8 == 0, "Unsupported step size");
    if constexpr (step_size == 1) {
      return (offset ^ (0x1 + 0x2 * (step_idx % 2 == 1) + 0x4 * (step_idx % 4 == 3))) + (step_idx % 8 == 7) * 8;
    } else if constexpr (step_size == 2) {
      return (offset ^ (0x2 + (0x4 * (step_idx % 2 == 1)))) + (step_idx % 4 == 3) * 8;
    } else if constexpr (step_size == 4) {
      return (offset ^ 0x4) + (step_idx % 2 == 1) * 8;
    } else {
      // step_size % 8 == 0
      return offset + step_size;
    }
  }

  template <uint32_t step_size, uint32_t row_stride>
  static __device__ __forceinline__ uint32_t advance_offset_by_row(uint32_t offset) {
    static_assert(step_size == 4 || step_size % 8 == 0, "Unsupported step size");
    if constexpr (step_size == 4) {
      return (offset ^ 0x4) + step_size * row_stride;
    } else {
      // step_size % 8 == 0
      return offset + step_size * row_stride;
    }
  }

  template<typename dtype = half>
  __device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4(R, smem_ptr);

    if constexpr (std::is_same<dtype, __nv_fp8_e4m3>::value ||
        std::is_same<dtype, __nv_fp8_e5m2>::value) {
      // Convert fp8 fragments to fp16
      constexpr __nv_fp8_interpretation_t interp = std::is_same<dtype, __nv_fp8_e4m3>::value ?
          __nv_fp8_interpretation_t::__NV_E4M3 : __nv_fp8_interpretation_t::__NV_E5M2;
      constexpr uint32_t FULL_MASK = 0xffffffff;

      uint32_t lane_id = threadIdx.x;
      uint32_t t_idx_0 = (lane_id & ~0x3) + ((lane_id & 0x2) >> 1);
      uint32_t t_idx_1 = t_idx_0 + 2;
      uint32_t shift = (lane_id & 0x1) * 16;

      auto cnv = [](__half2_raw v) -> uint32_t {
        uint32_t res = v.x;
        return res | v.y << 16;
      };

      uint32_t val_0 = __shfl_sync(FULL_MASK, R[0], t_idx_0);
      uint32_t val_1 = __shfl_sync(FULL_MASK, R[0], t_idx_1);
      R[0] = cnv(__nv_cvt_fp8x2_to_halfraw2(val_0 >> shift, interp));
      R[1] = cnv(__nv_cvt_fp8x2_to_halfraw2(val_1 >> shift, interp));

      val_0 = __shfl_sync(FULL_MASK, R[2], t_idx_0);
      val_1 = __shfl_sync(FULL_MASK, R[2], t_idx_1);
      R[2] = cnv(__nv_cvt_fp8x2_to_halfraw2(val_0 >> shift, interp));
      R[3] = cnv(__nv_cvt_fp8x2_to_halfraw2(val_1 >> shift, interp));
    }
  }

  __device__ __forceinline__ void stmatrix_m8n8x4(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::stmatrix_m8n8x4(R, smem_ptr);
  }

  template<typename dtype = half>
  __device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4_trans(R, smem_ptr);

    if constexpr (std::is_same<dtype, __nv_fp8_e4m3>::value ||
            std::is_same<dtype, __nv_fp8_e5m2>::value) {
      // Convert fp8 fragments to fp16
      constexpr __nv_fp8_interpretation_t interp = std::is_same<dtype, __nv_fp8_e4m3>::value ?
          __nv_fp8_interpretation_t::__NV_E4M3 : __nv_fp8_interpretation_t::__NV_E5M2;

      constexpr uint32_t FULL_MASK = 0xffffffff;

      uint32_t lane_id = threadIdx.x;
      uint32_t t_idx_0 = (lane_id & 0x3) + ((lane_id & ~0x7) >> 1);
      uint32_t t_idx_1 = t_idx_0 + 16;
      uint32_t shift = (lane_id & 0x4) * 4;

      auto cnv = [](__half2_raw v) -> uint32_t {
        uint32_t res = v.x;
        return res | v.y << 16;
      };

      auto shffle_acbd = [](uint32_t v) -> uint32_t {
        return (v & 0xff0000ff) + ((v >> 8) & 0x0000ff00) + ((v << 8) & 0x00ff0000);
      };

      uint32_t val_0 = shffle_acbd(__shfl_sync(FULL_MASK, R[0], t_idx_0));
      uint32_t val_1 = shffle_acbd(__shfl_sync(FULL_MASK, R[0], t_idx_1));
      R[0] = cnv(__nv_cvt_fp8x2_to_halfraw2(val_0 >> shift, interp));
      R[2] = cnv(__nv_cvt_fp8x2_to_halfraw2(val_1 >> shift, interp));

      val_0 = shffle_acbd(__shfl_sync(FULL_MASK, R[1], t_idx_0));
      val_1 = shffle_acbd(__shfl_sync(FULL_MASK, R[1], t_idx_1));
      R[1] = cnv(__nv_cvt_fp8x2_to_halfraw2(val_0 >> shift, interp));
      R[3] = cnv(__nv_cvt_fp8x2_to_halfraw2(val_1 >> shift, interp));
    }
  }

  template <cp_async::SharedMemFillMode fill_mode, typename T>
  __device__ __forceinline__ void load_128b_async(uint32_t offset, const T* gptr, bool predicate) {
    b128_t* smem_ptr = base + offset;
    cp_async::pred_load_128b<cp_async::PrefetchMode::kPrefetch, fill_mode>(
        smem_ptr, reinterpret_cast<const b128_t*>(gptr), predicate);
  }

  template <typename T>
  __device__ __forceinline__ void load_128b_async(uint32_t offset, const T* gptr) {
    b128_t* smem_ptr = base + offset;
    cp_async::load_128b<cp_async::PrefetchMode::kPrefetch>(smem_ptr,
                                                           reinterpret_cast<const b128_t*>(gptr));
  }

  template <typename T>
  __device__ __forceinline__ void store_128b(uint32_t offset, T* gptr) {
    *reinterpret_cast<b128_t*>(gptr) = *(base + offset);
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_PERMUTED_SMEM_CUH_
