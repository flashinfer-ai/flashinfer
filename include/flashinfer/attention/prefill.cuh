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
#ifndef FLASHINFER_PREFILL_CUH_
#define FLASHINFER_PREFILL_CUH_
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifdef FLASHINFER_ENABLE_FP8
#include <cuda_fp8.h>
#endif
#include <cuda_runtime.h>

#include <optional>
#include <tuple>

#include "../cp_async.cuh"
#include "../layout.cuh"
#include "../math.cuh"
#include "../mma.cuh"
#include "../page.cuh"
#include "../permuted_smem.cuh"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "cascade.cuh"
#include "handler.cuh"
#include "state.cuh"

namespace flashinfer {

namespace cg = cooperative_groups;
using cp_async::SharedMemFillMode;
using mma::MMAMode;

constexpr uint32_t warp_size = 32;

namespace {

template <typename DTypeQKAccum>
constexpr bool is_invalid_configuration(uint32_t num_frags_x, uint32_t num_frags_y,
                                        uint32_t num_frags_z, uint32_t num_warps_x,
                                        uint32_t num_warps_z) {
  return ((num_frags_y < 4) || (num_frags_y == 4 && num_frags_z % 2 == 1) ||
          (num_frags_y > 4 && num_frags_y % 8 != 0) ||
          (num_frags_x * (8 * num_frags_y + 2 * sizeof(DTypeQKAccum) * num_frags_z) >= 256));
}

/*!
 * \brief Return x - y if x > y, otherwise return 0.
 */
__device__ __forceinline__ uint32_t sub_if_greater_or_zero(uint32_t x, uint32_t y) {
  return (x > y) ? x - y : 0U;
}

template <uint32_t num_warps_x, uint32_t num_warps_z>
__device__ __forceinline__ uint32_t get_warp_idx_x() {
  if constexpr (num_warps_x == 1) {
    return 0;
  } else {
    return threadIdx.y;
  }
}

template <uint32_t num_warps_x, uint32_t num_warps_z>
__device__ __forceinline__ uint32_t get_warp_idx_z() {
  if constexpr (num_warps_z == 1) {
    return 0;
  } else {
    return threadIdx.z;
  }
}

template <uint32_t num_warps_x, uint32_t num_warps_z>
__device__ __forceinline__ uint32_t get_warp_idx() {
  return get_warp_idx_z<num_warps_x, num_warps_z>() * num_warps_x +
         get_warp_idx_x<num_warps_x, num_warps_z>();
}

enum class FragLayout {
  kRowMajor,
  kColMajor,
};

/*!
 * \brief Apply Llama style rotary embedding to two 16x16 fragments.
 * \tparam FragLayout The layout of the input fragments.
 * \tparam T The data type of the input fragments.
 * \param x_first_half First fragment x[offset:offset+16, j*16:(j+1)*16]
 * \param x_second_half Second fragment x[offset:offset*16, j*16+d/2:(j+1)*16+d/2]
 * \param rope_freq Rope frequency
 * \param offset The offset of the first row in both fragments.
 * \param scale A scale factor applied to the result (used to multiply sm_scale).
 * \note The sin/cos computation is slow, especially for A100 GPUs which has low
 *   non tensor-ops flops, will optimize in the future.
 */
template <FragLayout frag_layout, uint32_t group_size, typename T>
__device__ __forceinline__ void frag_apply_llama_rope(T* x_first_half, T* x_second_half,
                                                      const float* rope_freq, uint32_t offset,
                                                      float scale = 1.f) {
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    uint32_t i, j;
    if constexpr (frag_layout == FragLayout::kRowMajor) {
      // 0 1 | 4 5
      // ---------
      // 2 3 | 6 7
      i = ((reg_id % 4) / 2);
      j = (reg_id / 4);
    } else {
      // 0 1 | 2 3
      // ---------
      // 4 5 | 6 7
      i = reg_id / 4;
      j = (reg_id % 4) / 2;
    }
    __sincosf(float(offset + (8 / group_size) * i) * rope_freq[2 * j + reg_id % 2], &sin, &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin) * scale;
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin) * scale;
  }
}

template <FragLayout frag_layout, uint32_t group_size, typename T, typename IdType>
__device__ __forceinline__ void frag_apply_llama_rope_with_pos(T* x_first_half, T* x_second_half,
                                                               const float* rope_freq,
                                                               uint32_t offset,
                                                               const IdType* q_offset,
                                                               float scale = 1.f) {
  float pos[2] = {static_cast<float>(q_offset[offset]),
                  static_cast<float>(q_offset[offset + (8 / group_size)])};
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    uint32_t i, j;
    if constexpr (frag_layout == FragLayout::kRowMajor) {
      // 0 1 | 4 5
      // ---------
      // 2 3 | 6 7
      i = ((reg_id % 4) / 2);
      j = (reg_id / 4);
    } else {
      // 0 1 | 2 3
      // ---------
      // 4 5 | 6 7
      i = reg_id / 4;
      j = (reg_id % 4) / 2;
    }
    __sincosf(pos[i] * rope_freq[2 * j + reg_id % 2], &sin, &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin) * scale;
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin) * scale;
  }
}

/*!
 * \brief Produce k/v fragments from global memory to shared memory.
 * \tparam fill_mode The fill mode of the shared memory.
 * \tparam num_frags_y The number of fragments in y dimension.
 * \tparam num_frags_z The number of fragments in z dimension.
 * \tparam num_warps The number of warps in the threadblock.
 * \tparam kv_layout The layout of the input tensor.
 * \tparam group_size The number of qo heads that maps to a kv head (used in GQA).
 * \tparam T The data type of the input tensor.
 * \param smem The shared memory to store kv fragments.
 * \param gptr The global memory pointer.
 * \param qkv_info The tensor info of the input tensor.
 * \param kv_idx_base The base kv index.
 * \param kv_len The length of kv tensor.
 */
template <SharedMemFillMode fill_mode, uint32_t num_warps_x, uint32_t num_warps_z,
          uint32_t num_frags_y, uint32_t num_frags_z, typename T>
__device__ __forceinline__ void produce_kv(smem_t smem, uint32_t* smem_offset, T** gptr,
                                           const uint32_t kv_n_stride, const uint32_t kv_idx_base,
                                           const uint32_t kv_len, const uint32_t warp_idx,
                                           const uint32_t lane_idx) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_warps = num_warps_x * num_warps_z;
  constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<T>();
  uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / 8;
  // NOTE(Zihao): num_frags_z * 4 / num_warps_x = num_warps_z * num_frags_z * 4 / num_warps
  static_assert(num_frags_z * 4 % num_warps_x == 0);
#pragma unroll
  for (uint32_t i = 0; i < num_frags_z * 4 / num_warps_x; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < num_frags_y / 4; ++j) {
      smem.load_128b_async<fill_mode>(*smem_offset, *gptr, kv_idx < kv_len);
      *smem_offset = smem.advance_offset_by_column<8>(*smem_offset, j);
      *gptr += 8 * num_elems_per_128b<T>();
    }
    kv_idx += num_warps * 4;
    *smem_offset = smem.advance_offset_by_row<num_warps * 4, channel_size_128b_in>(*smem_offset) -
                   2 * num_frags_y;
    *gptr += num_warps * 4 * kv_n_stride - 2 * num_frags_y * num_elems_per_128b<T>();
  }
  *smem_offset -= num_warps_z * num_frags_z * 16 * channel_size_128b_in;
}

template <bool produce_v, uint32_t page_size, uint32_t num_warps_x, uint32_t num_warps_z,
          uint32_t num_frags_y, uint32_t num_frags_z, PageStorage page_storage, QKVLayout kv_layout,
          typename DType, typename IdType>
__device__ __forceinline__ void page_produce_kv(
    smem_t smem, uint32_t* smem_offset,
    paged_kv_t<page_storage, kv_layout, DType, IdType>& paged_kv, const uint32_t kv_idx_base,
    const uint32_t page_iter_base, const uint32_t kv_len, const IdType last_indptr,
    const uint32_t warp_idx, const uint32_t lane_idx) {
  constexpr SharedMemFillMode fill_mode =
      produce_v ? SharedMemFillMode::kFillZero : SharedMemFillMode::kNoFill;
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_warps = num_warps_x * num_warps_z;
  constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DType>();
  const uint32_t kv_head_idx = blockIdx.z;
  uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / 8;
  // NOTE(Zihao): num_frags_z * 4 / num_warps_x = num_warps_z * num_frags_z * 4 / num_warps
  static_assert(num_frags_z * 4 % num_warps_x == 0);
  if constexpr (page_size % 4 == 0) {
#pragma unroll
    for (uint32_t i = 0; i < num_frags_z * 4 / num_warps_x; ++i) {
      const uint32_t page_iter = page_iter_base + (4 * num_warps * i + warp_idx * 4) / page_size;
      const uint32_t entry_idx = (4 * num_warps * i + warp_idx * 4) % page_size + lane_idx / 8;
      DType* gptr = produce_v ? paged_kv.protective_get_v_ptr(
                                    page_iter, kv_head_idx, entry_idx,
                                    (lane_idx % 8) * num_elems_per_128b<DType>(), last_indptr)
                              : paged_kv.protective_get_k_ptr(
                                    page_iter, kv_head_idx, entry_idx,
                                    (lane_idx % 8) * num_elems_per_128b<DType>(), last_indptr);
#pragma unroll
      for (uint32_t j = 0; j < num_frags_y / 4; ++j) {
        smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
        *smem_offset = smem.advance_offset_by_column<8>(*smem_offset, j);
        gptr += 8 * num_elems_per_128b<DType>();
      }
      kv_idx += num_warps * 4;
      *smem_offset = smem.advance_offset_by_row<num_warps * 4, channel_size_128b_in>(*smem_offset) -
                     2 * num_frags_y;
    }
    *smem_offset -= num_warps_z * num_frags_z * 16 * channel_size_128b_in;
  } else {
#pragma unroll
    for (uint32_t i = 0; i < num_frags_z * 4 / num_warps_x; ++i) {
      const uint32_t page_iter =
          page_iter_base + (4 * num_warps * i + warp_idx * 4 + lane_idx / 8) / page_size;
      const uint32_t entry_idx = (4 * num_warps * i + warp_idx * 4 + lane_idx / 8) % page_size;
      DType* gptr = produce_v ? paged_kv.protective_get_v_ptr(
                                    page_iter, kv_head_idx, entry_idx,
                                    (lane_idx % 8) * num_elems_per_128b<DType>(), last_indptr)
                              : paged_kv.protective_get_k_ptr(
                                    page_iter, kv_head_idx, entry_idx,
                                    (lane_idx % 8) * num_elems_per_128b<DType>(), last_indptr);
#pragma unroll
      for (uint32_t j = 0; j < num_frags_y / 4; ++j) {
        smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
        *smem_offset = smem.advance_offset_by_column<8>(*smem_offset, j);
        gptr += 8 * num_elems_per_128b<DType>();
      }
      kv_idx += num_warps * 4;
      *smem_offset = smem.advance_offset_by_row<num_warps * 4, channel_size_128b_in>(*smem_offset) -
                     2 * num_frags_y;
    }
    *smem_offset -= num_warps_z * num_frags_z * 16 * channel_size_128b_in;
  }
}

template <uint32_t num_frags_y>
__device__ __forceinline__ void init_rope_freq(float (*rope_freq)[4],
                                               const float log2_rope_rcp_scale,
                                               const float log2_rope_rcp_theta) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  const uint32_t lane_idx = threadIdx.x;
#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_y / 2; ++fy) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      rope_freq[fy][j] = math::ptx_exp2(
          log2_rope_rcp_scale +
          log2_rope_rcp_theta *
              float(2 * ((fy * 16 + (j / 2) * 8 + (lane_idx % 4) * 2 + (j % 2)) % (head_dim / 2))) /
              float(head_dim));
    }
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_y, typename DTypeQKAccum>
__device__ __forceinline__ void init_states(float (*o_frag)[num_frags_y][8], DTypeQKAccum (*m)[2],
                                            float (*d)[2]) {
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        o_frag[fx][fy][reg_id] = 0.f;
      }
    }
  }
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      m[fx][j] = DTypeQKAccum(-5e4);
      d[fx][j] = 1.f;
    }
  }
}

template <uint32_t group_size, uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_x,
          uint32_t num_frags_y, typename DTypeIn>
__device__ __forceinline__ void load_q_global_smem(uint32_t q_idx_base,
                                                   const uint32_t qo_upper_bound,
                                                   DTypeIn* q_ptr_base, const uint32_t qo_n_stride,
                                                   const uint32_t qo_h_stride, smem_t* q_smem) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
  constexpr uint32_t num_warps = num_warps_x * num_warps_z;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warp_idx = get_warp_idx<num_warps_x, num_warps_z>();
  uint32_t q_smem_offset_w =
      smem_t::get_permuted_offset<channel_size_128b_in>(warp_idx * 4 + lane_idx / 8, lane_idx % 8);

  uint32_t q_idx = q_idx_base + (warp_idx * 4 + lane_idx / 8) / group_size;
  DTypeIn* q_ptr = q_ptr_base + ((warp_idx * 4 + lane_idx / 8) / group_size) * qo_n_stride +
                   ((warp_idx * 4 + lane_idx / 8) % group_size) * qo_h_stride;
  static_assert(num_frags_x * 4 % num_warps_z == 0);
  // NOTE(Zihao): num_warps_x * num_frags_x * 4 / num_warps = num_frags_x * 4 / num_warps_z
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x * 4 / num_warps_z; ++fx) {
    for (uint32_t fy = 0; fy < num_frags_y / 4; ++fy) {
      // load q fragment from gmem to smem
      q_smem->load_128b_async<SharedMemFillMode::kNoFill>(q_smem_offset_w, q_ptr,
                                                          q_idx < qo_upper_bound);
      q_smem_offset_w = q_smem->advance_offset_by_column<8>(q_smem_offset_w, fy);
      q_ptr += 8 * num_elems_per_128b<DTypeIn>();
    }
    q_idx += (num_warps * 4) / group_size;
    q_ptr += ((num_warps * 4) / group_size) * qo_n_stride -
             2 * num_frags_y * num_elems_per_128b<DTypeIn>();
    q_smem_offset_w =
        q_smem->advance_offset_by_row<num_warps * 4, channel_size_128b_in>(q_smem_offset_w) -
        2 * num_frags_y;
  }
}

template <uint32_t group_size, uint32_t num_frags_x, uint32_t num_frags_y, typename DTypeIn>
__device__ __forceinline__ void q_smem_inplace_apply_rotary_multiply_sm_scale(
    const uint32_t q_idx_base, const uint32_t qo_len, const uint32_t kv_len, smem_t* q_smem,
    uint32_t* q_smem_offset_r, float (*rope_freq)[4], const float sm_scale) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
  const uint32_t lane_idx = threadIdx.x;
  uint32_t q_frag_local[2][4];
  static_assert(num_frags_y % 4 == 0, "num_frags_y must be a multiple of 4");
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
    uint32_t q_idx = q_idx_base + (fx * 16 + lane_idx / 4) / group_size;
    uint32_t q_smem_offset_r_first_half = *q_smem_offset_r;
#pragma unroll
    for (uint32_t fyi = 0; fyi < num_frags_y / 2; ++fyi) {
      q_smem->ldmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
      uint32_t q_smem_offset_r_last_half =
          q_smem->advance_offset_by_column<num_frags_y>(q_smem_offset_r_first_half, 0);
      q_smem->ldmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
      frag_apply_llama_rope<FragLayout::kRowMajor, group_size, DTypeIn>(
          (DTypeIn*)q_frag_local[0], (DTypeIn*)q_frag_local[1], rope_freq[fyi],
          q_idx + kv_len - qo_len, sm_scale);
      q_smem->stmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
      q_smem->stmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
      q_smem_offset_r_first_half =
          q_smem->advance_offset_by_column<2>(q_smem_offset_r_first_half, fyi);
    }
    *q_smem_offset_r += 16 * channel_size_128b_in;
  }
  *q_smem_offset_r -= num_frags_x * 16 * channel_size_128b_in;
}

template <uint32_t group_size, uint32_t num_frags_x, uint32_t num_frags_y, typename DTypeIn,
          typename IdType>
__device__ __forceinline__ void q_smem_inplace_apply_rotary_with_pos_multiply_sm_scale(
    const uint32_t q_idx_base, const IdType* q_offset, smem_t* q_smem, uint32_t* q_smem_offset_r,
    float (*rope_freq)[4], const float sm_scale) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
  const uint32_t lane_idx = threadIdx.x;
  uint32_t q_frag_local[2][4];
  static_assert(num_frags_y % 4 == 0, "num_frags_y must be a multiple of 4");
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
    uint32_t q_idx = q_idx_base + (fx * 16 + lane_idx / 4) / group_size;
    uint32_t q_smem_offset_r_first_half = *q_smem_offset_r;
#pragma unroll
    for (uint32_t fyi = 0; fyi < num_frags_y / 2; ++fyi) {
      q_smem->ldmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
      uint32_t q_smem_offset_r_last_half =
          q_smem->advance_offset_by_column<num_frags_y>(q_smem_offset_r_first_half, 0);
      q_smem->ldmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
      frag_apply_llama_rope_with_pos<FragLayout::kRowMajor, group_size, DTypeIn>(
          (DTypeIn*)q_frag_local[0], (DTypeIn*)q_frag_local[1], rope_freq[fyi], q_idx, q_offset,
          sm_scale);
      q_smem->stmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
      q_smem->stmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
      q_smem_offset_r_first_half =
          q_smem->advance_offset_by_column<2>(q_smem_offset_r_first_half, fyi);
    }
    *q_smem_offset_r += 16 * channel_size_128b_in;
  }
  *q_smem_offset_r -= num_frags_x * 16 * channel_size_128b_in;
}

template <uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_x, uint32_t num_frags_y,
          typename DTypeIn>
__device__ __forceinline__ void q_smem_inplace_multiply_sm_scale(smem_t* q_smem,
                                                                 const float sm_scale,
                                                                 const uint32_t warp_idx,
                                                                 const uint32_t lane_idx) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_warps = num_warps_x * num_warps_z;
  constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
  // NOTE(Zihao): num_warps_x * num_frags_x * 16 * head_dim
#pragma unroll
  for (uint32_t i = 0; i < num_frags_x * head_dim / (num_warps_z * 16); ++i) {
    vec_t<DTypeIn, 8> tmp;
    tmp.load((DTypeIn*)(q_smem->base) + (i * num_warps + warp_idx) * 256 + lane_idx * 8);
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
      tmp[reg_id] *= sm_scale;
    }
    tmp.store((DTypeIn*)(q_smem->base) + (i * num_warps + warp_idx) * 256 + lane_idx * 8);
  }
}

template <uint32_t num_frags_y, uint32_t num_frags_z, typename DTypeIn>
__device__ __forceinline__ void k_smem_inplace_apply_rotary(
    const uint32_t kv_idx_base, smem_t* k_smem, uint32_t* k_smem_offset_r, float (*rope_freq)[4],
    const uint32_t warp_idx, const uint32_t lane_idx) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
  uint32_t k_frag_local[2][4];
  if constexpr (num_frags_y == 4) {
    // horizontal-axis: y
    // vertical-axis: z
    //         | 1-16       | 16-32      | 32-48      | 48-64      |
    // | 1-16  | warp_idx=0 | warp_idx=1 | warp_idx=0 | warp_idx=1 |
    // | 16-32 | warp_idx=2 | warp_idx=3 | warp_idx=2 | warp_idx=3 |
    static_assert(num_frags_z % 2 == 0,
                  "when num_frags_y == 4, num_frags_z must be a multiple of 2");
    uint32_t kv_idx = kv_idx_base + (warp_idx / 2) * 16 + lane_idx / 4;
    *k_smem_offset_r =
        (*k_smem_offset_r ^ (0x2 * (warp_idx % 2))) + (warp_idx / 2) * 16 * channel_size_128b_in;
#pragma unroll
    for (uint32_t i = 0; i < num_frags_z / 2; ++i) {
      // uint32_t fz = warp_idx / 2 + i * 2;
      uint32_t k_smem_offset_r_first_half = *k_smem_offset_r;
      uint32_t fyi = (warp_idx % 2);
      k_smem->ldmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
      uint32_t k_smem_offset_r_last_half =
          k_smem->advance_offset_by_column<4>(k_smem_offset_r_first_half, 0);
      k_smem->ldmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
      frag_apply_llama_rope<FragLayout::kColMajor, 1, DTypeIn>(
          (DTypeIn*)k_frag_local[0], (DTypeIn*)k_frag_local[1], rope_freq[fyi], kv_idx);
      k_smem->stmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
      k_smem->stmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
      *k_smem_offset_r += 32 * channel_size_128b_in;
      kv_idx += 32;
    }
    *k_smem_offset_r = (*k_smem_offset_r ^ (0x2 * (warp_idx % 2))) -
                       ((warp_idx / 2) + num_frags_z) * 16 * channel_size_128b_in;
  } else {
    static_assert(num_frags_y % 8 == 0);
    // horizontal axis: y
    // vertical axis: z
    //         | 1-16       | 16-32      | 32-48      | 48-64      | ...
    // | 1-16  | warp_idx=0 | warp_idx=1 | warp_idx=2 | warp_idx=3 | ...
    // | 16-32 | warp_idx=0 | warp_idx=1 | warp_idx=2 | warp_idx=3 | ...
    // ...
    uint32_t kv_idx = kv_idx_base + lane_idx / 4;
    *k_smem_offset_r = *k_smem_offset_r ^ (0x2 * warp_idx);
#pragma unroll
    for (uint32_t i = 0; i < num_frags_z; ++i) {
      uint32_t k_smem_offset_r_first_half = *k_smem_offset_r;
#pragma unroll
      for (uint32_t j = 0; j < num_frags_y / 8; ++j) {
        uint32_t fyi = warp_idx + j * 4;
        k_smem->ldmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
        uint32_t k_smem_offset_r_last_half =
            k_smem->advance_offset_by_column<num_frags_y>(k_smem_offset_r_first_half, 0);
        k_smem->ldmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
        frag_apply_llama_rope<FragLayout::kColMajor, 1, DTypeIn>(
            (DTypeIn*)k_frag_local[0], (DTypeIn*)k_frag_local[1], rope_freq[fyi], kv_idx);
        k_smem->stmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
        k_smem->stmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
        k_smem_offset_r_first_half =
            k_smem->advance_offset_by_column<8>(k_smem_offset_r_first_half, 0);
      }
      *k_smem_offset_r += 16 * channel_size_128b_in;
      kv_idx += 16;
    }
    *k_smem_offset_r =
        (*k_smem_offset_r ^ (0x2 * warp_idx)) - num_frags_z * 16 * channel_size_128b_in;
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z, typename DTypeIn,
          typename DTypeQKAccum>
__device__ __forceinline__ void compute_qk(smem_t* q_smem, uint32_t* q_smem_offset_r,
                                           smem_t* k_smem, uint32_t* k_smem_offset_r,
                                           DTypeQKAccum (*s_frag)[num_frags_z][8]) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
  uint32_t a_frag[num_frags_x][4], b_frag[4];
  // compute q*k^T
#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
      q_smem->ldmatrix_m8n8x4(*q_smem_offset_r, a_frag[fx]);
      *q_smem_offset_r = q_smem->advance_offset_by_row<16, channel_size_128b_in>(*q_smem_offset_r);
    }

    *q_smem_offset_r = q_smem->advance_offset_by_column<2>(*q_smem_offset_r, fy) -
                       num_frags_x * 16 * channel_size_128b_in;

#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      k_smem->ldmatrix_m8n8x4(*k_smem_offset_r, b_frag);
      *k_smem_offset_r = k_smem->advance_offset_by_row<16, channel_size_128b_in>(*k_smem_offset_r);
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        if constexpr (std::is_same<DTypeQKAccum, float>::value) {
          if (fy == 0) {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn, MMAMode::kInit>(s_frag[fx][fz],
                                                                               a_frag[fx], b_frag);
          } else {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn>(s_frag[fx][fz], a_frag[fx], b_frag);
          }
        } else if (std::is_same<DTypeQKAccum, half>::value) {
          if (fy == 0) {
            mma::mma_sync_m16n16k16_row_col_f16f16f16<MMAMode::kInit>((uint32_t*)s_frag[fx][fz],
                                                                      a_frag[fx], b_frag);
          } else {
            mma::mma_sync_m16n16k16_row_col_f16f16f16((uint32_t*)s_frag[fx][fz], a_frag[fx],
                                                      b_frag);
          }
        }
      }
    }
    *k_smem_offset_r = k_smem->advance_offset_by_column<2>(*k_smem_offset_r, fy) -
                       num_frags_z * 16 * channel_size_128b_in;
  }
  *q_smem_offset_r -= num_frags_y * 2;
  *k_smem_offset_r -= num_frags_y * 2;
}

template <uint32_t group_size, uint32_t num_frags_x, uint32_t num_frags_z, typename T>
__device__ __forceinline__ void apply_alibi_bias(const uint32_t qo_idx_base,
                                                 const uint32_t kv_idx_base, const int32_t q_offset,
                                                 float (*alibi_slope)[2],
                                                 T (*s_frag)[num_frags_z][8]) {
  const int32_t lane_idx = threadIdx.x;
#pragma unroll
  for (int32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (int32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
      for (int32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const int32_t q_idx = qo_idx_base +
                              (fx * 16 + lane_idx / 4 + 8 * ((reg_id % 4) / 2)) / group_size,
                      kv_idx = kv_idx_base + fz * 16 + 2 * (lane_idx % 4) + 8 * (reg_id / 4) +
                               reg_id % 2;
        s_frag[fx][fz][reg_id] +=
            T(alibi_slope[fx][(reg_id % 4) / 2]) * T(kv_idx - q_idx - q_offset);
      }
    }
  }
}

template <bool partition_kv, bool causal, uint32_t group_size, uint32_t num_frags_x,
          uint32_t num_frags_y, uint32_t num_frags_z, typename DTypeQKAccum>
__device__ __forceinline__ void mask_s(const uint32_t qo_idx_base, const uint32_t kv_idx_base,
                                       const uint32_t qo_len, const uint32_t kv_len,
                                       const uint32_t chunk_end,
                                       DTypeQKAccum (*s_frag)[num_frags_z][8]) {
  const uint32_t lane_idx = threadIdx.x;
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const uint32_t q_idx = qo_idx_base +
                               (fx * 16 + lane_idx / 4 + 8 * ((reg_id % 4) / 2)) / group_size,
                       kv_idx = kv_idx_base + fz * 16 + 2 * (lane_idx % 4) + 8 * (reg_id / 4) +
                                reg_id % 2;
        const bool out_of_boundary =
            (causal ? (kv_idx > kv_len + q_idx - qo_len || (partition_kv && kv_idx >= chunk_end))
                    : kv_idx >= chunk_end);
        s_frag[fx][fz][reg_id] = out_of_boundary ? DTypeQKAccum(-5e4) : s_frag[fx][fz][reg_id];
      }
    }
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z, typename DTypeQKAccum>
__device__ __forceinline__ void update_mdo_states(DTypeQKAccum (*s_frag)[num_frags_z][8],
                                                  float (*o_frag)[num_frags_y][8],
                                                  DTypeQKAccum (*m)[2], float (*d)[2]) {
  if constexpr (std::is_same<DTypeQKAccum, float>::value) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        float m_prev = m[fx][j];
#pragma unroll
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          float m_local = max(max(s_frag[fx][fz][j * 2 + 0], s_frag[fx][fz][j * 2 + 1]),
                              max(s_frag[fx][fz][j * 2 + 4], s_frag[fx][fz][j * 2 + 5]));
          m[fx][j] = max(m[fx][j], m_local);
        }
        m[fx][j] = max(m[fx][j], math::shfl_xor_sync(m[fx][j], 0x2));
        m[fx][j] = max(m[fx][j], math::shfl_xor_sync(m[fx][j], 0x1));

        float o_scale = math::ptx_exp2(m_prev - m[fx][j]);
        d[fx][j] *= o_scale;
#pragma unroll
        for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
          o_frag[fx][fy][j * 2 + 0] *= o_scale;
          o_frag[fx][fy][j * 2 + 1] *= o_scale;
          o_frag[fx][fy][j * 2 + 4] *= o_scale;
          o_frag[fx][fy][j * 2 + 5] *= o_scale;
        }
#pragma unroll
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          s_frag[fx][fz][j * 2 + 0] = math::ptx_exp2(s_frag[fx][fz][j * 2 + 0] - m[fx][j]);
          s_frag[fx][fz][j * 2 + 1] = math::ptx_exp2(s_frag[fx][fz][j * 2 + 1] - m[fx][j]);
          s_frag[fx][fz][j * 2 + 4] = math::ptx_exp2(s_frag[fx][fz][j * 2 + 4] - m[fx][j]);
          s_frag[fx][fz][j * 2 + 5] = math::ptx_exp2(s_frag[fx][fz][j * 2 + 5] - m[fx][j]);
        }
      }
    }
  } else if constexpr (std::is_same<DTypeQKAccum, half>::value) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
      half m_prev[2];
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        m_prev[j] = m[fx][j];
#pragma unroll
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          half2 m_local =
              __hmax2(*(half2*)&s_frag[fx][fz][j * 2], *(half2*)&s_frag[fx][fz][j * 2 + 4]);
          m[fx][j] = __hmax(m[fx][j], __hmax(m_local.x, m_local.y));
        }
      }
      *(half2*)&m[fx] = __hmax2(*(half2*)&m[fx], math::shfl_xor_sync(*(half2*)&m[fx], 0x2));
      *(half2*)&m[fx] = __hmax2(*(half2*)&m[fx], math::shfl_xor_sync(*(half2*)&m[fx], 0x1));
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        float o_scale = math::ptx_exp2(float(m_prev[j] - m[fx][j]));
        d[fx][j] *= o_scale;
#pragma unroll
        for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
          o_frag[fx][fy][j * 2 + 0] *= o_scale;
          o_frag[fx][fy][j * 2 + 1] *= o_scale;
          o_frag[fx][fy][j * 2 + 4] *= o_scale;
          o_frag[fx][fy][j * 2 + 5] *= o_scale;
        }
        half2 m2 = make_half2(m[fx][j], m[fx][j]);
#pragma unroll
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          *(half2*)&s_frag[fx][fz][j * 2] = math::ptx_exp2(*(half2*)&s_frag[fx][fz][j * 2] - m2);
          *(half2*)&s_frag[fx][fz][j * 2 + 4] =
              math::ptx_exp2(*(half2*)&s_frag[fx][fz][j * 2 + 4] - m2);
        }
      }
    }
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z, typename DTypeIn,
          typename DTypeQKAccum>
__device__ __forceinline__ void compute_sfm_v(smem_t* v_smem, uint32_t* v_smem_offset_r,
                                              DTypeQKAccum (*s_frag)[num_frags_z][8],
                                              float (*o_frag)[num_frags_y][8], float (*d)[2]) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();

  DTypeIn s_frag_f16[num_frags_x][num_frags_z][8];
  if constexpr (std::is_same<DTypeQKAccum, float>::value) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
        vec_cast<DTypeIn, float, 8>(s_frag_f16[fx][fz], s_frag[fx][fz]);
      }
    }
  }

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      if constexpr (std::is_same<DTypeQKAccum, float>::value) {
        mma::rowsum_f16f16f32(d[fx], s_frag_f16[fx][fz]);
      } else {
        mma::rowsum_f16f16f32(d[fx], s_frag[fx][fz]);
      }
    }
  }

#pragma unroll
  for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      uint32_t b_frag[4];
      v_smem->ldmatrix_m8n8x4_trans(*v_smem_offset_r, b_frag);
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        if constexpr (std::is_same<DTypeQKAccum, float>::value) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn>(
              o_frag[fx][fy], (uint32_t*)(s_frag_f16[fx][fz]), b_frag);
        } else {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn>(o_frag[fx][fy],
                                                             (uint32_t*)s_frag[fx][fz], b_frag);
        }
      }
      *v_smem_offset_r = v_smem->advance_offset_by_column<2>(*v_smem_offset_r, fy);
    }
    *v_smem_offset_r =
        v_smem->advance_offset_by_row<16, channel_size_128b_in>(*v_smem_offset_r) - 2 * num_frags_y;
  }
  *v_smem_offset_r -= 16 * num_frags_z * channel_size_128b_in;
}

template <uint32_t num_frags_x, uint32_t num_frags_y>
__device__ __forceinline__ void normalize_d(float (*o_frag)[num_frags_y][8], float (*d)[2]) {
  float d_rcp[num_frags_x][2];
  // compute reciprocal of d
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      d_rcp[fx][j] = math::ptx_rcp(d[fx][j]);
    }
  }

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        o_frag[fx][fy][reg_id] = o_frag[fx][fy][reg_id] * d_rcp[fx][(reg_id % 4) / 2];
      }
    }
  }
}

/*!
 * \brief Synchronize the states of the MDO kernel across the threadblock along threadIdx.z.
 */
template <uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_x, uint32_t num_frags_y,
          typename DTypeQKAccum>
__device__ __forceinline__ void threadblock_sync_mdo_states(float (*o_frag)[num_frags_y][8],
                                                            float* smem_workspace,
                                                            DTypeQKAccum (*m)[2], float (*d)[2],
                                                            const uint32_t warp_idx,
                                                            const uint32_t lane_idx) {
  // only necessary when blockDim.z > 1
  if constexpr (num_warps_z > 1) {
    float2* smem_md = (float2*)smem_workspace;
    // o: [num_warps, warp_size, 8]
    // md: [num_warps, num_frags_x, 2, warp_size, 2 (m/d)]
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        smem_md[((warp_idx * num_frags_x + fx) * 2 + j) * warp_size + lane_idx] =
            make_float2(float(m[fx][j]), d[fx][j]);
      }
    }

    float o_scale[num_frags_x][2];
    // synchronize m,d first
    __syncthreads();
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        float m_new = -5e4, d_new = 1.f;
#pragma unroll
        for (uint32_t i = 0; i < num_warps_z; ++i) {
          float2 md = smem_md[(((i * num_warps_x + get_warp_idx_x<num_warps_x, num_warps_z>()) *
                                    num_frags_x +
                                fx) *
                                   2 +
                               j) *
                                  warp_size +
                              lane_idx];
          float m_prev = m_new, d_prev = d_new;
          m_new = max(m_new, md.x);
          d_new = d_prev * math::ptx_exp2(m_prev - m_new) + md.y * math::ptx_exp2(md.x - m_new);
        }
        o_scale[fx][j] = math::ptx_exp2(float(m[fx][j]) - m_new);
        m[fx][j] = DTypeQKAccum(m_new);
        d[fx][j] = d_new;
      }
    }

    __syncthreads();

#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        vec_t<float, 8> o_new;
        o_new.fill(0.f);
        vec_t<float, 8>::memcpy(smem_workspace + (warp_idx * warp_size + lane_idx) * 8,
                                o_frag[fx][fy]);
        __syncthreads();
#pragma unroll
        for (uint32_t i = 0; i < num_warps_z; ++i) {
          vec_t<float, 8> oi;
          oi.load(smem_workspace +
                  ((i * num_warps_x + get_warp_idx_x<num_warps_x, num_warps_z>()) * warp_size +
                   lane_idx) *
                      8);
#pragma unroll
          for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
            o_new[reg_id] += oi[reg_id] * o_scale[fx][(reg_id % 4) / 2];
          }
        }
        o_new.store(o_frag[fx][fy]);
      }
    }
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_y, typename DTypeQKAccum>
__device__ __forceinline__ void grid_sync_mdo_states(float (*o_frag)[num_frags_y][8], float* tmp,
                                                     DTypeQKAccum (*m)[2], float (*d)[2]) {
  const uint32_t bx = blockIdx.x;
  const uint32_t num_chunks = gridDim.y;
  const uint32_t kv_head_idx = blockIdx.z;
  // aggregate global state
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      vec_t<float, 8>::memcpy(
          tmp + ((fx * num_frags_y + fy) * grid.size() + grid.thread_rank()) * 8, o_frag[fx][fy]);
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        o_frag[fx][fy][reg_id] = 0.f;
      }
    }
  }
  float2* tmp_md = (float2*)(tmp + num_frags_x * num_frags_y * 8 * grid.size());
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      tmp_md[(((fx * 2 + j) * grid.size() + grid.thread_rank())) * 2] =
          make_float2(float(m[fx][j]), d[fx][j]);
      m[fx][j] = DTypeQKAccum(-5e4);
      d[fx][j] = 1.f;
    }
  }

  grid.sync();

  for (uint32_t iter = 0; iter < num_chunks; ++iter) {
    float other_scale[num_frags_x][2];
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        float2 md =
            tmp_md[((fx * 2 + j) * grid.size() +
                    ((kv_head_idx * num_chunks + iter) * gridDim.x + bx) * block.num_threads() +
                    block.thread_rank()) *
                   2];
        float mi = md.x, di = md.y, m_prev = float(m[fx][j]);
        float m_new = max(m_prev, mi);
        m[fx][j] = m_new;
        float o_scale = math::ptx_exp2(m_prev - m_new);
        other_scale[fx][j] = math::ptx_exp2(mi - m_new);
        d[fx][j] = d[fx][j] * o_scale + di * other_scale[fx][j];
#pragma unroll
        for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
          o_frag[fx][fy][j * 2 + 0] *= o_scale;
          o_frag[fx][fy][j * 2 + 1] *= o_scale;
          o_frag[fx][fy][j * 2 + 4] *= o_scale;
          o_frag[fx][fy][j * 2 + 5] *= o_scale;
        }
      }
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        vec_t<float, 8> o_frag_i;
        o_frag_i.load(tmp +
                      ((fx * num_frags_y + fy) * grid.size() +
                       ((kv_head_idx * num_chunks + iter) * gridDim.x + bx) * block.num_threads() +
                       block.thread_rank()) *
                          8);
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
          o_frag[fx][fy][reg_id] += o_frag_i[reg_id] * other_scale[fx][(reg_id % 4) / 2];
        }
      }
    }
  }
}

template <uint32_t group_size, uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_x,
          uint32_t num_frags_y, typename DTypeOut>
__device__ __forceinline__ void write_o_reg_gmem(float (*o_frag)[num_frags_y][8], smem_t* o_smem,
                                                 DTypeOut* o_ptr_base, uint32_t o_idx_base,
                                                 const uint32_t qo_upper_bound,
                                                 const uint32_t qo_n_stride,
                                                 const uint32_t qo_h_stride) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_out = head_dim / num_elems_per_128b<DTypeOut>();
  constexpr uint32_t num_warps = num_warps_x * num_warps_z;
  const uint32_t warp_idx = get_warp_idx<num_warps_x, num_warps_z>();
  const uint32_t lane_idx = threadIdx.x;

  if (get_warp_idx_z<num_warps_x, num_warps_z>() == 0) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        uint32_t o_frag_f16[4];
        vec_cast<DTypeOut, float, 8>((DTypeOut*)o_frag_f16, o_frag[fx][fy]);
        uint32_t o_smem_offset_w = smem_t::get_permuted_offset<channel_size_128b_out>(
            (get_warp_idx_x<num_warps_x, num_warps_z>() * num_frags_x + fx) * 16 + lane_idx / 4,
            fy * 2);
        ((uint32_t*)(o_smem->base + o_smem_offset_w))[lane_idx % 4] = o_frag_f16[0];
        ((uint32_t*)(o_smem->base + o_smem_offset_w + 8 * channel_size_128b_out))[lane_idx % 4] =
            o_frag_f16[1];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1)))[lane_idx % 4] = o_frag_f16[2];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1) +
                     8 * channel_size_128b_out))[lane_idx % 4] = o_frag_f16[3];
      }
    }
  }

  __syncthreads();

  uint32_t o_smem_offset_w =
      smem_t::get_permuted_offset<channel_size_128b_out>(warp_idx * 4 + lane_idx / 8, lane_idx % 8);

  uint32_t o_idx = o_idx_base + (warp_idx * 4 + lane_idx / 8) / group_size;
  DTypeOut* o_ptr = o_ptr_base + ((warp_idx * 4 + lane_idx / 8) / group_size) * qo_n_stride +
                    ((warp_idx * 4 + lane_idx / 8) % group_size) * qo_h_stride;
  static_assert(num_frags_x * 4 % num_warps_z == 0);
  // NOTE(Zihao): num_warps_x * num_frags_x * 4 / num_warps = num_frags_x * 4 / num_warps_z
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x * 4 / num_warps_z; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y / 4; ++fy) {
      if (o_idx < qo_upper_bound) {
        o_smem->store_128b(o_smem_offset_w, o_ptr);
      }
      o_ptr += 8 * num_elems_per_128b<DTypeOut>();
      o_smem_offset_w = o_smem->advance_offset_by_column<8>(o_smem_offset_w, fy);
    }
    o_idx += (num_warps * 4) / group_size;
    o_ptr += ((num_warps * 4) / group_size) * qo_n_stride -
             2 * num_frags_y * num_elems_per_128b<DTypeOut>();
    o_smem_offset_w =
        o_smem->advance_offset_by_row<num_warps * 4, channel_size_128b_out>(o_smem_offset_w) -
        2 * num_frags_y;
  }
}

}  // namespace

/*!
 * \brief FlashAttention prefill CUDA kernel for a single request.
 * \tparam partition_kv Whether to split kv_len into chunks.
 * \tparam group_size The number of qo heads that maps to a kv head (used in GQA).
 * \tparam causal Whether to use causal attention.
 * \tparam kv_layout The layout of the input tensor.
 * \tparam pos_encoding_mode The positional encoding mode.
 * \tparam num_frags_x The number of fragments in x dimension.
 * \tparam num_frags_y The number of fragments in y dimension.
 * \tparam num_frags_z The number of fragments in z dimension.
 * \tparam num_warps The number of warps in the threadblock.
 * \tparam DTypeIn The data type of the input tensor.
 * \tparam DTypeOut The data type of the output tensor.
 * \param q The query tensor.
 * \param k The key tensor.
 * \param v The value tensor.
 * \param o The output tensor.
 * \param tmp The temporary buffer (used when partition_kv is true).
 * \param lse The logsumexp value.
 * \param qkv_info The tensor info of the input tensor.
 * \param sm_scale The scale factor applied to the softmax score.
 * \param log2_rope_rcp_scale log2(1/(rope_scale)), where rope_scale is the scaling
 *   factor used in RoPE interpolation.
 * \param log2_rope_rcp_theta log2(1/(rope_theta)), where rope_theta is the theta
 *   used in RoPE.
 */
template <bool partition_kv, uint32_t group_size, bool causal, QKVLayout kv_layout,
          PosEncodingMode pos_encoding_mode, uint32_t num_warps_x, uint32_t num_warps_z,
          uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z, typename DTypeIn,
          typename DTypeQKAccum, typename DTypeOut>
__global__ void SinglePrefillWithKVCacheKernel(
    DTypeIn* __restrict__ q, DTypeIn* __restrict__ k, DTypeIn* __restrict__ v,
    DTypeOut* __restrict__ o, void* __restrict__ tmp, float* __restrict__ lse,
    const tensor_info_t<kv_layout, group_size, num_frags_y * 16> qkv_info, float sm_scale,
    const float log2_rope_rcp_scale, const float log2_rope_rcp_theta) {
  constexpr uint32_t num_warps = num_warps_x * num_warps_z;
  static_assert(sizeof(DTypeIn) == 2);
  static_assert(sizeof(DTypeOut) == 2);
  sm_scale *= math::log2e;
  const uint32_t qo_len = qkv_info.qo_len;
  const uint32_t kv_len = qkv_info.kv_len;
  const uint32_t lane_idx = threadIdx.x, warp_idx = get_warp_idx<num_warps_x, num_warps_z>();
  const uint32_t bx = blockIdx.x, chunk_idx = blockIdx.y, kv_head_idx = blockIdx.z;
  float alibi_slopes[num_frags_x][2];
  if constexpr (pos_encoding_mode == PosEncodingMode::kALiBi) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t qo_head_idx =
            kv_head_idx * group_size + (lane_idx / 4 + j * 8 + fx * 16) % group_size;
        const uint32_t num_qo_heads = qkv_info.get_num_qo_heads();
        alibi_slopes[fx][j] = get_alibi_slope(qo_head_idx, num_qo_heads) * math::log2e;
      }
    }
  }

  const uint32_t num_chunks = gridDim.y;
  const uint32_t chunk_size = partition_kv ? ceil_div(kv_len, num_chunks) : kv_len;
  const uint32_t chunk_start = partition_kv ? chunk_idx * chunk_size : 0;
  const uint32_t chunk_end = partition_kv ? min((chunk_idx + 1) * chunk_size, kv_len) : kv_len;
  auto block = cg::this_thread_block();

  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
  constexpr uint32_t channel_size_128b_out = head_dim / num_elems_per_128b<DTypeOut>();

  static_assert(num_frags_z * num_frags_y % num_warps == 0);
  static_assert(group_size == 1 || group_size % 4 == 0);

  extern __shared__ uint8_t smem[];

  DTypeQKAccum s_frag[num_frags_x][num_frags_z][8];
  float o_frag[num_frags_x][num_frags_y][8];
  DTypeQKAccum m[num_frags_x][2];
  float d[num_frags_x][2];
  float rope_freq[num_frags_y / 2][4];
  if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
    init_rope_freq<num_frags_y>(rope_freq, log2_rope_rcp_scale, log2_rope_rcp_theta);
  }
  init_states<num_frags_x, num_frags_y>(o_frag, m, d);

  // cooperative fetch q fragment from gmem to reg
  const uint32_t qo_idx_base = (bx * num_warps_x * num_frags_x * 16) / group_size;
  const uint32_t kv_n_stride = qkv_info.get_kv_n_stride(), qo_n_stride = qkv_info.get_qo_n_stride(),
                 qo_h_stride = qkv_info.get_qo_h_stride();
  smem_t qo_smem(smem);
  DTypeIn* q_ptr_base =
      q + qkv_info.get_qo_elem_offset(qo_idx_base, kv_head_idx * group_size,
                                      (lane_idx % 8) * num_elems_per_128b<DTypeIn>());
  DTypeOut* o_ptr_base =
      partition_kv
          ? ((DTypeOut*)tmp) + chunk_idx * qkv_info.get_num_qo_heads() * head_dim +
                qkv_info.get_qo_elem_offset(qo_idx_base * num_chunks, kv_head_idx * group_size,
                                            (lane_idx % 8) * num_elems_per_128b<DTypeOut>())
          : o + qkv_info.get_qo_elem_offset(qo_idx_base, kv_head_idx * group_size,
                                            (lane_idx % 8) * num_elems_per_128b<DTypeOut>());
  uint32_t q_smem_offset_r = smem_t::get_permuted_offset<channel_size_128b_in>(
      get_warp_idx_x<num_warps_x, num_warps_z>() * num_frags_x * 16 + lane_idx % 16, lane_idx / 16);

  load_q_global_smem<group_size, num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
      qo_idx_base, qo_len, q_ptr_base, qo_n_stride, qo_h_stride, &qo_smem);

  cp_async::commit_group();
  cp_async::wait_group<0>();
  block.sync();

  if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
    q_smem_inplace_apply_rotary_multiply_sm_scale<group_size, num_frags_x, num_frags_y, DTypeIn>(
        qo_idx_base, qo_len, kv_len, &qo_smem, &q_smem_offset_r, rope_freq, sm_scale);
  } else {
    q_smem_inplace_multiply_sm_scale<num_warps_x, num_warps_z, num_frags_x, num_frags_y, DTypeIn>(
        &qo_smem, sm_scale, warp_idx, lane_idx);
  }

  smem_t k_smem(smem + (num_warps_x * num_frags_x) * 16 * head_dim * sizeof(DTypeIn)),
      v_smem(smem + (num_warps_x * num_frags_x + num_warps_z * num_frags_z) * 16 * head_dim *
                        sizeof(DTypeIn));

  const uint32_t num_iterations = ceil_div(
      causal ? min(chunk_end - chunk_start,
                   sub_if_greater_or_zero(
                       kv_len - qo_len + ((bx + 1) * num_frags_x * num_warps_x * 16) / group_size,
                       chunk_start))
             : chunk_end - chunk_start,
      num_warps_z * 16 * num_frags_z);

  const uint32_t mask_iteration =
      (causal ? min(chunk_end - chunk_start,
                    sub_if_greater_or_zero(
                        kv_len + (bx * num_warps_x * num_frags_x * 16) / group_size - qo_len,
                        chunk_start))
              : (chunk_end - chunk_start)) /
      (num_warps_z * 16 * num_frags_z);

  DTypeIn* k_ptr =
      k + qkv_info.get_kv_elem_offset(chunk_start + warp_idx * 4 + lane_idx / 8, kv_head_idx,
                                      (lane_idx % 8) * num_elems_per_128b<DTypeIn>());
  DTypeIn* v_ptr =
      v + qkv_info.get_kv_elem_offset(chunk_start + warp_idx * 4 + lane_idx / 8, kv_head_idx,
                                      (lane_idx % 8) * num_elems_per_128b<DTypeIn>());
  uint32_t k_smem_offset_r = smem_t::get_permuted_offset<channel_size_128b_in>(
               get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16 + 8 * (lane_idx / 16) +
                   lane_idx % 8,
               (lane_idx % 16) / 8),
           v_smem_offset_r = smem_t::get_permuted_offset<channel_size_128b_in>(
               get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16 + lane_idx % 16,
               lane_idx / 16),
           kv_smem_offset_w = smem_t::get_permuted_offset<channel_size_128b_in>(
               warp_idx * 4 + lane_idx / 8, lane_idx % 8);
  produce_kv<SharedMemFillMode::kNoFill, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
      k_smem, &kv_smem_offset_w, &k_ptr, kv_n_stride, chunk_start, chunk_end, warp_idx, lane_idx);
  cp_async::commit_group();
  produce_kv<SharedMemFillMode::kFillZero, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
      v_smem, &kv_smem_offset_w, &v_ptr, kv_n_stride, chunk_start, chunk_end, warp_idx, lane_idx);
  cp_async::commit_group();

#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    cp_async::wait_group<1>();
    block.sync();

    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      k_smem_inplace_apply_rotary<num_frags_y, num_frags_z, DTypeIn>(
          chunk_start + iter * (num_warps_z * 16 * num_frags_z), &k_smem, &k_smem_offset_r,
          rope_freq, warp_idx, lane_idx);
      block.sync();
    }

    // compute attention score
    compute_qk<num_frags_x, num_frags_y, num_frags_z, DTypeIn>(&qo_smem, &q_smem_offset_r, &k_smem,
                                                               &k_smem_offset_r, s_frag);

    if constexpr (pos_encoding_mode == PosEncodingMode::kALiBi) {
      apply_alibi_bias<group_size, num_frags_x, num_frags_z>(
          qo_idx_base +
              (get_warp_idx_x<num_warps_x, num_warps_z>() * num_frags_x * 16) / group_size,
          chunk_start + iter * (num_warps_z * 16 * num_frags_z) +
              get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16,
          int(kv_len) - int(qo_len), alibi_slopes, s_frag);
    }
    // apply mask
    if (iter >= mask_iteration) {
      mask_s<partition_kv, causal, group_size, num_frags_x, num_frags_y, num_frags_z>(
          qo_idx_base +
              (get_warp_idx_x<num_warps_x, num_warps_z>() * num_frags_x * 16) / group_size,
          chunk_start + iter * (num_warps_z * 16 * num_frags_z) +
              get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16,
          qo_len, kv_len, chunk_end, s_frag);
    }

    // compute m,d states in online softmax
    update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(s_frag, o_frag, m, d);

    block.sync();
    produce_kv<SharedMemFillMode::kNoFill, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        k_smem, &kv_smem_offset_w, &k_ptr, kv_n_stride,
        chunk_start + (iter + 1) * (num_warps_z * 16 * num_frags_z), chunk_end, warp_idx, lane_idx);
    cp_async::commit_group();
    cp_async::wait_group<1>();
    block.sync();

    // compute sfm*v
    compute_sfm_v<num_frags_x, num_frags_y, num_frags_z, DTypeIn>(&v_smem, &v_smem_offset_r, s_frag,
                                                                  o_frag, d);

    block.sync();
    produce_kv<SharedMemFillMode::kFillZero, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        v_smem, &kv_smem_offset_w, &v_ptr, kv_n_stride,
        chunk_start + (iter + 1) * (num_warps_z * 16 * num_frags_z), chunk_end, warp_idx, lane_idx);
    cp_async::commit_group();
  }
  cp_async::wait_group<0>();
  block.sync();

  // threadblock synchronization
  threadblock_sync_mdo_states<num_warps_x, num_warps_z, num_frags_x, num_frags_y, DTypeQKAccum>(
      o_frag, (float*)smem, m, d, warp_idx, lane_idx);

  // normalize d
  normalize_d<num_frags_x, num_frags_y>(o_frag, d);

  // write back
  write_o_reg_gmem<group_size, num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
      o_frag, &qo_smem, o_ptr_base, qo_idx_base, qo_len,
      partition_kv ? qo_n_stride * num_chunks : qo_n_stride, qo_h_stride);

  // write lse
  if (lse != nullptr || partition_kv) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t qo_head_idx =
            kv_head_idx * group_size + (lane_idx / 4 + j * 8 + fx * 16) % group_size;
        const uint32_t num_qo_heads = qkv_info.get_num_qo_heads();
        const uint32_t qo_idx =
            qo_idx_base + (get_warp_idx_x<num_warps_x, num_warps_z>() * num_frags_x * 16 +
                           lane_idx / 4 + j * 8 + fx * 16) /
                              group_size;
        if (qo_idx < qo_len) {
          if constexpr (partition_kv) {
            float* tmp_lse =
                (float*)(((DTypeOut*)tmp) + qo_len * num_chunks * num_qo_heads * head_dim);
            tmp_lse[(qo_idx * num_chunks + chunk_idx) * num_qo_heads + qo_head_idx] =
                math::ptx_log2(d[fx][j]) + float(m[fx][j]);
          } else {
            lse[qo_idx * num_qo_heads + qo_head_idx] = math::ptx_log2(d[fx][j]) + float(m[fx][j]);
          }
        }
      }
    }
  }
}

template <uint32_t group_size, bool causal, QKVLayout kv_layout, PosEncodingMode pos_encoding_mode,
          uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_x, uint32_t num_frags_y,
          uint32_t num_frags_z, typename DTypeIn, typename DTypeQKAccum, typename DTypeOut,
          typename IdType>
__global__ void BatchPrefillWithRaggedKVCacheKernel(
    DTypeIn* __restrict__ q, IdType* __restrict__ request_indices,
    IdType* __restrict__ tile_indices, IdType* __restrict__ qo_indptr, DTypeIn* __restrict__ k,
    DTypeIn* __restrict__ v, IdType* __restrict__ kv_indptr, IdType* __restrict__ q_offset,
    IdType* __restrict__ k_rope_pos_offset, DTypeOut* __restrict__ o, float* __restrict__ tmp,
    float* __restrict__ lse, uint32_t batch_size, float sm_scale, float log2_rope_rcp_scale,
    float log2_rope_rcp_theta) {
  static_assert(sizeof(DTypeIn) == 2);
  static_assert(sizeof(DTypeOut) == 2);
  sm_scale *= math::log2e;
  constexpr uint32_t head_dim = num_frags_y * 16;

  auto block = cg::this_thread_block();
  constexpr uint32_t num_warps = num_warps_x * num_warps_z;
  const uint32_t bx = blockIdx.x, lane_idx = threadIdx.x,
                 warp_idx = get_warp_idx<num_warps_x, num_warps_z>(), kv_head_idx = blockIdx.z;
  const uint32_t num_kv_heads = gridDim.z;
  const uint32_t request_idx = request_indices[bx], tile_idx = tile_indices[bx];
  constexpr uint32_t num_rows_per_cta = num_frags_x * num_warps_x * 16;
  const uint32_t qo_len = qo_indptr[request_idx + 1] - qo_indptr[request_idx],
                 kv_len = kv_indptr[request_idx + 1] - kv_indptr[request_idx];
  const tensor_info_t<kv_layout, group_size, num_frags_y * 16> qkv_info(qo_len, kv_len,
                                                                        num_kv_heads);
  float alibi_slopes[num_frags_x][2];
  if constexpr (pos_encoding_mode == PosEncodingMode::kALiBi) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t qo_head_idx =
            kv_head_idx * group_size + (lane_idx / 4 + j * 8 + fx * 16) % group_size;
        const uint32_t num_qo_heads = qkv_info.get_num_qo_heads();
        alibi_slopes[fx][j] = get_alibi_slope(qo_head_idx, num_qo_heads) * math::log2e;
      }
    }
  }
  const uint32_t qo_upper_bound = min(qo_len, (tile_idx + 1) * (num_rows_per_cta / group_size));

  constexpr bool partition_kv = false;
  constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
  constexpr uint32_t channel_size_128b_out = head_dim / num_elems_per_128b<DTypeOut>();

  static_assert(num_frags_z * num_frags_y % num_warps == 0);
  static_assert(group_size == 1 || group_size % 4 == 0);

  extern __shared__ uint8_t smem[];

  DTypeQKAccum s_frag[num_frags_x][num_frags_z][8];
  float o_frag[num_frags_x][num_frags_y][8];
  DTypeQKAccum m[num_frags_x][2];
  float d[num_frags_x][2];
  float rope_freq[num_frags_y / 2][4];

  if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
    init_rope_freq<num_frags_y>(rope_freq, log2_rope_rcp_scale, log2_rope_rcp_theta);
  }
  init_states<num_frags_x, num_frags_y>(o_frag, m, d);

  const uint32_t qo_idx_base = (tile_idx * num_warps_x * num_frags_x * 16) / group_size;
  const uint32_t kv_n_stride = qkv_info.get_kv_n_stride(), qo_n_stride = qkv_info.get_qo_n_stride(),
                 qo_h_stride = qkv_info.get_qo_h_stride();
  smem_t qo_smem(smem);

  DTypeIn* q_ptr_base = q + qkv_info.get_qo_elem_offset(
                                qo_indptr[request_idx] + qo_idx_base, kv_head_idx * group_size,
                                (lane_idx % 8) * num_elems_per_128b<DTypeIn>());
  DTypeIn* o_ptr_base = o + qkv_info.get_qo_elem_offset(
                                qo_indptr[request_idx] + qo_idx_base, kv_head_idx * group_size,
                                (lane_idx % 8) * num_elems_per_128b<DTypeOut>());

  uint32_t q_smem_offset_r = smem_t::get_permuted_offset<channel_size_128b_in>(
      get_warp_idx_x<num_warps_x, num_warps_z>() * num_frags_x * 16 + lane_idx % 16, lane_idx / 16);

  load_q_global_smem<group_size, num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
      qo_idx_base, qo_upper_bound, q_ptr_base, qo_n_stride, qo_h_stride, &qo_smem);

  cp_async::commit_group();
  cp_async::wait_group<0>();
  block.sync();

  if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
    if (!q_offset) {
      q_smem_inplace_apply_rotary_multiply_sm_scale<group_size, num_frags_x, num_frags_y, DTypeIn>(
          qo_idx_base, qo_len, kv_len, &qo_smem, &q_smem_offset_r, rope_freq, sm_scale);
    } else {
      q_smem_inplace_apply_rotary_with_pos_multiply_sm_scale<group_size, num_frags_x, num_frags_y,
                                                             DTypeIn>(
          qo_indptr[request_idx] + qo_idx_base, q_offset, &qo_smem, &q_smem_offset_r, rope_freq,
          sm_scale);
    }
  } else {
    q_smem_inplace_multiply_sm_scale<num_warps_x, num_warps_z, num_frags_x, num_frags_y, DTypeIn>(
        &qo_smem, sm_scale, warp_idx, lane_idx);
  }

  const uint32_t num_iterations = ceil_div(
      (causal ? min(kv_len, kv_len - qo_len +
                                ((tile_idx + 1) * num_frags_x * num_warps_x * 16) / group_size)
              : kv_len),
      16 * num_warps_z * num_frags_z);

  const uint32_t mask_iteration =
      (causal
           ? min(kv_len + (tile_idx * num_warps_x * num_frags_x * 16) / group_size - qo_len, kv_len)
           : kv_len) /
      (16 * num_warps_z * num_frags_z);

  smem_t k_smem(smem + (num_warps_x * num_frags_x) * 16 * head_dim * sizeof(DTypeIn)),
      v_smem(smem + (num_warps_x * num_frags_x + num_warps_z * num_frags_z) * 16 * head_dim *
                        sizeof(DTypeIn));

  uint32_t k_smem_offset_r = smem_t::get_permuted_offset<channel_size_128b_in>(
               get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16 + 8 * (lane_idx / 16) +
                   lane_idx % 8,
               (lane_idx % 16) / 8),
           v_smem_offset_r = smem_t::get_permuted_offset<channel_size_128b_in>(
               get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16 + lane_idx % 16,
               lane_idx / 16),
           kv_smem_offset_w = smem_t::get_permuted_offset<channel_size_128b_in>(
               warp_idx * 4 + lane_idx / 8, lane_idx % 8);

  DTypeIn* k_ptr =
      k + qkv_info.get_kv_elem_offset(kv_indptr[request_idx] + warp_idx * 4 + lane_idx / 8,
                                      kv_head_idx, (lane_idx % 8) * num_elems_per_128b<DTypeIn>());
  DTypeIn* v_ptr =
      v + qkv_info.get_kv_elem_offset(kv_indptr[request_idx] + warp_idx * 4 + lane_idx / 8,
                                      kv_head_idx, (lane_idx % 8) * num_elems_per_128b<DTypeIn>());

  produce_kv<SharedMemFillMode::kNoFill, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
      k_smem, &kv_smem_offset_w, &k_ptr, kv_n_stride, 0, kv_len, warp_idx, lane_idx);
  cp_async::commit_group();
  produce_kv<SharedMemFillMode::kFillZero, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
      v_smem, &kv_smem_offset_w, &v_ptr, kv_n_stride, 0, kv_len, warp_idx, lane_idx);
  cp_async::commit_group();

#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    cp_async::wait_group<1>();
    block.sync();

    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      k_smem_inplace_apply_rotary<num_frags_y, num_frags_z, DTypeIn>(
          (k_rope_pos_offset == nullptr ? 0 : k_rope_pos_offset[request_idx]) +
              iter * 16 * num_warps_z * num_frags_z,
          &k_smem, &k_smem_offset_r, rope_freq, warp_idx, lane_idx);
      block.sync();
    }

    // compute attention score
    compute_qk<num_frags_x, num_frags_y, num_frags_z, DTypeIn>(&qo_smem, &q_smem_offset_r, &k_smem,
                                                               &k_smem_offset_r, s_frag);

    if constexpr (pos_encoding_mode == PosEncodingMode::kALiBi) {
      // TODO(Zihao): handle the case that q_offset is specified
      apply_alibi_bias<group_size, num_frags_x, num_frags_z>(
          qo_idx_base + get_warp_idx_x<num_warps_x, num_warps_z>() * num_frags_x * 16,
          iter * 16 * num_warps_z * num_frags_z +
              get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16,
          int(kv_len) - int(qo_len), alibi_slopes, s_frag);
    }
    // apply mask
    if (iter >= mask_iteration) {
      mask_s<partition_kv, causal, group_size, num_frags_x, num_frags_y, num_frags_z>(
          qo_idx_base + get_warp_idx_x<num_warps_x, num_warps_z>() * num_frags_x * 16,
          iter * 16 * num_warps_z * num_frags_z +
              get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16,
          qo_len, kv_len, kv_len, s_frag);
    }

    // compute m,d states in online softmax
    update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(s_frag, o_frag, m, d);

    block.sync();
    produce_kv<SharedMemFillMode::kNoFill, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        k_smem, &kv_smem_offset_w, &k_ptr, kv_n_stride, (iter + 1) * 16 * num_warps_z * num_frags_z,
        kv_len, warp_idx, lane_idx);
    cp_async::commit_group();
    cp_async::wait_group<1>();
    block.sync();

    // compute sfm*v
    compute_sfm_v<num_frags_x, num_frags_y, num_frags_z, DTypeIn>(&v_smem, &v_smem_offset_r, s_frag,
                                                                  o_frag, d);

    block.sync();
    produce_kv<SharedMemFillMode::kFillZero, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        v_smem, &kv_smem_offset_w, &v_ptr, kv_n_stride, (iter + 1) * 16 * num_warps_z * num_frags_z,
        kv_len, warp_idx, lane_idx);
    cp_async::commit_group();
  }
  cp_async::wait_group<0>();
  block.sync();

  // threadblock synchronization
  threadblock_sync_mdo_states<num_warps_x, num_warps_z, num_frags_x, num_frags_y, DTypeQKAccum>(
      o_frag, (float*)smem, m, d, warp_idx, lane_idx);

  // normalize d
  normalize_d<num_frags_x, num_frags_y>(o_frag, d);

  // write back
  write_o_reg_gmem<group_size, num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
      o_frag, &qo_smem, o_ptr_base, qo_idx_base, qo_len, qo_n_stride, qo_h_stride);

  // write lse
  if (lse != nullptr) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t qo_head_idx =
            kv_head_idx * group_size + (lane_idx / 4 + j * 8 + fx * 16) % group_size;
        const uint32_t num_qo_heads = qkv_info.get_num_qo_heads();
        const uint32_t qo_idx =
            qo_idx_base + (get_warp_idx_x<num_warps_x, num_warps_z>() * num_warps_x * 16 +
                           lane_idx / 4 + j * 8 + fx * 16) /
                              group_size;
        if (qo_idx < qo_len) {
          lse[(qo_indptr[request_idx] + qo_idx) * num_qo_heads + qo_head_idx] =
              math::ptx_log2(d[fx][j]) + float(m[fx][j]);
        }
      }
    }
  }
}

template <uint32_t group_size, uint32_t page_size, bool causal, PosEncodingMode pos_encoding_mode,
          uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_x, uint32_t num_frags_y,
          uint32_t num_frags_z, PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn,
          typename DTypeQKAccum, typename DTypeOut, typename IdType>
__global__ void BatchPrefillWithPagedKVCacheKernel(
    IdType* __restrict__ request_indices, IdType* __restrict__ tile_indices,
    DTypeIn* __restrict__ q, paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
    IdType* __restrict__ qo_indptr, IdType* __restrict__ q_offset, DTypeOut* __restrict__ o,
    float* __restrict__ tmp, float* __restrict__ lse, float sm_scale, float log2_rope_rcp_scale,
    float log2_rope_rcp_theta) {
  static_assert(sizeof(DTypeIn) == 2);
  static_assert(sizeof(DTypeOut) == 2);
  sm_scale *= math::log2e;
  auto block = cg::this_thread_block();

  constexpr uint32_t num_warps = num_warps_x * num_warps_z;
  const uint32_t bx = blockIdx.x, lane_idx = threadIdx.x,
                 warp_idx = get_warp_idx<num_warps_x, num_warps_z>(), kv_head_idx = blockIdx.z;
  const uint32_t num_kv_heads = gridDim.z, num_qo_heads = num_kv_heads * group_size;
  float alibi_slopes[num_frags_x][2];
  if constexpr (pos_encoding_mode == PosEncodingMode::kALiBi) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t qo_head_idx =
            kv_head_idx * group_size + (lane_idx / 4 + j * 8 + fx * 16) % group_size;
        alibi_slopes[fx][j] = get_alibi_slope(qo_head_idx, num_qo_heads) * math::log2e;
      }
    }
  }
  const uint32_t request_idx = request_indices[bx], tile_idx = tile_indices[bx];
  constexpr uint32_t num_rows_per_cta = num_frags_x * num_warps_x * 16;
  const uint32_t qo_len = qo_indptr[request_idx + 1] - qo_indptr[request_idx],
                 kv_len = (paged_kv.indptr[request_idx + 1] - paged_kv.indptr[request_idx] - 1) *
                              paged_kv.page_size +
                          paged_kv.last_page_len[request_idx];
  const uint32_t qo_upper_bound = min(qo_len, (tile_idx + 1) * (num_rows_per_cta / group_size));

  constexpr bool partition_kv = false;
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
  constexpr uint32_t channel_size_128b_out = head_dim / num_elems_per_128b<DTypeOut>();

  static_assert(num_frags_z * num_frags_y % num_warps == 0);
  static_assert(group_size == 1 || group_size % 4 == 0);

  extern __shared__ uint8_t smem[];

  DTypeQKAccum s_frag[num_frags_x][num_frags_z][8];
  float o_frag[num_frags_x][num_frags_y][8];
  DTypeQKAccum m[num_frags_x][2];
  float d[num_frags_x][2];
  float rope_freq[num_frags_y / 2][4];

  if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
    init_rope_freq<num_frags_y>(rope_freq, log2_rope_rcp_scale, log2_rope_rcp_theta);
  }
  init_states<num_frags_x, num_frags_y>(o_frag, m, d);

  const uint32_t qo_idx_base = (tile_idx * num_warps * num_frags_x * 16) / group_size;
  const uint32_t qo_n_stride = get_n_stride_impl<QKVLayout::kNHD, head_dim>(num_qo_heads),
                 qo_h_stride = get_h_stride_impl<QKVLayout::kNHD, head_dim>(qo_len);
  smem_t qo_smem(smem);
  DTypeIn* q_ptr_base =
      q + get_elem_offset_impl<QKVLayout::kNHD, head_dim>(
              qo_indptr[request_idx] + qo_idx_base, kv_head_idx * group_size,
              (lane_idx % 8) * num_elems_per_128b<DTypeIn>(), qo_len, num_qo_heads);
  DTypeIn* o_ptr_base =
      o + get_elem_offset_impl<QKVLayout::kNHD, head_dim>(
              qo_indptr[request_idx] + qo_idx_base, kv_head_idx * group_size,
              (lane_idx % 8) * num_elems_per_128b<DTypeOut>(), qo_len, num_qo_heads);
  uint32_t q_smem_offset_r = smem_t::get_permuted_offset<channel_size_128b_in>(
      get_warp_idx<num_warps_x, num_warps_z>() * num_frags_x * 16 + lane_idx % 16, lane_idx / 16);

  load_q_global_smem<group_size, num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
      qo_idx_base, qo_upper_bound, q_ptr_base, qo_n_stride, qo_h_stride, &qo_smem);

  cp_async::commit_group();
  cp_async::wait_group<0>();
  block.sync();

  if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
    if (q_offset == nullptr) {
      q_smem_inplace_apply_rotary_multiply_sm_scale<group_size, num_frags_x, num_frags_y, DTypeIn>(
          qo_idx_base, qo_len, kv_len, &qo_smem, &q_smem_offset_r, rope_freq, sm_scale);
    } else {
      q_smem_inplace_apply_rotary_with_pos_multiply_sm_scale<group_size, num_frags_x, num_frags_y,
                                                             DTypeIn>(
          qo_indptr[request_idx] + qo_idx_base, q_offset, &qo_smem, &q_smem_offset_r, rope_freq,
          sm_scale);
    }
  } else {
    q_smem_inplace_multiply_sm_scale<num_warps_x, num_warps_z, num_frags_x, num_frags_y, DTypeIn>(
        &qo_smem, sm_scale, warp_idx, lane_idx);
  }

  smem_t k_smem(smem + (num_warps_x * num_frags_x) * 16 * head_dim * sizeof(DTypeIn)),
      v_smem(smem + (num_warps_x * num_frags_x + num_warps_z * num_frags_z) * 16 * head_dim *
                        sizeof(DTypeIn));

  uint32_t k_smem_offset_r = smem_t::get_permuted_offset<channel_size_128b_in>(
               get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16 + 8 * (lane_idx / 16) +
                   lane_idx % 8,
               (lane_idx % 16) / 8),
           v_smem_offset_r = smem_t::get_permuted_offset<channel_size_128b_in>(
               get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16 + lane_idx % 16,
               lane_idx / 16),
           kv_smem_offset_w = smem_t::get_permuted_offset<channel_size_128b_in>(
               warp_idx * 4 + lane_idx / 8, lane_idx % 8);
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  uint32_t page_iter_base = paged_kv.indptr[request_idx];
  page_produce_kv<false, page_size, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
      k_smem, &kv_smem_offset_w, paged_kv, 0, page_iter_base, kv_len, last_indptr, warp_idx,
      lane_idx);
  cp_async::commit_group();
  page_produce_kv<true, page_size, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
      v_smem, &kv_smem_offset_w, paged_kv, 0, page_iter_base, kv_len, last_indptr, warp_idx,
      lane_idx);
  cp_async::commit_group();

  const uint32_t num_iterations = ceil_div(
      (causal ? min(kv_len, kv_len - qo_len +
                                ((tile_idx + 1) * num_frags_x * num_warps_x * 16) / group_size)
              : kv_len),
      16 * num_warps_z * num_frags_z);

  const uint32_t mask_iteration =
      (causal
           ? min(kv_len + (tile_idx * num_warps_x * num_frags_x * 16) / group_size - qo_len, kv_len)
           : kv_len) /
      (16 * num_warps_z * num_frags_z);

#pragma unroll
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    cp_async::wait_group<1>();
    block.sync();

    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      k_smem_inplace_apply_rotary<num_frags_y, num_frags_z, DTypeIn>(
          (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[request_idx]) +
              iter * 16 * num_warps_z * num_frags_z,
          &k_smem, &k_smem_offset_r, rope_freq, warp_idx, lane_idx);
      block.sync();
    }

    // compute attention score
    compute_qk<num_frags_x, num_frags_y, num_frags_z, DTypeIn>(&qo_smem, &q_smem_offset_r, &k_smem,
                                                               &k_smem_offset_r, s_frag);

    if constexpr (pos_encoding_mode == PosEncodingMode::kALiBi) {
      // TODO(Zihao): handle the case that q_offset is specified
      apply_alibi_bias<group_size, num_frags_x, num_frags_z>(
          qo_idx_base + get_warp_idx_x<num_warps_x, num_warps_z>() * num_frags_x * 16,
          iter * 16 * num_warps_z * num_frags_z +
              get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16,
          int(kv_len) - int(qo_len), alibi_slopes, s_frag);
    }
    // apply mask
    if (iter >= mask_iteration) {
      mask_s<partition_kv, causal, group_size, num_frags_x, num_frags_y, num_frags_z>(
          qo_idx_base + get_warp_idx_x<num_warps_x, num_warps_z>() * num_frags_x * 16,
          iter * 16 * num_warps_z * num_frags_z +
              get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16,
          qo_len, kv_len, kv_len, s_frag);
    }

    // compute m,d states in online softmax
    update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(s_frag, o_frag, m, d);

    block.sync();
    page_iter_base += 16 * num_warps_z * num_frags_z / page_size;
    page_produce_kv<false, page_size, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        k_smem, &kv_smem_offset_w, paged_kv, (iter + 1) * 16 * num_warps_z * num_frags_z,
        page_iter_base, kv_len, last_indptr, warp_idx, lane_idx);
    cp_async::commit_group();
    cp_async::wait_group<1>();
    block.sync();

    // compute sfm*v
    compute_sfm_v<num_frags_x, num_frags_y, num_frags_z, DTypeIn>(&v_smem, &v_smem_offset_r, s_frag,
                                                                  o_frag, d);

    block.sync();
    page_produce_kv<true, page_size, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        v_smem, &kv_smem_offset_w, paged_kv, (iter + 1) * 16 * num_warps_z * num_frags_z,
        page_iter_base, kv_len, last_indptr, warp_idx, lane_idx);
    cp_async::commit_group();
  }
  cp_async::wait_group<0>();
  block.sync();

  // threadblock synchronization
  threadblock_sync_mdo_states<num_warps_x, num_warps_z, num_frags_x, num_frags_y, DTypeQKAccum>(
      o_frag, (float*)smem, m, d, warp_idx, lane_idx);

  // normalize d
  normalize_d<num_frags_x, num_frags_y>(o_frag, d);

  // write_back
  write_o_reg_gmem<group_size, num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
      o_frag, &qo_smem, o_ptr_base, qo_idx_base, qo_len, qo_n_stride, qo_h_stride);

  // write lse
  if (lse != nullptr) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t qo_head_idx =
            kv_head_idx * group_size + (lane_idx / 4 + j * 8 + fx * 16) % group_size;
        const uint32_t qo_idx =
            qo_idx_base + (get_warp_idx_x<num_warps_x, num_warps_z>() * num_frags_x * 16 +
                           lane_idx / 4 + j * 8 + fx * 16) /
                              group_size;
        if (qo_idx < qo_upper_bound) {
          lse[(qo_indptr[request_idx] + qo_idx) * num_qo_heads + qo_head_idx] =
              math::ptx_log2(d[fx][j]) + float(m[fx][j]);
        }
      }
    }
  }
}

/*!
 * \brief Estimate the temporary storage size and the maximum grid size for the
 *   cooperative SinglePrefillWithKVCacheKernel
 * \tparam DTypeIn The data type of input
 * \tparam DTypeOut The data type of output
 * \param tmp_size The estimated temporary storage size, return 0 if not use cooperative kernel.
 * \param max_grid_size The maximum grid size that can be used in a cooperative kernel.
 * \param num_qo_heads The number of query and output heads.
 * \param num_kv_heads The number of key and value heads.
 * \param qo_len The length of query and output.
 * \param kv_len The length of key and value.
 * \param head_dim The dimension of each head.
 * \param causal Whether to use causal attention.
 * \param kv_layout The layout of KV Cache.
 * \param pos_encoding_mode The positional encoding mode.
 * \param allow_fp16_qk_reduction Whether to allow accumulating q*k^T with fp16.
 * \param stream The cuda stream to execute the kernel on.
 * \return status Indicates whether CUDA calls are successful
 */
template <typename DTypeIn, typename DTypeOut>
cudaError_t SinglePrefillWithKVCacheWorkEstimation(
    uint32_t& tmp_size, uint32_t& max_grid_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t qo_len, uint32_t kv_len, uint32_t head_dim, bool causal = true,
    QKVLayout kv_layout = QKVLayout::kNHD,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
    bool allow_fp16_qk_reduction = false, cudaStream_t stream = nullptr) {
  if (kv_len < qo_len && causal) {
    std::ostringstream err_msg;
    err_msg << "When causal is true, kv_len must be greater than or equal to qo_len, "
            << "got kv_len " << kv_len << " and qo_len " << qo_len;
    throw std::invalid_argument(err_msg.str());
  }
  const uint32_t group_size = num_qo_heads / num_kv_heads;

  DISPATCH_ALLOW_FP16_QK_REDUCTION(
      allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION,
      {DISPATCH_NUM_FRAGS_X(
          (qo_len * group_size > 64 && head_dim < 256 ? 2 : 1), num_frags_x,
          {DISPATCH_GQA_GROUP_SIZE(
              group_size, GROUP_SIZE,
              {DISPATCH_CAUSAL(
                  causal, CAUSAL, {DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
                    constexpr uint32_t num_frags_y = HEAD_DIM / 16;
                    DISPATCH_POS_ENCODING_MODE(
                        pos_encoding_mode, pos_encoding_mode,
                        {DISPATCH_LAYOUT(kv_layout, KV_LAYOUT, {
                          using DTypeQKAccum =
                              typename std::conditional<ALLOW_FP16_QK_REDUCTION &&
                                                            std::is_same<DTypeIn, half>::value,
                                                        half, float>::type;

                          int dev_id = 0;
                          FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
                          int max_smem_per_sm = 0;
                          FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
                              &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                              dev_id));
                          // we expect each sm execute two threadblocks
                          const int max_smem_per_threadblock = max_smem_per_sm / 2;

                          constexpr uint32_t num_warps_x = 4;
                          constexpr uint32_t num_warps_z = 1;
                          const uint32_t max_num_frags_z_reg =
                              (HEAD_DIM == 128 && num_frags_x == 2 &&
                               pos_encoding_mode == PosEncodingMode::kRoPELlama &&
                               !allow_fp16_qk_reduction)
                                  ? 2
                                  : 4;
                          const uint32_t max_num_frags_z_smem =
                              (max_smem_per_threadblock / (16 * head_dim * sizeof(DTypeIn)) -
                               num_frags_x * num_warps_x) /
                              (2 * num_warps_z);

                          // control num_frags_z for maximum warp occupancy
                          DISPATCH_NUM_FRAGS_Z(
                              min(max_num_frags_z_smem, max_num_frags_z_reg), num_frags_z, {
                                if constexpr (is_invalid_configuration<DTypeQKAccum>(
                                                  num_frags_x, num_frags_y, num_frags_z,
                                                  num_warps_x, num_frags_z)) {
                                  // Invalid configuration, skip
                                  std::ostringstream err_msg;
                                  err_msg << "FlashInfer Internal Error: Invalid configuration : "
                                             "num_frags_x="
                                          << num_frags_x << " num_frags_y=" << num_frags_y
                                          << " num_frags_z=" << num_frags_z
                                          << " num_warps_x=" << num_warps_x
                                          << " num_warps_z=" << num_warps_z
                                          << " please create an issue "
                                             "(https://github.com/flashinfer-ai/flashinfer/issues)"
                                             " and report the issue to the developers.";
                                  throw std::invalid_argument(err_msg.str());
                                } else {
                                  constexpr uint32_t num_threads =
                                      (num_warps_x * num_warps_z) * warp_size;
                                  constexpr uint32_t num_rows_per_cta =
                                      num_frags_x * num_warps_x * 16;

                                  auto partition_kv_kernel = SinglePrefillWithKVCacheKernel<
                                      /*partition_kv=*/true, GROUP_SIZE, CAUSAL, KV_LAYOUT,
                                      pos_encoding_mode, num_warps_x, num_warps_z, num_frags_x,
                                      num_frags_y, num_frags_z, DTypeIn, DTypeQKAccum, DTypeOut>;
                                  tensor_info_t<KV_LAYOUT, GROUP_SIZE, HEAD_DIM> qkv_info(
                                      qo_len, kv_len, num_kv_heads);
                                  uint32_t smem_size =
                                      (num_frags_x * num_warps_x + num_frags_z * num_warps_z * 2) *
                                      16 * head_dim * sizeof(DTypeIn);
                                  FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                                      partition_kv_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                                  int num_blocks_per_sm = 0;
                                  int num_sm = 0;
                                  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
                                      &num_sm, cudaDevAttrMultiProcessorCount, dev_id));
                                  FLASHINFER_CUDA_CALL(
                                      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                          &num_blocks_per_sm, partition_kv_kernel, num_threads,
                                          smem_size));
                                  uint32_t max_num_kv_chunks =
                                      (num_blocks_per_sm * num_sm) /
                                      (num_kv_heads *
                                       ceil_div(qo_len * group_size, num_rows_per_cta));
                                  uint32_t num_chunks;
                                  if (max_num_kv_chunks > 0) {
                                    uint32_t chunk_size =
                                        max(ceil_div(kv_len, max_num_kv_chunks), 256);
                                    num_chunks = ceil_div(kv_len, chunk_size);
                                  } else {
                                    num_chunks = 0;
                                  }

                                  max_grid_size = num_blocks_per_sm * num_sm;
                                  if (num_chunks > 1) {
                                    uint32_t grid_size =
                                        32 * (num_warps_x * num_frags_z) *
                                        ceil_div(qo_len * group_size, num_rows_per_cta) *
                                        num_chunks * num_qo_heads;

                                    tmp_size = sizeof(DTypeOut) *
                                                   (num_chunks * num_qo_heads * qo_len * head_dim) +
                                               sizeof(float) * (num_chunks * num_qo_heads * qo_len);
                                  } else {
                                    tmp_size = 0;
                                  }
                                }
                              })
                        })})
                  })})})})});
  return cudaSuccess;
}

template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, QKVLayout KV_LAYOUT,
          PosEncodingMode pos_encoding_mode, bool ALLOW_FP16_QK_REDUCTION, bool CAUSAL,
          typename DTypeIn, typename DTypeOut>
cudaError_t SinglePrefillWithKVCacheDispatched(DTypeIn* q, DTypeIn* k, DTypeIn* v, DTypeOut* o,
                                               float* tmp, float* lse, uint32_t num_kv_heads,
                                               uint32_t qo_len, uint32_t kv_len, float sm_scale,
                                               float rope_scale, float rope_theta,
                                               cudaStream_t stream) {
  const float log2_rope_rcp_scale = -std::log2f(rope_scale);
  const float log2_rope_rcp_theta = -std::log2f(rope_theta);
  if (kv_len < qo_len && CAUSAL) {
    std::ostringstream err_msg;
    err_msg << "When causal is true, kv_len must be greater than or equal to qo_len, got kv_len"
            << kv_len << " and qo_len " << qo_len;
    throw std::invalid_argument(err_msg.str());
  }

  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
  DISPATCH_NUM_FRAGS_X((qo_len * GROUP_SIZE > 64 && HEAD_DIM < 256 ? 2 : 1), num_frags_x, {
    using DTypeQKAccum =
        typename std::conditional<ALLOW_FP16_QK_REDUCTION && std::is_same<DTypeIn, half>::value,
                                  half, float>::type;

    int dev_id = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    int max_smem_per_sm = 0;
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
    // we expect each sm execute two threadblocks
    const int max_smem_per_threadblock = max_smem_per_sm / 2;

    constexpr uint32_t num_warps_x = 2;
    constexpr uint32_t num_warps_z = 2;
    const uint32_t max_num_frags_z_reg =
        (HEAD_DIM == 128 && num_frags_x == 2 && pos_encoding_mode == PosEncodingMode::kRoPELlama &&
         !ALLOW_FP16_QK_REDUCTION)
            ? 2
            : 4;
    const uint32_t max_num_frags_z_smem =
        (max_smem_per_threadblock / (16 * HEAD_DIM * sizeof(DTypeIn)) - num_frags_x * num_warps_x) /
        (2 * num_warps_z);

    // control num_frags_z for maximum warp occupancy
    DISPATCH_NUM_FRAGS_Z(min(max_num_frags_z_smem, max_num_frags_z_reg), num_frags_z, {
      if constexpr (is_invalid_configuration<DTypeQKAccum>(num_frags_x, num_frags_y, num_frags_z,
                                                           num_warps_x, num_warps_z)) {
        // Invalid configuration, skip
        std::ostringstream err_msg;
        err_msg << "FlashInfer Internal Error: Invalid configuration : num_frags_x=" << num_frags_x
                << " num_frags_y=" << num_frags_y << " num_frags_z=" << num_frags_z
                << " num_warps_x=" << num_warps_x << " num_warps_z=" << num_warps_z
                << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                   " and report the issue to the developers.";
        throw std::invalid_argument(err_msg.str());
      } else {
        constexpr uint32_t num_threads = (num_warps_x * num_warps_z) * warp_size;
        constexpr uint32_t num_rows_per_cta = num_frags_x * num_warps_x * 16;
        auto partition_kv_kernel = SinglePrefillWithKVCacheKernel<
            /*partition_kv=*/true, GROUP_SIZE, CAUSAL, KV_LAYOUT, pos_encoding_mode, num_warps_x,
            num_warps_z, num_frags_x, num_frags_y, num_frags_z, DTypeIn, DTypeQKAccum, DTypeOut>;
        tensor_info_t<KV_LAYOUT, GROUP_SIZE, HEAD_DIM> qkv_info(qo_len, kv_len, num_kv_heads);
        uint32_t smem_size = (num_frags_x * num_warps_x + num_frags_z * num_warps_z * 2) * 16 *
                             HEAD_DIM * sizeof(DTypeIn);
        FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
            partition_kv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        int num_blocks_per_sm = 0;
        int num_sm = 0;
        FLASHINFER_CUDA_CALL(
            cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
        FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm, partition_kv_kernel, num_threads, smem_size));
        uint32_t max_num_kv_chunks =
            (num_blocks_per_sm * num_sm) /
            (num_kv_heads * ceil_div(qo_len * GROUP_SIZE, num_rows_per_cta));
        uint32_t num_chunks;
        if (max_num_kv_chunks > 0) {
          uint32_t chunk_size = max(ceil_div(kv_len, max_num_kv_chunks), 256);
          num_chunks = ceil_div(kv_len, chunk_size);
        } else {
          num_chunks = 0;
        }

        if (num_chunks <= 1 || tmp == nullptr) {
          // Enough parallelism, do not split-kv
          auto kernel = SinglePrefillWithKVCacheKernel<
              /*partition_kv=*/false, GROUP_SIZE, CAUSAL, KV_LAYOUT, pos_encoding_mode, num_warps_x,
              num_warps_z, num_frags_x, num_frags_y, num_frags_z, DTypeIn, DTypeQKAccum, DTypeOut>;
          void* args[] = {(void*)&q,
                          (void*)&k,
                          (void*)&v,
                          (void*)&o,
                          (void*)&tmp,
                          (void*)&lse,
                          (void*)&qkv_info,
                          (void*)&sm_scale,
                          (void*)&log2_rope_rcp_scale,
                          (void*)&log2_rope_rcp_theta};
          dim3 nblks(ceil_div(qo_len * GROUP_SIZE, num_rows_per_cta), 1, num_kv_heads);
          dim3 nthrs(32, num_warps_x, num_warps_z);
          FLASHINFER_CUDA_CALL(
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        } else {
          // Use cooperative groups to increase occupancy
          void* args[] = {(void*)&q,
                          (void*)&k,
                          (void*)&v,
                          (void*)&o,
                          (void*)&tmp,
                          (void*)&lse,
                          (void*)&qkv_info,
                          (void*)&sm_scale,
                          (void*)&log2_rope_rcp_scale,
                          (void*)&log2_rope_rcp_theta};
          dim3 nblks(ceil_div(qo_len * GROUP_SIZE, num_rows_per_cta), num_chunks, num_kv_heads);
          dim3 nthrs(32, num_warps_x, num_warps_z);
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)partition_kv_kernel, nblks, nthrs, args, smem_size, stream));
          const uint32_t num_qo_heads = num_kv_heads * GROUP_SIZE;
          FLASHINFER_CUDA_CALL(MergeStates(
              (DTypeOut*)tmp,
              (float*)(((DTypeOut*)tmp) + num_chunks * qo_len * num_qo_heads * HEAD_DIM), o, lse,
              num_chunks, qo_len, num_qo_heads, HEAD_DIM, stream));
        }
      }
    })
  });
  return cudaSuccess;
}

/*!
 * \brief FlashAttention prefill CUDA function for a single request.
 * \tparam DTypeIn The data type of input
 * \tparam DTypeOut The data type of output
 * \param q The query tensor.
 * \param k The key tensor.
 * \param v The value tensor.
 * \param o The output tensor.
 * \param tmp The temporary storage (only used for cooperative kernel).
 * \param lse The logsumexp values.
 * \param num_qo_heads The number of query and output heads.
 * \param num_kv_heads The number of key and value heads.
 * \param qo_len The length of query and output.
 * \param kv_len The length of key and value.
 * \param head_dim The dimension of each head.
 * \param causal Whether to use causal attention.
 * \param kv_layout The layout of input and output.
 * \param pos_encoding_mode The positional encoding mode.
 * \param allow_fp16_qk_reduction Whether to allow accumulating q*k^T with fp16.
 * \param rope_scale The scaling factor used in RoPE interpolation.
 * \param rope_theta The theta used in RoPE.
 * \param stream The cuda stream to execute the kernel on.
 * \return status Indicates whether CUDA calls are successful
 */
template <typename DTypeIn, typename DTypeOut>
cudaError_t SinglePrefillWithKVCache(DTypeIn* q, DTypeIn* k, DTypeIn* v, DTypeOut* o, float* tmp,
                                     float* lse, uint32_t num_qo_heads, uint32_t num_kv_heads,
                                     uint32_t qo_len, uint32_t kv_len, uint32_t head_dim,
                                     bool causal = true, QKVLayout kv_layout = QKVLayout::kNHD,
                                     PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
                                     bool allow_fp16_qk_reduction = false,
                                     std::optional<float> maybe_sm_scale = std::nullopt,
                                     float rope_scale = 1.f, float rope_theta = 1e4,
                                     cudaStream_t stream = nullptr) {
  const uint32_t group_size = num_qo_heads / num_kv_heads;
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  DISPATCH_ALLOW_FP16_QK_REDUCTION(
      allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION,
      {DISPATCH_GQA_GROUP_SIZE(
          group_size, GROUP_SIZE,
          {DISPATCH_CAUSAL(
              causal, CAUSAL,
              {DISPATCH_HEAD_DIM(
                  head_dim, HEAD_DIM,
                  {DISPATCH_POS_ENCODING_MODE(
                      pos_encoding_mode, pos_encoding_mode, {DISPATCH_LAYOUT(kv_layout, KV_LAYOUT, {
                        SinglePrefillWithKVCacheDispatched<GROUP_SIZE, HEAD_DIM, KV_LAYOUT,
                                                           pos_encoding_mode,
                                                           ALLOW_FP16_QK_REDUCTION, CAUSAL>(
                            q, k, v, o, tmp, lse, num_kv_heads, qo_len, kv_len, sm_scale,
                            rope_scale, rope_theta, stream);
                      })})})})})});
  return cudaSuccess;
}

template <uint32_t num_frags_x, uint32_t GROUP_SIZE, uint32_t HEAD_DIM, QKVLayout KV_LAYOUT,
          PosEncodingMode pos_encoding_mode, bool ALLOW_FP16_QK_REDUCTION, bool CAUSAL,
          typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(
    DTypeIn* q, IdType* request_indices, IdType* tile_indices, IdType* qo_indptr, DTypeIn* k,
    DTypeIn* v, IdType* kv_indptr, IdType* q_offset, IdType* k_rope_pos_offset, DTypeOut* o,
    float* tmp, float* lse, const uint32_t batch_size, const uint32_t num_qo_tiles,
    const uint32_t num_kv_heads, const float sm_scale, const float rope_scale,
    const float rope_theta, cudaStream_t stream = nullptr) {
  const float log2_rope_rcp_scale = -std::log2f(rope_scale);
  const float log2_rope_rcp_theta = -std::log2f(rope_theta);
  constexpr uint32_t num_warps_x = 4;
  constexpr uint32_t num_warps_z = 1;

  dim3 nblks(num_qo_tiles, 1, num_kv_heads);
  dim3 nthrs(32, num_warps_x, num_warps_z);
  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
  using DTypeQKAccum =
      typename std::conditional<ALLOW_FP16_QK_REDUCTION && std::is_same<DTypeIn, half>::value, half,
                                float>::type;

  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  int max_smem_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&max_smem_per_sm,
                                              cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
  // we expect each sm execute two threadblocks
  const int max_smem_per_threadblock = max_smem_per_sm / 2;

  const uint32_t max_num_frags_z_reg =
      (HEAD_DIM == 128 && num_frags_x == 2 && pos_encoding_mode == PosEncodingMode::kRoPELlama &&
       !ALLOW_FP16_QK_REDUCTION)
          ? 2
          : 4;
  const uint32_t max_num_frags_z_smem =
      (max_smem_per_threadblock / (16 * HEAD_DIM * sizeof(DTypeIn)) - num_frags_x * num_warps_x) /
      (2 * num_warps_z);

  DISPATCH_NUM_FRAGS_Z(min(max_num_frags_z_smem, max_num_frags_z_reg), num_frags_z, {
    if constexpr (is_invalid_configuration<DTypeQKAccum>(num_frags_x, num_frags_y, num_frags_z,
                                                         num_warps_x, num_warps_z)) {
      // Invalid configuration, skip
      std::ostringstream err_msg;
      err_msg << "FlashInfer Internal Error: Invalid configuration : num_frags_x=" << num_frags_x
              << " num_frags_y=" << num_frags_y << " num_frags_z=" << num_frags_z
              << " num_warps_x=" << num_warps_x << " num_warps_z=" << num_warps_z
              << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                 " and report the issue to the developers.";
      throw std::invalid_argument(err_msg.str());
    } else {
      auto kernel =
          BatchPrefillWithRaggedKVCacheKernel<GROUP_SIZE, CAUSAL, KV_LAYOUT, pos_encoding_mode,
                                              num_warps_x, num_warps_z, num_frags_x, num_frags_y,
                                              num_frags_z, DTypeIn, DTypeQKAccum, DTypeOut, IdType>;
      uint32_t smem_size = (num_frags_x * num_warps_x + num_frags_z * num_warps_z * 2) * 16 *
                           HEAD_DIM * sizeof(DTypeIn);
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      void* args[] = {(void*)&q,
                      (void*)&request_indices,
                      (void*)&tile_indices,
                      (void*)&qo_indptr,
                      (void*)&k,
                      (void*)&v,
                      (void*)&kv_indptr,
                      (void*)&q_offset,
                      (void*)&k_rope_pos_offset,
                      (void*)&o,
                      (void*)&tmp,
                      (void*)&lse,
                      (void*)&batch_size,
                      (void*)&sm_scale,
                      (void*)&log2_rope_rcp_scale,
                      (void*)&log2_rope_rcp_theta};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    }
  });
  return cudaSuccess;
}

template <PageStorage page_storage, QKVLayout kv_layout, uint32_t num_frags_x, uint32_t PAGE_SIZE,
          uint32_t GROUP_SIZE, uint32_t HEAD_DIM, PosEncodingMode pos_encoding_mode,
          bool ALLOW_FP16_QK_REDUCTION, bool CAUSAL, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(
    DTypeIn* q, IdType* request_indices, IdType* tile_indices, IdType* qo_indptr, IdType* q_offset,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv, DTypeOut* o, float* tmp,
    float* lse, uint32_t num_qo_tiles, float sm_scale, float rope_scale, float rope_theta,
    cudaStream_t stream) {
  const float log2_rope_rcp_scale = -std::log2f(rope_scale);
  const float log2_rope_rcp_theta = -std::log2f(rope_theta);
  constexpr uint32_t num_warps_x = 4;
  constexpr uint32_t num_warps_z = 1;
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t batch_size = paged_kv.batch_size;

  dim3 nblks(num_qo_tiles, 1, num_kv_heads);
  dim3 nthrs(32, num_warps_x, num_warps_z);

  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
  using DTypeQKAccum =
      typename std::conditional<ALLOW_FP16_QK_REDUCTION && std::is_same<DTypeIn, half>::value, half,
                                float>::type;

  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  int max_smem_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&max_smem_per_sm,
                                              cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
  // we expect each sm execute two threadblocks
  const int max_smem_per_threadblock = max_smem_per_sm / 2;

  const uint32_t max_num_frags_z_reg =
      (HEAD_DIM == 128 && num_frags_x == 2 && pos_encoding_mode == PosEncodingMode::kRoPELlama &&
       !ALLOW_FP16_QK_REDUCTION)
          ? 2
          : 4;
  const uint32_t max_num_frags_z_smem =
      (max_smem_per_threadblock / (16 * HEAD_DIM * sizeof(DTypeIn)) - num_frags_x * num_warps_x) /
      (2 * num_warps_z);

  DISPATCH_NUM_FRAGS_Z(min(max_num_frags_z_smem, max_num_frags_z_reg), num_frags_z, {
    if constexpr (is_invalid_configuration<DTypeQKAccum>(num_frags_x, num_frags_y, num_frags_z,
                                                         num_warps_x, num_warps_z)) {
      // Invalid configuration, skip
      std::ostringstream err_msg;
      err_msg << "FlashInfer Internal Error: Invalid configuration : num_frags_x=" << num_frags_x
              << " num_frags_y=" << num_frags_y << " num_frags_z=" << num_frags_z
              << " num_warps_x=" << num_warps_x << " num_warps_z=" << num_warps_z
              << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                 " and report the issue to the developers.";
      throw std::invalid_argument(err_msg.str());
    } else {
      auto kernel =
          BatchPrefillWithPagedKVCacheKernel<GROUP_SIZE, PAGE_SIZE, CAUSAL, pos_encoding_mode,
                                             num_warps_x, num_warps_z, num_frags_x, num_frags_y,
                                             num_frags_z, page_storage, kv_layout, DTypeIn,
                                             DTypeQKAccum, DTypeOut, IdType>;
      uint32_t smem_size = (num_frags_x * num_warps_x + num_frags_z * num_warps_z * 2) * 16 *
                           HEAD_DIM * sizeof(DTypeIn);
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      void* args[] = {(void*)&request_indices,
                      (void*)&tile_indices,
                      (void*)&q,
                      (void*)&paged_kv,
                      (void*)&qo_indptr,
                      (void*)&q_offset,
                      (void*)&o,
                      (void*)&tmp,
                      (void*)&lse,
                      (void*)&sm_scale,
                      (void*)&log2_rope_rcp_scale,
                      (void*)&log2_rope_rcp_theta};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    }
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_PREFILL_CUH_
