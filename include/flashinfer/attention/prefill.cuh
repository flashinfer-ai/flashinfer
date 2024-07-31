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

#include "../cp_async.cuh"
#include "../fastdiv.cuh"
#include "../layout.cuh"
#include "../math.cuh"
#include "../mma.cuh"
#include "../page.cuh"
#include "../permuted_smem.cuh"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "cascade.cuh"
#include "logits_post_hook.cuh"
#include "mask.cuh"
#include "warp_layout.cuh"

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
          (num_frags_y > 4 && num_frags_y % (2 * num_warps_x) != 0) ||
          (num_frags_x * (8 * num_frags_y + 2 * sizeof(DTypeQKAccum) * num_frags_z) >= 200));
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
template <typename T>
__device__ __forceinline__ void k_frag_apply_llama_rope(T* x_first_half, T* x_second_half,
                                                        const float* rope_freq,
                                                        const uint32_t kv_offset,
                                                        float scale = 1.f) {
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 2 3
    // ---------
    // 4 5 | 6 7
    uint32_t i = reg_id / 4, j = (reg_id % 4) / 2;
    __sincosf(float(kv_offset + 8 * i) * rope_freq[2 * j + reg_id % 2], &sin, &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin) * scale;
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin) * scale;
  }
}

template <typename T>
__device__ __forceinline__ void q_frag_apply_llama_rope(T* x_first_half, T* x_second_half,
                                                        const float* rope_freq,
                                                        const uint32_t qo_packed_offset,
                                                        const uint_fastdiv group_size,
                                                        float scale = 1.f) {
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 4 5
    // ---------
    // 2 3 | 6 7
    uint32_t i = ((reg_id % 4) / 2), j = (reg_id / 4);
    __sincosf(float((qo_packed_offset + 8 * i) / group_size) * rope_freq[2 * j + reg_id % 2], &sin,
              &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin) * scale;
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin) * scale;
  }
}

template <typename T, typename IdType>
__device__ __forceinline__ void q_frag_apply_llama_rope_with_pos(
    T* x_first_half, T* x_second_half, const float* rope_freq, const uint32_t qo_packed_offset,
    const uint_fastdiv group_size, const IdType* q_offset, float scale = 1.f) {
  float pos[2] = {static_cast<float>(q_offset[qo_packed_offset / group_size]),
                  static_cast<float>(q_offset[(qo_packed_offset + 8) / group_size])};
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 4 5
    // ---------
    // 2 3 | 6 7
    uint32_t i = ((reg_id % 4) / 2), j = (reg_id / 4);
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
                                           const uint32_t kv_len) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_warps = num_warps_x * num_warps_z;
  constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<T>();
  const uint32_t warp_idx = get_warp_idx<num_warps_x, num_warps_z>(), lane_idx = threadIdx.x;
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

template <bool produce_v, uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_y,
          uint32_t num_frags_z, PageStorage page_storage, QKVLayout kv_layout, typename DType,
          typename IdType>
__device__ __forceinline__ void page_produce_kv(
    smem_t smem, uint32_t* smem_offset,
    paged_kv_t<page_storage, kv_layout, DType, IdType>& paged_kv, const uint32_t kv_idx_base,
    const uint32_t packed_page_iter_base, const uint32_t kv_len, const IdType last_indptr) {
  constexpr SharedMemFillMode fill_mode =
      produce_v ? SharedMemFillMode::kFillZero : SharedMemFillMode::kNoFill;
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_warps = num_warps_x * num_warps_z;
  constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DType>();
  const uint32_t warp_idx = get_warp_idx<num_warps_x, num_warps_z>(), lane_idx = threadIdx.x;
  const uint32_t kv_head_idx = blockIdx.z;
  uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / 8;
  // NOTE(Zihao): num_frags_z * 4 / num_warps_x = num_warps_z * num_frags_z * 4 / num_warps
  static_assert(num_frags_z * 4 % num_warps_x == 0);
#pragma unroll
  for (uint32_t i = 0; i < num_frags_z * 4 / num_warps_x; ++i) {
    uint32_t page_iter, entry_idx;
    paged_kv.page_size.divmod(
        packed_page_iter_base + warp_idx * 4 + lane_idx / 8 + 4 * num_warps * i, page_iter,
        entry_idx);
    DType* gptr = produce_v
                      ? paged_kv.protective_get_v_ptr(page_iter, kv_head_idx, entry_idx,
                                                      (lane_idx % 8) * num_elems_per_128b<DType>(),
                                                      last_indptr)
                      : paged_kv.protective_get_k_ptr(page_iter, kv_head_idx, entry_idx,
                                                      (lane_idx % 8) * num_elems_per_128b<DType>(),
                                                      last_indptr);
#pragma unroll
    for (uint32_t j = 0; j < channel_size_128b_in / 8; ++j) {
      smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
      *smem_offset = smem.advance_offset_by_column<8>(*smem_offset, j);
      gptr += 8 * num_elems_per_128b<DType>();
    }
    kv_idx += num_warps * 4;
    *smem_offset = smem.advance_offset_by_row<num_warps * 4, channel_size_128b_in>(*smem_offset) -
                   channel_size_128b_in;
  }
  *smem_offset -= num_warps_z * num_frags_z * 16 * channel_size_128b_in;
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

template <uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_x, uint32_t num_frags_y,
          typename DTypeIn>
__device__ __forceinline__ void load_q_global_smem(uint32_t packed_offset,
                                                   const uint32_t qo_upper_bound,
                                                   DTypeIn* q_ptr_base, const uint32_t qo_n_stride,
                                                   const uint32_t qo_h_stride,
                                                   const uint_fastdiv group_size, smem_t* q_smem) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
  const uint32_t lane_idx = threadIdx.x, warp_idx_x = get_warp_idx_x<num_warps_x, num_warps_z>();

  if (get_warp_idx_z<num_warps_x, num_warps_z>() == 0) {
    uint32_t q_smem_offset_w = smem_t::get_permuted_offset<channel_size_128b_in>(
        warp_idx_x * num_frags_x * 16 + lane_idx / 8, lane_idx % 8);

#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        uint32_t q, r;
        group_size.divmod(packed_offset + lane_idx / 8 + fx * 16 + j * 4, q, r);
        const uint32_t q_idx = q;
        DTypeIn* q_ptr = q_ptr_base + q * qo_n_stride + r * qo_h_stride;
#pragma unroll
        for (uint32_t fyo = 0; fyo < num_frags_y / 4; ++fyo) {
          // load q fragment from gmem to smem
          q_smem->load_128b_async<SharedMemFillMode::kNoFill>(q_smem_offset_w, q_ptr,
                                                              q_idx < qo_upper_bound);
          q_smem_offset_w = q_smem->advance_offset_by_column<8>(q_smem_offset_w, fyo);
          q_ptr += 8 * num_elems_per_128b<DTypeIn>();
        }
        q_smem_offset_w = q_smem->advance_offset_by_row<4, channel_size_128b_in>(q_smem_offset_w) -
                          2 * num_frags_y;
      }
    }
  }
}

template <uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_x, uint32_t num_frags_y,
          typename DTypeIn>
__device__ __forceinline__ void q_smem_inplace_apply_rotary_multiply_sm_scale(
    const uint32_t q_packed_idx, const uint32_t qo_len, const uint32_t kv_len,
    const uint_fastdiv group_size, smem_t* q_smem, uint32_t* q_smem_offset_r, float (*rope_freq)[4],
    const float sm_scale) {
  if (get_warp_idx_z<num_warps_x, num_warps_z>() == 0) {
    constexpr uint32_t head_dim = num_frags_y * 16;
    constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
    const uint32_t lane_idx = threadIdx.x;
    uint32_t q_frag_local[2][4];
    static_assert(num_frags_y % 4 == 0, "num_frags_y must be a multiple of 4");
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
      uint32_t q_smem_offset_r_first_half = *q_smem_offset_r;
#pragma unroll
      for (uint32_t fyi = 0; fyi < num_frags_y / 2; ++fyi) {
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        uint32_t q_smem_offset_r_last_half =
            q_smem->advance_offset_by_column<num_frags_y>(q_smem_offset_r_first_half, 0);
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_frag_apply_llama_rope<DTypeIn>(
            (DTypeIn*)q_frag_local[0], (DTypeIn*)q_frag_local[1], rope_freq[fyi],
            q_packed_idx + kv_len * group_size - qo_len * group_size + fx * 16 + lane_idx / 4,
            group_size, sm_scale);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        q_smem_offset_r_first_half =
            q_smem->advance_offset_by_column<2>(q_smem_offset_r_first_half, fyi);
      }
      *q_smem_offset_r += 16 * channel_size_128b_in;
    }
    *q_smem_offset_r -= num_frags_x * 16 * channel_size_128b_in;
  }
}

template <uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_x, uint32_t num_frags_y,
          typename DTypeIn, typename IdType>
__device__ __forceinline__ void q_smem_inplace_apply_rotary_with_pos_multiply_sm_scale(
    const uint32_t q_packed_idx_base, const IdType* q_offset, smem_t* q_smem,
    const uint_fastdiv group_size, uint32_t* q_smem_offset_r, float (*rope_freq)[4],
    const float sm_scale) {
  if (get_warp_idx_z<num_warps_x, num_warps_z>() == 0) {
    constexpr uint32_t head_dim = num_frags_y * 16;
    constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
    const uint32_t lane_idx = threadIdx.x;
    uint32_t q_frag_local[2][4];
    static_assert(num_frags_y % 4 == 0, "num_frags_y must be a multiple of 4");
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
      uint32_t q_smem_offset_r_first_half = *q_smem_offset_r;
#pragma unroll
      for (uint32_t fyi = 0; fyi < num_frags_y / 2; ++fyi) {
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        uint32_t q_smem_offset_r_last_half =
            q_smem->advance_offset_by_column<num_frags_y>(q_smem_offset_r_first_half, 0);
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_frag_apply_llama_rope_with_pos<DTypeIn>(
            (DTypeIn*)q_frag_local[0], (DTypeIn*)q_frag_local[1], rope_freq[fyi],
            q_packed_idx_base + fx * 16 + lane_idx / 4, group_size, q_offset, sm_scale);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        q_smem_offset_r_first_half =
            q_smem->advance_offset_by_column<2>(q_smem_offset_r_first_half, fyi);
      }
      *q_smem_offset_r += 16 * channel_size_128b_in;
    }
    *q_smem_offset_r -= num_frags_x * 16 * channel_size_128b_in;
  }
}

template <uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_x, uint32_t num_frags_y,
          typename DTypeIn>
__device__ __forceinline__ void q_smem_inplace_multiply_sm_scale(smem_t* q_smem,
                                                                 const float sm_scale) {
  const uint32_t warp_idx = get_warp_idx<num_warps_x, num_warps_z>(), lane_idx = threadIdx.x;
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
  constexpr uint32_t num_warps = num_warps_x * num_warps_z;
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

template <uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_y, uint32_t num_frags_z,
          typename DTypeIn>
__device__ __forceinline__ void k_smem_inplace_apply_rotary(const uint32_t kv_idx_base,
                                                            smem_t* k_smem,
                                                            uint32_t* k_smem_offset_r,
                                                            float (*rope_freq)[4]) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
  uint32_t k_frag_local[2][4];
  const uint32_t lane_idx = threadIdx.x;
  if constexpr (num_frags_y == 4 && num_warps_x == 4) {
    static_assert(num_warps_z == 1);
    const uint32_t warp_idx = get_warp_idx_x<num_warps_x, num_warps_z>();
    // horizontal-axis: y
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
      k_frag_apply_llama_rope<DTypeIn>((DTypeIn*)k_frag_local[0], (DTypeIn*)k_frag_local[1],
                                       rope_freq[fyi], kv_idx);
      k_smem->stmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
      k_smem->stmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
      *k_smem_offset_r += 32 * channel_size_128b_in;
      kv_idx += 32;
    }
    *k_smem_offset_r = (*k_smem_offset_r ^ (0x2 * (warp_idx % 2))) -
                       ((warp_idx / 2) + num_frags_z) * 16 * channel_size_128b_in;
  } else {
    const uint32_t warp_idx_x = get_warp_idx_x<num_warps_x, num_warps_z>(),
                   warp_idx_z = get_warp_idx_z<num_warps_x, num_warps_z>();
    static_assert(num_frags_y % (2 * num_warps_x) == 0);
    // horizontal axis: y
    // vertical axis: z
    // | (warp_idx_z, warp_idx_x)       | 1-16   | 16-32  | 32-48  | 48-64  | ...
    // | 1-16*num_frags_z               | (0, 0) | (0, 1) | (0, 2) | (0, 3) | ...
    // | 16*num_frags_z-32*num_frags_z  | (1, 0) | (1, 1) | (1, 2) | (1, 3) | ...
    // ...
    uint32_t kv_idx = kv_idx_base + (warp_idx_z * num_frags_z * 16) + lane_idx / 4;
    *k_smem_offset_r = *k_smem_offset_r ^ (0x2 * warp_idx_x);
#pragma unroll
    for (uint32_t i = 0; i < num_frags_z; ++i) {
      uint32_t k_smem_offset_r_first_half = *k_smem_offset_r;
#pragma unroll
      for (uint32_t j = 0; j < num_frags_y / (2 * num_warps_x); ++j) {
        uint32_t fyi = warp_idx_x + j * num_warps_x;
        k_smem->ldmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
        uint32_t k_smem_offset_r_last_half =
            k_smem->advance_offset_by_column<num_frags_y>(k_smem_offset_r_first_half, 0);
        k_smem->ldmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
        k_frag_apply_llama_rope<DTypeIn>((DTypeIn*)k_frag_local[0], (DTypeIn*)k_frag_local[1],
                                         rope_freq[fyi], kv_idx);
        k_smem->stmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
        k_smem->stmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
        k_smem_offset_r_first_half =
            k_smem->advance_offset_by_column<2 * num_warps_x>(k_smem_offset_r_first_half, fyi);
      }
      *k_smem_offset_r += 16 * channel_size_128b_in;
      kv_idx += 16;
    }
    *k_smem_offset_r =
        (*k_smem_offset_r ^ (0x2 * warp_idx_x)) - num_frags_z * 16 * channel_size_128b_in;
  }
}

template <LogitsPostHook logits_post_hook, uint32_t num_frags_x, uint32_t num_frags_y,
          uint32_t num_frags_z, typename DTypeQ, typename DTypeK, typename DTypeQKAccum>
__device__ __forceinline__ void compute_qk(smem_t* q_smem, uint32_t* q_smem_offset_r,
                                           smem_t* k_smem, uint32_t* k_smem_offset_r,
                                           DTypeQKAccum (*s_frag)[num_frags_z][8],
                                           const float soft_cap) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_q = head_dim / num_elems_per_128b<DTypeQ>();
  constexpr uint32_t channel_size_128b_k = head_dim / num_elems_per_128b<DTypeK>();
  constexpr uint32_t mma_sz_q_128b = 16 / num_elems_per_128b<DTypeQ>();
  constexpr uint32_t mma_sz_k_128b = 16 / num_elems_per_128b<DTypeK>();
  uint32_t a_frag[num_frags_x][4], b_frag[4];
  // compute q*k^T
#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
      q_smem->ldmatrix_m8n8x4(*q_smem_offset_r, a_frag[fx]);
      *q_smem_offset_r = q_smem->advance_offset_by_row<16, channel_size_128b_q>(*q_smem_offset_r);
    }

    *q_smem_offset_r = q_smem->advance_offset_by_column<mma_sz_q_128b>(*q_smem_offset_r, fy) -
                       num_frags_x * 16 * channel_size_128b_q;

#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      k_smem->ldmatrix_m8n8x4<DTypeK>(*k_smem_offset_r, b_frag);
      *k_smem_offset_r = k_smem->advance_offset_by_row<16, channel_size_128b_k>(*k_smem_offset_r);
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        if constexpr (std::is_same<DTypeQKAccum, float>::value) {
          if (fy == 0) {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeQ, MMAMode::kInit>(s_frag[fx][fz],
                                                                               a_frag[fx], b_frag);
          } else {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeQ>(s_frag[fx][fz], a_frag[fx], b_frag);
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
    *k_smem_offset_r = k_smem->advance_offset_by_column<mma_sz_k_128b>(*k_smem_offset_r, fy) -
                       num_frags_z * 16 * channel_size_128b_k;
  }
  *q_smem_offset_r -= num_frags_y * mma_sz_q_128b;
  *k_smem_offset_r -= num_frags_y * mma_sz_k_128b;

  if constexpr (std::is_same<DTypeQKAccum, float>::value) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
          s_frag[fx][fz][reg_id] =
              apply_logits_post_hook<logits_post_hook>(s_frag[fx][fz][reg_id], soft_cap);
        }
      }
    }
  } else {
    static_assert(std::is_same<DTypeQKAccum, half>::value);
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
          *(half2*)(&s_frag[fx][fz][reg_id * 2]) = apply_logits_post_hook<logits_post_hook>(
              *(half2*)(&s_frag[fx][fz][reg_id * 2]), soft_cap);
        }
      }
    }
  }
}

template <uint32_t num_frags_x, uint32_t num_frags_z, typename T>
__device__ __forceinline__ void apply_alibi_bias(const uint32_t qo_packed_idx_base,
                                                 const uint32_t kv_idx_base, const int32_t q_offset,
                                                 const uint_fastdiv group_size,
                                                 float (*alibi_slope)[2],
                                                 T (*s_frag)[num_frags_z][8]) {
  const int32_t lane_idx = threadIdx.x;
#pragma unroll
  for (int32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (int32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
      for (int32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const int32_t q_idx =
                          (qo_packed_idx_base + fx * 16 + lane_idx / 4 + 8 * ((reg_id % 4) / 2)) /
                          group_size,
                      kv_idx = kv_idx_base + fz * 16 + 2 * (lane_idx % 4) + 8 * (reg_id / 4) +
                               reg_id % 2;
        s_frag[fx][fz][reg_id] +=
            T(alibi_slope[fx][(reg_id % 4) / 2]) * T(kv_idx - q_idx - q_offset);
      }
    }
  }
}

template <bool partition_kv, MaskMode mask_mode, uint32_t num_frags_x, uint32_t num_frags_y,
          uint32_t num_frags_z, typename DTypeQKAccum>
__device__ __forceinline__ void mask_s(const uint32_t qo_packed_idx_base,
                                       const uint32_t kv_idx_base, const uint32_t qo_len,
                                       const uint32_t kv_len, const uint32_t chunk_end,
                                       const uint_fastdiv group_size, uint8_t* custom_mask,
                                       DTypeQKAccum (*s_frag)[num_frags_z][8]) {
  const uint32_t lane_idx = threadIdx.x;
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const uint32_t q_idx =
                           (qo_packed_idx_base + fx * 16 + lane_idx / 4 + 8 * ((reg_id % 4) / 2)) /
                           group_size,
                       kv_idx = kv_idx_base + fz * 16 + 2 * (lane_idx % 4) + 8 * (reg_id / 4) +
                                reg_id % 2;
        const bool out_of_boundary =
            (mask_mode == MaskMode::kCausal
                 ? (kv_idx + qo_len > kv_len + q_idx || (partition_kv && kv_idx >= chunk_end))
                 : kv_idx >= chunk_end);
        s_frag[fx][fz][reg_id] =
            (out_of_boundary ||
             (mask_mode == MaskMode::kCustom && q_idx < qo_len &&
              !((custom_mask[(q_idx * kv_len + kv_idx) / 8] >> ((q_idx * kv_len + kv_idx) % 8)) &
                1)))
                ? DTypeQKAccum(-5e4)
                : s_frag[fx][fz][reg_id];
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

template <uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z, typename DTypeS,
          typename DTypeV, typename DTypeQKAccum>
__device__ __forceinline__ void compute_sfm_v(smem_t* v_smem, uint32_t* v_smem_offset_r,
                                              DTypeQKAccum (*s_frag)[num_frags_z][8],
                                              float (*o_frag)[num_frags_y][8], float (*d)[2]) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_s = head_dim / num_elems_per_128b<DTypeS>();
  constexpr uint32_t channel_size_128b_v = head_dim / num_elems_per_128b<DTypeV>();

  DTypeS s_frag_f16[num_frags_x][num_frags_z][8];
  if constexpr (std::is_same<DTypeQKAccum, float>::value) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
        vec_cast<DTypeS, float, 8>(s_frag_f16[fx][fz], s_frag[fx][fz]);
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
    constexpr uint32_t v_step_128b = 16 / num_elems_per_128b<DTypeV>();
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      uint32_t b_frag[4];
      v_smem->ldmatrix_m8n8x4_trans<DTypeV>(*v_smem_offset_r, b_frag);
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        if constexpr (std::is_same<DTypeQKAccum, float>::value) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeS>(
              o_frag[fx][fy], (uint32_t*)(s_frag_f16[fx][fz]), b_frag);
        } else {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeS>(o_frag[fx][fy],
                                                             (uint32_t*)s_frag[fx][fz], b_frag);
        }
      }
      *v_smem_offset_r = v_smem->advance_offset_by_column<v_step_128b>(*v_smem_offset_r, fy);
    }
    *v_smem_offset_r =
        v_smem->advance_offset_by_row<16, channel_size_128b_v>(*v_smem_offset_r) - v_step_128b * num_frags_y;
  }
  *v_smem_offset_r -= 16 * num_frags_z * channel_size_128b_v;
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

    float o_scale[num_frags_x][2][num_warps_z];
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

#pragma unroll
        for (uint32_t i = 0; i < num_warps_z; ++i) {
          float2 md = smem_md[(((i * num_warps_x + get_warp_idx_x<num_warps_x, num_warps_z>()) *
                                    num_frags_x +
                                fx) *
                                   2 +
                               j) *
                                  warp_size +
                              lane_idx];
          float mi = md.x;
          o_scale[fx][j][i] = math::ptx_exp2(float(mi - m_new));
        }
        m[fx][j] = DTypeQKAccum(m_new);
        d[fx][j] = d_new;
      }
    }

    __syncthreads();

    // the following code saves shared memory usage.
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
            o_new[reg_id] += oi[reg_id] * o_scale[fx][(reg_id % 4) / 2][i];
          }
        }
        o_new.store(o_frag[fx][fy]);
        __syncthreads();
      }
    }
  }
}

template <uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_x, uint32_t num_frags_y,
          typename DTypeOut>
__device__ __forceinline__ void write_o_reg_gmem(
    float (*o_frag)[num_frags_y][8], smem_t* o_smem, DTypeOut* o_ptr_base,
    const uint32_t o_packed_idx_base, const uint32_t qo_upper_bound, const uint32_t qo_n_stride,
    const uint32_t qo_h_stride, const uint_fastdiv group_size) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_out = head_dim / num_elems_per_128b<DTypeOut>();
  const uint32_t warp_idx_x = get_warp_idx_x<num_warps_x, num_warps_z>();
  const uint32_t lane_idx = threadIdx.x;

  if (get_warp_idx_z<num_warps_x, num_warps_z>() == 0) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        uint32_t o_frag_f16[4];
        vec_cast<DTypeOut, float, 8>((DTypeOut*)o_frag_f16, o_frag[fx][fy]);
        uint32_t o_smem_offset_w = smem_t::get_permuted_offset<channel_size_128b_out>(
            (warp_idx_x * num_frags_x + fx) * 16 + lane_idx / 4, fy * 2);
        ((uint32_t*)(o_smem->base + o_smem_offset_w))[lane_idx % 4] = o_frag_f16[0];
        ((uint32_t*)(o_smem->base + o_smem_offset_w + 8 * channel_size_128b_out))[lane_idx % 4] =
            o_frag_f16[1];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1)))[lane_idx % 4] = o_frag_f16[2];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1) +
                     8 * channel_size_128b_out))[lane_idx % 4] = o_frag_f16[3];
      }
    }

    uint32_t o_smem_offset_w = smem_t::get_permuted_offset<channel_size_128b_out>(
        warp_idx_x * num_frags_x * 16 + lane_idx / 8, lane_idx % 8);

#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        uint32_t q, r;
        group_size.divmod(o_packed_idx_base + lane_idx / 8 + fx * 16 + j * 4, q, r);
        const uint32_t o_idx = q;
        DTypeOut* o_ptr = o_ptr_base + q * qo_n_stride + r * qo_h_stride;
#pragma unroll
        for (uint32_t fyo = 0; fyo < num_frags_y / 4; ++fyo) {
          if (o_idx < qo_upper_bound) {
            o_smem->store_128b(o_smem_offset_w, o_ptr);
          }
          o_ptr += 8 * num_elems_per_128b<DTypeOut>();
          o_smem_offset_w = o_smem->advance_offset_by_column<8>(o_smem_offset_w, fyo);
        }
        o_smem_offset_w = o_smem->advance_offset_by_row<4, channel_size_128b_out>(o_smem_offset_w) -
                          2 * num_frags_y;
      }
    }
  }
}

}  // namespace

/*!
 * \brief FlashAttention prefill CUDA kernel for a single request.
 * \tparam partition_kv Whether to split kv_len into chunks.
 * \tparam mask_mode The mask mode used in the attention operation.
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
template <LogitsPostHook logits_post_hook, bool partition_kv, MaskMode mask_mode,
          QKVLayout kv_layout, PosEncodingMode pos_encoding_mode, uint32_t num_frags_x,
          uint32_t num_frags_y, uint32_t num_frags_z, uint32_t num_warps_x, uint32_t num_warps_z,
          typename DTypeQ, typename DTypeKV, typename DTypeQKAccum, typename DTypeOut>
__global__ void SinglePrefillWithKVCacheKernel(
    DTypeQ* __restrict__ q, DTypeKV* __restrict__ k, DTypeKV* __restrict__ v,
    uint8_t* __restrict__ custom_mask, DTypeOut* __restrict__ o, float* __restrict__ lse,
    const uint32_t qo_len, const uint32_t kv_len, const uint_fastdiv group_size,
    const float logits_soft_cap, float sm_scale, const float log2_rope_rcp_scale,
    const float log2_rope_rcp_theta) {
  // TODO[AP]: This is stub
  using DTypeIn = DTypeQ;
  static_assert(sizeof(DTypeIn) == 2);
  static_assert(sizeof(DTypeOut) == 2);
  sm_scale *=
      (logits_post_hook == LogitsPostHook::kNone ? math::log2e : math::ptx_rcp(logits_soft_cap));
  const uint32_t lane_idx = threadIdx.x, warp_idx = get_warp_idx<num_warps_x, num_warps_z>();
  const uint32_t bx = blockIdx.x, chunk_idx = blockIdx.y, kv_head_idx = blockIdx.z;
  const uint32_t num_kv_heads = gridDim.z, num_qo_heads = num_kv_heads * group_size;
  constexpr uint32_t num_rows_per_cta = num_frags_x * num_warps_x * 16;
  const tensor_info_t<kv_layout, num_frags_y * 16> qkv_info(qo_len, kv_len, num_qo_heads,
                                                            num_kv_heads);
  float alibi_slopes[num_frags_x][2];

  const uint32_t num_chunks = gridDim.y;
  const uint32_t chunk_size = partition_kv ? ceil_div(kv_len, num_chunks) : kv_len;
  const uint32_t chunk_start = partition_kv ? chunk_idx * chunk_size : 0;
  const uint32_t chunk_end = partition_kv ? min((chunk_idx + 1) * chunk_size, kv_len) : kv_len;
  auto block = cg::this_thread_block();

  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
  constexpr uint32_t channel_size_128b_out = head_dim / num_elems_per_128b<DTypeOut>();

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
  const uint32_t qo_packed_idx_base =
      (bx * num_warps_x + get_warp_idx_x<num_warps_x, num_warps_z>()) * num_frags_x * 16;
  const uint32_t kv_n_stride = qkv_info.get_kv_n_stride(), qo_n_stride = qkv_info.get_qo_n_stride(),
                 qo_h_stride = qkv_info.get_qo_h_stride();
  smem_t qo_smem(smem);
  DTypeIn* q_ptr_base =
      q + qkv_info.get_qo_elem_offset(0, kv_head_idx * group_size,
                                      (lane_idx % 8) * num_elems_per_128b<DTypeIn>());
  DTypeOut* o_ptr_base =
      partition_kv
          ? o + chunk_idx * num_qo_heads * head_dim +
                qkv_info.get_qo_elem_offset(0, kv_head_idx * group_size,
                                            (lane_idx % 8) * num_elems_per_128b<DTypeOut>())
          : o + qkv_info.get_qo_elem_offset(0, kv_head_idx * group_size,
                                            (lane_idx % 8) * num_elems_per_128b<DTypeOut>());
  uint32_t q_smem_offset_r = smem_t::get_permuted_offset<channel_size_128b_in>(
      get_warp_idx_x<num_warps_x, num_warps_z>() * num_frags_x * 16 + lane_idx % 16, lane_idx / 16);

  load_q_global_smem<num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
      qo_packed_idx_base, qo_len, q_ptr_base, qo_n_stride, qo_h_stride, group_size, &qo_smem);

  cp_async::commit_group();
  cp_async::wait_group<0>();
  block.sync();

  if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
    q_smem_inplace_apply_rotary_multiply_sm_scale<num_warps_x, num_warps_z, num_frags_x,
                                                  num_frags_y, DTypeIn>(
        qo_packed_idx_base, qo_len, kv_len, group_size, &qo_smem, &q_smem_offset_r, rope_freq,
        sm_scale);
  } else {
    q_smem_inplace_multiply_sm_scale<num_warps_x, num_warps_z, num_frags_x, num_frags_y, DTypeIn>(
        &qo_smem, sm_scale);
  }

  if constexpr (pos_encoding_mode == PosEncodingMode::kALiBi) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t qo_head_idx =
            kv_head_idx * group_size +
            (qo_packed_idx_base + lane_idx / 4 + j * 8 + fx * 16) % group_size;
        alibi_slopes[fx][j] = get_alibi_slope(qo_head_idx, num_qo_heads) * math::log2e;
      }
    }
  }

  smem_t k_smem(smem + (num_warps_x * num_frags_x) * 16 * head_dim * sizeof(DTypeIn)),
      v_smem(smem + (num_warps_x * num_frags_x + num_warps_z * num_frags_z) * 16 * head_dim *
                        sizeof(DTypeIn));

  const uint32_t num_iterations = ceil_div(
      mask_mode == MaskMode::kCausal
          ? min(chunk_end - chunk_start,
                sub_if_greater_or_zero(kv_len - qo_len + ((bx + 1) * num_rows_per_cta) / group_size,
                                       chunk_start))
          : chunk_end - chunk_start,
      16 * num_warps_z * num_frags_z);

  const uint32_t mask_iteration =
      (mask_mode == MaskMode::kCausal
           ? min(chunk_end - chunk_start,
                 sub_if_greater_or_zero(kv_len + (bx * num_rows_per_cta) / group_size - qo_len,
                                        chunk_start))
           : (chunk_end - chunk_start)) /
      (16 * num_warps_z * num_frags_z);

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
      k_smem, &kv_smem_offset_w, &k_ptr, kv_n_stride, chunk_start, chunk_end);
  cp_async::commit_group();
  produce_kv<SharedMemFillMode::kFillZero, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
      v_smem, &kv_smem_offset_w, &v_ptr, kv_n_stride, chunk_start, chunk_end);
  cp_async::commit_group();

#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    cp_async::wait_group<1>();
    block.sync();

    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      k_smem_inplace_apply_rotary<num_warps_x, num_warps_z, num_frags_y, num_frags_z, DTypeIn>(
          chunk_start + iter * 16 * num_warps_z * num_frags_z, &k_smem, &k_smem_offset_r,
          rope_freq);
      block.sync();
    }

    // compute attention score
    compute_qk<logits_post_hook, num_frags_x, num_frags_y, num_frags_z, DTypeQ, DTypeKV>(
        &qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag, logits_soft_cap);

    if constexpr (pos_encoding_mode == PosEncodingMode::kALiBi) {
      apply_alibi_bias<num_frags_x, num_frags_z>(
          qo_packed_idx_base,
          chunk_start +
              (iter * num_warps_z + get_warp_idx_z<num_warps_x, num_warps_z>()) * num_frags_z * 16,
          int(kv_len) - int(qo_len), group_size, alibi_slopes, s_frag);
    }
    // apply mask
    if constexpr (mask_mode == MaskMode::kCustom) {
      mask_s<partition_kv, mask_mode, num_frags_x, num_frags_y, num_frags_z>(
          qo_packed_idx_base,
          chunk_start +
              (iter * num_warps_z + get_warp_idx_z<num_warps_x, num_warps_z>()) * num_frags_z * 16,
          qo_len, kv_len, chunk_end, group_size, custom_mask, s_frag);
    } else {
      if (iter >= mask_iteration) {
        mask_s<partition_kv, mask_mode, num_frags_x, num_frags_y, num_frags_z>(
            qo_packed_idx_base,
            chunk_start + (iter * num_warps_z + get_warp_idx_z<num_warps_x, num_warps_z>()) *
                              num_frags_z * 16,
            qo_len, kv_len, chunk_end, group_size, nullptr, s_frag);
      }
    }

    // compute m,d states in online softmax
    update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(s_frag, o_frag, m, d);

    block.sync();
    produce_kv<SharedMemFillMode::kNoFill, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        k_smem, &kv_smem_offset_w, &k_ptr, kv_n_stride,
        chunk_start + (iter + 1) * 16 * num_warps_z * num_frags_z, chunk_end);
    cp_async::commit_group();
    cp_async::wait_group<1>();
    block.sync();

    // compute sfm*v
    compute_sfm_v<num_frags_x, num_frags_y, num_frags_z, DTypeQ, DTypeKV>(&v_smem, &v_smem_offset_r, s_frag,
                                                                  o_frag, d);

    block.sync();
    produce_kv<SharedMemFillMode::kFillZero, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        v_smem, &kv_smem_offset_w, &v_ptr, kv_n_stride,
        chunk_start + (iter + 1) * 16 * num_warps_z * num_frags_z, chunk_end);
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
  write_o_reg_gmem<num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
      o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
      partition_kv ? qo_n_stride * num_chunks : qo_n_stride, qo_h_stride, group_size);

  // write lse
  if (lse != nullptr || partition_kv) {
    if (get_warp_idx_z<num_warps_x, num_warps_z>() == 0) {
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          uint32_t q, r;
          group_size.divmod(qo_packed_idx_base + lane_idx / 4 + j * 8 + fx * 16, q, r);
          const uint32_t qo_head_idx = kv_head_idx * group_size + r;
          const uint32_t qo_idx = q;
          if (qo_idx < qo_len) {
            if constexpr (partition_kv) {
              lse[(qo_idx * num_chunks + chunk_idx) * num_qo_heads + qo_head_idx] =
                  math::ptx_log2(d[fx][j]) + float(m[fx][j]);
            } else {
              lse[qo_idx * num_qo_heads + qo_head_idx] = math::ptx_log2(d[fx][j]) + float(m[fx][j]);
            }
          }
        }
      }
    }
  }
}

template <bool partition_kv, LogitsPostHook logits_post_hook, MaskMode mask_mode,
          QKVLayout kv_layout, PosEncodingMode pos_encoding_mode, uint32_t num_frags_x,
          uint32_t num_frags_y, uint32_t num_frags_z, uint32_t num_warps_x, uint32_t num_warps_z,
          typename DTypeQ, typename DTypeKV, typename DTypeQKAccum, typename DTypeOut, typename IdType>
__global__ void BatchPrefillWithRaggedKVCacheKernel(
    DTypeQ* __restrict__ q, IdType* __restrict__ request_indices,
    IdType* __restrict__ q_tile_indices, IdType* __restrict__ kv_tile_indices,
    IdType* __restrict__ q_indptr, DTypeKV* __restrict__ k, DTypeKV* __restrict__ v,
    IdType* __restrict__ kv_indptr, uint8_t* __restrict__ custom_mask,
    IdType* __restrict__ qk_indptr, IdType* __restrict__ q_offset,
    IdType* __restrict__ k_rope_pos_offset, IdType* __restrict__ o_indptr, DTypeOut* __restrict__ o,
    float* __restrict__ lse, bool* __restrict__ block_valid_mask,
    IdType* __restrict__ kv_chunk_size_ptr, const uint_fastdiv group_size,
    const float logits_soft_cap, float sm_scale, const float log2_rope_rcp_scale,
    const float log2_rope_rcp_theta) {
  // TODO[AP]: Just stub
  using DTypeIn = DTypeQ;
  static_assert(sizeof(DTypeIn) == 2);
  static_assert(sizeof(DTypeOut) == 2);
  sm_scale *=
      (logits_post_hook == LogitsPostHook::kNone ? math::log2e : math::ptx_rcp(logits_soft_cap));
  constexpr uint32_t head_dim = num_frags_y * 16;
  const uint32_t kv_chunk_size = *kv_chunk_size_ptr;

  auto block = cg::this_thread_block();
  const uint32_t bx = blockIdx.x, lane_idx = threadIdx.x,
                 warp_idx = get_warp_idx<num_warps_x, num_warps_z>(), kv_head_idx = blockIdx.z;
  if (block_valid_mask && !block_valid_mask[bx]) {
    return;
  }
  const uint32_t num_kv_heads = gridDim.z, num_qo_heads = group_size * num_kv_heads;
  const uint32_t request_idx = request_indices[bx], qo_tile_idx = q_tile_indices[bx],
                 kv_tile_idx = kv_tile_indices[bx];
  constexpr uint32_t num_rows_per_cta = num_frags_x * num_warps_x * 16;
  const uint32_t qo_len = q_indptr[request_idx + 1] - q_indptr[request_idx],
                 kv_len = kv_indptr[request_idx + 1] - kv_indptr[request_idx];
  const uint32_t chunk_size = partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * chunk_size : 0;
  const uint32_t chunk_end = partition_kv ? min((kv_tile_idx + 1) * chunk_size, kv_len) : kv_len;
  const tensor_info_t<kv_layout, num_frags_y * 16> qkv_info(qo_len, kv_len, num_qo_heads,
                                                            num_kv_heads);
  float alibi_slopes[num_frags_x][2];
  const uint32_t qo_upper_bound =
      min(qo_len, ceil_div((qo_tile_idx + 1) * num_rows_per_cta, group_size));

  constexpr uint32_t channel_size_128b_in = head_dim / num_elems_per_128b<DTypeIn>();
  constexpr uint32_t channel_size_128b_out = head_dim / num_elems_per_128b<DTypeOut>();

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

  const uint32_t qo_packed_idx_base =
      (qo_tile_idx * num_warps_x + get_warp_idx_x<num_warps_x, num_warps_z>()) * num_frags_x * 16;
  const uint32_t kv_n_stride = qkv_info.get_kv_n_stride(), qo_n_stride = qkv_info.get_qo_n_stride(),
                 qo_h_stride = qkv_info.get_qo_h_stride();
  smem_t qo_smem(smem);

  DTypeIn* q_ptr_base =
      q + qkv_info.get_qo_elem_offset(q_indptr[request_idx], kv_head_idx * group_size,
                                      (lane_idx % 8) * num_elems_per_128b<DTypeIn>());

  DTypeIn* o_ptr_base =
      partition_kv
          ? o + kv_tile_idx * num_qo_heads * head_dim +
                qkv_info.get_qo_elem_offset(o_indptr[request_idx], kv_head_idx * group_size,
                                            (lane_idx % 8) * num_elems_per_128b<DTypeOut>())
          : o + qkv_info.get_qo_elem_offset(o_indptr[request_idx], kv_head_idx * group_size,
                                            (lane_idx % 8) * num_elems_per_128b<DTypeOut>());

  uint32_t q_smem_offset_r = smem_t::get_permuted_offset<channel_size_128b_in>(
      get_warp_idx_x<num_warps_x, num_warps_z>() * num_frags_x * 16 + lane_idx % 16, lane_idx / 16);

  load_q_global_smem<num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
      qo_packed_idx_base, qo_upper_bound, q_ptr_base, qo_n_stride, qo_h_stride, group_size,
      &qo_smem);

  cp_async::commit_group();
  cp_async::wait_group<0>();
  block.sync();

  if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
    if (!q_offset) {
      q_smem_inplace_apply_rotary_multiply_sm_scale<num_warps_x, num_warps_z, num_frags_x,
                                                    num_frags_y, DTypeIn>(
          qo_packed_idx_base, qo_len, kv_len, group_size, &qo_smem, &q_smem_offset_r, rope_freq,
          sm_scale);
    } else {
      q_smem_inplace_apply_rotary_with_pos_multiply_sm_scale<num_warps_x, num_warps_z, num_frags_x,
                                                             num_frags_y, DTypeIn>(
          qo_packed_idx_base, q_offset + q_indptr[request_idx], &qo_smem, group_size,
          &q_smem_offset_r, rope_freq, sm_scale);
    }
  } else {
    q_smem_inplace_multiply_sm_scale<num_warps_x, num_warps_z, num_frags_x, num_frags_y, DTypeIn>(
        &qo_smem, sm_scale);
  }

  if constexpr (pos_encoding_mode == PosEncodingMode::kALiBi) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t qo_head_idx =
            kv_head_idx * group_size +
            (qo_packed_idx_base + lane_idx / 4 + j * 8 + fx * 16) % group_size;
        alibi_slopes[fx][j] = get_alibi_slope(qo_head_idx, num_qo_heads) * math::log2e;
      }
    }
  }

  const uint32_t num_iterations =
      ceil_div((mask_mode == MaskMode::kCausal
                    ? min(chunk_end - chunk_start,
                          sub_if_greater_or_zero(
                              kv_len - qo_len + ((qo_tile_idx + 1) * num_rows_per_cta) / group_size,
                              chunk_start))
                    : chunk_end - chunk_start),
               16 * num_warps_z * num_frags_z);

  const uint32_t mask_iteration =
      (mask_mode == MaskMode::kCausal
           ? min(chunk_end - chunk_start,
                 sub_if_greater_or_zero(
                     kv_len + (qo_tile_idx * num_rows_per_cta) / group_size - qo_len, chunk_start))
           : chunk_end - chunk_start) /
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

  DTypeIn* k_ptr = k + qkv_info.get_kv_elem_offset(
                           kv_indptr[request_idx] + chunk_start + warp_idx * 4 + lane_idx / 8,
                           kv_head_idx, (lane_idx % 8) * num_elems_per_128b<DTypeIn>());
  DTypeIn* v_ptr = v + qkv_info.get_kv_elem_offset(
                           kv_indptr[request_idx] + chunk_start + warp_idx * 4 + lane_idx / 8,
                           kv_head_idx, (lane_idx % 8) * num_elems_per_128b<DTypeIn>());

  produce_kv<SharedMemFillMode::kNoFill, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
      k_smem, &kv_smem_offset_w, &k_ptr, kv_n_stride, chunk_start, chunk_end);
  cp_async::commit_group();
  produce_kv<SharedMemFillMode::kFillZero, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
      v_smem, &kv_smem_offset_w, &v_ptr, kv_n_stride, chunk_start, chunk_end);
  cp_async::commit_group();

#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    cp_async::wait_group<1>();
    block.sync();

    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      k_smem_inplace_apply_rotary<num_warps_x, num_warps_z, num_frags_y, num_frags_z, DTypeIn>(
          (k_rope_pos_offset == nullptr ? 0 : k_rope_pos_offset[request_idx]) + chunk_start +
              iter * 16 * num_warps_z * num_frags_z,
          &k_smem, &k_smem_offset_r, rope_freq);
      block.sync();
    }

    // compute attention score
    compute_qk<logits_post_hook, num_frags_x, num_frags_y, num_frags_z, DTypeQ, DTypeKV>(
        &qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag, logits_soft_cap);

    if constexpr (pos_encoding_mode == PosEncodingMode::kALiBi) {
      // TODO(Zihao): handle the case that q_offset is specified
      apply_alibi_bias<num_frags_x, num_frags_z>(
          qo_packed_idx_base,
          chunk_start +
              (iter * num_warps_z + get_warp_idx_z<num_warps_x, num_warps_z>()) * num_frags_z * 16,
          int(kv_len) - int(qo_len), group_size, alibi_slopes, s_frag);
    }
    // apply mask
    if constexpr (mask_mode == MaskMode::kCustom) {
      mask_s<partition_kv, mask_mode, num_frags_x, num_frags_y, num_frags_z>(
          qo_packed_idx_base,
          chunk_start +
              (iter * num_warps_z + get_warp_idx_z<num_warps_x, num_warps_z>()) * num_frags_z * 16,
          qo_len, kv_len, chunk_end, group_size, custom_mask + qk_indptr[request_idx], s_frag);
    } else {
      if (iter >= mask_iteration) {
        mask_s<partition_kv, mask_mode, num_frags_x, num_frags_y, num_frags_z>(
            qo_packed_idx_base,
            chunk_start + (iter * num_warps_z + get_warp_idx_z<num_warps_x, num_warps_z>()) *
                              num_frags_z * 16,
            qo_len, kv_len, chunk_end, group_size, nullptr, s_frag);
      }
    }

    // compute m,d states in online softmax
    update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(s_frag, o_frag, m, d);

    block.sync();
    produce_kv<SharedMemFillMode::kNoFill, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        k_smem, &kv_smem_offset_w, &k_ptr, kv_n_stride,
        chunk_start + (iter + 1) * 16 * num_warps_z * num_frags_z, kv_len);
    cp_async::commit_group();
    cp_async::wait_group<1>();
    block.sync();

    // compute sfm*v
    compute_sfm_v<num_frags_x, num_frags_y, num_frags_z, DTypeQ, DTypeKV>(
        &v_smem, &v_smem_offset_r, s_frag, o_frag, d);

    block.sync();
    produce_kv<SharedMemFillMode::kFillZero, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        v_smem, &kv_smem_offset_w, &v_ptr, kv_n_stride,
        chunk_start + (iter + 1) * 16 * num_warps_z * num_frags_z, kv_len);
    cp_async::commit_group();
  }
  cp_async::wait_group<0>();
  block.sync();

  // threadblock synchronization
  threadblock_sync_mdo_states<num_warps_x, num_warps_z, num_frags_x, num_frags_y, DTypeQKAccum>(
      o_frag, (float*)smem, m, d, warp_idx, lane_idx);

  // normalize d
  normalize_d<num_frags_x, num_frags_y>(o_frag, d);

  const uint32_t num_kv_chunks = ceil_div(kv_len, kv_chunk_size);

  // write back
  write_o_reg_gmem<num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
      o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
      partition_kv ? qo_n_stride * num_kv_chunks : qo_n_stride, qo_h_stride, group_size);

  // write lse
  if (lse != nullptr) {
    if (get_warp_idx_z<num_warps_x, num_warps_z>() == 0) {
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          uint32_t q, r;
          group_size.divmod(qo_packed_idx_base + lane_idx / 4 + j * 8 + fx * 16, q, r);
          const uint32_t qo_head_idx = kv_head_idx * group_size + r;
          const uint32_t qo_idx = q;
          if (qo_idx < qo_len) {
            if constexpr (partition_kv) {
              lse[(o_indptr[request_idx] + qo_idx * num_kv_chunks + kv_tile_idx) * num_qo_heads +
                  qo_head_idx] = math::ptx_log2(d[fx][j]) + float(m[fx][j]);
            } else {
              lse[(o_indptr[request_idx] + qo_idx) * num_qo_heads + qo_head_idx] =
                  math::ptx_log2(d[fx][j]) + float(m[fx][j]);
            }
          }
        }
      }
    }
  }
}

template <bool partition_kv, LogitsPostHook logits_post_hook, MaskMode mask_mode,
          PosEncodingMode pos_encoding_mode, uint32_t num_frags_x, uint32_t num_frags_y,
          uint32_t num_frags_z, uint32_t num_warps_x, uint32_t num_warps_z,
          PageStorage page_storage, QKVLayout kv_layout, typename DTypeQ, typename DTypeKV, 
          typename DTypeQKAccum, typename DTypeOut, typename IdType>
__global__ void BatchPrefillWithPagedKVCacheKernel(
    IdType* __restrict__ request_indices, IdType* __restrict__ q_tile_indices,
    IdType* __restrict__ kv_tile_indices, DTypeQ* __restrict__ q,
    paged_kv_t<page_storage, kv_layout, DTypeKV, IdType> paged_kv, IdType* __restrict__ q_indptr,
    uint8_t* __restrict__ custom_mask, IdType* __restrict__ qk_indptr,
    IdType* __restrict__ q_offset, IdType* __restrict__ o_indptr, DTypeOut* __restrict__ o,
    float* __restrict__ lse, bool* __restrict__ block_valid_mask,
    IdType* __restrict__ kv_chunk_size_ptr, const uint_fastdiv group_size,
    const float logits_soft_cap, float sm_scale, const float log2_rope_rcp_scale,
    const float log2_rope_rcp_theta) {
  static_assert(sizeof(DTypeQ) == 2);
  static_assert(sizeof(DTypeKV) == 2 || sizeof(DTypeKV) == 1);
  static_assert(sizeof(DTypeOut) == 2);
  sm_scale *=
      (logits_post_hook == LogitsPostHook::kNone ? math::log2e : math::ptx_rcp(logits_soft_cap));
  auto block = cg::this_thread_block();
  const uint32_t kv_chunk_size = *kv_chunk_size_ptr;

  const uint32_t bx = blockIdx.x, lane_idx = threadIdx.x,
                 warp_idx = get_warp_idx<num_warps_x, num_warps_z>(), kv_head_idx = blockIdx.z;
  if (block_valid_mask && !block_valid_mask[bx]) {
    return;
  }
  const uint32_t num_kv_heads = gridDim.z, num_qo_heads = num_kv_heads * group_size;
  float alibi_slopes[num_frags_x][2];

  const uint32_t request_idx = request_indices[bx], qo_tile_idx = q_tile_indices[bx],
                 kv_tile_idx = kv_tile_indices[bx];
  constexpr uint32_t num_rows_per_cta = num_frags_x * num_warps_x * 16;
  const uint32_t qo_len = q_indptr[request_idx + 1] - q_indptr[request_idx],
                 kv_len = (paged_kv.indptr[request_idx + 1] != paged_kv.indptr[request_idx])
                              ? (paged_kv.indptr[request_idx + 1] - paged_kv.indptr[request_idx] -
                                 1) * paged_kv.page_size +
                                    paged_kv.last_page_len[request_idx]
                              : 0;
  const uint32_t chunk_size = partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * chunk_size : 0;
  const uint32_t chunk_end = partition_kv ? min((kv_tile_idx + 1) * chunk_size, kv_len) : kv_len;
  const uint32_t qo_upper_bound =
      min(qo_len, ceil_div((qo_tile_idx + 1) * num_rows_per_cta, group_size));

  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_q = head_dim / num_elems_per_128b<DTypeQ>();
  constexpr uint32_t channel_size_128b_kv = head_dim / num_elems_per_128b<DTypeKV>();
  constexpr uint32_t channel_size_128b_out = head_dim / num_elems_per_128b<DTypeOut>();

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

  const uint32_t qo_packed_idx_base =
      (qo_tile_idx * num_warps_x + get_warp_idx_x<num_warps_x, num_warps_z>()) * num_frags_x * 16;
  const uint32_t qo_n_stride = get_n_stride_impl<QKVLayout::kNHD, head_dim>(num_qo_heads),
                 qo_h_stride = get_h_stride_impl<QKVLayout::kNHD, head_dim>(qo_len);
  smem_t qo_smem(smem);
  DTypeQ* q_ptr_base =
      q + get_elem_offset_impl<QKVLayout::kNHD, head_dim>(
              q_indptr[request_idx], kv_head_idx * group_size,
              (lane_idx % 8) * num_elems_per_128b<DTypeQ>(), qo_len, num_qo_heads);
  DTypeOut* o_ptr_base =
      partition_kv ? o + kv_tile_idx * num_qo_heads * head_dim +
                         get_elem_offset_impl<QKVLayout::kNHD, head_dim>(
                             o_indptr[request_idx], kv_head_idx * group_size,
                             (lane_idx % 8) * num_elems_per_128b<DTypeOut>(), qo_len, num_qo_heads)
                   : o + get_elem_offset_impl<QKVLayout::kNHD, head_dim>(
                             o_indptr[request_idx], kv_head_idx * group_size,
                             (lane_idx % 8) * num_elems_per_128b<DTypeOut>(), qo_len, num_qo_heads);
  uint32_t q_smem_offset_r = smem_t::get_permuted_offset<channel_size_128b_q>(
      get_warp_idx_x<num_warps_x, num_warps_z>() * num_frags_x * 16 + lane_idx % 16, lane_idx / 16);

  load_q_global_smem<num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
      qo_packed_idx_base, qo_upper_bound, q_ptr_base, qo_n_stride, qo_h_stride, group_size,
      &qo_smem);

  cp_async::commit_group();
  cp_async::wait_group<0>();
  block.sync();

  if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
    if (q_offset == nullptr) {
      q_smem_inplace_apply_rotary_multiply_sm_scale<num_warps_x, num_warps_z, num_frags_x,
                                                    num_frags_y, DTypeQ>(
          qo_packed_idx_base, qo_len, kv_len, group_size, &qo_smem, &q_smem_offset_r, rope_freq,
          sm_scale);
    } else {
      q_smem_inplace_apply_rotary_with_pos_multiply_sm_scale<num_warps_x, num_warps_z, num_frags_x,
                                                             num_frags_y, DTypeQ>(
          qo_packed_idx_base, q_offset + q_indptr[request_idx], &qo_smem, group_size,
          &q_smem_offset_r, rope_freq, sm_scale);
    }
  } else {
    q_smem_inplace_multiply_sm_scale<num_warps_x, num_warps_z, num_frags_x, num_frags_y, DTypeQ>(
        &qo_smem, sm_scale);
  }

  if constexpr (pos_encoding_mode == PosEncodingMode::kALiBi) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t qo_head_idx =
            kv_head_idx * group_size +
            (qo_packed_idx_base + lane_idx / 4 + j * 8 + fx * 16) % group_size;
        alibi_slopes[fx][j] = get_alibi_slope(qo_head_idx, num_qo_heads) * math::log2e;
      }
    }
  }

  smem_t k_smem(smem + (num_warps_x * num_frags_x) * 16 * head_dim * sizeof(DTypeQ)),
      v_smem(smem + (num_warps_x * num_frags_x) * 16 * head_dim * sizeof(DTypeQ) +
                    (num_warps_z * num_frags_z) * 16 * head_dim * sizeof(DTypeKV));

  auto col_idx = [](uint32_t idx) { return sizeof(DTypeKV) == 2 ? idx : 0; };
  uint32_t k_smem_offset_r = smem_t::get_permuted_offset<channel_size_128b_kv>(
               get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16 + 8 * (lane_idx / 16) +
                   lane_idx % 8,
               col_idx((lane_idx % 16) / 8)),
           v_smem_offset_r = smem_t::get_permuted_offset<channel_size_128b_kv>(
               get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16 + lane_idx % 16,
               col_idx(lane_idx / 16)),
           kv_smem_offset_w = smem_t::get_permuted_offset<channel_size_128b_kv>(
               warp_idx * 4 + lane_idx / 8, lane_idx % 8);
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  uint32_t packed_page_iter_base = paged_kv.indptr[request_idx] * paged_kv.page_size + chunk_start;
  page_produce_kv<false, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
      k_smem, &kv_smem_offset_w, paged_kv, chunk_start, packed_page_iter_base, chunk_end,
      last_indptr);
  cp_async::commit_group();
  page_produce_kv<true, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
      v_smem, &kv_smem_offset_w, paged_kv, chunk_start, packed_page_iter_base, chunk_end,
      last_indptr);
  cp_async::commit_group();

  const uint32_t num_iterations =
      ceil_div((mask_mode == MaskMode::kCausal
                    ? min(chunk_end - chunk_start,
                          sub_if_greater_or_zero(
                              kv_len - qo_len + ((qo_tile_idx + 1) * num_rows_per_cta) / group_size,
                              chunk_start))
                    : chunk_end - chunk_start),
               16 * num_warps_z * num_frags_z);

  const uint32_t mask_iteration =
      (mask_mode == MaskMode::kCausal
           ? min(chunk_end - chunk_start,
                 sub_if_greater_or_zero(
                     kv_len + (qo_tile_idx * num_rows_per_cta) / group_size - qo_len, chunk_start))
           : chunk_end - chunk_start) /
      (16 * num_warps_z * num_frags_z);

#pragma unroll
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    cp_async::wait_group<1>();
    block.sync();

    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      static_assert(std::is_same<DTypeKV, half>::value); // is unsupported for fp8
      k_smem_inplace_apply_rotary<num_warps_x, num_warps_z, num_frags_y, num_frags_z, DTypeKV>(
          (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[request_idx]) +
              chunk_start + iter * 16 * num_warps_z * num_frags_z,
          &k_smem, &k_smem_offset_r, rope_freq);
      block.sync();
    }

    // compute attention score
    compute_qk<logits_post_hook, num_frags_x, num_frags_y, num_frags_z, DTypeQ, DTypeKV>(
        &qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag, logits_soft_cap);

    if constexpr (pos_encoding_mode == PosEncodingMode::kALiBi) {
      // TODO(Zihao): handle the case that q_offset is specified
      apply_alibi_bias<num_frags_x, num_frags_z>(
          qo_packed_idx_base,
          chunk_start +
              (iter * num_warps_z + get_warp_idx_z<num_warps_x, num_warps_z>()) * num_frags_z * 16,
          int(kv_len) - int(qo_len), group_size, alibi_slopes, s_frag);
    }
    // apply mask
    if constexpr (mask_mode == MaskMode::kCustom) {
      mask_s<partition_kv, mask_mode, num_frags_x, num_frags_y, num_frags_z>(
          qo_packed_idx_base,
          chunk_start +
              (iter * num_warps_z + get_warp_idx_z<num_warps_x, num_warps_z>()) * num_frags_z * 16,
          qo_len, kv_len, chunk_end, group_size, custom_mask + qk_indptr[request_idx], s_frag);
    } else {
      if (iter >= mask_iteration) {
        mask_s<partition_kv, mask_mode, num_frags_x, num_frags_y, num_frags_z>(
            qo_packed_idx_base,
            chunk_start + (iter * num_warps_z + get_warp_idx_z<num_warps_x, num_warps_z>()) *
                              num_frags_z * 16,
            qo_len, kv_len, chunk_end, group_size, nullptr, s_frag);
      }
    }

    // compute m,d states in online softmax
    update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(s_frag, o_frag, m, d);

    block.sync();
    packed_page_iter_base += 16 * num_warps_z * num_frags_z;
    page_produce_kv<false, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        k_smem, &kv_smem_offset_w, paged_kv,
        chunk_start + (iter + 1) * 16 * num_warps_z * num_frags_z, packed_page_iter_base, chunk_end,
        last_indptr);
    cp_async::commit_group();
    cp_async::wait_group<1>();
    block.sync();

    // compute sfm*v
    compute_sfm_v<num_frags_x, num_frags_y, num_frags_z, DTypeQ, DTypeKV>(
      &v_smem, &v_smem_offset_r, s_frag, o_frag, d);

    block.sync();
    page_produce_kv<true, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        v_smem, &kv_smem_offset_w, paged_kv,
        chunk_start + (iter + 1) * 16 * num_warps_z * num_frags_z, packed_page_iter_base, chunk_end,
        last_indptr);
    cp_async::commit_group();
  }
  cp_async::wait_group<0>();
  block.sync();

  // threadblock synchronization
  threadblock_sync_mdo_states<num_warps_x, num_warps_z, num_frags_x, num_frags_y, DTypeQKAccum>(
      o_frag, (float*)smem, m, d, warp_idx, lane_idx);

  // normalize d
  normalize_d<num_frags_x, num_frags_y>(o_frag, d);

  const uint32_t num_kv_chunks = ceil_div(kv_len, kv_chunk_size);

  // write_back
  write_o_reg_gmem<num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
      o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
      partition_kv ? qo_n_stride * num_kv_chunks : qo_n_stride, qo_h_stride, group_size);

  // write lse
  if (lse != nullptr) {
    if (get_warp_idx_z<num_warps_x, num_warps_z>() == 0) {
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          uint32_t q, r;
          group_size.divmod(qo_packed_idx_base + lane_idx / 4 + j * 8 + fx * 16, q, r);
          const uint32_t qo_head_idx = kv_head_idx * group_size + r;
          const uint32_t qo_idx = q;
          if (qo_idx < qo_upper_bound) {
            if constexpr (partition_kv) {
              lse[(o_indptr[request_idx] + qo_idx * num_kv_chunks + kv_tile_idx) * num_qo_heads +
                  qo_head_idx] = math::ptx_log2(d[fx][j]) + float(m[fx][j]);
            } else {
              lse[(o_indptr[request_idx] + qo_idx) * num_qo_heads + qo_head_idx] =
                  math::ptx_log2(d[fx][j]) + float(m[fx][j]);
            }
          }
        }
      }
    }
  }
}

template <uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK, QKVLayout KV_LAYOUT,
          PosEncodingMode pos_encoding_mode, bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename DTypeQ, typename DTypeKV, typename DTypeOut>
cudaError_t SinglePrefillWithKVCacheDispatched(DTypeQ* q, DTypeKV* k, DTypeKV* v,
                                               uint8_t* custom_mask, DTypeOut* o, DTypeOut* tmp,
                                               float* lse, uint32_t num_qo_heads,
                                               uint32_t num_kv_heads, uint32_t qo_len,
                                               uint32_t kv_len, float logits_soft_cap,
                                               float sm_scale, float rope_scale, float rope_theta,
                                               cudaStream_t stream) {
  const float log2_rope_rcp_scale = -std::log2f(rope_scale);
  const float log2_rope_rcp_theta = -std::log2f(rope_theta);
  if (kv_len < qo_len && MASK_MODE == MaskMode::kCausal) {
    std::ostringstream err_msg;
    err_msg << "When mask_mode is set to MaskMode::kCausal, kv_len must be greater than or equal "
               "to qo_len, got kv_len"
            << kv_len << " and qo_len " << qo_len;
    throw std::invalid_argument(err_msg.str());
  }

  const uint32_t group_size = num_qo_heads / num_kv_heads;
  const uint_fastdiv group_size_fastdiv(group_size);
  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
  WarpLayout warp_layout;
  if (qo_len * group_size > 64 && HEAD_DIM < 256) {
    warp_layout = WarpLayout::k4x1x2;
  } else {
    warp_layout = WarpLayout::k4x1x1;
  }

  DISPATCH_WARP_LAYOUT(warp_layout, WARP_LAYOUT, {
    constexpr uint32_t num_frags_x = get_num_frags_x<WARP_LAYOUT>();
    // [AP] it's incorrect
    using DTypeQKAccum =
        typename std::conditional<ALLOW_FP16_QK_REDUCTION && std::is_same<DTypeQ, half>::value,
                                  half, float>::type;

    int dev_id = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    int max_smem_per_sm = 0;
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
    // we expect each sm execute two threadblocks
    const int max_smem_per_threadblock = max_smem_per_sm / 2;

    constexpr uint32_t num_warps_x = get_num_warps_x<WARP_LAYOUT>();
    constexpr uint32_t num_warps_z = get_num_warps_z<WARP_LAYOUT>();
    const uint32_t max_num_frags_z_reg =
        (HEAD_DIM == 128 && num_frags_x == 2 && pos_encoding_mode == PosEncodingMode::kRoPELlama &&
         !ALLOW_FP16_QK_REDUCTION)
            ? 2
            : 4;
    // [AP] it's incorrect
    const uint32_t max_num_frags_z_smem =
        (max_smem_per_threadblock / (16 * HEAD_DIM * sizeof(DTypeQ)) - num_frags_x * num_warps_x) /
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
        auto partition_kv_kernel =
            SinglePrefillWithKVCacheKernel<LOGITS_POST_HOOK, /*partition_kv=*/true, MASK_MODE,
                                           KV_LAYOUT, pos_encoding_mode, num_frags_x, num_frags_y,
                                           num_frags_z, num_warps_x, num_warps_z, DTypeQ, DTypeKV,
                                           DTypeQKAccum, DTypeOut>;
        uint32_t smem_size = num_frags_x * num_warps_x * 16 * HEAD_DIM * sizeof(DTypeQ) +
                             num_frags_z * num_warps_z * 2 * 16 * HEAD_DIM * sizeof(DTypeKV);
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
            (num_kv_heads * ceil_div(qo_len * group_size, num_rows_per_cta));
        uint32_t num_chunks;
        if (max_num_kv_chunks > 0) {
          uint32_t chunk_size = max(ceil_div(kv_len, max_num_kv_chunks), 256);
          num_chunks = ceil_div(kv_len, chunk_size);
        } else {
          num_chunks = 0;
        }

        if (num_chunks <= 1 || tmp == nullptr) {
          // Enough parallelism, do not split-kv
          auto kernel =
              SinglePrefillWithKVCacheKernel<LOGITS_POST_HOOK, /*partition_kv=*/false, MASK_MODE,
                                             KV_LAYOUT, pos_encoding_mode, num_frags_x, num_frags_y,
                                             num_frags_z, num_warps_x, num_warps_z, DTypeQ, DTypeKV,
                                             DTypeQKAccum, DTypeOut>;
          void* args[] = {(void*)&q,
                          (void*)&k,
                          (void*)&v,
                          (void*)&custom_mask,
                          (void*)&o,
                          (void*)&lse,
                          (void*)&qo_len,
                          (void*)&kv_len,
                          (void*)&group_size_fastdiv,
                          (void*)&logits_soft_cap,
                          (void*)&sm_scale,
                          (void*)&log2_rope_rcp_scale,
                          (void*)&log2_rope_rcp_theta};
          dim3 nblks(ceil_div(qo_len * group_size, num_rows_per_cta), 1, num_kv_heads);
          dim3 nthrs(32, num_warps_x, num_warps_z);
          FLASHINFER_CUDA_CALL(
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        } else {
          // Use cooperative groups to increase occupancy
          float* tmp_lse = (float*)(tmp + num_chunks * qo_len * num_qo_heads * HEAD_DIM);
          void* args[] = {(void*)&q,
                          (void*)&k,
                          (void*)&v,
                          (void*)&custom_mask,
                          (void*)&tmp,
                          (void*)&tmp_lse,
                          (void*)&qo_len,
                          (void*)&kv_len,
                          (void*)&group_size_fastdiv,
                          (void*)&logits_soft_cap,
                          (void*)&sm_scale,
                          (void*)&log2_rope_rcp_scale,
                          (void*)&log2_rope_rcp_theta};
          dim3 nblks(ceil_div(qo_len * group_size, num_rows_per_cta), num_chunks, num_kv_heads);
          dim3 nthrs(32, num_warps_x, num_warps_z);
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)partition_kv_kernel, nblks, nthrs, args, smem_size, stream));
          FLASHINFER_CUDA_CALL(MergeStates(tmp, tmp_lse, o, lse, num_chunks, qo_len, num_qo_heads,
                                           HEAD_DIM, stream));
        }
      }
    })
  });
  return cudaSuccess;
}

template <WarpLayout WARP_LAYOUT, uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK,
          QKVLayout KV_LAYOUT, PosEncodingMode pos_encoding_mode, bool ALLOW_FP16_QK_REDUCTION,
          MaskMode MASK_MODE, typename DTypeQ, typename DTypeKV, typename DTypeOut,
          typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(
    DTypeQ* q, IdType* request_indices, IdType* q_tile_indices, IdType* kv_tile_indices,
    IdType* q_indptr, DTypeKV* k, DTypeKV* v, IdType* kv_indptr, uint8_t* custom_mask,
    IdType* qk_indptr, IdType* q_offset, IdType* k_rope_pos_offset, IdType* o_indptr, DTypeOut* o,
    DTypeOut* tmp_v, float* tmp_s, float* lse, IdType* merge_indptr, bool* block_valid_mask,
    IdType* kv_chunk_size_ptr, uint32_t total_num_rows, uint32_t num_qo_heads,
    uint32_t padded_batch_size, uint32_t num_kv_heads, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, cudaStream_t stream = nullptr) {
  const float log2_rope_rcp_scale = -std::log2f(rope_scale);
  const float log2_rope_rcp_theta = -std::log2f(rope_theta);
  constexpr uint32_t num_frags_x = get_num_frags_x<WARP_LAYOUT>();
  constexpr uint32_t num_warps_x = get_num_warps_x<WARP_LAYOUT>();
  constexpr uint32_t num_warps_z = get_num_warps_z<WARP_LAYOUT>();
  const uint32_t group_size = num_qo_heads / num_kv_heads;
  const uint_fastdiv group_size_fastdiv(group_size);

  // TODO[AP]: Add support DTypeKV != DTypeQ
  using DTypeIn = DTypeQ;

  if (padded_batch_size == 0) {
    // No request, skip
    // this won't happen in CUDAGraph mode because we fixed the padded_batch_size
    return cudaSuccess;
  }

  dim3 nblks(padded_batch_size, 1, num_kv_heads);
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
      uint32_t smem_size = (num_frags_x * num_warps_x + num_frags_z * num_warps_z * 2) * 16 *
                           HEAD_DIM * sizeof(DTypeIn);
      if (tmp_v == nullptr) {
        // do not partition kv
        auto kernel = BatchPrefillWithRaggedKVCacheKernel<
            /*partition_kv=*/false, LOGITS_POST_HOOK, MASK_MODE, KV_LAYOUT, pos_encoding_mode,
            num_frags_x, num_frags_y, num_frags_z, num_warps_x, num_warps_z, DTypeQ, DTypeKV, DTypeQKAccum,
            DTypeOut, IdType>;
        FLASHINFER_CUDA_CALL(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        void* args[] = {(void*)&q,
                        (void*)&request_indices,
                        (void*)&q_tile_indices,
                        (void*)&kv_tile_indices,
                        (void*)&q_indptr,
                        (void*)&k,
                        (void*)&v,
                        (void*)&kv_indptr,
                        (void*)&custom_mask,
                        (void*)&qk_indptr,
                        (void*)&q_offset,
                        (void*)&k_rope_pos_offset,
                        (void*)&o_indptr,
                        (void*)&o,
                        (void*)&lse,
                        (void*)&block_valid_mask,
                        (void*)&kv_chunk_size_ptr,
                        (void*)&group_size_fastdiv,
                        (void*)&logits_soft_cap,
                        (void*)&sm_scale,
                        (void*)&log2_rope_rcp_scale,
                        (void*)&log2_rope_rcp_theta};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      } else {
        // partition kv
        auto kernel = BatchPrefillWithRaggedKVCacheKernel<
            /*partition_kv=*/true, LOGITS_POST_HOOK, MASK_MODE, KV_LAYOUT, pos_encoding_mode,
            num_frags_x, num_frags_y, num_frags_z, num_warps_x, num_warps_z, DTypeIn, DTypeIn, DTypeQKAccum,
            DTypeOut, IdType>;
        FLASHINFER_CUDA_CALL(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        void* args[] = {(void*)&q,
                        (void*)&request_indices,
                        (void*)&q_tile_indices,
                        (void*)&kv_tile_indices,
                        (void*)&q_indptr,
                        (void*)&k,
                        (void*)&v,
                        (void*)&kv_indptr,
                        (void*)&custom_mask,
                        (void*)&qk_indptr,
                        (void*)&q_offset,
                        (void*)&k_rope_pos_offset,
                        (void*)&o_indptr,
                        (void*)&tmp_v,
                        (void*)&tmp_s,
                        (void*)&block_valid_mask,
                        (void*)&kv_chunk_size_ptr,
                        (void*)&group_size_fastdiv,
                        (void*)&logits_soft_cap,
                        (void*)&sm_scale,
                        (void*)&log2_rope_rcp_scale,
                        (void*)&log2_rope_rcp_theta};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
            tmp_v, tmp_s, merge_indptr, o, lse, total_num_rows, num_qo_heads, HEAD_DIM, stream));
      }
    }
  });
  return cudaSuccess;
}

template <PageStorage page_storage, WarpLayout WARP_LAYOUT, uint32_t HEAD_DIM,
          LogitsPostHook LOGITS_POST_HOOK, QKVLayout kv_layout, PosEncodingMode pos_encoding_mode,
          bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename DTypeQ, typename DTypeKV,
          typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(
    DTypeQ* q, IdType* request_indices, IdType* q_tile_indices, IdType* kv_tile_indices,
    IdType* q_indptr, IdType* q_offset,
    paged_kv_t<page_storage, kv_layout, DTypeKV, IdType> paged_kv, uint8_t* custom_mask,
    IdType* qk_indptr, IdType* o_indptr, DTypeOut* o, DTypeOut* tmp_v, float* tmp_s, float* lse,
    IdType* merge_indptr, bool* block_valid_mask, IdType* kv_chunk_size_ptr,
    uint32_t total_num_rows, uint32_t num_qo_heads, uint32_t padded_batch_size,
    float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta,
    cudaStream_t stream) {
  const float log2_rope_rcp_scale = -std::log2f(rope_scale);
  const float log2_rope_rcp_theta = -std::log2f(rope_theta);
  constexpr uint32_t num_frags_x = get_num_frags_x<WARP_LAYOUT>();
  constexpr uint32_t num_warps_x = get_num_warps_x<WARP_LAYOUT>();
  constexpr uint32_t num_warps_z = get_num_warps_z<WARP_LAYOUT>();
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t group_size = num_qo_heads / num_kv_heads;
  const uint_fastdiv group_size_fastdiv(group_size);

  if (padded_batch_size == 0) {
    // No request, skip
    // this won't happen in CUDAGraph mode because we fixed the padded_batch_size
    return cudaSuccess;
  }

  dim3 nblks(padded_batch_size, 1, num_kv_heads);
  dim3 nthrs(32, num_warps_x, num_warps_z);

  constexpr uint32_t num_frags_y = HEAD_DIM / 16;
  using DTypeQKAccum =
      typename std::conditional<ALLOW_FP16_QK_REDUCTION && std::is_same<DTypeQ, half>::value, half,
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
      (max_smem_per_threadblock / (16 * HEAD_DIM * sizeof(DTypeQ)) - num_frags_x * num_warps_x) /
      (2 * num_warps_z * sizeof(DTypeKV) / sizeof(DTypeQ));

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
    } else if constexpr ((HEAD_DIM * sizeof(DTypeKV)) % 128 != 0) {
      std::ostringstream err_msg;
      err_msg << "Incompatible HEAD_DIM with DTypeKV. Should be multiples of 128 byte chunk."
                 "HEAD_DIM=" << HEAD_DIM << ", sizeof(DTypeKV)=" << sizeof(DTypeKV);
      throw std::invalid_argument(err_msg.str());
    } else {
      uint32_t smem_size = num_frags_x * num_warps_x * 16 * HEAD_DIM * sizeof(DTypeQ) + 
                           num_frags_z * num_warps_z * 2 * 16 * HEAD_DIM * sizeof(DTypeKV);

      if (tmp_v == nullptr) {
        // do not partition kv
        auto kernel = BatchPrefillWithPagedKVCacheKernel<
            /*partition_kv=*/false, LOGITS_POST_HOOK, MASK_MODE, pos_encoding_mode, num_frags_x,
            num_frags_y, num_frags_z, num_warps_x, num_warps_z, page_storage, kv_layout, DTypeQ,
            DTypeKV, DTypeQKAccum, DTypeOut, IdType>;

        FLASHINFER_CUDA_CALL(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        void* args[] = {(void*)&request_indices,
                        (void*)&q_tile_indices,
                        (void*)&kv_tile_indices,
                        (void*)&q,
                        (void*)&paged_kv,
                        (void*)&q_indptr,
                        (void*)&custom_mask,
                        (void*)&qk_indptr,
                        (void*)&q_offset,
                        (void*)&o_indptr,
                        (void*)&o,
                        (void*)&lse,
                        (void*)&block_valid_mask,
                        (void*)&kv_chunk_size_ptr,
                        (void*)&group_size_fastdiv,
                        (void*)&logits_soft_cap,
                        (void*)&sm_scale,
                        (void*)&log2_rope_rcp_scale,
                        (void*)&log2_rope_rcp_theta};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      } else {
        auto kernel = BatchPrefillWithPagedKVCacheKernel<
            /*partition_kv=*/true, LOGITS_POST_HOOK, MASK_MODE, pos_encoding_mode, num_frags_x,
            num_frags_y, num_frags_z, num_warps_x, num_warps_z, page_storage, kv_layout, DTypeQ,
            DTypeKV, DTypeQKAccum, DTypeOut, IdType>;

        FLASHINFER_CUDA_CALL(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        void* args[] = {(void*)&request_indices,
                        (void*)&q_tile_indices,
                        (void*)&kv_tile_indices,
                        (void*)&q,
                        (void*)&paged_kv,
                        (void*)&q_indptr,
                        (void*)&custom_mask,
                        (void*)&qk_indptr,
                        (void*)&q_offset,
                        (void*)&o_indptr,
                        (void*)&tmp_v,
                        (void*)&tmp_s,
                        (void*)&block_valid_mask,
                        (void*)&kv_chunk_size_ptr,
                        (void*)&group_size_fastdiv,
                        (void*)&logits_soft_cap,
                        (void*)&sm_scale,
                        (void*)&log2_rope_rcp_scale,
                        (void*)&log2_rope_rcp_theta};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
            tmp_v, tmp_s, merge_indptr, o, lse, total_num_rows, num_qo_heads, HEAD_DIM, stream));
      }
    }
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_PREFILL_CUH_
