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

#include <memory>
#ifdef FLASHINFER_ENABLE_FP8
#include <cuda_fp8.h>
#endif
#include <cuda_runtime.h>

#include <optional>
#include <tuple>

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

namespace flashinfer {

namespace cg = cooperative_groups;
using cp_async::SharedMemFillMode;
using mma::MMAMode;

constexpr uint32_t warp_size = 32;

namespace {

/*!
 * \brief Return x - y if x > y, otherwise return 0.
 */
__device__ __forceinline__ uint32_t sub_if_greater_or_zero(uint32_t x, uint32_t y) {
  return (x > y) ? x - y : 0U;
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

}  // namespace

template <LogitsPostHook LOGITS_POST_HOOK, MaskMode MASK_MODE, QKVLayout KV_LAYOUT,
          PosEncodingMode POS_ENCODING_MODE, uint32_t NUM_WARPS_X, uint32_t NUM_WARPS_Z,
          uint32_t NUM_FRAGS_X, uint32_t NUM_FRAGS_Y, uint32_t NUM_FRAGS_Z, typename DTypeIn,
          typename DTypeQKAccum, typename DTypeOut>
struct AttentionModules {
  static constexpr uint32_t HEAD_DIM = NUM_FRAGS_Y * 16;
  static constexpr uint32_t NUM_WARPS = NUM_WARPS_X * NUM_WARPS_Z;
  static constexpr uint32_t CHANNEL_SIZE_128B_IN = HEAD_DIM / num_elems_per_128b<DTypeIn>();
  static constexpr uint32_t CHANNEL_SIZE_128B_OUT = HEAD_DIM / num_elems_per_128b<DTypeOut>();
  static constexpr uint32_t NUM_ROWS_PER_CTA = NUM_FRAGS_X * NUM_WARPS_X * 16;
  static constexpr uint32_t NUM_COLS_PER_ITER = NUM_FRAGS_Z * NUM_WARPS_Z * 16;

  static constexpr bool IS_INVALID_CONFIGURATION =
      ((NUM_FRAGS_Y < 4) || (NUM_FRAGS_Y == 4 && NUM_FRAGS_Z % 2 == 1) ||
       (NUM_FRAGS_Y > 4 && NUM_FRAGS_Y % (2 * NUM_WARPS_X) != 0) ||
       (NUM_FRAGS_X * (8 * NUM_FRAGS_Y + 2 * sizeof(DTypeQKAccum) * NUM_FRAGS_Z) >= 256));

  static __device__ __forceinline__ uint32_t get_warp_idx_x() {
    if constexpr (NUM_WARPS_X == 1) {
      return 0;
    } else {
      return threadIdx.y;
    }
  }

  static __device__ __forceinline__ uint32_t get_warp_idx_z() {
    if constexpr (NUM_WARPS_Z == 1) {
      return 0;
    } else {
      return threadIdx.z;
    }
  }

  static __device__ __forceinline__ uint32_t get_warp_idx() {
    return get_warp_idx_z() * NUM_WARPS_X + get_warp_idx_x();
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
  template <SharedMemFillMode fill_mode, typename T>
  static __device__ __forceinline__ void produce_kv(smem_t smem, uint32_t* smem_offset, T** gptr,
                                                    const uint32_t kv_n_stride,
                                                    const uint32_t kv_idx_base,
                                                    const uint32_t kv_len) {
    const uint32_t warp_idx = get_warp_idx(), lane_idx = threadIdx.x;
    uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / 8;
    // NOTE(Zihao): num_frags_z * 4 / num_warps_x = num_warps_z * num_frags_z * 4 / num_warps
    static_assert(NUM_FRAGS_Z * 4 % NUM_WARPS_X == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_FRAGS_Z * 4 / NUM_WARPS_X; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < NUM_FRAGS_Y / 4; ++j) {
        smem.load_128b_async<fill_mode>(*smem_offset, *gptr, kv_idx < kv_len);
        *smem_offset = smem.advance_offset_by_column<8>(*smem_offset, j);
        *gptr += 8 * num_elems_per_128b<T>();
      }
      kv_idx += NUM_WARPS * 4;
      *smem_offset = smem.advance_offset_by_row<NUM_WARPS * 4, CHANNEL_SIZE_128B_IN>(*smem_offset) -
                     2 * NUM_FRAGS_Y;
      *gptr += NUM_WARPS * 4 * kv_n_stride - 2 * NUM_FRAGS_Y * num_elems_per_128b<T>();
    }
    *smem_offset -= NUM_COLS_PER_ITER * CHANNEL_SIZE_128B_IN;
  }

  template <bool produce_v, uint32_t page_size, PageStorage page_storage, typename IdType>
  static __device__ __forceinline__ void page_produce_kv(
      smem_t smem, uint32_t* smem_offset,
      paged_kv_t<page_storage, KV_LAYOUT, DTypeIn, IdType>& paged_kv, const uint32_t kv_idx_base,
      const uint32_t page_iter_base, const uint32_t kv_len, const IdType last_indptr) {
    constexpr SharedMemFillMode fill_mode =
        produce_v ? SharedMemFillMode::kFillZero : SharedMemFillMode::kNoFill;
    const uint32_t warp_idx = get_warp_idx(), lane_idx = threadIdx.x;
    const uint32_t kv_head_idx = blockIdx.z;
    uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / 8;
    // NOTE(Zihao): num_frags_z * 4 / num_warps_x = num_warps_z * num_frags_z * 4 / num_warps
    static_assert(NUM_FRAGS_Z * 4 % NUM_WARPS_X == 0);
    if constexpr (page_size % 4 == 0) {
#pragma unroll
      for (uint32_t i = 0; i < NUM_FRAGS_Z * 4 / NUM_WARPS_X; ++i) {
        const uint32_t page_iter = page_iter_base + (4 * NUM_WARPS * i + warp_idx * 4) / page_size;
        const uint32_t entry_idx = (4 * NUM_WARPS * i + warp_idx * 4) % page_size + lane_idx / 8;
        DTypeIn* gptr = produce_v
                            ? paged_kv.protective_get_v_ptr(
                                  page_iter, kv_head_idx, entry_idx,
                                  (lane_idx % 8) * num_elems_per_128b<DTypeIn>(), last_indptr)
                            : paged_kv.protective_get_k_ptr(
                                  page_iter, kv_head_idx, entry_idx,
                                  (lane_idx % 8) * num_elems_per_128b<DTypeIn>(), last_indptr);
#pragma unroll
        for (uint32_t j = 0; j < NUM_FRAGS_Y / 4; ++j) {
          smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
          *smem_offset = smem.advance_offset_by_column<8>(*smem_offset, j);
          gptr += 8 * num_elems_per_128b<DTypeIn>();
        }
        kv_idx += NUM_WARPS * 4;
        *smem_offset =
            smem.advance_offset_by_row<NUM_WARPS * 4, CHANNEL_SIZE_128B_IN>(*smem_offset) -
            2 * NUM_FRAGS_Y;
      }
      *smem_offset -= NUM_COLS_PER_ITER * CHANNEL_SIZE_128B_IN;
    } else {
#pragma unroll
      for (uint32_t i = 0; i < NUM_FRAGS_Z * 4 / NUM_WARPS_X; ++i) {
        const uint32_t page_iter =
            page_iter_base + (4 * NUM_WARPS * i + warp_idx * 4 + lane_idx / 8) / page_size;
        const uint32_t entry_idx = (4 * NUM_WARPS * i + warp_idx * 4 + lane_idx / 8) % page_size;
        DTypeIn* gptr = produce_v
                            ? paged_kv.protective_get_v_ptr(
                                  page_iter, kv_head_idx, entry_idx,
                                  (lane_idx % 8) * num_elems_per_128b<DTypeIn>(), last_indptr)
                            : paged_kv.protective_get_k_ptr(
                                  page_iter, kv_head_idx, entry_idx,
                                  (lane_idx % 8) * num_elems_per_128b<DTypeIn>(), last_indptr);
#pragma unroll
        for (uint32_t j = 0; j < NUM_FRAGS_Y / 4; ++j) {
          smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
          *smem_offset = smem.advance_offset_by_column<8>(*smem_offset, j);
          gptr += 8 * num_elems_per_128b<DTypeIn>();
        }
        kv_idx += NUM_WARPS * 4;
        *smem_offset =
            smem.advance_offset_by_row<NUM_WARPS * 4, CHANNEL_SIZE_128B_IN>(*smem_offset) -
            2 * NUM_FRAGS_Y;
      }
      *smem_offset -= NUM_COLS_PER_ITER * CHANNEL_SIZE_128B_IN;
    }
  }

  static __device__ __forceinline__ void init_rope_freq(float (*rope_freq)[4],
                                                        const float log2_rope_rcp_scale,
                                                        const float log2_rope_rcp_theta) {
    const uint32_t lane_idx = threadIdx.x;
#pragma unroll
    for (uint32_t fy = 0; fy < NUM_FRAGS_Y / 2; ++fy) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        rope_freq[fy][j] =
            math::ptx_exp2(log2_rope_rcp_scale +
                           log2_rope_rcp_theta *
                               float(2 * ((fy * 16 + (j / 2) * 8 + (lane_idx % 4) * 2 + (j % 2)) %
                                          (HEAD_DIM / 2))) /
                               float(HEAD_DIM));
      }
    }
  }

  static __device__ __forceinline__ void init_states(float (*o_frag)[NUM_FRAGS_Y][8],
                                                     DTypeQKAccum (*m)[2], float (*d)[2]) {
#pragma unroll
    for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
      for (uint32_t fy = 0; fy < NUM_FRAGS_Y; ++fy) {
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
          o_frag[fx][fy][reg_id] = 0.f;
        }
      }
    }
#pragma unroll
    for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        m[fx][j] = DTypeQKAccum(-5e4);
        d[fx][j] = 1.f;
      }
    }
  }

  static __device__ __forceinline__ void load_q_global_smem(
      uint32_t packed_offset, const uint32_t qo_upper_bound, DTypeIn* q_ptr_base,
      const uint32_t qo_n_stride, const uint32_t qo_h_stride, const uint_fastdiv group_size,
      smem_t* q_smem) {
    const uint32_t lane_idx = threadIdx.x;
    const uint32_t warp_idx = get_warp_idx();
    uint32_t q_smem_offset_w = smem_t::get_permuted_offset<CHANNEL_SIZE_128B_IN>(
        warp_idx * 4 + lane_idx / 8, lane_idx % 8);

    static_assert(NUM_FRAGS_X * 4 % NUM_WARPS_Z == 0);
    // NOTE(Zihao): num_warps_x * num_frags_x * 4 / num_warps = num_frags_x * 4 / num_warps_z
#pragma unroll
    for (uint32_t fx = 0; fx < NUM_FRAGS_X * 4 / NUM_WARPS_Z; ++fx) {
      uint32_t q, r;
      group_size.divmod(packed_offset + (NUM_WARPS * fx + warp_idx) * 4 + lane_idx / 8, q, r);
      const uint32_t q_idx = q;
      DTypeIn* q_ptr = q_ptr_base + q * qo_n_stride + r * qo_h_stride;
      for (uint32_t fy = 0; fy < NUM_FRAGS_Y / 4; ++fy) {
        // load q fragment from gmem to smem
        q_smem->load_128b_async<SharedMemFillMode::kNoFill>(q_smem_offset_w, q_ptr,
                                                            q_idx < qo_upper_bound);
        q_smem_offset_w = q_smem->advance_offset_by_column<8>(q_smem_offset_w, fy);
        q_ptr += 8 * num_elems_per_128b<DTypeIn>();
      }
      q_smem_offset_w =
          q_smem->advance_offset_by_row<NUM_WARPS * 4, CHANNEL_SIZE_128B_IN>(q_smem_offset_w) -
          2 * NUM_FRAGS_Y;
    }
  }

  static __device__ __forceinline__ void q_smem_inplace_apply_rotary_multiply_sm_scale(
      const uint32_t q_packed_idx, const uint32_t qo_len, const uint32_t kv_len,
      const uint_fastdiv group_size, smem_t* q_smem, uint32_t* q_smem_offset_r,
      float (*rope_freq)[4], const float sm_scale) {
    const uint32_t lane_idx = threadIdx.x;
    uint32_t q_frag_local[2][4];
    static_assert(NUM_FRAGS_Y % 4 == 0, "num_frags_y must be a multiple of 4");
    const uint32_t warp_idx_x = get_warp_idx_x();
#pragma unroll
    for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
      uint32_t q_smem_offset_r_first_half = *q_smem_offset_r;
#pragma unroll
      for (uint32_t fyi = 0; fyi < NUM_FRAGS_Y / 2; ++fyi) {
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        uint32_t q_smem_offset_r_last_half =
            q_smem->advance_offset_by_column<NUM_FRAGS_Y>(q_smem_offset_r_first_half, 0);
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_frag_apply_llama_rope<DTypeIn>((DTypeIn*)q_frag_local[0], (DTypeIn*)q_frag_local[1],
                                         rope_freq[fyi],
                                         q_packed_idx + kv_len * group_size - qo_len * group_size +
                                             (warp_idx_x * NUM_FRAGS_X + fx) * 16 + lane_idx / 4,
                                         group_size, sm_scale);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        q_smem_offset_r_first_half =
            q_smem->advance_offset_by_column<2>(q_smem_offset_r_first_half, fyi);
      }
      *q_smem_offset_r += 16 * CHANNEL_SIZE_128B_IN;
    }
    *q_smem_offset_r -= NUM_FRAGS_X * 16 * CHANNEL_SIZE_128B_IN;
  }

  template <typename IdType>
  static __device__ __forceinline__ void q_smem_inplace_apply_rotary_with_pos_multiply_sm_scale(
      const uint32_t q_packed_idx_base, const IdType* q_offset, smem_t* q_smem,
      const uint_fastdiv group_size, uint32_t* q_smem_offset_r, float (*rope_freq)[4],
      const float sm_scale) {
    const uint32_t lane_idx = threadIdx.x;
    uint32_t q_frag_local[2][4];
    static_assert(NUM_FRAGS_Y % 4 == 0, "num_frags_y must be a multiple of 4");
#pragma unroll
    for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
      uint32_t q_smem_offset_r_first_half = *q_smem_offset_r;
#pragma unroll
      for (uint32_t fyi = 0; fyi < NUM_FRAGS_Y / 2; ++fyi) {
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        uint32_t q_smem_offset_r_last_half =
            q_smem->advance_offset_by_column<NUM_FRAGS_Y>(q_smem_offset_r_first_half, 0);
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_frag_apply_llama_rope_with_pos<DTypeIn>(
            (DTypeIn*)q_frag_local[0], (DTypeIn*)q_frag_local[1], rope_freq[fyi],
            q_packed_idx_base + fx * 16 + lane_idx / 4, group_size, q_offset, sm_scale);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        q_smem_offset_r_first_half =
            q_smem->advance_offset_by_column<2>(q_smem_offset_r_first_half, fyi);
      }
      *q_smem_offset_r += 16 * CHANNEL_SIZE_128B_IN;
    }
    *q_smem_offset_r -= NUM_FRAGS_X * 16 * CHANNEL_SIZE_128B_IN;
  }

  static __device__ __forceinline__ void q_smem_inplace_multiply_sm_scale(smem_t* q_smem,
                                                                          const float sm_scale) {
    const uint32_t warp_idx = get_warp_idx(), lane_idx = threadIdx.x;
    // NOTE(Zihao): num_warps_x * num_frags_x * 16 * head_dim / (num_warps * 256)
#pragma unroll
    for (uint32_t i = 0; i < NUM_FRAGS_X * HEAD_DIM / (NUM_WARPS_Z * 16); ++i) {
      vec_t<DTypeIn, 8> tmp;
      tmp.load((DTypeIn*)(q_smem->base) + (i * NUM_WARPS + warp_idx) * 256 + lane_idx * 8);
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        tmp[reg_id] *= sm_scale;
      }
      tmp.store((DTypeIn*)(q_smem->base) + (i * NUM_WARPS + warp_idx) * 256 + lane_idx * 8);
    }
  }

  static __device__ __forceinline__ void k_smem_inplace_apply_rotary(const uint32_t kv_idx_base,
                                                                     smem_t* k_smem,
                                                                     uint32_t* k_smem_offset_r,
                                                                     float (*rope_freq)[4]) {
    uint32_t k_frag_local[2][4];
    const uint32_t lane_idx = threadIdx.x;
    if constexpr (NUM_FRAGS_Y == 4 && NUM_WARPS_X == 4) {
      static_assert(NUM_WARPS_Z == 1);
      const uint32_t warp_idx = get_warp_idx_x();
      // horizontal-axis: y
      // horizontal-axis: y
      // vertical-axis: z
      //         | 1-16       | 16-32      | 32-48      | 48-64      |
      // | 1-16  | warp_idx=0 | warp_idx=1 | warp_idx=0 | warp_idx=1 |
      // | 16-32 | warp_idx=2 | warp_idx=3 | warp_idx=2 | warp_idx=3 |
      static_assert(NUM_FRAGS_Z % 2 == 0,
                    "when num_frags_y == 4, num_frags_z must be a multiple of 2");
      uint32_t kv_idx = kv_idx_base + (warp_idx / 2) * 16 + lane_idx / 4;
      *k_smem_offset_r =
          (*k_smem_offset_r ^ (0x2 * (warp_idx % 2))) + (warp_idx / 2) * 16 * CHANNEL_SIZE_128B_IN;
#pragma unroll
      for (uint32_t i = 0; i < NUM_FRAGS_Z / 2; ++i) {
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
        *k_smem_offset_r += 32 * CHANNEL_SIZE_128B_IN;
        kv_idx += 32;
      }
      *k_smem_offset_r = (*k_smem_offset_r ^ (0x2 * (warp_idx % 2))) -
                         ((warp_idx / 2) + NUM_FRAGS_Z) * 16 * CHANNEL_SIZE_128B_IN;
    } else {
      const uint32_t warp_idx_x = get_warp_idx_x(), warp_idx_z = get_warp_idx_z();
      static_assert(NUM_FRAGS_Y % (2 * NUM_WARPS_X) == 0);
      // horizontal axis: y
      // vertical axis: z
      // | (warp_idx_z, warp_idx_x)       | 1-16   | 16-32  | 32-48  | 48-64  | ...
      // | 1-16*num_frags_z               | (0, 0) | (0, 1) | (0, 2) | (0, 3) | ...
      // | 16*num_frags_z-32*num_frags_z  | (1, 0) | (1, 1) | (1, 2) | (1, 3) | ...
      // ...
      uint32_t kv_idx = kv_idx_base + (warp_idx_z * NUM_FRAGS_Z * 16) + lane_idx / 4;
      *k_smem_offset_r = *k_smem_offset_r ^ (0x2 * warp_idx_x);
#pragma unroll
      for (uint32_t i = 0; i < NUM_FRAGS_Z; ++i) {
        uint32_t k_smem_offset_r_first_half = *k_smem_offset_r;
#pragma unroll
        for (uint32_t j = 0; j < NUM_FRAGS_Y / (2 * NUM_WARPS_X); ++j) {
          uint32_t fyi = warp_idx_x + j * NUM_WARPS_X;
          k_smem->ldmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
          uint32_t k_smem_offset_r_last_half =
              k_smem->advance_offset_by_column<NUM_FRAGS_Y>(k_smem_offset_r_first_half, fyi);
          k_smem->ldmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
          k_frag_apply_llama_rope<DTypeIn>((DTypeIn*)k_frag_local[0], (DTypeIn*)k_frag_local[1],
                                           rope_freq[fyi], kv_idx);
          k_smem->stmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
          k_smem->stmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
          k_smem_offset_r_first_half =
              k_smem->advance_offset_by_column<2 * NUM_WARPS_X>(k_smem_offset_r_first_half, fyi);
        }
        *k_smem_offset_r += 16 * CHANNEL_SIZE_128B_IN;
        kv_idx += 16;
      }
      *k_smem_offset_r =
          (*k_smem_offset_r ^ (0x2 * warp_idx_x)) - NUM_FRAGS_Z * 16 * CHANNEL_SIZE_128B_IN;
    }
  }

  static __device__ __forceinline__ void compute_qk(smem_t* q_smem, uint32_t* q_smem_offset_r,
                                                    smem_t* k_smem, uint32_t* k_smem_offset_r,
                                                    DTypeQKAccum (*s_frag)[NUM_FRAGS_Z][8]) {
    uint32_t a_frag[NUM_FRAGS_X][4], b_frag[4];
    // compute q*k^T
#pragma unroll
    for (uint32_t fy = 0; fy < NUM_FRAGS_Y; ++fy) {
#pragma unroll
      for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
        q_smem->ldmatrix_m8n8x4(*q_smem_offset_r, a_frag[fx]);
        *q_smem_offset_r =
            q_smem->advance_offset_by_row<16, CHANNEL_SIZE_128B_IN>(*q_smem_offset_r);
      }

      *q_smem_offset_r = q_smem->advance_offset_by_column<2>(*q_smem_offset_r, fy) -
                         NUM_FRAGS_X * 16 * CHANNEL_SIZE_128B_IN;

#pragma unroll
      for (uint32_t fz = 0; fz < NUM_FRAGS_Z; ++fz) {
        k_smem->ldmatrix_m8n8x4(*k_smem_offset_r, b_frag);
        *k_smem_offset_r =
            k_smem->advance_offset_by_row<16, CHANNEL_SIZE_128B_IN>(*k_smem_offset_r);
#pragma unroll
        for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
          if constexpr (std::is_same<DTypeQKAccum, float>::value) {
            if (fy == 0) {
              mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn, MMAMode::kInit>(
                  s_frag[fx][fz], a_frag[fx], b_frag);
            } else {
              mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn>(s_frag[fx][fz], a_frag[fx],
                                                                 b_frag);
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
                         NUM_FRAGS_Z * 16 * CHANNEL_SIZE_128B_IN;
    }
    *q_smem_offset_r -= NUM_FRAGS_Y * 2;
    *k_smem_offset_r -= NUM_FRAGS_Y * 2;

    if constexpr (std::is_same<DTypeQKAccum, float>::value) {
#pragma unroll
      for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
        for (uint32_t fz = 0; fz < NUM_FRAGS_Z; ++fz) {
#pragma unroll
          for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
            s_frag[fx][fz][reg_id] =
                apply_logits_post_hook<LOGITS_POST_HOOK>(s_frag[fx][fz][reg_id]);
          }
        }
      }
    } else {
      static_assert(std::is_same<DTypeQKAccum, half>::value);
#pragma unroll
      for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
        for (uint32_t fz = 0; fz < NUM_FRAGS_Z; ++fz) {
#pragma unroll
          for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
            *(half2*)(&s_frag[fx][fz][reg_id * 2]) =
                apply_logits_post_hook<LOGITS_POST_HOOK>(*(half2*)(&s_frag[fx][fz][reg_id * 2]));
          }
        }
      }
    }
  }

  template <typename T>
  static __device__ __forceinline__ void apply_alibi_bias(
      const uint32_t qo_packed_idx_base, const uint32_t kv_idx_base, const int32_t q_offset,
      const uint_fastdiv group_size, float (*alibi_slope)[2], T (*s_frag)[NUM_FRAGS_Z][8]) {
    const int32_t lane_idx = threadIdx.x;
#pragma unroll
    for (int32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
      for (int32_t fz = 0; fz < NUM_FRAGS_Z; ++fz) {
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

  template <bool partition_kv>
  static __device__ __forceinline__ void mask_s(const uint32_t qo_packed_idx_base,
                                                const uint32_t kv_idx_base, const uint32_t qo_len,
                                                const uint32_t kv_len, const uint32_t chunk_end,
                                                const uint_fastdiv group_size, float* custom_mask,
                                                DTypeQKAccum (*s_frag)[NUM_FRAGS_Z][8]) {
    const uint32_t lane_idx = threadIdx.x;
#pragma unroll
    for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
      for (uint32_t fz = 0; fz < NUM_FRAGS_Z; ++fz) {
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
          const uint32_t q_idx = (qo_packed_idx_base + fx * 16 + lane_idx / 4 +
                                  8 * ((reg_id % 4) / 2)) /
                                 group_size,
                         kv_idx = kv_idx_base + fz * 16 + 2 * (lane_idx % 4) + 8 * (reg_id / 4) +
                                  reg_id % 2;
          const bool out_of_boundary =
              (MASK_MODE == MaskMode::kCausal
                   ? (kv_idx > kv_len + q_idx - qo_len || (partition_kv && kv_idx >= chunk_end))
                   : kv_idx >= chunk_end);
          s_frag[fx][fz][reg_id] =
              out_of_boundary ? DTypeQKAccum(-5e4)
                              : s_frag[fx][fz][reg_id] +
                                    DTypeQKAccum((MASK_MODE == MaskMode::kCustom && q_idx < qo_len)
                                                     ? custom_mask[q_idx * kv_len + kv_idx]
                                                     : 0.f);
        }
      }
    }
  }

  static __device__ __forceinline__ void update_mdo_states(DTypeQKAccum (*s_frag)[NUM_FRAGS_Z][8],
                                                           float (*o_frag)[NUM_FRAGS_Y][8],
                                                           DTypeQKAccum (*m)[2], float (*d)[2]) {
    if constexpr (std::is_same<DTypeQKAccum, float>::value) {
#pragma unroll
      for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          float m_prev = m[fx][j];
#pragma unroll
          for (uint32_t fz = 0; fz < NUM_FRAGS_Z; ++fz) {
            float m_local = max(max(s_frag[fx][fz][j * 2 + 0], s_frag[fx][fz][j * 2 + 1]),
                                max(s_frag[fx][fz][j * 2 + 4], s_frag[fx][fz][j * 2 + 5]));
            m[fx][j] = max(m[fx][j], m_local);
          }
          m[fx][j] = max(m[fx][j], math::shfl_xor_sync(m[fx][j], 0x2));
          m[fx][j] = max(m[fx][j], math::shfl_xor_sync(m[fx][j], 0x1));

          float o_scale = math::ptx_exp2(m_prev - m[fx][j]);
          d[fx][j] *= o_scale;
#pragma unroll
          for (uint32_t fy = 0; fy < NUM_FRAGS_Y; ++fy) {
            o_frag[fx][fy][j * 2 + 0] *= o_scale;
            o_frag[fx][fy][j * 2 + 1] *= o_scale;
            o_frag[fx][fy][j * 2 + 4] *= o_scale;
            o_frag[fx][fy][j * 2 + 5] *= o_scale;
          }
#pragma unroll
          for (uint32_t fz = 0; fz < NUM_FRAGS_Z; ++fz) {
            s_frag[fx][fz][j * 2 + 0] = math::ptx_exp2(s_frag[fx][fz][j * 2 + 0] - m[fx][j]);
            s_frag[fx][fz][j * 2 + 1] = math::ptx_exp2(s_frag[fx][fz][j * 2 + 1] - m[fx][j]);
            s_frag[fx][fz][j * 2 + 4] = math::ptx_exp2(s_frag[fx][fz][j * 2 + 4] - m[fx][j]);
            s_frag[fx][fz][j * 2 + 5] = math::ptx_exp2(s_frag[fx][fz][j * 2 + 5] - m[fx][j]);
          }
        }
      }
    } else if constexpr (std::is_same<DTypeQKAccum, half>::value) {
#pragma unroll
      for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
        half m_prev[2];
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          m_prev[j] = m[fx][j];
#pragma unroll
          for (uint32_t fz = 0; fz < NUM_FRAGS_Z; ++fz) {
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
          for (uint32_t fy = 0; fy < NUM_FRAGS_Y; ++fy) {
            o_frag[fx][fy][j * 2 + 0] *= o_scale;
            o_frag[fx][fy][j * 2 + 1] *= o_scale;
            o_frag[fx][fy][j * 2 + 4] *= o_scale;
            o_frag[fx][fy][j * 2 + 5] *= o_scale;
          }
          half2 m2 = make_half2(m[fx][j], m[fx][j]);
#pragma unroll
          for (uint32_t fz = 0; fz < NUM_FRAGS_Z; ++fz) {
            *(half2*)&s_frag[fx][fz][j * 2] = math::ptx_exp2(*(half2*)&s_frag[fx][fz][j * 2] - m2);
            *(half2*)&s_frag[fx][fz][j * 2 + 4] =
                math::ptx_exp2(*(half2*)&s_frag[fx][fz][j * 2 + 4] - m2);
          }
        }
      }
    }
  }

  static __device__ __forceinline__ void compute_sfm_v(smem_t* v_smem, uint32_t* v_smem_offset_r,
                                                       DTypeQKAccum (*s_frag)[NUM_FRAGS_Z][8],
                                                       float (*o_frag)[NUM_FRAGS_Y][8],
                                                       float (*d)[2]) {
    DTypeIn s_frag_f16[NUM_FRAGS_X][NUM_FRAGS_Z][8];
    if constexpr (std::is_same<DTypeQKAccum, float>::value) {
#pragma unroll
      for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
        for (uint32_t fz = 0; fz < NUM_FRAGS_Z; ++fz) {
          vec_cast<DTypeIn, float, 8>(s_frag_f16[fx][fz], s_frag[fx][fz]);
        }
      }
    }

#pragma unroll
    for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
      for (uint32_t fz = 0; fz < NUM_FRAGS_Z; ++fz) {
        if constexpr (std::is_same<DTypeQKAccum, float>::value) {
          mma::rowsum_f16f16f32(d[fx], s_frag_f16[fx][fz]);
        } else {
          mma::rowsum_f16f16f32(d[fx], s_frag[fx][fz]);
        }
      }
    }

#pragma unroll
    for (uint32_t fz = 0; fz < NUM_FRAGS_Z; ++fz) {
#pragma unroll
      for (uint32_t fy = 0; fy < NUM_FRAGS_Y; ++fy) {
        uint32_t b_frag[4];
        v_smem->ldmatrix_m8n8x4_trans(*v_smem_offset_r, b_frag);
#pragma unroll
        for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
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
      *v_smem_offset_r = v_smem->advance_offset_by_row<16, CHANNEL_SIZE_128B_IN>(*v_smem_offset_r) -
                         2 * NUM_FRAGS_Y;
    }
    *v_smem_offset_r -= 16 * NUM_FRAGS_Z * CHANNEL_SIZE_128B_IN;
  }

  static __device__ __forceinline__ void normalize_d(float (*o_frag)[NUM_FRAGS_Y][8],
                                                     float (*d)[2]) {
    float d_rcp[NUM_FRAGS_X][2];
    // compute reciprocal of d
#pragma unroll
    for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        d_rcp[fx][j] = math::ptx_rcp(d[fx][j]);
      }
    }

#pragma unroll
    for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
      for (uint32_t fy = 0; fy < NUM_FRAGS_Y; ++fy) {
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
  static __device__ __forceinline__ void threadblock_sync_mdo_states(
      float (*o_frag)[NUM_FRAGS_Y][8], float* smem_workspace, DTypeQKAccum (*m)[2], float (*d)[2],
      const uint32_t warp_idx, const uint32_t lane_idx) {
    // only necessary when blockDim.z > 1
    if constexpr (NUM_WARPS_Z > 1) {
      float2* smem_md = (float2*)smem_workspace;
      // o: [num_warps, warp_size, 8]
      // md: [num_warps, num_frags_x, 2, warp_size, 2 (m/d)]
#pragma unroll
      for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          smem_md[((warp_idx * NUM_FRAGS_X + fx) * 2 + j) * warp_size + lane_idx] =
              make_float2(float(m[fx][j]), d[fx][j]);
        }
      }

      float o_scale[NUM_FRAGS_X][2][NUM_WARPS_Z];
      // synchronize m,d first
      __syncthreads();
#pragma unroll
      for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          float m_new = -5e4, d_new = 1.f;
#pragma unroll
          for (uint32_t i = 0; i < NUM_WARPS_Z; ++i) {
            float2 md =
                smem_md[(((i * NUM_WARPS_X + get_warp_idx_x()) * NUM_FRAGS_X + fx) * 2 + j) *
                            warp_size +
                        lane_idx];
            float m_prev = m_new, d_prev = d_new;
            m_new = max(m_new, md.x);
            d_new = d_prev * math::ptx_exp2(m_prev - m_new) + md.y * math::ptx_exp2(md.x - m_new);
          }

#pragma unroll
          for (uint32_t i = 0; i < NUM_WARPS_Z; ++i) {
            float2 md =
                smem_md[(((i * NUM_WARPS_X + get_warp_idx_x()) * NUM_FRAGS_X + fx) * 2 + j) *
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
      for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
        for (uint32_t fy = 0; fy < NUM_FRAGS_Y; ++fy) {
          vec_t<float, 8> o_new;
          o_new.fill(0.f);
          vec_t<float, 8>::memcpy(smem_workspace + (warp_idx * warp_size + lane_idx) * 8,
                                  o_frag[fx][fy]);
          __syncthreads();
#pragma unroll
          for (uint32_t i = 0; i < NUM_WARPS_Z; ++i) {
            vec_t<float, 8> oi;
            oi.load(smem_workspace +
                    ((i * NUM_WARPS_X + get_warp_idx_x()) * warp_size + lane_idx) * 8);
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

  static __device__ __forceinline__ void grid_sync_mdo_states(float (*o_frag)[NUM_FRAGS_Y][8],
                                                              float* tmp, DTypeQKAccum (*m)[2],
                                                              float (*d)[2]) {
    const uint32_t bx = blockIdx.x;
    const uint32_t num_chunks = gridDim.y;
    const uint32_t kv_head_idx = blockIdx.z;
    // aggregate global state
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
#pragma unroll
    for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
      for (uint32_t fy = 0; fy < NUM_FRAGS_Y; ++fy) {
        vec_t<float, 8>::memcpy(
            tmp + ((fx * NUM_FRAGS_Y + fy) * grid.size() + grid.thread_rank()) * 8, o_frag[fx][fy]);
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
          o_frag[fx][fy][reg_id] = 0.f;
        }
      }
    }
    float2* tmp_md = (float2*)(tmp + NUM_FRAGS_X * NUM_FRAGS_Y * 8 * grid.size());
#pragma unroll
    for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
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
      float other_scale[NUM_FRAGS_X][2];
#pragma unroll
      for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
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
          for (uint32_t fy = 0; fy < NUM_FRAGS_Y; ++fy) {
            o_frag[fx][fy][j * 2 + 0] *= o_scale;
            o_frag[fx][fy][j * 2 + 1] *= o_scale;
            o_frag[fx][fy][j * 2 + 4] *= o_scale;
            o_frag[fx][fy][j * 2 + 5] *= o_scale;
          }
        }
#pragma unroll
        for (uint32_t fy = 0; fy < NUM_FRAGS_Y; ++fy) {
          vec_t<float, 8> o_frag_i;
          o_frag_i.load(
              tmp + ((fx * NUM_FRAGS_Y + fy) * grid.size() +
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

  static __device__ __forceinline__ void write_o_reg_gmem(
      float (*o_frag)[NUM_FRAGS_Y][8], smem_t* o_smem, DTypeOut* o_ptr_base,
      const uint32_t o_packed_idx_base, const uint32_t qo_upper_bound, const uint32_t qo_n_stride,
      const uint32_t qo_h_stride, const uint_fastdiv group_size) {
    const uint32_t warp_idx = get_warp_idx();
    const uint32_t lane_idx = threadIdx.x;

    if (get_warp_idx_z() == 0) {
#pragma unroll
      for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
        for (uint32_t fy = 0; fy < NUM_FRAGS_Y; ++fy) {
          uint32_t o_frag_f16[4];
          vec_cast<DTypeOut, float, 8>((DTypeOut*)o_frag_f16, o_frag[fx][fy]);
          uint32_t o_smem_offset_w = smem_t::get_permuted_offset<CHANNEL_SIZE_128B_OUT>(
              (get_warp_idx_x() * NUM_FRAGS_X + fx) * 16 + lane_idx / 4, fy * 2);
          ((uint32_t*)(o_smem->base + o_smem_offset_w))[lane_idx % 4] = o_frag_f16[0];
          ((uint32_t*)(o_smem->base + o_smem_offset_w + 8 * CHANNEL_SIZE_128B_OUT))[lane_idx % 4] =
              o_frag_f16[1];
          ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1)))[lane_idx % 4] = o_frag_f16[2];
          ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1) +
                       8 * CHANNEL_SIZE_128B_OUT))[lane_idx % 4] = o_frag_f16[3];
        }
      }
    }

    __syncthreads();

    uint32_t o_smem_offset_w = smem_t::get_permuted_offset<CHANNEL_SIZE_128B_OUT>(
        warp_idx * 4 + lane_idx / 8, lane_idx % 8);

    static_assert(NUM_FRAGS_X * 4 % NUM_WARPS_Z == 0);
    // NOTE(Zihao): num_warps_x * num_frags_x * 4 / num_warps = num_frags_x * 4 / num_warps_z
#pragma unroll
    for (uint32_t fx = 0; fx < NUM_FRAGS_X * 4 / NUM_WARPS_Z; ++fx) {
      uint32_t q, r;
      group_size.divmod(o_packed_idx_base + (fx * NUM_WARPS + warp_idx) * 4 + lane_idx / 8, q, r);
      const uint32_t o_idx = q;
      DTypeOut* o_ptr = o_ptr_base + q * qo_n_stride + r * qo_h_stride;
#pragma unroll
      for (uint32_t fy = 0; fy < NUM_FRAGS_Y / 4; ++fy) {
        if (o_idx < qo_upper_bound) {
          o_smem->store_128b(o_smem_offset_w, o_ptr);
        }
        o_ptr += 8 * num_elems_per_128b<DTypeOut>();
        o_smem_offset_w = o_smem->advance_offset_by_column<8>(o_smem_offset_w, fy);
      }
      o_smem_offset_w =
          o_smem->advance_offset_by_row<NUM_WARPS * 4, CHANNEL_SIZE_128B_OUT>(o_smem_offset_w) -
          2 * NUM_FRAGS_Y;
    }
  }

  template <bool PARTITION_KV>
  static __device__ __forceinline__ void single_prefill_with_kv_cache_body(
      DTypeIn* q, DTypeIn* k, DTypeIn* v, float* custom_mask, DTypeOut* o, void* tmp, float* lse,
      const uint32_t qo_len, const uint32_t kv_len, const uint_fastdiv group_size, float sm_scale,
      const float log2_rope_rcp_scale, const float log2_rope_rcp_theta) {
    static_assert(sizeof(DTypeIn) == 2);
    static_assert(sizeof(DTypeOut) == 2);

    sm_scale *= (LOGITS_POST_HOOK == LogitsPostHook::kNone ? math::log2e : 1.f / 30.f);
    const uint32_t lane_idx = threadIdx.x, warp_idx = get_warp_idx();
    const uint32_t bx = blockIdx.x, chunk_idx = blockIdx.y, kv_head_idx = blockIdx.z;
    const uint32_t num_kv_heads = gridDim.z, num_qo_heads = num_kv_heads * group_size;
    const tensor_info_t<KV_LAYOUT, NUM_FRAGS_Y * 16> qkv_info(qo_len, kv_len, num_qo_heads,
                                                              num_kv_heads);
    float alibi_slopes[NUM_FRAGS_X][2];

    const uint32_t num_chunks = gridDim.y;
    const uint32_t chunk_size = PARTITION_KV ? ceil_div(kv_len, num_chunks) : kv_len;
    const uint32_t chunk_start = PARTITION_KV ? chunk_idx * chunk_size : 0;
    const uint32_t chunk_end = PARTITION_KV ? min((chunk_idx + 1) * chunk_size, kv_len) : kv_len;
    auto block = cg::this_thread_block();

    extern __shared__ uint8_t smem[];

    DTypeQKAccum s_frag[NUM_FRAGS_X][NUM_FRAGS_Z][8];
    float o_frag[NUM_FRAGS_X][NUM_FRAGS_Y][8];
    DTypeQKAccum m[NUM_FRAGS_X][2];
    float d[NUM_FRAGS_X][2];
    float rope_freq[NUM_FRAGS_Y / 2][4];
    if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      init_rope_freq(rope_freq, log2_rope_rcp_scale, log2_rope_rcp_theta);
    }
    init_states(o_frag, m, d);

    // cooperative fetch q fragment from gmem to reg
    const uint32_t qo_packed_idx_base = bx * NUM_ROWS_PER_CTA;
    const uint32_t kv_n_stride = qkv_info.get_kv_n_stride(),
                   qo_n_stride = qkv_info.get_qo_n_stride(),
                   qo_h_stride = qkv_info.get_qo_h_stride();
    smem_t qo_smem(smem);
    DTypeIn* q_ptr_base =
        q + qkv_info.get_qo_elem_offset(0, kv_head_idx * group_size,
                                        (lane_idx % 8) * num_elems_per_128b<DTypeIn>());
    DTypeOut* o_ptr_base =
        PARTITION_KV
            ? ((DTypeOut*)tmp) + chunk_idx * num_qo_heads * HEAD_DIM +
                  qkv_info.get_qo_elem_offset(0, kv_head_idx * group_size,
                                              (lane_idx % 8) * num_elems_per_128b<DTypeOut>())
            : o + qkv_info.get_qo_elem_offset(0, kv_head_idx * group_size,
                                              (lane_idx % 8) * num_elems_per_128b<DTypeOut>());
    uint32_t q_smem_offset_r = smem_t::get_permuted_offset<CHANNEL_SIZE_128B_IN>(
        get_warp_idx_x() * NUM_FRAGS_X * 16 + lane_idx % 16, lane_idx / 16);

    load_q_global_smem(qo_packed_idx_base, qo_len, q_ptr_base, qo_n_stride, qo_h_stride, group_size,
                       &qo_smem);

    cp_async::commit_group();
    cp_async::wait_group<0>();
    block.sync();

    if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      if (get_warp_idx_z() == 0) {
        q_smem_inplace_apply_rotary_multiply_sm_scale(qo_packed_idx_base, qo_len, kv_len,
                                                      group_size, &qo_smem, &q_smem_offset_r,
                                                      rope_freq, sm_scale);
      }
    } else {
      q_smem_inplace_multiply_sm_scale(&qo_smem, sm_scale);
    }

    if constexpr (POS_ENCODING_MODE == PosEncodingMode::kALiBi) {
#pragma unroll
      for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          const uint32_t qo_head_idx =
              kv_head_idx * group_size + (qo_packed_idx_base + get_warp_idx_x() * NUM_FRAGS_X * 16 +
                                          lane_idx / 4 + j * 8 + fx * 16) %
                                             group_size;
          alibi_slopes[fx][j] = get_alibi_slope(qo_head_idx, num_qo_heads) * math::log2e;
        }
      }
    }

    smem_t k_smem(smem + (NUM_WARPS_X * NUM_FRAGS_X) * 16 * HEAD_DIM * sizeof(DTypeIn)),
        v_smem(smem + (NUM_WARPS_X * NUM_FRAGS_X + NUM_WARPS_Z * NUM_FRAGS_Z) * 16 * HEAD_DIM *
                          sizeof(DTypeIn));

    const uint32_t num_iterations = ceil_div(
        MASK_MODE == MaskMode::kCausal
            ? min(chunk_end - chunk_start,
                  sub_if_greater_or_zero(
                      kv_len - qo_len + ((bx + 1) * NUM_ROWS_PER_CTA) / group_size, chunk_start))
            : chunk_end - chunk_start,
        NUM_COLS_PER_ITER);

    const uint32_t mask_iteration =
        (MASK_MODE == MaskMode::kCausal
             ? min(chunk_end - chunk_start,
                   sub_if_greater_or_zero(kv_len + (bx * NUM_ROWS_PER_CTA) / group_size - qo_len,
                                          chunk_start))
             : (chunk_end - chunk_start)) /
        NUM_COLS_PER_ITER;

    DTypeIn* k_ptr =
        k + qkv_info.get_kv_elem_offset(chunk_start + warp_idx * 4 + lane_idx / 8, kv_head_idx,
                                        (lane_idx % 8) * num_elems_per_128b<DTypeIn>());
    DTypeIn* v_ptr =
        v + qkv_info.get_kv_elem_offset(chunk_start + warp_idx * 4 + lane_idx / 8, kv_head_idx,
                                        (lane_idx % 8) * num_elems_per_128b<DTypeIn>());
    uint32_t k_smem_offset_r = smem_t::get_permuted_offset<CHANNEL_SIZE_128B_IN>(
                 get_warp_idx_z() * NUM_FRAGS_Z * 16 + 8 * (lane_idx / 16) + lane_idx % 8,
                 (lane_idx % 16) / 8),
             v_smem_offset_r = smem_t::get_permuted_offset<CHANNEL_SIZE_128B_IN>(
                 get_warp_idx_z() * NUM_FRAGS_Z * 16 + lane_idx % 16, lane_idx / 16),
             kv_smem_offset_w = smem_t::get_permuted_offset<CHANNEL_SIZE_128B_IN>(
                 warp_idx * 4 + lane_idx / 8, lane_idx % 8);
    produce_kv<SharedMemFillMode::kNoFill>(k_smem, &kv_smem_offset_w, &k_ptr, kv_n_stride,
                                           chunk_start, chunk_end);
    cp_async::commit_group();
    produce_kv<SharedMemFillMode::kFillZero>(v_smem, &kv_smem_offset_w, &v_ptr, kv_n_stride,
                                             chunk_start, chunk_end);
    cp_async::commit_group();

#pragma unroll 1
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
      cp_async::wait_group<1>();
      block.sync();

      if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
        k_smem_inplace_apply_rotary(chunk_start + iter * NUM_COLS_PER_ITER, &k_smem,
                                    &k_smem_offset_r, rope_freq);
        block.sync();
      }

      // compute attention score
      compute_qk(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);

      if constexpr (POS_ENCODING_MODE == PosEncodingMode::kALiBi) {
        apply_alibi_bias(
            qo_packed_idx_base + get_warp_idx_x() * NUM_FRAGS_X * 16,
            chunk_start + iter * NUM_COLS_PER_ITER + get_warp_idx_z() * NUM_FRAGS_Z * 16,
            int(kv_len) - int(qo_len), group_size, alibi_slopes, s_frag);
      }
      // apply mask
      if constexpr (MASK_MODE == MaskMode::kCustom) {
        mask_s<PARTITION_KV>(
            qo_packed_idx_base + get_warp_idx_x() * NUM_FRAGS_X * 16,
            chunk_start + iter * NUM_COLS_PER_ITER + get_warp_idx_z() * NUM_FRAGS_Z * 16, qo_len,
            kv_len, chunk_end, group_size, custom_mask, s_frag);
      } else {
        if (iter >= mask_iteration) {
          mask_s<PARTITION_KV>(
              qo_packed_idx_base + get_warp_idx_x() * NUM_FRAGS_X * 16,
              chunk_start + iter * NUM_COLS_PER_ITER + get_warp_idx_z() * NUM_FRAGS_Z * 16, qo_len,
              kv_len, chunk_end, group_size, nullptr, s_frag);
        }
      }

      // compute m,d states in online softmax
      update_mdo_states(s_frag, o_frag, m, d);

      block.sync();
      produce_kv<SharedMemFillMode::kNoFill>(k_smem, &kv_smem_offset_w, &k_ptr, kv_n_stride,
                                             chunk_start + (iter + 1) * NUM_COLS_PER_ITER,
                                             chunk_end);
      cp_async::commit_group();
      cp_async::wait_group<1>();
      block.sync();

      // compute sfm*v
      compute_sfm_v(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);

      block.sync();
      produce_kv<SharedMemFillMode::kFillZero>(v_smem, &kv_smem_offset_w, &v_ptr, kv_n_stride,
                                               chunk_start + (iter + 1) * NUM_COLS_PER_ITER,
                                               chunk_end);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    block.sync();

    // threadblock synchronization
    threadblock_sync_mdo_states(o_frag, (float*)smem, m, d, warp_idx, lane_idx);

    // normalize d
    normalize_d(o_frag, d);

    // write back
    write_o_reg_gmem(o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
                     PARTITION_KV ? qo_n_stride * num_chunks : qo_n_stride, qo_h_stride,
                     group_size);

    // write lse
    if (lse != nullptr || PARTITION_KV) {
#pragma unroll
      for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          uint32_t q, r;
          group_size.divmod(qo_packed_idx_base + get_warp_idx_x() * NUM_FRAGS_X * 16 +
                                lane_idx / 4 + j * 8 + fx * 16,
                            q, r);
          const uint32_t qo_idx = q, qo_head_idx = kv_head_idx * group_size + r;
          if (qo_idx < qo_len) {
            if constexpr (PARTITION_KV) {
              float* tmp_lse =
                  (float*)(((DTypeOut*)tmp) + qo_len * num_chunks * num_qo_heads * HEAD_DIM);
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

  template <typename IdType>
  static __device__ __forceinline__ void batch_prefill_with_ragged_kv_cache_body(
      DTypeIn* q, IdType* request_indices, IdType* tile_indices, IdType* qo_indptr, DTypeIn* k,
      DTypeIn* v, IdType* kv_indptr, float* custom_mask, IdType* qk_indptr, IdType* q_offset,
      IdType* k_rope_pos_offset, DTypeOut* o, float* tmp, float* lse, uint32_t batch_size,
      const uint_fastdiv group_size, float sm_scale, float log2_rope_rcp_scale,
      float log2_rope_rcp_theta) {
    static_assert(sizeof(DTypeIn) == 2);
    static_assert(sizeof(DTypeOut) == 2);
    sm_scale *= (LOGITS_POST_HOOK == LogitsPostHook::kNone ? math::log2e : 1.f / 30.f);

    auto block = cg::this_thread_block();
    const uint32_t bx = blockIdx.x, lane_idx = threadIdx.x, warp_idx = get_warp_idx(),
                   kv_head_idx = blockIdx.z;
    const uint32_t num_kv_heads = gridDim.z, num_qo_heads = group_size * num_kv_heads;
    const uint32_t request_idx = request_indices[bx], tile_idx = tile_indices[bx];
    const uint32_t qo_len = qo_indptr[request_idx + 1] - qo_indptr[request_idx],
                   kv_len = kv_indptr[request_idx + 1] - kv_indptr[request_idx];
    const tensor_info_t<KV_LAYOUT, NUM_FRAGS_Y * 16> qkv_info(qo_len, kv_len, num_qo_heads,
                                                              num_kv_heads);
    float alibi_slopes[NUM_FRAGS_X][2];
    const uint32_t qo_upper_bound =
        min(qo_len, ceil_div((tile_idx + 1) * NUM_ROWS_PER_CTA, group_size));
    constexpr bool PARTITION_KV = false;

    static_assert(NUM_FRAGS_Z * NUM_FRAGS_Y % NUM_WARPS == 0);

    extern __shared__ uint8_t smem[];

    DTypeQKAccum s_frag[NUM_FRAGS_X][NUM_FRAGS_Z][8];
    float o_frag[NUM_FRAGS_X][NUM_FRAGS_Y][8];
    DTypeQKAccum m[NUM_FRAGS_X][2];
    float d[NUM_FRAGS_X][2];
    float rope_freq[NUM_FRAGS_Y / 2][4];

    if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      init_rope_freq(rope_freq, log2_rope_rcp_scale, log2_rope_rcp_theta);
    }
    init_states(o_frag, m, d);

    const uint32_t qo_packed_idx_base = tile_idx * NUM_ROWS_PER_CTA;
    const uint32_t kv_n_stride = qkv_info.get_kv_n_stride(),
                   qo_n_stride = qkv_info.get_qo_n_stride(),
                   qo_h_stride = qkv_info.get_qo_h_stride();
    smem_t qo_smem(smem);

    DTypeIn* q_ptr_base =
        q + qkv_info.get_qo_elem_offset(qo_indptr[request_idx], kv_head_idx * group_size,
                                        (lane_idx % 8) * num_elems_per_128b<DTypeIn>());
    DTypeIn* o_ptr_base =
        o + qkv_info.get_qo_elem_offset(qo_indptr[request_idx], kv_head_idx * group_size,
                                        (lane_idx % 8) * num_elems_per_128b<DTypeOut>());

    uint32_t q_smem_offset_r = smem_t::get_permuted_offset<CHANNEL_SIZE_128B_IN>(
        get_warp_idx_x() * NUM_FRAGS_X * 16 + lane_idx % 16, lane_idx / 16);

    load_q_global_smem(qo_packed_idx_base, qo_upper_bound, q_ptr_base, qo_n_stride, qo_h_stride,
                       group_size, &qo_smem);

    cp_async::commit_group();
    cp_async::wait_group<0>();
    block.sync();

    if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      if (get_warp_idx_z() == 0) {
        if (!q_offset) {
          q_smem_inplace_apply_rotary_multiply_sm_scale(qo_packed_idx_base, qo_len, kv_len,
                                                        group_size, &qo_smem, &q_smem_offset_r,
                                                        rope_freq, sm_scale);
        } else {
          q_smem_inplace_apply_rotary_with_pos_multiply_sm_scale<IdType>(
              qo_packed_idx_base, q_offset + qo_indptr[request_idx], &qo_smem, group_size,
              &q_smem_offset_r, rope_freq, sm_scale);
        }
      }
    } else {
      q_smem_inplace_multiply_sm_scale(&qo_smem, sm_scale);
    }

    if constexpr (POS_ENCODING_MODE == PosEncodingMode::kALiBi) {
#pragma unroll
      for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          const uint32_t qo_head_idx =
              kv_head_idx * group_size + (qo_packed_idx_base + get_warp_idx_x() * NUM_FRAGS_X * 16 +
                                          lane_idx / 4 + j * 8 + fx * 16) %
                                             group_size;
          alibi_slopes[fx][j] = get_alibi_slope(qo_head_idx, num_qo_heads) * math::log2e;
        }
      }
    }

    const uint32_t num_iterations = ceil_div(
        (MASK_MODE == MaskMode::kCausal
             ? min(kv_len, kv_len - qo_len + ((tile_idx + 1) * NUM_ROWS_PER_CTA) / group_size)
             : kv_len),
        NUM_COLS_PER_ITER);

    const uint32_t mask_iteration =
        (MASK_MODE == MaskMode::kCausal
             ? min(kv_len + (tile_idx * NUM_ROWS_PER_CTA) / group_size - qo_len, kv_len)
             : kv_len) /
        NUM_COLS_PER_ITER;

    smem_t k_smem(smem + (NUM_WARPS_X * NUM_FRAGS_X) * 16 * HEAD_DIM * sizeof(DTypeIn)),
        v_smem(smem + (NUM_WARPS_X * NUM_FRAGS_X + NUM_WARPS_Z * NUM_FRAGS_Z) * 16 * HEAD_DIM *
                          sizeof(DTypeIn));

    uint32_t k_smem_offset_r = smem_t::get_permuted_offset<CHANNEL_SIZE_128B_IN>(
                 get_warp_idx_z() * NUM_FRAGS_Z * 16 + 8 * (lane_idx / 16) + lane_idx % 8,
                 (lane_idx % 16) / 8),
             v_smem_offset_r = smem_t::get_permuted_offset<CHANNEL_SIZE_128B_IN>(
                 get_warp_idx_z() * NUM_FRAGS_Z * 16 + lane_idx % 16, lane_idx / 16),
             kv_smem_offset_w = smem_t::get_permuted_offset<CHANNEL_SIZE_128B_IN>(
                 warp_idx * 4 + lane_idx / 8, lane_idx % 8);

    DTypeIn* k_ptr = k + qkv_info.get_kv_elem_offset(
                             kv_indptr[request_idx] + warp_idx * 4 + lane_idx / 8, kv_head_idx,
                             (lane_idx % 8) * num_elems_per_128b<DTypeIn>());
    DTypeIn* v_ptr = v + qkv_info.get_kv_elem_offset(
                             kv_indptr[request_idx] + warp_idx * 4 + lane_idx / 8, kv_head_idx,
                             (lane_idx % 8) * num_elems_per_128b<DTypeIn>());

    produce_kv<SharedMemFillMode::kNoFill>(k_smem, &kv_smem_offset_w, &k_ptr, kv_n_stride, 0,
                                           kv_len);
    cp_async::commit_group();
    produce_kv<SharedMemFillMode::kFillZero>(v_smem, &kv_smem_offset_w, &v_ptr, kv_n_stride, 0,
                                             kv_len);
    cp_async::commit_group();

#pragma unroll 1
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
      cp_async::wait_group<1>();
      block.sync();

      if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
        k_smem_inplace_apply_rotary(
            (k_rope_pos_offset == nullptr ? 0 : k_rope_pos_offset[request_idx]) +
                iter * NUM_COLS_PER_ITER,
            &k_smem, &k_smem_offset_r, rope_freq);
        block.sync();
      }

      // compute attention score
      compute_qk(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);

      if constexpr (POS_ENCODING_MODE == PosEncodingMode::kALiBi) {
        // TODO(Zihao): handle the case that q_offset is specified
        apply_alibi_bias(qo_packed_idx_base + get_warp_idx_x() * NUM_FRAGS_X * 16,
                         iter * NUM_COLS_PER_ITER + get_warp_idx_z() * NUM_FRAGS_Z * 16,
                         int(kv_len) - int(qo_len), group_size, alibi_slopes, s_frag);
      }
      // apply mask
      if constexpr (MASK_MODE == MaskMode::kCustom) {
        mask_s<PARTITION_KV>(qo_packed_idx_base + get_warp_idx_x() * NUM_FRAGS_X * 16,
                             iter * NUM_COLS_PER_ITER + get_warp_idx_z() * NUM_FRAGS_Z * 16, qo_len,
                             kv_len, kv_len, group_size, custom_mask + qk_indptr[request_idx],
                             s_frag);
      } else {
        if (iter >= mask_iteration) {
          mask_s<PARTITION_KV>(qo_packed_idx_base + get_warp_idx_x() * NUM_FRAGS_X * 16,
                               iter * NUM_COLS_PER_ITER + get_warp_idx_z() * NUM_FRAGS_Z * 16,
                               qo_len, kv_len, kv_len, group_size, nullptr, s_frag);
        }
      }

      // compute m,d states in online softmax
      update_mdo_states(s_frag, o_frag, m, d);

      block.sync();
      produce_kv<SharedMemFillMode::kNoFill>(k_smem, &kv_smem_offset_w, &k_ptr, kv_n_stride,
                                             (iter + 1) * NUM_COLS_PER_ITER, kv_len);
      cp_async::commit_group();
      cp_async::wait_group<1>();
      block.sync();

      // compute sfm*v
      compute_sfm_v(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);

      block.sync();
      produce_kv<SharedMemFillMode::kFillZero>(v_smem, &kv_smem_offset_w, &v_ptr, kv_n_stride,
                                               (iter + 1) * NUM_COLS_PER_ITER, kv_len);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    block.sync();

    // threadblock synchronization
    threadblock_sync_mdo_states(o_frag, (float*)smem, m, d, warp_idx, lane_idx);

    // normalize d
    normalize_d(o_frag, d);

    // write back
    write_o_reg_gmem(o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len, qo_n_stride,
                     qo_h_stride, group_size);

    // write lse
    if (lse != nullptr) {
#pragma unroll
      for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          uint32_t q, r;
          group_size.divmod(qo_packed_idx_base + get_warp_idx_x() * NUM_FRAGS_X * 16 +
                                lane_idx / 4 + j * 8 + fx * 16,
                            q, r);
          const uint32_t qo_idx = q, qo_head_idx = kv_head_idx * group_size + r;
          if (qo_idx < qo_len) {
            lse[(qo_indptr[request_idx] + qo_idx) * num_qo_heads + qo_head_idx] =
                math::ptx_log2(d[fx][j]) + float(m[fx][j]);
          }
        }
      }
    }
  }

  template <uint32_t PAGE_SIZE, PageStorage PAGE_STORAGE, typename IdType>
  static __device__ __forceinline__ void batch_prefill_with_paged_kv_cache_body(
      IdType* request_indices, IdType* tile_indices, DTypeIn* q,
      paged_kv_t<PAGE_STORAGE, KV_LAYOUT, DTypeIn, IdType> paged_kv, IdType* qo_indptr,
      float* custom_mask, IdType* qk_indptr, IdType* q_offset, DTypeOut* o, float* tmp, float* lse,
      const uint_fastdiv group_size, float sm_scale, float log2_rope_rcp_scale,
      float log2_rope_rcp_theta) {
    static_assert(sizeof(DTypeIn) == 2);
    static_assert(sizeof(DTypeOut) == 2);
    sm_scale *= (LOGITS_POST_HOOK == LogitsPostHook::kNone ? math::log2e : 1.f / 30.f);
    auto block = cg::this_thread_block();

    const uint32_t bx = blockIdx.x, lane_idx = threadIdx.x, warp_idx = get_warp_idx(),
                   kv_head_idx = blockIdx.z;
    const uint32_t num_kv_heads = gridDim.z, num_qo_heads = num_kv_heads * group_size;
    float alibi_slopes[NUM_FRAGS_X][2];

    const uint32_t request_idx = request_indices[bx], tile_idx = tile_indices[bx];
    const uint32_t qo_len = qo_indptr[request_idx + 1] - qo_indptr[request_idx],
                   kv_len = (paged_kv.indptr[request_idx + 1] - paged_kv.indptr[request_idx] - 1) *
                                PAGE_SIZE +
                            paged_kv.last_page_len[request_idx];
    const uint32_t qo_upper_bound =
        min(qo_len, ceil_div((tile_idx + 1) * NUM_ROWS_PER_CTA, group_size));

    constexpr bool PARTITION_KV = false;

    static_assert(NUM_FRAGS_Z * NUM_FRAGS_Y % NUM_WARPS == 0);

    extern __shared__ uint8_t smem[];

    DTypeQKAccum s_frag[NUM_FRAGS_X][NUM_FRAGS_Z][8];
    float o_frag[NUM_FRAGS_X][NUM_FRAGS_Y][8];
    DTypeQKAccum m[NUM_FRAGS_X][2];
    float d[NUM_FRAGS_X][2];
    float rope_freq[NUM_FRAGS_Y / 2][4];

    if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      init_rope_freq(rope_freq, log2_rope_rcp_scale, log2_rope_rcp_theta);
    }
    init_states(o_frag, m, d);

    const uint32_t qo_packed_idx_base = tile_idx * NUM_ROWS_PER_CTA;
    const uint32_t qo_n_stride = get_n_stride_impl<QKVLayout::kNHD, HEAD_DIM>(num_qo_heads),
                   qo_h_stride = get_h_stride_impl<QKVLayout::kNHD, HEAD_DIM>(qo_len);
    smem_t qo_smem(smem);
    DTypeIn* q_ptr_base =
        q + get_elem_offset_impl<QKVLayout::kNHD, HEAD_DIM>(
                qo_indptr[request_idx], kv_head_idx * group_size,
                (lane_idx % 8) * num_elems_per_128b<DTypeIn>(), qo_len, num_qo_heads);
    DTypeIn* o_ptr_base =
        o + get_elem_offset_impl<QKVLayout::kNHD, HEAD_DIM>(
                qo_indptr[request_idx], kv_head_idx * group_size,
                (lane_idx % 8) * num_elems_per_128b<DTypeOut>(), qo_len, num_qo_heads);
    uint32_t q_smem_offset_r = smem_t::get_permuted_offset<CHANNEL_SIZE_128B_IN>(
        get_warp_idx_x() * NUM_FRAGS_X * 16 + lane_idx % 16, lane_idx / 16);

    load_q_global_smem(qo_packed_idx_base, qo_upper_bound, q_ptr_base, qo_n_stride, qo_h_stride,
                       group_size, &qo_smem);

    cp_async::commit_group();
    cp_async::wait_group<0>();
    block.sync();

    if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      if (get_warp_idx_z() == 0) {
        if (q_offset == nullptr) {
          q_smem_inplace_apply_rotary_multiply_sm_scale(qo_packed_idx_base, qo_len, kv_len,
                                                        group_size, &qo_smem, &q_smem_offset_r,
                                                        rope_freq, sm_scale);
        } else {
          q_smem_inplace_apply_rotary_with_pos_multiply_sm_scale<IdType>(
              qo_packed_idx_base, q_offset + qo_indptr[request_idx], &qo_smem, group_size,
              &q_smem_offset_r, rope_freq, sm_scale);
        }
      }
    } else {
      q_smem_inplace_multiply_sm_scale(&qo_smem, sm_scale);
    }

    if constexpr (POS_ENCODING_MODE == PosEncodingMode::kALiBi) {
#pragma unroll
      for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          const uint32_t qo_head_idx =
              kv_head_idx * group_size + (qo_packed_idx_base + get_warp_idx_x() * NUM_WARPS_X * 16 +
                                          lane_idx / 4 + j * 8 + fx * 16) %
                                             group_size;
          alibi_slopes[fx][j] = get_alibi_slope(qo_head_idx, num_qo_heads) * math::log2e;
        }
      }
    }

    smem_t k_smem(smem + (NUM_WARPS_X * NUM_FRAGS_X) * 16 * HEAD_DIM * sizeof(DTypeIn)),
        v_smem(smem + (NUM_WARPS_X * NUM_FRAGS_X + NUM_WARPS_Z * NUM_FRAGS_Z) * 16 * HEAD_DIM *
                          sizeof(DTypeIn));

    uint32_t k_smem_offset_r = smem_t::get_permuted_offset<CHANNEL_SIZE_128B_IN>(
                 get_warp_idx_z() * NUM_FRAGS_Z * 16 + 8 * (lane_idx / 16) + lane_idx % 8,
                 (lane_idx % 16) / 8),
             v_smem_offset_r = smem_t::get_permuted_offset<CHANNEL_SIZE_128B_IN>(
                 get_warp_idx_z() * NUM_FRAGS_Z * 16 + lane_idx % 16, lane_idx / 16),
             kv_smem_offset_w = smem_t::get_permuted_offset<CHANNEL_SIZE_128B_IN>(
                 warp_idx * 4 + lane_idx / 8, lane_idx % 8);
    const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

    uint32_t page_iter_base = paged_kv.indptr[request_idx];
    page_produce_kv<false, PAGE_SIZE>(k_smem, &kv_smem_offset_w, paged_kv, 0, page_iter_base,
                                      kv_len, last_indptr);
    cp_async::commit_group();
    page_produce_kv<true, PAGE_SIZE>(v_smem, &kv_smem_offset_w, paged_kv, 0, page_iter_base, kv_len,
                                     last_indptr);
    cp_async::commit_group();

    const uint32_t num_iterations = ceil_div(
        (MASK_MODE == MaskMode::kCausal
             ? min(kv_len, kv_len - qo_len + ((tile_idx + 1) * NUM_ROWS_PER_CTA) / group_size)
             : kv_len),
        16 * NUM_WARPS_Z * NUM_FRAGS_Z);

    const uint32_t mask_iteration =
        (MASK_MODE == MaskMode::kCausal
             ? min(kv_len + (tile_idx * NUM_ROWS_PER_CTA) / group_size - qo_len, kv_len)
             : kv_len) /
        (16 * NUM_WARPS_Z * NUM_FRAGS_Z);

#pragma unroll
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
      cp_async::wait_group<1>();
      block.sync();

      if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
        k_smem_inplace_apply_rotary(
            (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[request_idx]) +
                iter * NUM_COLS_PER_ITER,
            &k_smem, &k_smem_offset_r, rope_freq);
        block.sync();
      }

      // compute attention score
      compute_qk(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r, s_frag);

      if constexpr (POS_ENCODING_MODE == PosEncodingMode::kALiBi) {
        // TODO(Zihao): handle the case that q_offset is specified
        apply_alibi_bias(qo_packed_idx_base + get_warp_idx_x() * NUM_FRAGS_X * 16,
                         iter * NUM_COLS_PER_ITER + get_warp_idx_z() * NUM_FRAGS_Z * 16,
                         int(kv_len) - int(qo_len), group_size, alibi_slopes, s_frag);
      }
      // apply mask
      if constexpr (MASK_MODE == MaskMode::kCustom) {
        mask_s<PARTITION_KV>(qo_packed_idx_base + get_warp_idx_x() * NUM_FRAGS_X * 16,
                             iter * NUM_COLS_PER_ITER + get_warp_idx_z() * NUM_FRAGS_Z * 16, qo_len,
                             kv_len, kv_len, group_size, custom_mask + qk_indptr[request_idx],
                             s_frag);
      } else {
        if (iter >= mask_iteration) {
          mask_s<PARTITION_KV>(qo_packed_idx_base + get_warp_idx_x() * NUM_FRAGS_X * 16,
                               iter * NUM_COLS_PER_ITER + get_warp_idx_z() * NUM_FRAGS_Z * 16,
                               qo_len, kv_len, kv_len, group_size, nullptr, s_frag);
        }
      }

      // compute m,d states in online softmax
      update_mdo_states(s_frag, o_frag, m, d);

      block.sync();
      page_iter_base += 16 * NUM_WARPS_Z * NUM_FRAGS_Z / PAGE_SIZE;
      page_produce_kv<false, PAGE_SIZE>(k_smem, &kv_smem_offset_w, paged_kv,
                                        (iter + 1) * NUM_COLS_PER_ITER, page_iter_base, kv_len,
                                        last_indptr);
      cp_async::commit_group();
      cp_async::wait_group<1>();
      block.sync();

      // compute sfm*v
      compute_sfm_v(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);

      block.sync();
      page_produce_kv<true, PAGE_SIZE>(v_smem, &kv_smem_offset_w, paged_kv,
                                       (iter + 1) * NUM_COLS_PER_ITER, page_iter_base, kv_len,
                                       last_indptr);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    block.sync();

    // threadblock synchronization
    threadblock_sync_mdo_states(o_frag, (float*)smem, m, d, warp_idx, lane_idx);

    // normalize d
    normalize_d(o_frag, d);

    // write_back
    write_o_reg_gmem(o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len, qo_n_stride,
                     qo_h_stride, group_size);

    // write lse
    if (lse != nullptr) {
#pragma unroll
      for (uint32_t fx = 0; fx < NUM_FRAGS_X; ++fx) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          uint32_t q, r;
          group_size.divmod(qo_packed_idx_base + get_warp_idx_x() * NUM_FRAGS_X * 16 +
                                lane_idx / 4 + j * 8 + fx * 16,
                            q, r);
          const uint32_t qo_idx = q, qo_head_idx = kv_head_idx * group_size + r;
          if (qo_idx < qo_upper_bound) {
            lse[(qo_indptr[request_idx] + qo_idx) * num_qo_heads + qo_head_idx] =
                math::ptx_log2(d[fx][j]) + float(m[fx][j]);
          }
        }
      }
    }
  }
};

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
          QKVLayout kv_layout, PosEncodingMode pos_encoding_mode, uint32_t num_warps_x,
          uint32_t num_warps_z, uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z,
          typename DTypeIn, typename DTypeQKAccum, typename DTypeOut>
__global__ void SinglePrefillWithKVCacheKernel(DTypeIn* __restrict__ q, DTypeIn* __restrict__ k,
                                               DTypeIn* __restrict__ v,
                                               float* __restrict__ custom_mask,
                                               DTypeOut* __restrict__ o, void* __restrict__ tmp,
                                               float* __restrict__ lse, const uint32_t qo_len,
                                               const uint32_t kv_len, const uint_fastdiv group_size,
                                               float sm_scale, const float log2_rope_rcp_scale,
                                               const float log2_rope_rcp_theta) {
  using mod_t = AttentionModules<logits_post_hook, mask_mode, kv_layout, pos_encoding_mode,
                                 num_warps_x, num_warps_z, num_frags_x, num_frags_y, num_frags_z,
                                 DTypeIn, DTypeQKAccum, DTypeOut>;
  mod_t::template single_prefill_with_kv_cache_body<partition_kv>(
      q, k, v, custom_mask, o, tmp, lse, qo_len, kv_len, group_size, sm_scale, log2_rope_rcp_scale,
      log2_rope_rcp_theta);
}

template <LogitsPostHook logits_post_hook, MaskMode mask_mode, QKVLayout kv_layout,
          PosEncodingMode pos_encoding_mode, uint32_t num_warps_x, uint32_t num_warps_z,
          uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z, typename DTypeIn,
          typename DTypeQKAccum, typename DTypeOut, typename IdType>
__global__ void BatchPrefillWithRaggedKVCacheKernel(
    DTypeIn* __restrict__ q, IdType* __restrict__ request_indices,
    IdType* __restrict__ tile_indices, IdType* __restrict__ qo_indptr, DTypeIn* __restrict__ k,
    DTypeIn* __restrict__ v, IdType* __restrict__ kv_indptr, float* __restrict__ custom_mask,
    IdType* __restrict__ qk_indptr, IdType* __restrict__ q_offset,
    IdType* __restrict__ k_rope_pos_offset, DTypeOut* __restrict__ o, float* __restrict__ tmp,
    float* __restrict__ lse, uint32_t batch_size, const uint_fastdiv group_size, float sm_scale,
    float log2_rope_rcp_scale, float log2_rope_rcp_theta) {
  using mod_t = AttentionModules<logits_post_hook, mask_mode, kv_layout, pos_encoding_mode,
                                 num_warps_x, num_warps_z, num_frags_x, num_frags_y, num_frags_z,
                                 DTypeIn, DTypeQKAccum, DTypeOut>;
  mod_t::template batch_prefill_with_ragged_kv_cache_body<IdType>(
      q, request_indices, tile_indices, qo_indptr, k, v, kv_indptr, custom_mask, qk_indptr,
      q_offset, k_rope_pos_offset, o, tmp, lse, batch_size, group_size, sm_scale,
      log2_rope_rcp_scale, log2_rope_rcp_theta);
}

template <LogitsPostHook logits_post_hook, uint32_t page_size, MaskMode mask_mode,
          PosEncodingMode pos_encoding_mode, uint32_t num_warps_x, uint32_t num_warps_z,
          uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z,
          PageStorage page_storage, QKVLayout kv_layout, typename DTypeIn, typename DTypeQKAccum,
          typename DTypeOut, typename IdType>
__global__ void BatchPrefillWithPagedKVCacheKernel(
    IdType* __restrict__ request_indices, IdType* __restrict__ tile_indices,
    DTypeIn* __restrict__ q, paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv,
    IdType* __restrict__ qo_indptr, float* __restrict__ custom_mask, IdType* __restrict__ qk_indptr,
    IdType* __restrict__ q_offset, DTypeOut* __restrict__ o, float* __restrict__ tmp,
    float* __restrict__ lse, const uint_fastdiv group_size, float sm_scale,
    float log2_rope_rcp_scale, float log2_rope_rcp_theta) {
  using mod_t = AttentionModules<logits_post_hook, mask_mode, kv_layout, pos_encoding_mode,
                                 num_warps_x, num_warps_z, num_frags_x, num_frags_y, num_frags_z,
                                 DTypeIn, DTypeQKAccum, DTypeOut>;
  mod_t::template batch_prefill_with_paged_kv_cache_body<page_size, page_storage, IdType>(
      request_indices, tile_indices, q, paged_kv, qo_indptr, custom_mask, qk_indptr, q_offset, o,
      tmp, lse, group_size, sm_scale, log2_rope_rcp_scale, log2_rope_rcp_theta);
}

template <uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK, QKVLayout KV_LAYOUT,
          PosEncodingMode pos_encoding_mode, bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename DTypeIn, typename DTypeOut>
cudaError_t SinglePrefillWithKVCacheDispatched(DTypeIn* q, DTypeIn* k, DTypeIn* v,
                                               float* custom_mask, DTypeOut* o, float* tmp,
                                               float* lse, uint32_t num_qo_heads,
                                               uint32_t num_kv_heads, uint32_t qo_len,
                                               uint32_t kv_len, float sm_scale, float rope_scale,
                                               float rope_theta, cudaStream_t stream) {
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
  DISPATCH_NUM_FRAGS_X((qo_len * group_size > 64 && HEAD_DIM < 256 ? 2 : 1), num_frags_x, {
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

    constexpr uint32_t num_warps_x = 1;
    constexpr uint32_t num_warps_z = 4;
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
      using ModT = AttentionModules<LOGITS_POST_HOOK, MASK_MODE, KV_LAYOUT, pos_encoding_mode,
                                    num_warps_x, num_warps_z, num_frags_x, num_frags_y, num_frags_z,
                                    DTypeIn, DTypeQKAccum, DTypeOut>;
      if constexpr (ModT::IS_INVALID_CONFIGURATION) {
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
                                           KV_LAYOUT, pos_encoding_mode, num_warps_x, num_warps_z,
                                           num_frags_x, num_frags_y, num_frags_z, DTypeIn,
                                           DTypeQKAccum, DTypeOut>;
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
                                             KV_LAYOUT, pos_encoding_mode, num_warps_x, num_warps_z,
                                             num_frags_x, num_frags_y, num_frags_z, DTypeIn,
                                             DTypeQKAccum, DTypeOut>;
          void* args[] = {(void*)&q,
                          (void*)&k,
                          (void*)&v,
                          (void*)&custom_mask,
                          (void*)&o,
                          (void*)&tmp,
                          (void*)&lse,
                          (void*)&qo_len,
                          (void*)&kv_len,
                          (void*)&group_size_fastdiv,
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
          void* args[] = {(void*)&q,
                          (void*)&k,
                          (void*)&v,
                          (void*)&custom_mask,
                          (void*)&o,
                          (void*)&tmp,
                          (void*)&lse,
                          (void*)&qo_len,
                          (void*)&kv_len,
                          (void*)&group_size_fastdiv,
                          (void*)&sm_scale,
                          (void*)&log2_rope_rcp_scale,
                          (void*)&log2_rope_rcp_theta};
          dim3 nblks(ceil_div(qo_len * group_size, num_rows_per_cta), num_chunks, num_kv_heads);
          dim3 nthrs(32, num_warps_x, num_warps_z);
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)partition_kv_kernel, nblks, nthrs, args, smem_size, stream));
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

template <uint32_t num_frags_x, uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK,
          QKVLayout KV_LAYOUT, PosEncodingMode pos_encoding_mode, bool ALLOW_FP16_QK_REDUCTION,
          MaskMode MASK_MODE, typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(
    DTypeIn* q, IdType* request_indices, IdType* tile_indices, IdType* qo_indptr, DTypeIn* k,
    DTypeIn* v, IdType* kv_indptr, float* custom_mask, IdType* qk_indptr, IdType* q_offset,
    IdType* k_rope_pos_offset, DTypeOut* o, float* tmp, float* lse, const uint32_t batch_size,
    const uint32_t num_qo_heads, const uint32_t num_qo_tiles, const uint32_t num_kv_heads,
    const float sm_scale, const float rope_scale, const float rope_theta,
    cudaStream_t stream = nullptr) {
  const float log2_rope_rcp_scale = -std::log2f(rope_scale);
  const float log2_rope_rcp_theta = -std::log2f(rope_theta);
  constexpr uint32_t num_warps_x = 1;
  constexpr uint32_t num_warps_z = 4;
  const uint32_t group_size = num_qo_heads / num_kv_heads;
  const uint_fastdiv group_size_fastdiv(group_size);

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
    using ModT = AttentionModules<LOGITS_POST_HOOK, MASK_MODE, KV_LAYOUT, pos_encoding_mode,
                                  num_warps_x, num_warps_z, num_frags_x, num_frags_y, num_frags_z,
                                  DTypeIn, DTypeQKAccum, DTypeOut>;
    if constexpr (ModT::IS_INVALID_CONFIGURATION) {
      // Invalid configuration, skip
      std::ostringstream err_msg;
      err_msg << "FlashInfer Internal Error: Invalid configuration : num_frags_x=" << num_frags_x
              << " num_frags_y=" << num_frags_y << " num_frags_z=" << num_frags_z
              << " num_warps_x=" << num_warps_x << " num_warps_z=" << num_warps_z
              << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                 " and report the issue to the developers.";
      throw std::invalid_argument(err_msg.str());
    } else {
      auto kernel = BatchPrefillWithRaggedKVCacheKernel<
          LOGITS_POST_HOOK, MASK_MODE, KV_LAYOUT, pos_encoding_mode, num_warps_x, num_warps_z,
          num_frags_x, num_frags_y, num_frags_z, DTypeIn, DTypeQKAccum, DTypeOut, IdType>;
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
                      (void*)&custom_mask,
                      (void*)&qk_indptr,
                      (void*)&q_offset,
                      (void*)&k_rope_pos_offset,
                      (void*)&o,
                      (void*)&tmp,
                      (void*)&lse,
                      (void*)&batch_size,
                      (void*)&group_size_fastdiv,
                      (void*)&sm_scale,
                      (void*)&log2_rope_rcp_scale,
                      (void*)&log2_rope_rcp_theta};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    }
  });
  return cudaSuccess;
}

template <PageStorage page_storage, uint32_t num_frags_x, uint32_t PAGE_SIZE, uint32_t HEAD_DIM,
          LogitsPostHook LOGITS_POST_HOOK, QKVLayout kv_layout, PosEncodingMode pos_encoding_mode,
          bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename DTypeIn, typename DTypeOut,
          typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(
    DTypeIn* q, IdType* request_indices, IdType* tile_indices, IdType* qo_indptr, IdType* q_offset,
    paged_kv_t<page_storage, kv_layout, DTypeIn, IdType> paged_kv, float* custom_mask,
    IdType* qk_indptr, DTypeOut* o, float* tmp, float* lse, uint32_t num_qo_heads,
    uint32_t num_qo_tiles, float sm_scale, float rope_scale, float rope_theta,
    cudaStream_t stream) {
  const float log2_rope_rcp_scale = -std::log2f(rope_scale);
  const float log2_rope_rcp_theta = -std::log2f(rope_theta);
  constexpr uint32_t num_warps_x = 1;
  constexpr uint32_t num_warps_z = 4;
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t batch_size = paged_kv.batch_size;
  const uint32_t group_size = num_qo_heads / num_kv_heads;
  const uint_fastdiv group_size_fastdiv(group_size);

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
    using ModT = AttentionModules<LOGITS_POST_HOOK, MASK_MODE, kv_layout, pos_encoding_mode,
                                  num_warps_x, num_warps_z, num_frags_x, num_frags_y, num_frags_z,
                                  DTypeIn, DTypeQKAccum, DTypeOut>;
    if constexpr (ModT::IS_INVALID_CONFIGURATION) {
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
          BatchPrefillWithPagedKVCacheKernel<LOGITS_POST_HOOK, PAGE_SIZE, MASK_MODE,
                                             pos_encoding_mode, num_warps_x, num_warps_z,
                                             num_frags_x, num_frags_y, num_frags_z, page_storage,
                                             kv_layout, DTypeIn, DTypeQKAccum, DTypeOut, IdType>;
      uint32_t smem_size = (num_frags_x * num_warps_x + num_frags_z * num_warps_z * 2) * 16 *
                           HEAD_DIM * sizeof(DTypeIn);
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      void* args[] = {(void*)&request_indices,
                      (void*)&tile_indices,
                      (void*)&q,
                      (void*)&paged_kv,
                      (void*)&qo_indptr,
                      (void*)&custom_mask,
                      (void*)&qk_indptr,
                      (void*)&q_offset,
                      (void*)&o,
                      (void*)&tmp,
                      (void*)&lse,
                      (void*)&group_size_fastdiv,
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
