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

#include <type_traits>

#ifdef USE_ROCM

#include <hip/hip_cooperative_groups.h>

#include "flashinfer/hip_cuda_type_utils.h"
// CUDA API Portable interfaces
#include "flashinfer/hip_defs.h"

#include <rocwmma/rocwmma.hpp>

#include <iostream>

// device print
#include <cstdio>

#else

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#endif // USE_ROCM

#include "../cp_async.cuh"
#include "../fastdiv.cuh"
#include "../frag_layout_swizzle.cuh"
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

#ifdef USE_ROCM
// TODO (yiakwy) : use AMD constants
constexpr uint32_t warp_size = 64;
#else
constexpr uint32_t warp_size = 32;
#endif

namespace {

template <PosEncodingMode pos_encoding_mode, typename DTypeKV, typename DTypeQKAccum>
constexpr bool is_invalid_configuration(uint32_t num_frags_x, uint32_t num_frags_y,
                                        uint32_t num_frags_z, uint32_t num_warps_x,
                                        uint32_t num_warps_z) {
  return ((num_frags_y < 4) || (num_frags_y == 4 && num_frags_z % 2 == 1) ||
          (num_frags_y > 4 && num_frags_y % (2 * num_warps_x) != 0) ||
          (num_frags_x * (8 * num_frags_y + 2 * sizeof(DTypeQKAccum) * num_frags_z) >= 256) ||
          (sizeof(DTypeKV) == 1 && num_frags_z * 2 % num_warps_x != 0) ||
          (sizeof(DTypeKV) == 1 && pos_encoding_mode == PosEncodingMode::kRoPELlama));
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
  static_assert(sizeof(T) == 2);
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
 * \tparam T The data type of the input tensor.
 * \param smem The shared memory to store kv fragments.
 * \param gptr The global memory pointer.
 * \param qkv_info The tensor info of the input tensor.
 * \param kv_idx_base The base kv index.
 * \param kv_len The length of kv tensor.
 */
template <SharedMemFillMode fill_mode, uint32_t num_warps_x, uint32_t num_warps_z,
          uint32_t num_frags_y, uint32_t num_frags_z, SwizzleMode swizzle_mode, typename T>
__device__ __forceinline__ void produce_kv(smem_t<swizzle_mode> smem, uint32_t* smem_offset,
                                           T** gptr, const uint32_t kv_stride_n,
                                           const uint32_t kv_idx_base, const uint32_t kv_len) {
  // NOTE(Zihao): for fp8, this function doesn't work for head_dim = 64 at the moment
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_warps = num_warps_x * num_warps_z;
  constexpr uint32_t channel_size_128b_kv = head_dim / num_elems_per_128b<T>();
  const uint32_t warp_idx = get_warp_idx<num_warps_x, num_warps_z>(), lane_idx = threadIdx.x;

  if constexpr (swizzle_mode == SwizzleMode::k128B) {
    // NOTE(yiakwy) : each thread of a 32 threads block, cooperatively load 128 bit (uint4/float4/halfx8) data from system memory to shared memory
    // kvsmem shape =(num_frags_z x 16, (num_frags_y / 4) * 64)
    // -- num_frags_y -->
    // kvsmem warps row/col 0               1                     ... 7
    //                      0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 ... 60 61 62 63
    //        0     0
    //              1
    //              2
    //              3
    //        1     0+4*1
    //        ..    ..
    //        3     0+4*3
    //              1+4*3
    //              2+4*3
    //              3+4*3
    //
    uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / 8;
    // NOTE(Zihao): num_frags_z * 4 / num_warps_x = num_warps_z * num_frags_z * 4 / num_warps
    static_assert(num_frags_z * 4 % num_warps_x == 0);

    // NOTE (yiakwy) : for kv = (1/*head*/, 16/*seq*/, 64), at least 128 rows will be loaded
#pragma unroll
    for (uint32_t i = 0; i < num_frags_z * 4 / num_warps_x && ((warp_idx * 4 + lane_idx / 8 + i * 16) < kv_len); ++i) { // for (uint32_t i = 0; i < num_frags_z * 4 / num_warps_x; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < num_frags_y / (8 / sizeof(T)); ++j) {

        // NOTE (yiakwy) : kvsmem[warp_idx * 4 + lane_idx / 8 + i * 16, lane_idx % 8 + j * 8] = kv[0, warp_idx * 4 + lane_idx / 8 + i * 16, lane_idx % 8 + j * 8]
        smem.template load_128b_async<fill_mode>(*smem_offset, *gptr, kv_idx < kv_len);

        T* kv_base_r = *gptr;
        T* kv_ptr = kv_base_r + (/* warp_idx * 4 + lane_idx / 8 */ + i * 16) * kv_stride_n + /* lane_idx % 8 */ + j * 8;

        b128_t* smem_ptr = smem.base + (/* warp_idx * 4 + lane_idx / 8 */ + i * 16) * 8 + /* lane_idx % 8 */ + j * 8;
        float16_t *s = reinterpret_cast<float16_t *>(smem_ptr);
        printf("[produce_kv] (i=%d,j=%d,warp_idx=%d), kv_smem[%d, %d] (%f..%f) = kv[H=0, N_CTX=%d/%d, %d](%f..%f,%f)\n", i, j, warp_idx, warp_idx * 4 + lane_idx / 8 + i * 16, lane_idx % 8 + j * 8, (float)(*s), (float)(*(s+7)), warp_idx * 4 + lane_idx / 8 + i * 16, kv_len, lane_idx % 8 + j * 8, (float)(*kv_ptr), (float)(*(kv_ptr+6)), (float)(*(kv_ptr+7)));

        *smem_offset = smem.template advance_offset_by_column<8>(*smem_offset, j);
        *gptr += 8 * num_elems_per_128b<T>();
      }
      kv_idx += num_warps * 4;

      // NOTE (yiakwy) : reset columns offset, ahead to next 16 rows
      *smem_offset =
          smem.template advance_offset_by_row<num_warps * 4, channel_size_128b_kv>(*smem_offset) -
          sizeof(T) * num_frags_y;
      // NOTE (yiakwy) : reset columns offset, ahead to next 16 rows
      *gptr += num_warps * 4 * kv_stride_n - sizeof(T) * num_frags_y * num_elems_per_128b<T>();
    }
    // NOTE (yiakwy) : reset kv smem pointer
    *smem_offset -= num_warps_z * num_frags_z * 16 * channel_size_128b_kv;
  } else {
    uint32_t kv_idx = kv_idx_base + warp_idx * 8 + lane_idx / 4;
    // NOTE(Zihao): num_frags_z * 2 / num_warps_x = num_warps_z * num_frags_z * 2 / num_warps
    static_assert(num_frags_z * 2 % num_warps_x == 0);
#pragma unroll
    for (uint32_t i = 0; i < num_frags_z * 2 / num_warps_x; ++i) {
      smem.template load_128b_async<fill_mode>(*smem_offset, *gptr, kv_idx < kv_len);
      *smem_offset =
          smem.template advance_offset_by_row<num_warps * 8, channel_size_128b_kv>(*smem_offset);
      kv_idx += num_warps * 8;
      *gptr += num_warps * 8 * kv_stride_n;
    }
    *smem_offset -= num_warps_z * num_frags_z * 16 * channel_size_128b_kv;
  }
}

template <bool produce_v, uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_y,
          uint32_t num_frags_z, PageStorage page_storage, SwizzleMode swizzle_mode, typename DType,
          typename IdType>
__device__ __forceinline__ void page_produce_kv(smem_t<swizzle_mode> smem, uint32_t* smem_offset,
                                                paged_kv_t<page_storage, DType, IdType>& paged_kv,
                                                const uint32_t kv_idx_base, const size_t* kv_offset,
                                                const uint32_t kv_len) {
  // NOTE(Zihao): for fp8, this function doesn't work for head_dim = 64 at the moment
  constexpr SharedMemFillMode fill_mode =
      produce_v ? SharedMemFillMode::kFillZero : SharedMemFillMode::kNoFill;
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_warps = num_warps_x * num_warps_z;
  constexpr uint32_t channel_size_128b_kv = head_dim / num_elems_per_128b<DType>();
  const uint32_t warp_idx = get_warp_idx<num_warps_x, num_warps_z>(), lane_idx = threadIdx.x;
  if constexpr (swizzle_mode == SwizzleMode::k128B) {
    uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / 8;
    // NOTE(Zihao): num_frags_z * 4 / num_warps_x = num_warps_z * num_frags_z * 4 / num_warps
    static_assert(num_frags_z * 4 % num_warps_x == 0);
#pragma unroll
    for (uint32_t i = 0; i < num_frags_z * 4 / num_warps_x; ++i) {
      DType* gptr = produce_v ? paged_kv.v_data + kv_offset[i] : paged_kv.k_data + kv_offset[i];
#pragma unroll
      for (uint32_t j = 0; j < num_frags_y / (8 / sizeof(DType)); ++j) {
        smem.template load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
        *smem_offset = smem.template advance_offset_by_column<8>(*smem_offset, j);
        gptr += 8 * num_elems_per_128b<DType>();
      }
      kv_idx += num_warps * 4;
      *smem_offset =
          smem.template advance_offset_by_row<num_warps * 4, channel_size_128b_kv>(*smem_offset) -
          sizeof(DType) * num_frags_y;
    }
    *smem_offset -= num_warps_z * num_frags_z * 16 * channel_size_128b_kv;
  } else {
    uint32_t kv_idx = kv_idx_base + warp_idx * 8 + lane_idx / 4;
    // NOTE(Zihao): num_frags_z * 2 / num_warps_x = num_warps_z * num_frags_z * 2 / num_warps
    static_assert(num_frags_z * 2 % num_warps_x == 0);
#pragma unroll
    for (uint32_t i = 0; i < num_frags_z * 2 / num_warps_x; ++i) {
      DType* gptr = produce_v ? paged_kv.v_data + kv_offset[i] : paged_kv.k_data + kv_offset[i];
      smem.template load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
      kv_idx += num_warps * 8;
      *smem_offset =
          smem.template advance_offset_by_row<num_warps * 8, channel_size_128b_kv>(*smem_offset);
    }
    *smem_offset -= num_warps_z * num_frags_z * 16 * channel_size_128b_kv;
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

template <uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_x, uint32_t num_frags_y,
          SwizzleMode swizzle_mode, typename DTypeQ>
__device__ __forceinline__ void load_q_global_smem(uint32_t packed_offset,
                                                   const uint32_t qo_upper_bound,
                                                   DTypeQ* q_ptr_base, const uint32_t q_stride_n,
                                                   const uint32_t q_stride_h,
                                                   const uint_fastdiv group_size,
                                                   smem_t<swizzle_mode>* q_smem) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_q = head_dim / num_elems_per_128b<DTypeQ>();
  const uint32_t lane_idx = threadIdx.x, warp_idx_x = get_warp_idx_x<num_warps_x, num_warps_z>();

  if (get_warp_idx_z<num_warps_x, num_warps_z>() == 0) {

    // TODO (yiakwy) : only half a warp concurrency if blockDim.x == 32 in ROCm platform

    // NOTE(yiakwy) : each thread of a 32 threads block, cooperatively load 128 bit (uint4/float4/halfx8) data from system memory to shared memory
    // qsmem shape = (_, (num_frags_y / 4) * 64 /*hidden_size*/)
    // -- frags y ->
    // qsmem row/col 0                       1                       ... 7               warp_idx {0..3}  
    //       0       0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 ... 60  61  62  63  0                |
    //       1       64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 ... 124 125 126 127 0                |
    //       2       .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  ... .   .   .   .   0               frags x
    //       3       .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  ... .   .   .   .   0                |
    //       ...     .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  ... .   .   .   .   0                |
    //       0+4*3   .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  ... .   .   .   .   0                v
    //       1+4*3   .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  ... .   .   .   .   0
    //       2+4*3   .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  ... .   .   .   .   0
    //       3+4*3   .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  ... .   .   .   .   0
    //  qsmem is (num_frags_x x 16) x 64 (128 bit) matrix fragment            
    uint32_t q_smem_offset_w = q_smem->template get_permuted_offset<channel_size_128b_q>(
        warp_idx_x * num_frags_x * 16 + lane_idx / 8, lane_idx % 8);

#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        uint32_t q, r;
        group_size.divmod(packed_offset + lane_idx / 8 + fx * 16 + j * 4, q, r);
        const uint32_t q_idx = q;

        // NOTE (yiakwy) : q_ptr = q[bz/*head*/, bx{0} * num_rows_per_cta{16} + warp_idx_x * num_frags_x * 16 + lane_idx / 8 + j * 4 /*seqlen*/, 0/*hdim*/] + (lane_idx % 8) * 8
        DTypeQ* q_ptr = q_ptr_base + q * q_stride_n + r * q_stride_h;

#pragma unroll
        for (uint32_t fyo = 0; fyo < num_frags_y / 4; ++fyo) {
          // load q fragment from gmem to smem
          // NOTE (yiakwy) : qsmem[warp_idx_x * num_frags_x * 16 + lane_idx / 8 + j * 4, lane_idx % 8] = q[bz/*head*/, warp_id_x * 16 + lane_idx / 8 + j * 4/*seqlen*/, 0/*hdim*/] + (lane_idx % 8) * 8
          if (qo_upper_bound >= 16) {
            q_smem->template load_128b_async<SharedMemFillMode::kFillZero>(q_smem_offset_w, q_ptr, q_idx < qo_upper_bound);
          } else {
            q_smem->template load_128b_async<SharedMemFillMode::kNoFill>(q_smem_offset_w, q_ptr, q_idx < qo_upper_bound);
          }

          // #ifdef DEBUG
          b128_t* smem_ptr = q_smem->base + (lane_idx / 8 + fx * 16 + j * 4 ) * 8 + lane_idx % 8;
          float16_t *s = reinterpret_cast<float16_t *>(smem_ptr);
          printf("[load q from global] (x=%d,z=%d,j=%d), q_smem[%d, %d](%f..%f) = q[H=%d,N_CTX=%d, %d](%f..%f)\n", threadIdx.x, threadIdx.z, j, lane_idx / 8 + j * 4, lane_idx % 8, (float)(*(s)), (float)(*(s+7)), 0, lane_idx / 8 + j * 4, (lane_idx % 8) * 8, (float)q_ptr[0], (float)q_ptr[7]);
          // #endif

          q_smem_offset_w = q_smem->template advance_offset_by_column<8>(q_smem_offset_w, fyo);

          // NOTE(yiakwy) : no need to increment at the last iteration
          if (fyo + 1 < num_frags_y / 4) {
            q_ptr += 8 * num_elems_per_128b<DTypeQ>();
          }
        }

        // TODO (yiakwy) : rewrite
        q_smem_offset_w =
            q_smem->template advance_offset_by_row<4, channel_size_128b_q>(q_smem_offset_w) -
            2 * num_frags_y;
      }
    }
  }
}

template <uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_x, uint32_t num_frags_y,
          SwizzleMode swizzle_mode, typename DTypeQ>
__device__ __forceinline__ void q_smem_inplace_apply_rotary_multiply_sm_scale(
    const uint32_t q_packed_idx, const uint32_t qo_len, const uint32_t kv_len,
    const uint_fastdiv group_size, smem_t<swizzle_mode>* q_smem, uint32_t* q_smem_offset_r,
    float (*rope_freq)[4], const float sm_scale) {
  if (get_warp_idx_z<num_warps_x, num_warps_z>() == 0) {
    constexpr uint32_t head_dim = num_frags_y * 16;
    constexpr uint32_t channel_size_128b_q = head_dim / num_elems_per_128b<DTypeQ>();
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
            q_smem->template advance_offset_by_column<num_frags_y>(q_smem_offset_r_first_half, 0);
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_frag_apply_llama_rope<DTypeQ>(
            (DTypeQ*)q_frag_local[0], (DTypeQ*)q_frag_local[1], rope_freq[fyi],
            q_packed_idx + kv_len * group_size - qo_len * group_size + fx * 16 + lane_idx / 4,
            group_size, sm_scale);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        q_smem_offset_r_first_half =
            q_smem->template advance_offset_by_column<2>(q_smem_offset_r_first_half, fyi);
      }
      *q_smem_offset_r += 16 * channel_size_128b_q;
    }
    *q_smem_offset_r -= num_frags_x * 16 * channel_size_128b_q;
  }
}

template <uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_x, uint32_t num_frags_y,
          SwizzleMode swizzle_mode, typename DTypeQ, typename IdType>
__device__ __forceinline__ void q_smem_inplace_apply_rotary_with_pos_multiply_sm_scale(
    const uint32_t q_packed_idx_base, const IdType* q_offset, smem_t<swizzle_mode>* q_smem,
    const uint_fastdiv group_size, uint32_t* q_smem_offset_r, float (*rope_freq)[4],
    const float sm_scale) {
  if (get_warp_idx_z<num_warps_x, num_warps_z>() == 0) {
    constexpr uint32_t head_dim = num_frags_y * 16;
    constexpr uint32_t channel_size_128b_q = head_dim / num_elems_per_128b<DTypeQ>();
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
            q_smem->template advance_offset_by_column<num_frags_y>(q_smem_offset_r_first_half, 0);
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_frag_apply_llama_rope_with_pos<DTypeQ>(
            (DTypeQ*)q_frag_local[0], (DTypeQ*)q_frag_local[1], rope_freq[fyi],
            q_packed_idx_base + fx * 16 + lane_idx / 4, group_size, q_offset, sm_scale);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        q_smem_offset_r_first_half =
            q_smem->template advance_offset_by_column<2>(q_smem_offset_r_first_half, fyi);
      }
      *q_smem_offset_r += 16 * channel_size_128b_q;
    }
    *q_smem_offset_r -= num_frags_x * 16 * channel_size_128b_q;
  }
}

template <uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_x, uint32_t num_frags_y,
          SwizzleMode swizzle_mode, typename DTypeQ>
__device__ __forceinline__ void q_smem_inplace_multiply_sm_scale(smem_t<swizzle_mode>* q_smem,
                                                                 const float sm_scale) {
  const uint32_t warp_idx = get_warp_idx<num_warps_x, num_warps_z>(), lane_idx = threadIdx.x;
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_q = head_dim / num_elems_per_128b<DTypeQ>();
  constexpr uint32_t num_warps = num_warps_x * num_warps_z;
#pragma unroll
  for (uint32_t i = 0; i < num_frags_x * head_dim / (num_warps_z * 16); ++i) {
    vec_t<DTypeQ, 8> tmp;
    tmp.load((DTypeQ*)(q_smem->base) + (i * num_warps + warp_idx) * 256 + lane_idx * 8);
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
      tmp[reg_id] *= sm_scale;
    }
    tmp.store((DTypeQ*)(q_smem->base) + (i * num_warps + warp_idx) * 256 + lane_idx * 8);
  }
}

template <uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_y, uint32_t num_frags_z,
          SwizzleMode swizzle_mode, typename DTypeKV>
__device__ __forceinline__ void k_smem_inplace_apply_rotary(const uint32_t kv_idx_base,
                                                            smem_t<swizzle_mode>* k_smem,
                                                            uint32_t* k_smem_offset_r,
                                                            float (*rope_freq)[4]) {
  static_assert(sizeof(DTypeKV) == 2);
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_kv = head_dim / num_elems_per_128b<DTypeKV>();
  uint32_t k_frag_local[2][4];
  const uint32_t lane_idx = threadIdx.x;
  if constexpr (num_frags_y == 4 && num_warps_x == 4) {
    static_assert(num_warps_z == 1);
    const uint32_t warp_idx = get_warp_idx_x<num_warps_x, num_warps_z>();
    // horizontal-axis: y
    // vertical-axis: z
    //         | 1-16       | 16-32      | 32-48      | 48-64      |
    // | 1-16  | warp_idx=0 | warp_idx=1 | warp_idx=0 | warp_idx=1 |
    // | 16-32 | warp_idx=2 | warp_idx=3 | warp_idx=2 | warp_idx=3 |
    static_assert(num_frags_z % 2 == 0,
                  "when num_frags_y == 4, num_frags_z must be a multiple of 2");
    uint32_t kv_idx = kv_idx_base + (warp_idx / 2) * 16 + lane_idx / 4;
    *k_smem_offset_r =
        (*k_smem_offset_r ^ (0x2 * (warp_idx % 2))) + (warp_idx / 2) * 16 * channel_size_128b_kv;
#pragma unroll
    for (uint32_t i = 0; i < num_frags_z / 2; ++i) {
      // uint32_t fz = warp_idx / 2 + i * 2;
      uint32_t k_smem_offset_r_first_half = *k_smem_offset_r;
      uint32_t fyi = (warp_idx % 2);
      k_smem->ldmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
      uint32_t k_smem_offset_r_last_half =
          k_smem->template advance_offset_by_column<4>(k_smem_offset_r_first_half, 0);
      k_smem->ldmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
      k_frag_apply_llama_rope<DTypeKV>((DTypeKV*)k_frag_local[0], (DTypeKV*)k_frag_local[1],
                                       rope_freq[fyi], kv_idx);
      k_smem->stmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
      k_smem->stmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
      *k_smem_offset_r += 32 * channel_size_128b_kv;
      kv_idx += 32;
    }
    *k_smem_offset_r = (*k_smem_offset_r ^ (0x2 * (warp_idx % 2))) -
                       ((warp_idx / 2) + num_frags_z) * 16 * channel_size_128b_kv;
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
            k_smem->template advance_offset_by_column<num_frags_y>(k_smem_offset_r_first_half, 0);
        k_smem->ldmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
        k_frag_apply_llama_rope<DTypeKV>((DTypeKV*)k_frag_local[0], (DTypeKV*)k_frag_local[1],
                                         rope_freq[fyi], kv_idx);
        k_smem->stmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
        k_smem->stmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
        k_smem_offset_r_first_half = k_smem->template advance_offset_by_column<2 * num_warps_x>(
            k_smem_offset_r_first_half, fyi);
      }
      *k_smem_offset_r += 16 * channel_size_128b_kv;
      kv_idx += 16;
    }
    *k_smem_offset_r =
        (*k_smem_offset_r ^ (0x2 * warp_idx_x)) - num_frags_z * 16 * channel_size_128b_kv;
  }
}

template <LogitsPostHook logits_post_hook, uint32_t num_frags_x, uint32_t num_frags_y,
          uint32_t num_frags_z, SwizzleMode swizzle_mode_q, SwizzleMode swizzle_mode_kv,
          typename DTypeQ, typename DTypeKV, typename DTypeQKAccum>
__device__ __forceinline__ void compute_qk(
    smem_t<swizzle_mode_q>* q_smem, uint32_t* q_smem_offset_r, smem_t<swizzle_mode_kv>* k_smem,
    uint32_t* k_smem_offset_r, DTypeQKAccum (*s_frag)[num_frags_z][8], const float soft_cap) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_q = head_dim / num_elems_per_128b<DTypeQ>();
  constexpr uint32_t channel_size_128b_kv = head_dim / num_elems_per_128b<DTypeKV>();

  #ifdef USE_ROCM

  // TODO (yiakwy) : REMOVE
  if (threadIdx.x == 0 && threadIdx.z == 0) {
    printf("[compute_qk] channel_size_128b_q=%d, channel_size_128b_kv=%d\n", channel_size_128b_q, channel_size_128b_kv);
    printf("[compute_qk] num_frags_x=%d, num_frags_y=%d, num_frags_z=%d\n", num_frags_x, num_frags_y, num_frags_z);
  }

  // NOTE(yiakwy) : each thread of 64=16x4 threads block, cooperatively loads 4 x consecutive fp16/bf16 data to cover 16x16 matrix frag
  uint32_t a_frag[num_frags_x][2];
  uint32_t b_frag[2];

  // hence 
  // TODO (yiakwy) : what if we change blckDim.x from 32 to 64
  uint32_t lane_id = threadIdx.x + threadIdx.z * blockDim.x;

  uint32_t lane_id_x = lane_id % 16;
  uint32_t lane_id_y = lane_id / 16;

  // TODO (yiakwy) : replace these variables later
  uint32_t warp_idx_x = get_warp_idx_x<1, 4>();
  uint32_t warp_idx_z = get_warp_idx_z<1, 4>();
  
  uint32_t warp_idx   = get_warp_idx<1, 4>();

  using float16_t = rocwmma::float16_t;

  using float16x4  = __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
  using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  
  #define MTX_FRAG_LDA (head_dim)
  // #define MTX_FRAG_LDB (head_dim)
  #define MTX_FRAG_LDB (num_frags_z * 16)

  #else
  
  // NOTE(yiakwy) : each thread of 32=8x4 threads block, cooperatively loads 2 x fp16/bf16 data, and repeat 4 (x4) times in 4 warps to cover 16x16 matrix frag
  uint32_t a_frag[num_frags_x][4], b_frag[4];

  #endif // USE_ROCM

  // compute q*k^T
  #ifdef USE_ROCM

  if (lane_id < 64U) {

#pragma unroll
  for (uint32_t fy = 0; fy < num_frags_y; ++fy) {

    // load q
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {

      // NOTE (yiakwy) : q_smem has shape of (num_frags_x, 16, 8x8), v_mfma_m16n16k16_fp16 will be applied 4 times along feat dim
      b128_t* smem_ptr = q_smem->base + (warp_idx_x * num_frags_x * 16) * channel_size_128b_q;
      float16_t *s = reinterpret_cast<float16_t *>(smem_ptr);
        
      float16x4 *a = reinterpret_cast<float16x4 *>(a_frag[fx]);

      // TODO (yiakwy) : replaced with more efficient load instruction
#pragma unroll
      for (uint32_t j=0; j < 4; j++) {
        // NOTE (yiakwy) : 16 threads loads 4 columns (16x4fp16) of data cooperatively
        uint32_t offset = lane_id_x * MTX_FRAG_LDA + j + lane_id_y * 4;
        s += offset;

        (*a)[j] = *(s);

        if (fx == 0 && fy == 0) {
        printf("[compute_qk] (x=%d, y=%d, j=%d) (fx=%d, fy=%d) a_mtx_frag[%d, %d]=%f\n", lane_id_x, lane_id_y, j, fx, fy, lane_id_x, j + lane_id_y * 4, (float)((*a)[j]));
        }
      }

      *q_smem_offset_r =
              q_smem->template advance_offset_by_row<16, channel_size_128b_q>(*q_smem_offset_r);
    } // num_frags_x
    
    // NOTE(yiakwy) : next to 16 = 2x8 columns
    *q_smem_offset_r = q_smem->template advance_offset_by_column<2>(*q_smem_offset_r, fy) -
                       num_frags_x * 16 * channel_size_128b_q;

    // load k
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {

      if constexpr (sizeof(DTypeKV) == 1) {
        assert(0 && "KV Cache with FP8 data type is not supported in ROCM");
      }

      b128_t* smem_ptr = k_smem->base + (warp_idx * 4) * channel_size_128b_q;
      float16_t *s = reinterpret_cast<float16_t *>(smem_ptr);

      float16x4 *b = reinterpret_cast<float16x4 *>(b_frag);

      // TODO (yiakwy) : replaced with more efficient load inst
#pragma unroll
      for (uint32_t j=0; j < 4; j++) {
        // NOTE (yiakwy) : loads 16 consecutive data of 1 row
        s += lane_id_x + lane_id_y * MTX_FRAG_LDB * 4 + j * MTX_FRAG_LDB;
        // s += lane_id_x * MTX_FRAG_LDB + j + lane_id_y * 4;

        (*b)[j] = *s;

        if (fy == 0 && fz == 0) {
        printf("[compute_qk] (x=%d, y=%d, j=%d) (fz=%d, fy=%d) b_mtx_frag[%d, %d]=%f\n", lane_id_x, lane_id_y, j, fz, fy, lane_id_y * 4 + j, lane_id_x, (float)((*b)[j]));
        }
      }

      // NOTE(yiakwy) : k is still in row-major layout
      *k_smem_offset_r =
          k_smem->template advance_offset_by_row<16, channel_size_128b_kv>(*k_smem_offset_r);

      // compute
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        float16x4 *a = reinterpret_cast<float16x4 *>(a_frag[fx]);
        float16x4 *b = reinterpret_cast<float16x4 *>(b_frag);

        if constexpr (std::is_same<DTypeQKAccum, float>::value) {
          floatx4 *d = reinterpret_cast<floatx4 *>(s_frag[fx][fz]);
          *d = __builtin_amdgcn_mfma_f32_16x16x16f16(*a, *b, *d, 0, 0, 0);
          __asm__ volatile("s_barrier" ::);

          if (fx == 0 && fy == 0 && fz == 0) { 

            for (uint32_t reg_id=0; reg_id < 4; reg_id++) {
              printf("[compute_qk] (x=%d, y=%d, reg_id=%d) s_frag[fx=%d][fy=0][fz=%d][%d, %d] = %f\n", lane_id_x, lane_id_y, reg_id, fx, fz, reg_id + lane_id_y * 4, lane_id_x, (*d)[reg_id]);
            }

          }
        } else {
          // TODO (yiakwy) : device cast fp32 to fp16
          assert(0 && "AMD v_mfma instruction does not support fp16 output.");
        }
      }
    }
    if constexpr (sizeof(DTypeKV) == 1) {
      assert(0 && "FP8 KV Cache will be suppported soon.");
    } else {
      *k_smem_offset_r = k_smem->template advance_offset_by_column<2>(*k_smem_offset_r, fy) -
                         num_frags_z * 16 * channel_size_128b_kv;
    }
  }

  } // if lane_id < 64
  #else

#pragma unroll
  // NOTE(yiakwy) each thead read 2 elments and repeat 4xnum_frags_y times , threads cooperatively loads 16x64 fp16 elements
  // 
  // frag_a:
  // Dtype=fp16/bf16
  //      cols             0  .. 15 16 .. 31 32 ..  63
  // frag_x\frag_y   rows  0        1        2  ..  3
  //  0              0
  //                 ..
  //                 15
  //  1              16
  //  
  //frag_b
  //Dtype=fp16/bf16
  //      cols            0  ..  15 16 .. 31 32 .. 63
  // frag_z\frag_y  rows  0          1 ..    2  .. 3
  // 0              0
  //                ..
  //                15
  // 1              16
  //
  for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
      q_smem->ldmatrix_m8n8x4(*q_smem_offset_r, a_frag[fx]);

      // NOTE (yiakwy) : move to the next 16 rows
      *q_smem_offset_r =
              q_smem->template advance_offset_by_row<16, channel_size_128b_q>(*q_smem_offset_r);

    }
    // NOTE(yiakwy) : next to 16 = 2x8 columns
    *q_smem_offset_r = q_smem->template advance_offset_by_column<2>(*q_smem_offset_r, fy) -
                       num_frags_x * 16 * channel_size_128b_q;

#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      if constexpr (sizeof(DTypeKV) == 1) {
        uint32_t b_frag_f8[2];
        if (fy % 2 == 0) {
          k_smem->ldmatrix_m8n8x4_left_half(*k_smem_offset_r, b_frag_f8);
        } else {
          k_smem->ldmatrix_m8n8x4_right_half(*k_smem_offset_r, b_frag_f8);
        }
        b_frag_f8[0] = frag_layout_swizzle_16b_to_8b(b_frag_f8[0]);
        b_frag_f8[1] = frag_layout_swizzle_16b_to_8b(b_frag_f8[1]);
        vec_cast<DTypeQ, DTypeKV>::template cast<8>((DTypeQ*)b_frag, (DTypeKV*)b_frag_f8);
      } else {
        k_smem->ldmatrix_m8n8x4(*k_smem_offset_r, b_frag);
      }
      *k_smem_offset_r =
          k_smem->template advance_offset_by_row<16, channel_size_128b_kv>(*k_smem_offset_r);

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
    if constexpr (sizeof(DTypeKV) == 1) {
      if (fy % 2 == 1) {
        *k_smem_offset_r = k_smem->template advance_offset_by_column<2>(*k_smem_offset_r, fy / 2);
      }
      *k_smem_offset_r -= num_frags_z * 16 * channel_size_128b_kv;
    } else {
      *k_smem_offset_r = k_smem->template advance_offset_by_column<2>(*k_smem_offset_r, fy) -
                         num_frags_z * 16 * channel_size_128b_kv;
    }
  }
  
#endif // USE_ROCM  
  
  *q_smem_offset_r -= num_frags_y * 2;
  *k_smem_offset_r -= num_frags_y * sizeof(DTypeKV);

  if constexpr (std::is_same<DTypeQKAccum, float>::value) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {

#ifdef USE_ROCM

#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
          s_frag[fx][fz][reg_id] =
              apply_logits_post_hook<logits_post_hook>(s_frag[fx][fz][reg_id], soft_cap);
        }

#else
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
          s_frag[fx][fz][reg_id] =
              apply_logits_post_hook<logits_post_hook>(s_frag[fx][fz][reg_id], soft_cap);
        }
#endif // USE_ROCM

      }
    }
  } else {
    static_assert(std::is_same<DTypeQKAccum, half>::value);
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {

#ifdef USE_ROCM

        for (uint32_t reg_id = 0; reg_id < 2; ++reg_id) {
          *(half2*)(&s_frag[fx][fz][reg_id * 2]) = apply_logits_post_hook<logits_post_hook>(
              *(half2*)(&s_frag[fx][fz][reg_id * 2]), soft_cap);
        }

#else
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
          *(half2*)(&s_frag[fx][fz][reg_id * 2]) = apply_logits_post_hook<logits_post_hook>(
              *(half2*)(&s_frag[fx][fz][reg_id * 2]), soft_cap);
        }
#endif

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

template <MaskMode mask_mode, uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z,
          typename DTypeQKAccum>
__device__ __forceinline__ void mask_s(const uint32_t qo_packed_idx_base,
                                       const uint32_t kv_idx_base, const uint32_t qo_len,
                                       const uint32_t kv_len, const uint32_t window_left,
                                       const uint32_t chunk_end, const uint_fastdiv group_size,
                                       uint8_t* custom_mask,
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
                 ? (kv_idx + qo_len > kv_len + q_idx || (kv_idx >= chunk_end) ||
                    kv_idx + qo_len + window_left < kv_len + q_idx)
                 : kv_idx >= chunk_end || kv_idx + qo_len + window_left < kv_len + q_idx);
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

template <uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z,
          SwizzleMode swizzle_mode, typename DTypeQ, typename DTypeKV, typename DTypeQKAccum>
__device__ __forceinline__ void compute_sfm_v(smem_t<swizzle_mode>* v_smem,
                                              uint32_t* v_smem_offset_r,
                                              DTypeQKAccum (*s_frag)[num_frags_z][8],
                                              float (*o_frag)[num_frags_y][8], float (*d)[2]) {
  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t channel_size_128b_kv = head_dim / num_elems_per_128b<DTypeKV>();

  #ifdef USE_ROCM
  using float16x4  = __attribute__((__vector_size__(4 * sizeof(rocwmma::float16_t)))) rocwmma::float16_t;
  using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  #endif

  DTypeQ s_frag_f16[num_frags_x][num_frags_z][8];
  if constexpr (std::is_same<DTypeQKAccum, float>::value) {
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
        vec_cast<DTypeQ, float>::template cast<8>(s_frag_f16[fx][fz], s_frag[fx][fz]);
      }
    }
  }

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
      
      #ifdef USE_ROCM
        // TODO (yiakwy) : check feasibility with v_mfma_fp32_m16xn16xk16_fp16
#pragma unroll
        for (int i=0; i < 8; ++i) {
          if constexpr (std::is_same<DTypeQKAccum, float>::value) {
            // device cast from half to float
            d[fx][i] += (float)s_frag_f16[fx][fz][i];
          } else {
            // device cast from half to float
            d[fx][i] += (float)s_frag[fx][fz][i];
          }
        }

      #else

      if constexpr (std::is_same<DTypeQKAccum, float>::value) {
        mma::rowsum_f16f16f32(d[fx], s_frag_f16[fx][fz]);
      } else {
        mma::rowsum_f16f16f32(d[fx], s_frag[fx][fz]);
      }

      #endif // USE_ROCM
    }
  }

#pragma unroll
  for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      uint32_t b_frag[4];
      if constexpr (sizeof(DTypeKV) == 1) {

        // TODO (yiakwy) : add FP8 support for KV Cache
        assert(0 && "FP8 KV Cache is not supported.");

        uint32_t b_frag_f8[2];
        if (fy % 2 == 0) {
          v_smem->ldmatrix_m8n8x4_trans_left_half(*v_smem_offset_r, b_frag_f8);
        } else {
          v_smem->ldmatrix_m8n8x4_trans_right_half(*v_smem_offset_r, b_frag_f8);
        }
        b_frag_f8[0] = frag_layout_swizzle_16b_to_8b_trans(b_frag_f8[0]);
        b_frag_f8[1] = frag_layout_swizzle_16b_to_8b_trans(b_frag_f8[1]);
        vec_cast<DTypeQ, DTypeKV>::template cast<8>((DTypeQ*)b_frag, (DTypeKV*)b_frag_f8);
        swap(b_frag[1], b_frag[2]);
      } else {
        
        #ifdef USE_ROCM
        
        b128_t* smem_ptr = v_smem->base + *v_smem_offset_r;

        float16x4 *b = reinterpret_cast<float16x4 *>(b_frag);

#pragma unroll
        for (int j=0; j < 4; j++) {
            (*b)[j] = (rocwmma::float16_t)(*b)[j];
        }

        #else
        v_smem->ldmatrix_m8n8x4_trans(*v_smem_offset_r, b_frag);
        #endif // USE_ROCM

      }
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {

        #ifdef USE_ROCM

        float16x4 *b = reinterpret_cast<float16x4 *>(b_frag);
        floatx4 *o = reinterpret_cast<floatx4 *>(o_frag[fx][fz]);

        if constexpr (std::is_same<DTypeQKAccum, float>::value) {
          float16x4 *a = reinterpret_cast<float16x4 *>(s_frag_f16[fx][fz]);
          *o = __builtin_amdgcn_mfma_f32_16x16x16f16(*a, *b, *o, 0, 0, 0);
        } else {
          float16x4 *a = reinterpret_cast<float16x4 *>(s_frag[fx][fz]);
          *o = __builtin_amdgcn_mfma_f32_16x16x16f16(*a, *b, *o, 0, 0, 0);
        }

        #else

        if constexpr (std::is_same<DTypeQKAccum, float>::value) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeQ>(
              o_frag[fx][fy], (uint32_t*)(s_frag_f16[fx][fz]), b_frag);
        } else {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeQ>(o_frag[fx][fy],
                                                            (uint32_t*)s_frag[fx][fz], b_frag);
        }

        #endif // USE_ROCM
      }

      // TODO (yiakwy) : fix 
      if constexpr (sizeof(DTypeKV) == 1) {
        if (fy % 2 == 1) {
          *v_smem_offset_r = v_smem->template advance_offset_by_column<2>(*v_smem_offset_r, fy / 2);
        }
      } else {
        *v_smem_offset_r = v_smem->template advance_offset_by_column<2>(*v_smem_offset_r, fy);
      }
    }
    *v_smem_offset_r =
        v_smem->template advance_offset_by_row<16, channel_size_128b_kv>(*v_smem_offset_r) -
        sizeof(DTypeKV) * num_frags_y;
  }
  *v_smem_offset_r -= 16 * num_frags_z * channel_size_128b_kv;
}

template <uint32_t num_frags_x, uint32_t num_frags_y, typename DTypeQKAccum>
__device__ __forceinline__ void normalize_d(float (*o_frag)[num_frags_y][8], DTypeQKAccum (*m)[2],
                                            float (*d)[2]) {
  float d_rcp[num_frags_x][2];
  // compute reciprocal of d
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      d_rcp[fx][j] = (m[fx][j] != DTypeQKAccum(-5e4)) ? math::ptx_rcp(d[fx][j]) : 0.f;
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
    float2* smem_md = (float2*)(smem_workspace + num_frags_x * num_frags_y * num_warps_x *
                                                     num_warps_z * warp_size * 8);
    // o: [num_warps, num_frags_x, num_frags_y, warp_size(32), 8]
    // md: [num_warps, num_frags_x, 2, warp_size(32), 2 (m/d)]
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        vec_t<float, 8>::memcpy(
            smem_workspace +
                (((warp_idx * num_frags_x + fx) * num_frags_y + fy) * warp_size + lane_idx) * 8,
            o_frag[fx][fy]);
      }
    }

#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        smem_md[((warp_idx * num_frags_x + fx) * 2 + j) * warp_size + lane_idx] =
            make_float2(float(m[fx][j]), d[fx][j]);
      }
    }

    // synchronize m,d first
    __syncthreads();
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
      float o_scale[2][num_warps_z];
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
          o_scale[j][i] = math::ptx_exp2(float(mi - m_new));
        }
        m[fx][j] = DTypeQKAccum(m_new);
        d[fx][j] = d_new;
      }

#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        vec_t<float, 8> o_new;
        o_new.fill(0.f);
#pragma unroll
        for (uint32_t i = 0; i < num_warps_z; ++i) {
          vec_t<float, 8> oi;
          oi.load(smem_workspace +
                  ((((i * num_warps_x + get_warp_idx_x<num_warps_x, num_warps_z>()) * num_frags_x +
                     fx) *
                        num_frags_y +
                    fy) *
                       warp_size +
                   lane_idx) *
                      8);
#pragma unroll
          for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
            o_new[reg_id] += oi[reg_id] * o_scale[(reg_id % 4) / 2][i];
          }
        }
        o_new.store(o_frag[fx][fy]);
      }
    }
  }
}

template <uint32_t num_warps_x, uint32_t num_warps_z, uint32_t num_frags_x, uint32_t num_frags_y,
          SwizzleMode swizzle_mode, typename DTypeOut>
__device__ __forceinline__ void write_o_reg_gmem(
    float (*o_frag)[num_frags_y][8], smem_t<swizzle_mode>* o_smem, DTypeOut* o_ptr_base,
    const uint32_t o_packed_idx_base, const uint32_t qo_upper_bound, const uint32_t o_stride_n,
    const uint32_t o_stride_h, const uint_fastdiv group_size) {
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
        vec_cast<DTypeOut, float>::template cast<8>((DTypeOut*)o_frag_f16, o_frag[fx][fy]);
#ifdef FLASHINFER_STMATRIX_M8N8X4_ENABLED
        uint32_t o_smem_offset_w = o_smem->template get_permuted_offset<channel_size_128b_out>(
            (warp_idx_x * num_frags_x + fx) * 16 + lane_idx % 16, fy * 2 + lane_idx / 16);
        o_smem->stmatrix_m8n8x4(o_smem_offset_w, o_frag_f16);
#else
        uint32_t o_smem_offset_w = o_smem->template get_permuted_offset<channel_size_128b_out>(
            (warp_idx_x * num_frags_x + fx) * 16 + lane_idx / 4, fy * 2);
        ((uint32_t*)(o_smem->base + o_smem_offset_w))[lane_idx % 4] = o_frag_f16[0];
        ((uint32_t*)(o_smem->base + o_smem_offset_w + 8 * channel_size_128b_out))[lane_idx % 4] =
            o_frag_f16[1];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1)))[lane_idx % 4] = o_frag_f16[2];
        ((uint32_t*)(o_smem->base + (o_smem_offset_w ^ 0x1) +
                     8 * channel_size_128b_out))[lane_idx % 4] = o_frag_f16[3];
#endif
      }
    }

    uint32_t o_smem_offset_w = o_smem->template get_permuted_offset<channel_size_128b_out>(
        warp_idx_x * num_frags_x * 16 + lane_idx / 8, lane_idx % 8);

#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        uint32_t q, r;
        group_size.divmod(o_packed_idx_base + lane_idx / 8 + fx * 16 + j * 4, q, r);
        const uint32_t o_idx = q;
        DTypeOut* o_ptr = o_ptr_base + q * o_stride_n + r * o_stride_h;
#pragma unroll
        for (uint32_t fyo = 0; fyo < num_frags_y / 4; ++fyo) {
          if (o_idx < qo_upper_bound) {
            o_smem->store_128b(o_smem_offset_w, o_ptr);
          }
          o_ptr += 8 * num_elems_per_128b<DTypeOut>();
          o_smem_offset_w = o_smem->template advance_offset_by_column<8>(o_smem_offset_w, fyo);
        }
        o_smem_offset_w =
            o_smem->template advance_offset_by_row<4, channel_size_128b_out>(o_smem_offset_w) -
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
 * \tparam pos_encoding_mode The positional encoding mode.
 * \tparam num_frags_x The number of fragments in x dimension.
 * \tparam num_frags_y The number of fragments in y dimension.
 * \tparam num_frags_z The number of fragments in z dimension.
 * \tparam num_warps The number of warps in the threadblock.
 * \tparam DTypeQ The data type of the query tensor.
 * \tparam DTypeKV The data type of the key/value tensor.
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
template <LogitsPostHook logits_post_hook, MaskMode mask_mode, PosEncodingMode pos_encoding_mode,
          uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z, uint32_t num_warps_x,
          uint32_t num_warps_z, typename DTypeQ, typename DTypeKV, typename DTypeQKAccum,
          typename DTypeOut>
__global__
__launch_bounds__(num_warps_x* num_warps_z* warp_size) void SinglePrefillWithKVCacheKernel(
    DTypeQ* __restrict__ q, DTypeKV* __restrict__ k, DTypeKV* __restrict__ v,
    uint8_t* __restrict__ custom_mask, DTypeOut* __restrict__ o, float* __restrict__ lse,
    const uint32_t qo_len, const uint32_t kv_len, const bool partition_kv,
    const uint_fastdiv group_size, const uint32_t q_stride_n, const uint32_t q_stride_h,
    const uint32_t kv_stride_n, const uint32_t kv_stride_h, const int32_t maybe_window_left,
    const float logits_soft_cap, float sm_scale, const float log2_rope_rcp_scale,
    const float log2_rope_rcp_theta) {
#if (__CUDA_ARCH__ < 800)
  if constexpr (std::is_same<DTypeQ, nv_bfloat16>::value) {
    FLASHINFER_RUNTIME_ASSERT("Prefill kernels do not support bf16 on sm75.");
  } else {
#endif
    static_assert(sizeof(DTypeQ) == 2);
    static_assert(sizeof(DTypeOut) == 2);
    sm_scale *=
        (logits_post_hook == LogitsPostHook::kNone ? math::log2e : math::ptx_rcp(logits_soft_cap));
    const uint32_t lane_idx = threadIdx.x, warp_idx = get_warp_idx<num_warps_x, num_warps_z>();
    const uint32_t bx = blockIdx.x, chunk_idx = blockIdx.y, kv_head_idx = blockIdx.z;
    const uint32_t num_kv_heads = gridDim.z, num_qo_heads = num_kv_heads * group_size;
    constexpr uint32_t num_rows_per_cta = num_frags_x * num_warps_x * 16;
    
    const tensor_info_t qkv_info(qo_len, kv_len, num_qo_heads, num_kv_heads, q_stride_n, q_stride_h,
                                 kv_stride_n, kv_stride_h, /*head_dim=*/num_frags_y * 16);

    float alibi_slopes[num_frags_x][2];

    const uint32_t num_chunks = gridDim.y;
    const uint32_t max_chunk_size = partition_kv ? ceil_div(kv_len, num_chunks) : kv_len;
    const uint32_t chunk_start = partition_kv ? chunk_idx * max_chunk_size : 0;
    const uint32_t chunk_end =
        partition_kv ? min((chunk_idx + 1) * max_chunk_size, kv_len) : kv_len;
    const uint32_t chunk_size = chunk_end - chunk_start;

    auto block = cg::this_thread_block();
    const uint32_t window_left = (maybe_window_left >= 0) ? maybe_window_left : kv_len;

    constexpr uint32_t head_dim = num_frags_y * 16;
    constexpr uint32_t channel_size_128b_q = head_dim / num_elems_per_128b<DTypeQ>(); // e.g.:64/8
    constexpr uint32_t channel_size_128b_kv = head_dim / num_elems_per_128b<DTypeKV>();
    constexpr uint32_t channel_size_128b_out = head_dim / num_elems_per_128b<DTypeOut>(); // e.g.: 64/4

    extern __shared__ uint8_t smem[];

    // NOTE(yiakwy) : e.g. 1x4 fragments, threads cooperatively ld/st 16 bf16/half elements in each frag
    DTypeQKAccum s_frag[num_frags_x][num_frags_z][8];
    // NOTE(yiakwy) : e.g. 1x4 fragments, threads cooperatively ld/st 16 bf16/half elements in each frag
    float o_frag[num_frags_x][num_frags_y][8];
    DTypeQKAccum m[num_frags_x][2];
    float d[num_frags_x][2];
    float rope_freq[num_frags_y / 2][4];
    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      init_rope_freq<num_frags_y>(rope_freq, log2_rope_rcp_scale, log2_rope_rcp_theta);
    }
    init_states<num_frags_x, num_frags_y>(o_frag, m, d);

    // TODO (yiakwy) : to be used by load_q_global_smem, double check to compute offset of q
    // cooperatively fetch q fragment from gmem to reg
    const uint32_t qo_packed_idx_base =
        (bx * num_warps_x + get_warp_idx_x<num_warps_x, num_warps_z>()) * num_frags_x * 16;

    constexpr SwizzleMode swizzle_mode_q = SwizzleMode::k128B;
    smem_t<swizzle_mode_q> qo_smem(smem);

    // TODO (yiakwy) : to be used by load_q_global_smem, double check to compute offset of q
    #ifdef USE_ROCM
    DTypeQ* q_ptr_base =
        q + qkv_info.get_q_elem_offset(0, kv_head_idx * group_size, 0/*threads related offset computed in function blocks*/);
    #else
    DTypeQ* q_ptr_base =
        q + qkv_info.get_q_elem_offset(0, kv_head_idx * group_size,
                                       (lane_idx % 8) * num_elems_per_128b<DTypeQ>());
    #endif // USE_ROCM-q
    
    #ifdef USE_ROCM
    DTypeOut* o_ptr_base =
        partition_kv
            ? o + chunk_idx * num_qo_heads * head_dim +
                  qkv_info.get_o_elem_offset(0, kv_head_idx * group_size, 0)
            : o + qkv_info.get_o_elem_offset(0, kv_head_idx * group_size, 0);
    #else
    DTypeOut* o_ptr_base =
        partition_kv
            ? o + chunk_idx * num_qo_heads * head_dim +
                  qkv_info.get_o_elem_offset(0, kv_head_idx * group_size,
                                             (lane_idx % 8) * num_elems_per_128b<DTypeOut>())
            : o + qkv_info.get_o_elem_offset(0, kv_head_idx * group_size,
                                             (lane_idx % 8) * num_elems_per_128b<DTypeOut>());
    #endif // USE_ROCM-o

    // TODO (yiakwy) : refactor
    #ifdef USE_ROCM
    // used by compute_qk for reading smem
    uint32_t q_smem_offset_r = qo_smem.template get_permuted_offset<channel_size_128b_q>(
        get_warp_idx_x<num_warps_x, num_warps_z>() * num_frags_x * 16, 0);
    #else
    uint32_t q_smem_offset_r = qo_smem.template get_permuted_offset<channel_size_128b_q>(
        get_warp_idx_x<num_warps_x, num_warps_z>() * num_frags_x * 16 + lane_idx % 16,
        lane_idx / 16);    
    #endif

    if (threadIdx.x == 0 && threadIdx.z == 0) {
      printf("[single prefill kernel] channel_size_128b_q = %d\n", channel_size_128b_q);
    }

    // NOTE(yiakwy) : FA2 outter loop (block level) load q first and iterate over sequence dimension inside a block
    load_q_global_smem<num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
        qo_packed_idx_base, qo_len, q_ptr_base, q_stride_n, q_stride_h, group_size, &qo_smem);

    cp_async::commit_group();
    cp_async::wait_group<0>();
    block.sync();

    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      q_smem_inplace_apply_rotary_multiply_sm_scale<num_warps_x, num_warps_z, num_frags_x,
                                                    num_frags_y, swizzle_mode_q, DTypeQ>(
          qo_packed_idx_base, qo_len, kv_len, group_size, &qo_smem, &q_smem_offset_r, rope_freq,
          sm_scale);
    } else {
      if (threadIdx.x==0 && threadIdx.z==0) {
        printf("[single prefill kernel] skip q_smem_inplace_multiply_sm_scale.\n");
      }
      // TODO (yiakwy) : recover
      /*
      q_smem_inplace_multiply_sm_scale<num_warps_x, num_warps_z, num_frags_x, num_frags_y,
                                       swizzle_mode_q, DTypeQ>(&qo_smem, sm_scale);
       */
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

    constexpr SwizzleMode swizzle_mode_kv =
        (sizeof(DTypeKV) == 1 && head_dim == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
    constexpr uint32_t kv_frag_rows = swizzle_mode_kv == SwizzleMode::k128B ? 4 : 8;
    constexpr uint32_t kv_frag_cols = swizzle_mode_kv == SwizzleMode::k128B ? 8 : 4;
    smem_t<swizzle_mode_kv> k_smem(smem +
                                   (num_warps_x * num_frags_x * sizeof(DTypeQ)) * 16 * head_dim),
        v_smem(smem + (num_warps_x * num_frags_x * sizeof(DTypeQ) +
                       num_warps_z * num_frags_z * sizeof(DTypeKV)) *
                          16 * head_dim);

    const uint32_t num_iterations = ceil_div(
        mask_mode == MaskMode::kCausal
            ? min(chunk_size,
                  sub_if_greater_or_zero(
                      kv_len - qo_len + ((bx + 1) * num_rows_per_cta) / group_size, chunk_start))
            : chunk_size,
        16 * num_warps_z * num_frags_z);

    const uint32_t window_iteration =
        ceil_div(sub_if_greater_or_zero(kv_len + (bx + 1) * num_rows_per_cta,
                                        qo_len + window_left + chunk_start),
                 (16 * num_warps_z * num_frags_z));

    const uint32_t mask_iteration =
        (mask_mode == MaskMode::kCausal
             ? min(chunk_size,
                   sub_if_greater_or_zero(kv_len + (bx * num_rows_per_cta) / group_size - qo_len,
                                          chunk_start))
             : chunk_size) /
        (16 * num_warps_z * num_frags_z);

    // TODO (yiakwy) : refactor
    #ifdef USE_ROCM

    DTypeKV* k_ptr =
        k + qkv_info.get_kv_elem_offset(
                chunk_start + warp_idx * kv_frag_rows + 0/* nvgpu : (lane_idx / 8) */, kv_head_idx,
                0/* nvgpu : (lane_idx % 8 ) * 8 */);
    DTypeKV* v_ptr =
        v + qkv_info.get_kv_elem_offset(
                chunk_start + warp_idx * kv_frag_rows + 0, kv_head_idx,
                0);
    // NOTE (yiakwy) : _w is used for storing (produce_kv) and _r is used for reading (compute_qk)
    // NOTE (yiakwy) : We reuse NV GPU uint4 loading layout
    uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<channel_size_128b_kv>(
                 get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16 +
                     0/* nvgpu : (lane_idx / 16) * 8 + lane_idx % 8 */,
                 0/* nvgpu : (lane_idx % 16) / 8) => {0, 1}*/),
             v_smem_offset_r = v_smem.template get_permuted_offset<channel_size_128b_kv>(
                 get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16 + 0 /* nvgpu : lane_idx % 16 => {0..15} */,
                 0/* nvgpu : lane_idx / 16 => {0, 1} */),
             kv_smem_offset_w = k_smem.template get_permuted_offset<channel_size_128b_kv>(
                 warp_idx * kv_frag_rows + lane_idx / kv_frag_cols, lane_idx % kv_frag_cols);
    
    #else
    DTypeKV* k_ptr =
        k + qkv_info.get_kv_elem_offset(
                chunk_start + warp_idx * kv_frag_rows + lane_idx / kv_frag_cols, kv_head_idx,
                (lane_idx % kv_frag_cols) * num_elems_per_128b<DTypeKV>());
    DTypeKV* v_ptr =
        v + qkv_info.get_kv_elem_offset(
                chunk_start + warp_idx * kv_frag_rows + lane_idx / kv_frag_cols, kv_head_idx,
                (lane_idx % kv_frag_cols) * num_elems_per_128b<DTypeKV>());
    uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<channel_size_128b_kv>(
                 get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16 +
                     8 * (lane_idx / 16) + lane_idx % 8,
                 (lane_idx % 16) / 8),
             v_smem_offset_r = v_smem.template get_permuted_offset<channel_size_128b_kv>(
                 get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16 + lane_idx % 16,
                 lane_idx / 16),
             kv_smem_offset_w = k_smem.template get_permuted_offset<channel_size_128b_kv>(
                 warp_idx * kv_frag_rows + lane_idx / kv_frag_cols, lane_idx % kv_frag_cols);
    #endif // USE_ROCM

    if (threadIdx.x==0 && threadIdx.z==0) {
    printf("[single prefill kernel] ===== producing key =====\n");
    }
    produce_kv<SharedMemFillMode::kNoFill, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        k_smem, &kv_smem_offset_w, &k_ptr, kv_stride_n, 0, chunk_size);
    cp_async::commit_group();
    
    if (threadIdx.x==0 && threadIdx.z==0) {
    printf("[single prefill kernel] ***** producing value *****\n");
    }
    produce_kv<SharedMemFillMode::kFillZero, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        v_smem, &kv_smem_offset_w, &v_ptr, kv_stride_n, 0, chunk_size);
    cp_async::commit_group();

    // NOTE (yiakwy) : kv inner loop
#pragma unroll 1
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
      cp_async::wait_group<1>();
      block.sync();

      if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
        // TODO (yiakwy) : recover
        /*
        k_smem_inplace_apply_rotary<num_warps_x, num_warps_z, num_frags_y, num_frags_z,
                                    swizzle_mode_kv, DTypeKV>(
            chunk_start + iter * 16 * num_warps_z * num_frags_z, &k_smem, &k_smem_offset_r,
            rope_freq);
        block.sync();
        */
      }

      // compute attention score
      if (threadIdx.x == 0 && threadIdx.z == 0) {
        printf("[single prefill kernel] start calling compute_qk...\n");
      }
      compute_qk<logits_post_hook, num_frags_x, num_frags_y, num_frags_z, swizzle_mode_q,
                 swizzle_mode_kv, DTypeQ, DTypeKV>(&qo_smem, &q_smem_offset_r, &k_smem,
                                                   &k_smem_offset_r, s_frag, logits_soft_cap);

      if constexpr (pos_encoding_mode == PosEncodingMode::kALiBi) {
        apply_alibi_bias<num_frags_x, num_frags_z>(
            qo_packed_idx_base,
            chunk_start + (iter * num_warps_z + get_warp_idx_z<num_warps_x, num_warps_z>()) *
                              num_frags_z * 16,
            int(kv_len) - int(qo_len), group_size, alibi_slopes, s_frag);
      }
      // apply mask
      if constexpr (mask_mode == MaskMode::kCustom) {
        mask_s<mask_mode, num_frags_x, num_frags_y, num_frags_z>(
            qo_packed_idx_base,
            chunk_start + (iter * num_warps_z + get_warp_idx_z<num_warps_x, num_warps_z>()) *
                              num_frags_z * 16,
            qo_len, kv_len, window_left, chunk_end, group_size, custom_mask, s_frag);
      } else {
        if (iter >= mask_iteration || iter < window_iteration) {
          mask_s<mask_mode, num_frags_x, num_frags_y, num_frags_z>(
              qo_packed_idx_base,
              chunk_start + (iter * num_warps_z + get_warp_idx_z<num_warps_x, num_warps_z>()) *
                                num_frags_z * 16,
              qo_len, kv_len, window_left, chunk_end, group_size, nullptr, s_frag);
        }
      }

      // compute m,d states in online softmax
      update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(s_frag, o_frag, m, d);

      block.sync();

      // TODO (yiakwy) : REMOVE
      if (threadIdx.x == 0 && threadIdx.z == 0) {
        printf("[single prefill kernel] calling pdate_mdo_states completes.");
      }

      // NOTE (yiakwy) : prepare the next loading
      if (iter + 1 < num_iterations) {

      produce_kv<SharedMemFillMode::kNoFill, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
          k_smem, &kv_smem_offset_w, &k_ptr, kv_stride_n,
          (iter + 1) * 16 * num_warps_z * num_frags_z, chunk_size);
      cp_async::commit_group();
      cp_async::wait_group<1>();
      block.sync();

      }
      
      // compute sfm*v
      compute_sfm_v<num_frags_x, num_frags_y, num_frags_z, swizzle_mode_kv, DTypeQ, DTypeKV>(
          &v_smem, &v_smem_offset_r, s_frag, o_frag, d);

      block.sync();

      // NOTE (yiakwy) : prepare the next loading
      if (iter + 1 < num_iterations) {

      produce_kv<SharedMemFillMode::kFillZero, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
          v_smem, &kv_smem_offset_w, &v_ptr, kv_stride_n,
          (iter + 1) * 16 * num_warps_z * num_frags_z, chunk_size);
      cp_async::commit_group();

      }
    }
  
    cp_async::wait_group<0>();
    block.sync();

    // threadblock synchronization
    threadblock_sync_mdo_states<num_warps_x, num_warps_z, num_frags_x, num_frags_y, DTypeQKAccum>(
        o_frag, (float*)smem, m, d, warp_idx, lane_idx);

    // normalize d
    normalize_d<num_frags_x, num_frags_y>(o_frag, m, d);

    // write back
    write_o_reg_gmem<num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
        o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
        /*o_stride_n=*/
        partition_kv ? num_qo_heads * head_dim * num_chunks : num_qo_heads * head_dim,
        /*o_stride_h=*/head_dim, group_size);

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
              if (partition_kv) {
                lse[(qo_idx * num_chunks + chunk_idx) * num_qo_heads + qo_head_idx] =
                    math::ptx_log2(d[fx][j]) + float(m[fx][j]);
              } else {
                // TODO (yiakwy) : REMOVE
                uint32_t warp_idx = get_warp_idx<num_warps_x, num_warps_z>();
                // if (warp_idx == 0) {
                printf("[write lse] (qo_idx=%d, qo_head_idx=%d), warp_idx=%d, (y, z)=(%d, %d), d[%d][%d]=%f, m[%d][%d]=%f", qo_idx, qo_head_idx, warp_idx, threadIdx.y, threadIdx.z, fx, j, d[fx][j], fx, j, float(m[fx][j]));
                // }
                lse[qo_idx * num_qo_heads + qo_head_idx] =
                    math::ptx_log2(d[fx][j]) + float(m[fx][j]);
              }
            }
          }
        }
      }
    }
#if (__CUDA_ARCH__ < 800)
  }
#endif
}

template <LogitsPostHook logits_post_hook, MaskMode mask_mode, PosEncodingMode pos_encoding_mode,
          uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z, uint32_t num_warps_x,
          uint32_t num_warps_z, typename DTypeQ, typename DTypeKV, typename DTypeQKAccum,
          typename DTypeOut, typename IdType>
__global__
__launch_bounds__(num_warps_x* num_warps_z* warp_size) void BatchPrefillWithRaggedKVCacheKernel(
    DTypeQ* __restrict__ q, IdType* __restrict__ request_indices,
    IdType* __restrict__ q_tile_indices, IdType* __restrict__ kv_tile_indices,
    IdType* __restrict__ q_indptr, DTypeKV* __restrict__ k, DTypeKV* __restrict__ v,
    IdType* __restrict__ kv_indptr, uint8_t* __restrict__ custom_mask,
    IdType* __restrict__ qk_indptr, IdType* __restrict__ q_offset,
    IdType* __restrict__ k_rope_pos_offset, IdType* __restrict__ o_indptr, DTypeOut* __restrict__ o,
    float* __restrict__ lse, bool* __restrict__ block_valid_mask,
    IdType* __restrict__ kv_chunk_size_ptr, const bool partition_kv, const uint_fastdiv group_size,
    const uint32_t q_stride_n, const uint32_t q_stride_h, const uint32_t kv_stride_n,
    const uint32_t kv_stride_h, const int32_t maybe_window_left, const float logits_soft_cap,
    float sm_scale, const float log2_rope_rcp_scale, const float log2_rope_rcp_theta) {
#if (__CUDA_ARCH__ < 800)
  if constexpr (std::is_same<DTypeQ, nv_bfloat16>::value) {
    FLASHINFER_RUNTIME_ASSERT("Prefill kernels do not support bf16 on sm75.");
  } else {
#endif
    static_assert(sizeof(DTypeQ) == 2);
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
    const uint32_t kv_len_safe = kv_len > 0 ? kv_len : 1;
    const uint32_t window_left = (maybe_window_left >= 0) ? maybe_window_left : kv_len;
    const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
    const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
    const uint32_t chunk_end =
        partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
    const uint32_t chunk_size = chunk_end - chunk_start;
    const tensor_info_t qkv_info(qo_len, kv_len, num_qo_heads, num_kv_heads, q_stride_n, q_stride_h,
                                 kv_stride_n, kv_stride_h, /*head_dim=*/num_frags_y * 16);
    float alibi_slopes[num_frags_x][2];
    const uint32_t qo_upper_bound =
        min(qo_len, ceil_div((qo_tile_idx + 1) * num_rows_per_cta, group_size));

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
    constexpr SwizzleMode swizzle_mode_q = SwizzleMode::k128B;
    smem_t<swizzle_mode_q> qo_smem(smem);

    DTypeQ* q_ptr_base =
        q + qkv_info.get_q_elem_offset(q_indptr[request_idx], kv_head_idx * group_size,
                                       (lane_idx % 8) * num_elems_per_128b<DTypeQ>());

    DTypeOut* o_ptr_base =
        partition_kv
            ? o + kv_tile_idx * num_qo_heads * head_dim +
                  qkv_info.get_o_elem_offset(o_indptr[request_idx], kv_head_idx * group_size,
                                             (lane_idx % 8) * num_elems_per_128b<DTypeOut>())
            : o + qkv_info.get_o_elem_offset(o_indptr[request_idx], kv_head_idx * group_size,
                                             (lane_idx % 8) * num_elems_per_128b<DTypeOut>());

    // 32x4 -> 16x8
    uint32_t q_smem_offset_r = qo_smem.template get_permuted_offset<channel_size_128b_q>(
        get_warp_idx_x<num_warps_x, num_warps_z>() * num_frags_x * 16 + lane_idx % 16,
        lane_idx / 16);

    load_q_global_smem<num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
        qo_packed_idx_base, qo_upper_bound, q_ptr_base, q_stride_n, q_stride_h, group_size,
        &qo_smem);

    cp_async::commit_group();
    cp_async::wait_group<0>();
    block.sync();

    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      if (!q_offset) {
        q_smem_inplace_apply_rotary_multiply_sm_scale<num_warps_x, num_warps_z, num_frags_x,
                                                      num_frags_y, swizzle_mode_q, DTypeQ>(
            qo_packed_idx_base, qo_len, kv_len, group_size, &qo_smem, &q_smem_offset_r, rope_freq,
            sm_scale);
      } else {
        q_smem_inplace_apply_rotary_with_pos_multiply_sm_scale<
            num_warps_x, num_warps_z, num_frags_x, num_frags_y, swizzle_mode_q, DTypeQ>(
            qo_packed_idx_base, q_offset + q_indptr[request_idx], &qo_smem, group_size,
            &q_smem_offset_r, rope_freq, sm_scale);
      }
    } else {
      q_smem_inplace_multiply_sm_scale<num_warps_x, num_warps_z, num_frags_x, num_frags_y,
                                       swizzle_mode_q, DTypeQ>(&qo_smem, sm_scale);
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

    const uint32_t num_iterations = ceil_div(
        (mask_mode == MaskMode::kCausal
             ? min(chunk_size,
                   sub_if_greater_or_zero(
                       kv_len - qo_len + ((qo_tile_idx + 1) * num_rows_per_cta) / group_size,
                       chunk_start))
             : chunk_size),
        16 * num_warps_z * num_frags_z);

    const uint32_t window_iteration =
        ceil_div(sub_if_greater_or_zero(kv_len + (bx + 1) * num_rows_per_cta,
                                        qo_len + window_left + chunk_start),
                 (16 * num_warps_z * num_frags_z));

    const uint32_t mask_iteration =
        (mask_mode == MaskMode::kCausal
             ? min(chunk_size, sub_if_greater_or_zero(
                                   kv_len + (qo_tile_idx * num_rows_per_cta) / group_size - qo_len,
                                   chunk_start))
             : chunk_size) /
        (16 * num_warps_z * num_frags_z);

    constexpr SwizzleMode swizzle_mode_kv =
        (sizeof(DTypeKV) == 1 && head_dim == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
    constexpr uint32_t kv_frag_rows = swizzle_mode_kv == SwizzleMode::k128B ? 4 : 8;
    constexpr uint32_t kv_frag_cols = swizzle_mode_kv == SwizzleMode::k128B ? 8 : 4;
    smem_t<swizzle_mode_kv> k_smem(smem +
                                   (num_warps_x * num_frags_x * sizeof(DTypeQ)) * 16 * head_dim),
        v_smem(smem + (num_warps_x * num_frags_x * sizeof(DTypeQ) +
                       num_warps_z * num_frags_z * sizeof(DTypeKV)) *
                          16 * head_dim);

    uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<channel_size_128b_kv>(
                 get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16 +
                     8 * (lane_idx / 16) + lane_idx % 8,
                 (lane_idx % 16) / 8),
             v_smem_offset_r = v_smem.template get_permuted_offset<channel_size_128b_kv>(
                 get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16 + lane_idx % 16,
                 lane_idx / 16),
             kv_smem_offset_w = k_smem.template get_permuted_offset<channel_size_128b_kv>(
                 warp_idx * kv_frag_rows + lane_idx / kv_frag_cols, lane_idx % kv_frag_cols);

    DTypeKV* k_ptr =
        k + qkv_info.get_kv_elem_offset(kv_indptr[request_idx] + chunk_start +
                                            warp_idx * kv_frag_rows + lane_idx / kv_frag_cols,
                                        kv_head_idx,
                                        (lane_idx % kv_frag_cols) * num_elems_per_128b<DTypeKV>());
    DTypeKV* v_ptr =
        v + qkv_info.get_kv_elem_offset(kv_indptr[request_idx] + chunk_start +
                                            warp_idx * kv_frag_rows + lane_idx / kv_frag_cols,
                                        kv_head_idx,
                                        (lane_idx % kv_frag_cols) * num_elems_per_128b<DTypeKV>());

    produce_kv<SharedMemFillMode::kNoFill, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        k_smem, &kv_smem_offset_w, &k_ptr, kv_stride_n, 0, chunk_size);
    cp_async::commit_group();
    produce_kv<SharedMemFillMode::kFillZero, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        v_smem, &kv_smem_offset_w, &v_ptr, kv_stride_n, 0, chunk_size);
    cp_async::commit_group();

#pragma unroll 1
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
      cp_async::wait_group<1>();
      block.sync();

      if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
        k_smem_inplace_apply_rotary<num_warps_x, num_warps_z, num_frags_y, num_frags_z,
                                    swizzle_mode_kv, DTypeKV>(
            (k_rope_pos_offset == nullptr ? 0 : k_rope_pos_offset[request_idx]) + chunk_start +
                iter * 16 * num_warps_z * num_frags_z,
            &k_smem, &k_smem_offset_r, rope_freq);
        block.sync();
      }

      // compute attention score
      compute_qk<logits_post_hook, num_frags_x, num_frags_y, num_frags_z, swizzle_mode_q,
                 swizzle_mode_kv, DTypeQ, DTypeKV>(&qo_smem, &q_smem_offset_r, &k_smem,
                                                   &k_smem_offset_r, s_frag, logits_soft_cap);

      if constexpr (pos_encoding_mode == PosEncodingMode::kALiBi) {
        // TODO(Zihao): handle the case that q_offset is specified
        apply_alibi_bias<num_frags_x, num_frags_z>(
            qo_packed_idx_base,
            chunk_start + (iter * num_warps_z + get_warp_idx_z<num_warps_x, num_warps_z>()) *
                              num_frags_z * 16,
            int(kv_len) - int(qo_len), group_size, alibi_slopes, s_frag);
      }
      // apply mask
      if constexpr (mask_mode == MaskMode::kCustom) {
        mask_s<mask_mode, num_frags_x, num_frags_y, num_frags_z>(
            qo_packed_idx_base,
            chunk_start + (iter * num_warps_z + get_warp_idx_z<num_warps_x, num_warps_z>()) *
                              num_frags_z * 16,
            qo_len, kv_len, window_left, chunk_end, group_size,
            custom_mask + qk_indptr[request_idx], s_frag);
      } else {
        if (iter >= mask_iteration || iter < window_iteration) {
          mask_s<mask_mode, num_frags_x, num_frags_y, num_frags_z>(
              qo_packed_idx_base,
              chunk_start + (iter * num_warps_z + get_warp_idx_z<num_warps_x, num_warps_z>()) *
                                num_frags_z * 16,
              qo_len, kv_len, window_left, chunk_end, group_size, nullptr, s_frag);
        }
      }

      // compute m,d states in online softmax
      update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(s_frag, o_frag, m, d);

      block.sync();
      produce_kv<SharedMemFillMode::kNoFill, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
          k_smem, &kv_smem_offset_w, &k_ptr, kv_stride_n,
          (iter + 1) * 16 * num_warps_z * num_frags_z, chunk_size);
      cp_async::commit_group();
      cp_async::wait_group<1>();
      block.sync();

      // compute sfm*v
      compute_sfm_v<num_frags_x, num_frags_y, num_frags_z, swizzle_mode_kv, DTypeQ, DTypeKV>(
          &v_smem, &v_smem_offset_r, s_frag, o_frag, d);

      block.sync();
      produce_kv<SharedMemFillMode::kFillZero, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
          v_smem, &kv_smem_offset_w, &v_ptr, kv_stride_n,
          (iter + 1) * 16 * num_warps_z * num_frags_z, chunk_size);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    block.sync();

    // threadblock synchronization
    threadblock_sync_mdo_states<num_warps_x, num_warps_z, num_frags_x, num_frags_y, DTypeQKAccum>(
        o_frag, (float*)smem, m, d, warp_idx, lane_idx);

    // normalize d
    normalize_d<num_frags_x, num_frags_y>(o_frag, m, d);

    const uint32_t num_kv_chunks = (kv_len_safe + kv_chunk_size - 1) / kv_chunk_size;

    // write back
    write_o_reg_gmem<num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
        o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
        /*o_stride_n=*/
        partition_kv ? num_qo_heads * head_dim * num_kv_chunks : num_qo_heads * head_dim,
        /*o_stride_h=*/head_dim, group_size);

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
              if (partition_kv) {
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
#if (__CUDA_ARCH__ < 800)
  }
#endif
}

template <LogitsPostHook logits_post_hook, MaskMode mask_mode, PosEncodingMode pos_encoding_mode,
          uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z, uint32_t num_warps_x,
          uint32_t num_warps_z, PageStorage page_storage, typename DTypeQ, typename DTypeKV,
          typename DTypeQKAccum, typename DTypeOut, typename IdType>
__global__
__launch_bounds__(num_warps_x* num_warps_z* warp_size) void BatchPrefillWithPagedKVCacheKernel(
    IdType* __restrict__ request_indices, IdType* __restrict__ q_tile_indices,
    IdType* __restrict__ kv_tile_indices, DTypeQ* __restrict__ q,
    paged_kv_t<page_storage, DTypeKV, IdType> paged_kv, IdType* __restrict__ q_indptr,
    uint8_t* __restrict__ custom_mask, IdType* __restrict__ qk_indptr,
    IdType* __restrict__ q_offset, IdType* __restrict__ o_indptr, DTypeOut* __restrict__ o,
    float* __restrict__ lse, bool* __restrict__ block_valid_mask,
    IdType* __restrict__ kv_chunk_size_ptr, const bool partition_kv, const uint_fastdiv group_size,
    int32_t maybe_window_left, const float logits_soft_cap, float sm_scale,
    const float log2_rope_rcp_scale, const float log2_rope_rcp_theta) {
#if (__CUDA_ARCH__ < 800)
  if constexpr (std::is_same<DTypeQ, nv_bfloat16>::value) {
    FLASHINFER_RUNTIME_ASSERT("Prefill kernels do not support bf16 on sm75.");
  } else {
#endif
    static_assert(sizeof(DTypeQ) == 2);
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
    const uint32_t kv_len_safe = kv_len > 0 ? kv_len : 1;
    const uint32_t window_left = (maybe_window_left >= 0) ? maybe_window_left : kv_len;
    const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
    const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
    const uint32_t chunk_end =
        partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
    const uint32_t chunk_size = chunk_end - chunk_start;
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
    const uint32_t q_stride_n = num_qo_heads * head_dim, q_stride_h = head_dim;
    constexpr SwizzleMode swizzle_mode_q = SwizzleMode::k128B;
    smem_t<swizzle_mode_q> qo_smem(smem);
    DTypeQ* q_ptr_base = q + get_elem_offset_impl(q_indptr[request_idx], kv_head_idx * group_size,
                                                  (lane_idx % 8) * num_elems_per_128b<DTypeQ>(),
                                                  q_stride_n, q_stride_h);
    DTypeOut* o_ptr_base =
        partition_kv ? o + kv_tile_idx * num_qo_heads * head_dim +
                           get_elem_offset_impl(o_indptr[request_idx], kv_head_idx * group_size,
                                                (lane_idx % 8) * num_elems_per_128b<DTypeOut>(),
                                                num_qo_heads * head_dim, head_dim)
                     : o + get_elem_offset_impl(o_indptr[request_idx], kv_head_idx * group_size,
                                                (lane_idx % 8) * num_elems_per_128b<DTypeOut>(),
                                                num_qo_heads * head_dim, head_dim);
    uint32_t q_smem_offset_r = qo_smem.template get_permuted_offset<channel_size_128b_q>(
        get_warp_idx_x<num_warps_x, num_warps_z>() * num_frags_x * 16 + lane_idx % 16,
        lane_idx / 16);

    load_q_global_smem<num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
        qo_packed_idx_base, qo_upper_bound, q_ptr_base, q_stride_n, q_stride_h, group_size,
        &qo_smem);

    cp_async::commit_group();
    cp_async::wait_group<0>();
    block.sync();

    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      if (q_offset == nullptr) {
        q_smem_inplace_apply_rotary_multiply_sm_scale<num_warps_x, num_warps_z, num_frags_x,
                                                      num_frags_y, swizzle_mode_q, DTypeQ>(
            qo_packed_idx_base, qo_len, kv_len, group_size, &qo_smem, &q_smem_offset_r, rope_freq,
            sm_scale);
      } else {
        q_smem_inplace_apply_rotary_with_pos_multiply_sm_scale<
            num_warps_x, num_warps_z, num_frags_x, num_frags_y, swizzle_mode_q, DTypeQ>(
            qo_packed_idx_base, q_offset + q_indptr[request_idx], &qo_smem, group_size,
            &q_smem_offset_r, rope_freq, sm_scale);
      }
    } else {
      q_smem_inplace_multiply_sm_scale<num_warps_x, num_warps_z, num_frags_x, num_frags_y,
                                       swizzle_mode_q, DTypeQ>(&qo_smem, sm_scale);
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

    constexpr SwizzleMode swizzle_mode_kv =
        (sizeof(DTypeKV) == 1 && head_dim == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
    constexpr uint32_t kv_frag_rows = swizzle_mode_kv == SwizzleMode::k128B ? 4 : 8;
    constexpr uint32_t kv_frag_cols = swizzle_mode_kv == SwizzleMode::k128B ? 8 : 4;
    smem_t<swizzle_mode_kv> k_smem(smem +
                                   (num_warps_x * num_frags_x * sizeof(DTypeQ)) * 16 * head_dim),
        v_smem(smem + (num_warps_x * num_frags_x * sizeof(DTypeQ) +
                       num_warps_z * num_frags_z * sizeof(DTypeKV)) *
                          16 * head_dim);
    size_t kv_offset[num_frags_z * (swizzle_mode_kv == SwizzleMode::k128B ? 4 : 2) / num_warps_x];

    uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<channel_size_128b_kv>(
                 get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16 +
                     8 * (lane_idx / 16) + lane_idx % 8,
                 (lane_idx % 16) / 8),
             v_smem_offset_r = v_smem.template get_permuted_offset<channel_size_128b_kv>(
                 get_warp_idx_z<num_warps_x, num_warps_z>() * num_frags_z * 16 + lane_idx % 16,
                 lane_idx / 16),
             kv_smem_offset_w = k_smem.template get_permuted_offset<channel_size_128b_kv>(
                 warp_idx * kv_frag_rows + lane_idx / kv_frag_cols, lane_idx % kv_frag_cols);
    const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

    uint32_t packed_page_iter_base =
        paged_kv.indptr[request_idx] * paged_kv.page_size + chunk_start;
#pragma unroll
    for (uint32_t i = 0;
         i < num_frags_z * (swizzle_mode_kv == SwizzleMode::k128B ? 4 : 2) / num_warps_x; ++i) {
      uint32_t page_iter, entry_idx;
      paged_kv.page_size.divmod(packed_page_iter_base + warp_idx * kv_frag_rows +
                                    lane_idx / kv_frag_cols +
                                    kv_frag_rows * num_warps_x * num_warps_z * i,
                                page_iter, entry_idx);
      kv_offset[i] = page_iter < last_indptr
                         ? paged_kv.get_elem_offset(
                               __ldg(paged_kv.indices + page_iter), kv_head_idx, entry_idx,
                               (lane_idx % kv_frag_cols) * num_elems_per_128b<DTypeKV>())
                         : 0;
    }
    page_produce_kv<false, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        k_smem, &kv_smem_offset_w, paged_kv, 0, kv_offset, chunk_size);
    cp_async::commit_group();
    page_produce_kv<true, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
        v_smem, &kv_smem_offset_w, paged_kv, 0, kv_offset, chunk_size);
    cp_async::commit_group();

    const uint32_t num_iterations = ceil_div(
        (mask_mode == MaskMode::kCausal
             ? min(chunk_size,
                   sub_if_greater_or_zero(
                       kv_len - qo_len + ((qo_tile_idx + 1) * num_rows_per_cta) / group_size,
                       chunk_start))
             : chunk_size),
        16 * num_warps_z * num_frags_z);

    const uint32_t window_iteration =
        ceil_div(sub_if_greater_or_zero(kv_len + (bx + 1) * num_rows_per_cta,
                                        qo_len + window_left + chunk_start),
                 (16 * num_warps_z * num_frags_z));

    const uint32_t mask_iteration =
        (mask_mode == MaskMode::kCausal
             ? min(chunk_size, sub_if_greater_or_zero(
                                   kv_len + (qo_tile_idx * num_rows_per_cta) / group_size - qo_len,
                                   chunk_start))
             : chunk_size) /
        (16 * num_warps_z * num_frags_z);

#pragma unroll 1
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
      packed_page_iter_base += 16 * num_warps_z * num_frags_z;
#pragma unroll
      for (uint32_t i = 0;
           i < num_frags_z * (swizzle_mode_kv == SwizzleMode::k128B ? 4 : 2) / num_warps_x; ++i) {
        uint32_t page_iter, entry_idx;
        paged_kv.page_size.divmod(packed_page_iter_base + warp_idx * kv_frag_rows +
                                      lane_idx / kv_frag_cols +
                                      kv_frag_rows * num_warps_x * num_warps_z * i,
                                  page_iter, entry_idx);
        kv_offset[i] = page_iter < last_indptr
                           ? paged_kv.get_elem_offset(
                                 __ldg(paged_kv.indices + page_iter), kv_head_idx, entry_idx,
                                 (lane_idx % kv_frag_cols) * num_elems_per_128b<DTypeKV>())
                           : 0;
      }
      cp_async::wait_group<1>();
      block.sync();

      if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
        k_smem_inplace_apply_rotary<num_warps_x, num_warps_z, num_frags_y, num_frags_z,
                                    swizzle_mode_kv, DTypeKV>(
            (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[request_idx]) +
                chunk_start + iter * 16 * num_warps_z * num_frags_z,
            &k_smem, &k_smem_offset_r, rope_freq);
        block.sync();
      }

      // compute attention score
      compute_qk<logits_post_hook, num_frags_x, num_frags_y, num_frags_z, swizzle_mode_q,
                 swizzle_mode_kv, DTypeQ, DTypeKV>(&qo_smem, &q_smem_offset_r, &k_smem,
                                                   &k_smem_offset_r, s_frag, logits_soft_cap);

      if constexpr (pos_encoding_mode == PosEncodingMode::kALiBi) {
        // TODO(Zihao): handle the case that q_offset is specified
        apply_alibi_bias<num_frags_x, num_frags_z>(
            qo_packed_idx_base,
            chunk_start + (iter * num_warps_z + get_warp_idx_z<num_warps_x, num_warps_z>()) *
                              num_frags_z * 16,
            int(kv_len) - int(qo_len), group_size, alibi_slopes, s_frag);
      }
      // apply mask
      if constexpr (mask_mode == MaskMode::kCustom) {
        mask_s<mask_mode, num_frags_x, num_frags_y, num_frags_z>(
            qo_packed_idx_base,
            chunk_start + (iter * num_warps_z + get_warp_idx_z<num_warps_x, num_warps_z>()) *
                              num_frags_z * 16,
            qo_len, kv_len, window_left, chunk_end, group_size,
            custom_mask + qk_indptr[request_idx], s_frag);
      } else {
        if (iter >= mask_iteration || iter < window_iteration) {
          mask_s<mask_mode, num_frags_x, num_frags_y, num_frags_z>(
              qo_packed_idx_base,
              chunk_start + (iter * num_warps_z + get_warp_idx_z<num_warps_x, num_warps_z>()) *
                                num_frags_z * 16,
              qo_len, kv_len, window_left, chunk_end, group_size, nullptr, s_frag);
        }
      }

      // compute m,d states in online softmax
      update_mdo_states<num_frags_x, num_frags_y, num_frags_z>(s_frag, o_frag, m, d);

      block.sync();
      page_produce_kv<false, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
          k_smem, &kv_smem_offset_w, paged_kv, (iter + 1) * 16 * num_warps_z * num_frags_z,
          kv_offset, chunk_size);
      cp_async::commit_group();
      cp_async::wait_group<1>();
      block.sync();

      // compute sfm*v
      compute_sfm_v<num_frags_x, num_frags_y, num_frags_z, swizzle_mode_kv, DTypeQ, DTypeKV>(
          &v_smem, &v_smem_offset_r, s_frag, o_frag, d);

      block.sync();
      page_produce_kv<true, num_warps_x, num_warps_z, num_frags_y, num_frags_z>(
          v_smem, &kv_smem_offset_w, paged_kv, (iter + 1) * 16 * num_warps_z * num_frags_z,
          kv_offset, chunk_size);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    block.sync();

    // threadblock synchronization
    threadblock_sync_mdo_states<num_warps_x, num_warps_z, num_frags_x, num_frags_y, DTypeQKAccum>(
        o_frag, (float*)smem, m, d, warp_idx, lane_idx);

    // normalize d
    normalize_d<num_frags_x, num_frags_y>(o_frag, m, d);

    const uint32_t num_kv_chunks = (kv_len_safe + kv_chunk_size - 1) / kv_chunk_size;

    // write_back
    write_o_reg_gmem<num_warps_x, num_warps_z, num_frags_x, num_frags_y>(
        o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
        /*o_stride_n=*/
        partition_kv ? num_qo_heads * head_dim * num_kv_chunks : num_qo_heads * head_dim,
        /*o_stride_h=*/head_dim, group_size);

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
              if (partition_kv) {
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
#if (__CUDA_ARCH__ < 800)
  }
#endif
}

template <uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK, PosEncodingMode pos_encoding_mode,
          bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename DTypeQ, typename DTypeKV,
          typename DTypeOut>
cudaError_t SinglePrefillWithKVCacheDispatched(
    DTypeQ* q, DTypeKV* k, DTypeKV* v, uint8_t* custom_mask, DTypeOut* o, DTypeOut* tmp, float* lse,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    uint32_t q_stride_n, uint32_t q_stride_h, uint32_t kv_stride_n, uint32_t kv_stride_h,
    int32_t window_left, float logits_soft_cap, float sm_scale, float rope_scale, float rope_theta,
    cudaStream_t stream) {
  const float log2_rope_rcp_scale = -std::log2f(rope_scale);
  const float log2_rope_rcp_theta = -std::log2f(rope_theta);
  // TODO (yiakwy) : REMOVE
  // e.x.: q: (1/*qo_heads*/, 2/*qo_len*/, 64) kv: (1/*kv_heads*/, 2/*kv_len*/, 64)
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
  int64_t unpacked_qo_len = qo_len * group_size;
  if (unpacked_qo_len > 64 && HEAD_DIM < 256) {
    warp_layout = WarpLayout::k4x1x2;
  } else {
    auto compute_capacity = GetCudaComputeCapability();
    #ifdef USE_ROCM
      // TODO (yiakwy) : tuning warp layout, ROCM 6.2 SDK output 9.4
      if (unpacked_qo_len > 16) {
        warp_layout = WarpLayout::k4x1x1;
      } else {
        warp_layout = WarpLayout::k1x4x1;
      }
    #else
    if (compute_capacity.first >= 8) {
      // Ampere or newer
      if (unpacked_qo_len > 16) {
        warp_layout = WarpLayout::k4x1x1;
      } else {
        warp_layout = WarpLayout::k1x4x1;
      }
    } else {
      // NOTE(Zihao): not enough shared memory on Turing for 1x4x1 layout
      warp_layout = WarpLayout::k4x1x1;
    }
    #endif
  }

  DISPATCH_WARP_LAYOUT(warp_layout, WARP_LAYOUT, {
    constexpr uint32_t num_frags_x = get_num_frags_x<WARP_LAYOUT>();
    using DTypeQKAccum =
        typename std::conditional<ALLOW_FP16_QK_REDUCTION && std::is_same<DTypeQ, half>::value,
                                  half, float>::type;

    int dev_id = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    int max_smem_per_sm = 0;
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
    // we expect each sm execute two threadblocks
    // TODO(Zihao): fix the following computation
    // TODO (yiakwy) : MI300X returns 64KB (i.e.: 2**16 addresable locations) for max_smem_per_sm, note for HEAD_DIM=64, DTypeQ=half (16 * HEAD_DIM * sizeof(DTypeQ) * 16) = 2**(4 + 6 + 1 + 4) = 2**15
    const int num_ctas_per_sm = max_smem_per_sm > (16 * HEAD_DIM * sizeof(DTypeQ) * 16) ? 2 : 1;
    const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

    constexpr uint32_t num_warps_x = get_num_warps_x<WARP_LAYOUT>();
    constexpr uint32_t num_warps_z = get_num_warps_z<WARP_LAYOUT>();
    const uint32_t max_num_frags_z_reg =
        (HEAD_DIM >= 128 && num_frags_x == 2 && pos_encoding_mode == PosEncodingMode::kRoPELlama &&
         !ALLOW_FP16_QK_REDUCTION)
            ? 2
            : (8 / num_frags_x);
    // TODO(Zihao): fix the following computation
    // NOTE(yiakwy) : for HEAD_DIM=64, DTypeQ=half and num_warps_z=4, max_num_frags_z_smem=32KB / 2**(4 + 6 + 1/*dtypeQ*/ + 1 + 2/*warp_size*/) = 2**15/(2**14 - delta) = 1
    /*
    const uint32_t max_num_frags_z_smem =
        (max_smem_per_threadblock / (16 * HEAD_DIM * sizeof(DTypeQ)) - num_frags_x * num_warps_x) /
        (2 * num_warps_z);
     */
    const uint32_t max_num_frags_z_smem =
        (max_smem_per_threadblock / (16 * HEAD_DIM * sizeof(DTypeQ))) /
        (2 * num_warps_z);

    // TODO (yiakwy) : fix here
    // control num_frags_z for maximum warp occupancy
    DISPATCH_NUM_FRAGS_Z(min(max_num_frags_z_smem, max_num_frags_z_reg), num_frags_z, {
      if constexpr (is_invalid_configuration<pos_encoding_mode, DTypeKV, DTypeQKAccum>(
                        num_frags_x, num_frags_y, num_frags_z, num_warps_x, num_warps_z)) {
        // Invalid configuration, skip
        std::ostringstream err_msg;
        err_msg << "FlashInfer Internal Error: Invalid configuration : num_frags_x=" << num_frags_x
                << " num_frags_y=" << num_frags_y << " num_frags_z=" << num_frags_z
                << " num_warps_x=" << num_warps_x << " num_warps_z=" << num_warps_z
                << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                   " and report the issue to the developers.";
        throw std::invalid_argument(err_msg.str());
      } else {
        constexpr uint32_t num_threads = (num_warps_x * num_warps_z) * warp_size; // 4x1x64=256
        constexpr uint32_t num_rows_per_cta = num_frags_x * num_warps_x * 16; // 1x4x16=64
        auto kernel =
            SinglePrefillWithKVCacheKernel<LOGITS_POST_HOOK, MASK_MODE, pos_encoding_mode,
                                           num_frags_x, num_frags_y, num_frags_z, num_warps_x,
                                           num_warps_z, DTypeQ, DTypeKV, DTypeQKAccum, DTypeOut>;
        // TODO(Zihao): fix the following computation
        uint32_t smem_size = (num_frags_x * num_warps_x * sizeof(DTypeQ) +
                              num_frags_z * num_warps_z * 2 * sizeof(DTypeQ)) *
                             16 * HEAD_DIM;
        FLASHINFER_CUDA_CALL(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        int num_blocks_per_sm = 0;
        int num_sm = 0;
        FLASHINFER_CUDA_CALL(
            cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
        FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm, kernel, num_threads, smem_size));
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

        // TODO(yiakwy) : REMOVE
        std::cout << "qo_len : " << qo_len << std::endl;
        std::cout << "kv_len : " << kv_len << std::endl;

        std::cout << "num_blocks_per_sm : " << num_blocks_per_sm << std::endl;
        std::cout << "max_num_kv_chunks : " << max_num_kv_chunks << std::endl;
        std::cout << "num_chunks : " << num_chunks << std::endl;
        
        std::cout << "num_rows_per_cta : " << num_rows_per_cta << std::endl;
        std::cout << "num_threads : " << num_threads << std::endl;
        std::cout << "num_warps_x : " << num_warps_x << std::endl;
        std::cout << "num_warps_z : " << num_warps_z << std::endl;

        if (num_chunks <= 1 || tmp == nullptr) {
          // Enough parallelism, do not split-kv
          bool partition_kv = false;
          void* args[] = {(void*)&q,
                          (void*)&k,
                          (void*)&v,
                          (void*)&custom_mask,
                          (void*)&o,
                          (void*)&lse,
                          (void*)&qo_len,
                          (void*)&kv_len,
                          (void*)&partition_kv,
                          (void*)&group_size_fastdiv,
                          (void*)&q_stride_n,
                          (void*)&q_stride_h,
                          (void*)&kv_stride_n,
                          (void*)&kv_stride_h,
                          (void*)&window_left,
                          (void*)&logits_soft_cap,
                          (void*)&sm_scale,
                          (void*)&log2_rope_rcp_scale,
                          (void*)&log2_rope_rcp_theta};
          dim3 nblks(ceil_div(qo_len * group_size, num_rows_per_cta), 1, num_kv_heads);
          dim3 nthrs(32, num_warps_x, num_warps_z);

          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        } else {
          // Use cooperative groups to increase occupancy
          bool partition_kv = true;
          float* tmp_lse = (float*)(tmp + num_chunks * qo_len * num_qo_heads * HEAD_DIM);
          void* args[] = {(void*)&q,
                          (void*)&k,
                          (void*)&v,
                          (void*)&custom_mask,
                          (void*)&tmp,
                          (void*)&tmp_lse,
                          (void*)&qo_len,
                          (void*)&kv_len,
                          (void*)&partition_kv,
                          (void*)&group_size_fastdiv,
                          (void*)&q_stride_n,
                          (void*)&q_stride_h,
                          (void*)&kv_stride_n,
                          (void*)&kv_stride_h,
                          (void*)&window_left,
                          (void*)&logits_soft_cap,
                          (void*)&sm_scale,
                          (void*)&log2_rope_rcp_scale,
                          (void*)&log2_rope_rcp_theta};
          dim3 nblks(ceil_div(qo_len * group_size, num_rows_per_cta), num_chunks, num_kv_heads);
          dim3 nthrs(32, num_warps_x, num_warps_z);
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
          FLASHINFER_CUDA_CALL(MergeStates(tmp, tmp_lse, o, lse, num_chunks, qo_len, num_qo_heads,
                                           HEAD_DIM, stream));
        }
      }
    })
  });
  return cudaSuccess;
}

template <WarpLayout WARP_LAYOUT, uint32_t HEAD_DIM, LogitsPostHook LOGITS_POST_HOOK,
          PosEncodingMode pos_encoding_mode, bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename DTypeQ, typename DTypeKV, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(
    DTypeQ* q, IdType* request_indices, IdType* q_tile_indices, IdType* kv_tile_indices,
    IdType* q_indptr, DTypeKV* k, DTypeKV* v, IdType* kv_indptr, uint8_t* custom_mask,
    IdType* qk_indptr, IdType* q_offset, IdType* k_rope_pos_offset, IdType* o_indptr, DTypeOut* o,
    DTypeOut* tmp_v, float* tmp_s, float* lse, IdType* merge_indptr, bool* block_valid_mask,
    IdType* kv_chunk_size_ptr, uint32_t total_num_rows, uint32_t num_qo_heads,
    uint32_t padded_batch_size, uint32_t num_kv_heads, uint32_t q_stride_n, uint32_t q_stride_h,
    uint32_t kv_stride_n, uint32_t kv_stride_h, int32_t window_left, float logits_soft_cap,
    float sm_scale, float rope_scale, float rope_theta, cudaStream_t stream = nullptr) {
  const float log2_rope_rcp_scale = -std::log2f(rope_scale);
  const float log2_rope_rcp_theta = -std::log2f(rope_theta);
  constexpr uint32_t num_frags_x = get_num_frags_x<WARP_LAYOUT>();
  constexpr uint32_t num_warps_x = get_num_warps_x<WARP_LAYOUT>();
  constexpr uint32_t num_warps_z = get_num_warps_z<WARP_LAYOUT>();
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
  // TODO(Zihao): fix the following computation
  const int num_ctas_per_sm = max_smem_per_sm > (16 * HEAD_DIM * sizeof(DTypeQ) * 16) ? 2 : 1;
  const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

  const uint32_t max_num_frags_z_reg =
      (HEAD_DIM >= 128 && num_frags_x == 2 && pos_encoding_mode == PosEncodingMode::kRoPELlama &&
       !ALLOW_FP16_QK_REDUCTION)
          ? 2
          : (8 / num_frags_x);
  // TODO(Zihao): fix the following computation
  const uint32_t max_num_frags_z_smem =
      (max_smem_per_threadblock / (16 * HEAD_DIM * sizeof(DTypeQ)) - num_frags_x * num_warps_x) /
      (2 * num_warps_z);

  DISPATCH_NUM_FRAGS_Z(min(max_num_frags_z_smem, max_num_frags_z_reg), num_frags_z, {
    if constexpr (is_invalid_configuration<pos_encoding_mode, DTypeKV, DTypeQKAccum>(
                      num_frags_x, num_frags_y, num_frags_z, num_warps_x, num_warps_z)) {
      // Invalid configuration, skip
      std::ostringstream err_msg;
      err_msg << "FlashInfer Internal Error: Invalid configuration : num_frags_x=" << num_frags_x
              << " num_frags_y=" << num_frags_y << " num_frags_z=" << num_frags_z
              << " num_warps_x=" << num_warps_x << " num_warps_z=" << num_warps_z
              << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                 " and report the issue to the developers.";
      throw std::invalid_argument(err_msg.str());
    } else {
      // TODO(Zihao): fix the following computation
      uint32_t smem_size = (num_frags_x * num_warps_x * sizeof(DTypeQ) +
                            num_frags_z * num_warps_z * 2 * sizeof(DTypeQ)) *
                           16 * HEAD_DIM;
      auto kernel = BatchPrefillWithRaggedKVCacheKernel<
          LOGITS_POST_HOOK, MASK_MODE, pos_encoding_mode, num_frags_x, num_frags_y, num_frags_z,
          num_warps_x, num_warps_z, DTypeQ, DTypeKV, DTypeQKAccum, DTypeOut, IdType>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      if (tmp_v == nullptr) {
        // do not partition kv
        bool partition_kv = false;

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
                        (void*)&partition_kv,
                        (void*)&group_size_fastdiv,
                        (void*)&q_stride_n,
                        (void*)&q_stride_h,
                        (void*)&kv_stride_n,
                        (void*)&kv_stride_h,
                        (void*)&window_left,
                        (void*)&logits_soft_cap,
                        (void*)&sm_scale,
                        (void*)&log2_rope_rcp_scale,
                        (void*)&log2_rope_rcp_theta};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      } else {
        // partition kv
        bool partition_kv = true;
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
                        (void*)&partition_kv,
                        (void*)&group_size_fastdiv,
                        (void*)&q_stride_n,
                        (void*)&q_stride_h,
                        (void*)&kv_stride_n,
                        (void*)&kv_stride_h,
                        (void*)&window_left,
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
          LogitsPostHook LOGITS_POST_HOOK, PosEncodingMode pos_encoding_mode,
          bool ALLOW_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename DTypeQ, typename DTypeKV,
          typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(
    DTypeQ* q, IdType* request_indices, IdType* q_tile_indices, IdType* kv_tile_indices,
    IdType* q_indptr, IdType* q_offset, paged_kv_t<page_storage, DTypeKV, IdType> paged_kv,
    uint8_t* custom_mask, IdType* qk_indptr, IdType* o_indptr, DTypeOut* o, DTypeOut* tmp_v,
    float* tmp_s, float* lse, IdType* merge_indptr, bool* block_valid_mask,
    IdType* kv_chunk_size_ptr, uint32_t total_num_rows, uint32_t num_qo_heads,
    uint32_t padded_batch_size, int32_t window_left, float logits_soft_cap, float sm_scale,
    float rope_scale, float rope_theta, cudaStream_t stream) {
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
  // TODO(Zihao): fix the following computation
  const int num_ctas_per_sm = max_smem_per_sm > (16 * HEAD_DIM * sizeof(DTypeQ) * 16) ? 2 : 1;
  const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

  const uint32_t max_num_frags_z_reg =
      (HEAD_DIM >= 128 && num_frags_x == 2 && pos_encoding_mode == PosEncodingMode::kRoPELlama &&
       !ALLOW_FP16_QK_REDUCTION)
          ? 2
          : (8 / num_frags_x);
  // TODO(Zihao): fix the following computation
  // NOTE (yiakwy) : fix max_num_frags_z_smem
  /*
  const uint32_t max_num_frags_z_smem =
      (max_smem_per_threadblock / (16 * HEAD_DIM * sizeof(DTypeQ)) - num_frags_x * num_warps_x) /
      (2 * num_warps_z);
   */
  const uint32_t max_num_frags_z_smem =
      (max_smem_per_threadblock / (16 * HEAD_DIM * sizeof(DTypeQ))) /
      (2 * num_warps_z);

  DISPATCH_NUM_FRAGS_Z(min(max_num_frags_z_smem, max_num_frags_z_reg), num_frags_z, {
    if constexpr (is_invalid_configuration<pos_encoding_mode, DTypeKV, DTypeQKAccum>(
                      num_frags_x, num_frags_y, num_frags_z, num_warps_x, num_warps_z)) {
      // Invalid configuration, skip
      std::ostringstream err_msg;
      err_msg << "FlashInfer Internal Error: Invalid configuration : num_frags_x=" << num_frags_x
              << " num_frags_y=" << num_frags_y << " num_frags_z=" << num_frags_z
              << " num_warps_x=" << num_warps_x << " num_warps_z=" << num_warps_z
              << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                 " and report the issue to the developers.";
      throw std::invalid_argument(err_msg.str());
    } else {
      // TODO(Zihao): fix the following computation
      uint32_t smem_size = (num_frags_x * num_warps_x * sizeof(DTypeQ) +
                            num_frags_z * num_warps_z * 2 * sizeof(DTypeQ)) *
                           16 * HEAD_DIM;
      auto kernel = BatchPrefillWithPagedKVCacheKernel<
          LOGITS_POST_HOOK, MASK_MODE, pos_encoding_mode, num_frags_x, num_frags_y, num_frags_z,
          num_warps_x, num_warps_z, page_storage, DTypeQ, DTypeKV, DTypeQKAccum, DTypeOut, IdType>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      if (tmp_v == nullptr) {
        // do not partition kv
        bool partition_kv = false;
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
                        (void*)&partition_kv,
                        (void*)&group_size_fastdiv,
                        (void*)&window_left,
                        (void*)&logits_soft_cap,
                        (void*)&sm_scale,
                        (void*)&log2_rope_rcp_scale,
                        (void*)&log2_rope_rcp_theta};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      } else {
        bool partition_kv = true;
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
                        (void*)&partition_kv,
                        (void*)&group_size_fastdiv,
                        (void*)&window_left,
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
