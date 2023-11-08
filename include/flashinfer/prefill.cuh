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
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "cp_async.cuh"
#include "layout.cuh"
#include "math.cuh"
#include "mma.cuh"
#include "page.cuh"
#include "permuted_smem.cuh"
#include "rope.cuh"
#include "state.cuh"
#include "utils.cuh"

namespace flashinfer {

namespace cg = cooperative_groups;

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
 * \tparam row_major Whether the input fragments are row major or not.
 * \tparam T The data type of the input fragments.
 * \param x_first_half First fragment x[offset:offset+16, j*16:(j+1)*16]
 * \param x_second_half Second fragment x[offset:offset*16, j*16+d/2:(j+1)*16+d/2]
 * \param rope_freq Rope frequency
 * \param offset The offset of the first row in both fragments.
 * \param scale A scale factor applied to the result (used to multiply sm_scale).
 * \note The sin/cos computation is slow, especially for A100 GPUs which has low
 *   non tensor-ops flops, will optimize in the future.
 */
template <bool row_major, typename T>
__device__ __forceinline__ void apply_llama_rope(T* x_first_half, T* x_second_half,
                                                 const float* rope_freq, uint32_t offset,
                                                 float scale = 1.f) {
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    uint32_t i, j;
    if constexpr (row_major) {
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
    __sincosf(float(offset + 8 * i) * rope_freq[2 * j + reg_id % 2], &sin, &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin) * scale;
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin) * scale;
  }
}

}  // namespace

/*!
 * \brief Produce k/v fragments from global memory to shared memory.
 * \tparam fill_zero Whether to fill zero for out-of-boundary elements in pred_load.
 * \tparam num_frags_y The number of fragments in y dimension.
 * \tparam num_frags_z The number of fragments in z dimension.
 * \tparam num_warps The number of warps in the threadblock.
 * \tparam layout The layout of the input tensor.
 * \tparam group_size The number of qo heads that maps to a kv head (used in GQA).
 * \tparam T The data type of the input tensor.
 * \param smem The shared memory to store kv fragments.
 * \param gptr The global memory pointer.
 * \param qkv_info The tensor info of the input tensor.
 * \param kv_idx_base The base kv index.
 * \param kv_len The length of kv tensor.
 */
template <bool fill_zero, uint32_t num_frags_y, uint32_t num_frags_z, uint32_t num_warps,
          QKVLayout layout, uint32_t group_size, typename T>
__device__ __forceinline__ void produce_kv(smem_t smem, uint32_t* smem_offset, T** gptr,
                                           const tensor_info_t<layout, group_size>& qkv_info,
                                           const uint32_t kv_idx_base, const uint32_t kv_len) {
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t num_cells_per_in_channel = num_frags_y * 16 / cell_capacity<T>();

  uint32_t kv_idx = kv_idx_base + ty * 4 + tx / 8;

#pragma unroll
  for (uint32_t i = 0; i < num_frags_z * 4 / num_warps; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < num_frags_y / 4; ++j) {
      smem.load_128b_async<fill_zero>(*smem_offset, *gptr, kv_idx < kv_len);
      *smem_offset += 16;
      *gptr += 8 * cell_capacity<T>();
    }
    kv_idx += num_warps * 4;
    *smem_offset += num_warps * 4 * num_cells_per_in_channel - 4 * num_frags_y;
    *gptr += num_warps * 4 * qkv_info.get_n_stride() - 2 * num_frags_y * cell_capacity<T>();
  }
  *smem_offset -= num_frags_z * 16 * num_cells_per_in_channel;
}

/*!
 * \brief FlashAttention prefill CUDA kernel for a single request.
 * \tparam cooperative Whether to use cooperative kernel (split kv_len into chunks).
 * \tparam group_size The number of qo heads that maps to a kv head (used in GQA).
 * \tparam causal Whether to use causal attention.
 * \tparam layout The layout of the input tensor.
 * \tparam rotary_mode The rotary mode.
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
 * \param tmp The temporary buffer (used in cooperative kernels).
 * \param qkv_info The tensor info of the input tensor.
 * \param sm_scale The scale factor applied to the softmax score.
 * \param log2_rope_rcp_scale log2(1/(rope_scale)), where rope_scale is the scaling
 *   factor used in RoPE interpolation.
 * \param log2_rope_rcp_theta log2(1/(rope_theta)), where rope_theta is the theta
 *   used in RoPE.
 */
template <bool cooperative, uint32_t group_size, bool causal, QKVLayout layout,
          RotaryMode rotary_mode, uint32_t num_frags_x, uint32_t num_frags_y, uint32_t num_frags_z,
          uint32_t num_warps, typename DTypeIn, typename DTypeOut>
__global__ void SinglePrefillWithKVCacheKernel(DTypeIn* __restrict__ q, DTypeIn* __restrict__ k,
                                               DTypeIn* __restrict__ v, DTypeOut* __restrict__ o,
                                               float* __restrict__ tmp,
                                               const tensor_info_t<layout, group_size> qkv_info,
                                               float sm_scale, const float log2_rope_rcp_scale,
                                               const float log2_rope_rcp_theta) {
  static_assert(sizeof(DTypeIn) == 2);
  static_assert(sizeof(DTypeOut) == 2);
  sm_scale *= math::log2e;
  const uint32_t qo_len = qkv_info.qo_len;
  const uint32_t kv_len = qkv_info.kv_len;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bx = blockIdx.x, chunk_idx = blockIdx.y, qo_head_idx = blockIdx.z,
                 kv_head_idx = qo_head_idx / group_size;
  const uint32_t num_chunks = gridDim.y;
  const uint32_t chunk_size = cooperative ? (kv_len + num_chunks - 1) / num_chunks : kv_len;
  const uint32_t chunk_start = cooperative ? chunk_idx * chunk_size : 0;
  const uint32_t chunk_end = cooperative ? min((chunk_idx + 1) * chunk_size, kv_len) : kv_len;
  auto block = cg::this_thread_block();

  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_cells_per_in_channel = head_dim / cell_capacity<DTypeIn>();
  constexpr uint32_t num_cells_per_out_channel = head_dim / cell_capacity<DTypeOut>();

  static_assert(num_frags_z * num_frags_y % num_warps == 0);

  extern __shared__ uint8_t smem[];

  float x_frag[num_frags_x][num_frags_z][8];
  float o_frag[num_frags_x][num_frags_y][8];
  float m[num_frags_x][2];
  float d[num_frags_x][2];
  float rope_freq[num_frags_y / 2][4];
  if constexpr (rotary_mode == RotaryMode::kLlama) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y / 2; ++fy) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        rope_freq[fy][j] = math::ptx_exp2(
            log2_rope_rcp_scale +
            log2_rope_rcp_theta *
                float(2 * ((fy * 16 + (j / 2) * 8 + (tx % 4) * 2 + (j % 2)) % (head_dim / 2))) /
                float(head_dim));
      }
    }
  }
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
      m[fx][j] = -5e4;
      d[fx][j] = 0.f;
    }
  }

  // cooperative fetch q fragment from gmem to reg
  smem_t q_smem(smem);
  uint32_t q_idx = (bx * num_warps + ty) * num_frags_x * 16 + tx / 4;
  uint32_t q_smem_offset_w =
      smem_t::get_permuted_offset<num_cells_per_in_channel>(ty * num_frags_x * 16 + tx / 4, tx % 4);
  uint32_t q_smem_offset_r = smem_t::get_permuted_offset<num_cells_per_in_channel>(
      ty * num_frags_x * 16 + tx % 16, tx / 16);
  DTypeIn* q_ptr =
      q + qkv_info.get_qo_elem_offset(q_idx, qo_head_idx, (tx % 4) * cell_capacity<DTypeIn>());
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
#pragma unroll
      for (uint32_t fyo = 0; fyo < num_frags_y / 2; ++fyo) {
        // load q fragment from gmem to smem
        q_smem.load_128b_async<false>(q_smem_offset_w, q_ptr, q_idx < qo_len);
        q_smem_offset_w += 8;
        q_ptr += 4 * cell_capacity<DTypeIn>();
      }
      q_idx += 8;
      q_smem_offset_w += 8 * num_cells_per_in_channel - 4 * num_frags_y;
      q_ptr += 8 * qkv_info.get_n_stride() - 2 * num_frags_y * cell_capacity<DTypeIn>();
    }
  }
  cp_async::commit_group();
  cp_async::wait_group<0>();
  block.sync();

  if constexpr (rotary_mode == RotaryMode::kLlama) {
    uint32_t q_frag_local[2][4];
    q_idx = (bx * num_warps + ty) * num_frags_x * 16 + tx / 4;
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fyi = 0; fyi < num_frags_y / 2; ++fyi) {
        q_smem.ldmatrix_m8n8x4(q_smem_offset_r, q_frag_local[0]);
        q_smem_offset_r += num_frags_y * 2;
        q_smem.ldmatrix_m8n8x4(q_smem_offset_r, q_frag_local[1]);
        apply_llama_rope<true, DTypeIn>((DTypeIn*)q_frag_local[0], (DTypeIn*)q_frag_local[1],
                                        rope_freq[fyi], q_idx + kv_len - qo_len, sm_scale);
        q_smem.stmatrix_m8n8x4(q_smem_offset_r, q_frag_local[1]);
        q_smem_offset_r -= num_frags_y * 2;
        q_smem.stmatrix_m8n8x4(q_smem_offset_r, q_frag_local[0]);
        q_smem_offset_r = (q_smem_offset_r ^ 0x2) + (fyi & 0x1) * 8;
      }
      q_smem_offset_r += 16 * num_cells_per_in_channel - 2 * num_frags_y;
      q_idx += 16;
    }
    q_smem_offset_r -= num_frags_x * 16 * num_cells_per_in_channel;
  } else {
#pragma unroll
    for (uint32_t i = 0; i < num_frags_x * 16 * head_dim / 256; ++i) {
      vec_t<DTypeIn, 8> tmp;
      tmp.load((DTypeIn*)(q_smem.base + ty * num_frags_x * 16 * num_cells_per_in_channel) +
               i * 256 + tx * 8);
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        tmp[reg_id] *= sm_scale;
      }
      tmp.store((DTypeIn*)(q_smem.base + ty * num_frags_x * 16 * num_cells_per_in_channel) +
                i * 256 + tx * 8);
    }
  }

  smem_t k_smem(smem + (num_warps * num_frags_x) * 16 * head_dim * sizeof(DTypeIn)),
      v_smem(smem + (num_warps * num_frags_x + num_frags_z) * 16 * head_dim * sizeof(DTypeIn));

  const uint32_t num_iterations =
      ((causal ? min(chunk_end - chunk_start,
                     sub_if_greater_or_zero(
                         kv_len - qo_len + ((bx + 1) * num_frags_x * num_warps) * 16, chunk_start))
               : chunk_end - chunk_start) +
       16 * num_frags_z - 1) /
      (16 * num_frags_z);

  const uint32_t mask_iteration =
      (causal ? min(chunk_end - chunk_start,
                    sub_if_greater_or_zero(kv_len + bx * num_warps * num_frags_x - qo_len,
                                           chunk_start))
              : (chunk_end - chunk_start)) /
      (16 * num_frags_z);

  DTypeIn* k_ptr = k + qkv_info.get_kv_elem_offset(chunk_start + ty * 4 + tx / 8, kv_head_idx,
                                                   (tx % 8) * cell_capacity<DTypeIn>());
  DTypeIn* v_ptr = v + qkv_info.get_kv_elem_offset(chunk_start + ty * 4 + tx / 8, kv_head_idx,
                                                   (tx % 8) * cell_capacity<DTypeIn>());
  uint32_t k_smem_offset_r = smem_t::get_permuted_offset<num_cells_per_in_channel>(
               8 * (tx / 16) + tx % 8, (tx % 16) / 8),
           v_smem_offset_r =
               smem_t::get_permuted_offset<num_cells_per_in_channel>(tx % 16, tx / 16),
           kv_smem_offset_w =
               smem_t::get_permuted_offset<num_cells_per_in_channel>(ty * 4 + tx / 8, tx % 8);
  produce_kv<false, num_frags_y, num_frags_z, num_warps>(k_smem, &kv_smem_offset_w, &k_ptr,
                                                         qkv_info, chunk_start, chunk_end);
  cp_async::commit_group();
  produce_kv<true, num_frags_y, num_frags_z, num_warps>(v_smem, &kv_smem_offset_w, &v_ptr, qkv_info,
                                                        chunk_start, chunk_end);
  cp_async::commit_group();

#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    cp_async::wait_group<1>();
    block.sync();

    if constexpr (rotary_mode == RotaryMode::kLlama) {
      uint32_t k_frag_local[2][4];
      uint32_t kv_idx = chunk_start + (iter * num_frags_z + ty / 2) * 16 + tx / 4;
      k_smem_offset_r =
          (k_smem_offset_r ^ (0x2 * (ty % 2))) + (ty / 2) * 16 * num_cells_per_in_channel;
#pragma unroll
      for (uint32_t i = 0; i < num_frags_z / 2; ++i) {
        // uint32_t fz = ty / 2 + i * 2;
#pragma unroll
        for (uint32_t j = 0; j < num_frags_y / 4; ++j) {
          uint32_t fyi = (ty % 2) + j * 2;
          k_smem.ldmatrix_m8n8x4(k_smem_offset_r, k_frag_local[0]);
          k_smem_offset_r += num_frags_y * 2;
          k_smem.ldmatrix_m8n8x4(k_smem_offset_r, k_frag_local[1]);
          apply_llama_rope<false, DTypeIn>((DTypeIn*)k_frag_local[0], (DTypeIn*)k_frag_local[1],
                                           rope_freq[fyi], kv_idx);
          k_smem.stmatrix_m8n8x4(k_smem_offset_r, k_frag_local[1]);
          k_smem_offset_r -= num_frags_y * 2;
          k_smem.stmatrix_m8n8x4(k_smem_offset_r, k_frag_local[0]);
          k_smem_offset_r += 8;
        }
        k_smem_offset_r += 32 * num_cells_per_in_channel - 2 * num_frags_y;
        kv_idx += 32;
      }
      k_smem_offset_r = (k_smem_offset_r ^ (0x2 * (ty % 2))) -
                        ((ty / 2) + num_frags_z) * 16 * num_cells_per_in_channel;
      block.sync();
    }

    // rotary_mode == RotaryMode::kNone
    uint32_t a_frag[num_frags_x][4], b_frag[4];

    // compute q*k^T
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
        q_smem.ldmatrix_m8n8x4(q_smem_offset_r, a_frag[fx]);
        q_smem_offset_r += 16 * num_cells_per_in_channel;
      }
      q_smem_offset_r =
          (q_smem_offset_r ^ 0x2) + (fy & 0x1) * 8 - num_frags_x * 16 * num_cells_per_in_channel;

#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
        k_smem.ldmatrix_m8n8x4(k_smem_offset_r, b_frag);
        k_smem_offset_r += 16 * num_cells_per_in_channel;
#pragma unroll
        for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
          if (fy == 0) {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn, true>(x_frag[fx][fz], a_frag[fx],
                                                                     b_frag);
          } else {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn>(x_frag[fx][fz], a_frag[fx], b_frag);
          }
        }
      }
      k_smem_offset_r =
          (k_smem_offset_r ^ 0x2) + (fy & 0x1) * 8 - num_frags_z * 16 * num_cells_per_in_channel;
    }
    q_smem_offset_r -= num_frags_y * 4;
    k_smem_offset_r -= num_frags_y * 4;

    // apply mask
    if (iter >= mask_iteration) {
      uint32_t q_idx_base = ((bx * num_warps + ty) * num_frags_x) * 16 + tx / 4,
               kv_idx_base = chunk_start + iter * 16 * num_frags_z + 2 * (tx % 4);
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
          for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
            const uint32_t q_idx = q_idx_base + 8 * ((reg_id % 4) / 2),
                           kv_idx = kv_idx_base + 8 * (reg_id / 4) + reg_id % 2;
            const bool out_of_boundary =
                (causal ? (kv_idx > kv_len + q_idx - qo_len || (cooperative && kv_idx >= chunk_end))
                        : kv_idx >= chunk_end);
            x_frag[fx][fz][reg_id] = out_of_boundary ? -5e4 : x_frag[fx][fz][reg_id];
          }
          kv_idx_base += 16;
        }
        kv_idx_base -= num_frags_z * 16;
        q_idx_base += 16;
      }
    }

    // compute m,d states in online softmax
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        float m_prev = m[fx][j];
#pragma unroll
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          float m_local = max(max(x_frag[fx][fz][j * 2 + 0], x_frag[fx][fz][j * 2 + 1]),
                              max(x_frag[fx][fz][j * 2 + 4], x_frag[fx][fz][j * 2 + 5]));
          m[fx][j] = max(m[fx][j], m_local);
        }
        m[fx][j] = max(m[fx][j], math::shfl_xor_sync(m[fx][j], 0x2));
        m[fx][j] = max(m[fx][j], math::shfl_xor_sync(m[fx][j], 0x1));
#pragma unroll
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          x_frag[fx][fz][j * 2 + 0] = math::ptx_exp2(x_frag[fx][fz][j * 2 + 0] - m[fx][j]);
          x_frag[fx][fz][j * 2 + 1] = math::ptx_exp2(x_frag[fx][fz][j * 2 + 1] - m[fx][j]);
          x_frag[fx][fz][j * 2 + 4] = math::ptx_exp2(x_frag[fx][fz][j * 2 + 4] - m[fx][j]);
          x_frag[fx][fz][j * 2 + 5] = math::ptx_exp2(x_frag[fx][fz][j * 2 + 5] - m[fx][j]);
        }
        float o_scale = math::ptx_exp2(m_prev - m[fx][j]);
        d[fx][j] *= o_scale;
#pragma unroll
        for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
          o_frag[fx][fy][j * 2 + 0] *= o_scale;
          o_frag[fx][fy][j * 2 + 1] *= o_scale;
          o_frag[fx][fy][j * 2 + 4] *= o_scale;
          o_frag[fx][fy][j * 2 + 5] *= o_scale;
        }
      }
    }

    block.sync();
    produce_kv<false, num_frags_y, num_frags_z, num_warps>(
        k_smem, &kv_smem_offset_w, &k_ptr, qkv_info, chunk_start + (iter + 1) * 16 * num_frags_z,
        chunk_end);
    cp_async::commit_group();
    cp_async::wait_group<1>();
    block.sync();

    __threadfence_block();

    // compute sfm*v
#pragma unroll
    for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        uint32_t b_frag[4];
        v_smem.ldmatrix_m8n8x4_trans(v_smem_offset_r, b_frag);
#pragma unroll
        for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
          uint32_t a_frag[4];
#pragma unroll
          for (uint32_t reg_id = fy * (8 / num_frags_y); reg_id < (fy + 1) * (8 / num_frags_y);
               ++reg_id) {
            d[fx][(reg_id % 4) / 2] += x_frag[fx][fz][reg_id];
          }
          vec_cast<DTypeIn, float, 8>((DTypeIn*)a_frag, x_frag[fx][fz]);
          mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn>(o_frag[fx][fy], a_frag, b_frag);
        }
        v_smem_offset_r = (v_smem_offset_r ^ 0x2) + (fy & 0x1) * 8;
      }
      v_smem_offset_r += 16 * num_cells_per_in_channel - 4 * num_frags_y;
    }
    v_smem_offset_r -= 16 * num_frags_z * num_cells_per_in_channel;
    block.sync();
    produce_kv<true, num_frags_y, num_frags_z, num_warps>(
        v_smem, &kv_smem_offset_w, &v_ptr, qkv_info, chunk_start + (iter + 1) * 16 * num_frags_z,
        chunk_end);
    cp_async::commit_group();
  }
  cp_async::wait_group<0>();
  block.sync();

#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      d[fx][j] += math::shfl_xor_sync(d[fx][j], 0x2);
      d[fx][j] += math::shfl_xor_sync(d[fx][j], 0x1);
    }
  }

  if constexpr (cooperative) {
    // aggregate global state
    auto grid = cg::this_grid();
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
    float* tmp_md = tmp + num_frags_x * num_frags_y * 8 * grid.size();
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        *(float2*)&tmp_md[((fx * 2 + j) * grid.size() + grid.thread_rank()) * 2] =
            make_float2(m[fx][j], d[fx][j]);
        m[fx][j] = -5e4;
        d[fx][j] = 0.f;
      }
    }

    grid.sync();

    for (uint32_t iter = 0; iter < num_chunks; ++iter) {
      float other_scale[num_frags_x][2];
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          float2 md = *(float2*)&tmp_md[((fx * 2 + j) * grid.size() +
                                         ((qo_head_idx * num_chunks + iter) * gridDim.x + bx) *
                                             block.num_threads() +
                                         block.thread_rank()) *
                                        2];
          float mi = md.x, di = md.y, m_prev = m[fx][j];
          m[fx][j] = max(m_prev, mi);
          float o_scale = math::ptx_exp2(m_prev - m[fx][j]);
          other_scale[fx][j] = math::ptx_exp2(mi - m[fx][j]);
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
          o_frag_i.load(
              tmp + ((fx * num_frags_y + fy) * grid.size() +
                     ((qo_head_idx * num_chunks + iter) * gridDim.x + bx) * block.num_threads() +
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

  // divide d
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        o_frag[fx][fy][reg_id] = __fdividef(o_frag[fx][fy][reg_id], d[fx][(reg_id % 4) / 2]);
      }
    }
  }

  // write back
  smem_t o_smem(smem);
  uint32_t o_smem_offset_w =
      smem_t::get_permuted_offset<num_cells_per_out_channel>(ty * num_frags_x * 16 + tx / 4, 0);
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
      vec_cast<DTypeOut, float, 2>((DTypeOut*)(o_smem.base + o_smem_offset_w) + (tx % 4) * 2,
                                   &o_frag[fx][fy][0]);
      vec_cast<DTypeOut, float, 2>(
          (DTypeOut*)(o_smem.base + o_smem_offset_w + 8 * num_cells_per_out_channel) + (tx % 4) * 2,
          &o_frag[fx][fy][2]);
      vec_cast<DTypeOut, float, 2>(
          (DTypeOut*)(o_smem.base + (o_smem_offset_w ^ 0x1)) + (tx % 4) * 2, &o_frag[fx][fy][4]);
      vec_cast<DTypeOut, float, 2>(
          (DTypeOut*)(o_smem.base + (o_smem_offset_w ^ 0x1) + 8 * num_cells_per_out_channel) +
              (tx % 4) * 2,
          &o_frag[fx][fy][6]);
      o_smem_offset_w = (o_smem_offset_w ^ 0x2) + (fy & 0x1) * 8;
    }
    o_smem_offset_w += 16 * num_cells_per_out_channel - num_frags_y * 4;
  }

  o_smem_offset_w = smem_t::get_permuted_offset<num_cells_per_out_channel>(
      ty * num_frags_x * 16 + tx / 4, tx % 4);
  uint32_t o_idx = (bx * num_warps + ty) * num_frags_x * 16 + tx / 4;
  DTypeOut* o_ptr =
      o + qkv_info.get_qo_elem_offset(o_idx, qo_head_idx, (tx % 4) * cell_capacity<DTypeOut>());
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
#pragma unroll
      for (uint32_t fyo = 0; fyo < num_frags_y / 2; ++fyo) {
        if (o_idx < qo_len) {
          o_smem.store_128b(o_smem_offset_w, o_ptr);
        }
        o_ptr += 4 * cell_capacity<DTypeOut>();
        o_smem_offset_w += 8;
      }
      o_idx += 8;
      o_smem_offset_w += 8 * num_cells_per_out_channel - 4 * num_frags_y;
      o_ptr += 8 * qkv_info.get_n_stride() - 2 * num_frags_y * cell_capacity<DTypeOut>();
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
 * \param layout The layout of input and output.
 * \param rotary_mode The rotary mode.
 * \param stream The cuda stream to execute the kernel on.
 * \return status Indicates whether CUDA calls are successful
 */
template <typename DTypeIn, typename DTypeOut>
cudaError_t SinglePrefillWithKVCacheWorkEstimation(uint32_t& tmp_size, uint32_t& max_grid_size,
                                                   uint32_t num_qo_heads, uint32_t num_kv_heads,
                                                   uint32_t qo_len, uint32_t kv_len,
                                                   uint32_t head_dim, bool causal = true,
                                                   QKVLayout layout = QKVLayout::kNHD,
                                                   RotaryMode rotary_mode = RotaryMode::kNone,
                                                   cudaStream_t stream = nullptr) {
  assert(kv_len >= qo_len);

  SWITCH_NUM_FRAGS_X(
      qo_len > 64, num_frags_x,
      {SWITCH_GQA_GROUP_SIZE(
          num_qo_heads / num_kv_heads, GROUP_SIZE,
          {SWITCH_CAUSAL(
              causal, CAUSAL, {SWITCH_HEAD_DIM_PREFILL(head_dim, HEAD_DIM, {
                constexpr uint32_t num_frags_y = HEAD_DIM / 16;
                SWITCH_ROTARY_MODE(
                    rotary_mode, ROTARY_MODE, {SWITCH_LAYOUT(layout, LAYOUT, {
                      int dev_id = 0;
                      FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
                      int max_smem_per_sm = 0;
                      FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
                          &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
                      // we expect each sm execute two threadblocks
                      const int max_smem_per_threadblock = max_smem_per_sm / 2;

                      constexpr uint32_t num_warps = 4UL;
                      const uint32_t max_num_frags_z_reg =
                          (HEAD_DIM == 128 && num_frags_x == 2 && ROTARY_MODE == RotaryMode::kLlama)
                              ? 2
                              : 4;
                      const uint32_t max_num_frags_z_smem =
                          (max_smem_per_threadblock / (16 * head_dim * sizeof(DTypeIn)) -
                           num_frags_x * num_warps) /
                          2;

                      // control num_frags_z for maximum warp occupancy
                      SWITCH_NUM_FRAGS_Z(
                          min(max_num_frags_z_smem, max_num_frags_z_reg), num_frags_z, {
                            constexpr uint32_t num_threads = num_warps * warp_size;
                            constexpr uint32_t num_rows_per_cta = num_frags_x * num_warps * 16;
                            auto cooperative_kernel = SinglePrefillWithKVCacheKernel<
                                true, GROUP_SIZE, CAUSAL, LAYOUT, ROTARY_MODE, num_frags_x,
                                num_frags_y, num_frags_z, num_warps, DTypeIn, DTypeOut>;
                            tensor_info_t<LAYOUT, GROUP_SIZE> qkv_info(qo_len, kv_len, num_kv_heads,
                                                                       HEAD_DIM);
                            uint32_t smem_size = (num_frags_x * num_warps + num_frags_z * 2) * 16 *
                                                 head_dim * sizeof(DTypeIn);
                            FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                                cooperative_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                smem_size));
                            int num_blocks_per_sm = 0;
                            int num_sm = 0;
                            FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
                                &num_sm, cudaDevAttrMultiProcessorCount, dev_id));
                            FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                &num_blocks_per_sm, cooperative_kernel, num_threads, smem_size));
                            uint32_t num_chunks =
                                min((num_blocks_per_sm * num_sm) /
                                        (num_qo_heads * (qo_len + (num_rows_per_cta - 1)) /
                                         num_rows_per_cta),
                                    kv_len / 512);

                            max_grid_size = num_blocks_per_sm * num_sm;
                            if (num_chunks > 1) {
                              uint32_t grid_size =
                                  32 * num_warps *
                                  ((qo_len + (num_rows_per_cta - 1)) / num_rows_per_cta) *
                                  num_chunks * num_qo_heads;
                              tmp_size = sizeof(float) *
                                         (4 * num_frags_x + num_frags_x * num_frags_y * 8) *
                                         grid_size;
                            } else {
                              tmp_size = 0;
                            }
                          })
                    })})
              })})})});
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
 * \param num_qo_heads The number of query and output heads.
 * \param num_kv_heads The number of key and value heads.
 * \param qo_len The length of query and output.
 * \param kv_len The length of key and value.
 * \param head_dim The dimension of each head.
 * \param causal Whether to use causal attention.
 * \param layout The layout of input and output.
 * \param rotary_mode The rotary mode.
 * \param rope_scale The scaling factor used in RoPE interpolation.
 * \param rope_theta The theta used in RoPE.
 * \param stream The cuda stream to execute the kernel on.
 * \return status Indicates whether CUDA calls are successful
 */
template <typename DTypeIn, typename DTypeOut>
cudaError_t SinglePrefillWithKVCache(DTypeIn* q, DTypeIn* k, DTypeIn* v, DTypeOut* o, float* tmp,
                                     uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t qo_len,
                                     uint32_t kv_len, uint32_t head_dim, bool causal = true,
                                     QKVLayout layout = QKVLayout::kNHD,
                                     RotaryMode rotary_mode = RotaryMode::kNone,
                                     float rope_scale = 1.f, float rope_theta = 1e4,
                                     cudaStream_t stream = nullptr) {
  const float sm_scale = 1.f / std::sqrt(float(head_dim));
  const float log2_rope_rcp_scale = -std::log2f(rope_scale);
  const float log2_rope_rcp_theta = -std::log2f(rope_theta);
  assert(kv_len >= qo_len);

  SWITCH_NUM_FRAGS_X(
      qo_len > 64, num_frags_x,
      {SWITCH_GQA_GROUP_SIZE(
          num_qo_heads / num_kv_heads, GROUP_SIZE,
          {SWITCH_CAUSAL(
              causal, CAUSAL, {SWITCH_HEAD_DIM_PREFILL(head_dim, HEAD_DIM, {
                constexpr uint32_t num_frags_y = HEAD_DIM / 16;
                SWITCH_ROTARY_MODE(
                    rotary_mode, ROTARY_MODE, {SWITCH_LAYOUT(layout, LAYOUT, {
                      int dev_id = 0;
                      FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
                      int max_smem_per_sm = 0;
                      FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
                          &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
                      // we expect each sm execute two threadblocks
                      const int max_smem_per_threadblock = max_smem_per_sm / 2;

                      constexpr uint32_t num_warps = 4UL;
                      const uint32_t max_num_frags_z_reg =
                          (HEAD_DIM == 128 && num_frags_x == 2 && ROTARY_MODE == RotaryMode::kLlama)
                              ? 2
                              : 4;
                      const uint32_t max_num_frags_z_smem =
                          (max_smem_per_threadblock / (16 * head_dim * sizeof(DTypeIn)) -
                           num_frags_x * num_warps) /
                          2;

                      // control num_frags_z for maximum warp occupancy
                      SWITCH_NUM_FRAGS_Z(
                          min(max_num_frags_z_smem, max_num_frags_z_reg), num_frags_z, {
                            constexpr uint32_t num_threads = num_warps * warp_size;
                            constexpr uint32_t num_rows_per_cta = num_frags_x * num_warps * 16;
                            auto cooperative_kernel = SinglePrefillWithKVCacheKernel<
                                true, GROUP_SIZE, CAUSAL, LAYOUT, ROTARY_MODE, num_frags_x,
                                num_frags_y, num_frags_z, num_warps, DTypeIn, DTypeOut>;
                            tensor_info_t<LAYOUT, GROUP_SIZE> qkv_info(qo_len, kv_len, num_kv_heads,
                                                                       HEAD_DIM);
                            uint32_t smem_size = (num_frags_x * num_warps + num_frags_z * 2) * 16 *
                                                 head_dim * sizeof(DTypeIn);
                            FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                                cooperative_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                smem_size));
                            int num_blocks_per_sm = 0;
                            int num_sm = 0;
                            FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
                                &num_sm, cudaDevAttrMultiProcessorCount, dev_id));
                            FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                &num_blocks_per_sm, cooperative_kernel, num_threads, smem_size));
                            uint32_t num_chunks =
                                min((num_blocks_per_sm * num_sm) /
                                        (num_qo_heads * (qo_len + (num_rows_per_cta - 1)) /
                                         num_rows_per_cta),
                                    kv_len / 512);

                            if (num_chunks <= 1 || tmp == nullptr) {
                              // Enough parallelism, do not use cooperative
                              // groups
                              auto kernel = SinglePrefillWithKVCacheKernel<
                                  false, GROUP_SIZE, CAUSAL, LAYOUT, ROTARY_MODE, num_frags_x,
                                  num_frags_y, num_frags_z, num_warps, DTypeIn, DTypeOut>;
                              void* args[] = {(void*)&q,
                                              (void*)&k,
                                              (void*)&v,
                                              (void*)&o,
                                              (void*)&tmp,
                                              (void*)&qkv_info,
                                              (void*)&sm_scale,
                                              (void*)&log2_rope_rcp_scale,
                                              (void*)&log2_rope_rcp_theta};
                              dim3 nblks((qo_len + (num_rows_per_cta - 1)) / num_rows_per_cta, 1,
                                         num_qo_heads);
                              dim3 nthrs(32, num_warps);
                              FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                                  kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                              FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs,
                                                                    args, smem_size, stream));
                            } else {
                              // Use cooperative groups to increase occupancy
                              void* args[] = {(void*)&q,
                                              (void*)&k,
                                              (void*)&v,
                                              (void*)&o,
                                              (void*)&tmp,
                                              (void*)&qkv_info,
                                              (void*)&sm_scale,
                                              (void*)&log2_rope_rcp_scale,
                                              (void*)&log2_rope_rcp_theta};
                              dim3 nblks((qo_len + (num_rows_per_cta - 1)) / num_rows_per_cta,
                                         num_chunks, num_qo_heads);
                              dim3 nthrs(32, num_warps);
                              FLASHINFER_CUDA_CALL(
                                  cudaLaunchCooperativeKernel((void*)cooperative_kernel, nblks,
                                                              nthrs, args, smem_size, stream));
                            }
                          })
                    })})
              })})})});
  return cudaSuccess;
}

/*!
 * \brief FlashAttention prefill CUDA function for multiple requests.
 * \tparam DTypeIn The data type of input
 * \tparam DTypeOut The data type of output
 * \tparam IdType The data type of index
 * \param q The query tensor.
 * \param paged_kv The paged kv-cache data structure.
 * \param q_indptr The index pointer of queries.
 * \param o The output tensor.
 * \param tmp The temporary storage (only used for cooperative kernel).
 * \param num_qo_heads The number of query and output heads.
 * \param causal Whether to use causal attention.
 * \param rotary_mode The rotary mode.
 * \param rope_scale The scaling factor used in RoPE interpolation.
 * \param rope_theta The theta used in RoPE.
 * \param stream The cuda stream to execute the kernel on.
 * \return status Indicates whether CUDA calls are successful
 * \note This implementation is not optimized, we will provide a much faster implementation in the
 *   future.
 */
template <typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithPagedKVCache(DTypeIn* q, paged_kv_t<DTypeIn, IdType> paged_kv,
                                         IdType* q_indptr, DTypeOut* o, float* tmp,
                                         uint32_t num_qo_heads, bool causal = true,
                                         RotaryMode rotary_mode = RotaryMode::kNone,
                                         float rope_scale = 1.f, float rope_theta = 1e4,
                                         cudaStream_t stream = nullptr) {
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t head_dim = paged_kv.head_dim;
  const uint32_t batch_size = paged_kv.batch_size;

  std::vector<IdType> q_indptr_h(paged_kv.batch_size + 1);
  std::vector<IdType> kv_indptr_h(paged_kv.batch_size + 1);

  FLASHINFER_CUDA_CALL(PagedKVCacheToRaggedTensorComputeIndptr(paged_kv, kv_indptr_h, stream));
  uint32_t nnz = kv_indptr_h.back();

  DTypeIn *keys = nullptr, *values = nullptr;
  IdType* kv_indptr = nullptr;
  FLASHINFER_CUDA_CALL(
      cudaMallocAsync(&keys, nnz * num_kv_heads * head_dim * sizeof(DTypeIn), stream));
  FLASHINFER_CUDA_CALL(
      cudaMallocAsync(&values, nnz * num_kv_heads * head_dim * sizeof(DTypeIn), stream));
  FLASHINFER_CUDA_CALL(cudaMallocAsync(&kv_indptr, (batch_size + 1) * sizeof(IdType), stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(q_indptr_h.data(), q_indptr,
                                       sizeof(IdType) * (paged_kv.batch_size + 1),
                                       cudaMemcpyDeviceToHost, stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(kv_indptr, kv_indptr_h.data(),
                                       sizeof(IdType) * (paged_kv.batch_size + 1),
                                       cudaMemcpyHostToDevice, stream));
  FLASHINFER_CUDA_CALL(cudaStreamSynchronize(stream));
  FLASHINFER_CUDA_CALL(PagedKVCacheToRaggedTensor(paged_kv, keys, values, kv_indptr, stream));
  FLASHINFER_CUDA_CALL(cudaStreamSynchronize(stream));

  for (uint32_t batch_idx = 0; batch_idx < paged_kv.batch_size; ++batch_idx) {
    if (q_indptr_h[batch_idx] == q_indptr_h[batch_idx + 1]) {
      continue;
    }
    SinglePrefillWithKVCache(q + q_indptr_h[batch_idx] * num_qo_heads * head_dim,
                             keys + kv_indptr_h[batch_idx] * num_kv_heads * head_dim,
                             values + kv_indptr_h[batch_idx] * num_kv_heads * head_dim,
                             o + q_indptr_h[batch_idx] * num_qo_heads * head_dim, nullptr,
                             num_qo_heads, num_kv_heads,
                             q_indptr_h[batch_idx + 1] - q_indptr_h[batch_idx],
                             kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx], head_dim, causal,
                             QKVLayout::kNHD, rotary_mode, rope_scale, rope_theta, stream);
  }
  FLASHINFER_CUDA_CALL(cudaFreeAsync(keys, stream));
  FLASHINFER_CUDA_CALL(cudaFreeAsync(values, stream));
  FLASHINFER_CUDA_CALL(cudaFreeAsync(kv_indptr, stream));
  FLASHINFER_CUDA_CALL(cudaStreamSynchronize(stream));

  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_PREFILL_CUH_
