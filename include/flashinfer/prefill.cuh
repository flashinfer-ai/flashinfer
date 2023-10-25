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

__device__ __forceinline__ uint32_t sub_if_greater_or_zero(uint32_t x,
                                                           uint32_t y) {
  return (x > y) ? x - y : 0U;
}

template <bool row_major, typename T>
__device__ __forceinline__ void apply_llama_rope(T* x_first_half,
                                                 T* x_second_half,
                                                 const float* rope_freq,
                                                 uint32_t offset,
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
    __sincosf(float(offset + 8 * i) * rope_freq[2 * j + reg_id % 2], &sin,
              &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] =
        (tmp * cos - (float)x_second_half[reg_id] * sin) * scale;
    x_second_half[reg_id] =
        ((float)x_second_half[reg_id] * cos + tmp * sin) * scale;
  }
}

}  // namespace

template <uint32_t num_frags_y, uint32_t num_frags_z, uint32_t num_warps,
          QKVLayout layout, uint32_t group_size, typename T>
__device__ __forceinline__ void produce_kv(
    smem_t* smem, T* gptr, const tensor_info_t<layout, group_size>& qkv_info,
    const uint32_t kv_idx_base, const uint32_t kv_len, const uint32_t head_idx,
    const uint32_t tx, const uint32_t ty) {
  constexpr uint32_t num_cells_per_head_in =
      num_frags_y * 16 / cell_capacity<T>();

  uint32_t kv_idx = kv_idx_base + ty * 4 + (tx % 16) / 4;
  smem->offset = smem_t::get_permuted_offset<num_cells_per_head_in>(
      ty * 4 + (tx % 16) / 4, (tx / 16) * 4 + tx % 4);
  gptr += qkv_info.get_kv_elem_offset(
      kv_idx, head_idx, ((tx / 16) * 4 + tx % 4) * cell_capacity<T>());

#pragma unroll
  for (uint32_t i = 0; i < num_frags_z * 4 / num_warps; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < num_frags_y / 4; ++j) {
      smem->load_128b_async(gptr, kv_idx < kv_len);
      smem->offset += 16;
      gptr += 8 * cell_capacity<T>();
    }
    kv_idx += num_warps * 4;
    smem->offset += num_warps * 4 * num_cells_per_head_in - 4 * num_frags_y;
    gptr += num_warps * 4 * qkv_info.get_n_stride() -
            2 * num_frags_y * cell_capacity<T>();
  }
}

template <bool pin_q_in_reg, bool cooperative, uint32_t group_size, bool causal,
          QKVLayout layout, RotaryMode rotary_mode, uint32_t num_frags_x,
          uint32_t num_frags_y, uint32_t num_frags_z, uint32_t num_stages_smem,
          uint32_t num_warps, typename DTypeIn, typename DTypeOut>
__global__ void SinglePrefillWithKVCacheKernel(
    DTypeIn* __restrict__ q, DTypeIn* __restrict__ k, DTypeIn* __restrict__ v,
    DTypeOut* __restrict__ o, float* __restrict__ tmp,
    const tensor_info_t<layout, group_size> qkv_info, const float sm_scale,
    const float log2_rope_inv_scale, const float log2_rope_inv_theta) {
  vec_t<DTypeIn, 2> sm_scale2;
  sm_scale2.fill(sm_scale * math::log2e);

  const uint32_t qo_len = qkv_info.qo_len;
  const uint32_t kv_len = qkv_info.kv_len;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bx = blockIdx.x, chunk_idx = blockIdx.y,
                 qo_head_idx = blockIdx.z,
                 kv_head_idx = qo_head_idx / group_size;
  const uint32_t num_chunks = gridDim.y;
  const uint32_t chunk_size =
      cooperative ? (kv_len + num_chunks - 1) / num_chunks : kv_len;
  const uint32_t chunk_start = cooperative ? chunk_idx * chunk_size : 0;
  const uint32_t chunk_end =
      cooperative ? min((chunk_idx + 1) * chunk_size, kv_len) : kv_len;
  auto block = cg::this_thread_block();

  constexpr uint32_t head_dim = num_frags_y * 16;
  constexpr uint32_t num_cells_per_head_in =
      head_dim / cell_capacity<DTypeIn>();
  constexpr uint32_t num_cells_per_head_out =
      head_dim / cell_capacity<DTypeOut>();

  static_assert(num_frags_z * num_frags_y % num_warps == 0);

  extern __shared__ uint8_t smem[];

  uint32_t q_frag[num_frags_x]
                 [pin_q_in_reg ? num_frags_y
                               : (rotary_mode == RotaryMode::kNone ? 1 : 2)][4];
  uint32_t b_frag[(rotary_mode == RotaryMode::kNone ? 1 : 2)][4];
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
            log2_rope_inv_scale +
            log2_rope_inv_theta *
                float(2 * ((fy * 16 + (j / 2) * 8 + (tx % 4) * 2 + (j % 2)) %
                           (head_dim / 2))) /
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
  q_smem.offset = smem_t::get_permuted_offset<num_cells_per_head_in>(
      ty * num_frags_x * 16 + tx / 4, tx % 4);
  DTypeIn* q_ptr =
      q + qkv_info.get_qo_elem_offset(q_idx, qo_head_idx,
                                      (tx % 4) * cell_capacity<DTypeIn>());
#pragma unroll
  for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
#pragma unroll
      for (uint32_t fyo = 0; fyo < num_frags_y / 2; ++fyo) {
        // load q fragment from gmem to smem
        q_smem.load_128b_async(q_ptr, q_idx < qo_len);
        q_smem.offset += 8;
        q_ptr += 4 * cell_capacity<DTypeIn>();
      }
      q_idx += 8;
      q_smem.offset += 8 * num_cells_per_head_in - 4 * num_frags_y;
      q_ptr += 8 * qkv_info.get_n_stride() -
               2 * num_frags_y * cell_capacity<DTypeIn>();
    }
  }
  cp_async::commit_group();
  cp_async::wait_group<0>();
  block.sync();

  // preprocess q fragment, multiply sm_scale and apply rotary
  if constexpr (pin_q_in_reg) {
    // pin q fragment in reg
    q_smem.offset = smem_t::get_permuted_offset<num_cells_per_head_in>(
        ty * num_frags_x * 16 + tx % 16, tx / 16);
    q_idx = (bx * num_warps + ty) * num_frags_x * 16 + tx / 4;
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        q_smem.ldmatrix_m8n8x4(q_frag[fx][fy]);
        q_smem.offset = (q_smem.offset ^ 0x2) + (fy & 0x1) * 8;
      }
      q_smem.offset += 16 * num_cells_per_head_in - 4 * num_frags_y;
      if constexpr (rotary_mode == RotaryMode::kLlama) {
#pragma unroll
        for (uint32_t fyi = 0; fyi < num_frags_y / 2; ++fyi) {
          apply_llama_rope<true, DTypeIn>(
              (DTypeIn*)q_frag[fx][fyi],
              (DTypeIn*)q_frag[fx][num_frags_y / 2 + fyi], rope_freq[fyi],
              q_idx + kv_len - qo_len, sm_scale * math::log2e);
        }
      } else {
#pragma unroll
        for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
          for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
            vec_t<DTypeIn, 2> tmp;
            tmp.load((DTypeIn*)&q_frag[fx][fy][reg_id]);
            tmp.data = tmp.data * sm_scale2.data;
            tmp.store((DTypeIn*)&q_frag[fx][fy][reg_id]);
          }
        }
      }
      q_idx += 16;
    }
  } else {
    // do not pin q fragment in reg
    if constexpr (rotary_mode == RotaryMode::kLlama) {
      q_smem.offset = smem_t::get_permuted_offset<num_cells_per_head_in>(
          ty * num_frags_x * 16 + tx % 16, tx / 16);
      q_idx = (bx * num_warps + ty) * num_frags_x * 16 + tx / 4;
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
        for (uint32_t fyi = 0; fyi < num_frags_y / 2; ++fyi) {
          q_smem.ldmatrix_m8n8x4(q_frag[0][0]);
          q_smem.offset += num_frags_y * 2;
          q_smem.ldmatrix_m8n8x4(q_frag[0][1]);
          apply_llama_rope<true, DTypeIn>(
              (DTypeIn*)q_frag[0][0], (DTypeIn*)q_frag[0][1], rope_freq[fyi],
              q_idx + kv_len - qo_len, sm_scale * math::log2e);
          q_smem.stmatrix_m8n8x4(q_frag[0][1]);
          q_smem.offset -= num_frags_y * 2;
          q_smem.stmatrix_m8n8x4(q_frag[0][0]);
          q_smem.offset = (q_smem.offset ^ 0x2) + (fyi & 0x1) * 8;
        }
        q_smem.offset += 16 * num_cells_per_head_in - 2 * num_frags_y;
        q_idx += 16;
      }
    } else {
#pragma unroll
      for (uint32_t i = 0; i < num_frags_x * 16 * head_dim / 64; ++i) {
        vec_t<DTypeIn, 2> tmp;
        tmp.load((DTypeIn*)(q_smem.base +
                            ty * num_frags_x * 16 * num_cells_per_head_in) +
                 i * 64 + tx * 2);
        tmp.data = tmp.data * sm_scale2.data;
        tmp.store((DTypeIn*)(q_smem.base +
                             ty * num_frags_x * 16 * num_cells_per_head_in) +
                  i * 64 + tx * 2);
      }
    }
  }

  smem_t k_smem[num_stages_smem];
  smem_t v_smem[num_stages_smem];
#pragma unroll
  for (uint32_t i = 0; i < num_stages_smem; ++i) {
    if constexpr (pin_q_in_reg) {
      k_smem[i].base =
          (cell_t*)(smem + (i * num_frags_z) * 16 * head_dim * sizeof(DTypeIn));
      v_smem[i].base = (cell_t*)(smem + ((num_stages_smem + i) * num_frags_z) *
                                            16 * head_dim * sizeof(DTypeIn));
    } else {
      k_smem[i].base =
          (cell_t*)(smem + (num_warps * num_frags_x + i * num_frags_z) * 16 *
                               head_dim * sizeof(DTypeIn));
      v_smem[i].base = (cell_t*)(smem + (num_warps * num_frags_x +
                                         (num_stages_smem + i) * num_frags_z) *
                                            16 * head_dim * sizeof(DTypeIn));
    }
  }

  const uint32_t num_iterations =
      ((causal ? min(chunk_end - chunk_start,
                     sub_if_greater_or_zero(
                         kv_len - qo_len +
                             ((bx + 1) * num_frags_x * num_warps) * 16,
                         chunk_start))
               : chunk_end - chunk_start) +
       16 * num_frags_z - 1) /
      (16 * num_frags_z);

  const uint32_t mask_iteration =
      (causal ? min(chunk_end - chunk_start,
                    sub_if_greater_or_zero(
                        kv_len + bx * num_warps * num_frags_x - qo_len,
                        chunk_start))
              : (chunk_end - chunk_start)) /
      (16 * num_frags_z);

#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
    const uint32_t stage_idx = iter;
    produce_kv<num_frags_y, num_frags_z, num_warps>(
        k_smem + stage_idx, k, qkv_info, chunk_start + iter * 16 * num_frags_z,
        chunk_end, kv_head_idx, tx, ty);
    cp_async::commit_group();
    produce_kv<num_frags_y, num_frags_z, num_warps>(
        v_smem + stage_idx, v, qkv_info, chunk_start + iter * 16 * num_frags_z,
        chunk_end, kv_head_idx, tx, ty);
    cp_async::commit_group();
  }

#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    const uint32_t stage_idx = iter % num_stages_smem;
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    // init x_frag with 0
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
          x_frag[fx][fz][reg_id] = 0.f;
        }
      }
    }

    if constexpr (rotary_mode == RotaryMode::kLlama) {
      // apply rotary on the fly
      if constexpr (!pin_q_in_reg) {
        q_smem.offset = smem_t::get_permuted_offset<num_cells_per_head_in>(
            ty * num_frags_x * 16 + tx % 16, tx / 16);
      }
      k_smem[stage_idx].offset =
          smem_t::get_permuted_offset<num_cells_per_head_in>(
              8 * (tx / 16) + tx % 8, (tx % 16) / 8);

      // compute q*k^T
#pragma unroll
      for (uint32_t fyi = 0; fyi < num_frags_y / 2; ++fyi) {
        if constexpr (!pin_q_in_reg) {
#pragma unroll
          for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
            q_smem.ldmatrix_m8n8x4(q_frag[fx][0]);
            q_smem.offset += num_frags_y * 2;
            q_smem.ldmatrix_m8n8x4(q_frag[fx][1]);
            q_smem.offset += 16 * num_cells_per_head_in - num_frags_y * 2;
          }
          q_smem.offset = (q_smem.offset ^ 0x2) + (fyi & 0x1) * 8 -
                          num_frags_x * 16 * num_cells_per_head_in;
        }
        uint32_t kv_idx = chunk_start + iter * 16 * num_frags_z + tx / 4;

#pragma unroll
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          k_smem[stage_idx].ldmatrix_m8n8x4(b_frag[0]);
          k_smem[stage_idx].offset += num_frags_y * 2;
          k_smem[stage_idx].ldmatrix_m8n8x4(b_frag[1]);
          k_smem[stage_idx].offset +=
              16 * num_cells_per_head_in - num_frags_y * 2;
          apply_llama_rope<false, DTypeIn>(
              (DTypeIn*)b_frag[0], (DTypeIn*)b_frag[1], rope_freq[fyi], kv_idx);
          kv_idx += 16;
#pragma unroll
          for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn>(
                x_frag[fx][fz], q_frag[fx][pin_q_in_reg ? fyi : 0], b_frag[0]);
            mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn>(
                x_frag[fx][fz],
                q_frag[fx][pin_q_in_reg ? fyi + num_frags_y / 2 : 1],
                b_frag[1]);
          }
        }
        k_smem[stage_idx].offset = (k_smem[stage_idx].offset ^ 0x2) +
                                   (fyi & 0x1) * 8 -
                                   num_frags_z * 16 * num_cells_per_head_in;
      }
    } else {
      // load k tile from smem to reg
      if constexpr (!pin_q_in_reg) {
        q_smem.offset = smem_t::get_permuted_offset<num_cells_per_head_in>(
            ty * num_frags_x * 16 + tx % 16, tx / 16);
      }
      k_smem[stage_idx].offset =
          smem_t::get_permuted_offset<num_cells_per_head_in>(
              8 * (tx / 16) + tx % 8, (tx % 16) / 8);

      // compute q*k^T
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        if constexpr (!pin_q_in_reg) {
#pragma unroll
          for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
            q_smem.ldmatrix_m8n8x4(q_frag[fx][0]);
            q_smem.offset += 16 * num_cells_per_head_in;
          }
          q_smem.offset = (q_smem.offset ^ 0x2) + (fy & 0x1) * 8 -
                          num_frags_x * 16 * num_cells_per_head_in;
        }

#pragma unroll
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
          k_smem[stage_idx].ldmatrix_m8n8x4(b_frag[0]);
          k_smem[stage_idx].offset += 16 * num_cells_per_head_in;
#pragma unroll
          for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn>(
                x_frag[fx][fz], q_frag[fx][pin_q_in_reg ? fy : 0], b_frag[0]);
          }
        }
        k_smem[stage_idx].offset = (k_smem[stage_idx].offset ^ 0x2) +
                                   (fy & 0x1) * 8 -
                                   num_frags_z * 16 * num_cells_per_head_in;
      }
    }

    // apply mask
    if (iter >= mask_iteration) {
      uint32_t q_idx_base = ((bx * num_warps + ty) * num_frags_x) * 16 + tx / 4,
               kv_idx_base =
                   chunk_start + iter * 16 * num_frags_z + 2 * (tx % 4);
#pragma unroll
      for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
        for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
#pragma unroll
          for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
            const uint32_t q_idx = q_idx_base + 8 * ((reg_id % 4) / 2),
                           kv_idx = kv_idx_base + 8 * (reg_id / 4) + reg_id % 2;
            const bool out_of_boundary =
                (causal ? (kv_idx > kv_len + q_idx - qo_len ||
                           (cooperative && kv_idx >= chunk_end))
                        : kv_idx >= chunk_end);
            x_frag[fx][fz][reg_id] =
                out_of_boundary ? -5e4 : x_frag[fx][fz][reg_id];
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
          float m_local =
              max(max(x_frag[fx][fz][j * 2 + 0], x_frag[fx][fz][j * 2 + 1]),
                  max(x_frag[fx][fz][j * 2 + 4], x_frag[fx][fz][j * 2 + 5]));
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
          x_frag[fx][fz][j * 2 + 0] =
              math::ptx_exp2(x_frag[fx][fz][j * 2 + 0] - m[fx][j]);
          x_frag[fx][fz][j * 2 + 1] =
              math::ptx_exp2(x_frag[fx][fz][j * 2 + 1] - m[fx][j]);
          x_frag[fx][fz][j * 2 + 4] =
              math::ptx_exp2(x_frag[fx][fz][j * 2 + 4] - m[fx][j]);
          x_frag[fx][fz][j * 2 + 5] =
              math::ptx_exp2(x_frag[fx][fz][j * 2 + 5] - m[fx][j]);
          d[fx][j] += x_frag[fx][fz][j * 2 + 0] + x_frag[fx][fz][j * 2 + 1] +
                      x_frag[fx][fz][j * 2 + 4] + x_frag[fx][fz][j * 2 + 5];
        }
      }
#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
        vec_t<DTypeIn, 8> tmp;
        tmp.cast_load(x_frag[fx][fz]);
        tmp.store((DTypeIn*)x_frag[fx][fz]);
      }
    }

    block.sync();
    produce_kv<num_frags_y, num_frags_z, num_warps>(
        k_smem + stage_idx, k, qkv_info,
        chunk_start + (iter + num_stages_smem) * 16 * num_frags_z, kv_len,
        kv_head_idx, tx, ty);
    cp_async::commit_group();
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();

    // load v tile from smem to reg
    v_smem[stage_idx].offset =
        smem_t::get_permuted_offset<num_cells_per_head_in>(tx % 16, tx / 16);

    // compute sfm*v
#pragma unroll
    for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
#pragma unroll
      for (uint32_t fz = 0; fz < num_frags_z; ++fz) {
        v_smem[stage_idx].ldmatrix_m8n8x4_trans(b_frag[0]);
        v_smem[stage_idx].offset += 16 * num_cells_per_head_in;
#pragma unroll
        for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeIn>(
              o_frag[fx][fy], (uint32_t*)x_frag[fx][fz], b_frag[0]);
        }
      }
      v_smem[stage_idx].offset = (v_smem[stage_idx].offset ^ 0x2) +
                                 (fy & 0x1) * 8 -
                                 num_frags_z * 16 * num_cells_per_head_in;
    }
    block.sync();
    produce_kv<num_frags_y, num_frags_z, num_warps>(
        v_smem + stage_idx, v, qkv_info,
        chunk_start + (iter + num_stages_smem) * 16 * num_frags_z, kv_len,
        kv_head_idx, tx, ty);
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
            tmp + ((fx * num_frags_y + fy) * grid.size() + grid.thread_rank()) *
                      8,
            o_frag[fx][fy]);
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
        *(float2*)&tmp_md[((fx * 2 + j) * grid.size() + grid.thread_rank()) *
                          2] = make_float2(m[fx][j], d[fx][j]);
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
          float2 md = *(
              float2*)&tmp_md[((fx * 2 + j) * grid.size() +
                               ((qo_head_idx * num_chunks + iter) * gridDim.x +
                                bx) *
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
          o_frag_i.load(tmp +
                        ((fx * num_frags_y + fy) * grid.size() +
                         ((qo_head_idx * num_chunks + iter) * gridDim.x + bx) *
                             block.num_threads() +
                         block.thread_rank()) *
                            8);
#pragma unroll
          for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
            o_frag[fx][fy][reg_id] +=
                o_frag_i[reg_id] * other_scale[fx][(reg_id % 4) / 2];
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
        o_frag[fx][fy][reg_id] =
            __fdividef(o_frag[fx][fy][reg_id], d[fx][(reg_id % 4) / 2]);
      }
    }
  }

  // write back
  smem_t o_smem(smem);
  if constexpr (std::is_same<DTypeOut, float>::value) {
    // TODO(Zihao)
  } else if constexpr (sizeof(DTypeOut) == 2) {
    o_smem.offset = smem_t::get_permuted_offset<num_cells_per_head_out>(
        ty * num_frags_x * 16 + tx / 4, 0);
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        vec_cast<DTypeOut, float, 2>(
            (DTypeOut*)(o_smem.base + o_smem.offset) + (tx % 4) * 2,
            &o_frag[fx][fy][0]);
        vec_cast<DTypeOut, float, 2>((DTypeOut*)(o_smem.base + o_smem.offset +
                                                 8 * num_cells_per_head_out) +
                                         (tx % 4) * 2,
                                     &o_frag[fx][fy][2]);
        vec_cast<DTypeOut, float, 2>(
            (DTypeOut*)(o_smem.base + (o_smem.offset ^ 0x1)) + (tx % 4) * 2,
            &o_frag[fx][fy][4]);
        vec_cast<DTypeOut, float, 2>(
            (DTypeOut*)(o_smem.base + (o_smem.offset ^ 0x1) +
                        8 * num_cells_per_head_out) +
                (tx % 4) * 2,
            &o_frag[fx][fy][6]);
        o_smem.offset = (o_smem.offset ^ 0x2) + (fy & 0x1) * 8;
      }
      o_smem.offset += 16 * num_cells_per_head_out - num_frags_y * 4;
    }

    o_smem.offset = smem_t::get_permuted_offset<num_cells_per_head_out>(
        ty * num_frags_x * 16 + tx % 16, tx / 16);
    uint32_t o_idx = (bx * num_warps + ty) * num_frags_x * 16 + tx % 16;
    DTypeOut* o_ptr =
        o + qkv_info.get_qo_elem_offset(o_idx, qo_head_idx,
                                        tx / 16 * cell_capacity<DTypeOut>());
#pragma unroll
    for (uint32_t fx = 0; fx < num_frags_x; ++fx) {
#pragma unroll
      for (uint32_t fy = 0; fy < num_frags_y; ++fy) {
        if (o_idx < qo_len) {
          o_smem.store_128b(o_ptr);
        }
        o_ptr += 2 * cell_capacity<DTypeOut>();
        o_smem.offset = (o_smem.offset ^ 0x2) + (fy & 0x1) * 8;
      }
      o_idx += 16;
      o_ptr += qkv_info.get_n_stride() * 16 -
               2 * num_frags_y * cell_capacity<DTypeOut>();
      o_smem.offset += 16 * num_cells_per_head_out - num_frags_y * 4;
    }
  } else {
    // NOTE(Zihao): Not implemented yet.
  }
}

template <typename DTypeIn, typename DTypeOut>
cudaError_t SinglePrefillWithKVCacheWorkEstimation(
    uint32_t& tmp_size, uint32_t& max_grid_size, uint32_t num_qo_heads,
    uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len, uint32_t head_dim,
    bool causal = true, QKVLayout layout = QKVLayout::kNHD,
    RotaryMode rotary_mode = RotaryMode::kNone, cudaStream_t stream = nullptr,
    uint32_t dev_id = 0) {
  SWITCH_NUM_FRAGS_X(
      qo_len > 64, NUM_FRAGS_X,
      {SWITCH_GQA_GROUP_SIZE(
          num_qo_heads / num_kv_heads, GROUP_SIZE,
          {SWITCH_CAUSAL(
              causal, CAUSAL,
              {SWITCH_HEAD_DIM(
                  head_dim, HEAD_DIM,
                  {SWITCH_ROTARY_MODE(
                      rotary_mode, ROTARY_MODE, {SWITCH_LAYOUT(layout, LAYOUT, {
                        constexpr bool pin_q_in_reg = false;
                        constexpr uint32_t num_frags_x = NUM_FRAGS_X;
                        constexpr uint32_t num_frags_y = HEAD_DIM / 16;
                        constexpr uint32_t num_frags_z = 2;
                        constexpr uint32_t num_warps = 4UL;
                        constexpr uint32_t num_stages_smem = 1;
                        constexpr uint32_t num_threads = num_warps * warp_size;
                        constexpr uint32_t num_rows_per_cta =
                            num_frags_x * num_warps * 16;
                        auto cooperative_kernel =
                            SinglePrefillWithKVCacheKernel<
                                pin_q_in_reg, true, GROUP_SIZE, CAUSAL, LAYOUT,
                                ROTARY_MODE, num_frags_x, num_frags_y,
                                num_frags_z, num_stages_smem, num_warps,
                                DTypeIn, DTypeOut>;
                        uint32_t smem_size =
                            (pin_q_in_reg
                                 ? max(num_frags_x * num_warps,
                                       2 * num_stages_smem * num_frags_z)
                                 : (num_frags_x * num_warps +
                                    2 * num_stages_smem * num_frags_z)) *
                            16 * head_dim * sizeof(DTypeIn);
                        int num_blocks_per_sm = 0;
                        int num_sm = 0;
                        FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
                            &num_sm, cudaDevAttrMultiProcessorCount, dev_id));
                        FLASHINFER_CUDA_CALL(
                            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                &num_blocks_per_sm, cooperative_kernel,
                                num_threads, smem_size));
                        max_grid_size = num_blocks_per_sm * num_sm;
                        uint32_t num_chunks =
                            min((num_blocks_per_sm * num_sm) /
                                    (num_qo_heads *
                                     (qo_len + (num_rows_per_cta - 1)) /
                                     num_rows_per_cta),
                                kv_len / 512);
                        if (num_chunks > 1) {
                          uint32_t grid_size =
                              32 * num_warps *
                              ((qo_len + (num_rows_per_cta - 1)) /
                               num_rows_per_cta) *
                              num_chunks * num_qo_heads;
                          tmp_size = sizeof(float) *
                                     (4 * num_frags_x +
                                      num_frags_x * num_frags_y * 8) *
                                     grid_size;
                        } else {
                          tmp_size = 0;
                        }
                      })})})})})});
  return cudaSuccess;
}

template <typename DTypeIn, typename DTypeOut>
cudaError_t SinglePrefillWithKVCache(
    DTypeIn* q, DTypeIn* k, DTypeIn* v, DTypeOut* o, float* tmp,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t qo_len,
    uint32_t kv_len, uint32_t head_dim, bool causal = true,
    QKVLayout layout = QKVLayout::kNHD,
    RotaryMode rotary_mode = RotaryMode::kNone, float rope_scale = 1.f,
    float rope_theta = 1e4, cudaStream_t stream = nullptr,
    uint32_t dev_id = 0) {
  const float sm_scale = 1.f / std::sqrt(float(head_dim));
  const float log2_rope_inv_scale = -std::log2f(rope_scale);
  const float log2_rope_inv_theta = -std::log2f(rope_theta);
  FLASHINFER_CUDA_CALL(cudaSetDevice(dev_id));
  assert(kv_len >= qo_len);

  SWITCH_NUM_FRAGS_X(
      qo_len > 64, NUM_FRAGS_X,
      {SWITCH_GQA_GROUP_SIZE(
          num_qo_heads / num_kv_heads, GROUP_SIZE,
          {SWITCH_CAUSAL(
              causal, CAUSAL,
              {SWITCH_HEAD_DIM(
                  head_dim, HEAD_DIM,
                  {SWITCH_ROTARY_MODE(
                      rotary_mode, ROTARY_MODE, {SWITCH_LAYOUT(layout, LAYOUT, {
                        constexpr bool pin_q_in_reg = false;
                        constexpr uint32_t num_frags_x = NUM_FRAGS_X;
                        constexpr uint32_t num_frags_y = HEAD_DIM / 16;
                        constexpr uint32_t num_frags_z = 2;
                        constexpr uint32_t num_warps = 4UL;
                        constexpr uint32_t num_stages_smem = 1;
                        constexpr uint32_t num_threads = num_warps * warp_size;
                        constexpr uint32_t num_rows_per_cta =
                            num_frags_x * num_warps * 16;
                        auto cooperative_kernel =
                            SinglePrefillWithKVCacheKernel<
                                pin_q_in_reg, true, GROUP_SIZE, CAUSAL, LAYOUT,
                                ROTARY_MODE, num_frags_x, num_frags_y,
                                num_frags_z, num_stages_smem, num_warps,
                                DTypeIn, DTypeOut>;
                        tensor_info_t<LAYOUT, GROUP_SIZE> qkv_info(
                            qo_len, kv_len, num_kv_heads, HEAD_DIM);
                        uint32_t smem_size =
                            (pin_q_in_reg
                                 ? max(num_frags_x * num_warps,
                                       2 * num_stages_smem * num_frags_z)
                                 : (num_frags_x * num_warps +
                                    2 * num_stages_smem * num_frags_z)) *
                            16 * head_dim * sizeof(DTypeIn);
                        int num_blocks_per_sm = 0;
                        int num_sm = 0;
                        FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
                            &num_sm, cudaDevAttrMultiProcessorCount, dev_id));
                        FLASHINFER_CUDA_CALL(
                            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                &num_blocks_per_sm, cooperative_kernel,
                                num_threads, smem_size));
                        uint32_t num_chunks =
                            min((num_blocks_per_sm * num_sm) /
                                    (num_qo_heads *
                                     (qo_len + (num_rows_per_cta - 1)) /
                                     num_rows_per_cta),
                                kv_len / 512);

                        if (num_chunks <= 1 || tmp == nullptr) {
                          // Enough parallelism, do not use cooperative groups
                          auto kernel = SinglePrefillWithKVCacheKernel<
                              pin_q_in_reg, false, GROUP_SIZE, CAUSAL, LAYOUT,
                              ROTARY_MODE, num_frags_x, num_frags_y,
                              num_frags_z, num_stages_smem, num_warps, DTypeIn,
                              DTypeOut>;
                          void* args[] = {(void*)&q,
                                          (void*)&k,
                                          (void*)&v,
                                          (void*)&o,
                                          (void*)&tmp,
                                          (void*)&qkv_info,
                                          (void*)&sm_scale,
                                          (void*)&log2_rope_inv_scale,
                                          (void*)&log2_rope_inv_theta};
                          dim3 nblks((qo_len + (num_rows_per_cta - 1)) /
                                         num_rows_per_cta,
                                     1, num_qo_heads);
                          dim3 nthrs(32, num_warps);
                          FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                              kernel,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              smem_size));
                          FLASHINFER_CUDA_CALL(
                              cudaLaunchKernel((void*)kernel, nblks, nthrs,
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
                                          (void*)&log2_rope_inv_scale,
                                          (void*)&log2_rope_inv_theta};
                          dim3 nblks((qo_len + (num_rows_per_cta - 1)) /
                                         num_rows_per_cta,
                                     num_chunks, num_qo_heads);
                          dim3 nthrs(32, num_warps);
                          FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
                              cooperative_kernel,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              smem_size));
                          FLASHINFER_CUDA_CALL(cudaLaunchCooperativeKernel(
                              (void*)cooperative_kernel, nblks, nthrs, args,
                              smem_size, stream));
                        }
                      })})})})})});
  return cudaSuccess;
}

template <typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t BatchPrefillWithPagedKVCache(
    DTypeIn* q, paged_kv_t<DTypeIn, IdType> paged_kv, IdType* q_indptr,
    DTypeOut* o, float* tmp, uint32_t num_qo_heads, bool causal = true,
    RotaryMode rotary_mode = RotaryMode::kNone, float rope_scale = 1.f,
    float rope_theta = 1e4, cudaStream_t stream = nullptr,
    uint32_t dev_id = 0) {
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t head_dim = paged_kv.head_dim;
  const uint32_t batch_size = paged_kv.batch_size;

  FLASHINFER_CUDA_CALL(cudaSetDevice(dev_id));
  std::vector<IdType> q_indptr_h(paged_kv.batch_size + 1);
  std::vector<IdType> kv_indptr_h(paged_kv.batch_size + 1);

  FLASHINFER_CUDA_CALL(PagedKVCacheToRaggedTensorComputeIndptr(
      paged_kv, kv_indptr_h, stream, dev_id));
  uint32_t nnz = kv_indptr_h.back();

  DTypeIn *keys = nullptr, *values = nullptr;
  IdType* kv_indptr = nullptr;
  FLASHINFER_CUDA_CALL(cudaMallocAsync(
      &keys, nnz * num_kv_heads * head_dim * sizeof(DTypeIn), stream));
  FLASHINFER_CUDA_CALL(cudaMallocAsync(
      &values, nnz * num_kv_heads * head_dim * sizeof(DTypeIn), stream));
  FLASHINFER_CUDA_CALL(
      cudaMallocAsync(&kv_indptr, (batch_size + 1) * sizeof(IdType), stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(
      q_indptr_h.data(), q_indptr, sizeof(IdType) * (paged_kv.batch_size + 1),
      cudaMemcpyDeviceToHost, stream));
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(
      kv_indptr, kv_indptr_h.data(), sizeof(IdType) * (paged_kv.batch_size + 1),
      cudaMemcpyHostToDevice, stream));
  FLASHINFER_CUDA_CALL(cudaStreamSynchronize(stream));
  FLASHINFER_CUDA_CALL(
      PagedKVCacheToRaggedTensor(paged_kv, keys, values, kv_indptr, stream));
  FLASHINFER_CUDA_CALL(cudaStreamSynchronize(stream));

  for (uint32_t batch_idx = 0; batch_idx < paged_kv.batch_size; ++batch_idx) {
    SinglePrefillWithKVCache(
        q + q_indptr_h[batch_idx] * num_qo_heads * head_dim,
        keys + kv_indptr_h[batch_idx] * num_kv_heads * head_dim,
        values + kv_indptr_h[batch_idx] * num_kv_heads * head_dim,
        o + q_indptr_h[batch_idx] * num_qo_heads * head_dim, nullptr,
        num_qo_heads, num_kv_heads,
        q_indptr_h[batch_idx + 1] - q_indptr_h[batch_idx],
        kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx], head_dim, causal,
        QKVLayout::kNHD, rotary_mode, rope_scale, rope_theta, stream, dev_id);
  }
  FLASHINFER_CUDA_CALL(cudaFreeAsync(keys, stream));
  FLASHINFER_CUDA_CALL(cudaFreeAsync(values, stream));
  FLASHINFER_CUDA_CALL(cudaFreeAsync(kv_indptr, stream));
  FLASHINFER_CUDA_CALL(cudaStreamSynchronize(stream));

  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_PREFILL_CUH_
