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

#pragma once

#include "../../arch/cp_async.cuh"
#include "../../arch/matrix_memory.cuh"
#include "../../arch/mma_sm120.cuh"
#include "../../arch/mma_sm120_nvfp4.cuh"
#include "../../common/zero_row.cuh"
#include "../../compute/nvfp4_vt.cuh"
#include "../../compute/warp_tiles.cuh"
#include "../../pipeline/staged_pipeline.cuh"
#include "gather.cuh"
#include "q_stage.cuh"
#include "resources.cuh"

namespace flashinfer::sparse_mla_sm120::nvfp4 {

template <ModelType MT, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE, bool DUAL_CACHE = false>
__global__ void __launch_bounds__(DECODE_BLOCK_THREADS) sparse_mla_decode_dsv4_nvfp4_kernel(
    const bf16* __restrict__ q, const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ indices, bf16* __restrict__ mid_out, float* __restrict__ mid_lse,
    bf16* __restrict__ output, float* __restrict__ out_lse, const float* __restrict__ attn_sink,
    const int* __restrict__ topk_length_ptr, const uint8_t* __restrict__ extra_kv_cache,
    const int32_t* __restrict__ extra_indices, const int* __restrict__ extra_topk_length_ptr,
    int extra_topk, int extra_page_block_size, size_t extra_page_stride_bytes, int num_tokens,
    int scratch_split_stride, int chunks_per_block, float sm_scale, size_t page_stride_bytes,
    bool write_direct) {
  using KV = KVCacheTraits<MT>;
  static_assert(MT == ModelType::DSV4);
  constexpr int D_NOPE = KV::D_NOPE;
  constexpr int D_ROPE_C = KV::D_ROPE;
  constexpr int D_QK = KV::D_QK;
  constexpr int D_V_C = KV::D_V;
  constexpr int NUM_K64_TILES = D_NOPE / 64;
  constexpr int VALID_HPB = (NUM_HEADS < HPB) ? NUM_HEADS : HPB;
  constexpr int PV_SCALE_GROUPS = D_NOPE / SF_VEC_SIZE;
  constexpr int PV_GROUPS_PER_WARP = (PV_SCALE_GROUPS + DECODE_N_WARPS - 1) / DECODE_N_WARPS;
  constexpr int PV_N8_TILES_PER_GROUP = SF_VEC_SIZE / 8;
  constexpr int ROPE_DIMS_PER_WARP = D_ROPE_C / DECODE_N_WARPS;
  constexpr int ROPE_N_TILES = ROPE_DIMS_PER_WARP / 8;
  constexpr int ROPE_K_ITERS = DECODE_CAND_WINDOW / 16;
  const int token_idx = blockIdx.x;
  const int h_start = blockIdx.y * HPB;
  const int split_idx = blockIdx.z;
  if (token_idx >= num_tokens) return;

  int topk_len = topk_length_ptr ? __ldg(topk_length_ptr + token_idx) : TOPK;
  topk_len = max(0, min(topk_len, TOPK));
  const int num_main_chunks = (topk_len + DECODE_CAND_WINDOW - 1) / DECODE_CAND_WINDOW;
  int extra_topk_len = 0;
  if constexpr (DUAL_CACHE) {
    extra_topk_len = extra_topk_length_ptr ? __ldg(extra_topk_length_ptr + token_idx) : extra_topk;
    extra_topk_len = max(0, min(extra_topk_len, extra_topk));
  }
  const int num_extra_chunks = (extra_topk_len + DECODE_CAND_WINDOW - 1) / DECODE_CAND_WINDOW;
  const int num_chunks = num_main_chunks + num_extra_chunks;
  const int chunk_lo = split_idx * chunks_per_block;
  const int chunk_hi = min(chunk_lo + chunks_per_block, num_chunks);
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const bool is_io = warp_id >= DECODE_N_WARPS;

  if (chunk_lo >= num_chunks) {
    if (!is_io) {
      if (write_direct) {
        for (int i = threadIdx.x; i < VALID_HPB * D_V_C; i += DECODE_MATH_THREADS) {
          output[((size_t)token_idx * NUM_HEADS + h_start) * D_V_C + i] = __float2bfloat16(0.f);
        }
        if (threadIdx.x < VALID_HPB) {
          const int h = h_start + threadIdx.x;
          out_lse[(size_t)token_idx * NUM_HEADS + h] =
              attn_sink ? __ldg(attn_sink + h) * LOG2E : -INFINITY;
        }
      } else if (threadIdx.x < VALID_HPB) {
        const int h = h_start + threadIdx.x;
        mid_lse[(size_t)token_idx * NUM_HEADS * scratch_split_stride +
                (size_t)h * scratch_split_stride + split_idx] = -1e30f;
      }
    }
    return;
  }

  extern __shared__ __align__(16) char smem_raw[];
  auto sm = DecodeNVFP4Smem<MT>::init(smem_raw);
  __shared__ bf16 sm_p_full[HPB][DECODE_CAND_WINDOW];
  const int32_t* idx_base = indices + (size_t)token_idx * TOPK;

  using Ring = pipeline::AsyncRing<DECODE_KV_BUF_COUNT>;
  if (threadIdx.x == 0) Ring::init(sm.mbar_full(0), sm.mbar_empty(0));
  __syncthreads();

  auto issue_gather = [&](int chunk, int buf) {
    gather_tile<PAGE_BLOCK_SIZE, DUAL_CACHE, DECODE_KV_SMEM_STRIDE, DECODE_IO_WARPS * 32>(
        kv_cache, idx_base, chunk, topk_len, false, PAGE_BLOCK_SIZE, page_stride_bytes,
        (warp_id - DECODE_N_WARPS) * 32 + lane, sm.kv_fp4(buf), sm.kv_rope(buf), sm.kv_sc(buf),
        sm.mbar_full(buf), num_main_chunks, extra_topk_len, extra_kv_cache,
        DUAL_CACHE ? extra_indices + size_t(token_idx) * extra_topk : nullptr,
        extra_page_block_size, extra_page_stride_bytes);
  };

  if (is_io) {
    uint32_t raw_free_phase = 1;
    int raw_slot = 0;
    for (int chunk = chunk_lo; chunk < chunk_hi; ++chunk) {
      const int buf = (chunk - chunk_lo) % DECODE_KV_BUF_COUNT;
      Ring::Free::wait(sm.mbar_empty(raw_slot), raw_free_phase);
      issue_gather(chunk, buf);
      Ring::advance(raw_slot, raw_free_phase);
    }
    return;
  }

  const int gid = lane >> 2;
  const int tid = lane & 3;
  const bf16* q_base = q + (size_t)token_idx * NUM_HEADS * D_QK + (size_t)h_start * D_QK;
  quantize_q_nvfp4_to_smem<DECODE_MATH_THREADS>(sm.q_fp4(), sm.q_sc(), sm.q_rope(), q_base,
                                                VALID_HPB);

  float acc_nope_groups[1][PV_GROUPS_PER_WARP][PV_N8_TILES_PER_GROUP][4] = {0};
  auto& acc_nope = acc_nope_groups[0];
  float acc_rope[ROPE_N_TILES][4] = {0};
  float global_max[2] = {-1e30f, -1e30f};
  float global_sum[2] = {0.f, 0.f};
  uint32_t raw_ready_phase = 0;
  int raw_consumer_slot = 0;

  for (int chunk = chunk_lo; chunk < chunk_hi; ++chunk) {
    const int buf = (chunk - chunk_lo) % DECODE_KV_BUF_COUNT;
    bool is_extra_chunk = false;
    int section_chunk = chunk;
    int section_len = topk_len;
    const int32_t* section_indices = idx_base;
    if constexpr (DUAL_CACHE) {
      is_extra_chunk = chunk >= num_main_chunks;
      if (is_extra_chunk) {
        section_chunk = chunk - num_main_chunks;
        section_len = extra_topk_len;
        section_indices = extra_indices + (size_t)token_idx * extra_topk;
      }
    }
    const int chunk_start = section_chunk * DECODE_CAND_WINDOW;
    const int chunk_end = min(chunk_start + DECODE_CAND_WINDOW, section_len);
    Ring::Ready::wait(sm.mbar_full(raw_consumer_slot), raw_ready_phase);

    uint8_t* sm_kv_fp4 = sm.kv_fp4(buf);
    uint8_t* sm_kv_sc = sm.kv_sc(buf);
    bf16* sm_kv_rope = sm.kv_rope(buf);

    float qk_groups[1][DECODE_QK_N_TILES][4] = {0};
    auto& qk = qk_groups[0];
    const int warp_first_cand = warp_id * DECODE_ENTRIES_PER_WARP;
    qk_nvfp4_nope_16x8<1, DSV4_NVFP4_Q_PACKED_STRIDE, DSV4_NVFP4_SCALE_STRIDE,
                       DECODE_KV_SMEM_STRIDE, DECODE_SCALE_BYTES_PER_TOKEN, NUM_K64_TILES>(
        qk_groups, sm.q_fp4(), sm.q_sc(), sm_kv_fp4, sm_kv_sc, warp_first_cand, lane);

#pragma unroll
    for (int ks = 0; ks < D_ROPE_C / 16; ++ks) {
      uint32_t a0, a1, a2, a3;
      ldmatrix_load_A_bf16(a0, a1, a2, a3, sm.q_rope() + ks * 16, D_ROPE_C, lane);
#pragma unroll
      for (int nt = 0; nt < DECODE_QK_N_TILES; ++nt) {
        const int cand_row_base = warp_first_cand + nt * 8;
        const bf16* rope_row = sm_kv_rope + (size_t)(cand_row_base + gid) * D_ROPE_C + ks * 16;
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(rope_row + tid * 2);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(rope_row + tid * 2 + 8);
        MmaBf16Result r =
            mma_bf16_m16n8k16(a0, a1, a2, a3, b0, b1, qk[nt][0], qk[nt][1], qk[nt][2], qk[nt][3]);
        qk[nt][0] = r.d0;
        qk[nt][1] = r.d1;
        qk[nt][2] = r.d2;
        qk[nt][3] = r.d3;
      }
    }

    bool valid_candidate[DECODE_QK_N_TILES][2];
#pragma unroll
    for (int nt = 0; nt < DECODE_QK_N_TILES; ++nt) {
      const int c0 = warp_first_cand + nt * 8 + tid * 2;
      const int c1 = c0 + 1;
      const int abs_c0 = chunk_start + c0;
      const int abs_c1 = chunk_start + c1;
      const int idx0 = abs_c0 < section_len ? section_indices[abs_c0] : -1;
      const int idx1 = abs_c1 < section_len ? section_indices[abs_c1] : -1;
      valid_candidate[nt][0] = abs_c0 < chunk_end && idx0 >= 0;
      valid_candidate[nt][1] = abs_c1 < chunk_end && idx1 >= 0;
      const float qk_scale = sm_scale * LOG2E;
      qk[nt][0] = valid_candidate[nt][0] ? qk[nt][0] * qk_scale : -1e30f;
      qk[nt][2] = valid_candidate[nt][0] ? qk[nt][2] * qk_scale : -1e30f;
      qk[nt][1] = valid_candidate[nt][1] ? qk[nt][1] * qk_scale : -1e30f;
      qk[nt][3] = valid_candidate[nt][1] ? qk[nt][3] * qk_scale : -1e30f;
    }

    float local_max[2] = {-1e30f, -1e30f};
#pragma unroll
    for (int nt = 0; nt < DECODE_QK_N_TILES; ++nt) {
      local_max[0] = fmaxf(local_max[0], fmaxf(qk[nt][0], qk[nt][1]));
      local_max[1] = fmaxf(local_max[1], fmaxf(qk[nt][2], qk[nt][3]));
    }
#pragma unroll
    for (int s = 2; s >= 1; s >>= 1) {
      local_max[0] = fmaxf(local_max[0], __shfl_xor_sync(0xffffffff, local_max[0], s));
      local_max[1] = fmaxf(local_max[1], __shfl_xor_sync(0xffffffff, local_max[1], s));
    }
    float local_sum[2] = {0.f, 0.f};
    float p[DECODE_QK_N_TILES][4];
#pragma unroll
    for (int nt = 0; nt < DECODE_QK_N_TILES; ++nt) {
      p[nt][0] = valid_candidate[nt][0] ? exp2f(qk[nt][0] - local_max[0]) : 0.f;
      p[nt][1] = valid_candidate[nt][1] ? exp2f(qk[nt][1] - local_max[0]) : 0.f;
      p[nt][2] = valid_candidate[nt][0] ? exp2f(qk[nt][2] - local_max[1]) : 0.f;
      p[nt][3] = valid_candidate[nt][1] ? exp2f(qk[nt][3] - local_max[1]) : 0.f;
      local_sum[0] += p[nt][0] + p[nt][1];
      local_sum[1] += p[nt][2] + p[nt][3];
    }
#pragma unroll
    for (int s = 2; s >= 1; s >>= 1) {
      local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], s);
      local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], s);
    }

    if (tid == 0) {
      sm.reduce_scratch()[warp_id * HPB + gid] = local_max[0];
      sm.reduce_scratch()[warp_id * HPB + gid + 8] = local_max[1];
      sm.reduce_scratch_second()[warp_id * HPB + gid] = local_sum[0];
      sm.reduce_scratch_second()[warp_id * HPB + gid + 8] = local_sum[1];
    }
    bar_sync_t<Dsv4Nvfp4Sync::MATH, DECODE_MATH_THREADS>();
    if (threadIdx.x < VALID_HPB) {
      const int h = threadIdx.x;
      float block_max = -1e30f;
#pragma unroll
      for (int w = 0; w < DECODE_N_WARPS; ++w)
        block_max = fmaxf(block_max, sm.reduce_scratch()[w * HPB + h]);
      float block_sum = 0.f;
#pragma unroll
      for (int w = 0; w < DECODE_N_WARPS; ++w)
        block_sum += sm.reduce_scratch_second()[w * HPB + h] *
                     exp2f(sm.reduce_scratch()[w * HPB + h] - block_max);
      sm.reduce_scratch()[h] = block_max;
      sm.reduce_scratch_second()[h] = block_sum;
    }
    bar_sync_t<Dsv4Nvfp4Sync::MATH, DECODE_MATH_THREADS>();

    const float block_max0 = sm.reduce_scratch()[gid];
    const float block_max1 = sm.reduce_scratch()[gid + 8];
    const float block_sum0 = sm.reduce_scratch_second()[gid];
    const float block_sum1 = sm.reduce_scratch_second()[gid + 8];
    const float new_max0 = fmaxf(global_max[0], block_max0);
    const float new_max1 = fmaxf(global_max[1], block_max1);
    const float alpha0 = global_max[0] > -1e29f ? exp2f(global_max[0] - new_max0) : 0.f;
    const float alpha1 = global_max[1] > -1e29f ? exp2f(global_max[1] - new_max1) : 0.f;
    const float block_rescale0 = exp2f(block_max0 - new_max0);
    const float block_rescale1 = exp2f(block_max1 - new_max1);
    const float warp_rescale0 = exp2f(local_max[0] - new_max0);
    const float warp_rescale1 = exp2f(local_max[1] - new_max1);

    if (chunk > chunk_lo) {
#pragma unroll
      for (int slot = 0; slot < PV_GROUPS_PER_WARP; ++slot) {
#pragma unroll
        for (int nt = 0; nt < PV_N8_TILES_PER_GROUP; ++nt) {
          acc_nope[slot][nt][0] *= alpha0;
          acc_nope[slot][nt][1] *= alpha0;
          acc_nope[slot][nt][2] *= alpha1;
          acc_nope[slot][nt][3] *= alpha1;
        }
      }
#pragma unroll
      for (int nt = 0; nt < ROPE_N_TILES; ++nt) {
        acc_rope[nt][0] *= alpha0;
        acc_rope[nt][1] *= alpha0;
        acc_rope[nt][2] *= alpha1;
        acc_rope[nt][3] *= alpha1;
      }
      global_sum[0] = global_sum[0] * alpha0 + block_sum0 * block_rescale0;
      global_sum[1] = global_sum[1] * alpha1 + block_sum1 * block_rescale1;
    } else {
      global_sum[0] = block_sum0 * block_rescale0;
      global_sum[1] = block_sum1 * block_rescale1;
    }
    global_max[0] = new_max0;
    global_max[1] = new_max1;

    float w_pre[DECODE_QK_N_TILES][4];
#pragma unroll
    for (int nt = 0; nt < DECODE_QK_N_TILES; ++nt) {
      w_pre[nt][0] = p[nt][0] * warp_rescale0;
      w_pre[nt][1] = p[nt][1] * warp_rescale0;
      w_pre[nt][2] = p[nt][2] * warp_rescale1;
      w_pre[nt][3] = p[nt][3] * warp_rescale1;
      const int c0 = nt * 8 + tid * 2;
      const int c1 = c0 + 1;
      sm_p_full[gid][warp_first_cand + c0] = __float2bfloat16(w_pre[nt][0]);
      sm_p_full[gid][warp_first_cand + c1] = __float2bfloat16(w_pre[nt][1]);
      sm_p_full[gid + 8][warp_first_cand + c0] = __float2bfloat16(w_pre[nt][2]);
      sm_p_full[gid + 8][warp_first_cand + c1] = __float2bfloat16(w_pre[nt][3]);
    }

    // Reuse the current QK source tile directly.  All 256 math threads
    // transpose/dequantize/requantize token-major paged V into the single
    // ephemeral CTA-local V^T stage; the producer warps have already started
    // gathering the next candidate tile into the other source buffer.
    bar_sync_t<Dsv4Nvfp4Sync::MATH, DECODE_MATH_THREADS>();
    prepare_nvfp4_vt_from_smem<DECODE_MATH_THREADS, DECODE_KV_SMEM_STRIDE>(
        sm_kv_fp4, sm_kv_sc, sm.vt_data(), sm.vt_sc());
    bar_sync_t<Dsv4Nvfp4Sync::MATH, DECODE_MATH_THREADS>();

    for (int task = threadIdx.x; task < HPB * DECODE_VT_SCALE_GROUPS; task += DECODE_MATH_THREADS) {
      const int head = task / DECODE_VT_SCALE_GROUPS;
      const int cand_group = task % DECODE_VT_SCALE_GROUPS;
      quantize_group16_to_nvfp4(
          &sm_p_full[head][cand_group * SF_VEC_SIZE],
          sm.p_fp4() + head * DECODE_W_PACKED_STRIDE + cand_group * FP4_PACKED_PER_GROUP,
          sm.w_sc() + head * DECODE_VT_SCALE_GROUPS + cand_group);
    }
    bar_sync_t<Dsv4Nvfp4Sync::MATH, DECODE_MATH_THREADS>();

    pv_nvfp4_vt_16x16<1, DECODE_W_PACKED_STRIDE, DECODE_N_WARPS>(
        acc_nope_groups, sm.p_fp4(), sm.w_sc(), sm.vt_data(), sm.vt_sc(), warp_id, lane);

    const int rope_dim_base = warp_id * ROPE_DIMS_PER_WARP;
#pragma unroll
    for (int ks = 0; ks < ROPE_K_ITERS; ++ks) {
      uint32_t a0, a1, a2, a3;
      ldmatrix_load_A_bf16(a0, a1, a2, a3, reinterpret_cast<const bf16*>(&sm_p_full[0][ks * 16]),
                           DECODE_CAND_WINDOW, lane);
#pragma unroll
      for (int nt = 0; nt < ROPE_N_TILES; ++nt) {
        const int n_col = rope_dim_base + nt * 8;
        const int k_base = ks * 16;
        const int ent0 = k_base + tid * 2;
        const int ent1 = ent0 + 1;
        const int ent8 = ent0 + 8;
        const int ent9 = ent0 + 9;
        const int col = n_col + gid;
        const uint16_t v0 =
            *reinterpret_cast<const uint16_t*>(sm_kv_rope + (size_t)ent0 * D_ROPE_C + col);
        const uint16_t v1 =
            *reinterpret_cast<const uint16_t*>(sm_kv_rope + (size_t)ent1 * D_ROPE_C + col);
        const uint16_t v8 =
            *reinterpret_cast<const uint16_t*>(sm_kv_rope + (size_t)ent8 * D_ROPE_C + col);
        const uint16_t v9 =
            *reinterpret_cast<const uint16_t*>(sm_kv_rope + (size_t)ent9 * D_ROPE_C + col);
        const uint32_t b0 = (uint32_t)v0 | ((uint32_t)v1 << 16);
        const uint32_t b1 = (uint32_t)v8 | ((uint32_t)v9 << 16);
        MmaBf16Result r = mma_bf16_m16n8k16(a0, a1, a2, a3, b0, b1, acc_rope[nt][0],
                                            acc_rope[nt][1], acc_rope[nt][2], acc_rope[nt][3]);
        acc_rope[nt][0] = r.d0;
        acc_rope[nt][1] = r.d1;
        acc_rope[nt][2] = r.d2;
        acc_rope[nt][3] = r.d3;
      }
    }

    bar_sync_t<Dsv4Nvfp4Sync::MATH, DECODE_MATH_THREADS>();
    if (threadIdx.x == 0) Ring::Free::publish(sm.mbar_empty(raw_consumer_slot));
    Ring::advance(raw_consumer_slot, raw_ready_phase);
  }

  if (write_direct) {
    if (warp_id == 0 && tid == 0) {
      float lse0 = global_sum[0] > 0.f ? log2f(global_sum[0]) + global_max[0] : -INFINITY;
      float lse1 = global_sum[1] > 0.f ? log2f(global_sum[1]) + global_max[1] : -INFINITY;
      float output_scale0 = 1.f;
      float output_scale1 = 1.f;
      if (attn_sink != nullptr) {
        const float sink0 = __ldg(attn_sink + h_start + gid) * LOG2E;
        const float max0 = fmaxf(lse0, sink0);
        const float attn_mass0 = lse0 > -1e29f ? exp2f(lse0 - max0) : 0.f;
        const float sink_mass0 = exp2f(sink0 - max0);
        const float total0 = attn_mass0 + sink_mass0;
        output_scale0 = total0 > 0.f ? attn_mass0 / total0 : 0.f;
        lse0 = total0 > 0.f ? log2f(total0) + max0 : -1e30f;
        if constexpr (VALID_HPB > 8) {
          const float sink1 = __ldg(attn_sink + h_start + gid + 8) * LOG2E;
          const float max1 = fmaxf(lse1, sink1);
          const float attn_mass1 = lse1 > -1e29f ? exp2f(lse1 - max1) : 0.f;
          const float sink_mass1 = exp2f(sink1 - max1);
          const float total1 = attn_mass1 + sink_mass1;
          output_scale1 = total1 > 0.f ? attn_mass1 / total1 : 0.f;
          lse1 = total1 > 0.f ? log2f(total1) + max1 : -1e30f;
        }
      }
      sm.reduce_scratch()[gid] = output_scale0;
      sm.reduce_scratch_second()[gid] = lse0;
      if constexpr (VALID_HPB > 8) {
        sm.reduce_scratch()[gid + 8] = output_scale1;
        sm.reduce_scratch_second()[gid + 8] = lse1;
      }
    }
    bar_sync_t<Dsv4Nvfp4Sync::MATH, DECODE_MATH_THREADS>();
  }

  const float direct_scale0 = write_direct ? sm.reduce_scratch()[gid] : 1.f;
  const float direct_scale1 = write_direct ? sm.reduce_scratch()[gid + 8] : 1.f;
  const float inv_sum0 = global_sum[0] > 0.f ? direct_scale0 / global_sum[0] : 0.f;
  const float inv_sum1 = global_sum[1] > 0.f ? direct_scale1 / global_sum[1] : 0.f;
  bf16* destination =
      write_direct
          ? output + ((size_t)token_idx * NUM_HEADS + h_start) * D_V_C
          : mid_out + (((size_t)token_idx * NUM_HEADS + h_start) * (size_t)scratch_split_stride +
                       split_idx) *
                          D_V_C;
  const size_t head_stride = write_direct ? D_V_C : (size_t)scratch_split_stride * D_V_C;

#pragma unroll
  for (int slot = 0; slot < PV_GROUPS_PER_WARP; ++slot) {
    const int scale_group = slot * DECODE_N_WARPS + warp_id;
    if (scale_group >= PV_SCALE_GROUPS) continue;
#pragma unroll
    for (int nt = 0; nt < PV_N8_TILES_PER_GROUP; ++nt) {
      const int d0 = scale_group * SF_VEC_SIZE + nt * 8 + tid * 2;
      const __nv_bfloat162 lo =
          __floats2bfloat162_rn(acc_nope[slot][nt][0] * inv_sum0, acc_nope[slot][nt][1] * inv_sum0);
      const __nv_bfloat162 hi =
          __floats2bfloat162_rn(acc_nope[slot][nt][2] * inv_sum1, acc_nope[slot][nt][3] * inv_sum1);
      *reinterpret_cast<__nv_bfloat162*>(&destination[(size_t)gid * head_stride + d0]) = lo;
      if constexpr (VALID_HPB > 8) {
        *reinterpret_cast<__nv_bfloat162*>(&destination[(size_t)(gid + 8) * head_stride + d0]) = hi;
      }
    }
  }
#pragma unroll
  for (int nt = 0; nt < ROPE_N_TILES; ++nt) {
    const int d0 = D_NOPE + warp_id * ROPE_DIMS_PER_WARP + nt * 8 + tid * 2;
    const __nv_bfloat162 lo =
        __floats2bfloat162_rn(acc_rope[nt][0] * inv_sum0, acc_rope[nt][1] * inv_sum0);
    const __nv_bfloat162 hi =
        __floats2bfloat162_rn(acc_rope[nt][2] * inv_sum1, acc_rope[nt][3] * inv_sum1);
    *reinterpret_cast<__nv_bfloat162*>(&destination[(size_t)gid * head_stride + d0]) = lo;
    if constexpr (VALID_HPB > 8) {
      *reinterpret_cast<__nv_bfloat162*>(&destination[(size_t)(gid + 8) * head_stride + d0]) = hi;
    }
  }
  if (warp_id == 0 && tid == 0) {
    if (write_direct) {
      out_lse[(size_t)token_idx * NUM_HEADS + h_start + gid] = sm.reduce_scratch_second()[gid];
      if constexpr (VALID_HPB > 8) {
        out_lse[(size_t)token_idx * NUM_HEADS + h_start + gid + 8] =
            sm.reduce_scratch_second()[gid + 8];
      }
    } else {
      const float lse0 = global_sum[0] > 0.f ? log2f(global_sum[0]) + global_max[0] : -1e30f;
      const float lse1 = global_sum[1] > 0.f ? log2f(global_sum[1]) + global_max[1] : -1e30f;
      const size_t lse_base = (size_t)token_idx * NUM_HEADS * scratch_split_stride +
                              (size_t)h_start * scratch_split_stride;
      mid_lse[lse_base + (size_t)gid * scratch_split_stride + split_idx] = lse0;
      if constexpr (VALID_HPB > 8) {
        mid_lse[lse_base + (size_t)(gid + 8) * scratch_split_stride + split_idx] = lse1;
      }
    }
  }
}

template <int NUM_HEADS>
__global__ void __launch_bounds__(DECODE_MERGE2_THREADS, 2)
    sparse_mla_decode_dsv4_nvfp4_merge2_kernel(const bf16* __restrict__ mid_out,
                                               const float* __restrict__ mid_lse,
                                               bf16* __restrict__ output,
                                               float* __restrict__ out_lse,
                                               const float* __restrict__ attn_sink,
                                               int num_tokens) {
  constexpr int D_V = 512;
  constexpr int VECS_PER_HEAD = D_V / 8;
  constexpr int H_BLOCKS = (NUM_HEADS + HPB - 1) / HPB;
  const int token_idx = blockIdx.x;
  const int head_block = blockIdx.y;
  if (token_idx >= num_tokens || head_block >= H_BLOCKS) return;
  const int h_start = head_block * HPB;
  constexpr int VALID_HPB = NUM_HEADS < HPB ? NUM_HEADS : HPB;
  __shared__ float weight0[HPB];
  __shared__ float weight1[HPB];

  if (threadIdx.x < VALID_HPB) {
    const int local_head = threadIdx.x;
    const int h = h_start + local_head;
    const float* lse_ptr = mid_lse + ((size_t)token_idx * NUM_HEADS + h) * 2;
    const float lse0 = lse_ptr[0];
    const float lse1 = lse_ptr[1];
    float global_max = fmaxf(lse0, lse1);
    if (global_max <= -1e29f) global_max = 0.f;
    float total = (lse0 > -1e29f ? exp2f(lse0 - global_max) : 0.f) +
                  (lse1 > -1e29f ? exp2f(lse1 - global_max) : 0.f);
    if (attn_sink != nullptr) {
      const float sink_log2 = __ldg(attn_sink + h) * LOG2E;
      if (total == 0.f) {
        global_max = sink_log2;
      } else if (sink_log2 > global_max) {
        total *= exp2f(global_max - sink_log2);
        global_max = sink_log2;
      }
      total += exp2f(sink_log2 - global_max);
    }
    const float inv_total = total > 0.f ? 1.f / total : 0.f;
    weight0[local_head] = (lse0 > -1e29f ? exp2f(lse0 - global_max) : 0.f) * inv_total;
    weight1[local_head] = (lse1 > -1e29f ? exp2f(lse1 - global_max) : 0.f) * inv_total;
    out_lse[(size_t)token_idx * NUM_HEADS + h] =
        total > 0.f ? log2f(total) + global_max : -INFINITY;
  }
  __syncthreads();

  for (int vec = threadIdx.x; vec < VALID_HPB * VECS_PER_HEAD; vec += DECODE_MERGE2_THREADS) {
    const int local_head = vec / VECS_PER_HEAD;
    const int dim = (vec % VECS_PER_HEAD) * 8;
    const bf16* partial =
        mid_out + ((size_t)token_idx * NUM_HEADS + h_start + local_head) * 2 * D_V + dim;
    const float w0 = weight0[local_head];
    const float w1 = weight1[local_head];
    // An empty non-direct split writes only mid_lse, leaving mid_out unwritten.
    // Do not read its output row when the merge weight is zero.
    uint4 packed0 = make_uint4(0, 0, 0, 0);
    uint4 packed1 = make_uint4(0, 0, 0, 0);
    if (w0 > 0.f) packed0 = *reinterpret_cast<const uint4*>(partial);
    if (w1 > 0.f) packed1 = *reinterpret_cast<const uint4*>(partial + D_V);
    const __nv_bfloat162* pairs0 = reinterpret_cast<const __nv_bfloat162*>(&packed0);
    const __nv_bfloat162* pairs1 = reinterpret_cast<const __nv_bfloat162*>(&packed1);
    uint4 merged;
    __nv_bfloat162* output_pairs = reinterpret_cast<__nv_bfloat162*>(&merged);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const float2 value0 = __bfloat1622float2(pairs0[pair]);
      const float2 value1 = __bfloat1622float2(pairs1[pair]);
      output_pairs[pair] =
          __floats2bfloat162_rn(value0.x * w0 + value1.x * w1, value0.y * w0 + value1.y * w1);
    }
    bf16* final_output =
        output + ((size_t)token_idx * NUM_HEADS + h_start + local_head) * D_V + dim;
    *reinterpret_cast<uint4*>(final_output) = merged;
  }
}

}  // namespace flashinfer::sparse_mla_sm120::nvfp4
