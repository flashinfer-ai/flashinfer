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
#include "../../common/lse.cuh"
#include "../../common/zero_row.cuh"
#include "../../compute/nvfp4_vt.cuh"
#include "../../compute/warp_tiles.cuh"
#include "../../pipeline/staged_pipeline.cuh"
#include "gather.cuh"
#include "q_stage.cuh"
#include "resources.cuh"

namespace flashinfer::sparse_mla_sm120::nvfp4 {

template <int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE, bool DUAL_CACHE = false>
__global__ void __launch_bounds__(STREAMING_BLOCK_THREADS, 1)
    sparse_mla_streaming_dsv4_nvfp4_kernel(
        const bf16* __restrict__ q, const uint8_t* __restrict__ kv_cache,
        const int32_t* __restrict__ indices, bf16* __restrict__ output, float* __restrict__ out_lse,
        bf16* __restrict__ mid_out, float* __restrict__ mid_lse,
        const float* __restrict__ attn_sink, const int* __restrict__ topk_length_ptr,
        const uint8_t* __restrict__ extra_kv_cache, const int32_t* __restrict__ extra_indices,
        const int* __restrict__ extra_topk_length_ptr, int extra_topk, int extra_page_block_size,
        size_t extra_page_stride_bytes, int num_tokens, int scratch_split_stride,
        int chunks_per_block, float sm_scale, size_t page_stride_bytes, bool write_direct,
        float lse_scale) {
  static_assert(NUM_HEADS <= STREAMING_HEADS_PER_CTA || NUM_HEADS % STREAMING_HEADS_PER_CTA == 0);
  constexpr int VALID_HEAD_GROUPS =
      NUM_HEADS < STREAMING_HEADS_PER_CTA ? (NUM_HEADS + HPB - 1) / HPB : STREAMING_HEAD_GROUPS;
  constexpr int HEADS_PER_CTA = VALID_HEAD_GROUPS * HPB;
  constexpr int D_NOPE = DSV4NVFP4Cache::D_NOPE;
  constexpr int D_ROPE = DSV4NVFP4Cache::D_ROPE;
  constexpr int D_QK = DSV4NVFP4Cache::D_QK;
  constexpr int D_V = DSV4NVFP4Cache::D_V;
  constexpr int HEAD_BLOCKS = NUM_HEADS / HEADS_PER_CTA;
  constexpr int NUM_K64_TILES = D_NOPE / 64;
  constexpr int PV_SCALE_GROUPS = D_NOPE / SF_VEC_SIZE;
  constexpr int PV_GROUPS_PER_WARP = (PV_SCALE_GROUPS + STREAMING_N_WARPS - 1) / STREAMING_N_WARPS;
  constexpr int PV_N8_TILES_PER_GROUP = SF_VEC_SIZE / 8;
  constexpr int ROPE_DIMS_PER_WARP = D_ROPE / STREAMING_N_WARPS;
  constexpr int ROPE_N_TILES = ROPE_DIMS_PER_WARP / 8;
  constexpr int ROPE_K_ITERS = STREAMING_CAND_WINDOW / 16;
  constexpr int REDUCE_GROUP_STRIDE = STREAMING_N_WARPS * HPB;
  const int token_idx = blockIdx.x;
  const int head_block = blockIdx.y;
  const int split_idx = blockIdx.z;
  if (token_idx >= num_tokens || head_block >= HEAD_BLOCKS) return;
  const int h_start = head_block * HEADS_PER_CTA;
  int topk_len = topk_length_ptr ? __ldg(topk_length_ptr + token_idx) : TOPK;
  topk_len = max(0, min(topk_len, TOPK));
  const int num_main_chunks = (topk_len + STREAMING_CAND_WINDOW - 1) / STREAMING_CAND_WINDOW;
  int extra_topk_len = 0;
  if constexpr (DUAL_CACHE) {
    extra_topk_len = extra_topk_length_ptr ? __ldg(extra_topk_length_ptr + token_idx) : extra_topk;
    extra_topk_len = max(0, min(extra_topk_len, extra_topk));
  }
  const int num_extra_chunks = (extra_topk_len + STREAMING_CAND_WINDOW - 1) / STREAMING_CAND_WINDOW;
  const int num_chunks = num_main_chunks + num_extra_chunks;
  const int chunk_lo = chunks_per_block > 0 ? split_idx * chunks_per_block : 0;
  const int chunk_hi =
      chunks_per_block > 0 ? min(chunk_lo + chunks_per_block, num_chunks) : num_chunks;
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const bool is_io = warp_id >= STREAMING_N_WARPS;

  extern __shared__ __align__(16) char smem_raw[];
  auto sm = StreamingNVFP4Smem::init(smem_raw);
  const int32_t* idx_base = indices + (size_t)token_idx * TOPK;

  using RawRing = pipeline::AsyncRing<STREAMING_KV_BUF_COUNT, 1, STREAMING_MATH_THREADS>;
  using VtRing = pipeline::AsyncRing<STREAMING_VT_PIPE_STAGES, STREAMING_IO_WARPS * 32,
                                     STREAMING_MATH_THREADS>;
  if (threadIdx.x == 0) {
    RawRing::init(sm.mbar_full(0), sm.mbar_empty(0));
    VtRing::init(sm.mbar_vt_full(0), sm.mbar_vt_empty(0));
  }
  __syncthreads();

  if (chunk_lo >= num_chunks) {
    if (!is_io) {
      for (int i = threadIdx.x; i < HEADS_PER_CTA * D_V; i += STREAMING_MATH_THREADS) {
        const int head = i / D_V;
        const int dim = i - head * D_V;
        if (write_direct) {
          output[((size_t)token_idx * NUM_HEADS + h_start + head) * D_V + dim] =
              __float2bfloat16(0.f);
        } else {
          mid_out[(((size_t)token_idx * NUM_HEADS + h_start + head) * scratch_split_stride +
                   split_idx) *
                      D_V +
                  dim] = __float2bfloat16(0.f);
        }
      }
      if (threadIdx.x < HEADS_PER_CTA) {
        const int h = h_start + threadIdx.x;
        if (write_direct) {
          out_lse[(size_t)token_idx * NUM_HEADS + h] =
              scale_output_lse(attn_sink ? __ldg(attn_sink + h) * LOG2E : -INFINITY, lse_scale);
        } else {
          mid_lse[((size_t)token_idx * NUM_HEADS + h) * scratch_split_stride + split_idx] = -1e30f;
        }
      }
    }
    return;
  }

  auto issue_gather = [&](int chunk, int buf) {
    gather_tile<PAGE_BLOCK_SIZE, DUAL_CACHE, STREAMING_KV_SMEM_STRIDE, STREAMING_GATHER_WARPS * 32>(
        kv_cache, idx_base, chunk, topk_len, false, PAGE_BLOCK_SIZE, page_stride_bytes,
        (warp_id - STREAMING_N_WARPS) * 32 + lane, sm.kv_fp4(buf), sm.kv_rope(buf), sm.kv_sc(buf),
        sm.mbar_full(buf), num_main_chunks, extra_topk_len, extra_kv_cache,
        DUAL_CACHE ? extra_indices + size_t(token_idx) * extra_topk : nullptr,
        extra_page_block_size, extra_page_stride_bytes);
  };

  if (is_io) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(STREAMING_IO_MAX_REGS));
    // KV gather remains double buffered.  The complete CTA-local V^T stage is
    // released immediately after P x V, so producer can gather N+1 meanwhile.
    uint32_t raw_free_phase = 1;
    uint32_t raw_ready_phase = 0;
    uint32_t vt_free_phase[STREAMING_VT_PIPE_STAGES] = {};
#pragma unroll
    for (int stage = 0; stage < STREAMING_VT_PIPE_STAGES; ++stage) vt_free_phase[stage] = 1;
    int raw_slot = 0;
    for (int chunk = chunk_lo; chunk < chunk_hi; ++chunk) {
      const int buf = (chunk - chunk_lo) % STREAMING_KV_BUF_COUNT;
      RawRing::Free::wait(sm.mbar_empty(raw_slot), raw_free_phase);
      if (threadIdx.x < STREAMING_MATH_THREADS + STREAMING_GATHER_WARPS * 32)
        issue_gather(chunk, buf);
      RawRing::Ready::wait(sm.mbar_full(raw_slot), raw_ready_phase);
      const int vt_slot = chunk % STREAMING_VT_PIPE_STAGES;
      VtRing::Free::wait(sm.mbar_vt_empty(vt_slot), vt_free_phase[vt_slot]);
      prepare_nvfp4_vt_from_smem<STREAMING_IO_WARPS * 32, STREAMING_KV_SMEM_STRIDE,
                                 STREAMING_MATH_THREADS>(sm.kv_fp4(buf), sm.kv_sc(buf),
                                                         sm.vt_data(vt_slot), sm.vt_sc(vt_slot));
      pipeline::RoleSync<Dsv4Nvfp4Sync::VT_PRODUCER, STREAMING_IO_WARPS * 32>::wait();
      VtRing::Ready::publish(sm.mbar_vt_full(vt_slot));
      vt_free_phase[vt_slot] ^= 1;
      if (++raw_slot == STREAMING_KV_BUF_COUNT) {
        raw_slot = 0;
        raw_free_phase ^= 1;
        raw_ready_phase ^= 1;
      }
    }
    return;
  }

  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(STREAMING_MATH_MAX_REGS));

  const int gid = lane >> 2;
  const int tid = lane & 3;
#pragma unroll
  for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
    const bf16* q_base =
        q + (size_t)token_idx * NUM_HEADS * D_QK + (size_t)(h_start + group * HPB) * D_QK;
    quantize_q_nvfp4_to_smem<STREAMING_MATH_THREADS, STREAMING_Q_FP4_STRIDE,
                             STREAMING_Q_SCALE_STRIDE>(sm.q_fp4(group), sm.q_sc(group),
                                                       sm.q_rope(group), q_base, HPB);
  }

  float acc_nope[STREAMING_HEAD_GROUPS][PV_GROUPS_PER_WARP][PV_N8_TILES_PER_GROUP][4] = {0};
  float acc_rope[STREAMING_HEAD_GROUPS][ROPE_N_TILES][4] = {0};
  float global_max[STREAMING_HEAD_GROUPS][2];
  float global_sum[STREAMING_HEAD_GROUPS][2];
#pragma unroll
  for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
    global_max[group][0] = global_max[group][1] = -1e30f;
    global_sum[group][0] = global_sum[group][1] = 0.f;
  }

  uint32_t raw_ready_phase = 0;
  uint32_t vt_ready_phase[STREAMING_VT_PIPE_STAGES] = {};
  int raw_consumer_slot = 0;
  for (int chunk = chunk_lo; chunk < chunk_hi; ++chunk) {
    const int buf = (chunk - chunk_lo) % STREAMING_KV_BUF_COUNT;
    int section_chunk = chunk;
    int section_len = topk_len;
    const int32_t* section_indices = idx_base;
    if constexpr (DUAL_CACHE) {
      if (chunk >= num_main_chunks) {
        section_chunk = chunk - num_main_chunks;
        section_len = extra_topk_len;
        section_indices = extra_indices + (size_t)token_idx * extra_topk;
      }
    }
    const int chunk_start = section_chunk * STREAMING_CAND_WINDOW;
    const int chunk_end = min(chunk_start + STREAMING_CAND_WINDOW, section_len);
    RawRing::Ready::wait(sm.mbar_full(raw_consumer_slot), raw_ready_phase);
    uint8_t* sm_kv_fp4 = sm.kv_fp4(buf);
    uint8_t* sm_kv_sc = sm.kv_sc(buf);
    bf16* sm_kv_rope = sm.kv_rope(buf);
    const int warp_first_cand = warp_id * STREAMING_ENTRIES_PER_WARP;

    float qk[STREAMING_HEAD_GROUPS][STREAMING_QK_N_TILES][4] = {0};
    qk_nvfp4_nope_16x8<VALID_HEAD_GROUPS, STREAMING_Q_FP4_STRIDE, STREAMING_Q_SCALE_STRIDE,
                       STREAMING_KV_SMEM_STRIDE, STREAMING_SCALE_BYTES_PER_TOKEN, NUM_K64_TILES>(
        qk, sm.q_fp4(0), sm.q_sc(0), sm_kv_fp4, sm_kv_sc, warp_first_cand, lane);

#pragma unroll
    for (int ks = 0; ks < D_ROPE / 16; ++ks) {
      const int cand_row_base = warp_first_cand;
      const bf16* rope_row = sm_kv_rope + (size_t)(cand_row_base + gid) * D_ROPE + ks * 16;
      const uint32_t b0 = *reinterpret_cast<const uint32_t*>(rope_row + tid * 2);
      const uint32_t b1 = *reinterpret_cast<const uint32_t*>(rope_row + tid * 2 + 8);
#pragma unroll
      for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
        uint32_t a0, a1, a2, a3;
        ldmatrix_load_A_bf16(a0, a1, a2, a3, sm.q_rope(group) + ks * 16, D_ROPE, lane);
        MmaBf16Result r = mma_bf16_m16n8k16(a0, a1, a2, a3, b0, b1, qk[group][0][0],
                                            qk[group][0][1], qk[group][0][2], qk[group][0][3]);
        qk[group][0][0] = r.d0;
        qk[group][0][1] = r.d1;
        qk[group][0][2] = r.d2;
        qk[group][0][3] = r.d3;
      }
    }

    float local_max[STREAMING_HEAD_GROUPS][2];
    float local_sum[STREAMING_HEAD_GROUPS][2];
    float p[STREAMING_HEAD_GROUPS][STREAMING_QK_N_TILES][4];
    float block_max_value[STREAMING_HEAD_GROUPS][2];
    const int c0 = warp_first_cand + tid * 2;
    const int c1 = c0 + 1;
    const int abs_c0 = chunk_start + c0;
    const int abs_c1 = chunk_start + c1;
    const int idx0 = abs_c0 < section_len ? section_indices[abs_c0] : -1;
    const int idx1 = abs_c1 < section_len ? section_indices[abs_c1] : -1;
    const bool valid_c0 = abs_c0 < chunk_end && idx0 >= 0;
    const bool valid_c1 = abs_c1 < chunk_end && idx1 >= 0;
    const float qk_scale = sm_scale * LOG2E;
#pragma unroll
    for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
      qk[group][0][0] = valid_c0 ? qk[group][0][0] * qk_scale : -1e30f;
      qk[group][0][2] = valid_c0 ? qk[group][0][2] * qk_scale : -1e30f;
      qk[group][0][1] = valid_c1 ? qk[group][0][1] * qk_scale : -1e30f;
      qk[group][0][3] = valid_c1 ? qk[group][0][3] * qk_scale : -1e30f;

      local_max[group][0] = fmaxf(qk[group][0][0], qk[group][0][1]);
      local_max[group][1] = fmaxf(qk[group][0][2], qk[group][0][3]);
#pragma unroll
      for (int s = 2; s >= 1; s >>= 1) {
        local_max[group][0] =
            fmaxf(local_max[group][0], __shfl_xor_sync(0xffffffff, local_max[group][0], s));
        local_max[group][1] =
            fmaxf(local_max[group][1], __shfl_xor_sync(0xffffffff, local_max[group][1], s));
      }
      p[group][0][0] = valid_c0 ? exp2f(qk[group][0][0] - local_max[group][0]) : 0.f;
      p[group][0][1] = valid_c1 ? exp2f(qk[group][0][1] - local_max[group][0]) : 0.f;
      p[group][0][2] = valid_c0 ? exp2f(qk[group][0][2] - local_max[group][1]) : 0.f;
      p[group][0][3] = valid_c1 ? exp2f(qk[group][0][3] - local_max[group][1]) : 0.f;
      local_sum[group][0] = p[group][0][0] + p[group][0][1];
      local_sum[group][1] = p[group][0][2] + p[group][0][3];
#pragma unroll
      for (int s = 2; s >= 1; s >>= 1) {
        local_sum[group][0] += __shfl_xor_sync(0xffffffff, local_sum[group][0], s);
        local_sum[group][1] += __shfl_xor_sync(0xffffffff, local_sum[group][1], s);
      }
      if (tid == 0) {
        const int base = (group * STREAMING_N_WARPS + warp_id) * HPB;
        sm.reduce_scratch()[base + gid] = local_max[group][0];
        sm.reduce_scratch()[base + gid + 8] = local_max[group][1];
      }
    }
    bar_sync_t<Dsv4Nvfp4Sync::MATH, STREAMING_MATH_THREADS>();

    if (threadIdx.x < HEADS_PER_CTA) {
      const int group = threadIdx.x / HPB;
      const int head = threadIdx.x % HPB;
      float block_max = -1e30f;
#pragma unroll
      for (int w = 0; w < STREAMING_N_WARPS; ++w)
        block_max =
            fmaxf(block_max, sm.reduce_scratch()[(group * STREAMING_N_WARPS + w) * HPB + head]);
      const int group_base = group * REDUCE_GROUP_STRIDE;
      sm.reduce_scratch()[group_base + head] = block_max;
    }
    bar_sync_t<Dsv4Nvfp4Sync::MATH, STREAMING_MATH_THREADS>();

#pragma unroll
    for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
      const int group_base = group * REDUCE_GROUP_STRIDE;
      block_max_value[group][0] = sm.reduce_scratch()[group_base + gid];
      block_max_value[group][1] = sm.reduce_scratch()[group_base + gid + 8];
    }
    // All warps must retain block max in registers before warp 0's slots are
    // recycled for the rescaled local sums.
    bar_sync_t<Dsv4Nvfp4Sync::MATH, STREAMING_MATH_THREADS>();

#pragma unroll
    for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
      if (tid == 0) {
        const int base = (group * STREAMING_N_WARPS + warp_id) * HPB;
        sm.reduce_scratch()[base + gid] =
            local_sum[group][0] * exp2f(local_max[group][0] - block_max_value[group][0]);
        sm.reduce_scratch()[base + gid + 8] =
            local_sum[group][1] * exp2f(local_max[group][1] - block_max_value[group][1]);
      }
    }
    bar_sync_t<Dsv4Nvfp4Sync::MATH, STREAMING_MATH_THREADS>();

    if (threadIdx.x < HEADS_PER_CTA) {
      const int group = threadIdx.x / HPB;
      const int head = threadIdx.x % HPB;
      float block_sum = 0.f;
#pragma unroll
      for (int w = 0; w < STREAMING_N_WARPS; ++w)
        block_sum += sm.reduce_scratch()[(group * STREAMING_N_WARPS + w) * HPB + head];
      sm.reduce_scratch()[group * REDUCE_GROUP_STRIDE + head] = block_sum;
    }
    bar_sync_t<Dsv4Nvfp4Sync::MATH, STREAMING_MATH_THREADS>();

#pragma unroll
    for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
      const int group_base = group * REDUCE_GROUP_STRIDE;
      const float block_max0 = block_max_value[group][0];
      const float block_max1 = block_max_value[group][1];
      const float block_sum0 = sm.reduce_scratch()[group_base + gid];
      const float block_sum1 = sm.reduce_scratch()[group_base + gid + 8];
      const float new_max0 = fmaxf(global_max[group][0], block_max0);
      const float new_max1 = fmaxf(global_max[group][1], block_max1);
      const float alpha0 =
          global_max[group][0] > -1e29f ? exp2f(global_max[group][0] - new_max0) : 0.f;
      const float alpha1 =
          global_max[group][1] > -1e29f ? exp2f(global_max[group][1] - new_max1) : 0.f;
      const float block_rescale0 = exp2f(block_max0 - new_max0);
      const float block_rescale1 = exp2f(block_max1 - new_max1);
      const float warp_rescale0 = exp2f(local_max[group][0] - new_max0);
      const float warp_rescale1 = exp2f(local_max[group][1] - new_max1);

      if (chunk > chunk_lo) {
#pragma unroll
        for (int slot = 0; slot < PV_GROUPS_PER_WARP; ++slot) {
#pragma unroll
          for (int nt = 0; nt < PV_N8_TILES_PER_GROUP; ++nt) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
              acc_nope[group][slot][nt][i] *= alpha0;
              acc_nope[group][slot][nt][i + 2] *= alpha1;
            }
          }
        }
#pragma unroll
        for (int nt = 0; nt < ROPE_N_TILES; ++nt) {
          acc_rope[group][nt][0] *= alpha0;
          acc_rope[group][nt][1] *= alpha0;
          acc_rope[group][nt][2] *= alpha1;
          acc_rope[group][nt][3] *= alpha1;
        }
        global_sum[group][0] = global_sum[group][0] * alpha0 + block_sum0 * block_rescale0;
        global_sum[group][1] = global_sum[group][1] * alpha1 + block_sum1 * block_rescale1;
      } else {
        global_sum[group][0] = block_sum0 * block_rescale0;
        global_sum[group][1] = block_sum1 * block_rescale1;
      }
      global_max[group][0] = new_max0;
      global_max[group][1] = new_max1;

      const float w0 = p[group][0][0] * warp_rescale0;
      const float w1 = p[group][0][1] * warp_rescale0;
      const float w2 = p[group][0][2] * warp_rescale1;
      const float w3 = p[group][0][3] * warp_rescale1;
      const int c0 = tid * 2;
      const int c1 = c0 + 1;
      bf16* p_group = sm.p_full(group);
      p_group[gid * STREAMING_P_STRIDE + warp_first_cand + c0] = __float2bfloat16(w0);
      p_group[gid * STREAMING_P_STRIDE + warp_first_cand + c1] = __float2bfloat16(w1);
      p_group[(gid + 8) * STREAMING_P_STRIDE + warp_first_cand + c0] = __float2bfloat16(w2);
      p_group[(gid + 8) * STREAMING_P_STRIDE + warp_first_cand + c1] = __float2bfloat16(w3);
    }

    // P production is local to the math consumer.  In parallel the IO warps
    // transpose/requantize the gathered 64-candidate V tile in shared memory.
    bar_sync_t<Dsv4Nvfp4Sync::MATH, STREAMING_MATH_THREADS>();

    for (int task = threadIdx.x; task < HEADS_PER_CTA * NVFP4_VT_SCALE_GROUPS;
         task += STREAMING_MATH_THREADS) {
      const int head = task / NVFP4_VT_SCALE_GROUPS;
      const int group = head / HPB;
      const int head_in_group = head % HPB;
      const int cand_group = task % NVFP4_VT_SCALE_GROUPS;
      quantize_group16_to_nvfp4(
          sm.p_full(group) + head_in_group * STREAMING_P_STRIDE + cand_group * SF_VEC_SIZE,
          sm.p_fp4(group) + head_in_group * STREAMING_W_PACKED_STRIDE +
              cand_group * FP4_PACKED_PER_GROUP,
          sm.w_sc(group) + head_in_group * NVFP4_VT_SCALE_GROUPS + cand_group);
    }
    bar_sync_t<Dsv4Nvfp4Sync::MATH, STREAMING_MATH_THREADS>();
    const int vt_slot = chunk % STREAMING_VT_PIPE_STAGES;
    VtRing::Ready::wait(sm.mbar_vt_full(vt_slot), vt_ready_phase[vt_slot]);

    pv_nvfp4_vt_16x16<VALID_HEAD_GROUPS, STREAMING_W_PACKED_STRIDE, STREAMING_N_WARPS>(
        acc_nope, sm.p_fp4(0), sm.w_sc(0), sm.vt_data(vt_slot), sm.vt_sc(vt_slot), warp_id, lane);

    bar_sync_t<Dsv4Nvfp4Sync::MATH, STREAMING_MATH_THREADS>();
    VtRing::Free::publish(sm.mbar_vt_empty(vt_slot));
    vt_ready_phase[vt_slot] ^= 1;

    const int rope_dim_base = warp_id * ROPE_DIMS_PER_WARP;
#pragma unroll
    for (int ks = 0; ks < ROPE_K_ITERS; ++ks) {
      uint32_t rope_a0[STREAMING_HEAD_GROUPS], rope_a1[STREAMING_HEAD_GROUPS];
      uint32_t rope_a2[STREAMING_HEAD_GROUPS], rope_a3[STREAMING_HEAD_GROUPS];
#pragma unroll
      for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
        ldmatrix_load_A_bf16(rope_a0[group], rope_a1[group], rope_a2[group], rope_a3[group],
                             sm.p_full(group) + ks * 16, STREAMING_P_STRIDE, lane);
      }
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
            *reinterpret_cast<const uint16_t*>(sm_kv_rope + (size_t)ent0 * D_ROPE + col);
        const uint16_t v1 =
            *reinterpret_cast<const uint16_t*>(sm_kv_rope + (size_t)ent1 * D_ROPE + col);
        const uint16_t v8 =
            *reinterpret_cast<const uint16_t*>(sm_kv_rope + (size_t)ent8 * D_ROPE + col);
        const uint16_t v9 =
            *reinterpret_cast<const uint16_t*>(sm_kv_rope + (size_t)ent9 * D_ROPE + col);
        const uint32_t b0 = (uint32_t)v0 | ((uint32_t)v1 << 16);
        const uint32_t b1 = (uint32_t)v8 | ((uint32_t)v9 << 16);
#pragma unroll
        for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
          MmaBf16Result r =
              mma_bf16_m16n8k16(rope_a0[group], rope_a1[group], rope_a2[group], rope_a3[group], b0,
                                b1, acc_rope[group][nt][0], acc_rope[group][nt][1],
                                acc_rope[group][nt][2], acc_rope[group][nt][3]);
          acc_rope[group][nt][0] = r.d0;
          acc_rope[group][nt][1] = r.d1;
          acc_rope[group][nt][2] = r.d2;
          acc_rope[group][nt][3] = r.d3;
        }
      }
    }

    bar_sync_t<Dsv4Nvfp4Sync::MATH, STREAMING_MATH_THREADS>();
    RawRing::Free::publish(sm.mbar_empty(raw_consumer_slot));
    RawRing::advance(raw_consumer_slot, raw_ready_phase);
  }

  float* final_output_scale = reinterpret_cast<float*>(sm.p_full(0));
  if (warp_id == 0 && tid == 0) {
#pragma unroll
    for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
      const float empty_lse = write_direct ? -INFINITY : -1e30f;
      float lse0 = global_sum[group][0] > 0.f ? log2f(global_sum[group][0]) + global_max[group][0]
                                              : empty_lse;
      float lse1 = global_sum[group][1] > 0.f ? log2f(global_sum[group][1]) + global_max[group][1]
                                              : empty_lse;
      float output_scale0 = 1.f;
      float output_scale1 = 1.f;
      const int h0 = h_start + group * HPB + gid;
      if (write_direct && attn_sink != nullptr) {
        const float sink0 = __ldg(attn_sink + h0) * LOG2E;
        const float max0 = fmaxf(lse0, sink0);
        const float attn_mass0 = lse0 > -1e29f ? exp2f(lse0 - max0) : 0.f;
        const float sink_mass0 = exp2f(sink0 - max0);
        const float total0 = attn_mass0 + sink_mass0;
        output_scale0 = total0 > 0.f ? attn_mass0 / total0 : 0.f;
        lse0 = total0 > 0.f ? log2f(total0) + max0 : -1e30f;
        const float sink1 = __ldg(attn_sink + h0 + 8) * LOG2E;
        const float max1 = fmaxf(lse1, sink1);
        const float attn_mass1 = lse1 > -1e29f ? exp2f(lse1 - max1) : 0.f;
        const float sink_mass1 = exp2f(sink1 - max1);
        const float total1 = attn_mass1 + sink_mass1;
        output_scale1 = total1 > 0.f ? attn_mass1 / total1 : 0.f;
        lse1 = total1 > 0.f ? log2f(total1) + max1 : -1e30f;
      }
      final_output_scale[group * HPB + gid] = output_scale0;
      final_output_scale[group * HPB + gid + 8] = output_scale1;
      if (write_direct) {
        out_lse[(size_t)token_idx * NUM_HEADS + h0] = scale_output_lse(lse0, lse_scale);
        out_lse[(size_t)token_idx * NUM_HEADS + h0 + 8] = scale_output_lse(lse1, lse_scale);
      } else {
        const size_t lse_base = (size_t)token_idx * NUM_HEADS * scratch_split_stride;
        mid_lse[lse_base + (size_t)h0 * scratch_split_stride + split_idx] = lse0;
        mid_lse[lse_base + (size_t)(h0 + 8) * scratch_split_stride + split_idx] = lse1;
      }
    }
  }
  bar_sync_t<Dsv4Nvfp4Sync::MATH, STREAMING_MATH_THREADS>();

#pragma unroll
  for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
    const float inv_sum0 = global_sum[group][0] > 0.f
                               ? final_output_scale[group * HPB + gid] / global_sum[group][0]
                               : 0.f;
    const float inv_sum1 = global_sum[group][1] > 0.f
                               ? final_output_scale[group * HPB + gid + 8] / global_sum[group][1]
                               : 0.f;
    const int group_h_start = h_start + group * HPB;
    bf16* destination =
        write_direct
            ? output + ((size_t)token_idx * NUM_HEADS + group_h_start) * D_V
            : mid_out + (((size_t)token_idx * NUM_HEADS + group_h_start) * scratch_split_stride +
                         split_idx) *
                            D_V;
    const size_t head_stride = write_direct ? D_V : (size_t)scratch_split_stride * D_V;
#pragma unroll
    for (int slot = 0; slot < PV_GROUPS_PER_WARP; ++slot) {
      const int scale_group = slot * STREAMING_N_WARPS + warp_id;
      if (scale_group >= PV_SCALE_GROUPS) continue;
#pragma unroll
      for (int nt = 0; nt < PV_N8_TILES_PER_GROUP; ++nt) {
        const int d0 = scale_group * SF_VEC_SIZE + nt * 8 + tid * 2;
        const __nv_bfloat162 lo = __floats2bfloat162_rn(acc_nope[group][slot][nt][0] * inv_sum0,
                                                        acc_nope[group][slot][nt][1] * inv_sum0);
        const __nv_bfloat162 hi = __floats2bfloat162_rn(acc_nope[group][slot][nt][2] * inv_sum1,
                                                        acc_nope[group][slot][nt][3] * inv_sum1);
        *reinterpret_cast<__nv_bfloat162*>(&destination[(size_t)gid * head_stride + d0]) = lo;
        *reinterpret_cast<__nv_bfloat162*>(&destination[(size_t)(gid + 8) * head_stride + d0]) = hi;
      }
    }
#pragma unroll
    for (int nt = 0; nt < ROPE_N_TILES; ++nt) {
      const int d0 = D_NOPE + warp_id * ROPE_DIMS_PER_WARP + nt * 8 + tid * 2;
      const __nv_bfloat162 lo = __floats2bfloat162_rn(acc_rope[group][nt][0] * inv_sum0,
                                                      acc_rope[group][nt][1] * inv_sum0);
      const __nv_bfloat162 hi = __floats2bfloat162_rn(acc_rope[group][nt][2] * inv_sum1,
                                                      acc_rope[group][nt][3] * inv_sum1);
      *reinterpret_cast<__nv_bfloat162*>(&destination[(size_t)gid * head_stride + d0]) = lo;
      *reinterpret_cast<__nv_bfloat162*>(&destination[(size_t)(gid + 8) * head_stride + d0]) = hi;
    }
  }
}

}  // namespace flashinfer::sparse_mla_sm120::nvfp4
