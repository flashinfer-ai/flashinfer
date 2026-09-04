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

#include "decode_dsv4_nvfp4_kernel.cuh"

namespace flashinfer::sparse_mla_sm120::nvfp4 {

constexpr int PREFILL_HEAD_GROUPS = 4;
constexpr int PREFILL_HEADS_PER_CTA = PREFILL_HEAD_GROUPS * HPB;
// A 64-head CTA amortizes one selected-V transpose over four 16-head groups.
// Eight producer warps sustain the online candidate-axis requantization while
// eight math warps retain four output accumulator groups.
constexpr int PREFILL_GATHER_WARPS = 2;
constexpr int PREFILL_IO_WARPS = 8;
constexpr int PREFILL_IO_MAX_REGS = 64;
constexpr int PREFILL_MATH_MAX_REGS = 192;
constexpr int PREFILL_VT_PIPE_STAGES = 1;
constexpr int PREFILL_KV_SMEM_STRIDE = DECODE_PACKED_NOPE_BYTES;
constexpr int PREFILL_Q_FP4_STRIDE = DECODE_PACKED_NOPE_BYTES;
constexpr int PREFILL_Q_SCALE_STRIDE = DSV4_NVFP4_NUM_SCALES;
constexpr int PREFILL_W_PACKED_STRIDE = DECODE_CAND_WINDOW / 2;
constexpr int PREFILL_BLOCK_THREADS = (DECODE_N_WARPS + PREFILL_IO_WARPS) * 32;
constexpr int PREFILL_MATH_THREADS = DECODE_MATH_THREADS;
// Padding breaks the 128-byte head-to-head alias in the probability staging
// matrix.  A multiple of eight BF16 elements preserves ldmatrix alignment
// while cutting the dominant P-quant shared-load bank conflict in half.
constexpr int PREFILL_P_STRIDE = DECODE_CAND_WINDOW + 8;

// Four 16-head groups share one selected-K gather and one transient V^T tile.
// The resulting 64-head CTA amortizes online V preparation while retaining the
// native NVFP4 QK/PV atoms validated by the decode path.
struct PrefillNVFP4Smem {
  static constexpr size_t SMEM_Q_ROPE = PREFILL_HEADS_PER_CTA * 64 * sizeof(bf16);
  static constexpr size_t SMEM_Q_FP4 = PREFILL_HEADS_PER_CTA * PREFILL_Q_FP4_STRIDE;
  static constexpr size_t SMEM_Q_SC = PREFILL_HEADS_PER_CTA * PREFILL_Q_SCALE_STRIDE;
  static constexpr size_t SMEM_KV_FP4_BUF = DECODE_CAND_WINDOW * PREFILL_KV_SMEM_STRIDE;
  static constexpr size_t SMEM_KV_SC_BUF = DECODE_CAND_WINDOW * DECODE_SCALE_BYTES_PER_TOKEN;
  static constexpr size_t SMEM_KV_ROPE_BUF = DECODE_CAND_WINDOW * 64 * sizeof(bf16);
  static constexpr size_t SMEM_MBAR_PAIR = 2 * sizeof(uint64_t);
  static constexpr size_t SMEM_MBAR_VT_PIPE = PREFILL_VT_PIPE_STAGES * sizeof(uint64_t);
  static constexpr size_t SMEM_REDUCE = PREFILL_HEAD_GROUPS * DECODE_N_WARPS * HPB * sizeof(float);
  static constexpr size_t SMEM_W_SC = PREFILL_HEADS_PER_CTA * DECODE_VT_SCALE_GROUPS;
  static constexpr size_t SMEM_W_FP4 = PREFILL_HEADS_PER_CTA * PREFILL_W_PACKED_STRIDE;
  // One CTA-local full V^T tile is consumed immediately after preparation and
  // never leaves shared memory.  KV itself remains double buffered.
  static constexpr size_t SMEM_VT_DATA = PREFILL_VT_PIPE_STAGES * DECODE_VT_DATA_BYTES;
  static constexpr size_t SMEM_VT_SC = PREFILL_VT_PIPE_STAGES * DECODE_VT_SCALE_BYTES;
  static constexpr size_t SMEM_P_FULL = PREFILL_HEADS_PER_CTA * PREFILL_P_STRIDE * sizeof(bf16);

  static constexpr size_t OFF_Q_ROPE = 0;
  static constexpr size_t OFF_Q_FP4 = OFF_Q_ROPE + SMEM_Q_ROPE;
  static constexpr size_t OFF_Q_SC = OFF_Q_FP4 + SMEM_Q_FP4;
  static constexpr size_t OFF_KV_FP4 = OFF_Q_SC + SMEM_Q_SC;
  static constexpr size_t OFF_KV_SC = OFF_KV_FP4 + DECODE_KV_BUF_COUNT * SMEM_KV_FP4_BUF;
  static constexpr size_t OFF_KV_ROPE = OFF_KV_SC + DECODE_KV_BUF_COUNT * SMEM_KV_SC_BUF;
  static constexpr size_t OFF_MBAR_FULL_UNALIGNED =
      OFF_KV_ROPE + DECODE_KV_BUF_COUNT * SMEM_KV_ROPE_BUF;
  static constexpr size_t OFF_MBAR_FULL = (OFF_MBAR_FULL_UNALIGNED + 15) / 16 * 16;
  static constexpr size_t OFF_MBAR_EMPTY = OFF_MBAR_FULL + SMEM_MBAR_PAIR;
  static constexpr size_t OFF_MBAR_VT_FULL = OFF_MBAR_EMPTY + SMEM_MBAR_PAIR;
  static constexpr size_t OFF_MBAR_VT_EMPTY = OFF_MBAR_VT_FULL + SMEM_MBAR_VT_PIPE;
  static constexpr size_t OFF_REDUCE = OFF_MBAR_VT_EMPTY + SMEM_MBAR_VT_PIPE;
  // Softmax reduction is dead before P quantization starts, and P operands
  // are dead before the next chunk's reduction.  Reuse that storage for W.
  static constexpr size_t OFF_W_SC = OFF_REDUCE;
  static constexpr size_t OFF_W_FP4_UNALIGNED = OFF_W_SC + SMEM_W_SC;
  static constexpr size_t OFF_W_FP4 = (OFF_W_FP4_UNALIGNED + 15) / 16 * 16;
  static constexpr size_t OFF_W_END = OFF_W_FP4 + SMEM_W_FP4;
  static constexpr size_t OFF_REDUCE_END = OFF_REDUCE + SMEM_REDUCE;
  static constexpr size_t OFF_VT_DATA_UNALIGNED =
      OFF_W_END > OFF_REDUCE_END ? OFF_W_END : OFF_REDUCE_END;
  static constexpr size_t OFF_VT_DATA = (OFF_VT_DATA_UNALIGNED + 15) / 16 * 16;
  static constexpr size_t OFF_VT_SC = OFF_VT_DATA + SMEM_VT_DATA;
  static constexpr size_t OFF_P_FULL_UNALIGNED = OFF_VT_SC + SMEM_VT_SC;
  static constexpr size_t OFF_P_FULL = (OFF_P_FULL_UNALIGNED + 15) / 16 * 16;
  static constexpr size_t SIZE = OFF_P_FULL + SMEM_P_FULL;

  char* base;

  __device__ static PrefillNVFP4Smem init(char* base) { return PrefillNVFP4Smem{base}; }
  __device__ __forceinline__ bf16* q_rope(int group) const {
    return reinterpret_cast<bf16*>(base + OFF_Q_ROPE) + group * HPB * 64;
  }
  __device__ __forceinline__ uint8_t* q_fp4(int group) const {
    return reinterpret_cast<uint8_t*>(base + OFF_Q_FP4) + group * HPB * PREFILL_Q_FP4_STRIDE;
  }
  __device__ __forceinline__ uint8_t* q_sc(int group) const {
    return reinterpret_cast<uint8_t*>(base + OFF_Q_SC) + group * HPB * PREFILL_Q_SCALE_STRIDE;
  }
  __device__ __forceinline__ uint8_t* kv_fp4(int parity) const {
    return reinterpret_cast<uint8_t*>(base + OFF_KV_FP4) + parity * SMEM_KV_FP4_BUF;
  }
  __device__ __forceinline__ uint8_t* kv_sc(int parity) const {
    return reinterpret_cast<uint8_t*>(base + OFF_KV_SC) + parity * SMEM_KV_SC_BUF;
  }
  __device__ __forceinline__ bf16* kv_rope(int parity) const {
    return reinterpret_cast<bf16*>(base + OFF_KV_ROPE) + parity * DECODE_CAND_WINDOW * 64;
  }
  __device__ __forceinline__ uint64_t* mbar_full(int parity) const {
    return reinterpret_cast<uint64_t*>(base + OFF_MBAR_FULL) + parity;
  }
  __device__ __forceinline__ uint64_t* mbar_empty(int parity) const {
    return reinterpret_cast<uint64_t*>(base + OFF_MBAR_EMPTY) + parity;
  }
  __device__ __forceinline__ uint64_t* mbar_vt_full(int stage) const {
    return reinterpret_cast<uint64_t*>(base + OFF_MBAR_VT_FULL) + stage;
  }
  __device__ __forceinline__ uint64_t* mbar_vt_empty(int stage) const {
    return reinterpret_cast<uint64_t*>(base + OFF_MBAR_VT_EMPTY) + stage;
  }
  __device__ __forceinline__ float* warp_max() const {
    return reinterpret_cast<float*>(base + OFF_REDUCE);
  }
  __device__ __forceinline__ uint8_t* w_sc(int group) const {
    return reinterpret_cast<uint8_t*>(base + OFF_W_SC) + group * HPB * DECODE_VT_SCALE_GROUPS;
  }
  __device__ __forceinline__ uint8_t* w_fp4(int group) const {
    return reinterpret_cast<uint8_t*>(base + OFF_W_FP4) + group * HPB * PREFILL_W_PACKED_STRIDE;
  }
  __device__ __forceinline__ uint8_t* vt_data(int stage) const {
    return reinterpret_cast<uint8_t*>(base + OFF_VT_DATA) + stage * DECODE_VT_DATA_BYTES;
  }
  __device__ __forceinline__ uint8_t* vt_sc(int stage) const {
    return reinterpret_cast<uint8_t*>(base + OFF_VT_SC) + stage * DECODE_VT_SCALE_BYTES;
  }
  __device__ __forceinline__ bf16* p_full(int group) const {
    return reinterpret_cast<bf16*>(base + OFF_P_FULL) + group * HPB * PREFILL_P_STRIDE;
  }
};

static_assert(PrefillNVFP4Smem::SIZE <= 99 * 1024);

template <int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE, bool DUAL_CACHE = false>
__global__ void __launch_bounds__(PREFILL_BLOCK_THREADS, 1) sparse_mla_prefill_dsv4_nvfp4_kernel(
    const bf16* __restrict__ q, const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ indices, bf16* __restrict__ output, float* __restrict__ out_lse,
    bf16* __restrict__ mid_out, float* __restrict__ mid_lse, const float* __restrict__ attn_sink,
    const int* __restrict__ topk_length_ptr, const uint8_t* __restrict__ extra_kv_cache,
    const int32_t* __restrict__ extra_indices, const int* __restrict__ extra_topk_length_ptr,
    int extra_topk, int extra_page_block_size, size_t stride_extra_kv_block, int num_tokens,
    int num_splits, int chunks_per_block, float sm_scale, size_t stride_kv_block,
    bool write_direct) {
  static_assert(NUM_HEADS <= PREFILL_HEADS_PER_CTA || NUM_HEADS % PREFILL_HEADS_PER_CTA == 0);
  constexpr int VALID_HEAD_GROUPS =
      NUM_HEADS < PREFILL_HEADS_PER_CTA ? (NUM_HEADS + HPB - 1) / HPB : PREFILL_HEAD_GROUPS;
  constexpr int HEADS_PER_CTA = VALID_HEAD_GROUPS * HPB;
  constexpr int D_NOPE = 448;
  constexpr int D_ROPE = 64;
  constexpr int D_QK = 512;
  constexpr int D_V = 512;
  constexpr int HEAD_BLOCKS = NUM_HEADS / HEADS_PER_CTA;
  constexpr int NUM_K64_TILES = D_NOPE / 64;
  constexpr int PV_SCALE_GROUPS = D_NOPE / SF_VEC_SIZE;
  constexpr int PV_GROUPS_PER_WARP = (PV_SCALE_GROUPS + DECODE_N_WARPS - 1) / DECODE_N_WARPS;
  constexpr int PV_N8_TILES_PER_GROUP = SF_VEC_SIZE / 8;
  constexpr int ROPE_DIMS_PER_WARP = D_ROPE / DECODE_N_WARPS;
  constexpr int ROPE_N_TILES = ROPE_DIMS_PER_WARP / 8;
  constexpr int ROPE_K_ITERS = DECODE_CAND_WINDOW / 16;
  constexpr int REDUCE_GROUP_STRIDE = DECODE_N_WARPS * HPB;
  const int token_idx = blockIdx.x;
  const int head_block = blockIdx.y;
  const int split_idx = blockIdx.z;
  if (token_idx >= num_tokens || head_block >= HEAD_BLOCKS) return;
  const int h_start = head_block * HEADS_PER_CTA;
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
  const int chunk_lo = chunks_per_block > 0 ? split_idx * chunks_per_block : 0;
  const int chunk_hi =
      chunks_per_block > 0 ? min(chunk_lo + chunks_per_block, num_chunks) : num_chunks;
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const bool is_io = warp_id >= DECODE_N_WARPS;

  extern __shared__ __align__(16) char smem_raw[];
  auto sm = PrefillNVFP4Smem::init(smem_raw);
  const int32_t* idx_base = indices + (size_t)token_idx * TOPK;

  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < DECODE_KV_BUF_COUNT; ++i) {
      mbarrier_init(sm.mbar_full(i), 1);
      mbarrier_init(sm.mbar_empty(i), 1);
    }
#pragma unroll
    for (int stage = 0; stage < PREFILL_VT_PIPE_STAGES; ++stage) {
      mbarrier_init(sm.mbar_vt_full(stage), 1);
      mbarrier_init(sm.mbar_vt_empty(stage), 1);
    }
  }
  __syncthreads();

  if (chunk_lo >= num_chunks) {
    if (!is_io) {
      for (int i = threadIdx.x; i < HEADS_PER_CTA * D_V; i += PREFILL_MATH_THREADS) {
        const int head = i / D_V;
        const int dim = i - head * D_V;
        if (write_direct) {
          output[((size_t)token_idx * NUM_HEADS + h_start + head) * D_V + dim] =
              __float2bfloat16(0.f);
        } else {
          mid_out[(((size_t)token_idx * NUM_HEADS + h_start + head) * num_splits + split_idx) *
                      D_V +
                  dim] = __float2bfloat16(0.f);
        }
      }
      if (threadIdx.x < HEADS_PER_CTA) {
        const int h = h_start + threadIdx.x;
        if (write_direct) {
          out_lse[(size_t)token_idx * NUM_HEADS + h] =
              attn_sink ? __ldg(attn_sink + h) * LOG2E : -1e30f;
        } else {
          mid_lse[((size_t)token_idx * NUM_HEADS + h) * num_splits + split_idx] = -1e30f;
        }
      }
    }
    return;
  }

  constexpr uint32_t BULK_NOPE_BYTES = DECODE_PACKED_NOPE_BYTES;
  constexpr uint32_t BULK_ROPE_BYTES = D_ROPE * sizeof(bf16);
  constexpr uint32_t BULK_TX_BYTES = DECODE_CAND_WINDOW * (BULK_NOPE_BYTES + BULK_ROPE_BYTES);

  auto issue_gather = [&](int chunk_idx, int buf) {
    bool is_extra = false;
    int section_chunk = chunk_idx;
    int section_len = topk_len;
    int section_page_block_size = PAGE_BLOCK_SIZE;
    size_t section_stride = stride_kv_block;
    const uint8_t* section_kv = kv_cache;
    const int32_t* section_indices = idx_base;
    if constexpr (DUAL_CACHE) {
      is_extra = chunk_idx >= num_main_chunks;
      if (is_extra) {
        section_chunk = chunk_idx - num_main_chunks;
        section_len = extra_topk_len;
        section_page_block_size = extra_page_block_size;
        section_stride = stride_extra_kv_block;
        section_kv = extra_kv_cache;
        section_indices = extra_indices + (size_t)token_idx * extra_topk;
      }
    }
    const int chunk_start = section_chunk * DECODE_CAND_WINDOW;
    const int chunk_end = min(chunk_start + DECODE_CAND_WINDOW, section_len);
    uint8_t* fp4_dst = sm.kv_fp4(buf);
    bf16* rope_dst = sm.kv_rope(buf);
    uint8_t* scale_dst = sm.kv_sc(buf);
    const int io_warp = warp_id - DECODE_N_WARPS;
    const int entry = io_warp * 32 + lane;
    const int cand = chunk_start + entry;
    const int idx_raw = cand < chunk_end ? section_indices[cand] : -1;

    uint4 s0 = make_uint4(0, 0, 0, 0);
    uint4 s1 = make_uint4(0, 0, 0, 0);
    if (idx_raw >= 0) {
      int page;
      int slot;
      if constexpr (DUAL_CACHE) {
        if (is_extra && section_page_block_size == 2) {
          page = idx_raw >> 1;
          slot = idx_raw & 1;
        } else {
          page = idx_raw >> 6;
          slot = idx_raw & 63;
        }
      } else {
        page = idx_raw / PAGE_BLOCK_SIZE;
        slot = idx_raw - page * PAGE_BLOCK_SIZE;
      }
      const uint8_t* scale_src = section_kv + (size_t)page * section_stride +
                                 (size_t)section_page_block_size * DECODE_DATA_BYTES_PER_TOKEN +
                                 (size_t)slot * DECODE_SCALE_BYTES_PER_TOKEN;
      s0 = *reinterpret_cast<const uint4*>(scale_src);
      s1 = *reinterpret_cast<const uint4*>(scale_src + sizeof(uint4));
    }
    *reinterpret_cast<uint4*>(scale_dst + (size_t)entry * DECODE_SCALE_BYTES_PER_TOKEN) = s0;
    *reinterpret_cast<uint4*>(scale_dst + (size_t)entry * DECODE_SCALE_BYTES_PER_TOKEN +
                              sizeof(uint4)) = s1;
    __threadfence_block();

    const int idx = idx_raw >= 0 ? idx_raw : 0;
    int page;
    int slot;
    if constexpr (DUAL_CACHE) {
      if (is_extra && section_page_block_size == 2) {
        page = idx >> 1;
        slot = idx & 1;
      } else {
        page = idx >> 6;
        slot = idx & 63;
      }
    } else {
      page = idx / PAGE_BLOCK_SIZE;
      slot = idx - page * PAGE_BLOCK_SIZE;
    }
    const uint8_t* data_src =
        section_kv + (size_t)page * section_stride + (size_t)slot * DECODE_DATA_BYTES_PER_TOKEN;
    if (io_warp == 0 && lane == 0) mbarrier_arrive_expect_tx(sm.mbar_full(buf), BULK_TX_BYTES);
    bar_sync_t<4, PREFILL_GATHER_WARPS * 32>();
    cp_async_bulk_g2s(fp4_dst + (size_t)entry * PREFILL_KV_SMEM_STRIDE, data_src, BULK_NOPE_BYTES,
                      sm.mbar_full(buf));
    cp_async_bulk_g2s(rope_dst + (size_t)entry * D_ROPE, data_src + DECODE_PACKED_NOPE_BYTES,
                      BULK_ROPE_BYTES, sm.mbar_full(buf));
  };

  if (is_io) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(PREFILL_IO_MAX_REGS));
    // KV gather remains double buffered.  The complete CTA-local V^T stage is
    // released immediately after P x V, so producer can gather N+1 meanwhile.
    uint32_t empty_phase = 1;
    uint32_t full_phase = 0;
    uint32_t vt_empty_phase[PREFILL_VT_PIPE_STAGES] = {};
#pragma unroll
    for (int stage = 0; stage < PREFILL_VT_PIPE_STAGES; ++stage) vt_empty_phase[stage] = 1;
    int state_idx = 0;
    for (int chunk = chunk_lo; chunk < chunk_hi; ++chunk) {
      const int buf = (chunk - chunk_lo) % DECODE_KV_BUF_COUNT;
      mbarrier_wait_parity(sm.mbar_empty(state_idx), empty_phase);
      if (threadIdx.x < PREFILL_MATH_THREADS + PREFILL_GATHER_WARPS * 32) issue_gather(chunk, buf);
      mbarrier_wait_parity(sm.mbar_full(state_idx), full_phase);
      const int stage = chunk % PREFILL_VT_PIPE_STAGES;
      mbarrier_wait_parity(sm.mbar_vt_empty(stage), vt_empty_phase[stage]);
      prepare_nvfp4_vt_from_smem<PREFILL_IO_WARPS * 32, PREFILL_KV_SMEM_STRIDE,
                                 PREFILL_MATH_THREADS>(sm.kv_fp4(buf), sm.kv_sc(buf),
                                                       sm.vt_data(stage), sm.vt_sc(stage));
      bar_sync_t<4, PREFILL_IO_WARPS * 32>();
      if (threadIdx.x == PREFILL_MATH_THREADS) mbarrier_arrive(sm.mbar_vt_full(stage));
      vt_empty_phase[stage] ^= 1;
      if (++state_idx == DECODE_KV_BUF_COUNT) {
        state_idx = 0;
        empty_phase ^= 1;
        full_phase ^= 1;
      }
    }
    return;
  }

  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(PREFILL_MATH_MAX_REGS));

  const int gid = lane >> 2;
  const int tid = lane & 3;
#pragma unroll
  for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
    const bf16* q_base =
        q + (size_t)token_idx * NUM_HEADS * D_QK + (size_t)(h_start + group * HPB) * D_QK;
    quantize_q_nvfp4_to_smem<PREFILL_MATH_THREADS, PREFILL_Q_FP4_STRIDE, PREFILL_Q_SCALE_STRIDE>(
        sm.q_fp4(group), sm.q_sc(group), sm.q_rope(group), q_base, HPB);
  }

  float acc_nope[PREFILL_HEAD_GROUPS][PV_GROUPS_PER_WARP][PV_N8_TILES_PER_GROUP][4] = {0};
  float acc_rope[PREFILL_HEAD_GROUPS][ROPE_N_TILES][4] = {0};
  float global_max[PREFILL_HEAD_GROUPS][2];
  float global_sum[PREFILL_HEAD_GROUPS][2];
#pragma unroll
  for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
    global_max[group][0] = global_max[group][1] = -1e30f;
    global_sum[group][0] = global_sum[group][1] = 0.f;
  }

  uint32_t cons_phase = 0;
  uint32_t vt_full_phase[PREFILL_VT_PIPE_STAGES] = {};
  int cons_idx = 0;
  for (int chunk = chunk_lo; chunk < chunk_hi; ++chunk) {
    const int buf = (chunk - chunk_lo) % DECODE_KV_BUF_COUNT;
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
    const int chunk_start = section_chunk * DECODE_CAND_WINDOW;
    const int chunk_end = min(chunk_start + DECODE_CAND_WINDOW, section_len);
    mbarrier_wait_parity(sm.mbar_full(cons_idx), cons_phase);
    uint8_t* sm_kv_fp4 = sm.kv_fp4(buf);
    uint8_t* sm_kv_sc = sm.kv_sc(buf);
    bf16* sm_kv_rope = sm.kv_rope(buf);
    const int warp_first_cand = warp_id * DECODE_ENTRIES_PER_WARP;

    float qk[PREFILL_HEAD_GROUPS][DECODE_QK_N_TILES][4] = {0};
#pragma unroll
    for (int kt = 0; kt < NUM_K64_TILES; ++kt) {
      const int cand_row_base = warp_first_cand;
      const uint32_t scale_b = *reinterpret_cast<const uint32_t*>(
          sm_kv_sc + (size_t)(cand_row_base + gid) * DECODE_SCALE_BYTES_PER_TOKEN + kt * 4);
      uint32_t b0, b1;
      ldmatrix_load_B_fp8(b0, b1,
                          sm_kv_fp4 + (size_t)cand_row_base * PREFILL_KV_SMEM_STRIDE + kt * 32,
                          PREFILL_KV_SMEM_STRIDE, lane);
#pragma unroll
      for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
        const int scale_row = gid + (lane & 1) * 8;
        const uint32_t scale_a = *reinterpret_cast<const uint32_t*>(
            sm.q_sc(group) + scale_row * PREFILL_Q_SCALE_STRIDE + kt * 4);
        uint32_t a0, a1, a2, a3;
        ldmatrix_load_A_fp8(a0, a1, a2, a3, sm.q_fp4(group) + kt * 32, PREFILL_Q_FP4_STRIDE, lane);
        MmaNvfp4Result r = mma_nvfp4_block_scaled_m16n8k64(a0, a1, a2, a3, b0, b1, qk[group][0][0],
                                                           qk[group][0][1], qk[group][0][2],
                                                           qk[group][0][3], scale_a, scale_b);
        qk[group][0][0] = r.d0;
        qk[group][0][1] = r.d1;
        qk[group][0][2] = r.d2;
        qk[group][0][3] = r.d3;
      }
    }

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

    float local_max[PREFILL_HEAD_GROUPS][2];
    float local_sum[PREFILL_HEAD_GROUPS][2];
    float p[PREFILL_HEAD_GROUPS][DECODE_QK_N_TILES][4];
    float block_max_value[PREFILL_HEAD_GROUPS][2];
#pragma unroll
    for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
      const int c0 = warp_first_cand + tid * 2;
      const int c1 = c0 + 1;
      const int abs_c0 = chunk_start + c0;
      const int abs_c1 = chunk_start + c1;
      const int idx0 = abs_c0 < section_len ? section_indices[abs_c0] : -1;
      const int idx1 = abs_c1 < section_len ? section_indices[abs_c1] : -1;
      if (abs_c0 >= chunk_end || idx0 < 0) qk[group][0][0] = qk[group][0][2] = -1e30f;
      if (abs_c1 >= chunk_end || idx1 < 0) qk[group][0][1] = qk[group][0][3] = -1e30f;
#pragma unroll
      for (int i = 0; i < 4; ++i) qk[group][0][i] *= sm_scale * LOG2E;

      local_max[group][0] = fmaxf(qk[group][0][0], qk[group][0][1]);
      local_max[group][1] = fmaxf(qk[group][0][2], qk[group][0][3]);
#pragma unroll
      for (int s = 2; s >= 1; s >>= 1) {
        local_max[group][0] =
            fmaxf(local_max[group][0], __shfl_xor_sync(0xffffffff, local_max[group][0], s));
        local_max[group][1] =
            fmaxf(local_max[group][1], __shfl_xor_sync(0xffffffff, local_max[group][1], s));
      }
      p[group][0][0] = exp2f(qk[group][0][0] - local_max[group][0]);
      p[group][0][1] = exp2f(qk[group][0][1] - local_max[group][0]);
      p[group][0][2] = exp2f(qk[group][0][2] - local_max[group][1]);
      p[group][0][3] = exp2f(qk[group][0][3] - local_max[group][1]);
      local_sum[group][0] = p[group][0][0] + p[group][0][1];
      local_sum[group][1] = p[group][0][2] + p[group][0][3];
#pragma unroll
      for (int s = 2; s >= 1; s >>= 1) {
        local_sum[group][0] += __shfl_xor_sync(0xffffffff, local_sum[group][0], s);
        local_sum[group][1] += __shfl_xor_sync(0xffffffff, local_sum[group][1], s);
      }
      if (tid == 0) {
        const int base = (group * DECODE_N_WARPS + warp_id) * HPB;
        sm.warp_max()[base + gid] = local_max[group][0];
        sm.warp_max()[base + gid + 8] = local_max[group][1];
      }
    }
    bar_sync_t<3, PREFILL_MATH_THREADS>();

    if (threadIdx.x < HEADS_PER_CTA) {
      const int group = threadIdx.x / HPB;
      const int head = threadIdx.x % HPB;
      float block_max = -1e30f;
#pragma unroll
      for (int w = 0; w < DECODE_N_WARPS; ++w)
        block_max = fmaxf(block_max, sm.warp_max()[(group * DECODE_N_WARPS + w) * HPB + head]);
      const int group_base = group * REDUCE_GROUP_STRIDE;
      sm.warp_max()[group_base + head] = block_max;
    }
    bar_sync_t<3, PREFILL_MATH_THREADS>();

#pragma unroll
    for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
      const int group_base = group * REDUCE_GROUP_STRIDE;
      block_max_value[group][0] = sm.warp_max()[group_base + gid];
      block_max_value[group][1] = sm.warp_max()[group_base + gid + 8];
    }
    // All warps must retain block max in registers before warp 0's slots are
    // recycled for the rescaled local sums.
    bar_sync_t<3, PREFILL_MATH_THREADS>();

#pragma unroll
    for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
      if (tid == 0) {
        const int base = (group * DECODE_N_WARPS + warp_id) * HPB;
        sm.warp_max()[base + gid] =
            local_sum[group][0] * exp2f(local_max[group][0] - block_max_value[group][0]);
        sm.warp_max()[base + gid + 8] =
            local_sum[group][1] * exp2f(local_max[group][1] - block_max_value[group][1]);
      }
    }
    bar_sync_t<3, PREFILL_MATH_THREADS>();

    if (threadIdx.x < HEADS_PER_CTA) {
      const int group = threadIdx.x / HPB;
      const int head = threadIdx.x % HPB;
      float block_sum = 0.f;
#pragma unroll
      for (int w = 0; w < DECODE_N_WARPS; ++w)
        block_sum += sm.warp_max()[(group * DECODE_N_WARPS + w) * HPB + head];
      sm.warp_max()[group * REDUCE_GROUP_STRIDE + head] = block_sum;
    }
    bar_sync_t<3, PREFILL_MATH_THREADS>();

#pragma unroll
    for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
      const int group_base = group * REDUCE_GROUP_STRIDE;
      const float block_max0 = block_max_value[group][0];
      const float block_max1 = block_max_value[group][1];
      const float block_sum0 = sm.warp_max()[group_base + gid];
      const float block_sum1 = sm.warp_max()[group_base + gid + 8];
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
      p_group[gid * PREFILL_P_STRIDE + warp_first_cand + c0] = __float2bfloat16(w0);
      p_group[gid * PREFILL_P_STRIDE + warp_first_cand + c1] = __float2bfloat16(w1);
      p_group[(gid + 8) * PREFILL_P_STRIDE + warp_first_cand + c0] = __float2bfloat16(w2);
      p_group[(gid + 8) * PREFILL_P_STRIDE + warp_first_cand + c1] = __float2bfloat16(w3);
    }

    // P production is local to the math consumer.  In parallel the IO warps
    // transpose/requantize the gathered 64-candidate V tile in shared memory.
    bar_sync_t<3, PREFILL_MATH_THREADS>();

    for (int task = threadIdx.x; task < HEADS_PER_CTA * DECODE_VT_SCALE_GROUPS;
         task += PREFILL_MATH_THREADS) {
      const int head = task / DECODE_VT_SCALE_GROUPS;
      const int group = head / HPB;
      const int head_in_group = head % HPB;
      const int cand_group = task % DECODE_VT_SCALE_GROUPS;
      quantize_bf16_group16_to_nvfp4(
          sm.p_full(group) + head_in_group * PREFILL_P_STRIDE + cand_group * SF_VEC_SIZE,
          sm.w_fp4(group) + head_in_group * PREFILL_W_PACKED_STRIDE +
              cand_group * FP4_PACKED_PER_GROUP,
          sm.w_sc(group) + head_in_group * DECODE_VT_SCALE_GROUPS + cand_group);
    }
    bar_sync_t<3, PREFILL_MATH_THREADS>();
    const int vt_stage = chunk % PREFILL_VT_PIPE_STAGES;
    mbarrier_wait_parity(sm.mbar_vt_full(vt_stage), vt_full_phase[vt_stage]);

    uint32_t p_scale_a[PREFILL_HEAD_GROUPS];
    uint32_t p_a0[PREFILL_HEAD_GROUPS], p_a1[PREFILL_HEAD_GROUPS];
    uint32_t p_a2[PREFILL_HEAD_GROUPS], p_a3[PREFILL_HEAD_GROUPS];
#pragma unroll
    for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
      const int p_scale_row = gid + (lane & 1) * 8;
      p_scale_a[group] =
          *reinterpret_cast<const uint32_t*>(sm.w_sc(group) + p_scale_row * DECODE_VT_SCALE_GROUPS);
      ldmatrix_load_A_fp8(p_a0[group], p_a1[group], p_a2[group], p_a3[group], sm.w_fp4(group),
                          PREFILL_W_PACKED_STRIDE, lane);
    }

#pragma unroll
    for (int slot = 0; slot < PV_GROUPS_PER_WARP; ++slot) {
      const int scale_group = slot * DECODE_N_WARPS + warp_id;
      if (scale_group >= PV_SCALE_GROUPS) continue;
      const int dim = scale_group * SF_VEC_SIZE;
      const uint8_t* vt_data = sm.vt_data(vt_stage);
      const uint8_t* vt_sc = sm.vt_sc(vt_stage);
      uint32_t b00, b01, b10, b11;
      ldmatrix_load_B_fp8(b00, b01, vt_data + dim * DECODE_VT_PACKED_K_BYTES,
                          DECODE_VT_PACKED_K_BYTES, lane);
      ldmatrix_load_B_fp8(b10, b11, vt_data + (dim + 8) * DECODE_VT_PACKED_K_BYTES,
                          DECODE_VT_PACKED_K_BYTES, lane);
      const uint32_t scale_b0 =
          *reinterpret_cast<const uint32_t*>(vt_sc + (dim + gid) * DECODE_VT_SCALE_GROUPS);
      const uint32_t scale_b1 =
          *reinterpret_cast<const uint32_t*>(vt_sc + (dim + 8 + gid) * DECODE_VT_SCALE_GROUPS);
#pragma unroll
      for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
        MmaNvfp4Result r0 =
            mma_nvfp4_block_scaled_m16n8k64(p_a0[group], p_a1[group], p_a2[group], p_a3[group], b00,
                                            b01, 0.f, 0.f, 0.f, 0.f, p_scale_a[group], scale_b0);
        MmaNvfp4Result r1 =
            mma_nvfp4_block_scaled_m16n8k64(p_a0[group], p_a1[group], p_a2[group], p_a3[group], b10,
                                            b11, 0.f, 0.f, 0.f, 0.f, p_scale_a[group], scale_b1);
        acc_nope[group][slot][0][0] += r0.d0;
        acc_nope[group][slot][0][1] += r0.d1;
        acc_nope[group][slot][0][2] += r0.d2;
        acc_nope[group][slot][0][3] += r0.d3;
        acc_nope[group][slot][1][0] += r1.d0;
        acc_nope[group][slot][1][1] += r1.d1;
        acc_nope[group][slot][1][2] += r1.d2;
        acc_nope[group][slot][1][3] += r1.d3;
      }
    }

    bar_sync_t<3, PREFILL_MATH_THREADS>();
    if (threadIdx.x == 0) mbarrier_arrive(sm.mbar_vt_empty(vt_stage));
    vt_full_phase[vt_stage] ^= 1;

    const int rope_dim_base = warp_id * ROPE_DIMS_PER_WARP;
#pragma unroll
    for (int ks = 0; ks < ROPE_K_ITERS; ++ks) {
      uint32_t rope_a0[PREFILL_HEAD_GROUPS], rope_a1[PREFILL_HEAD_GROUPS];
      uint32_t rope_a2[PREFILL_HEAD_GROUPS], rope_a3[PREFILL_HEAD_GROUPS];
#pragma unroll
      for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
        ldmatrix_load_A_bf16(rope_a0[group], rope_a1[group], rope_a2[group], rope_a3[group],
                             sm.p_full(group) + ks * 16, PREFILL_P_STRIDE, lane);
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

    bar_sync_t<3, PREFILL_MATH_THREADS>();
    if (threadIdx.x == 0) mbarrier_arrive(sm.mbar_empty(cons_idx));
    if (++cons_idx == DECODE_KV_BUF_COUNT) {
      cons_idx = 0;
      cons_phase ^= 1;
    }
  }

  float* final_output_scale = reinterpret_cast<float*>(sm.p_full(0));
  if (warp_id == 0 && tid == 0) {
#pragma unroll
    for (int group = 0; group < VALID_HEAD_GROUPS; ++group) {
      float lse0 =
          global_sum[group][0] > 0.f ? log2f(global_sum[group][0]) + global_max[group][0] : -1e30f;
      float lse1 =
          global_sum[group][1] > 0.f ? log2f(global_sum[group][1]) + global_max[group][1] : -1e30f;
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
        out_lse[(size_t)token_idx * NUM_HEADS + h0] = lse0;
        out_lse[(size_t)token_idx * NUM_HEADS + h0 + 8] = lse1;
      } else {
        const size_t lse_base = (size_t)token_idx * NUM_HEADS * num_splits;
        mid_lse[lse_base + (size_t)h0 * num_splits + split_idx] = lse0;
        mid_lse[lse_base + (size_t)(h0 + 8) * num_splits + split_idx] = lse1;
      }
    }
  }
  bar_sync_t<3, PREFILL_MATH_THREADS>();

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
            : mid_out +
                  (((size_t)token_idx * NUM_HEADS + group_h_start) * num_splits + split_idx) * D_V;
    const size_t head_stride = write_direct ? D_V : (size_t)num_splits * D_V;
#pragma unroll
    for (int slot = 0; slot < PV_GROUPS_PER_WARP; ++slot) {
      const int scale_group = slot * DECODE_N_WARPS + warp_id;
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
