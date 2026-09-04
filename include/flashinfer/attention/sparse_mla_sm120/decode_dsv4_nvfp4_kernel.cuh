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

#include "arch/barrier.cuh"
#include "arch/cp_async.cuh"
#include "arch/ldmatrix_sm120.cuh"
#include "arch/mma_sm120.cuh"
#include "arch/mma_sm120_nvfp4.cuh"
#include "common/d2_load_b_nvfp4.cuh"
#include "common/nvfp4_quant.cuh"
#include "model/kv_cache_traits.cuh"

namespace flashinfer::sparse_mla_sm120::nvfp4 {

constexpr int DECODE_N_WARPS = 8;
constexpr int DECODE_IO_WARPS = 2;
constexpr int DECODE_BLOCK_THREADS = (DECODE_N_WARPS + DECODE_IO_WARPS) * 32;
constexpr int DECODE_MATH_THREADS = DECODE_N_WARPS * 32;
constexpr int DECODE_MERGE2_THREADS = 512;
constexpr int DECODE_CAND_WINDOW = 64;
constexpr int DECODE_KV_BUF_COUNT = 2;
constexpr int DECODE_ENTRIES_PER_WARP = DECODE_CAND_WINDOW / DECODE_N_WARPS;
constexpr int DECODE_QK_N_TILES = DECODE_ENTRIES_PER_WARP / 8;
constexpr int DECODE_PACKED_NOPE_BYTES = 448 / 2;
constexpr int DECODE_DATA_BYTES_PER_TOKEN = DECODE_PACKED_NOPE_BYTES + 64 * sizeof(bf16);
constexpr int DECODE_SCALE_BYTES_PER_TOKEN = 32;
constexpr int DECODE_BYTES_PER_TOKEN = DECODE_DATA_BYTES_PER_TOKEN + DECODE_SCALE_BYTES_PER_TOKEN;
constexpr int DECODE_KV_SMEM_STRIDE = DECODE_PACKED_NOPE_BYTES + 16;
constexpr int DECODE_W_PACKED_STRIDE = DECODE_CAND_WINDOW / 2 + 16;
constexpr int DECODE_VT_PACKED_K_BYTES = DECODE_CAND_WINDOW / 2;
constexpr int DECODE_VT_SCALE_GROUPS = DECODE_CAND_WINDOW / SF_VEC_SIZE;
constexpr int DECODE_VT_DATA_BYTES = 448 * DECODE_VT_PACKED_K_BYTES;
constexpr int DECODE_VT_SCALE_BYTES = 448 * DECODE_VT_SCALE_GROUPS;

static_assert(DECODE_DATA_BYTES_PER_TOKEN == 352);
static_assert(DECODE_BYTES_PER_TOKEN == 384);
static_assert(DECODE_KV_SMEM_STRIDE == 240);

template <ModelType MT>
struct DecodeNVFP4Smem {
  using KV = KVCacheTraits<MT>;
  static_assert(MT == ModelType::DSV4);

  static constexpr size_t SMEM_Q_ROPE = HPB * KV::D_ROPE * sizeof(bf16);
  static constexpr size_t SMEM_Q_FP4 = HPB * DSV4_NVFP4_Q_PACKED_STRIDE;
  static constexpr size_t SMEM_Q_SC = HPB * DSV4_NVFP4_SCALE_STRIDE;
  static constexpr size_t SMEM_KV_FP4_BUF = DECODE_CAND_WINDOW * DECODE_KV_SMEM_STRIDE;
  static constexpr size_t SMEM_KV_SC_BUF = DECODE_CAND_WINDOW * DECODE_SCALE_BYTES_PER_TOKEN;
  static constexpr size_t SMEM_KV_ROPE_BUF = DECODE_CAND_WINDOW * KV::D_ROPE * sizeof(bf16);
  static constexpr size_t SMEM_MBAR_PAIR = 2 * sizeof(uint64_t);
  static constexpr size_t SMEM_REDUCE = 2 * DECODE_N_WARPS * HPB * sizeof(float);
  static constexpr size_t SMEM_W_SC = HPB * DECODE_VT_SCALE_GROUPS;
  static constexpr size_t SMEM_W_FP4 = HPB * DECODE_W_PACKED_STRIDE;
  static constexpr size_t SMEM_VT_DATA = DECODE_VT_DATA_BYTES;
  static constexpr size_t SMEM_VT_SC = DECODE_VT_SCALE_BYTES;

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
  static constexpr size_t OFF_MBAR_VT = OFF_MBAR_EMPTY + SMEM_MBAR_PAIR;
  static constexpr size_t OFF_REDUCE = OFF_MBAR_VT + sizeof(uint64_t);
  static constexpr size_t OFF_W_SC = OFF_REDUCE + SMEM_REDUCE;
  static constexpr size_t OFF_W_FP4_UNALIGNED = OFF_W_SC + SMEM_W_SC;
  static constexpr size_t OFF_W_FP4 = (OFF_W_FP4_UNALIGNED + 15) / 16 * 16;
  static constexpr size_t OFF_VT_DATA_UNALIGNED = OFF_W_FP4 + SMEM_W_FP4;
  static constexpr size_t OFF_VT_DATA = (OFF_VT_DATA_UNALIGNED + 15) / 16 * 16;
  static constexpr size_t OFF_VT_SC = OFF_VT_DATA + SMEM_VT_DATA;
  static constexpr size_t SIZE = OFF_VT_SC + SMEM_VT_SC;

  char* base;

  __device__ static DecodeNVFP4Smem init(char* base) { return DecodeNVFP4Smem{base}; }
  __device__ __forceinline__ bf16* q_rope() const {
    return reinterpret_cast<bf16*>(base + OFF_Q_ROPE);
  }
  __device__ __forceinline__ uint8_t* q_fp4() const {
    return reinterpret_cast<uint8_t*>(base + OFF_Q_FP4);
  }
  __device__ __forceinline__ uint8_t* q_sc() const {
    return reinterpret_cast<uint8_t*>(base + OFF_Q_SC);
  }
  __device__ __forceinline__ uint8_t* kv_fp4(int parity) const {
    return reinterpret_cast<uint8_t*>(base + OFF_KV_FP4 + parity * SMEM_KV_FP4_BUF);
  }
  __device__ __forceinline__ uint8_t* kv_sc(int parity) const {
    return reinterpret_cast<uint8_t*>(base + OFF_KV_SC + parity * SMEM_KV_SC_BUF);
  }
  __device__ __forceinline__ bf16* kv_rope(int parity) const {
    return reinterpret_cast<bf16*>(base + OFF_KV_ROPE + parity * SMEM_KV_ROPE_BUF);
  }
  __device__ __forceinline__ uint64_t* mbar_full(int parity) const {
    return reinterpret_cast<uint64_t*>(base + OFF_MBAR_FULL) + parity;
  }
  __device__ __forceinline__ uint64_t* mbar_empty(int parity) const {
    return reinterpret_cast<uint64_t*>(base + OFF_MBAR_EMPTY) + parity;
  }
  __device__ __forceinline__ uint64_t* mbar_vt() const {
    return reinterpret_cast<uint64_t*>(base + OFF_MBAR_VT);
  }
  __device__ __forceinline__ float* warp_max() const {
    return reinterpret_cast<float*>(base + OFF_REDUCE);
  }
  __device__ __forceinline__ float* warp_sum() const { return warp_max() + DECODE_N_WARPS * HPB; }
  __device__ __forceinline__ uint8_t* w_sc() const {
    return reinterpret_cast<uint8_t*>(base + OFF_W_SC);
  }
  __device__ __forceinline__ uint8_t* w_fp4() const {
    return reinterpret_cast<uint8_t*>(base + OFF_W_FP4);
  }
  __device__ __forceinline__ uint8_t* vt_data() const {
    return reinterpret_cast<uint8_t*>(base + OFF_VT_DATA);
  }
  __device__ __forceinline__ uint8_t* vt_sc() const {
    return reinterpret_cast<uint8_t*>(base + OFF_VT_SC);
  }
};

__device__ __forceinline__ float e2m1_code_to_float(uint8_t code) {
  constexpr float magnitude[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float value = magnitude[code & 7];
  return (code & 8) ? -value : value;
}

// Decode two packed E2M1 values with the native SM100+ conversion.  The
// prepare kernel consumes two source candidates at a time, so this avoids two
// independent scalar LUT lookups while preserving the low-/high-nibble order.
__device__ __forceinline__ float2 e2m1x2_code_to_float2(uint8_t codes) {
  uint32_t fp16x2;
  const uint32_t packed = codes;
  asm volatile(
      "{\n"
      ".reg .b8 fp4_byte;\n"
      "mov.b32 {fp4_byte, _, _, _}, %1;\n"
      "cvt.rn.f16x2.e2m1x2 %0, fp4_byte;\n"
      "}"
      : "=r"(fp16x2)
      : "r"(packed));
  const __half2 h2 = *reinterpret_cast<const __half2*>(&fp16x2);
  return __half22float2(h2);
}

__device__ __forceinline__ uint64_t transpose_e2m1_16x16_stage(uint64_t packed, int lane_in_group,
                                                               int distance, uint64_t low_mask) {
  const uint32_t packed_lo = static_cast<uint32_t>(packed);
  const uint32_t packed_hi = static_cast<uint32_t>(packed >> 32);
  const uint64_t partner =
      static_cast<uint64_t>(__shfl_xor_sync(0xffffffffu, packed_lo, distance, SF_VEC_SIZE)) |
      (static_cast<uint64_t>(__shfl_xor_sync(0xffffffffu, packed_hi, distance, SF_VEC_SIZE)) << 32);
  const int shift = distance * 4;
  if (lane_in_group & distance) {
    return ((partner & ~low_mask) >> shift) | (packed & ~low_mask);
  }
  return (packed & low_mask) | ((partner & low_mask) << shift);
}

// Transpose one 16-candidate x 16-dimension E2M1 tile entirely in registers.
// Four butterfly stages replace the per-output-lane gather of all 16 packed
// source values (8 shuffles per lane instead of 32).
__device__ __forceinline__ uint64_t transpose_e2m1_16x16(uint64_t packed, int lane_in_group) {
  packed = transpose_e2m1_16x16_stage(packed, lane_in_group, 8, 0x00000000ffffffffULL);
  packed = transpose_e2m1_16x16_stage(packed, lane_in_group, 4, 0x0000ffff0000ffffULL);
  packed = transpose_e2m1_16x16_stage(packed, lane_in_group, 2, 0x00ff00ff00ff00ffULL);
  return transpose_e2m1_16x16_stage(packed, lane_in_group, 1, 0x0f0f0f0f0f0f0f0fULL);
}

// Convert one token-major 64-candidate shared-memory tile into the
// candidate-reduction layout consumed by block-scaled P x V.  The conversion
// stays inside the CTA: source scales are absorbed, the 16x16 E2M1 tiles are
// transposed in registers, and the result is requantized into the ephemeral
// V^T operand.  KV_SMEM_STRIDE allows decode's padded source rows and
// prefill's compact rows to share the same implementation.
template <int WORKER_THREADS, int KV_SMEM_STRIDE, int THREAD_BASE = 0>
__device__ __forceinline__ void prepare_nvfp4_vt_from_smem(const uint8_t* __restrict__ kv_fp4,
                                                           const uint8_t* __restrict__ kv_sc,
                                                           uint8_t* __restrict__ vt_data,
                                                           uint8_t* __restrict__ vt_sc) {
  constexpr int NUM_DIM_GROUPS = 448 / SF_VEC_SIZE;
  constexpr int DIM_GROUPS_PER_ITER = WORKER_THREADS / 64;
  static_assert(WORKER_THREADS >= 64 && WORKER_THREADS % 64 == 0);

  const int worker_tid = threadIdx.x - THREAD_BASE;
  const int warp = worker_tid / 32;
  const int lane = worker_tid & 31;
  const int lane_in_group = lane & (SF_VEC_SIZE - 1);
  const int half_warp = lane / SF_VEC_SIZE;
  const int warp_cand_pair = warp & 1;
  const int cand_group = warp_cand_pair * 2 + half_warp;
  const int cand = cand_group * SF_VEC_SIZE + lane_in_group;

  for (int dim_group = warp / 2; dim_group < NUM_DIM_GROUPS; dim_group += DIM_GROUPS_PER_ITER) {
    const uint64_t packed = *reinterpret_cast<const uint64_t*>(
        kv_fp4 + (size_t)cand * KV_SMEM_STRIDE + dim_group * FP4_PACKED_PER_GROUP);
    const float source_scale =
        e4m3_byte_to_float(kv_sc[(size_t)cand * DECODE_SCALE_BYTES_PER_TOKEN + dim_group]);
    const uint64_t transposed = transpose_e2m1_16x16(packed, lane_in_group);

    float values[SF_VEC_SIZE];
#pragma unroll
    for (int source_pair = 0; source_pair < SF_VEC_SIZE / 2; ++source_pair) {
      const int source_lane0 = source_pair * 2;
      const int source_lane1 = source_lane0 + 1;
      const uint8_t codes = static_cast<uint8_t>(transposed >> (source_pair * 8));
      const float2 decoded = e2m1x2_code_to_float2(codes);
      const float scale0 = __shfl_sync(0xffffffffu, source_scale, source_lane0, SF_VEC_SIZE);
      const float scale1 = __shfl_sync(0xffffffffu, source_scale, source_lane1, SF_VEC_SIZE);
      values[source_lane0] = decoded.x * scale0;
      values[source_lane1] = decoded.y * scale1;
    }

    const int dim = dim_group * SF_VEC_SIZE + lane_in_group;
    uint2 quantized;
    uint8_t quantized_scale;
    quantize_fp32_group16_to_nvfp4_regs(values, quantized, quantized_scale);

    // Adjacent half-warps cover adjacent candidate groups.  Merge their
    // 8-byte results into one aligned 16-byte transaction; paired warps fill
    // the complete 64-candidate row.
    const int peer_lane = lane_in_group + SF_VEC_SIZE;
    const uint32_t peer_lo = __shfl_sync(0xffffffffu, quantized.x, peer_lane);
    const uint32_t peer_hi = __shfl_sync(0xffffffffu, quantized.y, peer_lane);
    const uint32_t peer_scale =
        __shfl_sync(0xffffffffu, static_cast<uint32_t>(quantized_scale), peer_lane);
    if (half_warp == 0) {
      *reinterpret_cast<uint4*>(vt_data + dim * DECODE_VT_PACKED_K_BYTES + warp_cand_pair * 16) =
          make_uint4(quantized.x, quantized.y, peer_lo, peer_hi);
      *reinterpret_cast<uint16_t*>(vt_sc + dim * DECODE_VT_SCALE_GROUPS + warp_cand_pair * 2) =
          static_cast<uint16_t>(quantized_scale) | static_cast<uint16_t>(peer_scale << 8);
    }
  }
}

template <ModelType MT, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE, bool DUAL_CACHE = false>
__global__ void __launch_bounds__(DECODE_BLOCK_THREADS) sparse_mla_decode_dsv4_nvfp4_kernel(
    const bf16* __restrict__ q, const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ indices, bf16* __restrict__ mid_out, float* __restrict__ mid_lse,
    bf16* __restrict__ output, float* __restrict__ out_lse, const float* __restrict__ attn_sink,
    const int* __restrict__ topk_length_ptr, const uint8_t* __restrict__ extra_kv_cache,
    const int32_t* __restrict__ extra_indices, const int* __restrict__ extra_topk_length_ptr,
    int extra_topk, int extra_page_block_size, size_t stride_extra_kv_block, int num_tokens,
    int num_splits, int chunks_per_block, float sm_scale, size_t stride_kv_block,
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
              attn_sink ? __ldg(attn_sink + h) * LOG2E : -1e30f;
        }
      } else if (threadIdx.x < VALID_HPB) {
        const int h = h_start + threadIdx.x;
        mid_lse[(size_t)token_idx * NUM_HEADS * num_splits + (size_t)h * num_splits + split_idx] =
            -1e30f;
      }
    }
    return;
  }

  extern __shared__ __align__(16) char smem_raw[];
  auto sm = DecodeNVFP4Smem<MT>::init(smem_raw);
  __shared__ bf16 sm_p_full[HPB][DECODE_CAND_WINDOW];
  const int32_t* idx_base = indices + (size_t)token_idx * TOPK;

  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < DECODE_KV_BUF_COUNT; ++i) {
      mbarrier_init(sm.mbar_full(i), 1);
      mbarrier_init(sm.mbar_empty(i), 1);
    }
  }
  __syncthreads();

  constexpr uint32_t BULK_NOPE_BYTES = DECODE_PACKED_NOPE_BYTES;
  constexpr uint32_t BULK_ROPE_BYTES = D_ROPE_C * sizeof(bf16);
  constexpr uint32_t BULK_TX_BYTES = DECODE_CAND_WINDOW * (BULK_NOPE_BYTES + BULK_ROPE_BYTES);

  auto issue_gather = [&](int chunk_idx, int buf) {
    bool is_extra = false;
    int section_chunk = chunk_idx;
    int section_len = topk_len;
    int section_topk = TOPK;
    int section_page_block_size = PAGE_BLOCK_SIZE;
    size_t section_stride = stride_kv_block;
    const uint8_t* section_kv = kv_cache;
    const int32_t* section_indices = idx_base;
    if constexpr (DUAL_CACHE) {
      is_extra = chunk_idx >= num_main_chunks;
      if (is_extra) {
        section_chunk = chunk_idx - num_main_chunks;
        section_len = extra_topk_len;
        section_topk = extra_topk;
        section_page_block_size = extra_page_block_size;
        section_stride = stride_extra_kv_block;
        section_kv = extra_kv_cache;
        section_indices = extra_indices + (size_t)token_idx * section_topk;
      }
    }
    const int chunk_start = section_chunk * DECODE_CAND_WINDOW;
    const int chunk_end = min(chunk_start + DECODE_CAND_WINDOW, section_len);
    uint8_t* fp4_dst = sm.kv_fp4(buf);
    bf16* rope_dst = sm.kv_rope(buf);
    uint8_t* scale_dst = sm.kv_sc(buf);

    const int io_warp = warp_id - DECODE_N_WARPS;
    const int entry = (io_warp & 1) * 32 + lane;
    const int cand = chunk_start + entry;
    const int idx_raw = (cand < chunk_end) ? section_indices[cand] : -1;
    if (io_warp < 2) {
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
    }

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
    bar_sync_t<4, DECODE_IO_WARPS * 32>();
    cp_async_bulk_g2s(fp4_dst + (size_t)entry * DECODE_KV_SMEM_STRIDE, data_src, BULK_NOPE_BYTES,
                      sm.mbar_full(buf));
    cp_async_bulk_g2s(rope_dst + (size_t)entry * D_ROPE_C, data_src + DECODE_PACKED_NOPE_BYTES,
                      BULK_ROPE_BYTES, sm.mbar_full(buf));
  };

  if (is_io) {
    uint32_t phase = 1;
    int state_idx = 0;
    for (int chunk = chunk_lo; chunk < chunk_hi; ++chunk) {
      const int buf = (chunk - chunk_lo) % DECODE_KV_BUF_COUNT;
      mbarrier_wait_parity(sm.mbar_empty(state_idx), phase);
      issue_gather(chunk, buf);
      if (++state_idx == DECODE_KV_BUF_COUNT) {
        state_idx = 0;
        phase ^= 1;
      }
    }
    return;
  }

  const int gid = lane >> 2;
  const int tid = lane & 3;
  const bf16* q_base = q + (size_t)token_idx * NUM_HEADS * D_QK + (size_t)h_start * D_QK;
  quantize_q_nvfp4_to_smem<DECODE_MATH_THREADS>(sm.q_fp4(), sm.q_sc(), sm.q_rope(), q_base,
                                                VALID_HPB);

  float acc_nope[PV_GROUPS_PER_WARP][PV_N8_TILES_PER_GROUP][4] = {0};
  float acc_rope[ROPE_N_TILES][4] = {0};
  float global_max[2] = {-1e30f, -1e30f};
  float global_sum[2] = {0.f, 0.f};
  uint32_t cons_phase = 0;
  int cons_idx = 0;

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
    mbarrier_wait_parity(sm.mbar_full(cons_idx), cons_phase);

    uint8_t* sm_kv_fp4 = sm.kv_fp4(buf);
    uint8_t* sm_kv_sc = sm.kv_sc(buf);
    bf16* sm_kv_rope = sm.kv_rope(buf);

    float qk[DECODE_QK_N_TILES][4] = {0};
    const int warp_first_cand = warp_id * DECODE_ENTRIES_PER_WARP;
#pragma unroll
    for (int kt = 0; kt < NUM_K64_TILES; ++kt) {
      const int scale_row = gid + (lane & 1) * 8;
      const uint32_t scale_a = *reinterpret_cast<const uint32_t*>(
          sm.q_sc() + scale_row * DSV4_NVFP4_SCALE_STRIDE + kt * 4);
      uint32_t a0, a1, a2, a3;
      ldmatrix_load_A_fp8(a0, a1, a2, a3, sm.q_fp4() + kt * 32, DSV4_NVFP4_Q_PACKED_STRIDE, lane);
#pragma unroll
      for (int nt = 0; nt < DECODE_QK_N_TILES; ++nt) {
        const int cand_row_base = warp_first_cand + nt * 8;
        const uint32_t scale_b = *reinterpret_cast<const uint32_t*>(
            sm_kv_sc + (size_t)(cand_row_base + gid) * DECODE_SCALE_BYTES_PER_TOKEN + kt * 4);
        uint32_t b0, b1;
        ldmatrix_load_B_fp8(b0, b1,
                            sm_kv_fp4 + (size_t)cand_row_base * DECODE_KV_SMEM_STRIDE + kt * 32,
                            DECODE_KV_SMEM_STRIDE, lane);
        MmaNvfp4Result r = mma_nvfp4_block_scaled_m16n8k64(
            a0, a1, a2, a3, b0, b1, qk[nt][0], qk[nt][1], qk[nt][2], qk[nt][3], scale_a, scale_b);
        qk[nt][0] = r.d0;
        qk[nt][1] = r.d1;
        qk[nt][2] = r.d2;
        qk[nt][3] = r.d3;
      }
    }

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

#pragma unroll
    for (int nt = 0; nt < DECODE_QK_N_TILES; ++nt) {
      const int c0 = warp_first_cand + nt * 8 + tid * 2;
      const int c1 = c0 + 1;
      const int abs_c0 = chunk_start + c0;
      const int abs_c1 = chunk_start + c1;
      const int idx0 = abs_c0 < section_len ? section_indices[abs_c0] : -1;
      const int idx1 = abs_c1 < section_len ? section_indices[abs_c1] : -1;
      if (abs_c0 >= chunk_end || idx0 < 0) qk[nt][0] = qk[nt][2] = -1e30f;
      if (abs_c1 >= chunk_end || idx1 < 0) qk[nt][1] = qk[nt][3] = -1e30f;
#pragma unroll
      for (int i = 0; i < 4; ++i) qk[nt][i] *= sm_scale * LOG2E;
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
      p[nt][0] = exp2f(qk[nt][0] - local_max[0]);
      p[nt][1] = exp2f(qk[nt][1] - local_max[0]);
      p[nt][2] = exp2f(qk[nt][2] - local_max[1]);
      p[nt][3] = exp2f(qk[nt][3] - local_max[1]);
      local_sum[0] += p[nt][0] + p[nt][1];
      local_sum[1] += p[nt][2] + p[nt][3];
    }
#pragma unroll
    for (int s = 2; s >= 1; s >>= 1) {
      local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], s);
      local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], s);
    }

    if (tid == 0) {
      sm.warp_max()[warp_id * HPB + gid] = local_max[0];
      sm.warp_max()[warp_id * HPB + gid + 8] = local_max[1];
      sm.warp_sum()[warp_id * HPB + gid] = local_sum[0];
      sm.warp_sum()[warp_id * HPB + gid + 8] = local_sum[1];
    }
    bar_sync_t<3, DECODE_MATH_THREADS>();
    if (threadIdx.x < VALID_HPB) {
      const int h = threadIdx.x;
      float block_max = -1e30f;
#pragma unroll
      for (int w = 0; w < DECODE_N_WARPS; ++w)
        block_max = fmaxf(block_max, sm.warp_max()[w * HPB + h]);
      float block_sum = 0.f;
#pragma unroll
      for (int w = 0; w < DECODE_N_WARPS; ++w)
        block_sum += sm.warp_sum()[w * HPB + h] * exp2f(sm.warp_max()[w * HPB + h] - block_max);
      sm.warp_max()[h] = block_max;
      sm.warp_sum()[h] = block_sum;
    }
    bar_sync_t<3, DECODE_MATH_THREADS>();

    const float block_max0 = sm.warp_max()[gid];
    const float block_max1 = sm.warp_max()[gid + 8];
    const float block_sum0 = sm.warp_sum()[gid];
    const float block_sum1 = sm.warp_sum()[gid + 8];
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
    bar_sync_t<3, DECODE_MATH_THREADS>();
    prepare_nvfp4_vt_from_smem<DECODE_MATH_THREADS, DECODE_KV_SMEM_STRIDE>(
        sm_kv_fp4, sm_kv_sc, sm.vt_data(), sm.vt_sc());
    bar_sync_t<3, DECODE_MATH_THREADS>();

    for (int task = threadIdx.x; task < HPB * DECODE_VT_SCALE_GROUPS; task += DECODE_MATH_THREADS) {
      const int head = task / DECODE_VT_SCALE_GROUPS;
      const int cand_group = task % DECODE_VT_SCALE_GROUPS;
      quantize_bf16_group16_to_nvfp4(
          &sm_p_full[head][cand_group * SF_VEC_SIZE],
          sm.w_fp4() + head * DECODE_W_PACKED_STRIDE + cand_group * FP4_PACKED_PER_GROUP,
          sm.w_sc() + head * DECODE_VT_SCALE_GROUPS + cand_group);
    }
    bar_sync_t<3, DECODE_MATH_THREADS>();

    const int p_scale_row = gid + (lane & 1) * 8;
    const uint32_t scale_a =
        *reinterpret_cast<const uint32_t*>(sm.w_sc() + p_scale_row * DECODE_VT_SCALE_GROUPS);
    uint32_t a0, a1, a2, a3;
    ldmatrix_load_A_fp8(a0, a1, a2, a3, sm.w_fp4(), DECODE_W_PACKED_STRIDE, lane);

#pragma unroll
    for (int slot = 0; slot < PV_GROUPS_PER_WARP; ++slot) {
      const int scale_group = slot * DECODE_N_WARPS + warp_id;
      if (scale_group >= PV_SCALE_GROUPS) continue;
      const int dim = scale_group * SF_VEC_SIZE;
      uint32_t b00, b01, b10, b11;
      ldmatrix_load_B_fp8(b00, b01, sm.vt_data() + dim * DECODE_VT_PACKED_K_BYTES,
                          DECODE_VT_PACKED_K_BYTES, lane);
      ldmatrix_load_B_fp8(b10, b11, sm.vt_data() + (dim + 8) * DECODE_VT_PACKED_K_BYTES,
                          DECODE_VT_PACKED_K_BYTES, lane);
      const uint32_t scale_b0 =
          *reinterpret_cast<const uint32_t*>(sm.vt_sc() + (dim + gid) * DECODE_VT_SCALE_GROUPS);
      const uint32_t scale_b1 =
          *reinterpret_cast<const uint32_t*>(sm.vt_sc() + (dim + 8 + gid) * DECODE_VT_SCALE_GROUPS);
      MmaNvfp4Result r0 = mma_nvfp4_block_scaled_m16n8k64(a0, a1, a2, a3, b00, b01, 0.f, 0.f, 0.f,
                                                          0.f, scale_a, scale_b0);
      MmaNvfp4Result r1 = mma_nvfp4_block_scaled_m16n8k64(a0, a1, a2, a3, b10, b11, 0.f, 0.f, 0.f,
                                                          0.f, scale_a, scale_b1);
      acc_nope[slot][0][0] += r0.d0;
      acc_nope[slot][0][1] += r0.d1;
      acc_nope[slot][0][2] += r0.d2;
      acc_nope[slot][0][3] += r0.d3;
      acc_nope[slot][1][0] += r1.d0;
      acc_nope[slot][1][1] += r1.d1;
      acc_nope[slot][1][2] += r1.d2;
      acc_nope[slot][1][3] += r1.d3;
    }

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

    bar_sync_t<3, DECODE_MATH_THREADS>();
    if (threadIdx.x == 0) mbarrier_arrive(sm.mbar_empty(cons_idx));
    if (++cons_idx == DECODE_KV_BUF_COUNT) {
      cons_idx = 0;
      cons_phase ^= 1;
    }
  }

  if (write_direct) {
    if (warp_id == 0 && tid == 0) {
      float lse0 = global_sum[0] > 0.f ? log2f(global_sum[0]) + global_max[0] : -1e30f;
      float lse1 = global_sum[1] > 0.f ? log2f(global_sum[1]) + global_max[1] : -1e30f;
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
      sm.warp_max()[gid] = output_scale0;
      sm.warp_sum()[gid] = lse0;
      if constexpr (VALID_HPB > 8) {
        sm.warp_max()[gid + 8] = output_scale1;
        sm.warp_sum()[gid + 8] = lse1;
      }
    }
    bar_sync_t<3, DECODE_MATH_THREADS>();
  }

  const float direct_scale0 = write_direct ? sm.warp_max()[gid] : 1.f;
  const float direct_scale1 = write_direct ? sm.warp_max()[gid + 8] : 1.f;
  const float inv_sum0 = global_sum[0] > 0.f ? direct_scale0 / global_sum[0] : 0.f;
  const float inv_sum1 = global_sum[1] > 0.f ? direct_scale1 / global_sum[1] : 0.f;
  bf16* destination =
      write_direct
          ? output + ((size_t)token_idx * NUM_HEADS + h_start) * D_V_C
          : mid_out + (((size_t)token_idx * NUM_HEADS + h_start) * (size_t)num_splits + split_idx) *
                          D_V_C;
  const size_t head_stride = write_direct ? D_V_C : (size_t)num_splits * D_V_C;

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
      out_lse[(size_t)token_idx * NUM_HEADS + h_start + gid] = sm.warp_sum()[gid];
      if constexpr (VALID_HPB > 8) {
        out_lse[(size_t)token_idx * NUM_HEADS + h_start + gid + 8] = sm.warp_sum()[gid + 8];
      }
    } else {
      const float lse0 = global_sum[0] > 0.f ? log2f(global_sum[0]) + global_max[0] : -1e30f;
      const float lse1 = global_sum[1] > 0.f ? log2f(global_sum[1]) + global_max[1] : -1e30f;
      const size_t lse_base =
          (size_t)token_idx * NUM_HEADS * num_splits + (size_t)h_start * num_splits;
      mid_lse[lse_base + (size_t)gid * num_splits + split_idx] = lse0;
      if constexpr (VALID_HPB > 8) {
        mid_lse[lse_base + (size_t)(gid + 8) * num_splits + split_idx] = lse1;
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
      if (sink_log2 > global_max) {
        total *= exp2f(global_max - sink_log2);
        global_max = sink_log2;
      }
      total += exp2f(sink_log2 - global_max);
    }
    const float inv_total = total > 0.f ? 1.f / total : 0.f;
    weight0[local_head] = (lse0 > -1e29f ? exp2f(lse0 - global_max) : 0.f) * inv_total;
    weight1[local_head] = (lse1 > -1e29f ? exp2f(lse1 - global_max) : 0.f) * inv_total;
    out_lse[(size_t)token_idx * NUM_HEADS + h] = total > 0.f ? log2f(total) + global_max : -1e30f;
  }
  __syncthreads();

  for (int vec = threadIdx.x; vec < VALID_HPB * VECS_PER_HEAD; vec += DECODE_MERGE2_THREADS) {
    const int local_head = vec / VECS_PER_HEAD;
    const int dim = (vec % VECS_PER_HEAD) * 8;
    const bf16* partial =
        mid_out + ((size_t)token_idx * NUM_HEADS + h_start + local_head) * 2 * D_V + dim;
    const float w0 = weight0[local_head];
    const float w1 = weight1[local_head];
    // A ragged split can contain no valid indices. Its LSE/weight is the
    // sentinel/zero pair, and its stage-1 accumulator is intentionally
    // irrelevant (it may contain NaN after masked MMA lanes). Do not read
    // that row: IEEE NaN * 0 would otherwise poison the merged output.
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
