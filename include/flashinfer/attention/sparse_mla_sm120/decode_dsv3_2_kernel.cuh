// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// (license header — full text mirrors the other files in this directory)

#pragma once

#include "arch/barrier.cuh"
#include "arch/cp_async.cuh"
#include "arch/ldmatrix_sm120.cuh"
#include "arch/mma_sm120.cuh"
#include "common/d2_load_b.cuh"
#include "common/fp8_quant.cuh"
#include "common/online_softmax.cuh"
#include "common/scale_mma.cuh"
#include "model/kv_cache_traits.cuh"
#include "model/scale_convert.cuh"

namespace flashinfer::sparse_mla_sm120 {

// Sparse MLA decode (V32-family): warp-spec pipeline forked from decode-dsv4
// (1 IO + 8 math warps, double-buffered KV, per-buffer mbarrier pairs)
// adapted to 656B inline-scale traits (D_NOPE=512, V_CHUNK=128,
// N_V_CHUNKS=4, V_HAS_ROPE=false). No dual cache.
//
// KV gmem layout per token (KV_GMEM_STRIDE=656):
//   [0     : 512)  FP8 e4m3 nope, 4 tiles × 128 elements
//   [512   : 528)  4 × FP32 scale (one per 128-elem tile)
//   [528   : 656)  BF16 rope, 64 elements × 2B
// DSv3.2 stores power-of-2 FP32 scales; GLM_NSA stores arbitrary FP32 scales.
// The IO warp does a single bulk per token that covers both nope and inline
// scales in one go (528 B), then a second bulk for rope (128 B). No scalar
// scale gather phase.
//
// Mbarrier pattern matches decode-dsv4: mbar_full[s] for IO→math (leader
// arrives with expect_tx, bulk completion decrements tx), mbar_empty[s]
// for math→IO drain. CTA-wide acq-rel sync (bar_sync<3, MATH_THREADS>)
// after mbarrier_wait_parity is required since the mbarrier wait has no
// implicit memory fence.

constexpr int DSV3_2_N_WARPS = 8;  // math warps
constexpr int DSV3_2_IO_WARPS = 1;
constexpr int DSV3_2_N_TOTAL_WARPS = DSV3_2_N_WARPS + DSV3_2_IO_WARPS;  // 9
constexpr int DSV3_2_BLOCK_THREADS = DSV3_2_N_TOTAL_WARPS * 32;         // 288
constexpr int DSV3_2_MATH_THREADS = DSV3_2_N_WARPS * 32;                // 256
constexpr int DSV3_2_IO_THREADS = DSV3_2_IO_WARPS * 32;                 // 32
constexpr int DSV3_2_CAND_WINDOW = 64;
constexpr int DSV3_2_BI = DSV3_2_CAND_WINDOW;
constexpr int DSV3_2_KV_BUF_COUNT = 2;
constexpr int DSV3_2_ENTRIES_PER_WARP = DSV3_2_BI / DSV3_2_N_WARPS;  // 8
constexpr int DSV3_2_QK_N_TILES = DSV3_2_ENTRIES_PER_WARP / 8;       // 1

struct DecodeDsv3_2Smem {
  using KV = KVCacheTraits<ModelType::DSV3_2>;

  static constexpr int N_V_CHUNKS = KV::D_NOPE / KV::QUANT_TILE;
  static constexpr size_t SMEM_Q_ROPE = HPB * KV::D_ROPE * sizeof(bf16);
  static constexpr size_t SMEM_Q_FP8 = HPB * KV::Q_NOPE_STRIDE;
  static constexpr size_t SMEM_Q_SC = HPB * KV::NUM_SCALES * sizeof(float);
  static constexpr size_t SMEM_KV_FP8_BUF = DSV3_2_BI * KV::KV_SMEM_STRIDE;
  static constexpr size_t SMEM_KV_ROPE_BUF = DSV3_2_BI * KV::D_ROPE * sizeof(bf16);
  static constexpr size_t SMEM_MBAR_PAIR = 2 * sizeof(uint64_t);
  static constexpr size_t SMEM_REDUCE = 2 * DSV3_2_N_WARPS * HPB * sizeof(float);
  static constexpr size_t SMEM_W_HEAD_SC = N_V_CHUNKS * HPB * sizeof(float);
  static constexpr size_t SMEM_W_FP8_BUF = HPB * (DSV3_2_BI + 16);

  static constexpr size_t OFF_Q_ROPE = 0;
  static constexpr size_t OFF_Q_FP8 = OFF_Q_ROPE + SMEM_Q_ROPE;
  static constexpr size_t OFF_Q_SC = OFF_Q_FP8 + SMEM_Q_FP8;
  static constexpr size_t OFF_KV_FP8 = OFF_Q_SC + SMEM_Q_SC;
  static constexpr size_t OFF_KV_ROPE = OFF_KV_FP8 + DSV3_2_KV_BUF_COUNT * SMEM_KV_FP8_BUF;
  static constexpr size_t OFF_MBAR_FULL_UNALIGNED =
      OFF_KV_ROPE + DSV3_2_KV_BUF_COUNT * SMEM_KV_ROPE_BUF;
  static constexpr size_t OFF_MBAR_FULL = (OFF_MBAR_FULL_UNALIGNED + 15) / 16 * 16;
  static constexpr size_t OFF_MBAR_EMPTY = OFF_MBAR_FULL + SMEM_MBAR_PAIR;
  static constexpr size_t OFF_REDUCE = OFF_MBAR_EMPTY + SMEM_MBAR_PAIR;
  static constexpr size_t OFF_W_HEAD_SC = OFF_REDUCE + SMEM_REDUCE;
  static constexpr size_t OFF_W_FP8 = OFF_W_HEAD_SC + SMEM_W_HEAD_SC;

  char* base;

  __device__ static DecodeDsv3_2Smem init(char* base) { return DecodeDsv3_2Smem{base}; }
  __device__ __forceinline__ bf16* q_rope() const {
    return reinterpret_cast<bf16*>(base + OFF_Q_ROPE);
  }
  __device__ __forceinline__ uint8_t* q_fp8() const {
    return reinterpret_cast<uint8_t*>(base + OFF_Q_FP8);
  }
  __device__ __forceinline__ float* q_sc() const {
    return reinterpret_cast<float*>(base + OFF_Q_SC);
  }
  __device__ __forceinline__ uint8_t* kv_fp8(int i) const {
    return reinterpret_cast<uint8_t*>(base + OFF_KV_FP8 + i * SMEM_KV_FP8_BUF);
  }
  __device__ __forceinline__ bf16* kv_rope(int i) const {
    return reinterpret_cast<bf16*>(base + OFF_KV_ROPE + i * SMEM_KV_ROPE_BUF);
  }
  __device__ __forceinline__ uint64_t* mbar_full(int i) const {
    return reinterpret_cast<uint64_t*>(base + OFF_MBAR_FULL) + i;
  }
  __device__ __forceinline__ uint64_t* mbar_empty(int i) const {
    return reinterpret_cast<uint64_t*>(base + OFF_MBAR_EMPTY) + i;
  }
  __device__ __forceinline__ float* reduce() const {
    return reinterpret_cast<float*>(base + OFF_REDUCE);
  }
  __device__ __forceinline__ float* warp_max() const { return reduce(); }
  __device__ __forceinline__ float* warp_sum() const { return reduce() + DSV3_2_N_WARPS * HPB; }
  __device__ __forceinline__ float* w_head_sc() const {
    return reinterpret_cast<float*>(base + OFF_W_HEAD_SC);
  }
  __device__ __forceinline__ uint8_t* w_fp8(int parity) const {
    return reinterpret_cast<uint8_t*>(base + OFF_W_FP8 + parity * SMEM_W_FP8_BUF);
  }
};

// No minBlocksPerSM hint on launch_bounds: kernel is smem-bound at 1
// block/SM regardless.
template <ModelType MT, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE>
__global__ void __launch_bounds__(DSV3_2_BLOCK_THREADS) sparse_mla_decode_dsv3_2_kernel(
    const bf16* __restrict__ Q,               // [num_tokens, num_heads, d_qk=576] bf16
    const uint8_t* __restrict__ KV_cache,     // FP8 paged (V32 INLINE layout, 656 B/token)
    const int32_t* __restrict__ indices,      // [num_tokens, topk] int32
    bf16* __restrict__ mid_out,               // [num_tokens, num_heads, num_splits, d_v=512] bf16
    float* __restrict__ mid_lse,              // [num_tokens, num_heads, num_splits] f32
    const int* __restrict__ topk_length_ptr,  // [num_tokens] or null
    int num_tokens, int num_splits, int chunks_per_block, float sm_scale, size_t stride_kv_block) {
  using KV = KVCacheTraits<MT>;
  static_assert(KV::D_QK == 576);
  constexpr int D_NOPE = KV::D_NOPE;                                // 512
  constexpr int D_ROPE_C = KV::D_ROPE;                              // 64
  constexpr int D_QK = KV::D_QK;                                    // 576
  constexpr int D_V_C = KV::D_V;                                    // 512
  constexpr int QUANT_TILE = KV::QUANT_TILE;                        // 128
  constexpr int NUM_SCALES = KV::NUM_SCALES;                        // 4
  constexpr int Q_NOPE_STRIDE = KV::Q_NOPE_STRIDE;                  // 528
  constexpr int KV_SMEM_STRIDE = KV::KV_SMEM_STRIDE;                // 528
  constexpr int SCALE_BYTES_PER_TOKEN = KV::SCALE_BYTES_PER_TOKEN;  // 16 (4 × FP32)
  // INLINE: bulk covers nope + scales together (528 B), rope follows at
  // gmem offset KV_ROPE_GMEM_OFFSET=528 with 128 B/entry.
  constexpr int KV_ROPE_OFFSET = KV::KV_ROPE_GMEM_OFFSET;  // 528
  constexpr int pbs = PAGE_BLOCK_SIZE;
  // Heads actually populated per CTA tile. For NUM_HEADS=8 (small TP),
  // only the first 8 head slots carry valid data; the kernel computes a
  // full HPB×CAND tile internally (zero-padded Q rows on invalid heads)
  // but only writes back NUM_HEADS heads.
  constexpr int VALID_HPB = (NUM_HEADS < HPB) ? NUM_HEADS : HPB;

  const int t_idx = blockIdx.x;
  const int h_block_idx = blockIdx.y;
  const int split_idx = blockIdx.z;
  if (t_idx >= num_tokens) return;

  const int h_start = h_block_idx * HPB;
  int topk_len = topk_length_ptr ? __ldg(topk_length_ptr + t_idx) : TOPK;
  topk_len = topk_len < 0 ? 0 : (topk_len > TOPK ? TOPK : topk_len);

  // Chunk range this block owns.
  const int num_chunks_total = (topk_len + DSV3_2_CAND_WINDOW - 1) / DSV3_2_CAND_WINDOW;
  const int chunk_lo = split_idx * chunks_per_block;
  const int chunk_hi = min(chunk_lo + chunks_per_block, num_chunks_total);

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const bool is_io = (warp_id >= DSV3_2_N_WARPS);

  // Early-exit splits: only math threads write LSE; IO has nothing to do.
  if (chunk_lo >= num_chunks_total) {
    if (!is_io && threadIdx.x < VALID_HPB) {
      const int h = h_start + threadIdx.x;
      const size_t lse_off =
          (size_t)t_idx * NUM_HEADS * num_splits + (size_t)h * num_splits + split_idx;
      mid_lse[lse_off] = -1e30f;
    }
    return;
  }

  constexpr int V_CHUNK = QUANT_TILE;                           // 128
  constexpr int N_V_CHUNKS = D_NOPE / V_CHUNK;                  // 4
  constexpr int NT_PER_WARP_XV = V_CHUNK / 8 / DSV3_2_N_WARPS;  // 2
  constexpr int XV_KSTEPS = DSV3_2_BI / 32;                     // 2
  constexpr int W_FP8_STRIDE = DSV3_2_BI + 16;                  // 80

  // ── Dynamic smem layout ────────────────────────────────────────
  // Single-buffer Q/scratch + double-buffered KV bufs.
  //   sm_q_rope    HPB * D_ROPE * 2B               = 2 KB
  //   sm_q_fp8     HPB * Q_NOPE_STRIDE             = 8.25 KB
  //   sm_q_sc      HPB * NUM_SCALES * 4B           = 256 B
  //   sm_kv_fp8    2 * BI * KV_SMEM_STRIDE         = 66 KB  (NoPE + INLINE scales)
  //   sm_kv_rope   2 * BI * D_ROPE * 2B            = 16 KB
  //   sm_reduce    2 * DSV3_2_N_WARPS * HPB * 4        = 1 KB
  //   sm_w_head_sc N_V_CHUNKS * HPB * 4            = 256 B
  //   sm_w_fp8 ×2  2 * HPB * (BI + 16)             = 2.5 KB
  //   Plus static sm_p_full HPB * BI * 2B          = 2 KB
  //   Total                                        ~ 98 KB
  // Under the 99 KB sm120a dynamic carveout, 1 block/SM.
  //
  // V32 has no separate sm_kv_sc_buf: the bulk that lands nope into
  // sm_kv_fp8 also lands the inline scale bytes (Q_NOPE_STRIDE = 528 =
  // D_NOPE 512 + SCALE_BYTES_PER_TOKEN 16), so the QK / XV stages read
  // scales directly out of sm_kv_fp8.
  extern __shared__ __align__(16) char smem_raw[];
  auto sm = DecodeDsv3_2Smem::init(smem_raw);

  __shared__ bf16 sm_p_full[HPB][DSV3_2_BI];  // 2 KB static
  const int32_t* idx_base = indices + (size_t)t_idx * TOPK;

  // ── Per-stage mbarrier init.
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < DSV3_2_KV_BUF_COUNT; ++s) {
      mbarrier_init(sm.mbar_full(s), 1);
      mbarrier_init(sm.mbar_empty(s), 1);
    }
  }
  __syncthreads();

  // ── TMA bulk constants ──
  // Per entry: NoPE+scales together (528 B) → sm_kv_fp8; RoPE (128 B) → sm_kv_rope.
  constexpr uint32_t V2_BULK_NOPESC_BYTES = (uint32_t)KV_SMEM_STRIDE;         // 528
  constexpr uint32_t V2_BULK_ROPE_BYTES = (uint32_t)D_ROPE_C * sizeof(bf16);  // 128
  constexpr uint32_t V2_BULK_TX_BYTES =
      (uint32_t)DSV3_2_BI * (V2_BULK_NOPESC_BYTES + V2_BULK_ROPE_BYTES);

  // IO gather: expect_tx + bulks. No scalar scale phase — scales travel
  // inside the NoPE+scales bulk.
  auto issue_gather = [&](int gather_chunk_idx, int buf) {
    const int g_start = gather_chunk_idx * DSV3_2_CAND_WINDOW;
    const int g_end = min(g_start + DSV3_2_CAND_WINDOW, topk_len);
    uint8_t* kv_fp8_dst = sm.kv_fp8(buf);
    bf16* kv_rope_dst = sm.kv_rope(buf);

    if (lane == 0) {
      mbarrier_arrive_expect_tx(sm.mbar_full(buf), V2_BULK_TX_BYTES);
    }

#pragma unroll
    for (int eo = 0; eo < DSV3_2_BI; eo += DSV3_2_IO_THREADS) {
      const int entry_idx = eo + lane;
      if (entry_idx >= DSV3_2_BI) break;
      const int cand_pos = g_start + entry_idx;
      const int idx_raw = (cand_pos < g_end) ? idx_base[cand_pos] : -1;
      const int idx = (idx_raw >= 0) ? idx_raw : 0;
      const int block_idx_g = idx / pbs;
      const int local_idx_g = idx - block_idx_g * pbs;
      const uint8_t* data_base = KV_cache + (size_t)block_idx_g * stride_kv_block +
                                 (size_t)local_idx_g * KV::KV_GMEM_STRIDE;
      // Bulk 1: NoPE + INLINE scales (528 B) → sm_kv_fp8 slot.
      cp_async_bulk_g2s(kv_fp8_dst + (size_t)entry_idx * KV_SMEM_STRIDE, data_base,
                        V2_BULK_NOPESC_BYTES, sm.mbar_full(buf));
      // Bulk 2: RoPE (128 B) → sm_kv_rope slot.
      cp_async_bulk_g2s(kv_rope_dst + (size_t)entry_idx * D_ROPE_C, data_base + KV_ROPE_OFFSET,
                        V2_BULK_ROPE_BYTES, sm.mbar_full(buf));
    }
  };

  if (is_io) {
    // Producer state: phase starts at 1 so first empty.wait(1) on a fresh
    // barrier (phase 0) returns immediately.
    uint32_t prod_phase = 1;
    int prod_idx = 0;
    for (int chunk_idx = chunk_lo; chunk_idx < chunk_hi; ++chunk_idx) {
      const int buf = (chunk_idx - chunk_lo) & 1;
      mbarrier_wait_parity(sm.mbar_empty(prod_idx), prod_phase);
      issue_gather(chunk_idx, buf);
      ++prod_idx;
      if (prod_idx == DSV3_2_KV_BUF_COUNT) {
        prod_idx = 0;
        prod_phase ^= 1;
      }
    }
    return;
  }

  // ──────────────────────────────────────────────────────────────
  // Math warps branch (warp_id < DSV3_2_N_WARPS = 8)
  // ──────────────────────────────────────────────────────────────

  const int gid = lane >> 2;
  const int tid = lane & 3;

  // Stage 0: Q quantization.
  const bf16* q_base = Q + (size_t)t_idx * NUM_HEADS * D_QK + (size_t)h_start * D_QK;
  quantize_q_to_smem<MT, DSV3_2_MATH_THREADS>(sm.q_fp8(), sm.q_sc(), sm.q_rope(), q_base,
                                              sm.reduce(), VALID_HPB);

  // Persistent state across chunks (per-thread registers).
  float acc_nope[N_V_CHUNKS][NT_PER_WARP_XV][4] = {0};
  float global_max[2] = {-1e30f, -1e30f};
  float global_sum[2] = {0.f, 0.f};

  // ── Chunk loop ─────────────────────────────────────────────────
  uint32_t cons_phase = 0;
  int cons_idx = 0;

  for (int chunk_idx = chunk_lo; chunk_idx < chunk_hi; ++chunk_idx) {
    const int buf = (chunk_idx - chunk_lo) & 1;
    const int split_cand_start = chunk_idx * DSV3_2_CAND_WINDOW;
    const int split_cand_end = min(split_cand_start + DSV3_2_CAND_WINDOW, topk_len);

    mbarrier_wait_parity(sm.mbar_full(cons_idx), cons_phase);
    bar_sync_t<3, DSV3_2_MATH_THREADS>();

    uint8_t* sm_kv_fp8 = sm.kv_fp8(buf);
    bf16* sm_kv_rope = sm.kv_rope(buf);

    // ── Stage 2 QK ────────────────────────────────────────────
    // K-side scales are inline at byte offset D_NOPE..D_NOPE+SCALE_BYTES_PER_TOKEN
    // within each KV_SMEM_STRIDE row.
    auto kv_scale_fp32 = [&](int cand, int blk) -> float {
      return *reinterpret_cast<const float*>(sm_kv_fp8 + (size_t)cand * KV_SMEM_STRIDE + D_NOPE +
                                             (size_t)blk * sizeof(float));
    };
    float qk[DSV3_2_QK_N_TILES][4] = {0};
    {
      const int warp_first_cand = warp_id * DSV3_2_ENTRIES_PER_WARP;
#pragma unroll
      for (int blk = 0; blk < NUM_SCALES; blk++) {
        uint8_t sfa = fp32_to_ue8m0(sm.q_sc()[(gid + (lane & 1) * 8) * NUM_SCALES + blk]);
#pragma unroll
        for (int nt = 0; nt < DSV3_2_QK_N_TILES; nt++) {
          const int cand_row_base = warp_first_cand + nt * 8;
          float acc0, acc1, acc2, acc3;
          init_qk_acc<KV::SCALE_FORMAT>(qk[nt], acc0, acc1, acc2, acc3);
          const uint8_t* k_scale_base =
              sm_kv_fp8 + (size_t)(cand_row_base + gid) * KV_SMEM_STRIDE + D_NOPE;
          uint8_t sfb = qk_k_scale_selector<KV>(k_scale_base, blk);
#pragma unroll
          for (int ks = 0; ks < QUANT_TILE / 32; ks++) {
            const int ko = blk * QUANT_TILE + ks * 32;
            uint32_t a0, a1, a2, a3, b0, b1;
            ldmatrix_load_A_fp8(a0, a1, a2, a3, sm.q_fp8() + ko, Q_NOPE_STRIDE, lane);
            ldmatrix_load_B_fp8(b0, b1, sm_kv_fp8 + (size_t)cand_row_base * KV_SMEM_STRIDE + ko,
                                KV_SMEM_STRIDE, lane);
            MmaFp8Result r = mma_fp8_block_scaled_m16n8k32(a0, a1, a2, a3, b0, b1, acc0, acc1, acc2,
                                                           acc3, sfa, sfb);
            acc0 = r.d0;
            acc1 = r.d1;
            acc2 = r.d2;
            acc3 = r.d3;
          }
          const int c0 = cand_row_base + tid * 2;
          const int c1 = c0 + 1;
          commit_qk_acc<KV>(qk[nt], acc0, acc1, acc2, acc3,
                            sm_kv_fp8 + (size_t)c0 * KV_SMEM_STRIDE + D_NOPE,
                            sm_kv_fp8 + (size_t)c1 * KV_SMEM_STRIDE + D_NOPE, blk);
        }
      }
    }
    {
      // K-rope B-operand loaded per-lane via scalar reads (ldmatrix.x2.trans
      // is wrong-layout for the N-outer rope smem; see decode-dsv4 note).
      const int warp_first_cand = warp_id * DSV3_2_ENTRIES_PER_WARP;
#pragma unroll
      for (int ks = 0; ks < D_ROPE_C / 16; ks++) {
        uint32_t a0, a1, a2, a3;
        ldmatrix_load_A_bf16(a0, a1, a2, a3, sm.q_rope() + ks * 16, D_ROPE_C, lane);
#pragma unroll
        for (int nt = 0; nt < DSV3_2_QK_N_TILES; nt++) {
          const int cand_row_base = warp_first_cand + nt * 8;
          const int entry = cand_row_base + gid;
          const bf16* kv_rope_row = sm_kv_rope + (size_t)entry * D_ROPE_C + ks * 16;
          uint32_t b0 = *reinterpret_cast<const uint32_t*>(kv_rope_row + tid * 2);
          uint32_t b1 = *reinterpret_cast<const uint32_t*>(kv_rope_row + tid * 2 + 8);
          MmaBf16Result r =
              mma_bf16_m16n8k16(a0, a1, a2, a3, b0, b1, qk[nt][0], qk[nt][1], qk[nt][2], qk[nt][3]);
          qk[nt][0] = r.d0;
          qk[nt][1] = r.d1;
          qk[nt][2] = r.d2;
          qk[nt][3] = r.d3;
        }
      }
    }

    // Mask invalid cands + sm_scale × LOG2E. Invalid = absolute cand position
    // past the per-token topk_length OR slot id = -1 (indexer-padded). The
    // IO warp gathered slot 0 into smem for the -1 case (idx clamped to 0);
    // setting qk = -inf kills the contribution in softmax.
    const int warp_first_cand = warp_id * DSV3_2_ENTRIES_PER_WARP;
#pragma unroll
    for (int nt = 0; nt < DSV3_2_QK_N_TILES; nt++) {
      const int c0 = warp_first_cand + nt * 8 + tid * 2;
      const int c1 = c0 + 1;
      const int abs_c0 = c0 + split_cand_start;
      const int abs_c1 = c1 + split_cand_start;
      const int idx0 = (abs_c0 < topk_len) ? idx_base[abs_c0] : -1;
      const int idx1 = (abs_c1 < topk_len) ? idx_base[abs_c1] : -1;
      if (abs_c0 >= split_cand_end || idx0 < 0) {
        qk[nt][0] = -1e30f;
        qk[nt][2] = -1e30f;
      }
      if (abs_c1 >= split_cand_end || idx1 < 0) {
        qk[nt][1] = -1e30f;
        qk[nt][3] = -1e30f;
      }
      qk[nt][0] *= sm_scale * LOG2E;
      qk[nt][1] *= sm_scale * LOG2E;
      qk[nt][2] *= sm_scale * LOG2E;
      qk[nt][3] *= sm_scale * LOG2E;
    }

    // Per-warp local max/sum.
    float local_max[2] = {-1e30f, -1e30f};
#pragma unroll
    for (int nt = 0; nt < DSV3_2_QK_N_TILES; nt++) {
      local_max[0] = fmaxf(local_max[0], fmaxf(qk[nt][0], qk[nt][1]));
      local_max[1] = fmaxf(local_max[1], fmaxf(qk[nt][2], qk[nt][3]));
    }
#pragma unroll
    for (int s = 2; s >= 1; s >>= 1) {
      local_max[0] = fmaxf(local_max[0], __shfl_xor_sync(0xffffffff, local_max[0], s));
      local_max[1] = fmaxf(local_max[1], __shfl_xor_sync(0xffffffff, local_max[1], s));
    }
    const bool valid_half0 = local_max[0] > -1e29f;
    const bool valid_half1 = local_max[1] > -1e29f;
    float local_sum[2] = {0.f, 0.f};
    float p[DSV3_2_QK_N_TILES][4];
#pragma unroll
    for (int nt = 0; nt < DSV3_2_QK_N_TILES; nt++) {
      p[nt][0] = valid_half0 ? exp2f(qk[nt][0] - local_max[0]) : 0.f;
      p[nt][1] = valid_half0 ? exp2f(qk[nt][1] - local_max[0]) : 0.f;
      p[nt][2] = valid_half1 ? exp2f(qk[nt][2] - local_max[1]) : 0.f;
      p[nt][3] = valid_half1 ? exp2f(qk[nt][3] - local_max[1]) : 0.f;
      local_sum[0] += p[nt][0] + p[nt][1];
      local_sum[1] += p[nt][2] + p[nt][3];
    }
#pragma unroll
    for (int s = 2; s >= 1; s >>= 1) {
      local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], s);
      local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], s);
    }

    // Cross-warp reduce.
    if (tid == 0) {
      sm.warp_max()[warp_id * HPB + gid] = local_max[0];
      sm.warp_max()[warp_id * HPB + gid + 8] = local_max[1];
      sm.warp_sum()[warp_id * HPB + gid] = local_sum[0];
      sm.warp_sum()[warp_id * HPB + gid + 8] = local_sum[1];
    }
    bar_sync_t<3, DSV3_2_MATH_THREADS>();
    if (threadIdx.x < VALID_HPB) {
      const int h = threadIdx.x;
      float wmax[DSV3_2_N_WARPS], wsum[DSV3_2_N_WARPS];
#pragma unroll
      for (int w = 0; w < DSV3_2_N_WARPS; w++) {
        wmax[w] = sm.warp_max()[w * HPB + h];
        wsum[w] = sm.warp_sum()[w * HPB + h];
      }
      float bmax = -1e30f;
#pragma unroll
      for (int w = 0; w < DSV3_2_N_WARPS; w++) bmax = fmaxf(bmax, wmax[w]);
      float bsum = 0.f;
#pragma unroll
      for (int w = 0; w < DSV3_2_N_WARPS; w++) bsum += wsum[w] * exp2f(wmax[w] - bmax);
      sm.warp_max()[h] = bmax;
      sm.warp_sum()[h] = bsum;
    }
    bar_sync_t<3, DSV3_2_MATH_THREADS>();

    const float block_local_max0 = sm.warp_max()[gid];
    const float block_local_max1 = sm.warp_max()[gid + 8];
    const float block_local_sum0 = sm.warp_sum()[gid];
    const float block_local_sum1 = sm.warp_sum()[gid + 8];

    // Online softmax update.
    float new_gmax0 = fmaxf(global_max[0], block_local_max0);
    float new_gmax1 = fmaxf(global_max[1], block_local_max1);
    const float alpha0 = (global_max[0] > -1e29f) ? exp2f(global_max[0] - new_gmax0) : 0.f;
    const float alpha1 = (global_max[1] > -1e29f) ? exp2f(global_max[1] - new_gmax1) : 0.f;
    const float block_rescale0 = exp2f(block_local_max0 - new_gmax0);
    const float block_rescale1 = exp2f(block_local_max1 - new_gmax1);
    // Per-warp rescale: drives spurious p ≡ 1 from all-invalid warps to ~0
    // before it contributes to sm_p_full (see decode-dsv4 comment).
    const float warp_rescale0 = exp2f(local_max[0] - new_gmax0);
    const float warp_rescale1 = exp2f(local_max[1] - new_gmax1);

    if (chunk_idx > chunk_lo) {
#pragma unroll
      for (int vc = 0; vc < N_V_CHUNKS; vc++) {
#pragma unroll
        for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
          acc_nope[vc][nt][0] *= alpha0;
          acc_nope[vc][nt][1] *= alpha0;
          acc_nope[vc][nt][2] *= alpha1;
          acc_nope[vc][nt][3] *= alpha1;
        }
      }
      global_sum[0] = global_sum[0] * alpha0 + block_local_sum0 * block_rescale0;
      global_sum[1] = global_sum[1] * alpha1 + block_local_sum1 * block_rescale1;
    } else {
      global_sum[0] = block_local_sum0 * block_rescale0;
      global_sum[1] = block_local_sum1 * block_rescale1;
    }
    global_max[0] = new_gmax0;
    global_max[1] = new_gmax1;

    // Stage 2.75: sm_p_full = p * warp_rescale.
    float w_pre[DSV3_2_QK_N_TILES][4];
#pragma unroll
    for (int nt = 0; nt < DSV3_2_QK_N_TILES; nt++) {
      w_pre[nt][0] = p[nt][0] * warp_rescale0;
      w_pre[nt][1] = p[nt][1] * warp_rescale0;
      w_pre[nt][2] = p[nt][2] * warp_rescale1;
      w_pre[nt][3] = p[nt][3] * warp_rescale1;
    }
    const int cand_col_base = warp_id * DSV3_2_ENTRIES_PER_WARP;
#pragma unroll
    for (int nt = 0; nt < DSV3_2_QK_N_TILES; nt++) {
      const int c0 = nt * 8 + tid * 2;
      const int c1 = c0 + 1;
      sm_p_full[gid][cand_col_base + c0] = __float2bfloat16(w_pre[nt][0]);
      sm_p_full[gid][cand_col_base + c1] = __float2bfloat16(w_pre[nt][1]);
      sm_p_full[gid + 8][cand_col_base + c0] = __float2bfloat16(w_pre[nt][2]);
      sm_p_full[gid + 8][cand_col_base + c1] = __float2bfloat16(w_pre[nt][3]);
    }
    // Zero-init sm_w_head_sc here (different smem buffer), single bar_sync
    // below covers both write groups.
    for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; i += DSV3_2_MATH_THREADS) {
      sm.w_head_sc()[i] = 0.f;
    }
    bar_sync_t<3, DSV3_2_MATH_THREADS>();

    // ── Stage 3 NoPE FP8 ──────────────────────────────────────
    // V32-family scales are inline FP32. DSv3.2 writes power-of-2 values; GLM
    // writes arbitrary values and uses the two-pass W residual below.
    {
      const int warp_first_cand_xv = warp_id * DSV3_2_ENTRIES_PER_WARP;
#pragma unroll
      for (int nt = 0; nt < DSV3_2_QK_N_TILES; nt++) {
        const int cand_e0 = warp_first_cand_xv + nt * 8 + tid * 2;
        const int cand_e1 = cand_e0 + 1;
#pragma unroll
        for (int vc = 0; vc < N_V_CHUNKS; vc++) {
          const float vsc0 = kv_scale_fp32(cand_e0, vc);
          const float vsc1 = kv_scale_fp32(cand_e1, vc);
          atomicMax(reinterpret_cast<int*>(&sm.w_head_sc()[vc * HPB + gid]),
                    __float_as_int(fmaxf(fabsf(w_pre[nt][0] * vsc0), fabsf(w_pre[nt][1] * vsc1))));
          atomicMax(reinterpret_cast<int*>(&sm.w_head_sc()[vc * HPB + gid + 8]),
                    __float_as_int(fmaxf(fabsf(w_pre[nt][2] * vsc0), fabsf(w_pre[nt][3] * vsc1))));
        }
      }
    }
    bar_sync_t<3, DSV3_2_MATH_THREADS>();
    for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; i += DSV3_2_MATH_THREADS) {
      sm.w_head_sc()[i] = fmaxf(sm.w_head_sc()[i], 1e-10f) / FP8_MAX;
    }
    bar_sync_t<3, DSV3_2_MATH_THREADS>();

#pragma unroll
    for (int vc = 0; vc < N_V_CHUNKS; vc++) {
      uint8_t* sm_w_fp8 = sm.w_fp8(vc & 1);
      const float sc0 = sm.w_head_sc()[vc * HPB + gid];
      const float sc1 = sm.w_head_sc()[vc * HPB + gid + 8];
      const float si0 = 1.f / sc0;
      const float si1 = 1.f / sc1;
      float xv_acc[NT_PER_WARP_XV][4] = {0};
#pragma unroll
      for (int wpass = 0; wpass < WeightFp8PassTraits<KV::SCALE_FORMAT>::PASSES; ++wpass) {
        if constexpr (KV::SCALE_FORMAT == ScaleFormat::ARBITRARY_FP32) {
          if (wpass > 0) bar_sync_t<3, DSV3_2_MATH_THREADS>();
        }
        const int warp_first_cand_xv = warp_id * DSV3_2_ENTRIES_PER_WARP;
#pragma unroll
        for (int nt = 0; nt < DSV3_2_QK_N_TILES; nt++) {
          const int cand_e0 = warp_first_cand_xv + nt * 8 + tid * 2;
          const int cand_e1 = cand_e0 + 1;
          const float vsc0 = kv_scale_fp32(cand_e0, vc);
          const float vsc1 = kv_scale_fp32(cand_e1, vc);
          const float wn00 = w_pre[nt][0] * vsc0 * si0;
          const float wn01 = w_pre[nt][1] * vsc1 * si0;
          const float wn10 = w_pre[nt][2] * vsc0 * si1;
          const float wn11 = w_pre[nt][3] * vsc1 * si1;
          Fp8WeightQuad wq =
              quantize_weight_quad_for_pass<KV::SCALE_FORMAT>(wn00, wn01, wn10, wn11, wpass);
          sm_w_fp8[(size_t)gid * W_FP8_STRIDE + cand_e0] = wq.h0_e0;
          sm_w_fp8[(size_t)gid * W_FP8_STRIDE + cand_e1] = wq.h0_e1;
          sm_w_fp8[(size_t)(gid + 8) * W_FP8_STRIDE + cand_e0] = wq.h1_e0;
          sm_w_fp8[(size_t)(gid + 8) * W_FP8_STRIDE + cand_e1] = wq.h1_e1;
        }
        bar_sync_t<3, DSV3_2_MATH_THREADS>();
#pragma unroll
        for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
          const int dim = vc * V_CHUNK + warp_id * (NT_PER_WARP_XV * 8) + nt * 8;
#pragma unroll
          for (int kstep = 0; kstep < XV_KSTEPS; kstep++) {
            const int ko = kstep * 32;
            uint32_t a0, a1, a2, a3, b0, b1;
            ldmatrix_load_A_fp8(a0, a1, a2, a3, sm_w_fp8 + ko, W_FP8_STRIDE, lane);
            d2_load_b_fp8<KV_SMEM_STRIDE>(b0, b1, sm_kv_fp8, kstep * 32, dim, lane);
            MmaFp8Result r = mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1, xv_acc[nt][0], xv_acc[nt][1],
                                              xv_acc[nt][2], xv_acc[nt][3]);
            xv_acc[nt][0] = r.d0;
            xv_acc[nt][1] = r.d1;
            xv_acc[nt][2] = r.d2;
            xv_acc[nt][3] = r.d3;
          }
        }
      }
#pragma unroll
      for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
        acc_nope[vc][nt][0] += xv_acc[nt][0] * sc0;
        acc_nope[vc][nt][1] += xv_acc[nt][1] * sc0;
        acc_nope[vc][nt][2] += xv_acc[nt][2] * sc1;
        acc_nope[vc][nt][3] += xv_acc[nt][3] * sc1;
      }
    }

    // V32 V_HAS_ROPE=false: no RoPE-side XV stage.

    // sm_p_full + sm_w_fp8 reuse next iter — math-only sync ensures Stage 3
    // is fully drained before consumer_release lets IO overwrite the slot.
    bar_sync_t<3, DSV3_2_MATH_THREADS>();

    // Release the slot to IO.
    if (threadIdx.x == 0) {
      mbarrier_arrive(sm.mbar_empty(cons_idx));
    }
    ++cons_idx;
    if (cons_idx == DSV3_2_KV_BUF_COUNT) {
      cons_idx = 0;
      cons_phase ^= 1;
    }
  }  // chunk loop

  // ── Write per-split partial output + LSE to mid_out / mid_lse ───
  // V32 D_V = D_NOPE = 512 (no RoPE segment in V).
  const float inv_g0 = (global_sum[0] > 0.f) ? (1.f / global_sum[0]) : 0.f;
  const float inv_g1 = (global_sum[1] > 0.f) ? (1.f / global_sum[1]) : 0.f;

  const size_t mid_o_base = ((size_t)t_idx * NUM_HEADS + h_start) * (size_t)num_splits * D_V_C +
                            (size_t)split_idx * D_V_C;

  // Pack adjacent (d0, d0+1) bf16 pairs into __nv_bfloat162 → STG.E.64.
#pragma unroll
  for (int vc = 0; vc < N_V_CHUNKS; vc++) {
#pragma unroll
    for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
      const int d0 = vc * V_CHUNK + warp_id * (NT_PER_WARP_XV * 8) + nt * 8 + tid * 2;
      const __nv_bfloat162 pair_lo =
          __floats2bfloat162_rn(acc_nope[vc][nt][0] * inv_g0, acc_nope[vc][nt][1] * inv_g0);
      const __nv_bfloat162 pair_hi =
          __floats2bfloat162_rn(acc_nope[vc][nt][2] * inv_g1, acc_nope[vc][nt][3] * inv_g1);
      *reinterpret_cast<__nv_bfloat162*>(
          &mid_out[mid_o_base + (size_t)gid * num_splits * D_V_C + d0]) = pair_lo;
      if constexpr (VALID_HPB > 8) {
        *reinterpret_cast<__nv_bfloat162*>(
            &mid_out[mid_o_base + (size_t)(gid + 8) * num_splits * D_V_C + d0]) = pair_hi;
      }
    }
  }
  if (warp_id == 0 && tid == 0) {
    const float lse0 = (global_sum[0] > 0.f) ? (log2f(global_sum[0]) + global_max[0]) : -1e30f;
    const float lse1 = (global_sum[1] > 0.f) ? (log2f(global_sum[1]) + global_max[1]) : -1e30f;
    const size_t lse_base = (size_t)t_idx * NUM_HEADS * num_splits + (size_t)h_start * num_splits;
    mid_lse[lse_base + (size_t)gid * num_splits + split_idx] = lse0;
    if constexpr (VALID_HPB > 8) {
      mid_lse[lse_base + (size_t)(gid + 8) * num_splits + split_idx] = lse1;
    }
  }
}

}  // namespace flashinfer::sparse_mla_sm120
