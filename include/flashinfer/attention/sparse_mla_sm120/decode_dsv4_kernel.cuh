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
#include "model/kv_cache_traits.cuh"
#include "model/scale_convert.cuh"

namespace flashinfer::sparse_mla_sm120 {

// Sparse MLA decode (DSv4): warp-specialized TMA gather, 8 math warps,
// double-buffered KV. IO warp drives cp.async.bulk into the two KV
// buffers; math warps consume. Each warp covers DSV4_BI/8 candidates and
// V_CHUNK/8 V-dims so per-thread acc_nope stays in registers.
//
// Per-buf mbarrier pairs: mbar_full[s] for IO→math (leader arrives with
// expect_tx, bulk completion decrements tx), mbar_empty[s] for math→IO
// drain (one math signaling thread arrives at end-of-iter).
//
// IMPORTANT: mbarrier.try_wait.parity has no implicit memory fence — the
// consumer must follow it with a CTA-wide acq-rel sync before reading
// smem, otherwise non-leader IO writes can be stale. We use
// bar_sync<3, MATH_THREADS> since only math warps participate.

constexpr int DSV4_N_WARPS = 8;  // math warps
constexpr int DSV4_IO_WARPS = 1;
constexpr int DSV4_N_TOTAL_WARPS = DSV4_N_WARPS + DSV4_IO_WARPS;  // 9
constexpr int DSV4_BLOCK_THREADS = DSV4_N_TOTAL_WARPS * 32;       // 288
constexpr int DSV4_MATH_THREADS = DSV4_N_WARPS * 32;              // 256
constexpr int DSV4_IO_THREADS = DSV4_IO_WARPS * 32;               // 32
constexpr int DSV4_CAND_WINDOW = 64;
constexpr int DSV4_BI = DSV4_CAND_WINDOW;
constexpr int DSV4_KV_BUF_COUNT = 2;
constexpr int DSV4_ENTRIES_PER_WARP = DSV4_BI / DSV4_N_WARPS;  // 8
constexpr int DSV4_QK_N_TILES = DSV4_ENTRIES_PER_WARP / 8;     // 1

template <ModelType MT>
struct DecodeDsv4Smem {
  using KV = KVCacheTraits<MT>;
  static_assert(MT == ModelType::DSV4);

  static constexpr int N_V_CHUNKS = KV::D_NOPE / KV::QUANT_TILE;
  static constexpr size_t SMEM_Q_ROPE = HPB * KV::D_ROPE * sizeof(bf16);
  static constexpr size_t SMEM_Q_FP8 = HPB * KV::Q_NOPE_STRIDE;
  static constexpr size_t SMEM_Q_SC = HPB * KV::NUM_SCALES * sizeof(float);
  static constexpr size_t SMEM_KV_FP8_BUF = DSV4_BI * KV::KV_SMEM_STRIDE;
  static constexpr size_t SMEM_KV_SC_BUF = DSV4_BI * KV::SCALE_BYTES_PER_TOKEN;
  static constexpr size_t SMEM_KV_ROPE_BUF = DSV4_BI * KV::D_ROPE * sizeof(bf16);
  static constexpr size_t SMEM_MBAR_PAIR = 2 * sizeof(uint64_t);
  static constexpr size_t SMEM_REDUCE = 2 * DSV4_N_WARPS * HPB * sizeof(float);
  static constexpr size_t SMEM_W_HEAD_SC = N_V_CHUNKS * HPB * sizeof(float);
  static constexpr size_t SMEM_W_FP8_BUF = HPB * (DSV4_BI + 16);

  static constexpr size_t OFF_Q_ROPE = 0;
  static constexpr size_t OFF_Q_FP8 = OFF_Q_ROPE + SMEM_Q_ROPE;
  static constexpr size_t OFF_Q_SC = OFF_Q_FP8 + SMEM_Q_FP8;
  static constexpr size_t OFF_KV_FP8 = OFF_Q_SC + SMEM_Q_SC;
  static constexpr size_t OFF_KV_SC = OFF_KV_FP8 + DSV4_KV_BUF_COUNT * SMEM_KV_FP8_BUF;
  static constexpr size_t OFF_KV_ROPE = OFF_KV_SC + DSV4_KV_BUF_COUNT * SMEM_KV_SC_BUF;
  static constexpr size_t OFF_MBAR_FULL_UNALIGNED =
      OFF_KV_ROPE + DSV4_KV_BUF_COUNT * SMEM_KV_ROPE_BUF;
  static constexpr size_t OFF_MBAR_FULL = (OFF_MBAR_FULL_UNALIGNED + 15) / 16 * 16;
  static constexpr size_t OFF_MBAR_EMPTY = OFF_MBAR_FULL + SMEM_MBAR_PAIR;
  static constexpr size_t OFF_REDUCE = OFF_MBAR_EMPTY + SMEM_MBAR_PAIR;
  static constexpr size_t OFF_W_HEAD_SC = OFF_REDUCE + SMEM_REDUCE;
  static constexpr size_t OFF_W_FP8 = OFF_W_HEAD_SC + SMEM_W_HEAD_SC;

  char* base;

  __device__ static DecodeDsv4Smem init(char* base) { return DecodeDsv4Smem{base}; }
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
  __device__ __forceinline__ uint8_t* kv_sc(int i) const {
    return reinterpret_cast<uint8_t*>(base + OFF_KV_SC + i * SMEM_KV_SC_BUF);
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
  __device__ __forceinline__ float* warp_sum() const { return reduce() + DSV4_N_WARPS * HPB; }
  __device__ __forceinline__ float* w_head_sc() const {
    return reinterpret_cast<float*>(base + OFF_W_HEAD_SC);
  }
  __device__ __forceinline__ uint8_t* w_fp8(int parity) const {
    return reinterpret_cast<uint8_t*>(base + OFF_W_FP8 + parity * SMEM_W_FP8_BUF);
  }
};

// No minBlocksPerSM on launch_bounds: kernel is smem-bound at 1 block/SM, and
// the unconstrained register budget avoids the per-warp spill the hint forces.
template <ModelType MT, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE>
__global__ void __launch_bounds__(DSV4_BLOCK_THREADS) sparse_mla_decode_dsv4_kernel(
    const bf16* __restrict__ Q,               // [num_tokens, num_heads, d_qk] bf16
    const uint8_t* __restrict__ KV_cache,     // FP8 paged (DSV4 footer layout)
    const int32_t* __restrict__ indices,      // [num_tokens, topk] int32
    bf16* __restrict__ mid_out,               // [num_tokens, num_heads, num_splits, d_v] bf16
    float* __restrict__ mid_lse,              // [num_tokens, num_heads, num_splits] f32
    const int* __restrict__ topk_length_ptr,  // [num_tokens] or null
    // Optional secondary KV cache (DSv4 C4A / C128A). When non-null, the extra
    // candidate window is concatenated after the main one; per-chunk dispatch
    // in IO + math routes each chunk to the correct source.
    const uint8_t* __restrict__ extra_KV_cache,     // nullable; may use different pbs
    const int32_t* __restrict__ extra_indices,      // [num_tokens, extra_topk]
    const int* __restrict__ extra_topk_length_ptr,  // [num_tokens] or null
    int extra_topk,                                 // 0 = no extra cache
    int pbs_extra,  // page_block_size for extra cache (e.g. 2 for DSv4 C128A)
    size_t stride_extra_kv_block, int num_tokens, int num_splits, int chunks_per_block,
    float sm_scale, size_t stride_kv_block) {
  using KV = KVCacheTraits<MT>;
  static_assert(MT == ModelType::DSV4, "decode-dsv4 currently DSV4-only");
  constexpr int D_NOPE = KV::D_NOPE;                                // 448
  constexpr int D_ROPE_C = KV::D_ROPE;                              // 64
  constexpr int D_QK = KV::D_QK;                                    // 512
  constexpr int D_V_C = KV::D_V;                                    // 512
  constexpr int QUANT_TILE = KV::QUANT_TILE;                        // 64
  constexpr int NUM_SCALES = KV::NUM_SCALES;                        // 7
  constexpr int Q_NOPE_STRIDE = KV::Q_NOPE_STRIDE;                  // 464
  constexpr int KV_SMEM_STRIDE = KV::KV_SMEM_STRIDE;                // 464
  constexpr int SCALE_BYTES_PER_TOKEN = KV::SCALE_BYTES_PER_TOKEN;  // 8
  constexpr int IO_STRIDE = D_NOPE + D_ROPE_C * 2;                  // 576
  constexpr int pbs = PAGE_BLOCK_SIZE;
  // Kernel always computes a full HPB×CAND tile (zero-Q-padded for unused
  // head slots); NUM_HEADS=8 small-TP shards write back only VALID_HPB rows.
  constexpr int VALID_HPB = (NUM_HEADS < HPB) ? NUM_HEADS : HPB;

  const int t_idx = blockIdx.x;
  const int h_block_idx = blockIdx.y;
  const int split_idx = blockIdx.z;
  if (t_idx >= num_tokens) return;

  const int h_start = h_block_idx * HPB;
  int topk_len = topk_length_ptr ? __ldg(topk_length_ptr + t_idx) : TOPK;
  topk_len = topk_len < 0 ? 0 : (topk_len > TOPK ? TOPK : topk_len);
  int extra_topk_len =
      (extra_KV_cache != nullptr)
          ? (extra_topk_length_ptr ? __ldg(extra_topk_length_ptr + t_idx) : extra_topk)
          : 0;
  extra_topk_len =
      extra_topk_len < 0 ? 0 : (extra_topk_len > extra_topk ? extra_topk : extra_topk_len);

  // Chunk range this block owns. Total chunks = main + extra (extra layout
  // is concatenated immediately after main; per-chunk dispatch in IO + math
  // routes to the right source).
  const int num_orig_chunks = (topk_len + DSV4_CAND_WINDOW - 1) / DSV4_CAND_WINDOW;
  const int num_extra_chunks = (extra_topk_len + DSV4_CAND_WINDOW - 1) / DSV4_CAND_WINDOW;
  const int num_chunks_total = num_orig_chunks + num_extra_chunks;
  const int chunk_lo = split_idx * chunks_per_block;
  const int chunk_hi = min(chunk_lo + chunks_per_block, num_chunks_total);

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const bool is_io = (warp_id >= DSV4_N_WARPS);

  // Early-exit splits: only math threads write LSE; IO warp has nothing to do.
  if (chunk_lo >= num_chunks_total) {
    if (!is_io && threadIdx.x < VALID_HPB) {
      const int h = h_start + threadIdx.x;
      const size_t lse_off =
          (size_t)t_idx * NUM_HEADS * num_splits + (size_t)h * num_splits + split_idx;
      mid_lse[lse_off] = -1e30f;
    }
    return;
  }

  constexpr int V_CHUNK = QUANT_TILE;                          // 64
  constexpr int N_V_CHUNKS = D_NOPE / V_CHUNK;                 // 7
  constexpr int NT_PER_WARP_XV = V_CHUNK / 8 / DSV4_N_WARPS;   // 1
  constexpr int XV_KSTEPS = DSV4_BI / 32;                      // 2
  constexpr int W_FP8_STRIDE = DSV4_BI + 16;                   // 80
  constexpr int ROPE_DIMS_PER_WARP = D_ROPE_C / DSV4_N_WARPS;  // 8
  constexpr int ROPE_N_TILES = ROPE_DIMS_PER_WARP / 8;         // 1
  constexpr int ROPE_K_ITERS = DSV4_BI / 16;                   // 4

  // ── Dynamic smem layout ────────────────────────────────────────
  // Single-buffer Q/scratch + double-buffered KV bufs.
  //   sm_q_rope    HPB * D_ROPE * 2B            =  2 KB
  //   sm_q_fp8     HPB * Q_NOPE_STRIDE          = 7.25 KB
  //   sm_q_sc      HPB * NUM_SCALES * 4B        = 0.44 KB
  //   sm_kv_fp8    2 * DSV4_BI * KV_SMEM_STRIDE   = 58 KB
  //   sm_kv_sc     2 * DSV4_BI * 8                = 1 KB
  //   sm_kv_rope   2 * DSV4_BI * D_ROPE * 2B      = 16 KB
  //   sm_reduce    2 * DSV4_N_WARPS * HPB * 4     = 1 KB  (8 warps)
  //   sm_w_head_sc N_V_CHUNKS * HPB * 4         = 448 B
  //   sm_w_fp8 ×2  2 * HPB * (DSV4_BI + 16)       = 2.5 KB  (double-buf
  //                across vc iters to drop the if-vc>0 bar_sync)
  //   Total                                     ~ 88 KB
  // Plus static sm_p_full HPB * DSV4_BI * 2B (bf16) = 2 KB.
  // Grand total ~ 90 KB (under 100 KB SM120 carveout, 1 block/SM).
  extern __shared__ __align__(16) char smem_raw[];
  auto sm = DecodeDsv4Smem<MT>::init(smem_raw);

  __shared__ bf16 sm_p_full[HPB][DSV4_BI];  // 2 KB static
  const int32_t* idx_base = indices + (size_t)t_idx * TOPK;

  // mbar_full: leader arrives + expect_tx, bulk completion drives tx.
  // mbar_empty: math signaling thread arrives. mbar.try_wait.parity has no
  // implicit memory fence — consumer must acq-rel via bar_sync after wait.
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < DSV4_KV_BUF_COUNT; ++s) {
      mbarrier_init(sm.mbar_full(s), 1);
      mbarrier_init(sm.mbar_empty(s), 1);
    }
  }
  __syncthreads();

  // ── TMA bulk constants ──
  constexpr uint32_t DSV4_BULK_NOPE_BYTES = (uint32_t)D_NOPE;                   // 448
  constexpr uint32_t DSV4_BULK_ROPE_BYTES = (uint32_t)D_ROPE_C * sizeof(bf16);  // 128
  constexpr uint32_t DSV4_BULK_TX_BYTES =
      (uint32_t)DSV4_BI * (DSV4_BULK_NOPE_BYTES + DSV4_BULK_ROPE_BYTES);

  // IO gather: scalar scales → fence → expect_tx + bulks.
  // Dispatch per-chunk: chunks [0, num_orig_chunks) read from main KV_cache +
  // indices; chunks [num_orig_chunks, num_chunks_total) read from the
  // extra_KV_cache + extra_indices. The smem layout is shared — math warps
  // don't care which source the data came from.
  auto issue_gather = [&](int gather_chunk_idx, int buf) {
    const bool is_extra = (gather_chunk_idx >= num_orig_chunks);
    const int chunk_in_section = is_extra ? (gather_chunk_idx - num_orig_chunks) : gather_chunk_idx;
    const int section_len = is_extra ? extra_topk_len : topk_len;
    const int g_start = chunk_in_section * DSV4_CAND_WINDOW;
    const int g_end = min(g_start + DSV4_CAND_WINDOW, section_len);
    const int32_t* section_idx_base =
        is_extra ? (extra_indices + (size_t)t_idx * extra_topk) : idx_base;
    const uint8_t* section_kv = is_extra ? extra_KV_cache : KV_cache;
    const size_t section_stride = is_extra ? stride_extra_kv_block : stride_kv_block;
    // Page block size of THIS section. Main is compile-time constexpr (typ.
    // 64); extra is runtime (DSv4 C128A passes 2). The 8-cycle runtime div
    // is dwarfed by the cp.async.bulk that follows.
    const int section_pbs = is_extra ? pbs_extra : pbs;
    uint8_t* kv_fp8_dst = sm.kv_fp8(buf);
    bf16* kv_rope_dst = sm.kv_rope(buf);
    uint8_t* kv_sc_dst = sm.kv_sc(buf);

#pragma unroll
    for (int eo = 0; eo < DSV4_BI; eo += DSV4_IO_THREADS) {
      const int entry_idx = eo + lane;
      if (entry_idx >= DSV4_BI) break;
      const int cand_pos = g_start + entry_idx;
      const bool is_valid_cand = (cand_pos < g_end);
      const int idx_raw = is_valid_cand ? section_idx_base[cand_pos] : -1;
      uint64_t scale_word = 0;
      if (idx_raw >= 0) {
        const int idx = idx_raw;
        const int block_idx_g = idx / section_pbs;
        const int local_idx_g = idx - block_idx_g * section_pbs;
        const uint8_t* scale_base = section_kv + (size_t)block_idx_g * section_stride +
                                    (size_t)section_pbs * IO_STRIDE +
                                    (size_t)local_idx_g * SCALE_BYTES_PER_TOKEN;
        scale_word = __ldg(reinterpret_cast<const uint64_t*>(scale_base));
      }
      *reinterpret_cast<uint64_t*>(kv_sc_dst + (size_t)entry_idx * SCALE_BYTES_PER_TOKEN) =
          scale_word;
    }
    __threadfence_block();

    if (lane == 0) {
      mbarrier_arrive_expect_tx(sm.mbar_full(buf), DSV4_BULK_TX_BYTES);
    }

    // Issue cp.async.bulk for NoPE (448 B/entry) + RoPE (128 B/entry).
    // Bulk completion decrements mbar tx; phase flips when arrival count
    // (1, by leader above) AND tx=0 both met.
#pragma unroll
    for (int eo = 0; eo < DSV4_BI; eo += DSV4_IO_THREADS) {
      const int entry_idx = eo + lane;
      if (entry_idx >= DSV4_BI) break;
      const int cand_pos = g_start + entry_idx;
      const int idx_raw = (cand_pos < g_end) ? section_idx_base[cand_pos] : -1;
      const int idx = (idx_raw >= 0) ? idx_raw : 0;
      const int block_idx_g = idx / section_pbs;
      const int local_idx_g = idx - block_idx_g * section_pbs;
      const uint8_t* data_base =
          section_kv + (size_t)block_idx_g * section_stride + (size_t)local_idx_g * IO_STRIDE;
      cp_async_bulk_g2s(kv_fp8_dst + (size_t)entry_idx * KV_SMEM_STRIDE, data_base,
                        DSV4_BULK_NOPE_BYTES, sm.mbar_full(buf));
      cp_async_bulk_g2s(kv_rope_dst + (size_t)entry_idx * D_ROPE_C, data_base + D_NOPE,
                        DSV4_BULK_ROPE_BYTES, sm.mbar_full(buf));
    }
  };

  if (is_io) {
    // Producer state: index, phase. Phase starts at 1 so first empty.wait(1)
    // on a freshly-initialized barrier (phase 0) returns immediately.
    uint32_t prod_phase = 1;
    int prod_idx = 0;
    for (int chunk_idx = chunk_lo; chunk_idx < chunk_hi; ++chunk_idx) {
      const int buf = (chunk_idx - chunk_lo) & 1;
      mbarrier_wait_parity(sm.mbar_empty(prod_idx), prod_phase);
      issue_gather(chunk_idx, buf);
      ++prod_idx;
      if (prod_idx == DSV4_KV_BUF_COUNT) {
        prod_idx = 0;
        prod_phase ^= 1;
      }
    }
    return;
  }

  // ──────────────────────────────────────────────────────────────
  // Math warps branch (warp_id < DSV4_N_WARPS = 4)
  // ──────────────────────────────────────────────────────────────

  const int gid = lane >> 2;
  const int tid = lane & 3;

  // Stage 0: Q quantization (math threads only; the helper uses bar:2
  // internally with count=DSV4_MATH_THREADS).
  const bf16* q_base = Q + (size_t)t_idx * NUM_HEADS * D_QK + (size_t)h_start * D_QK;
  quantize_q_to_smem<MT, DSV4_MATH_THREADS>(sm.q_fp8(), sm.q_sc(), sm.q_rope(), q_base, sm.reduce(),
                                            VALID_HPB);

  // Persistent state across chunks (per-thread registers).
  float acc_nope[N_V_CHUNKS][NT_PER_WARP_XV][4] = {0};
  float acc_rope[ROPE_N_TILES][4] = {0};
  float global_max[2] = {-1e30f, -1e30f};
  float global_sum[2] = {0.f, 0.f};

  // ── Chunk loop ─────────────────────────────────────────────────
  // Consumer state: starts at (idx=0, phase=0).
  uint32_t cons_phase = 0;
  int cons_idx = 0;

  for (int chunk_idx = chunk_lo; chunk_idx < chunk_hi; ++chunk_idx) {
    const int buf = (chunk_idx - chunk_lo) & 1;
    // Dispatch chunk to main vs extra section. The split_cand_{start,end}
    // pair is used by the math-warp mask: any candidate slot whose absolute
    // offset within its section ≥ section_len gets qk = -inf.
    const bool is_extra_chunk = (chunk_idx >= num_orig_chunks);
    const int chunk_in_section = is_extra_chunk ? (chunk_idx - num_orig_chunks) : chunk_idx;
    const int section_len = is_extra_chunk ? extra_topk_len : topk_len;
    const int split_cand_start = chunk_in_section * DSV4_CAND_WINDOW;
    const int split_cand_end = min(split_cand_start + DSV4_CAND_WINDOW, section_len);

    // Wait for IO to fill this buf (mbar_full tx + arrival both met).
    mbarrier_wait_parity(sm.mbar_full(cons_idx), cons_phase);
    // CTA-wide acquire after mbar wake. Without this, math reads see only
    // the view released by whichever IO thread triggered the phase flip —
    // other IO lanes' writes (e.g., last-head RoPE bytes) may be stale.
    bar_sync_t<3, DSV4_MATH_THREADS>();

    uint8_t* sm_kv_fp8 = sm.kv_fp8(buf);
    uint8_t* sm_kv_sc = sm.kv_sc(buf);
    bf16* sm_kv_rope = sm.kv_rope(buf);

    // ── Stage 2 QK ────────────────────────────────────────────
    float qk[DSV4_QK_N_TILES][4] = {0};
    {
      const int warp_first_cand = warp_id * DSV4_ENTRIES_PER_WARP;
#pragma unroll
      for (int blk = 0; blk < NUM_SCALES; blk++) {
        uint8_t sfa = fp32_to_ue8m0(sm.q_sc()[(gid + (lane & 1) * 8) * NUM_SCALES + blk]);
#pragma unroll
        for (int ks = 0; ks < QUANT_TILE / 32; ks++) {
          const int ko = blk * QUANT_TILE + ks * 32;
          uint32_t a0, a1, a2, a3;
          ldmatrix_load_A_fp8(a0, a1, a2, a3, sm.q_fp8() + ko, Q_NOPE_STRIDE, lane);
#pragma unroll
          for (int nt = 0; nt < DSV4_QK_N_TILES; nt++) {
            const int cand_row_base = warp_first_cand + nt * 8;
            uint8_t sfb = sm_kv_sc[(cand_row_base + gid) * SCALE_BYTES_PER_TOKEN + blk];
            uint32_t b0, b1;
            ldmatrix_load_B_fp8(b0, b1, sm_kv_fp8 + (size_t)cand_row_base * KV_SMEM_STRIDE + ko,
                                KV_SMEM_STRIDE, lane);
            MmaFp8Result r = mma_fp8_block_scaled_m16n8k32(
                a0, a1, a2, a3, b0, b1, qk[nt][0], qk[nt][1], qk[nt][2], qk[nt][3], sfa, sfb);
            qk[nt][0] = r.d0;
            qk[nt][1] = r.d1;
            qk[nt][2] = r.d2;
            qk[nt][3] = r.d3;
          }
        }
      }
    }
    {
      // K-rope B operand: scalar per-lane reads. mma.m16n8k16 needs each lane's
      // b0/b1 to hold consecutive K-rows of one N-entry, which ldmatrix.x2.trans
      // can't produce from the N-outer smem here (gid = lane>>2, tid = lane&3).
      const int warp_first_cand = warp_id * DSV4_ENTRIES_PER_WARP;
#pragma unroll
      for (int ks = 0; ks < D_ROPE_C / 16; ks++) {
        uint32_t a0, a1, a2, a3;
        ldmatrix_load_A_bf16(a0, a1, a2, a3, sm.q_rope() + ks * 16, D_ROPE_C, lane);
#pragma unroll
        for (int nt = 0; nt < DSV4_QK_N_TILES; nt++) {
          const int cand_row_base = warp_first_cand + nt * 8;
          // Per-lane scalar load: each lane reads from its OWN N-col entry.
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

    // Mask invalid cands + sm_scale × LOG2E. Invalid = position past
    // section_len OR slot id = -1 (indexer-padded; IO already gathered slot 0
    // into smem with idx clamped — masking to -inf kills it in softmax).
    const int32_t* section_idx_base =
        is_extra_chunk ? (extra_indices + (size_t)t_idx * extra_topk) : idx_base;
    const int warp_first_cand = warp_id * DSV4_ENTRIES_PER_WARP;
#pragma unroll
    for (int nt = 0; nt < DSV4_QK_N_TILES; nt++) {
      const int c0 = warp_first_cand + nt * 8 + tid * 2;
      const int c1 = c0 + 1;
      const int abs_c0 = c0 + split_cand_start;
      const int abs_c1 = c1 + split_cand_start;
      const int idx0 = (abs_c0 < section_len) ? section_idx_base[abs_c0] : -1;
      const int idx1 = (abs_c1 < section_len) ? section_idx_base[abs_c1] : -1;
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
    for (int nt = 0; nt < DSV4_QK_N_TILES; nt++) {
      local_max[0] = fmaxf(local_max[0], fmaxf(qk[nt][0], qk[nt][1]));
      local_max[1] = fmaxf(local_max[1], fmaxf(qk[nt][2], qk[nt][3]));
    }
#pragma unroll
    for (int s = 2; s >= 1; s >>= 1) {
      local_max[0] = fmaxf(local_max[0], __shfl_xor_sync(0xffffffff, local_max[0], s));
      local_max[1] = fmaxf(local_max[1], __shfl_xor_sync(0xffffffff, local_max[1], s));
    }
    float local_sum[2] = {0.f, 0.f};
    float p[DSV4_QK_N_TILES][4];
#pragma unroll
    for (int nt = 0; nt < DSV4_QK_N_TILES; nt++) {
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

    // Cross-warp reduce.
    if (tid == 0) {
      sm.warp_max()[warp_id * HPB + gid] = local_max[0];
      sm.warp_max()[warp_id * HPB + gid + 8] = local_max[1];
      sm.warp_sum()[warp_id * HPB + gid] = local_sum[0];
      sm.warp_sum()[warp_id * HPB + gid + 8] = local_sum[1];
    }
    bar_sync_t<3, DSV4_MATH_THREADS>();
    if (threadIdx.x < VALID_HPB) {
      const int h = threadIdx.x;
      float wmax[DSV4_N_WARPS], wsum[DSV4_N_WARPS];
#pragma unroll
      for (int w = 0; w < DSV4_N_WARPS; w++) {
        wmax[w] = sm.warp_max()[w * HPB + h];
        wsum[w] = sm.warp_sum()[w * HPB + h];
      }
      float bmax = -1e30f;
#pragma unroll
      for (int w = 0; w < DSV4_N_WARPS; w++) bmax = fmaxf(bmax, wmax[w]);
      float bsum = 0.f;
#pragma unroll
      for (int w = 0; w < DSV4_N_WARPS; w++) bsum += wsum[w] * exp2f(wmax[w] - bmax);
      sm.warp_max()[h] = bmax;
      sm.warp_sum()[h] = bsum;
    }
    bar_sync_t<3, DSV4_MATH_THREADS>();

    const float block_local_max0 = sm.warp_max()[gid];
    const float block_local_max1 = sm.warp_max()[gid + 8];
    const float block_local_sum0 = sm.warp_sum()[gid];
    const float block_local_sum1 = sm.warp_sum()[gid + 8];

    // Online softmax update.
    float new_gmax0 = fmaxf(global_max[0], block_local_max0);
    float new_gmax1 = fmaxf(global_max[1], block_local_max1);
    const float alpha0 = (global_max[0] > -1e29f) ? exp2f(global_max[0] - new_gmax0) : 0.f;
    const float alpha1 = (global_max[1] > -1e29f) ? exp2f(global_max[1] - new_gmax1) : 0.f;
    // Two distinct rescale factors:
    //   block_rescale = exp(block_local_max - new_gmax) — rescales the
    //     block-wide sum (already weighted by per-warp local_max during
    //     cross-warp reduction) into the new global frame. Used for the
    //     global_sum update.
    //   warp_rescale  = exp(local_max[w] - new_gmax) — rescales THIS
    //     warp's p (computed in the warp's own local_max frame) into the
    //     new global frame. CRITICAL for correctness when a warp covers
    //     only invalid candidates: the post-mask qk == -1e30 * sm_scale *
    //     LOG2E ≈ -6.38e28 becomes the warp's local_max[0], and softmax
    //     gives p ≡ 1 (exp2(qk - local_max) = exp2(0)). Without the
    //     per-warp factor, these spurious 1s would leak into sm_p_full
    //     and corrupt the RoPE MMA. The exp(local_max - new_gmax) factor
    //     drives them to ~0.
    const float block_rescale0 = exp2f(block_local_max0 - new_gmax0);
    const float block_rescale1 = exp2f(block_local_max1 - new_gmax1);
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
#pragma unroll
      for (int nt = 0; nt < ROPE_N_TILES; nt++) {
        acc_rope[nt][0] *= alpha0;
        acc_rope[nt][1] *= alpha0;
        acc_rope[nt][2] *= alpha1;
        acc_rope[nt][3] *= alpha1;
      }
      global_sum[0] = global_sum[0] * alpha0 + block_local_sum0 * block_rescale0;
      global_sum[1] = global_sum[1] * alpha1 + block_local_sum1 * block_rescale1;
    } else {
      global_sum[0] = block_local_sum0 * block_rescale0;
      global_sum[1] = block_local_sum1 * block_rescale1;
    }
    global_max[0] = new_gmax0;
    global_max[1] = new_gmax1;

    // Stage 2.75: sm_p_full = p * warp_rescale. Each warp uses its OWN
    // local_max-based rescale to kill all-invalid-warp contributions.
    float w_pre[DSV4_QK_N_TILES][4];
#pragma unroll
    for (int nt = 0; nt < DSV4_QK_N_TILES; nt++) {
      w_pre[nt][0] = p[nt][0] * warp_rescale0;
      w_pre[nt][1] = p[nt][1] * warp_rescale0;
      w_pre[nt][2] = p[nt][2] * warp_rescale1;
      w_pre[nt][3] = p[nt][3] * warp_rescale1;
    }
    const int cand_col_base = warp_id * DSV4_ENTRIES_PER_WARP;
#pragma unroll
    for (int nt = 0; nt < DSV4_QK_N_TILES; nt++) {
      const int c0 = nt * 8 + tid * 2;
      const int c1 = c0 + 1;
      sm_p_full[gid][cand_col_base + c0] = __float2bfloat16(w_pre[nt][0]);
      sm_p_full[gid][cand_col_base + c1] = __float2bfloat16(w_pre[nt][1]);
      sm_p_full[gid + 8][cand_col_base + c0] = __float2bfloat16(w_pre[nt][2]);
      sm_p_full[gid + 8][cand_col_base + c1] = __float2bfloat16(w_pre[nt][3]);
    }
    // Zero-init sm_w_head_sc here (different smem buffer than sm_p_full above),
    // so the single bar_sync below covers both write groups.
    for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; i += DSV4_MATH_THREADS) {
      sm.w_head_sc()[i] = 0.f;
    }
    bar_sync_t<3, DSV4_MATH_THREADS>();

    // ── Stage 3 NoPE FP8 ──────────────────────────────────────
    {
      const int warp_first_cand_xv = warp_id * DSV4_ENTRIES_PER_WARP;
#pragma unroll
      for (int nt = 0; nt < DSV4_QK_N_TILES; nt++) {
        const int cand_e0 = warp_first_cand_xv + nt * 8 + tid * 2;
        const int cand_e1 = cand_e0 + 1;
#pragma unroll
        for (int vc = 0; vc < N_V_CHUNKS; vc++) {
          const float vsc0 = ue8m0_to_fp32(sm_kv_sc[(size_t)cand_e0 * SCALE_BYTES_PER_TOKEN + vc]);
          const float vsc1 = ue8m0_to_fp32(sm_kv_sc[(size_t)cand_e1 * SCALE_BYTES_PER_TOKEN + vc]);
          atomicMax(reinterpret_cast<int*>(&sm.w_head_sc()[vc * HPB + gid]),
                    __float_as_int(fmaxf(fabsf(w_pre[nt][0] * vsc0), fabsf(w_pre[nt][1] * vsc1))));
          atomicMax(reinterpret_cast<int*>(&sm.w_head_sc()[vc * HPB + gid + 8]),
                    __float_as_int(fmaxf(fabsf(w_pre[nt][2] * vsc0), fabsf(w_pre[nt][3] * vsc1))));
        }
      }
    }
    bar_sync_t<3, DSV4_MATH_THREADS>();
    for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; i += DSV4_MATH_THREADS) {
      sm.w_head_sc()[i] = fmaxf(sm.w_head_sc()[i], 1e-10f) / FP8_MAX;
    }
    bar_sync_t<3, DSV4_MATH_THREADS>();

#pragma unroll
    for (int vc = 0; vc < N_V_CHUNKS; vc++) {
      // Double-buffered: vc=N quants into buf[N&1]; vc=N+1's quant targets the
      // OTHER buffer, so it cannot race with vc=N's MMA read. The bar_sync
      // after quant is the only sync needed within the vc loop.
      uint8_t* sm_w_fp8 = sm.w_fp8(vc & 1);
      // Phase 3 quant.
      {
        const int warp_first_cand_xv = warp_id * DSV4_ENTRIES_PER_WARP;
        const float si0 = 1.f / sm.w_head_sc()[vc * HPB + gid];
        const float si1 = 1.f / sm.w_head_sc()[vc * HPB + gid + 8];
#pragma unroll
        for (int nt = 0; nt < DSV4_QK_N_TILES; nt++) {
          const int cand_e0 = warp_first_cand_xv + nt * 8 + tid * 2;
          const int cand_e1 = cand_e0 + 1;
          const float vsc0 = ue8m0_to_fp32(sm_kv_sc[(size_t)cand_e0 * SCALE_BYTES_PER_TOKEN + vc]);
          const float vsc1 = ue8m0_to_fp32(sm_kv_sc[(size_t)cand_e1 * SCALE_BYTES_PER_TOKEN + vc]);
          __nv_fp8_e4m3 f00(fmaxf(FP8_MIN, fminf(FP8_MAX, w_pre[nt][0] * vsc0 * si0)));
          __nv_fp8_e4m3 f01(fmaxf(FP8_MIN, fminf(FP8_MAX, w_pre[nt][1] * vsc1 * si0)));
          __nv_fp8_e4m3 f10(fmaxf(FP8_MIN, fminf(FP8_MAX, w_pre[nt][2] * vsc0 * si1)));
          __nv_fp8_e4m3 f11(fmaxf(FP8_MIN, fminf(FP8_MAX, w_pre[nt][3] * vsc1 * si1)));
          sm_w_fp8[(size_t)gid * W_FP8_STRIDE + cand_e0] = f00.__x;
          sm_w_fp8[(size_t)gid * W_FP8_STRIDE + cand_e1] = f01.__x;
          sm_w_fp8[(size_t)(gid + 8) * W_FP8_STRIDE + cand_e0] = f10.__x;
          sm_w_fp8[(size_t)(gid + 8) * W_FP8_STRIDE + cand_e1] = f11.__x;
        }
      }
      bar_sync_t<3, DSV4_MATH_THREADS>();
      // Phase 4 FP8 MMA. Accumulate into persistent acc_nope[vc][nt][k].
      const float sc0 = sm.w_head_sc()[vc * HPB + gid];
      const float sc1 = sm.w_head_sc()[vc * HPB + gid + 8];
#pragma unroll
      for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
        const int dim = vc * V_CHUNK + warp_id * (NT_PER_WARP_XV * 8) + nt * 8;
        float xv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kstep = 0; kstep < XV_KSTEPS; kstep++) {
          const int ko = kstep * 32;
          uint32_t a0, a1, a2, a3, b0, b1;
          ldmatrix_load_A_fp8(a0, a1, a2, a3, sm_w_fp8 + ko, W_FP8_STRIDE, lane);
          d2_load_b_fp8<KV_SMEM_STRIDE>(b0, b1, sm_kv_fp8, kstep * 32, dim, lane);
          MmaFp8Result r = mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1, xv[0], xv[1], xv[2], xv[3]);
          xv[0] = r.d0;
          xv[1] = r.d1;
          xv[2] = r.d2;
          xv[3] = r.d3;
        }
        acc_nope[vc][nt][0] += xv[0] * sc0;
        acc_nope[vc][nt][1] += xv[1] * sc0;
        acc_nope[vc][nt][2] += xv[2] * sc1;
        acc_nope[vc][nt][3] += xv[3] * sc1;
      }
    }

    // ── Stage 3 RoPE bf16 ─────────────────────────────────────
    {
      const int rope_dim_base = warp_id * ROPE_DIMS_PER_WARP;
#pragma unroll
      for (int ks = 0; ks < ROPE_K_ITERS; ks++) {
        uint32_t a0, a1, a2, a3;
        ldmatrix_load_A_bf16(a0, a1, a2, a3, reinterpret_cast<const bf16*>(&sm_p_full[0][ks * 16]),
                             DSV4_BI, lane);
#pragma unroll
        for (int nt = 0; nt < ROPE_N_TILES; nt++) {
          const int n_col = rope_dim_base + nt * 8;
          const int k_base = ks * 16;
          const int ent0 = k_base + tid * 2;
          const int ent1 = ent0 + 1;
          const int ent8 = ent0 + 8;
          const int ent9 = ent0 + 9;
          const int col = n_col + gid;
          uint16_t v0 =
              *reinterpret_cast<const uint16_t*>(sm_kv_rope + (size_t)ent0 * D_ROPE_C + col);
          uint16_t v1 =
              *reinterpret_cast<const uint16_t*>(sm_kv_rope + (size_t)ent1 * D_ROPE_C + col);
          uint16_t v8 =
              *reinterpret_cast<const uint16_t*>(sm_kv_rope + (size_t)ent8 * D_ROPE_C + col);
          uint16_t v9 =
              *reinterpret_cast<const uint16_t*>(sm_kv_rope + (size_t)ent9 * D_ROPE_C + col);
          uint32_t b0 = (uint32_t)v0 | ((uint32_t)v1 << 16);
          uint32_t b1 = (uint32_t)v8 | ((uint32_t)v9 << 16);
          MmaBf16Result r = mma_bf16_m16n8k16(a0, a1, a2, a3, b0, b1, acc_rope[nt][0],
                                              acc_rope[nt][1], acc_rope[nt][2], acc_rope[nt][3]);
          acc_rope[nt][0] = r.d0;
          acc_rope[nt][1] = r.d1;
          acc_rope[nt][2] = r.d2;
          acc_rope[nt][3] = r.d3;
        }
      }
    }

    // sm_p_full + sm_w_fp8 reuse next iter — math-only sync ensures Stage 3
    // is fully drained before consumer_release lets IO overwrite the slot.
    bar_sync_t<3, DSV4_MATH_THREADS>();

    // Release the slot to IO (single signaling thread arrives mbar_empty).
    if (threadIdx.x == 0) {
      mbarrier_arrive(sm.mbar_empty(cons_idx));
    }
    ++cons_idx;
    if (cons_idx == DSV4_KV_BUF_COUNT) {
      cons_idx = 0;
      cons_phase ^= 1;
    }
  }  // chunk loop

  // ── Write per-split partial output + LSE to mid_out / mid_lse ───
  const float inv_g0 = (global_sum[0] > 0.f) ? (1.f / global_sum[0]) : 0.f;
  const float inv_g1 = (global_sum[1] > 0.f) ? (1.f / global_sum[1]) : 0.f;

  const size_t mid_o_base = ((size_t)t_idx * NUM_HEADS + h_start) * (size_t)num_splits * D_V_C +
                            (size_t)split_idx * D_V_C;

  // Pack adjacent (d0, d0+1) bf16 pairs into __nv_bfloat162 so the compiler
  // emits STG.E.64 instead of two STG.E.U16 — halves the global-store
  // instruction count and ~doubles sector-byte utilization (NCU A1.3 reported
  // 8.6 / 32 B/sector on these scalar stores, matching the unfused pattern).
  // d0 = warp_id*(NT_PER_WARP_XV*8) + nt*8 + tid*2 is always even ⇒ the
  // mid_out base+offset is 4-byte aligned, safe for __nv_bfloat162 access.
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
      // gid + 8 slot exists only when the kernel tile holds > 8 heads. For
      // NUM_HEADS=8 (small-TP configs) the second half is invalid and writing
      // would overflow mid_out's head dim.
      if constexpr (VALID_HPB > 8) {
        *reinterpret_cast<__nv_bfloat162*>(
            &mid_out[mid_o_base + (size_t)(gid + 8) * num_splits * D_V_C + d0]) = pair_hi;
      }
    }
  }
  {
    const int rope_dim_base = warp_id * ROPE_DIMS_PER_WARP;
#pragma unroll
    for (int nt = 0; nt < ROPE_N_TILES; nt++) {
      const int d0 = D_NOPE + rope_dim_base + nt * 8 + tid * 2;
      const __nv_bfloat162 pair_lo =
          __floats2bfloat162_rn(acc_rope[nt][0] * inv_g0, acc_rope[nt][1] * inv_g0);
      const __nv_bfloat162 pair_hi =
          __floats2bfloat162_rn(acc_rope[nt][2] * inv_g1, acc_rope[nt][3] * inv_g1);
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

// Merge kernel: collapse splits → final output + LSE.
//
// One block per (t, h) covering the full D_V via uint4 (8 bf16) vec loads.
// BLOCK_THREADS * DIMS_PER_THREAD must equal D_V_VAL. For D_V=512 the
// canonical instantiation is BLOCK_THREADS=64, DIMS_PER_THREAD=8 (one uint4
// per thread per split). Warp 0 computes the per-(t,h) LSE max/sum and
// broadcasts via smem so the inner accumulate loop reads each split's LSE
// from smem rather than re-fetching from L1/L2.
template <int NUM_HEADS, int D_V_VAL, int BLOCK_THREADS, int DIMS_PER_THREAD>
__global__ void __launch_bounds__(BLOCK_THREADS, 8) sparse_mla_decode_dsv4_merge_kernel(
    const bf16* __restrict__ mid_out, const float* __restrict__ mid_lse, bf16* __restrict__ output,
    float* __restrict__ out_lse,
    const float* __restrict__ attn_sink,  // [NUM_HEADS], nullable. natural-log domain.
    int num_tokens, int num_splits) {
  static_assert(BLOCK_THREADS % 32 == 0, "BLOCK_THREADS must be multiple of 32");
  static_assert(DIMS_PER_THREAD % 8 == 0, "DIMS_PER_THREAD must be multiple of 8 (uint4)");
  static_assert(BLOCK_THREADS * DIMS_PER_THREAD == D_V_VAL, "block must cover the full D_V row");
  constexpr int VECS_PER_THREAD = DIMS_PER_THREAD / 8;  // 1 for D_V=512, BT=64

  const int t_idx = blockIdx.x;
  const int h = blockIdx.y;
  if (t_idx >= num_tokens || h >= NUM_HEADS) return;
  const int tid = threadIdx.x;

  // Dynamic smem: cache runtime per-split LSE values so long C128A dual-cache
  // decode shapes are not capped by a compile-time split bound.
  extern __shared__ __align__(16) unsigned char merge_smem_raw[];
  float* sm_lse = reinterpret_cast<float*>(merge_smem_raw);
  __shared__ float sm_gmax;
  __shared__ float sm_inv_gsum;
  __shared__ float sm_glse;

  const float* lse_ptr = mid_lse + (size_t)t_idx * NUM_HEADS * num_splits + (size_t)h * num_splits;

  // Stage 0: cooperatively load mid_lse into smem.
  for (int sp = tid; sp < num_splits; sp += BLOCK_THREADS) {
    sm_lse[sp] = lse_ptr[sp];
  }
  __syncthreads();

  // Stage 1: warp-0 computes global_max + global_sum from sm_lse.
  if (tid < 32) {
    float local_max = -1e30f;
    for (int sp = tid; sp < num_splits; sp += 32) {
      local_max = fmaxf(local_max, sm_lse[sp]);
    }
#pragma unroll
    for (int s = 16; s >= 1; s >>= 1) {
      local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, s));
    }
    float gmax = (local_max > -1e29f) ? local_max : 0.f;

    float local_sum = 0.f;
    for (int sp = tid; sp < num_splits; sp += 32) {
      float lse_sp = sm_lse[sp];
      if (lse_sp > -1e29f) local_sum += exp2f(lse_sp - gmax);
    }
#pragma unroll
    for (int s = 16; s >= 1; s >>= 1) {
      local_sum += __shfl_xor_sync(0xffffffff, local_sum, s);
    }
    if (tid == 0) {
      // attn_sink (FlashMLA V4): the sink contributes a virtual logit at
      // exp2(sink_log2). Fold it into the normalizer so output and final
      // LSE both account for the sink's softmax mass.
      //   sum_with_sink = sum_partials + exp2(sink_log2 - gmax)
      //   glse           = log2(sum_with_sink) + gmax
      //                  = log2(exp2(glse_raw) + exp2(sink_log2))
      // Padded heads carry sink = -inf → exp2 = 0 → no-op (legacy semantics).
      float total_sum = local_sum;
      if (attn_sink != nullptr) {
        float sink_log2 = __ldg(attn_sink + h) * LOG2E;
        total_sum += exp2f(sink_log2 - gmax);
      }
      sm_gmax = gmax;
      sm_inv_gsum = (total_sum > 0.f) ? (1.f / total_sum) : 0.f;
      sm_glse = (total_sum > 0.f) ? (log2f(total_sum) + gmax) : -1e30f;
    }
  }
  __syncthreads();
  const float global_max = sm_gmax;
  const float inv_global_sum = sm_inv_gsum;

  // Stage 2: vec-load the partial outputs per split and accumulate.
  const bf16* mid_base = mid_out + ((size_t)t_idx * NUM_HEADS + h) * (size_t)num_splits * D_V_VAL;
  bf16* out_ptr = output + ((size_t)t_idx * NUM_HEADS + h) * D_V_VAL;
  const int dim_base = tid * DIMS_PER_THREAD;

  float acc[DIMS_PER_THREAD];
#pragma unroll
  for (int d = 0; d < DIMS_PER_THREAD; d++) acc[d] = 0.f;

  for (int sp = 0; sp < num_splits; sp++) {
    float lse_sp = sm_lse[sp];
    if (lse_sp <= -1e29f) continue;
    const float weight = exp2f(lse_sp - global_max);
    const bf16* row_base = mid_base + (size_t)sp * D_V_VAL + dim_base;
#pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; v++) {
      const uint4 packed = *reinterpret_cast<const uint4*>(row_base + v * 8);
      const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
      for (int p = 0; p < 4; p++) {
        const float2 f = __bfloat1622float2(pairs[p]);
        acc[v * 8 + p * 2 + 0] += weight * f.x;
        acc[v * 8 + p * 2 + 1] += weight * f.y;
      }
    }
  }

  // Stage 3: vec-store the normalized output.
#pragma unroll
  for (int v = 0; v < VECS_PER_THREAD; v++) {
    uint4 packed;
    __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
    for (int p = 0; p < 4; p++) {
      pairs[p] = __floats2bfloat162_rn(acc[v * 8 + p * 2 + 0] * inv_global_sum,
                                       acc[v * 8 + p * 2 + 1] * inv_global_sum);
    }
    *reinterpret_cast<uint4*>(out_ptr + dim_base + v * 8) = packed;
  }
  if (out_lse != nullptr && tid == 0) {
    out_lse[(size_t)t_idx * NUM_HEADS + h] = sm_glse;
  }
}

}  // namespace flashinfer::sparse_mla_sm120
