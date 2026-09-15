// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// (license header — full text mirrors the other files in this directory)

#pragma once

#include <type_traits>

#include "../../arch/cp_async.cuh"
#include "../../arch/matrix_memory.cuh"
#include "../../arch/mma_sm120.cuh"
#include "../../common/d2_load_b.cuh"
#include "../../common/zero_row.cuh"
#include "../../compute/online_softmax.cuh"
#include "../../compute/q_stage.cuh"
#include "../../compute/warp_tiles.cuh"
#include "../../execution/attention_params.cuh"
#include "../../model/kv_cache_traits.cuh"
#include "../../model/scale_convert.cuh"
#include "../../pipeline/staged_pipeline.cuh"
#include "resources.cuh"

struct Fp8DecodeGatherSchedule {
  static constexpr bool RAW_PIPELINE = false;
  static constexpr int EXTRA_THREADS = 0;
  static constexpr int MIN_BLOCKS = 0;
};

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

// The ordinary schedule omits minBlocksPerSM: it is smem-bound at 1 block/SM,
// and an unconstrained register budget avoids the spill forced by that hint.
// The mixed schedule supplies its own launch bound.
template <ModelType MT, int NUM_HEADS, int PAGE_BLOCK_SIZE,
          typename GatherSchedule = Fp8DecodeGatherSchedule>
__global__ void __launch_bounds__(
    (DecodeTileCfg<MT, GatherSchedule::RAW_PIPELINE>::template block_threads<GatherSchedule>()),
    GatherSchedule::MIN_BLOCKS)
    sparse_mla_decode_dsv4_kernel(
        const bf16* __restrict__ Q,            // [num_tokens, num_heads, d_qk] bf16
        const uint8_t* __restrict__ KV_cache,  // FP8 paged (DSV4 footer layout)
        const int32_t* __restrict__ indices,   // [num_tokens, topk] int32
        bf16* __restrict__ mid_out,   // [num_tokens, num_heads, scratch_split_stride, d_v] bf16
        float* __restrict__ mid_lse,  // [num_tokens, num_heads, scratch_split_stride] f32
        const int* __restrict__ topk_length_ptr,  // [num_tokens] or null
        // Optional secondary KV cache (DSv4 C4A / C128A). When non-null, the extra
        // candidate window is concatenated after the main one; per-chunk dispatch
        // in IO + math routes each chunk to the correct source.
        const uint8_t* __restrict__ extra_KV_cache,     // nullable; may use different pbs
        const int32_t* __restrict__ extra_indices,      // [num_tokens, extra_topk]
        const int* __restrict__ extra_topk_length_ptr,  // [num_tokens] or null
        int extra_topk,                                 // 0 = no extra cache
        int pbs_extra,  // page_block_size for extra cache (e.g. 2 for DSv4 C128A)
        size_t extra_page_stride_bytes, int num_tokens, int num_heads, int topk,
        int scratch_split_stride, int chunks_per_block, float sm_scale, size_t page_stride_bytes,
        // Row strides of (extra_)indices; either may exceed the row width when the
        // caller views a wider persistent buffer (last dim must stay contiguous).
        size_t indices_stride_elems, size_t extra_indices_stride_elems,
        std::conditional_t<MT == ModelType::DSV4, Dsv4PageDivisors, int> pages) {
  using KV = KVCacheTraits<MT>;
  using Cfg = DecodeTileCfg<MT, GatherSchedule::RAW_PIPELINE>;
  static_assert(MT == ModelType::DSV4 || MT == ModelType::DOTS3_SWA || MT == ModelType::DSV4_1,
                "decode-dsv4 serves the footer-scale model types (DSV4, DOTS3_SWA, DSV4_1)");
  constexpr int D_NOPE = KV::D_NOPE;                                // 448
  constexpr int D_ROPE_C = KV::D_ROPE;                              // 64
  constexpr int D_QK = KV::D_QK;                                    // 512
  constexpr int D_V_C = KV::D_V;                                    // 512
  constexpr int QUANT_TILE = KV::QUANT_TILE;                        // 64
  constexpr int NUM_SCALES = KV::NUM_SCALES;                        // 7
  constexpr int Q_NOPE_STRIDE = KV::Q_NOPE_STRIDE;                  // 464
  constexpr int KV_SMEM_STRIDE = KV::KV_SMEM_STRIDE;                // 464
  constexpr int SCALE_BYTES_PER_TOKEN = KV::SCALE_BYTES_PER_TOKEN;  // 8
  constexpr int IO_STRIDE = D_NOPE + D_ROPE_C * 2;
  using CacheLayout = std::conditional_t<MT == ModelType::DSV4_1, Dsv41Fp8Layout,
                                         FooterScaleLayout<IO_STRIDE, SCALE_BYTES_PER_TOKEN>>;
  const int pbs = [&]() {
    if constexpr (MT == ModelType::DSV4)
      return int(uint32_t(pages.main));
    else
      return MT == ModelType::DSV4_1 ? pages : PAGE_BLOCK_SIZE;
  }();
  // Kernel always computes a full HPB×CAND tile (zero-Q-padded for unused
  // head slots). NUM_HEADS == 0 selects the runtime-head-count instantiation:
  // one kernel per model type serves any num_heads <= 128. Q/output carry the
  // true num_heads stride while the mid scratch is HPB-aligned (gridDim.y *
  // HPB head rows per token), so both halves of the head tile write back
  // unconditionally and the merge kernel reads only h < num_heads. The
  // dedicated NUM_HEADS=8 instantiation keeps its true-H scratch, which makes
  // the second-half writeback a compile-time skip.
  constexpr bool RUNTIME_H = (NUM_HEADS == 0);
  constexpr int VALID_HPB = RUNTIME_H ? HPB : ((NUM_HEADS < HPB) ? NUM_HEADS : HPB);

  const int t_idx = blockIdx.x;
  const int h_block_idx = blockIdx.y;
  const int split_idx = blockIdx.z;
  if (t_idx >= num_tokens) return;

  const int h_start = h_block_idx * HPB;
  // Gmem row strides: Q/output use the true head count; the split-K scratch
  // is HPB-padded under RUNTIME_H so the (gid + 8) half-rows always exist.
  const int q_heads = RUNTIME_H ? num_heads : NUM_HEADS;
  const int mid_heads = RUNTIME_H ? (int)gridDim.y * HPB : NUM_HEADS;
  const int valid_h = RUNTIME_H ? min(num_heads - h_start, HPB) : VALID_HPB;
  // topk is the runtime indices-row width (the buffer bound); topk_length,
  // when given, is clamped to it.
  int topk_len = topk_length_ptr ? __ldg(topk_length_ptr + t_idx) : topk;
  topk_len = topk_len < 0 ? 0 : (topk_len > topk ? topk : topk_len);
  // Sliding-window models cap the candidate count at the window regardless of
  // what the caller passed, so omitting topk_length costs nothing: topk is
  // only the buffer width and the extra slots can never hold valid
  // candidates. Applied here rather than validated at the FFI so a caller
  // passing a too-large length cannot make the kernel over-scan either. The
  // host side rejects a buffer narrower than the window with a readable
  // error (SparseMlaSm120DecodeDsv4 in the JIT binding).
  if constexpr (Cfg::HAS_WINDOW) {
    topk_len = topk_len > Cfg::WINDOW ? Cfg::WINDOW : topk_len;
  }
  int extra_topk_len =
      (extra_KV_cache != nullptr)
          ? (extra_topk_length_ptr ? __ldg(extra_topk_length_ptr + t_idx) : extra_topk)
          : 0;
  extra_topk_len =
      extra_topk_len < 0 ? 0 : (extra_topk_len > extra_topk ? extra_topk : extra_topk_len);

  // Chunk range this block owns. Total chunks = main + extra (extra layout
  // is concatenated immediately after main; per-chunk dispatch in IO + math
  // routes to the right source).
  constexpr int SUBTILES = GatherSchedule::RAW_PIPELINE
                               ? kernels::dsv41_fp8::Dsv41MixedCacheDecodeResources::SUBTILES
                               : 1;
  constexpr int LOGICAL_WINDOW = Cfg::CAND_WINDOW * SUBTILES;
  const int num_orig_chunks = (topk_len + LOGICAL_WINDOW - 1) / LOGICAL_WINDOW * SUBTILES;
  const int num_extra_chunks = (extra_topk_len + LOGICAL_WINDOW - 1) / LOGICAL_WINDOW * SUBTILES;
  const int num_chunks_total = num_orig_chunks + num_extra_chunks;
  const int chunk_lo = split_idx * chunks_per_block * SUBTILES;
  const int chunk_hi = min(chunk_lo + chunks_per_block * SUBTILES, num_chunks_total);

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const bool is_io = (warp_id >= Cfg::N_WARPS);

  // Early-exit splits: only math threads write LSE; IO warp has nothing to do.
  if (chunk_lo >= num_chunks_total) {
    if (!is_io && threadIdx.x < valid_h) {
      const int h = h_start + threadIdx.x;
      const size_t lse_off = (size_t)t_idx * mid_heads * scratch_split_stride +
                             (size_t)h * scratch_split_stride + split_idx;
      mid_lse[lse_off] = -1e30f;
    }
    return;
  }

  constexpr int V_CHUNK = QUANT_TILE;  // DSV4 64, DOTS3_SWA 128, DSV4_1 32
  constexpr int N_V_CHUNKS = D_NOPE / V_CHUNK;
  constexpr int XV_FOLD = Cfg::XV_FOLD;    // W foldings per XV step (DSV4_1: 2)
  constexpr int XV_WARPS = Cfg::XV_WARPS;  // warps per chunk within a step
  constexpr int NT_PER_WARP_XV = V_CHUNK / 8 / XV_WARPS;
  // acc_nope's first index is the chunk-GROUP step, not the chunk: at
  // XV_FOLD=2 a warp's chunk within step vs is vs*2 + warp_id/XV_WARPS.
  constexpr int ACC_V_STEPS = N_V_CHUNKS / XV_FOLD;
  constexpr int XV_KSTEPS = Cfg::BI / 32;                      // 2
  constexpr int W_FP8_STRIDE = Cfg::BI + 16;                   // 80
  constexpr int ROPE_DIMS_PER_WARP = D_ROPE_C / Cfg::N_WARPS;  // 8
  constexpr int ROPE_N_TILES = ROPE_DIMS_PER_WARP / 8;         // 1
  constexpr int ROPE_K_ITERS = Cfg::BI / 16;                   // 4

  // Whether rope participates in V. DSV4: D_V = D_NOPE + D_ROPE, so the output
  // row carries a rope segment at [D_NOPE, D_V) fed by a P×V_rope MMA.
  // DOTS3_SWA: D_V = D_NOPE, no rope segment — the XV-rope stages below are
  // skipped entirely. This is a correctness guard, not an optimization: the
  // rope writeback indexes mid_out at D_NOPE + ..., which for D_V == D_NOPE
  // would run past the end of the row.
  // Note this is the *output* rope only; the QK rope path is unconditional.
  constexpr bool V_ROPE = KV::V_HAS_ROPE;
  static_assert(!V_ROPE || D_V_C == D_NOPE + D_ROPE_C,
                "V_HAS_ROPE implies the output row is nope followed by rope");
  static_assert(V_ROPE || D_V_C == D_NOPE, "without V rope the output row is pure nope");

  // ── Dynamic smem layout ────────────────────────────────────────
  // Single-buffer Q/scratch + double-buffered KV bufs.
  //   sm_q_rope    HPB * D_ROPE * 2B            =  2 KB
  //   sm_q_fp8     HPB * Q_NOPE_STRIDE          = 7.25 KB
  //   sm_q_sc      HPB * NUM_SCALES * 4B        = 0.44 KB
  //   sm_kv_fp8    2 * Cfg::BI * KV_SMEM_STRIDE   = 58 KB
  //   sm_kv_sc     2 * Cfg::BI * 8                = 1 KB
  //   sm_kv_rope   2 * Cfg::BI * D_ROPE * 2B      = 16 KB
  //   sm_reduce    2 * Cfg::N_WARPS * HPB * 4     = 1 KB  (8 warps)
  //   sm_w_head_sc N_V_CHUNKS * HPB * 4         = 448 B
  //   sm_w_fp8 ×2×XV_FOLD  W_FP8_SLOTS * HPB * (Cfg::BI + 16)
  //                                          = 2.5-5 KB (double-buf
  //                across step iters to drop the if-step>0 bar_sync)
  //   Total                                     ~ 88 KB
  // Plus static sm_p_full HPB * Cfg::BI * 2B (bf16) = 2 KB.
  // Grand total ~ 90 KB (under 100 KB SM120 carveout, 1 block/SM).
  extern __shared__ __align__(16) char smem_raw[];
  using Layout = Fp8DecodeSharedLayout<MT, GatherSchedule::RAW_PIPELINE>;
  auto sm = Layout::init(smem_raw);
  constexpr size_t RAW_OFFSET = Layout::OFF_W_FP8 + Layout::W_FP8_SLOTS * Layout::SMEM_W_FP8_BUF;

  // Only the XV-rope MMA consumes the bf16 P matrix; the XV-nope path reads the
  // FP8-quantized sm_w_fp8 instead. So with V_HAS_ROPE=false this buffer and the
  // stores feeding it are dead — size it away in the source rather than relying
  // on the optimizer to notice.
  __shared__ bf16 sm_p_full[HPB][V_ROPE ? Cfg::BI : 1];  // DSV4 2 KB static, DOTS3_SWA 0
  const int32_t* idx_base = indices + (size_t)t_idx * indices_stride_elems;

  // mbar_full: leader arrives + expect_tx, bulk completion drives tx.
  // mbar_empty: math signaling thread arrives. mbar.try_wait.parity has no
  // implicit memory fence — consumer must acq-rel via bar_sync after wait.
  using Ring = flashinfer::sparse_mla_sm120::pipeline::AsyncRing<Cfg::KV_BUF_COUNT>;
  if (threadIdx.x == 0) {
    Ring::init(sm.mbar_full(0), sm.mbar_empty(0));
    if constexpr (GatherSchedule::RAW_PIPELINE) {
      pipeline::BulkReady::init_slots<2>(GatherSchedule::ready(smem_raw + RAW_OFFSET));
    }
  }
  __syncthreads();
  if constexpr (GatherSchedule::RAW_PIPELINE) {
    static_assert(Cfg::MATH_THREADS == GatherSchedule::MATH_THREADS);
    struct Section {
      const uint8_t* kv;
      const int32_t* indices;
      size_t stride;
      int pbs, start, end;
      bool extra;
      __device__ int index(int row) const { return start + row < end ? indices[start + row] : -1; }
    };
    auto section = [&](int tile) {
      int chunk = chunk_lo + tile;
      bool extra = chunk >= num_orig_chunks;
      int start = (extra ? chunk - num_orig_chunks : chunk) * Cfg::BI;
      return Section{extra ? extra_KV_cache : KV_cache,
                     extra ? extra_indices + (size_t)t_idx * extra_indices_stride_elems : idx_base,
                     extra ? extra_page_stride_bytes : page_stride_bytes,
                     extra ? pbs_extra : pbs,
                     start,
                     min(start + Cfg::BI, extra ? extra_topk_len : topk_len),
                     extra};
    };
    if (is_io) {
      GatherSchedule::produce(sm, smem_raw + RAW_OFFSET, section, chunk_hi - chunk_lo);
      return;
    }
    GatherSchedule::math_registers();
  }

  // ── TMA bulk constants ──
  constexpr uint32_t DSV4_BULK_NOPE_BYTES = (uint32_t)D_NOPE;  // DSV4 448, DSV4_1 512
  constexpr uint32_t DSV4_BULK_ROPE_BYTES = (uint32_t)D_ROPE_C * sizeof(bf16);  // DSV4 128
  constexpr uint32_t DSV4_BULK_TX_BYTES =
      (uint32_t)Cfg::BI * (DSV4_BULK_NOPE_BYTES + DSV4_BULK_ROPE_BYTES);

  // IO gather: scalar scales → fence → expect_tx + bulks.
  // Dispatch per-chunk: chunks [0, num_orig_chunks) read from main KV_cache +
  // indices; chunks [num_orig_chunks, num_chunks_total) read from the
  // extra_KV_cache + extra_indices. The smem layout is shared — math warps
  // don't care which source the data came from.
  auto issue_gather = [&](int gather_chunk_idx, int buf) {
    const bool is_extra = (gather_chunk_idx >= num_orig_chunks);
    const int chunk_in_section = is_extra ? (gather_chunk_idx - num_orig_chunks) : gather_chunk_idx;
    const int section_len = is_extra ? extra_topk_len : topk_len;
    const int g_start = chunk_in_section * Cfg::CAND_WINDOW;
    const int g_end = min(g_start + Cfg::CAND_WINDOW, section_len);
    const int32_t* section_idx_base =
        is_extra ? (extra_indices + (size_t)t_idx * extra_indices_stride_elems) : idx_base;
    const uint8_t* section_kv = is_extra ? extra_KV_cache : KV_cache;
    const size_t section_stride = is_extra ? extra_page_stride_bytes : page_stride_bytes;
    const int section_pbs = is_extra ? pbs_extra : pbs;
    auto page_index = [&](int idx) {
      if constexpr (MT == ModelType::DSV4)
        return int(uint32_t(idx) / (is_extra ? pages.extra : pages.main));
      else
        return idx / section_pbs;
    };
    uint8_t* kv_fp8_dst = sm.kv_fp8(buf);
    bf16* kv_rope_dst = sm.kv_rope(buf);
    uint8_t* kv_sc_dst = sm.kv_sc(buf);

    // Per-lane entries (BI / IO_THREADS, compile-time): read each candidate's
    // index exactly once into registers — both the scale gather and the bulk
    // issues below derive from them. The per-lane index loads issue
    // back-to-back, as do the scale loads, so each round trip is paid once
    // per chunk instead of twice serialized.
    constexpr int EPW = Cfg::BI / Cfg::IO_THREADS;
    static_assert(Cfg::BI % Cfg::IO_THREADS == 0, "IO lanes must split BI evenly");
    int idx_raw[EPW];
#pragma unroll
    for (int e = 0; e < EPW; e++) {
      const int cand_pos = g_start + e * Cfg::IO_THREADS + lane;
      idx_raw[e] = (cand_pos < g_end) ? section_idx_base[cand_pos] : -1;
    }

    // Footer scale rows are 8B (DSV4, DOTS3_SWA) or 16B (DSV4_1); both stay
    // naturally aligned (the block stride and the in-block footer offset are
    // multiples of the row width).
    using ScaleWord = typename std::conditional<SCALE_BYTES_PER_TOKEN == 16, uint4, uint64_t>::type;
    ScaleWord scale_word[EPW];
#pragma unroll
    for (int e = 0; e < EPW; e++) {
      if constexpr (sizeof(ScaleWord) == 16) {
        scale_word[e] = make_uint4(0, 0, 0, 0);
      } else {
        scale_word[e] = 0;
      }
      if (idx_raw[e] >= 0) {
        const int idx = idx_raw[e];
        const int block_idx_g = page_index(idx);
        const int local_idx_g = idx - block_idx_g * section_pbs;
        const uint8_t* scale_base =
            section_kv + (size_t)block_idx_g * section_stride +
            CacheLayout::scale_offset(size_t(section_pbs), size_t(local_idx_g));
        scale_word[e] = __ldg(reinterpret_cast<const ScaleWord*>(scale_base));
      }
    }
#pragma unroll
    for (int e = 0; e < EPW; e++) {
      *reinterpret_cast<ScaleWord*>(kv_sc_dst + (size_t)(e * Cfg::IO_THREADS + lane) *
                                                    SCALE_BYTES_PER_TOKEN) = scale_word[e];
    }
    __threadfence_block();

    if (lane == 0) {
      pipeline::BulkReady::expect(sm.mbar_full(buf), DSV4_BULK_TX_BYTES);
    }

    // Issue cp.async.bulk for the FP8 data row (D_NOPE B/entry), plus the BF16
    // rope segment when the model has one (DSV4: 128 B/entry; DSV4_1: none).
    // Bulk completion decrements mbar tx; phase flips when arrival count
    // (1, by leader above) AND tx=0 both met.
#pragma unroll
    for (int e = 0; e < EPW; e++) {
      const int entry_idx = e * Cfg::IO_THREADS + lane;
      // Masked candidates gather the shared zero row, never a mutable cache
      // slot: a NaN there would leak through 0 * NaN in the value MMA.
      const bool valid = idx_raw[e] >= 0;
      const int idx = valid ? idx_raw[e] : 0;
      const int block_idx_g = page_index(idx);
      const int local_idx_g = idx - block_idx_g * section_pbs;
      const uint8_t* data_base = valid ? section_kv + (size_t)block_idx_g * section_stride +
                                             CacheLayout::data_offset(size_t(local_idx_g))
                                       : sparse_mla_zero_row;
      static_assert(DSV4_BULK_NOPE_BYTES + DSV4_BULK_ROPE_BYTES <= SPARSE_MLA_ZERO_ROW_BYTES);
      cp_async_bulk_g2s(kv_fp8_dst + (size_t)entry_idx * KV_SMEM_STRIDE, data_base,
                        DSV4_BULK_NOPE_BYTES, sm.mbar_full(buf));
      // DSV4_1 has no BF16 rope segment (rope lanes live in the FP8 row), so
      // there is no second bulk to issue.
      if constexpr (D_ROPE_C > 0) {
        cp_async_bulk_g2s(kv_rope_dst + (size_t)entry_idx * D_ROPE_C, data_base + D_NOPE,
                          DSV4_BULK_ROPE_BYTES, sm.mbar_full(buf));
      }
    }
  };

  if (is_io) {
    // Producer state: index, phase. Phase starts at 1 so first empty.wait(1)
    // on a freshly-initialized barrier (phase 0) returns immediately.
    uint32_t prod_phase = 1;
    int prod_idx = 0;
    for (int chunk_idx = chunk_lo; chunk_idx < chunk_hi; ++chunk_idx) {
      const int buf = (chunk_idx - chunk_lo) & 1;
      Ring::Free::wait(sm.mbar_empty(prod_idx), prod_phase);
      issue_gather(chunk_idx, buf);
      Ring::advance(prod_idx, prod_phase);
    }
    return;
  }

  // ──────────────────────────────────────────────────────────────
  // Math warps branch (warp_id < Cfg::N_WARPS = 4)
  // ──────────────────────────────────────────────────────────────

  const int gid = lane >> 2;
  const int tid = lane & 3;

  // Stage 0: Q quantization (math threads only; the helper uses bar:2
  // internally with count=Cfg::MATH_THREADS; rows past valid_h are zero-filled).
  const bf16* q_base = Q + (size_t)t_idx * q_heads * D_QK + (size_t)h_start * D_QK;
  quantize_q_to_smem<MT, Cfg::MATH_THREADS>(sm.q_fp8(), sm.q_sc(), sm.q_rope(), q_base, valid_h);

  // Persistent state across chunks (per-thread registers).
  float acc_nope[ACC_V_STEPS][NT_PER_WARP_XV][4] = {0};
  // Sized 1 when unused: a zero-length array is ill-formed, and every read is
  // behind `if constexpr (V_ROPE)`.
  float acc_rope[V_ROPE ? ROPE_N_TILES : 1][4] = {0};
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
    const int split_cand_start = chunk_in_section * Cfg::CAND_WINDOW;
    const int split_cand_end = min(split_cand_start + Cfg::CAND_WINDOW, section_len);

    if constexpr (GatherSchedule::RAW_PIPELINE) {
      if (!is_extra_chunk) Ring::Ready::wait(sm.mbar_full(cons_idx), cons_phase);
      GatherSchedule::KvReady::wait(buf);
    } else {
      // Wait for IO to fill this buf (mbar_full tx + arrival both met).
      Ring::Ready::wait(sm.mbar_full(cons_idx), cons_phase);
    }
    // CTA-wide acquire after the producer handoff. Without this, math reads
    // see only the view released
    // by whichever thread triggered the phase flip — other lanes' writes
    // (e.g., last-head RoPE bytes) may be stale.
    bar_sync_t<Cfg::MATH_BARRIER, Cfg::MATH_THREADS>();

    uint8_t* sm_kv_fp8 = sm.kv_fp8(buf);
    uint8_t* sm_kv_sc = sm.kv_sc(buf);
    bf16* sm_kv_rope = sm.kv_rope(buf);

    // ── Stage 2 QK ────────────────────────────────────────────
    float qk[Cfg::QK_N_TILES][4] = {0};
    static_assert(Cfg::QK_N_TILES == 1);
    qk_fp8_nope_16x8<KV>(qk[0], sm.q_fp8(), sm.q_sc(),
                         sm_kv_fp8 + (size_t)warp_id * Cfg::ENTRIES_PER_WARP * KV_SMEM_STRIDE,
                         sm_kv_sc + warp_id * Cfg::ENTRIES_PER_WARP * SCALE_BYTES_PER_TOKEN, lane);
    {
      // K-rope B operand: scalar per-lane reads. mma.m16n8k16 needs each lane's
      // b0/b1 to hold consecutive K-rows of one N-entry, which ldmatrix.x2.trans
      // can't produce from the N-outer smem here (gid = lane>>2, tid = lane&3).
      const int warp_first_cand = warp_id * Cfg::ENTRIES_PER_WARP;
#pragma unroll
      for (int ks = 0; ks < D_ROPE_C / 16; ks++) {
        uint32_t a0, a1, a2, a3;
        ldmatrix_load_A_bf16(a0, a1, a2, a3, sm.q_rope() + ks * 16, D_ROPE_C, lane);
#pragma unroll
        for (int nt = 0; nt < Cfg::QK_N_TILES; nt++) {
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
        is_extra_chunk ? (extra_indices + (size_t)t_idx * extra_indices_stride_elems) : idx_base;
    const int warp_first_cand = warp_id * Cfg::ENTRIES_PER_WARP;
#pragma unroll
    for (int nt = 0; nt < Cfg::QK_N_TILES; nt++) {
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
    for (int nt = 0; nt < Cfg::QK_N_TILES; nt++) {
      local_max[0] = fmaxf(local_max[0], fmaxf(qk[nt][0], qk[nt][1]));
      local_max[1] = fmaxf(local_max[1], fmaxf(qk[nt][2], qk[nt][3]));
    }
#pragma unroll
    for (int s = 2; s >= 1; s >>= 1) {
      local_max[0] = fmaxf(local_max[0], __shfl_xor_sync(0xffffffff, local_max[0], s));
      local_max[1] = fmaxf(local_max[1], __shfl_xor_sync(0xffffffff, local_max[1], s));
    }
    float local_sum[2] = {0.f, 0.f};
    float p[Cfg::QK_N_TILES][4];
#pragma unroll
    for (int nt = 0; nt < Cfg::QK_N_TILES; nt++) {
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
    bar_sync_t<Cfg::MATH_BARRIER, Cfg::MATH_THREADS>();
    if (threadIdx.x < VALID_HPB) {
      const int h = threadIdx.x;
      float wmax[Cfg::N_WARPS], wsum[Cfg::N_WARPS];
#pragma unroll
      for (int w = 0; w < Cfg::N_WARPS; w++) {
        wmax[w] = sm.warp_max()[w * HPB + h];
        wsum[w] = sm.warp_sum()[w * HPB + h];
      }
      float bmax = -1e30f;
#pragma unroll
      for (int w = 0; w < Cfg::N_WARPS; w++) bmax = fmaxf(bmax, wmax[w]);
      float bsum = 0.f;
#pragma unroll
      for (int w = 0; w < Cfg::N_WARPS; w++) bsum += wsum[w] * exp2f(wmax[w] - bmax);
      sm.warp_max()[h] = bmax;
      sm.warp_sum()[h] = bsum;
    }
    bar_sync_t<Cfg::MATH_BARRIER, Cfg::MATH_THREADS>();

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
      for (int vs = 0; vs < ACC_V_STEPS; vs++) {
#pragma unroll
        for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
          acc_nope[vs][nt][0] *= alpha0;
          acc_nope[vs][nt][1] *= alpha0;
          acc_nope[vs][nt][2] *= alpha1;
          acc_nope[vs][nt][3] *= alpha1;
        }
      }
      if constexpr (V_ROPE) {
#pragma unroll
        for (int nt = 0; nt < ROPE_N_TILES; nt++) {
          acc_rope[nt][0] *= alpha0;
          acc_rope[nt][1] *= alpha0;
          acc_rope[nt][2] *= alpha1;
          acc_rope[nt][3] *= alpha1;
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

    // Stage 2.75: sm_p_full = p * warp_rescale. Each warp uses its OWN
    // local_max-based rescale to kill all-invalid-warp contributions.
    // w_pre is shared: the XV-nope FP8 quantization below reads it too, so it
    // stays unconditional. Only the bf16 sm_p_full staging is rope-only.
    float w_pre[Cfg::QK_N_TILES][4];
#pragma unroll
    for (int nt = 0; nt < Cfg::QK_N_TILES; nt++) {
      w_pre[nt][0] = p[nt][0] * warp_rescale0;
      w_pre[nt][1] = p[nt][1] * warp_rescale0;
      w_pre[nt][2] = p[nt][2] * warp_rescale1;
      w_pre[nt][3] = p[nt][3] * warp_rescale1;
    }
    if constexpr (V_ROPE) {
      const int cand_col_base = warp_id * Cfg::ENTRIES_PER_WARP;
#pragma unroll
      for (int nt = 0; nt < Cfg::QK_N_TILES; nt++) {
        const int c0 = nt * 8 + tid * 2;
        const int c1 = c0 + 1;
        sm_p_full[gid][cand_col_base + c0] = __float2bfloat16(w_pre[nt][0]);
        sm_p_full[gid][cand_col_base + c1] = __float2bfloat16(w_pre[nt][1]);
        sm_p_full[gid + 8][cand_col_base + c0] = __float2bfloat16(w_pre[nt][2]);
        sm_p_full[gid + 8][cand_col_base + c1] = __float2bfloat16(w_pre[nt][3]);
      }
    }
    // Zero-init sm_w_head_sc here (different smem buffer than sm_p_full above),
    // so the single bar_sync below covers both write groups.
    for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; i += Cfg::MATH_THREADS) {
      sm.w_head_sc()[i] = 0.f;
    }
    bar_sync_t<Cfg::MATH_BARRIER, Cfg::MATH_THREADS>();

    // ── Stage 3 NoPE FP8 ──────────────────────────────────────
    {
      const int warp_first_cand_xv = warp_id * Cfg::ENTRIES_PER_WARP;
#pragma unroll
      for (int nt = 0; nt < Cfg::QK_N_TILES; nt++) {
        const int cand_e0 = warp_first_cand_xv + nt * 8 + tid * 2;
        const int cand_e1 = cand_e0 + 1;
#pragma unroll
        for (int vc = 0; vc < N_V_CHUNKS; vc++) {
          const float vsc0 =
              fp32_from_exponent_byte(sm_kv_sc[(size_t)cand_e0 * SCALE_BYTES_PER_TOKEN + vc]);
          const float vsc1 =
              fp32_from_exponent_byte(sm_kv_sc[(size_t)cand_e1 * SCALE_BYTES_PER_TOKEN + vc]);
          atomicMax(reinterpret_cast<int*>(&sm.w_head_sc()[vc * HPB + gid]),
                    __float_as_int(fmaxf(fabsf(w_pre[nt][0] * vsc0), fabsf(w_pre[nt][1] * vsc1))));
          atomicMax(reinterpret_cast<int*>(&sm.w_head_sc()[vc * HPB + gid + 8]),
                    __float_as_int(fmaxf(fabsf(w_pre[nt][2] * vsc0), fabsf(w_pre[nt][3] * vsc1))));
        }
      }
    }
    bar_sync_t<Cfg::MATH_BARRIER, Cfg::MATH_THREADS>();
    for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; i += Cfg::MATH_THREADS) {
      sm.w_head_sc()[i] = fmaxf(sm.w_head_sc()[i], 1e-10f) / FP8_MAX;
    }
    bar_sync_t<Cfg::MATH_BARRIER, Cfg::MATH_THREADS>();

#pragma unroll
    for (int vs = 0; vs < ACC_V_STEPS; vs++) {
      const int vc0 = vs * XV_FOLD;
      // Double-buffered by step parity: step N quants into parity N&1 while
      // step N-1's MMA reads the other buffer set, so quant cannot race the
      // reads. Each parity owns XV_FOLD W buffers (one folding per scale
      // group in the chunk group). The bar_sync after quant is the only sync
      // needed within the step loop.
      // Phase 3 quant: fold each candidate once per scale group in the step.
#pragma unroll
      for (int f = 0; f < XV_FOLD; f++) {
        const int vc = vc0 + f;
        uint8_t* sm_w_fp8 = sm.w_fp8((vs & 1) * XV_FOLD + f);
        const int warp_first_cand_xv = warp_id * Cfg::ENTRIES_PER_WARP;
        const float si0 = 1.f / sm.w_head_sc()[vc * HPB + gid];
        const float si1 = 1.f / sm.w_head_sc()[vc * HPB + gid + 8];
#pragma unroll
        for (int nt = 0; nt < Cfg::QK_N_TILES; nt++) {
          const int cand_e0 = warp_first_cand_xv + nt * 8 + tid * 2;
          const int cand_e1 = cand_e0 + 1;
          const float vsc0 =
              fp32_from_exponent_byte(sm_kv_sc[(size_t)cand_e0 * SCALE_BYTES_PER_TOKEN + vc]);
          const float vsc1 =
              fp32_from_exponent_byte(sm_kv_sc[(size_t)cand_e1 * SCALE_BYTES_PER_TOKEN + vc]);
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
      bar_sync_t<Cfg::MATH_BARRIER, Cfg::MATH_THREADS>();
      // Phase 4 FP8 MMA. This warp's chunk within the step and its W folding:
      // warp groups of XV_WARPS share one chunk, tiling its dims.
      const int vc = vc0 + warp_id / XV_WARPS;
      uint8_t* sm_w_fp8 = sm.w_fp8((vs & 1) * XV_FOLD + warp_id / XV_WARPS);
      const float sc0 = sm.w_head_sc()[vc * HPB + gid];
      const float sc1 = sm.w_head_sc()[vc * HPB + gid + 8];
#pragma unroll
      for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
        const int dim = vc * V_CHUNK + (warp_id % XV_WARPS) * (NT_PER_WARP_XV * 8) + nt * 8;
        float xv[4] = {0.f, 0.f, 0.f, 0.f};
        pv_fp8_d2_16x8<KV_SMEM_STRIDE, W_FP8_STRIDE, XV_KSTEPS>(xv, sm_w_fp8, sm_kv_fp8, dim, lane);
        acc_nope[vs][nt][0] += xv[0] * sc0;
        acc_nope[vs][nt][1] += xv[1] * sc0;
        acc_nope[vs][nt][2] += xv[2] * sc1;
        acc_nope[vs][nt][3] += xv[3] * sc1;
      }
    }

    // ── Stage 3 RoPE bf16 ─────────────────────────────────────
    // Skipped when V has no rope segment (DOTS3_SWA).
    if constexpr (V_ROPE) {
      const int rope_dim_base = warp_id * ROPE_DIMS_PER_WARP;
#pragma unroll
      for (int ks = 0; ks < ROPE_K_ITERS; ks++) {
        uint32_t a0, a1, a2, a3;
        ldmatrix_load_A_bf16(a0, a1, a2, a3, reinterpret_cast<const bf16*>(&sm_p_full[0][ks * 16]),
                             Cfg::BI, lane);
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
    bar_sync_t<Cfg::MATH_BARRIER, Cfg::MATH_THREADS>();

    // Release the slot to IO (single signaling thread arrives mbar_empty).
    if constexpr (GatherSchedule::RAW_PIPELINE) {
      GatherSchedule::KvFree::release(buf);
    } else if (threadIdx.x == 0) {
      Ring::Free::publish(sm.mbar_empty(cons_idx));
    }
    Ring::advance(cons_idx, cons_phase);
  }  // chunk loop

  // ── Write per-split partial output + LSE to mid_out / mid_lse ───
  const float inv_g0 = (global_sum[0] > 0.f) ? (1.f / global_sum[0]) : 0.f;
  const float inv_g1 = (global_sum[1] > 0.f) ? (1.f / global_sum[1]) : 0.f;

  const size_t mid_o_base =
      ((size_t)t_idx * mid_heads + h_start) * (size_t)scratch_split_stride * D_V_C +
      (size_t)split_idx * D_V_C;

  // Pack adjacent (d0, d0+1) bf16 pairs into __nv_bfloat162 so the compiler
  // emits STG.E.64 instead of two STG.E.U16 — halves the global-store
  // instruction count and ~doubles sector-byte utilization (NCU A1.3 reported
  // 8.6 / 32 B/sector on these scalar stores, matching the unfused pattern).
  // d0's + tid*2 term is always even ⇒ the mid_out base+offset is 4-byte
  // aligned, safe for __nv_bfloat162 access.
#pragma unroll
  for (int vs = 0; vs < ACC_V_STEPS; vs++) {
#pragma unroll
    for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
      const int vc = vs * XV_FOLD + warp_id / XV_WARPS;
      const int d0 = vc * V_CHUNK + (warp_id % XV_WARPS) * (NT_PER_WARP_XV * 8) + nt * 8 + tid * 2;
      const __nv_bfloat162 pair_lo =
          __floats2bfloat162_rn(acc_nope[vs][nt][0] * inv_g0, acc_nope[vs][nt][1] * inv_g0);
      const __nv_bfloat162 pair_hi =
          __floats2bfloat162_rn(acc_nope[vs][nt][2] * inv_g1, acc_nope[vs][nt][3] * inv_g1);
      *reinterpret_cast<__nv_bfloat162*>(
          &mid_out[mid_o_base + (size_t)gid * scratch_split_stride * D_V_C + d0]) = pair_lo;
      // gid + 8 slot exists only when the kernel tile holds > 8 heads. The
      // dedicated NUM_HEADS=8 instantiation strides mid_out by the true head
      // count, so the second half would overflow it; the runtime-H
      // instantiation HPB-aligns the scratch, making the write unconditional.
      if constexpr (VALID_HPB > 8) {
        *reinterpret_cast<__nv_bfloat162*>(
            &mid_out[mid_o_base + (size_t)(gid + 8) * scratch_split_stride * D_V_C + d0]) = pair_hi;
      }
    }
  }
  // Rope segment of the output row, at [D_NOPE, D_V). Absent when D_V == D_NOPE
  // (DOTS3_SWA) — writing it there would index past the end of the row.
  if constexpr (V_ROPE) {
    const int rope_dim_base = warp_id * ROPE_DIMS_PER_WARP;
#pragma unroll
    for (int nt = 0; nt < ROPE_N_TILES; nt++) {
      const int d0 = D_NOPE + rope_dim_base + nt * 8 + tid * 2;
      const __nv_bfloat162 pair_lo =
          __floats2bfloat162_rn(acc_rope[nt][0] * inv_g0, acc_rope[nt][1] * inv_g0);
      const __nv_bfloat162 pair_hi =
          __floats2bfloat162_rn(acc_rope[nt][2] * inv_g1, acc_rope[nt][3] * inv_g1);
      *reinterpret_cast<__nv_bfloat162*>(
          &mid_out[mid_o_base + (size_t)gid * scratch_split_stride * D_V_C + d0]) = pair_lo;
      if constexpr (VALID_HPB > 8) {
        *reinterpret_cast<__nv_bfloat162*>(
            &mid_out[mid_o_base + (size_t)(gid + 8) * scratch_split_stride * D_V_C + d0]) = pair_hi;
      }
    }
  }
  if (warp_id == 0 && tid == 0) {
    const float lse0 = (global_sum[0] > 0.f) ? (log2f(global_sum[0]) + global_max[0]) : -1e30f;
    const float lse1 = (global_sum[1] > 0.f) ? (log2f(global_sum[1]) + global_max[1]) : -1e30f;
    const size_t lse_base =
        (size_t)t_idx * mid_heads * scratch_split_stride + (size_t)h_start * scratch_split_stride;
    mid_lse[lse_base + (size_t)gid * scratch_split_stride + split_idx] = lse0;
    if constexpr (VALID_HPB > 8) {
      mid_lse[lse_base + (size_t)(gid + 8) * scratch_split_stride + split_idx] = lse1;
    }
  }
}

}  // namespace flashinfer::sparse_mla_sm120
