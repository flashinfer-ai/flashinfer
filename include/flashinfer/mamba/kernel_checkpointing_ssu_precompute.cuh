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
// =============================================================================
#ifndef FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_PRECOMPUTE_CUH_
#define FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_PRECOMPUTE_CUH_

#include <cute/tensor.hpp>

#include "checkpointing_ssu.cuh"
#include "common.cuh"  // PackedAligned
#include "kernel_checkpointing_ssu_common.cuh"

namespace flashinfer::mamba::checkpointing {

template <typename input_t, int NPREDICTED, int MAX_WINDOW, int DSTATE, int NUM_WARPS,
          int HEADS_PER_CTA>
struct CheckpointingSsuPrecomputeStorage {
  static constexpr int NPREDICTED_PAD_MMA_M = next_multiple_of<MMA_prop::M>(NPREDICTED);
  static constexpr int NPREDICTED_PAD_MMA_N = next_multiple_of<MMA_prop::N>(NPREDICTED);
  static constexpr int MAX_WINDOW_PAD_MMA_K = next_multiple_of<MMA_prop::K_SMALL>(MAX_WINDOW);
  static constexpr int NPREDICTED_SWIZZLE_R =
      next_multiple_of<SmemSwizzle<input_t>::ATOM_ROWS>(NPREDICTED);
  static constexpr int CB_NEW_REGS = NPREDICTED_PAD_MMA_M / 2;  // m16n8k16 → 8
  static constexpr int CB_OLD_REGS = MAX_WINDOW_PAD_MMA_K / 2;  // m16n8k{K_old} → 4 or 8
  float CB[CB_NEW_REGS * warpSize];      // C5 new-token raw C·B, fragA element-major (e*W+lane)
  float CB_old[CB_OLD_REGS * warpSize];  // C6 old-token raw C·old_B (no-write), element-major

  alignas(16) input_t B[NPREDICTED_PAD_MMA_N * DSTATE];      // matmul-1 N-operand (new B, swizzled)
  alignas(16) input_t C[NPREDICTED_SWIZZLE_R * DSTATE];      // matmul-1 A-operand (C, swizzled)
  alignas(16) input_t old_B[MAX_WINDOW_PAD_MMA_K * DSTATE];  // C6 N-operand (old B, no-write only)

  float dt[HEADS_PER_CTA][NPREDICTED];      // C1: raw dt (PHASE 1) → softplus in place (PHASE 2)
  float cumAdt[HEADS_PER_CTA][NPREDICTED];  // C2 scan output (PHASE 2)
  float old_dt[HEADS_PER_CTA][MAX_WINDOW];  // C6/C7 (NO-WRITE)
  float old_cumAdt[HEADS_PER_CTA][MAX_WINDOW];  // C6/C7 (NO-WRITE)
  float dt_bias[HEADS_PER_CTA];                 // per-head bias (PHASE 1)
  float A[HEADS_PER_CTA];                       // per-head A   (PHASE 1)
};

template <typename input_t, int NPREDICTED, int DSTATE, typename SmemT>
__device__ __forceinline__ void load_C(SmemT& smem, CheckpointingSsuParams const& params, int lane,
                                       int group_idx, int64_t outer, int seq_len) {
  auto const* __restrict__ C_ptr = reinterpret_cast<input_t const*>(params.C);
  int64_t const C_base = outer * params.C_stride_seq + (int64_t)group_idx * DSTATE;
  using CShape = cute::Shape<cute::Int<SmemT::NPREDICTED_SWIZZLE_R>, cute::Int<DSTATE>>;
  load_tile_async<CShape, NPREDICTED>(smem.C, C_ptr + C_base, params.C_stride_token, lane, seq_len);
  __pipeline_commit();
}

template <typename input_t, int NPREDICTED, int DSTATE, typename SmemT>
__device__ __forceinline__ void load_B(SmemT& smem, CheckpointingSsuParams const& params, int lane,
                                       int group_idx, int64_t outer, int seq_len) {
  auto const* __restrict__ B_ptr = reinterpret_cast<input_t const*>(params.B);
  int64_t const B_base = outer * params.B_stride_seq + (int64_t)group_idx * DSTATE;
  using BShape = cute::Shape<cute::Int<SmemT::NPREDICTED_PAD_MMA_N>, cute::Int<DSTATE>>;
  load_tile_async<BShape, NPREDICTED>(smem.B, B_ptr + B_base, params.B_stride_token, lane, seq_len);
  __pipeline_commit();
}

template <typename input_t, int MAX_WINDOW, int DSTATE, typename SmemT>
__device__ __forceinline__ void load_old_B(SmemT& smem, CheckpointingSsuParams const& params,
                                           int lane, int64_t cache_slot, int ring_start,
                                           int group_idx, int prev_k) {
  auto const* __restrict__ oldB_ptr = reinterpret_cast<input_t const*>(params.B_cache);
  int64_t const base =
      cache_slot * params.B_cache_stride_seq + (int64_t)group_idx * params.B_cache_stride_group;
  using OldBShape = cute::Shape<cute::Int<SmemT::MAX_WINDOW_PAD_MMA_K>, cute::Int<DSTATE>>;
  load_ring_tile_async<OldBShape, MAX_WINDOW>(smem.old_B, oldB_ptr + base,
                                              (int)params.B_cache_stride_pos, lane, ring_start,
                                              params.ring_buffer_len, prev_k);
  __pipeline_commit();  // NOT drained here — see load_C/load_B; caller drains once before the MMA.
}

template <typename dt_t, typename weight_t, typename matrixA_t, int NPREDICTED, int MAX_WINDOW,
          int HEADS_PER_CTA, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void load_phase1_coeffs(SmemT& smem,
                                                   CheckpointingSsuParams const& params, int warp,
                                                   int lane, int first_head, int seq_len,
                                                   int64_t outer, int seq, int64_t cache_slot,
                                                   int ring_start, bool must_checkpoint,
                                                   int prev_k) {
  auto const* __restrict__ A_ptr = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ dt_bias_ptr = reinterpret_cast<weight_t const*>(params.dt_bias);
  int const flat_tid = warp * warpSize + lane;
  constexpr int CTA_THREADS = NUM_WARPS * 32;
  // dt[t, head]: heads contiguous in gmem (stride 1), tokens strided (dt_stride_token).
  // idx → (t, h) with h = idx % HEADS_PER_CTA so consecutive threads hit consecutive
  // heads → coalesced read.
  auto const* __restrict__ dt_ptr = reinterpret_cast<dt_t const*>(params.dt);
  constexpr int DT_N = HEADS_PER_CTA * NPREDICTED;
#pragma unroll
  for (int idx = flat_tid; idx < DT_N; idx += CTA_THREADS) {
    int const h = idx % HEADS_PER_CTA;
    int const t = idx / HEADS_PER_CTA;
    float v = 0.f;
    if (t < seq_len)
      v = toFloat(dt_ptr[outer * params.dt_stride_seq + (int64_t)t * params.dt_stride_token +
                         first_head + h]);
    smem.dt[h][t] = v;  // raw; bias + softplus applied in PHASE 2
  }
  // Per-head dt_bias + A (one thread per head; heads contiguous → coalesced).
  if (flat_tid < HEADS_PER_CTA) {
    smem.A[flat_tid] = toFloat(A_ptr[first_head + flat_tid]);
    smem.dt_bias[flat_tid] = dt_bias_ptr ? toFloat(dt_bias_ptr[first_head + flat_tid]) : 0.f;
  }
  // Old-decay scan feeds ONLY the cb_old scaling in PHASE 2 (no-write path).  The
  // main recomputes its own decay in-registers from the dt ring (pre-gdc-pure
  // replay), so nothing is staged to gmem and the write path skips the scan.
  if (!must_checkpoint) {
    auto const* __restrict__ dtc_ptr = reinterpret_cast<float const*>(params.dt_cache);
    int64_t const dtc_base = cache_slot * params.dt_cache_stride_seq;
    int const scan_warp = flat_tid >> 5;
    int const scan_lane = flat_tid & 31;
    constexpr int NWARPS = CTA_THREADS / 32;
    for (int h = scan_warp; h < HEADS_PER_CTA; h += NWARPS) {
      int64_t const hh = (int64_t)(first_head + h);
      float const a = toFloat(A_ptr[hh]);
      int rr = ring_start + scan_lane;
      if (rr >= params.ring_buffer_len) rr -= params.ring_buffer_len;
      float const dv =
          (scan_lane < prev_k) ? dtc_ptr[dtc_base + hh * params.dt_cache_stride_head + rr] : 0.f;
      float scan = dv;
#pragma unroll
      for (int off = 1; off < MAX_WINDOW; off <<= 1) {
        float const up = __shfl_up_sync(constants::MASK_ALL_LANES, scan, off);
        if (scan_lane >= off) scan += up;
      }
      if (scan_lane < MAX_WINDOW) {
        smem.old_dt[h][scan_lane] = dv;
        smem.old_cumAdt[h][scan_lane] = (scan_lane < prev_k) ? a * scan : 0.f;
      }
    }
  }
}

template <typename dt_t, int NPREDICTED, bool DT_SOFTPLUS, typename SmemT>
__device__ __forceinline__ void load_dt(SmemT& smem, CheckpointingSsuParams const& params, int warp,
                                        int lane, int head, int64_t outer, float dt_bias_val,
                                        int seq_len) {
  if (lane < seq_len) {
    auto const* __restrict__ dt_ptr = reinterpret_cast<dt_t const*>(params.dt);
    int64_t const base = outer * params.dt_stride_seq + head;
    float dt_val = toFloat(dt_ptr[base + (int64_t)lane * params.dt_stride_token]) + dt_bias_val;
    if constexpr (DT_SOFTPLUS) dt_val = thresholded_softplus(dt_val);
    smem.dt[warp][lane] = dt_val;
  }
}

template <int NPREDICTED, typename SmemT>
__device__ __forceinline__ void compute_cumAdt_pw(SmemT& smem, int h, int lane, float A_val) {
  float val = (lane < NPREDICTED) ? A_val * smem.dt[h][lane] : 0.f;
  for (int offset = 1; offset < NPREDICTED; offset *= 2) {
    float other = __shfl_up_sync(constants::MASK_ALL_LANES, val, offset);
    if (lane >= offset) val += other;
  }
  if (lane < NPREDICTED) {
    smem.cumAdt[h][lane] = val;
  }
}

template <typename input_t, int NPREDICTED, int DSTATE, typename SmemT>
__device__ __forceinline__ void compute_cb_2warp(SmemT& smem, int warp, int lane) {
  using namespace cute;
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int NPREDICTED_PAD_MMA_N = SmemT::NPREDICTED_PAD_MMA_N;
  constexpr int N_HALF = NPREDICTED_PAD_MMA_M / 2;
  static_assert(N_HALF % MMA_prop::N == 0, "N_HALF must be a multiple of MMA::N");

  if constexpr (NPREDICTED_PAD_MMA_N == 8) {
    if (warp == 1) return;  // no valid B rows; cols [8,16) never read.
  }

  // ── Swizzled smem views (C/B byte-identical to the monolithic layout) ──
  auto layout_C =
      make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, DSTATE, NPREDICTED>();
  auto layout_B = make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_N, DSTATE>();
  Tensor smem_C = make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.C)), layout_C);
  Tensor smem_B = make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.B)), layout_B);

  // ── TiledMMA: _1x_1 = 32 threads, one [16, 8] atom ──
  auto tiled_mma =
      make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{}, Layout<Shape<_1, _1>>{});
  auto thr_mma = tiled_mma.get_slice(lane);

  constexpr int K_TILE = MMA_prop::K_BIG;
  Tensor smem_C_tiled = local_tile(smem_C, make_tile(Int<NPREDICTED_PAD_MMA_M>{}, Int<K_TILE>{}),
                                   make_coord(_0{}, _));
  Tensor smem_B_half =
      local_tile(smem_B, make_tile(Int<N_HALF>{}, Int<K_TILE>{}), make_coord(warp, _));

  Tensor frag_A = thr_mma.partition_fragment_A(smem_C_tiled(_, _, _0{}));
  Tensor frag_B = thr_mma.partition_fragment_B(smem_B_half(_, _, _0{}));
  auto layout_cb_half = make_layout(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_HALF>{}));
  Tensor frag_acc = thr_mma.partition_fragment_C(make_tensor((float*)nullptr, layout_cb_half));
  clear(frag_acc);

  auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, input_t>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(lane);
  Tensor smem_C_s2r = s2r_thr_A.partition_S(smem_C_tiled);
  Tensor frag_A_view = s2r_thr_A.retile_D(frag_A);
  auto s2r_B = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, input_t>{}, tiled_mma);
  auto s2r_thr_B = s2r_B.get_slice(lane);
  Tensor smem_B_s2r = s2r_thr_B.partition_S(smem_B_half);
  Tensor frag_B_view = s2r_thr_B.retile_D(frag_B);

  // ── Gemm: DSTATE/K_TILE K-tiles, 1 HMMA each ──
  constexpr int NUM_K_TILES = DSTATE / K_TILE;
#pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    cute::copy(s2r_A, smem_C_s2r(_, _, _, k), frag_A_view);
    cute::copy(s2r_B, smem_B_s2r(_, _, _, k), frag_B_view);
    cute::gemm(tiled_mma, frag_acc, frag_A, frag_B, frag_acc);
  }

  // ── Store the m16n8 accumulator ELEMENT-MAJOR → smem.CB (no swizzle) ──
  int const warp_base = warp * (int)size(frag_acc);  // 4 regs/lane (m16n8 C)
#pragma unroll
  for (int i = 0; i < size(frag_acc); ++i) {
    smem.CB[(warp_base + i) * warpSize + lane] = frag_acc(i);  // raw fp32
  }
}

template <typename input_t, int NPREDICTED, int MAX_WINDOW, int DSTATE, typename SmemT>
__device__ __forceinline__ void compute_cb_old_2warp(SmemT& smem, int warp_in_pair, int lane) {
  using namespace cute;
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  constexpr int N_HALF = MMA_prop::N;  // 8 = one m16n8 N-tile per warp
  static_assert(MAX_WINDOW_PAD_MMA_K % N_HALF == 0, "MAX_WINDOW_PAD_MMA_K must be a multiple of N");

  if constexpr (MAX_WINDOW_PAD_MMA_K == N_HALF) {
    if (warp_in_pair == 1) return;  // single N-tile; cols [8,16) never read.
  }

  // ── Swizzled smem views (C/old_B byte-identical to the monolithic layout) ──
  auto layout_C =
      make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, DSTATE, NPREDICTED>();
  auto layout_oldB = make_swizzled_layout_rc<input_t, MAX_WINDOW_PAD_MMA_K, DSTATE>();
  Tensor smem_C = make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.C)), layout_C);
  Tensor smem_oldB =
      make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.old_B)), layout_oldB);

  auto tiled_mma =
      make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{}, Layout<Shape<_1, _1>>{});
  auto thr_mma = tiled_mma.get_slice(lane);

  constexpr int K_TILE = MMA_prop::K_BIG;
  Tensor smem_C_tiled = local_tile(smem_C, make_tile(Int<NPREDICTED_PAD_MMA_M>{}, Int<K_TILE>{}),
                                   make_coord(_0{}, _));
  Tensor smem_oldB_half =
      local_tile(smem_oldB, make_tile(Int<N_HALF>{}, Int<K_TILE>{}), make_coord(warp_in_pair, _));

  Tensor frag_A = thr_mma.partition_fragment_A(smem_C_tiled(_, _, _0{}));
  Tensor frag_B = thr_mma.partition_fragment_B(smem_oldB_half(_, _, _0{}));
  auto layout_cb_half = make_layout(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_HALF>{}));
  Tensor frag_acc = thr_mma.partition_fragment_C(make_tensor((float*)nullptr, layout_cb_half));
  clear(frag_acc);

  auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, input_t>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(lane);
  Tensor smem_C_s2r = s2r_thr_A.partition_S(smem_C_tiled);
  Tensor frag_A_view = s2r_thr_A.retile_D(frag_A);
  auto s2r_B = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, input_t>{}, tiled_mma);
  auto s2r_thr_B = s2r_B.get_slice(lane);
  Tensor smem_oldB_s2r = s2r_thr_B.partition_S(smem_oldB_half);
  Tensor frag_B_view = s2r_thr_B.retile_D(frag_B);

  constexpr int NUM_K_TILES = DSTATE / K_TILE;
#pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    cute::copy(s2r_A, smem_C_s2r(_, _, _, k), frag_A_view);
    cute::copy(s2r_B, smem_oldB_s2r(_, _, _, k), frag_B_view);
    cute::gemm(tiled_mma, frag_acc, frag_A, frag_B, frag_acc);
  }

  int const warp_base = warp_in_pair * (int)size(frag_acc);  // 4 regs/lane (m16n8 C)
#pragma unroll
  for (int i = 0; i < size(frag_acc); ++i) {
    smem.CB_old[(warp_base + i) * warpSize + lane] = frag_acc(i);  // raw fp32
  }
}

template <bool IS_OLD, int REGS, typename input_t, typename SmemT>
__device__ __forceinline__ void scale_store_cb(SmemT& smem, int h, int lane, int seq_len,
                                               int prev_k, float const (&raw)[REGS],
                                               input_t* __restrict__ gmem_head) {
  constexpr int NUM_OUT_NTILES = SmemT::NPREDICTED_PAD_MMA_N / MMA_prop::N;
  constexpr int REGS_B_PER = REGS / 2;  // fragB regs per output N-tile (= K/4)
  constexpr int REGS_B = NUM_OUT_NTILES * REGS_B_PER;
  float total_old = 0.f;
  if constexpr (IS_OLD) {
    total_old = (prev_k > 0) ? smem.old_cumAdt[h][prev_k - 1] : 0.f;
  }
  int const r0 = lane / 4;
  int const c0 = (lane % 4) * 2;
  PackedAligned<input_t, REGS_B> packed;
#pragma unroll
  for (int g = 0; g < REGS_B; ++g) {
    int const on = g / REGS_B_PER;  // output N-tile (t-half): 0 → t=r0, 1 → t=r0+8
    int const r = g % REGS_B_PER;   // B-register within the output N-tile
    int const src = ((r >> 1) & 1) * 4 + (on << 1) + (r & 1);
    int const t = r0 + on * 8;
    int const c = c0 + (((r >> 1) & 1) << 3) + (r & 1);
    float val = 0.f;
    if constexpr (IS_OLD) {
      // C6: old tokens all precede the new ones — mask c<prev_k, no causal.
      if (c < prev_k && t < seq_len)
        val = raw[src] * __expf(smem.cumAdt[h][t] + total_old - smem.old_cumAdt[h][c]) *
              smem.old_dt[h][c];
    } else {
      // C5: causal CB_scaled[t,c]; per-head coeffs (this head's PHASE-1/scan data).
      if (c <= t && t < seq_len && c < seq_len)
        val = raw[src] * __expf(smem.cumAdt[h][t] - smem.cumAdt[h][c]) * smem.dt[h][c];
    }
    packed.val[g] = static_cast<input_t>(val);
  }
  reinterpret_cast<PackedAligned<input_t, REGS_B>*>(gmem_head)[lane] = packed;  // one STG
}

// -----------------------------------------------------------------------------
// PRECOMPUTE kernel.  Template params mirror checkpointing_ssu_kernel.
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, int NPREDICTED, int MAX_WINDOW, int DIM, int DSTATE,
          int HEADS_PER_GROUP, int HEADS_PER_CTA, int NUM_WARPS, bool DT_SOFTPLUS,
          bool VARLEN = false>
__global__ void checkpointing_ssu_precompute_kernel(CheckpointingSsuParams params) {
  using SmemT = CheckpointingSsuPrecomputeStorage<input_t, NPREDICTED, MAX_WINDOW, DSTATE,
                                                  NUM_WARPS, HEADS_PER_CTA>;
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int N_HALF = NPREDICTED_PAD_MMA_M / 2;

  extern __shared__ __align__(128) char smem_buf[];
  auto& smem = *reinterpret_cast<SmemT*>(smem_buf);

  // ── INTERNAL PDL (precompute → main): UNCONDITIONAL — the split's mechanism,
  cudaTriggerProgrammaticLaunchCompletion();  // main may co-launch now

  // ── Grid (batch, ngroups, head_tiles) ──
  int const seq = blockIdx.x;
  int const group_idx = blockIdx.y;
  int const head_tile = blockIdx.z;  // 0 for first cut (tiles==1)
  int const lane = threadIdx.x;
  int const warp = threadIdx.y;
  int const first_head = group_idx * HEADS_PER_GROUP + head_tile * HEADS_PER_CTA;

  // ── Per-slot setup (shared across the group's heads) ──
  auto const* __restrict__ sbi = reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  int64_t const cache_slot = sbi ? static_cast<int64_t>(sbi[seq]) : seq;
  if (cache_slot == params.pad_slot_id) return;

  // prev_k is on the critical path — must_checkpoint (and load_old_B's row extent) consume it
  // first, and there's no executed work before that consume to hide the load.  Issue it FIRST,
  // via __ldg (read-only path), so its latency overlaps the ring_start load instead of fully
  // stalling must_checkpoint on a cold dependent global load.
  auto const* __restrict__ prev_ptr = reinterpret_cast<int32_t const*>(params.prev_num_accepted);
  int const prev_k = __ldg(&prev_ptr[cache_slot]);
  auto const* __restrict__ ring_start_ptr = reinterpret_cast<int32_t const*>(params.ring_start);
  int const ring_start = __ldg(&ring_start_ptr[cache_slot]);

  int seq_len;
  int64_t outer;
  if constexpr (VARLEN) {
    auto const* __restrict__ cu = reinterpret_cast<int32_t const*>(params.cu_seqlens);
    int const bos = __ldg(&cu[seq]);
    int const eos = __ldg(&cu[seq + 1]);
    seq_len = eos - bos;
    if (seq_len <= 0) return;
    outer = (int64_t)bos;
  } else {
    seq_len = NPREDICTED;
    outer = (int64_t)seq;
  }
  bool const must_checkpoint = (prev_k + seq_len > MAX_WINDOW);
  // Ring: appends land at (ring_start + prev_k + i) % RING_BUFFER_LEN on BOTH
  // branches; a flush only advances ring_start (host-side, after the call).
  int const write_offset = prev_k;

  if (!must_checkpoint && warp >= 2 && warp < 4) {
    load_old_B<input_t, MAX_WINDOW, DSTATE>(smem, params, lane, cache_slot, ring_start, group_idx,
                                            prev_k);
  }

  // ── EXTERNAL PDL (conv1d → precompute): gated by ENABLE_PDL — conv1d produces B/C,
  // so wait before load_C/load_B.  PHASE 1 (conv1d-independent) runs HERE too, before
  // the wait, so its LDGs fill the conv1d wait window.  No-op without programmatic conv1d.
  if constexpr (ENABLE_PDL) {
    load_phase1_coeffs<dt_t, weight_t, matrixA_t, NPREDICTED, MAX_WINDOW, HEADS_PER_CTA, NUM_WARPS>(
        smem, params, warp, lane, first_head, seq_len, outer, seq, cache_slot, ring_start,
        must_checkpoint, prev_k);
    cudaGridDependencySynchronize();
  }

  // ── Load this group's C (W0-3 — the matmul warps; A-operand of both C·B and
  // C·old_B) + B (W0/1 only — the new-token N-operand) into swizzled smem (cp.async).
  // At NUM_WARPS>4 the extra warps W4+ run no matmul and never read C/B, so they skip
  // these loads (they earn their keep in PHASE 1 + PHASE 2 instead). ──
  if (warp < 4) {
    load_C<input_t, NPREDICTED, DSTATE>(smem, params, lane, group_idx, outer, seq_len);
  }
  if (warp < 2) {
    load_B<input_t, NPREDICTED, DSTATE>(smem, params, lane, group_idx, outer, seq_len);
  }

  // Standalone (!ENABLE_PDL): no conv1d wait to hide behind, so run PHASE 1 HERE to
  // overlap the B/C + old_B cp.async issued above instead of exposing its LDG latency.
  if constexpr (!ENABLE_PDL) {
    load_phase1_coeffs<dt_t, weight_t, matrixA_t, NPREDICTED, MAX_WINDOW, HEADS_PER_CTA, NUM_WARPS>(
        smem, params, warp, lane, first_head, seq_len, outer, seq, cache_slot, ring_start,
        must_checkpoint, prev_k);
  }

  // ── Drain ALL cp.async ONCE (C/B from load_C/load_B + old_B from load_old_B) BEFORE
  // any consumer — the helpers only commit, so the loads overlap in flight rather than
  // each draining its own group (which serialized B/C → old_B).  MUST precede
  // store_old_B (reads smem.B) AND the CB matmul (reads smem.B/C/old_B).  __syncwarp
  // publishes them within each warp (loads are redundant per-warp). ──
  __pipeline_wait_prior(0);
  __syncwarp();

  // ── old_B cache writeback (per-group, D-independent): the new B tokens become the
  // buffered "old" for the next step.  Reads the now-drained smem.B.  W0/1 hold smem.B;
  // the precompute owns this writeback (the main reads cb_scaled, never loads B).
  // store_old_B self-gates on head % HEADS_PER_GROUP — first_head qualifies. ──
  if (warp < 2) {
    store_old_B<input_t, NPREDICTED, DSTATE, HEADS_PER_GROUP>(smem, params, warp, lane, first_head,
                                                              group_idx, cache_slot, ring_start,
                                                              write_offset, seq_len);
  }

  // ── Raw matmul-1 — ONCE per group, by the FIRST 4 warps (no-write).  W0/1 → C·B (C5)
  // → smem.CB; W2/3 → C·old_B (C6, no-write) → smem.CB_old.  At NUM_WARPS>4 the extra
  // warps W4+ skip the matmul (the CB is per-group, computed once; the __syncthreads
  // below publishes it to them for PHASE 2).  Independent of PHASE 1 (different smem). ──
  if (warp < 2) {
    compute_cb_2warp<input_t, NPREDICTED, DSTATE>(smem, warp, lane);
  } else if (warp < 4 && !must_checkpoint) {
    compute_cb_old_2warp<input_t, NPREDICTED, MAX_WINDOW, DSTATE>(smem, warp - 2, lane);
  }

  // ── Publish PHASE-1 loads + the raw-CB stores cross-warp before PHASE 2 ──
  __syncthreads();

  // ── PHASE 2: warp-per-head, NO gmem loads.  Each loop iteration is one head:
  // bias+softplus (from smem) → warp scan → cumAdt; then store cumAdt_vec, the C7
  // cache tape, and scale_store_cb C5 / C6.  Every input is already in smem
  // (PHASE 1 + the scan), so the loop never stalls on a load.  NUM_ITER is
  // uniform, so every warp runs the same iteration count. ──
  auto* __restrict__ cb_gmem = reinterpret_cast<input_t*>(params.cb_scaled);   // CB_NEW_REGS/lane
  auto* __restrict__ cb_old_gmem = reinterpret_cast<input_t*>(params.cb_old);  // K_old/2 regs/lane
  auto* __restrict__ cumAdt_gmem = reinterpret_cast<float*>(params.cumAdt_vec);
  constexpr int CB_NEW_REGS = NPREDICTED_PAD_MMA_M / 2;         // element-major raw dump size (=8)
  constexpr int CB_OLD_REGS = SmemT::MAX_WINDOW_PAD_MMA_K / 2;  // element-major raw dump size
  // fragB (B-operand) STORE sizes = NUM_OUT_NTILES·(K/4) — the per-head gmem stride.  HALF of
  // CB_*_REGS at mtp≤8 (1 output N-tile); == CB_*_REGS at mtp>8 (2 t-halves, no shrink).
  constexpr int NUM_OUT_NTILES = SmemT::NPREDICTED_PAD_MMA_N / MMA_prop::N;
  constexpr int CB_NEW_REGS_B = NUM_OUT_NTILES * (NPREDICTED_PAD_MMA_M / 4);
  constexpr int CB_OLD_REGS_B = NUM_OUT_NTILES * (SmemT::MAX_WINDOW_PAD_MMA_K / 4);

  // Load the raw C·B / C·old_B fragA ONCE per warp (it's the same for all this
  // group's heads — the per-head decay scaling is applied below).  Element-major
  // (smem[e*32+lane]) → one conflict-free LDS per element.  Replaces the per-head
  // re-reads + the swizzled-store bank conflict.
  float raw_cb[CB_NEW_REGS];
#pragma unroll
  for (int e = 0; e < CB_NEW_REGS; ++e) raw_cb[e] = smem.CB[e * warpSize + lane];

  float raw_cb_old[CB_OLD_REGS];  // NO-WRITE only (loaded + used iff !must_checkpoint)

  if (!must_checkpoint) {
#pragma unroll
    for (int e = 0; e < CB_OLD_REGS; ++e) raw_cb_old[e] = smem.CB_old[e * warpSize + lane];
  }

  constexpr int NUM_ITER = (HEADS_PER_CTA + NUM_WARPS - 1) / NUM_WARPS;  // ceil, uniform
  int const h_base = warp * NUM_ITER;  // first head for this warp (contiguous block)
#pragma unroll
  for (int iter = 0; iter < NUM_ITER; ++iter) {
    int const h = h_base + iter;
    bool const has_head = (h < HEADS_PER_CTA);
    int const head = first_head + h;

    if (has_head) {
      // C1: bias + softplus on the pre-loaded raw dt (own-lane, in place).
      if (lane < seq_len) {
        float dv = smem.dt[h][lane] + smem.dt_bias[h];
        if constexpr (DT_SOFTPLUS) dv = thresholded_softplus(dv);
        smem.dt[h][lane] = dv;
      }
      // C2: cumsum scan (own-lane read + register shfl) → smem.cumAdt[h].
      compute_cumAdt_pw<NPREDICTED>(smem, h, lane, smem.A[h]);
      // Publish dt[h] (softplus) + cumAdt[h] cross-lane for scale_store_cb below.
      __syncwarp();

      // C5: scale the register-resident raw C·B by this head + STG → cb_scaled (fragB).
      auto* cb_gmem_head =
          cb_gmem + (int64_t)((seq * params.nheads + head) * warpSize * CB_NEW_REGS_B);
      scale_store_cb</*IS_OLD=*/false, CB_NEW_REGS, input_t>(smem, h, lane, seq_len, prev_k, raw_cb,
                                                             cb_gmem_head);
      // C6 (NO-WRITE): scale the register-resident raw C·old_B + STG → cb_old.
      if (!must_checkpoint) {
        auto* cb_old_head =
            cb_old_gmem + (int64_t)((seq * params.nheads + head) * warpSize * CB_OLD_REGS_B);
        scale_store_cb</*IS_OLD=*/true, CB_OLD_REGS, input_t>(smem, h, lane, seq_len, prev_k,
                                                              raw_cb_old, cb_old_head);
      }
    }
  }

  // Store pass: each warp stores its contiguous heads (h_base..h_base+NUM_ITER-1).
  // No __syncthreads needed: each warp reads only its own smem writes from the loop above.
  // Lane i covers element [i/NPREDICTED, i%NPREDICTED] within this warp's head block.
  auto* __restrict__ dt_cache_w = reinterpret_cast<float*>(params.dt_cache);
  constexpr int STORE_ELEMS = NUM_ITER * NPREDICTED;
#pragma unroll
  for (int i = lane; i < STORE_ELEMS; i += warpSize) {
    int const h_local = i / NPREDICTED;
    int const t = i % NPREDICTED;
    int const h = h_base + h_local;
    if (h >= HEADS_PER_CTA || t >= seq_len) continue;
    int const head = first_head + h;

    cumAdt_gmem[(int64_t)(seq * params.nheads + head) * NPREDICTED_PAD_MMA_M + t] =
        smem.cumAdt[h][t];
    int64_t const dt_w_base =
        cache_slot * params.dt_cache_stride_seq + (int64_t)head * params.dt_cache_stride_head;
    int dst = ring_start + write_offset + t;
    if (dst >= params.ring_buffer_len) dst -= params.ring_buffer_len;
    dt_cache_w[dt_w_base + dst] = smem.dt[h][t];
  }
}

}  // namespace flashinfer::mamba::checkpointing

#endif  // FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_PRECOMPUTE_CUH_
