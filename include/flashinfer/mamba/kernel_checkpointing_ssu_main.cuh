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
#ifndef FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_MAIN_CUH_
#define FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_MAIN_CUH_

// Brings ssu_checkpoint / ssu_nocheckpoint, CheckpointingSsuStorage,
// store_old_x, ENABLE_PDL — and (transitively) the shared load helpers in
// kernel_checkpointing_ssu_common.cuh (load_tile_async, load_state_*,
// the old_dt row loads, load_cb_fragA).
#include "kernel_checkpointing_ssu.cuh"

namespace flashinfer::mamba::checkpointing {

template <typename input_t, typename state_t, int NPREDICTED_, int MAX_WINDOW_, int D_PER_CTA,
          int DSTATE, int STATE_PIPE = 1, bool VARLEN = false>
struct CheckpointingSsuMainStorage {
  static constexpr int NPREDICTED = NPREDICTED_;
  static constexpr int MAX_WINDOW = MAX_WINDOW_;
  static constexpr int D_SMEM_COLS = next_multiple_of<SmemSwizzle<input_t>::ATOM_COLS>(D_PER_CTA);
  static constexpr int NPREDICTED_PAD_MMA_M = next_multiple_of<MMA_prop::M>(NPREDICTED);
  static constexpr int NPREDICTED_PAD_MMA_N = next_multiple_of<MMA_prop::N>(NPREDICTED);
  static constexpr int MAX_WINDOW_PAD_MMA_K = next_multiple_of<MMA_prop::K_SMALL>(MAX_WINDOW);
  static constexpr int NPREDICTED_SWIZZLE_R =
      next_multiple_of<SmemSwizzle<input_t>::ATOM_ROWS>(NPREDICTED);
  static constexpr int CB_ROW_STRIDE = SmemSwizzle<input_t>::ATOM_COLS;

  static constexpr int C_ELEMS = NPREDICTED_SWIZZLE_R * DSTATE;
  static constexpr int OLD_B_ELEMS = MAX_WINDOW_PAD_MMA_K * DSTATE;
  static constexpr int Z_ELEMS = NPREDICTED_SWIZZLE_R * D_SMEM_COLS;
  static constexpr int OLD_X_ELEMS = MAX_WINDOW_PAD_MMA_K * D_SMEM_COLS;
  static constexpr int X_ELEMS = NPREDICTED_PAD_MMA_M * D_SMEM_COLS;
  static constexpr int STATE_ELEMS = D_PER_CTA * DSTATE;
  static constexpr int CUMADT_ELEMS = NPREDICTED_PAD_MMA_M;
  // CB fragB-native blocks (operand swap): one warp of lane-major Packs (REGS_B =
  // NUM_OUT_NTILES·K/4 — HALF the old fragA store at mtp≤8).  Staged here so the output MMA's
  // B-operand loads from smem (LDS) instead of a just-in-time LDG.  Precompute writes this exact
  // layout (scale_store_cb).
  static constexpr int NUM_OUT_NTILES = NPREDICTED_PAD_MMA_N / MMA_prop::N;  // 1 (mtp≤8) or 2
  static constexpr int CB_NEW_REGS_B = NUM_OUT_NTILES * (NPREDICTED_PAD_MMA_M / 4);
  static constexpr int CB_OLD_REGS_B = NUM_OUT_NTILES * (MAX_WINDOW_PAD_MMA_K / 4);
  static constexpr int CB_NEW_ELEMS = 32 * CB_NEW_REGS_B;
  static constexpr int CB_OLD_ELEMS = 32 * CB_OLD_REGS_B;
  alignas(16) input_t C[STATE_PIPE * C_ELEMS];          // matmul-3 A-operand (conv1d output)
  alignas(16) input_t old_B[STATE_PIPE * OLD_B_ELEMS];  // replay B-operand (per-group cache)
  alignas(16) input_t z[STATE_PIPE * Z_ELEMS];
  alignas(16) input_t old_x[STATE_PIPE * OLD_X_ELEMS];
  float old_dt[STATE_PIPE * MAX_WINDOW];
  float cumAdt[STATE_PIPE * CUMADT_ELEMS];
  alignas(16) input_t x[STATE_PIPE * X_ELEMS];
  alignas(
      16) input_t cb_new[STATE_PIPE * CB_NEW_ELEMS];  // matmul-4 A-operand (precompute cb_scaled)
  alignas(16) input_t cb_old[STATE_PIPE * CB_OLD_ELEMS];  // no-write CB_old A-operand (precompute)
  alignas(16) state_t state[STATE_PIPE * STATE_ELEMS];
  // Work-unit metadata ring: one warp-width window (slot = unit % META_RING), refilled every
  // META_RING−STAGES units by fill_meta (see the kernel).  512 B — sized to stay under the
  // 4-blocks/SM smem cliff (57,728 B/block ×4 leaves 640 B/block on the 228 KB SM): tile is
  // re-derived at load (blockIdx.x + u·stride) and cache_slot is stored as int32 (cache slots
  // ≪ 2^31; pads canonicalized to -1) so the window fits with margin.
  static constexpr int META_RING = 32;
  static_assert((META_RING & (META_RING - 1)) == 0, "META_RING must be a power of two");
  int32_t meta_pnat[META_RING];
  int32_t meta_ring_start[META_RING];
  float meta_D[META_RING];
  float meta_A[META_RING];  // A[head] decay rate — the in-register cumAdt scan's multiplier
  int32_t meta_cache_slot[META_RING];
  // Varlen row geometry, batched into the ring like pnat/ring_start so is_valid / mc_of /
  // derive_head never touch cu_seqlens per unit (up to ~12 __ldg/unit across their re-derivation
  // sites — measured +5-7% at b=1024 before this).  bos and seq_len are PACKED into one int32
  // ((bos << SEQLEN_BITS) | seq_len): one STS/LDS on the meta path instead of two, half the
  // ring bytes; unpack is 2 ALU into the register queue.  seq_len fits SEQLEN_BITS by the
  // static_assert; bos < 2^23 tokens is asserted by the Python wrapper.  Dense: one 4 B dummy.
  static constexpr int SEQLEN_BITS = 8;
  static_assert(MAX_WINDOW <= (1 << SEQLEN_BITS) - 1, "seq_len must fit SEQLEN_BITS");
  int32_t meta_cu[VARLEN ? META_RING : 1];  // (cu[seq] << 8) | (cu[seq+1] - cu[seq])
};

template <typename input_t, typename state_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int D_PER_CTA, int DSTATE, int NUM_WARPS, bool IS_FIRST, typename SmemT>
__device__ __forceinline__ void load_head(SmemT& smem, CheckpointingSsuParams const& params,
                                          int lane, int warp, int d_tile, int head, int group_idx,
                                          int64_t cache_slot, int ring_start, int seq,
                                          int64_t outer, int seq_len, bool must_checkpoint,
                                          int prev_k, int tile_buf) {
  using namespace cute;
  int const d_tile_off = d_tile * D_PER_CTA;

  auto const* __restrict__ z_ptr = reinterpret_cast<input_t const*>(params.z);
  auto const* __restrict__ old_x_ptr = reinterpret_cast<input_t const*>(params.x_cache);
  auto const* __restrict__ old_B_ptr = reinterpret_cast<input_t const*>(params.B_cache);

  // Slot-distance strides (*_stride_seq) stay 64-bit — they can exceed 2^31
  // in production layouts; head/group strides are per-slot-internal.
  int64_t const ox_base = cache_slot * params.x_cache_stride_seq +
                          (int64_t)head * params.x_cache_stride_head + d_tile_off;
  int64_t const oB_base =
      cache_slot * params.B_cache_stride_seq + (int64_t)group_idx * params.B_cache_stride_group;

  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  using ZShape = cute::Shape<cute::Int<SmemT::NPREDICTED_SWIZZLE_R>, cute::Int<D_PER_CTA>>;
  using OldBShape = cute::Shape<cute::Int<MAX_WINDOW_PAD_MMA_K>, cute::Int<DSTATE>>;
  using OxShape = cute::Shape<cute::Int<MAX_WINDOW_PAD_MMA_K>, cute::Int<D_PER_CTA>>;

  // Per-tile smem slots: offset by tile_buf into the STATE_PIPE-strided ring arrays.
  auto* old_x_slot = smem.old_x + tile_buf * SmemT::OLD_X_ELEMS;
  float* old_dt_slot = smem.old_dt + tile_buf * MAX_WINDOW;
  auto* z_slot = smem.z + tile_buf * SmemT::Z_ELEMS;
  auto* old_B_slot = smem.old_B + tile_buf * SmemT::OLD_B_ELEMS;

  constexpr int SWIZZLE_ROWS = SmemSwizzle<input_t>::ATOM_ROWS;
  using OxHalfShape = cute::Shape<cute::Int<SWIZZLE_ROWS>, cute::Int<D_PER_CTA>>;
  int const ring_start_hi = ring_start + SWIZZLE_ROWS >= params.ring_buffer_len
                                ? ring_start + SWIZZLE_ROWS - params.ring_buffer_len
                                : ring_start + SWIZZLE_ROWS;
  if (warp == 0)
    load_ring_tile_async<OxHalfShape, /*COUNT=*/SWIZZLE_ROWS>(
        old_x_slot, old_x_ptr + ox_base, (int)params.x_cache_stride_pos, lane, ring_start,
        params.ring_buffer_len,
        /*count_rt=*/prev_k < SWIZZLE_ROWS ? prev_k : SWIZZLE_ROWS);
  if constexpr (MAX_WINDOW_PAD_MMA_K > SWIZZLE_ROWS)
    if (warp == 1)
      load_ring_tile_async<OxHalfShape, /*COUNT=*/SWIZZLE_ROWS>(
          old_x_slot + SWIZZLE_ROWS * SmemT::D_SMEM_COLS, old_x_ptr + ox_base,
          (int)params.x_cache_stride_pos, lane, ring_start_hi, params.ring_buffer_len,
          /*count_rt=*/prev_k - SWIZZLE_ROWS);
  if (must_checkpoint) {
    if constexpr (IS_FIRST)
      if (warp == 1)
        load_ring_tile_async<OldBShape, MAX_WINDOW>(old_B_slot, old_B_ptr + oB_base,
                                                    (int)params.B_cache_stride_pos, lane,
                                                    ring_start, params.ring_buffer_len);
  }
  if ((must_checkpoint || prev_k > 0) && warp == 2 && lane < MAX_WINDOW) {
    // dt rows from the RING — previous-step rows only, so PDL-safe on the
    // pre-gdc side.  BOTH paths consume them: the replay warp-scans the old
    // decay from these rows in registers (dB coeffs), the no-write output
    // scans them for β — nothing the precompute writes this step is read
    // before the output phase (see warp_scan_old_cumAdt).
    auto const* __restrict__ dtc_ptr = reinterpret_cast<float const*>(params.dt_cache);
    int64_t const dt_base = cache_slot * params.dt_cache_stride_seq +
                            (int64_t)(head * (int)params.dt_cache_stride_head);
    int ring_row = ring_start + lane;
    if (ring_row >= params.ring_buffer_len) ring_row -= params.ring_buffer_len;
    __pipeline_memcpy_async(&old_dt_slot[lane], &dtc_ptr[dt_base + ring_row], sizeof(float));
  }
  if (warp == 3 && z_ptr) {
    int64_t const z_base = outer * params.z_stride_seq + (int64_t)(head * DIM + d_tile_off);
    load_tile_async<ZShape, NPREDICTED>(z_slot, z_ptr + z_base, params.z_stride_token, lane,
                                        seq_len);
  }
  // NO commit/wait/syncwarp — process_head drains (see header).
}

template <typename state_t, int DIM, int D_PER_CTA, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void prefetch_state(SmemT& smem, CheckpointingSsuParams const& params,
                                               int lane, int warp, int d_tile, int head,
                                               int64_t cache_slot, int state_buf) {
  int const d_tile_off = d_tile * D_PER_CTA;
  auto const* __restrict__ state_ptr = reinterpret_cast<state_t const*>(params.state);
  int64_t const state_base =
      cache_slot * params.state_stride_seq + (int64_t)(head * DIM * DSTATE + d_tile_off * DSTATE);
  if constexpr (DIM == D_PER_CTA) {
    load_state_per_warp<state_t, D_PER_CTA, DSTATE, NUM_WARPS>(smem, state_ptr, state_base, warp,
                                                               lane, state_buf);
  } else {
    int const tid = warp * warpSize + lane;
    load_state_cta<state_t, D_PER_CTA, DSTATE, NUM_WARPS>(smem, state_ptr, state_base, tid,
                                                          state_buf);
  }
}

template <typename input_t, int NPREDICTED, int DIM, int D_PER_CTA, int DSTATE, bool IS_FIRST,
          typename SmemT>
__device__ __forceinline__ void load_x(SmemT& smem, CheckpointingSsuParams const& params, int lane,
                                       int warp, int d_tile, int head, int group_idx, int64_t outer,
                                       int seq_len, int tile_buf) {
  int const d_tile_off = d_tile * D_PER_CTA;
  auto const* __restrict__ C_ptr = reinterpret_cast<input_t const*>(params.C);
  auto const* __restrict__ x_ptr = reinterpret_cast<input_t const*>(params.x);
  int64_t const C_base = outer * params.C_stride_seq + (int64_t)(group_idx * DSTATE);
  int64_t const x_base = outer * params.x_stride_seq + (int64_t)(head * DIM + d_tile_off);
  using CShape = cute::Shape<cute::Int<SmemT::NPREDICTED_SWIZZLE_R>, cute::Int<DSTATE>>;
  using XShape = cute::Shape<cute::Int<SmemT::NPREDICTED_PAD_MMA_M>, cute::Int<D_PER_CTA>>;
  // C → slot, on W1.
  auto* C_slot = smem.C + tile_buf * SmemT::C_ELEMS;
  if constexpr (IS_FIRST)
    if (warp == 1)
      load_tile_async<CShape, NPREDICTED>(C_slot, C_ptr + C_base, params.C_stride_token, lane,
                                          seq_len);
  // x → slot, single-warp on W2 (W0-1 carry old_x, W3 carries z).
  auto* x_slot = smem.x + tile_buf * SmemT::X_ELEMS;
  if (warp == 2)
    load_tile_async<XShape, NPREDICTED>(x_slot, x_ptr + x_base, params.x_stride_token, lane,
                                        seq_len);
}

template <typename SmemT>
__device__ __forceinline__ void load_cumAdt(SmemT& smem, CheckpointingSsuParams const& params,
                                            int lane, int seq, int first_head, int seq_len,
                                            int tile_buf) {
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  auto const* __restrict__ cumAdt_ptr = reinterpret_cast<float const*>(params.cumAdt_vec);
  float* cumAdt_slot = smem.cumAdt + tile_buf * SmemT::CUMADT_ELEMS;
  if (lane < seq_len)
    cumAdt_slot[lane] =
        cumAdt_ptr[(int64_t)(seq * params.nheads + first_head) * NPREDICTED_PAD_MMA_M + lane];
}

template <typename SmemT>
__device__ __forceinline__ void load_cumAdt_async(SmemT& smem, CheckpointingSsuParams const& params,
                                                  int lane, int seq, int first_head, int seq_len,
                                                  int tile_buf) {
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  auto const* __restrict__ cumAdt_ptr = reinterpret_cast<float const*>(params.cumAdt_vec);
  float* cumAdt_slot = smem.cumAdt + tile_buf * SmemT::CUMADT_ELEMS;
  if (lane < seq_len)
    __pipeline_memcpy_async(
        &cumAdt_slot[lane],
        &cumAdt_ptr[(int64_t)(seq * params.nheads + first_head) * NPREDICTED_PAD_MMA_M + lane],
        sizeof(float));
}

template <typename input_t, typename SmemT>
__device__ __forceinline__ void load_cb_async(SmemT& smem, CheckpointingSsuParams const& params,
                                              int lane, int warp, int seq, int head,
                                              bool must_checkpoint, int tile_buf) {
  // fragB-native (operand swap): REGS_B = NUM_OUT_NTILES·K/4 regs/lane (precompute scale_store_cb).
  constexpr int CB_NEW_REGS_B = SmemT::CB_NEW_REGS_B;
  constexpr int CB_OLD_REGS_B = SmemT::CB_OLD_REGS_B;
  // Whole flattened offset stays 32-bit (batch·nheads·32·REGS_B ≪ 2³¹); widen only at the ptr add.
  auto const* __restrict__ cb_new_g = reinterpret_cast<input_t const*>(params.cb_scaled) +
                                      (int64_t)((seq * params.nheads + head) * 32 * CB_NEW_REGS_B);
  input_t* cb_new_s = smem.cb_new + tile_buf * SmemT::CB_NEW_ELEMS;
  __pipeline_memcpy_async(cb_new_s + lane * CB_NEW_REGS_B, cb_new_g + lane * CB_NEW_REGS_B,
                          CB_NEW_REGS_B * sizeof(input_t));
  if (!must_checkpoint) {  // no-write path also needs CB_old @ old_x
    auto const* __restrict__ cb_old_g =
        reinterpret_cast<input_t const*>(params.cb_old) +
        (int64_t)((seq * params.nheads + head) * 32 * CB_OLD_REGS_B);
    input_t* cb_old_s = smem.cb_old + tile_buf * SmemT::CB_OLD_ELEMS;
    __pipeline_memcpy_async(cb_old_s + lane * CB_OLD_REGS_B, cb_old_g + lane * CB_OLD_REGS_B,
                            CB_OLD_REGS_B * sizeof(input_t));
  }
}

// Old-token decay recomputed in registers from the dt ring (previous-step data,
// PDL-safe): cumAdt[k] = A · Σ_{j≤k} dt[j].  Same Hillis-Steele shfl scan and the
// same a·scan arithmetic order as the precompute's cb_old-scaling scan, so both
// kernels derive bit-identical decays from the ring.  Each warp scans redundantly
// (~4 shfl+add, no smem) — this removes the replay's only dependency on this-step
// precompute output, so tile 0 replays under G_pre0 alone with the post-gdc CB
// round trip still in flight (the racy pre-fix overlap, restored legally).
template <int MAX_WINDOW, typename SmemT>
__device__ __forceinline__ float warp_scan_old_cumAdt(SmemT const& smem, float A_val, int lane,
                                                      int prev_k, int tile_buf) {
  float const* old_dt_slot = smem.old_dt + tile_buf * MAX_WINDOW;
  float scan = (lane < prev_k) ? old_dt_slot[lane] : 0.f;  // prev_k <= MAX_WINDOW bounds the LDS
#pragma unroll
  for (int off = 1; off < MAX_WINDOW; off <<= 1) {
    float const up = __shfl_up_sync(constants::MASK_ALL_LANES, scan, off);
    if (lane >= off) scan += up;
  }
  return A_val * scan;
}

template <typename input_t, typename state_t, int D_PER_CTA, int DSTATE, typename SmemT,
          typename TiledMma, typename ThrMma, typename... FragY>
__device__ __forceinline__ void add_init_out_main(SmemT const& smem, TiledMma const& tiled_mma,
                                                  ThrMma const& thr_mma, int tid, int state_buf,
                                                  FragY&... frag_y) {
  using namespace cute;
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int K_TILE = cute::tile_size<2>(TiledMma{});
  constexpr int NUM_K_TILES = DSTATE / K_TILE;
  constexpr int M_TILE = cute::tile_size<0>(TiledMma{});  // 16·NUM_WARPS = D_PER_CTA
  static_assert(sizeof(state_t) != 1, "8-bit state goes through the dedicated 8-bit kernel");

  using AView = std::conditional_t<sizeof(state_t) == 2, MMA_prop::operand_t, state_t>;
  // A = state [D_PER_CTA, DSTATE] (dstate contiguous = K-major → x4 non-transpose LDSM),
  // slot-indexed.
  auto const layout_state = make_swizzled_layout_rc<AView, D_PER_CTA, DSTATE>();
  Tensor smem_state = make_tensor(
      make_smem_ptr(reinterpret_cast<AView const*>(smem.state + state_buf * SmemT::STATE_ELEMS)),
      layout_state);
  Tensor smem_state_ktiled =
      local_tile(smem_state, make_tile(Int<M_TILE>{}, Int<K_TILE>{}), make_coord(_0{}, _));
  // B = C [NPRED_pad, DSTATE] (dstate contiguous → x2 non-transpose LDSM), slot-indexed.
  auto const layout_C =
      make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, DSTATE, SmemT::NPREDICTED>();
  Tensor smem_C = make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(
                                  smem.C + state_buf * SmemT::C_ELEMS)),
                              layout_C);
  pipelined_kloop_gemm<3, NUM_K_TILES, state_t, input_t, MMA_prop::operand_t>(
      tiled_mma, thr_mma, tid, smem_state_ktiled, smem_C, frag_y...);
}

template <typename input_t, typename state_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS, int NUM_WARPS, bool MUST_CHECKPOINT,
          typename FragCBNew, typename FragCBOld, typename SmemT>
__device__ __forceinline__ void output_head_2k(SmemT& smem, CheckpointingSsuParams const& params,
                                               int lane, int warp, int d_tile, int head,
                                               int64_t cache_slot, int ring_start, int prev_k,
                                               int64_t out_seq_base, int write_offset, int seq_len,
                                               float D_val, float beta_extra, int tile_buf,
                                               FragCBNew const& frag_CB_new,
                                               FragCBOld const& frag_CB_old) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "output_head_2k requires 2-byte input type");
  static_assert(sizeof(state_t) == 2 || sizeof(state_t) == 4,
                "operand-swap output supports 2-byte (LDSM) or 4-byte (LDS.64) state");

  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int NPREDICTED_PAD_MMA_N = SmemT::NPREDICTED_PAD_MMA_N;
  constexpr int NPREDICTED_SWIZZLE_R = SmemT::NPREDICTED_SWIZZLE_R;
  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;
  int const tid = warp * warpSize + lane;

  // ── TiledMMA (OPERAND SWAP): warps split M = DIM (16 rows/warp); N = t (NPRED); output [d, t].
  // NUM_M_WARPS = D_PER_CTA/16 warps run the output MMA (== NUM_WARPS at d_split=1; fewer at
  // d_split=2, where the surplus warps still do store_state/store_old_x but skip the output). ──
  constexpr int NUM_M_WARPS = D_PER_CTA / MMA_prop::M;
  static_assert(D_PER_CTA % MMA_prop::M == 0, "D_PER_CTA must be a multiple of MMA::M (16)");
  bool const m_active = (warp < NUM_M_WARPS);
  auto tiled_mma = make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{},
                                  Layout<Shape<Int<NUM_M_WARPS>, _1>>{});
  auto thr_mma = tiled_mma.get_slice(tid);
  constexpr int N_TILE = cute::tile_size<1>(decltype(tiled_mma){});  // 8 = one t N-tile
  constexpr int M_TILE = cute::tile_size<0>(decltype(tiled_mma){});  // = D_PER_CTA
  constexpr int NUM_N_TILES = NPREDICTED_PAD_MMA_N / N_TILE;         // 1 (mtp≤8) or 2 (mtp>8)
  static_assert(NUM_N_TILES == 1 || NUM_N_TILES == 2, "output_head_2k: NUM_N_TILES must be 1 or 2");

  // ── smem transpose views [d, token]: x is OUT.2's A-operand source + the D·x read; z the z-gate
  // read.  x/old_x are stored [token,d] (d contiguous) → transpose-LDSM as the A operand.  z is
  // aliased-stored; its physical NPREDICTED_SWIZZLE_R rows == make_swizzled_layout_rc for token<8.
  // ──
  auto const* x_base = smem.x + tile_buf * NPREDICTED_PAD_MMA_M * D_SMEM_COLS;
  auto layout_x_trans =
      make_swizzled_layout_rc_transpose<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS>();
  Tensor smem_x_trans = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(x_base)), layout_x_trans);
  auto const* z_base = smem.z + tile_buf * NPREDICTED_SWIZZLE_R * D_SMEM_COLS;
  auto layout_z_trans =
      make_swizzled_layout_rc_transpose<input_t, NPREDICTED_SWIZZLE_R, D_SMEM_COLS>();
  Tensor smem_z_trans = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(z_base)), layout_z_trans);

  // ── x A-operand transpose LDSM (SM75_U16x8_LDSM_T: M=d 16 rows/warp, K=j 16) ──
  auto s2r_A_x = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_A_x = s2r_A_x.get_slice(tid);

  // ── Gmem output base (token 0, d = head·DIM + d_tile·D_PER_CTA); M=d stride 1, N=t stride token
  auto* __restrict__ output_ptr = reinterpret_cast<input_t*>(params.output);
  int64_t const out_base = out_seq_base + (int64_t)head * DIM + (int64_t)d_tile * D_PER_CTA;

  // ── Column (t) predicate identity; token = N so predicate on get<1> ──
  auto id_tile = make_identity_tensor(make_shape(Int<M_TILE>{}, Int<N_TILE>{}));
  auto id_part = thr_mma.partition_C(id_tile);
  float const* cumAdt_slot = smem.cumAdt + tile_buf * SmemT::CUMADT_ELEMS;

  if constexpr (MUST_CHECKPOINT) {
    // ── Write path: OUT.1 + store_state + (decay + CB@x + D·x + z-gate + scatter store) ──
    constexpr bool kSkipSmemToGmemState = (PHILOX_ROUNDS > 0) && std::is_same_v<state_t, __half>;

    auto epilogue = [&](auto& frag_y, int n) {
      // decay: frag_y[d,t] *= exp(cumAdt[t])  (t = N; broadcast over d = M) ──
      auto decay_bcast = make_tensor(
          make_smem_ptr(cumAdt_slot + n * N_TILE),
          make_layout(make_shape(Int<M_TILE>{}, Int<N_TILE>{}), make_stride(_0{}, _1{})));
      auto decay_part = thr_mma.partition_C(decay_bcast);
#pragma unroll
      for (int i = 0; i < size(frag_y); ++i) frag_y(i) *= __expf(decay_part(i));
      add_cbx_swapped<MMA_prop::operand_t, M_TILE, NPREDICTED_PAD_MMA_M>(
          frag_y, frag_CB_new[n], smem_x_trans, s2r_A_x, s2r_thr_A_x, thr_mma, tiled_mma, n);
      add_D_skip_swapped<input_t, M_TILE, N_TILE>(frag_y, smem_x_trans, thr_mma, D_val, n);
      compute_z_gating_swapped<input_t, M_TILE, N_TILE>(frag_y, smem_z_trans, thr_mma, params.z, n);
      auto gOut = make_tensor(
          make_gmem_ptr(output_ptr + out_base + (int64_t)(n * N_TILE) * params.out_stride_token),
          make_layout(make_shape(Int<M_TILE>{}, Int<N_TILE>{}),
                      make_stride(_1{}, params.out_stride_token)));
      auto gOut_part = thr_mma.partition_C(gOut);
#pragma unroll
      for (int i = 0; i < size(frag_y); ++i) {
        int const t = n * N_TILE + get<1>(id_part(i));
        if (t < seq_len) gOut_part(i) = static_cast<input_t>(frag_y(i));
      }
    };

    if constexpr (NUM_N_TILES == 2) {
      Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
      Tensor frag_y_1 = thr_mma.partition_fragment_C(id_tile);
      if (m_active)  // only the D_PER_CTA/16 M-warps run the output MMA
        add_init_out_main<input_t, state_t, D_PER_CTA, DSTATE>(smem, tiled_mma, thr_mma, tid,
                                                               tile_buf, frag_y_0, frag_y_1);
      if constexpr (!kSkipSmemToGmemState)  // store_state is cooperative over ALL warps
        store_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(smem, params, warp, lane, d_tile,
                                                                head, cache_slot, tile_buf);
      if (m_active) {
        epilogue(frag_y_0, 0);
        epilogue(frag_y_1, 1);
      }
    } else {
      Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
      if (m_active)  // only the D_PER_CTA/16 M-warps run the output MMA
        add_init_out_main<input_t, state_t, D_PER_CTA, DSTATE>(smem, tiled_mma, thr_mma, tid,
                                                               tile_buf, frag_y_0);
      if constexpr (!kSkipSmemToGmemState)  // store_state is cooperative over ALL warps
        store_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(smem, params, warp, lane, d_tile,
                                                                head, cache_slot, tile_buf);
      if (m_active) epilogue(frag_y_0, 0);
    }
  } else {
    // ── No-write path: OUT.1 + (β·decay + CB@x + CB_old@old_x + D·x + z-gate + scatter store) ──
    using MmaAtomOld = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG,
                                          MMA_prop::AtomK16, MMA_prop::AtomK8>;
    using LdsmAOld = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x8_LDSM_T,
                                        SM75_U16x4_LDSM_T>;
    auto tiled_mma_old =
        make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomOld>>{}, Layout<Shape<Int<NUM_M_WARPS>, _1>>{});
    auto thr_mma_old = tiled_mma_old.get_slice(tid);

    auto const* old_x_base = smem.old_x + tile_buf * MAX_WINDOW_PAD_MMA_K * D_SMEM_COLS;
    auto layout_old_x_trans =
        make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS>();
    Tensor smem_old_x_trans =
        make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(old_x_base)),
                    layout_old_x_trans);
    // old_x A-operand transpose LDSM (K=i = MAX_WINDOW_PAD_MMA_K ∈ {8,16}).
    auto s2r_A_old_x = make_tiled_copy_A(Copy_Atom<LdsmAOld, MMA_prop::operand_t>{}, tiled_mma_old);
    auto s2r_thr_A_old_x = s2r_A_old_x.get_slice(tid);

    // beta_extra (= exp of the old_cumAdt tail) arrives as a param, hoisted by the caller ahead
    // of the CB fragment loads — see compute_output_and_store.
    auto epilogue = [&](auto& frag_y, int n) {
      auto decay_bcast = make_tensor(
          make_smem_ptr(cumAdt_slot + n * N_TILE),
          make_layout(make_shape(Int<M_TILE>{}, Int<N_TILE>{}), make_stride(_0{}, _1{})));
      auto decay_part = thr_mma.partition_C(decay_bcast);
#pragma unroll
      for (int i = 0; i < size(frag_y); ++i) frag_y(i) *= beta_extra * __expf(decay_part(i));
      add_cbx_swapped<MMA_prop::operand_t, M_TILE, NPREDICTED_PAD_MMA_M>(
          frag_y, frag_CB_new[n], smem_x_trans, s2r_A_x, s2r_thr_A_x, thr_mma, tiled_mma, n);
      add_cbx_swapped<MMA_prop::operand_t, M_TILE, MAX_WINDOW_PAD_MMA_K>(
          frag_y, frag_CB_old[n], smem_old_x_trans, s2r_A_old_x, s2r_thr_A_old_x, thr_mma_old,
          tiled_mma_old, n);
      add_D_skip_swapped<input_t, M_TILE, N_TILE>(frag_y, smem_x_trans, thr_mma, D_val, n);
      compute_z_gating_swapped<input_t, M_TILE, N_TILE>(frag_y, smem_z_trans, thr_mma, params.z, n);
      auto gOut = make_tensor(
          make_gmem_ptr(output_ptr + out_base + (int64_t)(n * N_TILE) * params.out_stride_token),
          make_layout(make_shape(Int<M_TILE>{}, Int<N_TILE>{}),
                      make_stride(_1{}, params.out_stride_token)));
      auto gOut_part = thr_mma.partition_C(gOut);
#pragma unroll
      for (int i = 0; i < size(frag_y); ++i) {
        int const t = n * N_TILE + get<1>(id_part(i));
        if (t < seq_len) gOut_part(i) = static_cast<input_t>(frag_y(i));
      }
    };

    if constexpr (NUM_N_TILES == 2) {
      Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
      Tensor frag_y_1 = thr_mma.partition_fragment_C(id_tile);
      if (m_active) {  // only the D_PER_CTA/16 M-warps run the output MMA + epilogue
        add_init_out_main<input_t, state_t, D_PER_CTA, DSTATE>(smem, tiled_mma, thr_mma, tid,
                                                               tile_buf, frag_y_0, frag_y_1);
        epilogue(frag_y_0, 0);
        epilogue(frag_y_1, 1);
      }
    } else {
      Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
      if (m_active) {  // only the D_PER_CTA/16 M-warps run the output MMA + epilogue
        add_init_out_main<input_t, state_t, D_PER_CTA, DSTATE>(smem, tiled_mma, thr_mma, tid,
                                                               tile_buf, frag_y_0);
        epilogue(frag_y_0, 0);
      }
    }
  }

  // store_old_x: 128-thread cooperative copy (16×8 layout), only first 4 warps participate.
  if (warp < 4)
    store_old_x<input_t, NPREDICTED, DIM, D_PER_CTA>(smem, params, warp, lane, d_tile, head,
                                                     cache_slot, ring_start, write_offset, seq_len,
                                                     tile_buf);
}

template <typename input_t, typename state_t, int DIM, int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS,
          int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void replay_state_mma_main(
    SmemT& smem, CheckpointingSsuParams const& params, int warp, int lane, int prev_k, int d_tile,
    int64_t state_ptr_offset, state_t* state_w_base, int64_t rand_seed, float A_val,
    bool must_checkpoint, int tile_buf = 0) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "replay_state_mma requires 2-byte input type");
  static_assert(D_PER_CTA % 16 == 0, "D_PER_CTA must be divisible by 16 (m16n8 atom)");
  static_assert(D_PER_CTA >= 16, "D_PER_CTA must be at least 16");

  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;  // 8 or 16
  int const tid = warp * warpSize + lane;

  // Atom K matches the cache-window tile (MAX_WINDOW_PAD_MMA_K).
  //   K == MMA_prop::K_BIG   (16) → m16n8k16 + x4/x2 ldmatrix.trans
  //   K == MMA_prop::K_SMALL (8)  → m16n8k8  + x2/x1 ldmatrix.trans
  using MmaAtomType = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, MMA_prop::AtomK16,
                                         MMA_prop::AtomK8>;
  using LdsmA = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x8_LDSM_T,
                                   SM75_U16x4_LDSM_T>;
  using LdsmB = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x4_LDSM_T,
                                   SM75_U16x2_LDSM_T>;

  // Warp layout (M_WARPS, 4): always 4 warps along N=DSTATE; M_WARPS = NUM_WARPS/4 warps
  // split M=D_PER_CTA (each covers D_PER_CTA/M_WARPS/16 m-atoms + its own M-slice of the
  // A operand, cutting the redundant old_x LDSM by M_WARPS).  NUM_WARPS=4 → (1,4) = the
  // original _1x4 (byte-identical); 8 → (2,4); 16 → (4,4).
  // TODO(int8-8warp): the int8/fp8 amax (kernel_checkpointing_ssu_8bit.cuh) is warp-local
  // ONLY with full-N-per-warp (_W×1); this N-split layout would need a cross-warp amax
  // reduce.  bf16/fp16 are unaffected (deterministic state / order-free SR).
  constexpr int M_WARPS = NUM_WARPS / 4;
  static_assert(NUM_WARPS % 4 == 0, "replay_state_mma needs NUM_WARPS a multiple of 4");
  auto tiled_mma =
      make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomType>>{}, Layout<Shape<Int<M_WARPS>, _4>>{});
  auto thr_mma = tiled_mma.get_slice(tid);

  // Per-pass output tile is (D_PER_CTA/M_WARPS, N_PER_PASS).  N_PER_PASS = 4 warps × n8 = 32.
  constexpr int N_PER_PASS = 4 * MMA_prop::N;
  static_assert(DSTATE % N_PER_PASS == 0,
                "DSTATE must be divisible by 4 * MMA_prop::N for the (M_WARPS,4) warp layout");
  constexpr int NUM_N_PASSES = DSTATE / N_PER_PASS;

  // tile_buf selects the pipeline slot for all per-head smem arrays (old_dt, old_x).
  constexpr int MAX_WINDOW = SmemT::MAX_WINDOW;
  float const* old_dt_slot = smem.old_dt + tile_buf * MAX_WINDOW;
  float const cumAdt_lane = warp_scan_old_cumAdt<MAX_WINDOW>(smem, A_val, lane, prev_k, tile_buf);
  float total_cumAdt =
      (prev_k > 0) ? __shfl_sync(constants::MASK_ALL_LANES, cumAdt_lane, prev_k - 1) : 0.f;
  float total_decay = (prev_k > 0) ? __expf(total_cumAdt) : 1.f;

  // ── A operand: old_x [MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS] Swizzle<3,3,3>, transposed
  // view [M=D_SMEM_COLS, K=MAX_WINDOW_PAD_MMA_K].  D_SMEM_COLS may be padded above
  // D_PER_CTA when D_PER_CTA < swizzle atom; local_tile to D_PER_CTA
  // restricts the LDSM to the valid sub-tile.  Each warp loads the FULL M (4×
  // redundant across warps).  See header comment for traffic accounting. ──
  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;
  auto layout_A_full =
      make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS>();
  auto const* old_x_slot = smem.old_x + tile_buf * MAX_WINDOW_PAD_MMA_K * D_SMEM_COLS;
  Tensor smem_A_full = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(old_x_slot)), layout_A_full);
  Tensor smem_A = local_tile(smem_A_full, make_shape(Int<D_PER_CTA>{}, Int<MAX_WINDOW_PAD_MMA_K>{}),
                             make_coord(_0{}, _0{}));

  auto s2r_A = make_tiled_copy_A(Copy_Atom<LdsmA, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  Tensor smem_A_s2r = s2r_thr_A.partition_S(smem_A);
  Tensor frag_A = thr_mma.partition_fragment_A(make_tensor(
      (MMA_prop::operand_t*)0x0, make_shape(Int<D_PER_CTA>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
  Tensor frag_A_view = s2r_thr_A.retile_D(frag_A);

  cute::copy(s2r_A, smem_A_s2r, frag_A_view);
  // old_x is input_t == MMA_prop::operand_t (bf16) — no conversion needed.

  // ── B operand: old_B [MAX_WINDOW_PAD_MMA_K, DSTATE] swizzled, transposed view
  // [N=DSTATE, K=MAX_WINDOW_PAD_MMA_K].  Per pass loads N_PER_PASS=32 cols across
  // 4 warps; partition_S splits — each warp gets its disjoint 8-col slice. ──
  auto layout_B = make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, DSTATE>();
  Tensor smem_B_full = make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(
                                       smem.old_B + tile_buf * SmemT::OLD_B_ELEMS)),
                                   layout_B);

  auto s2r_B = make_tiled_copy_B(Copy_Atom<LdsmB, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_B = s2r_B.get_slice(tid);

  // ── State: per-CTA swizzle layout [D_PER_CTA, DSTATE].  tile_buf selects the
  // double-buffered slot (cross-head prefetch); 0 ⇒ the single original buffer. ──
  auto layout_state_swz = make_swizzled_layout_rc<state_t, D_PER_CTA, DSTATE>();
  state_t* state_base = reinterpret_cast<state_t*>(smem.state) + tile_buf * D_PER_CTA * DSTATE;

  // ── Per-pass identity for (row, col) coords ──
  // partition_C of an identity tensor of the per-pass output shape gives this
  // thread's (row, col) at every C-frag position, including warp-N offset.
  // Frag size per thread = (M_atoms=D_PER_CTA/16) × (N_atoms_per_warp=1) × 4 elts.
  auto id_tile = make_identity_tensor(make_shape(Int<D_PER_CTA>{}, Int<N_PER_PASS>{}));
  auto id_part = thr_mma.partition_C(id_tile);
  // Linear order from CuTe's column-major partition_C with m16n8 atom:
  //   i=0,1: same row (= row_lo of M-atom 0), adjacent cols (col_off, col_off+1)
  //   i=2,3: same row (= row_hi of M-atom 0), adjacent cols
  //   i=4,5: same row (= row_lo of M-atom 1)
  //   ... (V index 0..3 inside each m16n8, then M-atoms in M-major order)
  // Pair load at (i, i+1) covers two consecutive bf16 elts → one 32-bit LDS.

  // Precompute dB coefficients once — depend only on K (lane), not on N.
  // Same V-index → K-offset mapping as common's precompute_dB_coeff, but the
  // decay comes from the register scan (shfl) instead of an old_cumAdt LDS.
  constexpr int LANES_PER_N_COL = warpSize / MMA_prop::N;  // = 4 for m16n8k_
  constexpr int DB_COEFFS_PER_LANE = MAX_WINDOW_PAD_MMA_K / LANES_PER_N_COL;
  static_assert(DB_COEFFS_PER_LANE == 2 || DB_COEFFS_PER_LANE == 4,
                "DB_COEFFS_PER_LANE must be 2 (k8) or 4 (k16)");
  float dB_coeff[DB_COEFFS_PER_LANE];
#pragma unroll
  for (int i = 0; i < DB_COEFFS_PER_LANE; ++i) {
    int const k = (lane % 4) * 2 + (i & 1) + ((i & 2) << 2);
    float const ca_k = __shfl_sync(constants::MASK_ALL_LANES, cumAdt_lane, k);
    dB_coeff[i] = (k < prev_k) ? __expf(total_cumAdt - ca_k) * old_dt_slot[k] : 0.f;
  }

  using pair_t = Pair<state_t>;

  // Philox state amortized across 4 consecutive pair conversions: each call
  // returns 4 randints, all 4 get consumed before the next refresh (vs. 1-of-4
  // in the Triton-bit-equal layout — see writeback loop below).  Compile-time
  // pair_idx (n-loop and i-loop both unrolled) keeps `rand_idx[pair_idx & 3]`
  // as a known register access — no local-memory spill.
  constexpr bool kPhiloxF16 = (PHILOX_ROUNDS > 0) && std::is_same_v<state_t, __half>;
  [[maybe_unused]] uint32_t rand_idx[4];

  constexpr int PAIRS_PER_PASS = (D_PER_CTA / M_WARPS) / 8;  // = (per-warp M-atoms) × 2 row-pairs
  static_assert(NUM_N_PASSES % 2 == 0, "Cross-pass STG fusion requires even NUM_N_PASSES");

#pragma unroll
  for (int np = 0; np < NUM_N_PASSES; np += 2) {
    // Buffer of post-cvt_rs packed u32s for both passes (philox path only).
    [[maybe_unused]] uint32_t my_packed[2][PAIRS_PER_PASS];

#pragma unroll
    for (int local_n = 0; local_n < 2; ++local_n) {
      int const n = np + local_n;
      int const n_base = n * N_PER_PASS;

      // ── Allocate per-pass C-frag (4 × M_atoms fp32 elts/thread) ──
      Tensor frag_h = thr_mma.partition_fragment_C(
          make_tensor((float*)0x0, make_shape(Int<D_PER_CTA>{}, Int<N_PER_PASS>{})));

      // ── Load state × total_decay into frag_h. ──
#pragma unroll
      for (int i = 0; i < size(frag_h); i += 2) {
        int const row = get<0>(id_part(i));
        int const col = get<1>(id_part(i)) + n_base;
        int const off = layout_state_swz(row, col);
        pair_t const p = *reinterpret_cast<pair_t const*>(&state_base[off]);
        frag_h(i) = toFloat(p[cute::Int<0>{}]) * total_decay;
        frag_h(i + 1) = toFloat(p[cute::Int<1>{}]) * total_decay;
      }

      // ── LDSM.T per-pass B (per warp = 1 atom of 8 cols of N) ──
      Tensor smem_B_n =
          local_tile(smem_B_full, make_tile(Int<N_PER_PASS>{}, Int<MAX_WINDOW_PAD_MMA_K>{}),
                     make_coord(n, _0{}));
      auto smem_B_s2r_n = s2r_thr_B.partition_S(smem_B_n);

      Tensor frag_B = thr_mma.partition_fragment_B(make_tensor(
          (MMA_prop::operand_t*)0x0, make_shape(Int<N_PER_PASS>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
      auto frag_B_view = s2r_thr_B.retile_D(frag_B);

      cute::copy(s2r_B, smem_B_s2r_n, frag_B_view);

      compute_dB_scaling<DB_COEFFS_PER_LANE>(frag_B, dB_coeff);

      // ── HMMA: frag_h += frag_A @ frag_B ──
      cute::gemm(tiled_mma, frag_h, frag_A, frag_B, frag_h);

      // ── Smem write (always) + cvt_rs into my_packed (philox path) ──
#pragma unroll
      for (int i = 0; i < size(frag_h); i += 2) {
        int const row = get<0>(id_part(i));
        int const col = get<1>(id_part(i)) + n_base;
        int const off = layout_state_swz(row, col);

        // Smem write is always RN (matmul-3 reads it); only the gmem store is SR'd.
        pair_t const q = pack_float2<state_t>(make_float2(frag_h(i), frag_h(i + 1)));
        *reinterpret_cast<pair_t*>(&state_base[off]) = q;

        if constexpr (kPhiloxF16) {
          static_assert(sizeof(state_t) == 2, "STG.64 cooperative path requires 2-byte state_t");
          int const pair_idx = n * PAIRS_PER_PASS + i / 2;
          // Per-pair philox offset (consumed only by the helper's refresh branch).
          int64_t const philox_off =
              state_ptr_offset + (int64_t)(d_tile * D_PER_CTA + row) * DSTATE + col;
          my_packed[local_n][i / 2] = stochastic_round_pair_with_philox_refresh<PHILOX_ROUNDS>(
              frag_h(i), frag_h(i + 1), pair_idx, rand_seed, philox_off, rand_idx);
        }
      }
    }

    if constexpr (kPhiloxF16) {
      if (must_checkpoint) {
        exchange_ntile_state_store_global<PAIRS_PER_PASS, N_PER_PASS, DSTATE>(
            state_w_base, np, lane, my_packed, id_part);
      }
    }
  }
}

struct HeadMetaSSU {
  int tile{-1};
  int pnat{0};             // prev_num_accepted[cache_slot], batched at fill_meta
  int ring_start{0};       // ring_start[cache_slot], batched at fill_meta
  int32_t cache_slot{-1};  // -1 = pad / past-the-end (fill_meta canonicalizes pad_slot_id).
                           // int32 like the ring slot: no widening SHF on the LDS critical path;
                           // consumers promote inside their int64 address math (native IMAD.WIDE).
  float D_val{0.f};        // params.D[head] skip coeff
  float A_val{0.f};        // params.A[head] decay rate, batched at fill_meta
  int32_t seq_len{0};      // varlen only: cu[seq+1]-cu[seq], batched at fill_meta.  Dense never
                           // reads/writes these two — SROA drops the dead registers.
  int32_t bos{0};          // varlen only: cu[seq] (packed token offset)
};

// Working scalars derived from a meta-ring entry (derive_head output).
struct HeadCoords {
  int d_tile, seq, first_head, group_idx, ring_start, prev_k, seq_len;
  int64_t outer;
};

template <int NHEADS, int HEADS_PER_GROUP, int D_SPLIT, int NPREDICTED, bool VARLEN>
__device__ __forceinline__ HeadCoords derive_head(CheckpointingSsuParams const& /*params*/,
                                                  HeadMetaSSU const& meta) {
  HeadCoords coords;
  coords.first_head = meta.tile % NHEADS;                  // compile-time NHEADS divisor
  coords.d_tile = (meta.tile / NHEADS) % D_SPLIT;          // compile-time D_SPLIT divisor
  coords.seq = (meta.tile / NHEADS) / D_SPLIT;             // batch is the range, never a divisor
  coords.group_idx = coords.first_head / HEADS_PER_GROUP;  // compile-time HPG divisor
  coords.ring_start = meta.ring_start;
  coords.prev_k = meta.pnat;
  if constexpr (VARLEN) {
    coords.seq_len = meta.seq_len;
    coords.outer = (int64_t)meta.bos;
  } else {
    coords.seq_len = NPREDICTED;  // keep the compile-time constant
    coords.outer = (int64_t)coords.seq;
  }
  return coords;
}

template <typename input_t, typename state_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int D_PER_CTA, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void prefetch_async_pre_gdc(SmemT& smem,
                                                       CheckpointingSsuParams const& params,
                                                       int lane, int warp, HeadMetaSSU const& meta,
                                                       HeadCoords const& coords, int slot,
                                                       bool must_checkpoint) {
  prefetch_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(
      smem, params, lane, warp, coords.d_tile, coords.first_head, meta.cache_slot, slot);
  load_head<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
            /*IS_FIRST=*/true>(smem, params, lane, warp, coords.d_tile, coords.first_head,
                               coords.group_idx, meta.cache_slot, coords.ring_start, coords.seq,
                               coords.outer, coords.seq_len, must_checkpoint, coords.prev_k, slot);
}

template <typename input_t, int NPREDICTED, int DIM, int D_PER_CTA, int DSTATE, typename SmemT>
__device__ __forceinline__ void prefetch_async_post_gdc(SmemT& smem,
                                                        CheckpointingSsuParams const& params,
                                                        int lane, int warp,
                                                        HeadCoords const& coords, int slot,
                                                        bool must_checkpoint) {
  if (warp == 0) {
    load_cumAdt_async(smem, params, lane, coords.seq, coords.first_head, coords.seq_len, slot);
  } else if (warp == 1 || warp == 2) {
    load_x<input_t, NPREDICTED, DIM, D_PER_CTA, DSTATE, /*IS_FIRST=*/true>(
        smem, params, lane, warp, coords.d_tile, coords.first_head, coords.group_idx, coords.outer,
        coords.seq_len, slot);
  } else if (warp == 3) {
    load_cb_async<input_t>(smem, params, lane, warp, coords.seq, coords.first_head, must_checkpoint,
                           slot);
  }
}

template <typename input_t, typename state_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int D_PER_CTA, int DSTATE, int NUM_WARPS, int NHEADS, int HEADS_PER_GROUP, bool VARLEN,
          typename SmemT>
__device__ __forceinline__ void prefetch_async(SmemT& smem, CheckpointingSsuParams const& params,
                                               int lane, int warp, HeadMetaSSU const& meta,
                                               int slot, bool must_checkpoint) {
  constexpr int D_SPLIT = DIM / D_PER_CTA;
  auto const coords =
      derive_head<NHEADS, HEADS_PER_GROUP, D_SPLIT, NPREDICTED, VARLEN>(params, meta);
  prefetch_async_pre_gdc<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                         NUM_WARPS>(smem, params, lane, warp, meta, coords, slot, must_checkpoint);
  prefetch_async_post_gdc<input_t, NPREDICTED, DIM, D_PER_CTA, DSTATE>(
      smem, params, lane, warp, coords, slot, must_checkpoint);
}

template <typename input_t, typename weight_t, typename state_t, int NPREDICTED, int MAX_WINDOW,
          int DIM, int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS, int NUM_WARPS, int NHEADS,
          int HEADS_PER_GROUP, bool VARLEN, typename SmemT>
__device__ __forceinline__ void replay_state(SmemT& smem, CheckpointingSsuParams const& params,
                                             int lane, int warp, HeadMetaSSU const& meta,
                                             int slot) {
  constexpr int D_SPLIT = DIM / D_PER_CTA;
  auto const coords =
      derive_head<NHEADS, HEADS_PER_GROUP, D_SPLIT, NPREDICTED, VARLEN>(params, meta);
  int const d_tile = coords.d_tile, first_head = coords.first_head, prev_k = coords.prev_k;
  int64_t const rand_seed = (PHILOX_ROUNDS > 0) ? *params.rand_seed : 0;
  int64_t const state_ptr_offset =
      meta.cache_slot * params.state_stride_seq + (int64_t)first_head * DIM * DSTATE;
  state_t* const state_w_base = reinterpret_cast<state_t*>(params.state) + state_ptr_offset +
                                (int64_t)d_tile * D_PER_CTA * DSTATE;
  replay_state_mma_main<input_t, state_t, DIM, D_PER_CTA, DSTATE, PHILOX_ROUNDS, NUM_WARPS>(
      smem, params, warp, lane, prev_k, d_tile, state_ptr_offset, state_w_base, rand_seed,
      meta.A_val, /*must_checkpoint=*/true, slot);
}

template <typename input_t, typename weight_t, typename state_t, int NPREDICTED, int MAX_WINDOW,
          int DIM, int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS, int NUM_WARPS, int NHEADS,
          int HEADS_PER_GROUP, bool VARLEN, bool MUST_CHECKPOINT, typename SmemT>
__device__ __forceinline__ void compute_output_and_store(SmemT& smem,
                                                         CheckpointingSsuParams const& params,
                                                         int lane, int warp,
                                                         HeadMetaSSU const& meta, int slot) {
  using namespace cute;
  constexpr int D_SPLIT = DIM / D_PER_CTA;
  auto const coords =
      derive_head<NHEADS, HEADS_PER_GROUP, D_SPLIT, NPREDICTED, VARLEN>(params, meta);
  int const d_tile = coords.d_tile, first_head = coords.first_head, prev_k = coords.prev_k,
            seq_len = coords.seq_len;
  int64_t const outer = coords.outer;
  float const D_val = meta.D_val;  // batched at fill_meta, not reloaded here
  int const tid = warp * warpSize + lane;
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int NPREDICTED_PAD_MMA_N = SmemT::NPREDICTED_PAD_MMA_N;
  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  constexpr int NUM_OUT_NTILES = NPREDICTED_PAD_MMA_N / MMA_prop::N;  // 1 (mtp≤8) or 2 (mtp>8)
  constexpr int CB_NEW_REGS_B = SmemT::CB_NEW_REGS_B;  // NUM_OUT_NTILES · (K/4) total
  constexpr int CB_OLD_REGS_B = SmemT::CB_OLD_REGS_B;
  constexpr int CB_NEW_REGS_PER =
      CB_NEW_REGS_B / NUM_OUT_NTILES;  // one output N-tile's B-frag = K/4
  constexpr int CB_OLD_REGS_PER = CB_OLD_REGS_B / NUM_OUT_NTILES;
  using MmaAtomOld_t = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG,
                                          MMA_prop::AtomK16, MMA_prop::AtomK8>;
  // Operand swap: CB is the B-operand, warps split M=DIM (Shape<NUM_M_WARPS,1>, matching
  // output_head_2k).  ONE m16n8 N-atom per output N-tile → a partition_fragment_B ARRAY (a single
  // multi-N-tile partition_fragment_B trips make_fragment_like at mtp>8).  The precompute stored
  // REGS_B = NUM_OUT_NTILES·(K/4) regs/lane g-major (g = out_ntile·REGS_PER + r) — one vectorized
  // LDS, then distribute to the per-tile B-frags.
  constexpr int NUM_M_WARPS = D_PER_CTA / MMA_prop::M;
  auto tiled_mma_cb = make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{},
                                     Layout<Shape<Int<NUM_M_WARPS>, _1>>{});
  auto tiled_mma_old_cb =
      make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomOld_t>>{}, Layout<Shape<Int<NUM_M_WARPS>, _1>>{});
  auto thr_mma_cb = tiled_mma_cb.get_slice(tid);
  auto thr_mma_old_cb = tiled_mma_old_cb.get_slice(tid);
  using FragCBNewT = decltype(thr_mma_cb.partition_fragment_B(make_tensor(
      (MMA_prop::operand_t*)nullptr, make_shape(Int<MMA_prop::N>{}, Int<NPREDICTED_PAD_MMA_M>{}))));
  using FragCBOldT = decltype(thr_mma_old_cb.partition_fragment_B(make_tensor(
      (MMA_prop::operand_t*)nullptr, make_shape(Int<MMA_prop::N>{}, Int<MAX_WINDOW_PAD_MMA_K>{}))));
  FragCBNewT frag_CB_new[NUM_OUT_NTILES];
  FragCBOldT frag_CB_old[NUM_OUT_NTILES];
  // The fragment element is the MMA operand type (cutlass::bfloat16_t), bit-compatible with input_t
  // but a distinct C++ type — read the gmem/smem bytes AS that type so the assign is well-typed.
  using cb_new_frag_t = cute::remove_cvref_t<decltype(frag_CB_new[0](0))>;
  using cb_old_frag_t = cute::remove_cvref_t<decltype(frag_CB_old[0](0))>;
  // β = exp(cumAdt[prev_k−1]), warp-scanned from the dt ring — hoisted AHEAD of the CB
  // fragment loads + the whole output setup (the old_cumAdt LDS it replaces was the ncu v28
  // top short_scoreboard site when sunk to its FMUL consumer; the scan is shfl/ALU-only).
  // __expf(0) == 1.f exactly, so the prev_k==0 constant is bit-identical to the old ternary.
  float beta_extra = 1.f;
  if constexpr (!MUST_CHECKPOINT) {
    if (prev_k > 0)
      beta_extra = __expf(__shfl_sync(
          constants::MASK_ALL_LANES,
          warp_scan_old_cumAdt<MAX_WINDOW>(smem, meta.A_val, lane, prev_k, slot), prev_k - 1));
  }
  {
    input_t const* cb_new_s = smem.cb_new + slot * SmemT::CB_NEW_ELEMS;
    auto const raw =
        reinterpret_cast<PackedAligned<cb_new_frag_t, CB_NEW_REGS_B> const*>(cb_new_s)[lane];
#pragma unroll
    for (int on = 0; on < NUM_OUT_NTILES; ++on)
#pragma unroll
      for (int r = 0; r < CB_NEW_REGS_PER; ++r)
        frag_CB_new[on](r) = raw.val[on * CB_NEW_REGS_PER + r];
  }
  if constexpr (!MUST_CHECKPOINT) {
    input_t const* cb_old_s = smem.cb_old + slot * SmemT::CB_OLD_ELEMS;
    auto const raw =
        reinterpret_cast<PackedAligned<cb_old_frag_t, CB_OLD_REGS_B> const*>(cb_old_s)[lane];
#pragma unroll
    for (int on = 0; on < NUM_OUT_NTILES; ++on)
#pragma unroll
      for (int r = 0; r < CB_OLD_REGS_PER; ++r)
        frag_CB_old[on](r) = raw.val[on * CB_OLD_REGS_PER + r];
  }
  int64_t const out_seq_base = outer * params.out_stride_seq;
  // Ring: appends land at (ring_start + prev_k + i) % RING_BUFFER_LEN on both
  // branches; the flush advances ring_start host-side after the step.
  int const write_offset = prev_k;
  output_head_2k<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, PHILOX_ROUNDS,
                 NUM_WARPS, MUST_CHECKPOINT>(
      smem, params, lane, warp, d_tile, first_head, meta.cache_slot, meta.ring_start, prev_k,
      out_seq_base, write_offset, seq_len, D_val, beta_extra, slot, frag_CB_new, frag_CB_old);
}

template <typename input_t, typename weight_t, typename state_t, int NPREDICTED, int MAX_WINDOW,
          int DIM, int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS, int NUM_WARPS, int NHEADS,
          int HEADS_PER_GROUP, bool VARLEN, bool MUST_CHECKPOINT, typename SmemT>
__device__ __forceinline__ void process_head(SmemT& smem, CheckpointingSsuParams const& params,
                                             int lane, int warp, HeadMetaSSU const& meta,
                                             int slot) {
  if constexpr (MUST_CHECKPOINT) {
    replay_state<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                 PHILOX_ROUNDS, NUM_WARPS, NHEADS, HEADS_PER_GROUP, VARLEN>(smem, params, lane,
                                                                            warp, meta, slot);
    __syncthreads();  // publish the in-place-replayed state cross-warp before the output MMA
  }
  compute_output_and_store<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA,
                           DSTATE, PHILOX_ROUNDS, NUM_WARPS, NHEADS, HEADS_PER_GROUP, VARLEN,
                           MUST_CHECKPOINT>(smem, params, lane, warp, meta, slot);
}

// Reg cap derived from the measured-optimal blocks/SM per ring depth (64 K regs/SM, 128
// threads/block, 8-reg allocation granularity).  The smem and reg walls diverged when the kernel
// went persistent (smem wall: ~29 KB → 7 blocks at depth 1, ~57 KB → 4 at depth 2; natural reg
// demand ~118 → 4 blocks), so the cap picks the point on the spill-vs-occupancy curve.  Depth-1
// sweep (b=1024 bf16 mtp=8 pnat=4): cap 64 / 8 blk = 95.7 µs (spill-bound), 80 / 6 = 79.6
// (optimum — triton-replay-pm parity), 96 / 5 = 81.6, uncapped → ~150 regs / 3 blk = 95.6.
// Depth ≥2: 128 / 4 blk tuned earlier (118 regs, no spills; a 96 cap forced per-unit addressing
// recompute, and uncapping → 150 regs / 3 blocks, +20 µs).
constexpr int main_maxnreg(int num_stages) {
  int const target_blocks = (num_stages == 1) ? 6 : 4;
  int const cap = 65536 / (target_blocks * 128);
  return cap - cap % 8;
}

// =============================================================================
// Main kernel.  Template params mirror checkpointing_ssu_kernel so the launcher
// dispatches both with the same args.  NUM_STAGES = per-unit smem ring depth
// (each stage holds one work-unit's full bundle; the grid loop prefetches
// NUM_STAGES units ahead).  The launcher hard-codes 1 — the occupancy regime —
// and FLASHINFER_SSU_MAIN_PIPELINE_STAGES overrides it.
// =============================================================================
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int DSTATE, int HEADS_PER_GROUP, int PHILOX_ROUNDS, int NUM_WARPS, int D_SPLIT = 1,
          bool VARLEN = false, int NGROUPS = 1, int NUM_STAGES = 1>
__global__ __maxnreg__(main_maxnreg(NUM_STAGES)) void checkpointing_ssu_main_kernel(
    CheckpointingSsuParams params) {
  static_assert(DIM % D_SPLIT == 0, "DIM must be divisible by D_SPLIT");
  constexpr int D_PER_CTA = DIM / D_SPLIT;
  static_assert(D_PER_CTA >= 32, "D_PER_CTA must be >= 32 (output MMA m16n8 with _1×4 layout)");
  static_assert(NPREDICTED <= MAX_WINDOW, "NPREDICTED must be <= MAX_WINDOW");
  static_assert(MAX_WINDOW <= MMA_prop::K_BIG, "MAX_WINDOW must be <= MMA::K_BIG=16");
  assert(params.d_split == D_SPLIT);

  // ── EXTERNAL PDL: signal a programmatic DOWNSTREAM kernel that `output` is fully written
  // (ALL work-units done; the per-head cache writes are next-step-only).  Gated by ENABLE_PDL.
  // (The internal precompute→main chain uses the one-shot gdc_wait on the first work-unit.) ──
  if constexpr (ENABLE_PDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }

  using SmemT = CheckpointingSsuMainStorage<input_t, state_t, NPREDICTED, MAX_WINDOW, D_PER_CTA,
                                            DSTATE, NUM_STAGES, VARLEN>;
  extern __shared__ __align__(128) char smem_buf[];
  auto& smem = *reinterpret_cast<SmemT*>(smem_buf);

  // ── 1D grid-stride persistent loop over single-head work-units ──
  // Work-unit = (d_tile, seq, head); the launcher sizes the grid to
  // min(cta_per_sm·NUM_SMS, total_work) and each CTA grid-strides over work-units.  Default
  // cta_per_sm = occupancy ⇒ grid = the resident set, each CTA strides; bigger ⇒ grid==total_work.
  // Flatten head-fastest (head innermost) so consecutive work-units are consecutive heads of one
  // (d_tile, seq) → per-group C/old_B stay L2-hot.
  //
  // NHEADS = NGROUPS·HEADS_PER_GROUP is COMPILE-TIME, so the unflatten divides only by compile-time
  // constants (NHEADS, D_SPLIT, HEADS_PER_GROUP) — no runtime div/mod.  seq is the top quotient, so
  // `batch` is never a divisor; gridDim.x is the loop stride, never a divisor.
  constexpr int NHEADS = NGROUPS * HEADS_PER_GROUP;
  int const lane = threadIdx.x;
  int const warp = threadIdx.y;
  int const batch = (int)params.batch;
  int const total_work = D_SPLIT * batch * NHEADS;
  int const stride = (int)gridDim.x;

  auto const* __restrict__ sbi = reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  auto const* __restrict__ prev_ptr = reinterpret_cast<int32_t const*>(params.prev_num_accepted);
  auto const* __restrict__ cu = reinterpret_cast<int32_t const*>(params.cu_seqlens);
  auto const* __restrict__ D_ptr = reinterpret_cast<weight_t const*>(params.D);
  auto const* __restrict__ A_ptr = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ rsp = reinterpret_cast<int32_t const*>(params.ring_start);

  constexpr int STAGES = NUM_STAGES;
  constexpr int META_RING = SmemT::META_RING;
  static_assert(STAGES < META_RING, "meta window must cover the process+prefetch span");
  auto fill_meta = [&](int base) {
    int const unit = base + lane;
    int const work_unit = blockIdx.x + unit * stride;
    int const ring_slot = unit & (META_RING - 1);
    int32_t cache_slot = -1;  // canonical pad / past-the-end sentinel
    int pnat = 0, ring_start = 0;
    int32_t seq_len = 0, bos = 0;
    float D_val = 0.f, A_val = 0.f;
    if (work_unit < total_work) {
      int const seq = (work_unit / NHEADS) / D_SPLIT;  // compile-time NHEADS, D_SPLIT divisors
      int64_t const raw_slot = sbi ? static_cast<int64_t>(sbi[seq]) : seq;
      if (raw_slot != params.pad_slot_id) {
        cache_slot = static_cast<int32_t>(raw_slot);  // cache slots ≪ 2^31
        pnat = prev_ptr[raw_slot];
        ring_start = __ldg(rsp + raw_slot);
      }
      D_val = D_ptr ? toFloat(D_ptr[work_unit % NHEADS]) : 0.f;
      A_val = toFloat(A_ptr[work_unit % NHEADS]);
      if constexpr (VARLEN) {
        bos = __ldg(&cu[seq]);
        seq_len = __ldg(&cu[seq + 1]) - bos;
      }
    }
    smem.meta_cache_slot[ring_slot] = cache_slot;
    smem.meta_pnat[ring_slot] = pnat;
    smem.meta_ring_start[ring_slot] = ring_start;
    smem.meta_D[ring_slot] = D_val;
    smem.meta_A[ring_slot] = A_val;
    if constexpr (VARLEN) {
      smem.meta_cu[ring_slot] = (bos << SmemT::SEQLEN_BITS) | seq_len;
    }
    __syncwarp();
  };
  auto load_meta = [&](int unit) -> HeadMetaSSU {
    int const ring_slot = unit & (META_RING - 1);
    HeadMetaSSU meta;
    meta.tile = blockIdx.x + unit * stride;
    meta.cache_slot = smem.meta_cache_slot[ring_slot];
    meta.pnat = smem.meta_pnat[ring_slot];
    meta.ring_start = smem.meta_ring_start[ring_slot];
    meta.D_val = smem.meta_D[ring_slot];
    meta.A_val = smem.meta_A[ring_slot];
    if constexpr (VARLEN) {
      int32_t const packed = smem.meta_cu[ring_slot];  // one LDS for both fields
      meta.seq_len = packed & ((1 << SmemT::SEQLEN_BITS) - 1);
      meta.bos = packed >> SmemT::SEQLEN_BITS;  // bos < 2^23 ⇒ packed ≥ 0, arithmetic >> safe
    }
    return meta;
  };
  // valid / must_checkpoint from the ring entry alone — varlen's row length rides the ring
  // (batched at fill_meta), so neither touches cu_seqlens per unit.
  auto is_valid = [&](HeadMetaSSU const& meta) -> bool {
    if (meta.cache_slot < 0) return false;  // canonical pad / past-the-end sentinel (fill_meta)
    if constexpr (VARLEN) {
      return meta.seq_len > 0;  // empty varlen row ⇒ skip
    }
    return true;
  };
  auto mc_of = [&](HeadMetaSSU const& meta) -> bool {
    if constexpr (VARLEN) {
      return meta.pnat + meta.seq_len > MAX_WINDOW;
    }
    return meta.pnat + NPREDICTED > MAX_WINDOW;
  };

  // ── PROLOGUE: fill meta window 0, then tile 0 (smem buffer 0).  The pre-gdc cp.asyncs
  // (state + ring tiles — previous-step data only) are issued first so they fly during the
  // gdc wait; everything the precompute produces (cb / cumAdt / x / C) is issued strictly
  // after the gdc, and only the OUTPUT phase consumes it — the replay runs entirely on
  // pre-gdc data (old decay is warp-scanned from the dt ring), overlapping the G_post0
  // round trip. ──
  fill_meta(0);
  HeadMetaSSU const meta0 = load_meta(0);  // wu(0)=blockIdx.x < total_work (CTA has ≥1 work-unit)
  bool const valid0 = is_valid(meta0);
  bool const must_checkpoint0 = valid0 && mc_of(meta0);
  HeadCoords coords0{};  // derived once, reused by the post-gdc half across the gdc wait
  if (valid0) {
    coords0 = derive_head<NHEADS, HEADS_PER_GROUP, D_SPLIT, NPREDICTED, VARLEN>(params, meta0);
    prefetch_async_pre_gdc<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                           NUM_WARPS>(smem, params, lane, warp, meta0, coords0, /*slot=*/0,
                                      must_checkpoint0);
  }
  __pipeline_commit();  // G_pre0 (empty if tile 0 is a pad)

  cudaGridDependencySynchronize();  // gdc ONCE — precompute outputs (cb / cumAdt / x / C)
                                    // now visible

  if (valid0)
    prefetch_async_post_gdc<input_t, NPREDICTED, DIM, D_PER_CTA, DSTATE>(
        smem, params, lane, warp, coords0, /*slot=*/0, must_checkpoint0);
  __pipeline_commit();  // G_post0

  __pipeline_wait_prior(1);  // drain G_pre0 ONLY — the replay consumes no this-step precompute
                             // output (old decay is warp-scanned from the dt ring), so the
                             // G_post0 CB/cumAdt round trip stays in flight underneath it
  __syncthreads();           // publish the pre-gdc bundle cross-warp
  if (must_checkpoint0)
    replay_state<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                 PHILOX_ROUNDS, NUM_WARPS, NHEADS, HEADS_PER_GROUP, VARLEN>(smem, params, lane,
                                                                            warp, meta0, 0);
  __syncthreads();  // publish replayed state

  // Prefetch units 1..STAGES-1 (full bundles; gdc already fired) — unit p into smem buffer p,
  // seeding the register queue (meta_q[k] == meta(tile+k) at the top of steady iteration tile).
  HeadMetaSSU meta_q[STAGES];
#pragma unroll
  for (int p = 1; p < STAGES; ++p) {
    meta_q[p - 1] = load_meta(p);
    if (is_valid(meta_q[p - 1]))
      prefetch_async<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
                     NHEADS, HEADS_PER_GROUP, VARLEN>(smem, params, lane, warp, meta_q[p - 1],
                                                      /*slot=*/p, mc_of(meta_q[p - 1]));
    __pipeline_commit();  // G_full_p (empty if pad / past-the-end)
  }

  // FIFO: [G_post0, G_full_1 .. G_full_{STAGES-1}] = STAGES groups — the wait drains G_post0
  // (tile-0's CB/cumAdt, overlapped with the replay above) before tile-0's output consumes it.
  __pipeline_wait_prior(STAGES - 1);
  __syncthreads();  // publish tile-0 post-gdc data (C / x / cumAdt) cross-warp before its output
  if (valid0) {
    if (must_checkpoint0)
      compute_output_and_store<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA,
                               DSTATE, PHILOX_ROUNDS, NUM_WARPS, NHEADS, HEADS_PER_GROUP, VARLEN,
                               /*MUST_CHECKPOINT=*/true>(smem, params, lane, warp, meta0, 0);
    else
      compute_output_and_store<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA,
                               DSTATE, PHILOX_ROUNDS, NUM_WARPS, NHEADS, HEADS_PER_GROUP, VARLEN,
                               /*MUST_CHECKPOINT=*/false>(smem, params, lane, warp, meta0, 0);
  }
  __syncthreads();  // tile-0 output done reading buffer 0 before the tile-STAGES prefetch
                    // overwrites it

  // Prefetch unit STAGES into buffer 0 (freed by tile-0's output), completing the queue —
  // meta_q == {meta(1) .. meta(STAGES)} — and establishing the steady invariant for tile == 1.
  meta_q[STAGES - 1] = load_meta(STAGES);
  if (is_valid(meta_q[STAGES - 1]))
    prefetch_async<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
                   NHEADS, HEADS_PER_GROUP, VARLEN>(smem, params, lane, warp, meta_q[STAGES - 1],
                                                    /*slot=*/0, mc_of(meta_q[STAGES - 1]));
  __pipeline_commit();

  // ── STEADY STATE: process unit `tile` from the register queue, then prefetch unit tile+STAGES
  //    into the freed buffer `slot`.  Refill the smem meta window every META_RING−STAGES units
  //    (see the meta-pipeline comment above). ──
  int next_fill = META_RING - STAGES;
  for (int tile = 1; blockIdx.x + tile * stride < total_work; ++tile) {
    int const slot = tile % STAGES;  // smem buffer of unit `tile` (== buffer of unit tile+STAGES)
    if (tile == next_fill) {
      fill_meta(tile);
      next_fill += META_RING - STAGES;
    }
    HeadMetaSSU const meta = meta_q[0];  // register: appended STAGES iterations ago, LDS drained
    bool const valid = is_valid(meta);
    __pipeline_wait_prior(STAGES - 1);  // drain unit `tile`'s bundle (the oldest group)
    if (valid) {
      __syncthreads();  // publish the drained bundle (state + old-tiles + C/x/cumAdt) cross-warp
      if (mc_of(meta))
        process_head<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                     PHILOX_ROUNDS, NUM_WARPS, NHEADS, HEADS_PER_GROUP, VARLEN,
                     /*MUST_CHECKPOINT=*/true>(smem, params, lane, warp, meta, slot);
      else
        process_head<input_t, weight_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                     PHILOX_ROUNDS, NUM_WARPS, NHEADS, HEADS_PER_GROUP, VARLEN,
                     /*MUST_CHECKPOINT=*/false>(smem, params, lane, warp, meta, slot);
    }
    // Shift the queue (constant indices → registers) and append unit tile+STAGES from the smem
    // ring BEFORE the barrier (ring writes only happen at iteration tops, and no warp reaches the
    // next fill until all pass this barrier — the slot is stable): the barrier swallows the LDS
    // latency for the prefetch below, and the entry isn't consumed as a process-meta for STAGES
    // more iterations.
#pragma unroll
    for (int k = 0; k < STAGES - 1; ++k) meta_q[k] = meta_q[k + 1];
    meta_q[STAGES - 1] = load_meta(tile + STAGES);
    __syncthreads();  // output done reading `slot` before the prefetch below overwrites it

    if (is_valid(meta_q[STAGES - 1]))
      prefetch_async<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
                     NHEADS, HEADS_PER_GROUP, VARLEN>(smem, params, lane, warp, meta_q[STAGES - 1],
                                                      slot, mc_of(meta_q[STAGES - 1]));
    __pipeline_commit();  // keep the FIFO at STAGES groups (empty for pad / past-the-end)
  }
}

}  // namespace flashinfer::mamba::checkpointing

#endif  // FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_MAIN_CUH_
