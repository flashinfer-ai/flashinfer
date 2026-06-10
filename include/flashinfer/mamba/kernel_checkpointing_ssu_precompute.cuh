/*
 * Copyright (c) 2025 by FlashInfer team.
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
// Two-kernel split: PRECOMPUTE kernel (.plans/ssu_split.md, S3).
//
// PROTOTYPE / DESIGN DRAFT — structure for review, expect compile iteration.
//
// Computes the conv1d-coefficient block (equations C1/C2/C5 + the
// old_dt/old_cumAdt cache writes) and stores two scratch tensors the `main`
// kernel consumes:
//   cb_scaled : bf16, FRAGMENT-NATIVE [batch, nheads, lane(0..31), reg(0..7)] —
//               equation C5 laid out as matmul-4's fragA, so the main reads it
//               with one LDG.128 per thread straight into fragA.
//   decay_vec : f32, (batch, nheads, NPREDICTED_PAD_MMA_M) — exp(cumAdt[t]),
//               the per-head β factor for the main's OUT.1 (β·C@state).
//
// Launch granularity (see .plans/ssu_split.md "Granularity"):
//   grid = (batch, ngroups, ceil(HEADS_PER_GROUP / HEADS_PER_CTA))
//   First cut: HEADS_PER_CTA == HEADS_PER_GROUP → tiles=1 → 1 CTA/group.
//   The raw C·B matmul (C5's C·B contraction over DSTATE) is PER-GROUP — B/C
//   are (.,.,ngroups,dstate) — so it is computed ONCE per CTA and the result
//   (frag_acc, registers) is reused across the tile's heads; only the per-head
//   decay scaling differs.  That is the whole point of the per-group grid.
//
// CB_scaled store: one STG.128 per thread of a PackedAligned<input_t> (16 B) in
// fragA-native layout — the precompute never puts CB in (swizzled) smem and the
// main never LDSMs it; each side does a single 16 B vectorized transfer of the
// thread's 8-bf16 fragment.  (The two m16n8 N-tiles a thread accumulates ==
// fragA's 8 elements in the same order — see scale_store_cb_gmem.)
// =============================================================================
#ifndef FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_PRECOMPUTE_CUH_
#define FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_PRECOMPUTE_CUH_

#include <cute/tensor.hpp>

#include "checkpointing_ssu.cuh"
#include "common.cuh"  // PackedAligned
#include "kernel_checkpointing_ssu_common.cuh"

namespace flashinfer::mamba::checkpointing {

// -----------------------------------------------------------------------------
// Store one head's scaled CB to gmem in FRAGMENT-NATIVE layout, one STG.128 per
// thread.  The two m16n8 N-tile accumulators that produce CB (n=[0,8) ++
// n=[8,16), 4+4 f32/thread) have the SAME per-thread element order as `fragA`
// for matmul-4's `CB @ x` (8 bf16/thread, mma.m16n8k16 A operand).  So laying
// each thread's 8 scaled values out contiguously as [batch, head, lane, 8]
// lets the MAIN kernel read them with a single LDG.128 straight into fragA —
// no LDSM, no swizzle, no de-swizzle, neither side touches smem for CB.
//
// `raw_cb` = this lane's 8 raw C·B values in fragA element order:
//   e0=(r0,c0)   e1=(r0,c0+1) e2=(r1,c0)   e3=(r1,c0+1)   [N-tile 0]
//   e4=(r0,c0+8) e5=(r0,c0+9) e6=(r1,c0+8) e7=(r1,c0+9)   [N-tile 1]
// with r0=lane/4, r1=r0+8, c0=(lane%4)*2.
// cb_gmem_head = &cb_scaled[batch_slot, head, 0]  (32 × PackedAligned<input_t>).
template <typename input_t, typename SmemT>
__device__ __forceinline__ void scale_store_cb_gmem(
    float const (&raw_cb)[8], SmemT& smem, int lane, int seq_len,
    PackedAligned<input_t>* __restrict__ cb_gmem_head) {
  static_assert(PackedAligned<input_t>::count == 8,
                "cb_scaled fragA store assumes 8 input_t per 16 B pack (bf16)");
  int const r0 = lane / 4;
  int const c0 = (lane % 4) * 2;
  PackedAligned<input_t> packed;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    // fragA(m16n8k16) element e → (row t, col j), on the fly (folds at unroll):
    //   row half = (e >> 1) & 1  → t = r0 + 8*half
    //   N-tile   = (e >> 2) & 1  → j += 8*Ntile  (N-tile 1 covers cols [8,16))
    //   col pair =  e & 1        → j += pair
    int const t = r0 + (((e >> 1) & 1) << 3);
    int const j = c0 + (((e >> 2) & 1) << 3) + (e & 1);
    float val = 0.f;
    if (j <= t && t < seq_len && j < seq_len) {
      // C5: CB_scaled[t,j] = (C·B) * exp(cumAdt[t]-cumAdt[j]) * dt_proc[j].
      val = raw_cb[e] * __expf(smem.cumAdt[t] - smem.cumAdt[j]) * smem.dt_proc[j];
    }
    packed.val[e] = static_cast<input_t>(val);
  }
  cb_gmem_head[lane] = packed;  // one STG.128
}

// -----------------------------------------------------------------------------
// PRECOMPUTE kernel.  Template params mirror checkpointing_ssu_kernel.
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, int NPREDICTED, int MAX_WINDOW, int DIM, int DSTATE,
          int HEADS_PER_GROUP, int NUM_WARPS, bool VARLEN = false>
__global__ void checkpointing_ssu_precompute_kernel(CheckpointingSsuParams params) {
  using SmemT = CheckpointingSsuStorage<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, DSTATE>;
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int N_HALF = NPREDICTED_PAD_MMA_M / 2;
  // First cut: one CTA per (batch, group) — every head of the group handled by
  // this CTA's head loop (HEADS_PER_CTA == HEADS_PER_GROUP).
  constexpr int HEADS_PER_CTA = HEADS_PER_GROUP;  // TODO(tiling): heuristic for HPG=64.

  extern __shared__ __align__(128) char smem_buf[];
  auto& smem = *reinterpret_cast<SmemT*>(smem_buf);

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
  auto const* __restrict__ buf_idx_ptr = reinterpret_cast<int32_t const*>(params.cache_buf_idx);
  int const buf_read = __ldg(&buf_idx_ptr[cache_slot]);
  auto const* __restrict__ prev_ptr = reinterpret_cast<int32_t const*>(params.prev_num_accepted);
  int const prev_k = prev_ptr[cache_slot];

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
  int const buf_write = must_checkpoint ? (1 - buf_read) : buf_read;
  int const write_offset = must_checkpoint ? 0 : prev_k;

  if constexpr (ENABLE_PDL) {
    cudaGridDependencySynchronize();  // conv1d produces B/C — wait before the load.
  }

  // ── Load C, B for this GROUP into smem (shared across the head loop) ──
  // TODO(iterate): factor the C/B cp.async out of load_*_data — the precompute
  // needs only C and B (not state/old_x/x/z).  Reuse the same swizzled smem.C /
  // smem.B layouts so compute_cb_raw below indexes them identically.
  // load_group_BC<input_t, NPREDICTED, DSTATE>(smem, params, lane, warp, group_idx, outer,
  // seq_len);

  // ── Raw C·B MMA — ONCE per group, 2-warp N-split → swizzled smem.CB_scaled ──
  // TODO(iterate): like compute_CB_scaled_2warp (common.cuh:869-938) but store
  // the RAW (unscaled) accumulator to the swizzled smem.CB_scaled — the
  // decay/dt scaling is per-head and deferred to the per-warp LDSM below.
  // Warps 0/1 split the N axis; warps 2/3 idle here (the heavy B/C reads happen
  // once, not per-warp).
  // compute_cb_raw_2warp<input_t, NPREDICTED, DSTATE>(smem, warp, lane);

  // ── Warp-per-head ──
  // Each warp LDSMs the shared raw CB → fragA, scales by ITS head, STG.128s
  // fragA-native to gmem.  NUM_ITER is uniform across warps, so the iter==0
  // __syncthreads() is reached by every warp exactly once — deadlock-proof even
  // when HEADS_PER_CTA < NUM_WARPS (tail warps just no-op the h-guarded work).
  // The barrier is deferred past the cumAdt compute so the raw-CB smem store
  // (warps 0/1) overlaps it.
  auto const* __restrict__ A_ptr = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ dt_bias_ptr = reinterpret_cast<weight_t const*>(params.dt_bias);
  auto* __restrict__ cb_gmem = reinterpret_cast<PackedAligned<input_t>*>(params.cb_scaled);
  auto* __restrict__ decay_gmem = reinterpret_cast<float*>(params.decay_vec);

  constexpr int NUM_ITER = (HEADS_PER_CTA + NUM_WARPS - 1) / NUM_WARPS;  // ceil, uniform
#pragma unroll
  for (int iter = 0; iter < NUM_ITER; ++iter) {
    int const h = warp + iter * NUM_WARPS;
    bool const has_head = (h < HEADS_PER_CTA);
    int const head = first_head + h;

    if (has_head) {
      float const A_val = toFloat(A_ptr[head]);
      float const dt_bias_val = dt_bias_ptr ? toFloat(dt_bias_ptr[head]) : 0.f;
      // C1: dt_proc[t] = softplus(dt[outer,t,head] + dt_bias) → per-warp smem.
      // load_dt_proc<dt_t, NPREDICTED>(smem, params, warp, lane, head, outer,
      //                                dt_bias_val, seq_len);
      // C2: cumAdt + decay (warp scan) → per-warp smem.cumAdt[warp]/decay[warp].
      // compute_cumAdt_pw<NPREDICTED>(smem, warp, lane, A_val);
      (void)A_val;
      (void)dt_bias_val;
    }

    // Deferred barrier: overlaps the raw-CB smem store with the cumAdt compute.
    if (iter == 0) __syncthreads();

    if (has_head) {
      // decay_vec[batch, head, t] = exp(cumAdt[t]) (per-warp decay buffer).
      // if (lane < seq_len)
      //   decay_gmem[(seq*nheads + head)*NPREDICTED_PAD_MMA_M + lane] =
      //       smem.decay_pw(warp, lane);
      // LDSM the shared raw CB from smem.CB_scaled → raw_cb[8] (fragA order).
      float raw_cb[8];
      // ldsm_cb_raw<input_t>(smem, lane, raw_cb);
      (void)raw_cb;
      // C5: scale by this head's per-warp cumAdt/dt_proc + STG.128 fragA-native.
      auto* cb_gmem_head = cb_gmem + (int64_t)(seq * params.nheads + head) * warpSize;
      // scale_store_cb_gmem<input_t>(raw_cb, smem, warp, lane, seq_len, cb_gmem_head);
      (void)cb_gmem_head;
      // C7 + cache: store old_dt / old_cumAdt at write_offset (reuse
      // store_old_dt / store_old_cumAdt, common.cuh:1582+).
    }
  }
  (void)decay_gmem;
  (void)buf_write;
  (void)write_offset;
}

}  // namespace flashinfer::mamba::checkpointing

#endif  // FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_PRECOMPUTE_CUH_
