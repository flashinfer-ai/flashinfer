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
#ifndef FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_MAIN_8BIT_CUH_
#define FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_MAIN_8BIT_CUH_

// Quantized-state main for the precompute + main split.  Unlike the generic
// main, this is deliberately single-stage: the 8-bit replay uses a distinct
// M-sharded chain and per-row scale reduction, so pretending that the generic
// pipeline-depth knob applies would only create duplicate autotuner tactics.

#include "kernel_checkpointing_ssu_8bit.cuh"
#include "kernel_checkpointing_ssu_main.cuh"

namespace flashinfer::mamba::checkpointing {

// The precompute stores CB in the operand-swapped output MMA's B-fragment
// layout.  Rebuild the monolithic 8-bit kernel's swizzled matrix so its tested
// replay/output helpers can be reused unchanged.  One warp loads each matrix:
// W0 new-token CB, W1 old-token CB (no-write only).
template <bool IS_OLD, typename input_t, int NPREDICTED, int MAX_WINDOW, typename SmemT>
__device__ __forceinline__ void load_cb_8bit_main(SmemT& smem,
                                                  CheckpointingSsuParams const& params, int lane,
                                                  int seq, int head, bool must_checkpoint) {
  if constexpr (IS_OLD) {
    if (must_checkpoint) return;
  }
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int NPREDICTED_PAD_MMA_N = SmemT::NPREDICTED_PAD_MMA_N;
  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  constexpr int K = IS_OLD ? MAX_WINDOW_PAD_MMA_K : NPREDICTED_PAD_MMA_M;
  constexpr int NUM_OUT_NTILES = NPREDICTED_PAD_MMA_N / MMA_prop::N;
  constexpr int REGS_B_PER = K / 4;
  constexpr int REGS_B = NUM_OUT_NTILES * REGS_B_PER;
  constexpr int COL_OFFSET = IS_OLD ? NPREDICTED_PAD_MMA_M : 0;

  auto const* scratch = reinterpret_cast<input_t const*>(IS_OLD ? params.cb_old : params.cb_scaled);
  scratch += (int64_t)(seq * params.nheads + head) * warpSize * REGS_B;
  auto const packed = reinterpret_cast<PackedAligned<input_t, REGS_B> const*>(scratch)[lane];

  auto layout_cb =
      make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, SmemT::CB_ROW_STRIDE>();
  int const row_base = lane / 4;
  int const col_base = (lane % 4) * 2;
#pragma unroll
  for (int g = 0; g < REGS_B; ++g) {
    int const output_tile = g / REGS_B_PER;
    int const reg = g % REGS_B_PER;
    int const row = row_base + output_tile * MMA_prop::N;
    int const col = col_base + (((reg >> 1) & 1) << 3) + (reg & 1);
    smem.CB_scaled[layout_cb(row, COL_OFFSET + col)] = packed.val[g];
  }
}

template <typename input_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, int NPREDICTED, int MAX_WINDOW, int DIM, int DSTATE,
          int HEADS_PER_GROUP, int PHILOX_ROUNDS, bool VARLEN = false, int NGROUPS = 1>
__global__ void checkpointing_ssu_main_kernel_8bit(CheckpointingSsuParams params) {
  static_assert(sizeof(input_t) == 2, "8-bit split main requires 2-byte input");
  static_assert(sizeof(state_t) == 1, "8-bit split main requires int8/fp8 state");
  static_assert(DIM == 64, "8-bit split main requires DIM=64");
  static_assert(NPREDICTED <= MAX_WINDOW);
  static_assert(MAX_WINDOW <= MMA_prop::K_BIG);
  assert(params.d_split == 1);

  constexpr int NUM_WARPS = 4;
  constexpr int NHEADS = NGROUPS * HEADS_PER_GROUP;
  using SmemT =
      CheckpointingSsuStorage8bit<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, DSTATE>;
  extern __shared__ __align__(128) char smem_buf[];
  auto& smem = *reinterpret_cast<SmemT*>(smem_buf);

  int const lane = threadIdx.x;
  int const warp = threadIdx.y;
  int const total_work = static_cast<int>(params.batch) * NHEADS;
  auto const* sbi = reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  auto const* prev_ptr = reinterpret_cast<int32_t const*>(params.prev_num_accepted);
  auto const* ring_start_ptr = reinterpret_cast<int32_t const*>(params.ring_start);
  auto const* A_ptr = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* D_ptr = reinterpret_cast<weight_t const*>(params.D);
  auto const* cu = reinterpret_cast<int32_t const*>(params.cu_seqlens);

  // Match the generic split main's downstream PDL contract.  Cache writes are
  // next-step-only; a dependent consumer waits before reading output.
  if constexpr (ENABLE_PDL) cudaTriggerProgrammaticLaunchCompletion();

  bool first_work = true;
  for (int work = blockIdx.x; work < total_work; work += gridDim.x) {
    int const seq = work / NHEADS;
    int const head = work % NHEADS;
    int const group_idx = head / HEADS_PER_GROUP;
    int64_t const raw_slot = sbi ? static_cast<int64_t>(sbi[seq]) : seq;
    bool const valid = raw_slot != params.pad_slot_id;

    int prev_k = 0;
    int ring_start = 0;
    int seq_len = NPREDICTED;
    int64_t outer = seq;
    if (valid) {
      prev_k = __ldg(&prev_ptr[raw_slot]);
      ring_start = __ldg(&ring_start_ptr[raw_slot]);
      if constexpr (VARLEN) {
        int const bos = __ldg(&cu[seq]);
        int const eos = __ldg(&cu[seq + 1]);
        seq_len = eos - bos;
        outer = bos;
      }
    }
    bool const active = valid && seq_len > 0;
    bool const must_checkpoint = active && prev_k + seq_len > MAX_WINDOW;
    float const A_val = active ? toFloat(A_ptr[head]) : 0.f;
    float const D_val = active && D_ptr ? toFloat(D_ptr[head]) : 0.f;

    // Pre-gdc bundle contains only previous-step state/cache data (plus z),
    // and therefore overlaps the precompute.  The one-shot wait below makes
    // this step's CB/cumAdt and conv1d C/x visible.
    if (active) {
      prefetch_state<state_t, DIM, DIM, DSTATE, NUM_WARPS>(smem, params, lane, warp, 0, head,
                                                           raw_slot, 0);
      load_head<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, DIM, DSTATE, NUM_WARPS,
                /*IS_FIRST=*/true>(smem, params, lane, warp, 0, head, group_idx, raw_slot,
                                   ring_start, seq, outer, seq_len, must_checkpoint, prev_k, 0);
    }
    __pipeline_commit();

    if (first_work) {
      cudaGridDependencySynchronize();
      first_work = false;
    }

    if (active) {
      load_x<input_t, NPREDICTED, DIM, DIM, DSTATE, /*IS_FIRST=*/true>(
          smem, params, lane, warp, 0, head, group_idx, outer, seq_len, 0);
      if (warp == 2) load_cumAdt_async(smem, params, lane, seq, head, seq_len, 0);
      if (warp == 0)
        load_cb_8bit_main</*IS_OLD=*/false, input_t, NPREDICTED, MAX_WINDOW>(
            smem, params, lane, seq, head, must_checkpoint);
      if (warp == 1)
        load_cb_8bit_main</*IS_OLD=*/true, input_t, NPREDICTED, MAX_WINDOW>(
            smem, params, lane, seq, head, must_checkpoint);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    if (active) {
      // The quantized helpers use the monolithic old-cumAdt and decay arrays;
      // derive them from the split main's old-dt and cumAdt inputs.
      if (warp == 0) {
        float scan = lane < prev_k ? smem.old_dt[lane] : 0.f;
#pragma unroll
        for (int offset = 1; offset < MAX_WINDOW; offset <<= 1) {
          float const other = __shfl_up_sync(constants::MASK_ALL_LANES, scan, offset);
          if (lane >= offset) scan += other;
        }
        if (lane < MAX_WINDOW) smem.old_cumAdt[lane] = A_val * scan;
        if (lane < seq_len) smem.decay[lane] = __expf(smem.cumAdt[lane]);
      }
      __syncthreads();

      int64_t const out_seq_base = outer * params.out_stride_seq;
      if (must_checkpoint) {
        ssu_checkpoint_8bit<input_t, state_t, NPREDICTED, DIM, DIM, DSTATE, NUM_WARPS,
                            PHILOX_ROUNDS>(smem, params, warp, lane, prev_k, 0, out_seq_base, head,
                                           raw_slot, D_val, seq_len);
      } else {
        ssu_nocheckpoint_8bit<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, DIM, DSTATE,
                              NUM_WARPS>(smem, params, warp, lane, prev_k, 0, out_seq_base, head,
                                          raw_slot, D_val, seq_len);
      }
      store_old_x<input_t, NPREDICTED, DIM, DIM>(smem, params, warp, lane, 0, head, raw_slot,
                                                 ring_start, prev_k, seq_len);
    }
    __syncthreads();  // all consumers are done before the next work-unit reuses smem
  }
}

}  // namespace flashinfer::mamba::checkpointing

#endif  // FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_MAIN_8BIT_CUH_
