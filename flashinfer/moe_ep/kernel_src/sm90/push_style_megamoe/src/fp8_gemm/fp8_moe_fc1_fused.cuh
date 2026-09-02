/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-License-Identifier: MIT
 *
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <deep_gemm/fp8_gemm_impl.cuh>

#include "fp8_moe_scheduler.cuh"

namespace flashinfer::sm90_push_fp8 {

using namespace deep_gemm;

template <typename Scheduler, typename = void>
struct IsFp8MoeFc1Scheduler {
  static constexpr bool value = false;
};

template <typename Scheduler>
struct IsFp8MoeFc1Scheduler<Scheduler, decltype((void)Scheduler::kIsFp8MoeFc1Scheduler, void())> {
  static constexpr bool value = Scheduler::kIsFp8MoeFc1Scheduler;
};

// Fused SwiGLU + 1x128-quant epilogue over gate/up-interleaved weights;
// bit-exact with the unfused pipeline.
// The unfused activation kernel's TU is built with -use_fast_math, so the
// epilogue divisions use __fdividef to match its div.approx.f32; a plain
// `/` here would be div.rn.f32 and flip rounding-boundary values.
template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages, uint32_t kNumTMAThreads,
          uint32_t kNumMathThreadsPerGroup, uint32_t kNumTMAMulticast, typename SchedulerType,
          typename InputType>
__global__ void __launch_bounds__(
    get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M), 1)
    fp8_gemm_kernel_sm90_push_fc1_fused(__nv_fp8_e4m3* gmem_d_fp8, int64_t d_rows,
                                        float* gmem_sfa_out, int64_t sfa_out_stride,
                                        float* scales_b, InputType problem_input,
                                        __grid_constant__ const CUtensorMap tensor_map_a,
                                        __grid_constant__ const CUtensorMap tensor_map_b,
                                        __grid_constant__ const CUtensorMap tensor_map_scales_a) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ == 900))
  DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
  DG_STATIC_ASSERT(BLOCK_N == BLOCK_K, "Fused epilogue needs one N tile == one quant block");
  DG_STATIC_ASSERT(kNumTMAMulticast == 1, "Pair B tiles differ per CTA; no TMA multicast");
  DG_STATIC_ASSERT(IsFp8MoeFc1Scheduler<SchedulerType>::value,
                   "The fused FP8 FC1 kernel requires Fp8MoeFc1Scheduler");

  // Types
  using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
  using Barrier = cutlass::arch::ClusterTransactionBarrier;

  // smem: bf16 D staging replaced by bf16 gate staging + fp8 out staging;
  // A/B/scales_a stages match the unfused kernel byte for byte
  static constexpr uint32_t SMEM_ACT_SIZE = BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16);
  static constexpr uint32_t SMEM_OUT_SIZE = BLOCK_M * BLOCK_N * sizeof(__nv_fp8_e4m3);
  static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
  static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
  static constexpr uint32_t SMEM_SCALES_A_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
  static constexpr uint32_t SHAPE_K_SCALES = ceil_div(SHAPE_K, BLOCK_K);
  // TWO scales_b rows per task: [0, SHAPE_K_SCALES) gate, then the up row.
  static constexpr uint32_t SMEM_SCALES_B_SIZE =
      ceil_div<uint32_t>(2 * SHAPE_K_SCALES * sizeof(float), sizeof(Barrier)) * sizeof(Barrier);
  DG_STATIC_ASSERT((SMEM_ACT_SIZE + SMEM_OUT_SIZE) % 1024 == 0,
                   "Shared memory of A/B must be aligned to 1024 bytes");

  // Configs
  constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K;
  constexpr uint32_t kNumThreads =
      get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
  constexpr uint32_t kNumMathThreads = kNumThreads - kNumTMAThreads;
  constexpr uint32_t kNumIterations = ceil_div(SHAPE_K, kFullKOfAllStages);
  uint32_t const warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
  uint32_t const lane_idx = get_lane_id();

  if (threadIdx.x == kNumMathThreads) {
    cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_a));
    cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_b));
    cute::prefetch_tma_descriptor(
        reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_scales_a));
  }
  __syncwarp();

  extern __shared__ __align__(1024) uint8_t smem_buffer[];

  auto smem_act = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
  auto smem_out = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_ACT_SIZE);
  __nv_fp8_e4m3* smem_a[kNumStages];
  __nv_fp8_e4m3* smem_b[kNumStages];
  float* smem_scales_a[kNumStages];
  float* smem_scales_b;

  Barrier* full_barriers[kNumStages];
  Barrier* empty_barriers[kNumStages];

#pragma unroll
  for (int i = 0; i < kNumStages; ++i) {
    smem_a[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_ACT_SIZE + SMEM_OUT_SIZE +
                                                 i * SMEM_A_SIZE_PER_STAGE);
    smem_b[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_ACT_SIZE + SMEM_OUT_SIZE +
                                                 kNumStages * SMEM_A_SIZE_PER_STAGE +
                                                 i * SMEM_B_SIZE_PER_STAGE);
    smem_scales_a[i] =
        reinterpret_cast<float*>(smem_buffer + SMEM_ACT_SIZE + SMEM_OUT_SIZE +
                                 kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) +
                                 i * SMEM_SCALES_A_SIZE_PER_STAGE);
  }
  smem_scales_b = reinterpret_cast<float*>(
      smem_buffer + SMEM_ACT_SIZE + SMEM_OUT_SIZE +
      kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE));

  auto barrier_start_ptr =
      reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_scales_b) + SMEM_SCALES_B_SIZE);
#pragma unroll
  for (int i = 0; i < kNumStages; ++i) {
    full_barriers[i] = barrier_start_ptr + i;
    empty_barriers[i] = barrier_start_ptr + kNumStages + i;
  }

  // Initialize barriers (kNumTMAMulticast == 1 by static assert)
  if (threadIdx.x == kNumMathThreads) {
#pragma unroll
    for (int i = 0; i < kNumStages; ++i) {
      full_barriers[i]->init(1);
      empty_barriers[i]->init(kNumMathThreads / 32);
    }

    cutlass::arch::fence_view_async_shared();
  }

  __syncthreads();

  struct DivisibleK {};

  struct NotDivisibleK {};

  auto launch_k_iterations = [](auto const& func) {
    if constexpr (SHAPE_K % kFullKOfAllStages == 0) {
      for (int k_iter = 0; k_iter < kNumIterations; ++k_iter) func(k_iter, DivisibleK{});
    } else {
      for (int k_iter = 0; k_iter < kNumIterations - 1; ++k_iter) func(k_iter, DivisibleK{});
      func(kNumIterations - 1, NotDivisibleK{});
    }
  };

  constexpr int kNumTMARegisters = 40;
  constexpr int kNumMathRegisters = 232;

  // (m_block, PAIR) tasks consume 2 * kNumIterations barrier rounds, so
  // the parity term uses one flat, strictly increasing iteration counter.
  uint32_t m_block_idx, pair_idx;
  auto scheduler = SchedulerType(problem_input);

  if (threadIdx.x >= kNumMathThreads) {
    cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

    if (threadIdx.x == kNumMathThreads) {
      while (scheduler.get_next_block(m_block_idx, pair_idx)) {
#pragma unroll
        for (uint32_t phase = 0; phase < 2; ++phase) {
          launch_k_iterations([&](int k_iter, auto type) {
            constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
            constexpr int kNumInnerStages =
                kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
            DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

#pragma unroll
            for (uint32_t s = 0; s < kNumInnerStages; ++s) {
              empty_barriers[s]->wait(
                  ((scheduler.current_iter * 2 + phase) * kNumIterations + k_iter + 1) & 1);

              auto& full_barrier = *full_barriers[s];
              int k_idx = k_iter * kFullKOfAllStages + s * BLOCK_K;
              // A tile + A scales: SAME coordinates in both phases (the
              // second pass is the L2-reuse bandwidth win of pairing).
              tma_copy<kNumTMAMulticast>(&tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                         smem_a[s], k_idx, scheduler.get_global_m_idx(m_block_idx));
              tma_copy<kNumTMAMulticast>(
                  &tensor_map_scales_a, reinterpret_cast<uint64_t*>(&full_barrier),
                  smem_scales_a[s], scheduler.get_global_scales_a_idx(m_block_idx),
                  k_idx / BLOCK_K);

              // B tile: gate (phase 0) or up (phase 1) block of the pair
              tma_copy(&tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier), smem_b[s], k_idx,
                       scheduler.get_global_n_idx_phase(pair_idx, phase));
              full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE +
                                                SMEM_SCALES_A_SIZE_PER_STAGE);
            }

#pragma unroll
            for (uint32_t s = kNumInnerStages; s < kNumStages; ++s) {
              empty_barriers[s]->wait(
                  ((scheduler.current_iter * 2 + phase) * kNumIterations + k_iter + 1) & 1);
              full_barriers[s]->arrive();
            }
          });
        }
      }
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

    // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
    auto const math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / kNumMathThreadsPerGroup, 0);
    // WGMMA accum fragment rows of this thread within the BLOCK_M tile; the
    // fragment columns are i * 8 + col_base and i * 8 + col_base + 1.
    auto const r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8;
    auto const col_base = (lane_idx % 4) * 2;
    constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    constexpr uint32_t SHAPE_N_OUT = SHAPE_N / 2;  // de-interleaved output width (I)

    // Empty barrier arrival (multicast == 1: lane 0 of every math warp)
    auto empty_barrier_arrive = [&](int s) {
      lane_idx == 0 ? empty_barriers[s]->arrive() : void();
    };

    // Persistently schedule over (m_block, pair) tasks
    while (scheduler.get_next_block(m_block_idx, pair_idx)) {
      // the pair's two scales_b rows are gmem-adjacent by the interleaved
      // layout; BLOCK_N == BLOCK_K makes scale B uniform per tile
      if (threadIdx.x >= 32) {
        auto local_scales_b =
            scales_b +
            static_cast<uint64_t>(scheduler.get_scales_b_row_gate(pair_idx)) * SHAPE_K_SCALES;
#pragma unroll
        for (uint32_t i = threadIdx.x - 32; i < 2 * SHAPE_K_SCALES; i += kNumMathThreads - 32)
          st_shared(smem_scales_b + i, __ldg(local_scales_b + i));
      }
      cutlass::arch::NamedBarrier(kNumMathThreads).sync();

#pragma unroll
      for (uint32_t phase = 0; phase < 2; ++phase) {
        // Accumulation for WGMMA or CUDA promotion (fresh per phase)
        float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum] = {0};

        launch_k_iterations([&](int k_iter, auto type) {
          constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
          constexpr int kNumInnerStages =
              kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
          DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

#pragma unroll
          for (int s = 0; s < kNumInnerStages; ++s) {
            // Read this phase's B scale (uniform across the 128-wide tile)
            float scale_b_0 =
                ld_shared(smem_scales_b + phase * SHAPE_K_SCALES + k_iter * kNumStages + s);

            full_barriers[s]->wait(
                ((scheduler.current_iter * 2 + phase) * kNumIterations + k_iter) & 1);

            // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next
            // scheduled block polluting the results
            auto scale_a_0 = ld_shared(smem_scales_a[s] + r_0),
                 scale_a_1 = ld_shared(smem_scales_a[s] + r_1);

#pragma unroll
            for (int i = 0; i < WGMMA::kNumAccum; ++i) warpgroup_fence_operand(accum[i]);
            warpgroup_arrive();
#pragma unroll
            for (int k = 0; k < BLOCK_K / WGMMA::K; ++k) {
              auto desc_a =
                  make_smem_desc(smem_a[s] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
              auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);
              WGMMA::wgmma(desc_a, desc_b, accum, k);
            }
            warpgroup_commit_batch();
#pragma unroll
            for (int i = 0; i < WGMMA::kNumAccum; ++i) warpgroup_fence_operand(accum[i]);
            warpgroup_wait<0>();

            empty_barrier_arrive(s);

            // Promote with scales -- VERBATIM the unfused kernel's uniform-
            // scale-B path (the bitwise contract; do not reorder).
            float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
#pragma unroll
            for (int i = 0; i < WGMMA::kNumAccum / 4; ++i) {
              final_accum[i * 4 + 0] += scale_0_0 * accum[i * 4 + 0];
              final_accum[i * 4 + 1] += scale_0_0 * accum[i * 4 + 1];
              final_accum[i * 4 + 2] += scale_1_0 * accum[i * 4 + 2];
              final_accum[i * 4 + 3] += scale_1_0 * accum[i * 4 + 3];
            }
          }

#pragma unroll
          for (uint32_t s = kNumInnerStages; s < kNumStages; ++s) {
            full_barriers[s]->wait(
                ((scheduler.current_iter * 2 + phase) * kNumIterations + k_iter) & 1);
            empty_barrier_arrive(s);
          }
        });

        auto smem_act2 = reinterpret_cast<__nv_bfloat162*>(smem_act);
        if (phase == 0) {
          // stage gate as bf16 (the contract's FIRST rounding point);
          // each slot is written and read by the same thread
#pragma unroll
          for (int i = 0; i < WGMMA::kNumAccum / 4; ++i) {
            uint32_t c2 = (i * 8 + col_base) >> 1;
            smem_act2[r_0 * (BLOCK_N / 2) + c2] =
                __floats2bfloat162_rn(final_accum[i * 4 + 0], final_accum[i * 4 + 1]);
            smem_act2[r_1 * (BLOCK_N / 2) + c2] =
                __floats2bfloat162_rn(final_accum[i * 4 + 2], final_accum[i * 4 + 3]);
          }
        } else {
          // ---- fused SwiGLU + quant epilogue (phase 1: up tile ready) ----
          auto m_global_base = static_cast<int64_t>(scheduler.get_global_m_idx(m_block_idx));
          bool row0_valid = m_global_base + r_0 < scheduler.m_boundary;
          bool row1_valid = m_global_base + r_1 < scheduler.m_boundary;

          float amax_0 = 0.0f, amax_1 = 0.0f;
          __nv_bfloat162 t_0[WGMMA::kNumAccum / 4], t_1[WGMMA::kNumAccum / 4];
#pragma unroll
          for (int i = 0; i < WGMMA::kNumAccum / 4; ++i) {
            uint32_t c2 = (i * 8 + col_base) >> 1;
            // gate: staged bf16 -> f32 (matches FA reading h's gate half)
            float2 g_0 = __bfloat1622float2(smem_act2[r_0 * (BLOCK_N / 2) + c2]);
            float2 g_1 = __bfloat1622float2(smem_act2[r_1 * (BLOCK_N / 2) + c2]);
            // up: the contract's OTHER bf16 rounding point, then back to f32
            float u_00 = __bfloat162float(__float2bfloat16(final_accum[i * 4 + 0]));
            float u_01 = __bfloat162float(__float2bfloat16(final_accum[i * 4 + 1]));
            float u_10 = __bfloat162float(__float2bfloat16(final_accum[i * 4 + 2]));
            float u_11 = __bfloat162float(__float2bfloat16(final_accum[i * 4 + 3]));
            // silu(g) * u rounded through bf16, the unfused kernel's exact
            // expression; __fdividef per the cross-TU division contract
            // (see the kernel header)
            __nv_bfloat16 t_00 = __float2bfloat16(__fdividef(g_0.x, 1.0f + __expf(-g_0.x)) * u_00);
            __nv_bfloat16 t_01 = __float2bfloat16(__fdividef(g_0.y, 1.0f + __expf(-g_0.y)) * u_01);
            __nv_bfloat16 t_10 = __float2bfloat16(__fdividef(g_1.x, 1.0f + __expf(-g_1.x)) * u_10);
            __nv_bfloat16 t_11 = __float2bfloat16(__fdividef(g_1.y, 1.0f + __expf(-g_1.y)) * u_11);
            t_0[i] = __nv_bfloat162(t_00, t_01);
            t_1[i] = __nv_bfloat162(t_10, t_11);
            amax_0 =
                fmaxf(amax_0, fmaxf(fabsf(__bfloat162float(t_00)), fabsf(__bfloat162float(t_01))));
            amax_1 =
                fmaxf(amax_1, fmaxf(fabsf(__bfloat162float(t_10)), fabsf(__bfloat162float(t_11))));
          }
          // fmaxf matches the unfused kernel's reduction bit for bit: same
          // NaN drop, and it equals atomicMax-on-int-bits over non-negatives
          amax_0 = fmaxf(amax_0, __shfl_xor_sync(0xffffffff, amax_0, 1));
          amax_0 = fmaxf(amax_0, __shfl_xor_sync(0xffffffff, amax_0, 2));
          amax_1 = fmaxf(amax_1, __shfl_xor_sync(0xffffffff, amax_1, 1));
          amax_1 = fmaxf(amax_1, __shfl_xor_sync(0xffffffff, amax_1, 2));
          // __fdividef per the cross-TU division contract; this value is
          // also the sfa2 payload FC2 consumes
          float sc_0 = amax_0 > 0.0f ? __fdividef(amax_0, 448.0f) : 1.0f;
          float sc_1 = amax_1 > 0.0f ? __fdividef(amax_1, 448.0f) : 1.0f;

          // sfa column = pad_base[g] + (row - offsets[g]), stride P; the
          // guard only fires for malformed direct-FFI offsets
          int64_t sfa_col_base = scheduler.m_padded_4_offset + m_block_idx * BLOCK_M;
          if (lane_idx % 4 == 0) {
            if ((row0_valid && sfa_col_base + r_0 >= sfa_out_stride) ||
                (row1_valid && sfa_col_base + r_1 >= sfa_out_stride)) {
              printf("fc1_fused: sfa column %lld >= stride P %lld (bad offsets?)\n",
                     static_cast<long long>(sfa_col_base + r_1),
                     static_cast<long long>(sfa_out_stride));
              asm volatile("trap;");
            }
            if (row0_valid) gmem_sfa_out[pair_idx * sfa_out_stride + sfa_col_base + r_0] = sc_0;
            if (row1_valid) gmem_sfa_out[pair_idx * sfa_out_stride + sfa_col_base + r_1] = sc_1;
          }

          // Quantize (DIVISION by sc -- the contract's rounding form;
          // __fdividef for the same instruction-contract reason as above)
          // into the fp8 staging tile; 2 adjacent columns pack per u16.
          auto smem_out_u16 = reinterpret_cast<uint16_t*>(smem_out);
#pragma unroll
          for (int i = 0; i < WGMMA::kNumAccum / 4; ++i) {
            uint32_t c2 = (i * 8 + col_base) >> 1;
            float2 v_0 = __bfloat1622float2(t_0[i]);
            float2 v_1 = __bfloat1622float2(t_1[i]);
            // memcpy: a uint16_t read of q[2] would be alignment UB
            __nv_fp8_e4m3 q[2];
            uint16_t pk;
            q[0] = __nv_fp8_e4m3(fminf(fmaxf(__fdividef(v_0.x, sc_0), -448.0f), 448.0f));
            q[1] = __nv_fp8_e4m3(fminf(fmaxf(__fdividef(v_0.y, sc_0), -448.0f), 448.0f));
            memcpy(&pk, q, sizeof(pk));
            smem_out_u16[r_0 * (BLOCK_N / 2) + c2] = pk;
            q[0] = __nv_fp8_e4m3(fminf(fmaxf(__fdividef(v_1.x, sc_1), -448.0f), 448.0f));
            q[1] = __nv_fp8_e4m3(fminf(fmaxf(__fdividef(v_1.y, sc_1), -448.0f), 448.0f));
            memcpy(&pk, q, sizeof(pk));
            smem_out_u16[r_1 * (BLOCK_N / 2) + c2] = pk;
          }
          cutlass::arch::NamedBarrier(kNumMathThreads).sync();  // staging complete

          // Coalesced writeout: BLOCK_N bytes (8 x int4) per row, one row per
          // warp pass. Rows ascend, so break at the group boundary is safe.
          constexpr uint32_t kInt4PerLine = BLOCK_N / 16;
          int4 const* smem_out_int4 = reinterpret_cast<int4 const*>(smem_out);
          for (uint32_t line = warp_idx; line < BLOCK_M; line += kNumMathWarps) {
            if (m_global_base + line >= scheduler.m_boundary) break;
            // device-side D-capacity backstop (one predictable compare per
            // row; never fires for well-formed offsets)
            if (m_global_base + line >= d_rows) {
              printf("fc1_fused: output row %lld >= d_fp8 rows %lld (bad offsets?)\n",
                     static_cast<long long>(m_global_base + line), static_cast<long long>(d_rows));
              asm volatile("trap;");
            }
            if (lane_idx < kInt4PerLine) {
              int4* g_addr =
                  reinterpret_cast<int4*>(gmem_d_fp8 + (m_global_base + line) * SHAPE_N_OUT +
                                          pair_idx * BLOCK_N) +
                  lane_idx;
              *g_addr = smem_out_int4[line * kInt4PerLine + lane_idx];
            }
            __syncwarp();
          }
          // Order this task's staging reads before the next task's writes
          // (the act staging is single-owner, but the out staging is read
          // cooperatively above).
          cutlass::arch::NamedBarrier(kNumMathThreads).sync();
        }
      }
    }
  }
#else
  if (blockIdx.x == 0 and threadIdx.x == 0)
    DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

}  // namespace flashinfer::sm90_push_fp8
