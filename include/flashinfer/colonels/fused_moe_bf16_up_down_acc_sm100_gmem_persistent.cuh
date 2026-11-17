#pragma once

#include <cooperative_groups.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda/atomic>
#include <cuda/ptx>
#include <cuda/std/limits>
#include <type_traits>

#include "fused_moe_activation.cuh"
#include "fused_moe_helpers.cuh"

using bf16 = __nv_bfloat16;
namespace cg = cooperative_groups;

namespace moe_sm100_gmem_persistent {
template <int BLOCK_M, int PIPELINE_STAGES, int CLUSTER_SIZE, bool USE_RED_STORE,
          moe_activation::Kind ACTIVATION, int MMA_CTA_GROUP = BLOCK_M == 256 ? 2 : 1,
          int ROUTED_BLOCK_M = BLOCK_M, bool WEIGHT_EVICTION = false, int EPILOGUE_WARPS = 4,
          int PRODUCER_WARPS = 4, int EXTRA_BARRIERS = 64>
struct constants {
  static_assert(EPILOGUE_WARPS == 4 || EPILOGUE_WARPS == 8);
  static_assert(PRODUCER_WARPS == 4 || PRODUCER_WARPS == 8);
  static constexpr bool EVICT_WEIGHTS = WEIGHT_EVICTION;
  static constexpr int BM = BLOCK_M;
  // Group2 splits token rows and weight columns across adjacent CTAs.
  static constexpr int CTA_GROUP = MMA_CTA_GROUP;
  static constexpr int ROUTING_BM = ROUTED_BLOCK_M;
  static constexpr bool SPLIT_N_TMEM = CTA_GROUP == 2 && BM == 128;
  static constexpr int LOCAL_M = BM / CTA_GROUP;
  static constexpr bool RED_STORE = USE_RED_STORE;
  static constexpr int STAGES = PIPELINE_STAGES;
  static constexpr int BK = 64;
  // Each MMA group computes an N tile; matching M halves multicast X.
  static constexpr int WN = 4;
  static constexpr int CLUSTERS = CLUSTER_SIZE;
  static_assert(CLUSTERS % CTA_GROUP == 0);
  static constexpr int N_GROUPS = CLUSTERS / CTA_GROUP;
  static_assert(CLUSTERS == 1 || CLUSTERS == 2 || CLUSTERS == 4 || CLUSTERS == 8 || CLUSTERS == 16);
  static constexpr int MMA_THREADS = WN * 32;
  // Eight producer warps reduce per-warp gather4 issue serialization.
  static constexpr int PRODUCER_THREADS = PRODUCER_WARPS * 32;
  static constexpr int EPILOGUE_THREADS = EPILOGUE_WARPS * 32;
  // Warp 0 issues MMA and warp 1 manages TMEM; reuse idle warp 3 for CLC.
  static constexpr int SCHEDULER_WARP = 3;
  static constexpr int BLOCK_THREADS = MMA_THREADS + PRODUCER_THREADS + EPILOGUE_THREADS;
  static constexpr int EPILOGUE_WARPGROUPS = EPILOGUE_THREADS / 128;

  using Activation = moe_activation::Traits<ACTIVATION>;
  using ActivationParams = typename Activation::Params;
  static constexpr bool IS_GATED = Activation::GATED;
  // Ratio of up to down projection, gated has double the size of up projeciton
  // relative to down projection because it merges up projection and gate
  static constexpr int UD_RATIO = IS_GATED ? 2 : 1;
  static_assert(EPILOGUE_THREADS == 128 || EPILOGUE_THREADS == 256 || EPILOGUE_THREADS == 512);
  static_assert(BLOCK_THREADS <= 1024);

  static constexpr int MMA_N_UP = 256;
  static constexpr int TMEM_COLS = 512;
  static constexpr int EP_BUFFERS = TMEM_COLS / MMA_N_UP;
  // Blackwell's opt-in dynamic shared memory limit per block.
  static constexpr size_t SM100_MAX_SMEM = 232448;
  static constexpr uint32_t S_BITS_UP = 3;
  static constexpr uint32_t S_MODE_UP = 4 - S_BITS_UP;
  static constexpr uint32_t S_BITS_DOWN = 3;
  static constexpr uint32_t S_MODE_DOWN = 4 - S_BITS_DOWN;
  static constexpr int WS = BK * MMA_N_UP / CTA_GROUP;
  static constexpr int XS = BK * LOCAL_M;
  static constexpr int MMA_N_DOWN = MMA_N_UP;
  static constexpr int BN2 = MMA_N_DOWN;
  // Shift consecutive rows by four banks for conflict-free 16-byte lane stores.
  // Keep each row 16-byte aligned for the bulk reduction source.
  static constexpr int OUTPUT_STRIDE = BN2 / EPILOGUE_WARPGROUPS + 8;
  static constexpr int BK2 = BK;
  static constexpr int X_BLOCK_SIZE = 16;
  static_assert(LOCAL_M % X_BLOCK_SIZE == 0);
  static constexpr int TMA_TILES_DOWN = BK2 / 64;
  static constexpr int TILE_COLS_UP = MMA_N_UP / CTA_GROUP;
  static constexpr int TILE_INNER_DN = BK2 / TMA_TILES_DOWN;
  // Dense M128/group2 stores each N half in a different pair of TMEM warp partitions.
  static constexpr int EP_ROWS_PER_WARP = SPLIT_N_TMEM ? 32 : LOCAL_M / 4;
  static constexpr int EP_COLS_PER_WARP = SPLIT_N_TMEM ? MMA_N_UP / 2 : MMA_N_UP;
  static constexpr int EP_COLS_PER_WARPGROUP = EP_COLS_PER_WARP / EPILOGUE_WARPGROUPS;
  static constexpr int TPT = EP_ROWS_PER_WARP / 16;
  static constexpr uint32_t idesc_up = build_idesc<BM, MMA_N_UP, CTA_GROUP>();
  static constexpr uint32_t idesc_down = build_idesc<BM, MMA_N_DOWN, CTA_GROUP>();
  static constexpr int GATHERS_PER_CTA = LOCAL_M / (4 * N_GROUPS);
  static constexpr uint32_t TMA_RANK_X = 2;
  static constexpr uint32_t TMA_RANK_S = 2;
  static constexpr int SCRATCH_ROWS_PER_SM = std::min(ROUTING_BM, 128);
  static_assert(LOCAL_M <= SCRATCH_ROWS_PER_SM);
  // Every multicast peer must execute the same number of pipeline iterations.
  // Slower M128 fallback workers must not reserve a second pending tile.
  static constexpr int CLC_STAGES = SPLIT_N_TMEM ? 1 : 2;
  // Each CTA's epilogue, scheduler, and producers, plus one MMA leader per group.
  static constexpr int CLC_READERS =
      CLUSTERS * (EPILOGUE_THREADS + 1 + PRODUCER_THREADS) + N_GROUPS;

  struct smem_up {
    alignas(1024) bf16 w[STAGES * WS];
    alignas(1024) bf16 x[STAGES * XS];
  };

  struct smem_down_tma {
    alignas(1024) bf16 w[STAGES * WS];
    alignas(1024) bf16 x[STAGES * XS];
    // Each epilogue warp stages 16 rows for bulk output reduction.
    alignas(128) bf16 output[EPILOGUE_THREADS / 2 * OUTPUT_STRIDE];
  };

  using smem_down = std::conditional_t<RED_STORE, smem_up, smem_down_tma>;

  static constexpr int MAX_EXTRA_BARRIERS = EXTRA_BARRIERS;
  struct smem_persistent {
    alignas(8) uint64_t bar[2 * STAGES + 2 * EP_BUFFERS];
    alignas(16) int4 clc_response[CLC_STAGES];
    alignas(8) uint64_t clc_full[CLC_STAGES];
    alignas(8) uint64_t clc_empty[CLC_STAGES];
    uint32_t tmem_addr;
    alignas(8) uint64_t e_to_p_bar[MAX_EXTRA_BARRIERS];
  };

  static constexpr size_t smem_persistent_bytes = sizeof(smem_persistent);
  static constexpr size_t smem_persistent_bar_bytes = sizeof(decltype(smem_persistent::bar));
  static constexpr size_t smem_persistent_tmem_addr_bytes =
      sizeof(decltype(smem_persistent::tmem_addr));
  static constexpr size_t smem_up_bytes = sizeof(smem_up);
  static constexpr size_t smem_up_w_bytes = sizeof(decltype(smem_up::w));
  static constexpr size_t smem_up_x_bytes = sizeof(decltype(smem_up::x));
  static constexpr size_t smem_down_bytes = sizeof(smem_down);
  static constexpr size_t smem_down_w_bytes = sizeof(decltype(smem_down::w));
  static constexpr size_t smem_down_x_bytes = sizeof(decltype(smem_down::x));
  // Round persistent storage up to the overlay's 1024-byte alignment.
  static constexpr size_t OVERLAY_OFF = (smem_persistent_bytes + 1023) & ~size_t(1023);
  static constexpr size_t sMemSize = OVERLAY_OFF + std::max(smem_up_bytes, smem_down_bytes);

  static constexpr int MAX_STAGES = [] {
    // Each pipeline stage adds two barriers to the persistent storage.
    constexpr size_t metadata = smem_persistent_bytes - 2 * STAGES * sizeof(uint64_t);
    constexpr size_t stage_bytes = (WS + XS) * sizeof(bf16);
    constexpr size_t output_bytes = smem_down_bytes - smem_up_bytes;
    for (int stages = 1;; ++stages) {
      const size_t overlay = (metadata + 2 * stages * sizeof(uint64_t) + 1023) & ~size_t(1023);
      if (overlay + stages * stage_bytes + output_bytes > SM100_MAX_SMEM) return stages - 1;
    }
  }();
};

template <int STAGES>
struct x_tile_cursor {
  int stage = 0;
  bool reversed = false;

  __device__ __forceinline__ void advance() {
    stage += reversed ? -1 : 1;
    if (stage == STAGES)
      stage = 0;
    else if (stage == -1)
      stage = STAGES - 1;
  }

  __device__ __forceinline__ void reverse() {
    reversed = !reversed;
    // The next sweep starts at the tile consumed last.
    advance();
  }
};

template <typename C, bool REUSE_X>
__device__ __forceinline__ void fused_moe_bf16_wgmma_up_down_acc_body(
    const CUtensorMap& tensor_map_x, const CUtensorMap& tensor_map_w,
    const CUtensorMap& tensor_map_w2, const CUtensorMap& tensor_map_s,
    const CUtensorMap& tensor_map_s_partial, bf16* __restrict__ scratchpad, bf16* __restrict__ out,
    const int* __restrict__ sorted_token_ids, const int* __restrict__ expert_ids,
    const int* __restrict__ num_tokens_post_padded, const float* __restrict__ topk_weights,
    const int top_k, int M, int N, int K, float scaling_factor,
    typename C::ActivationParams activation_params) {
  // Keep Blackwell instructions out of other architectures in a shared fatbin.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 1000
  const int N2 = K;
  const int K2 = N / C::UD_RATIO;
  const int n_stages_up = K / C::BK;
  const int n_stages_down = N2 / C::BN2;
  const int k_stages_down = K2 / C::BK2;
  const int SCRATCH_COLS = K2;
  // Multicast peers execute the same number of pipeline iterations.
  const int UP_TILES =
      ((N + C::MMA_N_UP * C::N_GROUPS - 1) / (C::MMA_N_UP * C::N_GROUPS)) * C::N_GROUPS;
  const int LOCAL_UP_TILES = UP_TILES / C::N_GROUPS;
  const size_t OVERLAY_OFF = (offsetof(typename C::smem_persistent, e_to_p_bar) +
                              LOCAL_UP_TILES * sizeof(uint64_t) + 1023) &
                             ~size_t(1023);
  const int total_tokens = num_tokens_post_padded[0];
  const int total_blocks = const_ceil(total_tokens / float(C::BM));
  // Keep the same wave count while avoiding a sparsely populated final wave.
  // const int waves = const_ceil(total_blocks / float(max_workers));
  // const int persistent_stride = const_ceil(total_blocks / float(waves));
  const uint32_t block_start = blockIdx.x / C::CLUSTERS;
  if (block_start >= total_blocks) return;
  int cta_rank = 0;
  if constexpr (C::CLUSTERS > 1) {
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
  }
  const int group_rank = cta_rank % C::CTA_GROUP;
  const int n_group = cta_rank / C::CTA_GROUP;
  const uint16_t group_mask = uint16_t(((1u << C::CTA_GROUP) - 1) << (n_group * C::CTA_GROUP));
  const uint16_t x_mask = C::CTA_GROUP == 2
                              ? uint16_t((0x5555u << group_rank) & ((1u << C::CLUSTERS) - 1))
                              : uint16_t((1u << C::CLUSTERS) - 1);

  const int lane_id = threadIdx.x % 32;
  const bool is_scheduler = threadIdx.x / 32 == C::SCHEDULER_WARP;
  const bool is_epilogue =
      threadIdx.x < C::MMA_THREADS + C::PRODUCER_THREADS + C::EPILOGUE_THREADS &&
      threadIdx.x >= C::MMA_THREADS + C::PRODUCER_THREADS;
  const bool is_producer =
      threadIdx.x >= C::MMA_THREADS && threadIdx.x < C::MMA_THREADS + C::PRODUCER_THREADS;
  const bool is_mma = threadIdx.x < C::MMA_THREADS && !is_scheduler;
  const int warp_id = threadIdx.x / 32;

  extern __shared__ __align__(1024) uint8_t sh[];

  auto& smem_p = *reinterpret_cast<typename C::smem_persistent*>(sh);
  uint64_t* bar = smem_p.bar;
  // EPILOGUE BUFFERS
  uint64_t* e_to_c_bar = bar + 2 * C::STAGES;
  uint64_t* c_to_e_bar = e_to_c_bar + C::EP_BUFFERS;
  uint64_t* e_to_p_bar = smem_p.e_to_p_bar;
  uint64_t* clc_empty_bar = smem_p.clc_empty;
  uint64_t* clc_full_bar = smem_p.clc_full;

  typename C::smem_up& s = *reinterpret_cast<typename C::smem_up*>(sh + OVERLAY_OFF);
  typename C::smem_down& s_d = *reinterpret_cast<typename C::smem_down*>(sh + OVERLAY_OFF);

  // Since it reduces occupancy to 1, is there a reason to request less?
  if (threadIdx.x == 0) {
    for (int i = 0; i < C::STAGES; i++) {
      init_barrier(&bar[i], 1, 0);
      init_barrier(&bar[i + C::STAGES], C::N_GROUPS, 0);
    }
    for (int i = 0; i < C::EP_BUFFERS; i++) {
      init_barrier(e_to_c_bar + i, C::CTA_GROUP * C::EPILOGUE_THREADS, 0);
      init_barrier(c_to_e_bar + i, 1, 0);
    }
    for (int i = 0; i < LOCAL_UP_TILES; i++) {
      init_barrier(e_to_p_bar + i, C::EPILOGUE_THREADS, 0);
    }
    for (int i = 0; i < C::CLC_STAGES; i++) {
      init_barrier(clc_full_bar + i, 1, 0);
      expect_bytes(clc_full_bar + i, sizeof(int4));
      if (cta_rank == 0) {
        init_barrier(clc_empty_bar + i, C::CLC_READERS, 0);
      }
    }
  } else if (warp_id == 1) {
    tmem_alloc<C::CTA_GROUP>(__cvta_generic_to_shared(&smem_p.tmem_addr), C::TMEM_COLS);
  }

  if constexpr (C::CLUSTERS > 1) {
    cg::this_cluster().sync();
  } else {
    __syncthreads();
  }
  // Each CTA loads and multicasts the scratch tiles it publishes.
  uint32_t scratch_sm;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(scratch_sm));
  const uint32_t scratch_row_base = scratch_sm * C::SCRATCH_ROWS_PER_SM;
  uint32_t tmem_addr = smem_p.tmem_addr;
  int smem_stage = 0;
  int p = 0;
  x_tile_cursor<C::STAGES> x_cursor;
  int epilogue_stage = 0;
  int e_p = 0;
  int scratch_phase = 0;
  int clc_stage = 0;
  int c_f_p = 0;

  uint32_t local_block = block_start;
  uint32_t is_valid = local_block < total_blocks;

  if (is_mma && threadIdx.x == 0) {
    for (int i = 0; i < C::STAGES; i++) arrive(&bar[C::STAGES + i], C::N_GROUPS);
  } else if (is_epilogue) {
    // Seed both empty slots before the first MMA acquisition.
    for (int i = 0; i < C::EP_BUFFERS; i++) {
      arrive_mma_leader<C::CTA_GROUP>(e_to_c_bar + i, cta_rank);
    }
  } else if (is_scheduler) {
    if (lane_id == 0) {
      bool first_pass = true;
      while (true) {
        if (cta_rank == 0) {
          if (!first_pass) {
            wait<C::CLUSTERS>(clc_empty_bar + clc_stage, c_f_p ^ 1);
          }
          clc_try_cancel(smem_p.clc_response + clc_stage, clc_full_bar + clc_stage);
        }
        wait<C::CLUSTERS>(clc_full_bar + clc_stage, c_f_p);
        clc_query_response(smem_p.clc_response + clc_stage, is_valid, local_block);
        local_block /= C::CLUSTERS;
        // Observe each request before issuing the next, including the final failure.
        if (!is_valid || local_block >= total_blocks) break;
        expect_bytes(clc_full_bar + clc_stage, sizeof(int4));
        arrive<C::CLUSTERS, 0>(clc_empty_bar + clc_stage);
        if (++clc_stage == C::CLC_STAGES) {
          clc_stage = 0;
          c_f_p ^= 1;
          first_pass = false;
        }
      }
      // A cancelled padding block ends worker processing; a failed cancellation
      // ends CLC entirely, since retrying after observing failure is undefined.
      if (is_valid) {
        arrive<C::CLUSTERS, 0>(clc_empty_bar + clc_stage);
        if (cta_rank == 0) {
          // Every reader must release the terminal response before reuse.
          wait<C::CLUSTERS>(clc_empty_bar + clc_stage, c_f_p);
          do {
            // Reuse this one slot locally; peer schedulers have stopped.
            c_f_p ^= 1;
            expect_bytes(clc_full_bar + clc_stage, sizeof(int4));
            clc_try_cancel<false>(smem_p.clc_response + clc_stage, clc_full_bar + clc_stage);
            wait<C::CLUSTERS>(clc_full_bar + clc_stage, c_f_p);
            clc_query_response(smem_p.clc_response + clc_stage, is_valid, local_block);
          } while (is_valid);
        }
      }
    }
    is_valid = false;
  }

  if (is_producer) {
    const int producer_tid = threadIdx.x - C::MMA_THREADS;
    const int producer_wid = producer_tid / 32;
    const int gather_local = producer_wid + (producer_tid % 32) * (C::PRODUCER_THREADS / 32);
    const int gather_idx = n_group * C::GATHERS_PER_CTA + gather_local;
    int4 tokens;
    if (gather_local < C::GATHERS_PER_CTA) {
      tokens = reinterpret_cast<const int4*>(sorted_token_ids + local_block * C::BM +
                                             group_rank * C::LOCAL_M)[gather_idx];
    }
    while (is_valid) {
      // The host grid is an upper bound; cancelled padding CTAs have no token tile.
      const int current_block = local_block;
      // Fallback M128 halves share the expert entry of one routed M256 tile.
      const int exp_idx = expert_ids[current_block * C::BM / C::ROUTING_BM];

      if (gather_local < C::GATHERS_PER_CTA) {
        tokens.x /= top_k;
        tokens.y /= top_k;
        tokens.z /= top_k;
        tokens.w /= top_k;
      }
      for (int n_off = n_group * C::MMA_N_UP; n_off < UP_TILES * C::MMA_N_UP;
           n_off += C::N_GROUPS * C::MMA_N_UP) {
        const bool reuse_x = REUSE_X && n_stages_up >= C::STAGES && n_off != n_group * C::MMA_N_UP;
        if (reuse_x) x_cursor.reverse();
        for (int load_stage = 0; load_stage < n_stages_up; load_stage++) {
          if (smem_stage == C::STAGES) {
            p ^= 1;
            smem_stage = 0;
          }
          const int off = (x_cursor.reversed ? n_stages_up - 1 - load_stage : load_stage) * C::BK;
          const bool load_x = !reuse_x || load_stage >= C::STAGES;

          // Idle warps can miss a phase when the pipeline advances without them.
          if (producer_wid == 0 || (load_x && gather_local < C::GATHERS_PER_CTA))
            wait<C::CLUSTERS>(bar + C::STAGES + smem_stage, p);
          if (producer_wid == 0 && elect_sync()) {
            if (group_rank == 0)
              expect_bytes(bar + smem_stage,
                           C::CTA_GROUP * (C::WS + (load_x ? C::XS : 0)) * sizeof(bf16));
            load_async_4d<C::CTA_GROUP, C::EVICT_WEIGHTS>(
                s.w + smem_stage * C::WS, &tensor_map_w, bar + smem_stage, 0,
                n_off + group_rank * C::TILE_COLS_UP, off / 64, exp_idx);
          }

          if (load_x && gather_local < C::GATHERS_PER_CTA) {
            int idx = gather_idx;
            bf16* dst = s.x + x_cursor.stage * C::XS + idx * 4 * C::BK;
            load_async_2d_gather4<C::CLUSTERS, C::CTA_GROUP>(dst, &tensor_map_x, bar + smem_stage,
                                                             off, tokens, x_mask);
          }

          smem_stage++;
          x_cursor.advance();
          if (reuse_x && load_stage + 1 == C::STAGES)
            asm volatile("bar.sync 1, %0;" ::"n"(C::PRODUCER_THREADS) : "memory");
        }
      }

      // Valid routes precede padding. Count occupied scratch chunks across the
      // entire M tile so every CTA agrees on the group-2 transaction bytes.
      const bool has_x_block =
          producer_tid < C::BM / C::X_BLOCK_SIZE &&
          sorted_token_ids[current_block * C::BM + producer_tid * C::X_BLOCK_SIZE] < M * top_k;
      int x_blocks;
      asm volatile(
          "{ .reg .pred occupied;\n"
          "  setp.ne.u32 occupied, %1, 0;\n"
          "  bar.red.popc.u32 %0, 1, %2, occupied;\n"
          "}"
          : "=r"(x_blocks)
          : "r"(int(has_x_block)), "n"(C::PRODUCER_THREADS)
          : "memory");
      const int local_x_blocks =
          min(C::LOCAL_M / C::X_BLOCK_SIZE,
              max(0, x_blocks - group_rank * (C::LOCAL_M / C::X_BLOCK_SIZE)));
      wait<C::CLUSTERS>(clc_full_bar + clc_stage, c_f_p);
      clc_query_response(smem_p.clc_response + clc_stage, is_valid, local_block);
      local_block /= C::CLUSTERS;
      arrive<C::CLUSTERS, 0>(clc_empty_bar + clc_stage);
      if (++clc_stage == C::CLC_STAGES) {
        clc_stage = 0;
        c_f_p ^= 1;
      }
      is_valid = is_valid && (local_block < total_blocks);
      // Prefetch the next UP gather IDs while the current block runs DOWN.
      if (is_valid && gather_local < C::GATHERS_PER_CTA) {
        tokens = reinterpret_cast<const int4*>(sorted_token_ids + local_block * C::BM +
                                               group_rank * C::LOCAL_M)[gather_idx];
      }
      // Preserve X-buffer order at projection boundaries; the first DOWN sweep
      // loads fresh scratch tiles in forward K order as SwiGLU publishes them.
      bool reversed_down = false;
      for (int col = n_group; col < n_stages_down; col += C::N_GROUPS) {
        const bool reuse_x = REUSE_X && k_stages_down >= C::STAGES && col != n_group;
        if (reuse_x) {
          x_cursor.reverse();
          reversed_down = !reversed_down;
        }
        for (int load_stage = 0; load_stage < k_stages_down; load_stage++) {
          if (smem_stage == C::STAGES) {
            p ^= 1;
            smem_stage = 0;
          }
          if (producer_wid == 0 && elect_sync()) {
            const int k_off =
                (reversed_down ? k_stages_down - 1 - load_stage : load_stage) * C::BK2;
            const bool load_x = !reuse_x || load_stage >= C::STAGES;
            const int scratch_tile = k_off / (C::MMA_N_UP / C::UD_RATIO);
            // All BK2 chunks belong to the CTA that published this activation tile.
            const bool is_x_loader = load_x && n_group == scratch_tile % C::N_GROUPS;
            wait<C::CLUSTERS>(bar + C::STAGES + smem_stage, p);
            const int w_row_down = k_off;
            const int w_col_down = col * C::BN2 + group_rank * (C::BN2 / C::CTA_GROUP);

            if (group_rank == 0)
              expect_bytes(bar + smem_stage, (C::CTA_GROUP * C::WS +
                                              (load_x ? x_blocks * C::X_BLOCK_SIZE * C::BK2 : 0)) *
                                                 sizeof(bf16));
            load_async_4d<C::CTA_GROUP, C::EVICT_WEIGHTS>(
                s_d.w + smem_stage * C::WS, &tensor_map_w2, bar + smem_stage, 0, w_col_down,
                w_row_down / C::TILE_INNER_DN, exp_idx);
            // The publishing CTA loads X and multicasts it to every peer.
            if (is_x_loader) {
              // Issue weights before waiting for SwiGLU. The first sweep
              // observes every scratch tile; later sweeps reuse that data.
              if (col == n_group) {
                wait<C::CLUSTERS>(e_to_p_bar + static_cast<unsigned>(scratch_tile) / C::N_GROUPS,
                                  scratch_phase);
              }
              if (local_x_blocks == C::LOCAL_M / C::X_BLOCK_SIZE) {
                load_async<C::CLUSTERS, C::CTA_GROUP>(s_d.x + x_cursor.stage * C::XS, &tensor_map_s,
                                                      bar + smem_stage, scratch_row_base, k_off,
                                                      x_mask);
              } else {
                for (int block = 0; block < local_x_blocks; ++block) {
                  const int row = block * C::X_BLOCK_SIZE;
                  load_async<C::CLUSTERS, C::CTA_GROUP>(
                      s_d.x + x_cursor.stage * C::XS + row * C::BK2, &tensor_map_s_partial,
                      bar + smem_stage, scratch_row_base + row, k_off, x_mask);
                }
              }
            }
          }
          smem_stage++;
          x_cursor.advance();
        }
      }
      // Each SwiGLU-ready barrier completes once per processed token block.
      scratch_phase ^= 1;
      // Inactive DOWN producers rejoin before the next job's UP loads.
      if (is_valid) asm volatile("bar.sync 1, %0;" ::"n"(C::PRODUCER_THREADS) : "memory");
    }
  }
  // MMA
  else if (is_mma) {
    // Only group leaders issue MMA and consume CLC responses.
    if (group_rank == 0 && warp_id == 0 && elect_sync()) {
      while (is_valid) {
        const int current_block = local_block;
        for (int n_off = n_group * C::MMA_N_UP; n_off < UP_TILES * C::MMA_N_UP;
             n_off += C::N_GROUPS * C::MMA_N_UP) {
          if (REUSE_X && n_stages_up >= C::STAGES && n_off != n_group * C::MMA_N_UP)
            x_cursor.reverse();
          if (epilogue_stage == C::EP_BUFFERS) {
            e_p ^= 1;
            epilogue_stage = 0;
          }
          wait<C::CLUSTERS>(e_to_c_bar + epilogue_stage, e_p);
          for (int compute_stage = 0; compute_stage < n_stages_up; compute_stage += 1) {
            if (smem_stage == C::STAGES) {
              p ^= 1;
              smem_stage = 0;
            }
            wait<C::CLUSTERS>(bar + smem_stage, p);
            uint64_t desc_w = make_smem_descriptor<64, 1, C::S_MODE_UP>(s.w + smem_stage * C::WS);
            uint64_t desc_x =
                make_smem_descriptor<64, 1, C::S_MODE_UP>(s.x + x_cursor.stage * C::XS);
            for (int k = 0; k < C::BK; k += 16) {
              tcgen05_mma_f16<C::CTA_GROUP>(tmem_addr + epilogue_stage * C::MMA_N_UP, desc_x,
                                            desc_w, C::idesc_up, int(compute_stage != 0 || k != 0));
              desc_x += (32 >> 4);
              desc_w += (32 >> 4);
            }
            tcgen05_commit<C::CTA_GROUP, C::CLUSTERS>(
                __cvta_generic_to_shared(bar + C::STAGES + smem_stage));
            smem_stage++;
            x_cursor.advance();
          }
          // Separate completion barriers let SwiGLU trail UP by either tile.
          tcgen05_commit<C::CTA_GROUP>(__cvta_generic_to_shared(c_to_e_bar + epilogue_stage),
                                       group_mask);
          epilogue_stage++;
        }

        wait<C::CLUSTERS>(clc_full_bar + clc_stage, c_f_p);
        clc_query_response(smem_p.clc_response + clc_stage, is_valid, local_block);
        local_block /= C::CLUSTERS;
        arrive<C::CLUSTERS, 0>(clc_empty_bar + clc_stage, 1);
        if (++clc_stage == C::CLC_STAGES) {
          clc_stage = 0;
          c_f_p ^= 1;
        }
        is_valid = is_valid && (local_block < total_blocks);
        for (int compute_stage = n_group; compute_stage < n_stages_down;
             compute_stage += C::N_GROUPS) {
          if (REUSE_X && k_stages_down >= C::STAGES && compute_stage != n_group) x_cursor.reverse();
          if (epilogue_stage == C::EP_BUFFERS) {
            e_p ^= 1;
            epilogue_stage = 0;
          }
          wait<C::CLUSTERS>(e_to_c_bar + epilogue_stage, e_p);
          for (int k_stage = 0; k_stage < k_stages_down; k_stage++) {
            if (smem_stage == C::STAGES) {
              p ^= 1;
              smem_stage = 0;
            }
            wait<C::CLUSTERS>(bar + smem_stage, p);
            uint64_t desc_w =
                make_smem_descriptor<64, 1, C::S_MODE_DOWN>(s_d.w + smem_stage * C::WS);
            uint64_t desc_x =
                make_smem_descriptor<64, 1, C::S_MODE_UP>(s_d.x + x_cursor.stage * C::XS);
            for (int k = 0; k < C::BK; k += 16) {
              tcgen05_mma_f16<C::CTA_GROUP>(tmem_addr + epilogue_stage * C::MMA_N_UP, desc_x,
                                            desc_w, C::idesc_down, int(k_stage != 0 || k != 0));
              desc_x += (32 >> 4);
              desc_w += (32 >> 4);
            }
            tcgen05_commit<C::CTA_GROUP, C::CLUSTERS>(
                __cvta_generic_to_shared(bar + C::STAGES + smem_stage));
            smem_stage++;
            x_cursor.advance();
          }
          tcgen05_commit<C::CTA_GROUP>(__cvta_generic_to_shared(c_to_e_bar + epilogue_stage),
                                       group_mask);
          epilogue_stage++;
        }
        // Each SwiGLU-ready barrier completes once per processed token block.
        scratch_phase ^= 1;
      }
    }
  } else if (is_epilogue) {
    while (is_valid) {
      // The host grid is an upper bound; cancelled padding CTAs have no token tile.
      const int* local_token_ids = sorted_token_ids + local_block * C::BM + group_rank * C::LOCAL_M;
      int epilogue_tid = threadIdx.x - C::PRODUCER_THREADS - C::MMA_THREADS;

      const int epilogue_wid = (epilogue_tid / 32) % 4;
      const int epilogue_wgid = epilogue_tid / 128;
      const int epilogue_row_base =
          (C::SPLIT_N_TMEM ? epilogue_wid % 2 : epilogue_wid) * C::EP_ROWS_PER_WARP;
      const int epilogue_col_base = C::SPLIT_N_TMEM ? (epilogue_wid / 2) * C::EP_COLS_PER_WARP : 0;
      const int down_col_base = epilogue_wgid * C::EP_COLS_PER_WARPGROUP;
      const int output_warp = epilogue_tid / 32;

      float token_scale[C::TPT];
      int token_dst[C::TPT];
      for (int t = 0; t < C::TPT; t++) {
        const int token_idx = epilogue_row_base + t * 16 + lane_id % 16;
        const int routed_token = local_token_ids[token_idx];
        const int token = routed_token / top_k;
        token_dst[t] = token;
        token_scale[t] = token < M ? topk_weights[routed_token] * scaling_factor : 0.0f;
      }

      for (int n_off = n_group * C::MMA_N_UP; n_off < UP_TILES * C::MMA_N_UP;
           n_off += C::N_GROUPS * C::MMA_N_UP) {
        if (epilogue_stage == C::EP_BUFFERS) {
          e_p ^= 1;
          epilogue_stage = 0;
        }
        wait<C::CLUSTERS>(c_to_e_bar + epilogue_stage, e_p);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

        uint32_t acc_tmem = tmem_addr + epilogue_stage * C::MMA_N_UP;
        constexpr int up_cols_per_load = C::IS_GATED ? 32 : 16;
        float acc[C::IS_GATED ? 16 : 8];
        for (int row_off_tmem = 0; row_off_tmem < C::EP_ROWS_PER_WARP; row_off_tmem += 16) {
          // Warp groups share the TMEM rows and consume disjoint columns.
          const int token = token_dst[row_off_tmem / 16];
          for (int tn = epilogue_wgid * up_cols_per_load; tn < C::EP_COLS_PER_WARP;
               tn += C::EPILOGUE_WARPGROUPS * up_cols_per_load) {
            const int row = epilogue_wid * 32 + row_off_tmem;
            const int col = tn;
            const uint32_t addr = acc_tmem + (row << 16) + col;
            if constexpr (C::IS_GATED) {
              // Each lane loads eight gate values followed by eight up values.
              asm volatile(
                  "tcgen05.ld.sync.aligned.16x32bx2.x16.b32 {%0, %1, %2, "
                  "%3, %4, "
                  "%5, "
                  "%6, "
                  "%7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                  : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]), "=f"(acc[4]),
                    "=f"(acc[5]), "=f"(acc[6]), "=f"(acc[7]), "=f"(acc[8]), "=f"(acc[9]),
                    "=f"(acc[10]), "=f"(acc[11]), "=f"(acc[12]), "=f"(acc[13]), "=f"(acc[14]),
                    "=f"(acc[15])
                  : "r"(addr));
            } else {
              // Paired half-warps fill one 32-byte BF16 sector per row.
              asm volatile(
                  "tcgen05.ld.sync.aligned.16x32bx2.x8.b32 "
                  "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                  : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]), "=f"(acc[4]),
                    "=f"(acc[5]), "=f"(acc[6]), "=f"(acc[7])
                  : "r"(addr));
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;");

            const int scratch_row = epilogue_row_base + row_off_tmem + lane_id % 16;
            const int scratch_col =
                (n_off + epilogue_col_base + tn) / C::UD_RATIO + (lane_id / 16) * 8;
            bf16* dst = scratchpad + size_t(scratch_row_base) * SCRATCH_COLS +
                        scratch_row * SCRATCH_COLS + scratch_col;
            nv_bfloat162 activated[4];
            if constexpr (C::IS_GATED) {
              for (int i = 0; i < 4; i++) {
                activated[i] =
                    make_bfloat162(moe_activation::apply<typename C::Activation>(
                                       acc[2 * i], acc[2 * i + 8], activation_params),
                                   moe_activation::apply<typename C::Activation>(
                                       acc[2 * i + 1], acc[2 * i + 9], activation_params));
              }
            } else {
              for (int i = 0; i < 4; i++) {
                activated[i] = make_bfloat162(
                    moe_activation::apply<typename C::Activation>(acc[2 * i], activation_params),
                    moe_activation::apply<typename C::Activation>(acc[2 * i + 1],
                                                                  activation_params));
              }
            }
            if (UP_TILES * C::MMA_N_UP == N || (scratch_col < SCRATCH_COLS && token < M))
              __stcg(reinterpret_cast<float4*>(dst), *reinterpret_cast<float4*>(activated));
          }
        }
        // Publish global activation stores to the TMA async proxy.
        cuda::ptx::fence_proxy_async(cuda::ptx::space_global);
        arrive(e_to_p_bar + static_cast<unsigned>(n_off) / (C::MMA_N_UP * C::N_GROUPS));
        arrive_mma_leader<C::CTA_GROUP>(e_to_c_bar + epilogue_stage, cta_rank);
        epilogue_stage++;
      }

      for (int compute_stage = n_group; compute_stage < n_stages_down;
           compute_stage += C::N_GROUPS) {
        if (epilogue_stage == C::EP_BUFFERS) {
          e_p ^= 1;
          epilogue_stage = 0;
        }
        wait<C::CLUSTERS>(c_to_e_bar + epilogue_stage, e_p);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        float acc[8];
        nv_bfloat162 tile_res[4];
        for (int row_off = 0; row_off < C::EP_ROWS_PER_WARP; row_off += 16) {
          const int token = token_dst[row_off / 16];
          if constexpr (!C::RED_STORE) {
            // Both half-warps write these rows; all prior readers must finish.
            asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            __syncwarp();
          }
          // Each warp group owns a contiguous range and private staging rows.
          for (int tn = down_col_base / 16; tn < (down_col_base + C::EP_COLS_PER_WARPGROUP) / 16;
               ++tn) {
            const int row = epilogue_wid * 32 + row_off;
            const int col = tn * 16;
            const uint32_t addr = tmem_addr + epilogue_stage * C::MMA_N_UP + (row << 16) + col;
            // To coalesce stores we load 2 adjacent 32B chunks:
            // lane 0 cols 0-7 | lane 16 cols 8-15
            // lane 1 cols 0-7 | lane 17 cols 8-15 ...
            // After casting to bf16 lane 0 and 16 coalesce into 32B sector
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32 {%0, %1, %2, %3, %4, "
                "%5, "
                "%6, %7}, [%8], 8;"
                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]), "=f"(acc[4]),
                  "=f"(acc[5]), "=f"(acc[6]), "=f"(acc[7])
                : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const float scale = token_scale[row_off / 16];
            for (int i = 0; i < 4; i++) {
              tile_res[i] = __float22bfloat162_rn({acc[2 * i] * scale, acc[2 * i + 1] * scale});
            }
            float4 values = *reinterpret_cast<float4*>(tile_res);
            if constexpr (C::RED_STORE) {
              const int store_col =
                  compute_stage * C::BN2 + epilogue_col_base + tn * 16 + (lane_id / 16) * 8;
              if (token < M) {
                red_add_bf16x8(out + size_t(token) * N2 + store_col, values);
              }
            } else {
              const int output_row = output_warp * 16 + lane_id % 16;
              const int output_col = tn * 16 - down_col_base + (lane_id / 16) * 8;
              *reinterpret_cast<float4*>(s_d.output + output_row * C::OUTPUT_STRIDE + output_col) =
                  values;
            }
          }
          if (row_off + 16 >= C::EP_ROWS_PER_WARP) {
            // Accumulators have been read, so TMEM can be reused.
            arrive_mma_leader<C::CTA_GROUP>(e_to_c_bar + epilogue_stage, cta_rank);
          }
          if constexpr (!C::RED_STORE) {
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (lane_id < 16 && token < M) {
              const int output_row = output_warp * 16 + lane_id;
              asm volatile(
                  "cp.reduce.async.bulk.global.shared::cta.bulk_group."
                  "add.noftz.bf16 [%0], [%1], %2;" ::"l"(out + size_t(token) * N2 +
                                                         compute_stage * C::BN2 +
                                                         epilogue_col_base + down_col_base),
                  "r"(static_cast<uint32_t>(
                      __cvta_generic_to_shared(s_d.output + output_row * C::OUTPUT_STRIDE))),
                  "r"(uint32_t(C::EP_COLS_PER_WARPGROUP * sizeof(bf16)))
                  : "memory");
            }
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
          }
        }
        epilogue_stage++;
      }
      // Each SwiGLU-ready barrier completes once per processed token block.
      scratch_phase ^= 1;
      wait<C::CLUSTERS>(clc_full_bar + clc_stage, c_f_p);
      clc_query_response(smem_p.clc_response + clc_stage, is_valid, local_block);
      local_block /= C::CLUSTERS;
      arrive<C::CLUSTERS, 0>(clc_empty_bar + clc_stage);
      if (++clc_stage == C::CLC_STAGES) {
        clc_stage = 0;
        c_f_p ^= 1;
      }
      is_valid = is_valid && (local_block < total_blocks);
    }
  }
  // Keep TMEM allocated until every epilogue warp has consumed the final tile.
  if constexpr (C::CLUSTERS > 1)
    cg::this_cluster().sync();
  else
    __syncthreads();
  if (warp_id == 1) {
    tmem_dealloc<C::CTA_GROUP>(tmem_addr, C::TMEM_COLS);
  }
#else
  assert(false && "This fused MoE kernel requires SM100 or newer");
#endif
}

template <typename C, bool REUSE_X>
__global__ __cluster_dims__(C::CLUSTERS, 1, 1)
    __launch_bounds__(C::BLOCK_THREADS, 1) void fused_moe_bf16_wgmma_up_down_acc_kernel(
        const __grid_constant__ CUtensorMap tensor_map_x,
        const __grid_constant__ CUtensorMap tensor_map_w,
        const __grid_constant__ CUtensorMap tensor_map_w2,
        const __grid_constant__ CUtensorMap tensor_map_s,
        const __grid_constant__ CUtensorMap tensor_map_s_partial, bf16* __restrict__ scratchpad,
        bf16* __restrict__ out, const int* __restrict__ sorted_token_ids,
        const int* __restrict__ expert_ids, const int* __restrict__ num_tokens_post_padded,
        const float* __restrict__ topk_weights, const int top_k, int M, int N, int K,
        float scaling_factor, typename C::ActivationParams activation_params) {
  fused_moe_bf16_wgmma_up_down_acc_body<C, REUSE_X>(
      tensor_map_x, tensor_map_w, tensor_map_w2, tensor_map_s, tensor_map_s_partial, scratchpad,
      out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, N, K,
      scaling_factor, activation_params);
}

template <typename C, typename F, bool REUSE_X>
__global__
__launch_bounds__(C::BLOCK_THREADS, 1) void fused_moe_bf16_wgmma_up_down_acc_adaptive_kernel(
    const __grid_constant__ CUtensorMap tensor_map_x,
    const __grid_constant__ CUtensorMap tensor_map_w,
    const __grid_constant__ CUtensorMap tensor_map_w2,
    const __grid_constant__ CUtensorMap tensor_map_s,
    const __grid_constant__ CUtensorMap tensor_map_s_partial, bf16* __restrict__ scratchpad,
    bf16* __restrict__ out, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ num_tokens_post_padded,
    const float* __restrict__ topk_weights, const int top_k, int M, int N, int K,
    float scaling_factor, typename C::ActivationParams activation_params,
    const __grid_constant__ CUtensorMap tensor_map_s_fallback) {
#if __CUDA_ARCH__ >= 1000
  // Both branches must use the same tcgen05 CTA group throughout the kernel.
  static_assert(C::CTA_GROUP == 2 && F::CTA_GROUP == 2);
  static_assert(C::BLOCK_THREADS == F::BLOCK_THREADS);
  uint32_t cluster_size;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(cluster_size));
  if (cluster_size == C::CLUSTERS) {
    fused_moe_bf16_wgmma_up_down_acc_body<C, REUSE_X>(
        tensor_map_x, tensor_map_w, tensor_map_w2, tensor_map_s, tensor_map_s_partial, scratchpad,
        out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, N, K,
        scaling_factor, activation_params);
  } else {
    assert(cluster_size == F::CLUSTERS);
    fused_moe_bf16_wgmma_up_down_acc_body<F, REUSE_X>(
        tensor_map_x, tensor_map_w, tensor_map_w2, tensor_map_s_fallback, tensor_map_s_partial,
        scratchpad, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k,
        M, N, K, scaling_factor, activation_params);
  }
#endif
}

template <int BM, int STAGES, int CLUSTERS, bool RED_STORE, moe_activation::Kind ACTIVATION,
          bool PREFERRED_CLUSTERS, bool WEIGHT_EVICTION, int EPILOGUE_WARPS, int PRODUCER_WARPS,
          int EXTRA_BARRIERS>
bool launch_fused_moe_kernel_up_down_acc_impl(
    const bf16* x, bf16* w, bf16* w2, bf16* out, bf16* scratchpad, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded, const float* topk_weights,
    const int top_k, int M, int N, int K, int num_experts, int sorted_num, int block_m,
    float scaling_factor, typename moe_activation::Traits<ACTIVATION>::Params activation_params,
    cudaStream_t stream) {
  using C = constants<BM, STAGES, CLUSTERS, RED_STORE, ACTIVATION, BM == 256 ? 2 : 1, BM,
                      WEIGHT_EVICTION, EPILOGUE_WARPS, PRODUCER_WARPS, EXTRA_BARRIERS>;
  using F = constants<PREFERRED_CLUSTERS ? 128 : BM, STAGES,
                      PREFERRED_CLUSTERS ? CLUSTERS / 2 : CLUSTERS, RED_STORE, ACTIVATION,
                      PREFERRED_CLUSTERS ? 2 : C::CTA_GROUP, PREFERRED_CLUSTERS ? 256 : BM,
                      WEIGHT_EVICTION, EPILOGUE_WARPS, PRODUCER_WARPS, EXTRA_BARRIERS>;
  static_assert(!PREFERRED_CLUSTERS || (BM == 256 && CLUSTERS >= 4));
  if (N <= 0 || N > C::MAX_EXTRA_BARRIERS * C::MMA_N_UP * std::min(C::N_GROUPS, F::N_GROUPS))
    return false;
  const int N2 = K;
  const int K2 = N / C::UD_RATIO;
  constexpr size_t launch_smem = std::max(C::sMemSize, F::sMemSize);

  dim3 dimBlock(C::BLOCK_THREADS, 1, 1);
  int device;
  gpuErrchk(cudaGetDevice(&device));
  int num_sms;
  gpuErrchk(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  // CLC steals pending CTAs from the full work grid.
  const int block_count = (sorted_num + block_m - 1) / block_m;
  if (block_count == 0) return false;
  dim3 dimGrid(block_count * C::CLUSTERS, 1, 1);

  if constexpr (launch_smem > C::SM100_MAX_SMEM) {
    fprintf(stderr,
            "fused_moe: BM=%d N=%d WN=%d STAGES=%d requires %zu bytes of "
            "shared memory but SM100 max is %zu — skipping launch\n",
            BM, N, C::WN, STAGES, C::sMemSize, C::SM100_MAX_SMEM);
    return false;
  }
  if constexpr (launch_smem <= C::SM100_MAX_SMEM) {
    int max_smem_real;
    cudaDeviceGetAttribute(&max_smem_real, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (launch_smem > max_smem_real) return false;

    auto kernel = [] {
      if constexpr (PREFERRED_CLUSTERS)
        return fused_moe_bf16_wgmma_up_down_acc_adaptive_kernel<C, F, true>;
      else
        return fused_moe_bf16_wgmma_up_down_acc_kernel<C, true>;
    }();
    cudaFuncAttributes attributes{};
    const auto status = cudaFuncGetAttributes(&attributes, kernel);
    if (status != cudaSuccess) {
      cudaGetLastError();
      return false;
    }

    gpuErrchk(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, launch_smem));
    if constexpr (C::CLUSTERS > 8) {
      gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    }

    CUtensorMap tensor_map_w{};
    if (!encode_bf16_weight_map(&tensor_map_w, w, num_experts, N, K, C::TILE_COLS_UP, C::BK))
      return false;

    CUtensorMap tensor_map_x{};
    {
      uint64_t size[C::TMA_RANK_X] = {static_cast<uint64_t>(K), static_cast<uint64_t>(M)};
      uint64_t stride[C::TMA_RANK_X - 1] = {static_cast<uint64_t>(K) * sizeof(bf16)};
      // For tile::gather4 the bounding box along dim1 must be 1
      // (the 4 row coords provided to the instruction already define
      // the 4 rows to gather). Using anything other than 1 makes the
      // TMA load an illegal instruction. See PTX ISA 5.5.3.4.1.
      uint32_t box_size[C::TMA_RANK_X] = {C::BK, 1};
      uint32_t elem_stride[C::TMA_RANK_X] = {1, 1};
      auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
      CUresult res = cuTensorMapEncodeTiled(
          &tensor_map_x, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, C::TMA_RANK_X,
          const_cast<bf16*>(x), size, stride, box_size, elem_stride,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (res != CUDA_SUCCESS) return false;
      if (int(res) != 0) printf("tensor map x error %d\n", int(res));
    }

    // Create TMA descriptor for w2 (down projection)
    if constexpr (C::BN2 / 64 < C::WN / 4) {
      fprintf(stderr, "Too much warpgroups(%d) for block N(%d)\n", C::WN, C::BN2);
      return false;
    }
    CUtensorMap tensor_map_w2{};
    if (!encode_bf16_weight_map(&tensor_map_w2, w2, num_experts, N2, K2, C::BN2 / C::CTA_GROUP,
                                C::BK2))
      return false;

    static_assert(C::TMEM_COLS == 512);
    static_assert(C::SCRATCH_ROWS_PER_SM == F::SCRATCH_ROWS_PER_SM);
    const size_t scratch_rows = size_t(num_sms) * C::SCRATCH_ROWS_PER_SM;
    CUtensorMap tensor_map_s{}, tensor_map_s_fallback{}, tensor_map_s_partial{};
    CUtensorMap* scratch_maps[] = {&tensor_map_s_partial, &tensor_map_s, &tensor_map_s_fallback};
    const uint32_t scratch_box_rows[] = {C::X_BLOCK_SIZE, C::LOCAL_M, F::LOCAL_M};
    static_assert(C::X_BLOCK_SIZE == F::X_BLOCK_SIZE);
    for (int map = 0; map < (PREFERRED_CLUSTERS ? 3 : 2); ++map) {
      uint64_t size[C::TMA_RANK_S] = {static_cast<uint64_t>(K2), scratch_rows};
      uint64_t stride[C::TMA_RANK_S - 1] = {K2 * sizeof(bf16)};
      uint32_t box_size[C::TMA_RANK_S] = {C::BK2, scratch_box_rows[map]};
      uint32_t elem_stride[C::TMA_RANK_S] = {1, 1};
      auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
      CUresult res = cuTensorMapEncodeTiled(
          scratch_maps[map], CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, C::TMA_RANK_S,
          scratchpad, size, stride, box_size, elem_stride,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (res != CUDA_SUCCESS) return false;
    }

    if constexpr (!PREFERRED_CLUSTERS) {
      kernel<<<dimGrid, dimBlock, launch_smem, stream>>>(
          tensor_map_x, tensor_map_w, tensor_map_w2, tensor_map_s, tensor_map_s_partial, scratchpad,
          out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, N, K,
          scaling_factor, activation_params);
    } else {
      cudaLaunchAttribute attributes[2]{};
      // CUDA fills space left by preferred clusters with half-sized M128 clusters.
      attributes[0].id = cudaLaunchAttributeClusterDimension;
      attributes[0].val.clusterDim = {F::CLUSTERS, 1, 1};
      attributes[1].id = cudaLaunchAttributePreferredClusterDimension;
      attributes[1].val.preferredClusterDim = {C::CLUSTERS, 1, 1};
      cudaLaunchConfig_t config{};
      config.gridDim = dimGrid;
      config.blockDim = dimBlock;
      config.dynamicSmemBytes = launch_smem;
      config.stream = stream;
      config.attrs = attributes;
      config.numAttrs = 2;
      gpuErrchk(cudaLaunchKernelEx(&config, kernel, tensor_map_x, tensor_map_w, tensor_map_w2,
                                   tensor_map_s, tensor_map_s_partial, scratchpad, out,
                                   sorted_token_ids, expert_ids, num_tokens_post_padded,
                                   topk_weights, top_k, M, N, K, scaling_factor, activation_params,
                                   tensor_map_s_fallback));
    }
    gpuErrchk(cudaGetLastError());
    return true;
  }
  return false;
}

// clang-format off
template <int BM, int STAGES, int CLUSTERS, bool RED_STORE, moe_activation::Kind ACTIVATION, bool WEIGHT_EVICTION = false, int EPILOGUE_WARPS = 4, int PRODUCER_WARPS = 4, int EXTRA_BARRIERS = 64>
bool launch_fused_moe_kernel_up_down_acc(const bf16* x, bf16* w, bf16* w2, bf16* out, bf16* scratchpad, const int* sorted_token_ids, const int* expert_ids, const int* num_tokens_post_padded, const float* topk_weights, const int top_k, int M, int N, int K, int num_experts, int sorted_num, int block_m, float scaling_factor, typename moe_activation::Traits<ACTIVATION>::Params activation_params, cudaStream_t stream, bool cluster_fallback = false) {
    if (cluster_fallback) {
        if constexpr (BM == 256 && CLUSTERS >= 4) {
            return launch_fused_moe_kernel_up_down_acc_impl<BM, STAGES, CLUSTERS, RED_STORE, ACTIVATION, true, WEIGHT_EVICTION, EPILOGUE_WARPS, PRODUCER_WARPS, EXTRA_BARRIERS>(x, w, w2, out, scratchpad, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, N, K, num_experts, sorted_num, block_m, scaling_factor, activation_params, stream);
        } else {
            return false;
        }
    }
    return launch_fused_moe_kernel_up_down_acc_impl<BM, STAGES, CLUSTERS, RED_STORE, ACTIVATION, false, WEIGHT_EVICTION, EPILOGUE_WARPS, PRODUCER_WARPS, EXTRA_BARRIERS>(x, w, w2, out, scratchpad, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, N, K, num_experts, sorted_num, block_m, scaling_factor, activation_params, stream);
}

template <int BM, int STAGES, int CLUSTERS, bool RED_STORE, moe_activation::Kind ACTIVATION, bool WEIGHT_EVICTION = false, int EPILOGUE_WARPS = 4, int PRODUCER_WARPS = 4>
bool dispatch_barriers(const bf16* x, bf16* w, bf16* w2, bf16* out, bf16* scratchpad, const int* sorted_token_ids, const int* expert_ids, const int* num_tokens_post_padded, const float* topk_weights, const int top_k, int M, int N, int K, int num_experts, int sorted_num, int block_m, float scaling_factor, typename moe_activation::Traits<ACTIVATION>::Params activation_params, cudaStream_t stream, bool cluster_fallback = false) {
    // The fallback has fewer N groups, so each CTA may publish more tiles.
    const int groups = cluster_fallback ? CLUSTERS / 4 : CLUSTERS / (BM == 256 ? 2 : 1);
    const int barriers = (N - 1) / (256 * groups) + 1;
    auto launch = [&]<int capacity>() {
        return launch_fused_moe_kernel_up_down_acc<BM, STAGES, CLUSTERS, RED_STORE, ACTIVATION, WEIGHT_EVICTION, EPILOGUE_WARPS, PRODUCER_WARPS, capacity>(x, w, w2, out, scratchpad, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, N, K, num_experts, sorted_num, block_m, scaling_factor, activation_params, stream, cluster_fallback);
    };
    if (barriers <= 16)
        return launch.template operator()<16>();
    if (barriers <= 64)
        return launch.template operator()<64>();
    return false;
}

template <int BM, int CLUSTERS, bool RED_STORE, moe_activation::Kind ACTIVATION, bool WEIGHT_EVICTION = false, int EPILOGUE_WARPS = 4, int PRODUCER_WARPS = 4>
bool dispatch_stages(int stages, const bf16* x, bf16* w, bf16* w2, bf16* out, bf16* scratchpad, const int* sorted_token_ids, const int* expert_ids, const int* num_tokens_post_padded, const float* topk_weights, const int top_k, int M, int N, int K, int num_experts, int sorted_num, int block_m, float scaling_factor, typename moe_activation::Traits<ACTIVATION>::Params activation_params, cudaStream_t stream, bool cluster_fallback = false) {
    using C = constants<BM, 1, CLUSTERS, RED_STORE, ACTIVATION, BM == 256 ? 2 : 1, BM, WEIGHT_EVICTION, EPILOGUE_WARPS, PRODUCER_WARPS>;
    constexpr int MAX_STAGES = C::MAX_STAGES;
    if constexpr (MAX_STAGES > 0) {
        if (stages == MAX_STAGES)
            return dispatch_barriers<BM, MAX_STAGES, CLUSTERS, RED_STORE, ACTIVATION, WEIGHT_EVICTION, EPILOGUE_WARPS, PRODUCER_WARPS>(x, w, w2, out, scratchpad, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, N, K, num_experts, sorted_num, block_m, scaling_factor, activation_params, stream, cluster_fallback);
    }
    if constexpr (MAX_STAGES > 1) {
        if (stages == MAX_STAGES - 1)
            return dispatch_barriers<BM, MAX_STAGES - 1, CLUSTERS, RED_STORE, ACTIVATION, WEIGHT_EVICTION, EPILOGUE_WARPS, PRODUCER_WARPS>(x, w, w2, out, scratchpad, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, N, K, num_experts, sorted_num, block_m, scaling_factor, activation_params, stream, cluster_fallback);
    }
    return false;
}
// clang-format on

}  // namespace moe_sm100_gmem_persistent
