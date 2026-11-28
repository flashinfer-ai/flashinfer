#pragma once

#include <cooperative_groups.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp8.h>
#include <stdio.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cuda/atomic>
#include <cuda/ptx>
#include <cuda/std/limits>

#include "fused_moe_helpers.cuh"
namespace cg = cooperative_groups;

namespace w8a8_persistent {

template <int STAGES, int WN, int BM, int BK, int BN>
struct smem_up {
  alignas(1024) fp8 w[STAGES * WN * BK * BN];
  alignas(1024) fp8 x[STAGES * BK * BM];
};

template <int STAGES, int WN, int BM, int BK, int BN, bool DB_STORE>
struct smem_down {
  alignas(1024) fp8 w[STAGES * WN * BK * BN];
  alignas(1024) fp8 x[BM * WN * BN / 2];
  alignas(16) __nv_bfloat16 out[BM * ((DB_STORE ? 2 : 1) * BK * 2 + 8)];
};

template <int STAGES, int BM, int BK, int BN, int WN>
struct smem_persistent {
  alignas(8) uint64_t bar[2 * STAGES + 1];
  alignas(16) float topk_scales[BM];
};

// Scale buffers follow the fixed tile storage; only their lengths depend on K.
template <int STAGES, int BM, int BK, int BN, int WN, bool DB_STORE>
struct smem_layout {
  using Up = smem_up<STAGES, WN, BM, BK, BN>;
  using Down = smem_down<STAGES, WN, BM, BK, BN, DB_STORE>;
  using Persistent = smem_persistent<STAGES, BM, BK, BN, WN>;
  static constexpr size_t UP_END = offsetof(Up, x) + sizeof(Up::x);
  static constexpr size_t DOWN_END = offsetof(Down, out) + sizeof(Down::out);
  static constexpr size_t PERSISTENT_END =
      offsetof(Persistent, topk_scales) + sizeof(Persistent::topk_scales);
  size_t scale_x_bytes;
  size_t scale_up_bytes;
  size_t scale_down_bytes;
  size_t persistent_bytes;
  size_t overlay;
  size_t up_bytes;
  size_t down_bytes;
  size_t total;

  __host__ __device__ explicit smem_layout(int K)
      : scale_x_bytes(size_t(K / 128) * BM * sizeof(float)),
        scale_up_bytes(size_t(WN * BN / 128) * (K / 128) * sizeof(float)),
        scale_down_bytes(
            size_t((WN * BN / 2 / BK) * (K / BK) > BM ? (WN * BN / 2 / BK) * (K / BK) : BM) *
            sizeof(float)),
        persistent_bytes(PERSISTENT_END + scale_x_bytes + scale_up_bytes),
        overlay((persistent_bytes + 1023) & ~size_t(1023)),
        up_bytes(sizeof(Up)),
        down_bytes((DOWN_END + scale_down_bytes + 1023) & ~size_t(1023)),
        total(overlay + (up_bytes > down_bytes ? up_bytes : down_bytes)) {}
};

template <int BM, int BK, int BN, int WN, int STAGES, int PRODUCER_THREADS, bool DB_STORE,
          bool RED_STORE, bool OVERLAPPED = false>
__global__
__launch_bounds__(WN * 32 + PRODUCER_THREADS, 1) void fused_moe_w8a8_wgmma_up_down_acc_kernel(
    const fp8* __restrict__ x, const float* __restrict__ x_scale,
    const __grid_constant__ CUtensorMap tensor_map_w, const float* __restrict__ w_scale,
    const __grid_constant__ CUtensorMap tensor_map_w2, const float* __restrict__ w2_scale,
    __nv_bfloat16* __restrict__ out, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ num_tokens_post_padded,
    const float* __restrict__ topk_weights, const int top_k, int M, int N, int K,
    float scaling_factor, int* __restrict__ experts_done = nullptr) {
  // Hopper kernels share a fatbin with Blackwell kernels.
#if IS_SM90A
  constexpr int CONSUMER_THREADS = WN * 32;
  constexpr int WARPGROUPS = WN / 4;

  const int SPLIT_K = N / (WN * BN) == 0 ? 1 : N / (WN * BN);
  const int persistent_stride = gridDim.y / SPLIT_K;
  const int k_chunk = blockIdx.y % SPLIT_K;
  const int32_t block_start = blockIdx.y / SPLIT_K;
  const int total_tokens = num_tokens_post_padded[0];
  // Integer ceil: with --use_fast_math the float division can round up on
  // exact multiples, spilling into padding blocks whose expert_id is -1
  const int total_blocks = (total_tokens + BM - 1) / BM;

  if constexpr (OVERLAPPED) {
    // Dispatch doorbell. The overlapped fused allreduce spins on the
    // experts_done flags from a different stream. If the allreduce
    // grid is handed to the block scheduler first, its spinning blocks
    // occupy every SM and this grid can never become resident.
    // // The doorbell only shows the first block is resident. The rest must get
    // SMs before the allreduce blocks. This holds only at equal stream
    // priority. Do not give the allreduce stream a higher priority.
    if (threadIdx.x == 0 && blockIdx.y == 0) {
      cuda::atomic_ref<int, cuda::thread_scope_device> bell(*(experts_done + M * top_k * SPLIT_K));
      bell.store(1, cuda::memory_order_relaxed);
    }
  }

  if (block_start >= total_blocks) return;

  // TODO should not be hardcoded
  constexpr int block_shape[2] = {128, 128};

  constexpr uint32_t S_BITS_UP = 3;
  constexpr uint32_t S_MODE_UP = 4 - S_BITS_UP;

  constexpr uint32_t S_BITS_DOWN = WN * BN >= 256 ? 3 : WN * BN == 128 ? 2 : 1;
  constexpr uint32_t S_MODE_DOWN = 4 - S_BITS_DOWN;

  const int N2 = K;
  const int lane_id = threadIdx.x % 32;
  const bool is_producer = threadIdx.x >= CONSUMER_THREADS;
  const int warp_id = threadIdx.x / 32;
  const int wg_id = warp_id / 4;

  constexpr int BK2 = WN * BN / 2;
  constexpr int BN2 = BK * 2;

  // SMEM sizes
  constexpr int WS = WN * BK * BN;
  constexpr int XS = BK * BM;
  // Number of tiles along dim-2 of the 3D TMA descriptor
  constexpr int TILES_UP = BN * WN / 256;
  constexpr int TILES_DOWN = BN * WN / 256;
  // how many bytes we transfer per CP_ASYNC
  constexpr int TB = 16;
  // Thread offset per transfer
  constexpr int TO = TB / sizeof(fp8);
  // Tokens per thread
  constexpr int TPT = BM / 4;

  const int n_x_scales = K / 128;

  const int scale_rows_w_up = N / block_shape[1];
  const int scale_cols_w_up = K / block_shape[0];
  const int scale_rows_w_down = K / block_shape[1];
  const int scale_cols_w_down = (N / 2) / block_shape[0];
  constexpr int scale_cols_per_block_w_down = BK2 / block_shape[0];
  const int scales_down_to_load = scale_cols_per_block_w_down * scale_rows_w_down;

  constexpr int scales_per_stage_up = WN * BN / block_shape[0];
  constexpr int scale_rows_per_stage_down = BN2 / block_shape[0];

  constexpr int TILE_COLS_UP = BN * WN / TILES_UP;
  constexpr int TILE_INNER_DN = BK2 / TILES_DOWN;

  extern __shared__ __align__(1024) uint8_t sh[];
  using Layout = smem_layout<STAGES, BM, BK, BN, WN, DB_STORE>;
  const Layout layout(K);
  const size_t OVERLAY_OFF = layout.overlay;
  auto& smem_p = *reinterpret_cast<smem_persistent<STAGES, BM, BK, BN, WN>*>(sh);
  smem_up<STAGES, WN, BM, BK, BN>& s =
      *reinterpret_cast<smem_up<STAGES, WN, BM, BK, BN>*>(sh + OVERLAY_OFF);
  smem_down<STAGES, WN, BM, BK, BN, DB_STORE>& s_d =
      *reinterpret_cast<smem_down<STAGES, WN, BM, BK, BN, DB_STORE>*>(sh + OVERLAY_OFF);

  uint64_t* bar = smem_p.bar;
  uint64_t* epilogue_bar = smem_p.bar + 2 * STAGES;
  float* topk_scales = smem_p.topk_scales;
  float* scale_w_down = reinterpret_cast<float*>(sh + OVERLAY_OFF + Layout::DOWN_END);
  float* scale_x_up = reinterpret_cast<float*>(sh + Layout::PERSISTENT_END);
  float* scale_w_up =
      reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(scale_x_up) + layout.scale_x_bytes);

  if (threadIdx.x == 0) {
    for (int i = 0; i < STAGES; i++) {
      init_barrier(&bar[i], PRODUCER_THREADS + 1, 0);
      init_barrier(&bar[i + STAGES], CONSUMER_THREADS, 0);
    }
    init_barrier(epilogue_bar, 128, 0);
  }

  __syncthreads();
  auto consumer_sync = [&]() { asm volatile("bar.sync 0, %0;\n" ::"n"(CONSUMER_THREADS)); };
  auto global_sync = [&]() {
    asm volatile("bar.sync 1, %0;\n" ::"n"(PRODUCER_THREADS + CONSUMER_THREADS));
  };
  auto producer_sync = [&]() { asm volatile("bar.sync 2, %0;\n" ::"n"(PRODUCER_THREADS)); };
  auto warpgroup_sync = [&]() { asm volatile("bar.sync %0, %1;\n" ::"r"(3 + wg_id), "n"(128)); };

  const int n_stages_up = K / block_shape[0];
  const int n_stages_down = N2 / BN2;

  constexpr int TM = BM / 8;
  constexpr int TN = BN / 16;
  nv_bfloat162 f_acc[TN][TM][2];
  memset(f_acc, 0, sizeof(f_acc));

  // If we're running low on registers steal them from the producer
  // and not the consumer
  constexpr int PRODUCER_REGS = BM > 96 ? 24 : 56;
  constexpr int CONSUMER_REGS = BM > 96 ? 240 : 224;
  constexpr bool CHANGE_REG_ALLOC = BM > 88 || WN * BN >= 512;

  int p = 0;
  int smem_stage = 0;
  // PRODUCER
  if (is_producer) {
    // TODO does it really matter?
    if constexpr (CHANGE_REG_ALLOC) {
      reg_dealloc<PRODUCER_REGS>();
    }
    int producer_tid = threadIdx.x - CONSUMER_THREADS;
    constexpr int X_IT = const_ceil(XS / float(PRODUCER_THREADS * TO));
    static_assert((PRODUCER_THREADS * TO % 1024) == 0,
                  "X-load stride would intersect the swizzle mask; "
                  "recompute swizzle<S_BITS_UP>(i) per iteration instead");
    uint32_t tsrc[X_IT];
    int* token_ids = reinterpret_cast<int*>(scale_w_down);
    {
      const int exp_idx = expert_ids[block_start];
      int i = (producer_tid)*TO;
      if (producer_tid < BM) {
        token_ids[producer_tid] = sorted_token_ids[block_start * BM + producer_tid];
      }
      // Load token scales
      producer_sync();
      for (int r = 0; r < X_IT; r += 1) {
        if (i < XS) {
          int tdest = token_ids[r * (PRODUCER_THREADS / 8) + producer_tid / 8];
          tsrc[r] = tdest / top_k;
          if (producer_tid % 8 == 0 && tsrc[r] < M && i < XS) {
            uint32_t smem = __cvta_generic_to_shared(
                &topk_scales[r * (PRODUCER_THREADS / 8) + producer_tid / 8]);
            CP_ASYNC_CG4(smem, topk_weights + tdest, 4);
          }
        }
        i += PRODUCER_THREADS * TO;
      }
      // if the regions are overlapping we need to sync otherwise we could race
      // read/write from token_ids
      if constexpr (sizeof(s_d.x) + sizeof(s_d.out) < sizeof(s.x)) {
        producer_sync();
      }
      // Scale rows with two trailing floats require scalar copies for alignment.
      const int scale_vector = scale_cols_w_up % 4 == 0 ? 4 : 1;
      for (int col = producer_tid * scale_vector; col < scale_cols_w_up;
           col += PRODUCER_THREADS * scale_vector) {
        for (int scale_row = 0; scale_row < scales_per_stage_up; scale_row++) {
          uint32_t smem = __cvta_generic_to_shared(scale_w_up + scale_row * scale_cols_w_up + col);
          const float* src = w_scale + exp_idx * scale_rows_w_up * scale_cols_w_up +
                             (k_chunk * scales_per_stage_up + scale_row) * scale_cols_w_up + col;
          if (scale_vector == 4) {
            CP_ASYNC_CG(smem, src, 16);
          } else {
            CP_ASYNC_CG4(smem, src, 4);
          }
        }
      }
      for (int r = 0; r < X_IT; r += 1) {
        int token_idx = r * (PRODUCER_THREADS / 8) + producer_tid / 8;
        int row = tsrc[r];
        if (row < M && token_idx < BM) {
          for (int off = producer_tid % 8; off < n_x_scales; off += 8) {
            // Load and transpose to remove bank conficts
            uint32_t smem = __cvta_generic_to_shared(scale_x_up + token_idx + BM * off);
            CP_ASYNC_CG4(smem, &x_scale[n_x_scales * row + off], 4);
          }
        }
      }
      const int w_row_up = exp_idx * N + k_chunk * WN * BN;
      // Up projection X loading + Weights loading
      for (int load_stage = 0; load_stage < n_stages_up; load_stage++) {
        if (smem_stage == STAGES) {
          p ^= 1;
          smem_stage = 0;
        }
        const int off = load_stage * block_shape[0];
        int i = (producer_tid)*TO;
        int col = off + i % BK;
        int swizzled = swizzle<S_BITS_UP>(i);

        wait(bar + STAGES + smem_stage, p);

        for (int r = 0; r < X_IT; r += 1) {
          int row = tsrc[r];
          if (row < M && swizzled < XS) {
            uint32_t sm = __cvta_generic_to_shared(s.x + smem_stage * XS + swizzled);
            CP_ASYNC_CG(sm, reinterpret_cast<const float4*>(x + row * K + col), TB);
          }
          swizzled += PRODUCER_THREADS * TO;
        }

        cp_async_mbarrier_arrive(bar + smem_stage);

        if (producer_tid == 0) {
          cg::invoke_one(cg::coalesced_threads(), [&]() {
            expect_bytes(bar + smem_stage, WS * sizeof(fp8));
            load_async_3d(s.w + smem_stage * WS, &tensor_map_w, bar + smem_stage, off, 0,
                          w_row_up / TILE_COLS_UP);
          });
        }
        smem_stage++;
      }

      // prefetch token ids for next block if there is one
      if (block_start + persistent_stride < total_blocks) {
        int i = (producer_tid)*TO;
        for (int r = 0; r < X_IT; r += 1) {
          if (i < XS) {
            tsrc[r] = sorted_token_ids[(block_start + persistent_stride) * BM +
                                       r * (PRODUCER_THREADS / 8) + producer_tid / 8];
          }
          i += PRODUCER_THREADS * TO;
        }
      }

      const int base =
          exp_idx * scale_rows_w_down * scale_cols_w_down + k_chunk * scale_cols_per_block_w_down;
      for (int idx = producer_tid; idx < scales_down_to_load; idx += PRODUCER_THREADS) {
        uint32_t smem = __cvta_generic_to_shared(scale_w_down + idx);
        int row = idx % scale_rows_w_down;
        int col = idx / scale_rows_w_down;
        CP_ASYNC_CG4(smem, &w2_scale[base + row * scale_cols_w_down + col], 4);
      }

      // Down projection weights loading
      for (int load_stage = 0; load_stage < n_stages_down; load_stage++) {
        if (smem_stage == STAGES) {
          p ^= 1;
          smem_stage = 0;
        }
        int col = load_stage;
        wait(bar + STAGES + smem_stage, p);
        if (producer_tid == 0) {
          cg::invoke_one(cg::coalesced_threads(), [&]() {
            const int w_row_down = k_chunk * BK2;
            const int w_col_down = exp_idx * N2 + col * BN2;
            expect_bytes(bar + smem_stage, WS * sizeof(fp8));
            load_async_3d(s_d.w + smem_stage * WS, &tensor_map_w2, bar + smem_stage, 0, w_col_down,
                          w_row_down / TILE_INNER_DN);
          });
        }
        cp_async_mbarrier_arrive(bar + smem_stage);
        smem_stage++;
      }
    }
    // Persistent kernel loading
    for (int local_block = block_start + persistent_stride; local_block < total_blocks;
         local_block += persistent_stride) {
      const int exp_idx = expert_ids[local_block];
      int i = (producer_tid)*TO;
      const int w_row_up = exp_idx * N + k_chunk * WN * BN;

      for (int r = 0; r < X_IT; r += 1) {
        if (i < XS) {
          int tdest = tsrc[r];
          tsrc[r] = tdest / top_k;
          if (producer_tid % 8 == 0 && tsrc[r] < M && i < XS) {
            uint32_t smem = __cvta_generic_to_shared(
                &topk_scales[r * (PRODUCER_THREADS / 8) + producer_tid / 8]);
            CP_ASYNC_CG4(smem, topk_weights + tdest, 4);
          }
        }
        i += PRODUCER_THREADS * TO;
      }

      // Part where x, out and the scales might overlap so we only load weights
      for (int load_stage = 0; load_stage < STAGES; load_stage++) {
        if (smem_stage == STAGES) {
          p ^= 1;
          smem_stage = 0;
        }
        const int off = load_stage * block_shape[0];

        wait(bar + STAGES + smem_stage, p);

        if (producer_tid == 0) {
          cg::invoke_one(cg::coalesced_threads(), [&]() {
            expect_bytes(bar + smem_stage, WS * sizeof(fp8));
            load_async_3d(s.w + smem_stage * WS, &tensor_map_w, bar + smem_stage, off, 0,
                          w_row_up / TILE_COLS_UP);
          });
        }
        smem_stage++;
      }

      // Scale rows with two trailing floats require scalar copies for alignment.
      const int scale_vector = scale_cols_w_up % 4 == 0 ? 4 : 1;
      for (int col = producer_tid * scale_vector; col < scale_cols_w_up;
           col += PRODUCER_THREADS * scale_vector) {
        for (int scale_row = 0; scale_row < scales_per_stage_up; scale_row++) {
          uint32_t smem = __cvta_generic_to_shared(scale_w_up + scale_row * scale_cols_w_up + col);
          const float* src = w_scale + exp_idx * scale_rows_w_up * scale_cols_w_up +
                             (k_chunk * scales_per_stage_up + scale_row) * scale_cols_w_up + col;
          if (scale_vector == 4) {
            CP_ASYNC_CG(smem, src, 16);
          } else {
            CP_ASYNC_CG4(smem, src, 4);
          }
        }
      }
      for (int r = 0; r < X_IT; r += 1) {
        int token_idx = r * (PRODUCER_THREADS / 8) + producer_tid / 8;
        int row = tsrc[r];
        if (row < M && token_idx < BM) {
          for (int off = producer_tid % 8; off < n_x_scales; off += 8) {
            // Load and transpose to remove bank conficts
            uint32_t smem = __cvta_generic_to_shared(scale_x_up + token_idx + BM * off);
            CP_ASYNC_CG4(smem, &x_scale[n_x_scales * row + off], 4);
          }
        }
      }

      // if x and out are overlapped we need to wait for store to finish
      if constexpr (sizeof(s_d.x) < sizeof(s.x)) {
        global_sync();
      }

      // Now that all stages are done start loading X
      for (int load_stage = 0; load_stage < STAGES; load_stage++) {
        if (smem_stage == STAGES) {
          smem_stage = 0;
        }
        const int off = load_stage * block_shape[0];
        int i = (producer_tid)*TO;
        int col = off + i % BK;
        int swizzled = swizzle<S_BITS_UP>(i);

        for (int r = 0; r < X_IT; r += 1) {
          int row = tsrc[r];
          if (row < M && swizzled < XS) {
            uint32_t sm = __cvta_generic_to_shared(s.x + smem_stage * XS + swizzled);
            CP_ASYNC_CG(sm, reinterpret_cast<const float4*>(x + row * K + col), TB);
          }
          swizzled += PRODUCER_THREADS * TO;
        }

        cp_async_mbarrier_arrive(bar + smem_stage);
        smem_stage++;
      }

      // Up projection loading
      for (int load_stage = STAGES; load_stage < n_stages_up; load_stage++) {
        if (smem_stage == STAGES) {
          p ^= 1;
          smem_stage = 0;
        }
        const int off = load_stage * block_shape[0];
        int i = (producer_tid)*TO;
        int col = off + i % BK;
        int swizzled = swizzle<S_BITS_UP>(i);

        wait(bar + STAGES + smem_stage, p);

        for (int r = 0; r < X_IT; r += 1) {
          int row = tsrc[r];
          if (row < M && swizzled < XS) {
            uint32_t sm = __cvta_generic_to_shared(s.x + smem_stage * XS + swizzled);
            CP_ASYNC_CG(sm, reinterpret_cast<const float4*>(x + row * K + col), TB);
          }
          swizzled += PRODUCER_THREADS * TO;
        }

        cp_async_mbarrier_arrive(bar + smem_stage);

        if (producer_tid == 0) {
          cg::invoke_one(cg::coalesced_threads(), [&]() {
            expect_bytes(bar + smem_stage, WS * sizeof(fp8));
            load_async_3d(s.w + smem_stage * WS, &tensor_map_w, bar + smem_stage, off, 0,
                          w_row_up / TILE_COLS_UP);
          });
        }
        smem_stage++;
      }

      // pre fetch token ids
      if (local_block + persistent_stride < total_blocks) {
        int i = (producer_tid)*TO;
        for (int r = 0; r < X_IT; r += 1) {
          if (i < XS) {
            tsrc[r] = sorted_token_ids[(local_block + persistent_stride) * BM +
                                       r * (PRODUCER_THREADS / 8) + producer_tid / 8];
          }
          i += PRODUCER_THREADS * TO;
        }
      }

      const int base =
          exp_idx * scale_rows_w_down * scale_cols_w_down + k_chunk * scale_cols_per_block_w_down;
      for (int idx = producer_tid; idx < scales_down_to_load; idx += PRODUCER_THREADS) {
        uint32_t smem = __cvta_generic_to_shared(scale_w_down + idx);
        int row = idx % scale_rows_w_down;
        int col = idx / scale_rows_w_down;
        CP_ASYNC_CG4(smem, &w2_scale[base + row * scale_cols_w_down + col], 4);
      }

      // Down projection loading
      for (int load_stage = 0; load_stage < n_stages_down; load_stage++) {
        if (smem_stage == STAGES) {
          p ^= 1;
          smem_stage = 0;
        }
        int col = load_stage;
        wait(bar + STAGES + smem_stage, p);
        if (producer_tid == 0) {
          cg::invoke_one(cg::coalesced_threads(), [&]() {
            const int w_row_down = k_chunk * BK2;
            const int w_col_down = exp_idx * N2 + col * BN2;
            expect_bytes(bar + smem_stage, WS * sizeof(fp8));
            load_async_3d(s_d.w + smem_stage * WS, &tensor_map_w2, bar + smem_stage, 0, w_col_down,
                          w_row_down / TILE_INNER_DN);
          });
        }
        cp_async_mbarrier_arrive(bar + smem_stage);
        smem_stage++;
      }
    }
  }
  // CONSUMER
  else {
    if constexpr (CHANGE_REG_ALLOC) {
      reg_alloc<CONSUMER_REGS>();
    }
    // Empty barriers arrive instantly
    constexpr int SAVES_PER_WARP = BM / WN;
    for (int i = 0; i < STAGES; i++) arrive(&bar[STAGES + i]);

    int e_p = wg_id;
    if (wg_id == 0) {
      arrive(epilogue_bar);
    }

    for (int local_block = block_start; local_block < total_blocks;
         local_block += persistent_stride) {
      int token_src = M;
      int token_raw = M * top_k;
      if (lane_id < SAVES_PER_WARP) {
        token_raw = sorted_token_ids[local_block * BM + wg_id * (BM / 2) +
                                     (warp_id % 4) * SAVES_PER_WARP + lane_id];
        token_src = token_raw / top_k;
      }
      int toks[SAVES_PER_WARP];
      for (int k = 0; k < SAVES_PER_WARP; k++) {
        toks[k] = __shfl_sync(0xFFFFFFFF, token_src, k);
      }

      memset(f_acc, 0, sizeof(f_acc));
      for (int compute_stage = 0; compute_stage < n_stages_up; compute_stage += 1) {
        if (smem_stage == STAGES) {
          p ^= 1;
          smem_stage = 0;
        }
        wait(bar + smem_stage, p);

        const int scale_off = ((warp_id / 4) * (BN * 4) / 256) * 2 * K / 128;
        float scale_w[2];
        scale_w[0] = scale_w_up[scale_off + compute_stage];
        scale_w[1] = scale_w_up[scale_off + K / 128 + compute_stage];

        nv_bfloat162 scale_xw[TM][2];
        for (int tm = 0; tm < TM; tm++) {
          int token_idx = tm * 8 + (lane_id % 4) * 2;
          float2 x_sc = *reinterpret_cast<float2*>(scale_x_up + compute_stage * BM + token_idx);
          scale_xw[tm][0] = __nv_bfloat162(scale_w[0] * x_sc.x, scale_w[0] * x_sc.y);
          scale_xw[tm][1] = __nv_bfloat162(scale_w[1] * x_sc.x, scale_w[1] * x_sc.y);
        }

        constexpr int TN_I = BM < 32 ? TN : 1;
        float tile_acc[TN_I][TM][4];
        for (int tn = 0; tn < TN; tn += TN_I) {
          // https://github.com/NVIDIA/cutlass/discussions/1375
          // This is cursed
          for (int tni = 0; tni < TN_I; tni++) {
            for (int tm = 0; tm < TM; tm++) {
              for (int t = 0; t < 4; t++) asm volatile("" : "+f"(tile_acc[tni][tm][t])::"memory");
            }
          }
          cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
          warpgroup_arrive();
          for (int tni = 0; tni < TN_I; tni++) {
            fp8* sw = s.w + smem_stage * WS + (warp_id / 4) * (BN * 4) * BK + (tn + tni) * 64 * BK;
            fp8* sx = s.x + smem_stage * XS;
            uint64_t desc_w = make_smem_descriptor<64, 1, S_MODE_UP>(sw);
            uint64_t desc_x = make_smem_descriptor<64, 1, S_MODE_UP>(sx);
            wgmma<0, 1, 1, BM>(tile_acc[tni], desc_w, desc_x);
            wgmma<1, 1, 1, BM>(tile_acc[tni], desc_w + 1 * (32 >> 4), desc_x + 1 * (32 >> 4));
            wgmma<1, 1, 1, BM>(tile_acc[tni], desc_w + 2 * (32 >> 4), desc_x + 2 * (32 >> 4));
            wgmma<1, 1, 1, BM>(tile_acc[tni], desc_w + 3 * (32 >> 4), desc_x + 3 * (32 >> 4));
          }
          warpgroup_commit_batch();
          warpgroup_wait();
          if (tn == TN - TN_I) arrive(bar + STAGES + smem_stage);

          for (int tni = 0; tni < TN_I; tni++) {
            for (int tm = 0; tm < TM; tm++) {
              f_acc[tn + tni][tm][0] =
                  __hfma2(__nv_bfloat162(tile_acc[tni][tm][0], tile_acc[tni][tm][1]),
                          scale_xw[tm][0], f_acc[tn + tni][tm][0]);
              f_acc[tn + tni][tm][1] =
                  __hfma2(__nv_bfloat162(tile_acc[tni][tm][2], tile_acc[tni][tm][3]),
                          scale_xw[tm][1], f_acc[tn + tni][tm][1]);
            }
          }
        }
        smem_stage++;
      }

      constexpr float EPS = 1e-10;
      nv_bfloat162 token_max[TM] = {nv_bfloat162(EPS, EPS)};
      for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn++) {
          {
            const nv_bfloat162 one2 = __float2bfloat162_rn(1.0f);
            nv_bfloat162 xg = f_acc[tn][tm][0];
            nv_bfloat162 e = h2exp(__hneg2(xg));
            nv_bfloat162 sig = h2rcp(__hadd2(e, one2));
            f_acc[tn][tm][0] = __hmul2(__hmul2(xg, sig), f_acc[tn][tm][1]);
          }
          nv_bfloat162 abs = __habs2(f_acc[tn][tm][0]);
          token_max[tm] = __hmax2(abs, token_max[tm]);
        }
      }
      for (int tm = 0; tm < TM; tm++) {
        token_max[tm] = __hmax2(__shfl_xor_sync(0xFFFFFFFF, token_max[tm], 16), token_max[tm]);
        token_max[tm] = __hmax2(__shfl_xor_sync(0xFFFFFFFF, token_max[tm], 8), token_max[tm]);
        token_max[tm] = __hmax2(__shfl_xor_sync(0xFFFFFFFF, token_max[tm], 4), token_max[tm]);
      }

      constexpr int MAX_EXCHANGE = WARPGROUPS * BM * sizeof(nv_bfloat162) / sizeof(bf16);
      float4* block_max =
          reinterpret_cast<float4*>(s_d.out + sizeof(s_d.out) / sizeof(bf16)) - MAX_EXCHANGE;
      // If we're not overlapping with down proj we don't need to sync before writing to
      // block_max
      if constexpr (Layout::DOWN_END - MAX_EXCHANGE * sizeof(float4) < Layout::UP_END) {
        consumer_sync();
      }
      if (local_block > block_start) {
        asm volatile("cp.async.bulk.wait_group.read 0;");
        consumer_sync();
      }

      for (int tm = 0; tm < TM; tm++) {
        if (lane_id < 4) {
          int off = tm * 8 + (lane_id) * 2;
          reinterpret_cast<nv_bfloat162*>(block_max + off * WARPGROUPS)[warp_id] = token_max[tm];
        }
      }
      consumer_sync();

      constexpr float fp8_max_inv = 1 / 448.0;
      // 32 elements per TN after swiglu
      constexpr int QUANTS_PER_WG = const_ceil(float(TN * 32) / block_shape[0]);
      constexpr int QUANTS_TOTAL = BK2 / block_shape[0];
      // Reciprocal quant scale of the single quant group this wg
      // quantizes into (its own column slice).
      float token_scale_inv[TM][2];

      constexpr int warpgroups_to_cover_q = block_shape[0] / (TN * 32);
      for (int tm = 0; tm < TM; tm++) {
        int off = tm * 8 + (lane_id % 4) * 2;
        // Each wg quantizes its register rows into exactly one quant
        // group, so only that group's scale is needed here. Computing
        // all QUANTS_TOTAL scales (as before) doubled the block_max
        // reads and hmax reductions in this fully exposed
        // (tensor-idle) phase.
        {
          const int quant = QUANTS_TOTAL == 1 ? 0 : wg_id * QUANTS_PER_WG;
          nv_bfloat162 qmax = nv_bfloat162(EPS, EPS);
          int wg_off = quant * warpgroups_to_cover_q;
          for (int i = 0; i < warpgroups_to_cover_q; i++) {
            float4 bmax = block_max[off * WARPGROUPS + wg_off + i];
            qmax = __hmax2(*reinterpret_cast<nv_bfloat162*>(&bmax.x), qmax);
            qmax = __hmax2(*reinterpret_cast<nv_bfloat162*>(&bmax.y), qmax);
            qmax = __hmax2(*reinterpret_cast<nv_bfloat162*>(&bmax.z), qmax);
            qmax = __hmax2(*reinterpret_cast<nv_bfloat162*>(&bmax.w), qmax);
          }
          // Store reciprocals: one division per (tm, t) here instead
          // of one per (tm, t, tn) element in the loop below.
          token_scale_inv[tm][0] = 1.0f / (float(qmax.x) * fp8_max_inv);
          token_scale_inv[tm][1] = 1.0f / (float(qmax.y) * fp8_max_inv);
        }
        for (int t = 0; t < 2; t++) {
          for (int tn = 0; tn < TN; tn++) {
            float val = t == 0 ? f_acc[tn][tm][0].x : f_acc[tn][tm][0].y;
            float q = val * token_scale_inv[tm][t];
            int x_row = tm * 8 + (lane_id % 4) * 2 + t;
            int x_col_local = tn * 32 + (warp_id % 4) * 8 + lane_id / 4;
            // If we have more than one quant total, next one should be in the next
            // "block"
            // TODO explain better
            int wg_offset = QUANTS_TOTAL > 1 ? wg_id * QUANTS_PER_WG * BM * block_shape[0]
                                             : (warp_id / 4) * (TN * 32);
            int i = x_row * block_shape[0] + x_col_local;
            int swizzled = swizzle<S_BITS_DOWN>(wg_offset + i);
            s_d.x[swizzled] = fp8(q);
          }
        }
      }

      constexpr int MW = BM / 2;
      constexpr int TM2 = MW / 8;
      // Down projection is split by rows so each warp group needs the
      // quant scales of its own MW-token slice (not its register rows).
      float token_scale_down[QUANTS_TOTAL][TM2][2];
      for (int tm = 0; tm < TM2; tm++) {
        int off = wg_id * MW + tm * 8 + (lane_id % 4) * 2;
        const float2 topk_w = *reinterpret_cast<const float2*>(&topk_scales[off]);
        for (int quant = 0; quant < QUANTS_TOTAL; quant++) {
          nv_bfloat162 qmax = nv_bfloat162(EPS, EPS);
          int wg_off = quant * warpgroups_to_cover_q;
          for (int i = 0; i < warpgroups_to_cover_q; i++) {
            float4 bmax = block_max[off * WARPGROUPS + wg_off + i];
            qmax = __hmax2(*reinterpret_cast<nv_bfloat162*>(&bmax.x), qmax);
            qmax = __hmax2(*reinterpret_cast<nv_bfloat162*>(&bmax.y), qmax);
            qmax = __hmax2(*reinterpret_cast<nv_bfloat162*>(&bmax.z), qmax);
            qmax = __hmax2(*reinterpret_cast<nv_bfloat162*>(&bmax.w), qmax);
          }
          token_scale_down[quant][tm][0] = float(qmax.x) * fp8_max_inv * topk_w.x * scaling_factor;
          token_scale_down[quant][tm][1] = float(qmax.y) * fp8_max_inv * topk_w.y * scaling_factor;
        }
      }

      consumer_sync();

      constexpr int TND = BN2 / 64;
      float tile_acc[TND][TM2][4];
      __nv_bfloat162 out_tile[TND][TM2][2];
      constexpr int SAVE_ELEMS = (DB_STORE ? BN2 * 2 : BN2);
      constexpr int PAD = SAVE_ELEMS + 8;

      for (int compute_stage = 0; compute_stage < n_stages_down; compute_stage += 1) {
        if (smem_stage == STAGES) {
          p ^= 1;
          smem_stage = 0;
        }
        int col = compute_stage;
        float s_w[TND / 2];

        wait(bar + smem_stage, p);

#pragma unroll
        for (int quant = 0; quant < QUANTS_TOTAL; quant++) {
          // When QUANTS_TOTAL > 1, quant groups are stored in separate
          // contiguous BM*block_shape[0] regions (each with its own swizzle).
          // Each warp group reads its own MW-token slice.
          fp8* sx = s_d.x + quant * BM * block_shape[0] + wg_id * MW * block_shape[0];
          for (int i = 0; i < (TND / 2); i++) {
            s_w[i] = scale_w_down[quant * scale_rows_w_down + col * scale_rows_per_stage_down + i];
          }
          // In theory this should not be necessary because
          // we're using WGMMA with an accumulator scale of 0
          // however this produces an erroneous SASS with a
          // presumably unconfigured WGMMA instruction
          // current belief is that we're triggering some PTXAS bug
          memset(tile_acc, 0, sizeof(tile_acc));
          for (int tni = 0; tni < TND; tni++) {
            for (int tm = 0; tm < TM2; tm++) {
              for (int t = 0; t < 4; t++) asm volatile("" : "+f"(tile_acc[tni][tm][t])::"memory");
            }
          }
          cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
          warpgroup_arrive();

          for (int tni = 0; tni < TND; tni++) {
            fp8* sw =
                s_d.w + smem_stage * WS + quant * WS / TILES_DOWN + tni * 64 * (BK2 / TILES_DOWN);
            uint64_t desc_w = make_smem_descriptor<BK2 / (2 * TILES_DOWN), 1, S_MODE_DOWN>(sw);
            uint64_t desc_x = make_smem_descriptor<block_shape[0] / 2, 1, S_MODE_DOWN>(sx);
            wgmma<0, 1, 1, MW>(tile_acc[tni], desc_w, desc_x);
            desc_w += (32 >> 4);
            desc_x += (32 >> 4);
            for (int tk = 1; tk < 4; tk++) {
              wgmma<1, 1, 1, MW>(tile_acc[tni], desc_w, desc_x);
              desc_w += (32 >> 4);
              desc_x += (32 >> 4);
            }
          }
          warpgroup_commit_batch();
          warpgroup_wait();

          if (quant == QUANTS_TOTAL - 1) arrive(bar + STAGES + smem_stage);

          for (int tni = 0; tni < TND; tni++) {
            for (int tm = 0; tm < TM2; tm++) {
              nv_bfloat162 sc = __nv_bfloat162(token_scale_down[quant][tm][0] * s_w[tni / 2],
                                               token_scale_down[quant][tm][1] * s_w[tni / 2]);
              nv_bfloat162 p0 = __nv_bfloat162(tile_acc[tni][tm][0], tile_acc[tni][tm][1]);
              nv_bfloat162 p1 = __nv_bfloat162(tile_acc[tni][tm][2], tile_acc[tni][tm][3]);
              if (quant == 0) {
                out_tile[tni][tm][0] = __hmul2(p0, sc);
                out_tile[tni][tm][1] = __hmul2(p1, sc);
              } else {
                out_tile[tni][tm][0] = __hfma2(p0, sc, out_tile[tni][tm][0]);
                out_tile[tni][tm][1] = __hfma2(p1, sc, out_tile[tni][tm][1]);
              }
            }
          }
        }
        smem_stage++;

        constexpr int ST_GRANULARITY = DB_STORE ? 2 : 1;
        if (compute_stage > 0 && compute_stage % ST_GRANULARITY == 0) {
          if constexpr (RED_STORE) {
            warpgroup_sync();
          } else {
            asm volatile("cp.async.bulk.wait_group.read 0;");
            warpgroup_sync();
          }
        }
        int offset = DB_STORE ? (compute_stage % 2) * BN2 : 0;

        static_assert(TND % 2 == 0);
        for (int tn2 = 0; tn2 < TND; tn2 += 2) {
          for (int tm = 0; tm < TM2; tm++) {
            int out_row = wg_id * (BM / 2) + tm * 8 + lane_id % 8;
            int out_col = (warp_id % 4) * 16 + (lane_id & 8) + (tn2 + lane_id / 16) * 64 + offset;
            const uint32_t* lo = reinterpret_cast<uint32_t*>(out_tile[tn2][tm]);
            const uint32_t* hi = reinterpret_cast<uint32_t*>(out_tile[tn2 + 1][tm]);
            st_matrix_x4_trans(lo[0], lo[1], hi[0], hi[1],
                               __cvta_generic_to_shared(s_d.out + out_row * PAD + out_col));
          }
        }

        if ((DB_STORE && compute_stage % 2 == 1) || !DB_STORE ||
            compute_stage == n_stages_down - 1) {
          constexpr bool STAGGER = !RED_STORE || DB_STORE;
          if constexpr (STAGGER) {
            wait(epilogue_bar, e_p);
          }
          if constexpr (!RED_STORE) {
            cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
          }
          warpgroup_sync();

          int store_col = DB_STORE ? compute_stage - compute_stage % 2 : compute_stage;
          const int store_elems = DB_STORE && compute_stage % 2 == 1 ? SAVE_ELEMS : BN2;
          if constexpr (RED_STORE) {
            const int VECS_PER_ROW = store_elems / 8;
            for (int k = 0; k < SAVES_PER_WARP; k++) {
              int tok = toks[k];
              if (tok < M) {
                int row = warp_id * SAVES_PER_WARP + k;
                for (int v = lane_id; v < VECS_PER_ROW; v += 32) {
                  float4 val = *reinterpret_cast<const float4*>(s_d.out + row * PAD + v * 8);
                  red_add_bf16x8(out + tok * N2 + store_col * BN2 + v * 8, val);
                }
              }
            }
          } else {
            if (lane_id < SAVES_PER_WARP) {
              if (token_src < M) {
                int row = (warp_id)*SAVES_PER_WARP + lane_id;
                cuda::ptx::cp_reduce_async_bulk(
                    cuda::ptx::space_global, cuda::ptx::space_shared, cuda::ptx::op_add,
                    out + token_src * N2 + store_col * BN2, s_d.out + row * PAD,
                    store_elems * sizeof(__nv_bfloat16));
                cuda::ptx::cp_async_bulk_commit_group();
              }
            }
          }
          if constexpr (STAGGER) {
              // The last store neither waits for its turn nor passes it on: an arrive
              // without a wait lets this warpgroup complete two phases before the other
              // one checks its parity, and that one then waits forever.
              if (compute_stage != n_stages_down - 1)
                  arrive(epilogue_bar);
          }
        }
      }
      // if x and out are overlapped we need to signal that store has finished
      if constexpr (sizeof(s_d.x) < sizeof(s.x)) {
        if (local_block + persistent_stride < total_blocks) {
          if constexpr (!RED_STORE) {
            asm volatile("cp.async.bulk.wait_group.read 0;");
          }
          global_sync();
        }
      }
      if constexpr (OVERLAPPED) {
        if constexpr (RED_STORE) {
          // A token's row is written by every lane of its warp via
          // red.add, so order those before this lane's flag store.
          // Warp-level ordering is enough; no block barrier needed.
          __syncwarp();
          if (token_raw < M * top_k) {
            cuda::atomic_ref<int, cuda::thread_scope_device> flag(
                *(experts_done + token_raw * SPLIT_K + k_chunk));
            flag.store(1, cuda::memory_order_release);
          }
        } else {
          // A token's row is stored exclusively by its own lane's
          // cp_reduce_async_bulk groups, so a per-thread full wait
          // (writes visible in global memory, unlike the .read wait
          // above) followed by a release store is enough. Other
          // lanes proceed without any extra block-wide sync.
          if (token_raw < M * top_k) {
            asm volatile("cp.async.bulk.wait_group 0;");
            cuda::atomic_ref<int, cuda::thread_scope_device> flag(
                *(experts_done + token_raw * SPLIT_K + k_chunk));
            flag.store(1, cuda::memory_order_release);
          }
        }
      }
    }
  }
#else
  assert(false && "This fused MoE kernel requires SM90");
#endif
}

template <int BM, int BN, int WN, int STAGES, bool DB_STORE, bool RED_STORE>
bool launch_fused_moe_kernel_up_down_acc_impl(
    const fp8* x, const float* x_scale, fp8* w, const float* w_scale, fp8* w2,
    const float* w2_scale, __nv_bfloat16* out, const int* sorted_token_ids, const int* expert_ids,
    const int* num_tokens_post_padded, const float* topk_weights, const int top_k, int M, int N,
    int K, int num_experts, int sorted_num, int block_m, float scaling_factor, cudaStream_t stream,
    int* experts_done = nullptr) {
  constexpr int BK = 128;
  constexpr int PRODUCER_THREADS = 128;
  if (K <= 0 || K % (2 * BK) || N <= 0 || N % 256 || N % (BN * WN)) return false;
  int device, num_sms;
  gpuErrchk(cudaGetDevice(&device));
  gpuErrchk(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  dim3 dimBlock(32 * WN + PRODUCER_THREADS, 1, 1);
  const int SPLIT_K = N / (BN * WN) == 0 ? 1 : N / (BN * WN);
  const int block_count = std::min(num_sms / SPLIT_K, (int)std::ceil((float)sorted_num / block_m));
  if (block_count <= 0) return false;
  dim3 dimGrid(1, block_count * SPLIT_K, 1);

  using Layout = smem_layout<STAGES, BM, BK, BN, WN, DB_STORE>;
  const Layout layout(K);
  const size_t OVERLAY_OFF = layout.overlay;
  const size_t sMemSize = layout.total;

  if (sMemSize > SM90_MAX_SMEM) {
    fprintf(stderr,
            "fused_moe: BM=%d BN=%d WN=%d STAGES=%d requires %zu bytes of "
            "shared memory but SM90 max is %zu — skipping launch\n",
            BM, BN, WN, STAGES, sMemSize, SM90_MAX_SMEM);
    return false;
  }
  if (BN * WN > N) {
    fprintf(stderr, "Invalid config parameters BN*WN(%d) > N(%d), skipping launch", BN * WN, N);
    return false;
  }
  if (sMemSize <= SM90_MAX_SMEM && BN * WN <= N && N % (BN * WN) == 0) {
    int max_smem_real;
    cudaDeviceGetAttribute(&max_smem_real, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    assert(max_smem_real == SM90_MAX_SMEM);

    const bool overlapped = experts_done != nullptr;
    auto kernel_standard =
        fused_moe_w8a8_wgmma_up_down_acc_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS, DB_STORE,
                                                RED_STORE, false>;
    auto kernel_overlapped =
        fused_moe_w8a8_wgmma_up_down_acc_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS, DB_STORE,
                                                RED_STORE, true>;
    auto kernel = overlapped ? kernel_overlapped : kernel_standard;

    gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    constexpr int TILES_UP = BN * WN / 256;
    constexpr int TILES_DOWN = BN * WN / 256;
    // Create TMA descriptor for w (up projection)
    CUtensorMap tensor_map_w{};
    {
      constexpr uint32_t TILE_COLS_UP = BN * WN / TILES_UP;
      constexpr uint32_t rank = 3;
      uint64_t size[rank] = {static_cast<uint64_t>(K), static_cast<uint64_t>(TILE_COLS_UP),
                             static_cast<uint64_t>((N * num_experts) / TILE_COLS_UP)};
      uint64_t stride[rank - 1] = {static_cast<uint64_t>(K) * sizeof(fp8),
                                   static_cast<uint64_t>(TILE_COLS_UP) * K * sizeof(fp8)};
      uint32_t box_size[rank] = {BK, TILE_COLS_UP, static_cast<uint32_t>(TILES_UP)};
      uint32_t elem_stride[rank] = {1, 1, 1};
      auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
      CUresult res = cuTensorMapEncodeTiled(
          &tensor_map_w, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, w, size, stride,
          box_size, elem_stride, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    // Create TMA descriptor for w2 (down projection)
    constexpr int BK2 = WN * BN / 2;
    constexpr int BN2 = BK * 2;
    const int K2 = N / 2;
    const int N2 = K;
    if constexpr (BN2 / 64 < WN / 4) {
      fprintf(stderr, "Too much warpgroups(%d) for block N(%d)\n", WN, BN2);
      return false;
    }
    CUtensorMap tensor_map_w2{};
    {
      constexpr uint32_t TILE_INNER_DN = BK2 / TILES_DOWN;
      constexpr uint32_t rank = 3;
      // Reshape (K2, N2*E) into (TILE_INNER_DN, N2*E, K2/TILE_INNER_DN)
      // so that tile chunks end up in dim2 (outermost, varies slowest in smem fill)
      // producing contiguous per-tile regions in shared memory.
      uint64_t size[rank] = {static_cast<uint64_t>(TILE_INNER_DN),
                             static_cast<uint64_t>(N2 * num_experts),
                             static_cast<uint64_t>(K2 / TILE_INNER_DN)};
      uint64_t stride[rank - 1] = {static_cast<uint64_t>(K2) * sizeof(fp8),
                                   static_cast<uint64_t>(TILE_INNER_DN) * sizeof(fp8)};
      uint32_t box_size[rank] = {TILE_INNER_DN, BN2, static_cast<uint32_t>(TILES_DOWN)};
      uint32_t elem_stride[rank] = {1, 1, 1};
      auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
      CUresult res = cuTensorMapEncodeTiled(
          &tensor_map_w2, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, w2, size,
          stride, box_size, elem_stride, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          WN * BN >= 256   ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B
          : WN * BN == 128 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B
                           : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    kernel<<<dimGrid, dimBlock, sMemSize, stream>>>(
        x, x_scale, tensor_map_w, w_scale, tensor_map_w2, w2_scale, out, sorted_token_ids,
        expert_ids, num_tokens_post_padded, topk_weights, top_k, M, N, K, scaling_factor,
        experts_done);
    return true;
  }
  return false;
}

template <int BM, int BN, int WN, int STAGES, bool RED_STORE>
bool launch_fused_moe_kernel_up_down_acc(const fp8* x, const float* x_scale, fp8* w,
                                         const float* w_scale, fp8* w2, const float* w2_scale,
                                         __nv_bfloat16* out, const int* sorted_token_ids,
                                         const int* expert_ids, const int* num_tokens_post_padded,
                                         const float* topk_weights, const int top_k, int M, int N,
                                         int K, int num_experts, int sorted_num, int block_m,
                                         float scaling_factor, cudaStream_t stream,
                                         int* experts_done = nullptr) {
  if (K <= 0 || K % 256 || N <= 0 || N % 256 || N % (BN * WN)) return false;
  const smem_layout<STAGES, BM, 128, BN, WN, true> layout(K);
  auto launch = [&]<bool DB_STORE>() {
    return launch_fused_moe_kernel_up_down_acc_impl<BM, BN, WN, STAGES, DB_STORE, RED_STORE>(
        x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,
        num_tokens_post_padded, topk_weights, top_k, M, N, K, num_experts, sorted_num, block_m,
        scaling_factor, stream, experts_done);
  };
  if (layout.total <= SM90_MAX_SMEM) return launch.template operator()<true>();
  return launch.template operator()<false>();
}

}  // namespace w8a8_persistent
