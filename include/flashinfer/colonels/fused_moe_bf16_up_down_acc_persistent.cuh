#pragma once

#include <cooperative_groups.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include <cassert>
#include <cstdint>
#include <cuda/ptx>
#include <cuda/std/limits>

#include "fused_moe_activation.cuh"
#include "fused_moe_helpers.cuh"

using bf16 = __nv_bfloat16;
namespace cg = cooperative_groups;

namespace bf16_persistent {

template <int STAGES, int WN, int BM, int BK, int BN, int UP_FACTOR>
struct smem_up {
  // Matching stage strides keep DOWN prefetches from overwriting live UP operands.
  alignas(1024) bf16 w[STAGES * WN * BK * BN * 2 / UP_FACTOR];
  alignas(1024) bf16 x[STAGES * BK * BM];
};

template <int STAGES, int WN, int BM, int BK, int BN, bool DB_STORE, int UP_FACTOR>
struct smem_down {
  alignas(1024) bf16 w[STAGES * WN * BK * BN * 2 / UP_FACTOR];
  alignas(1024) bf16 x[BM * WN * BN / UP_FACTOR];
  alignas(16) bf16 out[BM * ((DB_STORE ? 2 : 1) * BK * 2 + 8)];
};

template <int STAGES, int BM, int BK, int BN, int WN>
struct smem_persistent {
  alignas(8) uint64_t bar[2 * STAGES + 1];
  float topk_scales[BM];
};

// Byte offset from the start of dynamic smem to the overlay region,
// Basically size of persistent smem rounded to next 1024 block
template <int STAGES, int BM, int BK, int BN, int WN>
static constexpr size_t smem_overlay_offset() {
  constexpr size_t raw = sizeof(smem_persistent<STAGES, BM, BK, BN, WN>);
  return (raw + 1023) & ~size_t(1023);
}

template <int BM, int BK, int BN, int WN, int STAGES, int PRODUCER_THREADS, bool DB_STORE,
          moe_activation::Kind ACTIVATION, bool OVERLAPPED = false>
__global__
__launch_bounds__(WN * 32 + PRODUCER_THREADS, 1) void fused_moe_bf16_wgmma_up_down_acc_kernel(
    const bf16* x, const __grid_constant__ CUtensorMap tensor_map_w,
    const __grid_constant__ CUtensorMap tensor_map_w2, bf16* __restrict__ out,
    const int* __restrict__ sorted_token_ids, const int* __restrict__ expert_ids,
    const int* __restrict__ num_tokens_post_padded, const float* __restrict__ topk_weights,
    const int top_k, int M, int N, int K, float scaling_factor,
    typename moe_activation::Traits<ACTIVATION>::Params activation_params,
    int* __restrict__ experts_done = nullptr) {
  // Hopper kernels share a fatbin with Blackwell kernels.
#if IS_SM90A
  using Activation = moe_activation::Traits<ACTIVATION>;
  constexpr int UP_FACTOR = Activation::GATED ? 2 : 1;
  constexpr int CONSUMER_THREADS = WN * 32;
  constexpr int WARPGROUPS = WN / 4;

  const int SPLIT_K = N / (WN * BN) == 0 ? 1 : N / (WN * BN);
  const int persistent_stride = gridDim.y / SPLIT_K;
  const int k_chunk = blockIdx.y % SPLIT_K;
  const int32_t block_start = blockIdx.y / SPLIT_K;
  const int total_tokens = num_tokens_post_padded[0];
  const int total_blocks = total_tokens / BM + (total_tokens % BM != 0);

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
  constexpr uint32_t S_BITS_UP = 3;
  constexpr uint32_t S_MODE_UP = 4 - S_BITS_UP;

  constexpr uint32_t S_BITS_DOWN = WN * BN >= 256 ? 3 : WN * BN == 128 ? 2 : 1;
  constexpr uint32_t S_MODE_DOWN = 4 - S_BITS_DOWN;

  const int K2 = N / UP_FACTOR;
  const int N2 = K;
  const int lane_id = threadIdx.x % 32;
  const bool is_producer = threadIdx.x >= CONSUMER_THREADS;
  const int warp_id = threadIdx.x / 32;
  const int wg_id = warp_id / 4;

  constexpr int BK2 = WN * BN / UP_FACTOR;
  constexpr int BN2 = BK * 2;

  // SMEM sizes
  constexpr int WS = WN * BK * BN;
  constexpr int WS_DOWN = WS * 2 / UP_FACTOR;
  constexpr int XS = BK * BM;
  // Number of tiles along dim-2 of the 3D TMA descriptor
  constexpr int TILES_UP = BN * WN / 256;
  constexpr int TILES_DOWN = BK2 / 64;
  // how many bytes we transfer per CP_ASYNC
  constexpr int TB = 16;
  // Thread offset per transfer
  constexpr int TO = TB / sizeof(bf16);
  // Tokens per thread
  constexpr int TPT = BM / 8;

  constexpr int TN2 = const_ceil(BN2 / float(64 * WARPGROUPS));

  extern __shared__ __align__(1024) uint8_t sh[];
  constexpr size_t OVERLAY_OFF = smem_overlay_offset<STAGES, BM, BK, BN, WN>();
  auto& smem_p = *reinterpret_cast<smem_persistent<STAGES, BM, BK, BN, WN>*>(sh);
  uint64_t* bar = smem_p.bar;
  uint64_t* epilogue_bar = smem_p.bar + 2 * STAGES;
  float* topk_scales = smem_p.topk_scales;

  smem_up<STAGES, WN, BM, BK, BN, UP_FACTOR>& s =
      *reinterpret_cast<smem_up<STAGES, WN, BM, BK, BN, UP_FACTOR>*>(sh + OVERLAY_OFF);
  smem_down<STAGES, WN, BM, BK, BN, DB_STORE, UP_FACTOR>& s_d =
      *reinterpret_cast<smem_down<STAGES, WN, BM, BK, BN, DB_STORE, UP_FACTOR>*>(sh + OVERLAY_OFF);

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

  const int n_stages_up = K / BK;
  const int n_stages_down = N2 / BN2;

  constexpr int TM = BM / 8;
  constexpr int TN = BN / 16;
  float f_acc[TN][TM][4];
  memset(f_acc, 0, sizeof(f_acc));

  // If we're running low on registers steal them from the producer
  // and not the consumer
  constexpr int PRODUCER_REGS = BM > 96 ? 24 : 56;
  constexpr int CONSUMER_REGS = BM > 96 ? 240 : 224;
  constexpr bool CHANGE_REG_ALLOC = BM > 88 || WN * BN >= 512;

  constexpr int TILE_COLS_UP = BN * WN / TILES_UP;
  constexpr int TILE_INNER_DN = BK2 / TILES_DOWN;

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
    uint32_t tsrc[X_IT];
    int* token_ids = reinterpret_cast<int*>(s.x);
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
      producer_sync();
      const int w_row_up = exp_idx * N + k_chunk * WN * BN;
      // Up projection X loading + Weights loading
      for (int load_stage = 0; load_stage < n_stages_up; load_stage++) {
        if (smem_stage == STAGES) {
          p ^= 1;
          smem_stage = 0;
        }
        const int off = load_stage * BK;
        int i = (producer_tid)*TO;
        int col = off + i % BK;
        int swizzled = swizzle<S_BITS_UP, 3, 3>(i);

        wait(bar + STAGES + smem_stage, p);

        for (int r = 0; r < X_IT; r += 1) {
          int row = tsrc[r];
          if (row < M && i < XS) {
            uint32_t sm = __cvta_generic_to_shared(s.x + smem_stage * XS + swizzled);
            CP_ASYNC_CG(sm, reinterpret_cast<const float4*>(x + row * K + col), TB);
          }
          i += PRODUCER_THREADS * TO;
          swizzled += PRODUCER_THREADS * TO;
        }

        cp_async_mbarrier_arrive(bar + smem_stage);

        if (producer_tid == 0) {
          cg::invoke_one(cg::coalesced_threads(), [&]() {
            expect_bytes(bar + smem_stage, WS * sizeof(bf16));
            load_async_3d(s.w + smem_stage * WS_DOWN, &tensor_map_w, bar + smem_stage, off, 0,
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
            expect_bytes(bar + smem_stage, WS_DOWN * sizeof(bf16));
            load_async_3d(s_d.w + smem_stage * WS_DOWN, &tensor_map_w2, bar + smem_stage, 0,
                          w_col_down, w_row_down / TILE_INNER_DN);
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

      // Part where x and out might overlap so we only load weights
      for (int load_stage = 0; load_stage < STAGES; load_stage++) {
        if (smem_stage == STAGES) {
          p ^= 1;
          smem_stage = 0;
        }
        const int off = load_stage * BK;

        wait(bar + STAGES + smem_stage, p);

        if (producer_tid == 0) {
          cg::invoke_one(cg::coalesced_threads(), [&]() {
            expect_bytes(bar + smem_stage, WS * sizeof(bf16));
            load_async_3d(s.w + smem_stage * WS_DOWN, &tensor_map_w, bar + smem_stage, off, 0,
                          w_row_up / TILE_COLS_UP);
          });
        }
        smem_stage++;
      }

      bool synced = false;
      // Now that all stages are done start loading X
      for (int load_stage = 0; load_stage < STAGES; load_stage++) {
        if (smem_stage == STAGES) {
          smem_stage = 0;
        }
        const int off = load_stage * BK;
        int i = (producer_tid)*TO;
        int col = off + i % BK;
        int swizzled = swizzle<S_BITS_UP, 3, 3>(i);
        // If x and out are overapped wait for out to be free
        if constexpr (sizeof(s_d.x) < sizeof(s.x)) {
          if (!synced && smem_stage * XS * sizeof(bf16) >= sizeof(s_d.x)) {
            global_sync();
            synced = true;
          }
        }
        for (int r = 0; r < X_IT; r += 1) {
          int row = tsrc[r];
          if (row < M && i < XS) {
            uint32_t sm = __cvta_generic_to_shared(s.x + smem_stage * XS + swizzled);
            CP_ASYNC_CG(sm, reinterpret_cast<const float4*>(x + row * K + col), TB);
          }
          i += PRODUCER_THREADS * TO;
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
        const int off = load_stage * BK;
        int i = (producer_tid)*TO;
        int col = off + i % BK;
        int swizzled = swizzle<S_BITS_UP, 3, 3>(i);

        wait(bar + STAGES + smem_stage, p);

        for (int r = 0; r < X_IT; r += 1) {
          int row = tsrc[r];
          if (row < M && i < XS) {
            uint32_t sm = __cvta_generic_to_shared(s.x + smem_stage * XS + swizzled);
            CP_ASYNC_CG(sm, reinterpret_cast<const float4*>(x + row * K + col), TB);
          }
          i += PRODUCER_THREADS * TO;
          swizzled += PRODUCER_THREADS * TO;
        }

        cp_async_mbarrier_arrive(bar + smem_stage);

        if (producer_tid == 0) {
          cg::invoke_one(cg::coalesced_threads(), [&]() {
            expect_bytes(bar + smem_stage, WS * sizeof(bf16));
            load_async_3d(s.w + smem_stage * WS_DOWN, &tensor_map_w, bar + smem_stage, off, 0,
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
            expect_bytes(bar + smem_stage, WS_DOWN * sizeof(bf16));
            load_async_3d(s_d.w + smem_stage * WS_DOWN, &tensor_map_w2, bar + smem_stage, 0,
                          w_col_down, w_row_down / TILE_INNER_DN);
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
      memset(f_acc, 0, sizeof(f_acc));
      for (int compute_stage = 0; compute_stage < n_stages_up; compute_stage += 1) {
        if (smem_stage == STAGES) {
          p ^= 1;
          smem_stage = 0;
        }
        wait(bar + smem_stage, p);

        // https://github.com/NVIDIA/cutlass/discussions/1375
        // This is cursed
        for (int tn = 0; tn < TN; tn++) {
          for (int tm = 0; tm < TM; tm++) {
            for (int t = 0; t < 4; t++) asm volatile("" : "+f"(f_acc[tn][tm][t])::"memory");
          }
        }
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
        warpgroup_arrive();
        for (int tn = 0; tn < TN; tn++) {
          bf16* sw = s.w + smem_stage * WS_DOWN + (warp_id / 4) * (BN * 4) * BK + tn * 64 * BK;
          bf16* sx = s.x + smem_stage * XS;
          uint64_t desc_w = make_smem_descriptor<64, 1, S_MODE_UP>(sw);
          uint64_t desc_x = make_smem_descriptor<64, 1, S_MODE_UP>(sx);
          wgmma_bf16<1, 1, 1, 0, 0, BM>(f_acc[tn], desc_w, desc_x);
          wgmma_bf16<1, 1, 1, 0, 0, BM>(f_acc[tn], desc_w + 1 * (32 >> 4), desc_x + 1 * (32 >> 4));
          wgmma_bf16<1, 1, 1, 0, 0, BM>(f_acc[tn], desc_w + 2 * (32 >> 4), desc_x + 2 * (32 >> 4));
          wgmma_bf16<1, 1, 1, 0, 0, BM>(f_acc[tn], desc_w + 3 * (32 >> 4), desc_x + 3 * (32 >> 4));
        }
        warpgroup_commit_batch();
        warpgroup_wait();
        arrive(bar + STAGES + smem_stage);

        smem_stage++;
      }

      const int exp_idx = expert_ids[local_block];
      int token_src = M;
      int token_raw = M * top_k;
      if (lane_id < SAVES_PER_WARP) {
        token_raw = sorted_token_ids[local_block * BM + wg_id * (BM / 2) +
                                     (warp_id % 4) * SAVES_PER_WARP + lane_id];
        token_src = token_raw / top_k;
      }

      for (int tn = 0; tn < TN; tn++) {
        for (int tm = 0; tm < TM; tm++) {
          if constexpr (Activation::GATED) {
            f_acc[tn][tm][0] = moe_activation::apply<Activation>(f_acc[tn][tm][0], f_acc[tn][tm][2],
                                                                 activation_params);
            f_acc[tn][tm][1] = moe_activation::apply<Activation>(f_acc[tn][tm][1], f_acc[tn][tm][3],
                                                                 activation_params);
          } else {
            for (int t = 0; t < 4; t++)
              f_acc[tn][tm][t] =
                  moe_activation::apply<Activation>(f_acc[tn][tm][t], activation_params);
          }
        }
      }

      if constexpr (WN > 4) {
        consumer_sync();
      }

      constexpr int BLOCKS_PER_WG = const_ceil(float(TN * (64 / UP_FACTOR)) / BK);
      constexpr int BLOCKS_TOTAL = BK2 / BK;

      for (int tm = 0; tm < TM; tm++) {
        for (int t = 0; t < 4 / UP_FACTOR; t++) {
          for (int tn = 0; tn < TN; tn++) {
            float val = f_acc[tn][tm][t];
            int x_row = tm * 8 + (lane_id % 4) * 2 + t % 2;
            int x_col_local = tn * (64 / UP_FACTOR) + (warp_id % 4) * (16 / UP_FACTOR) +
                              lane_id / 4 + (t / 2) * 8;
            int block_in_wg = x_col_local / BK;
            int col_in_block = x_col_local % BK;
            int wg_offset = BLOCKS_PER_WG != BLOCKS_TOTAL ? wg_id * BLOCKS_PER_WG * BM * BK : 0;
            int i = block_in_wg * BM * BK + x_row * BK + col_in_block;
            int swizzled = swizzle<S_BITS_DOWN, 3, 3>(wg_offset + i);
            s_d.x[swizzled] = bf16(val);
          }
        }
      }

      float token_scale[TPT];
      for (int t = 0; t < BM / 2 / 4; t += 2) {
        int token_idx = wg_id * (BM / 2) + (t / 2) * 8 + (lane_id % 4) * 2;
        const float2 topk_w = *reinterpret_cast<const float2*>(&topk_scales[token_idx]);
        token_scale[t] = topk_w.x * scaling_factor;
        token_scale[t + 1] = topk_w.y * scaling_factor;
      }
      consumer_sync();

      constexpr int TND = BN2 / 64;
      constexpr int MW = BM / 2;
      float tile_acc[TND][MW / 8][4];
      __nv_bfloat16 out_tile[TND][MW / 8][4];
      constexpr int SAVE_ELEMS = (DB_STORE ? BN2 * 2 : BN2);
      constexpr int PAD = SAVE_ELEMS + 8;
      constexpr int BLOCKS_PER_STAGE = BK2 / BK;

      for (int compute_stage = 0; compute_stage < n_stages_down; compute_stage += 1) {
        if (smem_stage == STAGES) {
          p ^= 1;
          smem_stage = 0;
        }

        memset(tile_acc, 0, sizeof(tile_acc));
        wait(bar + smem_stage, p);

        warpgroup_arrive();
#pragma unroll
        for (int block = 0; block < BLOCKS_PER_STAGE; block++) {
          // When BLOCKS_TOTAL > 1, BLOCKS are stored in separate
          // contiguous BM*BK regions (each with its own swizzle).
          // Each warp group reads its own MW-token slice.
          bf16* sx = s_d.x + block * BM * BK + wg_id * MW * BK;
          // I have no idea why but keeping this inside the loop somehow
          // makes it faster, initial assumption is that it brings control back
          // to the other warpgroup TODO investigate
          cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
          for (int tni = 0; tni < TND; tni++) {
            bf16* sw = s_d.w + smem_stage * WS_DOWN + block * WS_DOWN / TILES_DOWN +
                       tni * 64 * (BK2 / (TILES_DOWN));
            uint64_t desc_w = make_smem_descriptor<BK, 1, S_MODE_DOWN>(sw);
            uint64_t desc_x = make_smem_descriptor<BK, 1, S_MODE_DOWN>(sx);
            wgmma_bf16<1, 1, 1, 0, 0, MW>(tile_acc[tni], desc_w, desc_x);
            desc_w += (32 >> 4);
            desc_x += (32 >> 4);
            for (int tk = 1; tk < 4; tk++) {
              wgmma_bf16<1, 1, 1, 0, 0, MW>(tile_acc[tni], desc_w, desc_x);
              desc_w += (32 >> 4);
              desc_x += (32 >> 4);
            }
          }
        }
        warpgroup_commit_batch();
        warpgroup_wait();
        arrive(bar + STAGES + smem_stage);
        smem_stage++;

        for (int tni = 0; tni < TND; tni++) {
          for (int tm = 0; tm < MW / 8; tm++) {
            for (int t = 0; t < 4; t++) {
              float scale = token_scale[tm * 2 + t % 2];
              out_tile[tni][tm][t] = scale * tile_acc[tni][tm][t];
            }
          }
        }
        constexpr int ST_GRANULARITY = DB_STORE ? 2 : 1;
        if (compute_stage > 0 && compute_stage % ST_GRANULARITY == 0) {
          asm volatile("cp.async.bulk.wait_group 0;");
          warpgroup_sync();
        }
        int offset = DB_STORE ? (compute_stage % 2) * BN2 : 0;

        if constexpr (MW % 16 == 0) {
          for (int tn2 = 0; tn2 < TND; tn2++) {
            for (int tm = 0; tm < MW / 8; tm += 2) {
              int out_row = wg_id * (BM / 2) + tm * 8 + lane_id % 8 + (lane_id / 16) * 8;
              int out_col = (warp_id % 4) * 16 + (lane_id & 8) + tn2 * 64 + offset;
              st_matrix_x4_trans(reinterpret_cast<uint32_t*>(out_tile[tn2][tm]),
                                 __cvta_generic_to_shared(s_d.out + out_row * PAD + out_col));
            }
          }
        } else {
          for (int tn2 = 0; tn2 < TND; tn2++) {
            for (int tm = 0; tm < MW / 8; tm++) {
              int out_row = wg_id * (BM / 2) + tm * 8 + lane_id % 8;
              int out_col = (warp_id % 4) * 16 + (lane_id & 8) + tn2 * 64 + offset;
              st_matrix_x2_trans(reinterpret_cast<uint32_t*>(out_tile[tn2][tm]),
                                 __cvta_generic_to_shared(s_d.out + out_row * PAD + out_col));
            }
          }
        }

        if ((DB_STORE && compute_stage % 2 == 1) || !DB_STORE ||
            compute_stage == n_stages_down - 1) {
          if (compute_stage != n_stages_down - 1) wait(epilogue_bar, e_p);
          cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
          warpgroup_sync();

          int col = DB_STORE ? compute_stage - compute_stage % 2 : compute_stage;
          const int store_elems = DB_STORE && compute_stage % 2 == 1 ? SAVE_ELEMS : BN2;
          if (lane_id < SAVES_PER_WARP) {
            if (token_src < M) {
              int row = (warp_id)*SAVES_PER_WARP + lane_id;
              cuda::ptx::cp_reduce_async_bulk(cuda::ptx::space_global, cuda::ptx::space_shared,
                                              cuda::ptx::op_add, out + token_src * N2 + col * BN2,
                                              s_d.out + row * PAD,
                                              store_elems * sizeof(__nv_bfloat16));
            }
            cuda::ptx::cp_async_bulk_commit_group();
          }

          // The last store neither waits for its turn nor passes it on: an arrive
          // without a wait lets this warpgroup complete two phases before the other
          // one checks its parity, and that one then waits forever.
          if (compute_stage != n_stages_down - 1) arrive(epilogue_bar);
        }
      }
      if constexpr (sizeof(s_d.x) < sizeof(s.x)) {
        if (local_block + persistent_stride < total_blocks) {
          asm volatile("cp.async.bulk.wait_group 0;");
          global_sync();
        }
      }
      if constexpr (OVERLAPPED) {
        if (sizeof(s_d.x) >= sizeof(s.x) || local_block + persistent_stride >= total_blocks) {
          asm volatile("cp.async.bulk.wait_group 0;");
          consumer_sync();
        }
        if (token_raw < M * top_k) {
          cuda::atomic_ref<int, cuda::thread_scope_device> x(
              *(experts_done + token_raw * SPLIT_K + k_chunk));
          x.store(1, cuda::memory_order_release);
        }
      }
    }
  }
#else
  assert(false && "This fused MoE kernel requires SM90");
#endif
}

template <int BM, int BN, int WN, int STAGES, moe_activation::Kind ACTIVATION>
bool launch_fused_moe_kernel_up_down_acc(
    const bf16* x, bf16* w, bf16* w2, bf16* out, const int* sorted_token_ids, const int* expert_ids,
    const int* num_tokens_post_padded, const float* topk_weights, const int top_k, int M, int N,
    int K, int num_experts, int sorted_num, int block_m, float scaling_factor,
    typename moe_activation::Traits<ACTIVATION>::Params activation_params, cudaStream_t stream,
    int* experts_done = nullptr) {
  constexpr int UP_FACTOR = moe_activation::Traits<ACTIVATION>::GATED ? 2 : 1;
  constexpr int BK = 64;
  constexpr int PRODUCER_THREADS = 128;
  if (K <= 0 || K % (2 * BK) || N <= 0 || N % 256 || N % (BN * WN)) return false;
  // Persistent iterations prefetch a full stage ring before the consumer releases it.
  if (STAGES > K / BK) return false;
  int device, num_sms;
  gpuErrchk(cudaGetDevice(&device));
  gpuErrchk(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  dim3 dimBlock(32 * WN + PRODUCER_THREADS, 1, 1);
  const int SPLIT_K = N / (BN * WN) == 0 ? 1 : N / (BN * WN);
  const int block_count = std::min(num_sms / SPLIT_K, (int)std::ceil((float)sorted_num / block_m));
  if (block_count <= 0) return false;
  dim3 dimGrid(1, block_count * SPLIT_K, 1);

  constexpr size_t OVERLAY_OFF = smem_overlay_offset<STAGES, BM, BK, BN, WN>();
  constexpr bool DB_STORE =
      OVERLAY_OFF + sizeof(smem_down<STAGES, WN, BM, BK, BN, true, UP_FACTOR>) <= SM90_MAX_SMEM;
  constexpr size_t sMemSize =
      OVERLAY_OFF + std::max(sizeof(smem_up<STAGES, WN, BM, BK, BN, UP_FACTOR>),
                             sizeof(smem_down<STAGES, WN, BM, BK, BN, DB_STORE, UP_FACTOR>));

  if constexpr (sMemSize > SM90_MAX_SMEM) {
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
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&max_smem_real, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    assert(max_smem_real == SM90_MAX_SMEM);

    const bool overlapped = experts_done != nullptr;
    auto kernel_standard =
        fused_moe_bf16_wgmma_up_down_acc_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS, DB_STORE,
                                                ACTIVATION, false>;
    auto kernel_overlapped =
        fused_moe_bf16_wgmma_up_down_acc_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS, DB_STORE,
                                                ACTIVATION, true>;
    auto kernel = overlapped ? kernel_overlapped : kernel_standard;

    gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    constexpr int TILES_UP = BN * WN / 256;
    CUtensorMap tensor_map_w{};
    {
      constexpr uint32_t TILE_COLS_UP = BN * WN / TILES_UP;
      constexpr uint32_t rank = 3;
      uint64_t size[rank] = {static_cast<uint64_t>(K), static_cast<uint64_t>(TILE_COLS_UP),
                             static_cast<uint64_t>((N * num_experts) / TILE_COLS_UP)};
      uint64_t stride[rank - 1] = {static_cast<uint64_t>(K) * sizeof(bf16),
                                   static_cast<uint64_t>(TILE_COLS_UP) * K * sizeof(bf16)};
      uint32_t box_size[rank] = {BK, TILE_COLS_UP, static_cast<uint32_t>(TILES_UP)};
      uint32_t elem_stride[rank] = {1, 1, 1};
      auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
      CUresult res = cuTensorMapEncodeTiled(
          &tensor_map_w, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, w, size,
          stride, box_size, elem_stride, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    constexpr int BK2 = WN * BN / UP_FACTOR;
    constexpr int BN2 = BK * 2;
    constexpr int TILES_DOWN = BK2 / 64;
    const int K2 = N / UP_FACTOR;
    const int N2 = K;
    if constexpr (BN2 / 64 < WN / 4) {
      fprintf(stderr, "Too much warpgroups(%d) for block N(%d)\n", WN, BN2);
      return false;
    }
    CUtensorMap tensor_map_w2{};
    {
      constexpr uint32_t TILE_INNER_DN = BK2 / TILES_DOWN;
      constexpr uint32_t rank = 3;
      uint64_t size[rank] = {static_cast<uint64_t>(TILE_INNER_DN),
                             static_cast<uint64_t>(N2 * num_experts),
                             static_cast<uint64_t>(K2 / TILE_INNER_DN)};
      uint64_t stride[rank - 1] = {static_cast<uint64_t>(K2) * sizeof(bf16),
                                   static_cast<uint64_t>(TILE_INNER_DN) * sizeof(bf16)};
      uint32_t box_size[rank] = {TILE_INNER_DN, BN2, static_cast<uint32_t>(TILES_DOWN)};
      uint32_t elem_stride[rank] = {1, 1, 1};
      auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
      CUresult res = cuTensorMapEncodeTiled(
          &tensor_map_w2, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, w2, size,
          stride, box_size, elem_stride, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          WN * BN >= 256   ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B
          : WN * BN == 128 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B
                           : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    kernel<<<dimGrid, dimBlock, sMemSize, stream>>>(
        x, tensor_map_w, tensor_map_w2, out, sorted_token_ids, expert_ids, num_tokens_post_padded,
        topk_weights, top_k, M, N, K, scaling_factor, activation_params, experts_done);
    return true;
  }
  return false;
}

}  // namespace bf16_persistent
