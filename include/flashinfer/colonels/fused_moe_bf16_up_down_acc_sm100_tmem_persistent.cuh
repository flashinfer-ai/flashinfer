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
#include <cstdint>
#include <cuda/atomic>
#include <cuda/ptx>
#include <cuda/std/limits>

#include "fused_moe_activation.cuh"
#include "fused_moe_helpers.cuh"

using bf16 = __nv_bfloat16;
namespace cg = cooperative_groups;

namespace moe_sm100_tmem_persistent {

template <int BLOCK_M, int BLOCK_N, int PIPELINE_STAGES, moe_activation::Kind ACTIVATION>
struct constants {
  using Activation = moe_activation::Traits<ACTIVATION>;
  using ActivationParams = typename Activation::Params;
  static constexpr bool IS_GATED = Activation::GATED;
  static constexpr int UD_RATIO = IS_GATED ? 2 : 1;
  static constexpr int BM = BLOCK_M;
  static constexpr int STAGES = PIPELINE_STAGES;
  static constexpr int BK = 64;
  // Blackwell uses four MMA warps per CTA; additional warpgroups use clusters.
  static constexpr int WN = 4;
  static constexpr int BLN = BLOCK_N * WN;
  static constexpr int CLUSTERS = WN / 4;
  static constexpr int MMA_THREADS = WN * 32;
  static constexpr int PRODUCER_THREADS = 128;
  static constexpr int EPILOGUE_THREADS = 128;
  static constexpr int BLOCK_THREADS = MMA_THREADS + PRODUCER_THREADS + EPILOGUE_THREADS;

  // SwiGLU compacts the UP accumulator in place; DOWN uses the upper half.
  static constexpr int TMEM_COLS = BLN;
  // Blackwell's opt-in dynamic shared memory limit per block.
  static constexpr size_t SM100_MAX_SMEM = 232448;
  static constexpr uint32_t S_BITS_UP = 3;
  static constexpr uint32_t S_MODE_UP = 4 - S_BITS_UP;
  static constexpr uint32_t S_BITS_DOWN = 3;
  static constexpr uint32_t S_MODE_DOWN = 4 - S_BITS_DOWN;
  static constexpr int WS = BK * BLN;
  static constexpr int XS = BK * BM;
  static constexpr int BK2 = BLN / UD_RATIO;
  static constexpr int BN2 = BK * 2;
  static constexpr int WS_DOWN = BK2 * BN2;
  static constexpr int PAD = BN2 + 8;
  static constexpr int MMA_N_UP = 256;
  static constexpr int TILES_UP = BLN / MMA_N_UP;
  static constexpr int TILES_DOWN = BK2 / 64;
  static constexpr int TILE_COLS_UP = BLN / TILES_UP;
  static constexpr int TILE_INNER_DN = BK2 / TILES_DOWN;
  static constexpr int TPT = BM / 32;
  static constexpr uint32_t idesc_up = build_idesc<BM, MMA_N_UP>();
  static constexpr uint32_t idesc_down = build_idesc<BM, BN2>();
  static constexpr int LOADS_PER_WARP = BM / ((PRODUCER_THREADS / 32) * 4);
  static constexpr int SAVES_PER_WARP = BM / WN;
  static constexpr uint32_t TMA_RANK_W = 3;
  static constexpr uint32_t TMA_RANK_X = 2;

  struct smem_up {
    alignas(1024) bf16 w[STAGES * WS];
    alignas(1024) bf16 x[STAGES * XS];
  };

  template <bool DOUBLE_BUFFER>
  struct smem_down_storage {
    alignas(1024) bf16 w[STAGES * WS_DOWN];
    alignas(16) bf16 out[(DOUBLE_BUFFER ? 2 : 1) * BM * PAD];
  };

  struct smem_persistent {
    alignas(8) uint64_t bar[2 * STAGES + 3];
    uint32_t tmem_addr;
  };

  static constexpr size_t smem_persistent_bytes = sizeof(smem_persistent);
  static constexpr size_t smem_persistent_bar_bytes = sizeof(decltype(smem_persistent::bar));
  static constexpr size_t smem_persistent_tmem_addr_bytes =
      sizeof(decltype(smem_persistent::tmem_addr));
  // Round persistent storage up to the overlay's 1024-byte alignment.
  static constexpr size_t OVERLAY_OFF = (smem_persistent_bytes + 1023) & ~size_t(1023);
  static constexpr bool DB_STORE = OVERLAY_OFF + sizeof(smem_down_storage<true>) <= SM100_MAX_SMEM;
  using smem_down = smem_down_storage<DB_STORE>;

  static constexpr size_t smem_up_bytes = sizeof(smem_up);
  static constexpr size_t smem_up_w_bytes = sizeof(decltype(smem_up::w));
  static constexpr size_t smem_up_x_bytes = sizeof(decltype(smem_up::x));
  static constexpr size_t smem_down_bytes = sizeof(smem_down);
  static constexpr size_t smem_down_w_bytes = sizeof(decltype(smem_down::w));
  static constexpr size_t smem_down_out_bytes = sizeof(decltype(smem_down::out));
  static constexpr size_t sMemSize = OVERLAY_OFF + std::max(smem_up_bytes, smem_down_bytes);

  static constexpr int MAX_STAGES = [] {
    // Each pipeline stage adds two barriers to the persistent storage.
    constexpr size_t metadata = smem_persistent_bytes - 2 * STAGES * sizeof(uint64_t);
    constexpr size_t output_bytes = sizeof(smem_down_storage<false>) - smem_down_w_bytes;
    for (int stages = 1;; ++stages) {
      const size_t overlay = (metadata + 2 * stages * sizeof(uint64_t) + 1023) & ~size_t(1023);
      const size_t up = stages * (WS + XS) * sizeof(bf16);
      // The launcher uses a single output buffer when double buffering does not fit.
      const size_t down = stages * WS_DOWN * sizeof(bf16) + output_bytes;
      if (overlay + std::max(up, down) > SM100_MAX_SMEM) return stages - 1;
    }
  }();
};

template <typename C>
__global__ __cluster_dims__(C::CLUSTERS, 1, 1)
    __launch_bounds__(C::BLOCK_THREADS, 1) void fused_moe_bf16_wgmma_up_down_acc_kernel(
        const __grid_constant__ CUtensorMap tensor_map_x,
        const __grid_constant__ CUtensorMap tensor_map_w,
        const __grid_constant__ CUtensorMap tensor_map_w2, bf16* __restrict__ out,
        const int* __restrict__ sorted_token_ids, const int* __restrict__ expert_ids,
        const int* __restrict__ num_tokens_post_padded, const float* __restrict__ topk_weights,
        const int top_k, int M, int N, int K, float scaling_factor,
        typename C::ActivationParams activation_params) {
  // Keep Blackwell instructions out of other architectures in a shared fatbin.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 1000
  const int N2 = K;
  const int n_stages_up = K / C::BK;
  const int n_stages_down = N2 / C::BN2;
  const int SPLIT_K = (N + C::BLN - 1) / C::BLN;
  const int block_start = blockIdx.x / SPLIT_K;
  const int k_chunk = blockIdx.x % SPLIT_K;
  const int max_workers = gridDim.x / SPLIT_K;
  const int total_blocks = (num_tokens_post_padded[0] + C::BM - 1) / C::BM;
  if (total_blocks == 0) return;
  // Spread the final wave across workers, as in the persistent SMEM kernel.
  const int waves = (total_blocks + max_workers - 1) / max_workers;
  const int persistent_stride = (total_blocks + waves - 1) / waves;
  if (block_start >= persistent_stride) return;
  int cta_rank = 0;
  if constexpr (C::CLUSTERS > 1) {
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
  }

  const int lane_id = threadIdx.x % 32;
  const bool is_epilogue = threadIdx.x >= C::MMA_THREADS + C::PRODUCER_THREADS;
  const bool is_producer =
      threadIdx.x >= C::MMA_THREADS && threadIdx.x < C::MMA_THREADS + C::PRODUCER_THREADS;
  const bool is_mma = threadIdx.x < C::MMA_THREADS;
  const int warp_id = threadIdx.x / 32;

  extern __shared__ __align__(1024) uint8_t sh[];
  auto& smem_p = *reinterpret_cast<typename C::smem_persistent*>(sh);
  uint64_t* bar = smem_p.bar;
  uint64_t* e_to_c_bar = bar + 2 * C::STAGES;
  uint64_t* c_to_e_bar = e_to_c_bar + 1;
  uint64_t* e_to_p_bar = c_to_e_bar + 1;

  typename C::smem_up& s = *reinterpret_cast<typename C::smem_up*>(sh + C::OVERLAY_OFF);
  typename C::smem_down& s_d = *reinterpret_cast<typename C::smem_down*>(sh + C::OVERLAY_OFF);

  if (threadIdx.x == 0) {
    for (int i = 0; i < C::STAGES; i++) {
      init_barrier(&bar[i], C::BM / 4 + 1, 0);
      init_barrier(&bar[i + C::STAGES], 1, 0);
    }
    init_barrier(e_to_c_bar, C::EPILOGUE_THREADS, 0);
    init_barrier(c_to_e_bar, 1, 0);
    init_barrier(e_to_p_bar, C::EPILOGUE_THREADS, 0);
  } else if (warp_id == 1) {
    tmem_alloc(__cvta_generic_to_shared(&smem_p.tmem_addr), C::TMEM_COLS);
  }

  __syncthreads();
  uint32_t tmem_addr = smem_p.tmem_addr;
  auto epilogue_sync = [&]() { asm volatile("bar.sync 1, %0;\n" ::"n"(C::EPILOGUE_THREADS)); };

  int smem_stage = 0;
  int p = 0;
  int e_p = 0;
  int epilogue_p = 0;
  int producer_tile_p = 0;
  if (threadIdx.x == 0) {
    for (int i = 0; i < C::STAGES; i++) arrive(&bar[C::STAGES + i]);
  }
  for (int local_block = block_start; local_block < total_blocks;
       local_block += persistent_stride) {
    const int exp_idx = expert_ids[local_block];
    const int* local_token_ids =
        sorted_token_ids + local_block * C::CLUSTERS * C::BM + cta_rank * C::BM;
    // PRODUCER
    if (is_producer) {
      int producer_tid = threadIdx.x - C::MMA_THREADS;
      int producer_wid = producer_tid / 32;
      int4 tokens;
      if (local_block != block_start) {
        // UP's X buffer overlays DOWN's output staging; wait for all bulk stores.
        wait(e_to_p_bar, producer_tile_p);
        producer_tile_p ^= 1;
      }

      if (producer_tid % 32 < C::LOADS_PER_WARP) {
        tokens = reinterpret_cast<const int4*>(
            local_token_ids)[(producer_tid / 32) * C::LOADS_PER_WARP + producer_tid % 32];
        tokens.x /= top_k;
        tokens.y /= top_k;
        tokens.z /= top_k;
        tokens.w /= top_k;
      }
      for (int load_stage = 0; load_stage < n_stages_up; load_stage++) {
        if (smem_stage == C::STAGES) {
          p ^= 1;
          smem_stage = 0;
        }
        const int off = load_stage * C::BK;

        wait(bar + C::STAGES + smem_stage, p);
        if (producer_wid == 0 && elect_sync()) {
          expect_bytes(bar + smem_stage, C::WS * sizeof(bf16));
          for (int w_tile = 0; w_tile < C::TILES_UP; w_tile++) {
            load_async_3d(s.w + smem_stage * C::WS + w_tile * (C::WS / C::TILES_UP), &tensor_map_w,
                          bar + smem_stage, off, k_chunk * C::BLN + w_tile * C::TILE_COLS_UP,
                          exp_idx);
          }
        }

        if (producer_tid % 32 < C::LOADS_PER_WARP) {
          expect_bytes(bar + smem_stage, 4 * C::BK * sizeof(bf16));
          int idx = (producer_tid / 32) * C::LOADS_PER_WARP + producer_tid % 32;
          load_async_2d_gather4(s.x + smem_stage * C::XS + idx * 4 * C::BK, &tensor_map_x,
                                bar + smem_stage, off, tokens);
        }

        smem_stage++;
      }

      // Ungated DOWN stages are larger and can overlap multiple UP slots.
      if constexpr (!C::IS_GATED) {
        for (int i = 0; i < C::STAGES; i++) {
          wait(bar + C::STAGES + i, p ^ int(i < smem_stage));
        }
      }
      for (int load_stage = 0; load_stage < n_stages_down; load_stage++) {
        if (smem_stage == C::STAGES) {
          p ^= 1;
          smem_stage = 0;
        }
        if (producer_wid == 0 && elect_sync()) {
          int col = load_stage;
          wait(bar + C::STAGES + smem_stage, p);
          const int w_row_down = k_chunk * C::BK2;
          const int w_col_down = exp_idx * N2 + col * C::BN2;

          expect_bytes(bar + smem_stage, C::WS_DOWN * sizeof(bf16));
          load_async_3d(s_d.w + smem_stage * C::WS_DOWN, &tensor_map_w2, bar + smem_stage, 0,
                        w_col_down, w_row_down / C::TILE_INNER_DN);
          arrive(bar + smem_stage, C::BM / 4);
        }
        smem_stage++;
      }
      if constexpr (!C::IS_GATED) {
        for (int i = 0; i < C::STAGES; i++) {
          wait(bar + C::STAGES + i, p ^ int(i < smem_stage));
        }
      }
    }
    // MMA
    else if (is_mma) {
      if (warp_id == 0 && elect_sync()) {
        for (int compute_stage = 0; compute_stage < n_stages_up; compute_stage += 1) {
          if (smem_stage == C::STAGES) {
            p ^= 1;
            smem_stage = 0;
          }
          wait(bar + smem_stage, p);
          // Some sources say you need this sync but
          // https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2
          // explains why you actually dont
          // asm volatile("tcgen05.fence::after_thread_sync;");
          for (int w_tile = 0; w_tile < C::TILES_UP; w_tile++) {
            uint64_t desc_w = make_smem_descriptor<64, 1, C::S_MODE_UP>(
                s.w + smem_stage * C::WS + w_tile * (C::WS / C::TILES_UP));
            uint64_t desc_x = make_smem_descriptor<64, 1, C::S_MODE_UP>(s.x + smem_stage * C::XS);
            for (int k = 0; k < C::BK; k += 16) {
              tcgen05_mma_f16<C::CLUSTERS>(tmem_addr + w_tile * C::MMA_N_UP, desc_x, desc_w,
                                           C::idesc_up, 1 - int(compute_stage == 0 && k == 0));
              desc_x += (32 >> 4);
              desc_w += (32 >> 4);
            }
          }
          tcgen05_commit<C::CLUSTERS>(__cvta_generic_to_shared(bar + C::STAGES + smem_stage));
          smem_stage++;
        }
        tcgen05_commit<C::CLUSTERS>(__cvta_generic_to_shared(c_to_e_bar));
        uint32_t x_tmem = tmem_addr;
        uint32_t out_tmem = tmem_addr + C::BLN / 2;

        for (int compute_stage = 0; compute_stage < n_stages_down; compute_stage += 1) {
          if (smem_stage == C::STAGES) {
            p ^= 1;
            smem_stage = 0;
          }
          wait(bar + smem_stage, p);
          wait(e_to_c_bar, e_p);
          e_p ^= 1;
          for (int w_tile = 0; w_tile < C::TILES_DOWN; w_tile++) {
            uint64_t desc_w = make_smem_descriptor<C::BK, 1, C::S_MODE_DOWN>(
                s_d.w + smem_stage * C::WS_DOWN + w_tile * C::WS_DOWN / C::TILES_DOWN);
            x_tmem = tmem_addr;
            for (int k = 0; k < C::BK; k += 16) {
              tcgen05_mma_f16_tmem<C::CLUSTERS>(out_tmem, x_tmem + w_tile * 32, desc_w,
                                                C::idesc_down, 1 - int(w_tile == 0 && k == 0));
              desc_w += (32 >> 4);
              x_tmem += 8;
            }
          }
          tcgen05_commit<C::CLUSTERS>(__cvta_generic_to_shared(bar + C::STAGES + smem_stage));
          tcgen05_commit<C::CLUSTERS>(__cvta_generic_to_shared(c_to_e_bar));
          smem_stage++;
        }
        // Consume the final release before UP reuses both halves of TMEM.
        wait(e_to_c_bar, e_p);
        e_p ^= 1;
      }
    } else if (is_epilogue) {
      int epilogue_tid = threadIdx.x - C::PRODUCER_THREADS - C::MMA_THREADS;
      int epilogue_wid = epilogue_tid / 32;

      int token_src = M;
      if (lane_id < C::SAVES_PER_WARP)
        token_src = local_token_ids[epilogue_wid * C::SAVES_PER_WARP + lane_id] / top_k;

      wait(c_to_e_bar, epilogue_p);
      epilogue_p ^= 1;
      asm volatile("tcgen05.fence::after_thread_sync;");

      uint32_t x_tmem = tmem_addr;
      uint32_t out_tmem = tmem_addr + C::BLN / 2;
      float acc[8];
      for (int row_off = 0; row_off < C::BM / 4; row_off += 16) {
        for (int tn = 0; tn < C::BLN / 16; tn++) {
          const int row = epilogue_wid * 32 + row_off;
          const int col = tn * 16;
          const uint32_t addr = tmem_addr + (row << 16) + col;
          asm volatile(
              "tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0, %1, %2, %3, %4, %5, "
              "%6, "
              "%7}, [%8];"
              : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]), "=f"(acc[4]), "=f"(acc[5]),
                "=f"(acc[6]), "=f"(acc[7])
              : "r"(addr));
          asm volatile("tcgen05.wait::ld.sync.aligned;");

          if constexpr (C::IS_GATED) {
            nv_bfloat162 swiglu[2];
            swiglu[0] = make_bfloat162(
                moe_activation::apply<typename C::Activation>(acc[0], acc[4], activation_params),
                moe_activation::apply<typename C::Activation>(acc[1], acc[5], activation_params));
            swiglu[1] = make_bfloat162(
                moe_activation::apply<typename C::Activation>(acc[2], acc[6], activation_params),
                moe_activation::apply<typename C::Activation>(acc[3], acc[7], activation_params));
            const int store_col = tn * 4;
            const uint32_t store_addr = x_tmem + (row << 16) + store_col;
            asm volatile("tcgen05.st.sync.aligned.16x128b.x1.b32 [%2], {%0, %1};" ::"r"(
                             reinterpret_cast<uint32_t*>(swiglu)[0]),
                         "r"(reinterpret_cast<uint32_t*>(swiglu)[1]), "r"(store_addr));
          } else {
            nv_bfloat162 activated[4];
            for (int i = 0; i < 4; i++) {
              activated[i] = make_bfloat162(
                  moe_activation::apply<typename C::Activation>(acc[2 * i], activation_params),
                  moe_activation::apply<typename C::Activation>(acc[2 * i + 1], activation_params));
            }
            const uint32_t store_addr = x_tmem + (row << 16) + tn * 8;
            asm volatile(
                "tcgen05.st.sync.aligned.16x128b.x2.b32 [%4], {%0, %1, %2, "
                "%3};" ::"r"(reinterpret_cast<uint32_t*>(activated)[0]),
                "r"(reinterpret_cast<uint32_t*>(activated)[1]),
                "r"(reinterpret_cast<uint32_t*>(activated)[2]),
                "r"(reinterpret_cast<uint32_t*>(activated)[3]), "r"(store_addr));
          }
        }
      }
      // Publish every warp's SwiGLU stores before the leader starts the down MMA.
      asm volatile("tcgen05.wait::st.sync.aligned;");
      arrive(e_to_c_bar);

      float token_scale[C::TPT];
      for (int t = 0; t < C::TPT; t++) {
        int token_idx = epilogue_wid * C::TPT * 8 + t * 8 + lane_id / 4;
        const int routed_token = local_token_ids[token_idx];
        token_scale[t] =
            routed_token / top_k < M ? topk_weights[routed_token] * scaling_factor : 0.0f;
      }

      epilogue_sync();

      for (int compute_stage = 0; compute_stage < n_stages_down; compute_stage++) {
        int col = compute_stage;
        wait(c_to_e_bar, epilogue_p);
        epilogue_p ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;");
        float acc[8];
        nv_bfloat162 tile_res[C::BM / 64][C::BN2 / 16][4];
        for (int row_off = 0; row_off < C::BM / 4; row_off += 16) {
          for (int tn = 0; tn < C::BN2 / 16; tn++) {
            const int row = epilogue_wid * 32 + row_off;
            const int col = tn * 16;
            const uint32_t addr = out_tmem + (row << 16) + col;
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0, %1, %2, %3, %4, "
                "%5, "
                "%6, %7}, [%8];"
                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]), "=f"(acc[4]),
                  "=f"(acc[5]), "=f"(acc[6]), "=f"(acc[7])
                : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            auto& res = tile_res[row_off / 16][tn];
            res[0] = __float22bfloat162_rn(
                {acc[0] * token_scale[row_off / 8 + 0], acc[1] * token_scale[row_off / 8 + 0]});
            res[1] = __float22bfloat162_rn(
                {acc[2] * token_scale[row_off / 8 + 1], acc[3] * token_scale[row_off / 8 + 1]});
            res[2] = __float22bfloat162_rn(
                {acc[4] * token_scale[row_off / 8 + 0], acc[5] * token_scale[row_off / 8 + 0]});
            res[3] = __float22bfloat162_rn(
                {acc[6] * token_scale[row_off / 8 + 1], acc[7] * token_scale[row_off / 8 + 1]});
          }
        }
        // Release TMEM once the full tile is in registers so the next MMA can overlap
        // stores.
        asm volatile("tcgen05.fence::before_thread_sync;");
        arrive(e_to_c_bar);
        int stage_off = C::DB_STORE ? (compute_stage % 2) * C::BM * C::PAD : 0;
        if constexpr (C::DB_STORE) {
          asm volatile("cp.async.bulk.wait_group 1;");
        } else {
          asm volatile("cp.async.bulk.wait_group 0;");
        }
        epilogue_sync();
        for (int row_off = 0; row_off < C::BM / 4; row_off += 16) {
          for (int tn = 0; tn < C::BN2 / 16; tn++) {
            const int row = epilogue_wid * 32 + row_off;
            const int smem_row = C::BM == 128 ? row : epilogue_wid * 16;
            const int col = tn * 16;
            auto& res = tile_res[row_off / 16][tn];
            const int store_row = smem_row + lane_id % 16;
            const int store_col = col + (lane_id / 16) * 8;
            const uint32_t store_addr =
                __cvta_generic_to_shared(s_d.out + stage_off + store_row * C::PAD + store_col);
            st_matrix_x4(reinterpret_cast<uint32_t*>(res), store_addr);
          }
        }
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
        epilogue_sync();
        if (lane_id < C::SAVES_PER_WARP) {
          if (token_src < M) {
            int row = epilogue_wid * C::SAVES_PER_WARP + lane_id;
            cuda::ptx::cp_reduce_async_bulk(cuda::ptx::space_global, cuda::ptx::space_shared,
                                            cuda::ptx::op_add, out + token_src * K + col * C::BN2,
                                            s_d.out + stage_off + row * C::PAD,
                                            C::BN2 * sizeof(__nv_bfloat16));
          }
          cuda::ptx::cp_async_bulk_commit_group();
        }
      }
      // Drain every thread's bulk group before the next UP overwrites the overlay.
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
      arrive(e_to_p_bar);
    }
  }
  // Keep TMEM allocated until every epilogue warp has consumed the final tile.
  __syncthreads();
  if (warp_id == 1) {
    tmem_dealloc(tmem_addr, C::TMEM_COLS);
  }
#else
  assert(false && "This fused MoE kernel requires SM100 or newer");
#endif
}

template <int BM, int BN, int W, int STAGES, moe_activation::Kind ACTIVATION>
bool launch_fused_moe_kernel_up_down_acc(
    const bf16* x, bf16* w, bf16* w2, bf16* out, const int* sorted_token_ids, const int* expert_ids,
    const int* num_tokens_post_padded, const float* topk_weights, const int top_k, int M, int N,
    int K, int num_experts, int sorted_num, int block_m, float scaling_factor,
    typename moe_activation::Traits<ACTIVATION>::Params activation_params, cudaStream_t stream) {
  using C = constants<BM, BN, STAGES, ACTIVATION>;
  const int N2 = K;
  const int K2 = N / C::UD_RATIO;
  const int SPLIT_K = (N + C::BLN - 1) / C::BLN;
  dim3 dimBlock(C::BLOCK_THREADS, 1, 1);

  if constexpr (C::sMemSize > C::SM100_MAX_SMEM) {
    fprintf(stderr,
            "fused_moe: BM=%d BLN=%d WN=%d STAGES=%d requires %zu bytes of "
            "shared memory but SM100 max is %zu — skipping launch\n",
            BM, C::BLN, C::WN, STAGES, C::sMemSize, C::SM100_MAX_SMEM);
    return false;
  }
  if (C::BLN > ((N + 255) / 256) * 256) {
    fprintf(stderr, "Invalid config parameters BLN(%d) > N(%d), skipping launch", C::BLN, N);
    return false;
  }

  if constexpr (C::sMemSize <= C::SM100_MAX_SMEM) {
    if (!(N % C::BLN == 0 || (C::BLN == 256 && N % 256 != 0))) return false;
    int max_smem_real;
    int device;
    gpuErrchk(cudaGetDevice(&device));
    cudaDeviceGetAttribute(&max_smem_real, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (C::sMemSize > max_smem_real) return false;

    int num_sms;
    gpuErrchk(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
    const int block_count =
        std::min(std::max(1, num_sms / SPLIT_K), (sorted_num + block_m - 1) / block_m);
    if (block_count == 0) return true;
    dim3 dimGrid(block_count * SPLIT_K, 1, 1);

    auto kernel = fused_moe_bf16_wgmma_up_down_acc_kernel<C>;
    cudaFuncAttributes attributes{};
    const auto status = cudaFuncGetAttributes(&attributes, kernel);
    if (status != cudaSuccess) {
      cudaGetLastError();
      return false;
    }

    gpuErrchk(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::sMemSize));

    // A separate expert axis lets TMA zero-fill partial tiles without reading the next expert.
    CUtensorMap tensor_map_w{};
    {
      uint64_t size[C::TMA_RANK_W] = {static_cast<uint64_t>(K), static_cast<uint64_t>(N),
                                      static_cast<uint64_t>(num_experts)};
      uint64_t stride[C::TMA_RANK_W - 1] = {static_cast<uint64_t>(K) * sizeof(bf16),
                                            static_cast<uint64_t>(N) * K * sizeof(bf16)};
      uint32_t box_size[C::TMA_RANK_W] = {C::BK, C::TILE_COLS_UP, 1};
      uint32_t elem_stride[C::TMA_RANK_W] = {1, 1, 1};
      auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
      CUresult res = cuTensorMapEncodeTiled(
          &tensor_map_w, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, C::TMA_RANK_W, w,
          size, stride, box_size, elem_stride, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (res != CUDA_SUCCESS) return false;
    }

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
    {
      // Reshape (K2, N2*E) into (TILE_INNER_DN, N2*E, K2/TILE_INNER_DN)
      // so that tile chunks end up in dim2 (outermost, varies slowest in smem fill)
      // producing contiguous per-tile regions in shared memory.
      uint64_t size[C::TMA_RANK_W] = {static_cast<uint64_t>(C::TILE_INNER_DN),
                                      static_cast<uint64_t>(N2 * num_experts),
                                      static_cast<uint64_t>(K2 / C::TILE_INNER_DN)};
      uint64_t stride[C::TMA_RANK_W - 1] = {static_cast<uint64_t>(K2) * sizeof(bf16),
                                            static_cast<uint64_t>(C::TILE_INNER_DN) * sizeof(bf16)};
      uint32_t box_size[C::TMA_RANK_W] = {C::TILE_INNER_DN, C::BN2,
                                          static_cast<uint32_t>(C::TILES_DOWN)};
      uint32_t elem_stride[C::TMA_RANK_W] = {1, 1, 1};
      auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
      CUresult res = cuTensorMapEncodeTiled(
          &tensor_map_w2, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, C::TMA_RANK_W, w2,
          size, stride, box_size, elem_stride, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (res != CUDA_SUCCESS) return false;
    }

    kernel<<<dimGrid, dimBlock, C::sMemSize, stream>>>(
        tensor_map_x, tensor_map_w, tensor_map_w2, out, sorted_token_ids, expert_ids,
        num_tokens_post_padded, topk_weights, top_k, M, N, K, scaling_factor, activation_params);
    gpuErrchk(cudaGetLastError());
    return true;
  }
  return false;
}

// clang-format off
template <int BM, int BN, int W, moe_activation::Kind ACTIVATION>
bool dispatch_stages(int stages, const bf16* x, bf16* w, bf16* w2, bf16* out, const int* sorted_token_ids, const int* expert_ids, const int* num_tokens_post_padded, const float* topk_weights, const int top_k, int M, int N, int K, int num_experts, int sorted_num, int block_m, float scaling_factor, typename moe_activation::Traits<ACTIVATION>::Params activation_params, cudaStream_t stream) {
    using C = constants<BM, BN, 1, ACTIVATION>;
    constexpr int MAX_STAGES = C::MAX_STAGES;
    if constexpr (MAX_STAGES > 0) {
        if (stages == MAX_STAGES)
            return launch_fused_moe_kernel_up_down_acc<BM, BN, W, MAX_STAGES, ACTIVATION>(x, w, w2, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, N, K, num_experts, sorted_num, block_m, scaling_factor, activation_params, stream);
    }
    if constexpr (MAX_STAGES > 1) {
        if (stages == MAX_STAGES - 1)
            return launch_fused_moe_kernel_up_down_acc<BM, BN, W, MAX_STAGES - 1, ACTIVATION>(x, w, w2, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, N, K, num_experts, sorted_num, block_m, scaling_factor, activation_params, stream);
    }
    return false;
}
// clang-format on

}  // namespace moe_sm100_tmem_persistent
