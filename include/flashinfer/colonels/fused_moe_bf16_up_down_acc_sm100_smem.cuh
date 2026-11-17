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

namespace moe_sm100 {
template <int BLOCK_M, int BLOCK_N, int PIPELINE_STAGES, bool USE_RED_STORE,
          moe_activation::Kind ACTIVATION>
struct constants {
  static constexpr bool RED_STORE = USE_RED_STORE;
  using Activation = moe_activation::Traits<ACTIVATION>;
  using ActivationParams = typename Activation::Params;
  static constexpr bool IS_GATED = Activation::GATED;
  static constexpr int UD_RATIO = IS_GATED ? 2 : 1;
  static constexpr int BM = BLOCK_M;
  static constexpr int CTA_GROUP = BM == 256 ? 2 : 1;
  static constexpr int LOCAL_M = BM / CTA_GROUP;
  static constexpr int STAGES = PIPELINE_STAGES;
  static constexpr int BK = 64;
  // Native M256 MMA splits token rows and weight columns across two CTAs.
  static constexpr int WN = 4;
  static constexpr int BLN = BLOCK_N * WN;
  static constexpr int CLUSTERS = CTA_GROUP;
  static constexpr int MMA_THREADS = WN * 32;
  // Spread gather4 requests across eight warps to reduce per-warp issue serialization.
  static constexpr int PRODUCER_THREADS = 256;
  static constexpr int EPILOGUE_THREADS = 256;
  static constexpr int BLOCK_THREADS = MMA_THREADS + PRODUCER_THREADS + EPILOGUE_THREADS;
  static constexpr int EPILOGUE_WARPGROUPS = EPILOGUE_THREADS / 128;
  static_assert(EPILOGUE_THREADS == 128 || EPILOGUE_THREADS == 256 || EPILOGUE_THREADS == 512);
  static_assert(BLOCK_THREADS <= 1024);

  static constexpr int MMA_N_UP = 256;
  static constexpr int TMEM_COLS = 512;
  // UP and DOWN share a ring of accumulator slots, each with its own barrier pair.
  static constexpr int EP_BUFFERS = TMEM_COLS / MMA_N_UP;
  // Blackwell's opt-in dynamic shared memory limit per block.
  static constexpr size_t SM100_MAX_SMEM = 232448;
  static constexpr uint32_t S_BITS_UP = 3;
  static constexpr uint32_t S_MODE_UP = 4 - S_BITS_UP;
  static constexpr uint32_t S_BITS_DOWN = 3;
  static constexpr uint32_t S_MODE_DOWN = 4 - S_BITS_DOWN;
  static constexpr int WS = BK * MMA_N_UP / CTA_GROUP;
  static constexpr int XS = BK * LOCAL_M;
  static constexpr int MMA_N_DOWN = BLN / UD_RATIO >= 256 ? 256 : 128;
  static constexpr int BN2 = MMA_N_DOWN;
  static constexpr int BK2 = BK * MMA_N_UP / BN2;
  static constexpr int TMA_TILES_DOWN = BK2 / 64;
  static constexpr int TILE_COLS_UP = MMA_N_UP / CTA_GROUP;
  static constexpr int TILE_INNER_DN = BK2 / TMA_TILES_DOWN;
  static constexpr int TPT = LOCAL_M / 64;
  static constexpr uint32_t idesc_up = build_idesc<BM, MMA_N_UP, CTA_GROUP>();
  static constexpr uint32_t idesc_down = build_idesc<BM, MMA_N_DOWN, CTA_GROUP>();
  static constexpr int LOADS_PER_WARP = LOCAL_M / ((PRODUCER_THREADS / 32) * 4);
  static constexpr int LAST_UP_TILE = BLN / MMA_N_UP - 1;
  static constexpr int LAST_UP_K = LAST_UP_TILE * (MMA_N_UP / UD_RATIO);
  static constexpr int LAST_UP_SLOT = LAST_UP_TILE % EP_BUFFERS;
  static constexpr int LAST_UP_RELEASE_PHASE = (LAST_UP_TILE / EP_BUFFERS + 1) % 2;
  static constexpr uint32_t TMA_RANK_X = 2;

  struct smem_up {
    alignas(1024) bf16 w[STAGES * WS];
    alignas(1024) bf16 x[STAGES * XS];
  };

  static constexpr int LAST_UP_X = LOCAL_M * MMA_N_UP / UD_RATIO;
  static constexpr int OUTPUT_COLS = BN2 / EPILOGUE_WARPGROUPS;
  // Padding shifts consecutive rows by four banks for 16-byte lane stores.
  static constexpr int OUTPUT_STRIDE = OUTPUT_COLS + 8;
  static constexpr int OUTPUT_ELEMENTS = RED_STORE ? 0 : EPILOGUE_THREADS / 2 * OUTPUT_STRIDE;
  static constexpr int EARLY_UP_X_OFFSET = std::max(STAGES * XS, LAST_UP_X + OUTPUT_ELEMENTS);

  // The final activation tile reuses UP x after its completion barrier. Earlier
  // tiles stay beyond every UP input stage so their epilogues can overlap UP.
  // Bulk output staging occupies the gap immediately after the final activation tile.
  struct smem_down {
    alignas(1024) bf16 w[STAGES * WS];
    alignas(1024) bf16 x[EARLY_UP_X_OFFSET + LOCAL_M * LAST_UP_K];
  };

  __host__ __device__ static constexpr int down_x_offset(int k) {
    return k >= LAST_UP_K ? (k - LAST_UP_K) * LOCAL_M : EARLY_UP_X_OFFSET + k * LOCAL_M;
  }

  struct smem_persistent {
    alignas(8) uint64_t bar[2 * STAGES + 2 * EP_BUFFERS];
    uint32_t tmem_addr;
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
  static constexpr size_t smem_down_pad_bytes = (EARLY_UP_X_OFFSET - LAST_UP_X) * sizeof(bf16);
  static constexpr size_t smem_down_x_bytes = LOCAL_M * BLN / UD_RATIO * sizeof(bf16);
  // Round persistent storage up to the overlay's 1024-byte alignment.
  static constexpr size_t OVERLAY_OFF = (smem_persistent_bytes + 1023) & ~size_t(1023);
  static constexpr size_t sMemSize = OVERLAY_OFF + std::max(smem_up_bytes, smem_down_bytes);

  static constexpr int MAX_STAGES = [] {
    // Each pipeline stage adds two barriers to the persistent storage.
    constexpr size_t metadata = smem_persistent_bytes - 2 * STAGES * sizeof(uint64_t);
    for (int stages = 1;; ++stages) {
      const size_t overlay = (metadata + 2 * stages * sizeof(uint64_t) + 1023) & ~size_t(1023);
      const size_t x_elements =
          std::max(stages * XS, LAST_UP_X + OUTPUT_ELEMENTS) + LOCAL_M * LAST_UP_K;
      const size_t operands = (stages * WS + x_elements) * sizeof(bf16);
      if (overlay + ((operands + 1023) & ~size_t(1023)) > SM100_MAX_SMEM) return stages - 1;
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
  const int32_t warpM = blockIdx.y;
  if (warpM * C::BM >= num_tokens_post_padded[0]) return;
  const int exp_idx = expert_ids[warpM];
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

  const int* local_token_ids = sorted_token_ids + warpM * C::BM + cta_rank * C::LOCAL_M;

  extern __shared__ __align__(1024) uint8_t sh[];

  auto& smem_p = *reinterpret_cast<typename C::smem_persistent*>(sh);
  uint64_t* bar = smem_p.bar;
  // EPILOGUE BUFFERS
  uint64_t* e_to_c_bar = bar + 2 * C::STAGES;
  uint64_t* c_to_e_bar = e_to_c_bar + C::EP_BUFFERS;

  typename C::smem_up& s = *reinterpret_cast<typename C::smem_up*>(sh + C::OVERLAY_OFF);
  typename C::smem_down& s_d = *reinterpret_cast<typename C::smem_down*>(sh + C::OVERLAY_OFF);

  // Since it reduces occupancy to 1, is there a reason to request less?
  if (threadIdx.x == 0) {
    for (int i = 0; i < C::STAGES; i++) {
      init_barrier(&bar[i], C::CTA_GROUP == 2 ? 1 : C::LOCAL_M / 4 + 1, 0);
      init_barrier(&bar[i + C::STAGES], 1, 0);
    }
    for (int i = 0; i < C::EP_BUFFERS; i++) {
      init_barrier(e_to_c_bar + i, C::CTA_GROUP * C::EPILOGUE_THREADS, 0);
      init_barrier(c_to_e_bar + i, 1, 0);
    }
  } else if (warp_id == 1) {
    tmem_alloc<C::CTA_GROUP>(__cvta_generic_to_shared(&smem_p.tmem_addr), C::TMEM_COLS);
  }

  if constexpr (C::CLUSTERS > 1) {
    cg::this_cluster().sync();
  } else {
    __syncthreads();
  }
  uint32_t tmem_addr = smem_p.tmem_addr;

  int p = 0;
  // PRODUCER
  if (is_producer) {
    int producer_tid = threadIdx.x - C::MMA_THREADS;
    int producer_wid = producer_tid / 32;
    int smem_stage = 0;
    int4 tokens;

    if (producer_tid % 32 < C::LOADS_PER_WARP) {
      tokens = reinterpret_cast<const int4*>(
          local_token_ids)[(producer_tid / 32) * C::LOADS_PER_WARP + producer_tid % 32];
      tokens.x /= top_k;
      tokens.y /= top_k;
      tokens.z /= top_k;
      tokens.w /= top_k;
    }
    for (int n_off = 0; n_off < C::BLN; n_off += C::MMA_N_UP) {
      for (int load_stage = 0; load_stage < n_stages_up; load_stage++) {
        if (smem_stage == C::STAGES) {
          p ^= 1;
          smem_stage = 0;
        }
        const int off = load_stage * C::BK;

        wait<C::CLUSTERS>(bar + C::STAGES + smem_stage, p);
        if (producer_wid == 0 && elect_sync()) {
          if constexpr (C::CTA_GROUP == 2) {
            if (cta_rank == 0)
              expect_bytes(bar + smem_stage, C::CTA_GROUP * (C::WS + C::XS) * sizeof(bf16));
          } else {
            expect_bytes(bar + smem_stage, C::WS * sizeof(bf16));
          }
          load_async_4d<C::CTA_GROUP>(
              s.w + smem_stage * C::WS, &tensor_map_w, bar + smem_stage, 0,
              (blockIdx.x / C::CTA_GROUP) * C::BLN + n_off + cta_rank * C::TILE_COLS_UP, off / 64,
              exp_idx);
        }

        if (producer_tid % 32 < C::LOADS_PER_WARP) {
          if constexpr (C::CTA_GROUP == 1) expect_bytes(bar + smem_stage, 4 * C::BK * sizeof(bf16));
          int idx = (producer_tid / 32) * C::LOADS_PER_WARP + producer_tid % 32;
          load_async_2d_gather4<C::CLUSTERS, C::CTA_GROUP>(
              s.x + smem_stage * C::XS + idx * 4 * C::BK, &tensor_map_x, bar + smem_stage, off,
              tokens, uint16_t(1u << cta_rank));
        }

        smem_stage++;
      }
    }

    if (producer_wid == 0 && elect_sync()) {
      for (int col = 0; col < n_stages_down; col++) {
        for (int k_off = 0; k_off < C::BLN / C::UD_RATIO; k_off += C::BK2) {
          if (smem_stage == C::STAGES) {
            p ^= 1;
            smem_stage = 0;
          }
          wait<C::CLUSTERS>(bar + C::STAGES + smem_stage, p);
          const int w_row_down = (blockIdx.x / C::CTA_GROUP) * (C::BLN / C::UD_RATIO) + k_off;
          const int w_col_down = col * C::BN2 + cta_rank * (C::BN2 / C::CTA_GROUP);

          if (cta_rank == 0) expect_bytes(bar + smem_stage, C::CTA_GROUP * C::WS * sizeof(bf16));
          load_async_4d<C::CTA_GROUP>(s_d.w + smem_stage * C::WS, &tensor_map_w2, bar + smem_stage,
                                      0, w_col_down, w_row_down / C::TILE_INNER_DN, exp_idx);
          if constexpr (C::CTA_GROUP == 1) arrive(bar + smem_stage, C::LOCAL_M / 4);
          smem_stage++;
        }
      }
    }
  }
  // MMA
  else if (is_mma) {
    // Empty barriers arrive instantly
    int smem_stage = 0;
    int epilogue_stage = 0;
    int e_p = 0;
    if (warp_id == 0 && elect_sync()) {
      for (int i = 0; i < C::STAGES; i++) arrive(&bar[C::STAGES + i]);
    }
    if (cta_rank == 0 && warp_id == 0 && elect_sync()) {
      for (int n_off = 0; n_off < C::BLN; n_off += C::MMA_N_UP) {
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
          // Some sources say you need this sync but
          // https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2
          // explains why you actually dont
          // asm volatile("tcgen05.fence::after_thread_sync;");
          uint64_t desc_w = make_smem_descriptor<64, 1, C::S_MODE_UP>(s.w + smem_stage * C::WS);
          uint64_t desc_x = make_smem_descriptor<64, 1, C::S_MODE_UP>(s.x + smem_stage * C::XS);
          for (int k = 0; k < C::BK; k += 16) {
            tcgen05_mma_f16<C::CLUSTERS>(tmem_addr + epilogue_stage * C::MMA_N_UP, desc_x, desc_w,
                                         C::idesc_up, 1 - int(compute_stage == 0 && k == 0));
            desc_x += (32 >> 4);
            desc_w += (32 >> 4);
          }
          tcgen05_commit<C::CLUSTERS>(__cvta_generic_to_shared(bar + C::STAGES + smem_stage));
          smem_stage++;
        }
        // Separate completion barriers let SwiGLU trail UP by either tile.
        tcgen05_commit<C::CLUSTERS>(__cvta_generic_to_shared(c_to_e_bar + epilogue_stage));
        epilogue_stage++;
      }

      for (int compute_stage = 0; compute_stage < n_stages_down; compute_stage++) {
        if (epilogue_stage == C::EP_BUFFERS) {
          e_p ^= 1;
          epilogue_stage = 0;
        }
        wait<C::CLUSTERS>(e_to_c_bar + epilogue_stage, e_p);
        for (int k_off = 0; k_off < C::BLN / C::UD_RATIO; k_off += C::BK2) {
          if (smem_stage == C::STAGES) {
            p ^= 1;
            smem_stage = 0;
          }
          // The first DOWN accumulator wait already covers the penultimate UP tile.
          // Only wait for the final SwiGLU tile when its input columns are needed.
          // Check once: this barrier is reused by later DOWN epilogues.

          static_assert((C::MMA_N_UP / C::UD_RATIO) % C::BK2 == 0);
          if (compute_stage == 0 && k_off == C::LAST_UP_K) {
            wait<C::CLUSTERS>(e_to_c_bar + C::LAST_UP_SLOT, C::LAST_UP_RELEASE_PHASE);
          }
          wait<C::CLUSTERS>(bar + smem_stage, p);
          for (int w_tile = 0; w_tile < C::TMA_TILES_DOWN; w_tile++) {
            uint64_t desc_w = make_smem_descriptor<64, 1, C::S_MODE_DOWN>(
                s_d.w + smem_stage * C::WS + w_tile * C::WS / C::TMA_TILES_DOWN);
            uint64_t desc_x = make_smem_descriptor<64, 1, C::S_MODE_UP>(
                s_d.x + C::down_x_offset(k_off + w_tile * 64));
            for (int k = 0; k < 64; k += 16) {
              // Offset by MMA_N_UP because MM_N_DOWN can be smaller
              // and then we could write to overlapping buffer
              tcgen05_mma_f16<C::CLUSTERS>(tmem_addr + epilogue_stage * C::MMA_N_UP, desc_x, desc_w,
                                           C::idesc_down, int(k_off != 0 || w_tile != 0 || k != 0));
              desc_w += (32 >> 4);
              desc_x += (32 >> 4);
            }
          }
          tcgen05_commit<C::CLUSTERS>(__cvta_generic_to_shared(bar + C::STAGES + smem_stage));
          smem_stage++;
        }
        tcgen05_commit<C::CLUSTERS>(__cvta_generic_to_shared(c_to_e_bar + epilogue_stage));
        epilogue_stage++;
      }
    }
  } else if (is_epilogue) {
    int epilogue_stage = 0;
    int e_p = 0;
    int epilogue_tid = threadIdx.x - C::PRODUCER_THREADS - C::MMA_THREADS;

    const int epilogue_wid = (epilogue_tid / 32) % 4;
    const int epilogue_wgid = epilogue_tid / 128;

    // Seed both empty slots before the first MMA acquisition.
    for (int i = 0; i < C::EP_BUFFERS; i++) {
      arrive_mma_leader<C::CTA_GROUP>(e_to_c_bar + i, cta_rank);
    }

    for (int n_off = 0; n_off < C::BLN; n_off += C::MMA_N_UP) {
      if (epilogue_stage == C::EP_BUFFERS) {
        e_p ^= 1;
        epilogue_stage = 0;
      }
      wait<C::CLUSTERS>(c_to_e_bar + epilogue_stage, e_p);
      asm volatile("tcgen05.fence::after_thread_sync;");

      uint32_t acc_tmem = tmem_addr + epilogue_stage * C::MMA_N_UP;
      float acc[16];
      for (int row_off_tmem = 0; row_off_tmem < C::LOCAL_M / 4; row_off_tmem += 16) {
        // Warp groups share the TMEM rows and consume disjoint columns.
        for (int tn = epilogue_wgid * 32; tn < C::MMA_N_UP; tn += C::EPILOGUE_WARPGROUPS * 32) {
          const int row = epilogue_wid * 32 + row_off_tmem;
          const int col = tn;
          const uint32_t addr = acc_tmem + (row << 16) + col;
          asm volatile(
              "tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0, %1, %2, %3, %4, %5, "
              "%6, "
              "%7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
              : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]), "=f"(acc[4]), "=f"(acc[5]),
                "=f"(acc[6]), "=f"(acc[7]), "=f"(acc[8]), "=f"(acc[9]), "=f"(acc[10]),
                "=f"(acc[11]), "=f"(acc[12]), "=f"(acc[13]), "=f"(acc[14]), "=f"(acc[15])
              : "r"(addr));
          asm volatile("tcgen05.wait::ld.sync.aligned;");

          if constexpr (C::IS_GATED) {
            nv_bfloat162 swiglu[4];
            swiglu[0] = make_bfloat162(
                moe_activation::apply<typename C::Activation>(acc[0], acc[4], activation_params),
                moe_activation::apply<typename C::Activation>(acc[1], acc[5], activation_params));
            swiglu[1] = make_bfloat162(
                moe_activation::apply<typename C::Activation>(acc[2], acc[6], activation_params),
                moe_activation::apply<typename C::Activation>(acc[3], acc[7], activation_params));
            swiglu[2] = make_bfloat162(
                moe_activation::apply<typename C::Activation>(acc[8], acc[12], activation_params),
                moe_activation::apply<typename C::Activation>(acc[9], acc[13], activation_params));
            swiglu[3] = make_bfloat162(
                moe_activation::apply<typename C::Activation>(acc[10], acc[14], activation_params),
                moe_activation::apply<typename C::Activation>(acc[11], acc[15], activation_params));

            // BLOCKS are stored in separate
            // contiguous BM*BK regions (each with its own swizzle).
            int out_k = (n_off + tn) / 2;
            int smem_block = out_k / C::BK;
            int smem_col = out_k % C::BK + (lane_id / 16) * 8;
            int smem_row = epilogue_wid * (C::LOCAL_M / 4) + row_off_tmem + lane_id % 16;
            bf16* smem_addr =
                s_d.x + C::down_x_offset(smem_block * C::BK) + smem_row * C::BK + smem_col;
            uint32_t swizzled = swizzle<3, 4, 3>(__cvta_generic_to_shared(smem_addr));

            st_matrix_x4(reinterpret_cast<uint32_t*>(swiglu), swizzled);
          } else {
            for (int i = 0; i < 8; i++) {
              const nv_bfloat162 activated = make_bfloat162(
                  moe_activation::apply<typename C::Activation>(acc[2 * i], activation_params),
                  moe_activation::apply<typename C::Activation>(acc[2 * i + 1], activation_params));
              const int row =
                  epilogue_wid * (C::LOCAL_M / 4) + row_off_tmem + lane_id / 4 + (i % 2) * 8;
              const int col = n_off + tn + (lane_id % 4) * 2 + (i / 2) * 8;
              bf16* dst =
                  s_d.x + C::down_x_offset((col / C::BK) * C::BK) + row * C::BK + col % C::BK;
              const uint32_t swizzled = swizzle<3, 4, 3>(__cvta_generic_to_shared(dst));
              *reinterpret_cast<nv_bfloat162*>(__cvta_shared_to_generic(swizzled)) = activated;
            }
          }
        }
      }
      // Publish SwiGLU and finish TMEM reads before releasing this slot.
      cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
      arrive_mma_leader<C::CTA_GROUP>(e_to_c_bar + epilogue_stage, cta_rank);
      epilogue_stage++;
    }

    float token_scale[C::TPT];
    int token_dst[C::TPT];
    for (int t = 0; t < C::TPT; t++) {
      const int token_idx = epilogue_wid * (C::LOCAL_M / 4) + t * 16 + lane_id % 16;
      const int routed_token = local_token_ids[token_idx];
      token_scale[t] =
          routed_token / top_k < M ? topk_weights[routed_token] * scaling_factor : 0.0f;
      token_dst[t] = routed_token / top_k;
    }

    for (int compute_stage = 0; compute_stage < n_stages_down; compute_stage++) {
      if (epilogue_stage == C::EP_BUFFERS) {
        e_p ^= 1;
        epilogue_stage = 0;
      }
      wait<C::CLUSTERS>(c_to_e_bar + epilogue_stage, e_p);
      asm volatile("tcgen05.fence::after_thread_sync;");
      float acc[8];
      nv_bfloat162 tile_res[4];
      for (int row_off = 0; row_off < C::LOCAL_M / 4; row_off += 16) {
        const int token = token_dst[row_off / 16];
        if constexpr (!C::RED_STORE) {
          asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
          __syncwarp();
        }
        const int col_base = epilogue_wgid * C::OUTPUT_COLS;
        const int tn_begin = !C::RED_STORE ? col_base / 16 : epilogue_wgid;
        const int tn_end = !C::RED_STORE ? (col_base + C::OUTPUT_COLS) / 16 : C::BN2 / 16;
        const int tn_step = !C::RED_STORE ? 1 : C::EPILOGUE_WARPGROUPS;
        for (int tn = tn_begin; tn < tn_end; tn += tn_step) {
          const int row = epilogue_wid * 32 + row_off;
          const int col = tn * 16;
          const uint32_t addr = tmem_addr + epilogue_stage * C::MMA_N_UP + (row << 16) + col;
          // To coalesce stores we load 2 adjacent 32B chunks:
          // lane 0 cols 0-7 | lane 16 cols 8-15
          // lane 1 cols 0-7 | lane 17 cols 8-15 ...
          // After casting to bf16 lane 0 and 16 coalesce into 32B sector
          asm volatile(
              "tcgen05.ld.sync.aligned.16x32bx2.x8.b32 {%0, %1, %2, %3, %4, %5, "
              "%6, %7}, [%8], 8;"
              : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]), "=f"(acc[4]), "=f"(acc[5]),
                "=f"(acc[6]), "=f"(acc[7])
              : "r"(addr));
          asm volatile("tcgen05.wait::ld.sync.aligned;");
          const float scale = token_scale[row_off / 16];
          for (int i = 0; i < 4; i++) {
            tile_res[i] = __float22bfloat162_rn({acc[2 * i] * scale, acc[2 * i + 1] * scale});
          }
          float4 values = *reinterpret_cast<float4*>(tile_res);
          if constexpr (!C::RED_STORE) {
            const int output_row = (epilogue_tid / 32) * 16 + lane_id % 16;
            const int output_col = tn * 16 - col_base + (lane_id / 16) * 8;
            *reinterpret_cast<float4*>(s_d.x + C::LAST_UP_X + output_row * C::OUTPUT_STRIDE +
                                       output_col) = values;
          } else {
            const int store_col = compute_stage * C::BN2 + tn * 16 + (lane_id / 16) * 8;
            if (token < M) {
              red_add_bf16x8(out + size_t(token) * N2 + store_col, values);
            }
          }
        }
        // Every accumulator has been read; the reduction only needs SMEM now.
        if (row_off + 16 >= C::LOCAL_M / 4)
          arrive_mma_leader<C::CTA_GROUP>(e_to_c_bar + epilogue_stage, cta_rank);
        if constexpr (!C::RED_STORE) {
          __syncwarp();
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          if (lane_id < 16 && token < M) {
            const int output_row = (epilogue_tid / 32) * 16 + lane_id;
            asm volatile(
                "cp.reduce.async.bulk.global.shared::cta.bulk_group."
                "add.noftz.bf16 [%0], [%1], %2;" ::"l"(out + size_t(token) * N2 +
                                                       compute_stage * C::BN2 + col_base),
                "r"(static_cast<uint32_t>(__cvta_generic_to_shared(s_d.x + C::LAST_UP_X +
                                                                   output_row * C::OUTPUT_STRIDE))),
                "r"(uint32_t(C::OUTPUT_COLS * sizeof(bf16)))
                : "memory");
          }
          asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        }
      }
      epilogue_stage++;
    }
  }
  // Keep TMEM allocated until every epilogue warp has consumed the final tile.
  if constexpr (C::CLUSTERS > 1) {
    cg::this_cluster().sync();
  } else {
    __syncthreads();
  }
  if (warp_id == 1) {
    tmem_dealloc<C::CTA_GROUP>(tmem_addr, C::TMEM_COLS);
  }
#else
  assert(false && "This fused MoE kernel requires SM100 or newer");
#endif
}

template <int BM, int BN, int W, int STAGES, bool RED_STORE, moe_activation::Kind ACTIVATION>
bool launch_fused_moe_kernel_up_down_acc(
    const bf16* x, bf16* w, bf16* w2, bf16* out, const int* sorted_token_ids, const int* expert_ids,
    const int* num_tokens_post_padded, const float* topk_weights, const int top_k, int M, int N,
    int K, int num_experts, int sorted_num, int block_m, float scaling_factor,
    typename moe_activation::Traits<ACTIVATION>::Params activation_params, cudaStream_t stream) {
  using C = constants<BM, BN, STAGES, RED_STORE, ACTIVATION>;
  const int N2 = K;
  const int K2 = N / C::UD_RATIO;

  dim3 dimBlock(C::BLOCK_THREADS, 1, 1);
  dim3 dimGrid(C::CTA_GROUP * ((N + C::BLN - 1) / C::BLN), std::ceil((float)sorted_num / (block_m)),
               1);

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

  if constexpr (C::sMemSize <= C::SM100_MAX_SMEM && C::BLN % C::MMA_N_UP == 0) {
    int max_smem_real;
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&max_smem_real, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (C::sMemSize > max_smem_real) return false;

    auto kernel = fused_moe_bf16_wgmma_up_down_acc_kernel<C>;
    cudaFuncAttributes attributes{};
    const auto status = cudaFuncGetAttributes(&attributes, kernel);
    if (status != cudaSuccess) {
      cudaGetLastError();
      return false;
    }

    gpuErrchk(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::sMemSize));

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
    static_assert(C::BLN >= C::MMA_N_UP);

    if constexpr (C::BN2 / 64 < C::WN / 4) {
      fprintf(stderr, "Too much warpgroups(%d) for block N(%d)\n", C::WN, C::BN2);
      return false;
    }
    CUtensorMap tensor_map_w2{};
    if (!encode_bf16_weight_map(&tensor_map_w2, w2, num_experts, N2, K2, C::BN2 / C::CTA_GROUP,
                                C::BK2))
      return false;

    kernel<<<dimGrid, dimBlock, C::sMemSize, stream>>>(
        tensor_map_x, tensor_map_w, tensor_map_w2, out, sorted_token_ids, expert_ids,
        num_tokens_post_padded, topk_weights, top_k, M, N, K, scaling_factor, activation_params);
    gpuErrchk(cudaGetLastError());
    return true;
  }
  return false;
}

// clang-format off
template <int BM, int BN, int W, bool RED_STORE, moe_activation::Kind ACTIVATION>
bool dispatch_stages(int stages, const bf16* x, bf16* w, bf16* w2, bf16* out, const int* sorted_token_ids, const int* expert_ids, const int* num_tokens_post_padded, const float* topk_weights, const int top_k, int M, int N, int K, int num_experts, int sorted_num, int block_m, float scaling_factor, typename moe_activation::Traits<ACTIVATION>::Params activation_params, cudaStream_t stream) {
    using C = constants<BM, BN, 1, RED_STORE, ACTIVATION>;
    constexpr int MAX_STAGES = C::MAX_STAGES;
    if constexpr (MAX_STAGES > 0) {
        if (stages == MAX_STAGES)
            return launch_fused_moe_kernel_up_down_acc<BM, BN, W, MAX_STAGES, RED_STORE, ACTIVATION>(x, w, w2, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, N, K, num_experts, sorted_num, block_m, scaling_factor, activation_params, stream);
    }
    if constexpr (MAX_STAGES > 1) {
        if (stages == MAX_STAGES - 1)
            return launch_fused_moe_kernel_up_down_acc<BM, BN, W, MAX_STAGES - 1, RED_STORE, ACTIVATION>(x, w, w2, out, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, top_k, M, N, K, num_experts, sorted_num, block_m, scaling_factor, activation_params, stream);
    }
    return false;
}
// clang-format on

}  // namespace moe_sm100
