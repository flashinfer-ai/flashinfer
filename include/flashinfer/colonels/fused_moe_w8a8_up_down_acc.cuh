#pragma once

#include <cooperative_groups.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp8.h>
#include <stdio.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cuda/ptx>
#include <cuda/std/limits>

#include "fused_moe_helpers.cuh"
namespace cg = cooperative_groups;

template <int STAGES, int WN, int BM, int BK, int BN>
struct smem_up {
  alignas(1024) fp8 w[STAGES * WN * BK * BN];
  alignas(1024) fp8 x[STAGES * BK * BM];
};

template <int STAGES, int WN, int BM, int BK, int BN, bool DB_STORE>
struct smem_down {
  alignas(1024) fp8 w[STAGES * WN * BK * BN];
  alignas(1024) fp8 x[BM * WN * BN / 2];
  alignas(16) __nv_bfloat16 out[(DB_STORE ? 2 : 1) * BM * (BK * 2 + 8)];
};

template <int STAGES, int BM, int BK, int BN, int WN>
struct smem_persistent {
  alignas(8) uint64_t bar[2 * STAGES];
  float topk_scales[BM];
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
        scale_down_bytes(size_t(WN * BN / 2 / BK) * (K / BK) * sizeof(float)),
        persistent_bytes(PERSISTENT_END + scale_down_bytes),
        overlay((persistent_bytes + 1023) & ~size_t(1023)),
        up_bytes((UP_END + scale_x_bytes + scale_up_bytes + 1023) & ~size_t(1023)),
        down_bytes(sizeof(Down)),
        total(overlay + (up_bytes > down_bytes ? up_bytes : down_bytes)) {}
};

template <int BM, int BK, int BN, int WN, int STAGES, int PRODUCER_THREADS, bool DB_STORE>
__global__
__launch_bounds__(WN * 32 + PRODUCER_THREADS, 1, 1) void fused_moe_w8a8_wgmma_up_down_acc_kernel(
    const fp8* __restrict__ x, const float* __restrict__ x_scale,
    const __grid_constant__ CUtensorMap tensor_map_w, const float* __restrict__ w_scale,
    const __grid_constant__ CUtensorMap tensor_map_w2, const float* __restrict__ w2_scale,
    __nv_bfloat16* __restrict__ out, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids, const int* __restrict__ num_tokens_post_padded,
    const float* __restrict__ topk_weights, const int top_k, int M, int N, int K,
    float scaling_factor) {
  // Hopper kernels share a fatbin with Blackwell kernels.
#if IS_SM90A
  constexpr int CONSUMER_THREADS = WN * 32;
  constexpr int WARPGROUPS = WN / 4;
  const int32_t warpM = blockIdx.y + blockIdx.z * gridDim.y;
  if (warpM >= num_tokens_post_padded[0] / BM) return;
  const int exp_idx = expert_ids[warpM];

  // TODO should not be hardcoded
  constexpr int block_shape[2] = {128, 128};

  constexpr uint32_t S_BITS_UP = 3;
  constexpr uint32_t S_MODE_UP = 4 - S_BITS_UP;

  constexpr uint32_t S_BITS_DOWN = WN * BN >= 256 ? 3 : WN * BN == 128 ? 2 : 1;

  constexpr uint32_t S_MODE_DOWN = 4 - S_BITS_DOWN;

  const int K2 = N / 2;
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

  constexpr int TN2 = BN2 / (64 * WARPGROUPS);

  const int n_x_scales = K / 128;

  const int scale_rows_w_up = N / block_shape[1];
  const int scale_cols_w_up = K / block_shape[0];
  const int scale_rows_w_down = K / block_shape[1];
  const int scale_cols_w_down = (N / 2) / block_shape[0];
  constexpr int scale_cols_per_block_w_down = BK2 / block_shape[0];
  const int scales_down_to_load = scale_cols_per_block_w_down * scale_rows_w_down;

  constexpr int scales_per_stage_up = WN * BN / block_shape[0];
  constexpr int scale_rows_per_stage_down = BN2 / block_shape[0];

  extern __shared__ __align__(1024) uint8_t sh[];
  using Layout = smem_layout<STAGES, BM, BK, BN, WN, DB_STORE>;
  const Layout layout(K);
  const size_t OVERLAY_OFF = layout.overlay;
  auto& smem_p = *reinterpret_cast<smem_persistent<STAGES, BM, BK, BN, WN>*>(sh);
  uint64_t* bar = smem_p.bar;
  float* topk_scales = smem_p.topk_scales;
  float* scale_w_down = reinterpret_cast<float*>(sh + Layout::PERSISTENT_END);
  float* scale_x_up = reinterpret_cast<float*>(sh + OVERLAY_OFF + Layout::UP_END);
  float* scale_w_up =
      reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(scale_x_up) + layout.scale_x_bytes);

  smem_up<STAGES, WN, BM, BK, BN>& s =
      *reinterpret_cast<smem_up<STAGES, WN, BM, BK, BN>*>(sh + OVERLAY_OFF);

  if (threadIdx.x == 0) {
    for (int i = 0; i < STAGES; i++) {
      init_barrier(&bar[i], PRODUCER_THREADS + 1, 0);
      init_barrier(&bar[i + STAGES], CONSUMER_THREADS, 0);
    }
  }
  __syncthreads();
  auto consumer_sync = [&]() { asm volatile("bar.sync 0, %0;\n" ::"n"(CONSUMER_THREADS)); };

  const int n_stages_up = K / block_shape[0];
  const int n_stages_down = N2 / BN2;

  constexpr int TM = BM / 8;
  constexpr int TN = BN / 16;
  nv_bfloat162 f_acc[TN][TM][2];
  memset(f_acc, 0, sizeof(f_acc));

  // If we're running low on registers steal them from the producer
  // and not the consumer
  constexpr int PRODUCER_REGS = 24;
  constexpr int CONSUMER_REGS = 240;
  constexpr bool CHANGE_REG_ALLOC = BM > 88 || WN * BN >= 512;

  int p = 0;
  // PRODUCER
  if (is_producer) {
    // TODO does it really matter?
    if constexpr (CHANGE_REG_ALLOC) {
      reg_dealloc<PRODUCER_REGS>();
    }
    int producer_tid = threadIdx.x - CONSUMER_THREADS;
    constexpr int X_IT = const_ceil(XS / float(PRODUCER_THREADS * TO));
    uint32_t tsrc[X_IT];
    int i = (producer_tid)*TO;
    for (int r = 0; r < X_IT; r += 1) {
      if (i < XS) {
        int tdest = sorted_token_ids[warpM * BM + r * (PRODUCER_THREADS / 8) + producer_tid / 8];
        tsrc[r] = tdest / top_k;
        if (producer_tid % 8 == 0 && tsrc[r] < M && i < XS) {
          uint32_t smem =
              __cvta_generic_to_shared(&topk_scales[r * (PRODUCER_THREADS / 8) + producer_tid / 8]);
          CP_ASYNC_CG4(smem, topk_weights + tdest, 4);
        }
      }
      i += PRODUCER_THREADS * TO;
    }
    int smem_stage = 0;
    const int w_row_up = exp_idx * N + (blockIdx.x) * WN * BN;

    {
      // Scale rows with two trailing floats require scalar copies for alignment.
      const int scale_vector = scale_cols_w_up % 4 == 0 ? 4 : 1;
      for (int col = producer_tid * scale_vector; col < scale_cols_w_up;
           col += PRODUCER_THREADS * scale_vector) {
        for (int scale_row = 0; scale_row < scales_per_stage_up; scale_row++) {
          uint32_t smem = __cvta_generic_to_shared(scale_w_up + scale_row * scale_cols_w_up + col);
          const float* src = w_scale + exp_idx * scale_rows_w_up * scale_cols_w_up +
                             (blockIdx.x * scales_per_stage_up + scale_row) * scale_cols_w_up + col;
          if (scale_vector == 4) {
            CP_ASYNC_CG(smem, src, 16);
          } else {
            CP_ASYNC_CG4(smem, src, 4);
          }
        }
      }
    }
    {
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
    }
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
          constexpr int TILE_COLS_UP = BN * WN / TILES_UP;
          expect_bytes(bar + smem_stage, WS * sizeof(fp8));
          load_async_3d(s.w + smem_stage * WS, &tensor_map_w, bar + smem_stage, off, 0,
                        w_row_up / TILE_COLS_UP);
        });
      }
      smem_stage++;
    }

    smem_down<STAGES, WN, BM, BK, BN, DB_STORE>& s_d =
        *reinterpret_cast<smem_down<STAGES, WN, BM, BK, BN, DB_STORE>*>(sh + OVERLAY_OFF);

    const int base =
        exp_idx * scale_rows_w_down * scale_cols_w_down + blockIdx.x * scale_cols_per_block_w_down;
    for (int idx = producer_tid; idx < scales_down_to_load; idx += PRODUCER_THREADS) {
      uint32_t smem = __cvta_generic_to_shared(scale_w_down + idx);
      int row = idx % scale_rows_w_down;
      int col = idx / scale_rows_w_down;
      CP_ASYNC_CG4(smem, &w2_scale[base + row * scale_cols_w_down + col], 4);
    }

    for (int load_stage = 0; load_stage < n_stages_down; load_stage++) {
      if (smem_stage == STAGES) {
        p ^= 1;
        smem_stage = 0;
      }
      int col = (load_stage + exp_idx) % n_stages_down;
      wait(bar + STAGES + smem_stage, p);
      if (producer_tid == 0) {
        cg::invoke_one(cg::coalesced_threads(), [&]() {
          const int w_row_down = blockIdx.x * BK2;
          const int w_col_down = exp_idx * N2 + col * BN2;
          constexpr int TILE_INNER_DN = BK2 / TILES_DOWN;
          expect_bytes(bar + smem_stage, WS * sizeof(fp8));
          load_async_3d(s_d.w + smem_stage * WS, &tensor_map_w2, bar + smem_stage, 0, w_col_down,
                        w_row_down / TILE_INNER_DN);
        });
      }
      cp_async_mbarrier_arrive(bar + smem_stage);
      smem_stage++;
    }
  }
  // CONSUMER
  else {
    if constexpr (CHANGE_REG_ALLOC) {
      reg_alloc<CONSUMER_REGS>();
    }
    // Empty barriers arrive instantly
    constexpr int SAVES_PER_WARP = BM / WN;
    int token_src = M;
    if (lane_id < SAVES_PER_WARP)
      token_src = sorted_token_ids[warpM * BM + warp_id * SAVES_PER_WARP + lane_id] / top_k;
    for (int i = 0; i < STAGES; i++) arrive(&bar[STAGES + i]);

    int smem_stage = 0;
    for (int compute_stage = 0; compute_stage < n_stages_up; compute_stage += 1) {
      if (smem_stage == STAGES) {
        p ^= 1;
        smem_stage = 0;
      }
      wait(bar + smem_stage, p);

      float scale_x[TPT];
      for (int t = 0; t < TPT; t += 2) {
        int token_idx = (t / 2) * 8 + (lane_id % 4) * 2;
        float2 x_sc = *reinterpret_cast<float2*>(scale_x_up + compute_stage * BM + token_idx);
        scale_x[t] = x_sc.x;
        scale_x[t + 1] = x_sc.y;
      }
      const int scale_off = ((warp_id / 4) * (BN * 4) / 256) * 2 * K / 128;
      float scale_w[2];
      scale_w[0] = scale_w_up[scale_off + compute_stage];
      scale_w[1] = scale_w_up[scale_off + K / 128 + compute_stage];

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
                __hadd2(f_acc[tn + tni][tm][0],
                        __nv_bfloat162(scale_w[0] * scale_x[tm * 2] * tile_acc[tni][tm][0],
                                       scale_w[0] * scale_x[tm * 2 + 1] * tile_acc[tni][tm][1]));

            f_acc[tn + tni][tm][1] =
                __hadd2(f_acc[tn + tni][tm][1],
                        __nv_bfloat162(scale_w[1] * scale_x[tm * 2] * tile_acc[tni][tm][2],
                                       scale_w[1] * scale_x[tm * 2 + 1] * tile_acc[tni][tm][3]));
          }
        }
      }
      smem_stage++;
    }
    consumer_sync();
    smem_down<STAGES, WN, BM, BK, BN, DB_STORE>& s_d =
        *reinterpret_cast<smem_down<STAGES, WN, BM, BK, BN, DB_STORE>*>(sh + OVERLAY_OFF);
    float4* block_max = reinterpret_cast<float4*>(s_d.out);
    constexpr float EPS = 1e-10;
    nv_bfloat162 token_max[TM] = {nv_bfloat162(EPS, EPS)};
    for (int tn = 0; tn < TN; tn++) {
      for (int tm = 0; tm < TM; tm++) {
        f_acc[tn][tm][0].x = swiglu_mul(f_acc[tn][tm][0].x, f_acc[tn][tm][1].x);
        f_acc[tn][tm][0].y = swiglu_mul(f_acc[tn][tm][0].y, f_acc[tn][tm][1].y);
        nv_bfloat162 abs = __habs2(f_acc[tn][tm][0]);
        token_max[tm] = __hmax2(abs, token_max[tm]);
      }
    }
    for (int tm = 0; tm < TM; tm++) {
      token_max[tm] = __hmax2(__shfl_xor_sync(0xFFFFFFFF, token_max[tm], 16), token_max[tm]);
      token_max[tm] = __hmax2(__shfl_xor_sync(0xFFFFFFFF, token_max[tm], 8), token_max[tm]);
      token_max[tm] = __hmax2(__shfl_xor_sync(0xFFFFFFFF, token_max[tm], 4), token_max[tm]);
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
    float token_scale[QUANTS_TOTAL][TM][2];

    constexpr int warpgroups_to_cover_q = block_shape[0] / (TN * 32);
    for (int tm = 0; tm < TM; tm++) {
      int off = tm * 8 + (lane_id % 4) * 2;
      // Each wg computes ALL quant group scales by reading
      // block_max entries from every wg (not just its own).
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
        token_scale[quant][tm][0] = float(qmax.x) * fp8_max_inv;
        token_scale[quant][tm][1] = float(qmax.y) * fp8_max_inv;
      }
      for (int t = 0; t < 2; t++) {
        for (int tn = 0; tn < TN; tn++) {
          float val = t == 0 ? f_acc[tn][tm][0].x : f_acc[tn][tm][0].y;
          float scale;
          if constexpr (QUANTS_TOTAL == 1) {
            // Single quant group: all warp groups share the same scale
            scale = token_scale[0][tm][t];
          } else {
            // Yeah this shoud be wg_id*QUANTS_PER_WG but compiler is not smart
            // and it cannot assume that wg_id is in [0,1] and spills to local mem
            if (wg_id == 0)
              scale = token_scale[0 * QUANTS_PER_WG][tm][t];
            else if (wg_id == 1)
              scale = token_scale[1 * QUANTS_PER_WG][tm][t];
          }
          float q = val / scale;
          int x_row = tm * 8 + (lane_id % 4) * 2 + t;
          int x_col_local = tn * 32 + (warp_id % 4) * 8 + lane_id / 4;
          // If we have more than one quant total, next one should be in the next "block"
          // TODO explain better
          int wg_offset = QUANTS_TOTAL > 1 ? wg_id * QUANTS_PER_WG * BM * block_shape[0]
                                           : (warp_id / 4) * (TN * 32);
          int i = x_row * block_shape[0] + x_col_local;
          int swizzled = swizzle<S_BITS_DOWN>(wg_offset + i);
          s_d.x[swizzled] = fp8(q);
        }
      }
    }

    for (int t = 0; t < TPT; t += 2) {
      int token_idx = (t / 2) * 8 + (lane_id % 4) * 2;
      const float2 topk_w = *reinterpret_cast<const float2*>(&topk_scales[token_idx]);
      for (int quant = 0; quant < QUANTS_TOTAL; quant++) {
        token_scale[quant][t / 2][0] *= topk_w.x * scaling_factor;
        token_scale[quant][t / 2][1] *= topk_w.y * scaling_factor;
      }
    }

    consumer_sync();

    for (int compute_stage = 0; compute_stage < n_stages_down; compute_stage += 1) {
      if (smem_stage == STAGES) {
        p ^= 1;
        smem_stage = 0;
      }
      int col = (compute_stage + exp_idx) % n_stages_down;
      float s_w[TN2 / 2];

      constexpr int TN_I = BM < 32 ? TN2 : 1;
      float tile_acc[TN_I][TM][4];
      __nv_bfloat16 out_tile[TN2 / 2][TM][8];
      wait(bar + smem_stage, p);
      constexpr int PAD = BN2 + 8;
#pragma unroll
      for (int quant = 0; quant < QUANTS_TOTAL; quant++) {
        // When QUANTS_TOTAL > 1, quant groups are stored in separate
        // contiguous BM*block_shape[0] regions (each with its own swizzle).
        fp8* sx = s_d.x + quant * BM * block_shape[0];
        for (int i = 0; i < (TN2 / 2); i++) {
          s_w[i] = scale_w_down[quant * scale_rows_w_down + col * scale_rows_per_stage_down +
                                (warp_id / 4) * (TN2 / 2) + i];
        }
        for (int tn2 = 0; tn2 < TN2; tn2 += TN_I) {
          // In theory this should not be necessary because
          // we're using WGMMA with an accumulator scale of 0
          // however this produces an erroneous SASS with a
          // presumably unconfigured WGMMA instruction
          // current belief is that we're triggering some PTXAS bug
          memset(tile_acc, 0, sizeof(tile_acc));
          for (int tni = 0; tni < TN_I; tni++) {
            for (int tm = 0; tm < TM; tm++) {
              for (int t = 0; t < 4; t++) asm volatile("" : "+f"(tile_acc[tni][tm][t])::"memory");
            }
          }
          cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
          warpgroup_arrive();

          for (int tni = 0; tni < TN_I; tni++) {
            fp8* sw = s_d.w + smem_stage * WS + quant * WS / TILES_DOWN +
                      ((warp_id / 4) * TN2 + tn2 + tni) * 64 * (BK2 / TILES_DOWN);
            uint64_t desc_w = make_smem_descriptor<BK2 / (2 * TILES_DOWN), 1, S_MODE_DOWN>(sw);
            uint64_t desc_x = make_smem_descriptor<block_shape[0] / 2, 1, S_MODE_DOWN>(sx);
            wgmma<0, 1, 1, BM>(tile_acc[tni], desc_w, desc_x);
            desc_w += (32 >> 4);
            desc_x += (32 >> 4);
            for (int tk = 1; tk < 4; tk++) {
              wgmma<1, 1, 1, BM>(tile_acc[tni], desc_w, desc_x);
              desc_w += (32 >> 4);
              desc_x += (32 >> 4);
            }
          }
          warpgroup_commit_batch();
          warpgroup_wait();
          if (quant == QUANTS_TOTAL - 1 && tn2 == TN2 - TN_I) arrive(bar + STAGES + smem_stage);

          for (int tni = 0; tni < TN_I; tni += 2) {
            int tne = tn2 + tni;
            for (int tm = 0; tm < TM; tm++) {
              for (int t = 0; t < (TN_I >= 2 ? 8 : 4); t++) {
                int s = t % 2;
                if (quant == 0) {
                  out_tile[tne / 2][tm][(tne % 2) * 4 + t] =
                      token_scale[quant][tm][s] * tile_acc[tni + t / 4][tm][t % 4] * s_w[tne / 2];
                } else {
                  out_tile[tne / 2][tm][(tne % 2) * 4 + t] +=
                      token_scale[quant][tm][s] * tile_acc[tni + t / 4][tm][t % 4] * s_w[tne / 2];
                }
              }
            }
          }
        }
      }
      int stage_off = DB_STORE ? (compute_stage % 2) * BM * PAD : 0;
      if constexpr (DB_STORE) {
        asm volatile("cp.async.bulk.wait_group 1;");
      } else {
        asm volatile("cp.async.bulk.wait_group 0;");
      }

      consumer_sync();
      for (int tn2 = 0; tn2 < TN2; tn2 += 2) {
        for (int tm = 0; tm < TM; tm++) {
          int out_row = tm * 8 + lane_id % 8;
          int out_col = (warp_id / 4) * TN2 * 64 + (warp_id % 4) * 16 + (lane_id & 8) + tn2 * 64 +
                        (lane_id / 16) * 64;
          st_matrix_x4_trans(
              reinterpret_cast<uint32_t*>(out_tile[tn2 / 2][tm]),
              __cvta_generic_to_shared(s_d.out + stage_off + out_row * PAD + out_col));
        }
      }
      cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
      consumer_sync();
      if (lane_id < SAVES_PER_WARP) {
        if (token_src < M) {
          int row = warp_id * SAVES_PER_WARP + lane_id;
          cuda::ptx::cp_reduce_async_bulk(cuda::ptx::space_global, cuda::ptx::space_shared,
                                          cuda::ptx::op_add, out + token_src * N2 + col * BN2,
                                          s_d.out + stage_off + row * PAD,
                                          BN2 * sizeof(__nv_bfloat16));
        }
        cuda::ptx::cp_async_bulk_commit_group();
      }
      smem_stage++;
    }
  }
#else
  assert(false && "This fused MoE kernel requires SM90");
#endif
}

template <int BM, int BN, int WN, int STAGES, bool DB_STORE>
bool launch_fused_moe_kernel_up_down_acc_impl(
    const fp8* x, const float* x_scale, fp8* w, const float* w_scale, fp8* w2,
    const float* w2_scale, __nv_bfloat16* out, const int* sorted_token_ids, const int* expert_ids,
    const int* num_tokens_post_padded, const float* topk_weights, const int top_k, int M, int N,
    int K, int num_experts, int sorted_num, int block_m, float scaling_factor,
    cudaStream_t stream) {
  constexpr int BK = 128;
  constexpr int PRODUCER_THREADS = 128;
  if (K <= 0 || K % (2 * BK) || N <= 0 || N % 256 || N % (BN * WN)) return false;
  dim3 dimBlock(32 * WN + PRODUCER_THREADS, 1, 1);
  // Split token tiles across Y/Z to respect CUDA's 65535-block limit per axis.
  const int token_tiles = sorted_num / block_m + (sorted_num % block_m != 0);
  const int grid_z = (token_tiles + 65534) / 65535;
  const int grid_y = (token_tiles + grid_z - 1) / grid_z;
  dim3 dimGrid(N / (BN * WN), grid_y, grid_z);

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
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&max_smem_real, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    assert(max_smem_real == SM90_MAX_SMEM);

    gpuErrchk(cudaFuncSetAttribute(
        fused_moe_w8a8_wgmma_up_down_acc_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS, DB_STORE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

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

    fused_moe_w8a8_wgmma_up_down_acc_kernel<BM, BK, BN, WN, STAGES, PRODUCER_THREADS, DB_STORE>
        <<<dimGrid, dimBlock, sMemSize, stream>>>(
            x, x_scale, tensor_map_w, w_scale, tensor_map_w2, w2_scale, out, sorted_token_ids,
            expert_ids, num_tokens_post_padded, topk_weights, top_k, M, N, K, scaling_factor);
    gpuErrchk(cudaGetLastError());
    return true;
  }
  return false;
}

template <int BM, int BN, int WN, int STAGES>
bool launch_fused_moe_kernel_up_down_acc(const fp8* x, const float* x_scale, fp8* w,
                                         const float* w_scale, fp8* w2, const float* w2_scale,
                                         __nv_bfloat16* out, const int* sorted_token_ids,
                                         const int* expert_ids, const int* num_tokens_post_padded,
                                         const float* topk_weights, const int top_k, int M, int N,
                                         int K, int num_experts, int sorted_num, int block_m,
                                         float scaling_factor, cudaStream_t stream) {
  if (K <= 0 || K % 256 || N <= 0 || N % 256 || N % (BN * WN)) return false;
  const smem_layout<STAGES, BM, 128, BN, WN, true> layout(K);
  auto launch = [&]<bool DB_STORE>() {
    return launch_fused_moe_kernel_up_down_acc_impl<BM, BN, WN, STAGES, DB_STORE>(
        x, x_scale, w, w_scale, w2, w2_scale, out, sorted_token_ids, expert_ids,
        num_tokens_post_padded, topk_weights, top_k, M, N, K, num_experts, sorted_num, block_m,
        scaling_factor, stream);
  };
  if (layout.total <= SM90_MAX_SMEM) return launch.template operator()<true>();
  return launch.template operator()<false>();
}
