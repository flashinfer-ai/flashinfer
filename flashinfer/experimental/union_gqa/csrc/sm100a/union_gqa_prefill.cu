/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

/**
 * @file union_gqa_prefill.cu
 * @brief SM100 union-GQA prefill forward with two softmax warpgroups.
 *
 * The kernel uses two query stages:
 *   - WG0 softmax handles qstage 0: 8 query tokens x 16 heads = 128 M rows.
 *   - WG1 softmax handles qstage 1: 8 query tokens x 16 heads = 128 M rows.
 *   - One CTA therefore covers 16 query tokens = 256 logical M rows.
 *
 * KV tiles are N=64 and contain one 64-token sparse block. Each qstage owns
 * its metadata stream and K/V smem buffer, while TMEM keeps one S/P slot per
 * qstage plus one O accumulator per qstage.
 *
 * The two-warpgroup schedule overlaps the QK/PV pipeline with metadata and
 * TMA work. The softmax path interleaves MUFU and polynomial exponentials to
 * reduce special-function-unit pressure.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/bfloat16.h>
#include <math_constants.h>

#include <climits>
#include <cstdint>
#include <cute/arch/simd_sm100.hpp>
#include <cute/tensor.hpp>

#include "union_gqa_mma_noelect.cuh"
#include "union_gqa_prefill_api.cuh"
#include "union_gqa_utils.cuh"

#ifndef FLASHINFER_UNION_GQA_CAUSAL
#define FLASHINFER_UNION_GQA_CAUSAL 1
#endif

// Benchmark-only: measure a query suffix with a fully available KV history.
// The production build always starts at zero. Attention math is unchanged.
#ifndef FLASHINFER_UNION_GQA_BENCHMARK_Q_PREFIX
#define FLASHINFER_UNION_GQA_BENCHMARK_Q_PREFIX 0
#endif

namespace flashinfer::union_gqa {

struct UnionGQAPrefillParams {
  int s_q;
  int h_q;
  int h_kv;
  int d_qk;
  int d_v;
  int topk_blocks;
  int num_seq;
  int max_seqlen;
  float sm_scale;

  cutlass::bfloat16_t* q;
  cutlass::bfloat16_t* k;
  cutlass::bfloat16_t* v;
  int32_t* indices;
  int32_t* cu_seqlen;

  int stride_q_s_q;
  int stride_q_h_q;
  int stride_k_s_q;
  int stride_k_h_kv;
  int stride_v_s_q;
  int stride_v_h_kv;
  int stride_indices_s_q;
  int stride_indices_h_kv;

  cutlass::bfloat16_t* out;
  cudaStream_t stream;
};

namespace {

using namespace cute;

namespace kernel {

using bf16 = cutlass::bfloat16_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

static constexpr int D_QK = 128;
static constexpr int D_V = 128;
static constexpr int B_H = 16;
static constexpr int Q_STAGE_TOKENS = 8;
static constexpr int Q_STAGE = 2;
static constexpr int CTA_Q_TOKENS = Q_STAGE * Q_STAGE_TOKENS;
static constexpr int BENCHMARK_Q_PREFIX = FLASHINFER_UNION_GQA_BENCHMARK_Q_PREFIX;
static_assert(BENCHMARK_Q_PREFIX >= 0 && BENCHMARK_Q_PREFIX % CTA_Q_TOKENS == 0);
static constexpr int B_M = B_H * Q_STAGE_TOKENS;
// Public contract: every block index denotes exactly 64 consecutive KV tokens.
static constexpr int BLOCK_SIZE = 64;
static constexpr int B_N = BLOCK_SIZE;
static constexpr int SP_STAGE = 2;
static constexpr int MAX_NUM_KV_TILES = 128;
static constexpr int THREADS_PER_WARP = 32;
static constexpr int THREADS_PER_WARPGROUP = 128;
static constexpr int kThreads = 512;
static constexpr int BUFFER_ALIGN_BYTES = 1024;
static constexpr int WG_REG_SOFTMAX = 160;
static constexpr int WG_REG_CORRECTION = 96;
static constexpr int WG_REG_MMA_TMA = 96;

static_assert(B_N == BLOCK_SIZE);
static_assert(B_M == 128);

enum WG_SPEC_ROLE : int {
  WG_SPEC_ROLE_SOFTMAX0 = 0,
  WG_SPEC_ROLE_SOFTMAX1 = 1,
  WG_SPEC_ROLE_CORRECTION = 2,
  WG_SPEC_ROLE_MMA_TMA = 3,
};

struct tmem_cols {
  static constexpr int S00 = 0;
  static constexpr int S01 = 64;
  static constexpr int S10 = 128;
  static constexpr int S11 = 192;
  static constexpr int P00 = S00;
  static constexpr int P01 = S01;
  static constexpr int P10 = S10;
  static constexpr int P11 = S11;
  static constexpr int O0 = 256;
  static constexpr int O1 = 384;

  CUTE_HOST_DEVICE static constexpr int s(int qstage, int slot) {
    return qstage == 0 ? (slot == 0 ? S00 : S01) : (slot == 0 ? S10 : S11);
  }
  CUTE_HOST_DEVICE static constexpr int p(int qstage, int slot) {
    return qstage == 0 ? (slot == 0 ? P00 : P01) : (slot == 0 ? P10 : P11);
  }
  CUTE_HOST_DEVICE static constexpr int o(int stage) { return stage == 0 ? O0 : O1; }
};
static constexpr int TMEM_TOTAL_COLS = 512;

using SmemLayoutQ_SW128 = decltype(coalesce(
    tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<B_M>, Int<D_QK>>{}, Step<_1, _2>{}),
    Shape<_1, _1>{}));

using SmemLayoutQ_Copy = decltype(coalesce(
    tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{},
                  Shape<Int<B_H>, Int<D_QK>, Int<Q_STAGE_TOKENS>>{}, Step<_1, _3, _2>{}),
    Shape<_1, _1, _1>{}));

using SmemLayoutK_SW128 = decltype(coalesce(
    tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<B_N>, Int<D_QK>>{}, Step<_1, _2>{}),
    Shape<_1, _1>{}));

using SmemLayoutV_MN_SW128 = decltype(coalesce(
    tile_to_shape(UMMA::Layout_MN_SW128_Atom<bf16>{}, Shape<Int<D_V>, Int<B_N>>{}, Step<_2, _1>{}),
    Shape<_1, _1>{}));

using SmemLayoutOBuf =
    decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<B_M>, Int<D_V>>{}));

using SmemLayoutO_Copy = decltype(coalesce(
    tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{},
                  Shape<Int<B_H>, Int<D_V>, Int<Q_STAGE_TOKENS>>{}, Step<_1, _3, _2>{}),
    Shape<_1, _1, _1>{}));

using TiledMMA_QK = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_UnionGQA_SS_NOELECT<bf16, bf16, float, B_M, B_N, UMMA::Major::K,
                                          UMMA::Major::K>{}));

using TiledMMA_PV = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_UnionGQA_TS_NOELECT<bf16, bf16, float, B_M, D_V, UMMA::Major::K,
                                          UMMA::Major::MN>{}));

template <typename Shape_Q, typename TMA_Q, typename Shape_K, typename TMA_K, typename Shape_V,
          typename TMA_V, typename TmaO_t>
struct TmaParams {
  Shape_Q shape_Q;
  TMA_Q tma_Q;
  Shape_K shape_K;
  TMA_K tma_K;
  Shape_V shape_V;
  TMA_V tma_V;
  TmaO_t tma_O;
};

template <typename Shape_Q, typename TMA_Q, typename Shape_K, typename TMA_K, typename Shape_V,
          typename TMA_V, typename TmaO_t>
TmaParams(Shape_Q, TMA_Q, Shape_K, TMA_K, Shape_V, TMA_V, TmaO_t)
    -> TmaParams<Shape_Q, TMA_Q, Shape_K, TMA_K, Shape_V, TMA_V, TmaO_t>;

struct TmemTensorPlan {
  TiledMMA_QK tiled_mma_qk{};
  TiledMMA_PV tiled_mma_pv{};

  decltype(partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_M>, Int<B_N>>{})) tS0;
  decltype(partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_M>, Int<B_N>>{})) tS1;
  decltype(partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_M>, Int<B_N>>{})) tS2;
  decltype(partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_M>, Int<B_N>>{})) tS3;
  decltype(TiledMMA_PV{}.get_slice(_0{}).make_fragment_A(
      partition_shape_A(TiledMMA_PV{}, Shape<Int<B_M>, Int<B_N>>{}))) tP0;
  decltype(TiledMMA_PV{}.get_slice(_0{}).make_fragment_A(
      partition_shape_A(TiledMMA_PV{}, Shape<Int<B_M>, Int<B_N>>{}))) tP1;
  decltype(TiledMMA_PV{}.get_slice(_0{}).make_fragment_A(
      partition_shape_A(TiledMMA_PV{}, Shape<Int<B_M>, Int<B_N>>{}))) tP2;
  decltype(TiledMMA_PV{}.get_slice(_0{}).make_fragment_A(
      partition_shape_A(TiledMMA_PV{}, Shape<Int<B_M>, Int<B_N>>{}))) tP3;
  decltype(partition_fragment_C(TiledMMA_PV{}, Shape<Int<B_M>, Int<D_V>>{})) tO0;
  decltype(partition_fragment_C(TiledMMA_PV{}, Shape<Int<B_M>, Int<D_V>>{})) tO1;

  CUTE_DEVICE
  TmemTensorPlan()
      : tS0(partition_fragment_C(tiled_mma_qk, Shape<Int<B_M>, Int<B_N>>{})),
        tS1(partition_fragment_C(tiled_mma_qk, Shape<Int<B_M>, Int<B_N>>{})),
        tS2(partition_fragment_C(tiled_mma_qk, Shape<Int<B_M>, Int<B_N>>{})),
        tS3(partition_fragment_C(tiled_mma_qk, Shape<Int<B_M>, Int<B_N>>{})),
        tP0(tiled_mma_pv.get_slice(_0{}).make_fragment_A(
            partition_shape_A(tiled_mma_pv, Shape<Int<B_M>, Int<B_N>>{}))),
        tP1(tiled_mma_pv.get_slice(_0{}).make_fragment_A(
            partition_shape_A(tiled_mma_pv, Shape<Int<B_M>, Int<B_N>>{}))),
        tP2(tiled_mma_pv.get_slice(_0{}).make_fragment_A(
            partition_shape_A(tiled_mma_pv, Shape<Int<B_M>, Int<B_N>>{}))),
        tP3(tiled_mma_pv.get_slice(_0{}).make_fragment_A(
            partition_shape_A(tiled_mma_pv, Shape<Int<B_M>, Int<B_N>>{}))),
        tO0(partition_fragment_C(tiled_mma_pv, Shape<Int<B_M>, Int<D_V>>{})),
        tO1(partition_fragment_C(tiled_mma_pv, Shape<Int<B_M>, Int<D_V>>{})) {
    tS0.data().get() = tmem_cols::S00;
    tS1.data().get() = tmem_cols::S01;
    tS2.data().get() = tmem_cols::S10;
    tS3.data().get() = tmem_cols::S11;
    tP0.data().get() = tmem_cols::P00;
    tP1.data().get() = tmem_cols::P01;
    tP2.data().get() = tmem_cols::P10;
    tP3.data().get() = tmem_cols::P11;
    tO0.data().get() = tmem_cols::O0;
    tO1.data().get() = tmem_cols::O1;
  }

  CUTE_DEVICE auto& get_s(int qstage, int slot) {
    return qstage == 0 ? (slot == 0 ? tS0 : tS1) : (slot == 0 ? tS2 : tS3);
  }
  CUTE_DEVICE auto& get_p(int qstage, int slot) {
    return qstage == 0 ? (slot == 0 ? tP0 : tP1) : (slot == 0 ? tP2 : tP3);
  }
  CUTE_DEVICE auto& get_o(int stage) { return stage == 0 ? tO0 : tO1; }
};

static constexpr int SMEM_Q_ELEMS = cosize_v<SmemLayoutQ_SW128>;
static constexpr int SMEM_K_ELEMS = cosize_v<SmemLayoutK_SW128>;
static constexpr int SMEM_V_ELEMS = cosize_v<SmemLayoutV_MN_SW128>;
static constexpr int SMEM_O_ELEMS = cosize_v<SmemLayoutOBuf>;

static_assert(SMEM_Q_ELEMS == B_M * D_QK);
static_assert(SMEM_K_ELEMS >= B_N * D_QK);
static_assert(SMEM_V_ELEMS >= B_N * D_V);
static_assert(SMEM_Q_ELEMS == cosize_v<SmemLayoutQ_Copy>);
static_assert(SMEM_O_ELEMS == cosize_v<SmemLayoutO_Copy>);

struct SharedMemoryPlan {
  CUTE_ALIGNAS(BUFFER_ALIGN_BYTES) array_aligned<bf16, SMEM_Q_ELEMS> q[Q_STAGE];
  CUTE_ALIGNAS(BUFFER_ALIGN_BYTES) array_aligned<bf16, SMEM_O_ELEMS> o[Q_STAGE];
  CUTE_ALIGNAS(BUFFER_ALIGN_BYTES) array_aligned<bf16, SMEM_K_ELEMS> k[Q_STAGE];
  CUTE_ALIGNAS(BUFFER_ALIGN_BYTES) array_aligned<bf16, SMEM_V_ELEMS> v[Q_STAGE];

  CUTE_ALIGNAS(16) float o_scale[Q_STAGE][SP_STAGE][B_M];
  CUTE_ALIGNAS(16) float row_sum[Q_STAGE][B_M];
  CUTE_ALIGNAS(16) array_aligned<uint32_t, 1> tmem_start_addr;

  CUTE_ALIGNAS(16) int32_t num_union_tiles[Q_STAGE][2];
  CUTE_ALIGNAS(16) int32_t kv_indices[Q_STAGE][2][MAX_NUM_KV_TILES];
  CUTE_ALIGNAS(16) uint8_t kv_active_mask[Q_STAGE][2][MAX_NUM_KV_TILES];
  CUTE_ALIGNAS(16) uint8_t kv_valid_count[Q_STAGE][2][MAX_NUM_KV_TILES][Q_STAGE_TOKENS];

  transac_bar_t bar_metadata_ready[2];
  transac_bar_t bar_q_ready[Q_STAGE];
  transac_bar_t bar_q_free[Q_STAGE];
  transac_bar_t bar_softmax_ready[Q_STAGE][SP_STAGE];
  transac_bar_t bar_p_o_ready[Q_STAGE][SP_STAGE];
  transac_bar_t bar_correction_ready[Q_STAGE][SP_STAGE];
  transac_bar_t bar_o_acc_done[Q_STAGE];
  transac_bar_t bar_row_sum_ready[Q_STAGE];
  transac_bar_t bar_k_ready[Q_STAGE];
  transac_bar_t bar_k_free[Q_STAGE];
  transac_bar_t bar_v_ready[Q_STAGE];
  transac_bar_t bar_v_free[Q_STAGE];
  transac_bar_t bar_epilogue_ready[Q_STAGE];
  // A metadata wait for work i can occur after work i+1 finishes.
  // Keep those completions on separate barriers to avoid phase aliasing.
  transac_bar_t bar_epilogue_done[Q_STAGE][2];
  transac_bar_t bar_o_smem_free[Q_STAGE];

  CUTE_DEVICE auto get_q(int stage) {
    return make_tensor(make_smem_ptr(q[stage].data()), SmemLayoutQ_SW128{});
  }
  CUTE_DEVICE auto get_k(int qstage) {
    return make_tensor(make_smem_ptr(k[qstage].data()), SmemLayoutK_SW128{});
  }
  CUTE_DEVICE auto get_v(int qstage) {
    return make_tensor(make_smem_ptr(v[qstage].data()), SmemLayoutV_MN_SW128{});
  }
  CUTE_DEVICE auto get_o(int stage) {
    return make_tensor(make_smem_ptr(o[stage].data()), SmemLayoutOBuf{});
  }
};

static_assert(sizeof(SharedMemoryPlan) < 227 * 1024,
              "union-GQA SharedMemoryPlan exceeds SM100 smem budget");

}  // namespace kernel

using namespace kernel;
namespace nvutil = flashinfer::union_gqa::kernel_utils;

__device__ __forceinline__ void wait_previous_work_done(SharedMemoryPlan* plan, int work_iter) {
  if (work_iter > 0) {
    const int previous_work = work_iter - 1;
    const int slot = previous_work & 1;
    const int phase = (previous_work >> 1) & 1;
    plan->bar_epilogue_done[0][slot].wait(phase);
    plan->bar_epilogue_done[1][slot].wait(phase);
    asm volatile("tcgen05.fence::after_thread_sync;");
  }
}

__device__ __forceinline__ void wait_same_metadata_phase_done(SharedMemoryPlan* plan,
                                                              int work_iter) {
  if (work_iter >= 2) {
    const int previous_work = work_iter - 2;
    const int slot = previous_work & 1;
    const int phase = (previous_work >> 1) & 1;
    plan->bar_epilogue_done[0][slot].wait(phase);
    plan->bar_epilogue_done[1][slot].wait(phase);
    asm volatile("tcgen05.fence::after_thread_sync;");
  }
}

__device__ __forceinline__ int generate_union_metadata_warp(SharedMemoryPlan* plan,
                                                            const UnionGQAPrefillParams& params,
                                                            int offset_q, int cur_seq_len,
                                                            int q_pos_base, int h_kv_idx,
                                                            int input_kv_blocks, int meta_phase) {
  const int lane = threadIdx.x & (THREADS_PER_WARP - 1);
  const bool owns_query = lane < CTA_Q_TOKENS;
  const int q_in_cta = lane;
  const int qstage = owns_query ? (q_in_cta / Q_STAGE_TOKENS) : 0;
  const int query_in_stage = q_in_cta - qstage * Q_STAGE_TOKENS;
  const int q_local = q_pos_base + q_in_cta;
  const int q_global = offset_q + q_local;
  const int ind_base = q_global * params.stride_indices_s_q + h_kv_idx * params.stride_indices_h_kv;
  const uint32_t qstage_mask = ((1u << Q_STAGE_TOKENS) - 1u) << (qstage * Q_STAGE_TOKENS);
  const int qstage_leader = qstage * Q_STAGE_TOKENS;

  int cursor = 0;
  int out_tiles = 0;

  auto block_has_visible_token = [&](int blk) {
    if (blk < 0) {
      return false;
    }
#if FLASHINFER_UNION_GQA_CAUSAL
    const int last_valid_kv = (q_local < cur_seq_len - 1) ? q_local : (cur_seq_len - 1);
#else
    const int last_valid_kv = cur_seq_len - 1;
#endif
    // Compare in block space before multiplying so even a malformed large
    // int32 index cannot overflow into an in-range TMA address.
    return blk <= last_valid_kv / BLOCK_SIZE;
  };

  auto clear_metadata_tile = [&](int tile) {
    if (lane == qstage_leader) {
      plan->kv_indices[qstage][meta_phase][tile] = 0;
      plan->kv_active_mask[qstage][meta_phase][tile] = 0;
    }
    if (owns_query) {
      plan->kv_valid_count[qstage][meta_phase][tile][query_in_stage] = 0;
    }
    __syncwarp(qstage_mask);
  };

  if (owns_query) {
    while (out_tiles < MAX_NUM_KV_TILES) {
      int candidate = INT_MAX;
      if (q_local < cur_seq_len) {
        while (cursor < input_kv_blocks) {
          const int blk = params.indices[ind_base + cursor];
          if (block_has_visible_token(blk)) {
            candidate = blk;
            break;
          }
          ++cursor;
        }
      }

      const int min_blk = __reduce_min_sync(qstage_mask, candidate);
      if (min_blk == INT_MAX) {
        break;
      }

      const uint32_t active_mask32 = __ballot_sync(qstage_mask, candidate == min_blk);
      const uint32_t active_mask8 =
          (active_mask32 >> (qstage * Q_STAGE_TOKENS)) & ((1u << Q_STAGE_TOKENS) - 1u);

      if (candidate == min_blk) {
        ++cursor;
      }

      uint8_t valid_count = 0;
      if ((active_mask8 >> query_in_stage) & 1u) {
        const int kv_pos_base = min_blk * BLOCK_SIZE;
#if FLASHINFER_UNION_GQA_CAUSAL
        const int last_valid_kv = (q_local < cur_seq_len - 1) ? q_local : (cur_seq_len - 1);
#else
        const int last_valid_kv = cur_seq_len - 1;
#endif
        int count = last_valid_kv - kv_pos_base + 1;
        count = count < 0 ? 0 : count;
        count = count > B_N ? B_N : count;
        valid_count = static_cast<uint8_t>(count);
      }
      plan->kv_valid_count[qstage][meta_phase][out_tiles][query_in_stage] = valid_count;

      if (lane == qstage_leader) {
        plan->kv_indices[qstage][meta_phase][out_tiles] = min_blk;
        plan->kv_active_mask[qstage][meta_phase][out_tiles] = uint8_t(active_mask8);
      }
      __syncwarp(qstage_mask);
      ++out_tiles;
    }

    if (out_tiles == 0) {
      clear_metadata_tile(0);
      out_tiles = 1;
    }

    if (lane == qstage_leader) {
      plan->num_union_tiles[qstage][meta_phase] = out_tiles;
    }
  }

  __syncwarp();
  if (lane == 0) {
    cutlass::arch::fence_view_async_shared();
    plan->bar_metadata_ready[meta_phase].arrive();
  }
  return out_tiles;
}

template <class TmaParamsT>
__global__ __launch_bounds__(kThreads, 1) void union_gqa_prefill_kernel(
    __grid_constant__ const UnionGQAPrefillParams params,
    __grid_constant__ const TmaParamsT tma_params) {
  extern __shared__ char wksp_buf[];
  SharedMemoryPlan* plan = reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

  const int warpgroup_idx = threadIdx.x / THREADS_PER_WARPGROUP;
  const int warp_idx = threadIdx.x / THREADS_PER_WARP;
  const int lane_idx_in_warpgroup = threadIdx.x % THREADS_PER_WARPGROUP;

  const int max_q_tiles =
      (params.max_seqlen - BENCHMARK_Q_PREFIX + CTA_Q_TOKENS - 1) / CTA_Q_TOKENS;
  const int total_work = max_q_tiles * params.h_kv * params.num_seq;
  const int input_kv_blocks = params.topk_blocks;
  if (total_work == 0) return;

  struct WorkTile {
    int q_tile_idx;
    int h_kv_idx;
    int seq_idx;
    int offset_q;
    int cur_seq_len;
    int q_pos_base;
    bool valid;
  };

  auto get_work_tile = [&](int work_idx) {
    WorkTile work{};
    int linear_idx = work_idx;
    work.q_tile_idx = BENCHMARK_Q_PREFIX / CTA_Q_TOKENS + linear_idx % max_q_tiles;
    linear_idx /= max_q_tiles;
    work.h_kv_idx = linear_idx % params.h_kv;
    work.seq_idx = linear_idx / params.h_kv;
    const int seq_start = params.cu_seqlen[work.seq_idx];
    const int seq_end = params.cu_seqlen[work.seq_idx + 1];
    // CUDA-resident offsets cannot be checked on the host during graph capture.
    // Trap before any TMA load or output store can use an invalid address.
    if (params.cu_seqlen[0] != 0 || params.cu_seqlen[params.num_seq] != params.s_q ||
        seq_start < 0 || seq_end < seq_start || seq_end > params.s_q ||
        seq_end - seq_start > params.max_seqlen) {
      asm volatile("trap;");
    }
    work.offset_q = seq_start;
    work.cur_seq_len = seq_end - seq_start;
    work.q_pos_base = work.q_tile_idx * CTA_Q_TOKENS;
    work.valid = work.q_pos_base < work.cur_seq_len;
    return work;
  };

  auto qstage_valid = [&](const WorkTile& work, int qstage) {
    return work.q_pos_base + qstage * Q_STAGE_TOKENS < work.cur_seq_len;
  };

  auto item_phase = [&](int work_iter) { return work_iter & 1; };

  auto metadata_barrier_phase = [&](int work_iter) { return (work_iter >> 1) & 1; };

  auto update_dual_stage_and_phase = [&](int& stage, int& phase) {
    phase ^= stage;
    stage ^= 1;
  };

  if (threadIdx.x == 0) {
    plan->bar_metadata_ready[0].init(1);
    plan->bar_metadata_ready[1].init(1);
#pragma unroll
    for (int stage = 0; stage < Q_STAGE; ++stage) {
      plan->bar_q_ready[stage].init(1);
      plan->bar_q_free[stage].init(1);
#pragma unroll
      for (int slot = 0; slot < SP_STAGE; ++slot) {
        plan->bar_softmax_ready[stage][slot].init(1);
        plan->bar_p_o_ready[stage][slot].init(THREADS_PER_WARPGROUP * 2);
        plan->bar_correction_ready[stage][slot].init(THREADS_PER_WARPGROUP);
      }
      plan->bar_o_acc_done[stage].init(1);
      plan->bar_row_sum_ready[stage].init(THREADS_PER_WARPGROUP);
      plan->bar_epilogue_ready[stage].init(THREADS_PER_WARPGROUP);
      for (int slot = 0; slot < 2; ++slot) {
        plan->bar_epilogue_done[stage][slot].init(1);
      }
      plan->bar_o_smem_free[stage].init(1);
      plan->bar_q_free[stage].arrive();
      plan->bar_o_smem_free[stage].arrive();
    }
#pragma unroll
    for (int qstage = 0; qstage < Q_STAGE; ++qstage) {
      plan->bar_k_ready[qstage].init(1);
      plan->bar_k_free[qstage].init(1);
      plan->bar_v_ready[qstage].init(1);
      plan->bar_v_free[qstage].init(1);
      plan->bar_k_free[qstage].arrive();
      plan->bar_v_free[qstage].arrive();
    }
    cutlass::arch::fence_barrier_init();
  }

  if (warp_idx == 0) {
    cute::TMEM::Allocator1Sm().allocate(TMEM_TOTAL_COLS, plan->tmem_start_addr.data());
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  __syncthreads();

  const float softmax_scale = params.sm_scale;

  if (warpgroup_idx == WG_SPEC_ROLE_SOFTMAX0 || warpgroup_idx == WG_SPEC_ROLE_SOFTMAX1) {
    cutlass::arch::warpgroup_reg_alloc<WG_REG_SOFTMAX>();

    const int qstage = warpgroup_idx == WG_SPEC_ROLE_SOFTMAX0 ? 0 : 1;
    const int query_in_stage = lane_idx_in_warpgroup / B_H;
    const float softmax_scale_log2 = softmax_scale * 1.4426950408889634f;

    int sm_stage = 0;
    int sm_phase = 0;
    int work_iter = 0;
    for (int work_idx = int(blockIdx.x); work_idx < total_work; work_idx += int(gridDim.x)) {
      WorkTile work = get_work_tile(work_idx);
      if (!work.valid) continue;

      wait_previous_work_done(plan, work_iter);

      const int meta_phase = item_phase(work_iter);
      plan->bar_metadata_ready[meta_phase].wait(metadata_barrier_phase(work_iter));
      asm volatile("tcgen05.fence::after_thread_sync;");
      const int num_union_tiles = plan->num_union_tiles[qstage][meta_phase];

      float m_old = -CUDART_INF_F;
      float m = -CUDART_INF_F;
      float l = 0.0f;

      for (int kv_tile_idx = 0; kv_tile_idx < num_union_tiles; ++kv_tile_idx) {
        const int sm_slot = sm_stage;
        const int s_tmem_col = tmem_cols::s(qstage, sm_slot);
        const int p_tmem_col = tmem_cols::p(qstage, sm_slot);
        plan->bar_softmax_ready[qstage][sm_slot].wait(sm_phase);
        update_dual_stage_and_phase(sm_stage, sm_phase);
        asm volatile("tcgen05.fence::after_thread_sync;");

        const int valid_count =
            int(plan->kv_valid_count[qstage][meta_phase][kv_tile_idx][query_in_stage]);
        const bool row_empty = valid_count == 0;
        const uint32_t empty_mask = __ballot_sync(0xffffffffu, row_empty);
        const bool warp_all_empty = empty_mask == 0xffffffffu;

        float s_chunk[B_N];
        if (!warp_all_empty) {
          nvutil::tmem_load_32dp32b32x(s_tmem_col + 0, &s_chunk[0]);
          nvutil::tmem_load_32dp32b32x(s_tmem_col + 32, &s_chunk[32]);
          cutlass::arch::fence_view_async_tmem_load();
        }

        float o_scale = 1.0f;
        if (!row_empty) {
          float m_new;
          if (valid_count == B_N) {
            m_new = nvutil::row_max<B_N>(s_chunk, m_old);
          } else if (valid_count >= B_N / 2) {
#pragma unroll
            for (int j = B_N / 2; j < B_N; ++j) {
              if (j >= valid_count) {
                s_chunk[j] = -CUDART_INF_F;
              }
            }
            m_new = nvutil::row_max_range<0, B_N / 2>(s_chunk, m_old);
            m_new = nvutil::row_max_range<B_N / 2, B_N / 2>(s_chunk, m_new);
          } else {
#pragma unroll
            for (int j = 0; j < B_N / 2; ++j) {
              if (j >= valid_count) {
                s_chunk[j] = -CUDART_INF_F;
              }
            }
#pragma unroll
            for (int j = B_N / 2; j < B_N; ++j) {
              s_chunk[j] = -CUDART_INF_F;
            }
            m_new = nvutil::row_max_range<0, B_N / 2>(s_chunk, m_old);
          }

          const float acc_scale_log2 = (m_old - m_new) * softmax_scale_log2;
          if (kv_tile_idx == 0) {
            m = m_new;
          } else if (acc_scale_log2 >= -8.0f) {
            m = m_old;
          } else {
            m = m_new;
            o_scale = exp2f(acc_scale_log2);
            l *= o_scale;
          }
        }
        plan->o_scale[qstage][sm_slot][lane_idx_in_warpgroup] = o_scale;
        plan->bar_correction_ready[qstage][sm_slot].arrive();

        const float m_scaled_neg = -m * softmax_scale_log2;
        const float2 scale_pair = make_float2(softmax_scale_log2, softmax_scale_log2);
        const float2 m_neg_pair = make_float2(m_scaled_neg, m_scaled_neg);

        auto exp_pack_store_32 = [&](int s_start, int tmem_col) {
          uint32_t p_frag[16];
          if (row_empty) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
              p_frag[j] = 0u;
            }
          } else {
            float2 r_pair[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
              const int i = 2 * j;
              float2 s_raw = make_float2(s_chunk[s_start + i], s_chunk[s_start + i + 1]);
              float2 x_pair;
              cute::fma(x_pair, s_raw, scale_pair, m_neg_pair);
              if ((j & 1) == 0) {
                r_pair[j] = make_float2(exp2f(x_pair.x), exp2f(x_pair.y));
              } else {
                r_pair[j] = nvutil::ex2_emulation_3deg_pair(x_pair);
              }
            }

#pragma unroll
            for (int j = 0; j < 16; ++j) {
              const int i = 2 * j;
              s_chunk[s_start + i] = r_pair[j].x;
              s_chunk[s_start + i + 1] = r_pair[j].y;
              __nv_bfloat162 p01 = __float22bfloat162_rn(r_pair[j]);
              p_frag[j] = *reinterpret_cast<uint32_t const*>(&p01);
            }
          }
          nvutil::tmem_st_32dp32bNx<16>(tmem_col, p_frag);
        };

        exp_pack_store_32(0, p_tmem_col + 0);
        exp_pack_store_32(32, p_tmem_col + 16);
        cutlass::arch::fence_view_async_tmem_store();
        asm volatile("tcgen05.fence::before_thread_sync;");
        plan->bar_p_o_ready[qstage][sm_slot].arrive();

        if (!row_empty) {
          l = nvutil::row_sum<B_N>(s_chunk, l);
        }
        m_old = m;

        if (kv_tile_idx == num_union_tiles - 1) {
          plan->row_sum[qstage][lane_idx_in_warpgroup] = l;
          cutlass::arch::fence_view_async_shared();
          plan->bar_row_sum_ready[qstage].arrive();
        }
      }
      ++work_iter;
    }
  } else if (warpgroup_idx == WG_SPEC_ROLE_CORRECTION) {
    cutlass::arch::warpgroup_reg_dealloc<WG_REG_CORRECTION>();

    int corr_stage[Q_STAGE] = {0, 0};
    int corr_phase[Q_STAGE] = {0, 0};
    int pv_done_phase[Q_STAGE] = {0, 0};
    int work_iter = 0;
    for (int work_idx = int(blockIdx.x); work_idx < total_work; work_idx += int(gridDim.x)) {
      WorkTile work = get_work_tile(work_idx);
      if (!work.valid) continue;

      wait_previous_work_done(plan, work_iter);

      const int meta_phase = item_phase(work_iter);
      plan->bar_metadata_ready[meta_phase].wait(metadata_barrier_phase(work_iter));
      asm volatile("tcgen05.fence::after_thread_sync;");
      int num_union_tiles[Q_STAGE];
#pragma unroll
      for (int qstage = 0; qstage < Q_STAGE; ++qstage) {
        num_union_tiles[qstage] = plan->num_union_tiles[qstage][meta_phase];
      }
      const int max_union_tiles = max(num_union_tiles[0], num_union_tiles[1]);

#pragma unroll
      for (int qstage = 0; qstage < Q_STAGE; ++qstage) {
        const int corr_slot = corr_stage[qstage];
        plan->bar_correction_ready[qstage][corr_slot].wait(corr_phase[qstage]);
        plan->bar_p_o_ready[qstage][corr_slot].arrive();
        update_dual_stage_and_phase(corr_stage[qstage], corr_phase[qstage]);
      }

      for (int kv_tile_idx = 1; kv_tile_idx < max_union_tiles; ++kv_tile_idx) {
#pragma unroll
        for (int qstage = 0; qstage < Q_STAGE; ++qstage) {
          if (kv_tile_idx >= num_union_tiles[qstage]) {
            continue;
          }
          const int corr_slot = corr_stage[qstage];
          plan->bar_correction_ready[qstage][corr_slot].wait(corr_phase[qstage]);
          plan->bar_o_acc_done[qstage].wait(pv_done_phase[qstage]);
          pv_done_phase[qstage] ^= 1;
          asm volatile("tcgen05.fence::after_thread_sync;");

          const float scale_factor = plan->o_scale[qstage][corr_slot][lane_idx_in_warpgroup];
          const bool should_rescale = (__ballot_sync(0xffffffffu, scale_factor < 1.0f) != 0);
          if (should_rescale) {
            static constexpr int kCorrChunk = 16;
            float o_frag[kCorrChunk];
            const float2 scale_pair = make_float2(scale_factor, scale_factor);
            const int o_tmem_col = tmem_cols::o(qstage);
#pragma unroll
            for (int i = 0; i < D_V / kCorrChunk; ++i) {
              nvutil::tmem_ld_32dp32bNx<kCorrChunk>(o_tmem_col + i * kCorrChunk, o_frag);
              cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
              for (int j = 0; j < kCorrChunk; j += 2) {
                float2 o_pair = make_float2(o_frag[j], o_frag[j + 1]);
                o_pair = nvutil::mul_packed_f32x2(o_pair, scale_pair);
                o_frag[j] = o_pair.x;
                o_frag[j + 1] = o_pair.y;
              }
              nvutil::tmem_st_32dp32bNx<kCorrChunk>(o_tmem_col + i * kCorrChunk, o_frag);
            }
            cutlass::arch::fence_view_async_tmem_store();
          }
          asm volatile("tcgen05.fence::before_thread_sync;");
          plan->bar_p_o_ready[qstage][corr_slot].arrive();
          update_dual_stage_and_phase(corr_stage[qstage], corr_phase[qstage]);
        }
      }

#pragma unroll
      for (int qstage = 0; qstage < Q_STAGE; ++qstage) {
        plan->bar_o_acc_done[qstage].wait(pv_done_phase[qstage]);
        pv_done_phase[qstage] ^= 1;
        plan->bar_row_sum_ready[qstage].wait(work_iter & 1);
        asm volatile("tcgen05.fence::after_thread_sync;");

        const float rs = plan->row_sum[qstage][lane_idx_in_warpgroup];
        const float final_scale = rs == 0.0f ? 0.0f : 1.0f / rs;
        const int o_tmem_col = tmem_cols::o(qstage);
        auto sO = plan->get_o(qstage);
        plan->bar_o_smem_free[qstage].wait(work_iter & 1);
        asm volatile("tcgen05.fence::after_thread_sync;");

        static constexpr int kEpiChunk = 16;
        float o_frag[kEpiChunk];
        const int o_row = lane_idx_in_warpgroup;

#pragma unroll 1
        for (int i = 0; i < D_V / kEpiChunk; ++i) {
          nvutil::tmem_ld_32dp32bNx<kEpiChunk>(o_tmem_col + i * kEpiChunk, o_frag);
          cutlass::arch::fence_view_async_tmem_load();

#pragma unroll
          for (int j = 0; j < kEpiChunk; j += 8) {
            const int o_col = i * kEpiChunk + j;
            __nv_bfloat162 p0 = __float22bfloat162_rn(
                float2{o_frag[j + 0] * final_scale, o_frag[j + 1] * final_scale});
            __nv_bfloat162 p1 = __float22bfloat162_rn(
                float2{o_frag[j + 2] * final_scale, o_frag[j + 3] * final_scale});
            __nv_bfloat162 p2 = __float22bfloat162_rn(
                float2{o_frag[j + 4] * final_scale, o_frag[j + 5] * final_scale});
            __nv_bfloat162 p3 = __float22bfloat162_rn(
                float2{o_frag[j + 6] * final_scale, o_frag[j + 7] * final_scale});
            uint4 packed4{
                *reinterpret_cast<uint32_t const*>(&p0), *reinterpret_cast<uint32_t const*>(&p1),
                *reinterpret_cast<uint32_t const*>(&p2), *reinterpret_cast<uint32_t const*>(&p3)};
            *reinterpret_cast<uint4*>(&sO(o_row, o_col)) = packed4;
          }
        }
        cutlass::arch::fence_view_async_shared();
        asm volatile("tcgen05.fence::before_thread_sync;");
        plan->bar_epilogue_ready[qstage].arrive();
      }
      ++work_iter;
    }
  } else if (warpgroup_idx == WG_SPEC_ROLE_MMA_TMA) {
    cutlass::arch::warpgroup_reg_dealloc<WG_REG_MMA_TMA>();

    int k_ready_phase[Q_STAGE] = {0, 0};
    int v_ready_phase[Q_STAGE] = {0, 0};
    int k_tma_free_phase[Q_STAGE] = {0, 0};
    int v_tma_free_phase[Q_STAGE] = {0, 0};
    int qk_stage[Q_STAGE] = {0, 0};
    int p_phase[Q_STAGE] = {0, 0};
    int p_stage[Q_STAGE] = {0, 0};
    int work_iter = 0;

    for (int work_idx = int(blockIdx.x); work_idx < total_work; work_idx += int(gridDim.x)) {
      WorkTile work = get_work_tile(work_idx);
      if (!work.valid) continue;

      if (warp_idx == 12 && cute::elect_one_sync()) {
        TmemTensorPlan tmem;
        const int meta_phase = item_phase(work_iter);
        plan->bar_metadata_ready[meta_phase].wait(metadata_barrier_phase(work_iter));
        asm volatile("tcgen05.fence::after_thread_sync;");
        int num_union_tiles[Q_STAGE];
#pragma unroll
        for (int qstage = 0; qstage < Q_STAGE; ++qstage) {
          num_union_tiles[qstage] = plan->num_union_tiles[qstage][meta_phase];
        }
        const int max_union_tiles = max(num_union_tiles[0], num_union_tiles[1]);

        auto run_qk_for_stage = [&](int qstage, int sp_slot) {
          nvutil::umma_ss(tmem.tiled_mma_qk, plan->get_q(qstage), plan->get_k(qstage),
                          tmem.get_s(qstage, sp_slot), true);
        };

        auto run_pv_for_stage = [&](int qstage, bool clear_o) {
          const int cur_p_stage = p_stage[qstage];
          plan->bar_p_o_ready[qstage][cur_p_stage].wait(p_phase[qstage]);
          nvutil::umma_ts(tmem.tiled_mma_pv, tmem.get_p(qstage, cur_p_stage), plan->get_v(qstage),
                          tmem.get_o(qstage), clear_o);
          nvutil::umma_commit_arrive_noelect(plan->bar_o_acc_done[qstage]);
          update_dual_stage_and_phase(p_stage[qstage], p_phase[qstage]);
        };

        auto release_q_if_last_qk = [&](int qstage, int kv_tile_idx) {
          if (kv_tile_idx == num_union_tiles[qstage] - 1) {
            nvutil::umma_commit_arrive_noelect(plan->bar_q_free[qstage]);
          }
        };

        auto run_qk_tile = [&](int qstage, int kv_tile_idx) {
          const int cur_qk_stage = qk_stage[qstage];
          plan->bar_k_ready[qstage].wait(k_ready_phase[qstage]);
          asm volatile("tcgen05.fence::after_thread_sync;");
          run_qk_for_stage(qstage, cur_qk_stage);
          nvutil::umma_commit_arrive_noelect(plan->bar_softmax_ready[qstage][cur_qk_stage]);
          nvutil::umma_commit_arrive_noelect(plan->bar_k_free[qstage]);
          release_q_if_last_qk(qstage, kv_tile_idx);
          k_ready_phase[qstage] ^= 1;
          qk_stage[qstage] ^= 1;
        };

        auto run_pv_tile = [&](int qstage, int kv_tile_idx) {
          const bool clear_o = (kv_tile_idx == 0);
          plan->bar_v_ready[qstage].wait(v_ready_phase[qstage]);
          asm volatile("tcgen05.fence::after_thread_sync;");
          run_pv_for_stage(qstage, clear_o);
          nvutil::umma_commit_arrive_noelect(plan->bar_v_free[qstage]);
          v_ready_phase[qstage] ^= 1;
        };

// QK0 can run before the previous work item's epilogue/O-smem
// use completes because S/P are no longer used by that point.
#pragma unroll
        for (int qstage = 0; qstage < Q_STAGE; ++qstage) {
          plan->bar_q_ready[qstage].wait(item_phase(work_iter));
          asm volatile("tcgen05.fence::after_thread_sync;");
          if (num_union_tiles[qstage] > 0) {
            run_qk_tile(qstage, 0);
          }
        }

        wait_previous_work_done(plan, work_iter);

#pragma unroll
        for (int qstage = 0; qstage < Q_STAGE; ++qstage) {
          if (num_union_tiles[qstage] >= 2) {
            run_qk_tile(qstage, 1);
          }
        }

        for (int kv_tile_idx = 2; kv_tile_idx < max_union_tiles; ++kv_tile_idx) {
#pragma unroll
          for (int qstage = 0; qstage < Q_STAGE; ++qstage) {
            const int pv_tile_idx = kv_tile_idx - 2;
            if (pv_tile_idx < num_union_tiles[qstage]) {
              run_pv_tile(qstage, pv_tile_idx);
            }
            if (kv_tile_idx < num_union_tiles[qstage]) {
              run_qk_tile(qstage, kv_tile_idx);
            }
          }
        }

#pragma unroll
        for (int qstage = 0; qstage < Q_STAGE; ++qstage) {
          const int first_drain_tile = max_union_tiles > 1 ? max_union_tiles - 2 : 0;
          for (int pv_tile_idx = first_drain_tile; pv_tile_idx < num_union_tiles[qstage];
               ++pv_tile_idx) {
            run_pv_tile(qstage, pv_tile_idx);
          }
        }
      } else if (warp_idx == 13) {
        // V TMA + epilogue TMA store warp.
        if (cute::elect_one_sync()) {
          const int meta_phase = item_phase(work_iter);
          plan->bar_metadata_ready[meta_phase].wait(metadata_barrier_phase(work_iter));
          asm volatile("tcgen05.fence::after_thread_sync;");
          int num_union_tiles[Q_STAGE];
#pragma unroll
          for (int qstage = 0; qstage < Q_STAGE; ++qstage) {
            num_union_tiles[qstage] = plan->num_union_tiles[qstage][meta_phase];
          }
          const int max_union_tiles = max(num_union_tiles[0], num_union_tiles[1]);

          auto load_v_tile = [&](int qstage, int tile_idx) {
            plan->bar_v_free[qstage].wait(v_tma_free_phase[qstage]);

            auto mV = domain_offset(make_coord(_0{}, work.offset_q),
                                    tma_params.tma_V.get_tma_tensor(make_shape(
                                        params.d_v, params.s_q, params.h_kv))(_, _, work.h_kv_idx));
            const int blk_idx = plan->kv_indices[qstage][meta_phase][tile_idx];
            auto gV = flat_divide(mV, Tile<Int<D_V>, Int<B_N>>{})(_, _, _0{}, blk_idx);
            nvutil::tma_copy_g2s(
                tma_params.tma_V, gV,
                make_tensor(make_smem_ptr(plan->v[qstage].data()), SmemLayoutV_MN_SW128{}),
                plan->bar_v_ready[qstage], B_N * D_V * (uint32_t)sizeof(bf16),
                cute::TMA::CacheHintSm90::EVICT_LAST);
            v_tma_free_phase[qstage] ^= 1;
          };

          for (int tile_idx = 0; tile_idx < max_union_tiles; ++tile_idx) {
#pragma unroll
            for (int qstage = 0; qstage < Q_STAGE; ++qstage) {
              if (tile_idx < num_union_tiles[qstage]) {
                load_v_tile(qstage, tile_idx);
              }
            }
          }
        }

        bool any_tma_store = false;
#pragma unroll
        for (int qstage = 0; qstage < Q_STAGE; ++qstage) {
          plan->bar_epilogue_ready[qstage].wait(work_iter & 1);
          asm volatile("tcgen05.fence::after_thread_sync;");

          const int q_pos_base = work.q_pos_base + qstage * Q_STAGE_TOKENS;
          const int valid_tokens = min(Q_STAGE_TOKENS, max(0, work.cur_seq_len - q_pos_base));
          const bool full_stage = valid_tokens == Q_STAGE_TOKENS;
          any_tma_store |= full_stage;
          auto sO_copy = make_tensor(make_smem_ptr(plan->o[qstage].data()), SmemLayoutO_Copy{});
          if (full_stage && cute::elect_one_sync()) {
            plan->bar_epilogue_done[qstage][work_iter & 1].arrive();
            const int qstage_tile_idx = work.q_tile_idx * Q_STAGE + qstage;
            auto mO = domain_offset(
                make_coord(_0{}, _0{}, work.offset_q),
                tma_params.tma_O.get_tma_tensor(make_shape(params.h_q, params.d_v, params.s_q)));
            auto gO = flat_divide(mO, Tile<Int<B_H>, Int<D_V>, Int<Q_STAGE_TOKENS>>{})(
                _, _, _, work.h_kv_idx, _0{}, qstage_tile_idx);
            nvutil::tma_copy_s2g(tma_params.tma_O, sO_copy, gO);
            nvutil::tma_store_commit_group();
          } else if (!full_stage) {
            const int lane = threadIdx.x % THREADS_PER_WARP;
            if (lane == 0) {
              plan->bar_epilogue_done[qstage][work_iter & 1].arrive();
            }
            const int values_per_token = B_H * D_V;
            const int num_values = valid_tokens * values_per_token;
            for (int linear_idx = lane; linear_idx < num_values; linear_idx += THREADS_PER_WARP) {
              const int token = linear_idx / values_per_token;
              const int head_dim_idx = linear_idx - token * values_per_token;
              const int head = head_dim_idx / D_V;
              const int dim = head_dim_idx - head * D_V;
              const int global_token = work.offset_q + q_pos_base + token;
              const int global_head = work.h_kv_idx * B_H + head;
              const int64_t out_idx =
                  static_cast<int64_t>(global_token) * params.h_q * D_V + global_head * D_V + dim;
              params.out[out_idx] = sO_copy(head, dim, token);
            }
          }
          __syncwarp();
        }
        if (cute::elect_one_sync()) {
          if (any_tma_store) {
            nvutil::tma_store_wait_group_read<0>();
          }
#pragma unroll
          for (int qstage = 0; qstage < Q_STAGE; ++qstage) {
            plan->bar_o_smem_free[qstage].arrive();
          }
        }
      } else if (warp_idx == 14 && cute::elect_one_sync()) {
        // Q/K TMA warp.
        const int meta_phase = item_phase(work_iter);

#pragma unroll
        for (int qstage = 0; qstage < Q_STAGE; ++qstage) {
          if (!qstage_valid(work, qstage)) {
            plan->bar_q_ready[qstage].arrive();
            continue;
          }
          plan->bar_q_free[qstage].wait(item_phase(work_iter));
          const int qstage_tile_idx = work.q_tile_idx * Q_STAGE + qstage;
          auto mQ = domain_offset(
              make_coord(_0{}, _0{}, work.offset_q),
              tma_params.tma_Q.get_tma_tensor(make_shape(params.h_q, params.d_qk, params.s_q)));
          auto gQ = flat_divide(mQ, Tile<Int<B_H>, Int<D_QK>, Int<Q_STAGE_TOKENS>>{})(
              _, _, _, work.h_kv_idx, _0{}, qstage_tile_idx);
          auto sQ_copy = make_tensor(make_smem_ptr(plan->q[qstage].data()), SmemLayoutQ_Copy{});
          nvutil::tma_copy_g2s(tma_params.tma_Q, gQ, sQ_copy, plan->bar_q_ready[qstage],
                               B_H * D_QK * Q_STAGE_TOKENS * (uint32_t)sizeof(bf16),
                               cute::TMA::CacheHintSm90::EVICT_FIRST);
        }

        plan->bar_metadata_ready[meta_phase].wait(metadata_barrier_phase(work_iter));
        asm volatile("tcgen05.fence::after_thread_sync;");
        int num_union_tiles[Q_STAGE];
#pragma unroll
        for (int qstage = 0; qstage < Q_STAGE; ++qstage) {
          num_union_tiles[qstage] = plan->num_union_tiles[qstage][meta_phase];
        }
        const int max_union_tiles = max(num_union_tiles[0], num_union_tiles[1]);

        auto load_k_tile = [&](int qstage, int tile_idx) {
          plan->bar_k_free[qstage].wait(k_tma_free_phase[qstage]);

          auto mK = domain_offset(make_coord(work.offset_q, _0{}),
                                  tma_params.tma_K.get_tma_tensor(make_shape(
                                      params.s_q, params.d_qk, params.h_kv))(_, _, work.h_kv_idx));
          const int blk_idx = plan->kv_indices[qstage][meta_phase][tile_idx];
          auto gK = flat_divide(mK, Tile<Int<BLOCK_SIZE>, Int<D_QK>>{})(_, _, blk_idx, _0{});
          nvutil::tma_copy_g2s(
              tma_params.tma_K, gK,
              make_tensor(make_smem_ptr(plan->k[qstage].data()), SmemLayoutK_SW128{}),
              plan->bar_k_ready[qstage], B_N * D_QK * (uint32_t)sizeof(bf16),
              cute::TMA::CacheHintSm90::EVICT_LAST);
          k_tma_free_phase[qstage] ^= 1;
        };

#pragma unroll
        for (int qstage = 0; qstage < Q_STAGE; ++qstage) {
          if (num_union_tiles[qstage] > 0) {
            load_k_tile(qstage, 0);
          }
        }

        wait_previous_work_done(plan, work_iter);

        for (int tile_idx = 1; tile_idx < max_union_tiles; ++tile_idx) {
#pragma unroll
          for (int qstage = 0; qstage < Q_STAGE; ++qstage) {
            if (tile_idx < num_union_tiles[qstage]) {
              load_k_tile(qstage, tile_idx);
            }
          }
        }
      } else if (warp_idx == 15) {
        // Metadata warp.
        const int meta_phase = item_phase(work_iter);
        wait_same_metadata_phase_done(plan, work_iter);
        generate_union_metadata_warp(plan, params, work.offset_q, work.cur_seq_len, work.q_pos_base,
                                     work.h_kv_idx, input_kv_blocks, meta_phase);
        __syncwarp();
      }
      ++work_iter;
    }
  }

  __syncthreads();
  if (warp_idx == 0) {
    cute::TMEM::Allocator1Sm().free(0, TMEM_TOTAL_COLS);
  }
}

/**
 * @brief Build TMA descriptors and launch at most one persistent CTA per SM.
 * @return The CUDA setup or launch error, or cudaSuccess when there is no work.
 */
static cudaError_t launch_kernel(const UnionGQAPrefillParams& params) {
  using namespace cute;

  auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q);
  auto mQ_gmem = make_tensor(
      make_gmem_ptr(params.q),
      make_layout(shape_Q, make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)));
  auto tma_Q = cute::make_tma_copy(SM90_TMA_LOAD{}, mQ_gmem, SmemLayoutQ_Copy{});

  auto shape_K = make_shape(params.s_q, params.d_qk, params.h_kv);
  auto mK_gmem = make_tensor(
      make_gmem_ptr(params.k),
      make_layout(shape_K, make_stride(params.stride_k_s_q, _1{}, params.stride_k_h_kv)));
  auto tma_K = cute::make_tma_copy(SM90_TMA_LOAD{}, mK_gmem, SmemLayoutK_SW128{});

  auto shape_V = make_shape(params.d_v, params.s_q, params.h_kv);
  auto mV_gmem = make_tensor(
      make_gmem_ptr(params.v),
      make_layout(shape_V, make_stride(_1{}, params.stride_v_s_q, params.stride_v_h_kv)));
  auto tma_V = cute::make_tma_copy(SM90_TMA_LOAD{}, mV_gmem, SmemLayoutV_MN_SW128{});

  auto shape_O = make_shape(params.h_q, params.d_v, params.s_q);
  auto mO_gmem = make_tensor(
      make_gmem_ptr(params.out),
      make_layout(shape_O, make_stride(int(params.d_v), _1{}, int(params.h_q * params.d_v))));
  auto tma_O = cute::make_tma_copy(SM90_TMA_STORE{}, mO_gmem, SmemLayoutO_Copy{});

  auto tma_params = TmaParams{shape_Q, tma_Q, shape_K, tma_K, shape_V, tma_V, tma_O};

  const int max_q_tiles =
      (params.max_seqlen - BENCHMARK_Q_PREFIX + CTA_Q_TOKENS - 1) / CTA_Q_TOKENS;
  const int total_work = max_q_tiles * params.h_kv * params.num_seq;
  if (total_work == 0) return cudaSuccess;

  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  cudaDeviceProp prop{};
  status = cudaGetDeviceProperties(&prop, device);
  if (status != cudaSuccess) return status;
  int persistent_ctas =
      total_work < prop.multiProcessorCount ? total_work : prop.multiProcessorCount;
  dim3 grid(persistent_ctas);
  dim3 block(kThreads);
  constexpr size_t smem_bytes = sizeof(SharedMemoryPlan);

  auto kernel = &union_gqa_prefill_kernel<decltype(tma_params)>;
  status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  if (status != cudaSuccess) return status;
  kernel<<<grid, block, smem_bytes, params.stream>>>(params, tma_params);
  return cudaGetLastError();
}

}  // namespace

/**
 * @brief Pack validated contiguous BF16 inputs into the Union-GQA launch parameters.
 *
 * Block indices and cumulative sequence lengths reside on the device as int32 arrays.
 * The caller supplies SM100-compatible shapes, aligned pointers, and disjoint output.
 * @return The CUDA setup or launch status without synchronizing the supplied stream.
 */
cudaError_t LaunchUnionGQAPrefill(void* q, void* k, void* v, int32_t* block_indices,
                                  int32_t* cu_seqlens_q, void* out, int s_q, int h_q, int h_kv,
                                  int topk_blocks, int num_seq, int max_seqlen_q,
                                  float softmax_scale, cudaStream_t stream) {
  UnionGQAPrefillParams params{};
  params.s_q = s_q;
  params.h_q = h_q;
  params.h_kv = h_kv;
  params.d_qk = D_QK;
  params.d_v = D_V;
  params.topk_blocks = topk_blocks;
  params.num_seq = num_seq;
  params.max_seqlen = max_seqlen_q;
  params.sm_scale = softmax_scale;
  params.q = static_cast<cutlass::bfloat16_t*>(q);
  params.k = static_cast<cutlass::bfloat16_t*>(k);
  params.v = static_cast<cutlass::bfloat16_t*>(v);
  params.indices = block_indices;
  params.cu_seqlen = cu_seqlens_q;
  params.stride_q_s_q = D_QK * h_q;
  params.stride_q_h_q = D_QK;
  params.stride_k_s_q = D_QK * h_kv;
  params.stride_k_h_kv = D_QK;
  params.stride_v_s_q = D_V * h_kv;
  params.stride_v_h_kv = D_V;
  params.stride_indices_s_q = h_kv * topk_blocks;
  params.stride_indices_h_kv = topk_blocks;
  params.out = static_cast<cutlass::bfloat16_t*>(out);
  params.stream = stream;
  return launch_kernel(params);
}

}  // namespace flashinfer::union_gqa
