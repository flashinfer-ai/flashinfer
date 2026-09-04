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
#ifndef FLASHINFER_ATTENTION_HOPPER_DEQUANT_MAINLOOP_CUH_
#define FLASHINFER_ATTENTION_HOPPER_DEQUANT_MAINLOOP_CUH_

// Producer-side mainloops for a 16-bit query attending to an FP8 KV cache.
//
// The MMA warpgroups run the unmodified 16-bit consumer (mainloop_mma.cuh): they wait on the
// 16-bit K/V pipelines and read smem_k / smem_v exactly as for a 16-bit KV cache. The producer
// warpgroup fills those buffers in two steps:
//
//   1. K/V tiles are loaded as FP8 into small staging buffers (TMA for dense K/V, cp.async gather
//      for paged K/V), gated by the staging pipelines.
//   2. All 128 producer threads dequantize a staged tile into the 16-bit swizzled K/V layout
//      (exact conversion, the same values the FA2 mixed-precision kernel computes), publish it
//      on the 16-bit pipeline and release the staging stage.
//
// Because the staging buffers are separate from smem_k / smem_v, staging loads run ahead of the
// dequantization by NUM_STAGES_KV_STAGING tiles, and the V loads no longer wait for the epilogue
// (only the 16-bit V writes do, since smem_v aliases smem_o).

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <type_traits>

#include "../../fastdiv.cuh"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "named_barrier.cuh"
#include "utils.cuh"

namespace flashinfer {

using namespace cute;

// Empty stand-in for the staging pipelines of the 16-bit KV mainloops, so that the kernel can
// declare the staging state unconditionally.
struct NoStagingPipelines {
  template <typename SharedStorage>
  CUTLASS_DEVICE NoStagingPipelines(SharedStorage&, int) {}
};

template <typename CollectiveMainloop, bool KV_DEQUANT>
struct StagingPipelinesFor {
  using type = NoStagingPipelines;
};
template <typename CollectiveMainloop>
struct StagingPipelinesFor<CollectiveMainloop, true> {
  using type = typename CollectiveMainloop::StagingPipelines;
};

// Exact FP8 -> 16-bit conversion of 16 packed elements (four 32-bit words holding four FP8
// values each). Every finite FP8 value is representable in both fp16 and bf16, so all variants
// produce the same values as vec_cast (and hence the FA2 mixed-precision kernels); the bit
// tricks are the ones from fast_dequant_f8f16x4 with the byte placement folded into one PRMT.
template <typename DTypeKV, typename DTypeKVMma>
struct Fp8x16Dequantizer {
  static constexpr bool kE5M2 = std::is_same_v<DTypeKV, cutlass::float_e5m2_t>;
  static constexpr bool kHalf = std::is_same_v<DTypeKVMma, cutlass::half_t>;
  static_assert(kE5M2 || std::is_same_v<DTypeKV, cutlass::float_e4m3_t>);
  static_assert(kHalf || std::is_same_v<DTypeKVMma, cutlass::bfloat16_t>);

  // Elements keep their order: out[2*i] holds elements 0 and 1 of in[i] (low and high half),
  // out[2*i+1] holds elements 2 and 3.
  CUTLASS_DEVICE static void convert(const uint32_t (&in)[4], uint32_t (&out)[8]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t q = in[i];
      if constexpr (kE5M2 && kHalf) {
        // e5m2 is the top byte of an fp16: place the bytes and zero the low bytes.
        out[2 * i] = __byte_perm(q, 0, 0x1404);
        out[2 * i + 1] = __byte_perm(q, 0, 0x3424);
      } else if constexpr (!kE5M2 && kHalf) {
        // Hardware e4m3x2 -> f16x2 conversion (sm_89+); the low byte lands in the low half.
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;"
            : "=r"(out[2 * i])
            : "h"(static_cast<uint16_t>(q & 0xFFFFu)));
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;"
            : "=r"(out[2 * i + 1])
            : "h"(static_cast<uint16_t>(q >> 16)));
      } else {
        // bf16 target: keep the sign, move the 7 exponent/mantissa bits of the FP8 value
        // (at bits [14:8] after the byte placement) down to the bf16 exponent/mantissa fields
        // and fix up the exponent bias with one multiply, which also handles subnormals.
        constexpr int FP8_EXPONENT = kE5M2 ? 5 : 4;
        constexpr int BIAS_OFFSET = (1 << (8 - 1)) - (1 << (FP8_EXPONENT - 1));
        constexpr uint32_t BIAS_BITS = static_cast<uint32_t>(BIAS_OFFSET + 127)
                                       << 7;  // bf16 2^BIAS
        constexpr uint32_t BIAS2 = BIAS_BITS | (BIAS_BITS << 16);
        constexpr int SHIFT = 8 - FP8_EXPONENT;
        constexpr uint32_t PAYLOAD = (0x7F00u >> SHIFT) * 0x10001u;
        const uint32_t x01 = __byte_perm(q, 0, 0x1404);  // bytes 0, 1 -> high bytes of the halves
        const uint32_t x23 = __byte_perm(q, 0, 0x3424);  // bytes 2, 3
        const uint32_t y01 = (x01 & 0x80008000u) | ((x01 >> SHIFT) & PAYLOAD);
        const uint32_t y23 = (x23 & 0x80008000u) | ((x23 >> SHIFT) & PAYLOAD);
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(out[2 * i]) : "r"(y01), "r"(BIAS2));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(out[2 * i + 1]) : "r"(y23), "r"(BIAS2));
      }
    }
  }
};

// The part of the producer schedule shared by the dense and paged FP8-KV mainloops: the
// dequantization of staged tiles and the order in which tiles are staged and published.
template <typename Ktraits>
struct DequantKVSchedule {
  using DTypeKV = typename Ktraits::DTypeKV;
  using DTypeKVMma = typename Ktraits::DTypeKVMma;
  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using PipelineState = typename Ktraits::PipelineState;
  using StagingPipeline = typename Ktraits::StagingPipeline;
  using StagingPipelineState = typename Ktraits::StagingPipelineState;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SmemLayoutKStaging = typename Ktraits::SmemLayoutKStaging;
  using SmemLayoutVStaging = typename Ktraits::SmemLayoutVStaging;

  static constexpr int NUM_STAGES_KV_STAGING = Ktraits::NUM_STAGES_KV_STAGING;
  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  static constexpr int NUM_PRODUCER_THREADS = Ktraits::NUM_PRODUCER_THREADS;
  static constexpr int HEAD_DIM_QK = Ktraits::HEAD_DIM_QK;
  static constexpr int HEAD_DIM_VO = Ktraits::HEAD_DIM_VO;
  static constexpr int CTA_KV = Ktraits::CTA_KV;
  // 16 FP8 elements per 16-byte vector: one shared-memory load, one conversion, two 16-byte
  // stores (the 16-bit swizzle permutes 16-byte chunks, so the two halves are stored separately).
  static constexpr int VEC = 16;
  static_assert(NUM_PRODUCER_THREADS == cutlass::NumThreadsPerWarpGroup,
                "the whole producer warpgroup dequantizes");

  CUTLASS_DEVICE static uint32_t smem_addr(void const* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  }
  CUTLASS_DEVICE static void lds128(uint32_t addr, uint32_t (&v)[4]) {
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
                 : "r"(addr));
  }
  CUTLASS_DEVICE static void sts128(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2,
                                    uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :
                 : "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3));
  }

  // Dequantizes one (CTA_KV, HEAD_DIM) tile from the 8-bit staging layout into the 16-bit
  // swizzled K/V layout.
  //
  // Thread mapping: consecutive threads take consecutive rows of the same 16-byte column, so
  // the 8 threads of a quarter warp hit 8 different rows of a swizzle atom, i.e. 8 different
  // 16-byte bank groups, for both the 8-bit loads and the 16-bit stores. Each thread then
  // walks its column in steps of ROWS_PER_PASS rows. Both layouts stack 8-row swizzle atoms
  // along the row mode and ROWS_PER_PASS is a multiple of 8, so a step is a constant byte
  // offset that leaves the XOR pattern unchanged: the swizzled addresses are resolved once per
  // tile and the passes use immediate offsets. ROWS_PER_PASS is the largest multiple of 8 whose
  // pass fits in the warpgroup; head_dim 64 / 128 / 256 use all 128 threads, head_dim 192 uses
  // 96 (12 columns of 8 rows).
  template <int HEAD_DIM, typename TensorSrc, typename TensorDst>
  CUTLASS_DEVICE static void convert_tile(TensorSrc& sSrc, TensorDst& sDst, int src_stage,
                                          int dst_stage, int thread_idx) {
    constexpr int THREADS_PER_ROW = HEAD_DIM / VEC;
    constexpr int ROWS_PER_PASS = (NUM_PRODUCER_THREADS / THREADS_PER_ROW) / 8 * 8;
    constexpr int ACTIVE_THREADS = ROWS_PER_PASS * THREADS_PER_ROW;
    constexpr int NUM_PASSES = CTA_KV / ROWS_PER_PASS;
    static_assert(HEAD_DIM % VEC == 0 && ROWS_PER_PASS >= 8);
    static_assert(ACTIVE_THREADS <= NUM_PRODUCER_THREADS && CTA_KV % ROWS_PER_PASS == 0);
    if constexpr (ACTIVE_THREADS < NUM_PRODUCER_THREADS) {
      if (thread_idx >= ACTIVE_THREADS) {
        return;
      }
    }
    const int row = thread_idx % ROWS_PER_PASS;
    const int col = (thread_idx / ROWS_PER_PASS) * VEC;
    const uint32_t src = smem_addr(&sSrc(row, col, src_stage));
    const uint32_t dst_lo = smem_addr(&sDst(row, col, dst_stage));
    const uint32_t dst_hi = smem_addr(&sDst(row, col + VEC / 2, dst_stage));
    const uint32_t src_step = smem_addr(&sSrc(row + ROWS_PER_PASS, col, src_stage)) - src;
    const uint32_t dst_step = smem_addr(&sDst(row + ROWS_PER_PASS, col, dst_stage)) - dst_lo;

    // Software pipelined in groups of up to four passes: issue the loads of a group, then
    // convert and store. NUM_PASSES is 3..8 for the supported tiles.
    constexpr int GROUP = NUM_PASSES % 4 == 0 ? 4 : (NUM_PASSES % 3 == 0 ? 3 : 2);
    static_assert(NUM_PASSES % GROUP == 0);
#pragma unroll
    for (int g = 0; g < NUM_PASSES; g += GROUP) {
      uint32_t in[GROUP][4];
#pragma unroll
      for (int i = 0; i < GROUP; ++i) {
        lds128(src + (g + i) * src_step, in[i]);
      }
#pragma unroll
      for (int i = 0; i < GROUP; ++i) {
        uint32_t out[8];
        Fp8x16Dequantizer<DTypeKV, DTypeKVMma>::convert(in[i], out);
        const uint32_t off = (g + i) * dst_step;
        sts128(dst_lo + off, out[0], out[1], out[2], out[3]);
        sts128(dst_hi + off, out[4], out[5], out[6], out[7]);
      }
    }
  }

  // Waits for the next staged tile, dequantizes it into the next 16-bit stage and publishes it.
  template <int HEAD_DIM, typename TensorSrc, typename TensorDst>
  CUTLASS_DEVICE static void convert(TensorSrc& sSrc, TensorDst& sDst, StagingPipeline& staging,
                                     StagingPipelineState& staging_read, MainloopPipeline& pipeline,
                                     PipelineState& write, int thread_idx) {
    staging.consumer_wait(staging_read);
    pipeline.producer_acquire(write);
    convert_tile<HEAD_DIM>(sSrc, sDst, staging_read.index(), write.index(), thread_idx);
    // The 16-bit tile is written by the generic proxy and read by the GMMAs (async proxy).
    cutlass::arch::fence_view_async_shared();
    pipeline.producer_commit(write);
    staging.consumer_release(staging_read);
    ++staging_read;
    ++write;
  }

  // Runs the producer schedule for one work tile. Tiles are visited from last_tile down to
  // first_tile through next_tile(), and published in the order the consumer waits for them:
  // K(last), K(last-1), V(last), K(last-2), V(last-1), ..., K(first), V(first+1), V(first).
  // Each staging ring is kept NUM_STAGES_KV_STAGING tiles ahead of the dequantization: a tile is
  // staged as soon as the stage that the previous occupant released becomes free.
  //
  // issue_k / issue_v stage one tile (tile index, whether the tile may be partial), issue_q loads
  // Q, and all of them advance their pipeline state on every thread.
  template <typename SharedStorage, typename StagingPipelines, typename IssueQ, typename IssueK,
            typename IssueV, typename NextTile>
  CUTLASS_DEVICE static void run(SharedStorage& shared_storage, StagingPipelines& staging,
                                 MainloopPipeline& pipeline_k, MainloopPipeline& pipeline_v,
                                 PipelineState& smem_pipe_write_k, PipelineState& smem_pipe_write_v,
                                 int thread_idx, int work_idx, int last_tile, int first_tile,
                                 IssueQ&& issue_q, IssueK&& issue_k, IssueV&& issue_v,
                                 NextTile&& next_tile) {
    Tensor sKStaging =
        make_tensor(make_smem_ptr(shared_storage.smem_k_staging.data()), SmemLayoutKStaging{});
    Tensor sVStaging =
        make_tensor(make_smem_ptr(shared_storage.smem_v_staging.data()), SmemLayoutVStaging{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    int k_cursor = last_tile, v_cursor = last_tile;
    auto stage_next_k = [&] {
      if (k_cursor >= first_tile) {
        issue_k(k_cursor, /*maybe_partial=*/k_cursor == last_tile);
        k_cursor = next_tile(k_cursor);
      }
    };
    auto stage_next_v = [&] {
      if (v_cursor >= first_tile) {
        issue_v(v_cursor, /*maybe_partial=*/v_cursor == last_tile);
        v_cursor = next_tile(v_cursor);
      }
    };
    auto convert_k = [&] {
      convert<HEAD_DIM_QK>(sKStaging, sK, staging.k, staging.read_k, pipeline_k, smem_pipe_write_k,
                           thread_idx);
    };
    auto convert_v = [&] {
      convert<HEAD_DIM_VO>(sVStaging, sV, staging.v, staging.read_v, pipeline_v, smem_pipe_write_v,
                           thread_idx);
    };

#pragma unroll
    for (int i = 0; i < NUM_STAGES_KV_STAGING; ++i) {
      stage_next_k();
    }

    // Wait for the MMA warpgroups to say that smem_q is ready.
    cutlass::arch::NamedBarrier::sync(NUM_MMA_THREADS + NUM_PRODUCER_THREADS,
                                      static_cast<int>(NamedBarriers::kQueryEmpty));
    issue_q();

    // The staging buffers do not alias smem_o, so V can be staged before the epilogue of the
    // previous work tile has drained.
#pragma unroll
    for (int i = 0; i < NUM_STAGES_KV_STAGING; ++i) {
      stage_next_v();
    }

    convert_k();
    stage_next_k();

    // smem_v aliases smem_o: wait until the MMA warpgroups have written O of the previous work
    // tile before dequantizing V into it. See mainloop.cuh for why this is a cluster barrier.
    shared_storage.barrier_O.wait((work_idx + 1) % 2);

#pragma unroll 1
    for (int tile = last_tile;; tile = next_tile(tile)) {
      const int tile_next = next_tile(tile);
      if (tile_next >= first_tile) {
        convert_k();
        stage_next_k();
      }
      convert_v();
      stage_next_v();
      if (tile_next < first_tile) {
        break;
      }
    }
  }
};

// Dense (ragged or single) K/V: the staging buffers are filled by TMA, issued by one thread.
template <typename AdditionalParams, typename Ktraits, bool CAUSAL>
struct DequantCollectiveMainloop {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using TileShape_PDV = typename Ktraits::TileShape_PDV;
  static constexpr int CTA_Q = get<0>(TileShape_QKD{});
  static constexpr int CTA_KV = get<1>(TileShape_QKD{});

  static constexpr int NUM_STAGES = Ktraits::NUM_STAGES;
  static constexpr int NUM_STAGES_KV_STAGING = Ktraits::NUM_STAGES_KV_STAGING;
  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  static constexpr int NUM_PRODUCER_THREADS = Ktraits::NUM_PRODUCER_THREADS;
  static constexpr int HEAD_DIM_QK = Ktraits::HEAD_DIM_QK;
  static constexpr int HEAD_DIM_VO = Ktraits::HEAD_DIM_VO;
  static_assert(Ktraits::KV_DEQUANT && Ktraits::USE_TMA_LOAD_KV_STAGING);

  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  using GmemTiledCopyKV = cute::SM90_TMA_LOAD;

  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutKStaging = typename Ktraits::SmemLayoutKStaging;
  using SmemLayoutVStaging = typename Ktraits::SmemLayoutVStaging;

  using ShapeT = cute::Shape<int32_t, int32_t, int32_t>;
  using StrideT = cute::Shape<int64_t, _1, int64_t>;  // (N, D, H)
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  using TMA_Q = decltype(make_tma_copy(
      GmemTiledCopyQ{},
      make_tensor(make_gmem_ptr(static_cast<DTypeQ const*>(nullptr)),
                  repeat_like(StrideT{}, int32_t(0)), StrideT{}),
      SmemLayoutQ{}, select<0, 2>(TileShape_QKD{}), _1{}));  // no mcast for Q

  using TMA_K = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(make_gmem_ptr(static_cast<DTypeKV const*>(nullptr)),
                  repeat_like(StrideT{}, int32_t(0)), StrideT{}),
      take<0, 2>(SmemLayoutKStaging{}), select<1, 2>(TileShape_QKD{}), _1{}));  // no mcast

  using TMA_V = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(make_gmem_ptr(static_cast<DTypeKV const*>(nullptr)),
                  repeat_like(StrideT{}, int32_t(0)), StrideT{}),
      take<0, 2>(SmemLayoutVStaging{}), select<2, 1>(TileShape_PDV{}), _1{}));  // no mcast

  // The 16-bit K/V pipelines are signaled by the producer threads, not by the TMA unit.
  static constexpr bool USE_TMA_LOAD_KV = false;
  static constexpr bool KV_DEQUANT = true;
  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using StagingPipeline = typename Ktraits::StagingPipeline;
  using StagingPipelineState = typename Ktraits::StagingPipelineState;

  static constexpr uint32_t TmaTransactionBytesQ =
      static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<DTypeQ> / 8);
  static constexpr uint32_t TmaTransactionBytesKStaging = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutKStaging{})) * cutlass::sizeof_bits_v<DTypeKV> / 8);
  static constexpr uint32_t TmaTransactionBytesVStaging = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutVStaging{})) * cutlass::sizeof_bits_v<DTypeKV> / 8);

  // Same heuristic as the 16-bit mainloops (the GEMMs run in DTypeQ).
  static constexpr bool UseSchedulerBarrier = HEAD_DIM_VO <= 128;
  using WarpScheduler = WarpScheduler<Ktraits, UseSchedulerBarrier>;

  // Staging pipelines and their states; constructed by the producer warpgroup, which is both the
  // producer (the TMA-issuing thread) and the consumer (all 128 dequantizing threads).
  struct StagingPipelines {
    StagingPipeline k, v;
    StagingPipelineState write_k, read_k, write_v, read_v;

    template <typename SharedStorage>
    CUTLASS_DEVICE StagingPipelines(SharedStorage& shared_storage, int warp_group_thread_idx)
        : k(shared_storage.pipeline_k_staging,
            make_params(warp_group_thread_idx, TmaTransactionBytesKStaging),
            /*cluster_shape=*/Shape<_1, _1, _1>{}),
          v(shared_storage.pipeline_v_staging,
            make_params(warp_group_thread_idx, TmaTransactionBytesVStaging),
            /*cluster_shape=*/Shape<_1, _1, _1>{}),
          write_k(cutlass::make_producer_start_state<StagingPipeline>()),
          write_v(cutlass::make_producer_start_state<StagingPipeline>()) {}

    CUTLASS_DEVICE static typename StagingPipeline::Params make_params(int warp_group_thread_idx,
                                                                       uint32_t transaction_bytes) {
      typename StagingPipeline::Params params;
      params.role = StagingPipeline::ThreadCategory::ProducerConsumer;
      params.is_leader = warp_group_thread_idx == 0;
      params.num_consumers = NUM_PRODUCER_THREADS;
      params.transaction_bytes = transaction_bytes;
      return params;
    }
  };

  // Host side kernel arguments
  struct Arguments {
    DTypeQ const* Q_ptr;
    LayoutT layout_Q;
    DTypeKV const* K_ptr;
    LayoutT layout_K;
    DTypeKV const* V_ptr;
    LayoutT layout_V;
    int window_left;
    AdditionalParams additional_params;
  };

  // Device side kernel params
  struct Params {
    LayoutT layout_Q;
    LayoutT layout_K;
    LayoutT layout_V;
    TMA_Q tma_load_Q;
    TMA_K tma_load_K;
    TMA_V tma_load_V;
    int window_left;
    AdditionalParams additional_params;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.Q_ptr), args.layout_Q);
    TMA_Q tma_load_Q = make_tma_copy(GmemTiledCopyQ{}, mQ, SmemLayoutQ{},
                                     select<0, 2>(TileShape_QKD{}), _1{});  // no mcast for Q
    Tensor mK = make_tensor(make_gmem_ptr(args.K_ptr), args.layout_K);
    TMA_K tma_load_K = make_tma_copy(GmemTiledCopyKV{}, mK, SmemLayoutKStaging{}(_, _, _0{}),
                                     select<1, 2>(TileShape_QKD{}), _1{});  // no mcast
    Tensor mV = make_tensor(make_gmem_ptr(args.V_ptr), args.layout_V);
    TMA_V tma_load_V = make_tma_copy(GmemTiledCopyKV{}, mV, SmemLayoutVStaging{}(_, _, _0{}),
                                     select<2, 1>(TileShape_PDV{}), _1{});  // no mcast
    return {args.layout_Q, args.layout_K, args.layout_V,    tma_load_Q,
            tma_load_K,    tma_load_V,    args.window_left, args.additional_params};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_K.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_V.get_tma_descriptor());
  }

  CUTLASS_DEVICE
  int get_num_kv_tiles(Params const& mainloop_params, int q_tile_idx, const int qo_len,
                       const int kv_len) {
    // Function-local copies: cute::ceil_div takes references, and ODR-using the class-scope
    // constants from device code trips nvcc.
    static constexpr int CTA_Q = get<0>(TileShape_QKD{});
    static constexpr int CTA_KV = get<1>(TileShape_QKD{});
    int num_kv_tiles = cute::ceil_div(kv_len, CTA_KV);
    if constexpr (CAUSAL) {
      num_kv_tiles = std::min(num_kv_tiles,
                              cute::ceil_div((q_tile_idx + 1) * CTA_Q + kv_len - qo_len, CTA_KV));
    }
    return num_kv_tiles;
  }

  template <bool LEFT_SLIDING_WINDOW, typename BlockCoord, typename Scheduler,
            typename SharedStorage>
  CUTLASS_DEVICE void load(Params const& mainloop_params, MainloopPipeline pipeline_k,
                           MainloopPipeline pipeline_v, PipelineState& smem_pipe_write_k,
                           PipelineState& smem_pipe_write_v, StagingPipelines& staging,
                           SharedStorage& shared_storage, Scheduler& scheduler,
                           typename Scheduler::Params const& scheduler_params,
                           typename Scheduler::WorkTileInfo& work_tile_info,
                           BlockCoord const& block_coord, int work_idx,
                           const int num_kv_tiles_outside_items_window = 0,
                           const int num_kv_tiles_prefix = 0) {
    const int thread_idx = threadIdx.x;
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK =
        make_tensor(make_smem_ptr(shared_storage.smem_k_staging.data()), SmemLayoutKStaging{});
    Tensor sV =
        make_tensor(make_smem_ptr(shared_storage.smem_v_staging.data()), SmemLayoutVStaging{});

    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.layout_Q.shape());
    Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.layout_K.shape());
    Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(mainloop_params.layout_V.shape());

    auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
        block_coord;

    // Prepare the TMA loads
    Tensor gQ = get_local_tile_tensor(mQ, select<0, 2>(TileShape_QKD{}), qo_head_idx, qo_indptr,
                                      qo_len)(_, _, q_tile_idx);  // (Q, D)
    Tensor gK = get_local_tile_tensor(mK, select<1, 2>(TileShape_QKD{}), kv_head_idx, kv_indptr,
                                      kv_len);  // (K, D, _)
    Tensor gV = get_local_tile_tensor(mV, select<2, 1>(TileShape_PDV{}), kv_head_idx, kv_indptr,
                                      kv_len);  // (K, D, _)

    Tensor sQ_x = make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
    Tensor gQ_x = make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));
    auto [tQgQ, tQsQ] =
        tma_partition(mainloop_params.tma_load_Q, _0{}, Layout<_1>{}, group_modes<0, 2>(sQ_x),
                      group_modes<0, 2>(gQ_x));  // (TMA), (TMA)
    auto [tKgK, tKsK] =
        tma_partition(mainloop_params.tma_load_K, _0{}, Layout<_1>{}, group_modes<0, 2>(sK),
                      group_modes<0, 2>(gK));  // (TMA, k), (TMA, PIPE)
    auto [tVgV, tVsV] =
        tma_partition(mainloop_params.tma_load_V, _0{}, Layout<_1>{}, group_modes<0, 2>(sV),
                      group_modes<0, 2>(gV));  // (TMA, k), (TMA, PIPE)

    const int num_kv_tiles = get_num_kv_tiles(mainloop_params, q_tile_idx, qo_len, kv_len);
    const int last_tile = num_kv_tiles - 1;
    int first_tile = 0;
    if constexpr (LEFT_SLIDING_WINDOW) {
      first_tile = get_swa_begin_kv_tile_idx<CTA_Q, CTA_KV>(mainloop_params.window_left, q_tile_idx,
                                                            qo_len, kv_len);
    }

    // One thread issues the TMA loads; every thread advances the staging write states.
    const int lane_predicate = cute::elect_one_sync();
    const int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (thread_idx / 32) % 4, 0);
    const bool issue_thread = warp_idx_in_warpgroup == 0 && lane_predicate;

    auto issue_q = [&] {
      if (issue_thread) {
        shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
        copy(mainloop_params.tma_load_Q.with(
                 reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                     shared_storage.barrier_Q),
                 /*mcast_mask=*/0),
             tQgQ, tQsQ);
      }
    };
    auto issue_k = [&](int tile, bool /*maybe_partial*/) {
      if (issue_thread) {
        staging.k.producer_acquire(staging.write_k);
        copy(mainloop_params.tma_load_K.with(*staging.k.producer_get_barrier(staging.write_k),
                                             /*mcast_mask=*/0),
             tKgK(_, tile), tKsK(_, staging.write_k.index()));
      }
      ++staging.write_k;
    };
    auto issue_v = [&](int tile, bool /*maybe_partial*/) {
      if (issue_thread) {
        staging.v.producer_acquire(staging.write_v);
        copy(mainloop_params.tma_load_V.with(*staging.v.producer_get_barrier(staging.write_v),
                                             /*mcast_mask=*/0),
             tVgV(_, tile), tVsV(_, staging.write_v.index()));
      }
      ++staging.write_v;
    };
    auto next_tile = [](int tile) { return tile - 1; };

    DequantKVSchedule<Ktraits>::run(shared_storage, staging, pipeline_k, pipeline_v,
                                    smem_pipe_write_k, smem_pipe_write_v, thread_idx, work_idx,
                                    last_tile, first_tile, issue_q, issue_k, issue_v, next_tile);

    scheduler.prefetch_next_work(scheduler_params, work_tile_info);
    scheduler.broadcast_next_work(work_tile_info);
  }

  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline_k, MainloopPipeline pipeline_v,
                                PipelineState& smem_pipe_write_k, PipelineState& smem_pipe_write_v,
                                StagingPipelines& staging) {
    pipeline_k.producer_tail(smem_pipe_write_k);
    pipeline_v.producer_tail(smem_pipe_write_v);
    // Every staged tile has been dequantized (and its stage released) by this warpgroup, so the
    // staging pipelines are already drained.
  }
};

// Paged K/V: the staging buffers are filled by a cp.async gather through the page table, issued
// by all producer threads. The gather mirrors SparseCollectiveMainloop (sparse_mainloop.cuh).
template <typename AdditionalParams, typename Ktraits, bool CAUSAL, bool MULTIITEMSCORING = false>
struct SparseDequantCollectiveMainloop {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using IdType = typename Ktraits::IdType;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using TileShape_PDV = typename Ktraits::TileShape_PDV;
  static constexpr int CTA_Q = get<0>(TileShape_QKD{});
  static constexpr int CTA_KV = get<1>(TileShape_QKD{});

  static constexpr int NUM_STAGES = Ktraits::NUM_STAGES;
  static constexpr int NUM_STAGES_KV_STAGING = Ktraits::NUM_STAGES_KV_STAGING;
  static constexpr int HEAD_DIM_QK = Ktraits::HEAD_DIM_QK;
  static constexpr int HEAD_DIM_VO = Ktraits::HEAD_DIM_VO;
  static_assert(HEAD_DIM_QK == HEAD_DIM_VO);
  static constexpr int NUM_COPY_THREADS = cutlass::NumThreadsPerWarpGroup;
  static_assert(Ktraits::KV_DEQUANT && !Ktraits::USE_TMA_LOAD_KV_STAGING);
  static_assert(Ktraits::NUM_PRODUCER_THREADS == NUM_COPY_THREADS,
                "NUM_PRODUCER_THREADS must equal NUM_COPY_THREADS for sparse/paged KV loading");

  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  static constexpr auto AlignmentKV = 128 / cutlass::sizeof_bits<DTypeKV>::value;
  using AlignmentTypeKV = cute::uint_byte_t<static_cast<int>(sizeof(DTypeKV)) * AlignmentKV>;
  // NOTE(Zihao): use SM80_CP_ASYNC for sparse loading of KV-cache
  using GmemCopyAtomKV = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<AlignmentTypeKV>, DTypeKV>;
  using GmemTiledCopyK =
      decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
               GmemCopyAtomKV, NUM_COPY_THREADS, AlignmentKV,
               cutlass::detail::TagToStrideB_t<cutlass::layout::ColumnMajor>,
               decltype(cute::get<1>(TileShape_QKD{})), decltype(cute::get<2>(TileShape_QKD{}))>());
  using GmemTiledCopyV =
      decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
               GmemCopyAtomKV, NUM_COPY_THREADS, AlignmentKV,
               cutlass::detail::TagToStrideB_t<cutlass::layout::ColumnMajor>,
               decltype(cute::get<2>(TileShape_PDV{})), decltype(cute::get<1>(TileShape_PDV{}))>());

  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutKStaging = typename Ktraits::SmemLayoutKStaging;
  using SmemLayoutVStaging = typename Ktraits::SmemLayoutVStaging;

  using ShapeT = cute::Shape<int32_t, int32_t, int32_t>;
  using StrideT = cute::Shape<int64_t, _1, int64_t>;  // (N, D, H)
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  using TMA_Q = decltype(make_tma_copy(
      GmemTiledCopyQ{},
      make_tensor(make_gmem_ptr(static_cast<DTypeQ const*>(nullptr)),
                  repeat_like(StrideT{}, int32_t(0)), StrideT{}),
      SmemLayoutQ{}, select<0, 2>(TileShape_QKD{}), _1{}));  // no mcast for Q

  static constexpr bool USE_TMA_LOAD_KV = false;
  static constexpr bool KV_DEQUANT = true;
  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  static constexpr int NUM_PRODUCER_THREADS = Ktraits::NUM_PRODUCER_THREADS;
  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using StagingPipeline = typename Ktraits::StagingPipeline;
  using StagingPipelineState = typename Ktraits::StagingPipelineState;

  static constexpr uint32_t TmaTransactionBytesQ =
      static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<DTypeQ> / 8);

  static constexpr bool UseSchedulerBarrier = HEAD_DIM_VO <= 128;
  using WarpScheduler = WarpScheduler<Ktraits, UseSchedulerBarrier>;

  // Staging pipelines and their states; every producer thread issues cp.async into a stage
  // (producer) and dequantizes it (consumer).
  struct StagingPipelines {
    StagingPipeline k, v;
    StagingPipelineState write_k, read_k, write_v, read_v;

    template <typename SharedStorage>
    CUTLASS_DEVICE StagingPipelines(SharedStorage& shared_storage, int /*warp_group_thread_idx*/)
        : k(shared_storage.pipeline_k_staging, make_params()),
          v(shared_storage.pipeline_v_staging, make_params()),
          write_k(cutlass::make_producer_start_state<StagingPipeline>()),
          write_v(cutlass::make_producer_start_state<StagingPipeline>()) {}

    CUTLASS_DEVICE static typename StagingPipeline::Params make_params() {
      typename StagingPipeline::Params params;
      params.role = StagingPipeline::ThreadCategory::ProducerConsumer;
      params.producer_arv_count = NUM_COPY_THREADS;
      params.consumer_arv_count = NUM_COPY_THREADS;
      return params;
    }
  };

  // Host side kernel arguments
  struct Arguments {
    DTypeQ const* Q_ptr;
    LayoutT layout_Q;
    DTypeKV const* K_ptr;
    LayoutT layout_K;
    DTypeKV const* V_ptr;
    LayoutT layout_V;
    IdType const* kv_indices;
    int window_left;
    int64_t k_page_stride;  // Stride between pages for K (paged_k.stride(0))
    int64_t v_page_stride;  // Stride between pages for V (paged_v.stride(0))
    uint32_t page_size;     // Size of each page
    AdditionalParams additional_params;
  };

  // Device side kernel params
  struct Params {
    LayoutT layout_Q;
    LayoutT layout_K;
    LayoutT layout_V;
    TMA_Q tma_load_Q;
    DTypeKV* K_ptr;
    DTypeKV* V_ptr;
    IdType* kv_indices;
    int window_left;
    int64_t k_page_stride;   // Stride between pages for K
    int64_t v_page_stride;   // Stride between pages for V
    uint_fastdiv page_size;  // Size of each page (as fastdiv for efficient divmod)
    AdditionalParams additional_params;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.Q_ptr), args.layout_Q);
    TMA_Q tma_load_Q =
        make_tma_copy(GmemTiledCopyQ{}, mQ, SmemLayoutQ{}, select<0, 2>(TileShape_QKD{}), _1{});
    return {args.layout_Q,
            args.layout_K,
            args.layout_V,
            tma_load_Q,
            const_cast<DTypeKV*>(args.K_ptr),
            const_cast<DTypeKV*>(args.V_ptr),
            const_cast<IdType*>(args.kv_indices),
            args.window_left,
            args.k_page_stride,
            args.v_page_stride,
            uint_fastdiv(args.page_size),
            args.additional_params};
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_Q.get_tma_descriptor());
  }

  CUTLASS_DEVICE
  int get_num_kv_tiles(Params const& mainloop_params, int q_tile_idx, const int qo_len,
                       const int kv_len) {
    // Function-local copies: cute::ceil_div takes references, and ODR-using the class-scope
    // constants from device code trips nvcc.
    static constexpr int CTA_Q = get<0>(TileShape_QKD{});
    static constexpr int CTA_KV = get<1>(TileShape_QKD{});
    int num_kv_tiles = cute::ceil_div(kv_len, CTA_KV);
    if constexpr (CAUSAL || MULTIITEMSCORING) {
      num_kv_tiles = std::min(num_kv_tiles,
                              cute::ceil_div((q_tile_idx + 1) * CTA_Q + kv_len - qo_len, CTA_KV));
    }
    return num_kv_tiles;
  }

  template <bool LEFT_SLIDING_WINDOW, typename BlockCoord, typename Scheduler,
            typename SharedStorage>
  CUTLASS_DEVICE void load(Params const& mainloop_params, MainloopPipeline pipeline_k,
                           MainloopPipeline pipeline_v, PipelineState& smem_pipe_write_k,
                           PipelineState& smem_pipe_write_v, StagingPipelines& staging,
                           SharedStorage& shared_storage, Scheduler& scheduler,
                           typename Scheduler::Params const& scheduler_params,
                           typename Scheduler::WorkTileInfo& work_tile_info,
                           BlockCoord const& block_coord, int work_idx,
                           const int num_kv_tiles_outside_items_window = 0,
                           const int num_kv_tiles_prefix = 0) {
    const int thread_idx = threadIdx.x;
    const int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (thread_idx / 32) % 4, 0);
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK =
        make_tensor(make_smem_ptr(shared_storage.smem_k_staging.data()), SmemLayoutKStaging{});
    Tensor sV =
        make_tensor(make_smem_ptr(shared_storage.smem_v_staging.data()), SmemLayoutVStaging{});

    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.layout_Q.shape());

    auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
        block_coord;

    // Prepare the TMA loads
    Tensor gQ = get_local_tile_tensor(mQ, select<0, 2>(TileShape_QKD{}), qo_head_idx, qo_indptr,
                                      qo_len)(_, _, q_tile_idx);  // (Q, D)

    Tensor sQ_x = make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
    Tensor gQ_x = make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));
    auto [tQgQ, tQsQ] =
        tma_partition(mainloop_params.tma_load_Q, _0{}, Layout<_1>{}, group_modes<0, 2>(sQ_x),
                      group_modes<0, 2>(gQ_x));  // (TMA), (TMA)

    const int num_kv_tiles = get_num_kv_tiles(mainloop_params, q_tile_idx, qo_len, kv_len);
    const int last_tile = num_kv_tiles - 1;
    int first_tile = 0;
    if constexpr (LEFT_SLIDING_WINDOW) {
      first_tile = get_swa_begin_kv_tile_idx<CTA_Q, CTA_KV>(mainloop_params.window_left, q_tile_idx,
                                                            qo_len, kv_len);
    }

    // Store base pointers and indices for manual page table lookup
    DTypeKV* K_ptr_base = mainloop_params.K_ptr + kv_head_idx * stride<2>(mainloop_params.layout_K);
    DTypeKV* V_ptr_base = mainloop_params.V_ptr + kv_head_idx * stride<2>(mainloop_params.layout_V);
    IdType const* kv_indices_ptr = mainloop_params.kv_indices + kv_indptr;
    // Use the page stride (stride between pages) and stride within page
    const int64_t k_page_stride = mainloop_params.k_page_stride;
    const int64_t k_stride_n = stride<0>(mainloop_params.layout_K);

    // Create dummy tensors for partitioning with contiguous column-major layout
    // NOTE: We use a virtual contiguous layout for correct partitioning,
    // actual addressing uses page table lookup
    Tensor gK =
        make_tensor(make_gmem_ptr(static_cast<DTypeKV*>(nullptr)), make_shape(CTA_KV, HEAD_DIM_QK),
                    make_stride(HEAD_DIM_QK, _1{}));  // Column-major: (KV, D)
    Tensor gK_tiled =
        local_tile(gK, select<1, 2>(TileShape_QKD{}), make_coord(_, _0{}));  // (KV, D_K, kv)
    Tensor gV =
        make_tensor(make_gmem_ptr(static_cast<DTypeKV*>(nullptr)), make_shape(CTA_KV, HEAD_DIM_VO),
                    make_stride(HEAD_DIM_VO, _1{}));  // Column-major: (KV, D)
    Tensor gV_tiled =
        local_tile(gV, select<2, 1>(TileShape_PDV{}), make_coord(_, _0{}));  // (KV, D_V, kv)
    Tensor cK = cute::make_identity_tensor(gK_tiled.shape());
    Tensor cV = cute::make_identity_tensor(gV_tiled.shape());

    GmemTiledCopyK gmem_tiled_copy_k;
    GmemTiledCopyV gmem_tiled_copy_v;
    auto gmem_thr_copy_k = gmem_tiled_copy_k.get_slice(thread_idx);
    auto gmem_thr_copy_v = gmem_tiled_copy_v.get_slice(thread_idx);

    Tensor tKgK = gmem_thr_copy_k.partition_S(gK_tiled);  // (CPY, CPY_KV, CPY_D, kv)
    Tensor tKsK = gmem_thr_copy_k.partition_D(sK);        // (CPY, CPY_KV, CPY_D, PIPE)
    Tensor tVgV = gmem_thr_copy_v.partition_S(gV_tiled);  // (CPY, CPY_KV, CPY_D, kv)
    Tensor tVsV = gmem_thr_copy_v.partition_D(sV);        // (CPY, CPY_KV, CPY_D, PIPE)
    Tensor tKcK = gmem_thr_copy_k.partition_D(cK);        // (CPY, CPY_KV, CPY_D, kv)
    Tensor tVcV = gmem_thr_copy_v.partition_D(cV);        // (CPY, CPY_KV, CPY_D, kv)

    // Group organization based on partition strategy (see SparseCollectiveMainloop). Each thread
    // of a group looks up the page-table offset of one KV row; the rows a thread copies all live
    // in its own group, so the offsets are exchanged with warp shuffles.
    constexpr int NUM_KV_PER_ITER = decltype(size<1>(tKcK))::value;
    constexpr int KV_STRIDE = CTA_KV / NUM_KV_PER_ITER;
    constexpr int NUM_GROUPS = KV_STRIDE;
    constexpr int THREADS_PER_GROUP = NUM_COPY_THREADS / NUM_GROUPS;
    constexpr int NUM_ITERS_PER_GROUP = NUM_KV_PER_ITER;
    static_assert(NUM_ITERS_PER_GROUP <= THREADS_PER_GROUP);

    const int group_id = thread_idx / THREADS_PER_GROUP;
    const int thread_in_group = thread_idx % THREADS_PER_GROUP;

    // K and V of a tile are staged at different times, so the offset is recomputed for each
    // (one divmod and one page-table load per thread; K and V share strides, checked on the host).
    auto kv_offset_of_tile = [&](int kv_tile_idx, bool use_predicate) -> int64_t {
      const int kv_idx_read = kv_tile_idx * CTA_KV + group_id + thread_in_group * KV_STRIDE;
      const bool valid_read =
          thread_in_group < NUM_ITERS_PER_GROUP && (!use_predicate || kv_idx_read < kv_len);
      if (!valid_read) {
        return 0;
      }
      uint32_t page_iter, entry_idx;
      mainloop_params.page_size.divmod(kv_idx_read, page_iter, entry_idx);
      const IdType page_idx = kv_indices_ptr[page_iter];
      return page_idx * k_page_stride + entry_idx * k_stride_n;
    };

    auto load_kv_with_gather = [&](auto&& tXsX, auto&& tXcX, DTypeKV* base_ptr, int kv_tile_idx,
                                   int stage_idx, bool use_predicate, int64_t my_kv_offset) {
      using Vec = AlignmentTypeKV;
      constexpr int VecSize = sizeof(Vec) / sizeof(DTypeKV);
      const int kv_base_idx = kv_tile_idx * CTA_KV;
      auto dst = recast<Vec>(flatten(tXsX(_, _, _, stage_idx)));
      auto c = flatten(tXcX(_, _, _, kv_tile_idx));
      constexpr unsigned FULL_MASK = 0xffffffff;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(dst); ++i) {
        auto coord = c(VecSize * i);
        const int kv_offset = get<0>(coord);
        const int d_idx = get<1>(coord);
        const int kv_idx = kv_base_idx + kv_offset;
        const bool guard = !use_predicate || kv_idx < kv_len;
        const int src_thread = group_id * THREADS_PER_GROUP + kv_offset / KV_STRIDE;
        const int64_t base_offset = __shfl_sync(FULL_MASK, my_kv_offset, src_thread);
        Vec const* src_ptr = reinterpret_cast<Vec const*>(base_ptr + base_offset + d_idx);
        cutlass::arch::cp_async_zfill<sizeof(Vec), cutlass::arch::CacheOperation::Global>(
            &dst(i), src_ptr, guard);
      }
    };

    auto issue_q = [&] {
      if (warp_idx_in_warpgroup == 0) {
        const int lane_predicate = cute::elect_one_sync();
        if (lane_predicate) {
          shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
          copy(mainloop_params.tma_load_Q.with(
                   reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                       shared_storage.barrier_Q),
                   /*mcast_mask=*/0),
               tQgQ, tQsQ);
        }
      }
    };
    auto issue_k = [&](int tile, bool maybe_partial) {
      const int64_t my_kv_offset = kv_offset_of_tile(tile, maybe_partial);
      staging.k.producer_acquire(staging.write_k);
      load_kv_with_gather(tKsK, tKcK, K_ptr_base, tile, staging.write_k.index(), maybe_partial,
                          my_kv_offset);
      staging.k.producer_commit(staging.write_k, cutlass::arch::cpasync_barrier_arrive);
      ++staging.write_k;
    };
    auto issue_v = [&](int tile, bool maybe_partial) {
      const int64_t my_kv_offset = kv_offset_of_tile(tile, maybe_partial);
      staging.v.producer_acquire(staging.write_v);
      load_kv_with_gather(tVsV, tVcV, V_ptr_base, tile, staging.write_v.index(), maybe_partial,
                          my_kv_offset);
      staging.v.producer_commit(staging.write_v, cutlass::arch::cpasync_barrier_arrive);
      ++staging.write_v;
    };
    auto next_tile = [&](int kv_tile_idx) {
      int result = kv_tile_idx - 1;
      if constexpr (MULTIITEMSCORING) {
        if ((kv_tile_idx == num_kv_tiles_outside_items_window) &
            (kv_tile_idx >= num_kv_tiles_prefix)) {
          result = num_kv_tiles_prefix - 1;
        }
      }
      return result;
    };

    DequantKVSchedule<Ktraits>::run(shared_storage, staging, pipeline_k, pipeline_v,
                                    smem_pipe_write_k, smem_pipe_write_v, thread_idx, work_idx,
                                    last_tile, first_tile, issue_q, issue_k, issue_v, next_tile);

    scheduler.prefetch_next_work(scheduler_params, work_tile_info);
    scheduler.broadcast_next_work(work_tile_info);
  }

  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline_k, MainloopPipeline pipeline_v,
                                PipelineState& smem_pipe_write_k, PipelineState& smem_pipe_write_v,
                                StagingPipelines& staging) {
    pipeline_k.producer_tail(smem_pipe_write_k);
    pipeline_v.producer_tail(smem_pipe_write_v);
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_DEQUANT_MAINLOOP_CUH_
