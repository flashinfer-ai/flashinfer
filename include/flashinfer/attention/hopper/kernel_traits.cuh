/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_KERNEL_TRAITS_CUH_
#define FLASHINFER_ATTENTION_HOPPER_KERNEL_TRAITS_CUH_

#include <type_traits>

#include "../../cutlass_utils.cuh"
#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

namespace flashinfer {

using namespace cute;

template <typename MainloopPipeline, class DTypeQ, class DTypeKV, class DTypeOut, class IdType,
          int CTA_KV, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorageQKVO {
  cute::array_aligned<DTypeQ, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<DTypeKV, cute::cosize_v<SmemLayoutK>> smem_k;
  union {
    cute::array_aligned<DTypeKV, cute::cosize_v<SmemLayoutV>> smem_v;
    cute::array_aligned<DTypeOut, cute::cosize_v<SmemLayoutO>> smem_o;
  };
  struct {
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    cutlass::arch::ClusterBarrier barrier_O;
    typename MainloopPipeline::SharedStorage pipeline_k;
    typename MainloopPipeline::SharedStorage pipeline_v;
  };
};

template <bool USE_TMA_LOAD_KV, int HEAD_DIM_QK_, int HEAD_DIM_VO_, int CTA_Q_, int CTA_KV_,
          int NUM_STAGES_, typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_,
          typename AttentionVariant_>
struct AttentionKernelTraits {
  using AttentionVariant = AttentionVariant_;

  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  // Element type of the K/V tiles in shared memory, i.e. the B operands of the two GEMMs and the
  // type P is converted to. Equal to DTypeKV here; DequantAttentionKernelTraits (FP8 KV cache with
  // a 16-bit query) dequantizes K/V to DTypeQ on load and overrides it.
  using DTypeKVMma = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;
  using DTypeQKAccum = float;
  static constexpr bool KV_DEQUANT = false;

  static constexpr int CTA_Q = CTA_Q_;
  static_assert(CTA_Q % 64 == 0);
  static constexpr int CTA_KV = CTA_KV_;
  static constexpr int HEAD_DIM_QK = HEAD_DIM_QK_;
  static constexpr int HEAD_DIM_VO = HEAD_DIM_VO_;
  static_assert(HEAD_DIM_QK % 32 == 0);
  static_assert(HEAD_DIM_VO % 32 == 0);

  static constexpr int NUM_WARPS = ((CTA_Q / 64) + 1) * 4;
  static constexpr int NUM_THREADS = NUM_WARPS * cutlass::NumThreadsPerWarp;
  // NOTE(Zihao): the following constant should only be used when TMA is enabled,
  // where only one warp inside a warp group is used for TMA.
  static constexpr int NUM_PRODUCER_THREADS =
      USE_TMA_LOAD_KV ? cutlass::NumThreadsPerWarp : 4 * cutlass::NumThreadsPerWarp;

  using TileShape_QKD = Shape<Int<CTA_Q>, Int<CTA_KV>, Int<HEAD_DIM_QK>>;
  using TileShape_PDV = Shape<Int<CTA_Q>, Int<HEAD_DIM_VO>, Int<CTA_KV>>;

  static constexpr int NUM_STAGES = NUM_STAGES_;

  using AtomLayoutQKD = Layout<Shape<Int<CTA_Q / 64>, _1, _1>>;
  using TiledMmaQK = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<DTypeQ, DTypeKV, DTypeQKAccum, TileShape_QKD>(), AtomLayoutQKD{}));
  using TiledMmaPV = decltype(cute::make_tiled_mma(
      cute::GMMA::rs_op_selector<DTypeKV, DTypeKV, /*ElementAccum=*/float, TileShape_PDV,
                                 GMMA::Major::K, GMMA::Major::MN>(),
      AtomLayoutQKD{}));

  static constexpr int NUM_MMA_THREADS = size(TiledMmaQK{});

  using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeQ, decltype(cute::get<0>(TileShape_QKD{})),
                                   decltype(cute::get<2>(TileShape_QKD{}))>());
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_QKD{})));

  using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeKV, decltype(cute::get<1>(TileShape_QKD{})),
                                   decltype(cute::get<2>(TileShape_QKD{}))>());
  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtomK{},
      make_shape(shape<1>(TileShape_QKD{}), shape<2>(TileShape_QKD{}), Int<NUM_STAGES>{})));

  using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeKV, decltype(cute::get<2>(TileShape_PDV{})),
                                   decltype(cute::get<1>(TileShape_PDV{}))>());
  using SmemLayoutV = decltype(tile_to_shape(
      SmemLayoutAtomV{},
      make_shape(get<2>(TileShape_PDV{}), get<1>(TileShape_PDV{}), Int<NUM_STAGES>{})));

  // Note this is the transpose in terms of the view, not in terms of memory.
  using SmemLayoutVt = decltype(composition(
      SmemLayoutV{}, make_ordered_layout(make_shape(get<1>(TileShape_PDV{}),
                                                    get<2>(TileShape_PDV{}), Int<NUM_STAGES>{}),
                                         Step<_2, _1, _3>{})));

  using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeO, decltype(cute::get<0>(TileShape_PDV{})),
                                   decltype(cute::get<1>(TileShape_PDV{}))>());
  using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 1>(TileShape_PDV{})));
  using MainloopPipeline =
      std::conditional_t<USE_TMA_LOAD_KV, typename cutlass::PipelineTmaAsync<NUM_STAGES>,
                         typename cutlass::PipelineAsync<NUM_STAGES>>;
  using PipelineState = typename cutlass::PipelineState<NUM_STAGES>;

  using SharedStorage = SharedStorageQKVO<MainloopPipeline, DTypeQ, DTypeKV, DTypeO, IdType, CTA_KV,
                                          SmemLayoutQ, SmemLayoutK, SmemLayoutV, SmemLayoutO>;
};

// Shared storage of the FP8-KV kernels: the 16-bit Q/K/V/O buffers of SharedStorageQKVO plus the
// 8-bit staging buffers that K/V tiles are loaded into before the producer warpgroup dequantizes
// them into smem_k / smem_v. smem_v aliases smem_o exactly as in SharedStorageQKVO; the staging
// buffers alias nothing, so staging loads never have to wait for the epilogue.
template <typename MainloopPipeline, typename StagingPipeline, class DTypeQ, class DTypeKV,
          class DTypeKVMma, class DTypeOut, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV,
          class SmemLayoutO, class SmemLayoutKStaging, class SmemLayoutVStaging>
struct SharedStorageQKVODequant {
  cute::array_aligned<DTypeQ, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<DTypeKVMma, cute::cosize_v<SmemLayoutK>> smem_k;
  union {
    cute::array_aligned<DTypeKVMma, cute::cosize_v<SmemLayoutV>> smem_v;
    cute::array_aligned<DTypeOut, cute::cosize_v<SmemLayoutO>> smem_o;
  };
  cute::array_aligned<DTypeKV, cute::cosize_v<SmemLayoutKStaging>, 1024> smem_k_staging;
  cute::array_aligned<DTypeKV, cute::cosize_v<SmemLayoutVStaging>, 1024> smem_v_staging;
  struct {
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    cutlass::arch::ClusterBarrier barrier_O;
    typename MainloopPipeline::SharedStorage pipeline_k;
    typename MainloopPipeline::SharedStorage pipeline_v;
    typename StagingPipeline::SharedStorage pipeline_k_staging;
    typename StagingPipeline::SharedStorage pipeline_v_staging;
  };
};

// Kernel traits for a 16-bit query attending to an FP8 (e4m3 / e5m2) KV cache.
//
// The GEMMs run on the 16-bit tensor cores with K/V dequantized to DTypeQ, so everything the MMA
// warpgroups see (TiledMma*, SmemLayout{K,V,Vt}, smem_k, smem_v) is inherited from the 16-bit
// traits with DTypeKV := DTypeQ. What this struct adds is the 8-bit side: the staging layouts the
// K/V tiles are loaded into and the pipelines that hand staged tiles to the dequantizer.
//
// USE_TMA_LOAD_KV selects how the staging buffers are filled (TMA for dense K/V, cp.async gather
// for paged K/V). The kernel sees USE_TMA_LOAD_KV == false through the base: the 16-bit K/V
// pipelines are always thread-signaled (PipelineAsync) because the producer warpgroup, not the TMA
// unit, writes smem_k / smem_v, and all four producer warps take part in the load.
template <bool USE_TMA_LOAD_KV, int HEAD_DIM_QK_, int HEAD_DIM_VO_, int CTA_Q_, int CTA_KV_,
          int NUM_STAGES_, int NUM_STAGES_KV_STAGING_, typename DTypeQ_, typename DTypeKV_,
          typename DTypeO_, typename IdType_, typename AttentionVariant_>
struct DequantAttentionKernelTraits
    : AttentionKernelTraits</*USE_TMA_LOAD_KV=*/false, HEAD_DIM_QK_, HEAD_DIM_VO_, CTA_Q_, CTA_KV_,
                            NUM_STAGES_, DTypeQ_, /*DTypeKV_=*/DTypeQ_, DTypeO_, IdType_,
                            AttentionVariant_> {
  using Base =
      AttentionKernelTraits</*USE_TMA_LOAD_KV=*/false, HEAD_DIM_QK_, HEAD_DIM_VO_, CTA_Q_, CTA_KV_,
                            NUM_STAGES_, DTypeQ_, DTypeQ_, DTypeO_, IdType_, AttentionVariant_>;
  static_assert(cutlass::sizeof_bits_v<DTypeQ_> == 16, "the query must be 16-bit");
  static_assert(cutlass::sizeof_bits_v<DTypeKV_> == 8, "the KV cache must be 8-bit");

  using DTypeKV = DTypeKV_;    // KV cache element type in global memory and the staging buffers
  using DTypeKVMma = DTypeQ_;  // K/V (and P) element type fed to the GEMMs
  static constexpr bool KV_DEQUANT = true;
  static constexpr bool USE_TMA_LOAD_KV_STAGING = USE_TMA_LOAD_KV;
  static constexpr int NUM_STAGES_KV_STAGING = NUM_STAGES_KV_STAGING_;
  static_assert(NUM_STAGES_KV_STAGING >= 1);

  using TileShape_QKD = typename Base::TileShape_QKD;
  using TileShape_PDV = typename Base::TileShape_PDV;

  using SmemLayoutAtomKStaging =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, DTypeKV, decltype(cute::get<1>(TileShape_QKD{})),
               decltype(cute::get<2>(TileShape_QKD{}))>());
  using SmemLayoutKStaging = decltype(tile_to_shape(
      SmemLayoutAtomKStaging{}, make_shape(shape<1>(TileShape_QKD{}), shape<2>(TileShape_QKD{}),
                                           Int<NUM_STAGES_KV_STAGING>{})));
  using SmemLayoutAtomVStaging =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, DTypeKV, decltype(cute::get<2>(TileShape_PDV{})),
               decltype(cute::get<1>(TileShape_PDV{}))>());
  using SmemLayoutVStaging = decltype(tile_to_shape(
      SmemLayoutAtomVStaging{},
      make_shape(get<2>(TileShape_PDV{}), get<1>(TileShape_PDV{}), Int<NUM_STAGES_KV_STAGING>{})));

  using StagingPipeline =
      std::conditional_t<USE_TMA_LOAD_KV, cutlass::PipelineTmaAsync<NUM_STAGES_KV_STAGING>,
                         cutlass::PipelineAsync<NUM_STAGES_KV_STAGING>>;
  using StagingPipelineState = cutlass::PipelineState<NUM_STAGES_KV_STAGING>;

  using SharedStorage =
      SharedStorageQKVODequant<typename Base::MainloopPipeline, StagingPipeline, DTypeQ_, DTypeKV,
                               DTypeKVMma, DTypeO_, typename Base::SmemLayoutQ,
                               typename Base::SmemLayoutK, typename Base::SmemLayoutV,
                               typename Base::SmemLayoutO, SmemLayoutKStaging, SmemLayoutVStaging>;
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_KERNEL_TRAITS_CUH_
