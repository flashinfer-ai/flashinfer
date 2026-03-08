/*
 * Copyright (c) 2025 by FlashInfer team.
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
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "flashinfer/flat/ampere/collective/flat_collective_inverse.hpp"
#include "flashinfer/flat/ampere/collective/flat_collective_load.hpp"
#include "flashinfer/flat/cute_ext.hpp"
#include "flashinfer/flat/hopper/collective/flat_collective_load.hpp"
#include "flashinfer/flat/hopper/collective/flat_collective_store.hpp"
#include "flashinfer/flat/hopper/collective/flat_common.hpp"
#include "flashinfer/flat/hopper/collective/flat_named_barriers.hpp"
#include "flashinfer/flat/hopper/kernel/flat_options.hpp"
#include "flashinfer/flat/math_order_barrier.hpp"
#include "flashinfer/flat/unused.hpp"

// #define INLINE_LAMBDA [[gnu::always_inline]]
#define INLINE_LAMBDA __attribute__((always_inline))
// #define INLINE_LAMBDA [[msvc::forceinline]]

#define WORKAROUND_WGMMA_PERFORMANCE_LOSS() \
  if (thread_idx > 8192) {                  \
    __syncwarp();                           \
  }

namespace flat::collective {

struct DeltaRuleNamedBarriers : FlatSharedNamedBarriers {
  static constexpr int KKLaunched = FlatSharedNamedBarriers::NumBarriersUsed + 0;
  static constexpr int AuxMath = FlatSharedNamedBarriers::NumBarriersUsed + 1;
};

using namespace cute;
using flat::kernel::find_option_t;
using flat::kernel::Tag;

template <class Element_, class ElementAccumulatorQK_, class ElementAccumulatorKV_,
          class TileShape_,  // (seqlen_q, seqlen_kv, d)
          class LayoutQ_, class LayoutK_, class LayoutV_, class LayoutO_,  // (seqlen_q/k, d, h)
          class Options>
struct FlatMainloopTmaWarpSpecializedDeltaRule {
  using Element = Element_;
  using ElementAccumulatorQK = ElementAccumulatorQK_;
  using ElementAccumulatorO = ElementAccumulatorQK;
  using ElementAccumulatorKV = ElementAccumulatorKV_;
  using ElementO = Element;

  using TileShape = TileShape_;

  using LayoutQ = LayoutQ_;  // (seqlen_q, d, h)
  using LayoutK = LayoutK_;  // (seqlen_k, d, h)
  using LayoutV = LayoutV_;  // (seqlen_k, d, h)
  using LayoutO = LayoutO_;  // (seqlen_k, d, h)

  // Options
  static constexpr bool kIsPersistent =
      find_option_t<Tag::kIsPersistent, false_type, Options>::value;

  static constexpr bool kInitStateFromInput =
      find_option_t<Tag::kInitStateFromInput, false_type, Options>::value;

  static constexpr int NumLoadWarpGroups = 1;
  static constexpr int NumStateMmaWarpGroups = 2;
  static constexpr int NumAuxMmaWarpGroups = 1;

  static constexpr int StageCountQ = find_option_t<Tag::kStagesQ, Int<2>, Options>::value;
  static constexpr int StageCountK = find_option_t<Tag::kStagesK, Int<3>, Options>::value;
  static constexpr int StageCountV = find_option_t<Tag::kStagesV, Int<2>, Options>::value;

  static constexpr int NeedsAlpha =
      find_option_t<Tag::kNeedsAlpha, cute::true_type, Options>::value;
  static constexpr int NeedsBeta = find_option_t<Tag::kNeedsBeta, cute::true_type, Options>::value;

  static constexpr int NeedsDecay =
      find_option_t<Tag::kNeedsDecay, cute::false_type, Options>::value;
  static_assert(!NeedsDecay, "DeltaRule does not supports decay");

  static constexpr int NumLoadThreads = NumLoadWarpGroups * 128;
  static constexpr int NumStateMmaThreads = NumStateMmaWarpGroups * 128;
  static constexpr int NumAuxMmaThreads = NumAuxMmaWarpGroups * 128;

  static constexpr uint32_t OrderedBarrierId0 =
      uint32_t(cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
  static constexpr uint32_t OrderedBarrierId1 =
      uint32_t(cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);

  using OrderedMathBarriers = std::conditional_t<
      NumStateMmaWarpGroups == 2,
      OrderedNamedBarriers</*UseReservedNB=*/true, OrderedBarrierId0, OrderedBarrierId1>,
      OrderedNamedBarriers</*UseReservedNB=*/true, OrderedBarrierId0>>;

  using StagesQ = cutlass::gemm::collective::StageCount<StageCountQ>;
  using StagesK = cutlass::gemm::collective::StageCount<StageCountK>;
  using StagesV = cutlass::gemm::collective::StageCount<StageCountV>;
  using StagesO = cutlass::gemm::collective::StageCount<2>;
  using ClusterShape = Shape<_1, _1, _1>;

  using StagesQK = cutlass::gemm::collective::StageCount<2>;
  using StagesKK = cutlass::gemm::collective::StageCount<2>;

  using StagesAlphaBeta = cutlass::gemm::collective::StageCount<5>;

  static constexpr int Alignment = 16 / sizeof(Element);

  static constexpr auto BlkSeqQ = get<0>(TileShape{});   // Blk_Q
  static constexpr auto BlkSeqKV = get<1>(TileShape{});  // Blk_K/V
  static constexpr auto HeadSize = get<2>(TileShape{});  // D (Dq, Dk, Dv all equal)
  static constexpr auto HeadSizeQK = HeadSize;
  static constexpr auto HeadSizeV = HeadSize;

  using TileShapeQK = decltype(make_shape(BlkSeqQ, BlkSeqKV, HeadSizeQK));
  using TileShapeKK = decltype(make_shape(BlkSeqKV, BlkSeqKV, HeadSizeQK));
  using TileShapeKV = decltype(make_shape(HeadSizeV, HeadSizeQK, BlkSeqKV));
  static_assert(std::is_same_v<TileShapeQK, TileShapeKK>);

  using TileShapeO2 = decltype(make_shape(HeadSizeV, BlkSeqQ, BlkSeqKV));
  using TileShapeO1 = decltype(make_shape(HeadSizeV, BlkSeqQ, HeadSizeQK));

  static_assert(BlkSeqQ % 64 == 0);
  static_assert(BlkSeqQ == 64 || BlkSeqQ == 128);
  static constexpr bool IsQKCooperative = BlkSeqQ == 128;
  static constexpr bool IsKKCooperative = IsQKCooperative;

  using DummyStages = cutlass::gemm::collective::StageCount<2>;
  ;
  using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Element, LayoutQ, Alignment, Element,
      LayoutK, Alignment, ElementAccumulatorQK, TileShapeQK, ClusterShape, DummyStages,
      std::conditional_t<IsQKCooperative, cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                         cutlass::gemm::KernelTmaWarpSpecialized>>::CollectiveOp;

  using CollectiveMmaKV_G2S = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Element,
      decltype(select<1, 0, 2>(LayoutV{})), Alignment,  // direct TMA copy for GMEM -> SMEM
      Element, decltype(select<1, 0, 2>(LayoutK{})), Alignment, ElementAccumulatorKV, TileShapeKV,
      ClusterShape, DummyStages, cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  // raw layout for copy
  using SmemLayoutQ_SD =
      decltype(unstage_smem_layout(typename CollectiveMmaQK::SmemLayoutA{}, Int<StagesQ::value>{}));
  using SmemLayoutK_DS = decltype(unstage_smem_layout(typename CollectiveMmaKV_G2S::SmemLayoutB{},
                                                      Int<StagesK::value>{}));
  using SmemLayoutV_DS = decltype(unstage_smem_layout(typename CollectiveMmaKV_G2S::SmemLayoutA{},
                                                      Int<StagesV::value>{}));

  using RefLayoutV = decltype(make_layout(select<0, 2>(TileShapeKV{}), LayoutRight{}));
  using CollectiveMmaKV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Element, RefLayoutV,
      Alignment,  // needs a S2R transposition for MMA
      Element, decltype(select<1, 0, 2>(LayoutK{})), Alignment, ElementAccumulatorKV, TileShapeKV,
      ClusterShape, DummyStages, cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  using RefLayoutKV =
      decltype(make_layout(select<0, 1>(TileShapeKV{}), LayoutRight{}));  // (dv, dk)
  using CollectiveMmaO1 = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Element, RefLayoutKV, Alignment, Element,
      LayoutQ, Alignment, ElementAccumulatorO, TileShapeO1, ClusterShape, DummyStages,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  // (blk_q,blk_k) to align with O2 mma, LayoutRight to align with QK mma output
  using DesiredLayoutQK = decltype(make_layout(select<0, 1>(TileShapeQK{}), LayoutRight{}));
  using CollectiveMmaO2 = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Element, RefLayoutV, Alignment, Element,
      DesiredLayoutQK, Alignment, ElementAccumulatorO, TileShapeO2, ClusterShape, DummyStages,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  using TiledMmaQK = typename CollectiveMmaQK::TiledMma;  // Q@K^t
  using TiledMmaKV = decltype(convert_to_gmma_rs(typename CollectiveMmaKV::TiledMma{}));
  using TiledMmaO1 = decltype(convert_to_gmma_rs(typename CollectiveMmaO1::TiledMma{}));
  using TiledMmaO2 = decltype(convert_to_gmma_rs(typename CollectiveMmaO2::TiledMma{}));

  static constexpr int TiledMmaQKNumThreads = size(TiledMmaQK{});
  static_assert(size(TiledMmaQK{}) == NumAuxMmaThreads);

  static_assert(size(TiledMmaKV{}) == NumStateMmaThreads);
  static_assert(size(TiledMmaO1{}) == NumStateMmaThreads);
  static_assert(size(TiledMmaO2{}) == NumStateMmaThreads);

  using CollectiveStoreO =
      CollectiveStoreTma<TileShapeO1, ClusterShape, ElementO, ElementAccumulatorO,
                         /*Seme*/ ElementO, decltype(select<1, 0, 2>(LayoutO{})), StagesO::value>;

  // layout for compute
  using QKSmemLayoutQ = SmemLayoutQ_SD;
  using QKSmemLayoutK = decltype(select_layout<1, 0, 2>(SmemLayoutK_DS{}));

  using KVSmemLayoutK = SmemLayoutK_DS;
  using KVSmemLayoutV = SmemLayoutV_DS;

  // layout for compute output
  using SmemLayoutQK = decltype(tile_to_shape(
      GMMA::Layout_K_INTER_Atom<Element>{},
      flatten(make_shape(select<0, 1>(TileShapeQK{}), Int<StagesQK::value>{})),
      Step<_1, _2, _3>{}));
  using SmemLayoutO = typename CollectiveStoreO::SmemLayoutO;

  using SmemLayoutKK = decltype(tile_to_shape(
      GMMA::Layout_K_INTER_Atom<Element>{},
      flatten(make_shape(select<0, 1>(TileShapeQK{}), Int<StagesQK::value>{})),
      Step<_1, _2, _3>{}));

  using InverseType = cutlass::half_t;
  using CollectiveInverse = flat::collective::CollectiveInverse<InverseType, true, false>;

  using ElementAccumulatorSK = float;
  using TileShapeSK = decltype(make_shape(HeadSizeV, BlkSeqKV, HeadSizeQK));
  using CollectiveMmaSK =
      typename cutlass::gemm::collective::CollectiveBuilder<  // basically the same as O1
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Element, RefLayoutKV, Alignment,
          Element, LayoutK, Alignment, ElementAccumulatorSK, TileShapeSK, ClusterShape, DummyStages,
          cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  using ElementAccumulatorNewV = float;
  using TileShapeNewV = decltype(make_shape(HeadSizeV, BlkSeqKV, BlkSeqKV));
  using RefLayoutSK =
      decltype(make_layout(select<0, 2>(TileShapeNewV{}), LayoutRight{}));  // (dv, Blk)
  using DesiredLayoutKK = decltype(make_layout(select<1, 2>(TileShapeNewV{}), LayoutRight{}));  //
  using CollectiveMmaNewV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Element, RefLayoutSK, Alignment, Element,
      DesiredLayoutKK, Alignment, ElementAccumulatorKV, TileShapeNewV, ClusterShape, DummyStages,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  // FIXME: K@K^t are not exactly the same as Q@K^t, but similar enough
  using TiledMmaKK =
      typename CollectiveMmaQK::TiledMma;  // T = inv(I + strict_lower_triangular(K@K^t))
  using TiledMmaSK =
      decltype(convert_to_gmma_rs(typename CollectiveMmaSK::TiledMma{}));  // ??   = -S@K^t + V^t
  using TiledMmaNewV =
      decltype(convert_to_gmma_rs(typename CollectiveMmaNewV::TiledMma{}));  // NewV = ??@T^t

  static constexpr int TiledMmaKKNumThreads = size(TiledMmaKK{});
  static_assert(size(TiledMmaKK{}) == NumAuxMmaThreads);

  using GmemStrideAlphaBeta = Stride<int64_t, int32_t>;
  using GmemLayoutAlphaBeta = Layout<Shape<int64_t, int32_t>, GmemStrideAlphaBeta>;  // (seq, head)

  // (blk, pipe, cumsum_log/cumprod),
  //   0 for cumsum(log(alpha)) aka log(cumprod(alpha))
  //   1 for cumprod(alpha)
  //   2 for cumprod(alpha) * scale
  using AlphaCumSumLogIdx = _0;
  using AlphaCumProdIdx = _1;
  using AlphaCumProdScaleIdx = _2;

  using SmemLayoutAlpha =
      decltype(make_layout(make_shape(BlkSeqQ, Int<3>{}, Int<StagesAlphaBeta::value>{})));
  using SmemLayoutBeta = decltype(make_layout(make_shape(BlkSeqQ, Int<StagesAlphaBeta::value>{})));

  using MainloopQPipeline = cutlass::PipelineTmaAsync<StagesQ::value>;
  using MainloopKPipeline = cutlass::PipelineTmaAsync<StagesK::value>;
  using MainloopVPipeline = cutlass::PipelineTmaAsync<StagesV::value>;
  using MainloopOPipeline = typename CollectiveStoreO::Pipeline;

  using MainloopQKPipeline = cutlass::PipelineAsync<StagesQK::value>;
  using MainloopKKPipeline = cutlass::PipelineAsync<StagesKK::value>;

  using MainloopAlphaPipeline =
      std::conditional_t<NeedsAlpha, cutlass::PipelineAsync<StagesAlphaBeta::value>, Unused>;
  using MainloopBetaPipeline =
      std::conditional_t<NeedsBeta, cutlass::PipelineAsync<StagesAlphaBeta::value>, Unused>;

  using QPipelineState = typename cutlass::PipelineState<MainloopQPipeline::Stages>;
  using KPipelineState = typename cutlass::PipelineState<MainloopKPipeline::Stages>;
  using VPipelineState = typename cutlass::PipelineState<MainloopVPipeline::Stages>;
  using OPipelineState = typename CollectiveStoreO::PipelineState;

  using QKPipelineState = cutlass::PipelineState<MainloopQKPipeline::Stages>;
  using KKPipelineState = cutlass::PipelineState<MainloopKKPipeline::Stages>;

  using AlphaPipelineState =
      std::conditional_t<NeedsAlpha, cutlass::PipelineState<MainloopAlphaPipeline::Stages>, Unused>;
  using BetaPipelineState =
      std::conditional_t<NeedsBeta, cutlass::PipelineState<MainloopBetaPipeline::Stages>, Unused>;

  struct AlphaProcessor {
    CUTE_DEVICE
    AlphaProcessor(float scale) : scale_(scale) {}

    template <typename T>
    CUTE_DEVICE void operator()(T&& vecs) {
      constexpr int WarpSize = cutlass::NumThreadsPerWarp;
      int lane_id = cutlass::canonical_lane_idx();

      Tensor vecs_32 = flat_divide(
          std::forward<T>(vecs),
          make_tile(Int<WarpSize>{}));  // ((32), iter, cumsum_log/cumprod/cumprod_scale)
      Tensor vec_cumsum_log = vecs_32(make_coord(_), _, AlphaCumSumLogIdx{});
      Tensor vec_cumprod = vecs_32(make_coord(_), _, AlphaCumProdIdx{});
      Tensor vec_cumprod_s = vecs_32(make_coord(_), _, AlphaCumProdScaleIdx{});  // cumprod * scale
      Tensor frag = make_tensor<float>(size<1>(vec_cumprod));

      CUTE_UNROLL
      for (int iter = 0; iter < size(frag); ++iter) {
        frag(iter) = log2f(vec_cumsum_log(lane_id, iter) + 1e-10f);
      }

      CUTE_UNROLL
      for (int offset = 1; offset < WarpSize; offset *= 2) {
        CUTE_UNROLL
        for (int iter = 0; iter < size(frag); ++iter) {
          auto v = __shfl_up_sync(0xFFFFFFFF, frag(iter), offset);
          if (lane_id >= offset) {
            frag(iter) += v;
          }
        }
      }

      float sum = 0.0f;
      CUTE_UNROLL
      for (int iter = 1; iter < size(frag); ++iter) {
        sum += __shfl_sync(0xFFFFFFFF, frag(iter - 1), 31);
        frag(iter) += sum;
      }

      CUTE_UNROLL
      for (int iter = 0; iter < size(frag); ++iter) {
        vec_cumsum_log(lane_id, iter) = frag(iter);
        float cumprod = exp2f(frag(iter));
        vec_cumprod(lane_id, iter) = cumprod;
        vec_cumprod_s(lane_id, iter) = cumprod * scale_;
      }
    }

    float scale_ = 1.0f;
  };

  using BetaProcessor = Unused;
  // struct BetaProcessor {
  //   template <typename T>
  //   CUTE_DEVICE
  //   void operator()(T&& vec) {
  //     int lane_id = cutlass::canonical_lane_idx();
  //     int warp_size = cutlass::NumThreadsPerWarp;
  //     for (int i = lane_id; i < size(vec); i += warp_size) {
  //       auto val = vec(i);
  //       val = max(val, 1e-10f);  // clamp due to fusion with IKK before matrix inverse
  //       vec(i) = 1.0f / val;
  //     }
  //   }
  // };

  static constexpr int LoadQBytes = size(QKSmemLayoutQ{}(_, _, _0{})) * sizeof(Element);
  static constexpr int LoadKBytes = size(KVSmemLayoutK{}(_, _, _0{})) * sizeof(Element);
  static constexpr int LoadVBytes = size(KVSmemLayoutV{}(_, _, _0{})) * sizeof(Element);
  static constexpr int StoreOBytes = CollectiveStoreO::TmaTransactionBytes;

  using SharedStorageO = typename CollectiveStoreO::SharedStorage;

  struct SharedStorage {
    alignas(alignment_for_swizzle(
        QKSmemLayoutQ{})) cute::array_aligned<Element, cute::cosize_v<QKSmemLayoutQ>> smem_q;
    alignas(alignment_for_swizzle(
        KVSmemLayoutK{})) cute::array_aligned<Element, cute::cosize_v<KVSmemLayoutK>> smem_k;
    alignas(alignment_for_swizzle(
        KVSmemLayoutV{})) cute::array_aligned<Element, cute::cosize_v<KVSmemLayoutV>> smem_v;
    alignas(alignment_for_swizzle(
        SmemLayoutQK{})) cute::array_aligned<Element, cute::cosize_v<SmemLayoutQK>> smem_qk;
    alignas(alignment_for_swizzle(
        SmemLayoutKK{})) cute::array_aligned<InverseType, cute::cosize_v<SmemLayoutKK>> smem_kk;

    SharedStorageO smem_o;
    // TODO: make optional
    cute::array_aligned<float, cute::cosize_v<SmemLayoutBeta>> smem_beta;
    cute::array_aligned<float, cute::cosize_v<SmemLayoutAlpha>> smem_alpha;
  };

  using TMA_Q = typename CollectiveMmaQK::Params::TMA_A;
  using TMA_K = typename CollectiveMmaKV_G2S::Params::TMA_B;
  using TMA_V = typename CollectiveMmaKV_G2S::Params::TMA_A;
  using TMA_O = typename CollectiveStoreO::Params::TMA_O;

  using LoadQ = CollectiveLoadTma<LoadKind::kQ, MainloopQPipeline, Element, QKSmemLayoutQ, TMA_Q>;
  using LoadK = CollectiveLoadTma<LoadKind::kK, MainloopKPipeline, Element, KVSmemLayoutK, TMA_K>;
  using LoadV = CollectiveLoadTma<LoadKind::kV, MainloopVPipeline, Element, KVSmemLayoutV, TMA_V>;

  using LoadAlpha =
      CollectiveLoadVector<LoadKindVector::kAlpha, MainloopAlphaPipeline, float,
                           GmemLayoutAlphaBeta, float, SmemLayoutAlpha, AlphaProcessor>;
  using LoadBeta = CollectiveLoadVector<LoadKindVector::kBeta, MainloopBetaPipeline, float,
                                        GmemLayoutAlphaBeta, float, SmemLayoutBeta, BetaProcessor>;

  struct Arguments {  // clang-format off
    Element const* ptr_Q; LayoutQ dQ;
    Element const* ptr_K; LayoutK dK;
    Element const* ptr_V; LayoutV dV;
    Element*       ptr_O; LayoutO dO;
    float*        ptr_output_state; // layout fixed (kdim, vdim, num_heads, num_seqs):LayoutLeft{}
    float const*  ptr_input_state;
    float scale;
    float const* alpha_ptr; GmemStrideAlphaBeta alpha_stride;
    float const* beta_ptr;  GmemStrideAlphaBeta beta_stride;
  };  // clang-format on

  struct Params {
    TMA_Q tma_load_q;
    TMA_K tma_load_k;
    TMA_V tma_load_v;
    TMA_O tma_store_o;
    void* tensormaps;
    float scale;

    float* ptr_output_state;
    float const* ptr_input_state;

    float const* alpha_ptr;
    GmemLayoutAlphaBeta alpha_layout;
    float const* beta_ptr;
    GmemLayoutAlphaBeta beta_layout;
  };

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_size, Arguments const& args) {
    auto ratio = problem_size.num_q_heads > problem_size.num_v_heads
                     ? problem_size.num_q_heads / problem_size.num_v_heads
                     : problem_size.num_v_heads / problem_size.num_q_heads;

    constexpr bool IsGVAEnabled = find_option_t<Tag::kIsGVA, false_type, Options>::value;

    bool is_gqa_like = (problem_size.num_k_heads == problem_size.num_v_heads) &&
                       (problem_size.num_q_heads == ratio * problem_size.num_k_heads) &&
                       (problem_size.num_q_heads == ratio * problem_size.num_v_heads);

    bool is_gva_like = (problem_size.num_q_heads == problem_size.num_k_heads) &&
                       (problem_size.num_v_heads == ratio * problem_size.num_q_heads) &&
                       (problem_size.num_v_heads == ratio * problem_size.num_k_heads);
    return true && ((!IsGVAEnabled && is_gqa_like) || (IsGVAEnabled && is_gva_like)) &&
           (problem_size.head_size <= get<2>(TileShape{})) &&
           ((problem_size.head_size % Alignment) == 0);
  }

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_size, Arguments const& args,
                                        void* workspace) {
    int64_t s = problem_size.total_seqlen;
    int64_t t = problem_size.total_seqlen;
    int32_t d = problem_size.head_size;

    auto params_qk = CollectiveMmaQK::to_underlying_arguments(
        make_shape(s, t, d, problem_size.num_q_heads),
        typename CollectiveMmaQK::Arguments{
            args.ptr_Q, args.dQ, args.ptr_K, args.dK,  // never used, dummy
        },
        /*workspace=*/nullptr);

    auto params_kv_k = CollectiveMmaKV_G2S::to_underlying_arguments(
        make_shape(d, d, s, problem_size.num_k_heads),
        typename CollectiveMmaKV_G2S::Arguments{
            args.ptr_V, select<1, 0, 2>(args.dV),  // not used
            args.ptr_K, select<1, 0, 2>(args.dK),  // used as G2S for K
        },
        /*workspace=*/nullptr);

    auto params_kv_v = CollectiveMmaKV_G2S::to_underlying_arguments(
        make_shape(d, d, s, problem_size.num_v_heads),
        typename CollectiveMmaKV_G2S::Arguments{
            args.ptr_V, select<1, 0, 2>(args.dV),  // used as G2S for V
            args.ptr_K, select<1, 0, 2>(args.dK),  // not used
        },
        /*workspace=*/nullptr);

    auto params_o = CollectiveStoreO::to_underlying_arguments(
        make_shape(d, s, d, problem_size.num_o_heads),  // in O1
        // make_shape(d, s, s, problem_size.num_o_heads),  // in O2
        typename CollectiveStoreO::Arguments{args.ptr_O, select<1, 0, 2>(args.dO)}, workspace);

    return Params{
        .tma_load_q = params_qk.tma_load_a,
        .tma_load_k = params_kv_k.tma_load_b,
        .tma_load_v = params_kv_v.tma_load_a,
        .tma_store_o = params_o.tma_store_o,
        .tensormaps = params_o.tensormaps,
        .scale = args.scale,

        .ptr_output_state = args.ptr_output_state,
        .ptr_input_state = args.ptr_input_state,

        // TODO: refactor all name to varname_vartype
        .alpha_ptr = args.alpha_ptr,
        .alpha_layout = make_layout(make_shape(s, problem_size.num_sab_heads), args.alpha_stride),
        .beta_ptr = args.beta_ptr,
        .beta_layout = make_layout(make_shape(s, problem_size.num_sab_heads), args.beta_stride),
    };
  }

  static size_t get_workspace_size(Arguments const& args, int sm_count) {
    return CollectiveStoreO::get_workspace_size(sm_count);
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const& problem_shape,
                                              Arguments const& args, void* workspace,
                                              cudaStream_t stream) {
    return CollectiveStoreO::initialize_workspace(problem_shape, workspace, stream);
  }

  CUTE_DEVICE static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_k.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_v.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_store_o.get_tma_descriptor());
  }

  template <typename ProblemShape, typename LoadTileShape, typename WorkDesc>
  CUTE_DEVICE void load_qkv(Params const& params, ProblemShape const& problem_size,
                            LoadTileShape const& load_tile_shape, WorkDesc const& work_desc,
                            MainloopQPipeline& q_pipeline, QPipelineState& q_smem_pipe_write,
                            MainloopKPipeline& k_pipeline, KPipelineState& k_smem_pipe_write,
                            MainloopVPipeline& v_pipeline, VPipelineState& v_smem_pipe_write,
                            SharedStorage& storage) {
    int32_t num_blocks = ceil_div(work_desc.seq_len, get<0>(TileShape{}));
    uint32_t lane_predicate = cute::elect_one_sync();

    auto q_collective_load = LoadQ(params.tma_load_q, q_pipeline, storage.smem_q);
    auto k_collective_load = LoadK(params.tma_load_k, k_pipeline, storage.smem_k);
    auto v_collective_load = LoadV(params.tma_load_v, v_pipeline, storage.smem_v);

    auto q_src_dst = q_collective_load.partition_SD(problem_size, load_tile_shape, work_desc);
    auto k_src_dst = k_collective_load.partition_SD(problem_size, load_tile_shape, work_desc);
    auto v_src_dst = v_collective_load.partition_SD(problem_size, load_tile_shape, work_desc);

    CUTE_NO_UNROLL
    for (int blk = 0; blk < num_blocks; ++blk) {
      k_collective_load.step(k_src_dst, blk, k_smem_pipe_write, lane_predicate);
      q_collective_load.step(q_src_dst, blk, q_smem_pipe_write, lane_predicate);
      v_collective_load.step(v_src_dst, blk, v_smem_pipe_write, lane_predicate);
    }
  }

  template <typename ProblemShape, typename TileShape, typename WorkDesc>
  CUTE_DEVICE void load_beta(Params const& params, ProblemShape const& problem_size,
                             TileShape const& tile_shape, WorkDesc const& work_desc,
                             MainloopBetaPipeline& pipeline, BetaPipelineState& smem_pipe_write,
                             SharedStorage& storage) {
    int32_t num_blocks = ceil_div(work_desc.seq_len, get<0>(TileShape{}));

    // fuse post inverse diag(beta) into diagonal of IKK
    // auto collective_load = LoadBeta{params.beta_ptr, params.beta_layout, /*oob_value=*/1.0f,
    // pipeline, storage.smem_beta};
    auto collective_load = LoadBeta{params.beta_ptr, params.beta_layout, /*oob_value=*/0.0f,
                                    pipeline, storage.smem_beta};
    auto src_dst = collective_load.partition_SD(problem_size, tile_shape, work_desc);

    CUTE_NO_UNROLL
    for (int blk = 0; blk < num_blocks - 1; ++blk) {
      collective_load.step</*IsTail=*/false>(src_dst, blk, smem_pipe_write, num_blocks);
    }
    collective_load.step</*IsTail=*/true>(src_dst, num_blocks - 1, smem_pipe_write, num_blocks);
  }

  template <typename ProblemShape, typename TileShape, typename WorkDesc>
  CUTE_DEVICE void load_alpha(Params const& params, ProblemShape const& problem_size,
                              TileShape const& tile_shape, WorkDesc const& work_desc,
                              MainloopAlphaPipeline& pipeline, AlphaPipelineState& smem_pipe_write,
                              SharedStorage& storage) {
    int32_t num_blocks = ceil_div(work_desc.seq_len, get<0>(TileShape{}));

    auto collective_load = LoadAlpha{params.alpha_ptr, params.alpha_layout, /*oob_value=*/1.0f,
                                     pipeline, storage.smem_alpha};
    auto src_dst = collective_load.partition_SD(problem_size, tile_shape, work_desc);

    typename LoadAlpha::VectorProcessor processor{params.scale};

    CUTE_NO_UNROLL
    for (int blk = 0; blk < num_blocks - 1; ++blk) {
      collective_load.step</*IsTail=*/false>(src_dst, blk, smem_pipe_write, num_blocks, processor);
    }
    collective_load.step</*IsTail=*/true>(src_dst, num_blocks - 1, smem_pipe_write, num_blocks,
                                          processor);
  }

  template <typename ProblemSize, typename StoreTileShape, typename WorkDesc,
            typename PipelineState>
  CUTE_DEVICE void store(TMA_O const& tma_store, void* tensormaps, ProblemSize const& problem_size,
                         StoreTileShape const& store_tile_shape, WorkDesc const& work_desc,
                         MainloopOPipeline& pipeline, PipelineState& smem_pipe_read,
                         SharedStorageO& storage) {
    int32_t num_blocks = ceil_div(work_desc.seq_len, get<0>(TileShape{}));
    uint32_t lane_predicate = cute::elect_one_sync();

    auto collective_store = CollectiveStoreO{tma_store, pipeline, storage, tensormaps};
    auto src_dst = collective_store.partition_SD(problem_size, store_tile_shape, work_desc);

    CUTE_NO_UNROLL
    for (int blk = 0; blk < num_blocks; ++blk) {
      DPRINTF0_W("O collective_store.step smem_pipe_read:%d -> blk_idx:%d, num_blocks:%d\n",
                 smem_pipe_read.index(), blk, num_blocks);
      collective_store.step(problem_size, work_desc, src_dst, smem_pipe_read, blk, num_blocks,
                            lane_predicate);
    }
  }

  template <class ProblemShape, class WorkDesc>
  CUTE_DEVICE void compute(
      Params const& params, ProblemShape const& problem_size, WorkDesc const& work_desc,
      MainloopQPipeline& q_pipeline, QPipelineState& q_smem_pipe_read,
      MainloopKPipeline& k_pipeline, KPipelineState& k_smem_pipe_read,
      MainloopVPipeline& v_pipeline, VPipelineState& v_smem_pipe_read,
      MainloopOPipeline& o_pipeline, OPipelineState& o_smem_pipe_write,
      MainloopQKPipeline& qk_pipeline, QKPipelineState& qk_smem_pipe_read,
      MainloopKKPipeline& kk_pipeline, KKPipelineState& kk_smem_pipe_read,
      MainloopAlphaPipeline& alpha_pipeline, AlphaPipelineState& alpha_smem_pipe_read,
      // MainloopBetaPipeline& beta_pipeline, BetaPipelineState& beta_smem_pipe_read,
      OrderedMathBarriers& math_barriers, SharedStorage& storage) {
    // MAKE NVCC HAPPY!
    constexpr auto zero = Element{};

    int32_t num_blocks = ceil_div(work_desc.seq_len, get<0>(TileShape{}));
    DPRINTF0_WG("num_blocks: %d\n", num_blocks);

    int thread_idx = int(threadIdx.x) - NumLoadThreads;
    int warpgroup_idx = thread_idx / cutlass::NumThreadsPerWarpGroup;

    float scale = params.scale;

    // Tensor Beta  = make_tensor(make_smem_ptr(storage.smem_beta.data()), SmemLayoutBeta{});
    Tensor Alpha = make_tensor(make_smem_ptr(storage.smem_alpha.data()), SmemLayoutAlpha{});

    Tensor sQqk = make_tensor(make_smem_ptr(storage.smem_q.data()), QKSmemLayoutQ{});
    Tensor sKqk = make_tensor(make_smem_ptr(storage.smem_k.data()), QKSmemLayoutK{});
    Tensor sKkv = make_tensor(make_smem_ptr(storage.smem_k.data()), KVSmemLayoutK{});
    Tensor sVkv = make_tensor(make_smem_ptr(storage.smem_v.data()), KVSmemLayoutV{});
    Tensor sQK = make_tensor(make_smem_ptr(storage.smem_qk.data()), SmemLayoutQK{});
    Tensor sO = make_tensor(make_smem_ptr(storage.smem_o.data()), SmemLayoutO{});

    static_assert(sizeof(InverseType) == sizeof(Element));
    Tensor sKK_inv = make_tensor(make_smem_ptr(storage.smem_kk.data()), SmemLayoutKK{});
    Tensor sKK_opd = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(storage.smem_kk.data())),
                                 SmemLayoutKK{});

    ///////////////////////////////////////////////////////////////////////////
    // S@K  (-S K^T  +  V^T)
    auto sk_tiled_mma = TiledMmaSK{};
    auto sk_thr_mma = sk_tiled_mma.get_thread_slice(thread_idx);

    auto layout_SKAlpha = flatten(make_layout(  // broadcast Alpha vector to SK size
        make_layout(select<0, 1>(TileShapeSK{}), Stride<_0, _1>{}),  // (D, Blk_KV)
        select<1, 2>(SmemLayoutAlpha{})                              // (Idx, pipe)
        ));                                                          // (D, Blk_KV, Idx, pipe)

    auto tSKrAlpha = sk_thr_mma.partition_C(Alpha.compose(layout_SKAlpha))(
        _, _, _, AlphaCumProdIdx{}, _);  // (frag, iter_D, iter_Blk_Q, pipe)

    // tSKrV adds to tSKrSK (acc)
    using SK_V_S2R = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
    auto tSKrV_tiled_copy = make_tiled_copy_C(SK_V_S2R{}, sk_tiled_mma);
    auto tSKrV_thr_copy = tSKrV_tiled_copy.get_thread_slice(thread_idx);

    Tensor tSKsK = sk_thr_mma.partition_B(sKqk);
    Tensor tSKrK = sk_thr_mma.make_fragment_B(tSKsK);

    ///////////////////////////////////////////////////////////////////////////
    // NewV = (S@K result) @ T^t
    auto newv_tiled_mma = TiledMmaNewV{};
    auto newv_thr_mma = newv_tiled_mma.get_thread_slice(thread_idx);

    Tensor tNewVsB = newv_thr_mma.partition_B(sKK_opd);
    Tensor tNewVrB = newv_thr_mma.make_fragment_B(tNewVsB);

    ///////////////////////////////////////////////////////////////////////////
    // K@V
    auto kv_tiled_mma = TiledMmaKV{};
    auto kv_thr_mma = kv_tiled_mma.get_thread_slice(thread_idx);

    Tensor tKVrKV = partition_fragment_C(kv_thr_mma, select<0, 1>(TileShapeKV{}));

    // Tensor tKVrV    = kv_thr_mma.partition_fragment_A(sVkv(_, _, _0{}));  // mma src
    // Tensor tKVrV_cv = tKVrV_thr_copy.retile_D(tKVrV);                     // copy view dst
    // Tensor tKVsV    = tKVrV_thr_copy.partition_S(sVkv);                   // copy view src

    Tensor tKVsK = kv_thr_mma.partition_B(sKkv);
    Tensor tKVrK = kv_thr_mma.make_fragment_B(tKVsK);

    auto const cV = make_identity_tensor(Shape<Int<HeadSizeV>, Int<BlkSeqKV>>{});
    Tensor tKVcV = kv_thr_mma.partition_A(cV);

    ///////////////////////////////////////////////////////////////////////////
    // Q@K@V
    auto o1_tiled_mma = TiledMmaO1{};
    auto o1_thr_mma = o1_tiled_mma.get_thread_slice(thread_idx);
    auto o2_tiled_mma = TiledMmaO2{};
    auto o2_thr_mma = o2_tiled_mma.get_thread_slice(thread_idx);

    // A1 for Q@(KV)
    // Tensor tOrKV = make_acc_into_op<Element>(tKVrKV, typename TiledMmaO1::LayoutA_TV{});
    // B1 for Q@(KV)
    Tensor tOsQ = o1_thr_mma.partition_B(sQqk);
    Tensor tOrQ = o1_thr_mma.make_fragment_B(tOsQ);

    // A2 for QK@V
    // Tensor tOsV = o2_thr_mma.partition_A(sVkv);
    // Tensor tOrV = o2_thr_mma.make_fragment_A(tOsV);
    // B2 for QK@V
    Tensor tOsQK = o2_thr_mma.partition_B(sQK);
    Tensor tOrQK = o2_thr_mma.make_fragment_B(tOsQK);

    using O_R2S = typename CollectiveStoreO::CopyAtomR2S;
    auto tiled_copy_o = make_tiled_copy_C(O_R2S{}, o1_tiled_mma);
    auto thr_copy_o = tiled_copy_o.get_thread_slice(thread_idx);
    auto tOsO = thr_copy_o.partition_D(sO);

    auto const cO = make_identity_tensor(Shape<Int<HeadSizeQK>, Int<BlkSeqQ>>{});
    Tensor tOcO = o1_thr_mma.partition_C(cO);

    auto layout_OAlpha = flatten(make_layout(  // broadcast Alpha vector to O size
        make_layout(select<0, 1>(TileShapeO1{}), Stride<_0, _1>{}),  // (D, Blk_Q)
        select<1, 2>(SmemLayoutAlpha{})                              // (Idx, pipe)
        ));                                                          // (D, Blk_Q, Idx, pipe)

    auto tOrAlphaScale = o1_thr_mma.partition_C(Alpha.compose(layout_OAlpha))(
        _, _, _, AlphaCumProdScaleIdx{}, _);  // (frag, iter_D, iter_Blk_Q, pipe)

    auto const seq_idx = work_desc.seq_idx;
    auto const q_head_idx = work_desc.q_head_idx();
    auto const k_head_idx = work_desc.k_head_idx();
    auto const v_head_idx = work_desc.v_head_idx();

    auto sk_epi = [&](auto& tSKrSK, auto const& alpha_smem_pipe_read) INLINE_LAMBDA {
      if constexpr (NeedsAlpha) {
        transform(tSKrSK, tSKrAlpha(_, _, _, alpha_smem_pipe_read.index()), tSKrSK,
                  [&](auto sk, auto coeff) { return sk * coeff; });
      }
    };

    auto sk_load_v = [&](int pipe_idx) INLINE_LAMBDA {
      Tensor tSKrV = make_fragment_like<Element>(
          partition_fragment_C(sk_thr_mma, sVkv(_, _, _0{})));  // mma acc
      Tensor tSKrV_cv = tSKrV_thr_copy.retile_D(tSKrV);         // copy view dst
      Tensor tSKsV = tSKrV_thr_copy.partition_S(sVkv);          // copy view src
      copy(tSKrV_tiled_copy, tSKsV(_, _, _, pipe_idx), tSKrV_cv);
      return tSKrV;
    };

    auto kv_decay_v = [&](auto& tKVrV, auto const& alpha_smem_pipe_read, auto is_final_block_,
                          auto B) INLINE_LAMBDA {
      constexpr bool is_final_block = decltype(is_final_block_)::value;
      if constexpr (NeedsAlpha) {
        Tensor Alpha_cumsum_log = Alpha(_, AlphaCumSumLogIdx{}, alpha_smem_pipe_read.index());
        float block_coeff_log = Alpha_cumsum_log(B - 1);
        cute::transform(tKVrV, tKVcV, tKVrV, [&](auto val, auto coord) {
          auto tok = get<1>(coord);
          float coeff = [&] {
            if constexpr (!is_final_block) {
              return exp2f(block_coeff_log - Alpha_cumsum_log(tok));
            } else {
              return tok < B ? exp2f(block_coeff_log - Alpha_cumsum_log(tok)) : 0.0f;
            }
          }();
          return decltype(val)(val * coeff);
        });
      }
      if constexpr (is_final_block) {
        if constexpr (!NeedsAlpha) {
          cute::transform(tKVrV, tKVcV, tKVrV, [&](auto val, auto coord) {
            auto tok = get<1>(coord);
            return tok < B ? val : zero;  // mask v of tail oob values
          });
        }
      }
    };

    auto kv_load = [&](auto& tKVrKV) INLINE_LAMBDA {
      DPRINTF0_WG("[%d,%d,%d,%d]>> load tKVgKV -> tKVrKV\n", seq_idx, q_head_idx, k_head_idx,
                  v_head_idx);
      int num_state_heads = problem_size.num_sab_heads;
      int state_head_idx = work_desc.o_head_idx();
      auto gKV = make_tensor(make_gmem_ptr(params.ptr_input_state),
                             make_layout(make_shape(Int<HeadSizeQK>{}, Int<HeadSizeV>{},
                                                    num_state_heads, problem_size.num_seqs)))(
          _, _, state_head_idx, seq_idx);  // (KDim, VDim), K-contiguous

      auto tiled_copy_kv =
          make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, Element>{}, kv_tiled_mma);
      auto thr_copy_kv = tiled_copy_kv.get_thread_slice(thread_idx);

      auto tKVgKV = thr_copy_kv.partition_S(select_tensor<1, 0>(gKV));
      copy(tiled_copy_kv, tKVgKV, tKVrKV);
    };

    auto kv_store = [&]() INLINE_LAMBDA {  // tKVrKV is carried over whole mainloop
      DPRINTF0_WG("[%d,%d,%d,%d]>> save tKVrKV -> tKVgKV\n", seq_idx, q_head_idx, k_head_idx,
                  v_head_idx);
      int num_state_heads = problem_size.num_sab_heads;
      int state_head_idx = work_desc.o_head_idx();  // num_o_heads == num_sab_heads
      auto gKV = make_tensor(make_gmem_ptr(params.ptr_output_state),
                             make_layout(make_shape(Int<HeadSizeQK>{}, Int<HeadSizeV>{},
                                                    num_state_heads, problem_size.num_seqs)))(
          _, _, state_head_idx, seq_idx);  // (KDim, VDim), K-contiguous

      auto tiled_copy_kv =
          make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, Element>{}, kv_tiled_mma);
      auto thr_copy_kv = tiled_copy_kv.get_thread_slice(thread_idx);

      auto tKVgKV = thr_copy_kv.partition_D(select_tensor<1, 0>(gKV));
      copy(tiled_copy_kv, tKVrKV, tKVgKV);
    };

    auto o1_epi = [&](auto& tOrO1, auto const& alpha_smem_pipe_read) INLINE_LAMBDA {
      if constexpr (NeedsAlpha) {
        auto tOrAlphaScale_ = tOrAlphaScale(_, _, _, alpha_smem_pipe_read.index());
        CUTE_UNROLL
        for (int i = 0; i < size(tOrO1); ++i) {
          tOrO1(i) = tOrAlphaScale_(i) * tOrO1(i);
        }
      } else {
        CUTE_UNROLL
        for (int i = 0; i < size(tOrO1); ++i) {
          tOrO1(i) = scale * tOrO1(i);
        }
      }
    };

    auto o_store = [&](auto tOrO) INLINE_LAMBDA {
      auto tOrO_cvt = make_fragment_like<ElementO>(tOrO);
      copy(tOrO, tOrO_cvt);

      DPRINTF0_WG("compute: o_pipeline.producer_wait: smem_pipe_write:%d\n",
                  o_smem_pipe_write.index());
      o_pipeline.producer_acquire(o_smem_pipe_write);
      Tensor tOrO_cvt_cv = thr_copy_o.retile_S(tOrO_cvt);
      cutlass::arch::fence_view_async_shared();
      copy(tiled_copy_o, tOrO_cvt_cv, tOsO(_, _, _, o_smem_pipe_write.index()));
      cutlass::arch::fence_view_async_shared();
      o_pipeline.producer_commit(o_smem_pipe_write);
      ++o_smem_pipe_write;
    };

    auto compute_loop_body = [&](int blk, auto is_first_block_,
                                 auto is_final_block_) INLINE_LAMBDA {
      constexpr bool is_first_block = decltype(is_first_block_)::value;
      constexpr bool is_final_block = decltype(is_final_block_)::value;
      int B = is_final_block ? valid_seq_len(work_desc, blk) : BlkSeqKV;

      // 2.1 Q @ KV, NOTE: use old KV here
      DPRINTF0_WG("compute: q_pipeline.consumer_wait: smem_pipe_read:%d\n",
                  q_smem_pipe_read.index());
      q_pipeline.consumer_wait(q_smem_pipe_read);
      if constexpr (NeedsAlpha) {
        alpha_pipeline.consumer_wait(alpha_smem_pipe_read);
      }

      DPRINTF0_WG("[%d,%d,%d,%d]** dispatch O WGMMA\n", seq_idx, q_head_idx, k_head_idx,
                  v_head_idx);
      auto tOrO = partition_fragment_C(o1_thr_mma, select<0, 1>(TileShapeO1{}));
      if constexpr (is_first_block) {
        DPRINTF0_WG("compute: q_pipeline.consumer_release: smem_pipe_read:%d\n",
                    q_smem_pipe_read.index());
        q_pipeline.consumer_release(q_smem_pipe_read);
        ++q_smem_pipe_read;
      } else {
        Tensor tOrKV = make_acc_into_op<Element>(tKVrKV, typename TiledMmaO1::LayoutA_TV{});
        warpgroup_fence_operand(tOrKV);
        warpgroup_fence_operand(tOrO);
        math_barriers.ordered_or_wait(warpgroup_idx);
        warpgroup_arrive();
        gemm_zero_acc(o1_thr_mma, tOrKV, tOrQ(_, _, _, q_smem_pipe_read.index()), tOrO);
        warpgroup_commit_batch();  // q@kv batch
        math_barriers.notify_next_blocked(warpgroup_idx);
      }
      if constexpr (!is_first_block) {
        warpgroup_wait<0>();  // q@kv batch
        DPRINTF0_WG("compute: q_pipeline.consumer_release: smem_pipe_read:%d\n",
                    q_smem_pipe_read.index());
        q_pipeline.consumer_release(q_smem_pipe_read);
        ++q_smem_pipe_read;
        o1_epi(tOrO, alpha_smem_pipe_read);
      }

      DPRINTF0_WG("compute: k_pipeline.consumer_wait: smem_pipe_read:%d\n",
                  k_smem_pipe_read.index());
      k_pipeline.consumer_wait(k_smem_pipe_read);

      auto tSKrSK = partition_fragment_C(sk_thr_mma, sVkv(_, _, _0{}));
      if constexpr (!is_first_block) {
        auto tSKrS = make_acc_into_op<Element>(tKVrKV, typename TiledMmaSK::LayoutA_TV{});
        warpgroup_fence_operand(tSKrSK);
        warpgroup_fence_operand(tSKrS);
        math_barriers.ordered_or_wait(warpgroup_idx);
        warpgroup_arrive();
        gemm_zero_acc(sk_tiled_mma, tSKrS, tSKrK(_, _, _, k_smem_pipe_read.index()), tSKrSK);
        warpgroup_commit_batch();
        math_barriers.notify_next_blocked(warpgroup_idx);
        warpgroup_wait<0>();
      }

      DPRINTF0_WG("compute: v_pipeline.consumer_wait: smem_pipe_read:%d\n",
                  v_smem_pipe_read.index());
      v_pipeline.consumer_wait(v_smem_pipe_read);
      auto tSKrV = sk_load_v(v_smem_pipe_read.index());
      if constexpr (!is_first_block) {
        sk_epi(tSKrSK, alpha_smem_pipe_read);
        transform(tSKrV, tSKrSK, tSKrV, [](auto v, auto sk) { return v - Element(sk); });
      }

      kk_pipeline.consumer_wait(kk_smem_pipe_read);
      auto tNewVrA = make_acc_into_op<Element>(tSKrV, typename TiledMmaNewV::LayoutA_TV{});
      auto tNewVrC = partition_fragment_C(newv_thr_mma, select<0, 1>(TileShapeNewV{}));
      warpgroup_fence_operand(tNewVrA);
      warpgroup_fence_operand(tNewVrC);
      math_barriers.ordered_or_wait(warpgroup_idx);
      warpgroup_arrive();
      gemm_zero_acc(o1_thr_mma, tNewVrA, tNewVrB(_, _, _, kk_smem_pipe_read.index()), tNewVrC);
      warpgroup_commit_batch();  // new_v batch
      math_barriers.notify_next_blocked(warpgroup_idx);
      warpgroup_wait<0>();  // new_v batch
      DPRINTF0_WG("compute: v_pipeline.consumer_release: smem_pipe_read:%d\n",
                  v_smem_pipe_read.index());
      ++v_smem_pipe_read;  // NOTE: if we delay this increment after consumer_release, race
                           // condition happens, why?
      v_pipeline.consumer_release(v_smem_pipe_read);

      kk_pipeline.consumer_release(kk_smem_pipe_read);
      ++kk_smem_pipe_read;

      /////////////////////////////////////////////////////////////////////////
      // 2. compute qkv
      // 2.2 QK @ V, NOTE: use old KV here and QK is scaled
      qk_pipeline.consumer_wait(qk_smem_pipe_read);
      auto tOrV_or_tKVrV = make_acc_into_op<Element>(tNewVrC, typename TiledMmaKV::LayoutA_TV{});
      warpgroup_fence_operand(tOrV_or_tKVrV);
      warpgroup_fence_operand(tOrO);
      math_barriers.ordered_or_wait(warpgroup_idx);
      warpgroup_arrive();
      if constexpr (is_first_block) {
        gemm_zero_acc(o2_tiled_mma, tOrV_or_tKVrV, tOrQK(_, _, _, qk_smem_pipe_read.index()), tOrO);
      } else {
        gemm(o2_tiled_mma, tOrV_or_tKVrV, tOrQK(_, _, _, qk_smem_pipe_read.index()), tOrO);
      }
      warpgroup_commit_batch();  // qk@v batch
      math_barriers.notify_next_blocked(warpgroup_idx);
      warpgroup_wait<0>();  // qk@v batch
      qk_pipeline.consumer_release(qk_smem_pipe_read);
      ++qk_smem_pipe_read;
      o_store(tOrO);

      /////////////////////////////////////////////////////////////////////////
      // 3. update KV
      float block_coeff = 1.0f;
      if constexpr (NeedsAlpha) {
        block_coeff = Alpha(B - 1, AlphaCumProdIdx{}, alpha_smem_pipe_read.index());
      }

      cute::transform(tKVrKV, [&](auto kv) { return block_coeff * kv; });
      kv_decay_v(tOrV_or_tKVrV, alpha_smem_pipe_read, is_final_block_, B);

      DPRINTF0_WG("[%d,%d,%d,%d]** dispatch KV WGMMA\n", seq_idx, q_head_idx, k_head_idx,
                  v_head_idx);
      warpgroup_fence_operand(tOrV_or_tKVrV);
      warpgroup_fence_operand(tKVrKV);
      math_barriers.ordered_or_wait(warpgroup_idx);
      warpgroup_arrive();
      gemm(kv_tiled_mma, tOrV_or_tKVrV, tKVrK(_, _, _, k_smem_pipe_read.index()), tKVrKV);
      warpgroup_commit_batch();  // k@v batch
      math_barriers.notify_next_blocked(warpgroup_idx);
      warpgroup_wait<0>();

      DPRINTF0_WG("compute: k_pipeline.consumer_release: smem_pipe_read:%d\n",
                  k_smem_pipe_read.index());
      k_pipeline.consumer_release(k_smem_pipe_read);
      ++k_smem_pipe_read;

      if constexpr (NeedsAlpha) {
        alpha_pipeline.consumer_release(alpha_smem_pipe_read);
        ++alpha_smem_pipe_read;
      }
    };

    if constexpr (!kInitStateFromInput) {
      clear(tKVrKV);
      compute_loop_body(0, /*is_first_block_=*/cute::true_type{},
                        /*is_final_block_=*/cute::true_type{});
    } else {
      kv_load(tKVrKV);
      compute_loop_body(0, /*is_first_block_=*/cute::false_type{},
                        /*is_final_block_=*/cute::true_type{});
    }
    CUTE_NO_UNROLL
    for (int blk = 1; blk < num_blocks - 1; ++blk) {
      compute_loop_body(blk, /*is_first_block_=*/cute::false_type{},
                        /*is_final_block_=*/cute::false_type{});
    }
    if (num_blocks != 1) {
      compute_loop_body(num_blocks - 1, /*is_first_block_=*/cute::false_type{},
                        /*is_final_block_=*/cute::true_type{});
    }
    kv_store();
  }

  template <class ProblemShape, class WorkDesc>
  CUTE_DEVICE void compute_aux(Params const& params, ProblemShape const& problem_size,
                               WorkDesc const& work_desc, MainloopQPipeline& q_pipeline,
                               QPipelineState& q_smem_pipe_read, MainloopKPipeline& k_pipeline,
                               KPipelineState& k_smem_pipe_read, MainloopQKPipeline& qk_pipeline,
                               QKPipelineState& qk_smem_pipe_write, MainloopKKPipeline& kk_pipeline,
                               KKPipelineState& kk_smem_pipe_write,
                               MainloopAlphaPipeline& alpha_pipeline,
                               AlphaPipelineState& alpha_smem_pipe_read,
                               MainloopBetaPipeline& beta_pipeline,
                               BetaPipelineState& beta_smem_pipe_read, SharedStorage& storage) {
    int thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    float scale = params.scale;

    Tensor Beta = make_tensor(make_smem_ptr(storage.smem_beta.data()), SmemLayoutBeta{});
    Tensor Alpha = make_tensor(make_smem_ptr(storage.smem_alpha.data()), SmemLayoutAlpha{});

    Tensor sQqk = make_tensor(make_smem_ptr(storage.smem_q.data()), QKSmemLayoutQ{});
    Tensor sKqk = make_tensor(make_smem_ptr(storage.smem_k.data()), QKSmemLayoutK{});
    Tensor sKkv = make_tensor(make_smem_ptr(storage.smem_k.data()), KVSmemLayoutK{});
    Tensor sVkv = make_tensor(make_smem_ptr(storage.smem_v.data()), KVSmemLayoutV{});
    Tensor sQK = make_tensor(make_smem_ptr(storage.smem_qk.data()), SmemLayoutQK{});
    Tensor sO = make_tensor(make_smem_ptr(storage.smem_o.data()), SmemLayoutO{});

    static_assert(sizeof(InverseType) == sizeof(Element));
    Tensor sKK_inv = make_tensor(make_smem_ptr(storage.smem_kk.data()), SmemLayoutKK{});
    Tensor sKK_opd = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(storage.smem_kk.data())),
                                 SmemLayoutKK{});

    ///////////////////////////////////////////////////////////////////////////
    // Q@K
    auto qk_tiled_mma = TiledMmaQK{};
    auto qk_thr_mma = qk_tiled_mma.get_thread_slice(thread_idx);

    Tensor tQKsQ = qk_thr_mma.partition_A(sQqk);
    Tensor tQKsK = qk_thr_mma.partition_B(sKqk);
    Tensor tQKrQ = qk_thr_mma.make_fragment_A(tQKsQ);
    Tensor tQKrK = qk_thr_mma.make_fragment_B(tQKsK);

    auto cMqk = make_identity_tensor(select<0, 1>(TileShapeQK{}));  // (QTok, KTok)
    auto tQKcMqk = qk_thr_mma.partition_C(cMqk);                    // (idx) -> (tok_q, tok_k)

    ///////////////////////////////////////////////////////////////////////////
    // K@K  (basically I + strict_lower_triangular(K K^T)
    auto kk_tiled_mma = TiledMmaKK{};
    auto kk_thr_mma = kk_tiled_mma.get_thread_slice(thread_idx);
    Tensor tKKsK = kk_thr_mma.partition_B(sKqk);
    Tensor tKKrA = kk_thr_mma.make_fragment_A(tKKsK);
    Tensor tKKrB = kk_thr_mma.make_fragment_B(tKKsK);

    auto const& cMkk = cMqk;
    auto tKKcMkk = kk_thr_mma.partition_C(cMkk);

    auto const seq_idx = work_desc.seq_idx;
    auto const q_head_idx = work_desc.q_head_idx();
    auto const k_head_idx = work_desc.k_head_idx();
    auto const v_head_idx = work_desc.v_head_idx();

    auto qk_and_kk_epi = [&](auto& tQKrQK, auto& tKKrKK, auto const& alpha_smem_pipe_read,
                             auto const& beta_smem_pipe_read, auto is_final_block_,
                             auto B /*valid seqlen*/) {
      if constexpr (NeedsAlpha) {
        Tensor Alpha_cumsum_log = Alpha(_, AlphaCumSumLogIdx{}, alpha_smem_pipe_read.index());
        for_each(make_int_sequence<size(tKKcMkk)>{}, [&](auto i) {
          auto coord = tQKcMqk(i);
          auto [s, t] = coord;
          float alpha = exp2f(Alpha_cumsum_log(s) - Alpha_cumsum_log(t));
          tQKrQK(i) *= alpha * scale;
          tKKrKK(i) *= alpha;
        });
      } else {
        transform(tQKrQK, [scale](auto v) { return v * scale; });
      }

      if constexpr (NeedsBeta) {
        Tensor Beta_ = Beta(_, beta_smem_pipe_read.index());
        for_each(make_int_sequence<size(tKKcMkk)>{}, [&](auto i) {
          auto coord = tQKcMqk(i);
          auto [s, t] = coord;
          tKKrKK(i) *= Beta_(s);
        });
      }

      constexpr bool is_final_block = decltype(is_final_block_)::value;
      for_each(make_int_sequence<size(tKKcMkk)>{}, [&](auto i) {
        auto coord = tQKcMqk(i);
        auto [s, t] = coord;
        bool pred = s >= t;
        tQKrQK(i) = pred ? tQKrQK(i) : 0.0f;
        tKKrKK(i) =
            pred ? tKKrKK(i) : 0.0f;  // diagonal is garbage filled, will process during inversion
        if constexpr (is_final_block) {
          bool pred = s < B || t < B;
          tQKrQK(i) = pred ? tQKrQK(i) : 0.0f;
          tKKrKK(i) = pred ? tKKrKK(i) : 0.0f;
        }
      });
    };

    auto qk_store = [&](auto tQKrQK, auto const& qk_smem_pipe_write) {
      auto sQK_pipe_slice = sQK(_, _, qk_smem_pipe_write.index());

      static_assert(sizeof(Element) == 2);
      using CopyOpR2S = SM90_U32x4_STSM_N;
      auto tiled_copy_qk = make_tiled_copy_C(Copy_Atom<CopyOpR2S, Element>{}, qk_tiled_mma);
      auto thr_copy_qk = tiled_copy_qk.get_thread_slice(thread_idx);
      auto tQKsQK = thr_copy_qk.partition_D(sQK_pipe_slice);
      auto tQKrQK_cv = thr_copy_qk.retile_S(tQKrQK);
      auto tQKrQK_cvt_cv = make_fragment_like<Element>(tQKrQK_cv);
      cute::transform(tQKrQK_cv, tQKrQK_cvt_cv, [](auto v) { return Element(v); });
      copy(tiled_copy_qk, tQKrQK_cvt_cv, tQKsQK);
    };

    auto kk_store_and_inv = [&](auto tKKrKK, auto const& kk_smem_pipe_write) INLINE_LAMBDA {
      auto sKK_inv_pipe_slice = sKK_inv(_, _, kk_smem_pipe_write.index());

      static_assert(sizeof(Element) == 2);
      using CopyOpR2S = SM90_U32x4_STSM_N;
      auto tiled_store_kk = make_tiled_copy_C(Copy_Atom<CopyOpR2S, InverseType>{}, kk_tiled_mma);
      auto thr_store_kk = tiled_store_kk.get_thread_slice(thread_idx);
      auto tKKsKK = thr_store_kk.partition_D(sKK_inv_pipe_slice);
      auto tKKrKK_cv = thr_store_kk.retile_S(tKKrKK);
      auto tKKrKK_cvt_cv = make_fragment_like<InverseType>(tKKrKK_cv);
      cute::transform(tKKrKK_cv, tKKrKK_cvt_cv, [](auto v) { return InverseType(v); });
      copy(tiled_store_kk, tKKrKK_cvt_cv, tKKsKK);

      cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup,
                                                   DeltaRuleNamedBarriers::AuxMath);

      auto collective_inverse = CollectiveInverse(DeltaRuleNamedBarriers::AuxMath);
      collective_inverse.compute(sKK_inv_pipe_slice);

      // FIXME: we can ignore core matrices above diagonal
      if constexpr (NeedsBeta || !std::is_same_v<InverseType, Element>) {
        cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup,
                                                     DeltaRuleNamedBarriers::AuxMath);
        using CopyOpS2R = SM75_U32x4_LDSM_N;
        auto tiled_load_kk = make_tiled_copy_C(Copy_Atom<CopyOpS2R, InverseType>{}, kk_tiled_mma);
        auto thr_load_kk = tiled_load_kk.get_thread_slice(thread_idx);
        auto tKKrKK_cpy = make_fragment_like<InverseType>(tKKrKK_cvt_cv);
        auto tKKrKK_cvt = make_fragment_like<Element>(tKKrKK_cvt_cv);
        auto tKKcMkk_cv = thr_load_kk.retile_D(tKKcMkk);
        copy(tiled_load_kk, thr_load_kk.partition_S(sKK_inv_pipe_slice), tKKrKK_cpy);
        cute::transform(tKKrKK_cpy, tKKcMkk_cv, tKKrKK_cvt, [&](auto val, auto coord) {
          auto [_, t] = coord;
          if constexpr (NeedsBeta) {
            return Element(float(val) * Beta(t, beta_smem_pipe_read.index()));
          } else {
            return Element(val);
          }
        });
        copy(tiled_store_kk, tKKrKK_cvt, recast<Element>(tKKsKK));
      }
    };

    auto compute_aux_loop_body = [&](int blk, auto is_final_block_) INLINE_LAMBDA {
      constexpr bool is_final_block = decltype(is_final_block_)::value;

      int B = is_final_block ? valid_seq_len(work_desc, blk) : BlkSeqKV;

      Tensor tKKrKK = partition_fragment_C(TiledMmaKK{}, select<0, 1>(TileShapeKK{}));
      Tensor tQKrQK = partition_fragment_C(TiledMmaQK{}, select<0, 1>(TileShapeQK{}));

      k_pipeline.consumer_wait(k_smem_pipe_read);
      DPRINTF0_WG("[%d,%d,%d,%d]** dispatch KK WGMMA\n", seq_idx, q_head_idx, k_head_idx,
                  v_head_idx);
      warpgroup_fence_operand(tKKrKK);
      warpgroup_arrive();
      gemm_zero_acc(kk_tiled_mma, tKKrA(_, _, _, k_smem_pipe_read.index()),
                    tKKrB(_, _, _, k_smem_pipe_read.index()), tKKrKK);
      warpgroup_commit_batch();  // K@Kt batch

      q_pipeline.consumer_wait(q_smem_pipe_read);
      DPRINTF0_WG("[%d,%d,%d,%d]** dispatch QK WGMMA\n", seq_idx, q_head_idx, k_head_idx,
                  v_head_idx);
      warpgroup_fence_operand(tQKrQK);
      warpgroup_arrive();
      gemm_zero_acc(qk_tiled_mma, tQKrQ(_, _, _, q_smem_pipe_read.index()),
                    tQKrK(_, _, _, k_smem_pipe_read.index()), tQKrQK);
      warpgroup_commit_batch();  // Q@Kt batch

      // K@Kt and Q@Kt batch finished, we fused masking logic for qk and kk so wait for all of them
      warpgroup_wait<0>();

      k_pipeline.consumer_release(k_smem_pipe_read);
      ++k_smem_pipe_read;
      q_pipeline.consumer_release(q_smem_pipe_read);
      ++q_smem_pipe_read;

      if constexpr (NeedsAlpha) {
        alpha_pipeline.consumer_wait(alpha_smem_pipe_read);
      }
      if constexpr (NeedsBeta) {
        beta_pipeline.consumer_wait(beta_smem_pipe_read);
      }
      cutlass::arch::fence_view_async_shared();

      qk_and_kk_epi(tQKrQK, tKKrKK, alpha_smem_pipe_read, beta_smem_pipe_read, is_final_block_, B);

      kk_pipeline.producer_acquire(kk_smem_pipe_write);
      kk_store_and_inv(tKKrKK, kk_smem_pipe_write);
      cutlass::arch::fence_view_async_shared();
      kk_pipeline.producer_commit(kk_smem_pipe_write);
      ++kk_smem_pipe_write;

      qk_pipeline.producer_acquire(qk_smem_pipe_write);
      qk_store(tQKrQK, qk_smem_pipe_write);
      cutlass::arch::fence_view_async_shared();
      qk_pipeline.producer_commit(qk_smem_pipe_write);
      ++qk_smem_pipe_write;

      if constexpr (NeedsAlpha) {
        alpha_pipeline.consumer_release(alpha_smem_pipe_read);
        ++alpha_smem_pipe_read;
      }
      if constexpr (NeedsBeta) {
        beta_pipeline.consumer_release(beta_smem_pipe_read);
        ++beta_smem_pipe_read;
      }
    };

    int32_t num_blocks = ceil_div(work_desc.seq_len, get<0>(TileShape{}));
    CUTE_NO_UNROLL
    for (int blk = 0; blk < num_blocks - 1; ++blk) {
      compute_aux_loop_body(blk, /*is_final_block_=*/cute::false_type{});
    }
    compute_aux_loop_body(num_blocks - 1, /*is_final_block_=*/cute::true_type{});
  }

  template <typename WorkDesc>
  CUTE_DEVICE int valid_seq_len(WorkDesc work_desc, int blk_idx) {
    int remain_len = work_desc.seq_len - BlkSeqKV * blk_idx;
    return remain_len <= BlkSeqKV ? remain_len : BlkSeqKV;
  }
};

}  // namespace flat::collective
