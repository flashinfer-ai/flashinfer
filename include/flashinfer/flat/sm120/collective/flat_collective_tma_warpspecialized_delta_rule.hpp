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

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

// Reuse hopper code
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
  static_assert(std::is_same_v<ElementAccumulatorQK_, float>,
                "HMMA pipeline only supports float accumulator for QK matmul");
  static_assert(std::is_same_v<ElementAccumulatorKV_, float>,
                "HMMA pipeline only supports float accumulator for KV matmul");
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

  static constexpr bool kEnableCheckpointing =
      find_option_t<Tag::kEnableCheckpointing, false_type, Options>::value;

  static constexpr int NumLoadWarpGroups = 1;
  static constexpr int NumMmaWarpGroups = 2;

  static constexpr int StageCountQ = find_option_t<Tag::kStagesQ, Int<1>, Options>::value;
  static constexpr int StageCountK = find_option_t<Tag::kStagesK, Int<2>, Options>::value;
  static constexpr int StageCountV = find_option_t<Tag::kStagesV, Int<1>, Options>::value;

  static constexpr int NeedsAlpha =
      find_option_t<Tag::kNeedsAlpha, cute::true_type, Options>::value;
  static constexpr int NeedsBeta = find_option_t<Tag::kNeedsBeta, cute::true_type, Options>::value;

  static constexpr int NeedsDecay =
      find_option_t<Tag::kNeedsDecay, cute::false_type, Options>::value;
  static_assert(!NeedsDecay, "DeltaRule does not supports decay");

  static constexpr int NumLoadThreads = NumLoadWarpGroups * 128;
  static constexpr int NumMmaThreads = NumMmaWarpGroups * 128;

  static constexpr uint32_t OrderedBarrierId0 =
      uint32_t(cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
  static constexpr uint32_t OrderedBarrierId1 =
      uint32_t(cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);

  using OrderedMathBarriers = std::conditional_t<
      NumMmaWarpGroups == 2,
      OrderedNamedBarriers</*UseReservedNB=*/true, OrderedBarrierId0, OrderedBarrierId1>,
      OrderedNamedBarriers</*UseReservedNB=*/true, OrderedBarrierId0>>;

  using StagesQ = cutlass::gemm::collective::StageCount<StageCountQ>;
  using StagesK = cutlass::gemm::collective::StageCount<StageCountK>;
  using StagesV = cutlass::gemm::collective::StageCount<StageCountV>;
  using StagesO = cutlass::gemm::collective::StageCount<1>;
  using ClusterShape = Shape<_1, _1, _1>;

  using StagesAlphaBeta = cutlass::gemm::collective::StageCount<2>;

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

  using SmemLayoutQ_SD = decltype(tile_to_shape(
      GMMA::Layout_K_SW128_Atom<Element>{},
      make_shape(get<0>(TileShape{}), get<2>(TileShape{}), Int<StagesQ::value>{}),
      Step<_0, _1, _2>{}));
  using SmemLayoutK_SD = decltype(tile_to_shape(
      GMMA::Layout_K_SW128_Atom<Element>{},
      make_shape(get<1>(TileShape{}), get<2>(TileShape{}), Int<StagesK::value>{}),
      Step<_0, _1, _2>{}));
  using SmemLayoutV_SD = decltype(restage_smem_layout(SmemLayoutK_SD{}, Int<StagesV::value>{}));
  ;

  using SmemLayoutK_DS = decltype(select_layout<1, 0, 2>(SmemLayoutK_SD{}));
  using SmemLayoutV_DS = decltype(select_layout<1, 0, 2>(SmemLayoutV_SD{}));

  using MmaOp = std::conditional_t<std::is_same_v<Element, cutlass::bfloat16_t>,
                                   SM80_16x8x16_F32BF16BF16F32_TN, SM80_16x8x16_F32F16F16F32_TN>;

  using RefLayoutV = decltype(make_layout(select<0, 2>(TileShapeKV{}), LayoutRight{}));

  using RefLayoutKV =
      decltype(make_layout(select<0, 1>(TileShapeKV{}), LayoutRight{}));  // (dv, dk)

  // (blk_q,blk_k) to align with O2 mma, LayoutRight to align with QK mma output
  using DesiredLayoutQK = decltype(make_layout(select<0, 1>(TileShapeQK{}), LayoutRight{}));

  using TiledMmaQK = decltype(make_tiled_mma(MmaOp{}, Layout<_4, _1>{}, TileShapeQK{}));  // Q@K^t
  using TiledMmaKV = decltype(make_tiled_mma(MmaOp{}, Layout<_8, _1>{}, TileShapeKV{}));  // V @ K
  using TiledMmaO1 = decltype(make_tiled_mma(MmaOp{}, Layout<_8, _1>{}, TileShapeO1{}));  // KV @ Q
  using TiledMmaO2 = decltype(make_tiled_mma(MmaOp{}, Layout<_8, _1>{}, TileShapeO2{}));  // V @ QK

  static_assert(size(TiledMmaQK{}) == NumMmaThreads || size(TiledMmaQK{}) == NumMmaThreads / 2);

  static_assert(size(TiledMmaKV{}) == NumMmaThreads);
  static_assert(size(TiledMmaO1{}) == NumMmaThreads);
  static_assert(size(TiledMmaO2{}) == NumMmaThreads);

  using CollectiveStoreO =
      CollectiveStoreTma<TileShapeO1, ClusterShape, ElementO, ElementAccumulatorO,
                         /*Seme*/ ElementO, decltype(select<1, 0, 2>(LayoutO{})), StagesO::value>;

  // layout for compute output
  using LayoutAtom = Layout<Shape<_8, _8>, Stride<_8, _1>>;
  using SmemLayoutQK =
      decltype(tile_to_shape(LayoutAtom{}, select<0, 1>(TileShapeQK{}), Step<_1, _2>{}));
  using SmemLayoutKK =
      decltype(tile_to_shape(LayoutAtom{}, select<0, 1>(TileShapeQK{}), Step<_1, _2>{}));
  using SmemLayoutO = typename CollectiveStoreO::SmemLayoutO;

  using InverseType = cutlass::half_t;
  using CollectiveInverse = flat::collective::CollectiveInverse<InverseType, true, false>;

  using ElementAccumulatorSK = float;
  using TileShapeSK = decltype(make_shape(HeadSizeV, BlkSeqKV, HeadSizeQK));

  using ElementAccumulatorNewV = float;
  using TileShapeNewV = decltype(make_shape(HeadSizeV, BlkSeqKV, BlkSeqKV));
  using RefLayoutSK =
      decltype(make_layout(select<0, 2>(TileShapeNewV{}), LayoutRight{}));  // (dv, Blk)
  using DesiredLayoutKK = decltype(make_layout(select<1, 2>(TileShapeNewV{}), LayoutRight{}));  //

  using TiledMmaKK = decltype(make_tiled_mma(
      MmaOp{}, Layout<_4, _1>{}, TileShapeKK{}));  // T = inv(I + strict_lower_triangular(K@K^t))
  using TiledMmaSK =
      decltype(make_tiled_mma(MmaOp{}, Layout<_8, _1>{}, TileShapeSK{}));  // ?? = -S@K^t + V^t

  using TiledMmaNewV =
      decltype(make_tiled_mma(MmaOp{}, Layout<_8, _1>{}, TileShapeNewV{}));  // NewV = ??@T^t

  static_assert(size(TiledMmaKK{}) == size(TiledMmaQK{}));

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

  using MainloopAlphaPipeline =
      std::conditional_t<NeedsAlpha, cutlass::PipelineAsync<StagesAlphaBeta::value>, Unused>;
  using MainloopBetaPipeline =
      std::conditional_t<NeedsBeta, cutlass::PipelineAsync<StagesAlphaBeta::value>, Unused>;

  using QPipelineState = typename cutlass::PipelineState<MainloopQPipeline::Stages>;
  using KPipelineState = typename cutlass::PipelineState<MainloopKPipeline::Stages>;
  using VPipelineState = typename cutlass::PipelineState<MainloopVPipeline::Stages>;
  using OPipelineState = typename CollectiveStoreO::PipelineState;

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
        sum = __shfl_sync(0xFFFFFFFF, frag(iter - 1), 31);
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

  static constexpr int LoadQBytes = size(SmemLayoutQ_SD{}(_, _, _0{})) * sizeof(Element);
  static constexpr int LoadKBytes = size(SmemLayoutK_DS{}(_, _, _0{})) * sizeof(Element);
  static constexpr int LoadVBytes = size(SmemLayoutV_DS{}(_, _, _0{})) * sizeof(Element);
  static constexpr int StoreOBytes = CollectiveStoreO::TmaTransactionBytes;

  using SharedStorageO = typename CollectiveStoreO::SharedStorage;

  struct SharedStorage {
    alignas(alignment_for_swizzle(
        SmemLayoutQ_SD{})) cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ_SD>> smem_q;
    alignas(alignment_for_swizzle(
        SmemLayoutK_DS{})) cute::array_aligned<Element, cute::cosize_v<SmemLayoutK_DS>> smem_k;
    alignas(alignment_for_swizzle(
        SmemLayoutV_DS{})) cute::array_aligned<Element, cute::cosize_v<SmemLayoutV_DS>> smem_v;
    alignas(alignment_for_swizzle(
        SmemLayoutQK{})) cute::array_aligned<Element, cute::cosize_v<SmemLayoutQK>> smem_qk;
    alignas(alignment_for_swizzle(
        SmemLayoutKK{})) cute::array_aligned<InverseType, cute::cosize_v<SmemLayoutKK>> smem_kk;

    SharedStorageO smem_o;
    // TODO: make optional
    cute::array_aligned<float, cute::cosize_v<SmemLayoutBeta>> smem_beta;
    cute::array_aligned<float, cute::cosize_v<SmemLayoutAlpha>> smem_alpha;
  };

  using GmemTiledCopyQKV = cute::SM90_TMA_LOAD;
  using ShapeQKV = Shape<int64_t, int32_t, int32_t>;  // (seq, d, h)

  static_assert(size(ClusterShape{}) == 1, "mcast TMA not supported");
  static constexpr auto cluster_size_no_mcast = _1{};

  using TMA_Q = decltype(make_tma_copy<Element>(
      GmemTiledCopyQKV{},
      make_tensor(static_cast<Element const*>(nullptr),
                  make_layout(ShapeQKV{}, LayoutQ{})),  // LayoutQ is stride actually
      take<0, 2>(SmemLayoutQ_SD{}),                     // no Stages
      select<0, 2>(TileShape{}),                        // (seqlen, d)
      cluster_size_no_mcast));

  using TMA_K = decltype(make_tma_copy<Element>(
      GmemTiledCopyQKV{},
      select_tensor<1, 0, 2>(
          make_tensor(static_cast<Element const*>(nullptr),
                      make_layout(ShapeQKV{}, LayoutK{}))),  // LayoutK is stride actually
      take<0, 2>(SmemLayoutK_DS{}),                          // no Stages
      select<2, 1>(TileShape{}),                             // (d, seqlen)
      cluster_size_no_mcast));

  using TMA_V = decltype(make_tma_copy<Element>(
      GmemTiledCopyQKV{},
      select_tensor<1, 0, 2>(
          make_tensor(static_cast<Element const*>(nullptr),
                      make_layout(ShapeQKV{}, LayoutV{}))),  // LayoutV is stride actually
      take<0, 2>(SmemLayoutV_DS{}),                          // no Stages
      select<2, 1>(TileShape{}),                             // (d, seqlen)
      cluster_size_no_mcast));

  using TMA_O = typename CollectiveStoreO::Params::TMA_O;

  using LoadQ = CollectiveLoadTma<LoadKind::kQ, MainloopQPipeline, Element, SmemLayoutQ_SD, TMA_Q>;
  using LoadK = CollectiveLoadTma<LoadKind::kK, MainloopKPipeline, Element, SmemLayoutK_DS, TMA_K>;
  using LoadV = CollectiveLoadTma<LoadKind::kV, MainloopVPipeline, Element, SmemLayoutV_DS, TMA_V>;

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
    float*         ptr_state_checkpoints;     // [total_checkpoints, num_sab_heads, K, V]
    int64_t const* checkpoint_cu_starts;      // [num_seqs + 1]
    int32_t        checkpoint_every_n_tokens; // 0 = disabled, must be multiple of BlkSeqKV(64)
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

    float* ptr_state_checkpoints;
    int64_t const* checkpoint_cu_starts;
    int32_t checkpoint_every_n_tokens;
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

    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q),
                            make_layout(ShapeQKV(s, d, problem_size.num_q_heads), args.dQ));
    Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K),
                            make_layout(ShapeQKV(t, d, problem_size.num_k_heads), args.dK));
    Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V),
                            make_layout(ShapeQKV(t, d, problem_size.num_v_heads), args.dV));

    TMA_Q tma_load_q =
        make_tma_copy<Element>(GmemTiledCopyQKV{}, mQ, take<0, 2>(SmemLayoutQ_SD{}),  // no Stages
                               select<0, 2>(TileShape{}),  // (seqlen_q, d)
                               cluster_size_no_mcast);

    TMA_K tma_load_k = make_tma_copy<Element>(GmemTiledCopyQKV{}, select_tensor<1, 0, 2>(mK),
                                              take<0, 2>(SmemLayoutK_DS{}),  // no Stages
                                              select<2, 1>(TileShape{}),     // (d, seqlen_kv)
                                              cluster_size_no_mcast);

    TMA_V tma_load_v = make_tma_copy<Element>(GmemTiledCopyQKV{}, select_tensor<1, 0, 2>(mV),
                                              take<0, 2>(SmemLayoutV_DS{}),  // no Stages
                                              select<2, 1>(TileShape{}),     // (d, seqlen_kv)
                                              cluster_size_no_mcast);

    auto params_o = CollectiveStoreO::to_underlying_arguments(
        make_shape(d, s, d, problem_size.num_o_heads),  // in O1
        // make_shape(d, s, s, problem_size.num_o_heads),  // in O2
        typename CollectiveStoreO::Arguments{args.ptr_O, select<1, 0, 2>(args.dO)}, workspace);

    return Params{
        .tma_load_q = tma_load_q,
        .tma_load_k = tma_load_k,
        .tma_load_v = tma_load_v,
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

        .ptr_state_checkpoints = args.ptr_state_checkpoints,
        .checkpoint_cu_starts = args.checkpoint_cu_starts,
        .checkpoint_every_n_tokens = args.checkpoint_every_n_tokens,
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
  CUTE_DEVICE void compute(Params const& params, ProblemShape const& problem_size,
                           WorkDesc const& work_desc, MainloopQPipeline& q_pipeline,
                           QPipelineState& q_smem_pipe_read, MainloopKPipeline& k_pipeline,
                           KPipelineState& k_smem_pipe_read, MainloopVPipeline& v_pipeline,
                           VPipelineState& v_smem_pipe_read, MainloopOPipeline& o_pipeline,
                           OPipelineState& o_smem_pipe_write, MainloopAlphaPipeline& alpha_pipeline,
                           AlphaPipelineState& alpha_smem_pipe_read,
                           MainloopBetaPipeline& beta_pipeline,
                           BetaPipelineState& beta_smem_pipe_read,
                           OrderedMathBarriers& math_barriers, SharedStorage& storage) {
    // MAKE NVCC HAPPY!
    constexpr auto zero = Element{};

    int32_t num_blocks = ceil_div(work_desc.seq_len, get<0>(TileShape{}));
    DPRINTF0_WG("num_blocks: %d\n", num_blocks);

    int thread_idx = int(threadIdx.x) - NumLoadThreads;
    int warpgroup_idx = thread_idx / cutlass::NumThreadsPerWarpGroup;

    int kk_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
    int qk_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
    bool is_kk_wg = NumMmaWarpGroups == 1 || warpgroup_idx == 0;
    bool is_qk_wg = NumMmaWarpGroups == 1 || warpgroup_idx == 1;

    float scale = params.scale;

    Tensor Beta = make_tensor(make_smem_ptr(storage.smem_beta.data()), SmemLayoutBeta{});
    Tensor Alpha = make_tensor(make_smem_ptr(storage.smem_alpha.data()), SmemLayoutAlpha{});

    Tensor sQ_SD = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ_SD{});
    Tensor sK_SD = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK_SD{});
    Tensor sK_DS = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK_DS{});
    Tensor sV_DS = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV_DS{});
    Tensor sQK = make_tensor(make_smem_ptr(storage.smem_qk.data()), SmemLayoutQK{});
    Tensor sO = make_tensor(make_smem_ptr(storage.smem_o.data()), SmemLayoutO{});

    static_assert(sizeof(InverseType) == sizeof(Element));
    Tensor sKK_inv = make_tensor(make_smem_ptr(storage.smem_kk.data()), SmemLayoutKK{});
    Tensor sKK_opd = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(storage.smem_kk.data())),
                                 SmemLayoutKK{});

    ///////////////////////////////////////////////////////////////////////////
    // Q@K
    auto qk_tiled_mma = TiledMmaQK{};
    auto qk_thr_mma = qk_tiled_mma.get_thread_slice(qk_thread_idx);
    auto qk_tiled_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, qk_tiled_mma);
    auto qk_tiled_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, qk_tiled_mma);
    auto qk_thr_copy_A = qk_tiled_copy_A.get_thread_slice(qk_thread_idx);
    auto qk_thr_copy_B = qk_tiled_copy_B.get_thread_slice(qk_thread_idx);

    Tensor tQKrQ = qk_thr_mma.partition_fragment_A(sQ_SD(_, _, _0{}));
    Tensor tQKrQ_cv = qk_thr_copy_A.retile_D(tQKrQ);
    Tensor tQKsQ = qk_thr_copy_A.partition_S(sQ_SD);

    Tensor tQKrK = qk_thr_mma.partition_fragment_B(sK_SD(_, _, _0{}));
    Tensor tQKrK_cv = qk_thr_copy_B.retile_D(tQKrK);
    Tensor tQKsK = qk_thr_copy_B.partition_S(sK_SD);

    auto cMqk = make_identity_tensor(select<0, 1>(TileShapeQK{}));  // (QTok, KTok)
    auto tQKcMqk = qk_thr_mma.partition_C(cMqk);                    // (idx) -> (tok_q, tok_k)

    ///////////////////////////////////////////////////////////////////////////
    // K@K  (basically I + strict_lower_triangular(K K^T)
    auto kk_tiled_mma = TiledMmaKK{};
    auto kk_thr_mma = kk_tiled_mma.get_thread_slice(kk_thread_idx);
    auto kk_tiled_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, kk_tiled_mma);
    auto kk_tiled_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, kk_tiled_mma);
    auto kk_thr_copy_A = kk_tiled_copy_A.get_thread_slice(kk_thread_idx);
    auto kk_thr_copy_B = kk_tiled_copy_B.get_thread_slice(kk_thread_idx);

    Tensor tKKrA = kk_thr_mma.partition_fragment_A(sK_SD(_, _, _0{}));
    Tensor tKKrA_cv = kk_thr_copy_A.retile_D(tKKrA);
    Tensor tKKsA = kk_thr_copy_A.partition_S(sK_SD);

    Tensor tKKrB = kk_thr_mma.partition_fragment_B(sK_SD(_, _, _0{}));
    Tensor tKKrB_cv = kk_thr_copy_B.retile_D(tKKrB);
    Tensor tKKsB = kk_thr_copy_B.partition_S(sK_SD);

    auto const& cMkk = cMqk;
    auto tKKcMkk = kk_thr_mma.partition_C(cMkk);

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
    auto sk_tiled_copy_C = make_tiled_copy_C(Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, sk_tiled_mma);
    auto sk_thr_copy_C = sk_tiled_copy_C.get_thread_slice(thread_idx);
    auto sk_tiled_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, sk_tiled_mma);
    auto sk_thr_copy_B = sk_tiled_copy_B.get_thread_slice(thread_idx);

    Tensor tSKrK = sk_thr_mma.partition_fragment_B(sK_SD(_, _, _0{}));
    Tensor tSKrK_cv = sk_thr_copy_B.retile_D(tSKrK);
    Tensor tSKsK = sk_thr_copy_B.partition_S(sK_SD);

    ///////////////////////////////////////////////////////////////////////////
    // NewV = (S@K result) @ T^t
    auto newv_tiled_mma = TiledMmaNewV{};
    auto newv_thr_mma = newv_tiled_mma.get_thread_slice(thread_idx);
    auto newv_tiled_copy_B =
        make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, newv_tiled_mma);
    auto newv_thr_copy_B = newv_tiled_copy_B.get_thread_slice(thread_idx);

    Tensor tNewVrB = newv_thr_mma.partition_fragment_B(sKK_opd);
    Tensor tNewVrB_cv = newv_thr_copy_B.retile_D(tNewVrB);
    Tensor tNewVsB = newv_thr_copy_B.partition_S(sKK_opd);

    ///////////////////////////////////////////////////////////////////////////
    // K@V
    auto kv_tiled_mma = TiledMmaKV{};
    auto kv_thr_mma = kv_tiled_mma.get_thread_slice(thread_idx);
    auto kv_tiled_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, kv_tiled_mma);
    auto kv_thr_copy_B = kv_tiled_copy_B.get_thread_slice(thread_idx);

    Tensor tKVrKV = partition_fragment_C(kv_thr_mma, select<0, 1>(TileShapeKV{}));
    Tensor tKVrK = kv_thr_mma.partition_fragment_B(sK_DS(_, _, _0{}));
    Tensor tKVrK_cv = kv_thr_copy_B.retile_D(tKVrK);
    Tensor tKVsK = kv_thr_copy_B.partition_S(sK_DS);

    auto const cV = make_identity_tensor(Shape<Int<HeadSizeV>, Int<BlkSeqKV>>{});
    Tensor tKVcV = kv_thr_mma.partition_A(cV);

    ///////////////////////////////////////////////////////////////////////////
    // Q@K@V
    auto o1_tiled_mma = TiledMmaO1{};
    auto o1_thr_mma = o1_tiled_mma.get_thread_slice(thread_idx);
    auto o2_tiled_mma = TiledMmaO2{};
    auto o2_thr_mma = o2_tiled_mma.get_thread_slice(thread_idx);

    auto o1_tiled_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, o1_tiled_mma);
    auto o1_thr_copy_B = o1_tiled_copy_B.get_thread_slice(thread_idx);
    auto o2_tiled_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, o2_tiled_mma);
    auto o2_thr_copy_B = o2_tiled_copy_B.get_thread_slice(thread_idx);

    // A1 for Q@(KV)
    // Tensor tOrKV = make_acc_into_op<Element>(tKVrKV, typename TiledMmaO1::LayoutA_TV{});
    // B1 for Q@(KV)
    Tensor tOrQ = o1_thr_mma.partition_fragment_B(sQ_SD(_, _, _0{}));
    Tensor tOrQ_cv = o1_thr_copy_B.retile_D(tOrQ);
    Tensor tOsQ = o1_thr_copy_B.partition_S(sQ_SD);

    // A2 for QK@V
    // Tensor tOrV = make_acc_into_op<Element>(tKVrKV, typename TiledMmaO2::LayoutA_TV{});
    // B2 for QK@V
    Tensor tOrQK = o2_thr_mma.partition_fragment_B(sQK);
    Tensor tOrQK_cv = o2_thr_copy_B.retile_D(tOrQK);
    Tensor tOsQK = o2_thr_copy_B.partition_S(sQK);

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

    auto qk_or_kk_mask = [&](auto& tQKrQK, auto is_final_block_, auto B /*valid seqlen*/) {
      constexpr bool is_final_block = decltype(is_final_block_)::value;
      for_each(make_int_sequence<size(tKKcMkk)>{}, [&](auto i) {
        auto coord = tQKcMqk(i);
        auto [s, t] = coord;
        bool pred = s >= t;
        if constexpr (is_final_block) {
          pred = pred && (s < B || t < B);
        }
        // for tKKrKK diagonal is garbage filled, will be processed during inversion
        tQKrQK(i) = pred ? tQKrQK(i) : 0.0f;
      });
    };

    auto qk_epi = [&](auto& tQKrQK, auto const& alpha_smem_pipe_read) {
      if constexpr (NeedsAlpha) {
        Tensor Alpha_cumsum_log = Alpha(_, AlphaCumSumLogIdx{}, alpha_smem_pipe_read.index());
        for_each(make_int_sequence<size(tKKcMkk)>{}, [&](auto i) {
          auto coord = tQKcMqk(i);
          auto [s, t] = coord;
          float alpha = exp2f(Alpha_cumsum_log(s) - Alpha_cumsum_log(t));
          tQKrQK(i) *= alpha * scale;
        });
      } else {
        transform(tQKrQK, [scale](auto v) { return v * scale; });
      }
    };

    auto qk_store = [&](auto tQKrQK) {
      static_assert(sizeof(Element) == 2);
      using CopyOpR2S = SM90_U32x4_STSM_N;
      auto tiled_copy_qk = make_tiled_copy_C(Copy_Atom<CopyOpR2S, Element>{}, qk_tiled_mma);
      auto thr_copy_qk = tiled_copy_qk.get_thread_slice(qk_thread_idx);
      auto tQKsQK = thr_copy_qk.partition_D(sQK);
      auto tQKrQK_cv = thr_copy_qk.retile_S(tQKrQK);
      auto tQKrQK_cvt_cv = make_fragment_like<Element>(tQKrQK_cv);
      cute::transform(tQKrQK_cv, tQKrQK_cvt_cv, [](auto v) { return Element(v); });
      copy(tiled_copy_qk, tQKrQK_cvt_cv, tQKsQK);
    };

    auto kk_epi = [&](auto& tKKrKK, auto const& alpha_smem_pipe_read,
                      auto const& beta_smem_pipe_read) {
      if constexpr (NeedsAlpha) {
        Tensor Alpha_cumsum_log = Alpha(_, AlphaCumSumLogIdx{}, alpha_smem_pipe_read.index());
        for_each(make_int_sequence<size(tKKcMkk)>{}, [&](auto i) {
          auto coord = tQKcMqk(i);
          auto [s, t] = coord;
          float alpha = exp2f(Alpha_cumsum_log(s) - Alpha_cumsum_log(t));
          tKKrKK(i) *= alpha;
        });
      }

      if constexpr (NeedsBeta) {
        Tensor Beta_ = Beta(_, beta_smem_pipe_read.index());
        for_each(make_int_sequence<size(tKKcMkk)>{}, [&](auto i) {
          auto coord = tQKcMqk(i);
          auto [s, t] = coord;
          tKKrKK(i) *= Beta_(s);
        });
      }
    };

    auto kk_store_and_inv = [&](auto tKKrKK) INLINE_LAMBDA {
      static_assert(sizeof(Element) == 2);
      using CopyOpR2S = SM90_U32x4_STSM_N;
      auto tiled_store_kk = make_tiled_copy_C(Copy_Atom<CopyOpR2S, InverseType>{}, kk_tiled_mma);
      auto thr_store_kk = tiled_store_kk.get_thread_slice(kk_thread_idx);
      auto tKKsKK = thr_store_kk.partition_D(sKK_inv);
      auto tKKrKK_cv = thr_store_kk.retile_S(tKKrKK);
      auto tKKrKK_cvt_cv = make_fragment_like<InverseType>(tKKrKK_cv);
      cute::transform(tKKrKK_cv, tKKrKK_cvt_cv, [](auto v) { return InverseType(v); });
      copy(tiled_store_kk, tKKrKK_cvt_cv, tKKsKK);

      cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup,
                                                   DeltaRuleNamedBarriers::AuxMath);

      auto collective_inverse = CollectiveInverse(DeltaRuleNamedBarriers::AuxMath);
      collective_inverse.compute(sKK_inv);

      // FIXME: we can ignore core matrices above diagonal
      if constexpr (NeedsBeta || !std::is_same_v<InverseType, Element>) {
        cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup,
                                                     DeltaRuleNamedBarriers::AuxMath);
        using CopyOpS2R = SM75_U32x4_LDSM_N;
        auto tiled_load_kk = make_tiled_copy_C(Copy_Atom<CopyOpS2R, InverseType>{}, kk_tiled_mma);
        auto thr_load_kk = tiled_load_kk.get_thread_slice(kk_thread_idx);
        auto tKKrKK_cpy = make_fragment_like<InverseType>(tKKrKK_cvt_cv);
        auto tKKrKK_cvt = make_fragment_like<Element>(tKKrKK_cvt_cv);
        auto tKKcMkk_cv = thr_load_kk.retile_D(tKKcMkk);
        copy(tiled_load_kk, thr_load_kk.partition_S(sKK_inv), tKKrKK_cpy);
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

    auto sk_epi = [&](auto& tSKrSK, auto const& alpha_smem_pipe_read) INLINE_LAMBDA {
      if constexpr (NeedsAlpha) {
        transform(tSKrSK, tSKrAlpha(_, _, _, alpha_smem_pipe_read.index()), tSKrSK,
                  [&](auto sk, auto coeff) { return sk * coeff; });
      }
    };

    auto sk_load_v = [&](int pipe_idx) INLINE_LAMBDA {
      Tensor tSKrV =
          make_fragment_like<Element>(partition_fragment_C(sk_thr_mma, sV_DS(_, _, _0{})));
      Tensor tSKrV_cv = sk_thr_copy_C.retile_D(tSKrV);
      Tensor tSKsV = sk_thr_copy_C.partition_S(sV_DS);
      copy(sk_tiled_copy_C, tSKsV(_, _, _, pipe_idx), tSKrV_cv);
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

    auto kv_checkpoint_store = [&](int checkpoint_idx) INLINE_LAMBDA {
      if constexpr (kEnableCheckpointing) {
        DPRINTF0_WG("[%d,%d,%d,%d]>> save tKVrKV -> checkpoint[%d]\n", seq_idx, q_head_idx,
                    k_head_idx, v_head_idx, checkpoint_idx);
        int num_state_heads = problem_size.num_sab_heads;
        int state_head_idx = work_desc.o_head_idx();
        int64_t ckpt_offset = params.checkpoint_cu_starts[seq_idx] + checkpoint_idx;

        // Layout: [total_checkpoints, num_sab_heads, HeadSizeQK, HeadSizeV] LayoutLeft
        auto gKV =
            make_tensor(make_gmem_ptr(params.ptr_state_checkpoints +
                                      ckpt_offset * num_state_heads * HeadSizeQK * HeadSizeV +
                                      state_head_idx * HeadSizeQK * HeadSizeV),
                        make_layout(make_shape(Int<HeadSizeQK>{}, Int<HeadSizeV>{})));

        auto tiled_copy_kv =
            make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, Element>{}, kv_tiled_mma);
        auto thr_copy_kv = tiled_copy_kv.get_thread_slice(thread_idx);

        auto tKVgKV = thr_copy_kv.partition_D(select_tensor<1, 0>(gKV));
        copy(tiled_copy_kv, tKVrKV, tKVgKV);
      }
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

      DPRINTF0_WG("compute: k_pipeline.consumer_wait: smem_pipe_read:%d\n",
                  k_smem_pipe_read.index());
      k_pipeline.consumer_wait(k_smem_pipe_read);
      if constexpr (NeedsAlpha) {
        alpha_pipeline.consumer_wait(alpha_smem_pipe_read);
      }
      if constexpr (NeedsBeta) {
        beta_pipeline.consumer_wait(beta_smem_pipe_read);
      }
      do {
        if (!is_kk_wg) {
          __syncwarp();
          break;
        }
        DPRINTF0_WG("[%d,%d,%d,%d]** dispatch KK MMA\n", seq_idx, q_head_idx, k_head_idx,
                    v_head_idx);
        copy(kk_tiled_copy_A, tKKsA(_, _, _, k_smem_pipe_read.index()), tKKrA_cv);
        copy(kk_tiled_copy_B, tKKsB(_, _, _, k_smem_pipe_read.index()), tKKrB_cv);

        Tensor tKKrKK = partition_fragment_C(TiledMmaKK{}, select<0, 1>(TileShapeKK{}));
        clear(tKKrKK);
        gemm(kk_tiled_mma, tKKrA, tKKrB, tKKrKK);

        kk_epi(tKKrKK, alpha_smem_pipe_read, beta_smem_pipe_read);
        qk_or_kk_mask(tKKrKK, is_final_block_, B);
        kk_store_and_inv(tKKrKK);
      } while (0);
      if constexpr (NeedsBeta) {
        beta_pipeline.consumer_release(beta_smem_pipe_read);
        ++beta_smem_pipe_read;
      }

      DPRINTF0_WG("compute: q_pipeline.consumer_wait: smem_pipe_read:%d\n",
                  q_smem_pipe_read.index());
      q_pipeline.consumer_wait(q_smem_pipe_read);
      do {
        if (!is_qk_wg) {
          __syncwarp();
          break;
        }
        DPRINTF0_WG("[%d,%d,%d,%d]** dispatch QK MMA\n", seq_idx, q_head_idx, k_head_idx,
                    v_head_idx);
        copy(qk_tiled_copy_A, tQKsQ(_, _, _, q_smem_pipe_read.index()), tQKrQ_cv);
        copy(qk_tiled_copy_B, tQKsK(_, _, _, k_smem_pipe_read.index()), tQKrK_cv);

        Tensor tQKrQK = partition_fragment_C(TiledMmaQK{}, select<0, 1>(TileShapeQK{}));
        clear(tQKrQK);
        gemm(qk_tiled_mma, tQKrQ, tQKrK, tQKrQK);

        qk_epi(tQKrQK, alpha_smem_pipe_read);
        qk_or_kk_mask(tQKrQK, is_final_block_, B);
        qk_store(tQKrQK);
      } while (0);

      // 2.1 Q @ KV, NOTE: use old KV here
      auto tOrO = partition_fragment_C(o1_thr_mma, select<0, 1>(TileShapeO1{}));
      clear(tOrO);
      if constexpr (!is_first_block) {
        DPRINTF0_WG("[%d,%d,%d,%d]** dispatch O1 MMA\n", seq_idx, q_head_idx, k_head_idx,
                    v_head_idx);
        copy(o1_tiled_copy_B, tOsQ(_, _, _, q_smem_pipe_read.index()), tOrQ_cv);
        Tensor tOrKV = detail::SM80::make_acc_into_op<Element>(tKVrKV, o1_thr_mma);
        gemm(o1_thr_mma, tOrKV, tOrQ, tOrO);
        o1_epi(tOrO, alpha_smem_pipe_read);
      }
      DPRINTF0_WG("compute: q_pipeline.consumer_release: smem_pipe_read:%d\n",
                  q_smem_pipe_read.index());
      q_pipeline.consumer_release(q_smem_pipe_read);
      ++q_smem_pipe_read;

      auto tSKrSK = partition_fragment_C(sk_thr_mma, sV_DS(_, _, _0{}));
      if constexpr (!is_first_block) {
        auto tSKrS = detail::SM80::make_acc_into_op<Element>(tKVrKV, sk_tiled_mma);
        copy(sk_tiled_copy_B, tSKsK(_, _, _, k_smem_pipe_read.index()), tSKrK_cv);
        clear(tSKrSK);
        gemm(sk_tiled_mma, tSKrS, tSKrK, tSKrSK);
      }

      DPRINTF0_WG("compute: v_pipeline.consumer_wait: smem_pipe_read:%d\n",
                  v_smem_pipe_read.index());
      v_pipeline.consumer_wait(v_smem_pipe_read);
      auto tSKrV = sk_load_v(v_smem_pipe_read.index());
      if constexpr (!is_first_block) {
        sk_epi(tSKrSK, alpha_smem_pipe_read);
        transform(tSKrV, tSKrSK, tSKrV, [](auto v, auto sk) { return v - Element(sk); });
      }

      DPRINTF0_WG("[%d,%d,%d,%d]** dispatch NewV MMA\n", seq_idx, q_head_idx, k_head_idx,
                  v_head_idx);
      auto tNewVrA = detail::SM80::make_acc_into_op<Element>(tSKrV, newv_tiled_mma);
      auto tNewVrC = partition_fragment_C(newv_thr_mma, select<0, 1>(TileShapeNewV{}));
      math_barriers.ordered_or_wait(warpgroup_idx);
      copy(newv_tiled_copy_B, tNewVsB, tNewVrB_cv);
      clear(tNewVrC);
      gemm(newv_tiled_mma, tNewVrA, tNewVrB, tNewVrC);
      math_barriers.notify_next_blocked(warpgroup_idx);
      DPRINTF0_WG("compute: v_pipeline.consumer_release: smem_pipe_read:%d\n",
                  v_smem_pipe_read.index());
      v_pipeline.consumer_release(v_smem_pipe_read);
      ++v_smem_pipe_read;

      /////////////////////////////////////////////////////////////////////////
      // 2. compute qkv
      // 2.2 QK @ V, NOTE: use old KV here and QK is scaled
      DPRINTF0_WG("[%d,%d,%d,%d]** dispatch O2 MMA\n", seq_idx, q_head_idx, k_head_idx, v_head_idx);
      auto tOrV_or_tKVrV = detail::SM80::make_acc_into_op<Element>(tNewVrC, o2_tiled_mma);
      math_barriers.ordered_or_wait(warpgroup_idx);
      copy(o2_tiled_copy_B, tOsQK, tOrQK_cv);
      gemm(o2_tiled_mma, tOrV_or_tKVrV, tOrQK, tOrO);
      math_barriers.notify_next_blocked(warpgroup_idx);
      o_store(tOrO);

      /////////////////////////////////////////////////////////////////////////
      // 3. update KV
      float block_coeff = 1.0f;
      if constexpr (NeedsAlpha) {
        block_coeff = Alpha(B - 1, AlphaCumProdIdx{}, alpha_smem_pipe_read.index());
      }

      cute::transform(tKVrKV, [&](auto kv) { return block_coeff * kv; });
      kv_decay_v(tOrV_or_tKVrV, alpha_smem_pipe_read, is_final_block_, B);

      DPRINTF0_WG("[%d,%d,%d,%d]** dispatch KV MMA\n", seq_idx, q_head_idx, k_head_idx, v_head_idx);
      copy(kv_tiled_copy_B, tKVsK(_, _, _, k_smem_pipe_read.index()), tKVrK_cv);
      gemm(kv_tiled_mma, tOrV_or_tKVrV, tKVrK, tKVrKV);

      DPRINTF0_WG("compute: k_pipeline.consumer_release: smem_pipe_read:%d\n",
                  k_smem_pipe_read.index());
      k_pipeline.consumer_release(k_smem_pipe_read);
      ++k_smem_pipe_read;

      if constexpr (NeedsAlpha) {
        alpha_pipeline.consumer_release(alpha_smem_pipe_read);
        ++alpha_smem_pipe_read;
      }
    };

    int ckpt_blk_interval =
        (params.checkpoint_every_n_tokens > 0) ? params.checkpoint_every_n_tokens / BlkSeqKV : 0;
    int ckpt_count = 0;

    if constexpr (!kInitStateFromInput) {
      clear(tKVrKV);
      compute_loop_body(0, /*is_first_block_=*/cute::true_type{},
                        /*is_final_block_=*/cute::true_type{});
    } else {
      kv_load(tKVrKV);
      compute_loop_body(0, /*is_first_block_=*/cute::false_type{},
                        /*is_final_block_=*/cute::true_type{});
    }
    if constexpr (kEnableCheckpointing) {
      if (ckpt_blk_interval == 1) {
        kv_checkpoint_store(ckpt_count++);
      }
    }
    CUTE_NO_UNROLL
    for (int blk = 1; blk < num_blocks - 1; ++blk) {
      compute_loop_body(blk, /*is_first_block_=*/cute::false_type{},
                        /*is_final_block_=*/cute::false_type{});
      if constexpr (kEnableCheckpointing) {
        if ((blk + 1) % ckpt_blk_interval == 0) {
          kv_checkpoint_store(ckpt_count++);
        }
      }
    }
    if (num_blocks != 1) {
      compute_loop_body(num_blocks - 1, /*is_first_block_=*/cute::false_type{},
                        /*is_final_block_=*/cute::true_type{});
      // Only checkpoint on exact boundaries; the final (possibly partial) state
      // is always available via output_state from kv_store() below.
      if constexpr (kEnableCheckpointing) {
        if (num_blocks % ckpt_blk_interval == 0) {
          kv_checkpoint_store(ckpt_count);
        }
      }
    }
    kv_store();
  }

  template <typename WorkDesc>
  CUTE_DEVICE int valid_seq_len(WorkDesc work_desc, int blk_idx) {
    int remain_len = work_desc.seq_len - BlkSeqKV * blk_idx;
    return remain_len <= BlkSeqKV ? remain_len : BlkSeqKV;
  }
};

}  // namespace flat::collective
