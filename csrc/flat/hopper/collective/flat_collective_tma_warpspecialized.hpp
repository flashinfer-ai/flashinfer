#pragma once

#include "../../cute_ext.hpp"
#include "../collective/flat_collective_load.hpp"
#include "../collective/flat_collective_store.hpp"
#include "../collective/flat_common.hpp"
#include "../collective/flat_named_barriers.hpp"
#include "../kernel/flat_options.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

namespace flat::collective {

struct FlatNamedBarriers : FlatSharedNamedBarriers {
  static constexpr int QKLaunchedAndDecayComputed = FlatSharedNamedBarriers::NumBarriersUsed + 0;
};

using namespace cute;
using flat::kernel::find_option_t;
using flat::kernel::Tag;

template <class Element_, class ElementAccumulatorQK_, class ElementAccumulatorKV_,
          class TileShape_,  // (seqlen_q, seqlen_kv, d)
          class LayoutQ_, class LayoutK_, class LayoutV_, class LayoutO_,  // (seqlen_q/k, d, h)
          class Options>
struct FlatMainloopTmaWarpSpecialized {
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
  static constexpr int NumMmaWarpGroups =
      find_option_t<Tag::kNumMmaWarpGroups, Int<2>, Options>::value;
  static constexpr int StageCountQ = find_option_t<Tag::kStagesQ, Int<3>, Options>::value;
  static constexpr int StageCountK = find_option_t<Tag::kStagesK, Int<3>, Options>::value;
  static constexpr int StageCountV = find_option_t<Tag::kStagesV, Int<2>, Options>::value;

  static constexpr int NeedsScale =
      find_option_t<Tag::kNeedsScale, cute::true_type, Options>::value;
  static constexpr int NeedsDecay =
      find_option_t<Tag::kNeedsDecay, cute::true_type, Options>::value;

  static constexpr int NumLoadThreads = NumLoadWarpGroups * 128;
  static constexpr int NumMmaThreads = NumMmaWarpGroups * 128;

  using StagesQ = cutlass::gemm::collective::StageCount<StageCountQ>;
  using StagesK = cutlass::gemm::collective::StageCount<StageCountK>;
  using StagesV = cutlass::gemm::collective::StageCount<StageCountV>;
  using StagesO = cutlass::gemm::collective::StageCount<3>;
  using ClusterShape = Shape<_1, _1, _1>;

  // 16B alignment lets us use TMA
  static constexpr int Alignment = 16 / sizeof(Element);

  // Linear Attenion in math notation
  //   O_intra,i = (Qi @ Ki^T) @ Vi
  //   O_inter,i = Qi @ KVi
  //   KVi+1     = KVi + Ki^T @ Vi
  // with some elementwise opeartion omitted
  //
  // Following layout annotations are in M/N x K order to conform with MMA semantics
  //
  // Q, K and V are all in HeadSize-majored (K-major) mode in GMEM, then in mma operation form
  //   O_intra,i = mma(mma(Qi, Ki), select<1,0>(Vi))             QK: Blk x Blk  V:  Blk x Dv
  //   O_inter,i = mma(Qi, select<1, 0>(KVi))                    Q:  Blk x Dqk  KV: Dqk x Dv
  //   KVi+1     = KVi + mma(select<1,0>(Ki), select<1,0>(Vi))   KV: Dqk x Dv
  //
  // We want KV to be the A operand for MMA, we also want to avoid the KV transpose
  //   O_intra,i = mma(select<1,0>(Vi), mma(Qi, Ki))             GMEM  V: Blk x Dv   RMEM QK: Blk x
  //   Blk O_inter,i = mma(KVi, Qi)                                  RMEM KV:  Dv x Dqk  GMEM  Q:
  //   Blk x Dqk KVi+1     = KVi + mma(select<1,0>(Vi), select<1,0>(Ki))   GMEM  V: Blk x Dv   GMEM
  //   K: Dqk x Blk -> RMEM KV:  Dv x Dqk
  // This allows us to always keep KV in register.

  // We can further design G2S for Q and K as direct copy and G2S for V as transpose copy
  // The FINAL FORM will be:
  //   O_intra,i = mma(Vi, mma(Qi, Ki))                          SMEM  V:  Dv x Blk  RMEM QK: Blk x
  //   Blk O_inter,i = mma(KVi, Qi)                                  RMEM KV:  Dv x Dqk  SMEM  Q:
  //   Blk x Dqk KVi+1     = KVi + mma(Vi, select<1,0>(Ki))                SMEM  V:  Dv x Blk  SMEM
  //   K: Dqk x Blk -> RMEM KV:  Dv x Dqk

  static constexpr auto BlkSeqQ = get<0>(TileShape{});   // Blk_Q
  static constexpr auto BlkSeqKV = get<1>(TileShape{});  // Blk_K/V
  static constexpr auto HeadSize = get<2>(TileShape{});  // D (Dq, Dk, Dv all equal)
  static constexpr auto HeadSizeQK = HeadSize;
  static constexpr auto HeadSizeV = HeadSize;

  using TileShapeQK = decltype(make_shape(BlkSeqQ, BlkSeqKV, HeadSizeQK));
  using TileShapeKV = decltype(make_shape(HeadSizeV, HeadSizeQK, BlkSeqKV));

  // NOTE: describe MMA_O with respect to O_intra, aka, mma(V=(Dv,Blk_kv), QK=(??,Blk_kv)),
  // this might be useful when Blk is large, but HeadSize is small.
  // further notice, we need to Swap AB so that QK is operand A. All MMA planning shall be much more
  // natural than current ones.
  using TileShapeO2 = decltype(make_shape(HeadSizeV, BlkSeqQ, BlkSeqKV));

  // NOTE: describe MMA_O with respect to O_inter, aka, mma(KV=(Dv,Dqk), Q=(??,Dqk))
  // this might be useful when HeadSize is large, but Blk is small.
  // We arange the KV to be the operand A to MMA, so that we can totally avoid STS for KV.
  using TileShapeO1 = decltype(make_shape(HeadSizeV, BlkSeqQ, HeadSizeQK));

  static_assert(BlkSeqQ % 64 == 0);
  static_assert(BlkSeqQ == 64 || BlkSeqQ == 128);
  static constexpr bool IsQKCooperative = BlkSeqQ == 128;

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

  using TiledMmaQK = typename CollectiveMmaQK::TiledMma;
  using TiledMmaKV = decltype(convert_to_gmma_rs(typename CollectiveMmaKV::TiledMma{}));
  using TiledMmaO1 = decltype(convert_to_gmma_rs(typename CollectiveMmaO1::TiledMma{}));
  using TiledMmaO2 = decltype(convert_to_gmma_rs(typename CollectiveMmaO2::TiledMma{}));

  static constexpr int TiledMmaQKNumThreads = size(TiledMmaQK{});
  static_assert(size(TiledMmaQK{}) == (IsQKCooperative ? NumMmaThreads : 128));

  static_assert(size(TiledMmaKV{}) == NumMmaThreads);
  static_assert(size(TiledMmaO1{}) == NumMmaThreads);
  static_assert(size(TiledMmaO2{}) == NumMmaThreads);

  using CollectiveStoreO =
      CollectiveStoreTma<TileShapeO1, ClusterShape, ElementO, ElementAccumulatorO,
                         /*Seme*/ ElementO, decltype(select<1, 0, 2>(LayoutO{})), StagesO::value>;

  // layout for compute
  using QKSmemLayoutQ = SmemLayoutQ_SD;
  using QKSmemLayoutK = decltype(select_layout<1, 0, 2>(SmemLayoutK_DS{}));

  using KVSmemLayoutK = SmemLayoutK_DS;
  using KVSmemLayoutV = SmemLayoutV_DS;

  // layout for compute output
  using SmemLayoutQK =
      decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<Element>{}, select<0, 1>(TileShapeQK{})));
  using SmemLayoutO = typename CollectiveStoreO::SmemLayoutO;

  using MainloopQPipeline = cutlass::PipelineTmaAsync<StagesQ::value>;
  using MainloopKPipeline = cutlass::PipelineTmaAsync<StagesK::value>;
  using MainloopVPipeline = cutlass::PipelineTmaAsync<StagesV::value>;
  using MainloopOPipeline = typename CollectiveStoreO::Pipeline;

  using QPipelineState = typename cutlass::PipelineState<MainloopQPipeline::Stages>;
  using KPipelineState = typename cutlass::PipelineState<MainloopKPipeline::Stages>;
  using VPipelineState = typename cutlass::PipelineState<MainloopVPipeline::Stages>;
  using OPipelineState = typename CollectiveStoreO::PipelineState;

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

    SharedStorageO smem_o;
    cute::array_aligned<float, cute::max(BlkSeqQ, BlkSeqKV) + 1> decay;
  };

  using TMA_Q = typename CollectiveMmaQK::Params::TMA_A;
  using TMA_K = typename CollectiveMmaKV_G2S::Params::TMA_B;
  using TMA_V = typename CollectiveMmaKV_G2S::Params::TMA_A;
  using TMA_O = typename CollectiveStoreO::Params::TMA_O;

  using LoadQ = CollectiveLoadTma<LoadKind::kQ, MainloopQPipeline, Element, QKSmemLayoutQ, TMA_Q>;
  using LoadK = CollectiveLoadTma<LoadKind::kK, MainloopKPipeline, Element, KVSmemLayoutK, TMA_K>;
  using LoadV = CollectiveLoadTma<LoadKind::kV, MainloopVPipeline, Element, KVSmemLayoutV, TMA_V>;

  struct Arguments {  // clang-format off
    Element const* ptr_Q; LayoutQ dQ;
    Element const* ptr_K; LayoutK dK;
    Element const* ptr_V; LayoutV dV;
    Element*       ptr_O; LayoutO dO;
    float*        ptr_output_state; // layout fixed (kdim, vdim, num_heads, num_seqs):LayoutLeft{}
    float const*  ptr_input_state;
    float scale;
    float decay;
    float const* per_head_decay = nullptr;
    int decay_exponent_offset = 0;
  };  // clang-format on

  struct Params {
    TMA_Q tma_load_q;
    TMA_K tma_load_k;
    TMA_V tma_load_v;
    TMA_O tma_store_o;
    Element* ptr_O;
    LayoutV dO;
    void* tensormaps;
    float scale;

    float* ptr_output_state;
    float const* ptr_input_state;

    float decay;
    float const* per_head_decay;
    int decay_exponent_offset;
  };

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_size, Arguments const& args) {
    auto ratio = problem_size.num_q_heads / problem_size.num_k_heads;
    bool is_gqa_like = (problem_size.num_k_heads == problem_size.num_v_heads) &&
                       (problem_size.num_q_heads == ratio * problem_size.num_k_heads) &&
                       (problem_size.num_q_heads == ratio * problem_size.num_v_heads);

    return true && is_gqa_like && (problem_size.head_size <= get<2>(TileShape{})) &&
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

    auto params_kv = CollectiveMmaKV_G2S::to_underlying_arguments(
        make_shape(d, d, s, problem_size.num_k_heads),
        typename CollectiveMmaKV_G2S::Arguments{
            args.ptr_V, select<1, 0, 2>(args.dV),  // used as G2S for V
            args.ptr_K, select<1, 0, 2>(args.dK),  // used as G2S for K
        },
        /*workspace=*/nullptr);

    auto params_o = CollectiveStoreO::to_underlying_arguments(
        make_shape(d, s, d, problem_size.num_o_heads),  // in O1
        // make_shape(d, s, s, problem_size.num_o_heads),  // in O2
        typename CollectiveStoreO::Arguments{args.ptr_O, select<1, 0, 2>(args.dO)}, workspace);

    return Params{
        .tma_load_q = params_qk.tma_load_a,
        .tma_load_k = params_kv.tma_load_b,
        .tma_load_v = params_kv.tma_load_a,
        .tma_store_o = params_o.tma_store_o,
        .tensormaps = params_o.tensormaps,
        .scale = args.scale,

        .ptr_output_state = args.ptr_output_state,
        .ptr_input_state = args.ptr_input_state,

        .decay = args.decay,
        .per_head_decay = args.per_head_decay,
        .decay_exponent_offset = args.decay_exponent_offset,
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

  CUTE_DEVICE static void precompute_decay(float decay_factor, float* decay) {
    constexpr int len = cute::max(BlkSeqQ, BlkSeqKV) + 1;

    int lane_id = cutlass::canonical_lane_idx();
    float scale = decay_factor;

    for (int base = 0; base < len; base += cutlass::NumThreadsPerWarp) {
      float prod = scale;
      for (int offset = 1; offset < cutlass::NumThreadsPerWarp; offset *= 2) {
        auto v = __shfl_xor_sync(0xFFFFFFFF, prod, offset);
        if (lane_id > offset) {
          prod *= v;
        }
      }

      int vidx = base + lane_id;
      if (vidx == base) {
        prod = 1.0f;
      };  // correct prod as d^0, d^1, ..., d^tid, ... for first iter
      if (base != 0) {
        prod *= scale * decay[base - 1];
      }  // correct prod as d^(tid+1) for remaining iters
      if (vidx < len) {
        decay[vidx] = prod;
      }
    }
  }

  template <typename CollectiveLoad, typename TMA, typename ProblemShape, typename LoadTileShape,
            typename WorkDesc, class MainloopPipeline, class PipelineState, class SharedStorage>
  CUTE_DEVICE void load(TMA const& tma_load, ProblemShape const& problem_size,
                        LoadTileShape const& load_tile_shape, WorkDesc const& work_desc,
                        MainloopPipeline& pipeline, PipelineState& smem_pipe_write,
                        SharedStorage& storage) {
    int32_t num_blocks = ceil_div(work_desc.seq_len, get<0>(TileShape{}));
    uint32_t lane_predicate = cute::elect_one_sync();

    auto collective_load = CollectiveLoad{tma_load, pipeline, storage};
    auto src_dst = collective_load.partition_SD(problem_size, load_tile_shape, work_desc);

    CUTE_NO_UNROLL
    for (int blk = 0; blk < num_blocks; ++blk) {
      DPRINTF0_W("%s collective_load.step blk_idx:%d -> smem_pipe_write:%d, num_blocks:%d\n",
                 to_string(collective_load.kind), blk, smem_pipe_write.index(), num_blocks);
      collective_load.step(src_dst, blk, smem_pipe_write, lane_predicate);
    }
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
                           OPipelineState& o_smem_pipe_write, SharedStorage& storage) {
    // MAKE NVCC HAPPY!
    constexpr auto zero = Element{};

    int32_t num_blocks = ceil_div(work_desc.seq_len, get<0>(TileShape{}));
    DPRINTF0_WG("num_blocks: %d\n", num_blocks);

    int thread_idx = int(threadIdx.x) - NumLoadThreads;

    float scale = NeedsScale ? params.scale : 1.0f;

    bool is_per_head_decay = params.per_head_decay != nullptr;
    float decay_factor =
        is_per_head_decay ? params.per_head_decay[work_desc.q_head_idx()] : params.decay;
    float block_decay = 1.0f;
    bool warp_precompute = (thread_idx / cutlass::NumThreadsPerWarp) == (NumMmaWarpGroups * 4 - 1);
    if constexpr (NeedsDecay) {
      if (warp_precompute) {
        precompute_decay(decay_factor, storage.decay.data());
      }
    }

    auto valid_seq_len = [&](int blk) {  // in block
      int remain_len = work_desc.seq_len - BlkSeqKV * blk;
      return remain_len <= BlkSeqKV ? remain_len : BlkSeqKV;
    };

    Tensor sQqk = make_tensor(make_smem_ptr(storage.smem_q.data()), QKSmemLayoutQ{});
    Tensor sKqk = make_tensor(make_smem_ptr(storage.smem_k.data()), QKSmemLayoutK{});
    Tensor sKkv = make_tensor(make_smem_ptr(storage.smem_k.data()), KVSmemLayoutK{});
    Tensor sVkv = make_tensor(make_smem_ptr(storage.smem_v.data()), KVSmemLayoutV{});
    Tensor sQK = make_tensor(make_smem_ptr(storage.smem_qk.data()), SmemLayoutQK{});
    Tensor sO = make_tensor(make_smem_ptr(storage.smem_o.data()), SmemLayoutO{});

    ///////////////////////////////////////////////////////////////////////////
    // Q@K
    auto qk_tiled_mma = TiledMmaQK{};
    auto qk_thread_idx =
        IsQKCooperative ? thread_idx : (thread_idx % cutlass::NumThreadsPerWarpGroup);
    auto qk_thr_mma = qk_tiled_mma.get_thread_slice(qk_thread_idx);

    Tensor tQKsQ = qk_thr_mma.partition_A(sQqk);
    Tensor tQKsK = qk_thr_mma.partition_B(sKqk);
    Tensor tQKrQ = qk_thr_mma.make_fragment_A(tQKsQ);
    Tensor tQKrK = qk_thr_mma.make_fragment_B(tQKsK);

    auto cM0 = make_identity_tensor(select<0, 1>(TileShapeQK{}));  // (QTok, KTok)
    auto tQKcM0 = qk_thr_mma.partition_C(cM0);                     // (idx) -> (tok_q, tok_k)

    ///////////////////////////////////////////////////////////////////////////
    // K@V
    auto kv_tiled_mma = TiledMmaKV{};
    auto kv_thr_mma = kv_tiled_mma.get_thread_slice(thread_idx);

    using KV_S2R = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
    auto tKVrV_thr_copy = make_tiled_copy_A(KV_S2R{}, kv_tiled_mma).get_thread_slice(thread_idx);

    Tensor tKVrKV = partition_fragment_C(kv_thr_mma, select<0, 1>(TileShapeKV{}));

    Tensor tKVrV = kv_thr_mma.partition_fragment_A(sVkv(_, _, _0{}));  // mma src
    Tensor tKVrV_cv = tKVrV_thr_copy.retile_D(tKVrV);                  // copy view dst
    Tensor tKVsV = tKVrV_thr_copy.partition_S(sVkv);                   // copy view src

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

    using DecayLayout = Layout<decltype(select<0, 1>(TileShapeO1{})), Stride<_0, _1>>;
    auto tOrDecay = o1_thr_mma.partition_fragment_C(make_tensor<float>(DecayLayout{}));
    if constexpr (!NeedsDecay) {
      fill(tOrDecay, 1.0f);
    }

    auto const seq_idx = work_desc.seq_idx;
    auto const q_head_idx = work_desc.q_head_idx();
    auto const k_head_idx = work_desc.k_head_idx();
    auto const v_head_idx = work_desc.v_head_idx();

    bool const wg_compute_qk = IsQKCooperative || thread_idx / TiledMmaQKNumThreads == 0;

    auto qk_mask = [&](auto& tQKrQK, auto is_final_block_, auto B /*valid seqlen*/) {
      constexpr bool is_final_block = decltype(is_final_block_)::value;
      DPRINTF0_W("[%d,%d,%d,%d]** compute masked_QK\n", seq_idx, q_head_idx, k_head_idx,
                 v_head_idx);
      cute::transform(tQKrQK, tQKcM0, tQKrQK, [&](auto val, auto coord) {
        auto [s, t] = coord;
        auto scaled = s >= t ? scale * val : decltype(val){};  // also masked
        if constexpr (is_final_block) {
          scaled = s < B || t < B ? scaled : decltype(val){};
        }
        return scaled;
      });
      if constexpr (NeedsDecay) {
        cute::transform(tQKrQK, tQKcM0, tQKrQK, [&](auto val, auto coord) {
          auto [s, t] = coord;
          auto Lambda = s >= t ? storage.decay[s - t] : 0.0f;
          return val * Lambda;
        });
      }
    };

    auto qk_store = [&](auto const& tQKrQK) {
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

    auto kv_decay_v = [&](auto& tKVrV, auto is_final_block_, auto B) {
      constexpr bool is_final_block = decltype(is_final_block_)::value;
      if constexpr (NeedsDecay) {  // decay by Lambda * lambda^(-tok)
        cute::transform(tKVrV, tKVcV, tKVrV, [&](auto val, auto coord) {
          auto tok = get<1>(coord);
          float decay_v = [&] {
            if constexpr (!is_final_block) {
              return storage.decay[B - tok - params.decay_exponent_offset];
            } else {
              return tok < B ? storage.decay[B - tok - params.decay_exponent_offset] : 0.0f;
            }
          }();
          return decltype(val)(val * decay_v);
        });
      }
      if constexpr (is_final_block) {
        if constexpr (!NeedsDecay) {
          cute::transform(tKVrV, tKVcV, tKVrV, [&](auto val, auto coord) {
            auto tok = get<1>(coord);
            return tok < B ? val : zero;  // mask v of tail oob values
          });
        }
      }
    };

    // auto kv_mma = [&]() {};

    auto kv_epi = [&](auto& tKVrKV, auto& tKVrKV_inc, auto is_first_block_, auto is_final_block_,
                      auto B /*valid seqlen*/) {
      constexpr bool is_first_block = decltype(is_first_block_)::value;
      constexpr bool is_final_block = decltype(is_final_block_)::value;

      if constexpr (NeedsScale) {
        cute::transform(tKVrKV_inc, tKVrKV_inc, [&](auto val) { return val * scale; });
      }

      if constexpr (is_final_block || is_first_block) {
        if constexpr (NeedsDecay) {
          block_decay = storage.decay[B];
        }
      }
      cute::transform(tKVrKV, tKVrKV_inc, tKVrKV, [&](auto carried_kv, auto inc_kv) {
        return block_decay * carried_kv + inc_kv;
      });
    };

    auto kv_load = [&](auto& tKVrKV) {
      DPRINTF0_WG("[%d,%d,%d,%d]>> load tKVgKV -> tKVrKV\n", seq_idx, q_head_idx, k_head_idx,
                  v_head_idx);
      int num_state_heads = is_per_head_decay ? problem_size.num_q_heads : problem_size.num_k_heads;
      int state_head_idx = is_per_head_decay ? work_desc.q_head_idx() : work_desc.k_head_idx();
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

    auto kv_store = [&]() {  // tKVrKV is carried over whole mainloop
      bool is_lead_kv_head =
          (q_head_idx % (problem_size.num_q_heads / problem_size.num_k_heads)) == 0;
      if (is_per_head_decay || is_lead_kv_head) {
        DPRINTF0_WG("[%d,%d,%d,%d]>> save tKVrKV -> tKVgKV\n", seq_idx, q_head_idx, k_head_idx,
                    v_head_idx);
        int num_state_heads =
            is_per_head_decay ? problem_size.num_q_heads : problem_size.num_k_heads;
        int state_head_idx = is_per_head_decay ? work_desc.q_head_idx() : work_desc.k_head_idx();
        auto gKV = make_tensor(make_gmem_ptr(params.ptr_output_state),
                               make_layout(make_shape(Int<HeadSizeQK>{}, Int<HeadSizeV>{},
                                                      num_state_heads, problem_size.num_seqs)))(
            _, _, state_head_idx, seq_idx);  // (KDim, VDim), K-contiguous

        auto tiled_copy_kv =
            make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, Element>{}, kv_tiled_mma);
        auto thr_copy_kv = tiled_copy_kv.get_thread_slice(thread_idx);

        auto tKVgKV = thr_copy_kv.partition_D(select_tensor<1, 0>(gKV));
        copy(tKVrKV, tKVgKV);
      }
    };

    auto o1_load_decay = [&](auto& tOrDecay) {
      if constexpr (NeedsDecay) {
        cute::transform(tOcO, tOrDecay, [&](auto coord) {
          auto [_, tok] = coord;
          return storage.decay[tok + params.decay_exponent_offset];
        });
      }
    };

    auto o_epi_and_store = [&](auto tOrO1, auto tOrO2, auto const& tOrDecay, auto is_first_block_) {
      constexpr bool is_first_block = decltype(is_first_block_)::value;
      // cvt acc dtype to output dtype
      auto tOrO1_cv = thr_copy_o.retile_S(tOrO1);
      auto tOrO2_cv = thr_copy_o.retile_S(tOrO2);
      auto tOrDecay_cv = thr_copy_o.retile_S(tOrDecay);
      auto tOrO_cvt_cv = make_fragment_like<ElementO>(tOrO1_cv);
      DPRINTF0_WG("compute: o_pipeline.producer_wait: smem_pipe_write:%d\n",
                  o_smem_pipe_write.index());
      o_pipeline.producer_acquire(o_smem_pipe_write);
      if constexpr (is_first_block) {
        cute::transform(tOrO2_cv, tOrO_cvt_cv, [](auto v) { return ElementO(v); });
      } else {
        CUTE_UNROLL
        for (int i = 0; i < size(tOrO_cvt_cv); ++i) {
          tOrO_cvt_cv(i) = ElementO{fmaf(tOrDecay_cv(i), tOrO1_cv(i), tOrO2_cv(i))};
        }
      }
      cutlass::arch::fence_view_async_shared();
      copy(tiled_copy_o, tOrO_cvt_cv, tOsO(_, _, _, o_smem_pipe_write.index()));
      cutlass::arch::fence_view_async_shared();
      o_pipeline.producer_commit(o_smem_pipe_write);
      ++o_smem_pipe_write;
    };

    auto compute_loop_body = [&](int blk, auto is_first_block_, auto is_final_block_) {
      constexpr bool is_first_block = decltype(is_first_block_)::value;
      constexpr bool is_final_block = decltype(is_final_block_)::value;
      int B = is_final_block ? valid_seq_len(blk) : BlkSeqKV;

      DPRINTF0_WG("compute: q_pipeline.consumer_wait: smem_pipe_read:%d\n",
                  q_smem_pipe_read.index());
      q_pipeline.consumer_wait(q_smem_pipe_read);
      DPRINTF0_WG("compute: k_pipeline.consumer_wait: smem_pipe_read:%d\n",
                  k_smem_pipe_read.index());
      k_pipeline.consumer_wait(k_smem_pipe_read);

      Tensor tQKrQK = partition_fragment_C(qk_tiled_mma, select<0, 1>(TileShapeQK{}));
      if (wg_compute_qk) {
        DPRINTF0_WG("[%d,%d,%d,%d]** dispatch QK WGMMA\n", seq_idx, q_head_idx, k_head_idx,
                    v_head_idx);
        warpgroup_fence_operand(tQKrQK);
        warpgroup_arrive();
        gemm_zero_acc(qk_tiled_mma, tQKrQ(_, _, _, q_smem_pipe_read.index()),
                      tQKrK(_, _, _, k_smem_pipe_read.index()), tQKrQK);
        warpgroup_commit_batch();  // q@k batch
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads,
                                            FlatNamedBarriers::QKLaunchedAndDecayComputed);
        warpgroup_wait<0>();  // q@k batch finished
      } else {
        cutlass::arch::NamedBarrier::arrive_and_wait(NumMmaThreads,
                                                     FlatNamedBarriers::QKLaunchedAndDecayComputed);
      }
      if (wg_compute_qk) {
        qk_mask(tQKrQK, is_final_block_, B);
      }
      if (wg_compute_qk) {
        qk_store(tQKrQK);
      }

      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(NumMmaThreads,
                                        FlatNamedBarriers::AllMmaThreadsSync);  // qk -> smem

      /////////////////////////////////////////////////////////////////////////
      // 2. compute qkv
      // 2.1 Q @ KV, NOTE: use old KV here
      DPRINTF0_WG("[%d,%d,%d,%d]** dispatch O WGMMA\n", seq_idx, q_head_idx, k_head_idx,
                  v_head_idx);
      auto tOrO1 = partition_fragment_C(o1_thr_mma, select<0, 1>(TileShapeO1{}));
      if constexpr (!is_first_block) {
        Tensor tOrKV = make_acc_into_op<Element>(tKVrKV, typename TiledMmaO1::LayoutA_TV{});
        warpgroup_fence_operand(tOrKV);
        warpgroup_fence_operand(tOrO1);
        warpgroup_arrive();
        gemm_zero_acc(o1_thr_mma, tOrKV, tOrQ(_, _, _, q_smem_pipe_read.index()), tOrO1);
        warpgroup_commit_batch();  // q@kv batch
        warpgroup_wait<0>();       // q@kv batch
      }
      DPRINTF0_WG("compute: q_pipeline.consumer_release: smem_pipe_read:%d\n",
                  q_smem_pipe_read.index());
      q_pipeline.consumer_release(q_smem_pipe_read);
      ++q_smem_pipe_read;

      DPRINTF0_WG("compute: v_pipeline.consumer_wait: smem_pipe_read:%d\n",
                  v_smem_pipe_read.index());
      v_pipeline.consumer_wait(v_smem_pipe_read);
      copy(KV_S2R{}, tKVsV(_, _, _, v_smem_pipe_read.index()),
           tKVrV_cv);  // NOTE: tKVsV and tOrV have the same layout
      auto tOrO2 = make_fragment_like(tOrO1);
      warpgroup_fence_operand(tKVrV);
      warpgroup_fence_operand(tOrO2);
      warpgroup_arrive();
      gemm_zero_acc(o2_tiled_mma, tKVrV, tOrQK, tOrO2);
      warpgroup_commit_batch();  // qk@v batch
      warpgroup_wait<0>();       // qk@v batch
      if (blk == 0) {
        o1_load_decay(tOrDecay);
      }
      o_epi_and_store(tOrO1, tOrO2, tOrDecay, is_first_block_);

      /////////////////////////////////////////////////////////////////////////
      // 3. update KV
      DPRINTF0_WG("[%d,%d,%d,%d]** dispatch KV WGMMA\n", seq_idx, q_head_idx, k_head_idx,
                  v_head_idx);
      // copy(KV_S2R{}, tKVsV(_, _, _, v_smem_pipe_read.index()), tKVrV_cv);  // already loaded for
      // O2
      kv_decay_v(tKVrV, is_final_block_, B);
      auto tKVrKV_inc = make_tensor_like(tKVrKV);
      warpgroup_fence_operand(tKVrV);
      warpgroup_fence_operand(tKVrKV_inc);
      warpgroup_arrive();
      gemm_zero_acc(kv_tiled_mma, tKVrV, tKVrK(_, _, _, k_smem_pipe_read.index()), tKVrKV_inc);
      warpgroup_commit_batch();  // k@v batch
      warpgroup_wait<0>();

      DPRINTF0_WG("compute: k_pipeline.consumer_release: smem_pipe_read:%d\n",
                  k_smem_pipe_read.index());
      k_pipeline.consumer_release(k_smem_pipe_read);
      ++k_smem_pipe_read;

      kv_epi(tKVrKV, tKVrKV_inc, is_first_block_, is_final_block_, B);

      DPRINTF0_WG("compute: v_pipeline.consumer_release: smem_pipe_read:%d\n",
                  v_smem_pipe_read.index());
      v_pipeline.consumer_release(v_smem_pipe_read);
      ++v_smem_pipe_read;
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
};

}  // namespace flat::collective
