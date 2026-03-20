/*
# SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
# ==============================================================================
*/
#include <Bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x64x128u2_s5_et128x64_m256x64x16_cga2x1x1_16dp256b_rM_BN_transOut_schPd2x1x2x3_bN_tma_tmaSf_rgTma_clmp_swiGlu_dynB_sm100f.h>
namespace batchedGemm {


struct WorkIdStack {
  trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  cutlass::gemm::kernel::detail::
    PersistentTileSchedulerSm100<cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, 3>
      mScheduler;
  typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
    3>::WorkTileInfo workTileInfo;
  inline __device__ WorkIdStack(WorkIdSmem& workIdSmem,
                                WorkIdSmemBarrier& workIdSmemBarrier,
                                KernelParams const& params,
                                int32_t warpId,
                                int32_t barInitWarpId,
                                int32_t orderedSequenceGroupId)
    : mPipeline{workIdSmemBarrier.mBarriers, int32_t{1}, int32_t{704}, int32_t{0}, barInitWarpId}
    , mScheduler{&workIdSmem.workIdResponse[int32_t{0}],
                 typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<
                   cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
                   3>::Params{},
                 cute::block_id_in_cluster()}
    , workTileInfo{mScheduler.initial_work_tile_info(CuteFlatTuple378{})} {}
};
struct WorkThrottleBarrierStack {
  trtllm::dev::CutlassCpAsyncPipeline<3> mPipeline;
  inline __device__ WorkThrottleBarrierStack(
    WorkThrottleBarrierSmemBarrier& workThrottleBarrierSmemBarrier,
    int32_t warpId,
    int32_t barInitWarpId,
    int32_t orderedSequenceGroupId)
    : mPipeline{workThrottleBarrierSmemBarrier.mBarriers,
                warpId,
                int32_t{32},
                int32_t{32},
                barInitWarpId} {}
};
struct SmemBufferStack {
  int8_t* mPtr;
  inline __device__ SmemBufferStack(SmemBufferSmem& smemBufferSmem,
                                    int32_t warpId,
                                    int32_t barInitWarpId,
                                    int32_t orderedSequenceGroupId)
    : mPtr{&smemBufferSmem.mArray[int32_t{0}]} {}
};
struct SmemAStack {
  int8_t* mDepSmemPtr2;
  trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
    5,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  inline __device__ SmemAStack(SmemASmem& smemASmem,
                               SmemASmemBarrier& smemASmemBarrier,
                               SmemBufferSmem& smemBufferSmem,
                               SmemBufferStack& smemBufferStack,
                               int32_t warpId,
                               int32_t barInitWarpId,
                               int32_t orderedSequenceGroupId)
    : mDepSmemPtr2{&smemBufferSmem.mArray[int32_t{0}]}
    , mPipeline{smemASmemBarrier.mBarriers,
                warpId,
                int32_t{65536},
                ((warpId) == (barInitWarpId)) && (bool{cute::elect_one_sync()}),
                int32_t{1},
                CuteFlatTuple617{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId} {}
};
struct SmemBStack {
  int8_t* mDepSmemPtr2;
  trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
    5,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  inline __device__ SmemBStack(SmemBSmem& smemBSmem,
                               SmemBSmemBarrier& smemBSmemBarrier,
                               SmemBufferSmem& smemBufferSmem,
                               SmemBufferStack& smemBufferStack,
                               int32_t warpId,
                               int32_t barInitWarpId,
                               int32_t orderedSequenceGroupId)
    : mDepSmemPtr2{&smemBufferSmem.mArray[int32_t{0}]}
    , mPipeline{smemBSmemBarrier.mBarriers,
                warpId,
                int32_t{16384},
                ((warpId) == (barInitWarpId)) && (bool{cute::elect_one_sync()}),
                int32_t{1},
                CuteFlatTuple743{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId} {}
};
struct TmemStack {
  inline __device__ TmemStack(int32_t warpId,
                              int32_t barInitWarpId,
                              int32_t orderedSequenceGroupId) {}
};
struct Mma0Stack {
  trtllm::dev::CutlassUmmaAsyncPipeline<2, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  inline __device__ Mma0Stack(Mma0SmemBarrier& mma0SmemBarrier,
                              TmemStack& tmemStack,
                              int32_t warpId,
                              int32_t barInitWarpId,
                              int32_t orderedSequenceGroupId)
    : mPipeline{mma0SmemBarrier.mBarriers,
                warpId,
                int32_t{256},
                CuteFlatTuple866{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId} {}
};
struct GmemC0Stack {
  int8_t* mDepSmemPtr2;
  inline __device__ GmemC0Stack(GmemC0Smem& gmemC0Smem,
                                SmemBufferSmem& smemBufferSmem,
                                SmemBufferStack& smemBufferStack,
                                int32_t warpId,
                                int32_t barInitWarpId,
                                int32_t orderedSequenceGroupId)
    : mDepSmemPtr2{&smemBufferSmem.mArray[int32_t{0}]} {}
};
struct ClusterBarrierBuffersStack {
  trtllm::dev::CutlassClusterBarrier<cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>, true>
    mClusterBarrier;
  inline __device__ ClusterBarrierBuffersStack(
    ClusterBarrierBuffersSmemBarrier& clusterBarrierBuffersSmemBarrier,
    int32_t warpId,
    int32_t barInitWarpId,
    int32_t orderedSequenceGroupId)
    : mClusterBarrier{clusterBarrierBuffersSmemBarrier.mClusterSmemBarriers,
                      warpId,
                      int32_t{32},
                      int32_t{0}} {}
};
struct KernelState {
  int32_t const mNumNonExitingCtas;
  int32_t const mThreadIdx;
  uint32_t* const mTmemSwStatePtr;
  int32_t const mWarpIdx;
  inline __device__ KernelState(KernelParams const& params, uint32_t* tmemSwStatePtr)
    : mNumNonExitingCtas{params.ptrNumNonExitingCtas[int32_t{0}]}
    , mThreadIdx{reinterpret_cast<int32_t const&>(threadIdx.x)}
    , mTmemSwStatePtr{tmemSwStatePtr}
    , mWarpIdx{
        __shfl_sync(uint32_t{0xffffffff}, (mThreadIdx) / (int32_t{32}), int32_t{0}, int32_t{32})} {}
};
struct LoadTaskA {
  int32_t mCtaIdxY;
  int32_t mBatchIdx;
  int32_t mBatchLimit;
  int32_t mCtaOffsetK;
  int32_t mCtaIdxX;
  int32_t mCtaIdxZ;
  dim3 const mBlockInClusterIdx;
  uint16_t const mMmaCtaMask;
  inline __device__ LoadTaskA(KernelParams const& params,
                              KernelState const& state,
                              int32_t warpGrpStart)
    : mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mBatchIdx{int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}}
    , mBatchLimit{int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]}}
    , mCtaOffsetK{int32_t{0}}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)}
    , mBlockInClusterIdx{cute::block_id_in_cluster()}
    , mMmaCtaMask{uint16_t{
        __shfl_sync(uint32_t{0xffffffff}, trtllm::dev::getCtaMask(), int32_t{0}, int32_t{32})}} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{5})) && ((state.mWarpIdx) < (int32_t{6}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 SmemASmem& smemADstSmem,
                                 SmemAStack& smemADstStack,
                                 WorkThrottleBarrierStack& workThrottleBarrierDstStack,
                                 WorkIdSmem& workIdSrcSmem,
                                 WorkIdStack& workIdSrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<96>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsState{};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsReleaseState{};
    int32_t workIdConsToken{int32_t{0}};
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      5,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemAProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    int32_t smemAProdToken{int32_t{1}};
    trtllm::dev::CutlassCpAsyncPipeline<3>::PipelineState workThrottleBarrierProdState{int32_t{0},
                                                                                       int32_t{1},
                                                                                       int32_t{0}};
    int32_t workThrottleBarrierProdToken{int32_t{1}};
    do {
      int32_t paddedPerCtaK{(((params.k) + (int32_t{127})) / (int32_t{128})) * (int32_t{128})};
      int32_t loopEnd{paddedPerCtaK};
      bool const hasOneLoopIter{(int32_t{0}) < (loopEnd)};
      if ((mCtaIdxY) >= (state.mNumNonExitingCtas)) {
        goto ExitTileWithoutSignalingLabel;
      }
      mBatchIdx = int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]};
      mBatchLimit = int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]};
      //
      // smemA [HoistProdTryAcquire].
      //
      if ((int32_t{0}) < (loopEnd)) {
        smemAProdToken = smemADstStack.mPipeline.producer_try_acquire(smemAProdState);
      }
      //
      // Hoist the first iter.
      //
      //
      // workThrottleBarrier [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{4608}].
      //
      if ((mBlockInClusterIdx.x) == (int32_t{0})) {
        {
          workThrottleBarrierProdToken = workThrottleBarrierDstStack.mPipeline.producer_try_acquire(
            workThrottleBarrierProdState);
        }
      }
      if ((mBlockInClusterIdx.x) == (int32_t{0})) {
        workThrottleBarrierDstStack.mPipeline.producer_acquire(workThrottleBarrierProdState,
                                                               workThrottleBarrierProdToken);
      }
      //
      // workThrottleBarrier [ProdCommit, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{4608}].
      //
      if ((mBlockInClusterIdx.x) == (int32_t{0})) {
        {
          workThrottleBarrierDstStack.mPipeline.producer_commit(workThrottleBarrierProdState);
        }
        ++workThrottleBarrierProdState;
      }
      //
      // Loop body.
      //
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset389 = int32_t{0}; loopOffset389 < loopEnd;
           loopOffset389 += int32_t{128}) {
        bool const isLastLoopIter{((loopOffset389) + (int32_t{128})) >= (loopEnd)};
        //
        // gmemA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK3;
        { tileOffsetK3 = loopOffset389; }
        //
        // smemA [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          smemADstStack.mPipeline.producer_acquire(smemAProdState, smemAProdToken);
          if (((loopOffset389) + (int32_t{128})) < (loopEnd)) {
            smemAProdToken = smemADstStack.mPipeline.producer_try_acquire(
              trtllm::dev::makePipelineState(smemAProdState, int32_t{1}));
          }
        }
        //
        // smemA [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK5;
        tileOffsetK5 = tileOffsetK3;
        {
          uint64_t* barrier{smemADstStack.mPipeline.producer_get_barrier(smemAProdState)};
          int32_t index{smemAProdState.index()};
          {}
          {
            int8_t* smemBytesBasePtrA;
            int8_t* smemBytesStagePtrA;
            smemBytesBasePtrA =
              reinterpret_cast<int8_t*>(smemADstStack.mDepSmemPtr2) + (int32_t{0});
            smemBytesStagePtrA = smemBytesBasePtrA + ((index) * (int32_t{32768}));
            int32_t coords[4];
            coords[int32_t{0}] = int32_t{0};
            coords[int32_t{1}] = (mCtaIdxX) * (int32_t{128});
            coords[int32_t{2}] = (tileOffsetK5) / (int32_t{64});
            coords[int32_t{3}] = mBatchIdx;
            uint64_t* leadCtaMbar;
            leadCtaMbar = cuda_ptx::mapa(cuda_ptx::space_cluster_t{},
                                         barrier,
                                         int32_t{trtllm::dev::getLeadCtaRank()});
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::cp_async_bulk_tensor(
                cuda_ptx::space_cluster_t{},
                cuda_ptx::space_global_t{},
                cuda_ptx::cta_group_2_t{},
                &reinterpret_cast<cutlass::bfloat16_t*>(smemBytesStagePtrA)[int32_t{0}],
                params.tmaA,
                coords,
                leadCtaMbar,
                mMmaCtaMask);
            }
          }
        }
        //
        // smemA [ProdPreCommit, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset389) >= (int32_t{0})) {
        }
        //
        // smemA [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { smemADstStack.mPipeline.producer_commit(smemAProdState); }
          ++smemAProdState;
        }
        //
        // gmemA [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        //
        // smemA [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
      }
      //
      // Tail work.
      //
      //
      // gmemA [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
      //
      // workId [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
      //
      // smemA [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
    ExitTileWithSignalingLabel:
    ExitTileWithoutSignalingLabel:
      auto newWorkTileInfoTuple{
        workIdSrcStack.mScheduler.fetch_next_work(workIdSrcStack.workTileInfo,
                                                  workIdSrcStack.mPipeline,
                                                  workIdConsState)};
      workIdSrcStack.workTileInfo = cute::get<int32_t{0}>(newWorkTileInfoTuple);
      ++workIdConsState;
      mCtaIdxX = workIdSrcStack.workTileInfo.M_idx;
      mCtaIdxY = workIdSrcStack.workTileInfo.N_idx;
      mCtaIdxZ = workIdSrcStack.workTileInfo.L_idx;
    } while (workIdSrcStack.workTileInfo.is_valid_tile);
  }
};
struct LoadTaskB {
  int32_t mCtaIdxY;
  int32_t mBatchIdx;
  int32_t mBatchLimit;
  int32_t const mWarpGrpThreadIdx;
  int32_t mRoutedIndices[8];
  int32_t mCtaOffsetK;
  int32_t mCtaIdxX;
  int32_t mCtaIdxZ;
  int32_t const mWarpGrpWarpIdx;
  uint16_t const mMmaCtaMask;
  inline __device__ LoadTaskB(KernelParams const& params,
                              KernelState const& state,
                              int32_t warpGrpStart)
    : mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mBatchIdx{int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}}
    , mBatchLimit{int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]}}
    , mWarpGrpThreadIdx{min(int32_t{128},
                            max(int32_t{0}, (state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))))}
    , mCtaOffsetK{int32_t{0}}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)}
    , mWarpGrpWarpIdx{(state.mWarpIdx) - (warpGrpStart)}
    , mMmaCtaMask{uint16_t{
        __shfl_sync(uint32_t{0xffffffff}, trtllm::dev::getCtaMask(), int32_t{0}, int32_t{32})}} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{6})) && ((state.mWarpIdx) < (int32_t{10}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 SmemBSmem& smemBDstSmem,
                                 SmemBStack& smemBDstStack,
                                 WorkIdSmem& workIdSrcSmem,
                                 WorkIdStack& workIdSrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<96>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsState{};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsReleaseState{};
    int32_t workIdConsToken{int32_t{0}};
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      5,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemBProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    int32_t smemBProdToken{int32_t{1}};
    do {
      int32_t paddedPerCtaK{(((params.k) + (int32_t{127})) / (int32_t{128})) * (int32_t{128})};
      int32_t loopEnd{paddedPerCtaK};
      bool const hasOneLoopIter{(int32_t{0}) < (loopEnd)};
      if ((mCtaIdxY) >= (state.mNumNonExitingCtas)) {
        goto ExitTileWithoutSignalingLabel;
      }
      mBatchIdx = int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]};
      mBatchLimit = int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]};
      {
        if (((mWarpGrpWarpIdx) * (int32_t{4})) < (int32_t{32})) {
          mRoutedIndices[int32_t{0}] = int32_t{
            params.ptrRouteMap[((mWarpGrpWarpIdx) * (int32_t{4})) +
                               ((mCtaIdxY) * (int32_t{64}) +
                                ((int32_t{trtllm::dev::getCtaRankInPair()}) * (int32_t{32})))]};
          mRoutedIndices[int32_t{1}] = int32_t{
            params.ptrRouteMap[((mWarpGrpWarpIdx) * (int32_t{4}) + (int32_t{1})) +
                               ((mCtaIdxY) * (int32_t{64}) +
                                ((int32_t{trtllm::dev::getCtaRankInPair()}) * (int32_t{32})))]};
          mRoutedIndices[int32_t{2}] = int32_t{
            params.ptrRouteMap[((mWarpGrpWarpIdx) * (int32_t{4}) + (int32_t{2})) +
                               ((mCtaIdxY) * (int32_t{64}) +
                                ((int32_t{trtllm::dev::getCtaRankInPair()}) * (int32_t{32})))]};
          mRoutedIndices[int32_t{3}] = int32_t{
            params.ptrRouteMap[((mWarpGrpWarpIdx) * (int32_t{4}) + (int32_t{3})) +
                               ((mCtaIdxY) * (int32_t{64}) +
                                ((int32_t{trtllm::dev::getCtaRankInPair()}) * (int32_t{32})))]};
        }
      }
      {
        if ((((int32_t{4}) + (mWarpGrpWarpIdx)) * (int32_t{4})) < (int32_t{32})) {
          mRoutedIndices[int32_t{4}] = int32_t{
            params.ptrRouteMap[(((int32_t{4}) + (mWarpGrpWarpIdx)) * (int32_t{4})) +
                               ((mCtaIdxY) * (int32_t{64}) +
                                ((int32_t{trtllm::dev::getCtaRankInPair()}) * (int32_t{32})))]};
          mRoutedIndices[int32_t{5}] = int32_t{
            params.ptrRouteMap[(((int32_t{4}) + (mWarpGrpWarpIdx)) * (int32_t{4}) + (int32_t{1})) +
                               ((mCtaIdxY) * (int32_t{64}) +
                                ((int32_t{trtllm::dev::getCtaRankInPair()}) * (int32_t{32})))]};
          mRoutedIndices[int32_t{6}] = int32_t{
            params.ptrRouteMap[(((int32_t{4}) + (mWarpGrpWarpIdx)) * (int32_t{4}) + (int32_t{2})) +
                               ((mCtaIdxY) * (int32_t{64}) +
                                ((int32_t{trtllm::dev::getCtaRankInPair()}) * (int32_t{32})))]};
          mRoutedIndices[int32_t{7}] = int32_t{
            params.ptrRouteMap[(((int32_t{4}) + (mWarpGrpWarpIdx)) * (int32_t{4}) + (int32_t{3})) +
                               ((mCtaIdxY) * (int32_t{64}) +
                                ((int32_t{trtllm::dev::getCtaRankInPair()}) * (int32_t{32})))]};
        }
      }
      //
      // Hoist the first iter.
      //
      //
      // Loop body.
      //
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset503 = int32_t{0}; loopOffset503 < loopEnd;
           loopOffset503 += int32_t{128}) {
        bool const isLastLoopIter{((loopOffset503) + (int32_t{128})) >= (loopEnd)};
        //
        // gmemB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK4;
        { tileOffsetK4 = loopOffset503; }
        //
        // smemB [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          if ((loopOffset503) >= (int32_t{0})) {
            smemBProdToken = smemBDstStack.mPipeline.producer_try_acquire(smemBProdState);
          }
        }
        { smemBDstStack.mPipeline.producer_acquire(smemBProdState, smemBProdToken); }
        //
        // smemB [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK6;
        tileOffsetK6 = tileOffsetK4;
        {
          uint64_t* barrier{smemBDstStack.mPipeline.producer_get_barrier(smemBProdState)};
          int32_t index{smemBProdState.index()};
          {
            int8_t* smemBytesBasePtrB;
            int8_t* smemBytesStagePtrB;
            smemBytesBasePtrB =
              reinterpret_cast<int8_t*>(smemBDstStack.mDepSmemPtr2) + (int32_t{163840});
            smemBytesStagePtrB = smemBytesBasePtrB + ((index) * (int32_t{8192}));
            {
              int32_t coords[5];
              coords[int32_t{0}] = tileOffsetK6;
              coords[int32_t{1}] = mRoutedIndices[int32_t{0}];
              coords[int32_t{2}] = mRoutedIndices[int32_t{1}];
              coords[int32_t{3}] = mRoutedIndices[int32_t{2}];
              coords[int32_t{4}] = mRoutedIndices[int32_t{3}];
              uint64_t* leadCtaMbar;
              leadCtaMbar = cuda_ptx::mapa(cuda_ptx::space_cluster_t{},
                                           barrier,
                                           int32_t{trtllm::dev::getLeadCtaRank()});
              if ((bool{cute::elect_one_sync()}) &&
                  (((mWarpGrpWarpIdx) * (int32_t{4})) < (int32_t{32}))) {
                cuda_ptx::cp_async_bulk_tensor_tile_gather4(
                  cuda_ptx::space_cluster_t{},
                  cuda_ptx::space_global_t{},
                  cuda_ptx::cta_group_2_t{},
                  &reinterpret_cast<cutlass::bfloat16_t*>(
                    smemBytesStagePtrB)[((mWarpGrpWarpIdx) * (int32_t{4})) * (int32_t{64})],
                  params.tmaB,
                  coords,
                  leadCtaMbar,
                  mMmaCtaMask);
              }
              coords[int32_t{0}] += int32_t{64};
              if ((bool{cute::elect_one_sync()}) &&
                  (((mWarpGrpWarpIdx) * (int32_t{4})) < (int32_t{32}))) {
                cuda_ptx::cp_async_bulk_tensor_tile_gather4(
                  cuda_ptx::space_cluster_t{},
                  cuda_ptx::space_global_t{},
                  cuda_ptx::cta_group_2_t{},
                  &reinterpret_cast<cutlass::bfloat16_t*>(
                    smemBytesStagePtrB)[(((mWarpGrpWarpIdx) * (int32_t{4})) * (int32_t{64})) +
                                        (int32_t{2048})],
                  params.tmaB,
                  coords,
                  leadCtaMbar,
                  mMmaCtaMask);
              }
            }
            {
              int32_t coords[5];
              coords[int32_t{0}] = tileOffsetK6;
              coords[int32_t{1}] = mRoutedIndices[int32_t{4}];
              coords[int32_t{2}] = mRoutedIndices[int32_t{5}];
              coords[int32_t{3}] = mRoutedIndices[int32_t{6}];
              coords[int32_t{4}] = mRoutedIndices[int32_t{7}];
              uint64_t* leadCtaMbar;
              leadCtaMbar = cuda_ptx::mapa(cuda_ptx::space_cluster_t{},
                                           barrier,
                                           int32_t{trtllm::dev::getLeadCtaRank()});
              if ((bool{cute::elect_one_sync()}) &&
                  ((((int32_t{4}) + (mWarpGrpWarpIdx)) * (int32_t{4})) < (int32_t{32}))) {
                cuda_ptx::cp_async_bulk_tensor_tile_gather4(
                  cuda_ptx::space_cluster_t{},
                  cuda_ptx::space_global_t{},
                  cuda_ptx::cta_group_2_t{},
                  &reinterpret_cast<cutlass::bfloat16_t*>(
                    smemBytesStagePtrB)[(((int32_t{4}) + (mWarpGrpWarpIdx)) * (int32_t{4})) *
                                        (int32_t{64})],
                  params.tmaB,
                  coords,
                  leadCtaMbar,
                  mMmaCtaMask);
              }
              coords[int32_t{0}] += int32_t{64};
              if ((bool{cute::elect_one_sync()}) &&
                  ((((int32_t{4}) + (mWarpGrpWarpIdx)) * (int32_t{4})) < (int32_t{32}))) {
                cuda_ptx::cp_async_bulk_tensor_tile_gather4(
                  cuda_ptx::space_cluster_t{},
                  cuda_ptx::space_global_t{},
                  cuda_ptx::cta_group_2_t{},
                  &reinterpret_cast<cutlass::bfloat16_t*>(
                    smemBytesStagePtrB)[((((int32_t{4}) + (mWarpGrpWarpIdx)) * (int32_t{4})) *
                                         (int32_t{64})) +
                                        (int32_t{2048})],
                  params.tmaB,
                  coords,
                  leadCtaMbar,
                  mMmaCtaMask);
              }
            }
          }
        }
        //
        // smemB [ProdPreCommit, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset503) >= (int32_t{0})) {
        }
        //
        // smemB [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { smemBDstStack.mPipeline.producer_commit(smemBProdState); }
          ++smemBProdState;
        }
        //
        // gmemB [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        //
        // smemB [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
      }
      //
      // Tail work.
      //
      //
      // gmemB [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
      //
      // workId [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
      //
      // smemB [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
    ExitTileWithSignalingLabel:
    ExitTileWithoutSignalingLabel:
      auto newWorkTileInfoTuple{
        workIdSrcStack.mScheduler.fetch_next_work(workIdSrcStack.workTileInfo,
                                                  workIdSrcStack.mPipeline,
                                                  workIdConsState)};
      workIdSrcStack.workTileInfo = cute::get<int32_t{0}>(newWorkTileInfoTuple);
      ++workIdConsState;
      mCtaIdxX = workIdSrcStack.workTileInfo.M_idx;
      mCtaIdxY = workIdSrcStack.workTileInfo.N_idx;
      mCtaIdxZ = workIdSrcStack.workTileInfo.L_idx;
    } while (workIdSrcStack.workTileInfo.is_valid_tile);
    cudaTriggerProgrammaticLaunchCompletion();
  }
};
struct MmaTask0 {
  int32_t mCtaOffsetK;
  int32_t mCtaIdxX;
  int32_t mCtaIdxY;
  int32_t mCtaIdxZ;
  uint32_t const mTmemBaseOffset;
  inline __device__ MmaTask0(KernelParams const& params,
                             KernelState const& state,
                             int32_t warpGrpStart)
    : mCtaOffsetK{int32_t{0}}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)}
    , mTmemBaseOffset{uint32_t{
        __shfl_sync(uint32_t{0xffffffff}, (*state.mTmemSwStatePtr), int32_t{0}, int32_t{32})}} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{4})) && ((state.mWarpIdx) < (int32_t{5}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 Mma0Stack& mma0DstStack,
                                 SmemASmem& smemASrcSmem,
                                 SmemAStack& smemASrcStack,
                                 SmemBSmem& smemBSrcSmem,
                                 SmemBStack& smemBSrcStack,
                                 WorkIdSmem& workIdSrcSmem,
                                 WorkIdStack& workIdSrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<96>{});
    if (!((int32_t{cute::block_rank_in_cluster()}) == (int32_t{trtllm::dev::getLeadCtaRank()}))) {
      return;
    }
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      5,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemAConsState{};
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      5,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemAConsReleaseState{};
    int32_t smemAConsToken{int32_t{0}};
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      5,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemBConsState{};
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      5,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemBConsReleaseState{};
    int32_t smemBConsToken{int32_t{0}};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsState{};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsReleaseState{};
    int32_t workIdConsToken{int32_t{0}};
    trtllm::dev::CutlassUmmaAsyncPipeline<2,
                                          cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState mma0ProdState{int32_t{0}, int32_t{1}, int32_t{0}};
    int32_t mma0ProdStateStub{int32_t{1}};
    int32_t mma0ProdToken{int32_t{1}};
    do {
      int32_t paddedPerCtaK{(((params.k) + (int32_t{127})) / (int32_t{128})) * (int32_t{128})};
      int32_t loopEnd{paddedPerCtaK};
      if ((mCtaIdxY) >= (state.mNumNonExitingCtas)) {
        goto ExitTileWithoutSignalingLabel;
      }
      //
      // Loop body.
      //
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset641 = int32_t{0}; loopOffset641 < loopEnd;
           loopOffset641 += int32_t{256}) {
        //
        // Unrolled iter 0.
        //
        {
          bool const isFirstLoopIter{(loopOffset641) == (int32_t{0})};
          bool const isLastLoopIter{((loopOffset641) + (int32_t{128})) >= (loopEnd)};
          //
          // mma0 [ProdTryAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
          //
          if (isFirstLoopIter) {
            if ((loopOffset641) >= (int32_t{0})) {
              mma0ProdToken = mma0DstStack.mPipeline.producer_try_acquire(mma0ProdState);
            }
          }
          //
          // smemA [ConsTryWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
          //
          { smemAConsToken = smemASrcStack.mPipeline.consumer_try_wait(smemAConsState); }
          //
          // smemB [ConsTryWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
          //
          { smemBConsToken = smemBSrcStack.mPipeline.consumer_try_wait(smemBConsState); }
          //
          // mma0 [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{8388608}].
          //
          if (isFirstLoopIter) {
            mma0DstStack.mPipeline.producer_acquire(mma0ProdState, mma0ProdToken);
          }
          //
          // smemA [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{8388608}].
          //
          { smemASrcStack.mPipeline.consumer_wait(smemAConsState, smemAConsToken); }
          //
          // smemB [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{8388608}].
          //
          { smemBSrcStack.mPipeline.consumer_wait(smemBConsState, smemBConsToken); }
          //
          // smemA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
          //
          cutlass::bfloat16_t* smemPtrA5;
          {
            int32_t index{smemAConsState.index()};
            int8_t* smemBytesBasePtrA;
            smemBytesBasePtrA =
              reinterpret_cast<int8_t*>(smemASrcStack.mDepSmemPtr2) + (int32_t{0});
            int8_t* smemBytesStagePtrA;
            smemBytesStagePtrA = smemBytesBasePtrA + ((index) * (int32_t{32768}));
            smemPtrA5 = reinterpret_cast<cutlass::bfloat16_t*>(smemBytesStagePtrA) + (int32_t{0});
            ++smemAConsState;
          }
          //
          // smemB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
          //
          cutlass::bfloat16_t* smemPtrB6;
          {
            int32_t index{smemBConsState.index()};
            int8_t* smemBytesBasePtrB;
            smemBytesBasePtrB =
              reinterpret_cast<int8_t*>(smemBSrcStack.mDepSmemPtr2) + (int32_t{163840});
            int8_t* smemBytesStagePtrB;
            smemBytesStagePtrB = smemBytesBasePtrB + ((index) * (int32_t{8192}));
            smemPtrB6 = reinterpret_cast<cutlass::bfloat16_t*>(smemBytesStagePtrB) + (int32_t{0});
            ++smemBConsState;
          }
          //
          // mma0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
          //
          cutlass::bfloat16_t* smemPtrA8;
          cutlass::bfloat16_t* smemPtrB8;
          smemPtrA8 = smemPtrA5;
          smemPtrB8 = smemPtrB6;
          {
            int32_t index{mma0ProdState.index()};
            uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{64})))};
            uint32_t ptrTmemOffsetD{ptrTmemD};
            cutlass::bfloat16_t* ptrWithOffsetSmemA{(smemPtrA8 + int32_t{0})};
            cutlass::bfloat16_t* ptrWithOffsetSmemB{(smemPtrB8 + int32_t{0})};
            {
              uint32_t tmemPtrD{ptrTmemOffsetD};
              //
              // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
              //
              uint64_t smemDescA{
                trtllm::dev::createSmemDesc(ptrWithOffsetSmemA,
                                            uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                            uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
              //
              // leadingDimInBytes = 4096, strideInBytes = 1024, swizzleMode = 1.
              //
              uint64_t smemDescB{
                trtllm::dev::createSmemDesc(ptrWithOffsetSmemB,
                                            uint32_t{0x1000000 /*hi=256, lo=0*/},
                                            uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
              //
              // MMA inst for mi=0 ni=0 ki=0.
              //
              uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      false,
                                                                      int32_t{256},
                                                                      int32_t{64},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                      cuda_ptx::cta_group_2,
                                      tmemPtrD,
                                      smemDescA,
                                      smemDescB,
                                      utcmmaDesc_0_0_0,
                                      bool{(loopOffset641) != (int32_t{0})});
              }
              //
              // MMA inst for mi=0 ni=0 ki=1.
              //
              trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
              uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      false,
                                                                      int32_t{256},
                                                                      int32_t{64},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                      cuda_ptx::cta_group_2,
                                      tmemPtrD,
                                      smemDescA,
                                      smemDescB,
                                      utcmmaDesc_0_0_1,
                                      bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=2.
              //
              trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
              uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      false,
                                                                      int32_t{256},
                                                                      int32_t{64},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                      cuda_ptx::cta_group_2,
                                      tmemPtrD,
                                      smemDescA,
                                      smemDescB,
                                      utcmmaDesc_0_0_2,
                                      bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=3.
              //
              trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
              uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      false,
                                                                      int32_t{256},
                                                                      int32_t{64},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                      cuda_ptx::cta_group_2,
                                      tmemPtrD,
                                      smemDescA,
                                      smemDescB,
                                      utcmmaDesc_0_0_3,
                                      bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=4.
              //
              trtllm::dev::incrSmemAddr(smemDescA, int32_t{1018});
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{250});
              uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      false,
                                                                      int32_t{256},
                                                                      int32_t{64},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                      cuda_ptx::cta_group_2,
                                      tmemPtrD,
                                      smemDescA,
                                      smemDescB,
                                      utcmmaDesc_0_0_4,
                                      bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=5.
              //
              trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
              uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      false,
                                                                      int32_t{256},
                                                                      int32_t{64},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                      cuda_ptx::cta_group_2,
                                      tmemPtrD,
                                      smemDescA,
                                      smemDescB,
                                      utcmmaDesc_0_0_5,
                                      bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=6.
              //
              trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
              uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      false,
                                                                      int32_t{256},
                                                                      int32_t{64},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                      cuda_ptx::cta_group_2,
                                      tmemPtrD,
                                      smemDescA,
                                      smemDescB,
                                      utcmmaDesc_0_0_6,
                                      bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=7.
              //
              trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
              uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      false,
                                                                      int32_t{256},
                                                                      int32_t{64},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                      cuda_ptx::cta_group_2,
                                      tmemPtrD,
                                      smemDescA,
                                      smemDescB,
                                      utcmmaDesc_0_0_7,
                                      bool{true});
              }
            }
          }
          //
          // smemA [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
          //
          if ((loopOffset641) >= (int32_t{0})) {
            {
              smemASrcStack.mPipeline.consumer_release(smemAConsReleaseState);
            }
            ++smemAConsReleaseState;
          }
          //
          // smemB [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
          //
          if ((loopOffset641) >= (int32_t{0})) {
            {
              smemBSrcStack.mPipeline.consumer_release(smemBConsReleaseState);
            }
            ++smemBConsReleaseState;
          }
          //
          // mma0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
          //
          if (isLastLoopIter) {
            {
              mma0DstStack.mPipeline.producer_commit(mma0ProdState);
            }
            mma0ProdState += int32_t{2};
          }
        }
        //
        // Unrolled iter 1.
        //
        {
          bool const isFirstLoopIter{((int32_t{128}) + (loopOffset641)) == (int32_t{0})};
          bool const isLastLoopIter{(((int32_t{128}) + (loopOffset641)) + (int32_t{128})) >=
                                    (loopEnd)};
          //
          // mma0 [ProdTryAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
          //
          if (isFirstLoopIter) {
            if (((int32_t{128}) + (loopOffset641)) >= (int32_t{0})) {
              mma0ProdToken = mma0DstStack.mPipeline.producer_try_acquire(mma0ProdState);
            }
          }
          //
          // smemA [ConsTryWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
          //
          { smemAConsToken = smemASrcStack.mPipeline.consumer_try_wait(smemAConsState); }
          //
          // smemB [ConsTryWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
          //
          { smemBConsToken = smemBSrcStack.mPipeline.consumer_try_wait(smemBConsState); }
          //
          // mma0 [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{8388608}].
          //
          if (isFirstLoopIter) {
            mma0DstStack.mPipeline.producer_acquire(mma0ProdState, mma0ProdToken);
          }
          //
          // smemA [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{8388608}].
          //
          { smemASrcStack.mPipeline.consumer_wait(smemAConsState, smemAConsToken); }
          //
          // smemB [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{8388608}].
          //
          { smemBSrcStack.mPipeline.consumer_wait(smemBConsState, smemBConsToken); }
          //
          // smemA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
          //
          cutlass::bfloat16_t* smemPtrA5;
          {
            int32_t index{smemAConsState.index()};
            int8_t* smemBytesBasePtrA;
            smemBytesBasePtrA =
              reinterpret_cast<int8_t*>(smemASrcStack.mDepSmemPtr2) + (int32_t{0});
            int8_t* smemBytesStagePtrA;
            smemBytesStagePtrA = smemBytesBasePtrA + ((index) * (int32_t{32768}));
            smemPtrA5 = reinterpret_cast<cutlass::bfloat16_t*>(smemBytesStagePtrA) + (int32_t{0});
            ++smemAConsState;
          }
          //
          // smemB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
          //
          cutlass::bfloat16_t* smemPtrB6;
          {
            int32_t index{smemBConsState.index()};
            int8_t* smemBytesBasePtrB;
            smemBytesBasePtrB =
              reinterpret_cast<int8_t*>(smemBSrcStack.mDepSmemPtr2) + (int32_t{163840});
            int8_t* smemBytesStagePtrB;
            smemBytesStagePtrB = smemBytesBasePtrB + ((index) * (int32_t{8192}));
            smemPtrB6 = reinterpret_cast<cutlass::bfloat16_t*>(smemBytesStagePtrB) + (int32_t{0});
            ++smemBConsState;
          }
          //
          // mma0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
          //
          cutlass::bfloat16_t* smemPtrA8;
          cutlass::bfloat16_t* smemPtrB8;
          smemPtrA8 = smemPtrA5;
          smemPtrB8 = smemPtrB6;
          {
            int32_t index{mma0ProdState.index()};
            uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{64})))};
            uint32_t ptrTmemOffsetD{ptrTmemD};
            cutlass::bfloat16_t* ptrWithOffsetSmemA{(smemPtrA8 + int32_t{0})};
            cutlass::bfloat16_t* ptrWithOffsetSmemB{(smemPtrB8 + int32_t{0})};
            {
              uint32_t tmemPtrD{ptrTmemOffsetD};
              //
              // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
              //
              uint64_t smemDescA{
                trtllm::dev::createSmemDesc(ptrWithOffsetSmemA,
                                            uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                            uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
              //
              // leadingDimInBytes = 4096, strideInBytes = 1024, swizzleMode = 1.
              //
              uint64_t smemDescB{
                trtllm::dev::createSmemDesc(ptrWithOffsetSmemB,
                                            uint32_t{0x1000000 /*hi=256, lo=0*/},
                                            uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
              //
              // MMA inst for mi=0 ni=0 ki=0.
              //
              uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      false,
                                                                      int32_t{256},
                                                                      int32_t{64},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                      cuda_ptx::cta_group_2,
                                      tmemPtrD,
                                      smemDescA,
                                      smemDescB,
                                      utcmmaDesc_0_0_0,
                                      bool{((int32_t{128}) + (loopOffset641)) != (int32_t{0})});
              }
              //
              // MMA inst for mi=0 ni=0 ki=1.
              //
              trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
              uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      false,
                                                                      int32_t{256},
                                                                      int32_t{64},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                      cuda_ptx::cta_group_2,
                                      tmemPtrD,
                                      smemDescA,
                                      smemDescB,
                                      utcmmaDesc_0_0_1,
                                      bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=2.
              //
              trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
              uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      false,
                                                                      int32_t{256},
                                                                      int32_t{64},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                      cuda_ptx::cta_group_2,
                                      tmemPtrD,
                                      smemDescA,
                                      smemDescB,
                                      utcmmaDesc_0_0_2,
                                      bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=3.
              //
              trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
              uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      false,
                                                                      int32_t{256},
                                                                      int32_t{64},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                      cuda_ptx::cta_group_2,
                                      tmemPtrD,
                                      smemDescA,
                                      smemDescB,
                                      utcmmaDesc_0_0_3,
                                      bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=4.
              //
              trtllm::dev::incrSmemAddr(smemDescA, int32_t{1018});
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{250});
              uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      false,
                                                                      int32_t{256},
                                                                      int32_t{64},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                      cuda_ptx::cta_group_2,
                                      tmemPtrD,
                                      smemDescA,
                                      smemDescB,
                                      utcmmaDesc_0_0_4,
                                      bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=5.
              //
              trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
              uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      false,
                                                                      int32_t{256},
                                                                      int32_t{64},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                      cuda_ptx::cta_group_2,
                                      tmemPtrD,
                                      smemDescA,
                                      smemDescB,
                                      utcmmaDesc_0_0_5,
                                      bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=6.
              //
              trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
              uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      false,
                                                                      int32_t{256},
                                                                      int32_t{64},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                      cuda_ptx::cta_group_2,
                                      tmemPtrD,
                                      smemDescA,
                                      smemDescB,
                                      utcmmaDesc_0_0_6,
                                      bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=7.
              //
              trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
              uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      false,
                                                                      int32_t{256},
                                                                      int32_t{64},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                      cuda_ptx::cta_group_2,
                                      tmemPtrD,
                                      smemDescA,
                                      smemDescB,
                                      utcmmaDesc_0_0_7,
                                      bool{true});
              }
            }
          }
          //
          // smemA [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
          //
          if (((int32_t{128}) + (loopOffset641)) >= (int32_t{0})) {
            {
              smemASrcStack.mPipeline.consumer_release(smemAConsReleaseState);
            }
            ++smemAConsReleaseState;
          }
          //
          // smemB [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
          //
          if (((int32_t{128}) + (loopOffset641)) >= (int32_t{0})) {
            {
              smemBSrcStack.mPipeline.consumer_release(smemBConsReleaseState);
            }
            ++smemBConsReleaseState;
          }
          //
          // mma0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
          //
          if (isLastLoopIter) {
            {
              mma0DstStack.mPipeline.producer_commit(mma0ProdState);
            }
            mma0ProdState += int32_t{2};
          }
        }
      }
    //
    // Tail work.
    //
    ExitTileWithSignalingLabel:
    ExitTileWithoutSignalingLabel: {
      uint32_t stateStub_{mma0ProdState.count()};
      mma0ProdState =
        trtllm::dev::makePipelineState(trtllm::dev::makeProdStartStateFrom(mma0ProdState),
                                       mma0ProdStateStub);
      mma0ProdStateStub = stateStub_;
    }
      auto newWorkTileInfoTuple{
        workIdSrcStack.mScheduler.fetch_next_work(workIdSrcStack.workTileInfo,
                                                  workIdSrcStack.mPipeline,
                                                  workIdConsState)};
      workIdSrcStack.workTileInfo = cute::get<int32_t{0}>(newWorkTileInfoTuple);
      ++workIdConsState;
      mCtaIdxX = workIdSrcStack.workTileInfo.M_idx;
      mCtaIdxY = workIdSrcStack.workTileInfo.N_idx;
      mCtaIdxZ = workIdSrcStack.workTileInfo.L_idx;
    } while (workIdSrcStack.workTileInfo.is_valid_tile);
  }
};
struct EpilogueTask0 {
  int32_t const mWarpGrpWarpIdx;
  int32_t const mLaneIdx;
  int32_t const mWarpGrp4WarpIdx;
  int32_t const mWarpGrp4Idx;
  int32_t const mWarpRowIdx;
  int32_t const mQuadRowIdx;
  int32_t const mBaseRowIdx;
  int32_t const mLaneColIdx;
  int32_t const mBaseTmemCol;
  int32_t mCtaIdxY;
  int32_t mBatchIdx;
  int32_t mBatchLimit;
  float mClampLimit;
  float mGatedActAlpha;
  float mGatedActBeta;
  int32_t mCtaOffsetK;
  int32_t mCtaIdxX;
  int32_t mCtaIdxZ;
  uint32_t const mTmemBaseOffset;
  cutlass::Array<float, 64> frg9;
  int32_t const mGridDimX;
  inline __device__ EpilogueTask0(KernelParams const& params,
                                  KernelState const& state,
                                  int32_t warpGrpStart)
    : mWarpGrpWarpIdx{(state.mWarpIdx) - (warpGrpStart)}
    , mLaneIdx{(state.mThreadIdx) % (int32_t{32})}
    , mWarpGrp4WarpIdx{mWarpGrpWarpIdx}
    , mWarpGrp4Idx{int32_t{0}}
    , mWarpRowIdx{(mWarpGrp4WarpIdx) * (int32_t{32})}
    , mQuadRowIdx{((mLaneIdx) / (int32_t{4})) * (int32_t{2})}
    , mBaseRowIdx{(mWarpRowIdx) + (mQuadRowIdx)}
    , mLaneColIdx{((mLaneIdx) % (int32_t{4})) * (int32_t{2})}
    , mBaseTmemCol{mLaneColIdx}
    , mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mBatchIdx{int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}}
    , mBatchLimit{int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]}}
    , mClampLimit{float{3.4028235e+38}}
    , mGatedActAlpha{float{1}}
    , mGatedActBeta{float{0}}
    , mCtaOffsetK{int32_t{0}}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)}
    , mTmemBaseOffset{uint32_t{
        __shfl_sync(uint32_t{0xffffffff}, (*state.mTmemSwStatePtr), int32_t{0}, int32_t{32})}}
    , mGridDimX{reinterpret_cast<int32_t const&>(gridDim.x)} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{0})) && ((state.mWarpIdx) < (int32_t{4}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 GmemC0Smem& gmemC0DstSmem,
                                 GmemC0Stack& gmemC0DstStack,
                                 ClusterBarrierBuffersStack& clusterBarrierBuffersDstStack,
                                 Mma0Stack& mma0SrcStack,
                                 WorkIdSmem& workIdSrcSmem,
                                 WorkIdStack& workIdSrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<168>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassUmmaAsyncPipeline<
      2,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState mma0ConsState{};
    int32_t mma0ConsStateStub{int32_t{1}};
    int32_t mma0ConsToken{int32_t{0}};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsState{};
    int32_t workIdConsToken{int32_t{0}};
    do {
      int32_t paddedPerCtaK{(((params.k) + (int32_t{127})) / (int32_t{128})) * (int32_t{128})};
      int32_t loopEnd{paddedPerCtaK};
      bool const hasOneLoopIter{(int32_t{0}) < (loopEnd)};
      int32_t lastLoopOffset{int32_t{0}};
      if ((mCtaIdxY) >= (state.mNumNonExitingCtas)) {
        goto ExitTileWithoutSignalingLabel;
      }
      uint32_t tmemBaseWithStageOffset;
      tmemBaseWithStageOffset = mTmemBaseOffset;
      mBatchIdx = int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]};
      mBatchLimit = int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]};
      mClampLimit =
        float(bool{params.ptrClampLimit != nullptr})
          ? (float{params.ptrClampLimit[int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}]})
          : (float{3.4028235e+38});
      mGatedActAlpha =
        float(bool{params.ptrGatedActAlpha != nullptr})
          ? (float{params.ptrGatedActAlpha[int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}]})
          : (float{1});
      mGatedActBeta =
        float(bool{params.ptrGatedActBeta != nullptr})
          ? (float{params.ptrGatedActBeta[int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}]})
          : (float{0});
      //
      // Loop body.
      //
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset1000 = int32_t{0}; loopOffset1000 < loopEnd;
           loopOffset1000 += int32_t{128}) {
        bool const isFirstLoopIter{(loopOffset1000) == (int32_t{0})};
        //
        // gmemC0 [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        //
        // mma0 [ConsTailRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        lastLoopOffset = loopOffset1000;
      }
      //
      // Pull the last iter down.
      //
      //
      // mma0 [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if (hasOneLoopIter) {
        if (hasOneLoopIter) {
          mma0ConsToken = mma0SrcStack.mPipeline.consumer_try_wait(mma0ConsState);
        }
        mma0SrcStack.mPipeline.consumer_wait(mma0ConsState, mma0ConsToken);
      }
      //
      // mma0 [ConsWork (call 0), LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      uint32_t tmemBaseWithStageOffset8;
      if (hasOneLoopIter) {
        int32_t index{mma0ConsState.index()};
        uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{64})))};
        uint32_t ptrTmemOffsetD{ptrTmemD};
        tmemBaseWithStageOffset8 = ptrTmemOffsetD;
      }
      //
      // gmemC0 [ProdWork (call 0), LastIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      uint32_t tmemBaseWithStageOffset9;
      tmemBaseWithStageOffset9 = tmemBaseWithStageOffset8;
      if (hasOneLoopIter) {
        tmemBaseWithStageOffset = tmemBaseWithStageOffset9;
      }
      //
      // Tail work.
      //
      //
      // gmemC0 [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {
        //
        // Epilogue tile idxM=0 idxN=0.
        //
        {
          //
          // Load from Tmem to fragment.
          //
          {
            uint32_t tmemBasePtr{tmemBaseWithStageOffset};
            uint32_t(&dstSlice0)[32]{reinterpret_cast<uint32_t(&)[32]>(frg9[int32_t{0}])};
            cuda_ptx::tcgen05_ld_16x256b(dstSlice0,
                                         (tmemBasePtr) +
                                           (static_cast<uint32_t>((mWarpGrp4Idx) * (int32_t{64}))));
            uint32_t(&dstSlice1)[32]{reinterpret_cast<uint32_t(&)[32]>(frg9[int32_t{32}])};
            cuda_ptx::tcgen05_ld_16x256b(
              dstSlice1,
              (tmemBasePtr) + (static_cast<uint32_t>(((mWarpGrp4Idx) * (int32_t{64})) +
                                                     (int32_t{0x100000 /*hi=16, lo=0*/}))));
          }
          cutlass::arch::fence_view_async_tmem_load();
          cuda_ptx::cp_async_bulk_wait_group_read(cuda_ptx::n32_t<0>{});
          trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{4}) + (mWarpGrp4Idx));
          //
          // Store to Smem TmaAsyncGmemGatedAct.
          //
          int8_t* ptrSmemBase;
          cutlass::bfloat16_t* ptrSmem;
          ptrSmemBase = reinterpret_cast<int8_t*>(gmemC0DstStack.mDepSmemPtr2) +
                        ((mWarpGrp4Idx) * (int32_t{8192}) + (int32_t{204800}));
          ptrSmem = reinterpret_cast<cutlass::bfloat16_t*>(ptrSmemBase) + (int32_t{0});
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{((mBaseTmemCol) * (int32_t{64}) +
                                        (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
                                       (int32_t{64})};
              int32_t const smemOffsetInBytes{(((mBaseTmemCol) * (int32_t{64}) +
                                                (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
                                               (int32_t{16})) /
                                              (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            {
              frg9[int32_t{0}] =
                float{trtllm::dev::clamp(frg9[int32_t{0}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{2}] =
                float{trtllm::dev::clamp(frg9[int32_t{2}], float{-3.4028235e+38}, mClampLimit)};
              frg9[int32_t{32}] =
                float{trtllm::dev::clamp(frg9[int32_t{32}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{34}] =
                float{trtllm::dev::clamp(frg9[int32_t{34}], float{-3.4028235e+38}, mClampLimit)};
              cutlass::Array<float, 2> x0Array{frg9[int32_t{0}], frg9[int32_t{32}]};
              cutlass::Array<float, 2> x1Array{frg9[int32_t{2}], frg9[int32_t{34}]};
              cutlass::Array<float, 2> fusedScaleArray{(mGatedActAlpha) * (float{1.442695}),
                                                       (mGatedActAlpha) * (float{1.442695})};
              cutlass::Array<float, 2> betaScaleGateArray{mGatedActBeta, mGatedActBeta};
              cutlass::Array<float, 2> scaleGateArray{float{1}, float{1}};
              cutlass::Array<float, 2> x0ScaleGateArray;
              x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
              cutlass::Array<float, 2> x1ScaledArray;
              x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
              cutlass::Array<float, 2> actArray;
              actArray = trtllm::dev::sigmoid2_base2(x1ScaledArray);
              cutlass::Array<float, 2> swishArray;
              swishArray = trtllm::dev::fmul2(x1Array, actArray);
              cutlass::Array<float, 2> gatedActArray;
              gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, swishArray);
              frg9[int32_t{0}] = gatedActArray[int32_t{0}];
              frg9[int32_t{32}] = gatedActArray[int32_t{1}];
            }
            cutlass::Array<float, 2> accF2{frg9[int32_t{0}], frg9[int32_t{32}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{(((mBaseTmemCol) + (int32_t{1})) * (int32_t{64}) +
                                        (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
                                       (int32_t{64})};
              int32_t const smemOffsetInBytes{((((mBaseTmemCol) + (int32_t{1})) * (int32_t{64}) +
                                                (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
                                               (int32_t{16})) /
                                              (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            {
              frg9[int32_t{1}] =
                float{trtllm::dev::clamp(frg9[int32_t{1}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{3}] =
                float{trtllm::dev::clamp(frg9[int32_t{3}], float{-3.4028235e+38}, mClampLimit)};
              frg9[int32_t{33}] =
                float{trtllm::dev::clamp(frg9[int32_t{33}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{35}] =
                float{trtllm::dev::clamp(frg9[int32_t{35}], float{-3.4028235e+38}, mClampLimit)};
              cutlass::Array<float, 2> x0Array{frg9[int32_t{1}], frg9[int32_t{33}]};
              cutlass::Array<float, 2> x1Array{frg9[int32_t{3}], frg9[int32_t{35}]};
              cutlass::Array<float, 2> fusedScaleArray{(mGatedActAlpha) * (float{1.442695}),
                                                       (mGatedActAlpha) * (float{1.442695})};
              cutlass::Array<float, 2> betaScaleGateArray{mGatedActBeta, mGatedActBeta};
              cutlass::Array<float, 2> scaleGateArray{float{1}, float{1}};
              cutlass::Array<float, 2> x0ScaleGateArray;
              x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
              cutlass::Array<float, 2> x1ScaledArray;
              x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
              cutlass::Array<float, 2> actArray;
              actArray = trtllm::dev::sigmoid2_base2(x1ScaledArray);
              cutlass::Array<float, 2> swishArray;
              swishArray = trtllm::dev::fmul2(x1Array, actArray);
              cutlass::Array<float, 2> gatedActArray;
              gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, swishArray);
              frg9[int32_t{1}] = gatedActArray[int32_t{0}];
              frg9[int32_t{33}] = gatedActArray[int32_t{1}];
            }
            cutlass::Array<float, 2> accF2{frg9[int32_t{1}], frg9[int32_t{33}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{(((mBaseTmemCol) + (int32_t{8})) * (int32_t{64}) +
                                        (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
                                       (int32_t{64})};
              int32_t const smemOffsetInBytes{((((mBaseTmemCol) + (int32_t{8})) * (int32_t{64}) +
                                                (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
                                               (int32_t{16})) /
                                              (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            {
              frg9[int32_t{4}] =
                float{trtllm::dev::clamp(frg9[int32_t{4}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{6}] =
                float{trtllm::dev::clamp(frg9[int32_t{6}], float{-3.4028235e+38}, mClampLimit)};
              frg9[int32_t{36}] =
                float{trtllm::dev::clamp(frg9[int32_t{36}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{38}] =
                float{trtllm::dev::clamp(frg9[int32_t{38}], float{-3.4028235e+38}, mClampLimit)};
              cutlass::Array<float, 2> x0Array{frg9[int32_t{4}], frg9[int32_t{36}]};
              cutlass::Array<float, 2> x1Array{frg9[int32_t{6}], frg9[int32_t{38}]};
              cutlass::Array<float, 2> fusedScaleArray{(mGatedActAlpha) * (float{1.442695}),
                                                       (mGatedActAlpha) * (float{1.442695})};
              cutlass::Array<float, 2> betaScaleGateArray{mGatedActBeta, mGatedActBeta};
              cutlass::Array<float, 2> scaleGateArray{float{1}, float{1}};
              cutlass::Array<float, 2> x0ScaleGateArray;
              x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
              cutlass::Array<float, 2> x1ScaledArray;
              x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
              cutlass::Array<float, 2> actArray;
              actArray = trtllm::dev::sigmoid2_base2(x1ScaledArray);
              cutlass::Array<float, 2> swishArray;
              swishArray = trtllm::dev::fmul2(x1Array, actArray);
              cutlass::Array<float, 2> gatedActArray;
              gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, swishArray);
              frg9[int32_t{4}] = gatedActArray[int32_t{0}];
              frg9[int32_t{36}] = gatedActArray[int32_t{1}];
            }
            cutlass::Array<float, 2> accF2{frg9[int32_t{4}], frg9[int32_t{36}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{(((mBaseTmemCol) + (int32_t{9})) * (int32_t{64}) +
                                        (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
                                       (int32_t{64})};
              int32_t const smemOffsetInBytes{((((mBaseTmemCol) + (int32_t{9})) * (int32_t{64}) +
                                                (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
                                               (int32_t{16})) /
                                              (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            {
              frg9[int32_t{5}] =
                float{trtllm::dev::clamp(frg9[int32_t{5}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{7}] =
                float{trtllm::dev::clamp(frg9[int32_t{7}], float{-3.4028235e+38}, mClampLimit)};
              frg9[int32_t{37}] =
                float{trtllm::dev::clamp(frg9[int32_t{37}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{39}] =
                float{trtllm::dev::clamp(frg9[int32_t{39}], float{-3.4028235e+38}, mClampLimit)};
              cutlass::Array<float, 2> x0Array{frg9[int32_t{5}], frg9[int32_t{37}]};
              cutlass::Array<float, 2> x1Array{frg9[int32_t{7}], frg9[int32_t{39}]};
              cutlass::Array<float, 2> fusedScaleArray{(mGatedActAlpha) * (float{1.442695}),
                                                       (mGatedActAlpha) * (float{1.442695})};
              cutlass::Array<float, 2> betaScaleGateArray{mGatedActBeta, mGatedActBeta};
              cutlass::Array<float, 2> scaleGateArray{float{1}, float{1}};
              cutlass::Array<float, 2> x0ScaleGateArray;
              x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
              cutlass::Array<float, 2> x1ScaledArray;
              x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
              cutlass::Array<float, 2> actArray;
              actArray = trtllm::dev::sigmoid2_base2(x1ScaledArray);
              cutlass::Array<float, 2> swishArray;
              swishArray = trtllm::dev::fmul2(x1Array, actArray);
              cutlass::Array<float, 2> gatedActArray;
              gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, swishArray);
              frg9[int32_t{5}] = gatedActArray[int32_t{0}];
              frg9[int32_t{37}] = gatedActArray[int32_t{1}];
            }
            cutlass::Array<float, 2> accF2{frg9[int32_t{5}], frg9[int32_t{37}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{(((mBaseTmemCol) + (int32_t{16})) * (int32_t{64}) +
                                        (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
                                       (int32_t{64})};
              int32_t const smemOffsetInBytes{((((mBaseTmemCol) + (int32_t{16})) * (int32_t{64}) +
                                                (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
                                               (int32_t{16})) /
                                              (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            {
              frg9[int32_t{8}] =
                float{trtllm::dev::clamp(frg9[int32_t{8}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{10}] =
                float{trtllm::dev::clamp(frg9[int32_t{10}], float{-3.4028235e+38}, mClampLimit)};
              frg9[int32_t{40}] =
                float{trtllm::dev::clamp(frg9[int32_t{40}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{42}] =
                float{trtllm::dev::clamp(frg9[int32_t{42}], float{-3.4028235e+38}, mClampLimit)};
              cutlass::Array<float, 2> x0Array{frg9[int32_t{8}], frg9[int32_t{40}]};
              cutlass::Array<float, 2> x1Array{frg9[int32_t{10}], frg9[int32_t{42}]};
              cutlass::Array<float, 2> fusedScaleArray{(mGatedActAlpha) * (float{1.442695}),
                                                       (mGatedActAlpha) * (float{1.442695})};
              cutlass::Array<float, 2> betaScaleGateArray{mGatedActBeta, mGatedActBeta};
              cutlass::Array<float, 2> scaleGateArray{float{1}, float{1}};
              cutlass::Array<float, 2> x0ScaleGateArray;
              x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
              cutlass::Array<float, 2> x1ScaledArray;
              x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
              cutlass::Array<float, 2> actArray;
              actArray = trtllm::dev::sigmoid2_base2(x1ScaledArray);
              cutlass::Array<float, 2> swishArray;
              swishArray = trtllm::dev::fmul2(x1Array, actArray);
              cutlass::Array<float, 2> gatedActArray;
              gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, swishArray);
              frg9[int32_t{8}] = gatedActArray[int32_t{0}];
              frg9[int32_t{40}] = gatedActArray[int32_t{1}];
            }
            cutlass::Array<float, 2> accF2{frg9[int32_t{8}], frg9[int32_t{40}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{(((mBaseTmemCol) + (int32_t{17})) * (int32_t{64}) +
                                        (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
                                       (int32_t{64})};
              int32_t const smemOffsetInBytes{((((mBaseTmemCol) + (int32_t{17})) * (int32_t{64}) +
                                                (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
                                               (int32_t{16})) /
                                              (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            {
              frg9[int32_t{9}] =
                float{trtllm::dev::clamp(frg9[int32_t{9}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{11}] =
                float{trtllm::dev::clamp(frg9[int32_t{11}], float{-3.4028235e+38}, mClampLimit)};
              frg9[int32_t{41}] =
                float{trtllm::dev::clamp(frg9[int32_t{41}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{43}] =
                float{trtllm::dev::clamp(frg9[int32_t{43}], float{-3.4028235e+38}, mClampLimit)};
              cutlass::Array<float, 2> x0Array{frg9[int32_t{9}], frg9[int32_t{41}]};
              cutlass::Array<float, 2> x1Array{frg9[int32_t{11}], frg9[int32_t{43}]};
              cutlass::Array<float, 2> fusedScaleArray{(mGatedActAlpha) * (float{1.442695}),
                                                       (mGatedActAlpha) * (float{1.442695})};
              cutlass::Array<float, 2> betaScaleGateArray{mGatedActBeta, mGatedActBeta};
              cutlass::Array<float, 2> scaleGateArray{float{1}, float{1}};
              cutlass::Array<float, 2> x0ScaleGateArray;
              x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
              cutlass::Array<float, 2> x1ScaledArray;
              x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
              cutlass::Array<float, 2> actArray;
              actArray = trtllm::dev::sigmoid2_base2(x1ScaledArray);
              cutlass::Array<float, 2> swishArray;
              swishArray = trtllm::dev::fmul2(x1Array, actArray);
              cutlass::Array<float, 2> gatedActArray;
              gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, swishArray);
              frg9[int32_t{9}] = gatedActArray[int32_t{0}];
              frg9[int32_t{41}] = gatedActArray[int32_t{1}];
            }
            cutlass::Array<float, 2> accF2{frg9[int32_t{9}], frg9[int32_t{41}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{(((mBaseTmemCol) + (int32_t{24})) * (int32_t{64}) +
                                        (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
                                       (int32_t{64})};
              int32_t const smemOffsetInBytes{((((mBaseTmemCol) + (int32_t{24})) * (int32_t{64}) +
                                                (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
                                               (int32_t{16})) /
                                              (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            {
              frg9[int32_t{12}] =
                float{trtllm::dev::clamp(frg9[int32_t{12}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{14}] =
                float{trtllm::dev::clamp(frg9[int32_t{14}], float{-3.4028235e+38}, mClampLimit)};
              frg9[int32_t{44}] =
                float{trtllm::dev::clamp(frg9[int32_t{44}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{46}] =
                float{trtllm::dev::clamp(frg9[int32_t{46}], float{-3.4028235e+38}, mClampLimit)};
              cutlass::Array<float, 2> x0Array{frg9[int32_t{12}], frg9[int32_t{44}]};
              cutlass::Array<float, 2> x1Array{frg9[int32_t{14}], frg9[int32_t{46}]};
              cutlass::Array<float, 2> fusedScaleArray{(mGatedActAlpha) * (float{1.442695}),
                                                       (mGatedActAlpha) * (float{1.442695})};
              cutlass::Array<float, 2> betaScaleGateArray{mGatedActBeta, mGatedActBeta};
              cutlass::Array<float, 2> scaleGateArray{float{1}, float{1}};
              cutlass::Array<float, 2> x0ScaleGateArray;
              x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
              cutlass::Array<float, 2> x1ScaledArray;
              x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
              cutlass::Array<float, 2> actArray;
              actArray = trtllm::dev::sigmoid2_base2(x1ScaledArray);
              cutlass::Array<float, 2> swishArray;
              swishArray = trtllm::dev::fmul2(x1Array, actArray);
              cutlass::Array<float, 2> gatedActArray;
              gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, swishArray);
              frg9[int32_t{12}] = gatedActArray[int32_t{0}];
              frg9[int32_t{44}] = gatedActArray[int32_t{1}];
            }
            cutlass::Array<float, 2> accF2{frg9[int32_t{12}], frg9[int32_t{44}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{(((mBaseTmemCol) + (int32_t{25})) * (int32_t{64}) +
                                        (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
                                       (int32_t{64})};
              int32_t const smemOffsetInBytes{((((mBaseTmemCol) + (int32_t{25})) * (int32_t{64}) +
                                                (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
                                               (int32_t{16})) /
                                              (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            {
              frg9[int32_t{13}] =
                float{trtllm::dev::clamp(frg9[int32_t{13}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{15}] =
                float{trtllm::dev::clamp(frg9[int32_t{15}], float{-3.4028235e+38}, mClampLimit)};
              frg9[int32_t{45}] =
                float{trtllm::dev::clamp(frg9[int32_t{45}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{47}] =
                float{trtllm::dev::clamp(frg9[int32_t{47}], float{-3.4028235e+38}, mClampLimit)};
              cutlass::Array<float, 2> x0Array{frg9[int32_t{13}], frg9[int32_t{45}]};
              cutlass::Array<float, 2> x1Array{frg9[int32_t{15}], frg9[int32_t{47}]};
              cutlass::Array<float, 2> fusedScaleArray{(mGatedActAlpha) * (float{1.442695}),
                                                       (mGatedActAlpha) * (float{1.442695})};
              cutlass::Array<float, 2> betaScaleGateArray{mGatedActBeta, mGatedActBeta};
              cutlass::Array<float, 2> scaleGateArray{float{1}, float{1}};
              cutlass::Array<float, 2> x0ScaleGateArray;
              x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
              cutlass::Array<float, 2> x1ScaledArray;
              x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
              cutlass::Array<float, 2> actArray;
              actArray = trtllm::dev::sigmoid2_base2(x1ScaledArray);
              cutlass::Array<float, 2> swishArray;
              swishArray = trtllm::dev::fmul2(x1Array, actArray);
              cutlass::Array<float, 2> gatedActArray;
              gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, swishArray);
              frg9[int32_t{13}] = gatedActArray[int32_t{0}];
              frg9[int32_t{45}] = gatedActArray[int32_t{1}];
            }
            cutlass::Array<float, 2> accF2{frg9[int32_t{13}], frg9[int32_t{45}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{(((mBaseTmemCol) + (int32_t{32})) * (int32_t{64}) +
                                        (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
                                       (int32_t{64})};
              int32_t const smemOffsetInBytes{((((mBaseTmemCol) + (int32_t{32})) * (int32_t{64}) +
                                                (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
                                               (int32_t{16})) /
                                              (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            {
              frg9[int32_t{16}] =
                float{trtllm::dev::clamp(frg9[int32_t{16}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{18}] =
                float{trtllm::dev::clamp(frg9[int32_t{18}], float{-3.4028235e+38}, mClampLimit)};
              frg9[int32_t{48}] =
                float{trtllm::dev::clamp(frg9[int32_t{48}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{50}] =
                float{trtllm::dev::clamp(frg9[int32_t{50}], float{-3.4028235e+38}, mClampLimit)};
              cutlass::Array<float, 2> x0Array{frg9[int32_t{16}], frg9[int32_t{48}]};
              cutlass::Array<float, 2> x1Array{frg9[int32_t{18}], frg9[int32_t{50}]};
              cutlass::Array<float, 2> fusedScaleArray{(mGatedActAlpha) * (float{1.442695}),
                                                       (mGatedActAlpha) * (float{1.442695})};
              cutlass::Array<float, 2> betaScaleGateArray{mGatedActBeta, mGatedActBeta};
              cutlass::Array<float, 2> scaleGateArray{float{1}, float{1}};
              cutlass::Array<float, 2> x0ScaleGateArray;
              x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
              cutlass::Array<float, 2> x1ScaledArray;
              x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
              cutlass::Array<float, 2> actArray;
              actArray = trtllm::dev::sigmoid2_base2(x1ScaledArray);
              cutlass::Array<float, 2> swishArray;
              swishArray = trtllm::dev::fmul2(x1Array, actArray);
              cutlass::Array<float, 2> gatedActArray;
              gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, swishArray);
              frg9[int32_t{16}] = gatedActArray[int32_t{0}];
              frg9[int32_t{48}] = gatedActArray[int32_t{1}];
            }
            cutlass::Array<float, 2> accF2{frg9[int32_t{16}], frg9[int32_t{48}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{(((mBaseTmemCol) + (int32_t{33})) * (int32_t{64}) +
                                        (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
                                       (int32_t{64})};
              int32_t const smemOffsetInBytes{((((mBaseTmemCol) + (int32_t{33})) * (int32_t{64}) +
                                                (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
                                               (int32_t{16})) /
                                              (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            {
              frg9[int32_t{17}] =
                float{trtllm::dev::clamp(frg9[int32_t{17}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{19}] =
                float{trtllm::dev::clamp(frg9[int32_t{19}], float{-3.4028235e+38}, mClampLimit)};
              frg9[int32_t{49}] =
                float{trtllm::dev::clamp(frg9[int32_t{49}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{51}] =
                float{trtllm::dev::clamp(frg9[int32_t{51}], float{-3.4028235e+38}, mClampLimit)};
              cutlass::Array<float, 2> x0Array{frg9[int32_t{17}], frg9[int32_t{49}]};
              cutlass::Array<float, 2> x1Array{frg9[int32_t{19}], frg9[int32_t{51}]};
              cutlass::Array<float, 2> fusedScaleArray{(mGatedActAlpha) * (float{1.442695}),
                                                       (mGatedActAlpha) * (float{1.442695})};
              cutlass::Array<float, 2> betaScaleGateArray{mGatedActBeta, mGatedActBeta};
              cutlass::Array<float, 2> scaleGateArray{float{1}, float{1}};
              cutlass::Array<float, 2> x0ScaleGateArray;
              x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
              cutlass::Array<float, 2> x1ScaledArray;
              x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
              cutlass::Array<float, 2> actArray;
              actArray = trtllm::dev::sigmoid2_base2(x1ScaledArray);
              cutlass::Array<float, 2> swishArray;
              swishArray = trtllm::dev::fmul2(x1Array, actArray);
              cutlass::Array<float, 2> gatedActArray;
              gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, swishArray);
              frg9[int32_t{17}] = gatedActArray[int32_t{0}];
              frg9[int32_t{49}] = gatedActArray[int32_t{1}];
            }
            cutlass::Array<float, 2> accF2{frg9[int32_t{17}], frg9[int32_t{49}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{(((mBaseTmemCol) + (int32_t{40})) * (int32_t{64}) +
                                        (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
                                       (int32_t{64})};
              int32_t const smemOffsetInBytes{((((mBaseTmemCol) + (int32_t{40})) * (int32_t{64}) +
                                                (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
                                               (int32_t{16})) /
                                              (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            {
              frg9[int32_t{20}] =
                float{trtllm::dev::clamp(frg9[int32_t{20}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{22}] =
                float{trtllm::dev::clamp(frg9[int32_t{22}], float{-3.4028235e+38}, mClampLimit)};
              frg9[int32_t{52}] =
                float{trtllm::dev::clamp(frg9[int32_t{52}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{54}] =
                float{trtllm::dev::clamp(frg9[int32_t{54}], float{-3.4028235e+38}, mClampLimit)};
              cutlass::Array<float, 2> x0Array{frg9[int32_t{20}], frg9[int32_t{52}]};
              cutlass::Array<float, 2> x1Array{frg9[int32_t{22}], frg9[int32_t{54}]};
              cutlass::Array<float, 2> fusedScaleArray{(mGatedActAlpha) * (float{1.442695}),
                                                       (mGatedActAlpha) * (float{1.442695})};
              cutlass::Array<float, 2> betaScaleGateArray{mGatedActBeta, mGatedActBeta};
              cutlass::Array<float, 2> scaleGateArray{float{1}, float{1}};
              cutlass::Array<float, 2> x0ScaleGateArray;
              x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
              cutlass::Array<float, 2> x1ScaledArray;
              x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
              cutlass::Array<float, 2> actArray;
              actArray = trtllm::dev::sigmoid2_base2(x1ScaledArray);
              cutlass::Array<float, 2> swishArray;
              swishArray = trtllm::dev::fmul2(x1Array, actArray);
              cutlass::Array<float, 2> gatedActArray;
              gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, swishArray);
              frg9[int32_t{20}] = gatedActArray[int32_t{0}];
              frg9[int32_t{52}] = gatedActArray[int32_t{1}];
            }
            cutlass::Array<float, 2> accF2{frg9[int32_t{20}], frg9[int32_t{52}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{(((mBaseTmemCol) + (int32_t{41})) * (int32_t{64}) +
                                        (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
                                       (int32_t{64})};
              int32_t const smemOffsetInBytes{((((mBaseTmemCol) + (int32_t{41})) * (int32_t{64}) +
                                                (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
                                               (int32_t{16})) /
                                              (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            {
              frg9[int32_t{21}] =
                float{trtllm::dev::clamp(frg9[int32_t{21}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{23}] =
                float{trtllm::dev::clamp(frg9[int32_t{23}], float{-3.4028235e+38}, mClampLimit)};
              frg9[int32_t{53}] =
                float{trtllm::dev::clamp(frg9[int32_t{53}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{55}] =
                float{trtllm::dev::clamp(frg9[int32_t{55}], float{-3.4028235e+38}, mClampLimit)};
              cutlass::Array<float, 2> x0Array{frg9[int32_t{21}], frg9[int32_t{53}]};
              cutlass::Array<float, 2> x1Array{frg9[int32_t{23}], frg9[int32_t{55}]};
              cutlass::Array<float, 2> fusedScaleArray{(mGatedActAlpha) * (float{1.442695}),
                                                       (mGatedActAlpha) * (float{1.442695})};
              cutlass::Array<float, 2> betaScaleGateArray{mGatedActBeta, mGatedActBeta};
              cutlass::Array<float, 2> scaleGateArray{float{1}, float{1}};
              cutlass::Array<float, 2> x0ScaleGateArray;
              x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
              cutlass::Array<float, 2> x1ScaledArray;
              x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
              cutlass::Array<float, 2> actArray;
              actArray = trtllm::dev::sigmoid2_base2(x1ScaledArray);
              cutlass::Array<float, 2> swishArray;
              swishArray = trtllm::dev::fmul2(x1Array, actArray);
              cutlass::Array<float, 2> gatedActArray;
              gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, swishArray);
              frg9[int32_t{21}] = gatedActArray[int32_t{0}];
              frg9[int32_t{53}] = gatedActArray[int32_t{1}];
            }
            cutlass::Array<float, 2> accF2{frg9[int32_t{21}], frg9[int32_t{53}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{(((mBaseTmemCol) + (int32_t{48})) * (int32_t{64}) +
                                        (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
                                       (int32_t{64})};
              int32_t const smemOffsetInBytes{((((mBaseTmemCol) + (int32_t{48})) * (int32_t{64}) +
                                                (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
                                               (int32_t{16})) /
                                              (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            {
              frg9[int32_t{24}] =
                float{trtllm::dev::clamp(frg9[int32_t{24}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{26}] =
                float{trtllm::dev::clamp(frg9[int32_t{26}], float{-3.4028235e+38}, mClampLimit)};
              frg9[int32_t{56}] =
                float{trtllm::dev::clamp(frg9[int32_t{56}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{58}] =
                float{trtllm::dev::clamp(frg9[int32_t{58}], float{-3.4028235e+38}, mClampLimit)};
              cutlass::Array<float, 2> x0Array{frg9[int32_t{24}], frg9[int32_t{56}]};
              cutlass::Array<float, 2> x1Array{frg9[int32_t{26}], frg9[int32_t{58}]};
              cutlass::Array<float, 2> fusedScaleArray{(mGatedActAlpha) * (float{1.442695}),
                                                       (mGatedActAlpha) * (float{1.442695})};
              cutlass::Array<float, 2> betaScaleGateArray{mGatedActBeta, mGatedActBeta};
              cutlass::Array<float, 2> scaleGateArray{float{1}, float{1}};
              cutlass::Array<float, 2> x0ScaleGateArray;
              x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
              cutlass::Array<float, 2> x1ScaledArray;
              x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
              cutlass::Array<float, 2> actArray;
              actArray = trtllm::dev::sigmoid2_base2(x1ScaledArray);
              cutlass::Array<float, 2> swishArray;
              swishArray = trtllm::dev::fmul2(x1Array, actArray);
              cutlass::Array<float, 2> gatedActArray;
              gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, swishArray);
              frg9[int32_t{24}] = gatedActArray[int32_t{0}];
              frg9[int32_t{56}] = gatedActArray[int32_t{1}];
            }
            cutlass::Array<float, 2> accF2{frg9[int32_t{24}], frg9[int32_t{56}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{(((mBaseTmemCol) + (int32_t{49})) * (int32_t{64}) +
                                        (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
                                       (int32_t{64})};
              int32_t const smemOffsetInBytes{((((mBaseTmemCol) + (int32_t{49})) * (int32_t{64}) +
                                                (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
                                               (int32_t{16})) /
                                              (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            {
              frg9[int32_t{25}] =
                float{trtllm::dev::clamp(frg9[int32_t{25}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{27}] =
                float{trtllm::dev::clamp(frg9[int32_t{27}], float{-3.4028235e+38}, mClampLimit)};
              frg9[int32_t{57}] =
                float{trtllm::dev::clamp(frg9[int32_t{57}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{59}] =
                float{trtllm::dev::clamp(frg9[int32_t{59}], float{-3.4028235e+38}, mClampLimit)};
              cutlass::Array<float, 2> x0Array{frg9[int32_t{25}], frg9[int32_t{57}]};
              cutlass::Array<float, 2> x1Array{frg9[int32_t{27}], frg9[int32_t{59}]};
              cutlass::Array<float, 2> fusedScaleArray{(mGatedActAlpha) * (float{1.442695}),
                                                       (mGatedActAlpha) * (float{1.442695})};
              cutlass::Array<float, 2> betaScaleGateArray{mGatedActBeta, mGatedActBeta};
              cutlass::Array<float, 2> scaleGateArray{float{1}, float{1}};
              cutlass::Array<float, 2> x0ScaleGateArray;
              x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
              cutlass::Array<float, 2> x1ScaledArray;
              x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
              cutlass::Array<float, 2> actArray;
              actArray = trtllm::dev::sigmoid2_base2(x1ScaledArray);
              cutlass::Array<float, 2> swishArray;
              swishArray = trtllm::dev::fmul2(x1Array, actArray);
              cutlass::Array<float, 2> gatedActArray;
              gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, swishArray);
              frg9[int32_t{25}] = gatedActArray[int32_t{0}];
              frg9[int32_t{57}] = gatedActArray[int32_t{1}];
            }
            cutlass::Array<float, 2> accF2{frg9[int32_t{25}], frg9[int32_t{57}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{(((mBaseTmemCol) + (int32_t{56})) * (int32_t{64}) +
                                        (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
                                       (int32_t{64})};
              int32_t const smemOffsetInBytes{((((mBaseTmemCol) + (int32_t{56})) * (int32_t{64}) +
                                                (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
                                               (int32_t{16})) /
                                              (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            {
              frg9[int32_t{28}] =
                float{trtllm::dev::clamp(frg9[int32_t{28}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{30}] =
                float{trtllm::dev::clamp(frg9[int32_t{30}], float{-3.4028235e+38}, mClampLimit)};
              frg9[int32_t{60}] =
                float{trtllm::dev::clamp(frg9[int32_t{60}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{62}] =
                float{trtllm::dev::clamp(frg9[int32_t{62}], float{-3.4028235e+38}, mClampLimit)};
              cutlass::Array<float, 2> x0Array{frg9[int32_t{28}], frg9[int32_t{60}]};
              cutlass::Array<float, 2> x1Array{frg9[int32_t{30}], frg9[int32_t{62}]};
              cutlass::Array<float, 2> fusedScaleArray{(mGatedActAlpha) * (float{1.442695}),
                                                       (mGatedActAlpha) * (float{1.442695})};
              cutlass::Array<float, 2> betaScaleGateArray{mGatedActBeta, mGatedActBeta};
              cutlass::Array<float, 2> scaleGateArray{float{1}, float{1}};
              cutlass::Array<float, 2> x0ScaleGateArray;
              x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
              cutlass::Array<float, 2> x1ScaledArray;
              x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
              cutlass::Array<float, 2> actArray;
              actArray = trtllm::dev::sigmoid2_base2(x1ScaledArray);
              cutlass::Array<float, 2> swishArray;
              swishArray = trtllm::dev::fmul2(x1Array, actArray);
              cutlass::Array<float, 2> gatedActArray;
              gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, swishArray);
              frg9[int32_t{28}] = gatedActArray[int32_t{0}];
              frg9[int32_t{60}] = gatedActArray[int32_t{1}];
            }
            cutlass::Array<float, 2> accF2{frg9[int32_t{28}], frg9[int32_t{60}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{(((mBaseTmemCol) + (int32_t{57})) * (int32_t{64}) +
                                        (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
                                       (int32_t{64})};
              int32_t const smemOffsetInBytes{((((mBaseTmemCol) + (int32_t{57})) * (int32_t{64}) +
                                                (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
                                               (int32_t{16})) /
                                              (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            {
              frg9[int32_t{29}] =
                float{trtllm::dev::clamp(frg9[int32_t{29}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{31}] =
                float{trtllm::dev::clamp(frg9[int32_t{31}], float{-3.4028235e+38}, mClampLimit)};
              frg9[int32_t{61}] =
                float{trtllm::dev::clamp(frg9[int32_t{61}], -(mClampLimit), mClampLimit)};
              frg9[int32_t{63}] =
                float{trtllm::dev::clamp(frg9[int32_t{63}], float{-3.4028235e+38}, mClampLimit)};
              cutlass::Array<float, 2> x0Array{frg9[int32_t{29}], frg9[int32_t{61}]};
              cutlass::Array<float, 2> x1Array{frg9[int32_t{31}], frg9[int32_t{63}]};
              cutlass::Array<float, 2> fusedScaleArray{(mGatedActAlpha) * (float{1.442695}),
                                                       (mGatedActAlpha) * (float{1.442695})};
              cutlass::Array<float, 2> betaScaleGateArray{mGatedActBeta, mGatedActBeta};
              cutlass::Array<float, 2> scaleGateArray{float{1}, float{1}};
              cutlass::Array<float, 2> x0ScaleGateArray;
              x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
              cutlass::Array<float, 2> x1ScaledArray;
              x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
              cutlass::Array<float, 2> actArray;
              actArray = trtllm::dev::sigmoid2_base2(x1ScaledArray);
              cutlass::Array<float, 2> swishArray;
              swishArray = trtllm::dev::fmul2(x1Array, actArray);
              cutlass::Array<float, 2> gatedActArray;
              gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, swishArray);
              frg9[int32_t{29}] = gatedActArray[int32_t{0}];
              frg9[int32_t{61}] = gatedActArray[int32_t{1}];
            }
            cutlass::Array<float, 2> accF2{frg9[int32_t{29}], frg9[int32_t{61}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          cuda_ptx::fence_proxy_async(cuda_ptx::space_cluster_t{});
          cutlass::arch::fence_view_async_tmem_load();
          trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{4}) + (mWarpGrp4Idx));
          //
          // Issue TMA from smem to gmem.
          //
          if ((bool{cute::elect_one_sync()}) && ((mWarpGrp4WarpIdx) == (int32_t{0}))) {
            int8_t* ptrSmemBase;
            cutlass::bfloat16_t* ptrSmem;
            ptrSmemBase = reinterpret_cast<int8_t*>(gmemC0DstStack.mDepSmemPtr2) +
                          ((mWarpGrp4Idx) * (int32_t{8192}) + (int32_t{204800}));
            ptrSmem = reinterpret_cast<cutlass::bfloat16_t*>(ptrSmemBase) + (int32_t{0});
            int32_t coords[4];
            coords[int32_t{0}] = (mCtaIdxX) * (int32_t{64});
            coords[int32_t{1}] =
              (((int32_t{64}) - ((mBatchLimit) % (int32_t{64}))) % (int32_t{64})) +
              ((mWarpGrp4Idx) * (int32_t{64}));
            coords[int32_t{2}] = int32_t{0x40000000 /*1073741824*/};
            coords[int32_t{3}] =
              (((mCtaIdxY) * (int32_t{64})) +
               ((int32_t{0}) -
                (((int32_t{64}) - ((mBatchLimit) % (int32_t{64}))) % (int32_t{64})))) +
              (int32_t{0x40000000 /*1073741824*/});
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_global_t{},
                                           cuda_ptx::space_shared_t{},
                                           params.tmaC,
                                           coords,
                                           &ptrSmem[int32_t{0}]);
          }
          cuda_ptx::cp_async_bulk_commit_group();
          trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{4}) + (mWarpGrp4Idx));
        }
        {
          //
          // Skip all-reduce if on single device.
          //
        }
      }
    ExitTileWithSignalingLabel:
      //
      // mma0 [ConsTailRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if (hasOneLoopIter) {
        {
          mma0SrcStack.mPipeline.consumer_release(mma0ConsState);
        }
        mma0ConsState += int32_t{2};
      }
    ExitTileWithoutSignalingLabel: {
      uint32_t stateStub_{mma0ConsState.count()};
      mma0ConsState =
        trtllm::dev::makePipelineState(trtllm::dev::makeConsStartStateFrom(mma0ConsState),
                                       mma0ConsStateStub);
      mma0ConsStateStub = stateStub_;
    }
      auto newWorkTileInfoTuple{
        workIdSrcStack.mScheduler.fetch_next_work(workIdSrcStack.workTileInfo,
                                                  workIdSrcStack.mPipeline,
                                                  workIdConsState)};
      workIdSrcStack.workTileInfo = cute::get<int32_t{0}>(newWorkTileInfoTuple);
      ++workIdConsState;
      mCtaIdxX = workIdSrcStack.workTileInfo.M_idx;
      mCtaIdxY = workIdSrcStack.workTileInfo.N_idx;
      mCtaIdxZ = workIdSrcStack.workTileInfo.L_idx;
    } while (workIdSrcStack.workTileInfo.is_valid_tile);
  }
};
struct PaddingTask {
  int32_t mCtaIdxX;
  int32_t mCtaIdxY;
  int32_t mCtaIdxZ;
  inline __device__ PaddingTask(KernelParams const& params,
                                KernelState const& state,
                                int32_t warpGrpStart)
    : mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{11})) && ((state.mWarpIdx) < (int32_t{12}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 WorkIdSmem& workIdSrcSmem,
                                 WorkIdStack& workIdSrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<96>{});
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsState{};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsReleaseState{};
    int32_t workIdConsToken{int32_t{0}};
    do {
      if ((mCtaIdxY) >= (state.mNumNonExitingCtas)) {
        goto ExitTileWithoutSignalingLabel;
      }
      //
      // Tail work.
      //
      //
      // workId [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
    ExitTileWithSignalingLabel:
    ExitTileWithoutSignalingLabel:
      auto newWorkTileInfoTuple{
        workIdSrcStack.mScheduler.fetch_next_work(workIdSrcStack.workTileInfo,
                                                  workIdSrcStack.mPipeline,
                                                  workIdConsState)};
      workIdSrcStack.workTileInfo = cute::get<int32_t{0}>(newWorkTileInfoTuple);
      ++workIdConsState;
      mCtaIdxX = workIdSrcStack.workTileInfo.M_idx;
      mCtaIdxY = workIdSrcStack.workTileInfo.N_idx;
      mCtaIdxZ = workIdSrcStack.workTileInfo.L_idx;
    } while (workIdSrcStack.workTileInfo.is_valid_tile);
  }
};
struct WorkIdTask {
  dim3 const mBlockInClusterIdx;
  int32_t mCtaIdxX;
  int32_t mCtaIdxY;
  int32_t mCtaIdxZ;
  inline __device__ WorkIdTask(KernelParams const& params,
                               KernelState const& state,
                               int32_t warpGrpStart)
    : mBlockInClusterIdx{cute::block_id_in_cluster()}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{10})) && ((state.mWarpIdx) < (int32_t{11}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 WorkIdSmem& workIdDstSmem,
                                 WorkIdStack& workIdDstStack,
                                 WorkIdSmem& workIdSrcSmem,
                                 WorkIdStack& workIdSrcStack,
                                 WorkThrottleBarrierStack& workThrottleBarrierSrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<96>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsState{};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsReleaseState{};
    int32_t workIdConsToken{int32_t{0}};
    trtllm::dev::CutlassCpAsyncPipeline<3>::PipelineState workThrottleBarrierConsState{};
    trtllm::dev::CutlassCpAsyncPipeline<3>::PipelineState workThrottleBarrierConsReleaseState{};
    int32_t workThrottleBarrierConsToken{int32_t{0}};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdProdState{int32_t{0}, int32_t{1}, int32_t{0}};
    int32_t workIdProdToken{int32_t{1}};
    bool isProducer;
    isProducer = (mBlockInClusterIdx.x) == (int32_t{0});
    if (isProducer) {
      do {
        if ((mCtaIdxY) >= (state.mNumNonExitingCtas)) {
          goto ExitTileWithoutSignalingLabel;
        }
        workThrottleBarrierSrcStack.mPipeline.consumer_wait(workThrottleBarrierConsState,
                                                            workThrottleBarrierConsToken);
        workThrottleBarrierSrcStack.mPipeline.consumer_release(workThrottleBarrierConsState);
        ++workThrottleBarrierConsState;
      ExitTileWithSignalingLabel:
      ExitTileWithoutSignalingLabel:
        if ((mCtaIdxY) >= (state.mNumNonExitingCtas)) {
          workIdSrcStack.mPipeline.fast_drain_all(workIdSrcSmem.fastDrainStorage);
        }
        workIdProdState =
          workIdSrcStack.mScheduler.advance_to_next_work(workIdSrcStack.mPipeline.get_pipeline(),
                                                         workIdProdState);
        auto newWorkTileInfoTuple{
          workIdSrcStack.mScheduler.fetch_next_work(workIdSrcStack.workTileInfo,
                                                    workIdSrcStack.mPipeline,
                                                    workIdConsState)};
        workIdSrcStack.workTileInfo = cute::get<int32_t{0}>(newWorkTileInfoTuple);
        ++workIdConsState;
        mCtaIdxX = workIdSrcStack.workTileInfo.M_idx;
        mCtaIdxY = workIdSrcStack.workTileInfo.N_idx;
        mCtaIdxZ = workIdSrcStack.workTileInfo.L_idx;
      } while (workIdSrcStack.workTileInfo.is_valid_tile);
      workIdDstStack.mPipeline.producer_tail(workIdProdState);
    }
  }
};
extern "C" __global__ __launch_bounds__(384, 1)
  __cluster_dims__(2, 1, 1) void bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x64x128u2_s5_et128x64_m256x64x16_cga2x1x1_16dp256b_rM_BN_transOut_schPd2x1x2x3_bN_tma_tmaSf_rgTma_clmp_swiGlu_dynB_sm100f(
    CUTE_GRID_CONSTANT KernelParams const params) {
  extern __shared__ uint8_t smem__[];
  int32_t smemOffset__{int32_t{0}};
  smemOffset__ = (((smemOffset__) + (int32_t{1023})) / (int32_t{1024})) * (int32_t{1024});
  uint8_t* smemBufferSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemBufferSmem)});
  smemOffset__ = (((smemOffset__) + (int32_t{1023})) / (int32_t{1024})) * (int32_t{1024});
  uint8_t* smemASmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemASmem)});
  smemOffset__ = (((smemOffset__) + (int32_t{1023})) / (int32_t{1024})) * (int32_t{1024});
  uint8_t* smemBSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemBSmem)});
  smemOffset__ = (((smemOffset__) + (int32_t{1023})) / (int32_t{1024})) * (int32_t{1024});
  uint8_t* gmemC0SmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(GmemC0Smem)});
  smemOffset__ = (((smemOffset__) + (int32_t{15})) / (int32_t{16})) * (int32_t{16});
  uint8_t* workIdSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(WorkIdSmem)});
  uint32_t* TmemSwStatePtr{
    reinterpret_cast<uint32_t*>((reinterpret_cast<uint8_t*>(smem__) + smemOffset__))};
  smemOffset__ += int32_t{16};
  uint8_t* workIdSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(WorkIdSmemBarrier)});
  uint8_t* workThrottleBarrierSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(WorkThrottleBarrierSmemBarrier)});
  uint8_t* smemASmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemASmemBarrier)});
  uint8_t* smemBSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemBSmemBarrier)});
  uint8_t* mma0SmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(Mma0SmemBarrier)});
  uint8_t* clusterBarrierBuffersSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(ClusterBarrierBuffersSmemBarrier)});
  cudaGridDependencySynchronize();
  KernelState const state{params, TmemSwStatePtr};
  WorkIdSmem* workIdSmem{reinterpret_cast<WorkIdSmem*>(workIdSmemPtr)};
  WorkIdSmemBarrier* workIdSmemBarrier{reinterpret_cast<WorkIdSmemBarrier*>(workIdSmemBarrierPtr)};
  WorkIdStack workIdStack{(*workIdSmem),
                          (*workIdSmemBarrier),
                          params,
                          state.mWarpIdx,
                          int32_t{10},
                          int32_t{-1}};
  WorkThrottleBarrierSmemBarrier* workThrottleBarrierSmemBarrier{
    reinterpret_cast<WorkThrottleBarrierSmemBarrier*>(workThrottleBarrierSmemBarrierPtr)};
  WorkThrottleBarrierStack workThrottleBarrierStack{(*workThrottleBarrierSmemBarrier),
                                                    state.mWarpIdx,
                                                    int32_t{10},
                                                    int32_t{-1}};
  SmemBufferSmem* smemBufferSmem{reinterpret_cast<SmemBufferSmem*>(smemBufferSmemPtr)};
  SmemBufferStack smemBufferStack{(*smemBufferSmem), state.mWarpIdx, int32_t{0}, int32_t{-1}};
  SmemASmem* smemASmem{reinterpret_cast<SmemASmem*>(smemASmemPtr)};
  SmemASmemBarrier* smemASmemBarrier{reinterpret_cast<SmemASmemBarrier*>(smemASmemBarrierPtr)};
  SmemAStack smemAStack{(*smemASmem),
                        (*smemASmemBarrier),
                        (*smemBufferSmem),
                        smemBufferStack,
                        state.mWarpIdx,
                        int32_t{5},
                        int32_t{-1}};
  SmemBSmem* smemBSmem{reinterpret_cast<SmemBSmem*>(smemBSmemPtr)};
  SmemBSmemBarrier* smemBSmemBarrier{reinterpret_cast<SmemBSmemBarrier*>(smemBSmemBarrierPtr)};
  SmemBStack smemBStack{(*smemBSmem),
                        (*smemBSmemBarrier),
                        (*smemBufferSmem),
                        smemBufferStack,
                        state.mWarpIdx,
                        int32_t{6},
                        int32_t{-1}};
  TmemStack tmemStack{state.mWarpIdx, int32_t{0}, int32_t{-1}};
  Mma0SmemBarrier* mma0SmemBarrier{reinterpret_cast<Mma0SmemBarrier*>(mma0SmemBarrierPtr)};
  Mma0Stack mma0Stack{(*mma0SmemBarrier), tmemStack, state.mWarpIdx, int32_t{1}, int32_t{-1}};
  GmemC0Smem* gmemC0Smem{reinterpret_cast<GmemC0Smem*>(gmemC0SmemPtr)};
  GmemC0Stack gmemC0Stack{(*gmemC0Smem),
                          (*smemBufferSmem),
                          smemBufferStack,
                          state.mWarpIdx,
                          int32_t{0},
                          int32_t{-1}};
  ClusterBarrierBuffersSmemBarrier* clusterBarrierBuffersSmemBarrier{
    reinterpret_cast<ClusterBarrierBuffersSmemBarrier*>(clusterBarrierBuffersSmemBarrierPtr)};
  ClusterBarrierBuffersStack clusterBarrierBuffersStack{(*clusterBarrierBuffersSmemBarrier),
                                                        state.mWarpIdx,
                                                        int32_t{0},
                                                        int32_t{-1}};
  LoadTaskA loadTaskA{params, state, int32_t{5}};
  LoadTaskB loadTaskB{params, state, int32_t{6}};
  cutlass::arch::fence_barrier_init();
  cuda_ptx::fence_mbarrier_init(cuda_ptx::sem_release_t{}, cuda_ptx::scope_cluster_t{});
  cuda_ptx::barrier_cluster_arrive(cuda_ptx::sem_relaxed_t{});
  cuda_ptx::barrier_cluster_wait();
  if ((reinterpret_cast<int32_t const&>(threadIdx.x)) < (int32_t{32})) {
    cuda_ptx::tcgen05_alloc(cuda_ptx::cta_group_2_t{}, state.mTmemSwStatePtr, int32_t{128});
    cuda_ptx::tcgen05_relinquish_alloc_permit(cuda_ptx::cta_group_2_t{});
  }
  if ((bool{LoadTaskA::isSelected(params, state)}) ||
      (bool{LoadTaskB::isSelected(params, state)})) {
  } else {
    trtllm::dev::CutlassNamedBarrier::sync(224, 6);
  }
  if (bool{LoadTaskA::isSelected(params, state)}) {
    loadTaskA.execute(params,
                      state,
                      (*smemASmem),
                      smemAStack,
                      workThrottleBarrierStack,
                      (*workIdSmem),
                      workIdStack);
  } else {
    if (bool{LoadTaskB::isSelected(params, state)}) {
      loadTaskB.execute(params, state, (*smemBSmem), smemBStack, (*workIdSmem), workIdStack);
    } else {
      if (bool{MmaTask0::isSelected(params, state)}) {
        MmaTask0 mmaTask0{params, state, int32_t{4}};
        mmaTask0.execute(params,
                         state,
                         mma0Stack,
                         (*smemASmem),
                         smemAStack,
                         (*smemBSmem),
                         smemBStack,
                         (*workIdSmem),
                         workIdStack);
      } else {
        if (bool{EpilogueTask0::isSelected(params, state)}) {
          EpilogueTask0 epilogueTask0{params, state, int32_t{0}};
          epilogueTask0.execute(params,
                                state,
                                (*gmemC0Smem),
                                gmemC0Stack,
                                clusterBarrierBuffersStack,
                                mma0Stack,
                                (*workIdSmem),
                                workIdStack);
          trtllm::dev::CutlassNamedBarrier::sync(128, 7);
          int32_t const warpGrpThreadIdx{state.mThreadIdx};
          if ((warpGrpThreadIdx) < (int32_t{32})) {
            clusterBarrierBuffersStack.mClusterBarrier.sync();
            cuda_ptx::tcgen05_dealloc(cuda_ptx::cta_group_2_t{},
                                      uint32_t{__shfl_sync(uint32_t{0xffffffff},
                                                           (*state.mTmemSwStatePtr),
                                                           int32_t{0},
                                                           int32_t{32})},
                                      int32_t{128});
          }
        } else {
          if (bool{PaddingTask::isSelected(params, state)}) {
            PaddingTask paddingTask{params, state, int32_t{11}};
            paddingTask.execute(params, state, (*workIdSmem), workIdStack);
          } else {
            if (bool{WorkIdTask::isSelected(params, state)}) {
              WorkIdTask workIdTask{params, state, int32_t{10}};
              workIdTask.execute(params,
                                 state,
                                 (*workIdSmem),
                                 workIdStack,
                                 (*workIdSmem),
                                 workIdStack,
                                 workThrottleBarrierStack);
            }
          }
        }
      }
    }
  }
}
extern "C" __global__ void
bmm_Bfloat16_Bfloat16Bfloat16_Fp32_t128x64x128u2_s5_et128x64_m256x64x16_cga2x1x1_16dp256b_rM_BN_transOut_schPd2x1x2x3_bN_tma_tmaSf_rgTma_clmp_swiGlu_dynB_sm100fGetSmemSize(
  int32_t* outPtr) {
  int32_t size{0};
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemBufferSmem));
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemASmem));
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemBSmem));
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(GmemC0Smem));
  size = (size + 15) / 16 * 16;
  size += static_cast<int32_t>(sizeof(WorkIdSmem));
  size += 16;
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(WorkIdSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(WorkThrottleBarrierSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(SmemASmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(SmemBSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(Mma0SmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(ClusterBarrierBuffersSmemBarrier));
  outPtr[0] = size;
}

} // namespace batchedGemm
