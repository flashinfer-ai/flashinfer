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
#include <Bmm_Bfloat16_MxE2m1MxE4m3_Fp32_bA32_bB32_t128x128x256_s4_et128x64_m256x128x32_cga2x1x1_16dp256b_rM_TN_transOut_schPd2x1x2x3_biasM_bN_rgTma_clmp_dynB_sm100f.h>
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
    : mPipeline{workIdSmemBarrier.mBarriers, int32_t{1}, int32_t{672}, int32_t{0}, barInitWarpId}
    , mScheduler{&workIdSmem.workIdResponse[int32_t{0}],
                 typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<
                   cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
                   3>::Params{},
                 cute::block_id_in_cluster()}
    , workTileInfo{mScheduler.initial_work_tile_info(CuteFlatTuple390{})} {}
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
    4,
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
                int32_t{32768},
                ((warpId) == (barInitWarpId)) && (bool{cute::elect_one_sync()}),
                int32_t{1},
                CuteFlatTuple629{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId} {}
};
struct SmemBStack {
  int8_t* mDepSmemPtr2;
  trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
    4,
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
                int32_t{32768},
                ((warpId) == (barInitWarpId)) && (bool{cute::elect_one_sync()}),
                int32_t{1},
                CuteFlatTuple755{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId} {}
};
struct SmemSfAStack {
  trtllm::dev::CutlassTmaUmmaAsyncPipeline<4,
                                           cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
                                           cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  cutlass::float_ue8m0_t* mPtr;
  inline __device__ SmemSfAStack(SmemSfASmem& smemSfASmem,
                                 SmemSfASmemBarrier& smemSfASmemBarrier,
                                 int32_t warpId,
                                 int32_t barInitWarpId,
                                 int32_t orderedSequenceGroupId)
    : mPipeline{smemSfASmemBarrier.mBarriers,
                warpId,
                int32_t{2048},
                bool{cute::elect_one_sync()},
                CuteFlatTuple876{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId}
    , mPtr{&smemSfASmem.mArray[int32_t{0}][int32_t{0}]} {}
};
struct SmemSfBStack {
  trtllm::dev::CutlassTmaUmmaAsyncPipeline<4,
                                           cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
                                           cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  cutlass::float_ue8m0_t* mPtr;
  inline __device__ SmemSfBStack(SmemSfBSmem& smemSfBSmem,
                                 SmemSfBSmemBarrier& smemSfBSmemBarrier,
                                 int32_t warpId,
                                 int32_t barInitWarpId,
                                 int32_t orderedSequenceGroupId)
    : mPipeline{smemSfBSmemBarrier.mBarriers,
                warpId,
                int32_t{2048},
                bool{cute::elect_one_sync()},
                CuteFlatTuple999{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId}
    , mPtr{&smemSfBSmem.mArray[int32_t{0}][int32_t{0}]} {}
};
struct TmemSfAbStack {
  trtllm::dev::CutlassUmmaConsumerAsyncPipeline<
    4,
    false,
    false,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  inline __device__ TmemSfAbStack(TmemSfAbSmemBarrier& tmemSfAbSmemBarrier,
                                  int32_t warpId,
                                  int32_t barInitWarpId,
                                  int32_t orderedSequenceGroupId)
    : mPipeline{tmemSfAbSmemBarrier.mBarriers,
                warpId,
                int32_t{32},
                CuteFlatTuple1119{},
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
                CuteFlatTuple1236{},
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
    return ((state.mWarpIdx) >= (int32_t{6})) && ((state.mWarpIdx) < (int32_t{7}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 SmemASmem& smemADstSmem,
                                 SmemAStack& smemADstStack,
                                 WorkThrottleBarrierStack& workThrottleBarrierDstStack,
                                 WorkIdSmem& workIdSrcSmem,
                                 WorkIdStack& workIdSrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<96>{});
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsState{};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsReleaseState{};
    int32_t workIdConsToken{int32_t{0}};
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      4,
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
      int32_t paddedPerCtaK{(((params.k) + (int32_t{255})) / (int32_t{256})) * (int32_t{256})};
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
      for (int32_t loopOffset526 = int32_t{0}; loopOffset526 < loopEnd;
           loopOffset526 += int32_t{256}) {
        bool const isLastLoopIter{((loopOffset526) + (int32_t{256})) >= (loopEnd)};
        //
        // gmemA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK3;
        { tileOffsetK3 = loopOffset526; }
        //
        // smemA [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          smemADstStack.mPipeline.producer_acquire(smemAProdState, smemAProdToken);
          if (((loopOffset526) + (int32_t{256})) < (loopEnd)) {
            smemAProdToken = smemADstStack.mPipeline.producer_try_acquire(
              trtllm::dev::makePipelineState(smemAProdState, int32_t{1}));
          }
        }
        //
        // smemA [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK7;
        tileOffsetK7 = tileOffsetK3;
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
            int32_t coords[3];
            coords[int32_t{0}] = tileOffsetK7;
            coords[int32_t{1}] = (mCtaIdxX) * (int32_t{128});
            coords[int32_t{2}] = mBatchIdx;
            uint64_t* leadCtaMbar;
            leadCtaMbar = cuda_ptx::mapa(cuda_ptx::space_cluster_t{},
                                         barrier,
                                         int32_t{trtllm::dev::getLeadCtaRank()});
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::cp_async_bulk_tensor(
                cuda_ptx::space_cluster_t{},
                cuda_ptx::space_global_t{},
                cuda_ptx::cta_group_2_t{},
                &reinterpret_cast<cutlass::float_e4m3_t*>(smemBytesStagePtrA)[int32_t{0}],
                params.tmaA,
                coords,
                leadCtaMbar,
                mMmaCtaMask);
            }
            coords[int32_t{0}] += int32_t{128};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::cp_async_bulk_tensor(
                cuda_ptx::space_cluster_t{},
                cuda_ptx::space_global_t{},
                cuda_ptx::cta_group_2_t{},
                &reinterpret_cast<cutlass::float_e4m3_t*>(smemBytesStagePtrA)[int32_t{16384}],
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
        if ((loopOffset526) >= (int32_t{0})) {
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
  int32_t mCtaOffsetK;
  int32_t mCtaIdxX;
  int32_t mCtaIdxZ;
  uint16_t const mMmaCtaMask;
  inline __device__ LoadTaskB(KernelParams const& params,
                              KernelState const& state,
                              int32_t warpGrpStart)
    : mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mBatchIdx{int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}}
    , mBatchLimit{int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]}}
    , mCtaOffsetK{int32_t{0}}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)}
    , mMmaCtaMask{uint16_t{
        __shfl_sync(uint32_t{0xffffffff}, trtllm::dev::getCtaMask(), int32_t{0}, int32_t{32})}} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{4})) && ((state.mWarpIdx) < (int32_t{5}));
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
      4,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemBProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    int32_t smemBProdToken{int32_t{1}};
    do {
      int32_t paddedPerCtaK{(((params.k) + (int32_t{255})) / (int32_t{256})) * (int32_t{256})};
      int32_t loopEnd{paddedPerCtaK};
      bool const hasOneLoopIter{(int32_t{0}) < (loopEnd)};
      if ((mCtaIdxY) >= (state.mNumNonExitingCtas)) {
        goto ExitTileWithoutSignalingLabel;
      }
      mBatchIdx = int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]};
      mBatchLimit = int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]};
      //
      // smemB [HoistProdTryAcquire].
      //
      if ((int32_t{0}) < (loopEnd)) {
        smemBProdToken = smemBDstStack.mPipeline.producer_try_acquire(smemBProdState);
      }
      //
      // Hoist the first iter.
      //
      //
      // Loop body.
      //
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset633 = int32_t{0}; loopOffset633 < loopEnd;
           loopOffset633 += int32_t{256}) {
        bool const isLastLoopIter{((loopOffset633) + (int32_t{256})) >= (loopEnd)};
        //
        // gmemB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK4;
        { tileOffsetK4 = loopOffset633; }
        //
        // smemB [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          smemBDstStack.mPipeline.producer_acquire(smemBProdState, smemBProdToken);
          if (((loopOffset633) + (int32_t{256})) < (loopEnd)) {
            smemBProdToken = smemBDstStack.mPipeline.producer_try_acquire(
              trtllm::dev::makePipelineState(smemBProdState, int32_t{1}));
          }
        }
        //
        // smemB [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK8;
        tileOffsetK8 = tileOffsetK4;
        {
          uint64_t* barrier{smemBDstStack.mPipeline.producer_get_barrier(smemBProdState)};
          int32_t index{smemBProdState.index()};
          {}
          {
            int8_t* smemBytesBasePtrB;
            int8_t* smemBytesStagePtrB;
            smemBytesBasePtrB =
              reinterpret_cast<int8_t*>(smemBDstStack.mDepSmemPtr2) + (int32_t{131072});
            smemBytesStagePtrB = smemBytesBasePtrB + ((index) * (int32_t{16384}));
            int32_t coords[4];
            coords[int32_t{0}] = tileOffsetK8;
            coords[int32_t{1}] =
              (int32_t{trtllm::dev::getCtaRankInPair()}) * (int32_t{64}) +
              (((int32_t{128}) - ((mBatchLimit) % (int32_t{128}))) % (int32_t{128}));
            coords[int32_t{2}] = int32_t{0x40000000 /*1073741824*/};
            coords[int32_t{3}] =
              (((mCtaIdxY) * (int32_t{128})) +
               ((int32_t{0}) -
                (((int32_t{128}) - ((mBatchLimit) % (int32_t{128}))) % (int32_t{128})))) +
              (int32_t{0x40000000 /*1073741824*/});
            uint64_t* leadCtaMbar;
            leadCtaMbar = cuda_ptx::mapa(cuda_ptx::space_cluster_t{},
                                         barrier,
                                         int32_t{trtllm::dev::getLeadCtaRank()});
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::cp_async_bulk_tensor(
                cuda_ptx::space_cluster_t{},
                cuda_ptx::space_global_t{},
                cuda_ptx::cta_group_2_t{},
                &reinterpret_cast<cutlass::float_e4m3_t*>(smemBytesStagePtrB)[int32_t{0}],
                params.tmaB,
                coords,
                leadCtaMbar,
                mMmaCtaMask);
            }
            coords[int32_t{0}] += int32_t{128};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::cp_async_bulk_tensor(
                cuda_ptx::space_cluster_t{},
                cuda_ptx::space_global_t{},
                cuda_ptx::cta_group_2_t{},
                &reinterpret_cast<cutlass::float_e4m3_t*>(smemBytesStagePtrB)[int32_t{8192}],
                params.tmaB,
                coords,
                leadCtaMbar,
                mMmaCtaMask);
            }
          }
        }
        //
        // smemB [ProdPreCommit, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset633) >= (int32_t{0})) {
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
struct LoadSfATask {
  int32_t mCtaIdxY;
  int32_t mBatchIdx;
  int32_t mBatchLimit;
  int32_t mCtaOffsetK;
  int32_t mCtaIdxX;
  int32_t mCtaIdxZ;
  int32_t const mWarpGrpThreadIdx;
  inline __device__ LoadSfATask(KernelParams const& params,
                                KernelState const& state,
                                int32_t warpGrpStart)
    : mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mBatchIdx{int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}}
    , mBatchLimit{int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]}}
    , mCtaOffsetK{int32_t{0}}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)}
    , mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{7})) && ((state.mWarpIdx) < (int32_t{8}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 SmemSfASmem& smemSfADstSmem,
                                 SmemSfAStack& smemSfADstStack,
                                 WorkIdSmem& workIdSrcSmem,
                                 WorkIdStack& workIdSrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<96>{});
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsState{};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsReleaseState{};
    int32_t workIdConsToken{int32_t{0}};
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      4,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemSfAProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    int32_t smemSfAProdToken{int32_t{1}};
    do {
      int32_t paddedPerCtaK{(((params.k) + (int32_t{255})) / (int32_t{256})) * (int32_t{256})};
      int32_t loopEnd{paddedPerCtaK};
      if ((mCtaIdxY) >= (state.mNumNonExitingCtas)) {
        goto ExitTileWithoutSignalingLabel;
      }
      mBatchIdx = int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]};
      mBatchLimit = int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]};
      //
      // Unrolled head iter 0.
      //
      if ((int32_t{0}) < (loopEnd)) {
        //
        // gmemSfA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK5;
        { tileOffsetK5 = int32_t{0}; }
        //
        // smemSfA [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { smemSfAProdToken = smemSfADstStack.mPipeline.producer_try_acquire(smemSfAProdState); }
        }
        { smemSfADstStack.mPipeline.producer_acquire(smemSfAProdState, smemSfAProdToken); }
        //
        // smemSfA [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK9;
        tileOffsetK9 = tileOffsetK5;
        {
          uint64_t* barrier{smemSfADstStack.mPipeline.producer_get_barrier(smemSfAProdState)};
          int32_t index{smemSfAProdState.index()};
          {}
          {
            int32_t coords[4];
            coords[int32_t{0}] = int32_t{0};
            coords[int32_t{1}] = int32_t{0};
            coords[int32_t{2}] = (tileOffsetK9) / (int32_t{128});
            coords[int32_t{3}] = (mBatchIdx) * (params.tileStridePerBatch) + (mCtaIdxX);
            uint64_t* leadCtaMbar;
            leadCtaMbar = cuda_ptx::mapa(cuda_ptx::space_cluster_t{},
                                         barrier,
                                         int32_t{trtllm::dev::getLeadCtaRank()});
            uint16_t ctaMask{__shfl_sync(uint32_t{0xffffffff},
                                         trtllm::dev::getCtaMask(),
                                         int32_t{0},
                                         int32_t{32})};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             cuda_ptx::cta_group_2_t{},
                                             &smemSfADstSmem.mArray[index][int32_t{0}],
                                             params.tmaSfA,
                                             coords,
                                             leadCtaMbar,
                                             ctaMask);
            }
          }
        }
        //
        // gmemSfA [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        //
        // smemSfA [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
      }
      //
      // Loop body.
      //
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset772 = int32_t{256}; loopOffset772 < loopEnd;
           loopOffset772 += int32_t{256}) {
        bool const isFirstLoopIter{(loopOffset772) == (int32_t{256})};
        bool const isLastLoopIter{((loopOffset772) + (int32_t{256})) >= (loopEnd)};
        //
        // smemSfA [ProdPreCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset772) >= (int32_t{256})) {
        }
        //
        // smemSfA [ProdCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset772) >= (int32_t{256})) {
          {
            smemSfADstStack.mPipeline.producer_commit(smemSfAProdState);
          }
          ++smemSfAProdState;
        }
        //
        // gmemSfA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK5;
        { tileOffsetK5 = loopOffset772; }
        //
        // smemSfA [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          if ((loopOffset772) >= (int32_t{0})) {
            smemSfAProdToken = smemSfADstStack.mPipeline.producer_try_acquire(smemSfAProdState);
          }
        }
        { smemSfADstStack.mPipeline.producer_acquire(smemSfAProdState, smemSfAProdToken); }
        //
        // smemSfA [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK9;
        tileOffsetK9 = tileOffsetK5;
        {
          uint64_t* barrier{smemSfADstStack.mPipeline.producer_get_barrier(smemSfAProdState)};
          int32_t index{smemSfAProdState.index()};
          {}
          {
            int32_t coords[4];
            coords[int32_t{0}] = int32_t{0};
            coords[int32_t{1}] = int32_t{0};
            coords[int32_t{2}] = (tileOffsetK9) / (int32_t{128});
            coords[int32_t{3}] = (mBatchIdx) * (params.tileStridePerBatch) + (mCtaIdxX);
            uint64_t* leadCtaMbar;
            leadCtaMbar = cuda_ptx::mapa(cuda_ptx::space_cluster_t{},
                                         barrier,
                                         int32_t{trtllm::dev::getLeadCtaRank()});
            uint16_t ctaMask{__shfl_sync(uint32_t{0xffffffff},
                                         trtllm::dev::getCtaMask(),
                                         int32_t{0},
                                         int32_t{32})};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             cuda_ptx::cta_group_2_t{},
                                             &smemSfADstSmem.mArray[index][int32_t{0}],
                                             params.tmaSfA,
                                             coords,
                                             leadCtaMbar,
                                             ctaMask);
            }
          }
        }
        //
        // gmemSfA [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        //
        // smemSfA [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
      }
      //
      // Unrolled tail iter 0.
      //
      if ((loopEnd) > (int32_t{0})) {
        //
        // smemSfA [ProdPreCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {}
        //
        // smemSfA [ProdCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { smemSfADstStack.mPipeline.producer_commit(smemSfAProdState); }
          ++smemSfAProdState;
        }
      }
      //
      // Tail work.
      //
      //
      // gmemSfA [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
      //
      // workId [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
      //
      // smemSfA [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
struct LoadSfBTask {
  int32_t mCtaIdxY;
  int32_t mBatchIdx;
  int32_t mBatchLimit;
  int32_t mCtaOffsetK;
  int32_t mCtaIdxX;
  int32_t mCtaIdxZ;
  int32_t const mWarpGrpThreadIdx;
  inline __device__ LoadSfBTask(KernelParams const& params,
                                KernelState const& state,
                                int32_t warpGrpStart)
    : mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mBatchIdx{int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}}
    , mBatchLimit{int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]}}
    , mCtaOffsetK{int32_t{0}}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)}
    , mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{5})) && ((state.mWarpIdx) < (int32_t{6}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 SmemSfBSmem& smemSfBDstSmem,
                                 SmemSfBStack& smemSfBDstStack,
                                 WorkIdSmem& workIdSrcSmem,
                                 WorkIdStack& workIdSrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<96>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsState{};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsReleaseState{};
    int32_t workIdConsToken{int32_t{0}};
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      4,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemSfBProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    int32_t smemSfBProdToken{int32_t{1}};
    do {
      int32_t paddedPerCtaK{(((params.k) + (int32_t{255})) / (int32_t{256})) * (int32_t{256})};
      int32_t loopEnd{paddedPerCtaK};
      if ((mCtaIdxY) >= (state.mNumNonExitingCtas)) {
        goto ExitTileWithoutSignalingLabel;
      }
      mBatchIdx = int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]};
      mBatchLimit = int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]};
      //
      // Unrolled head iter 0.
      //
      if ((int32_t{0}) < (loopEnd)) {
        //
        // gmemSfB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK6;
        { tileOffsetK6 = int32_t{0}; }
        //
        // smemSfB [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { smemSfBProdToken = smemSfBDstStack.mPipeline.producer_try_acquire(smemSfBProdState); }
        }
        { smemSfBDstStack.mPipeline.producer_acquire(smemSfBProdState, smemSfBProdToken); }
        //
        // smemSfB [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK10;
        tileOffsetK10 = tileOffsetK6;
        {
          uint64_t* barrier{smemSfBDstStack.mPipeline.producer_get_barrier(smemSfBProdState)};
          int32_t index{smemSfBProdState.index()};
          {}
          {
            int32_t coords[4];
            coords[int32_t{0}] = int32_t{0};
            coords[int32_t{1}] = int32_t{0};
            coords[int32_t{2}] = (tileOffsetK10) / (int32_t{128});
            coords[int32_t{3}] = mCtaIdxY;
            uint64_t* leadCtaMbar;
            leadCtaMbar = cuda_ptx::mapa(cuda_ptx::space_cluster_t{},
                                         barrier,
                                         int32_t{trtllm::dev::getLeadCtaRank()});
            uint16_t ctaMask{__shfl_sync(uint32_t{0xffffffff},
                                         trtllm::dev::getCtaMask(),
                                         int32_t{0},
                                         int32_t{32})};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             cuda_ptx::cta_group_2_t{},
                                             &smemSfBDstSmem.mArray[index][int32_t{0}],
                                             params.tmaSfB,
                                             coords,
                                             leadCtaMbar,
                                             ctaMask);
            }
          }
        }
        //
        // gmemSfB [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        //
        // smemSfB [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
      }
      //
      // Loop body.
      //
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset918 = int32_t{256}; loopOffset918 < loopEnd;
           loopOffset918 += int32_t{256}) {
        bool const isFirstLoopIter{(loopOffset918) == (int32_t{256})};
        bool const isLastLoopIter{((loopOffset918) + (int32_t{256})) >= (loopEnd)};
        //
        // smemSfB [ProdPreCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset918) >= (int32_t{256})) {
        }
        //
        // smemSfB [ProdCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset918) >= (int32_t{256})) {
          {
            smemSfBDstStack.mPipeline.producer_commit(smemSfBProdState);
          }
          ++smemSfBProdState;
        }
        //
        // gmemSfB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK6;
        { tileOffsetK6 = loopOffset918; }
        //
        // smemSfB [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          if ((loopOffset918) >= (int32_t{0})) {
            smemSfBProdToken = smemSfBDstStack.mPipeline.producer_try_acquire(smemSfBProdState);
          }
        }
        { smemSfBDstStack.mPipeline.producer_acquire(smemSfBProdState, smemSfBProdToken); }
        //
        // smemSfB [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK10;
        tileOffsetK10 = tileOffsetK6;
        {
          uint64_t* barrier{smemSfBDstStack.mPipeline.producer_get_barrier(smemSfBProdState)};
          int32_t index{smemSfBProdState.index()};
          {}
          {
            int32_t coords[4];
            coords[int32_t{0}] = int32_t{0};
            coords[int32_t{1}] = int32_t{0};
            coords[int32_t{2}] = (tileOffsetK10) / (int32_t{128});
            coords[int32_t{3}] = mCtaIdxY;
            uint64_t* leadCtaMbar;
            leadCtaMbar = cuda_ptx::mapa(cuda_ptx::space_cluster_t{},
                                         barrier,
                                         int32_t{trtllm::dev::getLeadCtaRank()});
            uint16_t ctaMask{__shfl_sync(uint32_t{0xffffffff},
                                         trtllm::dev::getCtaMask(),
                                         int32_t{0},
                                         int32_t{32})};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             cuda_ptx::cta_group_2_t{},
                                             &smemSfBDstSmem.mArray[index][int32_t{0}],
                                             params.tmaSfB,
                                             coords,
                                             leadCtaMbar,
                                             ctaMask);
            }
          }
        }
        //
        // gmemSfB [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        //
        // smemSfB [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
      }
      //
      // Unrolled tail iter 0.
      //
      if ((loopEnd) > (int32_t{0})) {
        //
        // smemSfB [ProdPreCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {}
        //
        // smemSfB [ProdCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { smemSfBDstStack.mPipeline.producer_commit(smemSfBProdState); }
          ++smemSfBProdState;
        }
      }
      //
      // Tail work.
      //
      //
      // gmemSfB [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
      //
      // workId [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
      //
      // smemSfB [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
struct CopySfAbTask {
  int32_t mCtaOffsetK;
  int32_t mCtaIdxX;
  int32_t mCtaIdxY;
  int32_t mCtaIdxZ;
  uint32_t const mTmemBaseOffset;
  inline __device__ CopySfAbTask(KernelParams const& params,
                                 KernelState const& state,
                                 int32_t warpGrpStart)
    : mCtaOffsetK{int32_t{0}}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)}
    , mTmemBaseOffset{uint32_t{
        __shfl_sync(uint32_t{0xffffffff}, (*state.mTmemSwStatePtr), int32_t{0}, int32_t{32})}} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{8})) && ((state.mWarpIdx) < (int32_t{9}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 TmemSfAbStack& tmemSfAbDstStack,
                                 SmemSfASmem& smemSfASrcSmem,
                                 SmemSfAStack& smemSfASrcStack,
                                 SmemSfBSmem& smemSfBSrcSmem,
                                 SmemSfBStack& smemSfBSrcStack,
                                 WorkIdSmem& workIdSrcSmem,
                                 WorkIdStack& workIdSrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<96>{});
    if (!((int32_t{cute::block_rank_in_cluster()}) == (int32_t{trtllm::dev::getLeadCtaRank()}))) {
      return;
    }
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      4,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemSfAConsState{};
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      4,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState
      smemSfAConsReleaseState{};
    int32_t smemSfAConsToken{int32_t{0}};
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      4,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemSfBConsState{};
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      4,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState
      smemSfBConsReleaseState{};
    int32_t smemSfBConsToken{int32_t{0}};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsState{};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsReleaseState{};
    int32_t workIdConsToken{int32_t{0}};
    trtllm::dev::CutlassUmmaConsumerAsyncPipeline<
      4,
      false,
      false,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState tmemSfAbProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    int32_t tmemSfAbProdToken{int32_t{1}};
    do {
      int32_t paddedPerCtaK{(((params.k) + (int32_t{255})) / (int32_t{256})) * (int32_t{256})};
      int32_t loopEnd{paddedPerCtaK};
      if ((mCtaIdxY) >= (state.mNumNonExitingCtas)) {
        goto ExitTileWithoutSignalingLabel;
      }
      //
      // Loop body.
      //
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset1042 = int32_t{0}; loopOffset1042 < loopEnd;
           loopOffset1042 += int32_t{256}) {
        bool const isFirstLoopIter{(loopOffset1042) == (int32_t{0})};
        bool const isLastLoopIter{((loopOffset1042) + (int32_t{256})) >= (loopEnd)};
        //
        // tmemSfAb [ProdPreCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset1042) >= (int32_t{256})) {
        }
        //
        // tmemSfAb [ProdCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset1042) >= (int32_t{256})) {
          {
            tmemSfAbDstStack.mPipeline.producer_commit(tmemSfAbProdState);
          }
          ++tmemSfAbProdState;
        }
        //
        // smemSfA [ConsRelease, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset1042) >= (int32_t{256})) {
          {
            smemSfASrcStack.mPipeline.consumer_release(smemSfAConsReleaseState);
          }
          ++smemSfAConsReleaseState;
        }
        //
        // smemSfB [ConsRelease, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset1042) >= (int32_t{256})) {
          {
            smemSfBSrcStack.mPipeline.consumer_release(smemSfBConsReleaseState);
          }
          ++smemSfBConsReleaseState;
        }
        //
        // smemSfA [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { smemSfAConsToken = smemSfASrcStack.mPipeline.consumer_try_wait(smemSfAConsState); }
          smemSfASrcStack.mPipeline.consumer_wait(smemSfAConsState, smemSfAConsToken);
        }
        //
        // smemSfB [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { smemSfBConsToken = smemSfBSrcStack.mPipeline.consumer_try_wait(smemSfBConsState); }
          smemSfBSrcStack.mPipeline.consumer_wait(smemSfBConsState, smemSfBConsToken);
        }
        //
        // smemSfA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_ue8m0_t* smemPtrSfA9;
        {
          int32_t index{smemSfAConsState.index()};
          smemPtrSfA9 = smemSfASrcStack.mPtr + ((index) * (int32_t{1024}));
          ++smemSfAConsState;
        }
        //
        // smemSfB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_ue8m0_t* smemPtrSfB10;
        {
          int32_t index{smemSfBConsState.index()};
          smemPtrSfB10 = smemSfBSrcStack.mPtr + ((index) * (int32_t{1024}));
          ++smemSfBConsState;
        }
        //
        // tmemSfAb [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          if ((loopOffset1042) >= (int32_t{0})) {
            tmemSfAbProdToken = tmemSfAbDstStack.mPipeline.producer_try_acquire(tmemSfAbProdState);
          }
        }
        { tmemSfAbDstStack.mPipeline.producer_acquire(tmemSfAbProdState, tmemSfAbProdToken); }
        //
        // tmemSfAb [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_ue8m0_t* smemPtrSfA11;
        cutlass::float_ue8m0_t* smemPtrSfB11;
        smemPtrSfA11 = smemPtrSfA9;
        smemPtrSfB11 = smemPtrSfB10;
        {
          int32_t index{tmemSfAbProdState.index()};
          {
            uint32_t tmemBaseAddr{((mTmemBaseOffset) + (uint32_t{256})) +
                                  ((static_cast<uint32_t>(index)) * (uint32_t{8}))};
            uint64_t smemDesc{
              trtllm::dev::createSmemDesc(smemPtrSfA11, uint32_t{65536}, uint32_t{16392})};
            {
              {
                uint32_t tmemAddr{tmemBaseAddr};
                if (bool{cute::elect_one_sync()}) {
                  cuda_ptx::tcgen05_cp_32x128b_warpx4(cuda_ptx::cta_group_2_t{},
                                                      tmemAddr,
                                                      smemDesc);
                }
              }
            }
            {
              trtllm::dev::incrSmemAddr(smemDesc, int32_t{32});
              {
                uint32_t tmemAddr{(tmemBaseAddr) + (uint32_t{4})};
                if (bool{cute::elect_one_sync()}) {
                  cuda_ptx::tcgen05_cp_32x128b_warpx4(cuda_ptx::cta_group_2_t{},
                                                      tmemAddr,
                                                      smemDesc);
                }
              }
            }
          }
          {
            uint32_t tmemBaseAddr{((mTmemBaseOffset) + (uint32_t{288})) +
                                  ((static_cast<uint32_t>(index)) * (uint32_t{8}))};
            uint64_t smemDesc{
              trtllm::dev::createSmemDesc(smemPtrSfB11, uint32_t{65536}, uint32_t{16392})};
            {
              {
                uint32_t tmemAddr{tmemBaseAddr};
                if (bool{cute::elect_one_sync()}) {
                  cuda_ptx::tcgen05_cp_32x128b_warpx4(cuda_ptx::cta_group_2_t{},
                                                      tmemAddr,
                                                      smemDesc);
                }
              }
            }
            {
              trtllm::dev::incrSmemAddr(smemDesc, int32_t{32});
              {
                uint32_t tmemAddr{(tmemBaseAddr) + (uint32_t{4})};
                if (bool{cute::elect_one_sync()}) {
                  cuda_ptx::tcgen05_cp_32x128b_warpx4(cuda_ptx::cta_group_2_t{},
                                                      tmemAddr,
                                                      smemDesc);
                }
              }
            }
          }
        }
        //
        // smemSfA [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        //
        // smemSfB [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        //
        // tmemSfAb [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
      }
      //
      // Unrolled tail iter 0.
      //
      if ((loopEnd) > (int32_t{0})) {
        //
        // tmemSfAb [ProdPreCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopEnd) >= (int32_t{256})) {
        }
        //
        // tmemSfAb [ProdCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopEnd) >= (int32_t{256})) {
          {
            tmemSfAbDstStack.mPipeline.producer_commit(tmemSfAbProdState);
          }
          ++tmemSfAbProdState;
        }
        //
        // smemSfA [ConsRelease, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopEnd) >= (int32_t{256})) {
          {
            smemSfASrcStack.mPipeline.consumer_release(smemSfAConsReleaseState);
          }
          ++smemSfAConsReleaseState;
        }
        //
        // smemSfB [ConsRelease, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopEnd) >= (int32_t{256})) {
          {
            smemSfBSrcStack.mPipeline.consumer_release(smemSfBConsReleaseState);
          }
          ++smemSfBConsReleaseState;
        }
      }
      //
      // Tail work.
      //
      //
      // smemSfA [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
      //
      // smemSfB [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
      //
      // workId [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
      //
      // tmemSfAb [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
    return ((state.mWarpIdx) >= (int32_t{9})) && ((state.mWarpIdx) < (int32_t{10}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 Mma0Stack& mma0DstStack,
                                 SmemASmem& smemASrcSmem,
                                 SmemAStack& smemASrcStack,
                                 SmemBSmem& smemBSrcSmem,
                                 SmemBStack& smemBSrcStack,
                                 TmemSfAbStack& tmemSfAbSrcStack,
                                 WorkIdSmem& workIdSrcSmem,
                                 WorkIdStack& workIdSrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<96>{});
    if (!((int32_t{cute::block_rank_in_cluster()}) == (int32_t{trtllm::dev::getLeadCtaRank()}))) {
      return;
    }
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      4,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemAConsState{};
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      4,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemAConsReleaseState{};
    int32_t smemAConsToken{int32_t{0}};
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      4,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemBConsState{};
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      4,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemBConsReleaseState{};
    int32_t smemBConsToken{int32_t{0}};
    trtllm::dev::CutlassUmmaConsumerAsyncPipeline<
      4,
      false,
      false,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState tmemSfAbConsState{};
    trtllm::dev::CutlassUmmaConsumerAsyncPipeline<
      4,
      false,
      false,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState
      tmemSfAbConsReleaseState{};
    int32_t tmemSfAbConsToken{int32_t{0}};
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
      int32_t paddedPerCtaK{(((params.k) + (int32_t{255})) / (int32_t{256})) * (int32_t{256})};
      int32_t loopEnd{paddedPerCtaK};
      if ((mCtaIdxY) >= (state.mNumNonExitingCtas)) {
        goto ExitTileWithoutSignalingLabel;
      }
      //
      // Loop body.
      //
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset1257 = int32_t{0}; loopOffset1257 < loopEnd;
           loopOffset1257 += int32_t{256}) {
        bool const isFirstLoopIter{(loopOffset1257) == (int32_t{0})};
        bool const isLastLoopIter{((loopOffset1257) + (int32_t{256})) >= (loopEnd)};
        //
        // mma0 [ProdTryAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if (isFirstLoopIter) {
          if ((loopOffset1257) >= (int32_t{0})) {
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
        // tmemSfAb [ConsTryWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        { tmemSfAbConsToken = tmemSfAbSrcStack.mPipeline.consumer_try_wait(tmemSfAbConsState); }
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
        // tmemSfAb [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{8388608}].
        //
        { tmemSfAbSrcStack.mPipeline.consumer_wait(tmemSfAbConsState, tmemSfAbConsToken); }
        //
        // smemA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_e4m3_t* smemPtrA7;
        {
          int32_t index{smemAConsState.index()};
          int8_t* smemBytesBasePtrA;
          smemBytesBasePtrA = reinterpret_cast<int8_t*>(smemASrcStack.mDepSmemPtr2) + (int32_t{0});
          int8_t* smemBytesStagePtrA;
          smemBytesStagePtrA = smemBytesBasePtrA + ((index) * (int32_t{32768}));
          smemPtrA7 = reinterpret_cast<cutlass::float_e4m3_t*>(smemBytesStagePtrA) + (int32_t{0});
          ++smemAConsState;
        }
        //
        // smemB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_e4m3_t* smemPtrB8;
        {
          int32_t index{smemBConsState.index()};
          int8_t* smemBytesBasePtrB;
          smemBytesBasePtrB =
            reinterpret_cast<int8_t*>(smemBSrcStack.mDepSmemPtr2) + (int32_t{131072});
          int8_t* smemBytesStagePtrB;
          smemBytesStagePtrB = smemBytesBasePtrB + ((index) * (int32_t{16384}));
          smemPtrB8 = reinterpret_cast<cutlass::float_e4m3_t*>(smemBytesStagePtrB) + (int32_t{0});
          ++smemBConsState;
        }
        //
        // tmemSfAb [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        uint32_t tmemAddrSfA11;
        uint32_t tmemAddrSfB11;
        {
          int32_t index{tmemSfAbConsState.index()};
          tmemAddrSfA11 =
            ((mTmemBaseOffset) + (uint32_t{256})) + (static_cast<uint32_t>((index) * (int32_t{8})));
          tmemAddrSfB11 =
            ((mTmemBaseOffset) + (uint32_t{288})) + (static_cast<uint32_t>((index) * (int32_t{8})));
          ++tmemSfAbConsState;
        }
        //
        // mma0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_e4m3_t* smemPtrA13;
        cutlass::float_e4m3_t* smemPtrB13;
        uint32_t tmemAddrSfA13;
        uint32_t tmemAddrSfB13;
        smemPtrA13 = smemPtrA7;
        smemPtrB13 = smemPtrB8;
        tmemAddrSfA13 = tmemAddrSfA11;
        tmemAddrSfB13 = tmemAddrSfB11;
        {
          int32_t index{mma0ProdState.index()};
          uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{128})))};
          uint32_t ptrTmemOffsetD{ptrTmemD};
          cutlass::float_e4m3_t* ptrWithOffsetSmemA{(smemPtrA13 + int32_t{0})};
          cutlass::float_e4m3_t* ptrWithOffsetSmemB{(smemPtrB13 + int32_t{0})};
          {
            uint32_t tmemPtrD{ptrTmemOffsetD};
            uint32_t tmemPtrSfA{tmemAddrSfA13};
            uint32_t tmemPtrSfB{tmemAddrSfB13};
            //
            // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
            //
            uint64_t smemDescA{
              trtllm::dev::createSmemDesc(ptrWithOffsetSmemA,
                                          uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                          uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
            //
            // leadingDimInBytes = 8192, strideInBytes = 1024, swizzleMode = 1.
            //
            uint64_t smemDescB{
              trtllm::dev::createSmemDesc(ptrWithOffsetSmemB,
                                          uint32_t{0x2000000 /*hi=512, lo=0*/},
                                          uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
            //
            // MMA inst for mi=0 ni=0 ki=0.
            //
            uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc_block(int32_t{1},
                                                                          int32_t{5},
                                                                          int32_t{0},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{128},
                                                                          int32_t{32},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          false)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block32(cuda_ptx::kind_mxf8f6f4,
                                                        cuda_ptx::cta_group_2,
                                                        tmemPtrD,
                                                        smemDescA,
                                                        smemDescB,
                                                        utcmmaDesc_0_0_0,
                                                        tmemPtrSfA,
                                                        tmemPtrSfB,
                                                        bool{(loopOffset1257) != (int32_t{0})});
            }
            //
            // MMA inst for mi=0 ni=0 ki=1.
            //
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc_block(int32_t{1},
                                                                          int32_t{5},
                                                                          int32_t{0},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{128},
                                                                          int32_t{32},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{1},
                                                                          int32_t{1},
                                                                          false)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block32(cuda_ptx::kind_mxf8f6f4,
                                                        cuda_ptx::cta_group_2,
                                                        tmemPtrD,
                                                        smemDescA,
                                                        smemDescB,
                                                        utcmmaDesc_0_0_1,
                                                        tmemPtrSfA,
                                                        tmemPtrSfB,
                                                        bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=2.
            //
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc_block(int32_t{1},
                                                                          int32_t{5},
                                                                          int32_t{0},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{128},
                                                                          int32_t{32},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{2},
                                                                          int32_t{2},
                                                                          false)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block32(cuda_ptx::kind_mxf8f6f4,
                                                        cuda_ptx::cta_group_2,
                                                        tmemPtrD,
                                                        smemDescA,
                                                        smemDescB,
                                                        utcmmaDesc_0_0_2,
                                                        tmemPtrSfA,
                                                        tmemPtrSfB,
                                                        bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=3.
            //
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc_block(int32_t{1},
                                                                          int32_t{5},
                                                                          int32_t{0},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{128},
                                                                          int32_t{32},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{3},
                                                                          int32_t{3},
                                                                          false)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block32(cuda_ptx::kind_mxf8f6f4,
                                                        cuda_ptx::cta_group_2,
                                                        tmemPtrD,
                                                        smemDescA,
                                                        smemDescB,
                                                        utcmmaDesc_0_0_3,
                                                        tmemPtrSfA,
                                                        tmemPtrSfB,
                                                        bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=4.
            //
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{1018});
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{506});
            tmemPtrSfA += uint32_t{0x4 /*hi=0, lo=4*/};
            tmemPtrSfB += uint32_t{0x4 /*hi=0, lo=4*/};
            uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc_block(int32_t{1},
                                                                          int32_t{5},
                                                                          int32_t{0},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{128},
                                                                          int32_t{32},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          false)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block32(cuda_ptx::kind_mxf8f6f4,
                                                        cuda_ptx::cta_group_2,
                                                        tmemPtrD,
                                                        smemDescA,
                                                        smemDescB,
                                                        utcmmaDesc_0_0_4,
                                                        tmemPtrSfA,
                                                        tmemPtrSfB,
                                                        bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=5.
            //
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc_block(int32_t{1},
                                                                          int32_t{5},
                                                                          int32_t{0},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{128},
                                                                          int32_t{32},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{1},
                                                                          int32_t{1},
                                                                          false)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block32(cuda_ptx::kind_mxf8f6f4,
                                                        cuda_ptx::cta_group_2,
                                                        tmemPtrD,
                                                        smemDescA,
                                                        smemDescB,
                                                        utcmmaDesc_0_0_5,
                                                        tmemPtrSfA,
                                                        tmemPtrSfB,
                                                        bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=6.
            //
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc_block(int32_t{1},
                                                                          int32_t{5},
                                                                          int32_t{0},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{128},
                                                                          int32_t{32},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{2},
                                                                          int32_t{2},
                                                                          false)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block32(cuda_ptx::kind_mxf8f6f4,
                                                        cuda_ptx::cta_group_2,
                                                        tmemPtrD,
                                                        smemDescA,
                                                        smemDescB,
                                                        utcmmaDesc_0_0_6,
                                                        tmemPtrSfA,
                                                        tmemPtrSfB,
                                                        bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=7.
            //
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc_block(int32_t{1},
                                                                          int32_t{5},
                                                                          int32_t{0},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{128},
                                                                          int32_t{32},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{3},
                                                                          int32_t{3},
                                                                          false)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block32(cuda_ptx::kind_mxf8f6f4,
                                                        cuda_ptx::cta_group_2,
                                                        tmemPtrD,
                                                        smemDescA,
                                                        smemDescB,
                                                        utcmmaDesc_0_0_7,
                                                        tmemPtrSfA,
                                                        tmemPtrSfB,
                                                        bool{true});
            }
          }
        }
        //
        // smemA [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset1257) >= (int32_t{0})) {
          {
            smemASrcStack.mPipeline.consumer_release(smemAConsReleaseState);
          }
          ++smemAConsReleaseState;
        }
        //
        // smemB [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset1257) >= (int32_t{0})) {
          {
            smemBSrcStack.mPipeline.consumer_release(smemBConsReleaseState);
          }
          ++smemBConsReleaseState;
        }
        //
        // tmemSfAb [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset1257) >= (int32_t{0})) {
          {
            tmemSfAbSrcStack.mPipeline.consumer_release(tmemSfAbConsReleaseState);
          }
          ++tmemSfAbConsReleaseState;
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
  int32_t mCtaOffsetK;
  int32_t mCtaIdxX;
  int32_t mCtaIdxZ;
  uint32_t const mTmemBaseOffset;
  int32_t const mWarpGrpThreadIdx;
  cutlass::Array<float, 64> frg14;
  int32_t const mGridDimX;
  inline __device__ EpilogueTask0(KernelParams const& params,
                                  KernelState const& state,
                                  int32_t warpGrpStart)
    : mWarpGrpWarpIdx{(state.mWarpIdx) - (warpGrpStart)}
    , mLaneIdx{(state.mThreadIdx) % (int32_t{32})}
    , mWarpGrp4WarpIdx{mWarpGrpWarpIdx}
    , mWarpGrp4Idx{int32_t{0}}
    , mWarpRowIdx{(mWarpGrp4WarpIdx) * (int32_t{32})}
    , mQuadRowIdx{((mLaneIdx) / (int32_t{4})) * (int32_t{4})}
    , mBaseRowIdx{(mWarpRowIdx) + (mQuadRowIdx)}
    , mLaneColIdx{((mLaneIdx) % (int32_t{4})) * (int32_t{2})}
    , mBaseTmemCol{mLaneColIdx}
    , mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mBatchIdx{int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}}
    , mBatchLimit{int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]}}
    , mCtaOffsetK{int32_t{0}}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)}
    , mTmemBaseOffset{uint32_t{
        __shfl_sync(uint32_t{0xffffffff}, (*state.mTmemSwStatePtr), int32_t{0}, int32_t{32})}}
    , mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))}
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
      int32_t paddedPerCtaK{(((params.k) + (int32_t{255})) / (int32_t{256})) * (int32_t{256})};
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
      //
      // SmemBias::createFctProdVars.
      //
      int8_t* ptrSmemBaseBias;
      float* ptrSmemBias;
      ptrSmemBaseBias = reinterpret_cast<int8_t*>(gmemC0DstStack.mDepSmemPtr2) + (int32_t{212992});
      ptrSmemBias = reinterpret_cast<float*>(ptrSmemBaseBias) + (int32_t{0});
      //
      // Loading bias to SMEM.
      //
      {
        if (bool{reinterpret_cast<float const*>(params.ptrBias) != nullptr}) {
          if ((mWarpGrpThreadIdx) < (int32_t{128})) {
            int32_t offsetTileM{((mBatchIdx) * (params.tileStridePerBatch) + (mCtaIdxX)) *
                                (int32_t{128})};
            if (((offsetTileM) + (mWarpGrpThreadIdx)) < ((params.nm) * (params.numBatches))) {
              ptrSmemBias[mWarpGrpThreadIdx] = float{reinterpret_cast<float const*>(
                params.ptrBias)[(offsetTileM) + (mWarpGrpThreadIdx)]};
            }
          }
          trtllm::dev::CutlassNamedBarrier::sync(128, 8);
        }
      }
      //
      // Hoist the first iter.
      //
      //
      // Loop body.
      //
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset1507 = int32_t{0}; loopOffset1507 < loopEnd;
           loopOffset1507 += int32_t{256}) {
        //
        // gmemC0 [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        //
        // mma0 [ConsTailRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        lastLoopOffset = loopOffset1507;
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
      uint32_t tmemBaseWithStageOffset13;
      if (hasOneLoopIter) {
        int32_t index{mma0ConsState.index()};
        uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{128})))};
        uint32_t ptrTmemOffsetD{ptrTmemD};
        tmemBaseWithStageOffset13 = ptrTmemOffsetD;
      }
      //
      // gmemC0 [ProdWork (call 0), LastIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      uint32_t tmemBaseWithStageOffset14;
      tmemBaseWithStageOffset14 = tmemBaseWithStageOffset13;
      if (hasOneLoopIter) {
        tmemBaseWithStageOffset = tmemBaseWithStageOffset14;
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
            uint32_t(&dstSlice0)[32]{reinterpret_cast<uint32_t(&)[32]>(frg14[int32_t{0}])};
            cuda_ptx::tcgen05_ld_16x256b(dstSlice0,
                                         (tmemBasePtr) +
                                           (static_cast<uint32_t>((mWarpGrp4Idx) * (int32_t{64}))));
            uint32_t(&dstSlice1)[32]{reinterpret_cast<uint32_t(&)[32]>(frg14[int32_t{32}])};
            cuda_ptx::tcgen05_ld_16x256b(
              dstSlice1,
              (tmemBasePtr) + (static_cast<uint32_t>(((mWarpGrp4Idx) * (int32_t{64})) +
                                                     (int32_t{0x100000 /*hi=16, lo=0*/}))));
          }
          cutlass::arch::fence_view_async_tmem_load();
          //
          // Add bias.
          //
          if (bool{reinterpret_cast<float const*>(params.ptrBias) != nullptr}) {
            int32_t const warpRowIdx{(mWarpGrp4WarpIdx) * (int32_t{32})};
            int32_t const quadRowIdx{(mLaneIdx) / (int32_t{4})};
            int32_t const laneColIdx{((mLaneIdx) % (int32_t{4})) * (int32_t{2})};
            //
            // Add bias (0, 0).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{(laneColIdx) + ((mWarpGrp4Idx) * (int32_t{64}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{0}] = (frg14[int32_t{0}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 1).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{1}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{1}] = (frg14[int32_t{1}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 2).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{8}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{4}] = (frg14[int32_t{4}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 3).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{9}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{5}] = (frg14[int32_t{5}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 4).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{16}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{8}] = (frg14[int32_t{8}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 5).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{17}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{9}] = (frg14[int32_t{9}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 6).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{24}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{12}] = (frg14[int32_t{12}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 7).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{25}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{13}] = (frg14[int32_t{13}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 8).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{32}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{16}] = (frg14[int32_t{16}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 9).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{33}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{17}] = (frg14[int32_t{17}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 10).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{40}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{20}] = (frg14[int32_t{20}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 11).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{41}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{21}] = (frg14[int32_t{21}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 12).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{48}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{24}] = (frg14[int32_t{24}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 13).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{49}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{25}] = (frg14[int32_t{25}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 14).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{56}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{28}] = (frg14[int32_t{28}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 15).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{57}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{29}] = (frg14[int32_t{29}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 0).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{(laneColIdx) + ((mWarpGrp4Idx) * (int32_t{64}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{2}] = (frg14[int32_t{2}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 1).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{1}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{3}] = (frg14[int32_t{3}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 2).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{8}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{6}] = (frg14[int32_t{6}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 3).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{9}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{7}] = (frg14[int32_t{7}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 4).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{16}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{10}] = (frg14[int32_t{10}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 5).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{17}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{11}] = (frg14[int32_t{11}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 6).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{24}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{14}] = (frg14[int32_t{14}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 7).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{25}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{15}] = (frg14[int32_t{15}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 8).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{32}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{18}] = (frg14[int32_t{18}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 9).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{33}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{19}] = (frg14[int32_t{19}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 10).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{40}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{22}] = (frg14[int32_t{22}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 11).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{41}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{23}] = (frg14[int32_t{23}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 12).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{48}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{26}] = (frg14[int32_t{26}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 13).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{49}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{27}] = (frg14[int32_t{27}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 14).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{56}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{30}] = (frg14[int32_t{30}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 15).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{57}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{31}] = (frg14[int32_t{31}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 0).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{(laneColIdx) + ((mWarpGrp4Idx) * (int32_t{64}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{32}] = (frg14[int32_t{32}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 1).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{1}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{33}] = (frg14[int32_t{33}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 2).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{8}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{36}] = (frg14[int32_t{36}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 3).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{9}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{37}] = (frg14[int32_t{37}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 4).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{16}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{40}] = (frg14[int32_t{40}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 5).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{17}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{41}] = (frg14[int32_t{41}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 6).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{24}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{44}] = (frg14[int32_t{44}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 7).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{25}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{45}] = (frg14[int32_t{45}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 8).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{32}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{48}] = (frg14[int32_t{48}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 9).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{33}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{49}] = (frg14[int32_t{49}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 10).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{40}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{52}] = (frg14[int32_t{52}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 11).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{41}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{53}] = (frg14[int32_t{53}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 12).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{48}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{56}] = (frg14[int32_t{56}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 13).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{49}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{57}] = (frg14[int32_t{57}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 14).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{56}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{60}] = (frg14[int32_t{60}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 15).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{57}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{61}] = (frg14[int32_t{61}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 0).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{(laneColIdx) + ((mWarpGrp4Idx) * (int32_t{64}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{34}] = (frg14[int32_t{34}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 1).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{1}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{35}] = (frg14[int32_t{35}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 2).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{8}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{38}] = (frg14[int32_t{38}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 3).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{9}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{39}] = (frg14[int32_t{39}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 4).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{16}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{42}] = (frg14[int32_t{42}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 5).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{17}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{43}] = (frg14[int32_t{43}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 6).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{24}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{46}] = (frg14[int32_t{46}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 7).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{25}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{47}] = (frg14[int32_t{47}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 8).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{32}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{50}] = (frg14[int32_t{50}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 9).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{33}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{51}] = (frg14[int32_t{51}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 10).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{40}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{54}] = (frg14[int32_t{54}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 11).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{41}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{55}] = (frg14[int32_t{55}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 12).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{48}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{58}] = (frg14[int32_t{58}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 13).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{49}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{59}] = (frg14[int32_t{59}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 14).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{56}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{62}] = (frg14[int32_t{62}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 15).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         ((mWarpGrp4Idx) * (int32_t{64}) + (int32_t{57}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{63}] = (frg14[int32_t{63}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
          }
          cuda_ptx::cp_async_bulk_wait_group_read(cuda_ptx::n32_t<0>{});
          trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{7}) + (mWarpGrp4Idx));
          //
          // Store to Smem TmaAsyncGmemC.
          //
          int8_t* ptrSmemBase;
          cutlass::bfloat16_t* ptrSmem;
          ptrSmemBase = reinterpret_cast<int8_t*>(gmemC0DstStack.mDepSmemPtr2) +
                        ((mWarpGrp4Idx) * (int32_t{16384}) + (int32_t{196608}));
          ptrSmem = reinterpret_cast<cutlass::bfloat16_t*>(ptrSmemBase) + (int32_t{0});
          //
          // Smem store idxM=0 idxN=0.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                ((((mBaseTmemCol) * (int32_t{128}) + (mBaseRowIdx)) % (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                (((mBaseTmemCol) * (int32_t{128}) + (mBaseRowIdx)) / (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                (((((mBaseTmemCol) * (int32_t{128}) + (mBaseRowIdx)) % (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{0}],
                                           frg14[int32_t{2}],
                                           frg14[int32_t{32}],
                                           frg14[int32_t{34}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=1.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{1})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{1})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{1})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{1}],
                                           frg14[int32_t{3}],
                                           frg14[int32_t{33}],
                                           frg14[int32_t{35}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=2.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{8})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{8})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{8})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{4}],
                                           frg14[int32_t{6}],
                                           frg14[int32_t{36}],
                                           frg14[int32_t{38}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=3.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{9})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{9})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{9})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{5}],
                                           frg14[int32_t{7}],
                                           frg14[int32_t{37}],
                                           frg14[int32_t{39}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=4.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{16})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{16})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{16})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{8}],
                                           frg14[int32_t{10}],
                                           frg14[int32_t{40}],
                                           frg14[int32_t{42}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=5.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{17})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{17})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{17})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{9}],
                                           frg14[int32_t{11}],
                                           frg14[int32_t{41}],
                                           frg14[int32_t{43}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=6.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{24})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{24})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{24})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{12}],
                                           frg14[int32_t{14}],
                                           frg14[int32_t{44}],
                                           frg14[int32_t{46}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=7.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{25})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{25})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{25})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{13}],
                                           frg14[int32_t{15}],
                                           frg14[int32_t{45}],
                                           frg14[int32_t{47}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=8.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{32})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{32})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{32})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{16}],
                                           frg14[int32_t{18}],
                                           frg14[int32_t{48}],
                                           frg14[int32_t{50}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=9.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{33})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{33})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{33})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{17}],
                                           frg14[int32_t{19}],
                                           frg14[int32_t{49}],
                                           frg14[int32_t{51}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=10.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{40})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{40})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{40})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{20}],
                                           frg14[int32_t{22}],
                                           frg14[int32_t{52}],
                                           frg14[int32_t{54}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=11.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{41})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{41})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{41})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{21}],
                                           frg14[int32_t{23}],
                                           frg14[int32_t{53}],
                                           frg14[int32_t{55}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=12.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{48})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{48})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{48})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{24}],
                                           frg14[int32_t{26}],
                                           frg14[int32_t{56}],
                                           frg14[int32_t{58}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=13.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{49})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{49})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{49})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{25}],
                                           frg14[int32_t{27}],
                                           frg14[int32_t{57}],
                                           frg14[int32_t{59}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=14.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{56})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{56})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{56})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{28}],
                                           frg14[int32_t{30}],
                                           frg14[int32_t{60}],
                                           frg14[int32_t{62}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=15.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{57})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{57})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{57})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{29}],
                                           frg14[int32_t{31}],
                                           frg14[int32_t{61}],
                                           frg14[int32_t{63}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          cuda_ptx::fence_proxy_async(cuda_ptx::space_shared_t{});
          trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{7}) + (mWarpGrp4Idx));
          //
          // Issue TMA from smem to gmem.
          //
          if ((bool{cute::elect_one_sync()}) && ((mWarpGrp4WarpIdx) == (int32_t{0}))) {
            int8_t* ptrSmemBase;
            cutlass::bfloat16_t* ptrSmem;
            ptrSmemBase = reinterpret_cast<int8_t*>(gmemC0DstStack.mDepSmemPtr2) +
                          ((mWarpGrp4Idx) * (int32_t{16384}) + (int32_t{196608}));
            ptrSmem = reinterpret_cast<cutlass::bfloat16_t*>(ptrSmemBase) + (int32_t{0});
            int32_t coords[4];
            coords[int32_t{0}] = (mCtaIdxX) * (int32_t{128});
            coords[int32_t{1}] =
              (((int32_t{128}) - ((mBatchLimit) % (int32_t{128}))) % (int32_t{128})) +
              ((mWarpGrp4Idx) * (int32_t{64}));
            coords[int32_t{2}] = int32_t{0x40000000 /*1073741824*/};
            coords[int32_t{3}] =
              (((mCtaIdxY) * (int32_t{128})) +
               ((int32_t{0}) -
                (((int32_t{128}) - ((mBatchLimit) % (int32_t{128}))) % (int32_t{128})))) +
              (int32_t{0x40000000 /*1073741824*/});
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_global_t{},
                                           cuda_ptx::space_shared_t{},
                                           params.tmaC,
                                           coords,
                                           &ptrSmem[int32_t{0}]);
            coords[int32_t{0}] += int32_t{64};
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_global_t{},
                                           cuda_ptx::space_shared_t{},
                                           params.tmaC,
                                           coords,
                                           &ptrSmem[int32_t{4096}]);
          }
          cuda_ptx::cp_async_bulk_commit_group();
          trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{7}) + (mWarpGrp4Idx));
        }
        //
        // Epilogue tile idxM=0 idxN=1.
        //
        {
          //
          // Load from Tmem to fragment.
          //
          {
            uint32_t tmemBasePtr{tmemBaseWithStageOffset};
            uint32_t(&dstSlice0)[32]{reinterpret_cast<uint32_t(&)[32]>(frg14[int32_t{0}])};
            cuda_ptx::tcgen05_ld_16x256b(
              dstSlice0,
              (tmemBasePtr) +
                (static_cast<uint32_t>(((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}))));
            uint32_t(&dstSlice1)[32]{reinterpret_cast<uint32_t(&)[32]>(frg14[int32_t{32}])};
            cuda_ptx::tcgen05_ld_16x256b(
              dstSlice1,
              (tmemBasePtr) +
                (static_cast<uint32_t>((((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64})) +
                                       (int32_t{0x100000 /*hi=16, lo=0*/}))));
          }
          cutlass::arch::fence_view_async_tmem_load();
          //
          // Add bias.
          //
          if (bool{reinterpret_cast<float const*>(params.ptrBias) != nullptr}) {
            int32_t const warpRowIdx{(mWarpGrp4WarpIdx) * (int32_t{32})};
            int32_t const quadRowIdx{(mLaneIdx) / (int32_t{4})};
            int32_t const laneColIdx{((mLaneIdx) % (int32_t{4})) * (int32_t{2})};
            //
            // Add bias (0, 0).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{(laneColIdx) +
                                         (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{0}] = (frg14[int32_t{0}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 1).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{1}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{1}] = (frg14[int32_t{1}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 2).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{8}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{4}] = (frg14[int32_t{4}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 3).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{9}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{5}] = (frg14[int32_t{5}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 4).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{16}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{8}] = (frg14[int32_t{8}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 5).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{17}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{9}] = (frg14[int32_t{9}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 6).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{24}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{12}] = (frg14[int32_t{12}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 7).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{25}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{13}] = (frg14[int32_t{13}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 8).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{32}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{16}] = (frg14[int32_t{16}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 9).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{33}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{17}] = (frg14[int32_t{17}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 10).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{40}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{20}] = (frg14[int32_t{20}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 11).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{41}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{21}] = (frg14[int32_t{21}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 12).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{48}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{24}] = (frg14[int32_t{24}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 13).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{49}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{25}] = (frg14[int32_t{25}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 14).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{56}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{28}] = (frg14[int32_t{28}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (0, 15).
            //
            {
              int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{57}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{29}] = (frg14[int32_t{29}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 0).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{2}] = (frg14[int32_t{2}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 1).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{1}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{3}] = (frg14[int32_t{3}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 2).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{8}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{6}] = (frg14[int32_t{6}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 3).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{9}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{7}] = (frg14[int32_t{7}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 4).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{16}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{10}] = (frg14[int32_t{10}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 5).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{17}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{11}] = (frg14[int32_t{11}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 6).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{24}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{14}] = (frg14[int32_t{14}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 7).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{25}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{15}] = (frg14[int32_t{15}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 8).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{32}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{18}] = (frg14[int32_t{18}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 9).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{33}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{19}] = (frg14[int32_t{19}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 10).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{40}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{22}] = (frg14[int32_t{22}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 11).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{41}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{23}] = (frg14[int32_t{23}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 12).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{48}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{26}] = (frg14[int32_t{26}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 13).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{49}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{27}] = (frg14[int32_t{27}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 14).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{56}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{30}] = (frg14[int32_t{30}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (1, 15).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{57}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{31}] = (frg14[int32_t{31}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 0).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{32}] = (frg14[int32_t{32}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 1).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{1}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{33}] = (frg14[int32_t{33}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 2).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{8}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{36}] = (frg14[int32_t{36}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 3).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{9}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{37}] = (frg14[int32_t{37}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 4).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{16}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{40}] = (frg14[int32_t{40}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 5).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{17}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{41}] = (frg14[int32_t{41}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 6).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{24}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{44}] = (frg14[int32_t{44}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 7).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{25}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{45}] = (frg14[int32_t{45}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 8).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{32}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{48}] = (frg14[int32_t{48}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 9).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{33}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{49}] = (frg14[int32_t{49}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 10).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{40}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{52}] = (frg14[int32_t{52}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 11).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{41}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{53}] = (frg14[int32_t{53}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 12).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{48}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{56}] = (frg14[int32_t{56}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 13).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{49}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{57}] = (frg14[int32_t{57}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 14).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{56}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{60}] = (frg14[int32_t{60}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (2, 15).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{57}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{61}] = (frg14[int32_t{61}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 0).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{(laneColIdx) +
                                         (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{34}] = (frg14[int32_t{34}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 1).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{1}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{35}] = (frg14[int32_t{35}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 2).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{8}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{38}] = (frg14[int32_t{38}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 3).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{9}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{39}] = (frg14[int32_t{39}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 4).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{16}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{42}] = (frg14[int32_t{42}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 5).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{17}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{43}] = (frg14[int32_t{43}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 6).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{24}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{46}] = (frg14[int32_t{46}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 7).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{25}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{47}] = (frg14[int32_t{47}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 8).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{32}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{50}] = (frg14[int32_t{50}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 9).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{33}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{51}] = (frg14[int32_t{51}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 10).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{40}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{54}] = (frg14[int32_t{54}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 11).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{41}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{55}] = (frg14[int32_t{55}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 12).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{48}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{58}] = (frg14[int32_t{58}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 13).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{49}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{59}] = (frg14[int32_t{59}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 14).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{56}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{62}] = (frg14[int32_t{62}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
            //
            // Add bias (3, 15).
            //
            {
              int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
              int32_t const sharedColIdx{
                (laneColIdx) + (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}) + (int32_t{57}))};
              //
              // Loading bias to register.
              //
              frg14[int32_t{63}] = (frg14[int32_t{63}]) + (float{ptrSmemBias[sharedRowIdx]});
            }
          }
          cuda_ptx::cp_async_bulk_wait_group_read(cuda_ptx::n32_t<0>{});
          trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{7}) + (mWarpGrp4Idx));
          //
          // Store to Smem TmaAsyncGmemC.
          //
          int8_t* ptrSmemBase;
          cutlass::bfloat16_t* ptrSmem;
          ptrSmemBase = reinterpret_cast<int8_t*>(gmemC0DstStack.mDepSmemPtr2) +
                        ((mWarpGrp4Idx) * (int32_t{16384}) + (int32_t{196608}));
          ptrSmem = reinterpret_cast<cutlass::bfloat16_t*>(ptrSmemBase) + (int32_t{0});
          //
          // Smem store idxM=0 idxN=0.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                ((((mBaseTmemCol) * (int32_t{128}) + (mBaseRowIdx)) % (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                (((mBaseTmemCol) * (int32_t{128}) + (mBaseRowIdx)) / (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                (((((mBaseTmemCol) * (int32_t{128}) + (mBaseRowIdx)) % (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{0}],
                                           frg14[int32_t{2}],
                                           frg14[int32_t{32}],
                                           frg14[int32_t{34}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=1.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{1})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{1})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{1})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{1}],
                                           frg14[int32_t{3}],
                                           frg14[int32_t{33}],
                                           frg14[int32_t{35}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=2.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{8})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{8})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{8})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{4}],
                                           frg14[int32_t{6}],
                                           frg14[int32_t{36}],
                                           frg14[int32_t{38}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=3.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{9})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{9})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{9})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{5}],
                                           frg14[int32_t{7}],
                                           frg14[int32_t{37}],
                                           frg14[int32_t{39}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=4.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{16})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{16})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{16})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{8}],
                                           frg14[int32_t{10}],
                                           frg14[int32_t{40}],
                                           frg14[int32_t{42}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=5.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{17})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{17})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{17})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{9}],
                                           frg14[int32_t{11}],
                                           frg14[int32_t{41}],
                                           frg14[int32_t{43}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=6.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{24})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{24})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{24})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{12}],
                                           frg14[int32_t{14}],
                                           frg14[int32_t{44}],
                                           frg14[int32_t{46}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=7.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{25})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{25})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{25})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{13}],
                                           frg14[int32_t{15}],
                                           frg14[int32_t{45}],
                                           frg14[int32_t{47}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=8.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{32})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{32})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{32})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{16}],
                                           frg14[int32_t{18}],
                                           frg14[int32_t{48}],
                                           frg14[int32_t{50}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=9.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{33})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{33})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{33})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{17}],
                                           frg14[int32_t{19}],
                                           frg14[int32_t{49}],
                                           frg14[int32_t{51}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=10.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{40})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{40})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{40})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{20}],
                                           frg14[int32_t{22}],
                                           frg14[int32_t{52}],
                                           frg14[int32_t{54}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=11.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{41})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{41})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{41})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{21}],
                                           frg14[int32_t{23}],
                                           frg14[int32_t{53}],
                                           frg14[int32_t{55}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=12.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{48})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{48})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{48})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{24}],
                                           frg14[int32_t{26}],
                                           frg14[int32_t{56}],
                                           frg14[int32_t{58}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=13.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{49})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{49})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{49})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{25}],
                                           frg14[int32_t{27}],
                                           frg14[int32_t{57}],
                                           frg14[int32_t{59}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=14.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{56})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{56})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{56})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{28}],
                                           frg14[int32_t{30}],
                                           frg14[int32_t{60}],
                                           frg14[int32_t{62}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=15.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((((mBaseTmemCol) + (int32_t{57})) * (int32_t{128}) + (mBaseRowIdx)) %
                  (int32_t{128})) /
                 (int32_t{64})) *
                  (int32_t{64}) +
                ((((mBaseTmemCol) + (int32_t{57})) * (int32_t{128}) + (mBaseRowIdx)) /
                 (int32_t{128}))};
              int32_t const smemOffsetInBytes{
                ((smemRowIdx) * (int32_t{128})) +
                ((((((mBaseTmemCol) + (int32_t{57})) * (int32_t{128}) + (mBaseRowIdx)) %
                   (int32_t{64})) *
                  (int32_t{16})) /
                 (int32_t{8}))};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 4> accF4{frg14[int32_t{29}],
                                           frg14[int32_t{31}],
                                           frg14[int32_t{61}],
                                           frg14[int32_t{63}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          cuda_ptx::fence_proxy_async(cuda_ptx::space_shared_t{});
          trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{7}) + (mWarpGrp4Idx));
          //
          // Issue TMA from smem to gmem.
          //
          if ((bool{cute::elect_one_sync()}) && ((mWarpGrp4WarpIdx) == (int32_t{0}))) {
            int8_t* ptrSmemBase;
            cutlass::bfloat16_t* ptrSmem;
            ptrSmemBase = reinterpret_cast<int8_t*>(gmemC0DstStack.mDepSmemPtr2) +
                          ((mWarpGrp4Idx) * (int32_t{16384}) + (int32_t{196608}));
            ptrSmem = reinterpret_cast<cutlass::bfloat16_t*>(ptrSmemBase) + (int32_t{0});
            int32_t coords[4];
            coords[int32_t{0}] = (mCtaIdxX) * (int32_t{128});
            coords[int32_t{1}] =
              (((int32_t{128}) - ((mBatchLimit) % (int32_t{128}))) % (int32_t{128})) +
              (((int32_t{1}) + (mWarpGrp4Idx)) * (int32_t{64}));
            coords[int32_t{2}] = int32_t{0x40000000 /*1073741824*/};
            coords[int32_t{3}] =
              (((mCtaIdxY) * (int32_t{128})) +
               ((int32_t{0}) -
                (((int32_t{128}) - ((mBatchLimit) % (int32_t{128}))) % (int32_t{128})))) +
              (int32_t{0x40000000 /*1073741824*/});
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_global_t{},
                                           cuda_ptx::space_shared_t{},
                                           params.tmaC,
                                           coords,
                                           &ptrSmem[int32_t{0}]);
            coords[int32_t{0}] += int32_t{64};
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_global_t{},
                                           cuda_ptx::space_shared_t{},
                                           params.tmaC,
                                           coords,
                                           &ptrSmem[int32_t{4096}]);
          }
          cuda_ptx::cp_async_bulk_commit_group();
          trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{7}) + (mWarpGrp4Idx));
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
  __cluster_dims__(2, 1, 1) void bmm_Bfloat16_MxE2m1MxE4m3_Fp32_bA32_bB32_t128x128x256_s4_et128x64_m256x128x32_cga2x1x1_16dp256b_rM_TN_transOut_schPd2x1x2x3_biasM_bN_rgTma_clmp_dynB_sm100f(
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
  uint8_t* smemSfASmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemSfASmem)});
  smemOffset__ = (((smemOffset__) + (int32_t{1023})) / (int32_t{1024})) * (int32_t{1024});
  uint8_t* smemSfBSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemSfBSmem)});
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
  uint8_t* smemSfASmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemSfASmemBarrier)});
  uint8_t* smemSfBSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemSfBSmemBarrier)});
  uint8_t* tmemSfAbSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(TmemSfAbSmemBarrier)});
  uint8_t* mma0SmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(Mma0SmemBarrier)});
  uint8_t* clusterBarrierBuffersSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(ClusterBarrierBuffersSmemBarrier)});
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
                        int32_t{6},
                        int32_t{-1}};
  SmemBSmem* smemBSmem{reinterpret_cast<SmemBSmem*>(smemBSmemPtr)};
  SmemBSmemBarrier* smemBSmemBarrier{reinterpret_cast<SmemBSmemBarrier*>(smemBSmemBarrierPtr)};
  SmemBStack smemBStack{(*smemBSmem),
                        (*smemBSmemBarrier),
                        (*smemBufferSmem),
                        smemBufferStack,
                        state.mWarpIdx,
                        int32_t{4},
                        int32_t{-1}};
  SmemSfASmem* smemSfASmem{reinterpret_cast<SmemSfASmem*>(smemSfASmemPtr)};
  SmemSfASmemBarrier* smemSfASmemBarrier{
    reinterpret_cast<SmemSfASmemBarrier*>(smemSfASmemBarrierPtr)};
  SmemSfAStack smemSfAStack{(*smemSfASmem),
                            (*smemSfASmemBarrier),
                            state.mWarpIdx,
                            int32_t{7},
                            int32_t{-1}};
  SmemSfBSmem* smemSfBSmem{reinterpret_cast<SmemSfBSmem*>(smemSfBSmemPtr)};
  SmemSfBSmemBarrier* smemSfBSmemBarrier{
    reinterpret_cast<SmemSfBSmemBarrier*>(smemSfBSmemBarrierPtr)};
  SmemSfBStack smemSfBStack{(*smemSfBSmem),
                            (*smemSfBSmemBarrier),
                            state.mWarpIdx,
                            int32_t{5},
                            int32_t{-1}};
  TmemSfAbSmemBarrier* tmemSfAbSmemBarrier{
    reinterpret_cast<TmemSfAbSmemBarrier*>(tmemSfAbSmemBarrierPtr)};
  TmemSfAbStack tmemSfAbStack{(*tmemSfAbSmemBarrier), state.mWarpIdx, int32_t{8}, int32_t{-1}};
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
  LoadTaskA loadTaskA{params, state, int32_t{6}};
  LoadTaskB loadTaskB{params, state, int32_t{4}};
  cutlass::arch::fence_barrier_init();
  cuda_ptx::fence_mbarrier_init(cuda_ptx::sem_release_t{}, cuda_ptx::scope_cluster_t{});
  cuda_ptx::barrier_cluster_arrive(cuda_ptx::sem_relaxed_t{});
  cuda_ptx::barrier_cluster_wait();
  if ((reinterpret_cast<int32_t const&>(threadIdx.x)) < (int32_t{32})) {
    cuda_ptx::tcgen05_alloc(cuda_ptx::cta_group_2_t{}, state.mTmemSwStatePtr, int32_t{512});
    cuda_ptx::tcgen05_relinquish_alloc_permit(cuda_ptx::cta_group_2_t{});
  }
  if ((((bool{LoadTaskA::isSelected(params, state)}) ||
        (bool{LoadTaskB::isSelected(params, state)})) ||
       (bool{LoadSfATask::isSelected(params, state)})) ||
      (bool{LoadSfBTask::isSelected(params, state)})) {
  } else {
    trtllm::dev::CutlassNamedBarrier::sync(256, 9);
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
      if (bool{LoadSfATask::isSelected(params, state)}) {
        LoadSfATask loadSfATask{params, state, int32_t{7}};
        loadSfATask
          .execute(params, state, (*smemSfASmem), smemSfAStack, (*workIdSmem), workIdStack);
      } else {
        if (bool{LoadSfBTask::isSelected(params, state)}) {
          LoadSfBTask loadSfBTask{params, state, int32_t{5}};
          loadSfBTask
            .execute(params, state, (*smemSfBSmem), smemSfBStack, (*workIdSmem), workIdStack);
        } else {
          if (bool{CopySfAbTask::isSelected(params, state)}) {
            CopySfAbTask copySfAbTask{params, state, int32_t{8}};
            copySfAbTask.execute(params,
                                 state,
                                 tmemSfAbStack,
                                 (*smemSfASmem),
                                 smemSfAStack,
                                 (*smemSfBSmem),
                                 smemSfBStack,
                                 (*workIdSmem),
                                 workIdStack);
          } else {
            if (bool{MmaTask0::isSelected(params, state)}) {
              MmaTask0 mmaTask0{params, state, int32_t{9}};
              mmaTask0.execute(params,
                               state,
                               mma0Stack,
                               (*smemASmem),
                               smemAStack,
                               (*smemBSmem),
                               smemBStack,
                               tmemSfAbStack,
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
                trtllm::dev::CutlassNamedBarrier::sync(128, 10);
                int32_t const warpGrpThreadIdx{state.mThreadIdx};
                if ((warpGrpThreadIdx) < (int32_t{32})) {
                  clusterBarrierBuffersStack.mClusterBarrier.sync();
                  cuda_ptx::tcgen05_dealloc(cuda_ptx::cta_group_2_t{},
                                            uint32_t{__shfl_sync(uint32_t{0xffffffff},
                                                                 (*state.mTmemSwStatePtr),
                                                                 int32_t{0},
                                                                 int32_t{32})},
                                            int32_t{512});
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
    }
  }
}
extern "C" __global__ void
bmm_Bfloat16_MxE2m1MxE4m3_Fp32_bA32_bB32_t128x128x256_s4_et128x64_m256x128x32_cga2x1x1_16dp256b_rM_TN_transOut_schPd2x1x2x3_biasM_bN_rgTma_clmp_dynB_sm100fGetSmemSize(
  int32_t* outPtr) {
  int32_t size{0};
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemBufferSmem));
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemASmem));
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemBSmem));
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemSfASmem));
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemSfBSmem));
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
  size += static_cast<int32_t>(sizeof(SmemSfASmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(SmemSfBSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(TmemSfAbSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(Mma0SmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(ClusterBarrierBuffersSmemBarrier));
  outPtr[0] = size;
}

} // namespace batchedGemm
