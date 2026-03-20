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
#include <Bmm_Bfloat16_MxInt4Bfloat16_castBfloat16_Fp32_bA32_t128x64x256_s3_et128x64_m256x64x16_cga2x1x1_16dp256b_rM_BN_transOut_schPd2x1x2x3_bN_rgTma_clmp_dynB_sm100f.h>
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
    : mPipeline{workIdSmemBarrier.mBarriers, int32_t{1}, int32_t{960}, int32_t{0}, barInitWarpId}
    , mScheduler{&workIdSmem.workIdResponse[int32_t{0}],
                 typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<
                   cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
                   3>::Params{},
                 cute::block_id_in_cluster()}
    , workTileInfo{mScheduler.initial_work_tile_info(CuteFlatTuple385{})} {}
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
  trtllm::dev::CutlassTmaAsyncPipeline<3> mPipeline;
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
                int32_t{16384},
                ((warpId) == (barInitWarpId)) && (bool{cute::elect_one_sync()}),
                int32_t{128},
                CuteFlatTuple620{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId} {}
};
struct SmemBStack {
  int8_t* mDepSmemPtr2;
  trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
    3,
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
                CuteFlatTuple746{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId} {}
};
struct SmemSfAStack {
  trtllm::dev::CutlassTmaAsyncPipeline<3> mPipeline;
  cutlass::bfloat16_t* mPtr;
  inline __device__ SmemSfAStack(SmemSfASmem& smemSfASmem,
                                 SmemSfASmemBarrier& smemSfASmemBarrier,
                                 int32_t warpId,
                                 int32_t barInitWarpId,
                                 int32_t orderedSequenceGroupId)
    : mPipeline{smemSfASmemBarrier.mBarriers,
                warpId,
                int32_t{2048},
                ((warpId) == (barInitWarpId)) && (bool{cute::elect_one_sync()}),
                int32_t{128},
                CuteFlatTuple863{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId}
    , mPtr{&smemSfASmem.mArray[int32_t{0}][int32_t{0}]} {}
};
struct TmemCastAStack {
  trtllm::dev::CutlassUmmaConsumerAsyncPipeline<
    3,
    false,
    false,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  inline __device__ TmemCastAStack(TmemCastASmemBarrier& tmemCastASmemBarrier,
                                   int32_t warpId,
                                   int32_t barInitWarpId,
                                   int32_t orderedSequenceGroupId)
    : mPipeline{tmemCastASmemBarrier.mBarriers,
                warpId,
                int32_t{256},
                CuteFlatTuple986{},
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
                CuteFlatTuple1103{},
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
  inline __device__ LoadTaskA(KernelParams const& params,
                              KernelState const& state,
                              int32_t warpGrpStart)
    : mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mBatchIdx{int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}}
    , mBatchLimit{int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]}}
    , mCtaOffsetK{int32_t{0}}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)}
    , mBlockInClusterIdx{cute::block_id_in_cluster()} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{9})) && ((state.mWarpIdx) < (int32_t{10}));
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
    trtllm::dev::CutlassTmaAsyncPipeline<3>::PipelineState smemAProdState{int32_t{0},
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
      for (int32_t loopOffset480 = int32_t{0}; loopOffset480 < loopEnd;
           loopOffset480 += int32_t{256}) {
        bool const isLastLoopIter{((loopOffset480) + (int32_t{256})) >= (loopEnd)};
        //
        // gmemA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK3;
        { tileOffsetK3 = loopOffset480; }
        //
        // smemA [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          smemADstStack.mPipeline.producer_acquire(smemAProdState, smemAProdToken);
          if (((loopOffset480) + (int32_t{256})) < (loopEnd)) {
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
            smemBytesStagePtrA = smemBytesBasePtrA + ((index) * (int32_t{16384}));
            int32_t coords[4];
            coords[int32_t{0}] = int32_t{0};
            coords[int32_t{1}] = (mCtaIdxX) * (int32_t{128});
            coords[int32_t{2}] = (tileOffsetK7) / (int32_t{256});
            coords[int32_t{3}] = mBatchIdx;
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::cp_async_bulk_tensor(
                cuda_ptx::space_cluster_t{},
                cuda_ptx::space_global_t{},
                &reinterpret_cast<cutlass::int4b_t*>(smemBytesStagePtrA)[int32_t{0}],
                params.tmaA,
                coords,
                barrier);
            }
          }
        }
        //
        // smemA [ProdPreCommit, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset480) >= (int32_t{0})) {
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
    return ((state.mWarpIdx) >= (int32_t{8})) && ((state.mWarpIdx) < (int32_t{9}));
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
      3,
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
      for (int32_t loopOffset581 = int32_t{0}; loopOffset581 < loopEnd;
           loopOffset581 += int32_t{256}) {
        bool const isLastLoopIter{((loopOffset581) + (int32_t{256})) >= (loopEnd)};
        //
        // gmemB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK4;
        { tileOffsetK4 = loopOffset581; }
        //
        // smemB [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          smemBDstStack.mPipeline.producer_acquire(smemBProdState, smemBProdToken);
          if (((loopOffset581) + (int32_t{256})) < (loopEnd)) {
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
              reinterpret_cast<int8_t*>(smemBDstStack.mDepSmemPtr2) + (int32_t{49152});
            smemBytesStagePtrB = smemBytesBasePtrB + ((index) * (int32_t{16384}));
            int32_t coords[4];
            coords[int32_t{0}] = tileOffsetK8;
            coords[int32_t{1}] =
              (int32_t{trtllm::dev::getCtaRankInPair()}) * (int32_t{32}) +
              (((int32_t{64}) - ((mBatchLimit) % (int32_t{64}))) % (int32_t{64}));
            coords[int32_t{2}] = int32_t{0x40000000 /*1073741824*/};
            coords[int32_t{3}] =
              (((mCtaIdxY) * (int32_t{64})) +
               ((int32_t{0}) -
                (((int32_t{64}) - ((mBatchLimit) % (int32_t{64}))) % (int32_t{64})))) +
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
                &reinterpret_cast<cutlass::bfloat16_t*>(smemBytesStagePtrB)[int32_t{0}],
                params.tmaB,
                coords,
                leadCtaMbar,
                mMmaCtaMask);
            }
            coords[int32_t{0}] += int32_t{64};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::cp_async_bulk_tensor(
                cuda_ptx::space_cluster_t{},
                cuda_ptx::space_global_t{},
                cuda_ptx::cta_group_2_t{},
                &reinterpret_cast<cutlass::bfloat16_t*>(smemBytesStagePtrB)[int32_t{2048}],
                params.tmaB,
                coords,
                leadCtaMbar,
                mMmaCtaMask);
            }
            coords[int32_t{0}] += int32_t{64};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::cp_async_bulk_tensor(
                cuda_ptx::space_cluster_t{},
                cuda_ptx::space_global_t{},
                cuda_ptx::cta_group_2_t{},
                &reinterpret_cast<cutlass::bfloat16_t*>(smemBytesStagePtrB)[int32_t{4096}],
                params.tmaB,
                coords,
                leadCtaMbar,
                mMmaCtaMask);
            }
            coords[int32_t{0}] += int32_t{64};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::cp_async_bulk_tensor(
                cuda_ptx::space_cluster_t{},
                cuda_ptx::space_global_t{},
                cuda_ptx::cta_group_2_t{},
                &reinterpret_cast<cutlass::bfloat16_t*>(smemBytesStagePtrB)[int32_t{6144}],
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
        if ((loopOffset581) >= (int32_t{0})) {
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
    return ((state.mWarpIdx) >= (int32_t{10})) && ((state.mWarpIdx) < (int32_t{11}));
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
    trtllm::dev::CutlassTmaAsyncPipeline<3>::PipelineState smemSfAProdState{int32_t{0},
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
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemSfADstSmem.mArray[index][int32_t{0}],
                                             params.tmaSfA,
                                             coords,
                                             barrier);
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
      for (int32_t loopOffset727 = int32_t{256}; loopOffset727 < loopEnd;
           loopOffset727 += int32_t{256}) {
        bool const isFirstLoopIter{(loopOffset727) == (int32_t{256})};
        bool const isLastLoopIter{((loopOffset727) + (int32_t{256})) >= (loopEnd)};
        //
        // smemSfA [ProdPreCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset727) >= (int32_t{256})) {
        }
        //
        // smemSfA [ProdCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset727) >= (int32_t{256})) {
          {
            smemSfADstStack.mPipeline.producer_commit(smemSfAProdState);
          }
          ++smemSfAProdState;
        }
        //
        // gmemSfA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK5;
        { tileOffsetK5 = loopOffset727; }
        //
        // smemSfA [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          if ((loopOffset727) >= (int32_t{0})) {
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
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemSfADstSmem.mArray[index][int32_t{0}],
                                             params.tmaSfA,
                                             coords,
                                             barrier);
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
struct CastATask {
  int32_t mCtaOffsetK;
  int32_t mCtaIdxX;
  int32_t mCtaIdxY;
  int32_t mCtaIdxZ;
  int32_t const mWarpGrpThreadIdx;
  uint32_t const mTmemBaseOffset;
  inline __device__ CastATask(KernelParams const& params,
                              KernelState const& state,
                              int32_t warpGrpStart)
    : mCtaOffsetK{int32_t{0}}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)}
    , mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))}
    , mTmemBaseOffset{uint32_t{
        __shfl_sync(uint32_t{0xffffffff}, (*state.mTmemSwStatePtr), int32_t{0}, int32_t{32})}} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{4})) && ((state.mWarpIdx) < (int32_t{8}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 TmemCastAStack& tmemCastADstStack,
                                 SmemASmem& smemASrcSmem,
                                 SmemAStack& smemASrcStack,
                                 SmemSfASmem& smemSfASrcSmem,
                                 SmemSfAStack& smemSfASrcStack,
                                 WorkIdSmem& workIdSrcSmem,
                                 WorkIdStack& workIdSrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<112>{});
    trtllm::dev::CutlassTmaAsyncPipeline<3>::PipelineState smemAConsState{};
    trtllm::dev::CutlassTmaAsyncPipeline<3>::PipelineState smemAConsReleaseState{};
    int32_t smemAConsToken{int32_t{0}};
    trtllm::dev::CutlassTmaAsyncPipeline<3>::PipelineState smemSfAConsState{};
    trtllm::dev::CutlassTmaAsyncPipeline<3>::PipelineState smemSfAConsReleaseState{};
    int32_t smemSfAConsToken{int32_t{0}};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsState{};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsReleaseState{};
    int32_t workIdConsToken{int32_t{0}};
    trtllm::dev::CutlassUmmaConsumerAsyncPipeline<
      3,
      false,
      false,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState tmemCastAProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    int32_t tmemCastAProdToken{int32_t{1}};
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
      for (int32_t loopOffset843 = int32_t{0}; loopOffset843 < loopEnd;
           loopOffset843 += int32_t{256}) {
        bool const isFirstLoopIter{(loopOffset843) == (int32_t{0})};
        bool const isLastLoopIter{((loopOffset843) + (int32_t{256})) >= (loopEnd)};
        //
        // tmemCastA [ProdPreCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset843) >= (int32_t{256})) {
        }
        //
        // tmemCastA [ProdCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset843) >= (int32_t{256})) {
          {
            tmemCastADstStack.mPipeline.producer_commit(tmemCastAProdState);
          }
          ++tmemCastAProdState;
        }
        //
        // smemA [ConsRelease, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset843) >= (int32_t{256})) {
          trtllm::dev::CutlassNamedBarrier::sync(128, 1);
          { smemASrcStack.mPipeline.consumer_release(smemAConsReleaseState); }
          ++smemAConsReleaseState;
        }
        //
        // smemSfA [ConsRelease, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset843) >= (int32_t{256})) {
          trtllm::dev::CutlassNamedBarrier::sync(128, 3);
          { smemSfASrcStack.mPipeline.consumer_release(smemSfAConsReleaseState); }
          ++smemSfAConsReleaseState;
        }
        //
        // smemA [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { smemAConsToken = smemASrcStack.mPipeline.consumer_try_wait(smemAConsState); }
          smemASrcStack.mPipeline.consumer_wait(smemAConsState, smemAConsToken);
        }
        //
        // smemSfA [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { smemSfAConsToken = smemSfASrcStack.mPipeline.consumer_try_wait(smemSfAConsState); }
          smemSfASrcStack.mPipeline.consumer_wait(smemSfAConsState, smemSfAConsToken);
        }
        //
        // smemA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::int4b_t* smemPtrA7;
        {
          int32_t index{smemAConsState.index()};
          int8_t* smemBytesBasePtrA;
          smemBytesBasePtrA = reinterpret_cast<int8_t*>(smemASrcStack.mDepSmemPtr2) + (int32_t{0});
          int8_t* smemBytesStagePtrA;
          smemBytesStagePtrA = smemBytesBasePtrA + ((index) * (int32_t{16384}));
          smemPtrA7 = reinterpret_cast<cutlass::int4b_t*>(smemBytesStagePtrA) + (int32_t{0});
          ++smemAConsState;
        }
        //
        // smemSfA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::bfloat16_t* smemPtrSfA9;
        {
          int32_t index{smemSfAConsState.index()};
          smemPtrSfA9 = smemSfASrcStack.mPtr + ((index) * (int32_t{1024}));
          ++smemSfAConsState;
        }
        //
        // tmemCastA [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          if ((loopOffset843) >= (int32_t{0})) {
            tmemCastAProdToken =
              tmemCastADstStack.mPipeline.producer_try_acquire(tmemCastAProdState);
          }
        }
        { tmemCastADstStack.mPipeline.producer_acquire(tmemCastAProdState, tmemCastAProdToken); }
        //
        // tmemCastA [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::int4b_t* smemPtrA10;
        cutlass::bfloat16_t* smemPtrSfA10;
        smemPtrA10 = smemPtrA7;
        smemPtrSfA10 = smemPtrSfA9;
        {
          int32_t index{tmemCastAProdState.index()};
          {
            cutlass::Array<cutlass::int4b_t, 256> srcArray;
            cutlass::int4b_t* srcArrayPtr{srcArray.data()};
            cutlass::uint128_t* srcArrayPackedPtr;
            srcArrayPackedPtr = reinterpret_cast<cutlass::uint128_t*>(srcArrayPtr) + (int32_t{0});
            {
              //
              // Load elements k=0 to 31.
              //
              int32_t const smemRowIdx{((mWarpGrpThreadIdx) * (int32_t{256})) / (int32_t{256})};
              int32_t const smemOffsetInBytes{
                (((mWarpGrpThreadIdx) * (int32_t{256})) * (int32_t{4})) / (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              cutlass::uint128_t eltPack;
              eltPack = reinterpret_cast<cutlass::uint128_t*>(
                smemPtrA10)[((smemOffsetInBytes) ^ (swizzleMask)) / (int32_t{16})];
              srcArrayPackedPtr[int32_t{0}] = eltPack;
            }
            {
              //
              // Load elements k=32 to 63.
              //
              int32_t const smemRowIdx{((mWarpGrpThreadIdx) * (int32_t{256}) + (int32_t{32})) /
                                       (int32_t{256})};
              int32_t const smemOffsetInBytes{
                (((mWarpGrpThreadIdx) * (int32_t{256}) + (int32_t{32})) * (int32_t{4})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              cutlass::uint128_t eltPack;
              eltPack = reinterpret_cast<cutlass::uint128_t*>(
                smemPtrA10)[((smemOffsetInBytes) ^ (swizzleMask)) / (int32_t{16})];
              srcArrayPackedPtr[int32_t{1}] = eltPack;
            }
            {
              //
              // Load elements k=64 to 95.
              //
              int32_t const smemRowIdx{((mWarpGrpThreadIdx) * (int32_t{256}) + (int32_t{64})) /
                                       (int32_t{256})};
              int32_t const smemOffsetInBytes{
                (((mWarpGrpThreadIdx) * (int32_t{256}) + (int32_t{64})) * (int32_t{4})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              cutlass::uint128_t eltPack;
              eltPack = reinterpret_cast<cutlass::uint128_t*>(
                smemPtrA10)[((smemOffsetInBytes) ^ (swizzleMask)) / (int32_t{16})];
              srcArrayPackedPtr[int32_t{2}] = eltPack;
            }
            {
              //
              // Load elements k=96 to 127.
              //
              int32_t const smemRowIdx{((mWarpGrpThreadIdx) * (int32_t{256}) + (int32_t{96})) /
                                       (int32_t{256})};
              int32_t const smemOffsetInBytes{
                (((mWarpGrpThreadIdx) * (int32_t{256}) + (int32_t{96})) * (int32_t{4})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              cutlass::uint128_t eltPack;
              eltPack = reinterpret_cast<cutlass::uint128_t*>(
                smemPtrA10)[((smemOffsetInBytes) ^ (swizzleMask)) / (int32_t{16})];
              srcArrayPackedPtr[int32_t{3}] = eltPack;
            }
            {
              //
              // Load elements k=128 to 159.
              //
              int32_t const smemRowIdx{((mWarpGrpThreadIdx) * (int32_t{256}) + (int32_t{128})) /
                                       (int32_t{256})};
              int32_t const smemOffsetInBytes{
                (((mWarpGrpThreadIdx) * (int32_t{256}) + (int32_t{128})) * (int32_t{4})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              cutlass::uint128_t eltPack;
              eltPack = reinterpret_cast<cutlass::uint128_t*>(
                smemPtrA10)[((smemOffsetInBytes) ^ (swizzleMask)) / (int32_t{16})];
              srcArrayPackedPtr[int32_t{4}] = eltPack;
            }
            {
              //
              // Load elements k=160 to 191.
              //
              int32_t const smemRowIdx{((mWarpGrpThreadIdx) * (int32_t{256}) + (int32_t{160})) /
                                       (int32_t{256})};
              int32_t const smemOffsetInBytes{
                (((mWarpGrpThreadIdx) * (int32_t{256}) + (int32_t{160})) * (int32_t{4})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              cutlass::uint128_t eltPack;
              eltPack = reinterpret_cast<cutlass::uint128_t*>(
                smemPtrA10)[((smemOffsetInBytes) ^ (swizzleMask)) / (int32_t{16})];
              srcArrayPackedPtr[int32_t{5}] = eltPack;
            }
            {
              //
              // Load elements k=192 to 223.
              //
              int32_t const smemRowIdx{((mWarpGrpThreadIdx) * (int32_t{256}) + (int32_t{192})) /
                                       (int32_t{256})};
              int32_t const smemOffsetInBytes{
                (((mWarpGrpThreadIdx) * (int32_t{256}) + (int32_t{192})) * (int32_t{4})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              cutlass::uint128_t eltPack;
              eltPack = reinterpret_cast<cutlass::uint128_t*>(
                smemPtrA10)[((smemOffsetInBytes) ^ (swizzleMask)) / (int32_t{16})];
              srcArrayPackedPtr[int32_t{6}] = eltPack;
            }
            {
              //
              // Load elements k=224 to 255.
              //
              int32_t const smemRowIdx{((mWarpGrpThreadIdx) * (int32_t{256}) + (int32_t{224})) /
                                       (int32_t{256})};
              int32_t const smemOffsetInBytes{
                (((mWarpGrpThreadIdx) * (int32_t{256}) + (int32_t{224})) * (int32_t{4})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              cutlass::uint128_t eltPack;
              eltPack = reinterpret_cast<cutlass::uint128_t*>(
                smemPtrA10)[((smemOffsetInBytes) ^ (swizzleMask)) / (int32_t{16})];
              srcArrayPackedPtr[int32_t{7}] = eltPack;
            }
            cutlass::Array<cutlass::bfloat16_t, 8> srcSfArray;
            cutlass::bfloat16_t* srcSfArrayPtr{srcSfArray.data()};
            uint64_t* srcSfArrayPackedPtr;
            srcSfArrayPackedPtr = reinterpret_cast<uint64_t*>(srcSfArrayPtr) + (int32_t{0});
            {
              //
              // Load SFs k=0 to 3.
              //
              uint64_t sfPack;
              sfPack = reinterpret_cast<uint64_t*>(
                smemPtrSfA10)[((((mWarpGrpThreadIdx) / (int32_t{128})) * (int32_t{64})) *
                               (int32_t{128})) +
                              ((((mWarpGrpThreadIdx) % (int32_t{32})) * (int32_t{4})) +
                               (((mWarpGrpThreadIdx) % (int32_t{128})) / (int32_t{32})))];
              srcSfArrayPackedPtr[int32_t{0}] = sfPack;
            }
            {
              //
              // Load SFs k=4 to 7.
              //
              uint64_t sfPack;
              sfPack = reinterpret_cast<uint64_t*>(
                smemPtrSfA10)[(((((mWarpGrpThreadIdx) / (int32_t{128})) * (int32_t{64})) +
                                (int32_t{1})) *
                               (int32_t{128})) +
                              ((((mWarpGrpThreadIdx) % (int32_t{32})) * (int32_t{4})) +
                               (((mWarpGrpThreadIdx) % (int32_t{128})) / (int32_t{32})))];
              srcSfArrayPackedPtr[int32_t{1}] = sfPack;
            }
            cutlass::Array<uint32_t, 128> dstArrayPacked;
            cutlass::Array<cutlass::bfloat16_t, 256> tmpArrayBf16{
              trtllm::dev::castArray<cutlass::bfloat16_t, cutlass::int4b_t, 256>(srcArray)};
            cutlass::Array<cutlass::bfloat16_t, 8> tmpSfArrayBf16{srcSfArray};
            cutlass::bfloat16_t* tmpEltsBf16Ptr{tmpArrayBf16.data()};
            uint32_t* tmpEltsU32Ptr;
            tmpEltsU32Ptr = reinterpret_cast<uint32_t*>(tmpEltsBf16Ptr) + (int32_t{0});
            cutlass::bfloat16_t* tmpSfBf16Ptr{tmpSfArrayBf16.data()};
            uint32_t* tmpSfU32Ptr;
            tmpSfU32Ptr = reinterpret_cast<uint32_t*>(tmpSfBf16Ptr) + (int32_t{0});
            {
              //
              // Scale elements k=0 to 31.
              //
              uint32_t sfVec_H1_H0;
              uint32_t sfVec_H0_H0;
              sfVec_H1_H0 = tmpSfU32Ptr[int32_t{0}];
              trtllm::dev::prmt(sfVec_H0_H0, uint32_t{0}, sfVec_H1_H0, uint32_t{4112});
              uint32_t eltVec_0;
              eltVec_0 = tmpEltsU32Ptr[int32_t{0}];
              trtllm::dev::mulBf16x2(eltVec_0, sfVec_H0_H0);
              uint32_t scaledEltVec_0{trtllm::dev::mulBf16x2(eltVec_0, sfVec_H0_H0)};
              dstArrayPacked[int32_t{0}] = scaledEltVec_0;
              uint32_t eltVec_1;
              eltVec_1 = tmpEltsU32Ptr[int32_t{1}];
              trtllm::dev::mulBf16x2(eltVec_1, sfVec_H0_H0);
              uint32_t scaledEltVec_1{trtllm::dev::mulBf16x2(eltVec_1, sfVec_H0_H0)};
              dstArrayPacked[int32_t{1}] = scaledEltVec_1;
              uint32_t eltVec_2;
              eltVec_2 = tmpEltsU32Ptr[int32_t{2}];
              trtllm::dev::mulBf16x2(eltVec_2, sfVec_H0_H0);
              uint32_t scaledEltVec_2{trtllm::dev::mulBf16x2(eltVec_2, sfVec_H0_H0)};
              dstArrayPacked[int32_t{2}] = scaledEltVec_2;
              uint32_t eltVec_3;
              eltVec_3 = tmpEltsU32Ptr[int32_t{3}];
              trtllm::dev::mulBf16x2(eltVec_3, sfVec_H0_H0);
              uint32_t scaledEltVec_3{trtllm::dev::mulBf16x2(eltVec_3, sfVec_H0_H0)};
              dstArrayPacked[int32_t{3}] = scaledEltVec_3;
              uint32_t eltVec_4;
              eltVec_4 = tmpEltsU32Ptr[int32_t{4}];
              trtllm::dev::mulBf16x2(eltVec_4, sfVec_H0_H0);
              uint32_t scaledEltVec_4{trtllm::dev::mulBf16x2(eltVec_4, sfVec_H0_H0)};
              dstArrayPacked[int32_t{4}] = scaledEltVec_4;
              uint32_t eltVec_5;
              eltVec_5 = tmpEltsU32Ptr[int32_t{5}];
              trtllm::dev::mulBf16x2(eltVec_5, sfVec_H0_H0);
              uint32_t scaledEltVec_5{trtllm::dev::mulBf16x2(eltVec_5, sfVec_H0_H0)};
              dstArrayPacked[int32_t{5}] = scaledEltVec_5;
              uint32_t eltVec_6;
              eltVec_6 = tmpEltsU32Ptr[int32_t{6}];
              trtllm::dev::mulBf16x2(eltVec_6, sfVec_H0_H0);
              uint32_t scaledEltVec_6{trtllm::dev::mulBf16x2(eltVec_6, sfVec_H0_H0)};
              dstArrayPacked[int32_t{6}] = scaledEltVec_6;
              uint32_t eltVec_7;
              eltVec_7 = tmpEltsU32Ptr[int32_t{7}];
              trtllm::dev::mulBf16x2(eltVec_7, sfVec_H0_H0);
              uint32_t scaledEltVec_7{trtllm::dev::mulBf16x2(eltVec_7, sfVec_H0_H0)};
              dstArrayPacked[int32_t{7}] = scaledEltVec_7;
              uint32_t eltVec_8;
              eltVec_8 = tmpEltsU32Ptr[int32_t{8}];
              trtllm::dev::mulBf16x2(eltVec_8, sfVec_H0_H0);
              uint32_t scaledEltVec_8{trtllm::dev::mulBf16x2(eltVec_8, sfVec_H0_H0)};
              dstArrayPacked[int32_t{8}] = scaledEltVec_8;
              uint32_t eltVec_9;
              eltVec_9 = tmpEltsU32Ptr[int32_t{9}];
              trtllm::dev::mulBf16x2(eltVec_9, sfVec_H0_H0);
              uint32_t scaledEltVec_9{trtllm::dev::mulBf16x2(eltVec_9, sfVec_H0_H0)};
              dstArrayPacked[int32_t{9}] = scaledEltVec_9;
              uint32_t eltVec_10;
              eltVec_10 = tmpEltsU32Ptr[int32_t{10}];
              trtllm::dev::mulBf16x2(eltVec_10, sfVec_H0_H0);
              uint32_t scaledEltVec_10{trtllm::dev::mulBf16x2(eltVec_10, sfVec_H0_H0)};
              dstArrayPacked[int32_t{10}] = scaledEltVec_10;
              uint32_t eltVec_11;
              eltVec_11 = tmpEltsU32Ptr[int32_t{11}];
              trtllm::dev::mulBf16x2(eltVec_11, sfVec_H0_H0);
              uint32_t scaledEltVec_11{trtllm::dev::mulBf16x2(eltVec_11, sfVec_H0_H0)};
              dstArrayPacked[int32_t{11}] = scaledEltVec_11;
              uint32_t eltVec_12;
              eltVec_12 = tmpEltsU32Ptr[int32_t{12}];
              trtllm::dev::mulBf16x2(eltVec_12, sfVec_H0_H0);
              uint32_t scaledEltVec_12{trtllm::dev::mulBf16x2(eltVec_12, sfVec_H0_H0)};
              dstArrayPacked[int32_t{12}] = scaledEltVec_12;
              uint32_t eltVec_13;
              eltVec_13 = tmpEltsU32Ptr[int32_t{13}];
              trtllm::dev::mulBf16x2(eltVec_13, sfVec_H0_H0);
              uint32_t scaledEltVec_13{trtllm::dev::mulBf16x2(eltVec_13, sfVec_H0_H0)};
              dstArrayPacked[int32_t{13}] = scaledEltVec_13;
              uint32_t eltVec_14;
              eltVec_14 = tmpEltsU32Ptr[int32_t{14}];
              trtllm::dev::mulBf16x2(eltVec_14, sfVec_H0_H0);
              uint32_t scaledEltVec_14{trtllm::dev::mulBf16x2(eltVec_14, sfVec_H0_H0)};
              dstArrayPacked[int32_t{14}] = scaledEltVec_14;
              uint32_t eltVec_15;
              eltVec_15 = tmpEltsU32Ptr[int32_t{15}];
              trtllm::dev::mulBf16x2(eltVec_15, sfVec_H0_H0);
              uint32_t scaledEltVec_15{trtllm::dev::mulBf16x2(eltVec_15, sfVec_H0_H0)};
              dstArrayPacked[int32_t{15}] = scaledEltVec_15;
            }
            {
              //
              // Scale elements k=32 to 63.
              //
              uint32_t sfVec_H1_H0;
              uint32_t sfVec_H1_H1;
              sfVec_H1_H0 = tmpSfU32Ptr[int32_t{0}];
              trtllm::dev::prmt(sfVec_H1_H1, uint32_t{0}, sfVec_H1_H0, uint32_t{12850});
              uint32_t eltVec_0;
              eltVec_0 = tmpEltsU32Ptr[int32_t{16}];
              trtllm::dev::mulBf16x2(eltVec_0, sfVec_H1_H1);
              uint32_t scaledEltVec_0{trtllm::dev::mulBf16x2(eltVec_0, sfVec_H1_H1)};
              dstArrayPacked[int32_t{16}] = scaledEltVec_0;
              uint32_t eltVec_1;
              eltVec_1 = tmpEltsU32Ptr[int32_t{17}];
              trtllm::dev::mulBf16x2(eltVec_1, sfVec_H1_H1);
              uint32_t scaledEltVec_1{trtllm::dev::mulBf16x2(eltVec_1, sfVec_H1_H1)};
              dstArrayPacked[int32_t{17}] = scaledEltVec_1;
              uint32_t eltVec_2;
              eltVec_2 = tmpEltsU32Ptr[int32_t{18}];
              trtllm::dev::mulBf16x2(eltVec_2, sfVec_H1_H1);
              uint32_t scaledEltVec_2{trtllm::dev::mulBf16x2(eltVec_2, sfVec_H1_H1)};
              dstArrayPacked[int32_t{18}] = scaledEltVec_2;
              uint32_t eltVec_3;
              eltVec_3 = tmpEltsU32Ptr[int32_t{19}];
              trtllm::dev::mulBf16x2(eltVec_3, sfVec_H1_H1);
              uint32_t scaledEltVec_3{trtllm::dev::mulBf16x2(eltVec_3, sfVec_H1_H1)};
              dstArrayPacked[int32_t{19}] = scaledEltVec_3;
              uint32_t eltVec_4;
              eltVec_4 = tmpEltsU32Ptr[int32_t{20}];
              trtllm::dev::mulBf16x2(eltVec_4, sfVec_H1_H1);
              uint32_t scaledEltVec_4{trtllm::dev::mulBf16x2(eltVec_4, sfVec_H1_H1)};
              dstArrayPacked[int32_t{20}] = scaledEltVec_4;
              uint32_t eltVec_5;
              eltVec_5 = tmpEltsU32Ptr[int32_t{21}];
              trtllm::dev::mulBf16x2(eltVec_5, sfVec_H1_H1);
              uint32_t scaledEltVec_5{trtllm::dev::mulBf16x2(eltVec_5, sfVec_H1_H1)};
              dstArrayPacked[int32_t{21}] = scaledEltVec_5;
              uint32_t eltVec_6;
              eltVec_6 = tmpEltsU32Ptr[int32_t{22}];
              trtllm::dev::mulBf16x2(eltVec_6, sfVec_H1_H1);
              uint32_t scaledEltVec_6{trtllm::dev::mulBf16x2(eltVec_6, sfVec_H1_H1)};
              dstArrayPacked[int32_t{22}] = scaledEltVec_6;
              uint32_t eltVec_7;
              eltVec_7 = tmpEltsU32Ptr[int32_t{23}];
              trtllm::dev::mulBf16x2(eltVec_7, sfVec_H1_H1);
              uint32_t scaledEltVec_7{trtllm::dev::mulBf16x2(eltVec_7, sfVec_H1_H1)};
              dstArrayPacked[int32_t{23}] = scaledEltVec_7;
              uint32_t eltVec_8;
              eltVec_8 = tmpEltsU32Ptr[int32_t{24}];
              trtllm::dev::mulBf16x2(eltVec_8, sfVec_H1_H1);
              uint32_t scaledEltVec_8{trtllm::dev::mulBf16x2(eltVec_8, sfVec_H1_H1)};
              dstArrayPacked[int32_t{24}] = scaledEltVec_8;
              uint32_t eltVec_9;
              eltVec_9 = tmpEltsU32Ptr[int32_t{25}];
              trtllm::dev::mulBf16x2(eltVec_9, sfVec_H1_H1);
              uint32_t scaledEltVec_9{trtllm::dev::mulBf16x2(eltVec_9, sfVec_H1_H1)};
              dstArrayPacked[int32_t{25}] = scaledEltVec_9;
              uint32_t eltVec_10;
              eltVec_10 = tmpEltsU32Ptr[int32_t{26}];
              trtllm::dev::mulBf16x2(eltVec_10, sfVec_H1_H1);
              uint32_t scaledEltVec_10{trtllm::dev::mulBf16x2(eltVec_10, sfVec_H1_H1)};
              dstArrayPacked[int32_t{26}] = scaledEltVec_10;
              uint32_t eltVec_11;
              eltVec_11 = tmpEltsU32Ptr[int32_t{27}];
              trtllm::dev::mulBf16x2(eltVec_11, sfVec_H1_H1);
              uint32_t scaledEltVec_11{trtllm::dev::mulBf16x2(eltVec_11, sfVec_H1_H1)};
              dstArrayPacked[int32_t{27}] = scaledEltVec_11;
              uint32_t eltVec_12;
              eltVec_12 = tmpEltsU32Ptr[int32_t{28}];
              trtllm::dev::mulBf16x2(eltVec_12, sfVec_H1_H1);
              uint32_t scaledEltVec_12{trtllm::dev::mulBf16x2(eltVec_12, sfVec_H1_H1)};
              dstArrayPacked[int32_t{28}] = scaledEltVec_12;
              uint32_t eltVec_13;
              eltVec_13 = tmpEltsU32Ptr[int32_t{29}];
              trtllm::dev::mulBf16x2(eltVec_13, sfVec_H1_H1);
              uint32_t scaledEltVec_13{trtllm::dev::mulBf16x2(eltVec_13, sfVec_H1_H1)};
              dstArrayPacked[int32_t{29}] = scaledEltVec_13;
              uint32_t eltVec_14;
              eltVec_14 = tmpEltsU32Ptr[int32_t{30}];
              trtllm::dev::mulBf16x2(eltVec_14, sfVec_H1_H1);
              uint32_t scaledEltVec_14{trtllm::dev::mulBf16x2(eltVec_14, sfVec_H1_H1)};
              dstArrayPacked[int32_t{30}] = scaledEltVec_14;
              uint32_t eltVec_15;
              eltVec_15 = tmpEltsU32Ptr[int32_t{31}];
              trtllm::dev::mulBf16x2(eltVec_15, sfVec_H1_H1);
              uint32_t scaledEltVec_15{trtllm::dev::mulBf16x2(eltVec_15, sfVec_H1_H1)};
              dstArrayPacked[int32_t{31}] = scaledEltVec_15;
            }
            {
              //
              // Scale elements k=64 to 95.
              //
              uint32_t sfVec_H1_H0;
              uint32_t sfVec_H0_H0;
              sfVec_H1_H0 = tmpSfU32Ptr[int32_t{1}];
              trtllm::dev::prmt(sfVec_H0_H0, uint32_t{0}, sfVec_H1_H0, uint32_t{4112});
              uint32_t eltVec_0;
              eltVec_0 = tmpEltsU32Ptr[int32_t{32}];
              trtllm::dev::mulBf16x2(eltVec_0, sfVec_H0_H0);
              uint32_t scaledEltVec_0{trtllm::dev::mulBf16x2(eltVec_0, sfVec_H0_H0)};
              dstArrayPacked[int32_t{32}] = scaledEltVec_0;
              uint32_t eltVec_1;
              eltVec_1 = tmpEltsU32Ptr[int32_t{33}];
              trtllm::dev::mulBf16x2(eltVec_1, sfVec_H0_H0);
              uint32_t scaledEltVec_1{trtllm::dev::mulBf16x2(eltVec_1, sfVec_H0_H0)};
              dstArrayPacked[int32_t{33}] = scaledEltVec_1;
              uint32_t eltVec_2;
              eltVec_2 = tmpEltsU32Ptr[int32_t{34}];
              trtllm::dev::mulBf16x2(eltVec_2, sfVec_H0_H0);
              uint32_t scaledEltVec_2{trtllm::dev::mulBf16x2(eltVec_2, sfVec_H0_H0)};
              dstArrayPacked[int32_t{34}] = scaledEltVec_2;
              uint32_t eltVec_3;
              eltVec_3 = tmpEltsU32Ptr[int32_t{35}];
              trtllm::dev::mulBf16x2(eltVec_3, sfVec_H0_H0);
              uint32_t scaledEltVec_3{trtllm::dev::mulBf16x2(eltVec_3, sfVec_H0_H0)};
              dstArrayPacked[int32_t{35}] = scaledEltVec_3;
              uint32_t eltVec_4;
              eltVec_4 = tmpEltsU32Ptr[int32_t{36}];
              trtllm::dev::mulBf16x2(eltVec_4, sfVec_H0_H0);
              uint32_t scaledEltVec_4{trtllm::dev::mulBf16x2(eltVec_4, sfVec_H0_H0)};
              dstArrayPacked[int32_t{36}] = scaledEltVec_4;
              uint32_t eltVec_5;
              eltVec_5 = tmpEltsU32Ptr[int32_t{37}];
              trtllm::dev::mulBf16x2(eltVec_5, sfVec_H0_H0);
              uint32_t scaledEltVec_5{trtllm::dev::mulBf16x2(eltVec_5, sfVec_H0_H0)};
              dstArrayPacked[int32_t{37}] = scaledEltVec_5;
              uint32_t eltVec_6;
              eltVec_6 = tmpEltsU32Ptr[int32_t{38}];
              trtllm::dev::mulBf16x2(eltVec_6, sfVec_H0_H0);
              uint32_t scaledEltVec_6{trtllm::dev::mulBf16x2(eltVec_6, sfVec_H0_H0)};
              dstArrayPacked[int32_t{38}] = scaledEltVec_6;
              uint32_t eltVec_7;
              eltVec_7 = tmpEltsU32Ptr[int32_t{39}];
              trtllm::dev::mulBf16x2(eltVec_7, sfVec_H0_H0);
              uint32_t scaledEltVec_7{trtllm::dev::mulBf16x2(eltVec_7, sfVec_H0_H0)};
              dstArrayPacked[int32_t{39}] = scaledEltVec_7;
              uint32_t eltVec_8;
              eltVec_8 = tmpEltsU32Ptr[int32_t{40}];
              trtllm::dev::mulBf16x2(eltVec_8, sfVec_H0_H0);
              uint32_t scaledEltVec_8{trtllm::dev::mulBf16x2(eltVec_8, sfVec_H0_H0)};
              dstArrayPacked[int32_t{40}] = scaledEltVec_8;
              uint32_t eltVec_9;
              eltVec_9 = tmpEltsU32Ptr[int32_t{41}];
              trtllm::dev::mulBf16x2(eltVec_9, sfVec_H0_H0);
              uint32_t scaledEltVec_9{trtllm::dev::mulBf16x2(eltVec_9, sfVec_H0_H0)};
              dstArrayPacked[int32_t{41}] = scaledEltVec_9;
              uint32_t eltVec_10;
              eltVec_10 = tmpEltsU32Ptr[int32_t{42}];
              trtllm::dev::mulBf16x2(eltVec_10, sfVec_H0_H0);
              uint32_t scaledEltVec_10{trtllm::dev::mulBf16x2(eltVec_10, sfVec_H0_H0)};
              dstArrayPacked[int32_t{42}] = scaledEltVec_10;
              uint32_t eltVec_11;
              eltVec_11 = tmpEltsU32Ptr[int32_t{43}];
              trtllm::dev::mulBf16x2(eltVec_11, sfVec_H0_H0);
              uint32_t scaledEltVec_11{trtllm::dev::mulBf16x2(eltVec_11, sfVec_H0_H0)};
              dstArrayPacked[int32_t{43}] = scaledEltVec_11;
              uint32_t eltVec_12;
              eltVec_12 = tmpEltsU32Ptr[int32_t{44}];
              trtllm::dev::mulBf16x2(eltVec_12, sfVec_H0_H0);
              uint32_t scaledEltVec_12{trtllm::dev::mulBf16x2(eltVec_12, sfVec_H0_H0)};
              dstArrayPacked[int32_t{44}] = scaledEltVec_12;
              uint32_t eltVec_13;
              eltVec_13 = tmpEltsU32Ptr[int32_t{45}];
              trtllm::dev::mulBf16x2(eltVec_13, sfVec_H0_H0);
              uint32_t scaledEltVec_13{trtllm::dev::mulBf16x2(eltVec_13, sfVec_H0_H0)};
              dstArrayPacked[int32_t{45}] = scaledEltVec_13;
              uint32_t eltVec_14;
              eltVec_14 = tmpEltsU32Ptr[int32_t{46}];
              trtllm::dev::mulBf16x2(eltVec_14, sfVec_H0_H0);
              uint32_t scaledEltVec_14{trtllm::dev::mulBf16x2(eltVec_14, sfVec_H0_H0)};
              dstArrayPacked[int32_t{46}] = scaledEltVec_14;
              uint32_t eltVec_15;
              eltVec_15 = tmpEltsU32Ptr[int32_t{47}];
              trtllm::dev::mulBf16x2(eltVec_15, sfVec_H0_H0);
              uint32_t scaledEltVec_15{trtllm::dev::mulBf16x2(eltVec_15, sfVec_H0_H0)};
              dstArrayPacked[int32_t{47}] = scaledEltVec_15;
            }
            {
              //
              // Scale elements k=96 to 127.
              //
              uint32_t sfVec_H1_H0;
              uint32_t sfVec_H1_H1;
              sfVec_H1_H0 = tmpSfU32Ptr[int32_t{1}];
              trtllm::dev::prmt(sfVec_H1_H1, uint32_t{0}, sfVec_H1_H0, uint32_t{12850});
              uint32_t eltVec_0;
              eltVec_0 = tmpEltsU32Ptr[int32_t{48}];
              trtllm::dev::mulBf16x2(eltVec_0, sfVec_H1_H1);
              uint32_t scaledEltVec_0{trtllm::dev::mulBf16x2(eltVec_0, sfVec_H1_H1)};
              dstArrayPacked[int32_t{48}] = scaledEltVec_0;
              uint32_t eltVec_1;
              eltVec_1 = tmpEltsU32Ptr[int32_t{49}];
              trtllm::dev::mulBf16x2(eltVec_1, sfVec_H1_H1);
              uint32_t scaledEltVec_1{trtllm::dev::mulBf16x2(eltVec_1, sfVec_H1_H1)};
              dstArrayPacked[int32_t{49}] = scaledEltVec_1;
              uint32_t eltVec_2;
              eltVec_2 = tmpEltsU32Ptr[int32_t{50}];
              trtllm::dev::mulBf16x2(eltVec_2, sfVec_H1_H1);
              uint32_t scaledEltVec_2{trtllm::dev::mulBf16x2(eltVec_2, sfVec_H1_H1)};
              dstArrayPacked[int32_t{50}] = scaledEltVec_2;
              uint32_t eltVec_3;
              eltVec_3 = tmpEltsU32Ptr[int32_t{51}];
              trtllm::dev::mulBf16x2(eltVec_3, sfVec_H1_H1);
              uint32_t scaledEltVec_3{trtllm::dev::mulBf16x2(eltVec_3, sfVec_H1_H1)};
              dstArrayPacked[int32_t{51}] = scaledEltVec_3;
              uint32_t eltVec_4;
              eltVec_4 = tmpEltsU32Ptr[int32_t{52}];
              trtllm::dev::mulBf16x2(eltVec_4, sfVec_H1_H1);
              uint32_t scaledEltVec_4{trtllm::dev::mulBf16x2(eltVec_4, sfVec_H1_H1)};
              dstArrayPacked[int32_t{52}] = scaledEltVec_4;
              uint32_t eltVec_5;
              eltVec_5 = tmpEltsU32Ptr[int32_t{53}];
              trtllm::dev::mulBf16x2(eltVec_5, sfVec_H1_H1);
              uint32_t scaledEltVec_5{trtllm::dev::mulBf16x2(eltVec_5, sfVec_H1_H1)};
              dstArrayPacked[int32_t{53}] = scaledEltVec_5;
              uint32_t eltVec_6;
              eltVec_6 = tmpEltsU32Ptr[int32_t{54}];
              trtllm::dev::mulBf16x2(eltVec_6, sfVec_H1_H1);
              uint32_t scaledEltVec_6{trtllm::dev::mulBf16x2(eltVec_6, sfVec_H1_H1)};
              dstArrayPacked[int32_t{54}] = scaledEltVec_6;
              uint32_t eltVec_7;
              eltVec_7 = tmpEltsU32Ptr[int32_t{55}];
              trtllm::dev::mulBf16x2(eltVec_7, sfVec_H1_H1);
              uint32_t scaledEltVec_7{trtllm::dev::mulBf16x2(eltVec_7, sfVec_H1_H1)};
              dstArrayPacked[int32_t{55}] = scaledEltVec_7;
              uint32_t eltVec_8;
              eltVec_8 = tmpEltsU32Ptr[int32_t{56}];
              trtllm::dev::mulBf16x2(eltVec_8, sfVec_H1_H1);
              uint32_t scaledEltVec_8{trtllm::dev::mulBf16x2(eltVec_8, sfVec_H1_H1)};
              dstArrayPacked[int32_t{56}] = scaledEltVec_8;
              uint32_t eltVec_9;
              eltVec_9 = tmpEltsU32Ptr[int32_t{57}];
              trtllm::dev::mulBf16x2(eltVec_9, sfVec_H1_H1);
              uint32_t scaledEltVec_9{trtllm::dev::mulBf16x2(eltVec_9, sfVec_H1_H1)};
              dstArrayPacked[int32_t{57}] = scaledEltVec_9;
              uint32_t eltVec_10;
              eltVec_10 = tmpEltsU32Ptr[int32_t{58}];
              trtllm::dev::mulBf16x2(eltVec_10, sfVec_H1_H1);
              uint32_t scaledEltVec_10{trtllm::dev::mulBf16x2(eltVec_10, sfVec_H1_H1)};
              dstArrayPacked[int32_t{58}] = scaledEltVec_10;
              uint32_t eltVec_11;
              eltVec_11 = tmpEltsU32Ptr[int32_t{59}];
              trtllm::dev::mulBf16x2(eltVec_11, sfVec_H1_H1);
              uint32_t scaledEltVec_11{trtllm::dev::mulBf16x2(eltVec_11, sfVec_H1_H1)};
              dstArrayPacked[int32_t{59}] = scaledEltVec_11;
              uint32_t eltVec_12;
              eltVec_12 = tmpEltsU32Ptr[int32_t{60}];
              trtllm::dev::mulBf16x2(eltVec_12, sfVec_H1_H1);
              uint32_t scaledEltVec_12{trtllm::dev::mulBf16x2(eltVec_12, sfVec_H1_H1)};
              dstArrayPacked[int32_t{60}] = scaledEltVec_12;
              uint32_t eltVec_13;
              eltVec_13 = tmpEltsU32Ptr[int32_t{61}];
              trtllm::dev::mulBf16x2(eltVec_13, sfVec_H1_H1);
              uint32_t scaledEltVec_13{trtllm::dev::mulBf16x2(eltVec_13, sfVec_H1_H1)};
              dstArrayPacked[int32_t{61}] = scaledEltVec_13;
              uint32_t eltVec_14;
              eltVec_14 = tmpEltsU32Ptr[int32_t{62}];
              trtllm::dev::mulBf16x2(eltVec_14, sfVec_H1_H1);
              uint32_t scaledEltVec_14{trtllm::dev::mulBf16x2(eltVec_14, sfVec_H1_H1)};
              dstArrayPacked[int32_t{62}] = scaledEltVec_14;
              uint32_t eltVec_15;
              eltVec_15 = tmpEltsU32Ptr[int32_t{63}];
              trtllm::dev::mulBf16x2(eltVec_15, sfVec_H1_H1);
              uint32_t scaledEltVec_15{trtllm::dev::mulBf16x2(eltVec_15, sfVec_H1_H1)};
              dstArrayPacked[int32_t{63}] = scaledEltVec_15;
            }
            {
              //
              // Scale elements k=128 to 159.
              //
              uint32_t sfVec_H1_H0;
              uint32_t sfVec_H0_H0;
              sfVec_H1_H0 = tmpSfU32Ptr[int32_t{2}];
              trtllm::dev::prmt(sfVec_H0_H0, uint32_t{0}, sfVec_H1_H0, uint32_t{4112});
              uint32_t eltVec_0;
              eltVec_0 = tmpEltsU32Ptr[int32_t{64}];
              trtllm::dev::mulBf16x2(eltVec_0, sfVec_H0_H0);
              uint32_t scaledEltVec_0{trtllm::dev::mulBf16x2(eltVec_0, sfVec_H0_H0)};
              dstArrayPacked[int32_t{64}] = scaledEltVec_0;
              uint32_t eltVec_1;
              eltVec_1 = tmpEltsU32Ptr[int32_t{65}];
              trtllm::dev::mulBf16x2(eltVec_1, sfVec_H0_H0);
              uint32_t scaledEltVec_1{trtllm::dev::mulBf16x2(eltVec_1, sfVec_H0_H0)};
              dstArrayPacked[int32_t{65}] = scaledEltVec_1;
              uint32_t eltVec_2;
              eltVec_2 = tmpEltsU32Ptr[int32_t{66}];
              trtllm::dev::mulBf16x2(eltVec_2, sfVec_H0_H0);
              uint32_t scaledEltVec_2{trtllm::dev::mulBf16x2(eltVec_2, sfVec_H0_H0)};
              dstArrayPacked[int32_t{66}] = scaledEltVec_2;
              uint32_t eltVec_3;
              eltVec_3 = tmpEltsU32Ptr[int32_t{67}];
              trtllm::dev::mulBf16x2(eltVec_3, sfVec_H0_H0);
              uint32_t scaledEltVec_3{trtllm::dev::mulBf16x2(eltVec_3, sfVec_H0_H0)};
              dstArrayPacked[int32_t{67}] = scaledEltVec_3;
              uint32_t eltVec_4;
              eltVec_4 = tmpEltsU32Ptr[int32_t{68}];
              trtllm::dev::mulBf16x2(eltVec_4, sfVec_H0_H0);
              uint32_t scaledEltVec_4{trtllm::dev::mulBf16x2(eltVec_4, sfVec_H0_H0)};
              dstArrayPacked[int32_t{68}] = scaledEltVec_4;
              uint32_t eltVec_5;
              eltVec_5 = tmpEltsU32Ptr[int32_t{69}];
              trtllm::dev::mulBf16x2(eltVec_5, sfVec_H0_H0);
              uint32_t scaledEltVec_5{trtllm::dev::mulBf16x2(eltVec_5, sfVec_H0_H0)};
              dstArrayPacked[int32_t{69}] = scaledEltVec_5;
              uint32_t eltVec_6;
              eltVec_6 = tmpEltsU32Ptr[int32_t{70}];
              trtllm::dev::mulBf16x2(eltVec_6, sfVec_H0_H0);
              uint32_t scaledEltVec_6{trtllm::dev::mulBf16x2(eltVec_6, sfVec_H0_H0)};
              dstArrayPacked[int32_t{70}] = scaledEltVec_6;
              uint32_t eltVec_7;
              eltVec_7 = tmpEltsU32Ptr[int32_t{71}];
              trtllm::dev::mulBf16x2(eltVec_7, sfVec_H0_H0);
              uint32_t scaledEltVec_7{trtllm::dev::mulBf16x2(eltVec_7, sfVec_H0_H0)};
              dstArrayPacked[int32_t{71}] = scaledEltVec_7;
              uint32_t eltVec_8;
              eltVec_8 = tmpEltsU32Ptr[int32_t{72}];
              trtllm::dev::mulBf16x2(eltVec_8, sfVec_H0_H0);
              uint32_t scaledEltVec_8{trtllm::dev::mulBf16x2(eltVec_8, sfVec_H0_H0)};
              dstArrayPacked[int32_t{72}] = scaledEltVec_8;
              uint32_t eltVec_9;
              eltVec_9 = tmpEltsU32Ptr[int32_t{73}];
              trtllm::dev::mulBf16x2(eltVec_9, sfVec_H0_H0);
              uint32_t scaledEltVec_9{trtllm::dev::mulBf16x2(eltVec_9, sfVec_H0_H0)};
              dstArrayPacked[int32_t{73}] = scaledEltVec_9;
              uint32_t eltVec_10;
              eltVec_10 = tmpEltsU32Ptr[int32_t{74}];
              trtllm::dev::mulBf16x2(eltVec_10, sfVec_H0_H0);
              uint32_t scaledEltVec_10{trtllm::dev::mulBf16x2(eltVec_10, sfVec_H0_H0)};
              dstArrayPacked[int32_t{74}] = scaledEltVec_10;
              uint32_t eltVec_11;
              eltVec_11 = tmpEltsU32Ptr[int32_t{75}];
              trtllm::dev::mulBf16x2(eltVec_11, sfVec_H0_H0);
              uint32_t scaledEltVec_11{trtllm::dev::mulBf16x2(eltVec_11, sfVec_H0_H0)};
              dstArrayPacked[int32_t{75}] = scaledEltVec_11;
              uint32_t eltVec_12;
              eltVec_12 = tmpEltsU32Ptr[int32_t{76}];
              trtllm::dev::mulBf16x2(eltVec_12, sfVec_H0_H0);
              uint32_t scaledEltVec_12{trtllm::dev::mulBf16x2(eltVec_12, sfVec_H0_H0)};
              dstArrayPacked[int32_t{76}] = scaledEltVec_12;
              uint32_t eltVec_13;
              eltVec_13 = tmpEltsU32Ptr[int32_t{77}];
              trtllm::dev::mulBf16x2(eltVec_13, sfVec_H0_H0);
              uint32_t scaledEltVec_13{trtllm::dev::mulBf16x2(eltVec_13, sfVec_H0_H0)};
              dstArrayPacked[int32_t{77}] = scaledEltVec_13;
              uint32_t eltVec_14;
              eltVec_14 = tmpEltsU32Ptr[int32_t{78}];
              trtllm::dev::mulBf16x2(eltVec_14, sfVec_H0_H0);
              uint32_t scaledEltVec_14{trtllm::dev::mulBf16x2(eltVec_14, sfVec_H0_H0)};
              dstArrayPacked[int32_t{78}] = scaledEltVec_14;
              uint32_t eltVec_15;
              eltVec_15 = tmpEltsU32Ptr[int32_t{79}];
              trtllm::dev::mulBf16x2(eltVec_15, sfVec_H0_H0);
              uint32_t scaledEltVec_15{trtllm::dev::mulBf16x2(eltVec_15, sfVec_H0_H0)};
              dstArrayPacked[int32_t{79}] = scaledEltVec_15;
            }
            {
              //
              // Scale elements k=160 to 191.
              //
              uint32_t sfVec_H1_H0;
              uint32_t sfVec_H1_H1;
              sfVec_H1_H0 = tmpSfU32Ptr[int32_t{2}];
              trtllm::dev::prmt(sfVec_H1_H1, uint32_t{0}, sfVec_H1_H0, uint32_t{12850});
              uint32_t eltVec_0;
              eltVec_0 = tmpEltsU32Ptr[int32_t{80}];
              trtllm::dev::mulBf16x2(eltVec_0, sfVec_H1_H1);
              uint32_t scaledEltVec_0{trtllm::dev::mulBf16x2(eltVec_0, sfVec_H1_H1)};
              dstArrayPacked[int32_t{80}] = scaledEltVec_0;
              uint32_t eltVec_1;
              eltVec_1 = tmpEltsU32Ptr[int32_t{81}];
              trtllm::dev::mulBf16x2(eltVec_1, sfVec_H1_H1);
              uint32_t scaledEltVec_1{trtllm::dev::mulBf16x2(eltVec_1, sfVec_H1_H1)};
              dstArrayPacked[int32_t{81}] = scaledEltVec_1;
              uint32_t eltVec_2;
              eltVec_2 = tmpEltsU32Ptr[int32_t{82}];
              trtllm::dev::mulBf16x2(eltVec_2, sfVec_H1_H1);
              uint32_t scaledEltVec_2{trtllm::dev::mulBf16x2(eltVec_2, sfVec_H1_H1)};
              dstArrayPacked[int32_t{82}] = scaledEltVec_2;
              uint32_t eltVec_3;
              eltVec_3 = tmpEltsU32Ptr[int32_t{83}];
              trtllm::dev::mulBf16x2(eltVec_3, sfVec_H1_H1);
              uint32_t scaledEltVec_3{trtllm::dev::mulBf16x2(eltVec_3, sfVec_H1_H1)};
              dstArrayPacked[int32_t{83}] = scaledEltVec_3;
              uint32_t eltVec_4;
              eltVec_4 = tmpEltsU32Ptr[int32_t{84}];
              trtllm::dev::mulBf16x2(eltVec_4, sfVec_H1_H1);
              uint32_t scaledEltVec_4{trtllm::dev::mulBf16x2(eltVec_4, sfVec_H1_H1)};
              dstArrayPacked[int32_t{84}] = scaledEltVec_4;
              uint32_t eltVec_5;
              eltVec_5 = tmpEltsU32Ptr[int32_t{85}];
              trtllm::dev::mulBf16x2(eltVec_5, sfVec_H1_H1);
              uint32_t scaledEltVec_5{trtllm::dev::mulBf16x2(eltVec_5, sfVec_H1_H1)};
              dstArrayPacked[int32_t{85}] = scaledEltVec_5;
              uint32_t eltVec_6;
              eltVec_6 = tmpEltsU32Ptr[int32_t{86}];
              trtllm::dev::mulBf16x2(eltVec_6, sfVec_H1_H1);
              uint32_t scaledEltVec_6{trtllm::dev::mulBf16x2(eltVec_6, sfVec_H1_H1)};
              dstArrayPacked[int32_t{86}] = scaledEltVec_6;
              uint32_t eltVec_7;
              eltVec_7 = tmpEltsU32Ptr[int32_t{87}];
              trtllm::dev::mulBf16x2(eltVec_7, sfVec_H1_H1);
              uint32_t scaledEltVec_7{trtllm::dev::mulBf16x2(eltVec_7, sfVec_H1_H1)};
              dstArrayPacked[int32_t{87}] = scaledEltVec_7;
              uint32_t eltVec_8;
              eltVec_8 = tmpEltsU32Ptr[int32_t{88}];
              trtllm::dev::mulBf16x2(eltVec_8, sfVec_H1_H1);
              uint32_t scaledEltVec_8{trtllm::dev::mulBf16x2(eltVec_8, sfVec_H1_H1)};
              dstArrayPacked[int32_t{88}] = scaledEltVec_8;
              uint32_t eltVec_9;
              eltVec_9 = tmpEltsU32Ptr[int32_t{89}];
              trtllm::dev::mulBf16x2(eltVec_9, sfVec_H1_H1);
              uint32_t scaledEltVec_9{trtllm::dev::mulBf16x2(eltVec_9, sfVec_H1_H1)};
              dstArrayPacked[int32_t{89}] = scaledEltVec_9;
              uint32_t eltVec_10;
              eltVec_10 = tmpEltsU32Ptr[int32_t{90}];
              trtllm::dev::mulBf16x2(eltVec_10, sfVec_H1_H1);
              uint32_t scaledEltVec_10{trtllm::dev::mulBf16x2(eltVec_10, sfVec_H1_H1)};
              dstArrayPacked[int32_t{90}] = scaledEltVec_10;
              uint32_t eltVec_11;
              eltVec_11 = tmpEltsU32Ptr[int32_t{91}];
              trtllm::dev::mulBf16x2(eltVec_11, sfVec_H1_H1);
              uint32_t scaledEltVec_11{trtllm::dev::mulBf16x2(eltVec_11, sfVec_H1_H1)};
              dstArrayPacked[int32_t{91}] = scaledEltVec_11;
              uint32_t eltVec_12;
              eltVec_12 = tmpEltsU32Ptr[int32_t{92}];
              trtllm::dev::mulBf16x2(eltVec_12, sfVec_H1_H1);
              uint32_t scaledEltVec_12{trtllm::dev::mulBf16x2(eltVec_12, sfVec_H1_H1)};
              dstArrayPacked[int32_t{92}] = scaledEltVec_12;
              uint32_t eltVec_13;
              eltVec_13 = tmpEltsU32Ptr[int32_t{93}];
              trtllm::dev::mulBf16x2(eltVec_13, sfVec_H1_H1);
              uint32_t scaledEltVec_13{trtllm::dev::mulBf16x2(eltVec_13, sfVec_H1_H1)};
              dstArrayPacked[int32_t{93}] = scaledEltVec_13;
              uint32_t eltVec_14;
              eltVec_14 = tmpEltsU32Ptr[int32_t{94}];
              trtllm::dev::mulBf16x2(eltVec_14, sfVec_H1_H1);
              uint32_t scaledEltVec_14{trtllm::dev::mulBf16x2(eltVec_14, sfVec_H1_H1)};
              dstArrayPacked[int32_t{94}] = scaledEltVec_14;
              uint32_t eltVec_15;
              eltVec_15 = tmpEltsU32Ptr[int32_t{95}];
              trtllm::dev::mulBf16x2(eltVec_15, sfVec_H1_H1);
              uint32_t scaledEltVec_15{trtllm::dev::mulBf16x2(eltVec_15, sfVec_H1_H1)};
              dstArrayPacked[int32_t{95}] = scaledEltVec_15;
            }
            {
              //
              // Scale elements k=192 to 223.
              //
              uint32_t sfVec_H1_H0;
              uint32_t sfVec_H0_H0;
              sfVec_H1_H0 = tmpSfU32Ptr[int32_t{3}];
              trtllm::dev::prmt(sfVec_H0_H0, uint32_t{0}, sfVec_H1_H0, uint32_t{4112});
              uint32_t eltVec_0;
              eltVec_0 = tmpEltsU32Ptr[int32_t{96}];
              trtllm::dev::mulBf16x2(eltVec_0, sfVec_H0_H0);
              uint32_t scaledEltVec_0{trtllm::dev::mulBf16x2(eltVec_0, sfVec_H0_H0)};
              dstArrayPacked[int32_t{96}] = scaledEltVec_0;
              uint32_t eltVec_1;
              eltVec_1 = tmpEltsU32Ptr[int32_t{97}];
              trtllm::dev::mulBf16x2(eltVec_1, sfVec_H0_H0);
              uint32_t scaledEltVec_1{trtllm::dev::mulBf16x2(eltVec_1, sfVec_H0_H0)};
              dstArrayPacked[int32_t{97}] = scaledEltVec_1;
              uint32_t eltVec_2;
              eltVec_2 = tmpEltsU32Ptr[int32_t{98}];
              trtllm::dev::mulBf16x2(eltVec_2, sfVec_H0_H0);
              uint32_t scaledEltVec_2{trtllm::dev::mulBf16x2(eltVec_2, sfVec_H0_H0)};
              dstArrayPacked[int32_t{98}] = scaledEltVec_2;
              uint32_t eltVec_3;
              eltVec_3 = tmpEltsU32Ptr[int32_t{99}];
              trtllm::dev::mulBf16x2(eltVec_3, sfVec_H0_H0);
              uint32_t scaledEltVec_3{trtllm::dev::mulBf16x2(eltVec_3, sfVec_H0_H0)};
              dstArrayPacked[int32_t{99}] = scaledEltVec_3;
              uint32_t eltVec_4;
              eltVec_4 = tmpEltsU32Ptr[int32_t{100}];
              trtllm::dev::mulBf16x2(eltVec_4, sfVec_H0_H0);
              uint32_t scaledEltVec_4{trtllm::dev::mulBf16x2(eltVec_4, sfVec_H0_H0)};
              dstArrayPacked[int32_t{100}] = scaledEltVec_4;
              uint32_t eltVec_5;
              eltVec_5 = tmpEltsU32Ptr[int32_t{101}];
              trtllm::dev::mulBf16x2(eltVec_5, sfVec_H0_H0);
              uint32_t scaledEltVec_5{trtllm::dev::mulBf16x2(eltVec_5, sfVec_H0_H0)};
              dstArrayPacked[int32_t{101}] = scaledEltVec_5;
              uint32_t eltVec_6;
              eltVec_6 = tmpEltsU32Ptr[int32_t{102}];
              trtllm::dev::mulBf16x2(eltVec_6, sfVec_H0_H0);
              uint32_t scaledEltVec_6{trtllm::dev::mulBf16x2(eltVec_6, sfVec_H0_H0)};
              dstArrayPacked[int32_t{102}] = scaledEltVec_6;
              uint32_t eltVec_7;
              eltVec_7 = tmpEltsU32Ptr[int32_t{103}];
              trtllm::dev::mulBf16x2(eltVec_7, sfVec_H0_H0);
              uint32_t scaledEltVec_7{trtllm::dev::mulBf16x2(eltVec_7, sfVec_H0_H0)};
              dstArrayPacked[int32_t{103}] = scaledEltVec_7;
              uint32_t eltVec_8;
              eltVec_8 = tmpEltsU32Ptr[int32_t{104}];
              trtllm::dev::mulBf16x2(eltVec_8, sfVec_H0_H0);
              uint32_t scaledEltVec_8{trtllm::dev::mulBf16x2(eltVec_8, sfVec_H0_H0)};
              dstArrayPacked[int32_t{104}] = scaledEltVec_8;
              uint32_t eltVec_9;
              eltVec_9 = tmpEltsU32Ptr[int32_t{105}];
              trtllm::dev::mulBf16x2(eltVec_9, sfVec_H0_H0);
              uint32_t scaledEltVec_9{trtllm::dev::mulBf16x2(eltVec_9, sfVec_H0_H0)};
              dstArrayPacked[int32_t{105}] = scaledEltVec_9;
              uint32_t eltVec_10;
              eltVec_10 = tmpEltsU32Ptr[int32_t{106}];
              trtllm::dev::mulBf16x2(eltVec_10, sfVec_H0_H0);
              uint32_t scaledEltVec_10{trtllm::dev::mulBf16x2(eltVec_10, sfVec_H0_H0)};
              dstArrayPacked[int32_t{106}] = scaledEltVec_10;
              uint32_t eltVec_11;
              eltVec_11 = tmpEltsU32Ptr[int32_t{107}];
              trtllm::dev::mulBf16x2(eltVec_11, sfVec_H0_H0);
              uint32_t scaledEltVec_11{trtllm::dev::mulBf16x2(eltVec_11, sfVec_H0_H0)};
              dstArrayPacked[int32_t{107}] = scaledEltVec_11;
              uint32_t eltVec_12;
              eltVec_12 = tmpEltsU32Ptr[int32_t{108}];
              trtllm::dev::mulBf16x2(eltVec_12, sfVec_H0_H0);
              uint32_t scaledEltVec_12{trtllm::dev::mulBf16x2(eltVec_12, sfVec_H0_H0)};
              dstArrayPacked[int32_t{108}] = scaledEltVec_12;
              uint32_t eltVec_13;
              eltVec_13 = tmpEltsU32Ptr[int32_t{109}];
              trtllm::dev::mulBf16x2(eltVec_13, sfVec_H0_H0);
              uint32_t scaledEltVec_13{trtllm::dev::mulBf16x2(eltVec_13, sfVec_H0_H0)};
              dstArrayPacked[int32_t{109}] = scaledEltVec_13;
              uint32_t eltVec_14;
              eltVec_14 = tmpEltsU32Ptr[int32_t{110}];
              trtllm::dev::mulBf16x2(eltVec_14, sfVec_H0_H0);
              uint32_t scaledEltVec_14{trtllm::dev::mulBf16x2(eltVec_14, sfVec_H0_H0)};
              dstArrayPacked[int32_t{110}] = scaledEltVec_14;
              uint32_t eltVec_15;
              eltVec_15 = tmpEltsU32Ptr[int32_t{111}];
              trtllm::dev::mulBf16x2(eltVec_15, sfVec_H0_H0);
              uint32_t scaledEltVec_15{trtllm::dev::mulBf16x2(eltVec_15, sfVec_H0_H0)};
              dstArrayPacked[int32_t{111}] = scaledEltVec_15;
            }
            {
              //
              // Scale elements k=224 to 255.
              //
              uint32_t sfVec_H1_H0;
              uint32_t sfVec_H1_H1;
              sfVec_H1_H0 = tmpSfU32Ptr[int32_t{3}];
              trtllm::dev::prmt(sfVec_H1_H1, uint32_t{0}, sfVec_H1_H0, uint32_t{12850});
              uint32_t eltVec_0;
              eltVec_0 = tmpEltsU32Ptr[int32_t{112}];
              trtllm::dev::mulBf16x2(eltVec_0, sfVec_H1_H1);
              uint32_t scaledEltVec_0{trtllm::dev::mulBf16x2(eltVec_0, sfVec_H1_H1)};
              dstArrayPacked[int32_t{112}] = scaledEltVec_0;
              uint32_t eltVec_1;
              eltVec_1 = tmpEltsU32Ptr[int32_t{113}];
              trtllm::dev::mulBf16x2(eltVec_1, sfVec_H1_H1);
              uint32_t scaledEltVec_1{trtllm::dev::mulBf16x2(eltVec_1, sfVec_H1_H1)};
              dstArrayPacked[int32_t{113}] = scaledEltVec_1;
              uint32_t eltVec_2;
              eltVec_2 = tmpEltsU32Ptr[int32_t{114}];
              trtllm::dev::mulBf16x2(eltVec_2, sfVec_H1_H1);
              uint32_t scaledEltVec_2{trtllm::dev::mulBf16x2(eltVec_2, sfVec_H1_H1)};
              dstArrayPacked[int32_t{114}] = scaledEltVec_2;
              uint32_t eltVec_3;
              eltVec_3 = tmpEltsU32Ptr[int32_t{115}];
              trtllm::dev::mulBf16x2(eltVec_3, sfVec_H1_H1);
              uint32_t scaledEltVec_3{trtllm::dev::mulBf16x2(eltVec_3, sfVec_H1_H1)};
              dstArrayPacked[int32_t{115}] = scaledEltVec_3;
              uint32_t eltVec_4;
              eltVec_4 = tmpEltsU32Ptr[int32_t{116}];
              trtllm::dev::mulBf16x2(eltVec_4, sfVec_H1_H1);
              uint32_t scaledEltVec_4{trtllm::dev::mulBf16x2(eltVec_4, sfVec_H1_H1)};
              dstArrayPacked[int32_t{116}] = scaledEltVec_4;
              uint32_t eltVec_5;
              eltVec_5 = tmpEltsU32Ptr[int32_t{117}];
              trtllm::dev::mulBf16x2(eltVec_5, sfVec_H1_H1);
              uint32_t scaledEltVec_5{trtllm::dev::mulBf16x2(eltVec_5, sfVec_H1_H1)};
              dstArrayPacked[int32_t{117}] = scaledEltVec_5;
              uint32_t eltVec_6;
              eltVec_6 = tmpEltsU32Ptr[int32_t{118}];
              trtllm::dev::mulBf16x2(eltVec_6, sfVec_H1_H1);
              uint32_t scaledEltVec_6{trtllm::dev::mulBf16x2(eltVec_6, sfVec_H1_H1)};
              dstArrayPacked[int32_t{118}] = scaledEltVec_6;
              uint32_t eltVec_7;
              eltVec_7 = tmpEltsU32Ptr[int32_t{119}];
              trtllm::dev::mulBf16x2(eltVec_7, sfVec_H1_H1);
              uint32_t scaledEltVec_7{trtllm::dev::mulBf16x2(eltVec_7, sfVec_H1_H1)};
              dstArrayPacked[int32_t{119}] = scaledEltVec_7;
              uint32_t eltVec_8;
              eltVec_8 = tmpEltsU32Ptr[int32_t{120}];
              trtllm::dev::mulBf16x2(eltVec_8, sfVec_H1_H1);
              uint32_t scaledEltVec_8{trtllm::dev::mulBf16x2(eltVec_8, sfVec_H1_H1)};
              dstArrayPacked[int32_t{120}] = scaledEltVec_8;
              uint32_t eltVec_9;
              eltVec_9 = tmpEltsU32Ptr[int32_t{121}];
              trtllm::dev::mulBf16x2(eltVec_9, sfVec_H1_H1);
              uint32_t scaledEltVec_9{trtllm::dev::mulBf16x2(eltVec_9, sfVec_H1_H1)};
              dstArrayPacked[int32_t{121}] = scaledEltVec_9;
              uint32_t eltVec_10;
              eltVec_10 = tmpEltsU32Ptr[int32_t{122}];
              trtllm::dev::mulBf16x2(eltVec_10, sfVec_H1_H1);
              uint32_t scaledEltVec_10{trtllm::dev::mulBf16x2(eltVec_10, sfVec_H1_H1)};
              dstArrayPacked[int32_t{122}] = scaledEltVec_10;
              uint32_t eltVec_11;
              eltVec_11 = tmpEltsU32Ptr[int32_t{123}];
              trtllm::dev::mulBf16x2(eltVec_11, sfVec_H1_H1);
              uint32_t scaledEltVec_11{trtllm::dev::mulBf16x2(eltVec_11, sfVec_H1_H1)};
              dstArrayPacked[int32_t{123}] = scaledEltVec_11;
              uint32_t eltVec_12;
              eltVec_12 = tmpEltsU32Ptr[int32_t{124}];
              trtllm::dev::mulBf16x2(eltVec_12, sfVec_H1_H1);
              uint32_t scaledEltVec_12{trtllm::dev::mulBf16x2(eltVec_12, sfVec_H1_H1)};
              dstArrayPacked[int32_t{124}] = scaledEltVec_12;
              uint32_t eltVec_13;
              eltVec_13 = tmpEltsU32Ptr[int32_t{125}];
              trtllm::dev::mulBf16x2(eltVec_13, sfVec_H1_H1);
              uint32_t scaledEltVec_13{trtllm::dev::mulBf16x2(eltVec_13, sfVec_H1_H1)};
              dstArrayPacked[int32_t{125}] = scaledEltVec_13;
              uint32_t eltVec_14;
              eltVec_14 = tmpEltsU32Ptr[int32_t{126}];
              trtllm::dev::mulBf16x2(eltVec_14, sfVec_H1_H1);
              uint32_t scaledEltVec_14{trtllm::dev::mulBf16x2(eltVec_14, sfVec_H1_H1)};
              dstArrayPacked[int32_t{126}] = scaledEltVec_14;
              uint32_t eltVec_15;
              eltVec_15 = tmpEltsU32Ptr[int32_t{127}];
              trtllm::dev::mulBf16x2(eltVec_15, sfVec_H1_H1);
              uint32_t scaledEltVec_15{trtllm::dev::mulBf16x2(eltVec_15, sfVec_H1_H1)};
              dstArrayPacked[int32_t{127}] = scaledEltVec_15;
            }
            {
              uint32_t tmemBasePtr{mTmemBaseOffset};
              uint32_t const(&srcSlice0)[32]{
                reinterpret_cast<uint32_t const(&)[32]>(dstArrayPacked[int32_t{0}])};
              cuda_ptx::tcgen05_st_32x32b(
                (tmemBasePtr) +
                  (static_cast<uint32_t>((int32_t{128}) + ((index) * (int32_t{128})))),
                srcSlice0);
              uint32_t const(&srcSlice1)[32]{
                reinterpret_cast<uint32_t const(&)[32]>(dstArrayPacked[int32_t{32}])};
              cuda_ptx::tcgen05_st_32x32b(
                (tmemBasePtr) +
                  (static_cast<uint32_t>(((int32_t{128}) + ((index) * (int32_t{128}))) +
                                         (int32_t{0x20 /*hi=0, lo=32*/}))),
                srcSlice1);
              uint32_t const(&srcSlice2)[32]{
                reinterpret_cast<uint32_t const(&)[32]>(dstArrayPacked[int32_t{64}])};
              cuda_ptx::tcgen05_st_32x32b(
                (tmemBasePtr) +
                  (static_cast<uint32_t>(((int32_t{128}) + ((index) * (int32_t{128}))) +
                                         (int32_t{0x40 /*hi=0, lo=64*/}))),
                srcSlice2);
              uint32_t const(&srcSlice3)[32]{
                reinterpret_cast<uint32_t const(&)[32]>(dstArrayPacked[int32_t{96}])};
              cuda_ptx::tcgen05_st_32x32b(
                (tmemBasePtr) +
                  (static_cast<uint32_t>(((int32_t{128}) + ((index) * (int32_t{128}))) +
                                         (int32_t{0x60 /*hi=0, lo=96*/}))),
                srcSlice3);
            }
            cutlass::arch::fence_view_async_tmem_store();
          }
        }
        //
        // smemA [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        //
        // smemSfA [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        //
        // tmemCastA [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
      }
      //
      // Unrolled tail iter 0.
      //
      if ((loopEnd) > (int32_t{0})) {
        //
        // tmemCastA [ProdPreCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopEnd) >= (int32_t{256})) {
        }
        //
        // tmemCastA [ProdCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopEnd) >= (int32_t{256})) {
          {
            tmemCastADstStack.mPipeline.producer_commit(tmemCastAProdState);
          }
          ++tmemCastAProdState;
        }
        //
        // smemA [ConsRelease, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopEnd) >= (int32_t{256})) {
          trtllm::dev::CutlassNamedBarrier::sync(128, 1);
          { smemASrcStack.mPipeline.consumer_release(smemAConsReleaseState); }
          ++smemAConsReleaseState;
        }
        //
        // smemSfA [ConsRelease, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopEnd) >= (int32_t{256})) {
          trtllm::dev::CutlassNamedBarrier::sync(128, 3);
          { smemSfASrcStack.mPipeline.consumer_release(smemSfAConsReleaseState); }
          ++smemSfAConsReleaseState;
        }
      }
      //
      // Tail work.
      //
      //
      // smemA [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
      //
      // smemSfA [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
      //
      // workId [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
      //
      // tmemCastA [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
    return ((state.mWarpIdx) >= (int32_t{11})) && ((state.mWarpIdx) < (int32_t{12}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 Mma0Stack& mma0DstStack,
                                 TmemCastAStack& tmemCastASrcStack,
                                 SmemBSmem& smemBSrcSmem,
                                 SmemBStack& smemBSrcStack,
                                 WorkIdSmem& workIdSrcSmem,
                                 WorkIdStack& workIdSrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<96>{});
    if (!((int32_t{cute::block_rank_in_cluster()}) == (int32_t{trtllm::dev::getLeadCtaRank()}))) {
      return;
    }
    trtllm::dev::CutlassUmmaConsumerAsyncPipeline<
      3,
      false,
      false,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState tmemCastAConsState{};
    trtllm::dev::CutlassUmmaConsumerAsyncPipeline<
      3,
      false,
      false,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState
      tmemCastAConsReleaseState{};
    int32_t tmemCastAConsToken{int32_t{0}};
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      3,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemBConsState{};
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      3,
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
      int32_t paddedPerCtaK{(((params.k) + (int32_t{255})) / (int32_t{256})) * (int32_t{256})};
      int32_t loopEnd{paddedPerCtaK};
      if ((mCtaIdxY) >= (state.mNumNonExitingCtas)) {
        goto ExitTileWithoutSignalingLabel;
      }
      //
      // Loop body.
      //
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset1814 = int32_t{0}; loopOffset1814 < loopEnd;
           loopOffset1814 += int32_t{256}) {
        bool const isFirstLoopIter{(loopOffset1814) == (int32_t{0})};
        bool const isLastLoopIter{((loopOffset1814) + (int32_t{256})) >= (loopEnd)};
        //
        // mma0 [ProdTryAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if (isFirstLoopIter) {
          if ((loopOffset1814) >= (int32_t{0})) {
            mma0ProdToken = mma0DstStack.mPipeline.producer_try_acquire(mma0ProdState);
          }
        }
        //
        // tmemCastA [ConsTryWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        { tmemCastAConsToken = tmemCastASrcStack.mPipeline.consumer_try_wait(tmemCastAConsState); }
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
        // tmemCastA [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{8388608}].
        //
        { tmemCastASrcStack.mPipeline.consumer_wait(tmemCastAConsState, tmemCastAConsToken); }
        //
        // smemB [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{8388608}].
        //
        { smemBSrcStack.mPipeline.consumer_wait(smemBConsState, smemBConsToken); }
        //
        // tmemCastA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        uint32_t tmemAddrA10;
        {
          int32_t index{tmemCastAConsState.index()};
          tmemAddrA10 = (mTmemBaseOffset) +
                        ((uint32_t{128}) + (static_cast<uint32_t>((index) * (int32_t{128}))));
          ++tmemCastAConsState;
        }
        //
        // smemB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::bfloat16_t* smemPtrB8;
        {
          int32_t index{smemBConsState.index()};
          int8_t* smemBytesBasePtrB;
          smemBytesBasePtrB =
            reinterpret_cast<int8_t*>(smemBSrcStack.mDepSmemPtr2) + (int32_t{49152});
          int8_t* smemBytesStagePtrB;
          smemBytesStagePtrB = smemBytesBasePtrB + ((index) * (int32_t{16384}));
          smemPtrB8 = reinterpret_cast<cutlass::bfloat16_t*>(smemBytesStagePtrB) + (int32_t{0});
          ++smemBConsState;
        }
        //
        // mma0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        uint32_t tmemPtrA12;
        cutlass::bfloat16_t* smemPtrB12;
        tmemPtrA12 = tmemAddrA10;
        smemPtrB12 = smemPtrB8;
        {
          int32_t index{mma0ProdState.index()};
          uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{64})))};
          uint32_t ptrTmemOffsetD{ptrTmemD};
          uint32_t ptrTmemOffsetA{tmemPtrA12};
          cutlass::bfloat16_t* ptrWithOffsetSmemB{(smemPtrB12 + int32_t{0})};
          {
            uint32_t tmemPtrD{ptrTmemOffsetD};
            uint32_t tmemPtrA{ptrTmemOffsetA};
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
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_2,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_0,
                                           bool{(loopOffset1814) != (int32_t{0})});
            }
            //
            // MMA inst for mi=0 ni=0 ki=1.
            //
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
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
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_2,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_1,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=2.
            //
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
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
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_2,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_2,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=3.
            //
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
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
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_2,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_3,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=4.
            //
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
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
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_2,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_4,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=5.
            //
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
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
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_2,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_5,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=6.
            //
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
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
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_2,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_6,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=7.
            //
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
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
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_2,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_7,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=8.
            //
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{250});
            uint64_t utcmmaDesc_0_0_8{trtllm::dev::make_utcmma_desc(int32_t{1},
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
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_2,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_8,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=9.
            //
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            uint64_t utcmmaDesc_0_0_9{trtllm::dev::make_utcmma_desc(int32_t{1},
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
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_2,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_9,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=10.
            //
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            uint64_t utcmmaDesc_0_0_10{trtllm::dev::make_utcmma_desc(int32_t{1},
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
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_2,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_10,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=11.
            //
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            uint64_t utcmmaDesc_0_0_11{trtllm::dev::make_utcmma_desc(int32_t{1},
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
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_2,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_11,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=12.
            //
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{250});
            uint64_t utcmmaDesc_0_0_12{trtllm::dev::make_utcmma_desc(int32_t{1},
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
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_2,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_12,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=13.
            //
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            uint64_t utcmmaDesc_0_0_13{trtllm::dev::make_utcmma_desc(int32_t{1},
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
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_2,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_13,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=14.
            //
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            uint64_t utcmmaDesc_0_0_14{trtllm::dev::make_utcmma_desc(int32_t{1},
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
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_2,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_14,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=15.
            //
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            uint64_t utcmmaDesc_0_0_15{trtllm::dev::make_utcmma_desc(int32_t{1},
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
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_2,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_15,
                                           bool{true});
            }
          }
        }
        //
        // tmemCastA [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset1814) >= (int32_t{0})) {
          {
            tmemCastASrcStack.mPipeline.consumer_release(tmemCastAConsReleaseState);
          }
          ++tmemCastAConsReleaseState;
        }
        //
        // smemB [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset1814) >= (int32_t{0})) {
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
  cutlass::Array<float, 64> frg13;
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
    cuda_ptx::setmaxnreg_inc(cuda_ptx::n32_t<160>{});
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
      // Loop body.
      //
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset2072 = int32_t{0}; loopOffset2072 < loopEnd;
           loopOffset2072 += int32_t{256}) {
        bool const isFirstLoopIter{(loopOffset2072) == (int32_t{0})};
        //
        // gmemC0 [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        //
        // mma0 [ConsTailRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        lastLoopOffset = loopOffset2072;
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
      uint32_t tmemBaseWithStageOffset12;
      if (hasOneLoopIter) {
        int32_t index{mma0ConsState.index()};
        uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{64})))};
        uint32_t ptrTmemOffsetD{ptrTmemD};
        tmemBaseWithStageOffset12 = ptrTmemOffsetD;
      }
      //
      // gmemC0 [ProdWork (call 0), LastIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      uint32_t tmemBaseWithStageOffset13;
      tmemBaseWithStageOffset13 = tmemBaseWithStageOffset12;
      if (hasOneLoopIter) {
        tmemBaseWithStageOffset = tmemBaseWithStageOffset13;
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
            uint32_t(&dstSlice0)[32]{reinterpret_cast<uint32_t(&)[32]>(frg13[int32_t{0}])};
            cuda_ptx::tcgen05_ld_16x256b(dstSlice0,
                                         (tmemBasePtr) +
                                           (static_cast<uint32_t>((mWarpGrp4Idx) * (int32_t{64}))));
            uint32_t(&dstSlice1)[32]{reinterpret_cast<uint32_t(&)[32]>(frg13[int32_t{32}])};
            cuda_ptx::tcgen05_ld_16x256b(
              dstSlice1,
              (tmemBasePtr) + (static_cast<uint32_t>(((mWarpGrp4Idx) * (int32_t{64})) +
                                                     (int32_t{0x100000 /*hi=16, lo=0*/}))));
          }
          cutlass::arch::fence_view_async_tmem_load();
          cuda_ptx::cp_async_bulk_wait_group_read(cuda_ptx::n32_t<0>{});
          trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{6}) + (mWarpGrp4Idx));
          //
          // Store to Smem TmaAsyncGmemC.
          //
          int8_t* ptrSmemBase;
          cutlass::bfloat16_t* ptrSmem;
          ptrSmemBase = reinterpret_cast<int8_t*>(gmemC0DstStack.mDepSmemPtr2) +
                        ((mWarpGrp4Idx) * (int32_t{16384}) + (int32_t{98304}));
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
            cutlass::Array<float, 4> accF4{frg13[int32_t{0}],
                                           frg13[int32_t{2}],
                                           frg13[int32_t{32}],
                                           frg13[int32_t{34}]};
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
            cutlass::Array<float, 4> accF4{frg13[int32_t{1}],
                                           frg13[int32_t{3}],
                                           frg13[int32_t{33}],
                                           frg13[int32_t{35}]};
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
            cutlass::Array<float, 4> accF4{frg13[int32_t{4}],
                                           frg13[int32_t{6}],
                                           frg13[int32_t{36}],
                                           frg13[int32_t{38}]};
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
            cutlass::Array<float, 4> accF4{frg13[int32_t{5}],
                                           frg13[int32_t{7}],
                                           frg13[int32_t{37}],
                                           frg13[int32_t{39}]};
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
            cutlass::Array<float, 4> accF4{frg13[int32_t{8}],
                                           frg13[int32_t{10}],
                                           frg13[int32_t{40}],
                                           frg13[int32_t{42}]};
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
            cutlass::Array<float, 4> accF4{frg13[int32_t{9}],
                                           frg13[int32_t{11}],
                                           frg13[int32_t{41}],
                                           frg13[int32_t{43}]};
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
            cutlass::Array<float, 4> accF4{frg13[int32_t{12}],
                                           frg13[int32_t{14}],
                                           frg13[int32_t{44}],
                                           frg13[int32_t{46}]};
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
            cutlass::Array<float, 4> accF4{frg13[int32_t{13}],
                                           frg13[int32_t{15}],
                                           frg13[int32_t{45}],
                                           frg13[int32_t{47}]};
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
            cutlass::Array<float, 4> accF4{frg13[int32_t{16}],
                                           frg13[int32_t{18}],
                                           frg13[int32_t{48}],
                                           frg13[int32_t{50}]};
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
            cutlass::Array<float, 4> accF4{frg13[int32_t{17}],
                                           frg13[int32_t{19}],
                                           frg13[int32_t{49}],
                                           frg13[int32_t{51}]};
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
            cutlass::Array<float, 4> accF4{frg13[int32_t{20}],
                                           frg13[int32_t{22}],
                                           frg13[int32_t{52}],
                                           frg13[int32_t{54}]};
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
            cutlass::Array<float, 4> accF4{frg13[int32_t{21}],
                                           frg13[int32_t{23}],
                                           frg13[int32_t{53}],
                                           frg13[int32_t{55}]};
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
            cutlass::Array<float, 4> accF4{frg13[int32_t{24}],
                                           frg13[int32_t{26}],
                                           frg13[int32_t{56}],
                                           frg13[int32_t{58}]};
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
            cutlass::Array<float, 4> accF4{frg13[int32_t{25}],
                                           frg13[int32_t{27}],
                                           frg13[int32_t{57}],
                                           frg13[int32_t{59}]};
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
            cutlass::Array<float, 4> accF4{frg13[int32_t{28}],
                                           frg13[int32_t{30}],
                                           frg13[int32_t{60}],
                                           frg13[int32_t{62}]};
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
            cutlass::Array<float, 4> accF4{frg13[int32_t{29}],
                                           frg13[int32_t{31}],
                                           frg13[int32_t{61}],
                                           frg13[int32_t{63}]};
            cutlass::Array<cutlass::bfloat16_t, 4> scaledCvtAcc4{
              trtllm::dev::convert_float4_to_bfloat16(accF4)};
            {
              uint64_t convertedElts;
              convertedElts = reinterpret_cast<uint64_t&>(scaledCvtAcc4);
              reinterpret_cast<uint64_t*>(ptrSmem)[(smemOffset0) / (int32_t{4})] = convertedElts;
            }
          }
          cuda_ptx::fence_proxy_async(cuda_ptx::space_shared_t{});
          trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{6}) + (mWarpGrp4Idx));
          //
          // Issue TMA from smem to gmem.
          //
          if ((bool{cute::elect_one_sync()}) && ((mWarpGrp4WarpIdx) == (int32_t{0}))) {
            int8_t* ptrSmemBase;
            cutlass::bfloat16_t* ptrSmem;
            ptrSmemBase = reinterpret_cast<int8_t*>(gmemC0DstStack.mDepSmemPtr2) +
                          ((mWarpGrp4Idx) * (int32_t{16384}) + (int32_t{98304}));
            ptrSmem = reinterpret_cast<cutlass::bfloat16_t*>(ptrSmemBase) + (int32_t{0});
            int32_t coords[4];
            coords[int32_t{0}] = (mCtaIdxX) * (int32_t{128});
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
            coords[int32_t{0}] += int32_t{64};
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_global_t{},
                                           cuda_ptx::space_shared_t{},
                                           params.tmaC,
                                           coords,
                                           &ptrSmem[int32_t{4096}]);
          }
          cuda_ptx::cp_async_bulk_commit_group();
          trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{6}) + (mWarpGrp4Idx));
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
    return ((state.mWarpIdx) >= (int32_t{13})) && ((state.mWarpIdx) < (int32_t{16}));
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
    return ((state.mWarpIdx) >= (int32_t{12})) && ((state.mWarpIdx) < (int32_t{13}));
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
extern "C" __global__ __launch_bounds__(512, 1)
  __cluster_dims__(2, 1, 1) void bmm_Bfloat16_MxInt4Bfloat16_castBfloat16_Fp32_bA32_t128x64x256_s3_et128x64_m256x64x16_cga2x1x1_16dp256b_rM_BN_transOut_schPd2x1x2x3_bN_rgTma_clmp_dynB_sm100f(
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
  uint8_t* tmemCastASmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(TmemCastASmemBarrier)});
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
                          int32_t{12},
                          int32_t{-1}};
  WorkThrottleBarrierSmemBarrier* workThrottleBarrierSmemBarrier{
    reinterpret_cast<WorkThrottleBarrierSmemBarrier*>(workThrottleBarrierSmemBarrierPtr)};
  WorkThrottleBarrierStack workThrottleBarrierStack{(*workThrottleBarrierSmemBarrier),
                                                    state.mWarpIdx,
                                                    int32_t{12},
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
                        int32_t{9},
                        int32_t{-1}};
  SmemBSmem* smemBSmem{reinterpret_cast<SmemBSmem*>(smemBSmemPtr)};
  SmemBSmemBarrier* smemBSmemBarrier{reinterpret_cast<SmemBSmemBarrier*>(smemBSmemBarrierPtr)};
  SmemBStack smemBStack{(*smemBSmem),
                        (*smemBSmemBarrier),
                        (*smemBufferSmem),
                        smemBufferStack,
                        state.mWarpIdx,
                        int32_t{8},
                        int32_t{-1}};
  SmemSfASmem* smemSfASmem{reinterpret_cast<SmemSfASmem*>(smemSfASmemPtr)};
  SmemSfASmemBarrier* smemSfASmemBarrier{
    reinterpret_cast<SmemSfASmemBarrier*>(smemSfASmemBarrierPtr)};
  SmemSfAStack smemSfAStack{(*smemSfASmem),
                            (*smemSfASmemBarrier),
                            state.mWarpIdx,
                            int32_t{10},
                            int32_t{-1}};
  TmemCastASmemBarrier* tmemCastASmemBarrier{
    reinterpret_cast<TmemCastASmemBarrier*>(tmemCastASmemBarrierPtr)};
  TmemCastAStack tmemCastAStack{(*tmemCastASmemBarrier), state.mWarpIdx, int32_t{4}, int32_t{-1}};
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
  LoadTaskA loadTaskA{params, state, int32_t{9}};
  LoadTaskB loadTaskB{params, state, int32_t{8}};
  cutlass::arch::fence_barrier_init();
  cuda_ptx::fence_mbarrier_init(cuda_ptx::sem_release_t{}, cuda_ptx::scope_cluster_t{});
  cuda_ptx::barrier_cluster_arrive(cuda_ptx::sem_relaxed_t{});
  cuda_ptx::barrier_cluster_wait();
  if ((reinterpret_cast<int32_t const&>(threadIdx.x)) < (int32_t{32})) {
    cuda_ptx::tcgen05_alloc(cuda_ptx::cta_group_2_t{}, state.mTmemSwStatePtr, int32_t{512});
    cuda_ptx::tcgen05_relinquish_alloc_permit(cuda_ptx::cta_group_2_t{});
  }
  if (((bool{LoadTaskA::isSelected(params, state)}) ||
       (bool{LoadTaskB::isSelected(params, state)})) ||
      (bool{LoadSfATask::isSelected(params, state)})) {
  } else {
    trtllm::dev::CutlassNamedBarrier::sync(416, 8);
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
        LoadSfATask loadSfATask{params, state, int32_t{10}};
        loadSfATask
          .execute(params, state, (*smemSfASmem), smemSfAStack, (*workIdSmem), workIdStack);
      } else {
        if (bool{CastATask::isSelected(params, state)}) {
          CastATask castATask{params, state, int32_t{4}};
          castATask.execute(params,
                            state,
                            tmemCastAStack,
                            (*smemASmem),
                            smemAStack,
                            (*smemSfASmem),
                            smemSfAStack,
                            (*workIdSmem),
                            workIdStack);
        } else {
          if (bool{MmaTask0::isSelected(params, state)}) {
            MmaTask0 mmaTask0{params, state, int32_t{11}};
            mmaTask0.execute(params,
                             state,
                             mma0Stack,
                             tmemCastAStack,
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
              trtllm::dev::CutlassNamedBarrier::sync(128, 9);
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
                PaddingTask paddingTask{params, state, int32_t{13}};
                paddingTask.execute(params, state, (*workIdSmem), workIdStack);
              } else {
                if (bool{WorkIdTask::isSelected(params, state)}) {
                  WorkIdTask workIdTask{params, state, int32_t{12}};
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
extern "C" __global__ void
bmm_Bfloat16_MxInt4Bfloat16_castBfloat16_Fp32_bA32_t128x64x256_s3_et128x64_m256x64x16_cga2x1x1_16dp256b_rM_BN_transOut_schPd2x1x2x3_bN_rgTma_clmp_dynB_sm100fGetSmemSize(
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
  size += static_cast<int32_t>(sizeof(TmemCastASmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(Mma0SmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(ClusterBarrierBuffersSmemBarrier));
  outPtr[0] = size;
}

} // namespace batchedGemm
