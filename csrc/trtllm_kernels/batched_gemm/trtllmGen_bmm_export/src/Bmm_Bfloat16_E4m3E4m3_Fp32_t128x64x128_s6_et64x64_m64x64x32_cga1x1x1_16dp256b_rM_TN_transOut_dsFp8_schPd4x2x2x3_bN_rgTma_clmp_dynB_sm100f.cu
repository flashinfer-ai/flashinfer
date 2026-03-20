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
#include <Bmm_Bfloat16_E4m3E4m3_Fp32_t128x64x128_s6_et64x64_m64x64x32_cga1x1x1_16dp256b_rM_TN_transOut_dsFp8_schPd4x2x2x3_bN_rgTma_clmp_dynB_sm100f.h>
namespace batchedGemm {


struct WorkIdStack {
  trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  cutlass::gemm::kernel::detail::
    PersistentTileSchedulerSm100<cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, 3>
      mScheduler;
  typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
    3>::WorkTileInfo workTileInfo;
  inline __device__ WorkIdStack(WorkIdSmem& workIdSmem,
                                WorkIdSmemBarrier& workIdSmemBarrier,
                                KernelParams const& params,
                                int32_t warpId,
                                int32_t barInitWarpId,
                                int32_t orderedSequenceGroupId)
    : mPipeline{workIdSmemBarrier.mBarriers, int32_t{1}, int32_t{512}, int32_t{0}, barInitWarpId}
    , mScheduler{&workIdSmem.workIdResponse[int32_t{0}],
                 typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<
                   cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
                   3>::Params{},
                 cute::block_id_in_cluster()}
    , workTileInfo{mScheduler.initial_work_tile_info(CuteFlatTuple391{})} {}
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
  trtllm::dev::
    CutlassTmaMultiUmmaAsyncPipeline<6, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
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
                int32_t{16384},
                ((warpId) == (barInitWarpId)) && (bool{cute::elect_one_sync()}),
                int32_t{1},
                CuteFlatTuple626{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId} {}
};
struct SmemBStack {
  int8_t* mDepSmemPtr2;
  trtllm::dev::
    CutlassTmaMultiUmmaAsyncPipeline<6, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
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
                int32_t{8192},
                ((warpId) == (barInitWarpId)) && (bool{cute::elect_one_sync()}),
                int32_t{1},
                CuteFlatTuple748{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId} {}
};
struct SmemDeepSeekSfAbStack {
  trtllm::dev::CutlassCpAsyncPipeline<6> mPipeline;
  inline __device__ SmemDeepSeekSfAbStack(SmemDeepSeekSfAbSmem& smemDeepSeekSfAbSmem,
                                          SmemDeepSeekSfAbSmemBarrier& smemDeepSeekSfAbSmemBarrier,
                                          int32_t warpId,
                                          int32_t barInitWarpId,
                                          int32_t orderedSequenceGroupId)
    : mPipeline{smemDeepSeekSfAbSmemBarrier.mBarriers,
                warpId,
                int32_t{32},
                int32_t{256},
                barInitWarpId} {}
};
struct TmemStack {
  inline __device__ TmemStack(int32_t warpId,
                              int32_t barInitWarpId,
                              int32_t orderedSequenceGroupId) {}
};
struct Mma0Stack {
  trtllm::dev::CutlassUmmaAsyncPipeline<4, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  inline __device__ Mma0Stack(Mma0SmemBarrier& mma0SmemBarrier,
                              TmemStack& tmemStack,
                              int32_t warpId,
                              int32_t barInitWarpId,
                              int32_t orderedSequenceGroupId)
    : mPipeline{mma0SmemBarrier.mBarriers,
                warpId,
                int32_t{128},
                CuteFlatTuple981{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId} {}
};
struct Mma1Stack {
  trtllm::dev::CutlassUmmaAsyncPipeline<4, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  inline __device__ Mma1Stack(Mma1SmemBarrier& mma1SmemBarrier,
                              TmemStack& tmemStack,
                              int32_t warpId,
                              int32_t barInitWarpId,
                              int32_t orderedSequenceGroupId)
    : mPipeline{mma1SmemBarrier.mBarriers,
                warpId,
                int32_t{128},
                CuteFlatTuple1090{},
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
struct GmemC1Stack {
  int8_t* mDepSmemPtr2;
  inline __device__ GmemC1Stack(GmemC1Smem& gmemC1Smem,
                                SmemBufferSmem& smemBufferSmem,
                                SmemBufferStack& smemBufferStack,
                                int32_t warpId,
                                int32_t barInitWarpId,
                                int32_t orderedSequenceGroupId)
    : mDepSmemPtr2{&smemBufferSmem.mArray[int32_t{0}]} {}
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
  inline __device__ LoadTaskA(KernelParams const& params,
                              KernelState const& state,
                              int32_t warpGrpStart)
    : mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mBatchIdx{int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}}
    , mBatchLimit{int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]}}
    , mCtaOffsetK{int32_t{0}}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)} {}
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
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<48>{});
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsState{};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsReleaseState{};
    int32_t workIdConsToken{int32_t{0}};
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      6,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemAProdState{
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
      {
        {
          workThrottleBarrierProdToken = workThrottleBarrierDstStack.mPipeline.producer_try_acquire(
            workThrottleBarrierProdState);
        }
      }
      {
        workThrottleBarrierDstStack.mPipeline.producer_acquire(workThrottleBarrierProdState,
                                                               workThrottleBarrierProdToken);
      }
      //
      // workThrottleBarrier [ProdCommit, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{4608}].
      //
      {
        { workThrottleBarrierDstStack.mPipeline.producer_commit(workThrottleBarrierProdState); }
        ++workThrottleBarrierProdState;
      }
      //
      // Loop body.
      //
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset470 = int32_t{0}; loopOffset470 < loopEnd;
           loopOffset470 += int32_t{128}) {
        bool const isLastLoopIter{((loopOffset470) + (int32_t{128})) >= (loopEnd)};
        //
        // gmemA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK3;
        { tileOffsetK3 = loopOffset470; }
        //
        // smemA [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          smemADstStack.mPipeline.producer_acquire(smemAProdState, smemAProdToken);
          if (((loopOffset470) + (int32_t{128})) < (loopEnd)) {
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
            int32_t coords[3];
            coords[int32_t{0}] = tileOffsetK7;
            coords[int32_t{1}] = (mCtaIdxX) * (int32_t{128});
            coords[int32_t{2}] = mBatchIdx;
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::cp_async_bulk_tensor(
                cuda_ptx::space_cluster_t{},
                cuda_ptx::space_global_t{},
                &reinterpret_cast<cutlass::float_e4m3_t*>(smemBytesStagePtrA)[int32_t{0}],
                params.tmaA,
                coords,
                barrier);
            }
          }
        }
        //
        // smemA [ProdPreCommit, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset470) >= (int32_t{0})) {
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
  inline __device__ LoadTaskB(KernelParams const& params,
                              KernelState const& state,
                              int32_t warpGrpStart)
    : mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mBatchIdx{int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}}
    , mBatchLimit{int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]}}
    , mCtaOffsetK{int32_t{0}}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{10})) && ((state.mWarpIdx) < (int32_t{11}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 SmemBSmem& smemBDstSmem,
                                 SmemBStack& smemBDstStack,
                                 WorkIdSmem& workIdSrcSmem,
                                 WorkIdStack& workIdSrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<48>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsState{};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsReleaseState{};
    int32_t workIdConsToken{int32_t{0}};
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      6,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemBProdState{
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
      for (int32_t loopOffset570 = int32_t{0}; loopOffset570 < loopEnd;
           loopOffset570 += int32_t{128}) {
        bool const isLastLoopIter{((loopOffset570) + (int32_t{128})) >= (loopEnd)};
        //
        // gmemB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK4;
        { tileOffsetK4 = loopOffset570; }
        //
        // smemB [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          smemBDstStack.mPipeline.producer_acquire(smemBProdState, smemBProdToken);
          if (((loopOffset570) + (int32_t{128})) < (loopEnd)) {
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
              reinterpret_cast<int8_t*>(smemBDstStack.mDepSmemPtr2) + (int32_t{98304});
            smemBytesStagePtrB = smemBytesBasePtrB + ((index) * (int32_t{8192}));
            int32_t coords[4];
            coords[int32_t{0}] = tileOffsetK8;
            coords[int32_t{1}] = ((int32_t{64}) - ((mBatchLimit) % (int32_t{64}))) % (int32_t{64});
            coords[int32_t{2}] = int32_t{0x40000000 /*1073741824*/};
            coords[int32_t{3}] =
              (((mCtaIdxY) * (int32_t{64})) +
               ((int32_t{0}) -
                (((int32_t{64}) - ((mBatchLimit) % (int32_t{64}))) % (int32_t{64})))) +
              (int32_t{0x40000000 /*1073741824*/});
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::cp_async_bulk_tensor(
                cuda_ptx::space_cluster_t{},
                cuda_ptx::space_global_t{},
                &reinterpret_cast<cutlass::float_e4m3_t*>(smemBytesStagePtrB)[int32_t{0}],
                params.tmaB,
                coords,
                barrier);
            }
          }
        }
        //
        // smemB [ProdPreCommit, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset570) >= (int32_t{0})) {
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
struct LoadSfAbTask {
  int32_t mCtaOffsetK;
  int32_t mCtaIdxX;
  int32_t mCtaIdxY;
  int32_t mCtaIdxZ;
  int32_t const mWarpGrpThreadIdx;
  inline __device__ LoadSfAbTask(KernelParams const& params,
                                 KernelState const& state,
                                 int32_t warpGrpStart)
    : mCtaOffsetK{int32_t{0}}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)}
    , mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{11})) && ((state.mWarpIdx) < (int32_t{12}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 SmemDeepSeekSfAbSmem& smemDeepSeekSfAbDstSmem,
                                 SmemDeepSeekSfAbStack& smemDeepSeekSfAbDstStack,
                                 WorkIdSmem& workIdSrcSmem,
                                 WorkIdStack& workIdSrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<48>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsState{};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsReleaseState{};
    int32_t workIdConsToken{int32_t{0}};
    trtllm::dev::CutlassCpAsyncPipeline<6>::PipelineState smemDeepSeekSfAbProdState{int32_t{0},
                                                                                    int32_t{1},
                                                                                    int32_t{0}};
    int32_t smemDeepSeekSfAbProdToken{int32_t{1}};
    do {
      int32_t paddedPerCtaK{(((params.k) + (int32_t{127})) / (int32_t{128})) * (int32_t{128})};
      int32_t loopEnd{paddedPerCtaK};
      if ((mCtaIdxY) >= (state.mNumNonExitingCtas)) {
        goto ExitTileWithoutSignalingLabel;
      }
      float const* gmemDqSfsAct;
      gmemDqSfsAct = reinterpret_cast<float const*>(params.ptrSfB) + (int32_t{0});
      int32_t colDqSfsAct;
      colDqSfsAct = (mCtaIdxY) * (int32_t{64}) + (mWarpGrpThreadIdx);
      gmemDqSfsAct += ((mCtaIdxZ) * ((params.k) / (int32_t{128}))) *
                      (int32_t{params.ptrTotalNumPaddedTokens[int32_t{0}]});
      float const* gmemDqSfsWeights;
      gmemDqSfsWeights = reinterpret_cast<float const*>(params.ptrSfA) + (int32_t{0});
      gmemDqSfsWeights +=
        ((int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}) * (params.tileStridePerBatch) +
         (mCtaIdxX)) *
          ((params.k) / (int32_t{128})) +
        ((mCtaIdxZ) * ((params.k) / (int32_t{128})));
      //
      // smemDeepSeekSfAb [HoistProdTryAcquire].
      //
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
        // gmemSfB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK6;
        { tileOffsetK6 = int32_t{0}; }
        //
        // smemDeepSeekSfAb [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          {
            smemDeepSeekSfAbProdToken =
              smemDeepSeekSfAbDstStack.mPipeline.producer_try_acquire(smemDeepSeekSfAbProdState);
          }
        }
        {
          smemDeepSeekSfAbDstStack.mPipeline.producer_acquire(smemDeepSeekSfAbProdState,
                                                              smemDeepSeekSfAbProdToken);
        }
        //
        // smemDeepSeekSfAb [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK9;
        tileOffsetK9 = tileOffsetK5;
        {
          int32_t index{smemDeepSeekSfAbProdState.index()};
          float* dstSmemDqSfsAct{&smemDeepSeekSfAbDstSmem.mDqSfsAct[index][int32_t{0}]};
          dstSmemDqSfsAct += mWarpGrpThreadIdx;
          {
            float const* gmemDqSfsActAtIter{gmemDqSfsAct};
            float* dstSmemDqSfsActAtIter{dstSmemDqSfsAct};
            int32_t colIdx{colDqSfsAct};
            gmemDqSfsActAtIter += colIdx;
            dstSmemDqSfsActAtIter += int32_t{0};
            if ((colDqSfsAct) < (int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]})) {
              trtllm::dev::cpAsync(reinterpret_cast<uint32_t*>(dstSmemDqSfsActAtIter),
                                   reinterpret_cast<uint32_t const*>(gmemDqSfsActAtIter));
            }
          }
          {
            float const* gmemDqSfsActAtIter{gmemDqSfsAct};
            float* dstSmemDqSfsActAtIter{dstSmemDqSfsAct};
            int32_t colIdx{(colDqSfsAct) + (int32_t{32})};
            gmemDqSfsActAtIter += colIdx;
            dstSmemDqSfsActAtIter += int32_t{32};
            if (((colDqSfsAct) + (int32_t{32})) <
                (int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]})) {
              trtllm::dev::cpAsync(reinterpret_cast<uint32_t*>(dstSmemDqSfsActAtIter),
                                   reinterpret_cast<uint32_t const*>(gmemDqSfsActAtIter));
            }
          }
          float* dstSmemDqSfsWeights{&smemDeepSeekSfAbDstSmem.mDqSfsWeights[index][int32_t{0}]};
          if ((mWarpGrpThreadIdx) == (int32_t{0})) {
            trtllm::dev::cpAsync(dstSmemDqSfsWeights, gmemDqSfsWeights);
          }
          gmemDqSfsAct += int32_t{params.ptrTotalNumPaddedTokens[int32_t{0}]};
          ++gmemDqSfsWeights;
        }
        //
        // gmemSfA [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        //
        // gmemSfB [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        //
        // smemDeepSeekSfAb [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
      }
      //
      // Loop body.
      //
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset729 = int32_t{128}; loopOffset729 < loopEnd;
           loopOffset729 += int32_t{128}) {
        bool const isFirstLoopIter{(loopOffset729) == (int32_t{128})};
        bool const isLastLoopIter{((loopOffset729) + (int32_t{128})) >= (loopEnd)};
        //
        // smemDeepSeekSfAb [ProdPreCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset729) >= (int32_t{128})) {
        }
        //
        // smemDeepSeekSfAb [ProdCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset729) >= (int32_t{128})) {
          {
            smemDeepSeekSfAbDstStack.mPipeline.producer_commit(smemDeepSeekSfAbProdState);
          }
          ++smemDeepSeekSfAbProdState;
        }
        //
        // gmemSfA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK5;
        { tileOffsetK5 = loopOffset729; }
        //
        // gmemSfB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK6;
        { tileOffsetK6 = loopOffset729; }
        //
        // smemDeepSeekSfAb [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          if ((loopOffset729) >= (int32_t{0})) {
            smemDeepSeekSfAbProdToken =
              smemDeepSeekSfAbDstStack.mPipeline.producer_try_acquire(smemDeepSeekSfAbProdState);
          }
        }
        {
          smemDeepSeekSfAbDstStack.mPipeline.producer_acquire(smemDeepSeekSfAbProdState,
                                                              smemDeepSeekSfAbProdToken);
        }
        //
        // smemDeepSeekSfAb [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        int32_t tileOffsetK9;
        tileOffsetK9 = tileOffsetK5;
        {
          int32_t index{smemDeepSeekSfAbProdState.index()};
          float* dstSmemDqSfsAct{&smemDeepSeekSfAbDstSmem.mDqSfsAct[index][int32_t{0}]};
          dstSmemDqSfsAct += mWarpGrpThreadIdx;
          {
            float const* gmemDqSfsActAtIter{gmemDqSfsAct};
            float* dstSmemDqSfsActAtIter{dstSmemDqSfsAct};
            int32_t colIdx{colDqSfsAct};
            gmemDqSfsActAtIter += colIdx;
            dstSmemDqSfsActAtIter += int32_t{0};
            if ((colDqSfsAct) < (int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]})) {
              trtllm::dev::cpAsync(reinterpret_cast<uint32_t*>(dstSmemDqSfsActAtIter),
                                   reinterpret_cast<uint32_t const*>(gmemDqSfsActAtIter));
            }
          }
          {
            float const* gmemDqSfsActAtIter{gmemDqSfsAct};
            float* dstSmemDqSfsActAtIter{dstSmemDqSfsAct};
            int32_t colIdx{(colDqSfsAct) + (int32_t{32})};
            gmemDqSfsActAtIter += colIdx;
            dstSmemDqSfsActAtIter += int32_t{32};
            if (((colDqSfsAct) + (int32_t{32})) <
                (int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]})) {
              trtllm::dev::cpAsync(reinterpret_cast<uint32_t*>(dstSmemDqSfsActAtIter),
                                   reinterpret_cast<uint32_t const*>(gmemDqSfsActAtIter));
            }
          }
          float* dstSmemDqSfsWeights{&smemDeepSeekSfAbDstSmem.mDqSfsWeights[index][int32_t{0}]};
          if ((mWarpGrpThreadIdx) == (int32_t{0})) {
            trtllm::dev::cpAsync(dstSmemDqSfsWeights, gmemDqSfsWeights);
          }
          gmemDqSfsAct += int32_t{params.ptrTotalNumPaddedTokens[int32_t{0}]};
          ++gmemDqSfsWeights;
        }
        //
        // gmemSfA [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        //
        // gmemSfB [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        //
        // smemDeepSeekSfAb [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
      }
      //
      // Unrolled tail iter 0.
      //
      if ((loopEnd) > (int32_t{0})) {
        //
        // smemDeepSeekSfAb [ProdPreCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {}
        //
        // smemDeepSeekSfAb [ProdCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { smemDeepSeekSfAbDstStack.mPipeline.producer_commit(smemDeepSeekSfAbProdState); }
          ++smemDeepSeekSfAbProdState;
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
      // gmemSfB [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
      //
      // workId [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
      //
      // smemDeepSeekSfAb [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
    return ((state.mWarpIdx) >= (int32_t{8})) && ((state.mWarpIdx) < (int32_t{9}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 Mma0Stack& mma0DstStack,
                                 Mma1Stack& mma1DstStack,
                                 SmemASmem& smemASrcSmem,
                                 SmemAStack& smemASrcStack,
                                 SmemBSmem& smemBSrcSmem,
                                 SmemBStack& smemBSrcStack,
                                 WorkIdSmem& workIdSrcSmem,
                                 WorkIdStack& workIdSrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<48>{});
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      6,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemAConsState{};
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      6,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemAConsReleaseState{};
    int32_t smemAConsToken{int32_t{0}};
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      6,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemBConsState{};
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      6,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemBConsReleaseState{};
    int32_t smemBConsToken{int32_t{0}};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsState{};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsReleaseState{};
    int32_t workIdConsToken{int32_t{0}};
    trtllm::dev::CutlassUmmaAsyncPipeline<4,
                                          cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState mma0ProdState{int32_t{0}, int32_t{1}, int32_t{0}};
    int32_t mma0ProdStateStub{int32_t{1}};
    int32_t mma0ProdToken{int32_t{1}};
    trtllm::dev::CutlassUmmaAsyncPipeline<4,
                                          cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState mma1ProdState{int32_t{0}, int32_t{1}, int32_t{0}};
    int32_t mma1ProdStateStub{int32_t{1}};
    int32_t mma1ProdToken{int32_t{1}};
    do {
      int32_t paddedPerCtaK{(((params.k) + (int32_t{127})) / (int32_t{128})) * (int32_t{128})};
      int32_t loopEnd{paddedPerCtaK};
      if ((mCtaIdxY) >= (state.mNumNonExitingCtas)) {
        goto ExitTileWithoutSignalingLabel;
      }
      //
      // mma0 [HoistProdTryAcquire].
      //
      if ((int32_t{0}) < (loopEnd)) {
        mma0ProdToken = mma0DstStack.mPipeline.producer_try_acquire(mma0ProdState);
      }
      //
      // mma1 [HoistProdTryAcquire].
      //
      if ((int32_t{0}) < (loopEnd)) {
        mma1ProdToken = mma1DstStack.mPipeline.producer_try_acquire(mma1ProdState);
      }
      //
      // Loop body.
      //
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset885 = int32_t{0}; loopOffset885 < loopEnd;
           loopOffset885 += int32_t{128}) {
        bool const isFirstLoopIter{(loopOffset885) == (int32_t{0})};
        bool const isLastLoopIter{((loopOffset885) + (int32_t{128})) >= (loopEnd)};
        //
        // smemA [ConsTryWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        { smemAConsToken = smemASrcStack.mPipeline.consumer_try_wait(smemAConsState); }
        //
        // smemB [ConsTryWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        { smemBConsToken = smemBSrcStack.mPipeline.consumer_try_wait(smemBConsState); }
        //
        // mma0 [ProdTryAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          if ((loopOffset885) >= (int32_t{0})) {
            mma0ProdToken = mma0DstStack.mPipeline.producer_try_acquire(mma0ProdState);
          }
        }
        //
        // mma1 [ProdTryAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          if ((loopOffset885) >= (int32_t{0})) {
            mma1ProdToken = mma1DstStack.mPipeline.producer_try_acquire(mma1ProdState);
          }
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
        cutlass::float_e4m3_t* smemPtrA7;
        {
          int32_t index{smemAConsState.index()};
          int8_t* smemBytesBasePtrA;
          smemBytesBasePtrA = reinterpret_cast<int8_t*>(smemASrcStack.mDepSmemPtr2) + (int32_t{0});
          int8_t* smemBytesStagePtrA;
          smemBytesStagePtrA = smemBytesBasePtrA + ((index) * (int32_t{16384}));
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
            reinterpret_cast<int8_t*>(smemBSrcStack.mDepSmemPtr2) + (int32_t{98304});
          int8_t* smemBytesStagePtrB;
          smemBytesStagePtrB = smemBytesBasePtrB + ((index) * (int32_t{8192}));
          smemPtrB8 = reinterpret_cast<cutlass::float_e4m3_t*>(smemBytesStagePtrB) + (int32_t{0});
          ++smemBConsState;
        }
        //
        // mma0 [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{8388608}].
        //
        {
          mma0DstStack.mPipeline.producer_acquire(mma0ProdState, mma0ProdToken);
          if (((loopOffset885) + (int32_t{128})) < (loopEnd)) {
            mma0ProdToken = mma0DstStack.mPipeline.producer_try_acquire(
              trtllm::dev::makePipelineState(mma0ProdState, int32_t{1}));
          }
        }
        //
        // mma1 [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{8389632}].
        //
        {
          mma1DstStack.mPipeline.producer_acquire(mma1ProdState, mma1ProdToken);
          if (((loopOffset885) + (int32_t{128})) < (loopEnd)) {
            mma1ProdToken = mma1DstStack.mPipeline.producer_try_acquire(
              trtllm::dev::makePipelineState(mma1ProdState, int32_t{1}));
          }
        }
        //
        // mma0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_e4m3_t* smemPtrA11;
        cutlass::float_e4m3_t* smemPtrB11;
        smemPtrA11 = smemPtrA7;
        smemPtrB11 = smemPtrB8;
        {
          int32_t index{mma0ProdState.index()};
          uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{64})))};
          uint32_t ptrTmemOffsetD{ptrTmemD};
          cutlass::float_e4m3_t* ptrWithOffsetSmemA{(smemPtrA11 + int32_t{0})};
          cutlass::float_e4m3_t* ptrWithOffsetSmemB{(smemPtrB11 + int32_t{0})};
          {
            uint32_t tmemPtrD{ptrTmemOffsetD};
            //
            // leadingDimInBytes = 8192, strideInBytes = 1024, swizzleMode = 1.
            //
            uint64_t smemDescA{
              trtllm::dev::createSmemDesc(ptrWithOffsetSmemA,
                                          uint32_t{0x2000000 /*hi=512, lo=0*/},
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
            uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{64},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma(cuda_ptx::kind_f8f6f4,
                                    cuda_ptx::cta_group_1,
                                    tmemPtrD,
                                    smemDescA,
                                    smemDescB,
                                    utcmmaDesc_0_0_0,
                                    bool{false});
            }
            //
            // MMA inst for mi=0 ni=0 ki=1.
            //
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{64},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma(cuda_ptx::kind_f8f6f4,
                                    cuda_ptx::cta_group_1,
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
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{64},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma(cuda_ptx::kind_f8f6f4,
                                    cuda_ptx::cta_group_1,
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
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{64},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma(cuda_ptx::kind_f8f6f4,
                                    cuda_ptx::cta_group_1,
                                    tmemPtrD,
                                    smemDescA,
                                    smemDescB,
                                    utcmmaDesc_0_0_3,
                                    bool{true});
            }
          }
        }
        //
        // mma0 [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { mma0DstStack.mPipeline.producer_commit(mma0ProdState); }
          mma0ProdState += int32_t{2};
        }
        //
        // mma1 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_e4m3_t* smemPtrA12;
        cutlass::float_e4m3_t* smemPtrB12;
        smemPtrA12 = smemPtrA7;
        smemPtrB12 = smemPtrB8;
        {
          int32_t index{mma1ProdState.index()};
          uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{64})))};
          uint32_t ptrTmemOffsetD{(ptrTmemD) + (uint32_t{1048576})};
          cutlass::float_e4m3_t* ptrWithOffsetSmemA{(smemPtrA12 + int32_t{8192})};
          cutlass::float_e4m3_t* ptrWithOffsetSmemB{(smemPtrB12 + int32_t{0})};
          {
            uint32_t tmemPtrD{ptrTmemOffsetD};
            //
            // leadingDimInBytes = 8192, strideInBytes = 1024, swizzleMode = 1.
            //
            uint64_t smemDescA{
              trtllm::dev::createSmemDesc(ptrWithOffsetSmemA,
                                          uint32_t{0x2000000 /*hi=512, lo=0*/},
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
            uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{64},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma(cuda_ptx::kind_f8f6f4,
                                    cuda_ptx::cta_group_1,
                                    tmemPtrD,
                                    smemDescA,
                                    smemDescB,
                                    utcmmaDesc_0_0_0,
                                    bool{false});
            }
            //
            // MMA inst for mi=0 ni=0 ki=1.
            //
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{64},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma(cuda_ptx::kind_f8f6f4,
                                    cuda_ptx::cta_group_1,
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
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{64},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma(cuda_ptx::kind_f8f6f4,
                                    cuda_ptx::cta_group_1,
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
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{64},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma(cuda_ptx::kind_f8f6f4,
                                    cuda_ptx::cta_group_1,
                                    tmemPtrD,
                                    smemDescA,
                                    smemDescB,
                                    utcmmaDesc_0_0_3,
                                    bool{true});
            }
          }
        }
        //
        // mma1 [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { mma1DstStack.mPipeline.producer_commit(mma1ProdState); }
          mma1ProdState += int32_t{2};
        }
        //
        // smemA [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset885) >= (int32_t{0})) {
          {
            smemASrcStack.mPipeline.consumer_release(smemAConsReleaseState);
          }
          ++smemAConsReleaseState;
        }
        //
        // smemB [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset885) >= (int32_t{0})) {
          {
            smemBSrcStack.mPipeline.consumer_release(smemBConsReleaseState);
          }
          ++smemBConsReleaseState;
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
      {
        uint32_t stateStub_{mma1ProdState.count()};
        mma1ProdState =
          trtllm::dev::makePipelineState(trtllm::dev::makeProdStartStateFrom(mma1ProdState),
                                         mma1ProdStateStub);
        mma1ProdStateStub = stateStub_;
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
  int32_t const mLdtm16dp256bitTmemColIdx;
  int32_t const mLdtm16dp256bitTmemRowIdx;
  cutlass::Array<float, 32> frg13;
  int32_t const mGridDimX;
  inline __device__ EpilogueTask0(KernelParams const& params,
                                  KernelState const& state,
                                  int32_t warpGrpStart)
    : mWarpGrpWarpIdx{(state.mWarpIdx) - (warpGrpStart)}
    , mLaneIdx{(state.mThreadIdx) % (int32_t{32})}
    , mWarpGrp4WarpIdx{mWarpGrpWarpIdx}
    , mWarpGrp4Idx{int32_t{0}}
    , mWarpRowIdx{(mWarpGrp4WarpIdx) * (int32_t{16})}
    , mQuadRowIdx{((mLaneIdx) / (int32_t{4})) * (int32_t{2})}
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
    , mLdtm16dp256bitTmemColIdx{trtllm::dev::ldst16dp256bitTmemColIdx((mWarpGrpThreadIdx) %
                                                                      (int32_t{128}))}
    , mLdtm16dp256bitTmemRowIdx{trtllm::dev::ldst16dp256bitTmemRowIdx<int32_t{16}>(
        (mWarpGrpThreadIdx) % (int32_t{128}))}
    , mGridDimX{reinterpret_cast<int32_t const&>(gridDim.x)} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{0})) && ((state.mWarpIdx) < (int32_t{4}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 GmemC0Smem& gmemC0DstSmem,
                                 GmemC0Stack& gmemC0DstStack,
                                 Mma0Stack& mma0SrcStack,
                                 SmemDeepSeekSfAbSmem& smemDeepSeekSfAbSrcSmem,
                                 SmemDeepSeekSfAbStack& smemDeepSeekSfAbSrcStack,
                                 WorkIdSmem& workIdSrcSmem,
                                 WorkIdStack& workIdSrcStack) {
    cuda_ptx::setmaxnreg_inc(cuda_ptx::n32_t<160>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassUmmaAsyncPipeline<
      4,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState mma0ConsState{};
    int32_t mma0ConsStateStub{int32_t{1}};
    int32_t mma0ConsToken{int32_t{0}};
    trtllm::dev::CutlassCpAsyncPipeline<6>::PipelineState smemDeepSeekSfAbConsState{};
    int32_t smemDeepSeekSfAbConsToken{int32_t{0}};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsState{};
    int32_t workIdConsToken{int32_t{0}};
    do {
      int32_t paddedPerCtaK{(((params.k) + (int32_t{127})) / (int32_t{128})) * (int32_t{128})};
      int32_t loopEnd{paddedPerCtaK};
      if ((mCtaIdxY) >= (state.mNumNonExitingCtas)) {
        goto ExitTileWithoutSignalingLabel;
      }
      uint32_t tmemBaseWithStageOffset;
      tmemBaseWithStageOffset = mTmemBaseOffset;
      mBatchIdx = int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]};
      mBatchLimit = int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]};
      cutlass::Array<float, 32> regsAcc00;
      cutlass::Array<float, 32> regsPartialAcc00;
      CUTLASS_PRAGMA_UNROLL
      for (int32_t loopOffset1128 = int32_t{0}; loopOffset1128 < int32_t{32}; ++loopOffset1128) {
        regsAcc00[loopOffset1128] = float{0};
      }
      //
      // Unrolled head iter 0.
      //
      if ((int32_t{0}) < (loopEnd)) {
        //
        // mma0 [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { mma0ConsToken = mma0SrcStack.mPipeline.consumer_try_wait(mma0ConsState); }
          mma0SrcStack.mPipeline.consumer_wait(mma0ConsState, mma0ConsToken);
        }
        //
        // mma0 [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        uint32_t tmemBaseWithStageOffset11;
        {
          int32_t index{mma0ConsState.index()};
          uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{64})))};
          uint32_t ptrTmemOffsetD{ptrTmemD};
          tmemBaseWithStageOffset11 = ptrTmemOffsetD;
        }
        //
        // gmemC0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        uint32_t tmemBaseWithStageOffset13;
        tmemBaseWithStageOffset13 = tmemBaseWithStageOffset11;
        {
          {
            uint32_t tmemBasePtr{tmemBaseWithStageOffset13};
            uint32_t(&dstSlice0)[32]{
              reinterpret_cast<uint32_t(&)[32]>(regsPartialAcc00[int32_t{0}])};
            cuda_ptx::tcgen05_ld_16x256b(dstSlice0, tmemBasePtr);
          }
          cutlass::arch::fence_view_async_tmem_load();
        }
        //
        // mma0 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { mma0SrcStack.mPipeline.consumer_release(mma0ConsState); }
          mma0ConsState += int32_t{2};
        }
        //
        // smemDeepSeekSfAb [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          {
            smemDeepSeekSfAbConsToken =
              smemDeepSeekSfAbSrcStack.mPipeline.consumer_try_wait(smemDeepSeekSfAbConsState);
          }
          smemDeepSeekSfAbSrcStack.mPipeline.consumer_wait(smemDeepSeekSfAbConsState,
                                                           smemDeepSeekSfAbConsToken);
        }
        //
        // smemDeepSeekSfAb [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        float* smemDqSfsAct9;
        float* smemDqSfsWeights9;
        {
          int32_t index{smemDeepSeekSfAbConsState.index()};
          smemDqSfsAct9 = &smemDeepSeekSfAbSrcSmem.mDqSfsAct[index][int32_t{0}];
          smemDqSfsWeights9 = &smemDeepSeekSfAbSrcSmem.mDqSfsWeights[index][int32_t{0}];
        }
        //
        // gmemC0 [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
        //
        float* smemDqSfsAct13;
        float* smemDqSfsW13;
        tmemBaseWithStageOffset13 = tmemBaseWithStageOffset11;
        smemDqSfsAct13 = smemDqSfsAct9;
        smemDqSfsW13 = smemDqSfsWeights9;
        {
          float dqSfW{smemDqSfsW13[int32_t{0}]};
          float dqSfAb0{(smemDqSfsAct13[mLdtm16dp256bitTmemColIdx]) * (dqSfW)};
          float dqSfAb1{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{1})]) * (dqSfW)};
          float dqSfAb2{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{8})]) * (dqSfW)};
          float dqSfAb3{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{9})]) * (dqSfW)};
          float dqSfAb4{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{16})]) * (dqSfW)};
          float dqSfAb5{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{17})]) * (dqSfW)};
          float dqSfAb6{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{24})]) * (dqSfW)};
          float dqSfAb7{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{25})]) * (dqSfW)};
          float dqSfAb8{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{32})]) * (dqSfW)};
          float dqSfAb9{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{33})]) * (dqSfW)};
          float dqSfAb10{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{40})]) * (dqSfW)};
          float dqSfAb11{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{41})]) * (dqSfW)};
          float dqSfAb12{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{48})]) * (dqSfW)};
          float dqSfAb13{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{49})]) * (dqSfW)};
          float dqSfAb14{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{56})]) * (dqSfW)};
          float dqSfAb15{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{57})]) * (dqSfW)};
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb0, dqSfAb1};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{0}], regsAcc00[int32_t{1}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{0}],
                                         regsPartialAcc00[int32_t{1}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{0}] = tmp[int32_t{0}];
            regsAcc00[int32_t{1}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb0, dqSfAb1};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{2}], regsAcc00[int32_t{3}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{2}],
                                         regsPartialAcc00[int32_t{3}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{2}] = tmp[int32_t{0}];
            regsAcc00[int32_t{3}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb2, dqSfAb3};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{4}], regsAcc00[int32_t{5}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{4}],
                                         regsPartialAcc00[int32_t{5}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{4}] = tmp[int32_t{0}];
            regsAcc00[int32_t{5}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb2, dqSfAb3};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{6}], regsAcc00[int32_t{7}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{6}],
                                         regsPartialAcc00[int32_t{7}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{6}] = tmp[int32_t{0}];
            regsAcc00[int32_t{7}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb4, dqSfAb5};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{8}], regsAcc00[int32_t{9}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{8}],
                                         regsPartialAcc00[int32_t{9}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{8}] = tmp[int32_t{0}];
            regsAcc00[int32_t{9}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb4, dqSfAb5};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{10}], regsAcc00[int32_t{11}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{10}],
                                         regsPartialAcc00[int32_t{11}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{10}] = tmp[int32_t{0}];
            regsAcc00[int32_t{11}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb6, dqSfAb7};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{12}], regsAcc00[int32_t{13}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{12}],
                                         regsPartialAcc00[int32_t{13}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{12}] = tmp[int32_t{0}];
            regsAcc00[int32_t{13}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb6, dqSfAb7};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{14}], regsAcc00[int32_t{15}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{14}],
                                         regsPartialAcc00[int32_t{15}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{14}] = tmp[int32_t{0}];
            regsAcc00[int32_t{15}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb8, dqSfAb9};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{16}], regsAcc00[int32_t{17}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{16}],
                                         regsPartialAcc00[int32_t{17}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{16}] = tmp[int32_t{0}];
            regsAcc00[int32_t{17}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb8, dqSfAb9};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{18}], regsAcc00[int32_t{19}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{18}],
                                         regsPartialAcc00[int32_t{19}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{18}] = tmp[int32_t{0}];
            regsAcc00[int32_t{19}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb10, dqSfAb11};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{20}], regsAcc00[int32_t{21}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{20}],
                                         regsPartialAcc00[int32_t{21}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{20}] = tmp[int32_t{0}];
            regsAcc00[int32_t{21}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb10, dqSfAb11};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{22}], regsAcc00[int32_t{23}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{22}],
                                         regsPartialAcc00[int32_t{23}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{22}] = tmp[int32_t{0}];
            regsAcc00[int32_t{23}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb12, dqSfAb13};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{24}], regsAcc00[int32_t{25}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{24}],
                                         regsPartialAcc00[int32_t{25}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{24}] = tmp[int32_t{0}];
            regsAcc00[int32_t{25}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb12, dqSfAb13};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{26}], regsAcc00[int32_t{27}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{26}],
                                         regsPartialAcc00[int32_t{27}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{26}] = tmp[int32_t{0}];
            regsAcc00[int32_t{27}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb14, dqSfAb15};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{28}], regsAcc00[int32_t{29}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{28}],
                                         regsPartialAcc00[int32_t{29}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{28}] = tmp[int32_t{0}];
            regsAcc00[int32_t{29}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb14, dqSfAb15};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{30}], regsAcc00[int32_t{31}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{30}],
                                         regsPartialAcc00[int32_t{31}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{30}] = tmp[int32_t{0}];
            regsAcc00[int32_t{31}] = tmp[int32_t{1}];
          }
        }
        //
        // smemDeepSeekSfAb [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { smemDeepSeekSfAbSrcStack.mPipeline.consumer_release(smemDeepSeekSfAbConsState); }
          ++smemDeepSeekSfAbConsState;
        }
        //
        // gmemC0 [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
      }
      //
      // Loop body.
      //
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset1320 = int32_t{128}; loopOffset1320 < loopEnd;
           loopOffset1320 += int32_t{128}) {
        bool const isLastLoopIter{((loopOffset1320) + (int32_t{128})) >= (loopEnd)};
        //
        // mma0 [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { mma0ConsToken = mma0SrcStack.mPipeline.consumer_try_wait(mma0ConsState); }
          mma0SrcStack.mPipeline.consumer_wait(mma0ConsState, mma0ConsToken);
        }
        //
        // mma0 [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        uint32_t tmemBaseWithStageOffset11;
        {
          int32_t index{mma0ConsState.index()};
          uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{64})))};
          uint32_t ptrTmemOffsetD{ptrTmemD};
          tmemBaseWithStageOffset11 = ptrTmemOffsetD;
        }
        //
        // gmemC0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        uint32_t tmemBaseWithStageOffset13;
        tmemBaseWithStageOffset13 = tmemBaseWithStageOffset11;
        {
          {
            uint32_t tmemBasePtr{tmemBaseWithStageOffset13};
            uint32_t(&dstSlice0)[32]{
              reinterpret_cast<uint32_t(&)[32]>(regsPartialAcc00[int32_t{0}])};
            cuda_ptx::tcgen05_ld_16x256b(dstSlice0, tmemBasePtr);
          }
          cutlass::arch::fence_view_async_tmem_load();
        }
        //
        // mma0 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { mma0SrcStack.mPipeline.consumer_release(mma0ConsState); }
          mma0ConsState += int32_t{2};
        }
        //
        // smemDeepSeekSfAb [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          {
            smemDeepSeekSfAbConsToken =
              smemDeepSeekSfAbSrcStack.mPipeline.consumer_try_wait(smemDeepSeekSfAbConsState);
          }
          smemDeepSeekSfAbSrcStack.mPipeline.consumer_wait(smemDeepSeekSfAbConsState,
                                                           smemDeepSeekSfAbConsToken);
        }
        //
        // smemDeepSeekSfAb [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        float* smemDqSfsAct9;
        float* smemDqSfsWeights9;
        {
          int32_t index{smemDeepSeekSfAbConsState.index()};
          smemDqSfsAct9 = &smemDeepSeekSfAbSrcSmem.mDqSfsAct[index][int32_t{0}];
          smemDqSfsWeights9 = &smemDeepSeekSfAbSrcSmem.mDqSfsWeights[index][int32_t{0}];
        }
        //
        // gmemC0 [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
        //
        float* smemDqSfsAct13;
        float* smemDqSfsW13;
        tmemBaseWithStageOffset13 = tmemBaseWithStageOffset11;
        smemDqSfsAct13 = smemDqSfsAct9;
        smemDqSfsW13 = smemDqSfsWeights9;
        {
          float dqSfW{smemDqSfsW13[int32_t{0}]};
          float dqSfAb0{(smemDqSfsAct13[mLdtm16dp256bitTmemColIdx]) * (dqSfW)};
          float dqSfAb1{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{1})]) * (dqSfW)};
          float dqSfAb2{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{8})]) * (dqSfW)};
          float dqSfAb3{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{9})]) * (dqSfW)};
          float dqSfAb4{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{16})]) * (dqSfW)};
          float dqSfAb5{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{17})]) * (dqSfW)};
          float dqSfAb6{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{24})]) * (dqSfW)};
          float dqSfAb7{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{25})]) * (dqSfW)};
          float dqSfAb8{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{32})]) * (dqSfW)};
          float dqSfAb9{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{33})]) * (dqSfW)};
          float dqSfAb10{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{40})]) * (dqSfW)};
          float dqSfAb11{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{41})]) * (dqSfW)};
          float dqSfAb12{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{48})]) * (dqSfW)};
          float dqSfAb13{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{49})]) * (dqSfW)};
          float dqSfAb14{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{56})]) * (dqSfW)};
          float dqSfAb15{(smemDqSfsAct13[(mLdtm16dp256bitTmemColIdx) + (int32_t{57})]) * (dqSfW)};
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb0, dqSfAb1};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{0}], regsAcc00[int32_t{1}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{0}],
                                         regsPartialAcc00[int32_t{1}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{0}] = tmp[int32_t{0}];
            regsAcc00[int32_t{1}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb0, dqSfAb1};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{2}], regsAcc00[int32_t{3}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{2}],
                                         regsPartialAcc00[int32_t{3}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{2}] = tmp[int32_t{0}];
            regsAcc00[int32_t{3}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb2, dqSfAb3};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{4}], regsAcc00[int32_t{5}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{4}],
                                         regsPartialAcc00[int32_t{5}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{4}] = tmp[int32_t{0}];
            regsAcc00[int32_t{5}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb2, dqSfAb3};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{6}], regsAcc00[int32_t{7}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{6}],
                                         regsPartialAcc00[int32_t{7}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{6}] = tmp[int32_t{0}];
            regsAcc00[int32_t{7}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb4, dqSfAb5};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{8}], regsAcc00[int32_t{9}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{8}],
                                         regsPartialAcc00[int32_t{9}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{8}] = tmp[int32_t{0}];
            regsAcc00[int32_t{9}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb4, dqSfAb5};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{10}], regsAcc00[int32_t{11}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{10}],
                                         regsPartialAcc00[int32_t{11}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{10}] = tmp[int32_t{0}];
            regsAcc00[int32_t{11}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb6, dqSfAb7};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{12}], regsAcc00[int32_t{13}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{12}],
                                         regsPartialAcc00[int32_t{13}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{12}] = tmp[int32_t{0}];
            regsAcc00[int32_t{13}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb6, dqSfAb7};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{14}], regsAcc00[int32_t{15}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{14}],
                                         regsPartialAcc00[int32_t{15}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{14}] = tmp[int32_t{0}];
            regsAcc00[int32_t{15}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb8, dqSfAb9};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{16}], regsAcc00[int32_t{17}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{16}],
                                         regsPartialAcc00[int32_t{17}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{16}] = tmp[int32_t{0}];
            regsAcc00[int32_t{17}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb8, dqSfAb9};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{18}], regsAcc00[int32_t{19}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{18}],
                                         regsPartialAcc00[int32_t{19}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{18}] = tmp[int32_t{0}];
            regsAcc00[int32_t{19}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb10, dqSfAb11};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{20}], regsAcc00[int32_t{21}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{20}],
                                         regsPartialAcc00[int32_t{21}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{20}] = tmp[int32_t{0}];
            regsAcc00[int32_t{21}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb10, dqSfAb11};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{22}], regsAcc00[int32_t{23}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{22}],
                                         regsPartialAcc00[int32_t{23}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{22}] = tmp[int32_t{0}];
            regsAcc00[int32_t{23}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb12, dqSfAb13};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{24}], regsAcc00[int32_t{25}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{24}],
                                         regsPartialAcc00[int32_t{25}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{24}] = tmp[int32_t{0}];
            regsAcc00[int32_t{25}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb12, dqSfAb13};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{26}], regsAcc00[int32_t{27}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{26}],
                                         regsPartialAcc00[int32_t{27}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{26}] = tmp[int32_t{0}];
            regsAcc00[int32_t{27}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb14, dqSfAb15};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{28}], regsAcc00[int32_t{29}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{28}],
                                         regsPartialAcc00[int32_t{29}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{28}] = tmp[int32_t{0}];
            regsAcc00[int32_t{29}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb14, dqSfAb15};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{30}], regsAcc00[int32_t{31}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{30}],
                                         regsPartialAcc00[int32_t{31}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{30}] = tmp[int32_t{0}];
            regsAcc00[int32_t{31}] = tmp[int32_t{1}];
          }
        }
        //
        // smemDeepSeekSfAb [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { smemDeepSeekSfAbSrcStack.mPipeline.consumer_release(smemDeepSeekSfAbConsState); }
          ++smemDeepSeekSfAbConsState;
        }
        //
        // gmemC0 [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
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
          cuda_ptx::cp_async_bulk_wait_group_read(cuda_ptx::n32_t<0>{});
          trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{6}) + (mWarpGrp4Idx));
          //
          // Store to Smem TmaAsyncGmemC.
          //
          int8_t* ptrSmemBase;
          cutlass::bfloat16_t* ptrSmem;
          ptrSmemBase = reinterpret_cast<int8_t*>(gmemC0DstStack.mDepSmemPtr2) +
                        ((mWarpGrp4Idx) * (int32_t{8192}) + (int32_t{147456}));
          ptrSmem = reinterpret_cast<cutlass::bfloat16_t*>(ptrSmemBase) + (int32_t{0});
          //
          // Smem store idxM=0 idxN=0.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{((mBaseTmemCol) * (int32_t{64}) + (mBaseRowIdx)) /
                                       (int32_t{64})};
              int32_t const smemOffsetInBytes{
                (((mBaseTmemCol) * (int32_t{64}) + (mBaseRowIdx)) * (int32_t{16})) / (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{0}], regsAcc00[int32_t{2}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=1.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{1})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{1})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{1}], regsAcc00[int32_t{3}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=2.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{8})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{8})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{4}], regsAcc00[int32_t{6}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=3.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{9})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{9})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{5}], regsAcc00[int32_t{7}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=4.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{16})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{16})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{8}], regsAcc00[int32_t{10}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=5.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{17})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{17})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{9}], regsAcc00[int32_t{11}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=6.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{24})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{24})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{12}], regsAcc00[int32_t{14}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=7.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{25})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{25})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{13}], regsAcc00[int32_t{15}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=8.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{32})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{32})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{16}], regsAcc00[int32_t{18}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=9.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{33})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{33})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{17}], regsAcc00[int32_t{19}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=10.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{40})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{40})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{20}], regsAcc00[int32_t{22}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=11.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{41})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{41})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{21}], regsAcc00[int32_t{23}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=12.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{48})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{48})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{24}], regsAcc00[int32_t{26}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=13.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{49})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{49})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{25}], regsAcc00[int32_t{27}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=14.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{56})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{56})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{28}], regsAcc00[int32_t{30}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=15.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{57})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{57})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{29}], regsAcc00[int32_t{31}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
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
                          ((mWarpGrp4Idx) * (int32_t{8192}) + (int32_t{147456}));
            ptrSmem = reinterpret_cast<cutlass::bfloat16_t*>(ptrSmemBase) + (int32_t{0});
            int32_t coords[4];
            coords[int32_t{0}] = ((int32_t{2}) * (mCtaIdxX)) * (int32_t{64});
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
          trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{6}) + (mWarpGrp4Idx));
        }
        {
          //
          // Skip all-reduce if on single device.
          //
        }
      }
    ExitTileWithSignalingLabel:
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
struct EpilogueTask1 {
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
  int32_t const mLdtm16dp256bitTmemColIdx;
  int32_t const mLdtm16dp256bitTmemRowIdx;
  cutlass::Array<float, 32> frg14;
  int32_t const mGridDimX;
  inline __device__ EpilogueTask1(KernelParams const& params,
                                  KernelState const& state,
                                  int32_t warpGrpStart)
    : mWarpGrpWarpIdx{(state.mWarpIdx) - (warpGrpStart)}
    , mLaneIdx{(state.mThreadIdx) % (int32_t{32})}
    , mWarpGrp4WarpIdx{mWarpGrpWarpIdx}
    , mWarpGrp4Idx{int32_t{0}}
    , mWarpRowIdx{(mWarpGrp4WarpIdx) * (int32_t{16})}
    , mQuadRowIdx{((mLaneIdx) / (int32_t{4})) * (int32_t{2})}
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
    , mLdtm16dp256bitTmemColIdx{trtllm::dev::ldst16dp256bitTmemColIdx((mWarpGrpThreadIdx) %
                                                                      (int32_t{128}))}
    , mLdtm16dp256bitTmemRowIdx{trtllm::dev::ldst16dp256bitTmemRowIdx<int32_t{16}>(
        (mWarpGrpThreadIdx) % (int32_t{128}))}
    , mGridDimX{reinterpret_cast<int32_t const&>(gridDim.x)} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{4})) && ((state.mWarpIdx) < (int32_t{8}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 GmemC1Smem& gmemC1DstSmem,
                                 GmemC1Stack& gmemC1DstStack,
                                 Mma1Stack& mma1SrcStack,
                                 SmemDeepSeekSfAbSmem& smemDeepSeekSfAbSrcSmem,
                                 SmemDeepSeekSfAbStack& smemDeepSeekSfAbSrcStack,
                                 WorkIdSmem& workIdSrcSmem,
                                 WorkIdStack& workIdSrcStack) {
    cuda_ptx::setmaxnreg_inc(cuda_ptx::n32_t<160>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassUmmaAsyncPipeline<
      4,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState mma1ConsState{};
    int32_t mma1ConsStateStub{int32_t{1}};
    int32_t mma1ConsToken{int32_t{0}};
    trtllm::dev::CutlassCpAsyncPipeline<6>::PipelineState smemDeepSeekSfAbConsState{};
    int32_t smemDeepSeekSfAbConsToken{int32_t{0}};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsState{};
    int32_t workIdConsToken{int32_t{0}};
    do {
      int32_t paddedPerCtaK{(((params.k) + (int32_t{127})) / (int32_t{128})) * (int32_t{128})};
      int32_t loopEnd{paddedPerCtaK};
      if ((mCtaIdxY) >= (state.mNumNonExitingCtas)) {
        goto ExitTileWithoutSignalingLabel;
      }
      uint32_t tmemBaseWithStageOffset;
      tmemBaseWithStageOffset = mTmemBaseOffset;
      mBatchIdx = int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]};
      mBatchLimit = int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]};
      cutlass::Array<float, 32> regsAcc00;
      cutlass::Array<float, 32> regsPartialAcc00;
      CUTLASS_PRAGMA_UNROLL
      for (int32_t loopOffset1866 = int32_t{0}; loopOffset1866 < int32_t{32}; ++loopOffset1866) {
        regsAcc00[loopOffset1866] = float{0};
      }
      //
      // Unrolled head iter 0.
      //
      if ((int32_t{0}) < (loopEnd)) {
        //
        // mma1 [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { mma1ConsToken = mma1SrcStack.mPipeline.consumer_try_wait(mma1ConsState); }
          mma1SrcStack.mPipeline.consumer_wait(mma1ConsState, mma1ConsToken);
        }
        //
        // mma1 [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        uint32_t tmemBaseWithStageOffset12;
        {
          int32_t index{mma1ConsState.index()};
          uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{64})))};
          uint32_t ptrTmemOffsetD{(ptrTmemD) + (uint32_t{1048576})};
          tmemBaseWithStageOffset12 = ptrTmemOffsetD;
        }
        //
        // gmemC1 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        uint32_t tmemBaseWithStageOffset14;
        tmemBaseWithStageOffset14 = tmemBaseWithStageOffset12;
        {
          {
            uint32_t tmemBasePtr{tmemBaseWithStageOffset14};
            uint32_t(&dstSlice0)[32]{
              reinterpret_cast<uint32_t(&)[32]>(regsPartialAcc00[int32_t{0}])};
            cuda_ptx::tcgen05_ld_16x256b(dstSlice0, tmemBasePtr);
          }
          cutlass::arch::fence_view_async_tmem_load();
        }
        //
        // mma1 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { mma1SrcStack.mPipeline.consumer_release(mma1ConsState); }
          mma1ConsState += int32_t{2};
        }
        //
        // smemDeepSeekSfAb [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          {
            smemDeepSeekSfAbConsToken =
              smemDeepSeekSfAbSrcStack.mPipeline.consumer_try_wait(smemDeepSeekSfAbConsState);
          }
          smemDeepSeekSfAbSrcStack.mPipeline.consumer_wait(smemDeepSeekSfAbConsState,
                                                           smemDeepSeekSfAbConsToken);
        }
        //
        // smemDeepSeekSfAb [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        float* smemDqSfsAct9;
        float* smemDqSfsWeights9;
        {
          int32_t index{smemDeepSeekSfAbConsState.index()};
          smemDqSfsAct9 = &smemDeepSeekSfAbSrcSmem.mDqSfsAct[index][int32_t{0}];
          smemDqSfsWeights9 = &smemDeepSeekSfAbSrcSmem.mDqSfsWeights[index][int32_t{0}];
        }
        //
        // gmemC1 [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
        //
        float* smemDqSfsAct14;
        float* smemDqSfsW14;
        tmemBaseWithStageOffset14 = tmemBaseWithStageOffset12;
        smemDqSfsAct14 = smemDqSfsAct9;
        smemDqSfsW14 = smemDqSfsWeights9;
        {
          float dqSfW{smemDqSfsW14[int32_t{0}]};
          float dqSfAb0{(smemDqSfsAct14[mLdtm16dp256bitTmemColIdx]) * (dqSfW)};
          float dqSfAb1{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{1})]) * (dqSfW)};
          float dqSfAb2{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{8})]) * (dqSfW)};
          float dqSfAb3{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{9})]) * (dqSfW)};
          float dqSfAb4{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{16})]) * (dqSfW)};
          float dqSfAb5{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{17})]) * (dqSfW)};
          float dqSfAb6{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{24})]) * (dqSfW)};
          float dqSfAb7{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{25})]) * (dqSfW)};
          float dqSfAb8{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{32})]) * (dqSfW)};
          float dqSfAb9{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{33})]) * (dqSfW)};
          float dqSfAb10{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{40})]) * (dqSfW)};
          float dqSfAb11{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{41})]) * (dqSfW)};
          float dqSfAb12{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{48})]) * (dqSfW)};
          float dqSfAb13{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{49})]) * (dqSfW)};
          float dqSfAb14{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{56})]) * (dqSfW)};
          float dqSfAb15{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{57})]) * (dqSfW)};
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb0, dqSfAb1};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{0}], regsAcc00[int32_t{1}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{0}],
                                         regsPartialAcc00[int32_t{1}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{0}] = tmp[int32_t{0}];
            regsAcc00[int32_t{1}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb0, dqSfAb1};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{2}], regsAcc00[int32_t{3}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{2}],
                                         regsPartialAcc00[int32_t{3}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{2}] = tmp[int32_t{0}];
            regsAcc00[int32_t{3}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb2, dqSfAb3};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{4}], regsAcc00[int32_t{5}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{4}],
                                         regsPartialAcc00[int32_t{5}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{4}] = tmp[int32_t{0}];
            regsAcc00[int32_t{5}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb2, dqSfAb3};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{6}], regsAcc00[int32_t{7}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{6}],
                                         regsPartialAcc00[int32_t{7}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{6}] = tmp[int32_t{0}];
            regsAcc00[int32_t{7}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb4, dqSfAb5};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{8}], regsAcc00[int32_t{9}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{8}],
                                         regsPartialAcc00[int32_t{9}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{8}] = tmp[int32_t{0}];
            regsAcc00[int32_t{9}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb4, dqSfAb5};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{10}], regsAcc00[int32_t{11}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{10}],
                                         regsPartialAcc00[int32_t{11}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{10}] = tmp[int32_t{0}];
            regsAcc00[int32_t{11}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb6, dqSfAb7};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{12}], regsAcc00[int32_t{13}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{12}],
                                         regsPartialAcc00[int32_t{13}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{12}] = tmp[int32_t{0}];
            regsAcc00[int32_t{13}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb6, dqSfAb7};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{14}], regsAcc00[int32_t{15}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{14}],
                                         regsPartialAcc00[int32_t{15}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{14}] = tmp[int32_t{0}];
            regsAcc00[int32_t{15}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb8, dqSfAb9};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{16}], regsAcc00[int32_t{17}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{16}],
                                         regsPartialAcc00[int32_t{17}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{16}] = tmp[int32_t{0}];
            regsAcc00[int32_t{17}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb8, dqSfAb9};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{18}], regsAcc00[int32_t{19}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{18}],
                                         regsPartialAcc00[int32_t{19}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{18}] = tmp[int32_t{0}];
            regsAcc00[int32_t{19}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb10, dqSfAb11};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{20}], regsAcc00[int32_t{21}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{20}],
                                         regsPartialAcc00[int32_t{21}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{20}] = tmp[int32_t{0}];
            regsAcc00[int32_t{21}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb10, dqSfAb11};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{22}], regsAcc00[int32_t{23}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{22}],
                                         regsPartialAcc00[int32_t{23}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{22}] = tmp[int32_t{0}];
            regsAcc00[int32_t{23}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb12, dqSfAb13};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{24}], regsAcc00[int32_t{25}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{24}],
                                         regsPartialAcc00[int32_t{25}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{24}] = tmp[int32_t{0}];
            regsAcc00[int32_t{25}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb12, dqSfAb13};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{26}], regsAcc00[int32_t{27}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{26}],
                                         regsPartialAcc00[int32_t{27}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{26}] = tmp[int32_t{0}];
            regsAcc00[int32_t{27}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb14, dqSfAb15};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{28}], regsAcc00[int32_t{29}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{28}],
                                         regsPartialAcc00[int32_t{29}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{28}] = tmp[int32_t{0}];
            regsAcc00[int32_t{29}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb14, dqSfAb15};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{30}], regsAcc00[int32_t{31}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{30}],
                                         regsPartialAcc00[int32_t{31}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{30}] = tmp[int32_t{0}];
            regsAcc00[int32_t{31}] = tmp[int32_t{1}];
          }
        }
        //
        // smemDeepSeekSfAb [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { smemDeepSeekSfAbSrcStack.mPipeline.consumer_release(smemDeepSeekSfAbConsState); }
          ++smemDeepSeekSfAbConsState;
        }
        //
        // gmemC1 [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
      }
      //
      // Loop body.
      //
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset2058 = int32_t{128}; loopOffset2058 < loopEnd;
           loopOffset2058 += int32_t{128}) {
        bool const isLastLoopIter{((loopOffset2058) + (int32_t{128})) >= (loopEnd)};
        //
        // mma1 [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { mma1ConsToken = mma1SrcStack.mPipeline.consumer_try_wait(mma1ConsState); }
          mma1SrcStack.mPipeline.consumer_wait(mma1ConsState, mma1ConsToken);
        }
        //
        // mma1 [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        uint32_t tmemBaseWithStageOffset12;
        {
          int32_t index{mma1ConsState.index()};
          uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{64})))};
          uint32_t ptrTmemOffsetD{(ptrTmemD) + (uint32_t{1048576})};
          tmemBaseWithStageOffset12 = ptrTmemOffsetD;
        }
        //
        // gmemC1 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        uint32_t tmemBaseWithStageOffset14;
        tmemBaseWithStageOffset14 = tmemBaseWithStageOffset12;
        {
          {
            uint32_t tmemBasePtr{tmemBaseWithStageOffset14};
            uint32_t(&dstSlice0)[32]{
              reinterpret_cast<uint32_t(&)[32]>(regsPartialAcc00[int32_t{0}])};
            cuda_ptx::tcgen05_ld_16x256b(dstSlice0, tmemBasePtr);
          }
          cutlass::arch::fence_view_async_tmem_load();
        }
        //
        // mma1 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { mma1SrcStack.mPipeline.consumer_release(mma1ConsState); }
          mma1ConsState += int32_t{2};
        }
        //
        // smemDeepSeekSfAb [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          {
            smemDeepSeekSfAbConsToken =
              smemDeepSeekSfAbSrcStack.mPipeline.consumer_try_wait(smemDeepSeekSfAbConsState);
          }
          smemDeepSeekSfAbSrcStack.mPipeline.consumer_wait(smemDeepSeekSfAbConsState,
                                                           smemDeepSeekSfAbConsToken);
        }
        //
        // smemDeepSeekSfAb [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        float* smemDqSfsAct9;
        float* smemDqSfsWeights9;
        {
          int32_t index{smemDeepSeekSfAbConsState.index()};
          smemDqSfsAct9 = &smemDeepSeekSfAbSrcSmem.mDqSfsAct[index][int32_t{0}];
          smemDqSfsWeights9 = &smemDeepSeekSfAbSrcSmem.mDqSfsWeights[index][int32_t{0}];
        }
        //
        // gmemC1 [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
        //
        float* smemDqSfsAct14;
        float* smemDqSfsW14;
        tmemBaseWithStageOffset14 = tmemBaseWithStageOffset12;
        smemDqSfsAct14 = smemDqSfsAct9;
        smemDqSfsW14 = smemDqSfsWeights9;
        {
          float dqSfW{smemDqSfsW14[int32_t{0}]};
          float dqSfAb0{(smemDqSfsAct14[mLdtm16dp256bitTmemColIdx]) * (dqSfW)};
          float dqSfAb1{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{1})]) * (dqSfW)};
          float dqSfAb2{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{8})]) * (dqSfW)};
          float dqSfAb3{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{9})]) * (dqSfW)};
          float dqSfAb4{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{16})]) * (dqSfW)};
          float dqSfAb5{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{17})]) * (dqSfW)};
          float dqSfAb6{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{24})]) * (dqSfW)};
          float dqSfAb7{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{25})]) * (dqSfW)};
          float dqSfAb8{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{32})]) * (dqSfW)};
          float dqSfAb9{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{33})]) * (dqSfW)};
          float dqSfAb10{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{40})]) * (dqSfW)};
          float dqSfAb11{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{41})]) * (dqSfW)};
          float dqSfAb12{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{48})]) * (dqSfW)};
          float dqSfAb13{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{49})]) * (dqSfW)};
          float dqSfAb14{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{56})]) * (dqSfW)};
          float dqSfAb15{(smemDqSfsAct14[(mLdtm16dp256bitTmemColIdx) + (int32_t{57})]) * (dqSfW)};
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb0, dqSfAb1};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{0}], regsAcc00[int32_t{1}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{0}],
                                         regsPartialAcc00[int32_t{1}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{0}] = tmp[int32_t{0}];
            regsAcc00[int32_t{1}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb0, dqSfAb1};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{2}], regsAcc00[int32_t{3}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{2}],
                                         regsPartialAcc00[int32_t{3}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{2}] = tmp[int32_t{0}];
            regsAcc00[int32_t{3}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb2, dqSfAb3};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{4}], regsAcc00[int32_t{5}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{4}],
                                         regsPartialAcc00[int32_t{5}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{4}] = tmp[int32_t{0}];
            regsAcc00[int32_t{5}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb2, dqSfAb3};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{6}], regsAcc00[int32_t{7}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{6}],
                                         regsPartialAcc00[int32_t{7}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{6}] = tmp[int32_t{0}];
            regsAcc00[int32_t{7}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb4, dqSfAb5};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{8}], regsAcc00[int32_t{9}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{8}],
                                         regsPartialAcc00[int32_t{9}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{8}] = tmp[int32_t{0}];
            regsAcc00[int32_t{9}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb4, dqSfAb5};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{10}], regsAcc00[int32_t{11}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{10}],
                                         regsPartialAcc00[int32_t{11}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{10}] = tmp[int32_t{0}];
            regsAcc00[int32_t{11}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb6, dqSfAb7};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{12}], regsAcc00[int32_t{13}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{12}],
                                         regsPartialAcc00[int32_t{13}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{12}] = tmp[int32_t{0}];
            regsAcc00[int32_t{13}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb6, dqSfAb7};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{14}], regsAcc00[int32_t{15}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{14}],
                                         regsPartialAcc00[int32_t{15}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{14}] = tmp[int32_t{0}];
            regsAcc00[int32_t{15}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb8, dqSfAb9};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{16}], regsAcc00[int32_t{17}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{16}],
                                         regsPartialAcc00[int32_t{17}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{16}] = tmp[int32_t{0}];
            regsAcc00[int32_t{17}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb8, dqSfAb9};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{18}], regsAcc00[int32_t{19}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{18}],
                                         regsPartialAcc00[int32_t{19}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{18}] = tmp[int32_t{0}];
            regsAcc00[int32_t{19}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb10, dqSfAb11};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{20}], regsAcc00[int32_t{21}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{20}],
                                         regsPartialAcc00[int32_t{21}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{20}] = tmp[int32_t{0}];
            regsAcc00[int32_t{21}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb10, dqSfAb11};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{22}], regsAcc00[int32_t{23}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{22}],
                                         regsPartialAcc00[int32_t{23}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{22}] = tmp[int32_t{0}];
            regsAcc00[int32_t{23}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb12, dqSfAb13};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{24}], regsAcc00[int32_t{25}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{24}],
                                         regsPartialAcc00[int32_t{25}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{24}] = tmp[int32_t{0}];
            regsAcc00[int32_t{25}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb12, dqSfAb13};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{26}], regsAcc00[int32_t{27}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{26}],
                                         regsPartialAcc00[int32_t{27}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{26}] = tmp[int32_t{0}];
            regsAcc00[int32_t{27}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb14, dqSfAb15};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{28}], regsAcc00[int32_t{29}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{28}],
                                         regsPartialAcc00[int32_t{29}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{28}] = tmp[int32_t{0}];
            regsAcc00[int32_t{29}] = tmp[int32_t{1}];
          }
          {
            cutlass::Array<float, 2> dqSfsAb{dqSfAb14, dqSfAb15};
            cutlass::Array<float, 2> tmp{regsAcc00[int32_t{30}], regsAcc00[int32_t{31}]};
            cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{30}],
                                         regsPartialAcc00[int32_t{31}]};
            tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
            regsAcc00[int32_t{30}] = tmp[int32_t{0}];
            regsAcc00[int32_t{31}] = tmp[int32_t{1}];
          }
        }
        //
        // smemDeepSeekSfAb [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          { smemDeepSeekSfAbSrcStack.mPipeline.consumer_release(smemDeepSeekSfAbConsState); }
          ++smemDeepSeekSfAbConsState;
        }
        //
        // gmemC1 [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
      }
      //
      // Tail work.
      //
      //
      // gmemC1 [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {
        //
        // Epilogue tile idxM=0 idxN=0.
        //
        {
          cuda_ptx::cp_async_bulk_wait_group_read(cuda_ptx::n32_t<0>{});
          trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{8}) + (mWarpGrp4Idx));
          //
          // Store to Smem TmaAsyncGmemC.
          //
          int8_t* ptrSmemBase;
          cutlass::bfloat16_t* ptrSmem;
          ptrSmemBase = reinterpret_cast<int8_t*>(gmemC1DstStack.mDepSmemPtr2) +
                        ((mWarpGrp4Idx) * (int32_t{8192}) + (int32_t{155648}));
          ptrSmem = reinterpret_cast<cutlass::bfloat16_t*>(ptrSmemBase) + (int32_t{0});
          //
          // Smem store idxM=0 idxN=0.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{((mBaseTmemCol) * (int32_t{64}) + (mBaseRowIdx)) /
                                       (int32_t{64})};
              int32_t const smemOffsetInBytes{
                (((mBaseTmemCol) * (int32_t{64}) + (mBaseRowIdx)) * (int32_t{16})) / (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{0}], regsAcc00[int32_t{2}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=1.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{1})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{1})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{1}], regsAcc00[int32_t{3}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=2.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{8})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{8})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{4}], regsAcc00[int32_t{6}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=3.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{9})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{9})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{5}], regsAcc00[int32_t{7}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=4.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{16})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{16})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{8}], regsAcc00[int32_t{10}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=5.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{17})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{17})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{9}], regsAcc00[int32_t{11}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=6.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{24})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{24})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{12}], regsAcc00[int32_t{14}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=7.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{25})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{25})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{13}], regsAcc00[int32_t{15}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=8.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{32})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{32})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{16}], regsAcc00[int32_t{18}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=9.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{33})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{33})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{17}], regsAcc00[int32_t{19}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=10.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{40})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{40})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{20}], regsAcc00[int32_t{22}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=11.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{41})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{41})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{21}], regsAcc00[int32_t{23}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=12.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{48})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{48})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{24}], regsAcc00[int32_t{26}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=13.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{49})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{49})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{25}], regsAcc00[int32_t{27}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=14.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{56})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{56})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{28}], regsAcc00[int32_t{30}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          //
          // Smem store idxM=0 idxN=15.
          //
          {
            int32_t smemOffset0;
            {
              int32_t const smemRowIdx{
                (((mBaseTmemCol) + (int32_t{57})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{64})};
              int32_t const smemOffsetInBytes{
                ((((mBaseTmemCol) + (int32_t{57})) * (int32_t{64}) + (mBaseRowIdx)) *
                 (int32_t{16})) /
                (int32_t{8})};
              int32_t const swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
              smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{16});
            }
            cutlass::Array<float, 2> accF2{regsAcc00[int32_t{29}], regsAcc00[int32_t{31}]};
            cutlass::Array<cutlass::bfloat16_t, 2> scaledCvtAcc2{
              trtllm::dev::convert_float2_to_bfloat16(accF2)};
            {
              uint32_t convertedElts;
              convertedElts = reinterpret_cast<uint32_t&>(scaledCvtAcc2);
              reinterpret_cast<uint32_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
            }
          }
          cuda_ptx::fence_proxy_async(cuda_ptx::space_shared_t{});
          trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{8}) + (mWarpGrp4Idx));
          //
          // Issue TMA from smem to gmem.
          //
          if ((bool{cute::elect_one_sync()}) && ((mWarpGrp4WarpIdx) == (int32_t{0}))) {
            int8_t* ptrSmemBase;
            cutlass::bfloat16_t* ptrSmem;
            ptrSmemBase = reinterpret_cast<int8_t*>(gmemC1DstStack.mDepSmemPtr2) +
                          ((mWarpGrp4Idx) * (int32_t{8192}) + (int32_t{155648}));
            ptrSmem = reinterpret_cast<cutlass::bfloat16_t*>(ptrSmemBase) + (int32_t{0});
            int32_t coords[4];
            coords[int32_t{0}] = ((int32_t{2}) * (mCtaIdxX) + (int32_t{1})) * (int32_t{64});
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
          trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{8}) + (mWarpGrp4Idx));
        }
        {
          //
          // Skip all-reduce if on single device.
          //
        }
      }
    ExitTileWithSignalingLabel:
    ExitTileWithoutSignalingLabel: {
      uint32_t stateStub_{mma1ConsState.count()};
      mma1ConsState =
        trtllm::dev::makePipelineState(trtllm::dev::makeConsStartStateFrom(mma1ConsState),
                                       mma1ConsStateStub);
      mma1ConsStateStub = stateStub_;
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
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<48>{});
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsState{};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
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
  int32_t mCtaIdxX;
  int32_t mCtaIdxY;
  int32_t mCtaIdxZ;
  inline __device__ WorkIdTask(KernelParams const& params,
                               KernelState const& state,
                               int32_t warpGrpStart)
    : mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
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
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<48>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsState{};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdConsReleaseState{};
    int32_t workIdConsToken{int32_t{0}};
    trtllm::dev::CutlassCpAsyncPipeline<3>::PipelineState workThrottleBarrierConsState{};
    trtllm::dev::CutlassCpAsyncPipeline<3>::PipelineState workThrottleBarrierConsReleaseState{};
    int32_t workThrottleBarrierConsToken{int32_t{0}};
    trtllm::dev::CutlassWorkIdPipeline<3, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdProdState{int32_t{0}, int32_t{1}, int32_t{0}};
    int32_t workIdProdToken{int32_t{1}};
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
};
extern "C" __global__
__launch_bounds__(512, 1) void bmm_Bfloat16_E4m3E4m3_Fp32_t128x64x128_s6_et64x64_m64x64x32_cga1x1x1_16dp256b_rM_TN_transOut_dsFp8_schPd4x2x2x3_bN_rgTma_clmp_dynB_sm100f(
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
  smemOffset__ = (((smemOffset__) + (int32_t{1023})) / (int32_t{1024})) * (int32_t{1024});
  uint8_t* gmemC1SmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(GmemC1Smem)});
  smemOffset__ = (((smemOffset__) + (int32_t{15})) / (int32_t{16})) * (int32_t{16});
  uint8_t* workIdSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(WorkIdSmem)});
  smemOffset__ = (((smemOffset__) + (int32_t{15})) / (int32_t{16})) * (int32_t{16});
  uint8_t* smemDeepSeekSfAbSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemDeepSeekSfAbSmem)});
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
  uint8_t* smemDeepSeekSfAbSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemDeepSeekSfAbSmemBarrier)});
  uint8_t* mma0SmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(Mma0SmemBarrier)});
  uint8_t* mma1SmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(Mma1SmemBarrier)});
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
                        int32_t{10},
                        int32_t{-1}};
  SmemDeepSeekSfAbSmem* smemDeepSeekSfAbSmem{
    reinterpret_cast<SmemDeepSeekSfAbSmem*>(smemDeepSeekSfAbSmemPtr)};
  SmemDeepSeekSfAbSmemBarrier* smemDeepSeekSfAbSmemBarrier{
    reinterpret_cast<SmemDeepSeekSfAbSmemBarrier*>(smemDeepSeekSfAbSmemBarrierPtr)};
  SmemDeepSeekSfAbStack smemDeepSeekSfAbStack{(*smemDeepSeekSfAbSmem),
                                              (*smemDeepSeekSfAbSmemBarrier),
                                              state.mWarpIdx,
                                              int32_t{11},
                                              int32_t{-1}};
  TmemStack tmemStack{state.mWarpIdx, int32_t{0}, int32_t{-1}};
  Mma0SmemBarrier* mma0SmemBarrier{reinterpret_cast<Mma0SmemBarrier*>(mma0SmemBarrierPtr)};
  Mma0Stack mma0Stack{(*mma0SmemBarrier), tmemStack, state.mWarpIdx, int32_t{1}, int32_t{-1}};
  Mma1SmemBarrier* mma1SmemBarrier{reinterpret_cast<Mma1SmemBarrier*>(mma1SmemBarrierPtr)};
  Mma1Stack mma1Stack{(*mma1SmemBarrier), tmemStack, state.mWarpIdx, int32_t{4}, int32_t{-1}};
  GmemC0Smem* gmemC0Smem{reinterpret_cast<GmemC0Smem*>(gmemC0SmemPtr)};
  GmemC0Stack gmemC0Stack{(*gmemC0Smem),
                          (*smemBufferSmem),
                          smemBufferStack,
                          state.mWarpIdx,
                          int32_t{0},
                          int32_t{-1}};
  GmemC1Smem* gmemC1Smem{reinterpret_cast<GmemC1Smem*>(gmemC1SmemPtr)};
  GmemC1Stack gmemC1Stack{(*gmemC1Smem),
                          (*smemBufferSmem),
                          smemBufferStack,
                          state.mWarpIdx,
                          int32_t{0},
                          int32_t{-1}};
  LoadTaskA loadTaskA{params, state, int32_t{9}};
  LoadTaskB loadTaskB{params, state, int32_t{10}};
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  if ((reinterpret_cast<int32_t const&>(threadIdx.x)) < (int32_t{32})) {
    cuda_ptx::tcgen05_alloc(cuda_ptx::cta_group_1_t{}, state.mTmemSwStatePtr, int32_t{256});
    cuda_ptx::tcgen05_relinquish_alloc_permit(cuda_ptx::cta_group_1_t{});
  }
  if (((bool{LoadTaskA::isSelected(params, state)}) ||
       (bool{LoadTaskB::isSelected(params, state)})) ||
      (bool{LoadSfAbTask::isSelected(params, state)})) {
  } else {
    trtllm::dev::CutlassNamedBarrier::sync(416, 10);
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
      if (bool{LoadSfAbTask::isSelected(params, state)}) {
        LoadSfAbTask loadSfAbTask{params, state, int32_t{11}};
        loadSfAbTask.execute(params,
                             state,
                             (*smemDeepSeekSfAbSmem),
                             smemDeepSeekSfAbStack,
                             (*workIdSmem),
                             workIdStack);
      } else {
        if (bool{MmaTask0::isSelected(params, state)}) {
          MmaTask0 mmaTask0{params, state, int32_t{8}};
          mmaTask0.execute(params,
                           state,
                           mma0Stack,
                           mma1Stack,
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
                                  mma0Stack,
                                  (*smemDeepSeekSfAbSmem),
                                  smemDeepSeekSfAbStack,
                                  (*workIdSmem),
                                  workIdStack);
            trtllm::dev::CutlassNamedBarrier::sync(128, 11);
            int32_t const warpGrpThreadIdx{state.mThreadIdx};
            if ((warpGrpThreadIdx) < (int32_t{32})) {
              cuda_ptx::tcgen05_dealloc(cuda_ptx::cta_group_1_t{},
                                        uint32_t{__shfl_sync(uint32_t{0xffffffff},
                                                             (*state.mTmemSwStatePtr),
                                                             int32_t{0},
                                                             int32_t{32})},
                                        int32_t{256});
            }
          } else {
            if (bool{EpilogueTask1::isSelected(params, state)}) {
              EpilogueTask1 epilogueTask1{params, state, int32_t{4}};
              epilogueTask1.execute(params,
                                    state,
                                    (*gmemC1Smem),
                                    gmemC1Stack,
                                    mma1Stack,
                                    (*smemDeepSeekSfAbSmem),
                                    smemDeepSeekSfAbStack,
                                    (*workIdSmem),
                                    workIdStack);
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
bmm_Bfloat16_E4m3E4m3_Fp32_t128x64x128_s6_et64x64_m64x64x32_cga1x1x1_16dp256b_rM_TN_transOut_dsFp8_schPd4x2x2x3_bN_rgTma_clmp_dynB_sm100fGetSmemSize(
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
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(GmemC1Smem));
  size = (size + 15) / 16 * 16;
  size += static_cast<int32_t>(sizeof(WorkIdSmem));
  size = (size + 15) / 16 * 16;
  size += static_cast<int32_t>(sizeof(SmemDeepSeekSfAbSmem));
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
  size += static_cast<int32_t>(sizeof(SmemDeepSeekSfAbSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(Mma0SmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(Mma1SmemBarrier));
  outPtr[0] = size;
}

} // namespace batchedGemm
