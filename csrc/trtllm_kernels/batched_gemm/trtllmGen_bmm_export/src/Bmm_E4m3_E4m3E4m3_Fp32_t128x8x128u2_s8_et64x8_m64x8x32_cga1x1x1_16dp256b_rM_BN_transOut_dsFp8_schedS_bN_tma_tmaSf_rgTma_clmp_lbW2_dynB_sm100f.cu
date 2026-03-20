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
#include <Bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_cga1x1x1_16dp256b_rM_BN_transOut_dsFp8_schedS_bN_tma_tmaSf_rgTma_clmp_lbW2_dynB_sm100f.h>
namespace batchedGemm {


struct SmemBufferStack {
  int8_t* mPtr;
  inline __device__ SmemBufferStack(SmemBufferSmem& smemBufferSmem,
                                    int32_t warpId,
                                    int32_t barInitWarpId,
                                    int32_t orderedSequenceGroupId)
    : mPtr{&smemBufferSmem.mArray[int32_t{0}]} {}
};
struct SmemAStack {
  int8_t* mDepSmemPtr0;
  trtllm::dev::
    CutlassTmaMultiUmmaAsyncPipeline<8, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
      mPipeline;
  inline __device__ SmemAStack(SmemASmem& smemASmem,
                               SmemASmemBarrier& smemASmemBarrier,
                               SmemBufferSmem& smemBufferSmem,
                               SmemBufferStack& smemBufferStack,
                               int32_t warpId,
                               int32_t barInitWarpId,
                               int32_t orderedSequenceGroupId)
    : mDepSmemPtr0{&smemBufferSmem.mArray[int32_t{0}]}
    , mPipeline{smemASmemBarrier.mBarriers,
                warpId,
                int32_t{16384},
                ((warpId) == (barInitWarpId)) && (bool{cute::elect_one_sync()}),
                int32_t{1},
                CuteFlatTuple208{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId} {}
};
struct SmemBStack {
  int8_t* mDepSmemPtr0;
  trtllm::dev::
    CutlassTmaMultiUmmaAsyncPipeline<8, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
      mPipeline;
  inline __device__ SmemBStack(SmemBSmem& smemBSmem,
                               SmemBSmemBarrier& smemBSmemBarrier,
                               SmemBufferSmem& smemBufferSmem,
                               SmemBufferStack& smemBufferStack,
                               int32_t warpId,
                               int32_t barInitWarpId,
                               int32_t orderedSequenceGroupId)
    : mDepSmemPtr0{&smemBufferSmem.mArray[int32_t{0}]}
    , mPipeline{smemBSmemBarrier.mBarriers,
                warpId,
                int32_t{1024},
                ((warpId) == (barInitWarpId)) && (bool{cute::elect_one_sync()}),
                int32_t{1},
                CuteFlatTuple330{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId} {}
};
struct SmemDeepSeekSfAbStack {
  trtllm::dev::CutlassCpAsyncPipeline<8> mPipeline;
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
  trtllm::dev::CutlassUmmaAsyncPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  inline __device__ Mma0Stack(Mma0SmemBarrier& mma0SmemBarrier,
                              TmemStack& tmemStack,
                              int32_t warpId,
                              int32_t barInitWarpId,
                              int32_t orderedSequenceGroupId)
    : mPipeline{mma0SmemBarrier.mBarriers,
                warpId,
                int32_t{128},
                CuteFlatTuple563{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId} {}
};
struct Mma1Stack {
  trtllm::dev::CutlassUmmaAsyncPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  inline __device__ Mma1Stack(Mma1SmemBarrier& mma1SmemBarrier,
                              TmemStack& tmemStack,
                              int32_t warpId,
                              int32_t barInitWarpId,
                              int32_t orderedSequenceGroupId)
    : mPipeline{mma1SmemBarrier.mBarriers,
                warpId,
                int32_t{128},
                CuteFlatTuple672{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId} {}
};
struct GmemC0Stack {
  int8_t* mDepSmemPtr0;
  inline __device__ GmemC0Stack(GmemC0Smem& gmemC0Smem,
                                SmemBufferSmem& smemBufferSmem,
                                SmemBufferStack& smemBufferStack,
                                int32_t warpId,
                                int32_t barInitWarpId,
                                int32_t orderedSequenceGroupId)
    : mDepSmemPtr0{&smemBufferSmem.mArray[int32_t{0}]} {}
};
struct GmemC1Stack {
  int8_t* mDepSmemPtr0;
  inline __device__ GmemC1Stack(GmemC1Smem& gmemC1Smem,
                                SmemBufferSmem& smemBufferSmem,
                                SmemBufferStack& smemBufferStack,
                                int32_t warpId,
                                int32_t barInitWarpId,
                                int32_t orderedSequenceGroupId)
    : mDepSmemPtr0{&smemBufferSmem.mArray[int32_t{0}]} {}
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
  int32_t const mBatchIdx;
  int32_t const mBatchLimit;
  int32_t mCtaOffsetK;
  int32_t mCtaIdxX;
  inline __device__ LoadTaskA(KernelParams const& params,
                              KernelState const& state,
                              int32_t warpGrpStart)
    : mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mBatchIdx{int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}}
    , mBatchLimit{int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]}}
    , mCtaOffsetK{int32_t{0}}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{9})) && ((state.mWarpIdx) < (int32_t{10}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 SmemASmem& smemADstSmem,
                                 SmemAStack& smemADstStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<48>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      8,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemAProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    int32_t smemAProdToken{int32_t{1}};
    int32_t paddedPerCtaK{(((params.k) + (int32_t{127})) / (int32_t{128})) * (int32_t{128})};
    int32_t loopEnd{paddedPerCtaK};
    bool const hasOneLoopIter{(int32_t{0}) < (loopEnd)};
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
    // Loop body.
    //
    CUTLASS_PRAGMA_NO_UNROLL
    for (int32_t loopOffset257 = int32_t{0}; loopOffset257 < loopEnd;
         loopOffset257 += int32_t{128}) {
      bool const isLastLoopIter{((loopOffset257) + (int32_t{128})) >= (loopEnd)};
      //
      // gmemA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK1;
      { tileOffsetK1 = loopOffset257; }
      //
      // smemA [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {
        smemADstStack.mPipeline.producer_acquire(smemAProdState, smemAProdToken);
        if (((loopOffset257) + (int32_t{128})) < (loopEnd)) {
          smemAProdToken = smemADstStack.mPipeline.producer_try_acquire(
            trtllm::dev::makePipelineState(smemAProdState, int32_t{1}));
        }
      }
      //
      // smemA [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK5;
      tileOffsetK5 = tileOffsetK1;
      {
        uint64_t* barrier{smemADstStack.mPipeline.producer_get_barrier(smemAProdState)};
        int32_t index{smemAProdState.index()};
        {}
        {
          int8_t* smemBytesBasePtrA;
          int8_t* smemBytesStagePtrA;
          smemBytesBasePtrA = reinterpret_cast<int8_t*>(smemADstStack.mDepSmemPtr0) + (int32_t{0});
          smemBytesStagePtrA = smemBytesBasePtrA + ((index) * (int32_t{16384}));
          int32_t coords[4];
          coords[int32_t{0}] = int32_t{0};
          coords[int32_t{1}] = (mCtaIdxX) * (int32_t{128});
          coords[int32_t{2}] = (tileOffsetK5) / (int32_t{128});
          coords[int32_t{3}] = mBatchIdx;
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
      if ((loopOffset257) >= (int32_t{0})) {
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
    // smemA [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    {}
  ExitTileWithSignalingLabel:
  ExitTileWithoutSignalingLabel: {}
  }
};
struct LoadTaskB {
  int32_t mCtaIdxY;
  int32_t const mBatchIdx;
  int32_t const mBatchLimit;
  int32_t const mWarpGrpThreadIdx;
  int32_t mRoutedIndices[4];
  int32_t const mWarpGrpWarpIdx;
  int32_t mCtaOffsetK;
  inline __device__ LoadTaskB(KernelParams const& params,
                              KernelState const& state,
                              int32_t warpGrpStart)
    : mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mBatchIdx{int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}}
    , mBatchLimit{int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]}}
    , mWarpGrpThreadIdx{min(int32_t{64},
                            max(int32_t{0}, (state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))))}
    , mWarpGrpWarpIdx{(state.mWarpIdx) - (warpGrpStart)}
    , mCtaOffsetK{int32_t{0}} {
    cudaGridDependencySynchronize();
    {
      if ((((mWarpGrpWarpIdx) * (int32_t{4})) < (int32_t{8})) &&
          (bool{LoadTaskB::isSelected(params, state)})) {
        mRoutedIndices[int32_t{0}] = int32_t{
          params.ptrRouteMap[((mWarpGrpWarpIdx) * (int32_t{4})) + ((mCtaIdxY) * (int32_t{8}))]};
        mRoutedIndices[int32_t{1}] =
          int32_t{params.ptrRouteMap[((mWarpGrpWarpIdx) * (int32_t{4}) + (int32_t{1})) +
                                     ((mCtaIdxY) * (int32_t{8}))]};
        mRoutedIndices[int32_t{2}] =
          int32_t{params.ptrRouteMap[((mWarpGrpWarpIdx) * (int32_t{4}) + (int32_t{2})) +
                                     ((mCtaIdxY) * (int32_t{8}))]};
        mRoutedIndices[int32_t{3}] =
          int32_t{params.ptrRouteMap[((mWarpGrpWarpIdx) * (int32_t{4}) + (int32_t{3})) +
                                     ((mCtaIdxY) * (int32_t{8}))]};
      }
    }
  }
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{10})) && ((state.mWarpIdx) < (int32_t{12}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 SmemBSmem& smemBDstSmem,
                                 SmemBStack& smemBDstStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<48>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      8,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemBProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    int32_t smemBProdToken{int32_t{1}};
    int32_t paddedPerCtaK{(((params.k) + (int32_t{127})) / (int32_t{128})) * (int32_t{128})};
    int32_t loopEnd{paddedPerCtaK};
    bool const hasOneLoopIter{(int32_t{0}) < (loopEnd)};
    //
    // Hoist the first iter.
    //
    //
    // Loop body.
    //
    CUTLASS_PRAGMA_NO_UNROLL
    for (int32_t loopOffset339 = int32_t{0}; loopOffset339 < loopEnd;
         loopOffset339 += int32_t{128}) {
      bool const isLastLoopIter{((loopOffset339) + (int32_t{128})) >= (loopEnd)};
      //
      // gmemB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK2;
      { tileOffsetK2 = loopOffset339; }
      //
      // smemB [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {
        if ((loopOffset339) >= (int32_t{1024})) {
          smemBProdToken = smemBDstStack.mPipeline.producer_try_acquire(smemBProdState);
        }
      }
      { smemBDstStack.mPipeline.producer_acquire(smemBProdState, smemBProdToken); }
      //
      // smemB [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK6;
      tileOffsetK6 = tileOffsetK2;
      {
        uint64_t* barrier{smemBDstStack.mPipeline.producer_get_barrier(smemBProdState)};
        int32_t index{smemBProdState.index()};
        {
          int8_t* smemBytesBasePtrB;
          int8_t* smemBytesStagePtrB;
          smemBytesBasePtrB =
            reinterpret_cast<int8_t*>(smemBDstStack.mDepSmemPtr0) + (int32_t{131072});
          smemBytesStagePtrB = smemBytesBasePtrB + ((index) * (int32_t{1024}));
          {
            int32_t coords[5];
            coords[int32_t{0}] = tileOffsetK6;
            coords[int32_t{1}] = mRoutedIndices[int32_t{0}];
            coords[int32_t{2}] = mRoutedIndices[int32_t{1}];
            coords[int32_t{3}] = mRoutedIndices[int32_t{2}];
            coords[int32_t{4}] = mRoutedIndices[int32_t{3}];
            if ((bool{cute::elect_one_sync()}) &&
                (((mWarpGrpWarpIdx) * (int32_t{4})) < (int32_t{8}))) {
              cuda_ptx::cp_async_bulk_tensor_tile_gather4(
                cuda_ptx::space_shared_t{},
                cuda_ptx::space_global_t{},
                &reinterpret_cast<cutlass::float_e4m3_t*>(
                  smemBytesStagePtrB)[((mWarpGrpWarpIdx) * (int32_t{4})) * (int32_t{128})],
                params.tmaB,
                coords,
                barrier);
            }
          }
        }
      }
      //
      // smemB [ProdPreCommit, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if ((loopOffset339) >= (int32_t{0})) {
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
    // smemB [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    {}
  ExitTileWithSignalingLabel:
  ExitTileWithoutSignalingLabel: {}
    cudaTriggerProgrammaticLaunchCompletion();
  }
};
struct LoadSfAbTask {
  int32_t mCtaOffsetK;
  int32_t mCtaIdxY;
  int32_t mCtaIdxZ;
  int32_t const mWarpGrpThreadIdx;
  int32_t mCtaIdxX;
  inline __device__ LoadSfAbTask(KernelParams const& params,
                                 KernelState const& state,
                                 int32_t warpGrpStart)
    : mCtaOffsetK{int32_t{0}}
    , mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)}
    , mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{12})) && ((state.mWarpIdx) < (int32_t{13}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 SmemDeepSeekSfAbSmem& smemDeepSeekSfAbDstSmem,
                                 SmemDeepSeekSfAbStack& smemDeepSeekSfAbDstStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<48>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassCpAsyncPipeline<8>::PipelineState smemDeepSeekSfAbProdState{int32_t{0},
                                                                                    int32_t{1},
                                                                                    int32_t{0}};
    int32_t smemDeepSeekSfAbProdToken{int32_t{1}};
    int32_t paddedPerCtaK{(((params.k) + (int32_t{127})) / (int32_t{128})) * (int32_t{128})};
    int32_t loopEnd{paddedPerCtaK};
    float const* gmemDqSfsAct;
    gmemDqSfsAct = reinterpret_cast<float const*>(params.ptrSfB) + (int32_t{0});
    int32_t colDqSfsAct;
    colDqSfsAct = (mCtaIdxY) * (int32_t{8}) + (mWarpGrpThreadIdx);
    gmemDqSfsAct += ((mCtaIdxZ) * ((params.k) / (int32_t{128}))) * (params.numTokens);
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
      int32_t tileOffsetK3;
      { tileOffsetK3 = int32_t{0}; }
      //
      // gmemSfB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK4;
      { tileOffsetK4 = int32_t{0}; }
      //
      // smemDeepSeekSfAb [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      //
      // smemDeepSeekSfAb [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK7;
      tileOffsetK7 = tileOffsetK3;
      {
        int32_t index{smemDeepSeekSfAbProdState.index()};
        float* dstSmemDqSfsAct{&smemDeepSeekSfAbDstSmem.mDqSfsAct[index][int32_t{0}]};
        dstSmemDqSfsAct += mWarpGrpThreadIdx;
        {
          float const* gmemDqSfsActAtIter{gmemDqSfsAct};
          float* dstSmemDqSfsActAtIter{dstSmemDqSfsAct};
          int32_t colIdx{colDqSfsAct};
          gmemDqSfsActAtIter += int32_t{params.ptrRouteMap[colIdx]};
          dstSmemDqSfsActAtIter += int32_t{0};
          if ((colDqSfsAct) < (int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]})) {
            trtllm::dev::cpAsync(reinterpret_cast<uint32_t*>(dstSmemDqSfsActAtIter),
                                 reinterpret_cast<uint32_t const*>(gmemDqSfsActAtIter));
          }
        }
        float* dstSmemDqSfsWeights{&smemDeepSeekSfAbDstSmem.mDqSfsWeights[index][int32_t{0}]};
        if ((mWarpGrpThreadIdx) == (int32_t{0})) {
          trtllm::dev::cpAsync(dstSmemDqSfsWeights, gmemDqSfsWeights);
        }
        gmemDqSfsAct += params.numTokens;
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
    for (int32_t loopOffset462 = int32_t{128}; loopOffset462 < loopEnd;
         loopOffset462 += int32_t{128}) {
      bool const isFirstLoopIter{(loopOffset462) == (int32_t{128})};
      bool const isLastLoopIter{((loopOffset462) + (int32_t{128})) >= (loopEnd)};
      //
      // smemDeepSeekSfAb [ProdPreCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if ((loopOffset462) >= (int32_t{128})) {
      }
      //
      // smemDeepSeekSfAb [ProdCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if ((loopOffset462) >= (int32_t{128})) {
        {
          smemDeepSeekSfAbDstStack.mPipeline.producer_commit(smemDeepSeekSfAbProdState);
        }
        ++smemDeepSeekSfAbProdState;
      }
      //
      // gmemSfA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK3;
      { tileOffsetK3 = loopOffset462; }
      //
      // gmemSfB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK4;
      { tileOffsetK4 = loopOffset462; }
      //
      // smemDeepSeekSfAb [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {
        if ((loopOffset462) >= (int32_t{1024})) {
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
      int32_t tileOffsetK7;
      tileOffsetK7 = tileOffsetK3;
      {
        int32_t index{smemDeepSeekSfAbProdState.index()};
        float* dstSmemDqSfsAct{&smemDeepSeekSfAbDstSmem.mDqSfsAct[index][int32_t{0}]};
        dstSmemDqSfsAct += mWarpGrpThreadIdx;
        {
          float const* gmemDqSfsActAtIter{gmemDqSfsAct};
          float* dstSmemDqSfsActAtIter{dstSmemDqSfsAct};
          int32_t colIdx{colDqSfsAct};
          gmemDqSfsActAtIter += int32_t{params.ptrRouteMap[colIdx]};
          dstSmemDqSfsActAtIter += int32_t{0};
          if ((colDqSfsAct) < (int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]})) {
            trtllm::dev::cpAsync(reinterpret_cast<uint32_t*>(dstSmemDqSfsActAtIter),
                                 reinterpret_cast<uint32_t const*>(gmemDqSfsActAtIter));
          }
        }
        float* dstSmemDqSfsWeights{&smemDeepSeekSfAbDstSmem.mDqSfsWeights[index][int32_t{0}]};
        if ((mWarpGrpThreadIdx) == (int32_t{0})) {
          trtllm::dev::cpAsync(dstSmemDqSfsWeights, gmemDqSfsWeights);
        }
        gmemDqSfsAct += params.numTokens;
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
    // smemDeepSeekSfAb [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    {}
  ExitTileWithSignalingLabel:
  ExitTileWithoutSignalingLabel: {}
  }
};
struct MmaTask0 {
  int32_t mCtaOffsetK;
  uint32_t const mTmemBaseOffset;
  inline __device__ MmaTask0(KernelParams const& params,
                             KernelState const& state,
                             int32_t warpGrpStart)
    : mCtaOffsetK{int32_t{0}}
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
                                 SmemBStack& smemBSrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<48>{});
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      8,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemAConsState{};
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      8,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemAConsReleaseState{};
    int32_t smemAConsToken{int32_t{0}};
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      8,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemBConsState{};
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      8,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemBConsReleaseState{};
    int32_t smemBConsToken{int32_t{0}};
    trtllm::dev::CutlassUmmaAsyncPipeline<2,
                                          cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState mma0ProdState{int32_t{0}, int32_t{1}, int32_t{0}};
    int32_t mma0ProdToken{int32_t{1}};
    trtllm::dev::CutlassUmmaAsyncPipeline<2,
                                          cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState mma1ProdState{int32_t{0}, int32_t{1}, int32_t{0}};
    int32_t mma1ProdToken{int32_t{1}};
    int32_t paddedPerCtaK{(((params.k) + (int32_t{127})) / (int32_t{128})) * (int32_t{128})};
    int32_t loopEnd{paddedPerCtaK};
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
    for (int32_t loopOffset585 = int32_t{0}; loopOffset585 < loopEnd;
         loopOffset585 += int32_t{256}) {
      //
      // Unrolled iter 0.
      //
      {
        bool const isFirstLoopIter{(loopOffset585) == (int32_t{0})};
        bool const isLastLoopIter{((loopOffset585) + (int32_t{128})) >= (loopEnd)};
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
          if ((loopOffset585) >= (int32_t{0})) {
            mma0ProdToken = mma0DstStack.mPipeline.producer_try_acquire(mma0ProdState);
          }
        }
        //
        // mma1 [ProdTryAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          if ((loopOffset585) >= (int32_t{0})) {
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
        cutlass::float_e4m3_t* smemPtrA5;
        {
          int32_t index{smemAConsState.index()};
          int8_t* smemBytesBasePtrA;
          smemBytesBasePtrA = reinterpret_cast<int8_t*>(smemASrcStack.mDepSmemPtr0) + (int32_t{0});
          int8_t* smemBytesStagePtrA;
          smemBytesStagePtrA = smemBytesBasePtrA + ((index) * (int32_t{16384}));
          smemPtrA5 = reinterpret_cast<cutlass::float_e4m3_t*>(smemBytesStagePtrA) + (int32_t{0});
          ++smemAConsState;
        }
        //
        // smemB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_e4m3_t* smemPtrB6;
        {
          int32_t index{smemBConsState.index()};
          int8_t* smemBytesBasePtrB;
          smemBytesBasePtrB =
            reinterpret_cast<int8_t*>(smemBSrcStack.mDepSmemPtr0) + (int32_t{131072});
          int8_t* smemBytesStagePtrB;
          smemBytesStagePtrB = smemBytesBasePtrB + ((index) * (int32_t{1024}));
          smemPtrB6 = reinterpret_cast<cutlass::float_e4m3_t*>(smemBytesStagePtrB) + (int32_t{0});
          ++smemBConsState;
        }
        //
        // mma0 [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{8388608}].
        //
        {
          mma0DstStack.mPipeline.producer_acquire(mma0ProdState, mma0ProdToken);
          if (((loopOffset585) + (int32_t{128})) < (loopEnd)) {
            mma0ProdToken = mma0DstStack.mPipeline.producer_try_acquire(
              trtllm::dev::makePipelineState(mma0ProdState, int32_t{1}));
          }
        }
        //
        // mma1 [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{8389632}].
        //
        {
          mma1DstStack.mPipeline.producer_acquire(mma1ProdState, mma1ProdToken);
          if (((loopOffset585) + (int32_t{128})) < (loopEnd)) {
            mma1ProdToken = mma1DstStack.mPipeline.producer_try_acquire(
              trtllm::dev::makePipelineState(mma1ProdState, int32_t{1}));
          }
        }
        //
        // mma0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_e4m3_t* smemPtrA9;
        cutlass::float_e4m3_t* smemPtrB9;
        smemPtrA9 = smemPtrA5;
        smemPtrB9 = smemPtrB6;
        {
          int32_t index{mma0ProdState.index()};
          uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{8})))};
          uint32_t ptrTmemOffsetD{ptrTmemD};
          cutlass::float_e4m3_t* ptrWithOffsetSmemA{(smemPtrA9 + int32_t{0})};
          cutlass::float_e4m3_t* ptrWithOffsetSmemB{(smemPtrB9 + int32_t{0})};
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
            // leadingDimInBytes = 1024, strideInBytes = 1024, swizzleMode = 1.
            //
            uint64_t smemDescB{
              trtllm::dev::createSmemDesc(ptrWithOffsetSmemB,
                                          uint32_t{0x400000 /*hi=64, lo=0*/},
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
                                                                    int32_t{8},
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
                                                                    int32_t{8},
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
                                                                    int32_t{8},
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
                                                                    int32_t{8},
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
          ++mma0ProdState;
        }
        //
        // mma1 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_e4m3_t* smemPtrA10;
        cutlass::float_e4m3_t* smemPtrB10;
        smemPtrA10 = smemPtrA5;
        smemPtrB10 = smemPtrB6;
        {
          int32_t index{mma1ProdState.index()};
          uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{8})))};
          uint32_t ptrTmemOffsetD{(ptrTmemD) + (uint32_t{1048576})};
          cutlass::float_e4m3_t* ptrWithOffsetSmemA{(smemPtrA10 + int32_t{8192})};
          cutlass::float_e4m3_t* ptrWithOffsetSmemB{(smemPtrB10 + int32_t{0})};
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
            // leadingDimInBytes = 1024, strideInBytes = 1024, swizzleMode = 1.
            //
            uint64_t smemDescB{
              trtllm::dev::createSmemDesc(ptrWithOffsetSmemB,
                                          uint32_t{0x400000 /*hi=64, lo=0*/},
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
                                                                    int32_t{8},
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
                                                                    int32_t{8},
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
                                                                    int32_t{8},
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
                                                                    int32_t{8},
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
          ++mma1ProdState;
        }
        //
        // smemA [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset585) >= (int32_t{0})) {
          {
            smemASrcStack.mPipeline.consumer_release(smemAConsReleaseState);
          }
          ++smemAConsReleaseState;
        }
        //
        // smemB [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset585) >= (int32_t{0})) {
          {
            smemBSrcStack.mPipeline.consumer_release(smemBConsReleaseState);
          }
          ++smemBConsReleaseState;
        }
      }
      //
      // Unrolled iter 1.
      //
      {
        bool const isFirstLoopIter{((int32_t{128}) + (loopOffset585)) == (int32_t{0})};
        bool const isLastLoopIter{(((int32_t{128}) + (loopOffset585)) + (int32_t{128})) >=
                                  (loopEnd)};
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
          if (((int32_t{128}) + (loopOffset585)) >= (int32_t{0})) {
            mma0ProdToken = mma0DstStack.mPipeline.producer_try_acquire(mma0ProdState);
          }
        }
        //
        // mma1 [ProdTryAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        {
          if (((int32_t{128}) + (loopOffset585)) >= (int32_t{0})) {
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
        cutlass::float_e4m3_t* smemPtrA5;
        {
          int32_t index{smemAConsState.index()};
          int8_t* smemBytesBasePtrA;
          smemBytesBasePtrA = reinterpret_cast<int8_t*>(smemASrcStack.mDepSmemPtr0) + (int32_t{0});
          int8_t* smemBytesStagePtrA;
          smemBytesStagePtrA = smemBytesBasePtrA + ((index) * (int32_t{16384}));
          smemPtrA5 = reinterpret_cast<cutlass::float_e4m3_t*>(smemBytesStagePtrA) + (int32_t{0});
          ++smemAConsState;
        }
        //
        // smemB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_e4m3_t* smemPtrB6;
        {
          int32_t index{smemBConsState.index()};
          int8_t* smemBytesBasePtrB;
          smemBytesBasePtrB =
            reinterpret_cast<int8_t*>(smemBSrcStack.mDepSmemPtr0) + (int32_t{131072});
          int8_t* smemBytesStagePtrB;
          smemBytesStagePtrB = smemBytesBasePtrB + ((index) * (int32_t{1024}));
          smemPtrB6 = reinterpret_cast<cutlass::float_e4m3_t*>(smemBytesStagePtrB) + (int32_t{0});
          ++smemBConsState;
        }
        //
        // mma0 [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{8388608}].
        //
        {
          mma0DstStack.mPipeline.producer_acquire(mma0ProdState, mma0ProdToken);
          if ((((int32_t{128}) + (loopOffset585)) + (int32_t{128})) < (loopEnd)) {
            mma0ProdToken = mma0DstStack.mPipeline.producer_try_acquire(
              trtllm::dev::makePipelineState(mma0ProdState, int32_t{1}));
          }
        }
        //
        // mma1 [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{8389632}].
        //
        {
          mma1DstStack.mPipeline.producer_acquire(mma1ProdState, mma1ProdToken);
          if ((((int32_t{128}) + (loopOffset585)) + (int32_t{128})) < (loopEnd)) {
            mma1ProdToken = mma1DstStack.mPipeline.producer_try_acquire(
              trtllm::dev::makePipelineState(mma1ProdState, int32_t{1}));
          }
        }
        //
        // mma0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_e4m3_t* smemPtrA9;
        cutlass::float_e4m3_t* smemPtrB9;
        smemPtrA9 = smemPtrA5;
        smemPtrB9 = smemPtrB6;
        {
          int32_t index{mma0ProdState.index()};
          uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{8})))};
          uint32_t ptrTmemOffsetD{ptrTmemD};
          cutlass::float_e4m3_t* ptrWithOffsetSmemA{(smemPtrA9 + int32_t{0})};
          cutlass::float_e4m3_t* ptrWithOffsetSmemB{(smemPtrB9 + int32_t{0})};
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
            // leadingDimInBytes = 1024, strideInBytes = 1024, swizzleMode = 1.
            //
            uint64_t smemDescB{
              trtllm::dev::createSmemDesc(ptrWithOffsetSmemB,
                                          uint32_t{0x400000 /*hi=64, lo=0*/},
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
                                                                    int32_t{8},
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
                                                                    int32_t{8},
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
                                                                    int32_t{8},
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
                                                                    int32_t{8},
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
          ++mma0ProdState;
        }
        //
        // mma1 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_e4m3_t* smemPtrA10;
        cutlass::float_e4m3_t* smemPtrB10;
        smemPtrA10 = smemPtrA5;
        smemPtrB10 = smemPtrB6;
        {
          int32_t index{mma1ProdState.index()};
          uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{8})))};
          uint32_t ptrTmemOffsetD{(ptrTmemD) + (uint32_t{1048576})};
          cutlass::float_e4m3_t* ptrWithOffsetSmemA{(smemPtrA10 + int32_t{8192})};
          cutlass::float_e4m3_t* ptrWithOffsetSmemB{(smemPtrB10 + int32_t{0})};
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
            // leadingDimInBytes = 1024, strideInBytes = 1024, swizzleMode = 1.
            //
            uint64_t smemDescB{
              trtllm::dev::createSmemDesc(ptrWithOffsetSmemB,
                                          uint32_t{0x400000 /*hi=64, lo=0*/},
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
                                                                    int32_t{8},
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
                                                                    int32_t{8},
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
                                                                    int32_t{8},
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
                                                                    int32_t{8},
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
          ++mma1ProdState;
        }
        //
        // smemA [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if (((int32_t{128}) + (loopOffset585)) >= (int32_t{0})) {
          {
            smemASrcStack.mPipeline.consumer_release(smemAConsReleaseState);
          }
          ++smemAConsReleaseState;
        }
        //
        // smemB [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if (((int32_t{128}) + (loopOffset585)) >= (int32_t{0})) {
          {
            smemBSrcStack.mPipeline.consumer_release(smemBConsReleaseState);
          }
          ++smemBConsReleaseState;
        }
      }
    }
  //
  // Tail work.
  //
  ExitTileWithSignalingLabel:
  ExitTileWithoutSignalingLabel: {}
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
  int32_t const mBatchIdx;
  int32_t const mBatchLimit;
  float mScaleC;
  int32_t mCtaOffsetK;
  uint32_t const mTmemBaseOffset;
  int32_t const mWarpGrpThreadIdx;
  int32_t const mLdtm16dp256bitTmemColIdx;
  int32_t const mLdtm16dp256bitTmemRowIdx;
  int32_t mCtaIdxX;
  cutlass::Array<float, 4> frg11;
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
    , mScaleC{float{
        (params.ptrScaleC + int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]})[int32_t{0}]}}
    , mCtaOffsetK{int32_t{0}}
    , mTmemBaseOffset{uint32_t{
        __shfl_sync(uint32_t{0xffffffff}, (*state.mTmemSwStatePtr), int32_t{0}, int32_t{32})}}
    , mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))}
    , mLdtm16dp256bitTmemColIdx{trtllm::dev::ldst16dp256bitTmemColIdx((mWarpGrpThreadIdx) %
                                                                      (int32_t{128}))}
    , mLdtm16dp256bitTmemRowIdx{trtllm::dev::ldst16dp256bitTmemRowIdx<int32_t{16}>(
        (mWarpGrpThreadIdx) % (int32_t{128}))}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
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
                                 SmemDeepSeekSfAbStack& smemDeepSeekSfAbSrcStack) {
    cuda_ptx::setmaxnreg_inc(cuda_ptx::n32_t<160>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassUmmaAsyncPipeline<
      2,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState mma0ConsState{};
    int32_t mma0ConsToken{int32_t{0}};
    trtllm::dev::CutlassCpAsyncPipeline<8>::PipelineState smemDeepSeekSfAbConsState{};
    int32_t smemDeepSeekSfAbConsToken{int32_t{0}};
    int32_t paddedPerCtaK{(((params.k) + (int32_t{127})) / (int32_t{128})) * (int32_t{128})};
    int32_t loopEnd{paddedPerCtaK};
    uint32_t tmemBaseWithStageOffset;
    tmemBaseWithStageOffset = mTmemBaseOffset;
    cutlass::Array<float, 4> regsAcc00;
    cutlass::Array<float, 4> regsPartialAcc00;
    CUTLASS_PRAGMA_UNROLL
    for (int32_t loopOffset995 = int32_t{0}; loopOffset995 < int32_t{4}; ++loopOffset995) {
      regsAcc00[loopOffset995] = float{0};
    }
    float* gmemDqSfsC;
    gmemDqSfsC = reinterpret_cast<float*>(params.ptrSfC) + (int32_t{0});
    int8_t* ptrSmemBaseRowMax;
    float* ptrSmemRowMax;
    ptrSmemBaseRowMax = reinterpret_cast<int8_t*>(gmemC0DstStack.mDepSmemPtr0) + (int32_t{141312});
    ptrSmemRowMax = reinterpret_cast<float*>(ptrSmemBaseRowMax) + (int32_t{0});
    uint32_t uint32NegFltMax11;
    uint32NegFltMax11 = uint32_t{trtllm::dev::floatToUInt32ForAtomicMax(float{-3.4028235e+38})};
    CUTLASS_PRAGMA_UNROLL
    for (int32_t loopOffset1006 = mWarpGrpThreadIdx; loopOffset1006 < int32_t{8};
         loopOffset1006 += int32_t{128}) {
      reinterpret_cast<uint32_t*>(ptrSmemRowMax)[loopOffset1006] = uint32NegFltMax11;
    }
    trtllm::dev::CutlassNamedBarrier::sync(256, 1);
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
      uint32_t tmemBaseWithStageOffset9;
      {
        int32_t index{mma0ConsState.index()};
        uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{8})))};
        uint32_t ptrTmemOffsetD{ptrTmemD};
        tmemBaseWithStageOffset9 = ptrTmemOffsetD;
      }
      //
      // gmemC0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      uint32_t tmemBaseWithStageOffset11;
      tmemBaseWithStageOffset11 = tmemBaseWithStageOffset9;
      {
        {
          uint32_t tmemBasePtr{tmemBaseWithStageOffset11};
          uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(regsPartialAcc00[int32_t{0}])};
          cuda_ptx::tcgen05_ld_16x256b(dstSlice0, tmemBasePtr);
        }
        cutlass::arch::fence_view_async_tmem_load();
      }
      //
      // mma0 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {
        { mma0SrcStack.mPipeline.consumer_release(mma0ConsState); }
        ++mma0ConsState;
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
      float* smemDqSfsAct7;
      float* smemDqSfsWeights7;
      {
        int32_t index{smemDeepSeekSfAbConsState.index()};
        smemDqSfsAct7 = &smemDeepSeekSfAbSrcSmem.mDqSfsAct[index][int32_t{0}];
        smemDqSfsWeights7 = &smemDeepSeekSfAbSrcSmem.mDqSfsWeights[index][int32_t{0}];
      }
      //
      // gmemC0 [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
      //
      float* smemDqSfsAct11;
      float* smemDqSfsW11;
      tmemBaseWithStageOffset11 = tmemBaseWithStageOffset9;
      smemDqSfsAct11 = smemDqSfsAct7;
      smemDqSfsW11 = smemDqSfsWeights7;
      {
        float dqSfW{smemDqSfsW11[int32_t{0}]};
        float dqSfAb0{(smemDqSfsAct11[mLdtm16dp256bitTmemColIdx]) * (dqSfW)};
        float dqSfAb1{(smemDqSfsAct11[(mLdtm16dp256bitTmemColIdx) + (int32_t{1})]) * (dqSfW)};
        {
          cutlass::Array<float, 2> dqSfsAb{dqSfAb0, dqSfAb1};
          cutlass::Array<float, 2> tmp{regsAcc00[int32_t{0}], regsAcc00[int32_t{1}]};
          cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{0}], regsPartialAcc00[int32_t{1}]};
          tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
          regsAcc00[int32_t{0}] = tmp[int32_t{0}];
          regsAcc00[int32_t{1}] = tmp[int32_t{1}];
        }
        {
          cutlass::Array<float, 2> dqSfsAb{dqSfAb0, dqSfAb1};
          cutlass::Array<float, 2> tmp{regsAcc00[int32_t{2}], regsAcc00[int32_t{3}]};
          cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{2}], regsPartialAcc00[int32_t{3}]};
          tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
          regsAcc00[int32_t{2}] = tmp[int32_t{0}];
          regsAcc00[int32_t{3}] = tmp[int32_t{1}];
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
    for (int32_t loopOffset1087 = int32_t{128}; loopOffset1087 < loopEnd;
         loopOffset1087 += int32_t{128}) {
      bool const isLastLoopIter{((loopOffset1087) + (int32_t{128})) >= (loopEnd)};
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
      uint32_t tmemBaseWithStageOffset9;
      {
        int32_t index{mma0ConsState.index()};
        uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{8})))};
        uint32_t ptrTmemOffsetD{ptrTmemD};
        tmemBaseWithStageOffset9 = ptrTmemOffsetD;
      }
      //
      // gmemC0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      uint32_t tmemBaseWithStageOffset11;
      tmemBaseWithStageOffset11 = tmemBaseWithStageOffset9;
      {
        {
          uint32_t tmemBasePtr{tmemBaseWithStageOffset11};
          uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(regsPartialAcc00[int32_t{0}])};
          cuda_ptx::tcgen05_ld_16x256b(dstSlice0, tmemBasePtr);
        }
        cutlass::arch::fence_view_async_tmem_load();
      }
      //
      // mma0 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {
        { mma0SrcStack.mPipeline.consumer_release(mma0ConsState); }
        ++mma0ConsState;
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
      float* smemDqSfsAct7;
      float* smemDqSfsWeights7;
      {
        int32_t index{smemDeepSeekSfAbConsState.index()};
        smemDqSfsAct7 = &smemDeepSeekSfAbSrcSmem.mDqSfsAct[index][int32_t{0}];
        smemDqSfsWeights7 = &smemDeepSeekSfAbSrcSmem.mDqSfsWeights[index][int32_t{0}];
      }
      //
      // gmemC0 [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
      //
      float* smemDqSfsAct11;
      float* smemDqSfsW11;
      tmemBaseWithStageOffset11 = tmemBaseWithStageOffset9;
      smemDqSfsAct11 = smemDqSfsAct7;
      smemDqSfsW11 = smemDqSfsWeights7;
      {
        float dqSfW{smemDqSfsW11[int32_t{0}]};
        float dqSfAb0{(smemDqSfsAct11[mLdtm16dp256bitTmemColIdx]) * (dqSfW)};
        float dqSfAb1{(smemDqSfsAct11[(mLdtm16dp256bitTmemColIdx) + (int32_t{1})]) * (dqSfW)};
        {
          cutlass::Array<float, 2> dqSfsAb{dqSfAb0, dqSfAb1};
          cutlass::Array<float, 2> tmp{regsAcc00[int32_t{0}], regsAcc00[int32_t{1}]};
          cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{0}], regsPartialAcc00[int32_t{1}]};
          tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
          regsAcc00[int32_t{0}] = tmp[int32_t{0}];
          regsAcc00[int32_t{1}] = tmp[int32_t{1}];
        }
        {
          cutlass::Array<float, 2> dqSfsAb{dqSfAb0, dqSfAb1};
          cutlass::Array<float, 2> tmp{regsAcc00[int32_t{2}], regsAcc00[int32_t{3}]};
          cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{2}], regsPartialAcc00[int32_t{3}]};
          tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
          regsAcc00[int32_t{2}] = tmp[int32_t{0}];
          regsAcc00[int32_t{3}] = tmp[int32_t{1}];
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
      float qSfsC0[2]{float{1.3165538e-36}, float{1.3165538e-36}};
      //
      // Compute max for fragment idxM=0 idxN=0.
      //
      {
        trtllm::dev::reduceColMax16dp256bit<int32_t{1}, int32_t{1}, int32_t{1}, true>(qSfsC0,
                                                                                      regsAcc00);
        trtllm::dev::reduceColMax(qSfsC0,
                                  (ptrSmemRowMax + int32_t{0}),
                                  int32_t{256},
                                  mWarpGrpThreadIdx,
                                  int32_t{1});
      }
      qSfsC0[int32_t{0}] = (float{448}) / (qSfsC0[int32_t{0}]);
      qSfsC0[int32_t{1}] = (float{448}) / (qSfsC0[int32_t{1}]);
      int32_t tileColIdx{(mCtaIdxY) * (int32_t{8})};
      int32_t tileRowIdx{((mCtaIdxX) * (int32_t{128})) / (int32_t{128})};
      bool const isThreadInFirstRow{(mWarpGrpThreadIdx) < (int32_t{4})};
      bool const isFirstThreadInRow{((mWarpGrpThreadIdx) % (int32_t{4})) == (int32_t{0})};
      if ((((tileColIdx) + (mBaseTmemCol)) < (mBatchLimit)) && (isThreadInFirstRow)) {
        gmemDqSfsC[(tileRowIdx) * (int32_t{params.ptrTotalNumPaddedTokens[int32_t{0}]}) +
                   ((tileColIdx) + (mBaseTmemCol))] = (float{1}) / (qSfsC0[int32_t{0}]);
      }
      if ((((tileColIdx) + ((mBaseTmemCol) + (int32_t{1}))) < (mBatchLimit)) &&
          (isThreadInFirstRow)) {
        gmemDqSfsC[(tileRowIdx) * (int32_t{params.ptrTotalNumPaddedTokens[int32_t{0}]}) +
                   ((tileColIdx) + ((mBaseTmemCol) + (int32_t{1})))] =
          (float{1}) / (qSfsC0[int32_t{1}]);
      }
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
        cutlass::float_e4m3_t* ptrSmem;
        ptrSmemBase = reinterpret_cast<int8_t*>(gmemC0DstStack.mDepSmemPtr0) +
                      ((mWarpGrp4Idx) * (int32_t{512}) + (int32_t{139264}));
        ptrSmem = reinterpret_cast<cutlass::float_e4m3_t*>(ptrSmemBase) + (int32_t{0});
        //
        // Smem store idxM=0 idxN=0.
        //
        {
          int32_t smemOffset0;
          {
            int32_t const smemRowIdx{((mBaseTmemCol) * (int32_t{64}) + (mBaseRowIdx)) /
                                     (int32_t{128})};
            int32_t const smemOffsetInBytes{
              (((mBaseTmemCol) * (int32_t{64}) + (mBaseRowIdx)) * (int32_t{8})) / (int32_t{8})};
            int32_t const swizzleMask{((smemRowIdx) % (int32_t{4})) * (int32_t{16})};
            smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{8});
          }
          cutlass::Array<float, 2> scaleF2{qSfsC0[int32_t{0}], qSfsC0[int32_t{0}]};
          cutlass::Array<float, 2> accF2{regsAcc00[int32_t{0}], regsAcc00[int32_t{2}]};
          cutlass::Array<float, 2> scaledAccF2{trtllm::dev::fmul2(accF2, scaleF2)};
          cutlass::Array<cutlass::float_e4m3_t, 2> scaledCvtAcc2{
            trtllm::dev::convert_float2_to_e4m3(scaledAccF2)};
          {
            uint16_t convertedElts;
            convertedElts = reinterpret_cast<uint16_t&>(scaledCvtAcc2);
            reinterpret_cast<uint16_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
          }
        }
        //
        // Smem store idxM=0 idxN=1.
        //
        {
          int32_t smemOffset0;
          {
            int32_t const smemRowIdx{
              (((mBaseTmemCol) + (int32_t{1})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{128})};
            int32_t const smemOffsetInBytes{
              ((((mBaseTmemCol) + (int32_t{1})) * (int32_t{64}) + (mBaseRowIdx)) * (int32_t{8})) /
              (int32_t{8})};
            int32_t const swizzleMask{((smemRowIdx) % (int32_t{4})) * (int32_t{16})};
            smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{8});
          }
          cutlass::Array<float, 2> scaleF2{qSfsC0[int32_t{1}], qSfsC0[int32_t{1}]};
          cutlass::Array<float, 2> accF2{regsAcc00[int32_t{1}], regsAcc00[int32_t{3}]};
          cutlass::Array<float, 2> scaledAccF2{trtllm::dev::fmul2(accF2, scaleF2)};
          cutlass::Array<cutlass::float_e4m3_t, 2> scaledCvtAcc2{
            trtllm::dev::convert_float2_to_e4m3(scaledAccF2)};
          {
            uint16_t convertedElts;
            convertedElts = reinterpret_cast<uint16_t&>(scaledCvtAcc2);
            reinterpret_cast<uint16_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
          }
        }
        cuda_ptx::fence_proxy_async(cuda_ptx::space_shared_t{});
        trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{6}) + (mWarpGrp4Idx));
        //
        // Issue TMA from smem to gmem.
        //
        if ((bool{cute::elect_one_sync()}) && ((mWarpGrp4WarpIdx) == (int32_t{0}))) {
          int8_t* ptrSmemBase;
          cutlass::float_e4m3_t* ptrSmem;
          ptrSmemBase = reinterpret_cast<int8_t*>(gmemC0DstStack.mDepSmemPtr0) +
                        ((mWarpGrp4Idx) * (int32_t{512}) + (int32_t{139264}));
          ptrSmem = reinterpret_cast<cutlass::float_e4m3_t*>(ptrSmemBase) + (int32_t{0});
          int32_t coords[4];
          coords[int32_t{0}] = ((int32_t{2}) * (mCtaIdxX)) * (int32_t{64});
          coords[int32_t{1}] = (((int32_t{8}) - ((mBatchLimit) % (int32_t{8}))) % (int32_t{8})) +
                               ((mWarpGrp4Idx) * (int32_t{8}));
          coords[int32_t{2}] = int32_t{0x40000000 /*1073741824*/};
          coords[int32_t{3}] =
            (((mCtaIdxY) * (int32_t{8})) +
             ((int32_t{0}) - (((int32_t{8}) - ((mBatchLimit) % (int32_t{8}))) % (int32_t{8})))) +
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
  ExitTileWithoutSignalingLabel: {}
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
  int32_t const mBatchIdx;
  int32_t const mBatchLimit;
  float mScaleC;
  int32_t mCtaOffsetK;
  uint32_t const mTmemBaseOffset;
  int32_t const mWarpGrpThreadIdx;
  int32_t const mLdtm16dp256bitTmemColIdx;
  int32_t const mLdtm16dp256bitTmemRowIdx;
  cutlass::Array<float, 4> frg12;
  int32_t mCtaIdxX;
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
    , mScaleC{float{
        (params.ptrScaleC + int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]})[int32_t{0}]}}
    , mCtaOffsetK{int32_t{0}}
    , mTmemBaseOffset{uint32_t{
        __shfl_sync(uint32_t{0xffffffff}, (*state.mTmemSwStatePtr), int32_t{0}, int32_t{32})}}
    , mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))}
    , mLdtm16dp256bitTmemColIdx{trtllm::dev::ldst16dp256bitTmemColIdx((mWarpGrpThreadIdx) %
                                                                      (int32_t{128}))}
    , mLdtm16dp256bitTmemRowIdx{trtllm::dev::ldst16dp256bitTmemRowIdx<int32_t{16}>(
        (mWarpGrpThreadIdx) % (int32_t{128}))}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
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
                                 SmemDeepSeekSfAbStack& smemDeepSeekSfAbSrcStack) {
    cuda_ptx::setmaxnreg_inc(cuda_ptx::n32_t<160>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassUmmaAsyncPipeline<
      2,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState mma1ConsState{};
    int32_t mma1ConsToken{int32_t{0}};
    trtllm::dev::CutlassCpAsyncPipeline<8>::PipelineState smemDeepSeekSfAbConsState{};
    int32_t smemDeepSeekSfAbConsToken{int32_t{0}};
    int32_t paddedPerCtaK{(((params.k) + (int32_t{127})) / (int32_t{128})) * (int32_t{128})};
    int32_t loopEnd{paddedPerCtaK};
    uint32_t tmemBaseWithStageOffset;
    tmemBaseWithStageOffset = mTmemBaseOffset;
    cutlass::Array<float, 4> regsAcc00;
    cutlass::Array<float, 4> regsPartialAcc00;
    CUTLASS_PRAGMA_UNROLL
    for (int32_t loopOffset1297 = int32_t{0}; loopOffset1297 < int32_t{4}; ++loopOffset1297) {
      regsAcc00[loopOffset1297] = float{0};
    }
    float* gmemDqSfsC;
    gmemDqSfsC = reinterpret_cast<float*>(params.ptrSfC) + (int32_t{0});
    int8_t* ptrSmemBaseRowMax;
    float* ptrSmemRowMax;
    ptrSmemBaseRowMax = reinterpret_cast<int8_t*>(gmemC1DstStack.mDepSmemPtr0) + (int32_t{141312});
    ptrSmemRowMax = reinterpret_cast<float*>(ptrSmemBaseRowMax) + (int32_t{0});
    trtllm::dev::CutlassNamedBarrier::sync(256, 1);
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
      uint32_t tmemBaseWithStageOffset10;
      {
        int32_t index{mma1ConsState.index()};
        uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{8})))};
        uint32_t ptrTmemOffsetD{(ptrTmemD) + (uint32_t{1048576})};
        tmemBaseWithStageOffset10 = ptrTmemOffsetD;
      }
      //
      // gmemC1 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      uint32_t tmemBaseWithStageOffset12;
      tmemBaseWithStageOffset12 = tmemBaseWithStageOffset10;
      {
        {
          uint32_t tmemBasePtr{tmemBaseWithStageOffset12};
          uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(regsPartialAcc00[int32_t{0}])};
          cuda_ptx::tcgen05_ld_16x256b(dstSlice0, tmemBasePtr);
        }
        cutlass::arch::fence_view_async_tmem_load();
      }
      //
      // mma1 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {
        { mma1SrcStack.mPipeline.consumer_release(mma1ConsState); }
        ++mma1ConsState;
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
      float* smemDqSfsAct7;
      float* smemDqSfsWeights7;
      {
        int32_t index{smemDeepSeekSfAbConsState.index()};
        smemDqSfsAct7 = &smemDeepSeekSfAbSrcSmem.mDqSfsAct[index][int32_t{0}];
        smemDqSfsWeights7 = &smemDeepSeekSfAbSrcSmem.mDqSfsWeights[index][int32_t{0}];
      }
      //
      // gmemC1 [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
      //
      float* smemDqSfsAct12;
      float* smemDqSfsW12;
      tmemBaseWithStageOffset12 = tmemBaseWithStageOffset10;
      smemDqSfsAct12 = smemDqSfsAct7;
      smemDqSfsW12 = smemDqSfsWeights7;
      {
        float dqSfW{smemDqSfsW12[int32_t{0}]};
        float dqSfAb0{(smemDqSfsAct12[mLdtm16dp256bitTmemColIdx]) * (dqSfW)};
        float dqSfAb1{(smemDqSfsAct12[(mLdtm16dp256bitTmemColIdx) + (int32_t{1})]) * (dqSfW)};
        {
          cutlass::Array<float, 2> dqSfsAb{dqSfAb0, dqSfAb1};
          cutlass::Array<float, 2> tmp{regsAcc00[int32_t{0}], regsAcc00[int32_t{1}]};
          cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{0}], regsPartialAcc00[int32_t{1}]};
          tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
          regsAcc00[int32_t{0}] = tmp[int32_t{0}];
          regsAcc00[int32_t{1}] = tmp[int32_t{1}];
        }
        {
          cutlass::Array<float, 2> dqSfsAb{dqSfAb0, dqSfAb1};
          cutlass::Array<float, 2> tmp{regsAcc00[int32_t{2}], regsAcc00[int32_t{3}]};
          cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{2}], regsPartialAcc00[int32_t{3}]};
          tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
          regsAcc00[int32_t{2}] = tmp[int32_t{0}];
          regsAcc00[int32_t{3}] = tmp[int32_t{1}];
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
    for (int32_t loopOffset1384 = int32_t{128}; loopOffset1384 < loopEnd;
         loopOffset1384 += int32_t{128}) {
      bool const isLastLoopIter{((loopOffset1384) + (int32_t{128})) >= (loopEnd)};
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
      uint32_t tmemBaseWithStageOffset10;
      {
        int32_t index{mma1ConsState.index()};
        uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{8})))};
        uint32_t ptrTmemOffsetD{(ptrTmemD) + (uint32_t{1048576})};
        tmemBaseWithStageOffset10 = ptrTmemOffsetD;
      }
      //
      // gmemC1 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      uint32_t tmemBaseWithStageOffset12;
      tmemBaseWithStageOffset12 = tmemBaseWithStageOffset10;
      {
        {
          uint32_t tmemBasePtr{tmemBaseWithStageOffset12};
          uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(regsPartialAcc00[int32_t{0}])};
          cuda_ptx::tcgen05_ld_16x256b(dstSlice0, tmemBasePtr);
        }
        cutlass::arch::fence_view_async_tmem_load();
      }
      //
      // mma1 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {
        { mma1SrcStack.mPipeline.consumer_release(mma1ConsState); }
        ++mma1ConsState;
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
      float* smemDqSfsAct7;
      float* smemDqSfsWeights7;
      {
        int32_t index{smemDeepSeekSfAbConsState.index()};
        smemDqSfsAct7 = &smemDeepSeekSfAbSrcSmem.mDqSfsAct[index][int32_t{0}];
        smemDqSfsWeights7 = &smemDeepSeekSfAbSrcSmem.mDqSfsWeights[index][int32_t{0}];
      }
      //
      // gmemC1 [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
      //
      float* smemDqSfsAct12;
      float* smemDqSfsW12;
      tmemBaseWithStageOffset12 = tmemBaseWithStageOffset10;
      smemDqSfsAct12 = smemDqSfsAct7;
      smemDqSfsW12 = smemDqSfsWeights7;
      {
        float dqSfW{smemDqSfsW12[int32_t{0}]};
        float dqSfAb0{(smemDqSfsAct12[mLdtm16dp256bitTmemColIdx]) * (dqSfW)};
        float dqSfAb1{(smemDqSfsAct12[(mLdtm16dp256bitTmemColIdx) + (int32_t{1})]) * (dqSfW)};
        {
          cutlass::Array<float, 2> dqSfsAb{dqSfAb0, dqSfAb1};
          cutlass::Array<float, 2> tmp{regsAcc00[int32_t{0}], regsAcc00[int32_t{1}]};
          cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{0}], regsPartialAcc00[int32_t{1}]};
          tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
          regsAcc00[int32_t{0}] = tmp[int32_t{0}];
          regsAcc00[int32_t{1}] = tmp[int32_t{1}];
        }
        {
          cutlass::Array<float, 2> dqSfsAb{dqSfAb0, dqSfAb1};
          cutlass::Array<float, 2> tmp{regsAcc00[int32_t{2}], regsAcc00[int32_t{3}]};
          cutlass::Array<float, 2> reg{regsPartialAcc00[int32_t{2}], regsPartialAcc00[int32_t{3}]};
          tmp = trtllm::dev::ffma2(dqSfsAb, reg, tmp);
          regsAcc00[int32_t{2}] = tmp[int32_t{0}];
          regsAcc00[int32_t{3}] = tmp[int32_t{1}];
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
      float qSfsC0[2]{float{1.3165538e-36}, float{1.3165538e-36}};
      //
      // Compute max for fragment idxM=0 idxN=0.
      //
      {
        trtllm::dev::reduceColMax16dp256bit<int32_t{1}, int32_t{1}, int32_t{1}, true>(qSfsC0,
                                                                                      regsAcc00);
        trtllm::dev::reduceColMax(qSfsC0,
                                  (ptrSmemRowMax + int32_t{0}),
                                  int32_t{256},
                                  mWarpGrpThreadIdx,
                                  int32_t{1});
      }
      qSfsC0[int32_t{0}] = (float{448}) / (qSfsC0[int32_t{0}]);
      qSfsC0[int32_t{1}] = (float{448}) / (qSfsC0[int32_t{1}]);
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
        cutlass::float_e4m3_t* ptrSmem;
        ptrSmemBase = reinterpret_cast<int8_t*>(gmemC1DstStack.mDepSmemPtr0) +
                      ((mWarpGrp4Idx) * (int32_t{512}) + (int32_t{140288}));
        ptrSmem = reinterpret_cast<cutlass::float_e4m3_t*>(ptrSmemBase) + (int32_t{0});
        //
        // Smem store idxM=0 idxN=0.
        //
        {
          int32_t smemOffset0;
          {
            int32_t const smemRowIdx{((mBaseTmemCol) * (int32_t{64}) + (mBaseRowIdx)) /
                                     (int32_t{128})};
            int32_t const smemOffsetInBytes{
              (((mBaseTmemCol) * (int32_t{64}) + (mBaseRowIdx)) * (int32_t{8})) / (int32_t{8})};
            int32_t const swizzleMask{((smemRowIdx) % (int32_t{4})) * (int32_t{16})};
            smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{8});
          }
          cutlass::Array<float, 2> scaleF2{qSfsC0[int32_t{0}], qSfsC0[int32_t{0}]};
          cutlass::Array<float, 2> accF2{regsAcc00[int32_t{0}], regsAcc00[int32_t{2}]};
          cutlass::Array<float, 2> scaledAccF2{trtllm::dev::fmul2(accF2, scaleF2)};
          cutlass::Array<cutlass::float_e4m3_t, 2> scaledCvtAcc2{
            trtllm::dev::convert_float2_to_e4m3(scaledAccF2)};
          {
            uint16_t convertedElts;
            convertedElts = reinterpret_cast<uint16_t&>(scaledCvtAcc2);
            reinterpret_cast<uint16_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
          }
        }
        //
        // Smem store idxM=0 idxN=1.
        //
        {
          int32_t smemOffset0;
          {
            int32_t const smemRowIdx{
              (((mBaseTmemCol) + (int32_t{1})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{128})};
            int32_t const smemOffsetInBytes{
              ((((mBaseTmemCol) + (int32_t{1})) * (int32_t{64}) + (mBaseRowIdx)) * (int32_t{8})) /
              (int32_t{8})};
            int32_t const swizzleMask{((smemRowIdx) % (int32_t{4})) * (int32_t{16})};
            smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{8});
          }
          cutlass::Array<float, 2> scaleF2{qSfsC0[int32_t{1}], qSfsC0[int32_t{1}]};
          cutlass::Array<float, 2> accF2{regsAcc00[int32_t{1}], regsAcc00[int32_t{3}]};
          cutlass::Array<float, 2> scaledAccF2{trtllm::dev::fmul2(accF2, scaleF2)};
          cutlass::Array<cutlass::float_e4m3_t, 2> scaledCvtAcc2{
            trtllm::dev::convert_float2_to_e4m3(scaledAccF2)};
          {
            uint16_t convertedElts;
            convertedElts = reinterpret_cast<uint16_t&>(scaledCvtAcc2);
            reinterpret_cast<uint16_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
          }
        }
        cuda_ptx::fence_proxy_async(cuda_ptx::space_shared_t{});
        trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{8}) + (mWarpGrp4Idx));
        //
        // Issue TMA from smem to gmem.
        //
        if ((bool{cute::elect_one_sync()}) && ((mWarpGrp4WarpIdx) == (int32_t{0}))) {
          int8_t* ptrSmemBase;
          cutlass::float_e4m3_t* ptrSmem;
          ptrSmemBase = reinterpret_cast<int8_t*>(gmemC1DstStack.mDepSmemPtr0) +
                        ((mWarpGrp4Idx) * (int32_t{512}) + (int32_t{140288}));
          ptrSmem = reinterpret_cast<cutlass::float_e4m3_t*>(ptrSmemBase) + (int32_t{0});
          int32_t coords[4];
          coords[int32_t{0}] = ((int32_t{2}) * (mCtaIdxX) + (int32_t{1})) * (int32_t{64});
          coords[int32_t{1}] = (((int32_t{8}) - ((mBatchLimit) % (int32_t{8}))) % (int32_t{8})) +
                               ((mWarpGrp4Idx) * (int32_t{8}));
          coords[int32_t{2}] = int32_t{0x40000000 /*1073741824*/};
          coords[int32_t{3}] =
            (((mCtaIdxY) * (int32_t{8})) +
             ((int32_t{0}) - (((int32_t{8}) - ((mBatchLimit) % (int32_t{8}))) % (int32_t{8})))) +
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
  ExitTileWithoutSignalingLabel: {}
  }
};
struct PaddingTask {
  inline __device__ PaddingTask(KernelParams const& params,
                                KernelState const& state,
                                int32_t warpGrpStart) {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{13})) && ((state.mWarpIdx) < (int32_t{16}));
  }
  inline __device__ void execute(KernelParams const& params, KernelState const& state) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<48>{});
  //
  // Tail work.
  //
  ExitTileWithSignalingLabel:
  ExitTileWithoutSignalingLabel: {}
  }
};
extern "C" __global__
__launch_bounds__(512, 1) void bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_cga1x1x1_16dp256b_rM_BN_transOut_dsFp8_schedS_bN_tma_tmaSf_rgTma_clmp_lbW2_dynB_sm100f(
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
  uint8_t* smemDeepSeekSfAbSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemDeepSeekSfAbSmem)});
  uint32_t* TmemSwStatePtr{
    reinterpret_cast<uint32_t*>((reinterpret_cast<uint8_t*>(smem__) + smemOffset__))};
  smemOffset__ += int32_t{16};
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
  cudaGridDependencySynchronize();
  KernelState const state{params, TmemSwStatePtr};
  if ((reinterpret_cast<int32_t const&>(blockIdx.y)) >= (state.mNumNonExitingCtas)) {
    return;
  }
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
                                              int32_t{12},
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
    cuda_ptx::tcgen05_alloc(cuda_ptx::cta_group_1_t{}, state.mTmemSwStatePtr, int32_t{32});
    cuda_ptx::tcgen05_relinquish_alloc_permit(cuda_ptx::cta_group_1_t{});
  }
  if (((bool{LoadTaskA::isSelected(params, state)}) ||
       (bool{LoadTaskB::isSelected(params, state)})) ||
      (bool{LoadSfAbTask::isSelected(params, state)})) {
  } else {
    trtllm::dev::CutlassNamedBarrier::sync(384, 10);
  }
  if (bool{LoadTaskA::isSelected(params, state)}) {
    loadTaskA.execute(params, state, (*smemASmem), smemAStack);
  } else {
    if (bool{LoadTaskB::isSelected(params, state)}) {
      loadTaskB.execute(params, state, (*smemBSmem), smemBStack);
    } else {
      if (bool{LoadSfAbTask::isSelected(params, state)}) {
        LoadSfAbTask loadSfAbTask{params, state, int32_t{12}};
        loadSfAbTask.execute(params, state, (*smemDeepSeekSfAbSmem), smemDeepSeekSfAbStack);
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
                           smemBStack);
        } else {
          if (bool{EpilogueTask0::isSelected(params, state)}) {
            EpilogueTask0 epilogueTask0{params, state, int32_t{0}};
            epilogueTask0.execute(params,
                                  state,
                                  (*gmemC0Smem),
                                  gmemC0Stack,
                                  mma0Stack,
                                  (*smemDeepSeekSfAbSmem),
                                  smemDeepSeekSfAbStack);
            trtllm::dev::CutlassNamedBarrier::sync(128, 11);
            int32_t const warpGrpThreadIdx{state.mThreadIdx};
            if ((warpGrpThreadIdx) < (int32_t{32})) {
              cuda_ptx::tcgen05_dealloc(cuda_ptx::cta_group_1_t{},
                                        uint32_t{__shfl_sync(uint32_t{0xffffffff},
                                                             (*state.mTmemSwStatePtr),
                                                             int32_t{0},
                                                             int32_t{32})},
                                        int32_t{32});
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
                                    smemDeepSeekSfAbStack);
            } else {
              if (bool{PaddingTask::isSelected(params, state)}) {
                PaddingTask paddingTask{params, state, int32_t{13}};
                paddingTask.execute(params, state);
              }
            }
          }
        }
      }
    }
  }
}
extern "C" __global__ void
bmm_E4m3_E4m3E4m3_Fp32_t128x8x128u2_s8_et64x8_m64x8x32_cga1x1x1_16dp256b_rM_BN_transOut_dsFp8_schedS_bN_tma_tmaSf_rgTma_clmp_lbW2_dynB_sm100fGetSmemSize(
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
  size += static_cast<int32_t>(sizeof(SmemDeepSeekSfAbSmem));
  size += 16;
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
