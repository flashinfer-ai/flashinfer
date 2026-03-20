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
#include <Bmm_E2m1_E2m1E2m1_Fp32_bA16_bB16_bC16_t128x32x512u2_s5_et128x32_m256x32x64_cga2x1x1_16dp256b_rM_TN_transOut_schedS_biasM_bN_tma_tmaSf_rgTma_clmp_geGlu_dynB_sm100f.h>
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
    : mDepSmemPtr0{&smemBufferSmem.mArray[int32_t{0}]}
    , mPipeline{smemASmemBarrier.mBarriers,
                warpId,
                int32_t{65536},
                ((warpId) == (barInitWarpId)) && (bool{cute::elect_one_sync()}),
                int32_t{1},
                CuteFlatTuple209{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId} {}
};
struct SmemBStack {
  int8_t* mDepSmemPtr0;
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
    : mDepSmemPtr0{&smemBufferSmem.mArray[int32_t{0}]}
    , mPipeline{smemBSmemBarrier.mBarriers,
                warpId,
                int32_t{8192},
                ((warpId) == (barInitWarpId)) && (bool{cute::elect_one_sync()}),
                int32_t{1},
                CuteFlatTuple335{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId} {}
};
struct SmemSfAStack {
  trtllm::dev::CutlassTmaUmmaAsyncPipeline<5,
                                           cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
                                           cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  cutlass::float_e4m3_t* mPtr;
  inline __device__ SmemSfAStack(SmemSfASmem& smemSfASmem,
                                 SmemSfASmemBarrier& smemSfASmemBarrier,
                                 int32_t warpId,
                                 int32_t barInitWarpId,
                                 int32_t orderedSequenceGroupId)
    : mPipeline{smemSfASmemBarrier.mBarriers,
                warpId,
                int32_t{8192},
                bool{cute::elect_one_sync()},
                CuteFlatTuple456{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId}
    , mPtr{&smemSfASmem.mArray[int32_t{0}][int32_t{0}]} {}
};
struct SmemSfBStack {
  trtllm::dev::CutlassTmaAsyncPipeline<5> mPipeline;
  cutlass::float_e4m3_t* mPtr;
  inline __device__ SmemSfBStack(SmemSfBSmem& smemSfBSmem,
                                 SmemSfBSmemBarrier& smemSfBSmemBarrier,
                                 int32_t warpId,
                                 int32_t barInitWarpId,
                                 int32_t orderedSequenceGroupId)
    : mPipeline{smemSfBSmemBarrier.mBarriers,
                warpId,
                int32_t{1024},
                ((warpId) == (barInitWarpId)) && (bool{cute::elect_one_sync()}),
                int32_t{128},
                CuteFlatTuple575{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId}
    , mPtr{&smemSfBSmem.mArray[int32_t{0}][int32_t{0}]} {}
};
struct TmemSfAStack {
  trtllm::dev::CutlassUmmaConsumerAsyncPipeline<
    5,
    false,
    false,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  inline __device__ TmemSfAStack(TmemSfASmemBarrier& tmemSfASmemBarrier,
                                 int32_t warpId,
                                 int32_t barInitWarpId,
                                 int32_t orderedSequenceGroupId)
    : mPipeline{tmemSfASmemBarrier.mBarriers,
                warpId,
                int32_t{32},
                CuteFlatTuple698{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId} {}
};
struct TmemSfBStack {
  trtllm::dev::CutlassUmmaConsumerAsyncPipeline<
    5,
    false,
    false,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  inline __device__ TmemSfBStack(TmemSfBSmemBarrier& tmemSfBSmemBarrier,
                                 int32_t warpId,
                                 int32_t barInitWarpId,
                                 int32_t orderedSequenceGroupId)
    : mPipeline{tmemSfBSmemBarrier.mBarriers,
                warpId,
                int32_t{256},
                CuteFlatTuple810{},
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
  trtllm::dev::CutlassUmmaAsyncPipeline<1, cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  inline __device__ Mma0Stack(Mma0SmemBarrier& mma0SmemBarrier,
                              TmemStack& tmemStack,
                              int32_t warpId,
                              int32_t barInitWarpId,
                              int32_t orderedSequenceGroupId)
    : mPipeline{mma0SmemBarrier.mBarriers,
                warpId,
                int32_t{256},
                CuteFlatTuple927{},
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
  int32_t const mBatchIdx;
  int32_t const mBatchLimit;
  int32_t mCtaOffsetK;
  int32_t mCtaIdxX;
  uint16_t const mMmaCtaMask;
  inline __device__ LoadTaskA(KernelParams const& params,
                              KernelState const& state,
                              int32_t warpGrpStart)
    : mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mBatchIdx{int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}}
    , mBatchLimit{int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]}}
    , mCtaOffsetK{int32_t{0}}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , mMmaCtaMask{uint16_t{
        __shfl_sync(uint32_t{0xffffffff}, trtllm::dev::getCtaMask(), int32_t{0}, int32_t{32})}} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{20})) && ((state.mWarpIdx) < (int32_t{21}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 SmemASmem& smemADstSmem,
                                 SmemAStack& smemADstStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<48>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      5,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemAProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    int32_t smemAProdToken{int32_t{1}};
    int32_t paddedPerCtaK{(((params.k) + (int32_t{511})) / (int32_t{512})) * (int32_t{512})};
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
    for (int32_t loopOffset353 = int32_t{0}; loopOffset353 < loopEnd;
         loopOffset353 += int32_t{512}) {
      bool const isLastLoopIter{((loopOffset353) + (int32_t{512})) >= (loopEnd)};
      //
      // gmemA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK1;
      { tileOffsetK1 = loopOffset353; }
      //
      // smemA [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {
        smemADstStack.mPipeline.producer_acquire(smemAProdState, smemAProdToken);
        if (((loopOffset353) + (int32_t{512})) < (loopEnd)) {
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
          smemBytesStagePtrA = smemBytesBasePtrA + ((index) * (int32_t{32768}));
          int32_t coords[3];
          coords[int32_t{0}] = tileOffsetK5;
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
              &reinterpret_cast<cutlass::float_e2m1_t*>(smemBytesStagePtrA)[int32_t{0}],
              params.tmaA,
              coords,
              leadCtaMbar,
              mMmaCtaMask);
          }
          coords[int32_t{0}] += int32_t{256};
          if (bool{cute::elect_one_sync()}) {
            cuda_ptx::cp_async_bulk_tensor(
              cuda_ptx::space_cluster_t{},
              cuda_ptx::space_global_t{},
              cuda_ptx::cta_group_2_t{},
              &reinterpret_cast<cutlass::float_e2m1_t*>(smemBytesStagePtrA)[int32_t{16384}],
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
      if ((loopOffset353) >= (int32_t{0})) {
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
  uint16_t const mMmaCtaMask;
  inline __device__ LoadTaskB(KernelParams const& params,
                              KernelState const& state,
                              int32_t warpGrpStart)
    : mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mBatchIdx{int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}}
    , mBatchLimit{int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]}}
    , mWarpGrpThreadIdx{min(int32_t{128},
                            max(int32_t{0}, (state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))))}
    , mWarpGrpWarpIdx{(state.mWarpIdx) - (warpGrpStart)}
    , mCtaOffsetK{int32_t{0}}
    , mMmaCtaMask{uint16_t{
        __shfl_sync(uint32_t{0xffffffff}, trtllm::dev::getCtaMask(), int32_t{0}, int32_t{32})}} {
    cudaGridDependencySynchronize();
    {
      if ((((mWarpGrpWarpIdx) * (int32_t{4})) < (int32_t{16})) &&
          (bool{LoadTaskB::isSelected(params, state)})) {
        mRoutedIndices[int32_t{0}] = int32_t{
          params.ptrRouteMap[((mWarpGrpWarpIdx) * (int32_t{4})) +
                             ((mCtaIdxY) * (int32_t{32}) +
                              ((int32_t{trtllm::dev::getCtaRankInPair()}) * (int32_t{16})))]};
        mRoutedIndices[int32_t{1}] = int32_t{
          params.ptrRouteMap[((mWarpGrpWarpIdx) * (int32_t{4}) + (int32_t{1})) +
                             ((mCtaIdxY) * (int32_t{32}) +
                              ((int32_t{trtllm::dev::getCtaRankInPair()}) * (int32_t{16})))]};
        mRoutedIndices[int32_t{2}] = int32_t{
          params.ptrRouteMap[((mWarpGrpWarpIdx) * (int32_t{4}) + (int32_t{2})) +
                             ((mCtaIdxY) * (int32_t{32}) +
                              ((int32_t{trtllm::dev::getCtaRankInPair()}) * (int32_t{16})))]};
        mRoutedIndices[int32_t{3}] = int32_t{
          params.ptrRouteMap[((mWarpGrpWarpIdx) * (int32_t{4}) + (int32_t{3})) +
                             ((mCtaIdxY) * (int32_t{32}) +
                              ((int32_t{trtllm::dev::getCtaRankInPair()}) * (int32_t{16})))]};
      }
    }
  }
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{8})) && ((state.mWarpIdx) < (int32_t{12}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 SmemBSmem& smemBDstSmem,
                                 SmemBStack& smemBDstStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<48>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      5,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemBProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    int32_t smemBProdToken{int32_t{1}};
    int32_t paddedPerCtaK{(((params.k) + (int32_t{511})) / (int32_t{512})) * (int32_t{512})};
    int32_t loopEnd{paddedPerCtaK};
    bool const hasOneLoopIter{(int32_t{0}) < (loopEnd)};
    //
    // Hoist the first iter.
    //
    //
    // Loop body.
    //
    CUTLASS_PRAGMA_NO_UNROLL
    for (int32_t loopOffset441 = int32_t{0}; loopOffset441 < loopEnd;
         loopOffset441 += int32_t{512}) {
      bool const isLastLoopIter{((loopOffset441) + (int32_t{512})) >= (loopEnd)};
      //
      // gmemB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK2;
      { tileOffsetK2 = loopOffset441; }
      //
      // smemB [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {
        if ((loopOffset441) >= (int32_t{2560})) {
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
            reinterpret_cast<int8_t*>(smemBDstStack.mDepSmemPtr0) + (int32_t{163840});
          smemBytesStagePtrB = smemBytesBasePtrB + ((index) * (int32_t{4096}));
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
                (((mWarpGrpWarpIdx) * (int32_t{4})) < (int32_t{16}))) {
              cuda_ptx::cp_async_bulk_tensor_tile_gather4(
                cuda_ptx::space_cluster_t{},
                cuda_ptx::space_global_t{},
                cuda_ptx::cta_group_2_t{},
                &reinterpret_cast<cutlass::float_e2m1_t*>(
                  smemBytesStagePtrB)[(((mWarpGrpWarpIdx) * (int32_t{4})) * (int32_t{256})) /
                                      (int32_t{2})],
                params.tmaB,
                coords,
                leadCtaMbar,
                mMmaCtaMask);
            }
            coords[int32_t{0}] += int32_t{256};
            if ((bool{cute::elect_one_sync()}) &&
                (((mWarpGrpWarpIdx) * (int32_t{4})) < (int32_t{16}))) {
              cuda_ptx::cp_async_bulk_tensor_tile_gather4(
                cuda_ptx::space_cluster_t{},
                cuda_ptx::space_global_t{},
                cuda_ptx::cta_group_2_t{},
                &reinterpret_cast<cutlass::float_e2m1_t*>(
                  smemBytesStagePtrB)[((((mWarpGrpWarpIdx) * (int32_t{4})) * (int32_t{256})) +
                                       (int32_t{4096})) /
                                      (int32_t{2})],
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
      if ((loopOffset441) >= (int32_t{0})) {
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
struct LoadSfATask {
  int32_t mCtaIdxY;
  int32_t const mBatchIdx;
  int32_t const mBatchLimit;
  int32_t mCtaOffsetK;
  int32_t const mWarpGrpThreadIdx;
  int32_t mCtaIdxX;
  inline __device__ LoadSfATask(KernelParams const& params,
                                KernelState const& state,
                                int32_t warpGrpStart)
    : mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mBatchIdx{int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}}
    , mBatchLimit{int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]}}
    , mCtaOffsetK{int32_t{0}}
    , mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{21})) && ((state.mWarpIdx) < (int32_t{22}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 SmemSfASmem& smemSfADstSmem,
                                 SmemSfAStack& smemSfADstStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<48>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      5,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemSfAProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    int32_t smemSfAProdToken{int32_t{1}};
    int32_t paddedPerCtaK{(((params.k) + (int32_t{511})) / (int32_t{512})) * (int32_t{512})};
    int32_t loopEnd{paddedPerCtaK};
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
      // smemSfA [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
      { smemSfADstStack.mPipeline.producer_acquire(smemSfAProdState, smemSfAProdToken); }
      //
      // smemSfA [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK7;
      tileOffsetK7 = tileOffsetK3;
      {
        uint64_t* barrier{smemSfADstStack.mPipeline.producer_get_barrier(smemSfAProdState)};
        int32_t index{smemSfAProdState.index()};
        {}
        {
          int32_t coords[4];
          coords[int32_t{0}] = int32_t{0};
          coords[int32_t{1}] = int32_t{0};
          coords[int32_t{2}] = (tileOffsetK7) / (int32_t{64});
          coords[int32_t{3}] = (mBatchIdx) * (params.tileStridePerBatch) + (mCtaIdxX);
          uint64_t* leadCtaMbar;
          leadCtaMbar = cuda_ptx::mapa(cuda_ptx::space_cluster_t{},
                                       barrier,
                                       int32_t{trtllm::dev::getLeadCtaRank()});
          uint16_t ctaMask{
            __shfl_sync(uint32_t{0xffffffff}, trtllm::dev::getCtaMask(), int32_t{0}, int32_t{32})};
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
    for (int32_t loopOffset560 = int32_t{512}; loopOffset560 < loopEnd;
         loopOffset560 += int32_t{512}) {
      bool const isFirstLoopIter{(loopOffset560) == (int32_t{512})};
      bool const isLastLoopIter{((loopOffset560) + (int32_t{512})) >= (loopEnd)};
      //
      // smemSfA [ProdPreCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if ((loopOffset560) >= (int32_t{512})) {
      }
      //
      // smemSfA [ProdCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if ((loopOffset560) >= (int32_t{512})) {
        {
          smemSfADstStack.mPipeline.producer_commit(smemSfAProdState);
        }
        ++smemSfAProdState;
      }
      //
      // gmemSfA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK3;
      { tileOffsetK3 = loopOffset560; }
      //
      // smemSfA [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {
        if ((loopOffset560) >= (int32_t{2560})) {
          smemSfAProdToken = smemSfADstStack.mPipeline.producer_try_acquire(smemSfAProdState);
        }
      }
      { smemSfADstStack.mPipeline.producer_acquire(smemSfAProdState, smemSfAProdToken); }
      //
      // smemSfA [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK7;
      tileOffsetK7 = tileOffsetK3;
      {
        uint64_t* barrier{smemSfADstStack.mPipeline.producer_get_barrier(smemSfAProdState)};
        int32_t index{smemSfAProdState.index()};
        {}
        {
          int32_t coords[4];
          coords[int32_t{0}] = int32_t{0};
          coords[int32_t{1}] = int32_t{0};
          coords[int32_t{2}] = (tileOffsetK7) / (int32_t{64});
          coords[int32_t{3}] = (mBatchIdx) * (params.tileStridePerBatch) + (mCtaIdxX);
          uint64_t* leadCtaMbar;
          leadCtaMbar = cuda_ptx::mapa(cuda_ptx::space_cluster_t{},
                                       barrier,
                                       int32_t{trtllm::dev::getLeadCtaRank()});
          uint16_t ctaMask{
            __shfl_sync(uint32_t{0xffffffff}, trtllm::dev::getCtaMask(), int32_t{0}, int32_t{32})};
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
    // smemSfA [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    {}
  ExitTileWithSignalingLabel:
  ExitTileWithoutSignalingLabel: {}
  }
};
struct LoadSfBTask {
  int32_t mCtaIdxY;
  int32_t const mBatchIdx;
  int32_t const mBatchLimit;
  int32_t mCtaOffsetK;
  int32_t const mWarpGrpThreadIdx;
  int32_t const mWarpGrpWarpIdx;
  inline __device__ LoadSfBTask(KernelParams const& params,
                                KernelState const& state,
                                int32_t warpGrpStart)
    : mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , mBatchIdx{int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}}
    , mBatchLimit{int32_t{params.ptrCtaIdxXyToMnLimit[mCtaIdxY]}}
    , mCtaOffsetK{int32_t{0}}
    , mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))}
    , mWarpGrpWarpIdx{(state.mWarpIdx) - (warpGrpStart)} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{12})) && ((state.mWarpIdx) < (int32_t{20}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 SmemSfBSmem& smemSfBDstSmem,
                                 SmemSfBStack& smemSfBDstStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<48>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassTmaAsyncPipeline<5>::PipelineState smemSfBProdState{int32_t{0},
                                                                            int32_t{1},
                                                                            int32_t{0}};
    int32_t smemSfBProdToken{int32_t{1}};
    int32_t paddedPerCtaK{(((params.k) + (int32_t{511})) / (int32_t{512})) * (int32_t{512})};
    int32_t loopEnd{paddedPerCtaK};
    int32_t gmemRowIdxRouted0;
    int32_t gmemRowIdxRouted1;
    int32_t gmemRowIdxRouted2;
    int32_t gmemRowIdxRouted3;
    if (((mWarpGrpWarpIdx) * (int32_t{4})) < (int32_t{32})) {
      int32_t gmemRowIdxOffset{((mWarpGrpWarpIdx) * (int32_t{4})) + ((mCtaIdxY) * (int32_t{32}))};
      gmemRowIdxRouted0 = int32_t{params.ptrRouteMap[gmemRowIdxOffset]};
      gmemRowIdxRouted1 = int32_t{params.ptrRouteMap[(gmemRowIdxOffset) + (int32_t{1})]};
      gmemRowIdxRouted2 = int32_t{params.ptrRouteMap[(gmemRowIdxOffset) + (int32_t{2})]};
      gmemRowIdxRouted3 = int32_t{params.ptrRouteMap[(gmemRowIdxOffset) + (int32_t{3})]};
    }
    //
    // Unrolled head iter 0.
    //
    if ((int32_t{0}) < (loopEnd)) {
      //
      // gmemSfB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK4;
      { tileOffsetK4 = int32_t{0}; }
      //
      // smemSfB [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {}
      { smemSfBDstStack.mPipeline.producer_acquire(smemSfBProdState, smemSfBProdToken); }
      //
      // smemSfB [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK8;
      tileOffsetK8 = tileOffsetK4;
      {
        uint64_t* barrier{smemSfBDstStack.mPipeline.producer_get_barrier(smemSfBProdState)};
        int32_t index{smemSfBProdState.index()};
        {
          {
            int32_t coords[5];
            coords[int32_t{0}] = (tileOffsetK8) / (int32_t{16});
            coords[int32_t{1}] = gmemRowIdxRouted0;
            coords[int32_t{2}] = gmemRowIdxRouted1;
            coords[int32_t{3}] = gmemRowIdxRouted2;
            coords[int32_t{4}] = gmemRowIdxRouted3;
            if ((((mWarpGrpWarpIdx) * (int32_t{4})) < (int32_t{32})) &&
                (bool{cute::elect_one_sync()})) {
              cuda_ptx::cp_async_bulk_tensor_tile_gather4(
                cuda_ptx::space_shared_t{},
                cuda_ptx::space_global_t{},
                &reinterpret_cast<cutlass::float_e4m3_t*>(&smemSfBDstSmem.mArray[index][int32_t{
                  0}])[((mWarpGrpWarpIdx) * (int32_t{4})) * (int32_t{32})],
                params.tmaSfB,
                coords,
                barrier);
            }
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
    for (int32_t loopOffset693 = int32_t{512}; loopOffset693 < loopEnd;
         loopOffset693 += int32_t{512}) {
      bool const isFirstLoopIter{(loopOffset693) == (int32_t{512})};
      bool const isLastLoopIter{((loopOffset693) + (int32_t{512})) >= (loopEnd)};
      //
      // smemSfB [ProdPreCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if ((loopOffset693) >= (int32_t{512})) {
      }
      //
      // smemSfB [ProdCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if ((loopOffset693) >= (int32_t{512})) {
        {
          smemSfBDstStack.mPipeline.producer_commit(smemSfBProdState);
        }
        ++smemSfBProdState;
      }
      //
      // gmemSfB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK4;
      { tileOffsetK4 = loopOffset693; }
      //
      // smemSfB [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {
        if ((loopOffset693) >= (int32_t{2560})) {
          smemSfBProdToken = smemSfBDstStack.mPipeline.producer_try_acquire(smemSfBProdState);
        }
      }
      { smemSfBDstStack.mPipeline.producer_acquire(smemSfBProdState, smemSfBProdToken); }
      //
      // smemSfB [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK8;
      tileOffsetK8 = tileOffsetK4;
      {
        uint64_t* barrier{smemSfBDstStack.mPipeline.producer_get_barrier(smemSfBProdState)};
        int32_t index{smemSfBProdState.index()};
        {
          {
            int32_t coords[5];
            coords[int32_t{0}] = (tileOffsetK8) / (int32_t{16});
            coords[int32_t{1}] = gmemRowIdxRouted0;
            coords[int32_t{2}] = gmemRowIdxRouted1;
            coords[int32_t{3}] = gmemRowIdxRouted2;
            coords[int32_t{4}] = gmemRowIdxRouted3;
            if ((((mWarpGrpWarpIdx) * (int32_t{4})) < (int32_t{32})) &&
                (bool{cute::elect_one_sync()})) {
              cuda_ptx::cp_async_bulk_tensor_tile_gather4(
                cuda_ptx::space_shared_t{},
                cuda_ptx::space_global_t{},
                &reinterpret_cast<cutlass::float_e4m3_t*>(&smemSfBDstSmem.mArray[index][int32_t{
                  0}])[((mWarpGrpWarpIdx) * (int32_t{4})) * (int32_t{32})],
                params.tmaSfB,
                coords,
                barrier);
            }
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
    // smemSfB [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    {}
  ExitTileWithSignalingLabel:
  ExitTileWithoutSignalingLabel: {}
    cudaTriggerProgrammaticLaunchCompletion();
  }
};
struct CopySfATask {
  int32_t mCtaOffsetK;
  uint32_t const mTmemBaseOffset;
  inline __device__ CopySfATask(KernelParams const& params,
                                KernelState const& state,
                                int32_t warpGrpStart)
    : mCtaOffsetK{int32_t{0}}
    , mTmemBaseOffset{uint32_t{
        __shfl_sync(uint32_t{0xffffffff}, (*state.mTmemSwStatePtr), int32_t{0}, int32_t{32})}} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{22})) && ((state.mWarpIdx) < (int32_t{23}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 TmemSfAStack& tmemSfADstStack,
                                 SmemSfASmem& smemSfASrcSmem,
                                 SmemSfAStack& smemSfASrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<48>{});
    if (!((int32_t{cute::block_rank_in_cluster()}) == (int32_t{trtllm::dev::getLeadCtaRank()}))) {
      return;
    }
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      5,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState smemSfAConsState{};
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      5,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState
      smemSfAConsReleaseState{};
    int32_t smemSfAConsToken{int32_t{0}};
    trtllm::dev::CutlassUmmaConsumerAsyncPipeline<
      5,
      false,
      false,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState tmemSfAProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    int32_t tmemSfAProdToken{int32_t{1}};
    int32_t paddedPerCtaK{(((params.k) + (int32_t{511})) / (int32_t{512})) * (int32_t{512})};
    int32_t loopEnd{paddedPerCtaK};
    //
    // Loop body.
    //
    CUTLASS_PRAGMA_NO_UNROLL
    for (int32_t loopOffset787 = int32_t{0}; loopOffset787 < loopEnd;
         loopOffset787 += int32_t{512}) {
      bool const isFirstLoopIter{(loopOffset787) == (int32_t{0})};
      bool const isLastLoopIter{((loopOffset787) + (int32_t{512})) >= (loopEnd)};
      //
      // tmemSfA [ProdPreCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if ((loopOffset787) >= (int32_t{512})) {
      }
      //
      // tmemSfA [ProdCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if ((loopOffset787) >= (int32_t{512})) {
        {
          tmemSfADstStack.mPipeline.producer_commit(tmemSfAProdState);
        }
        ++tmemSfAProdState;
      }
      //
      // smemSfA [ConsRelease, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if ((loopOffset787) >= (int32_t{512})) {
        if ((loopOffset787) < ((loopEnd) - (int32_t{2048}))) {
          smemSfASrcStack.mPipeline.consumer_release(smemSfAConsReleaseState);
        }
        ++smemSfAConsReleaseState;
      }
      //
      // smemSfA [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {
        { smemSfAConsToken = smemSfASrcStack.mPipeline.consumer_try_wait(smemSfAConsState); }
        smemSfASrcStack.mPipeline.consumer_wait(smemSfAConsState, smemSfAConsToken);
      }
      //
      // smemSfA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      cutlass::float_e4m3_t* smemPtrSfA7;
      {
        int32_t index{smemSfAConsState.index()};
        smemPtrSfA7 = smemSfASrcStack.mPtr + ((index) * (int32_t{4096}));
        ++smemSfAConsState;
      }
      //
      // tmemSfA [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {
        if ((loopOffset787) >= (int32_t{2560})) {
          tmemSfAProdToken = tmemSfADstStack.mPipeline.producer_try_acquire(tmemSfAProdState);
        }
      }
      { tmemSfADstStack.mPipeline.producer_acquire(tmemSfAProdState, tmemSfAProdToken); }
      //
      // tmemSfA [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      cutlass::float_e4m3_t* smemPtrSfA9;
      smemPtrSfA9 = smemPtrSfA7;
      {
        int32_t index{tmemSfAProdState.index()};
        {
          uint32_t tmemBaseAddr{((mTmemBaseOffset) + (uint32_t{32})) +
                                ((static_cast<uint32_t>(index)) * (uint32_t{32}))};
          uint64_t smemDesc{
            trtllm::dev::createSmemDesc(smemPtrSfA9, uint32_t{65536}, uint32_t{16392})};
          {
            {
              uint32_t tmemAddr{tmemBaseAddr};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_cp_32x128b_warpx4(cuda_ptx::cta_group_2_t{}, tmemAddr, smemDesc);
              }
            }
          }
          {
            trtllm::dev::incrSmemAddr(smemDesc, int32_t{32});
            {
              uint32_t tmemAddr{(tmemBaseAddr) + (uint32_t{4})};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_cp_32x128b_warpx4(cuda_ptx::cta_group_2_t{}, tmemAddr, smemDesc);
              }
            }
          }
          {
            trtllm::dev::incrSmemAddr(smemDesc, int32_t{32});
            {
              uint32_t tmemAddr{(tmemBaseAddr) + (uint32_t{8})};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_cp_32x128b_warpx4(cuda_ptx::cta_group_2_t{}, tmemAddr, smemDesc);
              }
            }
          }
          {
            trtllm::dev::incrSmemAddr(smemDesc, int32_t{32});
            {
              uint32_t tmemAddr{(tmemBaseAddr) + (uint32_t{12})};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_cp_32x128b_warpx4(cuda_ptx::cta_group_2_t{}, tmemAddr, smemDesc);
              }
            }
          }
          {
            trtllm::dev::incrSmemAddr(smemDesc, int32_t{32});
            {
              uint32_t tmemAddr{(tmemBaseAddr) + (uint32_t{16})};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_cp_32x128b_warpx4(cuda_ptx::cta_group_2_t{}, tmemAddr, smemDesc);
              }
            }
          }
          {
            trtllm::dev::incrSmemAddr(smemDesc, int32_t{32});
            {
              uint32_t tmemAddr{(tmemBaseAddr) + (uint32_t{20})};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_cp_32x128b_warpx4(cuda_ptx::cta_group_2_t{}, tmemAddr, smemDesc);
              }
            }
          }
          {
            trtllm::dev::incrSmemAddr(smemDesc, int32_t{32});
            {
              uint32_t tmemAddr{(tmemBaseAddr) + (uint32_t{24})};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_cp_32x128b_warpx4(cuda_ptx::cta_group_2_t{}, tmemAddr, smemDesc);
              }
            }
          }
          {
            trtllm::dev::incrSmemAddr(smemDesc, int32_t{32});
            {
              uint32_t tmemAddr{(tmemBaseAddr) + (uint32_t{28})};
              if (bool{cute::elect_one_sync()}) {
                cuda_ptx::tcgen05_cp_32x128b_warpx4(cuda_ptx::cta_group_2_t{}, tmemAddr, smemDesc);
              }
            }
          }
        }
      }
      //
      // smemSfA [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      //
      // tmemSfA [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
    }
    //
    // Unrolled tail iter 0.
    //
    if ((loopEnd) > (int32_t{0})) {
      //
      // tmemSfA [ProdPreCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if ((loopEnd) >= (int32_t{512})) {
      }
      //
      // tmemSfA [ProdCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if ((loopEnd) >= (int32_t{512})) {
        {
          tmemSfADstStack.mPipeline.producer_commit(tmemSfAProdState);
        }
        ++tmemSfAProdState;
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
    // tmemSfA [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    {}
  ExitTileWithSignalingLabel:
  ExitTileWithoutSignalingLabel: {}
  }
};
struct CopySfBTask {
  int32_t mCtaOffsetK;
  uint32_t const mTmemBaseOffset;
  int32_t const mLaneIdx;
  inline __device__ CopySfBTask(KernelParams const& params,
                                KernelState const& state,
                                int32_t warpGrpStart)
    : mCtaOffsetK{int32_t{0}}
    , mTmemBaseOffset{uint32_t{
        __shfl_sync(uint32_t{0xffffffff}, (*state.mTmemSwStatePtr), int32_t{0}, int32_t{32})}}
    , mLaneIdx{(state.mThreadIdx) % (int32_t{32})} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{4})) && ((state.mWarpIdx) < (int32_t{8}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 TmemSfBStack& tmemSfBDstStack,
                                 SmemSfBSmem& smemSfBSrcSmem,
                                 SmemSfBStack& smemSfBSrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<72>{});
    trtllm::dev::CutlassTmaAsyncPipeline<5>::PipelineState smemSfBConsState{};
    trtllm::dev::CutlassTmaAsyncPipeline<5>::PipelineState smemSfBConsReleaseState{};
    int32_t smemSfBConsToken{int32_t{0}};
    trtllm::dev::CutlassUmmaConsumerAsyncPipeline<
      5,
      false,
      false,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState tmemSfBProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    int32_t tmemSfBProdToken{int32_t{1}};
    int32_t paddedPerCtaK{(((params.k) + (int32_t{511})) / (int32_t{512})) * (int32_t{512})};
    int32_t loopEnd{paddedPerCtaK};
    //
    // Loop body.
    //
    CUTLASS_PRAGMA_NO_UNROLL
    for (int32_t loopOffset949 = int32_t{0}; loopOffset949 < loopEnd;
         loopOffset949 += int32_t{512}) {
      bool const isFirstLoopIter{(loopOffset949) == (int32_t{0})};
      bool const isLastLoopIter{((loopOffset949) + (int32_t{512})) >= (loopEnd)};
      //
      // tmemSfB [ProdPreCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if ((loopOffset949) >= (int32_t{512})) {
      }
      //
      // tmemSfB [ProdCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if ((loopOffset949) >= (int32_t{512})) {
        {
          tmemSfBDstStack.mPipeline.producer_commit(tmemSfBProdState);
        }
        ++tmemSfBProdState;
      }
      //
      // smemSfB [ConsRelease, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if ((loopOffset949) >= (int32_t{512})) {
        trtllm::dev::CutlassNamedBarrier::sync(128, 4);
        if ((loopOffset949) < ((loopEnd) - (int32_t{2048}))) {
          smemSfBSrcStack.mPipeline.consumer_release(smemSfBConsReleaseState);
        }
        ++smemSfBConsReleaseState;
      }
      //
      // smemSfB [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {
        { smemSfBConsToken = smemSfBSrcStack.mPipeline.consumer_try_wait(smemSfBConsState); }
        smemSfBSrcStack.mPipeline.consumer_wait(smemSfBConsState, smemSfBConsToken);
      }
      //
      // smemSfB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      cutlass::float_e4m3_t* smemPtrSfB8;
      {
        int32_t index{smemSfBConsState.index()};
        smemPtrSfB8 = smemSfBSrcStack.mPtr + ((index) * (int32_t{1024}));
        ++smemSfBConsState;
      }
      //
      // tmemSfB [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {
        if ((loopOffset949) >= (int32_t{2560})) {
          tmemSfBProdToken = tmemSfBDstStack.mPipeline.producer_try_acquire(tmemSfBProdState);
        }
      }
      { tmemSfBDstStack.mPipeline.producer_acquire(tmemSfBProdState, tmemSfBProdToken); }
      //
      // tmemSfB [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      cutlass::float_e4m3_t* smemPtrSfB10;
      smemPtrSfB10 = smemPtrSfB8;
      {
        int32_t index{tmemSfBProdState.index()};
        {
          cutlass::uint128_t* smemVecPtr;
          smemVecPtr = reinterpret_cast<cutlass::uint128_t*>(smemPtrSfB10) + (int32_t{0});
          {
            cutlass::Array<cutlass::uint128_t, 1> sfArray;
            sfArray[int32_t{0}] = smemVecPtr[((mLaneIdx) * (int32_t{2})) ^
                                             (((mLaneIdx) / (int32_t{4})) % (int32_t{2}))];
            cutlass::Array<uint32_t, 1> sfArraySttm_0_0;
            sfArraySttm_0_0[int32_t{0}] =
              uint32_t{reinterpret_cast<uint32_t const(&)[4]>(sfArray[int32_t{0}])[int32_t{0}]};
            {
              uint32_t tmemBasePtr{((mTmemBaseOffset) + (uint32_t{192})) +
                                   ((static_cast<uint32_t>(index)) * (uint32_t{16}))};
              uint32_t const(&srcSlice0)[1]{
                reinterpret_cast<uint32_t const(&)[1]>(sfArraySttm_0_0[int32_t{0}])};
              cuda_ptx::tcgen05_st_32x32b(tmemBasePtr, srcSlice0);
            }
            cutlass::Array<uint32_t, 1> sfArraySttm_1_0;
            sfArraySttm_1_0[int32_t{0}] =
              uint32_t{reinterpret_cast<uint32_t const(&)[4]>(sfArray[int32_t{0}])[int32_t{1}]};
            {
              uint32_t tmemBasePtr{((mTmemBaseOffset) + (uint32_t{192})) +
                                   ((static_cast<uint32_t>(index)) * (uint32_t{16}))};
              uint32_t const(&srcSlice0)[1]{
                reinterpret_cast<uint32_t const(&)[1]>(sfArraySttm_1_0[int32_t{0}])};
              cuda_ptx::tcgen05_st_32x32b((tmemBasePtr) + (uint32_t{2}), srcSlice0);
            }
            cutlass::Array<uint32_t, 1> sfArraySttm_2_0;
            sfArraySttm_2_0[int32_t{0}] =
              uint32_t{reinterpret_cast<uint32_t const(&)[4]>(sfArray[int32_t{0}])[int32_t{2}]};
            {
              uint32_t tmemBasePtr{((mTmemBaseOffset) + (uint32_t{192})) +
                                   ((static_cast<uint32_t>(index)) * (uint32_t{16}))};
              uint32_t const(&srcSlice0)[1]{
                reinterpret_cast<uint32_t const(&)[1]>(sfArraySttm_2_0[int32_t{0}])};
              cuda_ptx::tcgen05_st_32x32b((tmemBasePtr) + (uint32_t{4}), srcSlice0);
            }
            cutlass::Array<uint32_t, 1> sfArraySttm_3_0;
            sfArraySttm_3_0[int32_t{0}] =
              uint32_t{reinterpret_cast<uint32_t const(&)[4]>(sfArray[int32_t{0}])[int32_t{3}]};
            {
              uint32_t tmemBasePtr{((mTmemBaseOffset) + (uint32_t{192})) +
                                   ((static_cast<uint32_t>(index)) * (uint32_t{16}))};
              uint32_t const(&srcSlice0)[1]{
                reinterpret_cast<uint32_t const(&)[1]>(sfArraySttm_3_0[int32_t{0}])};
              cuda_ptx::tcgen05_st_32x32b((tmemBasePtr) + (uint32_t{6}), srcSlice0);
            }
          }
          {
            cutlass::Array<cutlass::uint128_t, 1> sfArray;
            sfArray[int32_t{0}] = smemVecPtr[(((mLaneIdx) * (int32_t{2})) + (int32_t{1})) ^
                                             (((mLaneIdx) / (int32_t{4})) % (int32_t{2}))];
            cutlass::Array<uint32_t, 1> sfArraySttm_0_0;
            sfArraySttm_0_0[int32_t{0}] =
              uint32_t{reinterpret_cast<uint32_t const(&)[4]>(sfArray[int32_t{0}])[int32_t{0}]};
            {
              uint32_t tmemBasePtr{((mTmemBaseOffset) + (uint32_t{192})) +
                                   ((static_cast<uint32_t>(index)) * (uint32_t{16}))};
              uint32_t const(&srcSlice0)[1]{
                reinterpret_cast<uint32_t const(&)[1]>(sfArraySttm_0_0[int32_t{0}])};
              cuda_ptx::tcgen05_st_32x32b((tmemBasePtr) + (uint32_t{8}), srcSlice0);
            }
            cutlass::Array<uint32_t, 1> sfArraySttm_1_0;
            sfArraySttm_1_0[int32_t{0}] =
              uint32_t{reinterpret_cast<uint32_t const(&)[4]>(sfArray[int32_t{0}])[int32_t{1}]};
            {
              uint32_t tmemBasePtr{((mTmemBaseOffset) + (uint32_t{192})) +
                                   ((static_cast<uint32_t>(index)) * (uint32_t{16}))};
              uint32_t const(&srcSlice0)[1]{
                reinterpret_cast<uint32_t const(&)[1]>(sfArraySttm_1_0[int32_t{0}])};
              cuda_ptx::tcgen05_st_32x32b((tmemBasePtr) + (uint32_t{10}), srcSlice0);
            }
            cutlass::Array<uint32_t, 1> sfArraySttm_2_0;
            sfArraySttm_2_0[int32_t{0}] =
              uint32_t{reinterpret_cast<uint32_t const(&)[4]>(sfArray[int32_t{0}])[int32_t{2}]};
            {
              uint32_t tmemBasePtr{((mTmemBaseOffset) + (uint32_t{192})) +
                                   ((static_cast<uint32_t>(index)) * (uint32_t{16}))};
              uint32_t const(&srcSlice0)[1]{
                reinterpret_cast<uint32_t const(&)[1]>(sfArraySttm_2_0[int32_t{0}])};
              cuda_ptx::tcgen05_st_32x32b((tmemBasePtr) + (uint32_t{12}), srcSlice0);
            }
            cutlass::Array<uint32_t, 1> sfArraySttm_3_0;
            sfArraySttm_3_0[int32_t{0}] =
              uint32_t{reinterpret_cast<uint32_t const(&)[4]>(sfArray[int32_t{0}])[int32_t{3}]};
            {
              uint32_t tmemBasePtr{((mTmemBaseOffset) + (uint32_t{192})) +
                                   ((static_cast<uint32_t>(index)) * (uint32_t{16}))};
              uint32_t const(&srcSlice0)[1]{
                reinterpret_cast<uint32_t const(&)[1]>(sfArraySttm_3_0[int32_t{0}])};
              cuda_ptx::tcgen05_st_32x32b((tmemBasePtr) + (uint32_t{14}), srcSlice0);
            }
          }
          cutlass::arch::fence_view_async_tmem_store();
        }
      }
      //
      // smemSfB [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      //
      // tmemSfB [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
    }
    //
    // Unrolled tail iter 0.
    //
    if ((loopEnd) > (int32_t{0})) {
      //
      // tmemSfB [ProdPreCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if ((loopEnd) >= (int32_t{512})) {
      }
      //
      // tmemSfB [ProdCommit, Info{1}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if ((loopEnd) >= (int32_t{512})) {
        {
          tmemSfBDstStack.mPipeline.producer_commit(tmemSfBProdState);
        }
        ++tmemSfBProdState;
      }
    }
    //
    // Tail work.
    //
    //
    // smemSfB [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    {}
    //
    // tmemSfB [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
    return ((state.mWarpIdx) >= (int32_t{23})) && ((state.mWarpIdx) < (int32_t{24}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 Mma0Stack& mma0DstStack,
                                 SmemASmem& smemASrcSmem,
                                 SmemAStack& smemASrcStack,
                                 SmemBSmem& smemBSrcSmem,
                                 SmemBStack& smemBSrcStack,
                                 TmemSfAStack& tmemSfASrcStack,
                                 TmemSfBStack& tmemSfBSrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<48>{});
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
    trtllm::dev::CutlassUmmaConsumerAsyncPipeline<
      5,
      false,
      false,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState tmemSfAConsState{};
    trtllm::dev::CutlassUmmaConsumerAsyncPipeline<
      5,
      false,
      false,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState
      tmemSfAConsReleaseState{};
    int32_t tmemSfAConsToken{int32_t{0}};
    trtllm::dev::CutlassUmmaConsumerAsyncPipeline<
      5,
      false,
      false,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState tmemSfBConsState{};
    trtllm::dev::CutlassUmmaConsumerAsyncPipeline<
      5,
      false,
      false,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState
      tmemSfBConsReleaseState{};
    int32_t tmemSfBConsToken{int32_t{0}};
    trtllm::dev::CutlassUmmaAsyncPipeline<1,
                                          cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::
      PipelineState mma0ProdState{int32_t{0}, int32_t{1}, int32_t{0}};
    int32_t mma0ProdToken{int32_t{1}};
    int32_t paddedPerCtaK{(((params.k) + (int32_t{511})) / (int32_t{512})) * (int32_t{512})};
    int32_t loopEnd{paddedPerCtaK};
    //
    // Loop body.
    //
    CUTLASS_PRAGMA_NO_UNROLL
    for (int32_t loopOffset1129 = int32_t{0}; loopOffset1129 < loopEnd;
         loopOffset1129 += int32_t{1024}) {
      //
      // Unrolled iter 0.
      //
      {
        bool const isFirstLoopIter{(loopOffset1129) == (int32_t{0})};
        bool const isLastLoopIter{((loopOffset1129) + (int32_t{512})) >= (loopEnd)};
        //
        // mma0 [ProdTryAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if (isFirstLoopIter) {
          if ((loopOffset1129) >= (int32_t{0})) {
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
        // tmemSfA [ConsTryWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        { tmemSfAConsToken = tmemSfASrcStack.mPipeline.consumer_try_wait(tmemSfAConsState); }
        //
        // tmemSfB [ConsTryWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        { tmemSfBConsToken = tmemSfBSrcStack.mPipeline.consumer_try_wait(tmemSfBConsState); }
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
        // tmemSfA [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{8388608}].
        //
        { tmemSfASrcStack.mPipeline.consumer_wait(tmemSfAConsState, tmemSfAConsToken); }
        //
        // tmemSfB [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{8388608}].
        //
        { tmemSfBSrcStack.mPipeline.consumer_wait(tmemSfBConsState, tmemSfBConsToken); }
        //
        // smemA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_e2m1_t* smemPtrA5;
        {
          int32_t index{smemAConsState.index()};
          int8_t* smemBytesBasePtrA;
          smemBytesBasePtrA = reinterpret_cast<int8_t*>(smemASrcStack.mDepSmemPtr0) + (int32_t{0});
          int8_t* smemBytesStagePtrA;
          smemBytesStagePtrA = smemBytesBasePtrA + ((index) * (int32_t{32768}));
          smemPtrA5 = reinterpret_cast<cutlass::float_e2m1_t*>(smemBytesStagePtrA) + (int32_t{0});
          ++smemAConsState;
        }
        //
        // smemB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_e2m1_t* smemPtrB6;
        {
          int32_t index{smemBConsState.index()};
          int8_t* smemBytesBasePtrB;
          smemBytesBasePtrB =
            reinterpret_cast<int8_t*>(smemBSrcStack.mDepSmemPtr0) + (int32_t{163840});
          int8_t* smemBytesStagePtrB;
          smemBytesStagePtrB = smemBytesBasePtrB + ((index) * (int32_t{4096}));
          smemPtrB6 = reinterpret_cast<cutlass::float_e2m1_t*>(smemBytesStagePtrB) + (int32_t{0});
          ++smemBConsState;
        }
        //
        // tmemSfA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        uint32_t tmemAddrSfA9;
        {
          int32_t index{tmemSfAConsState.index()};
          tmemAddrSfA9 =
            ((mTmemBaseOffset) + (uint32_t{32})) + (static_cast<uint32_t>((index) * (int32_t{32})));
          ++tmemSfAConsState;
        }
        //
        // tmemSfB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        uint32_t tmemAddrSfB10;
        {
          int32_t index{tmemSfBConsState.index()};
          tmemAddrSfB10 = ((mTmemBaseOffset) + (uint32_t{192})) +
                          (static_cast<uint32_t>((index) * (int32_t{16})));
          ++tmemSfBConsState;
        }
        //
        // mma0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_e2m1_t* smemPtrA12;
        cutlass::float_e2m1_t* smemPtrB12;
        uint32_t tmemAddrSfA12;
        uint32_t tmemAddrSfB12;
        smemPtrA12 = smemPtrA5;
        smemPtrB12 = smemPtrB6;
        tmemAddrSfA12 = tmemAddrSfA9;
        tmemAddrSfB12 = tmemAddrSfB10;
        {
          int32_t index{mma0ProdState.index()};
          uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{32})))};
          uint32_t ptrTmemOffsetD{ptrTmemD};
          cutlass::float_e2m1_t* ptrWithOffsetSmemA{(smemPtrA12 + int32_t{0})};
          cutlass::float_e2m1_t* ptrWithOffsetSmemB{(smemPtrB12 + int32_t{0})};
          {
            uint32_t tmemPtrD{ptrTmemOffsetD};
            uint32_t tmemPtrSfA{tmemAddrSfA12};
            uint32_t tmemPtrSfB{tmemAddrSfB12};
            //
            // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
            //
            uint64_t smemDescA{
              trtllm::dev::createSmemDesc(ptrWithOffsetSmemA,
                                          uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                          uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
            //
            // leadingDimInBytes = 2048, strideInBytes = 1024, swizzleMode = 1.
            //
            uint64_t smemDescB{
              trtllm::dev::createSmemDesc(ptrWithOffsetSmemB,
                                          uint32_t{0x800000 /*hi=128, lo=0*/},
                                          uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
            //
            // MMA inst for mi=0 ni=0 ki=0.
            //
            uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc_block(int32_t{0},
                                                                          int32_t{1},
                                                                          int32_t{1},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{32},
                                                                          int32_t{64},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block16(cuda_ptx::kind_mxf4nvf4,
                                                        cuda_ptx::cta_group_2,
                                                        tmemPtrD,
                                                        smemDescA,
                                                        smemDescB,
                                                        utcmmaDesc_0_0_0,
                                                        tmemPtrSfA,
                                                        tmemPtrSfB,
                                                        bool{(loopOffset1129) != (int32_t{0})});
            }
            //
            // MMA inst for mi=0 ni=0 ki=1.
            //
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            tmemPtrSfA += uint32_t{0x4 /*hi=0, lo=4*/};
            tmemPtrSfB += uint32_t{0x2 /*hi=0, lo=2*/};
            uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc_block(int32_t{0},
                                                                          int32_t{1},
                                                                          int32_t{1},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{32},
                                                                          int32_t{64},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block16(cuda_ptx::kind_mxf4nvf4,
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
            tmemPtrSfA += uint32_t{0x4 /*hi=0, lo=4*/};
            tmemPtrSfB += uint32_t{0x2 /*hi=0, lo=2*/};
            uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc_block(int32_t{0},
                                                                          int32_t{1},
                                                                          int32_t{1},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{32},
                                                                          int32_t{64},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block16(cuda_ptx::kind_mxf4nvf4,
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
            tmemPtrSfA += uint32_t{0x4 /*hi=0, lo=4*/};
            tmemPtrSfB += uint32_t{0x2 /*hi=0, lo=2*/};
            uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc_block(int32_t{0},
                                                                          int32_t{1},
                                                                          int32_t{1},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{32},
                                                                          int32_t{64},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block16(cuda_ptx::kind_mxf4nvf4,
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
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{122});
            tmemPtrSfA += uint32_t{0x4 /*hi=0, lo=4*/};
            tmemPtrSfB += uint32_t{0x2 /*hi=0, lo=2*/};
            uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc_block(int32_t{0},
                                                                          int32_t{1},
                                                                          int32_t{1},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{32},
                                                                          int32_t{64},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block16(cuda_ptx::kind_mxf4nvf4,
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
            tmemPtrSfA += uint32_t{0x4 /*hi=0, lo=4*/};
            tmemPtrSfB += uint32_t{0x2 /*hi=0, lo=2*/};
            uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc_block(int32_t{0},
                                                                          int32_t{1},
                                                                          int32_t{1},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{32},
                                                                          int32_t{64},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block16(cuda_ptx::kind_mxf4nvf4,
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
            tmemPtrSfA += uint32_t{0x4 /*hi=0, lo=4*/};
            tmemPtrSfB += uint32_t{0x2 /*hi=0, lo=2*/};
            uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc_block(int32_t{0},
                                                                          int32_t{1},
                                                                          int32_t{1},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{32},
                                                                          int32_t{64},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block16(cuda_ptx::kind_mxf4nvf4,
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
            tmemPtrSfA += uint32_t{0x4 /*hi=0, lo=4*/};
            tmemPtrSfB += uint32_t{0x2 /*hi=0, lo=2*/};
            uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc_block(int32_t{0},
                                                                          int32_t{1},
                                                                          int32_t{1},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{32},
                                                                          int32_t{64},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block16(cuda_ptx::kind_mxf4nvf4,
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
        if ((loopOffset1129) >= (int32_t{0})) {
          {
            smemASrcStack.mPipeline.consumer_release(smemAConsReleaseState);
          }
          ++smemAConsReleaseState;
        }
        //
        // smemB [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset1129) >= (int32_t{0})) {
          {
            smemBSrcStack.mPipeline.consumer_release(smemBConsReleaseState);
          }
          ++smemBConsReleaseState;
        }
        //
        // tmemSfA [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset1129) >= (int32_t{0})) {
          {
            tmemSfASrcStack.mPipeline.consumer_release(tmemSfAConsReleaseState);
          }
          ++tmemSfAConsReleaseState;
        }
        //
        // tmemSfB [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if ((loopOffset1129) >= (int32_t{0})) {
          {
            tmemSfBSrcStack.mPipeline.consumer_release(tmemSfBConsReleaseState);
          }
          ++tmemSfBConsReleaseState;
        }
        //
        // mma0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if (isLastLoopIter) {
          {
            mma0DstStack.mPipeline.producer_commit(mma0ProdState);
          }
          ++mma0ProdState;
        }
      }
      //
      // Unrolled iter 1.
      //
      {
        bool const isFirstLoopIter{((int32_t{512}) + (loopOffset1129)) == (int32_t{0})};
        bool const isLastLoopIter{(((int32_t{512}) + (loopOffset1129)) + (int32_t{512})) >=
                                  (loopEnd)};
        //
        // mma0 [ProdTryAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if (isFirstLoopIter) {
          if (((int32_t{512}) + (loopOffset1129)) >= (int32_t{0})) {
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
        // tmemSfA [ConsTryWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        { tmemSfAConsToken = tmemSfASrcStack.mPipeline.consumer_try_wait(tmemSfAConsState); }
        //
        // tmemSfB [ConsTryWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        { tmemSfBConsToken = tmemSfBSrcStack.mPipeline.consumer_try_wait(tmemSfBConsState); }
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
        // tmemSfA [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{8388608}].
        //
        { tmemSfASrcStack.mPipeline.consumer_wait(tmemSfAConsState, tmemSfAConsToken); }
        //
        // tmemSfB [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{8388608}].
        //
        { tmemSfBSrcStack.mPipeline.consumer_wait(tmemSfBConsState, tmemSfBConsToken); }
        //
        // smemA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_e2m1_t* smemPtrA5;
        {
          int32_t index{smemAConsState.index()};
          int8_t* smemBytesBasePtrA;
          smemBytesBasePtrA = reinterpret_cast<int8_t*>(smemASrcStack.mDepSmemPtr0) + (int32_t{0});
          int8_t* smemBytesStagePtrA;
          smemBytesStagePtrA = smemBytesBasePtrA + ((index) * (int32_t{32768}));
          smemPtrA5 = reinterpret_cast<cutlass::float_e2m1_t*>(smemBytesStagePtrA) + (int32_t{0});
          ++smemAConsState;
        }
        //
        // smemB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_e2m1_t* smemPtrB6;
        {
          int32_t index{smemBConsState.index()};
          int8_t* smemBytesBasePtrB;
          smemBytesBasePtrB =
            reinterpret_cast<int8_t*>(smemBSrcStack.mDepSmemPtr0) + (int32_t{163840});
          int8_t* smemBytesStagePtrB;
          smemBytesStagePtrB = smemBytesBasePtrB + ((index) * (int32_t{4096}));
          smemPtrB6 = reinterpret_cast<cutlass::float_e2m1_t*>(smemBytesStagePtrB) + (int32_t{0});
          ++smemBConsState;
        }
        //
        // tmemSfA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        uint32_t tmemAddrSfA9;
        {
          int32_t index{tmemSfAConsState.index()};
          tmemAddrSfA9 =
            ((mTmemBaseOffset) + (uint32_t{32})) + (static_cast<uint32_t>((index) * (int32_t{32})));
          ++tmemSfAConsState;
        }
        //
        // tmemSfB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        uint32_t tmemAddrSfB10;
        {
          int32_t index{tmemSfBConsState.index()};
          tmemAddrSfB10 = ((mTmemBaseOffset) + (uint32_t{192})) +
                          (static_cast<uint32_t>((index) * (int32_t{16})));
          ++tmemSfBConsState;
        }
        //
        // mma0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        cutlass::float_e2m1_t* smemPtrA12;
        cutlass::float_e2m1_t* smemPtrB12;
        uint32_t tmemAddrSfA12;
        uint32_t tmemAddrSfB12;
        smemPtrA12 = smemPtrA5;
        smemPtrB12 = smemPtrB6;
        tmemAddrSfA12 = tmemAddrSfA9;
        tmemAddrSfB12 = tmemAddrSfB10;
        {
          int32_t index{mma0ProdState.index()};
          uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{32})))};
          uint32_t ptrTmemOffsetD{ptrTmemD};
          cutlass::float_e2m1_t* ptrWithOffsetSmemA{(smemPtrA12 + int32_t{0})};
          cutlass::float_e2m1_t* ptrWithOffsetSmemB{(smemPtrB12 + int32_t{0})};
          {
            uint32_t tmemPtrD{ptrTmemOffsetD};
            uint32_t tmemPtrSfA{tmemAddrSfA12};
            uint32_t tmemPtrSfB{tmemAddrSfB12};
            //
            // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
            //
            uint64_t smemDescA{
              trtllm::dev::createSmemDesc(ptrWithOffsetSmemA,
                                          uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                          uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
            //
            // leadingDimInBytes = 2048, strideInBytes = 1024, swizzleMode = 1.
            //
            uint64_t smemDescB{
              trtllm::dev::createSmemDesc(ptrWithOffsetSmemB,
                                          uint32_t{0x800000 /*hi=128, lo=0*/},
                                          uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
            //
            // MMA inst for mi=0 ni=0 ki=0.
            //
            uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc_block(int32_t{0},
                                                                          int32_t{1},
                                                                          int32_t{1},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{32},
                                                                          int32_t{64},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block16(
                cuda_ptx::kind_mxf4nvf4,
                cuda_ptx::cta_group_2,
                tmemPtrD,
                smemDescA,
                smemDescB,
                utcmmaDesc_0_0_0,
                tmemPtrSfA,
                tmemPtrSfB,
                bool{((int32_t{512}) + (loopOffset1129)) != (int32_t{0})});
            }
            //
            // MMA inst for mi=0 ni=0 ki=1.
            //
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            tmemPtrSfA += uint32_t{0x4 /*hi=0, lo=4*/};
            tmemPtrSfB += uint32_t{0x2 /*hi=0, lo=2*/};
            uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc_block(int32_t{0},
                                                                          int32_t{1},
                                                                          int32_t{1},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{32},
                                                                          int32_t{64},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block16(cuda_ptx::kind_mxf4nvf4,
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
            tmemPtrSfA += uint32_t{0x4 /*hi=0, lo=4*/};
            tmemPtrSfB += uint32_t{0x2 /*hi=0, lo=2*/};
            uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc_block(int32_t{0},
                                                                          int32_t{1},
                                                                          int32_t{1},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{32},
                                                                          int32_t{64},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block16(cuda_ptx::kind_mxf4nvf4,
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
            tmemPtrSfA += uint32_t{0x4 /*hi=0, lo=4*/};
            tmemPtrSfB += uint32_t{0x2 /*hi=0, lo=2*/};
            uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc_block(int32_t{0},
                                                                          int32_t{1},
                                                                          int32_t{1},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{32},
                                                                          int32_t{64},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block16(cuda_ptx::kind_mxf4nvf4,
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
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{122});
            tmemPtrSfA += uint32_t{0x4 /*hi=0, lo=4*/};
            tmemPtrSfB += uint32_t{0x2 /*hi=0, lo=2*/};
            uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc_block(int32_t{0},
                                                                          int32_t{1},
                                                                          int32_t{1},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{32},
                                                                          int32_t{64},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block16(cuda_ptx::kind_mxf4nvf4,
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
            tmemPtrSfA += uint32_t{0x4 /*hi=0, lo=4*/};
            tmemPtrSfB += uint32_t{0x2 /*hi=0, lo=2*/};
            uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc_block(int32_t{0},
                                                                          int32_t{1},
                                                                          int32_t{1},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{32},
                                                                          int32_t{64},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block16(cuda_ptx::kind_mxf4nvf4,
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
            tmemPtrSfA += uint32_t{0x4 /*hi=0, lo=4*/};
            tmemPtrSfB += uint32_t{0x2 /*hi=0, lo=2*/};
            uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc_block(int32_t{0},
                                                                          int32_t{1},
                                                                          int32_t{1},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{32},
                                                                          int32_t{64},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block16(cuda_ptx::kind_mxf4nvf4,
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
            tmemPtrSfA += uint32_t{0x4 /*hi=0, lo=4*/};
            tmemPtrSfB += uint32_t{0x2 /*hi=0, lo=2*/};
            uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc_block(int32_t{0},
                                                                          int32_t{1},
                                                                          int32_t{1},
                                                                          false,
                                                                          false,
                                                                          int32_t{256},
                                                                          int32_t{32},
                                                                          int32_t{64},
                                                                          false,
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          int32_t{0},
                                                                          true)};
            if (bool{cute::elect_one_sync()}) {
              cuda_ptx::tcgen05_mma_block_scale_block16(cuda_ptx::kind_mxf4nvf4,
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
        if (((int32_t{512}) + (loopOffset1129)) >= (int32_t{0})) {
          {
            smemASrcStack.mPipeline.consumer_release(smemAConsReleaseState);
          }
          ++smemAConsReleaseState;
        }
        //
        // smemB [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if (((int32_t{512}) + (loopOffset1129)) >= (int32_t{0})) {
          {
            smemBSrcStack.mPipeline.consumer_release(smemBConsReleaseState);
          }
          ++smemBConsReleaseState;
        }
        //
        // tmemSfA [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if (((int32_t{512}) + (loopOffset1129)) >= (int32_t{0})) {
          {
            tmemSfASrcStack.mPipeline.consumer_release(tmemSfAConsReleaseState);
          }
          ++tmemSfAConsReleaseState;
        }
        //
        // tmemSfB [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if (((int32_t{512}) + (loopOffset1129)) >= (int32_t{0})) {
          {
            tmemSfBSrcStack.mPipeline.consumer_release(tmemSfBConsReleaseState);
          }
          ++tmemSfBConsReleaseState;
        }
        //
        // mma0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        if (isLastLoopIter) {
          {
            mma0DstStack.mPipeline.producer_commit(mma0ProdState);
          }
          ++mma0ProdState;
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
  float mScaleGate;
  float mClampLimit;
  float mGatedActAlpha;
  float mGatedActBeta;
  int32_t mCtaOffsetK;
  uint32_t const mTmemBaseOffset;
  int32_t mCtaIdxX;
  int32_t const mWarpGrpThreadIdx;
  cutlass::Array<float, 32> frg13;
  int32_t const mLdtm16dp256bitTmemColIdx;
  int32_t const mLdtm16dp256bitTmemRowIdx;
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
    , mScaleGate{float{
        (params.ptrScaleGate + int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]})[int32_t{0}]}}
    , mClampLimit{float(bool{params.ptrClampLimit != nullptr})
                    ? (float{params.ptrClampLimit[int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}]})
                    : (float{3.4028235e+38})}
    , mGatedActAlpha{float(bool{params.ptrGatedActAlpha != nullptr})
                       ? (float{params.ptrGatedActAlpha[int32_t{
                           params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}]})
                       : (float{1})}
    , mGatedActBeta{float(bool{params.ptrGatedActBeta != nullptr})
                      ? (float{
                          params.ptrGatedActBeta[int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}]})
                      : (float{0})}
    , mCtaOffsetK{int32_t{0}}
    , mTmemBaseOffset{uint32_t{
        __shfl_sync(uint32_t{0xffffffff}, (*state.mTmemSwStatePtr), int32_t{0}, int32_t{32})}}
    , mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
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
                                 ClusterBarrierBuffersStack& clusterBarrierBuffersDstStack,
                                 Mma0Stack& mma0SrcStack) {
    cuda_ptx::setmaxnreg_inc(cuda_ptx::n32_t<160>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassUmmaAsyncPipeline<
      1,
      cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::PipelineState mma0ConsState{};
    int32_t mma0ConsToken{int32_t{0}};
    int32_t paddedPerCtaK{(((params.k) + (int32_t{511})) / (int32_t{512})) * (int32_t{512})};
    int32_t loopEnd{paddedPerCtaK};
    bool const hasOneLoopIter{(int32_t{0}) < (loopEnd)};
    int32_t lastLoopOffset{int32_t{0}};
    uint32_t tmemBaseWithStageOffset;
    tmemBaseWithStageOffset = mTmemBaseOffset;
    //
    // SmemBias::createFctProdVars.
    //
    int8_t* ptrSmemBaseBias;
    float* ptrSmemBias;
    ptrSmemBaseBias = reinterpret_cast<int8_t*>(gmemC0DstStack.mDepSmemPtr0) + (int32_t{186368});
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
            ptrSmemBias[mWarpGrpThreadIdx] = float{
              reinterpret_cast<float const*>(params.ptrBias)[(offsetTileM) + (mWarpGrpThreadIdx)]};
          }
        }
        trtllm::dev::CutlassNamedBarrier::sync(128, 9);
      }
    }
    mClampLimit = float(bool{params.ptrClampLimit != nullptr})
                    ? (float{params.ptrClampLimit[int32_t{params.ptrCtaIdxXyToBatchIdx[mCtaIdxY]}]})
                    : (float{3.4028235e+38});
    //
    // Hoist the first iter.
    //
    //
    // Loop body.
    //
    CUTLASS_PRAGMA_NO_UNROLL
    for (int32_t loopOffset1607 = int32_t{0}; loopOffset1607 < loopEnd;
         loopOffset1607 += int32_t{512}) {
      //
      // gmemC0 [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      //
      // mma0 [ConsTailRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      lastLoopOffset = loopOffset1607;
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
      uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{32})))};
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
          uint32_t(&dstSlice0)[16]{reinterpret_cast<uint32_t(&)[16]>(frg13[int32_t{0}])};
          cuda_ptx::tcgen05_ld_16x256b(dstSlice0,
                                       (tmemBasePtr) +
                                         (static_cast<uint32_t>((mWarpGrp4Idx) * (int32_t{32}))));
          uint32_t(&dstSlice1)[16]{reinterpret_cast<uint32_t(&)[16]>(frg13[int32_t{16}])};
          cuda_ptx::tcgen05_ld_16x256b(
            dstSlice1,
            (tmemBasePtr) + (static_cast<uint32_t>(((mWarpGrp4Idx) * (int32_t{32})) +
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
            int32_t const sharedColIdx{(laneColIdx) + ((mWarpGrp4Idx) * (int32_t{32}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{0}] = (frg13[int32_t{0}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (0, 1).
          //
          {
            int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{1}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{1}] = (frg13[int32_t{1}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (0, 2).
          //
          {
            int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{8}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{4}] = (frg13[int32_t{4}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (0, 3).
          //
          {
            int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{9}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{5}] = (frg13[int32_t{5}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (0, 4).
          //
          {
            int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{16}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{8}] = (frg13[int32_t{8}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (0, 5).
          //
          {
            int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{17}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{9}] = (frg13[int32_t{9}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (0, 6).
          //
          {
            int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{24}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{12}] = (frg13[int32_t{12}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (0, 7).
          //
          {
            int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{25}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{13}] = (frg13[int32_t{13}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (1, 0).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
            int32_t const sharedColIdx{(laneColIdx) + ((mWarpGrp4Idx) * (int32_t{32}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{2}] = (frg13[int32_t{2}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (1, 1).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{1}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{3}] = (frg13[int32_t{3}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (1, 2).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{8}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{6}] = (frg13[int32_t{6}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (1, 3).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{9}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{7}] = (frg13[int32_t{7}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (1, 4).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{16}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{10}] = (frg13[int32_t{10}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (1, 5).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{17}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{11}] = (frg13[int32_t{11}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (1, 6).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{24}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{14}] = (frg13[int32_t{14}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (1, 7).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{25}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{15}] = (frg13[int32_t{15}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (2, 0).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
            int32_t const sharedColIdx{(laneColIdx) + ((mWarpGrp4Idx) * (int32_t{32}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{16}] = (frg13[int32_t{16}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (2, 1).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{1}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{17}] = (frg13[int32_t{17}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (2, 2).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{8}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{20}] = (frg13[int32_t{20}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (2, 3).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{9}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{21}] = (frg13[int32_t{21}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (2, 4).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{16}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{24}] = (frg13[int32_t{24}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (2, 5).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{17}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{25}] = (frg13[int32_t{25}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (2, 6).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{24}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{28}] = (frg13[int32_t{28}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (2, 7).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{25}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{29}] = (frg13[int32_t{29}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (3, 0).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
            int32_t const sharedColIdx{(laneColIdx) + ((mWarpGrp4Idx) * (int32_t{32}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{18}] = (frg13[int32_t{18}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (3, 1).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{1}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{19}] = (frg13[int32_t{19}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (3, 2).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{8}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{22}] = (frg13[int32_t{22}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (3, 3).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{9}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{23}] = (frg13[int32_t{23}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (3, 4).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{16}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{26}] = (frg13[int32_t{26}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (3, 5).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{17}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{27}] = (frg13[int32_t{27}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (3, 6).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{24}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{30}] = (frg13[int32_t{30}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
          //
          // Add bias (3, 7).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{32}) + (int32_t{25}))};
            //
            // Loading bias to register.
            //
            frg13[int32_t{31}] = (frg13[int32_t{31}]) + (float{ptrSmemBias[sharedRowIdx]});
          }
        }
        //
        // Applying gated activation.
        //
        {
          frg13[int32_t{0}] =
            float{trtllm::dev::clamp(frg13[int32_t{0}], -(mClampLimit), mClampLimit)};
          frg13[int32_t{2}] =
            float{trtllm::dev::clamp(frg13[int32_t{2}], float{-3.4028235e+38}, mClampLimit)};
          frg13[int32_t{1}] =
            float{trtllm::dev::clamp(frg13[int32_t{1}], -(mClampLimit), mClampLimit)};
          frg13[int32_t{3}] =
            float{trtllm::dev::clamp(frg13[int32_t{3}], float{-3.4028235e+38}, mClampLimit)};
          frg13[int32_t{0}] = (frg13[int32_t{0}]) * (float{1});
          frg13[int32_t{1}] = (frg13[int32_t{1}]) * (float{1});
          cutlass::Array<float, 2> x0Array{frg13[int32_t{0}], frg13[int32_t{1}]};
          cutlass::Array<float, 2> x1Array{frg13[int32_t{2}], frg13[int32_t{3}]};
          cutlass::Array<float, 2> fusedScaleArray{(mScaleGate) * (mGatedActAlpha),
                                                   (mScaleGate) * (mGatedActAlpha)};
          cutlass::Array<float, 2> betaScaleGateArray{(mGatedActBeta) * (mScaleGate),
                                                      (mGatedActBeta) * (mScaleGate)};
          cutlass::Array<float, 2> scaleGateArray{mScaleGate, mScaleGate};
          cutlass::Array<float, 2> x0ScaleGateArray;
          x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
          cutlass::Array<float, 2> x1ScaledArray;
          x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
          float phi3612{trtllm::dev::phi(x1ScaledArray[int32_t{0}])};
          float phi3614{trtllm::dev::phi(x1ScaledArray[int32_t{1}])};
          cutlass::Array<float, 2> actArray{phi3612, phi3614};
          cutlass::Array<float, 2> x1PackedArray{x1Array[int32_t{0}], x1Array[int32_t{1}]};
          cutlass::Array<float, 2> geluArray;
          geluArray = trtllm::dev::fmul2(x1PackedArray, actArray);
          cutlass::Array<float, 2> gatedActArray;
          gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, geluArray);
          frg13[int32_t{0}] = gatedActArray[int32_t{0}];
          frg13[int32_t{1}] = gatedActArray[int32_t{1}];
        }
        {
          frg13[int32_t{16}] =
            float{trtllm::dev::clamp(frg13[int32_t{16}], -(mClampLimit), mClampLimit)};
          frg13[int32_t{18}] =
            float{trtllm::dev::clamp(frg13[int32_t{18}], float{-3.4028235e+38}, mClampLimit)};
          frg13[int32_t{17}] =
            float{trtllm::dev::clamp(frg13[int32_t{17}], -(mClampLimit), mClampLimit)};
          frg13[int32_t{19}] =
            float{trtllm::dev::clamp(frg13[int32_t{19}], float{-3.4028235e+38}, mClampLimit)};
          frg13[int32_t{16}] = (frg13[int32_t{16}]) * (float{1});
          frg13[int32_t{17}] = (frg13[int32_t{17}]) * (float{1});
          cutlass::Array<float, 2> x0Array{frg13[int32_t{16}], frg13[int32_t{17}]};
          cutlass::Array<float, 2> x1Array{frg13[int32_t{18}], frg13[int32_t{19}]};
          cutlass::Array<float, 2> fusedScaleArray{(mScaleGate) * (mGatedActAlpha),
                                                   (mScaleGate) * (mGatedActAlpha)};
          cutlass::Array<float, 2> betaScaleGateArray{(mGatedActBeta) * (mScaleGate),
                                                      (mGatedActBeta) * (mScaleGate)};
          cutlass::Array<float, 2> scaleGateArray{mScaleGate, mScaleGate};
          cutlass::Array<float, 2> x0ScaleGateArray;
          x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
          cutlass::Array<float, 2> x1ScaledArray;
          x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
          float phi3689{trtllm::dev::phi(x1ScaledArray[int32_t{0}])};
          float phi3691{trtllm::dev::phi(x1ScaledArray[int32_t{1}])};
          cutlass::Array<float, 2> actArray{phi3689, phi3691};
          cutlass::Array<float, 2> x1PackedArray{x1Array[int32_t{0}], x1Array[int32_t{1}]};
          cutlass::Array<float, 2> geluArray;
          geluArray = trtllm::dev::fmul2(x1PackedArray, actArray);
          cutlass::Array<float, 2> gatedActArray;
          gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, geluArray);
          frg13[int32_t{16}] = gatedActArray[int32_t{0}];
          frg13[int32_t{17}] = gatedActArray[int32_t{1}];
        }
        {
          frg13[int32_t{4}] =
            float{trtllm::dev::clamp(frg13[int32_t{4}], -(mClampLimit), mClampLimit)};
          frg13[int32_t{6}] =
            float{trtllm::dev::clamp(frg13[int32_t{6}], float{-3.4028235e+38}, mClampLimit)};
          frg13[int32_t{5}] =
            float{trtllm::dev::clamp(frg13[int32_t{5}], -(mClampLimit), mClampLimit)};
          frg13[int32_t{7}] =
            float{trtllm::dev::clamp(frg13[int32_t{7}], float{-3.4028235e+38}, mClampLimit)};
          frg13[int32_t{4}] = (frg13[int32_t{4}]) * (float{1});
          frg13[int32_t{5}] = (frg13[int32_t{5}]) * (float{1});
          cutlass::Array<float, 2> x0Array{frg13[int32_t{4}], frg13[int32_t{5}]};
          cutlass::Array<float, 2> x1Array{frg13[int32_t{6}], frg13[int32_t{7}]};
          cutlass::Array<float, 2> fusedScaleArray{(mScaleGate) * (mGatedActAlpha),
                                                   (mScaleGate) * (mGatedActAlpha)};
          cutlass::Array<float, 2> betaScaleGateArray{(mGatedActBeta) * (mScaleGate),
                                                      (mGatedActBeta) * (mScaleGate)};
          cutlass::Array<float, 2> scaleGateArray{mScaleGate, mScaleGate};
          cutlass::Array<float, 2> x0ScaleGateArray;
          x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
          cutlass::Array<float, 2> x1ScaledArray;
          x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
          float phi3775{trtllm::dev::phi(x1ScaledArray[int32_t{0}])};
          float phi3777{trtllm::dev::phi(x1ScaledArray[int32_t{1}])};
          cutlass::Array<float, 2> actArray{phi3775, phi3777};
          cutlass::Array<float, 2> x1PackedArray{x1Array[int32_t{0}], x1Array[int32_t{1}]};
          cutlass::Array<float, 2> geluArray;
          geluArray = trtllm::dev::fmul2(x1PackedArray, actArray);
          cutlass::Array<float, 2> gatedActArray;
          gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, geluArray);
          frg13[int32_t{4}] = gatedActArray[int32_t{0}];
          frg13[int32_t{5}] = gatedActArray[int32_t{1}];
        }
        {
          frg13[int32_t{20}] =
            float{trtllm::dev::clamp(frg13[int32_t{20}], -(mClampLimit), mClampLimit)};
          frg13[int32_t{22}] =
            float{trtllm::dev::clamp(frg13[int32_t{22}], float{-3.4028235e+38}, mClampLimit)};
          frg13[int32_t{21}] =
            float{trtllm::dev::clamp(frg13[int32_t{21}], -(mClampLimit), mClampLimit)};
          frg13[int32_t{23}] =
            float{trtllm::dev::clamp(frg13[int32_t{23}], float{-3.4028235e+38}, mClampLimit)};
          frg13[int32_t{20}] = (frg13[int32_t{20}]) * (float{1});
          frg13[int32_t{21}] = (frg13[int32_t{21}]) * (float{1});
          cutlass::Array<float, 2> x0Array{frg13[int32_t{20}], frg13[int32_t{21}]};
          cutlass::Array<float, 2> x1Array{frg13[int32_t{22}], frg13[int32_t{23}]};
          cutlass::Array<float, 2> fusedScaleArray{(mScaleGate) * (mGatedActAlpha),
                                                   (mScaleGate) * (mGatedActAlpha)};
          cutlass::Array<float, 2> betaScaleGateArray{(mGatedActBeta) * (mScaleGate),
                                                      (mGatedActBeta) * (mScaleGate)};
          cutlass::Array<float, 2> scaleGateArray{mScaleGate, mScaleGate};
          cutlass::Array<float, 2> x0ScaleGateArray;
          x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
          cutlass::Array<float, 2> x1ScaledArray;
          x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
          float phi3852{trtllm::dev::phi(x1ScaledArray[int32_t{0}])};
          float phi3854{trtllm::dev::phi(x1ScaledArray[int32_t{1}])};
          cutlass::Array<float, 2> actArray{phi3852, phi3854};
          cutlass::Array<float, 2> x1PackedArray{x1Array[int32_t{0}], x1Array[int32_t{1}]};
          cutlass::Array<float, 2> geluArray;
          geluArray = trtllm::dev::fmul2(x1PackedArray, actArray);
          cutlass::Array<float, 2> gatedActArray;
          gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, geluArray);
          frg13[int32_t{20}] = gatedActArray[int32_t{0}];
          frg13[int32_t{21}] = gatedActArray[int32_t{1}];
        }
        {
          frg13[int32_t{8}] =
            float{trtllm::dev::clamp(frg13[int32_t{8}], -(mClampLimit), mClampLimit)};
          frg13[int32_t{10}] =
            float{trtllm::dev::clamp(frg13[int32_t{10}], float{-3.4028235e+38}, mClampLimit)};
          frg13[int32_t{9}] =
            float{trtllm::dev::clamp(frg13[int32_t{9}], -(mClampLimit), mClampLimit)};
          frg13[int32_t{11}] =
            float{trtllm::dev::clamp(frg13[int32_t{11}], float{-3.4028235e+38}, mClampLimit)};
          frg13[int32_t{8}] = (frg13[int32_t{8}]) * (float{1});
          frg13[int32_t{9}] = (frg13[int32_t{9}]) * (float{1});
          cutlass::Array<float, 2> x0Array{frg13[int32_t{8}], frg13[int32_t{9}]};
          cutlass::Array<float, 2> x1Array{frg13[int32_t{10}], frg13[int32_t{11}]};
          cutlass::Array<float, 2> fusedScaleArray{(mScaleGate) * (mGatedActAlpha),
                                                   (mScaleGate) * (mGatedActAlpha)};
          cutlass::Array<float, 2> betaScaleGateArray{(mGatedActBeta) * (mScaleGate),
                                                      (mGatedActBeta) * (mScaleGate)};
          cutlass::Array<float, 2> scaleGateArray{mScaleGate, mScaleGate};
          cutlass::Array<float, 2> x0ScaleGateArray;
          x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
          cutlass::Array<float, 2> x1ScaledArray;
          x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
          float phi3938{trtllm::dev::phi(x1ScaledArray[int32_t{0}])};
          float phi3940{trtllm::dev::phi(x1ScaledArray[int32_t{1}])};
          cutlass::Array<float, 2> actArray{phi3938, phi3940};
          cutlass::Array<float, 2> x1PackedArray{x1Array[int32_t{0}], x1Array[int32_t{1}]};
          cutlass::Array<float, 2> geluArray;
          geluArray = trtllm::dev::fmul2(x1PackedArray, actArray);
          cutlass::Array<float, 2> gatedActArray;
          gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, geluArray);
          frg13[int32_t{8}] = gatedActArray[int32_t{0}];
          frg13[int32_t{9}] = gatedActArray[int32_t{1}];
        }
        {
          frg13[int32_t{24}] =
            float{trtllm::dev::clamp(frg13[int32_t{24}], -(mClampLimit), mClampLimit)};
          frg13[int32_t{26}] =
            float{trtllm::dev::clamp(frg13[int32_t{26}], float{-3.4028235e+38}, mClampLimit)};
          frg13[int32_t{25}] =
            float{trtllm::dev::clamp(frg13[int32_t{25}], -(mClampLimit), mClampLimit)};
          frg13[int32_t{27}] =
            float{trtllm::dev::clamp(frg13[int32_t{27}], float{-3.4028235e+38}, mClampLimit)};
          frg13[int32_t{24}] = (frg13[int32_t{24}]) * (float{1});
          frg13[int32_t{25}] = (frg13[int32_t{25}]) * (float{1});
          cutlass::Array<float, 2> x0Array{frg13[int32_t{24}], frg13[int32_t{25}]};
          cutlass::Array<float, 2> x1Array{frg13[int32_t{26}], frg13[int32_t{27}]};
          cutlass::Array<float, 2> fusedScaleArray{(mScaleGate) * (mGatedActAlpha),
                                                   (mScaleGate) * (mGatedActAlpha)};
          cutlass::Array<float, 2> betaScaleGateArray{(mGatedActBeta) * (mScaleGate),
                                                      (mGatedActBeta) * (mScaleGate)};
          cutlass::Array<float, 2> scaleGateArray{mScaleGate, mScaleGate};
          cutlass::Array<float, 2> x0ScaleGateArray;
          x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
          cutlass::Array<float, 2> x1ScaledArray;
          x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
          float phi4015{trtllm::dev::phi(x1ScaledArray[int32_t{0}])};
          float phi4017{trtllm::dev::phi(x1ScaledArray[int32_t{1}])};
          cutlass::Array<float, 2> actArray{phi4015, phi4017};
          cutlass::Array<float, 2> x1PackedArray{x1Array[int32_t{0}], x1Array[int32_t{1}]};
          cutlass::Array<float, 2> geluArray;
          geluArray = trtllm::dev::fmul2(x1PackedArray, actArray);
          cutlass::Array<float, 2> gatedActArray;
          gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, geluArray);
          frg13[int32_t{24}] = gatedActArray[int32_t{0}];
          frg13[int32_t{25}] = gatedActArray[int32_t{1}];
        }
        {
          frg13[int32_t{12}] =
            float{trtllm::dev::clamp(frg13[int32_t{12}], -(mClampLimit), mClampLimit)};
          frg13[int32_t{14}] =
            float{trtllm::dev::clamp(frg13[int32_t{14}], float{-3.4028235e+38}, mClampLimit)};
          frg13[int32_t{13}] =
            float{trtllm::dev::clamp(frg13[int32_t{13}], -(mClampLimit), mClampLimit)};
          frg13[int32_t{15}] =
            float{trtllm::dev::clamp(frg13[int32_t{15}], float{-3.4028235e+38}, mClampLimit)};
          frg13[int32_t{12}] = (frg13[int32_t{12}]) * (float{1});
          frg13[int32_t{13}] = (frg13[int32_t{13}]) * (float{1});
          cutlass::Array<float, 2> x0Array{frg13[int32_t{12}], frg13[int32_t{13}]};
          cutlass::Array<float, 2> x1Array{frg13[int32_t{14}], frg13[int32_t{15}]};
          cutlass::Array<float, 2> fusedScaleArray{(mScaleGate) * (mGatedActAlpha),
                                                   (mScaleGate) * (mGatedActAlpha)};
          cutlass::Array<float, 2> betaScaleGateArray{(mGatedActBeta) * (mScaleGate),
                                                      (mGatedActBeta) * (mScaleGate)};
          cutlass::Array<float, 2> scaleGateArray{mScaleGate, mScaleGate};
          cutlass::Array<float, 2> x0ScaleGateArray;
          x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
          cutlass::Array<float, 2> x1ScaledArray;
          x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
          float phi4101{trtllm::dev::phi(x1ScaledArray[int32_t{0}])};
          float phi4103{trtllm::dev::phi(x1ScaledArray[int32_t{1}])};
          cutlass::Array<float, 2> actArray{phi4101, phi4103};
          cutlass::Array<float, 2> x1PackedArray{x1Array[int32_t{0}], x1Array[int32_t{1}]};
          cutlass::Array<float, 2> geluArray;
          geluArray = trtllm::dev::fmul2(x1PackedArray, actArray);
          cutlass::Array<float, 2> gatedActArray;
          gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, geluArray);
          frg13[int32_t{12}] = gatedActArray[int32_t{0}];
          frg13[int32_t{13}] = gatedActArray[int32_t{1}];
        }
        {
          frg13[int32_t{28}] =
            float{trtllm::dev::clamp(frg13[int32_t{28}], -(mClampLimit), mClampLimit)};
          frg13[int32_t{30}] =
            float{trtllm::dev::clamp(frg13[int32_t{30}], float{-3.4028235e+38}, mClampLimit)};
          frg13[int32_t{29}] =
            float{trtllm::dev::clamp(frg13[int32_t{29}], -(mClampLimit), mClampLimit)};
          frg13[int32_t{31}] =
            float{trtllm::dev::clamp(frg13[int32_t{31}], float{-3.4028235e+38}, mClampLimit)};
          frg13[int32_t{28}] = (frg13[int32_t{28}]) * (float{1});
          frg13[int32_t{29}] = (frg13[int32_t{29}]) * (float{1});
          cutlass::Array<float, 2> x0Array{frg13[int32_t{28}], frg13[int32_t{29}]};
          cutlass::Array<float, 2> x1Array{frg13[int32_t{30}], frg13[int32_t{31}]};
          cutlass::Array<float, 2> fusedScaleArray{(mScaleGate) * (mGatedActAlpha),
                                                   (mScaleGate) * (mGatedActAlpha)};
          cutlass::Array<float, 2> betaScaleGateArray{(mGatedActBeta) * (mScaleGate),
                                                      (mGatedActBeta) * (mScaleGate)};
          cutlass::Array<float, 2> scaleGateArray{mScaleGate, mScaleGate};
          cutlass::Array<float, 2> x0ScaleGateArray;
          x0ScaleGateArray = trtllm::dev::ffma2(x0Array, scaleGateArray, betaScaleGateArray);
          cutlass::Array<float, 2> x1ScaledArray;
          x1ScaledArray = trtllm::dev::fmul2(x1Array, fusedScaleArray);
          float phi4178{trtllm::dev::phi(x1ScaledArray[int32_t{0}])};
          float phi4180{trtllm::dev::phi(x1ScaledArray[int32_t{1}])};
          cutlass::Array<float, 2> actArray{phi4178, phi4180};
          cutlass::Array<float, 2> x1PackedArray{x1Array[int32_t{0}], x1Array[int32_t{1}]};
          cutlass::Array<float, 2> geluArray;
          geluArray = trtllm::dev::fmul2(x1PackedArray, actArray);
          cutlass::Array<float, 2> gatedActArray;
          gatedActArray = trtllm::dev::fmul2(x0ScaleGateArray, geluArray);
          frg13[int32_t{28}] = gatedActArray[int32_t{0}];
          frg13[int32_t{29}] = gatedActArray[int32_t{1}];
        }
        frg13[int32_t{2}] = frg13[int32_t{16}];
        frg13[int32_t{3}] = frg13[int32_t{17}];
        frg13[int32_t{6}] = frg13[int32_t{20}];
        frg13[int32_t{7}] = frg13[int32_t{21}];
        frg13[int32_t{10}] = frg13[int32_t{24}];
        frg13[int32_t{11}] = frg13[int32_t{25}];
        frg13[int32_t{14}] = frg13[int32_t{28}];
        frg13[int32_t{15}] = frg13[int32_t{29}];
        //
        // Compute block scaling.
        //
        cutlass::Array<float, 8> sfArrayPreQuant;
        cutlass::Array<float, 8> blockAbsMaxArray;
        {
          //
          // Compute amax (0,0).
          //
          float localAbsMax0;
          float localAbsMax1;
          localAbsMax0 = fabsf(frg13[int32_t{0}]);
          localAbsMax1 = fabsf(frg13[int32_t{1}]);
          localAbsMax0 = fmaxf(localAbsMax0, fabsf(frg13[int32_t{2}]));
          localAbsMax1 = fmaxf(localAbsMax1, fabsf(frg13[int32_t{3}]));
          cutlass::Array<float, 2> localAbsMaxArray{localAbsMax0, localAbsMax1};
          cutlass::Array<float, 2> scaleCArray{mScaleC, mScaleC};
          cutlass::Array<float, 2> localBlockAbsMaxArray;
          localBlockAbsMaxArray = trtllm::dev::fmul2(localAbsMaxArray, scaleCArray);
          localBlockAbsMaxArray[int32_t{0}] =
            trtllm::dev::reduce_group_max_abs_f32<8, 4>(localBlockAbsMaxArray[int32_t{0}],
                                                        mLaneIdx);
          localBlockAbsMaxArray[int32_t{1}] =
            trtllm::dev::reduce_group_max_abs_f32<8, 4>(localBlockAbsMaxArray[int32_t{1}],
                                                        mLaneIdx);
          blockAbsMaxArray[int32_t{0}] = localBlockAbsMaxArray[int32_t{0}];
          blockAbsMaxArray[int32_t{1}] = localBlockAbsMaxArray[int32_t{1}];
        }
        {
          //
          // Compute amax (0,2).
          //
          float localAbsMax0;
          float localAbsMax1;
          localAbsMax0 = fabsf(frg13[int32_t{4}]);
          localAbsMax1 = fabsf(frg13[int32_t{5}]);
          localAbsMax0 = fmaxf(localAbsMax0, fabsf(frg13[int32_t{6}]));
          localAbsMax1 = fmaxf(localAbsMax1, fabsf(frg13[int32_t{7}]));
          cutlass::Array<float, 2> localAbsMaxArray{localAbsMax0, localAbsMax1};
          cutlass::Array<float, 2> scaleCArray{mScaleC, mScaleC};
          cutlass::Array<float, 2> localBlockAbsMaxArray;
          localBlockAbsMaxArray = trtllm::dev::fmul2(localAbsMaxArray, scaleCArray);
          localBlockAbsMaxArray[int32_t{0}] =
            trtllm::dev::reduce_group_max_abs_f32<8, 4>(localBlockAbsMaxArray[int32_t{0}],
                                                        mLaneIdx);
          localBlockAbsMaxArray[int32_t{1}] =
            trtllm::dev::reduce_group_max_abs_f32<8, 4>(localBlockAbsMaxArray[int32_t{1}],
                                                        mLaneIdx);
          blockAbsMaxArray[int32_t{2}] = localBlockAbsMaxArray[int32_t{0}];
          blockAbsMaxArray[int32_t{3}] = localBlockAbsMaxArray[int32_t{1}];
        }
        {
          //
          // Compute amax (0,4).
          //
          float localAbsMax0;
          float localAbsMax1;
          localAbsMax0 = fabsf(frg13[int32_t{8}]);
          localAbsMax1 = fabsf(frg13[int32_t{9}]);
          localAbsMax0 = fmaxf(localAbsMax0, fabsf(frg13[int32_t{10}]));
          localAbsMax1 = fmaxf(localAbsMax1, fabsf(frg13[int32_t{11}]));
          cutlass::Array<float, 2> localAbsMaxArray{localAbsMax0, localAbsMax1};
          cutlass::Array<float, 2> scaleCArray{mScaleC, mScaleC};
          cutlass::Array<float, 2> localBlockAbsMaxArray;
          localBlockAbsMaxArray = trtllm::dev::fmul2(localAbsMaxArray, scaleCArray);
          localBlockAbsMaxArray[int32_t{0}] =
            trtllm::dev::reduce_group_max_abs_f32<8, 4>(localBlockAbsMaxArray[int32_t{0}],
                                                        mLaneIdx);
          localBlockAbsMaxArray[int32_t{1}] =
            trtllm::dev::reduce_group_max_abs_f32<8, 4>(localBlockAbsMaxArray[int32_t{1}],
                                                        mLaneIdx);
          blockAbsMaxArray[int32_t{4}] = localBlockAbsMaxArray[int32_t{0}];
          blockAbsMaxArray[int32_t{5}] = localBlockAbsMaxArray[int32_t{1}];
        }
        {
          //
          // Compute amax (0,6).
          //
          float localAbsMax0;
          float localAbsMax1;
          localAbsMax0 = fabsf(frg13[int32_t{12}]);
          localAbsMax1 = fabsf(frg13[int32_t{13}]);
          localAbsMax0 = fmaxf(localAbsMax0, fabsf(frg13[int32_t{14}]));
          localAbsMax1 = fmaxf(localAbsMax1, fabsf(frg13[int32_t{15}]));
          cutlass::Array<float, 2> localAbsMaxArray{localAbsMax0, localAbsMax1};
          cutlass::Array<float, 2> scaleCArray{mScaleC, mScaleC};
          cutlass::Array<float, 2> localBlockAbsMaxArray;
          localBlockAbsMaxArray = trtllm::dev::fmul2(localAbsMaxArray, scaleCArray);
          localBlockAbsMaxArray[int32_t{0}] =
            trtllm::dev::reduce_group_max_abs_f32<8, 4>(localBlockAbsMaxArray[int32_t{0}],
                                                        mLaneIdx);
          localBlockAbsMaxArray[int32_t{1}] =
            trtllm::dev::reduce_group_max_abs_f32<8, 4>(localBlockAbsMaxArray[int32_t{1}],
                                                        mLaneIdx);
          blockAbsMaxArray[int32_t{6}] = localBlockAbsMaxArray[int32_t{0}];
          blockAbsMaxArray[int32_t{7}] = localBlockAbsMaxArray[int32_t{1}];
        }
        {
          //
          // Compute block SF (0,0).
          //
          float blockAbsMax0;
          blockAbsMax0 = blockAbsMaxArray[int32_t{0}];
          float blockAbsMax1;
          blockAbsMax1 = blockAbsMaxArray[int32_t{1}];
          cutlass::Array<float, 2> operandsArray{blockAbsMax0, blockAbsMax1};
          cutlass::Array<float, 2> reciprocalArray{(float{1}) / (float{6}),
                                                   (float{1}) / (float{6})};
          cutlass::Array<float, 2> blockSfHighArray;
          blockSfHighArray = trtllm::dev::fmul2(operandsArray, reciprocalArray);
          sfArrayPreQuant[int32_t{0}] = blockSfHighArray[int32_t{0}];
          sfArrayPreQuant[int32_t{1}] = blockSfHighArray[int32_t{1}];
        }
        {
          //
          // Compute block SF (0,2).
          //
          float blockAbsMax0;
          blockAbsMax0 = blockAbsMaxArray[int32_t{2}];
          float blockAbsMax1;
          blockAbsMax1 = blockAbsMaxArray[int32_t{3}];
          cutlass::Array<float, 2> operandsArray{blockAbsMax0, blockAbsMax1};
          cutlass::Array<float, 2> reciprocalArray{(float{1}) / (float{6}),
                                                   (float{1}) / (float{6})};
          cutlass::Array<float, 2> blockSfHighArray;
          blockSfHighArray = trtllm::dev::fmul2(operandsArray, reciprocalArray);
          sfArrayPreQuant[int32_t{2}] = blockSfHighArray[int32_t{0}];
          sfArrayPreQuant[int32_t{3}] = blockSfHighArray[int32_t{1}];
        }
        {
          //
          // Compute block SF (0,4).
          //
          float blockAbsMax0;
          blockAbsMax0 = blockAbsMaxArray[int32_t{4}];
          float blockAbsMax1;
          blockAbsMax1 = blockAbsMaxArray[int32_t{5}];
          cutlass::Array<float, 2> operandsArray{blockAbsMax0, blockAbsMax1};
          cutlass::Array<float, 2> reciprocalArray{(float{1}) / (float{6}),
                                                   (float{1}) / (float{6})};
          cutlass::Array<float, 2> blockSfHighArray;
          blockSfHighArray = trtllm::dev::fmul2(operandsArray, reciprocalArray);
          sfArrayPreQuant[int32_t{4}] = blockSfHighArray[int32_t{0}];
          sfArrayPreQuant[int32_t{5}] = blockSfHighArray[int32_t{1}];
        }
        {
          //
          // Compute block SF (0,6).
          //
          float blockAbsMax0;
          blockAbsMax0 = blockAbsMaxArray[int32_t{6}];
          float blockAbsMax1;
          blockAbsMax1 = blockAbsMaxArray[int32_t{7}];
          cutlass::Array<float, 2> operandsArray{blockAbsMax0, blockAbsMax1};
          cutlass::Array<float, 2> reciprocalArray{(float{1}) / (float{6}),
                                                   (float{1}) / (float{6})};
          cutlass::Array<float, 2> blockSfHighArray;
          blockSfHighArray = trtllm::dev::fmul2(operandsArray, reciprocalArray);
          sfArrayPreQuant[int32_t{6}] = blockSfHighArray[int32_t{0}];
          sfArrayPreQuant[int32_t{7}] = blockSfHighArray[int32_t{1}];
        }
        cutlass::Array<cutlass::float_e4m3_t, 8> sfArrayQuant{
          trtllm::dev::castArray<cutlass::float_e4m3_t, float, 8>(sfArrayPreQuant)};
        cutlass::Array<float, 8> sfArrayPostQuant{
          trtllm::dev::castArray<float, cutlass::float_e4m3_t, 8>(sfArrayQuant)};
        cutlass::Array<float, 16> finalAccArray;
        {
          //
          // Scale by block SF (0,0).
          //
          float decBlockSf;
          decBlockSf = sfArrayPostQuant[int32_t{0}];
          float encBlockSf;
          if ((decBlockSf) == (float{0})) {
            encBlockSf = float{0};
          } else {
            encBlockSf = (float{1}) / (decBlockSf);
          }
          finalAccArray[int32_t{0}] = (frg13[int32_t{0}]) * (encBlockSf);
          finalAccArray[int32_t{2}] = (frg13[int32_t{2}]) * (encBlockSf);
        }
        {
          //
          // Scale by block SF (0,1).
          //
          float decBlockSf;
          decBlockSf = sfArrayPostQuant[int32_t{1}];
          float encBlockSf;
          if ((decBlockSf) == (float{0})) {
            encBlockSf = float{0};
          } else {
            encBlockSf = (float{1}) / (decBlockSf);
          }
          finalAccArray[int32_t{1}] = (frg13[int32_t{1}]) * (encBlockSf);
          finalAccArray[int32_t{3}] = (frg13[int32_t{3}]) * (encBlockSf);
        }
        {
          //
          // Scale by block SF (0,2).
          //
          float decBlockSf;
          decBlockSf = sfArrayPostQuant[int32_t{2}];
          float encBlockSf;
          if ((decBlockSf) == (float{0})) {
            encBlockSf = float{0};
          } else {
            encBlockSf = (float{1}) / (decBlockSf);
          }
          finalAccArray[int32_t{4}] = (frg13[int32_t{4}]) * (encBlockSf);
          finalAccArray[int32_t{6}] = (frg13[int32_t{6}]) * (encBlockSf);
        }
        {
          //
          // Scale by block SF (0,3).
          //
          float decBlockSf;
          decBlockSf = sfArrayPostQuant[int32_t{3}];
          float encBlockSf;
          if ((decBlockSf) == (float{0})) {
            encBlockSf = float{0};
          } else {
            encBlockSf = (float{1}) / (decBlockSf);
          }
          finalAccArray[int32_t{5}] = (frg13[int32_t{5}]) * (encBlockSf);
          finalAccArray[int32_t{7}] = (frg13[int32_t{7}]) * (encBlockSf);
        }
        {
          //
          // Scale by block SF (0,4).
          //
          float decBlockSf;
          decBlockSf = sfArrayPostQuant[int32_t{4}];
          float encBlockSf;
          if ((decBlockSf) == (float{0})) {
            encBlockSf = float{0};
          } else {
            encBlockSf = (float{1}) / (decBlockSf);
          }
          finalAccArray[int32_t{8}] = (frg13[int32_t{8}]) * (encBlockSf);
          finalAccArray[int32_t{10}] = (frg13[int32_t{10}]) * (encBlockSf);
        }
        {
          //
          // Scale by block SF (0,5).
          //
          float decBlockSf;
          decBlockSf = sfArrayPostQuant[int32_t{5}];
          float encBlockSf;
          if ((decBlockSf) == (float{0})) {
            encBlockSf = float{0};
          } else {
            encBlockSf = (float{1}) / (decBlockSf);
          }
          finalAccArray[int32_t{9}] = (frg13[int32_t{9}]) * (encBlockSf);
          finalAccArray[int32_t{11}] = (frg13[int32_t{11}]) * (encBlockSf);
        }
        {
          //
          // Scale by block SF (0,6).
          //
          float decBlockSf;
          decBlockSf = sfArrayPostQuant[int32_t{6}];
          float encBlockSf;
          if ((decBlockSf) == (float{0})) {
            encBlockSf = float{0};
          } else {
            encBlockSf = (float{1}) / (decBlockSf);
          }
          finalAccArray[int32_t{12}] = (frg13[int32_t{12}]) * (encBlockSf);
          finalAccArray[int32_t{14}] = (frg13[int32_t{14}]) * (encBlockSf);
        }
        {
          //
          // Scale by block SF (0,7).
          //
          float decBlockSf;
          decBlockSf = sfArrayPostQuant[int32_t{7}];
          float encBlockSf;
          if ((decBlockSf) == (float{0})) {
            encBlockSf = float{0};
          } else {
            encBlockSf = (float{1}) / (decBlockSf);
          }
          finalAccArray[int32_t{13}] = (frg13[int32_t{13}]) * (encBlockSf);
          finalAccArray[int32_t{15}] = (frg13[int32_t{15}]) * (encBlockSf);
        }
        //
        // Store block scaling factors to Gmem.
        //
        {
          int32_t const threadIdxInGroup{(mLaneIdx) / (int32_t{4})};
          cutlass::float_e4m3_t* ptrSfOut;
          ptrSfOut = reinterpret_cast<cutlass::float_e4m3_t*>(params.ptrSfC) + (int32_t{0});
          int32_t offsetM{(mCtaIdxX) * (int32_t{64})};
          int32_t offsetN{(mWarpGrp4Idx) * (int32_t{32}) + ((mCtaIdxY) * (int32_t{32}))};
          cutlass::float_e4m3_t* sfArrayPtr{sfArrayQuant.data()};
          uint8_t* sfArrayPackedPtr;
          sfArrayPackedPtr = reinterpret_cast<uint8_t*>(sfArrayPtr) + (int32_t{0});
          {
            //
            // Store SF vector (0...,0).
            //
            uint8_t vecSf;
            int32_t localRowIdx;
            int32_t localColIdx;
            vecSf = sfArrayPackedPtr[int32_t{0}];
            localRowIdx = mLdtm16dp256bitTmemRowIdx;
            localColIdx = mLdtm16dp256bitTmemColIdx;
            if ((threadIdxInGroup) == (int32_t{1})) {
              vecSf = sfArrayPackedPtr[int32_t{1}];
              localRowIdx = mLdtm16dp256bitTmemRowIdx;
              localColIdx = (mLdtm16dp256bitTmemColIdx) + (int32_t{1});
            }
            if ((threadIdxInGroup) == (int32_t{2})) {
              vecSf = sfArrayPackedPtr[int32_t{2}];
              localRowIdx = mLdtm16dp256bitTmemRowIdx;
              localColIdx = (mLdtm16dp256bitTmemColIdx) + (int32_t{8});
            }
            if ((threadIdxInGroup) == (int32_t{3})) {
              vecSf = sfArrayPackedPtr[int32_t{3}];
              localRowIdx = mLdtm16dp256bitTmemRowIdx;
              localColIdx = (mLdtm16dp256bitTmemColIdx) + (int32_t{9});
            }
            if ((threadIdxInGroup) == (int32_t{4})) {
              vecSf = sfArrayPackedPtr[int32_t{4}];
              localRowIdx = mLdtm16dp256bitTmemRowIdx;
              localColIdx = (mLdtm16dp256bitTmemColIdx) + (int32_t{16});
            }
            if ((threadIdxInGroup) == (int32_t{5})) {
              vecSf = sfArrayPackedPtr[int32_t{5}];
              localRowIdx = mLdtm16dp256bitTmemRowIdx;
              localColIdx = (mLdtm16dp256bitTmemColIdx) + (int32_t{17});
            }
            if ((threadIdxInGroup) == (int32_t{6})) {
              vecSf = sfArrayPackedPtr[int32_t{6}];
              localRowIdx = mLdtm16dp256bitTmemRowIdx;
              localColIdx = (mLdtm16dp256bitTmemColIdx) + (int32_t{24});
            }
            if ((threadIdxInGroup) == (int32_t{7})) {
              vecSf = sfArrayPackedPtr[int32_t{7}];
              localRowIdx = mLdtm16dp256bitTmemRowIdx;
              localColIdx = (mLdtm16dp256bitTmemColIdx) + (int32_t{25});
            }
            int32_t eltIdxM{(offsetM) + (localRowIdx)};
            int32_t eltIdxN{(offsetN) + (localColIdx)};
            int32_t sfIdx0{(eltIdxN) / (int32_t{8})};
            int32_t sfIdx1{((eltIdxM) / (int32_t{16})) / (int32_t{4})};
            int32_t sfIdx2{(eltIdxN) % (int32_t{8})};
            int32_t sfIdx3{((eltIdxM) / (int32_t{16})) % (int32_t{4})};
            int32_t sfVecIdx{
              (((sfIdx0) * (((((params.nm) / (int32_t{2})) + (int32_t{63})) / (int32_t{64})) *
                            (int32_t{32}))) +
               ((sfIdx1) * (int32_t{32}))) +
              (((sfIdx2) * (int32_t{4})) + (sfIdx3))};
            if (((eltIdxN) < (mBatchLimit)) && ((eltIdxM) < ((params.nm) / (int32_t{2})))) {
              reinterpret_cast<uint8_t*>(ptrSfOut)[sfVecIdx] = vecSf;
            }
          }
        }
        cuda_ptx::cp_async_bulk_wait_group_read(cuda_ptx::n32_t<0>{});
        trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{8}) + (mWarpGrp4Idx));
        //
        // Store to Smem TmaAsyncGmemC.
        //
        int8_t* ptrSmemBase;
        cutlass::float_e2m1_t* ptrSmem;
        ptrSmemBase = reinterpret_cast<int8_t*>(gmemC0DstStack.mDepSmemPtr0) +
                      ((mWarpGrp4Idx) * (int32_t{1024}) + (int32_t{184320}));
        ptrSmem = reinterpret_cast<cutlass::float_e2m1_t*>(ptrSmemBase) + (int32_t{0});
        //
        // Smem store idxM=0 idxN=0.
        //
        {
          int32_t smemOffset0;
          {
            int32_t const smemRowIdx{((mBaseTmemCol) * (int32_t{64}) + (mBaseRowIdx)) /
                                     (int32_t{256})};
            int32_t const smemOffsetInBytes{
              (((mBaseTmemCol) * (int32_t{64}) + (mBaseRowIdx)) * (int32_t{4})) / (int32_t{8})};
            int32_t const swizzleMask{((smemRowIdx) % (int32_t{2})) * (int32_t{16})};
            smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{4});
          }
          cutlass::Array<float, 2> scaleF2{mScaleC, mScaleC};
          cutlass::Array<float, 2> accF2{finalAccArray[int32_t{0}], finalAccArray[int32_t{2}]};
          cutlass::Array<float, 2> scaledAccF2{trtllm::dev::fmul2(accF2, scaleF2)};
          cutlass::Array<cutlass::float_e2m1_t, 2> scaledCvtAcc2{
            trtllm::dev::convert_float2_to_e2m1(scaledAccF2)};
          {
            uint8_t convertedElts;
            convertedElts = reinterpret_cast<uint8_t&>(scaledCvtAcc2);
            reinterpret_cast<uint8_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
          }
        }
        //
        // Smem store idxM=0 idxN=1.
        //
        {
          int32_t smemOffset0;
          {
            int32_t const smemRowIdx{
              (((mBaseTmemCol) + (int32_t{1})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{256})};
            int32_t const smemOffsetInBytes{
              ((((mBaseTmemCol) + (int32_t{1})) * (int32_t{64}) + (mBaseRowIdx)) * (int32_t{4})) /
              (int32_t{8})};
            int32_t const swizzleMask{((smemRowIdx) % (int32_t{2})) * (int32_t{16})};
            smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{4});
          }
          cutlass::Array<float, 2> scaleF2{mScaleC, mScaleC};
          cutlass::Array<float, 2> accF2{finalAccArray[int32_t{1}], finalAccArray[int32_t{3}]};
          cutlass::Array<float, 2> scaledAccF2{trtllm::dev::fmul2(accF2, scaleF2)};
          cutlass::Array<cutlass::float_e2m1_t, 2> scaledCvtAcc2{
            trtllm::dev::convert_float2_to_e2m1(scaledAccF2)};
          {
            uint8_t convertedElts;
            convertedElts = reinterpret_cast<uint8_t&>(scaledCvtAcc2);
            reinterpret_cast<uint8_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
          }
        }
        //
        // Smem store idxM=0 idxN=2.
        //
        {
          int32_t smemOffset0;
          {
            int32_t const smemRowIdx{
              (((mBaseTmemCol) + (int32_t{8})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{256})};
            int32_t const smemOffsetInBytes{
              ((((mBaseTmemCol) + (int32_t{8})) * (int32_t{64}) + (mBaseRowIdx)) * (int32_t{4})) /
              (int32_t{8})};
            int32_t const swizzleMask{((smemRowIdx) % (int32_t{2})) * (int32_t{16})};
            smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{4});
          }
          cutlass::Array<float, 2> scaleF2{mScaleC, mScaleC};
          cutlass::Array<float, 2> accF2{finalAccArray[int32_t{4}], finalAccArray[int32_t{6}]};
          cutlass::Array<float, 2> scaledAccF2{trtllm::dev::fmul2(accF2, scaleF2)};
          cutlass::Array<cutlass::float_e2m1_t, 2> scaledCvtAcc2{
            trtllm::dev::convert_float2_to_e2m1(scaledAccF2)};
          {
            uint8_t convertedElts;
            convertedElts = reinterpret_cast<uint8_t&>(scaledCvtAcc2);
            reinterpret_cast<uint8_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
          }
        }
        //
        // Smem store idxM=0 idxN=3.
        //
        {
          int32_t smemOffset0;
          {
            int32_t const smemRowIdx{
              (((mBaseTmemCol) + (int32_t{9})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{256})};
            int32_t const smemOffsetInBytes{
              ((((mBaseTmemCol) + (int32_t{9})) * (int32_t{64}) + (mBaseRowIdx)) * (int32_t{4})) /
              (int32_t{8})};
            int32_t const swizzleMask{((smemRowIdx) % (int32_t{2})) * (int32_t{16})};
            smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{4});
          }
          cutlass::Array<float, 2> scaleF2{mScaleC, mScaleC};
          cutlass::Array<float, 2> accF2{finalAccArray[int32_t{5}], finalAccArray[int32_t{7}]};
          cutlass::Array<float, 2> scaledAccF2{trtllm::dev::fmul2(accF2, scaleF2)};
          cutlass::Array<cutlass::float_e2m1_t, 2> scaledCvtAcc2{
            trtllm::dev::convert_float2_to_e2m1(scaledAccF2)};
          {
            uint8_t convertedElts;
            convertedElts = reinterpret_cast<uint8_t&>(scaledCvtAcc2);
            reinterpret_cast<uint8_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
          }
        }
        //
        // Smem store idxM=0 idxN=4.
        //
        {
          int32_t smemOffset0;
          {
            int32_t const smemRowIdx{
              (((mBaseTmemCol) + (int32_t{16})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{256})};
            int32_t const smemOffsetInBytes{
              ((((mBaseTmemCol) + (int32_t{16})) * (int32_t{64}) + (mBaseRowIdx)) * (int32_t{4})) /
              (int32_t{8})};
            int32_t const swizzleMask{((smemRowIdx) % (int32_t{2})) * (int32_t{16})};
            smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{4});
          }
          cutlass::Array<float, 2> scaleF2{mScaleC, mScaleC};
          cutlass::Array<float, 2> accF2{finalAccArray[int32_t{8}], finalAccArray[int32_t{10}]};
          cutlass::Array<float, 2> scaledAccF2{trtllm::dev::fmul2(accF2, scaleF2)};
          cutlass::Array<cutlass::float_e2m1_t, 2> scaledCvtAcc2{
            trtllm::dev::convert_float2_to_e2m1(scaledAccF2)};
          {
            uint8_t convertedElts;
            convertedElts = reinterpret_cast<uint8_t&>(scaledCvtAcc2);
            reinterpret_cast<uint8_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
          }
        }
        //
        // Smem store idxM=0 idxN=5.
        //
        {
          int32_t smemOffset0;
          {
            int32_t const smemRowIdx{
              (((mBaseTmemCol) + (int32_t{17})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{256})};
            int32_t const smemOffsetInBytes{
              ((((mBaseTmemCol) + (int32_t{17})) * (int32_t{64}) + (mBaseRowIdx)) * (int32_t{4})) /
              (int32_t{8})};
            int32_t const swizzleMask{((smemRowIdx) % (int32_t{2})) * (int32_t{16})};
            smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{4});
          }
          cutlass::Array<float, 2> scaleF2{mScaleC, mScaleC};
          cutlass::Array<float, 2> accF2{finalAccArray[int32_t{9}], finalAccArray[int32_t{11}]};
          cutlass::Array<float, 2> scaledAccF2{trtllm::dev::fmul2(accF2, scaleF2)};
          cutlass::Array<cutlass::float_e2m1_t, 2> scaledCvtAcc2{
            trtllm::dev::convert_float2_to_e2m1(scaledAccF2)};
          {
            uint8_t convertedElts;
            convertedElts = reinterpret_cast<uint8_t&>(scaledCvtAcc2);
            reinterpret_cast<uint8_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
          }
        }
        //
        // Smem store idxM=0 idxN=6.
        //
        {
          int32_t smemOffset0;
          {
            int32_t const smemRowIdx{
              (((mBaseTmemCol) + (int32_t{24})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{256})};
            int32_t const smemOffsetInBytes{
              ((((mBaseTmemCol) + (int32_t{24})) * (int32_t{64}) + (mBaseRowIdx)) * (int32_t{4})) /
              (int32_t{8})};
            int32_t const swizzleMask{((smemRowIdx) % (int32_t{2})) * (int32_t{16})};
            smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{4});
          }
          cutlass::Array<float, 2> scaleF2{mScaleC, mScaleC};
          cutlass::Array<float, 2> accF2{finalAccArray[int32_t{12}], finalAccArray[int32_t{14}]};
          cutlass::Array<float, 2> scaledAccF2{trtllm::dev::fmul2(accF2, scaleF2)};
          cutlass::Array<cutlass::float_e2m1_t, 2> scaledCvtAcc2{
            trtllm::dev::convert_float2_to_e2m1(scaledAccF2)};
          {
            uint8_t convertedElts;
            convertedElts = reinterpret_cast<uint8_t&>(scaledCvtAcc2);
            reinterpret_cast<uint8_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
          }
        }
        //
        // Smem store idxM=0 idxN=7.
        //
        {
          int32_t smemOffset0;
          {
            int32_t const smemRowIdx{
              (((mBaseTmemCol) + (int32_t{25})) * (int32_t{64}) + (mBaseRowIdx)) / (int32_t{256})};
            int32_t const smemOffsetInBytes{
              ((((mBaseTmemCol) + (int32_t{25})) * (int32_t{64}) + (mBaseRowIdx)) * (int32_t{4})) /
              (int32_t{8})};
            int32_t const swizzleMask{((smemRowIdx) % (int32_t{2})) * (int32_t{16})};
            smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{4});
          }
          cutlass::Array<float, 2> scaleF2{mScaleC, mScaleC};
          cutlass::Array<float, 2> accF2{finalAccArray[int32_t{13}], finalAccArray[int32_t{15}]};
          cutlass::Array<float, 2> scaledAccF2{trtllm::dev::fmul2(accF2, scaleF2)};
          cutlass::Array<cutlass::float_e2m1_t, 2> scaledCvtAcc2{
            trtllm::dev::convert_float2_to_e2m1(scaledAccF2)};
          {
            uint8_t convertedElts;
            convertedElts = reinterpret_cast<uint8_t&>(scaledCvtAcc2);
            reinterpret_cast<uint8_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
          }
        }
        cuda_ptx::fence_proxy_async(cuda_ptx::space_shared_t{});
        trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{8}) + (mWarpGrp4Idx));
        //
        // Issue TMA from smem to gmem.
        //
        if ((bool{cute::elect_one_sync()}) && ((mWarpGrp4WarpIdx) == (int32_t{0}))) {
          int8_t* ptrSmemBase;
          cutlass::float_e2m1_t* ptrSmem;
          ptrSmemBase = reinterpret_cast<int8_t*>(gmemC0DstStack.mDepSmemPtr0) +
                        ((mWarpGrp4Idx) * (int32_t{1024}) + (int32_t{184320}));
          ptrSmem = reinterpret_cast<cutlass::float_e2m1_t*>(ptrSmemBase) + (int32_t{0});
          int32_t coords[4];
          coords[int32_t{0}] = (mCtaIdxX) * (int32_t{64});
          coords[int32_t{1}] = (((int32_t{32}) - ((mBatchLimit) % (int32_t{32}))) % (int32_t{32})) +
                               ((mWarpGrp4Idx) * (int32_t{32}));
          coords[int32_t{2}] = int32_t{0x40000000 /*1073741824*/};
          coords[int32_t{3}] =
            (((mCtaIdxY) * (int32_t{32})) +
             ((int32_t{0}) - (((int32_t{32}) - ((mBatchLimit) % (int32_t{32}))) % (int32_t{32})))) +
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
    //
    // mma0 [ConsTailRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    if (hasOneLoopIter) {
      {
        mma0SrcStack.mPipeline.consumer_release(mma0ConsState);
      }
      ++mma0ConsState;
    }
  ExitTileWithoutSignalingLabel: {}
  }
};
extern "C" __global__ __launch_bounds__(768, 1)
  __cluster_dims__(2, 1, 1) void bmm_E2m1_E2m1E2m1_Fp32_bA16_bB16_bC16_t128x32x512u2_s5_et128x32_m256x32x64_cga2x1x1_16dp256b_rM_TN_transOut_schedS_biasM_bN_tma_tmaSf_rgTma_clmp_geGlu_dynB_sm100f(
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
  uint32_t* TmemSwStatePtr{
    reinterpret_cast<uint32_t*>((reinterpret_cast<uint8_t*>(smem__) + smemOffset__))};
  smemOffset__ += int32_t{16};
  uint8_t* smemASmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemASmemBarrier)});
  uint8_t* smemBSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemBSmemBarrier)});
  uint8_t* smemSfASmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemSfASmemBarrier)});
  uint8_t* smemSfBSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemSfBSmemBarrier)});
  uint8_t* tmemSfASmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(TmemSfASmemBarrier)});
  uint8_t* tmemSfBSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(TmemSfBSmemBarrier)});
  uint8_t* mma0SmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(Mma0SmemBarrier)});
  uint8_t* clusterBarrierBuffersSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(ClusterBarrierBuffersSmemBarrier)});
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
                        int32_t{20},
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
                            int32_t{21},
                            int32_t{-1}};
  SmemSfBSmem* smemSfBSmem{reinterpret_cast<SmemSfBSmem*>(smemSfBSmemPtr)};
  SmemSfBSmemBarrier* smemSfBSmemBarrier{
    reinterpret_cast<SmemSfBSmemBarrier*>(smemSfBSmemBarrierPtr)};
  SmemSfBStack smemSfBStack{(*smemSfBSmem),
                            (*smemSfBSmemBarrier),
                            state.mWarpIdx,
                            int32_t{12},
                            int32_t{-1}};
  TmemSfASmemBarrier* tmemSfASmemBarrier{
    reinterpret_cast<TmemSfASmemBarrier*>(tmemSfASmemBarrierPtr)};
  TmemSfAStack tmemSfAStack{(*tmemSfASmemBarrier), state.mWarpIdx, int32_t{22}, int32_t{-1}};
  TmemSfBSmemBarrier* tmemSfBSmemBarrier{
    reinterpret_cast<TmemSfBSmemBarrier*>(tmemSfBSmemBarrierPtr)};
  TmemSfBStack tmemSfBStack{(*tmemSfBSmemBarrier), state.mWarpIdx, int32_t{4}, int32_t{-1}};
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
  LoadTaskA loadTaskA{params, state, int32_t{20}};
  LoadTaskB loadTaskB{params, state, int32_t{8}};
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
    trtllm::dev::CutlassNamedBarrier::sync(320, 10);
  }
  if (bool{LoadTaskA::isSelected(params, state)}) {
    loadTaskA.execute(params, state, (*smemASmem), smemAStack);
  } else {
    if (bool{LoadTaskB::isSelected(params, state)}) {
      loadTaskB.execute(params, state, (*smemBSmem), smemBStack);
    } else {
      if (bool{LoadSfATask::isSelected(params, state)}) {
        LoadSfATask loadSfATask{params, state, int32_t{21}};
        loadSfATask.execute(params, state, (*smemSfASmem), smemSfAStack);
      } else {
        if (bool{LoadSfBTask::isSelected(params, state)}) {
          LoadSfBTask loadSfBTask{params, state, int32_t{12}};
          loadSfBTask.execute(params, state, (*smemSfBSmem), smemSfBStack);
        } else {
          if (bool{CopySfATask::isSelected(params, state)}) {
            CopySfATask copySfATask{params, state, int32_t{22}};
            copySfATask.execute(params, state, tmemSfAStack, (*smemSfASmem), smemSfAStack);
          } else {
            if (bool{CopySfBTask::isSelected(params, state)}) {
              CopySfBTask copySfBTask{params, state, int32_t{4}};
              copySfBTask.execute(params, state, tmemSfBStack, (*smemSfBSmem), smemSfBStack);
            } else {
              if (bool{MmaTask0::isSelected(params, state)}) {
                MmaTask0 mmaTask0{params, state, int32_t{23}};
                mmaTask0.execute(params,
                                 state,
                                 mma0Stack,
                                 (*smemASmem),
                                 smemAStack,
                                 (*smemBSmem),
                                 smemBStack,
                                 tmemSfAStack,
                                 tmemSfBStack);
              } else {
                if (bool{EpilogueTask0::isSelected(params, state)}) {
                  EpilogueTask0 epilogueTask0{params, state, int32_t{0}};
                  epilogueTask0.execute(params,
                                        state,
                                        (*gmemC0Smem),
                                        gmemC0Stack,
                                        clusterBarrierBuffersStack,
                                        mma0Stack);
                  trtllm::dev::CutlassNamedBarrier::sync(128, 11);
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
bmm_E2m1_E2m1E2m1_Fp32_bA16_bB16_bC16_t128x32x512u2_s5_et128x32_m256x32x64_cga2x1x1_16dp256b_rM_TN_transOut_schedS_biasM_bN_tma_tmaSf_rgTma_clmp_geGlu_dynB_sm100fGetSmemSize(
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
  size += 16;
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(SmemASmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(SmemBSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(SmemSfASmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(SmemSfBSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(TmemSfASmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(TmemSfBSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(Mma0SmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(ClusterBarrierBuffersSmemBarrier));
  outPtr[0] = size;
}

} // namespace batchedGemm
