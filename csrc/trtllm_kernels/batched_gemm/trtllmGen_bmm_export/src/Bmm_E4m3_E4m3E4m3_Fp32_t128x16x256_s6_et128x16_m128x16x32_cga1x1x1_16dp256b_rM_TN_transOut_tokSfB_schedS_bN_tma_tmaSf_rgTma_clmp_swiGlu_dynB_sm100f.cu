#include <Bmm_E4m3_E4m3E4m3_Fp32_t128x16x256_s6_et128x16_m128x16x32_cga1x1x1_16dp256b_rM_TN_transOut_tokSfB_schedS_bN_tma_tmaSf_rgTma_clmp_swiGlu_dynB_sm100f.h>
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
    CutlassTmaMultiUmmaAsyncPipeline<6, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
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
                int32_t{32768},
                ((warpId) == (barInitWarpId)) && (bool{cute::elect_one_sync()}),
                int32_t{1},
                CuteFlatTuple195{},
                cute::true_type{},
                cute::true_type{},
                barInitWarpId} {}
};
struct SmemBStack {
  int8_t* mDepSmemPtr0;
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
    : mDepSmemPtr0{&smemBufferSmem.mArray[int32_t{0}]}
    , mPipeline{smemBSmemBarrier.mBarriers,
                warpId,
                int32_t{4096},
                ((warpId) == (barInitWarpId)) && (bool{cute::elect_one_sync()}),
                int32_t{1},
                CuteFlatTuple317{},
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
  trtllm::dev::CutlassUmmaAsyncPipeline<1, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  inline __device__ Mma0Stack(Mma0SmemBarrier& mma0SmemBarrier,
                              TmemStack& tmemStack,
                              int32_t warpId,
                              int32_t barInitWarpId,
                              int32_t orderedSequenceGroupId)
    : mPipeline{mma0SmemBarrier.mBarriers,
                warpId,
                int32_t{128},
                CuteFlatTuple436{},
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
    return ((state.mWarpIdx) >= (int32_t{5})) && ((state.mWarpIdx) < (int32_t{6}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 SmemASmem& smemADstSmem,
                                 SmemAStack& smemADstStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<48>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      6,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemAProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    int32_t smemAProdToken{int32_t{1}};
    int32_t paddedPerCtaK{(((params.k) + (int32_t{255})) / (int32_t{256})) * (int32_t{256})};
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
    for (int32_t loopOffset164 = int32_t{0}; loopOffset164 < loopEnd;
         loopOffset164 += int32_t{256}) {
      bool const isLastLoopIter{((loopOffset164) + (int32_t{256})) >= (loopEnd)};
      //
      // gmemA [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK1;
      { tileOffsetK1 = loopOffset164; }
      //
      // smemA [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {
        smemADstStack.mPipeline.producer_acquire(smemAProdState, smemAProdToken);
        if (((loopOffset164) + (int32_t{256})) < (loopEnd)) {
          smemAProdToken = smemADstStack.mPipeline.producer_try_acquire(
            trtllm::dev::makePipelineState(smemAProdState, int32_t{1}));
        }
      }
      //
      // smemA [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK3;
      tileOffsetK3 = tileOffsetK1;
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
          coords[int32_t{0}] = tileOffsetK3;
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
          coords[int32_t{0}] += int32_t{128};
          if (bool{cute::elect_one_sync()}) {
            cuda_ptx::cp_async_bulk_tensor(
              cuda_ptx::space_cluster_t{},
              cuda_ptx::space_global_t{},
              &reinterpret_cast<cutlass::float_e4m3_t*>(smemBytesStagePtrA)[int32_t{16384}],
              params.tmaA,
              coords,
              barrier);
          }
        }
      }
      //
      // smemA [ProdPreCommit, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if ((loopOffset164) >= (int32_t{0})) {
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
    , mWarpGrpThreadIdx{min(int32_t{128},
                            max(int32_t{0}, (state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))))}
    , mWarpGrpWarpIdx{(state.mWarpIdx) - (warpGrpStart)}
    , mCtaOffsetK{int32_t{0}} {
    cudaGridDependencySynchronize();
    {
      if ((((mWarpGrpWarpIdx) * (int32_t{4})) < (int32_t{16})) &&
          (bool{LoadTaskB::isSelected(params, state)})) {
        mRoutedIndices[int32_t{0}] = int32_t{
          params.ptrRouteMap[((mWarpGrpWarpIdx) * (int32_t{4})) + ((mCtaIdxY) * (int32_t{16}))]};
        mRoutedIndices[int32_t{1}] =
          int32_t{params.ptrRouteMap[((mWarpGrpWarpIdx) * (int32_t{4}) + (int32_t{1})) +
                                     ((mCtaIdxY) * (int32_t{16}))]};
        mRoutedIndices[int32_t{2}] =
          int32_t{params.ptrRouteMap[((mWarpGrpWarpIdx) * (int32_t{4}) + (int32_t{2})) +
                                     ((mCtaIdxY) * (int32_t{16}))]};
        mRoutedIndices[int32_t{3}] =
          int32_t{params.ptrRouteMap[((mWarpGrpWarpIdx) * (int32_t{4}) + (int32_t{3})) +
                                     ((mCtaIdxY) * (int32_t{16}))]};
      }
    }
  }
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{6})) && ((state.mWarpIdx) < (int32_t{10}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 SmemBSmem& smemBDstSmem,
                                 SmemBStack& smemBDstStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<48>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
      6,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemBProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    int32_t smemBProdToken{int32_t{1}};
    int32_t paddedPerCtaK{(((params.k) + (int32_t{255})) / (int32_t{256})) * (int32_t{256})};
    int32_t loopEnd{paddedPerCtaK};
    bool const hasOneLoopIter{(int32_t{0}) < (loopEnd)};
    //
    // Hoist the first iter.
    //
    //
    // Loop body.
    //
    CUTLASS_PRAGMA_NO_UNROLL
    for (int32_t loopOffset250 = int32_t{0}; loopOffset250 < loopEnd;
         loopOffset250 += int32_t{256}) {
      bool const isLastLoopIter{((loopOffset250) + (int32_t{256})) >= (loopEnd)};
      //
      // gmemB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK2;
      { tileOffsetK2 = loopOffset250; }
      //
      // smemB [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      {
        if ((loopOffset250) >= (int32_t{1536})) {
          smemBProdToken = smemBDstStack.mPipeline.producer_try_acquire(smemBProdState);
        }
      }
      { smemBDstStack.mPipeline.producer_acquire(smemBProdState, smemBProdToken); }
      //
      // smemB [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      int32_t tileOffsetK4;
      tileOffsetK4 = tileOffsetK2;
      {
        uint64_t* barrier{smemBDstStack.mPipeline.producer_get_barrier(smemBProdState)};
        int32_t index{smemBProdState.index()};
        {
          int8_t* smemBytesBasePtrB;
          int8_t* smemBytesStagePtrB;
          smemBytesBasePtrB =
            reinterpret_cast<int8_t*>(smemBDstStack.mDepSmemPtr0) + (int32_t{196608});
          smemBytesStagePtrB = smemBytesBasePtrB + ((index) * (int32_t{4096}));
          {
            int32_t coords[5];
            coords[int32_t{0}] = tileOffsetK4;
            coords[int32_t{1}] = mRoutedIndices[int32_t{0}];
            coords[int32_t{2}] = mRoutedIndices[int32_t{1}];
            coords[int32_t{3}] = mRoutedIndices[int32_t{2}];
            coords[int32_t{4}] = mRoutedIndices[int32_t{3}];
            if ((bool{cute::elect_one_sync()}) &&
                (((mWarpGrpWarpIdx) * (int32_t{4})) < (int32_t{16}))) {
              cuda_ptx::cp_async_bulk_tensor_tile_gather4(
                cuda_ptx::space_shared_t{},
                cuda_ptx::space_global_t{},
                &reinterpret_cast<cutlass::float_e4m3_t*>(
                  smemBytesStagePtrB)[((mWarpGrpWarpIdx) * (int32_t{4})) * (int32_t{128})],
                params.tmaB,
                coords,
                barrier);
            }
            coords[int32_t{0}] += int32_t{128};
            if ((bool{cute::elect_one_sync()}) &&
                (((mWarpGrpWarpIdx) * (int32_t{4})) < (int32_t{16}))) {
              cuda_ptx::cp_async_bulk_tensor_tile_gather4(
                cuda_ptx::space_shared_t{},
                cuda_ptx::space_global_t{},
                &reinterpret_cast<cutlass::float_e4m3_t*>(
                  smemBytesStagePtrB)[(((mWarpGrpWarpIdx) * (int32_t{4})) * (int32_t{128})) +
                                      (int32_t{2048})],
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
      if ((loopOffset250) >= (int32_t{0})) {
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
    return ((state.mWarpIdx) >= (int32_t{4})) && ((state.mWarpIdx) < (int32_t{5}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 Mma0Stack& mma0DstStack,
                                 SmemASmem& smemASrcSmem,
                                 SmemAStack& smemASrcStack,
                                 SmemBSmem& smemBSrcSmem,
                                 SmemBStack& smemBSrcStack) {
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
    trtllm::dev::CutlassUmmaAsyncPipeline<1,
                                          cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState mma0ProdState{int32_t{0}, int32_t{1}, int32_t{0}};
    int32_t mma0ProdToken{int32_t{1}};
    int32_t paddedPerCtaK{(((params.k) + (int32_t{255})) / (int32_t{256})) * (int32_t{256})};
    int32_t loopEnd{paddedPerCtaK};
    //
    // Loop body.
    //
    CUTLASS_PRAGMA_NO_UNROLL
    for (int32_t loopOffset342 = int32_t{0}; loopOffset342 < loopEnd;
         loopOffset342 += int32_t{256}) {
      bool const isFirstLoopIter{(loopOffset342) == (int32_t{0})};
      bool const isLastLoopIter{((loopOffset342) + (int32_t{256})) >= (loopEnd)};
      //
      // mma0 [ProdTryAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if (isFirstLoopIter) {
        if ((loopOffset342) >= (int32_t{0})) {
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
      cutlass::float_e4m3_t* smemPtrA3;
      {
        int32_t index{smemAConsState.index()};
        int8_t* smemBytesBasePtrA;
        smemBytesBasePtrA = reinterpret_cast<int8_t*>(smemASrcStack.mDepSmemPtr0) + (int32_t{0});
        int8_t* smemBytesStagePtrA;
        smemBytesStagePtrA = smemBytesBasePtrA + ((index) * (int32_t{32768}));
        smemPtrA3 = reinterpret_cast<cutlass::float_e4m3_t*>(smemBytesStagePtrA) + (int32_t{0});
        ++smemAConsState;
      }
      //
      // smemB [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      cutlass::float_e4m3_t* smemPtrB4;
      {
        int32_t index{smemBConsState.index()};
        int8_t* smemBytesBasePtrB;
        smemBytesBasePtrB =
          reinterpret_cast<int8_t*>(smemBSrcStack.mDepSmemPtr0) + (int32_t{196608});
        int8_t* smemBytesStagePtrB;
        smemBytesStagePtrB = smemBytesBasePtrB + ((index) * (int32_t{4096}));
        smemPtrB4 = reinterpret_cast<cutlass::float_e4m3_t*>(smemBytesStagePtrB) + (int32_t{0});
        ++smemBConsState;
      }
      //
      // mma0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      cutlass::float_e4m3_t* smemPtrA6;
      cutlass::float_e4m3_t* smemPtrB6;
      smemPtrA6 = smemPtrA3;
      smemPtrB6 = smemPtrB4;
      {
        int32_t index{mma0ProdState.index()};
        uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{16})))};
        uint32_t ptrTmemOffsetD{ptrTmemD};
        cutlass::float_e4m3_t* ptrWithOffsetSmemA{(smemPtrA6 + int32_t{0})};
        cutlass::float_e4m3_t* ptrWithOffsetSmemB{(smemPtrB6 + int32_t{0})};
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
          // leadingDimInBytes = 2048, strideInBytes = 1024, swizzleMode = 1.
          //
          uint64_t smemDescB{
            trtllm::dev::createSmemDesc(ptrWithOffsetSmemB,
                                        uint32_t{0x800000 /*hi=128, lo=0*/},
                                        uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
          //
          // MMA inst for mi=0 ni=0 ki=0.
          //
          uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
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
                                  bool{(loopOffset342) != (int32_t{0})});
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
                                                                  int32_t{128},
                                                                  int32_t{16},
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
                                                                  int32_t{128},
                                                                  int32_t{16},
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
                                                                  int32_t{128},
                                                                  int32_t{16},
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
          //
          // MMA inst for mi=0 ni=0 ki=4.
          //
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{1018});
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{122});
          uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{32},
                                                                  false,
                                                                  true)};
          if (bool{cute::elect_one_sync()}) {
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f8f6f4,
                                  cuda_ptx::cta_group_1,
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
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{32},
                                                                  false,
                                                                  true)};
          if (bool{cute::elect_one_sync()}) {
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f8f6f4,
                                  cuda_ptx::cta_group_1,
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
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{32},
                                                                  false,
                                                                  true)};
          if (bool{cute::elect_one_sync()}) {
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f8f6f4,
                                  cuda_ptx::cta_group_1,
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
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{32},
                                                                  false,
                                                                  true)};
          if (bool{cute::elect_one_sync()}) {
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f8f6f4,
                                  cuda_ptx::cta_group_1,
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
      if ((loopOffset342) >= (int32_t{0})) {
        {
          smemASrcStack.mPipeline.consumer_release(smemAConsReleaseState);
        }
        ++smemAConsReleaseState;
      }
      //
      // smemB [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      if ((loopOffset342) >= (int32_t{0})) {
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
        ++mma0ProdState;
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
  cutlass::Array<float, 16> frg7;
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
    , mGridDimX{reinterpret_cast<int32_t const&>(gridDim.x)} {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{0})) && ((state.mWarpIdx) < (int32_t{4}));
  }
  inline __device__ void execute(KernelParams const& params,
                                 KernelState const& state,
                                 GmemC0Smem& gmemC0DstSmem,
                                 GmemC0Stack& gmemC0DstStack,
                                 Mma0Stack& mma0SrcStack) {
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<160>{});
    cudaGridDependencySynchronize();
    trtllm::dev::CutlassUmmaAsyncPipeline<
      1,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState mma0ConsState{};
    int32_t mma0ConsToken{int32_t{0}};
    int32_t paddedPerCtaK{(((params.k) + (int32_t{255})) / (int32_t{256})) * (int32_t{256})};
    int32_t loopEnd{paddedPerCtaK};
    bool const hasOneLoopIter{(int32_t{0}) < (loopEnd)};
    int32_t lastLoopOffset{int32_t{0}};
    uint32_t tmemBaseWithStageOffset;
    tmemBaseWithStageOffset = mTmemBaseOffset;
    //
    // SmemPerTokenSfAb::createFctProdVars.
    //
    int8_t* ptrSmemBaseSmem;
    float* ptrSmemSmem;
    ptrSmemBaseSmem = reinterpret_cast<int8_t*>(gmemC0DstStack.mDepSmemPtr0) + (int32_t{223232});
    ptrSmemSmem = reinterpret_cast<float*>(ptrSmemBaseSmem) + (int32_t{0});
    //
    // Loading per-token SF to SMEM.
    //
    {
      float* ptrSmemB;
      ptrSmemB = reinterpret_cast<float*>(ptrSmemSmem) + (int32_t{0});
      if (bool{reinterpret_cast<cutlass::bfloat16_t const*>(params.ptrPerTokenSfB) != nullptr}) {
        if ((mWarpGrpThreadIdx) < (int32_t{16})) {
          int32_t offsetN{(mCtaIdxY) * (int32_t{16})};
          if (((mWarpGrpThreadIdx) + (offsetN)) < (mBatchLimit)) {
            ptrSmemB[mWarpGrpThreadIdx] = float{reinterpret_cast<cutlass::bfloat16_t const*>(
              params.ptrPerTokenSfB)[int32_t{params.ptrRouteMap[(mWarpGrpThreadIdx) + (offsetN)]}]};
          }
        }
      }
      if (bool{reinterpret_cast<cutlass::bfloat16_t const*>(params.ptrPerTokenSfB) != nullptr}) {
        trtllm::dev::CutlassNamedBarrier::sync(128, 5);
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
    for (int32_t loopOffset544 = int32_t{0}; loopOffset544 < loopEnd;
         loopOffset544 += int32_t{256}) {
      //
      // gmemC0 [ProdTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      //
      // mma0 [ConsTailRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      lastLoopOffset = loopOffset544;
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
    uint32_t tmemBaseWithStageOffset6;
    if (hasOneLoopIter) {
      int32_t index{mma0ConsState.index()};
      uint32_t ptrTmemD{(mTmemBaseOffset) + (static_cast<uint32_t>((index) * (int32_t{16})))};
      uint32_t ptrTmemOffsetD{ptrTmemD};
      tmemBaseWithStageOffset6 = ptrTmemOffsetD;
    }
    //
    // gmemC0 [ProdWork (call 0), LastIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    uint32_t tmemBaseWithStageOffset7;
    tmemBaseWithStageOffset7 = tmemBaseWithStageOffset6;
    if (hasOneLoopIter) {
      tmemBaseWithStageOffset = tmemBaseWithStageOffset7;
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
          uint32_t(&dstSlice0)[8]{reinterpret_cast<uint32_t(&)[8]>(frg7[int32_t{0}])};
          cuda_ptx::tcgen05_ld_16x256b(dstSlice0,
                                       (tmemBasePtr) +
                                         (static_cast<uint32_t>((mWarpGrp4Idx) * (int32_t{16}))));
          uint32_t(&dstSlice1)[8]{reinterpret_cast<uint32_t(&)[8]>(frg7[int32_t{8}])};
          cuda_ptx::tcgen05_ld_16x256b(
            dstSlice1,
            (tmemBasePtr) + (static_cast<uint32_t>(((mWarpGrp4Idx) * (int32_t{16})) +
                                                   (int32_t{0x100000 /*hi=16, lo=0*/}))));
        }
        cutlass::arch::fence_view_async_tmem_load();
        //
        // Compute per-token scaling.
        //
        if ((bool{reinterpret_cast<float const*>(params.ptrPerTokenSfA) != nullptr}) ||
            (bool{reinterpret_cast<float const*>(params.ptrPerTokenSfB) != nullptr})) {
          int32_t const warpRowIdx{(mWarpGrp4WarpIdx) * (int32_t{32})};
          int32_t const quadRowIdx{(mLaneIdx) / (int32_t{4})};
          int32_t const laneColIdx{((mLaneIdx) % (int32_t{4})) * (int32_t{2})};
          //
          // Compute per-token scaling (0, 0).
          //
          {
            int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
            int32_t const sharedColIdx{(laneColIdx) + ((mWarpGrp4Idx) * (int32_t{16}))};
            //
            // Loading per-token SF to register.
            //
            float* ptrSmemB;
            ptrSmemB = reinterpret_cast<float*>(ptrSmemSmem) + (int32_t{0});
            frg7[int32_t{0}] = (frg7[int32_t{0}]) * (float{ptrSmemB[sharedColIdx]});
          }
          //
          // Compute per-token scaling (0, 1).
          //
          {
            int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{16}) + (int32_t{1}))};
            //
            // Loading per-token SF to register.
            //
            float* ptrSmemB;
            ptrSmemB = reinterpret_cast<float*>(ptrSmemSmem) + (int32_t{0});
            frg7[int32_t{1}] = (frg7[int32_t{1}]) * (float{ptrSmemB[sharedColIdx]});
          }
          //
          // Compute per-token scaling (0, 2).
          //
          {
            int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{16}) + (int32_t{8}))};
            //
            // Loading per-token SF to register.
            //
            float* ptrSmemB;
            ptrSmemB = reinterpret_cast<float*>(ptrSmemSmem) + (int32_t{0});
            frg7[int32_t{4}] = (frg7[int32_t{4}]) * (float{ptrSmemB[sharedColIdx]});
          }
          //
          // Compute per-token scaling (0, 3).
          //
          {
            int32_t const sharedRowIdx{(warpRowIdx) + (quadRowIdx)};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{16}) + (int32_t{9}))};
            //
            // Loading per-token SF to register.
            //
            float* ptrSmemB;
            ptrSmemB = reinterpret_cast<float*>(ptrSmemSmem) + (int32_t{0});
            frg7[int32_t{5}] = (frg7[int32_t{5}]) * (float{ptrSmemB[sharedColIdx]});
          }
          //
          // Compute per-token scaling (1, 0).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
            int32_t const sharedColIdx{(laneColIdx) + ((mWarpGrp4Idx) * (int32_t{16}))};
            //
            // Loading per-token SF to register.
            //
            float* ptrSmemB;
            ptrSmemB = reinterpret_cast<float*>(ptrSmemSmem) + (int32_t{0});
            frg7[int32_t{2}] = (frg7[int32_t{2}]) * (float{ptrSmemB[sharedColIdx]});
          }
          //
          // Compute per-token scaling (1, 1).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{16}) + (int32_t{1}))};
            //
            // Loading per-token SF to register.
            //
            float* ptrSmemB;
            ptrSmemB = reinterpret_cast<float*>(ptrSmemSmem) + (int32_t{0});
            frg7[int32_t{3}] = (frg7[int32_t{3}]) * (float{ptrSmemB[sharedColIdx]});
          }
          //
          // Compute per-token scaling (1, 2).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{16}) + (int32_t{8}))};
            //
            // Loading per-token SF to register.
            //
            float* ptrSmemB;
            ptrSmemB = reinterpret_cast<float*>(ptrSmemSmem) + (int32_t{0});
            frg7[int32_t{6}] = (frg7[int32_t{6}]) * (float{ptrSmemB[sharedColIdx]});
          }
          //
          // Compute per-token scaling (1, 3).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{8})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{16}) + (int32_t{9}))};
            //
            // Loading per-token SF to register.
            //
            float* ptrSmemB;
            ptrSmemB = reinterpret_cast<float*>(ptrSmemSmem) + (int32_t{0});
            frg7[int32_t{7}] = (frg7[int32_t{7}]) * (float{ptrSmemB[sharedColIdx]});
          }
          //
          // Compute per-token scaling (2, 0).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
            int32_t const sharedColIdx{(laneColIdx) + ((mWarpGrp4Idx) * (int32_t{16}))};
            //
            // Loading per-token SF to register.
            //
            float* ptrSmemB;
            ptrSmemB = reinterpret_cast<float*>(ptrSmemSmem) + (int32_t{0});
            frg7[int32_t{8}] = (frg7[int32_t{8}]) * (float{ptrSmemB[sharedColIdx]});
          }
          //
          // Compute per-token scaling (2, 1).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{16}) + (int32_t{1}))};
            //
            // Loading per-token SF to register.
            //
            float* ptrSmemB;
            ptrSmemB = reinterpret_cast<float*>(ptrSmemSmem) + (int32_t{0});
            frg7[int32_t{9}] = (frg7[int32_t{9}]) * (float{ptrSmemB[sharedColIdx]});
          }
          //
          // Compute per-token scaling (2, 2).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{16}) + (int32_t{8}))};
            //
            // Loading per-token SF to register.
            //
            float* ptrSmemB;
            ptrSmemB = reinterpret_cast<float*>(ptrSmemSmem) + (int32_t{0});
            frg7[int32_t{12}] = (frg7[int32_t{12}]) * (float{ptrSmemB[sharedColIdx]});
          }
          //
          // Compute per-token scaling (2, 3).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{16})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{16}) + (int32_t{9}))};
            //
            // Loading per-token SF to register.
            //
            float* ptrSmemB;
            ptrSmemB = reinterpret_cast<float*>(ptrSmemSmem) + (int32_t{0});
            frg7[int32_t{13}] = (frg7[int32_t{13}]) * (float{ptrSmemB[sharedColIdx]});
          }
          //
          // Compute per-token scaling (3, 0).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
            int32_t const sharedColIdx{(laneColIdx) + ((mWarpGrp4Idx) * (int32_t{16}))};
            //
            // Loading per-token SF to register.
            //
            float* ptrSmemB;
            ptrSmemB = reinterpret_cast<float*>(ptrSmemSmem) + (int32_t{0});
            frg7[int32_t{10}] = (frg7[int32_t{10}]) * (float{ptrSmemB[sharedColIdx]});
          }
          //
          // Compute per-token scaling (3, 1).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{16}) + (int32_t{1}))};
            //
            // Loading per-token SF to register.
            //
            float* ptrSmemB;
            ptrSmemB = reinterpret_cast<float*>(ptrSmemSmem) + (int32_t{0});
            frg7[int32_t{11}] = (frg7[int32_t{11}]) * (float{ptrSmemB[sharedColIdx]});
          }
          //
          // Compute per-token scaling (3, 2).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{16}) + (int32_t{8}))};
            //
            // Loading per-token SF to register.
            //
            float* ptrSmemB;
            ptrSmemB = reinterpret_cast<float*>(ptrSmemSmem) + (int32_t{0});
            frg7[int32_t{14}] = (frg7[int32_t{14}]) * (float{ptrSmemB[sharedColIdx]});
          }
          //
          // Compute per-token scaling (3, 3).
          //
          {
            int32_t const sharedRowIdx{((warpRowIdx) + (quadRowIdx)) + (int32_t{24})};
            int32_t const sharedColIdx{(laneColIdx) +
                                       ((mWarpGrp4Idx) * (int32_t{16}) + (int32_t{9}))};
            //
            // Loading per-token SF to register.
            //
            float* ptrSmemB;
            ptrSmemB = reinterpret_cast<float*>(ptrSmemSmem) + (int32_t{0});
            frg7[int32_t{15}] = (frg7[int32_t{15}]) * (float{ptrSmemB[sharedColIdx]});
          }
        }
        cuda_ptx::cp_async_bulk_wait_group_read(cuda_ptx::n32_t<0>{});
        trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{4}) + (mWarpGrp4Idx));
        //
        // Store to Smem TmaAsyncGmemGatedAct.
        //
        int8_t* ptrSmemBase;
        cutlass::float_e4m3_t* ptrSmem;
        ptrSmemBase = reinterpret_cast<int8_t*>(gmemC0DstStack.mDepSmemPtr0) +
                      ((mWarpGrp4Idx) * (int32_t{1024}) + (int32_t{221184}));
        ptrSmem = reinterpret_cast<cutlass::float_e4m3_t*>(ptrSmemBase) + (int32_t{0});
        {
          int32_t smemOffset0;
          {
            int32_t const smemRowIdx{
              ((mBaseTmemCol) * (int32_t{64}) + (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
              (int32_t{128})};
            int32_t const smemOffsetInBytes{
              (((mBaseTmemCol) * (int32_t{64}) + (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
               (int32_t{8})) /
              (int32_t{8})};
            int32_t const swizzleMask{((smemRowIdx) % (int32_t{4})) * (int32_t{16})};
            smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{8});
          }
          cutlass::Array<float, 2> scaleF2{mScaleC, mScaleC};
          {
            frg7[int32_t{0}] =
              float{trtllm::dev::clamp(frg7[int32_t{0}], -(mClampLimit), mClampLimit)};
            frg7[int32_t{2}] =
              float{trtllm::dev::clamp(frg7[int32_t{2}], float{-3.4028235e+38}, mClampLimit)};
            frg7[int32_t{8}] =
              float{trtllm::dev::clamp(frg7[int32_t{8}], -(mClampLimit), mClampLimit)};
            frg7[int32_t{10}] =
              float{trtllm::dev::clamp(frg7[int32_t{10}], float{-3.4028235e+38}, mClampLimit)};
            frg7[int32_t{0}] = (frg7[int32_t{0}]) * (float{1});
            frg7[int32_t{8}] = (frg7[int32_t{8}]) * (float{1});
            cutlass::Array<float, 2> x0Array{frg7[int32_t{0}], frg7[int32_t{8}]};
            cutlass::Array<float, 2> x1Array{frg7[int32_t{2}], frg7[int32_t{10}]};
            cutlass::Array<float, 2> fusedScaleArray{
              ((mGatedActAlpha) * (float{1.442695})) * (mScaleGate),
              ((mGatedActAlpha) * (float{1.442695})) * (mScaleGate)};
            cutlass::Array<float, 2> betaScaleGateArray{(mGatedActBeta) * (mScaleGate),
                                                        (mGatedActBeta) * (mScaleGate)};
            cutlass::Array<float, 2> scaleGateArray{mScaleGate, mScaleGate};
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
            frg7[int32_t{0}] = gatedActArray[int32_t{0}];
            frg7[int32_t{8}] = gatedActArray[int32_t{1}];
          }
          cutlass::Array<float, 2> accF2{frg7[int32_t{0}], frg7[int32_t{8}]};
          cutlass::Array<float, 2> scaledAccF2{trtllm::dev::fmul2(accF2, scaleF2)};
          cutlass::Array<cutlass::float_e4m3_t, 2> scaledCvtAcc2{
            trtllm::dev::convert_float2_to_e4m3(scaledAccF2)};
          {
            uint16_t convertedElts;
            convertedElts = reinterpret_cast<uint16_t&>(scaledCvtAcc2);
            reinterpret_cast<uint16_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
          }
        }
        {
          int32_t smemOffset0;
          {
            int32_t const smemRowIdx{(((mBaseTmemCol) + (int32_t{1})) * (int32_t{64}) +
                                      (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
                                     (int32_t{128})};
            int32_t const smemOffsetInBytes{((((mBaseTmemCol) + (int32_t{1})) * (int32_t{64}) +
                                              (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
                                             (int32_t{8})) /
                                            (int32_t{8})};
            int32_t const swizzleMask{((smemRowIdx) % (int32_t{4})) * (int32_t{16})};
            smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{8});
          }
          cutlass::Array<float, 2> scaleF2{mScaleC, mScaleC};
          {
            frg7[int32_t{1}] =
              float{trtllm::dev::clamp(frg7[int32_t{1}], -(mClampLimit), mClampLimit)};
            frg7[int32_t{3}] =
              float{trtllm::dev::clamp(frg7[int32_t{3}], float{-3.4028235e+38}, mClampLimit)};
            frg7[int32_t{9}] =
              float{trtllm::dev::clamp(frg7[int32_t{9}], -(mClampLimit), mClampLimit)};
            frg7[int32_t{11}] =
              float{trtllm::dev::clamp(frg7[int32_t{11}], float{-3.4028235e+38}, mClampLimit)};
            frg7[int32_t{1}] = (frg7[int32_t{1}]) * (float{1});
            frg7[int32_t{9}] = (frg7[int32_t{9}]) * (float{1});
            cutlass::Array<float, 2> x0Array{frg7[int32_t{1}], frg7[int32_t{9}]};
            cutlass::Array<float, 2> x1Array{frg7[int32_t{3}], frg7[int32_t{11}]};
            cutlass::Array<float, 2> fusedScaleArray{
              ((mGatedActAlpha) * (float{1.442695})) * (mScaleGate),
              ((mGatedActAlpha) * (float{1.442695})) * (mScaleGate)};
            cutlass::Array<float, 2> betaScaleGateArray{(mGatedActBeta) * (mScaleGate),
                                                        (mGatedActBeta) * (mScaleGate)};
            cutlass::Array<float, 2> scaleGateArray{mScaleGate, mScaleGate};
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
            frg7[int32_t{1}] = gatedActArray[int32_t{0}];
            frg7[int32_t{9}] = gatedActArray[int32_t{1}];
          }
          cutlass::Array<float, 2> accF2{frg7[int32_t{1}], frg7[int32_t{9}]};
          cutlass::Array<float, 2> scaledAccF2{trtllm::dev::fmul2(accF2, scaleF2)};
          cutlass::Array<cutlass::float_e4m3_t, 2> scaledCvtAcc2{
            trtllm::dev::convert_float2_to_e4m3(scaledAccF2)};
          {
            uint16_t convertedElts;
            convertedElts = reinterpret_cast<uint16_t&>(scaledCvtAcc2);
            reinterpret_cast<uint16_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
          }
        }
        {
          int32_t smemOffset0;
          {
            int32_t const smemRowIdx{(((mBaseTmemCol) + (int32_t{8})) * (int32_t{64}) +
                                      (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
                                     (int32_t{128})};
            int32_t const smemOffsetInBytes{((((mBaseTmemCol) + (int32_t{8})) * (int32_t{64}) +
                                              (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
                                             (int32_t{8})) /
                                            (int32_t{8})};
            int32_t const swizzleMask{((smemRowIdx) % (int32_t{4})) * (int32_t{16})};
            smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{8});
          }
          cutlass::Array<float, 2> scaleF2{mScaleC, mScaleC};
          {
            frg7[int32_t{4}] =
              float{trtllm::dev::clamp(frg7[int32_t{4}], -(mClampLimit), mClampLimit)};
            frg7[int32_t{6}] =
              float{trtllm::dev::clamp(frg7[int32_t{6}], float{-3.4028235e+38}, mClampLimit)};
            frg7[int32_t{12}] =
              float{trtllm::dev::clamp(frg7[int32_t{12}], -(mClampLimit), mClampLimit)};
            frg7[int32_t{14}] =
              float{trtllm::dev::clamp(frg7[int32_t{14}], float{-3.4028235e+38}, mClampLimit)};
            frg7[int32_t{4}] = (frg7[int32_t{4}]) * (float{1});
            frg7[int32_t{12}] = (frg7[int32_t{12}]) * (float{1});
            cutlass::Array<float, 2> x0Array{frg7[int32_t{4}], frg7[int32_t{12}]};
            cutlass::Array<float, 2> x1Array{frg7[int32_t{6}], frg7[int32_t{14}]};
            cutlass::Array<float, 2> fusedScaleArray{
              ((mGatedActAlpha) * (float{1.442695})) * (mScaleGate),
              ((mGatedActAlpha) * (float{1.442695})) * (mScaleGate)};
            cutlass::Array<float, 2> betaScaleGateArray{(mGatedActBeta) * (mScaleGate),
                                                        (mGatedActBeta) * (mScaleGate)};
            cutlass::Array<float, 2> scaleGateArray{mScaleGate, mScaleGate};
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
            frg7[int32_t{4}] = gatedActArray[int32_t{0}];
            frg7[int32_t{12}] = gatedActArray[int32_t{1}];
          }
          cutlass::Array<float, 2> accF2{frg7[int32_t{4}], frg7[int32_t{12}]};
          cutlass::Array<float, 2> scaledAccF2{trtllm::dev::fmul2(accF2, scaleF2)};
          cutlass::Array<cutlass::float_e4m3_t, 2> scaledCvtAcc2{
            trtllm::dev::convert_float2_to_e4m3(scaledAccF2)};
          {
            uint16_t convertedElts;
            convertedElts = reinterpret_cast<uint16_t&>(scaledCvtAcc2);
            reinterpret_cast<uint16_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
          }
        }
        {
          int32_t smemOffset0;
          {
            int32_t const smemRowIdx{(((mBaseTmemCol) + (int32_t{9})) * (int32_t{64}) +
                                      (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) /
                                     (int32_t{128})};
            int32_t const smemOffsetInBytes{((((mBaseTmemCol) + (int32_t{9})) * (int32_t{64}) +
                                              (((mWarpRowIdx) / (int32_t{2})) + (mQuadRowIdx))) *
                                             (int32_t{8})) /
                                            (int32_t{8})};
            int32_t const swizzleMask{((smemRowIdx) % (int32_t{4})) * (int32_t{16})};
            smemOffset0 = (((smemOffsetInBytes) ^ (swizzleMask)) * (int32_t{8})) / (int32_t{8});
          }
          cutlass::Array<float, 2> scaleF2{mScaleC, mScaleC};
          {
            frg7[int32_t{5}] =
              float{trtllm::dev::clamp(frg7[int32_t{5}], -(mClampLimit), mClampLimit)};
            frg7[int32_t{7}] =
              float{trtllm::dev::clamp(frg7[int32_t{7}], float{-3.4028235e+38}, mClampLimit)};
            frg7[int32_t{13}] =
              float{trtllm::dev::clamp(frg7[int32_t{13}], -(mClampLimit), mClampLimit)};
            frg7[int32_t{15}] =
              float{trtllm::dev::clamp(frg7[int32_t{15}], float{-3.4028235e+38}, mClampLimit)};
            frg7[int32_t{5}] = (frg7[int32_t{5}]) * (float{1});
            frg7[int32_t{13}] = (frg7[int32_t{13}]) * (float{1});
            cutlass::Array<float, 2> x0Array{frg7[int32_t{5}], frg7[int32_t{13}]};
            cutlass::Array<float, 2> x1Array{frg7[int32_t{7}], frg7[int32_t{15}]};
            cutlass::Array<float, 2> fusedScaleArray{
              ((mGatedActAlpha) * (float{1.442695})) * (mScaleGate),
              ((mGatedActAlpha) * (float{1.442695})) * (mScaleGate)};
            cutlass::Array<float, 2> betaScaleGateArray{(mGatedActBeta) * (mScaleGate),
                                                        (mGatedActBeta) * (mScaleGate)};
            cutlass::Array<float, 2> scaleGateArray{mScaleGate, mScaleGate};
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
            frg7[int32_t{5}] = gatedActArray[int32_t{0}];
            frg7[int32_t{13}] = gatedActArray[int32_t{1}];
          }
          cutlass::Array<float, 2> accF2{frg7[int32_t{5}], frg7[int32_t{13}]};
          cutlass::Array<float, 2> scaledAccF2{trtllm::dev::fmul2(accF2, scaleF2)};
          cutlass::Array<cutlass::float_e4m3_t, 2> scaledCvtAcc2{
            trtllm::dev::convert_float2_to_e4m3(scaledAccF2)};
          {
            uint16_t convertedElts;
            convertedElts = reinterpret_cast<uint16_t&>(scaledCvtAcc2);
            reinterpret_cast<uint16_t*>(ptrSmem)[(smemOffset0) / (int32_t{2})] = convertedElts;
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
          cutlass::float_e4m3_t* ptrSmem;
          ptrSmemBase = reinterpret_cast<int8_t*>(gmemC0DstStack.mDepSmemPtr0) +
                        ((mWarpGrp4Idx) * (int32_t{1024}) + (int32_t{221184}));
          ptrSmem = reinterpret_cast<cutlass::float_e4m3_t*>(ptrSmemBase) + (int32_t{0});
          int32_t coords[4];
          coords[int32_t{0}] = (mCtaIdxX) * (int32_t{64});
          coords[int32_t{1}] = (((int32_t{16}) - ((mBatchLimit) % (int32_t{16}))) % (int32_t{16})) +
                               ((mWarpGrp4Idx) * (int32_t{16}));
          coords[int32_t{2}] = int32_t{0x40000000 /*1073741824*/};
          coords[int32_t{3}] =
            (((mCtaIdxY) * (int32_t{16})) +
             ((int32_t{0}) - (((int32_t{16}) - ((mBatchLimit) % (int32_t{16}))) % (int32_t{16})))) +
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
      ++mma0ConsState;
    }
  ExitTileWithoutSignalingLabel: {}
  }
};
struct PaddingTask {
  inline __device__ PaddingTask(KernelParams const& params,
                                KernelState const& state,
                                int32_t warpGrpStart) {}
  static inline __device__ bool isSelected(KernelParams const& params, KernelState const& state) {
    return ((state.mWarpIdx) >= (int32_t{10})) && ((state.mWarpIdx) < (int32_t{12}));
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
__launch_bounds__(384, 1) void bmm_E4m3_E4m3E4m3_Fp32_t128x16x256_s6_et128x16_m128x16x32_cga1x1x1_16dp256b_rM_TN_transOut_tokSfB_schedS_bN_tma_tmaSf_rgTma_clmp_swiGlu_dynB_sm100f(
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
  uint32_t* TmemSwStatePtr{
    reinterpret_cast<uint32_t*>((reinterpret_cast<uint8_t*>(smem__) + smemOffset__))};
  smemOffset__ += int32_t{16};
  uint8_t* smemASmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemASmemBarrier)});
  uint8_t* smemBSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemBSmemBarrier)});
  uint8_t* mma0SmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(Mma0SmemBarrier)});
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
  LoadTaskA loadTaskA{params, state, int32_t{5}};
  LoadTaskB loadTaskB{params, state, int32_t{6}};
  cutlass::arch::fence_barrier_init();
  __syncthreads();
  if ((reinterpret_cast<int32_t const&>(threadIdx.x)) < (int32_t{32})) {
    cuda_ptx::tcgen05_alloc(cuda_ptx::cta_group_1_t{}, state.mTmemSwStatePtr, int32_t{32});
    cuda_ptx::tcgen05_relinquish_alloc_permit(cuda_ptx::cta_group_1_t{});
  }
  if ((bool{LoadTaskA::isSelected(params, state)}) ||
      (bool{LoadTaskB::isSelected(params, state)})) {
  } else {
    trtllm::dev::CutlassNamedBarrier::sync(224, 6);
  }
  if (bool{LoadTaskA::isSelected(params, state)}) {
    loadTaskA.execute(params, state, (*smemASmem), smemAStack);
  } else {
    if (bool{LoadTaskB::isSelected(params, state)}) {
      loadTaskB.execute(params, state, (*smemBSmem), smemBStack);
    } else {
      if (bool{MmaTask0::isSelected(params, state)}) {
        MmaTask0 mmaTask0{params, state, int32_t{4}};
        mmaTask0
          .execute(params, state, mma0Stack, (*smemASmem), smemAStack, (*smemBSmem), smemBStack);
      } else {
        if (bool{EpilogueTask0::isSelected(params, state)}) {
          EpilogueTask0 epilogueTask0{params, state, int32_t{0}};
          epilogueTask0.execute(params, state, (*gmemC0Smem), gmemC0Stack, mma0Stack);
          trtllm::dev::CutlassNamedBarrier::sync(128, 7);
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
          if (bool{PaddingTask::isSelected(params, state)}) {
            PaddingTask paddingTask{params, state, int32_t{10}};
            paddingTask.execute(params, state);
          }
        }
      }
    }
  }
}
extern "C" __global__ void
bmm_E4m3_E4m3E4m3_Fp32_t128x16x256_s6_et128x16_m128x16x32_cga1x1x1_16dp256b_rM_TN_transOut_tokSfB_schedS_bN_tma_tmaSf_rgTma_clmp_swiGlu_dynB_sm100fGetSmemSize(
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
  size += 16;
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(SmemASmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(SmemBSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(Mma0SmemBarrier));
  outPtr[0] = size;
}

} // namespace batchedGemm
