#include <FmhaSm100fKernel_QFp16KvE4m3OFp16H256PagedKvDenseP64MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen.h>

// Res.cpp:137
// Fmha.h:845
struct SmemQStack {
  // Res.cpp:595
  trtllm::dev::CutlassTmaUmmaAsyncPipeline<1, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  // Res.cpp:208
  inline __device__ SmemQStack(SmemQSmem& smemQSmem,
                               SmemQSmemBarrier& smemQSmemBarrier,
                               int32_t warpId,
                               int32_t clusterDimX,
                               int32_t clusterDimY,
                               int32_t barInitWarpId,
                               int32_t orderedSequenceGroupId)
    : // Res.cpp:714
    mPipeline{smemQSmemBarrier.mBarriers,
              warpId,
              int32_t{8192},
              bool{cute::elect_one_sync()},
              CuteFlatTuple236{},
              cute::true_type{},
              cute::true_type{},
              barInitWarpId} {}
};
// Res.cpp:137
// Fmha.h:1117
struct SmemKvStack {
  // Res.cpp:595
  trtllm::dev::CutlassTmaAsyncPipeline<4> mPipeline;
  // MemBuffers.cpp:275
  cutlass::float_e4m3_t* mPtr;
  // Res.cpp:208
  inline __device__ SmemKvStack(SmemKvSmem& smemKvSmem,
                                SmemKvSmemBarrier& smemKvSmemBarrier,
                                int32_t warpId,
                                int32_t clusterDimX,
                                int32_t clusterDimY,
                                int32_t barInitWarpId,
                                int32_t orderedSequenceGroupId)
    : // Res.cpp:683
    mPipeline{smemKvSmemBarrier.mBarriers,
              warpId,
              int32_t{16384},
              ((warpId) == (barInitWarpId)) && (bool{cute::elect_one_sync()}),
              int32_t{128},
              CuteFlatTuple352{},
              cute::true_type{},
              cute::true_type{},
              barInitWarpId}
    , // MemBuffers.cpp:282
    mPtr{&smemKvSmem.mArray[int32_t{0}][int32_t{0}]} {}
};
// Res.cpp:137
// Fmha.h:1163
struct SmemTransformedKvStack {
  // Res.cpp:595
  trtllm::dev::CutlassUmmaConsumerAsyncPipeline<2, false, false> mPipeline;
  // MemBuffers.cpp:275
  cutlass::half_t* mPtr;
  // Res.cpp:208
  inline __device__ SmemTransformedKvStack(
    SmemTransformedKvSmem& smemTransformedKvSmem,
    SmemTransformedKvSmemBarrier& smemTransformedKvSmemBarrier,
    int32_t warpId,
    int32_t clusterDimX,
    int32_t clusterDimY,
    int32_t barInitWarpId,
    int32_t orderedSequenceGroupId)
    : // Res.cpp:787
    mPipeline{smemTransformedKvSmemBarrier.mBarriers,
              warpId,
              int32_t{128},
              CuteFlatTuple476{},
              cute::true_type{},
              cute::true_type{},
              barInitWarpId}
    , // MemBuffers.cpp:282
    mPtr{&smemTransformedKvSmem.mArray[int32_t{0}][int32_t{0}]} {}
};
// Res.cpp:137
// Fmha.h:1217
struct SmemPageOffsetsKvStack {
  // Res.cpp:595
  trtllm::dev::CutlassCpAsyncPipeline<6> mPipeline;
  // Res.cpp:208
  inline __device__ SmemPageOffsetsKvStack(
    SmemPageOffsetsKvSmem& smemPageOffsetsKvSmem,
    SmemPageOffsetsKvSmemBarrier& smemPageOffsetsKvSmemBarrier,
    int32_t warpId,
    int32_t clusterDimX,
    int32_t clusterDimY,
    int32_t barInitWarpId,
    int32_t orderedSequenceGroupId)
    : // Res.cpp:644
    mPipeline{smemPageOffsetsKvSmemBarrier.mBarriers,
              warpId,
              int32_t{32},
              int32_t{32},
              barInitWarpId} {}
};
// Res.cpp:137
// Fmha.h:1281
struct SmemPOStack {
  // Res.cpp:208
  inline __device__ SmemPOStack(SmemPOSmem& smemPOSmem,
                                int32_t warpId,
                                int32_t clusterDimX,
                                int32_t clusterDimY,
                                int32_t barInitWarpId,
                                int32_t orderedSequenceGroupId) {}
};
// Res.cpp:137
// Fmha.h:1304
struct SmemSoftmaxWarpGrpRed0Stack {
  // Res.cpp:208
  inline __device__ SmemSoftmaxWarpGrpRed0Stack(
    SmemSoftmaxWarpGrpRed0Smem& smemSoftmaxWarpGrpRed0Smem,
    int32_t warpId,
    int32_t clusterDimX,
    int32_t clusterDimY,
    int32_t barInitWarpId,
    int32_t orderedSequenceGroupId) {}
};
// Res.cpp:137
// Fmha.h:1334
struct SmemCorrWarpGrpRed1Stack {
  // Res.cpp:208
  inline __device__ SmemCorrWarpGrpRed1Stack(SmemCorrWarpGrpRed1Smem& smemCorrWarpGrpRed1Smem,
                                             int32_t warpId,
                                             int32_t clusterDimX,
                                             int32_t clusterDimY,
                                             int32_t barInitWarpId,
                                             int32_t orderedSequenceGroupId) {}
};
// Res.cpp:137
// Fmha.h:1866
struct TmemS0Stack {
  // MemBuffers.cpp:488
  float* mDepSmemPtr7;
  // Res.cpp:595
  trtllm::dev::CutlassUmmaAsyncPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  // TmemS.h:542
  int32_t const mNamedBarId;
  // TmemS.h:545
  int32_t const mInstId;
  // Res.cpp:208
  inline __device__ TmemS0Stack(TmemS0SmemBarrier& tmemS0SmemBarrier,
                                SmemSoftmaxWarpGrpRed0Smem& smemSoftmaxWarpGrpRed0Smem,
                                SmemSoftmaxWarpGrpRed0Stack& smemSoftmaxWarpGrpRed0Stack,
                                int32_t warpId,
                                int32_t clusterDimX,
                                int32_t clusterDimY,
                                int32_t barInitWarpId,
                                int32_t orderedSequenceGroupId,
                                int32_t namedBarId,
                                int32_t instId)
    : // MemBuffers.cpp:501
    mDepSmemPtr7{&smemSoftmaxWarpGrpRed0Smem.mArray[int32_t{0}]}
    , // Res.cpp:771
    mPipeline{tmemS0SmemBarrier.mBarriers,
              warpId,
              int32_t{128},
              CuteFlatTuple743{},
              cute::true_type{},
              cute::true_type{},
              barInitWarpId}
    , // TmemS.h:533
    mNamedBarId{namedBarId}
    , // TmemS.h:535
    mInstId{instId} {}
};
// Res.cpp:137
// SoftmaxSchedule.h:242
struct TmemSoftmaxLocal0Stack {
  // Res.cpp:595
  trtllm::dev::CutlassCpAsyncPipeline<2, true> mPipeline;
  // TmemSoftmax.h:194
  int32_t const mInstId;
  // Res.cpp:208
  inline __device__ TmemSoftmaxLocal0Stack(
    TmemSoftmaxLocal0SmemBarrier& tmemSoftmaxLocal0SmemBarrier,
    int32_t warpId,
    int32_t clusterDimX,
    int32_t clusterDimY,
    int32_t barInitWarpId,
    int32_t orderedSequenceGroupId,
    int32_t instId)
    : // Res.cpp:644
    mPipeline{tmemSoftmaxLocal0SmemBarrier.mBarriers,
              warpId,
              int32_t{128},
              int32_t{128},
              barInitWarpId}
    , // TmemSoftmax.h:187
    mInstId{instId} {}
};
// Res.cpp:137
// SoftmaxSchedule.h:253
struct TmemSoftmaxGlobal0Stack {
  // Res.cpp:208
  inline __device__ TmemSoftmaxGlobal0Stack(int32_t warpId,
                                            int32_t clusterDimX,
                                            int32_t clusterDimY,
                                            int32_t barInitWarpId,
                                            int32_t orderedSequenceGroupId) {}
};
// Res.cpp:137
// Fmha.h:1943
struct OrderP01Stack {
  // Res.cpp:208
  inline __device__ OrderP01Stack(OrderP01SmemBarrier& orderP01SmemBarrier,
                                  int32_t warpId,
                                  int32_t clusterDimX,
                                  int32_t clusterDimY,
                                  int32_t barInitWarpId,
                                  int32_t orderedSequenceGroupId) {}
};
// Res.cpp:137
// Fmha.h:1973
struct TmemP0Stack {
  // MemBuffers.cpp:488
  int8_t* mDepSmemPtr6;
  // Res.cpp:595
  trtllm::dev::CutlassCpAsyncPipeline<2, true> mPipeline;
  // TmemP.h:458
  int32_t const mNamedBarId;
  // TmemP.h:461
  int32_t const mInstId;
  // Res.cpp:208
  inline __device__ TmemP0Stack(TmemP0SmemBarrier& tmemP0SmemBarrier,
                                SmemPOSmem& smemPOSmem,
                                SmemPOStack& smemPOStack,
                                OrderP01SmemBarrier& orderP01SmemBarrier,
                                OrderP01Stack& orderP01Stack,
                                int32_t warpId,
                                int32_t clusterDimX,
                                int32_t clusterDimY,
                                int32_t barInitWarpId,
                                int32_t orderedSequenceGroupId,
                                int32_t namedBarId,
                                int32_t instId)
    : // MemBuffers.cpp:501
    mDepSmemPtr6{&smemPOSmem.mArray[int32_t{0}]}
    , // Res.cpp:644
    mPipeline{tmemP0SmemBarrier.mBarriers, warpId, int32_t{128}, int32_t{32}, barInitWarpId}
    , // TmemP.h:449
    mNamedBarId{namedBarId}
    , // TmemP.h:451
    mInstId{instId} {}
};
// Res.cpp:137
// Fmha.h:2060
struct TmemOStack {
  // MemBuffers.cpp:488
  int8_t* mDepSmemPtr6;
  // Res.cpp:595
  trtllm::dev::CutlassUmmaAsyncPipeline<1, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  // Res.cpp:208
  inline __device__ TmemOStack(TmemOSmemBarrier& tmemOSmemBarrier,
                               SmemPOSmem& smemPOSmem,
                               SmemPOStack& smemPOStack,
                               int32_t warpId,
                               int32_t clusterDimX,
                               int32_t clusterDimY,
                               int32_t barInitWarpId,
                               int32_t orderedSequenceGroupId)
    : // MemBuffers.cpp:501
    mDepSmemPtr6{&smemPOSmem.mArray[int32_t{0}]}
    , // Res.cpp:771
    mPipeline{tmemOSmemBarrier.mBarriers,
              warpId,
              int32_t{128},
              CuteFlatTuple1119{},
              cute::true_type{},
              cute::true_type{},
              barInitWarpId} {}
};
// Res.cpp:137
// Fmha.h:2077
struct TmemCorr0Stack {
  // MemBuffers.cpp:488
  float* mDepSmemPtr9;
  // MemBuffers.cpp:488
  int8_t* mDepSmemPtr6;
  // Res.cpp:208
  inline __device__ TmemCorr0Stack(SmemCorrWarpGrpRed1Smem& smemCorrWarpGrpRed1Smem,
                                   SmemCorrWarpGrpRed1Stack& smemCorrWarpGrpRed1Stack,
                                   SmemPOSmem& smemPOSmem,
                                   SmemPOStack& smemPOStack,
                                   int32_t warpId,
                                   int32_t clusterDimX,
                                   int32_t clusterDimY,
                                   int32_t barInitWarpId,
                                   int32_t orderedSequenceGroupId)
    : // MemBuffers.cpp:501
    mDepSmemPtr9{&smemCorrWarpGrpRed1Smem.mArray[int32_t{0}]}
    , // MemBuffers.cpp:501
    mDepSmemPtr6{&smemPOSmem.mArray[int32_t{0}]} {}
};
// Kernel.cpp:237
struct KernelState {
  // Kernel.cpp:57
  int32_t const mNumNonExitingCtas;
  // Kernel.cpp:59
  int32_t const mThreadIdx;
  // Kernel.cpp:62
  uint32_t* const mTmemSwStatePtr;
  // Kernel.cpp:64
  int32_t const mWarpIdx;
  // Kernel.cpp:66
  int32_t const mClusterDimX;
  // Kernel.cpp:68
  int32_t const mClusterDimY;
  // Kernel.cpp:70
  int32_t const mClusterDimZ;
  // Kernel.cpp:73
  inline __device__ KernelState(fmha::KernelParams const& params, uint32_t* tmemSwStatePtr)
    : // Kernel.cpp:2397
    mNumNonExitingCtas{int32_t{2147483647}}
    , // Kernel.cpp:197
    mThreadIdx{reinterpret_cast<int32_t const&>(threadIdx.x)}
    , // Kernel.cpp:105
    mTmemSwStatePtr{tmemSwStatePtr}
    , // Utils.h:188
    mWarpIdx{
      __shfl_sync(uint32_t{0xffffffff}, (mThreadIdx) / (int32_t{32}), int32_t{0}, int32_t{32})}
    , // Utils.h:188
    mClusterDimX{__shfl_sync(uint32_t{0xffffffff},
                             int32_t{trtllm::dev::getClusterDimX()},
                             int32_t{0},
                             int32_t{32})}
    , // Utils.h:188
    mClusterDimY{__shfl_sync(uint32_t{0xffffffff},
                             int32_t{trtllm::dev::getClusterDimY()},
                             int32_t{0},
                             int32_t{32})}
    , // Utils.h:188
    mClusterDimZ{__shfl_sync(uint32_t{0xffffffff},
                             int32_t{trtllm::dev::getClusterDimZ()},
                             int32_t{0},
                             int32_t{32})} {}
};
// Task.cpp:544
// Fmha.h:1571
struct LoadPageOffsetsTask {
  // Task.cpp:283
  int32_t mCtaIdxX;
  // Task.cpp:287
  int32_t mCtaIdxY;
  // Task.cpp:291
  int32_t mCtaIdxZ;
  // FmhaTask.h:220
  int32_t mHeadIdx;
  // FmhaTask.h:218
  int32_t mBatchIdx;
  // FmhaTask.h:208
  int32_t mSeqOffsetQ;
  // FmhaTask.h:210
  int32_t mSeqLenQ;
  // FmhaTask.h:224
  int32_t mCtaIdxQ;
  // FmhaTask.h:226
  int32_t mCtaIdxKv;
  // FmhaTask.h:214
  int32_t mSeqLenKv;
  // FmhaTask.h:216
  int32_t mNumCtasKv;
  // Task.cpp:371
  int32_t const mWarpGrpThreadIdx;
  // Task.cpp:551
  inline __device__ LoadPageOffsetsTask(fmha::KernelParams const& params,
                                        KernelState const& state,
                                        int32_t warpGrpStart)
    : // Kernel.cpp:194
    mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , // Kernel.cpp:195
    mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , // Kernel.cpp:196
    mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)}
    , // Task.cpp:287
    mHeadIdx{mCtaIdxY}
    , // Task.cpp:291
    mBatchIdx{mCtaIdxZ}
    , // FmhaTask.h:394
    mSeqOffsetQ{int32_t(bool{params.ptrCumSeqLensQ == nullptr})
                  ? ((mBatchIdx) * (params.mMaxSeqLenQ))
                  : (int32_t{params.ptrCumSeqLensQ[mBatchIdx]})}
    , // FmhaTask.h:410
    mSeqLenQ{int32_t(bool{params.ptrCumSeqLensQ == nullptr})
               ? (params.mMaxSeqLenQ)
               : ((int32_t{params.ptrCumSeqLensQ[(mBatchIdx) + (int32_t{1})]}) - (mSeqOffsetQ))}
    , // FmhaTask.h:516
    mCtaIdxQ{(mCtaIdxX) / (params.mMaxNumCtasKv)}
    , // FmhaTask.h:517
    mCtaIdxKv{(mCtaIdxX) % (params.mMaxNumCtasKv)}
    , // FmhaTask.h:543
    mSeqLenKv{(int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                             ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                             : (mBatchIdx)]}) -
              (((mSeqLenQ) - (int32_t{1})) - (mCtaIdxQ))}
    , // FmhaTask.h:565
    mNumCtasKv{
      int32_t{min(int32_t{((mSeqLenKv) + (int32_t{127})) / (int32_t{128})}, params.mMaxNumCtasKv)}}
    , // Task.cpp:379
    mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))} {}
  // Task.cpp:507
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:532
    return ((state.mWarpIdx) >= (int32_t{9})) && ((state.mWarpIdx) < (int32_t{10}));
  }
  // Task.cpp:454
  inline __device__ void execute(fmha::KernelParams const& params,
                                 KernelState const& state,
                                 SmemPageOffsetsKvSmem& smemPageOffsetsKvDstSmem,
                                 SmemPageOffsetsKvStack& smemPageOffsetsKvDstStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<56>{});
    // Task.cpp:1979
    trtllm::dev::CutlassCpAsyncPipeline<6>::PipelineState smemPageOffsetsKvProdState{int32_t{0},
                                                                                     int32_t{1},
                                                                                     int32_t{0}};
    // Task.cpp:1999
    int32_t smemPageOffsetsKvProdToken{int32_t{1}};
    // FmhaTask.h:582
    int32_t numLoopSteps;
    // FmhaTask.h:630
    if (((mCtaIdxQ) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
      // FmhaTask.h:668
      int32_t numSteps{((mSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
                       ((mNumCtasKv) * (int32_t{128}))};
      // FmhaTask.h:682
      numLoopSteps = numSteps;
    } else {
      // FmhaTask.h:651
      return;
    }
    // SmemPageOffsetsKv.h:204
    int32_t const* ptrPageIdxK5;
    // SmemPageOffsetsKv.h:210
    ptrPageIdxK5 =
      params.ptrPageIdxKv +
      (int32_t(params.mUseBlockSparseAttention)
         ? ((int32_t(params.mUsesSharedPagedKvIdx)
               ? ((mBatchIdx) * (params.mMaxNumPagesPerSeqKv))
               : (((mBatchIdx) * (params.mMaxNumPagesPerSeqKv)) * (int32_t{2}))) +
            (int32_t(params.mUsesSharedPagedKvIdx)
               ? (((params.mBatchSize) * (params.mMaxNumPagesPerSeqKv)) * (mHeadIdx))
               : ((((params.mBatchSize) * (params.mMaxNumPagesPerSeqKv)) * (mHeadIdx)) *
                  (int32_t{2}))))
         : (int32_t(params.mUsesSharedPagedKvIdx)
              ? ((mBatchIdx) * (params.mMaxNumPagesPerSeqKv))
              : (((mBatchIdx) * (params.mMaxNumPagesPerSeqKv)) * (int32_t{2}))));
    // SmemPageOffsetsKv.h:215
    int32_t const* ptrPageIdxV5;
    // SmemPageOffsetsKv.h:224
    if (params.mUsesSharedPagedKvIdx) {
      // SmemPageOffsetsKv.h:226
      ptrPageIdxV5 = ptrPageIdxK5;
    } else {
      // SmemPageOffsetsKv.h:228
      ptrPageIdxV5 = ptrPageIdxK5 + (params.mMaxNumPagesPerSeqKv);
    }
    // SmemPageOffsetsKv.h:302
    int32_t pageIdxUb5{(int32_t{((mSeqLenKv) + (int32_t{63})) / (int32_t{64})}) - (int32_t{1})};
    //
    // Loop body.
    //
    // Task.cpp:3350
    CUTLASS_PRAGMA_NO_UNROLL
    for (int32_t loopOffset403 = int32_t{0}; loopOffset403 < numLoopSteps;
         loopOffset403 += int32_t{16}) {
      // Task.cpp:3403
      bool const isFirstLoopIter{(loopOffset403) == (int32_t{0})};
      // Task.cpp:3423
      bool const isLastLoopIter{((loopOffset403) + (int32_t{16})) >= (numLoopSteps)};
      //
      // smemPageOffsetsKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:4948
      {
        // Task.cpp:4962
        if ((loopOffset403) >= (int32_t{0})) {
          // Task.cpp:4984
          smemPageOffsetsKvProdToken =
            smemPageOffsetsKvDstStack.mPipeline.producer_try_acquire(smemPageOffsetsKvProdState);
        }
      }
      // Task.cpp:1573
      // Task.cpp:4180
      {
        // Task.cpp:4210
        smemPageOffsetsKvDstStack.mPipeline.producer_acquire(smemPageOffsetsKvProdState,
                                                             smemPageOffsetsKvProdToken);
      }
      //
      // smemPageOffsetsKv [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:5038
      {
        // Task.cpp:5829
        int32_t index{smemPageOffsetsKvProdState.index()};
        // SmemPageOffsetsKv.h:390
        int32_t* ptrSmemPageOffsets;
        // SmemPageOffsetsKv.h:392
        ptrSmemPageOffsets = smemPageOffsetsKvDstSmem.mArray[index] + (mWarpGrpThreadIdx);
        // SmemPageOffsetsKv.h:430
        int32_t pageIdx{(((mCtaIdxKv) * (numLoopSteps) + (loopOffset403)) * (int32_t{2})) +
                        (mWarpGrpThreadIdx)};
        // SmemPageOffsetsKv.h:488
        trtllm::dev::cpAsync((ptrSmemPageOffsets + int32_t{0}),
                             (ptrPageIdxK5 + int32_t{min(pageIdx, pageIdxUb5)}),
                             int32_t{0},
                             int32_t{0},
                             int32_t{4});
      }
      //
      // smemPageOffsetsKv [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:4409
      {
        // Task.cpp:4427
        {
          // Task.cpp:4443
          smemPageOffsetsKvDstStack.mPipeline.producer_commit(smemPageOffsetsKvProdState);
        }
        // Task.cpp:43
        ++smemPageOffsetsKvProdState;
      }
      //
      // smemPageOffsetsKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:4948
      {
        // Task.cpp:4962
        if ((loopOffset403) >= (int32_t{0})) {
          // Task.cpp:4984
          smemPageOffsetsKvProdToken =
            smemPageOffsetsKvDstStack.mPipeline.producer_try_acquire(smemPageOffsetsKvProdState);
        }
      }
      // Task.cpp:1573
      // Task.cpp:4180
      {
        // Task.cpp:4210
        smemPageOffsetsKvDstStack.mPipeline.producer_acquire(smemPageOffsetsKvProdState,
                                                             smemPageOffsetsKvProdToken);
      }
      //
      // smemPageOffsetsKv [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:5038
      {
        // Task.cpp:5829
        int32_t index{smemPageOffsetsKvProdState.index()};
        // SmemPageOffsetsKv.h:390
        int32_t* ptrSmemPageOffsets;
        // SmemPageOffsetsKv.h:392
        ptrSmemPageOffsets = smemPageOffsetsKvDstSmem.mArray[index] + (mWarpGrpThreadIdx);
        // SmemPageOffsetsKv.h:430
        int32_t pageIdx{(((mCtaIdxKv) * (numLoopSteps) + (loopOffset403)) * (int32_t{2})) +
                        (mWarpGrpThreadIdx)};
        // SmemPageOffsetsKv.h:488
        trtllm::dev::cpAsync((ptrSmemPageOffsets + int32_t{0}),
                             (ptrPageIdxV5 + int32_t{min(pageIdx, pageIdxUb5)}),
                             int32_t{0},
                             int32_t{0},
                             int32_t{4});
      }
      //
      // smemPageOffsetsKv [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:4409
      {
        // Task.cpp:4427
        {
          // Task.cpp:4443
          smemPageOffsetsKvDstStack.mPipeline.producer_commit(smemPageOffsetsKvProdState);
        }
        // Task.cpp:43
        ++smemPageOffsetsKvProdState;
      }
    }
  //
  // Tail work.
  //
  // Task.cpp:3511
  ExitTileWithSignalingLabel:
  // Task.cpp:3518
  ExitTileWithoutSignalingLabel:
    // Task.cpp:3528
    {}
  }
};
// Task.cpp:544
// Fmha.h:1706
struct LoadTask {
  // Task.cpp:283
  int32_t mCtaIdxX;
  // Task.cpp:287
  int32_t mCtaIdxY;
  // Task.cpp:291
  int32_t mCtaIdxZ;
  // FmhaTask.h:220
  int32_t mHeadIdx;
  // FmhaTask.h:218
  int32_t mBatchIdx;
  // FmhaTask.h:208
  int32_t mSeqOffsetQ;
  // FmhaTask.h:210
  int32_t mSeqLenQ;
  // FmhaTask.h:224
  int32_t mCtaIdxQ;
  // FmhaTask.h:226
  int32_t mCtaIdxKv;
  // FmhaTask.h:214
  int32_t mSeqLenKv;
  // FmhaTask.h:216
  int32_t mNumCtasKv;
  // Task.cpp:551
  inline __device__ LoadTask(fmha::KernelParams const& params,
                             KernelState const& state,
                             int32_t warpGrpStart)
    : // Kernel.cpp:194
    mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , // Kernel.cpp:195
    mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , // Kernel.cpp:196
    mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)}
    , // Task.cpp:287
    mHeadIdx{mCtaIdxY}
    , // Task.cpp:291
    mBatchIdx{mCtaIdxZ}
    , // FmhaTask.h:394
    mSeqOffsetQ{int32_t(bool{params.ptrCumSeqLensQ == nullptr})
                  ? ((mBatchIdx) * (params.mMaxSeqLenQ))
                  : (int32_t{params.ptrCumSeqLensQ[mBatchIdx]})}
    , // FmhaTask.h:410
    mSeqLenQ{int32_t(bool{params.ptrCumSeqLensQ == nullptr})
               ? (params.mMaxSeqLenQ)
               : ((int32_t{params.ptrCumSeqLensQ[(mBatchIdx) + (int32_t{1})]}) - (mSeqOffsetQ))}
    , // FmhaTask.h:516
    mCtaIdxQ{(mCtaIdxX) / (params.mMaxNumCtasKv)}
    , // FmhaTask.h:517
    mCtaIdxKv{(mCtaIdxX) % (params.mMaxNumCtasKv)}
    , // FmhaTask.h:543
    mSeqLenKv{(int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                             ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                             : (mBatchIdx)]}) -
              (((mSeqLenQ) - (int32_t{1})) - (mCtaIdxQ))}
    , // FmhaTask.h:565
    mNumCtasKv{int32_t{
      min(int32_t{((mSeqLenKv) + (int32_t{127})) / (int32_t{128})}, params.mMaxNumCtasKv)}} {}
  // Task.cpp:507
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:532
    return ((state.mWarpIdx) >= (int32_t{11})) && ((state.mWarpIdx) < (int32_t{12}));
  }
  // Task.cpp:454
  inline __device__ void execute(fmha::KernelParams const& params,
                                 KernelState const& state,
                                 SmemQSmem& smemQDstSmem,
                                 SmemQStack& smemQDstStack,
                                 SmemKvSmem& smemKvDstSmem,
                                 SmemKvStack& smemKvDstStack,
                                 SmemPageOffsetsKvSmem& smemPageOffsetsKvSrcSmem,
                                 SmemPageOffsetsKvStack& smemPageOffsetsKvSrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<56>{});
    // Task.cpp:2079
    trtllm::dev::CutlassCpAsyncPipeline<6>::PipelineState smemPageOffsetsKvConsState{};
    // Task.cpp:2086
    trtllm::dev::CutlassCpAsyncPipeline<6>::PipelineState smemPageOffsetsKvConsReleaseState{};
    // Task.cpp:2100
    int32_t smemPageOffsetsKvConsToken{int32_t{0}};
    // Task.cpp:1979
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      1,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemQProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    // Task.cpp:1999
    int32_t smemQProdToken{int32_t{1}};
    // Task.cpp:1979
    trtllm::dev::CutlassTmaAsyncPipeline<4>::PipelineState smemKvProdState{int32_t{0},
                                                                           int32_t{1},
                                                                           int32_t{0}};
    // Task.cpp:1999
    int32_t smemKvProdToken{int32_t{1}};
    // SmemKv.h:749
    int32_t smemVoteIdx3{int32_t{0}};
    // FmhaTask.h:582
    int32_t numLoopSteps;
    // FmhaTask.h:630
    if (((mCtaIdxQ) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
      // FmhaTask.h:668
      int32_t numSteps{((mSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
                       ((mNumCtasKv) * (int32_t{128}))};
      // FmhaTask.h:682
      numLoopSteps = numSteps;
    } else {
      // FmhaTask.h:651
      return;
    }
    // Task.cpp:3168
    bool const hasOneLoopIter{(int32_t{0}) < (numLoopSteps)};
    // Task.cpp:3179
    int32_t lastLoopOffset{int32_t{0}};
    // SmemKv.h:668
    cutlass::AlignedArray<int32_t, 2> pageOffsets3;
    //
    // Hoist the first iter.
    //
    //
    // gmemQ [ConsWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    // GmemQkv.h:82
    int32_t idxQ00;
    // GmemQkv.h:83
    idxQ00 = mCtaIdxQ;
    // Task.cpp:1573
    if (hasOneLoopIter) {
    }
    //
    // gmemKv [ConsWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
    }
    //
    // smemQ [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4962
      {
        // Task.cpp:4984
        smemQProdToken = smemQDstStack.mPipeline.producer_try_acquire(smemQProdState);
      }
    }
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4210
      smemQDstStack.mPipeline.producer_acquire(smemQProdState, smemQProdToken);
    }
    //
    // smemQ [ProdWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // SmemQ.h:201
    int32_t prodIdxQ02;
    // Task.cpp:1477
    prodIdxQ02 = idxQ00;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4300
      uint64_t* barrier{smemQDstStack.mPipeline.producer_get_barrier(smemQProdState)};
      // Task.cpp:5829
      int32_t index{smemQProdState.index()};
      // Common.h:603
      cudaGridDependencySynchronize();
      // SmemTile.cpp:484
      int32_t coords[4];
      // SmemTile.cpp:491
      coords[int32_t{0}] = int32_t{0};
      // SmemTile.cpp:491
      coords[int32_t{1}] = int32_t{0};
      // SmemTile.cpp:491
      coords[int32_t{2}] = mHeadIdx;
      // SmemTile.cpp:491
      coords[int32_t{3}] = (prodIdxQ02) * (int32_t{1}) + (mSeqOffsetQ);
      // SmemTile.cpp:610
      if (bool{cute::elect_one_sync()}) {
        // CudaPtx.h:48
        cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                       cuda_ptx::space_global_t{},
                                       &smemQDstSmem.mArray[index][int32_t{0}],
                                       &params.tmaQ_,
                                       coords,
                                       barrier);
      }
      // SmemTile.cpp:619
      coords[int32_t{0}] += int32_t{64};
      // SmemTile.cpp:610
      if (bool{cute::elect_one_sync()}) {
        // CudaPtx.h:48
        cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                       cuda_ptx::space_global_t{},
                                       &smemQDstSmem.mArray[index][int32_t{1024}],
                                       &params.tmaQ_,
                                       coords,
                                       barrier);
      }
      // SmemTile.cpp:619
      coords[int32_t{0}] += int32_t{64};
      // SmemTile.cpp:610
      if (bool{cute::elect_one_sync()}) {
        // CudaPtx.h:48
        cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                       cuda_ptx::space_global_t{},
                                       &smemQDstSmem.mArray[index][int32_t{2048}],
                                       &params.tmaQ_,
                                       coords,
                                       barrier);
      }
      // SmemTile.cpp:619
      coords[int32_t{0}] += int32_t{64};
      // SmemTile.cpp:610
      if (bool{cute::elect_one_sync()}) {
        // CudaPtx.h:48
        cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                       cuda_ptx::space_global_t{},
                                       &smemQDstSmem.mArray[index][int32_t{3072}],
                                       &params.tmaQ_,
                                       coords,
                                       barrier);
      }
    }
    //
    // smemQ [ProdCommit, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4427
      {
        // Task.cpp:4443
        smemQDstStack.mPipeline.producer_commit(smemQProdState);
      }
      // Task.cpp:43
      ++smemQProdState;
    }
    //
    // smemPageOffsetsKv [ConsWait, FirstIter, FreqInfo{0, 16}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:2745
        smemPageOffsetsKvConsToken =
          smemPageOffsetsKvSrcStack.mPipeline.consumer_try_wait(smemPageOffsetsKvConsState);
      }
      // Task.cpp:2813
      smemPageOffsetsKvSrcStack.mPipeline.consumer_wait(smemPageOffsetsKvConsState,
                                                        smemPageOffsetsKvConsToken);
    }
    //
    // smemPageOffsetsKv [ConsWork (call 0), FirstIter, FreqInfo{0, 16}, UserTags{1}, Flags{0}].
    //
    // SmemPageOffsetsKv.h:320
    int32_t* ptrSmemPageOffsetsK5;
    // SmemPageOffsetsKv.h:326
    int32_t* ptrSmemPageOffsetsV5;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{smemPageOffsetsKvConsState.index()};
      // SmemPageOffsetsKv.h:349
      ptrSmemPageOffsetsK5 = smemPageOffsetsKvSrcSmem.mArray[index];
      // Task.cpp:43
      ++smemPageOffsetsKvConsState;
    }
    //
    // smemKv [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4962
      {
        // Task.cpp:4984
        smemKvProdToken = smemKvDstStack.mPipeline.producer_try_acquire(smemKvProdState);
      }
    }
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4210
      smemKvDstStack.mPipeline.producer_acquire(smemKvProdState, smemKvProdToken);
    }
    //
    // smemKv [ProdWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // SmemKv.h:772
    int32_t* ptrSmemPageOffsetsK3;
    // SmemKv.h:786
    int32_t* ptrSmemPageOffsetsV3;
    // Task.cpp:1477
    ptrSmemPageOffsetsK3 = ptrSmemPageOffsetsK5;
    // Task.cpp:1477
    ptrSmemPageOffsetsV3 = ptrSmemPageOffsetsV5;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4300
      uint64_t* barrier{smemKvDstStack.mPipeline.producer_get_barrier(smemKvProdState)};
      // Task.cpp:5829
      int32_t index{smemKvProdState.index()};
      // SmemKv.h:631
      int32_t adjustedGlobalTileIdx{(mCtaIdxKv) * (numLoopSteps)};
      // SmemKv.h:1404
      int32_t headDimOffset{int32_t{0}};
      // SmemKv.h:1529
      int32_t tokenOffset{int32_t{0}};
      // SmemKv.h:1652
      {
        // SmemKv.h:1669
        cutlass::AlignedArray<int32_t, 2> localPageOffsets03;
        // SmemKv.h:1685
        localPageOffsets03 = reinterpret_cast<cutlass::AlignedArray<int32_t, 2>*>(
          (ptrSmemPageOffsetsK3 + int32_t{0}))[int32_t{0}];
        // SmemKv.h:1705
        pageOffsets3[int32_t{0}] = localPageOffsets03[int32_t{0}];
        // SmemKv.h:1705
        pageOffsets3[int32_t{1}] = localPageOffsets03[int32_t{1}];
      }
      //
      // Load pageOffsets for headDimStageIdx = 0.
      //
      // SmemKv.h:1213
      {
        // SmemTile.cpp:484
        int32_t coords[4];
        // SmemTile.cpp:491
        coords[int32_t{0}] = headDimOffset;
        // SmemTile.cpp:491
        coords[int32_t{1}] = tokenOffset;
        // SmemTile.cpp:491
        coords[int32_t{2}] = mHeadIdx;
        // SmemTile.cpp:491
        coords[int32_t{3}] = pageOffsets3[int32_t{0}];
        // SmemTile.cpp:610
        if (bool{cute::elect_one_sync()}) {
          // CudaPtx.h:48
          cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                         cuda_ptx::space_global_t{},
                                         &smemKvDstSmem.mArray[index][int32_t{0}],
                                         &params.tmaK_,
                                         coords,
                                         barrier);
        }
      }
      // SmemKv.h:1213
      {
        // SmemTile.cpp:484
        int32_t coords[4];
        // SmemTile.cpp:491
        coords[int32_t{0}] = headDimOffset;
        // SmemTile.cpp:491
        coords[int32_t{1}] = tokenOffset;
        // SmemTile.cpp:491
        coords[int32_t{2}] = mHeadIdx;
        // SmemTile.cpp:491
        coords[int32_t{3}] = pageOffsets3[int32_t{1}];
        // SmemTile.cpp:610
        if (bool{cute::elect_one_sync()}) {
          // CudaPtx.h:48
          cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                         cuda_ptx::space_global_t{},
                                         &smemKvDstSmem.mArray[index][int32_t{8192}],
                                         &params.tmaK_,
                                         coords,
                                         barrier);
        }
      }
    }
    //
    // smemKv [ProdCommit, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4427
      {
        // Task.cpp:4443
        smemKvDstStack.mPipeline.producer_commit(smemKvProdState);
      }
      // Task.cpp:43
      ++smemKvProdState;
    }
    //
    // smemKv [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4962
      {
        // Task.cpp:4984
        smemKvProdToken = smemKvDstStack.mPipeline.producer_try_acquire(smemKvProdState);
      }
    }
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4210
      smemKvDstStack.mPipeline.producer_acquire(smemKvProdState, smemKvProdToken);
    }
    //
    // smemKv [ProdWork (call 1), FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1477
    ptrSmemPageOffsetsK3 = ptrSmemPageOffsetsK5;
    // Task.cpp:1477
    ptrSmemPageOffsetsV3 = ptrSmemPageOffsetsV5;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4300
      uint64_t* barrier{smemKvDstStack.mPipeline.producer_get_barrier(smemKvProdState)};
      // Task.cpp:5829
      int32_t index{smemKvProdState.index()};
      // SmemKv.h:631
      int32_t adjustedGlobalTileIdx{(mCtaIdxKv) * (numLoopSteps)};
      // SmemKv.h:1404
      int32_t headDimOffset{int32_t{128}};
      // SmemKv.h:1529
      int32_t tokenOffset{int32_t{0}};
      // SmemKv.h:1652

      //
      // Load pageOffsets for headDimStageIdx = 1.
      //
      // SmemKv.h:1213
      {
        // SmemTile.cpp:484
        int32_t coords[4];
        // SmemTile.cpp:491
        coords[int32_t{0}] = headDimOffset;
        // SmemTile.cpp:491
        coords[int32_t{1}] = tokenOffset;
        // SmemTile.cpp:491
        coords[int32_t{2}] = mHeadIdx;
        // SmemTile.cpp:491
        coords[int32_t{3}] = pageOffsets3[int32_t{0}];
        // SmemTile.cpp:610
        if (bool{cute::elect_one_sync()}) {
          // CudaPtx.h:48
          cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                         cuda_ptx::space_global_t{},
                                         &smemKvDstSmem.mArray[index][int32_t{0}],
                                         &params.tmaK_,
                                         coords,
                                         barrier);
        }
      }
      // SmemKv.h:1213
      {
        // SmemTile.cpp:484
        int32_t coords[4];
        // SmemTile.cpp:491
        coords[int32_t{0}] = headDimOffset;
        // SmemTile.cpp:491
        coords[int32_t{1}] = tokenOffset;
        // SmemTile.cpp:491
        coords[int32_t{2}] = mHeadIdx;
        // SmemTile.cpp:491
        coords[int32_t{3}] = pageOffsets3[int32_t{1}];
        // SmemTile.cpp:610
        if (bool{cute::elect_one_sync()}) {
          // CudaPtx.h:48
          cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                         cuda_ptx::space_global_t{},
                                         &smemKvDstSmem.mArray[index][int32_t{8192}],
                                         &params.tmaK_,
                                         coords,
                                         barrier);
        }
      }
    }
    //
    // smemKv [ProdCommit, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4427
      {
        // Task.cpp:4443
        smemKvDstStack.mPipeline.producer_commit(smemKvProdState);
      }
      // Task.cpp:43
      ++smemKvProdState;
    }
    //
    // smemPageOffsetsKv [ConsRelease, FirstIter, FreqInfo{0, 16}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:3772

    //
    // Loop body.
    //
    // Task.cpp:3350
    CUTLASS_PRAGMA_NO_UNROLL
    for (int32_t loopOffset723 = int32_t{0}; loopOffset723 < (numLoopSteps) - (int32_t{1});
         ++loopOffset723) {
      // Task.cpp:3423
      bool const isLastLoopIter{((loopOffset723) + (int32_t{1})) >=
                                ((numLoopSteps) - (int32_t{1}))};
      //
      // gmemKv [ConsWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:2893
      {}
      //
      // smemPageOffsetsKv [ConsWait, Info{0}, FreqInfo{1, 16}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:3772
      if ((((loopOffset723) + (int32_t{1})) % (int32_t{16})) == (int32_t{0})) {
        // Task.cpp:1573
        // Task.cpp:2781
        {
          // Task.cpp:1573
          // Task.cpp:2722
          {
            // Task.cpp:2745
            smemPageOffsetsKvConsToken =
              smemPageOffsetsKvSrcStack.mPipeline.consumer_try_wait(smemPageOffsetsKvConsState);
          }
          // Task.cpp:2813
          smemPageOffsetsKvSrcStack.mPipeline.consumer_wait(smemPageOffsetsKvConsState,
                                                            smemPageOffsetsKvConsToken);
        }
      }
      //
      // smemPageOffsetsKv [ConsWork (call 1), Info{0}, FreqInfo{1, 16}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:3772
      if ((((loopOffset723) + (int32_t{1})) % (int32_t{16})) == (int32_t{0})) {
        // Task.cpp:1573
        // Task.cpp:2893
        {
          // Task.cpp:5829
          int32_t index{smemPageOffsetsKvConsState.index()};
          // SmemPageOffsetsKv.h:349
          ptrSmemPageOffsetsK5 = smemPageOffsetsKvSrcSmem.mArray[index];
          // Task.cpp:43
          ++smemPageOffsetsKvConsState;
        }
      }
      //
      // smemKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:4948
      {
        // Task.cpp:4962
        if ((loopOffset723) >= (int32_t{0})) {
          // Task.cpp:4984
          smemKvProdToken = smemKvDstStack.mPipeline.producer_try_acquire(smemKvProdState);
        }
      }
      // Task.cpp:1573
      // Task.cpp:4180
      {
        // Task.cpp:4210
        smemKvDstStack.mPipeline.producer_acquire(smemKvProdState, smemKvProdToken);
      }
      //
      // smemKv [ProdWork (call 2), Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{0}].
      //
      // Task.cpp:1477
      ptrSmemPageOffsetsK3 = ptrSmemPageOffsetsK5;
      // Task.cpp:1477
      ptrSmemPageOffsetsV3 = ptrSmemPageOffsetsV5;
      // Task.cpp:1573
      // Task.cpp:5038
      {
        // Task.cpp:4300
        uint64_t* barrier{smemKvDstStack.mPipeline.producer_get_barrier(smemKvProdState)};
        // Task.cpp:5829
        int32_t index{smemKvProdState.index()};
        // SmemKv.h:631
        int32_t adjustedGlobalTileIdx{(mCtaIdxKv) * (numLoopSteps) +
                                      ((loopOffset723) + (int32_t{1}))};
        // SmemKv.h:1404
        int32_t headDimOffset{int32_t{0}};
        // SmemKv.h:1529
        int32_t tokenOffset{int32_t{0}};
        // SmemKv.h:1652
        {
          // SmemKv.h:1669
          cutlass::AlignedArray<int32_t, 2> localPageOffsets03;
          // SmemKv.h:1685
          localPageOffsets03 = reinterpret_cast<cutlass::AlignedArray<int32_t, 2>*>(
            (ptrSmemPageOffsetsK3 +
             (((loopOffset723) + (int32_t{1})) * (int32_t{2})) % (int32_t{32})))[int32_t{0}];
          // SmemKv.h:1705
          pageOffsets3[int32_t{0}] = localPageOffsets03[int32_t{0}];
          // SmemKv.h:1705
          pageOffsets3[int32_t{1}] = localPageOffsets03[int32_t{1}];
        }
        //
        // Load pageOffsets for headDimStageIdx = 0.
        //
        // SmemKv.h:1213
        {
          // SmemTile.cpp:484
          int32_t coords[4];
          // SmemTile.cpp:491
          coords[int32_t{0}] = headDimOffset;
          // SmemTile.cpp:491
          coords[int32_t{1}] = tokenOffset;
          // SmemTile.cpp:491
          coords[int32_t{2}] = mHeadIdx;
          // SmemTile.cpp:491
          coords[int32_t{3}] = pageOffsets3[int32_t{0}];
          // SmemTile.cpp:610
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{0}],
                                           &params.tmaK_,
                                           coords,
                                           barrier);
          }
        }
        // SmemKv.h:1213
        {
          // SmemTile.cpp:484
          int32_t coords[4];
          // SmemTile.cpp:491
          coords[int32_t{0}] = headDimOffset;
          // SmemTile.cpp:491
          coords[int32_t{1}] = tokenOffset;
          // SmemTile.cpp:491
          coords[int32_t{2}] = mHeadIdx;
          // SmemTile.cpp:491
          coords[int32_t{3}] = pageOffsets3[int32_t{1}];
          // SmemTile.cpp:610
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{8192}],
                                           &params.tmaK_,
                                           coords,
                                           barrier);
          }
        }
      }
      //
      // smemKv [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{0}].
      //
      // Task.cpp:4409
      {
        // Task.cpp:4427
        {
          // Task.cpp:4443
          smemKvDstStack.mPipeline.producer_commit(smemKvProdState);
        }
        // Task.cpp:43
        ++smemKvProdState;
      }
      //
      // smemKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:4948
      {
        // Task.cpp:4962
        if ((loopOffset723) >= (int32_t{0})) {
          // Task.cpp:4984
          smemKvProdToken = smemKvDstStack.mPipeline.producer_try_acquire(smemKvProdState);
        }
      }
      // Task.cpp:1573
      // Task.cpp:4180
      {
        // Task.cpp:4210
        smemKvDstStack.mPipeline.producer_acquire(smemKvProdState, smemKvProdToken);
      }
      //
      // smemKv [ProdWork (call 3), Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{0}].
      //
      // Task.cpp:1477
      ptrSmemPageOffsetsK3 = ptrSmemPageOffsetsK5;
      // Task.cpp:1477
      ptrSmemPageOffsetsV3 = ptrSmemPageOffsetsV5;
      // Task.cpp:1573
      // Task.cpp:5038
      {
        // Task.cpp:4300
        uint64_t* barrier{smemKvDstStack.mPipeline.producer_get_barrier(smemKvProdState)};
        // Task.cpp:5829
        int32_t index{smemKvProdState.index()};
        // SmemKv.h:631
        int32_t adjustedGlobalTileIdx{(mCtaIdxKv) * (numLoopSteps) +
                                      ((loopOffset723) + (int32_t{1}))};
        // SmemKv.h:1404
        int32_t headDimOffset{int32_t{128}};
        // SmemKv.h:1529
        int32_t tokenOffset{int32_t{0}};
        // SmemKv.h:1652

        //
        // Load pageOffsets for headDimStageIdx = 1.
        //
        // SmemKv.h:1213
        {
          // SmemTile.cpp:484
          int32_t coords[4];
          // SmemTile.cpp:491
          coords[int32_t{0}] = headDimOffset;
          // SmemTile.cpp:491
          coords[int32_t{1}] = tokenOffset;
          // SmemTile.cpp:491
          coords[int32_t{2}] = mHeadIdx;
          // SmemTile.cpp:491
          coords[int32_t{3}] = pageOffsets3[int32_t{0}];
          // SmemTile.cpp:610
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{0}],
                                           &params.tmaK_,
                                           coords,
                                           barrier);
          }
        }
        // SmemKv.h:1213
        {
          // SmemTile.cpp:484
          int32_t coords[4];
          // SmemTile.cpp:491
          coords[int32_t{0}] = headDimOffset;
          // SmemTile.cpp:491
          coords[int32_t{1}] = tokenOffset;
          // SmemTile.cpp:491
          coords[int32_t{2}] = mHeadIdx;
          // SmemTile.cpp:491
          coords[int32_t{3}] = pageOffsets3[int32_t{1}];
          // SmemTile.cpp:610
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{8192}],
                                           &params.tmaK_,
                                           coords,
                                           barrier);
          }
        }
      }
      //
      // smemKv [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{0}].
      //
      // Task.cpp:4409
      {
        // Task.cpp:4427
        {
          // Task.cpp:4443
          smemKvDstStack.mPipeline.producer_commit(smemKvProdState);
        }
        // Task.cpp:43
        ++smemKvProdState;
      }
      //
      // smemPageOffsetsKv [ConsRelease, Info{0}, FreqInfo{1, 16}, UserTags{1}, Flags{65536}].
      //
      // Task.cpp:3772
      if ((!(isLastLoopIter)) &&
          ((((loopOffset723) + (int32_t{1})) % (int32_t{16})) == (int32_t{15}))) {
        // Task.cpp:2533
        if ((loopOffset723) >= (int32_t{0})) {
          // Task.cpp:2561
          {
            // Task.cpp:2585
            smemPageOffsetsKvSrcStack.mPipeline.consumer_release(smemPageOffsetsKvConsReleaseState);
          }
          // Task.cpp:43
          ++smemPageOffsetsKvConsReleaseState;
        }
      }
      //
      // smemPageOffsetsKv [ConsWait, Info{0}, FreqInfo{0, 16}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:3772
      if (((loopOffset723) % (int32_t{16})) == (int32_t{0})) {
        // Task.cpp:1573
        // Task.cpp:2781
        {
          // Task.cpp:1573
          // Task.cpp:2722
          {
            // Task.cpp:2745
            smemPageOffsetsKvConsToken =
              smemPageOffsetsKvSrcStack.mPipeline.consumer_try_wait(smemPageOffsetsKvConsState);
          }
          // Task.cpp:2813
          smemPageOffsetsKvSrcStack.mPipeline.consumer_wait(smemPageOffsetsKvConsState,
                                                            smemPageOffsetsKvConsToken);
        }
      }
      //
      // smemPageOffsetsKv [ConsWork (call 2), Info{0}, FreqInfo{0, 16}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:3772
      if (((loopOffset723) % (int32_t{16})) == (int32_t{0})) {
        // Task.cpp:1573
        // Task.cpp:2893
        {
          // Task.cpp:5829
          int32_t index{smemPageOffsetsKvConsState.index()};
          // SmemPageOffsetsKv.h:349
          ptrSmemPageOffsetsV5 = smemPageOffsetsKvSrcSmem.mArray[index];
          // Task.cpp:43
          ++smemPageOffsetsKvConsState;
        }
      }
      //
      // smemKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:4948
      {
        // Task.cpp:4962
        if ((loopOffset723) >= (int32_t{0})) {
          // Task.cpp:4984
          smemKvProdToken = smemKvDstStack.mPipeline.producer_try_acquire(smemKvProdState);
        }
      }
      // Task.cpp:1573
      // Task.cpp:4180
      {
        // Task.cpp:4210
        smemKvDstStack.mPipeline.producer_acquire(smemKvProdState, smemKvProdToken);
      }
      //
      // smemKv [ProdWork (call 4), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1477
      ptrSmemPageOffsetsK3 = ptrSmemPageOffsetsK5;
      // Task.cpp:1477
      ptrSmemPageOffsetsV3 = ptrSmemPageOffsetsV5;
      // Task.cpp:1573
      // Task.cpp:5038
      {
        // Task.cpp:4300
        uint64_t* barrier{smemKvDstStack.mPipeline.producer_get_barrier(smemKvProdState)};
        // Task.cpp:5829
        int32_t index{smemKvProdState.index()};
        // SmemKv.h:631
        int32_t adjustedGlobalTileIdx{(mCtaIdxKv) * (numLoopSteps) + (loopOffset723)};
        // SmemKv.h:1404
        int32_t headDimOffset{int32_t{0}};
        // SmemKv.h:1529
        int32_t tokenOffset{int32_t{0}};
        // SmemKv.h:1652
        {
          // SmemKv.h:1669
          cutlass::AlignedArray<int32_t, 2> localPageOffsets03;
          // SmemKv.h:1685
          localPageOffsets03 = reinterpret_cast<cutlass::AlignedArray<int32_t, 2>*>(
            (ptrSmemPageOffsetsV3 + ((loopOffset723) * (int32_t{2})) % (int32_t{32})))[int32_t{0}];
          // SmemKv.h:1705
          pageOffsets3[int32_t{0}] = localPageOffsets03[int32_t{0}];
          // SmemKv.h:1705
          pageOffsets3[int32_t{1}] = localPageOffsets03[int32_t{1}];
        }
        //
        // Load pageOffsets for headDimStageIdx = 0.
        //
        // SmemKv.h:1213
        {
          // SmemTile.cpp:484
          int32_t coords[4];
          // SmemTile.cpp:491
          coords[int32_t{0}] = headDimOffset;
          // SmemTile.cpp:491
          coords[int32_t{1}] = tokenOffset;
          // SmemTile.cpp:491
          coords[int32_t{2}] = mHeadIdx;
          // SmemTile.cpp:491
          coords[int32_t{3}] = pageOffsets3[int32_t{0}];
          // SmemTile.cpp:610
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{0}],
                                           &params.tmaV_,
                                           coords,
                                           barrier);
          }
        }
        // SmemKv.h:1213
        {
          // SmemTile.cpp:484
          int32_t coords[4];
          // SmemTile.cpp:491
          coords[int32_t{0}] = headDimOffset;
          // SmemTile.cpp:491
          coords[int32_t{1}] = tokenOffset;
          // SmemTile.cpp:491
          coords[int32_t{2}] = mHeadIdx;
          // SmemTile.cpp:491
          coords[int32_t{3}] = pageOffsets3[int32_t{1}];
          // SmemTile.cpp:610
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{8192}],
                                           &params.tmaV_,
                                           coords,
                                           barrier);
          }
        }
      }
      //
      // smemKv [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:4409
      {
        // Task.cpp:4427
        {
          // Task.cpp:4443
          smemKvDstStack.mPipeline.producer_commit(smemKvProdState);
        }
        // Task.cpp:43
        ++smemKvProdState;
      }
      //
      // smemKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:4948
      {
        // Task.cpp:4962
        if ((loopOffset723) >= (int32_t{0})) {
          // Task.cpp:4984
          smemKvProdToken = smemKvDstStack.mPipeline.producer_try_acquire(smemKvProdState);
        }
      }
      // Task.cpp:1573
      // Task.cpp:4180
      {
        // Task.cpp:4210
        smemKvDstStack.mPipeline.producer_acquire(smemKvProdState, smemKvProdToken);
      }
      //
      // smemKv [ProdWork (call 5), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1477
      ptrSmemPageOffsetsK3 = ptrSmemPageOffsetsK5;
      // Task.cpp:1477
      ptrSmemPageOffsetsV3 = ptrSmemPageOffsetsV5;
      // Task.cpp:1573
      // Task.cpp:5038
      {
        // Task.cpp:4300
        uint64_t* barrier{smemKvDstStack.mPipeline.producer_get_barrier(smemKvProdState)};
        // Task.cpp:5829
        int32_t index{smemKvProdState.index()};
        // SmemKv.h:631
        int32_t adjustedGlobalTileIdx{(mCtaIdxKv) * (numLoopSteps) + (loopOffset723)};
        // SmemKv.h:1404
        int32_t headDimOffset{int32_t{128}};
        // SmemKv.h:1529
        int32_t tokenOffset{int32_t{0}};
        // SmemKv.h:1652

        //
        // Load pageOffsets for headDimStageIdx = 1.
        //
        // SmemKv.h:1213
        {
          // SmemTile.cpp:484
          int32_t coords[4];
          // SmemTile.cpp:491
          coords[int32_t{0}] = headDimOffset;
          // SmemTile.cpp:491
          coords[int32_t{1}] = tokenOffset;
          // SmemTile.cpp:491
          coords[int32_t{2}] = mHeadIdx;
          // SmemTile.cpp:491
          coords[int32_t{3}] = pageOffsets3[int32_t{0}];
          // SmemTile.cpp:610
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{0}],
                                           &params.tmaV_,
                                           coords,
                                           barrier);
          }
        }
        // SmemKv.h:1213
        {
          // SmemTile.cpp:484
          int32_t coords[4];
          // SmemTile.cpp:491
          coords[int32_t{0}] = headDimOffset;
          // SmemTile.cpp:491
          coords[int32_t{1}] = tokenOffset;
          // SmemTile.cpp:491
          coords[int32_t{2}] = mHeadIdx;
          // SmemTile.cpp:491
          coords[int32_t{3}] = pageOffsets3[int32_t{1}];
          // SmemTile.cpp:610
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{8192}],
                                           &params.tmaV_,
                                           coords,
                                           barrier);
          }
        }
      }
      //
      // smemKv [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:4409
      {
        // Task.cpp:4427
        {
          // Task.cpp:4443
          smemKvDstStack.mPipeline.producer_commit(smemKvProdState);
        }
        // Task.cpp:43
        ++smemKvProdState;
      }
      //
      // smemPageOffsetsKv [ConsRelease, Info{0}, FreqInfo{0, 16}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:3772
      if (((loopOffset723) % (int32_t{16})) == (int32_t{15})) {
        // Task.cpp:2533
        if ((loopOffset723) >= (int32_t{0})) {
          // Task.cpp:2561
          {
            // Task.cpp:2585
            smemPageOffsetsKvSrcStack.mPipeline.consumer_release(smemPageOffsetsKvConsReleaseState);
          }
          // Task.cpp:43
          ++smemPageOffsetsKvConsReleaseState;
        }
      }
      // Task.cpp:3457
      lastLoopOffset = loopOffset723;
    }
    //
    // Pull the last iter down.
    //
    // Task.cpp:3492
    if (((numLoopSteps) - (int32_t{1})) > (int32_t{0})) {
      // Task.cpp:3493
      ++lastLoopOffset;
    }
    //
    // smemPageOffsetsKv [ConsRelease, LastIter, FreqInfo{0, 16}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:2561
      {
        // Task.cpp:2585
        smemPageOffsetsKvSrcStack.mPipeline.consumer_release(smemPageOffsetsKvConsReleaseState);
      }
      // Task.cpp:43
      ++smemPageOffsetsKvConsReleaseState;
    }
    //
    // smemPageOffsetsKv [ConsWait, LastIter, FreqInfo{0, 16}, UserTags{2}, Flags{0}].
    //
    // Task.cpp:3772
    if (((lastLoopOffset) % (int32_t{16})) == (int32_t{0})) {
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:1573
        if (hasOneLoopIter) {
          // Task.cpp:2745
          smemPageOffsetsKvConsToken =
            smemPageOffsetsKvSrcStack.mPipeline.consumer_try_wait(smemPageOffsetsKvConsState);
        }
        // Task.cpp:2813
        smemPageOffsetsKvSrcStack.mPipeline.consumer_wait(smemPageOffsetsKvConsState,
                                                          smemPageOffsetsKvConsToken);
      }
    }
    //
    // smemPageOffsetsKv [ConsWork (call 3), LastIter, FreqInfo{0, 16}, UserTags{2}, Flags{0}].
    //
    // Task.cpp:3772
    if (((lastLoopOffset) % (int32_t{16})) == (int32_t{0})) {
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:5829
        int32_t index{smemPageOffsetsKvConsState.index()};
        // SmemPageOffsetsKv.h:349
        ptrSmemPageOffsetsV5 = smemPageOffsetsKvSrcSmem.mArray[index];
        // Task.cpp:43
        ++smemPageOffsetsKvConsState;
      }
    }
    //
    // smemKv [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4962
      if ((lastLoopOffset) >= (int32_t{0})) {
        // Task.cpp:4984
        smemKvProdToken = smemKvDstStack.mPipeline.producer_try_acquire(smemKvProdState);
      }
    }
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4210
      smemKvDstStack.mPipeline.producer_acquire(smemKvProdState, smemKvProdToken);
    }
    //
    // smemKv [ProdWork (call 6), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1477
    ptrSmemPageOffsetsK3 = ptrSmemPageOffsetsK5;
    // Task.cpp:1477
    ptrSmemPageOffsetsV3 = ptrSmemPageOffsetsV5;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4300
      uint64_t* barrier{smemKvDstStack.mPipeline.producer_get_barrier(smemKvProdState)};
      // Task.cpp:5829
      int32_t index{smemKvProdState.index()};
      // SmemKv.h:631
      int32_t adjustedGlobalTileIdx{(mCtaIdxKv) * (numLoopSteps) + (lastLoopOffset)};
      // SmemKv.h:1404
      int32_t headDimOffset{int32_t{0}};
      // SmemKv.h:1529
      int32_t tokenOffset{int32_t{0}};
      // SmemKv.h:1652
      {
        // SmemKv.h:1669
        cutlass::AlignedArray<int32_t, 2> localPageOffsets03;
        // SmemKv.h:1685
        localPageOffsets03 = reinterpret_cast<cutlass::AlignedArray<int32_t, 2>*>(
          (ptrSmemPageOffsetsV3 + ((lastLoopOffset) * (int32_t{2})) % (int32_t{32})))[int32_t{0}];
        // SmemKv.h:1705
        pageOffsets3[int32_t{0}] = localPageOffsets03[int32_t{0}];
        // SmemKv.h:1705
        pageOffsets3[int32_t{1}] = localPageOffsets03[int32_t{1}];
      }
      //
      // Load pageOffsets for headDimStageIdx = 0.
      //
      // SmemKv.h:1213
      {
        // SmemTile.cpp:484
        int32_t coords[4];
        // SmemTile.cpp:491
        coords[int32_t{0}] = headDimOffset;
        // SmemTile.cpp:491
        coords[int32_t{1}] = tokenOffset;
        // SmemTile.cpp:491
        coords[int32_t{2}] = mHeadIdx;
        // SmemTile.cpp:491
        coords[int32_t{3}] = pageOffsets3[int32_t{0}];
        // SmemTile.cpp:610
        if (bool{cute::elect_one_sync()}) {
          // CudaPtx.h:48
          cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                         cuda_ptx::space_global_t{},
                                         &smemKvDstSmem.mArray[index][int32_t{0}],
                                         &params.tmaV_,
                                         coords,
                                         barrier);
        }
      }
      // SmemKv.h:1213
      {
        // SmemTile.cpp:484
        int32_t coords[4];
        // SmemTile.cpp:491
        coords[int32_t{0}] = headDimOffset;
        // SmemTile.cpp:491
        coords[int32_t{1}] = tokenOffset;
        // SmemTile.cpp:491
        coords[int32_t{2}] = mHeadIdx;
        // SmemTile.cpp:491
        coords[int32_t{3}] = pageOffsets3[int32_t{1}];
        // SmemTile.cpp:610
        if (bool{cute::elect_one_sync()}) {
          // CudaPtx.h:48
          cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                         cuda_ptx::space_global_t{},
                                         &smemKvDstSmem.mArray[index][int32_t{8192}],
                                         &params.tmaV_,
                                         coords,
                                         barrier);
        }
      }
    }
    //
    // smemKv [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4427
      {
        // Task.cpp:4443
        smemKvDstStack.mPipeline.producer_commit(smemKvProdState);
      }
      // Task.cpp:43
      ++smemKvProdState;
    }
    //
    // smemKv [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4962
      if ((lastLoopOffset) >= (int32_t{0})) {
        // Task.cpp:4984
        smemKvProdToken = smemKvDstStack.mPipeline.producer_try_acquire(smemKvProdState);
      }
    }
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4210
      smemKvDstStack.mPipeline.producer_acquire(smemKvProdState, smemKvProdToken);
    }
    //
    // smemKv [ProdWork (call 7), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1477
    ptrSmemPageOffsetsK3 = ptrSmemPageOffsetsK5;
    // Task.cpp:1477
    ptrSmemPageOffsetsV3 = ptrSmemPageOffsetsV5;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4300
      uint64_t* barrier{smemKvDstStack.mPipeline.producer_get_barrier(smemKvProdState)};
      // Task.cpp:5829
      int32_t index{smemKvProdState.index()};
      // SmemKv.h:631
      int32_t adjustedGlobalTileIdx{(mCtaIdxKv) * (numLoopSteps) + (lastLoopOffset)};
      // SmemKv.h:1404
      int32_t headDimOffset{int32_t{128}};
      // SmemKv.h:1529
      int32_t tokenOffset{int32_t{0}};
      // SmemKv.h:1652

      //
      // Load pageOffsets for headDimStageIdx = 1.
      //
      // SmemKv.h:1213
      {
        // SmemTile.cpp:484
        int32_t coords[4];
        // SmemTile.cpp:491
        coords[int32_t{0}] = headDimOffset;
        // SmemTile.cpp:491
        coords[int32_t{1}] = tokenOffset;
        // SmemTile.cpp:491
        coords[int32_t{2}] = mHeadIdx;
        // SmemTile.cpp:491
        coords[int32_t{3}] = pageOffsets3[int32_t{0}];
        // SmemTile.cpp:610
        if (bool{cute::elect_one_sync()}) {
          // CudaPtx.h:48
          cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                         cuda_ptx::space_global_t{},
                                         &smemKvDstSmem.mArray[index][int32_t{0}],
                                         &params.tmaV_,
                                         coords,
                                         barrier);
        }
      }
      // SmemKv.h:1213
      {
        // SmemTile.cpp:484
        int32_t coords[4];
        // SmemTile.cpp:491
        coords[int32_t{0}] = headDimOffset;
        // SmemTile.cpp:491
        coords[int32_t{1}] = tokenOffset;
        // SmemTile.cpp:491
        coords[int32_t{2}] = mHeadIdx;
        // SmemTile.cpp:491
        coords[int32_t{3}] = pageOffsets3[int32_t{1}];
        // SmemTile.cpp:610
        if (bool{cute::elect_one_sync()}) {
          // CudaPtx.h:48
          cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                         cuda_ptx::space_global_t{},
                                         &smemKvDstSmem.mArray[index][int32_t{8192}],
                                         &params.tmaV_,
                                         coords,
                                         barrier);
        }
      }
    }
    //
    // smemKv [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4427
      {
        // Task.cpp:4443
        smemKvDstStack.mPipeline.producer_commit(smemKvProdState);
      }
      // Task.cpp:43
      ++smemKvProdState;
    }
    //
    // smemPageOffsetsKv [ConsRelease, LastIter, FreqInfo{0, 16}, UserTags{2}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:2561
      {
        // Task.cpp:2585
        smemPageOffsetsKvSrcStack.mPipeline.consumer_release(smemPageOffsetsKvConsReleaseState);
      }
      // Task.cpp:43
      ++smemPageOffsetsKvConsReleaseState;
    }
  //
  // Tail work.
  //
  // Task.cpp:3511
  ExitTileWithSignalingLabel:
  // Task.cpp:3518
  ExitTileWithoutSignalingLabel:
    // Task.cpp:3528
    {}
  }
};
// Task.cpp:544
// Fmha.h:2177
struct SoftmaxTask0 {
  // Task.cpp:283
  int32_t mCtaIdxX;
  // Task.cpp:287
  int32_t mCtaIdxY;
  // Task.cpp:291
  int32_t mCtaIdxZ;
  // FmhaTask.h:220
  int32_t mHeadIdx;
  // FmhaTask.h:218
  int32_t mBatchIdx;
  // FmhaTask.h:208
  int32_t mSeqOffsetQ;
  // FmhaTask.h:210
  int32_t mSeqLenQ;
  // FmhaTask.h:224
  int32_t mCtaIdxQ;
  // FmhaTask.h:226
  int32_t mCtaIdxKv;
  // FmhaTask.h:214
  int32_t mSeqLenKv;
  // FmhaTask.h:216
  int32_t mNumCtasKv;
  // Task.cpp:371
  int32_t const mWarpGrpThreadIdx;
  // TmemTile.cpp:422
  int32_t const mLdtm16dp256bitTmemColIdx;
  // TmemTile.cpp:445
  int32_t const mLdtm16dp256bitTmemRowIdx;
  // Task.cpp:691
  uint32_t const mTmemBaseOffset;
  // Task.cpp:551
  inline __device__ SoftmaxTask0(fmha::KernelParams const& params,
                                 KernelState const& state,
                                 int32_t warpGrpStart)
    : // Kernel.cpp:194
    mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , // Kernel.cpp:195
    mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , // Kernel.cpp:196
    mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)}
    , // Task.cpp:287
    mHeadIdx{mCtaIdxY}
    , // Task.cpp:291
    mBatchIdx{mCtaIdxZ}
    , // FmhaTask.h:394
    mSeqOffsetQ{int32_t(bool{params.ptrCumSeqLensQ == nullptr})
                  ? ((mBatchIdx) * (params.mMaxSeqLenQ))
                  : (int32_t{params.ptrCumSeqLensQ[mBatchIdx]})}
    , // FmhaTask.h:410
    mSeqLenQ{int32_t(bool{params.ptrCumSeqLensQ == nullptr})
               ? (params.mMaxSeqLenQ)
               : ((int32_t{params.ptrCumSeqLensQ[(mBatchIdx) + (int32_t{1})]}) - (mSeqOffsetQ))}
    , // FmhaTask.h:516
    mCtaIdxQ{(mCtaIdxX) / (params.mMaxNumCtasKv)}
    , // FmhaTask.h:517
    mCtaIdxKv{(mCtaIdxX) % (params.mMaxNumCtasKv)}
    , // FmhaTask.h:543
    mSeqLenKv{(int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                             ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                             : (mBatchIdx)]}) -
              (((mSeqLenQ) - (int32_t{1})) - (mCtaIdxQ))}
    , // FmhaTask.h:565
    mNumCtasKv{
      int32_t{min(int32_t{((mSeqLenKv) + (int32_t{127})) / (int32_t{128})}, params.mMaxNumCtasKv)}}
    , // Task.cpp:379
    mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))}
    , // TmemTile.cpp:432
    mLdtm16dp256bitTmemColIdx{
      trtllm::dev::ldst16dp256bitTmemColIdx((mWarpGrpThreadIdx) % (int32_t{128}))}
    , // TmemTile.cpp:453
    mLdtm16dp256bitTmemRowIdx{
      trtllm::dev::ldst16dp256bitTmemRowIdx<int32_t{32}>((mWarpGrpThreadIdx) % (int32_t{128}))}
    , // Kernel.cpp:2420
    mTmemBaseOffset{uint32_t{
      __shfl_sync(uint32_t{0xffffffff}, (*state.mTmemSwStatePtr), int32_t{0}, int32_t{32})}} {}
  // Task.cpp:507
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:532
    return ((state.mWarpIdx) >= (int32_t{0})) && ((state.mWarpIdx) < (int32_t{4}));
  }
  // Task.cpp:454
  inline __device__ void execute(fmha::KernelParams const& params,
                                 KernelState const& state,
                                 TmemSoftmaxLocal0Stack& tmemSoftmaxLocal0DstStack,
                                 TmemSoftmaxGlobal0Stack& tmemSoftmaxGlobal0DstStack,
                                 TmemP0Stack& tmemP0DstStack,
                                 TmemS0Stack& tmemS0SrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_inc(cuda_ptx::n32_t<184>{});
    // Task.cpp:2079
    trtllm::dev::CutlassUmmaAsyncPipeline<
      2,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState tmemS0ConsState{};
    // Task.cpp:2086
    trtllm::dev::CutlassUmmaAsyncPipeline<2,
                                          cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState tmemS0ConsReleaseState{};
    // Task.cpp:2100
    int32_t tmemS0ConsToken{int32_t{0}};
    // Task.cpp:1979
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState tmemSoftmaxLocal0ProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    // Task.cpp:1999
    int32_t tmemSoftmaxLocal0ProdToken{int32_t{1}};
    // Task.cpp:1979
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState tmemP0ProdState{int32_t{0},
                                                                                int32_t{1},
                                                                                int32_t{0}};
    // Task.cpp:1999
    int32_t tmemP0ProdToken{int32_t{1}};
    // FmhaTask.h:582
    int32_t numLoopSteps;
    // FmhaTask.h:630
    if (((mCtaIdxQ) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
      // FmhaTask.h:668
      int32_t numSteps{((mSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
                       ((mNumCtasKv) * (int32_t{128}))};
      // FmhaTask.h:682
      numLoopSteps = numSteps;
    } else {
      // FmhaTask.h:651
      return;
    }
    // Task.cpp:3168
    bool const hasOneLoopIter{(int32_t{0}) < (numLoopSteps)};
    // TmemS.h:647
    float oldMaxArray10[4];
    // TmemS.h:653
    float sumArray10[4]{float{0}, float{0}, float{0}, float{0}};
    // TmemS.h:665
    float newMaxArray10[4]{float{-3.4028235e+38},
                           float{-3.4028235e+38},
                           float{-3.4028235e+38},
                           float{-3.4028235e+38}};
    // TmemTile.cpp:373
    cutlass::Array<float, 16> regsQk;
    // TmemS.h:1354
    uint32_t uint32NegFltMax10{trtllm::dev::floatToUInt32ForAtomicMax(float{-3.4028235e+38})};
    // TmemS.h:1367
    CUTLASS_PRAGMA_UNROLL
    for (int32_t loopOffset1239 = mWarpGrpThreadIdx; loopOffset1239 < int32_t{16};
         loopOffset1239 += int32_t{128}) {
      // TmemS.h:1374
      reinterpret_cast<uint32_t*>(tmemS0SrcStack.mDepSmemPtr7)[loopOffset1239] = uint32NegFltMax10;
    }
    // TmemS.h:717
    trtllm::dev::CutlassNamedBarrier::sync(128, tmemS0SrcStack.mNamedBarId);
    // TmemSoftmax.h:515
    cudaGridDependencySynchronize();
    // TmemSoftmax.h:524
    float scaleSoftmaxLog212;
    // TmemSoftmax.h:529
    scaleSoftmaxLog212 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
                           ? (params.mScaleSoftmaxLog2)
                           : (float{params.ptrScaleSoftmaxLog2[int32_t{0}]});
    // TmemP.h:507
    uint32_t regsP[8];
    // TmemP.h:520
    cudaGridDependencySynchronize();
    // TmemP.h:527
    float scaleSoftmaxLog214;
    // TmemP.h:532
    scaleSoftmaxLog214 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
                           ? (params.mScaleSoftmaxLog2)
                           : (float{params.ptrScaleSoftmaxLog2[int32_t{0}]});
    //
    // Hoist the first iter.
    //
    //
    // tmemP0 [ProdWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{262656}].
    //
    //
    // tmemSoftmaxLocal0 [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{8}, Flags{1024}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4962
      {
        // Task.cpp:4984
        tmemSoftmaxLocal0ProdToken =
          tmemSoftmaxLocal0DstStack.mPipeline.producer_try_acquire(tmemSoftmaxLocal0ProdState);
      }
    }
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4210
      tmemSoftmaxLocal0DstStack.mPipeline.producer_acquire(tmemSoftmaxLocal0ProdState,
                                                           tmemSoftmaxLocal0ProdToken);
    }
    //
    // Loop body.
    //
    // Task.cpp:3350
    CUTLASS_PRAGMA_NO_UNROLL
    for (int32_t loopOffset1265 = int32_t{0}; loopOffset1265 < numLoopSteps; ++loopOffset1265) {
      // Task.cpp:3423
      bool const isLastLoopIter{((loopOffset1265) + (int32_t{1})) >= (numLoopSteps)};
      //
      // tmemS0 [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{8}, Flags{256}].
      //
      // Task.cpp:1573
      // Task.cpp:2781
      {
        // Task.cpp:1573
        // Task.cpp:2722
        {
          // Task.cpp:2745
          tmemS0ConsToken = tmemS0SrcStack.mPipeline.consumer_try_wait(tmemS0ConsState);
        }
        // Task.cpp:2813
        tmemS0SrcStack.mPipeline.consumer_wait(tmemS0ConsState, tmemS0ConsToken);
      }
      //
      // tmemS0 [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{8}, Flags{256}].
      //
      // TmemS.h:1053
      float* oldMaxPtr10;
      // TmemS.h:1058
      float* sumPtr10;
      // TmemS.h:1063
      float* newMaxPtr010;
      // TmemS.h:1068
      float* qkPtr010;
      // TmemS.h:1073
      float* newMaxPtr110;
      // TmemS.h:1078
      float* qkPtr110;
      // Task.cpp:1573
      // Task.cpp:2893
      {
        // Task.cpp:5829
        int32_t index{tmemS0ConsState.index()};
        // TmemS.h:1185
        oldMaxPtr10 = oldMaxArray10;
        // TmemS.h:1187
        sumPtr10 = sumArray10;
        // TmemS.h:1189
        newMaxPtr010 = newMaxArray10;
        // TmemS.h:1191
        newMaxPtr110 = newMaxArray10;
        // TmemS.h:1239
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset1286 = int32_t{0}; loopOffset1286 < int32_t{4}; ++loopOffset1286) {
          // TmemS.h:1250
          oldMaxArray10[loopOffset1286] = newMaxArray10[loopOffset1286];
        }
        //
        // The dense mask block.
        //
        // Mask.h:1760
        bool const allTilesAreCompleteK{((mSeqLenKv) % (int32_t{128})) == (int32_t{0})};
        // Mask.h:568
        int32_t const tileOffsetK{
          (((numLoopSteps) * (mCtaIdxKv) + (loopOffset1265)) * (int32_t{1}) + (int32_t{1})) *
          (int32_t{128})};
        // Mask.h:1824
        if ((tileOffsetK) <= (mSeqLenKv)) {
          // TmemTile.cpp:527
          {
            // TmemTile.cpp:529
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:545
            uint32_t(&dstSlice0)[8]{reinterpret_cast<uint32_t(&)[8]>(regsQk[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_16x256b(
              dstSlice0,
              (tmemBasePtr) +
                (static_cast<uint32_t>((index) * (int32_t{16}) +
                                       (int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                          ? (int32_t{0})
                                          : (int32_t{16})))));
            // TmemTile.cpp:545
            uint32_t(&dstSlice1)[8]{reinterpret_cast<uint32_t(&)[8]>(regsQk[int32_t{8}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_16x256b(
              dstSlice1,
              (tmemBasePtr) +
                (static_cast<uint32_t>(
                  ((index) * (int32_t{16}) + (int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                                ? (int32_t{0})
                                                : (int32_t{16}))) +
                  (int32_t{0x100000 /*hi=16, lo=0*/}))));
          }
          // Utils.h:248
          trtllm::dev::reduceColMax16dp256bit<int32_t{2}, int32_t{1}, int32_t{2}, false>(
            newMaxArray10,
            regsQk);
          // Utils.h:260
          trtllm::dev::reduceColMax(newMaxArray10,
                                    tmemS0SrcStack.mDepSmemPtr7,
                                    int32_t{128},
                                    mWarpGrpThreadIdx,
                                    tmemS0SrcStack.mNamedBarId);
        } else {
          // TmemTile.cpp:527
          {
            // TmemTile.cpp:529
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:545
            uint32_t(&dstSlice0)[8]{reinterpret_cast<uint32_t(&)[8]>(regsQk[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_16x256b(
              dstSlice0,
              (tmemBasePtr) +
                (static_cast<uint32_t>((index) * (int32_t{16}) +
                                       (int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                          ? (int32_t{0})
                                          : (int32_t{16})))));
            // TmemTile.cpp:545
            uint32_t(&dstSlice1)[8]{reinterpret_cast<uint32_t(&)[8]>(regsQk[int32_t{8}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_16x256b(
              dstSlice1,
              (tmemBasePtr) +
                (static_cast<uint32_t>(
                  ((index) * (int32_t{16}) + (int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                                ? (int32_t{0})
                                                : (int32_t{16}))) +
                  (int32_t{0x100000 /*hi=16, lo=0*/}))));
          }
          //
          // Apply the dense mask.
          //
          // Mask.h:568
          int32_t const tileOffsetK{
            (((numLoopSteps) * (mCtaIdxKv) + (loopOffset1265)) * (int32_t{1})) * (int32_t{128})};
          // Mask.h:891
          int32_t localIdxK0{mLdtm16dp256bitTmemRowIdx};
          // Mask.h:902
          if (((tileOffsetK) + (localIdxK0)) >= (mSeqLenKv)) {
            // Mask.h:906
            regsQk[int32_t{0}] = float{-3.4028235e+38};
            // Mask.h:906
            regsQk[int32_t{1}] = float{-3.4028235e+38};
            // Mask.h:906
            regsQk[int32_t{4}] = float{-3.4028235e+38};
            // Mask.h:906
            regsQk[int32_t{5}] = float{-3.4028235e+38};
          }
          // Mask.h:891
          int32_t localIdxK1{(mLdtm16dp256bitTmemRowIdx) + (int32_t{8})};
          // Mask.h:902
          if (((tileOffsetK) + (localIdxK1)) >= (mSeqLenKv)) {
            // Mask.h:906
            regsQk[int32_t{2}] = float{-3.4028235e+38};
            // Mask.h:906
            regsQk[int32_t{3}] = float{-3.4028235e+38};
            // Mask.h:906
            regsQk[int32_t{6}] = float{-3.4028235e+38};
            // Mask.h:906
            regsQk[int32_t{7}] = float{-3.4028235e+38};
          }
          // Mask.h:891
          int32_t localIdxK2{(mLdtm16dp256bitTmemRowIdx) + (int32_t{16})};
          // Mask.h:902
          if (((tileOffsetK) + (localIdxK2)) >= (mSeqLenKv)) {
            // Mask.h:906
            regsQk[int32_t{8}] = float{-3.4028235e+38};
            // Mask.h:906
            regsQk[int32_t{9}] = float{-3.4028235e+38};
            // Mask.h:906
            regsQk[int32_t{12}] = float{-3.4028235e+38};
            // Mask.h:906
            regsQk[int32_t{13}] = float{-3.4028235e+38};
          }
          // Mask.h:891
          int32_t localIdxK3{(mLdtm16dp256bitTmemRowIdx) + (int32_t{24})};
          // Mask.h:902
          if (((tileOffsetK) + (localIdxK3)) >= (mSeqLenKv)) {
            // Mask.h:906
            regsQk[int32_t{10}] = float{-3.4028235e+38};
            // Mask.h:906
            regsQk[int32_t{11}] = float{-3.4028235e+38};
            // Mask.h:906
            regsQk[int32_t{14}] = float{-3.4028235e+38};
            // Mask.h:906
            regsQk[int32_t{15}] = float{-3.4028235e+38};
          }
          // Utils.h:248
          trtllm::dev::reduceColMax16dp256bit<int32_t{2}, int32_t{1}, int32_t{2}, false>(
            newMaxArray10,
            regsQk);
          // Utils.h:260
          trtllm::dev::reduceColMax(newMaxArray10,
                                    tmemS0SrcStack.mDepSmemPtr7,
                                    int32_t{128},
                                    mWarpGrpThreadIdx,
                                    tmemS0SrcStack.mNamedBarId);
        }
        // TmemS.h:1327
        qkPtr010 = &regsQk[int32_t{0}];
        // TmemS.h:1329
        qkPtr110 = &regsQk[int32_t{0}];
        // Task.cpp:43
        ++tmemS0ConsState;
      }
      //
      // tmemSoftmaxLocal0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{8}, Flags{0}].
      //
      // TmemSoftmax.h:261
      float* oldMaxPtr11;
      // TmemSoftmax.h:267
      float* sumPtr11;
      // TmemSoftmax.h:273
      float* newMaxPtr11;
      // Task.cpp:1477
      oldMaxPtr11 = oldMaxPtr10;
      // Task.cpp:1477
      sumPtr11 = sumPtr10;
      // Task.cpp:1477
      newMaxPtr11 = newMaxPtr010;
      // Task.cpp:1573
      // Task.cpp:5038
      {
        // Task.cpp:5829
        int32_t index{tmemSoftmaxLocal0ProdState.index()};
        // TmemTile.cpp:373
        cutlass::Array<float, 8> stats;
        // TmemSoftmax.h:365
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset1358 = int32_t{0}; loopOffset1358 < int32_t{4}; ++loopOffset1358) {
          // TmemSoftmax.h:382
          stats[loopOffset1358] = oldMaxPtr11[loopOffset1358];
          // TmemSoftmax.h:384
          stats[(loopOffset1358) + (int32_t{4})] = newMaxPtr11[loopOffset1358];
        }
        // TmemTile.cpp:824
        {
          // TmemTile.cpp:826
          uint32_t tmemBasePtr{mTmemBaseOffset};
          // TmemTile.cpp:859
          uint32_t const(&srcSlice0)[8]{reinterpret_cast<uint32_t const(&)[8]>(stats[int32_t{0}])};
          // CudaPtx.h:48
          cuda_ptx::tcgen05_st_32x32b(
            (tmemBasePtr) +
              (static_cast<uint32_t>((index) * (int32_t{32}) +
                                     (int32_t((tmemSoftmaxLocal0DstStack.mInstId) == (int32_t{0}))
                                        ? (int32_t{32})
                                        : (int32_t{64})))),
            srcSlice0);
        }
        // TmemSoftmax.h:407
        cutlass::arch::fence_view_async_tmem_store();
      }
      //
      // tmemSoftmaxLocal0 [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{8}, Flags{0}].
      //
      // Task.cpp:4409
      {
        // Task.cpp:4427
        {
          // Task.cpp:4443
          tmemSoftmaxLocal0DstStack.mPipeline.producer_commit(tmemSoftmaxLocal0ProdState);
        }
        // Task.cpp:43
        ++tmemSoftmaxLocal0ProdState;
      }
      //
      // tmemP0 [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{6}, Flags{3145856}].
      //
      //
      // Skipped by flag SkipsProdAcquire.
      //
      //
      // tmemP0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{6}, Flags{3145856}].
      //
      // TmemP.h:555
      float* newMaxPtr14;
      // TmemP.h:560
      float* regsFp32P14;
      // Task.cpp:1477
      newMaxPtr14 = newMaxPtr110;
      // Task.cpp:1477
      regsFp32P14 = qkPtr110;
      // Task.cpp:1573
      // Task.cpp:5038
      {
        // Task.cpp:5829
        int32_t index{tmemP0ProdState.index()};
        // TmemP.h:1011
        float negScaledMaxArray[4];
        // TmemP.h:1029
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset1386 = int32_t{0}; loopOffset1386 < int32_t{4};
             loopOffset1386 += int32_t{2}) {
          // TmemP.h:1040
          float newMax0{newMaxPtr14[loopOffset1386]};
          // TmemP.h:1046
          float newMax1{newMaxPtr14[(loopOffset1386) + (int32_t{1})]};
          // Common.h:562
          if ((newMax0) == (float{-3.4028235e+38})) {
            // Common.h:564
            newMax0 = float{0};
          }
          // Common.h:562
          if ((newMax1) == (float{-3.4028235e+38})) {
            // Common.h:564
            newMax1 = float{0};
          }
          // Common.h:353
          cutlass::Array<float, 2> newMax2{newMax0, newMax1};
          // TmemP.h:1065
          float negLog2Scale{-(scaleSoftmaxLog214)};
          // Common.h:353
          cutlass::Array<float, 2> negLog2Scale2{negLog2Scale, negLog2Scale};
          // TmemP.h:1090
          newMax2 = trtllm::dev::fmul2(newMax2, negLog2Scale2);
          // TmemP.h:1101
          negScaledMaxArray[loopOffset1386] = newMax2[int32_t{0}];
          // TmemP.h:1102
          negScaledMaxArray[(loopOffset1386) + (int32_t{1})] = newMax2[int32_t{1}];
        }
        // TmemP.h:1597
        {
          // TmemP.h:1600
          cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
          // TmemP.h:750
          cutlass::Array<float, 2> vals;
          // TmemP.h:769
          cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                negScaledMaxArray[int32_t{1}]};
          // TmemP.h:778
          vals[int32_t{0}] = regsFp32P14[int32_t{0}];
          // TmemP.h:779
          vals[int32_t{1}] = regsFp32P14[int32_t{1}];
          // TmemP.h:787
          vals[int32_t{0}] =
            (log2Scale2[int32_t{0}]) * (vals[int32_t{0}]) + (negScaledMax[int32_t{0}]);
          // TmemP.h:796
          vals[int32_t{1}] =
            (log2Scale2[int32_t{1}]) * (vals[int32_t{1}]) + (negScaledMax[int32_t{1}]);
          // TmemP.h:819
          regsFp32P14[int32_t{0}] = vals[int32_t{0}];
          // TmemP.h:820
          regsFp32P14[int32_t{1}] = vals[int32_t{1}];
        }
        // TmemP.h:1597
        {
          // TmemP.h:1600
          cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
          // TmemP.h:750
          cutlass::Array<float, 2> vals;
          // TmemP.h:769
          cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                negScaledMaxArray[int32_t{1}]};
          // TmemP.h:778
          vals[int32_t{0}] = regsFp32P14[int32_t{2}];
          // TmemP.h:779
          vals[int32_t{1}] = regsFp32P14[int32_t{3}];
          // TmemP.h:812
          vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
          // TmemP.h:819
          regsFp32P14[int32_t{2}] = vals[int32_t{0}];
          // TmemP.h:820
          regsFp32P14[int32_t{3}] = vals[int32_t{1}];
        }
        // TmemP.h:1716
        regsFp32P14[int32_t{0}] = exp2f(regsFp32P14[int32_t{0}]);
        // TmemP.h:1745
        {
          // TmemP.h:1749
          cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
          // TmemP.h:750
          cutlass::Array<float, 2> vals;
          // TmemP.h:769
          cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{2}],
                                                negScaledMaxArray[int32_t{3}]};
          // TmemP.h:778
          vals[int32_t{0}] = regsFp32P14[int32_t{4}];
          // TmemP.h:779
          vals[int32_t{1}] = regsFp32P14[int32_t{5}];
          // TmemP.h:812
          vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
          // TmemP.h:819
          regsFp32P14[int32_t{4}] = vals[int32_t{0}];
          // TmemP.h:820
          regsFp32P14[int32_t{5}] = vals[int32_t{1}];
        }
        // TmemP.h:1786
        regsFp32P14[int32_t{1}] = exp2f(regsFp32P14[int32_t{1}]);
        // TmemP.h:1716
        regsFp32P14[int32_t{2}] = exp2f(regsFp32P14[int32_t{2}]);
        // TmemP.h:1745
        {
          // TmemP.h:1749
          cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
          // TmemP.h:750
          cutlass::Array<float, 2> vals;
          // TmemP.h:769
          cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{2}],
                                                negScaledMaxArray[int32_t{3}]};
          // TmemP.h:778
          vals[int32_t{0}] = regsFp32P14[int32_t{6}];
          // TmemP.h:779
          vals[int32_t{1}] = regsFp32P14[int32_t{7}];
          // TmemP.h:812
          vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
          // TmemP.h:819
          regsFp32P14[int32_t{6}] = vals[int32_t{0}];
          // TmemP.h:820
          regsFp32P14[int32_t{7}] = vals[int32_t{1}];
        }
        // TmemP.h:1786
        regsFp32P14[int32_t{3}] = exp2f(regsFp32P14[int32_t{3}]);
        // TmemP.h:1716
        regsFp32P14[int32_t{4}] = exp2f(regsFp32P14[int32_t{4}]);
        // TmemP.h:1745
        {
          // TmemP.h:1749
          cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
          // TmemP.h:750
          cutlass::Array<float, 2> vals;
          // TmemP.h:769
          cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                negScaledMaxArray[int32_t{1}]};
          // TmemP.h:778
          vals[int32_t{0}] = regsFp32P14[int32_t{8}];
          // TmemP.h:779
          vals[int32_t{1}] = regsFp32P14[int32_t{9}];
          // TmemP.h:812
          vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
          // TmemP.h:819
          regsFp32P14[int32_t{8}] = vals[int32_t{0}];
          // TmemP.h:820
          regsFp32P14[int32_t{9}] = vals[int32_t{1}];
        }
        // TmemP.h:1786
        regsFp32P14[int32_t{5}] = exp2f(regsFp32P14[int32_t{5}]);
        // TmemP.h:1716
        regsFp32P14[int32_t{6}] = exp2f(regsFp32P14[int32_t{6}]);
        // TmemP.h:1745
        {
          // TmemP.h:1749
          cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
          // TmemP.h:750
          cutlass::Array<float, 2> vals;
          // TmemP.h:769
          cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                negScaledMaxArray[int32_t{1}]};
          // TmemP.h:778
          vals[int32_t{0}] = regsFp32P14[int32_t{10}];
          // TmemP.h:779
          vals[int32_t{1}] = regsFp32P14[int32_t{11}];
          // TmemP.h:812
          vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
          // TmemP.h:819
          regsFp32P14[int32_t{10}] = vals[int32_t{0}];
          // TmemP.h:820
          regsFp32P14[int32_t{11}] = vals[int32_t{1}];
        }
        // TmemP.h:1786
        regsFp32P14[int32_t{7}] = exp2f(regsFp32P14[int32_t{7}]);
        // TmemP.h:1716
        regsFp32P14[int32_t{8}] = exp2f(regsFp32P14[int32_t{8}]);
        // TmemP.h:1745
        {
          // TmemP.h:1749
          cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
          // TmemP.h:750
          cutlass::Array<float, 2> vals;
          // TmemP.h:769
          cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{2}],
                                                negScaledMaxArray[int32_t{3}]};
          // TmemP.h:778
          vals[int32_t{0}] = regsFp32P14[int32_t{12}];
          // TmemP.h:779
          vals[int32_t{1}] = regsFp32P14[int32_t{13}];
          // TmemP.h:812
          vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
          // TmemP.h:819
          regsFp32P14[int32_t{12}] = vals[int32_t{0}];
          // TmemP.h:820
          regsFp32P14[int32_t{13}] = vals[int32_t{1}];
        }
        // TmemP.h:1786
        regsFp32P14[int32_t{9}] = exp2f(regsFp32P14[int32_t{9}]);
        // TmemP.h:1716
        regsFp32P14[int32_t{10}] = exp2f(regsFp32P14[int32_t{10}]);
        // TmemP.h:1745
        {
          // TmemP.h:1749
          cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
          // TmemP.h:750
          cutlass::Array<float, 2> vals;
          // TmemP.h:769
          cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{2}],
                                                negScaledMaxArray[int32_t{3}]};
          // TmemP.h:778
          vals[int32_t{0}] = regsFp32P14[int32_t{14}];
          // TmemP.h:779
          vals[int32_t{1}] = regsFp32P14[int32_t{15}];
          // TmemP.h:812
          vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
          // TmemP.h:819
          regsFp32P14[int32_t{14}] = vals[int32_t{0}];
          // TmemP.h:820
          regsFp32P14[int32_t{15}] = vals[int32_t{1}];
        }
        // TmemP.h:1786
        regsFp32P14[int32_t{11}] = exp2f(regsFp32P14[int32_t{11}]);
        // TmemP.h:1692
        {
          // TmemP.h:700
          float elt0;
          // TmemP.h:701
          float elt1;
          // TmemP.h:702
          float elt2;
          // TmemP.h:703
          float elt3;
          // TmemP.h:706
          elt0 = regsFp32P14[int32_t{0}];
          // TmemP.h:707
          elt1 = regsFp32P14[int32_t{1}];
          // TmemP.h:708
          elt2 = regsFp32P14[int32_t{2}];
          // TmemP.h:709
          elt3 = regsFp32P14[int32_t{3}];
          // TmemP.h:731
          regsP[int32_t{0}] = trtllm::dev::convert_float2_to_half(elt0, elt1);
          // TmemP.h:731
          regsP[int32_t{1}] = trtllm::dev::convert_float2_to_half(elt2, elt3);
        }
        // TmemP.h:1716
        regsFp32P14[int32_t{12}] = exp2f(regsFp32P14[int32_t{12}]);
        // TmemP.h:1786
        regsFp32P14[int32_t{13}] = exp2f(regsFp32P14[int32_t{13}]);
        // TmemP.h:1716
        regsFp32P14[int32_t{14}] = exp2f(regsFp32P14[int32_t{14}]);
        // TmemP.h:1786
        regsFp32P14[int32_t{15}] = exp2f(regsFp32P14[int32_t{15}]);
        // TmemP.h:1818
        {
          // TmemP.h:700
          float elt0;
          // TmemP.h:701
          float elt1;
          // TmemP.h:702
          float elt2;
          // TmemP.h:703
          float elt3;
          // TmemP.h:706
          elt0 = regsFp32P14[int32_t{4}];
          // TmemP.h:707
          elt1 = regsFp32P14[int32_t{5}];
          // TmemP.h:708
          elt2 = regsFp32P14[int32_t{6}];
          // TmemP.h:709
          elt3 = regsFp32P14[int32_t{7}];
          // TmemP.h:731
          regsP[int32_t{2}] = trtllm::dev::convert_float2_to_half(elt0, elt1);
          // TmemP.h:731
          regsP[int32_t{3}] = trtllm::dev::convert_float2_to_half(elt2, elt3);
        }
        // TmemP.h:1818
        {
          // TmemP.h:700
          float elt0;
          // TmemP.h:701
          float elt1;
          // TmemP.h:702
          float elt2;
          // TmemP.h:703
          float elt3;
          // TmemP.h:706
          elt0 = regsFp32P14[int32_t{8}];
          // TmemP.h:707
          elt1 = regsFp32P14[int32_t{9}];
          // TmemP.h:708
          elt2 = regsFp32P14[int32_t{10}];
          // TmemP.h:709
          elt3 = regsFp32P14[int32_t{11}];
          // TmemP.h:731
          regsP[int32_t{4}] = trtllm::dev::convert_float2_to_half(elt0, elt1);
          // TmemP.h:731
          regsP[int32_t{5}] = trtllm::dev::convert_float2_to_half(elt2, elt3);
        }
        // TmemP.h:1818
        {
          // TmemP.h:700
          float elt0;
          // TmemP.h:701
          float elt1;
          // TmemP.h:702
          float elt2;
          // TmemP.h:703
          float elt3;
          // TmemP.h:706
          elt0 = regsFp32P14[int32_t{12}];
          // TmemP.h:707
          elt1 = regsFp32P14[int32_t{13}];
          // TmemP.h:708
          elt2 = regsFp32P14[int32_t{14}];
          // TmemP.h:709
          elt3 = regsFp32P14[int32_t{15}];
          // TmemP.h:731
          regsP[int32_t{6}] = trtllm::dev::convert_float2_to_half(elt0, elt1);
          // TmemP.h:731
          regsP[int32_t{7}] = trtllm::dev::convert_float2_to_half(elt2, elt3);
        }
        // TmemP.h:1206
        cutlass::half_t* smemPtrP14;
        // TmemP.h:1208
        smemPtrP14 = reinterpret_cast<cutlass::half_t*>(tmemP0DstStack.mDepSmemPtr6) +
                     (((tmemP0DstStack.mInstId) + (index)) * (int32_t{2048}));
        // TmemP.h:1229
        trtllm::dev::storeTransposedSmem128x16b<int32_t{16}>(smemPtrP14, regsP, mWarpGrpThreadIdx);
        // TmemP.h:1231
        cutlass::arch::fence_view_async_shared();
      }
      //
      // tmemP0 [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{6}, Flags{3145856}].
      //
      // Task.cpp:4409
      {
        // Task.cpp:4427
        {
          //
          // Skipped by flag SkipsProdCommit.
          //
        }
        // Task.cpp:43
        ++tmemP0ProdState;
      }
      //
      // tmemSoftmaxLocal0 [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{10}, Flags{1024}].
      //
      // Task.cpp:1573
      // Task.cpp:4948
      {
        // Task.cpp:4962
        if ((loopOffset1265) >= (int32_t{0})) {
          // Task.cpp:4984
          tmemSoftmaxLocal0ProdToken =
            tmemSoftmaxLocal0DstStack.mPipeline.producer_try_acquire(tmemSoftmaxLocal0ProdState);
        }
      }
      // Task.cpp:1573
      // Task.cpp:4180
      {
        // Task.cpp:4210
        tmemSoftmaxLocal0DstStack.mPipeline.producer_acquire(tmemSoftmaxLocal0ProdState,
                                                             tmemSoftmaxLocal0ProdToken);
      }
      //
      // tmemS0 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{8}, Flags{1024}].
      //
      // Task.cpp:2533
      if ((loopOffset1265) >= (int32_t{0})) {
        // Task.cpp:2561
        {
          // Task.cpp:2585
          tmemS0SrcStack.mPipeline.consumer_release(tmemS0ConsReleaseState);
        }
        // Task.cpp:43
        ++tmemS0ConsReleaseState;
      }
      //
      // tmemSoftmaxGlobal0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // TmemSoftmax.h:545
      float* oldMaxPtr12;
      // TmemSoftmax.h:552
      float* sumPtr12;
      // TmemSoftmax.h:559
      float* newMaxPtr12;
      // TmemSoftmax.h:566
      float* pPtr12;
      // Task.cpp:1477
      oldMaxPtr12 = oldMaxPtr10;
      // Task.cpp:1477
      sumPtr12 = sumPtr10;
      // Task.cpp:1477
      newMaxPtr12 = newMaxPtr010;
      // Task.cpp:1477
      pPtr12 = qkPtr010;
      // Task.cpp:1573
      // Task.cpp:5038
      {
        // TmemSoftmax.h:1010
        {
          // Common.h:395
          cutlass::Array<float, 2> oldMax{float{oldMaxPtr12[int32_t{0}]},
                                          float{oldMaxPtr12[int32_t{1}]}};
          // Common.h:395
          cutlass::Array<float, 2> newMax{float{newMaxPtr12[int32_t{0}]},
                                          float{newMaxPtr12[int32_t{1}]}};
          // Common.h:353
          cutlass::Array<float, 2> scale2{float{1}, float{1}};
          // Common.h:133
          newMax[int32_t{0}] = -(newMax[int32_t{0}]);
          // Common.h:134
          newMax[int32_t{1}] = -(newMax[int32_t{1}]);
          // Common.h:137
          cutlass::Array<float, 2> maxDiff2{trtllm::dev::fadd2(oldMax, newMax)};
          // Common.h:155
          if (((maxDiff2[int32_t{0}]) != (float{0})) || ((maxDiff2[int32_t{1}]) != (float{0}))) {
            // Common.h:353
            cutlass::Array<float, 2> scale2_{scaleSoftmaxLog212, scaleSoftmaxLog212};
            // Common.h:161
            scale2 = trtllm::dev::fmul2(scale2_, maxDiff2);
            // Common.h:168
            scale2[int32_t{0}] = exp2f(scale2[int32_t{0}]);
            // Common.h:169
            scale2[int32_t{1}] = exp2f(scale2[int32_t{1}]);
          }
          // TmemSoftmax.h:1029
          cutlass::Array<float, 2> p0{pPtr12[int32_t{0}], pPtr12[int32_t{1}]};
          // Common.h:395
          cutlass::Array<float, 2> sum{float{sumPtr12[int32_t{0}]}, float{sumPtr12[int32_t{1}]}};
          // TmemSoftmax.h:1048
          sum = trtllm::dev::ffma2(scale2, sum, p0);
          // TmemSoftmax.h:1060
          cutlass::Array<float, 2> p1{pPtr12[int32_t{2}], pPtr12[int32_t{3}]};
          // TmemSoftmax.h:1076
          sum = trtllm::dev::fadd2(sum, p1);
          // TmemSoftmax.h:1083
          sumPtr12[int32_t{0}] = sum[int32_t{0}];
          // TmemSoftmax.h:1084
          sumPtr12[int32_t{1}] = sum[int32_t{1}];
          // TmemSoftmax.h:1060
          cutlass::Array<float, 2> p2{pPtr12[int32_t{8}], pPtr12[int32_t{9}]};
          // TmemSoftmax.h:1076
          sum = trtllm::dev::fadd2(sum, p2);
          // TmemSoftmax.h:1083
          sumPtr12[int32_t{0}] = sum[int32_t{0}];
          // TmemSoftmax.h:1084
          sumPtr12[int32_t{1}] = sum[int32_t{1}];
          // TmemSoftmax.h:1060
          cutlass::Array<float, 2> p3{pPtr12[int32_t{10}], pPtr12[int32_t{11}]};
          // TmemSoftmax.h:1076
          sum = trtllm::dev::fadd2(sum, p3);
          // TmemSoftmax.h:1083
          sumPtr12[int32_t{0}] = sum[int32_t{0}];
          // TmemSoftmax.h:1084
          sumPtr12[int32_t{1}] = sum[int32_t{1}];
        }
        // TmemSoftmax.h:1010
        {
          // Common.h:395
          cutlass::Array<float, 2> oldMax{float{oldMaxPtr12[int32_t{2}]},
                                          float{oldMaxPtr12[int32_t{3}]}};
          // Common.h:395
          cutlass::Array<float, 2> newMax{float{newMaxPtr12[int32_t{2}]},
                                          float{newMaxPtr12[int32_t{3}]}};
          // Common.h:353
          cutlass::Array<float, 2> scale2{float{1}, float{1}};
          // Common.h:133
          newMax[int32_t{0}] = -(newMax[int32_t{0}]);
          // Common.h:134
          newMax[int32_t{1}] = -(newMax[int32_t{1}]);
          // Common.h:137
          cutlass::Array<float, 2> maxDiff2{trtllm::dev::fadd2(oldMax, newMax)};
          // Common.h:155
          if (((maxDiff2[int32_t{0}]) != (float{0})) || ((maxDiff2[int32_t{1}]) != (float{0}))) {
            // Common.h:353
            cutlass::Array<float, 2> scale2_{scaleSoftmaxLog212, scaleSoftmaxLog212};
            // Common.h:161
            scale2 = trtllm::dev::fmul2(scale2_, maxDiff2);
            // Common.h:168
            scale2[int32_t{0}] = exp2f(scale2[int32_t{0}]);
            // Common.h:169
            scale2[int32_t{1}] = exp2f(scale2[int32_t{1}]);
          }
          // TmemSoftmax.h:1029
          cutlass::Array<float, 2> p0{pPtr12[int32_t{4}], pPtr12[int32_t{5}]};
          // Common.h:395
          cutlass::Array<float, 2> sum{float{sumPtr12[int32_t{2}]}, float{sumPtr12[int32_t{3}]}};
          // TmemSoftmax.h:1048
          sum = trtllm::dev::ffma2(scale2, sum, p0);
          // TmemSoftmax.h:1060
          cutlass::Array<float, 2> p1{pPtr12[int32_t{6}], pPtr12[int32_t{7}]};
          // TmemSoftmax.h:1076
          sum = trtllm::dev::fadd2(sum, p1);
          // TmemSoftmax.h:1083
          sumPtr12[int32_t{2}] = sum[int32_t{0}];
          // TmemSoftmax.h:1084
          sumPtr12[int32_t{3}] = sum[int32_t{1}];
          // TmemSoftmax.h:1060
          cutlass::Array<float, 2> p2{pPtr12[int32_t{12}], pPtr12[int32_t{13}]};
          // TmemSoftmax.h:1076
          sum = trtllm::dev::fadd2(sum, p2);
          // TmemSoftmax.h:1083
          sumPtr12[int32_t{2}] = sum[int32_t{0}];
          // TmemSoftmax.h:1084
          sumPtr12[int32_t{3}] = sum[int32_t{1}];
          // TmemSoftmax.h:1060
          cutlass::Array<float, 2> p3{pPtr12[int32_t{14}], pPtr12[int32_t{15}]};
          // TmemSoftmax.h:1076
          sum = trtllm::dev::fadd2(sum, p3);
          // TmemSoftmax.h:1083
          sumPtr12[int32_t{2}] = sum[int32_t{0}];
          // TmemSoftmax.h:1084
          sumPtr12[int32_t{3}] = sum[int32_t{1}];
        }
      }
      //
      // tmemSoftmaxLocal0 [ProdWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{10}, Flags{0}].
      //
      // Task.cpp:1477
      oldMaxPtr11 = oldMaxPtr10;
      // Task.cpp:1477
      sumPtr11 = sumPtr10;
      // Task.cpp:1477
      newMaxPtr11 = newMaxPtr010;
      // Task.cpp:1573
      if (isLastLoopIter) {
        // Task.cpp:5829
        int32_t index{tmemSoftmaxLocal0ProdState.index()};
        // TmemTile.cpp:373
        cutlass::Array<float, 8> stats;
        // TmemSoftmax.h:365
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset1644 = int32_t{0}; loopOffset1644 < int32_t{4}; ++loopOffset1644) {
          // TmemSoftmax.h:382
          stats[loopOffset1644] = sumPtr11[loopOffset1644];
          // TmemSoftmax.h:384
          stats[(loopOffset1644) + (int32_t{4})] = newMaxPtr11[loopOffset1644];
        }
        // TmemTile.cpp:824
        {
          // TmemTile.cpp:826
          uint32_t tmemBasePtr{mTmemBaseOffset};
          // TmemTile.cpp:859
          uint32_t const(&srcSlice0)[8]{reinterpret_cast<uint32_t const(&)[8]>(stats[int32_t{0}])};
          // CudaPtx.h:48
          cuda_ptx::tcgen05_st_32x32b(
            (tmemBasePtr) +
              (static_cast<uint32_t>((index) * (int32_t{32}) +
                                     (int32_t((tmemSoftmaxLocal0DstStack.mInstId) == (int32_t{0}))
                                        ? (int32_t{32})
                                        : (int32_t{64})))),
            srcSlice0);
        }
        // TmemSoftmax.h:407
        cutlass::arch::fence_view_async_tmem_store();
      }
      //
      // tmemSoftmaxLocal0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{10}, Flags{0}].
      //
      // Task.cpp:1573
      if (isLastLoopIter) {
        // Task.cpp:4427
        {
          // Task.cpp:4443
          tmemSoftmaxLocal0DstStack.mPipeline.producer_commit(tmemSoftmaxLocal0ProdState);
        }
        // Task.cpp:43
        ++tmemSoftmaxLocal0ProdState;
      }
      //
      // tmemSoftmaxLocal0 [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{2621952}].
      //
      // Task.cpp:1573
      if (isLastLoopIter) {
        // Task.cpp:4962
        if ((loopOffset1265) >= (int32_t{0})) {
          // Task.cpp:4984
          tmemSoftmaxLocal0ProdToken =
            tmemSoftmaxLocal0DstStack.mPipeline.producer_try_acquire(tmemSoftmaxLocal0ProdState);
        }
      }
      // Task.cpp:1573
      if (isLastLoopIter) {
        // Task.cpp:4210
        tmemSoftmaxLocal0DstStack.mPipeline.producer_acquire(tmemSoftmaxLocal0ProdState,
                                                             tmemSoftmaxLocal0ProdToken);
      }
      //
      // tmemSoftmaxLocal0 [ProdWork (call 2), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{2621952}].
      //
      //
      // tmemSoftmaxLocal0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{2621952}].
      //
      // Task.cpp:1573
      if (isLastLoopIter) {
        // Task.cpp:4427
        {
          //
          // Skipped by flag SkipsProdCommit.
          //
        }
      }
    }
  //
  // Tail work.
  //
  // Task.cpp:3511
  ExitTileWithSignalingLabel:
  // Task.cpp:3518
  ExitTileWithoutSignalingLabel:
    // Task.cpp:3528
    {}
  }
};
// Task.cpp:544
// Fmha.h:2358
struct CorrTask {
  // Task.cpp:283
  int32_t mCtaIdxX;
  // Task.cpp:287
  int32_t mCtaIdxY;
  // Task.cpp:291
  int32_t mCtaIdxZ;
  // FmhaTask.h:220
  int32_t mHeadIdx;
  // FmhaTask.h:218
  int32_t mBatchIdx;
  // FmhaTask.h:208
  int32_t mSeqOffsetQ;
  // FmhaTask.h:210
  int32_t mSeqLenQ;
  // FmhaTask.h:224
  int32_t mCtaIdxQ;
  // FmhaTask.h:226
  int32_t mCtaIdxKv;
  // FmhaTask.h:214
  int32_t mSeqLenKv;
  // FmhaTask.h:216
  int32_t mNumCtasKv;
  // Task.cpp:691
  uint32_t const mTmemBaseOffset;
  // Task.cpp:371
  int32_t const mWarpGrpThreadIdx;
  // TmemTile.cpp:422
  int32_t const mLdtm16dp256bitTmemColIdx;
  // TmemTile.cpp:445
  int32_t const mLdtm16dp256bitTmemRowIdx;
  // Task.cpp:266
  int32_t const mGridDimY;
  // Task.cpp:551
  inline __device__ CorrTask(fmha::KernelParams const& params,
                             KernelState const& state,
                             int32_t warpGrpStart)
    : // Kernel.cpp:194
    mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , // Kernel.cpp:195
    mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , // Kernel.cpp:196
    mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)}
    , // Task.cpp:287
    mHeadIdx{mCtaIdxY}
    , // Task.cpp:291
    mBatchIdx{mCtaIdxZ}
    , // FmhaTask.h:394
    mSeqOffsetQ{int32_t(bool{params.ptrCumSeqLensQ == nullptr})
                  ? ((mBatchIdx) * (params.mMaxSeqLenQ))
                  : (int32_t{params.ptrCumSeqLensQ[mBatchIdx]})}
    , // FmhaTask.h:410
    mSeqLenQ{int32_t(bool{params.ptrCumSeqLensQ == nullptr})
               ? (params.mMaxSeqLenQ)
               : ((int32_t{params.ptrCumSeqLensQ[(mBatchIdx) + (int32_t{1})]}) - (mSeqOffsetQ))}
    , // FmhaTask.h:516
    mCtaIdxQ{(mCtaIdxX) / (params.mMaxNumCtasKv)}
    , // FmhaTask.h:517
    mCtaIdxKv{(mCtaIdxX) % (params.mMaxNumCtasKv)}
    , // FmhaTask.h:543
    mSeqLenKv{(int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                             ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                             : (mBatchIdx)]}) -
              (((mSeqLenQ) - (int32_t{1})) - (mCtaIdxQ))}
    , // FmhaTask.h:565
    mNumCtasKv{
      int32_t{min(int32_t{((mSeqLenKv) + (int32_t{127})) / (int32_t{128})}, params.mMaxNumCtasKv)}}
    , // Kernel.cpp:2420
    mTmemBaseOffset{uint32_t{
      __shfl_sync(uint32_t{0xffffffff}, (*state.mTmemSwStatePtr), int32_t{0}, int32_t{32})}}
    , // Task.cpp:379
    mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))}
    , // TmemTile.cpp:432
    mLdtm16dp256bitTmemColIdx{
      trtllm::dev::ldst16dp256bitTmemColIdx((mWarpGrpThreadIdx) % (int32_t{128}))}
    , // TmemTile.cpp:453
    mLdtm16dp256bitTmemRowIdx{
      trtllm::dev::ldst16dp256bitTmemRowIdx<int32_t{16}>((mWarpGrpThreadIdx) % (int32_t{128}))}
    , // Kernel.cpp:192
    mGridDimY{reinterpret_cast<int32_t const&>(gridDim.y)} {}
  // Task.cpp:507
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:532
    return ((state.mWarpIdx) >= (int32_t{4})) && ((state.mWarpIdx) < (int32_t{8}));
  }
  // Task.cpp:454
  inline __device__ void execute(fmha::KernelParams const& params,
                                 KernelState const& state,
                                 TmemCorr0Stack& tmemCorr0DstStack,
                                 TmemSoftmaxLocal0Stack& tmemSoftmaxLocal0SrcStack,
                                 TmemOStack& tmemOSrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<88>{});
    // Task.cpp:2079
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState tmemSoftmaxLocal0ConsState{};
    // Task.cpp:2086
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState tmemSoftmaxLocal0ConsReleaseState{};
    // Task.cpp:2100
    int32_t tmemSoftmaxLocal0ConsToken{int32_t{0}};
    // Task.cpp:2079
    trtllm::dev::CutlassUmmaAsyncPipeline<
      1,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState tmemOConsState{};
    // Task.cpp:2086
    trtllm::dev::CutlassUmmaAsyncPipeline<
      1,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState tmemOConsReleaseState{};
    // Task.cpp:2100
    int32_t tmemOConsToken{int32_t{0}};
    // FmhaTask.h:582
    int32_t numLoopSteps;
    // FmhaTask.h:630
    if (((mCtaIdxQ) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
      // FmhaTask.h:668
      int32_t numSteps{((mSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
                       ((mNumCtasKv) * (int32_t{128}))};
      // FmhaTask.h:682
      numLoopSteps = numSteps;
    } else {
      // FmhaTask.h:648
      numLoopSteps = int32_t{0};
    }
    // Task.cpp:3168
    bool const hasOneLoopIter{(int32_t{0}) < (numLoopSteps)};
    // Task.cpp:3179
    int32_t lastLoopOffset{int32_t{0}};
    // TmemTile.cpp:373
    cutlass::Array<float, 8> frgStats11;
    // TmemCorr.h:1126
    cudaGridDependencySynchronize();
    // TmemCorr.h:1149
    float scaleSoftmaxLog216;
    // TmemCorr.h:1154
    scaleSoftmaxLog216 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
                           ? (params.mScaleSoftmaxLog2)
                           : (float{params.ptrScaleSoftmaxLog2[int32_t{0}]});
    //
    // Hoist the first iter.
    //
    //
    // tmemSoftmaxLocal0 [ConsWait, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:2745
        tmemSoftmaxLocal0ConsToken =
          tmemSoftmaxLocal0SrcStack.mPipeline.consumer_try_wait(tmemSoftmaxLocal0ConsState);
      }
      // Task.cpp:2813
      tmemSoftmaxLocal0SrcStack.mPipeline.consumer_wait(tmemSoftmaxLocal0ConsState,
                                                        tmemSoftmaxLocal0ConsToken);
    }
    //
    // tmemSoftmaxLocal0 [ConsWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{tmemSoftmaxLocal0ConsState.index()};
      // Task.cpp:43
      ++tmemSoftmaxLocal0ConsState;
    }
    //
    // tmemSoftmaxLocal0 [ConsRelease, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:2561
      {
        // Task.cpp:2585
        tmemSoftmaxLocal0SrcStack.mPipeline.consumer_release(tmemSoftmaxLocal0ConsReleaseState);
      }
      // Task.cpp:43
      ++tmemSoftmaxLocal0ConsReleaseState;
    }
    //
    // tmemCorr0 [ProdWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{512}].
    //
    //
    // Loop body.
    //
    // Task.cpp:3350
    CUTLASS_PRAGMA_NO_UNROLL
    for (int32_t loopOffset1746 = int32_t{0}; loopOffset1746 < (numLoopSteps) - (int32_t{1});
         ++loopOffset1746) {
      //
      // tmemSoftmaxLocal0 [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
      //
      // Task.cpp:1573
      // Task.cpp:2781
      {
        // Task.cpp:1573
        // Task.cpp:2722
        {
          // Task.cpp:2745
          tmemSoftmaxLocal0ConsToken =
            tmemSoftmaxLocal0SrcStack.mPipeline.consumer_try_wait(tmemSoftmaxLocal0ConsState);
        }
        // Task.cpp:2813
        tmemSoftmaxLocal0SrcStack.mPipeline.consumer_wait(tmemSoftmaxLocal0ConsState,
                                                          tmemSoftmaxLocal0ConsToken);
      }
      //
      // tmemSoftmaxLocal0 [ConsWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
      //
      // TmemSoftmax.h:231
      float* statsPtr111;
      // Task.cpp:1573
      // Task.cpp:2893
      {
        // Task.cpp:5829
        int32_t index{tmemSoftmaxLocal0ConsState.index()};
        // TmemTile.cpp:527
        {
          // TmemTile.cpp:529
          uint32_t tmemBasePtr{mTmemBaseOffset};
          // TmemTile.cpp:545
          uint32_t(&dstSlice0)[8]{reinterpret_cast<uint32_t(&)[8]>(frgStats11[int32_t{0}])};
          // CudaPtx.h:48
          cuda_ptx::tcgen05_ld_32x32b(
            dstSlice0,
            (tmemBasePtr) +
              (static_cast<uint32_t>((index) * (int32_t{32}) +
                                     (int32_t((tmemSoftmaxLocal0SrcStack.mInstId) == (int32_t{0}))
                                        ? (int32_t{32})
                                        : (int32_t{64})))));
        }
        // TmemSoftmax.h:327
        statsPtr111 = &frgStats11[int32_t{0}];
        // TmemSoftmax.h:330
        cutlass::arch::fence_view_async_tmem_load();
        // Task.cpp:43
        ++tmemSoftmaxLocal0ConsState;
      }
      //
      // tmemSoftmaxLocal0 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
      //
      // Task.cpp:2533
      if ((loopOffset1746) >= (int32_t{0})) {
        // Task.cpp:2561
        {
          // Task.cpp:2585
          tmemSoftmaxLocal0SrcStack.mPipeline.consumer_release(tmemSoftmaxLocal0ConsReleaseState);
        }
        // Task.cpp:43
        ++tmemSoftmaxLocal0ConsReleaseState;
      }
      //
      // tmemO [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:2781
      {
        // Task.cpp:1573
        // Task.cpp:2722
        {
          // Task.cpp:2745
          tmemOConsToken = tmemOSrcStack.mPipeline.consumer_try_wait(tmemOConsState);
        }
        // Task.cpp:2813
        tmemOSrcStack.mPipeline.consumer_wait(tmemOConsState, tmemOConsToken);
      }
      //
      // tmemO [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:2893
      {
        // Task.cpp:5829
        int32_t index{tmemOConsState.index()};
        // Task.cpp:43
        ++tmemOConsState;
      }
      //
      // tmemCorr0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
      //
      // TmemCorr.h:1184
      float* prodStatsPtr016;
      // Task.cpp:1477
      prodStatsPtr016 = statsPtr111;
      // Task.cpp:1573
      // Task.cpp:5038
      {
        // TmemCorr.h:416
        cutlass::Array<float, 4> scales16;
        // TmemCorr.h:428
        {
          // Common.h:353
          cutlass::Array<float, 2> oldMax{float{prodStatsPtr016[int32_t{0}]},
                                          float{prodStatsPtr016[int32_t{1}]}};
          // Common.h:353
          cutlass::Array<float, 2> newMax{float{prodStatsPtr016[int32_t{4}]},
                                          float{prodStatsPtr016[int32_t{5}]}};
          // Common.h:353
          cutlass::Array<float, 2> scale2{float{1}, float{1}};
          // Common.h:133
          newMax[int32_t{0}] = -(newMax[int32_t{0}]);
          // Common.h:134
          newMax[int32_t{1}] = -(newMax[int32_t{1}]);
          // Common.h:137
          cutlass::Array<float, 2> maxDiff2{trtllm::dev::fadd2(oldMax, newMax)};
          // Common.h:155
          if (((maxDiff2[int32_t{0}]) != (float{0})) || ((maxDiff2[int32_t{1}]) != (float{0}))) {
            // Common.h:353
            cutlass::Array<float, 2> scale2_{scaleSoftmaxLog216, scaleSoftmaxLog216};
            // Common.h:161
            scale2 = trtllm::dev::fmul2(scale2_, maxDiff2);
            // Common.h:168
            scale2[int32_t{0}] = exp2f(scale2[int32_t{0}]);
            // Common.h:169
            scale2[int32_t{1}] = exp2f(scale2[int32_t{1}]);
          }
          // TmemCorr.h:464
          scales16[int32_t{0}] = scale2[int32_t{0}];
          // TmemCorr.h:465
          scales16[int32_t{1}] = scale2[int32_t{1}];
        }
        // TmemCorr.h:428
        {
          // Common.h:353
          cutlass::Array<float, 2> oldMax{float{prodStatsPtr016[int32_t{2}]},
                                          float{prodStatsPtr016[int32_t{3}]}};
          // Common.h:353
          cutlass::Array<float, 2> newMax{float{prodStatsPtr016[int32_t{6}]},
                                          float{prodStatsPtr016[int32_t{7}]}};
          // Common.h:353
          cutlass::Array<float, 2> scale2{float{1}, float{1}};
          // Common.h:133
          newMax[int32_t{0}] = -(newMax[int32_t{0}]);
          // Common.h:134
          newMax[int32_t{1}] = -(newMax[int32_t{1}]);
          // Common.h:137
          cutlass::Array<float, 2> maxDiff2{trtllm::dev::fadd2(oldMax, newMax)};
          // Common.h:155
          if (((maxDiff2[int32_t{0}]) != (float{0})) || ((maxDiff2[int32_t{1}]) != (float{0}))) {
            // Common.h:353
            cutlass::Array<float, 2> scale2_{scaleSoftmaxLog216, scaleSoftmaxLog216};
            // Common.h:161
            scale2 = trtllm::dev::fmul2(scale2_, maxDiff2);
            // Common.h:168
            scale2[int32_t{0}] = exp2f(scale2[int32_t{0}]);
            // Common.h:169
            scale2[int32_t{1}] = exp2f(scale2[int32_t{1}]);
          }
          // TmemCorr.h:464
          scales16[int32_t{2}] = scale2[int32_t{0}];
          // TmemCorr.h:465
          scales16[int32_t{3}] = scale2[int32_t{1}];
        }
        // TmemCorr.h:1231
        bool skipsCorr{true};
        // TmemCorr.h:1249
        skipsCorr = (skipsCorr) && ((scales16[int32_t{0}]) == (float{1}));
        // TmemCorr.h:1249
        skipsCorr = (skipsCorr) && ((scales16[int32_t{1}]) == (float{1}));
        // TmemCorr.h:1249
        skipsCorr = (skipsCorr) && ((scales16[int32_t{2}]) == (float{1}));
        // TmemCorr.h:1249
        skipsCorr = (skipsCorr) && ((scales16[int32_t{3}]) == (float{1}));
        // TmemCorr.h:1257
        skipsCorr = __all_sync(uint32_t{-1}, skipsCorr);
        // TmemCorr.h:1259
        if (!(skipsCorr)) {
          //
          // The headDimStageIdx: 0.
          //
          // TmemCorr.h:1472
          for (int32_t loopOffset1829 = int32_t{0}; loopOffset1829 < int32_t{16};
               loopOffset1829 += int32_t{16}) {
            // TmemTile.cpp:373
            cutlass::Array<float, 16> tmemRegs016;
            // TmemTile.cpp:527
            {
              // TmemTile.cpp:529
              uint32_t tmemBasePtr{mTmemBaseOffset};
              // TmemTile.cpp:545
              uint32_t(&dstSlice0)[8]{reinterpret_cast<uint32_t(&)[8]>(tmemRegs016[int32_t{0}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_16x256b(
                dstSlice0,
                (tmemBasePtr) +
                  (static_cast<uint32_t>((int32_t{0x60 /*hi=0, lo=96*/}) + (loopOffset1829))));
              // TmemTile.cpp:545
              uint32_t(&dstSlice1)[8]{reinterpret_cast<uint32_t(&)[8]>(tmemRegs016[int32_t{8}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_16x256b(
                dstSlice1,
                (tmemBasePtr) +
                  (static_cast<uint32_t>(((int32_t{0x60 /*hi=0, lo=96*/}) + (loopOffset1829)) +
                                         (int32_t{0x100000 /*hi=16, lo=0*/}))));
            }
            // TmemCorr.h:1520
            {
              // TmemCorr.h:1540
              cutlass::Array<float, 2> localScales0{scales16[int32_t{0}], scales16[int32_t{1}]};
              // TmemCorr.h:1551
              cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{0}], tmemRegs016[int32_t{1}]};
              // TmemCorr.h:1563
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1566
              tmemRegs016[int32_t{0}] = vals0[int32_t{0}];
              // TmemCorr.h:1567
              tmemRegs016[int32_t{1}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1520
            {
              // TmemCorr.h:1540
              cutlass::Array<float, 2> localScales0{scales16[int32_t{0}], scales16[int32_t{1}]};
              // TmemCorr.h:1551
              cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{2}], tmemRegs016[int32_t{3}]};
              // TmemCorr.h:1563
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1566
              tmemRegs016[int32_t{2}] = vals0[int32_t{0}];
              // TmemCorr.h:1567
              tmemRegs016[int32_t{3}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1520
            {
              // TmemCorr.h:1540
              cutlass::Array<float, 2> localScales0{scales16[int32_t{2}], scales16[int32_t{3}]};
              // TmemCorr.h:1551
              cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{4}], tmemRegs016[int32_t{5}]};
              // TmemCorr.h:1563
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1566
              tmemRegs016[int32_t{4}] = vals0[int32_t{0}];
              // TmemCorr.h:1567
              tmemRegs016[int32_t{5}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1520
            {
              // TmemCorr.h:1540
              cutlass::Array<float, 2> localScales0{scales16[int32_t{2}], scales16[int32_t{3}]};
              // TmemCorr.h:1551
              cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{6}], tmemRegs016[int32_t{7}]};
              // TmemCorr.h:1563
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1566
              tmemRegs016[int32_t{6}] = vals0[int32_t{0}];
              // TmemCorr.h:1567
              tmemRegs016[int32_t{7}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1520
            {
              // TmemCorr.h:1540
              cutlass::Array<float, 2> localScales0{scales16[int32_t{0}], scales16[int32_t{1}]};
              // TmemCorr.h:1551
              cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{8}], tmemRegs016[int32_t{9}]};
              // TmemCorr.h:1563
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1566
              tmemRegs016[int32_t{8}] = vals0[int32_t{0}];
              // TmemCorr.h:1567
              tmemRegs016[int32_t{9}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1520
            {
              // TmemCorr.h:1540
              cutlass::Array<float, 2> localScales0{scales16[int32_t{0}], scales16[int32_t{1}]};
              // TmemCorr.h:1551
              cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{10}], tmemRegs016[int32_t{11}]};
              // TmemCorr.h:1563
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1566
              tmemRegs016[int32_t{10}] = vals0[int32_t{0}];
              // TmemCorr.h:1567
              tmemRegs016[int32_t{11}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1520
            {
              // TmemCorr.h:1540
              cutlass::Array<float, 2> localScales0{scales16[int32_t{2}], scales16[int32_t{3}]};
              // TmemCorr.h:1551
              cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{12}], tmemRegs016[int32_t{13}]};
              // TmemCorr.h:1563
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1566
              tmemRegs016[int32_t{12}] = vals0[int32_t{0}];
              // TmemCorr.h:1567
              tmemRegs016[int32_t{13}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1520
            {
              // TmemCorr.h:1540
              cutlass::Array<float, 2> localScales0{scales16[int32_t{2}], scales16[int32_t{3}]};
              // TmemCorr.h:1551
              cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{14}], tmemRegs016[int32_t{15}]};
              // TmemCorr.h:1563
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1566
              tmemRegs016[int32_t{14}] = vals0[int32_t{0}];
              // TmemCorr.h:1567
              tmemRegs016[int32_t{15}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1976
            {
              // TmemTile.cpp:824
              {
                // TmemTile.cpp:826
                uint32_t tmemBasePtr{mTmemBaseOffset};
                // TmemTile.cpp:859
                uint32_t const(&srcSlice0)[8]{
                  reinterpret_cast<uint32_t const(&)[8]>(tmemRegs016[int32_t{0}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_st_16x256b(
                  (tmemBasePtr) +
                    (static_cast<uint32_t>((int32_t{0x60 /*hi=0, lo=96*/}) + (loopOffset1829))),
                  srcSlice0);
                // TmemTile.cpp:859
                uint32_t const(&srcSlice1)[8]{
                  reinterpret_cast<uint32_t const(&)[8]>(tmemRegs016[int32_t{8}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_st_16x256b(
                  (tmemBasePtr) +
                    (static_cast<uint32_t>(((int32_t{0x60 /*hi=0, lo=96*/}) + (loopOffset1829)) +
                                           (int32_t{0x100000 /*hi=16, lo=0*/}))),
                  srcSlice1);
              }
            }
          }
          //
          // The headDimStageIdx: 1.
          //
          // TmemCorr.h:1472
          for (int32_t loopOffset1894 = int32_t{0}; loopOffset1894 < int32_t{16};
               loopOffset1894 += int32_t{16}) {
            // TmemTile.cpp:373
            cutlass::Array<float, 16> tmemRegs016;
            // TmemTile.cpp:527
            {
              // TmemTile.cpp:529
              uint32_t tmemBasePtr{mTmemBaseOffset};
              // TmemTile.cpp:545
              uint32_t(&dstSlice0)[8]{reinterpret_cast<uint32_t(&)[8]>(tmemRegs016[int32_t{0}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_16x256b(
                dstSlice0,
                (tmemBasePtr) +
                  (static_cast<uint32_t>((int32_t{0x70 /*hi=0, lo=112*/}) + (loopOffset1894))));
              // TmemTile.cpp:545
              uint32_t(&dstSlice1)[8]{reinterpret_cast<uint32_t(&)[8]>(tmemRegs016[int32_t{8}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_16x256b(
                dstSlice1,
                (tmemBasePtr) +
                  (static_cast<uint32_t>(((int32_t{0x70 /*hi=0, lo=112*/}) + (loopOffset1894)) +
                                         (int32_t{0x100000 /*hi=16, lo=0*/}))));
            }
            // TmemCorr.h:1520
            {
              // TmemCorr.h:1540
              cutlass::Array<float, 2> localScales0{scales16[int32_t{0}], scales16[int32_t{1}]};
              // TmemCorr.h:1551
              cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{0}], tmemRegs016[int32_t{1}]};
              // TmemCorr.h:1563
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1566
              tmemRegs016[int32_t{0}] = vals0[int32_t{0}];
              // TmemCorr.h:1567
              tmemRegs016[int32_t{1}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1520
            {
              // TmemCorr.h:1540
              cutlass::Array<float, 2> localScales0{scales16[int32_t{0}], scales16[int32_t{1}]};
              // TmemCorr.h:1551
              cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{2}], tmemRegs016[int32_t{3}]};
              // TmemCorr.h:1563
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1566
              tmemRegs016[int32_t{2}] = vals0[int32_t{0}];
              // TmemCorr.h:1567
              tmemRegs016[int32_t{3}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1520
            {
              // TmemCorr.h:1540
              cutlass::Array<float, 2> localScales0{scales16[int32_t{2}], scales16[int32_t{3}]};
              // TmemCorr.h:1551
              cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{4}], tmemRegs016[int32_t{5}]};
              // TmemCorr.h:1563
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1566
              tmemRegs016[int32_t{4}] = vals0[int32_t{0}];
              // TmemCorr.h:1567
              tmemRegs016[int32_t{5}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1520
            {
              // TmemCorr.h:1540
              cutlass::Array<float, 2> localScales0{scales16[int32_t{2}], scales16[int32_t{3}]};
              // TmemCorr.h:1551
              cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{6}], tmemRegs016[int32_t{7}]};
              // TmemCorr.h:1563
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1566
              tmemRegs016[int32_t{6}] = vals0[int32_t{0}];
              // TmemCorr.h:1567
              tmemRegs016[int32_t{7}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1520
            {
              // TmemCorr.h:1540
              cutlass::Array<float, 2> localScales0{scales16[int32_t{0}], scales16[int32_t{1}]};
              // TmemCorr.h:1551
              cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{8}], tmemRegs016[int32_t{9}]};
              // TmemCorr.h:1563
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1566
              tmemRegs016[int32_t{8}] = vals0[int32_t{0}];
              // TmemCorr.h:1567
              tmemRegs016[int32_t{9}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1520
            {
              // TmemCorr.h:1540
              cutlass::Array<float, 2> localScales0{scales16[int32_t{0}], scales16[int32_t{1}]};
              // TmemCorr.h:1551
              cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{10}], tmemRegs016[int32_t{11}]};
              // TmemCorr.h:1563
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1566
              tmemRegs016[int32_t{10}] = vals0[int32_t{0}];
              // TmemCorr.h:1567
              tmemRegs016[int32_t{11}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1520
            {
              // TmemCorr.h:1540
              cutlass::Array<float, 2> localScales0{scales16[int32_t{2}], scales16[int32_t{3}]};
              // TmemCorr.h:1551
              cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{12}], tmemRegs016[int32_t{13}]};
              // TmemCorr.h:1563
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1566
              tmemRegs016[int32_t{12}] = vals0[int32_t{0}];
              // TmemCorr.h:1567
              tmemRegs016[int32_t{13}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1520
            {
              // TmemCorr.h:1540
              cutlass::Array<float, 2> localScales0{scales16[int32_t{2}], scales16[int32_t{3}]};
              // TmemCorr.h:1551
              cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{14}], tmemRegs016[int32_t{15}]};
              // TmemCorr.h:1563
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1566
              tmemRegs016[int32_t{14}] = vals0[int32_t{0}];
              // TmemCorr.h:1567
              tmemRegs016[int32_t{15}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1976
            {
              // TmemTile.cpp:824
              {
                // TmemTile.cpp:826
                uint32_t tmemBasePtr{mTmemBaseOffset};
                // TmemTile.cpp:859
                uint32_t const(&srcSlice0)[8]{
                  reinterpret_cast<uint32_t const(&)[8]>(tmemRegs016[int32_t{0}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_st_16x256b(
                  (tmemBasePtr) +
                    (static_cast<uint32_t>((int32_t{0x70 /*hi=0, lo=112*/}) + (loopOffset1894))),
                  srcSlice0);
                // TmemTile.cpp:859
                uint32_t const(&srcSlice1)[8]{
                  reinterpret_cast<uint32_t const(&)[8]>(tmemRegs016[int32_t{8}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_st_16x256b(
                  (tmemBasePtr) +
                    (static_cast<uint32_t>(((int32_t{0x70 /*hi=0, lo=112*/}) + (loopOffset1894)) +
                                           (int32_t{0x100000 /*hi=16, lo=0*/}))),
                  srcSlice1);
              }
            }
          }
          // TmemCorr.h:1588
          cutlass::arch::fence_view_async_tmem_store();
        }
      }
      //
      // tmemO [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{1024}].
      //
      // Task.cpp:2533
      if ((loopOffset1746) >= (int32_t{0})) {
        // Task.cpp:2561
        {
          // Task.cpp:2585
          tmemOSrcStack.mPipeline.consumer_release(tmemOConsReleaseState);
        }
        // Task.cpp:43
        ++tmemOConsReleaseState;
      }
      //
      // tmemSoftmaxLocal0 [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // Task.cpp:3457
      lastLoopOffset = loopOffset1746;
    }
    //
    // Pull the last iter down.
    //
    // Task.cpp:3492
    if (((numLoopSteps) - (int32_t{1})) > (int32_t{0})) {
      // Task.cpp:3493
      ++lastLoopOffset;
    }
    //
    // tmemSoftmaxLocal0 [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:2745
        tmemSoftmaxLocal0ConsToken =
          tmemSoftmaxLocal0SrcStack.mPipeline.consumer_try_wait(tmemSoftmaxLocal0ConsState);
      }
      // Task.cpp:2813
      tmemSoftmaxLocal0SrcStack.mPipeline.consumer_wait(tmemSoftmaxLocal0ConsState,
                                                        tmemSoftmaxLocal0ConsToken);
    }
    //
    // tmemSoftmaxLocal0 [ConsWork (call 2), LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
    //
    // TmemSoftmax.h:231
    float* statsPtr211;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{tmemSoftmaxLocal0ConsState.index()};
      // TmemTile.cpp:527
      {
        // TmemTile.cpp:529
        uint32_t tmemBasePtr{mTmemBaseOffset};
        // TmemTile.cpp:545
        uint32_t(&dstSlice0)[8]{reinterpret_cast<uint32_t(&)[8]>(frgStats11[int32_t{0}])};
        // CudaPtx.h:48
        cuda_ptx::tcgen05_ld_32x32b(
          dstSlice0,
          (tmemBasePtr) +
            (static_cast<uint32_t>((index) * (int32_t{32}) +
                                   (int32_t((tmemSoftmaxLocal0SrcStack.mInstId) == (int32_t{0}))
                                      ? (int32_t{32})
                                      : (int32_t{64})))));
      }
      // TmemSoftmax.h:327
      statsPtr211 = &frgStats11[int32_t{0}];
      // TmemSoftmax.h:330
      cutlass::arch::fence_view_async_tmem_load();
      // Task.cpp:43
      ++tmemSoftmaxLocal0ConsState;
    }
    //
    // tmemSoftmaxLocal0 [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:2561
      {
        // Task.cpp:2585
        tmemSoftmaxLocal0SrcStack.mPipeline.consumer_release(tmemSoftmaxLocal0ConsReleaseState);
      }
      // Task.cpp:43
      ++tmemSoftmaxLocal0ConsReleaseState;
    }
    //
    // tmemO [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{34}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:2745
        tmemOConsToken = tmemOSrcStack.mPipeline.consumer_try_wait(tmemOConsState);
      }
      // Task.cpp:2813
      tmemOSrcStack.mPipeline.consumer_wait(tmemOConsState, tmemOConsToken);
    }
    //
    // tmemO [ConsWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{34}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{tmemOConsState.index()};
      // Task.cpp:43
      ++tmemOConsState;
    }
    //
    // tmemCorr0 [ProdWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // TmemCorr.h:1184
    float* prodStatsPtr116;
    // Task.cpp:1477
    prodStatsPtr116 = statsPtr211;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // TmemCorr.h:2829
      int32_t instIdxO{(mCtaIdxQ) * (int32_t{1})};
      // TmemCorr.h:2953
      int32_t seqOffsetO{(mSeqOffsetQ) + (instIdxO)};
      // TmemCorr.h:2958
      int32_t headIdxO;
      // TmemCorr.h:2962
      headIdxO = (mHeadIdx) * (int32_t{min(params.mNumHeadsQPerKv, int32_t{16})});
      // TmemCorr.h:2965
      int32_t headOffsetO{((mHeadIdx) * (int32_t{min(params.mNumHeadsQPerKv, int32_t{16})})) *
                          (int32_t{256})};
      // TmemCorr.h:2996
      int64_t ctaOffsetO{(static_cast<int64_t>(seqOffsetO)) *
                           (static_cast<int64_t>((params.mNumHeadsQ) * (int32_t{256}))) +
                         (static_cast<int64_t>(headOffsetO))};
      // TmemCorr.h:3010
      cutlass::half_t* ptrO{reinterpret_cast<cutlass::half_t*>(params.ptrO)};
      // TmemCorr.h:3015
      ptrO = ptrO + (ctaOffsetO);
      // TmemCorr.h:3045
      bool storesSoftmaxStats{reinterpret_cast<float*>(params.ptrSoftmaxStats) != nullptr};
      // TmemCorr.h:3051
      float* ptrSoftmaxStats;
      // TmemCorr.h:3053
      if (storesSoftmaxStats) {
        // TmemCorr.h:3057
        ptrSoftmaxStats = reinterpret_cast<float*>(params.ptrSoftmaxStats) +
                          (((seqOffsetO) * (params.mNumHeadsQ) +
                            ((mHeadIdx) * (int32_t{min(params.mNumHeadsQPerKv, int32_t{16})}))) *
                           (int32_t{2}));
      }
      // TmemCorr.h:2444
      int32_t numValidSlices{((int32_t{min(params.mNumHeadsQPerKv, int32_t{16})}) + (int32_t{3})) /
                             (int32_t{4})};
      // TmemCorr.h:2451
      int32_t numSlicesPerCta{((numValidSlices) + ((mNumCtasKv) - (int32_t{1}))) / (mNumCtasKv)};
      // TmemCorr.h:2459
      int32_t numCtasKvForReduction{((numValidSlices) + ((numSlicesPerCta) - (int32_t{1}))) /
                                    (numSlicesPerCta)};
      // TmemCorr.h:2467
      int32_t numReductionRowsPerCta{(numSlicesPerCta) * (int32_t{4})};
      // TmemCorr.h:2540
      float* ptrPartialStats;
      // TmemCorr.h:2543
      ptrPartialStats =
        reinterpret_cast<float*>(params.ptrPartialStats) +
        (((((mBatchIdx) * (mGridDimY) + (mHeadIdx)) * (params.mMaxNumCtasQ)) + (mCtaIdxQ)) *
         (((params.mMaxNumCtasKv) * (int32_t{16})) * (int32_t{2})));
      // TmemCorr.h:2576
      cutlass::half_t* ptrPartialO;
      // TmemCorr.h:2578
      ptrPartialO =
        reinterpret_cast<cutlass::half_t*>(params.ptrPartialO) +
        (((((mBatchIdx) * (mGridDimY) + (mHeadIdx)) * (params.mMaxNumCtasQ)) + (mCtaIdxQ)) *
         (((params.mMaxNumCtasKv) * (int32_t{16})) * (int32_t{256})));
      // TmemCorr.h:2592
      cutlass::half_t* ptrPartialCtaO{
        (ptrPartialO + ((mCtaIdxKv) * (int32_t{16})) * (int32_t{256}))};
      // TmemCorr.h:416
      cutlass::Array<float, 4> scales16;
      // TmemCorr.h:1962
      trtllm::dev::reduceColSum<int32_t{4}>(prodStatsPtr116,
                                            tmemCorr0DstStack.mDepSmemPtr9,
                                            int32_t{4},
                                            int32_t{128},
                                            mWarpGrpThreadIdx,
                                            int32_t{4});
      // TmemCorr.h:500
      {
        // TmemCorr.h:509
        float sum0{prodStatsPtr116[int32_t{0}]};
        // TmemCorr.h:515
        float sum1{prodStatsPtr116[int32_t{1}]};
        // TmemCorr.h:582
        prodStatsPtr116[int32_t{0}] = sum0;
        // TmemCorr.h:583
        prodStatsPtr116[int32_t{1}] = sum1;
        // TmemCorr.h:590
        scales16[int32_t{0}] = float(bool{params.ptrOutputScale == nullptr})
                                 ? (params.mOutputScale)
                                 : (float{params.ptrOutputScale[int32_t{0}]});
        // TmemCorr.h:591
        scales16[int32_t{1}] = float(bool{params.ptrOutputScale == nullptr})
                                 ? (params.mOutputScale)
                                 : (float{params.ptrOutputScale[int32_t{0}]});
      }
      // TmemCorr.h:500
      {
        // TmemCorr.h:509
        float sum0{prodStatsPtr116[int32_t{2}]};
        // TmemCorr.h:515
        float sum1{prodStatsPtr116[int32_t{3}]};
        // TmemCorr.h:582
        prodStatsPtr116[int32_t{2}] = sum0;
        // TmemCorr.h:583
        prodStatsPtr116[int32_t{3}] = sum1;
        // TmemCorr.h:590
        scales16[int32_t{2}] = float(bool{params.ptrOutputScale == nullptr})
                                 ? (params.mOutputScale)
                                 : (float{params.ptrOutputScale[int32_t{0}]});
        // TmemCorr.h:591
        scales16[int32_t{3}] = float(bool{params.ptrOutputScale == nullptr})
                                 ? (params.mOutputScale)
                                 : (float{params.ptrOutputScale[int32_t{0}]});
      }
      // TmemCorr.h:3797
      trtllm::dev::storeStatsForSwappedAb<int32_t{4}, false>(
        (prodStatsPtr116 + int32_t{4}),
        prodStatsPtr116,
        (ptrPartialStats + ((mCtaIdxKv) * (int32_t{16})) * (int32_t{2})),
        params.mNumHeadsQ,
        params.mNumHeadsQPerKv,
        mWarpGrpThreadIdx,
        int32_t{min(params.mNumHeadsQPerKv, int32_t{16})});
      //
      // The headDimStageIdx: 0.
      //
      // TmemCorr.h:1472
      for (int32_t loopOffset2068 = int32_t{0}; loopOffset2068 < int32_t{16};
           loopOffset2068 += int32_t{16}) {
        // TmemTile.cpp:373
        cutlass::Array<float, 16> tmemRegs016;
        // TmemTile.cpp:527
        {
          // TmemTile.cpp:529
          uint32_t tmemBasePtr{mTmemBaseOffset};
          // TmemTile.cpp:545
          uint32_t(&dstSlice0)[8]{reinterpret_cast<uint32_t(&)[8]>(tmemRegs016[int32_t{0}])};
          // CudaPtx.h:48
          cuda_ptx::tcgen05_ld_16x256b(
            dstSlice0,
            (tmemBasePtr) +
              (static_cast<uint32_t>((int32_t{0x60 /*hi=0, lo=96*/}) + (loopOffset2068))));
          // TmemTile.cpp:545
          uint32_t(&dstSlice1)[8]{reinterpret_cast<uint32_t(&)[8]>(tmemRegs016[int32_t{8}])};
          // CudaPtx.h:48
          cuda_ptx::tcgen05_ld_16x256b(
            dstSlice1,
            (tmemBasePtr) +
              (static_cast<uint32_t>(((int32_t{0x60 /*hi=0, lo=96*/}) + (loopOffset2068)) +
                                     (int32_t{0x100000 /*hi=16, lo=0*/}))));
        }
        // TmemCorr.h:3257
        uint32_t mRegsO16[8];
        // TmemCorr.h:1520
        {
          // TmemCorr.h:1540
          cutlass::Array<float, 2> localScales0{scales16[int32_t{0}], scales16[int32_t{1}]};
          // TmemCorr.h:1551
          cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{0}], tmemRegs016[int32_t{1}]};
          // TmemCorr.h:1563
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1566
          tmemRegs016[int32_t{0}] = vals0[int32_t{0}];
          // TmemCorr.h:1567
          tmemRegs016[int32_t{1}] = vals0[int32_t{1}];
          // TmemCorr.h:3487
          mRegsO16[int32_t{0}] =
            trtllm::dev::convert_float2_to_half(tmemRegs016[int32_t{0}], tmemRegs016[int32_t{1}]);
        }
        // TmemCorr.h:1520
        {
          // TmemCorr.h:1540
          cutlass::Array<float, 2> localScales0{scales16[int32_t{0}], scales16[int32_t{1}]};
          // TmemCorr.h:1551
          cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{2}], tmemRegs016[int32_t{3}]};
          // TmemCorr.h:1563
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1566
          tmemRegs016[int32_t{2}] = vals0[int32_t{0}];
          // TmemCorr.h:1567
          tmemRegs016[int32_t{3}] = vals0[int32_t{1}];
          // TmemCorr.h:3487
          mRegsO16[int32_t{1}] =
            trtllm::dev::convert_float2_to_half(tmemRegs016[int32_t{2}], tmemRegs016[int32_t{3}]);
        }
        // TmemCorr.h:1520
        {
          // TmemCorr.h:1540
          cutlass::Array<float, 2> localScales0{scales16[int32_t{2}], scales16[int32_t{3}]};
          // TmemCorr.h:1551
          cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{4}], tmemRegs016[int32_t{5}]};
          // TmemCorr.h:1563
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1566
          tmemRegs016[int32_t{4}] = vals0[int32_t{0}];
          // TmemCorr.h:1567
          tmemRegs016[int32_t{5}] = vals0[int32_t{1}];
          // TmemCorr.h:3487
          mRegsO16[int32_t{2}] =
            trtllm::dev::convert_float2_to_half(tmemRegs016[int32_t{4}], tmemRegs016[int32_t{5}]);
        }
        // TmemCorr.h:1520
        {
          // TmemCorr.h:1540
          cutlass::Array<float, 2> localScales0{scales16[int32_t{2}], scales16[int32_t{3}]};
          // TmemCorr.h:1551
          cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{6}], tmemRegs016[int32_t{7}]};
          // TmemCorr.h:1563
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1566
          tmemRegs016[int32_t{6}] = vals0[int32_t{0}];
          // TmemCorr.h:1567
          tmemRegs016[int32_t{7}] = vals0[int32_t{1}];
          // TmemCorr.h:3487
          mRegsO16[int32_t{3}] =
            trtllm::dev::convert_float2_to_half(tmemRegs016[int32_t{6}], tmemRegs016[int32_t{7}]);
        }
        // TmemCorr.h:1520
        {
          // TmemCorr.h:1540
          cutlass::Array<float, 2> localScales0{scales16[int32_t{0}], scales16[int32_t{1}]};
          // TmemCorr.h:1551
          cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{8}], tmemRegs016[int32_t{9}]};
          // TmemCorr.h:1563
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1566
          tmemRegs016[int32_t{8}] = vals0[int32_t{0}];
          // TmemCorr.h:1567
          tmemRegs016[int32_t{9}] = vals0[int32_t{1}];
          // TmemCorr.h:3487
          mRegsO16[int32_t{4}] =
            trtllm::dev::convert_float2_to_half(tmemRegs016[int32_t{8}], tmemRegs016[int32_t{9}]);
        }
        // TmemCorr.h:1520
        {
          // TmemCorr.h:1540
          cutlass::Array<float, 2> localScales0{scales16[int32_t{0}], scales16[int32_t{1}]};
          // TmemCorr.h:1551
          cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{10}], tmemRegs016[int32_t{11}]};
          // TmemCorr.h:1563
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1566
          tmemRegs016[int32_t{10}] = vals0[int32_t{0}];
          // TmemCorr.h:1567
          tmemRegs016[int32_t{11}] = vals0[int32_t{1}];
          // TmemCorr.h:3487
          mRegsO16[int32_t{5}] =
            trtllm::dev::convert_float2_to_half(tmemRegs016[int32_t{10}], tmemRegs016[int32_t{11}]);
        }
        // TmemCorr.h:1520
        {
          // TmemCorr.h:1540
          cutlass::Array<float, 2> localScales0{scales16[int32_t{2}], scales16[int32_t{3}]};
          // TmemCorr.h:1551
          cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{12}], tmemRegs016[int32_t{13}]};
          // TmemCorr.h:1563
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1566
          tmemRegs016[int32_t{12}] = vals0[int32_t{0}];
          // TmemCorr.h:1567
          tmemRegs016[int32_t{13}] = vals0[int32_t{1}];
          // TmemCorr.h:3487
          mRegsO16[int32_t{6}] =
            trtllm::dev::convert_float2_to_half(tmemRegs016[int32_t{12}], tmemRegs016[int32_t{13}]);
        }
        // TmemCorr.h:1520
        {
          // TmemCorr.h:1540
          cutlass::Array<float, 2> localScales0{scales16[int32_t{2}], scales16[int32_t{3}]};
          // TmemCorr.h:1551
          cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{14}], tmemRegs016[int32_t{15}]};
          // TmemCorr.h:1563
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1566
          tmemRegs016[int32_t{14}] = vals0[int32_t{0}];
          // TmemCorr.h:1567
          tmemRegs016[int32_t{15}] = vals0[int32_t{1}];
          // TmemCorr.h:3487
          mRegsO16[int32_t{7}] =
            trtllm::dev::convert_float2_to_half(tmemRegs016[int32_t{14}], tmemRegs016[int32_t{15}]);
        }
        // TmemCorr.h:3665
        trtllm::dev::reorganizeInSmemAndStoreToDstMem<int32_t{128}, int32_t{16}, false>(
          reinterpret_cast<cutlass::half_t*>(tmemCorr0DstStack.mDepSmemPtr6),
          ptrPartialCtaO,
          mRegsO16,
          int32_t{256},
          int32_t{min(params.mNumHeadsQPerKv, int32_t{16})},
          params.mNumHeadsQ,
          params.mNumHeadsQPerKv,
          int32_t{128},
          mWarpGrpThreadIdx,
          int32_t{3});
        // TmemCorr.h:3867
        trtllm::dev::CutlassNamedBarrier::sync(128, 3);
        // TmemCorr.h:3879
        ptrPartialCtaO += int32_t{128};
      }
      //
      // The headDimStageIdx: 1.
      //
      // TmemCorr.h:1472
      for (int32_t loopOffset2138 = int32_t{0}; loopOffset2138 < int32_t{16};
           loopOffset2138 += int32_t{16}) {
        // TmemTile.cpp:373
        cutlass::Array<float, 16> tmemRegs016;
        // TmemTile.cpp:527
        {
          // TmemTile.cpp:529
          uint32_t tmemBasePtr{mTmemBaseOffset};
          // TmemTile.cpp:545
          uint32_t(&dstSlice0)[8]{reinterpret_cast<uint32_t(&)[8]>(tmemRegs016[int32_t{0}])};
          // CudaPtx.h:48
          cuda_ptx::tcgen05_ld_16x256b(
            dstSlice0,
            (tmemBasePtr) +
              (static_cast<uint32_t>((int32_t{0x70 /*hi=0, lo=112*/}) + (loopOffset2138))));
          // TmemTile.cpp:545
          uint32_t(&dstSlice1)[8]{reinterpret_cast<uint32_t(&)[8]>(tmemRegs016[int32_t{8}])};
          // CudaPtx.h:48
          cuda_ptx::tcgen05_ld_16x256b(
            dstSlice1,
            (tmemBasePtr) +
              (static_cast<uint32_t>(((int32_t{0x70 /*hi=0, lo=112*/}) + (loopOffset2138)) +
                                     (int32_t{0x100000 /*hi=16, lo=0*/}))));
        }
        // TmemCorr.h:3257
        uint32_t mRegsO16[8];
        // TmemCorr.h:1520
        {
          // TmemCorr.h:1540
          cutlass::Array<float, 2> localScales0{scales16[int32_t{0}], scales16[int32_t{1}]};
          // TmemCorr.h:1551
          cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{0}], tmemRegs016[int32_t{1}]};
          // TmemCorr.h:1563
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1566
          tmemRegs016[int32_t{0}] = vals0[int32_t{0}];
          // TmemCorr.h:1567
          tmemRegs016[int32_t{1}] = vals0[int32_t{1}];
          // TmemCorr.h:3487
          mRegsO16[int32_t{0}] =
            trtllm::dev::convert_float2_to_half(tmemRegs016[int32_t{0}], tmemRegs016[int32_t{1}]);
        }
        // TmemCorr.h:1520
        {
          // TmemCorr.h:1540
          cutlass::Array<float, 2> localScales0{scales16[int32_t{0}], scales16[int32_t{1}]};
          // TmemCorr.h:1551
          cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{2}], tmemRegs016[int32_t{3}]};
          // TmemCorr.h:1563
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1566
          tmemRegs016[int32_t{2}] = vals0[int32_t{0}];
          // TmemCorr.h:1567
          tmemRegs016[int32_t{3}] = vals0[int32_t{1}];
          // TmemCorr.h:3487
          mRegsO16[int32_t{1}] =
            trtllm::dev::convert_float2_to_half(tmemRegs016[int32_t{2}], tmemRegs016[int32_t{3}]);
        }
        // TmemCorr.h:1520
        {
          // TmemCorr.h:1540
          cutlass::Array<float, 2> localScales0{scales16[int32_t{2}], scales16[int32_t{3}]};
          // TmemCorr.h:1551
          cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{4}], tmemRegs016[int32_t{5}]};
          // TmemCorr.h:1563
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1566
          tmemRegs016[int32_t{4}] = vals0[int32_t{0}];
          // TmemCorr.h:1567
          tmemRegs016[int32_t{5}] = vals0[int32_t{1}];
          // TmemCorr.h:3487
          mRegsO16[int32_t{2}] =
            trtllm::dev::convert_float2_to_half(tmemRegs016[int32_t{4}], tmemRegs016[int32_t{5}]);
        }
        // TmemCorr.h:1520
        {
          // TmemCorr.h:1540
          cutlass::Array<float, 2> localScales0{scales16[int32_t{2}], scales16[int32_t{3}]};
          // TmemCorr.h:1551
          cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{6}], tmemRegs016[int32_t{7}]};
          // TmemCorr.h:1563
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1566
          tmemRegs016[int32_t{6}] = vals0[int32_t{0}];
          // TmemCorr.h:1567
          tmemRegs016[int32_t{7}] = vals0[int32_t{1}];
          // TmemCorr.h:3487
          mRegsO16[int32_t{3}] =
            trtllm::dev::convert_float2_to_half(tmemRegs016[int32_t{6}], tmemRegs016[int32_t{7}]);
        }
        // TmemCorr.h:1520
        {
          // TmemCorr.h:1540
          cutlass::Array<float, 2> localScales0{scales16[int32_t{0}], scales16[int32_t{1}]};
          // TmemCorr.h:1551
          cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{8}], tmemRegs016[int32_t{9}]};
          // TmemCorr.h:1563
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1566
          tmemRegs016[int32_t{8}] = vals0[int32_t{0}];
          // TmemCorr.h:1567
          tmemRegs016[int32_t{9}] = vals0[int32_t{1}];
          // TmemCorr.h:3487
          mRegsO16[int32_t{4}] =
            trtllm::dev::convert_float2_to_half(tmemRegs016[int32_t{8}], tmemRegs016[int32_t{9}]);
        }
        // TmemCorr.h:1520
        {
          // TmemCorr.h:1540
          cutlass::Array<float, 2> localScales0{scales16[int32_t{0}], scales16[int32_t{1}]};
          // TmemCorr.h:1551
          cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{10}], tmemRegs016[int32_t{11}]};
          // TmemCorr.h:1563
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1566
          tmemRegs016[int32_t{10}] = vals0[int32_t{0}];
          // TmemCorr.h:1567
          tmemRegs016[int32_t{11}] = vals0[int32_t{1}];
          // TmemCorr.h:3487
          mRegsO16[int32_t{5}] =
            trtllm::dev::convert_float2_to_half(tmemRegs016[int32_t{10}], tmemRegs016[int32_t{11}]);
        }
        // TmemCorr.h:1520
        {
          // TmemCorr.h:1540
          cutlass::Array<float, 2> localScales0{scales16[int32_t{2}], scales16[int32_t{3}]};
          // TmemCorr.h:1551
          cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{12}], tmemRegs016[int32_t{13}]};
          // TmemCorr.h:1563
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1566
          tmemRegs016[int32_t{12}] = vals0[int32_t{0}];
          // TmemCorr.h:1567
          tmemRegs016[int32_t{13}] = vals0[int32_t{1}];
          // TmemCorr.h:3487
          mRegsO16[int32_t{6}] =
            trtllm::dev::convert_float2_to_half(tmemRegs016[int32_t{12}], tmemRegs016[int32_t{13}]);
        }
        // TmemCorr.h:1520
        {
          // TmemCorr.h:1540
          cutlass::Array<float, 2> localScales0{scales16[int32_t{2}], scales16[int32_t{3}]};
          // TmemCorr.h:1551
          cutlass::Array<float, 2> vals0{tmemRegs016[int32_t{14}], tmemRegs016[int32_t{15}]};
          // TmemCorr.h:1563
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1566
          tmemRegs016[int32_t{14}] = vals0[int32_t{0}];
          // TmemCorr.h:1567
          tmemRegs016[int32_t{15}] = vals0[int32_t{1}];
          // TmemCorr.h:3487
          mRegsO16[int32_t{7}] =
            trtllm::dev::convert_float2_to_half(tmemRegs016[int32_t{14}], tmemRegs016[int32_t{15}]);
        }
        // TmemCorr.h:3665
        trtllm::dev::reorganizeInSmemAndStoreToDstMem<int32_t{128}, int32_t{16}, false>(
          reinterpret_cast<cutlass::half_t*>(tmemCorr0DstStack.mDepSmemPtr6),
          ptrPartialCtaO,
          mRegsO16,
          int32_t{256},
          int32_t{min(params.mNumHeadsQPerKv, int32_t{16})},
          params.mNumHeadsQ,
          params.mNumHeadsQPerKv,
          int32_t{128},
          mWarpGrpThreadIdx,
          int32_t{3});
        // TmemCorr.h:3867
        trtllm::dev::CutlassNamedBarrier::sync(128, 3);
        // TmemCorr.h:3879
        ptrPartialCtaO += int32_t{128};
      }
      // TmemCorr.h:3341
      int32_t ctaIdxKvForReduction{trtllm::dev::recordCtaCompletion(
        (params.ptrMultiCtasKvCounter +
         (((mBatchIdx) * (mGridDimY) + (mHeadIdx)) * (params.mMaxNumCtasQ)) + (mCtaIdxQ)),
        reinterpret_cast<int32_t*>(tmemCorr0DstStack.mDepSmemPtr9),
        mWarpGrpThreadIdx,
        mNumCtasKv,
        numCtasKvForReduction,
        int32_t{128},
        int32_t{3})};
      // TmemCorr.h:3352
      if ((ctaIdxKvForReduction) < (numCtasKvForReduction)) {
        // TmemCorr.h:3394
        trtllm::dev::reducePartialO<int32_t{16},
                                    int32_t{256},
                                    int32_t{256},
                                    int32_t{128},
                                    false,
                                    false,
                                    false>(ptrO,
                                           ptrPartialO,
                                           ptrPartialStats,
                                           params.ptrAttentionSinks,
                                           ptrSoftmaxStats,
                                           scaleSoftmaxLog216,
                                           mNumCtasKv,
                                           mWarpGrpThreadIdx,
                                           ctaIdxKvForReduction,
                                           headIdxO,
                                           params.mNumHeadsQ,
                                           params.mNumHeadsQPerKvDivisor,
                                           int32_t{min(params.mNumHeadsQPerKv, int32_t{16})},
                                           storesSoftmaxStats);
      }
    }
    //
    // tmemO [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{34}, Flags{1024}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:2561
      {
        // Task.cpp:2585
        tmemOSrcStack.mPipeline.consumer_release(tmemOConsReleaseState);
      }
      // Task.cpp:43
      ++tmemOConsReleaseState;
    }
    //
    // Tail work.
    //
    //
    // tmemSoftmaxLocal0 [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    // Task.cpp:2668
    {}
  // Task.cpp:3511
  ExitTileWithSignalingLabel:
  // Task.cpp:3518
  ExitTileWithoutSignalingLabel:
    // Task.cpp:3528
    {}
    // Task.cpp:497
    cudaTriggerProgrammaticLaunchCompletion();
  }
};
// Task.cpp:544
// Fmha.h:2434
struct TransformKvTask {
  // Task.cpp:283
  int32_t mCtaIdxX;
  // Task.cpp:287
  int32_t mCtaIdxY;
  // Task.cpp:291
  int32_t mCtaIdxZ;
  // FmhaTask.h:220
  int32_t mHeadIdx;
  // FmhaTask.h:218
  int32_t mBatchIdx;
  // FmhaTask.h:208
  int32_t mSeqOffsetQ;
  // FmhaTask.h:210
  int32_t mSeqLenQ;
  // FmhaTask.h:224
  int32_t mCtaIdxQ;
  // FmhaTask.h:226
  int32_t mCtaIdxKv;
  // FmhaTask.h:214
  int32_t mSeqLenKv;
  // FmhaTask.h:216
  int32_t mNumCtasKv;
  // Task.cpp:371
  int32_t const mWarpGrpThreadIdx;
  // Task.cpp:551
  inline __device__ TransformKvTask(fmha::KernelParams const& params,
                                    KernelState const& state,
                                    int32_t warpGrpStart)
    : // Kernel.cpp:194
    mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , // Kernel.cpp:195
    mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , // Kernel.cpp:196
    mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)}
    , // Task.cpp:287
    mHeadIdx{mCtaIdxY}
    , // Task.cpp:291
    mBatchIdx{mCtaIdxZ}
    , // FmhaTask.h:394
    mSeqOffsetQ{int32_t(bool{params.ptrCumSeqLensQ == nullptr})
                  ? ((mBatchIdx) * (params.mMaxSeqLenQ))
                  : (int32_t{params.ptrCumSeqLensQ[mBatchIdx]})}
    , // FmhaTask.h:410
    mSeqLenQ{int32_t(bool{params.ptrCumSeqLensQ == nullptr})
               ? (params.mMaxSeqLenQ)
               : ((int32_t{params.ptrCumSeqLensQ[(mBatchIdx) + (int32_t{1})]}) - (mSeqOffsetQ))}
    , // FmhaTask.h:516
    mCtaIdxQ{(mCtaIdxX) / (params.mMaxNumCtasKv)}
    , // FmhaTask.h:517
    mCtaIdxKv{(mCtaIdxX) % (params.mMaxNumCtasKv)}
    , // FmhaTask.h:543
    mSeqLenKv{(int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                             ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                             : (mBatchIdx)]}) -
              (((mSeqLenQ) - (int32_t{1})) - (mCtaIdxQ))}
    , // FmhaTask.h:565
    mNumCtasKv{
      int32_t{min(int32_t{((mSeqLenKv) + (int32_t{127})) / (int32_t{128})}, params.mMaxNumCtasKv)}}
    , // Task.cpp:379
    mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))} {}
  // Task.cpp:507
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:532
    return ((state.mWarpIdx) >= (int32_t{12})) && ((state.mWarpIdx) < (int32_t{16}));
  }
  // Task.cpp:454
  inline __device__ void execute(fmha::KernelParams const& params,
                                 KernelState const& state,
                                 SmemTransformedKvSmem& smemTransformedKvDstSmem,
                                 SmemTransformedKvStack& smemTransformedKvDstStack,
                                 SmemKvSmem& smemKvSrcSmem,
                                 SmemKvStack& smemKvSrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_inc(cuda_ptx::n32_t<184>{});
    // Task.cpp:2079
    trtllm::dev::CutlassTmaAsyncPipeline<4>::PipelineState smemKvConsState{};
    // Task.cpp:2086
    trtllm::dev::CutlassTmaAsyncPipeline<4>::PipelineState smemKvConsReleaseState{};
    // Task.cpp:2100
    int32_t smemKvConsToken{int32_t{0}};
    // Task.cpp:1979
    trtllm::dev::CutlassUmmaConsumerAsyncPipeline<2, false, false>::PipelineState
      smemTransformedKvProdState{int32_t{0}, int32_t{1}, int32_t{0}};
    // Task.cpp:1999
    int32_t smemTransformedKvProdToken{int32_t{1}};
    // FmhaTask.h:582
    int32_t numLoopSteps;
    // FmhaTask.h:630
    if (((mCtaIdxQ) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
      // FmhaTask.h:668
      int32_t numSteps{((mSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
                       ((mNumCtasKv) * (int32_t{128}))};
      // FmhaTask.h:682
      numLoopSteps = numSteps;
    } else {
      // FmhaTask.h:651
      return;
    }
    // Task.cpp:3168
    bool const hasOneLoopIter{(int32_t{0}) < (numLoopSteps)};
    // Task.cpp:3179
    int32_t lastLoopOffset{int32_t{0}};
    //
    // Hoist the first iter.
    //
    //
    // smemKv [ConsWait, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:2745
        smemKvConsToken = smemKvSrcStack.mPipeline.consumer_try_wait(smemKvConsState);
      }
      // Task.cpp:2813
      smemKvSrcStack.mPipeline.consumer_wait(smemKvConsState, smemKvConsToken);
    }
    //
    // smemKv [ConsWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // SmemKv.h:199
    cutlass::float_e4m3_t* smemPtrK3;
    // SmemKv.h:206
    int32_t smemIdxK3;
    // SmemKv.h:214
    cutlass::float_e4m3_t* smemPtrV3;
    // SmemKv.h:221
    int32_t smemIdxV3;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{smemKvConsState.index()};
      // SmemKv.h:244
      smemPtrK3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
      // SmemKv.h:246
      smemIdxK3 = index;
      // Task.cpp:43
      ++smemKvConsState;
    }
    //
    // smemTransformedKv [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4962
      {
        // Task.cpp:4984
        smemTransformedKvProdToken =
          smemTransformedKvDstStack.mPipeline.producer_try_acquire(smemTransformedKvProdState);
      }
    }
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4210
      smemTransformedKvDstStack.mPipeline.producer_acquire(smemTransformedKvProdState,
                                                           smemTransformedKvProdToken);
    }
    //
    // smemTransformedKv [ProdWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // SmemTransformedKv.h:108
    cutlass::float_e4m3_t* smemPtrK4;
    // SmemTransformedKv.h:113
    int32_t smemIdxK4;
    // SmemTransformedKv.h:118
    cutlass::float_e4m3_t* smemPtrV4;
    // SmemTransformedKv.h:123
    int32_t smemIdxV4;
    // Task.cpp:1477
    smemPtrK4 = smemPtrK3;
    // Task.cpp:1477
    smemIdxK4 = smemIdxK3;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{smemTransformedKvProdState.index()};
      // SmemTransformedKv.h:214
      cutlass::float_e4m3_t* srcPtr;
      // SmemTransformedKv.h:215
      srcPtr = smemPtrK4 + ((smemIdxK4) * (int32_t{16384}));
      // SmemTransformedKv.h:241
      uint64_t srcBuffer4[16];
      // SmemTransformedKv.h:261
      CUTLASS_PRAGMA_UNROLL
      for (int32_t loopOffset2301 = int32_t{0}; loopOffset2301 < int32_t{2}; ++loopOffset2301) {
        // SmemTransformedKv.h:284
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset2303 = (loopOffset2301) * (int32_t{1024});
             loopOffset2303 < ((loopOffset2301) + (int32_t{1})) * (int32_t{1024});
             loopOffset2303 += int32_t{128}) {
          // SmemTransformedKv.h:293
          int32_t offset{(loopOffset2303) + (mWarpGrpThreadIdx)};
          // SmemTransformedKv.h:304
          srcBuffer4[(loopOffset2303) / (int32_t{128})] =
            reinterpret_cast<uint64_t*>(srcPtr)[offset];
        }
        // SmemTransformedKv.h:348
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset2307 = (loopOffset2301) * (int32_t{1024});
             loopOffset2307 < ((loopOffset2301) + (int32_t{1})) * (int32_t{1024});
             loopOffset2307 += int32_t{128}) {
          // SmemTransformedKv.h:356
          int32_t offset{(loopOffset2307) + (mWarpGrpThreadIdx)};
          // SmemTransformedKv.h:385
          cutlass::uint128_t dst{
            trtllm::dev::convertE4m3ToFp16(srcBuffer4[(loopOffset2307) / (int32_t{128})])};
          // SmemTransformedKv.h:397
          int32_t eltIdx{(offset) * (int32_t{8})};
          // SmemTile.cpp:389
          int32_t smemRowIdx{(((eltIdx) % (int32_t{128})) / (int32_t{64})) * (int32_t{128}) +
                             ((eltIdx) / (int32_t{128}))};
          // SmemTile.cpp:396
          int32_t smemOffsetInBytes{((smemRowIdx) * (int32_t{128})) +
                                    ((((eltIdx) % (int32_t{64})) * (int32_t{16})) / (int32_t{8}))};
          // SmemTile.cpp:416
          int32_t swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
          // SmemTransformedKv.h:414
          reinterpret_cast<cutlass::uint128_t*>(
            smemTransformedKvDstSmem
              .mArray[index])[((smemOffsetInBytes) ^ (swizzleMask)) / (int32_t{16})] = dst;
        }
      }
      // SmemTransformedKv.h:423
      cutlass::arch::fence_view_async_shared();
    }
    //
    // smemTransformedKv [ProdCommit, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4427
      {
        // Task.cpp:4443
        smemTransformedKvDstStack.mPipeline.producer_commit(smemTransformedKvProdState);
      }
      // Task.cpp:43
      ++smemTransformedKvProdState;
    }
    //
    // smemKv [ConsRelease, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{1024}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:2453
      trtllm::dev::CutlassNamedBarrier::sync(128, 6);
      // Task.cpp:2561
      {
        // Task.cpp:2585
        smemKvSrcStack.mPipeline.consumer_release(smemKvConsReleaseState);
      }
      // Task.cpp:43
      ++smemKvConsReleaseState;
    }
    //
    // smemKv [ConsWait, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:2745
        smemKvConsToken = smemKvSrcStack.mPipeline.consumer_try_wait(smemKvConsState);
      }
      // Task.cpp:2813
      smemKvSrcStack.mPipeline.consumer_wait(smemKvConsState, smemKvConsToken);
    }
    //
    // smemKv [ConsWork (call 1), FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{smemKvConsState.index()};
      // SmemKv.h:244
      smemPtrK3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
      // SmemKv.h:246
      smemIdxK3 = index;
      // Task.cpp:43
      ++smemKvConsState;
    }
    //
    // smemTransformedKv [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4962
      {
        // Task.cpp:4984
        smemTransformedKvProdToken =
          smemTransformedKvDstStack.mPipeline.producer_try_acquire(smemTransformedKvProdState);
      }
    }
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4210
      smemTransformedKvDstStack.mPipeline.producer_acquire(smemTransformedKvProdState,
                                                           smemTransformedKvProdToken);
    }
    //
    // smemTransformedKv [ProdWork (call 1), FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1477
    smemPtrK4 = smemPtrK3;
    // Task.cpp:1477
    smemIdxK4 = smemIdxK3;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{smemTransformedKvProdState.index()};
      // SmemTransformedKv.h:214
      cutlass::float_e4m3_t* srcPtr;
      // SmemTransformedKv.h:215
      srcPtr = smemPtrK4 + ((smemIdxK4) * (int32_t{16384}));
      // SmemTransformedKv.h:241
      uint64_t srcBuffer4[16];
      // SmemTransformedKv.h:261
      CUTLASS_PRAGMA_UNROLL
      for (int32_t loopOffset2375 = int32_t{0}; loopOffset2375 < int32_t{2}; ++loopOffset2375) {
        // SmemTransformedKv.h:284
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset2377 = (loopOffset2375) * (int32_t{1024});
             loopOffset2377 < ((loopOffset2375) + (int32_t{1})) * (int32_t{1024});
             loopOffset2377 += int32_t{128}) {
          // SmemTransformedKv.h:293
          int32_t offset{(loopOffset2377) + (mWarpGrpThreadIdx)};
          // SmemTransformedKv.h:304
          srcBuffer4[(loopOffset2377) / (int32_t{128})] =
            reinterpret_cast<uint64_t*>(srcPtr)[offset];
        }
        // SmemTransformedKv.h:348
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset2381 = (loopOffset2375) * (int32_t{1024});
             loopOffset2381 < ((loopOffset2375) + (int32_t{1})) * (int32_t{1024});
             loopOffset2381 += int32_t{128}) {
          // SmemTransformedKv.h:356
          int32_t offset{(loopOffset2381) + (mWarpGrpThreadIdx)};
          // SmemTransformedKv.h:385
          cutlass::uint128_t dst{
            trtllm::dev::convertE4m3ToFp16(srcBuffer4[(loopOffset2381) / (int32_t{128})])};
          // SmemTransformedKv.h:397
          int32_t eltIdx{(offset) * (int32_t{8})};
          // SmemTile.cpp:389
          int32_t smemRowIdx{(((eltIdx) % (int32_t{128})) / (int32_t{64})) * (int32_t{128}) +
                             ((eltIdx) / (int32_t{128}))};
          // SmemTile.cpp:396
          int32_t smemOffsetInBytes{((smemRowIdx) * (int32_t{128})) +
                                    ((((eltIdx) % (int32_t{64})) * (int32_t{16})) / (int32_t{8}))};
          // SmemTile.cpp:416
          int32_t swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
          // SmemTransformedKv.h:414
          reinterpret_cast<cutlass::uint128_t*>(
            smemTransformedKvDstSmem
              .mArray[index])[((smemOffsetInBytes) ^ (swizzleMask)) / (int32_t{16})] = dst;
        }
      }
      // SmemTransformedKv.h:423
      cutlass::arch::fence_view_async_shared();
    }
    //
    // smemTransformedKv [ProdCommit, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4427
      {
        // Task.cpp:4443
        smemTransformedKvDstStack.mPipeline.producer_commit(smemTransformedKvProdState);
      }
      // Task.cpp:43
      ++smemTransformedKvProdState;
    }
    //
    // smemKv [ConsRelease, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{1024}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:2453
      trtllm::dev::CutlassNamedBarrier::sync(128, 6);
      // Task.cpp:2561
      {
        // Task.cpp:2585
        smemKvSrcStack.mPipeline.consumer_release(smemKvConsReleaseState);
      }
      // Task.cpp:43
      ++smemKvConsReleaseState;
    }
    //
    // Loop body.
    //
    // Task.cpp:3350
    CUTLASS_PRAGMA_NO_UNROLL
    for (int32_t loopOffset2411 = int32_t{0}; loopOffset2411 < (numLoopSteps) - (int32_t{1});
         ++loopOffset2411) {
      //
      // smemKv [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:2781
      {
        // Task.cpp:1573
        // Task.cpp:2722
        {
          // Task.cpp:2745
          smemKvConsToken = smemKvSrcStack.mPipeline.consumer_try_wait(smemKvConsState);
        }
        // Task.cpp:2813
        smemKvSrcStack.mPipeline.consumer_wait(smemKvConsState, smemKvConsToken);
      }
      //
      // smemKv [ConsWork (call 2), Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:2893
      {
        // Task.cpp:5829
        int32_t index{smemKvConsState.index()};
        // SmemKv.h:244
        smemPtrK3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:246
        smemIdxK3 = index;
        // Task.cpp:43
        ++smemKvConsState;
      }
      //
      // smemTransformedKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:4948
      {
        // Task.cpp:4962
        if ((loopOffset2411) >= (int32_t{0})) {
          // Task.cpp:4984
          smemTransformedKvProdToken =
            smemTransformedKvDstStack.mPipeline.producer_try_acquire(smemTransformedKvProdState);
        }
      }
      // Task.cpp:1573
      // Task.cpp:4180
      {
        // Task.cpp:4210
        smemTransformedKvDstStack.mPipeline.producer_acquire(smemTransformedKvProdState,
                                                             smemTransformedKvProdToken);
      }
      //
      // smemTransformedKv [ProdWork (call 2), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1477
      smemPtrK4 = smemPtrK3;
      // Task.cpp:1477
      smemIdxK4 = smemIdxK3;
      // Task.cpp:1573
      // Task.cpp:5038
      {
        // Task.cpp:5829
        int32_t index{smemTransformedKvProdState.index()};
        // SmemTransformedKv.h:214
        cutlass::float_e4m3_t* srcPtr;
        // SmemTransformedKv.h:215
        srcPtr = smemPtrK4 + ((smemIdxK4) * (int32_t{16384}));
        // SmemTransformedKv.h:241
        uint64_t srcBuffer4[16];
        // SmemTransformedKv.h:261
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset2440 = int32_t{0}; loopOffset2440 < int32_t{2}; ++loopOffset2440) {
          // SmemTransformedKv.h:284
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset2442 = (loopOffset2440) * (int32_t{1024});
               loopOffset2442 < ((loopOffset2440) + (int32_t{1})) * (int32_t{1024});
               loopOffset2442 += int32_t{128}) {
            // SmemTransformedKv.h:293
            int32_t offset{(loopOffset2442) + (mWarpGrpThreadIdx)};
            // SmemTransformedKv.h:304
            srcBuffer4[(loopOffset2442) / (int32_t{128})] =
              reinterpret_cast<uint64_t*>(srcPtr)[offset];
          }
          // SmemTransformedKv.h:348
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset2446 = (loopOffset2440) * (int32_t{1024});
               loopOffset2446 < ((loopOffset2440) + (int32_t{1})) * (int32_t{1024});
               loopOffset2446 += int32_t{128}) {
            // SmemTransformedKv.h:356
            int32_t offset{(loopOffset2446) + (mWarpGrpThreadIdx)};
            // SmemTransformedKv.h:385
            cutlass::uint128_t dst{
              trtllm::dev::convertE4m3ToFp16(srcBuffer4[(loopOffset2446) / (int32_t{128})])};
            // SmemTransformedKv.h:397
            int32_t eltIdx{(offset) * (int32_t{8})};
            // SmemTile.cpp:389
            int32_t smemRowIdx{(((eltIdx) % (int32_t{128})) / (int32_t{64})) * (int32_t{128}) +
                               ((eltIdx) / (int32_t{128}))};
            // SmemTile.cpp:396
            int32_t smemOffsetInBytes{
              ((smemRowIdx) * (int32_t{128})) +
              ((((eltIdx) % (int32_t{64})) * (int32_t{16})) / (int32_t{8}))};
            // SmemTile.cpp:416
            int32_t swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
            // SmemTransformedKv.h:414
            reinterpret_cast<cutlass::uint128_t*>(
              smemTransformedKvDstSmem
                .mArray[index])[((smemOffsetInBytes) ^ (swizzleMask)) / (int32_t{16})] = dst;
          }
        }
        // SmemTransformedKv.h:423
        cutlass::arch::fence_view_async_shared();
      }
      //
      // smemTransformedKv [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:4409
      {
        // Task.cpp:4427
        {
          // Task.cpp:4443
          smemTransformedKvDstStack.mPipeline.producer_commit(smemTransformedKvProdState);
        }
        // Task.cpp:43
        ++smemTransformedKvProdState;
      }
      //
      // smemKv [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{1024}].
      //
      // Task.cpp:2533
      if ((loopOffset2411) >= (int32_t{0})) {
        // Task.cpp:2453
        trtllm::dev::CutlassNamedBarrier::sync(128, 6);
        // Task.cpp:2561
        {
          // Task.cpp:2585
          smemKvSrcStack.mPipeline.consumer_release(smemKvConsReleaseState);
        }
        // Task.cpp:43
        ++smemKvConsReleaseState;
      }
      //
      // smemKv [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:2781
      {
        // Task.cpp:1573
        // Task.cpp:2722
        {
          // Task.cpp:2745
          smemKvConsToken = smemKvSrcStack.mPipeline.consumer_try_wait(smemKvConsState);
        }
        // Task.cpp:2813
        smemKvSrcStack.mPipeline.consumer_wait(smemKvConsState, smemKvConsToken);
      }
      //
      // smemKv [ConsWork (call 3), Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:2893
      {
        // Task.cpp:5829
        int32_t index{smemKvConsState.index()};
        // SmemKv.h:244
        smemPtrK3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:246
        smemIdxK3 = index;
        // Task.cpp:43
        ++smemKvConsState;
      }
      //
      // smemTransformedKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:4948
      {
        // Task.cpp:4962
        if ((loopOffset2411) >= (int32_t{0})) {
          // Task.cpp:4984
          smemTransformedKvProdToken =
            smemTransformedKvDstStack.mPipeline.producer_try_acquire(smemTransformedKvProdState);
        }
      }
      // Task.cpp:1573
      // Task.cpp:4180
      {
        // Task.cpp:4210
        smemTransformedKvDstStack.mPipeline.producer_acquire(smemTransformedKvProdState,
                                                             smemTransformedKvProdToken);
      }
      //
      // smemTransformedKv [ProdWork (call 3), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1477
      smemPtrK4 = smemPtrK3;
      // Task.cpp:1477
      smemIdxK4 = smemIdxK3;
      // Task.cpp:1573
      // Task.cpp:5038
      {
        // Task.cpp:5829
        int32_t index{smemTransformedKvProdState.index()};
        // SmemTransformedKv.h:214
        cutlass::float_e4m3_t* srcPtr;
        // SmemTransformedKv.h:215
        srcPtr = smemPtrK4 + ((smemIdxK4) * (int32_t{16384}));
        // SmemTransformedKv.h:241
        uint64_t srcBuffer4[16];
        // SmemTransformedKv.h:261
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset2502 = int32_t{0}; loopOffset2502 < int32_t{2}; ++loopOffset2502) {
          // SmemTransformedKv.h:284
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset2504 = (loopOffset2502) * (int32_t{1024});
               loopOffset2504 < ((loopOffset2502) + (int32_t{1})) * (int32_t{1024});
               loopOffset2504 += int32_t{128}) {
            // SmemTransformedKv.h:293
            int32_t offset{(loopOffset2504) + (mWarpGrpThreadIdx)};
            // SmemTransformedKv.h:304
            srcBuffer4[(loopOffset2504) / (int32_t{128})] =
              reinterpret_cast<uint64_t*>(srcPtr)[offset];
          }
          // SmemTransformedKv.h:348
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset2508 = (loopOffset2502) * (int32_t{1024});
               loopOffset2508 < ((loopOffset2502) + (int32_t{1})) * (int32_t{1024});
               loopOffset2508 += int32_t{128}) {
            // SmemTransformedKv.h:356
            int32_t offset{(loopOffset2508) + (mWarpGrpThreadIdx)};
            // SmemTransformedKv.h:385
            cutlass::uint128_t dst{
              trtllm::dev::convertE4m3ToFp16(srcBuffer4[(loopOffset2508) / (int32_t{128})])};
            // SmemTransformedKv.h:397
            int32_t eltIdx{(offset) * (int32_t{8})};
            // SmemTile.cpp:389
            int32_t smemRowIdx{(((eltIdx) % (int32_t{128})) / (int32_t{64})) * (int32_t{128}) +
                               ((eltIdx) / (int32_t{128}))};
            // SmemTile.cpp:396
            int32_t smemOffsetInBytes{
              ((smemRowIdx) * (int32_t{128})) +
              ((((eltIdx) % (int32_t{64})) * (int32_t{16})) / (int32_t{8}))};
            // SmemTile.cpp:416
            int32_t swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
            // SmemTransformedKv.h:414
            reinterpret_cast<cutlass::uint128_t*>(
              smemTransformedKvDstSmem
                .mArray[index])[((smemOffsetInBytes) ^ (swizzleMask)) / (int32_t{16})] = dst;
          }
        }
        // SmemTransformedKv.h:423
        cutlass::arch::fence_view_async_shared();
      }
      //
      // smemTransformedKv [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:4409
      {
        // Task.cpp:4427
        {
          // Task.cpp:4443
          smemTransformedKvDstStack.mPipeline.producer_commit(smemTransformedKvProdState);
        }
        // Task.cpp:43
        ++smemTransformedKvProdState;
      }
      //
      // smemKv [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{1024}].
      //
      // Task.cpp:2533
      if ((loopOffset2411) >= (int32_t{0})) {
        // Task.cpp:2453
        trtllm::dev::CutlassNamedBarrier::sync(128, 6);
        // Task.cpp:2561
        {
          // Task.cpp:2585
          smemKvSrcStack.mPipeline.consumer_release(smemKvConsReleaseState);
        }
        // Task.cpp:43
        ++smemKvConsReleaseState;
      }
      //
      // smemKv [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:2781
      {
        // Task.cpp:1573
        // Task.cpp:2722
        {
          // Task.cpp:2745
          smemKvConsToken = smemKvSrcStack.mPipeline.consumer_try_wait(smemKvConsState);
        }
        // Task.cpp:2813
        smemKvSrcStack.mPipeline.consumer_wait(smemKvConsState, smemKvConsToken);
      }
      //
      // smemKv [ConsWork (call 4), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:2893
      {
        // Task.cpp:5829
        int32_t index{smemKvConsState.index()};
        // SmemKv.h:244
        smemPtrV3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:246
        smemIdxV3 = index;
        // Task.cpp:43
        ++smemKvConsState;
      }
      //
      // smemTransformedKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:4948
      {
        // Task.cpp:4962
        if ((loopOffset2411) >= (int32_t{0})) {
          // Task.cpp:4984
          smemTransformedKvProdToken =
            smemTransformedKvDstStack.mPipeline.producer_try_acquire(smemTransformedKvProdState);
        }
      }
      // Task.cpp:1573
      // Task.cpp:4180
      {
        // Task.cpp:4210
        smemTransformedKvDstStack.mPipeline.producer_acquire(smemTransformedKvProdState,
                                                             smemTransformedKvProdToken);
      }
      //
      // smemTransformedKv [ProdWork (call 4), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1477
      smemPtrV4 = smemPtrV3;
      // Task.cpp:1477
      smemIdxV4 = smemIdxV3;
      // Task.cpp:1573
      // Task.cpp:5038
      {
        // Task.cpp:5829
        int32_t index{smemTransformedKvProdState.index()};
        // SmemTransformedKv.h:214
        cutlass::float_e4m3_t* srcPtr;
        // SmemTransformedKv.h:215
        srcPtr = smemPtrV4 + ((smemIdxV4) * (int32_t{16384}));
        // SmemTransformedKv.h:241
        uint64_t srcBuffer4[16];
        // SmemTransformedKv.h:261
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset2564 = int32_t{0}; loopOffset2564 < int32_t{2}; ++loopOffset2564) {
          // SmemTransformedKv.h:284
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset2566 = (loopOffset2564) * (int32_t{1024});
               loopOffset2566 < ((loopOffset2564) + (int32_t{1})) * (int32_t{1024});
               loopOffset2566 += int32_t{128}) {
            // SmemTransformedKv.h:293
            int32_t offset{(loopOffset2566) + (mWarpGrpThreadIdx)};
            // SmemTransformedKv.h:304
            srcBuffer4[(loopOffset2566) / (int32_t{128})] =
              reinterpret_cast<uint64_t*>(srcPtr)[offset];
          }
          // SmemTransformedKv.h:348
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset2570 = (loopOffset2564) * (int32_t{1024});
               loopOffset2570 < ((loopOffset2564) + (int32_t{1})) * (int32_t{1024});
               loopOffset2570 += int32_t{128}) {
            // SmemTransformedKv.h:356
            int32_t offset{(loopOffset2570) + (mWarpGrpThreadIdx)};
            // SmemTransformedKv.h:385
            cutlass::uint128_t dst{
              trtllm::dev::convertE4m3ToFp16(srcBuffer4[(loopOffset2570) / (int32_t{128})])};
            // SmemTransformedKv.h:397
            int32_t eltIdx{(offset) * (int32_t{8})};
            // SmemTile.cpp:389
            int32_t smemRowIdx{(((eltIdx) % (int32_t{128})) / (int32_t{64})) * (int32_t{128}) +
                               ((eltIdx) / (int32_t{128}))};
            // SmemTile.cpp:396
            int32_t smemOffsetInBytes{
              ((smemRowIdx) * (int32_t{128})) +
              ((((eltIdx) % (int32_t{64})) * (int32_t{16})) / (int32_t{8}))};
            // SmemTile.cpp:416
            int32_t swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
            // SmemTransformedKv.h:414
            reinterpret_cast<cutlass::uint128_t*>(
              smemTransformedKvDstSmem
                .mArray[index])[((smemOffsetInBytes) ^ (swizzleMask)) / (int32_t{16})] = dst;
          }
        }
        // SmemTransformedKv.h:423
        cutlass::arch::fence_view_async_shared();
      }
      //
      // smemTransformedKv [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:4409
      {
        // Task.cpp:4427
        {
          // Task.cpp:4443
          smemTransformedKvDstStack.mPipeline.producer_commit(smemTransformedKvProdState);
        }
        // Task.cpp:43
        ++smemTransformedKvProdState;
      }
      //
      // smemKv [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{1024}].
      //
      // Task.cpp:2533
      if ((loopOffset2411) >= (int32_t{0})) {
        // Task.cpp:2453
        trtllm::dev::CutlassNamedBarrier::sync(128, 6);
        // Task.cpp:2561
        {
          // Task.cpp:2585
          smemKvSrcStack.mPipeline.consumer_release(smemKvConsReleaseState);
        }
        // Task.cpp:43
        ++smemKvConsReleaseState;
      }
      //
      // smemKv [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:2781
      {
        // Task.cpp:1573
        // Task.cpp:2722
        {
          // Task.cpp:2745
          smemKvConsToken = smemKvSrcStack.mPipeline.consumer_try_wait(smemKvConsState);
        }
        // Task.cpp:2813
        smemKvSrcStack.mPipeline.consumer_wait(smemKvConsState, smemKvConsToken);
      }
      //
      // smemKv [ConsWork (call 5), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:2893
      {
        // Task.cpp:5829
        int32_t index{smemKvConsState.index()};
        // SmemKv.h:244
        smemPtrV3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:246
        smemIdxV3 = index;
        // Task.cpp:43
        ++smemKvConsState;
      }
      //
      // smemTransformedKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:4948
      {
        // Task.cpp:4962
        if ((loopOffset2411) >= (int32_t{0})) {
          // Task.cpp:4984
          smemTransformedKvProdToken =
            smemTransformedKvDstStack.mPipeline.producer_try_acquire(smemTransformedKvProdState);
        }
      }
      // Task.cpp:1573
      // Task.cpp:4180
      {
        // Task.cpp:4210
        smemTransformedKvDstStack.mPipeline.producer_acquire(smemTransformedKvProdState,
                                                             smemTransformedKvProdToken);
      }
      //
      // smemTransformedKv [ProdWork (call 5), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1477
      smemPtrV4 = smemPtrV3;
      // Task.cpp:1477
      smemIdxV4 = smemIdxV3;
      // Task.cpp:1573
      // Task.cpp:5038
      {
        // Task.cpp:5829
        int32_t index{smemTransformedKvProdState.index()};
        // SmemTransformedKv.h:214
        cutlass::float_e4m3_t* srcPtr;
        // SmemTransformedKv.h:215
        srcPtr = smemPtrV4 + ((smemIdxV4) * (int32_t{16384}));
        // SmemTransformedKv.h:241
        uint64_t srcBuffer4[16];
        // SmemTransformedKv.h:261
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset2626 = int32_t{0}; loopOffset2626 < int32_t{2}; ++loopOffset2626) {
          // SmemTransformedKv.h:284
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset2628 = (loopOffset2626) * (int32_t{1024});
               loopOffset2628 < ((loopOffset2626) + (int32_t{1})) * (int32_t{1024});
               loopOffset2628 += int32_t{128}) {
            // SmemTransformedKv.h:293
            int32_t offset{(loopOffset2628) + (mWarpGrpThreadIdx)};
            // SmemTransformedKv.h:304
            srcBuffer4[(loopOffset2628) / (int32_t{128})] =
              reinterpret_cast<uint64_t*>(srcPtr)[offset];
          }
          // SmemTransformedKv.h:348
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset2632 = (loopOffset2626) * (int32_t{1024});
               loopOffset2632 < ((loopOffset2626) + (int32_t{1})) * (int32_t{1024});
               loopOffset2632 += int32_t{128}) {
            // SmemTransformedKv.h:356
            int32_t offset{(loopOffset2632) + (mWarpGrpThreadIdx)};
            // SmemTransformedKv.h:385
            cutlass::uint128_t dst{
              trtllm::dev::convertE4m3ToFp16(srcBuffer4[(loopOffset2632) / (int32_t{128})])};
            // SmemTransformedKv.h:397
            int32_t eltIdx{(offset) * (int32_t{8})};
            // SmemTile.cpp:389
            int32_t smemRowIdx{(((eltIdx) % (int32_t{128})) / (int32_t{64})) * (int32_t{128}) +
                               ((eltIdx) / (int32_t{128}))};
            // SmemTile.cpp:396
            int32_t smemOffsetInBytes{
              ((smemRowIdx) * (int32_t{128})) +
              ((((eltIdx) % (int32_t{64})) * (int32_t{16})) / (int32_t{8}))};
            // SmemTile.cpp:416
            int32_t swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
            // SmemTransformedKv.h:414
            reinterpret_cast<cutlass::uint128_t*>(
              smemTransformedKvDstSmem
                .mArray[index])[((smemOffsetInBytes) ^ (swizzleMask)) / (int32_t{16})] = dst;
          }
        }
        // SmemTransformedKv.h:423
        cutlass::arch::fence_view_async_shared();
      }
      //
      // smemTransformedKv [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:4409
      {
        // Task.cpp:4427
        {
          // Task.cpp:4443
          smemTransformedKvDstStack.mPipeline.producer_commit(smemTransformedKvProdState);
        }
        // Task.cpp:43
        ++smemTransformedKvProdState;
      }
      //
      // smemKv [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{1024}].
      //
      // Task.cpp:2533
      if ((loopOffset2411) >= (int32_t{0})) {
        // Task.cpp:2453
        trtllm::dev::CutlassNamedBarrier::sync(128, 6);
        // Task.cpp:2561
        {
          // Task.cpp:2585
          smemKvSrcStack.mPipeline.consumer_release(smemKvConsReleaseState);
        }
        // Task.cpp:43
        ++smemKvConsReleaseState;
      }
      // Task.cpp:3457
      lastLoopOffset = loopOffset2411;
    }
    //
    // Pull the last iter down.
    //
    // Task.cpp:3492
    if (((numLoopSteps) - (int32_t{1})) > (int32_t{0})) {
      // Task.cpp:3493
      ++lastLoopOffset;
    }
    //
    // smemKv [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:2745
        smemKvConsToken = smemKvSrcStack.mPipeline.consumer_try_wait(smemKvConsState);
      }
      // Task.cpp:2813
      smemKvSrcStack.mPipeline.consumer_wait(smemKvConsState, smemKvConsToken);
    }
    //
    // smemKv [ConsWork (call 6), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{smemKvConsState.index()};
      // SmemKv.h:244
      smemPtrV3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
      // SmemKv.h:246
      smemIdxV3 = index;
      // Task.cpp:43
      ++smemKvConsState;
    }
    //
    // smemTransformedKv [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4962
      if ((lastLoopOffset) >= (int32_t{0})) {
        // Task.cpp:4984
        smemTransformedKvProdToken =
          smemTransformedKvDstStack.mPipeline.producer_try_acquire(smemTransformedKvProdState);
      }
    }
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4210
      smemTransformedKvDstStack.mPipeline.producer_acquire(smemTransformedKvProdState,
                                                           smemTransformedKvProdToken);
    }
    //
    // smemTransformedKv [ProdWork (call 6), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1477
    smemPtrV4 = smemPtrV3;
    // Task.cpp:1477
    smemIdxV4 = smemIdxV3;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{smemTransformedKvProdState.index()};
      // SmemTransformedKv.h:214
      cutlass::float_e4m3_t* srcPtr;
      // SmemTransformedKv.h:215
      srcPtr = smemPtrV4 + ((smemIdxV4) * (int32_t{16384}));
      // SmemTransformedKv.h:241
      uint64_t srcBuffer4[16];
      // SmemTransformedKv.h:261
      CUTLASS_PRAGMA_UNROLL
      for (int32_t loopOffset2706 = int32_t{0}; loopOffset2706 < int32_t{2}; ++loopOffset2706) {
        // SmemTransformedKv.h:284
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset2708 = (loopOffset2706) * (int32_t{1024});
             loopOffset2708 < ((loopOffset2706) + (int32_t{1})) * (int32_t{1024});
             loopOffset2708 += int32_t{128}) {
          // SmemTransformedKv.h:293
          int32_t offset{(loopOffset2708) + (mWarpGrpThreadIdx)};
          // SmemTransformedKv.h:304
          srcBuffer4[(loopOffset2708) / (int32_t{128})] =
            reinterpret_cast<uint64_t*>(srcPtr)[offset];
        }
        // SmemTransformedKv.h:348
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset2712 = (loopOffset2706) * (int32_t{1024});
             loopOffset2712 < ((loopOffset2706) + (int32_t{1})) * (int32_t{1024});
             loopOffset2712 += int32_t{128}) {
          // SmemTransformedKv.h:356
          int32_t offset{(loopOffset2712) + (mWarpGrpThreadIdx)};
          // SmemTransformedKv.h:385
          cutlass::uint128_t dst{
            trtllm::dev::convertE4m3ToFp16(srcBuffer4[(loopOffset2712) / (int32_t{128})])};
          // SmemTransformedKv.h:397
          int32_t eltIdx{(offset) * (int32_t{8})};
          // SmemTile.cpp:389
          int32_t smemRowIdx{(((eltIdx) % (int32_t{128})) / (int32_t{64})) * (int32_t{128}) +
                             ((eltIdx) / (int32_t{128}))};
          // SmemTile.cpp:396
          int32_t smemOffsetInBytes{((smemRowIdx) * (int32_t{128})) +
                                    ((((eltIdx) % (int32_t{64})) * (int32_t{16})) / (int32_t{8}))};
          // SmemTile.cpp:416
          int32_t swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
          // SmemTransformedKv.h:414
          reinterpret_cast<cutlass::uint128_t*>(
            smemTransformedKvDstSmem
              .mArray[index])[((smemOffsetInBytes) ^ (swizzleMask)) / (int32_t{16})] = dst;
        }
      }
      // SmemTransformedKv.h:423
      cutlass::arch::fence_view_async_shared();
    }
    //
    // smemTransformedKv [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4427
      {
        // Task.cpp:4443
        smemTransformedKvDstStack.mPipeline.producer_commit(smemTransformedKvProdState);
      }
      // Task.cpp:43
      ++smemTransformedKvProdState;
    }
    //
    // smemKv [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{1024}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:2453
      trtllm::dev::CutlassNamedBarrier::sync(128, 6);
      // Task.cpp:2561
      {
        // Task.cpp:2585
        smemKvSrcStack.mPipeline.consumer_release(smemKvConsReleaseState);
      }
      // Task.cpp:43
      ++smemKvConsReleaseState;
    }
    //
    // smemKv [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:2745
        smemKvConsToken = smemKvSrcStack.mPipeline.consumer_try_wait(smemKvConsState);
      }
      // Task.cpp:2813
      smemKvSrcStack.mPipeline.consumer_wait(smemKvConsState, smemKvConsToken);
    }
    //
    // smemKv [ConsWork (call 7), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{smemKvConsState.index()};
      // SmemKv.h:244
      smemPtrV3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
      // SmemKv.h:246
      smemIdxV3 = index;
      // Task.cpp:43
      ++smemKvConsState;
    }
    //
    // smemTransformedKv [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4962
      if ((lastLoopOffset) >= (int32_t{0})) {
        // Task.cpp:4984
        smemTransformedKvProdToken =
          smemTransformedKvDstStack.mPipeline.producer_try_acquire(smemTransformedKvProdState);
      }
    }
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4210
      smemTransformedKvDstStack.mPipeline.producer_acquire(smemTransformedKvProdState,
                                                           smemTransformedKvProdToken);
    }
    //
    // smemTransformedKv [ProdWork (call 7), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1477
    smemPtrV4 = smemPtrV3;
    // Task.cpp:1477
    smemIdxV4 = smemIdxV3;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{smemTransformedKvProdState.index()};
      // SmemTransformedKv.h:214
      cutlass::float_e4m3_t* srcPtr;
      // SmemTransformedKv.h:215
      srcPtr = smemPtrV4 + ((smemIdxV4) * (int32_t{16384}));
      // SmemTransformedKv.h:241
      uint64_t srcBuffer4[16];
      // SmemTransformedKv.h:261
      CUTLASS_PRAGMA_UNROLL
      for (int32_t loopOffset2780 = int32_t{0}; loopOffset2780 < int32_t{2}; ++loopOffset2780) {
        // SmemTransformedKv.h:284
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset2782 = (loopOffset2780) * (int32_t{1024});
             loopOffset2782 < ((loopOffset2780) + (int32_t{1})) * (int32_t{1024});
             loopOffset2782 += int32_t{128}) {
          // SmemTransformedKv.h:293
          int32_t offset{(loopOffset2782) + (mWarpGrpThreadIdx)};
          // SmemTransformedKv.h:304
          srcBuffer4[(loopOffset2782) / (int32_t{128})] =
            reinterpret_cast<uint64_t*>(srcPtr)[offset];
        }
        // SmemTransformedKv.h:348
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset2786 = (loopOffset2780) * (int32_t{1024});
             loopOffset2786 < ((loopOffset2780) + (int32_t{1})) * (int32_t{1024});
             loopOffset2786 += int32_t{128}) {
          // SmemTransformedKv.h:356
          int32_t offset{(loopOffset2786) + (mWarpGrpThreadIdx)};
          // SmemTransformedKv.h:385
          cutlass::uint128_t dst{
            trtllm::dev::convertE4m3ToFp16(srcBuffer4[(loopOffset2786) / (int32_t{128})])};
          // SmemTransformedKv.h:397
          int32_t eltIdx{(offset) * (int32_t{8})};
          // SmemTile.cpp:389
          int32_t smemRowIdx{(((eltIdx) % (int32_t{128})) / (int32_t{64})) * (int32_t{128}) +
                             ((eltIdx) / (int32_t{128}))};
          // SmemTile.cpp:396
          int32_t smemOffsetInBytes{((smemRowIdx) * (int32_t{128})) +
                                    ((((eltIdx) % (int32_t{64})) * (int32_t{16})) / (int32_t{8}))};
          // SmemTile.cpp:416
          int32_t swizzleMask{((smemRowIdx) % (int32_t{8})) * (int32_t{16})};
          // SmemTransformedKv.h:414
          reinterpret_cast<cutlass::uint128_t*>(
            smemTransformedKvDstSmem
              .mArray[index])[((smemOffsetInBytes) ^ (swizzleMask)) / (int32_t{16})] = dst;
        }
      }
      // SmemTransformedKv.h:423
      cutlass::arch::fence_view_async_shared();
    }
    //
    // smemTransformedKv [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4427
      {
        // Task.cpp:4443
        smemTransformedKvDstStack.mPipeline.producer_commit(smemTransformedKvProdState);
      }
      // Task.cpp:43
      ++smemTransformedKvProdState;
    }
    //
    // smemKv [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{1024}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:2453
      trtllm::dev::CutlassNamedBarrier::sync(128, 6);
      // Task.cpp:2561
      {
        // Task.cpp:2585
        smemKvSrcStack.mPipeline.consumer_release(smemKvConsReleaseState);
      }
      // Task.cpp:43
      ++smemKvConsReleaseState;
    }
  //
  // Tail work.
  //
  // Task.cpp:3511
  ExitTileWithSignalingLabel:
  // Task.cpp:3518
  ExitTileWithoutSignalingLabel:
    // Task.cpp:3528
    {}
  }
};
// Task.cpp:544
// Fmha.h:2493
struct MmaTask {
  // Task.cpp:283
  int32_t mCtaIdxX;
  // Task.cpp:287
  int32_t mCtaIdxY;
  // Task.cpp:291
  int32_t mCtaIdxZ;
  // FmhaTask.h:220
  int32_t mHeadIdx;
  // FmhaTask.h:218
  int32_t mBatchIdx;
  // FmhaTask.h:208
  int32_t mSeqOffsetQ;
  // FmhaTask.h:210
  int32_t mSeqLenQ;
  // FmhaTask.h:224
  int32_t mCtaIdxQ;
  // FmhaTask.h:226
  int32_t mCtaIdxKv;
  // FmhaTask.h:214
  int32_t mSeqLenKv;
  // FmhaTask.h:216
  int32_t mNumCtasKv;
  // Task.cpp:691
  uint32_t const mTmemBaseOffset;
  // Task.cpp:551
  inline __device__ MmaTask(fmha::KernelParams const& params,
                            KernelState const& state,
                            int32_t warpGrpStart)
    : // Kernel.cpp:194
    mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , // Kernel.cpp:195
    mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , // Kernel.cpp:196
    mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)}
    , // Task.cpp:287
    mHeadIdx{mCtaIdxY}
    , // Task.cpp:291
    mBatchIdx{mCtaIdxZ}
    , // FmhaTask.h:394
    mSeqOffsetQ{int32_t(bool{params.ptrCumSeqLensQ == nullptr})
                  ? ((mBatchIdx) * (params.mMaxSeqLenQ))
                  : (int32_t{params.ptrCumSeqLensQ[mBatchIdx]})}
    , // FmhaTask.h:410
    mSeqLenQ{int32_t(bool{params.ptrCumSeqLensQ == nullptr})
               ? (params.mMaxSeqLenQ)
               : ((int32_t{params.ptrCumSeqLensQ[(mBatchIdx) + (int32_t{1})]}) - (mSeqOffsetQ))}
    , // FmhaTask.h:516
    mCtaIdxQ{(mCtaIdxX) / (params.mMaxNumCtasKv)}
    , // FmhaTask.h:517
    mCtaIdxKv{(mCtaIdxX) % (params.mMaxNumCtasKv)}
    , // FmhaTask.h:543
    mSeqLenKv{(int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                             ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                             : (mBatchIdx)]}) -
              (((mSeqLenQ) - (int32_t{1})) - (mCtaIdxQ))}
    , // FmhaTask.h:565
    mNumCtasKv{
      int32_t{min(int32_t{((mSeqLenKv) + (int32_t{127})) / (int32_t{128})}, params.mMaxNumCtasKv)}}
    , // Kernel.cpp:2420
    mTmemBaseOffset{uint32_t{
      __shfl_sync(uint32_t{0xffffffff}, (*state.mTmemSwStatePtr), int32_t{0}, int32_t{32})}} {}
  // Task.cpp:507
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:532
    return ((state.mWarpIdx) >= (int32_t{8})) && ((state.mWarpIdx) < (int32_t{9}));
  }
  // Task.cpp:454
  inline __device__ void execute(fmha::KernelParams const& params,
                                 KernelState const& state,
                                 TmemS0Stack& tmemS0DstStack,
                                 TmemOStack& tmemODstStack,
                                 SmemQSmem& smemQSrcSmem,
                                 SmemQStack& smemQSrcStack,
                                 SmemTransformedKvSmem& smemTransformedKvSrcSmem,
                                 SmemTransformedKvStack& smemTransformedKvSrcStack,
                                 TmemP0Stack& tmemP0SrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<56>{});
    // Task.cpp:2079
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      1,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemQConsState{};
    // Task.cpp:2086
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      1,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemQConsReleaseState{};
    // Task.cpp:2100
    int32_t smemQConsToken{int32_t{0}};
    // Task.cpp:2079
    trtllm::dev::CutlassUmmaConsumerAsyncPipeline<2, false, false>::PipelineState
      smemTransformedKvConsState{};
    // Task.cpp:2086
    trtllm::dev::CutlassUmmaConsumerAsyncPipeline<2, false, false>::PipelineState
      smemTransformedKvConsReleaseState{};
    // Task.cpp:2100
    int32_t smemTransformedKvConsToken{int32_t{0}};
    // Task.cpp:2079
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState tmemP0ConsState{};
    // Task.cpp:2086
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState tmemP0ConsReleaseState{};
    // Task.cpp:2100
    int32_t tmemP0ConsToken{int32_t{0}};
    // Task.cpp:1979
    trtllm::dev::CutlassUmmaAsyncPipeline<2,
                                          cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState tmemS0ProdState{int32_t{0}, int32_t{1}, int32_t{0}};
    // Task.cpp:1999
    int32_t tmemS0ProdToken{int32_t{1}};
    // Task.cpp:1979
    trtllm::dev::CutlassUmmaAsyncPipeline<1,
                                          cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState tmemOProdState{int32_t{0}, int32_t{1}, int32_t{0}};
    // Task.cpp:1999
    int32_t tmemOProdToken{int32_t{1}};
    // FmhaTask.h:582
    int32_t numLoopSteps;
    // FmhaTask.h:630
    if (((mCtaIdxQ) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
      // FmhaTask.h:668
      int32_t numSteps{((mSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
                       ((mNumCtasKv) * (int32_t{128}))};
      // FmhaTask.h:682
      numLoopSteps = numSteps;
    } else {
      // FmhaTask.h:651
      return;
    }
    // Task.cpp:3168
    bool const hasOneLoopIter{(int32_t{0}) < (numLoopSteps)};
    // Task.cpp:3179
    int32_t lastLoopOffset{int32_t{0}};
    //
    // Hoist the first iter.
    //
    //
    // smemQ [ConsWait, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:2745
        smemQConsToken = smemQSrcStack.mPipeline.consumer_try_wait(smemQConsState);
      }
      // Task.cpp:2813
      smemQSrcStack.mPipeline.consumer_wait(smemQConsState, smemQConsToken);
    }
    //
    // smemQ [ConsWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    // SmemQ.h:151
    cutlass::half_t* smemPtrQ0_2{&smemQSrcSmem.mArray[int32_t{0}][int32_t{0}]};
    // SmemQ.h:159
    int32_t smemIdxQ0_2;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{smemQConsState.index()};
      // SmemQ.h:188
      smemIdxQ0_2 = index;
      // Task.cpp:43
      ++smemQConsState;
    }
    //
    // tmemS0 [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4962
      {
        // Task.cpp:4984
        tmemS0ProdToken = tmemS0DstStack.mPipeline.producer_try_acquire(tmemS0ProdState);
      }
    }
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4210
      tmemS0DstStack.mPipeline.producer_acquire(tmemS0ProdState, tmemS0ProdToken);
    }
    //
    // smemTransformedKv [ConsWait, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:2745
        smemTransformedKvConsToken =
          smemTransformedKvSrcStack.mPipeline.consumer_try_wait(smemTransformedKvConsState);
      }
      // Task.cpp:2813
      smemTransformedKvSrcStack.mPipeline.consumer_wait(smemTransformedKvConsState,
                                                        smemTransformedKvConsToken);
    }
    //
    // smemTransformedKv [ConsWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // SmemKv.h:199
    cutlass::half_t* smemPtrK4;
    // SmemKv.h:206
    int32_t smemIdxK4;
    // SmemKv.h:214
    cutlass::half_t* smemPtrV4;
    // SmemKv.h:221
    int32_t smemIdxV4;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{smemTransformedKvConsState.index()};
      // SmemKv.h:267
      smemPtrK4 = &smemTransformedKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
      // SmemKv.h:304
      smemIdxK4 = index;
      // Task.cpp:43
      ++smemTransformedKvConsState;
    }
    //
    // tmemS0 [ProdWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    // TmemS.h:1123
    cutlass::half_t* smemPtrQ10;
    // TmemS.h:1128
    int32_t smemIdxQ10;
    // TmemS.h:1134
    cutlass::half_t* smemPtrK10;
    // TmemS.h:1139
    int32_t memIdxK10;
    // Task.cpp:1477
    smemPtrQ10 = smemPtrQ0_2;
    // Task.cpp:1477
    smemIdxQ10 = smemIdxQ0_2;
    // Task.cpp:1477
    smemPtrK10 = smemPtrK4;
    // Task.cpp:1477
    memIdxK10 = smemIdxK4;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{tmemS0ProdState.index()};
      // TmemS.h:1874
      cutlass::half_t* smemQ{smemPtrQ10};
      // TmemS.h:1895
      smemQ += (smemIdxQ10) * (int32_t{4096});
      // TmemS.h:1923
      cutlass::half_t* smemK{smemPtrK10};
      // TmemS.h:1929
      smemK += (memIdxK10) * (int32_t{16384});
      // Mma.cpp:618
      {
        // TmemTile.cpp:1755
        uint32_t tmemPtrD{
          (index) * (int32_t{16}) +
          (int32_t((tmemS0DstStack.mInstId) == (int32_t{0})) ? (int32_t{0}) : (int32_t{16}))};
        //
        // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
        //
        // Mma.cpp:203
        uint64_t smemDescA{
          trtllm::dev::createSmemDesc(smemK,
                                      uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                      uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
        //
        // leadingDimInBytes = 2048, strideInBytes = 1024, swizzleMode = 1.
        //
        // Mma.cpp:203
        uint64_t smemDescB{
          trtllm::dev::createSmemDesc(smemQ,
                                      uint32_t{0x800000 /*hi=128, lo=0*/},
                                      uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
        //
        // MMA inst for mi=0 ni=0 ki=0.
        //
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                false,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                false,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                false,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                false,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{1018});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{122});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                false,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                false,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                false,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                false,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
    // smemTransformedKv [ConsRelease, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:2561
      {
        // Task.cpp:2585
        smemTransformedKvSrcStack.mPipeline.consumer_release(smemTransformedKvConsReleaseState);
      }
      // Task.cpp:43
      ++smemTransformedKvConsReleaseState;
    }
    //
    // smemTransformedKv [ConsWait, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:2745
        smemTransformedKvConsToken =
          smemTransformedKvSrcStack.mPipeline.consumer_try_wait(smemTransformedKvConsState);
      }
      // Task.cpp:2813
      smemTransformedKvSrcStack.mPipeline.consumer_wait(smemTransformedKvConsState,
                                                        smemTransformedKvConsToken);
    }
    //
    // smemTransformedKv [ConsWork (call 1), FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{smemTransformedKvConsState.index()};
      // SmemKv.h:267
      smemPtrK4 = &smemTransformedKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
      // SmemKv.h:304
      smemIdxK4 = index;
      // Task.cpp:43
      ++smemTransformedKvConsState;
    }
    //
    // tmemS0 [ProdWork (call 1), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    // Task.cpp:1477
    smemPtrQ10 = smemPtrQ0_2;
    // Task.cpp:1477
    smemIdxQ10 = smemIdxQ0_2;
    // Task.cpp:1477
    smemPtrK10 = smemPtrK4;
    // Task.cpp:1477
    memIdxK10 = smemIdxK4;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{tmemS0ProdState.index()};
      // TmemS.h:1874
      cutlass::half_t* smemQ{smemPtrQ10};
      // TmemS.h:1895
      smemQ += (smemIdxQ10) * (int32_t{4096}) + (int32_t{2048});
      // TmemS.h:1923
      cutlass::half_t* smemK{smemPtrK10};
      // TmemS.h:1929
      smemK += (memIdxK10) * (int32_t{16384});
      // Mma.cpp:618
      {
        // TmemTile.cpp:1755
        uint32_t tmemPtrD{
          (index) * (int32_t{16}) +
          (int32_t((tmemS0DstStack.mInstId) == (int32_t{0})) ? (int32_t{0}) : (int32_t{16}))};
        //
        // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
        //
        // Mma.cpp:203
        uint64_t smemDescA{
          trtllm::dev::createSmemDesc(smemK,
                                      uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                      uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
        //
        // leadingDimInBytes = 2048, strideInBytes = 1024, swizzleMode = 1.
        //
        // Mma.cpp:203
        uint64_t smemDescB{
          trtllm::dev::createSmemDesc(smemQ,
                                      uint32_t{0x800000 /*hi=128, lo=0*/},
                                      uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
        //
        // MMA inst for mi=0 ni=0 ki=0.
        //
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                false,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                cuda_ptx::cta_group_1,
                                tmemPtrD,
                                smemDescA,
                                smemDescB,
                                utcmmaDesc_0_0_0,
                                bool{true});
        }
        //
        // MMA inst for mi=0 ni=0 ki=1.
        //
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                false,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                false,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                false,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{1018});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{122});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                false,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                false,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                false,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                false,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
    // smemTransformedKv [ConsRelease, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:2561
      {
        // Task.cpp:2585
        smemTransformedKvSrcStack.mPipeline.consumer_release(smemTransformedKvConsReleaseState);
      }
      // Task.cpp:43
      ++smemTransformedKvConsReleaseState;
    }
    //
    // tmemS0 [ProdCommit, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4427
      {
        // Task.cpp:4443
        tmemS0DstStack.mPipeline.producer_commit(tmemS0ProdState);
      }
      // Task.cpp:43
      ++tmemS0ProdState;
    }
    //
    // tmemS0 [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4962
      {
        // Task.cpp:4984
        tmemS0ProdToken = tmemS0DstStack.mPipeline.producer_try_acquire(tmemS0ProdState);
      }
    }
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4210
      tmemS0DstStack.mPipeline.producer_acquire(tmemS0ProdState, tmemS0ProdToken);
    }
    //
    // Loop body.
    //
    // Task.cpp:3350
    CUTLASS_PRAGMA_NO_UNROLL
    for (int32_t loopOffset3137 = int32_t{0}; loopOffset3137 < (numLoopSteps) - (int32_t{1});
         ++loopOffset3137) {
      //
      // smemTransformedKv [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:2781
      {
        // Task.cpp:1573
        // Task.cpp:2722
        {
          // Task.cpp:2745
          smemTransformedKvConsToken =
            smemTransformedKvSrcStack.mPipeline.consumer_try_wait(smemTransformedKvConsState);
        }
        // Task.cpp:2813
        smemTransformedKvSrcStack.mPipeline.consumer_wait(smemTransformedKvConsState,
                                                          smemTransformedKvConsToken);
      }
      //
      // smemTransformedKv [ConsWork (call 2), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:2893
      {
        // Task.cpp:5829
        int32_t index{smemTransformedKvConsState.index()};
        // SmemKv.h:267
        smemPtrK4 = &smemTransformedKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:304
        smemIdxK4 = index;
        // Task.cpp:43
        ++smemTransformedKvConsState;
      }
      //
      // tmemS0 [ProdWork (call 2), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // Task.cpp:1477
      smemPtrQ10 = smemPtrQ0_2;
      // Task.cpp:1477
      smemIdxQ10 = smemIdxQ0_2;
      // Task.cpp:1477
      smemPtrK10 = smemPtrK4;
      // Task.cpp:1477
      memIdxK10 = smemIdxK4;
      // Task.cpp:1573
      // Task.cpp:5038
      {
        // Task.cpp:5829
        int32_t index{tmemS0ProdState.index()};
        // TmemS.h:1874
        cutlass::half_t* smemQ{smemPtrQ10};
        // TmemS.h:1895
        smemQ += (smemIdxQ10) * (int32_t{4096});
        // TmemS.h:1923
        cutlass::half_t* smemK{smemPtrK10};
        // TmemS.h:1929
        smemK += (memIdxK10) * (int32_t{16384});
        // Mma.cpp:618
        {
          // TmemTile.cpp:1755
          uint32_t tmemPtrD{
            (index) * (int32_t{16}) +
            (int32_t((tmemS0DstStack.mInstId) == (int32_t{0})) ? (int32_t{0}) : (int32_t{16}))};
          //
          // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
          //
          // Mma.cpp:203
          uint64_t smemDescA{
            trtllm::dev::createSmemDesc(smemK,
                                        uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                        uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
          //
          // leadingDimInBytes = 2048, strideInBytes = 1024, swizzleMode = 1.
          //
          // Mma.cpp:203
          uint64_t smemDescB{
            trtllm::dev::createSmemDesc(smemQ,
                                        uint32_t{0x800000 /*hi=128, lo=0*/},
                                        uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
          //
          // MMA inst for mi=0 ni=0 ki=0.
          //
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{1018});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{122});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
      // smemTransformedKv [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:2533
      if ((loopOffset3137) >= (int32_t{0})) {
        // Task.cpp:2561
        {
          // Task.cpp:2585
          smemTransformedKvSrcStack.mPipeline.consumer_release(smemTransformedKvConsReleaseState);
        }
        // Task.cpp:43
        ++smemTransformedKvConsReleaseState;
      }
      //
      // smemTransformedKv [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:2781
      {
        // Task.cpp:1573
        // Task.cpp:2722
        {
          // Task.cpp:2745
          smemTransformedKvConsToken =
            smemTransformedKvSrcStack.mPipeline.consumer_try_wait(smemTransformedKvConsState);
        }
        // Task.cpp:2813
        smemTransformedKvSrcStack.mPipeline.consumer_wait(smemTransformedKvConsState,
                                                          smemTransformedKvConsToken);
      }
      //
      // smemTransformedKv [ConsWork (call 3), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:2893
      {
        // Task.cpp:5829
        int32_t index{smemTransformedKvConsState.index()};
        // SmemKv.h:267
        smemPtrK4 = &smemTransformedKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:304
        smemIdxK4 = index;
        // Task.cpp:43
        ++smemTransformedKvConsState;
      }
      //
      // tmemS0 [ProdWork (call 3), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // Task.cpp:1477
      smemPtrQ10 = smemPtrQ0_2;
      // Task.cpp:1477
      smemIdxQ10 = smemIdxQ0_2;
      // Task.cpp:1477
      smemPtrK10 = smemPtrK4;
      // Task.cpp:1477
      memIdxK10 = smemIdxK4;
      // Task.cpp:1573
      // Task.cpp:5038
      {
        // Task.cpp:5829
        int32_t index{tmemS0ProdState.index()};
        // TmemS.h:1874
        cutlass::half_t* smemQ{smemPtrQ10};
        // TmemS.h:1895
        smemQ += (smemIdxQ10) * (int32_t{4096}) + (int32_t{2048});
        // TmemS.h:1923
        cutlass::half_t* smemK{smemPtrK10};
        // TmemS.h:1929
        smemK += (memIdxK10) * (int32_t{16384});
        // Mma.cpp:618
        {
          // TmemTile.cpp:1755
          uint32_t tmemPtrD{
            (index) * (int32_t{16}) +
            (int32_t((tmemS0DstStack.mInstId) == (int32_t{0})) ? (int32_t{0}) : (int32_t{16}))};
          //
          // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
          //
          // Mma.cpp:203
          uint64_t smemDescA{
            trtllm::dev::createSmemDesc(smemK,
                                        uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                        uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
          //
          // leadingDimInBytes = 2048, strideInBytes = 1024, swizzleMode = 1.
          //
          // Mma.cpp:203
          uint64_t smemDescB{
            trtllm::dev::createSmemDesc(smemQ,
                                        uint32_t{0x800000 /*hi=128, lo=0*/},
                                        uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
          //
          // MMA inst for mi=0 ni=0 ki=0.
          //
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                  cuda_ptx::cta_group_1,
                                  tmemPtrD,
                                  smemDescA,
                                  smemDescB,
                                  utcmmaDesc_0_0_0,
                                  bool{true});
          }
          //
          // MMA inst for mi=0 ni=0 ki=1.
          //
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{1018});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{122});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
      // smemTransformedKv [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:2533
      if ((loopOffset3137) >= (int32_t{0})) {
        // Task.cpp:2561
        {
          // Task.cpp:2585
          smemTransformedKvSrcStack.mPipeline.consumer_release(smemTransformedKvConsReleaseState);
        }
        // Task.cpp:43
        ++smemTransformedKvConsReleaseState;
      }
      //
      // tmemS0 [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // Task.cpp:4409
      {
        // Task.cpp:4427
        {
          // Task.cpp:4443
          tmemS0DstStack.mPipeline.producer_commit(tmemS0ProdState);
        }
        // Task.cpp:43
        ++tmemS0ProdState;
      }
      //
      // tmemS0 [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:4948
      {
        // Task.cpp:4962
        if ((loopOffset3137) >= (int32_t{0})) {
          // Task.cpp:4984
          tmemS0ProdToken = tmemS0DstStack.mPipeline.producer_try_acquire(tmemS0ProdState);
        }
      }
      // Task.cpp:1573
      // Task.cpp:4180
      {
        // Task.cpp:4210
        tmemS0DstStack.mPipeline.producer_acquire(tmemS0ProdState, tmemS0ProdToken);
      }
      //
      // tmemP0 [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // TmemP.h:474
      int32_t stageIdxP14;
      // Task.cpp:1573
      // Task.cpp:2893
      {
        // Task.cpp:5829
        int32_t index{tmemP0ConsState.index()};
        // TmemP.h:488
        stageIdxP14 = index;
        // Task.cpp:43
        ++tmemP0ConsState;
      }
      //
      // tmemO [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{3}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:4948
      {
        // Task.cpp:4962
        if ((loopOffset3137) >= (int32_t{0})) {
          // Task.cpp:4984
          tmemOProdToken = tmemODstStack.mPipeline.producer_try_acquire(tmemOProdState);
        }
      }
      // Task.cpp:1573
      // Task.cpp:4180
      {
        // Task.cpp:4210
        tmemODstStack.mPipeline.producer_acquire(tmemOProdState, tmemOProdToken);
      }
      //
      // smemTransformedKv [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:2781
      {
        // Task.cpp:1573
        // Task.cpp:2722
        {
          // Task.cpp:2745
          smemTransformedKvConsToken =
            smemTransformedKvSrcStack.mPipeline.consumer_try_wait(smemTransformedKvConsState);
        }
        // Task.cpp:2813
        smemTransformedKvSrcStack.mPipeline.consumer_wait(smemTransformedKvConsState,
                                                          smemTransformedKvConsToken);
      }
      //
      // smemTransformedKv [ConsWork (call 4), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:2893
      {
        // Task.cpp:5829
        int32_t index{smemTransformedKvConsState.index()};
        // SmemKv.h:322
        smemPtrV4 = &smemTransformedKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:372
        smemIdxV4 = index;
        // Task.cpp:43
        ++smemTransformedKvConsState;
      }
      //
      // tmemO [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{3}, Flags{0}].
      //
      // TmemO.h:277
      cutlass::half_t* smemPtrV15;
      // TmemO.h:282
      int32_t memIdxV15;
      // TmemO.h:288
      int32_t smemIdxP15;
      // Task.cpp:1477
      smemPtrV15 = smemPtrV4;
      // Task.cpp:1477
      memIdxV15 = smemIdxV4;
      // Task.cpp:1477
      smemIdxP15 = stageIdxP14;
      // Task.cpp:1573
      // Task.cpp:5038
      {
        // Task.cpp:5829
        int32_t index{tmemOProdState.index()};
        // TmemO.h:367
        cutlass::half_t* smemP{reinterpret_cast<cutlass::half_t*>(tmemODstStack.mDepSmemPtr6)};
        // TmemO.h:381
        smemP += (smemIdxP15) * (int32_t{2048});
        // TmemO.h:493
        cutlass::half_t* smemV{smemPtrV15};
        // TmemO.h:505
        smemV = smemV + ((memIdxV15) * (int32_t{16384}));
        // TmemO.h:535
        bool readD{true};
        // TmemO.h:545
        if ((loopOffset3137) == (int32_t{0})) {
          // TmemO.h:547
          readD = false;
        }
        // Mma.cpp:618
        {
          // TmemTile.cpp:1755
          uint32_t tmemPtrD{(mTmemBaseOffset) + (uint32_t{96})};
          //
          // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
          //
          // Mma.cpp:203
          uint64_t smemDescA{
            trtllm::dev::createSmemDesc(smemV,
                                        uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                        uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
          //
          // leadingDimInBytes = 2048, strideInBytes = 1024, swizzleMode = 1.
          //
          // Mma.cpp:203
          uint64_t smemDescB{
            trtllm::dev::createSmemDesc(smemP,
                                        uint32_t{0x800000 /*hi=128, lo=0*/},
                                        uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
          //
          // MMA inst for mi=0 ni=0 ki=0.
          //
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                  cuda_ptx::cta_group_1,
                                  tmemPtrD,
                                  smemDescA,
                                  smemDescB,
                                  utcmmaDesc_0_0_0,
                                  bool{readD});
          }
          //
          // MMA inst for mi=0 ni=0 ki=1.
          //
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{122});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
      // smemTransformedKv [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:2533
      if ((loopOffset3137) >= (int32_t{0})) {
        // Task.cpp:2561
        {
          // Task.cpp:2585
          smemTransformedKvSrcStack.mPipeline.consumer_release(smemTransformedKvConsReleaseState);
        }
        // Task.cpp:43
        ++smemTransformedKvConsReleaseState;
      }
      //
      // smemTransformedKv [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:2781
      {
        // Task.cpp:1573
        // Task.cpp:2722
        {
          // Task.cpp:2745
          smemTransformedKvConsToken =
            smemTransformedKvSrcStack.mPipeline.consumer_try_wait(smemTransformedKvConsState);
        }
        // Task.cpp:2813
        smemTransformedKvSrcStack.mPipeline.consumer_wait(smemTransformedKvConsState,
                                                          smemTransformedKvConsToken);
      }
      //
      // smemTransformedKv [ConsWork (call 5), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1573
      // Task.cpp:2893
      {
        // Task.cpp:5829
        int32_t index{smemTransformedKvConsState.index()};
        // SmemKv.h:322
        smemPtrV4 = &smemTransformedKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:372
        smemIdxV4 = index;
        // Task.cpp:43
        ++smemTransformedKvConsState;
      }
      //
      // tmemO [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{3}, Flags{0}].
      //
      // Task.cpp:1477
      smemPtrV15 = smemPtrV4;
      // Task.cpp:1477
      memIdxV15 = smemIdxV4;
      // Task.cpp:1477
      smemIdxP15 = stageIdxP14;
      // Task.cpp:1573
      // Task.cpp:5038
      {
        // Task.cpp:5829
        int32_t index{tmemOProdState.index()};
        // TmemO.h:367
        cutlass::half_t* smemP{reinterpret_cast<cutlass::half_t*>(tmemODstStack.mDepSmemPtr6)};
        // TmemO.h:381
        smemP += (smemIdxP15) * (int32_t{2048});
        // TmemO.h:493
        cutlass::half_t* smemV{smemPtrV15};
        // TmemO.h:505
        smemV = smemV + ((memIdxV15) * (int32_t{16384}));
        // TmemO.h:535
        bool readD{true};
        // TmemO.h:545
        if ((loopOffset3137) == (int32_t{0})) {
          // TmemO.h:547
          readD = false;
        }
        // Mma.cpp:618
        {
          // TmemTile.cpp:1755
          uint32_t tmemPtrD{(mTmemBaseOffset) + (uint32_t{112})};
          //
          // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
          //
          // Mma.cpp:203
          uint64_t smemDescA{
            trtllm::dev::createSmemDesc(smemV,
                                        uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                        uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
          //
          // leadingDimInBytes = 2048, strideInBytes = 1024, swizzleMode = 1.
          //
          // Mma.cpp:203
          uint64_t smemDescB{
            trtllm::dev::createSmemDesc(smemP,
                                        uint32_t{0x800000 /*hi=128, lo=0*/},
                                        uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
          //
          // MMA inst for mi=0 ni=0 ki=0.
          //
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                  cuda_ptx::cta_group_1,
                                  tmemPtrD,
                                  smemDescA,
                                  smemDescB,
                                  utcmmaDesc_0_0_0,
                                  bool{readD});
          }
          //
          // MMA inst for mi=0 ni=0 ki=1.
          //
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{122});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1700
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1708
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
      // smemTransformedKv [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:2533
      if ((loopOffset3137) >= (int32_t{0})) {
        // Task.cpp:2561
        {
          // Task.cpp:2585
          smemTransformedKvSrcStack.mPipeline.consumer_release(smemTransformedKvConsReleaseState);
        }
        // Task.cpp:43
        ++smemTransformedKvConsReleaseState;
      }
      //
      // tmemO [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{3}, Flags{0}].
      //
      // Task.cpp:4409
      {
        // Task.cpp:4427
        {
          // Task.cpp:4443
          tmemODstStack.mPipeline.producer_commit(tmemOProdState);
        }
        // Task.cpp:43
        ++tmemOProdState;
      }
      // Task.cpp:3457
      lastLoopOffset = loopOffset3137;
    }
    //
    // Pull the last iter down.
    //
    // Task.cpp:3492
    if (((numLoopSteps) - (int32_t{1})) > (int32_t{0})) {
      // Task.cpp:3493
      ++lastLoopOffset;
    }
    //
    // smemQ [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:2561
      {
        // Task.cpp:2585
        smemQSrcStack.mPipeline.consumer_release(smemQConsReleaseState);
      }
      // Task.cpp:43
      ++smemQConsReleaseState;
    }
    //
    // tmemS0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4427
      {
        // Task.cpp:4443
        tmemS0DstStack.mPipeline.producer_commit(tmemS0ProdState);
      }
      // Task.cpp:43
      ++tmemS0ProdState;
    }
    //
    // tmemS0 [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4962
      if ((lastLoopOffset) >= (int32_t{0})) {
        // Task.cpp:4984
        tmemS0ProdToken = tmemS0DstStack.mPipeline.producer_try_acquire(tmemS0ProdState);
      }
    }
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4210
      tmemS0DstStack.mPipeline.producer_acquire(tmemS0ProdState, tmemS0ProdToken);
    }
    //
    // tmemP0 [ConsWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // TmemP.h:474
    int32_t stageIdxP14;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{tmemP0ConsState.index()};
      // TmemP.h:488
      stageIdxP14 = index;
      // Task.cpp:43
      ++tmemP0ConsState;
    }
    //
    // tmemO [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{3}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4962
      if ((lastLoopOffset) >= (int32_t{0})) {
        // Task.cpp:4984
        tmemOProdToken = tmemODstStack.mPipeline.producer_try_acquire(tmemOProdState);
      }
    }
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4210
      tmemODstStack.mPipeline.producer_acquire(tmemOProdState, tmemOProdToken);
    }
    //
    // smemTransformedKv [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:2745
        smemTransformedKvConsToken =
          smemTransformedKvSrcStack.mPipeline.consumer_try_wait(smemTransformedKvConsState);
      }
      // Task.cpp:2813
      smemTransformedKvSrcStack.mPipeline.consumer_wait(smemTransformedKvConsState,
                                                        smemTransformedKvConsToken);
    }
    //
    // smemTransformedKv [ConsWork (call 6), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{smemTransformedKvConsState.index()};
      // SmemKv.h:322
      smemPtrV4 = &smemTransformedKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
      // SmemKv.h:372
      smemIdxV4 = index;
      // Task.cpp:43
      ++smemTransformedKvConsState;
    }
    //
    // tmemO [ProdWork (call 2), LastIter, FreqInfo{0, 1}, UserTags{3}, Flags{0}].
    //
    // TmemO.h:277
    cutlass::half_t* smemPtrV15;
    // TmemO.h:282
    int32_t memIdxV15;
    // TmemO.h:288
    int32_t smemIdxP15;
    // Task.cpp:1477
    smemPtrV15 = smemPtrV4;
    // Task.cpp:1477
    memIdxV15 = smemIdxV4;
    // Task.cpp:1477
    smemIdxP15 = stageIdxP14;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{tmemOProdState.index()};
      // TmemO.h:367
      cutlass::half_t* smemP{reinterpret_cast<cutlass::half_t*>(tmemODstStack.mDepSmemPtr6)};
      // TmemO.h:381
      smemP += (smemIdxP15) * (int32_t{2048});
      // TmemO.h:493
      cutlass::half_t* smemV{smemPtrV15};
      // TmemO.h:505
      smemV = smemV + ((memIdxV15) * (int32_t{16384}));
      // TmemO.h:535
      bool readD{true};
      // TmemO.h:545
      if ((lastLoopOffset) == (int32_t{0})) {
        // TmemO.h:547
        readD = false;
      }
      // Mma.cpp:618
      {
        // TmemTile.cpp:1755
        uint32_t tmemPtrD{(mTmemBaseOffset) + (uint32_t{96})};
        //
        // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
        //
        // Mma.cpp:203
        uint64_t smemDescA{
          trtllm::dev::createSmemDesc(smemV,
                                      uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                      uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
        //
        // leadingDimInBytes = 2048, strideInBytes = 1024, swizzleMode = 1.
        //
        // Mma.cpp:203
        uint64_t smemDescB{
          trtllm::dev::createSmemDesc(smemP,
                                      uint32_t{0x800000 /*hi=128, lo=0*/},
                                      uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
        //
        // MMA inst for mi=0 ni=0 ki=0.
        //
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                true,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                cuda_ptx::cta_group_1,
                                tmemPtrD,
                                smemDescA,
                                smemDescB,
                                utcmmaDesc_0_0_0,
                                bool{readD});
        }
        //
        // MMA inst for mi=0 ni=0 ki=1.
        //
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                true,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                true,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                true,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{122});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                true,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                true,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                true,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                true,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
    // smemTransformedKv [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:2561
      {
        // Task.cpp:2585
        smemTransformedKvSrcStack.mPipeline.consumer_release(smemTransformedKvConsReleaseState);
      }
      // Task.cpp:43
      ++smemTransformedKvConsReleaseState;
    }
    //
    // smemTransformedKv [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:2745
        smemTransformedKvConsToken =
          smemTransformedKvSrcStack.mPipeline.consumer_try_wait(smemTransformedKvConsState);
      }
      // Task.cpp:2813
      smemTransformedKvSrcStack.mPipeline.consumer_wait(smemTransformedKvConsState,
                                                        smemTransformedKvConsToken);
    }
    //
    // smemTransformedKv [ConsWork (call 7), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{smemTransformedKvConsState.index()};
      // SmemKv.h:322
      smemPtrV4 = &smemTransformedKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
      // SmemKv.h:372
      smemIdxV4 = index;
      // Task.cpp:43
      ++smemTransformedKvConsState;
    }
    //
    // tmemO [ProdWork (call 3), LastIter, FreqInfo{0, 1}, UserTags{3}, Flags{0}].
    //
    // Task.cpp:1477
    smemPtrV15 = smemPtrV4;
    // Task.cpp:1477
    memIdxV15 = smemIdxV4;
    // Task.cpp:1477
    smemIdxP15 = stageIdxP14;
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:5829
      int32_t index{tmemOProdState.index()};
      // TmemO.h:367
      cutlass::half_t* smemP{reinterpret_cast<cutlass::half_t*>(tmemODstStack.mDepSmemPtr6)};
      // TmemO.h:381
      smemP += (smemIdxP15) * (int32_t{2048});
      // TmemO.h:493
      cutlass::half_t* smemV{smemPtrV15};
      // TmemO.h:505
      smemV = smemV + ((memIdxV15) * (int32_t{16384}));
      // TmemO.h:535
      bool readD{true};
      // TmemO.h:545
      if ((lastLoopOffset) == (int32_t{0})) {
        // TmemO.h:547
        readD = false;
      }
      // Mma.cpp:618
      {
        // TmemTile.cpp:1755
        uint32_t tmemPtrD{(mTmemBaseOffset) + (uint32_t{112})};
        //
        // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
        //
        // Mma.cpp:203
        uint64_t smemDescA{
          trtllm::dev::createSmemDesc(smemV,
                                      uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                      uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
        //
        // leadingDimInBytes = 2048, strideInBytes = 1024, swizzleMode = 1.
        //
        // Mma.cpp:203
        uint64_t smemDescB{
          trtllm::dev::createSmemDesc(smemP,
                                      uint32_t{0x800000 /*hi=128, lo=0*/},
                                      uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
        //
        // MMA inst for mi=0 ni=0 ki=0.
        //
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                true,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
                                cuda_ptx::cta_group_1,
                                tmemPtrD,
                                smemDescA,
                                smemDescB,
                                utcmmaDesc_0_0_0,
                                bool{readD});
        }
        //
        // MMA inst for mi=0 ni=0 ki=1.
        //
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                true,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                true,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                true,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{122});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                true,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                true,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                true,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
        // Mma.cpp:886
        trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
        // TmemTile.cpp:1600
        uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                int32_t{0},
                                                                int32_t{0},
                                                                true,
                                                                false,
                                                                int32_t{128},
                                                                int32_t{16},
                                                                int32_t{16},
                                                                false,
                                                                false)};
        // TmemTile.cpp:1700
        if (bool{cute::elect_one_sync()}) {
          // TmemTile.cpp:1708
          cuda_ptx::tcgen05_mma(cuda_ptx::kind_f16,
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
    // smemTransformedKv [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:2561
      {
        // Task.cpp:2585
        smemTransformedKvSrcStack.mPipeline.consumer_release(smemTransformedKvConsReleaseState);
      }
      // Task.cpp:43
      ++smemTransformedKvConsReleaseState;
    }
    //
    // tmemO [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{3}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4427
      {
        // Task.cpp:4443
        tmemODstStack.mPipeline.producer_commit(tmemOProdState);
      }
      // Task.cpp:43
      ++tmemOProdState;
    }
    //
    // tmemS0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    // Task.cpp:1573
    if (hasOneLoopIter) {
      // Task.cpp:4427
      {
        // Task.cpp:4443
        tmemS0DstStack.mPipeline.producer_commit(tmemS0ProdState);
      }
      // Task.cpp:43
      ++tmemS0ProdState;
    }
  //
  // Tail work.
  //
  // Task.cpp:3511
  ExitTileWithSignalingLabel:
  // Task.cpp:3518
  ExitTileWithoutSignalingLabel:
    // Task.cpp:3528
    {}
  }
};
// Task.cpp:544
// Fmha.h:2654
struct PaddingTask {
  // Task.cpp:551
  inline __device__ PaddingTask(fmha::KernelParams const& params,
                                KernelState const& state,
                                int32_t warpGrpStart) {}
  // Task.cpp:507
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:532
    return ((state.mWarpIdx) >= (int32_t{10})) && ((state.mWarpIdx) < (int32_t{11}));
  }
  // Task.cpp:454
  inline __device__ void execute(fmha::KernelParams const& params, KernelState const& state) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<56>{});
  //
  // Tail work.
  //
  // Task.cpp:3511
  ExitTileWithSignalingLabel:
  // Task.cpp:3518
  ExitTileWithoutSignalingLabel:
    // Task.cpp:3528
    {}
  }
};
extern "C" __global__
__launch_bounds__(512, 1) void fmhaSm100fKernel_QFp16KvE4m3OFp16H256PagedKvDenseP64MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen(
  CUTE_GRID_CONSTANT fmha::KernelParams const params) {
  // Kernel.cpp:1654
  trtllm::dev::prefetchTensorMap(&params.tmaQ_);
  // Kernel.cpp:1654
  trtllm::dev::prefetchTensorMap(&params.tmaK_);
  // Kernel.cpp:1654
  trtllm::dev::prefetchTensorMap(&params.tmaV_);
  // Kernel.cpp:1671
  extern __shared__ uint8_t smem__[];
  // Kernel.cpp:1682
  int32_t smemOffset__{int32_t{0}};
  // Kernel.cpp:1721
  smemOffset__ = (((smemOffset__) + (int32_t{1023})) / (int32_t{1024})) * (int32_t{1024});
  // Kernel.cpp:1725
  uint8_t* smemQSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemQSmem)});
  // Kernel.cpp:1721
  smemOffset__ = (((smemOffset__) + (int32_t{1023})) / (int32_t{1024})) * (int32_t{1024});
  // Kernel.cpp:1725
  uint8_t* smemKvSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemKvSmem)});
  // Kernel.cpp:1721
  smemOffset__ = (((smemOffset__) + (int32_t{1023})) / (int32_t{1024})) * (int32_t{1024});
  // Kernel.cpp:1725
  uint8_t* smemTransformedKvSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemTransformedKvSmem)});
  // Kernel.cpp:1721
  smemOffset__ = (((smemOffset__) + (int32_t{1023})) / (int32_t{1024})) * (int32_t{1024});
  // Kernel.cpp:1725
  uint8_t* smemPOSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemPOSmem)});
  // Kernel.cpp:1721
  smemOffset__ = (((smemOffset__) + (int32_t{127})) / (int32_t{128})) * (int32_t{128});
  // Kernel.cpp:1725
  uint8_t* smemPageOffsetsKvSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemPageOffsetsKvSmem)});
  // Kernel.cpp:1721
  smemOffset__ = (((smemOffset__) + (int32_t{15})) / (int32_t{16})) * (int32_t{16});
  // Kernel.cpp:1725
  uint8_t* smemSoftmaxWarpGrpRed0SmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemSoftmaxWarpGrpRed0Smem)});
  // Kernel.cpp:1721
  smemOffset__ = (((smemOffset__) + (int32_t{15})) / (int32_t{16})) * (int32_t{16});
  // Kernel.cpp:1725
  uint8_t* smemSoftmaxWarpGrpRed1SmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemSoftmaxWarpGrpRed0Smem)});
  // Kernel.cpp:1721
  smemOffset__ = (((smemOffset__) + (int32_t{15})) / (int32_t{16})) * (int32_t{16});
  // Kernel.cpp:1725
  uint8_t* smemCorrWarpGrpRed1SmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemCorrWarpGrpRed1Smem)});
  // Kernel.cpp:1698
  uint32_t* TmemSwStatePtr{
    reinterpret_cast<uint32_t*>((reinterpret_cast<uint8_t*>(smem__) + smemOffset__))};
  // Kernel.cpp:1706
  smemOffset__ += int32_t{16};
  // Kernel.cpp:1725
  uint8_t* smemQSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemQSmemBarrier)});
  // Kernel.cpp:1725
  uint8_t* smemKvSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemKvSmemBarrier)});
  // Kernel.cpp:1725
  uint8_t* smemTransformedKvSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemTransformedKvSmemBarrier)});
  // Kernel.cpp:1725
  uint8_t* smemPageOffsetsKvSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemPageOffsetsKvSmemBarrier)});
  // Kernel.cpp:1725
  uint8_t* tmemS0SmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(TmemS0SmemBarrier)});
  // Kernel.cpp:1725
  uint8_t* tmemSoftmaxLocal0SmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(TmemSoftmaxLocal0SmemBarrier)});
  // Kernel.cpp:1725
  uint8_t* orderP01SmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(OrderP01SmemBarrier)});
  // Kernel.cpp:1725
  uint8_t* tmemP0SmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(TmemP0SmemBarrier)});
  // Kernel.cpp:1725
  uint8_t* tmemOSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(TmemOSmemBarrier)});
  // Kernel.cpp:1762
  KernelState const state{params, TmemSwStatePtr};
  // Kernel.cpp:2212
  SmemQSmem* smemQSmem{reinterpret_cast<SmemQSmem*>(smemQSmemPtr)};
  // Kernel.cpp:2224
  SmemQSmemBarrier* smemQSmemBarrier{reinterpret_cast<SmemQSmemBarrier*>(smemQSmemBarrierPtr)};
  // Kernel.cpp:2279
  SmemQStack smemQStack{(*smemQSmem),
                        (*smemQSmemBarrier),
                        state.mWarpIdx,
                        state.mClusterDimX,
                        state.mClusterDimY,
                        int32_t{11},
                        int32_t{-1}};
  // Kernel.cpp:2212
  SmemKvSmem* smemKvSmem{reinterpret_cast<SmemKvSmem*>(smemKvSmemPtr)};
  // Kernel.cpp:2224
  SmemKvSmemBarrier* smemKvSmemBarrier{reinterpret_cast<SmemKvSmemBarrier*>(smemKvSmemBarrierPtr)};
  // Kernel.cpp:2279
  SmemKvStack smemKvStack{(*smemKvSmem),
                          (*smemKvSmemBarrier),
                          state.mWarpIdx,
                          state.mClusterDimX,
                          state.mClusterDimY,
                          int32_t{11},
                          int32_t{-1}};
  // Kernel.cpp:2212
  SmemTransformedKvSmem* smemTransformedKvSmem{
    reinterpret_cast<SmemTransformedKvSmem*>(smemTransformedKvSmemPtr)};
  // Kernel.cpp:2224
  SmemTransformedKvSmemBarrier* smemTransformedKvSmemBarrier{
    reinterpret_cast<SmemTransformedKvSmemBarrier*>(smemTransformedKvSmemBarrierPtr)};
  // Kernel.cpp:2279
  SmemTransformedKvStack smemTransformedKvStack{(*smemTransformedKvSmem),
                                                (*smemTransformedKvSmemBarrier),
                                                state.mWarpIdx,
                                                state.mClusterDimX,
                                                state.mClusterDimY,
                                                int32_t{4},
                                                int32_t{-1}};
  // Kernel.cpp:2212
  SmemPageOffsetsKvSmem* smemPageOffsetsKvSmem{
    reinterpret_cast<SmemPageOffsetsKvSmem*>(smemPageOffsetsKvSmemPtr)};
  // Kernel.cpp:2224
  SmemPageOffsetsKvSmemBarrier* smemPageOffsetsKvSmemBarrier{
    reinterpret_cast<SmemPageOffsetsKvSmemBarrier*>(smemPageOffsetsKvSmemBarrierPtr)};
  // Kernel.cpp:2279
  SmemPageOffsetsKvStack smemPageOffsetsKvStack{(*smemPageOffsetsKvSmem),
                                                (*smemPageOffsetsKvSmemBarrier),
                                                state.mWarpIdx,
                                                state.mClusterDimX,
                                                state.mClusterDimY,
                                                int32_t{9},
                                                int32_t{-1}};
  // Kernel.cpp:2212
  SmemPOSmem* smemPOSmem{reinterpret_cast<SmemPOSmem*>(smemPOSmemPtr)};
  // Kernel.cpp:2279
  SmemPOStack smemPOStack{(*smemPOSmem),
                          state.mWarpIdx,
                          state.mClusterDimX,
                          state.mClusterDimY,
                          int32_t{0},
                          int32_t{-1}};
  // Kernel.cpp:2212
  SmemSoftmaxWarpGrpRed0Smem* smemSoftmaxWarpGrpRed0Smem{
    reinterpret_cast<SmemSoftmaxWarpGrpRed0Smem*>(smemSoftmaxWarpGrpRed0SmemPtr)};
  // Kernel.cpp:2279
  SmemSoftmaxWarpGrpRed0Stack smemSoftmaxWarpGrpRed0Stack{(*smemSoftmaxWarpGrpRed0Smem),
                                                          state.mWarpIdx,
                                                          state.mClusterDimX,
                                                          state.mClusterDimY,
                                                          int32_t{0},
                                                          int32_t{-1}};
  // Kernel.cpp:2212
  SmemSoftmaxWarpGrpRed0Smem* smemSoftmaxWarpGrpRed1Smem{
    reinterpret_cast<SmemSoftmaxWarpGrpRed0Smem*>(smemSoftmaxWarpGrpRed1SmemPtr)};
  // Kernel.cpp:2279
  SmemSoftmaxWarpGrpRed0Stack smemSoftmaxWarpGrpRed1Stack{(*smemSoftmaxWarpGrpRed1Smem),
                                                          state.mWarpIdx,
                                                          state.mClusterDimX,
                                                          state.mClusterDimY,
                                                          int32_t{0},
                                                          int32_t{-1}};
  // Kernel.cpp:2212
  SmemCorrWarpGrpRed1Smem* smemCorrWarpGrpRed1Smem{
    reinterpret_cast<SmemCorrWarpGrpRed1Smem*>(smemCorrWarpGrpRed1SmemPtr)};
  // Kernel.cpp:2279
  SmemCorrWarpGrpRed1Stack smemCorrWarpGrpRed1Stack{(*smemCorrWarpGrpRed1Smem),
                                                    state.mWarpIdx,
                                                    state.mClusterDimX,
                                                    state.mClusterDimY,
                                                    int32_t{0},
                                                    int32_t{-1}};
  // Kernel.cpp:2224
  TmemS0SmemBarrier* tmemS0SmemBarrier{reinterpret_cast<TmemS0SmemBarrier*>(tmemS0SmemBarrierPtr)};
  // Kernel.cpp:2279
  TmemS0Stack tmemS0Stack{(*tmemS0SmemBarrier),
                          (*smemSoftmaxWarpGrpRed0Smem),
                          smemSoftmaxWarpGrpRed0Stack,
                          state.mWarpIdx,
                          state.mClusterDimX,
                          state.mClusterDimY,
                          int32_t{1},
                          int32_t{-1},
                          int32_t{1},
                          int32_t{0}};
  // Kernel.cpp:2224
  TmemSoftmaxLocal0SmemBarrier* tmemSoftmaxLocal0SmemBarrier{
    reinterpret_cast<TmemSoftmaxLocal0SmemBarrier*>(tmemSoftmaxLocal0SmemBarrierPtr)};
  // Kernel.cpp:2279
  TmemSoftmaxLocal0Stack tmemSoftmaxLocal0Stack{(*tmemSoftmaxLocal0SmemBarrier),
                                                state.mWarpIdx,
                                                state.mClusterDimX,
                                                state.mClusterDimY,
                                                int32_t{5},
                                                int32_t{-1},
                                                int32_t{0}};
  // Kernel.cpp:2279
  TmemSoftmaxGlobal0Stack tmemSoftmaxGlobal0Stack{state.mWarpIdx,
                                                  state.mClusterDimX,
                                                  state.mClusterDimY,
                                                  int32_t{0},
                                                  int32_t{-1}};
  // Kernel.cpp:2224
  OrderP01SmemBarrier* orderP01SmemBarrier{
    reinterpret_cast<OrderP01SmemBarrier*>(orderP01SmemBarrierPtr)};
  // Kernel.cpp:2279
  OrderP01Stack orderP01Stack{(*orderP01SmemBarrier),
                              state.mWarpIdx,
                              state.mClusterDimX,
                              state.mClusterDimY,
                              int32_t{0},
                              int32_t{-1}};
  // Kernel.cpp:2224
  TmemP0SmemBarrier* tmemP0SmemBarrier{reinterpret_cast<TmemP0SmemBarrier*>(tmemP0SmemBarrierPtr)};
  // Kernel.cpp:2279
  TmemP0Stack tmemP0Stack{(*tmemP0SmemBarrier),
                          (*smemPOSmem),
                          smemPOStack,
                          (*orderP01SmemBarrier),
                          orderP01Stack,
                          state.mWarpIdx,
                          state.mClusterDimX,
                          state.mClusterDimY,
                          int32_t{2},
                          int32_t{-1},
                          int32_t{3},
                          int32_t{0}};
  // Kernel.cpp:2224
  TmemOSmemBarrier* tmemOSmemBarrier{reinterpret_cast<TmemOSmemBarrier*>(tmemOSmemBarrierPtr)};
  // Kernel.cpp:2279
  TmemOStack tmemOStack{(*tmemOSmemBarrier),
                        (*smemPOSmem),
                        smemPOStack,
                        state.mWarpIdx,
                        state.mClusterDimX,
                        state.mClusterDimY,
                        int32_t{4},
                        int32_t{-1}};
  // Kernel.cpp:2279
  TmemCorr0Stack tmemCorr0Stack{(*smemCorrWarpGrpRed1Smem),
                                smemCorrWarpGrpRed1Stack,
                                (*smemPOSmem),
                                smemPOStack,
                                state.mWarpIdx,
                                state.mClusterDimX,
                                state.mClusterDimY,
                                int32_t{0},
                                int32_t{-1}};
  // Kernel.cpp:1858
  cutlass::arch::fence_barrier_init();
  // Kernel.cpp:2316
  if ((reinterpret_cast<int32_t const&>(threadIdx.x)) < (int32_t{32})) {
    // Kernel.cpp:2340
    cuda_ptx::tcgen05_alloc(cuda_ptx::cta_group_1_t{}, state.mTmemSwStatePtr, int32_t{512});
    // Kernel.cpp:2353
    cuda_ptx::tcgen05_relinquish_alloc_permit(cuda_ptx::cta_group_1_t{});
  }
  // Kernel.cpp:1882
  __syncthreads();
  // Kernel.cpp:2010
  if (bool{LoadPageOffsetsTask::isSelected(params, state)}) {
    // Kernel.cpp:2077
    LoadPageOffsetsTask loadPageOffsetsTask{params, state, int32_t{9}};
    // Kernel.cpp:2131
    loadPageOffsetsTask.execute(params, state, (*smemPageOffsetsKvSmem), smemPageOffsetsKvStack);
  } else {
    // Kernel.cpp:2010
    if (bool{LoadTask::isSelected(params, state)}) {
      // Kernel.cpp:2077
      LoadTask loadTask{params, state, int32_t{11}};
      // Kernel.cpp:2131
      loadTask.execute(params,
                       state,
                       (*smemQSmem),
                       smemQStack,
                       (*smemKvSmem),
                       smemKvStack,
                       (*smemPageOffsetsKvSmem),
                       smemPageOffsetsKvStack);
    } else {
      // Kernel.cpp:2010
      if (bool{SoftmaxTask0::isSelected(params, state)}) {
        // Kernel.cpp:2077
        SoftmaxTask0 softmaxTask0{params, state, int32_t{0}};
        // Kernel.cpp:2131
        softmaxTask0.execute(params,
                             state,
                             tmemSoftmaxLocal0Stack,
                             tmemSoftmaxGlobal0Stack,
                             tmemP0Stack,
                             tmemS0Stack);
      } else {
        // Kernel.cpp:2010
        if (bool{CorrTask::isSelected(params, state)}) {
          // Kernel.cpp:2077
          CorrTask corrTask{params, state, int32_t{4}};
          // Kernel.cpp:2131
          corrTask.execute(params, state, tmemCorr0Stack, tmemSoftmaxLocal0Stack, tmemOStack);
          // Task.cpp:5288
          trtllm::dev::CutlassNamedBarrier::sync(128, 11);
          // Task.cpp:5296
          int32_t const warpGrpThreadIdx{(state.mThreadIdx) - (int32_t{128})};
          // Task.cpp:5312
          if ((warpGrpThreadIdx) < (int32_t{32})) {
            // Task.cpp:5342
            cuda_ptx::tcgen05_dealloc(cuda_ptx::cta_group_1_t{},
                                      uint32_t{__shfl_sync(uint32_t{0xffffffff},
                                                           (*state.mTmemSwStatePtr),
                                                           int32_t{0},
                                                           int32_t{32})},
                                      int32_t{512});
          }
        } else {
          // Kernel.cpp:2010
          if (bool{TransformKvTask::isSelected(params, state)}) {
            // Kernel.cpp:2077
            TransformKvTask transformKvTask{params, state, int32_t{12}};
            // Kernel.cpp:2131
            transformKvTask.execute(params,
                                    state,
                                    (*smemTransformedKvSmem),
                                    smemTransformedKvStack,
                                    (*smemKvSmem),
                                    smemKvStack);
          } else {
            // Kernel.cpp:2010
            if (bool{MmaTask::isSelected(params, state)}) {
              // Kernel.cpp:2077
              MmaTask mmaTask{params, state, int32_t{8}};
              // Kernel.cpp:2131
              mmaTask.execute(params,
                              state,
                              tmemS0Stack,
                              tmemOStack,
                              (*smemQSmem),
                              smemQStack,
                              (*smemTransformedKvSmem),
                              smemTransformedKvStack,
                              tmemP0Stack);
            } else {
              // Kernel.cpp:2010
              if (bool{PaddingTask::isSelected(params, state)}) {
                // Kernel.cpp:2077
                PaddingTask paddingTask{params, state, int32_t{10}};
                // Kernel.cpp:2131
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
fmhaSm100fKernel_QFp16KvE4m3OFp16H256PagedKvDenseP64MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGenGetSmemSize(
  int32_t* outPtr) {
  int32_t size{0};
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemQSmem));
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemKvSmem));
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemTransformedKvSmem));
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemPOSmem));
  size = (size + 127) / 128 * 128;
  size += static_cast<int32_t>(sizeof(SmemPageOffsetsKvSmem));
  size = (size + 15) / 16 * 16;
  size += static_cast<int32_t>(sizeof(SmemSoftmaxWarpGrpRed0Smem));
  size = (size + 15) / 16 * 16;
  size += static_cast<int32_t>(sizeof(SmemSoftmaxWarpGrpRed0Smem));
  size = (size + 15) / 16 * 16;
  size += static_cast<int32_t>(sizeof(SmemCorrWarpGrpRed1Smem));
  size += 16;
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(SmemQSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(SmemKvSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(SmemTransformedKvSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(SmemPageOffsetsKvSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(TmemS0SmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(TmemSoftmaxLocal0SmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(OrderP01SmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(TmemP0SmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(TmemOSmemBarrier));
  outPtr[0] = size;
}
