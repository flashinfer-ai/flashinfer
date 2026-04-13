#include <FmhaSm100fKernel_QkvE4m3OBfloat16H64PagedKvSlidingOrChunkedCausalP64VarSeqQ128Kv128PersistentContext.h>

// Res.cpp:137
// Fmha.h:845
struct SmemQStack {
  // Res.cpp:595
  trtllm::dev::CutlassTmaUmmaAsyncPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  // Res.cpp:208
  inline __device__ SmemQStack(SmemQSmem& smemQSmem,
                               SmemQSmemBarrier& smemQSmemBarrier,
                               int32_t warpId,
                               int32_t clusterDimX,
                               int32_t clusterDimY,
                               int32_t barInitWarpId,
                               int32_t orderedSequenceGroupId)
    : // Res.cpp:719
    mPipeline{smemQSmemBarrier.mBarriers,
              warpId,
              int32_t{8192},
              bool{cute::elect_one_sync()},
              CuteFlatTuple213{},
              cute::true_type{},
              cute::true_type{},
              barInitWarpId} {}
};
// Res.cpp:137
// Fmha.h:1117
struct SmemKvStack {
  // Res.cpp:595
  trtllm::dev::CutlassTmaUmmaAsyncPipeline<3, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
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
    : // Res.cpp:719
    mPipeline{smemKvSmemBarrier.mBarriers,
              warpId,
              int32_t{8192},
              bool{cute::elect_one_sync()},
              CuteFlatTuple333{},
              cute::true_type{},
              cute::true_type{},
              barInitWarpId}
    , // MemBuffers.cpp:282
    mPtr{&smemKvSmem.mArray[int32_t{0}][int32_t{0}]} {}
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
// Fmha.h:1520
struct WorkIdStorageStack {
  // Res.cpp:595
  trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  // Res.cpp:1028
  cutlass::gemm::kernel::detail::
    PersistentTileSchedulerSm100<cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, 2>
      mScheduler;
  // Res.cpp:1031
  typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
    2>::WorkTileInfo workTileInfo;
  // Res.cpp:208
  inline __device__ WorkIdStorageStack(WorkIdStorageSmem& workIdStorageSmem,
                                       WorkIdStorageSmemBarrier& workIdStorageSmemBarrier,
                                       int32_t warpId,
                                       int32_t clusterDimX,
                                       int32_t clusterDimY,
                                       int32_t barInitWarpId,
                                       int32_t orderedSequenceGroupId)
    : // Res.cpp:810
    mPipeline{workIdStorageSmemBarrier.mBarriers,
              CuteFlatTuple576{},
              int32_t{1},
              int32_t{512},
              int32_t{0},
              barInitWarpId}
    , // Res.cpp:1046
    mScheduler{&workIdStorageSmem.workIdResponse[int32_t{0}],
               typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<
                 cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
                 2>::Params{},
               cute::block_id_in_cluster()}
    , // Res.cpp:1086
    workTileInfo{mScheduler.initial_work_tile_info(CuteFlatTuple772{})} {}
};
// Res.cpp:137
// Fmha.h:1525
struct WorkIdThrottleBarrierStack {
  // Res.cpp:595
  trtllm::dev::CutlassCpAsyncPipeline<2, true> mPipeline;
  // Res.cpp:208
  inline __device__ WorkIdThrottleBarrierStack(
    WorkIdThrottleBarrierSmemBarrier& workIdThrottleBarrierSmemBarrier,
    int32_t warpId,
    int32_t clusterDimX,
    int32_t clusterDimY,
    int32_t barInitWarpId,
    int32_t orderedSequenceGroupId)
    : // Res.cpp:644
    mPipeline{workIdThrottleBarrierSmemBarrier.mBarriers,
              warpId,
              int32_t{32},
              int32_t{32},
              barInitWarpId} {}
};
// Res.cpp:137
// Fmha.h:1890
struct TmemS0Stack {
  // Res.cpp:595
  trtllm::dev::CutlassUmmaAsyncPipeline<1, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  // TmemS.h:549
  int32_t const mNamedBarId;
  // TmemS.h:552
  int32_t const mInstId;
  // Res.cpp:208
  inline __device__ TmemS0Stack(TmemS0SmemBarrier& tmemS0SmemBarrier,
                                int32_t warpId,
                                int32_t clusterDimX,
                                int32_t clusterDimY,
                                int32_t barInitWarpId,
                                int32_t orderedSequenceGroupId,
                                int32_t namedBarId,
                                int32_t instId)
    : // Res.cpp:776
    mPipeline{tmemS0SmemBarrier.mBarriers,
              warpId,
              int32_t{128},
              CuteFlatTuple995{},
              cute::true_type{},
              cute::true_type{},
              barInitWarpId}
    , // TmemS.h:540
    mNamedBarId{namedBarId}
    , // TmemS.h:542
    mInstId{instId} {}
};
// Res.cpp:137
// SoftmaxSchedule.h:295
struct TmemSoftmaxLocal0Stack {
  // Res.cpp:595
  trtllm::dev::CutlassCpAsyncPipeline<1, true> mPipeline;
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
// SoftmaxSchedule.h:306
struct TmemSoftmaxGlobal0Stack {
  // Res.cpp:208
  inline __device__ TmemSoftmaxGlobal0Stack(int32_t warpId,
                                            int32_t clusterDimX,
                                            int32_t clusterDimY,
                                            int32_t barInitWarpId,
                                            int32_t orderedSequenceGroupId) {}
};
// Res.cpp:137
// Fmha.h:1987
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
// Fmha.h:2017
struct TmemP0Stack {
  // Res.cpp:544
  trtllm::dev::CutlassOrderedSequenceBarrier<1, 2> mOrderedSequence;
  // Res.cpp:595
  trtllm::dev::CutlassCpAsyncPipeline<1, true> mPipeline;
  // TmemP.h:472
  int32_t const mNamedBarId;
  // TmemP.h:475
  int32_t const mInstId;
  // Res.cpp:208
  inline __device__ TmemP0Stack(TmemP0SmemBarrier& tmemP0SmemBarrier,
                                OrderP01SmemBarrier& orderP01SmemBarrier,
                                OrderP01Stack& orderP01Stack,
                                int32_t warpId,
                                int32_t clusterDimX,
                                int32_t clusterDimY,
                                int32_t barInitWarpId,
                                int32_t orderedSequenceGroupId,
                                int32_t namedBarId,
                                int32_t instId)
    : // Res.cpp:560
    mOrderedSequence{orderP01SmemBarrier.mOrderedSequenceBarriers,
                     warpId,
                     orderedSequenceGroupId,
                     int32_t{128},
                     barInitWarpId}
    , // Res.cpp:644
    mPipeline{tmemP0SmemBarrier.mBarriers, warpId, int32_t{128}, int32_t{32}, barInitWarpId}
    , // TmemP.h:463
    mNamedBarId{namedBarId}
    , // TmemP.h:465
    mInstId{instId} {}
};
// Res.cpp:137
// Fmha.h:2104
struct TmemOStack {
  // Res.cpp:595
  trtllm::dev::CutlassUmmaAsyncPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  // Res.cpp:208
  inline __device__ TmemOStack(TmemOSmemBarrier& tmemOSmemBarrier,
                               int32_t warpId,
                               int32_t clusterDimX,
                               int32_t clusterDimY,
                               int32_t barInitWarpId,
                               int32_t orderedSequenceGroupId)
    : // Res.cpp:776
    mPipeline{tmemOSmemBarrier.mBarriers,
              warpId,
              int32_t{128},
              CuteFlatTuple1364{},
              cute::true_type{},
              cute::true_type{},
              barInitWarpId} {}
};
// Res.cpp:137
// Fmha.h:2121
struct TmemCorr0Stack {
  // Res.cpp:208
  inline __device__ TmemCorr0Stack(int32_t warpId,
                                   int32_t clusterDimX,
                                   int32_t clusterDimY,
                                   int32_t barInitWarpId,
                                   int32_t orderedSequenceGroupId) {}
};
// Res.cpp:137
// Fmha.h:2136
struct TmemCorr1Stack {
  // Res.cpp:208
  inline __device__ TmemCorr1Stack(int32_t warpId,
                                   int32_t clusterDimX,
                                   int32_t clusterDimY,
                                   int32_t barInitWarpId,
                                   int32_t orderedSequenceGroupId) {}
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
    : // Kernel.cpp:2401
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
// Task.cpp:559
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
  // FmhaTask.h:216
  int32_t mNumCtasKv;
  // FmhaTask.h:226
  int32_t mCtaIdxKv;
  // FmhaTask.h:224
  int32_t mCtaIdxQ;
  // FmhaTask.h:214
  int32_t mSeqLenKv;
  // FmhaTask.h:733
  int32_t mNumSkippedTilesKv;
  // Task.cpp:371
  int32_t const mWarpGrpThreadIdx;
  // Task.cpp:566
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
    , // Kernel.cpp:212
    mNumCtasKv{int32_t{1}}
    , // Kernel.cpp:210
    mCtaIdxKv{int32_t{0}}
    , // Task.cpp:283
    mCtaIdxQ{mCtaIdxX}
    , // FmhaTask.h:437
    mSeqLenKv{int32_t{
      params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                            ? ((((mHeadIdx) / (params.mNumHeadsQPerKv)) * (params.mBatchSize)) +
                               (mBatchIdx))
                            : (mBatchIdx)]}}
    , // Kernel.cpp:210
    mNumSkippedTilesKv{int32_t{0}}
    , // Task.cpp:379
    mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))} {}
  // Task.cpp:522
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:547
    return ((state.mWarpIdx) >= (int32_t{13})) && ((state.mWarpIdx) < (int32_t{14}));
  }
  // Task.cpp:454
  inline __device__ void execute(fmha::KernelParams const& params,
                                 KernelState const& state,
                                 SmemPageOffsetsKvSmem& smemPageOffsetsKvDstSmem,
                                 SmemPageOffsetsKvStack& smemPageOffsetsKvDstStack,
                                 WorkIdStorageSmem& workIdStorageSrcSmem,
                                 WorkIdStorageStack& workIdStorageSrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<56>{});
    // Task.cpp:2114
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsReleaseState{};
    // Task.cpp:2135
    int32_t workIdStorageConsToken{int32_t{0}};
    // Task.cpp:2013
    trtllm::dev::CutlassCpAsyncPipeline<6>::PipelineState smemPageOffsetsKvProdState{int32_t{0},
                                                                                     int32_t{1},
                                                                                     int32_t{0}};
    // Task.cpp:2033
    int32_t smemPageOffsetsKvProdToken{int32_t{1}};
    // Task.cpp:5485
    do {
      // FmhaTask.h:333
      int32_t currSeqCtaIdx;
      // FmhaTask.h:342
      currSeqCtaIdx = workIdStorageSrcStack.workTileInfo.M_idx;
      // FmhaTask.h:357
      mHeadIdx = workIdStorageSrcStack.workTileInfo.N_idx;
      // FmhaTask.h:361
      mBatchIdx = workIdStorageSrcStack.workTileInfo.L_idx;
      // FmhaTask.h:139
      mSeqOffsetQ = int32_t(bool{params.ptrCumSeqLensQ == nullptr})
                      ? ((mBatchIdx) * (params.mMaxSeqLenQ))
                      : (int32_t{params.ptrCumSeqLensQ[mBatchIdx]});
      // FmhaTask.h:139
      mSeqLenQ = int32_t(bool{params.ptrCumSeqLensQ == nullptr})
                   ? (params.mMaxSeqLenQ)
                   : ((int32_t{params.ptrCumSeqLensQ[(mBatchIdx) + (int32_t{1})]}) - (mSeqOffsetQ));
      // FmhaTask.h:491
      mCtaIdxQ = currSeqCtaIdx;
      // FmhaTask.h:139
      mSeqLenKv = int32_t{
        params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                              ? ((((mHeadIdx) / (params.mNumHeadsQPerKv)) * (params.mBatchSize)) +
                                 (mBatchIdx))
                              : (mBatchIdx)]};
      // FmhaTask.h:582
      int32_t numLoopSteps;
      // FmhaTask.h:592
      int32_t diffKvQ{(mSeqLenKv) - (mSeqLenQ)};
      // FmhaTask.h:597
      int32_t validSeqLenKv;
      // Common.h:63
      if ((params.mChunkedAttentionSizeLog2) > (int32_t{0})) {
        // FmhaTask.h:748
        mNumSkippedTilesKv =
          (((((mCtaIdxQ) * (int32_t{256})) + (diffKvQ)) >> (params.mChunkedAttentionSizeLog2))
           << (params.mChunkedAttentionSizeLog2)) /
          (int32_t{128});
      } else {
        // FmhaTask.h:767
        mNumSkippedTilesKv =
          (int32_t{max(int32_t{0},
                       ((((mCtaIdxQ) * (int32_t{256})) + (diffKvQ)) + (int32_t{1})) -
                         (params.mAttentionWindowSize))}) /
          (int32_t{128});
      }
      // FmhaTask.h:603
      validSeqLenKv =
        (int32_t{min((((mCtaIdxQ) * (int32_t{256})) + (diffKvQ)) + (int32_t{256}), mSeqLenKv)}) -
        ((mNumSkippedTilesKv) * (int32_t{128}));
      // FmhaTask.h:616
      mNumCtasKv =
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, int32_t{1})};
      // FmhaTask.h:630
      if ((((mCtaIdxQ) * (int32_t{256})) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
        // FmhaTask.h:668
        int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
                         ((mNumCtasKv) * (int32_t{128}))};
        // FmhaTask.h:682
        numLoopSteps = numSteps;
      } else {
        // FmhaTask.h:648
        numLoopSteps = int32_t{0};
      }
      // SmemPageOffsetsKv.h:204
      int32_t const* ptrPageIdxK4;
      // SmemPageOffsetsKv.h:210
      ptrPageIdxK4 = params.ptrPageIdxKv +
                     (int32_t(params.mUseBlockSparseAttention)
                        ? ((int32_t(params.mUsesSharedPagedKvIdx)
                              ? ((mBatchIdx) * (params.mMaxNumPagesPerSeqKv))
                              : (((mBatchIdx) * (params.mMaxNumPagesPerSeqKv)) * (int32_t{2}))) +
                           (int32_t(params.mUsesSharedPagedKvIdx)
                              ? (((params.mBatchSize) * (params.mMaxNumPagesPerSeqKv)) *
                                 ((mHeadIdx) / (params.mNumHeadsQPerKv)))
                              : ((((params.mBatchSize) * (params.mMaxNumPagesPerSeqKv)) *
                                  ((mHeadIdx) / (params.mNumHeadsQPerKv))) *
                                 (int32_t{2}))))
                        : (int32_t(params.mUsesSharedPagedKvIdx)
                             ? ((mBatchIdx) * (params.mMaxNumPagesPerSeqKv))
                             : (((mBatchIdx) * (params.mMaxNumPagesPerSeqKv)) * (int32_t{2}))));
      // SmemPageOffsetsKv.h:215
      int32_t const* ptrPageIdxV4;
      // SmemPageOffsetsKv.h:224
      if (params.mUsesSharedPagedKvIdx) {
        // SmemPageOffsetsKv.h:226
        ptrPageIdxV4 = ptrPageIdxK4;
      } else {
        // SmemPageOffsetsKv.h:228
        ptrPageIdxV4 = ptrPageIdxK4 + (params.mMaxNumPagesPerSeqKv);
      }
      // SmemPageOffsetsKv.h:279
      int32_t pageIdxLb4{
        ((((mCtaIdxQ) * (int32_t{256}) + ((mSeqLenKv) - (mSeqLenQ))) + (int32_t{1})) -
         (params.mAttentionWindowSize)) /
        (int32_t{64})};
      // SmemPageOffsetsKv.h:302
      int32_t pageIdxUb4{(int32_t{((mSeqLenKv) + (int32_t{63})) / (int32_t{64})}) - (int32_t{1})};
      //
      // Loop body.
      //
      // Task.cpp:3392
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset564 = int32_t{0}; loopOffset564 < numLoopSteps;
           loopOffset564 += int32_t{16}) {
        // Task.cpp:3445
        bool const isFirstLoopIter{(loopOffset564) == (int32_t{0})};
        // Task.cpp:3465
        bool const isLastLoopIter{((loopOffset564) + (int32_t{16})) >= (numLoopSteps)};
        //
        // smemPageOffsetsKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset564) >= (int32_t{0})) {
            // Task.cpp:5100
            smemPageOffsetsKvProdToken =
              smemPageOffsetsKvDstStack.mPipeline.producer_try_acquire(smemPageOffsetsKvProdState);
          }
        }
        // Task.cpp:1607
        // Task.cpp:4288
        {
          // Task.cpp:4318
          smemPageOffsetsKvDstStack.mPipeline.producer_acquire(smemPageOffsetsKvProdState,
                                                               smemPageOffsetsKvProdToken);
        }
        //
        // smemPageOffsetsKv [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{smemPageOffsetsKvProdState.index()};
          // SmemPageOffsetsKv.h:390
          int32_t* ptrSmemPageOffsets;
          // SmemPageOffsetsKv.h:392
          ptrSmemPageOffsets = smemPageOffsetsKvDstSmem.mArray[index] + (mWarpGrpThreadIdx);
          // SmemPageOffsetsKv.h:430
          int32_t pageIdx{
            (((mNumSkippedTilesKv) + ((mCtaIdxKv) * (numLoopSteps) + (loopOffset564))) *
             (int32_t{2})) +
            (mWarpGrpThreadIdx)};
          // SmemPageOffsetsKv.h:488
          trtllm::dev::cpAsync((ptrSmemPageOffsets + int32_t{0}),
                               (ptrPageIdxK4 + int32_t{min(pageIdx, pageIdxUb4)}),
                               int32_t{0},
                               int32_t{0},
                               int32_t{4});
        }
        //
        // smemPageOffsetsKv [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:4522
        {
          // Task.cpp:4540
          {
            // Task.cpp:4556
            smemPageOffsetsKvDstStack.mPipeline.producer_commit(smemPageOffsetsKvProdState);
          }
          // Task.cpp:43
          ++smemPageOffsetsKvProdState;
        }
        //
        // smemPageOffsetsKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset564) >= (int32_t{0})) {
            // Task.cpp:5100
            smemPageOffsetsKvProdToken =
              smemPageOffsetsKvDstStack.mPipeline.producer_try_acquire(smemPageOffsetsKvProdState);
          }
        }
        // Task.cpp:1607
        // Task.cpp:4288
        {
          // Task.cpp:4318
          smemPageOffsetsKvDstStack.mPipeline.producer_acquire(smemPageOffsetsKvProdState,
                                                               smemPageOffsetsKvProdToken);
        }
        //
        // smemPageOffsetsKv [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{smemPageOffsetsKvProdState.index()};
          // SmemPageOffsetsKv.h:390
          int32_t* ptrSmemPageOffsets;
          // SmemPageOffsetsKv.h:392
          ptrSmemPageOffsets = smemPageOffsetsKvDstSmem.mArray[index] + (mWarpGrpThreadIdx);
          // SmemPageOffsetsKv.h:430
          int32_t pageIdx{
            (((mNumSkippedTilesKv) + ((mCtaIdxKv) * (numLoopSteps) + (loopOffset564))) *
             (int32_t{2})) +
            (mWarpGrpThreadIdx)};
          // SmemPageOffsetsKv.h:488
          trtllm::dev::cpAsync((ptrSmemPageOffsets + int32_t{0}),
                               (ptrPageIdxV4 + int32_t{min(pageIdx, pageIdxUb4)}),
                               int32_t{0},
                               int32_t{0},
                               int32_t{4});
        }
        //
        // smemPageOffsetsKv [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
        //
        // Task.cpp:4522
        {
          // Task.cpp:4540
          {
            // Task.cpp:4556
            smemPageOffsetsKvDstStack.mPipeline.producer_commit(smemPageOffsetsKvProdState);
          }
          // Task.cpp:43
          ++smemPageOffsetsKvProdState;
        }
      }
    //
    // Tail work.
    //
    // Task.cpp:3553
    ExitTileWithSignalingLabel:
    // Task.cpp:3560
    ExitTileWithoutSignalingLabel:
      // Task.cpp:5532
      auto newWorkTileInfoTuple{
        workIdStorageSrcStack.mScheduler.fetch_next_work(workIdStorageSrcStack.workTileInfo,
                                                         workIdStorageSrcStack.mPipeline,
                                                         workIdStorageConsState)};
      // Task.cpp:5534
      workIdStorageSrcStack.workTileInfo = cute::get<int32_t{0}>(newWorkTileInfoTuple);
      // Task.cpp:5542
      ++workIdStorageConsState;
      // Task.cpp:5644
      mCtaIdxX = workIdStorageSrcStack.workTileInfo.M_idx;
      // Task.cpp:5645
      mCtaIdxY = workIdStorageSrcStack.workTileInfo.N_idx;
      // Task.cpp:5646
      mCtaIdxZ = workIdStorageSrcStack.workTileInfo.L_idx;
    } while (workIdStorageSrcStack.workTileInfo.is_valid_tile);
  }
};
// Task.cpp:559
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
  // FmhaTask.h:216
  int32_t mNumCtasKv;
  // FmhaTask.h:226
  int32_t mCtaIdxKv;
  // FmhaTask.h:224
  int32_t mCtaIdxQ;
  // FmhaTask.h:214
  int32_t mSeqLenKv;
  // FmhaTask.h:733
  int32_t mNumSkippedTilesKv;
  // Task.cpp:566
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
    , // Kernel.cpp:212
    mNumCtasKv{int32_t{1}}
    , // Kernel.cpp:210
    mCtaIdxKv{int32_t{0}}
    , // Task.cpp:283
    mCtaIdxQ{mCtaIdxX}
    , // FmhaTask.h:437
    mSeqLenKv{int32_t{
      params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                            ? ((((mHeadIdx) / (params.mNumHeadsQPerKv)) * (params.mBatchSize)) +
                               (mBatchIdx))
                            : (mBatchIdx)]}}
    , // Kernel.cpp:210
    mNumSkippedTilesKv{int32_t{0}} {}
  // Task.cpp:522
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:547
    return ((state.mWarpIdx) >= (int32_t{15})) && ((state.mWarpIdx) < (int32_t{16}));
  }
  // Task.cpp:454
  inline __device__ void execute(fmha::KernelParams const& params,
                                 KernelState const& state,
                                 SmemQSmem& smemQDstSmem,
                                 SmemQStack& smemQDstStack,
                                 SmemKvSmem& smemKvDstSmem,
                                 SmemKvStack& smemKvDstStack,
                                 WorkIdThrottleBarrierStack& workIdThrottleBarrierDstStack,
                                 SmemPageOffsetsKvSmem& smemPageOffsetsKvSrcSmem,
                                 SmemPageOffsetsKvStack& smemPageOffsetsKvSrcStack,
                                 WorkIdStorageSmem& workIdStorageSrcSmem,
                                 WorkIdStorageStack& workIdStorageSrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<56>{});
    // Task.cpp:2114
    trtllm::dev::CutlassCpAsyncPipeline<6>::PipelineState smemPageOffsetsKvConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassCpAsyncPipeline<6>::PipelineState smemPageOffsetsKvConsReleaseState{};
    // Task.cpp:2135
    int32_t smemPageOffsetsKvConsToken{int32_t{0}};
    // Task.cpp:2114
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsReleaseState{};
    // Task.cpp:2135
    int32_t workIdStorageConsToken{int32_t{0}};
    // Task.cpp:2013
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      2,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemQProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    // Task.cpp:2033
    int32_t smemQProdToken{int32_t{1}};
    // Task.cpp:2013
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      3,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemKvProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    // Task.cpp:2033
    int32_t smemKvProdToken{int32_t{1}};
    // Task.cpp:2013
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState workIdThrottleBarrierProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    // Task.cpp:2033
    int32_t workIdThrottleBarrierProdToken{int32_t{1}};
    // Task.cpp:5485
    do {
      // FmhaTask.h:333
      int32_t currSeqCtaIdx;
      // FmhaTask.h:342
      currSeqCtaIdx = workIdStorageSrcStack.workTileInfo.M_idx;
      // FmhaTask.h:357
      mHeadIdx = workIdStorageSrcStack.workTileInfo.N_idx;
      // FmhaTask.h:361
      mBatchIdx = workIdStorageSrcStack.workTileInfo.L_idx;
      // FmhaTask.h:139
      mSeqOffsetQ = int32_t(bool{params.ptrCumSeqLensQ == nullptr})
                      ? ((mBatchIdx) * (params.mMaxSeqLenQ))
                      : (int32_t{params.ptrCumSeqLensQ[mBatchIdx]});
      // FmhaTask.h:139
      mSeqLenQ = int32_t(bool{params.ptrCumSeqLensQ == nullptr})
                   ? (params.mMaxSeqLenQ)
                   : ((int32_t{params.ptrCumSeqLensQ[(mBatchIdx) + (int32_t{1})]}) - (mSeqOffsetQ));
      // FmhaTask.h:491
      mCtaIdxQ = currSeqCtaIdx;
      // FmhaTask.h:139
      mSeqLenKv = int32_t{
        params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                              ? ((((mHeadIdx) / (params.mNumHeadsQPerKv)) * (params.mBatchSize)) +
                                 (mBatchIdx))
                              : (mBatchIdx)]};
      // FmhaTask.h:582
      int32_t numLoopSteps;
      // FmhaTask.h:592
      int32_t diffKvQ{(mSeqLenKv) - (mSeqLenQ)};
      // FmhaTask.h:597
      int32_t validSeqLenKv;
      // Common.h:63
      if ((params.mChunkedAttentionSizeLog2) > (int32_t{0})) {
        // FmhaTask.h:748
        mNumSkippedTilesKv =
          (((((mCtaIdxQ) * (int32_t{256})) + (diffKvQ)) >> (params.mChunkedAttentionSizeLog2))
           << (params.mChunkedAttentionSizeLog2)) /
          (int32_t{128});
      } else {
        // FmhaTask.h:767
        mNumSkippedTilesKv =
          (int32_t{max(int32_t{0},
                       ((((mCtaIdxQ) * (int32_t{256})) + (diffKvQ)) + (int32_t{1})) -
                         (params.mAttentionWindowSize))}) /
          (int32_t{128});
      }
      // FmhaTask.h:603
      validSeqLenKv =
        (int32_t{min((((mCtaIdxQ) * (int32_t{256})) + (diffKvQ)) + (int32_t{256}), mSeqLenKv)}) -
        ((mNumSkippedTilesKv) * (int32_t{128}));
      // FmhaTask.h:616
      mNumCtasKv =
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, int32_t{1})};
      // FmhaTask.h:630
      if ((((mCtaIdxQ) * (int32_t{256})) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
        // FmhaTask.h:668
        int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
                         ((mNumCtasKv) * (int32_t{128}))};
        // FmhaTask.h:682
        numLoopSteps = numSteps;
      } else {
        // FmhaTask.h:648
        numLoopSteps = int32_t{0};
      }
      // Task.cpp:3203
      bool const hasOneLoopIter{(int32_t{0}) < (numLoopSteps)};
      // Task.cpp:3214
      int32_t lastLoopOffset{int32_t{0}};
      // SmemKv.h:720
      int32_t const headIdxKv3{(mHeadIdx) / (params.mNumHeadsQPerKv)};
      //
      // Hoist the first iter.
      //
      //
      // gmemQ [ConsWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // GmemQkv.h:82
      int32_t idxQ00;
      // GmemQkv.h:83
      idxQ00 = (mCtaIdxQ) * (int32_t{2});
      // GmemQkv.h:91
      int32_t idxQ10;
      // GmemQkv.h:92
      idxQ10 = ((mCtaIdxQ) * (int32_t{2})) + (int32_t{1});
      // Task.cpp:1607
      if (hasOneLoopIter) {
      }
      //
      // gmemKv [ConsWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
      }
      //
      // smemPageOffsetsKv [ConsWait, FirstIter, FreqInfo{0, 16}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:1607
        if (hasOneLoopIter) {
          // Task.cpp:2780
          smemPageOffsetsKvConsToken =
            smemPageOffsetsKvSrcStack.mPipeline.consumer_try_wait(smemPageOffsetsKvConsState);
        }
        // Task.cpp:2848
        smemPageOffsetsKvSrcStack.mPipeline.consumer_wait(smemPageOffsetsKvConsState,
                                                          smemPageOffsetsKvConsToken);
      }
      //
      // smemPageOffsetsKv [ConsWork (call 0), FirstIter, FreqInfo{0, 16}, UserTags{1}, Flags{0}].
      //
      // SmemPageOffsetsKv.h:320
      int32_t* ptrSmemPageOffsetsK4;
      // SmemPageOffsetsKv.h:326
      int32_t* ptrSmemPageOffsetsV4;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{smemPageOffsetsKvConsState.index()};
        // SmemPageOffsetsKv.h:349
        ptrSmemPageOffsetsK4 = smemPageOffsetsKvSrcSmem.mArray[index];
        // Task.cpp:43
        ++smemPageOffsetsKvConsState;
      }
      //
      // smemKv [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{17}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5078
        {
          // Task.cpp:5100
          smemKvProdToken = smemKvDstStack.mPipeline.producer_try_acquire(smemKvProdState);
        }
      }
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4318
        smemKvDstStack.mPipeline.producer_acquire(smemKvProdState, smemKvProdToken);
      }
      //
      // smemKv [ProdWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{17}, Flags{0}].
      //
      // SmemKv.h:772
      int32_t* ptrSmemPageOffsetsK3;
      // SmemKv.h:786
      int32_t* ptrSmemPageOffsetsV3;
      // Task.cpp:1511
      ptrSmemPageOffsetsK3 = ptrSmemPageOffsetsK4;
      // Task.cpp:1511
      ptrSmemPageOffsetsV3 = ptrSmemPageOffsetsV4;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4413
        uint64_t* barrier{smemKvDstStack.mPipeline.producer_get_barrier(smemKvProdState)};
        // Task.cpp:5945
        int32_t index{smemKvProdState.index()};
        // Common.h:599
        if (((mSeqLenKv) - (mSeqLenQ)) < (int32_t{128})) {
          // Common.h:603
          cudaGridDependencySynchronize();
        }
        // SmemKv.h:1430
        int32_t headDimOffset{int32_t{0}};
        // SmemKv.h:1555
        int32_t tokenOffset{int32_t{0}};
        //
        // Load pageOffsets for headDimStageIdx = 0.
        //
        // SmemKv.h:1695
        cutlass::AlignedArray<int32_t, 2> localPageOffsets03;
        // SmemKv.h:1711
        localPageOffsets03 = reinterpret_cast<cutlass::AlignedArray<int32_t, 2>*>(
          (ptrSmemPageOffsetsK3 + int32_t{0}))[int32_t{0}];
        // SmemKv.h:1236
        {
          // SmemTile.cpp:485
          int32_t coords[4];
          // SmemTile.cpp:492
          coords[int32_t{0}] = headDimOffset;
          // SmemTile.cpp:492
          coords[int32_t{1}] = tokenOffset;
          // SmemTile.cpp:492
          coords[int32_t{2}] = headIdxKv3;
          // SmemTile.cpp:492
          coords[int32_t{3}] = localPageOffsets03[int32_t{0}];
          // SmemTile.cpp:611
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
        // SmemKv.h:1236
        {
          // SmemTile.cpp:485
          int32_t coords[4];
          // SmemTile.cpp:492
          coords[int32_t{0}] = headDimOffset;
          // SmemTile.cpp:492
          coords[int32_t{1}] = tokenOffset;
          // SmemTile.cpp:492
          coords[int32_t{2}] = headIdxKv3;
          // SmemTile.cpp:492
          coords[int32_t{3}] = localPageOffsets03[int32_t{1}];
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{4096}],
                                           &params.tmaK_,
                                           coords,
                                           barrier);
          }
        }
      }
      //
      // smemKv [ProdCommit, FirstIter, FreqInfo{0, 1}, UserTags{17}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4540
        {
          // Task.cpp:4556
          smemKvDstStack.mPipeline.producer_commit(smemKvProdState);
        }
        // Task.cpp:43
        ++smemKvProdState;
      }
      //
      // smemQ [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5078
        {
          // Task.cpp:5100
          smemQProdToken = smemQDstStack.mPipeline.producer_try_acquire(smemQProdState);
        }
      }
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4318
        smemQDstStack.mPipeline.producer_acquire(smemQProdState, smemQProdToken);
      }
      //
      // smemQ [ProdWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // SmemQ.h:201
      int32_t prodIdxQ02;
      // Task.cpp:1511
      prodIdxQ02 = idxQ00;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4413
        uint64_t* barrier{smemQDstStack.mPipeline.producer_get_barrier(smemQProdState)};
        // Task.cpp:5945
        int32_t index{smemQProdState.index()};
        // Common.h:603
        cudaGridDependencySynchronize();
        // SmemTile.cpp:485
        int32_t coords[4];
        // SmemTile.cpp:492
        coords[int32_t{0}] = int32_t{0};
        // SmemTile.cpp:492
        coords[int32_t{1}] = int32_t{0};
        // SmemTile.cpp:492
        coords[int32_t{2}] = mHeadIdx;
        // SmemTile.cpp:492
        coords[int32_t{3}] = (prodIdxQ02) * (int32_t{128}) + (mSeqOffsetQ);
        // SmemTile.cpp:611
        if (bool{cute::elect_one_sync()}) {
          // CudaPtx.h:48
          cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                         cuda_ptx::space_global_t{},
                                         &smemQDstSmem.mArray[index][int32_t{0}],
                                         &params.tmaQ_,
                                         coords,
                                         barrier);
        }
      }
      //
      // smemQ [ProdCommit, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4540
        {
          // Task.cpp:4556
          smemQDstStack.mPipeline.producer_commit(smemQProdState);
        }
        // Task.cpp:43
        ++smemQProdState;
      }
      //
      // smemPageOffsetsKv [ConsRelease, FirstIter, FreqInfo{0, 16}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:3814

      //
      // smemQ [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5078
        {
          // Task.cpp:5100
          smemQProdToken = smemQDstStack.mPipeline.producer_try_acquire(smemQProdState);
        }
      }
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4318
        smemQDstStack.mPipeline.producer_acquire(smemQProdState, smemQProdToken);
      }
      //
      // smemQ [ProdWork (call 1), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // SmemQ.h:205
      int32_t prodIdxQ12;
      // Task.cpp:1511
      prodIdxQ02 = idxQ00;
      // Task.cpp:1511
      prodIdxQ12 = idxQ10;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4413
        uint64_t* barrier{smemQDstStack.mPipeline.producer_get_barrier(smemQProdState)};
        // Task.cpp:5945
        int32_t index{smemQProdState.index()};
        // SmemTile.cpp:485
        int32_t coords[4];
        // SmemTile.cpp:492
        coords[int32_t{0}] = int32_t{0};
        // SmemTile.cpp:492
        coords[int32_t{1}] = int32_t{0};
        // SmemTile.cpp:492
        coords[int32_t{2}] = mHeadIdx;
        // SmemTile.cpp:492
        coords[int32_t{3}] = (prodIdxQ12) * (int32_t{128}) + (mSeqOffsetQ);
        // SmemTile.cpp:611
        if (bool{cute::elect_one_sync()}) {
          // CudaPtx.h:48
          cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                         cuda_ptx::space_global_t{},
                                         &smemQDstSmem.mArray[index][int32_t{0}],
                                         &params.tmaQ_,
                                         coords,
                                         barrier);
        }
      }
      //
      // smemQ [ProdCommit, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4540
        {
          // Task.cpp:4556
          smemQDstStack.mPipeline.producer_commit(smemQProdState);
        }
        // Task.cpp:43
        ++smemQProdState;
      }
      //
      // smemPageOffsetsKv [ConsWait, FirstIter, FreqInfo{0, 16}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:1607
        if (hasOneLoopIter) {
          // Task.cpp:2780
          smemPageOffsetsKvConsToken =
            smemPageOffsetsKvSrcStack.mPipeline.consumer_try_wait(smemPageOffsetsKvConsState);
        }
        // Task.cpp:2848
        smemPageOffsetsKvSrcStack.mPipeline.consumer_wait(smemPageOffsetsKvConsState,
                                                          smemPageOffsetsKvConsToken);
      }
      //
      // smemPageOffsetsKv [ConsWork (call 1), FirstIter, FreqInfo{0, 16}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{smemPageOffsetsKvConsState.index()};
        // SmemPageOffsetsKv.h:349
        ptrSmemPageOffsetsV4 = smemPageOffsetsKvSrcSmem.mArray[index];
        // Task.cpp:43
        ++smemPageOffsetsKvConsState;
      }
      //
      // smemKv [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5078
        {
          // Task.cpp:5100
          smemKvProdToken = smemKvDstStack.mPipeline.producer_try_acquire(smemKvProdState);
        }
      }
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4318
        smemKvDstStack.mPipeline.producer_acquire(smemKvProdState, smemKvProdToken);
      }
      //
      // smemKv [ProdWork (call 1), FirstIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1511
      ptrSmemPageOffsetsK3 = ptrSmemPageOffsetsK4;
      // Task.cpp:1511
      ptrSmemPageOffsetsV3 = ptrSmemPageOffsetsV4;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4413
        uint64_t* barrier{smemKvDstStack.mPipeline.producer_get_barrier(smemKvProdState)};
        // Task.cpp:5945
        int32_t index{smemKvProdState.index()};
        // SmemKv.h:1430
        int32_t headDimOffset{int32_t{0}};
        // SmemKv.h:1555
        int32_t tokenOffset{int32_t{0}};
        //
        // Load pageOffsets for headDimStageIdx = 0.
        //
        // SmemKv.h:1695
        cutlass::AlignedArray<int32_t, 2> localPageOffsets03;
        // SmemKv.h:1711
        localPageOffsets03 = reinterpret_cast<cutlass::AlignedArray<int32_t, 2>*>(
          (ptrSmemPageOffsetsV3 + int32_t{0}))[int32_t{0}];
        // SmemKv.h:1236
        {
          // SmemTile.cpp:485
          int32_t coords[4];
          // SmemTile.cpp:492
          coords[int32_t{0}] = headDimOffset;
          // SmemTile.cpp:492
          coords[int32_t{1}] = tokenOffset;
          // SmemTile.cpp:492
          coords[int32_t{2}] = headIdxKv3;
          // SmemTile.cpp:492
          coords[int32_t{3}] = localPageOffsets03[int32_t{0}];
          // SmemTile.cpp:611
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
        // SmemKv.h:1236
        {
          // SmemTile.cpp:485
          int32_t coords[4];
          // SmemTile.cpp:492
          coords[int32_t{0}] = headDimOffset;
          // SmemTile.cpp:492
          coords[int32_t{1}] = tokenOffset;
          // SmemTile.cpp:492
          coords[int32_t{2}] = headIdxKv3;
          // SmemTile.cpp:492
          coords[int32_t{3}] = localPageOffsets03[int32_t{1}];
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{4096}],
                                           &params.tmaV_,
                                           coords,
                                           barrier);
          }
        }
      }
      //
      // smemKv [ProdCommit, FirstIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4540
        {
          // Task.cpp:4556
          smemKvDstStack.mPipeline.producer_commit(smemKvProdState);
        }
        // Task.cpp:43
        ++smemKvProdState;
      }
      //
      // smemPageOffsetsKv [ConsRelease, FirstIter, FreqInfo{0, 16}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:3814

      //
      // Loop body.
      //
      // Task.cpp:3392
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset968 = int32_t{0}; loopOffset968 < (numLoopSteps) - (int32_t{1});
           ++loopOffset968) {
        // Task.cpp:3465
        bool const isLastLoopIter{((loopOffset968) + (int32_t{1})) >=
                                  ((numLoopSteps) - (int32_t{1}))};
        //
        // gmemKv [ConsWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:2928
        {}
        //
        // smemPageOffsetsKv [ConsWait, Info{0}, FreqInfo{1, 16}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:3814
        if ((((loopOffset968) + (int32_t{1})) % (int32_t{16})) == (int32_t{0})) {
          // Task.cpp:1607
          // Task.cpp:2816
          {
            // Task.cpp:1607
            // Task.cpp:2757
            {
              // Task.cpp:2780
              smemPageOffsetsKvConsToken =
                smemPageOffsetsKvSrcStack.mPipeline.consumer_try_wait(smemPageOffsetsKvConsState);
            }
            // Task.cpp:2848
            smemPageOffsetsKvSrcStack.mPipeline.consumer_wait(smemPageOffsetsKvConsState,
                                                              smemPageOffsetsKvConsToken);
          }
        }
        //
        // smemPageOffsetsKv [ConsWork (call 2), Info{0}, FreqInfo{1, 16}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:3814
        if ((((loopOffset968) + (int32_t{1})) % (int32_t{16})) == (int32_t{0})) {
          // Task.cpp:1607
          // Task.cpp:2928
          {
            // Task.cpp:5945
            int32_t index{smemPageOffsetsKvConsState.index()};
            // SmemPageOffsetsKv.h:349
            ptrSmemPageOffsetsK4 = smemPageOffsetsKvSrcSmem.mArray[index];
            // Task.cpp:43
            ++smemPageOffsetsKvConsState;
          }
        }
        //
        // smemKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset968) >= (int32_t{0})) {
            // Task.cpp:5100
            smemKvProdToken = smemKvDstStack.mPipeline.producer_try_acquire(smemKvProdState);
          }
        }
        // Task.cpp:1607
        // Task.cpp:4288
        {
          // Task.cpp:4318
          smemKvDstStack.mPipeline.producer_acquire(smemKvProdState, smemKvProdToken);
        }
        //
        // smemKv [ProdWork (call 2), Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{0}].
        //
        // Task.cpp:1511
        ptrSmemPageOffsetsK3 = ptrSmemPageOffsetsK4;
        // Task.cpp:1511
        ptrSmemPageOffsetsV3 = ptrSmemPageOffsetsV4;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:4413
          uint64_t* barrier{smemKvDstStack.mPipeline.producer_get_barrier(smemKvProdState)};
          // Task.cpp:5945
          int32_t index{smemKvProdState.index()};
          // SmemKv.h:1430
          int32_t headDimOffset{int32_t{0}};
          // SmemKv.h:1555
          int32_t tokenOffset{int32_t{0}};
          //
          // Load pageOffsets for headDimStageIdx = 0.
          //
          // SmemKv.h:1695
          cutlass::AlignedArray<int32_t, 2> localPageOffsets03;
          // SmemKv.h:1711
          localPageOffsets03 = reinterpret_cast<cutlass::AlignedArray<int32_t, 2>*>(
            (ptrSmemPageOffsetsK3 +
             (((loopOffset968) + (int32_t{1})) * (int32_t{2})) % (int32_t{32})))[int32_t{0}];
          // SmemKv.h:1236
          {
            // SmemTile.cpp:485
            int32_t coords[4];
            // SmemTile.cpp:492
            coords[int32_t{0}] = headDimOffset;
            // SmemTile.cpp:492
            coords[int32_t{1}] = tokenOffset;
            // SmemTile.cpp:492
            coords[int32_t{2}] = headIdxKv3;
            // SmemTile.cpp:492
            coords[int32_t{3}] = localPageOffsets03[int32_t{0}];
            // SmemTile.cpp:611
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
          // SmemKv.h:1236
          {
            // SmemTile.cpp:485
            int32_t coords[4];
            // SmemTile.cpp:492
            coords[int32_t{0}] = headDimOffset;
            // SmemTile.cpp:492
            coords[int32_t{1}] = tokenOffset;
            // SmemTile.cpp:492
            coords[int32_t{2}] = headIdxKv3;
            // SmemTile.cpp:492
            coords[int32_t{3}] = localPageOffsets03[int32_t{1}];
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{4096}],
                                             &params.tmaK_,
                                             coords,
                                             barrier);
            }
          }
        }
        //
        // smemKv [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{0}].
        //
        // Task.cpp:4522
        {
          // Task.cpp:4540
          {
            // Task.cpp:4556
            smemKvDstStack.mPipeline.producer_commit(smemKvProdState);
          }
          // Task.cpp:43
          ++smemKvProdState;
        }
        //
        // smemPageOffsetsKv [ConsRelease, Info{0}, FreqInfo{1, 16}, UserTags{1}, Flags{65536}].
        //
        // Task.cpp:3814
        if ((!(isLastLoopIter)) &&
            ((((loopOffset968) + (int32_t{1})) % (int32_t{16})) == (int32_t{15}))) {
          // Task.cpp:2568
          if ((loopOffset968) >= (int32_t{0})) {
            // Task.cpp:2596
            {
              // Task.cpp:2620
              smemPageOffsetsKvSrcStack.mPipeline.consumer_release(
                smemPageOffsetsKvConsReleaseState);
            }
            // Task.cpp:43
            ++smemPageOffsetsKvConsReleaseState;
          }
        }
        //
        // smemPageOffsetsKv [ConsWait, Info{0}, FreqInfo{1, 16}, UserTags{2}, Flags{0}].
        //
        // Task.cpp:3814
        if ((((loopOffset968) + (int32_t{1})) % (int32_t{16})) == (int32_t{0})) {
          // Task.cpp:1607
          // Task.cpp:2816
          {
            // Task.cpp:1607
            // Task.cpp:2757
            {
              // Task.cpp:2780
              smemPageOffsetsKvConsToken =
                smemPageOffsetsKvSrcStack.mPipeline.consumer_try_wait(smemPageOffsetsKvConsState);
            }
            // Task.cpp:2848
            smemPageOffsetsKvSrcStack.mPipeline.consumer_wait(smemPageOffsetsKvConsState,
                                                              smemPageOffsetsKvConsToken);
          }
        }
        //
        // smemPageOffsetsKv [ConsWork (call 3), Info{0}, FreqInfo{1, 16}, UserTags{2}, Flags{0}].
        //
        // Task.cpp:3814
        if ((((loopOffset968) + (int32_t{1})) % (int32_t{16})) == (int32_t{0})) {
          // Task.cpp:1607
          // Task.cpp:2928
          {
            // Task.cpp:5945
            int32_t index{smemPageOffsetsKvConsState.index()};
            // SmemPageOffsetsKv.h:349
            ptrSmemPageOffsetsV4 = smemPageOffsetsKvSrcSmem.mArray[index];
            // Task.cpp:43
            ++smemPageOffsetsKvConsState;
          }
        }
        //
        // smemKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{12}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset968) >= (int32_t{0})) {
            // Task.cpp:5100
            smemKvProdToken = smemKvDstStack.mPipeline.producer_try_acquire(smemKvProdState);
          }
        }
        // Task.cpp:1607
        // Task.cpp:4288
        {
          // Task.cpp:4318
          smemKvDstStack.mPipeline.producer_acquire(smemKvProdState, smemKvProdToken);
        }
        //
        // smemKv [ProdWork (call 3), Info{0}, FreqInfo{0, 1}, UserTags{12}, Flags{0}].
        //
        // Task.cpp:1511
        ptrSmemPageOffsetsK3 = ptrSmemPageOffsetsK4;
        // Task.cpp:1511
        ptrSmemPageOffsetsV3 = ptrSmemPageOffsetsV4;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:4413
          uint64_t* barrier{smemKvDstStack.mPipeline.producer_get_barrier(smemKvProdState)};
          // Task.cpp:5945
          int32_t index{smemKvProdState.index()};
          // SmemKv.h:1430
          int32_t headDimOffset{int32_t{0}};
          // SmemKv.h:1555
          int32_t tokenOffset{int32_t{0}};
          //
          // Load pageOffsets for headDimStageIdx = 0.
          //
          // SmemKv.h:1695
          cutlass::AlignedArray<int32_t, 2> localPageOffsets03;
          // SmemKv.h:1711
          localPageOffsets03 = reinterpret_cast<cutlass::AlignedArray<int32_t, 2>*>(
            (ptrSmemPageOffsetsV3 +
             (((loopOffset968) + (int32_t{1})) * (int32_t{2})) % (int32_t{32})))[int32_t{0}];
          // SmemKv.h:1236
          {
            // SmemTile.cpp:485
            int32_t coords[4];
            // SmemTile.cpp:492
            coords[int32_t{0}] = headDimOffset;
            // SmemTile.cpp:492
            coords[int32_t{1}] = tokenOffset;
            // SmemTile.cpp:492
            coords[int32_t{2}] = headIdxKv3;
            // SmemTile.cpp:492
            coords[int32_t{3}] = localPageOffsets03[int32_t{0}];
            // SmemTile.cpp:611
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
          // SmemKv.h:1236
          {
            // SmemTile.cpp:485
            int32_t coords[4];
            // SmemTile.cpp:492
            coords[int32_t{0}] = headDimOffset;
            // SmemTile.cpp:492
            coords[int32_t{1}] = tokenOffset;
            // SmemTile.cpp:492
            coords[int32_t{2}] = headIdxKv3;
            // SmemTile.cpp:492
            coords[int32_t{3}] = localPageOffsets03[int32_t{1}];
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{4096}],
                                             &params.tmaV_,
                                             coords,
                                             barrier);
            }
          }
        }
        //
        // smemKv [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{12}, Flags{0}].
        //
        // Task.cpp:4522
        {
          // Task.cpp:4540
          {
            // Task.cpp:4556
            smemKvDstStack.mPipeline.producer_commit(smemKvProdState);
          }
          // Task.cpp:43
          ++smemKvProdState;
        }
        //
        // smemPageOffsetsKv [ConsRelease, Info{0}, FreqInfo{1, 16}, UserTags{2}, Flags{65536}].
        //
        // Task.cpp:3814
        if ((!(isLastLoopIter)) &&
            ((((loopOffset968) + (int32_t{1})) % (int32_t{16})) == (int32_t{15}))) {
          // Task.cpp:2568
          if ((loopOffset968) >= (int32_t{0})) {
            // Task.cpp:2596
            {
              // Task.cpp:2620
              smemPageOffsetsKvSrcStack.mPipeline.consumer_release(
                smemPageOffsetsKvConsReleaseState);
            }
            // Task.cpp:43
            ++smemPageOffsetsKvConsReleaseState;
          }
        }
        // Task.cpp:3499
        lastLoopOffset = loopOffset968;
      }
      //
      // Pull the last iter down.
      //
      // Task.cpp:3534
      if (((numLoopSteps) - (int32_t{1})) > (int32_t{0})) {
        // Task.cpp:3535
        ++lastLoopOffset;
      }
      //
      // workIdThrottleBarrier [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{4608}].
      //
      // Task.cpp:1607
      // Task.cpp:5064
      {
        // Task.cpp:5078
        if ((lastLoopOffset) >= (int32_t{0})) {
          // Task.cpp:5100
          workIdThrottleBarrierProdToken =
            workIdThrottleBarrierDstStack.mPipeline.producer_try_acquire(
              workIdThrottleBarrierProdState);
        }
      }
      // Task.cpp:1607
      // Task.cpp:4288
      {
        // Task.cpp:4318
        workIdThrottleBarrierDstStack.mPipeline.producer_acquire(workIdThrottleBarrierProdState,
                                                                 workIdThrottleBarrierProdToken);
      }
      //
      // workIdThrottleBarrier [ProdWork (call 0), LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{4608}].
      //
      //
      // workIdThrottleBarrier [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{4608}].
      //
      // Task.cpp:1607
      // Task.cpp:4507
      {
        // Task.cpp:4540
        {
          // Task.cpp:4556
          workIdThrottleBarrierDstStack.mPipeline.producer_commit(workIdThrottleBarrierProdState);
        }
        // Task.cpp:43
        ++workIdThrottleBarrierProdState;
      }
      //
      // smemPageOffsetsKv [ConsRelease, LastIter, FreqInfo{0, 16}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:2596
        {
          // Task.cpp:2620
          smemPageOffsetsKvSrcStack.mPipeline.consumer_release(smemPageOffsetsKvConsReleaseState);
        }
        // Task.cpp:43
        ++smemPageOffsetsKvConsReleaseState;
      }
      //
      // smemPageOffsetsKv [ConsRelease, LastIter, FreqInfo{0, 16}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:2596
        {
          // Task.cpp:2620
          smemPageOffsetsKvSrcStack.mPipeline.consumer_release(smemPageOffsetsKvConsReleaseState);
        }
        // Task.cpp:43
        ++smemPageOffsetsKvConsReleaseState;
      }
    //
    // Tail work.
    //
    // Task.cpp:3553
    ExitTileWithSignalingLabel:
    // Task.cpp:3560
    ExitTileWithoutSignalingLabel:
      // Task.cpp:5532
      auto newWorkTileInfoTuple{
        workIdStorageSrcStack.mScheduler.fetch_next_work(workIdStorageSrcStack.workTileInfo,
                                                         workIdStorageSrcStack.mPipeline,
                                                         workIdStorageConsState)};
      // Task.cpp:5534
      workIdStorageSrcStack.workTileInfo = cute::get<int32_t{0}>(newWorkTileInfoTuple);
      // Task.cpp:5542
      ++workIdStorageConsState;
      // Task.cpp:5644
      mCtaIdxX = workIdStorageSrcStack.workTileInfo.M_idx;
      // Task.cpp:5645
      mCtaIdxY = workIdStorageSrcStack.workTileInfo.N_idx;
      // Task.cpp:5646
      mCtaIdxZ = workIdStorageSrcStack.workTileInfo.L_idx;
    } while (workIdStorageSrcStack.workTileInfo.is_valid_tile);
  }
};
// Task.cpp:559
// Fmha.h:2221
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
  // FmhaTask.h:216
  int32_t mNumCtasKv;
  // FmhaTask.h:226
  int32_t mCtaIdxKv;
  // FmhaTask.h:224
  int32_t mCtaIdxQ;
  // FmhaTask.h:214
  int32_t mSeqLenKv;
  // FmhaTask.h:733
  int32_t mNumSkippedTilesKv;
  // Task.cpp:371
  int32_t const mWarpGrpThreadIdx;
  // Task.cpp:706
  uint32_t const mTmemBaseOffset;
  // Task.cpp:394
  int32_t const mWarpGrpWarpIdx;
  // Task.cpp:566
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
    , // Kernel.cpp:212
    mNumCtasKv{int32_t{1}}
    , // Kernel.cpp:210
    mCtaIdxKv{int32_t{0}}
    , // Task.cpp:283
    mCtaIdxQ{mCtaIdxX}
    , // FmhaTask.h:437
    mSeqLenKv{int32_t{
      params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                            ? ((((mHeadIdx) / (params.mNumHeadsQPerKv)) * (params.mBatchSize)) +
                               (mBatchIdx))
                            : (mBatchIdx)]}}
    , // Kernel.cpp:210
    mNumSkippedTilesKv{int32_t{0}}
    , // Task.cpp:379
    mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))}
    , // Kernel.cpp:2424
    mTmemBaseOffset{uint32_t{
      __shfl_sync(uint32_t{0xffffffff}, (*state.mTmemSwStatePtr), int32_t{0}, int32_t{32})}}
    , // Task.cpp:395
    mWarpGrpWarpIdx{(state.mWarpIdx) - (warpGrpStart)} {}
  // Task.cpp:522
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:547
    return ((state.mWarpIdx) >= (int32_t{0})) && ((state.mWarpIdx) < (int32_t{4}));
  }
  // Task.cpp:454
  inline __device__ void execute(fmha::KernelParams const& params,
                                 KernelState const& state,
                                 TmemSoftmaxLocal0Stack& tmemSoftmaxLocal0DstStack,
                                 TmemSoftmaxGlobal0Stack& tmemSoftmaxGlobal0DstStack,
                                 TmemP0Stack& tmemP0DstStack,
                                 TmemS0Stack& tmemS0SrcStack,
                                 WorkIdStorageSmem& workIdStorageSrcSmem,
                                 WorkIdStorageStack& workIdStorageSrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_inc(cuda_ptx::n32_t<184>{});
    // Task.cpp:2114
    trtllm::dev::CutlassUmmaAsyncPipeline<
      1,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState tmemS0ConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassUmmaAsyncPipeline<1,
                                          cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState tmemS0ConsReleaseState{};
    // Task.cpp:2135
    int32_t tmemS0ConsToken{int32_t{0}};
    // Task.cpp:2114
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsReleaseState{};
    // Task.cpp:2135
    int32_t workIdStorageConsToken{int32_t{0}};
    // Task.cpp:2013
    trtllm::dev::CutlassCpAsyncPipeline<1, true>::PipelineState tmemSoftmaxLocal0ProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    // Task.cpp:2033
    int32_t tmemSoftmaxLocal0ProdToken{int32_t{1}};
    // Task.cpp:2013
    trtllm::dev::CutlassCpAsyncPipeline<1, true>::PipelineState tmemP0ProdState{int32_t{0},
                                                                                int32_t{1},
                                                                                int32_t{0}};
    // Task.cpp:2033
    int32_t tmemP0ProdToken{int32_t{1}};
    // Task.cpp:5485
    do {
      // FmhaTask.h:333
      int32_t currSeqCtaIdx;
      // FmhaTask.h:342
      currSeqCtaIdx = workIdStorageSrcStack.workTileInfo.M_idx;
      // FmhaTask.h:357
      mHeadIdx = workIdStorageSrcStack.workTileInfo.N_idx;
      // FmhaTask.h:361
      mBatchIdx = workIdStorageSrcStack.workTileInfo.L_idx;
      // FmhaTask.h:139
      mSeqOffsetQ = int32_t(bool{params.ptrCumSeqLensQ == nullptr})
                      ? ((mBatchIdx) * (params.mMaxSeqLenQ))
                      : (int32_t{params.ptrCumSeqLensQ[mBatchIdx]});
      // FmhaTask.h:139
      mSeqLenQ = int32_t(bool{params.ptrCumSeqLensQ == nullptr})
                   ? (params.mMaxSeqLenQ)
                   : ((int32_t{params.ptrCumSeqLensQ[(mBatchIdx) + (int32_t{1})]}) - (mSeqOffsetQ));
      // FmhaTask.h:491
      mCtaIdxQ = currSeqCtaIdx;
      // FmhaTask.h:139
      mSeqLenKv = int32_t{
        params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                              ? ((((mHeadIdx) / (params.mNumHeadsQPerKv)) * (params.mBatchSize)) +
                                 (mBatchIdx))
                              : (mBatchIdx)]};
      // FmhaTask.h:582
      int32_t numLoopSteps;
      // FmhaTask.h:592
      int32_t diffKvQ{(mSeqLenKv) - (mSeqLenQ)};
      // FmhaTask.h:597
      int32_t validSeqLenKv;
      // Common.h:63
      if ((params.mChunkedAttentionSizeLog2) > (int32_t{0})) {
        // FmhaTask.h:748
        mNumSkippedTilesKv =
          (((((mCtaIdxQ) * (int32_t{256})) + (diffKvQ)) >> (params.mChunkedAttentionSizeLog2))
           << (params.mChunkedAttentionSizeLog2)) /
          (int32_t{128});
      } else {
        // FmhaTask.h:767
        mNumSkippedTilesKv =
          (int32_t{max(int32_t{0},
                       ((((mCtaIdxQ) * (int32_t{256})) + (diffKvQ)) + (int32_t{1})) -
                         (params.mAttentionWindowSize))}) /
          (int32_t{128});
      }
      // FmhaTask.h:603
      validSeqLenKv =
        (int32_t{min((((mCtaIdxQ) * (int32_t{256})) + (diffKvQ)) + (int32_t{256}), mSeqLenKv)}) -
        ((mNumSkippedTilesKv) * (int32_t{128}));
      // FmhaTask.h:616
      mNumCtasKv =
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, int32_t{1})};
      // FmhaTask.h:630
      if ((((mCtaIdxQ) * (int32_t{256})) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
        // FmhaTask.h:668
        int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
                         ((mNumCtasKv) * (int32_t{128}))};
        // FmhaTask.h:682
        numLoopSteps = numSteps;
      } else {
        // FmhaTask.h:648
        numLoopSteps = int32_t{0};
      }
      // Task.cpp:3203
      bool const hasOneLoopIter{(int32_t{0}) < (numLoopSteps)};
      // TmemS.h:654
      float oldMaxArray7[1];
      // TmemS.h:660
      float sumArray7[1]{float{0}};
      // TmemS.h:672
      float newMaxArray7[1]{float{-3.4028235e+38}};
      // TmemTile.cpp:373
      cutlass::Array<float, 128> regsQk;
      // TmemSoftmax.h:515
      cudaGridDependencySynchronize();
      // TmemSoftmax.h:524
      float scaleSoftmaxLog29;
      // TmemSoftmax.h:529
      scaleSoftmaxLog29 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
                            ? (params.mScaleSoftmaxLog2)
                            : (float{params.ptrScaleSoftmaxLog2[int32_t{0}]});
      // TmemTile.cpp:373
      cutlass::Array<uint32_t, 32> regsP;
      // TmemP.h:534
      cudaGridDependencySynchronize();
      // TmemP.h:541
      float scaleSoftmaxLog214;
      // TmemP.h:546
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
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5078
        {
          // Task.cpp:5100
          tmemSoftmaxLocal0ProdToken =
            tmemSoftmaxLocal0DstStack.mPipeline.producer_try_acquire(tmemSoftmaxLocal0ProdState);
        }
      }
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4318
        tmemSoftmaxLocal0DstStack.mPipeline.producer_acquire(tmemSoftmaxLocal0ProdState,
                                                             tmemSoftmaxLocal0ProdToken);
      }
      //
      // Loop body.
      //
      // Task.cpp:3392
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset1267 = int32_t{0}; loopOffset1267 < numLoopSteps; ++loopOffset1267) {
        // Task.cpp:3465
        bool const isLastLoopIter{((loopOffset1267) + (int32_t{1})) >= (numLoopSteps)};
        //
        // tmemS0 [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{8}, Flags{256}].
        //
        // Task.cpp:1607
        // Task.cpp:2816
        {
          // Task.cpp:1607
          // Task.cpp:2757
          {
            // Task.cpp:2780
            tmemS0ConsToken = tmemS0SrcStack.mPipeline.consumer_try_wait(tmemS0ConsState);
          }
          // Task.cpp:2848
          tmemS0SrcStack.mPipeline.consumer_wait(tmemS0ConsState, tmemS0ConsToken);
        }
        //
        // tmemS0 [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{8}, Flags{256}].
        //
        // TmemS.h:1060
        float* oldMaxPtr7;
        // TmemS.h:1065
        float* sumPtr7;
        // TmemS.h:1070
        float* newMaxPtr07;
        // TmemS.h:1075
        float* qkPtr07;
        // TmemS.h:1080
        float* newMaxPtr17;
        // TmemS.h:1085
        float* qkPtr17;
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{tmemS0ConsState.index()};
          // TmemS.h:1192
          oldMaxPtr7 = oldMaxArray7;
          // TmemS.h:1194
          sumPtr7 = sumArray7;
          // TmemS.h:1196
          newMaxPtr07 = newMaxArray7;
          // TmemS.h:1198
          newMaxPtr17 = newMaxArray7;
          // TmemS.h:1246
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset1288 = int32_t{0}; loopOffset1288 < int32_t{1}; ++loopOffset1288) {
            // TmemS.h:1257
            oldMaxArray7[loopOffset1288] = newMaxArray7[loopOffset1288];
          }
          // TmemS.h:1275
          float ilpMax0{newMaxArray7[int32_t{0}]};
          // TmemS.h:1275
          float ilpMax1{newMaxArray7[int32_t{0}]};
          // TmemS.h:1275
          float ilpMax2{newMaxArray7[int32_t{0}]};
          // TmemS.h:1275
          float ilpMax3{newMaxArray7[int32_t{0}]};
          //
          // The causal mask block.
          //
          // Mask.h:568
          int32_t const tileOffsetK{
            ((((numLoopSteps) * (mCtaIdxKv) + (loopOffset1267)) * (int32_t{1})) +
             (mNumSkippedTilesKv)) *
            (int32_t{128})};
          // Mask.h:1925
          bool isMaskSkipped{((tileOffsetK) + (int32_t{128})) <=
                             (((mCtaIdxQ) * (int32_t{256})) + ((mSeqLenKv) - (mSeqLenQ)))};
          // Mask.h:598
          bool isMaskSkippedBeginning;
          // Common.h:63
          if ((params.mChunkedAttentionSizeLog2) > (int32_t{0})) {
            // Mask.h:635
            isMaskSkippedBeginning =
              (((((mCtaIdxQ) * (int32_t{256})) + ((mSeqLenKv) - (mSeqLenQ))) + (int32_t{255})) >>
               (params.mChunkedAttentionSizeLog2)) ==
              ((((mCtaIdxQ) * (int32_t{256})) + ((mSeqLenKv) - (mSeqLenQ))) >>
               (params.mChunkedAttentionSizeLog2));
          } else {
            // Mask.h:648
            int32_t numBeginningMaskLoopSteps;
            // Mask.h:682
            if (((int32_t{max(
                   ((((mCtaIdxQ) * (int32_t{256})) + ((mSeqLenKv) - (mSeqLenQ))) + (int32_t{1})) -
                     (params.mAttentionWindowSize),
                   int32_t{0})}) /
                 (int32_t{128})) !=
                ((int32_t{max(
                   ((((mCtaIdxQ) * (int32_t{256})) + ((mSeqLenKv) - (mSeqLenQ))) + (int32_t{256})) -
                     (params.mAttentionWindowSize),
                   int32_t{0})}) /
                 (int32_t{128}))) {
              // Mask.h:686
              numBeginningMaskLoopSteps = int32_t{3};
            } else {
              // Mask.h:691
              numBeginningMaskLoopSteps = int32_t{2};
            }
            // Mask.h:698
            isMaskSkippedBeginning =
              ((numLoopSteps) * (mCtaIdxKv) + (loopOffset1267)) >= (numBeginningMaskLoopSteps);
          }
          // Mask.h:1936
          if ((isMaskSkipped) && (isMaskSkippedBeginning)) {
            // TmemTile.cpp:527
            {
              // TmemTile.cpp:529
              uint32_t tmemBasePtr{mTmemBaseOffset};
              // TmemTile.cpp:545
              uint32_t(&dstSlice0)[32]{reinterpret_cast<uint32_t(&)[32]>(regsQk[int32_t{0}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_32x32b(
                dstSlice0,
                (tmemBasePtr) +
                  (static_cast<uint32_t>(int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                           ? (int32_t{0})
                                           : (int32_t{128}))));
              // TmemTile.cpp:545
              uint32_t(&dstSlice1)[32]{reinterpret_cast<uint32_t(&)[32]>(regsQk[int32_t{32}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_32x32b(
                dstSlice1,
                (tmemBasePtr) +
                  (static_cast<uint32_t>((int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                            ? (int32_t{0})
                                            : (int32_t{128})) +
                                         (int32_t{0x20 /*hi=0, lo=32*/}))));
              // TmemTile.cpp:545
              uint32_t(&dstSlice2)[32]{reinterpret_cast<uint32_t(&)[32]>(regsQk[int32_t{64}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_32x32b(
                dstSlice2,
                (tmemBasePtr) +
                  (static_cast<uint32_t>((int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                            ? (int32_t{0})
                                            : (int32_t{128})) +
                                         (int32_t{0x40 /*hi=0, lo=64*/}))));
              // TmemTile.cpp:545
              uint32_t(&dstSlice3)[32]{reinterpret_cast<uint32_t(&)[32]>(regsQk[int32_t{96}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_32x32b(
                dstSlice3,
                (tmemBasePtr) +
                  (static_cast<uint32_t>((int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                            ? (int32_t{0})
                                            : (int32_t{128})) +
                                         (int32_t{0x60 /*hi=0, lo=96*/}))));
            }
            // TmemS.h:1681
            CUTLASS_PRAGMA_UNROLL
            for (int32_t loopOffset1323 = int32_t{0}; loopOffset1323 < int32_t{128};
                 loopOffset1323 += int32_t{4}) {
              // TmemS.h:1694
              ilpMax0 = fmaxf(ilpMax0, regsQk[loopOffset1323]);
              // TmemS.h:1694
              ilpMax1 = fmaxf(ilpMax1, regsQk[(loopOffset1323) + (int32_t{1})]);
              // TmemS.h:1694
              ilpMax2 = fmaxf(ilpMax2, regsQk[(loopOffset1323) + (int32_t{2})]);
              // TmemS.h:1694
              ilpMax3 = fmaxf(ilpMax3, regsQk[(loopOffset1323) + (int32_t{3})]);
            }
          } else {
            // TmemTile.cpp:527
            {
              // TmemTile.cpp:529
              uint32_t tmemBasePtr{mTmemBaseOffset};
              // TmemTile.cpp:545
              uint32_t(&dstSlice0)[32]{reinterpret_cast<uint32_t(&)[32]>(regsQk[int32_t{0}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_32x32b(
                dstSlice0,
                (tmemBasePtr) +
                  (static_cast<uint32_t>(int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                           ? (int32_t{0})
                                           : (int32_t{128}))));
              // TmemTile.cpp:545
              uint32_t(&dstSlice1)[32]{reinterpret_cast<uint32_t(&)[32]>(regsQk[int32_t{32}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_32x32b(
                dstSlice1,
                (tmemBasePtr) +
                  (static_cast<uint32_t>((int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                            ? (int32_t{0})
                                            : (int32_t{128})) +
                                         (int32_t{0x20 /*hi=0, lo=32*/}))));
              // TmemTile.cpp:545
              uint32_t(&dstSlice2)[32]{reinterpret_cast<uint32_t(&)[32]>(regsQk[int32_t{64}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_32x32b(
                dstSlice2,
                (tmemBasePtr) +
                  (static_cast<uint32_t>((int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                            ? (int32_t{0})
                                            : (int32_t{128})) +
                                         (int32_t{0x40 /*hi=0, lo=64*/}))));
              // TmemTile.cpp:545
              uint32_t(&dstSlice3)[32]{reinterpret_cast<uint32_t(&)[32]>(regsQk[int32_t{96}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_32x32b(
                dstSlice3,
                (tmemBasePtr) +
                  (static_cast<uint32_t>((int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                            ? (int32_t{0})
                                            : (int32_t{128})) +
                                         (int32_t{0x60 /*hi=0, lo=96*/}))));
            }
            //
            // Apply the causal mask.
            //
            // Mask.h:962
            int32_t const tileOffsetQ{
              (mCtaIdxQ) * (int32_t{256}) +
              ((tmemS0SrcStack.mInstId) * (int32_t{128}) + ((mSeqLenKv) - (mSeqLenQ)))};
            // Mask.h:568
            int32_t const tileOffsetK{
              ((((numLoopSteps) * (mCtaIdxKv) + (loopOffset1267)) * (int32_t{1})) +
               (mNumSkippedTilesKv)) *
              (int32_t{128})};
            // Mask.h:1006
            int32_t const tokenIdxQ{(tileOffsetQ) + (mWarpGrpThreadIdx)};
            //
            // Mask the elements outside the attention window/chunk.
            //
            // Mask.h:1024
            int32_t startIdxInWindow;
            // Mask.h:1026
            int32_t startUniformIdxInWindow;
            // Common.h:63
            if ((params.mChunkedAttentionSizeLog2) > (int32_t{0})) {
              // Mask.h:1068
              startIdxInWindow = ((tokenIdxQ) >> (params.mChunkedAttentionSizeLog2))
                                 << (params.mChunkedAttentionSizeLog2);
              // Mask.h:1069
              startUniformIdxInWindow =
                ((((tileOffsetQ) + ((mWarpGrpWarpIdx) * (int32_t{32}))) + (int32_t{31})) >>
                 (params.mChunkedAttentionSizeLog2))
                << (params.mChunkedAttentionSizeLog2);
            } else {
              // Mask.h:1049
              startIdxInWindow = ((tokenIdxQ) + (int32_t{1})) - (params.mAttentionWindowSize);
              // Mask.h:1050
              startUniformIdxInWindow =
                ((((tileOffsetQ) + ((mWarpGrpWarpIdx) * (int32_t{32}))) + (int32_t{31})) +
                 (int32_t{1})) -
                (params.mAttentionWindowSize);
            }
            // Mask.h:1075
            {
              // Mask.h:185
              int32_t uniformDist{(startUniformIdxInWindow) - (tileOffsetK)};
              // Mask.h:187
              int32_t clampedUniformDist{trtllm::dev::clamp(uniformDist, int32_t{0}, int32_t{128})};
              // Mask.h:207
              int32_t clampedDist{
                trtllm::dev::clamp((startIdxInWindow) - (tileOffsetK), int32_t{0}, int32_t{128})};
              // Mask.h:1078
              asm volatile("{\n"
                           ".reg .pred p<128>;\n"
                           "FirstToMask: .branchtargets R<129>;\n"
                           "brx.idx.uni %128, FirstToMask;\n"
                           "R0:\n"
                           "R1:\n"
                           "R2:\n"
                           "R3:\n"
                           "R4:\n"
                           "R5:\n"
                           "R6:\n"
                           "R7:\n"
                           "setp.le.s32 p0, %130, 0;\n"
                           "@p0 mov.b32 %0, %129;\n"
                           "setp.le.s32 p1, %130, 1;\n"
                           "@p1 mov.b32 %1, %129;\n"
                           "setp.le.s32 p2, %130, 2;\n"
                           "@p2 mov.b32 %2, %129;\n"
                           "setp.le.s32 p3, %130, 3;\n"
                           "@p3 mov.b32 %3, %129;\n"
                           "setp.le.s32 p4, %130, 4;\n"
                           "@p4 mov.b32 %4, %129;\n"
                           "setp.le.s32 p5, %130, 5;\n"
                           "@p5 mov.b32 %5, %129;\n"
                           "setp.le.s32 p6, %130, 6;\n"
                           "@p6 mov.b32 %6, %129;\n"
                           "setp.le.s32 p7, %130, 7;\n"
                           "@p7 mov.b32 %7, %129;\n"
                           "R8:\n"
                           "R9:\n"
                           "R10:\n"
                           "R11:\n"
                           "R12:\n"
                           "R13:\n"
                           "R14:\n"
                           "R15:\n"
                           "setp.le.s32 p8, %130, 8;\n"
                           "@p8 mov.b32 %8, %129;\n"
                           "setp.le.s32 p9, %130, 9;\n"
                           "@p9 mov.b32 %9, %129;\n"
                           "setp.le.s32 p10, %130, 10;\n"
                           "@p10 mov.b32 %10, %129;\n"
                           "setp.le.s32 p11, %130, 11;\n"
                           "@p11 mov.b32 %11, %129;\n"
                           "setp.le.s32 p12, %130, 12;\n"
                           "@p12 mov.b32 %12, %129;\n"
                           "setp.le.s32 p13, %130, 13;\n"
                           "@p13 mov.b32 %13, %129;\n"
                           "setp.le.s32 p14, %130, 14;\n"
                           "@p14 mov.b32 %14, %129;\n"
                           "setp.le.s32 p15, %130, 15;\n"
                           "@p15 mov.b32 %15, %129;\n"
                           "R16:\n"
                           "R17:\n"
                           "R18:\n"
                           "R19:\n"
                           "R20:\n"
                           "R21:\n"
                           "R22:\n"
                           "R23:\n"
                           "setp.le.s32 p16, %130, 16;\n"
                           "@p16 mov.b32 %16, %129;\n"
                           "setp.le.s32 p17, %130, 17;\n"
                           "@p17 mov.b32 %17, %129;\n"
                           "setp.le.s32 p18, %130, 18;\n"
                           "@p18 mov.b32 %18, %129;\n"
                           "setp.le.s32 p19, %130, 19;\n"
                           "@p19 mov.b32 %19, %129;\n"
                           "setp.le.s32 p20, %130, 20;\n"
                           "@p20 mov.b32 %20, %129;\n"
                           "setp.le.s32 p21, %130, 21;\n"
                           "@p21 mov.b32 %21, %129;\n"
                           "setp.le.s32 p22, %130, 22;\n"
                           "@p22 mov.b32 %22, %129;\n"
                           "setp.le.s32 p23, %130, 23;\n"
                           "@p23 mov.b32 %23, %129;\n"
                           "R24:\n"
                           "R25:\n"
                           "R26:\n"
                           "R27:\n"
                           "R28:\n"
                           "R29:\n"
                           "R30:\n"
                           "R31:\n"
                           "setp.le.s32 p24, %130, 24;\n"
                           "@p24 mov.b32 %24, %129;\n"
                           "setp.le.s32 p25, %130, 25;\n"
                           "@p25 mov.b32 %25, %129;\n"
                           "setp.le.s32 p26, %130, 26;\n"
                           "@p26 mov.b32 %26, %129;\n"
                           "setp.le.s32 p27, %130, 27;\n"
                           "@p27 mov.b32 %27, %129;\n"
                           "setp.le.s32 p28, %130, 28;\n"
                           "@p28 mov.b32 %28, %129;\n"
                           "setp.le.s32 p29, %130, 29;\n"
                           "@p29 mov.b32 %29, %129;\n"
                           "setp.le.s32 p30, %130, 30;\n"
                           "@p30 mov.b32 %30, %129;\n"
                           "setp.le.s32 p31, %130, 31;\n"
                           "@p31 mov.b32 %31, %129;\n"
                           "R32:\n"
                           "R33:\n"
                           "R34:\n"
                           "R35:\n"
                           "R36:\n"
                           "R37:\n"
                           "R38:\n"
                           "R39:\n"
                           "setp.le.s32 p32, %130, 32;\n"
                           "@p32 mov.b32 %32, %129;\n"
                           "setp.le.s32 p33, %130, 33;\n"
                           "@p33 mov.b32 %33, %129;\n"
                           "setp.le.s32 p34, %130, 34;\n"
                           "@p34 mov.b32 %34, %129;\n"
                           "setp.le.s32 p35, %130, 35;\n"
                           "@p35 mov.b32 %35, %129;\n"
                           "setp.le.s32 p36, %130, 36;\n"
                           "@p36 mov.b32 %36, %129;\n"
                           "setp.le.s32 p37, %130, 37;\n"
                           "@p37 mov.b32 %37, %129;\n"
                           "setp.le.s32 p38, %130, 38;\n"
                           "@p38 mov.b32 %38, %129;\n"
                           "setp.le.s32 p39, %130, 39;\n"
                           "@p39 mov.b32 %39, %129;\n"
                           "R40:\n"
                           "R41:\n"
                           "R42:\n"
                           "R43:\n"
                           "R44:\n"
                           "R45:\n"
                           "R46:\n"
                           "R47:\n"
                           "setp.le.s32 p40, %130, 40;\n"
                           "@p40 mov.b32 %40, %129;\n"
                           "setp.le.s32 p41, %130, 41;\n"
                           "@p41 mov.b32 %41, %129;\n"
                           "setp.le.s32 p42, %130, 42;\n"
                           "@p42 mov.b32 %42, %129;\n"
                           "setp.le.s32 p43, %130, 43;\n"
                           "@p43 mov.b32 %43, %129;\n"
                           "setp.le.s32 p44, %130, 44;\n"
                           "@p44 mov.b32 %44, %129;\n"
                           "setp.le.s32 p45, %130, 45;\n"
                           "@p45 mov.b32 %45, %129;\n"
                           "setp.le.s32 p46, %130, 46;\n"
                           "@p46 mov.b32 %46, %129;\n"
                           "setp.le.s32 p47, %130, 47;\n"
                           "@p47 mov.b32 %47, %129;\n"
                           "R48:\n"
                           "R49:\n"
                           "R50:\n"
                           "R51:\n"
                           "R52:\n"
                           "R53:\n"
                           "R54:\n"
                           "R55:\n"
                           "setp.le.s32 p48, %130, 48;\n"
                           "@p48 mov.b32 %48, %129;\n"
                           "setp.le.s32 p49, %130, 49;\n"
                           "@p49 mov.b32 %49, %129;\n"
                           "setp.le.s32 p50, %130, 50;\n"
                           "@p50 mov.b32 %50, %129;\n"
                           "setp.le.s32 p51, %130, 51;\n"
                           "@p51 mov.b32 %51, %129;\n"
                           "setp.le.s32 p52, %130, 52;\n"
                           "@p52 mov.b32 %52, %129;\n"
                           "setp.le.s32 p53, %130, 53;\n"
                           "@p53 mov.b32 %53, %129;\n"
                           "setp.le.s32 p54, %130, 54;\n"
                           "@p54 mov.b32 %54, %129;\n"
                           "setp.le.s32 p55, %130, 55;\n"
                           "@p55 mov.b32 %55, %129;\n"
                           "R56:\n"
                           "R57:\n"
                           "R58:\n"
                           "R59:\n"
                           "R60:\n"
                           "R61:\n"
                           "R62:\n"
                           "R63:\n"
                           "setp.le.s32 p56, %130, 56;\n"
                           "@p56 mov.b32 %56, %129;\n"
                           "setp.le.s32 p57, %130, 57;\n"
                           "@p57 mov.b32 %57, %129;\n"
                           "setp.le.s32 p58, %130, 58;\n"
                           "@p58 mov.b32 %58, %129;\n"
                           "setp.le.s32 p59, %130, 59;\n"
                           "@p59 mov.b32 %59, %129;\n"
                           "setp.le.s32 p60, %130, 60;\n"
                           "@p60 mov.b32 %60, %129;\n"
                           "setp.le.s32 p61, %130, 61;\n"
                           "@p61 mov.b32 %61, %129;\n"
                           "setp.le.s32 p62, %130, 62;\n"
                           "@p62 mov.b32 %62, %129;\n"
                           "setp.le.s32 p63, %130, 63;\n"
                           "@p63 mov.b32 %63, %129;\n"
                           "R64:\n"
                           "R65:\n"
                           "R66:\n"
                           "R67:\n"
                           "R68:\n"
                           "R69:\n"
                           "R70:\n"
                           "R71:\n"
                           "setp.le.s32 p64, %130, 64;\n"
                           "@p64 mov.b32 %64, %129;\n"
                           "setp.le.s32 p65, %130, 65;\n"
                           "@p65 mov.b32 %65, %129;\n"
                           "setp.le.s32 p66, %130, 66;\n"
                           "@p66 mov.b32 %66, %129;\n"
                           "setp.le.s32 p67, %130, 67;\n"
                           "@p67 mov.b32 %67, %129;\n"
                           "setp.le.s32 p68, %130, 68;\n"
                           "@p68 mov.b32 %68, %129;\n"
                           "setp.le.s32 p69, %130, 69;\n"
                           "@p69 mov.b32 %69, %129;\n"
                           "setp.le.s32 p70, %130, 70;\n"
                           "@p70 mov.b32 %70, %129;\n"
                           "setp.le.s32 p71, %130, 71;\n"
                           "@p71 mov.b32 %71, %129;\n"
                           "R72:\n"
                           "R73:\n"
                           "R74:\n"
                           "R75:\n"
                           "R76:\n"
                           "R77:\n"
                           "R78:\n"
                           "R79:\n"
                           "setp.le.s32 p72, %130, 72;\n"
                           "@p72 mov.b32 %72, %129;\n"
                           "setp.le.s32 p73, %130, 73;\n"
                           "@p73 mov.b32 %73, %129;\n"
                           "setp.le.s32 p74, %130, 74;\n"
                           "@p74 mov.b32 %74, %129;\n"
                           "setp.le.s32 p75, %130, 75;\n"
                           "@p75 mov.b32 %75, %129;\n"
                           "setp.le.s32 p76, %130, 76;\n"
                           "@p76 mov.b32 %76, %129;\n"
                           "setp.le.s32 p77, %130, 77;\n"
                           "@p77 mov.b32 %77, %129;\n"
                           "setp.le.s32 p78, %130, 78;\n"
                           "@p78 mov.b32 %78, %129;\n"
                           "setp.le.s32 p79, %130, 79;\n"
                           "@p79 mov.b32 %79, %129;\n"
                           "R80:\n"
                           "R81:\n"
                           "R82:\n"
                           "R83:\n"
                           "R84:\n"
                           "R85:\n"
                           "R86:\n"
                           "R87:\n"
                           "setp.le.s32 p80, %130, 80;\n"
                           "@p80 mov.b32 %80, %129;\n"
                           "setp.le.s32 p81, %130, 81;\n"
                           "@p81 mov.b32 %81, %129;\n"
                           "setp.le.s32 p82, %130, 82;\n"
                           "@p82 mov.b32 %82, %129;\n"
                           "setp.le.s32 p83, %130, 83;\n"
                           "@p83 mov.b32 %83, %129;\n"
                           "setp.le.s32 p84, %130, 84;\n"
                           "@p84 mov.b32 %84, %129;\n"
                           "setp.le.s32 p85, %130, 85;\n"
                           "@p85 mov.b32 %85, %129;\n"
                           "setp.le.s32 p86, %130, 86;\n"
                           "@p86 mov.b32 %86, %129;\n"
                           "setp.le.s32 p87, %130, 87;\n"
                           "@p87 mov.b32 %87, %129;\n"
                           "R88:\n"
                           "R89:\n"
                           "R90:\n"
                           "R91:\n"
                           "R92:\n"
                           "R93:\n"
                           "R94:\n"
                           "R95:\n"
                           "setp.le.s32 p88, %130, 88;\n"
                           "@p88 mov.b32 %88, %129;\n"
                           "setp.le.s32 p89, %130, 89;\n"
                           "@p89 mov.b32 %89, %129;\n"
                           "setp.le.s32 p90, %130, 90;\n"
                           "@p90 mov.b32 %90, %129;\n"
                           "setp.le.s32 p91, %130, 91;\n"
                           "@p91 mov.b32 %91, %129;\n"
                           "setp.le.s32 p92, %130, 92;\n"
                           "@p92 mov.b32 %92, %129;\n"
                           "setp.le.s32 p93, %130, 93;\n"
                           "@p93 mov.b32 %93, %129;\n"
                           "setp.le.s32 p94, %130, 94;\n"
                           "@p94 mov.b32 %94, %129;\n"
                           "setp.le.s32 p95, %130, 95;\n"
                           "@p95 mov.b32 %95, %129;\n"
                           "R96:\n"
                           "R97:\n"
                           "R98:\n"
                           "R99:\n"
                           "R100:\n"
                           "R101:\n"
                           "R102:\n"
                           "R103:\n"
                           "setp.le.s32 p96, %130, 96;\n"
                           "@p96 mov.b32 %96, %129;\n"
                           "setp.le.s32 p97, %130, 97;\n"
                           "@p97 mov.b32 %97, %129;\n"
                           "setp.le.s32 p98, %130, 98;\n"
                           "@p98 mov.b32 %98, %129;\n"
                           "setp.le.s32 p99, %130, 99;\n"
                           "@p99 mov.b32 %99, %129;\n"
                           "setp.le.s32 p100, %130, 100;\n"
                           "@p100 mov.b32 %100, %129;\n"
                           "setp.le.s32 p101, %130, 101;\n"
                           "@p101 mov.b32 %101, %129;\n"
                           "setp.le.s32 p102, %130, 102;\n"
                           "@p102 mov.b32 %102, %129;\n"
                           "setp.le.s32 p103, %130, 103;\n"
                           "@p103 mov.b32 %103, %129;\n"
                           "R104:\n"
                           "R105:\n"
                           "R106:\n"
                           "R107:\n"
                           "R108:\n"
                           "R109:\n"
                           "R110:\n"
                           "R111:\n"
                           "setp.le.s32 p104, %130, 104;\n"
                           "@p104 mov.b32 %104, %129;\n"
                           "setp.le.s32 p105, %130, 105;\n"
                           "@p105 mov.b32 %105, %129;\n"
                           "setp.le.s32 p106, %130, 106;\n"
                           "@p106 mov.b32 %106, %129;\n"
                           "setp.le.s32 p107, %130, 107;\n"
                           "@p107 mov.b32 %107, %129;\n"
                           "setp.le.s32 p108, %130, 108;\n"
                           "@p108 mov.b32 %108, %129;\n"
                           "setp.le.s32 p109, %130, 109;\n"
                           "@p109 mov.b32 %109, %129;\n"
                           "setp.le.s32 p110, %130, 110;\n"
                           "@p110 mov.b32 %110, %129;\n"
                           "setp.le.s32 p111, %130, 111;\n"
                           "@p111 mov.b32 %111, %129;\n"
                           "R112:\n"
                           "R113:\n"
                           "R114:\n"
                           "R115:\n"
                           "R116:\n"
                           "R117:\n"
                           "R118:\n"
                           "R119:\n"
                           "setp.le.s32 p112, %130, 112;\n"
                           "@p112 mov.b32 %112, %129;\n"
                           "setp.le.s32 p113, %130, 113;\n"
                           "@p113 mov.b32 %113, %129;\n"
                           "setp.le.s32 p114, %130, 114;\n"
                           "@p114 mov.b32 %114, %129;\n"
                           "setp.le.s32 p115, %130, 115;\n"
                           "@p115 mov.b32 %115, %129;\n"
                           "setp.le.s32 p116, %130, 116;\n"
                           "@p116 mov.b32 %116, %129;\n"
                           "setp.le.s32 p117, %130, 117;\n"
                           "@p117 mov.b32 %117, %129;\n"
                           "setp.le.s32 p118, %130, 118;\n"
                           "@p118 mov.b32 %118, %129;\n"
                           "setp.le.s32 p119, %130, 119;\n"
                           "@p119 mov.b32 %119, %129;\n"
                           "R120:\n"
                           "R121:\n"
                           "R122:\n"
                           "R123:\n"
                           "R124:\n"
                           "R125:\n"
                           "R126:\n"
                           "R127:\n"
                           "setp.le.s32 p120, %130, 120;\n"
                           "@p120 mov.b32 %120, %129;\n"
                           "setp.le.s32 p121, %130, 121;\n"
                           "@p121 mov.b32 %121, %129;\n"
                           "setp.le.s32 p122, %130, 122;\n"
                           "@p122 mov.b32 %122, %129;\n"
                           "setp.le.s32 p123, %130, 123;\n"
                           "@p123 mov.b32 %123, %129;\n"
                           "setp.le.s32 p124, %130, 124;\n"
                           "@p124 mov.b32 %124, %129;\n"
                           "setp.le.s32 p125, %130, 125;\n"
                           "@p125 mov.b32 %125, %129;\n"
                           "setp.le.s32 p126, %130, 126;\n"
                           "@p126 mov.b32 %126, %129;\n"
                           "setp.le.s32 p127, %130, 127;\n"
                           "@p127 mov.b32 %127, %129;\n"
                           "R128:\n"
                           "}\n"
                           : "+f"(regsQk[127]),
                             "+f"(regsQk[126]),
                             "+f"(regsQk[125]),
                             "+f"(regsQk[124]),
                             "+f"(regsQk[123]),
                             "+f"(regsQk[122]),
                             "+f"(regsQk[121]),
                             "+f"(regsQk[120]),
                             "+f"(regsQk[119]),
                             "+f"(regsQk[118]),
                             "+f"(regsQk[117]),
                             "+f"(regsQk[116]),
                             "+f"(regsQk[115]),
                             "+f"(regsQk[114]),
                             "+f"(regsQk[113]),
                             "+f"(regsQk[112]),
                             "+f"(regsQk[111]),
                             "+f"(regsQk[110]),
                             "+f"(regsQk[109]),
                             "+f"(regsQk[108]),
                             "+f"(regsQk[107]),
                             "+f"(regsQk[106]),
                             "+f"(regsQk[105]),
                             "+f"(regsQk[104]),
                             "+f"(regsQk[103]),
                             "+f"(regsQk[102]),
                             "+f"(regsQk[101]),
                             "+f"(regsQk[100]),
                             "+f"(regsQk[99]),
                             "+f"(regsQk[98]),
                             "+f"(regsQk[97]),
                             "+f"(regsQk[96]),
                             "+f"(regsQk[95]),
                             "+f"(regsQk[94]),
                             "+f"(regsQk[93]),
                             "+f"(regsQk[92]),
                             "+f"(regsQk[91]),
                             "+f"(regsQk[90]),
                             "+f"(regsQk[89]),
                             "+f"(regsQk[88]),
                             "+f"(regsQk[87]),
                             "+f"(regsQk[86]),
                             "+f"(regsQk[85]),
                             "+f"(regsQk[84]),
                             "+f"(regsQk[83]),
                             "+f"(regsQk[82]),
                             "+f"(regsQk[81]),
                             "+f"(regsQk[80]),
                             "+f"(regsQk[79]),
                             "+f"(regsQk[78]),
                             "+f"(regsQk[77]),
                             "+f"(regsQk[76]),
                             "+f"(regsQk[75]),
                             "+f"(regsQk[74]),
                             "+f"(regsQk[73]),
                             "+f"(regsQk[72]),
                             "+f"(regsQk[71]),
                             "+f"(regsQk[70]),
                             "+f"(regsQk[69]),
                             "+f"(regsQk[68]),
                             "+f"(regsQk[67]),
                             "+f"(regsQk[66]),
                             "+f"(regsQk[65]),
                             "+f"(regsQk[64]),
                             "+f"(regsQk[63]),
                             "+f"(regsQk[62]),
                             "+f"(regsQk[61]),
                             "+f"(regsQk[60]),
                             "+f"(regsQk[59]),
                             "+f"(regsQk[58]),
                             "+f"(regsQk[57]),
                             "+f"(regsQk[56]),
                             "+f"(regsQk[55]),
                             "+f"(regsQk[54]),
                             "+f"(regsQk[53]),
                             "+f"(regsQk[52]),
                             "+f"(regsQk[51]),
                             "+f"(regsQk[50]),
                             "+f"(regsQk[49]),
                             "+f"(regsQk[48]),
                             "+f"(regsQk[47]),
                             "+f"(regsQk[46]),
                             "+f"(regsQk[45]),
                             "+f"(regsQk[44]),
                             "+f"(regsQk[43]),
                             "+f"(regsQk[42]),
                             "+f"(regsQk[41]),
                             "+f"(regsQk[40]),
                             "+f"(regsQk[39]),
                             "+f"(regsQk[38]),
                             "+f"(regsQk[37]),
                             "+f"(regsQk[36]),
                             "+f"(regsQk[35]),
                             "+f"(regsQk[34]),
                             "+f"(regsQk[33]),
                             "+f"(regsQk[32]),
                             "+f"(regsQk[31]),
                             "+f"(regsQk[30]),
                             "+f"(regsQk[29]),
                             "+f"(regsQk[28]),
                             "+f"(regsQk[27]),
                             "+f"(regsQk[26]),
                             "+f"(regsQk[25]),
                             "+f"(regsQk[24]),
                             "+f"(regsQk[23]),
                             "+f"(regsQk[22]),
                             "+f"(regsQk[21]),
                             "+f"(regsQk[20]),
                             "+f"(regsQk[19]),
                             "+f"(regsQk[18]),
                             "+f"(regsQk[17]),
                             "+f"(regsQk[16]),
                             "+f"(regsQk[15]),
                             "+f"(regsQk[14]),
                             "+f"(regsQk[13]),
                             "+f"(regsQk[12]),
                             "+f"(regsQk[11]),
                             "+f"(regsQk[10]),
                             "+f"(regsQk[9]),
                             "+f"(regsQk[8]),
                             "+f"(regsQk[7]),
                             "+f"(regsQk[6]),
                             "+f"(regsQk[5]),
                             "+f"(regsQk[4]),
                             "+f"(regsQk[3]),
                             "+f"(regsQk[2]),
                             "+f"(regsQk[1]),
                             "+f"(regsQk[0])
                           : "r"(int32_t{__shfl_sync(uint32_t{0xffffffff},
                                                     (int32_t{128}) - (clampedUniformDist),
                                                     int32_t{0},
                                                     int32_t{32})}),
                             "f"(float{-3.4028235e+38}),
                             "r"((int32_t{128}) - (clampedDist)));
            }
            // Mask.h:1126
            {
              // Mask.h:185
              int32_t const uniformDist{
                (((tileOffsetQ) + ((mWarpGrpWarpIdx) * (int32_t{32}))) + (int32_t{1})) -
                (tileOffsetK)};
              // Mask.h:187
              int32_t clampedUniformDist{trtllm::dev::clamp(uniformDist, int32_t{0}, int32_t{128})};
              // Mask.h:207
              int32_t clampedDist{trtllm::dev::clamp(((tokenIdxQ) + (int32_t{1})) - (tileOffsetK),
                                                     int32_t{0},
                                                     int32_t{128})};
              // Mask.h:1128
              asm volatile("{\n"
                           ".reg .pred p<128>;\n"
                           "FirstToMask: .branchtargets R<129>;\n"
                           "brx.idx.uni %128, FirstToMask;\n"
                           "R0:\n"
                           "R1:\n"
                           "R2:\n"
                           "R3:\n"
                           "R4:\n"
                           "R5:\n"
                           "R6:\n"
                           "R7:\n"
                           "setp.le.s32 p0, %130, 0;\n"
                           "@p0 mov.b32 %0, %129;\n"
                           "setp.le.s32 p1, %130, 1;\n"
                           "@p1 mov.b32 %1, %129;\n"
                           "setp.le.s32 p2, %130, 2;\n"
                           "@p2 mov.b32 %2, %129;\n"
                           "setp.le.s32 p3, %130, 3;\n"
                           "@p3 mov.b32 %3, %129;\n"
                           "setp.le.s32 p4, %130, 4;\n"
                           "@p4 mov.b32 %4, %129;\n"
                           "setp.le.s32 p5, %130, 5;\n"
                           "@p5 mov.b32 %5, %129;\n"
                           "setp.le.s32 p6, %130, 6;\n"
                           "@p6 mov.b32 %6, %129;\n"
                           "setp.le.s32 p7, %130, 7;\n"
                           "@p7 mov.b32 %7, %129;\n"
                           "R8:\n"
                           "R9:\n"
                           "R10:\n"
                           "R11:\n"
                           "R12:\n"
                           "R13:\n"
                           "R14:\n"
                           "R15:\n"
                           "setp.le.s32 p8, %130, 8;\n"
                           "@p8 mov.b32 %8, %129;\n"
                           "setp.le.s32 p9, %130, 9;\n"
                           "@p9 mov.b32 %9, %129;\n"
                           "setp.le.s32 p10, %130, 10;\n"
                           "@p10 mov.b32 %10, %129;\n"
                           "setp.le.s32 p11, %130, 11;\n"
                           "@p11 mov.b32 %11, %129;\n"
                           "setp.le.s32 p12, %130, 12;\n"
                           "@p12 mov.b32 %12, %129;\n"
                           "setp.le.s32 p13, %130, 13;\n"
                           "@p13 mov.b32 %13, %129;\n"
                           "setp.le.s32 p14, %130, 14;\n"
                           "@p14 mov.b32 %14, %129;\n"
                           "setp.le.s32 p15, %130, 15;\n"
                           "@p15 mov.b32 %15, %129;\n"
                           "R16:\n"
                           "R17:\n"
                           "R18:\n"
                           "R19:\n"
                           "R20:\n"
                           "R21:\n"
                           "R22:\n"
                           "R23:\n"
                           "setp.le.s32 p16, %130, 16;\n"
                           "@p16 mov.b32 %16, %129;\n"
                           "setp.le.s32 p17, %130, 17;\n"
                           "@p17 mov.b32 %17, %129;\n"
                           "setp.le.s32 p18, %130, 18;\n"
                           "@p18 mov.b32 %18, %129;\n"
                           "setp.le.s32 p19, %130, 19;\n"
                           "@p19 mov.b32 %19, %129;\n"
                           "setp.le.s32 p20, %130, 20;\n"
                           "@p20 mov.b32 %20, %129;\n"
                           "setp.le.s32 p21, %130, 21;\n"
                           "@p21 mov.b32 %21, %129;\n"
                           "setp.le.s32 p22, %130, 22;\n"
                           "@p22 mov.b32 %22, %129;\n"
                           "setp.le.s32 p23, %130, 23;\n"
                           "@p23 mov.b32 %23, %129;\n"
                           "R24:\n"
                           "R25:\n"
                           "R26:\n"
                           "R27:\n"
                           "R28:\n"
                           "R29:\n"
                           "R30:\n"
                           "R31:\n"
                           "setp.le.s32 p24, %130, 24;\n"
                           "@p24 mov.b32 %24, %129;\n"
                           "setp.le.s32 p25, %130, 25;\n"
                           "@p25 mov.b32 %25, %129;\n"
                           "setp.le.s32 p26, %130, 26;\n"
                           "@p26 mov.b32 %26, %129;\n"
                           "setp.le.s32 p27, %130, 27;\n"
                           "@p27 mov.b32 %27, %129;\n"
                           "setp.le.s32 p28, %130, 28;\n"
                           "@p28 mov.b32 %28, %129;\n"
                           "setp.le.s32 p29, %130, 29;\n"
                           "@p29 mov.b32 %29, %129;\n"
                           "setp.le.s32 p30, %130, 30;\n"
                           "@p30 mov.b32 %30, %129;\n"
                           "setp.le.s32 p31, %130, 31;\n"
                           "@p31 mov.b32 %31, %129;\n"
                           "R32:\n"
                           "R33:\n"
                           "R34:\n"
                           "R35:\n"
                           "R36:\n"
                           "R37:\n"
                           "R38:\n"
                           "R39:\n"
                           "setp.le.s32 p32, %130, 32;\n"
                           "@p32 mov.b32 %32, %129;\n"
                           "setp.le.s32 p33, %130, 33;\n"
                           "@p33 mov.b32 %33, %129;\n"
                           "setp.le.s32 p34, %130, 34;\n"
                           "@p34 mov.b32 %34, %129;\n"
                           "setp.le.s32 p35, %130, 35;\n"
                           "@p35 mov.b32 %35, %129;\n"
                           "setp.le.s32 p36, %130, 36;\n"
                           "@p36 mov.b32 %36, %129;\n"
                           "setp.le.s32 p37, %130, 37;\n"
                           "@p37 mov.b32 %37, %129;\n"
                           "setp.le.s32 p38, %130, 38;\n"
                           "@p38 mov.b32 %38, %129;\n"
                           "setp.le.s32 p39, %130, 39;\n"
                           "@p39 mov.b32 %39, %129;\n"
                           "R40:\n"
                           "R41:\n"
                           "R42:\n"
                           "R43:\n"
                           "R44:\n"
                           "R45:\n"
                           "R46:\n"
                           "R47:\n"
                           "setp.le.s32 p40, %130, 40;\n"
                           "@p40 mov.b32 %40, %129;\n"
                           "setp.le.s32 p41, %130, 41;\n"
                           "@p41 mov.b32 %41, %129;\n"
                           "setp.le.s32 p42, %130, 42;\n"
                           "@p42 mov.b32 %42, %129;\n"
                           "setp.le.s32 p43, %130, 43;\n"
                           "@p43 mov.b32 %43, %129;\n"
                           "setp.le.s32 p44, %130, 44;\n"
                           "@p44 mov.b32 %44, %129;\n"
                           "setp.le.s32 p45, %130, 45;\n"
                           "@p45 mov.b32 %45, %129;\n"
                           "setp.le.s32 p46, %130, 46;\n"
                           "@p46 mov.b32 %46, %129;\n"
                           "setp.le.s32 p47, %130, 47;\n"
                           "@p47 mov.b32 %47, %129;\n"
                           "R48:\n"
                           "R49:\n"
                           "R50:\n"
                           "R51:\n"
                           "R52:\n"
                           "R53:\n"
                           "R54:\n"
                           "R55:\n"
                           "setp.le.s32 p48, %130, 48;\n"
                           "@p48 mov.b32 %48, %129;\n"
                           "setp.le.s32 p49, %130, 49;\n"
                           "@p49 mov.b32 %49, %129;\n"
                           "setp.le.s32 p50, %130, 50;\n"
                           "@p50 mov.b32 %50, %129;\n"
                           "setp.le.s32 p51, %130, 51;\n"
                           "@p51 mov.b32 %51, %129;\n"
                           "setp.le.s32 p52, %130, 52;\n"
                           "@p52 mov.b32 %52, %129;\n"
                           "setp.le.s32 p53, %130, 53;\n"
                           "@p53 mov.b32 %53, %129;\n"
                           "setp.le.s32 p54, %130, 54;\n"
                           "@p54 mov.b32 %54, %129;\n"
                           "setp.le.s32 p55, %130, 55;\n"
                           "@p55 mov.b32 %55, %129;\n"
                           "R56:\n"
                           "R57:\n"
                           "R58:\n"
                           "R59:\n"
                           "R60:\n"
                           "R61:\n"
                           "R62:\n"
                           "R63:\n"
                           "setp.le.s32 p56, %130, 56;\n"
                           "@p56 mov.b32 %56, %129;\n"
                           "setp.le.s32 p57, %130, 57;\n"
                           "@p57 mov.b32 %57, %129;\n"
                           "setp.le.s32 p58, %130, 58;\n"
                           "@p58 mov.b32 %58, %129;\n"
                           "setp.le.s32 p59, %130, 59;\n"
                           "@p59 mov.b32 %59, %129;\n"
                           "setp.le.s32 p60, %130, 60;\n"
                           "@p60 mov.b32 %60, %129;\n"
                           "setp.le.s32 p61, %130, 61;\n"
                           "@p61 mov.b32 %61, %129;\n"
                           "setp.le.s32 p62, %130, 62;\n"
                           "@p62 mov.b32 %62, %129;\n"
                           "setp.le.s32 p63, %130, 63;\n"
                           "@p63 mov.b32 %63, %129;\n"
                           "R64:\n"
                           "R65:\n"
                           "R66:\n"
                           "R67:\n"
                           "R68:\n"
                           "R69:\n"
                           "R70:\n"
                           "R71:\n"
                           "setp.le.s32 p64, %130, 64;\n"
                           "@p64 mov.b32 %64, %129;\n"
                           "setp.le.s32 p65, %130, 65;\n"
                           "@p65 mov.b32 %65, %129;\n"
                           "setp.le.s32 p66, %130, 66;\n"
                           "@p66 mov.b32 %66, %129;\n"
                           "setp.le.s32 p67, %130, 67;\n"
                           "@p67 mov.b32 %67, %129;\n"
                           "setp.le.s32 p68, %130, 68;\n"
                           "@p68 mov.b32 %68, %129;\n"
                           "setp.le.s32 p69, %130, 69;\n"
                           "@p69 mov.b32 %69, %129;\n"
                           "setp.le.s32 p70, %130, 70;\n"
                           "@p70 mov.b32 %70, %129;\n"
                           "setp.le.s32 p71, %130, 71;\n"
                           "@p71 mov.b32 %71, %129;\n"
                           "R72:\n"
                           "R73:\n"
                           "R74:\n"
                           "R75:\n"
                           "R76:\n"
                           "R77:\n"
                           "R78:\n"
                           "R79:\n"
                           "setp.le.s32 p72, %130, 72;\n"
                           "@p72 mov.b32 %72, %129;\n"
                           "setp.le.s32 p73, %130, 73;\n"
                           "@p73 mov.b32 %73, %129;\n"
                           "setp.le.s32 p74, %130, 74;\n"
                           "@p74 mov.b32 %74, %129;\n"
                           "setp.le.s32 p75, %130, 75;\n"
                           "@p75 mov.b32 %75, %129;\n"
                           "setp.le.s32 p76, %130, 76;\n"
                           "@p76 mov.b32 %76, %129;\n"
                           "setp.le.s32 p77, %130, 77;\n"
                           "@p77 mov.b32 %77, %129;\n"
                           "setp.le.s32 p78, %130, 78;\n"
                           "@p78 mov.b32 %78, %129;\n"
                           "setp.le.s32 p79, %130, 79;\n"
                           "@p79 mov.b32 %79, %129;\n"
                           "R80:\n"
                           "R81:\n"
                           "R82:\n"
                           "R83:\n"
                           "R84:\n"
                           "R85:\n"
                           "R86:\n"
                           "R87:\n"
                           "setp.le.s32 p80, %130, 80;\n"
                           "@p80 mov.b32 %80, %129;\n"
                           "setp.le.s32 p81, %130, 81;\n"
                           "@p81 mov.b32 %81, %129;\n"
                           "setp.le.s32 p82, %130, 82;\n"
                           "@p82 mov.b32 %82, %129;\n"
                           "setp.le.s32 p83, %130, 83;\n"
                           "@p83 mov.b32 %83, %129;\n"
                           "setp.le.s32 p84, %130, 84;\n"
                           "@p84 mov.b32 %84, %129;\n"
                           "setp.le.s32 p85, %130, 85;\n"
                           "@p85 mov.b32 %85, %129;\n"
                           "setp.le.s32 p86, %130, 86;\n"
                           "@p86 mov.b32 %86, %129;\n"
                           "setp.le.s32 p87, %130, 87;\n"
                           "@p87 mov.b32 %87, %129;\n"
                           "R88:\n"
                           "R89:\n"
                           "R90:\n"
                           "R91:\n"
                           "R92:\n"
                           "R93:\n"
                           "R94:\n"
                           "R95:\n"
                           "setp.le.s32 p88, %130, 88;\n"
                           "@p88 mov.b32 %88, %129;\n"
                           "setp.le.s32 p89, %130, 89;\n"
                           "@p89 mov.b32 %89, %129;\n"
                           "setp.le.s32 p90, %130, 90;\n"
                           "@p90 mov.b32 %90, %129;\n"
                           "setp.le.s32 p91, %130, 91;\n"
                           "@p91 mov.b32 %91, %129;\n"
                           "setp.le.s32 p92, %130, 92;\n"
                           "@p92 mov.b32 %92, %129;\n"
                           "setp.le.s32 p93, %130, 93;\n"
                           "@p93 mov.b32 %93, %129;\n"
                           "setp.le.s32 p94, %130, 94;\n"
                           "@p94 mov.b32 %94, %129;\n"
                           "setp.le.s32 p95, %130, 95;\n"
                           "@p95 mov.b32 %95, %129;\n"
                           "R96:\n"
                           "R97:\n"
                           "R98:\n"
                           "R99:\n"
                           "R100:\n"
                           "R101:\n"
                           "R102:\n"
                           "R103:\n"
                           "setp.le.s32 p96, %130, 96;\n"
                           "@p96 mov.b32 %96, %129;\n"
                           "setp.le.s32 p97, %130, 97;\n"
                           "@p97 mov.b32 %97, %129;\n"
                           "setp.le.s32 p98, %130, 98;\n"
                           "@p98 mov.b32 %98, %129;\n"
                           "setp.le.s32 p99, %130, 99;\n"
                           "@p99 mov.b32 %99, %129;\n"
                           "setp.le.s32 p100, %130, 100;\n"
                           "@p100 mov.b32 %100, %129;\n"
                           "setp.le.s32 p101, %130, 101;\n"
                           "@p101 mov.b32 %101, %129;\n"
                           "setp.le.s32 p102, %130, 102;\n"
                           "@p102 mov.b32 %102, %129;\n"
                           "setp.le.s32 p103, %130, 103;\n"
                           "@p103 mov.b32 %103, %129;\n"
                           "R104:\n"
                           "R105:\n"
                           "R106:\n"
                           "R107:\n"
                           "R108:\n"
                           "R109:\n"
                           "R110:\n"
                           "R111:\n"
                           "setp.le.s32 p104, %130, 104;\n"
                           "@p104 mov.b32 %104, %129;\n"
                           "setp.le.s32 p105, %130, 105;\n"
                           "@p105 mov.b32 %105, %129;\n"
                           "setp.le.s32 p106, %130, 106;\n"
                           "@p106 mov.b32 %106, %129;\n"
                           "setp.le.s32 p107, %130, 107;\n"
                           "@p107 mov.b32 %107, %129;\n"
                           "setp.le.s32 p108, %130, 108;\n"
                           "@p108 mov.b32 %108, %129;\n"
                           "setp.le.s32 p109, %130, 109;\n"
                           "@p109 mov.b32 %109, %129;\n"
                           "setp.le.s32 p110, %130, 110;\n"
                           "@p110 mov.b32 %110, %129;\n"
                           "setp.le.s32 p111, %130, 111;\n"
                           "@p111 mov.b32 %111, %129;\n"
                           "R112:\n"
                           "R113:\n"
                           "R114:\n"
                           "R115:\n"
                           "R116:\n"
                           "R117:\n"
                           "R118:\n"
                           "R119:\n"
                           "setp.le.s32 p112, %130, 112;\n"
                           "@p112 mov.b32 %112, %129;\n"
                           "setp.le.s32 p113, %130, 113;\n"
                           "@p113 mov.b32 %113, %129;\n"
                           "setp.le.s32 p114, %130, 114;\n"
                           "@p114 mov.b32 %114, %129;\n"
                           "setp.le.s32 p115, %130, 115;\n"
                           "@p115 mov.b32 %115, %129;\n"
                           "setp.le.s32 p116, %130, 116;\n"
                           "@p116 mov.b32 %116, %129;\n"
                           "setp.le.s32 p117, %130, 117;\n"
                           "@p117 mov.b32 %117, %129;\n"
                           "setp.le.s32 p118, %130, 118;\n"
                           "@p118 mov.b32 %118, %129;\n"
                           "setp.le.s32 p119, %130, 119;\n"
                           "@p119 mov.b32 %119, %129;\n"
                           "R120:\n"
                           "R121:\n"
                           "R122:\n"
                           "R123:\n"
                           "R124:\n"
                           "R125:\n"
                           "R126:\n"
                           "R127:\n"
                           "setp.le.s32 p120, %130, 120;\n"
                           "@p120 mov.b32 %120, %129;\n"
                           "setp.le.s32 p121, %130, 121;\n"
                           "@p121 mov.b32 %121, %129;\n"
                           "setp.le.s32 p122, %130, 122;\n"
                           "@p122 mov.b32 %122, %129;\n"
                           "setp.le.s32 p123, %130, 123;\n"
                           "@p123 mov.b32 %123, %129;\n"
                           "setp.le.s32 p124, %130, 124;\n"
                           "@p124 mov.b32 %124, %129;\n"
                           "setp.le.s32 p125, %130, 125;\n"
                           "@p125 mov.b32 %125, %129;\n"
                           "setp.le.s32 p126, %130, 126;\n"
                           "@p126 mov.b32 %126, %129;\n"
                           "setp.le.s32 p127, %130, 127;\n"
                           "@p127 mov.b32 %127, %129;\n"
                           "R128:\n"
                           "}\n"
                           : "+f"(regsQk[0]),
                             "+f"(regsQk[1]),
                             "+f"(regsQk[2]),
                             "+f"(regsQk[3]),
                             "+f"(regsQk[4]),
                             "+f"(regsQk[5]),
                             "+f"(regsQk[6]),
                             "+f"(regsQk[7]),
                             "+f"(regsQk[8]),
                             "+f"(regsQk[9]),
                             "+f"(regsQk[10]),
                             "+f"(regsQk[11]),
                             "+f"(regsQk[12]),
                             "+f"(regsQk[13]),
                             "+f"(regsQk[14]),
                             "+f"(regsQk[15]),
                             "+f"(regsQk[16]),
                             "+f"(regsQk[17]),
                             "+f"(regsQk[18]),
                             "+f"(regsQk[19]),
                             "+f"(regsQk[20]),
                             "+f"(regsQk[21]),
                             "+f"(regsQk[22]),
                             "+f"(regsQk[23]),
                             "+f"(regsQk[24]),
                             "+f"(regsQk[25]),
                             "+f"(regsQk[26]),
                             "+f"(regsQk[27]),
                             "+f"(regsQk[28]),
                             "+f"(regsQk[29]),
                             "+f"(regsQk[30]),
                             "+f"(regsQk[31]),
                             "+f"(regsQk[32]),
                             "+f"(regsQk[33]),
                             "+f"(regsQk[34]),
                             "+f"(regsQk[35]),
                             "+f"(regsQk[36]),
                             "+f"(regsQk[37]),
                             "+f"(regsQk[38]),
                             "+f"(regsQk[39]),
                             "+f"(regsQk[40]),
                             "+f"(regsQk[41]),
                             "+f"(regsQk[42]),
                             "+f"(regsQk[43]),
                             "+f"(regsQk[44]),
                             "+f"(regsQk[45]),
                             "+f"(regsQk[46]),
                             "+f"(regsQk[47]),
                             "+f"(regsQk[48]),
                             "+f"(regsQk[49]),
                             "+f"(regsQk[50]),
                             "+f"(regsQk[51]),
                             "+f"(regsQk[52]),
                             "+f"(regsQk[53]),
                             "+f"(regsQk[54]),
                             "+f"(regsQk[55]),
                             "+f"(regsQk[56]),
                             "+f"(regsQk[57]),
                             "+f"(regsQk[58]),
                             "+f"(regsQk[59]),
                             "+f"(regsQk[60]),
                             "+f"(regsQk[61]),
                             "+f"(regsQk[62]),
                             "+f"(regsQk[63]),
                             "+f"(regsQk[64]),
                             "+f"(regsQk[65]),
                             "+f"(regsQk[66]),
                             "+f"(regsQk[67]),
                             "+f"(regsQk[68]),
                             "+f"(regsQk[69]),
                             "+f"(regsQk[70]),
                             "+f"(regsQk[71]),
                             "+f"(regsQk[72]),
                             "+f"(regsQk[73]),
                             "+f"(regsQk[74]),
                             "+f"(regsQk[75]),
                             "+f"(regsQk[76]),
                             "+f"(regsQk[77]),
                             "+f"(regsQk[78]),
                             "+f"(regsQk[79]),
                             "+f"(regsQk[80]),
                             "+f"(regsQk[81]),
                             "+f"(regsQk[82]),
                             "+f"(regsQk[83]),
                             "+f"(regsQk[84]),
                             "+f"(regsQk[85]),
                             "+f"(regsQk[86]),
                             "+f"(regsQk[87]),
                             "+f"(regsQk[88]),
                             "+f"(regsQk[89]),
                             "+f"(regsQk[90]),
                             "+f"(regsQk[91]),
                             "+f"(regsQk[92]),
                             "+f"(regsQk[93]),
                             "+f"(regsQk[94]),
                             "+f"(regsQk[95]),
                             "+f"(regsQk[96]),
                             "+f"(regsQk[97]),
                             "+f"(regsQk[98]),
                             "+f"(regsQk[99]),
                             "+f"(regsQk[100]),
                             "+f"(regsQk[101]),
                             "+f"(regsQk[102]),
                             "+f"(regsQk[103]),
                             "+f"(regsQk[104]),
                             "+f"(regsQk[105]),
                             "+f"(regsQk[106]),
                             "+f"(regsQk[107]),
                             "+f"(regsQk[108]),
                             "+f"(regsQk[109]),
                             "+f"(regsQk[110]),
                             "+f"(regsQk[111]),
                             "+f"(regsQk[112]),
                             "+f"(regsQk[113]),
                             "+f"(regsQk[114]),
                             "+f"(regsQk[115]),
                             "+f"(regsQk[116]),
                             "+f"(regsQk[117]),
                             "+f"(regsQk[118]),
                             "+f"(regsQk[119]),
                             "+f"(regsQk[120]),
                             "+f"(regsQk[121]),
                             "+f"(regsQk[122]),
                             "+f"(regsQk[123]),
                             "+f"(regsQk[124]),
                             "+f"(regsQk[125]),
                             "+f"(regsQk[126]),
                             "+f"(regsQk[127])
                           : "r"(int32_t{__shfl_sync(uint32_t{0xffffffff},
                                                     clampedUniformDist,
                                                     int32_t{0},
                                                     int32_t{32})}),
                             "f"(float{-3.4028235e+38}),
                             "r"(clampedDist));
            }
            // TmemS.h:1681
            CUTLASS_PRAGMA_UNROLL
            for (int32_t loopOffset1363 = int32_t{0}; loopOffset1363 < int32_t{128};
                 loopOffset1363 += int32_t{4}) {
              // TmemS.h:1694
              ilpMax0 = fmaxf(ilpMax0, regsQk[loopOffset1363]);
              // TmemS.h:1694
              ilpMax1 = fmaxf(ilpMax1, regsQk[(loopOffset1363) + (int32_t{1})]);
              // TmemS.h:1694
              ilpMax2 = fmaxf(ilpMax2, regsQk[(loopOffset1363) + (int32_t{2})]);
              // TmemS.h:1694
              ilpMax3 = fmaxf(ilpMax3, regsQk[(loopOffset1363) + (int32_t{3})]);
            }
          }
          // TmemS.h:2197
          ilpMax0 = fmaxf(ilpMax0, ilpMax2);
          // TmemS.h:2197
          ilpMax1 = fmaxf(ilpMax1, ilpMax3);
          // TmemS.h:2216
          newMaxArray7[int32_t{0}] = fmaxf(ilpMax0, ilpMax1);
          // TmemS.h:1334
          qkPtr07 = &regsQk[int32_t{0}];
          // TmemS.h:1336
          qkPtr17 = &regsQk[int32_t{0}];
          // Task.cpp:43
          ++tmemS0ConsState;
        }
        //
        // tmemSoftmaxLocal0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{8}, Flags{0}].
        //
        // TmemSoftmax.h:261
        float* oldMaxPtr8;
        // TmemSoftmax.h:267
        float* sumPtr8;
        // TmemSoftmax.h:273
        float* newMaxPtr8;
        // Task.cpp:1511
        oldMaxPtr8 = oldMaxPtr7;
        // Task.cpp:1511
        sumPtr8 = sumPtr7;
        // Task.cpp:1511
        newMaxPtr8 = newMaxPtr07;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{tmemSoftmaxLocal0ProdState.index()};
          // TmemTile.cpp:373
          cutlass::Array<float, 2> stats;
          // TmemSoftmax.h:365
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset1385 = int32_t{0}; loopOffset1385 < int32_t{1}; ++loopOffset1385) {
            // TmemSoftmax.h:382
            stats[loopOffset1385] = oldMaxPtr8[loopOffset1385];
            // TmemSoftmax.h:384
            stats[(loopOffset1385) + (int32_t{1})] = newMaxPtr8[loopOffset1385];
          }
          // TmemTile.cpp:836
          {
            // TmemTile.cpp:838
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:871
            uint32_t const(&srcSlice0)[2]{
              reinterpret_cast<uint32_t const(&)[2]>(stats[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_st_32x32b(
              (tmemBasePtr) +
                (static_cast<uint32_t>(int32_t((tmemSoftmaxLocal0DstStack.mInstId) == (int32_t{0}))
                                         ? (int32_t{256})
                                         : (int32_t{288}))),
              srcSlice0);
          }
          // TmemSoftmax.h:407
          cutlass::arch::fence_view_async_tmem_store();
        }
        //
        // tmemSoftmaxLocal0 [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{8}, Flags{0}].
        //
        // Task.cpp:4522
        {
          // Task.cpp:4540
          {
            // Task.cpp:4556
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
        // TmemP.h:569
        float* newMaxPtr14;
        // TmemP.h:574
        float* regsFp32P14;
        // Task.cpp:1511
        newMaxPtr14 = newMaxPtr17;
        // Task.cpp:1511
        regsFp32P14 = qkPtr17;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{tmemP0ProdState.index()};
          // TmemP.h:1025
          float negScaledMaxArray[1];
          // TmemP.h:1128
          float newMax{newMaxPtr14[int32_t{0}]};
          // Common.h:562
          if ((newMax) == (float{-3.4028235e+38})) {
            // Common.h:564
            newMax = float{0};
          }
          // TmemP.h:1134
          float negScaledMax{-((newMax) * (scaleSoftmaxLog214))};
          // TmemP.h:1144
          negScaledMaxArray[int32_t{0}] = (negScaledMax) + (float{8.8073549});
          // TmemP.h:1655
          {
            // TmemP.h:1658
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{0}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{1}];
            // TmemP.h:801
            vals[int32_t{0}] =
              (log2Scale2[int32_t{0}]) * (vals[int32_t{0}]) + (negScaledMax[int32_t{0}]);
            // TmemP.h:810
            vals[int32_t{1}] =
              (log2Scale2[int32_t{1}]) * (vals[int32_t{1}]) + (negScaledMax[int32_t{1}]);
            // TmemP.h:833
            regsFp32P14[int32_t{0}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{1}] = vals[int32_t{1}];
          }
          // TmemP.h:1655
          {
            // TmemP.h:1658
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{2}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{3}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{2}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{3}] = vals[int32_t{1}];
          }
          // TmemP.h:1454
          tmemP0DstStack.mOrderedSequence.wait();
          // TmemP.h:1773
          regsFp32P14[int32_t{0}] = exp2f(regsFp32P14[int32_t{0}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{4}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{5}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{4}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{5}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{1}] = exp2f(regsFp32P14[int32_t{1}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{2}] = exp2f(regsFp32P14[int32_t{2}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{6}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{7}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{6}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{7}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{3}] = exp2f(regsFp32P14[int32_t{3}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{4}] = exp2f(regsFp32P14[int32_t{4}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{8}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{9}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{8}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{9}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{5}] = exp2f(regsFp32P14[int32_t{5}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{6}] = exp2f(regsFp32P14[int32_t{6}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{10}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{11}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{10}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{11}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{7}] = exp2f(regsFp32P14[int32_t{7}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{8}] = exp2f(regsFp32P14[int32_t{8}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{12}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{13}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{12}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{13}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{9}] = exp2f(regsFp32P14[int32_t{9}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{10}] = exp2f(regsFp32P14[int32_t{10}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{14}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{15}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{14}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{15}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{11}] = exp2f(regsFp32P14[int32_t{11}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{0}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{1}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{2}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{3}];
            // TmemP.h:745
            regsP[int32_t{0}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{12}] = exp2f(regsFp32P14[int32_t{12}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{16}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{17}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{16}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{17}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{13}] = exp2f(regsFp32P14[int32_t{13}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{14}] = exp2f(regsFp32P14[int32_t{14}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{18}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{19}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{18}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{19}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{15}] = exp2f(regsFp32P14[int32_t{15}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{4}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{5}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{6}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{7}];
            // TmemP.h:745
            regsP[int32_t{1}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{16}] = exp2f(regsFp32P14[int32_t{16}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{20}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{21}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{20}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{21}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{17}] = exp2f(regsFp32P14[int32_t{17}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{18}] = exp2f(regsFp32P14[int32_t{18}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{22}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{23}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{22}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{23}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{19}] = exp2f(regsFp32P14[int32_t{19}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{8}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{9}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{10}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{11}];
            // TmemP.h:745
            regsP[int32_t{2}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{20}] = exp2f(regsFp32P14[int32_t{20}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{24}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{25}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{24}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{25}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{21}] = exp2f(regsFp32P14[int32_t{21}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{22}] = exp2f(regsFp32P14[int32_t{22}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{26}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{27}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{26}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{27}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{23}] = exp2f(regsFp32P14[int32_t{23}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{12}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{13}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{14}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{15}];
            // TmemP.h:745
            regsP[int32_t{3}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{24}] = exp2f(regsFp32P14[int32_t{24}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{28}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{29}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{28}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{29}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{25}] = exp2f(regsFp32P14[int32_t{25}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{26}] = exp2f(regsFp32P14[int32_t{26}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{30}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{31}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{30}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{31}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{27}] = exp2f(regsFp32P14[int32_t{27}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{16}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{17}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{18}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{19}];
            // TmemP.h:745
            regsP[int32_t{4}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{28}] = exp2f(regsFp32P14[int32_t{28}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{32}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{33}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{32}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{33}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{29}] = exp2f(regsFp32P14[int32_t{29}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{30}] = exp2f(regsFp32P14[int32_t{30}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{34}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{35}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{34}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{35}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{31}] = exp2f(regsFp32P14[int32_t{31}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{20}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{21}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{22}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{23}];
            // TmemP.h:745
            regsP[int32_t{5}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{32}] = exp2f(regsFp32P14[int32_t{32}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{36}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{37}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{36}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{37}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{33}] = exp2f(regsFp32P14[int32_t{33}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{34}] = exp2f(regsFp32P14[int32_t{34}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{38}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{39}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{38}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{39}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{35}] = exp2f(regsFp32P14[int32_t{35}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{24}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{25}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{26}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{27}];
            // TmemP.h:745
            regsP[int32_t{6}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{36}] = exp2f(regsFp32P14[int32_t{36}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{40}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{41}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{40}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{41}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{37}] = exp2f(regsFp32P14[int32_t{37}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{38}] = exp2f(regsFp32P14[int32_t{38}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{42}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{43}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{42}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{43}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{39}] = exp2f(regsFp32P14[int32_t{39}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{28}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{29}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{30}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{31}];
            // TmemP.h:745
            regsP[int32_t{7}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{40}] = exp2f(regsFp32P14[int32_t{40}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{44}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{45}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{44}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{45}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{41}] = exp2f(regsFp32P14[int32_t{41}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{42}] = exp2f(regsFp32P14[int32_t{42}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{46}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{47}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{46}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{47}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{43}] = exp2f(regsFp32P14[int32_t{43}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{32}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{33}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{34}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{35}];
            // TmemP.h:745
            regsP[int32_t{8}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{44}] = exp2f(regsFp32P14[int32_t{44}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{48}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{49}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{48}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{49}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{45}] = exp2f(regsFp32P14[int32_t{45}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{46}] = exp2f(regsFp32P14[int32_t{46}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{50}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{51}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{50}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{51}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{47}] = exp2f(regsFp32P14[int32_t{47}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{36}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{37}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{38}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{39}];
            // TmemP.h:745
            regsP[int32_t{9}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{48}] = exp2f(regsFp32P14[int32_t{48}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{52}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{53}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{52}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{53}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{49}] = exp2f(regsFp32P14[int32_t{49}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{50}] = exp2f(regsFp32P14[int32_t{50}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{54}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{55}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{54}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{55}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{51}] = exp2f(regsFp32P14[int32_t{51}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{40}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{41}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{42}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{43}];
            // TmemP.h:745
            regsP[int32_t{10}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{52}] = exp2f(regsFp32P14[int32_t{52}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{56}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{57}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{56}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{57}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{53}] = exp2f(regsFp32P14[int32_t{53}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{54}] = exp2f(regsFp32P14[int32_t{54}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{58}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{59}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{58}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{59}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{55}] = exp2f(regsFp32P14[int32_t{55}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{44}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{45}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{46}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{47}];
            // TmemP.h:745
            regsP[int32_t{11}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{56}] = exp2f(regsFp32P14[int32_t{56}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{60}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{61}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{60}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{61}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{57}] = exp2f(regsFp32P14[int32_t{57}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{58}] = exp2f(regsFp32P14[int32_t{58}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{62}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{63}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{62}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{63}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{59}] = exp2f(regsFp32P14[int32_t{59}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{48}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{49}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{50}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{51}];
            // TmemP.h:745
            regsP[int32_t{12}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{60}] = exp2f(regsFp32P14[int32_t{60}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{64}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{65}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{64}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{65}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{61}] = exp2f(regsFp32P14[int32_t{61}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{62}] = exp2f(regsFp32P14[int32_t{62}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{66}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{67}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{66}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{67}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{63}] = exp2f(regsFp32P14[int32_t{63}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{52}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{53}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{54}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{55}];
            // TmemP.h:745
            regsP[int32_t{13}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{64}] = exp2f(regsFp32P14[int32_t{64}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{68}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{69}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{68}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{69}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{65}] = exp2f(regsFp32P14[int32_t{65}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{66}] = exp2f(regsFp32P14[int32_t{66}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{70}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{71}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{70}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{71}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{67}] = exp2f(regsFp32P14[int32_t{67}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{56}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{57}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{58}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{59}];
            // TmemP.h:745
            regsP[int32_t{14}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{68}] = exp2f(regsFp32P14[int32_t{68}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{72}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{73}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{72}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{73}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{69}] = exp2f(regsFp32P14[int32_t{69}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{70}] = exp2f(regsFp32P14[int32_t{70}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{74}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{75}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{74}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{75}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{71}] = exp2f(regsFp32P14[int32_t{71}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{60}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{61}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{62}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{63}];
            // TmemP.h:745
            regsP[int32_t{15}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{72}] = exp2f(regsFp32P14[int32_t{72}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{76}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{77}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{76}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{77}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{73}] = exp2f(regsFp32P14[int32_t{73}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{74}] = exp2f(regsFp32P14[int32_t{74}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{78}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{79}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{78}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{79}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{75}] = exp2f(regsFp32P14[int32_t{75}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{64}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{65}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{66}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{67}];
            // TmemP.h:745
            regsP[int32_t{16}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{76}] = exp2f(regsFp32P14[int32_t{76}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{80}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{81}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{80}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{81}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{77}] = exp2f(regsFp32P14[int32_t{77}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{78}] = exp2f(regsFp32P14[int32_t{78}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{82}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{83}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{82}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{83}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{79}] = exp2f(regsFp32P14[int32_t{79}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{68}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{69}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{70}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{71}];
            // TmemP.h:745
            regsP[int32_t{17}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{80}] = exp2f(regsFp32P14[int32_t{80}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{84}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{85}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{84}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{85}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{81}] = exp2f(regsFp32P14[int32_t{81}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{82}] = exp2f(regsFp32P14[int32_t{82}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{86}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{87}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{86}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{87}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{83}] = exp2f(regsFp32P14[int32_t{83}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{72}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{73}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{74}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{75}];
            // TmemP.h:745
            regsP[int32_t{18}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{84}] = exp2f(regsFp32P14[int32_t{84}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{88}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{89}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{88}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{89}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{85}] = exp2f(regsFp32P14[int32_t{85}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{86}] = exp2f(regsFp32P14[int32_t{86}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{90}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{91}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{90}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{91}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{87}] = exp2f(regsFp32P14[int32_t{87}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{76}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{77}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{78}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{79}];
            // TmemP.h:745
            regsP[int32_t{19}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{88}] = exp2f(regsFp32P14[int32_t{88}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{92}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{93}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{92}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{93}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{89}] = exp2f(regsFp32P14[int32_t{89}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{90}] = exp2f(regsFp32P14[int32_t{90}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{94}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{95}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{94}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{95}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{91}] = exp2f(regsFp32P14[int32_t{91}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{80}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{81}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{82}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{83}];
            // TmemP.h:745
            regsP[int32_t{20}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{92}] = exp2f(regsFp32P14[int32_t{92}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{96}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{97}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{96}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{97}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{93}] = exp2f(regsFp32P14[int32_t{93}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{94}] = exp2f(regsFp32P14[int32_t{94}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{98}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{99}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{98}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{99}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{95}] = exp2f(regsFp32P14[int32_t{95}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{84}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{85}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{86}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{87}];
            // TmemP.h:745
            regsP[int32_t{21}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{96}] = exp2f(regsFp32P14[int32_t{96}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{100}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{101}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{100}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{101}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{97}] = exp2f(regsFp32P14[int32_t{97}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{98}] = exp2f(regsFp32P14[int32_t{98}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{102}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{103}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{102}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{103}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{99}] = exp2f(regsFp32P14[int32_t{99}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{88}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{89}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{90}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{91}];
            // TmemP.h:745
            regsP[int32_t{22}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{100}] = exp2f(regsFp32P14[int32_t{100}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{104}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{105}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{104}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{105}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{101}] = exp2f(regsFp32P14[int32_t{101}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{102}] = exp2f(regsFp32P14[int32_t{102}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{106}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{107}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{106}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{107}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{103}] = exp2f(regsFp32P14[int32_t{103}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{92}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{93}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{94}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{95}];
            // TmemP.h:745
            regsP[int32_t{23}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{104}] = exp2f(regsFp32P14[int32_t{104}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{108}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{109}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{108}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{109}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{105}] = exp2f(regsFp32P14[int32_t{105}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{106}] = exp2f(regsFp32P14[int32_t{106}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{110}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{111}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{110}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{111}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{107}] = exp2f(regsFp32P14[int32_t{107}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{96}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{97}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{98}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{99}];
            // TmemP.h:745
            regsP[int32_t{24}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{108}] = exp2f(regsFp32P14[int32_t{108}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{112}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{113}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{112}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{113}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{109}] = exp2f(regsFp32P14[int32_t{109}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{110}] = exp2f(regsFp32P14[int32_t{110}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{114}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{115}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{114}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{115}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{111}] = exp2f(regsFp32P14[int32_t{111}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{100}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{101}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{102}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{103}];
            // TmemP.h:745
            regsP[int32_t{25}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{112}] = exp2f(regsFp32P14[int32_t{112}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{116}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{117}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{116}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{117}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{113}] = exp2f(regsFp32P14[int32_t{113}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{114}] = exp2f(regsFp32P14[int32_t{114}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{118}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{119}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{118}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{119}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{115}] = exp2f(regsFp32P14[int32_t{115}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{104}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{105}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{106}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{107}];
            // TmemP.h:745
            regsP[int32_t{26}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{116}] = exp2f(regsFp32P14[int32_t{116}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{120}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{121}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{120}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{121}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{117}] = exp2f(regsFp32P14[int32_t{117}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{118}] = exp2f(regsFp32P14[int32_t{118}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{122}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{123}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{122}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{123}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{119}] = exp2f(regsFp32P14[int32_t{119}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{108}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{109}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{110}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{111}];
            // TmemP.h:745
            regsP[int32_t{27}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{120}] = exp2f(regsFp32P14[int32_t{120}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{124}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{125}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{124}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{125}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{121}] = exp2f(regsFp32P14[int32_t{121}]);
          // TmemP.h:1438
          tmemP0DstStack.mOrderedSequence.arrive();
          // TmemP.h:1773
          regsFp32P14[int32_t{122}] = exp2f(regsFp32P14[int32_t{122}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog214, scaleSoftmaxLog214};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{0}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P14[int32_t{126}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P14[int32_t{127}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P14[int32_t{126}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P14[int32_t{127}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P14[int32_t{123}] = exp2f(regsFp32P14[int32_t{123}]);
          // TmemP.h:1749
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{112}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{113}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{114}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{115}];
            // TmemP.h:745
            regsP[int32_t{28}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P14[int32_t{124}] = exp2f(regsFp32P14[int32_t{124}]);
          // TmemP.h:1843
          regsFp32P14[int32_t{125}] = exp2f(regsFp32P14[int32_t{125}]);
          // TmemP.h:1773
          regsFp32P14[int32_t{126}] = exp2f(regsFp32P14[int32_t{126}]);
          // TmemP.h:1843
          regsFp32P14[int32_t{127}] = exp2f(regsFp32P14[int32_t{127}]);
          // TmemP.h:1875
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{116}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{117}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{118}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{119}];
            // TmemP.h:745
            regsP[int32_t{29}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1875
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{120}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{121}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{122}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{123}];
            // TmemP.h:745
            regsP[int32_t{30}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1875
          {
            // TmemP.h:714
            float elt0;
            // TmemP.h:715
            float elt1;
            // TmemP.h:716
            float elt2;
            // TmemP.h:717
            float elt3;
            // TmemP.h:720
            elt0 = regsFp32P14[int32_t{124}];
            // TmemP.h:721
            elt1 = regsFp32P14[int32_t{125}];
            // TmemP.h:722
            elt2 = regsFp32P14[int32_t{126}];
            // TmemP.h:723
            elt3 = regsFp32P14[int32_t{127}];
            // TmemP.h:745
            regsP[int32_t{31}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemTile.cpp:836
          {
            // TmemTile.cpp:838
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:871
            uint32_t const(&srcSlice0)[32]{
              reinterpret_cast<uint32_t const(&)[32]>(regsP[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_st_32x32b(
              (tmemBasePtr) +
                (static_cast<uint32_t>(int32_t((tmemP0DstStack.mInstId) == (int32_t{0}))
                                         ? (int32_t{0})
                                         : (int32_t{128}))),
              srcSlice0);
          }
          // TmemP.h:1420
          cutlass::arch::fence_view_async_tmem_store();
        }
        //
        // tmemP0 [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{6}, Flags{3145856}].
        //
        // Task.cpp:4522
        {
          // Task.cpp:4540
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
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset1267) >= (int32_t{0})) {
            // Task.cpp:5100
            tmemSoftmaxLocal0ProdToken =
              tmemSoftmaxLocal0DstStack.mPipeline.producer_try_acquire(tmemSoftmaxLocal0ProdState);
          }
        }
        // Task.cpp:1607
        // Task.cpp:4288
        {
          // Task.cpp:4318
          tmemSoftmaxLocal0DstStack.mPipeline.producer_acquire(tmemSoftmaxLocal0ProdState,
                                                               tmemSoftmaxLocal0ProdToken);
        }
        //
        // tmemS0 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{8}, Flags{1024}].
        //
        // Task.cpp:2568
        if ((loopOffset1267) >= (int32_t{0})) {
          // Task.cpp:2596
          {
            // Task.cpp:2620
            tmemS0SrcStack.mPipeline.consumer_release(tmemS0ConsReleaseState);
          }
          // Task.cpp:43
          ++tmemS0ConsReleaseState;
        }
        //
        // tmemSoftmaxGlobal0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        // TmemSoftmax.h:545
        float* oldMaxPtr9;
        // TmemSoftmax.h:552
        float* sumPtr9;
        // TmemSoftmax.h:559
        float* newMaxPtr9;
        // TmemSoftmax.h:566
        float* pPtr9;
        // Task.cpp:1511
        oldMaxPtr9 = oldMaxPtr7;
        // Task.cpp:1511
        sumPtr9 = sumPtr7;
        // Task.cpp:1511
        newMaxPtr9 = newMaxPtr07;
        // Task.cpp:1511
        pPtr9 = qkPtr07;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // TmemSoftmax.h:773
          asm volatile("fence.proxy.async.shared::cta;");
          // TmemSoftmax.h:857
          float localSum{float{0}};
          // TmemSoftmax.h:889
          cutlass::Array<float, 2> sum0{float{0}, float{0}};
          // TmemSoftmax.h:890
          cutlass::Array<float, 2> sum1{float{0}, float{0}};
          // TmemSoftmax.h:891
          cutlass::Array<float, 2> sum2{float{0}, float{0}};
          // TmemSoftmax.h:892
          cutlass::Array<float, 2> sum3{float{0}, float{0}};
          // TmemSoftmax.h:901
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset2494 = int32_t{0}; loopOffset2494 < int32_t{128};
               loopOffset2494 += int32_t{8}) {
            // TmemSoftmax.h:917
            cutlass::Array<float, 2> vals0;
            // TmemSoftmax.h:928
            vals0[int32_t{0}] = pPtr9[loopOffset2494];
            // TmemSoftmax.h:929
            vals0[int32_t{1}] = pPtr9[(loopOffset2494) + (int32_t{1})];
            // TmemSoftmax.h:938
            sum0 = trtllm::dev::fadd2(sum0, vals0);
            // TmemSoftmax.h:917
            cutlass::Array<float, 2> vals1;
            // TmemSoftmax.h:928
            vals1[int32_t{0}] = pPtr9[(loopOffset2494) + (int32_t{2})];
            // TmemSoftmax.h:929
            vals1[int32_t{1}] = pPtr9[(loopOffset2494) + (int32_t{3})];
            // TmemSoftmax.h:938
            sum1 = trtllm::dev::fadd2(sum1, vals1);
            // TmemSoftmax.h:917
            cutlass::Array<float, 2> vals2;
            // TmemSoftmax.h:928
            vals2[int32_t{0}] = pPtr9[(loopOffset2494) + (int32_t{4})];
            // TmemSoftmax.h:929
            vals2[int32_t{1}] = pPtr9[(loopOffset2494) + (int32_t{5})];
            // TmemSoftmax.h:938
            sum2 = trtllm::dev::fadd2(sum2, vals2);
            // TmemSoftmax.h:917
            cutlass::Array<float, 2> vals3;
            // TmemSoftmax.h:928
            vals3[int32_t{0}] = pPtr9[(loopOffset2494) + (int32_t{6})];
            // TmemSoftmax.h:929
            vals3[int32_t{1}] = pPtr9[(loopOffset2494) + (int32_t{7})];
            // TmemSoftmax.h:938
            sum3 = trtllm::dev::fadd2(sum3, vals3);
          }
          // TmemSoftmax.h:942
          sum0 = trtllm::dev::fadd2(sum0, sum1);
          // TmemSoftmax.h:943
          sum2 = trtllm::dev::fadd2(sum2, sum3);
          // TmemSoftmax.h:944
          sum0 = trtllm::dev::fadd2(sum0, sum2);
          // TmemSoftmax.h:951
          localSum = (localSum) + ((sum0[int32_t{0}]) + (sum0[int32_t{1}]));
          // Common.h:88
          float scale{float{1}};
          // Common.h:92
          float maxDiff{(float{oldMaxPtr9[int32_t{0}]}) - (float{newMaxPtr9[int32_t{0}]})};
          // Common.h:99
          if ((maxDiff) != (float{0})) {
            // Common.h:105
            scale = exp2f((scaleSoftmaxLog29) * (maxDiff));
          }
          // TmemSoftmax.h:815
          float expScale{scale};
          // TmemSoftmax.h:821
          float oldSum{sumPtr9[int32_t{0}]};
          // TmemSoftmax.h:826
          sumPtr9[int32_t{0}] = (expScale) * (oldSum) + (localSum);
        }
        //
        // tmemSoftmaxLocal0 [ProdWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{10}, Flags{0}].
        //
        // Task.cpp:1511
        oldMaxPtr8 = oldMaxPtr7;
        // Task.cpp:1511
        sumPtr8 = sumPtr7;
        // Task.cpp:1511
        newMaxPtr8 = newMaxPtr07;
        // Task.cpp:1607
        if (isLastLoopIter) {
          // Task.cpp:5945
          int32_t index{tmemSoftmaxLocal0ProdState.index()};
          // TmemTile.cpp:373
          cutlass::Array<float, 2> stats;
          // TmemSoftmax.h:365
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset2534 = int32_t{0}; loopOffset2534 < int32_t{1}; ++loopOffset2534) {
            // TmemSoftmax.h:382
            stats[loopOffset2534] = sumPtr8[loopOffset2534];
            // TmemSoftmax.h:384
            stats[(loopOffset2534) + (int32_t{1})] = newMaxPtr8[loopOffset2534];
          }
          // TmemTile.cpp:836
          {
            // TmemTile.cpp:838
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:871
            uint32_t const(&srcSlice0)[2]{
              reinterpret_cast<uint32_t const(&)[2]>(stats[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_st_32x32b(
              (tmemBasePtr) +
                (static_cast<uint32_t>(int32_t((tmemSoftmaxLocal0DstStack.mInstId) == (int32_t{0}))
                                         ? (int32_t{256})
                                         : (int32_t{288}))),
              srcSlice0);
          }
          // TmemSoftmax.h:407
          cutlass::arch::fence_view_async_tmem_store();
        }
        //
        // tmemSoftmaxLocal0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{10}, Flags{0}].
        //
        // Task.cpp:1607
        if (isLastLoopIter) {
          // Task.cpp:4540
          {
            // Task.cpp:4556
            tmemSoftmaxLocal0DstStack.mPipeline.producer_commit(tmemSoftmaxLocal0ProdState);
          }
          // Task.cpp:43
          ++tmemSoftmaxLocal0ProdState;
        }
        //
        // tmemSoftmaxLocal0 [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{2621952}].
        //
        // Task.cpp:1607
        if (isLastLoopIter) {
          // Task.cpp:5078
          if ((loopOffset1267) >= (int32_t{0})) {
            // Task.cpp:5100
            tmemSoftmaxLocal0ProdToken =
              tmemSoftmaxLocal0DstStack.mPipeline.producer_try_acquire(tmemSoftmaxLocal0ProdState);
          }
        }
        // Task.cpp:1607
        if (isLastLoopIter) {
          // Task.cpp:4318
          tmemSoftmaxLocal0DstStack.mPipeline.producer_acquire(tmemSoftmaxLocal0ProdState,
                                                               tmemSoftmaxLocal0ProdToken);
        }
        //
        // tmemSoftmaxLocal0 [ProdWork (call 2), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{2621952}].
        //
        //
        // tmemSoftmaxLocal0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{2621952}].
        //
        // Task.cpp:1607
        if (isLastLoopIter) {
          // Task.cpp:4540
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
    // Task.cpp:3553
    ExitTileWithSignalingLabel:
    // Task.cpp:3560
    ExitTileWithoutSignalingLabel:
      // Task.cpp:5532
      auto newWorkTileInfoTuple{
        workIdStorageSrcStack.mScheduler.fetch_next_work(workIdStorageSrcStack.workTileInfo,
                                                         workIdStorageSrcStack.mPipeline,
                                                         workIdStorageConsState)};
      // Task.cpp:5534
      workIdStorageSrcStack.workTileInfo = cute::get<int32_t{0}>(newWorkTileInfoTuple);
      // Task.cpp:5542
      ++workIdStorageConsState;
      // Task.cpp:5644
      mCtaIdxX = workIdStorageSrcStack.workTileInfo.M_idx;
      // Task.cpp:5645
      mCtaIdxY = workIdStorageSrcStack.workTileInfo.N_idx;
      // Task.cpp:5646
      mCtaIdxZ = workIdStorageSrcStack.workTileInfo.L_idx;
    } while (workIdStorageSrcStack.workTileInfo.is_valid_tile);
  }
};
// Task.cpp:559
// Fmha.h:2245
struct SoftmaxTask1 {
  // Task.cpp:522
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:547
    return ((state.mWarpIdx) >= (int32_t{4})) && ((state.mWarpIdx) < (int32_t{8}));
  }
};
// Task.cpp:559
// Fmha.h:2402
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
  // FmhaTask.h:216
  int32_t mNumCtasKv;
  // FmhaTask.h:226
  int32_t mCtaIdxKv;
  // FmhaTask.h:224
  int32_t mCtaIdxQ;
  // FmhaTask.h:214
  int32_t mSeqLenKv;
  // FmhaTask.h:733
  int32_t mNumSkippedTilesKv;
  // Task.cpp:706
  uint32_t const mTmemBaseOffset;
  // Task.cpp:371
  int32_t const mWarpGrpThreadIdx;
  // Task.cpp:566
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
    , // Kernel.cpp:212
    mNumCtasKv{int32_t{1}}
    , // Kernel.cpp:210
    mCtaIdxKv{int32_t{0}}
    , // Task.cpp:283
    mCtaIdxQ{mCtaIdxX}
    , // FmhaTask.h:437
    mSeqLenKv{int32_t{
      params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                            ? ((((mHeadIdx) / (params.mNumHeadsQPerKv)) * (params.mBatchSize)) +
                               (mBatchIdx))
                            : (mBatchIdx)]}}
    , // Kernel.cpp:210
    mNumSkippedTilesKv{int32_t{0}}
    , // Kernel.cpp:2424
    mTmemBaseOffset{uint32_t{
      __shfl_sync(uint32_t{0xffffffff}, (*state.mTmemSwStatePtr), int32_t{0}, int32_t{32})}}
    , // Task.cpp:379
    mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))} {}
  // Task.cpp:522
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:547
    return ((state.mWarpIdx) >= (int32_t{8})) && ((state.mWarpIdx) < (int32_t{12}));
  }
  // Task.cpp:454
  inline __device__ void execute(fmha::KernelParams const& params,
                                 KernelState const& state,
                                 TmemCorr0Stack& tmemCorr0DstStack,
                                 TmemCorr1Stack& tmemCorr1DstStack,
                                 TmemSoftmaxLocal0Stack& tmemSoftmaxLocal0SrcStack,
                                 TmemSoftmaxLocal0Stack& tmemSoftmaxLocal1SrcStack,
                                 TmemOStack& tmemOSrcStack,
                                 WorkIdStorageSmem& workIdStorageSrcSmem,
                                 WorkIdStorageStack& workIdStorageSrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<88>{});
    // Task.cpp:2114
    trtllm::dev::CutlassCpAsyncPipeline<1, true>::PipelineState tmemSoftmaxLocal0ConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassCpAsyncPipeline<1, true>::PipelineState tmemSoftmaxLocal0ConsReleaseState{};
    // Task.cpp:2135
    int32_t tmemSoftmaxLocal0ConsToken{int32_t{0}};
    // Task.cpp:2114
    trtllm::dev::CutlassCpAsyncPipeline<1, true>::PipelineState tmemSoftmaxLocal1ConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassCpAsyncPipeline<1, true>::PipelineState tmemSoftmaxLocal1ConsReleaseState{};
    // Task.cpp:2135
    int32_t tmemSoftmaxLocal1ConsToken{int32_t{0}};
    // Task.cpp:2114
    trtllm::dev::CutlassUmmaAsyncPipeline<
      2,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState tmemOConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassUmmaAsyncPipeline<
      2,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState tmemOConsReleaseState{};
    // Task.cpp:2135
    int32_t tmemOConsToken{int32_t{0}};
    // Task.cpp:2114
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsReleaseState{};
    // Task.cpp:2135
    int32_t workIdStorageConsToken{int32_t{0}};
    // Task.cpp:5485
    do {
      // FmhaTask.h:333
      int32_t currSeqCtaIdx;
      // FmhaTask.h:342
      currSeqCtaIdx = workIdStorageSrcStack.workTileInfo.M_idx;
      // FmhaTask.h:357
      mHeadIdx = workIdStorageSrcStack.workTileInfo.N_idx;
      // FmhaTask.h:361
      mBatchIdx = workIdStorageSrcStack.workTileInfo.L_idx;
      // FmhaTask.h:139
      mSeqOffsetQ = int32_t(bool{params.ptrCumSeqLensQ == nullptr})
                      ? ((mBatchIdx) * (params.mMaxSeqLenQ))
                      : (int32_t{params.ptrCumSeqLensQ[mBatchIdx]});
      // FmhaTask.h:139
      mSeqLenQ = int32_t(bool{params.ptrCumSeqLensQ == nullptr})
                   ? (params.mMaxSeqLenQ)
                   : ((int32_t{params.ptrCumSeqLensQ[(mBatchIdx) + (int32_t{1})]}) - (mSeqOffsetQ));
      // FmhaTask.h:491
      mCtaIdxQ = currSeqCtaIdx;
      // FmhaTask.h:139
      mSeqLenKv = int32_t{
        params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                              ? ((((mHeadIdx) / (params.mNumHeadsQPerKv)) * (params.mBatchSize)) +
                                 (mBatchIdx))
                              : (mBatchIdx)]};
      // FmhaTask.h:582
      int32_t numLoopSteps;
      // FmhaTask.h:592
      int32_t diffKvQ{(mSeqLenKv) - (mSeqLenQ)};
      // FmhaTask.h:597
      int32_t validSeqLenKv;
      // Common.h:63
      if ((params.mChunkedAttentionSizeLog2) > (int32_t{0})) {
        // FmhaTask.h:748
        mNumSkippedTilesKv =
          (((((mCtaIdxQ) * (int32_t{256})) + (diffKvQ)) >> (params.mChunkedAttentionSizeLog2))
           << (params.mChunkedAttentionSizeLog2)) /
          (int32_t{128});
      } else {
        // FmhaTask.h:767
        mNumSkippedTilesKv =
          (int32_t{max(int32_t{0},
                       ((((mCtaIdxQ) * (int32_t{256})) + (diffKvQ)) + (int32_t{1})) -
                         (params.mAttentionWindowSize))}) /
          (int32_t{128});
      }
      // FmhaTask.h:603
      validSeqLenKv =
        (int32_t{min((((mCtaIdxQ) * (int32_t{256})) + (diffKvQ)) + (int32_t{256}), mSeqLenKv)}) -
        ((mNumSkippedTilesKv) * (int32_t{128}));
      // FmhaTask.h:616
      mNumCtasKv =
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, int32_t{1})};
      // FmhaTask.h:630
      if ((((mCtaIdxQ) * (int32_t{256})) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
        // FmhaTask.h:668
        int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
                         ((mNumCtasKv) * (int32_t{128}))};
        // FmhaTask.h:682
        numLoopSteps = numSteps;
      } else {
        // FmhaTask.h:648
        numLoopSteps = int32_t{0};
      }
      // Task.cpp:3203
      bool const hasOneLoopIter{(int32_t{0}) < (numLoopSteps)};
      // Task.cpp:3214
      int32_t lastLoopOffset{int32_t{0}};
      // TmemTile.cpp:373
      cutlass::Array<float, 2> frgStats8;
      // TmemTile.cpp:373
      cutlass::Array<float, 2> frgStats11;
      // TmemCorr.h:1135
      cudaGridDependencySynchronize();
      // TmemCorr.h:1158
      float scaleSoftmaxLog217;
      // TmemCorr.h:1163
      scaleSoftmaxLog217 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
                             ? (params.mScaleSoftmaxLog2)
                             : (float{params.ptrScaleSoftmaxLog2[int32_t{0}]});
      // TmemCorr.h:1135
      cudaGridDependencySynchronize();
      // TmemCorr.h:1158
      float scaleSoftmaxLog218;
      // TmemCorr.h:1163
      scaleSoftmaxLog218 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
                             ? (params.mScaleSoftmaxLog2)
                             : (float{params.ptrScaleSoftmaxLog2[int32_t{0}]});
      //
      // Hoist the first iter.
      //
      //
      // tmemSoftmaxLocal0 [ConsWait, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:1607
        if (hasOneLoopIter) {
          // Task.cpp:2780
          tmemSoftmaxLocal0ConsToken =
            tmemSoftmaxLocal0SrcStack.mPipeline.consumer_try_wait(tmemSoftmaxLocal0ConsState);
        }
        // Task.cpp:2848
        tmemSoftmaxLocal0SrcStack.mPipeline.consumer_wait(tmemSoftmaxLocal0ConsState,
                                                          tmemSoftmaxLocal0ConsToken);
      }
      //
      // tmemSoftmaxLocal0 [ConsWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemSoftmaxLocal0ConsState.index()};
        // Task.cpp:43
        ++tmemSoftmaxLocal0ConsState;
      }
      //
      // tmemSoftmaxLocal0 [ConsRelease, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:2596
        {
          // Task.cpp:2620
          tmemSoftmaxLocal0SrcStack.mPipeline.consumer_release(tmemSoftmaxLocal0ConsReleaseState);
        }
        // Task.cpp:43
        ++tmemSoftmaxLocal0ConsReleaseState;
      }
      //
      // tmemCorr0 [ProdWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{512}].
      //
      //
      // tmemSoftmaxLocal1 [ConsWait, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:1607
        if (hasOneLoopIter) {
          // Task.cpp:2780
          tmemSoftmaxLocal1ConsToken =
            tmemSoftmaxLocal1SrcStack.mPipeline.consumer_try_wait(tmemSoftmaxLocal1ConsState);
        }
        // Task.cpp:2848
        tmemSoftmaxLocal1SrcStack.mPipeline.consumer_wait(tmemSoftmaxLocal1ConsState,
                                                          tmemSoftmaxLocal1ConsToken);
      }
      //
      // tmemSoftmaxLocal1 [ConsWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemSoftmaxLocal1ConsState.index()};
        // Task.cpp:43
        ++tmemSoftmaxLocal1ConsState;
      }
      //
      // tmemSoftmaxLocal1 [ConsRelease, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:2596
        {
          // Task.cpp:2620
          tmemSoftmaxLocal1SrcStack.mPipeline.consumer_release(tmemSoftmaxLocal1ConsReleaseState);
        }
        // Task.cpp:43
        ++tmemSoftmaxLocal1ConsReleaseState;
      }
      //
      // tmemCorr1 [ProdWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{512}].
      //
      //
      // Loop body.
      //
      // Task.cpp:3392
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset2706 = int32_t{0}; loopOffset2706 < (numLoopSteps) - (int32_t{1});
           ++loopOffset2706) {
        //
        // tmemSoftmaxLocal0 [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
        //
        // Task.cpp:1607
        // Task.cpp:2816
        {
          // Task.cpp:1607
          // Task.cpp:2757
          {
            // Task.cpp:2780
            tmemSoftmaxLocal0ConsToken =
              tmemSoftmaxLocal0SrcStack.mPipeline.consumer_try_wait(tmemSoftmaxLocal0ConsState);
          }
          // Task.cpp:2848
          tmemSoftmaxLocal0SrcStack.mPipeline.consumer_wait(tmemSoftmaxLocal0ConsState,
                                                            tmemSoftmaxLocal0ConsToken);
        }
        //
        // tmemSoftmaxLocal0 [ConsWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
        //
        // TmemSoftmax.h:231
        float* statsPtr18;
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{tmemSoftmaxLocal0ConsState.index()};
          // TmemTile.cpp:527
          {
            // TmemTile.cpp:529
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:545
            uint32_t(&dstSlice0)[2]{reinterpret_cast<uint32_t(&)[2]>(frgStats8[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_32x32b(
              dstSlice0,
              (tmemBasePtr) +
                (static_cast<uint32_t>(int32_t((tmemSoftmaxLocal0SrcStack.mInstId) == (int32_t{0}))
                                         ? (int32_t{256})
                                         : (int32_t{288}))));
          }
          // TmemSoftmax.h:327
          statsPtr18 = &frgStats8[int32_t{0}];
          // TmemSoftmax.h:330
          cutlass::arch::fence_view_async_tmem_load();
          // Task.cpp:43
          ++tmemSoftmaxLocal0ConsState;
        }
        //
        // tmemSoftmaxLocal0 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
        //
        // Task.cpp:2568
        if ((loopOffset2706) >= (int32_t{0})) {
          // Task.cpp:2596
          {
            // Task.cpp:2620
            tmemSoftmaxLocal0SrcStack.mPipeline.consumer_release(tmemSoftmaxLocal0ConsReleaseState);
          }
          // Task.cpp:43
          ++tmemSoftmaxLocal0ConsReleaseState;
        }
        //
        // tmemO [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:2816
        {
          // Task.cpp:1607
          // Task.cpp:2757
          {
            // Task.cpp:2780
            tmemOConsToken = tmemOSrcStack.mPipeline.consumer_try_wait(tmemOConsState);
          }
          // Task.cpp:2848
          tmemOSrcStack.mPipeline.consumer_wait(tmemOConsState, tmemOConsToken);
        }
        //
        // tmemO [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{tmemOConsState.index()};
          // Task.cpp:43
          ++tmemOConsState;
        }
        //
        // tmemCorr0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
        //
        // TmemCorr.h:1193
        float* prodStatsPtr017;
        // Task.cpp:1511
        prodStatsPtr017 = statsPtr18;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // TmemCorr.h:289
          cutlass::Array<float, 1> scales17;
          // Common.h:88
          float scale{float{1}};
          // Common.h:92
          float maxDiff{(float{prodStatsPtr017[int32_t{0}]}) -
                        (float{prodStatsPtr017[int32_t{1}]})};
          // Common.h:99
          if ((maxDiff) != (float{0})) {
            // Common.h:105
            scale = exp2f((scaleSoftmaxLog217) * (maxDiff));
          }
          // TmemCorr.h:316
          scales17[int32_t{0}] = scale;
          // TmemCorr.h:1240
          bool skipsCorr{true};
          // TmemCorr.h:1258
          skipsCorr = (skipsCorr) && ((scales17[int32_t{0}]) == (float{1}));
          // TmemCorr.h:1266
          skipsCorr = __all_sync(uint32_t{-1}, skipsCorr);
          // TmemCorr.h:1268
          if (!(skipsCorr)) {
            //
            // The headDimStageIdx: 0.
            //
            // TmemCorr.h:1486
            CUTLASS_PRAGMA_UNROLL
            for (int32_t loopOffset2761 = int32_t{0}; loopOffset2761 < int32_t{64};
                 loopOffset2761 += int32_t{64}) {
              // TmemTile.cpp:373
              cutlass::Array<float, 64> tmemRegs017;
              // TmemTile.cpp:527
              {
                // TmemTile.cpp:529
                uint32_t tmemBasePtr{mTmemBaseOffset};
                // TmemTile.cpp:545
                uint32_t(&dstSlice0)[16]{
                  reinterpret_cast<uint32_t(&)[16]>(tmemRegs017[int32_t{0}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_ld_32x32b(
                  dstSlice0,
                  (tmemBasePtr) +
                    (static_cast<uint32_t>((int32_t{0x140 /*hi=0, lo=320*/}) + (loopOffset2761))));
                // TmemTile.cpp:545
                uint32_t(&dstSlice1)[16]{
                  reinterpret_cast<uint32_t(&)[16]>(tmemRegs017[int32_t{16}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_ld_32x32b(
                  dstSlice1,
                  (tmemBasePtr) +
                    (static_cast<uint32_t>(((int32_t{0x140 /*hi=0, lo=320*/}) + (loopOffset2761)) +
                                           (int32_t{0x10 /*hi=0, lo=16*/}))));
                // TmemTile.cpp:545
                uint32_t(&dstSlice2)[16]{
                  reinterpret_cast<uint32_t(&)[16]>(tmemRegs017[int32_t{32}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_ld_32x32b(
                  dstSlice2,
                  (tmemBasePtr) +
                    (static_cast<uint32_t>(((int32_t{0x140 /*hi=0, lo=320*/}) + (loopOffset2761)) +
                                           (int32_t{0x20 /*hi=0, lo=32*/}))));
                // TmemTile.cpp:545
                uint32_t(&dstSlice3)[16]{
                  reinterpret_cast<uint32_t(&)[16]>(tmemRegs017[int32_t{48}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_ld_32x32b(
                  dstSlice3,
                  (tmemBasePtr) +
                    (static_cast<uint32_t>(((int32_t{0x140 /*hi=0, lo=320*/}) + (loopOffset2761)) +
                                           (int32_t{0x30 /*hi=0, lo=48*/}))));
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{0}], tmemRegs017[int32_t{1}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{0}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{1}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{2}], tmemRegs017[int32_t{3}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{2}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{3}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{4}], tmemRegs017[int32_t{5}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{4}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{5}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{6}], tmemRegs017[int32_t{7}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{6}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{7}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{8}], tmemRegs017[int32_t{9}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{8}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{9}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{10}], tmemRegs017[int32_t{11}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{10}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{11}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{12}], tmemRegs017[int32_t{13}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{12}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{13}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{14}], tmemRegs017[int32_t{15}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{14}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{15}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{16}], tmemRegs017[int32_t{17}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{16}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{17}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{18}], tmemRegs017[int32_t{19}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{18}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{19}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{20}], tmemRegs017[int32_t{21}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{20}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{21}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{22}], tmemRegs017[int32_t{23}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{22}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{23}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{24}], tmemRegs017[int32_t{25}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{24}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{25}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{26}], tmemRegs017[int32_t{27}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{26}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{27}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{28}], tmemRegs017[int32_t{29}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{28}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{29}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{30}], tmemRegs017[int32_t{31}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{30}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{31}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{32}], tmemRegs017[int32_t{33}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{32}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{33}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{34}], tmemRegs017[int32_t{35}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{34}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{35}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{36}], tmemRegs017[int32_t{37}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{36}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{37}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{38}], tmemRegs017[int32_t{39}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{38}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{39}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{40}], tmemRegs017[int32_t{41}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{40}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{41}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{42}], tmemRegs017[int32_t{43}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{42}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{43}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{44}], tmemRegs017[int32_t{45}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{44}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{45}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{46}], tmemRegs017[int32_t{47}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{46}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{47}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{48}], tmemRegs017[int32_t{49}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{48}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{49}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{50}], tmemRegs017[int32_t{51}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{50}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{51}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{52}], tmemRegs017[int32_t{53}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{52}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{53}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{54}], tmemRegs017[int32_t{55}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{54}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{55}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{56}], tmemRegs017[int32_t{57}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{56}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{57}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{58}], tmemRegs017[int32_t{59}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{58}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{59}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{60}], tmemRegs017[int32_t{61}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{60}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{61}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{62}], tmemRegs017[int32_t{63}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs017[int32_t{62}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs017[int32_t{63}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1997
              {
                // TmemTile.cpp:836
                {
                  // TmemTile.cpp:838
                  uint32_t tmemBasePtr{mTmemBaseOffset};
                  // TmemTile.cpp:871
                  uint32_t const(&srcSlice0)[16]{
                    reinterpret_cast<uint32_t const(&)[16]>(tmemRegs017[int32_t{0}])};
                  // CudaPtx.h:48
                  cuda_ptx::tcgen05_st_32x32b(
                    (tmemBasePtr) +
                      (static_cast<uint32_t>((int32_t{0x140 /*hi=0, lo=320*/}) + (loopOffset2761))),
                    srcSlice0);
                  // TmemTile.cpp:871
                  uint32_t const(&srcSlice1)[16]{
                    reinterpret_cast<uint32_t const(&)[16]>(tmemRegs017[int32_t{16}])};
                  // CudaPtx.h:48
                  cuda_ptx::tcgen05_st_32x32b(
                    (tmemBasePtr) + (static_cast<uint32_t>(
                                      ((int32_t{0x140 /*hi=0, lo=320*/}) + (loopOffset2761)) +
                                      (int32_t{0x10 /*hi=0, lo=16*/}))),
                    srcSlice1);
                  // TmemTile.cpp:871
                  uint32_t const(&srcSlice2)[16]{
                    reinterpret_cast<uint32_t const(&)[16]>(tmemRegs017[int32_t{32}])};
                  // CudaPtx.h:48
                  cuda_ptx::tcgen05_st_32x32b(
                    (tmemBasePtr) + (static_cast<uint32_t>(
                                      ((int32_t{0x140 /*hi=0, lo=320*/}) + (loopOffset2761)) +
                                      (int32_t{0x20 /*hi=0, lo=32*/}))),
                    srcSlice2);
                  // TmemTile.cpp:871
                  uint32_t const(&srcSlice3)[16]{
                    reinterpret_cast<uint32_t const(&)[16]>(tmemRegs017[int32_t{48}])};
                  // CudaPtx.h:48
                  cuda_ptx::tcgen05_st_32x32b(
                    (tmemBasePtr) + (static_cast<uint32_t>(
                                      ((int32_t{0x140 /*hi=0, lo=320*/}) + (loopOffset2761)) +
                                      (int32_t{0x30 /*hi=0, lo=48*/}))),
                    srcSlice3);
                }
              }
            }
            // TmemCorr.h:1602
            cutlass::arch::fence_view_async_tmem_store();
          }
        }
        //
        // tmemO [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{1024}].
        //
        // Task.cpp:2568
        if ((loopOffset2706) >= (int32_t{0})) {
          // Task.cpp:2596
          {
            // Task.cpp:2620
            tmemOSrcStack.mPipeline.consumer_release(tmemOConsReleaseState);
          }
          // Task.cpp:43
          ++tmemOConsReleaseState;
        }
        //
        // tmemSoftmaxLocal1 [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
        //
        // Task.cpp:1607
        // Task.cpp:2816
        {
          // Task.cpp:1607
          // Task.cpp:2757
          {
            // Task.cpp:2780
            tmemSoftmaxLocal1ConsToken =
              tmemSoftmaxLocal1SrcStack.mPipeline.consumer_try_wait(tmemSoftmaxLocal1ConsState);
          }
          // Task.cpp:2848
          tmemSoftmaxLocal1SrcStack.mPipeline.consumer_wait(tmemSoftmaxLocal1ConsState,
                                                            tmemSoftmaxLocal1ConsToken);
        }
        //
        // tmemSoftmaxLocal1 [ConsWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
        //
        // TmemSoftmax.h:231
        float* statsPtr111;
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{tmemSoftmaxLocal1ConsState.index()};
          // TmemTile.cpp:527
          {
            // TmemTile.cpp:529
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:545
            uint32_t(&dstSlice0)[2]{reinterpret_cast<uint32_t(&)[2]>(frgStats11[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_32x32b(
              dstSlice0,
              (tmemBasePtr) +
                (static_cast<uint32_t>(int32_t((tmemSoftmaxLocal1SrcStack.mInstId) == (int32_t{0}))
                                         ? (int32_t{256})
                                         : (int32_t{288}))));
          }
          // TmemSoftmax.h:327
          statsPtr111 = &frgStats11[int32_t{0}];
          // TmemSoftmax.h:330
          cutlass::arch::fence_view_async_tmem_load();
          // Task.cpp:43
          ++tmemSoftmaxLocal1ConsState;
        }
        //
        // tmemSoftmaxLocal1 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
        //
        // Task.cpp:2568
        if ((loopOffset2706) >= (int32_t{0})) {
          // Task.cpp:2596
          {
            // Task.cpp:2620
            tmemSoftmaxLocal1SrcStack.mPipeline.consumer_release(tmemSoftmaxLocal1ConsReleaseState);
          }
          // Task.cpp:43
          ++tmemSoftmaxLocal1ConsReleaseState;
        }
        //
        // tmemO [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:2816
        {
          // Task.cpp:1607
          // Task.cpp:2757
          {
            // Task.cpp:2780
            tmemOConsToken = tmemOSrcStack.mPipeline.consumer_try_wait(tmemOConsState);
          }
          // Task.cpp:2848
          tmemOSrcStack.mPipeline.consumer_wait(tmemOConsState, tmemOConsToken);
        }
        //
        // tmemO [ConsWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{tmemOConsState.index()};
          // Task.cpp:43
          ++tmemOConsState;
        }
        //
        // tmemCorr1 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
        //
        // TmemCorr.h:1193
        float* prodStatsPtr018;
        // Task.cpp:1511
        prodStatsPtr018 = statsPtr111;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // TmemCorr.h:289
          cutlass::Array<float, 1> scales18;
          // Common.h:88
          float scale{float{1}};
          // Common.h:92
          float maxDiff{(float{prodStatsPtr018[int32_t{0}]}) -
                        (float{prodStatsPtr018[int32_t{1}]})};
          // Common.h:99
          if ((maxDiff) != (float{0})) {
            // Common.h:105
            scale = exp2f((scaleSoftmaxLog218) * (maxDiff));
          }
          // TmemCorr.h:316
          scales18[int32_t{0}] = scale;
          // TmemCorr.h:1240
          bool skipsCorr{true};
          // TmemCorr.h:1258
          skipsCorr = (skipsCorr) && ((scales18[int32_t{0}]) == (float{1}));
          // TmemCorr.h:1266
          skipsCorr = __all_sync(uint32_t{-1}, skipsCorr);
          // TmemCorr.h:1268
          if (!(skipsCorr)) {
            //
            // The headDimStageIdx: 0.
            //
            // TmemCorr.h:1486
            CUTLASS_PRAGMA_UNROLL
            for (int32_t loopOffset3040 = int32_t{0}; loopOffset3040 < int32_t{64};
                 loopOffset3040 += int32_t{64}) {
              // TmemTile.cpp:373
              cutlass::Array<float, 64> tmemRegs018;
              // TmemTile.cpp:527
              {
                // TmemTile.cpp:529
                uint32_t tmemBasePtr{mTmemBaseOffset};
                // TmemTile.cpp:545
                uint32_t(&dstSlice0)[16]{
                  reinterpret_cast<uint32_t(&)[16]>(tmemRegs018[int32_t{0}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_ld_32x32b(
                  dstSlice0,
                  (tmemBasePtr) +
                    (static_cast<uint32_t>((int32_t{0x180 /*hi=0, lo=384*/}) + (loopOffset3040))));
                // TmemTile.cpp:545
                uint32_t(&dstSlice1)[16]{
                  reinterpret_cast<uint32_t(&)[16]>(tmemRegs018[int32_t{16}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_ld_32x32b(
                  dstSlice1,
                  (tmemBasePtr) +
                    (static_cast<uint32_t>(((int32_t{0x180 /*hi=0, lo=384*/}) + (loopOffset3040)) +
                                           (int32_t{0x10 /*hi=0, lo=16*/}))));
                // TmemTile.cpp:545
                uint32_t(&dstSlice2)[16]{
                  reinterpret_cast<uint32_t(&)[16]>(tmemRegs018[int32_t{32}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_ld_32x32b(
                  dstSlice2,
                  (tmemBasePtr) +
                    (static_cast<uint32_t>(((int32_t{0x180 /*hi=0, lo=384*/}) + (loopOffset3040)) +
                                           (int32_t{0x20 /*hi=0, lo=32*/}))));
                // TmemTile.cpp:545
                uint32_t(&dstSlice3)[16]{
                  reinterpret_cast<uint32_t(&)[16]>(tmemRegs018[int32_t{48}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_ld_32x32b(
                  dstSlice3,
                  (tmemBasePtr) +
                    (static_cast<uint32_t>(((int32_t{0x180 /*hi=0, lo=384*/}) + (loopOffset3040)) +
                                           (int32_t{0x30 /*hi=0, lo=48*/}))));
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{0}], tmemRegs018[int32_t{1}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{0}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{1}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{2}], tmemRegs018[int32_t{3}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{2}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{3}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{4}], tmemRegs018[int32_t{5}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{4}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{5}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{6}], tmemRegs018[int32_t{7}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{6}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{7}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{8}], tmemRegs018[int32_t{9}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{8}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{9}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{10}], tmemRegs018[int32_t{11}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{10}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{11}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{12}], tmemRegs018[int32_t{13}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{12}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{13}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{14}], tmemRegs018[int32_t{15}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{14}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{15}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{16}], tmemRegs018[int32_t{17}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{16}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{17}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{18}], tmemRegs018[int32_t{19}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{18}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{19}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{20}], tmemRegs018[int32_t{21}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{20}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{21}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{22}], tmemRegs018[int32_t{23}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{22}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{23}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{24}], tmemRegs018[int32_t{25}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{24}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{25}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{26}], tmemRegs018[int32_t{27}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{26}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{27}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{28}], tmemRegs018[int32_t{29}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{28}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{29}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{30}], tmemRegs018[int32_t{31}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{30}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{31}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{32}], tmemRegs018[int32_t{33}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{32}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{33}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{34}], tmemRegs018[int32_t{35}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{34}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{35}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{36}], tmemRegs018[int32_t{37}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{36}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{37}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{38}], tmemRegs018[int32_t{39}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{38}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{39}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{40}], tmemRegs018[int32_t{41}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{40}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{41}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{42}], tmemRegs018[int32_t{43}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{42}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{43}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{44}], tmemRegs018[int32_t{45}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{44}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{45}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{46}], tmemRegs018[int32_t{47}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{46}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{47}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{48}], tmemRegs018[int32_t{49}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{48}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{49}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{50}], tmemRegs018[int32_t{51}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{50}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{51}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{52}], tmemRegs018[int32_t{53}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{52}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{53}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{54}], tmemRegs018[int32_t{55}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{54}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{55}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{56}], tmemRegs018[int32_t{57}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{56}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{57}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{58}], tmemRegs018[int32_t{59}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{58}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{59}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{60}], tmemRegs018[int32_t{61}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{60}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{61}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{62}], tmemRegs018[int32_t{63}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs018[int32_t{62}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs018[int32_t{63}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1997
              {
                // TmemTile.cpp:836
                {
                  // TmemTile.cpp:838
                  uint32_t tmemBasePtr{mTmemBaseOffset};
                  // TmemTile.cpp:871
                  uint32_t const(&srcSlice0)[16]{
                    reinterpret_cast<uint32_t const(&)[16]>(tmemRegs018[int32_t{0}])};
                  // CudaPtx.h:48
                  cuda_ptx::tcgen05_st_32x32b(
                    (tmemBasePtr) +
                      (static_cast<uint32_t>((int32_t{0x180 /*hi=0, lo=384*/}) + (loopOffset3040))),
                    srcSlice0);
                  // TmemTile.cpp:871
                  uint32_t const(&srcSlice1)[16]{
                    reinterpret_cast<uint32_t const(&)[16]>(tmemRegs018[int32_t{16}])};
                  // CudaPtx.h:48
                  cuda_ptx::tcgen05_st_32x32b(
                    (tmemBasePtr) + (static_cast<uint32_t>(
                                      ((int32_t{0x180 /*hi=0, lo=384*/}) + (loopOffset3040)) +
                                      (int32_t{0x10 /*hi=0, lo=16*/}))),
                    srcSlice1);
                  // TmemTile.cpp:871
                  uint32_t const(&srcSlice2)[16]{
                    reinterpret_cast<uint32_t const(&)[16]>(tmemRegs018[int32_t{32}])};
                  // CudaPtx.h:48
                  cuda_ptx::tcgen05_st_32x32b(
                    (tmemBasePtr) + (static_cast<uint32_t>(
                                      ((int32_t{0x180 /*hi=0, lo=384*/}) + (loopOffset3040)) +
                                      (int32_t{0x20 /*hi=0, lo=32*/}))),
                    srcSlice2);
                  // TmemTile.cpp:871
                  uint32_t const(&srcSlice3)[16]{
                    reinterpret_cast<uint32_t const(&)[16]>(tmemRegs018[int32_t{48}])};
                  // CudaPtx.h:48
                  cuda_ptx::tcgen05_st_32x32b(
                    (tmemBasePtr) + (static_cast<uint32_t>(
                                      ((int32_t{0x180 /*hi=0, lo=384*/}) + (loopOffset3040)) +
                                      (int32_t{0x30 /*hi=0, lo=48*/}))),
                    srcSlice3);
                }
              }
            }
            // TmemCorr.h:1602
            cutlass::arch::fence_view_async_tmem_store();
          }
        }
        //
        // tmemO [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{1024}].
        //
        // Task.cpp:2568
        if ((loopOffset2706) >= (int32_t{0})) {
          // Task.cpp:2596
          {
            // Task.cpp:2620
            tmemOSrcStack.mPipeline.consumer_release(tmemOConsReleaseState);
          }
          // Task.cpp:43
          ++tmemOConsReleaseState;
        }
        //
        // tmemSoftmaxLocal0 [ConsTailWork, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        // Task.cpp:3499
        lastLoopOffset = loopOffset2706;
      }
      //
      // Pull the last iter down.
      //
      // Task.cpp:3534
      if (((numLoopSteps) - (int32_t{1})) > (int32_t{0})) {
        // Task.cpp:3535
        ++lastLoopOffset;
      }
      //
      // tmemSoftmaxLocal0 [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:1607
        if (hasOneLoopIter) {
          // Task.cpp:2780
          tmemSoftmaxLocal0ConsToken =
            tmemSoftmaxLocal0SrcStack.mPipeline.consumer_try_wait(tmemSoftmaxLocal0ConsState);
        }
        // Task.cpp:2848
        tmemSoftmaxLocal0SrcStack.mPipeline.consumer_wait(tmemSoftmaxLocal0ConsState,
                                                          tmemSoftmaxLocal0ConsToken);
      }
      //
      // tmemSoftmaxLocal0 [ConsWork (call 2), LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
      //
      // TmemSoftmax.h:231
      float* statsPtr28;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemSoftmaxLocal0ConsState.index()};
        // TmemTile.cpp:527
        {
          // TmemTile.cpp:529
          uint32_t tmemBasePtr{mTmemBaseOffset};
          // TmemTile.cpp:545
          uint32_t(&dstSlice0)[2]{reinterpret_cast<uint32_t(&)[2]>(frgStats8[int32_t{0}])};
          // CudaPtx.h:48
          cuda_ptx::tcgen05_ld_32x32b(
            dstSlice0,
            (tmemBasePtr) +
              (static_cast<uint32_t>(int32_t((tmemSoftmaxLocal0SrcStack.mInstId) == (int32_t{0}))
                                       ? (int32_t{256})
                                       : (int32_t{288}))));
        }
        // TmemSoftmax.h:327
        statsPtr28 = &frgStats8[int32_t{0}];
        // TmemSoftmax.h:330
        cutlass::arch::fence_view_async_tmem_load();
        // Task.cpp:43
        ++tmemSoftmaxLocal0ConsState;
      }
      //
      // tmemSoftmaxLocal0 [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:2596
        {
          // Task.cpp:2620
          tmemSoftmaxLocal0SrcStack.mPipeline.consumer_release(tmemSoftmaxLocal0ConsReleaseState);
        }
        // Task.cpp:43
        ++tmemSoftmaxLocal0ConsReleaseState;
      }
      //
      // tmemO [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{34}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:1607
        if (hasOneLoopIter) {
          // Task.cpp:2780
          tmemOConsToken = tmemOSrcStack.mPipeline.consumer_try_wait(tmemOConsState);
        }
        // Task.cpp:2848
        tmemOSrcStack.mPipeline.consumer_wait(tmemOConsState, tmemOConsToken);
      }
      //
      // tmemO [ConsWork (call 2), LastIter, FreqInfo{0, 1}, UserTags{34}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemOConsState.index()};
        // Task.cpp:43
        ++tmemOConsState;
      }
      //
      // tmemCorr0 [ProdWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // TmemCorr.h:1193
      float* prodStatsPtr117;
      // Task.cpp:1511
      prodStatsPtr117 = statsPtr28;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // TmemCorr.h:2860
        int32_t instIdxO{(mCtaIdxQ) * (int32_t{2})};
        // TmemCorr.h:2946
        bool const isInBoundsOut{(mWarpGrpThreadIdx) <
                                 ((mSeqLenQ) - ((instIdxO) * (int32_t{128})))};
        // TmemCorr.h:2965
        int32_t threadSeqOffsetO;
        // TmemCorr.h:2966
        int32_t threadHeadOffsetO;
        // TmemCorr.h:2976
        trtllm::dev::mapRowToHeadTokenIdx<false, true>(mWarpGrpThreadIdx,
                                                       threadHeadOffsetO,
                                                       threadSeqOffsetO,
                                                       params.mNumHeadsQPerKv);
        // TmemCorr.h:2984
        int32_t seqOffsetO{(mSeqOffsetQ) + ((instIdxO) * (int32_t{128}))};
        // TmemCorr.h:2989
        int32_t headIdxO;
        // TmemCorr.h:2993
        headIdxO = (mHeadIdx) + (threadHeadOffsetO);
        // TmemCorr.h:2996
        int32_t headOffsetO{((mHeadIdx) + (threadHeadOffsetO)) * (int32_t{64})};
        // TmemCorr.h:3027
        int64_t ctaOffsetO{(static_cast<int64_t>(seqOffsetO)) *
                             (static_cast<int64_t>((params.mNumHeadsQ) * (int32_t{64}))) +
                           (static_cast<int64_t>(headOffsetO))};
        // TmemCorr.h:3041
        cutlass::bfloat16_t* ptrO{reinterpret_cast<cutlass::bfloat16_t*>(params.ptrO)};
        // TmemCorr.h:3046
        ptrO = ptrO + (ctaOffsetO);
        // TmemCorr.h:3061
        int32_t threadOffsetO{(threadSeqOffsetO) * ((params.mNumHeadsQ) * (int32_t{64}))};
        // TmemCorr.h:3063
        ptrO = ptrO + (threadOffsetO);
        // TmemCorr.h:3076
        bool storesSoftmaxStats{reinterpret_cast<float*>(params.ptrSoftmaxStats) != nullptr};
        // TmemCorr.h:3082
        float* ptrSoftmaxStats;
        // TmemCorr.h:3084
        if (storesSoftmaxStats) {
          // TmemCorr.h:3088
          ptrSoftmaxStats = reinterpret_cast<float*>(params.ptrSoftmaxStats) +
                            (((seqOffsetO) * (params.mNumHeadsQ) + (mHeadIdx)) * (int32_t{2}));
        }
        // TmemCorr.h:289
        cutlass::Array<float, 1> scales17;
        // TmemCorr.h:330
        float finalSum17{prodStatsPtr117[int32_t{0}]};
        // TmemCorr.h:338
        float finalMax17{prodStatsPtr117[int32_t{1}]};
        // TmemCorr.h:1859
        float attentionSinkVal{int32_t{0}};
        // TmemCorr.h:1864
        if (bool{params.ptrAttentionSinks != nullptr}) {
          // TmemCorr.h:1884
          attentionSinkVal =
            (float{exp2f(((float{params.ptrAttentionSinks[headIdxO]}) * (float{1.442695})) -
                         ((finalMax17) * (scaleSoftmaxLog217)))}) *
            (float{448});
        }
        // TmemCorr.h:384
        prodStatsPtr117[int32_t{0}] = (finalSum17) + (attentionSinkVal);
        // TmemCorr.h:398
        scales17[int32_t{0}] = (float(bool{params.ptrOutputScale == nullptr})
                                  ? (params.mOutputScale)
                                  : (float{params.ptrOutputScale[int32_t{0}]})) /
                               ((finalSum17) + (attentionSinkVal));
        // TmemCorr.h:3898
        if (storesSoftmaxStats) {
          // TmemCorr.h:3902
          float scaleSoftmax{(scaleSoftmaxLog217) * (float{0.69314718})};
          // TmemCorr.h:3915
          (prodStatsPtr117 + int32_t{1})[int32_t{0}] =
            (float{(prodStatsPtr117 + int32_t{1})[int32_t{0}]}) * (scaleSoftmax);
          // TmemCorr.h:3926
          prodStatsPtr117[int32_t{0}] = (float{prodStatsPtr117[int32_t{0}]}) * (float{0.002232143});
          // TmemCorr.h:3981
          trtllm::dev::storeStatsForAb((prodStatsPtr117 + int32_t{1}),
                                       prodStatsPtr117,
                                       ptrSoftmaxStats,
                                       (threadSeqOffsetO) * (params.mNumHeadsQ) +
                                         (threadHeadOffsetO),
                                       mWarpGrpThreadIdx,
                                       true,
                                       (mSeqLenQ) - ((instIdxO) * (int32_t{128})));
        }
        //
        // The headDimStageIdx: 0.
        //
        // TmemCorr.h:1486
        for (int32_t loopOffset3363 = int32_t{0}; loopOffset3363 < int32_t{64};
             loopOffset3363 += int32_t{8}) {
          // TmemTile.cpp:373
          cutlass::Array<float, 8> tmemRegs017;
          // TmemTile.cpp:527
          {
            // TmemTile.cpp:529
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:545
            uint32_t(&dstSlice0)[8]{reinterpret_cast<uint32_t(&)[8]>(tmemRegs017[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_32x32b(
              dstSlice0,
              (tmemBasePtr) +
                (static_cast<uint32_t>((int32_t{0x140 /*hi=0, lo=320*/}) + (loopOffset3363))));
          }
          // TmemCorr.h:3438
          uint32_t mRegsO17[4];
          // TmemCorr.h:1534
          {
            // TmemCorr.h:1554
            cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
            // TmemCorr.h:1565
            cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{0}], tmemRegs017[int32_t{1}]};
            // TmemCorr.h:1577
            vals0 = trtllm::dev::fmul2(vals0, localScales0);
            // TmemCorr.h:1580
            tmemRegs017[int32_t{0}] = vals0[int32_t{0}];
            // TmemCorr.h:1581
            tmemRegs017[int32_t{1}] = vals0[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO17[int32_t{0}] = trtllm::dev::convert_float2_to_bfloat16(tmemRegs017[int32_t{0}],
                                                                           tmemRegs017[int32_t{1}]);
          }
          // TmemCorr.h:1534
          {
            // TmemCorr.h:1554
            cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
            // TmemCorr.h:1565
            cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{2}], tmemRegs017[int32_t{3}]};
            // TmemCorr.h:1577
            vals0 = trtllm::dev::fmul2(vals0, localScales0);
            // TmemCorr.h:1580
            tmemRegs017[int32_t{2}] = vals0[int32_t{0}];
            // TmemCorr.h:1581
            tmemRegs017[int32_t{3}] = vals0[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO17[int32_t{1}] = trtllm::dev::convert_float2_to_bfloat16(tmemRegs017[int32_t{2}],
                                                                           tmemRegs017[int32_t{3}]);
          }
          // TmemCorr.h:1534
          {
            // TmemCorr.h:1554
            cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
            // TmemCorr.h:1565
            cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{4}], tmemRegs017[int32_t{5}]};
            // TmemCorr.h:1577
            vals0 = trtllm::dev::fmul2(vals0, localScales0);
            // TmemCorr.h:1580
            tmemRegs017[int32_t{4}] = vals0[int32_t{0}];
            // TmemCorr.h:1581
            tmemRegs017[int32_t{5}] = vals0[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO17[int32_t{2}] = trtllm::dev::convert_float2_to_bfloat16(tmemRegs017[int32_t{4}],
                                                                           tmemRegs017[int32_t{5}]);
          }
          // TmemCorr.h:1534
          {
            // TmemCorr.h:1554
            cutlass::Array<float, 2> localScales0{scales17[int32_t{0}], scales17[int32_t{0}]};
            // TmemCorr.h:1565
            cutlass::Array<float, 2> vals0{tmemRegs017[int32_t{6}], tmemRegs017[int32_t{7}]};
            // TmemCorr.h:1577
            vals0 = trtllm::dev::fmul2(vals0, localScales0);
            // TmemCorr.h:1580
            tmemRegs017[int32_t{6}] = vals0[int32_t{0}];
            // TmemCorr.h:1581
            tmemRegs017[int32_t{7}] = vals0[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO17[int32_t{3}] = trtllm::dev::convert_float2_to_bfloat16(tmemRegs017[int32_t{6}],
                                                                           tmemRegs017[int32_t{7}]);
          }
          // TmemCorr.h:3744
          if (isInBoundsOut) {
            // Utils.h:647
            trtllm::dev::storeVec((ptrO + loopOffset3363), mRegsO17);
          }
        }
      }
      //
      // tmemO [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{34}, Flags{1024}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:2596
        {
          // Task.cpp:2620
          tmemOSrcStack.mPipeline.consumer_release(tmemOConsReleaseState);
        }
        // Task.cpp:43
        ++tmemOConsReleaseState;
      }
      //
      // tmemSoftmaxLocal1 [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:1607
        if (hasOneLoopIter) {
          // Task.cpp:2780
          tmemSoftmaxLocal1ConsToken =
            tmemSoftmaxLocal1SrcStack.mPipeline.consumer_try_wait(tmemSoftmaxLocal1ConsState);
        }
        // Task.cpp:2848
        tmemSoftmaxLocal1SrcStack.mPipeline.consumer_wait(tmemSoftmaxLocal1ConsState,
                                                          tmemSoftmaxLocal1ConsToken);
      }
      //
      // tmemSoftmaxLocal1 [ConsWork (call 2), LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
      //
      // TmemSoftmax.h:231
      float* statsPtr211;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemSoftmaxLocal1ConsState.index()};
        // TmemTile.cpp:527
        {
          // TmemTile.cpp:529
          uint32_t tmemBasePtr{mTmemBaseOffset};
          // TmemTile.cpp:545
          uint32_t(&dstSlice0)[2]{reinterpret_cast<uint32_t(&)[2]>(frgStats11[int32_t{0}])};
          // CudaPtx.h:48
          cuda_ptx::tcgen05_ld_32x32b(
            dstSlice0,
            (tmemBasePtr) +
              (static_cast<uint32_t>(int32_t((tmemSoftmaxLocal1SrcStack.mInstId) == (int32_t{0}))
                                       ? (int32_t{256})
                                       : (int32_t{288}))));
        }
        // TmemSoftmax.h:327
        statsPtr211 = &frgStats11[int32_t{0}];
        // TmemSoftmax.h:330
        cutlass::arch::fence_view_async_tmem_load();
        // Task.cpp:43
        ++tmemSoftmaxLocal1ConsState;
      }
      //
      // tmemSoftmaxLocal1 [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:2596
        {
          // Task.cpp:2620
          tmemSoftmaxLocal1SrcStack.mPipeline.consumer_release(tmemSoftmaxLocal1ConsReleaseState);
        }
        // Task.cpp:43
        ++tmemSoftmaxLocal1ConsReleaseState;
      }
      //
      // tmemO [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{36}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:1607
        if (hasOneLoopIter) {
          // Task.cpp:2780
          tmemOConsToken = tmemOSrcStack.mPipeline.consumer_try_wait(tmemOConsState);
        }
        // Task.cpp:2848
        tmemOSrcStack.mPipeline.consumer_wait(tmemOConsState, tmemOConsToken);
      }
      //
      // tmemO [ConsWork (call 3), LastIter, FreqInfo{0, 1}, UserTags{36}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemOConsState.index()};
        // Task.cpp:43
        ++tmemOConsState;
      }
      //
      // tmemCorr1 [ProdWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // TmemCorr.h:1193
      float* prodStatsPtr118;
      // Task.cpp:1511
      prodStatsPtr118 = statsPtr211;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // TmemCorr.h:2860
        int32_t instIdxO{(mCtaIdxQ) * (int32_t{2}) + (int32_t{1})};
        // TmemCorr.h:2946
        bool const isInBoundsOut{(mWarpGrpThreadIdx) <
                                 ((mSeqLenQ) - ((instIdxO) * (int32_t{128})))};
        // TmemCorr.h:2965
        int32_t threadSeqOffsetO;
        // TmemCorr.h:2966
        int32_t threadHeadOffsetO;
        // TmemCorr.h:2976
        trtllm::dev::mapRowToHeadTokenIdx<false, true>(mWarpGrpThreadIdx,
                                                       threadHeadOffsetO,
                                                       threadSeqOffsetO,
                                                       params.mNumHeadsQPerKv);
        // TmemCorr.h:2984
        int32_t seqOffsetO{(mSeqOffsetQ) + ((instIdxO) * (int32_t{128}))};
        // TmemCorr.h:2989
        int32_t headIdxO;
        // TmemCorr.h:2993
        headIdxO = (mHeadIdx) + (threadHeadOffsetO);
        // TmemCorr.h:2996
        int32_t headOffsetO{((mHeadIdx) + (threadHeadOffsetO)) * (int32_t{64})};
        // TmemCorr.h:3027
        int64_t ctaOffsetO{(static_cast<int64_t>(seqOffsetO)) *
                             (static_cast<int64_t>((params.mNumHeadsQ) * (int32_t{64}))) +
                           (static_cast<int64_t>(headOffsetO))};
        // TmemCorr.h:3041
        cutlass::bfloat16_t* ptrO{reinterpret_cast<cutlass::bfloat16_t*>(params.ptrO)};
        // TmemCorr.h:3046
        ptrO = ptrO + (ctaOffsetO);
        // TmemCorr.h:3061
        int32_t threadOffsetO{(threadSeqOffsetO) * ((params.mNumHeadsQ) * (int32_t{64}))};
        // TmemCorr.h:3063
        ptrO = ptrO + (threadOffsetO);
        // TmemCorr.h:3076
        bool storesSoftmaxStats{reinterpret_cast<float*>(params.ptrSoftmaxStats) != nullptr};
        // TmemCorr.h:3082
        float* ptrSoftmaxStats;
        // TmemCorr.h:3084
        if (storesSoftmaxStats) {
          // TmemCorr.h:3088
          ptrSoftmaxStats = reinterpret_cast<float*>(params.ptrSoftmaxStats) +
                            (((seqOffsetO) * (params.mNumHeadsQ) + (mHeadIdx)) * (int32_t{2}));
        }
        // TmemCorr.h:289
        cutlass::Array<float, 1> scales18;
        // TmemCorr.h:330
        float finalSum18{prodStatsPtr118[int32_t{0}]};
        // TmemCorr.h:338
        float finalMax18{prodStatsPtr118[int32_t{1}]};
        // TmemCorr.h:1859
        float attentionSinkVal{int32_t{0}};
        // TmemCorr.h:1864
        if (bool{params.ptrAttentionSinks != nullptr}) {
          // TmemCorr.h:1884
          attentionSinkVal =
            (float{exp2f(((float{params.ptrAttentionSinks[headIdxO]}) * (float{1.442695})) -
                         ((finalMax18) * (scaleSoftmaxLog218)))}) *
            (float{448});
        }
        // TmemCorr.h:384
        prodStatsPtr118[int32_t{0}] = (finalSum18) + (attentionSinkVal);
        // TmemCorr.h:398
        scales18[int32_t{0}] = (float(bool{params.ptrOutputScale == nullptr})
                                  ? (params.mOutputScale)
                                  : (float{params.ptrOutputScale[int32_t{0}]})) /
                               ((finalSum18) + (attentionSinkVal));
        // TmemCorr.h:3898
        if (storesSoftmaxStats) {
          // TmemCorr.h:3902
          float scaleSoftmax{(scaleSoftmaxLog218) * (float{0.69314718})};
          // TmemCorr.h:3915
          (prodStatsPtr118 + int32_t{1})[int32_t{0}] =
            (float{(prodStatsPtr118 + int32_t{1})[int32_t{0}]}) * (scaleSoftmax);
          // TmemCorr.h:3926
          prodStatsPtr118[int32_t{0}] = (float{prodStatsPtr118[int32_t{0}]}) * (float{0.002232143});
          // TmemCorr.h:3981
          trtllm::dev::storeStatsForAb((prodStatsPtr118 + int32_t{1}),
                                       prodStatsPtr118,
                                       ptrSoftmaxStats,
                                       (threadSeqOffsetO) * (params.mNumHeadsQ) +
                                         (threadHeadOffsetO),
                                       mWarpGrpThreadIdx,
                                       true,
                                       (mSeqLenQ) - ((instIdxO) * (int32_t{128})));
        }
        //
        // The headDimStageIdx: 0.
        //
        // TmemCorr.h:1486
        for (int32_t loopOffset3502 = int32_t{0}; loopOffset3502 < int32_t{64};
             loopOffset3502 += int32_t{8}) {
          // TmemTile.cpp:373
          cutlass::Array<float, 8> tmemRegs018;
          // TmemTile.cpp:527
          {
            // TmemTile.cpp:529
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:545
            uint32_t(&dstSlice0)[8]{reinterpret_cast<uint32_t(&)[8]>(tmemRegs018[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_32x32b(
              dstSlice0,
              (tmemBasePtr) +
                (static_cast<uint32_t>((int32_t{0x180 /*hi=0, lo=384*/}) + (loopOffset3502))));
          }
          // TmemCorr.h:3438
          uint32_t mRegsO18[4];
          // TmemCorr.h:1534
          {
            // TmemCorr.h:1554
            cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
            // TmemCorr.h:1565
            cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{0}], tmemRegs018[int32_t{1}]};
            // TmemCorr.h:1577
            vals0 = trtllm::dev::fmul2(vals0, localScales0);
            // TmemCorr.h:1580
            tmemRegs018[int32_t{0}] = vals0[int32_t{0}];
            // TmemCorr.h:1581
            tmemRegs018[int32_t{1}] = vals0[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO18[int32_t{0}] = trtllm::dev::convert_float2_to_bfloat16(tmemRegs018[int32_t{0}],
                                                                           tmemRegs018[int32_t{1}]);
          }
          // TmemCorr.h:1534
          {
            // TmemCorr.h:1554
            cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
            // TmemCorr.h:1565
            cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{2}], tmemRegs018[int32_t{3}]};
            // TmemCorr.h:1577
            vals0 = trtllm::dev::fmul2(vals0, localScales0);
            // TmemCorr.h:1580
            tmemRegs018[int32_t{2}] = vals0[int32_t{0}];
            // TmemCorr.h:1581
            tmemRegs018[int32_t{3}] = vals0[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO18[int32_t{1}] = trtllm::dev::convert_float2_to_bfloat16(tmemRegs018[int32_t{2}],
                                                                           tmemRegs018[int32_t{3}]);
          }
          // TmemCorr.h:1534
          {
            // TmemCorr.h:1554
            cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
            // TmemCorr.h:1565
            cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{4}], tmemRegs018[int32_t{5}]};
            // TmemCorr.h:1577
            vals0 = trtllm::dev::fmul2(vals0, localScales0);
            // TmemCorr.h:1580
            tmemRegs018[int32_t{4}] = vals0[int32_t{0}];
            // TmemCorr.h:1581
            tmemRegs018[int32_t{5}] = vals0[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO18[int32_t{2}] = trtllm::dev::convert_float2_to_bfloat16(tmemRegs018[int32_t{4}],
                                                                           tmemRegs018[int32_t{5}]);
          }
          // TmemCorr.h:1534
          {
            // TmemCorr.h:1554
            cutlass::Array<float, 2> localScales0{scales18[int32_t{0}], scales18[int32_t{0}]};
            // TmemCorr.h:1565
            cutlass::Array<float, 2> vals0{tmemRegs018[int32_t{6}], tmemRegs018[int32_t{7}]};
            // TmemCorr.h:1577
            vals0 = trtllm::dev::fmul2(vals0, localScales0);
            // TmemCorr.h:1580
            tmemRegs018[int32_t{6}] = vals0[int32_t{0}];
            // TmemCorr.h:1581
            tmemRegs018[int32_t{7}] = vals0[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO18[int32_t{3}] = trtllm::dev::convert_float2_to_bfloat16(tmemRegs018[int32_t{6}],
                                                                           tmemRegs018[int32_t{7}]);
          }
          // TmemCorr.h:3744
          if (isInBoundsOut) {
            // Utils.h:647
            trtllm::dev::storeVec((ptrO + loopOffset3502), mRegsO18);
          }
        }
      }
      //
      // tmemO [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{36}, Flags{1024}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:2596
        {
          // Task.cpp:2620
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
      // Task.cpp:2703
      {}
    // Task.cpp:3553
    ExitTileWithSignalingLabel:
    // Task.cpp:3560
    ExitTileWithoutSignalingLabel:
      // Task.cpp:5532
      auto newWorkTileInfoTuple{
        workIdStorageSrcStack.mScheduler.fetch_next_work(workIdStorageSrcStack.workTileInfo,
                                                         workIdStorageSrcStack.mPipeline,
                                                         workIdStorageConsState)};
      // Task.cpp:5534
      workIdStorageSrcStack.workTileInfo = cute::get<int32_t{0}>(newWorkTileInfoTuple);
      // Task.cpp:5542
      ++workIdStorageConsState;
      // Task.cpp:5644
      mCtaIdxX = workIdStorageSrcStack.workTileInfo.M_idx;
      // Task.cpp:5645
      mCtaIdxY = workIdStorageSrcStack.workTileInfo.N_idx;
      // Task.cpp:5646
      mCtaIdxZ = workIdStorageSrcStack.workTileInfo.L_idx;
    } while (workIdStorageSrcStack.workTileInfo.is_valid_tile);
    // Task.cpp:512
    cudaTriggerProgrammaticLaunchCompletion();
  }
};
// Task.cpp:559
// Fmha.h:2537
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
  // FmhaTask.h:216
  int32_t mNumCtasKv;
  // FmhaTask.h:226
  int32_t mCtaIdxKv;
  // FmhaTask.h:224
  int32_t mCtaIdxQ;
  // FmhaTask.h:214
  int32_t mSeqLenKv;
  // FmhaTask.h:733
  int32_t mNumSkippedTilesKv;
  // Task.cpp:706
  uint32_t const mTmemBaseOffset;
  // Task.cpp:566
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
    , // Kernel.cpp:212
    mNumCtasKv{int32_t{1}}
    , // Kernel.cpp:210
    mCtaIdxKv{int32_t{0}}
    , // Task.cpp:283
    mCtaIdxQ{mCtaIdxX}
    , // FmhaTask.h:437
    mSeqLenKv{int32_t{
      params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                            ? ((((mHeadIdx) / (params.mNumHeadsQPerKv)) * (params.mBatchSize)) +
                               (mBatchIdx))
                            : (mBatchIdx)]}}
    , // Kernel.cpp:210
    mNumSkippedTilesKv{int32_t{0}}
    , // Kernel.cpp:2424
    mTmemBaseOffset{uint32_t{
      __shfl_sync(uint32_t{0xffffffff}, (*state.mTmemSwStatePtr), int32_t{0}, int32_t{32})}} {}
  // Task.cpp:522
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:547
    return ((state.mWarpIdx) >= (int32_t{12})) && ((state.mWarpIdx) < (int32_t{13}));
  }
  // Task.cpp:454
  inline __device__ void execute(fmha::KernelParams const& params,
                                 KernelState const& state,
                                 TmemS0Stack& tmemS0DstStack,
                                 TmemS0Stack& tmemS1DstStack,
                                 TmemOStack& tmemODstStack,
                                 SmemQSmem& smemQSrcSmem,
                                 SmemQStack& smemQSrcStack,
                                 SmemKvSmem& smemKvSrcSmem,
                                 SmemKvStack& smemKvSrcStack,
                                 TmemP0Stack& tmemP0SrcStack,
                                 TmemP0Stack& tmemP1SrcStack,
                                 WorkIdStorageSmem& workIdStorageSrcSmem,
                                 WorkIdStorageStack& workIdStorageSrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<56>{});
    // Task.cpp:2114
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      2,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemQConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      2,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemQConsReleaseState{};
    // Task.cpp:2135
    int32_t smemQConsToken{int32_t{0}};
    // Task.cpp:2114
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      3,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemKvConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      3,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState
      smemKvConsReleaseState{};
    // Task.cpp:2135
    int32_t smemKvConsToken{int32_t{0}};
    // Task.cpp:2114
    trtllm::dev::CutlassCpAsyncPipeline<1, true>::PipelineState tmemP0ConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassCpAsyncPipeline<1, true>::PipelineState tmemP0ConsReleaseState{};
    // Task.cpp:2135
    int32_t tmemP0ConsToken{int32_t{0}};
    // Task.cpp:2114
    trtllm::dev::CutlassCpAsyncPipeline<1, true>::PipelineState tmemP1ConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassCpAsyncPipeline<1, true>::PipelineState tmemP1ConsReleaseState{};
    // Task.cpp:2135
    int32_t tmemP1ConsToken{int32_t{0}};
    // Task.cpp:2114
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsReleaseState{};
    // Task.cpp:2135
    int32_t workIdStorageConsToken{int32_t{0}};
    // Task.cpp:2013
    trtllm::dev::CutlassUmmaAsyncPipeline<1,
                                          cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState tmemS0ProdState{int32_t{0}, int32_t{1}, int32_t{0}};
    // Task.cpp:2033
    int32_t tmemS0ProdToken{int32_t{1}};
    // Task.cpp:2013
    trtllm::dev::CutlassUmmaAsyncPipeline<1,
                                          cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState tmemS1ProdState{int32_t{0}, int32_t{1}, int32_t{0}};
    // Task.cpp:2033
    int32_t tmemS1ProdToken{int32_t{1}};
    // Task.cpp:2013
    trtllm::dev::CutlassUmmaAsyncPipeline<2,
                                          cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState tmemOProdState{int32_t{0}, int32_t{1}, int32_t{0}};
    // Task.cpp:2033
    int32_t tmemOProdToken{int32_t{1}};
    //
    // tmemS0 [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{64}].
    //
    // Task.cpp:1607
    // Task.cpp:5064
    {
      // Task.cpp:5078
      {
        // Task.cpp:5100
        tmemS0ProdToken = tmemS0DstStack.mPipeline.producer_try_acquire(tmemS0ProdState);
      }
    }
    // Task.cpp:1607
    // Task.cpp:4288
    {
      // Task.cpp:4318
      tmemS0DstStack.mPipeline.producer_acquire(tmemS0ProdState, tmemS0ProdToken);
    }
    //
    // tmemS1 [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{64}].
    //
    // Task.cpp:1607
    // Task.cpp:5064
    {
      // Task.cpp:5078
      {
        // Task.cpp:5100
        tmemS1ProdToken = tmemS1DstStack.mPipeline.producer_try_acquire(tmemS1ProdState);
      }
    }
    // Task.cpp:1607
    // Task.cpp:4288
    {
      // Task.cpp:4318
      tmemS1DstStack.mPipeline.producer_acquire(tmemS1ProdState, tmemS1ProdToken);
    }
    // Task.cpp:5485
    do {
      // FmhaTask.h:333
      int32_t currSeqCtaIdx;
      // FmhaTask.h:342
      currSeqCtaIdx = workIdStorageSrcStack.workTileInfo.M_idx;
      // FmhaTask.h:357
      mHeadIdx = workIdStorageSrcStack.workTileInfo.N_idx;
      // FmhaTask.h:361
      mBatchIdx = workIdStorageSrcStack.workTileInfo.L_idx;
      // FmhaTask.h:139
      mSeqOffsetQ = int32_t(bool{params.ptrCumSeqLensQ == nullptr})
                      ? ((mBatchIdx) * (params.mMaxSeqLenQ))
                      : (int32_t{params.ptrCumSeqLensQ[mBatchIdx]});
      // FmhaTask.h:139
      mSeqLenQ = int32_t(bool{params.ptrCumSeqLensQ == nullptr})
                   ? (params.mMaxSeqLenQ)
                   : ((int32_t{params.ptrCumSeqLensQ[(mBatchIdx) + (int32_t{1})]}) - (mSeqOffsetQ));
      // FmhaTask.h:491
      mCtaIdxQ = currSeqCtaIdx;
      // FmhaTask.h:139
      mSeqLenKv = int32_t{
        params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                              ? ((((mHeadIdx) / (params.mNumHeadsQPerKv)) * (params.mBatchSize)) +
                                 (mBatchIdx))
                              : (mBatchIdx)]};
      // FmhaTask.h:582
      int32_t numLoopSteps;
      // FmhaTask.h:592
      int32_t diffKvQ{(mSeqLenKv) - (mSeqLenQ)};
      // FmhaTask.h:597
      int32_t validSeqLenKv;
      // Common.h:63
      if ((params.mChunkedAttentionSizeLog2) > (int32_t{0})) {
        // FmhaTask.h:748
        mNumSkippedTilesKv =
          (((((mCtaIdxQ) * (int32_t{256})) + (diffKvQ)) >> (params.mChunkedAttentionSizeLog2))
           << (params.mChunkedAttentionSizeLog2)) /
          (int32_t{128});
      } else {
        // FmhaTask.h:767
        mNumSkippedTilesKv =
          (int32_t{max(int32_t{0},
                       ((((mCtaIdxQ) * (int32_t{256})) + (diffKvQ)) + (int32_t{1})) -
                         (params.mAttentionWindowSize))}) /
          (int32_t{128});
      }
      // FmhaTask.h:603
      validSeqLenKv =
        (int32_t{min((((mCtaIdxQ) * (int32_t{256})) + (diffKvQ)) + (int32_t{256}), mSeqLenKv)}) -
        ((mNumSkippedTilesKv) * (int32_t{128}));
      // FmhaTask.h:616
      mNumCtasKv =
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, int32_t{1})};
      // FmhaTask.h:630
      if ((((mCtaIdxQ) * (int32_t{256})) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
        // FmhaTask.h:668
        int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
                         ((mNumCtasKv) * (int32_t{128}))};
        // FmhaTask.h:682
        numLoopSteps = numSteps;
      } else {
        // FmhaTask.h:648
        numLoopSteps = int32_t{0};
      }
      // Task.cpp:3203
      bool const hasOneLoopIter{(int32_t{0}) < (numLoopSteps)};
      // Task.cpp:3214
      int32_t lastLoopOffset{int32_t{0}};
      // SmemKv.h:169
      int32_t smemIndicesK[1];
      // SmemKv.h:181
      int32_t smemIndicesV[1];
      //
      // Hoist the first iter.
      //
      //
      // smemQ [ConsWait, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:1607
        if (hasOneLoopIter) {
          // Task.cpp:2780
          smemQConsToken = smemQSrcStack.mPipeline.consumer_try_wait(smemQConsState);
        }
        // Task.cpp:2848
        smemQSrcStack.mPipeline.consumer_wait(smemQConsState, smemQConsToken);
      }
      //
      // smemQ [ConsWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // SmemQ.h:151
      cutlass::float_e4m3_t* smemPtrQ0_2{&smemQSrcSmem.mArray[int32_t{0}][int32_t{0}]};
      // SmemQ.h:159
      int32_t smemIdxQ0_2;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{smemQConsState.index()};
        // SmemQ.h:188
        smemIdxQ0_2 = index;
        // Task.cpp:43
        ++smemQConsState;
      }
      //
      // tmemS0 [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{64}].
      //
      //
      // smemKv [ConsWait, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:1607
        if (hasOneLoopIter) {
          // Task.cpp:2780
          smemKvConsToken = smemKvSrcStack.mPipeline.consumer_try_wait(smemKvConsState);
        }
        // Task.cpp:2848
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
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{smemKvConsState.index()};
        // SmemKv.h:267
        smemPtrK3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:298
        smemIndicesK[int32_t{0}] = index;
        // SmemKv.h:304
        smemIdxK3 = smemIndicesK[int32_t{0}];
        // Task.cpp:43
        ++smemKvConsState;
      }
      //
      // tmemS0 [ProdWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // TmemS.h:1130
      cutlass::float_e4m3_t* smemPtrQ7;
      // TmemS.h:1135
      int32_t smemIdxQ7;
      // TmemS.h:1141
      cutlass::float_e4m3_t* smemPtrK7;
      // TmemS.h:1146
      int32_t memIdxK7;
      // Task.cpp:1511
      smemPtrQ7 = smemPtrQ0_2;
      // Task.cpp:1511
      smemIdxQ7 = smemIdxQ0_2;
      // Task.cpp:1511
      smemPtrK7 = smemPtrK3;
      // Task.cpp:1511
      memIdxK7 = smemIdxK3;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemS0ProdState.index()};
        // TmemS.h:1889
        cutlass::float_e4m3_t* smemQ{smemPtrQ7};
        // TmemS.h:1910
        smemQ += (smemIdxQ7) * (int32_t{8192});
        // TmemS.h:1938
        cutlass::float_e4m3_t* smemK{smemPtrK7};
        // TmemS.h:1944
        smemK += (memIdxK7) * (int32_t{8192});
        // Mma.cpp:618
        {
          // TmemTile.cpp:1765
          uint32_t tmemPtrD{int32_t((tmemS0DstStack.mInstId) == (int32_t{0})) ? (int32_t{0})
                                                                              : (int32_t{128})};
          //
          // leadingDimInBytes = 8192, strideInBytes = 512, swizzleMode = 2.
          //
          // Mma.cpp:203
          uint64_t smemDescA{
            trtllm::dev::createSmemDesc(smemQ,
                                        uint32_t{0x2000000 /*hi=512, lo=0*/},
                                        uint32_t{0x80004020 /*hi=32768, lo=16416*/})};
          //
          // leadingDimInBytes = 8192, strideInBytes = 512, swizzleMode = 2.
          //
          // Mma.cpp:203
          uint64_t smemDescB{
            trtllm::dev::createSmemDesc(smemK,
                                        uint32_t{0x2000000 /*hi=512, lo=0*/},
                                        uint32_t{0x80004020 /*hi=32768, lo=16416*/})};
          //
          // MMA inst for mi=0 ni=0 ki=0.
          //
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{128},
                                                                  int32_t{32},
                                                                  false,
                                                                  true)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{128},
                                                                  int32_t{32},
                                                                  false,
                                                                  true)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f8f6f4,
                                  cuda_ptx::cta_group_1,
                                  tmemPtrD,
                                  smemDescA,
                                  smemDescB,
                                  utcmmaDesc_0_0_1,
                                  bool{true});
          }
        }
      }
      //
      // tmemS0 [ProdCommit, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4540
        {
          // Task.cpp:4556
          tmemS0DstStack.mPipeline.producer_commit(tmemS0ProdState);
        }
        // Task.cpp:43
        ++tmemS0ProdState;
      }
      //
      // smemQ [ConsWait, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:1607
        if (hasOneLoopIter) {
          // Task.cpp:2780
          smemQConsToken = smemQSrcStack.mPipeline.consumer_try_wait(smemQConsState);
        }
        // Task.cpp:2848
        smemQSrcStack.mPipeline.consumer_wait(smemQConsState, smemQConsToken);
      }
      //
      // smemQ [ConsWork (call 1), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // SmemQ.h:151
      cutlass::float_e4m3_t* smemPtrQ1_2{&smemQSrcSmem.mArray[int32_t{0}][int32_t{0}]};
      // SmemQ.h:159
      int32_t smemIdxQ1_2;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{smemQConsState.index()};
        // SmemQ.h:188
        smemIdxQ1_2 = index;
        // Task.cpp:43
        ++smemQConsState;
      }
      //
      // tmemS1 [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{64}].
      //
      //
      // smemKv [ConsWork (call 1), FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{262144}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{smemKvConsState.index()};
        // SmemKv.h:267
        smemPtrK3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:304
        smemIdxK3 = smemIndicesK[int32_t{0}];
      }
      //
      // tmemS1 [ProdWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // TmemS.h:1130
      cutlass::float_e4m3_t* smemPtrQ10;
      // TmemS.h:1135
      int32_t smemIdxQ10;
      // TmemS.h:1141
      cutlass::float_e4m3_t* smemPtrK10;
      // TmemS.h:1146
      int32_t memIdxK10;
      // Task.cpp:1511
      smemPtrQ10 = smemPtrQ1_2;
      // Task.cpp:1511
      smemIdxQ10 = smemIdxQ1_2;
      // Task.cpp:1511
      smemPtrK10 = smemPtrK3;
      // Task.cpp:1511
      memIdxK10 = smemIdxK3;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemS1ProdState.index()};
        // TmemS.h:1889
        cutlass::float_e4m3_t* smemQ{smemPtrQ10};
        // TmemS.h:1910
        smemQ += (smemIdxQ10) * (int32_t{8192});
        // TmemS.h:1938
        cutlass::float_e4m3_t* smemK{smemPtrK10};
        // TmemS.h:1944
        smemK += (memIdxK10) * (int32_t{8192});
        // Mma.cpp:618
        {
          // TmemTile.cpp:1765
          uint32_t tmemPtrD{int32_t((tmemS1DstStack.mInstId) == (int32_t{0})) ? (int32_t{0})
                                                                              : (int32_t{128})};
          //
          // leadingDimInBytes = 8192, strideInBytes = 512, swizzleMode = 2.
          //
          // Mma.cpp:203
          uint64_t smemDescA{
            trtllm::dev::createSmemDesc(smemQ,
                                        uint32_t{0x2000000 /*hi=512, lo=0*/},
                                        uint32_t{0x80004020 /*hi=32768, lo=16416*/})};
          //
          // leadingDimInBytes = 8192, strideInBytes = 512, swizzleMode = 2.
          //
          // Mma.cpp:203
          uint64_t smemDescB{
            trtllm::dev::createSmemDesc(smemK,
                                        uint32_t{0x2000000 /*hi=512, lo=0*/},
                                        uint32_t{0x80004020 /*hi=32768, lo=16416*/})};
          //
          // MMA inst for mi=0 ni=0 ki=0.
          //
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{128},
                                                                  int32_t{32},
                                                                  false,
                                                                  true)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
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
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{128},
                                                                  int32_t{32},
                                                                  false,
                                                                  true)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
            cuda_ptx::tcgen05_mma(cuda_ptx::kind_f8f6f4,
                                  cuda_ptx::cta_group_1,
                                  tmemPtrD,
                                  smemDescA,
                                  smemDescB,
                                  utcmmaDesc_0_0_1,
                                  bool{true});
          }
        }
      }
      //
      // smemKv [ConsRelease, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:2596
        {
          // Task.cpp:2620
          smemKvSrcStack.mPipeline.consumer_release(smemKvConsReleaseState);
        }
        // Task.cpp:43
        ++smemKvConsReleaseState;
      }
      //
      // tmemS1 [ProdCommit, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4540
        {
          // Task.cpp:4556
          tmemS1DstStack.mPipeline.producer_commit(tmemS1ProdState);
        }
        // Task.cpp:43
        ++tmemS1ProdState;
      }
      //
      // Loop body.
      //
      // Task.cpp:3392
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset3832 = int32_t{0}; loopOffset3832 < (numLoopSteps) - (int32_t{1});
           ++loopOffset3832) {
        //
        // smemKv [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:2816
        {
          // Task.cpp:1607
          // Task.cpp:2757
          {
            // Task.cpp:2780
            smemKvConsToken = smemKvSrcStack.mPipeline.consumer_try_wait(smemKvConsState);
          }
          // Task.cpp:2848
          smemKvSrcStack.mPipeline.consumer_wait(smemKvConsState, smemKvConsToken);
        }
        //
        // smemKv [ConsWork (call 2), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{smemKvConsState.index()};
          // SmemKv.h:322
          smemPtrV3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
          // SmemKv.h:366
          smemIndicesV[int32_t{0}] = index;
          // SmemKv.h:372
          smemIdxV3 = smemIndicesV[int32_t{0}];
          // Task.cpp:43
          ++smemKvConsState;
        }
        //
        // tmemP0 [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{49152}].
        //
        //
        // Skipped by flag SkipsConsWait.
        //
        //
        // tmemP0 [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{49152}].
        //
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{tmemP0ConsState.index()};
          // Task.cpp:43
          ++tmemP0ConsState;
        }
        //
        // tmemP0 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{49152}].
        //
        // Task.cpp:2568
        if ((loopOffset3832) >= (int32_t{0})) {
          // Task.cpp:2596
          {
            //
            // Skipped by flag SkipsConsRelease.
            //
          }
          // Task.cpp:43
          ++tmemP0ConsReleaseState;
        }
        //
        // tmemO [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{26}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset3832) >= (int32_t{0})) {
            // Task.cpp:5100
            tmemOProdToken = tmemODstStack.mPipeline.producer_try_acquire(tmemOProdState);
          }
        }
        // Task.cpp:1607
        // Task.cpp:4288
        {
          // Task.cpp:4318
          tmemODstStack.mPipeline.producer_acquire(tmemOProdState, tmemOProdToken);
        }
        //
        // tmemS0 [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{1024}].
        //
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset3832) >= (int32_t{0})) {
            // Task.cpp:5100
            tmemS0ProdToken = tmemS0DstStack.mPipeline.producer_try_acquire(tmemS0ProdState);
          }
        }
        // Task.cpp:1607
        // Task.cpp:4288
        {
          // Task.cpp:4318
          tmemS0DstStack.mPipeline.producer_acquire(tmemS0ProdState, tmemS0ProdToken);
        }
        //
        // tmemO [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{26}, Flags{0}].
        //
        // TmemO.h:277
        cutlass::float_e4m3_t* smemPtrV16;
        // TmemO.h:282
        int32_t memIdxV16;
        // Task.cpp:1511
        smemPtrV16 = smemPtrV3;
        // Task.cpp:1511
        memIdxV16 = smemIdxV3;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{tmemOProdState.index()};
          // TmemO.h:493
          cutlass::float_e4m3_t* smemV{smemPtrV16};
          // TmemO.h:505
          smemV = smemV + ((memIdxV16) * (int32_t{8192}));
          // TmemO.h:535
          bool readD{true};
          // TmemO.h:545
          if ((loopOffset3832) == (int32_t{0})) {
            // TmemO.h:547
            readD = false;
          }
          // Mma.cpp:618
          {
            // TmemTile.cpp:1765
            uint32_t tmemPtrD{(mTmemBaseOffset) + (uint32_t{320})};
            // TmemTile.cpp:1521
            uint32_t tmemPtrA{mTmemBaseOffset};
            //
            // leadingDimInBytes = 0, strideInBytes = 512, swizzleMode = 2.
            //
            // Mma.cpp:203
            uint64_t smemDescB{
              trtllm::dev::createSmemDesc(smemV,
                                          uint32_t{0x0 /*hi=0, lo=0*/},
                                          uint32_t{0x80004020 /*hi=32768, lo=16416*/})};
            //
            // MMA inst for mi=0 ni=0 ki=0.
            //
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    true,
                                                                    int32_t{128},
                                                                    int32_t{64},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f8f6f4,
                                           cuda_ptx::cta_group_1,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_0,
                                           bool{readD});
            }
            //
            // MMA inst for mi=0 ni=0 ki=1.
            //
            // TmemTile.cpp:2041
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    true,
                                                                    int32_t{128},
                                                                    int32_t{64},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f8f6f4,
                                           cuda_ptx::cta_group_1,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_1,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=2.
            //
            // TmemTile.cpp:2041
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    true,
                                                                    int32_t{128},
                                                                    int32_t{64},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f8f6f4,
                                           cuda_ptx::cta_group_1,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_2,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=3.
            //
            // TmemTile.cpp:2041
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    true,
                                                                    int32_t{128},
                                                                    int32_t{64},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f8f6f4,
                                           cuda_ptx::cta_group_1,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_3,
                                           bool{true});
            }
          }
        }
        //
        // tmemO [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{26}, Flags{0}].
        //
        // Task.cpp:4522
        {
          // Task.cpp:4540
          {
            // Task.cpp:4556
            tmemODstStack.mPipeline.producer_commit(tmemOProdState);
          }
          // Task.cpp:43
          ++tmemOProdState;
        }
        //
        // smemKv [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:2816
        {
          // Task.cpp:1607
          // Task.cpp:2757
          {
            // Task.cpp:2780
            smemKvConsToken = smemKvSrcStack.mPipeline.consumer_try_wait(smemKvConsState);
          }
          // Task.cpp:2848
          smemKvSrcStack.mPipeline.consumer_wait(smemKvConsState, smemKvConsToken);
        }
        //
        // smemKv [ConsWork (call 3), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{smemKvConsState.index()};
          // SmemKv.h:267
          smemPtrK3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
          // SmemKv.h:298
          smemIndicesK[int32_t{0}] = index;
          // SmemKv.h:304
          smemIdxK3 = smemIndicesK[int32_t{0}];
          // Task.cpp:43
          ++smemKvConsState;
        }
        //
        // tmemS0 [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
        //
        // Task.cpp:1511
        smemPtrQ7 = smemPtrQ0_2;
        // Task.cpp:1511
        smemIdxQ7 = smemIdxQ0_2;
        // Task.cpp:1511
        smemPtrK7 = smemPtrK3;
        // Task.cpp:1511
        memIdxK7 = smemIdxK3;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{tmemS0ProdState.index()};
          // TmemS.h:1889
          cutlass::float_e4m3_t* smemQ{smemPtrQ7};
          // TmemS.h:1910
          smemQ += (smemIdxQ7) * (int32_t{8192});
          // TmemS.h:1938
          cutlass::float_e4m3_t* smemK{smemPtrK7};
          // TmemS.h:1944
          smemK += (memIdxK7) * (int32_t{8192});
          // Mma.cpp:618
          {
            // TmemTile.cpp:1765
            uint32_t tmemPtrD{int32_t((tmemS0DstStack.mInstId) == (int32_t{0})) ? (int32_t{0})
                                                                                : (int32_t{128})};
            //
            // leadingDimInBytes = 8192, strideInBytes = 512, swizzleMode = 2.
            //
            // Mma.cpp:203
            uint64_t smemDescA{
              trtllm::dev::createSmemDesc(smemQ,
                                          uint32_t{0x2000000 /*hi=512, lo=0*/},
                                          uint32_t{0x80004020 /*hi=32768, lo=16416*/})};
            //
            // leadingDimInBytes = 8192, strideInBytes = 512, swizzleMode = 2.
            //
            // Mma.cpp:203
            uint64_t smemDescB{
              trtllm::dev::createSmemDesc(smemK,
                                          uint32_t{0x2000000 /*hi=512, lo=0*/},
                                          uint32_t{0x80004020 /*hi=32768, lo=16416*/})};
            //
            // MMA inst for mi=0 ni=0 ki=0.
            //
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{128},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
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
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{128},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma(cuda_ptx::kind_f8f6f4,
                                    cuda_ptx::cta_group_1,
                                    tmemPtrD,
                                    smemDescA,
                                    smemDescB,
                                    utcmmaDesc_0_0_1,
                                    bool{true});
            }
          }
        }
        //
        // tmemS0 [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
        //
        // Task.cpp:4522
        {
          // Task.cpp:4540
          {
            // Task.cpp:4556
            tmemS0DstStack.mPipeline.producer_commit(tmemS0ProdState);
          }
          // Task.cpp:43
          ++tmemS0ProdState;
        }
        //
        // smemKv [ConsWork (call 4), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{262144}].
        //
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{smemKvConsState.index()};
          // SmemKv.h:322
          smemPtrV3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
          // SmemKv.h:372
          smemIdxV3 = smemIndicesV[int32_t{0}];
        }
        //
        // tmemP1 [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{49152}].
        //
        //
        // Skipped by flag SkipsConsWait.
        //
        //
        // tmemP1 [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{49152}].
        //
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{tmemP1ConsState.index()};
          // Task.cpp:43
          ++tmemP1ConsState;
        }
        //
        // tmemP1 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{49152}].
        //
        // Task.cpp:2568
        if ((loopOffset3832) >= (int32_t{0})) {
          // Task.cpp:2596
          {
            //
            // Skipped by flag SkipsConsRelease.
            //
          }
          // Task.cpp:43
          ++tmemP1ConsReleaseState;
        }
        //
        // tmemO [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{28}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset3832) >= (int32_t{0})) {
            // Task.cpp:5100
            tmemOProdToken = tmemODstStack.mPipeline.producer_try_acquire(tmemOProdState);
          }
        }
        // Task.cpp:1607
        // Task.cpp:4288
        {
          // Task.cpp:4318
          tmemODstStack.mPipeline.producer_acquire(tmemOProdState, tmemOProdToken);
        }
        //
        // tmemS1 [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{1024}].
        //
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset3832) >= (int32_t{0})) {
            // Task.cpp:5100
            tmemS1ProdToken = tmemS1DstStack.mPipeline.producer_try_acquire(tmemS1ProdState);
          }
        }
        // Task.cpp:1607
        // Task.cpp:4288
        {
          // Task.cpp:4318
          tmemS1DstStack.mPipeline.producer_acquire(tmemS1ProdState, tmemS1ProdToken);
        }
        //
        // tmemO [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{28}, Flags{0}].
        //
        // Task.cpp:1511
        smemPtrV16 = smemPtrV3;
        // Task.cpp:1511
        memIdxV16 = smemIdxV3;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{tmemOProdState.index()};
          // TmemO.h:493
          cutlass::float_e4m3_t* smemV{smemPtrV16};
          // TmemO.h:505
          smemV = smemV + ((memIdxV16) * (int32_t{8192}));
          // TmemO.h:535
          bool readD{true};
          // TmemO.h:545
          if ((loopOffset3832) == (int32_t{0})) {
            // TmemO.h:547
            readD = false;
          }
          // Mma.cpp:618
          {
            // TmemTile.cpp:1765
            uint32_t tmemPtrD{(mTmemBaseOffset) + (uint32_t{384})};
            // TmemTile.cpp:1521
            uint32_t tmemPtrA{(mTmemBaseOffset) + (uint32_t{128})};
            //
            // leadingDimInBytes = 0, strideInBytes = 512, swizzleMode = 2.
            //
            // Mma.cpp:203
            uint64_t smemDescB{
              trtllm::dev::createSmemDesc(smemV,
                                          uint32_t{0x0 /*hi=0, lo=0*/},
                                          uint32_t{0x80004020 /*hi=32768, lo=16416*/})};
            //
            // MMA inst for mi=0 ni=0 ki=0.
            //
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    true,
                                                                    int32_t{128},
                                                                    int32_t{64},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f8f6f4,
                                           cuda_ptx::cta_group_1,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_0,
                                           bool{readD});
            }
            //
            // MMA inst for mi=0 ni=0 ki=1.
            //
            // TmemTile.cpp:2041
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    true,
                                                                    int32_t{128},
                                                                    int32_t{64},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f8f6f4,
                                           cuda_ptx::cta_group_1,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_1,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=2.
            //
            // TmemTile.cpp:2041
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    true,
                                                                    int32_t{128},
                                                                    int32_t{64},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f8f6f4,
                                           cuda_ptx::cta_group_1,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_2,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=3.
            //
            // TmemTile.cpp:2041
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    true,
                                                                    int32_t{128},
                                                                    int32_t{64},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f8f6f4,
                                           cuda_ptx::cta_group_1,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_3,
                                           bool{true});
            }
          }
        }
        //
        // tmemO [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{28}, Flags{0}].
        //
        // Task.cpp:4522
        {
          // Task.cpp:4540
          {
            // Task.cpp:4556
            tmemODstStack.mPipeline.producer_commit(tmemOProdState);
          }
          // Task.cpp:43
          ++tmemOProdState;
        }
        //
        // smemKv [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
        //
        // Task.cpp:2568
        if ((loopOffset3832) >= (int32_t{0})) {
          // Task.cpp:2596
          {
            // Task.cpp:2620
            smemKvSrcStack.mPipeline.consumer_release(smemKvConsReleaseState);
          }
          // Task.cpp:43
          ++smemKvConsReleaseState;
        }
        //
        // smemKv [ConsWork (call 5), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{262144}].
        //
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{smemKvConsState.index()};
          // SmemKv.h:267
          smemPtrK3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
          // SmemKv.h:304
          smemIdxK3 = smemIndicesK[int32_t{0}];
        }
        //
        // tmemS1 [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
        //
        // Task.cpp:1511
        smemPtrQ10 = smemPtrQ1_2;
        // Task.cpp:1511
        smemIdxQ10 = smemIdxQ1_2;
        // Task.cpp:1511
        smemPtrK10 = smemPtrK3;
        // Task.cpp:1511
        memIdxK10 = smemIdxK3;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{tmemS1ProdState.index()};
          // TmemS.h:1889
          cutlass::float_e4m3_t* smemQ{smemPtrQ10};
          // TmemS.h:1910
          smemQ += (smemIdxQ10) * (int32_t{8192});
          // TmemS.h:1938
          cutlass::float_e4m3_t* smemK{smemPtrK10};
          // TmemS.h:1944
          smemK += (memIdxK10) * (int32_t{8192});
          // Mma.cpp:618
          {
            // TmemTile.cpp:1765
            uint32_t tmemPtrD{int32_t((tmemS1DstStack.mInstId) == (int32_t{0})) ? (int32_t{0})
                                                                                : (int32_t{128})};
            //
            // leadingDimInBytes = 8192, strideInBytes = 512, swizzleMode = 2.
            //
            // Mma.cpp:203
            uint64_t smemDescA{
              trtllm::dev::createSmemDesc(smemQ,
                                          uint32_t{0x2000000 /*hi=512, lo=0*/},
                                          uint32_t{0x80004020 /*hi=32768, lo=16416*/})};
            //
            // leadingDimInBytes = 8192, strideInBytes = 512, swizzleMode = 2.
            //
            // Mma.cpp:203
            uint64_t smemDescB{
              trtllm::dev::createSmemDesc(smemK,
                                          uint32_t{0x2000000 /*hi=512, lo=0*/},
                                          uint32_t{0x80004020 /*hi=32768, lo=16416*/})};
            //
            // MMA inst for mi=0 ni=0 ki=0.
            //
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{128},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
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
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{128},
                                                                    int32_t{32},
                                                                    false,
                                                                    true)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma(cuda_ptx::kind_f8f6f4,
                                    cuda_ptx::cta_group_1,
                                    tmemPtrD,
                                    smemDescA,
                                    smemDescB,
                                    utcmmaDesc_0_0_1,
                                    bool{true});
            }
          }
        }
        //
        // smemKv [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:2568
        if ((loopOffset3832) >= (int32_t{0})) {
          // Task.cpp:2596
          {
            // Task.cpp:2620
            smemKvSrcStack.mPipeline.consumer_release(smemKvConsReleaseState);
          }
          // Task.cpp:43
          ++smemKvConsReleaseState;
        }
        //
        // tmemS1 [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
        //
        // Task.cpp:4522
        {
          // Task.cpp:4540
          {
            // Task.cpp:4556
            tmemS1DstStack.mPipeline.producer_commit(tmemS1ProdState);
          }
          // Task.cpp:43
          ++tmemS1ProdState;
        }
        // Task.cpp:3499
        lastLoopOffset = loopOffset3832;
      }
      //
      // Pull the last iter down.
      //
      // Task.cpp:3534
      if (((numLoopSteps) - (int32_t{1})) > (int32_t{0})) {
        // Task.cpp:3535
        ++lastLoopOffset;
      }
      //
      // smemKv [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{6}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:1607
        if (hasOneLoopIter) {
          // Task.cpp:2780
          smemKvConsToken = smemKvSrcStack.mPipeline.consumer_try_wait(smemKvConsState);
        }
        // Task.cpp:2848
        smemKvSrcStack.mPipeline.consumer_wait(smemKvConsState, smemKvConsToken);
      }
      //
      // smemKv [ConsWork (call 6), LastIter, FreqInfo{0, 1}, UserTags{6}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{smemKvConsState.index()};
        // SmemKv.h:322
        smemPtrV3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:366
        smemIndicesV[int32_t{0}] = index;
        // SmemKv.h:372
        smemIdxV3 = smemIndicesV[int32_t{0}];
        // Task.cpp:43
        ++smemKvConsState;
      }
      //
      // tmemP0 [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{49152}].
      //
      //
      // Skipped by flag SkipsConsWait.
      //
      //
      // tmemP0 [ConsWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{49152}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemP0ConsState.index()};
        // Task.cpp:43
        ++tmemP0ConsState;
      }
      //
      // tmemP0 [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{49152}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:2596
        {
          //
          // Skipped by flag SkipsConsRelease.
          //
        }
        // Task.cpp:43
        ++tmemP0ConsReleaseState;
      }
      //
      // tmemO [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{26}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5078
        if ((lastLoopOffset) >= (int32_t{0})) {
          // Task.cpp:5100
          tmemOProdToken = tmemODstStack.mPipeline.producer_try_acquire(tmemOProdState);
        }
      }
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4318
        tmemODstStack.mPipeline.producer_acquire(tmemOProdState, tmemOProdToken);
      }
      //
      // tmemS0 [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{1540}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5078
        if ((lastLoopOffset) >= (int32_t{0})) {
          // Task.cpp:5100
          tmemS0ProdToken = tmemS0DstStack.mPipeline.producer_try_acquire(tmemS0ProdState);
        }
      }
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4318
        tmemS0DstStack.mPipeline.producer_acquire(tmemS0ProdState, tmemS0ProdToken);
      }
      //
      // tmemO [ProdWork (call 2), LastIter, FreqInfo{0, 1}, UserTags{26}, Flags{0}].
      //
      // TmemO.h:277
      cutlass::float_e4m3_t* smemPtrV16;
      // TmemO.h:282
      int32_t memIdxV16;
      // Task.cpp:1511
      smemPtrV16 = smemPtrV3;
      // Task.cpp:1511
      memIdxV16 = smemIdxV3;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemOProdState.index()};
        // TmemO.h:493
        cutlass::float_e4m3_t* smemV{smemPtrV16};
        // TmemO.h:505
        smemV = smemV + ((memIdxV16) * (int32_t{8192}));
        // TmemO.h:535
        bool readD{true};
        // TmemO.h:545
        if ((lastLoopOffset) == (int32_t{0})) {
          // TmemO.h:547
          readD = false;
        }
        // Mma.cpp:618
        {
          // TmemTile.cpp:1765
          uint32_t tmemPtrD{(mTmemBaseOffset) + (uint32_t{320})};
          // TmemTile.cpp:1521
          uint32_t tmemPtrA{mTmemBaseOffset};
          //
          // leadingDimInBytes = 0, strideInBytes = 512, swizzleMode = 2.
          //
          // Mma.cpp:203
          uint64_t smemDescB{
            trtllm::dev::createSmemDesc(smemV,
                                        uint32_t{0x0 /*hi=0, lo=0*/},
                                        uint32_t{0x80004020 /*hi=32768, lo=16416*/})};
          //
          // MMA inst for mi=0 ni=0 ki=0.
          //
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  true,
                                                                  int32_t{128},
                                                                  int32_t{64},
                                                                  int32_t{32},
                                                                  false,
                                                                  true)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
            cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f8f6f4,
                                         cuda_ptx::cta_group_1,
                                         tmemPtrD,
                                         tmemPtrA,
                                         smemDescB,
                                         utcmmaDesc_0_0_0,
                                         bool{readD});
          }
          //
          // MMA inst for mi=0 ni=0 ki=1.
          //
          // TmemTile.cpp:2041
          tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  true,
                                                                  int32_t{128},
                                                                  int32_t{64},
                                                                  int32_t{32},
                                                                  false,
                                                                  true)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
            cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f8f6f4,
                                         cuda_ptx::cta_group_1,
                                         tmemPtrD,
                                         tmemPtrA,
                                         smemDescB,
                                         utcmmaDesc_0_0_1,
                                         bool{true});
          }
          //
          // MMA inst for mi=0 ni=0 ki=2.
          //
          // TmemTile.cpp:2041
          tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  true,
                                                                  int32_t{128},
                                                                  int32_t{64},
                                                                  int32_t{32},
                                                                  false,
                                                                  true)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
            cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f8f6f4,
                                         cuda_ptx::cta_group_1,
                                         tmemPtrD,
                                         tmemPtrA,
                                         smemDescB,
                                         utcmmaDesc_0_0_2,
                                         bool{true});
          }
          //
          // MMA inst for mi=0 ni=0 ki=3.
          //
          // TmemTile.cpp:2041
          tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  true,
                                                                  int32_t{128},
                                                                  int32_t{64},
                                                                  int32_t{32},
                                                                  false,
                                                                  true)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
            cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f8f6f4,
                                         cuda_ptx::cta_group_1,
                                         tmemPtrD,
                                         tmemPtrA,
                                         smemDescB,
                                         utcmmaDesc_0_0_3,
                                         bool{true});
          }
        }
      }
      //
      // tmemO [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{26}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4540
        {
          // Task.cpp:4556
          tmemODstStack.mPipeline.producer_commit(tmemOProdState);
        }
        // Task.cpp:43
        ++tmemOProdState;
      }
      //
      // tmemS0 [ProdWork (call 2), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{516}].
      //
      //
      // tmemS0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{516}].
      //
      //
      // smemKv [ConsWork (call 7), LastIter, FreqInfo{0, 1}, UserTags{6}, Flags{262144}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{smemKvConsState.index()};
        // SmemKv.h:322
        smemPtrV3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:372
        smemIdxV3 = smemIndicesV[int32_t{0}];
      }
      //
      // smemQ [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:2596
        {
          // Task.cpp:2620
          smemQSrcStack.mPipeline.consumer_release(smemQConsReleaseState);
        }
        // Task.cpp:43
        ++smemQConsReleaseState;
      }
      //
      // smemQ [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:2596
        {
          // Task.cpp:2620
          smemQSrcStack.mPipeline.consumer_release(smemQConsReleaseState);
        }
        // Task.cpp:43
        ++smemQConsReleaseState;
      }
      //
      // tmemP1 [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{49152}].
      //
      //
      // Skipped by flag SkipsConsWait.
      //
      //
      // tmemP1 [ConsWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{49152}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemP1ConsState.index()};
        // Task.cpp:43
        ++tmemP1ConsState;
      }
      //
      // tmemP1 [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{49152}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:2596
        {
          //
          // Skipped by flag SkipsConsRelease.
          //
        }
        // Task.cpp:43
        ++tmemP1ConsReleaseState;
      }
      //
      // tmemO [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{28}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5078
        if ((lastLoopOffset) >= (int32_t{0})) {
          // Task.cpp:5100
          tmemOProdToken = tmemODstStack.mPipeline.producer_try_acquire(tmemOProdState);
        }
      }
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4318
        tmemODstStack.mPipeline.producer_acquire(tmemOProdState, tmemOProdToken);
      }
      //
      // tmemS1 [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{1540}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5078
        if ((lastLoopOffset) >= (int32_t{0})) {
          // Task.cpp:5100
          tmemS1ProdToken = tmemS1DstStack.mPipeline.producer_try_acquire(tmemS1ProdState);
        }
      }
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4318
        tmemS1DstStack.mPipeline.producer_acquire(tmemS1ProdState, tmemS1ProdToken);
      }
      //
      // tmemO [ProdWork (call 3), LastIter, FreqInfo{0, 1}, UserTags{28}, Flags{0}].
      //
      // Task.cpp:1511
      smemPtrV16 = smemPtrV3;
      // Task.cpp:1511
      memIdxV16 = smemIdxV3;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemOProdState.index()};
        // TmemO.h:493
        cutlass::float_e4m3_t* smemV{smemPtrV16};
        // TmemO.h:505
        smemV = smemV + ((memIdxV16) * (int32_t{8192}));
        // TmemO.h:535
        bool readD{true};
        // TmemO.h:545
        if ((lastLoopOffset) == (int32_t{0})) {
          // TmemO.h:547
          readD = false;
        }
        // Mma.cpp:618
        {
          // TmemTile.cpp:1765
          uint32_t tmemPtrD{(mTmemBaseOffset) + (uint32_t{384})};
          // TmemTile.cpp:1521
          uint32_t tmemPtrA{(mTmemBaseOffset) + (uint32_t{128})};
          //
          // leadingDimInBytes = 0, strideInBytes = 512, swizzleMode = 2.
          //
          // Mma.cpp:203
          uint64_t smemDescB{
            trtllm::dev::createSmemDesc(smemV,
                                        uint32_t{0x0 /*hi=0, lo=0*/},
                                        uint32_t{0x80004020 /*hi=32768, lo=16416*/})};
          //
          // MMA inst for mi=0 ni=0 ki=0.
          //
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  true,
                                                                  int32_t{128},
                                                                  int32_t{64},
                                                                  int32_t{32},
                                                                  false,
                                                                  true)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
            cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f8f6f4,
                                         cuda_ptx::cta_group_1,
                                         tmemPtrD,
                                         tmemPtrA,
                                         smemDescB,
                                         utcmmaDesc_0_0_0,
                                         bool{readD});
          }
          //
          // MMA inst for mi=0 ni=0 ki=1.
          //
          // TmemTile.cpp:2041
          tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  true,
                                                                  int32_t{128},
                                                                  int32_t{64},
                                                                  int32_t{32},
                                                                  false,
                                                                  true)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
            cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f8f6f4,
                                         cuda_ptx::cta_group_1,
                                         tmemPtrD,
                                         tmemPtrA,
                                         smemDescB,
                                         utcmmaDesc_0_0_1,
                                         bool{true});
          }
          //
          // MMA inst for mi=0 ni=0 ki=2.
          //
          // TmemTile.cpp:2041
          tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  true,
                                                                  int32_t{128},
                                                                  int32_t{64},
                                                                  int32_t{32},
                                                                  false,
                                                                  true)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
            cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f8f6f4,
                                         cuda_ptx::cta_group_1,
                                         tmemPtrD,
                                         tmemPtrA,
                                         smemDescB,
                                         utcmmaDesc_0_0_2,
                                         bool{true});
          }
          //
          // MMA inst for mi=0 ni=0 ki=3.
          //
          // TmemTile.cpp:2041
          tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  true,
                                                                  int32_t{128},
                                                                  int32_t{64},
                                                                  int32_t{32},
                                                                  false,
                                                                  true)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
            cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f8f6f4,
                                         cuda_ptx::cta_group_1,
                                         tmemPtrD,
                                         tmemPtrA,
                                         smemDescB,
                                         utcmmaDesc_0_0_3,
                                         bool{true});
          }
        }
      }
      //
      // tmemO [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{28}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4540
        {
          // Task.cpp:4556
          tmemODstStack.mPipeline.producer_commit(tmemOProdState);
        }
        // Task.cpp:43
        ++tmemOProdState;
      }
      //
      // tmemS1 [ProdWork (call 2), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{516}].
      //
      //
      // tmemS1 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{516}].
      //
      //
      // smemKv [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{6}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:2596
        {
          // Task.cpp:2620
          smemKvSrcStack.mPipeline.consumer_release(smemKvConsReleaseState);
        }
        // Task.cpp:43
        ++smemKvConsReleaseState;
      }
    //
    // Tail work.
    //
    // Task.cpp:3553
    ExitTileWithSignalingLabel:
    // Task.cpp:3560
    ExitTileWithoutSignalingLabel:
      // Task.cpp:5532
      auto newWorkTileInfoTuple{
        workIdStorageSrcStack.mScheduler.fetch_next_work(workIdStorageSrcStack.workTileInfo,
                                                         workIdStorageSrcStack.mPipeline,
                                                         workIdStorageConsState)};
      // Task.cpp:5534
      workIdStorageSrcStack.workTileInfo = cute::get<int32_t{0}>(newWorkTileInfoTuple);
      // Task.cpp:5542
      ++workIdStorageConsState;
      // Task.cpp:5644
      mCtaIdxX = workIdStorageSrcStack.workTileInfo.M_idx;
      // Task.cpp:5645
      mCtaIdxY = workIdStorageSrcStack.workTileInfo.N_idx;
      // Task.cpp:5646
      mCtaIdxZ = workIdStorageSrcStack.workTileInfo.L_idx;
    } while (workIdStorageSrcStack.workTileInfo.is_valid_tile);
    //
    // tmemS0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{516}].
    //
    // Task.cpp:4505
    {
      // Task.cpp:4540
      {
        // Task.cpp:4556
        tmemS0DstStack.mPipeline.producer_commit(tmemS0ProdState);
      }
      // Task.cpp:43
      ++tmemS0ProdState;
    }
    //
    // tmemS1 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{516}].
    //
    // Task.cpp:4505
    {
      // Task.cpp:4540
      {
        // Task.cpp:4556
        tmemS1DstStack.mPipeline.producer_commit(tmemS1ProdState);
      }
      // Task.cpp:43
      ++tmemS1ProdState;
    }
  }
};
// Task.cpp:559
// Fmha.h:2664
struct SchedulerTask {
  // Task.cpp:283
  int32_t mCtaIdxX;
  // Task.cpp:287
  int32_t mCtaIdxY;
  // Task.cpp:291
  int32_t mCtaIdxZ;
  // Task.cpp:566
  inline __device__ SchedulerTask(fmha::KernelParams const& params,
                                  KernelState const& state,
                                  int32_t warpGrpStart)
    : // Kernel.cpp:194
    mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , // Kernel.cpp:195
    mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , // Kernel.cpp:196
    mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)} {}
  // Task.cpp:522
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:547
    return ((state.mWarpIdx) >= (int32_t{14})) && ((state.mWarpIdx) < (int32_t{15}));
  }
  // Task.cpp:454
  inline __device__ void execute(fmha::KernelParams const& params,
                                 KernelState const& state,
                                 WorkIdStorageSmem& workIdStorageDstSmem,
                                 WorkIdStorageStack& workIdStorageDstStack,
                                 WorkIdStorageSmem& workIdStorageSrcSmem,
                                 WorkIdStorageStack& workIdStorageSrcStack,
                                 WorkIdThrottleBarrierStack& workIdThrottleBarrierSrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<56>{});
    // Task.cpp:2114
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsReleaseState{};
    // Task.cpp:2135
    int32_t workIdStorageConsToken{int32_t{0}};
    // Task.cpp:2114
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState workIdThrottleBarrierConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState
      workIdThrottleBarrierConsReleaseState{};
    // Task.cpp:2135
    int32_t workIdThrottleBarrierConsToken{int32_t{0}};
    // Task.cpp:2013
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageProdState{int32_t{0}, int32_t{1}, int32_t{0}};
    // Task.cpp:2033
    int32_t workIdStorageProdToken{int32_t{1}};
    // Task.cpp:6340
    do {
      // Task.cpp:6493
      workIdThrottleBarrierSrcStack.mPipeline.consumer_wait(workIdThrottleBarrierConsState,
                                                            workIdThrottleBarrierConsToken);
      // Task.cpp:6500
      workIdThrottleBarrierSrcStack.mPipeline.consumer_release(workIdThrottleBarrierConsState);
      // Task.cpp:6502
      ++workIdThrottleBarrierConsState;
      // Task.cpp:6464
      workIdStorageProdState = workIdStorageSrcStack.mScheduler.advance_to_next_work(
        workIdStorageSrcStack.mPipeline.get_pipeline(),
        workIdStorageProdState);
      // Task.cpp:5532
      auto newWorkTileInfoTuple{
        workIdStorageSrcStack.mScheduler.fetch_next_work(workIdStorageSrcStack.workTileInfo,
                                                         workIdStorageSrcStack.mPipeline,
                                                         workIdStorageConsState)};
      // Task.cpp:5534
      workIdStorageSrcStack.workTileInfo = cute::get<int32_t{0}>(newWorkTileInfoTuple);
      // Task.cpp:5542
      ++workIdStorageConsState;
      // Task.cpp:5644
      mCtaIdxX = workIdStorageSrcStack.workTileInfo.M_idx;
      // Task.cpp:5645
      mCtaIdxY = workIdStorageSrcStack.workTileInfo.N_idx;
      // Task.cpp:5646
      mCtaIdxZ = workIdStorageSrcStack.workTileInfo.L_idx;
    } while (workIdStorageSrcStack.workTileInfo.is_valid_tile);
    // Task.cpp:6423
    workIdStorageDstStack.mPipeline.producer_tail(workIdStorageProdState);
  }
};
extern "C" __global__
__launch_bounds__(512, 1) void fmhaSm100fKernel_QkvE4m3OBfloat16H64PagedKvSlidingOrChunkedCausalP64VarSeqQ128Kv128PersistentContext(
  CUTE_GRID_CONSTANT fmha::KernelParams const params) {
  // Kernel.cpp:1658
  trtllm::dev::prefetchTensorMap(&params.tmaQ_);
  // Kernel.cpp:1658
  trtllm::dev::prefetchTensorMap(&params.tmaK_);
  // Kernel.cpp:1658
  trtllm::dev::prefetchTensorMap(&params.tmaV_);
  // Kernel.cpp:1675
  extern __shared__ uint8_t smem__[];
  // Kernel.cpp:1686
  int32_t smemOffset__{int32_t{0}};
  // Kernel.cpp:1725
  smemOffset__ = (((smemOffset__) + (int32_t{1023})) / (int32_t{1024})) * (int32_t{1024});
  // Kernel.cpp:1729
  uint8_t* smemQSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemQSmem)});
  // Kernel.cpp:1725
  smemOffset__ = (((smemOffset__) + (int32_t{1023})) / (int32_t{1024})) * (int32_t{1024});
  // Kernel.cpp:1729
  uint8_t* smemKvSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemKvSmem)});
  // Kernel.cpp:1725
  smemOffset__ = (((smemOffset__) + (int32_t{127})) / (int32_t{128})) * (int32_t{128});
  // Kernel.cpp:1729
  uint8_t* smemPageOffsetsKvSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemPageOffsetsKvSmem)});
  // Kernel.cpp:1725
  smemOffset__ = (((smemOffset__) + (int32_t{15})) / (int32_t{16})) * (int32_t{16});
  // Kernel.cpp:1729
  uint8_t* workIdStorageSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(WorkIdStorageSmem)});
  // Kernel.cpp:1702
  uint32_t* TmemSwStatePtr{
    reinterpret_cast<uint32_t*>((reinterpret_cast<uint8_t*>(smem__) + smemOffset__))};
  // Kernel.cpp:1710
  smemOffset__ += int32_t{16};
  // Kernel.cpp:1729
  uint8_t* smemQSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemQSmemBarrier)});
  // Kernel.cpp:1729
  uint8_t* smemKvSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemKvSmemBarrier)});
  // Kernel.cpp:1729
  uint8_t* smemPageOffsetsKvSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemPageOffsetsKvSmemBarrier)});
  // Kernel.cpp:1729
  uint8_t* workIdStorageSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(WorkIdStorageSmemBarrier)});
  // Kernel.cpp:1729
  uint8_t* workIdThrottleBarrierSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(WorkIdThrottleBarrierSmemBarrier)});
  // Kernel.cpp:1729
  uint8_t* tmemS0SmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(TmemS0SmemBarrier)});
  // Kernel.cpp:1729
  uint8_t* tmemSoftmaxLocal0SmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(TmemSoftmaxLocal0SmemBarrier)});
  // Kernel.cpp:1729
  uint8_t* tmemS1SmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(TmemS0SmemBarrier)});
  // Kernel.cpp:1729
  uint8_t* tmemSoftmaxLocal1SmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(TmemSoftmaxLocal0SmemBarrier)});
  // Kernel.cpp:1729
  uint8_t* orderP01SmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(OrderP01SmemBarrier)});
  // Kernel.cpp:1729
  uint8_t* tmemP0SmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(TmemP0SmemBarrier)});
  // Kernel.cpp:1729
  uint8_t* tmemP1SmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(TmemP0SmemBarrier)});
  // Kernel.cpp:1729
  uint8_t* tmemOSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(TmemOSmemBarrier)});
  // Kernel.cpp:1766
  KernelState const state{params, TmemSwStatePtr};
  // Kernel.cpp:2216
  SmemQSmem* smemQSmem{reinterpret_cast<SmemQSmem*>(smemQSmemPtr)};
  // Kernel.cpp:2228
  SmemQSmemBarrier* smemQSmemBarrier{reinterpret_cast<SmemQSmemBarrier*>(smemQSmemBarrierPtr)};
  // Kernel.cpp:2283
  SmemQStack smemQStack{(*smemQSmem),
                        (*smemQSmemBarrier),
                        state.mWarpIdx,
                        state.mClusterDimX,
                        state.mClusterDimY,
                        int32_t{15},
                        int32_t{-1}};
  // Kernel.cpp:2216
  SmemKvSmem* smemKvSmem{reinterpret_cast<SmemKvSmem*>(smemKvSmemPtr)};
  // Kernel.cpp:2228
  SmemKvSmemBarrier* smemKvSmemBarrier{reinterpret_cast<SmemKvSmemBarrier*>(smemKvSmemBarrierPtr)};
  // Kernel.cpp:2283
  SmemKvStack smemKvStack{(*smemKvSmem),
                          (*smemKvSmemBarrier),
                          state.mWarpIdx,
                          state.mClusterDimX,
                          state.mClusterDimY,
                          int32_t{15},
                          int32_t{-1}};
  // Kernel.cpp:2216
  SmemPageOffsetsKvSmem* smemPageOffsetsKvSmem{
    reinterpret_cast<SmemPageOffsetsKvSmem*>(smemPageOffsetsKvSmemPtr)};
  // Kernel.cpp:2228
  SmemPageOffsetsKvSmemBarrier* smemPageOffsetsKvSmemBarrier{
    reinterpret_cast<SmemPageOffsetsKvSmemBarrier*>(smemPageOffsetsKvSmemBarrierPtr)};
  // Kernel.cpp:2283
  SmemPageOffsetsKvStack smemPageOffsetsKvStack{(*smemPageOffsetsKvSmem),
                                                (*smemPageOffsetsKvSmemBarrier),
                                                state.mWarpIdx,
                                                state.mClusterDimX,
                                                state.mClusterDimY,
                                                int32_t{13},
                                                int32_t{-1}};
  // Kernel.cpp:2216
  WorkIdStorageSmem* workIdStorageSmem{reinterpret_cast<WorkIdStorageSmem*>(workIdStorageSmemPtr)};
  // Kernel.cpp:2228
  WorkIdStorageSmemBarrier* workIdStorageSmemBarrier{
    reinterpret_cast<WorkIdStorageSmemBarrier*>(workIdStorageSmemBarrierPtr)};
  // Kernel.cpp:2283
  WorkIdStorageStack workIdStorageStack{(*workIdStorageSmem),
                                        (*workIdStorageSmemBarrier),
                                        state.mWarpIdx,
                                        state.mClusterDimX,
                                        state.mClusterDimY,
                                        int32_t{14},
                                        int32_t{-1}};
  // Kernel.cpp:2228
  WorkIdThrottleBarrierSmemBarrier* workIdThrottleBarrierSmemBarrier{
    reinterpret_cast<WorkIdThrottleBarrierSmemBarrier*>(workIdThrottleBarrierSmemBarrierPtr)};
  // Kernel.cpp:2283
  WorkIdThrottleBarrierStack workIdThrottleBarrierStack{(*workIdThrottleBarrierSmemBarrier),
                                                        state.mWarpIdx,
                                                        state.mClusterDimX,
                                                        state.mClusterDimY,
                                                        int32_t{14},
                                                        int32_t{-1}};
  // Kernel.cpp:2228
  TmemS0SmemBarrier* tmemS0SmemBarrier{reinterpret_cast<TmemS0SmemBarrier*>(tmemS0SmemBarrierPtr)};
  // Kernel.cpp:2283
  TmemS0Stack tmemS0Stack{(*tmemS0SmemBarrier),
                          state.mWarpIdx,
                          state.mClusterDimX,
                          state.mClusterDimY,
                          int32_t{1},
                          int32_t{-1},
                          int32_t{-1},
                          int32_t{0}};
  // Kernel.cpp:2228
  TmemSoftmaxLocal0SmemBarrier* tmemSoftmaxLocal0SmemBarrier{
    reinterpret_cast<TmemSoftmaxLocal0SmemBarrier*>(tmemSoftmaxLocal0SmemBarrierPtr)};
  // Kernel.cpp:2283
  TmemSoftmaxLocal0Stack tmemSoftmaxLocal0Stack{(*tmemSoftmaxLocal0SmemBarrier),
                                                state.mWarpIdx,
                                                state.mClusterDimX,
                                                state.mClusterDimY,
                                                int32_t{9},
                                                int32_t{-1},
                                                int32_t{0}};
  // Kernel.cpp:2283
  TmemSoftmaxGlobal0Stack tmemSoftmaxGlobal0Stack{state.mWarpIdx,
                                                  state.mClusterDimX,
                                                  state.mClusterDimY,
                                                  int32_t{0},
                                                  int32_t{-1}};
  // Kernel.cpp:2228
  TmemS0SmemBarrier* tmemS1SmemBarrier{reinterpret_cast<TmemS0SmemBarrier*>(tmemS1SmemBarrierPtr)};
  // Kernel.cpp:2283
  TmemS0Stack tmemS1Stack{(*tmemS1SmemBarrier),
                          state.mWarpIdx,
                          state.mClusterDimX,
                          state.mClusterDimY,
                          int32_t{5},
                          int32_t{-1},
                          int32_t{-1},
                          int32_t{1}};
  // Kernel.cpp:2228
  TmemSoftmaxLocal0SmemBarrier* tmemSoftmaxLocal1SmemBarrier{
    reinterpret_cast<TmemSoftmaxLocal0SmemBarrier*>(tmemSoftmaxLocal1SmemBarrierPtr)};
  // Kernel.cpp:2283
  TmemSoftmaxLocal0Stack tmemSoftmaxLocal1Stack{(*tmemSoftmaxLocal1SmemBarrier),
                                                state.mWarpIdx,
                                                state.mClusterDimX,
                                                state.mClusterDimY,
                                                int32_t{10},
                                                int32_t{-1},
                                                int32_t{1}};
  // Kernel.cpp:2283
  TmemSoftmaxGlobal0Stack tmemSoftmaxGlobal1Stack{state.mWarpIdx,
                                                  state.mClusterDimX,
                                                  state.mClusterDimY,
                                                  int32_t{0},
                                                  int32_t{-1}};
  // Kernel.cpp:2228
  OrderP01SmemBarrier* orderP01SmemBarrier{
    reinterpret_cast<OrderP01SmemBarrier*>(orderP01SmemBarrierPtr)};
  // Kernel.cpp:2283
  OrderP01Stack orderP01Stack{(*orderP01SmemBarrier),
                              state.mWarpIdx,
                              state.mClusterDimX,
                              state.mClusterDimY,
                              int32_t{0},
                              int32_t{-1}};
  // Kernel.cpp:2228
  TmemP0SmemBarrier* tmemP0SmemBarrier{reinterpret_cast<TmemP0SmemBarrier*>(tmemP0SmemBarrierPtr)};
  // Kernel.cpp:2283
  TmemP0Stack tmemP0Stack{(*tmemP0SmemBarrier),
                          (*orderP01SmemBarrier),
                          orderP01Stack,
                          state.mWarpIdx,
                          state.mClusterDimX,
                          state.mClusterDimY,
                          int32_t{2},
                          int32_t{0},
                          int32_t{-1},
                          int32_t{0}};
  // Kernel.cpp:2228
  TmemP0SmemBarrier* tmemP1SmemBarrier{reinterpret_cast<TmemP0SmemBarrier*>(tmemP1SmemBarrierPtr)};
  // Kernel.cpp:2283
  TmemP0Stack tmemP1Stack{(*tmemP1SmemBarrier),
                          (*orderP01SmemBarrier),
                          orderP01Stack,
                          state.mWarpIdx,
                          state.mClusterDimX,
                          state.mClusterDimY,
                          int32_t{6},
                          int32_t{1},
                          int32_t{-1},
                          int32_t{1}};
  // Kernel.cpp:2228
  TmemOSmemBarrier* tmemOSmemBarrier{reinterpret_cast<TmemOSmemBarrier*>(tmemOSmemBarrierPtr)};
  // Kernel.cpp:2283
  TmemOStack tmemOStack{(*tmemOSmemBarrier),
                        state.mWarpIdx,
                        state.mClusterDimX,
                        state.mClusterDimY,
                        int32_t{8},
                        int32_t{-1}};
  // Kernel.cpp:2283
  TmemCorr0Stack tmemCorr0Stack{state.mWarpIdx,
                                state.mClusterDimX,
                                state.mClusterDimY,
                                int32_t{0},
                                int32_t{-1}};
  // Kernel.cpp:2283
  TmemCorr1Stack tmemCorr1Stack{state.mWarpIdx,
                                state.mClusterDimX,
                                state.mClusterDimY,
                                int32_t{0},
                                int32_t{-1}};
  // Kernel.cpp:1862
  cutlass::arch::fence_barrier_init();
  // Kernel.cpp:2320
  if ((reinterpret_cast<int32_t const&>(threadIdx.x)) < (int32_t{32})) {
    // Kernel.cpp:2344
    cuda_ptx::tcgen05_alloc(cuda_ptx::cta_group_1_t{}, state.mTmemSwStatePtr, int32_t{512});
    // Kernel.cpp:2357
    cuda_ptx::tcgen05_relinquish_alloc_permit(cuda_ptx::cta_group_1_t{});
  }
  // Kernel.cpp:1886
  __syncthreads();
  // Kernel.cpp:2014
  if (bool{LoadPageOffsetsTask::isSelected(params, state)}) {
    // Kernel.cpp:2081
    LoadPageOffsetsTask loadPageOffsetsTask{params, state, int32_t{13}};
    // Kernel.cpp:2135
    loadPageOffsetsTask.execute(params,
                                state,
                                (*smemPageOffsetsKvSmem),
                                smemPageOffsetsKvStack,
                                (*workIdStorageSmem),
                                workIdStorageStack);
  } else {
    // Kernel.cpp:2014
    if (bool{LoadTask::isSelected(params, state)}) {
      // Kernel.cpp:2081
      LoadTask loadTask{params, state, int32_t{15}};
      // Kernel.cpp:2135
      loadTask.execute(params,
                       state,
                       (*smemQSmem),
                       smemQStack,
                       (*smemKvSmem),
                       smemKvStack,
                       workIdThrottleBarrierStack,
                       (*smemPageOffsetsKvSmem),
                       smemPageOffsetsKvStack,
                       (*workIdStorageSmem),
                       workIdStorageStack);
    } else {
      // Kernel.cpp:2014
      if ((bool{SoftmaxTask0::isSelected(params, state)}) ||
          (bool{SoftmaxTask1::isSelected(params, state)})) {
        // Kernel.cpp:2081
        SoftmaxTask0 softmaxTask0{
          params,
          state,
          int32_t(bool{SoftmaxTask0::isSelected(params, state)}) ? (int32_t{0}) : (int32_t{4})};
        // Kernel.cpp:2135
        softmaxTask0.execute(
          params,
          state,
          (SoftmaxTask0::isSelected(params, state)) ? (tmemSoftmaxLocal0Stack)
                                                    : (tmemSoftmaxLocal1Stack),
          (SoftmaxTask0::isSelected(params, state)) ? (tmemSoftmaxGlobal0Stack)
                                                    : (tmemSoftmaxGlobal1Stack),
          (SoftmaxTask0::isSelected(params, state)) ? (tmemP0Stack) : (tmemP1Stack),
          (SoftmaxTask0::isSelected(params, state)) ? (tmemS0Stack) : (tmemS1Stack),
          (SoftmaxTask0::isSelected(params, state)) ? ((*workIdStorageSmem))
                                                    : ((*workIdStorageSmem)),
          (SoftmaxTask0::isSelected(params, state)) ? (workIdStorageStack) : (workIdStorageStack));
      } else {
        // Kernel.cpp:2014
        if (bool{CorrTask::isSelected(params, state)}) {
          // Kernel.cpp:2081
          CorrTask corrTask{params, state, int32_t{8}};
          // Kernel.cpp:2135
          corrTask.execute(params,
                           state,
                           tmemCorr0Stack,
                           tmemCorr1Stack,
                           tmemSoftmaxLocal0Stack,
                           tmemSoftmaxLocal1Stack,
                           tmemOStack,
                           (*workIdStorageSmem),
                           workIdStorageStack);
          // Task.cpp:5404
          trtllm::dev::CutlassNamedBarrier::sync(128, 7);
          // Task.cpp:5412
          int32_t const warpGrpThreadIdx{(state.mThreadIdx) - (int32_t{256})};
          // Task.cpp:5428
          if ((warpGrpThreadIdx) < (int32_t{32})) {
            // Task.cpp:5458
            cuda_ptx::tcgen05_dealloc(cuda_ptx::cta_group_1_t{},
                                      uint32_t{__shfl_sync(uint32_t{0xffffffff},
                                                           (*state.mTmemSwStatePtr),
                                                           int32_t{0},
                                                           int32_t{32})},
                                      int32_t{512});
          }
        } else {
          // Kernel.cpp:2014
          if (bool{MmaTask::isSelected(params, state)}) {
            // Kernel.cpp:2081
            MmaTask mmaTask{params, state, int32_t{12}};
            // Kernel.cpp:2135
            mmaTask.execute(params,
                            state,
                            tmemS0Stack,
                            tmemS1Stack,
                            tmemOStack,
                            (*smemQSmem),
                            smemQStack,
                            (*smemKvSmem),
                            smemKvStack,
                            tmemP0Stack,
                            tmemP1Stack,
                            (*workIdStorageSmem),
                            workIdStorageStack);
          } else {
            // Kernel.cpp:2014
            if (bool{SchedulerTask::isSelected(params, state)}) {
              // Kernel.cpp:2081
              SchedulerTask schedulerTask{params, state, int32_t{14}};
              // Kernel.cpp:2135
              schedulerTask.execute(params,
                                    state,
                                    (*workIdStorageSmem),
                                    workIdStorageStack,
                                    (*workIdStorageSmem),
                                    workIdStorageStack,
                                    workIdThrottleBarrierStack);
            }
          }
        }
      }
    }
  }
}
extern "C" __global__ void
fmhaSm100fKernel_QkvE4m3OBfloat16H64PagedKvSlidingOrChunkedCausalP64VarSeqQ128Kv128PersistentContextGetSmemSize(
  int32_t* outPtr) {
  int32_t size{0};
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemQSmem));
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemKvSmem));
  size = (size + 127) / 128 * 128;
  size += static_cast<int32_t>(sizeof(SmemPageOffsetsKvSmem));
  size = (size + 15) / 16 * 16;
  size += static_cast<int32_t>(sizeof(WorkIdStorageSmem));
  size += 16;
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(SmemQSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(SmemKvSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(SmemPageOffsetsKvSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(WorkIdStorageSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(WorkIdThrottleBarrierSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(TmemS0SmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(TmemSoftmaxLocal0SmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(TmemS0SmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(TmemSoftmaxLocal0SmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(OrderP01SmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(TmemP0SmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(TmemP0SmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(TmemOSmemBarrier));
  outPtr[0] = size;
}
