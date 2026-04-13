#include <FmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausalP32VarSeqSkipsSoftmaxQ64Kv128PersistentKeepsAbForGen.h>

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
    : // Res.cpp:719
    mPipeline{smemQSmemBarrier.mBarriers,
              warpId,
              int32_t{32768},
              bool{cute::elect_one_sync()},
              CuteFlatTuple214{},
              cute::true_type{},
              cute::true_type{},
              barInitWarpId} {}
};
// Res.cpp:137
// Fmha.h:870
struct SmemSkipSoftmaxVoteStack {
  // Res.cpp:208
  inline __device__ SmemSkipSoftmaxVoteStack(SmemSkipSoftmaxVoteSmem& smemSkipSoftmaxVoteSmem,
                                             int32_t warpId,
                                             int32_t clusterDimX,
                                             int32_t clusterDimY,
                                             int32_t barInitWarpId,
                                             int32_t orderedSequenceGroupId) {}
};
// Res.cpp:137
// Fmha.h:1117
struct SmemKvStack {
  // MemBuffers.cpp:319
  int32_t* mDepSmemPtr3;
  // Res.cpp:595
  trtllm::dev::CutlassTmaUmmaAsyncPipeline<4, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  // MemBuffers.cpp:275
  cutlass::bfloat16_t* mPtr;
  // Res.cpp:208
  inline __device__ SmemKvStack(SmemKvSmem& smemKvSmem,
                                SmemKvSmemBarrier& smemKvSmemBarrier,
                                SmemSkipSoftmaxVoteSmem& smemSkipSoftmaxVoteSmem,
                                SmemSkipSoftmaxVoteStack& smemSkipSoftmaxVoteStack,
                                int32_t warpId,
                                int32_t clusterDimX,
                                int32_t clusterDimY,
                                int32_t barInitWarpId,
                                int32_t orderedSequenceGroupId)
    : // MemBuffers.cpp:332
    mDepSmemPtr3{&smemSkipSoftmaxVoteSmem.mArray[int32_t{0}]}
    , // Res.cpp:719
    mPipeline{smemKvSmemBarrier.mBarriers,
              warpId,
              int32_t{32768},
              bool{cute::elect_one_sync()},
              CuteFlatTuple350{},
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
              CuteFlatTuple593{},
              int32_t{1},
              int32_t{384},
              int32_t{0},
              barInitWarpId}
    , // Res.cpp:1046
    mScheduler{&workIdStorageSmem.workIdResponse[int32_t{0}],
               typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<
                 cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
                 2>::Params{},
               cute::block_id_in_cluster()}
    , // Res.cpp:1086
    workTileInfo{mScheduler.initial_work_tile_info(CuteFlatTuple789{})} {}
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
  // MemBuffers.cpp:488
  int32_t* mDepSmemPtr3;
  // Res.cpp:595
  trtllm::dev::CutlassUmmaAsyncPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  // TmemS.h:549
  int32_t const mNamedBarId;
  // TmemS.h:552
  int32_t const mInstId;
  // Res.cpp:208
  inline __device__ TmemS0Stack(TmemS0SmemBarrier& tmemS0SmemBarrier,
                                SmemSkipSoftmaxVoteSmem& smemSkipSoftmaxVoteSmem,
                                SmemSkipSoftmaxVoteStack& smemSkipSoftmaxVoteStack,
                                int32_t warpId,
                                int32_t clusterDimX,
                                int32_t clusterDimY,
                                int32_t barInitWarpId,
                                int32_t orderedSequenceGroupId,
                                int32_t namedBarId,
                                int32_t instId)
    : // MemBuffers.cpp:501
    mDepSmemPtr3{&smemSkipSoftmaxVoteSmem.mArray[int32_t{0}]}
    , // Res.cpp:776
    mPipeline{tmemS0SmemBarrier.mBarriers,
              warpId,
              int32_t{128},
              CuteFlatTuple1019{},
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
  // Res.cpp:595
  trtllm::dev::CutlassCpAsyncPipeline<2, true> mPipeline;
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
    : // Res.cpp:644
    mPipeline{tmemP0SmemBarrier.mBarriers, warpId, int32_t{128}, int32_t{32}, barInitWarpId}
    , // TmemP.h:463
    mNamedBarId{namedBarId}
    , // TmemP.h:465
    mInstId{instId} {}
};
// Res.cpp:137
// Fmha.h:2104
struct TmemOStack {
  // MemBuffers.cpp:488
  int32_t* mDepSmemPtr3;
  // Res.cpp:595
  trtllm::dev::CutlassUmmaAsyncPipeline<1, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  // Res.cpp:208
  inline __device__ TmemOStack(TmemOSmemBarrier& tmemOSmemBarrier,
                               SmemSkipSoftmaxVoteSmem& smemSkipSoftmaxVoteSmem,
                               SmemSkipSoftmaxVoteStack& smemSkipSoftmaxVoteStack,
                               int32_t warpId,
                               int32_t clusterDimX,
                               int32_t clusterDimY,
                               int32_t barInitWarpId,
                               int32_t orderedSequenceGroupId)
    : // MemBuffers.cpp:501
    mDepSmemPtr3{&smemSkipSoftmaxVoteSmem.mArray[int32_t{0}]}
    , // Res.cpp:776
    mPipeline{tmemOSmemBarrier.mBarriers,
              warpId,
              int32_t{128},
              CuteFlatTuple1388{},
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
    , // Task.cpp:283
    mCtaIdxQ{mCtaIdxX}
    , // FmhaTask.h:517
    mCtaIdxKv{int32_t{0}}
    , // FmhaTask.h:437
    mSeqLenKv{int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                            ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                            : (mBatchIdx)]}}
    , // Kernel.cpp:212
    mNumCtasKv{int32_t{1}}
    , // Task.cpp:379
    mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))} {}
  // Task.cpp:522
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:547
    return ((state.mWarpIdx) >= (int32_t{9})) && ((state.mWarpIdx) < (int32_t{10}));
  }
  // Task.cpp:454
  inline __device__ void execute(fmha::KernelParams const& params,
                                 KernelState const& state,
                                 SmemPageOffsetsKvSmem& smemPageOffsetsKvDstSmem,
                                 SmemPageOffsetsKvStack& smemPageOffsetsKvDstStack,
                                 WorkIdStorageSmem& workIdStorageSrcSmem,
                                 WorkIdStorageStack& workIdStorageSrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<112>{});
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
      // FmhaTask.h:521
      mCtaIdxQ = currSeqCtaIdx;
      // FmhaTask.h:525
      mCtaIdxKv = int32_t{0};
      // FmhaTask.h:139
      mSeqLenKv = int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                                ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                                : (mBatchIdx)]};
      // FmhaTask.h:582
      int32_t numLoopSteps;
      // FmhaTask.h:592
      int32_t diffKvQ{(mSeqLenKv) - (mSeqLenQ)};
      // FmhaTask.h:597
      int32_t validSeqLenKv;
      // FmhaTask.h:603
      validSeqLenKv = int32_t{
        min((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) + (diffKvQ)) + (params.mNumTokensPerCtaQ),
            mSeqLenKv)};
      // FmhaTask.h:616
      mNumCtasKv =
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, int32_t{1})};
      // FmhaTask.h:630
      if ((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) < (mSeqLenQ)) &&
          ((mCtaIdxKv) < (mNumCtasKv))) {
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
      int32_t pageIdxUb5{(int32_t{((mSeqLenKv) + (int32_t{31})) / (int32_t{32})}) - (int32_t{1})};
      //
      // Loop body.
      //
      // Task.cpp:3392
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset557 = int32_t{0}; loopOffset557 < numLoopSteps;
           loopOffset557 += int32_t{8}) {
        // Task.cpp:3445
        bool const isFirstLoopIter{(loopOffset557) == (int32_t{0})};
        // Task.cpp:3465
        bool const isLastLoopIter{((loopOffset557) + (int32_t{8})) >= (numLoopSteps)};
        //
        // smemPageOffsetsKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset557) >= (int32_t{0})) {
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
          int32_t pageIdx{(((mCtaIdxKv) * (numLoopSteps) + (loopOffset557)) * (int32_t{4})) +
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
          if ((loopOffset557) >= (int32_t{0})) {
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
          int32_t pageIdx{(((mCtaIdxKv) * (numLoopSteps) + (loopOffset557)) * (int32_t{4})) +
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
  // FmhaTask.h:224
  int32_t mCtaIdxQ;
  // FmhaTask.h:226
  int32_t mCtaIdxKv;
  // FmhaTask.h:214
  int32_t mSeqLenKv;
  // FmhaTask.h:216
  int32_t mNumCtasKv;
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
    , // Task.cpp:283
    mCtaIdxQ{mCtaIdxX}
    , // FmhaTask.h:517
    mCtaIdxKv{int32_t{0}}
    , // FmhaTask.h:437
    mSeqLenKv{int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                            ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                            : (mBatchIdx)]}}
    , // Kernel.cpp:212
    mNumCtasKv{int32_t{1}} {}
  // Task.cpp:522
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:547
    return ((state.mWarpIdx) >= (int32_t{11})) && ((state.mWarpIdx) < (int32_t{12}));
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
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<112>{});
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
      1,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemQProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    // Task.cpp:2033
    int32_t smemQProdToken{int32_t{1}};
    // Task.cpp:2013
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      4,
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
    // SmemKv.h:749
    int32_t smemVoteIdx4{int32_t{0}};
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
      // FmhaTask.h:521
      mCtaIdxQ = currSeqCtaIdx;
      // FmhaTask.h:525
      mCtaIdxKv = int32_t{0};
      // FmhaTask.h:139
      mSeqLenKv = int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                                ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                                : (mBatchIdx)]};
      // FmhaTask.h:582
      int32_t numLoopSteps;
      // FmhaTask.h:592
      int32_t diffKvQ{(mSeqLenKv) - (mSeqLenQ)};
      // FmhaTask.h:597
      int32_t validSeqLenKv;
      // FmhaTask.h:603
      validSeqLenKv = int32_t{
        min((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) + (diffKvQ)) + (params.mNumTokensPerCtaQ),
            mSeqLenKv)};
      // FmhaTask.h:616
      mNumCtasKv =
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, int32_t{1})};
      // FmhaTask.h:630
      if ((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) < (mSeqLenQ)) &&
          ((mCtaIdxKv) < (mNumCtasKv))) {
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
      // SmemKv.h:668
      cutlass::AlignedArray<int32_t, 4> pageOffsets4;
      // SmemKv.h:683
      bool skipsLoadingV4;
      //
      // Hoist the first iter.
      //
      //
      // workIdThrottleBarrier [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{4608}].
      //
      // Task.cpp:1607
      // Task.cpp:5064
      {
        // Task.cpp:5078
        {
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
      // workIdThrottleBarrier [ProdWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{4608}].
      //
      //
      // workIdThrottleBarrier [ProdCommit, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{4608}].
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
      // gmemQ [ConsWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // GmemQkv.h:82
      int32_t idxQ00;
      // GmemQkv.h:83
      idxQ00 = mCtaIdxQ;
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
        coords[int32_t{3}] = (prodIdxQ02) * (params.mNumTokensPerCtaQ) + (mSeqOffsetQ);
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
        // SmemTile.cpp:620
        coords[int32_t{0}] += int32_t{64};
        // SmemTile.cpp:611
        if (bool{cute::elect_one_sync()}) {
          // CudaPtx.h:48
          cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                         cuda_ptx::space_global_t{},
                                         &smemQDstSmem.mArray[index][int32_t{4096}],
                                         &params.tmaQ_,
                                         coords,
                                         barrier);
        }
        // SmemTile.cpp:620
        coords[int32_t{0}] += int32_t{64};
        // SmemTile.cpp:611
        if (bool{cute::elect_one_sync()}) {
          // CudaPtx.h:48
          cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                         cuda_ptx::space_global_t{},
                                         &smemQDstSmem.mArray[index][int32_t{8192}],
                                         &params.tmaQ_,
                                         coords,
                                         barrier);
        }
        // SmemTile.cpp:620
        coords[int32_t{0}] += int32_t{64};
        // SmemTile.cpp:611
        if (bool{cute::elect_one_sync()}) {
          // CudaPtx.h:48
          cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                         cuda_ptx::space_global_t{},
                                         &smemQDstSmem.mArray[index][int32_t{12288}],
                                         &params.tmaQ_,
                                         coords,
                                         barrier);
        }
        // SmemQ.h:327
        if (bool{cute::elect_one_sync()}) {
          // SmemQ.h:329
          trtllm::dev::completeTransaction(
            barrier,
            ((int32_t{64}) - ((params.mNumTokensPerCtaQ) * (params.mNumHeadsQPerKv))) *
              (int32_t{512}));
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
      // smemPageOffsetsKv [ConsWait, FirstIter, FreqInfo{0, 8}, UserTags{1}, Flags{0}].
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
      // smemPageOffsetsKv [ConsWork (call 0), FirstIter, FreqInfo{0, 8}, UserTags{1}, Flags{0}].
      //
      // SmemPageOffsetsKv.h:320
      int32_t* ptrSmemPageOffsetsK5;
      // SmemPageOffsetsKv.h:326
      int32_t* ptrSmemPageOffsetsV5;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{smemPageOffsetsKvConsState.index()};
        // SmemPageOffsetsKv.h:349
        ptrSmemPageOffsetsK5 = smemPageOffsetsKvSrcSmem.mArray[index];
        // Task.cpp:43
        ++smemPageOffsetsKvConsState;
      }
      //
      // smemKv [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
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
      // smemKv [ProdWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // SmemKv.h:772
      int32_t* ptrSmemPageOffsetsK4;
      // SmemKv.h:786
      int32_t* ptrSmemPageOffsetsV4;
      // Task.cpp:1511
      ptrSmemPageOffsetsK4 = ptrSmemPageOffsetsK5;
      // Task.cpp:1511
      ptrSmemPageOffsetsV4 = ptrSmemPageOffsetsV5;
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
        // SmemKv.h:1678
        {
          // SmemKv.h:1695
          cutlass::AlignedArray<int32_t, 4> localPageOffsets04;
          // SmemKv.h:1711
          localPageOffsets04 = reinterpret_cast<cutlass::AlignedArray<int32_t, 4>*>(
            (ptrSmemPageOffsetsK4 + int32_t{0}))[int32_t{0}];
          // SmemKv.h:1731
          pageOffsets4[int32_t{0}] = localPageOffsets04[int32_t{0}];
          // SmemKv.h:1731
          pageOffsets4[int32_t{1}] = localPageOffsets04[int32_t{1}];
          // SmemKv.h:1731
          pageOffsets4[int32_t{2}] = localPageOffsets04[int32_t{2}];
          // SmemKv.h:1731
          pageOffsets4[int32_t{3}] = localPageOffsets04[int32_t{3}];
        }
        //
        // Load pageOffsets for headDimStageIdx = 0.
        //
        // SmemKv.h:1236
        {
          // SmemTile.cpp:485
          int32_t coords[4];
          // SmemTile.cpp:492
          coords[int32_t{0}] = headDimOffset;
          // SmemTile.cpp:492
          coords[int32_t{1}] = tokenOffset;
          // SmemTile.cpp:492
          coords[int32_t{2}] = mHeadIdx;
          // SmemTile.cpp:492
          coords[int32_t{3}] = pageOffsets4[int32_t{0}];
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
          // SmemTile.cpp:620
          coords[int32_t{0}] += int32_t{64};
          // SmemTile.cpp:611
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
        // SmemKv.h:1236
        {
          // SmemTile.cpp:485
          int32_t coords[4];
          // SmemTile.cpp:492
          coords[int32_t{0}] = headDimOffset;
          // SmemTile.cpp:492
          coords[int32_t{1}] = tokenOffset;
          // SmemTile.cpp:492
          coords[int32_t{2}] = mHeadIdx;
          // SmemTile.cpp:492
          coords[int32_t{3}] = pageOffsets4[int32_t{1}];
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{2048}],
                                           &params.tmaK_,
                                           coords,
                                           barrier);
          }
          // SmemTile.cpp:620
          coords[int32_t{0}] += int32_t{64};
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{10240}],
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
          coords[int32_t{2}] = mHeadIdx;
          // SmemTile.cpp:492
          coords[int32_t{3}] = pageOffsets4[int32_t{2}];
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
          // SmemTile.cpp:620
          coords[int32_t{0}] += int32_t{64};
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{12288}],
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
          coords[int32_t{2}] = mHeadIdx;
          // SmemTile.cpp:492
          coords[int32_t{3}] = pageOffsets4[int32_t{3}];
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{6144}],
                                           &params.tmaK_,
                                           coords,
                                           barrier);
          }
          // SmemTile.cpp:620
          coords[int32_t{0}] += int32_t{64};
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{14336}],
                                           &params.tmaK_,
                                           coords,
                                           barrier);
          }
        }
      }
      //
      // smemKv [ProdCommit, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
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
      // smemKv [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
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
      // smemKv [ProdWork (call 1), FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1511
      ptrSmemPageOffsetsK4 = ptrSmemPageOffsetsK5;
      // Task.cpp:1511
      ptrSmemPageOffsetsV4 = ptrSmemPageOffsetsV5;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4413
        uint64_t* barrier{smemKvDstStack.mPipeline.producer_get_barrier(smemKvProdState)};
        // Task.cpp:5945
        int32_t index{smemKvProdState.index()};
        // SmemKv.h:1430
        int32_t headDimOffset{int32_t{128}};
        // SmemKv.h:1555
        int32_t tokenOffset{int32_t{0}};
        // SmemKv.h:1678

        //
        // Load pageOffsets for headDimStageIdx = 1.
        //
        // SmemKv.h:1236
        {
          // SmemTile.cpp:485
          int32_t coords[4];
          // SmemTile.cpp:492
          coords[int32_t{0}] = headDimOffset;
          // SmemTile.cpp:492
          coords[int32_t{1}] = tokenOffset;
          // SmemTile.cpp:492
          coords[int32_t{2}] = mHeadIdx;
          // SmemTile.cpp:492
          coords[int32_t{3}] = pageOffsets4[int32_t{0}];
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
          // SmemTile.cpp:620
          coords[int32_t{0}] += int32_t{64};
          // SmemTile.cpp:611
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
        // SmemKv.h:1236
        {
          // SmemTile.cpp:485
          int32_t coords[4];
          // SmemTile.cpp:492
          coords[int32_t{0}] = headDimOffset;
          // SmemTile.cpp:492
          coords[int32_t{1}] = tokenOffset;
          // SmemTile.cpp:492
          coords[int32_t{2}] = mHeadIdx;
          // SmemTile.cpp:492
          coords[int32_t{3}] = pageOffsets4[int32_t{1}];
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{2048}],
                                           &params.tmaK_,
                                           coords,
                                           barrier);
          }
          // SmemTile.cpp:620
          coords[int32_t{0}] += int32_t{64};
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{10240}],
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
          coords[int32_t{2}] = mHeadIdx;
          // SmemTile.cpp:492
          coords[int32_t{3}] = pageOffsets4[int32_t{2}];
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
          // SmemTile.cpp:620
          coords[int32_t{0}] += int32_t{64};
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{12288}],
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
          coords[int32_t{2}] = mHeadIdx;
          // SmemTile.cpp:492
          coords[int32_t{3}] = pageOffsets4[int32_t{3}];
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{6144}],
                                           &params.tmaK_,
                                           coords,
                                           barrier);
          }
          // SmemTile.cpp:620
          coords[int32_t{0}] += int32_t{64};
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{14336}],
                                           &params.tmaK_,
                                           coords,
                                           barrier);
          }
        }
      }
      //
      // smemKv [ProdCommit, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
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
      // smemPageOffsetsKv [ConsRelease, FirstIter, FreqInfo{0, 8}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:3814

      //
      // Loop body.
      //
      // Task.cpp:3392
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset1012 = int32_t{0}; loopOffset1012 < (numLoopSteps) - (int32_t{1});
           ++loopOffset1012) {
        // Task.cpp:3465
        bool const isLastLoopIter{((loopOffset1012) + (int32_t{1})) >=
                                  ((numLoopSteps) - (int32_t{1}))};
        //
        // gmemKv [ConsWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:2928
        {}
        //
        // smemPageOffsetsKv [ConsWait, Info{0}, FreqInfo{1, 8}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:3814
        if ((((loopOffset1012) + (int32_t{1})) % (int32_t{8})) == (int32_t{0})) {
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
        // smemPageOffsetsKv [ConsWork (call 1), Info{0}, FreqInfo{1, 8}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:3814
        if ((((loopOffset1012) + (int32_t{1})) % (int32_t{8})) == (int32_t{0})) {
          // Task.cpp:1607
          // Task.cpp:2928
          {
            // Task.cpp:5945
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
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset1012) >= (int32_t{0})) {
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
        ptrSmemPageOffsetsK4 = ptrSmemPageOffsetsK5;
        // Task.cpp:1511
        ptrSmemPageOffsetsV4 = ptrSmemPageOffsetsV5;
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
          // SmemKv.h:1678
          {
            // SmemKv.h:1695
            cutlass::AlignedArray<int32_t, 4> localPageOffsets04;
            // SmemKv.h:1711
            localPageOffsets04 = reinterpret_cast<cutlass::AlignedArray<int32_t, 4>*>(
              (ptrSmemPageOffsetsK4 +
               (((loopOffset1012) + (int32_t{1})) * (int32_t{4})) % (int32_t{32})))[int32_t{0}];
            // SmemKv.h:1731
            pageOffsets4[int32_t{0}] = localPageOffsets04[int32_t{0}];
            // SmemKv.h:1731
            pageOffsets4[int32_t{1}] = localPageOffsets04[int32_t{1}];
            // SmemKv.h:1731
            pageOffsets4[int32_t{2}] = localPageOffsets04[int32_t{2}];
            // SmemKv.h:1731
            pageOffsets4[int32_t{3}] = localPageOffsets04[int32_t{3}];
          }
          //
          // Load pageOffsets for headDimStageIdx = 0.
          //
          // SmemKv.h:1236
          {
            // SmemTile.cpp:485
            int32_t coords[4];
            // SmemTile.cpp:492
            coords[int32_t{0}] = headDimOffset;
            // SmemTile.cpp:492
            coords[int32_t{1}] = tokenOffset;
            // SmemTile.cpp:492
            coords[int32_t{2}] = mHeadIdx;
            // SmemTile.cpp:492
            coords[int32_t{3}] = pageOffsets4[int32_t{0}];
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
            // SmemTile.cpp:620
            coords[int32_t{0}] += int32_t{64};
            // SmemTile.cpp:611
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
          // SmemKv.h:1236
          {
            // SmemTile.cpp:485
            int32_t coords[4];
            // SmemTile.cpp:492
            coords[int32_t{0}] = headDimOffset;
            // SmemTile.cpp:492
            coords[int32_t{1}] = tokenOffset;
            // SmemTile.cpp:492
            coords[int32_t{2}] = mHeadIdx;
            // SmemTile.cpp:492
            coords[int32_t{3}] = pageOffsets4[int32_t{1}];
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{2048}],
                                             &params.tmaK_,
                                             coords,
                                             barrier);
            }
            // SmemTile.cpp:620
            coords[int32_t{0}] += int32_t{64};
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{10240}],
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
            coords[int32_t{2}] = mHeadIdx;
            // SmemTile.cpp:492
            coords[int32_t{3}] = pageOffsets4[int32_t{2}];
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
            // SmemTile.cpp:620
            coords[int32_t{0}] += int32_t{64};
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{12288}],
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
            coords[int32_t{2}] = mHeadIdx;
            // SmemTile.cpp:492
            coords[int32_t{3}] = pageOffsets4[int32_t{3}];
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{6144}],
                                             &params.tmaK_,
                                             coords,
                                             barrier);
            }
            // SmemTile.cpp:620
            coords[int32_t{0}] += int32_t{64};
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{14336}],
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
        // smemKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset1012) >= (int32_t{0})) {
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
        // smemKv [ProdWork (call 3), Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{0}].
        //
        // Task.cpp:1511
        ptrSmemPageOffsetsK4 = ptrSmemPageOffsetsK5;
        // Task.cpp:1511
        ptrSmemPageOffsetsV4 = ptrSmemPageOffsetsV5;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:4413
          uint64_t* barrier{smemKvDstStack.mPipeline.producer_get_barrier(smemKvProdState)};
          // Task.cpp:5945
          int32_t index{smemKvProdState.index()};
          // SmemKv.h:1430
          int32_t headDimOffset{int32_t{128}};
          // SmemKv.h:1555
          int32_t tokenOffset{int32_t{0}};
          // SmemKv.h:1678

          //
          // Load pageOffsets for headDimStageIdx = 1.
          //
          // SmemKv.h:1236
          {
            // SmemTile.cpp:485
            int32_t coords[4];
            // SmemTile.cpp:492
            coords[int32_t{0}] = headDimOffset;
            // SmemTile.cpp:492
            coords[int32_t{1}] = tokenOffset;
            // SmemTile.cpp:492
            coords[int32_t{2}] = mHeadIdx;
            // SmemTile.cpp:492
            coords[int32_t{3}] = pageOffsets4[int32_t{0}];
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
            // SmemTile.cpp:620
            coords[int32_t{0}] += int32_t{64};
            // SmemTile.cpp:611
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
          // SmemKv.h:1236
          {
            // SmemTile.cpp:485
            int32_t coords[4];
            // SmemTile.cpp:492
            coords[int32_t{0}] = headDimOffset;
            // SmemTile.cpp:492
            coords[int32_t{1}] = tokenOffset;
            // SmemTile.cpp:492
            coords[int32_t{2}] = mHeadIdx;
            // SmemTile.cpp:492
            coords[int32_t{3}] = pageOffsets4[int32_t{1}];
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{2048}],
                                             &params.tmaK_,
                                             coords,
                                             barrier);
            }
            // SmemTile.cpp:620
            coords[int32_t{0}] += int32_t{64};
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{10240}],
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
            coords[int32_t{2}] = mHeadIdx;
            // SmemTile.cpp:492
            coords[int32_t{3}] = pageOffsets4[int32_t{2}];
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
            // SmemTile.cpp:620
            coords[int32_t{0}] += int32_t{64};
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{12288}],
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
            coords[int32_t{2}] = mHeadIdx;
            // SmemTile.cpp:492
            coords[int32_t{3}] = pageOffsets4[int32_t{3}];
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{6144}],
                                             &params.tmaK_,
                                             coords,
                                             barrier);
            }
            // SmemTile.cpp:620
            coords[int32_t{0}] += int32_t{64};
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{14336}],
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
        // smemPageOffsetsKv [ConsRelease, Info{0}, FreqInfo{1, 8}, UserTags{1}, Flags{65536}].
        //
        // Task.cpp:3814
        if ((!(isLastLoopIter)) &&
            ((((loopOffset1012) + (int32_t{1})) % (int32_t{8})) == (int32_t{7}))) {
          // Task.cpp:2568
          if ((loopOffset1012) >= (int32_t{0})) {
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
        // smemPageOffsetsKv [ConsWait, Info{0}, FreqInfo{0, 8}, UserTags{2}, Flags{0}].
        //
        // Task.cpp:3814
        if (((loopOffset1012) % (int32_t{8})) == (int32_t{0})) {
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
        // smemPageOffsetsKv [ConsWork (call 2), Info{0}, FreqInfo{0, 8}, UserTags{2}, Flags{0}].
        //
        // Task.cpp:3814
        if (((loopOffset1012) % (int32_t{8})) == (int32_t{0})) {
          // Task.cpp:1607
          // Task.cpp:2928
          {
            // Task.cpp:5945
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
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset1012) >= (int32_t{0})) {
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
        // smemKv [ProdWork (call 4), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
        //
        // Task.cpp:1511
        ptrSmemPageOffsetsK4 = ptrSmemPageOffsetsK5;
        // Task.cpp:1511
        ptrSmemPageOffsetsV4 = ptrSmemPageOffsetsV5;
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
          // SmemKv.h:1307
          trtllm::dev::CutlassNamedBarrier::sync(160, (int32_t{1}) + (smemVoteIdx4));
          // SmemKv.h:1316
          int32_t voteVal;
          // SmemKv.h:1317
          voteVal = smemKvDstStack.mDepSmemPtr3[smemVoteIdx4];
          // SmemKv.h:1318
          voteVal = uint32_t{__shfl_sync(uint32_t{0xffffffff}, voteVal, int32_t{0}, int32_t{32})};
          // SmemKv.h:1327
          skipsLoadingV4 = (voteVal) == (int32_t{1});
          // SmemKv.h:1334
          smemVoteIdx4 = ((smemVoteIdx4) + (int32_t{1})) % (int32_t{2});
          // SmemKv.h:1342
          if (skipsLoadingV4) {
            // SmemKv.h:1346
            if (cute::elect_one_sync()) {
              // SmemKv.h:1349
              trtllm::dev::completeTransaction(barrier, int32_t{32768});
            }
          } else {
            // SmemKv.h:1678
            {
              // SmemKv.h:1695
              cutlass::AlignedArray<int32_t, 4> localPageOffsets04;
              // SmemKv.h:1711
              localPageOffsets04 = reinterpret_cast<cutlass::AlignedArray<int32_t, 4>*>(
                (ptrSmemPageOffsetsV4 +
                 ((loopOffset1012) * (int32_t{4})) % (int32_t{32})))[int32_t{0}];
              // SmemKv.h:1731
              pageOffsets4[int32_t{0}] = localPageOffsets04[int32_t{0}];
              // SmemKv.h:1731
              pageOffsets4[int32_t{1}] = localPageOffsets04[int32_t{1}];
              // SmemKv.h:1731
              pageOffsets4[int32_t{2}] = localPageOffsets04[int32_t{2}];
              // SmemKv.h:1731
              pageOffsets4[int32_t{3}] = localPageOffsets04[int32_t{3}];
            }
            //
            // Load pageOffsets for headDimStageIdx = 0.
            //
            // SmemKv.h:1236
            {
              // SmemTile.cpp:485
              int32_t coords[4];
              // SmemTile.cpp:492
              coords[int32_t{0}] = headDimOffset;
              // SmemTile.cpp:492
              coords[int32_t{1}] = tokenOffset;
              // SmemTile.cpp:492
              coords[int32_t{2}] = mHeadIdx;
              // SmemTile.cpp:492
              coords[int32_t{3}] = pageOffsets4[int32_t{0}];
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
              // SmemTile.cpp:620
              coords[int32_t{0}] += int32_t{64};
              // SmemTile.cpp:611
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
            // SmemKv.h:1236
            {
              // SmemTile.cpp:485
              int32_t coords[4];
              // SmemTile.cpp:492
              coords[int32_t{0}] = headDimOffset;
              // SmemTile.cpp:492
              coords[int32_t{1}] = tokenOffset;
              // SmemTile.cpp:492
              coords[int32_t{2}] = mHeadIdx;
              // SmemTile.cpp:492
              coords[int32_t{3}] = pageOffsets4[int32_t{1}];
              // SmemTile.cpp:611
              if (bool{cute::elect_one_sync()}) {
                // CudaPtx.h:48
                cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                               cuda_ptx::space_global_t{},
                                               &smemKvDstSmem.mArray[index][int32_t{2048}],
                                               &params.tmaV_,
                                               coords,
                                               barrier);
              }
              // SmemTile.cpp:620
              coords[int32_t{0}] += int32_t{64};
              // SmemTile.cpp:611
              if (bool{cute::elect_one_sync()}) {
                // CudaPtx.h:48
                cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                               cuda_ptx::space_global_t{},
                                               &smemKvDstSmem.mArray[index][int32_t{10240}],
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
              coords[int32_t{2}] = mHeadIdx;
              // SmemTile.cpp:492
              coords[int32_t{3}] = pageOffsets4[int32_t{2}];
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
              // SmemTile.cpp:620
              coords[int32_t{0}] += int32_t{64};
              // SmemTile.cpp:611
              if (bool{cute::elect_one_sync()}) {
                // CudaPtx.h:48
                cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                               cuda_ptx::space_global_t{},
                                               &smemKvDstSmem.mArray[index][int32_t{12288}],
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
              coords[int32_t{2}] = mHeadIdx;
              // SmemTile.cpp:492
              coords[int32_t{3}] = pageOffsets4[int32_t{3}];
              // SmemTile.cpp:611
              if (bool{cute::elect_one_sync()}) {
                // CudaPtx.h:48
                cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                               cuda_ptx::space_global_t{},
                                               &smemKvDstSmem.mArray[index][int32_t{6144}],
                                               &params.tmaV_,
                                               coords,
                                               barrier);
              }
              // SmemTile.cpp:620
              coords[int32_t{0}] += int32_t{64};
              // SmemTile.cpp:611
              if (bool{cute::elect_one_sync()}) {
                // CudaPtx.h:48
                cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                               cuda_ptx::space_global_t{},
                                               &smemKvDstSmem.mArray[index][int32_t{14336}],
                                               &params.tmaV_,
                                               coords,
                                               barrier);
              }
            }
          }
        }
        //
        // smemKv [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
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
        // smemKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset1012) >= (int32_t{0})) {
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
        // smemKv [ProdWork (call 5), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
        //
        // Task.cpp:1511
        ptrSmemPageOffsetsK4 = ptrSmemPageOffsetsK5;
        // Task.cpp:1511
        ptrSmemPageOffsetsV4 = ptrSmemPageOffsetsV5;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:4413
          uint64_t* barrier{smemKvDstStack.mPipeline.producer_get_barrier(smemKvProdState)};
          // Task.cpp:5945
          int32_t index{smemKvProdState.index()};
          // SmemKv.h:1430
          int32_t headDimOffset{int32_t{128}};
          // SmemKv.h:1555
          int32_t tokenOffset{int32_t{0}};
          // SmemKv.h:1342
          if (skipsLoadingV4) {
            // SmemKv.h:1346
            if (cute::elect_one_sync()) {
              // SmemKv.h:1349
              trtllm::dev::completeTransaction(barrier, int32_t{32768});
            }
          } else {
            // SmemKv.h:1678

            //
            // Load pageOffsets for headDimStageIdx = 1.
            //
            // SmemKv.h:1236
            {
              // SmemTile.cpp:485
              int32_t coords[4];
              // SmemTile.cpp:492
              coords[int32_t{0}] = headDimOffset;
              // SmemTile.cpp:492
              coords[int32_t{1}] = tokenOffset;
              // SmemTile.cpp:492
              coords[int32_t{2}] = mHeadIdx;
              // SmemTile.cpp:492
              coords[int32_t{3}] = pageOffsets4[int32_t{0}];
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
              // SmemTile.cpp:620
              coords[int32_t{0}] += int32_t{64};
              // SmemTile.cpp:611
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
            // SmemKv.h:1236
            {
              // SmemTile.cpp:485
              int32_t coords[4];
              // SmemTile.cpp:492
              coords[int32_t{0}] = headDimOffset;
              // SmemTile.cpp:492
              coords[int32_t{1}] = tokenOffset;
              // SmemTile.cpp:492
              coords[int32_t{2}] = mHeadIdx;
              // SmemTile.cpp:492
              coords[int32_t{3}] = pageOffsets4[int32_t{1}];
              // SmemTile.cpp:611
              if (bool{cute::elect_one_sync()}) {
                // CudaPtx.h:48
                cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                               cuda_ptx::space_global_t{},
                                               &smemKvDstSmem.mArray[index][int32_t{2048}],
                                               &params.tmaV_,
                                               coords,
                                               barrier);
              }
              // SmemTile.cpp:620
              coords[int32_t{0}] += int32_t{64};
              // SmemTile.cpp:611
              if (bool{cute::elect_one_sync()}) {
                // CudaPtx.h:48
                cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                               cuda_ptx::space_global_t{},
                                               &smemKvDstSmem.mArray[index][int32_t{10240}],
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
              coords[int32_t{2}] = mHeadIdx;
              // SmemTile.cpp:492
              coords[int32_t{3}] = pageOffsets4[int32_t{2}];
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
              // SmemTile.cpp:620
              coords[int32_t{0}] += int32_t{64};
              // SmemTile.cpp:611
              if (bool{cute::elect_one_sync()}) {
                // CudaPtx.h:48
                cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                               cuda_ptx::space_global_t{},
                                               &smemKvDstSmem.mArray[index][int32_t{12288}],
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
              coords[int32_t{2}] = mHeadIdx;
              // SmemTile.cpp:492
              coords[int32_t{3}] = pageOffsets4[int32_t{3}];
              // SmemTile.cpp:611
              if (bool{cute::elect_one_sync()}) {
                // CudaPtx.h:48
                cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                               cuda_ptx::space_global_t{},
                                               &smemKvDstSmem.mArray[index][int32_t{6144}],
                                               &params.tmaV_,
                                               coords,
                                               barrier);
              }
              // SmemTile.cpp:620
              coords[int32_t{0}] += int32_t{64};
              // SmemTile.cpp:611
              if (bool{cute::elect_one_sync()}) {
                // CudaPtx.h:48
                cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                               cuda_ptx::space_global_t{},
                                               &smemKvDstSmem.mArray[index][int32_t{14336}],
                                               &params.tmaV_,
                                               coords,
                                               barrier);
              }
            }
          }
        }
        //
        // smemKv [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
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
        // smemPageOffsetsKv [ConsRelease, Info{0}, FreqInfo{0, 8}, UserTags{2}, Flags{0}].
        //
        // Task.cpp:3814
        if (((loopOffset1012) % (int32_t{8})) == (int32_t{7})) {
          // Task.cpp:2568
          if ((loopOffset1012) >= (int32_t{0})) {
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
        lastLoopOffset = loopOffset1012;
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
      // smemPageOffsetsKv [ConsRelease, LastIter, FreqInfo{0, 8}, UserTags{1}, Flags{0}].
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
      // smemPageOffsetsKv [ConsWait, LastIter, FreqInfo{0, 8}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:3814
      if (((lastLoopOffset) % (int32_t{8})) == (int32_t{0})) {
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
      }
      //
      // smemPageOffsetsKv [ConsWork (call 3), LastIter, FreqInfo{0, 8}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:3814
      if (((lastLoopOffset) % (int32_t{8})) == (int32_t{0})) {
        // Task.cpp:1607
        if (hasOneLoopIter) {
          // Task.cpp:5945
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
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5078
        if ((lastLoopOffset) >= (int32_t{0})) {
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
      // smemKv [ProdWork (call 6), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1511
      ptrSmemPageOffsetsK4 = ptrSmemPageOffsetsK5;
      // Task.cpp:1511
      ptrSmemPageOffsetsV4 = ptrSmemPageOffsetsV5;
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
        // SmemKv.h:1307
        trtllm::dev::CutlassNamedBarrier::sync(160, (int32_t{1}) + (smemVoteIdx4));
        // SmemKv.h:1316
        int32_t voteVal;
        // SmemKv.h:1317
        voteVal = smemKvDstStack.mDepSmemPtr3[smemVoteIdx4];
        // SmemKv.h:1318
        voteVal = uint32_t{__shfl_sync(uint32_t{0xffffffff}, voteVal, int32_t{0}, int32_t{32})};
        // SmemKv.h:1327
        skipsLoadingV4 = (voteVal) == (int32_t{1});
        // SmemKv.h:1334
        smemVoteIdx4 = ((smemVoteIdx4) + (int32_t{1})) % (int32_t{2});
        // SmemKv.h:1342
        if (skipsLoadingV4) {
          // SmemKv.h:1346
          if (cute::elect_one_sync()) {
            // SmemKv.h:1349
            trtllm::dev::completeTransaction(barrier, int32_t{32768});
          }
        } else {
          // SmemKv.h:1678
          {
            // SmemKv.h:1695
            cutlass::AlignedArray<int32_t, 4> localPageOffsets04;
            // SmemKv.h:1711
            localPageOffsets04 = reinterpret_cast<cutlass::AlignedArray<int32_t, 4>*>(
              (ptrSmemPageOffsetsV4 +
               ((lastLoopOffset) * (int32_t{4})) % (int32_t{32})))[int32_t{0}];
            // SmemKv.h:1731
            pageOffsets4[int32_t{0}] = localPageOffsets04[int32_t{0}];
            // SmemKv.h:1731
            pageOffsets4[int32_t{1}] = localPageOffsets04[int32_t{1}];
            // SmemKv.h:1731
            pageOffsets4[int32_t{2}] = localPageOffsets04[int32_t{2}];
            // SmemKv.h:1731
            pageOffsets4[int32_t{3}] = localPageOffsets04[int32_t{3}];
          }
          //
          // Load pageOffsets for headDimStageIdx = 0.
          //
          // SmemKv.h:1236
          {
            // SmemTile.cpp:485
            int32_t coords[4];
            // SmemTile.cpp:492
            coords[int32_t{0}] = headDimOffset;
            // SmemTile.cpp:492
            coords[int32_t{1}] = tokenOffset;
            // SmemTile.cpp:492
            coords[int32_t{2}] = mHeadIdx;
            // SmemTile.cpp:492
            coords[int32_t{3}] = pageOffsets4[int32_t{0}];
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
            // SmemTile.cpp:620
            coords[int32_t{0}] += int32_t{64};
            // SmemTile.cpp:611
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
          // SmemKv.h:1236
          {
            // SmemTile.cpp:485
            int32_t coords[4];
            // SmemTile.cpp:492
            coords[int32_t{0}] = headDimOffset;
            // SmemTile.cpp:492
            coords[int32_t{1}] = tokenOffset;
            // SmemTile.cpp:492
            coords[int32_t{2}] = mHeadIdx;
            // SmemTile.cpp:492
            coords[int32_t{3}] = pageOffsets4[int32_t{1}];
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{2048}],
                                             &params.tmaV_,
                                             coords,
                                             barrier);
            }
            // SmemTile.cpp:620
            coords[int32_t{0}] += int32_t{64};
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{10240}],
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
            coords[int32_t{2}] = mHeadIdx;
            // SmemTile.cpp:492
            coords[int32_t{3}] = pageOffsets4[int32_t{2}];
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
            // SmemTile.cpp:620
            coords[int32_t{0}] += int32_t{64};
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{12288}],
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
            coords[int32_t{2}] = mHeadIdx;
            // SmemTile.cpp:492
            coords[int32_t{3}] = pageOffsets4[int32_t{3}];
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{6144}],
                                             &params.tmaV_,
                                             coords,
                                             barrier);
            }
            // SmemTile.cpp:620
            coords[int32_t{0}] += int32_t{64};
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{14336}],
                                             &params.tmaV_,
                                             coords,
                                             barrier);
            }
          }
        }
      }
      //
      // smemKv [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
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
      // smemKv [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5078
        if ((lastLoopOffset) >= (int32_t{0})) {
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
      // smemKv [ProdWork (call 7), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1511
      ptrSmemPageOffsetsK4 = ptrSmemPageOffsetsK5;
      // Task.cpp:1511
      ptrSmemPageOffsetsV4 = ptrSmemPageOffsetsV5;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4413
        uint64_t* barrier{smemKvDstStack.mPipeline.producer_get_barrier(smemKvProdState)};
        // Task.cpp:5945
        int32_t index{smemKvProdState.index()};
        // SmemKv.h:1430
        int32_t headDimOffset{int32_t{128}};
        // SmemKv.h:1555
        int32_t tokenOffset{int32_t{0}};
        // SmemKv.h:1342
        if (skipsLoadingV4) {
          // SmemKv.h:1346
          if (cute::elect_one_sync()) {
            // SmemKv.h:1349
            trtllm::dev::completeTransaction(barrier, int32_t{32768});
          }
        } else {
          // SmemKv.h:1678

          //
          // Load pageOffsets for headDimStageIdx = 1.
          //
          // SmemKv.h:1236
          {
            // SmemTile.cpp:485
            int32_t coords[4];
            // SmemTile.cpp:492
            coords[int32_t{0}] = headDimOffset;
            // SmemTile.cpp:492
            coords[int32_t{1}] = tokenOffset;
            // SmemTile.cpp:492
            coords[int32_t{2}] = mHeadIdx;
            // SmemTile.cpp:492
            coords[int32_t{3}] = pageOffsets4[int32_t{0}];
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
            // SmemTile.cpp:620
            coords[int32_t{0}] += int32_t{64};
            // SmemTile.cpp:611
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
          // SmemKv.h:1236
          {
            // SmemTile.cpp:485
            int32_t coords[4];
            // SmemTile.cpp:492
            coords[int32_t{0}] = headDimOffset;
            // SmemTile.cpp:492
            coords[int32_t{1}] = tokenOffset;
            // SmemTile.cpp:492
            coords[int32_t{2}] = mHeadIdx;
            // SmemTile.cpp:492
            coords[int32_t{3}] = pageOffsets4[int32_t{1}];
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{2048}],
                                             &params.tmaV_,
                                             coords,
                                             barrier);
            }
            // SmemTile.cpp:620
            coords[int32_t{0}] += int32_t{64};
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{10240}],
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
            coords[int32_t{2}] = mHeadIdx;
            // SmemTile.cpp:492
            coords[int32_t{3}] = pageOffsets4[int32_t{2}];
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
            // SmemTile.cpp:620
            coords[int32_t{0}] += int32_t{64};
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{12288}],
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
            coords[int32_t{2}] = mHeadIdx;
            // SmemTile.cpp:492
            coords[int32_t{3}] = pageOffsets4[int32_t{3}];
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{6144}],
                                             &params.tmaV_,
                                             coords,
                                             barrier);
            }
            // SmemTile.cpp:620
            coords[int32_t{0}] += int32_t{64};
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{14336}],
                                             &params.tmaV_,
                                             coords,
                                             barrier);
            }
          }
        }
      }
      //
      // smemKv [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
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
      // smemPageOffsetsKv [ConsRelease, LastIter, FreqInfo{0, 8}, UserTags{2}, Flags{0}].
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
  int32_t const mLdtm16dp32bitTmemColIdx;
  // TmemTile.cpp:445
  int32_t const mLdtm16dp32bitTmemRowIdx;
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
    , // Task.cpp:283
    mCtaIdxQ{mCtaIdxX}
    , // FmhaTask.h:517
    mCtaIdxKv{int32_t{0}}
    , // FmhaTask.h:437
    mSeqLenKv{int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                            ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                            : (mBatchIdx)]}}
    , // Kernel.cpp:212
    mNumCtasKv{int32_t{1}}
    , // Task.cpp:379
    mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))}
    , // TmemTile.cpp:432
    mLdtm16dp32bitTmemColIdx{
      trtllm::dev::ldst16dp32bitTmemColIdx<int32_t{64}>((mWarpGrpThreadIdx) % (int32_t{128}))}
    , // TmemTile.cpp:453
    mLdtm16dp32bitTmemRowIdx{
      trtllm::dev::ldst16dp32bitTmemRowIdx<int32_t{16}>((mWarpGrpThreadIdx) % (int32_t{128}))}
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
    cuda_ptx::setmaxnreg_inc(cuda_ptx::n32_t<200>{});
    // Task.cpp:2114
    trtllm::dev::CutlassUmmaAsyncPipeline<
      2,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState tmemS0ConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassUmmaAsyncPipeline<2,
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
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState tmemSoftmaxLocal0ProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    // Task.cpp:2033
    int32_t tmemSoftmaxLocal0ProdToken{int32_t{1}};
    // Task.cpp:2013
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState tmemP0ProdState{int32_t{0},
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
      // FmhaTask.h:521
      mCtaIdxQ = currSeqCtaIdx;
      // FmhaTask.h:525
      mCtaIdxKv = int32_t{0};
      // FmhaTask.h:139
      mSeqLenKv = int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                                ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                                : (mBatchIdx)]};
      // FmhaTask.h:582
      int32_t numLoopSteps;
      // FmhaTask.h:592
      int32_t diffKvQ{(mSeqLenKv) - (mSeqLenQ)};
      // FmhaTask.h:597
      int32_t validSeqLenKv;
      // FmhaTask.h:603
      validSeqLenKv = int32_t{
        min((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) + (diffKvQ)) + (params.mNumTokensPerCtaQ),
            mSeqLenKv)};
      // FmhaTask.h:616
      mNumCtasKv =
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, int32_t{1})};
      // FmhaTask.h:630
      if ((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) < (mSeqLenQ)) &&
          ((mCtaIdxKv) < (mNumCtasKv))) {
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
      float oldMaxArray8[1];
      // TmemS.h:660
      float sumArray8[1]{float{0}};
      // TmemS.h:672
      float newMaxArray8[1]{float{-3.4028235e+38}};
      // TmemTile.cpp:373
      cutlass::Array<float, 64> regsQk;
      // TmemS.h:746
      float adjustedSkipSoftmaxThreshold{(params.mSkipSoftmaxThresholdScaleFactor) /
                                         (static_cast<float>(mSeqLenKv))};
      // TmemS.h:763
      cudaGridDependencySynchronize();
      // TmemS.h:770
      float scaleSoftmaxLog28;
      // TmemS.h:775
      scaleSoftmaxLog28 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
                            ? (params.mScaleSoftmaxLog2)
                            : (float{params.ptrScaleSoftmaxLog2[int32_t{0}]});
      // TmemSoftmax.h:515
      cudaGridDependencySynchronize();
      // TmemSoftmax.h:524
      float scaleSoftmaxLog210;
      // TmemSoftmax.h:529
      scaleSoftmaxLog210 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
                             ? (params.mScaleSoftmaxLog2)
                             : (float{params.ptrScaleSoftmaxLog2[int32_t{0}]});
      // TmemTile.cpp:373
      cutlass::Array<uint32_t, 32> regsP;
      // TmemP.h:534
      cudaGridDependencySynchronize();
      // TmemP.h:541
      float scaleSoftmaxLog212;
      // TmemP.h:546
      scaleSoftmaxLog212 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
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
      for (int32_t loopOffset1867 = int32_t{0}; loopOffset1867 < numLoopSteps; ++loopOffset1867) {
        // Task.cpp:3465
        bool const isLastLoopIter{((loopOffset1867) + (int32_t{1})) >= (numLoopSteps)};
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
        float* oldMaxPtr8;
        // TmemS.h:1065
        float* sumPtr8;
        // TmemS.h:1070
        float* newMaxPtr08;
        // TmemS.h:1075
        float* qkPtr08;
        // TmemS.h:1080
        float* newMaxPtr18;
        // TmemS.h:1085
        float* qkPtr18;
        // TmemS.h:1102
        bool warpSkipsSoftmax8;
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{tmemS0ConsState.index()};
          // TmemS.h:1192
          oldMaxPtr8 = oldMaxArray8;
          // TmemS.h:1194
          sumPtr8 = sumArray8;
          // TmemS.h:1196
          newMaxPtr08 = newMaxArray8;
          // TmemS.h:1198
          newMaxPtr18 = newMaxArray8;
          // TmemS.h:1207
          float reducedMaxArray8[1];
          // TmemS.h:1214
          reducedMaxArray8[int32_t{0}] = float{-3.4028235e+38};
          // TmemS.h:1246
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset1891 = int32_t{0}; loopOffset1891 < int32_t{1}; ++loopOffset1891) {
            // TmemS.h:1257
            oldMaxArray8[loopOffset1891] = newMaxArray8[loopOffset1891];
          }
          // TmemS.h:1275
          float ilpMax0{float{-3.4028235e+38}};
          // TmemS.h:1275
          float ilpMax1{float{-3.4028235e+38}};
          // TmemS.h:1275
          float ilpMax2{float{-3.4028235e+38}};
          // TmemS.h:1275
          float ilpMax3{float{-3.4028235e+38}};
          // TmemS.h:2043
          if ((mWarpGrpThreadIdx) == (int32_t{0})) {
            // TmemS.h:2055
            tmemS0SrcStack.mDepSmemPtr3[index] = int32_t{1};
          }
          // TmemS.h:2065
          trtllm::dev::CutlassNamedBarrier::sync(128, (int32_t{3}) + (index));
          //
          // The causal mask block.
          //
          // Mask.h:568
          int32_t const tileOffsetK{
            (((numLoopSteps) * (mCtaIdxKv) + (loopOffset1867)) * (int32_t{1})) * (int32_t{128})};
          // Mask.h:1925
          bool isMaskSkipped{
            ((tileOffsetK) + (int32_t{128})) <=
            (((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) + ((mSeqLenKv) - (mSeqLenQ)))};
          // Mask.h:1936
          if (isMaskSkipped) {
            // TmemTile.cpp:527
            {
              // TmemTile.cpp:529
              uint32_t tmemBasePtr{mTmemBaseOffset};
              // TmemTile.cpp:545
              uint32_t(&dstSlice0)[64]{reinterpret_cast<uint32_t(&)[64]>(regsQk[int32_t{0}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_16x32bx2(
                dstSlice0,
                (tmemBasePtr) +
                  (static_cast<uint32_t>((index) * (int32_t{128}) +
                                         (int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                            ? (int32_t{0})
                                            : (int32_t{128})))),
                cuda_ptx::n32_t<64>{});
            }
            // TmemS.h:1681
            CUTLASS_PRAGMA_UNROLL
            for (int32_t loopOffset1913 = int32_t{0}; loopOffset1913 < int32_t{64};
                 loopOffset1913 += int32_t{4}) {
              // TmemS.h:1694
              ilpMax0 = fmaxf(ilpMax0, regsQk[loopOffset1913]);
              // TmemS.h:1694
              ilpMax1 = fmaxf(ilpMax1, regsQk[(loopOffset1913) + (int32_t{1})]);
              // TmemS.h:1694
              ilpMax2 = fmaxf(ilpMax2, regsQk[(loopOffset1913) + (int32_t{2})]);
              // TmemS.h:1694
              ilpMax3 = fmaxf(ilpMax3, regsQk[(loopOffset1913) + (int32_t{3})]);
            }
          } else {
            // TmemTile.cpp:527
            {
              // TmemTile.cpp:529
              uint32_t tmemBasePtr{mTmemBaseOffset};
              // TmemTile.cpp:545
              uint32_t(&dstSlice0)[64]{reinterpret_cast<uint32_t(&)[64]>(regsQk[int32_t{0}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_16x32bx2(
                dstSlice0,
                (tmemBasePtr) +
                  (static_cast<uint32_t>((index) * (int32_t{128}) +
                                         (int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                            ? (int32_t{0})
                                            : (int32_t{128})))),
                cuda_ptx::n32_t<64>{});
            }
            //
            // Apply the causal mask.
            //
            // Mask.h:962
            int32_t const tileOffsetQ{(mCtaIdxQ) * (params.mNumTokensPerCtaQ) +
                                      ((mSeqLenKv) - (mSeqLenQ))};
            // Mask.h:568
            int32_t const tileOffsetK{
              (((numLoopSteps) * (mCtaIdxKv) + (loopOffset1867)) * (int32_t{1})) * (int32_t{128})};
            // Mask.h:1003
            int32_t localTokenIdxQ{(mLdtm16dp32bitTmemRowIdx) / (params.mNumHeadsQPerKvDivisor)};
            // Mask.h:1006
            int32_t tokenIdxQ{(tileOffsetQ) + (localTokenIdxQ)};
            // Mask.h:47
            CUTLASS_PRAGMA_UNROLL
            for (int32_t loopOffset1928 = int32_t{0}; loopOffset1928 < int32_t{64};
                 ++loopOffset1928) {
              // Mask.h:65
              if ((((tileOffsetK) + (mLdtm16dp32bitTmemColIdx)) + (loopOffset1928)) > (tokenIdxQ)) {
                // Mask.h:69
                regsQk[loopOffset1928] = float{-3.4028235e+38};
              }
            }
            // TmemS.h:1681
            CUTLASS_PRAGMA_UNROLL
            for (int32_t loopOffset1934 = int32_t{0}; loopOffset1934 < int32_t{64};
                 loopOffset1934 += int32_t{4}) {
              // TmemS.h:1694
              ilpMax0 = fmaxf(ilpMax0, regsQk[loopOffset1934]);
              // TmemS.h:1694
              ilpMax1 = fmaxf(ilpMax1, regsQk[(loopOffset1934) + (int32_t{1})]);
              // TmemS.h:1694
              ilpMax2 = fmaxf(ilpMax2, regsQk[(loopOffset1934) + (int32_t{2})]);
              // TmemS.h:1694
              ilpMax3 = fmaxf(ilpMax3, regsQk[(loopOffset1934) + (int32_t{3})]);
            }
          }
          // TmemS.h:2197
          ilpMax0 = fmaxf(ilpMax0, ilpMax2);
          // TmemS.h:2197
          ilpMax1 = fmaxf(ilpMax1, ilpMax3);
          // TmemS.h:2216
          reducedMaxArray8[int32_t{0}] = fmaxf(ilpMax0, ilpMax1);
          // TmemS.h:2227
          reducedMaxArray8[int32_t{0}] =
            fmaxf(reducedMaxArray8[int32_t{0}],
                  __shfl_xor_sync(uint32_t{0xffffffff}, reducedMaxArray8[int32_t{0}], int32_t{16}));
          // TmemS.h:570
          bool threadCanSkip{(float{exp2f((scaleSoftmaxLog28) * ((reducedMaxArray8[int32_t{0}]) -
                                                                 (newMaxArray8[int32_t{0}])))}) <
                             (adjustedSkipSoftmaxThreshold)};
          // TmemS.h:615
          bool warpCanSkipSoftmax{__all_sync(uint32_t{-1}, threadCanSkip)};
          // TmemS.h:2263
          warpSkipsSoftmax8 = warpCanSkipSoftmax;
          // TmemS.h:2270
          if (!(warpSkipsSoftmax8)) {
            // TmemS.h:2279
            newMaxArray8[int32_t{0}] =
              fmaxf(reducedMaxArray8[int32_t{0}], newMaxArray8[int32_t{0}]);
          }
          // TmemS.h:2095
          if (cute::elect_one_sync()) {
            // TmemS.h:2132
            atomicAnd(&tmemS0SrcStack.mDepSmemPtr3[index], warpSkipsSoftmax8);
          }
          // TmemS.h:2147
          trtllm::dev::CutlassNamedBarrier::arrive(160, (int32_t{1}) + (index));
          // TmemS.h:1334
          qkPtr08 = &regsQk[int32_t{0}];
          // TmemS.h:1336
          qkPtr18 = &regsQk[int32_t{0}];
          // Task.cpp:43
          ++tmemS0ConsState;
        }
        //
        // tmemSoftmaxLocal0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{8}, Flags{0}].
        //
        // TmemSoftmax.h:261
        float* oldMaxPtr9;
        // TmemSoftmax.h:267
        float* sumPtr9;
        // TmemSoftmax.h:273
        float* newMaxPtr9;
        // Task.cpp:1511
        oldMaxPtr9 = oldMaxPtr8;
        // Task.cpp:1511
        sumPtr9 = sumPtr8;
        // Task.cpp:1511
        newMaxPtr9 = newMaxPtr08;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{tmemSoftmaxLocal0ProdState.index()};
          // TmemTile.cpp:373
          cutlass::Array<float, 2> stats;
          // TmemSoftmax.h:365
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset1969 = int32_t{0}; loopOffset1969 < int32_t{1}; ++loopOffset1969) {
            // TmemSoftmax.h:382
            stats[loopOffset1969] = oldMaxPtr9[loopOffset1969];
            // TmemSoftmax.h:384
            stats[(loopOffset1969) + (int32_t{1})] = newMaxPtr9[loopOffset1969];
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
                (static_cast<uint32_t>((index) * (int32_t{128}) +
                                       (int32_t((tmemSoftmaxLocal0DstStack.mInstId) == (int32_t{0}))
                                          ? (int32_t{0})
                                          : (int32_t{128})))),
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
        float* newMaxPtr12;
        // TmemP.h:574
        float* regsFp32P12;
        // TmemP.h:589
        bool warpSkipsSoftmax12;
        // Task.cpp:1511
        newMaxPtr12 = newMaxPtr18;
        // Task.cpp:1511
        regsFp32P12 = qkPtr18;
        // Task.cpp:1511
        warpSkipsSoftmax12 = warpSkipsSoftmax8;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{tmemP0ProdState.index()};
          // TmemP.h:1181
          if (warpSkipsSoftmax12) {
            // TmemP.h:2350
            CUTLASS_PRAGMA_UNROLL
            for (int32_t loopOffset2001 = int32_t{0}; loopOffset2001 < int32_t{32};
                 ++loopOffset2001) {
              // TmemP.h:2354
              regsP[loopOffset2001] = int32_t{0};
            }
          } else {
            // TmemP.h:1025
            float negScaledMaxArray[1];
            // TmemP.h:1128
            float newMax{newMaxPtr12[int32_t{0}]};
            // TmemP.h:1134
            float negScaledMax{-((newMax) * (scaleSoftmaxLog212))};
            // TmemP.h:1144
            negScaledMaxArray[int32_t{0}] = negScaledMax;
            // TmemP.h:1655
            {
              // TmemP.h:1658
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{0}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{1}];
              // TmemP.h:801
              vals[int32_t{0}] =
                (log2Scale2[int32_t{0}]) * (vals[int32_t{0}]) + (negScaledMax[int32_t{0}]);
              // TmemP.h:810
              vals[int32_t{1}] =
                (log2Scale2[int32_t{1}]) * (vals[int32_t{1}]) + (negScaledMax[int32_t{1}]);
              // TmemP.h:833
              regsFp32P12[int32_t{0}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{1}] = vals[int32_t{1}];
            }
            // TmemP.h:1655
            {
              // TmemP.h:1658
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{2}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{3}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{2}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{3}] = vals[int32_t{1}];
            }
            // TmemP.h:1773
            regsFp32P12[int32_t{0}] = exp2f(regsFp32P12[int32_t{0}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{4}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{5}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{4}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{5}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{1}] = exp2f(regsFp32P12[int32_t{1}]);
            // TmemP.h:1773
            regsFp32P12[int32_t{2}] = exp2f(regsFp32P12[int32_t{2}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{6}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{7}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{6}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{7}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{3}] = exp2f(regsFp32P12[int32_t{3}]);
            // TmemP.h:1773
            regsFp32P12[int32_t{4}] = exp2f(regsFp32P12[int32_t{4}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{8}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{9}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{8}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{9}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{5}] = exp2f(regsFp32P12[int32_t{5}]);
            // TmemP.h:1773
            regsFp32P12[int32_t{6}] = exp2f(regsFp32P12[int32_t{6}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{10}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{11}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{10}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{11}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{7}] = exp2f(regsFp32P12[int32_t{7}]);
            // TmemP.h:1773
            regsFp32P12[int32_t{8}] = exp2f(regsFp32P12[int32_t{8}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{12}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{13}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{12}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{13}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{9}] = exp2f(regsFp32P12[int32_t{9}]);
            // TmemP.h:1773
            regsFp32P12[int32_t{10}] = exp2f(regsFp32P12[int32_t{10}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{14}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{15}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{14}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{15}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{11}] = exp2f(regsFp32P12[int32_t{11}]);
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
              elt0 = regsFp32P12[int32_t{0}];
              // TmemP.h:721
              elt1 = regsFp32P12[int32_t{1}];
              // TmemP.h:722
              elt2 = regsFp32P12[int32_t{2}];
              // TmemP.h:723
              elt3 = regsFp32P12[int32_t{3}];
              // TmemP.h:745
              regsP[int32_t{0}] = trtllm::dev::convert_float2_to_bfloat16(elt0, elt1);
              // TmemP.h:745
              regsP[int32_t{1}] = trtllm::dev::convert_float2_to_bfloat16(elt2, elt3);
            }
            // TmemP.h:1773
            regsFp32P12[int32_t{12}] = exp2f(regsFp32P12[int32_t{12}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{16}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{17}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{16}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{17}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{13}] = exp2f(regsFp32P12[int32_t{13}]);
            // TmemP.h:1773
            regsFp32P12[int32_t{14}] = exp2f(regsFp32P12[int32_t{14}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{18}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{19}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{18}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{19}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{15}] = exp2f(regsFp32P12[int32_t{15}]);
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
              elt0 = regsFp32P12[int32_t{4}];
              // TmemP.h:721
              elt1 = regsFp32P12[int32_t{5}];
              // TmemP.h:722
              elt2 = regsFp32P12[int32_t{6}];
              // TmemP.h:723
              elt3 = regsFp32P12[int32_t{7}];
              // TmemP.h:745
              regsP[int32_t{2}] = trtllm::dev::convert_float2_to_bfloat16(elt0, elt1);
              // TmemP.h:745
              regsP[int32_t{3}] = trtllm::dev::convert_float2_to_bfloat16(elt2, elt3);
            }
            // TmemP.h:1773
            regsFp32P12[int32_t{16}] = exp2f(regsFp32P12[int32_t{16}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{20}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{21}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{20}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{21}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{17}] = exp2f(regsFp32P12[int32_t{17}]);
            // TmemP.h:1773
            regsFp32P12[int32_t{18}] = exp2f(regsFp32P12[int32_t{18}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{22}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{23}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{22}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{23}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{19}] = exp2f(regsFp32P12[int32_t{19}]);
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
              elt0 = regsFp32P12[int32_t{8}];
              // TmemP.h:721
              elt1 = regsFp32P12[int32_t{9}];
              // TmemP.h:722
              elt2 = regsFp32P12[int32_t{10}];
              // TmemP.h:723
              elt3 = regsFp32P12[int32_t{11}];
              // TmemP.h:745
              regsP[int32_t{4}] = trtllm::dev::convert_float2_to_bfloat16(elt0, elt1);
              // TmemP.h:745
              regsP[int32_t{5}] = trtllm::dev::convert_float2_to_bfloat16(elt2, elt3);
            }
            // TmemP.h:1773
            regsFp32P12[int32_t{20}] = exp2f(regsFp32P12[int32_t{20}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{24}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{25}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{24}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{25}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{21}] = exp2f(regsFp32P12[int32_t{21}]);
            // TmemP.h:1773
            regsFp32P12[int32_t{22}] = exp2f(regsFp32P12[int32_t{22}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{26}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{27}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{26}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{27}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{23}] = exp2f(regsFp32P12[int32_t{23}]);
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
              elt0 = regsFp32P12[int32_t{12}];
              // TmemP.h:721
              elt1 = regsFp32P12[int32_t{13}];
              // TmemP.h:722
              elt2 = regsFp32P12[int32_t{14}];
              // TmemP.h:723
              elt3 = regsFp32P12[int32_t{15}];
              // TmemP.h:745
              regsP[int32_t{6}] = trtllm::dev::convert_float2_to_bfloat16(elt0, elt1);
              // TmemP.h:745
              regsP[int32_t{7}] = trtllm::dev::convert_float2_to_bfloat16(elt2, elt3);
            }
            // TmemP.h:1773
            regsFp32P12[int32_t{24}] = exp2f(regsFp32P12[int32_t{24}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{28}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{29}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{28}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{29}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{25}] = exp2f(regsFp32P12[int32_t{25}]);
            // TmemP.h:1773
            regsFp32P12[int32_t{26}] = exp2f(regsFp32P12[int32_t{26}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{30}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{31}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{30}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{31}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{27}] = exp2f(regsFp32P12[int32_t{27}]);
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
              elt0 = regsFp32P12[int32_t{16}];
              // TmemP.h:721
              elt1 = regsFp32P12[int32_t{17}];
              // TmemP.h:722
              elt2 = regsFp32P12[int32_t{18}];
              // TmemP.h:723
              elt3 = regsFp32P12[int32_t{19}];
              // TmemP.h:745
              regsP[int32_t{8}] = trtllm::dev::convert_float2_to_bfloat16(elt0, elt1);
              // TmemP.h:745
              regsP[int32_t{9}] = trtllm::dev::convert_float2_to_bfloat16(elt2, elt3);
            }
            // TmemP.h:1773
            regsFp32P12[int32_t{28}] = exp2f(regsFp32P12[int32_t{28}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{32}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{33}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{32}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{33}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{29}] = exp2f(regsFp32P12[int32_t{29}]);
            // TmemP.h:1773
            regsFp32P12[int32_t{30}] = exp2f(regsFp32P12[int32_t{30}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{34}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{35}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{34}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{35}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{31}] = exp2f(regsFp32P12[int32_t{31}]);
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
              elt0 = regsFp32P12[int32_t{20}];
              // TmemP.h:721
              elt1 = regsFp32P12[int32_t{21}];
              // TmemP.h:722
              elt2 = regsFp32P12[int32_t{22}];
              // TmemP.h:723
              elt3 = regsFp32P12[int32_t{23}];
              // TmemP.h:745
              regsP[int32_t{10}] = trtllm::dev::convert_float2_to_bfloat16(elt0, elt1);
              // TmemP.h:745
              regsP[int32_t{11}] = trtllm::dev::convert_float2_to_bfloat16(elt2, elt3);
            }
            // TmemP.h:1773
            regsFp32P12[int32_t{32}] = exp2f(regsFp32P12[int32_t{32}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{36}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{37}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{36}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{37}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{33}] = exp2f(regsFp32P12[int32_t{33}]);
            // TmemP.h:1773
            regsFp32P12[int32_t{34}] = exp2f(regsFp32P12[int32_t{34}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{38}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{39}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{38}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{39}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{35}] = exp2f(regsFp32P12[int32_t{35}]);
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
              elt0 = regsFp32P12[int32_t{24}];
              // TmemP.h:721
              elt1 = regsFp32P12[int32_t{25}];
              // TmemP.h:722
              elt2 = regsFp32P12[int32_t{26}];
              // TmemP.h:723
              elt3 = regsFp32P12[int32_t{27}];
              // TmemP.h:745
              regsP[int32_t{12}] = trtllm::dev::convert_float2_to_bfloat16(elt0, elt1);
              // TmemP.h:745
              regsP[int32_t{13}] = trtllm::dev::convert_float2_to_bfloat16(elt2, elt3);
            }
            // TmemP.h:1773
            regsFp32P12[int32_t{36}] = exp2f(regsFp32P12[int32_t{36}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{40}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{41}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{40}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{41}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{37}] = exp2f(regsFp32P12[int32_t{37}]);
            // TmemP.h:1773
            regsFp32P12[int32_t{38}] = exp2f(regsFp32P12[int32_t{38}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{42}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{43}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{42}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{43}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{39}] = exp2f(regsFp32P12[int32_t{39}]);
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
              elt0 = regsFp32P12[int32_t{28}];
              // TmemP.h:721
              elt1 = regsFp32P12[int32_t{29}];
              // TmemP.h:722
              elt2 = regsFp32P12[int32_t{30}];
              // TmemP.h:723
              elt3 = regsFp32P12[int32_t{31}];
              // TmemP.h:745
              regsP[int32_t{14}] = trtllm::dev::convert_float2_to_bfloat16(elt0, elt1);
              // TmemP.h:745
              regsP[int32_t{15}] = trtllm::dev::convert_float2_to_bfloat16(elt2, elt3);
            }
            // TmemP.h:1773
            regsFp32P12[int32_t{40}] = exp2f(regsFp32P12[int32_t{40}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{44}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{45}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{44}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{45}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{41}] = exp2f(regsFp32P12[int32_t{41}]);
            // TmemP.h:1773
            regsFp32P12[int32_t{42}] = exp2f(regsFp32P12[int32_t{42}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{46}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{47}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{46}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{47}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{43}] = exp2f(regsFp32P12[int32_t{43}]);
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
              elt0 = regsFp32P12[int32_t{32}];
              // TmemP.h:721
              elt1 = regsFp32P12[int32_t{33}];
              // TmemP.h:722
              elt2 = regsFp32P12[int32_t{34}];
              // TmemP.h:723
              elt3 = regsFp32P12[int32_t{35}];
              // TmemP.h:745
              regsP[int32_t{16}] = trtllm::dev::convert_float2_to_bfloat16(elt0, elt1);
              // TmemP.h:745
              regsP[int32_t{17}] = trtllm::dev::convert_float2_to_bfloat16(elt2, elt3);
            }
            // TmemP.h:1773
            regsFp32P12[int32_t{44}] = exp2f(regsFp32P12[int32_t{44}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{48}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{49}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{48}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{49}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{45}] = exp2f(regsFp32P12[int32_t{45}]);
            // TmemP.h:1773
            regsFp32P12[int32_t{46}] = exp2f(regsFp32P12[int32_t{46}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{50}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{51}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{50}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{51}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{47}] = exp2f(regsFp32P12[int32_t{47}]);
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
              elt0 = regsFp32P12[int32_t{36}];
              // TmemP.h:721
              elt1 = regsFp32P12[int32_t{37}];
              // TmemP.h:722
              elt2 = regsFp32P12[int32_t{38}];
              // TmemP.h:723
              elt3 = regsFp32P12[int32_t{39}];
              // TmemP.h:745
              regsP[int32_t{18}] = trtllm::dev::convert_float2_to_bfloat16(elt0, elt1);
              // TmemP.h:745
              regsP[int32_t{19}] = trtllm::dev::convert_float2_to_bfloat16(elt2, elt3);
            }
            // TmemP.h:1773
            regsFp32P12[int32_t{48}] = exp2f(regsFp32P12[int32_t{48}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{52}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{53}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{52}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{53}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{49}] = exp2f(regsFp32P12[int32_t{49}]);
            // TmemP.h:1773
            regsFp32P12[int32_t{50}] = exp2f(regsFp32P12[int32_t{50}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{54}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{55}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{54}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{55}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{51}] = exp2f(regsFp32P12[int32_t{51}]);
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
              elt0 = regsFp32P12[int32_t{40}];
              // TmemP.h:721
              elt1 = regsFp32P12[int32_t{41}];
              // TmemP.h:722
              elt2 = regsFp32P12[int32_t{42}];
              // TmemP.h:723
              elt3 = regsFp32P12[int32_t{43}];
              // TmemP.h:745
              regsP[int32_t{20}] = trtllm::dev::convert_float2_to_bfloat16(elt0, elt1);
              // TmemP.h:745
              regsP[int32_t{21}] = trtllm::dev::convert_float2_to_bfloat16(elt2, elt3);
            }
            // TmemP.h:1773
            regsFp32P12[int32_t{52}] = exp2f(regsFp32P12[int32_t{52}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{56}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{57}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{56}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{57}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{53}] = exp2f(regsFp32P12[int32_t{53}]);
            // TmemP.h:1773
            regsFp32P12[int32_t{54}] = exp2f(regsFp32P12[int32_t{54}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{58}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{59}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{58}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{59}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{55}] = exp2f(regsFp32P12[int32_t{55}]);
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
              elt0 = regsFp32P12[int32_t{44}];
              // TmemP.h:721
              elt1 = regsFp32P12[int32_t{45}];
              // TmemP.h:722
              elt2 = regsFp32P12[int32_t{46}];
              // TmemP.h:723
              elt3 = regsFp32P12[int32_t{47}];
              // TmemP.h:745
              regsP[int32_t{22}] = trtllm::dev::convert_float2_to_bfloat16(elt0, elt1);
              // TmemP.h:745
              regsP[int32_t{23}] = trtllm::dev::convert_float2_to_bfloat16(elt2, elt3);
            }
            // TmemP.h:1773
            regsFp32P12[int32_t{56}] = exp2f(regsFp32P12[int32_t{56}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{60}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{61}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{60}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{61}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{57}] = exp2f(regsFp32P12[int32_t{57}]);
            // TmemP.h:1773
            regsFp32P12[int32_t{58}] = exp2f(regsFp32P12[int32_t{58}]);
            // TmemP.h:1802
            {
              // TmemP.h:1806
              cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog212, scaleSoftmaxLog212};
              // TmemP.h:764
              cutlass::Array<float, 2> vals;
              // TmemP.h:783
              cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                    negScaledMaxArray[int32_t{0}]};
              // TmemP.h:792
              vals[int32_t{0}] = regsFp32P12[int32_t{62}];
              // TmemP.h:793
              vals[int32_t{1}] = regsFp32P12[int32_t{63}];
              // TmemP.h:826
              vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
              // TmemP.h:833
              regsFp32P12[int32_t{62}] = vals[int32_t{0}];
              // TmemP.h:834
              regsFp32P12[int32_t{63}] = vals[int32_t{1}];
            }
            // TmemP.h:1843
            regsFp32P12[int32_t{59}] = exp2f(regsFp32P12[int32_t{59}]);
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
              elt0 = regsFp32P12[int32_t{48}];
              // TmemP.h:721
              elt1 = regsFp32P12[int32_t{49}];
              // TmemP.h:722
              elt2 = regsFp32P12[int32_t{50}];
              // TmemP.h:723
              elt3 = regsFp32P12[int32_t{51}];
              // TmemP.h:745
              regsP[int32_t{24}] = trtllm::dev::convert_float2_to_bfloat16(elt0, elt1);
              // TmemP.h:745
              regsP[int32_t{25}] = trtllm::dev::convert_float2_to_bfloat16(elt2, elt3);
            }
            // TmemP.h:1773
            regsFp32P12[int32_t{60}] = exp2f(regsFp32P12[int32_t{60}]);
            // TmemP.h:1843
            regsFp32P12[int32_t{61}] = exp2f(regsFp32P12[int32_t{61}]);
            // TmemP.h:1773
            regsFp32P12[int32_t{62}] = exp2f(regsFp32P12[int32_t{62}]);
            // TmemP.h:1843
            regsFp32P12[int32_t{63}] = exp2f(regsFp32P12[int32_t{63}]);
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
              elt0 = regsFp32P12[int32_t{52}];
              // TmemP.h:721
              elt1 = regsFp32P12[int32_t{53}];
              // TmemP.h:722
              elt2 = regsFp32P12[int32_t{54}];
              // TmemP.h:723
              elt3 = regsFp32P12[int32_t{55}];
              // TmemP.h:745
              regsP[int32_t{26}] = trtllm::dev::convert_float2_to_bfloat16(elt0, elt1);
              // TmemP.h:745
              regsP[int32_t{27}] = trtllm::dev::convert_float2_to_bfloat16(elt2, elt3);
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
              elt0 = regsFp32P12[int32_t{56}];
              // TmemP.h:721
              elt1 = regsFp32P12[int32_t{57}];
              // TmemP.h:722
              elt2 = regsFp32P12[int32_t{58}];
              // TmemP.h:723
              elt3 = regsFp32P12[int32_t{59}];
              // TmemP.h:745
              regsP[int32_t{28}] = trtllm::dev::convert_float2_to_bfloat16(elt0, elt1);
              // TmemP.h:745
              regsP[int32_t{29}] = trtllm::dev::convert_float2_to_bfloat16(elt2, elt3);
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
              elt0 = regsFp32P12[int32_t{60}];
              // TmemP.h:721
              elt1 = regsFp32P12[int32_t{61}];
              // TmemP.h:722
              elt2 = regsFp32P12[int32_t{62}];
              // TmemP.h:723
              elt3 = regsFp32P12[int32_t{63}];
              // TmemP.h:745
              regsP[int32_t{30}] = trtllm::dev::convert_float2_to_bfloat16(elt0, elt1);
              // TmemP.h:745
              regsP[int32_t{31}] = trtllm::dev::convert_float2_to_bfloat16(elt2, elt3);
            }
          }
          // TmemTile.cpp:836
          {
            // TmemTile.cpp:838
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:871
            uint32_t const(&srcSlice0)[16]{
              reinterpret_cast<uint32_t const(&)[16]>(regsP[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_st_16x32bx2(
              (tmemBasePtr) +
                (static_cast<uint32_t>((index) * (int32_t{128}) +
                                       (int32_t((tmemP0DstStack.mInstId) == (int32_t{0}))
                                          ? (int32_t{32})
                                          : (int32_t{160})))),
              cuda_ptx::n32_t<32>{},
              srcSlice0);
            // TmemTile.cpp:871
            uint32_t const(&srcSlice1)[16]{
              reinterpret_cast<uint32_t const(&)[16]>(regsP[int32_t{16}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_st_16x32bx2(
              (tmemBasePtr) +
                (static_cast<uint32_t>(
                  ((index) * (int32_t{128}) + (int32_t((tmemP0DstStack.mInstId) == (int32_t{0}))
                                                 ? (int32_t{32})
                                                 : (int32_t{160}))) +
                  (int32_t{0x10 /*hi=0, lo=16*/}))),
              cuda_ptx::n32_t<32>{},
              srcSlice1);
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
          if ((loopOffset1867) >= (int32_t{0})) {
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
        // tmemS0 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{8}, Flags{66560}].
        //
        // Task.cpp:3814
        if (!(isLastLoopIter)) {
          // Task.cpp:2568
          if ((loopOffset1867) >= (int32_t{0})) {
            // Task.cpp:2596
            {
              // Task.cpp:2620
              tmemS0SrcStack.mPipeline.consumer_release(tmemS0ConsReleaseState);
            }
            // Task.cpp:43
            ++tmemS0ConsReleaseState;
          }
        }
        //
        // tmemSoftmaxGlobal0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        // TmemSoftmax.h:545
        float* oldMaxPtr10;
        // TmemSoftmax.h:552
        float* sumPtr10;
        // TmemSoftmax.h:559
        float* newMaxPtr10;
        // TmemSoftmax.h:566
        float* pPtr10;
        // TmemSoftmax.h:573
        bool warpSkipsSums10;
        // Task.cpp:1511
        oldMaxPtr10 = oldMaxPtr8;
        // Task.cpp:1511
        sumPtr10 = sumPtr8;
        // Task.cpp:1511
        newMaxPtr10 = newMaxPtr08;
        // Task.cpp:1511
        pPtr10 = qkPtr08;
        // Task.cpp:1511
        warpSkipsSums10 = warpSkipsSoftmax8;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // TmemSoftmax.h:773
          asm volatile("fence.proxy.async.shared::cta;");
          // TmemSoftmax.h:610
          if (!(warpSkipsSums10)) {
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
            for (int32_t loopOffset2594 = int32_t{0}; loopOffset2594 < int32_t{64};
                 loopOffset2594 += int32_t{8}) {
              // TmemSoftmax.h:917
              cutlass::Array<float, 2> vals0;
              // TmemSoftmax.h:928
              vals0[int32_t{0}] = pPtr10[loopOffset2594];
              // TmemSoftmax.h:929
              vals0[int32_t{1}] = pPtr10[(loopOffset2594) + (int32_t{1})];
              // TmemSoftmax.h:938
              sum0 = trtllm::dev::fadd2(sum0, vals0);
              // TmemSoftmax.h:917
              cutlass::Array<float, 2> vals1;
              // TmemSoftmax.h:928
              vals1[int32_t{0}] = pPtr10[(loopOffset2594) + (int32_t{2})];
              // TmemSoftmax.h:929
              vals1[int32_t{1}] = pPtr10[(loopOffset2594) + (int32_t{3})];
              // TmemSoftmax.h:938
              sum1 = trtllm::dev::fadd2(sum1, vals1);
              // TmemSoftmax.h:917
              cutlass::Array<float, 2> vals2;
              // TmemSoftmax.h:928
              vals2[int32_t{0}] = pPtr10[(loopOffset2594) + (int32_t{4})];
              // TmemSoftmax.h:929
              vals2[int32_t{1}] = pPtr10[(loopOffset2594) + (int32_t{5})];
              // TmemSoftmax.h:938
              sum2 = trtllm::dev::fadd2(sum2, vals2);
              // TmemSoftmax.h:917
              cutlass::Array<float, 2> vals3;
              // TmemSoftmax.h:928
              vals3[int32_t{0}] = pPtr10[(loopOffset2594) + (int32_t{6})];
              // TmemSoftmax.h:929
              vals3[int32_t{1}] = pPtr10[(loopOffset2594) + (int32_t{7})];
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
            float maxDiff{(float{oldMaxPtr10[int32_t{0}]}) - (float{newMaxPtr10[int32_t{0}]})};
            // Common.h:99
            if ((maxDiff) != (float{0})) {
              // Common.h:105
              scale = exp2f((scaleSoftmaxLog210) * (maxDiff));
            }
            // TmemSoftmax.h:815
            float expScale{scale};
            // TmemSoftmax.h:821
            float oldSum{sumPtr10[int32_t{0}]};
            // TmemSoftmax.h:826
            sumPtr10[int32_t{0}] = (expScale) * (oldSum) + (localSum);
          }
        }
        //
        // tmemSoftmaxLocal0 [ProdWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{10}, Flags{0}].
        //
        // Task.cpp:1511
        oldMaxPtr9 = oldMaxPtr8;
        // Task.cpp:1511
        sumPtr9 = sumPtr8;
        // Task.cpp:1511
        newMaxPtr9 = newMaxPtr08;
        // Task.cpp:1607
        if (isLastLoopIter) {
          // Task.cpp:5945
          int32_t index{tmemSoftmaxLocal0ProdState.index()};
          // TmemTile.cpp:373
          cutlass::Array<float, 2> stats;
          // TmemSoftmax.h:365
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset2634 = int32_t{0}; loopOffset2634 < int32_t{1}; ++loopOffset2634) {
            // TmemSoftmax.h:382
            stats[loopOffset2634] = sumPtr9[loopOffset2634];
            // TmemSoftmax.h:384
            stats[(loopOffset2634) + (int32_t{1})] = newMaxPtr9[loopOffset2634];
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
                (static_cast<uint32_t>((index) * (int32_t{128}) +
                                       (int32_t((tmemSoftmaxLocal0DstStack.mInstId) == (int32_t{0}))
                                          ? (int32_t{0})
                                          : (int32_t{128})))),
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
        // tmemSoftmaxLocal0 [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{512}].
        //
        // Task.cpp:1607
        if (isLastLoopIter) {
          // Task.cpp:5078
          if ((loopOffset1867) >= (int32_t{0})) {
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
        // tmemSoftmaxLocal0 [ProdWork (call 2), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{512}].
        //
        //
        // tmemSoftmaxLocal0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{512}].
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
        // tmemS0 [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{16}, Flags{1280}].
        //
        // Task.cpp:1607
        if (isLastLoopIter) {
          // Task.cpp:2596
          {
            // Task.cpp:2620
            tmemS0SrcStack.mPipeline.consumer_release(tmemS0ConsReleaseState);
            // Task.cpp:2627
            asm volatile("fence.proxy.async.shared::cta;");
          }
          // Task.cpp:43
          ++tmemS0ConsReleaseState;
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
  // FmhaTask.h:224
  int32_t mCtaIdxQ;
  // FmhaTask.h:226
  int32_t mCtaIdxKv;
  // FmhaTask.h:214
  int32_t mSeqLenKv;
  // FmhaTask.h:216
  int32_t mNumCtasKv;
  // Task.cpp:706
  uint32_t const mTmemBaseOffset;
  // Task.cpp:371
  int32_t const mWarpGrpThreadIdx;
  // TmemTile.cpp:422
  int32_t const mLdtm16dp32bitTmemColIdx;
  // TmemTile.cpp:445
  int32_t const mLdtm16dp32bitTmemRowIdx;
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
    , // Task.cpp:283
    mCtaIdxQ{mCtaIdxX}
    , // FmhaTask.h:517
    mCtaIdxKv{int32_t{0}}
    , // FmhaTask.h:437
    mSeqLenKv{int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                            ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                            : (mBatchIdx)]}}
    , // Kernel.cpp:212
    mNumCtasKv{int32_t{1}}
    , // Kernel.cpp:2424
    mTmemBaseOffset{uint32_t{
      __shfl_sync(uint32_t{0xffffffff}, (*state.mTmemSwStatePtr), int32_t{0}, int32_t{32})}}
    , // Task.cpp:379
    mWarpGrpThreadIdx{(state.mThreadIdx) - ((warpGrpStart) * (int32_t{32}))}
    , // TmemTile.cpp:432
    mLdtm16dp32bitTmemColIdx{
      trtllm::dev::ldst16dp32bitTmemColIdx<int32_t{64}>((mWarpGrpThreadIdx) % (int32_t{128}))}
    , // TmemTile.cpp:453
    mLdtm16dp32bitTmemRowIdx{
      trtllm::dev::ldst16dp32bitTmemRowIdx<int32_t{16}>((mWarpGrpThreadIdx) % (int32_t{128}))} {}
  // Task.cpp:522
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:547
    return ((state.mWarpIdx) >= (int32_t{4})) && ((state.mWarpIdx) < (int32_t{8}));
  }
  // Task.cpp:454
  inline __device__ void execute(fmha::KernelParams const& params,
                                 KernelState const& state,
                                 TmemCorr0Stack& tmemCorr0DstStack,
                                 TmemSoftmaxLocal0Stack& tmemSoftmaxLocal0SrcStack,
                                 TmemOStack& tmemOSrcStack,
                                 WorkIdStorageSmem& workIdStorageSrcSmem,
                                 WorkIdStorageStack& workIdStorageSrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_inc(cuda_ptx::n32_t<192>{});
    // Task.cpp:2114
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState tmemSoftmaxLocal0ConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState tmemSoftmaxLocal0ConsReleaseState{};
    // Task.cpp:2135
    int32_t tmemSoftmaxLocal0ConsToken{int32_t{0}};
    // Task.cpp:2114
    trtllm::dev::CutlassUmmaAsyncPipeline<
      1,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState tmemOConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassUmmaAsyncPipeline<
      1,
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
      // FmhaTask.h:521
      mCtaIdxQ = currSeqCtaIdx;
      // FmhaTask.h:525
      mCtaIdxKv = int32_t{0};
      // FmhaTask.h:139
      mSeqLenKv = int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                                ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                                : (mBatchIdx)]};
      // FmhaTask.h:582
      int32_t numLoopSteps;
      // FmhaTask.h:592
      int32_t diffKvQ{(mSeqLenKv) - (mSeqLenQ)};
      // FmhaTask.h:597
      int32_t validSeqLenKv;
      // FmhaTask.h:603
      validSeqLenKv = int32_t{
        min((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) + (diffKvQ)) + (params.mNumTokensPerCtaQ),
            mSeqLenKv)};
      // FmhaTask.h:616
      mNumCtasKv =
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, int32_t{1})};
      // FmhaTask.h:630
      if ((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) < (mSeqLenQ)) &&
          ((mCtaIdxKv) < (mNumCtasKv))) {
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
      cutlass::Array<float, 2> frgStats9;
      // TmemCorr.h:1135
      cudaGridDependencySynchronize();
      // TmemCorr.h:1158
      float scaleSoftmaxLog214;
      // TmemCorr.h:1163
      scaleSoftmaxLog214 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
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
      // Loop body.
      //
      // Task.cpp:3392
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset2775 = int32_t{0}; loopOffset2775 < (numLoopSteps) - (int32_t{1});
           ++loopOffset2775) {
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
        float* statsPtr19;
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
            uint32_t(&dstSlice0)[2]{reinterpret_cast<uint32_t(&)[2]>(frgStats9[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_32x32b(
              dstSlice0,
              (tmemBasePtr) +
                (static_cast<uint32_t>((index) * (int32_t{128}) +
                                       (int32_t((tmemSoftmaxLocal0SrcStack.mInstId) == (int32_t{0}))
                                          ? (int32_t{0})
                                          : (int32_t{128})))));
          }
          // TmemSoftmax.h:327
          statsPtr19 = &frgStats9[int32_t{0}];
          // TmemSoftmax.h:330
          cutlass::arch::fence_view_async_tmem_load();
          // Task.cpp:43
          ++tmemSoftmaxLocal0ConsState;
        }
        //
        // tmemSoftmaxLocal0 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
        //
        // Task.cpp:2568
        if ((loopOffset2775) >= (int32_t{0})) {
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
        float* prodStatsPtr014;
        // Task.cpp:1511
        prodStatsPtr014 = statsPtr19;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // TmemCorr.h:289
          cutlass::Array<float, 1> scales14;
          // Common.h:88
          float scale{float{1}};
          // Common.h:92
          float maxDiff{(float{prodStatsPtr014[int32_t{0}]}) -
                        (float{prodStatsPtr014[int32_t{1}]})};
          // Common.h:99
          if ((maxDiff) != (float{0})) {
            // Common.h:105
            scale = exp2f((scaleSoftmaxLog214) * (maxDiff));
          }
          // TmemCorr.h:316
          scales14[int32_t{0}] = scale;
          // TmemCorr.h:1240
          bool skipsCorr{true};
          // TmemCorr.h:1258
          skipsCorr = (skipsCorr) && ((scales14[int32_t{0}]) == (float{1}));
          // TmemCorr.h:1266
          skipsCorr = __all_sync(uint32_t{-1}, skipsCorr);
          // TmemCorr.h:1268
          if (!(skipsCorr)) {
            //
            // The headDimStageIdx: 0.
            //
            // TmemCorr.h:1486
            for (int32_t loopOffset2830 = int32_t{0}; loopOffset2830 < int32_t{64};
                 loopOffset2830 += int32_t{64}) {
              // TmemTile.cpp:373
              cutlass::Array<float, 64> tmemRegs014;
              // TmemTile.cpp:527
              {
                // TmemTile.cpp:529
                uint32_t tmemBasePtr{mTmemBaseOffset};
                // TmemTile.cpp:545
                uint32_t(&dstSlice0)[64]{
                  reinterpret_cast<uint32_t(&)[64]>(tmemRegs014[int32_t{0}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_ld_16x32bx2(
                  dstSlice0,
                  (tmemBasePtr) +
                    (static_cast<uint32_t>((int32_t{0x100 /*hi=0, lo=256*/}) + (loopOffset2830))),
                  cuda_ptx::n32_t<64>{});
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{0}], tmemRegs014[int32_t{1}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{0}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{1}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{2}], tmemRegs014[int32_t{3}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{2}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{3}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{4}], tmemRegs014[int32_t{5}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{4}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{5}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{6}], tmemRegs014[int32_t{7}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{6}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{7}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{8}], tmemRegs014[int32_t{9}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{8}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{9}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{10}], tmemRegs014[int32_t{11}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{10}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{11}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{12}], tmemRegs014[int32_t{13}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{12}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{13}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{14}], tmemRegs014[int32_t{15}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{14}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{15}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{16}], tmemRegs014[int32_t{17}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{16}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{17}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{18}], tmemRegs014[int32_t{19}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{18}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{19}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{20}], tmemRegs014[int32_t{21}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{20}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{21}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{22}], tmemRegs014[int32_t{23}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{22}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{23}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{24}], tmemRegs014[int32_t{25}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{24}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{25}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{26}], tmemRegs014[int32_t{27}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{26}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{27}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{28}], tmemRegs014[int32_t{29}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{28}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{29}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{30}], tmemRegs014[int32_t{31}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{30}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{31}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{32}], tmemRegs014[int32_t{33}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{32}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{33}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{34}], tmemRegs014[int32_t{35}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{34}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{35}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{36}], tmemRegs014[int32_t{37}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{36}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{37}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{38}], tmemRegs014[int32_t{39}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{38}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{39}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{40}], tmemRegs014[int32_t{41}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{40}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{41}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{42}], tmemRegs014[int32_t{43}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{42}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{43}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{44}], tmemRegs014[int32_t{45}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{44}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{45}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{46}], tmemRegs014[int32_t{47}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{46}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{47}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{48}], tmemRegs014[int32_t{49}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{48}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{49}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{50}], tmemRegs014[int32_t{51}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{50}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{51}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{52}], tmemRegs014[int32_t{53}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{52}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{53}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{54}], tmemRegs014[int32_t{55}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{54}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{55}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{56}], tmemRegs014[int32_t{57}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{56}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{57}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{58}], tmemRegs014[int32_t{59}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{58}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{59}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{60}], tmemRegs014[int32_t{61}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{60}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{61}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{62}], tmemRegs014[int32_t{63}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{62}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{63}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1997
              {
                // TmemTile.cpp:836
                {
                  // TmemTile.cpp:838
                  uint32_t tmemBasePtr{mTmemBaseOffset};
                  // TmemTile.cpp:871
                  uint32_t const(&srcSlice0)[64]{
                    reinterpret_cast<uint32_t const(&)[64]>(tmemRegs014[int32_t{0}])};
                  // CudaPtx.h:48
                  cuda_ptx::tcgen05_st_16x32bx2(
                    (tmemBasePtr) +
                      (static_cast<uint32_t>((int32_t{0x100 /*hi=0, lo=256*/}) + (loopOffset2830))),
                    cuda_ptx::n32_t<64>{},
                    srcSlice0);
                }
              }
            }
            //
            // The headDimStageIdx: 1.
            //
            // TmemCorr.h:1486
            for (int32_t loopOffset3035 = int32_t{0}; loopOffset3035 < int32_t{64};
                 loopOffset3035 += int32_t{64}) {
              // TmemTile.cpp:373
              cutlass::Array<float, 64> tmemRegs014;
              // TmemTile.cpp:527
              {
                // TmemTile.cpp:529
                uint32_t tmemBasePtr{mTmemBaseOffset};
                // TmemTile.cpp:545
                uint32_t(&dstSlice0)[64]{
                  reinterpret_cast<uint32_t(&)[64]>(tmemRegs014[int32_t{0}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_ld_16x32bx2(
                  dstSlice0,
                  (tmemBasePtr) +
                    (static_cast<uint32_t>((int32_t{0x180 /*hi=0, lo=384*/}) + (loopOffset3035))),
                  cuda_ptx::n32_t<64>{});
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{0}], tmemRegs014[int32_t{1}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{0}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{1}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{2}], tmemRegs014[int32_t{3}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{2}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{3}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{4}], tmemRegs014[int32_t{5}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{4}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{5}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{6}], tmemRegs014[int32_t{7}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{6}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{7}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{8}], tmemRegs014[int32_t{9}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{8}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{9}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{10}], tmemRegs014[int32_t{11}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{10}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{11}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{12}], tmemRegs014[int32_t{13}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{12}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{13}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{14}], tmemRegs014[int32_t{15}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{14}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{15}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{16}], tmemRegs014[int32_t{17}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{16}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{17}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{18}], tmemRegs014[int32_t{19}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{18}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{19}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{20}], tmemRegs014[int32_t{21}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{20}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{21}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{22}], tmemRegs014[int32_t{23}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{22}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{23}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{24}], tmemRegs014[int32_t{25}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{24}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{25}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{26}], tmemRegs014[int32_t{27}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{26}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{27}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{28}], tmemRegs014[int32_t{29}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{28}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{29}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{30}], tmemRegs014[int32_t{31}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{30}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{31}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{32}], tmemRegs014[int32_t{33}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{32}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{33}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{34}], tmemRegs014[int32_t{35}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{34}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{35}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{36}], tmemRegs014[int32_t{37}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{36}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{37}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{38}], tmemRegs014[int32_t{39}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{38}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{39}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{40}], tmemRegs014[int32_t{41}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{40}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{41}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{42}], tmemRegs014[int32_t{43}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{42}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{43}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{44}], tmemRegs014[int32_t{45}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{44}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{45}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{46}], tmemRegs014[int32_t{47}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{46}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{47}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{48}], tmemRegs014[int32_t{49}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{48}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{49}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{50}], tmemRegs014[int32_t{51}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{50}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{51}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{52}], tmemRegs014[int32_t{53}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{52}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{53}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{54}], tmemRegs014[int32_t{55}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{54}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{55}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{56}], tmemRegs014[int32_t{57}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{56}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{57}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{58}], tmemRegs014[int32_t{59}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{58}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{59}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{60}], tmemRegs014[int32_t{61}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{60}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{61}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{62}], tmemRegs014[int32_t{63}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs014[int32_t{62}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs014[int32_t{63}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1997
              {
                // TmemTile.cpp:836
                {
                  // TmemTile.cpp:838
                  uint32_t tmemBasePtr{mTmemBaseOffset};
                  // TmemTile.cpp:871
                  uint32_t const(&srcSlice0)[64]{
                    reinterpret_cast<uint32_t const(&)[64]>(tmemRegs014[int32_t{0}])};
                  // CudaPtx.h:48
                  cuda_ptx::tcgen05_st_16x32bx2(
                    (tmemBasePtr) +
                      (static_cast<uint32_t>((int32_t{0x180 /*hi=0, lo=384*/}) + (loopOffset3035))),
                    cuda_ptx::n32_t<64>{},
                    srcSlice0);
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
        if ((loopOffset2775) >= (int32_t{0})) {
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
        lastLoopOffset = loopOffset2775;
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
      float* statsPtr29;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemSoftmaxLocal0ConsState.index()};
        // TmemTile.cpp:527
        {
          // TmemTile.cpp:529
          uint32_t tmemBasePtr{mTmemBaseOffset};
          // TmemTile.cpp:545
          uint32_t(&dstSlice0)[2]{reinterpret_cast<uint32_t(&)[2]>(frgStats9[int32_t{0}])};
          // CudaPtx.h:48
          cuda_ptx::tcgen05_ld_32x32b(
            dstSlice0,
            (tmemBasePtr) +
              (static_cast<uint32_t>((index) * (int32_t{128}) +
                                     (int32_t((tmemSoftmaxLocal0SrcStack.mInstId) == (int32_t{0}))
                                        ? (int32_t{0})
                                        : (int32_t{128})))));
        }
        // TmemSoftmax.h:327
        statsPtr29 = &frgStats9[int32_t{0}];
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
      // tmemSoftmaxLocal0 [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{1536}].
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
      // tmemSoftmaxLocal0 [ConsWork (call 3), LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{1536}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemSoftmaxLocal0ConsState.index()};
        // Task.cpp:43
        ++tmemSoftmaxLocal0ConsState;
      }
      //
      // tmemSoftmaxLocal0 [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{1536}].
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
      // tmemO [ConsWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{34}, Flags{0}].
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
      float* prodStatsPtr114;
      // Task.cpp:1511
      prodStatsPtr114 = statsPtr29;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // TmemCorr.h:2860
        int32_t instIdxO{(mCtaIdxQ) * (int32_t{1})};
        // TmemCorr.h:2907
        int32_t numValidRowsO{
          (int32_t{min(params.mNumTokensPerCtaQ,
                       (mSeqLenQ) - ((instIdxO) * (params.mNumTokensPerCtaQ)))}) *
          (params.mNumHeadsQPerKv)};
        // TmemCorr.h:2927
        bool const isInBoundsOut{(mLdtm16dp32bitTmemRowIdx) < (numValidRowsO)};
        // TmemCorr.h:2965
        int32_t threadSeqOffsetO;
        // TmemCorr.h:2966
        int32_t threadHeadOffsetO;
        // TmemCorr.h:2976
        trtllm::dev::mapRowToHeadTokenIdx<true, true>(mLdtm16dp32bitTmemRowIdx,
                                                      threadHeadOffsetO,
                                                      threadSeqOffsetO,
                                                      params.mNumHeadsQPerKv);
        // TmemCorr.h:2984
        int32_t seqOffsetO{(mSeqOffsetQ) + ((instIdxO) * (params.mNumTokensPerCtaQ))};
        // TmemCorr.h:2989
        int32_t headIdxO;
        // TmemCorr.h:2993
        headIdxO =
          ((mHeadIdx) * (int32_t{min(params.mNumHeadsQPerKv, int32_t{64})})) + (threadHeadOffsetO);
        // TmemCorr.h:2996
        int32_t headOffsetO{(((mHeadIdx) * (int32_t{min(params.mNumHeadsQPerKv, int32_t{64})})) +
                             (threadHeadOffsetO)) *
                            (int32_t{256})};
        // TmemCorr.h:3027
        int64_t ctaOffsetO{(static_cast<int64_t>(seqOffsetO)) *
                             (static_cast<int64_t>((params.mNumHeadsQ) * (int32_t{256}))) +
                           (static_cast<int64_t>((headOffsetO) + (mLdtm16dp32bitTmemColIdx)))};
        // TmemCorr.h:3041
        cutlass::bfloat16_t* ptrO{reinterpret_cast<cutlass::bfloat16_t*>(params.ptrO)};
        // TmemCorr.h:3046
        ptrO = ptrO + (ctaOffsetO);
        // TmemCorr.h:3061
        int32_t threadOffsetO{(threadSeqOffsetO) * ((params.mNumHeadsQ) * (int32_t{256}))};
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
                            (((seqOffsetO) * (params.mNumHeadsQ) +
                              ((mHeadIdx) * (int32_t{min(params.mNumHeadsQPerKv, int32_t{64})}))) *
                             (int32_t{2}));
        }
        // TmemCorr.h:289
        cutlass::Array<float, 1> scales14;
        // TmemCorr.h:330
        float finalSum14{prodStatsPtr114[int32_t{0}]};
        // TmemCorr.h:338
        float finalMax14{prodStatsPtr114[int32_t{1}]};
        // TmemCorr.h:354
        finalSum14 =
          (finalSum14) + (float{__shfl_xor_sync(uint32_t{0xffffffff}, finalSum14, int32_t{16})});
        // TmemCorr.h:1859
        float attentionSinkVal{int32_t{0}};
        // TmemCorr.h:1864
        if (bool{params.ptrAttentionSinks != nullptr}) {
          // TmemCorr.h:1884
          attentionSinkVal =
            float{exp2f(((float{params.ptrAttentionSinks[headIdxO]}) * (float{1.442695})) -
                        ((finalMax14) * (scaleSoftmaxLog214)))};
        }
        // TmemCorr.h:384
        prodStatsPtr114[int32_t{0}] = (finalSum14) + (attentionSinkVal);
        // TmemCorr.h:398
        scales14[int32_t{0}] = (float(bool{params.ptrOutputScale == nullptr})
                                  ? (params.mOutputScale)
                                  : (float{params.ptrOutputScale[int32_t{0}]})) /
                               ((finalSum14) + (attentionSinkVal));
        // TmemCorr.h:3898
        if (storesSoftmaxStats) {
          // TmemCorr.h:3902
          float scaleSoftmax{(scaleSoftmaxLog214) * (float{0.69314718})};
          // TmemCorr.h:3915
          (prodStatsPtr114 + int32_t{1})[int32_t{0}] =
            (float{(prodStatsPtr114 + int32_t{1})[int32_t{0}]}) * (scaleSoftmax);
          // TmemCorr.h:3981
          trtllm::dev::storeStatsForAb((prodStatsPtr114 + int32_t{1}),
                                       prodStatsPtr114,
                                       ptrSoftmaxStats,
                                       (threadSeqOffsetO) * (params.mNumHeadsQ) +
                                         (threadHeadOffsetO),
                                       mLdtm16dp32bitTmemRowIdx,
                                       ((mWarpGrpThreadIdx) % (int32_t{32})) < (int32_t{16}),
                                       numValidRowsO);
        }
        //
        // The headDimStageIdx: 0.
        //
        // TmemCorr.h:1486
        for (int32_t loopOffset3371 = int32_t{0}; loopOffset3371 < int32_t{64};
             loopOffset3371 += int32_t{8}) {
          // TmemTile.cpp:373
          cutlass::Array<float, 8> tmemRegs014;
          // TmemTile.cpp:527
          {
            // TmemTile.cpp:529
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:545
            uint32_t(&dstSlice0)[8]{reinterpret_cast<uint32_t(&)[8]>(tmemRegs014[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_16x32bx2(
              dstSlice0,
              (tmemBasePtr) +
                (static_cast<uint32_t>((int32_t{0x100 /*hi=0, lo=256*/}) + (loopOffset3371))),
              cuda_ptx::n32_t<64>{});
          }
          // TmemCorr.h:3438
          uint32_t mRegsO14[4];
          // TmemCorr.h:1534
          {
            // TmemCorr.h:1554
            cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
            // TmemCorr.h:1565
            cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{0}], tmemRegs014[int32_t{1}]};
            // TmemCorr.h:1577
            vals0 = trtllm::dev::fmul2(vals0, localScales0);
            // TmemCorr.h:1580
            tmemRegs014[int32_t{0}] = vals0[int32_t{0}];
            // TmemCorr.h:1581
            tmemRegs014[int32_t{1}] = vals0[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO14[int32_t{0}] = trtllm::dev::convert_float2_to_bfloat16(tmemRegs014[int32_t{0}],
                                                                           tmemRegs014[int32_t{1}]);
          }
          // TmemCorr.h:1534
          {
            // TmemCorr.h:1554
            cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
            // TmemCorr.h:1565
            cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{2}], tmemRegs014[int32_t{3}]};
            // TmemCorr.h:1577
            vals0 = trtllm::dev::fmul2(vals0, localScales0);
            // TmemCorr.h:1580
            tmemRegs014[int32_t{2}] = vals0[int32_t{0}];
            // TmemCorr.h:1581
            tmemRegs014[int32_t{3}] = vals0[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO14[int32_t{1}] = trtllm::dev::convert_float2_to_bfloat16(tmemRegs014[int32_t{2}],
                                                                           tmemRegs014[int32_t{3}]);
          }
          // TmemCorr.h:1534
          {
            // TmemCorr.h:1554
            cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
            // TmemCorr.h:1565
            cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{4}], tmemRegs014[int32_t{5}]};
            // TmemCorr.h:1577
            vals0 = trtllm::dev::fmul2(vals0, localScales0);
            // TmemCorr.h:1580
            tmemRegs014[int32_t{4}] = vals0[int32_t{0}];
            // TmemCorr.h:1581
            tmemRegs014[int32_t{5}] = vals0[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO14[int32_t{2}] = trtllm::dev::convert_float2_to_bfloat16(tmemRegs014[int32_t{4}],
                                                                           tmemRegs014[int32_t{5}]);
          }
          // TmemCorr.h:1534
          {
            // TmemCorr.h:1554
            cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
            // TmemCorr.h:1565
            cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{6}], tmemRegs014[int32_t{7}]};
            // TmemCorr.h:1577
            vals0 = trtllm::dev::fmul2(vals0, localScales0);
            // TmemCorr.h:1580
            tmemRegs014[int32_t{6}] = vals0[int32_t{0}];
            // TmemCorr.h:1581
            tmemRegs014[int32_t{7}] = vals0[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO14[int32_t{3}] = trtllm::dev::convert_float2_to_bfloat16(tmemRegs014[int32_t{6}],
                                                                           tmemRegs014[int32_t{7}]);
          }
          // TmemCorr.h:3744
          if (isInBoundsOut) {
            // Utils.h:647
            trtllm::dev::storeVec((ptrO + loopOffset3371), mRegsO14);
          }
        }
        // TmemCorr.h:4063
        ptrO += int32_t{128};
        //
        // The headDimStageIdx: 1.
        //
        // TmemCorr.h:1486
        for (int32_t loopOffset3413 = int32_t{0}; loopOffset3413 < int32_t{64};
             loopOffset3413 += int32_t{8}) {
          // TmemTile.cpp:373
          cutlass::Array<float, 8> tmemRegs014;
          // TmemTile.cpp:527
          {
            // TmemTile.cpp:529
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:545
            uint32_t(&dstSlice0)[8]{reinterpret_cast<uint32_t(&)[8]>(tmemRegs014[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_16x32bx2(
              dstSlice0,
              (tmemBasePtr) +
                (static_cast<uint32_t>((int32_t{0x180 /*hi=0, lo=384*/}) + (loopOffset3413))),
              cuda_ptx::n32_t<64>{});
          }
          // TmemCorr.h:3438
          uint32_t mRegsO14[4];
          // TmemCorr.h:1534
          {
            // TmemCorr.h:1554
            cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
            // TmemCorr.h:1565
            cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{0}], tmemRegs014[int32_t{1}]};
            // TmemCorr.h:1577
            vals0 = trtllm::dev::fmul2(vals0, localScales0);
            // TmemCorr.h:1580
            tmemRegs014[int32_t{0}] = vals0[int32_t{0}];
            // TmemCorr.h:1581
            tmemRegs014[int32_t{1}] = vals0[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO14[int32_t{0}] = trtllm::dev::convert_float2_to_bfloat16(tmemRegs014[int32_t{0}],
                                                                           tmemRegs014[int32_t{1}]);
          }
          // TmemCorr.h:1534
          {
            // TmemCorr.h:1554
            cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
            // TmemCorr.h:1565
            cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{2}], tmemRegs014[int32_t{3}]};
            // TmemCorr.h:1577
            vals0 = trtllm::dev::fmul2(vals0, localScales0);
            // TmemCorr.h:1580
            tmemRegs014[int32_t{2}] = vals0[int32_t{0}];
            // TmemCorr.h:1581
            tmemRegs014[int32_t{3}] = vals0[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO14[int32_t{1}] = trtllm::dev::convert_float2_to_bfloat16(tmemRegs014[int32_t{2}],
                                                                           tmemRegs014[int32_t{3}]);
          }
          // TmemCorr.h:1534
          {
            // TmemCorr.h:1554
            cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
            // TmemCorr.h:1565
            cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{4}], tmemRegs014[int32_t{5}]};
            // TmemCorr.h:1577
            vals0 = trtllm::dev::fmul2(vals0, localScales0);
            // TmemCorr.h:1580
            tmemRegs014[int32_t{4}] = vals0[int32_t{0}];
            // TmemCorr.h:1581
            tmemRegs014[int32_t{5}] = vals0[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO14[int32_t{2}] = trtllm::dev::convert_float2_to_bfloat16(tmemRegs014[int32_t{4}],
                                                                           tmemRegs014[int32_t{5}]);
          }
          // TmemCorr.h:1534
          {
            // TmemCorr.h:1554
            cutlass::Array<float, 2> localScales0{scales14[int32_t{0}], scales14[int32_t{0}]};
            // TmemCorr.h:1565
            cutlass::Array<float, 2> vals0{tmemRegs014[int32_t{6}], tmemRegs014[int32_t{7}]};
            // TmemCorr.h:1577
            vals0 = trtllm::dev::fmul2(vals0, localScales0);
            // TmemCorr.h:1580
            tmemRegs014[int32_t{6}] = vals0[int32_t{0}];
            // TmemCorr.h:1581
            tmemRegs014[int32_t{7}] = vals0[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO14[int32_t{3}] = trtllm::dev::convert_float2_to_bfloat16(tmemRegs014[int32_t{6}],
                                                                           tmemRegs014[int32_t{7}]);
          }
          // TmemCorr.h:3744
          if (isInBoundsOut) {
            // Utils.h:647
            trtllm::dev::storeVec((ptrO + loopOffset3413), mRegsO14);
          }
        }
        // TmemCorr.h:4063
        ptrO += int32_t{128};
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
  // FmhaTask.h:224
  int32_t mCtaIdxQ;
  // FmhaTask.h:226
  int32_t mCtaIdxKv;
  // FmhaTask.h:214
  int32_t mSeqLenKv;
  // FmhaTask.h:216
  int32_t mNumCtasKv;
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
    , // Task.cpp:283
    mCtaIdxQ{mCtaIdxX}
    , // FmhaTask.h:517
    mCtaIdxKv{int32_t{0}}
    , // FmhaTask.h:437
    mSeqLenKv{int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                            ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                            : (mBatchIdx)]}}
    , // Kernel.cpp:212
    mNumCtasKv{int32_t{1}}
    , // Kernel.cpp:2424
    mTmemBaseOffset{uint32_t{
      __shfl_sync(uint32_t{0xffffffff}, (*state.mTmemSwStatePtr), int32_t{0}, int32_t{32})}} {}
  // Task.cpp:522
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:547
    return ((state.mWarpIdx) >= (int32_t{8})) && ((state.mWarpIdx) < (int32_t{9}));
  }
  // Task.cpp:454
  inline __device__ void execute(fmha::KernelParams const& params,
                                 KernelState const& state,
                                 TmemS0Stack& tmemS0DstStack,
                                 TmemOStack& tmemODstStack,
                                 SmemQSmem& smemQSrcSmem,
                                 SmemQStack& smemQSrcStack,
                                 SmemKvSmem& smemKvSrcSmem,
                                 SmemKvStack& smemKvSrcStack,
                                 TmemP0Stack& tmemP0SrcStack,
                                 WorkIdStorageSmem& workIdStorageSrcSmem,
                                 WorkIdStorageStack& workIdStorageSrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<112>{});
    // Task.cpp:2114
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      1,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemQConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      1,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemQConsReleaseState{};
    // Task.cpp:2135
    int32_t smemQConsToken{int32_t{0}};
    // Task.cpp:2114
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      4,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemKvConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      4,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState
      smemKvConsReleaseState{};
    // Task.cpp:2135
    int32_t smemKvConsToken{int32_t{0}};
    // Task.cpp:2114
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState tmemP0ConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState tmemP0ConsReleaseState{};
    // Task.cpp:2135
    int32_t tmemP0ConsToken{int32_t{0}};
    // Task.cpp:2114
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsReleaseState{};
    // Task.cpp:2135
    int32_t workIdStorageConsToken{int32_t{0}};
    // Task.cpp:2013
    trtllm::dev::CutlassUmmaAsyncPipeline<2,
                                          cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState tmemS0ProdState{int32_t{0}, int32_t{1}, int32_t{0}};
    // Task.cpp:2033
    int32_t tmemS0ProdToken{int32_t{1}};
    // Task.cpp:2013
    trtllm::dev::CutlassUmmaAsyncPipeline<1,
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
    // tmemS0 [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{8256}].
    //
    // Task.cpp:1607
    // Task.cpp:5064
    {
      // Task.cpp:5078
      {
        // Task.cpp:5100
        tmemS0ProdToken = tmemS0DstStack.mPipeline.producer_try_acquire(
          trtllm::dev::makePipelineState(tmemS0ProdState, int32_t{1}));
      }
    }
    // Task.cpp:1607
    // Task.cpp:4288
    {
      // Task.cpp:4318
      tmemS0DstStack.mPipeline.producer_acquire(
        trtllm::dev::makePipelineState(tmemS0ProdState, int32_t{1}),
        tmemS0ProdToken);
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
      // FmhaTask.h:521
      mCtaIdxQ = currSeqCtaIdx;
      // FmhaTask.h:525
      mCtaIdxKv = int32_t{0};
      // FmhaTask.h:139
      mSeqLenKv = int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                                ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                                : (mBatchIdx)]};
      // FmhaTask.h:582
      int32_t numLoopSteps;
      // FmhaTask.h:592
      int32_t diffKvQ{(mSeqLenKv) - (mSeqLenQ)};
      // FmhaTask.h:597
      int32_t validSeqLenKv;
      // FmhaTask.h:603
      validSeqLenKv = int32_t{
        min((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) + (diffKvQ)) + (params.mNumTokensPerCtaQ),
            mSeqLenKv)};
      // FmhaTask.h:616
      mNumCtasKv =
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, int32_t{1})};
      // FmhaTask.h:630
      if ((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) < (mSeqLenQ)) &&
          ((mCtaIdxKv) < (mNumCtasKv))) {
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
      // TmemO.h:261
      bool skipsBmm2;
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
      cutlass::bfloat16_t* smemPtrQ0_2{&smemQSrcSmem.mArray[int32_t{0}][int32_t{0}]};
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
      cutlass::bfloat16_t* smemPtrK4;
      // SmemKv.h:206
      int32_t smemIdxK4;
      // SmemKv.h:214
      cutlass::bfloat16_t* smemPtrV4;
      // SmemKv.h:221
      int32_t smemIdxV4;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{smemKvConsState.index()};
        // SmemKv.h:267
        smemPtrK4 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:304
        smemIdxK4 = index;
        // Task.cpp:43
        ++smemKvConsState;
      }
      //
      // tmemS0 [ProdWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // TmemS.h:1130
      cutlass::bfloat16_t* smemPtrQ8;
      // TmemS.h:1135
      int32_t smemIdxQ8;
      // TmemS.h:1141
      cutlass::bfloat16_t* smemPtrK8;
      // TmemS.h:1146
      int32_t memIdxK8;
      // Task.cpp:1511
      smemPtrQ8 = smemPtrQ0_2;
      // Task.cpp:1511
      smemIdxQ8 = smemIdxQ0_2;
      // Task.cpp:1511
      smemPtrK8 = smemPtrK4;
      // Task.cpp:1511
      memIdxK8 = smemIdxK4;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemS0ProdState.index()};
        // TmemS.h:1889
        cutlass::bfloat16_t* smemQ{smemPtrQ8};
        // TmemS.h:1910
        smemQ += (smemIdxQ8) * (int32_t{16384});
        // TmemS.h:1938
        cutlass::bfloat16_t* smemK{smemPtrK8};
        // TmemS.h:1944
        smemK += (memIdxK8) * (int32_t{16384});
        // Mma.cpp:618
        {
          // TmemTile.cpp:1765
          uint32_t tmemPtrD{
            (index) * (int32_t{128}) +
            (int32_t((tmemS0DstStack.mInstId) == (int32_t{0})) ? (int32_t{0}) : (int32_t{128}))};
          //
          // leadingDimInBytes = 8192, strideInBytes = 1024, swizzleMode = 1.
          //
          // Mma.cpp:203
          uint64_t smemDescA{
            trtllm::dev::createSmemDesc(smemQ,
                                        uint32_t{0x2000000 /*hi=512, lo=0*/},
                                        uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
          //
          // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
          //
          // Mma.cpp:203
          uint64_t smemDescB{
            trtllm::dev::createSmemDesc(smemK,
                                        uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                        uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
          //
          // MMA inst for mi=0 ni=0 ki=0.
          //
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
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
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
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
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
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
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
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
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{506});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{1018});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
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
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
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
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
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
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
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
      // smemKv [ConsWork (call 1), FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{smemKvConsState.index()};
        // SmemKv.h:267
        smemPtrK4 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:304
        smemIdxK4 = index;
        // Task.cpp:43
        ++smemKvConsState;
      }
      //
      // tmemS0 [ProdWork (call 1), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // Task.cpp:1511
      smemPtrQ8 = smemPtrQ0_2;
      // Task.cpp:1511
      smemIdxQ8 = smemIdxQ0_2;
      // Task.cpp:1511
      smemPtrK8 = smemPtrK4;
      // Task.cpp:1511
      memIdxK8 = smemIdxK4;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemS0ProdState.index()};
        // TmemS.h:1889
        cutlass::bfloat16_t* smemQ{smemPtrQ8};
        // TmemS.h:1910
        smemQ += (smemIdxQ8) * (int32_t{16384}) + (int32_t{8192});
        // TmemS.h:1938
        cutlass::bfloat16_t* smemK{smemPtrK8};
        // TmemS.h:1944
        smemK += (memIdxK8) * (int32_t{16384});
        // Mma.cpp:618
        {
          // TmemTile.cpp:1765
          uint32_t tmemPtrD{
            (index) * (int32_t{128}) +
            (int32_t((tmemS0DstStack.mInstId) == (int32_t{0})) ? (int32_t{0}) : (int32_t{128}))};
          //
          // leadingDimInBytes = 8192, strideInBytes = 1024, swizzleMode = 1.
          //
          // Mma.cpp:203
          uint64_t smemDescA{
            trtllm::dev::createSmemDesc(smemQ,
                                        uint32_t{0x2000000 /*hi=512, lo=0*/},
                                        uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
          //
          // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
          //
          // Mma.cpp:203
          uint64_t smemDescB{
            trtllm::dev::createSmemDesc(smemK,
                                        uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                        uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
          //
          // MMA inst for mi=0 ni=0 ki=0.
          //
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
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
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
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
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
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
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
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
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{506});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{1018});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
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
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
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
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
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
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{128},
                                                                  int32_t{16},
                                                                  false,
                                                                  false)};
          // TmemTile.cpp:1710
          if (bool{cute::elect_one_sync()}) {
            // TmemTile.cpp:1718
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
      // tmemS0 [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{8256}].
      //
      //
      // Loop body.
      //
      // Task.cpp:3392
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset3811 = int32_t{0}; loopOffset3811 < (numLoopSteps) - (int32_t{1});
           ++loopOffset3811) {
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
        // smemKv [ConsWork (call 2), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{smemKvConsState.index()};
          // SmemKv.h:267
          smemPtrK4 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
          // SmemKv.h:304
          smemIdxK4 = index;
          // Task.cpp:43
          ++smemKvConsState;
        }
        //
        // tmemS0 [ProdWork (call 2), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        // Task.cpp:1511
        smemPtrQ8 = smemPtrQ0_2;
        // Task.cpp:1511
        smemIdxQ8 = smemIdxQ0_2;
        // Task.cpp:1511
        smemPtrK8 = smemPtrK4;
        // Task.cpp:1511
        memIdxK8 = smemIdxK4;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{tmemS0ProdState.index()};
          // TmemS.h:1889
          cutlass::bfloat16_t* smemQ{smemPtrQ8};
          // TmemS.h:1910
          smemQ += (smemIdxQ8) * (int32_t{16384});
          // TmemS.h:1938
          cutlass::bfloat16_t* smemK{smemPtrK8};
          // TmemS.h:1944
          smemK += (memIdxK8) * (int32_t{16384});
          // Mma.cpp:618
          {
            // TmemTile.cpp:1765
            uint32_t tmemPtrD{
              (index) * (int32_t{128}) +
              (int32_t((tmemS0DstStack.mInstId) == (int32_t{0})) ? (int32_t{0}) : (int32_t{128}))};
            //
            // leadingDimInBytes = 8192, strideInBytes = 1024, swizzleMode = 1.
            //
            // Mma.cpp:203
            uint64_t smemDescA{
              trtllm::dev::createSmemDesc(smemQ,
                                          uint32_t{0x2000000 /*hi=512, lo=0*/},
                                          uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
            //
            // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
            //
            // Mma.cpp:203
            uint64_t smemDescB{
              trtllm::dev::createSmemDesc(smemK,
                                          uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                          uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
            //
            // MMA inst for mi=0 ni=0 ki=0.
            //
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
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
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
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
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
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
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
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
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{506});
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{1018});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
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
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
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
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
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
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
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
        // smemKv [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:2568
        if ((loopOffset3811) >= (int32_t{0})) {
          // Task.cpp:2596
          {
            // Task.cpp:2620
            smemKvSrcStack.mPipeline.consumer_release(smemKvConsReleaseState);
          }
          // Task.cpp:43
          ++smemKvConsReleaseState;
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
          smemPtrK4 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
          // SmemKv.h:304
          smemIdxK4 = index;
          // Task.cpp:43
          ++smemKvConsState;
        }
        //
        // tmemS0 [ProdWork (call 3), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        // Task.cpp:1511
        smemPtrQ8 = smemPtrQ0_2;
        // Task.cpp:1511
        smemIdxQ8 = smemIdxQ0_2;
        // Task.cpp:1511
        smemPtrK8 = smemPtrK4;
        // Task.cpp:1511
        memIdxK8 = smemIdxK4;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{tmemS0ProdState.index()};
          // TmemS.h:1889
          cutlass::bfloat16_t* smemQ{smemPtrQ8};
          // TmemS.h:1910
          smemQ += (smemIdxQ8) * (int32_t{16384}) + (int32_t{8192});
          // TmemS.h:1938
          cutlass::bfloat16_t* smemK{smemPtrK8};
          // TmemS.h:1944
          smemK += (memIdxK8) * (int32_t{16384});
          // Mma.cpp:618
          {
            // TmemTile.cpp:1765
            uint32_t tmemPtrD{
              (index) * (int32_t{128}) +
              (int32_t((tmemS0DstStack.mInstId) == (int32_t{0})) ? (int32_t{0}) : (int32_t{128}))};
            //
            // leadingDimInBytes = 8192, strideInBytes = 1024, swizzleMode = 1.
            //
            // Mma.cpp:203
            uint64_t smemDescA{
              trtllm::dev::createSmemDesc(smemQ,
                                          uint32_t{0x2000000 /*hi=512, lo=0*/},
                                          uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
            //
            // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
            //
            // Mma.cpp:203
            uint64_t smemDescB{
              trtllm::dev::createSmemDesc(smemK,
                                          uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                          uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
            //
            // MMA inst for mi=0 ni=0 ki=0.
            //
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
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
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
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
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
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
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
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
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{506});
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{1018});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
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
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
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
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
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
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
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
        // smemKv [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:2568
        if ((loopOffset3811) >= (int32_t{0})) {
          // Task.cpp:2596
          {
            // Task.cpp:2620
            smemKvSrcStack.mPipeline.consumer_release(smemKvConsReleaseState);
          }
          // Task.cpp:43
          ++smemKvConsReleaseState;
        }
        //
        // tmemS0 [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
        // tmemS0 [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset3811) >= (int32_t{0})) {
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
        // tmemP0 [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        // TmemP.h:488
        int32_t stageIdxP12;
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{tmemP0ConsState.index()};
          // TmemP.h:502
          stageIdxP12 = index;
          // Task.cpp:43
          ++tmemP0ConsState;
        }
        //
        // tmemO [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{3}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset3811) >= (int32_t{0})) {
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
        // smemKv [ConsWork (call 4), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{smemKvConsState.index()};
          // SmemKv.h:322
          smemPtrV4 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
          // SmemKv.h:372
          smemIdxV4 = index;
          // Task.cpp:43
          ++smemKvConsState;
        }
        //
        // tmemO [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{3}, Flags{0}].
        //
        // TmemO.h:277
        cutlass::bfloat16_t* smemPtrV13;
        // TmemO.h:282
        int32_t memIdxV13;
        // TmemO.h:288
        int32_t smemIdxP13;
        // Task.cpp:1511
        smemPtrV13 = smemPtrV4;
        // Task.cpp:1511
        memIdxV13 = smemIdxV4;
        // Task.cpp:1511
        smemIdxP13 = stageIdxP12;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{tmemOProdState.index()};
          // TmemO.h:730
          int32_t* localSmemSkipMmaVotePtr;
          // TmemO.h:748
          int32_t voteVal;
          // TmemO.h:762
          localSmemSkipMmaVotePtr = tmemODstStack.mDepSmemPtr3;
          // TmemO.h:766
          voteVal = localSmemSkipMmaVotePtr[smemIdxP13];
          // TmemO.h:768
          voteVal = uint32_t{__shfl_sync(uint32_t{0xffffffff}, voteVal, int32_t{0}, int32_t{32})};
          // TmemO.h:644
          skipsBmm2 = (voteVal) == (int32_t{1});
          // TmemO.h:653
          if (!(skipsBmm2)) {
            // TmemO.h:493
            cutlass::bfloat16_t* smemV{smemPtrV13};
            // TmemO.h:505
            smemV = smemV + ((memIdxV13) * (int32_t{16384}));
            // TmemO.h:535
            bool readD{true};
            // TmemO.h:545
            if ((loopOffset3811) == (int32_t{0})) {
              // TmemO.h:547
              readD = false;
            }
            // Mma.cpp:618
            {
              // TmemTile.cpp:1765
              uint32_t tmemPtrD{(mTmemBaseOffset) + (uint32_t{256})};
              // TmemTile.cpp:1521
              uint32_t tmemPtrA{
                (mTmemBaseOffset) +
                ((uint32_t{32}) + ((static_cast<uint32_t>(smemIdxP13)) * (uint32_t{128})))};
              //
              // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
              //
              // Mma.cpp:203
              uint64_t smemDescB{
                trtllm::dev::createSmemDesc(smemV,
                                            uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                            uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
              //
              // MMA inst for mi=0 ni=0 ki=0.
              //
              // TmemTile.cpp:1610
              uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      true,
                                                                      int32_t{64},
                                                                      int32_t{128},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              // TmemTile.cpp:1710
              if (bool{cute::elect_one_sync()}) {
                // TmemTile.cpp:1718
                cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
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
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      true,
                                                                      int32_t{64},
                                                                      int32_t{128},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              // TmemTile.cpp:1710
              if (bool{cute::elect_one_sync()}) {
                // TmemTile.cpp:1718
                cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
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
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      true,
                                                                      int32_t{64},
                                                                      int32_t{128},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              // TmemTile.cpp:1710
              if (bool{cute::elect_one_sync()}) {
                // TmemTile.cpp:1718
                cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
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
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      true,
                                                                      int32_t{64},
                                                                      int32_t{128},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              // TmemTile.cpp:1710
              if (bool{cute::elect_one_sync()}) {
                // TmemTile.cpp:1718
                cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                             cuda_ptx::cta_group_1,
                                             tmemPtrD,
                                             tmemPtrA,
                                             smemDescB,
                                             utcmmaDesc_0_0_3,
                                             bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=4.
              //
              // TmemTile.cpp:2041
              tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
              // Mma.cpp:886
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
              // TmemTile.cpp:1610
              uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      true,
                                                                      int32_t{64},
                                                                      int32_t{128},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              // TmemTile.cpp:1710
              if (bool{cute::elect_one_sync()}) {
                // TmemTile.cpp:1718
                cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                             cuda_ptx::cta_group_1,
                                             tmemPtrD,
                                             tmemPtrA,
                                             smemDescB,
                                             utcmmaDesc_0_0_4,
                                             bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=5.
              //
              // TmemTile.cpp:2041
              tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
              // Mma.cpp:886
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
              // TmemTile.cpp:1610
              uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      true,
                                                                      int32_t{64},
                                                                      int32_t{128},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              // TmemTile.cpp:1710
              if (bool{cute::elect_one_sync()}) {
                // TmemTile.cpp:1718
                cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                             cuda_ptx::cta_group_1,
                                             tmemPtrD,
                                             tmemPtrA,
                                             smemDescB,
                                             utcmmaDesc_0_0_5,
                                             bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=6.
              //
              // TmemTile.cpp:2041
              tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
              // Mma.cpp:886
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
              // TmemTile.cpp:1610
              uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      true,
                                                                      int32_t{64},
                                                                      int32_t{128},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              // TmemTile.cpp:1710
              if (bool{cute::elect_one_sync()}) {
                // TmemTile.cpp:1718
                cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                             cuda_ptx::cta_group_1,
                                             tmemPtrD,
                                             tmemPtrA,
                                             smemDescB,
                                             utcmmaDesc_0_0_6,
                                             bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=7.
              //
              // TmemTile.cpp:2041
              tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
              // Mma.cpp:886
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
              // TmemTile.cpp:1610
              uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      true,
                                                                      int32_t{64},
                                                                      int32_t{128},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              // TmemTile.cpp:1710
              if (bool{cute::elect_one_sync()}) {
                // TmemTile.cpp:1718
                cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                             cuda_ptx::cta_group_1,
                                             tmemPtrD,
                                             tmemPtrA,
                                             smemDescB,
                                             utcmmaDesc_0_0_7,
                                             bool{true});
              }
            }
          }
        }
        //
        // smemKv [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
        //
        // Task.cpp:2568
        if ((loopOffset3811) >= (int32_t{0})) {
          // Task.cpp:2596
          {
            // Task.cpp:2620
            smemKvSrcStack.mPipeline.consumer_release(smemKvConsReleaseState);
          }
          // Task.cpp:43
          ++smemKvConsReleaseState;
        }
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
        // smemKv [ConsWork (call 5), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{smemKvConsState.index()};
          // SmemKv.h:322
          smemPtrV4 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
          // SmemKv.h:372
          smemIdxV4 = index;
          // Task.cpp:43
          ++smemKvConsState;
        }
        //
        // tmemO [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{3}, Flags{0}].
        //
        // Task.cpp:1511
        smemPtrV13 = smemPtrV4;
        // Task.cpp:1511
        memIdxV13 = smemIdxV4;
        // Task.cpp:1511
        smemIdxP13 = stageIdxP12;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{tmemOProdState.index()};
          // TmemO.h:653
          if (!(skipsBmm2)) {
            // TmemO.h:493
            cutlass::bfloat16_t* smemV{smemPtrV13};
            // TmemO.h:505
            smemV = smemV + ((memIdxV13) * (int32_t{16384}));
            // TmemO.h:535
            bool readD{true};
            // TmemO.h:545
            if ((loopOffset3811) == (int32_t{0})) {
              // TmemO.h:547
              readD = false;
            }
            // Mma.cpp:618
            {
              // TmemTile.cpp:1765
              uint32_t tmemPtrD{(mTmemBaseOffset) + (uint32_t{384})};
              // TmemTile.cpp:1521
              uint32_t tmemPtrA{
                (mTmemBaseOffset) +
                ((uint32_t{32}) + ((static_cast<uint32_t>(smemIdxP13)) * (uint32_t{128})))};
              //
              // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
              //
              // Mma.cpp:203
              uint64_t smemDescB{
                trtllm::dev::createSmemDesc(smemV,
                                            uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                            uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
              //
              // MMA inst for mi=0 ni=0 ki=0.
              //
              // TmemTile.cpp:1610
              uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      true,
                                                                      int32_t{64},
                                                                      int32_t{128},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              // TmemTile.cpp:1710
              if (bool{cute::elect_one_sync()}) {
                // TmemTile.cpp:1718
                cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
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
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      true,
                                                                      int32_t{64},
                                                                      int32_t{128},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              // TmemTile.cpp:1710
              if (bool{cute::elect_one_sync()}) {
                // TmemTile.cpp:1718
                cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
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
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      true,
                                                                      int32_t{64},
                                                                      int32_t{128},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              // TmemTile.cpp:1710
              if (bool{cute::elect_one_sync()}) {
                // TmemTile.cpp:1718
                cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
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
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      true,
                                                                      int32_t{64},
                                                                      int32_t{128},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              // TmemTile.cpp:1710
              if (bool{cute::elect_one_sync()}) {
                // TmemTile.cpp:1718
                cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                             cuda_ptx::cta_group_1,
                                             tmemPtrD,
                                             tmemPtrA,
                                             smemDescB,
                                             utcmmaDesc_0_0_3,
                                             bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=4.
              //
              // TmemTile.cpp:2041
              tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
              // Mma.cpp:886
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
              // TmemTile.cpp:1610
              uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      true,
                                                                      int32_t{64},
                                                                      int32_t{128},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              // TmemTile.cpp:1710
              if (bool{cute::elect_one_sync()}) {
                // TmemTile.cpp:1718
                cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                             cuda_ptx::cta_group_1,
                                             tmemPtrD,
                                             tmemPtrA,
                                             smemDescB,
                                             utcmmaDesc_0_0_4,
                                             bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=5.
              //
              // TmemTile.cpp:2041
              tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
              // Mma.cpp:886
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
              // TmemTile.cpp:1610
              uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      true,
                                                                      int32_t{64},
                                                                      int32_t{128},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              // TmemTile.cpp:1710
              if (bool{cute::elect_one_sync()}) {
                // TmemTile.cpp:1718
                cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                             cuda_ptx::cta_group_1,
                                             tmemPtrD,
                                             tmemPtrA,
                                             smemDescB,
                                             utcmmaDesc_0_0_5,
                                             bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=6.
              //
              // TmemTile.cpp:2041
              tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
              // Mma.cpp:886
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
              // TmemTile.cpp:1610
              uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      true,
                                                                      int32_t{64},
                                                                      int32_t{128},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              // TmemTile.cpp:1710
              if (bool{cute::elect_one_sync()}) {
                // TmemTile.cpp:1718
                cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                             cuda_ptx::cta_group_1,
                                             tmemPtrD,
                                             tmemPtrA,
                                             smemDescB,
                                             utcmmaDesc_0_0_6,
                                             bool{true});
              }
              //
              // MMA inst for mi=0 ni=0 ki=7.
              //
              // TmemTile.cpp:2041
              tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
              // Mma.cpp:886
              trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
              // TmemTile.cpp:1610
              uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                      int32_t{1},
                                                                      int32_t{1},
                                                                      false,
                                                                      true,
                                                                      int32_t{64},
                                                                      int32_t{128},
                                                                      int32_t{16},
                                                                      false,
                                                                      false)};
              // TmemTile.cpp:1710
              if (bool{cute::elect_one_sync()}) {
                // TmemTile.cpp:1718
                cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                             cuda_ptx::cta_group_1,
                                             tmemPtrD,
                                             tmemPtrA,
                                             smemDescB,
                                             utcmmaDesc_0_0_7,
                                             bool{true});
              }
            }
          }
        }
        //
        // smemKv [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
        //
        // Task.cpp:2568
        if ((loopOffset3811) >= (int32_t{0})) {
          // Task.cpp:2596
          {
            // Task.cpp:2620
            smemKvSrcStack.mPipeline.consumer_release(smemKvConsReleaseState);
          }
          // Task.cpp:43
          ++smemKvConsReleaseState;
        }
        //
        // tmemO [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{3}, Flags{0}].
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
        // Task.cpp:3499
        lastLoopOffset = loopOffset3811;
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
      // smemQ [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
      // tmemS0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{4}].
      //
      //
      // tmemS0 [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{8192}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5078
        if ((lastLoopOffset) >= (int32_t{0})) {
          // Task.cpp:5100
          tmemS0ProdToken = tmemS0DstStack.mPipeline.producer_try_acquire(
            trtllm::dev::makePipelineState(tmemS0ProdState, int32_t{1}));
        }
      }
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:4318
        tmemS0DstStack.mPipeline.producer_acquire(
          trtllm::dev::makePipelineState(tmemS0ProdState, int32_t{1}),
          tmemS0ProdToken);
      }
      //
      // tmemP0 [ConsWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // TmemP.h:488
      int32_t stageIdxP12;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemP0ConsState.index()};
        // TmemP.h:502
        stageIdxP12 = index;
        // Task.cpp:43
        ++tmemP0ConsState;
      }
      //
      // tmemO [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{3}, Flags{0}].
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
      // smemKv [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
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
      // smemKv [ConsWork (call 6), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{smemKvConsState.index()};
        // SmemKv.h:322
        smemPtrV4 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:372
        smemIdxV4 = index;
        // Task.cpp:43
        ++smemKvConsState;
      }
      //
      // tmemO [ProdWork (call 2), LastIter, FreqInfo{0, 1}, UserTags{3}, Flags{0}].
      //
      // TmemO.h:277
      cutlass::bfloat16_t* smemPtrV13;
      // TmemO.h:282
      int32_t memIdxV13;
      // TmemO.h:288
      int32_t smemIdxP13;
      // Task.cpp:1511
      smemPtrV13 = smemPtrV4;
      // Task.cpp:1511
      memIdxV13 = smemIdxV4;
      // Task.cpp:1511
      smemIdxP13 = stageIdxP12;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemOProdState.index()};
        // TmemO.h:730
        int32_t* localSmemSkipMmaVotePtr;
        // TmemO.h:748
        int32_t voteVal;
        // TmemO.h:762
        localSmemSkipMmaVotePtr = tmemODstStack.mDepSmemPtr3;
        // TmemO.h:766
        voteVal = localSmemSkipMmaVotePtr[smemIdxP13];
        // TmemO.h:768
        voteVal = uint32_t{__shfl_sync(uint32_t{0xffffffff}, voteVal, int32_t{0}, int32_t{32})};
        // TmemO.h:644
        skipsBmm2 = (voteVal) == (int32_t{1});
        // TmemO.h:653
        if (!(skipsBmm2)) {
          // TmemO.h:493
          cutlass::bfloat16_t* smemV{smemPtrV13};
          // TmemO.h:505
          smemV = smemV + ((memIdxV13) * (int32_t{16384}));
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
            uint32_t tmemPtrD{(mTmemBaseOffset) + (uint32_t{256})};
            // TmemTile.cpp:1521
            uint32_t tmemPtrA{
              (mTmemBaseOffset) +
              ((uint32_t{32}) + ((static_cast<uint32_t>(smemIdxP13)) * (uint32_t{128})))};
            //
            // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
            //
            // Mma.cpp:203
            uint64_t smemDescB{
              trtllm::dev::createSmemDesc(smemV,
                                          uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                          uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
            //
            // MMA inst for mi=0 ni=0 ki=0.
            //
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    true,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
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
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    true,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
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
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    true,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
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
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    true,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_1,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_3,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=4.
            //
            // TmemTile.cpp:2041
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    true,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_1,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_4,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=5.
            //
            // TmemTile.cpp:2041
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    true,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_1,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_5,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=6.
            //
            // TmemTile.cpp:2041
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    true,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_1,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_6,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=7.
            //
            // TmemTile.cpp:2041
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    true,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_1,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_7,
                                           bool{true});
            }
          }
        }
      }
      //
      // smemKv [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
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
      // smemKv [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
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
      // smemKv [ConsWork (call 7), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{smemKvConsState.index()};
        // SmemKv.h:322
        smemPtrV4 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:372
        smemIdxV4 = index;
        // Task.cpp:43
        ++smemKvConsState;
      }
      //
      // tmemO [ProdWork (call 3), LastIter, FreqInfo{0, 1}, UserTags{3}, Flags{0}].
      //
      // Task.cpp:1511
      smemPtrV13 = smemPtrV4;
      // Task.cpp:1511
      memIdxV13 = smemIdxV4;
      // Task.cpp:1511
      smemIdxP13 = stageIdxP12;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemOProdState.index()};
        // TmemO.h:653
        if (!(skipsBmm2)) {
          // TmemO.h:493
          cutlass::bfloat16_t* smemV{smemPtrV13};
          // TmemO.h:505
          smemV = smemV + ((memIdxV13) * (int32_t{16384}));
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
            uint32_t tmemPtrA{
              (mTmemBaseOffset) +
              ((uint32_t{32}) + ((static_cast<uint32_t>(smemIdxP13)) * (uint32_t{128})))};
            //
            // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
            //
            // Mma.cpp:203
            uint64_t smemDescB{
              trtllm::dev::createSmemDesc(smemV,
                                          uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                          uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
            //
            // MMA inst for mi=0 ni=0 ki=0.
            //
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    true,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
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
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    true,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
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
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    true,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
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
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    true,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_1,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_3,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=4.
            //
            // TmemTile.cpp:2041
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    true,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_1,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_4,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=5.
            //
            // TmemTile.cpp:2041
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_5{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    true,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_1,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_5,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=6.
            //
            // TmemTile.cpp:2041
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_6{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    true,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_1,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_6,
                                           bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=7.
            //
            // TmemTile.cpp:2041
            tmemPtrA += uint32_t{0x8 /*hi=0, lo=8*/};
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{128});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_7{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    true,
                                                                    int32_t{64},
                                                                    int32_t{128},
                                                                    int32_t{16},
                                                                    false,
                                                                    false)};
            // TmemTile.cpp:1710
            if (bool{cute::elect_one_sync()}) {
              // TmemTile.cpp:1718
              cuda_ptx::tcgen05_mma_tmem_a(cuda_ptx::kind_f16,
                                           cuda_ptx::cta_group_1,
                                           tmemPtrD,
                                           tmemPtrA,
                                           smemDescB,
                                           utcmmaDesc_0_0_7,
                                           bool{true});
            }
          }
        }
      }
      //
      // smemKv [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
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
      // tmemO [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{3}, Flags{0}].
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
    // tmemS0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{4}].
    //
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
    // tmemS0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{4}].
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
    // tmemS0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{4}].
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
    return ((state.mWarpIdx) >= (int32_t{10})) && ((state.mWarpIdx) < (int32_t{11}));
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
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<112>{});
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
__launch_bounds__(384, 1) void fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausalP32VarSeqSkipsSoftmaxQ64Kv128PersistentKeepsAbForGen(
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
  uint8_t* smemSkipSoftmaxVoteSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemSkipSoftmaxVoteSmem)});
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
  uint8_t* orderP01SmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(OrderP01SmemBarrier)});
  // Kernel.cpp:1729
  uint8_t* tmemP0SmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
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
                        int32_t{11},
                        int32_t{-1}};
  // Kernel.cpp:2216
  SmemSkipSoftmaxVoteSmem* smemSkipSoftmaxVoteSmem{
    reinterpret_cast<SmemSkipSoftmaxVoteSmem*>(smemSkipSoftmaxVoteSmemPtr)};
  // Kernel.cpp:2283
  SmemSkipSoftmaxVoteStack smemSkipSoftmaxVoteStack{(*smemSkipSoftmaxVoteSmem),
                                                    state.mWarpIdx,
                                                    state.mClusterDimX,
                                                    state.mClusterDimY,
                                                    int32_t{0},
                                                    int32_t{-1}};
  // Kernel.cpp:2216
  SmemKvSmem* smemKvSmem{reinterpret_cast<SmemKvSmem*>(smemKvSmemPtr)};
  // Kernel.cpp:2228
  SmemKvSmemBarrier* smemKvSmemBarrier{reinterpret_cast<SmemKvSmemBarrier*>(smemKvSmemBarrierPtr)};
  // Kernel.cpp:2283
  SmemKvStack smemKvStack{(*smemKvSmem),
                          (*smemKvSmemBarrier),
                          (*smemSkipSoftmaxVoteSmem),
                          smemSkipSoftmaxVoteStack,
                          state.mWarpIdx,
                          state.mClusterDimX,
                          state.mClusterDimY,
                          int32_t{11},
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
                                                int32_t{9},
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
                                        int32_t{10},
                                        int32_t{-1}};
  // Kernel.cpp:2228
  WorkIdThrottleBarrierSmemBarrier* workIdThrottleBarrierSmemBarrier{
    reinterpret_cast<WorkIdThrottleBarrierSmemBarrier*>(workIdThrottleBarrierSmemBarrierPtr)};
  // Kernel.cpp:2283
  WorkIdThrottleBarrierStack workIdThrottleBarrierStack{(*workIdThrottleBarrierSmemBarrier),
                                                        state.mWarpIdx,
                                                        state.mClusterDimX,
                                                        state.mClusterDimY,
                                                        int32_t{10},
                                                        int32_t{-1}};
  // Kernel.cpp:2228
  TmemS0SmemBarrier* tmemS0SmemBarrier{reinterpret_cast<TmemS0SmemBarrier*>(tmemS0SmemBarrierPtr)};
  // Kernel.cpp:2283
  TmemS0Stack tmemS0Stack{(*tmemS0SmemBarrier),
                          (*smemSkipSoftmaxVoteSmem),
                          smemSkipSoftmaxVoteStack,
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
                                                int32_t{5},
                                                int32_t{-1},
                                                int32_t{0}};
  // Kernel.cpp:2283
  TmemSoftmaxGlobal0Stack tmemSoftmaxGlobal0Stack{state.mWarpIdx,
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
                          int32_t{-1},
                          int32_t{-1},
                          int32_t{0}};
  // Kernel.cpp:2228
  TmemOSmemBarrier* tmemOSmemBarrier{reinterpret_cast<TmemOSmemBarrier*>(tmemOSmemBarrierPtr)};
  // Kernel.cpp:2283
  TmemOStack tmemOStack{(*tmemOSmemBarrier),
                        (*smemSkipSoftmaxVoteSmem),
                        smemSkipSoftmaxVoteStack,
                        state.mWarpIdx,
                        state.mClusterDimX,
                        state.mClusterDimY,
                        int32_t{4},
                        int32_t{-1}};
  // Kernel.cpp:2283
  TmemCorr0Stack tmemCorr0Stack{state.mWarpIdx,
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
    LoadPageOffsetsTask loadPageOffsetsTask{params, state, int32_t{9}};
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
      LoadTask loadTask{params, state, int32_t{11}};
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
      if (bool{SoftmaxTask0::isSelected(params, state)}) {
        // Kernel.cpp:2081
        SoftmaxTask0 softmaxTask0{params, state, int32_t{0}};
        // Kernel.cpp:2135
        softmaxTask0.execute(params,
                             state,
                             tmemSoftmaxLocal0Stack,
                             tmemSoftmaxGlobal0Stack,
                             tmemP0Stack,
                             tmemS0Stack,
                             (*workIdStorageSmem),
                             workIdStorageStack);
      } else {
        // Kernel.cpp:2014
        if (bool{CorrTask::isSelected(params, state)}) {
          // Kernel.cpp:2081
          CorrTask corrTask{params, state, int32_t{4}};
          // Kernel.cpp:2135
          corrTask.execute(params,
                           state,
                           tmemCorr0Stack,
                           tmemSoftmaxLocal0Stack,
                           tmemOStack,
                           (*workIdStorageSmem),
                           workIdStorageStack);
          // Task.cpp:5404
          trtllm::dev::CutlassNamedBarrier::sync(128, 10);
          // Task.cpp:5412
          int32_t const warpGrpThreadIdx{(state.mThreadIdx) - (int32_t{128})};
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
            MmaTask mmaTask{params, state, int32_t{8}};
            // Kernel.cpp:2135
            mmaTask.execute(params,
                            state,
                            tmemS0Stack,
                            tmemOStack,
                            (*smemQSmem),
                            smemQStack,
                            (*smemKvSmem),
                            smemKvStack,
                            tmemP0Stack,
                            (*workIdStorageSmem),
                            workIdStorageStack);
          } else {
            // Kernel.cpp:2014
            if (bool{SchedulerTask::isSelected(params, state)}) {
              // Kernel.cpp:2081
              SchedulerTask schedulerTask{params, state, int32_t{10}};
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
fmhaSm100fKernel_QkvBfloat16OBfloat16H256PagedKvCausalP32VarSeqSkipsSoftmaxQ64Kv128PersistentKeepsAbForGenGetSmemSize(
  int32_t* outPtr) {
  int32_t size{0};
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemQSmem));
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemKvSmem));
  size = (size + 127) / 128 * 128;
  size += static_cast<int32_t>(sizeof(SmemPageOffsetsKvSmem));
  size = (size + 15) / 16 * 16;
  size += static_cast<int32_t>(sizeof(SmemSkipSoftmaxVoteSmem));
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
  size += static_cast<int32_t>(sizeof(OrderP01SmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(TmemP0SmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(TmemOSmemBarrier));
  outPtr[0] = size;
}
