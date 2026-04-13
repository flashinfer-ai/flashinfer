#include <FmhaSm100fKernel_QBfloat16KvE4m3OBfloat16H64PagedKvSlidingOrChunkedCausalP32VarSeqQ8Kv128PersistentSwapsAbForGen.h>

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
              int32_t{1024},
              bool{cute::elect_one_sync()},
              CuteFlatTuple239{},
              cute::true_type{},
              cute::true_type{},
              barInitWarpId} {}
};
// Res.cpp:137
// Fmha.h:1117
struct SmemKvStack {
  // Res.cpp:595
  trtllm::dev::CutlassTmaAsyncPipeline<9> mPipeline;
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
              int32_t{8192},
              ((warpId) == (barInitWarpId)) && (bool{cute::elect_one_sync()}),
              int32_t{128},
              CuteFlatTuple355{},
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
  cutlass::bfloat16_t* mPtr;
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
              CuteFlatTuple479{},
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
// Fmha.h:1263
struct SmemPStack {
  // Res.cpp:208
  inline __device__ SmemPStack(SmemPSmem& smemPSmem,
                               int32_t warpId,
                               int32_t clusterDimX,
                               int32_t clusterDimY,
                               int32_t barInitWarpId,
                               int32_t orderedSequenceGroupId) {}
};
// Res.cpp:137
// Fmha.h:1271
struct SmemOStack {
  // Res.cpp:208
  inline __device__ SmemOStack(SmemOSmem& smemOSmem,
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
// Fmha.h:1520
struct WorkIdStorageStack {
  // Res.cpp:595
  trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  // Res.cpp:1023
  cutlass::gemm::kernel::detail::
    PersistentTileSchedulerSm100<cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, 2>
      mScheduler;
  // Res.cpp:1026
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
    : // Res.cpp:805
    mPipeline{workIdStorageSmemBarrier.mBarriers,
              CuteFlatTuple755{},
              int32_t{1},
              int32_t{512},
              int32_t{0},
              barInitWarpId}
    , // Res.cpp:1041
    mScheduler{&workIdStorageSmem.workIdResponse[int32_t{0}],
               typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<
                 cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
                 2>::Params{},
               cute::block_id_in_cluster()}
    , // Res.cpp:1081
    workTileInfo{mScheduler.initial_work_tile_info(CuteFlatTuple951{})} {}
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
// Fmha.h:1866
struct TmemS0Stack {
  // MemBuffers.cpp:488
  float* mDepSmemPtr8;
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
    mDepSmemPtr8{&smemSoftmaxWarpGrpRed0Smem.mArray[int32_t{0}]}
    , // Res.cpp:771
    mPipeline{tmemS0SmemBarrier.mBarriers,
              warpId,
              int32_t{128},
              CuteFlatTuple1181{},
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
                                SmemPSmem& smemPSmem,
                                SmemPStack& smemPStack,
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
    mDepSmemPtr6{&smemPSmem.mArray[int32_t{0}]}
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
                               SmemPSmem& smemPSmem,
                               SmemPStack& smemPStack,
                               int32_t warpId,
                               int32_t clusterDimX,
                               int32_t clusterDimY,
                               int32_t barInitWarpId,
                               int32_t orderedSequenceGroupId)
    : // MemBuffers.cpp:501
    mDepSmemPtr6{&smemPSmem.mArray[int32_t{0}]}
    , // Res.cpp:771
    mPipeline{tmemOSmemBarrier.mBarriers,
              warpId,
              int32_t{128},
              CuteFlatTuple1557{},
              cute::true_type{},
              cute::true_type{},
              barInitWarpId} {}
};
// Res.cpp:137
// Fmha.h:2077
struct TmemCorr0Stack {
  // MemBuffers.cpp:488
  float* mDepSmemPtr10;
  // MemBuffers.cpp:488
  int8_t* mDepSmemPtr7;
  // Res.cpp:208
  inline __device__ TmemCorr0Stack(SmemCorrWarpGrpRed1Smem& smemCorrWarpGrpRed1Smem,
                                   SmemCorrWarpGrpRed1Stack& smemCorrWarpGrpRed1Stack,
                                   SmemOSmem& smemOSmem,
                                   SmemOStack& smemOStack,
                                   int32_t warpId,
                                   int32_t clusterDimX,
                                   int32_t clusterDimY,
                                   int32_t barInitWarpId,
                                   int32_t orderedSequenceGroupId)
    : // MemBuffers.cpp:501
    mDepSmemPtr10{&smemCorrWarpGrpRed1Smem.mArray[int32_t{0}]}
    , // MemBuffers.cpp:501
    mDepSmemPtr7{&smemOSmem.mArray[int32_t{0}]} {}
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
  // FmhaTask.h:733
  int32_t mNumSkippedTilesKv;
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
    , // Task.cpp:283
    mCtaIdxQ{mCtaIdxX}
    , // FmhaTask.h:517
    mCtaIdxKv{int32_t{0}}
    , // FmhaTask.h:543
    mSeqLenKv{(int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                             ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                             : (mBatchIdx)]}) -
              (((mSeqLenQ) - (int32_t{1})) - (mCtaIdxQ))}
    , // Kernel.cpp:212
    mNumCtasKv{int32_t{1}}
    , // Kernel.cpp:210
    mNumSkippedTilesKv{int32_t{0}}
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
                                 SmemPageOffsetsKvStack& smemPageOffsetsKvDstStack,
                                 WorkIdStorageSmem& workIdStorageSrcSmem,
                                 WorkIdStorageStack& workIdStorageSrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<56>{});
    // Task.cpp:2079
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsState{};
    // Task.cpp:2086
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsReleaseState{};
    // Task.cpp:2100
    int32_t workIdStorageConsToken{int32_t{0}};
    // Task.cpp:1979
    trtllm::dev::CutlassCpAsyncPipeline<6>::PipelineState smemPageOffsetsKvProdState{int32_t{0},
                                                                                     int32_t{1},
                                                                                     int32_t{0}};
    // Task.cpp:1999
    int32_t smemPageOffsetsKvProdToken{int32_t{1}};
    // Task.cpp:5369
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
      mSeqLenKv =
        (int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                       ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                       : (mBatchIdx)]}) -
        (((mSeqLenQ) - (int32_t{1})) - (mCtaIdxQ));
      // FmhaTask.h:582
      int32_t numLoopSteps;
      // FmhaTask.h:592
      int32_t diffKvQ{(mSeqLenKv) - (mSeqLenQ)};
      // FmhaTask.h:597
      int32_t validSeqLenKv;
      // Common.h:63
      if ((params.mChunkedAttentionSizeLog2) > (int32_t{0})) {
        // FmhaTask.h:748
        mNumSkippedTilesKv = ((((mCtaIdxQ) + (diffKvQ)) >> (params.mChunkedAttentionSizeLog2))
                              << (params.mChunkedAttentionSizeLog2)) /
                             (int32_t{128});
      } else {
        // FmhaTask.h:767
        mNumSkippedTilesKv = (int32_t{max(int32_t{0},
                                          (((mCtaIdxQ) + (diffKvQ)) + (int32_t{1})) -
                                            (params.mAttentionWindowSize))}) /
                             (int32_t{128});
      }
      // FmhaTask.h:603
      validSeqLenKv = (int32_t{min(((mCtaIdxQ) + (diffKvQ)) + (int32_t{1}), mSeqLenKv)}) -
                      ((mNumSkippedTilesKv) * (int32_t{128}));
      // FmhaTask.h:616
      mNumCtasKv =
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, int32_t{1})};
      // FmhaTask.h:630
      if (((mCtaIdxQ) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
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
      // SmemPageOffsetsKv.h:279
      int32_t pageIdxLb5{
        ((((mCtaIdxQ) * (int32_t{1}) + ((mSeqLenKv) - (mSeqLenQ))) + (int32_t{1})) -
         (params.mAttentionWindowSize)) /
        (int32_t{32})};
      // SmemPageOffsetsKv.h:302
      int32_t pageIdxUb5{(int32_t{((mSeqLenKv) + (int32_t{31})) / (int32_t{32})}) - (int32_t{1})};
      //
      // Loop body.
      //
      // Task.cpp:3350
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset612 = int32_t{0}; loopOffset612 < numLoopSteps;
           loopOffset612 += int32_t{8}) {
        // Task.cpp:3403
        bool const isFirstLoopIter{(loopOffset612) == (int32_t{0})};
        // Task.cpp:3423
        bool const isLastLoopIter{((loopOffset612) + (int32_t{8})) >= (numLoopSteps)};
        //
        // smemPageOffsetsKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:1573
        // Task.cpp:4948
        {
          // Task.cpp:4962
          if ((loopOffset612) >= (int32_t{0})) {
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
          int32_t pageIdx{
            (((mNumSkippedTilesKv) + ((mCtaIdxKv) * (numLoopSteps) + (loopOffset612))) *
             (int32_t{4})) +
            (mWarpGrpThreadIdx)};
          // SmemPageOffsetsKv.h:449
          if (((pageIdx) < (pageIdxLb5)) || ((pageIdx) > (pageIdxUb5))) {
            // SmemPageOffsetsKv.h:451
            (*ptrSmemPageOffsets) = int32_t{-1};
          } else {
            // SmemPageOffsetsKv.h:461
            trtllm::dev::cpAsync(ptrSmemPageOffsets, (ptrPageIdxK5 + pageIdx));
          }
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
          if ((loopOffset612) >= (int32_t{0})) {
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
          int32_t pageIdx{
            (((mNumSkippedTilesKv) + ((mCtaIdxKv) * (numLoopSteps) + (loopOffset612))) *
             (int32_t{4})) +
            (mWarpGrpThreadIdx)};
          // SmemPageOffsetsKv.h:449
          if (((pageIdx) < (pageIdxLb5)) || ((pageIdx) > (pageIdxUb5))) {
            // SmemPageOffsetsKv.h:451
            (*ptrSmemPageOffsets) = int32_t{-1};
          } else {
            // SmemPageOffsetsKv.h:461
            trtllm::dev::cpAsync(ptrSmemPageOffsets, (ptrPageIdxV5 + pageIdx));
          }
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
      // Task.cpp:5416
      auto newWorkTileInfoTuple{
        workIdStorageSrcStack.mScheduler.fetch_next_work(workIdStorageSrcStack.workTileInfo,
                                                         workIdStorageSrcStack.mPipeline,
                                                         workIdStorageConsState)};
      // Task.cpp:5418
      workIdStorageSrcStack.workTileInfo = cute::get<int32_t{0}>(newWorkTileInfoTuple);
      // Task.cpp:5426
      ++workIdStorageConsState;
      // Task.cpp:5528
      mCtaIdxX = workIdStorageSrcStack.workTileInfo.M_idx;
      // Task.cpp:5529
      mCtaIdxY = workIdStorageSrcStack.workTileInfo.N_idx;
      // Task.cpp:5530
      mCtaIdxZ = workIdStorageSrcStack.workTileInfo.L_idx;
    } while (workIdStorageSrcStack.workTileInfo.is_valid_tile);
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
  // FmhaTask.h:733
  int32_t mNumSkippedTilesKv;
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
    , // Task.cpp:283
    mCtaIdxQ{mCtaIdxX}
    , // FmhaTask.h:517
    mCtaIdxKv{int32_t{0}}
    , // FmhaTask.h:543
    mSeqLenKv{(int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                             ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                             : (mBatchIdx)]}) -
              (((mSeqLenQ) - (int32_t{1})) - (mCtaIdxQ))}
    , // Kernel.cpp:212
    mNumCtasKv{int32_t{1}}
    , // Kernel.cpp:210
    mNumSkippedTilesKv{int32_t{0}} {}
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
                                 WorkIdThrottleBarrierStack& workIdThrottleBarrierDstStack,
                                 SmemPageOffsetsKvSmem& smemPageOffsetsKvSrcSmem,
                                 SmemPageOffsetsKvStack& smemPageOffsetsKvSrcStack,
                                 WorkIdStorageSmem& workIdStorageSrcSmem,
                                 WorkIdStorageStack& workIdStorageSrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<56>{});
    // Task.cpp:2079
    trtllm::dev::CutlassCpAsyncPipeline<6>::PipelineState smemPageOffsetsKvConsState{};
    // Task.cpp:2086
    trtllm::dev::CutlassCpAsyncPipeline<6>::PipelineState smemPageOffsetsKvConsReleaseState{};
    // Task.cpp:2100
    int32_t smemPageOffsetsKvConsToken{int32_t{0}};
    // Task.cpp:2079
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsState{};
    // Task.cpp:2086
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsReleaseState{};
    // Task.cpp:2100
    int32_t workIdStorageConsToken{int32_t{0}};
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
    trtllm::dev::CutlassTmaAsyncPipeline<9>::PipelineState smemKvProdState{int32_t{0},
                                                                           int32_t{1},
                                                                           int32_t{0}};
    // Task.cpp:1999
    int32_t smemKvProdToken{int32_t{1}};
    // Task.cpp:1979
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState workIdThrottleBarrierProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    // Task.cpp:1999
    int32_t workIdThrottleBarrierProdToken{int32_t{1}};
    // SmemKv.h:749
    int32_t smemVoteIdx3{int32_t{0}};
    // Task.cpp:5369
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
      mSeqLenKv =
        (int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                       ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                       : (mBatchIdx)]}) -
        (((mSeqLenQ) - (int32_t{1})) - (mCtaIdxQ));
      // FmhaTask.h:582
      int32_t numLoopSteps;
      // FmhaTask.h:592
      int32_t diffKvQ{(mSeqLenKv) - (mSeqLenQ)};
      // FmhaTask.h:597
      int32_t validSeqLenKv;
      // Common.h:63
      if ((params.mChunkedAttentionSizeLog2) > (int32_t{0})) {
        // FmhaTask.h:748
        mNumSkippedTilesKv = ((((mCtaIdxQ) + (diffKvQ)) >> (params.mChunkedAttentionSizeLog2))
                              << (params.mChunkedAttentionSizeLog2)) /
                             (int32_t{128});
      } else {
        // FmhaTask.h:767
        mNumSkippedTilesKv = (int32_t{max(int32_t{0},
                                          (((mCtaIdxQ) + (diffKvQ)) + (int32_t{1})) -
                                            (params.mAttentionWindowSize))}) /
                             (int32_t{128});
      }
      // FmhaTask.h:603
      validSeqLenKv = (int32_t{min(((mCtaIdxQ) + (diffKvQ)) + (int32_t{1}), mSeqLenKv)}) -
                      ((mNumSkippedTilesKv) * (int32_t{128}));
      // FmhaTask.h:616
      mNumCtasKv =
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, int32_t{1})};
      // FmhaTask.h:630
      if (((mCtaIdxQ) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
        // FmhaTask.h:668
        int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
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
      //
      // Hoist the first iter.
      //
      //
      // workIdThrottleBarrier [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{4608}].
      //
      // Task.cpp:1573
      // Task.cpp:4948
      {
        // Task.cpp:4962
        {
          // Task.cpp:4984
          workIdThrottleBarrierProdToken =
            workIdThrottleBarrierDstStack.mPipeline.producer_try_acquire(
              workIdThrottleBarrierProdState);
        }
      }
      // Task.cpp:1573
      // Task.cpp:4180
      {
        // Task.cpp:4210
        workIdThrottleBarrierDstStack.mPipeline.producer_acquire(workIdThrottleBarrierProdState,
                                                                 workIdThrottleBarrierProdToken);
      }
      //
      // workIdThrottleBarrier [ProdWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{4608}].
      //
      //
      // workIdThrottleBarrier [ProdCommit, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{4608}].
      //
      // Task.cpp:1573
      // Task.cpp:4394
      {
        // Task.cpp:4427
        {
          // Task.cpp:4443
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
      // smemPageOffsetsKv [ConsWait, FirstIter, FreqInfo{0, 8}, UserTags{1}, Flags{0}].
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
      // smemPageOffsetsKv [ConsWork (call 0), FirstIter, FreqInfo{0, 8}, UserTags{1}, Flags{0}].
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
        // SmemKv.h:1404
        int32_t headDimOffset{int32_t{0}};
        // SmemKv.h:1529
        int32_t tokenOffset{int32_t{0}};
        //
        // Load pageOffsets for headDimStageIdx = 0.
        //
        // SmemKv.h:1669
        cutlass::AlignedArray<int32_t, 4> localPageOffsets03;
        // SmemKv.h:1685
        localPageOffsets03 = reinterpret_cast<cutlass::AlignedArray<int32_t, 4>*>(
          (ptrSmemPageOffsetsK3 + int32_t{0}))[int32_t{0}];
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
          coords[int32_t{3}] = localPageOffsets03[int32_t{0}];
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
          coords[int32_t{3}] = localPageOffsets03[int32_t{1}];
          // SmemTile.cpp:610
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{2048}],
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
          coords[int32_t{3}] = localPageOffsets03[int32_t{2}];
          // SmemTile.cpp:610
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
          coords[int32_t{3}] = localPageOffsets03[int32_t{3}];
          // SmemTile.cpp:610
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{6144}],
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
      // smemPageOffsetsKv [ConsRelease, FirstIter, FreqInfo{0, 8}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:3772

      //
      // Loop body.
      //
      // Task.cpp:3350
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset925 = int32_t{0}; loopOffset925 < (numLoopSteps) - (int32_t{1});
           ++loopOffset925) {
        // Task.cpp:3423
        bool const isLastLoopIter{((loopOffset925) + (int32_t{1})) >=
                                  ((numLoopSteps) - (int32_t{1}))};
        //
        // gmemKv [ConsWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        // Task.cpp:1573
        // Task.cpp:2893
        {}
        //
        // smemPageOffsetsKv [ConsWait, Info{0}, FreqInfo{1, 8}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:3772
        if ((((loopOffset925) + (int32_t{1})) % (int32_t{8})) == (int32_t{0})) {
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
        // smemPageOffsetsKv [ConsWork (call 1), Info{0}, FreqInfo{1, 8}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:3772
        if ((((loopOffset925) + (int32_t{1})) % (int32_t{8})) == (int32_t{0})) {
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
          if ((loopOffset925) >= (int32_t{0})) {
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
        // smemKv [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{0}].
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
          // SmemKv.h:1404
          int32_t headDimOffset{int32_t{0}};
          // SmemKv.h:1529
          int32_t tokenOffset{int32_t{0}};
          //
          // Load pageOffsets for headDimStageIdx = 0.
          //
          // SmemKv.h:1669
          cutlass::AlignedArray<int32_t, 4> localPageOffsets03;
          // SmemKv.h:1685
          localPageOffsets03 = reinterpret_cast<cutlass::AlignedArray<int32_t, 4>*>(
            (ptrSmemPageOffsetsK3 +
             (((loopOffset925) + (int32_t{1})) * (int32_t{4})) % (int32_t{32})))[int32_t{0}];
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
            coords[int32_t{3}] = localPageOffsets03[int32_t{0}];
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
            coords[int32_t{3}] = localPageOffsets03[int32_t{1}];
            // SmemTile.cpp:610
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{2048}],
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
            coords[int32_t{3}] = localPageOffsets03[int32_t{2}];
            // SmemTile.cpp:610
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
            coords[int32_t{3}] = localPageOffsets03[int32_t{3}];
            // SmemTile.cpp:610
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{6144}],
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
        // smemPageOffsetsKv [ConsRelease, Info{0}, FreqInfo{1, 8}, UserTags{1}, Flags{65536}].
        //
        // Task.cpp:3772
        if ((!(isLastLoopIter)) &&
            ((((loopOffset925) + (int32_t{1})) % (int32_t{8})) == (int32_t{7}))) {
          // Task.cpp:2533
          if ((loopOffset925) >= (int32_t{0})) {
            // Task.cpp:2561
            {
              // Task.cpp:2585
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
        // Task.cpp:3772
        if (((loopOffset925) % (int32_t{8})) == (int32_t{0})) {
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
        // smemPageOffsetsKv [ConsWork (call 2), Info{0}, FreqInfo{0, 8}, UserTags{2}, Flags{0}].
        //
        // Task.cpp:3772
        if (((loopOffset925) % (int32_t{8})) == (int32_t{0})) {
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
          if ((loopOffset925) >= (int32_t{0})) {
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
        // smemKv [ProdWork (call 2), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
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
          // SmemKv.h:1404
          int32_t headDimOffset{int32_t{0}};
          // SmemKv.h:1529
          int32_t tokenOffset{int32_t{0}};
          //
          // Load pageOffsets for headDimStageIdx = 0.
          //
          // SmemKv.h:1669
          cutlass::AlignedArray<int32_t, 4> localPageOffsets03;
          // SmemKv.h:1685
          localPageOffsets03 = reinterpret_cast<cutlass::AlignedArray<int32_t, 4>*>(
            (ptrSmemPageOffsetsV3 + ((loopOffset925) * (int32_t{4})) % (int32_t{32})))[int32_t{0}];
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
            coords[int32_t{3}] = localPageOffsets03[int32_t{0}];
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
            coords[int32_t{3}] = localPageOffsets03[int32_t{1}];
            // SmemTile.cpp:610
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{2048}],
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
            coords[int32_t{3}] = localPageOffsets03[int32_t{2}];
            // SmemTile.cpp:610
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
            coords[int32_t{3}] = localPageOffsets03[int32_t{3}];
            // SmemTile.cpp:610
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemKvDstSmem.mArray[index][int32_t{6144}],
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
        // smemPageOffsetsKv [ConsRelease, Info{0}, FreqInfo{0, 8}, UserTags{2}, Flags{0}].
        //
        // Task.cpp:3772
        if (((loopOffset925) % (int32_t{8})) == (int32_t{7})) {
          // Task.cpp:2533
          if ((loopOffset925) >= (int32_t{0})) {
            // Task.cpp:2561
            {
              // Task.cpp:2585
              smemPageOffsetsKvSrcStack.mPipeline.consumer_release(
                smemPageOffsetsKvConsReleaseState);
            }
            // Task.cpp:43
            ++smemPageOffsetsKvConsReleaseState;
          }
        }
        // Task.cpp:3457
        lastLoopOffset = loopOffset925;
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
      // smemPageOffsetsKv [ConsRelease, LastIter, FreqInfo{0, 8}, UserTags{1}, Flags{0}].
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
      // smemPageOffsetsKv [ConsWait, LastIter, FreqInfo{0, 8}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:3772
      if (((lastLoopOffset) % (int32_t{8})) == (int32_t{0})) {
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
      // smemPageOffsetsKv [ConsWork (call 3), LastIter, FreqInfo{0, 8}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:3772
      if (((lastLoopOffset) % (int32_t{8})) == (int32_t{0})) {
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
      // smemKv [ProdWork (call 3), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
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
        // SmemKv.h:1404
        int32_t headDimOffset{int32_t{0}};
        // SmemKv.h:1529
        int32_t tokenOffset{int32_t{0}};
        //
        // Load pageOffsets for headDimStageIdx = 0.
        //
        // SmemKv.h:1669
        cutlass::AlignedArray<int32_t, 4> localPageOffsets03;
        // SmemKv.h:1685
        localPageOffsets03 = reinterpret_cast<cutlass::AlignedArray<int32_t, 4>*>(
          (ptrSmemPageOffsetsV3 + ((lastLoopOffset) * (int32_t{4})) % (int32_t{32})))[int32_t{0}];
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
          coords[int32_t{3}] = localPageOffsets03[int32_t{0}];
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
          coords[int32_t{3}] = localPageOffsets03[int32_t{1}];
          // SmemTile.cpp:610
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{2048}],
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
          coords[int32_t{3}] = localPageOffsets03[int32_t{2}];
          // SmemTile.cpp:610
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
          coords[int32_t{3}] = localPageOffsets03[int32_t{3}];
          // SmemTile.cpp:610
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKvDstSmem.mArray[index][int32_t{6144}],
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
      // smemPageOffsetsKv [ConsRelease, LastIter, FreqInfo{0, 8}, UserTags{2}, Flags{0}].
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
      // Task.cpp:5416
      auto newWorkTileInfoTuple{
        workIdStorageSrcStack.mScheduler.fetch_next_work(workIdStorageSrcStack.workTileInfo,
                                                         workIdStorageSrcStack.mPipeline,
                                                         workIdStorageConsState)};
      // Task.cpp:5418
      workIdStorageSrcStack.workTileInfo = cute::get<int32_t{0}>(newWorkTileInfoTuple);
      // Task.cpp:5426
      ++workIdStorageConsState;
      // Task.cpp:5528
      mCtaIdxX = workIdStorageSrcStack.workTileInfo.M_idx;
      // Task.cpp:5529
      mCtaIdxY = workIdStorageSrcStack.workTileInfo.N_idx;
      // Task.cpp:5530
      mCtaIdxZ = workIdStorageSrcStack.workTileInfo.L_idx;
    } while (workIdStorageSrcStack.workTileInfo.is_valid_tile);
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
  // FmhaTask.h:733
  int32_t mNumSkippedTilesKv;
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
    , // Task.cpp:283
    mCtaIdxQ{mCtaIdxX}
    , // FmhaTask.h:517
    mCtaIdxKv{int32_t{0}}
    , // FmhaTask.h:543
    mSeqLenKv{(int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                             ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                             : (mBatchIdx)]}) -
              (((mSeqLenQ) - (int32_t{1})) - (mCtaIdxQ))}
    , // Kernel.cpp:212
    mNumCtasKv{int32_t{1}}
    , // Kernel.cpp:210
    mNumSkippedTilesKv{int32_t{0}}
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
                                 TmemS0Stack& tmemS0SrcStack,
                                 WorkIdStorageSmem& workIdStorageSrcSmem,
                                 WorkIdStorageStack& workIdStorageSrcStack) {
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
    // Task.cpp:2079
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsState{};
    // Task.cpp:2086
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsReleaseState{};
    // Task.cpp:2100
    int32_t workIdStorageConsToken{int32_t{0}};
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
    // Task.cpp:5369
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
      mSeqLenKv =
        (int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                       ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                       : (mBatchIdx)]}) -
        (((mSeqLenQ) - (int32_t{1})) - (mCtaIdxQ));
      // FmhaTask.h:582
      int32_t numLoopSteps;
      // FmhaTask.h:592
      int32_t diffKvQ{(mSeqLenKv) - (mSeqLenQ)};
      // FmhaTask.h:597
      int32_t validSeqLenKv;
      // Common.h:63
      if ((params.mChunkedAttentionSizeLog2) > (int32_t{0})) {
        // FmhaTask.h:748
        mNumSkippedTilesKv = ((((mCtaIdxQ) + (diffKvQ)) >> (params.mChunkedAttentionSizeLog2))
                              << (params.mChunkedAttentionSizeLog2)) /
                             (int32_t{128});
      } else {
        // FmhaTask.h:767
        mNumSkippedTilesKv = (int32_t{max(int32_t{0},
                                          (((mCtaIdxQ) + (diffKvQ)) + (int32_t{1})) -
                                            (params.mAttentionWindowSize))}) /
                             (int32_t{128});
      }
      // FmhaTask.h:603
      validSeqLenKv = (int32_t{min(((mCtaIdxQ) + (diffKvQ)) + (int32_t{1}), mSeqLenKv)}) -
                      ((mNumSkippedTilesKv) * (int32_t{128}));
      // FmhaTask.h:616
      mNumCtasKv =
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, int32_t{1})};
      // FmhaTask.h:630
      if (((mCtaIdxQ) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
        // FmhaTask.h:668
        int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
                         ((mNumCtasKv) * (int32_t{128}))};
        // FmhaTask.h:682
        numLoopSteps = numSteps;
      } else {
        // FmhaTask.h:648
        numLoopSteps = int32_t{0};
      }
      // Task.cpp:3168
      bool const hasOneLoopIter{(int32_t{0}) < (numLoopSteps)};
      // TmemS.h:647
      float oldMaxArray13[2];
      // TmemS.h:653
      float sumArray13[2]{float{0}, float{0}};
      // TmemS.h:665
      float newMaxArray13[2]{float{-3.4028235e+38}, float{-3.4028235e+38}};
      // TmemTile.cpp:373
      cutlass::Array<float, 8> regsQk;
      // TmemS.h:1354
      uint32_t uint32NegFltMax13{trtllm::dev::floatToUInt32ForAtomicMax(float{-3.4028235e+38})};
      // TmemS.h:1367
      CUTLASS_PRAGMA_UNROLL
      for (int32_t loopOffset1330 = mWarpGrpThreadIdx; loopOffset1330 < int32_t{8};
           loopOffset1330 += int32_t{128}) {
        // TmemS.h:1374
        reinterpret_cast<uint32_t*>(tmemS0SrcStack.mDepSmemPtr8)[loopOffset1330] =
          uint32NegFltMax13;
      }
      // TmemS.h:717
      trtllm::dev::CutlassNamedBarrier::sync(128, tmemS0SrcStack.mNamedBarId);
      // TmemSoftmax.h:515
      cudaGridDependencySynchronize();
      // TmemSoftmax.h:524
      float scaleSoftmaxLog215;
      // TmemSoftmax.h:529
      scaleSoftmaxLog215 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
                             ? (params.mScaleSoftmaxLog2)
                             : (float{params.ptrScaleSoftmaxLog2[int32_t{0}]});
      // TmemP.h:507
      uint32_t regsP[4];
      // TmemP.h:520
      cudaGridDependencySynchronize();
      // TmemP.h:527
      float scaleSoftmaxLog217;
      // TmemP.h:532
      scaleSoftmaxLog217 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
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
      for (int32_t loopOffset1356 = int32_t{0}; loopOffset1356 < numLoopSteps; ++loopOffset1356) {
        // Task.cpp:3423
        bool const isLastLoopIter{((loopOffset1356) + (int32_t{1})) >= (numLoopSteps)};
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
        float* oldMaxPtr13;
        // TmemS.h:1058
        float* sumPtr13;
        // TmemS.h:1063
        float* newMaxPtr013;
        // TmemS.h:1068
        float* qkPtr013;
        // TmemS.h:1073
        float* newMaxPtr113;
        // TmemS.h:1078
        float* qkPtr113;
        // Task.cpp:1573
        // Task.cpp:2893
        {
          // Task.cpp:5829
          int32_t index{tmemS0ConsState.index()};
          // TmemS.h:1185
          oldMaxPtr13 = oldMaxArray13;
          // TmemS.h:1187
          sumPtr13 = sumArray13;
          // TmemS.h:1189
          newMaxPtr013 = newMaxArray13;
          // TmemS.h:1191
          newMaxPtr113 = newMaxArray13;
          // TmemS.h:1239
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset1377 = int32_t{0}; loopOffset1377 < int32_t{2}; ++loopOffset1377) {
            // TmemS.h:1250
            oldMaxArray13[loopOffset1377] = newMaxArray13[loopOffset1377];
          }
          //
          // The dense mask block.
          //
          // Mask.h:1760
          bool const allTilesAreCompleteK{((mSeqLenKv) % (int32_t{128})) == (int32_t{0})};
          // Mask.h:568
          int32_t const tileOffsetK{
            ((((numLoopSteps) * (mCtaIdxKv) + (loopOffset1356)) * (int32_t{1}) + (int32_t{1})) +
             (mNumSkippedTilesKv)) *
            (int32_t{128})};
          // Mask.h:1783
          bool const isFullTileK{(tileOffsetK) < (mSeqLenKv)};
          // Mask.h:1824
          if (((allTilesAreCompleteK) || (isFullTileK)) &&
              (((((mSeqLenKv) - (params.mAttentionWindowSize)) % (int32_t{128})) == (int32_t{0})) ||
               (((numLoopSteps) * (mCtaIdxKv) + (loopOffset1356)) > (int32_t{0})))) {
            // TmemTile.cpp:527
            {
              // TmemTile.cpp:529
              uint32_t tmemBasePtr{mTmemBaseOffset};
              // TmemTile.cpp:545
              uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(regsQk[int32_t{0}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_16x256b(
                dstSlice0,
                (tmemBasePtr) +
                  (static_cast<uint32_t>((index) * (int32_t{8}) +
                                         (int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                            ? (int32_t{0})
                                            : (int32_t{8})))));
              // TmemTile.cpp:545
              uint32_t(&dstSlice1)[4]{reinterpret_cast<uint32_t(&)[4]>(regsQk[int32_t{4}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_16x256b(
                dstSlice1,
                (tmemBasePtr) +
                  (static_cast<uint32_t>(
                    ((index) * (int32_t{8}) + (int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                                 ? (int32_t{0})
                                                 : (int32_t{8}))) +
                    (int32_t{0x100000 /*hi=16, lo=0*/}))));
            }
            // Utils.h:248
            trtllm::dev::reduceColMax16dp256bit<int32_t{2}, int32_t{1}, int32_t{1}, false>(
              newMaxArray13,
              regsQk);
            // Utils.h:260
            trtllm::dev::reduceColMax(newMaxArray13,
                                      tmemS0SrcStack.mDepSmemPtr8,
                                      int32_t{128},
                                      mWarpGrpThreadIdx,
                                      tmemS0SrcStack.mNamedBarId);
          } else {
            // TmemTile.cpp:527
            {
              // TmemTile.cpp:529
              uint32_t tmemBasePtr{mTmemBaseOffset};
              // TmemTile.cpp:545
              uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(regsQk[int32_t{0}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_16x256b(
                dstSlice0,
                (tmemBasePtr) +
                  (static_cast<uint32_t>((index) * (int32_t{8}) +
                                         (int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                            ? (int32_t{0})
                                            : (int32_t{8})))));
              // TmemTile.cpp:545
              uint32_t(&dstSlice1)[4]{reinterpret_cast<uint32_t(&)[4]>(regsQk[int32_t{4}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_16x256b(
                dstSlice1,
                (tmemBasePtr) +
                  (static_cast<uint32_t>(
                    ((index) * (int32_t{8}) + (int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                                 ? (int32_t{0})
                                                 : (int32_t{8}))) +
                    (int32_t{0x100000 /*hi=16, lo=0*/}))));
            }
            //
            // Apply the dense mask.
            //
            // Mask.h:568
            int32_t const tileOffsetK{
              ((((numLoopSteps) * (mCtaIdxKv) + (loopOffset1356)) * (int32_t{1})) +
               (mNumSkippedTilesKv)) *
              (int32_t{128})};
            // Mask.h:862
            int32_t localIdxK{mLdtm16dp256bitTmemRowIdx};
            // Mask.h:407
            int32_t clampedDistLeft{((tileOffsetK) + (localIdxK)) -
                                    ((mSeqLenKv) - (params.mAttentionWindowSize))};
            // Mask.h:409
            int32_t clampedDistRight{((tileOffsetK) + (localIdxK)) - (mSeqLenKv)};
            // Mask.h:872
            asm volatile("{\n"
                         ".reg .pred p<4>;\n"
                         "setp.lt.s32 p0, %9, 0;\n"
                         "@p0 mov.b32 %0, %8;\n"
                         "@p0 mov.b32 %1, %8;\n"
                         "setp.lt.s32 p1, %9, -8;\n"
                         "@p1 mov.b32 %2, %8;\n"
                         "@p1 mov.b32 %3, %8;\n"
                         "setp.lt.s32 p2, %9, -16;\n"
                         "@p2 mov.b32 %4, %8;\n"
                         "@p2 mov.b32 %5, %8;\n"
                         "setp.lt.s32 p3, %9, -24;\n"
                         "@p3 mov.b32 %6, %8;\n"
                         "@p3 mov.b32 %7, %8;\n"
                         "setp.ge.s32 p0, %10, 0;\n"
                         "@p0 mov.b32 %0, %8;\n"
                         "@p0 mov.b32 %1, %8;\n"
                         "setp.ge.s32 p1, %10, -8;\n"
                         "@p1 mov.b32 %2, %8;\n"
                         "@p1 mov.b32 %3, %8;\n"
                         "setp.ge.s32 p2, %10, -16;\n"
                         "@p2 mov.b32 %4, %8;\n"
                         "@p2 mov.b32 %5, %8;\n"
                         "setp.ge.s32 p3, %10, -24;\n"
                         "@p3 mov.b32 %6, %8;\n"
                         "@p3 mov.b32 %7, %8;\n"
                         "}\n"
                         : "+f"(regsQk[0]),
                           "+f"(regsQk[1]),
                           "+f"(regsQk[2]),
                           "+f"(regsQk[3]),
                           "+f"(regsQk[4]),
                           "+f"(regsQk[5]),
                           "+f"(regsQk[6]),
                           "+f"(regsQk[7])
                         : "f"(float{-3.4028235e+38}), "r"(clampedDistLeft), "r"(clampedDistRight));
            // Utils.h:248
            trtllm::dev::reduceColMax16dp256bit<int32_t{2}, int32_t{1}, int32_t{1}, false>(
              newMaxArray13,
              regsQk);
            // Utils.h:260
            trtllm::dev::reduceColMax(newMaxArray13,
                                      tmemS0SrcStack.mDepSmemPtr8,
                                      int32_t{128},
                                      mWarpGrpThreadIdx,
                                      tmemS0SrcStack.mNamedBarId);
          }
          // TmemS.h:1327
          qkPtr013 = &regsQk[int32_t{0}];
          // TmemS.h:1329
          qkPtr113 = &regsQk[int32_t{0}];
          // Task.cpp:43
          ++tmemS0ConsState;
        }
        //
        // tmemSoftmaxLocal0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{8}, Flags{0}].
        //
        // TmemSoftmax.h:261
        float* oldMaxPtr14;
        // TmemSoftmax.h:267
        float* sumPtr14;
        // TmemSoftmax.h:273
        float* newMaxPtr14;
        // Task.cpp:1477
        oldMaxPtr14 = oldMaxPtr13;
        // Task.cpp:1477
        sumPtr14 = sumPtr13;
        // Task.cpp:1477
        newMaxPtr14 = newMaxPtr013;
        // Task.cpp:1573
        // Task.cpp:5038
        {
          // Task.cpp:5829
          int32_t index{tmemSoftmaxLocal0ProdState.index()};
          // TmemTile.cpp:373
          cutlass::Array<float, 4> stats;
          // TmemSoftmax.h:365
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset1422 = int32_t{0}; loopOffset1422 < int32_t{2}; ++loopOffset1422) {
            // TmemSoftmax.h:382
            stats[loopOffset1422] = oldMaxPtr14[loopOffset1422];
            // TmemSoftmax.h:384
            stats[(loopOffset1422) + (int32_t{2})] = newMaxPtr14[loopOffset1422];
          }
          // TmemTile.cpp:824
          {
            // TmemTile.cpp:826
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:859
            uint32_t const(&srcSlice0)[4]{
              reinterpret_cast<uint32_t const(&)[4]>(stats[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_st_32x32b(
              (tmemBasePtr) +
                (static_cast<uint32_t>((index) * (int32_t{32}) +
                                       (int32_t((tmemSoftmaxLocal0DstStack.mInstId) == (int32_t{0}))
                                          ? (int32_t{16})
                                          : (int32_t{48})))),
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
        float* newMaxPtr17;
        // TmemP.h:560
        float* regsFp32P17;
        // Task.cpp:1477
        newMaxPtr17 = newMaxPtr113;
        // Task.cpp:1477
        regsFp32P17 = qkPtr113;
        // Task.cpp:1573
        // Task.cpp:5038
        {
          // Task.cpp:5829
          int32_t index{tmemP0ProdState.index()};
          // TmemP.h:1011
          float negScaledMaxArray[2];
          // TmemP.h:1029
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset1450 = int32_t{0}; loopOffset1450 < int32_t{2};
               loopOffset1450 += int32_t{2}) {
            // TmemP.h:1040
            float newMax0{newMaxPtr17[loopOffset1450]};
            // TmemP.h:1046
            float newMax1{newMaxPtr17[(loopOffset1450) + (int32_t{1})]};
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
            float negLog2Scale{-(scaleSoftmaxLog217)};
            // Common.h:353
            cutlass::Array<float, 2> negLog2Scale2{negLog2Scale, negLog2Scale};
            // TmemP.h:1090
            newMax2 = trtllm::dev::fmul2(newMax2, negLog2Scale2);
            // TmemP.h:1101
            negScaledMaxArray[loopOffset1450] = newMax2[int32_t{0}];
            // TmemP.h:1102
            negScaledMaxArray[(loopOffset1450) + (int32_t{1})] = newMax2[int32_t{1}];
          }
          // TmemP.h:1597
          {
            // TmemP.h:1600
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog217, scaleSoftmaxLog217};
            // TmemP.h:750
            cutlass::Array<float, 2> vals;
            // TmemP.h:769
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{1}]};
            // TmemP.h:778
            vals[int32_t{0}] = regsFp32P17[int32_t{0}];
            // TmemP.h:779
            vals[int32_t{1}] = regsFp32P17[int32_t{1}];
            // TmemP.h:787
            vals[int32_t{0}] =
              (log2Scale2[int32_t{0}]) * (vals[int32_t{0}]) + (negScaledMax[int32_t{0}]);
            // TmemP.h:796
            vals[int32_t{1}] =
              (log2Scale2[int32_t{1}]) * (vals[int32_t{1}]) + (negScaledMax[int32_t{1}]);
            // TmemP.h:819
            regsFp32P17[int32_t{0}] = vals[int32_t{0}];
            // TmemP.h:820
            regsFp32P17[int32_t{1}] = vals[int32_t{1}];
          }
          // TmemP.h:1597
          {
            // TmemP.h:1600
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog217, scaleSoftmaxLog217};
            // TmemP.h:750
            cutlass::Array<float, 2> vals;
            // TmemP.h:769
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{1}]};
            // TmemP.h:778
            vals[int32_t{0}] = regsFp32P17[int32_t{2}];
            // TmemP.h:779
            vals[int32_t{1}] = regsFp32P17[int32_t{3}];
            // TmemP.h:812
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:819
            regsFp32P17[int32_t{2}] = vals[int32_t{0}];
            // TmemP.h:820
            regsFp32P17[int32_t{3}] = vals[int32_t{1}];
          }
          // TmemP.h:1716
          regsFp32P17[int32_t{0}] = exp2f(regsFp32P17[int32_t{0}]);
          // TmemP.h:1745
          {
            // TmemP.h:1749
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog217, scaleSoftmaxLog217};
            // TmemP.h:750
            cutlass::Array<float, 2> vals;
            // TmemP.h:769
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{1}]};
            // TmemP.h:778
            vals[int32_t{0}] = regsFp32P17[int32_t{4}];
            // TmemP.h:779
            vals[int32_t{1}] = regsFp32P17[int32_t{5}];
            // TmemP.h:812
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:819
            regsFp32P17[int32_t{4}] = vals[int32_t{0}];
            // TmemP.h:820
            regsFp32P17[int32_t{5}] = vals[int32_t{1}];
          }
          // TmemP.h:1786
          regsFp32P17[int32_t{1}] = exp2f(regsFp32P17[int32_t{1}]);
          // TmemP.h:1716
          regsFp32P17[int32_t{2}] = exp2f(regsFp32P17[int32_t{2}]);
          // TmemP.h:1745
          {
            // TmemP.h:1749
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog217, scaleSoftmaxLog217};
            // TmemP.h:750
            cutlass::Array<float, 2> vals;
            // TmemP.h:769
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{1}]};
            // TmemP.h:778
            vals[int32_t{0}] = regsFp32P17[int32_t{6}];
            // TmemP.h:779
            vals[int32_t{1}] = regsFp32P17[int32_t{7}];
            // TmemP.h:812
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:819
            regsFp32P17[int32_t{6}] = vals[int32_t{0}];
            // TmemP.h:820
            regsFp32P17[int32_t{7}] = vals[int32_t{1}];
          }
          // TmemP.h:1786
          regsFp32P17[int32_t{3}] = exp2f(regsFp32P17[int32_t{3}]);
          // TmemP.h:1716
          regsFp32P17[int32_t{4}] = exp2f(regsFp32P17[int32_t{4}]);
          // TmemP.h:1786
          regsFp32P17[int32_t{5}] = exp2f(regsFp32P17[int32_t{5}]);
          // TmemP.h:1716
          regsFp32P17[int32_t{6}] = exp2f(regsFp32P17[int32_t{6}]);
          // TmemP.h:1786
          regsFp32P17[int32_t{7}] = exp2f(regsFp32P17[int32_t{7}]);
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
            elt0 = regsFp32P17[int32_t{0}];
            // TmemP.h:707
            elt1 = regsFp32P17[int32_t{1}];
            // TmemP.h:708
            elt2 = regsFp32P17[int32_t{2}];
            // TmemP.h:709
            elt3 = regsFp32P17[int32_t{3}];
            // TmemP.h:731
            regsP[int32_t{0}] = trtllm::dev::convert_float2_to_bfloat16(elt0, elt1);
            // TmemP.h:731
            regsP[int32_t{1}] = trtllm::dev::convert_float2_to_bfloat16(elt2, elt3);
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
            elt0 = regsFp32P17[int32_t{4}];
            // TmemP.h:707
            elt1 = regsFp32P17[int32_t{5}];
            // TmemP.h:708
            elt2 = regsFp32P17[int32_t{6}];
            // TmemP.h:709
            elt3 = regsFp32P17[int32_t{7}];
            // TmemP.h:731
            regsP[int32_t{2}] = trtllm::dev::convert_float2_to_bfloat16(elt0, elt1);
            // TmemP.h:731
            regsP[int32_t{3}] = trtllm::dev::convert_float2_to_bfloat16(elt2, elt3);
          }
          // TmemP.h:1206
          cutlass::bfloat16_t* smemPtrP17;
          // TmemP.h:1208
          smemPtrP17 = reinterpret_cast<cutlass::bfloat16_t*>(tmemP0DstStack.mDepSmemPtr6) +
                       (((tmemP0DstStack.mInstId) + (index)) * (int32_t{1024}));
          // TmemP.h:1229
          trtllm::dev::storeTransposedSmem128x16b<int32_t{8}>(smemPtrP17, regsP, mWarpGrpThreadIdx);
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
          if ((loopOffset1356) >= (int32_t{0})) {
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
        if ((loopOffset1356) >= (int32_t{0})) {
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
        float* oldMaxPtr15;
        // TmemSoftmax.h:552
        float* sumPtr15;
        // TmemSoftmax.h:559
        float* newMaxPtr15;
        // TmemSoftmax.h:566
        float* pPtr15;
        // Task.cpp:1477
        oldMaxPtr15 = oldMaxPtr13;
        // Task.cpp:1477
        sumPtr15 = sumPtr13;
        // Task.cpp:1477
        newMaxPtr15 = newMaxPtr013;
        // Task.cpp:1477
        pPtr15 = qkPtr013;
        // Task.cpp:1573
        // Task.cpp:5038
        {
          // TmemSoftmax.h:1010
          {
            // Common.h:395
            cutlass::Array<float, 2> oldMax{float{oldMaxPtr15[int32_t{0}]},
                                            float{oldMaxPtr15[int32_t{1}]}};
            // Common.h:395
            cutlass::Array<float, 2> newMax{float{newMaxPtr15[int32_t{0}]},
                                            float{newMaxPtr15[int32_t{1}]}};
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
              cutlass::Array<float, 2> scale2_{scaleSoftmaxLog215, scaleSoftmaxLog215};
              // Common.h:161
              scale2 = trtllm::dev::fmul2(scale2_, maxDiff2);
              // Common.h:168
              scale2[int32_t{0}] = exp2f(scale2[int32_t{0}]);
              // Common.h:169
              scale2[int32_t{1}] = exp2f(scale2[int32_t{1}]);
            }
            // TmemSoftmax.h:1029
            cutlass::Array<float, 2> p0{pPtr15[int32_t{0}], pPtr15[int32_t{1}]};
            // Common.h:395
            cutlass::Array<float, 2> sum{float{sumPtr15[int32_t{0}]}, float{sumPtr15[int32_t{1}]}};
            // TmemSoftmax.h:1048
            sum = trtllm::dev::ffma2(scale2, sum, p0);
            // TmemSoftmax.h:1060
            cutlass::Array<float, 2> p1{pPtr15[int32_t{2}], pPtr15[int32_t{3}]};
            // TmemSoftmax.h:1076
            sum = trtllm::dev::fadd2(sum, p1);
            // TmemSoftmax.h:1083
            sumPtr15[int32_t{0}] = sum[int32_t{0}];
            // TmemSoftmax.h:1084
            sumPtr15[int32_t{1}] = sum[int32_t{1}];
            // TmemSoftmax.h:1060
            cutlass::Array<float, 2> p2{pPtr15[int32_t{4}], pPtr15[int32_t{5}]};
            // TmemSoftmax.h:1076
            sum = trtllm::dev::fadd2(sum, p2);
            // TmemSoftmax.h:1083
            sumPtr15[int32_t{0}] = sum[int32_t{0}];
            // TmemSoftmax.h:1084
            sumPtr15[int32_t{1}] = sum[int32_t{1}];
            // TmemSoftmax.h:1060
            cutlass::Array<float, 2> p3{pPtr15[int32_t{6}], pPtr15[int32_t{7}]};
            // TmemSoftmax.h:1076
            sum = trtllm::dev::fadd2(sum, p3);
            // TmemSoftmax.h:1083
            sumPtr15[int32_t{0}] = sum[int32_t{0}];
            // TmemSoftmax.h:1084
            sumPtr15[int32_t{1}] = sum[int32_t{1}];
          }
        }
        //
        // tmemSoftmaxLocal0 [ProdWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{10}, Flags{0}].
        //
        // Task.cpp:1477
        oldMaxPtr14 = oldMaxPtr13;
        // Task.cpp:1477
        sumPtr14 = sumPtr13;
        // Task.cpp:1477
        newMaxPtr14 = newMaxPtr013;
        // Task.cpp:1573
        if (isLastLoopIter) {
          // Task.cpp:5829
          int32_t index{tmemSoftmaxLocal0ProdState.index()};
          // TmemTile.cpp:373
          cutlass::Array<float, 4> stats;
          // TmemSoftmax.h:365
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset1613 = int32_t{0}; loopOffset1613 < int32_t{2}; ++loopOffset1613) {
            // TmemSoftmax.h:382
            stats[loopOffset1613] = sumPtr14[loopOffset1613];
            // TmemSoftmax.h:384
            stats[(loopOffset1613) + (int32_t{2})] = newMaxPtr14[loopOffset1613];
          }
          // TmemTile.cpp:824
          {
            // TmemTile.cpp:826
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:859
            uint32_t const(&srcSlice0)[4]{
              reinterpret_cast<uint32_t const(&)[4]>(stats[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_st_32x32b(
              (tmemBasePtr) +
                (static_cast<uint32_t>((index) * (int32_t{32}) +
                                       (int32_t((tmemSoftmaxLocal0DstStack.mInstId) == (int32_t{0}))
                                          ? (int32_t{16})
                                          : (int32_t{48})))),
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
          if ((loopOffset1356) >= (int32_t{0})) {
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
      // Task.cpp:5416
      auto newWorkTileInfoTuple{
        workIdStorageSrcStack.mScheduler.fetch_next_work(workIdStorageSrcStack.workTileInfo,
                                                         workIdStorageSrcStack.mPipeline,
                                                         workIdStorageConsState)};
      // Task.cpp:5418
      workIdStorageSrcStack.workTileInfo = cute::get<int32_t{0}>(newWorkTileInfoTuple);
      // Task.cpp:5426
      ++workIdStorageConsState;
      // Task.cpp:5528
      mCtaIdxX = workIdStorageSrcStack.workTileInfo.M_idx;
      // Task.cpp:5529
      mCtaIdxY = workIdStorageSrcStack.workTileInfo.N_idx;
      // Task.cpp:5530
      mCtaIdxZ = workIdStorageSrcStack.workTileInfo.L_idx;
    } while (workIdStorageSrcStack.workTileInfo.is_valid_tile);
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
  // FmhaTask.h:733
  int32_t mNumSkippedTilesKv;
  // Task.cpp:691
  uint32_t const mTmemBaseOffset;
  // Task.cpp:371
  int32_t const mWarpGrpThreadIdx;
  // TmemTile.cpp:422
  int32_t const mLdtm16dp256bitTmemColIdx;
  // TmemTile.cpp:445
  int32_t const mLdtm16dp256bitTmemRowIdx;
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
    , // Task.cpp:283
    mCtaIdxQ{mCtaIdxX}
    , // FmhaTask.h:517
    mCtaIdxKv{int32_t{0}}
    , // FmhaTask.h:543
    mSeqLenKv{(int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                             ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                             : (mBatchIdx)]}) -
              (((mSeqLenQ) - (int32_t{1})) - (mCtaIdxQ))}
    , // Kernel.cpp:212
    mNumCtasKv{int32_t{1}}
    , // Kernel.cpp:210
    mNumSkippedTilesKv{int32_t{0}}
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
      trtllm::dev::ldst16dp256bitTmemRowIdx<int32_t{16}>((mWarpGrpThreadIdx) % (int32_t{128}))} {}
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
                                 TmemOStack& tmemOSrcStack,
                                 WorkIdStorageSmem& workIdStorageSrcSmem,
                                 WorkIdStorageStack& workIdStorageSrcStack) {
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
    // Task.cpp:2079
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsState{};
    // Task.cpp:2086
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsReleaseState{};
    // Task.cpp:2100
    int32_t workIdStorageConsToken{int32_t{0}};
    // Task.cpp:5369
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
      mSeqLenKv =
        (int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                       ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                       : (mBatchIdx)]}) -
        (((mSeqLenQ) - (int32_t{1})) - (mCtaIdxQ));
      // FmhaTask.h:582
      int32_t numLoopSteps;
      // FmhaTask.h:592
      int32_t diffKvQ{(mSeqLenKv) - (mSeqLenQ)};
      // FmhaTask.h:597
      int32_t validSeqLenKv;
      // Common.h:63
      if ((params.mChunkedAttentionSizeLog2) > (int32_t{0})) {
        // FmhaTask.h:748
        mNumSkippedTilesKv = ((((mCtaIdxQ) + (diffKvQ)) >> (params.mChunkedAttentionSizeLog2))
                              << (params.mChunkedAttentionSizeLog2)) /
                             (int32_t{128});
      } else {
        // FmhaTask.h:767
        mNumSkippedTilesKv = (int32_t{max(int32_t{0},
                                          (((mCtaIdxQ) + (diffKvQ)) + (int32_t{1})) -
                                            (params.mAttentionWindowSize))}) /
                             (int32_t{128});
      }
      // FmhaTask.h:603
      validSeqLenKv = (int32_t{min(((mCtaIdxQ) + (diffKvQ)) + (int32_t{1}), mSeqLenKv)}) -
                      ((mNumSkippedTilesKv) * (int32_t{128}));
      // FmhaTask.h:616
      mNumCtasKv =
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, int32_t{1})};
      // FmhaTask.h:630
      if (((mCtaIdxQ) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
        // FmhaTask.h:668
        int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
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
      cutlass::Array<float, 4> frgStats14;
      // TmemCorr.h:1126
      cudaGridDependencySynchronize();
      // TmemCorr.h:1149
      float scaleSoftmaxLog219;
      // TmemCorr.h:1154
      scaleSoftmaxLog219 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
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
      for (int32_t loopOffset1748 = int32_t{0}; loopOffset1748 < (numLoopSteps) - (int32_t{1});
           ++loopOffset1748) {
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
        float* statsPtr114;
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
            uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(frgStats14[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_32x32b(
              dstSlice0,
              (tmemBasePtr) +
                (static_cast<uint32_t>((index) * (int32_t{32}) +
                                       (int32_t((tmemSoftmaxLocal0SrcStack.mInstId) == (int32_t{0}))
                                          ? (int32_t{16})
                                          : (int32_t{48})))));
          }
          // TmemSoftmax.h:327
          statsPtr114 = &frgStats14[int32_t{0}];
          // TmemSoftmax.h:330
          cutlass::arch::fence_view_async_tmem_load();
          // Task.cpp:43
          ++tmemSoftmaxLocal0ConsState;
        }
        //
        // tmemSoftmaxLocal0 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
        //
        // Task.cpp:2533
        if ((loopOffset1748) >= (int32_t{0})) {
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
        float* prodStatsPtr019;
        // Task.cpp:1477
        prodStatsPtr019 = statsPtr114;
        // Task.cpp:1573
        // Task.cpp:5038
        {
          // TmemCorr.h:416
          cutlass::Array<float, 2> scales19;
          // TmemCorr.h:428
          {
            // Common.h:353
            cutlass::Array<float, 2> oldMax{float{prodStatsPtr019[int32_t{0}]},
                                            float{prodStatsPtr019[int32_t{1}]}};
            // Common.h:353
            cutlass::Array<float, 2> newMax{float{prodStatsPtr019[int32_t{2}]},
                                            float{prodStatsPtr019[int32_t{3}]}};
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
              cutlass::Array<float, 2> scale2_{scaleSoftmaxLog219, scaleSoftmaxLog219};
              // Common.h:161
              scale2 = trtllm::dev::fmul2(scale2_, maxDiff2);
              // Common.h:168
              scale2[int32_t{0}] = exp2f(scale2[int32_t{0}]);
              // Common.h:169
              scale2[int32_t{1}] = exp2f(scale2[int32_t{1}]);
            }
            // TmemCorr.h:464
            scales19[int32_t{0}] = scale2[int32_t{0}];
            // TmemCorr.h:465
            scales19[int32_t{1}] = scale2[int32_t{1}];
          }
          // TmemCorr.h:1231
          bool skipsCorr{true};
          // TmemCorr.h:1249
          skipsCorr = (skipsCorr) && ((scales19[int32_t{0}]) == (float{1}));
          // TmemCorr.h:1249
          skipsCorr = (skipsCorr) && ((scales19[int32_t{1}]) == (float{1}));
          // TmemCorr.h:1257
          skipsCorr = __all_sync(uint32_t{-1}, skipsCorr);
          // TmemCorr.h:1259
          if (!(skipsCorr)) {
            //
            // The headDimStageIdx: 0.
            //
            // TmemCorr.h:1472
            for (int32_t loopOffset1813 = int32_t{0}; loopOffset1813 < int32_t{4};
                 loopOffset1813 += int32_t{8}) {
              // TmemTile.cpp:373
              cutlass::Array<float, 4> tmemRegs019;
              // TmemTile.cpp:527
              {
                // TmemTile.cpp:529
                uint32_t tmemBasePtr{mTmemBaseOffset};
                // TmemTile.cpp:545
                uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(tmemRegs019[int32_t{0}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_ld_16x256b(
                  dstSlice0,
                  (tmemBasePtr) +
                    (static_cast<uint32_t>((int32_t{0x50 /*hi=0, lo=80*/}) + (loopOffset1813))));
              }
              // TmemCorr.h:1520
              {
                // TmemCorr.h:1540
                cutlass::Array<float, 2> localScales0{scales19[int32_t{0}], scales19[int32_t{1}]};
                // TmemCorr.h:1551
                cutlass::Array<float, 2> vals0{tmemRegs019[int32_t{0}], tmemRegs019[int32_t{1}]};
                // TmemCorr.h:1563
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1566
                tmemRegs019[int32_t{0}] = vals0[int32_t{0}];
                // TmemCorr.h:1567
                tmemRegs019[int32_t{1}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1520
              {
                // TmemCorr.h:1540
                cutlass::Array<float, 2> localScales0{scales19[int32_t{0}], scales19[int32_t{1}]};
                // TmemCorr.h:1551
                cutlass::Array<float, 2> vals0{tmemRegs019[int32_t{2}], tmemRegs019[int32_t{3}]};
                // TmemCorr.h:1563
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1566
                tmemRegs019[int32_t{2}] = vals0[int32_t{0}];
                // TmemCorr.h:1567
                tmemRegs019[int32_t{3}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1976
              {
                // TmemTile.cpp:824
                {
                  // TmemTile.cpp:826
                  uint32_t tmemBasePtr{mTmemBaseOffset};
                  // TmemTile.cpp:859
                  uint32_t const(&srcSlice0)[4]{
                    reinterpret_cast<uint32_t const(&)[4]>(tmemRegs019[int32_t{0}])};
                  // CudaPtx.h:48
                  cuda_ptx::tcgen05_st_16x256b(
                    (tmemBasePtr) +
                      (static_cast<uint32_t>((int32_t{0x50 /*hi=0, lo=80*/}) + (loopOffset1813))),
                    srcSlice0);
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
        if ((loopOffset1748) >= (int32_t{0})) {
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
        lastLoopOffset = loopOffset1748;
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
      float* statsPtr214;
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:5829
        int32_t index{tmemSoftmaxLocal0ConsState.index()};
        // TmemTile.cpp:527
        {
          // TmemTile.cpp:529
          uint32_t tmemBasePtr{mTmemBaseOffset};
          // TmemTile.cpp:545
          uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(frgStats14[int32_t{0}])};
          // CudaPtx.h:48
          cuda_ptx::tcgen05_ld_32x32b(
            dstSlice0,
            (tmemBasePtr) +
              (static_cast<uint32_t>((index) * (int32_t{32}) +
                                     (int32_t((tmemSoftmaxLocal0SrcStack.mInstId) == (int32_t{0}))
                                        ? (int32_t{16})
                                        : (int32_t{48})))));
        }
        // TmemSoftmax.h:327
        statsPtr214 = &frgStats14[int32_t{0}];
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
      float* prodStatsPtr119;
      // Task.cpp:1477
      prodStatsPtr119 = statsPtr214;
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // TmemCorr.h:2829
        int32_t instIdxO{(mCtaIdxQ) * (int32_t{1})};
        // TmemCorr.h:2953
        int32_t seqOffsetO{(mSeqOffsetQ) + (instIdxO)};
        // TmemCorr.h:2958
        int32_t headIdxO;
        // TmemCorr.h:2962
        headIdxO = (mHeadIdx) * (int32_t{min(params.mNumHeadsQPerKv, int32_t{8})});
        // TmemCorr.h:2965
        int32_t headOffsetO{((mHeadIdx) * (int32_t{min(params.mNumHeadsQPerKv, int32_t{8})})) *
                            (int32_t{64})};
        // TmemCorr.h:2996
        int64_t ctaOffsetO{(static_cast<int64_t>(seqOffsetO)) *
                             (static_cast<int64_t>((params.mNumHeadsQ) * (int32_t{64}))) +
                           (static_cast<int64_t>(headOffsetO))};
        // TmemCorr.h:3010
        cutlass::bfloat16_t* ptrO{reinterpret_cast<cutlass::bfloat16_t*>(params.ptrO)};
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
                              ((mHeadIdx) * (int32_t{min(params.mNumHeadsQPerKv, int32_t{8})}))) *
                             (int32_t{2}));
        }
        // TmemCorr.h:416
        cutlass::Array<float, 2> scales19;
        // TmemCorr.h:1962
        trtllm::dev::reduceColSum<int32_t{2}>(prodStatsPtr119,
                                              tmemCorr0DstStack.mDepSmemPtr10,
                                              int32_t{4},
                                              int32_t{128},
                                              mWarpGrpThreadIdx,
                                              int32_t{4});
        // TmemCorr.h:500
        {
          // TmemCorr.h:509
          float sum0{prodStatsPtr119[int32_t{0}]};
          // TmemCorr.h:515
          float sum1{prodStatsPtr119[int32_t{1}]};
          // TmemCorr.h:1883
          float attentionSinkVal0{int32_t{0}};
          // TmemCorr.h:1886
          float attentionSinkVal1{int32_t{0}};
          // TmemCorr.h:1891
          if (bool{params.ptrAttentionSinks != nullptr}) {
            // TmemCorr.h:1916
            attentionSinkVal0 = float{exp2f(
              ((float{params.ptrAttentionSinks[int32_t{min((headIdxO) + (mLdtm16dp256bitTmemColIdx),
                                                           (params.mNumHeadsQ) - (int32_t{1}))}]}) *
               (float{1.442695})) -
              ((float{prodStatsPtr119[int32_t{2}]}) * (scaleSoftmaxLog219)))};
            // TmemCorr.h:1917
            attentionSinkVal1 =
              float{exp2f(((float{params.ptrAttentionSinks[int32_t{
                             min((headIdxO) + ((mLdtm16dp256bitTmemColIdx) + (int32_t{1})),
                                 (params.mNumHeadsQ) - (int32_t{1}))}]}) *
                           (float{1.442695})) -
                          ((float{prodStatsPtr119[int32_t{3}]}) * (scaleSoftmaxLog219)))};
          }
          // TmemCorr.h:582
          prodStatsPtr119[int32_t{0}] = (sum0) + (attentionSinkVal0);
          // TmemCorr.h:583
          prodStatsPtr119[int32_t{1}] = (sum1) + (attentionSinkVal1);
          // TmemCorr.h:590
          scales19[int32_t{0}] = (float(bool{params.ptrOutputScale == nullptr})
                                    ? (params.mOutputScale)
                                    : (float{params.ptrOutputScale[int32_t{0}]})) /
                                 ((sum0) + (attentionSinkVal0));
          // TmemCorr.h:591
          scales19[int32_t{1}] = (float(bool{params.ptrOutputScale == nullptr})
                                    ? (params.mOutputScale)
                                    : (float{params.ptrOutputScale[int32_t{0}]})) /
                                 ((sum1) + (attentionSinkVal1));
        }
        // TmemCorr.h:3714
        if (storesSoftmaxStats) {
          // TmemCorr.h:3718
          float scaleSoftmax{(scaleSoftmaxLog219) * (float{0.69314718})};
          // TmemCorr.h:3731
          (prodStatsPtr119 + int32_t{2})[int32_t{0}] =
            (float{(prodStatsPtr119 + int32_t{2})[int32_t{0}]}) * (scaleSoftmax);
          // TmemCorr.h:3731
          (prodStatsPtr119 + int32_t{2})[int32_t{1}] =
            (float{(prodStatsPtr119 + int32_t{2})[int32_t{1}]}) * (scaleSoftmax);
          // TmemCorr.h:3797
          trtllm::dev::storeStatsForSwappedAb<int32_t{2}, false>(
            (prodStatsPtr119 + int32_t{2}),
            prodStatsPtr119,
            ptrSoftmaxStats,
            params.mNumHeadsQ,
            params.mNumHeadsQPerKv,
            mWarpGrpThreadIdx,
            int32_t{min(params.mNumHeadsQPerKv, int32_t{8})});
        }
        //
        // The headDimStageIdx: 0.
        //
        // TmemCorr.h:1472
        for (int32_t loopOffset1944 = int32_t{0}; loopOffset1944 < int32_t{4};
             loopOffset1944 += int32_t{8}) {
          // TmemTile.cpp:373
          cutlass::Array<float, 4> tmemRegs019;
          // TmemTile.cpp:527
          {
            // TmemTile.cpp:529
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:545
            uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(tmemRegs019[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_16x256b(
              dstSlice0,
              (tmemBasePtr) +
                (static_cast<uint32_t>((int32_t{0x50 /*hi=0, lo=80*/}) + (loopOffset1944))));
          }
          // TmemCorr.h:3257
          uint32_t mRegsO19[2];
          // TmemCorr.h:1520
          {
            // TmemCorr.h:1540
            cutlass::Array<float, 2> localScales0{scales19[int32_t{0}], scales19[int32_t{1}]};
            // TmemCorr.h:1551
            cutlass::Array<float, 2> vals0{tmemRegs019[int32_t{0}], tmemRegs019[int32_t{1}]};
            // TmemCorr.h:1563
            vals0 = trtllm::dev::fmul2(vals0, localScales0);
            // TmemCorr.h:1566
            tmemRegs019[int32_t{0}] = vals0[int32_t{0}];
            // TmemCorr.h:1567
            tmemRegs019[int32_t{1}] = vals0[int32_t{1}];
            // TmemCorr.h:3487
            mRegsO19[int32_t{0}] = trtllm::dev::convert_float2_to_bfloat16(tmemRegs019[int32_t{0}],
                                                                           tmemRegs019[int32_t{1}]);
          }
          // TmemCorr.h:1520
          {
            // TmemCorr.h:1540
            cutlass::Array<float, 2> localScales0{scales19[int32_t{0}], scales19[int32_t{1}]};
            // TmemCorr.h:1551
            cutlass::Array<float, 2> vals0{tmemRegs019[int32_t{2}], tmemRegs019[int32_t{3}]};
            // TmemCorr.h:1563
            vals0 = trtllm::dev::fmul2(vals0, localScales0);
            // TmemCorr.h:1566
            tmemRegs019[int32_t{2}] = vals0[int32_t{0}];
            // TmemCorr.h:1567
            tmemRegs019[int32_t{3}] = vals0[int32_t{1}];
            // TmemCorr.h:3487
            mRegsO19[int32_t{1}] = trtllm::dev::convert_float2_to_bfloat16(tmemRegs019[int32_t{2}],
                                                                           tmemRegs019[int32_t{3}]);
          }
          // TmemCorr.h:3665
          trtllm::dev::reorganizeInSmemAndStoreToDstMem<int32_t{64}, int32_t{8}, false>(
            reinterpret_cast<cutlass::bfloat16_t*>(tmemCorr0DstStack.mDepSmemPtr7),
            ptrO,
            mRegsO19,
            int32_t{64},
            int32_t{min(params.mNumHeadsQPerKv, int32_t{8})},
            params.mNumHeadsQ,
            params.mNumHeadsQPerKv,
            int32_t{128},
            mWarpGrpThreadIdx,
            int32_t{3});
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
      // Task.cpp:5416
      auto newWorkTileInfoTuple{
        workIdStorageSrcStack.mScheduler.fetch_next_work(workIdStorageSrcStack.workTileInfo,
                                                         workIdStorageSrcStack.mPipeline,
                                                         workIdStorageConsState)};
      // Task.cpp:5418
      workIdStorageSrcStack.workTileInfo = cute::get<int32_t{0}>(newWorkTileInfoTuple);
      // Task.cpp:5426
      ++workIdStorageConsState;
      // Task.cpp:5528
      mCtaIdxX = workIdStorageSrcStack.workTileInfo.M_idx;
      // Task.cpp:5529
      mCtaIdxY = workIdStorageSrcStack.workTileInfo.N_idx;
      // Task.cpp:5530
      mCtaIdxZ = workIdStorageSrcStack.workTileInfo.L_idx;
    } while (workIdStorageSrcStack.workTileInfo.is_valid_tile);
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
  // FmhaTask.h:733
  int32_t mNumSkippedTilesKv;
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
    , // Task.cpp:283
    mCtaIdxQ{mCtaIdxX}
    , // FmhaTask.h:517
    mCtaIdxKv{int32_t{0}}
    , // FmhaTask.h:543
    mSeqLenKv{(int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                             ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                             : (mBatchIdx)]}) -
              (((mSeqLenQ) - (int32_t{1})) - (mCtaIdxQ))}
    , // Kernel.cpp:212
    mNumCtasKv{int32_t{1}}
    , // Kernel.cpp:210
    mNumSkippedTilesKv{int32_t{0}}
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
                                 SmemKvStack& smemKvSrcStack,
                                 WorkIdStorageSmem& workIdStorageSrcSmem,
                                 WorkIdStorageStack& workIdStorageSrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_inc(cuda_ptx::n32_t<184>{});
    // Task.cpp:2079
    trtllm::dev::CutlassTmaAsyncPipeline<9>::PipelineState smemKvConsState{};
    // Task.cpp:2086
    trtllm::dev::CutlassTmaAsyncPipeline<9>::PipelineState smemKvConsReleaseState{};
    // Task.cpp:2100
    int32_t smemKvConsToken{int32_t{0}};
    // Task.cpp:2079
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsState{};
    // Task.cpp:2086
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsReleaseState{};
    // Task.cpp:2100
    int32_t workIdStorageConsToken{int32_t{0}};
    // Task.cpp:1979
    trtllm::dev::CutlassUmmaConsumerAsyncPipeline<2, false, false>::PipelineState
      smemTransformedKvProdState{int32_t{0}, int32_t{1}, int32_t{0}};
    // Task.cpp:1999
    int32_t smemTransformedKvProdToken{int32_t{1}};
    // Task.cpp:5369
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
      mSeqLenKv =
        (int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                       ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                       : (mBatchIdx)]}) -
        (((mSeqLenQ) - (int32_t{1})) - (mCtaIdxQ));
      // FmhaTask.h:582
      int32_t numLoopSteps;
      // FmhaTask.h:592
      int32_t diffKvQ{(mSeqLenKv) - (mSeqLenQ)};
      // FmhaTask.h:597
      int32_t validSeqLenKv;
      // Common.h:63
      if ((params.mChunkedAttentionSizeLog2) > (int32_t{0})) {
        // FmhaTask.h:748
        mNumSkippedTilesKv = ((((mCtaIdxQ) + (diffKvQ)) >> (params.mChunkedAttentionSizeLog2))
                              << (params.mChunkedAttentionSizeLog2)) /
                             (int32_t{128});
      } else {
        // FmhaTask.h:767
        mNumSkippedTilesKv = (int32_t{max(int32_t{0},
                                          (((mCtaIdxQ) + (diffKvQ)) + (int32_t{1})) -
                                            (params.mAttentionWindowSize))}) /
                             (int32_t{128});
      }
      // FmhaTask.h:603
      validSeqLenKv = (int32_t{min(((mCtaIdxQ) + (diffKvQ)) + (int32_t{1}), mSeqLenKv)}) -
                      ((mNumSkippedTilesKv) * (int32_t{128}));
      // FmhaTask.h:616
      mNumCtasKv =
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, int32_t{1})};
      // FmhaTask.h:630
      if (((mCtaIdxQ) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
        // FmhaTask.h:668
        int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
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
        srcPtr = smemPtrK4 + ((smemIdxK4) * (int32_t{8192}));
        // SmemTransformedKv.h:241
        uint64_t srcBuffer4[8];
        // SmemTransformedKv.h:261
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset2089 = int32_t{0}; loopOffset2089 < int32_t{2}; ++loopOffset2089) {
          // SmemTransformedKv.h:284
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset2091 = (loopOffset2089) * (int32_t{512});
               loopOffset2091 < ((loopOffset2089) + (int32_t{1})) * (int32_t{512});
               loopOffset2091 += int32_t{128}) {
            // SmemTransformedKv.h:293
            int32_t offset{(loopOffset2091) + (mWarpGrpThreadIdx)};
            // SmemTransformedKv.h:304
            srcBuffer4[(loopOffset2091) / (int32_t{128})] =
              reinterpret_cast<uint64_t*>(srcPtr)[offset];
          }
          // SmemTransformedKv.h:348
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset2095 = (loopOffset2089) * (int32_t{512});
               loopOffset2095 < ((loopOffset2089) + (int32_t{1})) * (int32_t{512});
               loopOffset2095 += int32_t{128}) {
            // SmemTransformedKv.h:356
            int32_t offset{(loopOffset2095) + (mWarpGrpThreadIdx)};
            // SmemTransformedKv.h:385
            cutlass::uint128_t dst{
              trtllm::dev::convertE4m3ToBfloat16(srcBuffer4[(loopOffset2095) / (int32_t{128})])};
            // SmemTransformedKv.h:397
            int32_t eltIdx{(offset) * (int32_t{8})};
            // SmemTile.cpp:369
            int32_t smemRowIdx{(eltIdx) / (int32_t{64})};
            // SmemTile.cpp:372
            int32_t smemOffsetInBytes{((eltIdx) * (int32_t{16})) / (int32_t{8})};
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
      for (int32_t loopOffset2125 = int32_t{0}; loopOffset2125 < (numLoopSteps) - (int32_t{1});
           ++loopOffset2125) {
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
        // smemKv [ConsWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{0}].
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
          if ((loopOffset2125) >= (int32_t{0})) {
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
        // smemTransformedKv [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
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
          srcPtr = smemPtrK4 + ((smemIdxK4) * (int32_t{8192}));
          // SmemTransformedKv.h:241
          uint64_t srcBuffer4[8];
          // SmemTransformedKv.h:261
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset2154 = int32_t{0}; loopOffset2154 < int32_t{2}; ++loopOffset2154) {
            // SmemTransformedKv.h:284
            CUTLASS_PRAGMA_UNROLL
            for (int32_t loopOffset2156 = (loopOffset2154) * (int32_t{512});
                 loopOffset2156 < ((loopOffset2154) + (int32_t{1})) * (int32_t{512});
                 loopOffset2156 += int32_t{128}) {
              // SmemTransformedKv.h:293
              int32_t offset{(loopOffset2156) + (mWarpGrpThreadIdx)};
              // SmemTransformedKv.h:304
              srcBuffer4[(loopOffset2156) / (int32_t{128})] =
                reinterpret_cast<uint64_t*>(srcPtr)[offset];
            }
            // SmemTransformedKv.h:348
            CUTLASS_PRAGMA_UNROLL
            for (int32_t loopOffset2160 = (loopOffset2154) * (int32_t{512});
                 loopOffset2160 < ((loopOffset2154) + (int32_t{1})) * (int32_t{512});
                 loopOffset2160 += int32_t{128}) {
              // SmemTransformedKv.h:356
              int32_t offset{(loopOffset2160) + (mWarpGrpThreadIdx)};
              // SmemTransformedKv.h:385
              cutlass::uint128_t dst{
                trtllm::dev::convertE4m3ToBfloat16(srcBuffer4[(loopOffset2160) / (int32_t{128})])};
              // SmemTransformedKv.h:397
              int32_t eltIdx{(offset) * (int32_t{8})};
              // SmemTile.cpp:369
              int32_t smemRowIdx{(eltIdx) / (int32_t{64})};
              // SmemTile.cpp:372
              int32_t smemOffsetInBytes{((eltIdx) * (int32_t{16})) / (int32_t{8})};
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
        if ((loopOffset2125) >= (int32_t{0})) {
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
        // smemKv [ConsWork (call 2), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
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
          if ((loopOffset2125) >= (int32_t{0})) {
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
        // smemTransformedKv [ProdWork (call 2), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
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
          srcPtr = smemPtrV4 + ((smemIdxV4) * (int32_t{8192}));
          // SmemTransformedKv.h:241
          uint64_t srcBuffer4[8];
          // SmemTransformedKv.h:261
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset2216 = int32_t{0}; loopOffset2216 < int32_t{2}; ++loopOffset2216) {
            // SmemTransformedKv.h:284
            CUTLASS_PRAGMA_UNROLL
            for (int32_t loopOffset2218 = (loopOffset2216) * (int32_t{512});
                 loopOffset2218 < ((loopOffset2216) + (int32_t{1})) * (int32_t{512});
                 loopOffset2218 += int32_t{128}) {
              // SmemTransformedKv.h:293
              int32_t offset{(loopOffset2218) + (mWarpGrpThreadIdx)};
              // SmemTransformedKv.h:304
              srcBuffer4[(loopOffset2218) / (int32_t{128})] =
                reinterpret_cast<uint64_t*>(srcPtr)[offset];
            }
            // SmemTransformedKv.h:348
            CUTLASS_PRAGMA_UNROLL
            for (int32_t loopOffset2222 = (loopOffset2216) * (int32_t{512});
                 loopOffset2222 < ((loopOffset2216) + (int32_t{1})) * (int32_t{512});
                 loopOffset2222 += int32_t{128}) {
              // SmemTransformedKv.h:356
              int32_t offset{(loopOffset2222) + (mWarpGrpThreadIdx)};
              // SmemTransformedKv.h:385
              cutlass::uint128_t dst{
                trtllm::dev::convertE4m3ToBfloat16(srcBuffer4[(loopOffset2222) / (int32_t{128})])};
              // SmemTransformedKv.h:397
              int32_t eltIdx{(offset) * (int32_t{8})};
              // SmemTile.cpp:369
              int32_t smemRowIdx{(eltIdx) / (int32_t{64})};
              // SmemTile.cpp:372
              int32_t smemOffsetInBytes{((eltIdx) * (int32_t{16})) / (int32_t{8})};
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
        if ((loopOffset2125) >= (int32_t{0})) {
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
        lastLoopOffset = loopOffset2125;
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
      // smemKv [ConsWork (call 3), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
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
      // smemTransformedKv [ProdWork (call 3), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
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
        srcPtr = smemPtrV4 + ((smemIdxV4) * (int32_t{8192}));
        // SmemTransformedKv.h:241
        uint64_t srcBuffer4[8];
        // SmemTransformedKv.h:261
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset2296 = int32_t{0}; loopOffset2296 < int32_t{2}; ++loopOffset2296) {
          // SmemTransformedKv.h:284
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset2298 = (loopOffset2296) * (int32_t{512});
               loopOffset2298 < ((loopOffset2296) + (int32_t{1})) * (int32_t{512});
               loopOffset2298 += int32_t{128}) {
            // SmemTransformedKv.h:293
            int32_t offset{(loopOffset2298) + (mWarpGrpThreadIdx)};
            // SmemTransformedKv.h:304
            srcBuffer4[(loopOffset2298) / (int32_t{128})] =
              reinterpret_cast<uint64_t*>(srcPtr)[offset];
          }
          // SmemTransformedKv.h:348
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset2302 = (loopOffset2296) * (int32_t{512});
               loopOffset2302 < ((loopOffset2296) + (int32_t{1})) * (int32_t{512});
               loopOffset2302 += int32_t{128}) {
            // SmemTransformedKv.h:356
            int32_t offset{(loopOffset2302) + (mWarpGrpThreadIdx)};
            // SmemTransformedKv.h:385
            cutlass::uint128_t dst{
              trtllm::dev::convertE4m3ToBfloat16(srcBuffer4[(loopOffset2302) / (int32_t{128})])};
            // SmemTransformedKv.h:397
            int32_t eltIdx{(offset) * (int32_t{8})};
            // SmemTile.cpp:369
            int32_t smemRowIdx{(eltIdx) / (int32_t{64})};
            // SmemTile.cpp:372
            int32_t smemOffsetInBytes{((eltIdx) * (int32_t{16})) / (int32_t{8})};
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
      // Task.cpp:5416
      auto newWorkTileInfoTuple{
        workIdStorageSrcStack.mScheduler.fetch_next_work(workIdStorageSrcStack.workTileInfo,
                                                         workIdStorageSrcStack.mPipeline,
                                                         workIdStorageConsState)};
      // Task.cpp:5418
      workIdStorageSrcStack.workTileInfo = cute::get<int32_t{0}>(newWorkTileInfoTuple);
      // Task.cpp:5426
      ++workIdStorageConsState;
      // Task.cpp:5528
      mCtaIdxX = workIdStorageSrcStack.workTileInfo.M_idx;
      // Task.cpp:5529
      mCtaIdxY = workIdStorageSrcStack.workTileInfo.N_idx;
      // Task.cpp:5530
      mCtaIdxZ = workIdStorageSrcStack.workTileInfo.L_idx;
    } while (workIdStorageSrcStack.workTileInfo.is_valid_tile);
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
  // FmhaTask.h:733
  int32_t mNumSkippedTilesKv;
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
    , // Task.cpp:283
    mCtaIdxQ{mCtaIdxX}
    , // FmhaTask.h:517
    mCtaIdxKv{int32_t{0}}
    , // FmhaTask.h:543
    mSeqLenKv{(int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                             ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                             : (mBatchIdx)]}) -
              (((mSeqLenQ) - (int32_t{1})) - (mCtaIdxQ))}
    , // Kernel.cpp:212
    mNumCtasKv{int32_t{1}}
    , // Kernel.cpp:210
    mNumSkippedTilesKv{int32_t{0}}
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
                                 TmemP0Stack& tmemP0SrcStack,
                                 WorkIdStorageSmem& workIdStorageSrcSmem,
                                 WorkIdStorageStack& workIdStorageSrcStack) {
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
    // Task.cpp:2079
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsState{};
    // Task.cpp:2086
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsReleaseState{};
    // Task.cpp:2100
    int32_t workIdStorageConsToken{int32_t{0}};
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
    //
    // tmemS0 [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{64}].
    //
    // Task.cpp:1573
    // Task.cpp:4948
    {
      // Task.cpp:4962
      {
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
    // tmemS0 [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{8256}].
    //
    // Task.cpp:1573
    // Task.cpp:4948
    {
      // Task.cpp:4962
      {
        // Task.cpp:4984
        tmemS0ProdToken = tmemS0DstStack.mPipeline.producer_try_acquire(
          trtllm::dev::makePipelineState(tmemS0ProdState, int32_t{1}));
      }
    }
    // Task.cpp:1573
    // Task.cpp:4180
    {
      // Task.cpp:4210
      tmemS0DstStack.mPipeline.producer_acquire(
        trtllm::dev::makePipelineState(tmemS0ProdState, int32_t{1}),
        tmemS0ProdToken);
    }
    // Task.cpp:5369
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
      mSeqLenKv =
        (int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                       ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                       : (mBatchIdx)]}) -
        (((mSeqLenQ) - (int32_t{1})) - (mCtaIdxQ));
      // FmhaTask.h:582
      int32_t numLoopSteps;
      // FmhaTask.h:592
      int32_t diffKvQ{(mSeqLenKv) - (mSeqLenQ)};
      // FmhaTask.h:597
      int32_t validSeqLenKv;
      // Common.h:63
      if ((params.mChunkedAttentionSizeLog2) > (int32_t{0})) {
        // FmhaTask.h:748
        mNumSkippedTilesKv = ((((mCtaIdxQ) + (diffKvQ)) >> (params.mChunkedAttentionSizeLog2))
                              << (params.mChunkedAttentionSizeLog2)) /
                             (int32_t{128});
      } else {
        // FmhaTask.h:767
        mNumSkippedTilesKv = (int32_t{max(int32_t{0},
                                          (((mCtaIdxQ) + (diffKvQ)) + (int32_t{1})) -
                                            (params.mAttentionWindowSize))}) /
                             (int32_t{128});
      }
      // FmhaTask.h:603
      validSeqLenKv = (int32_t{min(((mCtaIdxQ) + (diffKvQ)) + (int32_t{1}), mSeqLenKv)}) -
                      ((mNumSkippedTilesKv) * (int32_t{128}));
      // FmhaTask.h:616
      mNumCtasKv =
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, int32_t{1})};
      // FmhaTask.h:630
      if (((mCtaIdxQ) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
        // FmhaTask.h:668
        int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
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
      cutlass::bfloat16_t* smemPtrQ0_2{&smemQSrcSmem.mArray[int32_t{0}][int32_t{0}]};
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
      // tmemS0 [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{64}].
      //
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
      cutlass::bfloat16_t* smemPtrK4;
      // SmemKv.h:206
      int32_t smemIdxK4;
      // SmemKv.h:214
      cutlass::bfloat16_t* smemPtrV4;
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
      cutlass::bfloat16_t* smemPtrQ13;
      // TmemS.h:1128
      int32_t smemIdxQ13;
      // TmemS.h:1134
      cutlass::bfloat16_t* smemPtrK13;
      // TmemS.h:1139
      int32_t memIdxK13;
      // Task.cpp:1477
      smemPtrQ13 = smemPtrQ0_2;
      // Task.cpp:1477
      smemIdxQ13 = smemIdxQ0_2;
      // Task.cpp:1477
      smemPtrK13 = smemPtrK4;
      // Task.cpp:1477
      memIdxK13 = smemIdxK4;
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:5829
        int32_t index{tmemS0ProdState.index()};
        // TmemS.h:1874
        cutlass::bfloat16_t* smemQ{smemPtrQ13};
        // TmemS.h:1895
        smemQ += (smemIdxQ13) * (int32_t{512});
        // TmemS.h:1923
        cutlass::bfloat16_t* smemK{smemPtrK13};
        // TmemS.h:1929
        smemK += (memIdxK13) * (int32_t{8192});
        // Mma.cpp:618
        {
          // TmemTile.cpp:1755
          uint32_t tmemPtrD{
            (index) * (int32_t{8}) +
            (int32_t((tmemS0DstStack.mInstId) == (int32_t{0})) ? (int32_t{0}) : (int32_t{8}))};
          //
          // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
          //
          // Mma.cpp:203
          uint64_t smemDescA{
            trtllm::dev::createSmemDesc(smemK,
                                        uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                        uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
          //
          // leadingDimInBytes = 1024, strideInBytes = 1024, swizzleMode = 1.
          //
          // Mma.cpp:203
          uint64_t smemDescB{
            trtllm::dev::createSmemDesc(smemQ,
                                        uint32_t{0x400000 /*hi=64, lo=0*/},
                                        uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
          //
          // MMA inst for mi=0 ni=0 ki=0.
          //
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{8},
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
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{8},
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
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{8},
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
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{8},
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
      // tmemS0 [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{8256}].
      //
      //
      // Loop body.
      //
      // Task.cpp:3350
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset2541 = int32_t{0}; loopOffset2541 < (numLoopSteps) - (int32_t{1});
           ++loopOffset2541) {
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
        // smemTransformedKv [ConsWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
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
        // tmemS0 [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        // Task.cpp:1477
        smemPtrQ13 = smemPtrQ0_2;
        // Task.cpp:1477
        smemIdxQ13 = smemIdxQ0_2;
        // Task.cpp:1477
        smemPtrK13 = smemPtrK4;
        // Task.cpp:1477
        memIdxK13 = smemIdxK4;
        // Task.cpp:1573
        // Task.cpp:5038
        {
          // Task.cpp:5829
          int32_t index{tmemS0ProdState.index()};
          // TmemS.h:1874
          cutlass::bfloat16_t* smemQ{smemPtrQ13};
          // TmemS.h:1895
          smemQ += (smemIdxQ13) * (int32_t{512});
          // TmemS.h:1923
          cutlass::bfloat16_t* smemK{smemPtrK13};
          // TmemS.h:1929
          smemK += (memIdxK13) * (int32_t{8192});
          // Mma.cpp:618
          {
            // TmemTile.cpp:1755
            uint32_t tmemPtrD{
              (index) * (int32_t{8}) +
              (int32_t((tmemS0DstStack.mInstId) == (int32_t{0})) ? (int32_t{0}) : (int32_t{8}))};
            //
            // leadingDimInBytes = 16384, strideInBytes = 1024, swizzleMode = 1.
            //
            // Mma.cpp:203
            uint64_t smemDescA{
              trtllm::dev::createSmemDesc(smemK,
                                          uint32_t{0x4000000 /*hi=1024, lo=0*/},
                                          uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
            //
            // leadingDimInBytes = 1024, strideInBytes = 1024, swizzleMode = 1.
            //
            // Mma.cpp:203
            uint64_t smemDescB{
              trtllm::dev::createSmemDesc(smemQ,
                                          uint32_t{0x400000 /*hi=64, lo=0*/},
                                          uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
            //
            // MMA inst for mi=0 ni=0 ki=0.
            //
            // TmemTile.cpp:1600
            uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{8},
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
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{8},
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
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{8},
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
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    false,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{8},
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
          }
        }
        //
        // smemTransformedKv [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:2533
        if ((loopOffset2541) >= (int32_t{0})) {
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
          if ((loopOffset2541) >= (int32_t{0})) {
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
        int32_t stageIdxP17;
        // Task.cpp:1573
        // Task.cpp:2893
        {
          // Task.cpp:5829
          int32_t index{tmemP0ConsState.index()};
          // TmemP.h:488
          stageIdxP17 = index;
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
          if ((loopOffset2541) >= (int32_t{0})) {
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
        // smemTransformedKv [ConsWork (call 2), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
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
        cutlass::bfloat16_t* smemPtrV18;
        // TmemO.h:282
        int32_t memIdxV18;
        // TmemO.h:288
        int32_t smemIdxP18;
        // Task.cpp:1477
        smemPtrV18 = smemPtrV4;
        // Task.cpp:1477
        memIdxV18 = smemIdxV4;
        // Task.cpp:1477
        smemIdxP18 = stageIdxP17;
        // Task.cpp:1573
        // Task.cpp:5038
        {
          // Task.cpp:5829
          int32_t index{tmemOProdState.index()};
          // TmemO.h:367
          cutlass::bfloat16_t* smemP{
            reinterpret_cast<cutlass::bfloat16_t*>(tmemODstStack.mDepSmemPtr6)};
          // TmemO.h:381
          smemP += (smemIdxP18) * (int32_t{1024});
          // TmemO.h:493
          cutlass::bfloat16_t* smemV{smemPtrV18};
          // TmemO.h:505
          smemV = smemV + ((memIdxV18) * (int32_t{8192}));
          // TmemO.h:535
          bool readD{true};
          // TmemO.h:545
          if ((loopOffset2541) == (int32_t{0})) {
            // TmemO.h:547
            readD = false;
          }
          // Mma.cpp:618
          {
            // TmemTile.cpp:1755
            uint32_t tmemPtrD{(mTmemBaseOffset) + (uint32_t{80})};
            //
            // leadingDimInBytes = 0, strideInBytes = 1024, swizzleMode = 1.
            //
            // Mma.cpp:203
            uint64_t smemDescA{
              trtllm::dev::createSmemDesc(smemV,
                                          uint32_t{0x0 /*hi=0, lo=0*/},
                                          uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
            //
            // leadingDimInBytes = 1024, strideInBytes = 1024, swizzleMode = 1.
            //
            // Mma.cpp:203
            uint64_t smemDescB{
              trtllm::dev::createSmemDesc(smemP,
                                          uint32_t{0x400000 /*hi=64, lo=0*/},
                                          uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
            //
            // MMA inst for mi=0 ni=0 ki=0.
            //
            // TmemTile.cpp:1600
            uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    true,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{8},
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
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    true,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{8},
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
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    true,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{8},
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
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    true,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{8},
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
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{58});
            // TmemTile.cpp:1600
            uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    true,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{8},
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
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    true,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{8},
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
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    true,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{8},
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
                                                                    int32_t{1},
                                                                    int32_t{1},
                                                                    true,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{8},
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
        if ((loopOffset2541) >= (int32_t{0})) {
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
        lastLoopOffset = loopOffset2541;
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
      // tmemS0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{4}].
      //
      //
      // tmemS0 [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{8192}].
      //
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:4962
        if ((lastLoopOffset) >= (int32_t{0})) {
          // Task.cpp:4984
          tmemS0ProdToken = tmemS0DstStack.mPipeline.producer_try_acquire(
            trtllm::dev::makePipelineState(tmemS0ProdState, int32_t{1}));
        }
      }
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:4210
        tmemS0DstStack.mPipeline.producer_acquire(
          trtllm::dev::makePipelineState(tmemS0ProdState, int32_t{1}),
          tmemS0ProdToken);
      }
      //
      // tmemP0 [ConsWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // TmemP.h:474
      int32_t stageIdxP17;
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:5829
        int32_t index{tmemP0ConsState.index()};
        // TmemP.h:488
        stageIdxP17 = index;
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
      // smemTransformedKv [ConsWork (call 3), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
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
      // tmemO [ProdWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{3}, Flags{0}].
      //
      // TmemO.h:277
      cutlass::bfloat16_t* smemPtrV18;
      // TmemO.h:282
      int32_t memIdxV18;
      // TmemO.h:288
      int32_t smemIdxP18;
      // Task.cpp:1477
      smemPtrV18 = smemPtrV4;
      // Task.cpp:1477
      memIdxV18 = smemIdxV4;
      // Task.cpp:1477
      smemIdxP18 = stageIdxP17;
      // Task.cpp:1573
      if (hasOneLoopIter) {
        // Task.cpp:5829
        int32_t index{tmemOProdState.index()};
        // TmemO.h:367
        cutlass::bfloat16_t* smemP{
          reinterpret_cast<cutlass::bfloat16_t*>(tmemODstStack.mDepSmemPtr6)};
        // TmemO.h:381
        smemP += (smemIdxP18) * (int32_t{1024});
        // TmemO.h:493
        cutlass::bfloat16_t* smemV{smemPtrV18};
        // TmemO.h:505
        smemV = smemV + ((memIdxV18) * (int32_t{8192}));
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
          uint32_t tmemPtrD{(mTmemBaseOffset) + (uint32_t{80})};
          //
          // leadingDimInBytes = 0, strideInBytes = 1024, swizzleMode = 1.
          //
          // Mma.cpp:203
          uint64_t smemDescA{
            trtllm::dev::createSmemDesc(smemV,
                                        uint32_t{0x0 /*hi=0, lo=0*/},
                                        uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
          //
          // leadingDimInBytes = 1024, strideInBytes = 1024, swizzleMode = 1.
          //
          // Mma.cpp:203
          uint64_t smemDescB{
            trtllm::dev::createSmemDesc(smemP,
                                        uint32_t{0x400000 /*hi=64, lo=0*/},
                                        uint32_t{0x40004040 /*hi=16384, lo=16448*/})};
          //
          // MMA inst for mi=0 ni=0 ki=0.
          //
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  true,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{8},
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
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  true,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{8},
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
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  true,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{8},
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
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  true,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{8},
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
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{58});
          // TmemTile.cpp:1600
          uint64_t utcmmaDesc_0_0_4{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  true,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{8},
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
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  true,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{8},
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
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  true,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{8},
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
                                                                  int32_t{1},
                                                                  int32_t{1},
                                                                  true,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{8},
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
    // tmemS0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{4}].
    //
    //
    // Tail work.
    //
    // Task.cpp:3511
    ExitTileWithSignalingLabel:
    // Task.cpp:3518
    ExitTileWithoutSignalingLabel:
      // Task.cpp:5416
      auto newWorkTileInfoTuple{
        workIdStorageSrcStack.mScheduler.fetch_next_work(workIdStorageSrcStack.workTileInfo,
                                                         workIdStorageSrcStack.mPipeline,
                                                         workIdStorageConsState)};
      // Task.cpp:5418
      workIdStorageSrcStack.workTileInfo = cute::get<int32_t{0}>(newWorkTileInfoTuple);
      // Task.cpp:5426
      ++workIdStorageConsState;
      // Task.cpp:5528
      mCtaIdxX = workIdStorageSrcStack.workTileInfo.M_idx;
      // Task.cpp:5529
      mCtaIdxY = workIdStorageSrcStack.workTileInfo.N_idx;
      // Task.cpp:5530
      mCtaIdxZ = workIdStorageSrcStack.workTileInfo.L_idx;
    } while (workIdStorageSrcStack.workTileInfo.is_valid_tile);
    //
    // tmemS0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{4}].
    //
    // Task.cpp:4392
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
    // tmemS0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{4}].
    //
    // Task.cpp:4392
    {
      // Task.cpp:4427
      {
        // Task.cpp:4443
        tmemS0DstStack.mPipeline.producer_commit(tmemS0ProdState);
      }
      // Task.cpp:43
      ++tmemS0ProdState;
    }
  }
};
// Task.cpp:544
// Fmha.h:2620
struct SchedulerTask {
  // Task.cpp:283
  int32_t mCtaIdxX;
  // Task.cpp:287
  int32_t mCtaIdxY;
  // Task.cpp:291
  int32_t mCtaIdxZ;
  // Task.cpp:551
  inline __device__ SchedulerTask(fmha::KernelParams const& params,
                                  KernelState const& state,
                                  int32_t warpGrpStart)
    : // Kernel.cpp:194
    mCtaIdxX{reinterpret_cast<int32_t const&>(blockIdx.x)}
    , // Kernel.cpp:195
    mCtaIdxY{reinterpret_cast<int32_t const&>(blockIdx.y)}
    , // Kernel.cpp:196
    mCtaIdxZ{reinterpret_cast<int32_t const&>(blockIdx.z)} {}
  // Task.cpp:507
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:532
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
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<56>{});
    // Task.cpp:2079
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsState{};
    // Task.cpp:2086
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageConsReleaseState{};
    // Task.cpp:2100
    int32_t workIdStorageConsToken{int32_t{0}};
    // Task.cpp:2079
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState workIdThrottleBarrierConsState{};
    // Task.cpp:2086
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState
      workIdThrottleBarrierConsReleaseState{};
    // Task.cpp:2100
    int32_t workIdThrottleBarrierConsToken{int32_t{0}};
    // Task.cpp:1979
    trtllm::dev::CutlassWorkIdPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::
      PipelineState workIdStorageProdState{int32_t{0}, int32_t{1}, int32_t{0}};
    // Task.cpp:1999
    int32_t workIdStorageProdToken{int32_t{1}};
    // Task.cpp:6218
    do {
      // Task.cpp:6371
      workIdThrottleBarrierSrcStack.mPipeline.consumer_wait(workIdThrottleBarrierConsState,
                                                            workIdThrottleBarrierConsToken);
      // Task.cpp:6378
      workIdThrottleBarrierSrcStack.mPipeline.consumer_release(workIdThrottleBarrierConsState);
      // Task.cpp:6380
      ++workIdThrottleBarrierConsState;
      // Task.cpp:6342
      workIdStorageProdState = workIdStorageSrcStack.mScheduler.advance_to_next_work(
        workIdStorageSrcStack.mPipeline.get_pipeline(),
        workIdStorageProdState);
      // Task.cpp:5416
      auto newWorkTileInfoTuple{
        workIdStorageSrcStack.mScheduler.fetch_next_work(workIdStorageSrcStack.workTileInfo,
                                                         workIdStorageSrcStack.mPipeline,
                                                         workIdStorageConsState)};
      // Task.cpp:5418
      workIdStorageSrcStack.workTileInfo = cute::get<int32_t{0}>(newWorkTileInfoTuple);
      // Task.cpp:5426
      ++workIdStorageConsState;
      // Task.cpp:5528
      mCtaIdxX = workIdStorageSrcStack.workTileInfo.M_idx;
      // Task.cpp:5529
      mCtaIdxY = workIdStorageSrcStack.workTileInfo.N_idx;
      // Task.cpp:5530
      mCtaIdxZ = workIdStorageSrcStack.workTileInfo.L_idx;
    } while (workIdStorageSrcStack.workTileInfo.is_valid_tile);
    // Task.cpp:6301
    workIdStorageDstStack.mPipeline.producer_tail(workIdStorageProdState);
  }
};
extern "C" __global__
__launch_bounds__(512, 1) void fmhaSm100fKernel_QBfloat16KvE4m3OBfloat16H64PagedKvSlidingOrChunkedCausalP32VarSeqQ8Kv128PersistentSwapsAbForGen(
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
  uint8_t* smemPSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemPSmem)});
  // Kernel.cpp:1721
  smemOffset__ = (((smemOffset__) + (int32_t{127})) / (int32_t{128})) * (int32_t{128});
  // Kernel.cpp:1725
  uint8_t* smemPageOffsetsKvSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemPageOffsetsKvSmem)});
  // Kernel.cpp:1721
  smemOffset__ = (((smemOffset__) + (int32_t{127})) / (int32_t{128})) * (int32_t{128});
  // Kernel.cpp:1725
  uint8_t* smemOSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemOSmem)});
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
  // Kernel.cpp:1721
  smemOffset__ = (((smemOffset__) + (int32_t{15})) / (int32_t{16})) * (int32_t{16});
  // Kernel.cpp:1725
  uint8_t* workIdStorageSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(WorkIdStorageSmem)});
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
  uint8_t* workIdStorageSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(WorkIdStorageSmemBarrier)});
  // Kernel.cpp:1725
  uint8_t* workIdThrottleBarrierSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1741
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(WorkIdThrottleBarrierSmemBarrier)});
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
  SmemPSmem* smemPSmem{reinterpret_cast<SmemPSmem*>(smemPSmemPtr)};
  // Kernel.cpp:2279
  SmemPStack smemPStack{(*smemPSmem),
                        state.mWarpIdx,
                        state.mClusterDimX,
                        state.mClusterDimY,
                        int32_t{0},
                        int32_t{-1}};
  // Kernel.cpp:2212
  SmemOSmem* smemOSmem{reinterpret_cast<SmemOSmem*>(smemOSmemPtr)};
  // Kernel.cpp:2279
  SmemOStack smemOStack{(*smemOSmem),
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
  // Kernel.cpp:2212
  WorkIdStorageSmem* workIdStorageSmem{reinterpret_cast<WorkIdStorageSmem*>(workIdStorageSmemPtr)};
  // Kernel.cpp:2224
  WorkIdStorageSmemBarrier* workIdStorageSmemBarrier{
    reinterpret_cast<WorkIdStorageSmemBarrier*>(workIdStorageSmemBarrierPtr)};
  // Kernel.cpp:2279
  WorkIdStorageStack workIdStorageStack{(*workIdStorageSmem),
                                        (*workIdStorageSmemBarrier),
                                        state.mWarpIdx,
                                        state.mClusterDimX,
                                        state.mClusterDimY,
                                        int32_t{10},
                                        int32_t{-1}};
  // Kernel.cpp:2224
  WorkIdThrottleBarrierSmemBarrier* workIdThrottleBarrierSmemBarrier{
    reinterpret_cast<WorkIdThrottleBarrierSmemBarrier*>(workIdThrottleBarrierSmemBarrierPtr)};
  // Kernel.cpp:2279
  WorkIdThrottleBarrierStack workIdThrottleBarrierStack{(*workIdThrottleBarrierSmemBarrier),
                                                        state.mWarpIdx,
                                                        state.mClusterDimX,
                                                        state.mClusterDimY,
                                                        int32_t{10},
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
                          (*smemPSmem),
                          smemPStack,
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
                        (*smemPSmem),
                        smemPStack,
                        state.mWarpIdx,
                        state.mClusterDimX,
                        state.mClusterDimY,
                        int32_t{4},
                        int32_t{-1}};
  // Kernel.cpp:2279
  TmemCorr0Stack tmemCorr0Stack{(*smemCorrWarpGrpRed1Smem),
                                smemCorrWarpGrpRed1Stack,
                                (*smemOSmem),
                                smemOStack,
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
    loadPageOffsetsTask.execute(params,
                                state,
                                (*smemPageOffsetsKvSmem),
                                smemPageOffsetsKvStack,
                                (*workIdStorageSmem),
                                workIdStorageStack);
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
                       workIdThrottleBarrierStack,
                       (*smemPageOffsetsKvSmem),
                       smemPageOffsetsKvStack,
                       (*workIdStorageSmem),
                       workIdStorageStack);
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
                             tmemS0Stack,
                             (*workIdStorageSmem),
                             workIdStorageStack);
      } else {
        // Kernel.cpp:2010
        if (bool{CorrTask::isSelected(params, state)}) {
          // Kernel.cpp:2077
          CorrTask corrTask{params, state, int32_t{4}};
          // Kernel.cpp:2131
          corrTask.execute(params,
                           state,
                           tmemCorr0Stack,
                           tmemSoftmaxLocal0Stack,
                           tmemOStack,
                           (*workIdStorageSmem),
                           workIdStorageStack);
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
                                    smemKvStack,
                                    (*workIdStorageSmem),
                                    workIdStorageStack);
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
                              tmemP0Stack,
                              (*workIdStorageSmem),
                              workIdStorageStack);
            } else {
              // Kernel.cpp:2010
              if (bool{SchedulerTask::isSelected(params, state)}) {
                // Kernel.cpp:2077
                SchedulerTask schedulerTask{params, state, int32_t{10}};
                // Kernel.cpp:2131
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
}
extern "C" __global__ void
fmhaSm100fKernel_QBfloat16KvE4m3OBfloat16H64PagedKvSlidingOrChunkedCausalP32VarSeqQ8Kv128PersistentSwapsAbForGenGetSmemSize(
  int32_t* outPtr) {
  int32_t size{0};
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemQSmem));
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemKvSmem));
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemTransformedKvSmem));
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemPSmem));
  size = (size + 127) / 128 * 128;
  size += static_cast<int32_t>(sizeof(SmemPageOffsetsKvSmem));
  size = (size + 127) / 128 * 128;
  size += static_cast<int32_t>(sizeof(SmemOSmem));
  size = (size + 15) / 16 * 16;
  size += static_cast<int32_t>(sizeof(SmemSoftmaxWarpGrpRed0Smem));
  size = (size + 15) / 16 * 16;
  size += static_cast<int32_t>(sizeof(SmemSoftmaxWarpGrpRed0Smem));
  size = (size + 15) / 16 * 16;
  size += static_cast<int32_t>(sizeof(SmemCorrWarpGrpRed1Smem));
  size = (size + 15) / 16 * 16;
  size += static_cast<int32_t>(sizeof(WorkIdStorageSmem));
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
