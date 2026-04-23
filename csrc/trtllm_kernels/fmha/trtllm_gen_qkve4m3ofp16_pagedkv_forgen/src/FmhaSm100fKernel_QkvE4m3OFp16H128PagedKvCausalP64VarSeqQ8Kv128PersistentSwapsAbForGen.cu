#include <FmhaSm100fKernel_QkvE4m3OFp16H128PagedKvCausalP64VarSeqQ8Kv128PersistentSwapsAbForGen.h>

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
              int32_t{1024},
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
  trtllm::dev::CutlassTmaUmmaAsyncPipeline<9, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
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
              int32_t{16384},
              bool{cute::elect_one_sync()},
              CuteFlatTuple356{},
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
              CuteFlatTuple635{},
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
    workTileInfo{mScheduler.initial_work_tile_info(CuteFlatTuple831{})} {}
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
  float* mDepSmemPtr7;
  // Res.cpp:595
  trtllm::dev::CutlassUmmaAsyncPipeline<1, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  // TmemS.h:549
  int32_t const mNamedBarId;
  // TmemS.h:552
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
    , // Res.cpp:776
    mPipeline{tmemS0SmemBarrier.mBarriers,
              warpId,
              int32_t{128},
              CuteFlatTuple1061{},
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
  // MemBuffers.cpp:488
  int8_t* mDepSmemPtr5;
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
    mDepSmemPtr5{&smemPSmem.mArray[int32_t{0}]}
    , // Res.cpp:560
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
  // MemBuffers.cpp:488
  int8_t* mDepSmemPtr5;
  // Res.cpp:595
  trtllm::dev::CutlassUmmaAsyncPipeline<2, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
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
    mDepSmemPtr5{&smemPSmem.mArray[int32_t{0}]}
    , // Res.cpp:776
    mPipeline{tmemOSmemBarrier.mBarriers,
              warpId,
              int32_t{128},
              CuteFlatTuple1444{},
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
  // MemBuffers.cpp:488
  float* mDepSmemPtr9;
  // MemBuffers.cpp:488
  int8_t* mDepSmemPtr6;
  // Res.cpp:208
  inline __device__ TmemCorr1Stack(SmemCorrWarpGrpRed1Smem& smemCorrWarpGrpRed1Smem,
                                   SmemCorrWarpGrpRed1Stack& smemCorrWarpGrpRed1Stack,
                                   SmemOSmem& smemOSmem,
                                   SmemOStack& smemOStack,
                                   int32_t warpId,
                                   int32_t clusterDimX,
                                   int32_t clusterDimY,
                                   int32_t barInitWarpId,
                                   int32_t orderedSequenceGroupId)
    : // MemBuffers.cpp:501
    mDepSmemPtr9{&smemCorrWarpGrpRed1Smem.mArray[int32_t{0}]}
    , // MemBuffers.cpp:501
    mDepSmemPtr6{&smemOSmem.mArray[int32_t{0}]} {}
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
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{255})) / (int32_t{256})}, int32_t{1})};
      // FmhaTask.h:630
      if ((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) < (mSeqLenQ)) &&
          ((mCtaIdxKv) < (mNumCtasKv))) {
        // FmhaTask.h:668
        int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{256})) - (int32_t{1}))) /
                         ((mNumCtasKv) * (int32_t{256}))};
        // FmhaTask.h:682
        numLoopSteps = (numSteps) * (int32_t{2});
      } else {
        // FmhaTask.h:648
        numLoopSteps = int32_t{0};
      }
      // SmemPageOffsetsKv.h:204
      int32_t const* ptrPageIdxK4;
      // SmemPageOffsetsKv.h:210
      ptrPageIdxK4 =
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
      int32_t const* ptrPageIdxV4;
      // SmemPageOffsetsKv.h:224
      if (params.mUsesSharedPagedKvIdx) {
        // SmemPageOffsetsKv.h:226
        ptrPageIdxV4 = ptrPageIdxK4;
      } else {
        // SmemPageOffsetsKv.h:228
        ptrPageIdxV4 = ptrPageIdxK4 + (params.mMaxNumPagesPerSeqKv);
      }
      // SmemPageOffsetsKv.h:302
      int32_t pageIdxUb4{(int32_t{((mSeqLenKv) + (int32_t{63})) / (int32_t{64})}) - (int32_t{1})};
      //
      // Loop body.
      //
      // Task.cpp:3392
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset563 = int32_t{0}; loopOffset563 < numLoopSteps;
           loopOffset563 += int32_t{16}) {
        // Task.cpp:3445
        bool const isFirstLoopIter{(loopOffset563) == (int32_t{0})};
        // Task.cpp:3465
        bool const isLastLoopIter{((loopOffset563) + (int32_t{16})) >= (numLoopSteps)};
        //
        // smemPageOffsetsKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset563) >= (int32_t{0})) {
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
          int32_t pageIdx{(((mCtaIdxKv) * (numLoopSteps) + (loopOffset563)) * (int32_t{2})) +
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
          if ((loopOffset563) >= (int32_t{0})) {
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
          int32_t pageIdx{(((mCtaIdxKv) * (numLoopSteps) + (loopOffset563)) * (int32_t{2})) +
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
      9,
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
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{255})) / (int32_t{256})}, int32_t{1})};
      // FmhaTask.h:630
      if ((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) < (mSeqLenQ)) &&
          ((mCtaIdxKv) < (mNumCtasKv))) {
        // FmhaTask.h:668
        int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{256})) - (int32_t{1}))) /
                         ((mNumCtasKv) * (int32_t{256}))};
        // FmhaTask.h:682
        numLoopSteps = (numSteps) * (int32_t{2});
      } else {
        // FmhaTask.h:648
        numLoopSteps = int32_t{0};
      }
      // Task.cpp:3203
      bool const hasOneLoopIter{(int32_t{0}) < (numLoopSteps)};
      // Task.cpp:3214
      int32_t lastLoopOffset{int32_t{0}};
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
        // SmemQ.h:327
        if (bool{cute::elect_one_sync()}) {
          // SmemQ.h:329
          trtllm::dev::completeTransaction(
            barrier,
            ((int32_t{8}) - ((params.mNumTokensPerCtaQ) * (params.mNumHeadsQPerKv))) *
              (int32_t{128}));
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
          coords[int32_t{2}] = mHeadIdx;
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
          coords[int32_t{2}] = mHeadIdx;
          // SmemTile.cpp:492
          coords[int32_t{3}] = localPageOffsets03[int32_t{1}];
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
      // smemPageOffsetsKv [ConsRelease, FirstIter, FreqInfo{0, 16}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:3814

      //
      // Loop body.
      //
      // Task.cpp:3392
      CUTLASS_PRAGMA_NO_UNROLL
      for (int32_t loopOffset846 = int32_t{0}; loopOffset846 < (numLoopSteps) - (int32_t{1});
           ++loopOffset846) {
        // Task.cpp:3465
        bool const isLastLoopIter{((loopOffset846) + (int32_t{1})) >=
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
        if ((((loopOffset846) + (int32_t{1})) % (int32_t{16})) == (int32_t{0})) {
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
        // smemPageOffsetsKv [ConsWork (call 1), Info{0}, FreqInfo{1, 16}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:3814
        if ((((loopOffset846) + (int32_t{1})) % (int32_t{16})) == (int32_t{0})) {
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
          if ((loopOffset846) >= (int32_t{0})) {
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
        // smemKv [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{9}, Flags{0}].
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
             (((loopOffset846) + (int32_t{1})) * (int32_t{2})) % (int32_t{32})))[int32_t{0}];
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
            coords[int32_t{2}] = mHeadIdx;
            // SmemTile.cpp:492
            coords[int32_t{3}] = localPageOffsets03[int32_t{1}];
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
            ((((loopOffset846) + (int32_t{1})) % (int32_t{16})) == (int32_t{15}))) {
          // Task.cpp:2568
          if ((loopOffset846) >= (int32_t{0})) {
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
        // smemPageOffsetsKv [ConsWait, Info{0}, FreqInfo{0, 16}, UserTags{2}, Flags{0}].
        //
        // Task.cpp:3814
        if (((loopOffset846) % (int32_t{16})) == (int32_t{0})) {
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
        // smemPageOffsetsKv [ConsWork (call 2), Info{0}, FreqInfo{0, 16}, UserTags{2}, Flags{0}].
        //
        // Task.cpp:3814
        if (((loopOffset846) % (int32_t{16})) == (int32_t{0})) {
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
        // smemKv [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset846) >= (int32_t{0})) {
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
        // smemKv [ProdWork (call 2), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
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
            (ptrSmemPageOffsetsV3 + ((loopOffset846) * (int32_t{2})) % (int32_t{32})))[int32_t{0}];
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
            coords[int32_t{2}] = mHeadIdx;
            // SmemTile.cpp:492
            coords[int32_t{3}] = localPageOffsets03[int32_t{1}];
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
        // smemPageOffsetsKv [ConsRelease, Info{0}, FreqInfo{0, 16}, UserTags{2}, Flags{0}].
        //
        // Task.cpp:3814
        if (((loopOffset846) % (int32_t{16})) == (int32_t{15})) {
          // Task.cpp:2568
          if ((loopOffset846) >= (int32_t{0})) {
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
        lastLoopOffset = loopOffset846;
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
      // smemPageOffsetsKv [ConsWait, LastIter, FreqInfo{0, 16}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:3814
      if (((lastLoopOffset) % (int32_t{16})) == (int32_t{0})) {
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
      // smemPageOffsetsKv [ConsWork (call 3), LastIter, FreqInfo{0, 16}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:3814
      if (((lastLoopOffset) % (int32_t{16})) == (int32_t{0})) {
        // Task.cpp:1607
        if (hasOneLoopIter) {
          // Task.cpp:5945
          int32_t index{smemPageOffsetsKvConsState.index()};
          // SmemPageOffsetsKv.h:349
          ptrSmemPageOffsetsV4 = smemPageOffsetsKvSrcSmem.mArray[index];
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
      // smemKv [ProdWork (call 3), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
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
          (ptrSmemPageOffsetsV3 + ((lastLoopOffset) * (int32_t{2})) % (int32_t{32})))[int32_t{0}];
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
          coords[int32_t{2}] = mHeadIdx;
          // SmemTile.cpp:492
          coords[int32_t{3}] = localPageOffsets03[int32_t{1}];
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
  // Task.cpp:706
  uint32_t const mTmemBaseOffset;
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
    mLdtm16dp256bitTmemColIdx{
      trtllm::dev::ldst16dp256bitTmemColIdx((mWarpGrpThreadIdx) % (int32_t{128}))}
    , // TmemTile.cpp:453
    mLdtm16dp256bitTmemRowIdx{
      trtllm::dev::ldst16dp256bitTmemRowIdx<int32_t{32}>((mWarpGrpThreadIdx) % (int32_t{128}))}
    , // Kernel.cpp:2424
    mTmemBaseOffset{uint32_t{
      __shfl_sync(uint32_t{0xffffffff}, (*state.mTmemSwStatePtr), int32_t{0}, int32_t{32})}} {}
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
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{255})) / (int32_t{256})}, int32_t{1})};
      // FmhaTask.h:630
      if ((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) < (mSeqLenQ)) &&
          ((mCtaIdxKv) < (mNumCtasKv))) {
        // FmhaTask.h:668
        int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{256})) - (int32_t{1}))) /
                         ((mNumCtasKv) * (int32_t{256}))};
        // FmhaTask.h:682
        numLoopSteps = numSteps;
      } else {
        // FmhaTask.h:648
        numLoopSteps = int32_t{0};
      }
      // Task.cpp:3203
      bool const hasOneLoopIter{(int32_t{0}) < (numLoopSteps)};
      // TmemS.h:654
      float oldMaxArray12[2];
      // TmemS.h:660
      float sumArray12[2]{float{0}, float{0}};
      // TmemS.h:672
      float newMaxArray12[2]{float{-3.4028235e+38}, float{-3.4028235e+38}};
      // TmemTile.cpp:373
      cutlass::Array<float, 8> regsQk;
      // TmemS.h:1361
      uint32_t uint32NegFltMax12{trtllm::dev::floatToUInt32ForAtomicMax(float{-3.4028235e+38})};
      // TmemS.h:1374
      CUTLASS_PRAGMA_UNROLL
      for (int32_t loopOffset1186 = mWarpGrpThreadIdx; loopOffset1186 < int32_t{8};
           loopOffset1186 += int32_t{128}) {
        // TmemS.h:1381
        reinterpret_cast<uint32_t*>(tmemS0SrcStack.mDepSmemPtr7)[loopOffset1186] =
          uint32NegFltMax12;
      }
      // TmemS.h:724
      trtllm::dev::CutlassNamedBarrier::sync(128, tmemS0SrcStack.mNamedBarId);
      // TmemSoftmax.h:515
      cudaGridDependencySynchronize();
      // TmemSoftmax.h:524
      float scaleSoftmaxLog214;
      // TmemSoftmax.h:529
      scaleSoftmaxLog214 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
                             ? (params.mScaleSoftmaxLog2)
                             : (float{params.ptrScaleSoftmaxLog2[int32_t{0}]});
      // TmemP.h:521
      uint32_t regsP[2];
      // TmemP.h:534
      cudaGridDependencySynchronize();
      // TmemP.h:541
      float scaleSoftmaxLog219;
      // TmemP.h:546
      scaleSoftmaxLog219 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
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
      for (int32_t loopOffset1212 = int32_t{0}; loopOffset1212 < numLoopSteps; ++loopOffset1212) {
        // Task.cpp:3465
        bool const isLastLoopIter{((loopOffset1212) + (int32_t{1})) >= (numLoopSteps)};
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
        float* oldMaxPtr12;
        // TmemS.h:1065
        float* sumPtr12;
        // TmemS.h:1070
        float* newMaxPtr012;
        // TmemS.h:1075
        float* qkPtr012;
        // TmemS.h:1080
        float* newMaxPtr112;
        // TmemS.h:1085
        float* qkPtr112;
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{tmemS0ConsState.index()};
          // TmemS.h:1192
          oldMaxPtr12 = oldMaxArray12;
          // TmemS.h:1194
          sumPtr12 = sumArray12;
          // TmemS.h:1196
          newMaxPtr012 = newMaxArray12;
          // TmemS.h:1198
          newMaxPtr112 = newMaxArray12;
          // TmemS.h:1246
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset1233 = int32_t{0}; loopOffset1233 < int32_t{2}; ++loopOffset1233) {
            // TmemS.h:1257
            oldMaxArray12[loopOffset1233] = newMaxArray12[loopOffset1233];
          }
          //
          // The causal mask block.
          //
          // Mask.h:568
          int32_t const tileOffsetK{
            (((numLoopSteps) * (mCtaIdxKv) + (loopOffset1212)) * (int32_t{2}) +
             (tmemS0SrcStack.mInstId)) *
            (int32_t{128})};
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
              uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(regsQk[int32_t{0}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_16x256b(
                dstSlice0,
                (tmemBasePtr) +
                  (static_cast<uint32_t>(int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                           ? (int32_t{0})
                                           : (int32_t{8}))));
              // TmemTile.cpp:545
              uint32_t(&dstSlice1)[4]{reinterpret_cast<uint32_t(&)[4]>(regsQk[int32_t{4}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_16x256b(
                dstSlice1,
                (tmemBasePtr) +
                  (static_cast<uint32_t>((int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                            ? (int32_t{0})
                                            : (int32_t{8})) +
                                         (int32_t{0x100000 /*hi=16, lo=0*/}))));
            }
            // Utils.h:248
            trtllm::dev::reduceColMax16dp256bit<int32_t{2}, int32_t{1}, int32_t{1}, false>(
              newMaxArray12,
              regsQk);
            // Utils.h:260
            trtllm::dev::reduceColMax(newMaxArray12,
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
              uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(regsQk[int32_t{0}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_16x256b(
                dstSlice0,
                (tmemBasePtr) +
                  (static_cast<uint32_t>(int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                           ? (int32_t{0})
                                           : (int32_t{8}))));
              // TmemTile.cpp:545
              uint32_t(&dstSlice1)[4]{reinterpret_cast<uint32_t(&)[4]>(regsQk[int32_t{4}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_16x256b(
                dstSlice1,
                (tmemBasePtr) +
                  (static_cast<uint32_t>((int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                            ? (int32_t{0})
                                            : (int32_t{8})) +
                                         (int32_t{0x100000 /*hi=16, lo=0*/}))));
            }
            //
            // Apply the causal mask.
            //
            // Mask.h:1180
            int32_t const tileOffsetQ{(mCtaIdxQ) * (params.mNumTokensPerCtaQ) +
                                      ((mSeqLenKv) - (mSeqLenQ))};
            // Mask.h:568
            int32_t const tileOffsetK{
              (((numLoopSteps) * (mCtaIdxKv) + (loopOffset1212)) * (int32_t{2}) +
               (tmemS0SrcStack.mInstId)) *
              (int32_t{128})};
            // Mask.h:1206
            int32_t startIdxInWindow0;
            // Mask.h:1206
            int32_t startIdxInWindow1;
            // Mask.h:1225
            int32_t localTokenIdxQ0{(mLdtm16dp256bitTmemColIdx) / (params.mNumHeadsQPerKvDivisor)};
            // Mask.h:1239
            int32_t const idxQ0{min((tileOffsetQ) + (localTokenIdxQ0), (mSeqLenKv) - (int32_t{1}))};
            // Mask.h:1225
            int32_t localTokenIdxQ1{((mLdtm16dp256bitTmemColIdx) + (int32_t{1})) /
                                    (params.mNumHeadsQPerKvDivisor)};
            // Mask.h:1239
            int32_t const idxQ1{min((tileOffsetQ) + (localTokenIdxQ1), (mSeqLenKv) - (int32_t{1}))};
            // Mask.h:1286
            int32_t const idxK0{(tileOffsetK) + (mLdtm16dp256bitTmemRowIdx)};
            // Mask.h:1286
            int32_t const idxK1{(tileOffsetK) + ((mLdtm16dp256bitTmemRowIdx) + (int32_t{8}))};
            // Mask.h:1286
            int32_t const idxK2{(tileOffsetK) + ((mLdtm16dp256bitTmemRowIdx) + (int32_t{16}))};
            // Mask.h:1286
            int32_t const idxK3{(tileOffsetK) + ((mLdtm16dp256bitTmemRowIdx) + (int32_t{24}))};
            // Mask.h:1311
            if ((idxK0) > (idxQ0)) {
              // Mask.h:1315
              regsQk[int32_t{0}] = float{-3.4028235e+38};
            }
            // Mask.h:1311
            if ((idxK1) > (idxQ0)) {
              // Mask.h:1315
              regsQk[int32_t{2}] = float{-3.4028235e+38};
            }
            // Mask.h:1311
            if ((idxK2) > (idxQ0)) {
              // Mask.h:1315
              regsQk[int32_t{4}] = float{-3.4028235e+38};
            }
            // Mask.h:1311
            if ((idxK3) > (idxQ0)) {
              // Mask.h:1315
              regsQk[int32_t{6}] = float{-3.4028235e+38};
            }
            // Mask.h:1311
            if ((idxK0) > (idxQ1)) {
              // Mask.h:1315
              regsQk[int32_t{1}] = float{-3.4028235e+38};
            }
            // Mask.h:1311
            if ((idxK1) > (idxQ1)) {
              // Mask.h:1315
              regsQk[int32_t{3}] = float{-3.4028235e+38};
            }
            // Mask.h:1311
            if ((idxK2) > (idxQ1)) {
              // Mask.h:1315
              regsQk[int32_t{5}] = float{-3.4028235e+38};
            }
            // Mask.h:1311
            if ((idxK3) > (idxQ1)) {
              // Mask.h:1315
              regsQk[int32_t{7}] = float{-3.4028235e+38};
            }
            // Utils.h:248
            trtllm::dev::reduceColMax16dp256bit<int32_t{2}, int32_t{1}, int32_t{1}, false>(
              newMaxArray12,
              regsQk);
            // Utils.h:260
            trtllm::dev::reduceColMax(newMaxArray12,
                                      tmemS0SrcStack.mDepSmemPtr7,
                                      int32_t{128},
                                      mWarpGrpThreadIdx,
                                      tmemS0SrcStack.mNamedBarId);
          }
          // TmemS.h:1334
          qkPtr012 = &regsQk[int32_t{0}];
          // TmemS.h:1336
          qkPtr112 = &regsQk[int32_t{0}];
          // Task.cpp:43
          ++tmemS0ConsState;
        }
        //
        // tmemSoftmaxLocal0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{8}, Flags{0}].
        //
        // TmemSoftmax.h:261
        float* oldMaxPtr13;
        // TmemSoftmax.h:267
        float* sumPtr13;
        // TmemSoftmax.h:273
        float* newMaxPtr13;
        // Task.cpp:1511
        oldMaxPtr13 = oldMaxPtr12;
        // Task.cpp:1511
        sumPtr13 = sumPtr12;
        // Task.cpp:1511
        newMaxPtr13 = newMaxPtr012;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{tmemSoftmaxLocal0ProdState.index()};
          // TmemTile.cpp:373
          cutlass::Array<float, 4> stats;
          // TmemSoftmax.h:365
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset1316 = int32_t{0}; loopOffset1316 < int32_t{2}; ++loopOffset1316) {
            // TmemSoftmax.h:382
            stats[loopOffset1316] = oldMaxPtr13[loopOffset1316];
            // TmemSoftmax.h:384
            stats[(loopOffset1316) + (int32_t{2})] = newMaxPtr13[loopOffset1316];
          }
          // TmemTile.cpp:836
          {
            // TmemTile.cpp:838
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:871
            uint32_t const(&srcSlice0)[4]{
              reinterpret_cast<uint32_t const(&)[4]>(stats[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_st_32x32b(
              (tmemBasePtr) +
                (static_cast<uint32_t>(int32_t((tmemSoftmaxLocal0DstStack.mInstId) == (int32_t{0}))
                                         ? (int32_t{16})
                                         : (int32_t{48}))),
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
        float* newMaxPtr19;
        // TmemP.h:574
        float* regsFp32P19;
        // Task.cpp:1511
        newMaxPtr19 = newMaxPtr112;
        // Task.cpp:1511
        regsFp32P19 = qkPtr112;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{tmemP0ProdState.index()};
          // TmemP.h:1025
          float negScaledMaxArray[2];
          // TmemP.h:1043
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset1344 = int32_t{0}; loopOffset1344 < int32_t{2};
               loopOffset1344 += int32_t{2}) {
            // TmemP.h:1054
            float newMax0{newMaxPtr19[loopOffset1344]};
            // TmemP.h:1060
            float newMax1{newMaxPtr19[(loopOffset1344) + (int32_t{1})]};
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
            // TmemP.h:1079
            float negLog2Scale{-(scaleSoftmaxLog219)};
            // Common.h:353
            cutlass::Array<float, 2> negLog2Scale2{negLog2Scale, negLog2Scale};
            // Common.h:353
            cutlass::Array<float, 2> log2E4m3Scale2{float{8.8073549}, float{8.8073549}};
            // TmemP.h:1104
            newMax2 = trtllm::dev::ffma2(newMax2, negLog2Scale2, log2E4m3Scale2);
            // TmemP.h:1115
            negScaledMaxArray[loopOffset1344] = newMax2[int32_t{0}];
            // TmemP.h:1116
            negScaledMaxArray[(loopOffset1344) + (int32_t{1})] = newMax2[int32_t{1}];
          }
          // TmemP.h:1655
          {
            // TmemP.h:1658
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog219, scaleSoftmaxLog219};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{1}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P19[int32_t{0}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P19[int32_t{1}];
            // TmemP.h:801
            vals[int32_t{0}] =
              (log2Scale2[int32_t{0}]) * (vals[int32_t{0}]) + (negScaledMax[int32_t{0}]);
            // TmemP.h:810
            vals[int32_t{1}] =
              (log2Scale2[int32_t{1}]) * (vals[int32_t{1}]) + (negScaledMax[int32_t{1}]);
            // TmemP.h:833
            regsFp32P19[int32_t{0}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P19[int32_t{1}] = vals[int32_t{1}];
          }
          // TmemP.h:1655
          {
            // TmemP.h:1658
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog219, scaleSoftmaxLog219};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{1}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P19[int32_t{2}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P19[int32_t{3}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P19[int32_t{2}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P19[int32_t{3}] = vals[int32_t{1}];
          }
          // TmemP.h:1454
          tmemP0DstStack.mOrderedSequence.wait();
          // TmemP.h:1773
          regsFp32P19[int32_t{0}] = exp2f(regsFp32P19[int32_t{0}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog219, scaleSoftmaxLog219};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{1}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P19[int32_t{4}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P19[int32_t{5}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P19[int32_t{4}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P19[int32_t{5}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P19[int32_t{1}] = exp2f(regsFp32P19[int32_t{1}]);
          // TmemP.h:1438
          tmemP0DstStack.mOrderedSequence.arrive();
          // TmemP.h:1773
          regsFp32P19[int32_t{2}] = exp2f(regsFp32P19[int32_t{2}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog219, scaleSoftmaxLog219};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{1}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P19[int32_t{6}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P19[int32_t{7}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P19[int32_t{6}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P19[int32_t{7}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P19[int32_t{3}] = exp2f(regsFp32P19[int32_t{3}]);
          // TmemP.h:1773
          regsFp32P19[int32_t{4}] = exp2f(regsFp32P19[int32_t{4}]);
          // TmemP.h:1843
          regsFp32P19[int32_t{5}] = exp2f(regsFp32P19[int32_t{5}]);
          // TmemP.h:1773
          regsFp32P19[int32_t{6}] = exp2f(regsFp32P19[int32_t{6}]);
          // TmemP.h:1843
          regsFp32P19[int32_t{7}] = exp2f(regsFp32P19[int32_t{7}]);
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
            elt0 = regsFp32P19[int32_t{0}];
            // TmemP.h:721
            elt1 = regsFp32P19[int32_t{1}];
            // TmemP.h:722
            elt2 = regsFp32P19[int32_t{2}];
            // TmemP.h:723
            elt3 = regsFp32P19[int32_t{3}];
            // TmemP.h:745
            regsP[int32_t{0}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
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
            elt0 = regsFp32P19[int32_t{4}];
            // TmemP.h:721
            elt1 = regsFp32P19[int32_t{5}];
            // TmemP.h:722
            elt2 = regsFp32P19[int32_t{6}];
            // TmemP.h:723
            elt3 = regsFp32P19[int32_t{7}];
            // TmemP.h:745
            regsP[int32_t{1}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1255
          cutlass::float_e4m3_t* smemPtrP19;
          // TmemP.h:1257
          smemPtrP19 = reinterpret_cast<cutlass::float_e4m3_t*>(tmemP0DstStack.mDepSmemPtr5) +
                       ((tmemP0DstStack.mInstId) * (int32_t{1024}));
          // TmemP.h:1278
          trtllm::dev::storeTransposedSmem8b<int32_t{8}, int32_t{128}>(smemPtrP19,
                                                                       regsP,
                                                                       mWarpGrpThreadIdx);
          // TmemP.h:1280
          cutlass::arch::fence_view_async_shared();
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
          if ((loopOffset1212) >= (int32_t{0})) {
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
        if ((loopOffset1212) >= (int32_t{0})) {
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
        float* oldMaxPtr14;
        // TmemSoftmax.h:552
        float* sumPtr14;
        // TmemSoftmax.h:559
        float* newMaxPtr14;
        // TmemSoftmax.h:566
        float* pPtr14;
        // Task.cpp:1511
        oldMaxPtr14 = oldMaxPtr12;
        // Task.cpp:1511
        sumPtr14 = sumPtr12;
        // Task.cpp:1511
        newMaxPtr14 = newMaxPtr012;
        // Task.cpp:1511
        pPtr14 = qkPtr012;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // TmemSoftmax.h:1010
          {
            // Common.h:395
            cutlass::Array<float, 2> oldMax{float{oldMaxPtr14[int32_t{0}]},
                                            float{oldMaxPtr14[int32_t{1}]}};
            // Common.h:395
            cutlass::Array<float, 2> newMax{float{newMaxPtr14[int32_t{0}]},
                                            float{newMaxPtr14[int32_t{1}]}};
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
              cutlass::Array<float, 2> scale2_{scaleSoftmaxLog214, scaleSoftmaxLog214};
              // Common.h:161
              scale2 = trtllm::dev::fmul2(scale2_, maxDiff2);
              // Common.h:168
              scale2[int32_t{0}] = exp2f(scale2[int32_t{0}]);
              // Common.h:169
              scale2[int32_t{1}] = exp2f(scale2[int32_t{1}]);
            }
            // TmemSoftmax.h:1029
            cutlass::Array<float, 2> p0{pPtr14[int32_t{0}], pPtr14[int32_t{1}]};
            // Common.h:395
            cutlass::Array<float, 2> sum{float{sumPtr14[int32_t{0}]}, float{sumPtr14[int32_t{1}]}};
            // TmemSoftmax.h:1048
            sum = trtllm::dev::ffma2(scale2, sum, p0);
            // TmemSoftmax.h:1060
            cutlass::Array<float, 2> p1{pPtr14[int32_t{2}], pPtr14[int32_t{3}]};
            // TmemSoftmax.h:1076
            sum = trtllm::dev::fadd2(sum, p1);
            // TmemSoftmax.h:1083
            sumPtr14[int32_t{0}] = sum[int32_t{0}];
            // TmemSoftmax.h:1084
            sumPtr14[int32_t{1}] = sum[int32_t{1}];
            // TmemSoftmax.h:1060
            cutlass::Array<float, 2> p2{pPtr14[int32_t{4}], pPtr14[int32_t{5}]};
            // TmemSoftmax.h:1076
            sum = trtllm::dev::fadd2(sum, p2);
            // TmemSoftmax.h:1083
            sumPtr14[int32_t{0}] = sum[int32_t{0}];
            // TmemSoftmax.h:1084
            sumPtr14[int32_t{1}] = sum[int32_t{1}];
            // TmemSoftmax.h:1060
            cutlass::Array<float, 2> p3{pPtr14[int32_t{6}], pPtr14[int32_t{7}]};
            // TmemSoftmax.h:1076
            sum = trtllm::dev::fadd2(sum, p3);
            // TmemSoftmax.h:1083
            sumPtr14[int32_t{0}] = sum[int32_t{0}];
            // TmemSoftmax.h:1084
            sumPtr14[int32_t{1}] = sum[int32_t{1}];
          }
        }
        //
        // tmemSoftmaxLocal0 [ProdWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{10}, Flags{0}].
        //
        // Task.cpp:1511
        oldMaxPtr13 = oldMaxPtr12;
        // Task.cpp:1511
        sumPtr13 = sumPtr12;
        // Task.cpp:1511
        newMaxPtr13 = newMaxPtr012;
        // Task.cpp:1607
        if (isLastLoopIter) {
          // Task.cpp:5945
          int32_t index{tmemSoftmaxLocal0ProdState.index()};
          // TmemTile.cpp:373
          cutlass::Array<float, 4> stats;
          // TmemSoftmax.h:365
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset1508 = int32_t{0}; loopOffset1508 < int32_t{2}; ++loopOffset1508) {
            // TmemSoftmax.h:382
            stats[loopOffset1508] = sumPtr13[loopOffset1508];
            // TmemSoftmax.h:384
            stats[(loopOffset1508) + (int32_t{2})] = newMaxPtr13[loopOffset1508];
          }
          // TmemTile.cpp:836
          {
            // TmemTile.cpp:838
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:871
            uint32_t const(&srcSlice0)[4]{
              reinterpret_cast<uint32_t const(&)[4]>(stats[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_st_32x32b(
              (tmemBasePtr) +
                (static_cast<uint32_t>(int32_t((tmemSoftmaxLocal0DstStack.mInstId) == (int32_t{0}))
                                         ? (int32_t{16})
                                         : (int32_t{48}))),
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
          if ((loopOffset1212) >= (int32_t{0})) {
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
  int32_t const mLdtm16dp256bitTmemColIdx;
  // TmemTile.cpp:445
  int32_t const mLdtm16dp256bitTmemRowIdx;
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
    mLdtm16dp256bitTmemColIdx{
      trtllm::dev::ldst16dp256bitTmemColIdx((mWarpGrpThreadIdx) % (int32_t{128}))}
    , // TmemTile.cpp:453
    mLdtm16dp256bitTmemRowIdx{
      trtllm::dev::ldst16dp256bitTmemRowIdx<int32_t{16}>((mWarpGrpThreadIdx) % (int32_t{128}))} {}
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
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{255})) / (int32_t{256})}, int32_t{1})};
      // FmhaTask.h:630
      if ((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) < (mSeqLenQ)) &&
          ((mCtaIdxKv) < (mNumCtasKv))) {
        // FmhaTask.h:668
        int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{256})) - (int32_t{1}))) /
                         ((mNumCtasKv) * (int32_t{256}))};
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
      cutlass::Array<float, 4> frgStats13;
      // TmemTile.cpp:373
      cutlass::Array<float, 4> frgStats16;
      // TmemCorr.h:1135
      cudaGridDependencySynchronize();
      // TmemCorr.h:1158
      float scaleSoftmaxLog222;
      // TmemCorr.h:1163
      scaleSoftmaxLog222 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
                             ? (params.mScaleSoftmaxLog2)
                             : (float{params.ptrScaleSoftmaxLog2[int32_t{0}]});
      // TmemCorr.h:1135
      cudaGridDependencySynchronize();
      // TmemCorr.h:1158
      float scaleSoftmaxLog223;
      // TmemCorr.h:1163
      scaleSoftmaxLog223 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
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
      for (int32_t loopOffset1676 = int32_t{0}; loopOffset1676 < (numLoopSteps) - (int32_t{1});
           ++loopOffset1676) {
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
        float* statsPtr113;
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
            uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(frgStats13[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_32x32b(
              dstSlice0,
              (tmemBasePtr) +
                (static_cast<uint32_t>(int32_t((tmemSoftmaxLocal0SrcStack.mInstId) == (int32_t{0}))
                                         ? (int32_t{16})
                                         : (int32_t{48}))));
          }
          // TmemSoftmax.h:327
          statsPtr113 = &frgStats13[int32_t{0}];
          // TmemSoftmax.h:330
          cutlass::arch::fence_view_async_tmem_load();
          // Task.cpp:43
          ++tmemSoftmaxLocal0ConsState;
        }
        //
        // tmemSoftmaxLocal0 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
        //
        // Task.cpp:2568
        if ((loopOffset1676) >= (int32_t{0})) {
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
        float* prodStatsPtr022;
        // Task.cpp:1511
        prodStatsPtr022 = statsPtr113;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // TmemCorr.h:425
          cutlass::Array<float, 2> scales22;
          // TmemCorr.h:437
          {
            // Common.h:353
            cutlass::Array<float, 2> oldMax{float{prodStatsPtr022[int32_t{0}]},
                                            float{prodStatsPtr022[int32_t{1}]}};
            // Common.h:353
            cutlass::Array<float, 2> newMax{float{prodStatsPtr022[int32_t{2}]},
                                            float{prodStatsPtr022[int32_t{3}]}};
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
              cutlass::Array<float, 2> scale2_{scaleSoftmaxLog222, scaleSoftmaxLog222};
              // Common.h:161
              scale2 = trtllm::dev::fmul2(scale2_, maxDiff2);
              // Common.h:168
              scale2[int32_t{0}] = exp2f(scale2[int32_t{0}]);
              // Common.h:169
              scale2[int32_t{1}] = exp2f(scale2[int32_t{1}]);
            }
            // TmemCorr.h:473
            scales22[int32_t{0}] = scale2[int32_t{0}];
            // TmemCorr.h:474
            scales22[int32_t{1}] = scale2[int32_t{1}];
          }
          // TmemCorr.h:1240
          bool skipsCorr{true};
          // TmemCorr.h:1258
          skipsCorr = (skipsCorr) && ((scales22[int32_t{0}]) == (float{1}));
          // TmemCorr.h:1258
          skipsCorr = (skipsCorr) && ((scales22[int32_t{1}]) == (float{1}));
          // TmemCorr.h:1266
          skipsCorr = __all_sync(uint32_t{-1}, skipsCorr);
          // TmemCorr.h:1268
          if (!(skipsCorr)) {
            //
            // The headDimStageIdx: 0.
            //
            // TmemCorr.h:1486
            CUTLASS_PRAGMA_UNROLL
            for (int32_t loopOffset1741 = int32_t{0}; loopOffset1741 < int32_t{8};
                 loopOffset1741 += int32_t{8}) {
              // TmemTile.cpp:373
              cutlass::Array<float, 8> tmemRegs022;
              // TmemTile.cpp:527
              {
                // TmemTile.cpp:529
                uint32_t tmemBasePtr{mTmemBaseOffset};
                // TmemTile.cpp:545
                uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(tmemRegs022[int32_t{0}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_ld_16x256b(
                  dstSlice0,
                  (tmemBasePtr) +
                    (static_cast<uint32_t>((int32_t{0x50 /*hi=0, lo=80*/}) + (loopOffset1741))));
                // TmemTile.cpp:545
                uint32_t(&dstSlice1)[4]{reinterpret_cast<uint32_t(&)[4]>(tmemRegs022[int32_t{4}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_ld_16x256b(
                  dstSlice1,
                  (tmemBasePtr) +
                    (static_cast<uint32_t>(((int32_t{0x50 /*hi=0, lo=80*/}) + (loopOffset1741)) +
                                           (int32_t{0x100000 /*hi=16, lo=0*/}))));
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales22[int32_t{0}], scales22[int32_t{1}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs022[int32_t{0}], tmemRegs022[int32_t{1}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs022[int32_t{0}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs022[int32_t{1}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales22[int32_t{0}], scales22[int32_t{1}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs022[int32_t{2}], tmemRegs022[int32_t{3}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs022[int32_t{2}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs022[int32_t{3}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales22[int32_t{0}], scales22[int32_t{1}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs022[int32_t{4}], tmemRegs022[int32_t{5}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs022[int32_t{4}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs022[int32_t{5}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales22[int32_t{0}], scales22[int32_t{1}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs022[int32_t{6}], tmemRegs022[int32_t{7}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs022[int32_t{6}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs022[int32_t{7}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1997
              {
                // TmemTile.cpp:836
                {
                  // TmemTile.cpp:838
                  uint32_t tmemBasePtr{mTmemBaseOffset};
                  // TmemTile.cpp:871
                  uint32_t const(&srcSlice0)[4]{
                    reinterpret_cast<uint32_t const(&)[4]>(tmemRegs022[int32_t{0}])};
                  // CudaPtx.h:48
                  cuda_ptx::tcgen05_st_16x256b(
                    (tmemBasePtr) +
                      (static_cast<uint32_t>((int32_t{0x50 /*hi=0, lo=80*/}) + (loopOffset1741))),
                    srcSlice0);
                  // TmemTile.cpp:871
                  uint32_t const(&srcSlice1)[4]{
                    reinterpret_cast<uint32_t const(&)[4]>(tmemRegs022[int32_t{4}])};
                  // CudaPtx.h:48
                  cuda_ptx::tcgen05_st_16x256b(
                    (tmemBasePtr) +
                      (static_cast<uint32_t>(((int32_t{0x50 /*hi=0, lo=80*/}) + (loopOffset1741)) +
                                             (int32_t{0x100000 /*hi=16, lo=0*/}))),
                    srcSlice1);
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
        if ((loopOffset1676) >= (int32_t{0})) {
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
        float* statsPtr116;
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
            uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(frgStats16[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_32x32b(
              dstSlice0,
              (tmemBasePtr) +
                (static_cast<uint32_t>(int32_t((tmemSoftmaxLocal1SrcStack.mInstId) == (int32_t{0}))
                                         ? (int32_t{16})
                                         : (int32_t{48}))));
          }
          // TmemSoftmax.h:327
          statsPtr116 = &frgStats16[int32_t{0}];
          // TmemSoftmax.h:330
          cutlass::arch::fence_view_async_tmem_load();
          // Task.cpp:43
          ++tmemSoftmaxLocal1ConsState;
        }
        //
        // tmemSoftmaxLocal1 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
        //
        // Task.cpp:2568
        if ((loopOffset1676) >= (int32_t{0})) {
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
        float* prodStatsPtr023;
        // Task.cpp:1511
        prodStatsPtr023 = statsPtr116;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // TmemCorr.h:425
          cutlass::Array<float, 2> scales23;
          // TmemCorr.h:437
          {
            // Common.h:353
            cutlass::Array<float, 2> oldMax{float{prodStatsPtr023[int32_t{0}]},
                                            float{prodStatsPtr023[int32_t{1}]}};
            // Common.h:353
            cutlass::Array<float, 2> newMax{float{prodStatsPtr023[int32_t{2}]},
                                            float{prodStatsPtr023[int32_t{3}]}};
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
              cutlass::Array<float, 2> scale2_{scaleSoftmaxLog223, scaleSoftmaxLog223};
              // Common.h:161
              scale2 = trtllm::dev::fmul2(scale2_, maxDiff2);
              // Common.h:168
              scale2[int32_t{0}] = exp2f(scale2[int32_t{0}]);
              // Common.h:169
              scale2[int32_t{1}] = exp2f(scale2[int32_t{1}]);
            }
            // TmemCorr.h:473
            scales23[int32_t{0}] = scale2[int32_t{0}];
            // TmemCorr.h:474
            scales23[int32_t{1}] = scale2[int32_t{1}];
          }
          // TmemCorr.h:1240
          bool skipsCorr{true};
          // TmemCorr.h:1258
          skipsCorr = (skipsCorr) && ((scales23[int32_t{0}]) == (float{1}));
          // TmemCorr.h:1258
          skipsCorr = (skipsCorr) && ((scales23[int32_t{1}]) == (float{1}));
          // TmemCorr.h:1266
          skipsCorr = __all_sync(uint32_t{-1}, skipsCorr);
          // TmemCorr.h:1268
          if (!(skipsCorr)) {
            //
            // The headDimStageIdx: 0.
            //
            // TmemCorr.h:1486
            CUTLASS_PRAGMA_UNROLL
            for (int32_t loopOffset1854 = int32_t{0}; loopOffset1854 < int32_t{8};
                 loopOffset1854 += int32_t{8}) {
              // TmemTile.cpp:373
              cutlass::Array<float, 8> tmemRegs023;
              // TmemTile.cpp:527
              {
                // TmemTile.cpp:529
                uint32_t tmemBasePtr{mTmemBaseOffset};
                // TmemTile.cpp:545
                uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(tmemRegs023[int32_t{0}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_ld_16x256b(
                  dstSlice0,
                  (tmemBasePtr) +
                    (static_cast<uint32_t>((int32_t{0x58 /*hi=0, lo=88*/}) + (loopOffset1854))));
                // TmemTile.cpp:545
                uint32_t(&dstSlice1)[4]{reinterpret_cast<uint32_t(&)[4]>(tmemRegs023[int32_t{4}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_ld_16x256b(
                  dstSlice1,
                  (tmemBasePtr) +
                    (static_cast<uint32_t>(((int32_t{0x58 /*hi=0, lo=88*/}) + (loopOffset1854)) +
                                           (int32_t{0x100000 /*hi=16, lo=0*/}))));
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales23[int32_t{0}], scales23[int32_t{1}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs023[int32_t{0}], tmemRegs023[int32_t{1}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs023[int32_t{0}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs023[int32_t{1}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales23[int32_t{0}], scales23[int32_t{1}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs023[int32_t{2}], tmemRegs023[int32_t{3}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs023[int32_t{2}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs023[int32_t{3}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales23[int32_t{0}], scales23[int32_t{1}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs023[int32_t{4}], tmemRegs023[int32_t{5}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs023[int32_t{4}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs023[int32_t{5}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1534
              {
                // TmemCorr.h:1554
                cutlass::Array<float, 2> localScales0{scales23[int32_t{0}], scales23[int32_t{1}]};
                // TmemCorr.h:1565
                cutlass::Array<float, 2> vals0{tmemRegs023[int32_t{6}], tmemRegs023[int32_t{7}]};
                // TmemCorr.h:1577
                vals0 = trtllm::dev::fmul2(vals0, localScales0);
                // TmemCorr.h:1580
                tmemRegs023[int32_t{6}] = vals0[int32_t{0}];
                // TmemCorr.h:1581
                tmemRegs023[int32_t{7}] = vals0[int32_t{1}];
              }
              // TmemCorr.h:1997
              {
                // TmemTile.cpp:836
                {
                  // TmemTile.cpp:838
                  uint32_t tmemBasePtr{mTmemBaseOffset};
                  // TmemTile.cpp:871
                  uint32_t const(&srcSlice0)[4]{
                    reinterpret_cast<uint32_t const(&)[4]>(tmemRegs023[int32_t{0}])};
                  // CudaPtx.h:48
                  cuda_ptx::tcgen05_st_16x256b(
                    (tmemBasePtr) +
                      (static_cast<uint32_t>((int32_t{0x58 /*hi=0, lo=88*/}) + (loopOffset1854))),
                    srcSlice0);
                  // TmemTile.cpp:871
                  uint32_t const(&srcSlice1)[4]{
                    reinterpret_cast<uint32_t const(&)[4]>(tmemRegs023[int32_t{4}])};
                  // CudaPtx.h:48
                  cuda_ptx::tcgen05_st_16x256b(
                    (tmemBasePtr) +
                      (static_cast<uint32_t>(((int32_t{0x58 /*hi=0, lo=88*/}) + (loopOffset1854)) +
                                             (int32_t{0x100000 /*hi=16, lo=0*/}))),
                    srcSlice1);
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
        if ((loopOffset1676) >= (int32_t{0})) {
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
        lastLoopOffset = loopOffset1676;
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
      float* statsPtr213;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemSoftmaxLocal0ConsState.index()};
        // TmemTile.cpp:527
        {
          // TmemTile.cpp:529
          uint32_t tmemBasePtr{mTmemBaseOffset};
          // TmemTile.cpp:545
          uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(frgStats13[int32_t{0}])};
          // CudaPtx.h:48
          cuda_ptx::tcgen05_ld_32x32b(
            dstSlice0,
            (tmemBasePtr) +
              (static_cast<uint32_t>(int32_t((tmemSoftmaxLocal0SrcStack.mInstId) == (int32_t{0}))
                                       ? (int32_t{16})
                                       : (int32_t{48}))));
        }
        // TmemSoftmax.h:327
        statsPtr213 = &frgStats13[int32_t{0}];
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
      // Task.cpp:1607
      if (hasOneLoopIter) {
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
      float* statsPtr216;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemSoftmaxLocal1ConsState.index()};
        // TmemTile.cpp:527
        {
          // TmemTile.cpp:529
          uint32_t tmemBasePtr{mTmemBaseOffset};
          // TmemTile.cpp:545
          uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(frgStats16[int32_t{0}])};
          // CudaPtx.h:48
          cuda_ptx::tcgen05_ld_32x32b(
            dstSlice0,
            (tmemBasePtr) +
              (static_cast<uint32_t>(int32_t((tmemSoftmaxLocal1SrcStack.mInstId) == (int32_t{0}))
                                       ? (int32_t{16})
                                       : (int32_t{48}))));
        }
        // TmemSoftmax.h:327
        statsPtr216 = &frgStats16[int32_t{0}];
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
      // TmemCorr.h:2344
      float* finalProdStatsPtr023;
      // TmemCorr.h:2349
      float* finalProdStatsPtr123;
      // Task.cpp:1511
      finalProdStatsPtr023 = statsPtr213;
      // Task.cpp:1511
      finalProdStatsPtr123 = statsPtr216;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // TmemCorr.h:2860
        int32_t instIdxO{(mCtaIdxQ) * (int32_t{1})};
        // TmemCorr.h:2907
        int32_t numValidRowsO{
          (int32_t{min(params.mNumTokensPerCtaQ,
                       (mSeqLenQ) - ((instIdxO) * (params.mNumTokensPerCtaQ)))}) *
          (params.mNumHeadsQPerKv)};
        // TmemCorr.h:2984
        int32_t seqOffsetO{(mSeqOffsetQ) + ((instIdxO) * (params.mNumTokensPerCtaQ))};
        // TmemCorr.h:2989
        int32_t headIdxO;
        // TmemCorr.h:2993
        headIdxO = (mHeadIdx) * (int32_t{min(params.mNumHeadsQPerKv, int32_t{8})});
        // TmemCorr.h:2996
        int32_t headOffsetO{((mHeadIdx) * (int32_t{min(params.mNumHeadsQPerKv, int32_t{8})})) *
                            (int32_t{128})};
        // TmemCorr.h:3027
        int64_t ctaOffsetO{(static_cast<int64_t>(seqOffsetO)) *
                             (static_cast<int64_t>((params.mNumHeadsQ) * (int32_t{128}))) +
                           (static_cast<int64_t>(headOffsetO))};
        // TmemCorr.h:3041
        cutlass::half_t* ptrO{reinterpret_cast<cutlass::half_t*>(params.ptrO)};
        // TmemCorr.h:3046
        ptrO = ptrO + (ctaOffsetO);
        // TmemCorr.h:3076
        bool storesSoftmaxStats{reinterpret_cast<float*>(params.ptrSoftmaxStats) != nullptr};
        // TmemCorr.h:3082
        float* ptrSoftmaxStats;
        // TmemCorr.h:3084
        if (storesSoftmaxStats) {
          // TmemCorr.h:3088
          ptrSoftmaxStats = reinterpret_cast<float*>(params.ptrSoftmaxStats) +
                            (((seqOffsetO) * (params.mNumHeadsQ) +
                              ((mHeadIdx) * (int32_t{min(params.mNumHeadsQPerKv, int32_t{8})}))) *
                             (int32_t{2}));
        }
        // TmemCorr.h:804
        float scales23[2][2];
        // TmemCorr.h:824
        {
          // Common.h:353
          cutlass::Array<float, 2> sums0{float{finalProdStatsPtr023[int32_t{0}]},
                                         float{finalProdStatsPtr023[int32_t{1}]}};
          // Common.h:353
          cutlass::Array<float, 2> sums1{float{finalProdStatsPtr123[int32_t{0}]},
                                         float{finalProdStatsPtr123[int32_t{1}]}};
          // Common.h:353
          cutlass::Array<float, 2> maxs0{float{finalProdStatsPtr023[int32_t{2}]},
                                         float{finalProdStatsPtr023[int32_t{3}]}};
          // Common.h:353
          cutlass::Array<float, 2> maxs1{float{finalProdStatsPtr123[int32_t{2}]},
                                         float{finalProdStatsPtr123[int32_t{3}]}};
          // TmemCorr.h:862
          float max0{fmaxf(float{finalProdStatsPtr023[int32_t{2}]},
                           float{finalProdStatsPtr123[int32_t{2}]})};
          // TmemCorr.h:864
          float max1{fmaxf(float{finalProdStatsPtr023[int32_t{3}]},
                           float{finalProdStatsPtr123[int32_t{3}]})};
          // Common.h:353
          cutlass::Array<float, 2> maxs{max0, max1};
          // Common.h:353
          cutlass::Array<float, 2> scalesI0{float{1}, float{1}};
          // Common.h:353
          cutlass::Array<float, 2> scalesI1{float{1}, float{1}};
          // Common.h:198
          maxs[int32_t{0}] = -(maxs[int32_t{0}]);
          // Common.h:199
          maxs[int32_t{1}] = -(maxs[int32_t{1}]);
          // Common.h:202
          cutlass::Array<float, 2> maxDiffI0{trtllm::dev::fadd2(maxs0, maxs)};
          // Common.h:203
          cutlass::Array<float, 2> maxDiffI1{trtllm::dev::fadd2(maxs1, maxs)};
          // Common.h:230
          if (((((maxDiffI0[int32_t{0}]) != (float{0})) ||
                ((maxDiffI0[int32_t{1}]) != (float{0}))) ||
               ((maxDiffI1[int32_t{0}]) != (float{0}))) ||
              ((maxDiffI1[int32_t{1}]) != (float{0}))) {
            // Common.h:353
            cutlass::Array<float, 2> scales2_{scaleSoftmaxLog223, scaleSoftmaxLog223};
            // Common.h:237
            scalesI0 = trtllm::dev::fmul2(scales2_, maxDiffI0);
            // Common.h:238
            scalesI1 = trtllm::dev::fmul2(scales2_, maxDiffI1);
            // Common.h:247
            scalesI0[int32_t{0}] = exp2f(scalesI0[int32_t{0}]);
            // Common.h:248
            scalesI0[int32_t{1}] = exp2f(scalesI0[int32_t{1}]);
            // Common.h:249
            scalesI1[int32_t{0}] = exp2f(scalesI1[int32_t{0}]);
            // Common.h:250
            scalesI1[int32_t{1}] = exp2f(scalesI1[int32_t{1}]);
          }
          // TmemCorr.h:880
          cutlass::Array<float, 2> sums;
          // TmemCorr.h:883
          sums = trtllm::dev::ffma2(scalesI0, sums0, trtllm::dev::fmul2(scalesI1, sums1));
          // TmemCorr.h:893
          finalProdStatsPtr023[int32_t{0}] = sums[int32_t{0}];
          // TmemCorr.h:894
          finalProdStatsPtr023[int32_t{1}] = sums[int32_t{1}];
          // TmemCorr.h:896
          finalProdStatsPtr023[int32_t{2}] = max0;
          // TmemCorr.h:897
          finalProdStatsPtr023[int32_t{3}] = max1;
          // TmemCorr.h:914
          scales23[int32_t{0}][int32_t{0}] = scalesI0[int32_t{0}];
          // TmemCorr.h:915
          scales23[int32_t{0}][int32_t{1}] = scalesI0[int32_t{1}];
          // TmemCorr.h:916
          scales23[int32_t{1}][int32_t{0}] = scalesI1[int32_t{0}];
          // TmemCorr.h:917
          scales23[int32_t{1}][int32_t{1}] = scalesI1[int32_t{1}];
        }
        // TmemCorr.h:1983
        trtllm::dev::reduceColSum<int32_t{2}>(finalProdStatsPtr023,
                                              tmemCorr1DstStack.mDepSmemPtr9,
                                              int32_t{4},
                                              int32_t{128},
                                              mWarpGrpThreadIdx,
                                              int32_t{4});
        // TmemCorr.h:933
        {
          // TmemCorr.h:942
          float sum0{finalProdStatsPtr023[int32_t{0}]};
          // TmemCorr.h:948
          float sum1{finalProdStatsPtr023[int32_t{1}]};
          // TmemCorr.h:955
          float max0{finalProdStatsPtr023[int32_t{2}]};
          // TmemCorr.h:961
          float max1{finalProdStatsPtr023[int32_t{3}]};
          // TmemCorr.h:1904
          float attentionSinkVal0{int32_t{0}};
          // TmemCorr.h:1907
          float attentionSinkVal1{int32_t{0}};
          // TmemCorr.h:1912
          if (bool{params.ptrAttentionSinks != nullptr}) {
            // TmemCorr.h:1937
            attentionSinkVal0 =
              (float{exp2f(((float{params.ptrAttentionSinks[int32_t{min(
                              (headIdxO) + ((mLdtm16dp256bitTmemColIdx) % (params.mNumHeadsQPerKv)),
                              (params.mNumHeadsQ) - (int32_t{1}))}]}) *
                            (float{1.442695})) -
                           ((max0) * (scaleSoftmaxLog223)))}) *
              (float{448});
            // TmemCorr.h:1938
            attentionSinkVal1 =
              (float{exp2f(((float{params.ptrAttentionSinks[int32_t{
                              min((headIdxO) + (((mLdtm16dp256bitTmemColIdx) + (int32_t{1})) %
                                                (params.mNumHeadsQPerKv)),
                                  (params.mNumHeadsQ) - (int32_t{1}))}]}) *
                            (float{1.442695})) -
                           ((max1) * (scaleSoftmaxLog223)))}) *
              (float{448});
          }
          // TmemCorr.h:1023
          finalProdStatsPtr023[int32_t{0}] = (sum0) + (attentionSinkVal0);
          // TmemCorr.h:1024
          finalProdStatsPtr023[int32_t{1}] = (sum1) + (attentionSinkVal1);
          // Common.h:353
          cutlass::Array<float, 2> normScales{(float(bool{params.ptrOutputScale == nullptr})
                                                 ? (params.mOutputScale)
                                                 : (float{params.ptrOutputScale[int32_t{0}]})) /
                                                ((sum0) + (attentionSinkVal0)),
                                              (float(bool{params.ptrOutputScale == nullptr})
                                                 ? (params.mOutputScale)
                                                 : (float{params.ptrOutputScale[int32_t{0}]})) /
                                                ((sum1) + (attentionSinkVal1))};
          // Common.h:377
          cutlass::Array<float, 2> expScales0{scales23[int32_t{0}][int32_t{0}],
                                              scales23[int32_t{0}][int32_t{1}]};
          // Common.h:377
          cutlass::Array<float, 2> expScales1{scales23[int32_t{1}][int32_t{0}],
                                              scales23[int32_t{1}][int32_t{1}]};
          // TmemCorr.h:1051
          cutlass::Array<float, 2> finalScales0;
          // TmemCorr.h:1054
          finalScales0 = trtllm::dev::fmul2(normScales, expScales0);
          // TmemCorr.h:1064
          scales23[int32_t{0}][int32_t{0}] = finalScales0[int32_t{0}];
          // TmemCorr.h:1065
          scales23[int32_t{0}][int32_t{1}] = finalScales0[int32_t{1}];
          // TmemCorr.h:1069
          cutlass::Array<float, 2> finalScales1;
          // TmemCorr.h:1072
          finalScales1 = trtllm::dev::fmul2(normScales, expScales1);
          // TmemCorr.h:1082
          scales23[int32_t{1}][int32_t{0}] = finalScales1[int32_t{0}];
          // TmemCorr.h:1083
          scales23[int32_t{1}][int32_t{1}] = finalScales1[int32_t{1}];
        }
        // TmemCorr.h:3898
        if (storesSoftmaxStats) {
          // TmemCorr.h:3902
          float scaleSoftmax{(scaleSoftmaxLog223) * (float{0.69314718})};
          // TmemCorr.h:3915
          (finalProdStatsPtr023 + int32_t{2})[int32_t{0}] =
            (float{(finalProdStatsPtr023 + int32_t{2})[int32_t{0}]}) * (scaleSoftmax);
          // TmemCorr.h:3926
          finalProdStatsPtr023[int32_t{0}] =
            (float{finalProdStatsPtr023[int32_t{0}]}) * (float{0.002232143});
          // TmemCorr.h:3915
          (finalProdStatsPtr023 + int32_t{2})[int32_t{1}] =
            (float{(finalProdStatsPtr023 + int32_t{2})[int32_t{1}]}) * (scaleSoftmax);
          // TmemCorr.h:3926
          finalProdStatsPtr023[int32_t{1}] =
            (float{finalProdStatsPtr023[int32_t{1}]}) * (float{0.002232143});
          // TmemCorr.h:3981
          trtllm::dev::storeStatsForSwappedAb<int32_t{2}, true>((finalProdStatsPtr023 + int32_t{2}),
                                                                finalProdStatsPtr023,
                                                                ptrSoftmaxStats,
                                                                params.mNumHeadsQ,
                                                                params.mNumHeadsQPerKv,
                                                                mWarpGrpThreadIdx,
                                                                numValidRowsO);
        }
        // TmemCorr.h:1674
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset2100 = int32_t{0}; loopOffset2100 < int32_t{8};
             loopOffset2100 += int32_t{8}) {
          // TmemTile.cpp:373
          cutlass::Array<float, 8> tmemRegs023;
          // TmemTile.cpp:527
          {
            // TmemTile.cpp:529
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:545
            uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(tmemRegs023[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_16x256b(
              dstSlice0,
              (tmemBasePtr) + (static_cast<uint32_t>((int32_t{80}) + (loopOffset2100))));
            // TmemTile.cpp:545
            uint32_t(&dstSlice1)[4]{reinterpret_cast<uint32_t(&)[4]>(tmemRegs023[int32_t{4}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_16x256b(
              dstSlice1,
              (tmemBasePtr) + (static_cast<uint32_t>(((int32_t{80}) + (loopOffset2100)) +
                                                     (int32_t{0x100000 /*hi=16, lo=0*/}))));
          }
          // TmemTile.cpp:373
          cutlass::Array<float, 8> tmemRegs123;
          // TmemTile.cpp:527
          {
            // TmemTile.cpp:529
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:545
            uint32_t(&dstSlice0)[4]{reinterpret_cast<uint32_t(&)[4]>(tmemRegs123[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_16x256b(
              dstSlice0,
              (tmemBasePtr) + (static_cast<uint32_t>((int32_t{88}) + (loopOffset2100))));
            // TmemTile.cpp:545
            uint32_t(&dstSlice1)[4]{reinterpret_cast<uint32_t(&)[4]>(tmemRegs123[int32_t{4}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_16x256b(
              dstSlice1,
              (tmemBasePtr) + (static_cast<uint32_t>(((int32_t{88}) + (loopOffset2100)) +
                                                     (int32_t{0x100000 /*hi=16, lo=0*/}))));
          }
          // TmemCorr.h:3438
          uint32_t mRegsO23[4];
          // TmemCorr.h:1715
          {
            // TmemCorr.h:1725
            cutlass::Array<float, 2> vals023;
            // TmemCorr.h:1730
            cutlass::Array<float, 2> vals123;
            // TmemCorr.h:1763
            vals023[int32_t{0}] = tmemRegs023[int32_t{0}];
            // TmemCorr.h:1764
            vals023[int32_t{1}] = tmemRegs023[int32_t{1}];
            // TmemCorr.h:1765
            vals123[int32_t{0}] = tmemRegs123[int32_t{0}];
            // TmemCorr.h:1766
            vals123[int32_t{1}] = tmemRegs123[int32_t{1}];
            // Common.h:377
            cutlass::Array<float, 2> scales0_0{scales23[int32_t{0}][int32_t{0}],
                                               scales23[int32_t{0}][int32_t{1}]};
            // Common.h:377
            cutlass::Array<float, 2> scales1_0{scales23[int32_t{1}][int32_t{0}],
                                               scales23[int32_t{1}][int32_t{1}]};
            // TmemCorr.h:1787
            vals023 =
              trtllm::dev::ffma2(vals023, scales0_0, trtllm::dev::fmul2(vals123, scales1_0));
            // TmemCorr.h:1793
            tmemRegs023[int32_t{0}] = vals023[int32_t{0}];
            // TmemCorr.h:1794
            tmemRegs023[int32_t{1}] = vals023[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO23[int32_t{0}] =
              trtllm::dev::convert_float2_to_half(tmemRegs023[int32_t{0}], tmemRegs023[int32_t{1}]);
          }
          // TmemCorr.h:1715
          {
            // TmemCorr.h:1725
            cutlass::Array<float, 2> vals023;
            // TmemCorr.h:1730
            cutlass::Array<float, 2> vals123;
            // TmemCorr.h:1763
            vals023[int32_t{0}] = tmemRegs023[int32_t{2}];
            // TmemCorr.h:1764
            vals023[int32_t{1}] = tmemRegs023[int32_t{3}];
            // TmemCorr.h:1765
            vals123[int32_t{0}] = tmemRegs123[int32_t{2}];
            // TmemCorr.h:1766
            vals123[int32_t{1}] = tmemRegs123[int32_t{3}];
            // Common.h:377
            cutlass::Array<float, 2> scales0_0{scales23[int32_t{0}][int32_t{0}],
                                               scales23[int32_t{0}][int32_t{1}]};
            // Common.h:377
            cutlass::Array<float, 2> scales1_0{scales23[int32_t{1}][int32_t{0}],
                                               scales23[int32_t{1}][int32_t{1}]};
            // TmemCorr.h:1787
            vals023 =
              trtllm::dev::ffma2(vals023, scales0_0, trtllm::dev::fmul2(vals123, scales1_0));
            // TmemCorr.h:1793
            tmemRegs023[int32_t{2}] = vals023[int32_t{0}];
            // TmemCorr.h:1794
            tmemRegs023[int32_t{3}] = vals023[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO23[int32_t{1}] =
              trtllm::dev::convert_float2_to_half(tmemRegs023[int32_t{2}], tmemRegs023[int32_t{3}]);
          }
          // TmemCorr.h:1715
          {
            // TmemCorr.h:1725
            cutlass::Array<float, 2> vals023;
            // TmemCorr.h:1730
            cutlass::Array<float, 2> vals123;
            // TmemCorr.h:1763
            vals023[int32_t{0}] = tmemRegs023[int32_t{4}];
            // TmemCorr.h:1764
            vals023[int32_t{1}] = tmemRegs023[int32_t{5}];
            // TmemCorr.h:1765
            vals123[int32_t{0}] = tmemRegs123[int32_t{4}];
            // TmemCorr.h:1766
            vals123[int32_t{1}] = tmemRegs123[int32_t{5}];
            // Common.h:377
            cutlass::Array<float, 2> scales0_0{scales23[int32_t{0}][int32_t{0}],
                                               scales23[int32_t{0}][int32_t{1}]};
            // Common.h:377
            cutlass::Array<float, 2> scales1_0{scales23[int32_t{1}][int32_t{0}],
                                               scales23[int32_t{1}][int32_t{1}]};
            // TmemCorr.h:1787
            vals023 =
              trtllm::dev::ffma2(vals023, scales0_0, trtllm::dev::fmul2(vals123, scales1_0));
            // TmemCorr.h:1793
            tmemRegs023[int32_t{4}] = vals023[int32_t{0}];
            // TmemCorr.h:1794
            tmemRegs023[int32_t{5}] = vals023[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO23[int32_t{2}] =
              trtllm::dev::convert_float2_to_half(tmemRegs023[int32_t{4}], tmemRegs023[int32_t{5}]);
          }
          // TmemCorr.h:1715
          {
            // TmemCorr.h:1725
            cutlass::Array<float, 2> vals023;
            // TmemCorr.h:1730
            cutlass::Array<float, 2> vals123;
            // TmemCorr.h:1763
            vals023[int32_t{0}] = tmemRegs023[int32_t{6}];
            // TmemCorr.h:1764
            vals023[int32_t{1}] = tmemRegs023[int32_t{7}];
            // TmemCorr.h:1765
            vals123[int32_t{0}] = tmemRegs123[int32_t{6}];
            // TmemCorr.h:1766
            vals123[int32_t{1}] = tmemRegs123[int32_t{7}];
            // Common.h:377
            cutlass::Array<float, 2> scales0_0{scales23[int32_t{0}][int32_t{0}],
                                               scales23[int32_t{0}][int32_t{1}]};
            // Common.h:377
            cutlass::Array<float, 2> scales1_0{scales23[int32_t{1}][int32_t{0}],
                                               scales23[int32_t{1}][int32_t{1}]};
            // TmemCorr.h:1787
            vals023 =
              trtllm::dev::ffma2(vals023, scales0_0, trtllm::dev::fmul2(vals123, scales1_0));
            // TmemCorr.h:1793
            tmemRegs023[int32_t{6}] = vals023[int32_t{0}];
            // TmemCorr.h:1794
            tmemRegs023[int32_t{7}] = vals023[int32_t{1}];
            // TmemCorr.h:3664
            mRegsO23[int32_t{3}] =
              trtllm::dev::convert_float2_to_half(tmemRegs023[int32_t{6}], tmemRegs023[int32_t{7}]);
          }
          // TmemCorr.h:3849
          trtllm::dev::reorganizeInSmemAndStoreToDstMem<int32_t{128}, int32_t{8}, true>(
            reinterpret_cast<cutlass::half_t*>(tmemCorr1DstStack.mDepSmemPtr6),
            ptrO,
            mRegsO23,
            int32_t{128},
            numValidRowsO,
            params.mNumHeadsQ,
            params.mNumHeadsQPerKvDivisor,
            int32_t{128},
            mWarpGrpThreadIdx,
            int32_t{3});
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
      9,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemKvConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      9,
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
        int32_t{min(int32_t{((validSeqLenKv) + (int32_t{255})) / (int32_t{256})}, int32_t{1})};
      // FmhaTask.h:630
      if ((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) < (mSeqLenQ)) &&
          ((mCtaIdxKv) < (mNumCtasKv))) {
        // FmhaTask.h:668
        int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{256})) - (int32_t{1}))) /
                         ((mNumCtasKv) * (int32_t{256}))};
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
      // smemKv [ConsWait, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
        // SmemKv.h:304
        smemIdxK3 = index;
        // Task.cpp:43
        ++smemKvConsState;
      }
      //
      // tmemS0 [ProdWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // TmemS.h:1130
      cutlass::float_e4m3_t* smemPtrQ12;
      // TmemS.h:1135
      int32_t smemIdxQ12;
      // TmemS.h:1141
      cutlass::float_e4m3_t* smemPtrK12;
      // TmemS.h:1146
      int32_t memIdxK12;
      // Task.cpp:1511
      smemPtrQ12 = smemPtrQ0_2;
      // Task.cpp:1511
      smemIdxQ12 = smemIdxQ0_2;
      // Task.cpp:1511
      smemPtrK12 = smemPtrK3;
      // Task.cpp:1511
      memIdxK12 = smemIdxK3;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemS0ProdState.index()};
        // TmemS.h:1889
        cutlass::float_e4m3_t* smemQ{smemPtrQ12};
        // TmemS.h:1910
        smemQ += (smemIdxQ12) * (int32_t{1024});
        // TmemS.h:1938
        cutlass::float_e4m3_t* smemK{smemPtrK12};
        // TmemS.h:1944
        smemK += (memIdxK12) * (int32_t{16384});
        // Mma.cpp:618
        {
          // TmemTile.cpp:1765
          uint32_t tmemPtrD{int32_t((tmemS0DstStack.mInstId) == (int32_t{0})) ? (int32_t{0})
                                                                              : (int32_t{8})};
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
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{8},
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
                                                                  int32_t{8},
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
          //
          // MMA inst for mi=0 ni=0 ki=2.
          //
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{8},
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
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{8},
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
                                  utcmmaDesc_0_0_3,
                                  bool{true});
          }
        }
      }
      //
      // smemKv [ConsRelease, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
      // tmemS1 [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{64}].
      //
      //
      // smemKv [ConsWait, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
        smemPtrK3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:304
        smemIdxK3 = index;
        // Task.cpp:43
        ++smemKvConsState;
      }
      //
      // tmemS1 [ProdWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // TmemS.h:1130
      cutlass::float_e4m3_t* smemPtrQ15;
      // TmemS.h:1135
      int32_t smemIdxQ15;
      // TmemS.h:1141
      cutlass::float_e4m3_t* smemPtrK15;
      // TmemS.h:1146
      int32_t memIdxK15;
      // Task.cpp:1511
      smemPtrQ15 = smemPtrQ0_2;
      // Task.cpp:1511
      smemIdxQ15 = smemIdxQ0_2;
      // Task.cpp:1511
      smemPtrK15 = smemPtrK3;
      // Task.cpp:1511
      memIdxK15 = smemIdxK3;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemS1ProdState.index()};
        // TmemS.h:1889
        cutlass::float_e4m3_t* smemQ{smemPtrQ15};
        // TmemS.h:1910
        smemQ += (smemIdxQ15) * (int32_t{1024});
        // TmemS.h:1938
        cutlass::float_e4m3_t* smemK{smemPtrK15};
        // TmemS.h:1944
        smemK += (memIdxK15) * (int32_t{16384});
        // Mma.cpp:618
        {
          // TmemTile.cpp:1765
          uint32_t tmemPtrD{int32_t((tmemS1DstStack.mInstId) == (int32_t{0})) ? (int32_t{0})
                                                                              : (int32_t{8})};
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
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{8},
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
                                                                  int32_t{8},
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
          //
          // MMA inst for mi=0 ni=0 ki=2.
          //
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{8},
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
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  false,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{8},
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
                                  utcmmaDesc_0_0_3,
                                  bool{true});
          }
        }
      }
      //
      // smemKv [ConsRelease, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
      for (int32_t loopOffset2495 = int32_t{0}; loopOffset2495 < (numLoopSteps) - (int32_t{1});
           ++loopOffset2495) {
        //
        // tmemS0 [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset2495) >= (int32_t{0})) {
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
        // tmemO [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset2495) >= (int32_t{0})) {
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
        // smemKv [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
          // SmemKv.h:372
          smemIdxV3 = index;
          // Task.cpp:43
          ++smemKvConsState;
        }
        //
        // tmemO [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
        //
        // TmemO.h:277
        cutlass::float_e4m3_t* smemPtrV21;
        // TmemO.h:282
        int32_t memIdxV21;
        // Task.cpp:1511
        smemPtrV21 = smemPtrV3;
        // Task.cpp:1511
        memIdxV21 = smemIdxV3;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{tmemOProdState.index()};
          // TmemO.h:367
          cutlass::float_e4m3_t* smemP{
            reinterpret_cast<cutlass::float_e4m3_t*>(tmemODstStack.mDepSmemPtr5)};
          // TmemO.h:381
          smemP += int32_t{0};
          // TmemO.h:493
          cutlass::float_e4m3_t* smemV{smemPtrV21};
          // TmemO.h:505
          smemV = smemV + ((memIdxV21) * (int32_t{16384}));
          // TmemO.h:535
          bool readD{true};
          // TmemO.h:545
          if ((loopOffset2495) == (int32_t{0})) {
            // TmemO.h:547
            readD = false;
          }
          // Mma.cpp:618
          {
            // TmemTile.cpp:1765
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
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    true,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{8},
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
                                    bool{readD});
            }
            //
            // MMA inst for mi=0 ni=0 ki=1.
            //
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{256});
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    true,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{8},
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
            //
            // MMA inst for mi=0 ni=0 ki=2.
            //
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{256});
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    true,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{8},
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
                                    utcmmaDesc_0_0_2,
                                    bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=3.
            //
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{256});
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    true,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{8},
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
                                    utcmmaDesc_0_0_3,
                                    bool{true});
            }
          }
        }
        //
        // smemKv [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        // Task.cpp:2568
        if ((loopOffset2495) >= (int32_t{0})) {
          // Task.cpp:2596
          {
            // Task.cpp:2620
            smemKvSrcStack.mPipeline.consumer_release(smemKvConsReleaseState);
          }
          // Task.cpp:43
          ++smemKvConsReleaseState;
        }
        //
        // tmemO [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
        // smemKv [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
          // SmemKv.h:304
          smemIdxK3 = index;
          // Task.cpp:43
          ++smemKvConsState;
        }
        //
        // tmemS0 [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        // Task.cpp:1511
        smemPtrQ12 = smemPtrQ0_2;
        // Task.cpp:1511
        smemIdxQ12 = smemIdxQ0_2;
        // Task.cpp:1511
        smemPtrK12 = smemPtrK3;
        // Task.cpp:1511
        memIdxK12 = smemIdxK3;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{tmemS0ProdState.index()};
          // TmemS.h:1889
          cutlass::float_e4m3_t* smemQ{smemPtrQ12};
          // TmemS.h:1910
          smemQ += (smemIdxQ12) * (int32_t{1024});
          // TmemS.h:1938
          cutlass::float_e4m3_t* smemK{smemPtrK12};
          // TmemS.h:1944
          smemK += (memIdxK12) * (int32_t{16384});
          // Mma.cpp:618
          {
            // TmemTile.cpp:1765
            uint32_t tmemPtrD{int32_t((tmemS0DstStack.mInstId) == (int32_t{0})) ? (int32_t{0})
                                                                                : (int32_t{8})};
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
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{8},
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
                                                                    int32_t{8},
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
            //
            // MMA inst for mi=0 ni=0 ki=2.
            //
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{8},
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
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{8},
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
                                    utcmmaDesc_0_0_3,
                                    bool{true});
            }
          }
        }
        //
        // smemKv [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        // Task.cpp:2568
        if ((loopOffset2495) >= (int32_t{0})) {
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
        // tmemS1 [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset2495) >= (int32_t{0})) {
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
        // tmemO [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:5064
        {
          // Task.cpp:5078
          if ((loopOffset2495) >= (int32_t{0})) {
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
        // smemKv [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
          smemPtrV3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
          // SmemKv.h:372
          smemIdxV3 = index;
          // Task.cpp:43
          ++smemKvConsState;
        }
        //
        // tmemO [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
        //
        // Task.cpp:1511
        smemPtrV21 = smemPtrV3;
        // Task.cpp:1511
        memIdxV21 = smemIdxV3;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{tmemOProdState.index()};
          // TmemO.h:367
          cutlass::float_e4m3_t* smemP{
            reinterpret_cast<cutlass::float_e4m3_t*>(tmemODstStack.mDepSmemPtr5)};
          // TmemO.h:381
          smemP += int32_t{1024};
          // TmemO.h:493
          cutlass::float_e4m3_t* smemV{smemPtrV21};
          // TmemO.h:505
          smemV = smemV + ((memIdxV21) * (int32_t{16384}));
          // TmemO.h:535
          bool readD{true};
          // TmemO.h:545
          if ((loopOffset2495) == (int32_t{0})) {
            // TmemO.h:547
            readD = false;
          }
          // Mma.cpp:618
          {
            // TmemTile.cpp:1765
            uint32_t tmemPtrD{(mTmemBaseOffset) + (uint32_t{88})};
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
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    true,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{8},
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
                                    bool{readD});
            }
            //
            // MMA inst for mi=0 ni=0 ki=1.
            //
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{256});
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    true,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{8},
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
            //
            // MMA inst for mi=0 ni=0 ki=2.
            //
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{256});
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    true,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{8},
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
                                    utcmmaDesc_0_0_2,
                                    bool{true});
            }
            //
            // MMA inst for mi=0 ni=0 ki=3.
            //
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{256});
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    true,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{8},
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
                                    utcmmaDesc_0_0_3,
                                    bool{true});
            }
          }
        }
        //
        // smemKv [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        // Task.cpp:2568
        if ((loopOffset2495) >= (int32_t{0})) {
          // Task.cpp:2596
          {
            // Task.cpp:2620
            smemKvSrcStack.mPipeline.consumer_release(smemKvConsReleaseState);
          }
          // Task.cpp:43
          ++smemKvConsReleaseState;
        }
        //
        // tmemO [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
        // smemKv [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
        // smemKv [ConsWork (call 5), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
        //
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{smemKvConsState.index()};
          // SmemKv.h:267
          smemPtrK3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
          // SmemKv.h:304
          smemIdxK3 = index;
          // Task.cpp:43
          ++smemKvConsState;
        }
        //
        // tmemS1 [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        // Task.cpp:1511
        smemPtrQ15 = smemPtrQ0_2;
        // Task.cpp:1511
        smemIdxQ15 = smemIdxQ0_2;
        // Task.cpp:1511
        smemPtrK15 = smemPtrK3;
        // Task.cpp:1511
        memIdxK15 = smemIdxK3;
        // Task.cpp:1607
        // Task.cpp:5154
        {
          // Task.cpp:5945
          int32_t index{tmemS1ProdState.index()};
          // TmemS.h:1889
          cutlass::float_e4m3_t* smemQ{smemPtrQ15};
          // TmemS.h:1910
          smemQ += (smemIdxQ15) * (int32_t{1024});
          // TmemS.h:1938
          cutlass::float_e4m3_t* smemK{smemPtrK15};
          // TmemS.h:1944
          smemK += (memIdxK15) * (int32_t{16384});
          // Mma.cpp:618
          {
            // TmemTile.cpp:1765
            uint32_t tmemPtrD{int32_t((tmemS1DstStack.mInstId) == (int32_t{0})) ? (int32_t{0})
                                                                                : (int32_t{8})};
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
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{8},
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
                                                                    int32_t{8},
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
            //
            // MMA inst for mi=0 ni=0 ki=2.
            //
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{2});
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{8},
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
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    false,
                                                                    false,
                                                                    int32_t{128},
                                                                    int32_t{8},
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
                                    utcmmaDesc_0_0_3,
                                    bool{true});
            }
          }
        }
        //
        // smemKv [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
        //
        // Task.cpp:2568
        if ((loopOffset2495) >= (int32_t{0})) {
          // Task.cpp:2596
          {
            // Task.cpp:2620
            smemKvSrcStack.mPipeline.consumer_release(smemKvConsReleaseState);
          }
          // Task.cpp:43
          ++smemKvConsReleaseState;
        }
        //
        // tmemS1 [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
        lastLoopOffset = loopOffset2495;
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
      // tmemS0 [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
      // tmemO [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
      // smemKv [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
        smemPtrV3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:372
        smemIdxV3 = index;
        // Task.cpp:43
        ++smemKvConsState;
      }
      //
      // tmemO [ProdWork (call 2), LastIter, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
      //
      // TmemO.h:277
      cutlass::float_e4m3_t* smemPtrV21;
      // TmemO.h:282
      int32_t memIdxV21;
      // Task.cpp:1511
      smemPtrV21 = smemPtrV3;
      // Task.cpp:1511
      memIdxV21 = smemIdxV3;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemOProdState.index()};
        // TmemO.h:367
        cutlass::float_e4m3_t* smemP{
          reinterpret_cast<cutlass::float_e4m3_t*>(tmemODstStack.mDepSmemPtr5)};
        // TmemO.h:381
        smemP += int32_t{0};
        // TmemO.h:493
        cutlass::float_e4m3_t* smemV{smemPtrV21};
        // TmemO.h:505
        smemV = smemV + ((memIdxV21) * (int32_t{16384}));
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
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{8},
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
                                  bool{readD});
          }
          //
          // MMA inst for mi=0 ni=0 ki=1.
          //
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{256});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{8},
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
          //
          // MMA inst for mi=0 ni=0 ki=2.
          //
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{256});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{8},
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
                                  utcmmaDesc_0_0_2,
                                  bool{true});
          }
          //
          // MMA inst for mi=0 ni=0 ki=3.
          //
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{256});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{8},
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
                                  utcmmaDesc_0_0_3,
                                  bool{true});
          }
        }
      }
      //
      // smemKv [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
      // tmemO [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
      // tmemS1 [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
      // tmemO [ProdAcquire, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
      // smemKv [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
        smemPtrV3 = &smemKvSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:372
        smemIdxV3 = index;
        // Task.cpp:43
        ++smemKvConsState;
      }
      //
      // tmemO [ProdWork (call 3), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1511
      smemPtrV21 = smemPtrV3;
      // Task.cpp:1511
      memIdxV21 = smemIdxV3;
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemOProdState.index()};
        // TmemO.h:367
        cutlass::float_e4m3_t* smemP{
          reinterpret_cast<cutlass::float_e4m3_t*>(tmemODstStack.mDepSmemPtr5)};
        // TmemO.h:381
        smemP += int32_t{1024};
        // TmemO.h:493
        cutlass::float_e4m3_t* smemV{smemPtrV21};
        // TmemO.h:505
        smemV = smemV + ((memIdxV21) * (int32_t{16384}));
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
          uint32_t tmemPtrD{(mTmemBaseOffset) + (uint32_t{88})};
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
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_0{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{8},
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
                                  bool{readD});
          }
          //
          // MMA inst for mi=0 ni=0 ki=1.
          //
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{256});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{8},
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
          //
          // MMA inst for mi=0 ni=0 ki=2.
          //
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{256});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{8},
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
                                  utcmmaDesc_0_0_2,
                                  bool{true});
          }
          //
          // MMA inst for mi=0 ni=0 ki=3.
          //
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{256});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{128},
                                                                  int32_t{8},
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
                                  utcmmaDesc_0_0_3,
                                  bool{true});
          }
        }
      }
      //
      // smemKv [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
      // tmemO [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
      // tmemS1 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{4}].
      //
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
    // tmemS1 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{4}].
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
__launch_bounds__(512, 1) void fmhaSm100fKernel_QkvE4m3OFp16H128PagedKvCausalP64VarSeqQ8Kv128PersistentSwapsAbForGen(
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
  smemOffset__ = (((smemOffset__) + (int32_t{1023})) / (int32_t{1024})) * (int32_t{1024});
  // Kernel.cpp:1729
  uint8_t* smemPSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemPSmem)});
  // Kernel.cpp:1725
  smemOffset__ = (((smemOffset__) + (int32_t{127})) / (int32_t{128})) * (int32_t{128});
  // Kernel.cpp:1729
  uint8_t* smemPageOffsetsKvSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemPageOffsetsKvSmem)});
  // Kernel.cpp:1725
  smemOffset__ = (((smemOffset__) + (int32_t{127})) / (int32_t{128})) * (int32_t{128});
  // Kernel.cpp:1729
  uint8_t* smemOSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemOSmem)});
  // Kernel.cpp:1725
  smemOffset__ = (((smemOffset__) + (int32_t{15})) / (int32_t{16})) * (int32_t{16});
  // Kernel.cpp:1729
  uint8_t* smemSoftmaxWarpGrpRed0SmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemSoftmaxWarpGrpRed0Smem)});
  // Kernel.cpp:1725
  smemOffset__ = (((smemOffset__) + (int32_t{15})) / (int32_t{16})) * (int32_t{16});
  // Kernel.cpp:1729
  uint8_t* smemSoftmaxWarpGrpRed1SmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemSoftmaxWarpGrpRed0Smem)});
  // Kernel.cpp:1725
  smemOffset__ = (((smemOffset__) + (int32_t{15})) / (int32_t{16})) * (int32_t{16});
  // Kernel.cpp:1729
  uint8_t* smemCorrWarpGrpRed1SmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemCorrWarpGrpRed1Smem)});
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
  SmemPSmem* smemPSmem{reinterpret_cast<SmemPSmem*>(smemPSmemPtr)};
  // Kernel.cpp:2283
  SmemPStack smemPStack{(*smemPSmem),
                        state.mWarpIdx,
                        state.mClusterDimX,
                        state.mClusterDimY,
                        int32_t{0},
                        int32_t{-1}};
  // Kernel.cpp:2216
  SmemOSmem* smemOSmem{reinterpret_cast<SmemOSmem*>(smemOSmemPtr)};
  // Kernel.cpp:2283
  SmemOStack smemOStack{(*smemOSmem),
                        state.mWarpIdx,
                        state.mClusterDimX,
                        state.mClusterDimY,
                        int32_t{0},
                        int32_t{-1}};
  // Kernel.cpp:2216
  SmemSoftmaxWarpGrpRed0Smem* smemSoftmaxWarpGrpRed0Smem{
    reinterpret_cast<SmemSoftmaxWarpGrpRed0Smem*>(smemSoftmaxWarpGrpRed0SmemPtr)};
  // Kernel.cpp:2283
  SmemSoftmaxWarpGrpRed0Stack smemSoftmaxWarpGrpRed0Stack{(*smemSoftmaxWarpGrpRed0Smem),
                                                          state.mWarpIdx,
                                                          state.mClusterDimX,
                                                          state.mClusterDimY,
                                                          int32_t{0},
                                                          int32_t{-1}};
  // Kernel.cpp:2216
  SmemSoftmaxWarpGrpRed0Smem* smemSoftmaxWarpGrpRed1Smem{
    reinterpret_cast<SmemSoftmaxWarpGrpRed0Smem*>(smemSoftmaxWarpGrpRed1SmemPtr)};
  // Kernel.cpp:2283
  SmemSoftmaxWarpGrpRed0Stack smemSoftmaxWarpGrpRed1Stack{(*smemSoftmaxWarpGrpRed1Smem),
                                                          state.mWarpIdx,
                                                          state.mClusterDimX,
                                                          state.mClusterDimY,
                                                          int32_t{0},
                                                          int32_t{-1}};
  // Kernel.cpp:2216
  SmemCorrWarpGrpRed1Smem* smemCorrWarpGrpRed1Smem{
    reinterpret_cast<SmemCorrWarpGrpRed1Smem*>(smemCorrWarpGrpRed1SmemPtr)};
  // Kernel.cpp:2283
  SmemCorrWarpGrpRed1Stack smemCorrWarpGrpRed1Stack{(*smemCorrWarpGrpRed1Smem),
                                                    state.mWarpIdx,
                                                    state.mClusterDimX,
                                                    state.mClusterDimY,
                                                    int32_t{0},
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
                          (*smemSoftmaxWarpGrpRed0Smem),
                          smemSoftmaxWarpGrpRed0Stack,
                          state.mWarpIdx,
                          state.mClusterDimX,
                          state.mClusterDimY,
                          int32_t{1},
                          int32_t{-1},
                          int32_t{1},
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
                          (*smemSoftmaxWarpGrpRed1Smem),
                          smemSoftmaxWarpGrpRed1Stack,
                          state.mWarpIdx,
                          state.mClusterDimX,
                          state.mClusterDimY,
                          int32_t{5},
                          int32_t{-1},
                          int32_t{2},
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
                          (*smemPSmem),
                          smemPStack,
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
                          (*smemPSmem),
                          smemPStack,
                          (*orderP01SmemBarrier),
                          orderP01Stack,
                          state.mWarpIdx,
                          state.mClusterDimX,
                          state.mClusterDimY,
                          int32_t{6},
                          int32_t{1},
                          int32_t{3},
                          int32_t{1}};
  // Kernel.cpp:2228
  TmemOSmemBarrier* tmemOSmemBarrier{reinterpret_cast<TmemOSmemBarrier*>(tmemOSmemBarrierPtr)};
  // Kernel.cpp:2283
  TmemOStack tmemOStack{(*tmemOSmemBarrier),
                        (*smemPSmem),
                        smemPStack,
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
  TmemCorr1Stack tmemCorr1Stack{(*smemCorrWarpGrpRed1Smem),
                                smemCorrWarpGrpRed1Stack,
                                (*smemOSmem),
                                smemOStack,
                                state.mWarpIdx,
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
          trtllm::dev::CutlassNamedBarrier::sync(128, 11);
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
fmhaSm100fKernel_QkvE4m3OFp16H128PagedKvCausalP64VarSeqQ8Kv128PersistentSwapsAbForGenGetSmemSize(
  int32_t* outPtr) {
  int32_t size{0};
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemQSmem));
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemKvSmem));
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
