#include <FmhaSm100fKernel_QkvE4m3OE2m1H64PagedKvCausalP16MultiCtasKvCgaVarSeqSkipsSoftmaxQ32Kv128StaticSwapsAbForGen.h>

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
              int32_t{2048},
              bool{cute::elect_one_sync()},
              CuteFlatTuple249{},
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
// Fmha.h:1041
struct SmemKStack {
  // Res.cpp:595
  trtllm::dev::CutlassTmaUmmaAsyncPipeline<9, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  // MemBuffers.cpp:275
  cutlass::float_e4m3_t* mPtr;
  // Res.cpp:208
  inline __device__ SmemKStack(SmemKSmem& smemKSmem,
                               SmemKSmemBarrier& smemKSmemBarrier,
                               int32_t warpId,
                               int32_t clusterDimX,
                               int32_t clusterDimY,
                               int32_t barInitWarpId,
                               int32_t orderedSequenceGroupId)
    : // Res.cpp:719
    mPipeline{smemKSmemBarrier.mBarriers,
              warpId,
              int32_t{8192},
              bool{cute::elect_one_sync()},
              CuteFlatTuple378{},
              cute::true_type{},
              cute::true_type{},
              barInitWarpId}
    , // MemBuffers.cpp:282
    mPtr{&smemKSmem.mArray[int32_t{0}][int32_t{0}]} {}
};
// Res.cpp:137
// Fmha.h:1060
struct SmemVStack {
  // MemBuffers.cpp:319
  int32_t* mDepSmemPtr3;
  // Res.cpp:595
  trtllm::dev::CutlassTmaUmmaAsyncPipeline<9, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  // MemBuffers.cpp:275
  cutlass::float_e4m3_t* mPtr;
  // Res.cpp:208
  inline __device__ SmemVStack(SmemVSmem& smemVSmem,
                               SmemVSmemBarrier& smemVSmemBarrier,
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
    mPipeline{smemVSmemBarrier.mBarriers,
              warpId,
              int32_t{8192},
              bool{cute::elect_one_sync()},
              CuteFlatTuple510{},
              cute::true_type{},
              cute::true_type{},
              barInitWarpId}
    , // MemBuffers.cpp:282
    mPtr{&smemVSmem.mArray[int32_t{0}][int32_t{0}]} {}
};
// Res.cpp:137
// Fmha.h:1193
struct SmemPageOffsetsKStack {
  // Res.cpp:595
  trtllm::dev::CutlassCpAsyncPipeline<6> mPipeline;
  // Res.cpp:208
  inline __device__ SmemPageOffsetsKStack(SmemPageOffsetsKSmem& smemPageOffsetsKSmem,
                                          SmemPageOffsetsKSmemBarrier& smemPageOffsetsKSmemBarrier,
                                          int32_t warpId,
                                          int32_t clusterDimX,
                                          int32_t clusterDimY,
                                          int32_t barInitWarpId,
                                          int32_t orderedSequenceGroupId)
    : // Res.cpp:644
    mPipeline{smemPageOffsetsKSmemBarrier.mBarriers,
              warpId,
              int32_t{32},
              int32_t{32},
              barInitWarpId} {}
};
// Res.cpp:137
// Fmha.h:1201
struct SmemPageOffsetsVStack {
  // Res.cpp:595
  trtllm::dev::CutlassCpAsyncPipeline<6> mPipeline;
  // Res.cpp:208
  inline __device__ SmemPageOffsetsVStack(SmemPageOffsetsVSmem& smemPageOffsetsVSmem,
                                          SmemPageOffsetsVSmemBarrier& smemPageOffsetsVSmemBarrier,
                                          int32_t warpId,
                                          int32_t clusterDimX,
                                          int32_t clusterDimY,
                                          int32_t barInitWarpId,
                                          int32_t orderedSequenceGroupId)
    : // Res.cpp:644
    mPipeline{smemPageOffsetsVSmemBarrier.mBarriers,
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
// Fmha.h:1433
struct ClusterTransactionBarrierBuffersStack {
  // MemBuffers.cpp:94
  trtllm::dev::CutlassClusterTransactionBarrier<
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mClusterBarrier;
  // Res.cpp:208
  inline __device__ ClusterTransactionBarrierBuffersStack(
    ClusterTransactionBarrierBuffersSmemBarrier& clusterTransactionBarrierBuffersSmemBarrier,
    int32_t warpId,
    int32_t clusterDimX,
    int32_t clusterDimY,
    int32_t barInitWarpId,
    int32_t orderedSequenceGroupId)
    : // MemBuffers.cpp:100
    mClusterBarrier{clusterTransactionBarrierBuffersSmemBarrier.mClusterSmemBarriers,
                    warpId,
                    int32_t{1},
                    int32_t{34816},
                    int32_t{0}} {}
};
// Res.cpp:137
// Fmha.h:1475
struct SmemCgaReductionStack {
  // Res.cpp:208
  inline __device__ SmemCgaReductionStack(SmemCgaReductionSmem& smemCgaReductionSmem,
                                          int32_t warpId,
                                          int32_t clusterDimX,
                                          int32_t clusterDimY,
                                          int32_t barInitWarpId,
                                          int32_t orderedSequenceGroupId) {}
};
// Res.cpp:137
// Fmha.h:1890
struct TmemS0Stack {
  // MemBuffers.cpp:488
  float* mDepSmemPtr9;
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
                                SmemSoftmaxWarpGrpRed0Smem& smemSoftmaxWarpGrpRed0Smem,
                                SmemSoftmaxWarpGrpRed0Stack& smemSoftmaxWarpGrpRed0Stack,
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
    mDepSmemPtr9{&smemSoftmaxWarpGrpRed0Smem.mArray[int32_t{0}]}
    , // MemBuffers.cpp:501
    mDepSmemPtr3{&smemSkipSoftmaxVoteSmem.mArray[int32_t{0}]}
    , // Res.cpp:776
    mPipeline{tmemS0SmemBarrier.mBarriers,
              warpId,
              int32_t{128},
              CuteFlatTuple928{},
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
  // MemBuffers.cpp:488
  int8_t* mDepSmemPtr8;
  // Res.cpp:595
  trtllm::dev::CutlassCpAsyncPipeline<2, true> mPipeline;
  // TmemP.h:472
  int32_t const mNamedBarId;
  // TmemP.h:475
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
    mDepSmemPtr8{&smemPOSmem.mArray[int32_t{0}]}
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
  int8_t* mDepSmemPtr8;
  // MemBuffers.cpp:488
  int32_t* mDepSmemPtr3;
  // Res.cpp:595
  trtllm::dev::CutlassUmmaAsyncPipeline<1, cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>
    mPipeline;
  // Res.cpp:208
  inline __device__ TmemOStack(TmemOSmemBarrier& tmemOSmemBarrier,
                               SmemPOSmem& smemPOSmem,
                               SmemPOStack& smemPOStack,
                               SmemSkipSoftmaxVoteSmem& smemSkipSoftmaxVoteSmem,
                               SmemSkipSoftmaxVoteStack& smemSkipSoftmaxVoteStack,
                               int32_t warpId,
                               int32_t clusterDimX,
                               int32_t clusterDimY,
                               int32_t barInitWarpId,
                               int32_t orderedSequenceGroupId)
    : // MemBuffers.cpp:501
    mDepSmemPtr8{&smemPOSmem.mArray[int32_t{0}]}
    , // MemBuffers.cpp:501
    mDepSmemPtr3{&smemSkipSoftmaxVoteSmem.mArray[int32_t{0}]}
    , // Res.cpp:776
    mPipeline{tmemOSmemBarrier.mBarriers,
              warpId,
              int32_t{128},
              CuteFlatTuple1311{},
              cute::true_type{},
              cute::true_type{},
              barInitWarpId} {}
};
// Res.cpp:137
// Fmha.h:2121
struct TmemCorr0Stack {
  // MemBuffers.cpp:488
  float* mDepSmemPtr11;
  // MemBuffers.cpp:488
  int8_t* mDepSmemPtr8;
  // MemBuffers.cpp:488
  int8_t* mDepSmemPtr13;
  // MemBuffers.cpp:521
  trtllm::dev::CutlassClusterTransactionBarrier<
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>* mClusterBarrierPtr12;
  // Res.cpp:208
  inline __device__ TmemCorr0Stack(
    SmemCorrWarpGrpRed1Smem& smemCorrWarpGrpRed1Smem,
    SmemCorrWarpGrpRed1Stack& smemCorrWarpGrpRed1Stack,
    SmemPOSmem& smemPOSmem,
    SmemPOStack& smemPOStack,
    SmemCgaReductionSmem& smemCgaReductionSmem,
    SmemCgaReductionStack& smemCgaReductionStack,
    ClusterTransactionBarrierBuffersSmemBarrier& clusterTransactionBarrierBuffersSmemBarrier,
    ClusterTransactionBarrierBuffersStack& clusterTransactionBarrierBuffersStack,
    int32_t warpId,
    int32_t clusterDimX,
    int32_t clusterDimY,
    int32_t barInitWarpId,
    int32_t orderedSequenceGroupId)
    : // MemBuffers.cpp:501
    mDepSmemPtr11{&smemCorrWarpGrpRed1Smem.mArray[int32_t{0}]}
    , // MemBuffers.cpp:501
    mDepSmemPtr8{&smemPOSmem.mArray[int32_t{0}]}
    , // MemBuffers.cpp:501
    mDepSmemPtr13{&smemCgaReductionSmem.mArray[int32_t{0}]}
    , // MemBuffers.cpp:534
    mClusterBarrierPtr12{&clusterTransactionBarrierBuffersStack.mClusterBarrier} {}
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
    , // FmhaTask.h:516
    mCtaIdxQ{(mCtaIdxX) / (params.mMaxNumCtasKv)}
    , // FmhaTask.h:517
    mCtaIdxKv{(mCtaIdxX) % (params.mMaxNumCtasKv)}
    , // FmhaTask.h:437
    mSeqLenKv{int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                            ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                            : (mBatchIdx)]}}
    , // FmhaTask.h:565
    mNumCtasKv{
      int32_t{min(int32_t{((mSeqLenKv) + (int32_t{127})) / (int32_t{128})}, params.mMaxNumCtasKv)}}
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
                                 SmemPageOffsetsKSmem& smemPageOffsetsKDstSmem,
                                 SmemPageOffsetsKStack& smemPageOffsetsKDstStack,
                                 SmemPageOffsetsVSmem& smemPageOffsetsVDstSmem,
                                 SmemPageOffsetsVStack& smemPageOffsetsVDstStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<112>{});
    // Task.cpp:2013
    trtllm::dev::CutlassCpAsyncPipeline<6>::PipelineState smemPageOffsetsKProdState{int32_t{0},
                                                                                    int32_t{1},
                                                                                    int32_t{0}};
    // Task.cpp:2033
    int32_t smemPageOffsetsKProdToken{int32_t{1}};
    // Task.cpp:2013
    trtllm::dev::CutlassCpAsyncPipeline<6>::PipelineState smemPageOffsetsVProdState{int32_t{0},
                                                                                    int32_t{1},
                                                                                    int32_t{0}};
    // Task.cpp:2033
    int32_t smemPageOffsetsVProdToken{int32_t{1}};
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
    mNumCtasKv = int32_t{
      min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, params.mMaxNumCtasKv)};
    // FmhaTask.h:630
    if ((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
      // FmhaTask.h:668
      int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
                       ((mNumCtasKv) * (int32_t{128}))};
      // FmhaTask.h:682
      numLoopSteps = numSteps;
    } else {
      // FmhaTask.h:651
      return;
    }
    // SmemPageOffsetsKv.h:204
    int32_t const* ptrPageIdxK6;
    // SmemPageOffsetsKv.h:210
    ptrPageIdxK6 =
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
    int32_t const* ptrPageIdxV6;
    // SmemPageOffsetsKv.h:224
    if (params.mUsesSharedPagedKvIdx) {
      // SmemPageOffsetsKv.h:226
      ptrPageIdxV6 = ptrPageIdxK6;
    } else {
      // SmemPageOffsetsKv.h:228
      ptrPageIdxV6 = ptrPageIdxK6 + (params.mMaxNumPagesPerSeqKv);
    }
    // SmemPageOffsetsKv.h:302
    int32_t pageIdxUb6{(int32_t{((mSeqLenKv) + (int32_t{15})) / (int32_t{16})}) - (int32_t{1})};
    // SmemPageOffsetsKv.h:204
    int32_t const* ptrPageIdxK7;
    // SmemPageOffsetsKv.h:210
    ptrPageIdxK7 =
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
    int32_t const* ptrPageIdxV7;
    // SmemPageOffsetsKv.h:224
    if (params.mUsesSharedPagedKvIdx) {
      // SmemPageOffsetsKv.h:226
      ptrPageIdxV7 = ptrPageIdxK7;
    } else {
      // SmemPageOffsetsKv.h:228
      ptrPageIdxV7 = ptrPageIdxK7 + (params.mMaxNumPagesPerSeqKv);
    }
    // SmemPageOffsetsKv.h:302
    int32_t pageIdxUb7{(int32_t{((mSeqLenKv) + (int32_t{15})) / (int32_t{16})}) - (int32_t{1})};
    //
    // Loop body.
    //
    // Task.cpp:3392
    CUTLASS_PRAGMA_NO_UNROLL
    for (int32_t loopOffset473 = int32_t{0}; loopOffset473 < numLoopSteps;
         loopOffset473 += int32_t{4}) {
      // Task.cpp:3445
      bool const isFirstLoopIter{(loopOffset473) == (int32_t{0})};
      // Task.cpp:3465
      bool const isLastLoopIter{((loopOffset473) + (int32_t{4})) >= (numLoopSteps)};
      //
      // smemPageOffsetsK [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1607
      // Task.cpp:5064
      {
        // Task.cpp:5078
        if ((loopOffset473) >= (int32_t{0})) {
          // Task.cpp:5100
          smemPageOffsetsKProdToken =
            smemPageOffsetsKDstStack.mPipeline.producer_try_acquire(smemPageOffsetsKProdState);
        }
      }
      // Task.cpp:1607
      // Task.cpp:4288
      {
        // Task.cpp:4318
        smemPageOffsetsKDstStack.mPipeline.producer_acquire(smemPageOffsetsKProdState,
                                                            smemPageOffsetsKProdToken);
      }
      //
      // smemPageOffsetsK [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1607
      // Task.cpp:5154
      {
        // Task.cpp:5945
        int32_t index{smemPageOffsetsKProdState.index()};
        // SmemPageOffsetsKv.h:390
        int32_t* ptrSmemPageOffsets;
        // SmemPageOffsetsKv.h:392
        ptrSmemPageOffsets = smemPageOffsetsKDstSmem.mArray[index] + (mWarpGrpThreadIdx);
        // SmemPageOffsetsKv.h:430
        int32_t pageIdx{(((mCtaIdxKv) * (numLoopSteps) + (loopOffset473)) * (int32_t{8})) +
                        (mWarpGrpThreadIdx)};
        // SmemPageOffsetsKv.h:488
        trtllm::dev::cpAsync((ptrSmemPageOffsets + int32_t{0}),
                             (ptrPageIdxK6 + int32_t{min(pageIdx, pageIdxUb6)}),
                             int32_t{0},
                             int32_t{0},
                             int32_t{4});
      }
      //
      // smemPageOffsetsK [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:4522
      {
        // Task.cpp:4540
        {
          // Task.cpp:4556
          smemPageOffsetsKDstStack.mPipeline.producer_commit(smemPageOffsetsKProdState);
        }
        // Task.cpp:43
        ++smemPageOffsetsKProdState;
      }
      //
      // smemPageOffsetsV [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:1607
      // Task.cpp:5064
      {
        // Task.cpp:5078
        if ((loopOffset473) >= (int32_t{0})) {
          // Task.cpp:5100
          smemPageOffsetsVProdToken =
            smemPageOffsetsVDstStack.mPipeline.producer_try_acquire(smemPageOffsetsVProdState);
        }
      }
      // Task.cpp:1607
      // Task.cpp:4288
      {
        // Task.cpp:4318
        smemPageOffsetsVDstStack.mPipeline.producer_acquire(smemPageOffsetsVProdState,
                                                            smemPageOffsetsVProdToken);
      }
      //
      // smemPageOffsetsV [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:1607
      // Task.cpp:5154
      {
        // Task.cpp:5945
        int32_t index{smemPageOffsetsVProdState.index()};
        // SmemPageOffsetsKv.h:390
        int32_t* ptrSmemPageOffsets;
        // SmemPageOffsetsKv.h:392
        ptrSmemPageOffsets = smemPageOffsetsVDstSmem.mArray[index] + (mWarpGrpThreadIdx);
        // SmemPageOffsetsKv.h:430
        int32_t pageIdx{(((mCtaIdxKv) * (numLoopSteps) + (loopOffset473)) * (int32_t{8})) +
                        (mWarpGrpThreadIdx)};
        // SmemPageOffsetsKv.h:488
        trtllm::dev::cpAsync((ptrSmemPageOffsets + int32_t{0}),
                             (ptrPageIdxV7 + int32_t{min(pageIdx, pageIdxUb7)}),
                             int32_t{0},
                             int32_t{0},
                             int32_t{4});
      }
      //
      // smemPageOffsetsV [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:4522
      {
        // Task.cpp:4540
        {
          // Task.cpp:4556
          smemPageOffsetsVDstStack.mPipeline.producer_commit(smemPageOffsetsVProdState);
        }
        // Task.cpp:43
        ++smemPageOffsetsVProdState;
      }
    }
  //
  // Tail work.
  //
  // Task.cpp:3553
  ExitTileWithSignalingLabel:
  // Task.cpp:3560
  ExitTileWithoutSignalingLabel:
    // Task.cpp:3570
    {}
  }
};
// Task.cpp:559
// Fmha.h:1625
struct LoadTaskQk {
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
  inline __device__ LoadTaskQk(fmha::KernelParams const& params,
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
    , // FmhaTask.h:437
    mSeqLenKv{int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                            ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                            : (mBatchIdx)]}}
    , // FmhaTask.h:565
    mNumCtasKv{int32_t{
      min(int32_t{((mSeqLenKv) + (int32_t{127})) / (int32_t{128})}, params.mMaxNumCtasKv)}} {}
  // Task.cpp:522
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:547
    return ((state.mWarpIdx) >= (int32_t{10})) && ((state.mWarpIdx) < (int32_t{11}));
  }
  // Task.cpp:454
  inline __device__ void execute(fmha::KernelParams const& params,
                                 KernelState const& state,
                                 SmemQSmem& smemQDstSmem,
                                 SmemQStack& smemQDstStack,
                                 SmemKSmem& smemKDstSmem,
                                 SmemKStack& smemKDstStack,
                                 SmemPageOffsetsKSmem& smemPageOffsetsKSrcSmem,
                                 SmemPageOffsetsKStack& smemPageOffsetsKSrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<112>{});
    // Task.cpp:2114
    trtllm::dev::CutlassCpAsyncPipeline<6>::PipelineState smemPageOffsetsKConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassCpAsyncPipeline<6>::PipelineState smemPageOffsetsKConsReleaseState{};
    // Task.cpp:2135
    int32_t smemPageOffsetsKConsToken{int32_t{0}};
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
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemKProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    // Task.cpp:2033
    int32_t smemKProdToken{int32_t{1}};
    // SmemKv.h:749
    int32_t smemVoteIdx4{int32_t{0}};
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
    mNumCtasKv = int32_t{
      min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, params.mMaxNumCtasKv)};
    // FmhaTask.h:630
    if ((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
      // FmhaTask.h:668
      int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
                       ((mNumCtasKv) * (int32_t{128}))};
      // FmhaTask.h:682
      numLoopSteps = numSteps;
    } else {
      // FmhaTask.h:651
      return;
    }
    // Task.cpp:3203
    bool const hasOneLoopIter{(int32_t{0}) < (numLoopSteps)};
    // Task.cpp:3214
    int32_t lastLoopOffset{int32_t{0}};
    // SmemKv.h:683
    bool skipsLoadingV4;
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
          ((int32_t{32}) - ((params.mNumTokensPerCtaQ) * (params.mNumHeadsQPerKv))) *
            (int32_t{64}));
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
    // smemPageOffsetsK [ConsWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{2}].
    //
    // SmemPageOffsetsKv.h:320
    int32_t* ptrSmemPageOffsetsK6;
    // SmemPageOffsetsKv.h:326
    int32_t* ptrSmemPageOffsetsV6;
    //
    // Loop body.
    //
    // Task.cpp:3392
    CUTLASS_PRAGMA_NO_UNROLL
    for (int32_t loopOffset621 = int32_t{0}; loopOffset621 < numLoopSteps; ++loopOffset621) {
      // Task.cpp:3465
      bool const isLastLoopIter{((loopOffset621) + (int32_t{1})) >= (numLoopSteps)};
      //
      // gmemKv [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // Task.cpp:1607
      // Task.cpp:2928
      {}
      //
      // smemPageOffsetsK [ConsWait, Info{0}, FreqInfo{0, 4}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:3814
      if (((loopOffset621) % (int32_t{4})) == (int32_t{0})) {
        // Task.cpp:1607
        // Task.cpp:2816
        {
          // Task.cpp:1607
          // Task.cpp:2757
          {
            // Task.cpp:2780
            smemPageOffsetsKConsToken =
              smemPageOffsetsKSrcStack.mPipeline.consumer_try_wait(smemPageOffsetsKConsState);
          }
          // Task.cpp:2848
          smemPageOffsetsKSrcStack.mPipeline.consumer_wait(smemPageOffsetsKConsState,
                                                           smemPageOffsetsKConsToken);
        }
      }
      //
      // smemPageOffsetsK [ConsWork (call 0), Info{0}, FreqInfo{0, 4}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:3814
      if (((loopOffset621) % (int32_t{4})) == (int32_t{0})) {
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{smemPageOffsetsKConsState.index()};
          // SmemPageOffsetsKv.h:349
          ptrSmemPageOffsetsK6 = smemPageOffsetsKSrcSmem.mArray[index];
          // Task.cpp:43
          ++smemPageOffsetsKConsState;
        }
      }
      //
      // smemK [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1607
      // Task.cpp:5064
      {
        // Task.cpp:5078
        if ((loopOffset621) >= (int32_t{0})) {
          // Task.cpp:5100
          smemKProdToken = smemKDstStack.mPipeline.producer_try_acquire(smemKProdState);
        }
      }
      // Task.cpp:1607
      // Task.cpp:4288
      {
        // Task.cpp:4318
        smemKDstStack.mPipeline.producer_acquire(smemKProdState, smemKProdToken);
      }
      //
      // smemK [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // SmemKv.h:772
      int32_t* ptrSmemPageOffsetsK4;
      // Task.cpp:1511
      ptrSmemPageOffsetsK4 = ptrSmemPageOffsetsK6;
      // Task.cpp:1607
      // Task.cpp:5154
      {
        // Task.cpp:4413
        uint64_t* barrier{smemKDstStack.mPipeline.producer_get_barrier(smemKProdState)};
        // Task.cpp:5945
        int32_t index{smemKProdState.index()};
        // SmemKv.h:631
        int32_t adjustedGlobalTileIdx{(mCtaIdxKv) * (numLoopSteps) + (loopOffset621)};
        // SmemKv.h:1430
        int32_t headDimOffset{int32_t{0}};
        // SmemKv.h:1555
        int32_t tokenOffset{int32_t{0}};
        //
        // Load pageOffsets for headDimStageIdx = 0.
        //
        // SmemKv.h:1695
        cutlass::AlignedArray<int32_t, 4> localPageOffsets04;
        // SmemKv.h:1711
        localPageOffsets04 = reinterpret_cast<cutlass::AlignedArray<int32_t, 4>*>(
          (ptrSmemPageOffsetsK4 + ((loopOffset621) * (int32_t{8})) % (int32_t{32})))[int32_t{0}];
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
          coords[int32_t{3}] = localPageOffsets04[int32_t{0}];
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKDstSmem.mArray[index][int32_t{0}],
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
          coords[int32_t{3}] = localPageOffsets04[int32_t{1}];
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKDstSmem.mArray[index][int32_t{1024}],
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
          coords[int32_t{3}] = localPageOffsets04[int32_t{2}];
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKDstSmem.mArray[index][int32_t{2048}],
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
          coords[int32_t{3}] = localPageOffsets04[int32_t{3}];
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKDstSmem.mArray[index][int32_t{3072}],
                                           &params.tmaK_,
                                           coords,
                                           barrier);
          }
        }
        //
        // Load pageOffsets for headDimStageIdx = 0.
        //
        // SmemKv.h:1695
        cutlass::AlignedArray<int32_t, 4> localPageOffsets14;
        // SmemKv.h:1711
        localPageOffsets14 = reinterpret_cast<cutlass::AlignedArray<int32_t, 4>*>(
          (ptrSmemPageOffsetsK4 +
           (((loopOffset621) * (int32_t{8})) + (int32_t{4})) % (int32_t{32})))[int32_t{0}];
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
          coords[int32_t{3}] = localPageOffsets14[int32_t{0}];
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKDstSmem.mArray[index][int32_t{4096}],
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
          coords[int32_t{3}] = localPageOffsets14[int32_t{1}];
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKDstSmem.mArray[index][int32_t{5120}],
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
          coords[int32_t{3}] = localPageOffsets14[int32_t{2}];
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKDstSmem.mArray[index][int32_t{6144}],
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
          coords[int32_t{3}] = localPageOffsets14[int32_t{3}];
          // SmemTile.cpp:611
          if (bool{cute::elect_one_sync()}) {
            // CudaPtx.h:48
            cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                           cuda_ptx::space_global_t{},
                                           &smemKDstSmem.mArray[index][int32_t{7168}],
                                           &params.tmaK_,
                                           coords,
                                           barrier);
          }
        }
      }
      //
      // smemK [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:4522
      {
        // Task.cpp:4540
        {
          // Task.cpp:4556
          smemKDstStack.mPipeline.producer_commit(smemKProdState);
        }
        // Task.cpp:43
        ++smemKProdState;
      }
      //
      // smemPageOffsetsK [ConsRelease, Info{0}, FreqInfo{0, 4}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:3814
      if ((((loopOffset621) % (int32_t{4})) == (int32_t{3})) ||
          ((loopOffset621) == ((numLoopSteps) - (int32_t{1})))) {
        // Task.cpp:2568
        if ((loopOffset621) >= (int32_t{0})) {
          // Task.cpp:2596
          {
            // Task.cpp:2620
            smemPageOffsetsKSrcStack.mPipeline.consumer_release(smemPageOffsetsKConsReleaseState);
          }
          // Task.cpp:43
          ++smemPageOffsetsKConsReleaseState;
        }
      }
      // Task.cpp:3499
      lastLoopOffset = loopOffset621;
    }
  //
  // Pull the last iter down.
  //
  //
  // Tail work.
  //
  // Task.cpp:3553
  ExitTileWithSignalingLabel:
  // Task.cpp:3560
  ExitTileWithoutSignalingLabel:
    // Task.cpp:3570
    {}
  }
};
// Task.cpp:559
// Fmha.h:1638
struct LoadTaskV {
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
  inline __device__ LoadTaskV(fmha::KernelParams const& params,
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
    , // FmhaTask.h:437
    mSeqLenKv{int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                            ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                            : (mBatchIdx)]}}
    , // FmhaTask.h:565
    mNumCtasKv{int32_t{
      min(int32_t{((mSeqLenKv) + (int32_t{127})) / (int32_t{128})}, params.mMaxNumCtasKv)}} {}
  // Task.cpp:522
  static inline __device__ bool isSelected(fmha::KernelParams const& params,
                                           KernelState const& state) {
    // Task.cpp:547
    return ((state.mWarpIdx) >= (int32_t{11})) && ((state.mWarpIdx) < (int32_t{12}));
  }
  // Task.cpp:454
  inline __device__ void execute(fmha::KernelParams const& params,
                                 KernelState const& state,
                                 SmemVSmem& smemVDstSmem,
                                 SmemVStack& smemVDstStack,
                                 SmemPageOffsetsVSmem& smemPageOffsetsVSrcSmem,
                                 SmemPageOffsetsVStack& smemPageOffsetsVSrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<112>{});
    // Task.cpp:2114
    trtllm::dev::CutlassCpAsyncPipeline<6>::PipelineState smemPageOffsetsVConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassCpAsyncPipeline<6>::PipelineState smemPageOffsetsVConsReleaseState{};
    // Task.cpp:2135
    int32_t smemPageOffsetsVConsToken{int32_t{0}};
    // Task.cpp:2013
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      9,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemVProdState{
      int32_t{0},
      int32_t{1},
      int32_t{0}};
    // Task.cpp:2033
    int32_t smemVProdToken{int32_t{1}};
    // SmemKv.h:749
    int32_t smemVoteIdx5{int32_t{0}};
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
    mNumCtasKv = int32_t{
      min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, params.mMaxNumCtasKv)};
    // FmhaTask.h:630
    if ((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
      // FmhaTask.h:668
      int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
                       ((mNumCtasKv) * (int32_t{128}))};
      // FmhaTask.h:682
      numLoopSteps = numSteps;
    } else {
      // FmhaTask.h:651
      return;
    }
    // Task.cpp:3203
    bool const hasOneLoopIter{(int32_t{0}) < (numLoopSteps)};
    // Task.cpp:3214
    int32_t lastLoopOffset{int32_t{0}};
    // SmemKv.h:683
    bool skipsLoadingV5;
    //
    // Hoist the first iter.
    //
    //
    // smemPageOffsetsV [ConsWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{2}, Flags{2}].
    //
    // SmemPageOffsetsKv.h:320
    int32_t* ptrSmemPageOffsetsK7;
    // SmemPageOffsetsKv.h:326
    int32_t* ptrSmemPageOffsetsV7;
    //
    // Loop body.
    //
    // Task.cpp:3392
    CUTLASS_PRAGMA_NO_UNROLL
    for (int32_t loopOffset814 = int32_t{0}; loopOffset814 < numLoopSteps; ++loopOffset814) {
      // Task.cpp:3465
      bool const isLastLoopIter{((loopOffset814) + (int32_t{1})) >= (numLoopSteps)};
      //
      // gmemKv [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // Task.cpp:1607
      // Task.cpp:2928
      {}
      //
      // smemPageOffsetsV [ConsWait, Info{0}, FreqInfo{0, 4}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:3814
      if (((loopOffset814) % (int32_t{4})) == (int32_t{0})) {
        // Task.cpp:1607
        // Task.cpp:2816
        {
          // Task.cpp:1607
          // Task.cpp:2757
          {
            // Task.cpp:2780
            smemPageOffsetsVConsToken =
              smemPageOffsetsVSrcStack.mPipeline.consumer_try_wait(smemPageOffsetsVConsState);
          }
          // Task.cpp:2848
          smemPageOffsetsVSrcStack.mPipeline.consumer_wait(smemPageOffsetsVConsState,
                                                           smemPageOffsetsVConsToken);
        }
      }
      //
      // smemPageOffsetsV [ConsWork (call 0), Info{0}, FreqInfo{0, 4}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:3814
      if (((loopOffset814) % (int32_t{4})) == (int32_t{0})) {
        // Task.cpp:1607
        // Task.cpp:2928
        {
          // Task.cpp:5945
          int32_t index{smemPageOffsetsVConsState.index()};
          // SmemPageOffsetsKv.h:349
          ptrSmemPageOffsetsV7 = smemPageOffsetsVSrcSmem.mArray[index];
          // Task.cpp:43
          ++smemPageOffsetsVConsState;
        }
      }
      //
      // smemV [ProdAcquire, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1607
      // Task.cpp:5064
      {
        // Task.cpp:5078
        if ((loopOffset814) >= (int32_t{0})) {
          // Task.cpp:5100
          smemVProdToken = smemVDstStack.mPipeline.producer_try_acquire(smemVProdState);
        }
      }
      // Task.cpp:1607
      // Task.cpp:4288
      {
        // Task.cpp:4318
        smemVDstStack.mPipeline.producer_acquire(smemVProdState, smemVProdToken);
      }
      //
      // smemV [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // SmemKv.h:786
      int32_t* ptrSmemPageOffsetsV5;
      // Task.cpp:1511
      ptrSmemPageOffsetsV5 = ptrSmemPageOffsetsV7;
      // Task.cpp:1607
      // Task.cpp:5154
      {
        // Task.cpp:4413
        uint64_t* barrier{smemVDstStack.mPipeline.producer_get_barrier(smemVProdState)};
        // Task.cpp:5945
        int32_t index{smemVProdState.index()};
        // SmemKv.h:631
        int32_t adjustedGlobalTileIdx{(mCtaIdxKv) * (numLoopSteps) + (loopOffset814)};
        // SmemKv.h:1430
        int32_t headDimOffset{int32_t{0}};
        // SmemKv.h:1555
        int32_t tokenOffset{int32_t{0}};
        // SmemKv.h:1307
        trtllm::dev::CutlassNamedBarrier::sync(160, (int32_t{5}) + (smemVoteIdx5));
        // SmemKv.h:1316
        int32_t voteVal;
        // SmemKv.h:1317
        voteVal = smemVDstStack.mDepSmemPtr3[smemVoteIdx5];
        // SmemKv.h:1318
        voteVal = uint32_t{__shfl_sync(uint32_t{0xffffffff}, voteVal, int32_t{0}, int32_t{32})};
        // SmemKv.h:1327
        skipsLoadingV5 = (voteVal) == (int32_t{1});
        // SmemKv.h:1334
        smemVoteIdx5 = ((smemVoteIdx5) + (int32_t{1})) % (int32_t{2});
        // SmemKv.h:1342
        if (skipsLoadingV5) {
          // SmemKv.h:1346
          if (cute::elect_one_sync()) {
            // SmemKv.h:1349
            trtllm::dev::completeTransaction(barrier, int32_t{8192});
          }
        } else {
          //
          // Load pageOffsets for headDimStageIdx = 0.
          //
          // SmemKv.h:1695
          cutlass::AlignedArray<int32_t, 4> localPageOffsets05;
          // SmemKv.h:1711
          localPageOffsets05 = reinterpret_cast<cutlass::AlignedArray<int32_t, 4>*>(
            (ptrSmemPageOffsetsV5 + ((loopOffset814) * (int32_t{8})) % (int32_t{32})))[int32_t{0}];
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
            coords[int32_t{3}] = localPageOffsets05[int32_t{0}];
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemVDstSmem.mArray[index][int32_t{0}],
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
            coords[int32_t{3}] = localPageOffsets05[int32_t{1}];
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemVDstSmem.mArray[index][int32_t{1024}],
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
            coords[int32_t{3}] = localPageOffsets05[int32_t{2}];
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemVDstSmem.mArray[index][int32_t{2048}],
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
            coords[int32_t{3}] = localPageOffsets05[int32_t{3}];
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemVDstSmem.mArray[index][int32_t{3072}],
                                             &params.tmaV_,
                                             coords,
                                             barrier);
            }
          }
          //
          // Load pageOffsets for headDimStageIdx = 0.
          //
          // SmemKv.h:1695
          cutlass::AlignedArray<int32_t, 4> localPageOffsets15;
          // SmemKv.h:1711
          localPageOffsets15 = reinterpret_cast<cutlass::AlignedArray<int32_t, 4>*>(
            (ptrSmemPageOffsetsV5 +
             (((loopOffset814) * (int32_t{8})) + (int32_t{4})) % (int32_t{32})))[int32_t{0}];
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
            coords[int32_t{3}] = localPageOffsets15[int32_t{0}];
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemVDstSmem.mArray[index][int32_t{4096}],
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
            coords[int32_t{3}] = localPageOffsets15[int32_t{1}];
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemVDstSmem.mArray[index][int32_t{5120}],
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
            coords[int32_t{3}] = localPageOffsets15[int32_t{2}];
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemVDstSmem.mArray[index][int32_t{6144}],
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
            coords[int32_t{3}] = localPageOffsets15[int32_t{3}];
            // SmemTile.cpp:611
            if (bool{cute::elect_one_sync()}) {
              // CudaPtx.h:48
              cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                             cuda_ptx::space_global_t{},
                                             &smemVDstSmem.mArray[index][int32_t{7168}],
                                             &params.tmaV_,
                                             coords,
                                             barrier);
            }
          }
        }
      }
      //
      // smemV [ProdCommit, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:4522
      {
        // Task.cpp:4540
        {
          // Task.cpp:4556
          smemVDstStack.mPipeline.producer_commit(smemVProdState);
        }
        // Task.cpp:43
        ++smemVProdState;
      }
      //
      // smemPageOffsetsV [ConsRelease, Info{0}, FreqInfo{0, 4}, UserTags{2}, Flags{0}].
      //
      // Task.cpp:3814
      if ((((loopOffset814) % (int32_t{4})) == (int32_t{3})) ||
          ((loopOffset814) == ((numLoopSteps) - (int32_t{1})))) {
        // Task.cpp:2568
        if ((loopOffset814) >= (int32_t{0})) {
          // Task.cpp:2596
          {
            // Task.cpp:2620
            smemPageOffsetsVSrcStack.mPipeline.consumer_release(smemPageOffsetsVConsReleaseState);
          }
          // Task.cpp:43
          ++smemPageOffsetsVConsReleaseState;
        }
      }
      // Task.cpp:3499
      lastLoopOffset = loopOffset814;
    }
  //
  // Pull the last iter down.
  //
  //
  // Tail work.
  //
  // Task.cpp:3553
  ExitTileWithSignalingLabel:
  // Task.cpp:3560
  ExitTileWithoutSignalingLabel:
    // Task.cpp:3570
    {}
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
    , // FmhaTask.h:516
    mCtaIdxQ{(mCtaIdxX) / (params.mMaxNumCtasKv)}
    , // FmhaTask.h:517
    mCtaIdxKv{(mCtaIdxX) % (params.mMaxNumCtasKv)}
    , // FmhaTask.h:437
    mSeqLenKv{int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                            ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                            : (mBatchIdx)]}}
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
                                 TmemS0Stack& tmemS0SrcStack) {
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
    mNumCtasKv = int32_t{
      min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, params.mMaxNumCtasKv)};
    // FmhaTask.h:630
    if ((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
      // FmhaTask.h:668
      int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
                       ((mNumCtasKv) * (int32_t{128}))};
      // FmhaTask.h:682
      numLoopSteps = numSteps;
    } else {
      // FmhaTask.h:651
      return;
    }
    // Task.cpp:3203
    bool const hasOneLoopIter{(int32_t{0}) < (numLoopSteps)};
    // TmemS.h:654
    float oldMaxArray14[8];
    // TmemS.h:660
    float
      sumArray14[8]{float{0}, float{0}, float{0}, float{0}, float{0}, float{0}, float{0}, float{0}};
    // TmemS.h:672
    float newMaxArray14[8]{float{-3.4028235e+38},
                           float{-3.4028235e+38},
                           float{-3.4028235e+38},
                           float{-3.4028235e+38},
                           float{-3.4028235e+38},
                           float{-3.4028235e+38},
                           float{-3.4028235e+38},
                           float{-3.4028235e+38}};
    // TmemTile.cpp:373
    cutlass::Array<float, 32> regsQk;
    // TmemS.h:1361
    uint32_t uint32NegFltMax14{trtllm::dev::floatToUInt32ForAtomicMax(float{-3.4028235e+38})};
    // TmemS.h:1374
    CUTLASS_PRAGMA_UNROLL
    for (int32_t loopOffset1021 = mWarpGrpThreadIdx; loopOffset1021 < int32_t{32};
         loopOffset1021 += int32_t{128}) {
      // TmemS.h:1381
      reinterpret_cast<uint32_t*>(tmemS0SrcStack.mDepSmemPtr9)[loopOffset1021] = uint32NegFltMax14;
    }
    // TmemS.h:724
    trtllm::dev::CutlassNamedBarrier::sync(128, tmemS0SrcStack.mNamedBarId);
    // TmemS.h:746
    float adjustedSkipSoftmaxThreshold{(params.mSkipSoftmaxThresholdScaleFactor) /
                                       (static_cast<float>(mSeqLenKv))};
    // TmemS.h:763
    cudaGridDependencySynchronize();
    // TmemS.h:770
    float scaleSoftmaxLog214;
    // TmemS.h:775
    scaleSoftmaxLog214 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
                           ? (params.mScaleSoftmaxLog2)
                           : (float{params.ptrScaleSoftmaxLog2[int32_t{0}]});
    // TmemSoftmax.h:515
    cudaGridDependencySynchronize();
    // TmemSoftmax.h:524
    float scaleSoftmaxLog216;
    // TmemSoftmax.h:529
    scaleSoftmaxLog216 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
                           ? (params.mScaleSoftmaxLog2)
                           : (float{params.ptrScaleSoftmaxLog2[int32_t{0}]});
    // TmemP.h:521
    uint32_t regsP[8];
    // TmemP.h:534
    cudaGridDependencySynchronize();
    // TmemP.h:541
    float scaleSoftmaxLog218;
    // TmemP.h:546
    scaleSoftmaxLog218 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
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
    for (int32_t loopOffset1051 = int32_t{0}; loopOffset1051 < numLoopSteps; ++loopOffset1051) {
      // Task.cpp:3465
      bool const isLastLoopIter{((loopOffset1051) + (int32_t{1})) >= (numLoopSteps)};
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
      float* oldMaxPtr14;
      // TmemS.h:1065
      float* sumPtr14;
      // TmemS.h:1070
      float* newMaxPtr014;
      // TmemS.h:1075
      float* qkPtr014;
      // TmemS.h:1080
      float* newMaxPtr114;
      // TmemS.h:1085
      float* qkPtr114;
      // TmemS.h:1102
      bool warpSkipsSoftmax14;
      // Task.cpp:1607
      // Task.cpp:2928
      {
        // Task.cpp:5945
        int32_t index{tmemS0ConsState.index()};
        // TmemS.h:1192
        oldMaxPtr14 = oldMaxArray14;
        // TmemS.h:1194
        sumPtr14 = sumArray14;
        // TmemS.h:1196
        newMaxPtr014 = newMaxArray14;
        // TmemS.h:1198
        newMaxPtr114 = newMaxArray14;
        // TmemS.h:1207
        float reducedMaxArray14[8];
        // TmemS.h:1214
        reducedMaxArray14[int32_t{0}] = float{-3.4028235e+38};
        // TmemS.h:1214
        reducedMaxArray14[int32_t{1}] = float{-3.4028235e+38};
        // TmemS.h:1214
        reducedMaxArray14[int32_t{2}] = float{-3.4028235e+38};
        // TmemS.h:1214
        reducedMaxArray14[int32_t{3}] = float{-3.4028235e+38};
        // TmemS.h:1214
        reducedMaxArray14[int32_t{4}] = float{-3.4028235e+38};
        // TmemS.h:1214
        reducedMaxArray14[int32_t{5}] = float{-3.4028235e+38};
        // TmemS.h:1214
        reducedMaxArray14[int32_t{6}] = float{-3.4028235e+38};
        // TmemS.h:1214
        reducedMaxArray14[int32_t{7}] = float{-3.4028235e+38};
        // TmemS.h:1246
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset1082 = int32_t{0}; loopOffset1082 < int32_t{8}; ++loopOffset1082) {
          // TmemS.h:1257
          oldMaxArray14[loopOffset1082] = newMaxArray14[loopOffset1082];
        }
        //
        // The causal mask block.
        //
        // Mask.h:568
        int32_t const tileOffsetK{
          (((numLoopSteps) * (mCtaIdxKv) + (loopOffset1051)) * (int32_t{1})) * (int32_t{128})};
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
            uint32_t(&dstSlice0)[16]{reinterpret_cast<uint32_t(&)[16]>(regsQk[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_16x256b(
              dstSlice0,
              (tmemBasePtr) +
                (static_cast<uint32_t>((index) * (int32_t{32}) +
                                       (int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                          ? (int32_t{0})
                                          : (int32_t{32})))));
            // TmemTile.cpp:545
            uint32_t(&dstSlice1)[16]{reinterpret_cast<uint32_t(&)[16]>(regsQk[int32_t{16}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_16x256b(
              dstSlice1,
              (tmemBasePtr) +
                (static_cast<uint32_t>(
                  ((index) * (int32_t{32}) + (int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                                ? (int32_t{0})
                                                : (int32_t{32}))) +
                  (int32_t{0x100000 /*hi=16, lo=0*/}))));
          }
          // Utils.h:248
          trtllm::dev::reduceColMax16dp256bit<int32_t{2}, int32_t{1}, int32_t{4}, false>(
            reducedMaxArray14,
            regsQk);
          // Utils.h:260
          trtllm::dev::reduceColMax(
            reducedMaxArray14,
            (tmemS0SrcStack.mDepSmemPtr9 + ((loopOffset1051) % (int32_t{2})) * (int32_t{128})),
            int32_t{128},
            mWarpGrpThreadIdx,
            tmemS0SrcStack.mNamedBarId);
        } else {
          // TmemTile.cpp:527
          {
            // TmemTile.cpp:529
            uint32_t tmemBasePtr{mTmemBaseOffset};
            // TmemTile.cpp:545
            uint32_t(&dstSlice0)[16]{reinterpret_cast<uint32_t(&)[16]>(regsQk[int32_t{0}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_16x256b(
              dstSlice0,
              (tmemBasePtr) +
                (static_cast<uint32_t>((index) * (int32_t{32}) +
                                       (int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                          ? (int32_t{0})
                                          : (int32_t{32})))));
            // TmemTile.cpp:545
            uint32_t(&dstSlice1)[16]{reinterpret_cast<uint32_t(&)[16]>(regsQk[int32_t{16}])};
            // CudaPtx.h:48
            cuda_ptx::tcgen05_ld_16x256b(
              dstSlice1,
              (tmemBasePtr) +
                (static_cast<uint32_t>(
                  ((index) * (int32_t{32}) + (int32_t((tmemS0SrcStack.mInstId) == (int32_t{0}))
                                                ? (int32_t{0})
                                                : (int32_t{32}))) +
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
            (((numLoopSteps) * (mCtaIdxKv) + (loopOffset1051)) * (int32_t{1})) * (int32_t{128})};
          // Mask.h:1206
          int32_t startIdxInWindow0;
          // Mask.h:1206
          int32_t startIdxInWindow1;
          // Mask.h:1206
          int32_t startIdxInWindow2;
          // Mask.h:1206
          int32_t startIdxInWindow3;
          // Mask.h:1206
          int32_t startIdxInWindow4;
          // Mask.h:1206
          int32_t startIdxInWindow5;
          // Mask.h:1206
          int32_t startIdxInWindow6;
          // Mask.h:1206
          int32_t startIdxInWindow7;
          // Mask.h:1225
          int32_t localTokenIdxQ0{(mLdtm16dp256bitTmemColIdx) / (params.mNumHeadsQPerKvDivisor)};
          // Mask.h:1239
          int32_t const idxQ0{min((tileOffsetQ) + (localTokenIdxQ0), (mSeqLenKv) - (int32_t{1}))};
          // Mask.h:1225
          int32_t localTokenIdxQ1{((mLdtm16dp256bitTmemColIdx) + (int32_t{1})) /
                                  (params.mNumHeadsQPerKvDivisor)};
          // Mask.h:1239
          int32_t const idxQ1{min((tileOffsetQ) + (localTokenIdxQ1), (mSeqLenKv) - (int32_t{1}))};
          // Mask.h:1225
          int32_t localTokenIdxQ2{((mLdtm16dp256bitTmemColIdx) + (int32_t{8})) /
                                  (params.mNumHeadsQPerKvDivisor)};
          // Mask.h:1239
          int32_t const idxQ2{min((tileOffsetQ) + (localTokenIdxQ2), (mSeqLenKv) - (int32_t{1}))};
          // Mask.h:1225
          int32_t localTokenIdxQ3{((mLdtm16dp256bitTmemColIdx) + (int32_t{9})) /
                                  (params.mNumHeadsQPerKvDivisor)};
          // Mask.h:1239
          int32_t const idxQ3{min((tileOffsetQ) + (localTokenIdxQ3), (mSeqLenKv) - (int32_t{1}))};
          // Mask.h:1225
          int32_t localTokenIdxQ4{((mLdtm16dp256bitTmemColIdx) + (int32_t{16})) /
                                  (params.mNumHeadsQPerKvDivisor)};
          // Mask.h:1239
          int32_t const idxQ4{min((tileOffsetQ) + (localTokenIdxQ4), (mSeqLenKv) - (int32_t{1}))};
          // Mask.h:1225
          int32_t localTokenIdxQ5{((mLdtm16dp256bitTmemColIdx) + (int32_t{17})) /
                                  (params.mNumHeadsQPerKvDivisor)};
          // Mask.h:1239
          int32_t const idxQ5{min((tileOffsetQ) + (localTokenIdxQ5), (mSeqLenKv) - (int32_t{1}))};
          // Mask.h:1225
          int32_t localTokenIdxQ6{((mLdtm16dp256bitTmemColIdx) + (int32_t{24})) /
                                  (params.mNumHeadsQPerKvDivisor)};
          // Mask.h:1239
          int32_t const idxQ6{min((tileOffsetQ) + (localTokenIdxQ6), (mSeqLenKv) - (int32_t{1}))};
          // Mask.h:1225
          int32_t localTokenIdxQ7{((mLdtm16dp256bitTmemColIdx) + (int32_t{25})) /
                                  (params.mNumHeadsQPerKvDivisor)};
          // Mask.h:1239
          int32_t const idxQ7{min((tileOffsetQ) + (localTokenIdxQ7), (mSeqLenKv) - (int32_t{1}))};
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
            regsQk[int32_t{16}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK3) > (idxQ0)) {
            // Mask.h:1315
            regsQk[int32_t{18}] = float{-3.4028235e+38};
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
            regsQk[int32_t{17}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK3) > (idxQ1)) {
            // Mask.h:1315
            regsQk[int32_t{19}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK0) > (idxQ2)) {
            // Mask.h:1315
            regsQk[int32_t{4}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK1) > (idxQ2)) {
            // Mask.h:1315
            regsQk[int32_t{6}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK2) > (idxQ2)) {
            // Mask.h:1315
            regsQk[int32_t{20}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK3) > (idxQ2)) {
            // Mask.h:1315
            regsQk[int32_t{22}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK0) > (idxQ3)) {
            // Mask.h:1315
            regsQk[int32_t{5}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK1) > (idxQ3)) {
            // Mask.h:1315
            regsQk[int32_t{7}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK2) > (idxQ3)) {
            // Mask.h:1315
            regsQk[int32_t{21}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK3) > (idxQ3)) {
            // Mask.h:1315
            regsQk[int32_t{23}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK0) > (idxQ4)) {
            // Mask.h:1315
            regsQk[int32_t{8}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK1) > (idxQ4)) {
            // Mask.h:1315
            regsQk[int32_t{10}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK2) > (idxQ4)) {
            // Mask.h:1315
            regsQk[int32_t{24}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK3) > (idxQ4)) {
            // Mask.h:1315
            regsQk[int32_t{26}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK0) > (idxQ5)) {
            // Mask.h:1315
            regsQk[int32_t{9}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK1) > (idxQ5)) {
            // Mask.h:1315
            regsQk[int32_t{11}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK2) > (idxQ5)) {
            // Mask.h:1315
            regsQk[int32_t{25}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK3) > (idxQ5)) {
            // Mask.h:1315
            regsQk[int32_t{27}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK0) > (idxQ6)) {
            // Mask.h:1315
            regsQk[int32_t{12}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK1) > (idxQ6)) {
            // Mask.h:1315
            regsQk[int32_t{14}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK2) > (idxQ6)) {
            // Mask.h:1315
            regsQk[int32_t{28}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK3) > (idxQ6)) {
            // Mask.h:1315
            regsQk[int32_t{30}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK0) > (idxQ7)) {
            // Mask.h:1315
            regsQk[int32_t{13}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK1) > (idxQ7)) {
            // Mask.h:1315
            regsQk[int32_t{15}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK2) > (idxQ7)) {
            // Mask.h:1315
            regsQk[int32_t{29}] = float{-3.4028235e+38};
          }
          // Mask.h:1311
          if ((idxK3) > (idxQ7)) {
            // Mask.h:1315
            regsQk[int32_t{31}] = float{-3.4028235e+38};
          }
          // Utils.h:248
          trtllm::dev::reduceColMax16dp256bit<int32_t{2}, int32_t{1}, int32_t{4}, false>(
            reducedMaxArray14,
            regsQk);
          // Utils.h:260
          trtllm::dev::reduceColMax(
            reducedMaxArray14,
            (tmemS0SrcStack.mDepSmemPtr9 + ((loopOffset1051) % (int32_t{2})) * (int32_t{128})),
            int32_t{128},
            mWarpGrpThreadIdx,
            tmemS0SrcStack.mNamedBarId);
        }
        // TmemS.h:587
        int32_t numSkippedStats{int32_t{0}};
        // TmemS.h:601
        if ((float{exp2f((scaleSoftmaxLog214) *
                         ((reducedMaxArray14[int32_t{0}]) - (newMaxArray14[int32_t{0}])))}) <
            (adjustedSkipSoftmaxThreshold)) {
          // TmemS.h:603
          ++numSkippedStats;
        }
        // TmemS.h:601
        if ((float{exp2f((scaleSoftmaxLog214) *
                         ((reducedMaxArray14[int32_t{1}]) - (newMaxArray14[int32_t{1}])))}) <
            (adjustedSkipSoftmaxThreshold)) {
          // TmemS.h:603
          ++numSkippedStats;
        }
        // TmemS.h:601
        if ((float{exp2f((scaleSoftmaxLog214) *
                         ((reducedMaxArray14[int32_t{2}]) - (newMaxArray14[int32_t{2}])))}) <
            (adjustedSkipSoftmaxThreshold)) {
          // TmemS.h:603
          ++numSkippedStats;
        }
        // TmemS.h:601
        if ((float{exp2f((scaleSoftmaxLog214) *
                         ((reducedMaxArray14[int32_t{3}]) - (newMaxArray14[int32_t{3}])))}) <
            (adjustedSkipSoftmaxThreshold)) {
          // TmemS.h:603
          ++numSkippedStats;
        }
        // TmemS.h:601
        if ((float{exp2f((scaleSoftmaxLog214) *
                         ((reducedMaxArray14[int32_t{4}]) - (newMaxArray14[int32_t{4}])))}) <
            (adjustedSkipSoftmaxThreshold)) {
          // TmemS.h:603
          ++numSkippedStats;
        }
        // TmemS.h:601
        if ((float{exp2f((scaleSoftmaxLog214) *
                         ((reducedMaxArray14[int32_t{5}]) - (newMaxArray14[int32_t{5}])))}) <
            (adjustedSkipSoftmaxThreshold)) {
          // TmemS.h:603
          ++numSkippedStats;
        }
        // TmemS.h:601
        if ((float{exp2f((scaleSoftmaxLog214) *
                         ((reducedMaxArray14[int32_t{6}]) - (newMaxArray14[int32_t{6}])))}) <
            (adjustedSkipSoftmaxThreshold)) {
          // TmemS.h:603
          ++numSkippedStats;
        }
        // TmemS.h:601
        if ((float{exp2f((scaleSoftmaxLog214) *
                         ((reducedMaxArray14[int32_t{7}]) - (newMaxArray14[int32_t{7}])))}) <
            (adjustedSkipSoftmaxThreshold)) {
          // TmemS.h:603
          ++numSkippedStats;
        }
        // TmemS.h:570
        bool threadCanSkip{(numSkippedStats) == (int32_t{8})};
        // TmemS.h:615
        bool warpCanSkipSoftmax{__all_sync(uint32_t{-1}, threadCanSkip)};
        // TmemS.h:2263
        warpSkipsSoftmax14 = warpCanSkipSoftmax;
        // TmemS.h:2270
        if (!(warpSkipsSoftmax14)) {
          // TmemS.h:2279
          newMaxArray14[int32_t{0}] =
            fmaxf(reducedMaxArray14[int32_t{0}], newMaxArray14[int32_t{0}]);
          // TmemS.h:2279
          newMaxArray14[int32_t{1}] =
            fmaxf(reducedMaxArray14[int32_t{1}], newMaxArray14[int32_t{1}]);
          // TmemS.h:2279
          newMaxArray14[int32_t{2}] =
            fmaxf(reducedMaxArray14[int32_t{2}], newMaxArray14[int32_t{2}]);
          // TmemS.h:2279
          newMaxArray14[int32_t{3}] =
            fmaxf(reducedMaxArray14[int32_t{3}], newMaxArray14[int32_t{3}]);
          // TmemS.h:2279
          newMaxArray14[int32_t{4}] =
            fmaxf(reducedMaxArray14[int32_t{4}], newMaxArray14[int32_t{4}]);
          // TmemS.h:2279
          newMaxArray14[int32_t{5}] =
            fmaxf(reducedMaxArray14[int32_t{5}], newMaxArray14[int32_t{5}]);
          // TmemS.h:2279
          newMaxArray14[int32_t{6}] =
            fmaxf(reducedMaxArray14[int32_t{6}], newMaxArray14[int32_t{6}]);
          // TmemS.h:2279
          newMaxArray14[int32_t{7}] =
            fmaxf(reducedMaxArray14[int32_t{7}], newMaxArray14[int32_t{7}]);
        }
        // TmemS.h:2093
        if ((mWarpGrpThreadIdx) == (int32_t{0})) {
          // TmemS.h:2123
          tmemS0SrcStack.mDepSmemPtr3[index] =
            int32_t(warpSkipsSoftmax14) ? (int32_t{1}) : (int32_t{0});
        }
        // TmemS.h:2147
        trtllm::dev::CutlassNamedBarrier::arrive(160, (int32_t{5}) + (index));
        // TmemS.h:1361
        uint32_t uint32NegFltMax14{trtllm::dev::floatToUInt32ForAtomicMax(float{-3.4028235e+38})};
        // TmemS.h:1374
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset1319 = mWarpGrpThreadIdx; loopOffset1319 < int32_t{32};
             loopOffset1319 += int32_t{128}) {
          // TmemS.h:1381
          reinterpret_cast<uint32_t*>(
            (tmemS0SrcStack.mDepSmemPtr9 + (((loopOffset1051) % (int32_t{2})) ^ (int32_t{1})) *
                                             (int32_t{128})))[loopOffset1319] = uint32NegFltMax14;
        }
        // TmemS.h:1334
        qkPtr014 = &regsQk[int32_t{0}];
        // TmemS.h:1336
        qkPtr114 = &regsQk[int32_t{0}];
        // Task.cpp:43
        ++tmemS0ConsState;
      }
      //
      // tmemSoftmaxLocal0 [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{8}, Flags{0}].
      //
      // TmemSoftmax.h:261
      float* oldMaxPtr15;
      // TmemSoftmax.h:267
      float* sumPtr15;
      // TmemSoftmax.h:273
      float* newMaxPtr15;
      // Task.cpp:1511
      oldMaxPtr15 = oldMaxPtr14;
      // Task.cpp:1511
      sumPtr15 = sumPtr14;
      // Task.cpp:1511
      newMaxPtr15 = newMaxPtr014;
      // Task.cpp:1607
      // Task.cpp:5154
      {
        // Task.cpp:5945
        int32_t index{tmemSoftmaxLocal0ProdState.index()};
        // TmemTile.cpp:373
        cutlass::Array<float, 16> stats;
        // TmemSoftmax.h:365
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset1335 = int32_t{0}; loopOffset1335 < int32_t{8}; ++loopOffset1335) {
          // TmemSoftmax.h:382
          stats[loopOffset1335] = oldMaxPtr15[loopOffset1335];
          // TmemSoftmax.h:384
          stats[(loopOffset1335) + (int32_t{8})] = newMaxPtr15[loopOffset1335];
        }
        // TmemTile.cpp:836
        {
          // TmemTile.cpp:838
          uint32_t tmemBasePtr{mTmemBaseOffset};
          // TmemTile.cpp:871
          uint32_t const(&srcSlice0)[16]{
            reinterpret_cast<uint32_t const(&)[16]>(stats[int32_t{0}])};
          // CudaPtx.h:48
          cuda_ptx::tcgen05_st_32x32b(
            (tmemBasePtr) +
              (static_cast<uint32_t>((index) * (int32_t{32}) +
                                     (int32_t((tmemSoftmaxLocal0DstStack.mInstId) == (int32_t{0}))
                                        ? (int32_t{64})
                                        : (int32_t{96})))),
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
      float* newMaxPtr18;
      // TmemP.h:574
      float* regsFp32P18;
      // TmemP.h:589
      bool warpSkipsSoftmax18;
      // Task.cpp:1511
      newMaxPtr18 = newMaxPtr114;
      // Task.cpp:1511
      regsFp32P18 = qkPtr114;
      // Task.cpp:1511
      warpSkipsSoftmax18 = warpSkipsSoftmax14;
      // Task.cpp:1607
      // Task.cpp:5154
      {
        // Task.cpp:5945
        int32_t index{tmemP0ProdState.index()};
        // TmemP.h:1181
        if (warpSkipsSoftmax18) {
          // TmemP.h:2350
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset1367 = int32_t{0}; loopOffset1367 < int32_t{8}; ++loopOffset1367) {
            // TmemP.h:2354
            regsP[loopOffset1367] = int32_t{0};
          }
        } else {
          // TmemP.h:1025
          float negScaledMaxArray[8];
          // TmemP.h:1043
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset1371 = int32_t{0}; loopOffset1371 < int32_t{8};
               loopOffset1371 += int32_t{2}) {
            // TmemP.h:1054
            float newMax0{newMaxPtr18[loopOffset1371]};
            // TmemP.h:1060
            float newMax1{newMaxPtr18[(loopOffset1371) + (int32_t{1})]};
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
            float negLog2Scale{-(scaleSoftmaxLog218)};
            // Common.h:353
            cutlass::Array<float, 2> negLog2Scale2{negLog2Scale, negLog2Scale};
            // Common.h:353
            cutlass::Array<float, 2> log2E4m3Scale2{float{8.8073549}, float{8.8073549}};
            // TmemP.h:1104
            newMax2 = trtllm::dev::ffma2(newMax2, negLog2Scale2, log2E4m3Scale2);
            // TmemP.h:1115
            negScaledMaxArray[loopOffset1371] = newMax2[int32_t{0}];
            // TmemP.h:1116
            negScaledMaxArray[(loopOffset1371) + (int32_t{1})] = newMax2[int32_t{1}];
          }
          // TmemP.h:1655
          {
            // TmemP.h:1658
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog218, scaleSoftmaxLog218};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{1}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P18[int32_t{0}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P18[int32_t{1}];
            // TmemP.h:801
            vals[int32_t{0}] =
              (log2Scale2[int32_t{0}]) * (vals[int32_t{0}]) + (negScaledMax[int32_t{0}]);
            // TmemP.h:810
            vals[int32_t{1}] =
              (log2Scale2[int32_t{1}]) * (vals[int32_t{1}]) + (negScaledMax[int32_t{1}]);
            // TmemP.h:833
            regsFp32P18[int32_t{0}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P18[int32_t{1}] = vals[int32_t{1}];
          }
          // TmemP.h:1655
          {
            // TmemP.h:1658
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog218, scaleSoftmaxLog218};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{1}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P18[int32_t{2}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P18[int32_t{3}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P18[int32_t{2}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P18[int32_t{3}] = vals[int32_t{1}];
          }
          // TmemP.h:1773
          regsFp32P18[int32_t{0}] = exp2f(regsFp32P18[int32_t{0}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog218, scaleSoftmaxLog218};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{2}],
                                                  negScaledMaxArray[int32_t{3}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P18[int32_t{4}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P18[int32_t{5}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P18[int32_t{4}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P18[int32_t{5}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P18[int32_t{1}] = exp2f(regsFp32P18[int32_t{1}]);
          // TmemP.h:1773
          regsFp32P18[int32_t{2}] = exp2f(regsFp32P18[int32_t{2}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog218, scaleSoftmaxLog218};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{2}],
                                                  negScaledMaxArray[int32_t{3}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P18[int32_t{6}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P18[int32_t{7}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P18[int32_t{6}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P18[int32_t{7}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P18[int32_t{3}] = exp2f(regsFp32P18[int32_t{3}]);
          // TmemP.h:1773
          regsFp32P18[int32_t{4}] = exp2f(regsFp32P18[int32_t{4}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog218, scaleSoftmaxLog218};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{4}],
                                                  negScaledMaxArray[int32_t{5}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P18[int32_t{8}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P18[int32_t{9}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P18[int32_t{8}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P18[int32_t{9}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P18[int32_t{5}] = exp2f(regsFp32P18[int32_t{5}]);
          // TmemP.h:1773
          regsFp32P18[int32_t{6}] = exp2f(regsFp32P18[int32_t{6}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog218, scaleSoftmaxLog218};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{4}],
                                                  negScaledMaxArray[int32_t{5}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P18[int32_t{10}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P18[int32_t{11}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P18[int32_t{10}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P18[int32_t{11}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P18[int32_t{7}] = exp2f(regsFp32P18[int32_t{7}]);
          // TmemP.h:1773
          regsFp32P18[int32_t{8}] = exp2f(regsFp32P18[int32_t{8}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog218, scaleSoftmaxLog218};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{6}],
                                                  negScaledMaxArray[int32_t{7}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P18[int32_t{12}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P18[int32_t{13}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P18[int32_t{12}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P18[int32_t{13}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P18[int32_t{9}] = exp2f(regsFp32P18[int32_t{9}]);
          // TmemP.h:1773
          regsFp32P18[int32_t{10}] = exp2f(regsFp32P18[int32_t{10}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog218, scaleSoftmaxLog218};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{6}],
                                                  negScaledMaxArray[int32_t{7}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P18[int32_t{14}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P18[int32_t{15}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P18[int32_t{14}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P18[int32_t{15}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P18[int32_t{11}] = exp2f(regsFp32P18[int32_t{11}]);
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
            elt0 = regsFp32P18[int32_t{0}];
            // TmemP.h:721
            elt1 = regsFp32P18[int32_t{1}];
            // TmemP.h:722
            elt2 = regsFp32P18[int32_t{2}];
            // TmemP.h:723
            elt3 = regsFp32P18[int32_t{3}];
            // TmemP.h:745
            regsP[int32_t{0}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P18[int32_t{12}] = exp2f(regsFp32P18[int32_t{12}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog218, scaleSoftmaxLog218};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{1}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P18[int32_t{16}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P18[int32_t{17}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P18[int32_t{16}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P18[int32_t{17}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P18[int32_t{13}] = exp2f(regsFp32P18[int32_t{13}]);
          // TmemP.h:1773
          regsFp32P18[int32_t{14}] = exp2f(regsFp32P18[int32_t{14}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog218, scaleSoftmaxLog218};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{0}],
                                                  negScaledMaxArray[int32_t{1}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P18[int32_t{18}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P18[int32_t{19}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P18[int32_t{18}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P18[int32_t{19}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P18[int32_t{15}] = exp2f(regsFp32P18[int32_t{15}]);
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
            elt0 = regsFp32P18[int32_t{4}];
            // TmemP.h:721
            elt1 = regsFp32P18[int32_t{5}];
            // TmemP.h:722
            elt2 = regsFp32P18[int32_t{6}];
            // TmemP.h:723
            elt3 = regsFp32P18[int32_t{7}];
            // TmemP.h:745
            regsP[int32_t{1}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P18[int32_t{16}] = exp2f(regsFp32P18[int32_t{16}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog218, scaleSoftmaxLog218};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{2}],
                                                  negScaledMaxArray[int32_t{3}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P18[int32_t{20}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P18[int32_t{21}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P18[int32_t{20}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P18[int32_t{21}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P18[int32_t{17}] = exp2f(regsFp32P18[int32_t{17}]);
          // TmemP.h:1773
          regsFp32P18[int32_t{18}] = exp2f(regsFp32P18[int32_t{18}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog218, scaleSoftmaxLog218};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{2}],
                                                  negScaledMaxArray[int32_t{3}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P18[int32_t{22}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P18[int32_t{23}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P18[int32_t{22}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P18[int32_t{23}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P18[int32_t{19}] = exp2f(regsFp32P18[int32_t{19}]);
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
            elt0 = regsFp32P18[int32_t{8}];
            // TmemP.h:721
            elt1 = regsFp32P18[int32_t{9}];
            // TmemP.h:722
            elt2 = regsFp32P18[int32_t{10}];
            // TmemP.h:723
            elt3 = regsFp32P18[int32_t{11}];
            // TmemP.h:745
            regsP[int32_t{2}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P18[int32_t{20}] = exp2f(regsFp32P18[int32_t{20}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog218, scaleSoftmaxLog218};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{4}],
                                                  negScaledMaxArray[int32_t{5}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P18[int32_t{24}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P18[int32_t{25}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P18[int32_t{24}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P18[int32_t{25}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P18[int32_t{21}] = exp2f(regsFp32P18[int32_t{21}]);
          // TmemP.h:1773
          regsFp32P18[int32_t{22}] = exp2f(regsFp32P18[int32_t{22}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog218, scaleSoftmaxLog218};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{4}],
                                                  negScaledMaxArray[int32_t{5}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P18[int32_t{26}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P18[int32_t{27}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P18[int32_t{26}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P18[int32_t{27}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P18[int32_t{23}] = exp2f(regsFp32P18[int32_t{23}]);
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
            elt0 = regsFp32P18[int32_t{12}];
            // TmemP.h:721
            elt1 = regsFp32P18[int32_t{13}];
            // TmemP.h:722
            elt2 = regsFp32P18[int32_t{14}];
            // TmemP.h:723
            elt3 = regsFp32P18[int32_t{15}];
            // TmemP.h:745
            regsP[int32_t{3}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P18[int32_t{24}] = exp2f(regsFp32P18[int32_t{24}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog218, scaleSoftmaxLog218};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{6}],
                                                  negScaledMaxArray[int32_t{7}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P18[int32_t{28}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P18[int32_t{29}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P18[int32_t{28}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P18[int32_t{29}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P18[int32_t{25}] = exp2f(regsFp32P18[int32_t{25}]);
          // TmemP.h:1773
          regsFp32P18[int32_t{26}] = exp2f(regsFp32P18[int32_t{26}]);
          // TmemP.h:1802
          {
            // TmemP.h:1806
            cutlass::Array<float, 2> log2Scale2{scaleSoftmaxLog218, scaleSoftmaxLog218};
            // TmemP.h:764
            cutlass::Array<float, 2> vals;
            // TmemP.h:783
            cutlass::Array<float, 2> negScaledMax{negScaledMaxArray[int32_t{6}],
                                                  negScaledMaxArray[int32_t{7}]};
            // TmemP.h:792
            vals[int32_t{0}] = regsFp32P18[int32_t{30}];
            // TmemP.h:793
            vals[int32_t{1}] = regsFp32P18[int32_t{31}];
            // TmemP.h:826
            vals = trtllm::dev::fadd2(trtllm::dev::fmul2(log2Scale2, vals), negScaledMax);
            // TmemP.h:833
            regsFp32P18[int32_t{30}] = vals[int32_t{0}];
            // TmemP.h:834
            regsFp32P18[int32_t{31}] = vals[int32_t{1}];
          }
          // TmemP.h:1843
          regsFp32P18[int32_t{27}] = exp2f(regsFp32P18[int32_t{27}]);
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
            elt0 = regsFp32P18[int32_t{16}];
            // TmemP.h:721
            elt1 = regsFp32P18[int32_t{17}];
            // TmemP.h:722
            elt2 = regsFp32P18[int32_t{18}];
            // TmemP.h:723
            elt3 = regsFp32P18[int32_t{19}];
            // TmemP.h:745
            regsP[int32_t{4}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
          // TmemP.h:1773
          regsFp32P18[int32_t{28}] = exp2f(regsFp32P18[int32_t{28}]);
          // TmemP.h:1843
          regsFp32P18[int32_t{29}] = exp2f(regsFp32P18[int32_t{29}]);
          // TmemP.h:1773
          regsFp32P18[int32_t{30}] = exp2f(regsFp32P18[int32_t{30}]);
          // TmemP.h:1843
          regsFp32P18[int32_t{31}] = exp2f(regsFp32P18[int32_t{31}]);
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
            elt0 = regsFp32P18[int32_t{20}];
            // TmemP.h:721
            elt1 = regsFp32P18[int32_t{21}];
            // TmemP.h:722
            elt2 = regsFp32P18[int32_t{22}];
            // TmemP.h:723
            elt3 = regsFp32P18[int32_t{23}];
            // TmemP.h:745
            regsP[int32_t{5}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
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
            elt0 = regsFp32P18[int32_t{24}];
            // TmemP.h:721
            elt1 = regsFp32P18[int32_t{25}];
            // TmemP.h:722
            elt2 = regsFp32P18[int32_t{26}];
            // TmemP.h:723
            elt3 = regsFp32P18[int32_t{27}];
            // TmemP.h:745
            regsP[int32_t{6}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
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
            elt0 = regsFp32P18[int32_t{28}];
            // TmemP.h:721
            elt1 = regsFp32P18[int32_t{29}];
            // TmemP.h:722
            elt2 = regsFp32P18[int32_t{30}];
            // TmemP.h:723
            elt3 = regsFp32P18[int32_t{31}];
            // TmemP.h:745
            regsP[int32_t{7}] = trtllm::dev::convert_float4_to_e4m3(elt0, elt1, elt2, elt3);
          }
        }
        // TmemP.h:1255
        cutlass::float_e4m3_t* smemPtrP18;
        // TmemP.h:1257
        smemPtrP18 = reinterpret_cast<cutlass::float_e4m3_t*>(tmemP0DstStack.mDepSmemPtr8) +
                     (((tmemP0DstStack.mInstId) + (index)) * (int32_t{4096}));
        // TmemP.h:1278
        trtllm::dev::storeTransposedSmem8b<int32_t{32}, int32_t{128}>(smemPtrP18,
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
        if ((loopOffset1051) >= (int32_t{0})) {
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
      if ((loopOffset1051) >= (int32_t{0})) {
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
      float* oldMaxPtr16;
      // TmemSoftmax.h:552
      float* sumPtr16;
      // TmemSoftmax.h:559
      float* newMaxPtr16;
      // TmemSoftmax.h:566
      float* pPtr16;
      // TmemSoftmax.h:573
      bool warpSkipsSums16;
      // Task.cpp:1511
      oldMaxPtr16 = oldMaxPtr14;
      // Task.cpp:1511
      sumPtr16 = sumPtr14;
      // Task.cpp:1511
      newMaxPtr16 = newMaxPtr014;
      // Task.cpp:1511
      pPtr16 = qkPtr014;
      // Task.cpp:1511
      warpSkipsSums16 = warpSkipsSoftmax14;
      // Task.cpp:1607
      // Task.cpp:5154
      {
        // TmemSoftmax.h:610
        if (!(warpSkipsSums16)) {
          // TmemSoftmax.h:1010
          {
            // Common.h:395
            cutlass::Array<float, 2> oldMax{float{oldMaxPtr16[int32_t{0}]},
                                            float{oldMaxPtr16[int32_t{1}]}};
            // Common.h:395
            cutlass::Array<float, 2> newMax{float{newMaxPtr16[int32_t{0}]},
                                            float{newMaxPtr16[int32_t{1}]}};
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
            // TmemSoftmax.h:1029
            cutlass::Array<float, 2> p0{pPtr16[int32_t{0}], pPtr16[int32_t{1}]};
            // Common.h:395
            cutlass::Array<float, 2> sum{float{sumPtr16[int32_t{0}]}, float{sumPtr16[int32_t{1}]}};
            // TmemSoftmax.h:1048
            sum = trtllm::dev::ffma2(scale2, sum, p0);
            // TmemSoftmax.h:1060
            cutlass::Array<float, 2> p1{pPtr16[int32_t{2}], pPtr16[int32_t{3}]};
            // TmemSoftmax.h:1076
            sum = trtllm::dev::fadd2(sum, p1);
            // TmemSoftmax.h:1083
            sumPtr16[int32_t{0}] = sum[int32_t{0}];
            // TmemSoftmax.h:1084
            sumPtr16[int32_t{1}] = sum[int32_t{1}];
            // TmemSoftmax.h:1060
            cutlass::Array<float, 2> p2{pPtr16[int32_t{16}], pPtr16[int32_t{17}]};
            // TmemSoftmax.h:1076
            sum = trtllm::dev::fadd2(sum, p2);
            // TmemSoftmax.h:1083
            sumPtr16[int32_t{0}] = sum[int32_t{0}];
            // TmemSoftmax.h:1084
            sumPtr16[int32_t{1}] = sum[int32_t{1}];
            // TmemSoftmax.h:1060
            cutlass::Array<float, 2> p3{pPtr16[int32_t{18}], pPtr16[int32_t{19}]};
            // TmemSoftmax.h:1076
            sum = trtllm::dev::fadd2(sum, p3);
            // TmemSoftmax.h:1083
            sumPtr16[int32_t{0}] = sum[int32_t{0}];
            // TmemSoftmax.h:1084
            sumPtr16[int32_t{1}] = sum[int32_t{1}];
          }
          // TmemSoftmax.h:1010
          {
            // Common.h:395
            cutlass::Array<float, 2> oldMax{float{oldMaxPtr16[int32_t{2}]},
                                            float{oldMaxPtr16[int32_t{3}]}};
            // Common.h:395
            cutlass::Array<float, 2> newMax{float{newMaxPtr16[int32_t{2}]},
                                            float{newMaxPtr16[int32_t{3}]}};
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
            // TmemSoftmax.h:1029
            cutlass::Array<float, 2> p0{pPtr16[int32_t{4}], pPtr16[int32_t{5}]};
            // Common.h:395
            cutlass::Array<float, 2> sum{float{sumPtr16[int32_t{2}]}, float{sumPtr16[int32_t{3}]}};
            // TmemSoftmax.h:1048
            sum = trtllm::dev::ffma2(scale2, sum, p0);
            // TmemSoftmax.h:1060
            cutlass::Array<float, 2> p1{pPtr16[int32_t{6}], pPtr16[int32_t{7}]};
            // TmemSoftmax.h:1076
            sum = trtllm::dev::fadd2(sum, p1);
            // TmemSoftmax.h:1083
            sumPtr16[int32_t{2}] = sum[int32_t{0}];
            // TmemSoftmax.h:1084
            sumPtr16[int32_t{3}] = sum[int32_t{1}];
            // TmemSoftmax.h:1060
            cutlass::Array<float, 2> p2{pPtr16[int32_t{20}], pPtr16[int32_t{21}]};
            // TmemSoftmax.h:1076
            sum = trtllm::dev::fadd2(sum, p2);
            // TmemSoftmax.h:1083
            sumPtr16[int32_t{2}] = sum[int32_t{0}];
            // TmemSoftmax.h:1084
            sumPtr16[int32_t{3}] = sum[int32_t{1}];
            // TmemSoftmax.h:1060
            cutlass::Array<float, 2> p3{pPtr16[int32_t{22}], pPtr16[int32_t{23}]};
            // TmemSoftmax.h:1076
            sum = trtllm::dev::fadd2(sum, p3);
            // TmemSoftmax.h:1083
            sumPtr16[int32_t{2}] = sum[int32_t{0}];
            // TmemSoftmax.h:1084
            sumPtr16[int32_t{3}] = sum[int32_t{1}];
          }
          // TmemSoftmax.h:1010
          {
            // Common.h:395
            cutlass::Array<float, 2> oldMax{float{oldMaxPtr16[int32_t{4}]},
                                            float{oldMaxPtr16[int32_t{5}]}};
            // Common.h:395
            cutlass::Array<float, 2> newMax{float{newMaxPtr16[int32_t{4}]},
                                            float{newMaxPtr16[int32_t{5}]}};
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
            // TmemSoftmax.h:1029
            cutlass::Array<float, 2> p0{pPtr16[int32_t{8}], pPtr16[int32_t{9}]};
            // Common.h:395
            cutlass::Array<float, 2> sum{float{sumPtr16[int32_t{4}]}, float{sumPtr16[int32_t{5}]}};
            // TmemSoftmax.h:1048
            sum = trtllm::dev::ffma2(scale2, sum, p0);
            // TmemSoftmax.h:1060
            cutlass::Array<float, 2> p1{pPtr16[int32_t{10}], pPtr16[int32_t{11}]};
            // TmemSoftmax.h:1076
            sum = trtllm::dev::fadd2(sum, p1);
            // TmemSoftmax.h:1083
            sumPtr16[int32_t{4}] = sum[int32_t{0}];
            // TmemSoftmax.h:1084
            sumPtr16[int32_t{5}] = sum[int32_t{1}];
            // TmemSoftmax.h:1060
            cutlass::Array<float, 2> p2{pPtr16[int32_t{24}], pPtr16[int32_t{25}]};
            // TmemSoftmax.h:1076
            sum = trtllm::dev::fadd2(sum, p2);
            // TmemSoftmax.h:1083
            sumPtr16[int32_t{4}] = sum[int32_t{0}];
            // TmemSoftmax.h:1084
            sumPtr16[int32_t{5}] = sum[int32_t{1}];
            // TmemSoftmax.h:1060
            cutlass::Array<float, 2> p3{pPtr16[int32_t{26}], pPtr16[int32_t{27}]};
            // TmemSoftmax.h:1076
            sum = trtllm::dev::fadd2(sum, p3);
            // TmemSoftmax.h:1083
            sumPtr16[int32_t{4}] = sum[int32_t{0}];
            // TmemSoftmax.h:1084
            sumPtr16[int32_t{5}] = sum[int32_t{1}];
          }
          // TmemSoftmax.h:1010
          {
            // Common.h:395
            cutlass::Array<float, 2> oldMax{float{oldMaxPtr16[int32_t{6}]},
                                            float{oldMaxPtr16[int32_t{7}]}};
            // Common.h:395
            cutlass::Array<float, 2> newMax{float{newMaxPtr16[int32_t{6}]},
                                            float{newMaxPtr16[int32_t{7}]}};
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
            // TmemSoftmax.h:1029
            cutlass::Array<float, 2> p0{pPtr16[int32_t{12}], pPtr16[int32_t{13}]};
            // Common.h:395
            cutlass::Array<float, 2> sum{float{sumPtr16[int32_t{6}]}, float{sumPtr16[int32_t{7}]}};
            // TmemSoftmax.h:1048
            sum = trtllm::dev::ffma2(scale2, sum, p0);
            // TmemSoftmax.h:1060
            cutlass::Array<float, 2> p1{pPtr16[int32_t{14}], pPtr16[int32_t{15}]};
            // TmemSoftmax.h:1076
            sum = trtllm::dev::fadd2(sum, p1);
            // TmemSoftmax.h:1083
            sumPtr16[int32_t{6}] = sum[int32_t{0}];
            // TmemSoftmax.h:1084
            sumPtr16[int32_t{7}] = sum[int32_t{1}];
            // TmemSoftmax.h:1060
            cutlass::Array<float, 2> p2{pPtr16[int32_t{28}], pPtr16[int32_t{29}]};
            // TmemSoftmax.h:1076
            sum = trtllm::dev::fadd2(sum, p2);
            // TmemSoftmax.h:1083
            sumPtr16[int32_t{6}] = sum[int32_t{0}];
            // TmemSoftmax.h:1084
            sumPtr16[int32_t{7}] = sum[int32_t{1}];
            // TmemSoftmax.h:1060
            cutlass::Array<float, 2> p3{pPtr16[int32_t{30}], pPtr16[int32_t{31}]};
            // TmemSoftmax.h:1076
            sum = trtllm::dev::fadd2(sum, p3);
            // TmemSoftmax.h:1083
            sumPtr16[int32_t{6}] = sum[int32_t{0}];
            // TmemSoftmax.h:1084
            sumPtr16[int32_t{7}] = sum[int32_t{1}];
          }
        }
      }
      //
      // tmemSoftmaxLocal0 [ProdWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{10}, Flags{0}].
      //
      // Task.cpp:1511
      oldMaxPtr15 = oldMaxPtr14;
      // Task.cpp:1511
      sumPtr15 = sumPtr14;
      // Task.cpp:1511
      newMaxPtr15 = newMaxPtr014;
      // Task.cpp:1607
      if (isLastLoopIter) {
        // Task.cpp:5945
        int32_t index{tmemSoftmaxLocal0ProdState.index()};
        // TmemTile.cpp:373
        cutlass::Array<float, 16> stats;
        // TmemSoftmax.h:365
        CUTLASS_PRAGMA_UNROLL
        for (int32_t loopOffset1817 = int32_t{0}; loopOffset1817 < int32_t{8}; ++loopOffset1817) {
          // TmemSoftmax.h:382
          stats[loopOffset1817] = sumPtr15[loopOffset1817];
          // TmemSoftmax.h:384
          stats[(loopOffset1817) + (int32_t{8})] = newMaxPtr15[loopOffset1817];
        }
        // TmemTile.cpp:836
        {
          // TmemTile.cpp:838
          uint32_t tmemBasePtr{mTmemBaseOffset};
          // TmemTile.cpp:871
          uint32_t const(&srcSlice0)[16]{
            reinterpret_cast<uint32_t const(&)[16]>(stats[int32_t{0}])};
          // CudaPtx.h:48
          cuda_ptx::tcgen05_st_32x32b(
            (tmemBasePtr) +
              (static_cast<uint32_t>((index) * (int32_t{32}) +
                                     (int32_t((tmemSoftmaxLocal0DstStack.mInstId) == (int32_t{0}))
                                        ? (int32_t{64})
                                        : (int32_t{96})))),
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
        if ((loopOffset1051) >= (int32_t{0})) {
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
    // Task.cpp:3570
    {}
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
  // Task.cpp:266
  int32_t const mGridDimY;
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
    , // FmhaTask.h:516
    mCtaIdxQ{(mCtaIdxX) / (params.mMaxNumCtasKv)}
    , // FmhaTask.h:517
    mCtaIdxKv{(mCtaIdxX) % (params.mMaxNumCtasKv)}
    , // FmhaTask.h:437
    mSeqLenKv{int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                            ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                            : (mBatchIdx)]}}
    , // FmhaTask.h:565
    mNumCtasKv{
      int32_t{min(int32_t{((mSeqLenKv) + (int32_t{127})) / (int32_t{128})}, params.mMaxNumCtasKv)}}
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
      trtllm::dev::ldst16dp256bitTmemRowIdx<int32_t{16}>((mWarpGrpThreadIdx) % (int32_t{128}))}
    , // Kernel.cpp:192
    mGridDimY{reinterpret_cast<int32_t const&>(gridDim.y)} {}
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
                                 TmemOStack& tmemOSrcStack) {
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
    mNumCtasKv = int32_t{
      min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, params.mMaxNumCtasKv)};
    // FmhaTask.h:630
    if ((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
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
    cutlass::Array<float, 16> frgStats15;
    // TmemCorr.h:1135
    cudaGridDependencySynchronize();
    // TmemCorr.h:1158
    float scaleSoftmaxLog220;
    // TmemCorr.h:1163
    scaleSoftmaxLog220 = float(bool{params.ptrScaleSoftmaxLog2 == nullptr})
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
    for (int32_t loopOffset1923 = int32_t{0}; loopOffset1923 < (numLoopSteps) - (int32_t{1});
         ++loopOffset1923) {
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
      float* statsPtr115;
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
          uint32_t(&dstSlice0)[16]{reinterpret_cast<uint32_t(&)[16]>(frgStats15[int32_t{0}])};
          // CudaPtx.h:48
          cuda_ptx::tcgen05_ld_32x32b(
            dstSlice0,
            (tmemBasePtr) +
              (static_cast<uint32_t>((index) * (int32_t{32}) +
                                     (int32_t((tmemSoftmaxLocal0SrcStack.mInstId) == (int32_t{0}))
                                        ? (int32_t{64})
                                        : (int32_t{96})))));
        }
        // TmemSoftmax.h:327
        statsPtr115 = &frgStats15[int32_t{0}];
        // TmemSoftmax.h:330
        cutlass::arch::fence_view_async_tmem_load();
        // Task.cpp:43
        ++tmemSoftmaxLocal0ConsState;
      }
      //
      // tmemSoftmaxLocal0 [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{1024}].
      //
      // Task.cpp:2568
      if ((loopOffset1923) >= (int32_t{0})) {
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
      float* prodStatsPtr020;
      // Task.cpp:1511
      prodStatsPtr020 = statsPtr115;
      // Task.cpp:1607
      // Task.cpp:5154
      {
        // TmemCorr.h:425
        cutlass::Array<float, 8> scales20;
        // TmemCorr.h:437
        {
          // Common.h:353
          cutlass::Array<float, 2> oldMax{float{prodStatsPtr020[int32_t{0}]},
                                          float{prodStatsPtr020[int32_t{1}]}};
          // Common.h:353
          cutlass::Array<float, 2> newMax{float{prodStatsPtr020[int32_t{8}]},
                                          float{prodStatsPtr020[int32_t{9}]}};
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
            cutlass::Array<float, 2> scale2_{scaleSoftmaxLog220, scaleSoftmaxLog220};
            // Common.h:161
            scale2 = trtllm::dev::fmul2(scale2_, maxDiff2);
            // Common.h:168
            scale2[int32_t{0}] = exp2f(scale2[int32_t{0}]);
            // Common.h:169
            scale2[int32_t{1}] = exp2f(scale2[int32_t{1}]);
          }
          // TmemCorr.h:473
          scales20[int32_t{0}] = scale2[int32_t{0}];
          // TmemCorr.h:474
          scales20[int32_t{1}] = scale2[int32_t{1}];
        }
        // TmemCorr.h:437
        {
          // Common.h:353
          cutlass::Array<float, 2> oldMax{float{prodStatsPtr020[int32_t{2}]},
                                          float{prodStatsPtr020[int32_t{3}]}};
          // Common.h:353
          cutlass::Array<float, 2> newMax{float{prodStatsPtr020[int32_t{10}]},
                                          float{prodStatsPtr020[int32_t{11}]}};
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
            cutlass::Array<float, 2> scale2_{scaleSoftmaxLog220, scaleSoftmaxLog220};
            // Common.h:161
            scale2 = trtllm::dev::fmul2(scale2_, maxDiff2);
            // Common.h:168
            scale2[int32_t{0}] = exp2f(scale2[int32_t{0}]);
            // Common.h:169
            scale2[int32_t{1}] = exp2f(scale2[int32_t{1}]);
          }
          // TmemCorr.h:473
          scales20[int32_t{2}] = scale2[int32_t{0}];
          // TmemCorr.h:474
          scales20[int32_t{3}] = scale2[int32_t{1}];
        }
        // TmemCorr.h:437
        {
          // Common.h:353
          cutlass::Array<float, 2> oldMax{float{prodStatsPtr020[int32_t{4}]},
                                          float{prodStatsPtr020[int32_t{5}]}};
          // Common.h:353
          cutlass::Array<float, 2> newMax{float{prodStatsPtr020[int32_t{12}]},
                                          float{prodStatsPtr020[int32_t{13}]}};
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
            cutlass::Array<float, 2> scale2_{scaleSoftmaxLog220, scaleSoftmaxLog220};
            // Common.h:161
            scale2 = trtllm::dev::fmul2(scale2_, maxDiff2);
            // Common.h:168
            scale2[int32_t{0}] = exp2f(scale2[int32_t{0}]);
            // Common.h:169
            scale2[int32_t{1}] = exp2f(scale2[int32_t{1}]);
          }
          // TmemCorr.h:473
          scales20[int32_t{4}] = scale2[int32_t{0}];
          // TmemCorr.h:474
          scales20[int32_t{5}] = scale2[int32_t{1}];
        }
        // TmemCorr.h:437
        {
          // Common.h:353
          cutlass::Array<float, 2> oldMax{float{prodStatsPtr020[int32_t{6}]},
                                          float{prodStatsPtr020[int32_t{7}]}};
          // Common.h:353
          cutlass::Array<float, 2> newMax{float{prodStatsPtr020[int32_t{14}]},
                                          float{prodStatsPtr020[int32_t{15}]}};
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
            cutlass::Array<float, 2> scale2_{scaleSoftmaxLog220, scaleSoftmaxLog220};
            // Common.h:161
            scale2 = trtllm::dev::fmul2(scale2_, maxDiff2);
            // Common.h:168
            scale2[int32_t{0}] = exp2f(scale2[int32_t{0}]);
            // Common.h:169
            scale2[int32_t{1}] = exp2f(scale2[int32_t{1}]);
          }
          // TmemCorr.h:473
          scales20[int32_t{6}] = scale2[int32_t{0}];
          // TmemCorr.h:474
          scales20[int32_t{7}] = scale2[int32_t{1}];
        }
        // TmemCorr.h:1240
        bool skipsCorr{true};
        // TmemCorr.h:1258
        skipsCorr = (skipsCorr) && ((scales20[int32_t{0}]) == (float{1}));
        // TmemCorr.h:1258
        skipsCorr = (skipsCorr) && ((scales20[int32_t{1}]) == (float{1}));
        // TmemCorr.h:1258
        skipsCorr = (skipsCorr) && ((scales20[int32_t{2}]) == (float{1}));
        // TmemCorr.h:1258
        skipsCorr = (skipsCorr) && ((scales20[int32_t{3}]) == (float{1}));
        // TmemCorr.h:1258
        skipsCorr = (skipsCorr) && ((scales20[int32_t{4}]) == (float{1}));
        // TmemCorr.h:1258
        skipsCorr = (skipsCorr) && ((scales20[int32_t{5}]) == (float{1}));
        // TmemCorr.h:1258
        skipsCorr = (skipsCorr) && ((scales20[int32_t{6}]) == (float{1}));
        // TmemCorr.h:1258
        skipsCorr = (skipsCorr) && ((scales20[int32_t{7}]) == (float{1}));
        // TmemCorr.h:1266
        skipsCorr = __all_sync(uint32_t{-1}, skipsCorr);
        // TmemCorr.h:1268
        if (!(skipsCorr)) {
          //
          // The headDimStageIdx: 0.
          //
          // TmemCorr.h:1486
          CUTLASS_PRAGMA_UNROLL
          for (int32_t loopOffset2042 = int32_t{0}; loopOffset2042 < int32_t{16};
               loopOffset2042 += int32_t{32}) {
            // TmemTile.cpp:373
            cutlass::Array<float, 16> tmemRegs020;
            // TmemTile.cpp:527
            {
              // TmemTile.cpp:529
              uint32_t tmemBasePtr{mTmemBaseOffset};
              // TmemTile.cpp:545
              uint32_t(&dstSlice0)[16]{reinterpret_cast<uint32_t(&)[16]>(tmemRegs020[int32_t{0}])};
              // CudaPtx.h:48
              cuda_ptx::tcgen05_ld_16x256b(
                dstSlice0,
                (tmemBasePtr) +
                  (static_cast<uint32_t>((int32_t{0x80 /*hi=0, lo=128*/}) + (loopOffset2042))));
            }
            // TmemCorr.h:1534
            {
              // TmemCorr.h:1554
              cutlass::Array<float, 2> localScales0{scales20[int32_t{0}], scales20[int32_t{1}]};
              // TmemCorr.h:1565
              cutlass::Array<float, 2> vals0{tmemRegs020[int32_t{0}], tmemRegs020[int32_t{1}]};
              // TmemCorr.h:1577
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1580
              tmemRegs020[int32_t{0}] = vals0[int32_t{0}];
              // TmemCorr.h:1581
              tmemRegs020[int32_t{1}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1534
            {
              // TmemCorr.h:1554
              cutlass::Array<float, 2> localScales0{scales20[int32_t{0}], scales20[int32_t{1}]};
              // TmemCorr.h:1565
              cutlass::Array<float, 2> vals0{tmemRegs020[int32_t{2}], tmemRegs020[int32_t{3}]};
              // TmemCorr.h:1577
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1580
              tmemRegs020[int32_t{2}] = vals0[int32_t{0}];
              // TmemCorr.h:1581
              tmemRegs020[int32_t{3}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1534
            {
              // TmemCorr.h:1554
              cutlass::Array<float, 2> localScales0{scales20[int32_t{2}], scales20[int32_t{3}]};
              // TmemCorr.h:1565
              cutlass::Array<float, 2> vals0{tmemRegs020[int32_t{4}], tmemRegs020[int32_t{5}]};
              // TmemCorr.h:1577
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1580
              tmemRegs020[int32_t{4}] = vals0[int32_t{0}];
              // TmemCorr.h:1581
              tmemRegs020[int32_t{5}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1534
            {
              // TmemCorr.h:1554
              cutlass::Array<float, 2> localScales0{scales20[int32_t{2}], scales20[int32_t{3}]};
              // TmemCorr.h:1565
              cutlass::Array<float, 2> vals0{tmemRegs020[int32_t{6}], tmemRegs020[int32_t{7}]};
              // TmemCorr.h:1577
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1580
              tmemRegs020[int32_t{6}] = vals0[int32_t{0}];
              // TmemCorr.h:1581
              tmemRegs020[int32_t{7}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1534
            {
              // TmemCorr.h:1554
              cutlass::Array<float, 2> localScales0{scales20[int32_t{4}], scales20[int32_t{5}]};
              // TmemCorr.h:1565
              cutlass::Array<float, 2> vals0{tmemRegs020[int32_t{8}], tmemRegs020[int32_t{9}]};
              // TmemCorr.h:1577
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1580
              tmemRegs020[int32_t{8}] = vals0[int32_t{0}];
              // TmemCorr.h:1581
              tmemRegs020[int32_t{9}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1534
            {
              // TmemCorr.h:1554
              cutlass::Array<float, 2> localScales0{scales20[int32_t{4}], scales20[int32_t{5}]};
              // TmemCorr.h:1565
              cutlass::Array<float, 2> vals0{tmemRegs020[int32_t{10}], tmemRegs020[int32_t{11}]};
              // TmemCorr.h:1577
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1580
              tmemRegs020[int32_t{10}] = vals0[int32_t{0}];
              // TmemCorr.h:1581
              tmemRegs020[int32_t{11}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1534
            {
              // TmemCorr.h:1554
              cutlass::Array<float, 2> localScales0{scales20[int32_t{6}], scales20[int32_t{7}]};
              // TmemCorr.h:1565
              cutlass::Array<float, 2> vals0{tmemRegs020[int32_t{12}], tmemRegs020[int32_t{13}]};
              // TmemCorr.h:1577
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1580
              tmemRegs020[int32_t{12}] = vals0[int32_t{0}];
              // TmemCorr.h:1581
              tmemRegs020[int32_t{13}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1534
            {
              // TmemCorr.h:1554
              cutlass::Array<float, 2> localScales0{scales20[int32_t{6}], scales20[int32_t{7}]};
              // TmemCorr.h:1565
              cutlass::Array<float, 2> vals0{tmemRegs020[int32_t{14}], tmemRegs020[int32_t{15}]};
              // TmemCorr.h:1577
              vals0 = trtllm::dev::fmul2(vals0, localScales0);
              // TmemCorr.h:1580
              tmemRegs020[int32_t{14}] = vals0[int32_t{0}];
              // TmemCorr.h:1581
              tmemRegs020[int32_t{15}] = vals0[int32_t{1}];
            }
            // TmemCorr.h:1997
            {
              // TmemTile.cpp:836
              {
                // TmemTile.cpp:838
                uint32_t tmemBasePtr{mTmemBaseOffset};
                // TmemTile.cpp:871
                uint32_t const(&srcSlice0)[16]{
                  reinterpret_cast<uint32_t const(&)[16]>(tmemRegs020[int32_t{0}])};
                // CudaPtx.h:48
                cuda_ptx::tcgen05_st_16x256b(
                  (tmemBasePtr) +
                    (static_cast<uint32_t>((int32_t{0x80 /*hi=0, lo=128*/}) + (loopOffset2042))),
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
      if ((loopOffset1923) >= (int32_t{0})) {
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
      lastLoopOffset = loopOffset1923;
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
    float* statsPtr215;
    // Task.cpp:1607
    if (hasOneLoopIter) {
      // Task.cpp:5945
      int32_t index{tmemSoftmaxLocal0ConsState.index()};
      // TmemTile.cpp:527
      {
        // TmemTile.cpp:529
        uint32_t tmemBasePtr{mTmemBaseOffset};
        // TmemTile.cpp:545
        uint32_t(&dstSlice0)[16]{reinterpret_cast<uint32_t(&)[16]>(frgStats15[int32_t{0}])};
        // CudaPtx.h:48
        cuda_ptx::tcgen05_ld_32x32b(
          dstSlice0,
          (tmemBasePtr) +
            (static_cast<uint32_t>((index) * (int32_t{32}) +
                                   (int32_t((tmemSoftmaxLocal0SrcStack.mInstId) == (int32_t{0}))
                                      ? (int32_t{64})
                                      : (int32_t{96})))));
      }
      // TmemSoftmax.h:327
      statsPtr215 = &frgStats15[int32_t{0}];
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
    float* prodStatsPtr120;
    // Task.cpp:1511
    prodStatsPtr120 = statsPtr215;
    // Task.cpp:1607
    if (hasOneLoopIter) {
      // TmemCorr.h:2860
      int32_t instIdxO{(mCtaIdxQ) * (int32_t{1})};
      // TmemCorr.h:2907
      int32_t numValidRowsO{(int32_t{min(params.mNumTokensPerCtaQ,
                                         (mSeqLenQ) - ((instIdxO) * (params.mNumTokensPerCtaQ)))}) *
                            (params.mNumHeadsQPerKv)};
      // TmemCorr.h:2984
      int32_t seqOffsetO{(mSeqOffsetQ) + ((instIdxO) * (params.mNumTokensPerCtaQ))};
      // TmemCorr.h:2989
      int32_t headIdxO;
      // TmemCorr.h:2993
      headIdxO = (mHeadIdx) * (int32_t{min(params.mNumHeadsQPerKv, int32_t{32})});
      // TmemCorr.h:2996
      int32_t headOffsetO{((mHeadIdx) * (int32_t{min(params.mNumHeadsQPerKv, int32_t{32})})) *
                          (int32_t{64})};
      // TmemCorr.h:3027
      int64_t ctaOffsetO{(static_cast<int64_t>(seqOffsetO)) *
                           (static_cast<int64_t>((params.mNumHeadsQ) * (int32_t{64}))) +
                         (static_cast<int64_t>(headOffsetO))};
      // TmemCorr.h:3041
      cutlass::float_e2m1_t* ptrO{reinterpret_cast<cutlass::float_e2m1_t*>(params.ptrO)};
      // TmemCorr.h:3046
      ptrO = ptrO + ((ctaOffsetO) / (int64_t{2}));
      // TmemCorr.h:3076
      bool storesSoftmaxStats{reinterpret_cast<float*>(params.ptrSoftmaxStats) != nullptr};
      // TmemCorr.h:3082
      float* ptrSoftmaxStats;
      // TmemCorr.h:3084
      if (storesSoftmaxStats) {
        // TmemCorr.h:3088
        ptrSoftmaxStats = reinterpret_cast<float*>(params.ptrSoftmaxStats) +
                          (((seqOffsetO) * (params.mNumHeadsQ) +
                            ((mHeadIdx) * (int32_t{min(params.mNumHeadsQPerKv, int32_t{32})}))) *
                           (int32_t{2}));
      }
      // TmemCorr.h:3103
      float mScaleInvSf{trtllm::dev::reciprocal_approximate_ftz(
        float(bool{params.ptrScaleSfO == nullptr}) ? (params.mScaleSfO)
                                                   : (float{params.ptrScaleSfO[int32_t{0}]}))};
      // TmemCorr.h:3119
      int32_t sfRowIdx{(seqOffsetO) + (params.mStartTokenIdxSfO)};
      // TmemCorr.h:3145
      int64_t baseSfOffset{trtllm::dev::getSfOffset(
        int32_t{0},
        ((mHeadIdx) * (int32_t{min(params.mNumHeadsQPerKv, int32_t{32})})) * (int32_t{4}),
        (params.mNumHeadsQ) * (int32_t{4}))};
      // TmemCorr.h:3151
      cutlass::float_e4m3_t* ptrSfO;
      // TmemCorr.h:3153
      ptrSfO = reinterpret_cast<cutlass::float_e4m3_t*>(params.ptrSfO) + (baseSfOffset);
      // TmemCorr.h:2475
      int32_t numValidSlices{((numValidRowsO) + (int32_t{15})) / (int32_t{16})};
      // TmemCorr.h:2482
      int32_t numSlicesPerCta{((numValidSlices) + ((mNumCtasKv) - (int32_t{1}))) / (mNumCtasKv)};
      // TmemCorr.h:2490
      int32_t numCtasKvForReduction{((numValidSlices) + ((numSlicesPerCta) - (int32_t{1}))) /
                                    (numSlicesPerCta)};
      // TmemCorr.h:2498
      int32_t numReductionRowsPerCta{(numSlicesPerCta) * (int32_t{16})};
      // TmemCorr.h:2519
      int32_t numValidRowsInCta{
        min(numReductionRowsPerCta, (numValidRowsO) - ((mCtaIdxKv) * (numReductionRowsPerCta)))};
      // TmemCorr.h:2530
      int32_t completionBytes{(int32_t{34816}) -
                              (((numValidRowsInCta) * (mNumCtasKv)) * (int32_t{136}))};
      // TmemCorr.h:2688
      cutlass::half_t* ptrPartialSmemO{
        reinterpret_cast<cutlass::half_t*>(tmemCorr0DstStack.mDepSmemPtr13)};
      // TmemCorr.h:2695
      cutlass::half_t* ptrPartialCtaSmemO{
        (ptrPartialSmemO + ((mCtaIdxKv) * (numReductionRowsPerCta)) * (int32_t{64}))};
      // TmemCorr.h:2708
      float* ptrPartialSmemStats{reinterpret_cast<float*>(
        (ptrPartialSmemO + ((mNumCtasKv) * (numReductionRowsPerCta)) * (int32_t{64})))};
      // TmemCorr.h:2717
      float* ptrPartialCtaSmemStats{
        (ptrPartialSmemStats + ((mCtaIdxKv) * (numReductionRowsPerCta)) * (int32_t{2}))};
      // TmemCorr.h:425
      cutlass::Array<float, 8> scales20;
      // TmemCorr.h:1983
      trtllm::dev::reduceColSum<int32_t{8}>(prodStatsPtr120,
                                            tmemCorr0DstStack.mDepSmemPtr11,
                                            int32_t{4},
                                            int32_t{128},
                                            mWarpGrpThreadIdx,
                                            int32_t{4});
      // TmemCorr.h:509
      {
        // TmemCorr.h:518
        float sum0{prodStatsPtr120[int32_t{0}]};
        // TmemCorr.h:524
        float sum1{prodStatsPtr120[int32_t{1}]};
        // TmemCorr.h:591
        prodStatsPtr120[int32_t{0}] = sum0;
        // TmemCorr.h:592
        prodStatsPtr120[int32_t{1}] = sum1;
        // TmemCorr.h:599
        scales20[int32_t{0}] = (float(bool{params.ptrOutputScale == nullptr})
                                  ? (params.mOutputScale)
                                  : (float{params.ptrOutputScale[int32_t{0}]})) /
                               (float{448});
        // TmemCorr.h:600
        scales20[int32_t{1}] = (float(bool{params.ptrOutputScale == nullptr})
                                  ? (params.mOutputScale)
                                  : (float{params.ptrOutputScale[int32_t{0}]})) /
                               (float{448});
      }
      // TmemCorr.h:509
      {
        // TmemCorr.h:518
        float sum0{prodStatsPtr120[int32_t{2}]};
        // TmemCorr.h:524
        float sum1{prodStatsPtr120[int32_t{3}]};
        // TmemCorr.h:591
        prodStatsPtr120[int32_t{2}] = sum0;
        // TmemCorr.h:592
        prodStatsPtr120[int32_t{3}] = sum1;
        // TmemCorr.h:599
        scales20[int32_t{2}] = (float(bool{params.ptrOutputScale == nullptr})
                                  ? (params.mOutputScale)
                                  : (float{params.ptrOutputScale[int32_t{0}]})) /
                               (float{448});
        // TmemCorr.h:600
        scales20[int32_t{3}] = (float(bool{params.ptrOutputScale == nullptr})
                                  ? (params.mOutputScale)
                                  : (float{params.ptrOutputScale[int32_t{0}]})) /
                               (float{448});
      }
      // TmemCorr.h:509
      {
        // TmemCorr.h:518
        float sum0{prodStatsPtr120[int32_t{4}]};
        // TmemCorr.h:524
        float sum1{prodStatsPtr120[int32_t{5}]};
        // TmemCorr.h:591
        prodStatsPtr120[int32_t{4}] = sum0;
        // TmemCorr.h:592
        prodStatsPtr120[int32_t{5}] = sum1;
        // TmemCorr.h:599
        scales20[int32_t{4}] = (float(bool{params.ptrOutputScale == nullptr})
                                  ? (params.mOutputScale)
                                  : (float{params.ptrOutputScale[int32_t{0}]})) /
                               (float{448});
        // TmemCorr.h:600
        scales20[int32_t{5}] = (float(bool{params.ptrOutputScale == nullptr})
                                  ? (params.mOutputScale)
                                  : (float{params.ptrOutputScale[int32_t{0}]})) /
                               (float{448});
      }
      // TmemCorr.h:509
      {
        // TmemCorr.h:518
        float sum0{prodStatsPtr120[int32_t{6}]};
        // TmemCorr.h:524
        float sum1{prodStatsPtr120[int32_t{7}]};
        // TmemCorr.h:591
        prodStatsPtr120[int32_t{6}] = sum0;
        // TmemCorr.h:592
        prodStatsPtr120[int32_t{7}] = sum1;
        // TmemCorr.h:599
        scales20[int32_t{6}] = (float(bool{params.ptrOutputScale == nullptr})
                                  ? (params.mOutputScale)
                                  : (float{params.ptrOutputScale[int32_t{0}]})) /
                               (float{448});
        // TmemCorr.h:600
        scales20[int32_t{7}] = (float(bool{params.ptrOutputScale == nullptr})
                                  ? (params.mOutputScale)
                                  : (float{params.ptrOutputScale[int32_t{0}]})) /
                               (float{448});
      }
      // TmemCorr.h:3981
      trtllm::dev::storeStatsForSwappedAb<int32_t{8}, false>((prodStatsPtr120 + int32_t{8}),
                                                             prodStatsPtr120,
                                                             ptrPartialCtaSmemStats,
                                                             tmemCorr0DstStack.mClusterBarrierPtr12,
                                                             numReductionRowsPerCta,
                                                             mWarpGrpThreadIdx,
                                                             numValidRowsO);
      //
      // The headDimStageIdx: 0.
      //
      // TmemCorr.h:1486
      for (int32_t loopOffset2233 = int32_t{0}; loopOffset2233 < int32_t{16};
           loopOffset2233 += int32_t{32}) {
        // TmemTile.cpp:373
        cutlass::Array<float, 16> tmemRegs020;
        // TmemTile.cpp:527
        {
          // TmemTile.cpp:529
          uint32_t tmemBasePtr{mTmemBaseOffset};
          // TmemTile.cpp:545
          uint32_t(&dstSlice0)[16]{reinterpret_cast<uint32_t(&)[16]>(tmemRegs020[int32_t{0}])};
          // CudaPtx.h:48
          cuda_ptx::tcgen05_ld_16x256b(
            dstSlice0,
            (tmemBasePtr) +
              (static_cast<uint32_t>((int32_t{0x80 /*hi=0, lo=128*/}) + (loopOffset2233))));
        }
        // TmemCorr.h:3438
        uint32_t mRegsO20[8];
        // TmemCorr.h:1534
        {
          // TmemCorr.h:1554
          cutlass::Array<float, 2> localScales0{scales20[int32_t{0}], scales20[int32_t{1}]};
          // TmemCorr.h:1565
          cutlass::Array<float, 2> vals0{tmemRegs020[int32_t{0}], tmemRegs020[int32_t{1}]};
          // TmemCorr.h:1577
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1580
          tmemRegs020[int32_t{0}] = vals0[int32_t{0}];
          // TmemCorr.h:1581
          tmemRegs020[int32_t{1}] = vals0[int32_t{1}];
          // TmemCorr.h:3664
          mRegsO20[int32_t{0}] =
            trtllm::dev::convert_float2_to_half(tmemRegs020[int32_t{0}], tmemRegs020[int32_t{1}]);
        }
        // TmemCorr.h:1534
        {
          // TmemCorr.h:1554
          cutlass::Array<float, 2> localScales0{scales20[int32_t{0}], scales20[int32_t{1}]};
          // TmemCorr.h:1565
          cutlass::Array<float, 2> vals0{tmemRegs020[int32_t{2}], tmemRegs020[int32_t{3}]};
          // TmemCorr.h:1577
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1580
          tmemRegs020[int32_t{2}] = vals0[int32_t{0}];
          // TmemCorr.h:1581
          tmemRegs020[int32_t{3}] = vals0[int32_t{1}];
          // TmemCorr.h:3664
          mRegsO20[int32_t{1}] =
            trtllm::dev::convert_float2_to_half(tmemRegs020[int32_t{2}], tmemRegs020[int32_t{3}]);
        }
        // TmemCorr.h:1534
        {
          // TmemCorr.h:1554
          cutlass::Array<float, 2> localScales0{scales20[int32_t{2}], scales20[int32_t{3}]};
          // TmemCorr.h:1565
          cutlass::Array<float, 2> vals0{tmemRegs020[int32_t{4}], tmemRegs020[int32_t{5}]};
          // TmemCorr.h:1577
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1580
          tmemRegs020[int32_t{4}] = vals0[int32_t{0}];
          // TmemCorr.h:1581
          tmemRegs020[int32_t{5}] = vals0[int32_t{1}];
          // TmemCorr.h:3664
          mRegsO20[int32_t{2}] =
            trtllm::dev::convert_float2_to_half(tmemRegs020[int32_t{4}], tmemRegs020[int32_t{5}]);
        }
        // TmemCorr.h:1534
        {
          // TmemCorr.h:1554
          cutlass::Array<float, 2> localScales0{scales20[int32_t{2}], scales20[int32_t{3}]};
          // TmemCorr.h:1565
          cutlass::Array<float, 2> vals0{tmemRegs020[int32_t{6}], tmemRegs020[int32_t{7}]};
          // TmemCorr.h:1577
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1580
          tmemRegs020[int32_t{6}] = vals0[int32_t{0}];
          // TmemCorr.h:1581
          tmemRegs020[int32_t{7}] = vals0[int32_t{1}];
          // TmemCorr.h:3664
          mRegsO20[int32_t{3}] =
            trtllm::dev::convert_float2_to_half(tmemRegs020[int32_t{6}], tmemRegs020[int32_t{7}]);
        }
        // TmemCorr.h:1534
        {
          // TmemCorr.h:1554
          cutlass::Array<float, 2> localScales0{scales20[int32_t{4}], scales20[int32_t{5}]};
          // TmemCorr.h:1565
          cutlass::Array<float, 2> vals0{tmemRegs020[int32_t{8}], tmemRegs020[int32_t{9}]};
          // TmemCorr.h:1577
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1580
          tmemRegs020[int32_t{8}] = vals0[int32_t{0}];
          // TmemCorr.h:1581
          tmemRegs020[int32_t{9}] = vals0[int32_t{1}];
          // TmemCorr.h:3664
          mRegsO20[int32_t{4}] =
            trtllm::dev::convert_float2_to_half(tmemRegs020[int32_t{8}], tmemRegs020[int32_t{9}]);
        }
        // TmemCorr.h:1534
        {
          // TmemCorr.h:1554
          cutlass::Array<float, 2> localScales0{scales20[int32_t{4}], scales20[int32_t{5}]};
          // TmemCorr.h:1565
          cutlass::Array<float, 2> vals0{tmemRegs020[int32_t{10}], tmemRegs020[int32_t{11}]};
          // TmemCorr.h:1577
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1580
          tmemRegs020[int32_t{10}] = vals0[int32_t{0}];
          // TmemCorr.h:1581
          tmemRegs020[int32_t{11}] = vals0[int32_t{1}];
          // TmemCorr.h:3664
          mRegsO20[int32_t{5}] =
            trtllm::dev::convert_float2_to_half(tmemRegs020[int32_t{10}], tmemRegs020[int32_t{11}]);
        }
        // TmemCorr.h:1534
        {
          // TmemCorr.h:1554
          cutlass::Array<float, 2> localScales0{scales20[int32_t{6}], scales20[int32_t{7}]};
          // TmemCorr.h:1565
          cutlass::Array<float, 2> vals0{tmemRegs020[int32_t{12}], tmemRegs020[int32_t{13}]};
          // TmemCorr.h:1577
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1580
          tmemRegs020[int32_t{12}] = vals0[int32_t{0}];
          // TmemCorr.h:1581
          tmemRegs020[int32_t{13}] = vals0[int32_t{1}];
          // TmemCorr.h:3664
          mRegsO20[int32_t{6}] =
            trtllm::dev::convert_float2_to_half(tmemRegs020[int32_t{12}], tmemRegs020[int32_t{13}]);
        }
        // TmemCorr.h:1534
        {
          // TmemCorr.h:1554
          cutlass::Array<float, 2> localScales0{scales20[int32_t{6}], scales20[int32_t{7}]};
          // TmemCorr.h:1565
          cutlass::Array<float, 2> vals0{tmemRegs020[int32_t{14}], tmemRegs020[int32_t{15}]};
          // TmemCorr.h:1577
          vals0 = trtllm::dev::fmul2(vals0, localScales0);
          // TmemCorr.h:1580
          tmemRegs020[int32_t{14}] = vals0[int32_t{0}];
          // TmemCorr.h:1581
          tmemRegs020[int32_t{15}] = vals0[int32_t{1}];
          // TmemCorr.h:3664
          mRegsO20[int32_t{7}] =
            trtllm::dev::convert_float2_to_half(tmemRegs020[int32_t{14}], tmemRegs020[int32_t{15}]);
        }
        // TmemCorr.h:3849
        trtllm::dev::reorganizeInSmemAndStoreToDstMem<int32_t{64}, int32_t{32}, false>(
          reinterpret_cast<cutlass::half_t*>(tmemCorr0DstStack.mDepSmemPtr8),
          ptrPartialCtaSmemO,
          tmemCorr0DstStack.mClusterBarrierPtr12,
          mRegsO20,
          int32_t{64},
          numValidRowsO,
          numReductionRowsPerCta,
          params.mNumHeadsQ,
          params.mNumHeadsQPerKvDivisor,
          int32_t{128},
          mWarpGrpThreadIdx,
          int32_t{3});
      }
      // TmemCorr.h:3533
      if ((mCtaIdxKv) < (numCtasKvForReduction)) {
        // TmemCorr.h:3571
        trtllm::dev::
          reducePartialO<int32_t{32}, int32_t{64}, int32_t{64}, int32_t{128}, true, true, true>(
            ptrO,
            ptrPartialSmemO,
            ptrPartialSmemStats,
            params.ptrAttentionSinks,
            ptrSoftmaxStats,
            tmemCorr0DstStack.mClusterBarrierPtr12,
            completionBytes,
            scaleSoftmaxLog220,
            mNumCtasKv,
            mWarpGrpThreadIdx,
            mCtaIdxKv,
            headIdxO,
            params.mNumHeadsQ,
            params.mNumHeadsQPerKvDivisor,
            numValidRowsO,
            storesSoftmaxStats,
            ptrSfO,
            float(bool{params.ptrScaleSfO == nullptr}) ? (params.mScaleSfO)
                                                       : (float{params.ptrScaleSfO[int32_t{0}]}),
            sfRowIdx);
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
    // Task.cpp:3570
    {}
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
    , // FmhaTask.h:516
    mCtaIdxQ{(mCtaIdxX) / (params.mMaxNumCtasKv)}
    , // FmhaTask.h:517
    mCtaIdxKv{(mCtaIdxX) % (params.mMaxNumCtasKv)}
    , // FmhaTask.h:437
    mSeqLenKv{int32_t{params.ptrSeqLensKv[int32_t(params.mUseBlockSparseAttention)
                                            ? (((mHeadIdx) * (params.mBatchSize)) + (mBatchIdx))
                                            : (mBatchIdx)]}}
    , // FmhaTask.h:565
    mNumCtasKv{
      int32_t{min(int32_t{((mSeqLenKv) + (int32_t{127})) / (int32_t{128})}, params.mMaxNumCtasKv)}}
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
                                 SmemKSmem& smemKSrcSmem,
                                 SmemKStack& smemKSrcStack,
                                 SmemVSmem& smemVSrcSmem,
                                 SmemVStack& smemVSrcStack,
                                 TmemP0Stack& tmemP0SrcStack) {
    // Task.cpp:463
    cuda_ptx::setmaxnreg_dec(cuda_ptx::n32_t<112>{});
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
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemKConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      9,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemKConsReleaseState{};
    // Task.cpp:2135
    int32_t smemKConsToken{int32_t{0}};
    // Task.cpp:2114
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      9,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemVConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassTmaUmmaAsyncPipeline<
      9,
      cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::PipelineState smemVConsReleaseState{};
    // Task.cpp:2135
    int32_t smemVConsToken{int32_t{0}};
    // Task.cpp:2114
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState tmemP0ConsState{};
    // Task.cpp:2121
    trtllm::dev::CutlassCpAsyncPipeline<2, true>::PipelineState tmemP0ConsReleaseState{};
    // Task.cpp:2135
    int32_t tmemP0ConsToken{int32_t{0}};
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
    mNumCtasKv = int32_t{
      min(int32_t{((validSeqLenKv) + (int32_t{127})) / (int32_t{128})}, params.mMaxNumCtasKv)};
    // FmhaTask.h:630
    if ((((mCtaIdxQ) * (params.mNumTokensPerCtaQ)) < (mSeqLenQ)) && ((mCtaIdxKv) < (mNumCtasKv))) {
      // FmhaTask.h:668
      int32_t numSteps{((validSeqLenKv) + (((mNumCtasKv) * (int32_t{128})) - (int32_t{1}))) /
                       ((mNumCtasKv) * (int32_t{128}))};
      // FmhaTask.h:682
      numLoopSteps = numSteps;
    } else {
      // FmhaTask.h:651
      return;
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
    // tmemS0 [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    // Task.cpp:1607
    if (hasOneLoopIter) {
      // Task.cpp:5078
      {
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
    // smemK [ConsWait, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1607
    if (hasOneLoopIter) {
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:2780
        smemKConsToken = smemKSrcStack.mPipeline.consumer_try_wait(smemKConsState);
      }
      // Task.cpp:2848
      smemKSrcStack.mPipeline.consumer_wait(smemKConsState, smemKConsToken);
    }
    //
    // smemK [ConsWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // SmemKv.h:199
    cutlass::float_e4m3_t* smemPtrK4;
    // SmemKv.h:206
    int32_t smemIdxK4;
    // SmemKv.h:214
    cutlass::float_e4m3_t* smemPtrV4;
    // SmemKv.h:221
    int32_t smemIdxV4;
    // Task.cpp:1607
    if (hasOneLoopIter) {
      // Task.cpp:5945
      int32_t index{smemKConsState.index()};
      // SmemKv.h:267
      smemPtrK4 = &smemKSrcSmem.mArray[int32_t{0}][int32_t{0}];
      // SmemKv.h:304
      smemIdxK4 = index;
      // Task.cpp:43
      ++smemKConsState;
    }
    //
    // tmemS0 [ProdWork (call 0), FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    // TmemS.h:1130
    cutlass::float_e4m3_t* smemPtrQ14;
    // TmemS.h:1135
    int32_t smemIdxQ14;
    // TmemS.h:1141
    cutlass::float_e4m3_t* smemPtrK14;
    // TmemS.h:1146
    int32_t memIdxK14;
    // Task.cpp:1511
    smemPtrQ14 = smemPtrQ0_2;
    // Task.cpp:1511
    smemIdxQ14 = smemIdxQ0_2;
    // Task.cpp:1511
    smemPtrK14 = smemPtrK4;
    // Task.cpp:1511
    memIdxK14 = smemIdxK4;
    // Task.cpp:1607
    if (hasOneLoopIter) {
      // Task.cpp:5945
      int32_t index{tmemS0ProdState.index()};
      // TmemS.h:1889
      cutlass::float_e4m3_t* smemQ{smemPtrQ14};
      // TmemS.h:1910
      smemQ += (smemIdxQ14) * (int32_t{2048});
      // TmemS.h:1938
      cutlass::float_e4m3_t* smemK{smemPtrK14};
      // TmemS.h:1944
      smemK += (memIdxK14) * (int32_t{8192});
      // Mma.cpp:618
      {
        // TmemTile.cpp:1765
        uint32_t tmemPtrD{
          (index) * (int32_t{32}) +
          (int32_t((tmemS0DstStack.mInstId) == (int32_t{0})) ? (int32_t{0}) : (int32_t{32}))};
        //
        // leadingDimInBytes = 8192, strideInBytes = 512, swizzleMode = 2.
        //
        // Mma.cpp:203
        uint64_t smemDescA{
          trtllm::dev::createSmemDesc(smemK,
                                      uint32_t{0x2000000 /*hi=512, lo=0*/},
                                      uint32_t{0x80004020 /*hi=32768, lo=16416*/})};
        //
        // leadingDimInBytes = 2048, strideInBytes = 512, swizzleMode = 2.
        //
        // Mma.cpp:203
        uint64_t smemDescB{
          trtllm::dev::createSmemDesc(smemQ,
                                      uint32_t{0x800000 /*hi=128, lo=0*/},
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
                                                                int32_t{32},
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
                                                                int32_t{32},
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
    // smemK [ConsRelease, FirstIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // Task.cpp:1607
    if (hasOneLoopIter) {
      // Task.cpp:2596
      {
        // Task.cpp:2620
        smemKSrcStack.mPipeline.consumer_release(smemKConsReleaseState);
      }
      // Task.cpp:43
      ++smemKConsReleaseState;
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
    // tmemS0 [ProdAcquire, FirstIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
    //
    // Task.cpp:1607
    if (hasOneLoopIter) {
      // Task.cpp:5078
      {
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
    // Loop body.
    //
    // Task.cpp:3392
    CUTLASS_PRAGMA_NO_UNROLL
    for (int32_t loopOffset2493 = int32_t{0}; loopOffset2493 < (numLoopSteps) - (int32_t{1});
         ++loopOffset2493) {
      //
      // smemK [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1607
      // Task.cpp:2816
      {
        // Task.cpp:1607
        // Task.cpp:2757
        {
          // Task.cpp:2780
          smemKConsToken = smemKSrcStack.mPipeline.consumer_try_wait(smemKConsState);
        }
        // Task.cpp:2848
        smemKSrcStack.mPipeline.consumer_wait(smemKConsState, smemKConsToken);
      }
      //
      // smemK [ConsWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:1607
      // Task.cpp:2928
      {
        // Task.cpp:5945
        int32_t index{smemKConsState.index()};
        // SmemKv.h:267
        smemPtrK4 = &smemKSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:304
        smemIdxK4 = index;
        // Task.cpp:43
        ++smemKConsState;
      }
      //
      // tmemS0 [ProdWork (call 1), Info{0}, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
      //
      // Task.cpp:1511
      smemPtrQ14 = smemPtrQ0_2;
      // Task.cpp:1511
      smemIdxQ14 = smemIdxQ0_2;
      // Task.cpp:1511
      smemPtrK14 = smemPtrK4;
      // Task.cpp:1511
      memIdxK14 = smemIdxK4;
      // Task.cpp:1607
      // Task.cpp:5154
      {
        // Task.cpp:5945
        int32_t index{tmemS0ProdState.index()};
        // TmemS.h:1889
        cutlass::float_e4m3_t* smemQ{smemPtrQ14};
        // TmemS.h:1910
        smemQ += (smemIdxQ14) * (int32_t{2048});
        // TmemS.h:1938
        cutlass::float_e4m3_t* smemK{smemPtrK14};
        // TmemS.h:1944
        smemK += (memIdxK14) * (int32_t{8192});
        // Mma.cpp:618
        {
          // TmemTile.cpp:1765
          uint32_t tmemPtrD{
            (index) * (int32_t{32}) +
            (int32_t((tmemS0DstStack.mInstId) == (int32_t{0})) ? (int32_t{0}) : (int32_t{32}))};
          //
          // leadingDimInBytes = 8192, strideInBytes = 512, swizzleMode = 2.
          //
          // Mma.cpp:203
          uint64_t smemDescA{
            trtllm::dev::createSmemDesc(smemK,
                                        uint32_t{0x2000000 /*hi=512, lo=0*/},
                                        uint32_t{0x80004020 /*hi=32768, lo=16416*/})};
          //
          // leadingDimInBytes = 2048, strideInBytes = 512, swizzleMode = 2.
          //
          // Mma.cpp:203
          uint64_t smemDescB{
            trtllm::dev::createSmemDesc(smemQ,
                                        uint32_t{0x800000 /*hi=128, lo=0*/},
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
                                                                  int32_t{32},
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
                                                                  int32_t{32},
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
      // smemK [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
      //
      // Task.cpp:2568
      if ((loopOffset2493) >= (int32_t{0})) {
        // Task.cpp:2596
        {
          // Task.cpp:2620
          smemKSrcStack.mPipeline.consumer_release(smemKConsReleaseState);
        }
        // Task.cpp:43
        ++smemKConsReleaseState;
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
        if ((loopOffset2493) >= (int32_t{0})) {
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
      int32_t stageIdxP18;
      // Task.cpp:1607
      // Task.cpp:2928
      {
        // Task.cpp:5945
        int32_t index{tmemP0ConsState.index()};
        // TmemP.h:502
        stageIdxP18 = index;
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
        if ((loopOffset2493) >= (int32_t{0})) {
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
      // smemV [ConsWait, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:1607
      // Task.cpp:2816
      {
        // Task.cpp:1607
        // Task.cpp:2757
        {
          // Task.cpp:2780
          smemVConsToken = smemVSrcStack.mPipeline.consumer_try_wait(smemVConsState);
        }
        // Task.cpp:2848
        smemVSrcStack.mPipeline.consumer_wait(smemVConsState, smemVConsToken);
      }
      //
      // smemV [ConsWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // SmemKv.h:199
      cutlass::float_e4m3_t* smemPtrK5;
      // SmemKv.h:206
      int32_t smemIdxK5;
      // SmemKv.h:214
      cutlass::float_e4m3_t* smemPtrV5;
      // SmemKv.h:221
      int32_t smemIdxV5;
      // Task.cpp:1607
      // Task.cpp:2928
      {
        // Task.cpp:5945
        int32_t index{smemVConsState.index()};
        // SmemKv.h:322
        smemPtrV5 = &smemVSrcSmem.mArray[int32_t{0}][int32_t{0}];
        // SmemKv.h:372
        smemIdxV5 = index;
        // Task.cpp:43
        ++smemVConsState;
      }
      //
      // tmemO [ProdWork (call 0), Info{0}, FreqInfo{0, 1}, UserTags{3}, Flags{0}].
      //
      // TmemO.h:277
      cutlass::float_e4m3_t* smemPtrV19;
      // TmemO.h:282
      int32_t memIdxV19;
      // TmemO.h:288
      int32_t smemIdxP19;
      // Task.cpp:1511
      smemPtrV19 = smemPtrV5;
      // Task.cpp:1511
      memIdxV19 = smemIdxV5;
      // Task.cpp:1511
      smemIdxP19 = stageIdxP18;
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
        voteVal = localSmemSkipMmaVotePtr[smemIdxP19];
        // TmemO.h:768
        voteVal = uint32_t{__shfl_sync(uint32_t{0xffffffff}, voteVal, int32_t{0}, int32_t{32})};
        // TmemO.h:644
        skipsBmm2 = (voteVal) == (int32_t{1});
        // TmemO.h:653
        if (!(skipsBmm2)) {
          // TmemO.h:367
          cutlass::float_e4m3_t* smemP{
            reinterpret_cast<cutlass::float_e4m3_t*>(tmemODstStack.mDepSmemPtr8)};
          // TmemO.h:381
          smemP += (smemIdxP19) * (int32_t{4096});
          // TmemO.h:493
          cutlass::float_e4m3_t* smemV{smemPtrV19};
          // TmemO.h:505
          smemV = smemV + ((memIdxV19) * (int32_t{8192}));
          // TmemO.h:535
          bool readD{true};
          // TmemO.h:545
          if ((loopOffset2493) == (int32_t{0})) {
            // TmemO.h:547
            readD = false;
          }
          // Mma.cpp:618
          {
            // TmemTile.cpp:1765
            uint32_t tmemPtrD{(mTmemBaseOffset) + (uint32_t{128})};
            //
            // leadingDimInBytes = 0, strideInBytes = 512, swizzleMode = 2.
            //
            // Mma.cpp:203
            uint64_t smemDescA{
              trtllm::dev::createSmemDesc(smemV,
                                          uint32_t{0x0 /*hi=0, lo=0*/},
                                          uint32_t{0x80004020 /*hi=32768, lo=16416*/})};
            //
            // leadingDimInBytes = 4096, strideInBytes = 1024, swizzleMode = 1.
            //
            // Mma.cpp:203
            uint64_t smemDescB{
              trtllm::dev::createSmemDesc(smemP,
                                          uint32_t{0x1000000 /*hi=256, lo=0*/},
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
                                                                    int32_t{64},
                                                                    int32_t{32},
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
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    true,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{32},
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
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    true,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{32},
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
            trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
            // Mma.cpp:886
            trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
            // TmemTile.cpp:1610
            uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                    int32_t{0},
                                                                    int32_t{0},
                                                                    true,
                                                                    false,
                                                                    int32_t{64},
                                                                    int32_t{32},
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
      }
      //
      // smemV [ConsRelease, Info{0}, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
      //
      // Task.cpp:2568
      if ((loopOffset2493) >= (int32_t{0})) {
        // Task.cpp:2596
        {
          // Task.cpp:2620
          smemVSrcStack.mPipeline.consumer_release(smemVConsReleaseState);
        }
        // Task.cpp:43
        ++smemVConsReleaseState;
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
      lastLoopOffset = loopOffset2493;
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
    // tmemS0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
    // tmemP0 [ConsWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{1}, Flags{0}].
    //
    // TmemP.h:488
    int32_t stageIdxP18;
    // Task.cpp:1607
    if (hasOneLoopIter) {
      // Task.cpp:5945
      int32_t index{tmemP0ConsState.index()};
      // TmemP.h:502
      stageIdxP18 = index;
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
    // smemV [ConsWait, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1607
    if (hasOneLoopIter) {
      // Task.cpp:1607
      if (hasOneLoopIter) {
        // Task.cpp:2780
        smemVConsToken = smemVSrcStack.mPipeline.consumer_try_wait(smemVConsState);
      }
      // Task.cpp:2848
      smemVSrcStack.mPipeline.consumer_wait(smemVConsState, smemVConsToken);
    }
    //
    // smemV [ConsWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // SmemKv.h:214
    cutlass::float_e4m3_t* smemPtrV5;
    // SmemKv.h:221
    int32_t smemIdxV5;
    // Task.cpp:1607
    if (hasOneLoopIter) {
      // Task.cpp:5945
      int32_t index{smemVConsState.index()};
      // SmemKv.h:322
      smemPtrV5 = &smemVSrcSmem.mArray[int32_t{0}][int32_t{0}];
      // SmemKv.h:372
      smemIdxV5 = index;
      // Task.cpp:43
      ++smemVConsState;
    }
    //
    // tmemO [ProdWork (call 1), LastIter, FreqInfo{0, 1}, UserTags{3}, Flags{0}].
    //
    // TmemO.h:277
    cutlass::float_e4m3_t* smemPtrV19;
    // TmemO.h:282
    int32_t memIdxV19;
    // TmemO.h:288
    int32_t smemIdxP19;
    // Task.cpp:1511
    smemPtrV19 = smemPtrV5;
    // Task.cpp:1511
    memIdxV19 = smemIdxV5;
    // Task.cpp:1511
    smemIdxP19 = stageIdxP18;
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
      voteVal = localSmemSkipMmaVotePtr[smemIdxP19];
      // TmemO.h:768
      voteVal = uint32_t{__shfl_sync(uint32_t{0xffffffff}, voteVal, int32_t{0}, int32_t{32})};
      // TmemO.h:644
      skipsBmm2 = (voteVal) == (int32_t{1});
      // TmemO.h:653
      if (!(skipsBmm2)) {
        // TmemO.h:367
        cutlass::float_e4m3_t* smemP{
          reinterpret_cast<cutlass::float_e4m3_t*>(tmemODstStack.mDepSmemPtr8)};
        // TmemO.h:381
        smemP += (smemIdxP19) * (int32_t{4096});
        // TmemO.h:493
        cutlass::float_e4m3_t* smemV{smemPtrV19};
        // TmemO.h:505
        smemV = smemV + ((memIdxV19) * (int32_t{8192}));
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
          uint32_t tmemPtrD{(mTmemBaseOffset) + (uint32_t{128})};
          //
          // leadingDimInBytes = 0, strideInBytes = 512, swizzleMode = 2.
          //
          // Mma.cpp:203
          uint64_t smemDescA{
            trtllm::dev::createSmemDesc(smemV,
                                        uint32_t{0x0 /*hi=0, lo=0*/},
                                        uint32_t{0x80004020 /*hi=32768, lo=16416*/})};
          //
          // leadingDimInBytes = 4096, strideInBytes = 1024, swizzleMode = 1.
          //
          // Mma.cpp:203
          uint64_t smemDescB{
            trtllm::dev::createSmemDesc(smemP,
                                        uint32_t{0x1000000 /*hi=256, lo=0*/},
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
                                                                  int32_t{64},
                                                                  int32_t{32},
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
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_1{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{32},
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
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_2{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{32},
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
          trtllm::dev::incrSmemAddr(smemDescA, int32_t{128});
          // Mma.cpp:886
          trtllm::dev::incrSmemAddr(smemDescB, int32_t{2});
          // TmemTile.cpp:1610
          uint64_t utcmmaDesc_0_0_3{trtllm::dev::make_utcmma_desc(int32_t{1},
                                                                  int32_t{0},
                                                                  int32_t{0},
                                                                  true,
                                                                  false,
                                                                  int32_t{64},
                                                                  int32_t{32},
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
    }
    //
    // smemV [ConsRelease, LastIter, FreqInfo{0, 1}, UserTags{4}, Flags{0}].
    //
    // Task.cpp:1607
    if (hasOneLoopIter) {
      // Task.cpp:2596
      {
        // Task.cpp:2620
        smemVSrcStack.mPipeline.consumer_release(smemVConsReleaseState);
      }
      // Task.cpp:43
      ++smemVConsReleaseState;
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
    // tmemS0 [ProdCommit, LastIter, FreqInfo{0, 1}, UserTags{0}, Flags{0}].
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
  // Tail work.
  //
  // Task.cpp:3553
  ExitTileWithSignalingLabel:
  // Task.cpp:3560
  ExitTileWithoutSignalingLabel:
    // Task.cpp:3570
    {}
  }
};
extern "C" __global__
__launch_bounds__(384, 1) void fmhaSm100fKernel_QkvE4m3OE2m1H64PagedKvCausalP16MultiCtasKvCgaVarSeqSkipsSoftmaxQ32Kv128StaticSwapsAbForGen(
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
  uint8_t* smemKSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemKSmem)});
  // Kernel.cpp:1725
  smemOffset__ = (((smemOffset__) + (int32_t{1023})) / (int32_t{1024})) * (int32_t{1024});
  // Kernel.cpp:1729
  uint8_t* smemVSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemVSmem)});
  // Kernel.cpp:1725
  smemOffset__ = (((smemOffset__) + (int32_t{1023})) / (int32_t{1024})) * (int32_t{1024});
  // Kernel.cpp:1729
  uint8_t* smemPOSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemPOSmem)});
  // Kernel.cpp:1725
  smemOffset__ = (((smemOffset__) + (int32_t{127})) / (int32_t{128})) * (int32_t{128});
  // Kernel.cpp:1729
  uint8_t* smemPageOffsetsKSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemPageOffsetsKSmem)});
  // Kernel.cpp:1725
  smemOffset__ = (((smemOffset__) + (int32_t{127})) / (int32_t{128})) * (int32_t{128});
  // Kernel.cpp:1729
  uint8_t* smemPageOffsetsVSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemPageOffsetsVSmem)});
  // Kernel.cpp:1725
  smemOffset__ = (((smemOffset__) + (int32_t{63})) / (int32_t{64})) * (int32_t{64});
  // Kernel.cpp:1729
  uint8_t* smemCgaReductionSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemCgaReductionSmem)});
  // Kernel.cpp:1725
  smemOffset__ = (((smemOffset__) + (int32_t{15})) / (int32_t{16})) * (int32_t{16});
  // Kernel.cpp:1729
  uint8_t* smemSkipSoftmaxVoteSmemPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemSkipSoftmaxVoteSmem)});
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
  uint8_t* smemKSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemKSmemBarrier)});
  // Kernel.cpp:1729
  uint8_t* smemVSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemVSmemBarrier)});
  // Kernel.cpp:1729
  uint8_t* smemPageOffsetsKSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemPageOffsetsKSmemBarrier)});
  // Kernel.cpp:1729
  uint8_t* smemPageOffsetsVSmemBarrierPtr{(reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ += static_cast<int32_t>(uint64_t{sizeof(SmemPageOffsetsVSmemBarrier)});
  // Kernel.cpp:1729
  uint8_t* clusterTransactionBarrierBuffersSmemBarrierPtr{
    (reinterpret_cast<uint8_t*>(smem__) + smemOffset__)};
  // Kernel.cpp:1745
  smemOffset__ +=
    static_cast<int32_t>(uint64_t{sizeof(ClusterTransactionBarrierBuffersSmemBarrier)});
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
                        int32_t{10},
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
  SmemKSmem* smemKSmem{reinterpret_cast<SmemKSmem*>(smemKSmemPtr)};
  // Kernel.cpp:2228
  SmemKSmemBarrier* smemKSmemBarrier{reinterpret_cast<SmemKSmemBarrier*>(smemKSmemBarrierPtr)};
  // Kernel.cpp:2283
  SmemKStack smemKStack{(*smemKSmem),
                        (*smemKSmemBarrier),
                        state.mWarpIdx,
                        state.mClusterDimX,
                        state.mClusterDimY,
                        int32_t{10},
                        int32_t{-1}};
  // Kernel.cpp:2216
  SmemVSmem* smemVSmem{reinterpret_cast<SmemVSmem*>(smemVSmemPtr)};
  // Kernel.cpp:2228
  SmemVSmemBarrier* smemVSmemBarrier{reinterpret_cast<SmemVSmemBarrier*>(smemVSmemBarrierPtr)};
  // Kernel.cpp:2283
  SmemVStack smemVStack{(*smemVSmem),
                        (*smemVSmemBarrier),
                        (*smemSkipSoftmaxVoteSmem),
                        smemSkipSoftmaxVoteStack,
                        state.mWarpIdx,
                        state.mClusterDimX,
                        state.mClusterDimY,
                        int32_t{11},
                        int32_t{-1}};
  // Kernel.cpp:2216
  SmemPageOffsetsKSmem* smemPageOffsetsKSmem{
    reinterpret_cast<SmemPageOffsetsKSmem*>(smemPageOffsetsKSmemPtr)};
  // Kernel.cpp:2228
  SmemPageOffsetsKSmemBarrier* smemPageOffsetsKSmemBarrier{
    reinterpret_cast<SmemPageOffsetsKSmemBarrier*>(smemPageOffsetsKSmemBarrierPtr)};
  // Kernel.cpp:2283
  SmemPageOffsetsKStack smemPageOffsetsKStack{(*smemPageOffsetsKSmem),
                                              (*smemPageOffsetsKSmemBarrier),
                                              state.mWarpIdx,
                                              state.mClusterDimX,
                                              state.mClusterDimY,
                                              int32_t{9},
                                              int32_t{-1}};
  // Kernel.cpp:2216
  SmemPageOffsetsVSmem* smemPageOffsetsVSmem{
    reinterpret_cast<SmemPageOffsetsVSmem*>(smemPageOffsetsVSmemPtr)};
  // Kernel.cpp:2228
  SmemPageOffsetsVSmemBarrier* smemPageOffsetsVSmemBarrier{
    reinterpret_cast<SmemPageOffsetsVSmemBarrier*>(smemPageOffsetsVSmemBarrierPtr)};
  // Kernel.cpp:2283
  SmemPageOffsetsVStack smemPageOffsetsVStack{(*smemPageOffsetsVSmem),
                                              (*smemPageOffsetsVSmemBarrier),
                                              state.mWarpIdx,
                                              state.mClusterDimX,
                                              state.mClusterDimY,
                                              int32_t{9},
                                              int32_t{-1}};
  // Kernel.cpp:2216
  SmemPOSmem* smemPOSmem{reinterpret_cast<SmemPOSmem*>(smemPOSmemPtr)};
  // Kernel.cpp:2283
  SmemPOStack smemPOStack{(*smemPOSmem),
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
  // Kernel.cpp:2228
  ClusterTransactionBarrierBuffersSmemBarrier* clusterTransactionBarrierBuffersSmemBarrier{
    reinterpret_cast<ClusterTransactionBarrierBuffersSmemBarrier*>(
      clusterTransactionBarrierBuffersSmemBarrierPtr)};
  // Kernel.cpp:2283
  ClusterTransactionBarrierBuffersStack clusterTransactionBarrierBuffersStack{
    (*clusterTransactionBarrierBuffersSmemBarrier),
    state.mWarpIdx,
    state.mClusterDimX,
    state.mClusterDimY,
    int32_t{0},
    int32_t{-1}};
  // Kernel.cpp:2216
  SmemCgaReductionSmem* smemCgaReductionSmem{
    reinterpret_cast<SmemCgaReductionSmem*>(smemCgaReductionSmemPtr)};
  // Kernel.cpp:2283
  SmemCgaReductionStack smemCgaReductionStack{(*smemCgaReductionSmem),
                                              state.mWarpIdx,
                                              state.mClusterDimX,
                                              state.mClusterDimY,
                                              int32_t{0},
                                              int32_t{-1}};
  // Kernel.cpp:2228
  TmemS0SmemBarrier* tmemS0SmemBarrier{reinterpret_cast<TmemS0SmemBarrier*>(tmemS0SmemBarrierPtr)};
  // Kernel.cpp:2283
  TmemS0Stack tmemS0Stack{(*tmemS0SmemBarrier),
                          (*smemSoftmaxWarpGrpRed0Smem),
                          smemSoftmaxWarpGrpRed0Stack,
                          (*smemSkipSoftmaxVoteSmem),
                          smemSkipSoftmaxVoteStack,
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
  // Kernel.cpp:2228
  TmemOSmemBarrier* tmemOSmemBarrier{reinterpret_cast<TmemOSmemBarrier*>(tmemOSmemBarrierPtr)};
  // Kernel.cpp:2283
  TmemOStack tmemOStack{(*tmemOSmemBarrier),
                        (*smemPOSmem),
                        smemPOStack,
                        (*smemSkipSoftmaxVoteSmem),
                        smemSkipSoftmaxVoteStack,
                        state.mWarpIdx,
                        state.mClusterDimX,
                        state.mClusterDimY,
                        int32_t{4},
                        int32_t{-1}};
  // Kernel.cpp:2283
  TmemCorr0Stack tmemCorr0Stack{(*smemCorrWarpGrpRed1Smem),
                                smemCorrWarpGrpRed1Stack,
                                (*smemPOSmem),
                                smemPOStack,
                                (*smemCgaReductionSmem),
                                smemCgaReductionStack,
                                (*clusterTransactionBarrierBuffersSmemBarrier),
                                clusterTransactionBarrierBuffersStack,
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
  // Kernel.cpp:1888
  cute::cluster_sync();
  // Kernel.cpp:2014
  if (bool{LoadPageOffsetsTask::isSelected(params, state)}) {
    // Kernel.cpp:2081
    LoadPageOffsetsTask loadPageOffsetsTask{params, state, int32_t{9}};
    // Kernel.cpp:2135
    loadPageOffsetsTask.execute(params,
                                state,
                                (*smemPageOffsetsKSmem),
                                smemPageOffsetsKStack,
                                (*smemPageOffsetsVSmem),
                                smemPageOffsetsVStack);
  } else {
    // Kernel.cpp:2014
    if (bool{LoadTaskQk::isSelected(params, state)}) {
      // Kernel.cpp:2081
      LoadTaskQk loadTaskQk{params, state, int32_t{10}};
      // Kernel.cpp:2135
      loadTaskQk.execute(params,
                         state,
                         (*smemQSmem),
                         smemQStack,
                         (*smemKSmem),
                         smemKStack,
                         (*smemPageOffsetsKSmem),
                         smemPageOffsetsKStack);
    } else {
      // Kernel.cpp:2014
      if (bool{LoadTaskV::isSelected(params, state)}) {
        // Kernel.cpp:2081
        LoadTaskV loadTaskV{params, state, int32_t{11}};
        // Kernel.cpp:2135
        loadTaskV.execute(params,
                          state,
                          (*smemVSmem),
                          smemVStack,
                          (*smemPageOffsetsVSmem),
                          smemPageOffsetsVStack);
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
                               tmemS0Stack);
        } else {
          // Kernel.cpp:2014
          if (bool{CorrTask::isSelected(params, state)}) {
            // Kernel.cpp:2081
            CorrTask corrTask{params, state, int32_t{4}};
            // Kernel.cpp:2135
            corrTask.execute(params, state, tmemCorr0Stack, tmemSoftmaxLocal0Stack, tmemOStack);
            // Task.cpp:5404
            trtllm::dev::CutlassNamedBarrier::sync(128, 13);
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
                              (*smemKSmem),
                              smemKStack,
                              (*smemVSmem),
                              smemVStack,
                              tmemP0Stack);
            }
          }
        }
      }
    }
  }
}
extern "C" __global__ void
fmhaSm100fKernel_QkvE4m3OE2m1H64PagedKvCausalP16MultiCtasKvCgaVarSeqSkipsSoftmaxQ32Kv128StaticSwapsAbForGenGetSmemSize(
  int32_t* outPtr) {
  int32_t size{0};
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemQSmem));
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemKSmem));
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemVSmem));
  size = (size + 1023) / 1024 * 1024;
  size += static_cast<int32_t>(sizeof(SmemPOSmem));
  size = (size + 127) / 128 * 128;
  size += static_cast<int32_t>(sizeof(SmemPageOffsetsKSmem));
  size = (size + 127) / 128 * 128;
  size += static_cast<int32_t>(sizeof(SmemPageOffsetsVSmem));
  size = (size + 63) / 64 * 64;
  size += static_cast<int32_t>(sizeof(SmemCgaReductionSmem));
  size = (size + 15) / 16 * 16;
  size += static_cast<int32_t>(sizeof(SmemSkipSoftmaxVoteSmem));
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
  size += static_cast<int32_t>(sizeof(SmemKSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(SmemVSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(SmemPageOffsetsKSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(SmemPageOffsetsVSmemBarrier));
  size = (size + 0) / 1 * 1;
  size += static_cast<int32_t>(sizeof(ClusterTransactionBarrierBuffersSmemBarrier));
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
