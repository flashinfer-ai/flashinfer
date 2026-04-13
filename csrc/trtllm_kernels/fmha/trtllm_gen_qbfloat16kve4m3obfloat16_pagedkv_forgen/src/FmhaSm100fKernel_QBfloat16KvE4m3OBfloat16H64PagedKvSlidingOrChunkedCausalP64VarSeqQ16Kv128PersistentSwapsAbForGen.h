#pragma once

#include <cuda_ptx/cuda_ptx.h>
#include <cuda/std/cstdint>
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/atom/copy_traits_sm80.hpp>
#include <cute/atom/copy_traits_sm90.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/builders/sm90_common.inl>
#include <cutlass/gemm/kernel/sm100_tile_scheduler.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <trtllm/dev/CuteUtils.h>
#include <trtllm/dev/CutlassBarrier.h>
#include <trtllm/dev/CutlassPipeline.h>
#include <trtllm/dev/CutlassUtils.h>
#include <trtllm/dev/Schedule.h>
#include <trtllm/dev/SmemTile.h>
#include <trtllm/dev/TmemTile.h>
#include <trtllm/dev/Utils.h>
#include <trtllm/dev/FastMath.h>
#include <trtllm/dev/Fp4Utils.h>
#include <trtllm/dev/ReduceCol.h>
#include <trtllm/dev/ReduceRow.h>
#include <trtllm/dev/ReduceMultiCtasKv.h>
#include <trtllm/dev/StoreGmemO.h>
#include <trtllm/dev/StoreSmemP.h>
#include <trtllm/dev/StoreSmemOut.h>
#include <KernelParams.h>

// SmemQ.h:127
using CuteLayout80 = cute::Layout<cute::Int<int32_t{1024}>, cute::Int<int32_t{1}>>;
// SmemKv.h:124
using CuteLayout85 = cute::Layout<cute::Int<int32_t{8192}>, cute::Int<int32_t{1}>>;
// SmemKv.h:124
using CuteLayout90 = cute::Layout<cute::Int<int32_t{8192}>, cute::Int<int32_t{1}>>;
// SmemPageOffsetsKv.h:81
using CuteLayout95 = cute::Layout<cute::Int<int32_t{32}>, cute::Int<int32_t{1}>>;
// Fmha.h:1268
using CuteLayout100 = cute::Layout<cute::Int<int32_t{8192}>, cute::Int<int32_t{1}>>;
// Fmha.h:1276
using CuteLayout105 = cute::Layout<cute::Int<int32_t{2048}>, cute::Int<int32_t{1}>>;
// Fmha.h:1308
using CuteLayout110 = cute::Layout<cute::Int<int32_t{64}>, cute::Int<int32_t{1}>>;
// Fmha.h:1318
using CuteLayout115 = cute::Layout<cute::Int<int32_t{64}>, cute::Int<int32_t{1}>>;
// Fmha.h:1339
using CuteLayout120 = cute::Layout<cute::Int<int32_t{64}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple235 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:619
using CuteFlatTuple239 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple355 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple479 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple597 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple748 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:814
using CuteFlatTuple755 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:1069
using CuteFlatTuple951 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple1059 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple1177 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:619
using CuteFlatTuple1181 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple1295 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple1431 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple1553 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:619
using CuteFlatTuple1557 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;

// Res.cpp:125
// Fmha.h:845
struct __align__(1024) SmemQSmem {
  // MemBuffers.cpp:251
  cutlass::bfloat16_t mArray[1][1024];
};
// Res.cpp:81
// Fmha.h:845
struct SmemQSmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassTmaUmmaAsyncPipeline<
    1,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
// Res.cpp:125
// Fmha.h:1117
struct __align__(1024) SmemKvSmem {
  // MemBuffers.cpp:251
  cutlass::float_e4m3_t mArray[9][8192];
};
// Res.cpp:81
// Fmha.h:1117
struct SmemKvSmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassTmaAsyncPipeline<9>::SharedStorage mBarriers;
};
// Res.cpp:125
// Fmha.h:1163
struct __align__(1024) SmemTransformedKvSmem {
  // MemBuffers.cpp:251
  cutlass::bfloat16_t mArray[2][8192];
};
// Res.cpp:81
// Fmha.h:1163
struct SmemTransformedKvSmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassUmmaConsumerAsyncPipeline<2, false, false>::SharedStorage mBarriers;
};
// Res.cpp:125
// Fmha.h:1217
struct __align__(128) SmemPageOffsetsKvSmem {
  // MemBuffers.cpp:251
  int32_t mArray[6][32];
};
// Res.cpp:81
// Fmha.h:1217
struct SmemPageOffsetsKvSmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassCpAsyncPipeline<6>::SharedStorage mBarriers;
};
// Res.cpp:125
// Fmha.h:1263
struct __align__(1024) SmemPSmem {
  // MemBuffers.cpp:251
  int8_t mArray[8192];
};
// Res.cpp:125
// Fmha.h:1271
struct __align__(128) SmemOSmem {
  // MemBuffers.cpp:251
  int8_t mArray[2048];
};
// Res.cpp:125
// Fmha.h:1304
struct __align__(16) SmemSoftmaxWarpGrpRed0Smem {
  // MemBuffers.cpp:251
  float mArray[64];
};
// Res.cpp:125
// Fmha.h:1334
struct __align__(16) SmemCorrWarpGrpRed1Smem {
  // MemBuffers.cpp:251
  float mArray[64];
};
// Res.cpp:125
// Fmha.h:1520
struct __align__(16) WorkIdStorageSmem {
  // Res.cpp:999
  alignas(16) typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
    2>::CLCResponse workIdResponse[2];
  // Res.cpp:1012
  alignas(16) typename trtllm::dev::ClcFastDrain<4>::SharedStorage fastDrainStorage;
};
// Res.cpp:81
// Fmha.h:1520
struct WorkIdStorageSmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassWorkIdPipeline<
    2,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
// Res.cpp:81
// Fmha.h:1525
struct WorkIdThrottleBarrierSmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassCpAsyncPipeline<2, true>::SharedStorage mBarriers;
};
// Res.cpp:81
// Fmha.h:1866
struct TmemS0SmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassUmmaAsyncPipeline<
    2,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
// Res.cpp:81
// SoftmaxSchedule.h:242
struct TmemSoftmaxLocal0SmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassCpAsyncPipeline<2, true>::SharedStorage mBarriers;
};
// Res.cpp:81
// Fmha.h:1943
struct OrderP01SmemBarrier {
  // Res.cpp:458
  typename trtllm::dev::CutlassOrderedSequenceBarrier<1, 2>::SharedStorage mOrderedSequenceBarriers;
};
// Res.cpp:81
// Fmha.h:1973
struct TmemP0SmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassCpAsyncPipeline<2, true>::SharedStorage mBarriers;
};
// Res.cpp:81
// Fmha.h:2060
struct TmemOSmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassUmmaAsyncPipeline<
    1,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
