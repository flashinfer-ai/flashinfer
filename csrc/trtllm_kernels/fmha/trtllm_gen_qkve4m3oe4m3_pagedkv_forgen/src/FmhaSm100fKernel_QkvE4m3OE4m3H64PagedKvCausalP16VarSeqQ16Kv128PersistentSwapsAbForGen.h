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
// SmemPageOffsetsKv.h:81
using CuteLayout90 = cute::Layout<cute::Int<int32_t{32}>, cute::Int<int32_t{1}>>;
// Fmha.h:1268
using CuteLayout95 = cute::Layout<cute::Int<int32_t{4096}>, cute::Int<int32_t{1}>>;
// Fmha.h:1276
using CuteLayout100 = cute::Layout<cute::Int<int32_t{1024}>, cute::Int<int32_t{1}>>;
// Fmha.h:1308
using CuteLayout105 = cute::Layout<cute::Int<int32_t{64}>, cute::Int<int32_t{1}>>;
// Fmha.h:1318
using CuteLayout110 = cute::Layout<cute::Int<int32_t{64}>, cute::Int<int32_t{1}>>;
// Fmha.h:1339
using CuteLayout115 = cute::Layout<cute::Int<int32_t{128}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple232 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:619
using CuteFlatTuple236 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple352 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:619
using CuteFlatTuple356 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple477 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple628 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:819
using CuteFlatTuple635 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:1074
using CuteFlatTuple831 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple939 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple1057 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:619
using CuteFlatTuple1061 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple1175 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple1318 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple1440 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:619
using CuteFlatTuple1444 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;

// Res.cpp:125
// Fmha.h:845
struct __align__(1024) SmemQSmem {
  // MemBuffers.cpp:251
  cutlass::float_e4m3_t mArray[2][1024];
};
// Res.cpp:81
// Fmha.h:845
struct SmemQSmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassTmaUmmaAsyncPipeline<
    2,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
// Res.cpp:125
// Fmha.h:1117
struct __align__(1024) SmemKvSmem {
  // MemBuffers.cpp:251
  cutlass::float_e4m3_t mArray[18][8192];
};
// Res.cpp:81
// Fmha.h:1117
struct SmemKvSmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassTmaUmmaAsyncPipeline<
    18,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
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
  int8_t mArray[4096];
};
// Res.cpp:125
// Fmha.h:1271
struct __align__(128) SmemOSmem {
  // MemBuffers.cpp:251
  int8_t mArray[1024];
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
  float mArray[128];
};
// Res.cpp:125
// Fmha.h:1520
struct __align__(16) WorkIdStorageSmem {
  // Res.cpp:1004
  alignas(16) typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
    2>::CLCResponse workIdResponse[2];
  // Res.cpp:1017
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
// Fmha.h:1890
struct TmemS0SmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassUmmaAsyncPipeline<
    1,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
// Res.cpp:81
// SoftmaxSchedule.h:295
struct TmemSoftmaxLocal0SmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassCpAsyncPipeline<1, true>::SharedStorage mBarriers;
};
// Res.cpp:81
// Fmha.h:1987
struct OrderP01SmemBarrier {
  // Res.cpp:458
  typename trtllm::dev::CutlassOrderedSequenceBarrier<1, 2>::SharedStorage mOrderedSequenceBarriers;
};
// Res.cpp:81
// Fmha.h:2017
struct TmemP0SmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassCpAsyncPipeline<1, true>::SharedStorage mBarriers;
};
// Res.cpp:81
// Fmha.h:2104
struct TmemOSmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassUmmaAsyncPipeline<
    2,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
