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
using CuteLayout80 = cute::Layout<cute::Int<int32_t{32768}>, cute::Int<int32_t{1}>>;
// Fmha.h:875
using CuteLayout85 = cute::Layout<cute::Int<int32_t{2}>, cute::Int<int32_t{1}>>;
// SmemKv.h:124
using CuteLayout90 = cute::Layout<cute::Int<int32_t{16384}>, cute::Int<int32_t{1}>>;
// SmemPageOffsetsKv.h:81
using CuteLayout95 = cute::Layout<cute::Int<int32_t{32}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple212 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:619
using CuteFlatTuple216 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple348 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:619
using CuteFlatTuple352 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple473 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple591 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:619
using CuteFlatTuple595 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple709 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple838 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:609
using CuteFlatTuple960 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
// Res.cpp:619
using CuteFlatTuple964 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;

// Res.cpp:125
// Fmha.h:845
struct __align__(1024) SmemQSmem {
  // MemBuffers.cpp:251
  cutlass::float_e4m3_t mArray[2][32768];
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
// Fmha.h:870
struct __align__(16) SmemSkipSoftmaxVoteSmem {
  // MemBuffers.cpp:251
  int32_t mArray[2];
};
// Res.cpp:125
// Fmha.h:1117
struct __align__(1024) SmemKvSmem {
  // MemBuffers.cpp:251
  cutlass::float_e4m3_t mArray[9][16384];
};
// Res.cpp:81
// Fmha.h:1117
struct SmemKvSmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassTmaUmmaAsyncPipeline<
    9,
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
// Res.cpp:81
// Fmha.h:1890
struct TmemS0SmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassUmmaAsyncPipeline<
    2,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
// Res.cpp:81
// SoftmaxSchedule.h:295
struct TmemSoftmaxLocal0SmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassCpAsyncPipeline<2, true>::SharedStorage mBarriers;
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
  typename trtllm::dev::CutlassCpAsyncPipeline<2, true>::SharedStorage mBarriers;
};
// Res.cpp:81
// Fmha.h:2104
struct TmemOSmemBarrier {
  // Res.cpp:445
  typename trtllm::dev::CutlassUmmaAsyncPipeline<
    1,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
