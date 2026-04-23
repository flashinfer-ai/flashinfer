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
#include <trtllm/dev/Fp4Utils.h>
#include <KernelParams.h>
namespace batchedGemm {


using CuteLayout48 = cute::Layout<cute::Int<int32_t{178944}>, cute::Int<int32_t{1}>>;
using CuteLayout53 = cute::Layout<cute::Int<int32_t{32768}>, cute::Int<int32_t{1}>>;
using CuteLayout58 = cute::Layout<cute::Int<int32_t{4096}>, cute::Int<int32_t{1}>>;
using CuteLayout63 = cute::Layout<cute::Int<int32_t{1024}>, cute::Int<int32_t{1}>>;
using CuteLayout68 = cute::Layout<cute::Int<int32_t{128}>, cute::Int<int32_t{1}>>;
using CuteLayout77 = cute::Layout<cute::Int<int32_t{2048}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple207 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple211 =
  cute::tuple<cute::Int<int32_t{2}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple333 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple337 =
  cute::tuple<cute::Int<int32_t{2}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple454 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple458 =
  cute::tuple<cute::Int<int32_t{2}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple577 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple696 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple700 =
  cute::tuple<cute::Int<int32_t{2}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple808 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple812 =
  cute::tuple<cute::Int<int32_t{2}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple925 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple929 =
  cute::tuple<cute::Int<int32_t{2}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;

struct __align__(1024) SmemBufferSmem {
  int8_t mArray[178944];
};
struct __align__(1024) SmemASmem{};
struct SmemASmemBarrier {
  typename trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
    5,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
struct __align__(1024) SmemBSmem{};
struct SmemBSmemBarrier {
  typename trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
    5,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
struct __align__(1024) SmemSfASmem {
  cutlass::float_ue8m0_t mArray[5][1024];
};
struct SmemSfASmemBarrier {
  typename trtllm::dev::CutlassTmaUmmaAsyncPipeline<
    5,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
struct __align__(1024) SmemSfBSmem {
  cutlass::float_ue8m0_t mArray[5][128];
};
struct SmemSfBSmemBarrier {
  typename trtllm::dev::CutlassTmaAsyncPipeline<5>::SharedStorage mBarriers;
};
struct TmemSfASmemBarrier {
  typename trtllm::dev::CutlassUmmaConsumerAsyncPipeline<
    5,
    false,
    false,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
struct TmemSfBSmemBarrier {
  typename trtllm::dev::CutlassUmmaConsumerAsyncPipeline<
    5,
    false,
    false,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
struct Mma0SmemBarrier {
  typename trtllm::dev::CutlassUmmaAsyncPipeline<
    1,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
struct __align__(1024) GmemC0Smem{};
struct ClusterBarrierBuffersSmemBarrier {
  typename trtllm::dev::CutlassClusterBarrier<cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
                                              true>::SharedStorage mClusterSmemBarriers;
};

} // namespace batchedGemm
