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
#include <trtllm/dev/ReduceRow.h>
#include <trtllm/dev/ReduceCol.h>
#include <trtllm/dev/Fp4Utils.h>
#include <KernelParams.h>
namespace batchedGemm {


using CuteLayout48 = cute::Layout<cute::Int<int32_t{182784}>, cute::Int<int32_t{1}>>;
using CuteLayout53 = cute::Layout<cute::Int<int32_t{16384}>, cute::Int<int32_t{1}>>;
using CuteLayout58 = cute::Layout<cute::Int<int32_t{16384}>, cute::Int<int32_t{1}>>;
using CuteLayout67 = cute::Layout<cute::Int<int32_t{8192}>, cute::Int<int32_t{1}>>;
using CuteLayout76 = cute::Layout<cute::Int<int32_t{8192}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple190 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple389 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple495 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple624 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple746 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple865 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple979 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple1088 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;

struct __align__(16) WorkIdSmem {
  alignas(16) typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>,
    3>::CLCResponse workIdResponse[3];
  alignas(16) typename trtllm::dev::ClcFastDrain<4>::SharedStorage fastDrainStorage;
};
struct WorkIdSmemBarrier {
  typename trtllm::dev::CutlassWorkIdPipeline<
    3,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
struct WorkThrottleBarrierSmemBarrier {
  typename trtllm::dev::CutlassCpAsyncPipeline<3>::SharedStorage mBarriers;
};
struct __align__(1024) SmemBufferSmem {
  int8_t mArray[182784];
};
struct __align__(1024) SmemASmem{};
struct SmemASmemBarrier {
  typename trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
    5,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
struct __align__(1024) SmemBSmem{};
struct SmemBSmemBarrier {
  typename trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
    5,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
struct __align__(16) SmemDeepSeekSfAbSmem {
  float mDqSfsAct[5][128];
  float mDqSfsWeights[5][1];
};
struct SmemDeepSeekSfAbSmemBarrier {
  typename trtllm::dev::CutlassCpAsyncPipeline<5>::SharedStorage mBarriers;
};
struct Mma0SmemBarrier {
  typename trtllm::dev::CutlassUmmaAsyncPipeline<
    4,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
struct Mma1SmemBarrier {
  typename trtllm::dev::CutlassUmmaAsyncPipeline<
    4,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
struct __align__(1024) GmemC0Smem{};
struct __align__(1024) GmemC1Smem{};

} // namespace batchedGemm
