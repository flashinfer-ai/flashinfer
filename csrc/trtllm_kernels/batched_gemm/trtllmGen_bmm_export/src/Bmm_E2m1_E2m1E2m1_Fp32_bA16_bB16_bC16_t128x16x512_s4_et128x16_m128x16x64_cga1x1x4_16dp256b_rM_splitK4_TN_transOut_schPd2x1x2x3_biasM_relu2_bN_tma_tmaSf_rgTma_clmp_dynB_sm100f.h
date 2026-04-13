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


using CuteLayout48 = cute::Layout<cute::Int<int32_t{180992}>, cute::Int<int32_t{1}>>;
using CuteLayout53 = cute::Layout<cute::Int<int32_t{65536}>, cute::Int<int32_t{1}>>;
using CuteLayout58 = cute::Layout<cute::Int<int32_t{8192}>, cute::Int<int32_t{1}>>;
using CuteLayout63 = cute::Layout<cute::Int<int32_t{4096}>, cute::Int<int32_t{1}>>;
using CuteLayout68 = cute::Layout<cute::Int<int32_t{512}>, cute::Int<int32_t{1}>>;
using CuteLayout79 = cute::Layout<cute::Int<int32_t{8192}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple193 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple392 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{4}>>;
using CuteFlatTuple498 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple627 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple749 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple866 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple985 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple1104 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple1212 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple1325 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;

struct __align__(16) WorkIdSmem {
  alignas(16) typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<4>>,
    3>::CLCResponse workIdResponse[3];
  alignas(16) typename trtllm::dev::ClcFastDrain<4>::SharedStorage fastDrainStorage;
};
struct WorkIdSmemBarrier {
  typename trtllm::dev::CutlassWorkIdPipeline<
    3,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<4>>>::SharedStorage mBarriers;
};
struct WorkThrottleBarrierSmemBarrier {
  typename trtllm::dev::CutlassCpAsyncPipeline<3>::SharedStorage mBarriers;
};
struct __align__(1024) SmemBufferSmem {
  int8_t mArray[180992];
};
struct __align__(1024) SmemASmem{};
struct SmemASmemBarrier {
  typename trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
    4,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
struct __align__(1024) SmemBSmem{};
struct SmemBSmemBarrier {
  typename trtllm::dev::CutlassTmaMultiUmmaAsyncPipeline<
    4,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
struct __align__(1024) SmemSfASmem {
  cutlass::float_e4m3_t mArray[4][4096];
};
struct SmemSfASmemBarrier {
  typename trtllm::dev::CutlassTmaUmmaAsyncPipeline<
    4,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
struct __align__(1024) SmemSfBSmem {
  cutlass::float_e4m3_t mArray[4][512];
};
struct SmemSfBSmemBarrier {
  typename trtllm::dev::CutlassTmaAsyncPipeline<4>::SharedStorage mBarriers;
};
struct TmemSfASmemBarrier {
  typename trtllm::dev::CutlassUmmaConsumerAsyncPipeline<4, false, false>::SharedStorage mBarriers;
};
struct TmemSfBSmemBarrier {
  typename trtllm::dev::CutlassUmmaConsumerAsyncPipeline<4, false, false>::SharedStorage mBarriers;
};
struct Mma0SmemBarrier {
  typename trtllm::dev::CutlassUmmaAsyncPipeline<
    2,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
struct ClusterBarrier0SmemBarrier {
  typename trtllm::dev::CutlassClusterBarrier<
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<4>>>::SharedStorage mClusterSmemBarriers;
};
struct __align__(1024) GmemC0Smem {
  uint64_t mSplitKBar[1];
};

} // namespace batchedGemm
