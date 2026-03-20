/*
# SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
# ==============================================================================
*/
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


using CuteLayout48 = cute::Layout<cute::Int<int32_t{184832}>, cute::Int<int32_t{1}>>;
using CuteLayout53 = cute::Layout<cute::Int<int32_t{16384}>, cute::Int<int32_t{1}>>;
using CuteLayout58 = cute::Layout<cute::Int<int32_t{32768}>, cute::Int<int32_t{1}>>;
using CuteLayout63 = cute::Layout<cute::Int<int32_t{512}>, cute::Int<int32_t{1}>>;
using CuteLayout68 = cute::Layout<cute::Int<int32_t{1024}>, cute::Int<int32_t{1}>>;
using CuteLayout77 = cute::Layout<cute::Int<int32_t{32768}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple191 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple390 =
  cute::tuple<cute::Int<int32_t{2}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple496 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple625 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple629 =
  cute::tuple<cute::Int<int32_t{2}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple751 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple755 =
  cute::tuple<cute::Int<int32_t{2}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple872 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple876 =
  cute::tuple<cute::Int<int32_t{2}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple995 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple999 =
  cute::tuple<cute::Int<int32_t{2}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple1115 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple1119 =
  cute::tuple<cute::Int<int32_t{2}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple1232 =
  cute::tuple<cute::Int<int32_t{1}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;
using CuteFlatTuple1236 =
  cute::tuple<cute::Int<int32_t{2}>, cute::Int<int32_t{1}>, cute::Int<int32_t{1}>>;

struct __align__(16) WorkIdSmem {
  alignas(16) typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
    3>::CLCResponse workIdResponse[3];
  alignas(16) typename trtllm::dev::ClcFastDrain<4>::SharedStorage fastDrainStorage;
};
struct WorkIdSmemBarrier {
  typename trtllm::dev::CutlassWorkIdPipeline<
    3,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
struct WorkThrottleBarrierSmemBarrier {
  typename trtllm::dev::CutlassCpAsyncPipeline<3>::SharedStorage mBarriers;
};
struct __align__(1024) SmemBufferSmem {
  int8_t mArray[184832];
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
  cutlass::float_ue8m0_t mArray[5][512];
};
struct SmemSfASmemBarrier {
  typename trtllm::dev::CutlassTmaUmmaAsyncPipeline<
    5,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
struct __align__(1024) SmemSfBSmem {
  cutlass::float_ue8m0_t mArray[5][1024];
};
struct SmemSfBSmemBarrier {
  typename trtllm::dev::CutlassTmaUmmaAsyncPipeline<
    5,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>,
    cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>>::SharedStorage mBarriers;
};
struct TmemSfAbSmemBarrier {
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
