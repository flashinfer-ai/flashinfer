/*
 * Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_MXFP8_GEMM_TEMPLATE_SM120_H_
#define FLASHINFER_MXFP8_GEMM_TEMPLATE_SM120_H_

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // #ifndef _WIN32

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "flashinfer/arch_condition.h"
#include "flashinfer/cutlass_utils.cuh"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif  // #ifndef _WIN32

namespace flashinfer {
namespace gemm {
using namespace cute;

#ifdef ENABLE_BF16
using SafeBF16Sm120 = __nv_bfloat16;
#else
using SafeBF16Sm120 = void;
#endif

template <typename T, typename CTA_M_, typename CTA_N_, typename CTA_K_, bool SWAP_AB_>
size_t genericMxfp8GemmKernelLauncherSm120(void* D, void const* A, void const* B,
                                           void const* input_sf, void const* weight_sf, int m,
                                           int n, int k, int batch_count,
                                           CutlassGemmConfig gemmConfig, char* workspace,
                                           size_t const workspaceBytes, cudaStream_t stream,
                                           int* occupancy);

#ifdef PLACEHOLDER_KERNELS

#define INSTANTIATE_MXFP8_GEMM_KERNEL_LAUNCHER_SM120(T, CTA_M_, CTA_N_, CTA_K_, SWAP_AB_)        \
  template <>                                                                                    \
  size_t genericMxfp8GemmKernelLauncherSm120<T, cute::Int<CTA_M_>, cute::Int<CTA_N_>,            \
                                             cute::Int<CTA_K_>, SWAP_AB_>(                       \
      void* D, void const* A, void const* B, void const* input_sf, void const* weight_sf, int m, \
      int n, int k, int batch_count, CutlassGemmConfig gemmConfig, char* workspace,              \
      const size_t workspaceBytes, cudaStream_t stream, int* occupancy) {                        \
    throw std::runtime_error(                                                                    \
        "MXFP8 SM120 gemm kernel is not compiled with support for "                              \
        "this Architecture.");                                                                   \
  }

#else

// SM120 block-scaled MXFP8 GEMM kernel instantiation macro.
// Key differences from SM100:
//   - arch::Sm120 (no 2SM MMA)
//   - ClusterShape always static Shape<1,1,1> (no programmatic multicast on SM120)
//   - KernelScheduleAuto: CUTLASS auto-selects Cooperative vs Pingpong based on tile shape
//   - EpilogueScheduleAuto: CUTLASS auto-selects epilogue schedule
#define INSTANTIATE_MXFP8_GEMM_KERNEL_LAUNCHER_SM120(T, CTA_M_, CTA_N_, CTA_K_, SWAP_AB_)         \
  struct DeviceGemmMxfp8GemmSm120_##T##_##CTA_M_##_##CTA_N_##_##CTA_K_##SWAP_AB_ {                \
    using OutElementType = flashinfer::cutlass_dtype<T>::type;                                    \
    using CTAShape = cute::Shape<cute::Int<CTA_M_>, cute::Int<CTA_N_>, cute::Int<CTA_K_>>;        \
    /* SM120 only supports ClusterShape 1x1x1 (no programmatic multicast) */                      \
    using ClusterShape = cute::Shape<_1, _1, _1>;                                                 \
    using ElementType = cutlass::float_e4m3_t;                                                    \
    using Arch = cutlass::arch::Sm120;                                                            \
    /* Input A */                                                                                 \
    using ElementA = ElementType;                                                                 \
    using LayoutA = cutlass::layout::RowMajor;                                                    \
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementType>::value;             \
    /* Input B */                                                                                 \
    using ElementB = ElementType;                                                                 \
    using LayoutB = cutlass::layout::ColumnMajor;                                                 \
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementType>::value;             \
    /* Input C */                                                                                 \
    using ElementC = void;                                                                        \
    using LayoutC =                                                                               \
        std::conditional_t<SWAP_AB_, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;    \
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<OutElementType>::value;          \
                                                                                                  \
    using SFType = cutlass::float_ue8m0_t;                                                        \
    using ElementCompute = float;                                                                 \
    using ElementAccumulator = float;                                                             \
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;                              \
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;                     \
    /* Let CUTLASS auto-select: Cooperative for large N, Pingpong for small N */                  \
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;                 \
    using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;                       \
                                                                                                  \
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<         \
        Arch, OperatorClass, CTAShape, ClusterShape, EpilogueTileType, ElementAccumulator,        \
        ElementCompute, ElementC, LayoutC, AlignmentC, OutElementType, LayoutC, AlignmentC,       \
        EpilogueSchedule,                                                                         \
        cutlass::epilogue::fusion::LinearCombination<OutElementType, float, void,                 \
                                                     float>>::CollectiveOp;                       \
                                                                                                  \
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<             \
        Arch, OperatorClass, cutlass::mx_float8_t<ElementA>, LayoutA, AlignmentA,                 \
        cutlass::mx_float8_t<ElementB>, LayoutB, AlignmentB, ElementAccumulator, CTAShape,        \
        ClusterShape,                                                                             \
        std::conditional_t<CTA_N_ == 256 || CTA_M_ == 256,                                        \
                           cutlass::gemm::collective::StageCount<2>,                              \
                           cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(    \
                               sizeof(typename CollectiveEpilogue::SharedStorage))>>,             \
        MainloopSchedule>::CollectiveOp;                                                          \
                                                                                                  \
    /* Guard: only run on SM12x devices */                                                        \
    template <typename Base>                                                                      \
    struct Sm12xOnly : Base {                                                                     \
      using typename Base::Params;                                                                \
      CUTLASS_DEVICE                                                                              \
      void operator()(Params const& params, char* smem_buf) {                                     \
        if constexpr (flashinfer::arch::is_major_v<12>) {                                         \
          this->Base::operator()(params, smem_buf);                                               \
        } else {                                                                                  \
          if (cute::thread0()) {                                                                  \
            printf("%s : This kernel shall only run on SM12x devices.\n", __PRETTY_FUNCTION__);   \
            __trap();                                                                             \
          }                                                                                       \
        }                                                                                         \
      }                                                                                           \
    };                                                                                            \
    using GemmKernel =                                                                            \
        Sm12xOnly<cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,           \
                                                       CollectiveMainloop, CollectiveEpilogue,    \
                                                       cutlass::gemm::PersistentScheduler>>;      \
                                                                                                  \
    using Gemm = typename cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;                \
  };                                                                                              \
                                                                                                  \
  template <typename Gemm>                                                                        \
  typename Gemm::Arguments prepareGemmArgsSm120_##T##_##CTA_M_##_##CTA_N_##_##CTA_K_##SWAP_AB_(   \
      void* D, void const* A, void const* B, void const* input_sf, void const* weight_sf, int m,  \
      int n, int k, int batch_count) {                                                            \
    using Sm1xxBlkScaledConfig =                                                                  \
        typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;                      \
    using ElementA = typename Gemm::ElementA;                                                     \
    using ElementB = typename Gemm::ElementB;                                                     \
    using ElementSFA = cutlass::float_ue8m0_t;                                                    \
    using ElementSFB = cutlass::float_ue8m0_t;                                                    \
    using ElementC = void;                                                                        \
    using ElementD = typename Gemm::ElementD;                                                     \
    using ElementCompute = float;                                                                 \
                                                                                                  \
    typename Gemm::Arguments operator_args;                                                       \
    operator_args.mode = cutlass::gemm::GemmUniversalMode::kGemm;                                 \
    auto& fusion_args = operator_args.epilogue.thread;                                            \
    fusion_args.alpha_ptr = nullptr; /* MXFP8 has no global scale */                              \
                                                                                                  \
    operator_args.problem_shape = cute::make_shape(m, n, k, batch_count);                         \
                                                                                                  \
    operator_args.mainloop.ptr_A = static_cast<ElementA const*>(A);                               \
    operator_args.mainloop.ptr_B = static_cast<ElementB const*>(B);                               \
    operator_args.mainloop.ptr_SFA = static_cast<ElementSFA const*>(input_sf);                    \
    operator_args.mainloop.ptr_SFB = static_cast<ElementSFB const*>(weight_sf);                   \
    operator_args.epilogue.ptr_C = static_cast<ElementC const*>(D);                               \
    operator_args.epilogue.ptr_D = static_cast<ElementD*>(D);                                     \
                                                                                                  \
    operator_args.mainloop.dA = cutlass::make_cute_packed_stride(                                 \
        typename Gemm::GemmKernel::StrideA{}, {m, k, batch_count});                               \
    operator_args.mainloop.dB = cutlass::make_cute_packed_stride(                                 \
        typename Gemm::GemmKernel::StrideB{}, {n, k, batch_count});                               \
    operator_args.epilogue.dC = cutlass::make_cute_packed_stride(                                 \
        typename Gemm::GemmKernel::StrideC{}, {m, n, batch_count});                               \
    operator_args.epilogue.dD = operator_args.epilogue.dC;                                        \
                                                                                                  \
    operator_args.mainloop.layout_SFA =                                                           \
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(operator_args.problem_shape);                \
    operator_args.mainloop.layout_SFB =                                                           \
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(operator_args.problem_shape);                \
                                                                                                  \
    if constexpr (!std::is_const_v<decltype(operator_args.scheduler.max_swizzle_size)>) {         \
      operator_args.scheduler.max_swizzle_size = 1;                                               \
    }                                                                                             \
    if constexpr (!std::is_const_v<decltype(operator_args.scheduler.raster_order)>) {             \
      using Enum_t = decltype(operator_args.scheduler.raster_order);                              \
      operator_args.scheduler.raster_order = Enum_t::Heuristic;                                   \
    }                                                                                             \
    /* SM120: ClusterShape is always 1x1x1, no fallback needed */                                 \
    operator_args.hw_info.cluster_shape = dim3(1, 1, 1);                                          \
                                                                                                  \
    return operator_args;                                                                         \
  }                                                                                               \
                                                                                                  \
  template <>                                                                                     \
  size_t genericMxfp8GemmKernelLauncherSm120<T, cute::Int<CTA_M_>, cute::Int<CTA_N_>,             \
                                             cute::Int<CTA_K_>, SWAP_AB_>(                        \
      void* D, void const* A, void const* B, void const* input_sf, void const* weight_sf, int m,  \
      int n, int k, int batch_count, CutlassGemmConfig gemmConfig, char* workspace,               \
      const size_t workspaceBytes, cudaStream_t stream, int* occupancy) {                         \
    using ElementOutput__ =                                                                       \
        typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value,       \
                                                cutlass::half_t, T>::type;                        \
    using ElementOutput_ = typename cutlass::platform::conditional<                               \
        cutlass::platform::is_same<ElementOutput__, float>::value, float, ElementOutput__>::type; \
    using ElementOutput = typename cutlass::platform::conditional<                                \
        cutlass::platform::is_same<ElementOutput_, SafeBF16Sm120>::value, cutlass::bfloat16_t,    \
        ElementOutput_>::type;                                                                    \
                                                                                                  \
    using Mxfp8GemmOperator =                                                                     \
        DeviceGemmMxfp8GemmSm120_##T##_##CTA_M_##_##CTA_N_##_##CTA_K_##SWAP_AB_::Gemm;            \
    Mxfp8GemmOperator gemm;                                                                       \
    auto args = [&]() {                                                                           \
      if constexpr (SWAP_AB_) {                                                                   \
        return prepareGemmArgsSm120_##T##_##CTA_M_##_##CTA_N_##_##CTA_K_##SWAP_AB_<               \
            Mxfp8GemmOperator>(D, B, A, weight_sf, input_sf, n, m, k, batch_count);               \
      } else {                                                                                    \
        return prepareGemmArgsSm120_##T##_##CTA_M_##_##CTA_N_##_##CTA_K_##SWAP_AB_<               \
            Mxfp8GemmOperator>(D, A, B, input_sf, weight_sf, m, n, k, batch_count);               \
      }                                                                                           \
    }();                                                                                          \
    /* Return workspace size */                                                                   \
    size_t required_workspace = gemm.get_workspace_size(args);                                    \
    if (!A && !B && !D) {                                                                         \
      return required_workspace;                                                                  \
    }                                                                                             \
    if (required_workspace > workspaceBytes) {                                                    \
      std::string errMsg("Requested workspace size insufficient. Required " +                     \
                         std::to_string(required_workspace) + ", got " +                          \
                         std::to_string(workspaceBytes));                                         \
      throw std::runtime_error("[MXFP8 SM120 gemm Runner] " + errMsg);                            \
    }                                                                                             \
    auto can_implement = gemm.can_implement(args);                                                \
    if (can_implement != cutlass::Status::kSuccess) {                                             \
      std::string errMsg = "MXFP8 SM120 Gemm cutlass kernel will fail for params. Error: " +      \
                           std::string(cutlass::cutlassGetStatusString(can_implement));           \
      throw std::runtime_error("[MXFP8 SM120 gemm Runner] " + errMsg);                            \
    }                                                                                             \
    auto initStatus = gemm.initialize(args, workspace, stream);                                   \
    if (initStatus != cutlass::Status::kSuccess) {                                                \
      std::string errMsg = "Failed to initialize cutlass MXFP8 gemm on sm120. Error: " +          \
                           std::string(cutlass::cutlassGetStatusString(initStatus));              \
      throw std::runtime_error("[MXFP8 SM120 gemm Runner] " + errMsg);                            \
    }                                                                                             \
    auto runStatus = gemm.run(args, workspace, stream, nullptr, /*enablePDL=*/true);              \
    if (runStatus != cutlass::Status::kSuccess) {                                                 \
      std::string errMsg = "Failed to run cutlass MXFP8 gemm on sm120. Error: " +                 \
                           std::string(cutlass::cutlassGetStatusString(runStatus));               \
      throw std::runtime_error("[MXFP8 SM120 gemm Runner] " + errMsg);                            \
    }                                                                                             \
    return required_workspace;                                                                    \
  }

#endif  // PLACEHOLDER_KERNELS

}  // namespace gemm
}  // namespace flashinfer
#endif  // FLASHINFER_MXFP8_GEMM_TEMPLATE_SM120_H_
