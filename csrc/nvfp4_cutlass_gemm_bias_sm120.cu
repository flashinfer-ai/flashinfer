/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "flashinfer/cutlass_utils.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace gemm {
namespace cutlass_bias_sm120 {

using namespace cute;

template <int CtaM, int CtaN, int CtaK, bool SwapAB, bool UseStreamK>
struct KernelConfig {
  using Output = cutlass::bfloat16_t;
  using CtaShape = Shape<Int<CtaM>, Int<CtaN>, Int<CtaK>>;
  using ClusterShape = Shape<_1, _1, _1>;
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutOutput =
      std::conditional_t<SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  using ElementC = void;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using FusionOperation = std::conditional_t<
      SwapAB, cutlass::epilogue::fusion::LinCombPerRowBias<Output, float, Output, ElementC, float>,
      cutlass::epilogue::fusion::LinCombPerColBias<Output, float, Output, ElementC, float>>;

  static constexpr int kAlignmentAB = 32;
  static constexpr int kAlignmentOutput = 128 / cutlass::sizeof_bits<Output>::value;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, CtaShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC,
      LayoutOutput, kAlignmentOutput, Output, LayoutOutput, kAlignmentOutput,
      cutlass::epilogue::TmaWarpSpecialized, FusionOperation>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementA, LayoutA,
      kAlignmentAB, ElementB, LayoutB, kAlignmentAB, ElementAccumulator, CtaShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  using Scheduler = std::conditional_t<UseStreamK, cutlass::gemm::StreamKScheduler,
                                       cutlass::gemm::StaticPersistentScheduler>;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                           CollectiveEpilogue, Scheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Gemm>
typename Gemm::Arguments PrepareArguments(void* output, void const* activation, void const* weight,
                                          void const* activation_scale, void const* weight_scale,
                                          float const* alpha, void const* bias, int m, int n,
                                          int k) {
  using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using Output = typename Gemm::ElementD;

  typename Gemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = make_shape(m, n, k, 1);
  args.mainloop.ptr_A = static_cast<cutlass::float_e2m1_t const*>(activation);
  args.mainloop.ptr_B = static_cast<cutlass::float_e2m1_t const*>(weight);
  args.mainloop.ptr_SFA = static_cast<cutlass::float_ue4m3_t const*>(activation_scale);
  args.mainloop.ptr_SFB = static_cast<cutlass::float_ue4m3_t const*>(weight_scale);
  args.mainloop.dA =
      cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, make_shape(m, k, 1));
  args.mainloop.dB =
      cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, make_shape(n, k, 1));
  args.mainloop.layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(args.problem_shape);
  args.mainloop.layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(args.problem_shape);

  args.epilogue.ptr_C = nullptr;
  args.epilogue.ptr_D = static_cast<Output*>(output);
  args.epilogue.dC =
      cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, make_shape(m, n, 1));
  args.epilogue.dD = args.epilogue.dC;
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;
  args.epilogue.thread.alpha_ptr = alpha;
  args.epilogue.thread.beta_ptr = nullptr;
  args.epilogue.thread.bias_ptr = static_cast<Output const*>(bias);

  if constexpr (!std::is_const_v<decltype(args.scheduler.max_swizzle_size)>) {
    args.scheduler.max_swizzle_size = 1;
  }
  if constexpr (!std::is_const_v<decltype(args.scheduler.raster_order)>) {
    using RasterOrder = decltype(args.scheduler.raster_order);
    args.scheduler.raster_order = RasterOrder::Heuristic;
  }
  args.hw_info.cluster_shape = dim3(1, 1, 1);
  args.hw_info.cluster_shape_fallback = dim3(1, 1, 1);
  return args;
}

template <int CtaM, int CtaN, int CtaK, bool SwapAB, bool UseStreamK>
std::size_t Launch(void* output, void const* activation, void const* weight,
                   void const* activation_scale, void const* weight_scale, float const* alpha,
                   void const* bias, int m, int n, int k, void* workspace,
                   std::size_t workspace_bytes, cudaStream_t stream) {
  using Gemm = typename KernelConfig<CtaM, CtaN, CtaK, SwapAB, UseStreamK>::Gemm;
  Gemm gemm;
  auto args = PrepareArguments<Gemm>(output, activation, weight, activation_scale, weight_scale,
                                     alpha, bias, m, n, k);
  std::size_t const required = gemm.get_workspace_size(args);
  if (activation == nullptr && weight == nullptr && output == nullptr) return required;
  if (required > workspace_bytes) {
    throw std::runtime_error("CUTLASS NVFP4 bias GEMM workspace is too small: required " +
                             std::to_string(required) + ", got " + std::to_string(workspace_bytes));
  }
  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS NVFP4 bias GEMM cannot implement tactic: " +
                             std::string(cutlass::cutlassGetStatusString(status)));
  }
  status = gemm.initialize(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS NVFP4 bias GEMM initialization failed: " +
                             std::string(cutlass::cutlassGetStatusString(status)));
  }
  status = gemm.run(args, workspace, stream, nullptr, /*enable_pdl=*/true);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS NVFP4 bias GEMM launch failed: " +
                             std::string(cutlass::cutlassGetStatusString(status)));
  }
  return required;
}

#define DISPATCH_TILE(CTA_M, CTA_N, CTA_K)                                                      \
  switch (tactic % 4) {                                                                         \
    case 0:                                                                                     \
      return Launch<CTA_M, CTA_N, CTA_K, true, false>(output, weight, activation, weight_scale, \
                                                      activation_scale, alpha, bias, n, m, k,   \
                                                      workspace, workspace_bytes, stream);      \
    case 1:                                                                                     \
      return Launch<CTA_M, CTA_N, CTA_K, false, false>(                                         \
          output, activation, weight, activation_scale, weight_scale, alpha, bias, m, n, k,     \
          workspace, workspace_bytes, stream);                                                  \
    case 2:                                                                                     \
      return Launch<CTA_M, CTA_N, CTA_K, true, true>(output, weight, activation, weight_scale,  \
                                                     activation_scale, alpha, bias, n, m, k,    \
                                                     workspace, workspace_bytes, stream);       \
    default:                                                                                    \
      return Launch<CTA_M, CTA_N, CTA_K, false, true>(                                          \
          output, activation, weight, activation_scale, weight_scale, alpha, bias, m, n, k,     \
          workspace, workspace_bytes, stream);                                                  \
  }

std::size_t Dispatch(void* output, void const* activation, void const* weight,
                     void const* activation_scale, void const* weight_scale, float const* alpha,
                     void const* bias, int m, int n, int k, void* workspace,
                     std::size_t workspace_bytes, cudaStream_t stream, int tactic) {
  if (tactic < 0 || tactic >= 32) {
    throw std::invalid_argument("CUTLASS NVFP4 bias GEMM tactic must be in [0, 32)");
  }
  switch (tactic / 4) {
    case 0:
      DISPATCH_TILE(128, 32, 128);
    case 1:
      DISPATCH_TILE(128, 32, 256);
    case 2:
      DISPATCH_TILE(128, 64, 128);
    case 3:
      DISPATCH_TILE(128, 64, 256);
    case 4:
      DISPATCH_TILE(128, 128, 128);
    case 5:
      DISPATCH_TILE(128, 128, 256);
    case 6:
      DISPATCH_TILE(256, 128, 128);
    default:
      DISPATCH_TILE(128, 256, 128);
  }
}

#undef DISPATCH_TILE

}  // namespace cutlass_bias_sm120
}  // namespace gemm
}  // namespace flashinfer

namespace torch_ext {
namespace {

constexpr auto kPackedFp4 = dl_uint8;
constexpr auto kScaleType = dl_uint8;

void CheckProblem(TensorView activation, TensorView weight, TensorView activation_scale,
                  TensorView weight_scale, TensorView alpha, TensorView bias, TensorView output,
                  TensorView workspace) {
  CHECK_INPUT_AND_TYPE(activation, kPackedFp4);
  CHECK_INPUT_AND_TYPE(weight, kPackedFp4);
  CHECK_INPUT_AND_TYPE(activation_scale, kScaleType);
  CHECK_INPUT_AND_TYPE(weight_scale, kScaleType);
  CHECK_INPUT_AND_TYPE(alpha, dl_float32);
  CHECK_INPUT_AND_TYPE(bias, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(output, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(workspace, dl_uint8);
  CHECK_DEVICE(weight, activation);
  CHECK_DEVICE(activation_scale, activation);
  CHECK_DEVICE(weight_scale, activation);
  CHECK_DEVICE(alpha, activation);
  CHECK_DEVICE(bias, activation);
  CHECK_DEVICE(output, activation);
  CHECK_DEVICE(workspace, activation);
  CHECK_DIM(2, activation);
  CHECK_DIM(2, weight);
  CHECK_DIM(1, bias);
  CHECK_DIM(2, output);

  int64_t const m = activation.size(0);
  int64_t const k_packed = activation.size(1);
  int64_t const n = weight.size(0);
  TVM_FFI_ICHECK_EQ(weight.size(1), k_packed) << "packed K dimensions must match";
  TVM_FFI_ICHECK_EQ(output.size(0), m) << "output M dimension mismatch";
  TVM_FFI_ICHECK_EQ(output.size(1), n) << "output N dimension mismatch";
  TVM_FFI_ICHECK_EQ(bias.size(0), n) << "bias must have shape [N]";
  TVM_FFI_ICHECK_EQ(alpha.numel(), 1) << "alpha must contain one device f32 value";
}

}  // namespace

int64_t nvfp4_cutlass_gemm_bias_workspace_size(int64_t m, int64_t n, int64_t k, int64_t tactic) {
  return static_cast<int64_t>(flashinfer::gemm::cutlass_bias_sm120::Dispatch(
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, static_cast<int>(m),
      static_cast<int>(n), static_cast<int>(k), nullptr, 0, nullptr, static_cast<int>(tactic)));
}

void nvfp4_cutlass_gemm_bias(TensorView activation, TensorView weight, TensorView activation_scale,
                             TensorView weight_scale, TensorView alpha, TensorView bias,
                             TensorView output, TensorView workspace, int64_t tactic) {
  CheckProblem(activation, weight, activation_scale, weight_scale, alpha, bias, output, workspace);
  int const m = static_cast<int>(activation.size(0));
  int const n = static_cast<int>(weight.size(0));
  int const k = static_cast<int>(activation.size(1) * 2);
  flashinfer::gemm::cutlass_bias_sm120::Dispatch(
      output.data_ptr(), activation.data_ptr(), weight.data_ptr(), activation_scale.data_ptr(),
      weight_scale.data_ptr(), static_cast<float const*>(alpha.data_ptr()), bias.data_ptr(), m, n,
      k, workspace.data_ptr(), static_cast<std::size_t>(workspace.numel()),
      get_stream(activation.device()), static_cast<int>(tactic));
}

int64_t nvfp4_cutlass_gemm_bias_tactic_num() { return 32; }

}  // namespace torch_ext

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_cutlass_gemm_bias, torch_ext::nvfp4_cutlass_gemm_bias);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_cutlass_gemm_bias_workspace_size,
                              torch_ext::nvfp4_cutlass_gemm_bias_workspace_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_cutlass_gemm_bias_tactic_num,
                              torch_ext::nvfp4_cutlass_gemm_bias_tactic_num);
