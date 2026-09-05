/*
 * Copyright (c) 2026 by FlashInfer team.
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

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#include "flashinfer/gemm/dual_bf16_weight/dispatch.cuh"
#include "flashinfer/gemm/dual_bf16_weight/kernel_1sm.cuh"
#include "flashinfer/gemm/dual_bf16_weight/kernel_2sm.cuh"
#include "flashinfer/gemm/dual_bf16_weight/kernel_splitk.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer::gemm::dual_bf16_weight {
namespace {

template <class Value>
constexpr Value align_up(Value value, Value alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

bool valid_shape(ProblemShape const& shape) {
  return shape.token_count > 0 && shape.output_channel_count > 0 && shape.reduction_size > 0 &&
         (shape.reduction_size % kReductionTile) == 0;
}

one_sm::OutputType one_sm_output_type(OutputType output_type) {
  return output_type == OutputType::kFloat32 ? one_sm::OutputType::kFloat32
                                             : one_sm::OutputType::kBFloat16;
}

two_sm::OutputType two_sm_output_type(OutputType output_type) {
  return output_type == OutputType::kFloat32 ? two_sm::OutputType::kFloat32
                                             : two_sm::OutputType::kBFloat16;
}

split_k::OutputType split_k_output_type(OutputType output_type) {
  return output_type == OutputType::kFloat32 ? split_k::OutputType::kFloat32
                                             : split_k::OutputType::kBFloat16;
}

template <int StageCount>
constexpr int one_sm_shared_memory_bytes() {
  using namespace cute;
  using MmaAtom =
      SM100_MMA_F16BF16_SS<one_sm::Input, one_sm::Input, float, one_sm::kOutputChannelTile,
                           one_sm::kTokenTile, UMMA::Major::K, UMMA::Major::K>;
  using TiledMma = decltype(make_tiled_mma(MmaAtom{}));
  using WeightMmaShape = decltype(partition_shape_A(
      TiledMma{}, make_shape(Int<one_sm::kOutputChannelTile>{}, Int<one_sm::kReductionTile>{})));
  using ActivationMmaShape = decltype(partition_shape_B(
      TiledMma{}, make_shape(Int<one_sm::kTokenTile>{}, Int<one_sm::kReductionTile>{})));
  using WeightSmemLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<one_sm::Input>{}, append(WeightMmaShape{}, Int<StageCount>{}),
      Step<_1, _2, _3>{}));
  using ActivationSmemLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<one_sm::Input>{}, append(ActivationMmaShape{}, Int<StageCount>{}),
      Step<_1, _2, _3>{}));
  using Storage = one_sm::detail::SharedStorage<WeightSmemLayout, ActivationSmemLayout, StageCount>;
  return sizeof(Storage);
}

cudaError_t current_sm100_properties(cudaDeviceProp* properties) {
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaGetDeviceProperties(properties, device);
  if (status != cudaSuccess) {
    return status;
  }
  if (properties->major != 10 || properties->minor != 0) {
    return cudaErrorInvalidDevice;
  }
  return cudaSuccess;
}

}  // namespace

char const* kernel_kind_name(KernelKind kind) {
  switch (kind) {
    case KernelKind::kSplitK1Sm:
      return "split-K 1SM";
    case KernelKind::kPersistent1Sm:
      return "persistent 1SM";
    case KernelKind::kCluster2Sm:
      return "cluster 2SM";
  }
  return "unknown";
}

char const* output_type_name(OutputType output_type) {
  return output_type == OutputType::kFloat32 ? "FP32" : "BF16";
}

cudaError_t select_kernel_config(ProblemShape const& shape, KernelConfig* config) {
  if (config == nullptr || !valid_shape(shape)) {
    return cudaErrorInvalidValue;
  }

  cudaDeviceProp properties{};
  cudaError_t status = current_sm100_properties(&properties);
  if (status != cudaSuccess) {
    return status;
  }

  *config = KernelConfig{};
  config->shape = shape;

  // Split-K requires at least two 128-wide reduction tiles. K=128 is routed
  // through the non-split 1SM/2SM paths instead of returning an invalid config.
  if (shape.token_count <= kSplitKMaximumTokenCount && shape.reduction_size >= 2 * kReductionTile) {
    split_k::KernelConfig selected =
        split_k::select_kernel_config(shape.token_count, shape.output_channel_count,
                                      shape.reduction_size, properties.multiProcessorCount);
    if (selected.split_k == 0) {
      return cudaErrorInvalidValue;
    }

    config->kind = KernelKind::kSplitK1Sm;
    config->output_channel_tile = split_k::kOutputChannelTile;
    config->token_tile = selected.token_tile;
    config->split_k = selected.split_k;
    config->stage_count = selected.stage_count;
    config->grid_size = selected.grid_size;
    config->shared_memory_bytes = selected.shared_memory_bytes;
    config->partial_workspace_bytes = align_up<std::size_t>(
        split_k::partial_workspace_bytes(selected, shape.token_count, shape.output_channel_count),
        alignof(int));
    config->counter_workspace_bytes = split_k::counter_workspace_bytes(selected);
    config->workspace_bytes = config->partial_workspace_bytes + config->counter_workspace_bytes;
    return cudaSuccess;
  }

  if (shape.token_count < kTwoSmMinimumTokenCount &&
      (shape.output_channel_count % one_sm::kOutputChannelTile) == 0) {
    config->kind = KernelKind::kPersistent1Sm;
    config->output_channel_tile = one_sm::kOutputChannelTile;
    config->token_tile = one_sm::kTokenTile;
    config->split_k = 1;
    config->stage_count = 6;
    config->shared_memory_bytes = one_sm_shared_memory_bytes<6>();
    int output_tiles = shape.output_channel_count / one_sm::kOutputChannelTile;
    int token_tiles = (shape.token_count + one_sm::kTokenTile - 1) / one_sm::kTokenTile;
    int total_tiles = output_tiles * token_tiles;
    config->grid_size =
        total_tiles < properties.multiProcessorCount ? total_tiles : properties.multiProcessorCount;
    return cudaSuccess;
  }

  two_sm::KernelConfig selected =
      two_sm::select_kernel_config(shape.token_count, shape.output_channel_count);
  if (selected.output_channel_tile == 0 || selected.token_tile == 0) {
    return cudaErrorInvalidValue;
  }

  config->kind = KernelKind::kCluster2Sm;
  config->output_channel_tile = selected.output_channel_tile;
  config->token_tile = selected.token_tile;
  config->split_k = 1;
  config->stage_count = selected.stage_count;
  config->shared_memory_bytes = selected.shared_memory_bytes;
  int output_tiles = (shape.output_channel_count + selected.output_channel_tile - 1) /
                     selected.output_channel_tile;
  int token_tiles = (shape.token_count + selected.token_tile - 1) / selected.token_tile;
  int cluster_count = output_tiles * token_tiles;
  int resident_cluster_limit = properties.multiProcessorCount / 2;
  if (cluster_count > resident_cluster_limit) {
    cluster_count = resident_cluster_limit;
  }
  config->grid_size = cluster_count * 2;
  config->used_compatibility_fallback = shape.token_count < kTwoSmMinimumTokenCount;
  return cudaSuccess;
}

cudaError_t launch(Arguments const& arguments, KernelConfig const& config, void* workspace,
                   std::size_t workspace_bytes, cudaStream_t stream) {
  if (arguments.output == nullptr || arguments.activation == nullptr ||
      arguments.weight_high == nullptr || arguments.weight_low == nullptr ||
      arguments.token_count != config.shape.token_count ||
      arguments.output_channel_count != config.shape.output_channel_count ||
      arguments.reduction_size != config.shape.reduction_size ||
      (arguments.output_type != OutputType::kFloat32 &&
       arguments.output_type != OutputType::kBFloat16)) {
    return cudaErrorInvalidValue;
  }

  if (config.kind == KernelKind::kSplitK1Sm) {
    if (workspace == nullptr || workspace_bytes < config.workspace_bytes) {
      return cudaErrorInvalidValue;
    }
    auto workspace_address = reinterpret_cast<std::uintptr_t>(workspace);
    float* partial_output = reinterpret_cast<float*>(workspace_address);
    int* counters = reinterpret_cast<int*>(workspace_address + config.partial_workspace_bytes);

    // A tiny per-tile memset is deliberately part of the captured stream work:
    // it makes arbitrary caller-owned buffers and recovery after failed launches
    // correct. The split-K kernel also resets counters after successful use.
    cudaError_t status = cudaMemsetAsync(counters, 0, config.counter_workspace_bytes, stream);
    if (status != cudaSuccess) {
      return status;
    }

    split_k::Arguments kernel_arguments{arguments.output,
                                        partial_output,
                                        counters,
                                        arguments.activation,
                                        arguments.weight_high,
                                        arguments.weight_low,
                                        arguments.token_count,
                                        arguments.output_channel_count,
                                        arguments.reduction_size,
                                        split_k_output_type(arguments.output_type)};
    return split_k::launch(kernel_arguments, config.split_k, config.stage_count, stream);
  }

  if (config.kind == KernelKind::kPersistent1Sm) {
    one_sm::Arguments kernel_arguments{arguments.output,
                                       arguments.activation,
                                       arguments.weight_high,
                                       arguments.weight_low,
                                       arguments.token_count,
                                       arguments.output_channel_count,
                                       arguments.reduction_size,
                                       kLowScale,
                                       one_sm_output_type(arguments.output_type)};
    return one_sm::launch(kernel_arguments, config.stage_count, stream);
  }

  two_sm::Arguments kernel_arguments{
      arguments.output,         arguments.activation,
      arguments.weight_high,    arguments.weight_low,
      arguments.token_count,    arguments.output_channel_count,
      arguments.reduction_size, two_sm_output_type(arguments.output_type)};
  return two_sm::launch(kernel_arguments, stream);
}

}  // namespace flashinfer::gemm::dual_bf16_weight

namespace flashinfer::gemm {
namespace {

using dual_bf16_weight::Arguments;
using dual_bf16_weight::Input;
using dual_bf16_weight::KernelConfig;
using dual_bf16_weight::OutputType;
using dual_bf16_weight::ProblemShape;

void check_problem_shape(int64_t m, int64_t n, int64_t k) {
  TVM_FFI_ICHECK_GT(m, 0) << "M must be positive";
  TVM_FFI_ICHECK_GT(n, 0) << "N must be positive";
  TVM_FFI_ICHECK_GT(k, 0) << "K must be positive";
  TVM_FFI_ICHECK_EQ(k % dual_bf16_weight::kReductionTile, 0)
      << "K must be a multiple of " << dual_bf16_weight::kReductionTile;
  TVM_FFI_ICHECK_LE(m, std::numeric_limits<int>::max()) << "M exceeds the int32 dispatcher range";
  TVM_FFI_ICHECK_LE(n, std::numeric_limits<int>::max()) << "N exceeds the int32 dispatcher range";
  TVM_FFI_ICHECK_LE(k, std::numeric_limits<int>::max()) << "K exceeds the int32 dispatcher range";
}

void check_tma_alignment(TensorView tensor, char const* name) {
  constexpr std::uintptr_t kTmaAlignment = 16;
  auto address = reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
  TVM_FFI_ICHECK_EQ(address % kTmaAlignment, 0)
      << name << " data pointer must be 16-byte aligned for TMA";
}

KernelConfig make_config(int64_t m, int64_t n, int64_t k) {
  check_problem_shape(m, n, k);
  KernelConfig config{};
  cudaError_t status = dual_bf16_weight::select_kernel_config(
      ProblemShape{static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)}, &config);
  if (status == cudaErrorInvalidDevice) {
    TVM_FFI_THROW(ValueError)
        << "dual BF16 weight GEMM requires an exact SM100 (compute capability "
           "10.0) device";
  }
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "failed to select dual BF16 weight GEMM configuration: " << cudaGetErrorString(status);
  return config;
}

}  // namespace

int64_t dual_bf16_weight_gemm_workspace_size(int64_t m, int64_t n, int64_t k, int64_t device_id) {
  ffi::CUDADeviceGuard device_guard(static_cast<int>(device_id));
  auto config = make_config(m, n, k);
  TVM_FFI_ICHECK_LE(config.workspace_bytes,
                    static_cast<std::size_t>(std::numeric_limits<int64_t>::max()));
  return static_cast<int64_t>(config.workspace_bytes);
}

int64_t dual_bf16_weight_gemm_kernel_kind(int64_t m, int64_t n, int64_t k, int64_t device_id) {
  ffi::CUDADeviceGuard device_guard(static_cast<int>(device_id));
  auto config = make_config(m, n, k);
  return static_cast<int64_t>(config.kind);
}

void dual_bf16_weight_gemm(TensorView activation, TensorView weight_high, TensorView weight_low,
                           TensorView output, TensorView workspace_buffer) {
  CHECK_INPUT_AND_TYPE(activation, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(weight_high, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(weight_low, dl_bfloat16);
  CHECK_INPUT(workspace_buffer);
  CHECK_INPUT(output);
  CHECK_INPUT_TYPE(workspace_buffer, dl_uint8);

  TVM_FFI_ICHECK(output.dtype() == dl_bfloat16 || output.dtype() == dl_float32)
      << "output must be bfloat16 or float32";
  TVM_FFI_ICHECK_EQ(activation.ndim(), 2) << "activation must have shape [M, K]";
  TVM_FFI_ICHECK_EQ(weight_high.ndim(), 2) << "weight_high must have shape [N, K]";
  TVM_FFI_ICHECK_EQ(weight_low.ndim(), 2) << "weight_low must have shape [N, K]";
  TVM_FFI_ICHECK_EQ(output.ndim(), 2) << "output must have shape [M, N]";

  CHECK_DEVICE(activation, weight_high);
  CHECK_DEVICE(activation, weight_low);
  CHECK_DEVICE(activation, output);
  CHECK_DEVICE(activation, workspace_buffer);

  int64_t m = activation.size(0);
  int64_t k = activation.size(1);
  int64_t n = weight_high.size(0);
  TVM_FFI_ICHECK_EQ(weight_high.size(1), k)
      << "weight_high must have shape [N, K] with K matching activation";
  TVM_FFI_ICHECK_EQ(weight_low.size(0), n) << "weight_low and weight_high must have the same shape";
  TVM_FFI_ICHECK_EQ(weight_low.size(1), k) << "weight_low and weight_high must have the same shape";
  TVM_FFI_ICHECK_EQ(output.size(0), m) << "output must have shape [M, N]";
  TVM_FFI_ICHECK_EQ(output.size(1), n) << "output must have shape [M, N]";

  ffi::CUDADeviceGuard device_guard(activation.device().device_id);
  KernelConfig config = make_config(m, n, k);
  check_tma_alignment(activation, "activation");
  check_tma_alignment(weight_high, "weight_high");
  check_tma_alignment(weight_low, "weight_low");
  check_tma_alignment(output, "output");
  std::size_t provided_workspace_bytes =
      static_cast<std::size_t>(workspace_buffer.numel()) *
      static_cast<std::size_t>(get_element_size(workspace_buffer));
  TVM_FFI_ICHECK_GE(provided_workspace_bytes, config.workspace_bytes)
      << "workspace_buffer is too small: need " << config.workspace_bytes << " bytes, got "
      << provided_workspace_bytes;
  if (config.workspace_bytes != 0) {
    check_tma_alignment(workspace_buffer, "workspace_buffer");
  }

  OutputType output_type =
      output.dtype() == dl_float32 ? OutputType::kFloat32 : OutputType::kBFloat16;
  Arguments arguments{output.data_ptr(),
                      static_cast<Input const*>(activation.data_ptr()),
                      static_cast<Input const*>(weight_high.data_ptr()),
                      static_cast<Input const*>(weight_low.data_ptr()),
                      static_cast<int>(m),
                      static_cast<int>(n),
                      static_cast<int>(k),
                      output_type};

  void* workspace = config.workspace_bytes == 0 ? nullptr : workspace_buffer.data_ptr();
  cudaError_t status = dual_bf16_weight::launch(
      arguments, config, workspace, provided_workspace_bytes, get_stream(activation.device()));
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "dual BF16 weight GEMM launch failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer::gemm

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::gemm::dual_bf16_weight_gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(workspace_size,
                              flashinfer::gemm::dual_bf16_weight_gemm_workspace_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel_kind, flashinfer::gemm::dual_bf16_weight_gemm_kernel_kind);
