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
#include <algorithm>
#include <cstdint>
#include <flashinfer/blackwell_softmax.cuh>
#include <flashinfer/sampling.cuh>
#include <limits>

#include "tvm_ffi_utils.h"

using namespace flashinfer;
using tvm::ffi::Optional;

namespace {

constexpr int kBootstrapThreads = 256;
constexpr int kRowwiseThreads = 512;
constexpr int kWarpThreads = 128;
constexpr int kMr515Threads = 512;
constexpr int kWarpRowsPerCta = 4;
constexpr int kMaxSplits = 64;
constexpr size_t kBootstrapDynamicSmemBytes = 128;
constexpr size_t kRowwiseDynamicSmemBytes = 128;
constexpr size_t kWarpDynamicSmemBytes = 0;
constexpr size_t kMr515DynamicSmemBytes = 128;

enum class ParameterKind : int {
  kNone = 0,
  kScalar = 1,
  kPerRow = 2,
};

// Stable numeric values are intentionally exposed by softmax_route for tests
// and benchmark evidence. Keep this selection in C++ so an API-level
// correctness test cannot silently pass through OnlineSoftmax fallback.
enum class SoftmaxRoute : int64_t {
  kFallback = 0,
  kWarp = 1,
  kRowwise = 2,
  kBootstrap = 3,
  kMr515V32000T512 = 4,
};

bool use_mr515_none_row(uint32_t rows) {
  switch (rows) {
    case 16:
    case 32:
    case 64:
    case 128:
    case 512:
    case 1024:
      return true;
    default:
      return false;
  }
}

bool use_mr515_kernel(uint32_t rows, uint32_t vocab_size, ParameterKind parameter_kind,
                      float temperature_val, bool enable_pdl, int device_major, int device_minor) {
  if (device_major != 10 || device_minor != 3 || vocab_size != 32000) {
    return false;
  }
  if (parameter_kind == ParameterKind::kNone) {
    return !enable_pdl && use_mr515_none_row(rows);
  }
  return parameter_kind == ParameterKind::kScalar && enable_pdl && rows == 64 &&
         temperature_val == 1.0f;
}

bool use_warp_kernel(uint32_t rows, uint32_t vocab_size, ParameterKind parameter_kind) {
  return rows <= 128 && vocab_size <= 257 &&
         (parameter_kind == ParameterKind::kScalar || parameter_kind == ParameterKind::kPerRow);
}

bool use_rowwise_kernel(uint32_t rows, uint32_t vocab_size, ParameterKind parameter_kind) {
  const bool small_low_row = rows <= 32 && vocab_size <= 16384;
  const bool dense_aligned_mid_row = rows > 128 && rows <= 384 && vocab_size >= 24576 &&
                                     vocab_size <= 256000 && vocab_size % 4 == 0 &&
                                     parameter_kind == ParameterKind::kNone;
  const bool dense_aligned_high_row_narrow = rows > 384 && rows <= 1024 && vocab_size >= 24576 &&
                                             vocab_size <= 32000 && vocab_size % 4 == 0;
  const bool dense_aligned_high_row_wide = rows > 512 && rows <= 1024 && vocab_size > 32000 &&
                                           vocab_size <= 64000 && vocab_size % 4 == 0 &&
                                           parameter_kind == ParameterKind::kNone;
  const bool measured_large_odd = rows > 128 && rows <= 512 && vocab_size >= 24576 &&
                                  vocab_size <= 131072 && vocab_size % 4 != 0;
  return small_low_row || dense_aligned_mid_row || dense_aligned_high_row_narrow ||
         dense_aligned_high_row_wide || measured_large_odd;
}

SoftmaxRoute select_softmax_route(uint32_t rows, uint32_t vocab_size, ParameterKind parameter_kind,
                                  float temperature_val, bool enable_pdl, int device_major,
                                  int device_minor) {
  if (rows == 0 || vocab_size == 0 ||
      static_cast<uint64_t>(rows) * vocab_size >
          static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return SoftmaxRoute::kFallback;
  }
  if (use_mr515_kernel(rows, vocab_size, parameter_kind, temperature_val, enable_pdl, device_major,
                       device_minor)) {
    return SoftmaxRoute::kMr515V32000T512;
  }
  if (use_warp_kernel(rows, vocab_size, parameter_kind)) {
    return SoftmaxRoute::kWarp;
  }
  if (use_rowwise_kernel(rows, vocab_size, parameter_kind)) {
    return SoftmaxRoute::kRowwise;
  }
  return SoftmaxRoute::kBootstrap;
}

cudaError_t launch_noncooperative_kernel(const void* kernel, dim3 grid, dim3 block, void** args,
                                         size_t dynamic_smem_bytes, bool enable_pdl,
                                         cudaStream_t stream) {
  if (!enable_pdl) {
    return cudaLaunchKernel(kernel, grid, block, args, dynamic_smem_bytes, stream);
  }
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = dynamic_smem_bytes;
  config.stream = stream;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = enable_pdl ? 1 : 0;
  config.attrs = &attribute;
  config.numAttrs = 1;
  return cudaLaunchKernelExC(&config, kernel, args);
}

cudaError_t launch_cooperative_kernel(const void* kernel, dim3 grid, dim3 block, void** args,
                                      size_t dynamic_smem_bytes, bool enable_pdl,
                                      cudaStream_t stream) {
  if (!enable_pdl) {
    return cudaLaunchCooperativeKernel(kernel, grid, block, args, dynamic_smem_bytes, stream);
  }
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = dynamic_smem_bytes;
  config.stream = stream;
  cudaLaunchAttribute attributes[2]{};
  attributes[0].id = cudaLaunchAttributeCooperative;
  attributes[0].val.cooperative = 1;
  attributes[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[1].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attributes;
  config.numAttrs = 2;
  return cudaLaunchKernelExC(&config, kernel, args);
}

cudaError_t launch_blackwell_softmax(float* logits, float* output, float* temperature_arr,
                                     float temperature_val, ParameterKind parameter_kind,
                                     uint32_t rows, uint32_t vocab_size, void* workspace,
                                     size_t workspace_bytes, bool enable_pdl, bool is_sm103,
                                     SoftmaxRoute* selected_route, cudaStream_t stream) {
  const SoftmaxRoute route =
      select_softmax_route(rows, vocab_size, parameter_kind, temperature_val, enable_pdl,
                           /*device_major=*/10, is_sm103 ? 3 : 0);
  *selected_route = route;
  if (route == SoftmaxRoute::kFallback) {
    return cudaErrorNotSupported;
  }

  float* parameter = temperature_arr != nullptr ? temperature_arr : logits;
  int rows_i = static_cast<int>(rows);
  int vocab_size_i = static_cast<int>(vocab_size);
  // Keep the existing sm_100a launch path unchanged. The high bit is an
  // sm_103a integration-only PDL flag; adapted kernels strip it before
  // evaluating the original 0/1/2 parameter-kind ABI.
  const bool launch_with_pdl = is_sm103 && enable_pdl;
  int parameter_kind_i = static_cast<int>(parameter_kind) | (launch_with_pdl ? 4 : 0);

  if (route == SoftmaxRoute::kMr515V32000T512) {
    // TEMP_KIND=0 never dereferences this ABI slot. A null dummy tells the
    // adapted frozen payload to execute the required PDL wait/signal pair.
    float* mr515_temperature = launch_with_pdl ? nullptr : parameter;
    void* args[] = {&logits, &mr515_temperature, &output, &temperature_val};
    return launch_noncooperative_kernel(
        reinterpret_cast<const void*>(kernel_mr474_manual_softmax_exp2_t512_vec4), dim3(rows),
        dim3(kMr515Threads), args, kMr515DynamicSmemBytes, launch_with_pdl, stream);
  }

  if (route == SoftmaxRoute::kWarp) {
    void* args[] = {&logits,       &parameter,        &output,         &rows_i,
                    &vocab_size_i, &parameter_kind_i, &temperature_val};
    return launch_noncooperative_kernel(
        reinterpret_cast<const void*>(kernel_flashinfer_blackwell_softmax_followup_warp),
        dim3(ceil_div(rows, static_cast<uint32_t>(kWarpRowsPerCta))), dim3(kWarpThreads), args,
        kWarpDynamicSmemBytes, launch_with_pdl, stream);
  }

  if (route == SoftmaxRoute::kRowwise) {
    void* args[] = {&logits,       &parameter,        &output,         &rows_i,
                    &vocab_size_i, &parameter_kind_i, &temperature_val};
    return launch_noncooperative_kernel(
        reinterpret_cast<const void*>(kernel_flashinfer_blackwell_softmax_followup_rowwise),
        dim3(rows), dim3(kRowwiseThreads), args, kRowwiseDynamicSmemBytes, launch_with_pdl, stream);
  }

  int active_blocks_per_sm = 0;
  cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm, kernel_flashinfer_blackwell_softmax_bootstrap_seed, kBootstrapThreads,
      kBootstrapDynamicSmemBytes);
  if (status != cudaSuccess) {
    return status;
  }
  if (active_blocks_per_sm <= 0) {
    return cudaErrorNotSupported;
  }

  int sm_count = 0;
  int device = 0;
  if ((status = cudaGetDevice(&device)) != cudaSuccess ||
      (status = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device)) !=
          cudaSuccess) {
    return status;
  }

  const uint64_t tiles_per_row =
      ceil_div(static_cast<uint64_t>(vocab_size), static_cast<uint64_t>(kBootstrapThreads));
  const uint64_t cooperative_capacity =
      static_cast<uint64_t>(active_blocks_per_sm) * static_cast<uint64_t>(sm_count);
  const uint64_t provisional_grid = std::max<uint64_t>(
      1, std::min<uint64_t>(static_cast<uint64_t>(rows) * tiles_per_row, cooperative_capacity));
  const uint64_t split_capacity =
      provisional_grid >= rows ? provisional_grid / static_cast<uint64_t>(rows) : 1;
  const int splits = static_cast<int>(
      std::max<uint64_t>(1, std::min<uint64_t>({tiles_per_row, kMaxSplits, split_capacity})));
  const uint64_t grid = std::max<uint64_t>(
      1, std::min<uint64_t>(provisional_grid, static_cast<uint64_t>(rows) * splits));
  const size_t partial_count = static_cast<size_t>(rows) * static_cast<size_t>(splits);
  const size_t required_workspace_bytes = partial_count * 2 * sizeof(float);
  if (required_workspace_bytes > workspace_bytes) {
    return cudaErrorNotSupported;
  }

  auto* partial_max = static_cast<float*>(workspace);
  auto* partial_sum = partial_max + partial_count;
  int splits_i = splits;
  void* args[] = {&logits, &parameter,    &output,   &partial_max,      &partial_sum,
                  &rows_i, &vocab_size_i, &splits_i, &parameter_kind_i, &temperature_val};
  return launch_cooperative_kernel(
      reinterpret_cast<const void*>(kernel_flashinfer_blackwell_softmax_bootstrap_seed),
      dim3(static_cast<uint32_t>(grid)), dim3(kBootstrapThreads), args, kBootstrapDynamicSmemBytes,
      launch_with_pdl, stream);
}

ParameterKind validate_logits_and_temperature(TensorView logits,
                                              Optional<TensorView> maybe_temperature_arr,
                                              bool temperature_is_none) {
  CHECK_INPUT(logits);
  CHECK_DIM(2, logits);
  CHECK_INPUT_TYPE(logits, dl_float32);
  TVM_FFI_ICHECK_GT(logits.size(0), 0) << "logits must contain at least one row";
  TVM_FFI_ICHECK_GT(logits.size(1), 0) << "logits must contain at least one vocabulary entry";
  TVM_FFI_ICHECK_LE(static_cast<uint64_t>(logits.size(0)),
                    static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));
  TVM_FFI_ICHECK_LE(static_cast<uint64_t>(logits.size(1)),
                    static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));

  if (maybe_temperature_arr.has_value()) {
    const TensorView temperature_arr = maybe_temperature_arr.value();
    CHECK_INPUT(temperature_arr);
    CHECK_DIM(1, temperature_arr);
    CHECK_INPUT_TYPE(temperature_arr, dl_float32);
    CHECK_DEVICE(temperature_arr, logits);
    TVM_FFI_ICHECK_EQ(temperature_arr.size(0), logits.size(0))
        << "temperature length must equal logits.size(0)";
    TVM_FFI_ICHECK(!temperature_is_none)
        << "temperature_is_none must be false when a temperature tensor is provided";
    return ParameterKind::kPerRow;
  }
  return temperature_is_none ? ParameterKind::kNone : ParameterKind::kScalar;
}

void validate_softmax_io(TensorView workspace_buffer, TensorView logits, TensorView output) {
  CHECK_INPUT(workspace_buffer);
  CHECK_DIM(1, workspace_buffer);
  CHECK_DEVICE(workspace_buffer, logits);
  CHECK_INPUT(output);
  CHECK_DIM(2, output);
  CHECK_INPUT_TYPE(output, dl_float32);
  CHECK_DEVICE(output, logits);
  CHECK_SHAPE(output, logits);
  TVM_FFI_ICHECK_NE(output.data_ptr(), logits.data_ptr())
      << "output must be fresh and must not alias logits";
}

SoftmaxRoute query_softmax_route(TensorView logits, Optional<TensorView> maybe_temperature_arr,
                                 double temperature_val, bool enable_pdl,
                                 bool temperature_is_none) {
  const ParameterKind parameter_kind =
      validate_logits_and_temperature(logits, maybe_temperature_arr, temperature_is_none);
  ffi::CUDADeviceGuard device_guard(logits.device().device_id);
  int device_major = 0;
  int device_minor = 0;
  cudaError_t status = cudaDeviceGetAttribute(&device_major, cudaDevAttrComputeCapabilityMajor,
                                              logits.device().device_id);
  if (status == cudaSuccess) {
    status = cudaDeviceGetAttribute(&device_minor, cudaDevAttrComputeCapabilityMinor,
                                    logits.device().device_id);
  }
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Blackwell Softmax route query failed with error code " << cudaGetErrorString(status);
  if (device_major != 10 || (device_minor != 0 && device_minor != 3)) {
    return SoftmaxRoute::kFallback;
  }
  return select_softmax_route(
      static_cast<uint32_t>(logits.size(0)), static_cast<uint32_t>(logits.size(1)), parameter_kind,
      static_cast<float>(temperature_val), enable_pdl, device_major, device_minor);
}

void blackwell_softmax_impl(TensorView workspace_buffer, TensorView logits, TensorView output,
                            Optional<TensorView> maybe_temperature_arr, double temperature_val,
                            bool enable_pdl, bool temperature_is_none, bool is_sm103) {
  const ParameterKind parameter_kind =
      validate_logits_and_temperature(logits, maybe_temperature_arr, temperature_is_none);
  validate_softmax_io(workspace_buffer, logits, output);

  const auto rows = static_cast<uint32_t>(logits.size(0));
  const auto vocab_size = static_cast<uint32_t>(logits.size(1));
  const bool has_temperature_arr = maybe_temperature_arr.has_value();

  ffi::CUDADeviceGuard device_guard(logits.device().device_id);
  auto stream = get_stream(logits.device());
  auto* logits_ptr = static_cast<float*>(logits.data_ptr());
  auto* output_ptr = static_cast<float*>(output.data_ptr());
  auto* temperature_ptr =
      has_temperature_arr ? static_cast<float*>(maybe_temperature_arr.value().data_ptr()) : nullptr;
  const size_t workspace_bytes = get_element_size(workspace_buffer) * workspace_buffer.size(0);

  SoftmaxRoute selected_route = SoftmaxRoute::kFallback;
  cudaError_t status = launch_blackwell_softmax(
      logits_ptr, output_ptr, temperature_ptr, static_cast<float>(temperature_val), parameter_kind,
      rows, vocab_size, workspace_buffer.data_ptr(), workspace_bytes, enable_pdl, is_sm103,
      &selected_route, stream);
  // The promoted MR515 route is fail-closed: it must never be hidden by an
  // OnlineSoftmax fallback if its actual launch fails.
  if (status == cudaErrorNotSupported && selected_route != SoftmaxRoute::kMr515V32000T512) {
    status = sampling::OnlineSoftmax<float>(logits_ptr, output_ptr, rows, vocab_size,
                                            temperature_ptr, static_cast<float>(temperature_val),
                                            workspace_buffer.data_ptr(), workspace_bytes,
                                            enable_pdl, stream);
  }
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Blackwell Softmax route " << static_cast<int64_t>(selected_route)
      << " failed with error code " << cudaGetErrorString(status);
}

}  // namespace

void blackwell_softmax(TensorView workspace_buffer, TensorView logits, TensorView output,
                       Optional<TensorView> maybe_temperature_arr, double temperature_val,
                       bool enable_pdl, bool temperature_is_none, bool is_sm103) {
  blackwell_softmax_impl(workspace_buffer, logits, output, maybe_temperature_arr, temperature_val,
                         enable_pdl, temperature_is_none, is_sm103);
}

int64_t blackwell_softmax_route(TensorView logits, Optional<TensorView> maybe_temperature_arr,
                                double temperature_val, bool enable_pdl, bool temperature_is_none) {
  return static_cast<int64_t>(query_softmax_route(logits, maybe_temperature_arr, temperature_val,
                                                  enable_pdl, temperature_is_none));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(softmax, blackwell_softmax);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(softmax_route, blackwell_softmax_route);
