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
constexpr int kMaxSplits = 64;
constexpr size_t kDynamicSmemBytes = 128;

enum class ParameterKind : int {
  kNone = 0,
  kScalar = 1,
  kPerRow = 2,
};

bool use_rowwise_kernel(uint32_t rows, uint32_t vocab_size, ParameterKind parameter_kind) {
  const bool small_low_row = rows <= 32 && vocab_size <= 16384;
  const bool dense_aligned_mid_row = rows > 128 && rows <= 384 && vocab_size >= 24576 &&
                                     vocab_size <= 256000 && vocab_size % 4 == 0 &&
                                     parameter_kind == ParameterKind::kNone;
  const bool dense_aligned_high_row_narrow = rows > 384 && rows <= 1024 && vocab_size >= 24576 &&
                                             vocab_size <= 32000 && vocab_size % 4 == 0 &&
                                             parameter_kind == ParameterKind::kNone;
  const bool measured_large_odd = rows > 128 && rows <= 512 && vocab_size >= 24576 &&
                                  vocab_size <= 131072 && vocab_size % 4 != 0;
  return small_low_row || dense_aligned_mid_row || dense_aligned_high_row_narrow ||
         measured_large_odd;
}

cudaError_t launch_blackwell_softmax(float* logits, float* output, float* temperature_arr,
                                     float temperature_val, ParameterKind parameter_kind,
                                     uint32_t rows, uint32_t vocab_size, void* workspace,
                                     size_t workspace_bytes, cudaStream_t stream) {
  if (rows == 0 || vocab_size == 0 ||
      static_cast<uint64_t>(rows) * vocab_size >
          static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return cudaErrorNotSupported;
  }

  float* parameter = temperature_arr != nullptr ? temperature_arr : logits;
  int rows_i = static_cast<int>(rows);
  int vocab_size_i = static_cast<int>(vocab_size);
  int parameter_kind_i = static_cast<int>(parameter_kind);

  if (use_rowwise_kernel(rows, vocab_size, parameter_kind)) {
    void* args[] = {&logits,       &parameter,        &output,         &rows_i,
                    &vocab_size_i, &parameter_kind_i, &temperature_val};
    return cudaLaunchKernel(
        reinterpret_cast<const void*>(kernel_flashinfer_blackwell_softmax_followup_rowwise),
        dim3(rows), dim3(kRowwiseThreads), args, kDynamicSmemBytes, stream);
  }

  int active_blocks_per_sm = 0;
  cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm, kernel_flashinfer_blackwell_softmax_bootstrap_seed, kBootstrapThreads,
      kDynamicSmemBytes);
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
  return cudaLaunchCooperativeKernel(
      reinterpret_cast<const void*>(kernel_flashinfer_blackwell_softmax_bootstrap_seed),
      dim3(static_cast<uint32_t>(grid)), dim3(kBootstrapThreads), args, kDynamicSmemBytes, stream);
}

}  // namespace

void blackwell_softmax(TensorView workspace_buffer, TensorView logits, TensorView output,
                       Optional<TensorView> maybe_temperature_arr, double temperature_val,
                       bool enable_pdl, bool temperature_is_none) {
  CHECK_INPUT(workspace_buffer);
  CHECK_INPUT(logits);
  CHECK_INPUT(output);
  CHECK_DIM(2, logits);

  const auto rows = static_cast<uint32_t>(logits.size(0));
  const auto vocab_size = static_cast<uint32_t>(logits.size(1));
  const bool has_temperature_arr = maybe_temperature_arr.has_value();
  const ParameterKind parameter_kind =
      has_temperature_arr ? ParameterKind::kPerRow
                          : (temperature_is_none ? ParameterKind::kNone : ParameterKind::kScalar);

  ffi::CUDADeviceGuard device_guard(logits.device().device_id);
  auto stream = get_stream(logits.device());
  auto* logits_ptr = static_cast<float*>(logits.data_ptr());
  auto* output_ptr = static_cast<float*>(output.data_ptr());
  auto* temperature_ptr =
      has_temperature_arr ? static_cast<float*>(maybe_temperature_arr.value().data_ptr()) : nullptr;
  const size_t workspace_bytes = get_element_size(workspace_buffer) * workspace_buffer.size(0);

  cudaError_t status = launch_blackwell_softmax(
      logits_ptr, output_ptr, temperature_ptr, static_cast<float>(temperature_val), parameter_kind,
      rows, vocab_size, workspace_buffer.data_ptr(), workspace_bytes, stream);
  if (status == cudaErrorNotSupported) {
    status = sampling::OnlineSoftmax<float>(logits_ptr, output_ptr, rows, vocab_size,
                                            temperature_ptr, static_cast<float>(temperature_val),
                                            workspace_buffer.data_ptr(), workspace_bytes,
                                            enable_pdl, stream);
  }
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Blackwell Softmax failed with error code " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(softmax, blackwell_softmax);
