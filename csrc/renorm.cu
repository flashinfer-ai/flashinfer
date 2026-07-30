/*
 * Copyright (c) 2024 by FlashInfer team.
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
#include <cstdint>
#include <flashinfer/air_top_p.cuh>
#include <flashinfer/sampling.cuh>

#include "sampling_utils.h"
#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Optional;

namespace {

struct MemoryRange {
  std::uintptr_t begin;
  std::uintptr_t end;
};

MemoryRange get_memory_range(const TensorView& tensor) {
  auto begin = reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
  auto size = static_cast<std::uintptr_t>(tensor.numel() * get_element_size(tensor));
  return {begin, begin + size};
}

bool overlaps(const MemoryRange& lhs, const MemoryRange& rhs) {
  return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

void check_no_overlap(const TensorView& lhs, const TensorView& rhs, const char* lhs_name,
                      const char* rhs_name) {
  if (overlaps(get_memory_range(lhs), get_memory_range(rhs))) {
    TVM_FFI_THROW(ValueError) << lhs_name << " must not overlap " << rhs_name;
  }
}

}  // namespace

void top_p_renorm_probs(TensorView probs, TensorView renorm_probs,
                        Optional<TensorView> maybe_top_p_arr, double top_p_val,
                        bool is_deterministic, TensorView workspace) {
  CHECK_INPUT_AND_TYPE(probs, dl_float32);
  CHECK_INPUT_AND_TYPE(renorm_probs, dl_float32);
  CHECK_INPUT_AND_TYPE(workspace, dl_uint8);
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  CHECK_SHAPE(probs, renorm_probs);
  CHECK_DEVICE(probs, renorm_probs);
  CHECK_DEVICE(probs, workspace);
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  check_tensor_param(maybe_top_p_arr, probs);
  bool has_top_p_arr = maybe_top_p_arr.has_value();
  if (has_top_p_arr) {
    CHECK_INPUT_AND_TYPE(maybe_top_p_arr.value(), dl_float32);
    CHECK_DEVICE(probs, maybe_top_p_arr.value());
  }

  auto probs_range = get_memory_range(probs);
  auto renorm_probs_range = get_memory_range(renorm_probs);
  if (overlaps(probs_range, renorm_probs_range) && (probs_range.begin != renorm_probs_range.begin ||
                                                    probs_range.end != renorm_probs_range.end)) {
    TVM_FFI_THROW(ValueError) << "probs and renorm_probs must either alias exactly or not overlap";
  }
  check_no_overlap(workspace, probs, "workspace", "probs");
  check_no_overlap(workspace, renorm_probs, "workspace", "renorm_probs");
  if (has_top_p_arr) {
    check_no_overlap(workspace, maybe_top_p_arr.value(), "workspace", "top_p");
    check_no_overlap(renorm_probs, maybe_top_p_arr.value(), "renorm_probs", "top_p");
  }
  if (vocab_size >= sampling::air_top_p::NUM_BUCKETS &&
      reinterpret_cast<std::uintptr_t>(workspace.data_ptr()) %
              alignof(sampling::air_top_p::Counter<float>) !=
          0) {
    TVM_FFI_THROW(ValueError) << "workspace must be aligned to "
                              << alignof(sampling::air_top_p::Counter<float>) << " bytes";
  }

  ffi::CUDADeviceGuard device_guard(probs.device().device_id);
  auto stream = get_stream(probs.device());

  float* top_p_arr_ptr =
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr.value().data_ptr()) : nullptr;

  cudaError_t status;
  // Fallback to ternary search for small vocab where radix precision is insufficient
  if (vocab_size < sampling::air_top_p::NUM_BUCKETS) {
    status = sampling::TopPRenormProb<float>(
        static_cast<float*>(probs.data_ptr()), static_cast<float*>(renorm_probs.data_ptr()),
        top_p_arr_ptr, batch_size, top_p_val, vocab_size, stream);
  } else if (is_deterministic) {
    status = sampling::air_top_p::AirTopPRenormProb<true, float>(
        static_cast<float*>(probs.data_ptr()), static_cast<float*>(renorm_probs.data_ptr()),
        top_p_arr_ptr, batch_size, top_p_val, vocab_size, workspace.data_ptr(), stream);
  } else {
    status = sampling::air_top_p::AirTopPRenormProb<false, float>(
        static_cast<float*>(probs.data_ptr()), static_cast<float*>(renorm_probs.data_ptr()),
        top_p_arr_ptr, batch_size, top_p_val, vocab_size, workspace.data_ptr(), stream);
  }
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "TopPRenormProb failed with error code " << cudaGetErrorString(status);
}

void top_k_renorm_probs(TensorView probs, TensorView renorm_probs,
                        Optional<TensorView> maybe_top_k_arr, int64_t top_k_val,
                        TensorView row_states_buffer) {
  CHECK_INPUT(probs);
  CHECK_INPUT(row_states_buffer);
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  check_tensor_param(maybe_top_k_arr, probs);
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  ffi::CUDADeviceGuard device_guard(probs.device().device_id);
  auto stream = get_stream(probs.device());

  cudaError_t status;
  auto dtype = probs.dtype();

  // Use radix-based top-k with dtype dispatch for FP32/FP16/BF16
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(dtype, c_type, [&] {
    status = sampling::RadixTopKRenormProbMultiCTA<c_type, int>(
        static_cast<c_type*>(probs.data_ptr()), static_cast<c_type*>(renorm_probs.data_ptr()),
        has_top_k_arr ? static_cast<int*>(maybe_top_k_arr.value().data_ptr()) : nullptr, batch_size,
        top_k_val, vocab_size, static_cast<sampling::RadixRowState*>(row_states_buffer.data_ptr()),
        stream);
    return true;
  });

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "TopKRenormProb failed with error code " << cudaGetErrorString(status);
}

void top_k_mask_logits(TensorView logits, TensorView mask_logits,
                       Optional<TensorView> maybe_top_k_arr, int64_t top_k_val,
                       TensorView row_states_buffer) {
  CHECK_INPUT(logits);
  CHECK_INPUT(row_states_buffer);
  CHECK_DIM(2, logits);  // logits: (batch_size, vocab_size)
  unsigned int batch_size = logits.size(0);
  unsigned int vocab_size = logits.size(1);
  check_tensor_param(maybe_top_k_arr, logits);
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  ffi::CUDADeviceGuard device_guard(logits.device().device_id);
  auto stream = get_stream(logits.device());

  cudaError_t status;
  auto dtype = logits.dtype();

  // Use radix-based top-k with auto-selection (single-CTA for small vocab, multi-CTA for large
  // vocab)
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(dtype, c_type, [&] {
    status = sampling::RadixTopKMaskLogitsMultiCTA<c_type, int>(
        static_cast<c_type*>(logits.data_ptr()), static_cast<c_type*>(mask_logits.data_ptr()),
        has_top_k_arr ? static_cast<int*>(maybe_top_k_arr.value().data_ptr()) : nullptr, batch_size,
        top_k_val, vocab_size, static_cast<sampling::RadixRowState*>(row_states_buffer.data_ptr()),
        stream);
    return true;
  });

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "TopKMaskLogits failed with error code " << cudaGetErrorString(status);
}
