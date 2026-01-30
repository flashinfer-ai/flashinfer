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
#include <flashinfer/sampling.cuh>

#include "sampling_utils.h"
#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Optional;

// Helper function to validate seed/offset tensors for sampling operations
inline void validate_seed_offset_tensors(const Optional<TensorView>& maybe_seed_arr,
                                         const Optional<TensorView>& maybe_offset_arr,
                                         const TensorView& reference_tensor) {
  if (maybe_seed_arr.has_value()) {
    CHECK_INPUT(maybe_seed_arr.value());
    CHECK_DIM(1, maybe_seed_arr.value());
    TVM_FFI_ICHECK(maybe_seed_arr.value().dtype() == dl_int64 ||
                   maybe_seed_arr.value().dtype() == dl_uint64)
        << "seed tensor must be int64 or uint64";
    CHECK_DEVICE(maybe_seed_arr.value(), reference_tensor);
  }
  if (maybe_offset_arr.has_value()) {
    CHECK_INPUT(maybe_offset_arr.value());
    CHECK_DIM(1, maybe_offset_arr.value());
    TVM_FFI_ICHECK(maybe_offset_arr.value().dtype() == dl_int64 ||
                   maybe_offset_arr.value().dtype() == dl_uint64)
        << "offset tensor must be int64 or uint64";
    CHECK_DEVICE(maybe_offset_arr.value(), reference_tensor);
  }
}

void softmax(TensorView workspace_buffer, TensorView logits, TensorView output,
             Optional<TensorView> maybe_temperature_arr, double temperature_val, bool enable_pdl) {
  CHECK_INPUT(workspace_buffer);
  CHECK_INPUT(logits);
  CHECK_INPUT(output);
  CHECK_DIM(2, logits);  // logits: (batch_size, vocab_size)
  unsigned int batch_size = logits.size(0);
  unsigned int vocab_size = logits.size(1);

  bool has_temperature_arr = maybe_temperature_arr.has_value();

  ffi::CUDADeviceGuard device_guard(logits.device().device_id);
  auto stream = get_stream(logits.device());
  cudaError_t status = sampling::OnlineSoftmax<float>(
      static_cast<float*>(logits.data_ptr()), static_cast<float*>(output.data_ptr()), batch_size,
      vocab_size,
      has_temperature_arr ? static_cast<float*>(maybe_temperature_arr.value().data_ptr()) : nullptr,
      temperature_val, workspace_buffer.data_ptr(),
      get_element_size(workspace_buffer) * workspace_buffer.size(0), enable_pdl, stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "OnlineSoftmax failed with error code " << cudaGetErrorString(status);
}

void sampling_from_logits(TensorView logits, TensorView output, Optional<TensorView> maybe_indices,
                          bool deterministic, Optional<TensorView> maybe_seed_arr,
                          uint64_t seed_val, Optional<TensorView> maybe_offset_arr,
                          uint64_t offset_val) {
  CHECK_INPUT(logits);
  CHECK_DIM(2, logits);  // logits: (batch_size, vocab_size)
  CHECK_MAYBE_INPUT_TYPES(maybe_indices, dl_int32, dl_int64);
  CHECK_MAYBE_SAME_DTYPE(maybe_indices, output);
  validate_seed_offset_tensors(maybe_seed_arr, maybe_offset_arr, logits);

  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = logits.size(1);

  ffi::CUDADeviceGuard device_guard(logits.device().device_id);
  auto stream = get_stream(logits.device());

  DISPATCH_DLPACK_IDTYPE_TO_CTYPE(output.dtype(), IdType, [&] {
    cudaError_t status = sampling::SamplingFromLogits<float, IdType>(
        static_cast<float*>(logits.data_ptr()), static_cast<IdType*>(output.data_ptr()),
        maybe_indices.has_value() ? static_cast<IdType*>(maybe_indices.value().data_ptr())
                                  : nullptr,
        batch_size, vocab_size, deterministic,
        maybe_seed_arr.has_value() ? static_cast<uint64_t*>(maybe_seed_arr.value().data_ptr())
                                   : nullptr,
        seed_val,
        maybe_offset_arr.has_value() ? static_cast<uint64_t*>(maybe_offset_arr.value().data_ptr())
                                     : nullptr,
        offset_val, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "SamplingFromLogits failed with error code " << cudaGetErrorString(status);
    return true;
  });
}

void sampling_from_probs(TensorView probs, TensorView output, Optional<TensorView> maybe_indices,
                         bool deterministic, Optional<TensorView> maybe_seed_arr, uint64_t seed_val,
                         Optional<TensorView> maybe_offset_arr, uint64_t offset_val) {
  CHECK_INPUT(probs);
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  CHECK_MAYBE_INPUT_TYPES(maybe_indices, dl_int32, dl_int64);
  CHECK_MAYBE_SAME_DTYPE(maybe_indices, output);
  validate_seed_offset_tensors(maybe_seed_arr, maybe_offset_arr, probs);

  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = probs.size(1);

  ffi::CUDADeviceGuard device_guard(probs.device().device_id);
  auto stream = get_stream(probs.device());

  DISPATCH_DLPACK_IDTYPE_TO_CTYPE(output.dtype(), IdType, [&] {
    cudaError_t status = sampling::SamplingFromProb<float, IdType>(
        static_cast<float*>(probs.data_ptr()), static_cast<IdType*>(output.data_ptr()),
        maybe_indices.has_value() ? static_cast<IdType*>(maybe_indices.value().data_ptr())
                                  : nullptr,
        batch_size, vocab_size, deterministic,
        maybe_seed_arr.has_value() ? static_cast<uint64_t*>(maybe_seed_arr.value().data_ptr())
                                   : nullptr,
        seed_val,
        maybe_offset_arr.has_value() ? static_cast<uint64_t*>(maybe_offset_arr.value().data_ptr())
                                     : nullptr,
        offset_val, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "SamplingFromProbs failed with error code " << cudaGetErrorString(status);
    return true;
  });
}

void top_p_sampling_from_probs(TensorView probs, TensorView output,
                               Optional<TensorView> maybe_indices,
                               Optional<TensorView> maybe_top_p_arr, double top_p_val,
                               bool deterministic, Optional<TensorView> maybe_seed_arr,
                               uint64_t seed_val, Optional<TensorView> maybe_offset_arr,
                               uint64_t offset_val) {
  CHECK_INPUT(probs);
  CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  CHECK_MAYBE_INPUT_TYPES(maybe_indices, dl_int32, dl_int64);
  CHECK_MAYBE_SAME_DTYPE(maybe_indices, output);
  validate_seed_offset_tensors(maybe_seed_arr, maybe_offset_arr, probs);

  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = probs.size(1);
  check_tensor_param(maybe_top_p_arr, probs);
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  ffi::CUDADeviceGuard device_guard(probs.device().device_id);
  auto stream = get_stream(probs.device());

  DISPATCH_DLPACK_IDTYPE_TO_CTYPE(output.dtype(), IdType, [&] {
    cudaError_t status = sampling::TopPSamplingFromProb<float, IdType>(
        static_cast<float*>(probs.data_ptr()), static_cast<IdType*>(output.data_ptr()),
        maybe_indices.has_value() ? static_cast<IdType*>(maybe_indices.value().data_ptr())
                                  : nullptr,
        has_top_p_arr ? static_cast<float*>(maybe_top_p_arr.value().data_ptr()) : nullptr,
        batch_size, top_p_val, vocab_size, deterministic,
        maybe_seed_arr.has_value() ? static_cast<uint64_t*>(maybe_seed_arr.value().data_ptr())
                                   : nullptr,
        seed_val,
        maybe_offset_arr.has_value() ? static_cast<uint64_t*>(maybe_offset_arr.value().data_ptr())
                                     : nullptr,
        offset_val, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "TopPSamplingFromProbs failed with error code " << cudaGetErrorString(status);
    return true;
  });
}

void top_k_sampling_from_probs(TensorView probs, TensorView output,
                               Optional<TensorView> maybe_indices,
                               Optional<TensorView> maybe_top_k_arr, int64_t top_k_val,
                               bool deterministic, Optional<TensorView> maybe_seed_arr,
                               uint64_t seed_val, Optional<TensorView> maybe_offset_arr,
                               uint64_t offset_val) {
  CHECK_INPUT(probs);
  CHECK_INPUT(output);
  CHECK_DEVICE(output, probs);
  CHECK_DIM(2, probs);   // probs: (batch_size, vocab_size)
  CHECK_DIM(1, output);  // output: (batch_size)
  CHECK_MAYBE_INPUT_TYPES(maybe_indices, dl_int32, dl_int64);
  CHECK_MAYBE_SAME_DTYPE(maybe_indices, output);
  validate_seed_offset_tensors(maybe_seed_arr, maybe_offset_arr, probs);

  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = probs.size(1);
  check_tensor_param(maybe_top_k_arr, probs);
  bool has_top_k_arr = maybe_top_k_arr.has_value();

  ffi::CUDADeviceGuard device_guard(probs.device().device_id);
  auto stream = get_stream(probs.device());

  DISPATCH_DLPACK_IDTYPE_TO_CTYPE(output.dtype(), IdType, [&] {
    cudaError_t status = sampling::TopKSamplingFromProb<float, IdType>(
        static_cast<float*>(probs.data_ptr()), static_cast<IdType*>(output.data_ptr()),
        maybe_indices.has_value() ? static_cast<IdType*>(maybe_indices.value().data_ptr())
                                  : nullptr,
        has_top_k_arr ? static_cast<float*>(maybe_top_k_arr.value().data_ptr()) : nullptr,
        batch_size, top_k_val, vocab_size, deterministic,
        maybe_seed_arr.has_value() ? static_cast<uint64_t*>(maybe_seed_arr.value().data_ptr())
                                   : nullptr,
        seed_val,
        maybe_offset_arr.has_value() ? static_cast<uint64_t*>(maybe_offset_arr.value().data_ptr())
                                     : nullptr,
        offset_val, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "TopKSamplingFromProbs failed with error code " << cudaGetErrorString(status);
    return true;
  });
}

void min_p_sampling_from_probs(TensorView probs, TensorView output,
                               Optional<TensorView> maybe_indices,
                               Optional<TensorView> maybe_min_p_arr, double min_p_val,
                               bool deterministic, Optional<TensorView> maybe_seed_arr,
                               uint64_t seed_val, Optional<TensorView> maybe_offset_arr,
                               uint64_t offset_val) {
  CHECK_INPUT(probs);
  CHECK_INPUT(output);
  CHECK_DEVICE(output, probs);
  CHECK_DIM(2, probs);   // probs: (batch_size, vocab_size)
  CHECK_DIM(1, output);  // output: (batch_size)
  CHECK_MAYBE_INPUT_TYPES(maybe_indices, dl_int32, dl_int64);
  CHECK_MAYBE_SAME_DTYPE(maybe_indices, output);
  validate_seed_offset_tensors(maybe_seed_arr, maybe_offset_arr, probs);

  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = probs.size(1);
  check_tensor_param(maybe_min_p_arr, probs);
  bool has_min_p_arr = maybe_min_p_arr.has_value();

  ffi::CUDADeviceGuard device_guard(probs.device().device_id);
  auto stream = get_stream(probs.device());

  DISPATCH_DLPACK_IDTYPE_TO_CTYPE(output.dtype(), IdType, [&] {
    cudaError_t status = sampling::MinPSamplingFromProb<float, IdType>(
        static_cast<float*>(probs.data_ptr()),
        has_min_p_arr ? static_cast<float*>(maybe_min_p_arr.value().data_ptr()) : nullptr,
        static_cast<IdType*>(output.data_ptr()),
        maybe_indices.has_value() ? static_cast<IdType*>(maybe_indices.value().data_ptr())
                                  : nullptr,
        batch_size, min_p_val, vocab_size, deterministic,
        maybe_seed_arr.has_value() ? static_cast<uint64_t*>(maybe_seed_arr.value().data_ptr())
                                   : nullptr,
        seed_val,
        maybe_offset_arr.has_value() ? static_cast<uint64_t*>(maybe_offset_arr.value().data_ptr())
                                     : nullptr,
        offset_val, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "MinPSamplingFromProb failed with error code " << cudaGetErrorString(status);
    return true;
  });
}

void top_k_top_p_sampling_from_probs(TensorView probs, TensorView output,
                                     Optional<TensorView> maybe_indices,
                                     Optional<TensorView> maybe_top_k_arr, double top_k_val,
                                     Optional<TensorView> maybe_top_p_arr, double top_p_val,
                                     bool deterministic, Optional<TensorView> maybe_seed_arr,
                                     uint64_t seed_val, Optional<TensorView> maybe_offset_arr,
                                     uint64_t offset_val) {
  CHECK_INPUT(probs);
  CHECK_INPUT(output);
  CHECK_DEVICE(output, probs);
  CHECK_DIM(2, probs);   // probs: (batch_size, vocab_size)
  CHECK_DIM(1, output);  // output: (batch_size)
  CHECK_MAYBE_INPUT_TYPES(maybe_indices, dl_int32, dl_int64);
  CHECK_MAYBE_SAME_DTYPE(maybe_indices, output);
  validate_seed_offset_tensors(maybe_seed_arr, maybe_offset_arr, probs);

  unsigned int batch_size = output.size(0);
  unsigned int vocab_size = probs.size(1);
  check_tensor_param(maybe_top_k_arr, probs);
  check_tensor_param(maybe_top_p_arr, probs);
  bool has_top_k_arr = maybe_top_k_arr.has_value();
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  ffi::CUDADeviceGuard device_guard(probs.device().device_id);
  auto stream = get_stream(probs.device());

  DISPATCH_DLPACK_IDTYPE_TO_CTYPE(output.dtype(), IdType, [&] {
    cudaError_t status = sampling::TopKTopPSamplingFromProb<float, IdType>(
        static_cast<float*>(probs.data_ptr()),
        has_top_k_arr ? static_cast<IdType*>(maybe_top_k_arr.value().data_ptr()) : nullptr,
        has_top_p_arr ? static_cast<float*>(maybe_top_p_arr.value().data_ptr()) : nullptr,
        static_cast<IdType*>(output.data_ptr()),
        maybe_indices.has_value() ? static_cast<IdType*>(maybe_indices.value().data_ptr())
                                  : nullptr,
        batch_size, top_k_val, top_p_val, vocab_size, deterministic,
        maybe_seed_arr.has_value() ? static_cast<uint64_t*>(maybe_seed_arr.value().data_ptr())
                                   : nullptr,
        seed_val,
        maybe_offset_arr.has_value() ? static_cast<uint64_t*>(maybe_offset_arr.value().data_ptr())
                                     : nullptr,
        offset_val, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "TopKTopPSamplingFromProbs failed with error code " << cudaGetErrorString(status);
    return true;
  });
}

void chain_speculative_sampling(TensorView draft_probs, TensorView draft_token_ids,
                                TensorView target_probs, TensorView output_token_ids,
                                TensorView output_accepted_token_num,
                                TensorView output_emitted_draft_token_num, bool deterministic,
                                Optional<TensorView> maybe_seed_arr, uint64_t seed_val,
                                Optional<TensorView> maybe_offset_arr, uint64_t offset_val) {
  CHECK_INPUT(draft_probs);
  CHECK_INPUT(draft_token_ids);
  CHECK_INPUT(target_probs);
  CHECK_DEVICE(draft_token_ids, draft_probs);
  CHECK_DEVICE(target_probs, draft_probs);
  validate_seed_offset_tensors(maybe_seed_arr, maybe_offset_arr, draft_probs);

  CHECK_DIM(3, draft_probs);      // draft_probs: (batch_size, num_speculate_tokens, vocab_size)
  CHECK_DIM(2, draft_token_ids);  // draft_token_ids: (batch_size, num_speculate_tokens)
  CHECK_DIM(3, target_probs);  // target_probs: (batch_size, num_speculate_tokens + 1, vocab_size)
  unsigned int batch_size = draft_probs.size(0);
  unsigned int num_speculate_tokens = draft_probs.size(1);
  unsigned int vocab_size = draft_probs.size(2);
  TVM_FFI_ICHECK_EQ(batch_size, draft_token_ids.size(0));
  TVM_FFI_ICHECK_EQ(batch_size, target_probs.size(0));
  TVM_FFI_ICHECK_EQ(num_speculate_tokens + 1, target_probs.size(1));
  TVM_FFI_ICHECK_EQ(vocab_size, target_probs.size(2));
  TVM_FFI_ICHECK_EQ(batch_size, output_accepted_token_num.size(0));
  TVM_FFI_ICHECK_EQ(batch_size, output_emitted_draft_token_num.size(0));

  ffi::CUDADeviceGuard device_guard(draft_probs.device().device_id);
  auto stream = get_stream(draft_probs.device());
  cudaError_t status = sampling::ChainSpeculativeSampling<float, int>(
      static_cast<float*>(draft_probs.data_ptr()), static_cast<int*>(draft_token_ids.data_ptr()),
      static_cast<float*>(target_probs.data_ptr()), static_cast<int*>(output_token_ids.data_ptr()),
      static_cast<int*>(output_accepted_token_num.data_ptr()),
      static_cast<int*>(output_emitted_draft_token_num.data_ptr()), batch_size,
      num_speculate_tokens, vocab_size, deterministic,
      maybe_seed_arr.has_value() ? static_cast<uint64_t*>(maybe_seed_arr.value().data_ptr())
                                 : nullptr,
      seed_val,
      maybe_offset_arr.has_value() ? static_cast<uint64_t*>(maybe_offset_arr.value().data_ptr())
                                   : nullptr,
      offset_val, stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "ChainSpeculativeSampling failed with error code " << cudaGetErrorString(status);
}
