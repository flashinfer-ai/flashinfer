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
#include <flashinfer/sampling_hy3.cuh>
#include <limits>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Optional;

namespace sampler_hy3 = flashinfer::sampling::hy3;

// HY3 fused logits processing and Gumbel-max sampling optimized for SM100/B200.
// The CUDA implementation is derived from Tencent/HPC-Ops and retains its MIT
// license in include/flashinfer/sampling_hy3.cuh. This wrapper accepts
// caller-owned workspace so the hot path performs no cudaMalloc/free.
// seed must be greater than zero when maybe_gumbel_noise is absent.
void fused_sampling_from_logits_hy3(
    TensorView workspace_buffer, TensorView logits, TensorView output,
    Optional<TensorView> maybe_penalty_mask, Optional<TensorView> maybe_slot_id,
    Optional<TensorView> maybe_repetition_penalty, double repetition_penalty_val,
    Optional<TensorView> maybe_temperature, double temperature_val, int64_t softmax_policy,
    Optional<TensorView> maybe_top_k, int64_t top_k_val, Optional<TensorView> maybe_top_p,
    double top_p_val, int64_t max_top_k, Optional<TensorView> maybe_gumbel_noise,
    Optional<TensorView> maybe_draft_token_ids, int64_t sm_count, uint64_t seed, uint64_t offset,
    bool temperature_only) {
  CHECK_INPUT_AND_TYPE(workspace_buffer, dl_uint8);
  CHECK_DIM(1, workspace_buffer);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(workspace_buffer.data_ptr()) % alignof(float), 0U)
      << "workspace_buffer address must be aligned to four bytes";
  CHECK_CUDA(logits);
  CHECK_DIM(2, logits);
  TVM_FFI_ICHECK_EQ(logits.stride(1), 1) << "logits inner dimension must be contiguous";
  TVM_FFI_ICHECK(logits.dtype() == dl_float32 || logits.dtype() == dl_bfloat16)
      << "logits dtype must be float32 or bfloat16";
  CHECK_INPUT_AND_TYPE(output, dl_int32);
  CHECK_DIM(2, output);
  CHECK_DEVICE(workspace_buffer, logits);
  CHECK_DEVICE(output, logits);

  const int64_t batch_size_i64 = logits.size(0);
  const int64_t vocab_size_i64 = logits.size(1);
  TVM_FFI_ICHECK_GT(batch_size_i64, 0) << "batch_size must be positive";
  TVM_FFI_ICHECK_EQ(vocab_size_i64, sampler_hy3::kVocabSize)
      << "HY3 fused sampler currently supports vocab_size=" << sampler_hy3::kVocabSize;
  TVM_FFI_ICHECK_EQ(output.size(0), batch_size_i64);
  TVM_FFI_ICHECK_EQ(output.size(1), 1);
  TVM_FFI_ICHECK_LE(batch_size_i64, std::numeric_limits<int>::max());
  TVM_FFI_ICHECK_LE(logits.stride(0), std::numeric_limits<int>::max());
  TVM_FFI_ICHECK_GE(logits.stride(0), vocab_size_i64);
  TVM_FFI_ICHECK_GT(sm_count, 0);
  TVM_FFI_ICHECK_LE(sm_count, std::numeric_limits<int>::max());

  auto check_optional_vector = [&](const Optional<TensorView>& maybe_tensor, DLDataType dtype,
                                   const char* name) {
    if (!maybe_tensor.has_value()) return;
    const TensorView& tensor = maybe_tensor.value();
    CHECK_CUDA(tensor);
    CHECK_CONTIGUOUS(tensor);
    CHECK_DEVICE(tensor, logits);
    TVM_FFI_ICHECK_EQ(tensor.dtype(), dtype) << name << " has an invalid dtype";
    TVM_FFI_ICHECK_EQ(tensor.ndim(), 1) << name << " must have rank 1";
    TVM_FFI_ICHECK_EQ(tensor.size(0), batch_size_i64) << name << " must have shape [batch_size]";
  };

  check_optional_vector(maybe_repetition_penalty, dl_float32, "repetition_penalty");
  check_optional_vector(maybe_temperature, dl_float32, "temperature");
  check_optional_vector(maybe_top_p, dl_float32, "top_p");
  check_optional_vector(maybe_draft_token_ids, dl_int64, "draft_token_ids");
  if (maybe_top_k.has_value()) {
    const TensorView& top_k = maybe_top_k.value();
    CHECK_CUDA(top_k);
    CHECK_CONTIGUOUS(top_k);
    CHECK_DEVICE(top_k, logits);
    CHECK_DIM(1, top_k);
    TVM_FFI_ICHECK_EQ(top_k.size(0), batch_size_i64);
    TVM_FFI_ICHECK(top_k.dtype() == dl_int32 || top_k.dtype() == dl_int64)
        << "top_k dtype must be int32 or int64";
  }
  if (maybe_gumbel_noise.has_value()) {
    const TensorView& noise = maybe_gumbel_noise.value();
    CHECK_INPUT_AND_TYPE(noise, dl_float32);
    CHECK_DEVICE(noise, logits);
    CHECK_DIM(2, noise);
    TVM_FFI_ICHECK_EQ(noise.size(0), batch_size_i64);
    TVM_FFI_ICHECK_EQ(noise.size(1), vocab_size_i64);
  } else {
    TVM_FFI_ICHECK_GT(seed, 0) << "seed must be > 0 without external gumbel_noise";
  }

  const int logits_dtype = logits.dtype() == dl_float32 ? 0 : 1;
  const size_t workspace_size =
      static_cast<size_t>(workspace_buffer.size(0)) * get_element_size(workspace_buffer);
  ffi::CUDADeviceGuard device_guard(logits.device().device_id);
  cudaStream_t stream = get_stream(logits.device());
  cudaError_t status = cudaSuccess;

  if (temperature_only) {
    TVM_FFI_ICHECK(!maybe_penalty_mask.has_value() && !maybe_slot_id.has_value() &&
                   !maybe_repetition_penalty.has_value() && repetition_penalty_val == 0.0 &&
                   !maybe_top_k.has_value() && top_k_val == 0 && !maybe_top_p.has_value() &&
                   top_p_val == 0.0 && softmax_policy == sampler_hy3::kSoftmaxNone)
        << "temperature-only path received an incompatible sampler feature";
    TVM_FFI_ICHECK(maybe_temperature.has_value() || temperature_val > 0.0)
        << "temperature-only path requires a positive scalar or per-row temperature";
    status = sampler_hy3::launch_temperature(
        static_cast<int32_t*>(output.data_ptr()), logits.data_ptr(), logits_dtype,
        static_cast<int>(logits.stride(0)),
        maybe_temperature.has_value()
            ? static_cast<const float*>(maybe_temperature.value().data_ptr())
            : nullptr,
        static_cast<float>(temperature_val),
        maybe_gumbel_noise.has_value()
            ? static_cast<const float*>(maybe_gumbel_noise.value().data_ptr())
            : nullptr,
        maybe_draft_token_ids.has_value()
            ? static_cast<const int64_t*>(maybe_draft_token_ids.value().data_ptr())
            : nullptr,
        static_cast<int>(batch_size_i64), static_cast<int>(vocab_size_i64),
        workspace_buffer.data_ptr(), workspace_size, static_cast<int>(sm_count), seed, offset,
        stream);
  } else {
    TVM_FFI_ICHECK(!maybe_draft_token_ids.has_value())
        << "draft_token_ids is supported only by the temperature-only path";
    TVM_FFI_ICHECK_GE(softmax_policy, sampler_hy3::kSoftmaxNone);
    TVM_FFI_ICHECK_LE(softmax_policy, sampler_hy3::kSoftmaxAfterTopK);
    TVM_FFI_ICHECK(max_top_k == 32 || max_top_k == 64) << "max_top_k must be 32 or 64";
    TVM_FFI_ICHECK_EQ(maybe_penalty_mask.has_value(), maybe_slot_id.has_value())
        << "penalty_mask and slot_id must be provided together";

    int penalty_rows = 0;
    int penalty_row_stride = 0;
    uint8_t* penalty_mask_ptr = nullptr;
    const int32_t* slot_id_ptr = nullptr;
    if (maybe_penalty_mask.has_value()) {
      const TensorView& penalty_mask = maybe_penalty_mask.value();
      CHECK_INPUT_AND_TYPE(penalty_mask, dl_uint8);
      CHECK_DEVICE(penalty_mask, logits);
      CHECK_DIM(2, penalty_mask);
      TVM_FFI_ICHECK_GE(penalty_mask.size(0), batch_size_i64);
      TVM_FFI_ICHECK_GE(penalty_mask.size(1), (vocab_size_i64 + 7) / 8);
      TVM_FFI_ICHECK_EQ(penalty_mask.stride(0) % 4, 0)
          << "penalty_mask row stride must be a multiple of four bytes";
      TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(penalty_mask.data_ptr()) % alignof(uint32_t),
                        0U)
          << "penalty_mask address must be aligned to four bytes";
      TVM_FFI_ICHECK_LE(penalty_mask.size(0), std::numeric_limits<int>::max());
      TVM_FFI_ICHECK_LE(penalty_mask.stride(0), std::numeric_limits<int>::max());
      check_optional_vector(maybe_slot_id, dl_int32, "slot_id");
      penalty_rows = static_cast<int>(penalty_mask.size(0));
      penalty_row_stride = static_cast<int>(penalty_mask.stride(0));
      penalty_mask_ptr = static_cast<uint8_t*>(penalty_mask.data_ptr());
      slot_id_ptr = static_cast<const int32_t*>(maybe_slot_id.value().data_ptr());
    }

    const bool has_repetition_penalty =
        maybe_repetition_penalty.has_value() || repetition_penalty_val > 0.0;
    const bool has_top_k = maybe_top_k.has_value() || top_k_val > 0;
    const bool has_top_p = maybe_top_p.has_value() || top_p_val > 0.0;
    TVM_FFI_ICHECK(!has_repetition_penalty || maybe_penalty_mask.has_value())
        << "repetition_penalty requires penalty_mask and slot_id";
    TVM_FFI_ICHECK(!has_top_p || has_top_k) << "top_p requires top_k";
    TVM_FFI_ICHECK(!has_top_p || softmax_policy != sampler_hy3::kSoftmaxNone)
        << "top_p requires softmax_policy != NONE";
    TVM_FFI_ICHECK(softmax_policy == sampler_hy3::kSoftmaxNone || has_top_p)
        << "softmax_policy != NONE requires top_p";

    const int top_k_element_bytes =
        !maybe_top_k.has_value() ? 0 : (maybe_top_k.value().dtype() == dl_int32 ? 4 : 8);
    status = sampler_hy3::launch_heavy(
        static_cast<int32_t*>(output.data_ptr()), logits.data_ptr(), logits_dtype, penalty_mask_ptr,
        slot_id_ptr,
        maybe_repetition_penalty.has_value()
            ? static_cast<const float*>(maybe_repetition_penalty.value().data_ptr())
            : nullptr,
        static_cast<float>(repetition_penalty_val),
        maybe_temperature.has_value()
            ? static_cast<const float*>(maybe_temperature.value().data_ptr())
            : nullptr,
        static_cast<float>(temperature_val), static_cast<int>(softmax_policy),
        maybe_top_k.has_value() ? maybe_top_k.value().data_ptr() : nullptr, top_k_element_bytes,
        static_cast<int>(top_k_val),
        maybe_top_p.has_value() ? static_cast<const float*>(maybe_top_p.value().data_ptr())
                                : nullptr,
        static_cast<float>(top_p_val),
        maybe_gumbel_noise.has_value()
            ? static_cast<const float*>(maybe_gumbel_noise.value().data_ptr())
            : nullptr,
        static_cast<int>(batch_size_i64), static_cast<int>(vocab_size_i64), penalty_rows,
        penalty_row_stride, static_cast<int>(logits.stride(0)), static_cast<int>(max_top_k),
        workspace_buffer.data_ptr(), workspace_size, static_cast<int>(sm_count), seed, offset,
        stream);
  }
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "FusedSamplingFromLogitsHY3 failed with error code " << cudaGetErrorString(status);
}
