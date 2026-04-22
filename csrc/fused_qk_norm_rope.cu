/*
 * Copyright (c) 2025 by FlashInfer team.
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
#include <flashinfer/fused_qk_norm_rope.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

void fused_qk_norm_rope_run(TensorView qkv_in, TensorView q_weight, TensorView k_weight,
                            TensorView q_out, TensorView k_out, TensorView v_out,
                            int64_t num_tokens, int64_t seq_len, int64_t ppf, int64_t pph,
                            int64_t ppw, int64_t num_frame_channels, int64_t num_height_channels,
                            int64_t num_width_channels, int64_t num_heads_q, int64_t num_heads_k,
                            int64_t num_heads_v, int64_t head_dim, double eps, double base,
                            bool interleave, double factor, double low, double high,
                            double attention_factor, bool is_qk_norm, bool output_fp8,
                            double output_quant_scale, double v_quant_scale) {
  CHECK_INPUT(qkv_in);
  CHECK_INPUT(q_weight);
  CHECK_INPUT(k_weight);
  CHECK_CUDA(q_out);
  CHECK_CONTIGUOUS(q_out);
  CHECK_CUDA(k_out);
  CHECK_CONTIGUOUS(k_out);
  CHECK_CUDA(v_out);
  CHECK_CONTIGUOUS(v_out);

  CHECK_INPUT_TYPE(qkv_in, dl_bfloat16);
  CHECK_INPUT_TYPE(q_weight, dl_bfloat16);
  CHECK_INPUT_TYPE(k_weight, dl_bfloat16);

  ffi::CUDADeviceGuard device_guard(qkv_in.device().device_id);
  const cudaStream_t stream = get_stream(qkv_in.device());

  int num_sms;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, qkv_in.device().device_id);

  launchFusedQKNormRope(
      qkv_in.data_ptr(), q_out.data_ptr(), k_out.data_ptr(), v_out.data_ptr(),
      static_cast<int>(num_tokens), static_cast<int>(seq_len), static_cast<int>(ppf),
      static_cast<int>(pph), static_cast<int>(ppw), static_cast<int>(num_frame_channels),
      static_cast<int>(num_height_channels), static_cast<int>(num_width_channels),
      static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),
      static_cast<int>(num_heads_v), static_cast<int>(head_dim), static_cast<float>(eps),
      q_weight.data_ptr(), k_weight.data_ptr(), static_cast<float>(base), interleave,
      static_cast<float>(factor), static_cast<float>(low), static_cast<float>(high),
      static_cast<float>(attention_factor), stream, is_qk_norm, num_sms, output_fp8,
      static_cast<float>(output_quant_scale), static_cast<float>(v_quant_scale));
}
