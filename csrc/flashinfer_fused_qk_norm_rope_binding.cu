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
#include "tvm_ffi_utils.h"

void fused_qk_norm_rope_run(TensorView qkv_in, TensorView q_weight, TensorView k_weight,
                            TensorView q_out, TensorView k_out, TensorView v_out,
                            int64_t num_tokens, int64_t seq_len, int64_t ppf, int64_t pph,
                            int64_t ppw, int64_t num_frame_channels, int64_t num_height_channels,
                            int64_t num_width_channels, int64_t num_heads_q, int64_t num_heads_k,
                            int64_t num_heads_v, int64_t head_dim, double eps, double base,
                            bool interleave, double factor, double low, double high,
                            double attention_factor, bool is_qk_norm, bool output_fp8,
                            double output_quant_scale, double v_quant_scale);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, fused_qk_norm_rope_run);
