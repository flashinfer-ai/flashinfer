/*
 * Copyright (c) 2023-2025 by FlashInfer team.
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

// FMHAv2 JIT Binding
// This file exports the fmha_v2_run function via TVM FFI

#include <fused_multihead_attention.h>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;
using Attention_input_layout = fmha::Attention_input_layout;

// void fmha_v2_paged_run(ffi::TensorView q, ffi::TensorView k, ffi::TensorView v, ffi::TensorView
// o,
//                  ffi::TensorView workspace_buffer, size_t workspace_buffer_size_in_bytes,
//                  ffi::TensorView block_tables, int page_size, ffi::TensorView seq_lens,
//                  ffi::TensorView cum_seq_lens_q, ffi::TensorView cum_seq_lens_kv,
//                  Attention_input_layout input_layout, Data_type output_dtype, int max_q_len,
//                  int max_kv_len, int batch_size, int64_t mask_mode_code, float scale_softmax,
//                  float scale_bmm1, float scale_bmm2, int window_left, int chunked_attention_size,
//                  bool has_alibi, float softcapping_scale, Optional<ffi::TensorView>
//                  softmax_stats, Optional<ffi::TensorView> sinks);

// void fmha_v2_ragged_run(ffi::TensorView q, ffi::TensorView k, ffi::TensorView v, ffi::TensorView
// o,
//                  ffi::TensorView workspace_buffer, size_t workspace_buffer_size_in_bytes,
//                  ffi::TensorView block_tables, int page_size, ffi::TensorView seq_lens,
//                  ffi::TensorView cum_seq_lens_q, ffi::TensorView cum_seq_lens_kv,
//                  Attention_input_layout input_layout, Data_type output_dtype, int max_q_len,
//                  int max_kv_len, int batch_size, int64_t mask_mode_code, float scale_softmax,
//                  float scale_bmm1, float scale_bmm2, int window_left, int chunked_attention_size,
//                  bool has_alibi, float softcapping_scale, Optional<ffi::TensorView>
//                  softmax_stats, Optional<ffi::TensorView> sinks);

// // FMHAv2 attention operator
// TVM_FFI_DLL_EXPORT_TYPED_FUNC(paged_run, fmha_v2_paged_run);
// TVM_FFI_DLL_EXPORT_TYPED_FUNC(ragged_run, fmha_v2_ragged_run);

void fmha_v2_run(ffi::TensorView q, ffi::TensorView k, ffi::TensorView v, ffi::TensorView o,
                 ffi::TensorView workspace_buffer, size_t workspace_buffer_size_in_bytes,
                 Optional<ffi::TensorView> maybe_block_tables, int page_size,
                 ffi::TensorView seq_lens, ffi::TensorView cum_seq_lens_q,
                 ffi::TensorView cum_seq_lens_kv, int input_layout, int output_dtype, int max_q_len,
                 int max_kv_len, int batch_size, int total_q_tokens, int total_kv_tokens,
                 int64_t mask_mode_code, float scale_softmax, float scale_bmm1, float scale_bmm2,
                 int window_left, int chunked_attention_size, bool has_alibi,
                 float softcapping_scale, ffi::TensorView scale_bmm2_d,
                 Optional<ffi::TensorView> softmax_stats, Optional<ffi::TensorView> sinks);

// FMHAv2 attention operator
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, fmha_v2_run);
