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
#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace sm90_w4a16_moe {

int64_t sm90_nvfp4_w4a16_moe_decode_buffer_bytes(int64_t num_tokens);
int64_t sm90_nvfp4_w4a16_moe_decode_counter_words();
int64_t sm90_nvfp4_w4a16_moe_decode_scratch_bytes(int64_t device);
void sm90_nvfp4_w4a16_moe_decode_run(TensorView hidden_states, TensorView topk_ids,
                                     TensorView topk_weights, TensorView w13, TensorView w13_scale,
                                     TensorView w13_up_scale, TensorView w2, TensorView w2_scale,
                                     TensorView norm_weight, TensorView out, TensorView buffer,
                                     TensorView counters, TensorView scratch, int64_t num_experts,
                                     double eps, bool enable_pdl);

int64_t sm90_nvfp4_w4a16_moe_prefill_counter_words();
int64_t sm90_nvfp4_w4a16_moe_prefill_map_bytes();
int64_t sm90_nvfp4_w4a16_moe_prefill_slots(int64_t chunk_tokens);
void sm90_nvfp4_w4a16_moe_prefill_init(TensorView xt, TensorView h, TensorView maps);
void sm90_nvfp4_w4a16_moe_prefill_run(TensorView hidden_states, TensorView topk_ids,
                                      TensorView topk_weights, TensorView w13, TensorView w13_scale,
                                      TensorView w13_up_scale, TensorView w2, TensorView w2_scale,
                                      TensorView norm_weight, TensorView out, TensorView recv,
                                      TensorView xt, TensorView maps, TensorView rev,
                                      TensorView unit_counts, TensorView counters,
                                      int64_t num_experts, int64_t chunk_tokens, double eps,
                                      bool enable_pdl);

}  // namespace sm90_w4a16_moe
}  // namespace flashinfer

using namespace flashinfer::sm90_w4a16_moe;

TVM_FFI_DLL_EXPORT_TYPED_FUNC(decode_buffer_bytes, sm90_nvfp4_w4a16_moe_decode_buffer_bytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(decode_counter_words, sm90_nvfp4_w4a16_moe_decode_counter_words);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(decode_scratch_bytes, sm90_nvfp4_w4a16_moe_decode_scratch_bytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(decode_run, sm90_nvfp4_w4a16_moe_decode_run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prefill_counter_words, sm90_nvfp4_w4a16_moe_prefill_counter_words);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prefill_map_bytes, sm90_nvfp4_w4a16_moe_prefill_map_bytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prefill_slots, sm90_nvfp4_w4a16_moe_prefill_slots);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prefill_init, sm90_nvfp4_w4a16_moe_prefill_init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prefill_run, sm90_nvfp4_w4a16_moe_prefill_run);
