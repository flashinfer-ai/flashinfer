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
#include "tvm_ffi_utils.h"

namespace flashinfer::fused_moe {
using tvm::ffi::TensorView;

void B12xDirectNVFP4FusedMoe(TensorView hidden_states, TensorView topk_ids, TensorView topk_weights,
                             TensorView gemm1_weights, TensorView gemm1_scales,
                             TensorView gemm2_weights, TensorView gemm2_scales,
                             TensorView expert_map, TensorView hidden_quantized,
                             TensorView hidden_scales, TensorView intermediate_quantized,
                             TensorView intermediate_scales, TensorView output,
                             int64_t outputs_per_warp, int64_t num_threads,
                             double hidden_global_encode_scale,
                             double intermediate_global_encode_scale, int64_t run_down);
}  // namespace flashinfer::fused_moe

TVM_FFI_DLL_EXPORT_TYPED_FUNC(b12x_direct_nvfp4_fused_moe,
                              flashinfer::fused_moe::B12xDirectNVFP4FusedMoe);
