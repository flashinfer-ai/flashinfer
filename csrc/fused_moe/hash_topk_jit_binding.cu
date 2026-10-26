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

namespace flashinfer::dsv4_hash_topk {
using tvm::ffi::TensorView;
void HashTopK(TensorView router_logits, TensorView input_id, TensorView tid2eid,
              TensorView topk_weights, TensorView topk_ids, double routed_scaling_factor,
              bool launch_with_pdl);
}  // namespace flashinfer::dsv4_hash_topk

TVM_FFI_DLL_EXPORT_TYPED_FUNC(hash_topk, flashinfer::dsv4_hash_topk::HashTopK);
