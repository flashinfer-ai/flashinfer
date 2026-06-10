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
#include <cmath>
#include <cstdint>

#include "flashinfer/fused_moe/hash_topk.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer::dsv4_hash_topk {

using tvm::ffi::TensorView;

/*!
 * \brief TVM-FFI entry point for DSv4 hash-based MoE routing.
 *
 *   router_logits : [num_tokens, num_routed_experts] float32
 *   input_id      : [num_tokens]                     int64
 *   tid2eid       : [vocab, topk]                    int32   (token -> expert table)
 *   topk_weights  : [num_tokens, topk_fused]         float32 (output)
 *   topk_ids      : [num_tokens, topk_fused]         int32   (output)
 *
 * topk_fused = topk + num_shared_experts is inferred from the output shape.
 */
void HashTopK(TensorView router_logits, TensorView input_id, TensorView tid2eid,
              TensorView topk_weights, TensorView topk_ids, double routed_scaling_factor,
              bool launch_with_pdl) {
  // CUDA + contiguous + dtype.
  CHECK_INPUT_AND_TYPE(router_logits, dl_float32);
  CHECK_INPUT_AND_TYPE(input_id, dl_int64);
  CHECK_INPUT_AND_TYPE(tid2eid, dl_int32);
  CHECK_INPUT_AND_TYPE(topk_weights, dl_float32);
  CHECK_INPUT_AND_TYPE(topk_ids, dl_int32);

  // Same CUDA device for all tensors.
  CHECK_DEVICE(input_id, router_logits);
  CHECK_DEVICE(tid2eid, router_logits);
  CHECK_DEVICE(topk_weights, router_logits);
  CHECK_DEVICE(topk_ids, router_logits);

  // Ranks.
  CHECK_DIM(2, router_logits);
  CHECK_DIM(1, input_id);
  CHECK_DIM(2, tid2eid);
  CHECK_DIM(2, topk_weights);
  CHECK_DIM(2, topk_ids);

  const int64_t num_tokens = router_logits.sizes()[0];
  const int64_t num_routed_experts = router_logits.sizes()[1];
  const int64_t topk = tid2eid.sizes()[1];
  const int64_t topk_fused = topk_ids.sizes()[1];
  const int64_t num_shared_experts = topk_fused - topk;

  TVM_FFI_ICHECK(num_routed_experts >= 1) << "num_routed_experts must be >= 1";
  TVM_FFI_ICHECK(topk >= 1) << "topk must be >= 1";
  TVM_FFI_ICHECK(input_id.numel() == num_tokens) << "input_id length must equal num_tokens";
  TVM_FFI_ICHECK(topk_weights.sizes()[0] == num_tokens && topk_ids.sizes()[0] == num_tokens)
      << "output rows must equal num_tokens";
  TVM_FFI_ICHECK(topk_weights.sizes()[1] == topk_fused)
      << "topk_weights cols must equal topk_fused";
  TVM_FFI_ICHECK(num_shared_experts == 0 || num_shared_experts == 1)
      << "num_shared_experts (topk_fused - topk) must be 0 or 1";
  TVM_FFI_ICHECK(topk_fused <= flashinfer::fused_moe::kHashTopKWarpThreads)
      << "topk_fused must be <= warp size (32)";
  if (num_shared_experts > 0) {
    TVM_FFI_ICHECK(std::isfinite(routed_scaling_factor) && routed_scaling_factor > 0.0)
        << "routed_scaling_factor must be positive and finite when a shared expert is fused";
  }

  if (num_tokens == 0) {
    return;
  }

  auto stream = get_stream(router_logits.device());

  const cudaError_t status = flashinfer::fused_moe::LaunchHashTopK(
      static_cast<const float*>(router_logits.data_ptr()),
      static_cast<const int64_t*>(input_id.data_ptr()),
      static_cast<const int32_t*>(tid2eid.data_ptr()), static_cast<int32_t*>(topk_ids.data_ptr()),
      static_cast<float*>(topk_weights.data_ptr()), static_cast<uint32_t>(num_tokens),
      static_cast<uint32_t>(topk), static_cast<uint32_t>(num_routed_experts),
      static_cast<uint32_t>(num_shared_experts), static_cast<float>(routed_scaling_factor),
      launch_with_pdl, stream);

  TVM_FFI_ICHECK(status == cudaSuccess)
      << "hash_topk kernel launch failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer::dsv4_hash_topk
