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
#include <cstdint>

#include "flashinfer/fused_moe/hash_topk.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer::dsv4_hash_topk {

using tvm::ffi::TensorView;

// DSv4 hash-based MoE routing.
//
//   router_logits : [num_tokens, num_routed_experts] float32
//   input_id      : [num_tokens]                     int64
//   tid2eid       : [vocab, topk]                    int32   (token -> expert table)
//   topk_weights  : [num_tokens, topk_fused]         float32 (output)
//   topk_ids      : [num_tokens, topk_fused]         int32   (output)
//
// topk_fused = topk + num_shared_experts is inferred from the output shape.
void HashTopK(TensorView router_logits, TensorView input_id, TensorView tid2eid,
              TensorView topk_weights, TensorView topk_ids, double routed_scaling_factor,
              bool launch_with_pdl) {
  TVM_FFI_ICHECK(router_logits.dim() == 2) << "router_logits must be a 2D Tensor";
  TVM_FFI_ICHECK(input_id.dim() == 1) << "input_id must be a 1D Tensor";
  TVM_FFI_ICHECK(tid2eid.dim() == 2) << "tid2eid must be a 2D Tensor";
  TVM_FFI_ICHECK(topk_weights.dim() == 2) << "topk_weights must be a 2D Tensor";
  TVM_FFI_ICHECK(topk_ids.dim() == 2) << "topk_ids must be a 2D Tensor";

  const int64_t num_tokens = router_logits.sizes()[0];
  const int64_t num_routed_experts = router_logits.sizes()[1];
  const int64_t topk = tid2eid.sizes()[1];
  const int64_t topk_fused = topk_ids.sizes()[1];
  const int64_t num_shared_experts = topk_fused - topk;

  TVM_FFI_ICHECK(
      router_logits.device().device_type == kDLCUDA && input_id.device().device_type == kDLCUDA &&
      tid2eid.device().device_type == kDLCUDA && topk_weights.device().device_type == kDLCUDA &&
      topk_ids.device().device_type == kDLCUDA)
      << "all tensors must be CUDA tensors";
  TVM_FFI_ICHECK(encode_dlpack_dtype(router_logits.dtype()) == float32_code)
      << "router_logits must be float32";
  TVM_FFI_ICHECK(encode_dlpack_dtype(input_id.dtype()) == int64_code) << "input_id must be int64";
  TVM_FFI_ICHECK(encode_dlpack_dtype(tid2eid.dtype()) == int32_code) << "tid2eid must be int32";
  TVM_FFI_ICHECK(encode_dlpack_dtype(topk_weights.dtype()) == float32_code)
      << "topk_weights must be float32";
  TVM_FFI_ICHECK(encode_dlpack_dtype(topk_ids.dtype()) == int32_code) << "topk_ids must be int32";

  TVM_FFI_ICHECK(input_id.numel() == num_tokens) << "input_id length must equal num_tokens";
  TVM_FFI_ICHECK(tid2eid.sizes()[0] >= 1 && topk >= 1) << "tid2eid must be [vocab, topk]";
  TVM_FFI_ICHECK(topk_weights.sizes()[0] == num_tokens && topk_ids.sizes()[0] == num_tokens)
      << "output rows must equal num_tokens";
  TVM_FFI_ICHECK(topk_weights.sizes()[1] == topk_fused)
      << "topk_weights cols must equal topk_fused";
  TVM_FFI_ICHECK(num_shared_experts >= 0) << "topk_fused must be >= topk";
  TVM_FFI_ICHECK(topk_fused <= flashinfer::fused_moe::kHashTopKWarpThreads)
      << "topk_fused must be <= warp size (32)";

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
