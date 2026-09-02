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

#include "flashinfer/fused_moe/alphamoe_router.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer::alphamoe_router {

using tvm::ffi::TensorView;

/*!
 * \brief TVM-FFI entry point for the fused AlphaMoE gating router.
 *
 *   router_logits           : [num_tokens, num_experts]  float32 (input)
 *   topk_weights            : [num_tokens, top_k]        float32 (output)
 *   topk_ids                : [num_tokens, top_k]        int32   (output)
 *   sorted_token_ids        : [max_blocks * block_m]     int32   (output)
 *   expert_ids              : [max_blocks]               int32   (output)
 *   num_tokens_post_padded  : [1]                        int32   (output)
 *   expert_counts           : [num_experts]              int32   (output)
 *   expert_offsets          : [num_experts + 1]          int32   (output)
 *   expert_scatter_offsets  : [num_experts]              int32   (output)
 *   scratch                 : [see alphamoe_router_scratch_ints] int32 (workspace)
 *
 * (top_k, block_m, has_shared_expert) come in as scalars; every output shape
 * is derived from them and the input shape, so the launch is fixed under CUDA
 * graph capture and no host synchronization happens anywhere on this path.
 */
void AlphaMoeFusedRouter(TensorView router_logits, TensorView topk_weights,
                         TensorView topk_ids, TensorView sorted_token_ids,
                         TensorView expert_ids,
                         TensorView num_tokens_post_padded,
                         TensorView expert_counts, TensorView expert_offsets,
                         TensorView expert_scatter_offsets, TensorView scratch,
                         int64_t top_k, int64_t block_m,
                         bool has_shared_expert) {
  CHECK_INPUT_AND_TYPE(router_logits, dl_float32);
  CHECK_INPUT_AND_TYPE(topk_weights, dl_float32);
  CHECK_INPUT_AND_TYPE(topk_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(sorted_token_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(expert_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(num_tokens_post_padded, dl_int32);
  CHECK_INPUT_AND_TYPE(expert_counts, dl_int32);
  CHECK_INPUT_AND_TYPE(expert_offsets, dl_int32);
  CHECK_INPUT_AND_TYPE(expert_scatter_offsets, dl_int32);
  CHECK_INPUT_AND_TYPE(scratch, dl_int32);

  CHECK_DEVICE(topk_weights, router_logits);
  CHECK_DEVICE(topk_ids, router_logits);
  CHECK_DEVICE(sorted_token_ids, router_logits);
  CHECK_DEVICE(expert_ids, router_logits);
  CHECK_DEVICE(num_tokens_post_padded, router_logits);
  CHECK_DEVICE(expert_counts, router_logits);
  CHECK_DEVICE(expert_offsets, router_logits);
  CHECK_DEVICE(expert_scatter_offsets, router_logits);
  CHECK_DEVICE(scratch, router_logits);

  CHECK_DIM(2, router_logits);
  CHECK_DIM(2, topk_weights);
  CHECK_DIM(2, topk_ids);
  CHECK_DIM(1, sorted_token_ids);
  CHECK_DIM(1, expert_ids);
  CHECK_DIM(1, num_tokens_post_padded);
  CHECK_DIM(1, expert_counts);
  CHECK_DIM(1, expert_offsets);
  CHECK_DIM(1, expert_scatter_offsets);
  CHECK_DIM(1, scratch);

  const int64_t num_tokens = router_logits.sizes()[0];
  const int64_t num_experts = router_logits.sizes()[1];

  const auto params = flashinfer::fused_moe::make_alphamoe_router_params(
      static_cast<int>(num_tokens), static_cast<int>(num_experts),
      static_cast<int>(top_k), static_cast<int>(block_m),
      has_shared_expert ? 1 : 0);

  TVM_FFI_ICHECK(topk_weights.sizes()[0] == num_tokens &&
                 topk_weights.sizes()[1] == top_k &&
                 topk_ids.sizes()[0] == num_tokens &&
                 topk_ids.sizes()[1] == top_k)
      << "topk outputs must be [num_tokens, top_k]";
  TVM_FFI_ICHECK(expert_counts.numel() == num_experts &&
                 expert_scatter_offsets.numel() == num_experts)
      << "expert counts/scatter offsets must be [num_experts]";
  TVM_FFI_ICHECK(expert_offsets.numel() == num_experts + 1)
      << "expert_offsets must be [num_experts + 1]";
  TVM_FFI_ICHECK(num_tokens_post_padded.numel() == 1)
      << "num_tokens_post_padded must be [1]";
  TVM_FFI_ICHECK(sorted_token_ids.numel() == params.slots)
      << "sorted_token_ids must be [max_blocks * block_m] = " << params.slots;
  TVM_FFI_ICHECK(expert_ids.numel() == params.max_blocks)
      << "expert_ids must be [max_blocks] = " << params.max_blocks;
  TVM_FFI_ICHECK(scratch.numel() >=
                 flashinfer::fused_moe::alphamoe_router_scratch_ints(params))
      << "scratch too small for the generic path";

  auto stream = get_stream(router_logits.device());

  flashinfer::fused_moe::alphamoe_router_forward(
      params, static_cast<const float*>(router_logits.data_ptr()),
      static_cast<float*>(topk_weights.data_ptr()),
      static_cast<int32_t*>(topk_ids.data_ptr()),
      static_cast<int32_t*>(expert_counts.data_ptr()),
      static_cast<int32_t*>(expert_offsets.data_ptr()),
      static_cast<int32_t*>(expert_scatter_offsets.data_ptr()),
      static_cast<int32_t*>(num_tokens_post_padded.data_ptr()),
      static_cast<int32_t*>(expert_ids.data_ptr()),
      static_cast<int32_t*>(sorted_token_ids.data_ptr()),
      static_cast<int32_t*>(scratch.data_ptr()), stream);
}

}  // namespace flashinfer::alphamoe_router
