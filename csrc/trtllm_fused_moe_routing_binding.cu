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
#include <algorithm>
#include <cstdint>

#include "flashinfer/trtllm/fused_moe/runner.h"
#include "tvm_ffi_utils.h"

namespace flashinfer::trtllm_gen_routing {

namespace btg = batchedGemm::trtllm::gen;
using tensorrt_llm::kernels::trtllmgen_moe::RoutingMethodType;
namespace Routing = tensorrt_llm::kernels::trtllmgen_moe::Routing;
using tvm::ffi::Optional;
using tvm::ffi::TensorView;

/*!
 * \brief Standalone TVM-FFI entry point for the trtllm-gen MoE routing stage.
 *
 * Runs the same Routing::Runner dispatcher that the fused MoE launchers invoke
 * before their GEMMs, but surfaces the routing outputs to the caller instead of
 * feeding them into a GEMM workspace. All output tensors are caller-allocated;
 * see flashinfer/fused_moe/trtllm_gen_routing.py for the expected shapes.
 *
 *   routing_logits              : [num_tokens, num_experts] float32 or bfloat16
 *   routing_bias                : [num_experts] float32 or bfloat16 (optional)
 *   topk_packed                 : [num_tokens, top_k + num_fused_shared_experts] int32 (output)
 *                                 PackedScoreIdx layout: (idx << 16) | bf16 score bits
 *   topk_weights                : [num_tokens, top_k + num_fused_shared_experts] bfloat16 (output)
 *   expert_count_histogram      : [>= max(2 * total_num_experts, 512)] int32 (scratch/output)
 *   total_num_padded_tokens     : [1] int32 (output)
 *   expanded_idx_to_permuted_idx: [num_tokens, top_k + num_fused_shared_experts] int32 (output)
 *   permuted_idx_to_token_idx   : [>= max_num_padded_tokens] int32 (output)
 *   cta_idx_xy_to_batch_idx     : [>= max_num_ctas] int32 (output)
 *   cta_idx_xy_to_mn_limit      : [>= max_num_ctas] int32 (output)
 *   num_non_exiting_ctas        : [1] int32 (output)
 *
 * The expert-weight output dtype is always bfloat16: Routing::Runner hard-codes
 * mDtypeOutput = Bfloat16 for every routing method (see
 * csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_runner.cu).
 */
void trtllm_gen_routing(TensorView routing_logits, Optional<TensorView> routing_bias,
                        TensorView topk_packed, TensorView topk_weights,
                        TensorView expert_count_histogram, TensorView total_num_padded_tokens,
                        TensorView expanded_idx_to_permuted_idx,
                        TensorView permuted_idx_to_token_idx, TensorView cta_idx_xy_to_batch_idx,
                        TensorView cta_idx_xy_to_mn_limit, TensorView num_non_exiting_ctas,
                        int64_t top_k, int64_t num_fused_shared_experts, int64_t n_group,
                        int64_t topk_group, int64_t local_expert_offset, int64_t local_num_experts,
                        double routed_scaling_factor, int64_t tile_tokens_dim,
                        int64_t routing_method_type, bool norm_topk_prob, bool enable_pdl) {
  CHECK_INPUT(routing_logits);
  CHECK_DIM(2, routing_logits);
  TVM_FFI_ICHECK(routing_logits.dtype() == dl_float32 || routing_logits.dtype() == dl_bfloat16)
      << "routing_logits must be float32 or bfloat16";

  int64_t const num_tokens = routing_logits.sizes()[0];
  int64_t const num_experts = routing_logits.sizes()[1];
  int64_t const total_experts_per_token = top_k + num_fused_shared_experts;
  int64_t const total_num_experts = num_experts + num_fused_shared_experts;

  TVM_FFI_ICHECK(num_tokens >= 1) << "num_tokens must be >= 1";
  TVM_FFI_ICHECK(top_k >= 1 && top_k <= num_experts) << "top_k must be between 1 and num_experts";
  TVM_FFI_ICHECK(num_fused_shared_experts >= 0) << "num_fused_shared_experts must be non-negative";
  TVM_FFI_ICHECK(local_num_experts >= 1 && local_num_experts <= num_experts)
      << "local_num_experts must be between 1 and num_experts";
  TVM_FFI_ICHECK(local_expert_offset >= 0 && local_expert_offset + local_num_experts <= num_experts)
      << "expert offset and count must be within valid range";
  TVM_FFI_ICHECK(tile_tokens_dim >= 1 && (tile_tokens_dim & (tile_tokens_dim - 1)) == 0)
      << "tile_tokens_dim must be a positive power of two, got " << tile_tokens_dim;
  TVM_FFI_ICHECK(routing_method_type >= 0 &&
                 routing_method_type < static_cast<int64_t>(RoutingMethodType::Unspecified))
      << "invalid routing_method_type " << routing_method_type;

  btg::Dtype const dtype_logits =
      routing_logits.dtype() == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;
  btg::Dtype dtype_bias = btg::Dtype::Bfloat16;
  void* routing_bias_ptr = nullptr;
  if (routing_bias.has_value()) {
    auto const& bias = routing_bias.value();
    CHECK_INPUT(bias);
    CHECK_DEVICE(bias, routing_logits);
    TVM_FFI_ICHECK(bias.dtype() == dl_float32 || bias.dtype() == dl_bfloat16)
        << "routing_bias must be float32 or bfloat16";
    TVM_FFI_ICHECK(bias.numel() == num_experts) << "routing_bias must have num_experts elements";
    dtype_bias = bias.dtype() == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;
    routing_bias_ptr = const_cast<void*>(bias.data_ptr());
  }

  // Output buffer validation. Sizes mirror FusedMoeLauncher::prepare_routing_common().
  int32_t const max_num_ctas = Routing::getMaxNumCtasInBatchDim(num_tokens, total_experts_per_token,
                                                                total_num_experts, tile_tokens_dim);
  int32_t const max_num_padded_tokens = Routing::getMaxPermutedPaddedCount(
      num_tokens, total_experts_per_token, total_num_experts, tile_tokens_dim);
  int64_t const histogram_size = std::max<int64_t>(total_num_experts * 2, 256 * 2);

  auto check_i32_output = [&](TensorView const& t, char const* name, int64_t min_numel) {
    CHECK_INPUT_AND_TYPE(t, dl_int32);
    CHECK_DEVICE(t, routing_logits);
    TVM_FFI_ICHECK(t.numel() >= min_numel)
        << name << " must have at least " << min_numel << " elements, got " << t.numel();
  };
  check_i32_output(topk_packed, "topk_packed", num_tokens * total_experts_per_token);
  check_i32_output(expert_count_histogram, "expert_count_histogram", histogram_size);
  check_i32_output(total_num_padded_tokens, "total_num_padded_tokens", 1);
  check_i32_output(expanded_idx_to_permuted_idx, "expanded_idx_to_permuted_idx",
                   num_tokens * total_experts_per_token);
  check_i32_output(permuted_idx_to_token_idx, "permuted_idx_to_token_idx", max_num_padded_tokens);
  check_i32_output(cta_idx_xy_to_batch_idx, "cta_idx_xy_to_batch_idx", max_num_ctas);
  check_i32_output(cta_idx_xy_to_mn_limit, "cta_idx_xy_to_mn_limit", max_num_ctas);
  check_i32_output(num_non_exiting_ctas, "num_non_exiting_ctas", 1);

  CHECK_INPUT_AND_TYPE(topk_weights, dl_bfloat16);
  CHECK_DEVICE(topk_weights, routing_logits);
  TVM_FFI_ICHECK(topk_weights.numel() >= num_tokens * total_experts_per_token)
      << "topk_weights must have at least num_tokens * (top_k + num_fused_shared_experts) "
         "elements";

  auto stream = get_stream(routing_logits.device());
  Routing::Runner routing_runner(tile_tokens_dim);

  // FromLogits mode only: expert_ids == nullptr makes the dispatcher read
  // routing_logits. dtypeElt / useRoutingScalesOnInput / useDeepSeekFp8 /
  // numTokensPerExpert are unused by the routing dispatcher (they exist for
  // signature parity with the fused launcher) — pass benign values.
  routing_runner.run(
      const_cast<void*>(routing_logits.data_ptr()), routing_bias_ptr, num_tokens, num_experts,
      top_k, num_fused_shared_experts, n_group, topk_group, local_expert_offset, local_num_experts,
      static_cast<float>(routed_scaling_factor), static_cast<int32_t*>(topk_packed.data_ptr()),
      static_cast<int32_t*>(expert_count_histogram.data_ptr()),
      static_cast<int32_t*>(total_num_padded_tokens.data_ptr()),
      static_cast<int32_t*>(expanded_idx_to_permuted_idx.data_ptr()),
      /*permutedIdxToExpandedIdx=*/nullptr,
      static_cast<int32_t*>(permuted_idx_to_token_idx.data_ptr()),
      /*expertIds=*/nullptr, topk_weights.data_ptr(),
      /*numTokensPerExpert=*/nullptr, static_cast<int32_t*>(cta_idx_xy_to_batch_idx.data_ptr()),
      static_cast<int32_t*>(cta_idx_xy_to_mn_limit.data_ptr()),
      static_cast<int32_t*>(num_non_exiting_ctas.data_ptr()),
      /*dtypeElt=*/btg::Dtype::Bfloat16, dtype_bias,
      /*useRoutingScalesOnInput=*/false, /*useDeepSeekFp8=*/false,
      static_cast<RoutingMethodType>(routing_method_type), stream, dtype_logits, norm_topk_prob,
      /*routing_replay_out=*/nullptr, enable_pdl);
}

}  // namespace flashinfer::trtllm_gen_routing

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_gen_routing,
                              flashinfer::trtllm_gen_routing::trtllm_gen_routing);
