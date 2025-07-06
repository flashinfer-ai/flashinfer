/*
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
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <flashinfer/exception.h>
#include <nvrtc.h>
#include <torch/all.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"
#include "flashinfer/trtllm/fused_moe/DevKernel.h"
#include "flashinfer/trtllm/fused_moe/RoutingKernel.h"
#include "flashinfer/trtllm/fused_moe/runner.h"

namespace tensorrt_llm {
// HACK from: cpp/tensorrt_llm/kernels/quantization.h
inline int computeFP4LinearLayoutSFSize(int totalRow, int totalColumn) {
  return totalRow * totalColumn;
}
}  // namespace tensorrt_llm

namespace flashinfer {

void trtllm_fp8_per_tensor_scale_moe_launcher(
    at::Tensor routing_logits, at::Tensor routing_bias, at::Tensor hidden_states,
    at::Tensor gemm1_weights, at::Tensor output1_scales_scalar,
    at::Tensor output1_scales_gate_scalar, at::Tensor gemm2_weights,
    at::Tensor output2_scales_scalar, at::Tensor output, at::Tensor workspace_buffer,
    int64_t num_experts, int64_t top_k, int64_t n_group, int64_t topk_group,
    int64_t intermediate_size, int64_t local_expert_offset, int64_t local_num_experts,
    double routed_scaling_factor, bool use_routing_scales_on_input, int64_t tile_tokens_dim,
    int64_t routing_method_type) {
  auto device = hidden_states.device();
  const auto stream = at::cuda::getCurrentCUDAStream(device.index());

  int32_t seq_len = hidden_states.size(0);
  int32_t hidden_size = hidden_states.size(1);

  // Setup MoE runner args
  tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::MoERunnerArgs args;

  // Convert PyTorch dtype to TensorRT-LLM dtype
  auto dtype = hidden_states.dtype();
  if (dtype == at::ScalarType::Half) {
    args.mDtypeElt = batchedGemm::trtllm::gen::Dtype::Fp16;
  } else if (dtype == at::ScalarType::BFloat16) {
    args.mDtypeElt = batchedGemm::trtllm::gen::Dtype::Bfloat16;
  } else if (dtype == at::ScalarType::Float8_e4m3fn) {
    args.mDtypeElt = batchedGemm::trtllm::gen::Dtype::E4m3;
  } else {
    TORCH_CHECK(false, "Unsupported input dtype for MoE: ", dtype);
  }

  args.routing_logits = routing_logits.data_ptr();
  args.routing_bias = routing_bias.data_ptr();
  args.hidden_states = hidden_states.data_ptr();
  args.gemm1_weights = gemm1_weights.data_ptr();
  args.output1_scales_scalar = output1_scales_scalar.data_ptr<float>();
  args.output1_scales_gate_scalar = output1_scales_gate_scalar.data_ptr<float>();
  args.output2_scales_scalar = output2_scales_scalar.data_ptr<float>();
  args.gemm2_weights = gemm2_weights.data_ptr();
  args.num_tokens = seq_len;
  args.num_experts = num_experts;
  args.hidden_size = hidden_size;
  args.top_k = top_k;
  args.n_group = n_group;
  args.topk_group = topk_group;
  args.local_expert_offset = local_expert_offset;
  args.local_num_experts = local_num_experts;
  args.routed_scaling_factor = routed_scaling_factor;
  args.intermediate_size = intermediate_size;
  args.mUseDeepSeekFp8 = false;  // Not using deepseek fp8 in this interface
  args.mUseRoutingScalesOnInput = use_routing_scales_on_input;
  args.output = output.data_ptr();
  args.output_scale = nullptr;

  // Print debug information about the arguments

  if (use_routing_scales_on_input) {
    TORCH_CHECK(routing_logits.scalar_type() == at::ScalarType::BFloat16,
                "routing_logits must be bfloat16.");
  } else {
    TORCH_CHECK(routing_logits.scalar_type() == at::ScalarType::Float,
                "routing_logits must be float.");
  }
  TORCH_CHECK(routing_logits.dim() == 2, "routing_logits must be 2D.");
  TORCH_CHECK(routing_logits.sizes()[1] == num_experts, "routing_logits has incorrect shape.");
  TORCH_CHECK(routing_bias.scalar_type() == at::ScalarType::BFloat16,
              "routing_bias must be bfloat16.");
  TORCH_CHECK(routing_bias.dim() == 1, "routing_bias must be 1D.");
  TORCH_CHECK(routing_bias.sizes()[0] == num_experts, "routing_bias has incorrect shape.");

  if (n_group <= 0 || topk_group <= 0) {
    TORCH_CHECK(top_k == 1, "Current routing kernel (no groups) only supports top_k=1.");
  } else {
    TORCH_CHECK(top_k <= 8, "Current routing kernel (with groups) only supports top_k<=8.");
    TORCH_CHECK(topk_group <= 4,
                "Current routing kernel (with groups) only supports topk_group<=4.");
    TORCH_CHECK(topk_group <= n_group, "n_group must not be smaller than topk_group.");
    TORCH_CHECK(num_experts % n_group == 0, "num_experts must be divisible by n_group");
    // This check ensures we have enough experts in the selected groups to handle the top_k routing
    TORCH_CHECK(top_k < (topk_group * num_experts / n_group),
                "top_k must be less than total number of experts in selected groups");
  }
  TORCH_CHECK(num_experts % 4 == 0,
              "Routing kernel expects that num_experts must be divisible by 4");
  TORCH_CHECK(num_experts > top_k, "num_experts must be greater than top_k");

  int32_t max_num_padded_tokens =
      tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::getMaxPermutedPaddedCount(
          args.num_tokens, top_k, num_experts, tile_tokens_dim);
  int32_t max_num_ctas =
      tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::getMaxNumCtasInBatchDim(
          args.num_tokens, args.top_k, args.num_experts, tile_tokens_dim);

  // Calculate workspace offsets
  size_t offset = 0;
  auto align_offset = [&offset](size_t alignment) {
    offset = (offset + alignment - 1) / alignment * alignment;
  };

  // Routing workspace tensors
  size_t num_tokens_per_expert_size = num_experts * sizeof(int32_t);
  align_offset(sizeof(int32_t));
  size_t num_tokens_per_expert_offset = offset;
  offset += num_tokens_per_expert_size;

  size_t total_num_padded_tokens_size = sizeof(int32_t);
  align_offset(sizeof(int32_t));
  size_t total_num_padded_tokens_offset = offset;
  offset += total_num_padded_tokens_size;

  size_t expanded_idx_to_permuted_idx_size = seq_len * top_k * sizeof(int32_t);
  align_offset(sizeof(int32_t));
  size_t expanded_idx_to_permuted_idx_offset = offset;
  offset += expanded_idx_to_permuted_idx_size;

  size_t permuted_idx_to_token_idx_size = max_num_padded_tokens * sizeof(int32_t);
  align_offset(sizeof(int32_t));
  size_t permuted_idx_to_token_idx_offset = offset;
  offset += permuted_idx_to_token_idx_size;

  size_t expert_weights_size = seq_len * top_k * sizeof(uint16_t);  // BFloat16
  align_offset(sizeof(uint16_t));
  size_t expert_weights_offset = offset;
  offset += expert_weights_size;

  size_t expert_indexes_size = seq_len * top_k * sizeof(int32_t);
  align_offset(sizeof(int32_t));
  size_t expert_indexes_offset = offset;
  offset += expert_indexes_size;

  size_t expert_count_histogram_size = 2 * 256 * sizeof(int32_t);
  align_offset(sizeof(int32_t));
  size_t expert_count_histogram_offset = offset;
  offset += expert_count_histogram_size;

  // CTA workspace tensors
  size_t cta_idx_xy_to_batch_idx_size = max_num_ctas * sizeof(int32_t);
  align_offset(sizeof(int32_t));
  size_t cta_idx_xy_to_batch_idx_offset = offset;
  offset += cta_idx_xy_to_batch_idx_size;

  size_t cta_idx_xy_to_mn_limit_size = max_num_ctas * sizeof(int32_t);
  align_offset(sizeof(int32_t));
  size_t cta_idx_xy_to_mn_limit_offset = offset;
  offset += cta_idx_xy_to_mn_limit_size;

  size_t num_non_exiting_ctas_size = sizeof(int32_t);
  align_offset(sizeof(int32_t));
  size_t num_non_exiting_ctas_offset = offset;
  offset += num_non_exiting_ctas_size;

  // Intermediate computation tensors
  size_t gemm1_output_size = max_num_padded_tokens * 2 * intermediate_size * sizeof(uint8_t);
  align_offset(16);  // 16 bytes alignment for TMA
  size_t gemm1_output_offset = offset;
  offset += gemm1_output_size;

  size_t gemm1_output_scale_size =
      (2 * intermediate_size / 128) * max_num_padded_tokens * sizeof(float);
  align_offset(sizeof(float));
  size_t gemm1_output_scale_offset = offset;
  offset += gemm1_output_scale_size;

  size_t activation_output_size = max_num_padded_tokens * intermediate_size * sizeof(uint8_t);
  align_offset(16);  // 16 bytes alignment for TMA
  size_t activation_output_offset = offset;
  offset += activation_output_size;

  size_t activation_output_scale_size =
      (intermediate_size / 128) * max_num_padded_tokens * sizeof(float);
  align_offset(sizeof(float));
  size_t activation_output_scale_offset = offset;
  offset += activation_output_scale_size;

  size_t gemm2_output_size =
      max_num_padded_tokens * hidden_size * sizeof(uint16_t);  // Changed to BFloat16
  align_offset(16);                                            // 16 bytes alignment for TMA
  size_t gemm2_output_offset = offset;
  offset += gemm2_output_size;

  // Create the MoE runner to get BMM workspace sizes
  tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::Runner moe_runner(
      args.mDtypeElt, args.mUseDeepSeekFp8, tile_tokens_dim);
  auto [bmm1WorkspaceSize, bmm2WorkspaceSize] = moe_runner.getWorkspaceSizeInBytes(args);

  // BMM workspaces
  align_offset(256);  // Align to 256 bytes for BMM workspaces
  size_t bmm1_workspace_offset = offset;
  offset += bmm1WorkspaceSize;

  align_offset(256);
  size_t bmm2_workspace_offset = offset;
  offset += bmm2WorkspaceSize;

  size_t total_workspace_size = offset;

  TORCH_CHECK(workspace_buffer.size(0) >= total_workspace_size,
              "Workspace buffer too small. Required: ", total_workspace_size,
              ", Available: ", workspace_buffer.size(0));

  // Get base pointer to workspace buffer
  char* workspace_ptr = static_cast<char*>(workspace_buffer.data_ptr());

  // Setup MoE workspace with allocated pointers
  tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::MoEWorkspace workspace;

  // Setup workspace pointers for routing
  workspace.total_num_padded_tokens =
      reinterpret_cast<int32_t*>(workspace_ptr + total_num_padded_tokens_offset);
  workspace.total_max_padded_tokens = max_num_padded_tokens;
  workspace.ProjUpTileN = tile_tokens_dim;
  workspace.routing_expert_indexes =
      reinterpret_cast<int32_t*>(workspace_ptr + expert_indexes_offset);
  workspace.permuted_idx_size = workspace.total_num_padded_tokens;
  workspace.expanded_idx_to_permuted_idx =
      reinterpret_cast<int32_t*>(workspace_ptr + expanded_idx_to_permuted_idx_offset);
  workspace.permuted_idx_to_token_idx =
      reinterpret_cast<int32_t*>(workspace_ptr + permuted_idx_to_token_idx_offset);
  workspace.expert_weights = workspace_ptr + expert_weights_offset;
  workspace.cta_idx_xy_to_batch_idx =
      reinterpret_cast<int32_t*>(workspace_ptr + cta_idx_xy_to_batch_idx_offset);
  workspace.cta_idx_xy_to_mn_limit =
      reinterpret_cast<int32_t*>(workspace_ptr + cta_idx_xy_to_mn_limit_offset);
  workspace.num_non_exiting_ctas =
      reinterpret_cast<int32_t*>(workspace_ptr + num_non_exiting_ctas_offset);

  // Perform routing computation using routing_logits and routing_bias
  // This computes expert selections and weights that will be used by the MoE kernel
  tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::Runner routing_runner(tile_tokens_dim);

  routing_runner.run(
      routing_logits.data_ptr(), routing_bias.data_ptr(), args.num_tokens, args.num_experts,
      args.top_k, args.n_group, args.topk_group, args.local_expert_offset, args.local_num_experts,
      args.routed_scaling_factor, workspace.routing_expert_indexes,
      reinterpret_cast<int32_t*>(workspace_ptr + expert_count_histogram_offset),
      workspace.total_num_padded_tokens, workspace.expanded_idx_to_permuted_idx,
      nullptr /*permuted_idx_to_expanded_idx*/, workspace.permuted_idx_to_token_idx,
      workspace.expert_weights,
      reinterpret_cast<int32_t*>(workspace_ptr + num_tokens_per_expert_offset),
      workspace.cta_idx_xy_to_batch_idx, workspace.cta_idx_xy_to_mn_limit,
      workspace.num_non_exiting_ctas, args.mDtypeElt, use_routing_scales_on_input,
      false /* use_deep_seek_fp8 */,
      static_cast<tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::RoutingMethodType>(
          routing_method_type),
      stream);

  // Setup remaining workspace pointers for MoE
  workspace.gemm1_output = workspace_ptr + gemm1_output_offset;
  workspace.gemm1_output_scale =
      reinterpret_cast<float*>(workspace_ptr + gemm1_output_scale_offset);
  workspace.activation_output = workspace_ptr + activation_output_offset;
  workspace.activation_output_scale =
      reinterpret_cast<float*>(workspace_ptr + activation_output_scale_offset);
  workspace.gemm2_output = workspace_ptr + gemm2_output_offset;
  workspace.gemm2_output_scale = nullptr;
  workspace.bmm1_workspace = workspace_ptr + bmm1_workspace_offset;
  workspace.bmm2_workspace = workspace_ptr + bmm2_workspace_offset;

  // Run the MoE kernel using the computed routing results
  moe_runner.run(args, workspace, device.index(), stream);
}

void trtllm_fp8_per_tensor_scale_moe(at::Tensor routing_logits, at::Tensor routing_bias,
                                     at::Tensor hidden_states, at::Tensor gemm1_weights,
                                     at::Tensor output1_scales_scalar,
                                     at::Tensor output1_scales_gate_scalar,
                                     at::Tensor gemm2_weights, at::Tensor output2_scales_scalar,
                                     at::Tensor output, at::Tensor workspace_buffer,
                                     int64_t num_experts, int64_t top_k, int64_t n_group,
                                     int64_t topk_group, int64_t intermediate_size,
                                     int64_t local_expert_offset, int64_t local_num_experts,
                                     double routed_scaling_factor, bool use_routing_scales_on_input,
                                     int64_t tile_tokens_dim, int64_t routing_method_type) {
  auto dtype = hidden_states.dtype();
  if (dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16 ||
      dtype == at::ScalarType::Float8_e4m3fn) {
    trtllm_fp8_per_tensor_scale_moe_launcher(
        routing_logits, routing_bias, hidden_states, gemm1_weights, output1_scales_scalar,
        output1_scales_gate_scalar, gemm2_weights, output2_scales_scalar, output, workspace_buffer,
        num_experts, top_k, n_group, topk_group, intermediate_size, local_expert_offset,
        local_num_experts, routed_scaling_factor, use_routing_scales_on_input, tile_tokens_dim,
        routing_method_type);
  } else {
    TORCH_CHECK(false, "Unsupported input type: ", dtype);
  }
}

void trtllm_fp8_block_scale_moe_launcher(at::Tensor routing_logits, at::Tensor routing_bias,
                                         at::Tensor hidden_states, at::Tensor hidden_states_scale,
                                         at::Tensor gemm1_weights, at::Tensor gemm1_weights_scale,
                                         at::Tensor gemm2_weights, at::Tensor gemm2_weights_scale,
                                         at::Tensor output, at::Tensor workspace_buffer,
                                         int64_t num_experts, int64_t top_k, int64_t n_group,
                                         int64_t topk_group, int64_t intermediate_size,
                                         int64_t local_expert_offset, int64_t local_num_experts,
                                         double routed_scaling_factor, int64_t tile_tokens_dim,
                                         int64_t routing_method_type) {
  auto device = hidden_states.device();
  const auto stream = at::cuda::getCurrentCUDAStream(device.index());

  int32_t seq_len = hidden_states.size(0);
  int32_t hidden_size = hidden_states.size(1);

  // Setup MoE runner args
  tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::MoERunnerArgs args;

  // Convert PyTorch dtype to TensorRT-LLM dtype
  auto dtype = hidden_states.dtype();
  if (dtype == at::ScalarType::Half) {
    args.mDtypeElt = batchedGemm::trtllm::gen::Dtype::Fp16;
  } else if (dtype == at::ScalarType::BFloat16) {
    args.mDtypeElt = batchedGemm::trtllm::gen::Dtype::Bfloat16;
  } else if (dtype == at::ScalarType::Float8_e4m3fn) {
    args.mDtypeElt = batchedGemm::trtllm::gen::Dtype::E4m3;
  } else {
    TORCH_CHECK(false, "Unsupported input dtype for MoE: ", dtype);
  }

  args.routing_logits = routing_logits.data_ptr();
  args.routing_bias = routing_bias.data_ptr();
  args.hidden_states = hidden_states.data_ptr();
  args.hidden_states_scale = hidden_states_scale.data_ptr();
  args.gemm1_weights = gemm1_weights.data_ptr();
  args.gemm1_weights_scale = gemm1_weights_scale.data_ptr();
  args.gemm2_weights = gemm2_weights.data_ptr();
  args.gemm2_weights_scale = gemm2_weights_scale.data_ptr();
  args.num_tokens = seq_len;
  args.num_experts = num_experts;
  args.hidden_size = hidden_size;
  args.top_k = top_k;
  args.n_group = n_group;
  args.topk_group = topk_group;
  args.local_expert_offset = local_expert_offset;
  args.local_num_experts = local_num_experts;
  args.routed_scaling_factor = routed_scaling_factor;
  args.intermediate_size = intermediate_size;
  args.mUseDeepSeekFp8 = true;            // hard-coded?
  args.mUseRoutingScalesOnInput = false;  // Default value for block scale
  args.output = output.data_ptr();
  args.output_scale = nullptr;

  TORCH_CHECK(routing_logits.scalar_type() == at::ScalarType::Float,
              "routing_logits must be float.");
  TORCH_CHECK(routing_logits.dim() == 2, "routing_logits must be 2D.");
  TORCH_CHECK(routing_logits.sizes()[1] == num_experts, "routing_logits has incorrect shape.");
  TORCH_CHECK(routing_bias.scalar_type() == at::ScalarType::BFloat16 ||
                  routing_bias.scalar_type() == at::ScalarType::Float,
              "routing_bias must be bfloat16 or float.");
  TORCH_CHECK(routing_bias.dim() == 1, "routing_bias must be 1D.");
  TORCH_CHECK(routing_bias.sizes()[0] == num_experts, "routing_bias has incorrect shape.");

  if (n_group <= 0 || topk_group <= 0) {
    TORCH_CHECK(top_k == 1, "Current routing kernel (no groups) only supports top_k=1.");
  } else {
    TORCH_CHECK(top_k <= 8, "Current routing kernel (with groups) only supports top_k<=8.");
    TORCH_CHECK(topk_group <= 4,
                "Current routing kernel (with groups) only supports topk_group<=4.");
    TORCH_CHECK(topk_group <= n_group, "n_group must not be smaller than topk_group.");
    TORCH_CHECK(num_experts % n_group == 0, "num_experts must be divisible by n_group");
    // This check ensures we have enough experts in the selected groups to handle the top_k routing
    TORCH_CHECK(top_k < (topk_group * num_experts / n_group),
                "top_k must be less than total number of experts in selected groups");
  }
  TORCH_CHECK(num_experts % 4 == 0,
              "Routing kernel expects that num_experts must be divisible by 4");
  TORCH_CHECK(num_experts > top_k, "num_experts must be greater than top_k");

  int32_t max_num_padded_tokens =
      tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::getMaxPermutedPaddedCount(
          args.num_tokens, top_k, num_experts, tile_tokens_dim);
  int32_t max_num_ctas =
      tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::getMaxNumCtasInBatchDim(
          args.num_tokens, args.top_k, args.num_experts, tile_tokens_dim);

  // Calculate workspace offsets
  size_t offset = 0;
  auto align_offset = [&offset](size_t alignment) {
    offset = (offset + alignment - 1) / alignment * alignment;
  };

  // Routing workspace tensors
  size_t num_tokens_per_expert_size = num_experts * sizeof(int32_t);
  align_offset(sizeof(int32_t));
  size_t num_tokens_per_expert_offset = offset;
  offset += num_tokens_per_expert_size;

  size_t total_num_padded_tokens_size = sizeof(int32_t);
  align_offset(sizeof(int32_t));
  size_t total_num_padded_tokens_offset = offset;
  offset += total_num_padded_tokens_size;

  size_t expanded_idx_to_permuted_idx_size = seq_len * top_k * sizeof(int32_t);
  align_offset(sizeof(int32_t));
  size_t expanded_idx_to_permuted_idx_offset = offset;
  offset += expanded_idx_to_permuted_idx_size;

  size_t permuted_idx_to_token_idx_size = max_num_padded_tokens * sizeof(int32_t);
  align_offset(sizeof(int32_t));
  size_t permuted_idx_to_token_idx_offset = offset;
  offset += permuted_idx_to_token_idx_size;

  size_t expert_weights_size = seq_len * top_k * sizeof(uint16_t);  // BFloat16 for block scale
  align_offset(sizeof(uint16_t));
  size_t expert_weights_offset = offset;
  offset += expert_weights_size;

  size_t expert_indexes_size = seq_len * top_k * sizeof(int32_t);
  align_offset(sizeof(int32_t));
  size_t expert_indexes_offset = offset;
  offset += expert_indexes_size;

  size_t expert_count_histogram_size = 2 * 256 * sizeof(int32_t);
  align_offset(sizeof(int32_t));
  size_t expert_count_histogram_offset = offset;
  offset += expert_count_histogram_size;

  // CTA workspace tensors
  size_t cta_idx_xy_to_batch_idx_size = max_num_ctas * sizeof(int32_t);
  align_offset(sizeof(int32_t));
  size_t cta_idx_xy_to_batch_idx_offset = offset;
  offset += cta_idx_xy_to_batch_idx_size;

  size_t cta_idx_xy_to_mn_limit_size = max_num_ctas * sizeof(int32_t);
  align_offset(sizeof(int32_t));
  size_t cta_idx_xy_to_mn_limit_offset = offset;
  offset += cta_idx_xy_to_mn_limit_size;

  size_t num_non_exiting_ctas_size = sizeof(int32_t);
  align_offset(sizeof(int32_t));
  size_t num_non_exiting_ctas_offset = offset;
  offset += num_non_exiting_ctas_size;

  // Intermediate computation tensors for block scale
  size_t gemm1_output_size =
      max_num_padded_tokens * 2 * intermediate_size * sizeof(uint8_t);  // fp8
  align_offset(16);  // 16 bytes alignment for TMA
  size_t gemm1_output_offset = offset;
  offset += gemm1_output_size;

  size_t gemm1_output_scale_size =
      (2 * intermediate_size / 128) * max_num_padded_tokens * sizeof(float);
  align_offset(sizeof(float));
  size_t gemm1_output_scale_offset = offset;
  offset += gemm1_output_scale_size;

  size_t activation_output_size =
      max_num_padded_tokens * intermediate_size * sizeof(uint8_t);  // fp8
  align_offset(16);                                                 // 16 bytes alignment for TMA
  size_t activation_output_offset = offset;
  offset += activation_output_size;

  size_t activation_output_scale_size =
      (intermediate_size / 128) * max_num_padded_tokens * sizeof(float);
  align_offset(sizeof(float));
  size_t activation_output_scale_offset = offset;
  offset += activation_output_scale_size;

  size_t gemm2_output_size = max_num_padded_tokens * hidden_size * sizeof(uint16_t);  // BFloat16
  align_offset(16);  // 16 bytes alignment for TMA
  size_t gemm2_output_offset = offset;
  offset += gemm2_output_size;

  // Create the MoE runner to get BMM workspace sizes
  tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::Runner moe_runner(
      args.mDtypeElt, args.mUseDeepSeekFp8, tile_tokens_dim);
  auto [bmm1WorkspaceSize, bmm2WorkspaceSize] = moe_runner.getWorkspaceSizeInBytes(args);

  // BMM workspaces
  align_offset(256);  // Align to 256 bytes for BMM workspaces
  size_t bmm1_workspace_offset = offset;
  offset += bmm1WorkspaceSize;

  align_offset(256);
  size_t bmm2_workspace_offset = offset;
  offset += bmm2WorkspaceSize;

  size_t total_workspace_size = offset;

  TORCH_CHECK(workspace_buffer.size(0) >= total_workspace_size,
              "Workspace buffer too small. Required: ", total_workspace_size,
              ", Available: ", workspace_buffer.size(0));

  // Get base pointer to workspace buffer
  char* workspace_ptr = static_cast<char*>(workspace_buffer.data_ptr());

  // Setup MoE workspace with allocated pointers
  tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::MoEWorkspace workspace;

  // Setup workspace pointers for routing
  workspace.total_num_padded_tokens =
      reinterpret_cast<int32_t*>(workspace_ptr + total_num_padded_tokens_offset);
  workspace.total_max_padded_tokens = max_num_padded_tokens;
  workspace.ProjUpTileN = tile_tokens_dim;
  workspace.routing_expert_indexes =
      reinterpret_cast<int32_t*>(workspace_ptr + expert_indexes_offset);
  workspace.permuted_idx_size = workspace.total_num_padded_tokens;
  workspace.expanded_idx_to_permuted_idx =
      reinterpret_cast<int32_t*>(workspace_ptr + expanded_idx_to_permuted_idx_offset);
  workspace.permuted_idx_to_token_idx =
      reinterpret_cast<int32_t*>(workspace_ptr + permuted_idx_to_token_idx_offset);
  workspace.expert_weights = workspace_ptr + expert_weights_offset;
  workspace.cta_idx_xy_to_batch_idx =
      reinterpret_cast<int32_t*>(workspace_ptr + cta_idx_xy_to_batch_idx_offset);
  workspace.cta_idx_xy_to_mn_limit =
      reinterpret_cast<int32_t*>(workspace_ptr + cta_idx_xy_to_mn_limit_offset);
  workspace.num_non_exiting_ctas =
      reinterpret_cast<int32_t*>(workspace_ptr + num_non_exiting_ctas_offset);

  // Perform routing computation using routing_logits and routing_bias
  // This computes expert selections and weights that will be used by the MoE kernel
  tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::Runner routing_runner(tile_tokens_dim);

  routing_runner.run(
      routing_logits.data_ptr(), routing_bias.data_ptr(), args.num_tokens, args.num_experts,
      args.top_k, args.n_group, args.topk_group, args.local_expert_offset, args.local_num_experts,
      args.routed_scaling_factor, workspace.routing_expert_indexes,
      reinterpret_cast<int32_t*>(workspace_ptr + expert_count_histogram_offset),
      workspace.total_num_padded_tokens, workspace.expanded_idx_to_permuted_idx,
      nullptr /*permuted_idx_to_expanded_idx*/, workspace.permuted_idx_to_token_idx,
      workspace.expert_weights,
      reinterpret_cast<int32_t*>(workspace_ptr + num_tokens_per_expert_offset),
      workspace.cta_idx_xy_to_batch_idx, workspace.cta_idx_xy_to_mn_limit,
      workspace.num_non_exiting_ctas, args.mDtypeElt, false /* use_routing_scales_on_input */,
      true /* use_deep_seek_fp8 */,
      static_cast<tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::RoutingMethodType>(
          routing_method_type),
      stream);

  // Setup remaining workspace pointers for MoE
  workspace.gemm1_output = workspace_ptr + gemm1_output_offset;
  workspace.gemm1_output_scale =
      reinterpret_cast<float*>(workspace_ptr + gemm1_output_scale_offset);
  workspace.activation_output = workspace_ptr + activation_output_offset;
  workspace.activation_output_scale =
      reinterpret_cast<float*>(workspace_ptr + activation_output_scale_offset);
  workspace.gemm2_output = workspace_ptr + gemm2_output_offset;
  workspace.gemm2_output_scale = nullptr;
  workspace.bmm1_workspace = workspace_ptr + bmm1_workspace_offset;
  workspace.bmm2_workspace = workspace_ptr + bmm2_workspace_offset;

  // Run the MoE kernel using the computed routing results
  moe_runner.run(args, workspace, device.index(), stream);
}

void trtllm_fp8_block_scale_moe(at::Tensor routing_logits, at::Tensor routing_bias,
                                at::Tensor hidden_states, at::Tensor hidden_states_scale,
                                at::Tensor gemm1_weights, at::Tensor gemm1_weights_scale,
                                at::Tensor gemm2_weights, at::Tensor gemm2_weights_scale,
                                at::Tensor output, at::Tensor workspace_buffer, int64_t num_experts,
                                int64_t top_k, int64_t n_group, int64_t topk_group,
                                int64_t intermediate_size, int64_t local_expert_offset,
                                int64_t local_num_experts, double routed_scaling_factor,
                                int64_t tile_tokens_dim, int64_t routing_method_type) {
  auto dtype = hidden_states.dtype();
  if (dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16 ||
      dtype == at::ScalarType::Float8_e4m3fn) {
    trtllm_fp8_block_scale_moe_launcher(
        routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights,
        gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, output, workspace_buffer,
        num_experts, top_k, n_group, topk_group, intermediate_size, local_expert_offset,
        local_num_experts, routed_scaling_factor, tile_tokens_dim, routing_method_type);
  } else {
    TORCH_CHECK(false, "Unsupported input type: ", dtype);
  }
}

namespace trtllm_cubin_loader {
#include <flashinfer/cubin_loader.h>
}

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("trtllm_fp8_per_tensor_scale_moe", trtllm_fp8_per_tensor_scale_moe);
  m.def("trtllm_fp8_block_scale_moe", trtllm_fp8_block_scale_moe);
}

}  // namespace flashinfer
