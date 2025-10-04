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
#include <cuda_runtime.h>
#include <flashinfer/exception.h>
#include <nvrtc.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/GemmGatedActOptions.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"
#include "flashinfer/trtllm/fused_moe/DevKernel.h"
#include "flashinfer/trtllm/fused_moe/RoutingKernel.h"
#include "flashinfer/trtllm/fused_moe/runner.h"
#include "nv_internal/tensorrt_llm/kernels/quantization.h"
#include "nv_internal/tensorrt_llm/thop/utils.h"
#include "tvm_ffi_utils.h"

namespace flashinfer {

namespace btg = batchedGemm::trtllm::gen;
using batchedGemm::gemm::MatrixLayout;
using tensorrt_llm::kernels::trtllmgen_moe::MoE::GatedActType;
using tensorrt_llm::kernels::trtllmgen_moe::Routing::RoutingMethodType;

/*

Abstraction layers:

1. TORCH_LIBRARY_FRAGMENT bindings
These are currently the same signature as the public python APIs.
We strive to make the python interface relatively stable
and the naming of parameters meaningful to the users.

2. FusedMoeLauncher
This performs checks and preparations for the execution,
organized in several stages, see FusedMoeLauncher::run().

3. MoE::Runner
Orchestrate and dispatch all the kernels executions to fulfill the requested operation.
This includes PermuteGemm1, Gemm2, activation (if not fused), and finalize.

4. TrtllmGenBatchedGemmRunner
This provides tactic selection if not determined yet at the public API (or auto-tuning)

5. BatchedGemm Runner
The low-level gemm kernel executor which is updated together with the kernels.

6. BatchedGemmInterface
Driver calls take place to carry out the gemm operations.
*/

class FusedMoeLauncher {
 protected:
  Tensor const* routing_logits{};
  Tensor const* routing_bias{};
  Tensor const* hidden_states{};
  Tensor const* gemm1_weights{};
  Tensor const* output1_scales_scalar{};
  Tensor const* output1_scales_gate_scalar{};
  Tensor const* gemm2_weights{};
  Tensor const* output2_scales_scalar{};

  int64_t tile_tokens_dim{};
  int64_t routing_method_type{};
  bool use_shuffled_weight{};
  MatrixLayout weight_layout{MatrixLayout::MajorK};

  std::tuple<int, int> device_version;
  std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs> args;
  tensorrt_llm::kernels::trtllmgen_moe::MoE::MoEWorkspace workspace;

  btg::Dtype mDtypeAct{btg::Dtype::Bfloat16};
  btg::Dtype mDtypeWeights{btg::Dtype::Bfloat16};
  GatedActType gated_act_type{GatedActType::SwiGlu};

  // Initialize common data necessary for later.
  // May throw exception from TVM_FFI_ICHECK.
  void init_common(Tensor const* routing_logits, Tensor const* routing_bias,
                   Tensor const* hidden_states, Tensor const* gemm1_weights,
                   Tensor const* gemm2_weights,
                   std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
                   int64_t tile_tokens_dim, int64_t routing_method_type, bool use_shuffled_weight,
                   int64_t weight_layout, int64_t gated_act_type);

  // Routing logits [num_tokens, num_experts]
  void check_routing_logits_shape() const {
    TVM_FFI_ICHECK_EQ(routing_logits->ndim, 2) << "routing_logits must be 2D.";
    TVM_FFI_ICHECK_EQ(routing_logits->shape[0], hidden_states->shape[0])
        << "routing_logits and hidden_states must have the same number of tokens.";
    TVM_FFI_ICHECK_EQ(routing_logits->shape[1], args->num_experts)
        << "routing_logits dim1 must match num_experts.";
  }

  // Routing bias [num_experts]
  void check_routing_bias_shape() const {
    if (routing_bias != nullptr) {
      TVM_FFI_ICHECK_EQ(routing_bias->ndim, 1) << "routing_bias must be 1D.";
      TVM_FFI_ICHECK_EQ(routing_bias->shape[0], args->num_experts)
          << "routing_bias has incorrect shape.";
    }
  }

  // Hidden states [num_tokens, hidden_size]
  void check_hidden_states_shape() const {
    TVM_FFI_ICHECK_EQ(hidden_states->ndim, 2) << "hidden_states must be 2D.";
    TVM_FFI_ICHECK_EQ(hidden_states->shape[1], args->intermediate_size)
        << "hidden_states has incorrect shape.";
  }

  // GEMM1 or GEMM2 weights [num_experts, M, K] or [num_experts, K/block_k, M, block_k]
  void check_weights_shape(std::string which_weights) const {
    Tensor const* weights{};
    if (which_weights == "gemm1") {
      weights = gemm1_weights;
    } else if (which_weights == "gemm2") {
      weights = gemm2_weights;
    } else {
      TVM_FFI_LOG_AND_THROW(InternalError) << "Internal error: which_weights = " << which_weights;
    }

    int64_t Mn = 0, K = 0;
    if (weight_layout == MatrixLayout::MajorK) {
      // MajorK [num_experts, M, K]
      Mn = weights->shape[1];
      K = weights->shape[2];
    } else if (weight_layout == MatrixLayout::BlockMajorK) {
      // BlockMajorK [num_experts, K/block_k, M, block_k]
      Mn = weights->shape[2];
      int64_t block_k = weights->shape[3];
      K = weights->shape[1] * block_k;
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported weight_layout: " << weight_layout;
    }
    TVM_FFI_ICHECK_EQ(weights->shape[0], args->num_experts)
        << which_weights << " weights expert dimension must match num_experts";
    if (which_weights == "gemm1") {
      TVM_FFI_ICHECK_EQ(Mn % 2, 0) << which_weights << " weights Mn dimension must be even.";
      TVM_FFI_ICHECK_EQ(args->intermediate_size, Mn / 2)
          << "intermediate_size has incorrect shape.";
      TVM_FFI_ICHECK_EQ(K, hidden_states->shape[1])
          << which_weights << " weights K dimension must be equal to hidden_size.";
    } else if (which_weights == "gemm2") {
      TVM_FFI_ICHECK_EQ(K, args->intermediate_size)
          << which_weights << " weights K dimension must be equal to intermediate_size.";
    }
  }

  void check_routing_common() const {
    TVM_FFI_ICHECK(args->top_k > 0 && args->top_k <= args->num_experts)
        << "top_k must be between 1 and num_experts";
    TVM_FFI_ICHECK(args->local_num_experts > 0 && args->local_num_experts <= args->num_experts)
        << "local_num_experts must be between 1 and num_experts";
    TVM_FFI_ICHECK(args->local_expert_offset >= 0 &&
                   args->local_expert_offset + args->local_num_experts <= args->num_experts)
        << "expert offset and count must be within valid range";

    check_routing_logits_shape();

    if (routing_bias) {
      check_routing_bias_shape();
    }
  }

  // Routing phase workspace tensors (allocated in prepare_routing() or prepare_routing_common())
  Tensor num_tokens_per_expert;
  Tensor total_num_padded_tokens;
  Tensor expanded_idx_to_permuted_idx;
  Tensor permuted_idx_to_token_idx;
  Tensor expert_weights;
  Tensor expert_indexes;
  Tensor expert_count_histogram;
  Tensor cta_idx_xy_to_batch_idx;
  Tensor cta_idx_xy_to_mn_limit;
  Tensor num_non_exiting_ctas;

  void prepare_routing_common() {
    // Allocate routing phase workspace tensors
    int32_t max_num_padded_tokens =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxPermutedPaddedCount(
            args->num_tokens, args->top_k, args->num_experts, tile_tokens_dim);

    // Common routing workspace tensors allocation
    num_tokens_per_expert = alloc_tensor({args->num_experts}, dl_int32, routing_logits->device);

    total_num_padded_tokens = alloc_tensor({1}, dl_int32, routing_logits->device);

    expanded_idx_to_permuted_idx =
        alloc_tensor({args->num_tokens * args->top_k}, dl_int32, routing_logits->device);

    permuted_idx_to_token_idx =
        alloc_tensor({max_num_padded_tokens}, dl_int32, routing_logits->device);

    expert_indexes =
        alloc_tensor({args->num_tokens, args->top_k}, dl_int32, routing_logits->device);

    // expert_weights allocation should be done by derived class since data type could vary

    int64_t const size_of_expert_count_histogram = std::max(args->num_experts * 2, 256 * 2);
    expert_count_histogram = alloc_tensor({size_of_expert_count_histogram},
                                          dl_int32,  // 256 is the max number of threads per block
                                                     // and max number of experts
                                          routing_logits->device);

    int32_t max_num_ctas = tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxNumCtasInBatchDim(
        args->num_tokens, args->top_k, args->num_experts, tile_tokens_dim);

    cta_idx_xy_to_batch_idx = alloc_tensor({max_num_ctas}, dl_int32, routing_logits->device);

    cta_idx_xy_to_mn_limit = alloc_tensor({max_num_ctas}, dl_int32, routing_logits->device);

    num_non_exiting_ctas = alloc_tensor({1}, dl_int32, routing_logits->device);

    workspace.total_num_padded_tokens = static_cast<int*>(total_num_padded_tokens->data);
    workspace.total_max_padded_tokens = max_num_padded_tokens;
    workspace.ProjUpTileN = tile_tokens_dim;
    workspace.routing_expert_indexes = static_cast<int*>(expert_indexes->data);
    workspace.permuted_idx_size = static_cast<int*>(total_num_padded_tokens->data);
    workspace.expanded_idx_to_permuted_idx = static_cast<int*>(expanded_idx_to_permuted_idx->data);
    workspace.permuted_idx_to_token_idx = static_cast<int*>(permuted_idx_to_token_idx->data);
    // workspace.expert_weights will be set by derived class after expert_weights allocation
    workspace.cta_idx_xy_to_batch_idx = static_cast<int*>(cta_idx_xy_to_batch_idx->data);
    workspace.cta_idx_xy_to_mn_limit = static_cast<int*>(cta_idx_xy_to_mn_limit->data);
    workspace.num_non_exiting_ctas = static_cast<int*>(num_non_exiting_ctas->data);
  }

  void check_moe_common() const {
    // Hidden states [num_tokens, hidden_size]
    TVM_FFI_ICHECK_EQ(hidden_states->ndim, 2) << "hidden_states must be 2D.";
  }

  // MoE computation phase workspace tensors (allocated in prepare_moe() or prepare_moe_common())
  Tensor gemm1_output;
  Tensor activation_output;
  Tensor gemm2_output;
  Tensor workspace_fc1;
  Tensor workspace_fc2;
  Tensor output;
  int64_t moe_tactic{-1};
  std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner> moe_runner;

  void prepare_moe_common(int64_t& moe_tactic) {
    using RunnerType = tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner;
    moe_runner = std::make_unique<RunnerType>(this->mDtypeAct, this->mDtypeWeights,
                                              args->mUseDeepSeekFp8, (int32_t)tile_tokens_dim,
                                              static_cast<GatedActType>(this->gated_act_type),
                                              this->use_shuffled_weight, this->weight_layout);

    if (moe_tactic == -1) {
      moe_tactic = moe_runner->getDefaultValidConfigIndex(
          args->top_k, args->hidden_size, args->intermediate_size, args->local_num_experts,
          args->num_tokens);
    }
    this->moe_tactic = moe_tactic;

    auto workspace_sizes = moe_runner->getWorkspaceSizeInBytes(*args, moe_tactic);
    workspace_fc1 = alloc_tensor({std::get<0>(workspace_sizes)}, dl_int8, hidden_states->device);
    workspace_fc2 = alloc_tensor({std::get<1>(workspace_sizes)}, dl_int8, hidden_states->device);
    workspace.bmm1_workspace = workspace_fc1->data;
    workspace.bmm2_workspace = workspace_fc2->data;
  }

 public:
  virtual void check_routing() const = 0;
  virtual void prepare_routing() = 0;
  virtual void check_moe() const = 0;
  virtual void prepare_moe(int64_t& moe_tactic) = 0;

  // Main entry point for all the executions.
  // Do initializations prior to calling this as the initializations are different for bf16, fp8 and
  // fp4. The executions are non-blocking by default.
  Array<Tensor> run(int64_t moe_tactic, bool enable_pdl = true) {
    check_routing();
    prepare_routing();

    // Execute routing
    tensorrt_llm::kernels::trtllmgen_moe::Routing::Runner routing_runner(tile_tokens_dim);
    cudaStream_t routing_stream = get_stream(routing_logits->device);
    routing_runner.run(routing_logits->data, args->routing_bias, args->num_tokens,
                       args->num_experts, args->top_k, args->n_group, args->topk_group,
                       args->local_expert_offset, args->local_num_experts,
                       args->routed_scaling_factor, static_cast<int*>(expert_indexes->data),
                       static_cast<int*>(expert_count_histogram->data),
                       static_cast<int*>(total_num_padded_tokens->data),
                       static_cast<int*>(expanded_idx_to_permuted_idx->data),
                       nullptr /*permuted_idx_to_expanded_idx->data*/,
                       static_cast<int*>(permuted_idx_to_token_idx->data), expert_weights->data,
                       static_cast<int*>(num_tokens_per_expert->data),
                       static_cast<int*>(cta_idx_xy_to_batch_idx->data),
                       static_cast<int*>(cta_idx_xy_to_mn_limit->data),
                       static_cast<int*>(num_non_exiting_ctas->data), args->mDtypeElt, false, true,
                       static_cast<RoutingMethodType>(routing_method_type), routing_stream);

    check_moe();
    // if moe_tactic is -1, it will be set to the default valid config index
    prepare_moe(moe_tactic);

    // Execute MoE
    cudaStream_t moe_stream = get_stream(hidden_states->device);
    moe_runner->run(*args, workspace, hidden_states->device.device_id, moe_stream, moe_tactic,
                    enable_pdl);

    if (args->do_finalize) {
      return {output};
    }
    return {gemm2_output, expert_weights, expanded_idx_to_permuted_idx};
  }
};

void FusedMoeLauncher::init_common(
    Tensor const* routing_logits, Tensor const* routing_bias, Tensor const* hidden_states,
    Tensor const* gemm1_weights, Tensor const* gemm2_weights,
    std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
    int64_t tile_tokens_dim, int64_t routing_method_type, bool use_shuffled_weight,
    int64_t weight_layout, int64_t gated_act_type) {
  // Check devicearchitecture: Blackwell (SM 10.x) required
  TVM_FFI_ICHECK(hidden_states != nullptr) << "hidden_states is required";
  auto device = hidden_states->device.device_id;
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  TVM_FFI_ICHECK_EQ(major, 10) << "BF16 MoE requires 10.x architecture. Current device has SM "
                               << major << minor;
  this->device_version = std::make_tuple(major, minor);

  this->routing_logits = routing_logits;
  this->routing_bias = routing_bias;
  this->hidden_states = hidden_states;
  this->gemm1_weights = gemm1_weights;
  this->gemm2_weights = gemm2_weights;

  args->routing_logits = routing_logits->data;
  args->routing_bias = routing_bias ? routing_bias->data : nullptr;
  args->hidden_states = hidden_states->data;
  args->gemm1_weights = gemm1_weights->data;
  args->gemm2_weights = gemm2_weights->data;

  this->args = std::move(args);
  this->tile_tokens_dim = tile_tokens_dim;
  this->routing_method_type = routing_method_type;
  this->use_shuffled_weight = use_shuffled_weight;
  TVM_FFI_ICHECK(0 <= weight_layout && weight_layout <= 2)
      << "the value of weight_layout is not recognized";
  this->weight_layout = static_cast<MatrixLayout>(weight_layout);
  TVM_FFI_ICHECK(0 <= gated_act_type && gated_act_type <= 1)
      << "the value of gated_act_type is not recognized";
  this->gated_act_type = static_cast<GatedActType>(gated_act_type);
}

class Bf16MoeLauncher : public FusedMoeLauncher {
 public:
  Bf16MoeLauncher() = default;

  void init(Tensor const& routing_logits, Optional<Tensor> const& routing_bias,
            Tensor const& hidden_states, Tensor const& gemm1_weights, Tensor const& gemm2_weights,
            std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
            int64_t tile_tokens_dim, int64_t routing_method_type, bool use_shuffled_weight,
            int64_t weight_layout) {
    constexpr int64_t gated_act_type =
        static_cast<int64_t>(GatedActType::SwiGlu);  // not exposed in api for now

    // Do base class init and perform common checks
    FusedMoeLauncher::init_common(
        &routing_logits, routing_bias.has_value() ? &routing_bias.value() : nullptr, &hidden_states,
        &gemm1_weights, &gemm2_weights, std::move(args), tile_tokens_dim, routing_method_type,
        use_shuffled_weight, weight_layout, gated_act_type);
  }

  void check_routing() const override {
    FusedMoeLauncher::check_routing_common();

    // TODO n_group, topk_group validation?
  }

  void prepare_routing() override {
    FusedMoeLauncher::prepare_routing_common();

    args->mDtypeElt = btg::Dtype::Bfloat16;
    args->mDtypeExpW = btg::Dtype::Bfloat16;
    args->mUseDeepSeekFp8 = false;

    auto const routing_bias_dtype = dl_bfloat16;
    expert_weights =
        alloc_tensor({args->num_tokens, args->top_k}, routing_bias_dtype, routing_logits->device);

    workspace.expert_weights = expert_weights->data;
  }

  void check_moe() const override {
    FusedMoeLauncher::check_moe_common();

    TVM_FFI_ICHECK_EQ(weight_layout, MatrixLayout::BlockMajorK)
        << "BF16 Moe: weight_layout must be BlockMajorK";
    check_weights_shape("gemm1");
    check_weights_shape("gemm2");

    TVM_FFI_ICHECK_EQ(args->intermediate_size % 128, 0)
        << "the second dimension of weights must be a multiple of 128.";
  }

  void prepare_moe(int64_t& moe_tactic) override {
    // in the next line moe_tactic is passed by reference so modification will be propagated back
    // here
    FusedMoeLauncher::prepare_moe_common(moe_tactic);

    int32_t max_num_padded_tokens = workspace.total_max_padded_tokens;
    gemm1_output = alloc_tensor({max_num_padded_tokens, args->intermediate_size}, dl_bfloat16,
                                hidden_states->device);
    activation_output = alloc_tensor({max_num_padded_tokens, args->intermediate_size}, dl_bfloat16,
                                     hidden_states->device);
    gemm2_output = alloc_tensor({max_num_padded_tokens, args->hidden_size}, dl_bfloat16,
                                hidden_states->device);

    workspace.hidden_states_scale_linear = nullptr;
    workspace.gemm1_output = gemm1_output->data;
    workspace.gemm1_output_scale = nullptr;  // BF16 doesn't use scale tensors
    workspace.activation_output = activation_output->data;
    workspace.activation_output_scale = nullptr;  // BF16 doesn't use scale tensors
    workspace.gemm2_output = gemm2_output->data;
    workspace.gemm2_output_scale = nullptr;

    output =
        alloc_tensor({args->num_tokens, args->hidden_size}, dl_bfloat16, hidden_states->device);
    args->output = output->data;
    args->output_scale = nullptr;
  }
};

Tensor trtllm_bf16_moe(Tensor const& routing_logits, Optional<Tensor> const& routing_bias,
                       Tensor const& hidden_states, Tensor const& gemm1_weights,
                       Tensor const& gemm2_weights, int64_t num_experts, int64_t top_k,
                       int64_t n_group, int64_t topk_group, int64_t intermediate_size,
                       int64_t local_expert_offset, int64_t local_num_experts,
                       int64_t tile_tokens_dim, int64_t routing_method_type,
                       bool use_shuffled_weight, int64_t weight_layout, int64_t moe_tactic,
                       bool enable_pdl) {
  // Just some basic type validation first and leave more checks to the launcher
  TVM_FFI_ICHECK(routing_logits->dtype == dl_float32 || routing_logits->dtype == dl_bfloat16)
      << "BF16 MoE: routing_logits must be bfloat16 or float.";
  if (routing_bias.has_value()) {
    TVM_FFI_ICHECK_EQ(routing_bias.value()->dtype, dl_bfloat16)
        << "BF16 MoE: routing_bias must be bfloat16.";
  }
  TVM_FFI_ICHECK_EQ(hidden_states->dtype, dl_bfloat16)
      << "BF16 MoE: hidden_states must be bfloat16.";
  TVM_FFI_ICHECK_EQ(gemm1_weights->dtype, dl_bfloat16)
      << "BF16 MoE: gemm1_weights must be bfloat16.";
  TVM_FFI_ICHECK_EQ(gemm2_weights->dtype, dl_bfloat16)
      << "BF16 MoE: gemm2_weights must be bfloat16.";

  // Save params to MoE arguments
  auto args = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>();
  args->num_tokens = hidden_states->shape[0];
  args->num_experts = num_experts;
  args->hidden_size = hidden_states->shape[1];
  args->hidden_size_output = args->hidden_size;
  args->top_k = top_k;
  args->n_group = n_group;
  args->topk_group = topk_group;
  args->local_expert_offset = local_expert_offset;
  args->local_num_experts = local_num_experts;
  args->intermediate_size = intermediate_size;

  Bf16MoeLauncher launcher;
  launcher.init(routing_logits, routing_bias, hidden_states, gemm1_weights, gemm2_weights,
                std::move(args), tile_tokens_dim, routing_method_type, use_shuffled_weight,
                weight_layout);
  auto data = launcher.run(moe_tactic, enable_pdl)[0];
  return data;
}
using tvm::ffi::Array;
using tvm::ffi::Optional;

Tensor trtllm_fp8_per_tensor_scale_moe_launcher(
    Tensor routing_logits, Optional<Tensor> routing_bias, Tensor hidden_states,
    Tensor gemm1_weights, Tensor output1_scales_scalar, Tensor output1_scales_gate_scalar,
    Tensor gemm2_weights, Tensor output2_scales_scalar, int64_t const num_experts,
    int64_t const top_k, int64_t const n_group, int64_t const topk_group,
    int64_t const intermediate_size, int64_t const local_expert_offset,
    int64_t const local_num_experts, double const routed_scaling_factor,
    bool const use_routing_scales_on_input, int64_t const tile_tokens_dim,
    int64_t const routing_method_type, bool enable_pdl) {
  static const std::tuple<int, int> device_props = [hidden_states] {
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                           hidden_states->device.device_id);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                           hidden_states->device.device_id);
    return std::make_tuple(major, minor);
  }();

  TVM_FFI_ICHECK_EQ(std::get<0>(device_props), 10)
      << "This kernel requires 10.x architecture. Current device has SM "
      << std::get<0>(device_props) << std::get<1>(device_props);

  if (use_routing_scales_on_input) {
    TVM_FFI_ICHECK_EQ(routing_logits->dtype, dl_bfloat16) << "routing_logits must be bfloat16.";
  } else {
    TVM_FFI_ICHECK_EQ(routing_logits->dtype, dl_float32) << "routing_logits must be float.";
  }
  TVM_FFI_ICHECK_EQ(routing_logits->ndim, 2) << "routing_logits must be 2D.";
  TVM_FFI_ICHECK_EQ(routing_logits->shape[1], num_experts) << "routing_logits has incorrect shape.";
  if (routing_bias.has_value()) {
    TVM_FFI_ICHECK_EQ(routing_bias.value()->dtype, dl_bfloat16) << "routing_bias must be bfloat16.";
    TVM_FFI_ICHECK_EQ(routing_bias.value()->ndim, 1) << "routing_bias must be 1D.";
    TVM_FFI_ICHECK_EQ(routing_bias.value()->shape[0], num_experts)
        << "routing_bias has incorrect shape.";
  }

  if (n_group <= 0 || topk_group <= 0) {
    TVM_FFI_ICHECK_EQ(top_k, 1) << "Current routing kernel (no groups) only supports top_k=1.";
  } else {
    TVM_FFI_ICHECK_LE(top_k, 8) << "Current routing kernel (with groups) only supports top_k<=8.";
    TVM_FFI_ICHECK_LE(topk_group, 4)
        << "Current routing kernel (with groups) only supports topk_group<=4.";
    TVM_FFI_ICHECK_LE(topk_group, n_group) << "n_group must not be smaller than topk_group.";
    TVM_FFI_ICHECK_EQ(num_experts % n_group, 0) << "num_experts must be divisible by n_group";
    // This check ensures we have enough experts in the selected groups to handle the top_k routing
    TVM_FFI_ICHECK_LT(top_k, (topk_group * num_experts / n_group))
        << "top_k must be less than total number of experts in selected groups";
  }
  TVM_FFI_ICHECK_EQ(num_experts % 4, 0)
      << "Routing kernel expects that num_experts must be divisible by 4";
  TVM_FFI_ICHECK_GT(num_experts, top_k) << "num_experts must be greater than top_k";
  TVM_FFI_ICHECK_LE(local_num_experts + local_expert_offset, num_experts)
      << "num_experts must be greater or equal to local_num_experts + local_expert_offset";

  tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs args;
  tensorrt_llm::kernels::trtllmgen_moe::MoE::MoEWorkspace workspace;

  // Convert PyTorch dtype to TensorRT-LLM dtype
  auto dtype = hidden_states->dtype;
  if (dtype == dl_float16) {
    args.mDtypeElt = btg::Dtype::Fp16;
  } else if (dtype == dl_bfloat16) {
    args.mDtypeElt = btg::Dtype::Bfloat16;
  } else if (dtype == dl_float8_e4m3fn) {
    args.mDtypeElt = btg::Dtype::E4m3;
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported input dtype for MoE.";
  }

  args.routing_logits = routing_logits->data;
  auto const routing_bias_dtype =
      routing_bias.has_value() ? routing_bias.value()->dtype : dl_bfloat16;
  args.routing_bias = routing_bias.has_value() ? routing_bias.value()->data : nullptr;
  args.hidden_states = hidden_states->data;
  args.gemm1_weights = gemm1_weights->data;
  args.output1_scales_scalar = static_cast<float*>(output1_scales_scalar->data);
  args.output1_scales_gate_scalar = static_cast<float*>(output1_scales_gate_scalar->data);
  args.gemm2_weights = gemm2_weights->data;
  args.output2_scales_scalar = static_cast<float*>(output2_scales_scalar->data);
  args.num_tokens = hidden_states->shape[0];
  args.num_experts = num_experts;
  args.hidden_size = hidden_states->shape[1];
  args.hidden_size_output = args.hidden_size;
  args.top_k = top_k;
  args.n_group = n_group;
  args.topk_group = topk_group;
  args.local_expert_offset = local_expert_offset;
  args.local_num_experts = local_num_experts;
  args.routed_scaling_factor = routed_scaling_factor;
  args.intermediate_size = intermediate_size;
  args.mUseRoutingScalesOnInput = use_routing_scales_on_input;

  // allocate workspace for routing kernel
  Tensor num_tokens_per_expert = alloc_tensor({num_experts}, dl_int32, routing_logits->device);
  int32_t max_num_padded_tokens =
      tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxPermutedPaddedCount(
          args.num_tokens, top_k, num_experts, tile_tokens_dim);
  Tensor total_num_padded_tokens = alloc_tensor({1}, dl_int32, routing_logits->device);
  Tensor expanded_idx_to_permuted_idx =
      alloc_tensor({args.num_tokens * args.top_k}, dl_int32, routing_logits->device);
  Tensor permuted_idx_to_token_idx =
      alloc_tensor({max_num_padded_tokens}, dl_int32, routing_logits->device);
  Tensor expert_weights =
      alloc_tensor({args.num_tokens, args.top_k}, routing_bias_dtype, routing_logits->device);
  Tensor expert_indexes =
      alloc_tensor({args.num_tokens, args.top_k}, dl_int32, routing_logits->device);
  Tensor expert_count_histogram = alloc_tensor(
      {2 * 256},
      dl_int32,  // 256 is the max number of threads per block and max number of experts
      routing_logits->device);

  // allocate workspace for activation/gemm/finalize kernels
  // Tensor gemm1_output = alloc_tensor({max_num_padded_tokens, 2 * intermediate_size},
  //                                    dl_float8_e4m3fn, hidden_states->device);
  // Tensor activation_output = alloc_tensor({max_num_padded_tokens, intermediate_size},
  //                                         dl_float8_e4m3fn, hidden_states->device);
  Tensor gemm1_output =
      alloc_tensor({max_num_padded_tokens, 2 * intermediate_size}, dl_uint8, hidden_states->device);
  Tensor gemm1_output_scale = alloc_tensor({2 * intermediate_size / 128, max_num_padded_tokens},
                                           dl_float32, hidden_states->device);
  Tensor activation_output =
      alloc_tensor({max_num_padded_tokens, intermediate_size}, dl_uint8, hidden_states->device);
  Tensor activation_output_scale = alloc_tensor({intermediate_size / 128, max_num_padded_tokens},
                                                dl_float32, hidden_states->device);
  Tensor gemm2_output =
      alloc_tensor({max_num_padded_tokens, args.hidden_size}, dl_bfloat16, hidden_states->device);

  int32_t max_num_ctas = tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxNumCtasInBatchDim(
      args.num_tokens, args.top_k, args.num_experts, tile_tokens_dim);
  Tensor cta_idx_xy_to_batch_idx = alloc_tensor({max_num_ctas}, dl_int32, routing_logits->device);
  Tensor cta_idx_xy_to_mn_limit = alloc_tensor({max_num_ctas}, dl_int32, routing_logits->device);
  Tensor num_non_exiting_ctas = alloc_tensor({1}, dl_int32, routing_logits->device);

  tensorrt_llm::kernels::trtllmgen_moe::Routing::Runner routing_runner(tile_tokens_dim);
  cudaStream_t stream = get_stream(routing_logits->device);
  routing_runner.run(
      routing_logits->data, args.routing_bias, args.num_tokens, args.num_experts, args.top_k,
      args.n_group, args.topk_group, args.local_expert_offset, args.local_num_experts,
      args.routed_scaling_factor, static_cast<int*>(expert_indexes->data),
      static_cast<int*>(expert_count_histogram->data),
      static_cast<int*>(total_num_padded_tokens->data),
      static_cast<int*>(expanded_idx_to_permuted_idx->data),
      nullptr /*static_cast<int*>(permuted_idx_to_expanded_idx->data)*/,
      static_cast<int*>(permuted_idx_to_token_idx->data), expert_weights->data,
      static_cast<int*>(num_tokens_per_expert->data),
      static_cast<int*>(cta_idx_xy_to_batch_idx->data),
      static_cast<int*>(cta_idx_xy_to_mn_limit->data),
      static_cast<int*>(num_non_exiting_ctas->data), args.mDtypeElt, use_routing_scales_on_input,
      false /* use_deep_seek_fp8 */, static_cast<RoutingMethodType>(routing_method_type), stream);

  // MoE kernel except routing
  TVM_FFI_ICHECK_EQ(hidden_states->dtype, dl_float8_e4m3fn) << "hidden_states must be fp8.";
  TVM_FFI_ICHECK_EQ(gemm1_weights->dtype, dl_float8_e4m3fn) << "gemm1_weights must be fp8.";
  TVM_FFI_ICHECK_EQ(gemm1_weights->ndim, 3) << "gemm1_weights must be 3D.";
  TVM_FFI_ICHECK_EQ(gemm1_weights->shape[1] % 2, 0)
      << "the second dimension of weights must be even.";
  TVM_FFI_ICHECK_EQ(intermediate_size, gemm1_weights->shape[1] / 2)
      << "intermediate_size has incorrect shape.";
  TVM_FFI_ICHECK_EQ(gemm1_weights->shape[2], hidden_states->shape[1])
      << "the third dimension of weights must be equal to hidden_size.";
  TVM_FFI_ICHECK_EQ(intermediate_size % 128, 0)
      << "the second dimension of weights must be a multiple of 128.";

  TVM_FFI_ICHECK_EQ(output1_scales_scalar->dtype, dl_float32)
      << "output1_scales_scalar must be float.";
  TVM_FFI_ICHECK_EQ(output1_scales_scalar->ndim, 1) << "output1_scales_scalar must be 1D.";
  TVM_FFI_ICHECK_EQ(output1_scales_scalar->shape[0], local_num_experts)
      << "output1_scales_scalar has incorrect dim 0.";
  TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar->dtype, dl_float32)
      << "output1_scales_gate_scalar must be float.";
  TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar->ndim, 1)
      << "output1_scales_gate_scalar must be 1D.";
  TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar->shape[0], local_num_experts)
      << "output1_scales_gate_scalar has incorrect dim 0.";

  TVM_FFI_ICHECK_EQ(gemm2_weights->dtype, dl_float8_e4m3fn) << "gemm2_weights must be fp8.";
  TVM_FFI_ICHECK_EQ(gemm2_weights->ndim, 3) << "gemm2_weights must be 3D.";
  TVM_FFI_ICHECK_EQ(gemm2_weights->shape[2], intermediate_size)
      << "the third dimension of weights must be equal to intermediate_size.";

  TVM_FFI_ICHECK_EQ(output2_scales_scalar->dtype, dl_float32)
      << "output2_scales_scalar must be float.";
  TVM_FFI_ICHECK_EQ(output2_scales_scalar->ndim, 1) << "output2_scales_scalar must be 1D.";
  TVM_FFI_ICHECK_EQ(output2_scales_scalar->shape[0], local_num_experts)
      << "output2_scales_scalar has incorrect dim 0.";

  // allocate output
  Tensor output =
      alloc_tensor({args.num_tokens, args.hidden_size}, dl_bfloat16, hidden_states->device);

  // setup workspace
  workspace.total_num_padded_tokens = static_cast<int*>(total_num_padded_tokens->data);
  workspace.total_max_padded_tokens = max_num_padded_tokens;
  workspace.ProjUpTileN = tile_tokens_dim;
  workspace.routing_expert_indexes = static_cast<int*>(expert_indexes->data);
  workspace.permuted_idx_size = static_cast<int*>(total_num_padded_tokens->data);
  workspace.expanded_idx_to_permuted_idx = static_cast<int*>(
      expanded_idx_to_permuted_idx->data);  // Needed by activation/finalize kernels
  workspace.permuted_idx_to_token_idx =
      static_cast<int*>(permuted_idx_to_token_idx->data);  // Needed by permuteGemm1 kernel
  workspace.expert_weights = expert_weights->data;         // Consumed by finalize kernel

  workspace.cta_idx_xy_to_batch_idx = static_cast<int*>(cta_idx_xy_to_batch_idx->data);
  workspace.cta_idx_xy_to_mn_limit = static_cast<int*>(cta_idx_xy_to_mn_limit->data);
  workspace.num_non_exiting_ctas = static_cast<int*>(num_non_exiting_ctas->data);

  // gemm1 intermediate ws
  workspace.gemm1_output = gemm1_output->data;
  workspace.gemm1_output_scale = static_cast<float*>(gemm1_output_scale->data);
  // activation intermediate ws
  workspace.activation_output = activation_output->data;
  workspace.activation_output_scale = static_cast<float*>(activation_output_scale->data);
  // gemm2 intermediate ws
  workspace.gemm2_output = gemm2_output->data;
  workspace.gemm2_output_scale = nullptr;
  args.output = output->data;
  args.output_scale = nullptr;

  tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner moe_runner(
      args.mDtypeElt, args.mUseDeepSeekFp8, tile_tokens_dim, /*useShuffledMatrixA*/ true);

  auto const moeConfigIndex =
      moe_runner.getDefaultValidConfigIndex(args.top_k, args.hidden_size, args.intermediate_size,
                                            args.local_num_experts, args.num_tokens);

  auto workspace_sizes = moe_runner.getWorkspaceSizeInBytes(args, moeConfigIndex);
  Tensor workspace_fc1 =
      alloc_tensor({std::get<0>(workspace_sizes)}, dl_int8, hidden_states->device);
  Tensor workspace_fc2 =
      alloc_tensor({std::get<1>(workspace_sizes)}, dl_int8, hidden_states->device);
  workspace.bmm1_workspace = workspace_fc1->data;
  workspace.bmm2_workspace = workspace_fc2->data;
  cudaStream_t moe_stream = get_stream(hidden_states->device);
  moe_runner.run(args, workspace, hidden_states->device.device_id, moe_stream, moeConfigIndex,
                 enable_pdl);
  return output;
}

Tensor trtllm_fp8_per_tensor_scale_moe(
    Tensor routing_logits, Optional<Tensor> routing_bias, Tensor hidden_states,
    Tensor gemm1_weights, Tensor output1_scales_scalar, Tensor output1_scales_gate_scalar,
    Tensor gemm2_weights, Tensor output2_scales_scalar, int64_t num_experts, int64_t top_k,
    int64_t n_group, int64_t topk_group, int64_t intermediate_size, int64_t local_expert_offset,
    int64_t local_num_experts, double routed_scaling_factor, bool use_routing_scales_on_input,
    int64_t tile_tokens_dim, int64_t routing_method_type, bool enable_pdl) {
  auto dtype = hidden_states->dtype;
  if (dtype == dl_float16 || dtype == dl_bfloat16 || dtype == dl_float8_e4m3fn) {
    return trtllm_fp8_per_tensor_scale_moe_launcher(
        routing_logits, routing_bias, hidden_states, gemm1_weights, output1_scales_scalar,
        output1_scales_gate_scalar, gemm2_weights, output2_scales_scalar, num_experts, top_k,
        n_group, topk_group, intermediate_size, local_expert_offset, local_num_experts,
        routed_scaling_factor, use_routing_scales_on_input, tile_tokens_dim, routing_method_type,
        enable_pdl);
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported input dtype.";
  }
}

void trtllm_fp8_block_scale_moe_launcher(
    Tensor routing_logits, Optional<Tensor> routing_bias, Tensor hidden_states,
    Tensor hidden_states_scale, Tensor gemm1_weights, Tensor gemm1_weights_scale,
    Tensor gemm2_weights, Tensor gemm2_weights_scale, Tensor output, int64_t const num_experts,
    int64_t const top_k, int64_t const n_group, int64_t const topk_group,
    int64_t const intermediate_size, int64_t const local_expert_offset,
    int64_t const local_num_experts, double const routed_scaling_factor,
    int64_t const tile_tokens_dim, int64_t const routing_method_type,
    tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner& moe_runner, int64_t moeConfigIndex,
    bool enable_pdl) {
  static const std::tuple<int, int> device_props = [hidden_states] {
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                           hidden_states->device.device_id);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                           hidden_states->device.device_id);
    return std::make_tuple(major, minor);
  }();

  TVM_FFI_ICHECK_EQ(std::get<0>(device_props), 10)
      << "This kernel requires 10.x architecture. Current device has SM "
      << std::get<0>(device_props) << std::get<1>(device_props);

  TVM_FFI_ICHECK_EQ(routing_logits->dtype, dl_float32) << "routing_logits must be float.";
  TVM_FFI_ICHECK_EQ(routing_logits->ndim, 2) << "routing_logits must be 2D.";
  TVM_FFI_ICHECK_EQ(routing_logits->shape[0], hidden_states->shape[0])
      << "routing_logits and hidden_states must have the same number of tokens.";
  TVM_FFI_ICHECK_EQ(routing_logits->shape[1], num_experts)
      << "routing_logits dim1 must match num_experts.";
  if (routing_bias.has_value()) {
    TVM_FFI_ICHECK(routing_bias.value()->dtype == dl_bfloat16 ||
                   routing_bias.value()->dtype == dl_float32)
        << "routing_bias must be bfloat16 or float.";
    TVM_FFI_ICHECK_EQ(routing_bias.value()->ndim, 1) << "routing_bias must be 1D.";
    TVM_FFI_ICHECK_EQ(routing_bias.value()->shape[0], num_experts)
        << "routing_bias has incorrect shape.";
  }

  if (n_group <= 0 || topk_group <= 0) {
    TVM_FFI_ICHECK_EQ(top_k, 1) << "Current routing kernel (no groups) only supports top_k=1.";
  } else {
    TVM_FFI_ICHECK_LE(top_k, 8) << "Current routing kernel (with groups) only supports top_k<=8.";
    TVM_FFI_ICHECK_LE(topk_group, 4)
        << "Current routing kernel (with groups) only supports topk_group<=4.";
    TVM_FFI_ICHECK_LE(topk_group, n_group) << "n_group must not be smaller than topk_group.";
    TVM_FFI_ICHECK_EQ(num_experts % n_group, 0) << "num_experts must be divisible by n_group";
    // This check ensures we have enough experts in the selected groups to handle the top_k routing
    TVM_FFI_ICHECK_LT(top_k, (topk_group * num_experts / n_group))
        << "top_k must be less than total number of experts in selected groups";
  }
  TVM_FFI_ICHECK_EQ(num_experts % 4, 0)
      << "Routing kernel expects that num_experts must be divisible by 4";
  TVM_FFI_ICHECK_GT(num_experts, top_k) << "num_experts must be greater than top_k";

  tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs args;
  tensorrt_llm::kernels::trtllmgen_moe::MoE::MoEWorkspace workspace;

  // Convert PyTorch dtype to TensorRT-LLM dtype
  auto dtype = hidden_states->dtype;
  if (dtype == dl_float16) {
    args.mDtypeElt = btg::Dtype::Fp16;
  } else if (dtype == dl_bfloat16) {
    args.mDtypeElt = btg::Dtype::Bfloat16;
  } else if (dtype == dl_float8_e4m3fn) {
    args.mDtypeElt = btg::Dtype::E4m3;
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported input dtype for MoE.";
  }

  auto const routing_bias_dtype =
      routing_bias.has_value() ? routing_bias.value()->dtype : dl_bfloat16;
  args.mDtypeExpW = routing_bias_dtype == dl_bfloat16 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;
  args.routing_logits = static_cast<float*>(routing_logits->data);
  args.routing_bias = routing_bias.has_value() ? routing_bias.value()->data : nullptr;
  args.hidden_states = hidden_states->data;
  args.hidden_states_scale = static_cast<float*>(hidden_states_scale->data);
  args.gemm1_weights = gemm1_weights->data;
  args.gemm1_weights_scale = static_cast<float*>(gemm1_weights_scale->data);
  args.gemm2_weights = gemm2_weights->data;
  args.gemm2_weights_scale = static_cast<float*>(gemm2_weights_scale->data);
  args.num_tokens = hidden_states->shape[0];
  args.num_experts = num_experts;
  args.hidden_size = hidden_states->shape[1];
  args.hidden_size_output = args.hidden_size;
  args.top_k = top_k;
  args.n_group = n_group;
  args.topk_group = topk_group;
  args.local_expert_offset = local_expert_offset;
  args.local_num_experts = local_num_experts;
  args.routed_scaling_factor = routed_scaling_factor;
  args.intermediate_size = intermediate_size;
  args.mUseDeepSeekFp8 = true;

  // allocate workspace for routing kernel
  Tensor num_tokens_per_expert = alloc_tensor({num_experts}, dl_int32, routing_logits->device);
  int32_t max_num_padded_tokens =
      tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxPermutedPaddedCount(
          args.num_tokens, top_k, num_experts, tile_tokens_dim);
  Tensor total_num_padded_tokens = alloc_tensor({1}, dl_int32, routing_logits->device);
  Tensor expanded_idx_to_permuted_idx =
      alloc_tensor({args.num_tokens * args.top_k}, dl_int32, routing_logits->device);
  Tensor permuted_idx_to_token_idx =
      alloc_tensor({max_num_padded_tokens}, dl_int32, routing_logits->device);
  Tensor expert_weights =
      alloc_tensor({args.num_tokens, args.top_k}, routing_bias_dtype, routing_logits->device);
  Tensor expert_indexes =
      alloc_tensor({args.num_tokens, args.top_k}, dl_int32, routing_logits->device);
  int64_t const size_of_expert_count_histogram = std::max(num_experts * 2, int64_t(256 * 2));
  Tensor expert_count_histogram = alloc_tensor(
      {size_of_expert_count_histogram},
      dl_int32,  // 256 is the max number of threads per block and max number of experts
      routing_logits->device);

  // allocate workspace for activation/gemm/finalize kernels
  // Tensor gemm1_output = alloc_tensor({max_num_padded_tokens, 2 * intermediate_size},
  //                                    dl_float8_e4m3fn, hidden_states->device);
  // Tensor activation_output = alloc_tensor({max_num_padded_tokens, intermediate_size},
  //                                         dl_float8_e4m3fn, hidden_states->device);
  Tensor gemm1_output =
      alloc_tensor({max_num_padded_tokens, 2 * intermediate_size}, dl_uint8, hidden_states->device);
  Tensor gemm1_output_scale = alloc_tensor({2 * intermediate_size / 128, max_num_padded_tokens},
                                           dl_float32, hidden_states->device);
  Tensor activation_output =
      alloc_tensor({max_num_padded_tokens, intermediate_size}, dl_uint8, hidden_states->device);
  Tensor activation_output_scale = alloc_tensor({intermediate_size / 128, max_num_padded_tokens},
                                                dl_float32, hidden_states->device);
  Tensor gemm2_output =
      alloc_tensor({max_num_padded_tokens, args.hidden_size}, dl_bfloat16, hidden_states->device);

  int32_t max_num_ctas = tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxNumCtasInBatchDim(
      args.num_tokens, args.top_k, args.num_experts, tile_tokens_dim);
  Tensor cta_idx_xy_to_batch_idx = alloc_tensor({max_num_ctas}, dl_int32, routing_logits->device);
  Tensor cta_idx_xy_to_mn_limit = alloc_tensor({max_num_ctas}, dl_int32, routing_logits->device);
  Tensor num_non_exiting_ctas = alloc_tensor({1}, dl_int32, routing_logits->device);

  tensorrt_llm::kernels::trtllmgen_moe::Routing::Runner routing_runner(tile_tokens_dim);
  cudaStream_t stream = get_stream(routing_logits->device);
  routing_runner.run(static_cast<float*>(routing_logits->data), args.routing_bias, args.num_tokens,
                     args.num_experts, args.top_k, args.n_group, args.topk_group,
                     args.local_expert_offset, args.local_num_experts, args.routed_scaling_factor,
                     static_cast<int*>(expert_indexes->data),
                     static_cast<int*>(expert_count_histogram->data),
                     static_cast<int*>(total_num_padded_tokens->data),
                     static_cast<int*>(expanded_idx_to_permuted_idx->data),
                     nullptr /*static_cast<int*>(permuted_idx_to_expanded_idx->data)*/,
                     static_cast<int*>(permuted_idx_to_token_idx->data), expert_weights->data,
                     static_cast<int*>(num_tokens_per_expert->data),
                     static_cast<int*>(cta_idx_xy_to_batch_idx->data),
                     static_cast<int*>(cta_idx_xy_to_mn_limit->data),
                     static_cast<int*>(num_non_exiting_ctas->data), args.mDtypeElt, false, true,
                     static_cast<RoutingMethodType>(routing_method_type), stream);

  // MoE kernel except routing
  TVM_FFI_ICHECK_EQ(hidden_states->dtype, dl_float8_e4m3fn) << "hidden_states must be fp8.";
  TVM_FFI_ICHECK_EQ(hidden_states_scale->dtype, dl_float32) << "hidden_states_scale must be float.";
  TVM_FFI_ICHECK_EQ(hidden_states_scale->ndim, 2) << "hidden_states_scale must be 2D.";
  TVM_FFI_ICHECK_EQ(hidden_states_scale->shape[0], hidden_states->shape[1] / 128)
      << "hidden_states_scale dim0 must match hidden_states dim1 / 128.";
  TVM_FFI_ICHECK_EQ(hidden_states_scale->shape[1], args.num_tokens)
      << "hidden_states_scale dim1 must match num_tokens.";
  TVM_FFI_ICHECK_EQ(gemm1_weights->dtype, dl_float8_e4m3fn) << "gemm1_weights must be fp8.";

  TVM_FFI_ICHECK(gemm1_weights->ndim == 3 || gemm1_weights->ndim == 4)
      << "gemm1_weights must be 3D or 4D.";
  {
    int64_t Mn = 0, K = 0;
    if (gemm1_weights->ndim == 3) {
      // MajorK [num_experts, M, K]
      Mn = gemm1_weights->shape[1];
      K = gemm1_weights->shape[2];
    } else if (gemm1_weights->ndim == 4) {
      // BlockMajorK [num_experts, K/block_k, M, block_k]
      Mn = gemm1_weights->shape[2];
      int64_t block_k = gemm1_weights->shape[3];
      K = gemm1_weights->shape[1] * block_k;
    }
    TVM_FFI_ICHECK_EQ(Mn % 2, 0) << "the second dimension of weights must be even.";
    TVM_FFI_ICHECK_EQ(intermediate_size, Mn / 2) << "intermediate_size has incorrect shape.";
    TVM_FFI_ICHECK_EQ(K, hidden_states->shape[1])
        << "the third dimension of weights must be equal to hidden_size.";
  }
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale->dtype, dl_float32) << "gemm1_weights_scale must be float.";
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale->ndim, 3) << "gemm1_weights_scale must be 3D.";

  TVM_FFI_ICHECK_EQ(gemm1_weights_scale->shape[0], local_num_experts)
      << "gemm1_weights_scale has incorrect shape.";
  TVM_FFI_ICHECK_EQ(intermediate_size % 128, 0)
      << "the second dimension of weights must be a multiple of 128.";
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale->shape[1], 2 * intermediate_size / 128)
      << "gemm1_weights_scale has incorrect shape.";
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale->shape[2], args.hidden_size / 128)
      << "gemm1_weights_scale has incorrect shape.";
  TVM_FFI_ICHECK_EQ(gemm2_weights->dtype, dl_float8_e4m3fn) << "gemm2_weights must be fp8.";

  TVM_FFI_ICHECK(gemm2_weights->ndim == 3 || gemm2_weights->ndim == 4)
      << "gemm2_weights must be 3D or 4D.";
  {
    int64_t K = 0;
    if (gemm2_weights->ndim == 3) {
      // MajorK [num_experts, M, K]
      K = gemm2_weights->shape[2];
    } else if (gemm2_weights->ndim == 4) {
      // BlockMajorK [num_experts, K/block_k, M, block_k]
      int64_t block_k = gemm2_weights->shape[3];
      K = gemm2_weights->shape[1] * block_k;
    }
    TVM_FFI_ICHECK_EQ(K, intermediate_size)
        << "the third dimension of weights must be equal to intermediate_size.";
  }
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale->dtype, dl_float32) << "gemm2_weights_scale must be float.";
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale->ndim, 3) << "gemm2_weights_scale must be 3D.";
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale->shape[0], local_num_experts)
      << "gemm2_weights_scale has incorrect shape.";
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale->shape[1], args.hidden_size / 128)
      << "gemm2_weights_scale has incorrect shape.";
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale->shape[2], intermediate_size / 128)
      << "gemm2_weights_scale has incorrect shape.";

  TVM_FFI_ICHECK_EQ(output->shape[0], args.num_tokens) << "output has incorrect shape.";
  TVM_FFI_ICHECK_EQ(output->shape[1], args.hidden_size) << "output has incorrect shape.";
  TVM_FFI_ICHECK_EQ(output->dtype, dl_bfloat16) << "output must be bf16.";

  // setup workspace
  workspace.total_num_padded_tokens = static_cast<int*>(total_num_padded_tokens->data);
  workspace.total_max_padded_tokens = max_num_padded_tokens;
  workspace.ProjUpTileN = tile_tokens_dim;
  workspace.routing_expert_indexes = static_cast<int*>(expert_indexes->data);
  workspace.permuted_idx_size = static_cast<int*>(total_num_padded_tokens->data);
  workspace.expanded_idx_to_permuted_idx = static_cast<int*>(
      expanded_idx_to_permuted_idx->data);  // Needed by activation/finalize kernels
  workspace.permuted_idx_to_token_idx =
      static_cast<int*>(permuted_idx_to_token_idx->data);  // Needed by permuteGemm1 kernel
  workspace.expert_weights = expert_weights->data;         // Consumed by finalize kernel

  workspace.cta_idx_xy_to_batch_idx = static_cast<int*>(cta_idx_xy_to_batch_idx->data);
  workspace.cta_idx_xy_to_mn_limit = static_cast<int*>(cta_idx_xy_to_mn_limit->data);
  workspace.num_non_exiting_ctas = static_cast<int*>(num_non_exiting_ctas->data);

  // gemm1 intermediate ws
  workspace.gemm1_output = gemm1_output->data;
  workspace.gemm1_output_scale = static_cast<float*>(gemm1_output_scale->data);
  // activation intermediate ws
  workspace.activation_output = activation_output->data;
  workspace.activation_output_scale = static_cast<float*>(activation_output_scale->data);
  // gemm2 intermediate ws
  workspace.gemm2_output = gemm2_output->data;
  workspace.gemm2_output_scale = nullptr;
  args.output = output->data;
  args.output_scale = nullptr;

  auto workspace_sizes = moe_runner.getWorkspaceSizeInBytes(args, moeConfigIndex);
  Tensor workspace_fc1 =
      alloc_tensor({std::get<0>(workspace_sizes)}, dl_int8, hidden_states->device);
  Tensor workspace_fc2 =
      alloc_tensor({std::get<1>(workspace_sizes)}, dl_int8, hidden_states->device);
  workspace.bmm1_workspace = workspace_fc1->data;
  workspace.bmm2_workspace = workspace_fc2->data;

  cudaStream_t moe_stream = get_stream(hidden_states->device);
  moe_runner.run(args, workspace, hidden_states->device.device_id, moe_stream, moeConfigIndex,
                 enable_pdl);
}

void trtllm_fp8_block_scale_moe(Tensor routing_logits, Optional<Tensor> routing_bias,
                                Tensor hidden_states, Tensor hidden_states_scale,
                                Tensor gemm1_weights, Tensor gemm1_weights_scale,
                                Tensor gemm2_weights, Tensor gemm2_weights_scale, Tensor output,
                                int64_t num_experts, int64_t top_k, int64_t n_group,
                                int64_t topk_group, int64_t intermediate_size,
                                int64_t local_expert_offset, int64_t local_num_experts,
                                double routed_scaling_factor, int64_t tile_tokens_dim,
                                int64_t routing_method_type, bool use_shuffled_weight,
                                int64_t weight_layout, bool enable_pdl) {
  auto dtype = hidden_states->dtype;
  if (dtype == dl_float16 || dtype == dl_bfloat16 || dtype == dl_float8_e4m3fn) {
    using RunnerType = tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner;

    btg::Dtype mDtypeElt{btg::Dtype::E4m3};  // FP8 runner so hard-coded
    bool mUseDeepSeekFp8{true};              // Always true for BlockScaleMoe

    TVM_FFI_ICHECK(0 <= weight_layout && weight_layout <= 2)
        << "the value of weight_layout is not recognized";

    // Properly initialize the runner using make_unique like in the original code
    auto mRunner = std::make_unique<RunnerType>(
        mDtypeElt, mUseDeepSeekFp8, tile_tokens_dim, use_shuffled_weight,
        static_cast<batchedGemm::gemm::MatrixLayout>(weight_layout));

    // Always use fallback config (equivalent to moeConfigIndex == -1 case from original code)
    auto const num_tokens = hidden_states->shape[0];
    auto const hidden_size = hidden_states->shape[1];

    int64_t moeConfigIndex = mRunner->getDefaultValidConfigIndex(
        top_k, hidden_size, intermediate_size, local_num_experts, num_tokens);

    return trtllm_fp8_block_scale_moe_launcher(
        routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights,
        gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, output, num_experts, top_k,
        n_group, topk_group, intermediate_size, local_expert_offset, local_num_experts,
        routed_scaling_factor, tile_tokens_dim, routing_method_type, *mRunner, moeConfigIndex,
        enable_pdl);
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported hidden state dtype.";
  }
}

// TODO(siyuan): This launcher supports flexible weight and activation types.
// We should cleanup other launchers and only use this one in the future.
Array<Tensor> trtllm_fp4_block_scale_moe_launcher(
    Optional<Tensor> routing_logits, Tensor expert_indices, Tensor expert_weights,
    Optional<Tensor> routing_bias, Tensor hidden_states, Optional<Tensor> hidden_states_scale,
    Tensor gemm1_weights, Tensor gemm1_weights_scale, Optional<Tensor> gemm1_bias,
    Optional<Tensor> gemm1_alpha, Optional<Tensor> gemm1_beta, Optional<Tensor> gemm1_clamp_limit,
    Tensor gemm2_weights, Tensor gemm2_weights_scale, Optional<Tensor> gemm2_bias,
    Optional<Tensor> output1_scales_scalar, Optional<Tensor> output1_scales_gate_scalar,
    Optional<Tensor> output2_scales_scalar, int64_t const num_experts, int64_t const top_k,
    Optional<int64_t> const n_group, Optional<int64_t> const topk_group,
    int64_t const intermediate_size, int64_t const local_expert_offset,
    int64_t const local_num_experts, Optional<double> const routed_scaling_factor,
    int64_t const tile_tokens_dim, int64_t const routing_method_type, bool const do_finalize,
    tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner& moe_runner, btg::Dtype dtype_act,
    btg::Dtype dtype_weights, int64_t const moeConfigIndex, bool enable_pdl, Tensor output) {
  static const std::tuple<int, int> device_props = [hidden_states] {
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                           hidden_states->device.device_id);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                           hidden_states->device.device_id);
    return std::make_tuple(major, minor);
  }();

  TVM_FFI_ICHECK_EQ(std::get<0>(device_props), 10)
      << "This kernel requires 10.x architecture. Current device has SM "
      << std::get<0>(device_props) << std::get<1>(device_props);

  TVM_FFI_ICHECK(dtype_act == btg::Dtype::E2m1 || dtype_act == btg::Dtype::Bfloat16 ||
                 dtype_act == btg::Dtype::E4m3 || dtype_act == btg::Dtype::MxE4m3)
      << "Only E2m1, Bfloat16, MxE4m3 and E4m3 are supported by block scale MoE";
  if (dtype_act == btg::Dtype::E2m1) {
    TVM_FFI_ICHECK(dtype_weights == btg::Dtype::E2m1)
        << "Only E2m1 and MxE2m1 are supported by block scale MoE with E2m1 activation";
    TVM_FFI_ICHECK(hidden_states_scale.has_value())
        << "hidden_states_scale is required for E2m1 activation";
    TVM_FFI_ICHECK(output1_scales_scalar.has_value())
        << "output1_scales_scalar is required for E2m1 activation";
    TVM_FFI_ICHECK(output1_scales_gate_scalar.has_value())
        << "output1_scales_gate_scalar is required for E2m1 activation";
    TVM_FFI_ICHECK(output2_scales_scalar.has_value())
        << "output2_scales_scalar is required for E2m1 activation";
  } else if (dtype_act == btg::Dtype::Bfloat16 || dtype_act == btg::Dtype::E4m3 ||
             dtype_act == btg::Dtype::MxE4m3) {
    TVM_FFI_ICHECK(dtype_weights == btg::Dtype::MxE2m1)
        << "Only MxE2m1 weights are supported by block scale MoE with Bfloat16, E4m3 or "
           "MxE4m3 activation";
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported act dtype.";
  }

  if (dtype_act == btg::Dtype::E4m3) {
    TVM_FFI_ICHECK(output1_scales_scalar.has_value())
        << "output1_scales_scalar is required for E4m3 activation";
    TVM_FFI_ICHECK(output1_scales_gate_scalar.has_value())
        << "output1_scales_gate_scalar is required for E4m3 activation";
    TVM_FFI_ICHECK(output2_scales_scalar.has_value())
        << "output2_scales_scalar is required for E4m3 activation";
  }

  if (routing_logits.has_value()) {
    TVM_FFI_ICHECK(routing_logits.value()->dtype == dl_float32 ||
                   routing_logits.value()->dtype == dl_bfloat16)
        << "routing_logits must be float or bfloat16.";
    TVM_FFI_ICHECK_EQ(routing_logits.value()->ndim, 2) << "routing_logits must be 2D.";
    TVM_FFI_ICHECK_EQ(routing_logits.value()->shape[1], num_experts)
        << "routing_logits has incorrect shape.";
  }
  if (routing_bias.has_value()) {
    TVM_FFI_ICHECK_EQ(routing_bias.value()->dtype, dl_bfloat16) << "routing_bias must be bfloat16.";
    TVM_FFI_ICHECK_EQ(routing_bias.value()->ndim, 1) << "routing_bias must be 1D.";
    TVM_FFI_ICHECK_EQ(routing_bias.value()->shape[0], num_experts)
        << "routing_bias has incorrect shape.";
  }

  if (n_group.value_or(0) != 0) {
    TVM_FFI_ICHECK(static_cast<RoutingMethodType>(routing_method_type) ==
                   RoutingMethodType::DeepSeekV3)
        << "Routing kernel with groups implies DeepSeekV3 routing method.";
    TVM_FFI_ICHECK(topk_group.has_value()) << "if n_group is given, topk_group must be given";
    TVM_FFI_ICHECK_EQ(num_experts % n_group.value(), 0)
        << "num_experts must be divisible by n_group";
    TVM_FFI_ICHECK(top_k <= 8 && top_k > 0)
        << "Current routing kernel (with groups) only supports top_k<=8 && top_k>0.";
    TVM_FFI_ICHECK(topk_group.value() <= 4 && topk_group.value() > 0)
        << "Current routing kernel only (with groups) supports topk_group<=4 && topk_group > 0.";
    TVM_FFI_ICHECK_LE(topk_group.value(), n_group.value())
        << "n_group must not be smaller than topk_group.";
    // This check ensures we have enough experts in the selected groups to handle the top_k routing
    TVM_FFI_ICHECK_LT(top_k, (topk_group.value() * num_experts / n_group.value()))
        << "top_k must be less than total number of experts in selected groups";
  } else if (static_cast<RoutingMethodType>(routing_method_type) ==
                 RoutingMethodType::Renormalize ||
             static_cast<RoutingMethodType>(routing_method_type) ==
                 RoutingMethodType::RenormalizeNaive ||
             static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::TopK) {
    TVM_FFI_ICHECK(top_k <= 8 && top_k > 0)
        << "Current routing kernel (no groups, renormalize/topk) only supports top_k<=8 && "
           "top_k>0.";
  } else if (static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::Llama4) {
    TVM_FFI_ICHECK_EQ(top_k, 1)
        << "Current routing kernel (no groups, Llama4) only supports top_k=1.";
  }

  TVM_FFI_ICHECK_EQ(num_experts % 4, 0)
      << "Routing kernel expects that num_experts must be divisible by 4";
  TVM_FFI_ICHECK_GT(num_experts, top_k) << "num_experts must be greater than top_k";

  tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs args;
  tensorrt_llm::kernels::trtllmgen_moe::MoE::MoEWorkspace workspace;

  // setup args
  // note: the assumption is that output data type is always Bfloat16 (the default)
  auto routing_bias_dtype = dl_bfloat16;
  if (routing_bias.has_value()) {
    routing_bias_dtype = routing_bias.value()->dtype;
  } else if (routing_logits.has_value()) {
    routing_bias_dtype = routing_logits.value()->dtype;
  }
  args.mDtypeElt = dtype_act;
  args.mDtypeExpW = routing_bias_dtype == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;
  args.routing_logits = routing_logits.has_value() ? routing_logits.value()->data : nullptr;
  args.routing_bias = routing_bias.has_value() ? routing_bias.value()->data : nullptr;
  args.hidden_states = hidden_states->data;
  args.hidden_states_scale =
      hidden_states_scale.has_value() ? hidden_states_scale.value()->data : nullptr;
  args.gemm1_weights = gemm1_weights->data;
  args.gemm1_weights_scale = gemm1_weights_scale->data;
  args.gemm1_bias =
      gemm1_bias.has_value() ? static_cast<float*>(gemm1_bias.value()->data) : nullptr;
  args.gemm1_alpha =
      gemm1_alpha.has_value() ? static_cast<float*>(gemm1_alpha.value()->data) : nullptr;
  args.gemm1_beta =
      gemm1_beta.has_value() ? static_cast<float*>(gemm1_beta.value()->data) : nullptr;
  args.gemm1_clamp_limit = gemm1_clamp_limit.has_value()
                               ? static_cast<float*>(gemm1_clamp_limit.value()->data)
                               : nullptr;
  args.gemm2_weights = gemm2_weights->data;
  args.gemm2_weights_scale = gemm2_weights_scale->data;
  args.gemm2_bias =
      gemm2_bias.has_value() ? static_cast<float*>(gemm2_bias.value()->data) : nullptr;
  args.num_tokens = hidden_states->shape[0];
  args.num_experts = num_experts;
  // * 2 to compensate for the fact that sizeof(hidden_states.dtype) is 1 because we pack 2 e2m1
  // into 1 byte.
  auto const hidden_states_hidden_size =
      dtype_act == btg::Dtype::E2m1 ? hidden_states->shape[1] * 2 : hidden_states->shape[1];
  args.hidden_size = hidden_states_hidden_size;
  args.hidden_size_output = args.hidden_size;
  args.top_k = top_k;
  args.n_group = n_group.value_or(0);
  args.topk_group = topk_group.value_or(0);
  args.local_expert_offset = local_expert_offset;
  args.local_num_experts = local_num_experts;
  args.routed_scaling_factor = routed_scaling_factor.value_or(1.0);
  args.intermediate_size = intermediate_size;

  // allocate workspace for routing kernel
  Tensor num_tokens_per_expert = alloc_tensor({num_experts}, dl_int32, hidden_states->device);
  int32_t max_num_padded_tokens =
      tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxPermutedPaddedCount(
          args.num_tokens, top_k, num_experts, tile_tokens_dim);
  Tensor total_num_padded_tokens = alloc_tensor({1}, dl_int32, hidden_states->device);
  Tensor expanded_idx_to_permuted_idx =
      alloc_tensor({args.num_tokens, args.top_k}, dl_int32, hidden_states->device);

  Tensor permuted_idx_to_token_idx =
      alloc_tensor({max_num_padded_tokens}, dl_int32, hidden_states->device);
  // Tensor expert_weights = alloc_tensor(
  //     {args.num_tokens, args.top_k}, routing_bias_dtype, hidden_states->device);
  // Tensor expert_indexes = alloc_tensor(
  //     {args.num_tokens, args.top_k}, dl_int32, hidden_states->device);
  int constexpr MAX_NUM_EXPERTS = 384;
  Tensor expert_count_histogram = alloc_tensor(
      {2 * MAX_NUM_EXPERTS},
      dl_int32,  // 256 is the max number of threads per block and max number of experts
      hidden_states->device);

  auto const sf_vec_size = dtype_weights == btg::Dtype::MxE2m1 ? 32 : 16;

  // allocate workspace for activation/gemm/finalize kernels
  auto const gemm1_output_hidden =
      dtype_act == btg::Dtype::E2m1 ? intermediate_size / 2 : intermediate_size;
  // Tensor gemm1_output = alloc_tensor(
  //     {max_num_padded_tokens, gemm1_output_hidden},
  //     dtype_act == btg::Dtype::Bfloat16 ? dl_bfloat16 : dl_float8_e4m3fn, hidden_states->device);
  Tensor gemm1_output = alloc_tensor({max_num_padded_tokens, gemm1_output_hidden},
                                     dtype_act == btg::Dtype::Bfloat16 ? dl_bfloat16 : dl_uint8,
                                     hidden_states->device);

  Optional<Tensor> gemm1_output_scale = std::nullopt;
  if (dtype_act == btg::Dtype::E2m1 || dtype_act == btg::Dtype::MxE4m3) {
    int64_t sf_size = tensorrt_llm::computeSwizzledLayoutSFSize(max_num_padded_tokens,
                                                                intermediate_size / sf_vec_size);
    // gemm1_output_scale = alloc_tensor({sf_size}, dl_float8_e4m3fn, hidden_states->device);
    gemm1_output_scale = alloc_tensor({sf_size}, dl_uint8, hidden_states->device);
  }

  Tensor gemm2_output =
      alloc_tensor({max_num_padded_tokens, args.hidden_size}, dl_bfloat16, hidden_states->device);

  int32_t max_num_ctas = tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxNumCtasInBatchDim(
      args.num_tokens, args.top_k, args.num_experts, tile_tokens_dim);
  Tensor cta_idx_xy_to_batch_idx = alloc_tensor({max_num_ctas}, dl_int32, hidden_states->device);
  Tensor cta_idx_xy_to_mn_limit = alloc_tensor({max_num_ctas}, dl_int32, hidden_states->device);
  Tensor num_non_exiting_ctas = alloc_tensor({1}, dl_int32, hidden_states->device);

  //
  // TopK routing
  //

  tensorrt_llm::kernels::trtllmgen_moe::Routing::Runner routing_runner(tile_tokens_dim);
  cudaStream_t stream = get_stream(hidden_states->device);
  routing_runner.run(args.routing_logits, args.routing_bias, args.num_tokens, args.num_experts,
                     args.top_k, args.n_group, args.topk_group, args.local_expert_offset,
                     args.local_num_experts, args.routed_scaling_factor,
                     static_cast<int*>(expert_indices->data),
                     static_cast<int*>(expert_count_histogram->data),
                     static_cast<int*>(total_num_padded_tokens->data),
                     static_cast<int*>(expanded_idx_to_permuted_idx->data),
                     nullptr, /*static_cast<int*>(permuted_idx_to_expanded_idx->data),*/
                     static_cast<int*>(permuted_idx_to_token_idx->data), expert_weights->data,
                     static_cast<int*>(num_tokens_per_expert->data),
                     static_cast<int*>(cta_idx_xy_to_batch_idx->data),
                     static_cast<int*>(cta_idx_xy_to_mn_limit->data),
                     static_cast<int*>(num_non_exiting_ctas->data), args.mDtypeElt,
                     false /* use_routing_scales_on_input */, false /* use_deep_seek_fp8 */,
                     static_cast<RoutingMethodType>(routing_method_type), stream);

  //
  // FC13 (gemm1) + FC2 (gemm2)
  //

  if (dtype_act == btg::Dtype::E2m1) {
    TVM_FFI_ICHECK_EQ(hidden_states->dtype, dl_uint8) << "hidden_states must be byte.";
  } else if (dtype_act == btg::Dtype::E4m3 || dtype_act == btg::Dtype::MxE4m3) {
    TVM_FFI_ICHECK_EQ(hidden_states->dtype, dl_float8_e4m3fn) << "hidden_states must be fp8.";
  } else if (dtype_act == btg::Dtype::Bfloat16) {
    TVM_FFI_ICHECK_EQ(hidden_states->dtype, dl_bfloat16) << "hidden_states must be bfloat16.";
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported act dtype.";
  }

  if (hidden_states_scale.has_value()) {
    TVM_FFI_ICHECK_EQ(hidden_states_scale.value()->dtype, dl_float8_e4m3fn)
        << "hidden_states_scale must be fp8.";

    TVM_FFI_ICHECK_EQ(
        get_numel(hidden_states_scale.value()),
        tensorrt_llm::computeLinearLayoutSFSize(args.num_tokens, args.hidden_size / sf_vec_size))
        << "hidden_states_scale has incorrect size";
  }

  TVM_FFI_ICHECK_EQ(gemm1_weights->dtype, dl_uint8) << "gemm1_weights must be byte.";

  TVM_FFI_ICHECK_EQ(gemm1_weights->ndim, 3) << "gemm1_weights must be 3D.";
  TVM_FFI_ICHECK_EQ(gemm1_weights->shape[1] % 2, 0)
      << "the second dimension of weights must be even.";
  TVM_FFI_ICHECK_EQ(intermediate_size, gemm1_weights->shape[1] / 2)
      << "intermediate_size has incorrect dim 1.";
  // This check passes even though the actual shape of the weights[2] and hidden_states[1] is
  // 2 times larger due to the fact that 2 e2m1 are packed into 1 byte.
  TVM_FFI_ICHECK_EQ(
      gemm1_weights->shape[2],
      (dtype_act == btg::Dtype::E2m1 ? hidden_states->shape[1] : hidden_states->shape[1] / 2))
      << "the third dimension of weights must be equal to hidden_size.";

  TVM_FFI_ICHECK_EQ(gemm1_weights_scale->dtype, dl_float8_e4m3fn)
      << "gemm1_weights_scale must be fp8.";

  TVM_FFI_ICHECK_EQ(gemm1_weights_scale->ndim, 3) << "gemm1_weights_scale must be 3D.";
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale->shape[0], local_num_experts)
      << "gemm1_weights_scale has incorrect dim 0.";
  TVM_FFI_ICHECK_EQ(intermediate_size % sf_vec_size, 0)
      << "the second dimension of weights must be a multiple of ",
      sf_vec_size;
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale->shape[1], 2 * intermediate_size)
      << "gemm1_weights_scale has incorrect dim 1.";
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale->shape[2], args.hidden_size / sf_vec_size)
      << "gemm1_weights_scale has incorrect dim 2.";

  if (gemm1_bias.has_value()) {
    TVM_FFI_ICHECK_EQ(gemm1_bias.value()->dtype, dl_float32)
        << "gemm1_bias must be float, got "
        << tvm::ffi::DLDataTypeToString(gemm1_bias.value()->dtype);
    TVM_FFI_ICHECK_EQ(gemm1_bias.value()->ndim, 2) << "gemm1_bias must be 2D.";
    TVM_FFI_ICHECK_EQ(gemm1_bias.value()->shape[0], local_num_experts)
        << "gemm1_bias has incorrect dim 0.";
    TVM_FFI_ICHECK_EQ(gemm1_bias.value()->shape[1], 2 * intermediate_size)
        << "gemm1_bias has incorrect dim 1.";
  }

  if (gemm1_alpha.has_value()) {
    TVM_FFI_ICHECK_EQ(gemm1_alpha.value()->dtype, dl_float32)
        << "gemm1_alpha must be float, got "
        << tvm::ffi::DLDataTypeToString(gemm1_alpha.value()->dtype);
    TVM_FFI_ICHECK_EQ(gemm1_alpha.value()->ndim, 1) << "gemm1_alpha must be 1D.";
    TVM_FFI_ICHECK_EQ(gemm1_alpha.value()->shape[0], local_num_experts)
        << "gemm1_alpha has incorrect dim 0.";
  }
  if (gemm1_beta.has_value()) {
    TVM_FFI_ICHECK_EQ(gemm1_beta.value()->dtype, dl_float32)
        << "gemm1_beta must be float, got "
        << tvm::ffi::DLDataTypeToString(gemm1_beta.value()->dtype);
    TVM_FFI_ICHECK_EQ(gemm1_beta.value()->ndim, 1) << "gemm1_beta must be 1D.";
    TVM_FFI_ICHECK_EQ(gemm1_beta.value()->shape[0], local_num_experts)
        << "gemm1_beta has incorrect dim 0.";
  }

  TVM_FFI_ICHECK_EQ(gemm2_weights->dtype, dl_uint8) << "gemm2_weights must be byte.";

  TVM_FFI_ICHECK_EQ(gemm2_weights->ndim, 3) << "gemm2_weights must be 3D.";
  // / 2 to compensate for the fact that we pack 2 e2m1 into 1 byte.
  TVM_FFI_ICHECK_EQ(gemm2_weights->shape[2], intermediate_size / 2)
      << "the third dimension of weights must be equal to intermediate_size.";

  TVM_FFI_ICHECK_EQ(gemm2_weights_scale->dtype, dl_float8_e4m3fn)
      << "gemm2_weights_scale must be fp8.";

  TVM_FFI_ICHECK_EQ(gemm2_weights_scale->ndim, 3) << "gemm2_weights_scale must be 3D.";
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale->shape[0], local_num_experts)
      << "gemm2_weights_scale has incorrect dim 0.";
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale->shape[1], args.hidden_size)
      << "gemm2_weights_scale has incorrect dim 1.";
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale->shape[2], intermediate_size / sf_vec_size)
      << "gemm2_weights_scale has incorrect dim 2.";

  if (output1_scales_scalar.has_value()) {
    TVM_FFI_ICHECK_EQ(output1_scales_scalar.value()->dtype, dl_float32)
        << "output1_scales_scalar must be float.";
    TVM_FFI_ICHECK_EQ(output1_scales_scalar.value()->ndim, 1)
        << "output1_scales_scalar must be 1D.";
    TVM_FFI_ICHECK_EQ(output1_scales_scalar.value()->shape[0], local_num_experts)
        << "output1_scales_scalar has incorrect dim 0.";
  }

  if (output1_scales_gate_scalar.has_value()) {
    TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar.value()->dtype, dl_float32)
        << "output1_scales_gate_scalar must be float.";
    TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar.value()->ndim, 1)
        << "output1_scales_gate_scalar must be 1D.";
    TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar.value()->shape[0], local_num_experts)
        << "output1_scales_gate_scalar has incorrect dim 0.";
  }

  if (output2_scales_scalar.has_value()) {
    TVM_FFI_ICHECK_EQ(output2_scales_scalar.value()->dtype, dl_float32)
        << "output2_scales_scalar must be float.";
    TVM_FFI_ICHECK_EQ(output2_scales_scalar.value()->ndim, 1)
        << "output2_scales_scalar must be 1D.";
    TVM_FFI_ICHECK_EQ(output2_scales_scalar.value()->shape[0], local_num_experts)
        << "output2_scales_scalar has incorrect dim 0.";
  }

  // setup workspace
  workspace.total_num_padded_tokens = static_cast<int*>(total_num_padded_tokens->data);
  workspace.total_max_padded_tokens = max_num_padded_tokens;
  workspace.ProjUpTileN = tile_tokens_dim;
  workspace.routing_expert_indexes = static_cast<int*>(expert_indices->data);
  workspace.permuted_idx_size = static_cast<int*>(total_num_padded_tokens->data);
  workspace.expanded_idx_to_permuted_idx =
      static_cast<int*>(expanded_idx_to_permuted_idx->data);  // Needed by permute/finalize kernels
  workspace.permuted_idx_to_token_idx =
      static_cast<int*>(permuted_idx_to_token_idx->data);  // Needed by permuteGemm1 kernel
  workspace.expert_weights = expert_weights->data;         // Consumed by finalize kernel

  workspace.cta_idx_xy_to_batch_idx = static_cast<int*>(cta_idx_xy_to_batch_idx->data);
  workspace.cta_idx_xy_to_mn_limit = static_cast<int*>(cta_idx_xy_to_mn_limit->data);
  workspace.num_non_exiting_ctas = static_cast<int*>(num_non_exiting_ctas->data);

  workspace.hidden_states_scale_linear = nullptr;

  // gemm1 intermediate ws
  workspace.gemm1_output = gemm1_output->data;
  workspace.gemm1_output_scale = gemm1_output_scale.has_value()
                                     ? static_cast<float*>(gemm1_output_scale.value()->data)
                                     : nullptr;

  // gemm2 intermediate ws
  workspace.gemm2_output = gemm2_output->data;
  workspace.gemm2_output_scale = nullptr;
  args.output = output->data;
  args.output_scale = nullptr;
  args.output1_scales_scalar = output1_scales_scalar.has_value()
                                   ? static_cast<float*>(output1_scales_scalar.value()->data)
                                   : nullptr;
  args.output1_scales_gate_scalar =
      output1_scales_gate_scalar.has_value()
          ? static_cast<float*>(output1_scales_gate_scalar.value()->data)
          : nullptr;
  args.output2_scales_scalar = output2_scales_scalar.has_value()
                                   ? static_cast<float*>(output2_scales_scalar.value()->data)
                                   : nullptr;
  args.do_finalize = do_finalize;

  auto const workspace_sizes = moe_runner.getWorkspaceSizeInBytes(args, moeConfigIndex);

  Tensor workspace_fc1 =
      alloc_tensor({std::get<0>(workspace_sizes)}, dl_int8, hidden_states->device);
  Tensor workspace_fc2 =
      alloc_tensor({std::get<1>(workspace_sizes)}, dl_int8, hidden_states->device);
  workspace.bmm1_workspace = workspace_fc1->data;
  workspace.bmm2_workspace = workspace_fc2->data;
  cudaStream_t moe_stream = get_stream(hidden_states->device);
  moe_runner.run(args, workspace, hidden_states->device.device_id, moe_stream, moeConfigIndex,
                 enable_pdl);

  if (!do_finalize) {
    return {gemm2_output, expert_weights, expanded_idx_to_permuted_idx};
  }
  return {output};
}

Array<Tensor> trtllm_fp4_block_scale_moe(
    Optional<Tensor> routing_logits, Tensor topk_ids, Tensor expert_weights,
    Optional<Tensor> routing_bias, Tensor hidden_states, Optional<Tensor> hidden_states_scale,
    Tensor gemm1_weights, Tensor gemm1_weights_scale, Optional<Tensor> gemm1_bias,
    Optional<Tensor> gemm1_alpha, Optional<Tensor> gemm1_beta, Optional<Tensor> gemm1_clamp_limit,
    Tensor gemm2_weights, Tensor gemm2_weights_scale, Optional<Tensor> gemm2_bias,
    Optional<Tensor> output1_scales_scalar, Optional<Tensor> output1_scales_gate_scalar,
    Optional<Tensor> output2_scales_scalar, int64_t num_experts, int64_t top_k,
    Optional<int64_t> n_group, Optional<int64_t> topk_group, int64_t intermediate_size,
    int64_t local_expert_offset, int64_t local_num_experts, Optional<double> routed_scaling_factor,
    int64_t tile_tokens_dim, int64_t routing_method_type, bool do_finalize, bool enable_pdl,
    int64_t gated_act_type, Tensor output, int64_t config_index) {
  using RunnerType = tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner;

  int const num_tokens = hidden_states->shape[0];
  int hidden_size = hidden_states->shape[1];
  if (hidden_states->dtype == dl_uint8) hidden_size *= 2;
  int hidden_states_scale_vec_size = -1;
  if (hidden_states_scale.has_value()) {
    hidden_states_scale_vec_size =
        (num_tokens * hidden_size) / get_numel(hidden_states_scale.value());
  }
  int weight_scale_vec_size =
      (local_num_experts * intermediate_size * 2 * hidden_size) / get_numel(gemm1_weights_scale);
  TVM_FFI_ICHECK(weight_scale_vec_size == 16 || weight_scale_vec_size == 32)
      << "unsupported weight_scale_vec_size.";
  auto mDtypeWeights = weight_scale_vec_size == 16 ? btg::Dtype::E2m1 : btg::Dtype::MxE2m1;

  TVM_FFI_ICHECK(gemm1_weights->dtype == dl_uint8 && gemm2_weights->dtype == dl_uint8)
      << "weights must be fp4 packed in uint8.";
  TVM_FFI_ICHECK(hidden_states->dtype == dl_uint8 || hidden_states->dtype == dl_bfloat16 ||
                 hidden_states->dtype == dl_float8_e4m3fn)
      << "hidden_states must be bf16, fp8 or uint8 (packed fp4).";
  auto mDtypeAct = btg::Dtype::Bfloat16;
  if (hidden_states->dtype == dl_uint8) {
    TVM_FFI_ICHECK(hidden_states_scale.has_value() &&
                   hidden_states_scale.value()->dtype == dl_float8_e4m3fn)
        << "hidden_states_scale must be provided for fp4 activation.";
    if (hidden_states_scale_vec_size == 16) {
      mDtypeAct = btg::Dtype::E2m1;
    } else if (hidden_states_scale_vec_size == 32) {
      mDtypeAct = btg::Dtype::MxE2m1;
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported hidden state scale shape.";
    }
  } else if (hidden_states->dtype == dl_float8_e4m3fn) {
    if (hidden_states_scale.has_value()) {
      if (hidden_states_scale_vec_size == 32) {
        mDtypeAct = btg::Dtype::MxE4m3;
      } else {
        TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported hidden state scale shape.";
      }
    } else {
      mDtypeAct = btg::Dtype::E4m3;
    }
  }
  bool mUseDeepSeekFp8{false};  // FP4 doesn't use DeepSeek FP8

  // Properly initialize the runner using make_unique like in the original code
  auto mRunner = std::make_unique<RunnerType>(
      mDtypeAct, mDtypeWeights, mUseDeepSeekFp8, (int32_t)tile_tokens_dim,
      static_cast<GatedActType>(gated_act_type), /*useShuffledMatrixA*/ true);

  if (config_index == -1) {
    config_index = mRunner->getDefaultValidConfigIndex(top_k, hidden_size, intermediate_size,
                                                       local_num_experts, num_tokens);
  }

  return trtllm_fp4_block_scale_moe_launcher(
      routing_logits, topk_ids, expert_weights, routing_bias, hidden_states, hidden_states_scale,
      gemm1_weights, gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta, gemm1_clamp_limit,
      gemm2_weights, gemm2_weights_scale, gemm2_bias, output1_scales_scalar,
      output1_scales_gate_scalar, output2_scales_scalar, num_experts, top_k, n_group, topk_group,
      intermediate_size, local_expert_offset, local_num_experts, routed_scaling_factor,
      tile_tokens_dim, routing_method_type, do_finalize, *mRunner, mDtypeAct, mDtypeWeights,
      config_index, enable_pdl, output);
}

int64_t trtllm_get_default_moe_configs(int64_t const tile_tokens_dim, int64_t const dtype_act_,
                                       int64_t const dtype_weights_, bool const useDeepSeekFp8,
                                       int64_t const top_k, int64_t const hidden_size,
                                       int64_t const intermediate_size,
                                       int64_t const num_local_experts,
                                       int64_t const gated_act_type, int64_t const num_tokens) {
  auto dtype_act = static_cast<btg::Dtype>(dtype_act_);
  auto dtype_weights = static_cast<btg::Dtype>(dtype_weights_);
  tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner moe_runner(
      dtype_act, dtype_weights, useDeepSeekFp8, (int32_t)tile_tokens_dim,
      static_cast<GatedActType>(gated_act_type), /*useShuffledMatrixA*/ true);
  return moe_runner.getDefaultValidConfigIndex(top_k, hidden_size, intermediate_size,
                                               num_local_experts, num_tokens);
}

Array<int64_t> trtllm_get_valid_moe_configs(int64_t const tile_tokens_dim, int64_t const dtype_act_,
                                            int64_t const dtype_weights_, bool const useDeepSeekFp8,
                                            int64_t const top_k, int64_t const hidden_size,
                                            int64_t const intermediate_size,
                                            int64_t const num_local_experts,
                                            int64_t const gated_act_type,
                                            int64_t const num_tokens) {
  auto dtype_act = static_cast<btg::Dtype>(dtype_act_);
  auto dtype_weights = static_cast<btg::Dtype>(dtype_weights_);
  tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner moe_runner(
      dtype_act, dtype_weights, useDeepSeekFp8, (int32_t)tile_tokens_dim,
      static_cast<GatedActType>(gated_act_type), /*useShuffledMatrixA*/ true);
  return moe_runner.getValidConfigIndices(top_k, hidden_size, intermediate_size, num_local_experts,
                                          num_tokens);
}

namespace trtllm_cubin_loader {
#include <flashinfer/cubin_loader.h>
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_bf16_moe, trtllm_bf16_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp8_per_tensor_scale_moe, trtllm_fp8_per_tensor_scale_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp8_block_scale_moe, trtllm_fp8_block_scale_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp4_block_scale_moe, trtllm_fp4_block_scale_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_get_default_moe_configs, trtllm_get_default_moe_configs);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_get_valid_moe_configs, trtllm_get_valid_moe_configs);

}  // namespace flashinfer
