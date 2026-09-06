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
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <flashinfer/exception.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "flashinfer/fused_moe/da_moe.cuh"
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
using tensorrt_llm::kernels::trtllmgen_moe::MoE::ActivationType;
using tensorrt_llm::kernels::trtllmgen_moe::Routing::RoutingMethodType;
using tvm::ffi::Array;
using tvm::ffi::Optional;

enum class RoutingInputMode {
  FromLogits,          // Mode 1: Compute routing from logits
  PackedPrecomputed,   // Mode 2: Pre-computed with packed (score << 16 | id) format
  UnpackedPrecomputed  // Mode 3: Pre-computed with separate topk_ids and topk_weights
};

/** Typed storage for one tile of graph-stable routing metadata. */
struct RoutingMetadataBuffers {
  // Stable FFI width.
  // Sync with flashinfer/fused_moe/core.py:TrtllmMoERoutingMetadataSlot.tensors.
  static constexpr int64_t kNumTensors = 9;
  // FFI[0] live padded size.
  // Sync with flashinfer/fused_moe/core.py:TrtllmMoERoutingMetadataSlot.total_num_padded_tokens.
  Tensor total_num_padded_tokens;
  // FFI[1] expanded-to-permuted map.
  // Sync with
  // flashinfer/fused_moe/core.py:TrtllmMoERoutingMetadataSlot.expanded_idx_to_permuted_idx.
  Tensor expanded_idx_to_permuted_idx;
  // FFI[2] permuted-to-token map.
  // Sync with flashinfer/fused_moe/core.py:TrtllmMoERoutingMetadataSlot.permuted_idx_to_token_idx.
  Tensor permuted_idx_to_token_idx;
  // FFI[3] routing weights.
  // Sync with flashinfer/fused_moe/core.py:TrtllmMoERoutingMetadataSlot.expert_weights.
  Tensor expert_weights;
  // FFI[4] histogram scratch.
  // Sync with flashinfer/fused_moe/core.py:TrtllmMoERoutingMetadataSlot.expert_count_histogram.
  Tensor expert_count_histogram;
  // FFI[5] expert counts.
  // Sync with flashinfer/fused_moe/core.py:TrtllmMoERoutingMetadataSlot.num_tokens_per_expert.
  Tensor num_tokens_per_expert;
  // FFI[6] CTA batch map.
  // Sync with flashinfer/fused_moe/core.py:TrtllmMoERoutingMetadataSlot.cta_idx_xy_to_batch_idx.
  Tensor cta_idx_xy_to_batch_idx;
  // FFI[7] CTA limits.
  // Sync with flashinfer/fused_moe/core.py:TrtllmMoERoutingMetadataSlot.cta_idx_xy_to_mn_limit.
  Tensor cta_idx_xy_to_mn_limit;
  // FFI[8] live CTA count.
  // Sync with flashinfer/fused_moe/core.py:TrtllmMoERoutingMetadataSlot.num_non_exiting_ctas.
  Tensor num_non_exiting_ctas;

  /** Decode one exact stable nine-tensor routing ABI. */
  static RoutingMetadataBuffers from_ffi(Array<Tensor> const& tensors) {
    TVM_FFI_ICHECK_EQ(tensors.size(), kNumTensors)
        << "Routing metadata requires exactly nine tensors.";
    return from_flat_ffi(tensors, 0);
  }

  /** Decode one routing record from a flattened multi-tile FFI array. */
  static RoutingMetadataBuffers from_flat_ffi(Array<Tensor> const& tensors, int64_t offset) {
    TVM_FFI_ICHECK_GE(offset, 0) << "Routing metadata offset must be nonnegative.";
    TVM_FFI_ICHECK_GE(tensors.size(), offset + kNumTensors)
        << "Routing metadata requires nine tensors at the requested offset.";
    return {tensors[offset],     tensors[offset + 1], tensors[offset + 2],
            tensors[offset + 3], tensors[offset + 4], tensors[offset + 5],
            tensors[offset + 6], tensors[offset + 7], tensors[offset + 8]};
  }

  /** Encode one typed routing record in the existing public FFI order. */
  Array<Tensor> to_ffi() const {
    return {total_num_padded_tokens, expanded_idx_to_permuted_idx, permuted_idx_to_token_idx,
            expert_weights,          expert_count_histogram,       num_tokens_per_expert,
            cta_idx_xy_to_batch_idx, cta_idx_xy_to_mn_limit,       num_non_exiting_ctas};
  }

  /** Bind this routing record to a concrete MoE workspace and tile geometry. */
  template <typename Workspace>
  void bind(Workspace& workspace, int64_t tile_tokens_dim) const {
    workspace.total_num_padded_tokens = static_cast<int32_t*>(total_num_padded_tokens.data_ptr());
    workspace.permuted_idx_size = workspace.total_num_padded_tokens;
    workspace.total_max_padded_tokens = static_cast<int32_t>(permuted_idx_to_token_idx.size(0) - 1);
    workspace.ProjUpTileN = static_cast<int32_t>(tile_tokens_dim);
    workspace.expanded_idx_to_permuted_idx =
        static_cast<int32_t*>(expanded_idx_to_permuted_idx.data_ptr());
    workspace.permuted_idx_to_token_idx =
        static_cast<int32_t*>(permuted_idx_to_token_idx.data_ptr());
    workspace.permuted_idx_to_expanded_idx = nullptr;
    workspace.expert_weights = expert_weights.data_ptr();
    workspace.cta_idx_xy_to_batch_idx = static_cast<int32_t*>(cta_idx_xy_to_batch_idx.data_ptr());
    workspace.cta_idx_xy_to_mn_limit = static_cast<int32_t*>(cta_idx_xy_to_mn_limit.data_ptr());
    workspace.num_non_exiting_ctas = static_cast<int32_t*>(num_non_exiting_ctas.data_ptr());
  }
};

/** Typed storage used to canonicalize live routing logits. */
struct CanonicalRoutingBuffers {
  // Stable FFI width.
  // Sync with flashinfer/fused_moe/core.py:TRTLLMCanonicalRouting.tensors.
  static constexpr int64_t kNumTensors = 11;
  // FFI[0] native int16 replay IDs produced by the ordinary router.
  // Sync with flashinfer/fused_moe/core.py:TRTLLMCanonicalRouting.routing_replay_ids.
  Tensor routing_replay_ids;
  // FFI[1] canonical weights.
  // Sync with flashinfer/fused_moe/core.py:TRTLLMCanonicalRouting.expert_weights.
  Tensor expert_weights;
  // FFI[2] conventional packed scratch required by the ordinary router ABI.
  // Sync with flashinfer/fused_moe/core.py:TRTLLMCanonicalRouting.packed_router_scratch.
  Tensor packed_scratch;
  // FFI[3] expert counts.
  // Sync with flashinfer/fused_moe/core.py:TRTLLMCanonicalRouting.scratch.
  Tensor num_tokens_per_expert;
  // FFI[4] padded size.
  // Sync with flashinfer/fused_moe/core.py:TRTLLMCanonicalRouting.scratch.
  Tensor total_num_padded_tokens;
  // FFI[5] expanded-to-permuted map.
  // Sync with flashinfer/fused_moe/core.py:TRTLLMCanonicalRouting.scratch.
  Tensor expanded_idx_to_permuted_idx;
  // FFI[6] permuted-to-token map.
  // Sync with flashinfer/fused_moe/core.py:TRTLLMCanonicalRouting.scratch.
  Tensor permuted_idx_to_token_idx;
  // FFI[7] histogram scratch.
  // Sync with flashinfer/fused_moe/core.py:TRTLLMCanonicalRouting.scratch.
  Tensor expert_count_histogram;
  // FFI[8] CTA batch map.
  // Sync with flashinfer/fused_moe/core.py:TRTLLMCanonicalRouting.scratch.
  Tensor cta_idx_xy_to_batch_idx;
  // FFI[9] CTA limits.
  // Sync with flashinfer/fused_moe/core.py:TRTLLMCanonicalRouting.scratch.
  Tensor cta_idx_xy_to_mn_limit;
  // FFI[10] live CTA count.
  // Sync with flashinfer/fused_moe/core.py:TRTLLMCanonicalRouting.scratch.
  Tensor num_non_exiting_ctas;

  /** Decode the stable canonical-routing FFI tensor order. */
  static CanonicalRoutingBuffers from_ffi(Array<Tensor> const& tensors) {
    TVM_FFI_ICHECK_EQ(tensors.size(), kNumTensors)
        << "Canonical routing requires eleven stable tensors.";
    return {tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5],
            tensors[6], tensors[7], tensors[8], tensors[9], tensors[10]};
  }

  /** Encode canonical-routing storage in its existing public FFI order. */
  Array<Tensor> to_ffi() const {
    return {routing_replay_ids,        expert_weights,          packed_scratch,
            num_tokens_per_expert,     total_num_padded_tokens, expanded_idx_to_permuted_idx,
            permuted_idx_to_token_idx, expert_count_histogram,  cta_idx_xy_to_batch_idx,
            cta_idx_xy_to_mn_limit,    num_non_exiting_ctas};
  }
};

/** Typed graph-stable buffers for the BF16 body ABI. */
struct BF16DABodyBuffers {
  // Stable BF16 FFI width.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  static constexpr int64_t kNumTensors = 4;
  // FFI[0] FC1 output.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor gemm1_output;
  // FFI[1] FC2 output.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor gemm2_output;
  // FFI[2] FC1 scratch.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor workspace_fc1;
  // FFI[3] FC2 scratch.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor workspace_fc2;

  /** Decode a BF16 body from the public FFI order. */
  static BF16DABodyBuffers from_ffi(Array<Tensor> const& tensors) {
    TVM_FFI_ICHECK_EQ(tensors.size(), kNumTensors)
        << "A BF16 DA body requires four prepared tensors.";
    return {tensors[0], tensors[1], tensors[2], tensors[3]};
  }

  /** Encode a BF16 body in the public FFI order. */
  Array<Tensor> to_ffi() const {
    return {gemm1_output, gemm2_output, workspace_fc1, workspace_fc2};
  }

  /** Bind BF16 body buffers to the BF16 runner workspace ABI. */
  template <typename Workspace>
  void bind(Workspace& workspace) const {
    workspace.hidden_states_scale_linear = nullptr;
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = nullptr;
    workspace.activation_output = nullptr;
    workspace.activation_output_scale = nullptr;
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;
    workspace.bmm1_workspace = workspace_fc1.data_ptr();
    workspace.bmm2_workspace = workspace_fc2.data_ptr();
  }
};

/** Typed graph-stable buffers for the FP8 per-tensor body ABI. */
struct FP8PerTensorDABodyBuffers {
  // Stable FP8-per-tensor FFI width.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  static constexpr int64_t kNumTensors = 5;
  // FFI[0] FC1 output.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor gemm1_output;
  // FFI[1] FC1 scale.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor gemm1_output_scale;
  // FFI[2] FC2 output.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor gemm2_output;
  // FFI[3] FC1 scratch.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor workspace_fc1;
  // FFI[4] FC2 scratch.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor workspace_fc2;

  /** Decode an FP8 per-tensor body from the public FFI order. */
  static FP8PerTensorDABodyBuffers from_ffi(Array<Tensor> const& tensors) {
    TVM_FFI_ICHECK_EQ(tensors.size(), kNumTensors)
        << "An FP8 per-tensor DA body requires five prepared tensors.";
    return {tensors[0], tensors[1], tensors[2], tensors[3], tensors[4]};
  }

  /** Encode an FP8 per-tensor body in the public FFI order. */
  Array<Tensor> to_ffi() const {
    return {gemm1_output, gemm1_output_scale, gemm2_output, workspace_fc1, workspace_fc2};
  }

  /** Bind FP8 per-tensor buffers to their concrete runner workspace ABI. */
  template <typename Workspace>
  void bind(Workspace& workspace) const {
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = static_cast<float*>(gemm1_output_scale.data_ptr());
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.bmm1_workspace = workspace_fc1.data_ptr();
    workspace.bmm2_workspace = workspace_fc2.data_ptr();
  }
};

/** Typed graph-stable buffers for the DeepSeek FP8 block-scale body ABI. */
struct DeepSeekFP8DABodyBuffers {
  // Stable DeepSeek-FP8 FFI width.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  static constexpr int64_t kNumTensors = 7;
  // FFI[0] FC1 output.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor gemm1_output;
  // FFI[1] FC1 scale.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor gemm1_output_scale;
  // FFI[2] activation output.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor activation_output;
  // FFI[3] activation scale.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor activation_output_scale;
  // FFI[4] FC2 output.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor gemm2_output;
  // FFI[5] FC1 scratch.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor workspace_fc1;
  // FFI[6] FC2 scratch.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor workspace_fc2;

  /** Decode a DeepSeek FP8 body from the public FFI order. */
  static DeepSeekFP8DABodyBuffers from_ffi(Array<Tensor> const& tensors) {
    TVM_FFI_ICHECK_EQ(tensors.size(), kNumTensors)
        << "A DeepSeek FP8 DA body requires seven prepared tensors.";
    return {tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], tensors[6]};
  }

  /** Encode a DeepSeek FP8 body in the public FFI order. */
  Array<Tensor> to_ffi() const {
    return {gemm1_output, gemm1_output_scale, activation_output, activation_output_scale,
            gemm2_output, workspace_fc1,      workspace_fc2};
  }

  /** Bind DeepSeek FP8 buffers to the DeepSeek block-scale runner ABI. */
  template <typename Workspace>
  void bind(Workspace& workspace) const {
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = static_cast<float*>(gemm1_output_scale.data_ptr());
    workspace.activation_output = activation_output.data_ptr();
    workspace.activation_output_scale = static_cast<float*>(activation_output_scale.data_ptr());
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.bmm1_workspace = workspace_fc1.data_ptr();
    workspace.bmm2_workspace = workspace_fc2.data_ptr();
  }
};

/** Typed graph-stable buffers for the MXFP8 block-scale body ABI. */
struct MXFP8DABodyBuffers {
  // Stable MXFP8 FFI width.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  static constexpr int64_t kNumTensors = 5;
  // FFI[0] FC1 output.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor gemm1_output;
  // FFI[1] FC1 scale.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor gemm1_output_scale;
  // FFI[2] FC2 output.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor gemm2_output;
  // FFI[3] FC1 scratch.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor workspace_fc1;
  // FFI[4] FC2 scratch.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor workspace_fc2;

  /** Decode an MXFP8 body from the public FFI order. */
  static MXFP8DABodyBuffers from_ffi(Array<Tensor> const& tensors) {
    TVM_FFI_ICHECK_EQ(tensors.size(), kNumTensors)
        << "An MXFP8 DA body requires five prepared tensors.";
    return {tensors[0], tensors[1], tensors[2], tensors[3], tensors[4]};
  }

  /** Encode an MXFP8 body in the public FFI order. */
  Array<Tensor> to_ffi() const {
    return {gemm1_output, gemm1_output_scale, gemm2_output, workspace_fc1, workspace_fc2};
  }

  /** Bind MXFP8 buffers to the MXFP8 block-scale runner ABI. */
  template <typename Workspace>
  void bind(Workspace& workspace) const {
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = static_cast<float*>(gemm1_output_scale.data_ptr());
    workspace.activation_output = nullptr;
    workspace.activation_output_scale = nullptr;
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.bmm1_workspace = workspace_fc1.data_ptr();
    workspace.bmm2_workspace = workspace_fc2.data_ptr();
  }
};

/** Typed graph-stable buffers for the FP4 body ABI. */
struct FP4DABodyBuffers {
  // Stable FP4 FFI width.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  static constexpr int64_t kNumTensors = 7;
  // FFI[0] FC1 output.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor gemm1_output;
  // FFI[1] FC1 scale.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor gemm1_output_scale;
  // FFI[2] activation output.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor activation_output;
  // FFI[3] FC2 token scales.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor per_token_scales_fc2;
  // FFI[4] FC2 output.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor gemm2_output;
  // FFI[5] FC1 scratch.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor workspace_fc1;
  // FFI[6] FC2 scratch.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor workspace_fc2;

  /** Decode an FP4 body from the public FFI order. */
  static FP4DABodyBuffers from_ffi(Array<Tensor> const& tensors) {
    TVM_FFI_ICHECK_EQ(tensors.size(), kNumTensors)
        << "An FP4 DA body requires seven prepared tensors.";
    return {tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], tensors[6]};
  }

  /** Encode an FP4 body in the public FFI order. */
  Array<Tensor> to_ffi() const {
    return {gemm1_output, gemm1_output_scale, activation_output, per_token_scales_fc2,
            gemm2_output, workspace_fc1,      workspace_fc2};
  }

  /** Bind FP4 buffers and optional live token scales to the FP4 runner ABI. */
  template <typename Workspace>
  void bind(Workspace& workspace, Optional<TensorView> const& per_token_scales) const {
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = gemm1_output_scale.numel() == 0
                                       ? nullptr
                                       : static_cast<float*>(gemm1_output_scale.data_ptr());
    workspace.activation_output =
        activation_output.numel() == 0 ? nullptr : activation_output.data_ptr();
    workspace.activation_output_scale = workspace.gemm1_output_scale;
    workspace.token_scales =
        per_token_scales.has_value() ? per_token_scales.value().data_ptr() : nullptr;
    workspace.token_scales_fc2 =
        per_token_scales_fc2.numel() == 0 ? nullptr : per_token_scales_fc2.data_ptr();
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;
    workspace.hidden_states_scale_linear = nullptr;
    workspace.bmm1_workspace = workspace_fc1.data_ptr();
    workspace.bmm2_workspace = workspace_fc2.data_ptr();
  }
};

/** Typed graph-stable buffers for the MXINT4 body ABI. */
struct MXINT4DABodyBuffers {
  // Stable MXINT4 FFI width.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  static constexpr int64_t kNumTensors = 4;
  // FFI[0] FC1 output.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor gemm1_output;
  // FFI[1] FC2 output.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor gemm2_output;
  // FFI[2] FC1 scratch.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor workspace_fc1;
  // FFI[3] FC2 scratch.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaBodyWorkspace.tensors.
  Tensor workspace_fc2;

  /** Decode an MXINT4 body from the public FFI order. */
  static MXINT4DABodyBuffers from_ffi(Array<Tensor> const& tensors) {
    TVM_FFI_ICHECK_EQ(tensors.size(), kNumTensors)
        << "An MXINT4 DA body requires four prepared tensors.";
    return {tensors[0], tensors[1], tensors[2], tensors[3]};
  }

  /** Encode an MXINT4 body in the public FFI order. */
  Array<Tensor> to_ffi() const {
    return {gemm1_output, gemm2_output, workspace_fc1, workspace_fc2};
  }

  /** Bind MXINT4 buffers to the MXINT4 runner workspace ABI. */
  template <typename Workspace>
  void bind(Workspace& workspace) const {
    workspace.hidden_states_scale_linear = nullptr;
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = nullptr;
    workspace.activation_output = nullptr;
    workspace.activation_output_scale = nullptr;
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;
    workspace.bmm1_workspace = workspace_fc1.data_ptr();
    workspace.bmm2_workspace = workspace_fc2.data_ptr();
  }
};

/** Typed result of one ordinary MoE launch before TVM-FFI result encoding. */
struct MoeRunResultBuffers {
  // FFI[0] final or FC2 output.
  // Sync with flashinfer/fused_moe/core.py:_unpack_trtllm_moe_output.
  Tensor primary_output;
  // FFI[1] routing weights; an undefined tensor preserves the borrowed-buffer slot.
  // Sync with flashinfer/fused_moe/core.py:_unpack_trtllm_moe_output.
  Tensor expert_weights;
  // Optional next FFI item, layout map.
  // Sync with flashinfer/fused_moe/core.py:_unpack_trtllm_moe_output.
  Optional<Tensor> expanded_to_permuted_indices;
  // Optional activation output.
  // Sync with flashinfer/fused_moe/core.py:_unpack_trtllm_moe_output.
  Optional<Tensor> activation_output;
  // Whether FFI[0] is finalized output rather than an unfinalized FC2 buffer.
  // Sync with flashinfer/fused_moe/core.py:_unpack_trtllm_moe_output.
  bool is_finalized{true};

  /** Construct a result whose stable FFI layout is fixed by its output mode. */
  MoeRunResultBuffers(bool finalized, Tensor primary)
      : primary_output(std::move(primary)), is_finalized(finalized) {}

  /** Encode the typed launch result in the existing public FFI tensor order. */
  Array<Tensor> to_ffi() const {
    Array<Tensor> tensors{primary_output};
    // Unfinalized calls always reserve FFI[1] for expert weights. Some launchers borrow the
    // caller's buffer, so the undefined tensor is intentional and Python substitutes its input.
    if (!is_finalized) {
      tensors.push_back(expert_weights);
    }
    if (expanded_to_permuted_indices.has_value()) {
      tensors.push_back(expanded_to_permuted_indices.value());
    }
    if (activation_output.has_value()) {
      tensors.push_back(activation_output.value());
    }
    return tensors;
  }
};

/** Typed CUDA-owned state carried across the two TVM-FFI SWITCH capture calls. */
struct DASwitchCaptureState {
  // Native FFI capture-ID slot.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaSwitchCaptureState.native.
  static constexpr int64_t kCaptureIdIndex = 0;
  // Native FFI SWITCH-node slot.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaSwitchCaptureState.native.
  static constexpr int64_t kConditionalNodeIndex = 1;
  // Native FFI preamble-node slot.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaSwitchCaptureState.native.
  static constexpr int64_t kParallelWorkNodeIndex = 2;
  // Native FFI selector-node slot.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaSwitchCaptureState.native.
  static constexpr int64_t kSelectorNodeIndex = 3;
  // Native FFI body-count slot.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaSwitchCaptureState.BODY_COUNT_INDEX.
  static constexpr int64_t kBodyCountIndex = 4;
  // Native FFI fixed header width.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaSwitchCaptureState.HEADER_SIZE.
  static constexpr int64_t kHeaderSize = 5;
  // Minimum SWITCH fanout.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaSwitchCaptureState.MINIMUM_BODY_COUNT.
  static constexpr int64_t kMinimumBodyCount = 2;

  // Native FFI[0] capture generation.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaSwitchCaptureState.native.
  unsigned long long capture_id;
  // Native FFI[1] installed SWITCH node.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaSwitchCaptureState.native.
  cudaGraphNode_t conditional_node;
  // Native FFI[2] live preamble node.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaSwitchCaptureState.native.
  cudaGraphNode_t parallel_work_node;
  // Native FFI[3] device selector node.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaSwitchCaptureState.native.
  cudaGraphNode_t selector_node;
  // Native FFI[5:] child graphs.
  // Sync with flashinfer/fused_moe/core.py:TrtllmDaSwitchCaptureState.body_graph_handles.
  std::vector<cudaGraph_t> body_graphs;

  /** Decode and validate the stable integer-handle FFI representation. */
  static DASwitchCaptureState from_ffi(Array<int64_t> const& state) {
    TVM_FFI_ICHECK_GE(state.size(), kHeaderSize + kMinimumBodyCount)
        << "DA SWITCH capture state is incomplete.";
    int64_t const body_count = state[kBodyCountIndex];
    TVM_FFI_ICHECK_GE(body_count, kMinimumBodyCount)
        << "DA SWITCH capture state requires at least two body graphs.";
    TVM_FFI_ICHECK_LE(body_count, da_moe::kDAMaxBodies)
        << "DA SWITCH capture state exceeds immutable body capacity.";
    TVM_FFI_ICHECK_EQ(state.size(), body_count + kHeaderSize)
        << "DA SWITCH body graph state has an invalid length.";

    std::vector<cudaGraph_t> body_graphs;
    body_graphs.reserve(body_count);
    for (int64_t body_index = 0; body_index < body_count; ++body_index) {
      body_graphs.push_back(reinterpret_cast<cudaGraph_t>(state[kHeaderSize + body_index]));
    }
    return {static_cast<unsigned long long>(state[kCaptureIdIndex]),
            reinterpret_cast<cudaGraphNode_t>(state[kConditionalNodeIndex]),
            reinterpret_cast<cudaGraphNode_t>(state[kParallelWorkNodeIndex]),
            reinterpret_cast<cudaGraphNode_t>(state[kSelectorNodeIndex]), std::move(body_graphs)};
  }

  /** Encode named CUDA graph handles in the stable public FFI order. */
  Array<int64_t> to_ffi() const {
    Array<int64_t> state{
        static_cast<int64_t>(capture_id), reinterpret_cast<int64_t>(conditional_node),
        reinterpret_cast<int64_t>(parallel_work_node), reinterpret_cast<int64_t>(selector_node),
        static_cast<int64_t>(body_graphs.size())};
    for (auto body_graph : body_graphs) {
      state.push_back(reinterpret_cast<int64_t>(body_graph));
    }
    return state;
  }
};

/// Return log2(value) for powers of two and -1 for arbitrary positive tile sizes.
inline int32_t computeRoutingLog2(int64_t value) {
  int64_t n = value;
  int32_t out = 0;
  while (n >>= 1) {
    ++out;
  }
  return (int64_t{1} << out) == value ? out : -1;
}

/// Validate the public, graph-stable input contract shared by allocation and population.
template <typename OptionalWeights>
inline RoutingInputMode validateMultiTileRoutingInputs(
    TensorView const& topk_ids, int64_t num_experts, int64_t top_k, int64_t local_expert_offset,
    int64_t local_num_experts, Array<int64_t> const& tile_tokens_dims, int64_t routing_input_mode,
    OptionalWeights const& topk_weights) {
  // First constrain immutable tile capacity and the caller-visible routing tensor geometry.
  TVM_FFI_ICHECK_GT(tile_tokens_dims.size(), 0) << "tile_tokens_dims must be non-empty.";
  TVM_FFI_ICHECK_LE(tile_tokens_dims.size(),
                    moe::dev::routing::routingPrecomputed::kMaxRoutingMetadataTiles)
      << "tile_tokens_dims exceeds the compiled fused routing capacity.";
  for (int64_t tile_tokens_dim : tile_tokens_dims) {
    TVM_FFI_ICHECK_GT(tile_tokens_dim, 0) << "tile_tokens_dims entries must be positive.";
  }
  TVM_FFI_ICHECK_EQ(topk_ids.device().device_type, kDLCUDA) << "topk_ids must be a CUDA tensor.";
  TVM_FFI_ICHECK_EQ(topk_ids.ndim(), 2) << "topk_ids must be 2D.";
  TVM_FFI_ICHECK(topk_ids.IsContiguous()) << "topk_ids must be contiguous.";
  TVM_FFI_ICHECK_GT(topk_ids.size(0), 0) << "topk_ids must contain at least one token.";
  TVM_FFI_ICHECK_EQ(topk_ids.size(1), top_k) << "topk_ids dim1 must match top_k.";
  TVM_FFI_ICHECK_GT(top_k, 0) << "top_k must be positive.";
  TVM_FFI_ICHECK_LE(top_k, 8) << "fused multi-tile routing supports top_k <= 8.";
  TVM_FFI_ICHECK_GE(num_experts, top_k) << "num_experts must be at least top_k.";
  TVM_FFI_ICHECK_LE(num_experts, da_moe::kDAMaxExperts)
      << "fused multi-tile routing supports num_experts <= " << da_moe::kDAMaxExperts << ".";
  TVM_FFI_ICHECK(local_num_experts > 0 && local_num_experts <= num_experts)
      << "local_num_experts must be between 1 and num_experts.";
  TVM_FFI_ICHECK(local_expert_offset >= 0 && local_expert_offset + local_num_experts <= num_experts)
      << "the local expert range must lie within num_experts.";
  TVM_FFI_ICHECK_LE(topk_ids.size(0),
                    moe::dev::routing::routingPrecomputed::maxTokensMultiTileCluster(num_experts))
      << "the token count exceeds the fused multi-tile cluster topology.";

  // Decode the routing representation once, then validate only the corresponding weight ABI.
  auto const input_mode = static_cast<RoutingInputMode>(routing_input_mode);
  TVM_FFI_ICHECK(input_mode == RoutingInputMode::PackedPrecomputed ||
                 input_mode == RoutingInputMode::UnpackedPrecomputed)
      << "multi-tile routing requires packed or unpacked precomputed routing.";
  if (input_mode == RoutingInputMode::UnpackedPrecomputed) {
    TVM_FFI_ICHECK(topk_ids.dtype() == dl_int16 || topk_ids.dtype() == dl_int32)
        << "unpacked topk_ids must be int16 or int32.";
    TVM_FFI_ICHECK(topk_weights.has_value())
        << "unpacked precomputed routing requires topk_weights.";
    auto const& weights = topk_weights.value();
    TVM_FFI_ICHECK(weights.dtype() == dl_bfloat16 || weights.dtype() == dl_float32)
        << "topk_weights must be bfloat16 or float32.";
    TVM_FFI_ICHECK_EQ(weights.ndim(), 2) << "topk_weights must be 2D.";
    TVM_FFI_ICHECK_EQ(weights.size(0), topk_ids.size(0))
        << "topk_weights dim0 must match topk_ids.";
    TVM_FFI_ICHECK_EQ(weights.size(1), top_k) << "topk_weights dim1 must match top_k.";
    TVM_FFI_ICHECK(weights.IsContiguous()) << "topk_weights must be contiguous.";
    TVM_FFI_ICHECK_EQ(weights.device().device_type, kDLCUDA)
        << "topk_weights must be a CUDA tensor.";
    TVM_FFI_ICHECK_EQ(weights.device().device_id, topk_ids.device().device_id)
        << "topk_weights and topk_ids must be on the same device.";
  } else {
    TVM_FFI_ICHECK_EQ(topk_ids.dtype(), dl_int32) << "packed topk_ids must be int32.";
    TVM_FFI_ICHECK(!topk_weights.has_value())
        << "packed precomputed routing carries weights inside topk_ids.";
  }

  // The clustered preamble is admitted only on the architecture family it was compiled for.
  int major = 0;
  int minor = 0;
  CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                          topk_ids.device().device_id));
  CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                          topk_ids.device().device_id));
  TVM_FFI_ICHECK_EQ(major, 10) << "fused multi-tile routing requires SM10.x; got SM" << major
                               << minor << ".";
  return input_mode;
}

/// Validate one tile's metadata storage against its exact routing geometry.
inline void validateRoutingMetadata(RoutingMetadataBuffers const& metadata, int64_t num_tokens,
                                    int64_t top_k, int64_t num_experts, int64_t tile_tokens_dim,
                                    DLDevice device, DLDataType expert_weights_dtype) {
  // Recompute exact extent bounds from the body tile so a mismatched FFI record fails early.
  int32_t const max_num_padded_tokens =
      tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxPermutedPaddedCount(
          num_tokens, top_k, num_experts, tile_tokens_dim);
  int32_t const max_num_ctas =
      tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxNumCtasInBatchDim(
          num_tokens, top_k, num_experts, tile_tokens_dim);
  int64_t const histogram_size = std::max<int64_t>(num_experts * 2, 256 * 2);

  auto check_1d = [device](Tensor const& tensor, DLDataType dtype, int64_t min_size,
                           char const* name) {
    TVM_FFI_ICHECK_EQ(tensor.ndim(), 1) << name << " must be 1D.";
    TVM_FFI_ICHECK_GE(tensor.size(0), min_size) << name << " is too small.";
    TVM_FFI_ICHECK_EQ(tensor.dtype(), dtype) << name << " has incorrect dtype.";
    TVM_FFI_ICHECK_EQ(tensor.device().device_type, device.device_type)
        << name << " is on the wrong device type.";
    TVM_FFI_ICHECK_EQ(tensor.device().device_id, device.device_id)
        << name << " is on the wrong device.";
    TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous.";
  };
  check_1d(metadata.total_num_padded_tokens, dl_int32, 1, "total_num_padded_tokens");
  check_1d(metadata.expanded_idx_to_permuted_idx, dl_int32, num_tokens * top_k,
           "expanded_idx_to_permuted_idx");
  check_1d(metadata.permuted_idx_to_token_idx, dl_int32, max_num_padded_tokens + 1,
           "permuted_idx_to_token_idx");
  check_1d(metadata.expert_count_histogram, dl_int32, histogram_size, "expert_count_histogram");
  check_1d(metadata.num_tokens_per_expert, dl_int32, num_experts, "num_tokens_per_expert");
  check_1d(metadata.cta_idx_xy_to_batch_idx, dl_int32, max_num_ctas, "cta_idx_xy_to_batch_idx");
  check_1d(metadata.cta_idx_xy_to_mn_limit, dl_int32, max_num_ctas, "cta_idx_xy_to_mn_limit");
  check_1d(metadata.num_non_exiting_ctas, dl_int32, 1, "num_non_exiting_ctas");

  // Weight storage is the only two-dimensional field and preserves the caller's numeric dtype.
  Tensor const& expert_weights = metadata.expert_weights;
  TVM_FFI_ICHECK_EQ(expert_weights.ndim(), 2) << "expert_weights must be 2D.";
  TVM_FFI_ICHECK_EQ(expert_weights.size(0), num_tokens)
      << "expert_weights dim0 must match num_tokens.";
  TVM_FFI_ICHECK_EQ(expert_weights.size(1), top_k) << "expert_weights dim1 must match top_k.";
  TVM_FFI_ICHECK_EQ(expert_weights.dtype(), expert_weights_dtype)
      << "expert_weights has incorrect dtype.";
  TVM_FFI_ICHECK_EQ(expert_weights.device().device_type, device.device_type)
      << "expert_weights is on the wrong device type.";
  TVM_FFI_ICHECK_EQ(expert_weights.device().device_id, device.device_id)
      << "expert_weights is on the wrong device.";
  TVM_FFI_ICHECK(expert_weights.IsContiguous()) << "expert_weights must be contiguous.";
}

/// Build the framework-independent descriptor consumed by the fused routing kernel.
inline moe::dev::routing::routingPrecomputed::Data makePrecomputedRoutingData(
    TensorView const& topk_ids, RoutingInputMode input_mode, int64_t num_experts, int64_t top_k,
    int64_t local_expert_offset, int64_t local_num_experts, int64_t tile_tokens_dim,
    RoutingMetadataBuffers const& metadata) {
  // Bind the representation-specific input pointer while keeping every output tile-local.
  moe::dev::routing::routingPrecomputed::Data data;
  data.mDtypeOutput =
      metadata.expert_weights.dtype() == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;
  data.mUsePdl = false;
  data.mPtrTopKPacked = input_mode == RoutingInputMode::PackedPrecomputed
                            ? const_cast<void*>(topk_ids.data_ptr())
                            : nullptr;
  data.mPtrTopKIds = nullptr;
  data.mPtrPrecomputedExpertIds =
      input_mode == RoutingInputMode::UnpackedPrecomputed ? topk_ids.data_ptr() : nullptr;
  data.mExpertIdType = input_mode == RoutingInputMode::PackedPrecomputed
                           ? moe::dev::routing::routingPrecomputed::ExpertIdType::Packed
                       : topk_ids.dtype() == dl_int16
                           ? moe::dev::routing::routingPrecomputed::ExpertIdType::Int16
                           : moe::dev::routing::routingPrecomputed::ExpertIdType::Int32;
  data.mPtrExpertCounts = static_cast<int32_t*>(metadata.expert_count_histogram.data_ptr());
  data.mPtrPermutedIdxSize = static_cast<int32_t*>(metadata.total_num_padded_tokens.data_ptr());
  data.mPtrExpandedIdxToPermutedIdx =
      static_cast<int32_t*>(metadata.expanded_idx_to_permuted_idx.data_ptr());
  data.mPtrPermutedIdxToExpandedIdx = nullptr;
  data.mPtrPermutedIdxToTokenIdx =
      static_cast<int32_t*>(metadata.permuted_idx_to_token_idx.data_ptr());
  data.mPtrTopKWeights = metadata.expert_weights.data_ptr();
  data.mPtrCtaIdxXyToBatchIdx = static_cast<int32_t*>(metadata.cta_idx_xy_to_batch_idx.data_ptr());
  data.mPtrCtaIdxXyToMnLimit = static_cast<int32_t*>(metadata.cta_idx_xy_to_mn_limit.data_ptr());
  data.mPtrNumNonExitingCtas = static_cast<int32_t*>(metadata.num_non_exiting_ctas.data_ptr());
  data.mPtrNumTokensPerExpert = static_cast<int32_t*>(metadata.num_tokens_per_expert.data_ptr());
  data.mPtrScores = nullptr;
  // Copy shape and expert-partition values that must agree across all fused tile descriptors.
  data.mNumTokens = topk_ids.size(0);
  data.mNumExperts = num_experts;
  data.mNumFusedSharedExperts = 0;
  data.mTopK = top_k;
  data.mTotalExpertsPerToken = top_k;
  data.mPaddingLog2 = computeRoutingLog2(tile_tokens_dim);
  data.mTileTokensDim = tile_tokens_dim;
  data.mLocalExpertsStartIdx = local_expert_offset;
  data.mLocalExpertsStrideLog2 = 0;
  data.mNumLocalExperts = local_num_experts;
  return data;
}

/** Decode the flattened public multi-tile routing ABI into typed records once. */
inline std::vector<RoutingMetadataBuffers> routingMetadataFromFfi(
    Array<Tensor> const& flat_routing_metadata, int64_t num_tiles) {
  TVM_FFI_ICHECK_EQ(flat_routing_metadata.size(), num_tiles * RoutingMetadataBuffers::kNumTensors)
      << "Flat routing metadata must contain nine tensors per tile.";
  std::vector<RoutingMetadataBuffers> records;
  records.reserve(num_tiles);
  for (int64_t tile_index = 0; tile_index < num_tiles; ++tile_index) {
    records.push_back(RoutingMetadataBuffers::from_flat_ffi(
        flat_routing_metadata, tile_index * RoutingMetadataBuffers::kNumTensors));
  }
  return records;
}

/** Encode typed multi-tile routing records into the existing flattened FFI ABI. */
inline Array<Tensor> routingMetadataToFfi(
    std::vector<RoutingMetadataBuffers> const& routing_metadata) {
  Array<Tensor> flat;
  for (RoutingMetadataBuffers const& record : routing_metadata) {
    for (Tensor const& tensor : record.to_ffi()) {
      flat.push_back(tensor);
    }
  }
  return flat;
}

/// Validate and materialize host launch descriptors for all prepared routing tiles.
inline std::vector<moe::dev::routing::routingPrecomputed::Data> makeMultiTileRoutingData(
    TensorView topk_ids, int64_t num_experts, int64_t top_k, int64_t local_expert_offset,
    int64_t local_num_experts, Array<int64_t> const& tile_tokens_dims,
    std::vector<RoutingMetadataBuffers> const& routing_metadata, RoutingInputMode input_mode,
    Optional<TensorView> const& topk_weights, cudaStream_t stream) {
  TVM_FFI_ICHECK_EQ(routing_metadata.size(), tile_tokens_dims.size())
      << "Routing metadata must contain one typed record per tile.";
  int64_t const num_tokens = topk_ids.size(0);
  DLDataType const expert_weights_dtype = input_mode == RoutingInputMode::UnpackedPrecomputed
                                              ? topk_weights.value().dtype()
                                              : dl_bfloat16;

  // Validate and bind each typed record independently because output extents depend on tile-N.
  std::vector<moe::dev::routing::routingPrecomputed::Data> routing_data;
  routing_data.reserve(tile_tokens_dims.size());
  for (int64_t tile_index = 0; tile_index < tile_tokens_dims.size(); ++tile_index) {
    int64_t const tile_tokens_dim = tile_tokens_dims[tile_index];
    RoutingMetadataBuffers const& metadata = routing_metadata[tile_index];
    validateRoutingMetadata(metadata, num_tokens, top_k, num_experts, tile_tokens_dim,
                            topk_ids.device(), expert_weights_dtype);

    // Preserve stable metadata addresses while refreshing caller-owned unpacked weights in place.
    if (input_mode == RoutingInputMode::UnpackedPrecomputed &&
        metadata.expert_weights.data_ptr() != topk_weights.value().data_ptr()) {
      auto const& weights = topk_weights.value();
      size_t const num_bytes = static_cast<size_t>(weights.numel()) * weights.dtype().bits / 8;
      CHECK_CUDA_ERROR(cudaMemcpyAsync(metadata.expert_weights.data_ptr(), weights.data_ptr(),
                                       num_bytes, cudaMemcpyDeviceToDevice, stream));
    }
    routing_data.push_back(makePrecomputedRoutingData(topk_ids, input_mode, num_experts, top_k,
                                                      local_expert_offset, local_num_experts,
                                                      tile_tokens_dim, metadata));
  }
  return routing_data;
}
namespace {

__global__ void cast_fp32_to_bf16_kernel(__nv_bfloat16* output, float const* input,
                                         int64_t num_elements) {
  int64_t const index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < num_elements) {
    output[index] = __float2bfloat16_rn(input[index]);
  }
}

void cast_fp32_to_bf16(void* output, void const* input, int64_t num_elements, cudaStream_t stream) {
  constexpr int32_t block_size = 256;
  int32_t const grid_size = static_cast<int32_t>((num_elements + block_size - 1) / block_size);
  cast_fp32_to_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
      static_cast<__nv_bfloat16*>(output), static_cast<float const*>(input), num_elements);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "Failed to convert routing scales to BF16.";
}

}  // namespace

// Validate routing_replay_out tensor properties.
// NOTE: dim0 >= num_tokens is intentionally NOT checked — with CUDA graphs the buffer
// is pre-allocated at maximum batch size and reused across steps with varying num_tokens.
static void validate_routing_replay_out(TensorView const& replay, TensorView const& hidden_states,
                                        int64_t top_k) {
  TVM_FFI_ICHECK(replay.device().device_type == kDLCUDA)
      << "routing_replay_out must be a CUDA tensor";
  TVM_FFI_ICHECK(replay.device().device_id == hidden_states.device().device_id)
      << "routing_replay_out must be on the same device as hidden_states";
  TVM_FFI_ICHECK(replay.ndim() == 2) << "routing_replay_out must be 2D [num_tokens, top_k]";
  TVM_FFI_ICHECK(replay.size(1) == top_k) << "routing_replay_out dim1 must equal top_k";
  TVM_FFI_ICHECK((replay.dtype() == DLDataType{kDLInt, 16, 1}))
      << "routing_replay_out must be int16 dtype";
  TVM_FFI_ICHECK(replay.IsContiguous())
      << "routing_replay_out must be contiguous (packed row-major)";
}

enum class Fp8QuantizationType {
  NoneFp8,
  DeepSeekFp8,
  MxFp8,
  PerTensorFp8,
  PerChannelFp8,
};

inline std::string fp8QuantizationTypeToString(Fp8QuantizationType quantization_type) {
  switch (quantization_type) {
    default:
    case Fp8QuantizationType::NoneFp8:
      return "NoneFp8";
    case Fp8QuantizationType::DeepSeekFp8:
      return "DeepSeekFp8";
    case Fp8QuantizationType::MxFp8:
      return "MxFp8";
    case Fp8QuantizationType::PerTensorFp8:
      return "PerTensorFp8";
    case Fp8QuantizationType::PerChannelFp8:
      return "PerChannelFp8";
  }
}

inline ActivationType validateAndCastActivationType(int64_t act_type) {
  TVM_FFI_ICHECK(act_type >= 0 && act_type < static_cast<int64_t>(ActivationType::InvalidType))
      << "Invalid activation type: " << act_type;
  return static_cast<ActivationType>(act_type);
}

inline bool hasOptionalGemm1ActivationParams(Optional<TensorView> const& gemm1_alpha,
                                             Optional<TensorView> const& gemm1_beta,
                                             Optional<TensorView> const& gemm1_clamp_limit) {
  return gemm1_alpha.has_value() || gemm1_beta.has_value() || gemm1_clamp_limit.has_value();
}

// MxFp8 applies these in the fused FC1 epilogue of the trtllm-gen cubins; DeepSeekFp8 has no
// fused activation and applies them in the separate activation kernel
// (moe::dev::activation::run). Both consume the values as-is: FP8 block scaling carries no
// scalar dequant factor, so no host-side rescaling of the limit is needed.
inline void validateFp8BlockScaleGemm1ActivationParams(
    Optional<TensorView> const& gemm1_alpha, Optional<TensorView> const& gemm1_beta,
    Optional<TensorView> const& gemm1_clamp_limit, Fp8QuantizationType quantization_type,
    ActivationType activation_type) {
  if (!hasOptionalGemm1ActivationParams(gemm1_alpha, gemm1_beta, gemm1_clamp_limit)) {
    return;
  }
  TVM_FFI_ICHECK(quantization_type == Fp8QuantizationType::MxFp8 ||
                 quantization_type == Fp8QuantizationType::DeepSeekFp8)
      << "gemm1_alpha, gemm1_beta, and gemm1_clamp_limit are only supported for "
         "Fp8QuantizationType::MxFp8 and Fp8QuantizationType::DeepSeekFp8 in FP8 block scale "
         "MoE, got "
      << fp8QuantizationTypeToString(quantization_type) << ".";
  TVM_FFI_ICHECK(activation_type == ActivationType::Swiglu)
      << "gemm1_alpha, gemm1_beta, and gemm1_clamp_limit are only supported for "
         "ActivationType::Swiglu.";
}

// Utility function to compute the next power of two
inline int32_t nextPowerOfTwo(float value) {
  int32_t n = static_cast<int32_t>(std::ceil(value));
  if (n <= 1) return 1;

  // If n is already a power of 2, return it
  if ((n & (n - 1)) == 0) return n;

  // Find the next power of 2
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;

  return n;
}

std::set<int32_t> computeSelectedTileN(std::vector<int32_t> const& supported_tile_nums,
                                       int64_t const num_tokens, int64_t const top_k,
                                       int64_t const num_local_experts) {
  TVM_FFI_ICHECK(!supported_tile_nums.empty()) << "supported_tile_nums must not be empty.";
  float const avg_tokens_per_expert = static_cast<float>(num_tokens * top_k) / num_local_experts;
  // NOTE: This differs from Python AutoTuner bucketing:
  // - AutoTuner maps raw num_tokens with last_positive_power_of_2 (round-down).
  // - Here we map derived avg_tokens_per_expert and use nextPowerOfTwo (round-up).
  // Because they round different quantities in different directions, cache bucket and runtime
  // tile candidates can diverge; launcher-side tactic resolution handles that mismatch.
  // assume supported_tile_nums is sorted
  int32_t tile_tokens_dim = std::clamp(nextPowerOfTwo(avg_tokens_per_expert),
                                       supported_tile_nums.front(), supported_tile_nums.back());
  auto it = std::find(supported_tile_nums.begin(), supported_tile_nums.end(), tile_tokens_dim);
  FLASHINFER_CHECK(
      it != supported_tile_nums.end(), "computeSelectedTileN expected exact tile ", tile_tokens_dim,
      " in supported_tile_nums (size=", supported_tile_nums.size(),
      "). Please keep supported_tile_nums as a dense power-of-2 ladder for this launcher.");

  // Candidate tile set centered on the heuristic tile.
  // This function returns nearby candidates (not a single final tile):
  //   center, +1, +2, and -1 neighbors when available.
  // Final tile choice is made later (autotuner-provided tile if valid, otherwise fallback policy).
  std::set<int32_t> selected_tile_nums;
  selected_tile_nums.insert(tile_tokens_dim);
  if (std::next(it) != supported_tile_nums.end()) {
    selected_tile_nums.insert(*std::next(it));
    if (std::next(std::next(it)) != supported_tile_nums.end()) {
      selected_tile_nums.insert(*std::next(std::next(it)));
    }
  }
  if (it != supported_tile_nums.begin()) {
    selected_tile_nums.insert(*std::prev(it));
  }

  return selected_tile_nums;
}

int64_t selectDefaultTileN(std::vector<int32_t> const& supported_tile_nums,
                           int64_t const num_tokens, int64_t const top_k,
                           int64_t const num_local_experts) {
  auto selected = computeSelectedTileN(supported_tile_nums, num_tokens, top_k, num_local_experts);
  TVM_FFI_ICHECK(!selected.empty()) << "No selected tile_N candidates for current MoE input.";
  return *selected.begin();
}

// Resolve the (tile_N, config) pair passed from Python side, applying fallback logic
// when tile_N is -1.
std::pair<int64_t, int64_t> resolveMoeTileAndConfig(Array<int64_t> const& config_index,
                                                    std::vector<int32_t> const& supported_tile_nums,
                                                    int64_t const num_tokens, int64_t const top_k,
                                                    int64_t const num_local_experts) {
  // Python side convention: tactic is [tile_N, config]
  TVM_FFI_ICHECK(config_index.size() == 2)
      << "Invalid tactic, expected to be [tile_N, config], but got array of size "
      << config_index.size();
  const int64_t tile_N = config_index[0];
  const int64_t config = config_index[1];

  if (tile_N == -1 || config == -1) {
    // Use fallback tactic
    auto const default_tile_N =
        selectDefaultTileN(supported_tile_nums, num_tokens, top_k, num_local_experts);
    return {default_tile_N, -1};
  }

  return {tile_N, config};
}

// Validate the FC1 bias tensor against the selected BiasType. Currently only
// BiasType::None and BiasType::Mn are exercised from the flashinfer MoE path
inline void check_gemm1_bias_mn(Optional<TensorView> const& gemm1_bias,
                                batchedGemm::gemm::BiasType bias_type, int32_t num_tokens,
                                int32_t top_k, int32_t intermediate_size) {
  if (bias_type == batchedGemm::gemm::BiasType::None) {
    TVM_FFI_ICHECK(!gemm1_bias.has_value())
        << "gemm1_bias is provided when gemm1_bias_type is None";
    return;
  }
  TVM_FFI_ICHECK(bias_type == batchedGemm::gemm::BiasType::Mn)
      << "flashinfer MoE only supports gemm1_bias_type in {None, Mn}; got "
      << static_cast<int64_t>(bias_type);
  TVM_FFI_ICHECK(gemm1_bias.has_value())
      << "gemm1_bias must be provided when gemm1_bias_type is Mn";
  auto const& bias = gemm1_bias.value();
  TVM_FFI_ICHECK_EQ(bias.dtype(), dl_bfloat16) << "gemm1_bias must be bfloat16.";
  TVM_FFI_ICHECK_EQ(bias.ndim(), 3)
      << "gemm1_bias must have shape [num_tokens, top_k, 2 * intermediate_size].";
  TVM_FFI_ICHECK_EQ(bias.size(0), num_tokens)
      << "gemm1_bias must have shape [num_tokens, top_k, 2 * intermediate_size].";
  TVM_FFI_ICHECK_EQ(bias.size(1), top_k)
      << "gemm1_bias must have shape [num_tokens, top_k, 2 * intermediate_size].";
  TVM_FFI_ICHECK_EQ(bias.size(2), 2 * intermediate_size)
      << "gemm1_bias must have shape [num_tokens, top_k, 2 * intermediate_size].";
}

class FusedMoeLauncher {
 protected:
  Optional<TensorView> routing_logits;
  Optional<TensorView> routing_bias;
  TensorView hidden_states;
  TensorView gemm1_weights;
  Optional<TensorView> gemm1_bias;
  Optional<TensorView> output1_scales_scalar;
  Optional<TensorView> output1_scales_gate_scalar;
  TensorView gemm2_weights;
  Optional<TensorView> output2_scales_scalar;
  Optional<TensorView> per_token_scales;
  Tensor per_token_scales_fc2;
  bool use_per_channel_scaling_gemm1{false};
  bool use_per_channel_scaling_gemm2{false};

  int64_t tile_tokens_dim{};
  int64_t routing_method_type{};
  bool use_shuffled_weight{};
  batchedGemm::gemm::MatrixLayout weight_layout{batchedGemm::gemm::MatrixLayout::MajorK};
  batchedGemm::gemm::BiasType gemm1_bias_type{batchedGemm::gemm::BiasType::None};

  std::tuple<int, int> device_version;
  std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs> args;
  tensorrt_llm::kernels::trtllmgen_moe::MoE::MoEWorkspace workspace;

  btg::Dtype mDtypeAct{btg::Dtype::Bfloat16};
  btg::Dtype mDtypeWeights{btg::Dtype::Bfloat16};
  btg::Dtype mRoutingBiasDtype{
      btg::Dtype::Bfloat16};  // Dtype for expert weights in routing, based on routing bias
  btg::Dtype mRoutingLogitsDtype{btg::Dtype::Bfloat16};
  bool norm_topk_prob{true};
  ActivationType activation_type{ActivationType::Swiglu};

  // Optional routing replay output: [num_tokens, top_k] int16 tensor
  Optional<TensorView> routing_replay_out;

  int64_t intermediate_size_factor{2};

 public:
  // Constructor that initializes all TensorView members
  FusedMoeLauncher(const Optional<TensorView>& routing_logits,
                   const Optional<TensorView>& routing_bias, const TensorView& hidden_states,
                   const TensorView& gemm1_weights, const Optional<TensorView>& gemm1_bias,
                   const Optional<TensorView>& output1_scales_scalar,
                   const Optional<TensorView>& output1_scales_gate_scalar,
                   const TensorView& gemm2_weights,
                   const Optional<TensorView>& output2_scales_scalar,
                   const Optional<TensorView>& per_token_scales,
                   RoutingInputMode routing_input_mode = RoutingInputMode::FromLogits)
      : routing_input_mode_(routing_input_mode),
        routing_logits(routing_logits),
        routing_bias(routing_bias),
        hidden_states(hidden_states),
        gemm1_weights(gemm1_weights),
        gemm1_bias(gemm1_bias),
        output1_scales_scalar(output1_scales_scalar),
        output1_scales_gate_scalar(output1_scales_gate_scalar),
        gemm2_weights(gemm2_weights),
        output2_scales_scalar(output2_scales_scalar),
        per_token_scales(per_token_scales),
        tile_tokens_dim{},
        routing_method_type{},
        use_shuffled_weight{},
        weight_layout{batchedGemm::gemm::MatrixLayout::MajorK},
        mDtypeAct{btg::Dtype::Bfloat16},
        mDtypeWeights{btg::Dtype::Bfloat16},
        activation_type{ActivationType::Swiglu},
        intermediate_size_factor{2} {}

 public:
  void set_routing_replay_out(const Optional<TensorView>& replay_out) {
    routing_replay_out = replay_out;
  }

 protected:
  // Initialize common data necessary for later.
  // May throw exception from TVM_FFI_ICHECK.
  void init_common(std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
                   int64_t tile_tokens_dim, int64_t routing_method_type, bool use_shuffled_weight,
                   int64_t weight_layout, ActivationType activation_type, int64_t gemm1_bias_type,
                   bool norm_topk_prob = true);

  // Routing logits [num_tokens, num_experts]
  void check_routing_logits() const {
    if (routing_logits.has_value()) {
      // Check shape
      TVM_FFI_ICHECK_EQ(routing_logits.value().ndim(), 2) << "routing_logits must be 2D.";
      TVM_FFI_ICHECK_EQ(routing_logits.value().size(0), hidden_states.size(0))
          << "routing_logits and hidden_states must have the same number of tokens.";
      TVM_FFI_ICHECK_EQ(routing_logits.value().size(1), args->num_experts)
          << "routing_logits dim1 must match num_experts.";

      // Check dtype
      TVM_FFI_ICHECK(routing_logits.value().dtype() == dl_float32 ||
                     routing_logits.value().dtype() == dl_bfloat16)
          << "routing_logits must be float or bfloat16.";
    }
  }

  // Empty placeholder tensors may still carry a non-null data_ptr.
  static bool has_precomputed(TensorView const& tensor) {
    return tensor.ndim() == 2 && tensor.size(0) > 0;
  }

  static btg::Dtype expert_weights_dtype(TensorView const& weights) {
    return weights.dtype() == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;
  }

  bool is_unpacked_routing() const {
    return routing_input_mode_ == RoutingInputMode::UnpackedPrecomputed;
  }

  int32_t* unpacked_expert_ids(TensorView const& indices) const {
    return is_unpacked_routing() ? static_cast<int32_t*>(const_cast<void*>(indices.data_ptr()))
                                 : nullptr;
  }

  virtual int32_t* precomputed_expert_ids() const { return nullptr; }

  RoutingInputMode routing_input_mode_;

  // Routing bias [num_experts]
  void check_routing_bias_shape() const {
    if (routing_bias.has_value()) {
      TVM_FFI_ICHECK(routing_bias.value().dtype() == dl_bfloat16 ||
                     routing_bias.value().dtype() == dl_float32)
          << "routing_bias must be bfloat16 or float.";
      TVM_FFI_ICHECK_EQ(routing_bias.value().ndim(), 1) << "routing_bias must be 1D.";
      TVM_FFI_ICHECK_EQ(routing_bias.value().size(0), args->num_experts)
          << "routing_bias has incorrect shape.";
    }
  }

  // Hidden states [num_tokens, hidden_size]
  void check_hidden_states_shape() const {
    TVM_FFI_ICHECK_EQ(hidden_states.ndim(), 2) << "hidden_states must be 2D.";
    TVM_FFI_ICHECK_EQ(hidden_states.size(1), args->intermediate_size)
        << "hidden_states has incorrect shape.";
  }

  // GEMM1 or GEMM2 weights [num_experts, M, K] or [num_experts, K/block_k, M, block_k]
  void check_weights_shape(std::string which_weights) const {
    TensorView weights = (which_weights == "gemm1") ? gemm1_weights : gemm2_weights;
    if (which_weights != "gemm1" && which_weights != "gemm2") {
      TVM_FFI_LOG_AND_THROW(InternalError) << "Internal error: which_weights = " << which_weights;
    }

    int64_t Mn = 0, K = 0;
    if (weight_layout == batchedGemm::gemm::MatrixLayout::MajorK) {
      // MajorK [num_experts, M, K]
      Mn = weights.size(1);
      K = weights.size(2);
    } else if (weight_layout == batchedGemm::gemm::MatrixLayout::BlockMajorK) {
      // BlockMajorK [num_experts, K/block_k, M, block_k]
      Mn = weights.size(2);
      int64_t block_k = weights.size(3);
      K = weights.size(1) * block_k;
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "Unsupported weight_layout: " << (int)weight_layout;
    }
    if (which_weights == "gemm1") {
      // Gated MoE activations (e.g. Swiglu/Geglu) pack gate+up projections in GEMM1,
      // so Mn = 2 * intermediate_size and must be even.
      if (intermediate_size_factor == 2) {
        TVM_FFI_ICHECK_EQ(Mn % 2, 0) << which_weights << " weights Mn dimension must be even.";
      }
      // Non-gated activations (e.g. Relu2) use a single projection in GEMM1,
      // so Mn = intermediate_size. This check covers both gated and non-gated cases.
      TVM_FFI_ICHECK_EQ(args->intermediate_size * intermediate_size_factor, Mn)
          << "intermediate_size has incorrect shape.";
      TVM_FFI_ICHECK_EQ(K, hidden_states.size(1))
          << which_weights << " weights K dimension must be equal to hidden_size.";
    } else if (which_weights == "gemm2") {
      // GEMM2 always consumes the post-activation hidden of size intermediate_size.
      TVM_FFI_ICHECK_EQ(K, args->intermediate_size)
          << which_weights << " weights K dimension must be equal to intermediate_size.";
    }
    if (args->num_fused_shared_experts > 0) {
      TVM_FFI_ICHECK_EQ(weights.size(0), args->local_num_experts + args->num_fused_shared_experts)
          << which_weights
          << " weights dim 0 must be local_num_experts + num_fused_shared_experts.";
    }
  }

  void check_optional_per_expert_float_tensor(Optional<TensorView> const& tensor,
                                              std::string const& tensor_name) const {
    if (!tensor.has_value()) {
      return;
    }
    auto const& value = tensor.value();
    TVM_FFI_ICHECK(value.device().device_type == kDLCUDA)
        << tensor_name << " must be a CUDA tensor.";
    TVM_FFI_ICHECK(value.device().device_id == hidden_states.device().device_id)
        << tensor_name << " must be on the same device as hidden_states.";
    TVM_FFI_ICHECK_EQ(value.dtype(), dl_float32) << tensor_name << " must be float32.";
    TVM_FFI_ICHECK_EQ(value.ndim(), 1) << tensor_name << " must be 1D.";
    // The batched GEMM indexes per-expert tensors by local batch entry, and fused
    // shared experts occupy the rows after the routed local experts, so the tensor
    // must cover local_num_experts + num_fused_shared_experts rows.
    TVM_FFI_ICHECK_EQ(value.size(0), args->local_num_experts + args->num_fused_shared_experts)
        << tensor_name << " must have shape [local_num_experts + num_fused_shared_experts].";
    TVM_FFI_ICHECK(value.IsContiguous()) << tensor_name << " must be contiguous.";
  }

  void check_routing_common() const {
    TVM_FFI_ICHECK(args->top_k > 0 && args->top_k <= args->num_experts)
        << "top_k must be between 1 and num_experts";
    TVM_FFI_ICHECK(args->local_num_experts > 0 && args->local_num_experts <= args->num_experts)
        << "local_num_experts must be between 1 and num_experts";
    TVM_FFI_ICHECK(args->local_expert_offset >= 0 &&
                   args->local_expert_offset + args->local_num_experts <= args->num_experts)
        << "expert offset and count must be within valid range";

    check_routing_logits();

    if (routing_bias.has_value()) {
      check_routing_bias_shape();
    }
  }

  // Routing phase workspace tensors (allocated in prepare_routing() or prepare_routing_common())
  Tensor num_tokens_per_expert;
  Tensor total_num_padded_tokens;
  Tensor expanded_idx_to_permuted_idx;
  Tensor permuted_idx_to_token_idx;
  Tensor permuted_idx_to_expanded_idx;
  // Launcher-owned routing weights, returned by run() on the do_finalize=false
  // rows. Stays empty when a derived launcher borrows a caller-supplied buffer
  // instead, in which case only workspace.expert_weights points at it and the
  // caller must substitute its own buffer for that slot.
  Tensor expert_weights;
  Tensor expert_indexes;
  Tensor expert_count_histogram;
  Tensor cta_idx_xy_to_batch_idx;
  Tensor cta_idx_xy_to_mn_limit;
  Tensor num_non_exiting_ctas;

  void* permuted_idx_to_expanded_idx_ptr() const {
    return permuted_idx_to_expanded_idx.defined() ? permuted_idx_to_expanded_idx.data_ptr()
                                                  : nullptr;
  }

  void prepare_routing_common() {
    int32_t const totalExpertsPerToken = args->top_k + args->num_fused_shared_experts;
    int32_t const totalNumExperts = args->num_experts + args->num_fused_shared_experts;

    // Allocate routing phase workspace tensors
    num_tokens_per_expert = alloc_tensor({totalNumExperts}, dl_int32, hidden_states.device());
    int32_t max_num_padded_tokens =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxPermutedPaddedCount(
            args->num_tokens, totalExpertsPerToken, totalNumExperts, tile_tokens_dim);

    total_num_padded_tokens = alloc_tensor({1}, dl_int32, hidden_states.device());

    expanded_idx_to_permuted_idx =
        alloc_tensor({args->num_tokens * totalExpertsPerToken}, dl_int32, hidden_states.device());

    // WAR: the routed batched-GEMM kernels read one int32 past the end of the route map.
    // TODO: drop the +1 once the fixed kernel cubins land.
    permuted_idx_to_token_idx =
        alloc_tensor({max_num_padded_tokens + 1}, dl_int32, hidden_states.device());

    if (gemm1_bias_type == batchedGemm::gemm::BiasType::Mn) {
      permuted_idx_to_expanded_idx =
          alloc_tensor({max_num_padded_tokens}, dl_int32, hidden_states.device());
    }

    expert_indexes =
        alloc_tensor({args->num_tokens, totalExpertsPerToken}, dl_int32, hidden_states.device());

    // expert_weights allocation should be done by derived class since data type could vary

    int64_t const size_of_expert_count_histogram = std::max(totalNumExperts * 2, 256 * 2);
    expert_count_histogram = alloc_tensor({size_of_expert_count_histogram},
                                          dl_int32,  // 256 is the max number of threads per block
                                                     // and max number of experts
                                          hidden_states.device());

    int32_t max_num_ctas = tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxNumCtasInBatchDim(
        args->num_tokens, totalExpertsPerToken, totalNumExperts, tile_tokens_dim);

    cta_idx_xy_to_batch_idx = alloc_tensor({max_num_ctas}, dl_int32, hidden_states.device());

    cta_idx_xy_to_mn_limit = alloc_tensor({max_num_ctas}, dl_int32, hidden_states.device());

    num_non_exiting_ctas = alloc_tensor({1}, dl_int32, hidden_states.device());

    workspace.total_num_padded_tokens = static_cast<int*>(total_num_padded_tokens.data_ptr());
    workspace.total_max_padded_tokens = max_num_padded_tokens;
    workspace.ProjUpTileN = tile_tokens_dim;
    workspace.routing_expert_indexes = static_cast<int*>(expert_indexes.data_ptr());
    workspace.permuted_idx_size = static_cast<int*>(total_num_padded_tokens.data_ptr());
    workspace.expanded_idx_to_permuted_idx =
        static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr());
    workspace.permuted_idx_to_token_idx = static_cast<int*>(permuted_idx_to_token_idx.data_ptr());
    workspace.permuted_idx_to_expanded_idx = static_cast<int*>(permuted_idx_to_expanded_idx_ptr());
    // workspace.expert_weights will be set by derived class after expert_weights allocation
    workspace.cta_idx_xy_to_batch_idx = static_cast<int*>(cta_idx_xy_to_batch_idx.data_ptr());
    workspace.cta_idx_xy_to_mn_limit = static_cast<int*>(cta_idx_xy_to_mn_limit.data_ptr());
    workspace.num_non_exiting_ctas = static_cast<int*>(num_non_exiting_ctas.data_ptr());
  }

  void check_moe_common() const {
    // Hidden states [num_tokens, hidden_size]
    TVM_FFI_ICHECK_EQ(hidden_states.ndim(), 2) << "hidden_states must be 2D.";
    check_gemm1_bias_mn(gemm1_bias, gemm1_bias_type, args->num_tokens, args->top_k,
                        args->intermediate_size);
  }

  // MoE computation phase workspace tensors (allocated in prepare_moe() or prepare_moe_common())
  Tensor gemm1_output;
  Tensor activation_output;
  Tensor gemm2_output;
  Tensor workspace_fc1;
  Tensor workspace_fc2;
  Tensor output;
  int64_t moe_tactic{-1};
  // Non-owning; points into the thread-local runner cache in prepare_moe_common().
  tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner* moe_runner{nullptr};

  /** Resolve and retain the typed MoE runner for one complete tactic. */
  void prepare_moe_runner(int64_t& moe_tactic) {
    using RunnerType = tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner;
    bool usePerTokenScalingGemm1 = per_token_scales.has_value() || args->mUseRoutingScalesOnInput;
    // FIXME(siyuan): currently only nvfp4 x nvfp4 uses per-token scaling in both FC1 and FC2
    bool usePerTokenScalingGemm2 = per_token_scales.has_value() && mDtypeAct == btg::Dtype::E2m1;
    // For FP8 block-scale (E4m3 activations, E4m3 weights) with DeepSeek FP8 and no
    // gemm1 bias, use the weights-only Runner constructor to match the original kernel
    // path and numerics. DSFp8 + biasMn routes through the unified constructor below
    // (which accepts gemm1_bias_type).
    bool const useWeightsOnlyConstructor =
        this->mDtypeAct == btg::Dtype::E4m3 && this->mDtypeWeights == btg::Dtype::E4m3 &&
        args->mUseDeepSeekFp8 && args->gemm1_bias_type == batchedGemm::gemm::BiasType::None;
    bool const usePerChannelScalingGemm1 = use_per_channel_scaling_gemm1;
    bool const usePerChannelScalingGemm2 = use_per_channel_scaling_gemm2;

    // A Runner contains only constructor-derived kernel metadata and config indices. Reuse it on
    // the same host thread instead of rebuilding and filtering the global config table per call.
    std::tuple const runnerKey{static_cast<int64_t>(this->mDtypeAct),
                               static_cast<int64_t>(this->mDtypeWeights),
                               args->mUseDeepSeekFp8,
                               static_cast<int32_t>(tile_tokens_dim),
                               static_cast<int64_t>(this->activation_type),
                               this->use_shuffled_weight,
                               static_cast<int64_t>(this->weight_layout),
                               static_cast<int64_t>(args->gemm1_bias_type),
                               usePerTokenScalingGemm1,
                               usePerTokenScalingGemm2,
                               usePerChannelScalingGemm1,
                               usePerChannelScalingGemm2,
                               useWeightsOnlyConstructor,
                               std::get<0>(device_version),
                               std::get<1>(device_version),
                               hidden_states.device().device_id};
    using RunnerCacheKey = std::remove_const_t<decltype(runnerKey)>;
    static thread_local std::map<RunnerCacheKey, RunnerType> runnerCache;

    auto runnerIt = runnerCache.find(runnerKey);
    if (runnerIt == runnerCache.end()) {
      if (useWeightsOnlyConstructor) {
        runnerIt =
            runnerCache
                .try_emplace(runnerKey, this->mDtypeWeights, args->mUseDeepSeekFp8,
                             static_cast<int32_t>(tile_tokens_dim), this->use_shuffled_weight,
                             this->weight_layout, usePerTokenScalingGemm1, usePerTokenScalingGemm2,
                             usePerChannelScalingGemm1, usePerChannelScalingGemm2)
                .first;
      } else {
        runnerIt =
            runnerCache
                .try_emplace(runnerKey, this->mDtypeAct, this->mDtypeWeights, args->mUseDeepSeekFp8,
                             static_cast<int32_t>(tile_tokens_dim), this->activation_type,
                             this->use_shuffled_weight, this->weight_layout, args->gemm1_bias_type,
                             usePerTokenScalingGemm1, usePerTokenScalingGemm2,
                             usePerChannelScalingGemm1, usePerChannelScalingGemm2)
                .first;
      }
    }
    moe_runner = &runnerIt->second;

    int32_t const effectiveTopK = args->top_k + args->num_fused_shared_experts;
    int32_t const effectiveLocalExperts = args->local_num_experts + args->num_fused_shared_experts;

    if (moe_tactic == -1) {
      moe_tactic = moe_runner->getDefaultValidConfigIndex(effectiveTopK, args->hidden_size,
                                                          args->intermediate_size,
                                                          effectiveLocalExperts, args->num_tokens);
    }
    FLASHINFER_CHECK(moe_runner->isValidConfigIndex(moe_tactic, effectiveTopK, args->hidden_size,
                                                    args->intermediate_size, effectiveLocalExperts,
                                                    args->num_tokens),
                     "Invalid MoE tactic ", moe_tactic, " for tile_N=", tile_tokens_dim,
                     ". This often indicates a stale or mismatched autotuner cache entry.");
    this->moe_tactic = moe_tactic;
  }

  /** Allocate the FC workspaces required by the resolved complete tactic. */
  void prepare_moe_common(int64_t& moe_tactic) {
    prepare_moe_runner(moe_tactic);
    auto workspace_sizes = moe_runner->getWorkspaceSizeInBytes(*args, moe_tactic);
    workspace_fc1 = alloc_tensor({std::get<0>(workspace_sizes)}, dl_int8, hidden_states.device());
    workspace_fc2 = alloc_tensor({std::get<1>(workspace_sizes)}, dl_int8, hidden_states.device());
    workspace.bmm1_workspace = workspace_fc1.data_ptr();
    workspace.bmm2_workspace = workspace_fc2.data_ptr();
  }

  /** Bind one tile's graph-stable routing metadata to the common MoE workspace. */
  void bind_routing_metadata(RoutingMetadataBuffers const& metadata) {
    metadata.bind(workspace, tile_tokens_dim);
  }

 public:
  virtual void check_routing() const = 0;
  virtual void prepare_routing() = 0;
  virtual void check_moe() const = 0;
  virtual void prepare_moe(int64_t& moe_tactic) = 0;

  // Main entry point for all the executions.
  // Do initializations prior to calling this as the initializations are different for bf16, fp8 and
  // fp4. The executions are non-blocking by default.
  //
  // Return-array layout depending on (do_finalize, return_activation_output):
  //
  // | do_finalize | return_activation_output | Returned tensors                                  |
  // |-------------|--------------------------|---------------------------------------------------|
  // | true  | false | [output]                                                                   |
  // | true  | true  | [output, expanded_idx_to_permuted_idx, gemm1_output]                       |
  // | false | false | [gemm2_output, expert_weights, expanded_idx_to_permuted_idx]               |
  // | false | true  | [gemm2_output, expert_weights, expanded_idx_to_permuted_idx, gemm1_output] |
  //
  // The `gemm1_output` slot carries the post-activation FC1 output with shape
  // [num_padded_tokens, intermediate_size].
  //
  // `expanded_idx_to_permuted_idx` is appended whenever a permuted-layout
  // tensor (`gemm2_output` or `gemm1_output`) is returned, so the caller can
  // always unpermute back to (token, slot) order.
  virtual MoeRunResultBuffers run(int64_t moe_tactic, bool enable_pdl = true,
                                  bool use_routing_scales_on_input = false,
                                  bool use_deep_seek_fp8 = false,
                                  bool return_activation_output = false) {
    ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
    check_routing();
    prepare_routing();

    // Execute routing
    tensorrt_llm::kernels::trtllmgen_moe::Routing::Runner routing_runner(tile_tokens_dim);
    cudaStream_t routing_stream = get_stream(hidden_states.device());

    int32_t* expert_ids_param = precomputed_expert_ids();

    int16_t* replay_ptr = nullptr;
    if (routing_replay_out.has_value()) {
      replay_ptr = reinterpret_cast<int16_t*>(routing_replay_out.value().data_ptr());
    }

    routing_runner.run(
        args->routing_logits, args->routing_bias, args->num_tokens, args->num_experts, args->top_k,
        args->num_fused_shared_experts, args->n_group, args->topk_group, args->local_expert_offset,
        args->local_num_experts, args->routed_scaling_factor, workspace.routing_expert_indexes,
        static_cast<int*>(expert_count_histogram.data_ptr()),
        static_cast<int*>(total_num_padded_tokens.data_ptr()),
        static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr()),
        workspace.permuted_idx_to_expanded_idx,
        static_cast<int*>(permuted_idx_to_token_idx.data_ptr()), expert_ids_param,
        workspace.expert_weights, static_cast<int*>(num_tokens_per_expert.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_batch_idx.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_mn_limit.data_ptr()),
        static_cast<int*>(num_non_exiting_ctas.data_ptr()), args->mDtypeElt, mRoutingBiasDtype,
        use_routing_scales_on_input, use_deep_seek_fp8,
        static_cast<RoutingMethodType>(routing_method_type), routing_stream, mRoutingLogitsDtype,
        norm_topk_prob, replay_ptr, enable_pdl);

    check_moe();
    prepare_moe(moe_tactic);

    cudaStream_t moe_stream = get_stream(hidden_states.device());
    moe_runner->run(*args, workspace, hidden_states.device().device_id, moe_stream, moe_tactic,
                    enable_pdl);

    MoeRunResultBuffers result(args->do_finalize, args->do_finalize ? output : gemm2_output);
    if (!args->do_finalize) {
      result.expert_weights = FusedMoeLauncher::expert_weights;
    }
    // Always surface the permutation map when the caller gets any
    // permuted-layout buffer back, so gemm1/gemm2 outputs can be reordered.
    if (!args->do_finalize || return_activation_output) {
      result.expanded_to_permuted_indices = expanded_idx_to_permuted_idx;
    }
    if (return_activation_output) {
      result.activation_output = gemm1_output;
    }
    return result;
  }
};

void FusedMoeLauncher::init_common(
    std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
    int64_t tile_tokens_dim, int64_t routing_method_type, bool use_shuffled_weight,
    int64_t weight_layout, ActivationType activation_type, int64_t gemm1_bias_type,
    bool norm_topk_prob) {
  // Check devicearchitecture: Blackwell (SM 10.x) required
  auto device = hidden_states.device().device_id;
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  TVM_FFI_ICHECK(major == 10 || major == 12)
      << "MoE kernel requires SM 10.x or SM 12.x architecture. Current device has SM " << major
      << minor;
  this->device_version = std::make_tuple(major, minor);

  args->routing_logits = routing_logits.has_value() ? routing_logits.value().data_ptr() : nullptr;
  args->routing_bias = routing_bias.has_value() ? routing_bias.value().data_ptr() : nullptr;
  args->hidden_states = hidden_states.data_ptr();
  args->gemm1_weights = gemm1_weights.data_ptr();
  args->gemm2_weights = gemm2_weights.data_ptr();
  args->gemm1_bias = gemm1_bias.has_value() ? gemm1_bias.value().data_ptr() : nullptr;
  auto bias_type_enum = static_cast<batchedGemm::gemm::BiasType>(gemm1_bias_type);
  args->gemm1_bias_type = bias_type_enum;

  // Fused shared experts do not yet support expert parallelism (EP).
  //
  // The routing kernel assigns each fused shared expert the global id
  // (num_experts + k) and the permutation pipeline maps a global expert id to a
  // weight-tensor row as (global_id - local_expert_offset) (see
  // include/flashinfer/trtllm/fused_moe/RoutingKernel.cuh). A shared expert
  // therefore lands at its intended local row (local_num_experts + k) only when
  // local_expert_offset + local_num_experts == num_experts, i.e. when every
  // routed expert is local to this rank. Under EP (local_expert_offset > 0, or
  // local_num_experts < num_experts) this silently produces wrong results, so
  // reject it explicitly until the kernel learns to map the shared-expert id
  // range independently of the routed-expert window.
  TVM_FFI_ICHECK(args->num_fused_shared_experts == 0 ||
                 (args->local_expert_offset == 0 && args->local_num_experts == args->num_experts))
      << "Fused shared experts (num_fused_shared_experts > 0) are currently only supported "
         "without expert parallelism, i.e. local_expert_offset == 0 and "
         "local_num_experts == num_experts. Got num_fused_shared_experts="
      << args->num_fused_shared_experts << ", local_expert_offset=" << args->local_expert_offset
      << ", local_num_experts=" << args->local_num_experts << ", num_experts=" << args->num_experts
      << ".";

  this->args = std::move(args);
  this->tile_tokens_dim = tile_tokens_dim;
  this->routing_method_type = routing_method_type;
  this->use_shuffled_weight = use_shuffled_weight;
  TVM_FFI_ICHECK(0 <= weight_layout && weight_layout <= 2)
      << "the value of weight_layout is not recognized";
  this->weight_layout = static_cast<batchedGemm::gemm::MatrixLayout>(weight_layout);
  this->activation_type = activation_type;
  this->intermediate_size_factor = isGatedActivation(activation_type) ? 2 : 1;
  this->norm_topk_prob = norm_topk_prob;
  this->gemm1_bias_type = bias_type_enum;
}

class Bf16MoeLauncher : public FusedMoeLauncher {
 public:
  static constexpr std::array<int32_t, 5> mSupportedTileNums = {8, 16, 32, 64, 128};

  Bf16MoeLauncher(Optional<TensorView> const& routing_logits,
                  Optional<TensorView> const& routing_bias, TensorView const& expert_indices,
                  TensorView const& expert_weights, TensorView const& hidden_states,
                  TensorView const& gemm1_weights, TensorView const& gemm2_weights,
                  Optional<TensorView> const& gemm1_bias, Optional<TensorView> const& gemm1_alpha,
                  Optional<TensorView> const& gemm1_beta,
                  Optional<TensorView> const& gemm1_clamp_limit,
                  RoutingInputMode routing_input_mode)
      : FusedMoeLauncher(routing_logits, routing_bias, hidden_states, gemm1_weights, gemm1_bias,
                         Optional<TensorView>(), Optional<TensorView>(), gemm2_weights,
                         Optional<TensorView>(), Optional<TensorView>(), routing_input_mode),
        expert_indices(expert_indices),
        expert_weights(expert_weights),
        gemm1_alpha(gemm1_alpha),
        gemm1_beta(gemm1_beta),
        gemm1_clamp_limit(gemm1_clamp_limit) {}

  void init(std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
            int64_t tile_tokens_dim, int64_t routing_method_type, bool use_shuffled_weight,
            int64_t weight_layout, ActivationType activation_type, int64_t gemm1_bias_type,
            bool norm_topk_prob = true) {
    FusedMoeLauncher::init_common(std::move(args), tile_tokens_dim, routing_method_type,
                                  use_shuffled_weight, weight_layout, activation_type,
                                  gemm1_bias_type, norm_topk_prob);
  }

  void check_routing() const override {
    FusedMoeLauncher::check_routing_common();
    if (has_precomputed(expert_indices)) {
      TVM_FFI_ICHECK_EQ(expert_indices.ndim(), 2) << "expert_indices must be 2D.";
      TVM_FFI_ICHECK_EQ(expert_indices.size(0), hidden_states.size(0))
          << "expert_indices and hidden_states must have same number of tokens.";
      TVM_FFI_ICHECK_EQ(expert_indices.size(1), args->top_k)
          << "expert_indices dim1 must match top_k.";
      TVM_FFI_ICHECK_EQ(expert_indices.dtype(), dl_int32) << "expert_indices must be int32.";
    }
    if (is_unpacked_routing()) {
      TVM_FFI_ICHECK(has_precomputed(expert_indices))
          << "expert_indices must be a 2D [num_tokens, top_k] tensor for unpacked precomputed "
             "routing.";
      TVM_FFI_ICHECK(expert_weights.dtype() == dl_bfloat16 || expert_weights.dtype() == dl_float32)
          << "expert_weights must be bfloat16 or float32 for unpacked precomputed routing.";
      TVM_FFI_ICHECK_EQ(expert_weights.ndim(), 2) << "expert_weights must be 2D.";
      TVM_FFI_ICHECK_EQ(expert_weights.size(0), hidden_states.size(0))
          << "expert_weights and hidden_states must have same number of tokens.";
      TVM_FFI_ICHECK_EQ(expert_weights.size(1), args->top_k)
          << "expert_weights dim1 must match top_k.";
    }

    // TODO n_group, topk_group validation?
  }

  void prepare_routing() override {
    FusedMoeLauncher::prepare_routing_common();

    args->mDtypeElt = btg::Dtype::Bfloat16;
    args->mUseDeepSeekFp8 = false;

    // Set expert weights dtype based on routing bias
    auto const routing_bias_dtype =
        routing_bias.has_value() ? routing_bias.value().dtype() : dl_bfloat16;
    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;

    auto const routing_logits_dtype =
        routing_logits.has_value() ? routing_logits.value().dtype() : dl_bfloat16;
    mRoutingLogitsDtype =
        routing_logits_dtype == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;

    bool has_precomputed_indices = has_precomputed(expert_indices);
    if (has_precomputed_indices) {
      // Use expert_indices directly
      workspace.routing_expert_indexes =
          static_cast<int*>(const_cast<void*>(expert_indices.data_ptr()));
    }
    bool has_precomputed_weights = has_precomputed(expert_weights);
    if (has_precomputed_weights) {
      if (is_unpacked_routing()) {
        args->mDtypeExpW = expert_weights_dtype(expert_weights);
      }
      workspace.expert_weights = const_cast<void*>(expert_weights.data_ptr());
    } else {
      // Allocate the routing-output buffer as bf16 to match the kernel's output
      // (mDtypeOutput is always Bfloat16 in trtllm_fused_moe_runner.cu, never the
      // logits dtype); a fp32 alloc would mislabel bf16 data when this buffer is
      // surfaced to the caller verbatim on do_finalize=false. See #3595.
      FusedMoeLauncher::expert_weights =
          alloc_tensor({args->num_tokens, args->top_k}, dl_bfloat16, hidden_states.device());
      workspace.expert_weights = FusedMoeLauncher::expert_weights.data_ptr();
    }
  }

  int32_t* precomputed_expert_ids() const override { return unpacked_expert_ids(expert_indices); }

  void check_moe() const override {
    FusedMoeLauncher::check_moe_common();

    TVM_FFI_ICHECK(weight_layout == batchedGemm::gemm::MatrixLayout::BlockMajorK)
        << "BF16 Moe: weight_layout must be BlockMajorK";
    check_weights_shape("gemm1");
    check_weights_shape("gemm2");
    check_optional_per_expert_float_tensor(gemm1_alpha, "gemm1_alpha");
    check_optional_per_expert_float_tensor(gemm1_beta, "gemm1_beta");
    check_optional_per_expert_float_tensor(gemm1_clamp_limit, "gemm1_clamp_limit");
    if (gemm1_alpha.has_value() || gemm1_beta.has_value() || gemm1_clamp_limit.has_value()) {
      TVM_FFI_ICHECK(activation_type == ActivationType::Swiglu)
          << "gemm1_alpha, gemm1_beta, and gemm1_clamp_limit are only supported for "
             "ActivationType::Swiglu.";
    }

    TVM_FFI_ICHECK_EQ(args->intermediate_size % 128, 0)
        << "the second dimension of weights must be a multiple of 128.";
  }

  /** Allocate and bind one ordinary BF16 MoE body workspace. */
  void prepare_moe(int64_t& moe_tactic) override {
    FusedMoeLauncher::prepare_moe_common(moe_tactic);

    // Blackwell TMA may select BASE_128KB address generation from the logical tensor-map shape.
    // Keep at least 128 KiB mapped from each activation base, as the quantized launchers do below.
    int32_t max_num_padded_tokens = workspace.total_max_padded_tokens;
    int32_t max_num_padded_tokens_gemm1 =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
            max_num_padded_tokens, args->intermediate_size,
            btg::dtypeGetNumBits(btg::Dtype::Bfloat16));
    int32_t max_num_padded_tokens_gemm2 =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
            max_num_padded_tokens, args->hidden_size, btg::dtypeGetNumBits(btg::Dtype::Bfloat16));
    gemm1_output = alloc_tensor({max_num_padded_tokens_gemm1, args->intermediate_size}, dl_bfloat16,
                                hidden_states.device());
    gemm2_output = alloc_tensor({max_num_padded_tokens_gemm2, args->hidden_size}, dl_bfloat16,
                                hidden_states.device());

    workspace.hidden_states_scale_linear = nullptr;
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = nullptr;
    workspace.activation_output = nullptr;
    workspace.activation_output_scale = nullptr;
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;

    // Only the finalize step writes `output`; when do_finalize is false the
    // result is taken from gemm2_output instead, so skip this allocation.
    if (args->do_finalize && args->output == nullptr) {
      output =
          alloc_tensor({args->num_tokens, args->hidden_size}, dl_bfloat16, hidden_states.device());
      args->output = output.data_ptr();
    }
    args->output_scale = nullptr;
    args->gemm1_alpha =
        gemm1_alpha.has_value() ? static_cast<float*>(gemm1_alpha.value().data_ptr()) : nullptr;
    args->gemm1_beta =
        gemm1_beta.has_value() ? static_cast<float*>(gemm1_beta.value().data_ptr()) : nullptr;
    args->gemm1_clamp_limit = gemm1_clamp_limit.has_value()
                                  ? static_cast<float*>(gemm1_clamp_limit.value().data_ptr())
                                  : nullptr;
  }

  /** Allocate graph-stable BF16 body buffers without launching routing or GEMMs. */
  BF16DABodyBuffers prepare_da_body(RoutingMetadataBuffers const& routing_metadata,
                                    int64_t moe_tactic) {
    ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
    TVM_FFI_ICHECK(args->do_finalize) << "BF16 DA bodies require finalized output.";
    bind_routing_metadata(routing_metadata);
    args->mDtypeExpW = routing_metadata.expert_weights.dtype() == dl_float32 ? btg::Dtype::Fp32
                                                                             : btg::Dtype::Bfloat16;
    check_moe();
    prepare_moe(moe_tactic);
    return {gemm1_output, gemm2_output, workspace_fc1, workspace_fc2};
  }

  /** Launch one BF16 body from prepared routing metadata and exact ABI buffers. */
  void run_da_body(RoutingMetadataBuffers const& routing_metadata,
                   BF16DABodyBuffers const& prepared, int64_t moe_tactic, bool enable_pdl) {
    ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
    TVM_FFI_ICHECK(args->do_finalize) << "BF16 DA bodies require finalized output.";
    bind_routing_metadata(routing_metadata);
    args->mDtypeExpW = routing_metadata.expert_weights.dtype() == dl_float32 ? btg::Dtype::Fp32
                                                                             : btg::Dtype::Bfloat16;
    prepare_moe_runner(moe_tactic);
    prepared.bind(workspace);
    args->gemm1_alpha = nullptr;
    args->gemm1_beta = nullptr;
    args->gemm1_clamp_limit = nullptr;
    cudaStream_t stream = get_stream(hidden_states.device());
    moe_runner->run(*args, workspace, hidden_states.device().device_id, stream, moe_tactic,
                    enable_pdl);
  }

  static Array<Array<int64_t>> getValidConfigs(int64_t top_k, int64_t hidden_size,
                                               int64_t intermediate_size, int64_t num_local_experts,
                                               int64_t num_tokens, int64_t act_type,
                                               bool use_shuffled_weight, int64_t weight_layout,
                                               batchedGemm::gemm::BiasType gemm1_bias_type) {
    Array<Array<int64_t>> valid_configs;

    std::vector<int32_t> supported_tile_nums(mSupportedTileNums.begin(), mSupportedTileNums.end());
    std::set<int32_t> selected_tile_nums =
        computeSelectedTileN(supported_tile_nums, num_tokens, top_k, num_local_experts);

    for (int32_t tile_N : selected_tile_nums) {
      auto moe_runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
          btg::Dtype::Bfloat16,  // dtype_act
          btg::Dtype::Bfloat16,  // dtype_weights
          false,                 // useDeepSeekFp8
          tile_N, static_cast<ActivationType>(act_type), use_shuffled_weight,
          static_cast<batchedGemm::gemm::MatrixLayout>(weight_layout), gemm1_bias_type);

      auto cfgs = moe_runner->getValidConfigIndices(top_k, hidden_size, intermediate_size,
                                                    num_local_experts, num_tokens);

      for (auto cfg : cfgs) {
        valid_configs.push_back({tile_N, cfg});
      }
    }

    return valid_configs;
  }

 private:
  TensorView expert_weights;
  TensorView expert_indices;
  Optional<TensorView> gemm1_alpha;
  Optional<TensorView> gemm1_beta;
  Optional<TensorView> gemm1_clamp_limit;
};

class Fp8PerTensorLauncher : public FusedMoeLauncher {
 public:
  static constexpr std::array<int32_t, 5> mSupportedTileNums = {8, 16, 32, 64, 128};

  Fp8PerTensorLauncher(Optional<TensorView> const& routing_logits,
                       Optional<TensorView> const& routing_bias, TensorView const& hidden_states,
                       TensorView const& gemm1_weights, TensorView const& output1_scales_scalar,
                       TensorView const& output1_scales_gate_scalar,
                       TensorView const& gemm2_weights, TensorView const& output2_scales_scalar,
                       Optional<TensorView> const& expert_indices = Optional<TensorView>(),
                       Optional<TensorView> const& expert_weights = Optional<TensorView>(),
                       RoutingInputMode routing_input_mode = RoutingInputMode::FromLogits)
      : FusedMoeLauncher(routing_logits, routing_bias, hidden_states, gemm1_weights,
                         Optional<TensorView>(), Optional<TensorView>(output1_scales_scalar),
                         Optional<TensorView>(output1_scales_gate_scalar), gemm2_weights,
                         Optional<TensorView>(output2_scales_scalar), Optional<TensorView>(),
                         routing_input_mode),
        expert_indices(expert_indices),
        expert_weights(expert_weights),
        use_routing_scales_on_input(false) {}

  bool has_precomputed_routing() const {
    return expert_indices.has_value() && expert_indices.value().ndim() == 2 &&
           expert_indices.value().size(0) > 0;
  }

  void init(std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
            int64_t tile_tokens_dim, int64_t routing_method_type, bool use_shuffled_weight,
            int64_t weight_layout, bool use_routing_scales_on_input_param,
            ActivationType activation_type, bool norm_topk_prob = true) {
    this->use_routing_scales_on_input = use_routing_scales_on_input_param;
    args->mUseRoutingScalesOnInput = use_routing_scales_on_input_param;

    auto dtype = hidden_states.dtype();
    if (dtype == dl_float16) {
      mDtypeAct = btg::Dtype::Fp16;
    } else if (dtype == dl_bfloat16) {
      mDtypeAct = btg::Dtype::Bfloat16;
    } else if (dtype == dl_float8_e4m3fn) {
      mDtypeAct = btg::Dtype::E4m3;
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported input dtype for FP8 MoE.";
    }
    mDtypeWeights = btg::Dtype::E4m3;

    FusedMoeLauncher::init_common(
        std::move(args), tile_tokens_dim, routing_method_type, use_shuffled_weight, weight_layout,
        activation_type, static_cast<int64_t>(batchedGemm::gemm::BiasType::None), norm_topk_prob);
  }

  void check_routing() const override {
    FusedMoeLauncher::check_routing_common();
    if (has_precomputed_routing()) {
      TVM_FFI_ICHECK_EQ(expert_indices.value().size(0), hidden_states.size(0))
          << "expert_indices and hidden_states must have same number of tokens.";
      TVM_FFI_ICHECK_EQ(expert_indices.value().size(1), args->top_k)
          << "expert_indices dim1 must match top_k.";
      TVM_FFI_ICHECK_EQ(expert_indices.value().dtype(), dl_int32)
          << "expert_indices must be int32.";
    }
    if (is_unpacked_routing()) {
      TVM_FFI_ICHECK(has_precomputed_routing())
          << "expert_indices must be a 2D [num_tokens, top_k] tensor for unpacked precomputed "
             "routing.";
      TVM_FFI_ICHECK(expert_weights.has_value())
          << "expert_weights is required for unpacked precomputed routing.";
      auto const& weights = expert_weights.value();
      TVM_FFI_ICHECK(weights.dtype() == dl_bfloat16 || weights.dtype() == dl_float32)
          << "expert_weights must be bfloat16 or float32 for unpacked precomputed routing.";
      TVM_FFI_ICHECK_EQ(weights.ndim(), 2) << "expert_weights must be 2D.";
      TVM_FFI_ICHECK_EQ(weights.size(0), hidden_states.size(0))
          << "expert_weights and hidden_states must have same number of tokens.";
      TVM_FFI_ICHECK_EQ(weights.size(1), args->top_k) << "expert_weights dim1 must match top_k.";
    }
  }

  void prepare_routing() override {
    FusedMoeLauncher::prepare_routing_common();

    if (has_precomputed_routing()) {
      workspace.routing_expert_indexes =
          static_cast<int*>(const_cast<void*>(expert_indices.value().data_ptr()));
    }

    auto dtype = hidden_states.dtype();
    if (dtype == dl_float16) {
      args->mDtypeElt = btg::Dtype::Fp16;
    } else if (dtype == dl_bfloat16) {
      args->mDtypeElt = btg::Dtype::Bfloat16;
    } else if (dtype == dl_float8_e4m3fn) {
      args->mDtypeElt = btg::Dtype::E4m3;
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported input dtype for MoE.";
    }

    args->mDtypeOut = btg::Dtype::Bfloat16;
    args->mUseDeepSeekFp8 = false;

    auto const routing_bias_dtype =
        routing_bias.has_value() ? routing_bias.value().dtype() : dl_bfloat16;
    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;

    auto const routing_logits_dtype =
        routing_logits.has_value() ? routing_logits.value().dtype() : dl_bfloat16;
    mRoutingLogitsDtype =
        routing_logits_dtype == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;

    if (is_unpacked_routing()) {
      auto const& weights = expert_weights.value();
      args->mDtypeExpW = expert_weights_dtype(weights);
      workspace.expert_weights = const_cast<void*>(weights.data_ptr());
    } else {
      FusedMoeLauncher::expert_weights =
          alloc_tensor({args->num_tokens, args->top_k}, dl_bfloat16, hidden_states.device());
      workspace.expert_weights = FusedMoeLauncher::expert_weights.data_ptr();
    }
    if (static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::Llama4) {
      if (is_unpacked_routing() && expert_weights.value().dtype() == dl_float32) {
        auto const& weights = expert_weights.value();
        routing_scales =
            alloc_tensor({args->num_tokens, args->top_k}, dl_bfloat16, hidden_states.device());
        cast_fp32_to_bf16(routing_scales.data_ptr(), weights.data_ptr(), weights.numel(),
                          get_stream(hidden_states.device()));
        workspace.token_scales = routing_scales.data_ptr();
      } else {
        workspace.token_scales = workspace.expert_weights;
      }
    }
  }

  int32_t* precomputed_expert_ids() const override {
    return expert_indices.has_value() ? unpacked_expert_ids(expert_indices.value()) : nullptr;
  }

  void check_moe() const override {
    FusedMoeLauncher::check_moe_common();

    TVM_FFI_ICHECK(output1_scales_scalar.has_value())
        << "output1_scales_scalar is required for FP8 MoE";
    TVM_FFI_ICHECK_EQ(output1_scales_scalar.value().dtype(), dl_float32)
        << "output1_scales_scalar must be float.";
    TVM_FFI_ICHECK_EQ(output1_scales_scalar.value().ndim(), 1)
        << "output1_scales_scalar must be 1D.";
    TVM_FFI_ICHECK_EQ(output1_scales_scalar.value().size(0), args->local_num_experts)
        << "output1_scales_scalar has incorrect dim 0.";

    TVM_FFI_ICHECK(output1_scales_gate_scalar.has_value())
        << "output1_scales_gate_scalar is required for FP8 MoE";
    TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar.value().dtype(), dl_float32)
        << "output1_scales_gate_scalar must be float.";
    TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar.value().ndim(), 1)
        << "output1_scales_gate_scalar must be 1D.";
    TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar.value().size(0), args->local_num_experts)
        << "output1_scales_gate_scalar has incorrect dim 0.";

    TVM_FFI_ICHECK(output2_scales_scalar.has_value())
        << "output2_scales_scalar is required for FP8 MoE";
    TVM_FFI_ICHECK_EQ(output2_scales_scalar.value().dtype(), dl_float32)
        << "output2_scales_scalar must be float.";
    TVM_FFI_ICHECK_EQ(output2_scales_scalar.value().ndim(), 1)
        << "output2_scales_scalar must be 1D.";
    TVM_FFI_ICHECK_EQ(output2_scales_scalar.value().size(0), args->local_num_experts)
        << "output2_scales_scalar has incorrect dim 0.";

    TVM_FFI_ICHECK(hidden_states.dtype() == dl_float8_e4m3fn ||
                   hidden_states.dtype() == dl_float16 || hidden_states.dtype() == dl_bfloat16)
        << "FP8 MoE: hidden_states must be float8_e4m3fn, float16, or bfloat16.";
    TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_float8_e4m3fn)
        << "FP8 MoE: gemm1_weights must be float8_e4m3fn.";
    TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_float8_e4m3fn)
        << "FP8 MoE: gemm2_weights must be float8_e4m3fn.";
  }

  /** Allocate and bind one ordinary FP8 per-tensor MoE body workspace. */
  void prepare_moe(int64_t& moe_tactic) override {
    FusedMoeLauncher::prepare_moe_common(moe_tactic);

    int32_t max_num_padded_tokens_gemm1 = workspace.total_max_padded_tokens + args->num_experts;
    int32_t max_num_padded_tokens_gemm2 = workspace.total_max_padded_tokens;

    gemm1_output = alloc_tensor({max_num_padded_tokens_gemm1, 2 * args->intermediate_size},
                                dl_uint8, hidden_states.device());
    gemm1_output_scale =
        alloc_tensor({2 * args->intermediate_size / 128, max_num_padded_tokens_gemm1}, dl_float32,
                     hidden_states.device());

    gemm2_output = alloc_tensor({max_num_padded_tokens_gemm2, args->hidden_size}, dl_bfloat16,
                                hidden_states.device());

    workspace.hidden_states_scale_linear = nullptr;
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = static_cast<float*>(gemm1_output_scale.data_ptr());
    workspace.activation_output = nullptr;
    workspace.activation_output_scale = nullptr;
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;

    // Only the finalize step writes `output`; when do_finalize is false the
    // result is taken from gemm2_output instead, so skip this allocation.
    if (args->do_finalize && args->output == nullptr) {
      output =
          alloc_tensor({args->num_tokens, args->hidden_size}, dl_bfloat16, hidden_states.device());
      args->output = output.data_ptr();
    }
    args->output_scale = nullptr;

    // Set scale pointers
    TVM_FFI_ICHECK(output1_scales_scalar.has_value());
    TVM_FFI_ICHECK(output1_scales_gate_scalar.has_value());
    TVM_FFI_ICHECK(output2_scales_scalar.has_value());

    args->output1_scales_scalar = static_cast<float*>(output1_scales_scalar.value().data_ptr());
    args->output1_scales_gate_scalar =
        static_cast<float*>(output1_scales_gate_scalar.value().data_ptr());
    args->output2_scales_scalar = static_cast<float*>(output2_scales_scalar.value().data_ptr());
  }

  /** Allocate graph-stable FP8 per-tensor buffers for one exact routed body. */
  FP8PerTensorDABodyBuffers prepare_da_body(RoutingMetadataBuffers const& routing_metadata,
                                            int64_t moe_tactic) {
    ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
    TVM_FFI_ICHECK(args->do_finalize) << "FP8 per-tensor DA bodies require finalized output.";
    bind_routing_metadata(routing_metadata);
    args->mDtypeExpW = expert_weights_dtype(routing_metadata.expert_weights);
    check_moe();
    prepare_moe(moe_tactic);
    return {gemm1_output, gemm1_output_scale, gemm2_output, workspace_fc1, workspace_fc2};
  }

  /** Launch one exact FP8 per-tensor body from prepared routing metadata. */
  void run_da_body(RoutingMetadataBuffers const& routing_metadata,
                   FP8PerTensorDABodyBuffers const& prepared, int64_t moe_tactic, bool enable_pdl) {
    // Rebind the selected tile's routing record before resolving the complete body tactic.
    ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
    TVM_FFI_ICHECK(args->do_finalize) << "FP8 per-tensor DA bodies require finalized output.";
    bind_routing_metadata(routing_metadata);
    args->mDtypeExpW = expert_weights_dtype(routing_metadata.expert_weights);
    prepare_moe_runner(moe_tactic);
    workspace.hidden_states_scale_linear = nullptr;
    workspace.activation_output = nullptr;
    workspace.activation_output_scale = nullptr;
    workspace.gemm2_output_scale = nullptr;
    prepared.bind(workspace);
    TVM_FFI_ICHECK(output1_scales_scalar.has_value());
    TVM_FFI_ICHECK(output1_scales_gate_scalar.has_value());
    TVM_FFI_ICHECK(output2_scales_scalar.has_value());
    args->output1_scales_scalar = static_cast<float*>(output1_scales_scalar.value().data_ptr());
    args->output1_scales_gate_scalar =
        static_cast<float*>(output1_scales_gate_scalar.value().data_ptr());
    args->output2_scales_scalar = static_cast<float*>(output2_scales_scalar.value().data_ptr());
    // Preserve Llama4's routing-scale convention while launching through the typed FP8 ABI.
    if (static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::Llama4) {
      workspace.token_scales = workspace.expert_weights;
    }
    cudaStream_t stream = get_stream(hidden_states.device());
    moe_runner->run(*args, workspace, hidden_states.device().device_id, stream, moe_tactic,
                    enable_pdl);
  }

 private:
  Optional<TensorView> expert_indices;
  Optional<TensorView> expert_weights;
  bool use_routing_scales_on_input;
  Tensor gemm1_output_scale;
  Tensor activation_output_scale;
  Tensor routing_scales;

 public:
  static Array<Array<int64_t>> getValidConfigs(int64_t top_k, int64_t hidden_size,
                                               int64_t intermediate_size, int64_t num_local_experts,
                                               int64_t num_tokens, int64_t act_type,
                                               bool use_shuffled_weight, int64_t weight_layout,
                                               btg::Dtype dtype_act, btg::Dtype dtype_weights,
                                               bool use_routing_scales_on_input) {
    Array<Array<int64_t>> valid_configs;

    std::vector<int32_t> supported_tile_nums(mSupportedTileNums.begin(), mSupportedTileNums.end());
    std::set<int32_t> selected_tile_nums =
        computeSelectedTileN(supported_tile_nums, num_tokens, top_k, num_local_experts);

    for (int32_t tile_N : selected_tile_nums) {
      auto moe_runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
          dtype_act, dtype_weights,
          false,  // useDeepSeekFp8
          tile_N, static_cast<ActivationType>(act_type), use_shuffled_weight,
          static_cast<batchedGemm::gemm::MatrixLayout>(weight_layout),
          // FP8 per-tensor doesn't use Mn-bias (LoRA) cubins.
          /*gemm1BiasType*/ batchedGemm::gemm::BiasType::None,
          /*usePerTokenScalingGemm1*/ use_routing_scales_on_input,
          /*usePerTokenScalingGemm2*/ false, false, false);

      auto cfgs = moe_runner->getValidConfigIndices(top_k, hidden_size, intermediate_size,
                                                    num_local_experts, num_tokens);

      for (auto cfg : cfgs) {
        valid_configs.push_back({tile_N, cfg});
      }
    }

    return valid_configs;
  }
};

class Fp8PerChannelLauncher : public FusedMoeLauncher {
 public:
  static constexpr std::array<int32_t, 5> mSupportedTileNums = {8, 16, 32, 64, 128};

  Fp8PerChannelLauncher(Optional<TensorView> const& routing_logits,
                        Optional<TensorView> const& routing_bias, TensorView const& hidden_states,
                        TensorView const& hidden_states_scale, TensorView const& gemm1_weights,
                        TensorView const& gemm1_per_channel_weight_scale,
                        TensorView const& output1_scales_scalar,
                        TensorView const& output1_scales_gate_scalar,
                        TensorView const& gemm2_weights,
                        TensorView const& gemm2_per_channel_weight_scale,
                        TensorView const& output2_scales_scalar, TensorView const& expert_indices,
                        TensorView const& expert_weights)
      : FusedMoeLauncher(routing_logits, routing_bias, hidden_states, gemm1_weights,
                         Optional<TensorView>(), Optional<TensorView>(output1_scales_scalar),
                         Optional<TensorView>(output1_scales_gate_scalar), gemm2_weights,
                         Optional<TensorView>(output2_scales_scalar),
                         Optional<TensorView>(hidden_states_scale)),
        hidden_states_scale_(hidden_states_scale),
        gemm1_per_channel_weight_scale_(gemm1_per_channel_weight_scale),
        gemm2_per_channel_weight_scale_(gemm2_per_channel_weight_scale),
        expert_indices_(expert_indices),
        expert_weights_(expert_weights) {
    use_per_channel_scaling_gemm1 = true;
    use_per_channel_scaling_gemm2 = true;
  }

  void init(std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
            int64_t tile_tokens_dim, int64_t routing_method_type, bool use_shuffled_weight,
            int64_t weight_layout, bool use_routing_scales_on_input, ActivationType activation_type,
            bool norm_topk_prob = true) {
    args->mUseRoutingScalesOnInput = use_routing_scales_on_input;

    auto dtype = hidden_states.dtype();
    if (dtype == dl_float16) {
      mDtypeAct = btg::Dtype::Fp16;
    } else if (dtype == dl_bfloat16) {
      mDtypeAct = btg::Dtype::Bfloat16;
    } else if (dtype == dl_float8_e4m3fn) {
      mDtypeAct = btg::Dtype::E4m3;
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "Unsupported input dtype for FP8 per-channel MoE.";
    }
    mDtypeWeights = btg::Dtype::E4m3;

    FusedMoeLauncher::init_common(
        std::move(args), tile_tokens_dim, routing_method_type, use_shuffled_weight, weight_layout,
        activation_type, static_cast<int64_t>(batchedGemm::gemm::BiasType::None), norm_topk_prob);
  }

  void check_routing() const override {
    FusedMoeLauncher::check_routing_common();
    TVM_FFI_ICHECK(routing_logits.has_value() || has_precomputed(expert_indices_))
        << "Either routing_logits or expert_indices must be provided.";
    if (args->mUseRoutingScalesOnInput && routing_logits.has_value()) {
      TVM_FFI_ICHECK_EQ(routing_logits.value().dtype(), dl_bfloat16)
          << "routing_logits must be bfloat16 when routing scales are applied on input.";
    }
    if (has_precomputed(expert_indices_)) {
      TVM_FFI_ICHECK_EQ(expert_indices_.size(0), hidden_states.size(0))
          << "expert_indices and hidden_states must have same number of tokens.";
      TVM_FFI_ICHECK_EQ(expert_indices_.size(1), args->top_k)
          << "expert_indices dim1 must match top_k.";
      TVM_FFI_ICHECK_EQ(expert_indices_.dtype(), dl_int32) << "expert_indices must be int32.";
    }
  }

  void prepare_routing() override {
    FusedMoeLauncher::prepare_routing_common();

    args->mDtypeElt = mDtypeAct;
    args->mDtypeOut = btg::Dtype::Bfloat16;
    args->mUseDeepSeekFp8 = false;

    auto const routing_bias_dtype =
        routing_bias.has_value() ? routing_bias.value().dtype() : dl_bfloat16;
    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;

    auto const routing_logits_dtype =
        routing_logits.has_value() ? routing_logits.value().dtype() : dl_bfloat16;
    mRoutingLogitsDtype =
        routing_logits_dtype == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;

    if (has_precomputed(expert_indices_)) {
      workspace.routing_expert_indexes =
          static_cast<int*>(const_cast<void*>(expert_indices_.data_ptr()));
    }

    if (has_precomputed(expert_weights_)) {
      if (is_unpacked_routing()) {
        args->mDtypeExpW = expert_weights_dtype(expert_weights_);
      }
      workspace.expert_weights = const_cast<void*>(expert_weights_.data_ptr());
    } else {
      FusedMoeLauncher::expert_weights =
          alloc_tensor({args->num_tokens, args->top_k}, dl_bfloat16, hidden_states.device());
      workspace.expert_weights = FusedMoeLauncher::expert_weights.data_ptr();
    }

    if (static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::Llama4) {
      workspace.token_scales = workspace.expert_weights;
    }
  }

  void check_moe() const override {
    FusedMoeLauncher::check_moe_common();

    auto gemm1_scale_dim1 = intermediate_size_factor * args->intermediate_size;

    TVM_FFI_ICHECK_EQ(hidden_states_scale_.dtype(), dl_float32)
        << "hidden_states_scale must be float32.";
    TVM_FFI_ICHECK_EQ(hidden_states_scale_.ndim(), 2)
        << "hidden_states_scale must be 2D [num_tokens, 1].";
    TVM_FFI_ICHECK_EQ(hidden_states_scale_.size(0), args->num_tokens)
        << "hidden_states_scale dim 0 must match num_tokens.";
    TVM_FFI_ICHECK_EQ(hidden_states_scale_.size(1), 1) << "hidden_states_scale dim 1 must be 1.";
    TVM_FFI_ICHECK(hidden_states_scale_.IsContiguous())
        << "hidden_states_scale must be contiguous.";

    TVM_FFI_ICHECK_EQ(gemm1_per_channel_weight_scale_.dtype(), dl_float32)
        << "gemm1_per_channel_weight_scale must be float32.";
    TVM_FFI_ICHECK_EQ(gemm1_per_channel_weight_scale_.ndim(), 2)
        << "gemm1_per_channel_weight_scale must be 2D [local_num_experts, "
           "intermediate_size_factor*intermediate_size].";
    TVM_FFI_ICHECK_EQ(gemm1_per_channel_weight_scale_.size(0), args->local_num_experts)
        << "gemm1_per_channel_weight_scale dim 0 must match local_num_experts.";
    TVM_FFI_ICHECK_EQ(gemm1_per_channel_weight_scale_.size(1), gemm1_scale_dim1)
        << "gemm1_per_channel_weight_scale dim 1 must be " << intermediate_size_factor
        << "*intermediate_size=" << gemm1_scale_dim1 << ".";

    TVM_FFI_ICHECK_EQ(output1_scales_scalar.value().dtype(), dl_float32)
        << "output1_scales_scalar must be float32.";
    TVM_FFI_ICHECK_EQ(output1_scales_scalar.value().ndim(), 1)
        << "output1_scales_scalar must be 1D [local_num_experts].";
    TVM_FFI_ICHECK_EQ(output1_scales_scalar.value().size(0), args->local_num_experts)
        << "output1_scales_scalar dim 0 must match local_num_experts.";

    TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar.value().dtype(), dl_float32)
        << "output1_scales_gate_scalar must be float32.";
    TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar.value().ndim(), 1)
        << "output1_scales_gate_scalar must be 1D [local_num_experts].";
    TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar.value().size(0), args->local_num_experts)
        << "output1_scales_gate_scalar dim 0 must match local_num_experts.";

    TVM_FFI_ICHECK_EQ(output2_scales_scalar.value().dtype(), dl_float32)
        << "output2_scales_scalar must be float32.";
    TVM_FFI_ICHECK_EQ(output2_scales_scalar.value().ndim(), 1)
        << "output2_scales_scalar must be 1D [local_num_experts].";
    TVM_FFI_ICHECK_EQ(output2_scales_scalar.value().size(0), args->local_num_experts)
        << "output2_scales_scalar dim 0 must match local_num_experts.";

    TVM_FFI_ICHECK_EQ(gemm2_per_channel_weight_scale_.dtype(), dl_float32)
        << "gemm2_per_channel_weight_scale must be float32.";
    TVM_FFI_ICHECK_EQ(gemm2_per_channel_weight_scale_.ndim(), 2)
        << "gemm2_per_channel_weight_scale must be 2D [local_num_experts, hidden_size].";
    TVM_FFI_ICHECK_EQ(gemm2_per_channel_weight_scale_.size(0), args->local_num_experts)
        << "gemm2_per_channel_weight_scale dim 0 must match local_num_experts.";
    TVM_FFI_ICHECK_EQ(gemm2_per_channel_weight_scale_.size(1), args->hidden_size)
        << "gemm2_per_channel_weight_scale dim 1 must match hidden_size.";

    TVM_FFI_ICHECK(hidden_states.dtype() == dl_float8_e4m3fn ||
                   hidden_states.dtype() == dl_float16 || hidden_states.dtype() == dl_bfloat16)
        << "FP8 per-channel MoE: hidden_states must be float8_e4m3fn, float16, or bfloat16.";
    TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_float8_e4m3fn)
        << "FP8 per-channel MoE: gemm1_weights must be float8_e4m3fn.";
    TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_float8_e4m3fn)
        << "FP8 per-channel MoE: gemm2_weights must be float8_e4m3fn.";
  }

  void prepare_moe(int64_t& moe_tactic) override {
    FusedMoeLauncher::prepare_moe_common(moe_tactic);

    int32_t max_num_padded_tokens_gemm1 = workspace.total_max_padded_tokens + args->num_experts;
    int32_t max_num_padded_tokens_gemm2 = workspace.total_max_padded_tokens;

    gemm1_output = alloc_tensor(
        {max_num_padded_tokens_gemm1, intermediate_size_factor * args->intermediate_size}, dl_uint8,
        hidden_states.device());
    gemm1_output_scale = alloc_tensor(
        {intermediate_size_factor * args->intermediate_size / 128, max_num_padded_tokens_gemm1},
        dl_float32, hidden_states.device());

    gemm2_output = alloc_tensor({max_num_padded_tokens_gemm2, args->hidden_size}, dl_bfloat16,
                                hidden_states.device());

    workspace.hidden_states_scale_linear = nullptr;
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = static_cast<float*>(gemm1_output_scale.data_ptr());
    workspace.activation_output = nullptr;
    workspace.activation_output_scale = nullptr;
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;

    if (args->do_finalize && args->output == nullptr) {
      output =
          alloc_tensor({args->num_tokens, args->hidden_size}, dl_bfloat16, hidden_states.device());
      args->output = output.data_ptr();
    }
    args->output_scale = nullptr;

    args->output1_scales_scalar = static_cast<float*>(output1_scales_scalar.value().data_ptr());
    args->output1_scales_gate_scalar =
        static_cast<float*>(output1_scales_gate_scalar.value().data_ptr());
    args->output2_scales_scalar = static_cast<float*>(output2_scales_scalar.value().data_ptr());

    args->hidden_states_scale = const_cast<void*>(hidden_states_scale_.data_ptr());
    args->gemm1_per_channel_weight_scale =
        static_cast<float*>(gemm1_per_channel_weight_scale_.data_ptr());
    args->gemm2_per_channel_weight_scale =
        static_cast<float*>(gemm2_per_channel_weight_scale_.data_ptr());
  }

 private:
  TensorView hidden_states_scale_;
  TensorView gemm1_per_channel_weight_scale_;
  TensorView gemm2_per_channel_weight_scale_;
  TensorView expert_indices_;
  TensorView expert_weights_;
  Tensor gemm1_output_scale;

 public:
  static Array<Array<int64_t>> getValidConfigs(int64_t top_k, int64_t hidden_size,
                                               int64_t intermediate_size, int64_t num_local_experts,
                                               int64_t num_tokens, int64_t act_type,
                                               bool use_shuffled_weight, int64_t weight_layout,
                                               btg::Dtype dtype_act, btg::Dtype dtype_weights) {
    Array<Array<int64_t>> valid_configs;

    std::vector<int32_t> supported_tile_nums(mSupportedTileNums.begin(), mSupportedTileNums.end());
    std::set<int32_t> selected_tile_nums =
        computeSelectedTileN(supported_tile_nums, num_tokens, top_k, num_local_experts);

    for (int32_t tile_N : selected_tile_nums) {
      auto moe_runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
          dtype_act, dtype_weights,
          false,  // useDeepSeekFp8
          tile_N, static_cast<ActivationType>(act_type), use_shuffled_weight,
          static_cast<batchedGemm::gemm::MatrixLayout>(weight_layout),
          /*gemm1BiasType*/ batchedGemm::gemm::BiasType::None,
          /*usePerTokenScalingGemm1*/ true,
          /*usePerTokenScalingGemm2*/ false,
          /*usePerChannelScalingGemm1*/ true,
          /*usePerChannelScalingGemm2*/ true);

      auto cfgs = moe_runner->getValidConfigIndices(top_k, hidden_size, intermediate_size,
                                                    num_local_experts, num_tokens);

      for (auto cfg : cfgs) {
        valid_configs.push_back({tile_N, cfg});
      }
    }

    return valid_configs;
  }
};

class Fp8BlockScaleLauncher : public FusedMoeLauncher {
 public:
  static constexpr std::array<int32_t, 5> mBaseSupportedTileNums = {8, 16, 32, 64, 128};

  static std::vector<int32_t> getSupportedTileNums(Fp8QuantizationType quantization_type) {
    std::vector<int32_t> tiles(mBaseSupportedTileNums.begin(), mBaseSupportedTileNums.end());
    if (quantization_type == Fp8QuantizationType::MxFp8) {
      tiles.push_back(256);
    }
    return tiles;
  }

  Fp8BlockScaleLauncher(Optional<TensorView> const& routing_logits,
                        Optional<TensorView> const& routing_bias, TensorView const& hidden_states,
                        TensorView const& hidden_states_scale, TensorView const& gemm1_weights,
                        TensorView const& gemm1_weights_scale,
                        Optional<TensorView> const& gemm1_bias,
                        Optional<TensorView> const& gemm1_alpha,
                        Optional<TensorView> const& gemm1_beta,
                        Optional<TensorView> const& gemm1_clamp_limit,
                        TensorView const& gemm2_weights, TensorView const& gemm2_weights_scale,
                        TensorView const& expert_indices, TensorView const& expert_weights,
                        Fp8QuantizationType quantization_type, RoutingInputMode routing_input_mode)
      : FusedMoeLauncher(routing_logits, routing_bias, hidden_states, gemm1_weights, gemm1_bias,
                         Optional<TensorView>(), Optional<TensorView>(), gemm2_weights,
                         Optional<TensorView>(), Optional<TensorView>(), routing_input_mode),
        hidden_states_scale(hidden_states_scale),
        gemm1_weights_scale(gemm1_weights_scale),
        gemm1_alpha(gemm1_alpha),
        gemm1_beta(gemm1_beta),
        gemm1_clamp_limit(gemm1_clamp_limit),
        gemm2_weights_scale(gemm2_weights_scale),
        expert_indices(expert_indices),
        expert_weights(expert_weights),
        quantization_type(quantization_type) {}

  void init(std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
            int64_t tile_tokens_dim, int64_t routing_method_type, bool use_shuffled_weight,
            int64_t weight_layout, ActivationType activation_type, int64_t gemm1_bias_type,
            bool norm_topk_prob = true) {
    if (quantization_type == Fp8QuantizationType::MxFp8) {
      mDtypeAct = btg::Dtype::MxE4m3;
      mDtypeWeights = btg::Dtype::MxE4m3;
    } else {
      mDtypeAct = btg::Dtype::E4m3;
      mDtypeWeights = btg::Dtype::E4m3;
    }

    auto dtype = hidden_states.dtype();
    if (dtype == dl_float16) {
      args->mDtypeElt = btg::Dtype::Fp16;
    } else if (dtype == dl_bfloat16) {
      args->mDtypeElt = btg::Dtype::Bfloat16;
    } else if (dtype == dl_float8_e4m3fn) {
      args->mDtypeElt = btg::Dtype::E4m3;
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported input dtype for MoE.";
    }

    // Output is always bfloat16 for FP8 block scale
    args->mDtypeOut = btg::Dtype::Bfloat16;

    FusedMoeLauncher::init_common(std::move(args), tile_tokens_dim, routing_method_type,
                                  use_shuffled_weight, weight_layout, activation_type,
                                  gemm1_bias_type, norm_topk_prob);
  }

  void check_routing() const override {
    if (has_precomputed(expert_indices)) {
      TVM_FFI_ICHECK_EQ(expert_indices.ndim(), 2) << "expert_indices must be 2D.";
      TVM_FFI_ICHECK_EQ(expert_indices.size(0), hidden_states.size(0))
          << "expert_indices and hidden_states must have same number of tokens.";
      TVM_FFI_ICHECK_EQ(expert_indices.size(1), args->top_k)
          << "expert_indices dim1 must match top_k.";
      TVM_FFI_ICHECK_EQ(expert_indices.dtype(), dl_int32) << "expert_indices must be int32.";
    }
    if (is_unpacked_routing()) {
      TVM_FFI_ICHECK(has_precomputed(expert_indices))
          << "expert_indices must be a 2D [num_tokens, top_k] tensor for unpacked precomputed "
             "routing.";
      TVM_FFI_ICHECK(expert_weights.dtype() == dl_bfloat16 || expert_weights.dtype() == dl_float32)
          << "expert_weights must be bfloat16 or float32 for unpacked precomputed routing.";
      TVM_FFI_ICHECK_EQ(expert_weights.ndim(), 2) << "expert_weights must be 2D.";
      TVM_FFI_ICHECK_EQ(expert_weights.size(0), hidden_states.size(0))
          << "expert_weights and hidden_states must have same number of tokens.";
      TVM_FFI_ICHECK_EQ(expert_weights.size(1), args->top_k)
          << "expert_weights dim1 must match top_k.";
    }

    FusedMoeLauncher::check_routing_common();

    if (static_cast<RoutingMethodType>(routing_method_type) != RoutingMethodType::DeepSeekV3) {
      TVM_FFI_ICHECK(args->n_group <= 1)
          << "Current routing kernel (no groups) only supports n_group <= 1";
      TVM_FFI_ICHECK(args->topk_group <= 1)
          << "Current routing kernel (no groups) only supports topk_group <= 1";
    }

    if (static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::DeepSeekV3) {
      TVM_FFI_ICHECK(args->n_group != 0) << "n_group should not be zero for DeepSeekV3 routing";
      TVM_FFI_ICHECK(args->topk_group != 0) << "if n_group is given, topk_group must be given";
      TVM_FFI_ICHECK_EQ(args->num_experts % args->n_group, 0)
          << "num_experts must be divisible by n_group";
      // DeepSeekV3 routing supports top_k up to:
      // - 8  when num_experts <= 384 (NumKimiK2Experts)
      // - 22 when num_experts > 384 (NumNemotronExperts path)
      // Keep this in sync with LAUNCH_ROUTING_DEEPSEEK in trtllm_fused_moe_routing_deepseek.cu.
      constexpr int32_t kNumKimiK2Experts = 384;  // same as in trtllm_fused_moe_routing_deepseek.cu
      int32_t max_supported_top_k = args->num_experts <= kNumKimiK2Experts ? 8 : 22;
      TVM_FFI_ICHECK(args->top_k <= max_supported_top_k && args->top_k > 0)
          << "Current routing kernel (with groups) only supports top_k<=" << max_supported_top_k
          << " && top_k>0 for num_experts=" << args->num_experts << ".";
      TVM_FFI_ICHECK(args->topk_group <= 4 && args->topk_group > 0)
          << "Current routing kernel only (with groups) supports topk_group<=4 && topk_group > 0.";
      TVM_FFI_ICHECK_LE(args->topk_group, args->n_group)
          << "n_group must not be smaller than topk_group.";
      TVM_FFI_ICHECK_LT(args->top_k, (args->topk_group * args->num_experts / args->n_group))
          << "top_k must be less than total number of experts in selected groups";
    } else if (static_cast<RoutingMethodType>(routing_method_type) ==
                   RoutingMethodType::Renormalize ||
               static_cast<RoutingMethodType>(routing_method_type) ==
                   RoutingMethodType::RenormalizeNaive ||
               static_cast<RoutingMethodType>(routing_method_type) ==
                   RoutingMethodType::SigmoidRenorm ||
               static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::Sigmoid ||
               static_cast<RoutingMethodType>(routing_method_type) ==
                   RoutingMethodType::TopKSigmoid) {
      TVM_FFI_ICHECK(args->top_k <= 32 && args->top_k > 0)
          << "Current routing kernel (no groups) only supports top_k<=32 && top_k>0.";
    } else if (static_cast<RoutingMethodType>(routing_method_type) == RoutingMethodType::Llama4) {
      TVM_FFI_ICHECK_EQ(args->top_k, 1)
          << "Current routing kernel (no groups, Llama4) only supports top_k=1.";
    }

    TVM_FFI_ICHECK_EQ(args->num_experts % 4, 0)
        << "Routing kernel expects that num_experts must be divisible by 4";
    TVM_FFI_ICHECK_GT(args->num_experts, args->top_k) << "num_experts must be greater than top_k";
    TVM_FFI_ICHECK_LE(args->local_num_experts + args->local_expert_offset, args->num_experts)
        << "num_experts must be greater or equal to local_num_experts + local_expert_offset";
  }

  void prepare_routing() override {
    FusedMoeLauncher::prepare_routing_common();

    auto dtype = hidden_states.dtype();
    if (dtype == dl_float16) {
      args->mDtypeElt = btg::Dtype::Fp16;
    } else if (dtype == dl_bfloat16) {
      args->mDtypeElt = btg::Dtype::Bfloat16;
    } else if (dtype == dl_float8_e4m3fn) {
      args->mDtypeElt = btg::Dtype::E4m3;
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported input dtype for MoE.";
    }

    args->mUseDeepSeekFp8 = quantization_type == Fp8QuantizationType::DeepSeekFp8;
    bool has_precomputed_indices = has_precomputed(expert_indices);
    if (has_precomputed_indices) {
      // Use expert_indices directly
      workspace.routing_expert_indexes =
          static_cast<int*>(const_cast<void*>(expert_indices.data_ptr()));
    } else {
      // Use routing_logits directly
      args->routing_logits = static_cast<float*>(routing_logits.value().data_ptr());
    }
    // Set expert weights dtype based on routing bias
    auto const routing_bias_dtype =
        routing_bias.has_value() ? routing_bias.value().dtype() : dl_bfloat16;
    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;
    int32_t const totalExpertsPerToken = args->top_k + args->num_fused_shared_experts;

    auto const routing_logits_dtype =
        routing_logits.has_value() ? routing_logits.value().dtype() : dl_bfloat16;
    mRoutingLogitsDtype =
        routing_logits_dtype == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;

    bool has_precomputed_weights = has_precomputed(expert_weights);
    if (!has_precomputed_weights) {
      // Allocate the routing-output buffer as bf16 to match the kernel's output
      // (always Bfloat16, never the logits dtype); a fp32 alloc would mislabel
      // bf16 data when this buffer is surfaced to the caller verbatim on
      // do_finalize=false. See #3595.
      FusedMoeLauncher::expert_weights = alloc_tensor({args->num_tokens, totalExpertsPerToken},
                                                      dl_bfloat16, hidden_states.device());
      workspace.expert_weights = FusedMoeLauncher::expert_weights.data_ptr();
    } else {
      if (is_unpacked_routing()) {
        args->mDtypeExpW = expert_weights_dtype(expert_weights);
      }
      workspace.expert_weights = const_cast<void*>(expert_weights.data_ptr());
    }
  }

  int32_t* precomputed_expert_ids() const override { return unpacked_expert_ids(expert_indices); }

  void check_moe() const override {
    FusedMoeLauncher::check_moe_common();

    TVM_FFI_ICHECK_EQ(hidden_states.dtype(), dl_float8_e4m3fn) << "hidden_states must be fp8.";
    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      TVM_FFI_ICHECK_EQ(hidden_states_scale.dtype(), dl_float32)
          << "hidden_states_scale must be float.";
      TVM_FFI_ICHECK_EQ(hidden_states_scale.ndim(), 2) << "hidden_states_scale must be 2D.";
      TVM_FFI_ICHECK_EQ(hidden_states_scale.size(0), hidden_states.size(1) / 128)
          << "hidden_states_scale dim0 must match hidden_states dim1 / 128.";
      TVM_FFI_ICHECK_EQ(hidden_states_scale.size(1), args->num_tokens)
          << "hidden_states_scale dim1 must match num_tokens.";
    } else if (quantization_type == Fp8QuantizationType::MxFp8) {
      TVM_FFI_CHECK(weight_layout == batchedGemm::gemm::MatrixLayout::MajorK,
                    "weight_layout must be MajorK for MxFp8.");
      TVM_FFI_ICHECK_EQ(hidden_states_scale.dtype(), dl_uint8);
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "trtllm_fp8_block_scale_moe only supports DeepSeekFp8 or MxFp8.";
    }

    TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_float8_e4m3fn) << "gemm1_weights must be fp8.";
    TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_float8_e4m3fn) << "gemm2_weights must be fp8.";
    check_optional_per_expert_float_tensor(gemm1_alpha, "gemm1_alpha");
    check_optional_per_expert_float_tensor(gemm1_beta, "gemm1_beta");
    check_optional_per_expert_float_tensor(gemm1_clamp_limit, "gemm1_clamp_limit");

    int64_t const totalLocalExperts = args->local_num_experts + args->num_fused_shared_experts;
    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      TVM_FFI_ICHECK_EQ(gemm1_weights_scale.dtype(), dl_float32)
          << "gemm1_weights_scale must be float.";
      TVM_FFI_ICHECK_EQ(gemm1_weights_scale.ndim(), 3) << "gemm1_weights_scale must be 3D.";
      TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(0), totalLocalExperts)
          << "gemm1_weights_scale has incorrect dim 0.";
      TVM_FFI_ICHECK_EQ(args->intermediate_size % 128, 0)
          << "intermediate_size must be a multiple of 128.";
      TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(1),
                        intermediate_size_factor * args->intermediate_size / 128)
          << "gemm1_weights_scale has incorrect shape.";
      TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(2), args->hidden_size / 128)
          << "gemm1_weights_scale has incorrect shape.";
    } else if (quantization_type == Fp8QuantizationType::MxFp8) {
      TVM_FFI_ICHECK_EQ(gemm1_weights_scale.dtype(), dl_uint8)
          << "gemm1_weights_scale must be uint8.";
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "trtllm_fp8_block_scale_moe only supports DeepSeekFp8 or MxFp8.";
    }

    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      TVM_FFI_ICHECK_EQ(gemm2_weights_scale.dtype(), dl_float32)
          << "gemm2_weights_scale must be float.";
      TVM_FFI_ICHECK_EQ(gemm2_weights_scale.ndim(), 3) << "gemm2_weights_scale must be 3D.";
      TVM_FFI_ICHECK_EQ(gemm2_weights_scale.size(0), totalLocalExperts)
          << "gemm2_weights_scale has incorrect dim 0.";
      TVM_FFI_ICHECK_EQ(gemm2_weights_scale.size(1), args->hidden_size / 128)
          << "gemm2_weights_scale has incorrect shape.";
      TVM_FFI_ICHECK_EQ(gemm2_weights_scale.size(2), args->intermediate_size / 128)
          << "gemm2_weights_scale has incorrect shape.";
    } else if (quantization_type == Fp8QuantizationType::MxFp8) {
      TVM_FFI_ICHECK_EQ(gemm2_weights_scale.dtype(), dl_uint8)
          << "gemm2_weights_scale must be uint8.";
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "trtllm_fp8_block_scale_moe only supports DeepSeekFp8 or MxFp8.";
    }

    check_weights_shape("gemm1");
    check_weights_shape("gemm2");

    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      TVM_FFI_ICHECK_EQ(args->intermediate_size % 128, 0)
          << "intermediate_size must be a multiple of 128.";
    }
  }

  void prepare_moe(int64_t& moe_tactic) override {
    FusedMoeLauncher::prepare_moe_common(moe_tactic);

    // Calculate max_num_padded_tokens for gemm1 and gemm2 using maybeGetMinTokenCount
    int32_t max_num_padded_tokens_gemm1 =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
            workspace.total_max_padded_tokens, args->intermediate_size,
            btg::dtypeGetNumBits(args->mDtypeElt));
    int32_t max_num_padded_tokens_gemm2 =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
            workspace.total_max_padded_tokens, args->hidden_size,
            btg::dtypeGetNumBits(args->mDtypeOut));

    // DeepSeek has unfused activation function so it must allocate using intermediate_size_factor
    auto const gemm1_output_hidden = quantization_type == Fp8QuantizationType::DeepSeekFp8
                                         ? intermediate_size_factor * args->intermediate_size
                                         : args->intermediate_size;
    gemm1_output = alloc_tensor({max_num_padded_tokens_gemm1, gemm1_output_hidden}, dl_uint8,
                                hidden_states.device());

    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      gemm1_output_scale = alloc_tensor({intermediate_size_factor * args->intermediate_size / 128,
                                         workspace.total_max_padded_tokens},
                                        dl_float32, hidden_states.device());
    } else if (quantization_type == Fp8QuantizationType::MxFp8) {
      // MxFP8 fuses the activation so no need for intermediate_size_factor
      int64_t sf_size = tensorrt_llm::computeSwizzledLayoutSFSize(max_num_padded_tokens_gemm1,
                                                                  args->intermediate_size / 32);
      gemm1_output_scale = alloc_tensor({sf_size}, dl_uint8, hidden_states.device());
    }

    // DeepSeek FP8 doesn't fuse the activation
    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      activation_output = alloc_tensor({max_num_padded_tokens_gemm1, args->intermediate_size},
                                       dl_uint8, hidden_states.device());
      activation_output_scale =
          alloc_tensor({args->intermediate_size / 128, max_num_padded_tokens_gemm1}, dl_float32,
                       hidden_states.device());
    }

    gemm2_output = alloc_tensor({max_num_padded_tokens_gemm2, args->hidden_size}, dl_bfloat16,
                                hidden_states.device());

    workspace.hidden_states_scale_linear = nullptr;
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = static_cast<float*>(gemm1_output_scale.data_ptr());
    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      workspace.activation_output = activation_output.data_ptr();
      workspace.activation_output_scale = static_cast<float*>(activation_output_scale.data_ptr());
    }
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;

    // Only the finalize step writes `output`; when do_finalize is false the
    // result is taken from gemm2_output instead, so skip this allocation.
    if (args->do_finalize && args->output == nullptr) {
      output =
          alloc_tensor({args->num_tokens, args->hidden_size}, dl_bfloat16, hidden_states.device());
      args->output = output.data_ptr();
    }
    args->output_scale = nullptr;

    args->hidden_states_scale = static_cast<float*>(hidden_states_scale.data_ptr());
    args->gemm1_weights_scale = static_cast<float*>(gemm1_weights_scale.data_ptr());
    args->gemm1_alpha =
        gemm1_alpha.has_value() ? static_cast<float*>(gemm1_alpha.value().data_ptr()) : nullptr;
    args->gemm1_beta =
        gemm1_beta.has_value() ? static_cast<float*>(gemm1_beta.value().data_ptr()) : nullptr;
    args->gemm1_clamp_limit = gemm1_clamp_limit.has_value()
                                  ? static_cast<float*>(gemm1_clamp_limit.value().data_ptr())
                                  : nullptr;
    args->gemm2_weights_scale = static_cast<float*>(gemm2_weights_scale.data_ptr());
  }

  /** Allocate graph-stable buffers for one exact DeepSeek FP8 body. */
  DeepSeekFP8DABodyBuffers prepare_deepseek_da_body(RoutingMetadataBuffers const& routing_metadata,
                                                    int64_t moe_tactic) {
    ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
    TVM_FFI_ICHECK(quantization_type == Fp8QuantizationType::DeepSeekFp8);
    TVM_FFI_ICHECK(args->do_finalize) << "DeepSeek FP8 DA bodies require finalized output.";
    bind_routing_metadata(routing_metadata);
    args->mDtypeExpW = routing_metadata.expert_weights.dtype() == dl_float32 ? btg::Dtype::Fp32
                                                                             : btg::Dtype::Bfloat16;
    args->mUseDeepSeekFp8 = true;
    check_moe();
    prepare_moe(moe_tactic);
    return {gemm1_output, gemm1_output_scale, activation_output, activation_output_scale,
            gemm2_output, workspace_fc1,      workspace_fc2};
  }

  /** Allocate graph-stable buffers for one exact MXFP8 body. */
  MXFP8DABodyBuffers prepare_mxfp8_da_body(RoutingMetadataBuffers const& routing_metadata,
                                           int64_t moe_tactic) {
    ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
    TVM_FFI_ICHECK(quantization_type == Fp8QuantizationType::MxFp8);
    TVM_FFI_ICHECK(args->do_finalize) << "MXFP8 DA bodies require finalized output.";
    bind_routing_metadata(routing_metadata);
    args->mDtypeExpW = routing_metadata.expert_weights.dtype() == dl_float32 ? btg::Dtype::Fp32
                                                                             : btg::Dtype::Bfloat16;
    args->mUseDeepSeekFp8 = false;
    check_moe();
    prepare_moe(moe_tactic);
    return {gemm1_output, gemm1_output_scale, gemm2_output, workspace_fc1, workspace_fc2};
  }

  /** Launch one exact DeepSeek FP8 body from its typed prepared buffers. */
  void run_deepseek_da_body(RoutingMetadataBuffers const& routing_metadata,
                            DeepSeekFP8DABodyBuffers const& prepared, int64_t moe_tactic,
                            bool enable_pdl) {
    // Resolve the DeepSeek routing-weight ABI before binding the lane-owned maximum workspace.
    ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
    TVM_FFI_ICHECK(quantization_type == Fp8QuantizationType::DeepSeekFp8);
    TVM_FFI_ICHECK(args->do_finalize) << "DeepSeek FP8 DA bodies require finalized output.";
    bind_routing_metadata(routing_metadata);
    args->mDtypeExpW = routing_metadata.expert_weights.dtype() == dl_float32 ? btg::Dtype::Fp32
                                                                             : btg::Dtype::Bfloat16;
    args->mUseDeepSeekFp8 = true;
    prepare_moe_runner(moe_tactic);
    workspace.hidden_states_scale_linear = nullptr;
    prepared.bind(workspace);
    workspace.gemm2_output_scale = nullptr;
    args->hidden_states_scale = static_cast<float*>(hidden_states_scale.data_ptr());
    args->gemm1_weights_scale = static_cast<float*>(gemm1_weights_scale.data_ptr());
    args->gemm1_alpha = nullptr;
    args->gemm1_beta = nullptr;
    args->gemm1_clamp_limit = nullptr;
    args->gemm2_weights_scale = static_cast<float*>(gemm2_weights_scale.data_ptr());
    // Launch the complete body after every dtype-specific scale pointer is bound.
    cudaStream_t stream = get_stream(hidden_states.device());
    moe_runner->run(*args, workspace, hidden_states.device().device_id, stream, moe_tactic,
                    enable_pdl);
  }

  /** Launch one exact MXFP8 body from its typed prepared buffers. */
  void run_mxfp8_da_body(RoutingMetadataBuffers const& routing_metadata,
                         MXFP8DABodyBuffers const& prepared, int64_t moe_tactic, bool enable_pdl) {
    // Resolve the MXFP8 routing-weight ABI before binding the lane-owned maximum workspace.
    ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
    TVM_FFI_ICHECK(quantization_type == Fp8QuantizationType::MxFp8);
    TVM_FFI_ICHECK(args->do_finalize) << "MXFP8 DA bodies require finalized output.";
    bind_routing_metadata(routing_metadata);
    args->mDtypeExpW = routing_metadata.expert_weights.dtype() == dl_float32 ? btg::Dtype::Fp32
                                                                             : btg::Dtype::Bfloat16;
    args->mUseDeepSeekFp8 = false;
    prepare_moe_runner(moe_tactic);
    workspace.hidden_states_scale_linear = nullptr;
    prepared.bind(workspace);
    workspace.gemm2_output_scale = nullptr;
    args->hidden_states_scale = static_cast<float*>(hidden_states_scale.data_ptr());
    args->gemm1_weights_scale = static_cast<float*>(gemm1_weights_scale.data_ptr());
    args->gemm1_alpha = nullptr;
    args->gemm1_beta = nullptr;
    args->gemm1_clamp_limit = nullptr;
    args->gemm2_weights_scale = static_cast<float*>(gemm2_weights_scale.data_ptr());
    // Launch the complete body after every dtype-specific scale pointer is bound.
    cudaStream_t stream = get_stream(hidden_states.device());
    moe_runner->run(*args, workspace, hidden_states.device().device_id, stream, moe_tactic,
                    enable_pdl);
  }

 private:
  TensorView hidden_states_scale;
  TensorView gemm1_weights_scale;
  Optional<TensorView> gemm1_alpha;
  Optional<TensorView> gemm1_beta;
  Optional<TensorView> gemm1_clamp_limit;
  TensorView gemm2_weights_scale;
  Tensor gemm1_output_scale;
  Tensor activation_output_scale;
  TensorView expert_indices;
  TensorView expert_weights;
  Fp8QuantizationType quantization_type;

 public:
  // Override to handle pre-computed routing.
  MoeRunResultBuffers run(int64_t moe_tactic, bool enable_pdl = true,
                          bool use_routing_scales_on_input = false, bool use_deep_seek_fp8 = false,
                          bool return_activation_output = false) override {
    ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
    check_routing();
    prepare_routing();

    cudaStream_t routing_stream = get_stream(hidden_states.device());
    tensorrt_llm::kernels::trtllmgen_moe::Routing::Runner routing_runner(tile_tokens_dim);

    bool use_precomputed = has_precomputed(expert_indices);
    // When using pre-computed routing, pass nullptr as routing_logits to tell the
    // routing runner to use the pre-computed expert indices from workspace.routing_expert_indexes
    int16_t* replay_ptr = nullptr;
    if (routing_replay_out.has_value()) {
      replay_ptr = reinterpret_cast<int16_t*>(routing_replay_out.value().data_ptr());
    }

    routing_runner.run(
        use_precomputed ? nullptr : args->routing_logits, args->routing_bias, args->num_tokens,
        args->num_experts, args->top_k, args->num_fused_shared_experts, args->n_group,
        args->topk_group, args->local_expert_offset, args->local_num_experts,
        args->routed_scaling_factor, workspace.routing_expert_indexes,
        static_cast<int*>(expert_count_histogram.data_ptr()),
        static_cast<int*>(total_num_padded_tokens.data_ptr()),
        static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr()),
        workspace.permuted_idx_to_expanded_idx,
        static_cast<int*>(permuted_idx_to_token_idx.data_ptr()), precomputed_expert_ids(),
        workspace.expert_weights, static_cast<int*>(num_tokens_per_expert.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_batch_idx.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_mn_limit.data_ptr()),
        static_cast<int*>(num_non_exiting_ctas.data_ptr()), args->mDtypeElt, mRoutingBiasDtype,
        use_routing_scales_on_input, use_deep_seek_fp8,
        static_cast<RoutingMethodType>(routing_method_type), routing_stream, mRoutingLogitsDtype,
        norm_topk_prob, replay_ptr, enable_pdl);

    check_moe();
    prepare_moe(moe_tactic);

    cudaStream_t moe_stream = get_stream(hidden_states.device());
    moe_runner->run(*args, workspace, hidden_states.device().device_id, moe_stream, moe_tactic,
                    enable_pdl);

    MoeRunResultBuffers result(args->do_finalize, args->do_finalize ? output : gemm2_output);
    if (!args->do_finalize) {
      result.expert_weights = FusedMoeLauncher::expert_weights;
    }
    if (!args->do_finalize || return_activation_output) {
      result.expanded_to_permuted_indices = expanded_idx_to_permuted_idx;
    }
    if (return_activation_output) {
      // For DSFp8, gemm1_output is the pre-activation FC1 output (shape [M, 2*I])
      // and the post-activation tensor lives in activation_output (shape [M, I]).
      // MxFp8 fuses SwiGLU into FC1 so gemm1_output IS already post-activation.
      result.activation_output =
          quantization_type == Fp8QuantizationType::DeepSeekFp8 ? activation_output : gemm1_output;
    }
    return result;
  }

  static Array<Array<int64_t>> getValidConfigs(
      int64_t top_k, int64_t hidden_size, int64_t intermediate_size, int64_t num_local_experts,
      int64_t num_tokens, bool use_shuffled_weight, int64_t weight_layout, btg::Dtype dtype_act,
      btg::Dtype dtype_weights, Fp8QuantizationType quantization_type, int64_t act_type,
      batchedGemm::gemm::BiasType gemm1_bias_type) {
    Array<Array<int64_t>> valid_configs;
    auto activation_type = validateAndCastActivationType(act_type);

    auto supported_tile_nums = getSupportedTileNums(quantization_type);
    std::set<int32_t> selected_tile_nums =
        computeSelectedTileN(supported_tile_nums, num_tokens, top_k, num_local_experts);

    for (int32_t tile_N : selected_tile_nums) {
      std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner> moe_runner;
      // Keep getValidConfigs constructor path aligned with runtime prepare_moe_common().
      // DSFp8 without bias uses the weights-only constructor;
      // DSFp8 + biasMn routes through the unified constructor below.
      if (quantization_type == Fp8QuantizationType::DeepSeekFp8 && dtype_act == btg::Dtype::E4m3 &&
          dtype_weights == btg::Dtype::E4m3 &&
          gemm1_bias_type == batchedGemm::gemm::BiasType::None) {
        TVM_FFI_ICHECK(static_cast<int>(activation_type) ==
                       static_cast<int>(ActivationType::Swiglu))
            << "DeepSeekFp8 only supports ActivationType::Swiglu, got "
            << static_cast<int>(activation_type) << ".";
        moe_runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
            dtype_weights, true /* useDeepSeekFp8 */, tile_N, use_shuffled_weight,
            static_cast<batchedGemm::gemm::MatrixLayout>(weight_layout));
      } else {
        // Under current trtllm_get_valid_moe_configs() dispatch rules, this else-path is
        // reached by FP8 block-scale MXFP8 (dtype_act=dtype_weights=MxE4m3) and by
        // DeepSeek FP8 with a gemm1 bias (BiasType::Mn).
        moe_runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
            dtype_act,                                              // dtypeAct
            dtype_weights,                                          // dtypeWeights
            quantization_type == Fp8QuantizationType::DeepSeekFp8,  // useDeepSeekFp8
            tile_N, activation_type, use_shuffled_weight,
            static_cast<batchedGemm::gemm::MatrixLayout>(weight_layout), gemm1_bias_type);
      }

      auto cfgs = moe_runner->getValidConfigIndices(top_k, hidden_size, intermediate_size,
                                                    num_local_experts, num_tokens);

      for (auto cfg : cfgs) {
        valid_configs.push_back({tile_N, cfg});
      }
    }

    return valid_configs;
  }
};

class MxInt4BlockScaleLauncher : public FusedMoeLauncher {
 public:
  static constexpr std::array<int32_t, 5> mSupportedTileNums = {8, 16, 32, 64, 128};

  MxInt4BlockScaleLauncher(Optional<TensorView> const& routing_logits,
                           Optional<TensorView> const& routing_bias,
                           TensorView const& expert_indices, TensorView const& expert_weights,
                           TensorView const& hidden_states, TensorView const& gemm1_weights,
                           TensorView const& gemm1_weights_scale,
                           Optional<TensorView> const& gemm1_alpha,
                           Optional<TensorView> const& gemm1_beta,
                           Optional<TensorView> const& gemm1_clamp_limit,
                           Optional<TensorView> const& gemm1_bias, TensorView const& gemm2_weights,
                           TensorView const& gemm2_weights_scale)
      : FusedMoeLauncher(routing_logits, routing_bias, hidden_states, gemm1_weights, gemm1_bias,
                         Optional<TensorView>(), Optional<TensorView>(), gemm2_weights,
                         Optional<TensorView>(), Optional<TensorView>()),
        gemm1_alpha(gemm1_alpha),
        gemm1_beta(gemm1_beta),
        gemm1_clamp_limit(gemm1_clamp_limit),
        gemm1_weights_scale(gemm1_weights_scale),
        gemm2_weights_scale(gemm2_weights_scale),
        expert_indices(expert_indices),
        expert_weights(expert_weights) {}

  void init(std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
            int64_t tile_tokens_dim, int64_t routing_method_type, int64_t gemm1_bias_type,
            bool norm_topk_prob = true) {
    // currently only support mxint4 x bf16
    auto dtype = hidden_states.dtype();
    if (dtype == dl_bfloat16) {
      args->mDtypeElt = btg::Dtype::Bfloat16;
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported input dtype for MoE.";
    }
    args->mDtypeOut = btg::Dtype::Bfloat16;

    mDtypeAct = btg::Dtype::Bfloat16;
    mDtypeWeights = btg::Dtype::MxInt4;

    FusedMoeLauncher::init_common(
        std::move(args), tile_tokens_dim, routing_method_type,
        /*use_shuffled_weight=*/true,
        static_cast<int64_t>(batchedGemm::gemm::MatrixLayout::BlockMajorK), ActivationType::Swiglu,
        gemm1_bias_type, norm_topk_prob);
  }

  void check_routing() const override {
    FusedMoeLauncher::check_routing_common();
    if (expert_indices.ndim() == 2 && expert_indices.size(0) > 0) {
      TVM_FFI_ICHECK_EQ(expert_indices.ndim(), 2) << "expert_indices must be 2D.";
      TVM_FFI_ICHECK_EQ(expert_indices.size(0), hidden_states.size(0))
          << "expert_indices and hidden_states must have same number of tokens.";
      TVM_FFI_ICHECK_EQ(expert_indices.size(1), args->top_k)
          << "expert_indices dim1 must match top_k.";
      TVM_FFI_ICHECK_EQ(expert_indices.dtype(), dl_int32) << "expert_indices must be int32.";
    }
  }

  void prepare_routing() override {
    FusedMoeLauncher::prepare_routing_common();

    args->mDtypeElt = mDtypeAct;
    args->mUseDeepSeekFp8 = false;
    // Set expert weights dtype based on routing bias
    auto const routing_bias_dtype =
        routing_bias.has_value() ? routing_bias.value().dtype() : dl_bfloat16;
    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;

    auto const routing_logits_dtype =
        routing_logits.has_value() ? routing_logits.value().dtype() : dl_bfloat16;
    mRoutingLogitsDtype =
        routing_logits_dtype == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;

    bool has_precomputed_indices = expert_indices.ndim() == 2 && expert_indices.size(0) > 0;
    if (has_precomputed_indices) {
      workspace.routing_expert_indexes =
          static_cast<int*>(const_cast<void*>(expert_indices.data_ptr()));
    }
    bool has_precomputed_weights = expert_weights.ndim() == 2 && expert_weights.size(0) > 0;
    if (has_precomputed_weights) {
      workspace.expert_weights = const_cast<void*>(expert_weights.data_ptr());
    } else {
      // Allocate the routing-output buffer as bf16 to match the kernel's output
      // (always Bfloat16, never the logits dtype); a fp32 alloc would mislabel
      // bf16 data when this buffer is surfaced to the caller verbatim on
      // do_finalize=false. See #3595.
      FusedMoeLauncher::expert_weights =
          alloc_tensor({args->num_tokens, args->top_k}, dl_bfloat16, hidden_states.device());
      workspace.expert_weights = FusedMoeLauncher::expert_weights.data_ptr();
    }
  }

  void check_moe() const override {
    FusedMoeLauncher::check_moe_common();

    TVM_FFI_ICHECK(mDtypeAct == btg::Dtype::Bfloat16)
        << "Only Bfloat16 is supported by MxInt4 block scale MoE";

    TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_uint8) << "gemm1_weights must be uint8.";
    TVM_FFI_ICHECK_EQ(gemm1_weights_scale.dtype(), dl_bfloat16)
        << "gemm1_weights_scale must be bf16.";
    TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_uint8) << "gemm2_weights must be uint8.";
    TVM_FFI_ICHECK_EQ(gemm2_weights_scale.dtype(), dl_bfloat16)
        << "gemm2_weights_scale must be bf16.";
  }

  void prepare_moe(int64_t& moe_tactic) override {
    args->hidden_states = hidden_states.data_ptr();
    args->hidden_states_scale = nullptr;
    args->gemm1_weights = gemm1_weights.data_ptr();
    args->gemm1_weights_scale = gemm1_weights_scale.data_ptr();
    args->gemm1_alpha =
        gemm1_alpha.has_value() ? static_cast<float*>(gemm1_alpha.value().data_ptr()) : nullptr;
    args->gemm1_beta =
        gemm1_beta.has_value() ? static_cast<float*>(gemm1_beta.value().data_ptr()) : nullptr;
    args->gemm1_clamp_limit = gemm1_clamp_limit.has_value()
                                  ? static_cast<float*>(gemm1_clamp_limit.value().data_ptr())
                                  : nullptr;
    args->gemm2_weights = gemm2_weights.data_ptr();
    args->gemm2_weights_scale = gemm2_weights_scale.data_ptr();
    args->output1_scales_scalar = nullptr;
    args->output1_scales_gate_scalar = nullptr;
    args->output2_scales_scalar = nullptr;

    FusedMoeLauncher::prepare_moe_common(moe_tactic);

    max_num_padded_tokens_gemm1 =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
            workspace.total_max_padded_tokens, args->intermediate_size,
            btg::dtypeGetNumBits(mDtypeAct));
    max_num_padded_tokens_gemm2 =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
            workspace.total_max_padded_tokens, args->hidden_size,
            btg::dtypeGetNumBits(btg::Dtype::Bfloat16));  // Output is always BF16

    auto const gemm1_output_hidden = args->intermediate_size;
    gemm1_output = alloc_tensor({max_num_padded_tokens_gemm1, gemm1_output_hidden}, dl_bfloat16,
                                hidden_states.device());

    // Allocate gemm2_output
    gemm2_output = alloc_tensor({max_num_padded_tokens_gemm2, args->hidden_size}, dl_bfloat16,
                                hidden_states.device());

    // Setup workspace pointers
    workspace.hidden_states_scale_linear = nullptr;  // MxInt4 doesn't use linear scale
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = nullptr;
    // Note: activation_output and activation_output_scale are set by the base class
    // prepare_moe_common() when gated activation is used
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;
  }

  /** Allocate graph-stable MXINT4 buffers for one exact routed body. */
  MXINT4DABodyBuffers prepare_da_body(RoutingMetadataBuffers const& routing_metadata,
                                      int64_t moe_tactic) {
    ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
    TVM_FFI_ICHECK(args->do_finalize) << "MXINT4 DA bodies require finalized output.";
    bind_routing_metadata(routing_metadata);
    args->mDtypeExpW = routing_metadata.expert_weights.dtype() == dl_float32 ? btg::Dtype::Fp32
                                                                             : btg::Dtype::Bfloat16;
    check_moe();
    prepare_moe(moe_tactic);
    return {gemm1_output, gemm2_output, workspace_fc1, workspace_fc2};
  }

  /** Launch one exact MXINT4 body from prepared routing metadata. */
  void run_da_body(RoutingMetadataBuffers const& routing_metadata,
                   MXINT4DABodyBuffers const& prepared, int64_t moe_tactic, bool enable_pdl) {
    ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
    TVM_FFI_ICHECK(args->do_finalize) << "MXINT4 DA bodies require finalized output.";
    bind_routing_metadata(routing_metadata);
    args->mDtypeExpW = routing_metadata.expert_weights.dtype() == dl_float32 ? btg::Dtype::Fp32
                                                                             : btg::Dtype::Bfloat16;
    prepare_moe_runner(moe_tactic);
    prepared.bind(workspace);
    args->hidden_states_scale = nullptr;
    args->gemm1_weights_scale = gemm1_weights_scale.data_ptr();
    args->gemm1_alpha = nullptr;
    args->gemm1_beta = nullptr;
    args->gemm1_clamp_limit = nullptr;
    args->gemm2_weights_scale = gemm2_weights_scale.data_ptr();
    cudaStream_t stream = get_stream(hidden_states.device());
    moe_runner->run(*args, workspace, hidden_states.device().device_id, stream, moe_tactic,
                    enable_pdl);
  }

 private:
  Optional<TensorView> gemm1_alpha;
  Optional<TensorView> gemm1_beta;
  Optional<TensorView> gemm1_clamp_limit;
  TensorView gemm1_weights_scale;
  TensorView gemm2_weights_scale;
  TensorView expert_indices;
  TensorView expert_weights;
  int32_t max_num_padded_tokens_gemm1{};
  int32_t max_num_padded_tokens_gemm2{};

 public:
  static Array<Array<int64_t>> getValidConfigs(int64_t top_k, int64_t hidden_size,
                                               int64_t intermediate_size, int64_t num_local_experts,
                                               int64_t num_tokens,
                                               batchedGemm::gemm::BiasType gemm1_bias_type) {
    Array<Array<int64_t>> valid_configs;

    std::vector<int32_t> tile_sizes(mSupportedTileNums.begin(), mSupportedTileNums.end());
    std::set<int32_t> selected_tile_nums =
        computeSelectedTileN(tile_sizes, num_tokens, top_k, num_local_experts);

    for (int32_t tile_N : selected_tile_nums) {
      auto moe_runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
          btg::Dtype::Bfloat16, btg::Dtype::MxInt4,
          false,  // useDeepSeekFp8
          tile_N, ActivationType::Swiglu, /*useShuffledMatrix*/ true,
          batchedGemm::gemm::MatrixLayout::BlockMajorK, gemm1_bias_type);

      auto cfgs = moe_runner->getValidConfigIndices(top_k, hidden_size, intermediate_size,
                                                    num_local_experts, num_tokens);

      for (auto cfg : cfgs) {
        valid_configs.push_back({tile_N, cfg});
      }
    }

    return valid_configs;
  }
};

class FP4BlockScaleLauncher : public FusedMoeLauncher {
 public:
  static constexpr std::array<int32_t, 4> mBaseSupportedTileNums = {8, 16, 32, 64};

  static std::vector<int32_t> getSupportedTileNums(btg::Dtype dtype_act, btg::Dtype dtype_weights) {
    std::vector<int32_t> tiles(mBaseSupportedTileNums.begin(), mBaseSupportedTileNums.end());
    if (dtype_act != btg::Dtype::Bfloat16) {
      tiles.push_back(128);
#ifndef TLLM_RUBIN_FEATURES
      // Keep tactic enumeration aligned with the public BMM artifact. The
      // previously separate Rubin BMM pin had no 192-tile FP4 kernels (its only
      // 192-tile kernels were FP8), and launchers are built eagerly for every
      // advertised tile, so advertising 192 there failed runner construction
      // outright. The consolidated multi-arch BMM pin does ship sm107a 192-tile
      // FP4 kernels, so this guard can be dropped once Rubin is re-verified.
      if ((dtype_weights == btg::Dtype::E2m1 && dtype_act == btg::Dtype::E2m1) ||
          (dtype_weights == btg::Dtype::MxE2m1 && dtype_act == btg::Dtype::MxE4m3)) {
        tiles.push_back(192);
      }
#endif
      tiles.push_back(256);
    }
    return tiles;
  }

  FP4BlockScaleLauncher(
      RoutingInputMode routing_input_mode, Optional<TensorView> const& routing_logits,
      Optional<TensorView> const& routing_bias, TensorView const& hidden_states,
      Optional<TensorView> const& hidden_states_scale, TensorView const& gemm1_weights,
      TensorView const& gemm1_weights_scale, Optional<TensorView> const& gemm1_bias,
      Optional<TensorView> const& gemm1_alpha, Optional<TensorView> const& gemm1_beta,
      Optional<TensorView> const& gemm1_clamp_limit, TensorView const& gemm2_weights,
      TensorView const& gemm2_weights_scale, Optional<TensorView> const& gemm2_bias,
      Optional<TensorView> const& output1_scales_scalar,
      Optional<TensorView> const& output1_scales_gate_scalar,
      Optional<TensorView> const& output2_scales_scalar,
      Optional<TensorView> const& per_token_scales, TensorView const& topk_ids,
      TensorView const& topk_weights)
      : FusedMoeLauncher(routing_logits, routing_bias, hidden_states, gemm1_weights, gemm1_bias,
                         output1_scales_scalar, output1_scales_gate_scalar, gemm2_weights,
                         output2_scales_scalar, per_token_scales, routing_input_mode),
        hidden_states_scale(hidden_states_scale),
        gemm1_weights_scale(gemm1_weights_scale),
        gemm1_alpha(gemm1_alpha),
        gemm1_beta(gemm1_beta),
        gemm1_clamp_limit(gemm1_clamp_limit),
        gemm2_weights_scale(gemm2_weights_scale),
        gemm2_bias(gemm2_bias),
        topk_ids(topk_ids),
        topk_weights(topk_weights) {}

  void init(std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
            int64_t tile_tokens_dim, int64_t routing_method_type, bool use_shuffled_weight,
            int64_t weight_layout, ActivationType activation_type, btg::Dtype dtype_act,
            btg::Dtype dtype_weights, int64_t gemm1_bias_type, bool norm_topk_prob = true) {
    // Set data types
    args->mDtypeElt = dtype_act;
    args->mDtypeOut = btg::Dtype::Bfloat16;  // Output is always BF16 for FP4
    args->mUseDeepSeekFp8 = false;           // FP4 doesn't use DeepSeek FP8

    mDtypeAct = dtype_act;
    mDtypeWeights = dtype_weights;

    FusedMoeLauncher::init_common(std::move(args), tile_tokens_dim, routing_method_type,
                                  use_shuffled_weight, weight_layout, activation_type,
                                  gemm1_bias_type, norm_topk_prob);
  }

  void check_routing() const override {
    // First call base class common routing checks
    FusedMoeLauncher::check_routing_common();

    if (routing_input_mode_ == RoutingInputMode::UnpackedPrecomputed) {
      TVM_FFI_ICHECK_EQ(topk_ids.dtype(), dl_int32)
          << "topk_ids must be int32 for unpacked precomputed routing.";
      TVM_FFI_ICHECK(topk_weights.dtype() == dl_bfloat16 || topk_weights.dtype() == dl_float32)
          << "topk_weights must be bfloat16 or float32 for unpacked precomputed routing.";
    }

    if (args->num_fused_shared_experts > 0) {
      int64_t const totalExpertsPerToken = args->top_k + args->num_fused_shared_experts;
      TVM_FFI_ICHECK_EQ(topk_ids.numel(),
                        static_cast<int64_t>(args->num_tokens) * totalExpertsPerToken)
          << "topk_ids must have num_tokens * (top_k + num_fused_shared_experts) elements";
      TVM_FFI_ICHECK_EQ(topk_weights.numel(),
                        static_cast<int64_t>(args->num_tokens) * totalExpertsPerToken)
          << "topk_weights must have num_tokens * (top_k + num_fused_shared_experts) elements";
    }
  }

  void prepare_routing() override {
    FusedMoeLauncher::prepare_routing_common();

    workspace.routing_expert_indexes = static_cast<int*>(const_cast<void*>(topk_ids.data_ptr()));
    workspace.expert_weights = const_cast<void*>(topk_weights.data_ptr());

    args->mDtypeElt = mDtypeAct;
    auto routing_bias_dtype = routing_bias.has_value() ? routing_bias.value().dtype() : dl_bfloat16;
    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;

    auto const routing_logits_dtype =
        routing_logits.has_value() ? routing_logits.value().dtype() : dl_bfloat16;
    mRoutingLogitsDtype =
        routing_logits_dtype == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;

    if (routing_input_mode_ == RoutingInputMode::UnpackedPrecomputed) {
      args->mDtypeExpW =
          topk_weights.dtype() == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;
    }
  }

  void check_moe() const override {
    TVM_FFI_ICHECK(mDtypeAct == btg::Dtype::E2m1 || mDtypeAct == btg::Dtype::Bfloat16 ||
                   mDtypeAct == btg::Dtype::E4m3 || mDtypeAct == btg::Dtype::MxE4m3)
        << "Only E2m1, Bfloat16, MxE4m3 and E4m3 are supported by Fp4 block scale MoE";

    if (mDtypeAct == btg::Dtype::E2m1) {
      TVM_FFI_ICHECK(mDtypeWeights == btg::Dtype::E2m1)
          << "Only E2m1 and MxE2m1 are supported by block scale MoE with E2m1 activation";
      TVM_FFI_ICHECK(hidden_states_scale.has_value())
          << "hidden_states_scale is required for E2m1 activation";
      TVM_FFI_ICHECK(output1_scales_scalar.has_value())
          << "output1_scales_scalar is required for E2m1 activation";
      TVM_FFI_ICHECK(output1_scales_gate_scalar.has_value())
          << "output1_scales_gate_scalar is required for E2m1 activation";
      TVM_FFI_ICHECK(output2_scales_scalar.has_value())
          << "output2_scales_scalar is required for E2m1 activation";
    } else if (mDtypeAct == btg::Dtype::Bfloat16 || mDtypeAct == btg::Dtype::E4m3 ||
               mDtypeAct == btg::Dtype::MxE4m3) {
      TVM_FFI_ICHECK(mDtypeWeights == btg::Dtype::MxE2m1)
          << "Only MxE2m1 weights are supported by block scale MoE with Bfloat16, E4m3 or "
             "MxE4m3 activation";
    }

    if (mDtypeAct == btg::Dtype::E4m3) {
      TVM_FFI_ICHECK(output1_scales_scalar.has_value())
          << "output1_scales_scalar is required for E4m3 activation";
      TVM_FFI_ICHECK(output1_scales_gate_scalar.has_value())
          << "output1_scales_gate_scalar is required for E4m3 activation";
      TVM_FFI_ICHECK(output2_scales_scalar.has_value())
          << "output2_scales_scalar is required for E4m3 activation";
    }

    TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_uint8) << "gemm1_weights must be byte.";
    TVM_FFI_ICHECK_EQ(gemm1_weights_scale.dtype(), dl_float8_e4m3fn)
        << "gemm1_weights_scale must be fp8.";
    TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_uint8) << "gemm2_weights must be byte.";
    TVM_FFI_ICHECK_EQ(gemm2_weights_scale.dtype(), dl_float8_e4m3fn)
        << "gemm2_weights_scale must be fp8.";

    if (args->num_fused_shared_experts > 0) {
      int64_t const totalLocalExperts = args->local_num_experts + args->num_fused_shared_experts;
      TVM_FFI_ICHECK_EQ(gemm1_weights.size(0), totalLocalExperts)
          << "gemm1 weights dim 0 must be local_num_experts + num_fused_shared_experts.";
      TVM_FFI_ICHECK_EQ(gemm2_weights.size(0), totalLocalExperts)
          << "gemm2 weights dim 0 must be local_num_experts + num_fused_shared_experts.";

      auto check_expert_major = [&](Optional<TensorView> const& tensor, char const* name,
                                    int32_t expected_ndim) {
        if (!tensor.has_value()) {
          return;
        }
        TVM_FFI_ICHECK_EQ(tensor.value().ndim(), expected_ndim)
            << name << " must be " << expected_ndim << "D.";
        TVM_FFI_ICHECK_EQ(tensor.value().size(0), totalLocalExperts)
            << name << " dim 0 must be local_num_experts + num_fused_shared_experts.";
      };
      check_expert_major(output1_scales_scalar, "output1_scales_scalar", 1);
      check_expert_major(output1_scales_gate_scalar, "output1_scales_gate_scalar", 1);
      check_expert_major(output2_scales_scalar, "output2_scales_scalar", 1);
      check_expert_major(gemm1_bias, "gemm1_bias", 2);
      check_expert_major(gemm2_bias, "gemm2_bias", 2);

      check_optional_per_expert_float_tensor(gemm1_alpha, "gemm1_alpha");
      check_optional_per_expert_float_tensor(gemm1_beta, "gemm1_beta");
      check_optional_per_expert_float_tensor(gemm1_clamp_limit, "gemm1_clamp_limit");
    }
  }

  /** Bind the exact FP4 launcher inputs to the common typed runner arguments. */
  void configure_moe_args() {
    args->hidden_states = hidden_states.data_ptr();
    args->hidden_states_scale =
        hidden_states_scale.has_value() ? hidden_states_scale.value().data_ptr() : nullptr;
    args->gemm1_weights = gemm1_weights.data_ptr();
    args->gemm1_weights_scale = gemm1_weights_scale.data_ptr();
    args->gemm1_alpha =
        gemm1_alpha.has_value() ? static_cast<float*>(gemm1_alpha.value().data_ptr()) : nullptr;
    args->gemm1_beta =
        gemm1_beta.has_value() ? static_cast<float*>(gemm1_beta.value().data_ptr()) : nullptr;
    args->gemm1_clamp_limit = gemm1_clamp_limit.has_value()
                                  ? static_cast<float*>(gemm1_clamp_limit.value().data_ptr())
                                  : nullptr;
    args->gemm2_weights = gemm2_weights.data_ptr();
    args->gemm2_weights_scale = gemm2_weights_scale.data_ptr();
    args->gemm1_bias =
        gemm1_bias.has_value() ? static_cast<float*>(gemm1_bias.value().data_ptr()) : nullptr;
    args->gemm2_bias =
        gemm2_bias.has_value() ? static_cast<float*>(gemm2_bias.value().data_ptr()) : nullptr;
    args->output1_scales_scalar =
        output1_scales_scalar.has_value()
            ? static_cast<float*>(output1_scales_scalar.value().data_ptr())
            : nullptr;
    args->output1_scales_gate_scalar =
        output1_scales_gate_scalar.has_value()
            ? static_cast<float*>(output1_scales_gate_scalar.value().data_ptr())
            : nullptr;
    args->output2_scales_scalar =
        output2_scales_scalar.has_value()
            ? static_cast<float*>(output2_scales_scalar.value().data_ptr())
            : nullptr;
  }

  /** Allocate and bind one ordinary FP4 MoE body workspace. */
  void prepare_moe(int64_t& moe_tactic) override {
    configure_moe_args();

    FusedMoeLauncher::prepare_moe_common(moe_tactic);

    auto const sf_vec_size = mDtypeWeights == btg::Dtype::MxE2m1 ? 32 : 16;

    max_num_padded_tokens_gemm1 =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
            workspace.total_max_padded_tokens, args->intermediate_size,
            btg::dtypeGetNumBits(mDtypeAct));
    max_num_padded_tokens_gemm2 =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::maybeGetMinTokenCount(
            workspace.total_max_padded_tokens, args->hidden_size,
            btg::dtypeGetNumBits(btg::Dtype::Bfloat16));  // Output is always BF16

    auto const gemm1_output_hidden =
        mDtypeAct == btg::Dtype::E2m1 ? args->intermediate_size / 2 : args->intermediate_size;
    if (mDtypeAct == btg::Dtype::E2m1 || mDtypeAct == btg::Dtype::MxE4m3) {
      int64_t sf_size = tensorrt_llm::computeSwizzledLayoutSFSize(
          max_num_padded_tokens_gemm1, args->intermediate_size / sf_vec_size);
      gemm1_output_scale = alloc_tensor({sf_size}, dl_uint8, hidden_states.device());
    }
    if (!per_token_scales.has_value()) {
      gemm1_output = alloc_tensor({max_num_padded_tokens_gemm1, gemm1_output_hidden},
                                  mDtypeAct == btg::Dtype::Bfloat16 ? dl_bfloat16 : dl_uint8,
                                  hidden_states.device());
    } else {  // FC1 output is Bfloat16
      TVM_FFI_ICHECK(mDtypeAct == btg::Dtype::E2m1)
          << "NvFP4 MoE: currently only support NvFP4 x NvFP4 when using per-token scaling.";
      // When per-token scales are used, the FC1 output is always BF16 and will be quantized
      gemm1_output = alloc_tensor({max_num_padded_tokens_gemm1, args->intermediate_size},
                                  dl_bfloat16, hidden_states.device());
      // The per-token NvFP4 quant needs to stage the output for running the explicit quant kernel
      activation_output = alloc_tensor({max_num_padded_tokens_gemm1, gemm1_output_hidden}, dl_uint8,
                                       hidden_states.device());
      per_token_scales_fc2 =
          alloc_tensor({max_num_padded_tokens_gemm1}, dl_float32, hidden_states.device());
    }

    // Allocate gemm2_output
    gemm2_output = alloc_tensor({max_num_padded_tokens_gemm2, args->hidden_size}, dl_bfloat16,
                                hidden_states.device());

    // Setup workspace pointers
    workspace.hidden_states_scale_linear = nullptr;  // FP4 doesn't use linear scale
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm1_output_scale = gemm1_output_scale.has_value()
                                       ? static_cast<float*>(gemm1_output_scale.value().data_ptr())
                                       : nullptr;
    if (per_token_scales.has_value()) {
      workspace.token_scales = per_token_scales.value().data_ptr();
      workspace.activation_output = activation_output.data_ptr();
      workspace.activation_output_scale = workspace.gemm1_output_scale;
      workspace.token_scales_fc2 = per_token_scales_fc2.data_ptr();
    }
    workspace.gemm2_output = gemm2_output.data_ptr();
    workspace.gemm2_output_scale = nullptr;
  }

  /** Allocate the graph-stable buffers for one FP4 body without launching it. */
  FP4DABodyBuffers prepare_da_body(RoutingMetadataBuffers const& routing_metadata,
                                   int64_t moe_tactic) {
    ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
    bind_routing_metadata(routing_metadata);
    args->mDtypeExpW = routing_metadata.expert_weights.dtype() == dl_float32 ? btg::Dtype::Fp32
                                                                             : btg::Dtype::Bfloat16;
    check_moe();
    prepare_moe(moe_tactic);

    Tensor empty = alloc_tensor({0}, dl_uint8, hidden_states.device());
    return {gemm1_output,
            gemm1_output_scale.has_value() ? gemm1_output_scale.value() : empty,
            activation_output.defined() ? activation_output : empty,
            per_token_scales_fc2.defined() ? per_token_scales_fc2 : empty,
            gemm2_output,
            workspace_fc1,
            workspace_fc2};
  }

  /** Launch one FP4 body from precomputed routing metadata and prepared buffers. */
  void run_da_body(RoutingMetadataBuffers const& routing_metadata, FP4DABodyBuffers const& prepared,
                   int64_t moe_tactic, bool enable_pdl) {
    ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
    configure_moe_args();
    bind_routing_metadata(routing_metadata);
    args->mDtypeExpW = routing_metadata.expert_weights.dtype() == dl_float32 ? btg::Dtype::Fp32
                                                                             : btg::Dtype::Bfloat16;
    prepare_moe_runner(moe_tactic);
    prepared.bind(workspace, per_token_scales);

    cudaStream_t stream = get_stream(hidden_states.device());
    moe_runner->run(*args, workspace, hidden_states.device().device_id, stream, moe_tactic,
                    enable_pdl);
  }

  /** Return the launcher-owned finalized output handle used by the existing DA result ABI. */
  Tensor finalized_da_output() const { return output; }

 private:
  Optional<TensorView> hidden_states_scale;
  TensorView gemm1_weights_scale;
  Optional<TensorView> gemm1_alpha;
  Optional<TensorView> gemm1_beta;
  Optional<TensorView> gemm1_clamp_limit;
  TensorView gemm2_weights_scale;
  Optional<TensorView> gemm2_bias;
  int32_t max_num_padded_tokens_gemm1{};
  int32_t max_num_padded_tokens_gemm2{};
  Optional<Tensor> gemm1_output_scale;
  TensorView topk_ids;      // [num_tokens, top_k] - pre-computed or output top-k expert indices
  TensorView topk_weights;  // [num_tokens, top_k] - pre-computed or output top-k routing weights

 public:
  MoeRunResultBuffers run(int64_t moe_tactic, bool enable_pdl = true,
                          bool use_routing_scales_on_input = false, bool use_deep_seek_fp8 = false,
                          bool return_activation_output = false) override {
    ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
    check_routing();
    prepare_routing();

    // Execute routing
    tensorrt_llm::kernels::trtllmgen_moe::Routing::Runner routing_runner(tile_tokens_dim);
    cudaStream_t routing_stream = get_stream(hidden_states.device());

    // Set routing kernel parameters based on mode (see RoutingInputMode enum for documentation)
    int32_t* expert_ids_param = nullptr;   // INPUT: pre-computed expert IDs (Mode 3 only)
    void* expert_weights_param = nullptr;  // INPUT or OUTPUT depending on mode

    switch (routing_input_mode_) {
      case RoutingInputMode::FromLogits:
        // Mode 1: Kernel computes routing, writes weights to expert_weights_param (OUTPUT)
        expert_ids_param = nullptr;
        expert_weights_param = topk_weights.data_ptr();
        break;

      case RoutingInputMode::PackedPrecomputed:
        // Mode 2: Kernel unpacks from topk_ids, writes weights to expert_weights_param (OUTPUT)
        expert_ids_param = nullptr;
        expert_weights_param = topk_weights.data_ptr();
        break;

      case RoutingInputMode::UnpackedPrecomputed:
        // Mode 3: Both are INPUTS, kernel uses them directly
        expert_ids_param = static_cast<int32_t*>(topk_ids.data_ptr());
        expert_weights_param = topk_weights.data_ptr();
        break;
    }

    int16_t* replay_ptr = nullptr;
    if (routing_replay_out.has_value()) {
      replay_ptr = reinterpret_cast<int16_t*>(routing_replay_out.value().data_ptr());
    }

    routing_runner.run(args->routing_logits, args->routing_bias, args->num_tokens,
                       args->num_experts, args->top_k, args->num_fused_shared_experts,
                       args->n_group, args->topk_group, args->local_expert_offset,
                       args->local_num_experts, args->routed_scaling_factor,
                       static_cast<int*>(topk_ids.data_ptr()),
                       static_cast<int*>(expert_count_histogram.data_ptr()),
                       static_cast<int*>(total_num_padded_tokens.data_ptr()),
                       static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr()),
                       static_cast<int*>(permuted_idx_to_expanded_idx_ptr()),
                       static_cast<int*>(permuted_idx_to_token_idx.data_ptr()), expert_ids_param,
                       expert_weights_param, static_cast<int*>(num_tokens_per_expert.data_ptr()),
                       static_cast<int*>(cta_idx_xy_to_batch_idx.data_ptr()),
                       static_cast<int*>(cta_idx_xy_to_mn_limit.data_ptr()),
                       static_cast<int*>(num_non_exiting_ctas.data_ptr()), args->mDtypeElt,
                       mRoutingBiasDtype, use_routing_scales_on_input, use_deep_seek_fp8,
                       static_cast<RoutingMethodType>(routing_method_type), routing_stream,
                       mRoutingLogitsDtype, norm_topk_prob, replay_ptr, enable_pdl);

    check_moe();
    prepare_moe(moe_tactic);

    cudaStream_t moe_stream = get_stream(hidden_states.device());
    moe_runner->run(*args, workspace, hidden_states.device().device_id, moe_stream, moe_tactic,
                    enable_pdl);

    MoeRunResultBuffers result(args->do_finalize, args->do_finalize ? output : gemm2_output);
    if (!args->do_finalize) {
      result.expert_weights = FusedMoeLauncher::expert_weights;
    }
    if (!args->do_finalize || return_activation_output) {
      result.expanded_to_permuted_indices = expanded_idx_to_permuted_idx;
    }
    if (return_activation_output) {
      result.activation_output = gemm1_output;
    }
    return result;
  }

  static Array<Array<int64_t>> getValidConfigs(int64_t top_k, int64_t hidden_size,
                                               int64_t intermediate_size, int64_t num_local_experts,
                                               int64_t num_tokens, int64_t act_type,
                                               btg::Dtype dtype_act, btg::Dtype dtype_weights,
                                               bool use_per_token_scaling,
                                               batchedGemm::gemm::BiasType gemm1_bias_type) {
    Array<Array<int64_t>> valid_configs;

    std::vector<int32_t> tile_sizes = getSupportedTileNums(dtype_act, dtype_weights);
    std::set<int32_t> selected_tile_nums =
        computeSelectedTileN(tile_sizes, num_tokens, top_k, num_local_experts);

    for (int32_t tile_N : selected_tile_nums) {
      auto moe_runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
          dtype_act, dtype_weights,
          false,  // useDeepSeekFp8
          tile_N, static_cast<ActivationType>(act_type),
          /*useShuffledMatrix*/ true,
          /*weight_layout*/ batchedGemm::gemm::MatrixLayout::MajorK,
          /*gemm1BiasType*/ gemm1_bias_type,
          /*usePerTokenScalingGemm1*/ use_per_token_scaling,
          // Match prepare_moe_common(): only NVFP4 uses the explicit
          // per-token scale operand for FC2.
          /*usePerTokenScalingGemm2*/
          use_per_token_scaling && dtype_act == btg::Dtype::E2m1, false, false);

      auto cfgs = moe_runner->getValidConfigIndices(top_k, hidden_size, intermediate_size,
                                                    num_local_experts, num_tokens);

      for (auto cfg : cfgs) {
        valid_configs.push_back({tile_N, cfg});
      }
    }

    return valid_configs;
  }
};

Array<Tensor> trtllm_bf16_moe(
    int64_t routing_input_mode, Optional<TensorView> const& routing_logits,
    Optional<TensorView> const& routing_bias, TensorView const& expert_indices,
    TensorView const& expert_weights, TensorView const& hidden_states,
    TensorView const& gemm1_weights, TensorView const& gemm2_weights,
    Optional<TensorView> const& gemm1_lora_delta, Optional<TensorView> const& gemm1_alpha,
    Optional<TensorView> const& gemm1_beta, Optional<TensorView> const& gemm1_clamp_limit,
    TensorView output, int64_t num_experts, int64_t top_k, Optional<int64_t> n_group,
    Optional<int64_t> topk_group, int64_t intermediate_size, int64_t local_expert_offset,
    int64_t local_num_experts, Optional<double> routed_scaling_factor, int64_t routing_method_type,
    bool use_shuffled_weight, int64_t weight_layout, bool do_finalize, bool enable_pdl,
    Array<int64_t> moe_tactic, int64_t activation_type, bool norm_topk_prob,
    Optional<TensorView> routing_replay_out, Array<Tensor> da_routing_metadata,
    Array<Tensor> da_body_workspace, bool is_da_body_preparation) {
  // Just some basic type validation first and leave more checks to the launcher
  if (routing_logits.has_value()) {
    TVM_FFI_ICHECK(routing_logits.value().dtype() == dl_float32 ||
                   routing_logits.value().dtype() == dl_bfloat16)
        << "BF16 MoE: routing_logits must be bfloat16 or float.";
  }
  TVM_FFI_ICHECK_EQ(hidden_states.dtype(), dl_bfloat16)
      << "BF16 MoE: hidden_states must be bfloat16.";
  TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_bfloat16)
      << "BF16 MoE: gemm1_weights must be bfloat16.";
  TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_bfloat16)
      << "BF16 MoE: gemm2_weights must be bfloat16.";

  if (routing_replay_out.has_value()) {
    validate_routing_replay_out(routing_replay_out.value(), hidden_states, top_k);
  }

  auto const num_tokens = hidden_states.size(0);
  auto const hidden_size = hidden_states.size(1);
  auto const activation = validateAndCastActivationType(activation_type);

  auto const gemm1_bias_type_enum = gemm1_lora_delta.has_value()
                                        ? batchedGemm::gemm::BiasType::Mn
                                        : batchedGemm::gemm::BiasType::None;

  // Calculate supported tile sizes
  std::vector<int32_t> mSupportedTileN(Bf16MoeLauncher::mSupportedTileNums.begin(),
                                       Bf16MoeLauncher::mSupportedTileNums.end());
  // Build launchers for ALL supported tiles (not just the computeSelectedTileN subset)
  // so that autotuner-cached tactics always find their tile_N in the map.
  // Launcher creation is cheap (no GPU allocation until run()), so this is safe.

  // Create a map of launchers for each tile size
  std::unordered_map<int32_t, std::unique_ptr<Bf16MoeLauncher>> launchers_map;

  for (int32_t curr_tile_N : mSupportedTileN) {
    // Create MoE arguments for this launcher
    auto args = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>();
    args->num_tokens = num_tokens;
    args->num_experts = num_experts;
    args->hidden_size = hidden_size;
    args->hidden_size_output = args->hidden_size;
    args->top_k = top_k;
    args->n_group = n_group.value_or(0);
    args->topk_group = topk_group.value_or(0);
    args->routed_scaling_factor = routed_scaling_factor.value_or(1.0);
    args->local_expert_offset = local_expert_offset;
    args->local_num_experts = local_num_experts;
    args->intermediate_size = intermediate_size;
    args->do_finalize = do_finalize;
    args->output = output.data_ptr();
    args->output_scale = nullptr;

    // Create and initialize launcher for this tile size
    auto launcher = std::make_unique<Bf16MoeLauncher>(
        routing_logits, routing_bias, expert_indices, expert_weights, hidden_states, gemm1_weights,
        gemm2_weights, gemm1_lora_delta, gemm1_alpha, gemm1_beta, gemm1_clamp_limit,
        static_cast<RoutingInputMode>(routing_input_mode));
    launcher->init(std::move(args), curr_tile_N, routing_method_type, use_shuffled_weight,
                   weight_layout, activation, static_cast<int64_t>(gemm1_bias_type_enum),
                   norm_topk_prob);
    launcher->set_routing_replay_out(routing_replay_out);

    launchers_map[curr_tile_N] = std::move(launcher);
  }

  auto const [tile_N, config] =
      resolveMoeTileAndConfig(moe_tactic, mSupportedTileN, num_tokens, top_k, local_num_experts);

  // Get the launcher for the selected tile_N
  auto launcher_it = launchers_map.find(static_cast<int32_t>(tile_N));
  FLASHINFER_CHECK(launcher_it != launchers_map.end(),
                   "Internal error: missing BF16 MoE launcher for tile_N=", tile_N);
  auto& selected_launcher = launcher_it->second;

  if (is_da_body_preparation) {
    TVM_FFI_ICHECK_EQ(da_body_workspace.size(), 0)
        << "DA preparation cannot consume an existing body workspace.";
    auto const routing = RoutingMetadataBuffers::from_ffi(da_routing_metadata);
    return selected_launcher->prepare_da_body(routing, config).to_ffi();
  }
  if (!da_routing_metadata.empty()) {
    auto const routing = RoutingMetadataBuffers::from_ffi(da_routing_metadata);
    auto const body = BF16DABodyBuffers::from_ffi(da_body_workspace);
    selected_launcher->run_da_body(routing, body, config, enable_pdl);
    return {};
  }
  TVM_FFI_ICHECK_EQ(da_body_workspace.size(), 0)
      << "Ordinary BF16 MoE cannot consume a DA body workspace.";

  // Run the launcher - it will create its own runner internally
  return selected_launcher
      ->run(config, enable_pdl,
            /*use_routing_scales_on_input=*/false,
            /*use_deep_seek_fp8=*/false, gemm1_lora_delta.has_value())
      .to_ffi();
}

Array<Tensor> trtllm_fp8_per_tensor_scale_moe(
    TensorView routing_logits, Optional<TensorView> routing_bias, TensorView hidden_states,
    TensorView gemm1_weights, TensorView output1_scales_scalar,
    TensorView output1_scales_gate_scalar, TensorView gemm2_weights,
    TensorView output2_scales_scalar, TensorView output, int64_t num_experts, int64_t top_k,
    Optional<int64_t> n_group, Optional<int64_t> topk_group, int64_t intermediate_size,
    int64_t local_expert_offset, int64_t local_num_experts, Optional<double> routed_scaling_factor,
    bool use_routing_scales_on_input, int64_t routing_method_type, bool do_finalize,
    bool enable_pdl, Array<int64_t> config_index, int64_t activation_type, bool norm_topk_prob,
    Optional<TensorView> routing_replay_out, Array<Tensor> da_routing_metadata,
    Array<Tensor> da_body_workspace, bool is_da_body_preparation) {
  // Basic type validation
  auto dtype = hidden_states.dtype();
  auto activation = validateAndCastActivationType(activation_type);

  TVM_FFI_ICHECK(dtype == dl_float8_e4m3fn || dtype == dl_float16 || dtype == dl_bfloat16)
      << "FP8 MoE: hidden_states must be float8_e4m3fn, float16, or bfloat16.";
  TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_float8_e4m3fn)
      << "FP8 MoE: gemm1_weights must be float8_e4m3fn.";
  TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_float8_e4m3fn)
      << "FP8 MoE: gemm2_weights must be float8_e4m3fn.";
  TVM_FFI_ICHECK_EQ(output1_scales_scalar.dtype(), dl_float32)
      << "FP8 MoE: output1_scales_scalar must be float32.";
  TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar.dtype(), dl_float32)
      << "FP8 MoE: output1_scales_gate_scalar must be float32.";
  TVM_FFI_ICHECK_EQ(output2_scales_scalar.dtype(), dl_float32)
      << "FP8 MoE: output2_scales_scalar must be float32.";

  if (routing_replay_out.has_value()) {
    validate_routing_replay_out(routing_replay_out.value(), hidden_states, top_k);
  }

  auto const num_tokens = hidden_states.size(0);
  auto const hidden_size = hidden_states.size(1);

  // Use default values that match the original function behavior
  bool use_shuffled_weight = true;  // Original uses /*useShuffledMatrix*/ true
  int64_t weight_layout = 0;        // Default to MajorK

  // Calculate supported tile sizes
  std::vector<int32_t> mSupportedTileN(Fp8PerTensorLauncher::mSupportedTileNums.begin(),
                                       Fp8PerTensorLauncher::mSupportedTileNums.end());
  // Build launchers for ALL supported tiles so autotuner-cached tactics always find their tile_N.

  // Create a map of launchers for each tile size
  std::unordered_map<int32_t, std::unique_ptr<Fp8PerTensorLauncher>> launchers_map;

  for (int32_t curr_tile_N : mSupportedTileN) {
    // Create MoE arguments for this launcher
    auto args = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>();
    args->num_tokens = num_tokens;
    args->num_experts = num_experts;
    args->hidden_size = hidden_size;
    args->hidden_size_output = args->hidden_size;
    args->top_k = top_k;
    args->n_group = n_group.value_or(0);
    args->topk_group = topk_group.value_or(0);
    args->local_expert_offset = local_expert_offset;
    args->local_num_experts = local_num_experts;
    args->intermediate_size = intermediate_size;
    args->routed_scaling_factor = routed_scaling_factor.value_or(1.0);
    args->do_finalize = do_finalize;
    args->output = output.data_ptr();
    args->output_scale = nullptr;

    // Create and initialize launcher for this tile size
    auto launcher = std::make_unique<Fp8PerTensorLauncher>(
        Optional<TensorView>(routing_logits), routing_bias, hidden_states, gemm1_weights,
        output1_scales_scalar, output1_scales_gate_scalar, gemm2_weights, output2_scales_scalar);
    launcher->init(std::move(args), curr_tile_N, routing_method_type, use_shuffled_weight,
                   weight_layout, use_routing_scales_on_input, activation, norm_topk_prob);
    launcher->set_routing_replay_out(routing_replay_out);

    launchers_map[curr_tile_N] = std::move(launcher);
  }

  auto const [tile_N, config] =
      resolveMoeTileAndConfig(config_index, mSupportedTileN, num_tokens, top_k, local_num_experts);

  // Get the launcher for the selected tile_N
  auto launcher_it = launchers_map.find(static_cast<int32_t>(tile_N));
  FLASHINFER_CHECK(launcher_it != launchers_map.end(),
                   "Internal error: missing FP8 per-tensor MoE launcher for tile_N=", tile_N);
  auto& selected_launcher = launcher_it->second;

  if (is_da_body_preparation) {
    TVM_FFI_ICHECK_EQ(da_body_workspace.size(), 0)
        << "DA preparation cannot consume an existing body workspace.";
    auto const routing = RoutingMetadataBuffers::from_ffi(da_routing_metadata);
    return selected_launcher->prepare_da_body(routing, config).to_ffi();
  }
  if (!da_routing_metadata.empty()) {
    auto const routing = RoutingMetadataBuffers::from_ffi(da_routing_metadata);
    auto const body = FP8PerTensorDABodyBuffers::from_ffi(da_body_workspace);
    selected_launcher->run_da_body(routing, body, config, enable_pdl);
    return {};
  }
  TVM_FFI_ICHECK_EQ(da_body_workspace.size(), 0)
      << "Ordinary FP8 per-tensor MoE cannot consume a DA body workspace.";

  // Run the launcher - it will create its own runner internally
  return selected_launcher->run(config, enable_pdl, use_routing_scales_on_input).to_ffi();
}

Array<Tensor> trtllm_fp8_per_tensor_scale_routed_moe(
    int64_t routing_input_mode, TensorView expert_indices, TensorView expert_weights,
    Optional<TensorView> routing_bias, TensorView hidden_states, TensorView gemm1_weights,
    TensorView output1_scales_scalar, TensorView output1_scales_gate_scalar,
    TensorView gemm2_weights, TensorView output2_scales_scalar, TensorView output,
    int64_t num_experts, int64_t top_k, Optional<int64_t> n_group, Optional<int64_t> topk_group,
    int64_t intermediate_size, int64_t local_expert_offset, int64_t local_num_experts,
    Optional<double> routed_scaling_factor, bool use_routing_scales_on_input,
    int64_t routing_method_type, bool do_finalize, bool enable_pdl, Array<int64_t> config_index,
    int64_t activation_type, bool norm_topk_prob, Optional<TensorView> routing_replay_out,
    Array<Tensor> da_routing_metadata, Array<Tensor> da_body_workspace,
    bool is_da_body_preparation) {
  // Basic type validation
  auto const dtype = hidden_states.dtype();
  auto const activation = validateAndCastActivationType(activation_type);

  TVM_FFI_ICHECK_EQ(expert_indices.dtype(), dl_int32) << "FP8 MoE: expert_indices must be int32.";
  TVM_FFI_ICHECK(expert_indices.device().device_type == kDLCUDA)
      << "FP8 MoE: expert_indices must be a CUDA tensor.";
  TVM_FFI_ICHECK(expert_indices.device().device_id == hidden_states.device().device_id)
      << "FP8 MoE: expert_indices must be on the same device as hidden_states.";
  TVM_FFI_ICHECK_EQ(expert_indices.ndim(), 2)
      << "FP8 MoE: expert_indices must be 2D [num_tokens, top_k].";
  TVM_FFI_ICHECK_EQ(expert_indices.size(0), hidden_states.size(0))
      << "FP8 MoE: expert_indices and hidden_states must have the same number of tokens.";
  TVM_FFI_ICHECK_EQ(expert_indices.size(1), top_k)
      << "FP8 MoE: expert_indices dim1 must match top_k.";
  TVM_FFI_ICHECK(expert_indices.IsContiguous()) << "FP8 MoE: expert_indices must be contiguous.";
  TVM_FFI_ICHECK(dtype == dl_float8_e4m3fn || dtype == dl_float16 || dtype == dl_bfloat16)
      << "FP8 MoE: hidden_states must be float8_e4m3fn, float16, or bfloat16.";
  TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_float8_e4m3fn)
      << "FP8 MoE: gemm1_weights must be float8_e4m3fn.";
  TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_float8_e4m3fn)
      << "FP8 MoE: gemm2_weights must be float8_e4m3fn.";
  TVM_FFI_ICHECK_EQ(output1_scales_scalar.dtype(), dl_float32)
      << "FP8 MoE: output1_scales_scalar must be float32.";
  TVM_FFI_ICHECK_EQ(output1_scales_gate_scalar.dtype(), dl_float32)
      << "FP8 MoE: output1_scales_gate_scalar must be float32.";
  TVM_FFI_ICHECK_EQ(output2_scales_scalar.dtype(), dl_float32)
      << "FP8 MoE: output2_scales_scalar must be float32.";

  if (routing_replay_out.has_value()) {
    validate_routing_replay_out(routing_replay_out.value(), hidden_states, top_k);
  }

  auto const num_tokens = hidden_states.size(0);
  auto const hidden_size = hidden_states.size(1);

  // Use default values that match the original function behavior
  bool const use_shuffled_weight = true;  // Original uses /*useShuffledMatrix*/ true
  int64_t const weight_layout = 0;        // Default to MajorK

  // Calculate supported tile sizes
  std::vector<int32_t> mSupportedTileN(Fp8PerTensorLauncher::mSupportedTileNums.begin(),
                                       Fp8PerTensorLauncher::mSupportedTileNums.end());
  // Build launchers for ALL supported tiles so autotuner-cached tactics always find their tile_N.

  // Create a map of launchers for each tile size
  std::unordered_map<int32_t, std::unique_ptr<Fp8PerTensorLauncher>> launchers_map;

  for (int32_t curr_tile_N : mSupportedTileN) {
    // Create MoE arguments for this launcher
    auto args = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>();
    args->num_tokens = num_tokens;
    args->num_experts = num_experts;
    args->hidden_size = hidden_size;
    args->hidden_size_output = args->hidden_size;
    args->top_k = top_k;
    args->n_group = n_group.value_or(0);
    args->topk_group = topk_group.value_or(0);
    args->local_expert_offset = local_expert_offset;
    args->local_num_experts = local_num_experts;
    args->intermediate_size = intermediate_size;
    args->routed_scaling_factor = routed_scaling_factor.value_or(1.0);
    args->do_finalize = do_finalize;
    args->output = output.data_ptr();
    args->output_scale = nullptr;

    auto launcher = std::make_unique<Fp8PerTensorLauncher>(
        Optional<TensorView>(), routing_bias, hidden_states, gemm1_weights, output1_scales_scalar,
        output1_scales_gate_scalar, gemm2_weights, output2_scales_scalar,
        Optional<TensorView>(expert_indices), Optional<TensorView>(expert_weights),
        static_cast<RoutingInputMode>(routing_input_mode));
    launcher->init(std::move(args), curr_tile_N, routing_method_type, use_shuffled_weight,
                   weight_layout, use_routing_scales_on_input, activation, norm_topk_prob);
    launcher->set_routing_replay_out(routing_replay_out);

    launchers_map[curr_tile_N] = std::move(launcher);
  }

  auto const [tile_N, config] =
      resolveMoeTileAndConfig(config_index, mSupportedTileN, num_tokens, top_k, local_num_experts);

  // Get the launcher for the selected tile_N
  auto launcher_it = launchers_map.find(static_cast<int32_t>(tile_N));
  FLASHINFER_CHECK(launcher_it != launchers_map.end(),
                   "Internal error: missing FP8 per-tensor MoE launcher for tile_N=", tile_N);
  auto& selected_launcher = launcher_it->second;

  if (is_da_body_preparation) {
    TVM_FFI_ICHECK_EQ(da_body_workspace.size(), 0)
        << "DA preparation cannot consume an existing body workspace.";
    auto const routing = RoutingMetadataBuffers::from_ffi(da_routing_metadata);
    return selected_launcher->prepare_da_body(routing, config).to_ffi();
  }
  if (!da_routing_metadata.empty()) {
    auto const routing = RoutingMetadataBuffers::from_ffi(da_routing_metadata);
    auto const body = FP8PerTensorDABodyBuffers::from_ffi(da_body_workspace);
    selected_launcher->run_da_body(routing, body, config, enable_pdl);
    return {};
  }
  TVM_FFI_ICHECK_EQ(da_body_workspace.size(), 0)
      << "Ordinary routed FP8 per-tensor MoE cannot consume a DA body workspace.";

  // Run the launcher - it will create its own runner internally
  return selected_launcher->run(config, enable_pdl, use_routing_scales_on_input).to_ffi();
}

Array<Tensor> trtllm_fp8_per_channel_scale_moe(
    Optional<TensorView> routing_logits, TensorView expert_indices, TensorView expert_weights,
    Optional<TensorView> routing_bias, TensorView hidden_states, TensorView hidden_states_scale,
    TensorView gemm1_weights, TensorView gemm1_per_channel_weight_scale,
    TensorView output1_scales_scalar, TensorView output1_scales_gate_scalar,
    TensorView gemm2_weights, TensorView gemm2_per_channel_weight_scale,
    TensorView output2_scales_scalar, TensorView output, int64_t num_experts, int64_t top_k,
    Optional<int64_t> n_group, Optional<int64_t> topk_group, int64_t intermediate_size,
    int64_t local_expert_offset, int64_t local_num_experts, Optional<double> routed_scaling_factor,
    bool use_routing_scales_on_input, int64_t routing_method_type, bool do_finalize,
    bool enable_pdl, Array<int64_t> config_index, int64_t activation_type, bool norm_topk_prob) {
  auto const activation = validateAndCastActivationType(activation_type);
  auto const num_tokens = hidden_states.size(0);
  auto const hidden_size = hidden_states.size(1);
  bool constexpr use_shuffled_weight = true;
  int64_t constexpr weight_layout = 0;

  std::vector<int32_t> supported_tile_nums(Fp8PerChannelLauncher::mSupportedTileNums.begin(),
                                           Fp8PerChannelLauncher::mSupportedTileNums.end());
  auto const [tile_N, config] = resolveMoeTileAndConfig(config_index, supported_tile_nums,
                                                        num_tokens, top_k, local_num_experts);

  auto args = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>();
  args->num_tokens = num_tokens;
  args->num_experts = num_experts;
  args->hidden_size = hidden_size;
  args->hidden_size_output = hidden_size;
  args->top_k = top_k;
  args->n_group = n_group.value_or(0);
  args->topk_group = topk_group.value_or(0);
  args->local_expert_offset = local_expert_offset;
  args->local_num_experts = local_num_experts;
  args->intermediate_size = intermediate_size;
  args->routed_scaling_factor = routed_scaling_factor.value_or(1.0);
  args->do_finalize = do_finalize;
  args->output = output.data_ptr();

  auto launcher = std::make_unique<Fp8PerChannelLauncher>(
      routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights,
      gemm1_per_channel_weight_scale, output1_scales_scalar, output1_scales_gate_scalar,
      gemm2_weights, gemm2_per_channel_weight_scale, output2_scales_scalar, expert_indices,
      expert_weights);
  launcher->init(std::move(args), tile_N, routing_method_type, use_shuffled_weight, weight_layout,
                 use_routing_scales_on_input, activation, norm_topk_prob);

  return launcher->run(config, enable_pdl, use_routing_scales_on_input).to_ffi();
}

Array<Tensor> trtllm_fp8_block_scale_moe(
    int64_t routing_input_mode, Optional<TensorView> routing_logits, TensorView expert_indices,
    TensorView expert_weights, Optional<TensorView> routing_bias, TensorView hidden_states,
    TensorView hidden_states_scale, TensorView gemm1_weights, TensorView gemm1_weights_scale,
    Optional<TensorView> gemm1_lora_delta, Optional<TensorView> gemm1_alpha,
    Optional<TensorView> gemm1_beta, Optional<TensorView> gemm1_clamp_limit,
    TensorView gemm2_weights, TensorView gemm2_weights_scale, TensorView output,
    int64_t num_experts, int64_t top_k, Optional<int64_t> num_fused_shared_experts,
    Optional<int64_t> n_group, Optional<int64_t> topk_group, int64_t intermediate_size,
    int64_t local_expert_offset, int64_t local_num_experts, Optional<double> routed_scaling_factor,
    int64_t routing_method_type, bool use_shuffled_weight, int64_t weight_layout, bool do_finalize,
    bool enable_pdl, Array<int64_t> config_index, Fp8QuantizationType quantization_type,
    int64_t act_type, bool norm_topk_prob, Optional<TensorView> routing_replay_out,
    Array<Tensor> da_routing_metadata, Array<Tensor> da_body_workspace,
    bool is_da_body_preparation) {
  auto activation_type = validateAndCastActivationType(act_type);
  validateFp8BlockScaleGemm1ActivationParams(gemm1_alpha, gemm1_beta, gemm1_clamp_limit,
                                             quantization_type, activation_type);
  // DeepSeekFp8 currently uses a TRTLLM runner that hardwires Swiglu activation semantics.
  // Fail for any other activation to avoid silently running incorrect activation behavior.
  if (quantization_type == Fp8QuantizationType::DeepSeekFp8 &&
      activation_type != ActivationType::Swiglu) {
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "DeepSeekFp8 only supports ActivationType::Swiglu in this runner path. "
        << "Received activation_type=" << static_cast<int>(activation_type);
  }

  // Basic type validation
  auto dtype = hidden_states.dtype();

  // Either routing_logits or expert_indices must be provided
  // expert_indices is a packed tensor: (expert_id << 16) | (weight_bf16.view(int16))
  bool use_routing_logits = routing_logits.has_value();
  // Check ndim==2 and size>0 because empty placeholder tensors may have non-null data_ptr
  bool use_precomputed_routing = expert_indices.ndim() == 2 && expert_indices.size(0) > 0;

  TVM_FFI_ICHECK(use_routing_logits || use_precomputed_routing)
      << "Either routing_logits or expert_indices must be provided.";

  (void)use_routing_logits;
  TVM_FFI_ICHECK(dtype == dl_float16 || dtype == dl_bfloat16 || dtype == dl_float8_e4m3fn)
      << "FP8 block scale MoE: hidden_states must be fp16, bf16, or fp8.";
  if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
    TVM_FFI_ICHECK_EQ(hidden_states_scale.dtype(), dl_float32)
        << "FP8 block scale MoE: hidden_states_scale must be float32.";
  } else if (quantization_type == Fp8QuantizationType::MxFp8) {
    TVM_FFI_ICHECK_EQ(hidden_states_scale.dtype(), dl_uint8)
        << "FP8 block scale MoE: hidden_states_scale must be uint8.";
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "trtllm_fp8_block_scale_moe only supports DeepSeekFp8 or MxFp8.";
  }
  TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_float8_e4m3fn)
      << "FP8 block scale MoE: gemm1_weights must be fp8.";
  TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_float8_e4m3fn)
      << "FP8 block scale MoE: gemm2_weights must be fp8.";
  if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
    TVM_FFI_ICHECK_EQ(gemm1_weights_scale.dtype(), dl_float32)
        << "FP8 block scale MoE: gemm1_weights_scale must be float32.";
    TVM_FFI_ICHECK_EQ(gemm2_weights_scale.dtype(), dl_float32)
        << "FP8 block scale MoE: gemm2_weights_scale must be float32.";
  } else if (quantization_type == Fp8QuantizationType::MxFp8) {
    TVM_FFI_ICHECK_EQ(gemm1_weights_scale.dtype(), dl_uint8)
        << "FP8 block scale MoE: gemm1_weights_scale must be uint8.";
    TVM_FFI_ICHECK_EQ(gemm2_weights_scale.dtype(), dl_uint8)
        << "FP8 block scale MoE: gemm2_weights_scale must be uint8.";
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "trtllm_fp8_block_scale_moe only supports DeepSeekFp8 or MxFp8.";
  }

  if (quantization_type == Fp8QuantizationType::MxFp8) {
    TVM_FFI_ICHECK(use_shuffled_weight) << "use_shuffled_weight must be true for MxFp8.";
    TVM_FFI_ICHECK(weight_layout == 0) << "weight_layout must be 0 for MxFp8.";
  }

  if (routing_replay_out.has_value()) {
    // Replay records at stride top_k + nfse, mismatching the [num_tokens, top_k] layout.
    TVM_FFI_ICHECK(num_fused_shared_experts.value_or(0) == 0)
        << "routing_replay_out is not supported with num_fused_shared_experts > 0";
    validate_routing_replay_out(routing_replay_out.value(), hidden_states, top_k);
  }

  auto const num_tokens = hidden_states.size(0);
  auto const hidden_size = hidden_states.size(1);
  auto const gemm1_bias_type_enum = gemm1_lora_delta.has_value()
                                        ? batchedGemm::gemm::BiasType::Mn
                                        : batchedGemm::gemm::BiasType::None;
  auto const dtype_act =
      quantization_type == Fp8QuantizationType::DeepSeekFp8 ? btg::Dtype::E4m3 : btg::Dtype::MxE4m3;
  auto const dtype_weights =
      quantization_type == Fp8QuantizationType::DeepSeekFp8 ? btg::Dtype::E4m3 : btg::Dtype::MxE4m3;

  int64_t const nFusedShared = num_fused_shared_experts.value_or(0);
  int64_t const totalExpertsPerToken = top_k + nFusedShared;
  int64_t const totalLocalExperts = local_num_experts + nFusedShared;

  auto supported_tile_nums = Fp8BlockScaleLauncher::getSupportedTileNums(quantization_type);
  // Build launchers for ALL supported tiles so autotuner-cached tactics always find their tile_N.

  // Create a map of launchers for each tile size
  std::unordered_map<int32_t, std::unique_ptr<Fp8BlockScaleLauncher>> launchers_map;

  for (int32_t curr_tile_N : supported_tile_nums) {
    // Create MoE arguments for this launcher
    auto args = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>();
    args->num_tokens = num_tokens;
    args->num_experts = num_experts;
    args->num_fused_shared_experts = nFusedShared;
    args->hidden_size = hidden_size;
    args->hidden_size_output = args->hidden_size;
    args->top_k = top_k;
    args->n_group = n_group.value_or(0);
    args->topk_group = topk_group.value_or(0);
    args->local_expert_offset = local_expert_offset;
    args->local_num_experts = local_num_experts;
    args->intermediate_size = intermediate_size;
    args->routed_scaling_factor = routed_scaling_factor.value_or(1.0);
    args->do_finalize = do_finalize;
    args->output = output.data_ptr();
    args->output_scale = nullptr;

    // Create and initialize launcher for this tile size
    auto launcher = std::make_unique<Fp8BlockScaleLauncher>(
        routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights,
        gemm1_weights_scale, gemm1_lora_delta, gemm1_alpha, gemm1_beta, gemm1_clamp_limit,
        gemm2_weights, gemm2_weights_scale, expert_indices, expert_weights, quantization_type,
        static_cast<RoutingInputMode>(routing_input_mode));
    launcher->init(std::move(args), curr_tile_N, routing_method_type, use_shuffled_weight,
                   weight_layout, activation_type, static_cast<int64_t>(gemm1_bias_type_enum),
                   norm_topk_prob);
    launcher->set_routing_replay_out(routing_replay_out);

    launchers_map[curr_tile_N] = std::move(launcher);
  }

  // Use the fused totals (routed + shared experts) so the fallback tile/config
  // selection matches prepare_moe(), which validates the chosen tactic against
  // effectiveTopK / effectiveLocalExperts.
  auto const [tile_N, config] = resolveMoeTileAndConfig(
      config_index, supported_tile_nums, num_tokens, totalExpertsPerToken, totalLocalExperts);

  // Get the launcher for the selected tile_N
  auto launcher_it = launchers_map.find(static_cast<int32_t>(tile_N));
  FLASHINFER_CHECK(launcher_it != launchers_map.end(),
                   "Internal error: missing FP8 block-scale MoE launcher for tile_N=", tile_N);
  auto& selected_launcher = launcher_it->second;

  if (is_da_body_preparation) {
    TVM_FFI_ICHECK_EQ(da_body_workspace.size(), 0)
        << "DA preparation cannot consume an existing body workspace.";
    auto const routing = RoutingMetadataBuffers::from_ffi(da_routing_metadata);
    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      return selected_launcher->prepare_deepseek_da_body(routing, config).to_ffi();
    }
    return selected_launcher->prepare_mxfp8_da_body(routing, config).to_ffi();
  }
  if (!da_routing_metadata.empty()) {
    auto const routing = RoutingMetadataBuffers::from_ffi(da_routing_metadata);
    if (quantization_type == Fp8QuantizationType::DeepSeekFp8) {
      auto const body = DeepSeekFP8DABodyBuffers::from_ffi(da_body_workspace);
      selected_launcher->run_deepseek_da_body(routing, body, config, enable_pdl);
    } else {
      auto const body = MXFP8DABodyBuffers::from_ffi(da_body_workspace);
      selected_launcher->run_mxfp8_da_body(routing, body, config, enable_pdl);
    }
    return {};
  }
  TVM_FFI_ICHECK_EQ(da_body_workspace.size(), 0)
      << "Ordinary FP8 block-scale MoE cannot consume a DA body workspace.";

  // Run the launcher with DeepSeek FP8 enabled - it will create its own runner internally
  return selected_launcher
      ->run(config, enable_pdl, false /* use_routing_scales_on_input */,
            quantization_type == Fp8QuantizationType::DeepSeekFp8 /* use_deep_seek_fp8 */,
            gemm1_lora_delta.has_value())
      .to_ffi();
}

Array<Tensor> trtllm_fp4_block_scale_moe(
    int64_t routing_input_mode, Optional<TensorView> routing_logits, TensorView topk_ids,
    TensorView topk_weights, Optional<TensorView> routing_bias, TensorView hidden_states,
    Optional<TensorView> hidden_states_scale, TensorView gemm1_weights,
    TensorView gemm1_weights_scale, Optional<TensorView> gemm1_bias,
    Optional<TensorView> gemm1_lora_delta, Optional<TensorView> gemm1_alpha,
    Optional<TensorView> gemm1_beta, Optional<TensorView> gemm1_clamp_limit,
    TensorView gemm2_weights, TensorView gemm2_weights_scale, Optional<TensorView> gemm2_bias,
    Optional<TensorView> output1_scales_scalar, Optional<TensorView> output1_scales_gate_scalar,
    Optional<TensorView> output2_scales_scalar, Optional<TensorView> per_token_scales,
    int64_t num_experts, int64_t top_k, Optional<int64_t> num_fused_shared_experts,
    Optional<int64_t> n_group, Optional<int64_t> topk_group, int64_t intermediate_size,
    int64_t local_expert_offset, int64_t local_num_experts, Optional<double> routed_scaling_factor,
    int64_t routing_method_type, bool do_finalize, bool enable_pdl, int64_t act_type,
    TensorView output, Array<int64_t> config_index, bool norm_topk_prob,
    Optional<TensorView> routing_replay_out, Array<Tensor> da_routing_metadata,
    Array<Tensor> da_body_workspace, bool is_da_body_preparation) {
  auto const gemm1_bias_type_enum = gemm1_lora_delta.has_value()
                                        ? batchedGemm::gemm::BiasType::Mn
                                        : batchedGemm::gemm::BiasType::None;
  // Determine data types based on input format
  int const num_tokens = hidden_states.size(0);
  int hidden_size = hidden_states.size(1);
  if (hidden_states.dtype() == dl_uint8) hidden_size *= 2;

  int64_t const nFusedShared = num_fused_shared_experts.value_or(0);
  int64_t const totalExpertsPerToken = top_k + nFusedShared;
  int64_t const totalLocalExperts = local_num_experts + nFusedShared;

  int64_t hidden_states_scale_vec_size = -1;
  if (hidden_states_scale.has_value()) {
    hidden_states_scale_vec_size =
        (static_cast<int64_t>(num_tokens) * hidden_size) / hidden_states_scale.value().numel();
  }
  int64_t intermediate_size_factor =
      isGatedActivation(static_cast<ActivationType>(act_type)) ? 2 : 1;
  int64_t logical_scale_count =
      totalLocalExperts * intermediate_size * intermediate_size_factor * hidden_size;
  int64_t weight_scale_vec_size_raw = logical_scale_count / gemm1_weights_scale.numel();

  // Snap to nearest valid sf_vec_size (16 or 32).
  // The raw value may be slightly smaller than the true vec_size because
  // block_scale_interleave pads scale columns to a multiple of 4, inflating numel().
  int64_t weight_scale_vec_size = weight_scale_vec_size_raw > 16 ? 32 : 16;

  // Round-trip validation: the unpadded scale count must not exceed actual numel
  // (padding only adds elements, never removes them).
  int64_t expected_unpadded = logical_scale_count / weight_scale_vec_size;
  TVM_FFI_ICHECK(gemm1_weights_scale.numel() >= expected_unpadded)
      << "weight scale tensor too small: numel=" << gemm1_weights_scale.numel()
      << " but expected at least " << expected_unpadded
      << " for sf_vec_size=" << weight_scale_vec_size;

  auto mDtypeWeights = weight_scale_vec_size == 16 ? btg::Dtype::E2m1 : btg::Dtype::MxE2m1;

  if (routing_bias.has_value()) {
    TVM_FFI_ICHECK(routing_bias.value().dtype() == dl_bfloat16 ||
                   routing_bias.value().dtype() == dl_float32)
        << "routing_bias must be bfloat16 or float.";

    TVM_FFI_ICHECK_EQ(routing_bias.value().ndim(), 1) << "routing_bias must be 1D.";
    TVM_FFI_ICHECK_EQ(routing_bias.value().size(0), num_experts)
        << "routing_bias has incorrect shape.";
  }

  if (routing_replay_out.has_value()) {
    // Replay records at stride top_k + nfse, mismatching the [num_tokens, top_k] layout.
    TVM_FFI_ICHECK(nFusedShared == 0)
        << "routing_replay_out is not supported with num_fused_shared_experts > 0";
    validate_routing_replay_out(routing_replay_out.value(), hidden_states, top_k);
  }

  // Determine activation type
  TVM_FFI_ICHECK(gemm1_weights.dtype() == dl_uint8 && gemm2_weights.dtype() == dl_uint8)
      << "weights must be fp4 packed in uint8.";
  TVM_FFI_ICHECK(hidden_states.dtype() == dl_uint8 || hidden_states.dtype() == dl_bfloat16 ||
                 hidden_states.dtype() == dl_float8_e4m3fn)
      << "hidden_states must be bf16, fp8 or uint8 (packed fp4).";

  auto mDtypeAct = btg::Dtype::Bfloat16;
  if (hidden_states.dtype() == dl_uint8) {
    TVM_FFI_ICHECK(hidden_states_scale.has_value() &&
                   hidden_states_scale.value().dtype() == dl_float8_e4m3fn)
        << "hidden_states_scale must be provided for fp4 activation.";
    if (hidden_states_scale_vec_size == 16) {
      mDtypeAct = btg::Dtype::E2m1;
    } else if (hidden_states_scale_vec_size == 32) {
      mDtypeAct = btg::Dtype::MxE2m1;
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported hidden state scale shape.";
    }
  } else if (hidden_states.dtype() == dl_float8_e4m3fn) {
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

  // Determine supported tile sizes
  std::vector<int32_t> mSupportedTileN =
      FP4BlockScaleLauncher::getSupportedTileNums(mDtypeAct, mDtypeWeights);
  // Build launchers for ALL supported tiles so autotuner-cached tactics always find their tile_N.

  // Create a map of launchers for each tile size
  std::unordered_map<int32_t, std::unique_ptr<FP4BlockScaleLauncher>> launchers_map;

  for (int32_t curr_tile_N : mSupportedTileN) {
    // Create MoE arguments for this launcher
    auto args = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>();
    args->num_tokens = num_tokens;
    args->num_experts = num_experts;
    // For E2m1, hidden_size is already multiplied by 2 above, so use it directly
    args->hidden_size = hidden_size;
    args->hidden_size_output = output.size(1) > 0 ? output.size(1) : hidden_size / 2;
    args->top_k = top_k;
    args->num_fused_shared_experts = nFusedShared;
    args->n_group = n_group.value_or(0);
    args->topk_group = topk_group.value_or(0);
    args->local_expert_offset = local_expert_offset;
    args->local_num_experts = local_num_experts;
    args->intermediate_size = intermediate_size;
    args->routed_scaling_factor = routed_scaling_factor.value_or(1.0);
    args->do_finalize = do_finalize;
    args->output = output.data_ptr();
    args->output_scale = nullptr;

    // gemm1_bias and gemm1_lora_delta are mutually exclusive
    auto const& gemm1_bias_effective = gemm1_lora_delta.has_value() ? gemm1_lora_delta : gemm1_bias;

    // Create and initialize launcher for this tile size
    auto launcher = std::make_unique<FP4BlockScaleLauncher>(
        static_cast<RoutingInputMode>(routing_input_mode), routing_logits, routing_bias,
        hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale,
        gemm1_bias_effective, gemm1_alpha, gemm1_beta, gemm1_clamp_limit, gemm2_weights,
        gemm2_weights_scale, gemm2_bias, output1_scales_scalar, output1_scales_gate_scalar,
        output2_scales_scalar, per_token_scales, topk_ids, topk_weights);
    launcher->init(std::move(args), curr_tile_N, routing_method_type, /*use_shuffled_weight=*/true,
                   /*weight_layout=*/0, static_cast<ActivationType>(act_type), mDtypeAct,
                   mDtypeWeights, static_cast<int64_t>(gemm1_bias_type_enum), norm_topk_prob);
    launcher->set_routing_replay_out(routing_replay_out);

    launchers_map[curr_tile_N] = std::move(launcher);
  }

  auto const [tile_N, config] = resolveMoeTileAndConfig(config_index, mSupportedTileN, num_tokens,
                                                        totalExpertsPerToken, totalLocalExperts);

  // Get the launcher for the selected tile_N
  auto launcher_it = launchers_map.find(static_cast<int32_t>(tile_N));
  FLASHINFER_CHECK(launcher_it != launchers_map.end(),
                   "Internal error: missing FP4 block-scale MoE launcher for tile_N=", tile_N);
  auto& selected_launcher = launcher_it->second;

  if (is_da_body_preparation) {
    TVM_FFI_ICHECK_EQ(da_body_workspace.size(), 0)
        << "DA body preparation does not accept an existing body workspace.";
    auto const routing = RoutingMetadataBuffers::from_ffi(da_routing_metadata);
    return selected_launcher->prepare_da_body(routing, config).to_ffi();
  }
  if (!da_routing_metadata.empty()) {
    auto const routing = RoutingMetadataBuffers::from_ffi(da_routing_metadata);
    auto const body = FP4DABodyBuffers::from_ffi(da_body_workspace);
    selected_launcher->run_da_body(routing, body, config, enable_pdl);
    if (do_finalize) {
      return {selected_launcher->finalized_da_output()};
    }
    return {body.gemm2_output, routing.expert_weights, routing.expanded_idx_to_permuted_idx};
  }
  TVM_FFI_ICHECK_EQ(da_body_workspace.size(), 0)
      << "A DA body workspace requires routing metadata.";

  // Run the launcher - it will create its own runner internally
  return selected_launcher
      ->run(config, enable_pdl,
            /*use_routing_scales_on_input=*/false,
            /*use_deep_seek_fp8=*/false, gemm1_lora_delta.has_value())
      .to_ffi();
}

Array<Tensor> trtllm_mxint4_block_scale_moe(
    Optional<TensorView> const& routing_logits, Optional<TensorView> routing_bias,
    TensorView const& expert_indices, TensorView const& expert_weights, TensorView hidden_states,
    TensorView gemm1_weights, TensorView gemm1_weights_scale, Optional<TensorView> gemm1_alpha,
    Optional<TensorView> gemm1_beta, Optional<TensorView> gemm1_clamp_limit,
    Optional<TensorView> gemm1_lora_delta, TensorView gemm2_weights, TensorView gemm2_weights_scale,
    int64_t num_experts, int64_t top_k, Optional<int64_t> n_group, Optional<int64_t> topk_group,
    int64_t intermediate_size, int64_t local_expert_offset, int64_t local_num_experts,
    Optional<double> routed_scaling_factor, int64_t routing_method_type, bool do_finalize,
    bool enable_pdl, TensorView output, Array<int64_t> config_index, bool norm_topk_prob,
    Optional<TensorView> routing_replay_out, Array<Tensor> da_routing_metadata,
    Array<Tensor> da_body_workspace, bool is_da_body_preparation) {
  // Determine data types based on input format
  int const num_tokens = hidden_states.size(0);
  int hidden_size = hidden_states.size(1);

  auto gemm1_bias_type_enum = gemm1_lora_delta.has_value() ? batchedGemm::gemm::BiasType::Mn
                                                           : batchedGemm::gemm::BiasType::None;

  // Just some basic type validation first and leave more checks to the launcher

  int weight_scale_vec_size =
      (local_num_experts * intermediate_size * 2 * hidden_size) / gemm1_weights_scale.numel();

  TVM_FFI_ICHECK(weight_scale_vec_size == 32) << "unsupported weight_scale_vec_size.";

  if (routing_logits.has_value()) {
    TVM_FFI_ICHECK(routing_logits.value().dtype() == dl_float32 ||
                   routing_logits.value().dtype() == dl_bfloat16)
        << "routing_logits must be float or bfloat16.";
    TVM_FFI_ICHECK_EQ(routing_logits.value().ndim(), 2) << "routing_logits must be 2D.";
    TVM_FFI_ICHECK_EQ(routing_logits.value().size(1), num_experts)
        << "routing_logits has incorrect shape.";
  }
  if (routing_bias.has_value()) {
    TVM_FFI_ICHECK(routing_bias.value().dtype() == dl_bfloat16 ||
                   routing_bias.value().dtype() == dl_float32)
        << "routing_bias must be bfloat16 or float.";
    TVM_FFI_ICHECK_EQ(routing_bias.value().ndim(), 1) << "routing_bias must be 1D.";
    TVM_FFI_ICHECK_EQ(routing_bias.value().size(0), num_experts)
        << "routing_bias has incorrect shape.";
  }

  if (routing_replay_out.has_value()) {
    validate_routing_replay_out(routing_replay_out.value(), hidden_states, top_k);
  }

  // Determine activation type
  TVM_FFI_ICHECK(gemm1_weights.dtype() == dl_uint8 && gemm2_weights.dtype() == dl_uint8)
      << "weights must be int4 packed in uint8.";
  TVM_FFI_ICHECK(hidden_states.dtype() == dl_bfloat16) << "hidden_states must be bf16.";

  // Determine supported tile sizes
  std::vector<int32_t> mSupportedTileN(MxInt4BlockScaleLauncher::mSupportedTileNums.begin(),
                                       MxInt4BlockScaleLauncher::mSupportedTileNums.end());
  // Build launchers for ALL supported tiles so autotuner-cached tactics always find their tile_N.

  // Create a map of launchers for each tile size
  std::unordered_map<int32_t, std::unique_ptr<MxInt4BlockScaleLauncher>> launchers_map;

  for (int32_t curr_tile_N : mSupportedTileN) {
    // Create MoE arguments for this launcher
    auto args = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>();
    args->num_tokens = num_tokens;
    args->num_experts = num_experts;
    // For E2m1, hidden_size is already multiplied by 2 above, so use it directly
    args->hidden_size = hidden_size;
    args->hidden_size_output = args->hidden_size;
    args->top_k = top_k;
    args->n_group = n_group.value_or(0);
    args->topk_group = topk_group.value_or(0);
    args->local_expert_offset = local_expert_offset;
    args->local_num_experts = local_num_experts;
    args->intermediate_size = intermediate_size;
    args->routed_scaling_factor = routed_scaling_factor.value_or(1.0);
    args->do_finalize = do_finalize;
    args->output = output.data_ptr();
    args->output_scale = nullptr;

    // Create and initialize launcher for this tile size
    auto launcher = std::make_unique<MxInt4BlockScaleLauncher>(
        routing_logits, routing_bias, expert_indices, expert_weights, hidden_states, gemm1_weights,
        gemm1_weights_scale, gemm1_alpha, gemm1_beta, gemm1_clamp_limit, gemm1_lora_delta,
        gemm2_weights, gemm2_weights_scale);
    launcher->init(std::move(args), curr_tile_N, routing_method_type,
                   static_cast<int64_t>(gemm1_bias_type_enum), norm_topk_prob);
    launcher->set_routing_replay_out(routing_replay_out);

    launchers_map[curr_tile_N] = std::move(launcher);
  }

  auto const [tile_N, config] =
      resolveMoeTileAndConfig(config_index, mSupportedTileN, num_tokens, top_k, local_num_experts);

  // Get the launcher for the selected tile_N
  auto launcher_it = launchers_map.find(static_cast<int32_t>(tile_N));
  FLASHINFER_CHECK(launcher_it != launchers_map.end(),
                   "Internal error: missing MXINT4 block-scale MoE launcher for tile_N=", tile_N);
  auto& selected_launcher = launcher_it->second;

  if (is_da_body_preparation) {
    TVM_FFI_ICHECK_EQ(da_body_workspace.size(), 0)
        << "DA preparation cannot consume an existing body workspace.";
    auto const routing = RoutingMetadataBuffers::from_ffi(da_routing_metadata);
    return selected_launcher->prepare_da_body(routing, config).to_ffi();
  }
  if (!da_routing_metadata.empty()) {
    auto const routing = RoutingMetadataBuffers::from_ffi(da_routing_metadata);
    auto const body = MXINT4DABodyBuffers::from_ffi(da_body_workspace);
    selected_launcher->run_da_body(routing, body, config, enable_pdl);
    return {};
  }
  TVM_FFI_ICHECK_EQ(da_body_workspace.size(), 0)
      << "Ordinary MXINT4 MoE cannot consume a DA body workspace.";

  // Run the launcher - it will create its own runner internally
  return selected_launcher
      ->run(config, enable_pdl,
            /*use_routing_scales_on_input=*/false,
            /*use_deep_seek_fp8=*/false, gemm1_lora_delta.has_value())
      .to_ffi();
}

Array<Array<int64_t>> trtllm_get_valid_moe_configs(
    int64_t const dtype_act_, int64_t const dtype_weights_,
    Fp8QuantizationType fp8_quantization_type, int64_t const top_k, int64_t const hidden_size,
    int64_t const intermediate_size, int64_t const num_local_experts, int64_t const act_type,
    bool const use_shuffled_weight, int64_t const weight_layout, bool const use_per_token_scaling,
    int64_t const num_tokens, bool has_gemm1_lora_delta) {
  auto activation_type = validateAndCastActivationType(act_type);
  auto dtype_act = static_cast<btg::Dtype>(dtype_act_);
  auto dtype_weights = static_cast<btg::Dtype>(dtype_weights_);
  auto gemm1_bias_type_enum =
      has_gemm1_lora_delta ? batchedGemm::gemm::BiasType::Mn : batchedGemm::gemm::BiasType::None;

  if (dtype_act == btg::Dtype::Bfloat16 && dtype_weights == btg::Dtype::MxInt4) {
    // MxInt4 MoE
    return MxInt4BlockScaleLauncher::getValidConfigs(
        top_k, hidden_size, intermediate_size, num_local_experts, num_tokens, gemm1_bias_type_enum);
  }
  if (dtype_act == btg::Dtype::Bfloat16 && dtype_weights == btg::Dtype::Bfloat16) {
    // BF16 MoE
    return Bf16MoeLauncher::getValidConfigs(
        top_k, hidden_size, intermediate_size, num_local_experts, num_tokens, act_type,
        use_shuffled_weight, weight_layout, gemm1_bias_type_enum);

  } else if (fp8_quantization_type == Fp8QuantizationType::DeepSeekFp8 &&
             dtype_act == btg::Dtype::E4m3 && dtype_weights == btg::Dtype::E4m3) {
    if (activation_type != ActivationType::Swiglu) {
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "DeepSeekFp8 only supports ActivationType::Swiglu, " << "got act_type=" << act_type
          << ".";
    }
    // FP8 block scale (DeepSeek)
    return Fp8BlockScaleLauncher::getValidConfigs(
        top_k, hidden_size, intermediate_size, num_local_experts, num_tokens, use_shuffled_weight,
        weight_layout, dtype_act, dtype_weights, fp8_quantization_type, act_type,
        gemm1_bias_type_enum);
  } else if (fp8_quantization_type == Fp8QuantizationType::PerChannelFp8 &&
             dtype_act == btg::Dtype::E4m3 && dtype_weights == btg::Dtype::E4m3) {
    // FP8 per-channel scale
    return Fp8PerChannelLauncher::getValidConfigs(
        top_k, hidden_size, intermediate_size, num_local_experts, num_tokens, act_type,
        use_shuffled_weight, weight_layout, dtype_act, dtype_weights);
  } else if (fp8_quantization_type == Fp8QuantizationType::MxFp8 &&
             dtype_act == btg::Dtype::MxE4m3 && dtype_weights == btg::Dtype::MxE4m3) {
    // FP8 block scale (MxFp8)
    return Fp8BlockScaleLauncher::getValidConfigs(
        top_k, hidden_size, intermediate_size, num_local_experts, num_tokens, use_shuffled_weight,
        weight_layout, dtype_act, dtype_weights, fp8_quantization_type, act_type,
        gemm1_bias_type_enum);
  } else if ((fp8_quantization_type == Fp8QuantizationType::PerTensorFp8 ||
              fp8_quantization_type == Fp8QuantizationType::NoneFp8) &&
             dtype_weights == btg::Dtype::E4m3) {
    if (has_gemm1_lora_delta) {
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "FP8 per-tensor MoE does not support lora delta";
    }
    if (!isGatedActivation(activation_type)) {
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "FP8 per-tensor currently supports gated activations only, "
          << "got act_type=" << act_type << ".";
    }
    return Fp8PerTensorLauncher::getValidConfigs(
        top_k, hidden_size, intermediate_size, num_local_experts, num_tokens, act_type,
        use_shuffled_weight, weight_layout, dtype_act, dtype_weights, use_per_token_scaling);
  } else if (fp8_quantization_type == Fp8QuantizationType::PerChannelFp8 &&
             dtype_weights == btg::Dtype::E4m3) {
    // FP8 per-channel with bf16/fp16 activations (E4m3/E4m3 case handled above).
    return Fp8PerChannelLauncher::getValidConfigs(
        top_k, hidden_size, intermediate_size, num_local_experts, num_tokens, act_type,
        use_shuffled_weight, weight_layout, dtype_act, dtype_weights);
  } else if (dtype_weights == btg::Dtype::E2m1 || dtype_weights == btg::Dtype::MxE2m1) {
    // FP4 block scale
    return FP4BlockScaleLauncher::getValidConfigs(
        top_k, hidden_size, intermediate_size, num_local_experts, num_tokens, act_type, dtype_act,
        dtype_weights, use_per_token_scaling, gemm1_bias_type_enum);
  }

  TVM_FFI_LOG_AND_THROW(NotImplementedError)
      << "Unsupported data type combination for getValidConfigs: " << "dtype_act="
      << static_cast<int>(dtype_act) << ", dtype_weights=" << static_cast<int>(dtype_weights)
      << ", fp8_quantization_type=" << fp8QuantizationTypeToString(fp8_quantization_type);

  // Unreachable code - added to suppress compiler warning
  return Array<Array<int64_t>>();
}

/// Return valid complete tactics decomposed into tile-N, FC1, FC2, and anchor coordinates.
Array<Array<int64_t>> trtllm_get_valid_moe_factorizations(
    int64_t const dtype_act_, int64_t const dtype_weights_,
    Fp8QuantizationType fp8_quantization_type, int64_t const top_k, int64_t const hidden_size,
    int64_t const intermediate_size, int64_t const num_local_experts, int64_t const act_type,
    bool const use_shuffled_weight, int64_t const weight_layout, bool const use_per_token_scaling,
    int64_t const num_tokens, bool has_gemm1_lora_delta) {
  // Start from complete valid tactics so every factorized coordinate remains executable.
  auto const completeTactics = trtllm_get_valid_moe_configs(
      dtype_act_, dtype_weights_, fp8_quantization_type, top_k, hidden_size, intermediate_size,
      num_local_experts, act_type, use_shuffled_weight, weight_layout, use_per_token_scaling,
      num_tokens, has_gemm1_lora_delta);
  auto const dtypeAct = static_cast<btg::Dtype>(dtype_act_);
  auto const dtypeWeights = static_cast<btg::Dtype>(dtype_weights_);
  auto const activationType = validateAndCastActivationType(act_type);
  auto const matrixLayout = static_cast<batchedGemm::gemm::MatrixLayout>(weight_layout);
  auto const gemm1BiasType =
      has_gemm1_lora_delta ? batchedGemm::gemm::BiasType::Mn : batchedGemm::gemm::BiasType::None;
  bool const useDeepSeekFp8 = fp8_quantization_type == Fp8QuantizationType::DeepSeekFp8;
  bool const usePerTokenScalingGemm2 = use_per_token_scaling && dtypeAct == btg::Dtype::E2m1;
  bool const useWeightsOnlyConstructor = dtypeAct == btg::Dtype::E4m3 &&
                                         dtypeWeights == btg::Dtype::E4m3 && useDeepSeekFp8 &&
                                         gemm1BiasType == batchedGemm::gemm::BiasType::None;

  // Reuse one dtype-correct runner per tile while decomposing tactics into FC coordinates.
  std::map<int64_t, std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>> runners;
  Array<Array<int64_t>> result;
  for (auto const& completeTactic : completeTactics) {
    TVM_FFI_ICHECK_EQ(completeTactic.size(), 2);
    int64_t const tileN = completeTactic[0];
    int64_t const configIndex = completeTactic[1];
    auto runnerIt = runners.find(tileN);
    if (runnerIt == runners.end()) {
      std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner> runner;
      if (useWeightsOnlyConstructor) {
        runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
            dtypeWeights, useDeepSeekFp8, static_cast<int>(tileN), use_shuffled_weight,
            matrixLayout, use_per_token_scaling, usePerTokenScalingGemm2, false, false);
      } else {
        runner = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner>(
            dtypeAct, dtypeWeights, useDeepSeekFp8, static_cast<int>(tileN), activationType,
            use_shuffled_weight, matrixLayout, gemm1BiasType, use_per_token_scaling,
            usePerTokenScalingGemm2, false, false);
      }
      runnerIt = runners.emplace(tileN, std::move(runner)).first;
    }
    auto const components = runnerIt->second->getConfigComponents(configIndex);
    auto const anchorIndex = runnerIt->second->getDefaultValidConfigIndex(
        top_k, hidden_size, intermediate_size, num_local_experts, num_tokens);
    result.push_back({tileN, configIndex, components.gemm1Config, components.gemm2Config,
                      configIndex == anchorIndex ? 1 : 0});
  }
  return result;
}

/** Allocate stable native replay outputs and scratch for one live-logits router launch. */
Array<Tensor> trtllm_moe_allocate_canonical_routing(TensorView routing_logits, int64_t top_k,
                                                    int64_t tile_tokens_dim) {
  // Validate the public router geometry before deriving any graph-stable allocation extents.
  ffi::CUDADeviceGuard device_guard(routing_logits.device().device_id);
  TVM_FFI_ICHECK_EQ(routing_logits.ndim(), 2) << "routing_logits must be two-dimensional.";
  TVM_FFI_ICHECK(routing_logits.dtype() == dl_bfloat16 || routing_logits.dtype() == dl_float32)
      << "routing_logits must be bfloat16 or float32.";
  int64_t const num_tokens = routing_logits.size(0);
  int64_t const num_experts = routing_logits.size(1);
  TVM_FFI_ICHECK(top_k > 0 && top_k <= num_experts) << "top_k must be between one and num_experts.";
  TVM_FFI_ICHECK_GT(tile_tokens_dim, 0) << "tile_tokens_dim must be positive.";
  int32_t const max_num_padded_tokens =
      tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxPermutedPaddedCount(
          num_tokens, top_k, num_experts, tile_tokens_dim);
  int32_t const max_num_ctas =
      tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxNumCtasInBatchDim(
          num_tokens, top_k, num_experts, tile_tokens_dim);
  int64_t const histogram_size = std::max<int64_t>(num_experts * 2, 256 * 2);
  // Allocate the complete typed record once so later capture only mutates tensor contents.
  CanonicalRoutingBuffers buffers{
      alloc_tensor({num_tokens, top_k}, dl_int16, routing_logits.device()),
      alloc_tensor({num_tokens, top_k}, dl_bfloat16, routing_logits.device()),
      alloc_tensor({num_tokens, top_k}, dl_int32, routing_logits.device()),
      alloc_tensor({num_experts}, dl_int32, routing_logits.device()),
      alloc_tensor({1}, dl_int32, routing_logits.device()),
      alloc_tensor({num_tokens * top_k}, dl_int32, routing_logits.device()),
      alloc_tensor({max_num_padded_tokens + 1}, dl_int32, routing_logits.device()),
      alloc_tensor({histogram_size}, dl_int32, routing_logits.device()),
      alloc_tensor({max_num_ctas}, dl_int32, routing_logits.device()),
      alloc_tensor({max_num_ctas}, dl_int32, routing_logits.device()),
      alloc_tensor({1}, dl_int32, routing_logits.device())};
  return buffers.to_ffi();
}

/** Launch the real TRTLLM router once and retain both conventional and replay outputs. */
void trtllm_moe_canonicalize_routing(
    TensorView routing_logits, Optional<TensorView> routing_bias, TensorView hidden_states,
    Array<Tensor> canonical, int64_t top_k, Optional<int64_t> n_group, Optional<int64_t> topk_group,
    int64_t local_expert_offset, int64_t local_num_experts, Optional<double> routed_scaling_factor,
    int64_t routing_method_type, bool use_routing_scales_on_input, bool use_deep_seek_fp8,
    bool norm_topk_prob, bool enable_pdl, int64_t tile_tokens_dim) {
  // Decode the public tensor array once and retain named fields through routing.
  ffi::CUDADeviceGuard device_guard(routing_logits.device().device_id);
  CanonicalRoutingBuffers const buffers = CanonicalRoutingBuffers::from_ffi(canonical);
  int64_t const num_tokens = routing_logits.size(0);
  int64_t const num_experts = routing_logits.size(1);
  TVM_FFI_ICHECK_EQ(hidden_states.size(0), num_tokens)
      << "hidden_states and routing_logits must have the same token count.";
  TVM_FFI_ICHECK(local_num_experts > 0 && local_expert_offset + local_num_experts <= num_experts)
      << "the local expert range must lie within routing_logits.";

  btg::Dtype dtype_elt;
  if (hidden_states.dtype() == dl_float16) {
    dtype_elt = btg::Dtype::Fp16;
  } else if (hidden_states.dtype() == dl_bfloat16) {
    dtype_elt = btg::Dtype::Bfloat16;
  } else if (hidden_states.dtype() == dl_float8_e4m3fn) {
    dtype_elt = btg::Dtype::E4m3;
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "Unsupported activation dtype for canonical routing.";
  }
  btg::Dtype const routing_bias_dtype =
      routing_bias.has_value() && routing_bias.value().dtype() == dl_float32 ? btg::Dtype::Fp32
                                                                             : btg::Dtype::Bfloat16;
  btg::Dtype const routing_logits_dtype =
      routing_logits.dtype() == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;

  // Run the production router into graph-stable storage, including its native int16 replay IDs.
  tensorrt_llm::kernels::trtllmgen_moe::Routing::Runner routing_runner(tile_tokens_dim);
  cudaStream_t stream = get_stream(routing_logits.device());
  routing_runner.run(
      const_cast<void*>(routing_logits.data_ptr()),
      routing_bias.has_value() ? const_cast<void*>(routing_bias.value().data_ptr()) : nullptr,
      num_tokens, num_experts, top_k, 0, n_group.value_or(0), topk_group.value_or(0),
      local_expert_offset, local_num_experts, routed_scaling_factor.value_or(1.0),
      static_cast<int*>(buffers.packed_scratch.data_ptr()),
      static_cast<int*>(buffers.expert_count_histogram.data_ptr()),
      static_cast<int*>(buffers.total_num_padded_tokens.data_ptr()),
      static_cast<int*>(buffers.expanded_idx_to_permuted_idx.data_ptr()), nullptr,
      static_cast<int*>(buffers.permuted_idx_to_token_idx.data_ptr()), nullptr,
      buffers.expert_weights.data_ptr(),
      static_cast<int*>(buffers.num_tokens_per_expert.data_ptr()),
      static_cast<int*>(buffers.cta_idx_xy_to_batch_idx.data_ptr()),
      static_cast<int*>(buffers.cta_idx_xy_to_mn_limit.data_ptr()),
      static_cast<int*>(buffers.num_non_exiting_ctas.data_ptr()), dtype_elt, routing_bias_dtype,
      use_routing_scales_on_input, use_deep_seek_fp8,
      static_cast<RoutingMethodType>(routing_method_type), stream, routing_logits_dtype,
      norm_topk_prob, static_cast<int16_t*>(buffers.routing_replay_ids.data_ptr()), enable_pdl);
}

/// Allocate graph-stable routing metadata storage for each requested tile without launching work.
Array<Tensor> trtllm_moe_allocate_routing_metadata_multi_tile(
    TensorView topk_ids, int64_t num_experts, int64_t top_k, int64_t local_expert_offset,
    int64_t local_num_experts, Array<int64_t> tile_tokens_dims, int64_t routing_input_mode,
    Optional<Tensor> topk_weights) {
  // Validate shared routing inputs once before allocating tile-dependent output records.
  ffi::CUDADeviceGuard device_guard(topk_ids.device().device_id);
  auto const input_mode = validateMultiTileRoutingInputs(
      topk_ids, num_experts, top_k, local_expert_offset, local_num_experts, tile_tokens_dims,
      routing_input_mode, topk_weights);
  int64_t const num_tokens = topk_ids.size(0);
  int64_t const histogram_size = std::max<int64_t>(num_experts * 2, 256 * 2);
  DLDataType const expert_weights_dtype = input_mode == RoutingInputMode::UnpackedPrecomputed
                                              ? topk_weights.value().dtype()
                                              : dl_bfloat16;

  // Allocate one typed record per tile while preserving borrowed unpacked routing weights.
  std::vector<RoutingMetadataBuffers> routing_metadata;
  routing_metadata.reserve(tile_tokens_dims.size());
  for (int64_t tile_tokens_dim : tile_tokens_dims) {
    int32_t const max_num_padded_tokens =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxPermutedPaddedCount(
            num_tokens, top_k, num_experts, tile_tokens_dim);
    int32_t const max_num_ctas =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxNumCtasInBatchDim(
            num_tokens, top_k, num_experts, tile_tokens_dim);

    RoutingMetadataBuffers metadata{
        alloc_tensor({1}, dl_int32, topk_ids.device()),
        alloc_tensor({num_tokens * top_k}, dl_int32, topk_ids.device()),
        alloc_tensor({max_num_padded_tokens + 1}, dl_int32, topk_ids.device()),
        input_mode == RoutingInputMode::UnpackedPrecomputed
            ? topk_weights.value()
            : alloc_tensor({num_tokens, top_k}, dl_bfloat16, topk_ids.device()),
        alloc_tensor({histogram_size}, dl_int32, topk_ids.device()),
        alloc_tensor({num_experts}, dl_int32, topk_ids.device()),
        alloc_tensor({max_num_ctas}, dl_int32, topk_ids.device()),
        alloc_tensor({max_num_ctas}, dl_int32, topk_ids.device()),
        alloc_tensor({1}, dl_int32, topk_ids.device())};
    validateRoutingMetadata(metadata, num_tokens, top_k, num_experts, tile_tokens_dim,
                            topk_ids.device(), expert_weights_dtype);
    routing_metadata.push_back(std::move(metadata));
  }
  return routingMetadataToFfi(routing_metadata);
}

/// Return the exact native token bound for the fused multi-tile DA preamble.
int64_t trtllm_moe_max_da_multi_tile_tokens(int64_t num_experts) {
  TVM_FFI_ICHECK_GT(num_experts, 0) << "num_experts must be positive.";
  TVM_FFI_ICHECK_LE(num_experts, da_moe::kDAMaxExperts)
      << "num_experts exceeds the compiled DA capacity.";
  return moe::dev::routing::routingPrecomputed::maxTokensMultiTileCluster(num_experts);
}

/// Populate all preallocated tile metadata with one routing-method-neutral CUDA kernel launch.
void trtllm_moe_populate_routing_metadata_multi_tile(
    TensorView topk_ids, int64_t num_experts, int64_t top_k, int64_t local_expert_offset,
    int64_t local_num_experts, Array<int64_t> tile_tokens_dims, Array<Tensor> flat_routing_metadata,
    int64_t routing_input_mode, Optional<TensorView> topk_weights) {
  ffi::CUDADeviceGuard device_guard(topk_ids.device().device_id);
  auto const input_mode = validateMultiTileRoutingInputs(
      topk_ids, num_experts, top_k, local_expert_offset, local_num_experts, tile_tokens_dims,
      routing_input_mode, topk_weights);
  cudaStream_t stream = get_stream(topk_ids.device());
  auto const routing_metadata =
      routingMetadataFromFfi(flat_routing_metadata, tile_tokens_dims.size());
  auto routing_data = makeMultiTileRoutingData(topk_ids, num_experts, top_k, local_expert_offset,
                                               local_num_experts, tile_tokens_dims,
                                               routing_metadata, input_mode, topk_weights, stream);

  moe::dev::routing::routingPrecomputed::runMultiTileCluster(
      routing_data.data(), static_cast<int32_t>(routing_data.size()), stream);
}

/// Inspect whether the active outer capture may safely reuse one workspace lane.
Array<int64_t> trtllm_moe_inspect_da_workspace_lane(TensorView device_anchor,
                                                    int64_t expected_capture_id,
                                                    int64_t previous_conditional_node_handle) {
  ffi::CUDADeviceGuard device_guard(device_anchor.device().device_id);
  cudaStream_t stream = get_stream(device_anchor.device());
  da_moe::ActiveCaptureContext context{};
  CHECK_CUDA_ERROR(da_moe::GetActiveCaptureContext(stream, &context));
  TVM_FFI_ICHECK(context.status == cudaStreamCaptureStatusActive && context.graph != nullptr)
      << "DA workspace-lane inspection requires an active outer CUDA Graph capture.";
  bool is_serialized = false;
  CHECK_CUDA_ERROR(da_moe::ValidateWorkspaceLaneSequence(
      context, static_cast<unsigned long long>(expected_capture_id),
      reinterpret_cast<cudaGraphNode_t>(previous_conditional_node_handle), &is_serialized));
  return {static_cast<int64_t>(context.capture_id), static_cast<int64_t>(is_serialized)};
}

/// Inject parallel multi-tile routing, selector, and an empty SWITCH into an outer capture.
Array<int64_t> trtllm_moe_begin_da_switch_capture(
    TensorView topk_ids, int64_t num_experts, int64_t top_k, int64_t local_expert_offset,
    int64_t local_num_experts, Array<int64_t> tile_tokens_dims, Array<Tensor> flat_routing_metadata,
    int64_t routing_input_mode, Optional<TensorView> topk_weights, TensorView exemplar_spectra,
    TensorView exemplar_body_indices, int64_t num_selector_exemplars, TensorView selected_body,
    int64_t num_bodies, int64_t expected_capture_id, int64_t previous_conditional_node_handle) {
  // Reject invalid plan capacity before mutating the caller's active outer graph.
  ffi::CUDADeviceGuard device_guard(topk_ids.device().device_id);
  TVM_FFI_ICHECK_GT(num_bodies, 1) << "A DA SWITCH requires at least two bodies.";
  TVM_FFI_ICHECK_LE(num_bodies, da_moe::kDAMaxBodies)
      << "DA body count exceeds immutable SWITCH capacity.";
  TVM_FFI_ICHECK_GT(num_selector_exemplars, 0) << "A DA selector requires at least one exemplar.";
  TVM_FFI_ICHECK_LE(num_selector_exemplars, da_moe::kDAMaxExemplars)
      << "DA selector exemplar count exceeds immutable capacity.";

  auto const input_mode = validateMultiTileRoutingInputs(
      topk_ids, num_experts, top_k, local_expert_offset, local_num_experts, tile_tokens_dims,
      routing_input_mode, topk_weights);
  cudaStream_t stream = get_stream(topk_ids.device());
  da_moe::ActiveCaptureContext original{};
  CHECK_CUDA_ERROR(da_moe::GetActiveCaptureContext(stream, &original));
  TVM_FFI_ICHECK(original.status == cudaStreamCaptureStatusActive && original.graph != nullptr)
      << "DA SWITCH injection requires an active outer CUDA Graph capture.";
  bool is_workspace_lane_serialized = false;
  CHECK_CUDA_ERROR(da_moe::ValidateWorkspaceLaneSequence(
      original, static_cast<unsigned long long>(expected_capture_id),
      reinterpret_cast<cudaGraphNode_t>(previous_conditional_node_handle),
      &is_workspace_lane_serialized));
  TVM_FFI_ICHECK(is_workspace_lane_serialized)
      << "DA workspace lane is not ordered after its previous invocation.";

  // Dispatch the fused preamble first, then rewind the capture frontier to create a sibling root.
  auto const routing_metadata =
      routingMetadataFromFfi(flat_routing_metadata, tile_tokens_dims.size());
  auto routing_data = makeMultiTileRoutingData(topk_ids, num_experts, top_k, local_expert_offset,
                                               local_num_experts, tile_tokens_dims,
                                               routing_metadata, input_mode, topk_weights, stream);
  moe::dev::routing::routingPrecomputed::runMultiTileCluster(
      routing_data.data(), static_cast<int32_t>(routing_data.size()), stream);
  da_moe::ActiveCaptureContext after_parallel_work{};
  CHECK_CUDA_ERROR(da_moe::GetActiveCaptureContext(stream, &after_parallel_work));
  CHECK_CUDA_ERROR(da_moe::SetCaptureDependencies(stream, original.dependencies.data(),
                                                  original.dependencies.size()));

  // Dispatch the device selector from the original frontier using the routing representation ABI.
  cudaGraphConditionalHandle conditional_handle = 0;
  CHECK_CUDA_ERROR(cudaGraphConditionalHandleCreate(&conditional_handle, original.graph, 0,
                                                    cudaGraphCondAssignDefault));
  int64_t const assignment_numel = topk_ids.numel();
  bool const packed_ids = input_mode == RoutingInputMode::PackedPrecomputed;
  if (packed_ids) {
    da_moe::DASelectorKernel<da_moe::kDAMaxExperts, da_moe::kDAMaxExemplars, true>
        <<<1, da_moe::kDASelectorBlockThreads, 0, stream>>>(
            static_cast<int32_t const*>(topk_ids.data_ptr()), assignment_numel, num_experts,
            static_cast<float const*>(exemplar_spectra.data_ptr()),
            static_cast<int32_t const*>(exemplar_body_indices.data_ptr()),
            static_cast<int>(num_selector_exemplars), conditional_handle,
            static_cast<int32_t*>(selected_body.data_ptr()));
  } else if (topk_ids.dtype() == dl_int16) {
    da_moe::DASelectorKernel<da_moe::kDAMaxExperts, da_moe::kDAMaxExemplars, false, int16_t>
        <<<1, da_moe::kDASelectorBlockThreads, 0, stream>>>(
            static_cast<int16_t const*>(topk_ids.data_ptr()), assignment_numel, num_experts,
            static_cast<float const*>(exemplar_spectra.data_ptr()),
            static_cast<int32_t const*>(exemplar_body_indices.data_ptr()),
            static_cast<int>(num_selector_exemplars), conditional_handle,
            static_cast<int32_t*>(selected_body.data_ptr()));
  } else {
    da_moe::DASelectorKernel<da_moe::kDAMaxExperts, da_moe::kDAMaxExemplars, false>
        <<<1, da_moe::kDASelectorBlockThreads, 0, stream>>>(
            static_cast<int32_t const*>(topk_ids.data_ptr()), assignment_numel, num_experts,
            static_cast<float const*>(exemplar_spectra.data_ptr()),
            static_cast<int32_t const*>(exemplar_body_indices.data_ptr()),
            static_cast<int>(num_selector_exemplars), conditional_handle,
            static_cast<int32_t*>(selected_body.data_ptr()));
  }
  CHECK_CUDA_ERROR(cudaPeekAtLastError());
  da_moe::ActiveCaptureContext after_selector{};
  CHECK_CUDA_ERROR(da_moe::GetActiveCaptureContext(stream, &after_selector));

  // Join both independent roots at one conditional node whose child graphs are populated later.
  std::vector<cudaGraphNode_t> switch_dependencies = after_parallel_work.dependencies;
  switch_dependencies.insert(switch_dependencies.end(), after_selector.dependencies.begin(),
                             after_selector.dependencies.end());
  cudaGraphNodeParams conditional_params{};
  conditional_params.type = cudaGraphNodeTypeConditional;
  conditional_params.conditional.handle = conditional_handle;
  conditional_params.conditional.type = cudaGraphCondTypeSwitch;
  conditional_params.conditional.size = static_cast<unsigned int>(num_bodies);
  cudaGraphNode_t conditional_node = nullptr;
  CHECK_CUDA_ERROR(da_moe::AddGraphNode(&conditional_node, original.graph,
                                        switch_dependencies.data(), switch_dependencies.size(),
                                        &conditional_params));

  DASwitchCaptureState state{original.capture_id,
                             conditional_node,
                             after_parallel_work.dependencies.back(),
                             after_selector.dependencies.back(),
                             {}};
  state.body_graphs.reserve(num_bodies);
  for (int64_t body_index = 0; body_index < num_bodies; ++body_index) {
    state.body_graphs.push_back(conditional_params.conditional.phGraph_out[body_index]);
  }
  return state.to_ffi();
}

/// Begin direct stream capture into one CUDA-owned conditional body graph.
void trtllm_moe_begin_da_body_capture(int64_t device_id, int64_t auxiliary_stream_handle,
                                      int64_t body_graph_handle) {
  ffi::CUDADeviceGuard device_guard(device_id);
  auto stream = reinterpret_cast<cudaStream_t>(auxiliary_stream_handle);
  auto graph = reinterpret_cast<cudaGraph_t>(body_graph_handle);
  CHECK_CUDA_ERROR(cudaStreamBeginCaptureToGraph(stream, graph, nullptr, nullptr, 0,
                                                 cudaStreamCaptureModeRelaxed));
}

/// Create one private nonblocking stream for direct conditional-body capture.
int64_t trtllm_moe_create_da_body_capture_stream(int64_t device_id) {
  ffi::CUDADeviceGuard device_guard(device_id);
  cudaStream_t stream = nullptr;
  CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  return reinterpret_cast<int64_t>(stream);
}

/// Destroy one private conditional-body capture stream after its Python owner expires.
void trtllm_moe_destroy_da_body_capture_stream(int64_t device_id, int64_t stream_handle) {
  ffi::CUDADeviceGuard device_guard(device_id);
  auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

/// End direct stream capture and require CUDA to return the requested body graph.
void trtllm_moe_end_da_body_capture(int64_t device_id, int64_t auxiliary_stream_handle,
                                    int64_t body_graph_handle) {
  ffi::CUDADeviceGuard device_guard(device_id);
  auto stream = reinterpret_cast<cudaStream_t>(auxiliary_stream_handle);
  auto expected_graph = reinterpret_cast<cudaGraph_t>(body_graph_handle);
  cudaGraph_t captured_graph = nullptr;
  CHECK_CUDA_ERROR(cudaStreamEndCapture(stream, &captured_graph));
  TVM_FFI_ICHECK_EQ(captured_graph, expected_graph)
      << "CUDA returned a different DA conditional body graph.";
}

/// Join the populated SWITCH to the outer capture and return inspected topology facts.
Array<int64_t> trtllm_moe_finish_da_switch_capture(TensorView device_anchor, Array<int64_t> state) {
  // Rehydrate the named CUDA handles and prove this is still the same active capture generation.
  ffi::CUDADeviceGuard device_guard(device_anchor.device().device_id);
  auto const capture_state = DASwitchCaptureState::from_ffi(state);
  int64_t const body_count = capture_state.body_graphs.size();
  cudaStream_t stream = get_stream(device_anchor.device());
  da_moe::ActiveCaptureContext context{};
  CHECK_CUDA_ERROR(da_moe::GetActiveCaptureContext(stream, &context));
  TVM_FFI_ICHECK_EQ(context.capture_id, capture_state.capture_id)
      << "DA SWITCH capture generation changed before body completion.";
  auto conditional_node = capture_state.conditional_node;
  CHECK_CUDA_ERROR(da_moe::SetCaptureDependencies(stream, &conditional_node, 1));

  // Inspect the completed outer topology only after joining the SWITCH into the stream frontier.
  size_t outer_node_count = 0;
  size_t outer_edge_count = 0;
  CHECK_CUDA_ERROR(cudaGraphGetNodes(context.graph, nullptr, &outer_node_count));
  CHECK_CUDA_ERROR(da_moe::GetGraphEdgeCount(context.graph, &outer_edge_count));
  auto parallel_node = capture_state.parallel_work_node;
  auto selector_node = capture_state.selector_node;
  std::vector<cudaGraphNode_t> parallel_dependencies;
  std::vector<cudaGraphNode_t> selector_dependencies;
  CHECK_CUDA_ERROR(da_moe::GetGraphNodeDependencies(parallel_node, &parallel_dependencies));
  CHECK_CUDA_ERROR(da_moe::GetGraphNodeDependencies(selector_node, &selector_dependencies));

  // Encode the stable inspection ABI, followed by one node count per conditional child body.
  Array<int64_t> topology;
  topology.push_back(static_cast<int64_t>(capture_state.capture_id));
  topology.push_back(static_cast<int64_t>(outer_node_count));
  topology.push_back(static_cast<int64_t>(outer_edge_count));
  topology.push_back(1);
  topology.push_back(body_count);
  topology.push_back(static_cast<int64_t>(selector_dependencies.size()));
  topology.push_back(static_cast<int64_t>(parallel_dependencies.size()));
  topology.push_back(
      da_moe::HaveSameGraphDependencies(selector_dependencies, parallel_dependencies) ? 1 : 0);
  // Native preflight proved this invocation is ordered after the lane's prior conditional.
  topology.push_back(1);
  // This finish call contributes exactly one serial invocation to the cumulative topology.
  topology.push_back(1);
  for (int64_t body_index = 0; body_index < body_count; ++body_index) {
    size_t node_count = 0;
    auto body_graph = capture_state.body_graphs[body_index];
    CHECK_CUDA_ERROR(cudaGraphGetNodes(body_graph, nullptr, &node_count));
    topology.push_back(static_cast<int64_t>(node_count));
  }
  return topology;
}

namespace trtllm_cubin_loader {
#include <flashinfer/cubin_loader.h>
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_bf16_moe, trtllm_bf16_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp8_per_tensor_scale_moe, trtllm_fp8_per_tensor_scale_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp8_per_tensor_scale_routed_moe,
                              trtllm_fp8_per_tensor_scale_routed_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp8_per_channel_scale_moe, trtllm_fp8_per_channel_scale_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp8_block_scale_moe, trtllm_fp8_block_scale_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp4_block_scale_moe, trtllm_fp4_block_scale_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_mxint4_block_scale_moe, trtllm_mxint4_block_scale_moe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_get_valid_moe_configs, trtllm_get_valid_moe_configs);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_get_valid_moe_factorizations,
                              trtllm_get_valid_moe_factorizations);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_moe_allocate_canonical_routing,
                              trtllm_moe_allocate_canonical_routing);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_moe_canonicalize_routing, trtllm_moe_canonicalize_routing);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_moe_allocate_routing_metadata_multi_tile,
                              trtllm_moe_allocate_routing_metadata_multi_tile);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_moe_max_da_multi_tile_tokens,
                              trtllm_moe_max_da_multi_tile_tokens);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_moe_populate_routing_metadata_multi_tile,
                              trtllm_moe_populate_routing_metadata_multi_tile);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_moe_begin_da_switch_capture,
                              trtllm_moe_begin_da_switch_capture);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_moe_inspect_da_workspace_lane,
                              trtllm_moe_inspect_da_workspace_lane);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_moe_create_da_body_capture_stream,
                              trtllm_moe_create_da_body_capture_stream);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_moe_destroy_da_body_capture_stream,
                              trtllm_moe_destroy_da_body_capture_stream);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_moe_begin_da_body_capture, trtllm_moe_begin_da_body_capture);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_moe_end_da_body_capture, trtllm_moe_end_da_body_capture);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_moe_finish_da_switch_capture,
                              trtllm_moe_finish_da_switch_capture);

}  // namespace flashinfer
