/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#ifndef CAKE_STEPFUN_FULL
#error "cake_stepfun_stages.cu is part of the full-path Cake StepFun module (-DCAKE_STEPFUN_FULL)"
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "cake_stepfun_routing_tail.cuh"
#include "flashinfer/exception.h"
#include "flashinfer/trtllm/fused_moe/runner.h"

namespace tensorrt_llm {
namespace kernels {
namespace trtllmgen_moe {
namespace cake_stepfun {

namespace btg = batchedGemm::trtllm::gen;
namespace generated = flashinfer::cake_stepfun::generated;

namespace {

generated::TensorLayout denseLayout(void const* data, std::initializer_list<int64_t> dims) {
  generated::TensorLayout layout{};
  layout.data = data;
  layout.rank = static_cast<int>(dims.size());
  int64_t stride = 1;
  int index = layout.rank;
  for (auto it = std::rbegin(dims); it != std::rend(dims); ++it) {
    --index;
    layout.dimensions[index] = *it;
    layout.strides[index] = stride;
    stride *= *it;
  }
  layout.elements = stride;
  return layout;
}

cudaLaunchAttribute pdlAttribute() {
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  return attribute;
}

std::string tileList(int family) {
  std::string tiles;
  for (size_t index = 0; index < generated::kFc2KernelCount; ++index) {
    if (generated::kFc2Kernels[index].family != family) continue;
    tiles += (tiles.empty() ? "" : ", ") + std::to_string(generated::kFc2Kernels[index].tile_n);
  }
  return tiles.empty() ? "none" : tiles;
}

char const* fc2FamilyName(int family) {
  switch (family) {
    case generated::kFc2Bf16:
      return "bf16";
    case generated::kFc2Fp8PerTensor:
      return "fp8";
    case generated::kFc2MxFp8:
      return "mxfp8";
    case generated::kFc2Nvfp4:
      return "nvfp4";
    case generated::kFc2Nvfp4PerToken:
      return "nvfp4_bf16tok";
    default:
      return "unsupported";
  }
}

// The generated-manifest FC2 family served by the Gemm2::Runner constructor arguments, or -1.
int selectFc2Family(btg::Dtype dtypeAct, btg::Dtype dtypeWeights, btg::Dtype dtypeOut,
                    bool useDeepSeekFp8, bool useShuffledMatrix,
                    batchedGemm::gemm::MatrixLayout weightLayout, bool usePerTokenScaling,
                    bool usePerChannelScaling) {
  using batchedGemm::gemm::MatrixLayout;
  if (dtypeOut != btg::Dtype::Bfloat16 || useDeepSeekFp8 || !useShuffledMatrix ||
      usePerChannelScaling) {
    return -1;
  }
  if (dtypeAct == btg::Dtype::E2m1 && dtypeWeights == btg::Dtype::E2m1 &&
      weightLayout == MatrixLayout::MajorK) {
    return usePerTokenScaling ? generated::kFc2Nvfp4PerToken : generated::kFc2Nvfp4;
  }
  if (usePerTokenScaling) return -1;
  if (dtypeAct == btg::Dtype::Bfloat16 && dtypeWeights == btg::Dtype::Bfloat16 &&
      weightLayout == MatrixLayout::BlockMajorK) {
    return generated::kFc2Bf16;
  }
  if (dtypeAct == btg::Dtype::E4m3 && dtypeWeights == btg::Dtype::E4m3 &&
      weightLayout == MatrixLayout::MajorK) {
    return generated::kFc2Fp8PerTensor;
  }
  if (dtypeAct == btg::Dtype::MxE4m3 && dtypeWeights == btg::Dtype::MxE4m3 &&
      weightLayout == MatrixLayout::MajorK) {
    return generated::kFc2MxFp8;
  }
  return -1;
}

template <typename Spec>
void configureSmem(Spec const& spec, std::vector<bool>& configured, size_t index,
                   char const* stage) {
  if (configured[index]) return;
  cudaError_t const rc = spec.configure(spec.dynamic_smem_bytes);
  FLASHINFER_CHECK(rc == cudaSuccess, "Cake StepFun ", stage,
                   ": cudaFuncSetAttribute(MaxDynamicSharedMemorySize=", spec.dynamic_smem_bytes,
                   ") failed for ", spec.symbol, ": ", cudaGetErrorString(rc));
  configured[index] = true;
}

}  // namespace

// ------------------------------------------------------------------------------------------------
// Routing
// ------------------------------------------------------------------------------------------------

bool routingWritesBenignTail() {
  for (size_t index = 0; index < generated::kRoutingKernelCount; ++index) {
    if (!generated::kRoutingKernels[index].writes_benign_tail) return false;
  }
  return generated::kRoutingKernelCount > 0;
}

void RoutingRunner::run(void* routingLogits, void* routingBias, int32_t numTokens,
                        int32_t numExperts, int32_t topK, int32_t numFusedSharedExpert,
                        int32_t nGroup, int32_t topkGroup, int32_t localExpertOffset,
                        int32_t localNumExperts, float routedScalingFactor,
                        int32_t* routingExpertIndexes, int32_t* expertCountHistogram,
                        int32_t* permutedIdxSize, int32_t* expandedIdxToPermutedIdx,
                        int32_t* permutedIdxToExpandedIdx, int32_t* permutedIdxToTokenIdx,
                        int32_t* expertIds, void* expertWeights, int32_t* numTokensPerExpert,
                        int32_t* ctaIdxXyToBatchIdx, int32_t* ctaIdxXyToMnLimit,
                        int32_t* numNonExitingCtas, btg::Dtype dtypeElt, btg::Dtype dtypeBias,
                        bool useRoutingScalesOnInput, bool useDeepSeekFp8,
                        Routing::RoutingMethodType routingMethodType, cudaStream_t stream,
                        btg::Dtype dtypeLogits, bool normTopkProb, int16_t* routing_replay_out,
                        bool enable_pdl) {
  (void)routingBias;
  (void)dtypeBias;
  (void)dtypeElt;
  (void)normTopkProb;
  (void)nGroup;
  (void)topkGroup;
  (void)routedScalingFactor;
  FLASHINFER_CHECK(routingMethodType == Routing::RoutingMethodType::Renormalize,
                   "Cake StepFun routing serves RoutingMethodType::Renormalize only, got ",
                   Routing::serializeMoeRoutingMethodType(routingMethodType));
  FLASHINFER_CHECK(expertIds == nullptr && routingLogits != nullptr,
                   "Cake StepFun routing computes the expert selection from routing_logits; "
                   "pre-computed expert ids are not supported");
  FLASHINFER_CHECK(numFusedSharedExpert == 0,
                   "Cake StepFun routing does not support fused shared experts");
  FLASHINFER_CHECK(routing_replay_out == nullptr,
                   "Cake StepFun routing does not write routing_replay_out");
  FLASHINFER_CHECK(permutedIdxToExpandedIdx == nullptr,
                   "Cake StepFun routing does not write permuted_idx_to_expanded_idx (GEMM1 Mn "
                   "bias rows are not supported)");
  FLASHINFER_CHECK(!useRoutingScalesOnInput && !useDeepSeekFp8,
                   "Cake StepFun routing does not support routing scales on the input or "
                   "DeepSeek FP8");
  FLASHINFER_CHECK(numTokensPerExpert == nullptr,
                   "Cake StepFun routing does not write num_tokens_per_expert");
  FLASHINFER_CHECK(dtypeLogits == btg::Dtype::Fp32 || dtypeLogits == btg::Dtype::Bfloat16,
                   "Cake StepFun routing reads float32 or bfloat16 routing logits");
  int const logitsDtype = dtypeLogits == btg::Dtype::Fp32 ? 0 : 1;

  generated::RoutingKernelSpec const* spec = nullptr;
  for (size_t index = 0; index < generated::kRoutingKernelCount; ++index) {
    auto const& candidate = generated::kRoutingKernels[index];
    if (candidate.logits_dtype == logitsDtype && candidate.min_tokens <= numTokens &&
        numTokens <= candidate.max_tokens) {
      spec = &candidate;
      break;
    }
  }
  FLASHINFER_CHECK(spec != nullptr, "No Cake StepFun routing kernel serves num_tokens=", numTokens,
                   " with ", logitsDtype == 0 ? "float32" : "bfloat16", " logits");

  generated::RoutingArgs args{};
  args.routing_logits = routingLogits;
  args.topk_packed = routingExpertIndexes;
  args.topk_weights = expertWeights;
  args.expert_count_histogram = expertCountHistogram;
  args.total_num_padded_tokens = permutedIdxSize;
  args.expanded_idx_to_permuted_idx = expandedIdxToPermutedIdx;
  args.permuted_idx_to_token_idx = permutedIdxToTokenIdx;
  args.cta_idx_xy_to_batch_idx = ctaIdxXyToBatchIdx;
  args.cta_idx_xy_to_mn_limit = ctaIdxXyToMnLimit;
  args.num_non_exiting_ctas = numNonExitingCtas;
  args.num_tokens = numTokens;
  args.num_experts = numExperts;
  args.top_k = topK;
  args.local_expert_offset = localExpertOffset;
  args.local_num_experts = localNumExperts;
  args.tile_tokens_dim = mTileTokensDim;
  args.max_num_ctas =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);

  static std::vector<bool> configured(generated::kRoutingKernelCount, false);
  configureSmem(*spec, configured, static_cast<size_t>(spec - generated::kRoutingKernels),
                "routing");

  cudaLaunchConfig_t config{};
  if (spec->grid_rule == generated::RoutingGrid::kFixed) {
    config.gridDim = dim3(spec->grid[0], spec->grid[1], spec->grid[2]);
  } else {
    FLASHINFER_CHECK(spec->tokens_per_cta > 0, "Cake StepFun routing kernel ", spec->symbol,
                     " declares no tokens_per_cta");
    config.gridDim = dim3(
        static_cast<unsigned>((numTokens + spec->tokens_per_cta - 1) / spec->tokens_per_cta), 1u,
        1u);
  }
  config.blockDim = dim3(spec->block[0], spec->block[1], spec->block[2]);
  config.dynamicSmemBytes = spec->dynamic_smem_bytes;
  config.stream = stream;
  std::array<cudaLaunchAttribute, 2> attributes{};
  unsigned numAttrs = 0;
  if (enable_pdl) attributes[numAttrs++] = pdlAttribute();
  if (spec->cluster_attribute) {
    attributes[numAttrs].id = cudaLaunchAttributeClusterDimension;
    attributes[numAttrs].val.clusterDim.x = spec->cluster[0];
    attributes[numAttrs].val.clusterDim.y = spec->cluster[1];
    attributes[numAttrs].val.clusterDim.z = spec->cluster[2];
    ++numAttrs;
  }
  config.attrs = attributes.data();
  config.numAttrs = numAttrs;
  cudaError_t const launched = spec->submit(&config, args);
  FLASHINFER_CHECK(launched == cudaSuccess, "Cake StepFun routing launch failed for ", spec->symbol,
                   " num_tokens=", numTokens, " : ", cudaGetErrorString(launched));
}

// ------------------------------------------------------------------------------------------------
// FC2
// ------------------------------------------------------------------------------------------------

Fc2Runner::Fc2Runner(btg::Dtype dtypeAct, btg::Dtype dtypeWeights, btg::Dtype outputDtype,
                     bool useDeepSeekFp8, int tileTokensDim, bool useShuffledMatrix,
                     batchedGemm::gemm::MatrixLayout weightLayout, bool usePerTokenScaling,
                     bool usePerChannelScaling)
    : mDtypeAct(dtypeAct),
      mDtypeWeights(dtypeWeights),
      mDtypeOut(outputDtype),
      mTileTokensDim(tileTokensDim),
      mFamily(selectFc2Family(dtypeAct, dtypeWeights, outputDtype, useDeepSeekFp8,
                              useShuffledMatrix, weightLayout, usePerTokenScaling,
                              usePerChannelScaling)) {
  if (mFamily >= 0) {
    for (size_t index = 0; index < generated::kFc2KernelCount; ++index) {
      auto const& spec = generated::kFc2Kernels[index];
      if (spec.family == mFamily && spec.tile_n == tileTokensDim) {
        mKernels.push_back(static_cast<int32_t>(index));
      }
    }
  }
  mSmemConfigured.assign(generated::kFc2KernelCount, false);
}

bool Fc2Runner::shapeSupported(int32_t configIndex, int32_t hiddenSize,
                               int32_t intermediateSize) const {
  if (std::find(mKernels.begin(), mKernels.end(), configIndex) == mKernels.end()) return false;
  auto const& spec = generated::kFc2Kernels[configIndex];
  int64_t const gridM = hiddenSize / spec.output_rows_per_cta;
  return intermediateSize > 0 && intermediateSize % spec.block_k == 0 && hiddenSize > 0 &&
         hiddenSize % spec.output_rows_per_cta == 0 && gridM % spec.cluster[0] == 0;
}

size_t Fc2Runner::getWorkspaceSizeInBytes(int32_t, int32_t, int32_t, int32_t, int32_t,
                                          int32_t) const {
  return 0;
}

int32_t Fc2Runner::getDefaultValidConfigIndex(int32_t, int32_t hiddenSize,
                                              int32_t intermediateSize, int32_t, int32_t) const {
  for (int32_t index : mKernels) {
    if (shapeSupported(index, hiddenSize, intermediateSize)) return index;
  }
  FLASHINFER_CHECK(false, "No Cake StepFun FC2 kernel (", fc2FamilyName(mFamily),
                   ") serves tile_N=", mTileTokensDim, " at hidden_size=", hiddenSize,
                   ", intermediate_size=", intermediateSize,
                   " (exported tiles for this family: ", tileList(mFamily),
                   "; hidden_size must be a multiple of the kernel's output rows per CTA times "
                   "its cluster size and intermediate_size a multiple of its K tile).");
  return -1;
}

bool Fc2Runner::isValidConfigIndex(int32_t configIndex, int32_t, int32_t hiddenSize,
                                   int32_t intermediateSize, int32_t, int32_t) const {
  return shapeSupported(configIndex, hiddenSize, intermediateSize);
}

std::vector<int64_t> Fc2Runner::getPassingConfigIndices() const {
  return std::vector<int64_t>(mKernels.begin(), mKernels.end());
}

generated::SfLayout Fc2Runner::sfLayoutA(int32_t configIndex) const {
  FLASHINFER_CHECK(std::find(mKernels.begin(), mKernels.end(), configIndex) != mKernels.end(),
                   "Invalid Cake StepFun FC2 config index ", configIndex, " for ",
                   fc2FamilyName(mFamily), " tile_N=", mTileTokensDim);
  return generated::kFc2Kernels[configIndex].sf_layout_a;
}

void Fc2Runner::run(void* permutedHiddenState, void* permutedHiddenStateScale, void* weight,
                    void* weightScale, void* perTokenScales, void* perChannelScales,
                    float* outputScalesScalar, float* ptrBias, void* output, void* outputScale,
                    int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
                    int32_t numExperts, int32_t numTokens, int32_t* ptrNumNonExitingCtas,
                    int32_t* ptrTotalNumPaddedTokens, int32_t* ptrCtaIdxXyToBatchIdx,
                    int32_t* ptrCtaIdxXyToMnLimit, void* bmm2Workspace, int device,
                    cudaStream_t stream, int32_t configIndex, bool enable_pdl,
                    int32_t validIntermediateSize, int32_t validHiddenSize) {
  (void)bmm2Workspace;
  (void)device;
  char const* const family = fc2FamilyName(mFamily);
  FLASHINFER_CHECK(shapeSupported(configIndex, hiddenSize, intermediateSize),
                   "Invalid Cake StepFun FC2 config index ", configIndex, " for ", family,
                   " tile_N=", mTileTokensDim, ", hidden_size=", hiddenSize,
                   ", intermediate_size=", intermediateSize);
  FLASHINFER_CHECK(ptrBias == nullptr, "Cake StepFun FC2 does not consume a GEMM2 bias");
  FLASHINFER_CHECK(perChannelScales == nullptr,
                   "Cake StepFun FC2 does not consume per-channel scales");
  FLASHINFER_CHECK(outputScale == nullptr, "Cake StepFun FC2 writes bf16 without output scales");
  FLASHINFER_CHECK((validHiddenSize < 0 || validHiddenSize == hiddenSize) &&
                       (validIntermediateSize < 0 || validIntermediateSize == intermediateSize),
                   "Cake StepFun FC2 does not support valid (unpadded) dimensions smaller than "
                   "the padded hidden_size / intermediate_size");
  FLASHINFER_CHECK(ptrNumNonExitingCtas != nullptr && ptrCtaIdxXyToBatchIdx != nullptr &&
                       ptrCtaIdxXyToMnLimit != nullptr && ptrTotalNumPaddedTokens != nullptr,
                   "Cake StepFun FC2 requires the complete routing arrays");
  bool const blockScaled = mFamily == generated::kFc2Nvfp4 ||
                           mFamily == generated::kFc2Nvfp4PerToken ||
                           mFamily == generated::kFc2MxFp8;
  FLASHINFER_CHECK(!blockScaled || (permutedHiddenStateScale != nullptr && weightScale != nullptr),
                   "Cake StepFun FC2 (", family, ") requires activation and weight block scales");
  bool const scalarScaled = mFamily == generated::kFc2Nvfp4 ||
                            mFamily == generated::kFc2Nvfp4PerToken ||
                            mFamily == generated::kFc2Fp8PerTensor;
  FLASHINFER_CHECK(!scalarScaled || outputScalesScalar != nullptr, "Cake StepFun FC2 (", family,
                   ") requires the per-expert output2 scales");
  FLASHINFER_CHECK(mFamily != generated::kFc2Nvfp4PerToken || perTokenScales != nullptr,
                   "Cake StepFun FC2 (nvfp4_bf16tok) requires the fp32 per-token scales");
  FLASHINFER_CHECK(mFamily == generated::kFc2Nvfp4PerToken || perTokenScales == nullptr,
                   "Cake StepFun FC2 (", family, ") does not consume per-token scales");

  auto const& spec = generated::kFc2Kernels[configIndex];
  int64_t const E = numExperts;
  int64_t const H = hiddenSize;
  int64_t const I = intermediateSize;
  int32_t const gridM = hiddenSize / spec.output_rows_per_cta;
  int32_t const gridN =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
  int64_t const maxPaddedTokens =
      Routing::getMaxPermutedPaddedCount(numTokens, topK, numExperts, mTileTokensDim);
  int32_t const kTiles = intermediateSize / spec.block_k;

  generated::Fc2Args args{};
  generated::TensorLayout weightLayout{}, weightScaleLayout{}, activationLayout{},
      activationScaleLayout{}, outputLayout{};
  // Weight / scale views follow the FC1 conventions with the roles of H and I exchanged (weights on
  // M = hidden size, K = intermediate size); see README.md.
  switch (mFamily) {
    case generated::kFc2Nvfp4:
    case generated::kFc2Nvfp4PerToken:
      weightLayout = denseLayout(weight, {E, H, I / 2});
      weightScaleLayout = denseLayout(weightScale, {E * gridM, I / 64, int64_t{2}, int64_t{256}});
      activationLayout = denseLayout(permutedHiddenState, {maxPaddedTokens, I / 2});
      activationScaleLayout = denseLayout(permutedHiddenStateScale, {maxPaddedTokens, I / 16});
      break;
    case generated::kFc2Bf16:
      weightLayout = denseLayout(weight, {E, I / 64, H, int64_t{64}});
      activationLayout = denseLayout(permutedHiddenState, {maxPaddedTokens, I});
      break;
    case generated::kFc2Fp8PerTensor:
      weightLayout = denseLayout(weight, {E, H, I});
      activationLayout = denseLayout(permutedHiddenState, {maxPaddedTokens, I});
      break;
    case generated::kFc2MxFp8:
      weightLayout = denseLayout(weight, {E, H, I});
      weightScaleLayout = denseLayout(weightScale, {E * gridM, I / 128, int64_t{2}, int64_t{256}});
      activationLayout = denseLayout(permutedHiddenState, {maxPaddedTokens, I});
      activationScaleLayout = denseLayout(permutedHiddenStateScale, {maxPaddedTokens, I / 32});
      break;
    default:
      FLASHINFER_CHECK(false, "Cake StepFun FC2: unknown family ", mFamily);
  }
  outputLayout = denseLayout(output, {maxPaddedTokens, H});
  FLASHINFER_CHECK(spec.encode_a != nullptr && spec.encode_a(&args.A, weightLayout),
                   "Cake StepFun FC2 (", family, "): failed to encode the weight tensor map");
  if (spec.encode_sfa != nullptr) {
    FLASHINFER_CHECK(spec.encode_sfa(&args.SFA, weightScaleLayout), "Cake StepFun FC2 (", family,
                     "): failed to encode the weight block-scale tensor map");
  }
  if (spec.encode_b != nullptr) {
    FLASHINFER_CHECK(spec.encode_b(&args.B_map, activationLayout), "Cake StepFun FC2 (", family,
                     "): failed to encode the activation tensor map");
  }
  if (spec.encode_sfb != nullptr) {
    FLASHINFER_CHECK(spec.encode_sfb(&args.SFB_map, activationScaleLayout), "Cake StepFun FC2 (",
                     family, "): failed to encode the activation block-scale tensor map");
  }
  if (spec.encode_c != nullptr) {
    FLASHINFER_CHECK(spec.encode_c(&args.C_map, outputLayout), "Cake StepFun FC2 (", family,
                     "): failed to encode the output tensor map");
  }
  args.B_ptr = permutedHiddenState;
  args.SFB_ptr = permutedHiddenStateScale;
  args.C_ptr = output;
  args.per_token_scale = static_cast<float*>(perTokenScales);
  args.tile_expert = ptrCtaIdxXyToBatchIdx;
  args.tile_mn_limit = ptrCtaIdxXyToMnLimit;
  args.total_tiles = ptrNumNonExitingCtas;
  args.total_num_padded_tokens = ptrTotalNumPaddedTokens;
  args.work_counter = nullptr;
  args.scale_c = outputScalesScalar;
  args.N_out = hiddenSize;
  args.K = intermediateSize;
  args.grid_m = gridM;
  args.grid_n = gridN;
  args.K_tiles = kTiles;

  configureSmem(spec, mSmemConfigured, static_cast<size_t>(configIndex), "FC2");

  if (!spec.bounds_acquired_tiles && !routingWritesBenignTail()) {
    cudaError_t const padded =
        launchRoutingTail(ptrCtaIdxXyToBatchIdx, ptrCtaIdxXyToMnLimit, /*routeMap=*/nullptr,
                          ptrNumNonExitingCtas, gridN, mTileTokensDim, enable_pdl, stream);
    FLASHINFER_CHECK(padded == cudaSuccess, "Cake StepFun FC2 routing-tail launch failed for ",
                     spec.symbol, " grid_n=", gridN, " : ", cudaGetErrorString(padded));
  }

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(gridM), static_cast<unsigned>(gridN), 1u);
  config.blockDim = dim3(spec.block[0], spec.block[1], spec.block[2]);
  config.dynamicSmemBytes = spec.dynamic_smem_bytes;
  config.stream = stream;
  std::array<cudaLaunchAttribute, 2> attributes{};
  unsigned numAttrs = 0;
  if (enable_pdl) attributes[numAttrs++] = pdlAttribute();
  if (spec.cluster_attribute) {
    attributes[numAttrs].id = cudaLaunchAttributeClusterDimension;
    attributes[numAttrs].val.clusterDim.x = spec.cluster[0];
    attributes[numAttrs].val.clusterDim.y = spec.cluster[1];
    attributes[numAttrs].val.clusterDim.z = spec.cluster[2];
    ++numAttrs;
  }
  config.attrs = attributes.data();
  config.numAttrs = numAttrs;
  cudaError_t const launched = spec.submit(&config, args);
  FLASHINFER_CHECK(launched == cudaSuccess, "Cake StepFun FC2 launch failed for ", spec.symbol,
                   " grid=(", gridM, ",", gridN, ") : ", cudaGetErrorString(launched));
}

// ------------------------------------------------------------------------------------------------
// NVFP4 per-token requantization
// ------------------------------------------------------------------------------------------------

namespace requant {

void run(int32_t numExpanded, int32_t innerDim, __nv_bfloat16 const* input, float globalScaleInv,
         float e4m3Max, int32_t const* expandedIdxToPermutedIdx, uint8_t* output,
         uint8_t* outputScale, float* perTokenScaleOut, generated::SfLayout layout,
         cudaStream_t stream, bool enablePdl) {
  FLASHINFER_CHECK(input != nullptr && expandedIdxToPermutedIdx != nullptr && output != nullptr &&
                       outputScale != nullptr && perTokenScaleOut != nullptr,
                   "Cake StepFun requantization requires every operand");
  generated::RequantKernelSpec const* spec = nullptr;
  for (size_t index = 0; index < generated::kRequantKernelCount; ++index) {
    if (generated::kRequantKernels[index].sf_layout == layout) {
      spec = &generated::kRequantKernels[index];
      break;
    }
  }
  FLASHINFER_CHECK(spec != nullptr,
                   "No Cake StepFun requantization kernel writes block-scale layout ",
                   static_cast<int>(layout), " (the layout the selected FC2 kernel reads)");
  FLASHINFER_CHECK(spec->rows_per_cta > 0, "Cake StepFun requantization kernel ", spec->symbol,
                   " declares no rows_per_cta");
  generated::RequantArgs args{};
  args.input = input;
  args.expanded_idx_to_permuted_idx = expandedIdxToPermutedIdx;
  args.output = output;
  args.output_scale = outputScale;
  args.per_token_scale = perTokenScaleOut;
  args.global_scale_inv = globalScaleInv;
  args.e4m3_max = e4m3Max;
  args.num_expanded = numExpanded;
  args.inner_dim = innerDim;

  static std::vector<bool> configured(generated::kRequantKernelCount, false);
  configureSmem(*spec, configured, static_cast<size_t>(spec - generated::kRequantKernels),
                "requantization");
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(
      static_cast<unsigned>((numExpanded + spec->rows_per_cta - 1) / spec->rows_per_cta), 1u, 1u);
  config.blockDim = dim3(spec->block[0], spec->block[1], spec->block[2]);
  config.dynamicSmemBytes = spec->dynamic_smem_bytes;
  config.stream = stream;
  cudaLaunchAttribute attribute = pdlAttribute();
  config.attrs = &attribute;
  config.numAttrs = enablePdl ? 1u : 0u;
  cudaError_t const launched = spec->submit(&config, args);
  FLASHINFER_CHECK(launched == cudaSuccess, "Cake StepFun requantization launch failed for ",
                   spec->symbol, " num_expanded=", numExpanded, " : ", cudaGetErrorString(launched));
}

}  // namespace requant

// ------------------------------------------------------------------------------------------------
// Finalize
// ------------------------------------------------------------------------------------------------

namespace finalize {

void run(moe::dev::finalize::Data const& data, cudaStream_t stream) {
  FLASHINFER_CHECK(!data.mUseDeepSeekFp8, "Cake StepFun finalize does not serve DeepSeek FP8");
  FLASHINFER_CHECK(data.mDtypeElt == btg::Dtype::Bfloat16 && data.mDtypeExpW == btg::Dtype::Bfloat16,
                   "Cake StepFun finalize reads bf16 FC2 output and bf16 expert weights");
  FLASHINFER_CHECK(data.inDqSfsPtr == nullptr && data.outDqSfsPtr == nullptr,
                   "Cake StepFun finalize consumes no dequantization scales");
  FLASHINFER_CHECK(data.expertWeightsPtr != nullptr,
                   "Cake StepFun finalize requires the routing expert weights");
  // Same variant rule as the native dispatcher: the scalar kernel below 1184 CTAs (148 SMs x 8
  // blocks of the native kernel's occupancy), the vector-load kernel otherwise.
  int const numThreads = 256;
  int const numBlocksX = (data.hiddenDim - 1 + numThreads) / numThreads;
  int const numBlocksY = std::min(8192, data.numTokens);
  generated::FinalizeVariant const variant = numBlocksX * numBlocksY < 1184
                                                 ? generated::FinalizeVariant::kScalar
                                                 : generated::FinalizeVariant::kVector;
  generated::FinalizeKernelSpec const* spec = nullptr;
  for (size_t index = 0; index < generated::kFinalizeKernelCount; ++index) {
    if (generated::kFinalizeKernels[index].variant == variant) {
      spec = &generated::kFinalizeKernels[index];
      break;
    }
  }
  FLASHINFER_CHECK(spec != nullptr, "No Cake StepFun finalize kernel of variant ",
                   variant == generated::FinalizeVariant::kScalar ? "scalar" : "vector",
                   " (hidden_dim=", data.hiddenDim, ", num_tokens=", data.numTokens, ")");
  FLASHINFER_CHECK(spec->max_top_k <= 0 || data.topK <= spec->max_top_k,
                   "Cake StepFun finalize kernel ", spec->symbol, " supports top_k <= ",
                   spec->max_top_k, ", got ", data.topK);

  generated::FinalizeArgs args{};
  args.input = static_cast<__nv_bfloat16 const*>(data.inPtr);
  args.expert_weights = static_cast<__nv_bfloat16 const*>(data.expertWeightsPtr);
  args.output = static_cast<__nv_bfloat16*>(data.outPtr);
  args.expanded_idx_to_permuted_idx = data.expandedIdxToPermutedIdx;
  args.total_num_padded_tokens = data.totalNumPaddedTokens;
  args.hidden_dim = data.hiddenDim;
  args.hidden_dim_padded = data.hiddenDimPadded;
  args.num_tokens = data.numTokens;
  args.num_experts = data.numExperts;
  args.top_k = data.topK;

  static std::vector<bool> configured(generated::kFinalizeKernelCount, false);
  configureSmem(*spec, configured, static_cast<size_t>(spec - generated::kFinalizeKernels),
                "finalize");
  cudaLaunchConfig_t config{};
  config.gridDim = variant == generated::FinalizeVariant::kScalar
                       ? dim3(static_cast<unsigned>(numBlocksX), static_cast<unsigned>(numBlocksY),
                              1u)
                       : dim3(static_cast<unsigned>(data.numTokens), 1u, 1u);
  config.blockDim = dim3(spec->block[0], spec->block[1], spec->block[2]);
  config.dynamicSmemBytes = spec->dynamic_smem_bytes;
  config.stream = stream;
  cudaLaunchAttribute attribute = pdlAttribute();
  config.attrs = &attribute;
  config.numAttrs = data.mUsePdl ? 1u : 0u;
  cudaError_t const launched = spec->submit(&config, args);
  FLASHINFER_CHECK(launched == cudaSuccess, "Cake StepFun finalize launch failed for ",
                   spec->symbol, " num_tokens=", data.numTokens, " : ",
                   cudaGetErrorString(launched));
}

}  // namespace finalize

}  // namespace cake_stepfun
}  // namespace trtllmgen_moe
}  // namespace kernels
}  // namespace tensorrt_llm
