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

#ifndef CAKE_STEPFUN_FC1
#error "cake_stepfun_fc1_runner.cu is part of the Cake StepFun fused-MoE module (-DCAKE_STEPFUN_FC1)"
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdlib>

#include "flashinfer/exception.h"
#include "flashinfer/trtllm/fused_moe/runner.h"
#include "generated/cake_stepfun_generated_manifest.cuh"

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

// Routing tail for kernels that do not bound the tiles they acquire through cluster launch control
// by num_non_exiting_ctas (Fc1KernelSpec::bounds_acquired_tiles == false; they take total_tiles and
// exit only their initial CTA). trtllm-gen routing writes cta_idx_xy_to_batch_idx and
// cta_idx_xy_to_mn_limit for the first *numNonExitingCtas CTAs and permuted_idx_to_token_idx for
// their token slots only; a running CTA of such a kernel processes every cancelled CTA it acquires,
// so the entries in [*numNonExitingCtas, gridN) must describe a benign tile: expert 0, zero valid
// rows (mn_limit = tile * tileN) and padded token slots (-1). The kernel is launched in-stream between
// routing and FC1 with the FC1's programmatic-dependent-launch attribute; it waits for routing
// before reading the count and releases the FC1 once the tail is written.
constexpr unsigned kRoutingTailBlocks = 4;
constexpr unsigned kRoutingTailThreads = 256;

__global__ void __launch_bounds__(kRoutingTailThreads)
    padRoutingTailKernel(int32_t* tileExpert, int32_t* tileMnLimit, int32_t* routeMap,
                         int32_t const* numNonExitingCtas, int32_t gridN, int32_t tileN) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  int32_t const first = *numNonExitingCtas;
  int32_t const stride = static_cast<int32_t>(gridDim.x * blockDim.x);
  int32_t const lane = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  for (int32_t tile = first + lane; tile < gridN; tile += stride) {
    tileExpert[tile] = 0;
    tileMnLimit[tile] = tile * tileN;
  }
  for (int32_t slot = first * tileN + lane; slot < gridN * tileN; slot += stride) {
    routeMap[slot] = -1;
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// The generated-manifest family served by the native cubin selection arguments, or -1.
int selectFamily(btg::Dtype dtypeAct, btg::Dtype dtypeWeights, btg::Dtype dtypeOutput,
                 bool useDeepSeekFp8, MoE::ActivationType activationType, bool useShuffledMatrix,
                 batchedGemm::gemm::MatrixLayout weightLayout, batchedGemm::gemm::BiasType biasType,
                 bool usePerTokenScaling, bool usePerChannelScaling) {
  using batchedGemm::gemm::MatrixLayout;
  // Every exported family is the fused SwiGLU-Step FC1 over trtllm-shuffled weights without a
  // GEMM1 bias or per-channel scales.
  if (activationType != MoE::ActivationType::SwigluStep || !useShuffledMatrix ||
      biasType != batchedGemm::gemm::BiasType::None || usePerChannelScaling || useDeepSeekFp8) {
    return -1;
  }
  if (dtypeAct == btg::Dtype::E2m1 && dtypeWeights == btg::Dtype::E2m1 &&
      weightLayout == MatrixLayout::MajorK) {
    if (dtypeOutput == btg::Dtype::E2m1 && !usePerTokenScaling) return generated::kFc1Nvfp4;
    if (dtypeOutput == btg::Dtype::Bfloat16 && usePerTokenScaling) {
      return generated::kFc1Nvfp4PerToken;
    }
    return -1;
  }
  if (usePerTokenScaling) return -1;
  if (dtypeAct == btg::Dtype::Bfloat16 && dtypeWeights == btg::Dtype::Bfloat16 &&
      dtypeOutput == btg::Dtype::Bfloat16 && weightLayout == MatrixLayout::BlockMajorK) {
    return generated::kFc1Bf16;
  }
  if (dtypeAct == btg::Dtype::E4m3 && dtypeWeights == btg::Dtype::E4m3 &&
      dtypeOutput == btg::Dtype::E4m3 && weightLayout == MatrixLayout::MajorK) {
    return generated::kFc1Fp8PerTensor;
  }
  if (dtypeAct == btg::Dtype::MxE4m3 && dtypeWeights == btg::Dtype::MxE4m3 &&
      dtypeOutput == btg::Dtype::MxE4m3 && weightLayout == MatrixLayout::MajorK) {
    return generated::kFc1MxFp8;
  }
  return -1;
}

char const* familyName(int family) {
  switch (family) {
    case generated::kFc1Nvfp4:
      return "nvfp4";
    case generated::kFc1Nvfp4PerToken:
      return "nvfp4_bf16tok";
    case generated::kFc1Bf16:
      return "bf16";
    case generated::kFc1Fp8PerTensor:
      return "fp8";
    case generated::kFc1MxFp8:
      return "mxfp8";
    default:
      return "native";
  }
}

}  // namespace

Fc1Runner::Fc1Runner(btg::Dtype dtypeAct, btg::Dtype dtypeWeights, btg::Dtype dtypeOutput,
                     bool useDeepSeekFp8, int tileTokensDim, MoE::ActivationType activationType,
                     bool useShuffledMatrix, batchedGemm::gemm::MatrixLayout weightLayout,
                     batchedGemm::gemm::BiasType biasType, bool usePerTokenScaling,
                     bool usePerChannelScaling)
    : mDtypeAct(dtypeAct),
      mDtypeWeights(dtypeWeights),
      mDtypeOutput(dtypeOutput),
      mTileTokensDim(tileTokensDim),
      mActType(activationType),
      mBiasType(biasType),
      mFamily(selectFamily(dtypeAct, dtypeWeights, dtypeOutput, useDeepSeekFp8, activationType,
                           useShuffledMatrix, weightLayout, biasType, usePerTokenScaling,
                           usePerChannelScaling)) {
  if (mFamily >= 0) {
    for (size_t index = 0; index < generated::kFc1KernelCount; ++index) {
      auto const& spec = generated::kFc1Kernels[index];
      if (spec.family == mFamily && spec.tile_n == tileTokensDim) {
        mKernels.push_back(static_cast<int32_t>(index));
      }
    }
  }
  mSmemConfigured.assign(generated::kFc1KernelCount, false);
  if (mKernels.empty()) {
    mFamily = -1;
    mNative.emplace(dtypeAct, dtypeWeights, dtypeOutput, useDeepSeekFp8, tileTokensDim,
                    activationType, useShuffledMatrix, weightLayout, biasType, usePerTokenScaling,
                    usePerChannelScaling);
  }
}

bool Fc1Runner::shapeSupported(int32_t configIndex, int32_t hiddenSize,
                               int32_t intermediateSize) const {
  if (std::find(mKernels.begin(), mKernels.end(), configIndex) == mKernels.end()) {
    return false;
  }
  auto const& spec = generated::kFc1Kernels[configIndex];
  // K streams in whole BLOCK_K tiles (the block-scale groups divide BLOCK_K); the interleaved
  // up/gate output rows tile by output_rows_per_cta (128 physical weight rows) and a cluster of
  // cluster[0] CTAs covers adjacent output-row tiles.
  int64_t const gridM = intermediateSize / spec.output_rows_per_cta;
  return hiddenSize > 0 && hiddenSize % spec.block_k == 0 && intermediateSize > 0 &&
         intermediateSize % spec.output_rows_per_cta == 0 && gridM % spec.cluster[0] == 0;
}

size_t Fc1Runner::getWorkspaceSizeInBytes(int32_t topK, int32_t hiddenSize,
                                          int32_t intermediateSize, int32_t numExperts,
                                          int32_t numTokens, int32_t configIndex) const {
  if (mNative) {
    return mNative->getWorkspaceSizeInBytes(topK, hiddenSize, intermediateSize, numExperts,
                                            numTokens, configIndex);
  }
  return 0;
}

int32_t Fc1Runner::getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize,
                                              int32_t intermediateSize, int32_t numExperts,
                                              int32_t numTokens) const {
  if (mNative) {
    return mNative->getDefaultValidConfigIndex(topK, hiddenSize, intermediateSize, numExperts,
                                               numTokens);
  }
  for (int32_t index : mKernels) {
    if (shapeSupported(index, hiddenSize, intermediateSize)) {
      return index;
    }
  }
  FLASHINFER_CHECK(false, "No Cake StepFun FC1 kernel (", familyName(mFamily),
                   ") for tile_N=", mTileTokensDim, " accepts hidden_size=", hiddenSize,
                   ", intermediate_size=", intermediateSize,
                   " (hidden_size must be a multiple of the kernel K tile and intermediate_size a "
                   "multiple of its output rows per CTA times the cluster size).");
  return -1;
}

bool Fc1Runner::isValidConfigIndex(int32_t configIndex, int32_t topK, int32_t hiddenSize,
                                   int32_t intermediateSize, int32_t numExperts,
                                   int32_t numTokens) const {
  if (mNative) {
    return mNative->isValidConfigIndex(configIndex, topK, hiddenSize, intermediateSize,
                                       numExperts, numTokens);
  }
  return shapeSupported(configIndex, hiddenSize, intermediateSize);
}

std::vector<int64_t> Fc1Runner::getPassingConfigIndices() const {
  if (mNative) {
    return mNative->getPassingConfigIndices();
  }
  return std::vector<int64_t>(mKernels.begin(), mKernels.end());
}

void Fc1Runner::run(void* hiddenState, void* hiddenStateScale, void* weight, void* weightScale,
                    void* perTokenScales, void* perChannelScales, float* outputScalesScalar,
                    float* outputScalesGateScalar, void* ptrBias, float* ptrGatedActAlpha,
                    float* ptrGatedActBeta, float* ptrClampLimit,
                    int32_t* permutedIdxToBiasRowIdx, void* output, void* outputScale,
                    int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
                    int32_t numExperts, int32_t numTokens, int32_t* permutedIdxToTokenIdx,
                    int32_t* ptrNumNonExitingCtas, int32_t* ptrTotalNumPaddedTokens,
                    int32_t* ptrCtaIdxXyToBatchIdx, int32_t* ptrCtaIdxXyToMnLimit,
                    void* bmm1Workspace, bool useRoutingScalesOnInput, int device,
                    cudaStream_t stream, int32_t configIndex, bool enable_pdl,
                    int32_t validHiddenSize, int32_t validIntermediateSize) {
  if (mNative) {
    mNative->run(hiddenState, hiddenStateScale, weight, weightScale, perTokenScales,
                 perChannelScales, outputScalesScalar, outputScalesGateScalar, ptrBias,
                 ptrGatedActAlpha, ptrGatedActBeta, ptrClampLimit, permutedIdxToBiasRowIdx, output,
                 outputScale, topK, hiddenSize, intermediateSize, numExperts, numTokens,
                 permutedIdxToTokenIdx, ptrNumNonExitingCtas, ptrTotalNumPaddedTokens,
                 ptrCtaIdxXyToBatchIdx, ptrCtaIdxXyToMnLimit, bmm1Workspace,
                 useRoutingScalesOnInput, device, stream, configIndex, enable_pdl,
                 validHiddenSize, validIntermediateSize);
    return;
  }
  char const* const family = familyName(mFamily);
  FLASHINFER_CHECK(shapeSupported(configIndex, hiddenSize, intermediateSize),
                   "Invalid Cake StepFun FC1 config index ", configIndex, " for ", family,
                   " tile_N=", mTileTokensDim, ", hidden_size=", hiddenSize,
                   ", intermediate_size=", intermediateSize);
  FLASHINFER_CHECK(ptrClampLimit != nullptr, "Cake StepFun FC1 (", family,
                   ") requires an explicit per-expert clamp limit (gemm1_clamp_limit)");
  FLASHINFER_CHECK(ptrGatedActAlpha == nullptr && ptrGatedActBeta == nullptr,
                   "SwigluStep accepts gemm1_clamp_limit only; gemm1_alpha / gemm1_beta must be "
                   "absent");
  FLASHINFER_CHECK(ptrBias == nullptr && permutedIdxToBiasRowIdx == nullptr,
                   "Cake StepFun FC1 does not consume a GEMM1 bias");
  FLASHINFER_CHECK(perChannelScales == nullptr,
                   "Cake StepFun FC1 does not consume per-channel scales");
  FLASHINFER_CHECK(!useRoutingScalesOnInput,
                   "Cake StepFun FC1 does not apply routing scales on the input");
  FLASHINFER_CHECK((validHiddenSize < 0 || validHiddenSize == hiddenSize) &&
                       (validIntermediateSize < 0 || validIntermediateSize == intermediateSize),
                   "Cake StepFun FC1 does not support valid (unpadded) dimensions smaller than "
                   "the padded hidden_size / intermediate_size");
  FLASHINFER_CHECK(ptrNumNonExitingCtas != nullptr && permutedIdxToTokenIdx != nullptr &&
                       ptrCtaIdxXyToBatchIdx != nullptr && ptrCtaIdxXyToMnLimit != nullptr,
                   "Cake StepFun FC1 requires the complete trtllm-gen routing arrays");
  bool const blockScaled =
      mFamily == generated::kFc1Nvfp4 || mFamily == generated::kFc1Nvfp4PerToken ||
      mFamily == generated::kFc1MxFp8;
  bool const scalarScaled = mFamily == generated::kFc1Nvfp4 ||
                            mFamily == generated::kFc1Nvfp4PerToken ||
                            mFamily == generated::kFc1Fp8PerTensor;
  FLASHINFER_CHECK(!blockScaled || (hiddenStateScale != nullptr && weightScale != nullptr),
                   "Cake StepFun FC1 (", family,
                   ") requires activation and weight block scales");
  FLASHINFER_CHECK(!scalarScaled ||
                       (outputScalesScalar != nullptr && outputScalesGateScalar != nullptr),
                   "Cake StepFun FC1 (", family, ") requires per-expert output1 scales");
  FLASHINFER_CHECK(mFamily != generated::kFc1Nvfp4PerToken || perTokenScales != nullptr,
                   "Cake StepFun FC1 (nvfp4_bf16tok) requires the fp32 per-token activation scales");
  FLASHINFER_CHECK(mFamily == generated::kFc1Nvfp4PerToken || perTokenScales == nullptr,
                   "Cake StepFun FC1 (", family, ") does not consume per-token scales");
  FLASHINFER_CHECK(mFamily == generated::kFc1Nvfp4PerToken || mFamily == generated::kFc1Bf16 ||
                       mFamily == generated::kFc1Fp8PerTensor || outputScale != nullptr,
                   "Cake StepFun FC1 (", family, ") requires the output block-scale buffer");

  auto const& spec = generated::kFc1Kernels[configIndex];
  int64_t const E = numExperts;
  int64_t const H = hiddenSize;
  int64_t const I = intermediateSize;
  int64_t const T = numTokens;
  int32_t const gridM = intermediateSize / spec.output_rows_per_cta;
  int32_t const gridN =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
  int64_t const maxPaddedTokens =
      Routing::getMaxPermutedPaddedCount(numTokens, topK, numExperts, mTileTokensDim);
  int32_t const kTiles = hiddenSize / spec.block_k;

  generated::Fc1Args args{};
  generated::TensorLayout weightLayout{}, weightScaleLayout{}, activationLayout{},
      activationScaleLayout{}, outputLayout{};
  switch (mFamily) {
    case generated::kFc1Nvfp4:
    case generated::kFc1Nvfp4PerToken:
      // Weights [E, 2I, H/2] packed E2m1 bytes; 128x4 block scales viewed per 128-row weight
      // tile [E * gridM, H/64, 2, 256]; activations [T, H/2] bytes with linear [T, H/16] scales.
      weightLayout = denseLayout(weight, {E, 2 * I, H / 2});
      weightScaleLayout = denseLayout(weightScale, {E * gridM, H / 64, int64_t{2}, int64_t{256}});
      activationLayout = denseLayout(hiddenState, {T, H / 2});
      activationScaleLayout = denseLayout(hiddenStateScale, {T, H / 16});
      // E2m1 output [maxPadded, I/2] bytes; the bf16 per-token output is a plain pointer.
      outputLayout = denseLayout(output, {maxPaddedTokens, I / 2});
      break;
    case generated::kFc1Bf16:
      // BlockMajorK bf16 weights [E, H/64, 2I, 64]; bf16 activations [T, H]; bf16 output.
      weightLayout = denseLayout(weight, {E, H / 64, 2 * I, int64_t{64}});
      activationLayout = denseLayout(hiddenState, {T, H});
      outputLayout = denseLayout(output, {maxPaddedTokens, I});
      break;
    case generated::kFc1Fp8PerTensor:
      // Shuffled MajorK E4m3 weights [E, 2I, H]; E4m3 activations [T, H]; E4m3 output
      // [maxPadded, I] (FlashInfer's buffer is wider; the kernel's row stride is I).
      weightLayout = denseLayout(weight, {E, 2 * I, H});
      activationLayout = denseLayout(hiddenState, {T, H});
      outputLayout = denseLayout(output, {maxPaddedTokens, I});
      break;
    case generated::kFc1MxFp8:
      // Shuffled MajorK MxE4m3 weights [E, 2I, H] with swizzled UE8M0 scales viewed per 128-row
      // weight tile [E * gridM, H/128, 2, 256]; activations [T, H] with linear [T, H/32] scales.
      weightLayout = denseLayout(weight, {E, 2 * I, H});
      weightScaleLayout = denseLayout(weightScale, {E * gridM, H / 128, int64_t{2}, int64_t{256}});
      activationLayout = denseLayout(hiddenState, {T, H});
      activationScaleLayout = denseLayout(hiddenStateScale, {T, H / 32});
      outputLayout = denseLayout(output, {maxPaddedTokens, I});
      break;
    default:
      FLASHINFER_CHECK(false, "Cake StepFun FC1: unknown family ", mFamily);
  }
  FLASHINFER_CHECK(spec.encode_a != nullptr && spec.encode_a(&args.A, weightLayout),
                   "Cake StepFun FC1 (", family, "): failed to encode the weight tensor map");
  if (spec.encode_sfa != nullptr) {
    FLASHINFER_CHECK(spec.encode_sfa(&args.SFA, weightScaleLayout), "Cake StepFun FC1 (", family,
                     "): failed to encode the weight block-scale tensor map");
  }
  if (spec.encode_b != nullptr) {
    FLASHINFER_CHECK(spec.encode_b(&args.B_map, activationLayout), "Cake StepFun FC1 (", family,
                     "): failed to encode the activation tensor map");
  }
  if (spec.encode_sfb != nullptr) {
    FLASHINFER_CHECK(spec.encode_sfb(&args.SFB_map, activationScaleLayout), "Cake StepFun FC1 (",
                     family, "): failed to encode the activation block-scale tensor map");
  }
  if (spec.encode_c != nullptr) {
    FLASHINFER_CHECK(spec.encode_c(&args.C_map, outputLayout), "Cake StepFun FC1 (", family,
                     "): failed to encode the output tensor map");
  }
  args.B_ptr = hiddenState;
  args.SFB_ptr = hiddenStateScale;
  args.C_ptr = output;
  // SFC is the output block-scale buffer, or the per-token activation scales the bf16 per-token
  // kernels gather by token index.
  args.SFC_ptr = mFamily == generated::kFc1Nvfp4PerToken ? perTokenScales : outputScale;
  args.route_map = permutedIdxToTokenIdx;
  args.tile_expert = ptrCtaIdxXyToBatchIdx;
  args.tile_mn_limit = ptrCtaIdxXyToMnLimit;
  args.total_tiles = ptrNumNonExitingCtas;
  args.work_counter = nullptr;
  args.scale_c = outputScalesScalar;
  args.scale_gate = outputScalesGateScalar;
  args.clamp_limit = ptrClampLimit;
  // The SwiGLU-Step epilogue forces alpha = 1 and beta = 0 and reads neither; a valid per-expert
  // vector keeps any kernel read in bounds.
  args.act_alpha = outputScalesScalar != nullptr ? outputScalesScalar : ptrClampLimit;
  args.act_beta = args.act_alpha;
  args.M_out = intermediateSize;
  args.K = hiddenSize;
  args.grid_m = gridM;
  args.grid_n = gridN;
  args.K_tiles = kTiles;

  if (!mSmemConfigured[configIndex]) {
    cudaError_t const configured = spec.configure(spec.dynamic_smem_bytes);
    FLASHINFER_CHECK(configured == cudaSuccess,
                     "Cake StepFun FC1: cudaFuncSetAttribute(MaxDynamicSharedMemorySize=",
                     spec.dynamic_smem_bytes, ") failed for ", spec.symbol, ": ",
                     cudaGetErrorString(configured));
    mSmemConfigured[configIndex] = true;
  }

  // Diagnostic override: CAKE_STEPFUN_FC1_PDL=0 launches the FC1 kernel without the programmatic
  // stream-serialization attribute, =1 launches it with the attribute, while the rest of the
  // pipeline keeps the caller's PDL setting; unset follows enable_pdl.
  static int const fc1PdlOverride = [] {
    char const* value = std::getenv("CAKE_STEPFUN_FC1_PDL");
    if (value == nullptr || value[0] == '\0' || value[1] != '\0') return -1;
    return value[0] == '0' ? 0 : value[0] == '1' ? 1 : -1;
  }();
  bool const fc1Pdl = fc1PdlOverride < 0 ? enable_pdl : fc1PdlOverride == 1;
  cudaLaunchAttribute pdlAttribute{};
  pdlAttribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdlAttribute.val.programmaticStreamSerializationAllowed = 1;

  if (!spec.bounds_acquired_tiles && mPadRoutingTail) {
    cudaLaunchConfig_t tailConfig{};
    tailConfig.gridDim = dim3(kRoutingTailBlocks, 1u, 1u);
    tailConfig.blockDim = dim3(kRoutingTailThreads, 1u, 1u);
    tailConfig.dynamicSmemBytes = 0;
    tailConfig.stream = stream;
    tailConfig.attrs = &pdlAttribute;
    tailConfig.numAttrs = fc1Pdl ? 1u : 0u;
    cudaError_t const padded =
        cudaLaunchKernelEx(&tailConfig, padRoutingTailKernel, ptrCtaIdxXyToBatchIdx,
                           ptrCtaIdxXyToMnLimit, permutedIdxToTokenIdx,
                           static_cast<int32_t const*>(ptrNumNonExitingCtas), gridN,
                           mTileTokensDim);
    FLASHINFER_CHECK(padded == cudaSuccess, "Cake StepFun FC1 routing-tail launch failed for ",
                     spec.symbol, " grid_n=", gridN, " : ", cudaGetErrorString(padded));
  }

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(gridM), static_cast<unsigned>(gridN), 1u);
  config.blockDim = dim3(spec.block[0], spec.block[1], spec.block[2]);
  config.dynamicSmemBytes = spec.dynamic_smem_bytes;
  config.stream = stream;
  std::array<cudaLaunchAttribute, 2> attributes{};
  unsigned numAttrs = 0;
  if (fc1Pdl) {
    attributes[numAttrs] = pdlAttribute;
    ++numAttrs;
  }
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
  FLASHINFER_CHECK(launched == cudaSuccess, "Cake StepFun FC1 launch failed for ", spec.symbol,
                   " grid=(", gridM, ",", gridN, ") : ", cudaGetErrorString(launched));
}

}  // namespace cake_stepfun
}  // namespace trtllmgen_moe
}  // namespace kernels
}  // namespace tensorrt_llm
