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
#pragma once

// Cake StepFun FC1 runner: the FC1 (GEMM1 + fused SwiGLU-Step activation) stage of the trtllm-gen
// fused-MoE pipeline executed by exported Cake kernels. It replaces PermuteGemm1::Runner inside
// MoE::Runner when the module is compiled with -DCAKE_STEPFUN_FC1 and exposes the same surface:
// constructor, workspace query, config-index validity, and run() over the trtllm-gen routing ABI
// (permuted_idx_to_token_idx, cta_idx_xy_to_batch_idx, cta_idx_xy_to_mn_limit,
// num_non_exiting_ctas). Routing, FC2, and finalize stay the native trtllm-gen kernels.
//
// The exported kernels form five families selected by the constructor's dtype / layout / scaling
// arguments exactly as the native cubin selection does: NVFP4 (E2m1 output), NVFP4 with the fp32
// per-token activation scale (bf16 output), BF16 (BlockMajorK weights), per-tensor FP8 and MXFP8,
// each over the tokens-per-CTA tiles the generated inventory covers. Config indices are positions
// in the generated kernel table. (dtype, tile) pairs without an exported Cake kernel fall back to
// the native PermuteGemm1 runner so the fused-MoE tile planner keeps its complete candidate set.
//
// This header is included from include/flashinfer/trtllm/fused_moe/runner.h after the PermuteGemm1
// and Routing declarations it relies on.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace tensorrt_llm {
namespace kernels {
namespace trtllmgen_moe {
namespace cake_stepfun {

class Fc1Runner {
 public:
  explicit Fc1Runner(batchedGemm::trtllm::gen::Dtype dtypeAct,
                     batchedGemm::trtllm::gen::Dtype dtypeWeights,
                     batchedGemm::trtllm::gen::Dtype dtypeOutput, bool useDeepSeekFp8,
                     int tileTokensDim, MoE::ActivationType activationType,
                     bool useShuffledMatrix, batchedGemm::gemm::MatrixLayout weightLayout,
                     batchedGemm::gemm::BiasType biasType, bool usePerTokenScaling,
                     bool usePerChannelScaling);

  size_t getWorkspaceSizeInBytes(int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
                                 int32_t numExperts, int32_t numTokens, int32_t configIndex) const;

  [[nodiscard]] int32_t getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize,
                                                   int32_t intermediateSize, int32_t numExperts,
                                                   int32_t numTokens) const;

  [[nodiscard]] bool isValidConfigIndex(int32_t configIndex, int32_t topK, int32_t hiddenSize,
                                        int32_t intermediateSize, int32_t numExperts,
                                        int32_t numTokens) const;

  [[nodiscard]] std::vector<int64_t> getPassingConfigIndices() const;

  // True when this (dtype, tile) runs the exported Cake kernels (false: native fallback).
  [[nodiscard]] bool usesCakeKernels() const { return mNative == std::nullopt; }

  // Generated-manifest family index of this runner (-1 on the native fallback).
  [[nodiscard]] int family() const { return mFamily; }

  // Same argument list as PermuteGemm1::Runner::run. Per family the Cake kernels consume:
  //  NVFP4            hiddenState E2m1, hiddenStateScale linear E4m3 blocks, weight / weightScale
  //                   (trtllm-shuffled E2m1 + 128x4 block scales), outputScalesScalar /
  //                   outputScalesGateScalar / ptrClampLimit (raw units); output E2m1 with 8x4
  //                   block scales in outputScale.
  //  NVFP4 per-token  as NVFP4 plus perTokenScales (fp32 per token); output bf16 (FlashInfer
  //                   quantizes the GEMM2 input afterwards).
  //  BF16             hiddenState bf16, weight bf16 BlockMajorK, ptrClampLimit (physical units);
  //                   output bf16.
  //  FP8 per-tensor   hiddenState / weight E4m3 (shuffled MajorK), outputScalesScalar /
  //                   outputScalesGateScalar / ptrClampLimit (raw units); output E4m3.
  //  MXFP8            hiddenState / weight MxE4m3 with UE8M0 block scales (linear activation
  //                   scales, swizzled weight scales), ptrClampLimit (physical units); output
  //                   MxE4m3 with swizzled block scales in outputScale.
  void run(void* hiddenState, void* hiddenStateScale, void* weight, void* weightScale,
           void* perTokenScales, void* perChannelScales, float* outputScalesScalar,
           float* outputScalesGateScalar, void* ptrBias, float* ptrGatedActAlpha,
           float* ptrGatedActBeta, float* ptrClampLimit, int32_t* permutedIdxToBiasRowIdx,
           void* output, void* outputScale, int32_t topK, int32_t hiddenSize,
           int32_t intermediateSize, int32_t numExperts, int32_t numTokens,
           int32_t* permutedIdxToTokenIdx, int32_t* ptrNumNonExitingCtas,
           int32_t* ptrTotalNumPaddedTokens, int32_t* ptrCtaIdxXyToBatchIdx,
           int32_t* ptrCtaIdxXyToMnLimit, void* bmm1Workspace, bool useRoutingScalesOnInput,
           int device, cudaStream_t stream, int32_t configIndex, bool enable_pdl,
           int32_t validHiddenSize = -1, int32_t validIntermediateSize = -1);

  // Read by MoE::Runner::run exactly as PermuteGemm1::Runner's members of the same names.
  batchedGemm::trtllm::gen::Dtype mDtypeAct;
  batchedGemm::trtllm::gen::Dtype mDtypeWeights;
  batchedGemm::trtllm::gen::Dtype mDtypeOutput;
  int32_t mTileTokensDim;
  MoE::ActivationType mActType;
  batchedGemm::gemm::BiasType mBiasType{batchedGemm::gemm::BiasType::None};

 private:
  bool shapeSupported(int32_t configIndex, int32_t hiddenSize, int32_t intermediateSize) const;

  int mFamily{-1};
  // Generated kernel table positions serving (mFamily, mTileTokensDim) on this module's arch.
  std::vector<int32_t> mKernels;
  // Dynamic shared memory opt-in done once per kernel on this runner's device.
  mutable std::vector<bool> mSmemConfigured;
  // Native FC1 for (dtype, tile) pairs without an exported Cake kernel.
  std::optional<PermuteGemm1::Runner> mNative;
};

}  // namespace cake_stepfun
}  // namespace trtllmgen_moe
}  // namespace kernels
}  // namespace tensorrt_llm
