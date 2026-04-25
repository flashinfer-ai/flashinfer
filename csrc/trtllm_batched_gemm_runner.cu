/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstring>
#include <vector>

#include "flashinfer/trtllm/batched_gemm/KernelRunner.h"
// #include "tensorrt_llm/common/assert.h"
#include "flashinfer/exception.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/BatchedGemmInterface.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/Enums.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"
#include "flashinfer/trtllm/common.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"

namespace tensorrt_llm {
namespace kernels {

using namespace batchedGemm::batchedGemm;
using namespace batchedGemm::gemm;
using namespace batchedGemm::trtllm::gen;

static BatchedGemmInterface::ModuleCache globalTrtllmGenBatchedGemmModuleCache;

static inline bool skipQuirks(BatchedGemmConfig const& config) {
  // Skip kernels that are known to hang/crash. Keep a record here for future reference.
  auto const& options = config.mOptions;
  // FC1 128x128 batchM mNumWarpsLoadSfA=4 (=2 is ok)
  // bmm_E2m1_E2m1E2m1_Fp32_bA16_bB16_bC16_t128x128x256_s6_et128x64_m256x128x64_c2x1x1_32dp32b_rN_TN_schPd2x1x2x3_biasFp32N_bM_tma_ldgstsSf_rgTma_clmp_swiGlu_lsfaW4_dynB_sm100f
  bool const isKnownHangingSm100fSwigluLsfaW4Family =
      config.mSm == SmVersion::Sm100f && options.mDtypeA == tg::Dtype::E2m1 &&
      options.mDtypeB == tg::Dtype::E2m1 && options.mTileM == 128 && options.mTileN == 128 &&
      options.mTileK == 256 && options.mNumStages == 6 && options.mClusterDimX == 2 &&
      options.mClusterDimY == 1 && options.mClusterDimZ == 1 &&
      !doesRouteImplUseNoRoute(options.mRouteImpl) && !options.mTransposeMmaOutput &&
      options.mTileScheduler == TileScheduler::Persistent && options.mFusedAct &&
      options.mActType == batchedGemm::gemmGatedAct::ActType::SwiGlu &&
      options.mNumWarpsLoadSfA == 4;

  return isKnownHangingSm100fSwigluLsfaW4Family;
}

std::vector<int64_t> prioritizePredefinedConfigs(
    int m, int n, int k, std::vector<int64_t> const& sortedIndices,
    batchedGemm::batchedGemm::BatchedGemmConfig const* configs) {
  // Function to bubble up the pre-determined config.
  auto bubbleUpConfig = [&configs](std::vector<int64_t> const& sortedIndices,
                                   auto&& pred) -> std::vector<int64_t> {
    std::vector<int64_t> prioritizedIndices_;
    // Copy matching configs to new vector
    std::copy_if(sortedIndices.begin(), sortedIndices.end(),
                 std::back_inserter(prioritizedIndices_), [&configs, &pred](int idx) {
                   BatchedGemmConfig const& config = configs[idx];
                   return (pred(config));
                 });
    // Copy the rest of the configs to new vector, if not already copied
    std::copy_if(sortedIndices.begin(), sortedIndices.end(),
                 std::back_inserter(prioritizedIndices_), [&prioritizedIndices_](int idx) {
                   return std::find(prioritizedIndices_.begin(), prioritizedIndices_.end(), idx) ==
                          prioritizedIndices_.end();
                 });
    return prioritizedIndices_;
  };

  // Init empty vector
  std::vector<int64_t> prioritizedIndices;

  //
  // Dummy
  //

  if (n /* out_dim */ == 0 && k /* in_dim */ == 0) {
    auto pred = [](BatchedGemmConfig const& config) {
      BatchedGemmOptions const& options = config.mOptions;
      return options.mNumStagesA == 4 && options.mNumStagesB == 4 && options.mNumStagesMma == 2 &&
             options.mTileK == 256 && options.mTileScheduler == TileScheduler::Persistent;
    };
    prioritizedIndices = bubbleUpConfig(sortedIndices, pred);
  }
  //
  // Fall back
  //
  else {
    prioritizedIndices = sortedIndices;
  }

  return prioritizedIndices;
}

static inline void setProblemDimensions(BatchedGemmData& gemmData, bool transposeMmaOutput,
                                        int32_t m, int32_t n, int32_t k,
                                        std::vector<int32_t> const& batchedTokens,
                                        int32_t numTokens, int32_t numBatches,
                                        int32_t maxNumCtasInBatchDim) {
  gemmData.mProblemDimensions.mNumBatches = numBatches;
  gemmData.mProblemDimensions.mNumTokens = numTokens;
  gemmData.mProblemDimensions.mBatchM = !transposeMmaOutput;
  gemmData.mProblemDimensions.mBatchedM =
      transposeMmaOutput ? std::vector<int32_t>{} : batchedTokens;
  gemmData.mProblemDimensions.mBatchedN =
      transposeMmaOutput ? batchedTokens : std::vector<int32_t>{};
  gemmData.mProblemDimensions.mM = transposeMmaOutput ? n : m;
  gemmData.mProblemDimensions.mN = transposeMmaOutput ? m : n;
  gemmData.mProblemDimensions.mK = k;
  gemmData.mProblemDimensions.mValidM = gemmData.mProblemDimensions.mM;
  gemmData.mProblemDimensions.mValidN = gemmData.mProblemDimensions.mN;
  gemmData.mProblemDimensions.mValidK = gemmData.mProblemDimensions.mK;
  gemmData.mProblemDimensions.mRank = 0;
  gemmData.mProblemDimensions.mWorldSize = 1;
  gemmData.mProblemDimensions.mMaxNumCtasInTokenDim = maxNumCtasInBatchDim;
}

TrtllmGenBatchedGemmRunner::TrtllmGenBatchedGemmRunner(
    TrtllmGenBatchedGemmRunnerOptions const& options_)
    : mOptions(options_) {
  // Select a GEMM kernel config to use
  auto const bmm = BatchedGemmInterface();
  auto const configs = bmm.getBatchedGemmConfigs();

  mPassingConfigIndices.clear();

  for (size_t i = 0; i < bmm.getNumBatchedGemmConfigs(); ++i) {
    // The kernel config.
    auto const& config = configs[i];
    auto const& options = config.mOptions;
    // The tile size in CGA granularity.
    auto const tileSize = options.mTransposeMmaOutput ? options.mTileN * options.mClusterDimY
                                                      : options.mTileM * options.mClusterDimX;
    // Check if kernel dtype matches runner config.
    bool const dtypeMatch =
        options.mTransposeMmaOutput
            ? (options.mDtypeA == mOptions.dtypeB && options.mDtypeB == mOptions.dtypeA)
            : (options.mDtypeA == mOptions.dtypeA && options.mDtypeB == mOptions.dtypeB);
    // Check if kernel weight layout matches runner config.
    bool const layoutAndShuffleMatch =
        options.mTransposeMmaOutput ? (options.mUseShuffledMatrix == mOptions.useShuffledMatrix &&
                                       options.mLayoutA == mOptions.weightLayout)
                                    : (options.mUseShuffledMatrix == mOptions.useShuffledMatrix &&
                                       options.mLayoutB == mOptions.weightLayout);
    if (dtypeMatch && options.mDtypeC == mOptions.dtypeC &&
        options.mUseDeepSeekFp8 == mOptions.deepSeekFp8 &&
        (!doesRouteImplUseNoRoute(options.mRouteImpl)) == mOptions.routeAct &&
        options.mFusedAct == mOptions.fusedAct && options.mIsStaticBatch == mOptions.staticBatch &&
        tileSize == mOptions.tileSize && layoutAndShuffleMatch) {
      if (options.mFusedAct) {
        if (options.mActType != static_cast<batchedGemm::gemmGatedAct::ActType>(mOptions.actType)) {
          continue;
        }
      }
      if ((int64_t)options.mEltwiseActType != (int64_t)mOptions.eltwiseActType) {
        continue;
      }
      if (skipQuirks(config)) {
        continue;
      }
      if (options.mEpilogueTileM == mOptions.epilogueTileM) {
        mPassingConfigIndices.push_back(i);
      }
    }
  }

  std::ostringstream error_msg;
  error_msg << "No kernel found for the given options: "
            << "mDtypeA: " << tg::dtypeToString(mOptions.dtypeA)
            << ", mDtypeB: " << tg::dtypeToString(mOptions.dtypeB)
            << ", mDtypeC: " << tg::dtypeToString(mOptions.dtypeC)
            << ", mUseDeepSeekFp8: " << mOptions.deepSeekFp8
            << ", mActType: " << (int64_t)mOptions.actType
            << ", mEltwiseActType: " << (int64_t)mOptions.eltwiseActType
            << ", mTransposeMmaOutput: auto-tuned"
            << ", mRouteAct: " << mOptions.routeAct << ", mFusedAct: " << mOptions.fusedAct
            << ", mIsStaticBatch: " << mOptions.staticBatch << ", mTileSize: " << mOptions.tileSize;
  FLASHINFER_CHECK(!mPassingConfigIndices.empty(), error_msg.str());
}

size_t TrtllmGenBatchedGemmRunner::getWorkspaceSizeInBytes(
    int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens, int32_t numTokens,
    int32_t numBatches, int32_t maxNumCtasInBatchDim, int32_t configIndex) const {
  auto bmm = BatchedGemmInterface();
  auto const configs = bmm.getBatchedGemmConfigs();
  auto const& config = configs[configIndex];

  BatchedGemmData gemmData{};
  setProblemDimensions(gemmData, config.mOptions.mTransposeMmaOutput, m, n, k, batchedTokens,
                       numTokens, numBatches, maxNumCtasInBatchDim);

  return bmm.getWorkspaceSizeInBytes(config, gemmData);
}

void TrtllmGenBatchedGemmRunner::run(
    int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens, int32_t numTokens,
    int32_t numBatches, int32_t maxNumCtasInBatchDim, void const* a, void const* sfA, void const* b,
    void const* sfB, void const* perTokensSfA, void const* perTokensSfB, float const* scaleC,
    float const* scaleGateC, float const* ptrBias, float const* ptrAlpha, float const* ptrBeta,
    float const* ptrClampLimit, void* c, void* outSfC, int32_t const* routeMap,
    int32_t const* totalNumPaddedTokens, int32_t const* ctaIdxXyToBatchIdx,
    int32_t const* ctaIdxXyToMnLimit, int32_t const* numNonExitingCtas, void* workspace,
    CUstream stream, int device, int32_t configIndex, bool enable_pdl) {
  auto bmm = BatchedGemmInterface();

  BatchedGemmData gemmData{};

  auto const configs = bmm.getBatchedGemmConfigs();

  auto const& config = configs[configIndex];
  bool const transposeMmaOutput = config.mOptions.mTransposeMmaOutput;

  FLASHINFER_CHECK(numBatches > 0, "Batched GEMM requires numBatches > 0");
  if (!mOptions.staticBatch) {
    FLASHINFER_CHECK(totalNumPaddedTokens,
                     "Batched GEMM with dynamic batching requires totalNumPaddedTokens");
    FLASHINFER_CHECK(ctaIdxXyToBatchIdx,
                     "Batched GEMM with dynamic batching requires ctaIdxXyToBatchIdx");
    FLASHINFER_CHECK(ctaIdxXyToMnLimit,
                     "Batched GEMM with dynamic batching requires ctaIdxXyToMnLimit");
    FLASHINFER_CHECK(numNonExitingCtas,
                     "Batched GEMM with dynamic batching requires numNonExitingCtas");
  }

  if (!mOptions.staticBatch && numTokens != 0) {
    FLASHINFER_CHECK(maxNumCtasInBatchDim > 0,
                     "Batched GEMM with dynamic batching requires maxNumCtasInBatchDim > 0");
  }

  if (mOptions.routeAct) {
    FLASHINFER_CHECK(routeMap, "Batched GEMM with routeAct requires routeMap");
    FLASHINFER_CHECK(numTokens > 0, "Batched GEMM with routeAct requires numTokens > 0");
  }

  // Dims
  setProblemDimensions(gemmData, transposeMmaOutput, m, n, k, batchedTokens, numTokens, numBatches,
                       maxNumCtasInBatchDim);

  // Inputs
  gemmData.mInputBuffers.mPtrA = transposeMmaOutput ? b : a;
  gemmData.mInputBuffers.mPtrSfA = transposeMmaOutput ? sfB : sfA;
  gemmData.mInputBuffers.mPtrB = transposeMmaOutput ? a : b;
  gemmData.mInputBuffers.mPtrSfB = transposeMmaOutput ? sfA : sfB;
  gemmData.mInputBuffers.mPtrScaleC = scaleC;
  gemmData.mInputBuffers.mPtrScaleGate = scaleGateC;
  // For simplicity pass set scaleAct to scaleGateC
  gemmData.mInputBuffers.mPtrScaleAct = scaleGateC;
  gemmData.mInputBuffers.mPtrPerTokenSfA = transposeMmaOutput ? perTokensSfB : perTokensSfA;
  gemmData.mInputBuffers.mPtrPerTokenSfB = transposeMmaOutput ? perTokensSfA : perTokensSfB;
  gemmData.mInputBuffers.mPtrBias = ptrBias;
  gemmData.mInputBuffers.mPtrGatedActAlpha = ptrAlpha;
  gemmData.mInputBuffers.mPtrGatedActBeta = ptrBeta;
  gemmData.mInputBuffers.mPtrClampLimit = ptrClampLimit;

  gemmData.mInputBuffers.mPtrRouteMap = routeMap;

  // Pointer to total number of padded tokens
  gemmData.mInputBuffers.mPtrTotalNumPaddedTokens = totalNumPaddedTokens;
  gemmData.mInputBuffers.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
  gemmData.mInputBuffers.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
  gemmData.mInputBuffers.mPtrNumNonExitingCtas = numNonExitingCtas;

  // Outputs
  gemmData.mOutputBuffers.mPtrC = c;
  gemmData.mOutputBuffers.mPtrSfC = outSfC;

  int32_t multiProcessorCount;
  cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device);

  if (getBoolEnv("TRTLLM_BATCHED_GEMM_PRINT_NAME")) {
    FLASHINFER_LOG("NumBatches", numBatches, ", MaxNumCgasInBatchDim", maxNumCtasInBatchDim,
                   ", MaxNumCtasInBatchDim", maxNumCtasInBatchDim, ", ShapeMNK",
                   gemmData.mProblemDimensions.mM, gemmData.mProblemDimensions.mN,
                   gemmData.mProblemDimensions.mK, ", ValidShapeMNK",
                   gemmData.mProblemDimensions.mValidM, gemmData.mProblemDimensions.mValidN,
                   gemmData.mProblemDimensions.mValidK, ", Kernel", config.mFunctionName);
  }

  // FIXME once we start using all-reduce in the epilogue of the bmm this can be moved elsewhere
  bmm.runInitBeforeWorldSync(config, gemmData, static_cast<void*>(stream));

  auto const err =
      bmm.run(config, workspace, gemmData, static_cast<void*>(stream), multiProcessorCount,
              enable_pdl, /*pinnedHostBuffer=*/nullptr, globalTrtllmGenBatchedGemmModuleCache);

  FLASHINFER_CHECK(err == 0,
                   "Error occurred when running GEMM!"
                   " (numBatches: ",
                   numBatches, ", GemmMNK: ", m, " ", n, " ", k, ", Kernel: ", config.mFunctionName,
                   ", transposeMmaOutput: ", transposeMmaOutput, ", configIndex: ", configIndex,
                   ", maxNumCtasInBatchDim: ", maxNumCtasInBatchDim,
                   ", maxNumCtasInBatchDim: ", maxNumCtasInBatchDim, ")");
}

void TrtllmGenBatchedGemmRunner::run(int32_t m, int32_t n, int32_t k,
                                     std::vector<int32_t> const& batchedTokens, void const* a,
                                     void const* sfA, void const* b, void const* sfB, void* c,
                                     void* outSfC, void* workspace, CUstream stream, int device,
                                     int32_t configIndex, bool enable_pdl) {
  // Dispatch with block scaling factors and with static batching.
  run(m, n, k, batchedTokens, /* numTokens */ 0, batchedTokens.size(), /* maxNumCtasInBatchDim */ 0,
      a, sfA, b, sfB,
      /* perTokensSfA */ nullptr, /* perTokensSfB */ nullptr,
      /* scaleC */ nullptr, /* scaleGateC */ nullptr, /* ptrBias */ nullptr, /* ptrAlpha */ nullptr,
      /* ptrBeta */ nullptr, /* ptrClampLimit */ nullptr, c, outSfC,
      /* routeMap */ nullptr, /* totalNumPaddedTokens */ nullptr,
      /* ctaIdxXyToBatchIdx */ nullptr, /* ctaIdxXyToMnLimit */ nullptr,
      /* numNonExitingCtas */ nullptr, workspace, stream, device, configIndex, enable_pdl);
}

void TrtllmGenBatchedGemmRunner::run(int32_t m, int32_t n, int32_t k,
                                     std::vector<int32_t> const& batchedTokens, void const* a,
                                     void const* sfA, void const* b, void const* sfB,
                                     float const* ptrBias, float const* ptrAlpha,
                                     float const* ptrBeta, float const* ptrClampLimit, void* c,
                                     void* outSfC, void* workspace, CUstream stream, int device,
                                     int32_t configIndex, bool enable_pdl) {
  // Dispatch with block scaling factors and with static batching.
  run(m, n, k, batchedTokens, /* numTokens */ 0, batchedTokens.size(), /* maxNumCtasInBatchDim */ 0,
      a, sfA, b, sfB,
      /* perTokensSfA */ nullptr, /* perTokensSfB */ nullptr,
      /* scaleC */ nullptr, /* scaleGateC */ nullptr, ptrBias, ptrAlpha, ptrBeta, ptrClampLimit, c,
      outSfC,
      /* routeMap */ nullptr, /* totalNumPaddedTokens */ nullptr,
      /* ctaIdxXyToBatchIdx */ nullptr, /* ctaIdxXyToMnLimit */ nullptr,
      /* numNonExitingCtas */ nullptr, workspace, stream, device, configIndex, enable_pdl);
}

void TrtllmGenBatchedGemmRunner::run(int32_t m, int32_t n, int32_t k,
                                     std::vector<int32_t> const& batchedTokens, void const* a,
                                     void const* b, float const* scaleC, float const* scaleGateC,
                                     void* c, void* workspace, CUstream stream, int device,
                                     int32_t configIndex, bool enable_pdl) {
  // Dispatch with block scaling factors and with static batching.
  run(m, n, k, batchedTokens, /* numTokens */ 0, batchedTokens.size(), /* maxNumCtasInBatchDim */ 0,
      a,
      /* sfA */ nullptr, b, /* sfB */ nullptr, /* perTokensSfA */ nullptr,
      /* perTokensSfB */ nullptr, scaleC, scaleGateC, /* ptrBias */ nullptr, /* ptrAlpha */ nullptr,
      /* ptrBeta */ nullptr, /* ptrClampLimit */ nullptr, c,
      /* outSfC */ nullptr,
      /* routeMap */ nullptr, /* totalNumPaddedTokens */ nullptr,
      /* ctaIdxXyToBatchIdx */ nullptr, /* ctaIdxXyToMnLimit */ nullptr,
      /* numNonExitingCtas */ nullptr, workspace, stream, device, configIndex, enable_pdl);
}

std::vector<int64_t> TrtllmGenBatchedGemmRunner::getValidConfigIndices(
    int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens, int32_t numTokens,
    int32_t numBatches, int32_t maxNumCtasInBatchDim) const {
  auto const bmm = BatchedGemmInterface();
  auto const configs = bmm.getBatchedGemmConfigs();

  int32_t multiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();

  auto cmpFunc = [&configs, &bmm, &multiProcessorCount, &m, &n, &k, &batchedTokens, &numTokens,
                  &numBatches, &maxNumCtasInBatchDim](int64_t idx0, int64_t idx1) {
    auto const& optionsA = configs[idx0].mOptions;
    auto const& optionsB = configs[idx1].mOptions;
    int32_t sizeK = k;

    // Keep comparator stable across mixed transpose modes.
    if (optionsA.mTransposeMmaOutput != optionsB.mTransposeMmaOutput) {
      return optionsA.mTransposeMmaOutput;
    }

    // Tier 0: K < tileK, prefer higher efficiency.
    if (optionsA.mTileK != optionsB.mTileK) {
      // Both waste computation, prefer higher efficiency.
      if (sizeK <= optionsA.mTileK && sizeK <= optionsB.mTileK) {
        double eff_a = (double)sizeK / optionsA.mTileK;
        double eff_b = (double)sizeK / optionsB.mTileK;
        return eff_a > eff_b;
      }
      // If either can be utilized, sort by tileK.
      else {
        return optionsA.mTileK > optionsB.mTileK;
      }
    }

    // Tier 1: When tileK is the same, prefer unroll loop 2x for mma.
    if (optionsA.mUseUnrollLoop2xForMma != optionsB.mUseUnrollLoop2xForMma) {
      return optionsA.mUseUnrollLoop2xForMma;
    }

    // Tier 2+: When previous comparators are the same, prefer higher tileM.
    if (optionsA.mTileM != optionsB.mTileM) {
      return optionsA.mTileM > optionsB.mTileM;
    }

    // Tier 2+: When previous comparators are the same, prefer higher tileN.
    if (optionsA.mTileN != optionsB.mTileN) {
      return optionsA.mTileN > optionsB.mTileN;
    }

    // Tier 2+: When previous comparators are the same, and when the number of estimated CTAs is on
    // the larger side, prefer persistent tile scheduler.
    if (optionsA.mTileScheduler != optionsB.mTileScheduler) {
      BatchedGemmData gemmData{};
      setProblemDimensions(gemmData, optionsA.mTransposeMmaOutput, m, n, k, batchedTokens,
                           numTokens, numBatches, maxNumCtasInBatchDim);
      auto options = bmm.getOptionsFromConfigAndData(configs[idx0], gemmData);
      auto numCtas = bmm.getNumCtas(options, gemmData.mProblemDimensions.mMaxNumCtasInTokenDim);
      if (numCtas > multiProcessorCount) {
        return optionsA.mTileScheduler == batchedGemm::gemm::TileScheduler::Persistent;
      } else {
        return optionsB.mTileScheduler == batchedGemm::gemm::TileScheduler::Persistent;
      }
    }

    return false;
  };

  // Sort configs by options.
  std::vector<int64_t> sortedIndices = mPassingConfigIndices;
  std::sort(sortedIndices.begin(), sortedIndices.end(), cmpFunc);

  // Special rules for corner cases, if applicable.
  std::vector<int64_t> prioritizedIndices =
      prioritizePredefinedConfigs(m, n, k, sortedIndices, configs);

  // Filter out invalid configs.
  std::vector<int64_t> validConfigIndices;
  for (auto const& configIndex : prioritizedIndices) {
    BatchedGemmData gemmData{};
    auto const transposeMmaOutput = configs[configIndex].mOptions.mTransposeMmaOutput;
    setProblemDimensions(gemmData, transposeMmaOutput, m, n, k, batchedTokens, numTokens,
                         numBatches, maxNumCtasInBatchDim);
    auto isValidConfig = bmm.isValidConfig(configs[configIndex], gemmData);
    if (isValidConfig) {
      validConfigIndices.push_back(configIndex);
    }
  }

  std::ostringstream error_msg;
  if (validConfigIndices.empty()) {
    int64_t numTransposeConfigs = 0;
    for (auto const& configIndex : prioritizedIndices) {
      if (configs[configIndex].mOptions.mTransposeMmaOutput) {
        ++numTransposeConfigs;
      }
    }
    error_msg << "No valid config found for the given problem shape"
              << " (m=" << m << ", n=" << n << ", k=" << k << ", numTokens=" << numTokens
              << ", numBatches=" << numBatches << ", maxNumCtasInBatchDim=" << maxNumCtasInBatchDim
              << ", passingConfigs=" << mPassingConfigIndices.size()
              << ", prioritizedConfigs=" << prioritizedIndices.size()
              << ", transposeConfigs=" << numTransposeConfigs
              << ", nonTransposeConfigs=" << (prioritizedIndices.size() - numTransposeConfigs)
              << ")";
  }
  FLASHINFER_CHECK(!validConfigIndices.empty(), error_msg.str());

  return validConfigIndices;
}

int64_t TrtllmGenBatchedGemmRunner::getDefaultValidConfigIndex(
    int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens, int32_t numTokens,
    int32_t numBatches, int32_t maxNumCtasInBatchDim) const {
  auto const validConfigIndices =
      getValidConfigIndices(m, n, k, batchedTokens, numTokens, numBatches, maxNumCtasInBatchDim);

  return validConfigIndices[0];
}

bool TrtllmGenBatchedGemmRunner::isValidConfigIndex(int32_t configIndex, int32_t m, int32_t n,
                                                    int32_t k,
                                                    std::vector<int32_t> const& batchedTokens,
                                                    int32_t numTokens, int32_t numBatches,
                                                    int32_t maxNumCtasInBatchDim) const {
  auto const bmm = BatchedGemmInterface();
  auto const configs = bmm.getBatchedGemmConfigs();
  auto const& config = configs[configIndex];

  BatchedGemmData gemmData{};
  setProblemDimensions(gemmData, config.mOptions.mTransposeMmaOutput, m, n, k, batchedTokens,
                       numTokens, numBatches, maxNumCtasInBatchDim);

  return bmm.isValidConfig(config, gemmData);
}

}  // namespace kernels
}  // namespace tensorrt_llm
