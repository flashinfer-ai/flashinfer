/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include <cuda.h>

#include <cstdint>
#include <cuda/std/cfloat>
#include <iterator>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "../../exception.h"
#include "../../utils.cuh"
#include "../common.h"
#include "cuda_runtime_api.h"
#include "flashInferMetaInfo.h"
#include "fmhaReduction.h"
#include "fmhaRunnerParams.h"
#include "kernelParams.h"
#include "lse.cuh"

#ifdef TLLM_GEN_FMHA_CUBIN_PATH
static const std::string tllm_gen_fmha_cubin_path = std::string(TLLM_GEN_FMHA_CUBIN_PATH);
#else
static_assert(false, "TLLM_GEN_FMHA_CUBIN_PATH macro is not defined when compiling");
#endif

#ifdef TLLM_GEN_FMHA_METAINFO_HASH
static const std::string tllm_gen_fmha_metainfo_hash = std::string(TLLM_GEN_FMHA_METAINFO_HASH);
#else
static_assert(false, "TLLM_GEN_FMHA_METAINFO_HASH macro is not defined when compiling");
#endif

namespace flashinfer::trtllm_cubin_loader {
std::string getCubin(const std::string& kernelName, const std::string& sha256);
}  // namespace flashinfer::trtllm_cubin_loader
using flashinfer::trtllm_cubin_loader::getCubin;

// Check if two SM values are family/specific versions of the same architecture
// Returns true only if one is a family version and the other is a compatible specific version
constexpr bool isFamilySpecificSMPair(int sm1, int sm2) {
  if ((sm1 == kSM_100f && (sm2 == kSM_100 || sm2 == kSM_103)) ||
      (sm2 == kSM_100f && (sm1 == kSM_100 || sm1 == kSM_103))) {
    return true;
  }
  return false;
}

constexpr bool isSMCompatible(int gpuSM, int kernelSM) {
  if (gpuSM == kSM_103) {
    return kernelSM == kSM_100f || kernelSM == kSM_103;
  } else if (gpuSM == kSM_100) {
    return kernelSM == kSM_100f || kernelSM == kSM_100;
  }

  return gpuSM == kernelSM;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
class TllmGenFmhaKernel {
 public:
  // The parameters for launching the kernel.
  // maxNumCtasQ, maxNumCtasKv, numCtasX, numCtasY, numCtasZ, clusterDimX
  struct CtaLaunchParams {
    // The maximum number of CTAs in Q dimension.
    int mMaxNumCtasQ;
    // The maximum number of CTAs in Kv dimension.
    int mMaxNumCtasKv;
    // The number of CTAs in X dimension.
    int mNumCtasX;
    // The number of CTAs in Y dimension.
    int mNumCtasY;
    // The number of CTAs in Z dimension.
    int mNumCtasZ;
    // The cluster size in the X dimension.
    int mClusterDimX;
  };

 public:
  using KernelMeta = tensorrt_llm::kernels::TllmGenFmhaKernelMetaInfo;
  using RunnerParams = TllmGenFmhaRunnerParams;
  using SelectKernelParams = TllmGenSelectKernelParams;

  // Ctor.
  TllmGenFmhaKernel(KernelMeta const* pMetaStart, unsigned int nMetaCount, Data_type dtypeQ,
                    Data_type dtypeKv, Data_type dtypeOut, unsigned int smArch)
      : mDtypeQ(dtypeQ),
        mDtypeKv(dtypeKv),
        mDtypeOut(dtypeOut),
        mKernelMeta(pMetaStart),
        mKernelMetaCount(nMetaCount),
        mSM(smArch) {}

  void loadKernels() {
    for (unsigned int i = 0; i < mKernelMetaCount; ++i) {
      auto const& kernelMeta = mKernelMeta[i];
      IKL_LOG_DEBUG("Checking tllmgen attention kernel %s", kernelMeta.mFuncName);
      if (isSMCompatible(mSM, kernelMeta.mSM) && kernelMeta.mDataTypeQ == mDtypeQ &&
          kernelMeta.mDataTypeKv == mDtypeKv && kernelMeta.mDataTypeO == mDtypeOut) {
        // Store metadata for later use.
        IKL_LOG_DEBUG("Adding tllmgen attention kernel %s", kernelMeta.mFuncName);
        // Check for hash conflicts.
        uint64_t hash = hashID(kernelMeta);
        if (mKernelMetaMap.find(hash) != mKernelMetaMap.end()) {
          // The kernelMeta of the existing kernel.
          auto const& existingKernelMeta = mKernelMeta[mKernelMetaMap.at(hash)];
          // Allow conflicts only if they are family/specific versions of the same architecture.
          FLASHINFER_CHECK(isFamilySpecificSMPair(existingKernelMeta.mSM, kernelMeta.mSM),
                           "Hash conflicts exist between %s and %s.", existingKernelMeta.mFuncName,
                           kernelMeta.mFuncName);

          // Prefer specific SM version over family version (replace if existing is family).
          if (existingKernelMeta.mSM == kSM_100f) {
            mKernelMetaMap[hash] = i;
          }
        } else {
          mKernelMetaMap[hash] = i;
        }
      }
    }
  }

  size_t getNumLoadedKernels() const { return mKernelMetaMap.size(); }

  inline uint64_t hashID(int qkvLayout, int maskType, int kernelType, int scheduler,
                         int multiCtasKvMode, int headDimPerCtaV, int headDimQk, int headDimV,
                         int tileSizeQ, int tileSizeKv, int numTokensPerPage, bool reuseSmemKForV,
                         bool uses2CtaMma, bool sparseMla, bool skipsSoftmax) const {
    FLASHINFER_CHECK((headDimPerCtaV >= 32) && (headDimQk >= 32) && (headDimV >= 32) &&
                         (headDimPerCtaV <= 1024) && (headDimQk <= 1024) && (headDimV <= 1024),
                     "Expect (32 <= headDim <= 1024), got headDimPerCtaV=%d, headDimQk=%d, "
                     "headDimV=%d",
                     headDimPerCtaV, headDimQk, headDimV);
    // The numTokensPerPage must be power of 2.
    FLASHINFER_CHECK((numTokensPerPage & (numTokensPerPage - 1)) == 0,
                     "The numTokensPerPage must be power of 2.");
    FLASHINFER_CHECK(tileSizeQ <= 128 && tileSizeKv <= 128,
                     "The tileSizeQ and tileSizeKv must be <= 128.");
    FLASHINFER_CHECK((tileSizeQ & (tileSizeQ - 1)) == 0 && (tileSizeKv & (tileSizeKv - 1)) == 0,
                     "The tileSizeQ and tileSizeKv must be power of 2.");
    FLASHINFER_CHECK(tileSizeKv == 64 || tileSizeKv == 128, "The tileSizeKv must be 64 or 128.");
    // Format of the hash key:
    // Bit 0  - 3 : qkvLayout.
    // Bit 4  - 7 : maskType.
    // Bit 8  - 11: kernelType.
    // Bit 12 - 15: tileScheduler.
    // Bit 16 - 17: multiCtasKvMode.
    // Bit 18 - 25: (headDimPerCtaV >> 3).
    // Bit 26 - 33: (headDimQk >> 3).
    // Bit 34 - 41: (headDimV >> 3).
    // Bit 42 - 43: (tileSizeKv >> 6).
    // Bit 44 - 48: (log2(numTokensPerPage)).
    // Bit 49 - 52: (log2(tileSizeQ)).
    // Bit 53 - 53: reuseSmemKForV.
    // Bit 54 - 54: uses2CtaMma.
    // Bit 55 - 55: sparseMla.
    // Bit 56 - 56: skipsSoftmax.
    return (static_cast<uint64_t>(qkvLayout) << 0) | (static_cast<uint64_t>(maskType) << 4) |
           (static_cast<uint64_t>(kernelType) << 8) | (static_cast<uint64_t>(scheduler) << 12) |
           (static_cast<uint64_t>(multiCtasKvMode) << 16) |
           (static_cast<uint64_t>(headDimPerCtaV >> 3) << 18) |
           (static_cast<uint64_t>(headDimQk >> 3) << 26) |
           (static_cast<uint64_t>(headDimV >> 3) << 34) |
           (static_cast<uint64_t>(tileSizeKv >> 6) << 42) |
           (static_cast<uint64_t>(log2(numTokensPerPage)) << 44) |
           (static_cast<uint64_t>(log2(tileSizeQ)) << 49) |
           (static_cast<uint64_t>(reuseSmemKForV) << 53) |
           (static_cast<uint64_t>(uses2CtaMma) << 54) | (static_cast<uint64_t>(sparseMla) << 55) |
           (static_cast<uint64_t>(skipsSoftmax) << 56);
  }

  uint64_t hashID(KernelMeta const& kernelMeta) const {
    return hashID(kernelMeta.mQkvLayout, kernelMeta.mMaskType, kernelMeta.mKernelType,
                  kernelMeta.mTileScheduler, kernelMeta.mMultiCtasKvMode,
                  kernelMeta.mHeadDimPerCtaV, kernelMeta.mHeadDimQk, kernelMeta.mHeadDimV,
                  kernelMeta.mTileSizeQ, kernelMeta.mTileSizeKv, kernelMeta.mNumTokensPerPage,
                  kernelMeta.mReuseSmemKForV, kernelMeta.m2CtaMma, kernelMeta.mSparseMla,
                  kernelMeta.mSkipsSoftmaxWhenPossible);
  }

  std::pair<bool, std::string> checkIfKernelExist(RunnerParams const& params) const {
    // The selectKernelParams that might be updated.
    SelectKernelParams selectKernelParams{params};
    // Select the kernel.
    selectKernel(params, selectKernelParams);
    // Hash the runner params.
    auto [hashId, info] = hashFromRunnerParams(params, selectKernelParams);
    return std::make_pair(mKernelMetaMap.find(hashId) != mKernelMetaMap.end(), info);
  }

  // start here
  void run(RunnerParams const& params) const {
    // The selectKernelParams that might be updated.
    SelectKernelParams selectKernelParams{params};
    // The parameters for launching the kernel.
    CtaLaunchParams ctaLaunchParams;
    // The iteration index (used to detect a deadlock of selecting new kernels).
    int selectKernelIter = 0;
    // While loop.
    while (true) {
      // Any value >= 2 should work here, but we set it larger in case that we
      // might have more complicated heuristic in the future.
      FLASHINFER_CHECK(selectKernelIter < 8,
                       "A deadlock is detected when selecting trtllm-gen kernels.");

      // Select the kernel.
      selectKernel(params, selectKernelParams);
      // Load the kernel.
      auto [func, kernelMeta] = loadKernel(params, selectKernelParams);

      // Compute the number of CTAs in X, Y and Z dimension and the cluster size in the X dimension.
      computeCtaAndClusterConfig(ctaLaunchParams, params, kernelMeta, selectKernelParams);

      // Need to select a new kernel if mSelectNewKernel is true.
      if (selectKernelParams.mSelectNewKernel) {
        selectKernelIter++;
        continue;
      }

      // Prepare the kernel parameters.
      auto kernelParams = KernelParams::setKernelParams(
          params, kernelMeta, ctaLaunchParams.mMaxNumCtasQ, ctaLaunchParams.mMaxNumCtasKv);

      // Prepare kernel parameters list for cuLaunchKernelEx.
      void* kernelParamsList[] = {&kernelParams};
      CUlaunchConfig launch_config;
      launch_config.blockDimX = kernelMeta.mThreadsPerCTA;
      launch_config.blockDimY = 1;
      launch_config.blockDimZ = 1;
      launch_config.gridDimX = ctaLaunchParams.mNumCtasX;
      launch_config.gridDimY = ctaLaunchParams.mNumCtasY;
      launch_config.gridDimZ = ctaLaunchParams.mNumCtasZ;
      launch_config.hStream = params.stream;
      launch_config.sharedMemBytes = kernelMeta.mSharedMemBytes;

      // Debug info.
      IKL_LOG_DEBUG("TRTLLM-Gen launch info (in TllmGenFmhaKernel %s, %s, %s, %d): kernelName = %s",
                    toStr(mDtypeQ), toStr(mDtypeKv), toStr(mDtypeOut), mSM, kernelMeta.mFuncName);
      IKL_LOG_DEBUG(
          "TRTLLM-Gen launch info: maxSeqLenQ = %d, "
          "maxSeqLenKv = %d, "
          "numHeadsQ = %d, "
          "numHeadsKv = %d, batchSize = %d, kernelType = %d",
          params.mMaxSeqLenQ, params.mMaxSeqLenKv, params.mNumHeadsQ, params.mNumHeadsKv,
          params.mBatchSize, static_cast<int>(params.mKernelType));
      IKL_LOG_DEBUG(
          "TRTLLM-Gen launch info: numCtasX = %d, numCtasY = %d, numCtasZ = %d, clusterDimX = %d",
          ctaLaunchParams.mNumCtasX, ctaLaunchParams.mNumCtasY, ctaLaunchParams.mNumCtasZ,
          ctaLaunchParams.mClusterDimX);

      CUlaunchAttribute launch_attribute[3];
      launch_attribute[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      launch_attribute[0].value.clusterDim.x = ctaLaunchParams.mClusterDimX;
      launch_attribute[0].value.clusterDim.y = 1;
      launch_attribute[0].value.clusterDim.z = 1;
      launch_attribute[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      launch_attribute[1].value.clusterSchedulingPolicyPreference =
          ctaLaunchParams.mClusterDimX > 1 ? CU_CLUSTER_SCHEDULING_POLICY_SPREAD
                                           : CU_CLUSTER_SCHEDULING_POLICY_DEFAULT;
      launch_attribute[2].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      launch_attribute[2].value.programmaticStreamSerializationAllowed = params.enable_pdl;

      launch_config.attrs = launch_attribute;
      launch_config.numAttrs = 3;
      // Add setting for non-portable cluster size.
      if (ctaLaunchParams.mClusterDimX > 8) {
        cuErrCheck(cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                                      1  // Enable non-portable cluster sizes
                                      ));
      }

      // Force using GmemReduction for the multiCtasKvMode if the CgaSmemReduction needs more than
      // one wave (due to the cluster occupancy limit).
      // TODO: find a better heuristic of using CgaSmemReduction.
      if (isCgaSmemReduction(selectKernelParams.mMultiCtasKvMode)) {
        // The maximum number of active clusters that could co-exist.
        int maxActiveClusters = 1;
        cuErrCheck(cuOccupancyMaxActiveClusters(&maxActiveClusters, func, &launch_config));
        // Use the GmemReduction instead if it needs more than one wave.
        if (maxActiveClusters * ctaLaunchParams.mClusterDimX <
            (ctaLaunchParams.mNumCtasX * ctaLaunchParams.mNumCtasY * ctaLaunchParams.mNumCtasZ)) {
          selectKernelParams.mForceGmemReduction = true;
          selectKernelParams.mMultiCtasKvMode = MultiCtasKvMode::GmemReduction;
          // continue to select a new kernel.
          continue;
        }
      }
      cuErrCheck(cuLaunchKernelEx(&launch_config, func, kernelParamsList, nullptr));

      // Run the separate reduction kernel if needed.
      tensorrt_llm::kernels::runFmhaReduction(kernelMeta, kernelParams, params.mMultiProcessorCount,
                                              params.enable_pdl, params.stream);

      if (params.lsePtr != nullptr) {
        flashinfer::ComputeLSEFromMD(params.softmaxStatsPtr, params.lsePtr,
                                     params.mSumOfSeqLensQ * params.mNumHeadsQ, params.enable_pdl,
                                     params.stream);
      }
      // Break the while op.
      break;
    }
  }

 private:
  // Is it MLA generation kernel ?
  inline bool isMlaGenKernel(RunnerParams const& params) const {
    return params.mHeadDimQk == 576 && params.mHeadDimV == 512;
  }

  // Compute the number of CTAs in X, Y and Z dimension and the cluster size in the X dimension.
  void computeCtaAndClusterConfig(CtaLaunchParams& ctaLaunchParams, RunnerParams const& params,
                                  KernelMeta const& kernelMeta,
                                  SelectKernelParams& selectKernelParams) const {
    bool isDsv3MinLatencyMode = params.mBatchSize == 1 && params.mMaxSeqLenQ >= 1 &&
                                params.mMaxSeqLenQ <= 16 && params.mHeadDimQk == 576 &&
                                params.mHeadDimV == 512;
    // Do we need to select a new kernel ?
    selectKernelParams.mSelectNewKernel = false;

    // The number of Ctas per Q sequence.
    int numCtasPerSeqQ = (params.mMaxSeqLenQ + kernelMeta.mStepQ - 1) / kernelMeta.mStepQ;
    // The generation-phase kernels might need to group both tokensQ and headsQ into one CTA.
    if (params.mMaxSeqLenQ > 1 && !isContextKernel(params.mKernelType)) {
      // Each CTA handles one tokenQ by default for spec-decoding generation kernel.
      if (!kernelMeta.mGroupsTokensHeadsQ) {
        numCtasPerSeqQ = params.mMaxSeqLenQ;
      } else {
        // Compute numTokensPerCtaQ where each CTA must process complete numGroupedHeadsQ.
        // Note that each CTA must process complete numHeadsQPerKv.
        int numTokensPerCtaQ = kernelMeta.mStepQ / params.mNumHeadsQPerKv;
        // Group both headsQ and tokensQ into one CTA.
        numCtasPerSeqQ = flashinfer::ceil_div(params.mMaxSeqLenQ, numTokensPerCtaQ);
      }
    }

    // Compute the grid dimension Y.
    int numHeadsPerCta =
        kernelMeta.mGroupsHeadsQ ? std::min(params.mNumHeadsQPerKv, kernelMeta.mStepQ) : 1;
    int numCtasForAllHeadsQ = params.mNumHeadsQ / numHeadsPerCta;
    FLASHINFER_CHECK(numHeadsPerCta * numCtasForAllHeadsQ == params.mNumHeadsQ,
                     "The numHeadsQ/numHeadsKv is not supported.");
    // Take the number of headDim CTAs.
    FLASHINFER_CHECK(kernelMeta.mHeadDimV % selectKernelParams.mHeadDimPerCtaV == 0,
                     "The headDimPerCtaV is not supported.");
    int numCtasPerHeadDim = kernelMeta.mHeadDimV / selectKernelParams.mHeadDimPerCtaV;
    // Compute the current numCtasX.
    int numCtasX = numCtasPerSeqQ;
    // Update the numCtasY.
    int numCtasY = numCtasForAllHeadsQ * numCtasPerHeadDim;
    // Compute the grid dimension Z.
    int numCtasZ = params.mBatchSize;
    // The 2CtaMma kernels will use 2 Ctas in the x dimension (only used by MLA generation kernels)
    // for heads, so numCtasPerHeadDim and numCtasForAllHeadsQ will be handled by the 2Ctas in the x
    // dimension.
    if (isMlaGenKernel(params) && selectKernelParams.mUses2CtaMma) {
      FLASHINFER_CHECK(numCtasForAllHeadsQ == 2 && numCtasPerHeadDim == 2,
                       "Internal error: numCtasPerHeadDim should be 2.");
      numCtasX *= 2;
      numCtasY /= (numCtasForAllHeadsQ * numCtasPerHeadDim);
    }

    // First split the seqLenKv into multiple CTAs if the utilization is not full.
    // The number of Ctas per KV sequence.
    int numCtasPerSeqKv = 1;
    // Consider the multiCtasKvMode for better GPU utilization.
    if (isMultiCtasKvEnabled(selectKernelParams.mMultiCtasKvMode)) {
      // The maximum attention window (the maximum number of tokensKv that will be attended to).
      int maxAttentionWindow{params.mMaxSeqLenKv};
      // The sparseMla only selects topK tokensKv.
      if (params.mSparseMla) {
        maxAttentionWindow = std::min(params.mMaxSeqLenKv, params.mSparseMlaTopK);
      }
      // Some of the tilesKv will be skipped if the sliding window attention or chunked attention is
      // used.
      if (isSlidingOrChunkedCausalMask(selectKernelParams.mMaskType)) {
        if (params.mMaxSeqLenKv > params.mAttentionWindowSize) {
          // Consider that the first tileKv might contain tokensKv that is out of the attention
          // window.
          maxAttentionWindow =
              std::min(params.mMaxSeqLenKv, params.mAttentionWindowSize + kernelMeta.mStepKv - 1);
        } else {
          maxAttentionWindow = std::min(params.mMaxSeqLenKv, params.mChunkedAttentionSize);
        }
      }

      // The maximum number Ctas per Kv sequence, which makes sure that each CtaKv has work to do.
      // The factor of 2 is applied here to ensure the reduction overhead does not outweigh the
      // benefits of a shorter mainloop.
      int const maxNumCtasPerSeqKv =
          (maxAttentionWindow + 2 * kernelMeta.mStepKv - 1) / (2 * kernelMeta.mStepKv);
      // Compute numCtasPerSeqKv.
      numCtasPerSeqKv = std::min(
          maxNumCtasPerSeqKv,
          std::max(1, int32_t(params.mMultiProcessorCount / (numCtasX * numCtasY * numCtasZ))));
      // Update the numCtasX.
      numCtasX *= numCtasPerSeqKv;
      // The current total number of CTAs.
      int totalNumCtas = numCtasX * numCtasZ * numCtasY;
      // Disable the multiCtasKvMode if there is only one CtaKv.
      if (numCtasPerSeqKv <= 1) {
        selectKernelParams.mMultiCtasKvMode = MultiCtasKvMode::Disabled;
        // Enable the persistent scheduler for better performance.
        selectKernelParams.mTileScheduler = TileScheduler::Persistent;
        // Need to select a different kernel.
        selectKernelParams.mSelectNewKernel = true;
      } else if (totalNumCtas < params.mMultiProcessorCount && isMlaGenKernel(params) &&
                 !params.mSparseMla && selectKernelParams.mTileSizeKv == 128 &&
                 getEnvUseTileSizeKv64ForTrtllmGen()) {
        // Use smaller tileSizeKv to fully utilize the SMs.
        selectKernelParams.mTileSizeKv = 64;
        // Need to select a different kernel.
        selectKernelParams.mSelectNewKernel = true;
      }

      // Enable the CgaSmemReduction if the numCtasPerSeqKv <= 16 as the maximum cluster dimension
      // is 16. Only the swapsMmaAbForGeneration kernel supports the CgaSmemReduction for now.
      if (!isDsv3MinLatencyMode && numCtasPerSeqKv > 1 && numCtasPerSeqKv <= 16 &&
          isSwapsMmaAbForGenerationKernel(selectKernelParams.mKernelType) &&
          isGmemReduction(selectKernelParams.mMultiCtasKvMode) &&
          !selectKernelParams.mForceGmemReduction) {
        selectKernelParams.mMultiCtasKvMode = MultiCtasKvMode::CgaSmemReduction;
        // Need to select a different kernel.
        selectKernelParams.mSelectNewKernel = true;
      }

      // Add the debug info when multiCtasKvMode is enabled.
      if (numCtasPerSeqKv > 1) {
        IKL_LOG_DEBUG(
            "TRTLLM-Gen launch info: multiCtasKvMode is enabled with tileSizeKv = %d, "
            "numCtasPerSeqKv = %d, "
            "numCtasPerSeqQ = "
            "%d, numCtasY = %d, numCtasZ = %d",
            selectKernelParams.mTileSizeKv, numCtasPerSeqKv, numCtasPerSeqQ, numCtasY, numCtasZ);
      }
    }

    // The cluster size in the X dimension.
    int clusterDimX = selectKernelParams.mUses2CtaMma ? 2 : 1;
    if (isCgaSmemReduction(selectKernelParams.mMultiCtasKvMode)) {
      // Note 2CtaMma and CgaSmemReduction cannot be used together currently.
      clusterDimX *= numCtasPerSeqKv;
    }

    // Compute the current number of CTAs in total.
    int totalNumCtas = numCtasX * numCtasZ * numCtasY;

    // Then split the headDimV into multiple CTAs if there are still unused SMs.
    if (isMlaGenKernel(params) && !selectKernelParams.mReuseSmemKForV &&
        !selectKernelParams.mSelectNewKernel && !selectKernelParams.mUses2CtaMma) {
      // Split the headDimV into multiple CTAs if the utilization is not full.
      // It doesn't work with reuseSmemKForV currently.
      // TODO: find better heuristic of splitting headDimV across multiple CTAs.

      int corrFactor = isDsv3MinLatencyMode ? 1 : 2;
      if (selectKernelParams.mHeadDimPerCtaV == 512 &&
          totalNumCtas * corrFactor <= params.mMultiProcessorCount) {
        // Use smaller headDimPerCtaV to fully utilize the SMs.
        selectKernelParams.mHeadDimPerCtaV =
            totalNumCtas * 2 * corrFactor <= params.mMultiProcessorCount ? 128 : 256;
        // Need to select a different kernel.
        selectKernelParams.mSelectNewKernel = true;
      }
    }

    // Update the parameters for launching the kernel.
    ctaLaunchParams.mMaxNumCtasQ = numCtasPerSeqQ;
    ctaLaunchParams.mMaxNumCtasKv = numCtasPerSeqKv;
    ctaLaunchParams.mNumCtasX = numCtasX;
    ctaLaunchParams.mNumCtasY = numCtasY;
    ctaLaunchParams.mNumCtasZ = numCtasZ;
    ctaLaunchParams.mClusterDimX = clusterDimX;
  }

  // Determine if we should use the SwapsMmaAbForGeneration kernel for MLA generation.
  bool useSwapsMmaAbMlaGenKernel(RunnerParams const& params) const {
    // Use the SwapsMmaAbForGeneration kernel for MLA generation when the following conditions are
    // met:
    // 1. The seqLenPerCtaKv <= 1024 based on the benchmark results (this might be fine-tuned
    // later).
    // 2. The numCtas (after splitting the heads across multiple CTAs) <=
    // params.mMultiProcessorCount.

    // The maximum number Ctas per Kv sequence, which makes sure that each CtaKv has work to do.
    // Here we assume the stepKv is 256.
    int const maxNumCtasPerSeqKv = flashinfer::ceil_div(params.mMaxSeqLenKv, 256);
    ;
    // The number of Ctas.
    int const numCtas = static_cast<int32_t>(params.mBatchSize * params.mMaxSeqLenQ *
                                             flashinfer::ceil_div(params.mNumHeadsQPerKv, 16));
    // Compute numCtasPerSeqKv.
    int const numCtasPerSeqKv =
        std::min(maxNumCtasPerSeqKv, std::max(1, int32_t(params.mMultiProcessorCount / numCtas)));
    // Compute the seqLenPerCtaKv.
    int const seqLenPerCtaKv = flashinfer::ceil_div(params.mMaxSeqLenKv, numCtasPerSeqKv);
    // Whether we should use the SwapsMmaAbForGeneration kernel for MLA generation.
    return seqLenPerCtaKv <= 1024 && numCtas <= params.mMultiProcessorCount;
  }

  // Select the MLA generation kernel.
  void selectMlaGenerationKernel(RunnerParams const& params,
                                 SelectKernelParams& selectKernelParams) const {
    // We use the low-latency kernel (SwapsMmaAbForGeneration with tileSizeQ = 16) when any of the
    // following conditions are met:
    // 1. The number of headsQPerKv is <= 32.
    // 2. The number of headsQPerKv is < 128 for sparseMla.
    // 3. The seqLenPerCtaKv <= 1024 based on the benchmark results (this might be fine-tuned later)
    // and
    //    the numCtas (after splitting the heads across multiple CTAs) <=
    //    params.mMultiProcessorCount.
    // The sparseMla kernel will always use the 2CTA high-throughput kernel.

    // The kernel type.
    FmhaKernelType& kernelType = selectKernelParams.mKernelType;
    // The tile size for Q.
    int& tileSizeQ = selectKernelParams.mTileSizeQ;

    // Check the conditions.
    if (params.mNumHeadsQPerKv <= 32 || (params.mSparseMla && params.mNumHeadsQPerKv < 128) ||
        useSwapsMmaAbMlaGenKernel(params)) {
      kernelType = FmhaKernelType::SwapsMmaAbForGeneration;
      // Currently, only tileSizeQ = 8 or 16 are supported.
      tileSizeQ = params.mNumHeadsQPerKv <= 8 ? 8 : 16;
    } else {
      // Otherwise, we use the high-throughput kernel.
      kernelType = FmhaKernelType::KeepsMmaAbForGeneration;
      // Use the tileSizeQ = 64 for MLA high-throughput generation kernels.
      tileSizeQ = 64;
      // Always use the separate reduction kernel.
      if (isMultiCtasKvEnabled(selectKernelParams.mMultiCtasKvMode)) {
        selectKernelParams.mMultiCtasKvMode = MultiCtasKvMode::GmemReductionWithSeparateKernel;
      }
      // The keepsMmaAbForGeneration sparseMla kernels only support numHeadsQPerKv = 128.
      FLASHINFER_CHECK(
          !params.mSparseMla || params.mNumHeadsQPerKv == 128,
          "The keepsMmaAbForGeneration sparseMla kernels only support numHeadsQPerKv = 128, got %d",
          params.mNumHeadsQPerKv);
      // The 2CTA keepsMmaAbForGeneration kernel is used when the numHeadsQPerKv is 128.
      if (params.mNumHeadsQPerKv == 128) {
        selectKernelParams.mUses2CtaMma = true;
        // Each Cta only handles 256 headDimV.
        selectKernelParams.mHeadDimPerCtaV = 256;
      }
    }
  }

  // Selects a heuristic tileSizeQ if groupsTokensHeadsQ is true.
  void selectTileSizeQForGqaGeneration(RunnerParams const& params,
                                       SelectKernelParams& selectKernelParams) const {
    // Define the per-tile mainloop cost model for different tileSizeQ choices.
    std::unordered_map<int, float> kernelMainloopCost = {
        {128, 2.2},  // Cost factor when tileSizeQ = 128
        {64, 1.68},  // Cost factor when tileSizeQ = 64
        {32, 1.48},  // Cost factor when tileSizeQ = 32
        {16, 1.2},   // Cost factor when tileSizeQ = 16
        {8, 1.0}     // Cost factor when tileSizeQ = 8
    };

    // Define the per-tile reduction cost model for different tileSizeQ choices.
    std::unordered_map<int, float> kernelReductionCost = {
        {128, 1.32},  // Reduction cost factor when tileSizeQ = 128
        {64, 1.2},    // Reduction cost factor when tileSizeQ = 64
        {32, 1.08},   // Reduction cost factor when tileSizeQ = 32
        {16, 1.03},   // Reduction cost factor when tileSizeQ = 16
        {8, 1.0}      // Reduction cost factor when tileSizeQ = 8
    };

    // The reduction cost emulated as a sequence length factor.
    float const kernelReductionSeqLenFactor = 128.0f;

    // The parameters for launching the kernel.
    CtaLaunchParams ctaLaunchParams;
    // The copy of the selectKernelParams, which makes sure it won't modify the original
    // selectKernelParams when computing the number of CTAs.
    SelectKernelParams selectKernelParamsCopy = selectKernelParams;
    // Load the kernel.
    auto [func, kernelMeta] = loadKernel(params, selectKernelParamsCopy);
    // Compute numCtasX, numCtasY and numCtasZ.
    computeCtaAndClusterConfig(ctaLaunchParams, params, kernelMeta, selectKernelParamsCopy);

    // If there are no free SMs or tileSizeQ is already the smallest one, skip the heuristic
    // selection.
    if (ctaLaunchParams.mNumCtasX * ctaLaunchParams.mNumCtasY * ctaLaunchParams.mNumCtasZ * 2 >
            params.mMultiProcessorCount ||
        selectKernelParamsCopy.mTileSizeQ <= 8) {
      // No need to select the kernel further.
      return;
    }

    // Candidate tile sizes for tileSizeQ to explore.
    int const candidateTileSizesQ[] = {128, 64, 32, 16, 8};

    // The default tileSizeQ.
    int defaultTileSizeQ = selectKernelParamsCopy.mTileSizeQ;
    // The selected tileSizeQ.
    int selectedTileSizeQ = selectKernelParamsCopy.mTileSizeQ;

    // The minimum modeling kernel time.
    float globalModelingKernelTime = FLT_MAX;
    // Loop over each candidate tile size.
    for (int tileSizeQ : candidateTileSizesQ) {
      // Only consider candidates <= default tileSizeQ and ensure each CTA processes full
      // numHeadsQPerKv.
      if (tileSizeQ > defaultTileSizeQ || tileSizeQ < params.mNumHeadsQPerKv) {
        continue;
      }

      selectKernelParamsCopy.mTileSizeQ = tileSizeQ;
      if (tileSizeQ >= 64) {
        selectKernelParamsCopy.mKernelType = FmhaKernelType::KeepsMmaAbForGeneration;
      } else {
        selectKernelParamsCopy.mKernelType = FmhaKernelType::SwapsMmaAbForGeneration;
      }

      // Load the kernel.
      std::tie(func, kernelMeta) = loadKernel(params, selectKernelParamsCopy);

      // Compute the number of CTAs.
      computeCtaAndClusterConfig(ctaLaunchParams, params, kernelMeta, selectKernelParamsCopy);

      // Compute the seqLenPerCtaKv.
      int32_t seqLenPerCtaKv =
          flashinfer::ceil_div(flashinfer::ceil_div(params.mMaxSeqLenKv, kernelMeta.mStepKv),
                               ctaLaunchParams.mMaxNumCtasKv) *
          kernelMeta.mStepKv;

      // Compute the modeling kernel time = mainloop cost + reduction cost.
      float modelingKernelTime = kernelMainloopCost.at(tileSizeQ) * seqLenPerCtaKv +
                                 kernelReductionCost.at(tileSizeQ) * kernelReductionSeqLenFactor *
                                     ctaLaunchParams.mMaxNumCtasKv;

      // Compute the total number of CTAs.
      int32_t numCtas =
          ctaLaunchParams.mNumCtasX * ctaLaunchParams.mNumCtasY * ctaLaunchParams.mNumCtasZ;
      // Compute the number of waves.
      int32_t numWaves = flashinfer::ceil_div(numCtas, params.mMultiProcessorCount);
      // Compute the total modeling kernel time.
      modelingKernelTime *= numWaves;

      // If this candidate has a lower time than the global minimum, update the global minimum.
      if (modelingKernelTime < globalModelingKernelTime) {
        globalModelingKernelTime = modelingKernelTime;
        selectedTileSizeQ = tileSizeQ;
      }
    }

    // Update the tileSizeQ.
    selectKernelParams.mTileSizeQ = selectedTileSizeQ;
    // Update the kernel type.
    if (selectKernelParams.mTileSizeQ >= 64) {
      selectKernelParams.mKernelType = FmhaKernelType::KeepsMmaAbForGeneration;
    } else {
      selectKernelParams.mKernelType = FmhaKernelType::SwapsMmaAbForGeneration;
    }
  }

  // Selects a heuristic kernel for GQA generation.
  void selectGqGenerationKernel(RunnerParams const& params,
                                SelectKernelParams& selectKernelParams) const {
    // The kernel type.
    FmhaKernelType& kernelType = selectKernelParams.mKernelType;
    // The tile size for Q.
    int& tileSizeQ = selectKernelParams.mTileSizeQ;

    // Mixed precision kernels don't work with groupsTokensHeadsQ = true for now.
    if (mDtypeQ != mDtypeKv || mDtypeOut == DATA_TYPE_E2M1) {
      tileSizeQ = params.mNumHeadsQPerKv <= 8 ? 8 : 16;
      kernelType = FmhaKernelType::SwapsMmaAbForGeneration;
      return;
    }

    // The number of tokensQ and headsQ that can be grouped into one CTA.
    int numTokensHeadsQ = params.mNumHeadsQPerKv * params.mMaxSeqLenQ;
    // When numHeadsQPerKv >= 64, use KeepsMmaAbForGeneration kernel.
    if (numTokensHeadsQ <= 8) {
      tileSizeQ = 8;
      kernelType = FmhaKernelType::SwapsMmaAbForGeneration;
    } else if (numTokensHeadsQ <= 16) {
      tileSizeQ = 16;
      kernelType = FmhaKernelType::SwapsMmaAbForGeneration;
    } else if (numTokensHeadsQ <= 32) {
      tileSizeQ = 32;
      kernelType = FmhaKernelType::SwapsMmaAbForGeneration;
    } else if (numTokensHeadsQ <= 64) {
      tileSizeQ = 64;
      kernelType = FmhaKernelType::KeepsMmaAbForGeneration;
    } else {
      tileSizeQ = 128;
      kernelType = FmhaKernelType::KeepsMmaAbForGeneration;
    }

    // When maxSeqLenQ > 1, use an experimental kernel-timing model to select the best kernel that
    // groups both tokensQ and headsQ into one CTA.
    if (params.mMaxSeqLenQ > 1) {
      selectTileSizeQForGqaGeneration(params, selectKernelParams);
    }
  }

  // Select a kernel based on the heuristic.
  void selectKernel(RunnerParams const& params, SelectKernelParams& selectKernelParams) const {
    // Select the kernel based on the kernel type.
    if (isGenerationKernel(params.mKernelType) && isMlaGenKernel(params)) {
      selectMlaGenerationKernel(params, selectKernelParams);
    } else if (isGenerationKernel(params.mKernelType)) {
      selectGqGenerationKernel(params, selectKernelParams);
    }

    // Enable sliding window or chunked causal if the max kv sequence length exceeds attention
    // window size or chunked attention size. This is supported by causal-mask context kernels and
    // generation-phase kernels.
    if ((selectKernelParams.mMaskType == TrtllmGenAttentionMaskType::Causal ||
         !isContextKernel(params.mKernelType)) &&
        (params.mMaxSeqLenKv > params.mAttentionWindowSize ||
         params.mChunkedAttentionSize != INT_MAX)) {
      FLASHINFER_CHECK(
          params.mMaxSeqLenKv <= params.mAttentionWindowSize ||
              params.mMaxSeqLenKv <= params.mChunkedAttentionSize,
          "Sliding window attention and chunked attention should not be used together");
      selectKernelParams.mMaskType = TrtllmGenAttentionMaskType::SlidingOrChunkedCausal;
    }

    // SparseMla kernels use a fixed numTokensPerPage = 1.
    if (params.mSparseMla) {
      selectKernelParams.mNumTokensPerPage = 1;
    } else if (!isPagedKv(params.mQkvLayout)) {
      // NumTokensPerPage is set to 0 when not selecting pagedKv-layout kernels.
      selectKernelParams.mNumTokensPerPage = 0;
    }
  }

  std::pair<uint64_t, std::string> hashFromRunnerParams(
      RunnerParams const& params, SelectKernelParams& selectKernelParams) const {
    // Debug info.
    std::string info =
        "qkvLayout=" + std::to_string(static_cast<int>(params.mQkvLayout)) +
        ", maskType=" + std::to_string(static_cast<int>(selectKernelParams.mMaskType)) +
        ", kernelType=" + std::to_string(static_cast<int>(selectKernelParams.mKernelType)) +
        ", tileScheduler=" + std::to_string(static_cast<int>(selectKernelParams.mTileScheduler)) +
        ", multiCtasKvMode=" +
        std::to_string(static_cast<int>(selectKernelParams.mMultiCtasKvMode)) +
        ", headDimPerCtaV=" + std::to_string(selectKernelParams.mHeadDimPerCtaV) +
        ", headDimQk=" + std::to_string(params.mHeadDimQk) +
        ", headDimV=" + std::to_string(params.mHeadDimV) +
        ", tileSizeQ=" + std::to_string(selectKernelParams.mTileSizeQ) +
        ", tileSizeKv=" + std::to_string(selectKernelParams.mTileSizeKv) +
        ", numTokensPerPage=" + std::to_string(selectKernelParams.mNumTokensPerPage) +
        ", reuseSmemKForV=" + std::to_string(selectKernelParams.mReuseSmemKForV) +
        ", uses2CtaMma=" + std::to_string(selectKernelParams.mUses2CtaMma) +
        ", sparseMla=" + std::to_string(params.mSparseMla) +
        ", skipsSoftmax=" + std::to_string(selectKernelParams.mSkipsSoftmaxWhenPossible);
    IKL_LOG_DEBUG(
        "Searching for kernel traits (%d available) in TllmGenFmhaKernel(%s, %s, %s, %d) %s",
        getNumLoadedKernels(), toStr(mDtypeQ), toStr(mDtypeKv), toStr(mDtypeOut), mSM,
        info.c_str());

    return std::make_pair(
        hashID(static_cast<int>(params.mQkvLayout), static_cast<int>(selectKernelParams.mMaskType),
               static_cast<int>(selectKernelParams.mKernelType),
               static_cast<int>(selectKernelParams.mTileScheduler),
               static_cast<int>(selectKernelParams.mMultiCtasKvMode),
               selectKernelParams.mHeadDimPerCtaV, params.mHeadDimQk, params.mHeadDimV,
               selectKernelParams.mTileSizeQ, selectKernelParams.mTileSizeKv,
               selectKernelParams.mNumTokensPerPage, selectKernelParams.mReuseSmemKForV,
               selectKernelParams.mUses2CtaMma, params.mSparseMla,
               selectKernelParams.mSkipsSoftmaxWhenPossible),
        info);
  }

  // Load a single kernel (called by `run()` when needed).
  std::pair<CUfunction, KernelMeta> loadKernel(RunnerParams const& params,
                                               SelectKernelParams& selectKernelParams) const {
    // Hash the runner params.
    auto [hashId, info] = hashFromRunnerParams(params, selectKernelParams);
    auto const findMetaIter = mKernelMetaMap.find(hashId);
    // The meta index.
    auto const metaIndex = findMetaIter->second;

    // Add debug info when kernels are not found.
    FLASHINFER_CHECK(findMetaIter != mKernelMetaMap.end(), "Trtllm-gen kernels not found: " + info);

    // Load the function if not found.
    if (mFunctions.find(hashId) == mFunctions.end()) {
      // Load the kernel on-demand.
      auto const& kernelMeta = mKernelMeta[metaIndex];
      CUmodule hmod{0};
      std::string kernelName(kernelMeta.mFuncName);

      // Check if the module is already loaded.
      auto findModuleIter = mModules.find(kernelMeta.mFuncName);
      auto capitalizeFirst = [](std::string str) {
        if (!str.empty()) {
          str[0] = std::toupper(str[0]);
        }
        return str;
      };
      if (findModuleIter == mModules.end()) {
        // Load the module.
        std::string cubin_path = tllm_gen_fmha_cubin_path + "/" + kernelMeta.mFuncName + ".cubin";
        std::string cubin = getCubin(cubin_path, kernelMeta.sha256);
        if (cubin.empty()) {
          throw std::runtime_error("Failed to load cubin for " + kernelName);
        }
        cuErrCheck(cuModuleLoadData(&hmod, cubin.data()));
        mModules[kernelName] = hmod;
      } else {
        hmod = findModuleIter->second;
      }

      // Load the function.
      KernelInfo funcInfo;
      funcInfo.mMetaInfoIndex = metaIndex;
      cuErrCheck(cuModuleGetFunction(&funcInfo.mDeviceFunction, hmod, kernelMeta.mFuncName));

      if (kernelMeta.mSharedMemBytes >= 48 * 1024) {
        cuErrCheck(cuFuncSetAttribute(funcInfo.mDeviceFunction,
                                      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      kernelMeta.mSharedMemBytes));
      }

      // Cache the loaded function.
      mFunctions[hashId] = funcInfo;
    }

    // Retrieve the loaded kernel.
    auto const& kernelInfo = mFunctions.at(hashId);
    auto const& kernelMeta = mKernelMeta[kernelInfo.mMetaInfoIndex];
    CUfunction func = kernelInfo.mDeviceFunction;

    // Return the function and kernelMeta.
    return std::make_pair(func, kernelMeta);
  }

  Data_type mDtypeQ, mDtypeKv, mDtypeOut;
  KernelMeta const* mKernelMeta;
  unsigned int mKernelMetaCount;
  unsigned int mSM;
  mutable std::unordered_map<std::string, CUmodule> mModules;

  mutable std::unordered_map<uint64_t, unsigned int> mKernelMetaMap;

  struct KernelInfo {
    unsigned int mMetaInfoIndex;
    CUfunction mDeviceFunction;
  };

  mutable std::unordered_map<uint64_t, KernelInfo> mFunctions;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

class TllmFmhaKernelFactory {
 public:
  using KernelType = TllmGenFmhaKernel;

  KernelType const* getKernels(const typename KernelType::KernelMeta* pKernelList,
                               unsigned int nbKernels, Data_type dtypeQ, Data_type dtypeKv,
                               Data_type dtypeOut, unsigned int sm) {
    static std::mutex s_mutex;
    std::lock_guard<std::mutex> lg(s_mutex);

    auto const id = hashID(dtypeQ, dtypeKv, dtypeOut, sm);
    auto const findIter = mKernels.find(id);
    if (findIter == mKernels.end()) {
      KernelType* newKernel = new KernelType{pKernelList, nbKernels, dtypeQ, dtypeKv, dtypeOut, sm};
      newKernel->loadKernels();
      mKernels.insert(std::make_pair(id, std::unique_ptr<KernelType>(newKernel)));
      IKL_LOG_DEBUG(
          "Loading new kernel for dtypeQ=%s, dtypeKv=%s, dtypeOut=%s, sm=%d with %d loaded kernels",
          toStr(dtypeQ), toStr(dtypeKv), toStr(dtypeOut), sm, newKernel->getNumLoadedKernels());
      return newKernel;
    }
    return findIter->second.get();
  }

  static TllmFmhaKernelFactory& Get() {
    int deviceId;
    cudaGetDevice(&deviceId);
    static std::unique_ptr<TllmFmhaKernelFactory> sFactory[32] = {nullptr};
    if (sFactory[deviceId] == nullptr) {
      FLASHINFER_CHECK(deviceId < 32, "Invalid deviceId %d (max is 32 devices)", deviceId);
      sFactory[deviceId] = std::make_unique<TllmFmhaKernelFactory>(TllmFmhaKernelFactory());
    }

    return *(sFactory[deviceId]);
  }

 private:
  TllmFmhaKernelFactory() = default;

  inline uint64_t hashID(Data_type dtypeQ, Data_type dtypeKv, Data_type dtypeOut,
                         unsigned int sm) const {
    return static_cast<uint64_t>(sm) | static_cast<uint64_t>(dtypeQ) << 16 |
           static_cast<uint64_t>(dtypeKv) << 20 | static_cast<uint64_t>(dtypeOut) << 24;
  }

  std::unordered_map<uint64_t, const std::unique_ptr<KernelType>> mKernels;
};

inline TllmGenFmhaKernel const* getTllmFmhaKernels(Data_type dtypeQ, Data_type dtypeKv,
                                                   Data_type dtypeOut, unsigned int sm) {
#ifndef EXCLUDE_SM_100
  return TllmFmhaKernelFactory::Get().getKernels(
      tensorrt_llm::kernels::sTllmGenFmhaKernelMetaInfos,
      sizeof(tensorrt_llm::kernels::sTllmGenFmhaKernelMetaInfos) /
          sizeof(tensorrt_llm::kernels::sTllmGenFmhaKernelMetaInfos[0]),
      dtypeQ, dtypeKv, dtypeOut, sm);
#else
  return nullptr;
#endif  // EXCLUDE_SM_100
}
