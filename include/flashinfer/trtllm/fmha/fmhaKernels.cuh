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
#include <iterator>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "../../utils.cuh"
#include "../common.h"
#include "cubin/kernelMetaInfo.h"
#include "cuda_runtime_api.h"
#include "fmhaRunnerParams.h"
#include "kernelParams.h"

namespace flashinfer::trtllm_cubin_loader {
std::string getCubin(const std::string& kernelName, const std::string& sha256);
std::string getMetaInfo(const std::string& name, const std::string& sha256,
                        const std::string& extension);
}  // namespace flashinfer::trtllm_cubin_loader
using flashinfer::trtllm_cubin_loader::getCubin;
using flashinfer::trtllm_cubin_loader::getMetaInfo;

////////////////////////////////////////////////////////////////////////////////////////////////////
class TllmGenFmhaKernel {
 public:
  using KernelMeta = TllmGenFmhaKernelMetaInfo;
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
      if (kernelMeta.mSM == mSM && kernelMeta.mDataTypeQ == mDtypeQ &&
          kernelMeta.mDataTypeKv == mDtypeKv && kernelMeta.mDataTypeO == mDtypeOut) {
        // Store metadata for later use.
        mKernelMetaMap[hashID(kernelMeta)] = i;
      }
    }
  }

  size_t getNumLoadedKernels() const { return mKernelMetaMap.size(); }

  inline uint64_t hashID(int qkvLayout, int maskType, int kernelType, int scheduler,
                         int multiCtasKvMode, int headDimPerCtaV, int headDimQk, int headDimV,
                         int tileSizeKv, int numTokensPerPage, int maxNumHeadsQPerKvInCta,
                         bool reuseSmemKForV, bool uses2CtaMma) const {
    TORCH_CHECK((headDimPerCtaV >= 32) && (headDimQk >= 32) && (headDimV >= 32) &&
                    (headDimPerCtaV <= 2048) && (headDimQk <= 2048) && (headDimV <= 2048) &&
                    (numTokensPerPage <= 128),
                "Expect (32 <= headDim <= 2048) && (numTokensPerPage <= 128), "
                "got headDimPerCtaV=%d, headDimQk=%d, "
                "headDimV=%d, numTokensPerPage=%d",
                headDimPerCtaV, headDimQk, headDimV, numTokensPerPage);
    TORCH_CHECK(maxNumHeadsQPerKvInCta <= 128, "The maxNumHeadsQPerKvInCta <= 128 is required.");
    TORCH_CHECK(tileSizeKv == 64 || tileSizeKv == 128, "The tileSizeKv must be 64 or 128.");
    // Format of the hash key:
    // Bit 0  - 3 : qkvLayout.
    // Bit 4  - 7 : maskType.
    // Bit 8  - 11: kernelType.
    // Bit 12 - 15: tileScheduler.
    // Bit 16 - 17: multiCtasKvMode.
    // Bit 18 - 24: (headDimPerCtaV >> 5).
    // Bit 25 - 31: (headDimQk >> 5).
    // Bit 32 - 38: (headDimV >> 5).
    // Bit 39 - 40: (tileSizeKv >> 6).
    // Bit 41 - 48: numTokensPerPage.
    // Bit 49 - 56: maxNumHeadsQPerKvInCta.
    // Bit 57 - 57: reuseSmemKForV.
    // Bit 58 - 58: uses2CtaMma.
    return (static_cast<uint64_t>(qkvLayout) << 0) | (static_cast<uint64_t>(maskType) << 4) |
           (static_cast<uint64_t>(kernelType) << 8) | (static_cast<uint64_t>(scheduler) << 12) |
           (static_cast<uint64_t>(multiCtasKvMode) << 16) |
           (static_cast<uint64_t>(headDimPerCtaV >> 5) << 18) |
           (static_cast<uint64_t>(headDimQk >> 5) << 25) |
           (static_cast<uint64_t>(headDimV >> 5) << 32) |
           (static_cast<uint64_t>(tileSizeKv >> 6) << 39) |
           (static_cast<uint64_t>(numTokensPerPage) << 41) |
           (static_cast<uint64_t>(maxNumHeadsQPerKvInCta) << 49) |
           (static_cast<uint64_t>(reuseSmemKForV) << 57) |
           (static_cast<uint64_t>(uses2CtaMma) << 58);
  }

  uint64_t hashID(KernelMeta const& kernelMeta) const {
    return hashID(kernelMeta.mQkvLayout, kernelMeta.mMaskType, kernelMeta.mKernelType,
                  kernelMeta.mTileScheduler, kernelMeta.mMultiCtasKvMode,
                  kernelMeta.mHeadDimPerCtaV, kernelMeta.mHeadDimQk, kernelMeta.mHeadDimV,
                  kernelMeta.mTileSizeKv, kernelMeta.mNumTokensPerPage,
                  kernelMeta.mMaxNumHeadsQPerKvInCta, kernelMeta.mReuseSmemKForV,
                  kernelMeta.m2CtaMma);
  }

  std::pair<bool, std::string> checkIfKernelExist(RunnerParams const& params) const {
    // The selectKernelParams that might be updated.
    SelectKernelParams selectKernelParams{params};
    auto [hashId, info] = hashFromRunnerParams(params, selectKernelParams);
    return std::make_pair(mKernelMetaMap.find(hashId) != mKernelMetaMap.end(), info);
  }
  // start here
  void run(RunnerParams const& params) const {
    // The selectKernelParams that might be updated.
    SelectKernelParams selectKernelParams{params};
    // The iteration index (used to detect a deadlock of selecting new kernels).
    int selectKernelIter = 0;
    // While loop.
    while (true) {
      // Any value >= 2 should work here, but we set it larger in case that we
      // might have more complicated heuristic in the future.
      TORCH_CHECK(selectKernelIter < 8,
                  "A deadlock is detected when selecting trtllm-gen kernels.");
      auto [hashId, info] = hashFromRunnerParams(params, selectKernelParams);
      auto const findMetaIter = mKernelMetaMap.find(hashId);

      // Add debug info when kernels are not found.
      TORCH_CHECK(findMetaIter != mKernelMetaMap.end(), "Trtllm-gen kernels not found: " + info);

      //  auto const& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
      auto const findFuncIter = mFunctions.find(hashId);
      if (findFuncIter == mFunctions.end()) {
        // Load the kernel on-demand.
        loadKernel(hashId, findMetaIter->second);
      }
      // Retrieve the loaded kernel.
      auto const& kernelInfo = mFunctions.at(hashId);
      auto const& kernelMeta = mKernelMeta[kernelInfo.mMetaInfoIndex];
      CUfunction func = kernelInfo.mDeviceFunction;

      // Compute the number of CTAs in X, Y and Z dimension and the cluster size in the X dimension.
      auto [maxNumCtasQ, maxNumCtasKv, numCtasX, numCtasY, numCtasZ, clusterDimX] =
          computeCtaAndClusterConfig(params, kernelMeta, selectKernelParams);
      // Need to select a new kernel if mSelectNewKernel is true.
      if (selectKernelParams.mSelectNewKernel) {
        selectKernelIter++;
        continue;
      }

      // Prepare the kernel parameters.
      auto kernelParams =
          KernelParams::setKernelParams(params, kernelMeta, maxNumCtasQ, maxNumCtasKv);

      // Prepare kernel parameters list for cuLaunchKernelEx.
      void* kernelParamsList[] = {&kernelParams};
      CUlaunchConfig launch_config;
      launch_config.blockDimX = kernelMeta.mThreadsPerCTA;
      launch_config.blockDimY = 1;
      launch_config.blockDimZ = 1;
      launch_config.gridDimX = numCtasX;
      launch_config.gridDimY = numCtasY;
      launch_config.gridDimZ = numCtasZ;
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
          numCtasX, numCtasY, numCtasZ, clusterDimX);

      CUlaunchAttribute launch_attribute[3];
      launch_attribute[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      launch_attribute[0].value.clusterDim.x = clusterDimX;
      launch_attribute[0].value.clusterDim.y = 1;
      launch_attribute[0].value.clusterDim.z = 1;
      launch_attribute[1].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      launch_attribute[1].value.clusterSchedulingPolicyPreference =
          clusterDimX > 1 ? CU_CLUSTER_SCHEDULING_POLICY_SPREAD
                          : CU_CLUSTER_SCHEDULING_POLICY_DEFAULT;
      launch_attribute[2].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      launch_attribute[2].value.programmaticStreamSerializationAllowed = getEnvEnablePDL();

      launch_config.attrs = launch_attribute;
      launch_config.numAttrs = 3;
      // Add setting for non-portable cluster size.
      if (clusterDimX > 8) {
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
        if (maxActiveClusters * clusterDimX < (numCtasX * numCtasY * numCtasZ)) {
          selectKernelParams.mForceGmemReduction = true;
          selectKernelParams.mMultiCtasKvMode = MultiCtasKvMode::GmemReduction;
          // continue to select a new kernel.
          continue;
        }
      }
      cuErrCheck(cuLaunchKernelEx(&launch_config, func, kernelParamsList, nullptr));
      // Break the while op.
      break;
    }
  }

  static std::string getCubinPath() {
    const char* env_hash = std::getenv("FLASHINFER_CUBIN_ARTIFACTORY_HASH");
    std::string hash =
        env_hash ? std::string(env_hash) : "52e676342c67a3772e06f10b84600044c0c22b76";
    std::string cubin_path = hash + "/fmha/trtllm-gen/";
    return cubin_path;
  }

 private:
  // Is it MLA generation kernel ?
  inline bool isMlaGenKernel(RunnerParams const& params) const {
    return params.mHeadDimQk == 576 && params.mHeadDimV == 512;
  }

  // Compute the number of CTAs in X, Y and Z dimension and the cluster size in the X dimension.
  using CtaClusterInfo = std::tuple<int, int, int, int, int, int>;

  CtaClusterInfo computeCtaAndClusterConfig(RunnerParams const& params,
                                            KernelMeta const& kernelMeta,
                                            SelectKernelParams& selectKernelParams) const {
    bool isDsv3MinLatencyMode = params.mBatchSize == 1 && params.mMaxSeqLenQ >= 1 &&
                                params.mMaxSeqLenQ <= 16 && params.mHeadDimQk == 576 &&
                                params.mHeadDimV == 512;
    // Do we need to select a new kernel ?
    selectKernelParams.mSelectNewKernel = false;

    // The number of Ctas per Q sequence.
    int numCtasPerSeqQ = (params.mMaxSeqLenQ + kernelMeta.mStepQ - 1) / kernelMeta.mStepQ;
    // Each CTA handles one tokenQ by default for spec-decoding generation kernel, which is used to
    // emulate causal masking (like MTP or Eagle3). Note this will be changed later when the
    // high-throughput spec-decoding generation kernels are integrated.
    if (params.mMaxSeqLenQ > 1 && !isContextKernel(params.mKernelType)) {
      numCtasPerSeqQ = params.mMaxSeqLenQ;
    }

    // Compute the grid dimension Y.
    int numHeadsPerCta = kernelMeta.mGroupsHeadsQ
                             ? std::min(params.mNumHeadsQPerKv, kernelMeta.mMaxNumHeadsQPerKvInCta)
                             : 1;
    int numCtasForAllHeadsQ = params.mNumHeadsQ / numHeadsPerCta;
    TORCH_CHECK(numHeadsPerCta * numCtasForAllHeadsQ == params.mNumHeadsQ,
                "The numHeadsQ/numHeadsKv is not supported.");
    // Take the number of headDim CTAs.
    TORCH_CHECK(kernelMeta.mHeadDimV % selectKernelParams.mHeadDimPerCtaV == 0,
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
      TORCH_CHECK(numCtasForAllHeadsQ == 2 && numCtasPerHeadDim == 2,
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
      int const maxNumCtasPerSeqKv =
          (maxAttentionWindow + kernelMeta.mStepKv - 1) / kernelMeta.mStepKv;
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
                 selectKernelParams.mTileSizeKv == 128 && getEnvUseTileSizeKv64ForTrtllmGen()) {
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

    // Return the number of CTAs for X, Y and Z dimension and the cluster size in the X dimension.
    return std::make_tuple(numCtasPerSeqQ, numCtasPerSeqKv, numCtasX, numCtasY, numCtasZ,
                           clusterDimX);
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
                                             divUp(params.mNumHeadsQPerKv, 16));
    // Compute numCtasPerSeqKv.
    int const numCtasPerSeqKv =
        std::min(maxNumCtasPerSeqKv, std::max(1, int32_t(params.mMultiProcessorCount / numCtas)));
    // Compute the seqLenPerCtaKv.
    int const seqLenPerCtaKv = flashinfer::ceil_div(params.mMaxSeqLenKv, numCtasPerSeqKv);
    // Whether we should use the SwapsMmaAbForGeneration kernel for MLA generation.
    return seqLenPerCtaKv <= 1024 && numCtas <= params.mMultiProcessorCount;
  }

  std::pair<uint64_t, std::string> hashFromRunnerParams(
      RunnerParams const& params, SelectKernelParams& selectKernelParams) const {
    // The updated kernel type.
    FmhaKernelType& kernelType = selectKernelParams.mKernelType;
    // Generation kernelType will use either SwapsMmaAbForGeneration or KeepsMmaAbForGeneration.
    if (isGenerationKernel(params.mKernelType) && isMlaGenKernel(params)) {
      // We use the low-latency kernel (SwapsMmaAbForGeneration with tileSizeQ = 16) when any of the
      // following conditions are met:
      // 1. The number of headsQPerKv is <= 32.
      // 2. The seqLenPerCtaKv <= 1024 based on the benchmark results (this might be fine-tuned
      // later) and
      //    the numCtas (after splitting the heads across multiple CTAs) <=
      //    params.mMultiProcessorCount.

      // Check the conditions.
      if (params.mNumHeadsQPerKv <= 32 || useSwapsMmaAbMlaGenKernel(params)) {
        kernelType = FmhaKernelType::SwapsMmaAbForGeneration;
      } else {
        // Otherwise, we use the high-throughput kernel.
        kernelType = FmhaKernelType::KeepsMmaAbForGeneration;
        // The 2CTA keepsMmaAbForGeneration kernel is used when the numHeadsQPerKv is 128.
        if (params.mNumHeadsQPerKv == 128) {
          selectKernelParams.mUses2CtaMma = true;
          // Each Cta only handles 256 headDimV.
          selectKernelParams.mHeadDimPerCtaV = 256;
        }
      }
    } else if (isGenerationKernel(params.mKernelType)) {
      kernelType = (params.mNumHeadsQPerKv <= 16 && params.mHeadDimQk != 32)
                       ? FmhaKernelType::SwapsMmaAbForGeneration
                       : FmhaKernelType::KeepsMmaAbForGeneration;
    }

    // The maximum number of headsQPerKv that the kernel can support in one Cta.
    int maxNumHeadsQPerKvInCta = 1;
    if (isSwapsMmaAbForGenerationKernel(kernelType)) {
      // Set the corresponding maxNumHeadsQPerKvInCta (tileSizeQ) for low-latency generation
      // kernels.
      maxNumHeadsQPerKvInCta = (params.mNumHeadsQPerKv <= 8) ? 8 : 16;
      TORCH_CHECK((maxNumHeadsQPerKvInCta == 8 || maxNumHeadsQPerKvInCta == 16) &&
                      (params.mNumHeadsQPerKv < maxNumHeadsQPerKvInCta ||
                       params.mNumHeadsQPerKv % maxNumHeadsQPerKvInCta == 0),
                  "Not supported");
    } else if (isKeepsMmaAbForGenerationKernel(kernelType)) {
      // Use the maxNumHeadsQPerKvInCta (tileSizeQ) = 64 for MLA high-throughput generation kernels.
      maxNumHeadsQPerKvInCta = isMlaGenKernel(params) ? 64 : 32;
      TORCH_CHECK((params.mNumHeadsQPerKv < maxNumHeadsQPerKvInCta ||
                   params.mNumHeadsQPerKv % maxNumHeadsQPerKvInCta == 0),
                  "Not supported");
    } else if (isContextKernel(kernelType)) {
      TORCH_CHECK(maxNumHeadsQPerKvInCta == 1, "Not supported");
    }

    // The mask type.
    selectKernelParams.mMaskType = params.mMaskType;
    // Enable sliding window or chunked causal if the max kv sequence length exceeds attention
    // window size or chunked attention size. This is supported by causal-mask context kernels and
    // generation-phase kernels.
    if ((selectKernelParams.mMaskType == TrtllmGenAttentionMaskType::Causal ||
         !isContextKernel(params.mKernelType)) &&
        (params.mMaxSeqLenKv > params.mAttentionWindowSize ||
         params.mChunkedAttentionSize != INT_MAX)) {
      TORCH_CHECK(params.mMaxSeqLenKv <= params.mAttentionWindowSize ||
                      params.mMaxSeqLenKv <= params.mChunkedAttentionSize,
                  "Sliding window attention and chunked attention should not be used together");
      selectKernelParams.mMaskType = TrtllmGenAttentionMaskType::SlidingOrChunkedCausal;
    }
    // NumTokensPerPage is set to 0 when not selecting pagedKv-layout kernels.
    int numTokensPerPage = (!isPagedKv(params.mQkvLayout)) ? 0 : params.mNumTokensPerPage;

    // Debug info.
    std::string info =
        "qkvLayout=" + std::to_string(static_cast<int>(params.mQkvLayout)) +
        ", maskType=" + std::to_string(static_cast<int>(selectKernelParams.mMaskType)) +
        ", kernelType=" + std::to_string(static_cast<int>(kernelType)) +
        ", tileScheduler=" + std::to_string(static_cast<int>(selectKernelParams.mTileScheduler)) +
        ", multiCtasKvMode=" +
        std::to_string(static_cast<int>(selectKernelParams.mMultiCtasKvMode)) +
        ", headDimPerCtaV=" + std::to_string(selectKernelParams.mHeadDimPerCtaV) +
        ", headDimQk=" + std::to_string(params.mHeadDimQk) +
        ", headDimV=" + std::to_string(params.mHeadDimV) +
        ", tileSizeKv=" + std::to_string(selectKernelParams.mTileSizeKv) +
        ", numTokensPerPage=" + std::to_string(numTokensPerPage) +
        ", maxNumHeadsQPerKvInCta=" + std::to_string(maxNumHeadsQPerKvInCta) +
        ", reuseSmemKForV=" + std::to_string(selectKernelParams.mReuseSmemKForV) +
        ", uses2CtaMma=" + std::to_string(selectKernelParams.mUses2CtaMma);
    IKL_LOG_DEBUG(
        "Searching for kernel traits (%d available) in TllmGenFmhaKernel(%s, %s, %s, %d) %s",
        getNumLoadedKernels(), toStr(mDtypeQ), toStr(mDtypeKv), toStr(mDtypeOut), mSM,
        info.c_str());

    return std::make_pair(
        hashID(static_cast<int>(params.mQkvLayout), static_cast<int>(selectKernelParams.mMaskType),
               static_cast<int>(kernelType), static_cast<int>(selectKernelParams.mTileScheduler),
               static_cast<int>(selectKernelParams.mMultiCtasKvMode),
               selectKernelParams.mHeadDimPerCtaV, params.mHeadDimQk, params.mHeadDimV,
               selectKernelParams.mTileSizeKv, numTokensPerPage, maxNumHeadsQPerKvInCta,
               selectKernelParams.mReuseSmemKForV, selectKernelParams.mUses2CtaMma),
        info);
  }

  // Load a single kernel (called by `run()` when needed).
  void loadKernel(uint64_t hashId, unsigned int metaIndex) const {
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
      std::string cubin_path = TllmGenFmhaKernel::getCubinPath() + kernelMeta.mFuncName;
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

  KernelType const* getKernels(Data_type dtypeQ, Data_type dtypeKv, Data_type dtypeOut,
                               unsigned int sm) {
    static std::mutex s_mutex;
    std::lock_guard<std::mutex> lg(s_mutex);

    if (!metainfo_loaded) {
      std::string metainfo_raw =
          getMetaInfo(TllmGenFmhaKernel::getCubinPath() + "flashInferMetaInfo",
                      "8c5630020c0452fb1cd1ea7e3b8fdbb7bf94f71bd899ed5b704a490bdb4f7368", ".h");
      metainfo = KernelType::KernelMeta::loadFromMetaInfoRaw(metainfo_raw);
      metainfo_loaded = true;
    }

    auto const id = hashID(dtypeQ, dtypeKv, dtypeOut, sm);
    auto const findIter = mKernels.find(id);
    if (findIter == mKernels.end()) {
      KernelType* newKernel =
          new KernelType{metainfo.data(), metainfo.size(), dtypeQ, dtypeKv, dtypeOut, sm};
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
      TORCH_CHECK(deviceId < 32, "Invalid deviceId %d (max is 32 devices)", deviceId);
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
  std::vector<KernelType::KernelMeta> metainfo;
  bool metainfo_loaded = false;
};

inline TllmGenFmhaKernel const* getTllmFmhaKernels(Data_type dtypeQ, Data_type dtypeKv,
                                                   Data_type dtypeOut, unsigned int sm) {
#ifndef EXCLUDE_SM_100
  return TllmFmhaKernelFactory::Get().getKernels(dtypeQ, dtypeKv, dtypeOut, sm);
#else
  return nullptr;
#endif  // EXCLUDE_SM_100
}
