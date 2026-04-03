/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <vector>

#include "../common.h"
#include "flashInferMetaInfo.h"

namespace flashinfer::trtllm_fmha_meta {
struct TllmGenFmhaKernelMetaInfoAdapter {
  Data_type mDataTypeQ;
  Data_type mDataTypeKv;
  Data_type mDataTypeO;
  int mTileSizeQ;
  int mTileSizeKv;
  int mStepQ;
  int mStepKv;
  int mHeadDimPerCtaV;
  int mHeadDimQk;
  int mHeadDimV;
  int mSM;
  const unsigned char* mCubin;
  unsigned int mCubinSize;
  const char* mFuncName;
  int mSharedMemBytes;
  int mThreadsPerCTA;
  int mQkvLayout;
  int mNumTokensPerPage;
  int mMaskType;
  int mKernelType;
  int mTileScheduler;
  int mMultiCtasKvMode;
  int mNumEltsPerSageAttnBlkQ;
  int mNumEltsPerSageAttnBlkK;
  int mNumEltsPerSageAttnBlkP;
  int mNumEltsPerSageAttnBlkV;
  bool mGroupsHeadsQ;
  bool mGroupsTokensHeadsQ;
  bool mReuseSmemKForV;
  bool m2CtaMma;
  bool mSparseMla;
  bool mSkipsSoftmaxWhenPossible;
  Data_type mDataTypeQkReinterpret;
  bool mReserved1;
  bool mReserved2;
  bool mIsKernelVx;
  const char* sha256;

  inline bool isKernelVx() const { return mIsKernelVx; }
};

inline TllmGenFmhaKernelMetaInfoAdapter toAdapter(
    tensorrt_llm::kernels::TllmGenFmhaKernelMetaInfo const& meta) {
  return {meta.mDataTypeQ,
          meta.mDataTypeKv,
          meta.mDataTypeO,
          meta.mTileSizeQ,
          meta.mTileSizeKv,
          meta.mStepQ,
          meta.mStepKv,
          meta.mHeadDimPerCtaV,
          meta.mHeadDimQk,
          meta.mHeadDimV,
          meta.mSM,
          meta.mCubin,
          meta.mCubinSize,
          meta.mFuncName,
          meta.mSharedMemBytes,
          meta.mThreadsPerCTA,
          meta.mQkvLayout,
          meta.mNumTokensPerPage,
          meta.mMaskType,
          meta.mKernelType,
          meta.mTileScheduler,
          meta.mMultiCtasKvMode,
          0,
          0,
          0,
          0,
          meta.mGroupsHeadsQ,
          meta.mGroupsTokensHeadsQ,
          meta.mReuseSmemKForV,
          meta.m2CtaMma,
          meta.mSparseMla,
          meta.mSkipsSoftmaxWhenPossible,
          meta.mDataTypeQ,
          meta.mReserved1,
          meta.mReserved2,
          false,
          meta.sha256};
}

inline TllmGenFmhaKernelMetaInfoAdapter toAdapter(
    tensorrt_llm::kernels::TllmGenFmhaKernelMetaInfoVx const& meta) {
  return {meta.mDataTypeQ,
          meta.mDataTypeKv,
          meta.mDataTypeO,
          meta.mTileSizeQ,
          meta.mTileSizeKv,
          meta.mStepQ,
          meta.mStepKv,
          meta.mHeadDimPerCtaV,
          meta.mHeadDimQk,
          meta.mHeadDimV,
          meta.mSM,
          meta.mCubin,
          meta.mCubinSize,
          meta.mFuncName,
          meta.mSharedMemBytes,
          meta.mThreadsPerCTA,
          meta.mQkvLayout,
          meta.mNumTokensPerPage,
          meta.mMaskType,
          meta.mKernelType,
          meta.mTileScheduler,
          meta.mMultiCtasKvMode,
          meta.mNumEltsPerSageAttnBlkQ,
          meta.mNumEltsPerSageAttnBlkK,
          meta.mNumEltsPerSageAttnBlkP,
          meta.mNumEltsPerSageAttnBlkV,
          meta.mGroupsHeadsQ,
          false,
          meta.mReuseSmemKForV,
          meta.m2CtaMma,
          meta.mSparseMla,
          meta.mSkipsSoftmaxWhenPossible,
          meta.mDataTypeQkReinterpret,
          false,
          false,
          true,
          meta.sha256};
}

inline const std::vector<TllmGenFmhaKernelMetaInfoAdapter>& getAllKernelMetaInfos() {
  static std::vector<TllmGenFmhaKernelMetaInfoAdapter> metas = [] {
    std::vector<TllmGenFmhaKernelMetaInfoAdapter> combined;
    combined.reserve(sizeof(tensorrt_llm::kernels::sTllmGenFmhaKernelMetaInfos) /
                         sizeof(tensorrt_llm::kernels::sTllmGenFmhaKernelMetaInfos[0]) +
                     sizeof(tensorrt_llm::kernels::sTllmGenFmhaKernelMetaInfosVx) /
                         sizeof(tensorrt_llm::kernels::sTllmGenFmhaKernelMetaInfosVx[0]));
    for (auto const& meta : tensorrt_llm::kernels::sTllmGenFmhaKernelMetaInfos) {
      auto adapter = toAdapter(meta);
      adapter.mIsKernelVx = false;
      combined.push_back(adapter);
    }
    for (auto const& meta : tensorrt_llm::kernels::sTllmGenFmhaKernelMetaInfosVx) {
      auto adapter = toAdapter(meta);
      adapter.mIsKernelVx = true;
      combined.push_back(adapter);
    }
    return combined;
  }();
  return metas;
}

// Emits Trtllm standard-style metainfo to run reductions for LLM workloads only.
inline tensorrt_llm::kernels::TllmGenFmhaKernelMetaInfo toLlm(
    TllmGenFmhaKernelMetaInfoAdapter const& meta) {
  return {meta.mDataTypeQ,
          meta.mDataTypeKv,
          meta.mDataTypeO,
          meta.mTileSizeQ,
          meta.mTileSizeKv,
          meta.mStepQ,
          meta.mStepKv,
          meta.mHeadDimPerCtaV,
          meta.mHeadDimQk,
          meta.mHeadDimV,
          meta.mSM,
          meta.mCubin,
          meta.mCubinSize,
          meta.mFuncName,
          meta.mSharedMemBytes,
          meta.mThreadsPerCTA,
          meta.mQkvLayout,
          meta.mNumTokensPerPage,
          meta.mMaskType,
          meta.mKernelType,
          meta.mTileScheduler,
          meta.mMultiCtasKvMode,
          meta.mGroupsHeadsQ,
          meta.mGroupsTokensHeadsQ,
          meta.mReuseSmemKForV,
          meta.m2CtaMma,
          meta.mSparseMla,
          meta.mSkipsSoftmaxWhenPossible,
          meta.mReserved1,
          meta.mReserved2,
          meta.sha256};
}
}  // namespace flashinfer::trtllm_fmha_meta
