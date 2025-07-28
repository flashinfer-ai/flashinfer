/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "GemmOptions.h"

namespace gemm {

namespace tensorrt_llm {
namespace kernels {
// clang-format off

#define TLLM_GEN_COMMIT "c603ed2"
#define TLLM_GEN_EXPORT_VERSION "6.0"
#define TLLM_GEN_GEMM_CONFIG_HASH "a123e72"

static constexpr size_t tllmGenGemmListLen = 4;

#ifndef EXCLUDE_SM_100
inline unsigned char* Gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_et64x16_m64x16x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedS_sm100a_cubin = nullptr;
inline unsigned char* Gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_et64x32_m64x32x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedP_sm100a_cubin = nullptr;
inline unsigned char* Gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_et64x32_m64x32x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedS_sm100a_cubin = nullptr;
inline unsigned char* Gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_et64x8_m64x8x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedP_sm100a_cubin = nullptr;
#endif // EXCLUDE_SM_100

#ifndef EXCLUDE_SM_100
inline unsigned int Gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_et64x16_m64x16x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedS_sm100a_cubin_len = 0;
inline unsigned int Gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_et64x32_m64x32x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedP_sm100a_cubin_len = 0;
inline unsigned int Gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_et64x32_m64x32x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedS_sm100a_cubin_len = 0;
inline unsigned int Gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_et64x8_m64x8x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedP_sm100a_cubin_len = 0;
#endif // EXCLUDE_SM_100


static const gemm::GemmConfig tllmGenGemmList[] = {
#ifndef EXCLUDE_SM_100
{Gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_et64x16_m64x16x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedS_sm100a_cubin, Gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_et64x16_m64x16x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedS_sm100a_cubin_len, 121856, "gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_et64x16_m64x16x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedS_sm100a", 384, "ec338f39ce74011f230100d3c74cc5a11b298cdcb7330a06e42250bcb0693641", { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mBiasType */ gemm::BiasType(0)
, /* mBlockK */ -1
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 1
, /* mDtypeAcc */ trtllm::gen::Dtype(1056776)
, /* mDtypeA */ trtllm::gen::Dtype(1050629)
, /* mDtypeB */ trtllm::gen::Dtype(1050629)
, /* mDtypeC */ trtllm::gen::Dtype(1052672)
, /* mDtypeMmaA */ trtllm::gen::Dtype(1050629)
, /* mDtypeMmaB */ trtllm::gen::Dtype(1050629)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueLdtmDps */ 16
, /* mEpilogueLdtmBits */ 256
, /* mEpilogueTileM */ 64
, /* mEpilogueTileN */ 16
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistLoadTaskInit */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mLayoutA */ gemm::MatrixLayout(0)
, /* mLayoutB */ gemm::MatrixLayout(0)
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaKind */ trtllm::gen::MmaKind(2)
, /* mMmaM */ 64
, /* mMmaN */ 16
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 1
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 6
, /* mNumStagesMma */ 2
, /* mNumStagesMmaWithinWorkTile */ 2
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mPatchF2fp */ 0
, /* mUseShuffledMatrixA */ 0
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(0)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 16
, /* mTileK */ 128
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 1
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, gemm::SmVersion::Sm100a },
{Gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_et64x32_m64x32x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedP_sm100a_cubin, Gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_et64x32_m64x32x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedP_sm100a_cubin_len, 138240, "gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_et64x32_m64x32x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedP_sm100a", 416, "f3fd905fb287c39d083acf7cf208c8ade64e8dae54e1a133e25913bd59130e5c", { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mBiasType */ gemm::BiasType(0)
, /* mBlockK */ -1
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 1
, /* mDtypeAcc */ trtllm::gen::Dtype(1056776)
, /* mDtypeA */ trtllm::gen::Dtype(1050629)
, /* mDtypeB */ trtllm::gen::Dtype(1050629)
, /* mDtypeC */ trtllm::gen::Dtype(1052672)
, /* mDtypeMmaA */ trtllm::gen::Dtype(1050629)
, /* mDtypeMmaB */ trtllm::gen::Dtype(1050629)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueLdtmDps */ 16
, /* mEpilogueLdtmBits */ 256
, /* mEpilogueTileM */ 64
, /* mEpilogueTileN */ 32
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistLoadTaskInit */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mLayoutA */ gemm::MatrixLayout(0)
, /* mLayoutB */ gemm::MatrixLayout(0)
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaKind */ trtllm::gen::MmaKind(2)
, /* mMmaM */ 64
, /* mMmaN */ 32
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 1
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 6
, /* mNumStagesMma */ 2
, /* mNumStagesMmaWithinWorkTile */ 2
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mPatchF2fp */ 0
, /* mUseShuffledMatrixA */ 0
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(0)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 32
, /* mTileK */ 128
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 1
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(1)
 }, gemm::SmVersion::Sm100a },
{Gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_et64x32_m64x32x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedS_sm100a_cubin, Gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_et64x32_m64x32x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedS_sm100a_cubin_len, 138240, "gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_et64x32_m64x32x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedS_sm100a", 384, "5e5546c91a51d769841533ae5339845737922cb7837213fe69050adb059baa3d", { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mBiasType */ gemm::BiasType(0)
, /* mBlockK */ -1
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 1
, /* mDtypeAcc */ trtllm::gen::Dtype(1056776)
, /* mDtypeA */ trtllm::gen::Dtype(1050629)
, /* mDtypeB */ trtllm::gen::Dtype(1050629)
, /* mDtypeC */ trtllm::gen::Dtype(1052672)
, /* mDtypeMmaA */ trtllm::gen::Dtype(1050629)
, /* mDtypeMmaB */ trtllm::gen::Dtype(1050629)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueLdtmDps */ 16
, /* mEpilogueLdtmBits */ 256
, /* mEpilogueTileM */ 64
, /* mEpilogueTileN */ 32
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistLoadTaskInit */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mLayoutA */ gemm::MatrixLayout(0)
, /* mLayoutB */ gemm::MatrixLayout(0)
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaKind */ trtllm::gen::MmaKind(2)
, /* mMmaM */ 64
, /* mMmaN */ 32
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 1
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 6
, /* mNumStagesMma */ 2
, /* mNumStagesMmaWithinWorkTile */ 2
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mPatchF2fp */ 0
, /* mUseShuffledMatrixA */ 0
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(0)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 32
, /* mTileK */ 128
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 1
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(0)
 }, gemm::SmVersion::Sm100a },
{Gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_et64x8_m64x8x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedP_sm100a_cubin, Gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_et64x8_m64x8x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedP_sm100a_cubin_len, 113664, "gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_et64x8_m64x8x32_cga1x1x1_16dp256b_s6_TN_transOut_noShflA_dsFp8_schedP_sm100a", 416, "1f5da4a1effb9237f8542a909a6c6ebfb6586d78727f24a99a76cfa71b4a1bd8", { /* mAllReduceAlgo */ gemm::AllReduceAlgo(0)
, /* mBiasType */ gemm::BiasType(0)
, /* mBlockK */ -1
, /* mClusterDimX */ 1
, /* mClusterDimY */ 1
, /* mClusterDimZ */ 1
, /* mDtypeAcc */ trtllm::gen::Dtype(1056776)
, /* mDtypeA */ trtllm::gen::Dtype(1050629)
, /* mDtypeB */ trtllm::gen::Dtype(1050629)
, /* mDtypeC */ trtllm::gen::Dtype(1052672)
, /* mDtypeMmaA */ trtllm::gen::Dtype(1050629)
, /* mDtypeMmaB */ trtllm::gen::Dtype(1050629)
, /* mEnablesEarlyExit */ 0
, /* mEnablesDelayedEarlyExit */ 0
, /* mEnablesGlobalPtxKnobs */ 1
, /* mEpilogueLdtmDps */ 16
, /* mEpilogueLdtmBits */ 256
, /* mEpilogueTileM */ 64
, /* mEpilogueTileN */ 8
, /* mGridTriggerSecondaryA */ 0
, /* mGridTriggerSecondaryB */ 0
, /* mGridWaitForPrimaryEarlyExit */ 1
, /* mGridWaitForPrimaryA */ 1
, /* mGridWaitForPrimaryB */ 1
, /* mHoistLoadTaskInit */ 1
, /* mHoistMmaTaskTryWaits */ 0
, /* mK */ 2048
, /* mKernelTraits */ {}
, /* mLayoutA */ gemm::MatrixLayout(0)
, /* mLayoutB */ gemm::MatrixLayout(0)
, /* mM */ 256
, /* mMmaK */ 32
, /* mMmaKind */ trtllm::gen::MmaKind(2)
, /* mMmaM */ 64
, /* mMmaN */ 8
, /* mMockAllReduce */ 0
, /* mN */ 256
, /* mNumSlicesForSplitK */ 1
, /* mNumSlicesForSliceK */ 1
, /* mNumStages */ 6
, /* mNumStagesMma */ 2
, /* mNumStagesMmaWithinWorkTile */ 2
, /* mNumStagesMmaAcrossWorkTile */ 1
, /* mNumStagesWorkId */ 3
, /* mOutputDebugTensors */ 0
, /* mPatchF2fp */ 0
, /* mUseShuffledMatrixA */ 0
, /* mSliceK */ 0
, /* mSplitK */ gemm::SplitK(0)
, /* mTransposeMmaOutput */ 1
, /* mTileM */ 128
, /* mTileN */ 8
, /* mTileK */ 128
, /* mUseUnrollLoop2xForMma */ 1
, /* mUseCustomMmaSchedule */ 1
, /* mUseHoistTryWaitForCustomMmaSchedule */ 0
, /* mUseDeepSeekFp8 */ 1
, /* mUsePerTokenSfA */ 0
, /* mUsePerTokenSfB */ 0
, /* mUseTmaStore */ 1
, /* mUseTwoTmaLoadWarps */ 1
, /* mUseTwoMmaWarps */ 0
, /* mSfLayoutA */ trtllm::gen::SfLayout(3)
, /* mSfLayoutB */ trtllm::gen::SfLayout(3)
, /* mSfLayoutC */ trtllm::gen::SfLayout(3)
, /* mSfReshapeFactor */ 1
, /* mTileScheduler */ gemm::TileScheduler(1)
 }, gemm::SmVersion::Sm100a },
#endif // EXCLUDE_SM_100
};
// clang-format on
}  // namespace kernels
}  // namespace tensorrt_llm
}  // namespace gemm
