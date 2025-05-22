/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
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

#include <flashinfer/trtllm/fmha/kernelParams.h>
// Helper to print Data_type
inline const char* dataTypeToString(Data_type dt) {
  switch (dt) {
    case DATA_TYPE_FP16:
      return "FP16";
    case DATA_TYPE_BF16:
      return "BF16";
    case DATA_TYPE_FP32:
      return "FP32";
    default:
      return "UNKNOWN";
  }
}

static const struct TllmGenFmhaKernelMetaInfo {
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
  const char* mFuncName;
  int mSharedMemBytes;
  int mThreadsPerCTA;
  int mQkvLayout;
  int mNumTokensPerPage;
  int mMaskType;
  int mKernelType;
  int mMaxNumHeadsQPerKvInCta;
  int mTileScheduler;
  bool mGroupsHeadsQ;
  bool mMultiCtasKvMode;
  bool mReuseSmemKForV;
  bool m2CtaMma;
  const char* sha256;

  void print() const {
    std::cout << "TllmGenFmhaKernelMetaInfo {\n";
    std::cout << "  mDataTypeQ: " << dataTypeToString(mDataTypeQ) << "\n";
    std::cout << "  mDataTypeKv: " << dataTypeToString(mDataTypeKv) << "\n";
    std::cout << "  mDataTypeO: " << dataTypeToString(mDataTypeO) << "\n";
    std::cout << "  mTileSizeQ: " << mTileSizeQ << "\n";
    std::cout << "  mTileSizeKv: " << mTileSizeKv << "\n";
    std::cout << "  mStepQ: " << mStepQ << "\n";
    std::cout << "  mStepKv: " << mStepKv << "\n";
    std::cout << "  mHeadDimPerCtaV: " << mHeadDimPerCtaV << "\n";
    std::cout << "  mHeadDimQk: " << mHeadDimQk << "\n";
    std::cout << "  mHeadDimV: " << mHeadDimV << "\n";
    std::cout << "  mSM: " << mSM << "\n";
    std::cout << "  mFuncName: " << (mFuncName ? mFuncName : "null") << "\n";
    std::cout << "  mSharedMemBytes: " << mSharedMemBytes << "\n";
    std::cout << "  mThreadsPerCTA: " << mThreadsPerCTA << "\n";
    std::cout << "  mQkvLayout: " << mQkvLayout << "\n";
    std::cout << "  mNumTokensPerPage: " << mNumTokensPerPage << "\n";
    std::cout << "  mMaskType: " << mMaskType << "\n";
    std::cout << "  mKernelType: " << mKernelType << "\n";
    std::cout << "  mMaxNumHeadsQPerKvInCta: " << mMaxNumHeadsQPerKvInCta << "\n";
    std::cout << "  mTileScheduler: " << mTileScheduler << "\n";
    std::cout << "  mGroupsHeadsQ: " << std::boolalpha << mGroupsHeadsQ << "\n";
    std::cout << "  mMultiCtasKvMode: " << std::boolalpha << mMultiCtasKvMode << "\n";
    std::cout << "  mReuseSmemKForV: " << std::boolalpha << mReuseSmemKForV << "\n";
    std::cout << "  m2CtaMma: " << std::boolalpha << m2CtaMma << "\n";
    std::cout << "  sha256: " << (sha256 ? sha256 : "null") << "\n";
    std::cout << "}\n";
  }
} sTllmGenFmhaKernelMetaInfos[] = {
// clang-format off
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 141968, 512, 2, 64, 0, 2, 8, 0, true, false, false, false, "12415769d9b85f0d8cce3dd8f4e6ec0d24a0e863d73ca75d5f1bab3ebf07688a"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 141968, 512, 2, 16, 0, 2, 8, 0, true, false, false, false, "7b3c9397803282ff6ccb20a11b741041f8fdb2433b58a20ae59dc0599b0f414b"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 141968, 512, 2, 32, 0, 2, 8, 0, true, false, false, false, "fe72f3b2f0b73d72d81441b0573c7193b37d8f01fe28c7396146fea495e024d2"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64MultiCtasKvModeVarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 141968, 512, 2, 64, 0, 2, 8, 0, true, true, false, false, "a0f1e8c6112e5cdf72eb35d33112269f8c85bb79f7b25bf15f67c935656cb343"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ8TileSizeKv128PersistentSwapsMmaAbForGeneration", 146064, 512, 2, 16, 0, 2, 8, 1, true, false, false, false, "0b3c97dff88a6ead6c74c567039114ccc95bd84552064b1efb7fb4e99517a040"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32MultiCtasKvModeVarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 141968, 512, 2, 32, 0, 2, 8, 0, true, true, false, false, "c66327a27f29fef42465876c564c286e00d3700032abfee19c1008d92712c10d"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16MultiCtasKvModeVarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 141968, 512, 2, 16, 0, 2, 8, 0, true, true, false, false, "394f99c775b0cd62fed8b2ac53932a4f703a18396c26de661d5eca48ac347ddd"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ8TileSizeKv128PersistentSwapsMmaAbForGeneration", 146064, 512, 2, 32, 0, 2, 8, 1, true, false, false, false, "21a39ae8b4d97c24b9a671aa35d3a16d13afa4c1254b6ac6d01b480b0c40912b"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ8TileSizeKv128PersistentSwapsMmaAbForGeneration", 146064, 512, 2, 64, 0, 2, 8, 1, true, false, false, false, "a08ebd7d8c4683916729d198390da664f337b03ed61dea93cd1c71c3c3b75f3e"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 142992, 512, 2, 64, 0, 2, 8, 0, true, false, false, false, "7dcad5594f6f0f5a2d1899a1fce8498f99deb87c687c7cb3477fb8fd096d80bb"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 142992, 512, 2, 32, 0, 2, 8, 0, true, false, false, false, "d9c73692af4ceca4b0daee48a7495a20a80bedf1bf35cc8a3544d8c08f11e5b4"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 142992, 512, 2, 16, 0, 2, 8, 0, true, false, false, false, "4bb4f3e5f42d53c6fa147f8181c85885f6d0af8f9354674f7613e929cae03cd8"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ8TileSizeKv128PersistentSwapsMmaAbForGeneration", 147088, 512, 2, 32, 0, 2, 8, 1, true, false, false, false, "ddb7b4199f2c3b0b1bcd5dd407131142f6735683e3b82f816949a3c7b06b8820"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ8TileSizeKv128PersistentSwapsMmaAbForGeneration", 147088, 512, 2, 64, 0, 2, 8, 1, true, false, false, false, "b8efc638750fd565d68fb76046f3c5bb650b28d25885996bb0d58c409da827c5"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64MultiCtasKvModeVarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 142992, 512, 2, 64, 0, 2, 8, 0, true, true, false, false, "dd4e45937939a0b4985a481fd321dc43bc277221030bb36c3872df3accd2272c"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32MultiCtasKvModeVarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 142992, 512, 2, 32, 0, 2, 8, 0, true, true, false, false, "e0fd5dc99dab9fc2b464ee3f2dadb0c8a308da0d4e32587f76afb70f6a55aa54"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16MultiCtasKvModeVarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 142992, 512, 2, 16, 0, 2, 8, 0, true, true, false, false, "c959c55555985ef24abddab0f658c98ceee43b94cf6515a402326fbc59764242"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ8TileSizeKv128PersistentSwapsMmaAbForGeneration", 147088, 512, 2, 16, 0, 2, 8, 1, true, false, false, false, "fb15a1b2d8e392d25bb33a64ba43954aa0e48d597943977c63df77be9c09111d"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 167952, 512, 2, 64, 0, 3, 128, 0, true, false, false, false, "ca0a3cfa51215ca5630633a0ed8a5bdfe6bd35532f322b635ec7cf099d09e913"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 167952, 512, 2, 32, 0, 3, 128, 0, true, false, false, false, "38bdd71768aa7258dce2e2a3ce2ae3d9e1eaa97d40d7d437c29fe6b82bb5e3cf"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32MultiCtasKvModeVarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 167968, 512, 2, 32, 0, 3, 128, 0, true, true, false, false, "dcfb583c4eeb167eba28802a74c428d428824b3f48e4fb1a919d1ce98a495ea8"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ128TileSizeKv128PersistentKeepsMmaAbForGeneration", 200720, 512, 2, 32, 0, 3, 128, 1, true, false, false, false, "8cf5321f8ed898427376d61d231f5bd5622d3a29a3826790434a7747d27aa910"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64MultiCtasKvModeVarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 167968, 512, 2, 64, 0, 3, 128, 0, true, true, false, false, "1ae9cf2e8bdb43fa1b50e79a1d5f1854c4bcdb6f9413b3731205059fdfc0c725"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ128TileSizeKv128PersistentKeepsMmaAbForGeneration", 200720, 512, 2, 64, 0, 3, 128, 1, true, false, false, false, "54b1e32c5394b4d1b94ae584c4602f3b4d89c6ee74af1da49763ae99faa83bd8"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ128TileSizeKv128PersistentKeepsMmaAbForGeneration", 200720, 512, 2, 16, 0, 3, 128, 1, true, false, false, false, "89d4c6b3de8b8d56b0d7bea6932a65c164dcc157a55bc90bc885c7dd0cd08df4"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 167952, 512, 2, 16, 0, 3, 128, 0, true, false, false, false, "71c091f10881a3ac6becc2171358fc938a8d8d29846ac0421e6b448a94ca27f4"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16MultiCtasKvModeVarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 167968, 512, 2, 16, 0, 3, 128, 0, true, true, false, false, "785553d8a56eb8537847f0002e1cfc225b8e2dd1688bfb1bd1fb9be9691ecc1d"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 168976, 512, 2, 64, 0, 3, 128, 0, true, false, false, false, "4b27bbcd19771d6bfc0b2b7d582a8518a8908f87a4e7bd5da25bf2e298b4486d"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 168976, 512, 2, 16, 0, 3, 128, 0, true, false, false, false, "e0db921302ae8d0fa5940020ddeea6f33edb1c55ed737cf5dee19abaa5ed1256"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 168976, 512, 2, 32, 0, 3, 128, 0, true, false, false, false, "0287df54a58efd1f186c5b0890e3ca071f9b404ca852430ae334b6515f950deb"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ128TileSizeKv128PersistentKeepsMmaAbForGeneration", 201744, 512, 2, 64, 0, 3, 128, 1, true, false, false, false, "a13e4ace1bc64d61f82fa37f742fd4b4d3f3edf5227ef6f143a45a120c521824"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ128TileSizeKv128PersistentKeepsMmaAbForGeneration", 201744, 512, 2, 32, 0, 3, 128, 1, true, false, false, false, "47cb90799e88e8ac847c4ed7228a6b20260babf76a95f5074fa95fddb684a6a0"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ128TileSizeKv128PersistentKeepsMmaAbForGeneration", 201744, 512, 2, 16, 0, 3, 128, 1, true, false, false, false, "b6ec521c13f8f337c44c45f44de586f85ea9ffe6ecb11430e89478ecbabc523b"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64MultiCtasKvModeVarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 168992, 512, 2, 64, 0, 3, 128, 0, true, true, false, false, "93ed4516663f413d700e8061a82776c8b728db59535a16a42cf1ac3488e7287b"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32MultiCtasKvModeVarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 168992, 512, 2, 32, 0, 3, 128, 0, true, true, false, false, "134b40197b618d922e09640130830f0107055590bfe658bd3b6e11acbfc85dad"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16MultiCtasKvModeVarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 168992, 512, 2, 16, 0, 3, 128, 0, true, true, false, false, "48e8f61657a02ff5550f1b4bd733fd94b6272f997b5bb0f1f4b5cbf04c3154e2"},
// clang-format on
    };
