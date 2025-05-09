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
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 141968, 512, 2, 32, 0, 2, 8, 0, true, false, false, false, "221385b07d7f8d654b3b3b4fd02228c80afa3e96d262080285314abceed0a9fe"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 141968, 512, 2, 64, 0, 2, 8, 0, true, false, false, false, "b8395c3b0b6699d3e5f63227e6660b3cd7ba0bb12134d7fb74c165b403c95bef"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 141968, 512, 2, 16, 0, 2, 8, 0, true, false, false, false, "880243eb22c2196ec62a82e311ad3afd8c691fceb0ce4bb5c58ae34671d1e9ac"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32MultiCtasKvModeVarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 141968, 512, 2, 32, 0, 2, 8, 0, true, true, false, false, "81825e4e234846ad9ec396a219115745296fc18ddb438fd646b99145f2cc2c0c"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64MultiCtasKvModeVarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 141968, 512, 2, 64, 0, 2, 8, 0, true, true, false, false, "958d617c1837772951a4bd2675103a6893539265c23d65ad96270b0ceb117765"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ8TileSizeKv128PersistentSwapsMmaAbForGeneration", 146064, 512, 2, 64, 0, 2, 8, 1, true, false, false, false, "dfd53c4c28eee59c19f338177880af54fba93271ba6a0e778a66cb2c43cc0e8a"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ8TileSizeKv128PersistentSwapsMmaAbForGeneration", 146064, 512, 2, 32, 0, 2, 8, 1, true, false, false, false, "80d6e0b2dc4a76e3888242d45cdb10ee32faf52c94fa83021d89d8516167d77d"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ8TileSizeKv128PersistentSwapsMmaAbForGeneration", 146064, 512, 2, 16, 0, 2, 8, 1, true, false, false, false, "c936ebcf76bd6c38ebee94a15d994482fcc91ead2ef1cbfdbfa32c773aeaf405"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16MultiCtasKvModeVarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 141968, 512, 2, 16, 0, 2, 8, 0, true, true, false, false, "6496e4a4da29153cf45efca1867cea7e663aeedc89766258977626046605299f"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 142992, 512, 2, 32, 0, 2, 8, 0, true, false, false, false, "0f6d4911c18031ebf31f9c7607560229c9b2c12f030748b89f22a4999bb8afbc"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 142992, 512, 2, 64, 0, 2, 8, 0, true, false, false, false, "c802fefe90c7881bdd88b08706717ef1dff99114aa79b0af1a22286d90bcc2f9"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 142992, 512, 2, 16, 0, 2, 8, 0, true, false, false, false, "94b46ad77fdbc89a369918b9fb2b4a263dc11029bdb14bc64c527f9a8573bd1a"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ8TileSizeKv128PersistentSwapsMmaAbForGeneration", 147088, 512, 2, 64, 0, 2, 8, 1, true, false, false, false, "1c137e3c7408dc448adecdadc902ade577fb1d83d975d4c906fd9cd8082ddc39"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ8TileSizeKv128PersistentSwapsMmaAbForGeneration", 147088, 512, 2, 32, 0, 2, 8, 1, true, false, false, false, "86ce7509cf30ea08d5446f1dce4d935ea6a31d469d54e98eb7eb311e0fb89b29"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32MultiCtasKvModeVarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 142992, 512, 2, 32, 0, 2, 8, 0, true, true, false, false, "ae23112acf99eb65213521769c5871a953e6668e76158aa9c1195b1468cc8dd8"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64MultiCtasKvModeVarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 142992, 512, 2, 64, 0, 2, 8, 0, true, true, false, false, "3569bb6adb665fdce3730ec6f142d9f948ff22b4b5546fa3ce236cd10760c0e9"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16MultiCtasKvModeVarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 142992, 512, 2, 16, 0, 2, 8, 0, true, true, false, false, "7c4e6e3d517dd1a82593087e0a69cbc518a6e7fb562bd043f657d1093211c9e3"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ8TileSizeKv128PersistentSwapsMmaAbForGeneration", 147088, 512, 2, 16, 0, 2, 8, 1, true, false, false, false, "4d5cc2651be8f97a2a13d8daddb6bf472596bc797d4618ee39894064cd28a068"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 167952, 512, 2, 64, 0, 3, 128, 0, true, false, false, false, "92be23c688dcb30b8f4369e25d4ba34e64b6e78c27bf88de07d5a8271d06b132"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 167952, 512, 2, 32, 0, 3, 128, 0, true, false, false, false, "b021db22b47a9328e87f7233d37848f77e3e4b4a362d4db593b9452b124218b7"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 167952, 512, 2, 16, 0, 3, 128, 0, true, false, false, false, "2d867d79bd61231eb912c85a3d1b2193bb9771bc8150c4801d77a5f65810d50f"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32MultiCtasKvModeVarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 167968, 512, 2, 32, 0, 3, 128, 0, true, true, false, false, "beced06006eb668edd42e662df9437ea74810b216fff87cb0e119bf6dc135cb8"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ128TileSizeKv128PersistentKeepsMmaAbForGeneration", 200720, 512, 2, 32, 0, 3, 128, 1, true, false, false, false, "ea1f0f4cc528029e68c74354a3f2c1c1085bd9c58b9ec7851bc1885b5dece80b"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ128TileSizeKv128PersistentKeepsMmaAbForGeneration", 200720, 512, 2, 16, 0, 3, 128, 1, true, false, false, false, "9fadac31c8be3863054a90a7fd054d7ee563fb95f518dba0962b83397e6d837a"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64MultiCtasKvModeVarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 167968, 512, 2, 64, 0, 3, 128, 0, true, true, false, false, "282c7d2b537c61e1f88134e33dc43dd9a3bb197a0eeb2498ea4818a4eb1f5b07"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ128TileSizeKv128PersistentKeepsMmaAbForGeneration", 200720, 512, 2, 64, 0, 3, 128, 1, true, false, false, false, "5778e075b3d0bd48a9a3f44556638a8a0a92f4aece233cfbffc15db317b96d64"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16MultiCtasKvModeVarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 167968, 512, 2, 16, 0, 3, 128, 0, true, true, false, false, "925a69f158af6a96719be39713cb0fed93146d2b7b4fabb472dd6144ec7179f5"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 168976, 512, 2, 64, 0, 3, 128, 0, true, false, false, false, "c3cb9c0824c7ecace359d18376ad94dcb7fdc7f5b0855b45d3647a20e04fe198"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ128TileSizeKv128PersistentKeepsMmaAbForGeneration", 201744, 512, 2, 64, 0, 3, 128, 1, true, false, false, false, "a7048d01cbe1004fb872fcc5d3860c8a4a3bbced155b5cba9a02d6bd958e47c0"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 168976, 512, 2, 32, 0, 3, 128, 0, true, false, false, false, "6df37d0e3586e16cca944270725b57c405b5f4187925222cdc67218b00b80335"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 168976, 512, 2, 16, 0, 3, 128, 0, true, false, false, false, "d8cdea48969a85ee6a30a7c5b72f295a1e0bf91b49791b054bd6764d65cb218f"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32MultiCtasKvModeVarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 168992, 512, 2, 32, 0, 3, 128, 0, true, true, false, false, "200f7bb7c2319b10077412aaa4a4d109edef05526a19c8c34bc1299449becff3"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ128TileSizeKv128PersistentKeepsMmaAbForGeneration", 201744, 512, 2, 32, 0, 3, 128, 1, true, false, false, false, "e3f33de8fd40c89a44d8f57f4c03d0419d7137a3261576ac0d7e507f8293edab"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ128TileSizeKv128PersistentKeepsMmaAbForGeneration", 201744, 512, 2, 16, 0, 3, 128, 1, true, false, false, false, "521ca354ede861f0f31544a4a73fe9807bbdaf13f0979c14f6800a222490debb"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64MultiCtasKvModeVarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 168992, 512, 2, 64, 0, 3, 128, 0, true, true, false, false, "9c6fe66aea1a236e13758b6c72238bae2a4f4c485165e1c805bf981edbbdcd55"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16MultiCtasKvModeVarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 168992, 512, 2, 16, 0, 3, 128, 0, true, true, false, false, "501848e5cae5395781ef47920db17c5a37a0b9bf958b03525f430c89bab1fd9f"},
// clang-format on
    };
