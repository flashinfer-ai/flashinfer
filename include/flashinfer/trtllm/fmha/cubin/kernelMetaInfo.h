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
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16MultiCtasKvModeVarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 201760, 512, 2, 16, 0, 3, 8, 0, true, true, false, false, "9dd5456f07ed949ab8758dda016a603643fb126782225af07621832fbe80fba5"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16MultiCtasKvModeVarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 145040, 512, 2, 16, 0, 2, 8, 0, true, true, false, false, "10cc7593c14877034ce2f8b79345bab59cde355ca960c5d4f626fc6ddd1df2a7"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ128TileSizeKv128PersistentKeepsMmaAbForGeneration", 201744, 512, 2, 16, 0, 3, 8, 1, true, false, false, false, "fe1f5c8589fafdf71aaa1336e250ed81d24dfd34baed9541bdff6e2a75331743"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 201744, 512, 2, 16, 0, 3, 8, 0, true, false, false, false, "0f306582db9b829cd29faafaf70831fe4e52b0683014734829e498bebb211d8b"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ8TileSizeKv128PersistentSwapsMmaAbForGeneration", 147088, 512, 2, 16, 0, 2, 8, 1, true, false, false, false, "635e378ba1b367c9927f664799cf7eef5b0b97f8d6661b1dddf36a6c6355f062"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 145040, 512, 2, 16, 0, 2, 8, 0, true, false, false, false, "5db0e5fc5bf964a40afeac74919cbe491a4c19a41909d40ed3802d9b701006d3"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32MultiCtasKvModeVarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 201760, 512, 2, 32, 0, 3, 8, 0, true, true, false, false, "e36c03dc4e87e2b36c146c68b021136c92a7d43a6fd2d799310b426d422e89e5"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32MultiCtasKvModeVarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 145040, 512, 2, 32, 0, 2, 8, 0, true, true, false, false, "5e394837a62ec7c1c0ec4dd7c5759ef1d64ae0978da24107d2294921b890ed1c"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ128TileSizeKv128PersistentKeepsMmaAbForGeneration", 201744, 512, 2, 32, 0, 3, 8, 1, true, false, false, false, "6735d151f6c1588de3e76f10636f6907c8ed9d9d1f393b18a8f684bfcd607a25"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 201744, 512, 2, 32, 0, 3, 8, 0, true, false, false, false, "a5d84de511f32185c77243efae8cd8ba1ae9240daff519377a66898a6c323003"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ8TileSizeKv128PersistentSwapsMmaAbForGeneration", 147088, 512, 2, 32, 0, 2, 8, 1, true, false, false, false, "18477e80d16e6c2f9f0e3b2c283bcabc6fda1b3ef7ac05834f807b1b217e9b87"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 145040, 512, 2, 32, 0, 2, 8, 0, true, false, false, false, "b1070c5d5ad41df637ac68f6450cf8012da8e73ec223be50d8ab1a27974dee9a"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64MultiCtasKvModeVarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 201760, 512, 2, 64, 0, 3, 8, 0, true, true, false, false, "c91022d24178d1fc8060b9672cf1b1c5c29fcbab70e5908a6cb3b4eed201ffc0"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64MultiCtasKvModeVarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 145040, 512, 2, 64, 0, 2, 8, 0, true, true, false, false, "f2031ecc847078144ef746cc306a91c7cc13c331fb952a840d36432f8099460a"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ128TileSizeKv128PersistentKeepsMmaAbForGeneration", 201744, 512, 2, 64, 0, 3, 8, 1, true, false, false, false, "9ee52611cc485a950ff296b4aefa0f34d6294ca92e4f0b709f60202b361f71cd"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 201744, 512, 2, 64, 0, 3, 8, 0, true, false, false, false, "50fd883e206f2c106ea6dddd96d21e1075360bae233f6f7fbb596970c7dfb16a"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ8TileSizeKv128PersistentSwapsMmaAbForGeneration", 147088, 512, 2, 64, 0, 2, 8, 1, true, false, false, false, "0e4ca851036b41d726a1e6f6e033b771ff8c127b88e7aaad9fd45e56b966e922"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvE4m3AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 145040, 512, 2, 64, 0, 2, 8, 0, true, false, false, false, "53e45d8d9b6ff7b171062044f0dce6204bad3c67d056d1c3ede595d53cb053d6"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16MultiCtasKvModeVarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 200736, 512, 2, 16, 0, 3, 8, 0, true, true, false, false, "482cebdc44c02a378db6248b0a8c0e131f4bf5bf3b0b4519f8859255cfb40962"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16MultiCtasKvModeVarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 144016, 512, 2, 16, 0, 2, 8, 0, true, true, false, false, "9baec066fd9887fecf8d1bb0b88adcb01c7e43c583f7ad6d4b6eaca45f0930ee"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ128TileSizeKv128PersistentKeepsMmaAbForGeneration", 200720, 512, 2, 16, 0, 3, 8, 1, true, false, false, false, "abd1014322d70f6ef676290eba0512261ff15c6da141e62b656b8ad20d119ebe"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 200720, 512, 2, 16, 0, 3, 8, 0, true, false, false, false, "ecf03c4c2e39e57fea8a74a897c97a68edbd86ce9e081e4f605e5885f2925a0f"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ8TileSizeKv128PersistentSwapsMmaAbForGeneration", 146064, 512, 2, 16, 0, 2, 8, 1, true, false, false, false, "fceb1759fdda361c6fa24e9747b65a584ba4b40cffcb4ce572d5c648d267678a"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP16VarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 144016, 512, 2, 16, 0, 2, 8, 0, true, false, false, false, "8fd3cf7ef4e980f82e954a758019a20b15a2796a1cddf67b12dd98a2c7a16028"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32MultiCtasKvModeVarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 200736, 512, 2, 32, 0, 3, 8, 0, true, true, false, false, "8b91e49d957af0bff352bfa2df1310075cde6909e577f1b2bf53e99ed24e0d3b"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32MultiCtasKvModeVarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 144016, 512, 2, 32, 0, 2, 8, 0, true, true, false, false, "0330a174ceebc0af169c7c929196e2421ba58a18faf4b150df5d84daf44bdb67"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ128TileSizeKv128PersistentKeepsMmaAbForGeneration", 200720, 512, 2, 32, 0, 3, 8, 1, true, false, false, false, "a9ddcc55a54efc8fe7b12bcef7f0ca3702d6dc52494f90acf806dac484c84338"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 200720, 512, 2, 32, 0, 3, 8, 0, true, false, false, false, "9a0b6f967eeecc021716463287a7c1e1a68d495c4ef5fbda1657e49217fe03f1"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ8TileSizeKv128PersistentSwapsMmaAbForGeneration", 146064, 512, 2, 32, 0, 2, 8, 1, true, false, false, false, "88e6cec0201727bbc24a31e1f1d6862042dbe2474af925c0ae023db16abe29e3"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP32VarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 144016, 512, 2, 32, 0, 2, 8, 0, true, false, false, false, "f6494082ccc0214b7e9ac367e2bca8768a051ce8615228d89d930311f8ffe3d6"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64MultiCtasKvModeVarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 200736, 512, 2, 64, 0, 3, 8, 0, true, true, false, false, "fef49370c9a7891f18fe9dbf7d89a0473204ab73ffa65a17effbd0f907ed3742"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64MultiCtasKvModeVarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 144016, 512, 2, 64, 0, 2, 8, 0, true, true, false, false, "2deec436d6868fc9d9387c23bef802b3d9f7560821f7009521d8d52680461369"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ128TileSizeKv128PersistentKeepsMmaAbForGeneration", 200720, 512, 2, 64, 0, 3, 8, 1, true, false, false, false, "0edd2a76ac4f730c8a45c70fa7e240688cb86c9b66091fbfc2e0c9004649d5e9"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 128, 128, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ128TileSizeKv128StaticKeepsMmaAbForGeneration", 200720, 512, 2, 64, 0, 3, 8, 0, true, false, false, false, "9bc5ade51284c932e80ec3e61f936002879d6cccdc1d35aed6fb2238126836a5"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ8TileSizeKv128PersistentSwapsMmaAbForGeneration", 146064, 512, 2, 64, 0, 2, 8, 1, true, false, false, false, "31414a24b8b2e612a8b2388b3c93f196dc62f77d7f8a9651e298f9b0cbafdf5b"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, DATA_TYPE_FP16, 8, 128, 8, 256, 128, 128, 128, kSM_100, "fmhaSm100Kernel_QFp16KvFp16AccFp32OFp16HQk128HV128LayoutPagedKvMaskDenseP64VarSeqLenTileSizeQ8TileSizeKv128StaticSwapsMmaAbForGeneration", 144016, 512, 2, 64, 0, 2, 8, 0, true, false, false, false, "2211bebf1e936e47a0dfc64f036d53bb2da020356c466603c469c62e83789227"},
// clang-format on
    };
