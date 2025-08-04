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
    case DATA_TYPE_E4M3:
      return "E4M3";
    case DATA_TYPE_E2M1:
      return "E2M1";
    default:
      return "UNKNOWN";
  }
}

inline Data_type stringToDataType(std::string str) {
  if (str == "DATA_TYPE_FP16") return DATA_TYPE_FP16;
  if (str == "DATA_TYPE_BF16") return DATA_TYPE_BF16;
  if (str == "DATA_TYPE_FP32") return DATA_TYPE_FP32;
  if (str == "DATA_TYPE_E4M3") return DATA_TYPE_E4M3;
  if (str == "DATA_TYPE_E2M1") return DATA_TYPE_E2M1;
  return DATA_TYPE_UNKNOWN;
}

inline int stringToArch(std::string str) {
  if (str == "kSM_90") return kSM_90;
  if (str == "kSM_100") return kSM_100;
  if (str == "kSM_120") return kSM_120;
  return 0;
}

struct TllmGenFmhaKernelMetaInfo {
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
  int mMultiCtasKvMode;
  bool mGroupsHeadsQ;
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

  static TllmGenFmhaKernelMetaInfo fromString(std::string code) {
    std::vector<std::string> param_list;
    std::string current_param = "";
    for (int i = 0; i < code.size(); ++i) {
      if (code[i] != ' ' && code[i] != ',' && code[i] != '"') {
        current_param += code[i];
      }
      if (code[i] == ',') {
        param_list.push_back(current_param);
        current_param = "";
      }
    }
    param_list.push_back(current_param);
    assert(param_list.size() == 25);
    const char* mFuncName = strdup(param_list[11].c_str());
    const char* sha256 = strdup(param_list[24].c_str());
    return TllmGenFmhaKernelMetaInfo{stringToDataType(param_list[0]),
                                     stringToDataType(param_list[1]),
                                     stringToDataType(param_list[2]),
                                     std::stoi(param_list[3]),
                                     std::stoi(param_list[4]),
                                     std::stoi(param_list[5]),
                                     std::stoi(param_list[6]),
                                     std::stoi(param_list[7]),
                                     std::stoi(param_list[8]),
                                     std::stoi(param_list[9]),
                                     stringToArch(param_list[10]),
                                     mFuncName,
                                     std::stoi(param_list[12]),
                                     std::stoi(param_list[13]),
                                     std::stoi(param_list[14]),
                                     std::stoi(param_list[15]),
                                     std::stoi(param_list[16]),
                                     std::stoi(param_list[17]),
                                     std::stoi(param_list[18]),
                                     std::stoi(param_list[19]),
                                     std::stoi(param_list[20]),
                                     param_list[21] == "true" ? true : false,
                                     param_list[22] == "true" ? true : false,
                                     param_list[23] == "true" ? true : false,
                                     sha256};
  };

  static std::vector<TllmGenFmhaKernelMetaInfo> loadFromMetaInfoRaw(std::string metainfo_raw) {
    std::vector<TllmGenFmhaKernelMetaInfo> metainfo;
    int left_braces = std::count(metainfo_raw.begin(), metainfo_raw.end(), '{');
    int right_braces = std::count(metainfo_raw.begin(), metainfo_raw.end(), '}');
    assert(left_braces == right_braces);
    int left_brace_pos = -1;
    for (int i = 0; i < metainfo_raw.size(); ++i) {
      if (metainfo_raw[i] == '{') {
        left_brace_pos = i;
      } else if (metainfo_raw[i] == '}') {
        metainfo.push_back(TllmGenFmhaKernelMetaInfo::fromString(
            metainfo_raw.substr(left_brace_pos + 1, i - left_brace_pos - 1)));
      }
    }
    return metainfo;
  };
};
