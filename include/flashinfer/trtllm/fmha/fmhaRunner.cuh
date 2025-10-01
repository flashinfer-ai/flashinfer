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

#include <cuda_runtime.h>

#include "../../exception.h"
#include "fmhaKernels.cuh"
#include "fmhaRunnerParams.h"

class TllmGenFmhaRunner {
 public:
  // Constructor.
  explicit TllmGenFmhaRunner(Data_type dtypeQ, Data_type dtypeKv, Data_type dtypeOut)
      : mSM(getSMVersion()), mDtypeQ(dtypeQ), mDtypeKv(dtypeKv), mDtypeOut(dtypeOut) {
    FLASHINFER_CHECK(mSM == kSM_100 || mSM == kSM_103, "Unsupported architecture");
    FLASHINFER_CHECK(
        mDtypeQ == DATA_TYPE_E4M3 || mDtypeQ == DATA_TYPE_FP16 || mDtypeQ == DATA_TYPE_BF16,
        "Unsupported Q data type: " + std::string(toStr(mDtypeQ)));
    FLASHINFER_CHECK(
        mDtypeKv == DATA_TYPE_E4M3 || mDtypeKv == DATA_TYPE_FP16 || mDtypeKv == DATA_TYPE_BF16,
        "Unsupported Kv data type: " + std::string(toStr(mDtypeKv)));
    FLASHINFER_CHECK(mDtypeOut == DATA_TYPE_E4M3 || mDtypeOut == DATA_TYPE_FP16 ||
                         mDtypeOut == DATA_TYPE_BF16 || mDtypeOut == DATA_TYPE_E2M1,
                     "Unsupported Output data type: " + std::string(toStr(mDtypeOut)));
    mKernel = getTllmFmhaKernels(mDtypeQ, mDtypeKv, mDtypeOut, mSM);
  }

  TllmGenFmhaRunner() = default;

  // Check if fmha is supported.
  bool isSupported(TllmGenFmhaRunnerParams const& runnerParams) {
    return mKernel->checkIfKernelExist(runnerParams).first;
  }

  // Check if fmha is supported with additional info.
  std::pair<bool, std::string> isSupportedWithInfo(
      TllmGenFmhaRunnerParams const& runnerParams) const {
    return mKernel->checkIfKernelExist(runnerParams);
  }

  // Run the fmha kernel.
  void run(TllmGenFmhaRunnerParams const& runnerParams) { mKernel->run(runnerParams); }

 private:
  // The input/output datatype.
  Data_type mDtypeQ, mDtypeKv, mDtypeOut;
  // The SM version.
  int mSM;
  // The class that stores all the kernels.
  TllmGenFmhaKernel const* mKernel;
};
