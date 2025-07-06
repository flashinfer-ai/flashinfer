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

#include <flashinfer/trtllm/fmha/fmhaRunner.cuh>

TllmGenFmhaRunner::TllmGenFmhaRunner(Data_type dtypeQ, Data_type dtypeKv, Data_type dtypeOut)
    : mSM(getSMVersion()), mDtypeQ(dtypeQ), mDtypeKv(dtypeKv), mDtypeOut(dtypeOut) {
  TORCH_CHECK(mSM == kSM_100, "Unsupported architecture");
  TORCH_CHECK(mDtypeQ == DATA_TYPE_E4M3 || mDtypeQ == DATA_TYPE_FP16 || mDtypeQ == DATA_TYPE_BF16,
              "Unsupported Q data type");
  TORCH_CHECK(
      mDtypeKv == DATA_TYPE_E4M3 || mDtypeKv == DATA_TYPE_FP16 || mDtypeKv == DATA_TYPE_BF16,
      "Unsupported Kv data type");
  TORCH_CHECK(
      mDtypeOut == DATA_TYPE_E4M3 || mDtypeOut == DATA_TYPE_FP16 || mDtypeOut == DATA_TYPE_BF16,
      "Unsupported Output data type");
  mKernel = getTllmFmhaKernels(mDtypeQ, mDtypeKv, mDtypeOut, mSM);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TllmGenFmhaRunner::run(TllmGenFmhaRunnerParams const& runnerParams) {
  mKernel->run(runnerParams);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TllmGenFmhaRunner::isSupported(TllmGenFmhaRunnerParams const& runnerParams) const {
  return mKernel->checkIfKernelExist(runnerParams).first;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::pair<bool, std::string> TllmGenFmhaRunner::isSupportedWithInfo(
    TllmGenFmhaRunnerParams const& runnerParams) const {
  return mKernel->checkIfKernelExist(runnerParams);
}
