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

#include "fmhaKernels.cuh"
#include "fmhaRunnerParams.h"

class TllmGenFmhaRunner {
 public:
  // Constructor.
  explicit TllmGenFmhaRunner(Data_type dtypeQ, Data_type dtypeKv, Data_type dtypeOut);

  TllmGenFmhaRunner() = default;

  // Check if fmha is supported.
  bool isSupported(TllmGenFmhaRunnerParams const& runnerParams) const;

  // Check if fmha is supported with additional info.
  std::pair<bool, std::string> isSupportedWithInfo(
      TllmGenFmhaRunnerParams const& runnerParams) const;

  // Run the fmha kernel.
  void run(TllmGenFmhaRunnerParams const&);

 private:
  // The input/output datatype.
  Data_type mDtypeQ, mDtypeKv, mDtypeOut;
  // The SM version.
  int mSM;
  // The class that stores all the kernels.
  TllmGenFmhaKernel const* mKernel;
};
