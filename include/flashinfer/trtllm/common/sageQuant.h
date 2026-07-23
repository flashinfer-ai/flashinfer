/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
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

#include <cuda.h>
#include <cuda_runtime.h>
#include <flashinfer/trtllm/common.h>

namespace flashinfer::trtllm {

struct SageQuantParams {
  // Required arguments for SageQuantQk (Q or K).
  int sumSeqLensQk{};
  int numHeads{};
  int headDim{};
  int tokenBlockSize{};
  bool kSmooth{false};
  void const* ptrQk{nullptr};
  void* ptrQkQuant{nullptr};
  Data_type inputType{DATA_TYPE_FP16};
  Data_type quantType{DATA_TYPE_E4M3};
  float* ptrQkScale{nullptr};
  float* ptrKMean{nullptr};
  // Optional arguments for SageQuantV.
  // vStage: 0: disabled, 1: collect scales, 2: quantize.
  int vStage{};
  int sumSeqLensV{};
  int numHeadsV{};
  void const* ptrV{nullptr};
  void* ptrVQuant{nullptr};
  float* ptrVScale{nullptr};
  // Hardware information.
  int smCount{};
  cudaStream_t stream{};
};

void invokeSageQuant(SageQuantParams const& params);

}  // namespace flashinfer::trtllm
