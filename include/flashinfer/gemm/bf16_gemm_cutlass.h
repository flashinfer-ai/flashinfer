/*
 * Copyright (c) 2025, FlashInfer.
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

#ifndef FLASHINFER_BF16_GEMM_CUTLASS_H_
#define FLASHINFER_BF16_GEMM_CUTLASS_H_

#include <cuda_runtime_api.h>

#include <vector>

#include "flashinfer/gemm/cutlass_gemm_configs.h"

namespace flashinfer {
namespace gemm {

class CutlassBf16GemmRunnerInterface {
 public:
  CutlassBf16GemmRunnerInterface() = default;
  virtual ~CutlassBf16GemmRunnerInterface() = default;

  virtual void gemm(__nv_bfloat16 const* A, __nv_bfloat16 const* B, void* D, int m, int n, int k,
                    int b, CutlassGemmConfig gemmConfig, char* workspacePtr,
                    size_t const workspaceBytes, cudaStream_t stream) = 0;

  virtual size_t getWorkspaceSize(int m, int n, int k) = 0;

  virtual std::vector<CutlassGemmConfig> getConfigs() const = 0;
};

template <typename T>
class CutlassBf16GemmRunner : public CutlassBf16GemmRunnerInterface {
 public:
  CutlassBf16GemmRunner() = default;
  ~CutlassBf16GemmRunner() = default;

  void gemm(__nv_bfloat16 const* A, __nv_bfloat16 const* B, void* D, int m, int n, int k, int b,
            CutlassGemmConfig gemmConfig, char* workspacePtr, size_t const workspaceBytes,
            cudaStream_t stream) override;
  size_t getWorkspaceSize(int m, int n, int k) override;
  std::vector<CutlassGemmConfig> getConfigs() const override;

 private:
  size_t getWorkspaceSizeImpl(int m, int n, int k);
};

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_BF16_GEMM_CUTLASS_H_
