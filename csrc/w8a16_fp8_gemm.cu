/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/gemm/w8a16_fp8_gemm_sm90.cuh>

#include "tvm_ffi_utils.h"
#include "w8a16_fp8_gemm_checks.h"
#include "w8a16_fp8_gemm_config.inc"

using namespace flashinfer;

namespace {

// Hopper (sm_90a): variant codes 2..W8A16Sm90MaxAlgoCode (code 1 is reserved).
struct W8A16Launcher {
  using Plan = gemm::W8A16Sm90Plan;
  static constexpr const char* Name = "w8a16_fp8_gemm";
  static constexpr int FirstAlgoCode = 2;
  static constexpr int MaxAlgoCode = gemm::W8A16Sm90MaxAlgoCode;
  static constexpr const char* SliceRule =
      "K must be a multiple of 64 whose k-tile count (K / 64) divides into 2..8 equal slices of "
      "at least 8 k-tiles";
  static cudaError_t Resolve(int M, int N, int K, int num_sms, int64_t algo, Plan* plan) {
    return gemm::W8A16Sm90Resolve(M, N, K, num_sms, algo, plan);
  }
  template <typename T>
  static cudaError_t WorkspaceBytes(const Plan& plan, int M, int N, int K, int device_id,
                                    int num_sms, size_t* bytes) {
    return gemm::W8A16Fp8GemmSm90WorkspaceBytes<T>(plan, M, N, K, device_id, num_sms, bytes);
  }
  template <typename T>
  static cudaError_t Run(const Plan& plan, const T* act, const __nv_fp8_e4m3* weight,
                         const float* scale, const T* bias, T* out, int M, int N, int K,
                         void* workspace, size_t workspace_bytes, int device_id, int num_sms,
                         cudaStream_t stream) {
    return gemm::W8A16Fp8GemmSm90<T>(plan, act, weight, scale, bias, out, M, N, K, workspace,
                                     workspace_bytes, device_id, num_sms, stream);
  }
};

}  // namespace

// The contract of both functions is in w8a16_fp8_gemm_checks.h; this module accepts
// activations of W8A16ActType only.
void w8a16_fp8_gemm(TensorView out, TensorView act, TensorView weight, TensorView scale,
                    ffi::Optional<TensorView> bias, ffi::Optional<TensorView> workspace,
                    int64_t algo) {
  W8A16LaunchGemm<W8A16Launcher, W8A16ActType>(out, act, weight, scale, bias, workspace, algo,
                                               W8A16_ACT_DTYPE_CODE);
}

int64_t w8a16_fp8_gemm_workspace_bytes(int64_t M, int64_t N, int64_t K, int64_t device_id,
                                       int64_t algo) {
  return W8A16QueryWorkspaceBytes<W8A16Launcher, W8A16ActType>(M, N, K, device_id, algo);
}
