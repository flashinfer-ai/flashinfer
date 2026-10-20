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
#include <flashinfer/gemm/w8a16_fp8_gemm_sm100.cuh>

#include "tvm_ffi_utils.h"
#include "w8a16_fp8_gemm_checks.h"
#include "w8a16_fp8_gemm_config.inc"

using namespace flashinfer;

namespace {

constexpr int FirstAlgoCode = 1;
constexpr const char* SliceRule =
    "K must be a multiple of 128 whose k-tile count (K / 128) has a divisor in 2..8";

}  // namespace

// Blackwell (sm_100a) launcher; the contract is in w8a16_fp8_gemm_checks.h.
// `algo` is the variant code (0 = the launcher's cost model, 1.. = one variant; see
// W8A16Sm100Algo and flashinfer.gemm.w8a16_fp8_gemm's `algo` argument).
// The activation dtype is fixed per module (W8A16ActType, rendered from the Jinja config);
// the Python side loads the module matching act.dtype.
void w8a16_fp8_gemm(TensorView out, TensorView act, TensorView weight, TensorView scale,
                    ffi::Optional<TensorView> bias, ffi::Optional<TensorView> workspace,
                    int64_t algo) {
  const W8A16Operands ops =
      W8A16CheckOperands(out, act, weight, scale, bias, workspace, W8A16_ACT_DTYPE_CODE);
  gemm::W8A16Sm100Plan plan;
  W8A16CheckPlan(gemm::W8A16Sm100Resolve(ops.M, ops.N, ops.K, ops.num_sms, algo, &plan), algo,
                 FirstAlgoCode, gemm::W8A16Sm100MaxAlgoCode, ops.K, SliceRule);

  ffi::CUDADeviceGuard device_guard(ops.device_id);
  const cudaStream_t stream = get_stream(act.device());
  const cudaError_t status = gemm::W8A16Fp8GemmSm100<W8A16ActType>(
      plan, static_cast<const W8A16ActType*>(act.data_ptr()),
      static_cast<const __nv_fp8_e4m3*>(weight.data_ptr()),
      static_cast<const float*>(scale.data_ptr()), static_cast<const W8A16ActType*>(ops.bias),
      static_cast<W8A16ActType*>(out.data_ptr()), ops.M, ops.N, ops.K, ops.workspace,
      ops.workspace_bytes, ops.device_id, ops.num_sms, stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "w8a16_fp8_gemm (sm100) failed with error code " << cudaGetErrorString(status);
}

// Workspace (bytes) the chosen variant needs for this problem: the split-K fp32 partials
// and / or the CUTLASS scheduler workspace (stream-K); 0 when none.  Rejects the same
// requests the call would, so the caller learns before allocating anything.
int64_t w8a16_fp8_gemm_workspace_bytes(int64_t M, int64_t N, int64_t K, int64_t device_id,
                                       int64_t algo) {
  W8A16CheckDims(M, N, K);
  const int num_sms = W8A16NumSms(static_cast<int>(device_id));
  gemm::W8A16Sm100Plan plan;
  W8A16CheckPlan(gemm::W8A16Sm100Resolve(static_cast<int>(M), static_cast<int>(N),
                                         static_cast<int>(K), num_sms, algo, &plan),
                 algo, FirstAlgoCode, gemm::W8A16Sm100MaxAlgoCode, K, SliceRule);
  // Both activation dtypes share the mainloop / epilogue layout, so the answer is the same
  // from either dtype's module.
  size_t bytes = 0;
  W8A16CheckWorkspaceBytes(gemm::W8A16Fp8GemmSm100WorkspaceBytes<W8A16ActType>(
                               plan, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                               static_cast<int>(device_id), num_sms, &bytes),
                           M, N, plan.splits);
  return static_cast<int64_t>(bytes);
}
