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
#include <flashinfer/gemm/w8a16_fp8_gemm.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

// out[M, N] = act[M, K] @ (fp8_weight[N, K] * scale).T (+ bias[N])
// `algo` selects the kernel variant on Blackwell; the Hopper launcher keeps its own
// tile / split-K cost model (env overrides FLASHINFER_W8A16_FP8_TILE / _SPLITS) and
// only accepts 0 (auto) here.
void w8a16_fp8_gemm(TensorView out, TensorView act, TensorView weight, TensorView scale,
                    ffi::Optional<TensorView> bias, bool weight_prepared,
                    ffi::Optional<TensorView> workspace, int64_t algo) {
  TVM_FFI_ICHECK(algo == 0) << "algo is Blackwell-only; the Hopper launcher picks the tile";
  CHECK_INPUT(act);     // [M, K], fp16/bf16, contiguous
  CHECK_INPUT(weight);  // [N, K], fp8 e4m3, contiguous
  CHECK_INPUT(out);     // [M, N], act dtype, contiguous
  CHECK_INPUT(scale);   // [1], fp32
  CHECK_DEVICE(act, weight);
  CHECK_DEVICE(act, out);
  CHECK_DEVICE(act, scale);
  CHECK_DIM(2, act);
  CHECK_DIM(2, weight);
  CHECK_DIM(2, out);

  const int64_t M = act.size(0);
  const int64_t K = act.size(1);
  const int64_t N = weight.size(0);
  TVM_FFI_ICHECK_EQ(weight.size(1), K) << "weight must be [N, K] with K matching act";
  TVM_FFI_ICHECK_EQ(out.size(0), M);
  TVM_FFI_ICHECK_EQ(out.size(1), N);
  TVM_FFI_ICHECK(encode_dlpack_dtype(weight.dtype()) == float8_e4m3fn_code)
      << "weight must be float8_e4m3fn";
  TVM_FFI_ICHECK(encode_dlpack_dtype(scale.dtype()) == float32_code) << "scale must be float32";
  TVM_FFI_ICHECK_EQ(scale.numel(), 1) << "scale is one per-tensor value";
  TVM_FFI_ICHECK(out.dtype() == act.dtype()) << "out must have the activation dtype";
  TVM_FFI_ICHECK(M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX);

  const void* bias_ptr = nullptr;
  if (bias.has_value()) {
    const auto& b = bias.value();
    CHECK_INPUT(b);
    CHECK_DEVICE(act, b);
    CHECK_DIM(1, b);
    TVM_FFI_ICHECK_EQ(b.size(0), N) << "bias must be [N]";
    TVM_FFI_ICHECK(b.dtype() == act.dtype()) << "bias must have the activation dtype";
    bias_ptr = b.data_ptr();
  }

  float* ws_ptr = nullptr;
  int64_t ws_bytes = 0;
  if (workspace.has_value()) {
    const auto& wsv = workspace.value();
    CHECK_INPUT(wsv);
    CHECK_DEVICE(act, wsv);
    TVM_FFI_ICHECK(encode_dlpack_dtype(wsv.dtype()) == float32_code) << "workspace must be float32";
    ws_ptr = static_cast<float*>(wsv.data_ptr());
    ws_bytes = wsv.numel() * 4;
  }
  int num_sms = 0;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, act.device().device_id);

  ffi::CUDADeviceGuard device_guard(act.device().device_id);
  const cudaStream_t stream = get_stream(act.device());

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(act.dtype(), c_type, [&] {
    cudaError_t status = gemm::W8A16Fp8Gemm<c_type>(
        static_cast<const c_type*>(act.data_ptr()),
        static_cast<const __nv_fp8_e4m3*>(weight.data_ptr()),
        static_cast<const float*>(scale.data_ptr()), static_cast<const c_type*>(bias_ptr),
        static_cast<c_type*>(out.data_ptr()), static_cast<int>(M), static_cast<int>(N),
        static_cast<int>(K), weight_prepared, ws_ptr, ws_bytes, num_sms, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "w8a16_fp8_gemm failed with error code " << cudaGetErrorString(status);
    return true;
  });
}

// Workspace (bytes) the split-K path needs for this problem; 0 when it does not split.
int64_t w8a16_fp8_gemm_workspace_bytes(int64_t M, int64_t N, int64_t K, int64_t device_id,
                                       int64_t algo) {
  TVM_FFI_ICHECK(algo == 0) << "algo is Blackwell-only; the Hopper launcher picks the tile";
  int num_sms = 0;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, static_cast<int>(device_id));
  return gemm::W8A16Fp8GemmWorkspaceBytes(static_cast<int>(M), static_cast<int>(N),
                                          static_cast<int>(K), num_sms);
}
