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
#include "w8a16_fp8_gemm_config.inc"

using namespace flashinfer;

// 16-byte base-address check (TMA descriptors, 16-byte vector loads / stores); CHECK_INPUT
// covers device and contiguity only.
#define W8A16_CHECK_ALIGNED(x)                                          \
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>((x).data_ptr()) % 16 == 0) \
      << #x " must start at a 16-byte aligned address (storage offset of a view)"

// Blackwell (sm_100a) launcher: same contract as w8a16_fp8_gemm (csrc/w8a16_fp8_gemm.cu).
// out[M, N] = act[M, K] @ (fp8_weight[N, K] * scale).T (+ bias[N])
// `algo` is the tile code (0 = cost model, 1.. = W8A16Sm100Tile); see
// flashinfer.gemm.w8a16_fp8_gemm's `algo` argument.
// The activation dtype is fixed per module (W8A16ActType, rendered from the Jinja config);
// the Python side loads the module matching act.dtype.
void w8a16_fp8_gemm(TensorView out, TensorView act, TensorView weight, TensorView scale,
                    ffi::Optional<TensorView> bias, ffi::Optional<TensorView> workspace,
                    int64_t algo) {
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
  TVM_FFI_ICHECK(algo >= 0 && algo <= gemm::kW8A16Sm100MaxTileCode)
      << "algo must be 0 (auto) or a tile code 1.." << gemm::kW8A16Sm100MaxTileCode;

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
  TVM_FFI_ICHECK(encode_dlpack_dtype(act.dtype()) == W8A16_ACT_DTYPE_CODE)
      << "this w8a16_fp8_gemm module was built for another activation dtype";
  TVM_FFI_ICHECK(out.dtype() == act.dtype()) << "out must have the activation dtype";
  TVM_FFI_ICHECK(M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX);
  // M = 0 is handled by the Python caller (empty output, no launch); here the tile
  // schedulers divide by the tile counts, so every dimension must be positive.
  TVM_FFI_ICHECK(M > 0 && N > 0 && K > 0)
      << "M, N, K must be positive, got " << M << " " << N << " " << K;
  TVM_FFI_ICHECK(K % 16 == 0) << "K must be a multiple of 16 (16-byte TMA rows of fp8)";
  TVM_FFI_ICHECK(N % 8 == 0) << "N must be a multiple of 8 (16-byte output rows)";
  // TMA descriptors and the 16-byte vector loads / stores (epilogue, split-K reduce)
  // need 16-byte aligned base addresses; a contiguous view at an odd storage offset
  // is not, and a misaligned pointer faults the CUDA context instead of this call.
  W8A16_CHECK_ALIGNED(act);
  W8A16_CHECK_ALIGNED(weight);
  W8A16_CHECK_ALIGNED(out);

  const void* bias_ptr = nullptr;
  if (bias.has_value()) {
    const auto& b = bias.value();
    CHECK_INPUT(b);
    CHECK_DEVICE(act, b);
    CHECK_DIM(1, b);
    TVM_FFI_ICHECK_EQ(b.size(0), N) << "bias must be [N]";
    TVM_FFI_ICHECK(b.dtype() == act.dtype()) << "bias must have the activation dtype";
    W8A16_CHECK_ALIGNED(b);
    bias_ptr = b.data_ptr();
  }

  void* ws_ptr = nullptr;
  size_t ws_bytes = 0;
  if (workspace.has_value()) {
    const auto& wsv = workspace.value();
    CHECK_INPUT(wsv);
    CHECK_DEVICE(act, wsv);
    W8A16_CHECK_ALIGNED(wsv);
    ws_ptr = wsv.data_ptr();
    ws_bytes = static_cast<size_t>(wsv.numel()) * (wsv.dtype().bits / 8);
  }
  const int device_id = act.device().device_id;
  int num_sms = 0;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);

  ffi::CUDADeviceGuard device_guard(device_id);
  const cudaStream_t stream = get_stream(act.device());

  const cudaError_t status = gemm::W8A16Fp8GemmSm100<W8A16ActType>(
      static_cast<const W8A16ActType*>(act.data_ptr()),
      static_cast<const __nv_fp8_e4m3*>(weight.data_ptr()),
      static_cast<const float*>(scale.data_ptr()), static_cast<const W8A16ActType*>(bias_ptr),
      static_cast<W8A16ActType*>(out.data_ptr()), static_cast<int>(M), static_cast<int>(N),
      static_cast<int>(K), ws_ptr, ws_bytes, device_id, num_sms, stream, static_cast<int>(algo));
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "w8a16_fp8_gemm (sm100) failed with error code " << cudaGetErrorString(status);
}

// Workspace (bytes) the CUTLASS kernel needs for this problem; 0 for the persistent scheduler.
int64_t w8a16_fp8_gemm_workspace_bytes(int64_t M, int64_t N, int64_t K, int64_t device_id,
                                       int64_t algo) {
  TVM_FFI_ICHECK(M > 0 && N > 0 && K > 0 && M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX)
      << "M, N, K must be positive 32-bit values, got " << M << " " << N << " " << K;
  int num_sms = 0;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, static_cast<int>(device_id));
  // Both activation dtypes share the mainloop/epilogue layout, so the answer is the same
  // from either dtype's module.
  return static_cast<int64_t>(gemm::W8A16Fp8GemmSm100WorkspaceBytes<W8A16ActType>(
      static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), static_cast<int>(device_id),
      num_sms, static_cast<int>(algo)));
}
