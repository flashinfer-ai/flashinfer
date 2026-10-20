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
#ifndef FLASHINFER_CSRC_W8A16_FP8_GEMM_CHECKS_H_
#define FLASHINFER_CSRC_W8A16_FP8_GEMM_CHECKS_H_

// Argument validation shared by the Hopper and Blackwell w8a16_fp8_gemm launchers
// (csrc/w8a16_fp8_gemm.cu, csrc/w8a16_fp8_gemm_sm100.cu).  Both launchers expose the same
// contract; only the variant codes and the kernels behind them differ.
//
//   out[M, N] = act[M, K] @ (fp8_weight[N, K] * scale).T (+ bias[N])
//
// act [M, K] fp16/bf16, weight [N, K] fp8 e4m3, scale one fp32 value, bias [N] in act's
// dtype, out [M, N] in act's dtype; everything contiguous, on one device, 16-byte aligned.
// The workspace is a byte buffer of any dtype (the kernels take a void* and a byte count).
// `out` and the workspace are written while the inputs are still being read (other tiles,
// the split-K reduction), so neither may share bytes with act / weight / scale / bias, nor
// with each other; the inputs may alias one another freely.

#include <cstdint>

#include "tvm_ffi_utils.h"

// 16-byte base-address check (TMA descriptors, 16-byte vector loads / stores); CHECK_INPUT
// covers device and contiguity only.  A contiguous view at an odd storage offset is not
// aligned, and a misaligned pointer faults the CUDA context instead of this call.
#define W8A16_CHECK_ALIGNED(x)                                          \
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>((x).data_ptr()) % 16 == 0) \
      << #x " must start at a 16-byte aligned address (storage offset of a view)"

// [begin, end) of the bytes a contiguous tensor covers (none for an empty one).
struct W8A16ByteSpan {
  uintptr_t begin = 0, end = 0;
};
inline W8A16ByteSpan W8A16SpanOf(const TensorView& t) {
  W8A16ByteSpan s;
  if (t.numel() == 0) return s;
  s.begin = reinterpret_cast<uintptr_t>(t.data_ptr());
  s.end = s.begin + static_cast<uintptr_t>(t.numel()) * (t.dtype().bits / 8);
  return s;
}
inline void W8A16CheckDisjoint(const char* written, const W8A16ByteSpan& w, const char* read,
                               const W8A16ByteSpan& r) {
  TVM_FFI_ICHECK(w.begin >= w.end || r.begin >= r.end || w.begin >= r.end || r.begin >= w.end)
      << written << " must not overlap " << read << " in memory (" << written
      << " is written while " << read << " is still being read)";
}

// The validated operands of one call, as the kernels take them.
struct W8A16Operands {
  int M, N, K;
  const void* bias = nullptr;  // nullptr: no bias
  void* workspace = nullptr;   // nullptr: none given (fine when the variant needs 0 bytes)
  size_t workspace_bytes = 0;
  int device_id = 0;
  int num_sms = 0;
};

inline int W8A16NumSms(int device_id) {
  int num_sms = 0;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);
  return num_sms;
}

// Shape rules the workspace query shares with the call.  M = 0 is handled by the Python
// caller (empty output, no launch); here the tile schedulers divide by the tile counts, so
// every dimension must be positive, and the kernels index with 32-bit ints.
inline void W8A16CheckDims(int64_t M, int64_t N, int64_t K) {
  TVM_FFI_ICHECK(M > 0 && N > 0 && K > 0 && M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX)
      << "M, N, K must be positive 32-bit values, got " << M << " " << N << " " << K;
  TVM_FFI_ICHECK(K % 16 == 0) << "K must be a multiple of 16 (16-byte TMA rows of fp8), got " << K;
  TVM_FFI_ICHECK(N % 8 == 0) << "N must be a multiple of 8 (16-byte output rows), got " << N;
}

// Validates every tensor argument of w8a16_fp8_gemm against the contract above.
// `act_dtype_code` is the DLPack code the module was built for (W8A16_ACT_DTYPE_CODE).
inline W8A16Operands W8A16CheckOperands(const TensorView& out, const TensorView& act,
                                        const TensorView& weight, const TensorView& scale,
                                        const ffi::Optional<TensorView>& bias,
                                        const ffi::Optional<TensorView>& workspace,
                                        int act_dtype_code) {
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
  TVM_FFI_ICHECK(encode_dlpack_dtype(act.dtype()) == act_dtype_code)
      << "this w8a16_fp8_gemm module was built for another activation dtype";
  TVM_FFI_ICHECK(out.dtype() == act.dtype()) << "out must have the activation dtype";
  W8A16CheckDims(M, N, K);
  W8A16_CHECK_ALIGNED(act);
  W8A16_CHECK_ALIGNED(weight);
  W8A16_CHECK_ALIGNED(out);

  W8A16Operands ops;
  ops.M = static_cast<int>(M);
  ops.N = static_cast<int>(N);
  ops.K = static_cast<int>(K);
  if (bias.has_value()) {
    const auto& b = bias.value();
    CHECK_INPUT(b);
    CHECK_DEVICE(act, b);
    CHECK_DIM(1, b);
    TVM_FFI_ICHECK_EQ(b.size(0), N) << "bias must be [N]";
    TVM_FFI_ICHECK(b.dtype() == act.dtype()) << "bias must have the activation dtype";
    W8A16_CHECK_ALIGNED(b);
    ops.bias = b.data_ptr();
  }
  if (workspace.has_value()) {
    const auto& wsv = workspace.value();
    CHECK_INPUT(wsv);
    CHECK_DEVICE(act, wsv);
    W8A16_CHECK_ALIGNED(wsv);
    ops.workspace = wsv.data_ptr();
    ops.workspace_bytes = static_cast<size_t>(wsv.numel()) * (wsv.dtype().bits / 8);
  }
  // Aliasing: the two buffers the call writes against everything it reads.
  const W8A16ByteSpan out_span = W8A16SpanOf(out);
  const W8A16ByteSpan ws_span =
      workspace.has_value() ? W8A16SpanOf(workspace.value()) : W8A16ByteSpan{};
  W8A16CheckDisjoint("out", out_span, "act", W8A16SpanOf(act));
  W8A16CheckDisjoint("out", out_span, "weight", W8A16SpanOf(weight));
  W8A16CheckDisjoint("out", out_span, "scale", W8A16SpanOf(scale));
  W8A16CheckDisjoint("workspace", ws_span, "act", W8A16SpanOf(act));
  W8A16CheckDisjoint("workspace", ws_span, "weight", W8A16SpanOf(weight));
  W8A16CheckDisjoint("workspace", ws_span, "scale", W8A16SpanOf(scale));
  W8A16CheckDisjoint("workspace", ws_span, "out", out_span);
  if (bias.has_value()) {
    W8A16CheckDisjoint("out", out_span, "bias", W8A16SpanOf(bias.value()));
    W8A16CheckDisjoint("workspace", ws_span, "bias", W8A16SpanOf(bias.value()));
  }
  ops.device_id = act.device().device_id;
  ops.num_sms = W8A16NumSms(ops.device_id);
  return ops;
}

// Turns the status of W8A16Sm90Resolve / W8A16Sm100Resolve into an error with the reason,
// before any workspace is sized or anything is launched.  `first_code`..`max_code` is the
// range of variant codes, `slice_rule` says when the split-K variant can slice K.
inline void W8A16CheckPlan(cudaError_t status, int64_t algo, int first_code, int max_code,
                           int64_t K, const char* slice_rule) {
  TVM_FFI_ICHECK(status != cudaErrorInvalidValue)
      << "algo must be 0 (auto) or a variant code " << first_code << ".." << max_code << ", got "
      << algo;
  TVM_FFI_ICHECK(status != cudaErrorNotSupported)
      << "the split-K variant (algo " << algo << ") cannot slice K=" << K << ": " << slice_rule;
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "w8a16_fp8_gemm could not resolve a variant: " << cudaGetErrorString(status);
}

// Turns the status of W8A16Fp8GemmSm90WorkspaceBytes / W8A16Fp8GemmSm100WorkspaceBytes into
// an error: the only failure is a byte count that leaves the signed 64-bit range the query
// returns in (`splits` x M x N fp32 partials of a forced split-K at extreme dimensions).
inline void W8A16CheckWorkspaceBytes(cudaError_t status, int64_t M, int64_t N, int splits) {
  TVM_FFI_ICHECK(status == cudaSuccess) << "the workspace for M=" << M << ", N=" << N << " in "
                                        << splits << " K-slices exceeds the 64-bit byte range";
}

#endif  // FLASHINFER_CSRC_W8A16_FP8_GEMM_CHECKS_H_
