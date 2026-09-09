/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include "flashinfer/gemm/nvfp4_smooth_quantize_sm100.cuh"
#if defined(FLASHINFER_ENABLE_SVDQ_SM120_K1_K2_FUSION)
#include "flashinfer/gemm/nvfp4_smooth_quantize_lora_down_sm120.cuh"
#endif
#include "tvm_ffi_utils.h"

namespace torch_ext {

namespace {

int32_t getMultiProcessorCount(int32_t device_index) {
  static thread_local int32_t cached_multi_processor_count = -1;
  static thread_local int32_t cached_device_index = -1;

  if (device_index == cached_device_index && cached_multi_processor_count != -1) {
    return cached_multi_processor_count;
  }
  int32_t count;
  cudaError_t cudaStatus =
      cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device_index);
  TVM_FFI_ICHECK(cudaStatus == cudaSuccess)
      << "Failed to get device attribute: " << cudaGetErrorString(cudaStatus);
  cached_multi_processor_count = count;
  cached_device_index = device_index;
  return count;
}

}  // namespace

// (xq, sf) = NVFP4-quantize(x * pqs) in one pass: the per-input-channel pre_quant_scale smoothing
// is fused into the quantize -- byte-identical to a separate x_hat = x*s elementwise pass followed
// by fp4_quantize. x [m, n] bf16, pqs [n] bf16, global_scale f32 (numel >= 1, the per-tensor
// scale). Outputs are caller-allocated: xq [m, n/2] uint8 (packed e2m1), sf uint8 with numel >=
// ceil(m/128)*128 * ceil((n/16)/4)*4 (swizzled 128x4 UE4M3 block scales, vec size 16). SM100+.
void nvfp4_quantize_smooth(TensorView x, TensorView pqs, TensorView global_scale, TensorView xq,
                           TensorView sf, bool enable_pdl) {
  CHECK_INPUT_AND_TYPE(x, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(pqs, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(global_scale, dl_float32);
  CHECK_INPUT_AND_TYPE(xq, dl_uint8);
  CHECK_INPUT_AND_TYPE(sf, dl_uint8);
  CHECK_DEVICE(pqs, x);
  CHECK_DEVICE(global_scale, x);
  CHECK_DEVICE(xq, x);
  CHECK_DEVICE(sf, x);

  TVM_FFI_ICHECK_EQ(x.ndim(), 2) << "x must be [m, n]";
  int const m = static_cast<int>(x.size(0));
  int const n = static_cast<int>(x.size(1));
  TVM_FFI_ICHECK_EQ(n % 16, 0) << "n must be divisible by 16 (NVFP4 SF vector size)";
  TVM_FFI_ICHECK_EQ(pqs.numel(), n) << "pqs must have n elements";
  TVM_FFI_ICHECK_GE(global_scale.numel(), 1) << "global_scale must contain at least one element";

  TVM_FFI_ICHECK_EQ(xq.ndim(), 2) << "xq must be [m, n/2]";
  TVM_FFI_ICHECK_EQ(xq.size(0), m) << "xq must be [m, n/2]";
  TVM_FFI_ICHECK_EQ(xq.size(1), n / 2) << "xq must be [m, n/2]";
  // Swizzled 128x4 SF layout: rows padded to a multiple of 128, SF columns (n/16) to a multiple
  // of 4.
  int64_t const sfSize =
      static_cast<int64_t>((m + 128 - 1) / 128 * 128) * ((n / 16 + 4 - 1) / 4 * 4);
  TVM_FFI_ICHECK_GE(sf.numel(), sfSize)
      << "sf is smaller than the required swizzled scale layout (" << sfSize << " bytes)";

  int const smCount = getMultiProcessorCount(x.device().device_id);
  auto stream = get_stream(x.device());
  flashinfer::gemm::nvfp4_smooth_quantize(
      xq.data_ptr(), sf.data_ptr(), x.data_ptr(), pqs.data_ptr(),
      static_cast<float const*>(global_scale.data_ptr()), m, n, smCount, stream, enable_pdl);
}

#if defined(FLASHINFER_ENABLE_SVDQ_SM120_K1_K2_FUSION)
void nvfp4_quantize_smooth_lora_down_sm120_impl(TensorView x, TensorView pqs,
                                                TensorView global_scale, TensorView l2t_smoothed,
                                                TensorView xq, TensorView sf, TensorView down,
                                                bool signal_m537_quant_ready,
                                                int geometry_variant = 0) {
  CHECK_INPUT_AND_TYPE(x, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(pqs, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(global_scale, dl_float32);
  CHECK_INPUT_AND_TYPE(l2t_smoothed, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(xq, dl_uint8);
  CHECK_INPUT_AND_TYPE(sf, dl_uint8);
  CHECK_INPUT_AND_TYPE(down, dl_bfloat16);
  CHECK_DEVICE(pqs, x);
  CHECK_DEVICE(global_scale, x);
  CHECK_DEVICE(l2t_smoothed, x);
  CHECK_DEVICE(xq, x);
  CHECK_DEVICE(sf, x);
  CHECK_DEVICE(down, x);

  TVM_FFI_ICHECK_EQ(x.ndim(), 2) << "x must be 2-D [m, k]";
  int const m = static_cast<int>(x.size(0));
  int const k = static_cast<int>(x.size(1));
  // Was a list of ~51 exact (M, K) pairs, kept in step by hand with the producer
  // dispatcher and with SM120_FUSED_LINEAR_MK on the Python side. Those three
  // copies drifted apart once, which is why the list existed -- but the drift
  // was a consequence of writing the same fact down three times.
  //
  // The producer decides now. Every launcher validates its own preconditions and
  // refuses with cudaErrorInvalidValue, and the status check below turns that
  // into an error naming the shape. A shape nobody measured either runs or is
  // refused by the thing that would have had to run it, rather than by a list of
  // where the measurements happened to be taken.
  TVM_FFI_ICHECK_EQ(pqs.numel(), k) << "pqs must have k elements";
  TVM_FFI_ICHECK_GE(global_scale.numel(), 1) << "global_scale must contain at least one element";
  TVM_FFI_ICHECK_EQ(l2t_smoothed.ndim(), 2) << "l2t_smoothed must be [k, 32]";
  TVM_FFI_ICHECK_EQ(l2t_smoothed.size(0), k) << "l2t_smoothed must be [k, 32]";
  TVM_FFI_ICHECK_EQ(l2t_smoothed.size(1), 32) << "l2t_smoothed must be [k, 32]";
  TVM_FFI_ICHECK_EQ(xq.ndim(), 2) << "xq must be [m, k/2]";
  TVM_FFI_ICHECK_EQ(xq.size(0), m) << "xq must be [m, k/2]";
  TVM_FFI_ICHECK_EQ(xq.size(1), k / 2) << "xq must be [m, k/2]";
  int64_t const sfSize = static_cast<int64_t>((m + 127) / 128 * 128) * (k / 16);
  TVM_FFI_ICHECK_GE(sf.numel(), sfSize)
      << "sf is smaller than the required swizzled scale layout (" << sfSize << " bytes)";
  TVM_FFI_ICHECK_EQ(down.ndim(), 2) << "down must be [m, 32]";
  TVM_FFI_ICHECK_EQ(down.size(0), m) << "down must be [m, 32]";
  TVM_FFI_ICHECK_EQ(down.size(1), 32) << "down must be [m, 32]";

  auto stream = get_stream(x.device());
  cudaError_t const status = flashinfer::gemm::nvfp4_smooth_quantize_lora_down_sm120(
      x.data_ptr(), pqs.data_ptr(), static_cast<float const*>(global_scale.data_ptr()),
      xq.data_ptr(), sf.data_ptr(), l2t_smoothed.data_ptr(), down.data_ptr(), m, k, stream,
      signal_m537_quant_ready, geometry_variant);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "failed to launch SM120 fused smooth-quantize + LoRA-down kernel: "
      << cudaGetErrorString(status);
}

void nvfp4_quantize_smooth_lora_down_sm120(TensorView x, TensorView pqs, TensorView global_scale,
                                           TensorView l2t_smoothed, TensorView xq, TensorView sf,
                                           TensorView down) {
  nvfp4_quantize_smooth_lora_down_sm120_impl(x, pqs, global_scale, l2t_smoothed, xq, sf, down,
                                             false);
}

// Geometry-selecting entry for the fused route.
//
// Deliberately separate from the seven-argument export above rather than an
// added parameter on it: that symbol is bound by TVM FFI and called from Python
// with exactly seven arguments, so widening it would change an ABI that tests
// and the diagnostic probe both depend on. This one is internal -- only
// dispatch_svdquant_producer_sm120 calls it -- so the export stays as it was.
void nvfp4_quantize_smooth_lora_down_geometry_sm120(TensorView x, TensorView pqs,
                                                    TensorView global_scale,
                                                    TensorView l2t_smoothed, TensorView xq,
                                                    TensorView sf, TensorView down,
                                                    int geometry_variant) {
  nvfp4_quantize_smooth_lora_down_sm120_impl(x, pqs, global_scale, l2t_smoothed, xq, sf, down,
                                             false, geometry_variant);
}

// Runtime-M/K entry, for measuring what the templated shape specialisation is
// worth. Takes the geometry explicitly because there is no per-shape pin to
// read it from -- that is the whole point of the variant.
void nvfp4_quantize_smooth_lora_down_dyn_sm120(TensorView x, TensorView pqs,
                                               TensorView global_scale, TensorView l2t_smoothed,
                                               TensorView xq, TensorView sf, TensorView down,
                                               int family, int tiling0, int tiling1, int tiling2,
                                               int address_policy) {
  CHECK_INPUT_AND_TYPE(x, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(pqs, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(global_scale, dl_float32);
  CHECK_INPUT_AND_TYPE(l2t_smoothed, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(xq, dl_uint8);
  CHECK_INPUT_AND_TYPE(sf, dl_uint8);
  CHECK_INPUT_AND_TYPE(down, dl_bfloat16);
  int const m = static_cast<int>(x.size(0));
  int const k = static_cast<int>(x.size(1));
  cudaStream_t stream = get_stream(x.device());
  cudaError_t status = flashinfer::gemm::nvfp4_smooth_quantize_lora_down_family_sm120(
      x.data_ptr(), pqs.data_ptr(), static_cast<float const*>(global_scale.data_ptr()),
      xq.data_ptr(), sf.data_ptr(), l2t_smoothed.data_ptr(), down.data_ptr(), m, k, family, tiling0,
      tiling1, tiling2, address_policy, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "nvfp4_quantize_smooth_lora_down_dyn_sm120 failed: " << cudaGetErrorString(status);
}

void nvfp4_quantize_smooth_lora_down_overlap_sm120(TensorView x, TensorView pqs,
                                                   TensorView global_scale, TensorView l2t_smoothed,
                                                   TensorView xq, TensorView sf, TensorView down) {
  nvfp4_quantize_smooth_lora_down_sm120_impl(x, pqs, global_scale, l2t_smoothed, xq, sf, down,
                                             true);
}
// Direct-only diagnostic entry for the accepted n16 geometry of the (512,
// 3072) native K12 producer. The symbol used by the acceptance A/B trace is
// retained for focused diagnostics, while rows 0 and 28 reach the same geometry
// through the shared production entry and nvfp4_svdquant_linear_sm120. No
// Python routing, tactic table, route ABI, JIT source list or combined C++ path
// references this diagnostic symbol.
//
// The layout contract below repeats the production entry's checks rather than
// refactoring them into a shared helper, so the production validation and its
// error strings stay byte-for-byte unchanged.
void nvfp4_quantize_smooth_lora_down_m512_k3072_n16_experimental_sm120(TensorView x, TensorView pqs,
                                                                       TensorView global_scale,
                                                                       TensorView l2t_smoothed,
                                                                       TensorView xq, TensorView sf,
                                                                       TensorView down) {
  CHECK_INPUT_AND_TYPE(x, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(pqs, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(global_scale, dl_float32);
  CHECK_INPUT_AND_TYPE(l2t_smoothed, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(xq, dl_uint8);
  CHECK_INPUT_AND_TYPE(sf, dl_uint8);
  CHECK_INPUT_AND_TYPE(down, dl_bfloat16);
  CHECK_DEVICE(pqs, x);
  CHECK_DEVICE(global_scale, x);
  CHECK_DEVICE(l2t_smoothed, x);
  CHECK_DEVICE(xq, x);
  CHECK_DEVICE(sf, x);
  CHECK_DEVICE(down, x);

  // Exactly the diagnostic shape and nothing else -- in particular not the
  // (64, 3072) and (27280, 3072) neighbours the production entry admits, whose
  // producers are entirely different kernels.
  constexpr int kDiagnosticM = 512;
  constexpr int kDiagnosticK = 3072;
  constexpr int kDiagnosticRank = 32;
  TVM_FFI_ICHECK_EQ(x.ndim(), 2)
      << "x must be exactly the experimental SM120 (512, 3072) n16 K12 shape";
  int const m = static_cast<int>(x.size(0));
  int const k = static_cast<int>(x.size(1));
  TVM_FFI_ICHECK(m == kDiagnosticM && k == kDiagnosticK)
      << "x must be exactly the experimental SM120 (512, 3072) n16 K12 shape";
  TVM_FFI_ICHECK_EQ(pqs.numel(), k) << "pqs must have k elements";
  TVM_FFI_ICHECK_GE(global_scale.numel(), 1) << "global_scale must contain at least one element";
  TVM_FFI_ICHECK_EQ(l2t_smoothed.ndim(), 2) << "l2t_smoothed must be [k, 32]";
  TVM_FFI_ICHECK_EQ(l2t_smoothed.size(0), k) << "l2t_smoothed must be [k, 32]";
  TVM_FFI_ICHECK_EQ(l2t_smoothed.size(1), kDiagnosticRank) << "l2t_smoothed must be [k, 32]";
  TVM_FFI_ICHECK_EQ(xq.ndim(), 2) << "xq must be [m, k/2]";
  TVM_FFI_ICHECK_EQ(xq.size(0), m) << "xq must be [m, k/2]";
  TVM_FFI_ICHECK_EQ(xq.size(1), k / 2) << "xq must be [m, k/2]";
  int64_t const sfSize = static_cast<int64_t>((m + 127) / 128 * 128) * (k / 16);
  TVM_FFI_ICHECK_GE(sf.numel(), sfSize)
      << "sf is smaller than the required swizzled scale layout (" << sfSize << " bytes)";
  TVM_FFI_ICHECK_EQ(down.ndim(), 2) << "down must be [m, 32]";
  TVM_FFI_ICHECK_EQ(down.size(0), m) << "down must be [m, 32]";
  TVM_FFI_ICHECK_EQ(down.size(1), kDiagnosticRank) << "down must be [m, 32]";

  auto stream = get_stream(x.device());
  cudaError_t const status =
      flashinfer::gemm::nvfp4_smooth_quantize_lora_down_m512_k3072_n16_experimental_sm120(
          x.data_ptr(), pqs.data_ptr(), static_cast<float const*>(global_scale.data_ptr()),
          xq.data_ptr(), sf.data_ptr(), l2t_smoothed.data_ptr(), down.data_ptr(), m, k, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "failed to launch the experimental SM120 M512/K3072 n16 fused "
         "smooth-quantize + LoRA-down kernel: "
      << cudaGetErrorString(status);
}
#endif

}  // namespace torch_ext

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_quantize_smooth, torch_ext::nvfp4_quantize_smooth);
#if defined(FLASHINFER_ENABLE_SVDQ_SM120_K1_K2_FUSION)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_quantize_smooth_lora_down_dyn_sm120,
                              torch_ext::nvfp4_quantize_smooth_lora_down_dyn_sm120);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_quantize_smooth_lora_down_sm120,
                              torch_ext::nvfp4_quantize_smooth_lora_down_sm120);
// Diagnostic-only export; deliberately not referenced by any production path.
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    nvfp4_quantize_smooth_lora_down_m512_k3072_n16_experimental_sm120,
    torch_ext::nvfp4_quantize_smooth_lora_down_m512_k3072_n16_experimental_sm120);
#endif
