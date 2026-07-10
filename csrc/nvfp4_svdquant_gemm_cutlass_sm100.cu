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
// TVM-FFI binding for the SM100 SVDQuant fused NVFP4 GEMM:
//   nvfp4_svdquant_gemm : out = alpha * (A @ Bᵀ) + (D @ L1ᵀ) [+ bias], the residual NVFP4 GEMM
//                         fused with the rank-r LoRA-up via a 2nd bf16 tcgen05 MMA in the same
//                         TMEM accumulator (custom CUTLASS SM100 block-scaled collective).
//                         r is inferred from d/l1 and must be a positive multiple of 32.
//   nvfp4_svdquant_gemm_tactic_num : number of kernel tactics for the autotuner.

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

#include "flashinfer/gemm/nvfp4_svdquant_gemm_cutlass.h"
#include "flashinfer/gemm/nvfp4_svdquant_gemm_template_sm100.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace flashinfer {
namespace gemm {

using svdquant_detail::KernelShape;
using svdquant_detail::resolve_tactic;
using svdquant_detail::RuntimeTactic;

size_t nvfp4_svdquant_gemm_workspace_size(int m, int n, int k, int tactic) {
  RuntimeTactic const runtime_tactic = resolve_tactic(tactic);
  switch (runtime_tactic.kernel_shape) {
    case KernelShape::k1Sm128x256x128:
      return svdquant_detail::workspace_size_for_tactic<
          svdquant_detail::Tactic1Sm128x256x128Config>(m, n, k, runtime_tactic);
    case KernelShape::k2Sm256x256x128:
      return svdquant_detail::workspace_size_for_tactic<
          svdquant_detail::Tactic2Sm256x256x128Config>(m, n, k, runtime_tactic);
    case KernelShape::k1Sm128x128x128:
      return svdquant_detail::workspace_size_for_tactic<
          svdquant_detail::Tactic1Sm128x128x128Config>(m, n, k, runtime_tactic);
    case KernelShape::k2Sm256x192x128:
      return svdquant_detail::workspace_size_for_tactic<
          svdquant_detail::Tactic2Sm256x192x128Config>(m, n, k, runtime_tactic);
    case KernelShape::k1Sm128x64x128:
      return svdquant_detail::workspace_size_for_tactic<svdquant_detail::Tactic1Sm128x64x128Config>(
          m, n, k, runtime_tactic);
    case KernelShape::k1Sm128x128x256:
      return svdquant_detail::workspace_size_for_tactic<
          svdquant_detail::Tactic1Sm128x128x256Config>(m, n, k, runtime_tactic);
    case KernelShape::k2Sm256x128x256:
      return svdquant_detail::workspace_size_for_tactic<
          svdquant_detail::Tactic2Sm256x128x256Config>(m, n, k, runtime_tactic);
    case KernelShape::k2Sm256x256x256:
      return svdquant_detail::workspace_size_for_tactic<
          svdquant_detail::Tactic2Sm256x256x256Config>(m, n, k, runtime_tactic);
  }
  throw std::invalid_argument("nvfp4_svdquant_gemm_workspace_size: invalid kernel shape");
}

// Fused residual NVFP4 GEMM + rank-r LoRA-up: D @ L1ᵀ via the 2nd bf16 tcgen05 MMA in the custom
// collective. 1/alpha folded into L1 so the epilogue yields alpha*residual + D@L1ᵀ + bias.
void nvfp4_svdquant_gemm_run(void* out, void const* A, void const* B, void const* sfa,
                             void const* sfb, float const* alpha, void const* D, void const* L1,
                             void const* bias, int m, int n, int k, int lora_rank, char* ws,
                             size_t wsBytes, cudaStream_t stream, int tactic, bool enable_pdl) {
  RuntimeTactic const runtime_tactic = resolve_tactic(tactic);
  switch (runtime_tactic.kernel_shape) {
    case KernelShape::k1Sm128x256x128:
      return svdquant_detail::run_tactic<svdquant_detail::Tactic1Sm128x256x128Config>(
          out, A, B, sfa, sfb, alpha, D, L1, bias, m, n, k, lora_rank, ws, wsBytes, stream,
          runtime_tactic, enable_pdl);
    case KernelShape::k2Sm256x256x128:
      return svdquant_detail::run_tactic<svdquant_detail::Tactic2Sm256x256x128Config>(
          out, A, B, sfa, sfb, alpha, D, L1, bias, m, n, k, lora_rank, ws, wsBytes, stream,
          runtime_tactic, enable_pdl);
    case KernelShape::k1Sm128x128x128:
      return svdquant_detail::run_tactic<svdquant_detail::Tactic1Sm128x128x128Config>(
          out, A, B, sfa, sfb, alpha, D, L1, bias, m, n, k, lora_rank, ws, wsBytes, stream,
          runtime_tactic, enable_pdl);
    case KernelShape::k2Sm256x192x128:
      return svdquant_detail::run_tactic<svdquant_detail::Tactic2Sm256x192x128Config>(
          out, A, B, sfa, sfb, alpha, D, L1, bias, m, n, k, lora_rank, ws, wsBytes, stream,
          runtime_tactic, enable_pdl);
    case KernelShape::k1Sm128x64x128:
      return svdquant_detail::run_tactic<svdquant_detail::Tactic1Sm128x64x128Config>(
          out, A, B, sfa, sfb, alpha, D, L1, bias, m, n, k, lora_rank, ws, wsBytes, stream,
          runtime_tactic, enable_pdl);
    case KernelShape::k1Sm128x128x256:
      return svdquant_detail::run_tactic<svdquant_detail::Tactic1Sm128x128x256Config>(
          out, A, B, sfa, sfb, alpha, D, L1, bias, m, n, k, lora_rank, ws, wsBytes, stream,
          runtime_tactic, enable_pdl);
    case KernelShape::k2Sm256x128x256:
      return svdquant_detail::run_tactic<svdquant_detail::Tactic2Sm256x128x256Config>(
          out, A, B, sfa, sfb, alpha, D, L1, bias, m, n, k, lora_rank, ws, wsBytes, stream,
          runtime_tactic, enable_pdl);
    case KernelShape::k2Sm256x256x256:
      return svdquant_detail::run_tactic<svdquant_detail::Tactic2Sm256x256x256Config>(
          out, A, B, sfa, sfb, alpha, D, L1, bias, m, n, k, lora_rank, ws, wsBytes, stream,
          runtime_tactic, enable_pdl);
  }
  throw std::invalid_argument("nvfp4_svdquant_gemm_run: invalid kernel shape");
}

}  // namespace gemm
}  // namespace flashinfer

namespace torch_ext {

namespace {

constexpr auto FLOAT4_E2M1X2 = dl_uint8;  // packed e2m1
constexpr auto SF_DTYPE = dl_uint8;       // swizzled ue4m3 block scales

// ceil(rows / 128) * 128 * ceil(cols / 4) * 4, the 128x4-swizzled block-scale layout size.
inline int64_t swizzled_sf_size(int64_t rows, int64_t sfCols) {
  auto pad = [](int64_t x, int64_t y) { return (x + y - 1) / y * y; };
  return pad(rows, 128) * pad(sfCols, 4);
}

}  // namespace

// out = alpha * (A @ Bᵀ) + (D @ L1ᵀ) [+ bias]. a = quant(x_hat) [m, k/2] uint8, b [n, k/2] uint8
// (packed e2m1), a_sf/b_sf swizzled UE4M3 block scales, alpha f32[1] (residual dequant scale).
// D [m, r] = x_hat @ L2ᵀ (bf16) and L1 [n, r] = svdquant_lora_b / alpha (bf16; 1/alpha folded so
// the epilogue out = alpha * acc yields the LoRA); the LoRA rank r is inferred from d/l1 and must
// be a positive multiple of 32. out [m, n] bf16, allocated by the caller.
void nvfp4_svdquant_gemm(TensorView a, TensorView b, TensorView a_sf, TensorView b_sf,
                         TensorView alpha, TensorView d, TensorView l1,
                         Optional<TensorView> const& bias, TensorView out,
                         TensorView workspace_buffer, int64_t tactic, bool enable_pdl) {
  CHECK_INPUT_AND_TYPE(a, FLOAT4_E2M1X2);
  CHECK_INPUT_AND_TYPE(b, FLOAT4_E2M1X2);
  CHECK_INPUT_AND_TYPE(a_sf, SF_DTYPE);
  CHECK_INPUT_AND_TYPE(b_sf, SF_DTYPE);
  CHECK_INPUT_AND_TYPE(alpha, dl_float32);
  CHECK_INPUT_AND_TYPE(d, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(l1, dl_bfloat16);
  CHECK_DEVICE(b, a);
  CHECK_DEVICE(a_sf, a);
  CHECK_DEVICE(b_sf, a);
  CHECK_DEVICE(alpha, a);
  CHECK_DEVICE(d, a);
  CHECK_DEVICE(l1, a);
  CHECK_DEVICE(out, a);
  CHECK_DEVICE(workspace_buffer, a);

  TVM_FFI_ICHECK_EQ(a.ndim(), 2) << "a must be [m, k/2]";
  TVM_FFI_ICHECK_EQ(b.ndim(), 2) << "b must be [n, k/2]";
  int64_t const m = a.size(0);
  int64_t const kPacked = a.size(1);
  int64_t const k = kPacked * 2;
  int64_t const n = b.size(0);
  TVM_FFI_ICHECK_EQ(b.size(1), kPacked) << "a and b inner dimensions mismatch";
  TVM_FFI_ICHECK(n > 0 && k > 0) << "n and k must be positive";
  TVM_FFI_ICHECK(n % 32 == 0 && k % 32 == 0) << "n and k must be divisible by 32";
  TVM_FFI_ICHECK_GE(alpha.numel(), 1) << "alpha must contain at least one element";
  TVM_FFI_ICHECK_GE(a_sf.numel(), swizzled_sf_size(m, k / 16))
      << "a_sf is smaller than the required swizzled scale layout";
  TVM_FFI_ICHECK_GE(b_sf.numel(), swizzled_sf_size(n, k / 16))
      << "b_sf is smaller than the required swizzled scale layout";
  TVM_FFI_ICHECK(d.ndim() == 2 && d.size(0) == m)
      << "d must have shape [m, r] (rank-r LoRA-down output)";
  int64_t const loraRank = d.size(1);
  TVM_FFI_ICHECK(loraRank >= 32 && loraRank % 32 == 0)
      << "the LoRA rank (d/l1 inner dimension) must be a positive multiple of 32, got " << loraRank;
  TVM_FFI_ICHECK_EQ(reinterpret_cast<std::uintptr_t>(d.data_ptr()) % 16, 0)
      << "d must be 16-byte aligned for TMA";
  TVM_FFI_ICHECK(l1.ndim() == 2 && l1.size(0) == n && l1.size(1) == loraRank)
      << "l1 must have shape [n, r] with the same LoRA rank as d (pre-divided by alpha)";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<std::uintptr_t>(l1.data_ptr()) % 16, 0)
      << "l1 must be 16-byte aligned for TMA";
  TVM_FFI_ICHECK(out.ndim() == 2 && out.size(0) == m && out.size(1) == n)
      << "out must have shape [m, n]";
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(out.dtype()), bfloat16_code)
      << "nvfp4_svdquant_gemm currently supports bf16 output only";

  void const* biasPtr = nullptr;
  if (bias.has_value()) {
    auto const& biasTensor = bias.value();
    CHECK_INPUT_AND_TYPE(biasTensor, dl_bfloat16);
    CHECK_DEVICE(biasTensor, a);
    TVM_FFI_ICHECK(biasTensor.ndim() == 1 && biasTensor.size(0) == n) << "bias must have shape [n]";
    biasPtr = biasTensor.data_ptr();
  }

  TVM_FFI_ICHECK(tactic >= -1 && tactic < flashinfer::gemm::kNvfp4SvdquantGemmNumTactics)
      << "invalid NVFP4 SVDQuant tactic: " << tactic;
  int const tacticId = tactic < 0 ? 0 : static_cast<int>(tactic);

  // Empty batch: out is [0, n], nothing to compute.
  if (m == 0) return;

  size_t const requiredWorkspaceBytes = flashinfer::gemm::nvfp4_svdquant_gemm_workspace_size(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), tacticId);
  auto stream = get_stream(a.device());

  auto runKernel = [&](void* workspace) {
    flashinfer::gemm::nvfp4_svdquant_gemm_run(
        out.data_ptr(), a.data_ptr(), b.data_ptr(), a_sf.data_ptr(), b_sf.data_ptr(),
        static_cast<float const*>(alpha.data_ptr()), d.data_ptr(), l1.data_ptr(), biasPtr,
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), static_cast<int>(loraRank),
        reinterpret_cast<char*>(workspace), requiredWorkspaceBytes, stream, tacticId, enable_pdl);
  };

  int64_t const provided_workspace_size =
      workspace_buffer.numel() * get_element_size(workspace_buffer);
  if (provided_workspace_size < static_cast<int64_t>(requiredWorkspaceBytes)) {
    Tensor new_workspace = alloc_tensor({static_cast<int64_t>(requiredWorkspaceBytes)},
                                        DLDataType{kDLInt, 8, 1}, a.device());
    runKernel(new_workspace.data_ptr());
  } else {
    runKernel(workspace_buffer.data_ptr());
  }
}

int64_t nvfp4_svdquant_gemm_tactic_num() { return flashinfer::gemm::kNvfp4SvdquantGemmNumTactics; }

}  // namespace torch_ext

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_svdquant_gemm, torch_ext::nvfp4_svdquant_gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_svdquant_gemm_tactic_num,
                              torch_ext::nvfp4_svdquant_gemm_tactic_num);
