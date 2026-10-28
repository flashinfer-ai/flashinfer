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
// The Blackwell W8A16 kernels of this module's activation dtype, compiled apart from the
// TVM-FFI launcher (w8a16_fp8_gemm_sm100.cu, which declares these two instantiations extern).
// This unit must not include tvm_ffi_utils.h: its global `using tvm::ffi::Tensor` makes the
// unqualified `Tensor` in CUTLASS's sm100 epilogue ambiguous with cute::Tensor under nvcc
// 12.9 and 13.x.
#include <flashinfer/gemm/w8a16_fp8_gemm_sm100.cuh>

#include "w8a16_fp8_gemm_config.inc"

namespace flashinfer::gemm {

template cudaError_t W8A16Fp8GemmSm100<W8A16ActType>(
    const W8A16Sm100Plan& plan, const W8A16ActType* act, const __nv_fp8_e4m3* weight,
    const float* scale, const W8A16ActType* bias, W8A16ActType* out, int M, int N, int K,
    void* workspace, size_t workspace_bytes, int device_id, int num_sms, cudaStream_t stream);
template cudaError_t W8A16Fp8GemmSm100WorkspaceBytes<W8A16ActType>(const W8A16Sm100Plan& plan,
                                                                   int M, int N, int K,
                                                                   int device_id, int num_sms,
                                                                   size_t* bytes);

}  // namespace flashinfer::gemm
