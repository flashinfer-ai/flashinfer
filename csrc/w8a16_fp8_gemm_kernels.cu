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
// The Hopper W8A16 kernels of this module's activation dtype, compiled apart from the TVM-FFI
// launcher (w8a16_fp8_gemm.cu, which declares these two instantiations extern).  Same layout
// as Blackwell (w8a16_fp8_gemm_sm100_kernels.cu), where it is required; this unit likewise
// includes no TVM-FFI header.
#include <flashinfer/gemm/w8a16_fp8_gemm_sm90.cuh>

#include "w8a16_fp8_gemm_config.inc"

namespace flashinfer::gemm {

template cudaError_t W8A16Fp8GemmSm90<W8A16ActType>(
    const W8A16Sm90Plan& plan, const W8A16ActType* act, const __nv_fp8_e4m3* weight,
    const float* scale, const W8A16ActType* bias, W8A16ActType* out, int M, int N, int K,
    void* workspace, size_t workspace_bytes, int device_id, int num_sms, cudaStream_t stream);
template cudaError_t W8A16Fp8GemmSm90WorkspaceBytes<W8A16ActType>(const W8A16Sm90Plan& plan, int M,
                                                                  int N, int K, int device_id,
                                                                  int num_sms, size_t* bytes);

}  // namespace flashinfer::gemm
