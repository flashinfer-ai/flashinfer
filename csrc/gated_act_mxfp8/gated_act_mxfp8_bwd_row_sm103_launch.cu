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
#include <cuda.h>
#include <cuda_runtime.h>

#include "gated_act_mxfp8_bwd_row_sm103.cu"
#include "gated_act_mxfp8_launch.cuh"

namespace flashinfer::gated_act_mxfp8 {

cudaError_t LaunchBackwardRowSm103(__nv_bfloat16* input, __nv_bfloat16* grad,
                                   CUtensorMap row_act, CUtensorMap row_gate, uint8_t* row_scales,
                                   int m, int k, cudaStream_t stream) {
  kernel_gated_act_mxfp8_bwd_row_direct_64x64_sm103<<<dim3(k / 64, m / 32), 128, 4096, stream>>>(
      input, grad, row_act, row_gate, row_scales, m, k);
  return cudaGetLastError();
}

}  // namespace flashinfer::gated_act_mxfp8
