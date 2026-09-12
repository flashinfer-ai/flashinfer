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

#include "gated_act_mxfp8_fwd_row_noalloc.cu"
#include "gated_act_mxfp8_launch.cuh"

namespace flashinfer::gated_act_mxfp8 {

cudaError_t LaunchForwardRowNoAllocate(__nv_bfloat16* input, CUtensorMap row_output,
                                       uint8_t* row_scales, int m, int k, cudaStream_t stream) {
  kernel_gated_act_mxfp8_fwd_row_direct_128x64_noalloc<<<dim3(k / 128, m / 32), 256, 4096,
                                                         stream>>>(input, row_output, row_scales, m,
                                                                   k);
  return cudaGetLastError();
}

}  // namespace flashinfer::gated_act_mxfp8
