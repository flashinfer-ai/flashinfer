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
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace flashinfer::gated_act_mxfp8 {

cudaError_t LaunchForwardRow(__nv_bfloat16* input, CUtensorMap row_output, uint8_t* row_scales,
                             int m, int k, cudaStream_t stream);

cudaError_t LaunchForwardRowNoAllocate(__nv_bfloat16* input, CUtensorMap row_output,
                                       uint8_t* row_scales, int m, int k, cudaStream_t stream);

cudaError_t LaunchBackwardRow(__nv_bfloat16* input, __nv_bfloat16* grad, CUtensorMap row_act,
                              CUtensorMap row_gate, uint8_t* row_scales, int m, int k,
                              cudaStream_t stream);

cudaError_t LaunchBackwardRowSm103(__nv_bfloat16* input, __nv_bfloat16* grad, CUtensorMap row_act,
                                   CUtensorMap row_gate, uint8_t* row_scales, int m, int k,
                                   cudaStream_t stream);

cudaError_t LaunchForwardCol(CUtensorMap gate, CUtensorMap up, CUtensorMap col_output,
                             uint8_t* col_scales, int m, int k, cudaStream_t stream);

cudaError_t LaunchBackwardCol(CUtensorMap gate, CUtensorMap up, CUtensorMap grad,
                              CUtensorMap col_act, CUtensorMap col_gate, uint8_t* col_scales, int m,
                              int k, cudaStream_t stream);

cudaError_t LaunchForwardBoth(__nv_bfloat16* input, CUtensorMap row_output, CUtensorMap col_output,
                              uint8_t* row_scales, uint8_t* col_scales, int m, int k,
                              cudaStream_t stream);

cudaError_t LaunchForwardBothNoAllocate(__nv_bfloat16* input, CUtensorMap row_output,
                                        CUtensorMap col_output, uint8_t* row_scales,
                                        uint8_t* col_scales, int m, int k, cudaStream_t stream);

cudaError_t LaunchBackwardBoth(__nv_bfloat16* input, __nv_bfloat16* grad, CUtensorMap row_act,
                               CUtensorMap row_gate, CUtensorMap col_act, CUtensorMap col_gate,
                               uint8_t* row_scales, uint8_t* col_scales, int m, int k,
                               cudaStream_t stream);

}  // namespace flashinfer::gated_act_mxfp8
