/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer::union_gqa {

cudaError_t LaunchUnionGQAPrefill(void* q, void* k, void* v, int32_t* block_indices,
                                  int32_t* cu_seqlens_q, void* out, int total_q, int num_qo_heads,
                                  int num_kv_heads, int topk_blocks, int batch_size,
                                  int max_seqlen_q, float softmax_scale, cudaStream_t stream);

}  // namespace flashinfer::union_gqa
