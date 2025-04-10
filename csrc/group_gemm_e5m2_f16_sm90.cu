/*
 * Copyright (c) 2024 by FlashInfer team.
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
#include <flashinfer/gemm/group_gemm_sm90.cuh>

using namespace flashinfer;
using namespace flashinfer::group_gemm;

namespace flashinfer {
namespace group_gemm {

template cudaError_t CutlassSegmentGEMMSM90Run<cutlass::float_e5m2_t, cutlass::half_t>(
    void* float_buffer, size_t float_buffer_size_in_bytes, void* int_buffer,
    size_t int_buffer_size_in_bytes, void* all_problems, int64_t batch_size, void* x, void* w,
    void* y, void* x_stride, void* w_stride, void* y_stride, bool weight_column_major,
    cudaStream_t stream);

};  // namespace group_gemm
};  // namespace flashinfer
