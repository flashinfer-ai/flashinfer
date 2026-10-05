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
#ifndef FLASHINFER_ATTENTION_QSA_OUTPUT_GATE_CUH_
#define FLASHINFER_ATTENTION_QSA_OUTPUT_GATE_CUH_

#include <cuda_runtime.h>

#include <cstdint>

#include "../../utils.cuh"
#include "../../vec_dtypes.cuh"

namespace flashinfer {

// Built without -use_fast_math (see flashinfer/jit/qsa_ops.py): the accurate exponential and
// an IEEE divide, so a logistic below the smallest normal float stays subnormal instead of
// being flushed to zero.
__forceinline__ __device__ float qsa_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// out = attention * sigmoid(gate), formed in float and stored once. `attention` may be taller
// than `out` (an attention plan's row padding; only `out`'s rows are read) and may be `out`
// itself, so neither is restrict; `gate` may not overlap `out`. A thread takes VEC_SIZE
// contiguous features of one (row, head) line, and the host picks the widest vector every
// line's start and strides allow: two bytes at a time ran at a fifth of the rate.
template <typename DType, uint32_t VEC_SIZE>
__global__ void QSAOutputGateKernel(const DType* attention, const DType* __restrict__ gate,
                                    DType* out, uint32_t rows, uint32_t num_heads,
                                    uint32_t vectors_per_line, uint32_t threads_per_line,
                                    uint32_t lines_per_block, int64_t attention_row_stride,
                                    int64_t attention_head_stride, int64_t gate_row_stride,
                                    int64_t gate_head_stride, int64_t out_row_stride,
                                    int64_t out_head_stride) {
  const uint32_t line_in_block = threadIdx.x / threads_per_line;
  const uint32_t vector_in_line = threadIdx.x - line_in_block * threads_per_line;
  if (line_in_block >= lines_per_block) return;
  const uint64_t lines = static_cast<uint64_t>(rows) * num_heads;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * lines_per_block;
  for (uint64_t line = static_cast<uint64_t>(blockIdx.x) * lines_per_block + line_in_block;
       line < lines; line += stride) {
    const uint32_t row = static_cast<uint32_t>(line / num_heads);
    const uint32_t head = static_cast<uint32_t>(line - static_cast<uint64_t>(row) * num_heads);
    const DType* attention_line =
        attention + row * attention_row_stride + head * attention_head_stride;
    const DType* gate_line = gate + row * gate_row_stride + head * gate_head_stride;
    DType* out_line = out + row * out_row_stride + head * out_head_stride;
    for (uint32_t vector = vector_in_line; vector < vectors_per_line; vector += threads_per_line) {
      vec_t<DType, VEC_SIZE> value, weight;
      value.load(attention_line + vector * VEC_SIZE);
      weight.load(gate_line + vector * VEC_SIZE);
#pragma unroll
      for (uint32_t i = 0; i < VEC_SIZE; ++i) {
        value[i] = static_cast<DType>(static_cast<float>(value[i]) *
                                      qsa_sigmoid(static_cast<float>(weight[i])));
      }
      value.store(out_line + vector * VEC_SIZE);
    }
  }
}

template <typename DType>
cudaError_t QSAOutputGate(const DType* attention, const DType* gate, DType* out, uint32_t rows,
                          uint32_t num_heads, uint32_t head_dim, int64_t attention_row_stride,
                          int64_t attention_head_stride, int64_t gate_row_stride,
                          int64_t gate_head_stride, int64_t out_row_stride, int64_t out_head_stride,
                          cudaStream_t stream) {
  if (rows == 0 || num_heads == 0 || head_dim == 0) return cudaSuccess;
  constexpr uint32_t kMaxThreads = 256;
  // Strides are in elements: a line starts on a vector boundary only if the base pointer does
  // and both strides are multiples of the vector.
  const auto fits = [&](uint32_t vec) {
    const auto aligned = [&](const void* base, int64_t row_stride, int64_t head_stride) {
      return reinterpret_cast<uintptr_t>(base) % (vec * sizeof(DType)) == 0 &&
             row_stride % vec == 0 && head_stride % vec == 0;
    };
    return head_dim % vec == 0 && aligned(attention, attention_row_stride, attention_head_stride) &&
           aligned(gate, gate_row_stride, gate_head_stride) &&
           aligned(out, out_row_stride, out_head_stride);
  };
  uint32_t vec_size = 16 / sizeof(DType);
  while (vec_size > 1 && !fits(vec_size)) vec_size /= 2;
  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    const uint32_t vectors_per_line = head_dim / VEC_SIZE;
    const uint32_t threads_per_line =
        vectors_per_line < kMaxThreads ? vectors_per_line : kMaxThreads;
    const uint32_t lines_per_block = kMaxThreads / threads_per_line;
    const uint64_t lines = static_cast<uint64_t>(rows) * num_heads;
    const uint64_t blocks_needed = ceil_div(lines, static_cast<uint64_t>(lines_per_block));
    constexpr uint64_t kMaxGridX = 2147483647ULL;
    const uint32_t blocks =
        static_cast<uint32_t>(blocks_needed < kMaxGridX ? blocks_needed : kMaxGridX);
    QSAOutputGateKernel<DType, VEC_SIZE><<<blocks, threads_per_line * lines_per_block, 0, stream>>>(
        attention, gate, out, rows, num_heads, vectors_per_line, threads_per_line, lines_per_block,
        attention_row_stride, attention_head_stride, gate_row_stride, gate_head_stride,
        out_row_stride, out_head_stride);
    FLASHINFER_CUDA_CALL(cudaGetLastError());
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_QSA_OUTPUT_GATE_CUH_
