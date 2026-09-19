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

#include "../utils.cuh"

namespace flashinfer {

// The logistic function at float precision. This module is built without
// -use_fast_math -- see flashinfer/jit/qsa_output_gate.py -- so this is the
// accurate exponential and an IEEE divide, and a result below the smallest
// normal float stays subnormal instead of being flushed away.
__forceinline__ __device__ float qsa_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// The attention output of a gated model is multiplied elementwise by
// sigmoid(gate) before it leaves the attention step. A kernel that fuses this
// into its own epilogue never has to round twice; one that returns the
// attention output alone leaves the caller to do it, and doing it with a chain
// of elementwise ops both rounds the sigmoid to the output dtype and, under
// CUDA graph capture, puts every intermediate in the graph's private pool.
//
// This kernel does it in one pass and holds nothing: the attention value is
// read at the output dtype (so its rounding is already what a fused epilogue
// would have produced), the gate is widened to float, the product is formed in
// float, and the result is stored once.
//
// It also absorbs the row padding an attention plan with a fixed row count
// leaves behind: `attention` may be taller than `out`, and only the first
// `out` rows are read.
//
// `attention` and `out` may name the same storage: scaling in place is what a
// caller with no row padding wants, so neither is marked restrict. `gate` is
// read-only and is required not to overlap `out`, which the entry point checks.
//
// The feature axis is the contiguous one in all three tensors, so a thread
// takes a whole 16-byte vector of it. Reading two bytes at a time instead ran
// at about a fifth of the rate on a kernel that is nothing but memory traffic
// (204 GB/s against 1026).
template <typename DType, uint32_t VEC_SIZE>
__global__ void QSAOutputGateKernel(const DType* attention, const DType* __restrict__ gate,
                                    DType* out, uint32_t rows, uint32_t num_heads,
                                    uint32_t head_dim, uint32_t vectors_per_line,
                                    uint32_t threads_per_line, uint32_t lines_per_block,
                                    int64_t attention_row_stride, int64_t attention_head_stride,
                                    int64_t gate_row_stride, int64_t gate_head_stride,
                                    int64_t out_row_stride, int64_t out_head_stride) {
  using Vec = uint4;
  static_assert(sizeof(Vec) == VEC_SIZE * sizeof(DType), "the vector is 16 bytes wide");
  // A line is one (row, head): `threads_per_line` threads cover it, each taking
  // every `threads_per_line`-th vector, and a block holds as many lines as its
  // threads allow. Lines beyond one grid are walked by a stride.
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
      const uint32_t base = vector * VEC_SIZE;
      const Vec value = *reinterpret_cast<const Vec*>(attention_line + base);
      const Vec weight = *reinterpret_cast<const Vec*>(gate_line + base);
      Vec result;
      const DType* value_elems = reinterpret_cast<const DType*>(&value);
      const DType* weight_elems = reinterpret_cast<const DType*>(&weight);
      DType* result_elems = reinterpret_cast<DType*>(&result);
#pragma unroll
      for (uint32_t i = 0; i < VEC_SIZE; ++i) {
        result_elems[i] = static_cast<DType>(static_cast<float>(value_elems[i]) *
                                             qsa_sigmoid(static_cast<float>(weight_elems[i])));
      }
      *reinterpret_cast<Vec*>(out_line + base) = result;
    }
  }
}

// The same thing one element at a time, for a head the vector width does not
// divide or a tensor whose lines are not 16-byte aligned.
template <typename DType>
__global__ void QSAOutputGateScalarKernel(const DType* attention, const DType* __restrict__ gate,
                                          DType* out, uint32_t rows, uint32_t num_heads,
                                          uint32_t head_dim, int64_t attention_row_stride,
                                          int64_t attention_head_stride, int64_t gate_row_stride,
                                          int64_t gate_head_stride, int64_t out_row_stride,
                                          int64_t out_head_stride) {
  const uint32_t head = blockIdx.x;
  if (head >= num_heads) return;
  for (uint32_t row = blockIdx.y; row < rows; row += gridDim.y) {
    const DType* attention_line =
        attention + row * attention_row_stride + head * attention_head_stride;
    const DType* gate_line = gate + row * gate_row_stride + head * gate_head_stride;
    DType* out_line = out + row * out_row_stride + head * out_head_stride;
    for (uint32_t column = threadIdx.x; column < head_dim; column += blockDim.x) {
      out_line[column] = static_cast<DType>(static_cast<float>(attention_line[column]) *
                                            qsa_sigmoid(static_cast<float>(gate_line[column])));
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
  constexpr uint32_t kVecSize = 16 / sizeof(DType);
  constexpr uint32_t kMaxThreads = 256;
  constexpr uint32_t kMaxGridY = 65535;

  // Every line the vectorized kernel touches has to start on a 16-byte
  // boundary and be a whole number of vectors long. Strides are in elements,
  // so a stride that is not a multiple of the vector width moves a line off the
  // boundary even when the base pointer is on it.
  const auto aligned = [&](const void* base, int64_t row_stride, int64_t head_stride) {
    return reinterpret_cast<uintptr_t>(base) % 16 == 0 && row_stride % kVecSize == 0 &&
           head_stride % kVecSize == 0;
  };
  const bool vectorizable = head_dim % kVecSize == 0 &&
                            aligned(attention, attention_row_stride, attention_head_stride) &&
                            aligned(gate, gate_row_stride, gate_head_stride) &&
                            aligned(out, out_row_stride, out_head_stride);

  if (vectorizable) {
    const uint32_t vectors_per_line = head_dim / kVecSize;
    // Whole lines per block, so no thread straddles two of them. A line wider
    // than a block is walked by its threads instead.
    const uint32_t threads_per_line =
        vectors_per_line < kMaxThreads ? vectors_per_line : kMaxThreads;
    const uint32_t lines_per_block = kMaxThreads / threads_per_line;
    const uint32_t threads = threads_per_line * lines_per_block;
    const uint64_t lines = static_cast<uint64_t>(rows) * num_heads;
    const uint64_t blocks_needed = ceil_div(lines, static_cast<uint64_t>(lines_per_block));
    constexpr uint64_t kMaxGridX = 2147483647ULL;
    const uint32_t blocks =
        static_cast<uint32_t>(blocks_needed < kMaxGridX ? blocks_needed : kMaxGridX);
    auto kernel = QSAOutputGateKernel<DType, kVecSize>;
    void* args[] = {(void*)&attention,
                    (void*)&gate,
                    (void*)&out,
                    (void*)&rows,
                    (void*)&num_heads,
                    (void*)&head_dim,
                    (void*)&vectors_per_line,
                    (void*)&threads_per_line,
                    (void*)&lines_per_block,
                    (void*)&attention_row_stride,
                    (void*)&attention_head_stride,
                    (void*)&gate_row_stride,
                    (void*)&gate_head_stride,
                    (void*)&out_row_stride,
                    (void*)&out_head_stride};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, blocks, threads, args, 0, stream));
    return cudaSuccess;
  }

  const uint32_t warp_aligned = round_up(head_dim, 32u);
  const uint32_t threads = warp_aligned < kMaxThreads ? warp_aligned : kMaxThreads;
  const dim3 grid(num_heads, rows < kMaxGridY ? rows : kMaxGridY);
  auto kernel = QSAOutputGateScalarKernel<DType>;
  void* args[] = {(void*)&attention,
                  (void*)&gate,
                  (void*)&out,
                  (void*)&rows,
                  (void*)&num_heads,
                  (void*)&head_dim,
                  (void*)&attention_row_stride,
                  (void*)&attention_head_stride,
                  (void*)&gate_row_stride,
                  (void*)&gate_head_stride,
                  (void*)&out_row_stride,
                  (void*)&out_head_stride};
  FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, grid, threads, args, 0, stream));
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_QSA_OUTPUT_GATE_CUH_
