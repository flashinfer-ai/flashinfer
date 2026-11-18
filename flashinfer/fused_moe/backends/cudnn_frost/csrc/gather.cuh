// Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace frost_gather {

// Routing is kept on the device. The inverse index feeds FC1 directly, while
// mapping retains the surrounding pipeline's forward/fused-COMBINE contract.
__global__ void indices(const int32_t* ids, int32_t* cursors, int32_t* mapping,
                        int32_t* token_index, int32_t* row_experts, int rows, int topk, int experts,
                        int32_t* slots = nullptr, float* combined_scores = nullptr,
                        const float* scores = nullptr, const float* alpha = nullptr) {
  for (int64_t r = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; r < rows;
       r += int64_t(blockDim.x) * gridDim.x) {
    const int raw = ids[r];
    const bool valid = raw >= 0 && raw < experts;
    const int expert = valid ? raw : 0;
    const int dest = atomicAdd(cursors + expert, 1);
    token_index[dest] = valid ? r / topk : -1;
    if (row_experts) row_experts[dest] = expert;
    if (combined_scores) {
      mapping[dest] = valid ? r / topk : -1;
      slots[dest] = r % topk;
      combined_scores[r] = valid ? scores[r] * alpha[expert] : 0.f;
    } else {
      mapping[r] = dest;
    }
  }
}

inline int scale_pitch(int columns) { return (columns + 15) / 16 * 16; }

// Undo the input's global F8_128x4 layout once per original token. FC1 gathers
// these linear rows and creates its local MMA scale atoms inside the mainloop.
__global__ void linear_scales(const uint8_t* source, uint8_t* output, int tokens, int columns,
                              int pitch, bool swizzled) {
  const int words_per_row = pitch / 4;
  for (int64_t word = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       word < int64_t(tokens) * words_per_row; word += int64_t(blockDim.x) * gridDim.x) {
    const int row = word / words_per_row;
    const int col = (word % words_per_row) * 4;
    uint32_t value = 0;
    if (col < columns) {
      const int64_t offset = swizzled ? int64_t(row / 128) * 128 * columns + (col / 4) * 512 +
                                            (row % 32) * 16 + ((row % 128) / 32) * 4
                                      : int64_t(row) * columns + col;
      value = *reinterpret_cast<const uint32_t*>(source + offset);
    }
    reinterpret_cast<uint32_t*>(output)[word] = value;
  }
}

}  // namespace frost_gather
