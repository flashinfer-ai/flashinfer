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
#include <cstdint>

#include "tvm_ffi_utils.h"

namespace {

constexpr int kWarps = 8;
constexpr int kSlabRows = 16;

__device__ __forceinline__ int64_t SlabBytes(int cols) { return kSlabRows * (1 + cols / 2); }

// One warp per scale row. The row's base is the lowest start among the 16-value windows that
// cover the most of its bytes; the base and the row's 4-bit codes (0 outside the window) go to
// its slab of `fixed`, and the count of bytes outside to `row_exceptions`.
__global__ void CsfEncodeRowsKernel(const uint8_t* __restrict__ scales, uint8_t* __restrict__ fixed,
                                    int32_t* __restrict__ row_exceptions, int64_t num_rows,
                                    int cols) {
  __shared__ uint32_t hist[kWarps][256];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + warp;
  if (row >= num_rows) return;
  uint32_t* h = hist[warp];
  for (int i = lane; i < 256; i += 32) h[i] = 0;
  __syncwarp();
  const uint8_t* src = scales + row * cols;
  for (int c = lane; c < cols; c += 32) atomicAdd(&h[src[c]], 1u);
  __syncwarp();
  // The cover sits above the low byte, 255 - start in it: the max is the lowest best start.
  uint32_t best = 0;
  for (int start = lane; start <= 240; start += 32) {
    uint32_t cover = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) cover += h[start + i];
    best = max(best, (cover << 8) | (255 - start));
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    best = max(best, __shfl_xor_sync(0xffffffff, best, offset));
  }
  const int base = 255 - static_cast<int>(best & 0xff);
  const int r = static_cast<int>(row % kSlabRows);
  uint8_t* slab = fixed + row / kSlabRows * SlabBytes(cols);
  uint8_t* codes = slab + kSlabRows + r * (cols / 2);
  for (int i = lane; i < cols / 2; i += 32) {
    const int lo = src[2 * i] - base, hi = src[2 * i + 1] - base;
    codes[i] = (lo >= 0 && lo < 16 ? lo : 0) | (hi >= 0 && hi < 16 ? hi : 0) << 4;
  }
  if (lane == 0) {
    slab[r] = base;
    row_exceptions[row] = cols - static_cast<int>(best >> 8);
  }
}

// One warp per scale row with exceptions: writes them from `row_offsets[row]` in column
// order, each the flat `row * cols + col` position within the matrix in the low 24 bits and
// the scale byte in the high 8.
__global__ void CsfEncodeExceptionsKernel(const uint8_t* __restrict__ scales,
                                          const uint8_t* __restrict__ fixed,
                                          const int64_t* __restrict__ row_offsets,
                                          uint32_t* __restrict__ exceptions, int64_t num_rows,
                                          int rows, int cols) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + warp;
  if (row >= num_rows) return;
  int64_t out = row_offsets[row];
  if (out == row_offsets[row + 1]) return;
  const int base = fixed[row / kSlabRows * SlabBytes(cols) + row % kSlabRows];
  const uint8_t* src = scales + row * cols;
  const uint32_t first = static_cast<uint32_t>(row % rows * cols);
  for (int c0 = 0; c0 < cols; c0 += 32) {
    const int c = c0 + lane;
    const int value = c < cols ? src[c] : base;
    const bool outside = value < base || value > base + 15;
    const uint32_t mask = __ballot_sync(0xffffffff, outside);
    if (outside) {
      exceptions[out + __popc(mask & ((1u << lane) - 1))] =
          (first + c) | static_cast<uint32_t>(value) << 24;
    }
    out += __popc(mask);
  }
}

void CheckCsfScales(TensorView scales, TensorView fixed) {
  CHECK_INPUT_AND_TYPE(scales, dl_uint8);
  CHECK_INPUT_AND_TYPE(fixed, dl_uint8);
  CHECK_DIM(3, scales);
  CHECK_DIM(3, fixed);
  const int64_t rows = scales.size(1), cols = scales.size(2);
  TVM_FFI_ICHECK(rows % kSlabRows == 0 && cols % 2 == 0 && cols > 0 && rows * cols <= 1 << 24)
      << "CSF scales need rows % 16 == 0, even cols and rows * cols <= 2^24, got " << rows << " x "
      << cols;
  TVM_FFI_ICHECK(fixed.size(0) == scales.size(0) && fixed.size(1) == rows / kSlabRows &&
                 fixed.size(2) == kSlabRows * (1 + cols / 2))
      << "fixed must be [num_matrices, rows / 16, 16 * (1 + cols / 2)]";
}

int NumBlocks(int64_t num_rows) { return static_cast<int>((num_rows + kWarps - 1) / kWarps); }

}  // namespace

void csf_encode_rows(TensorView scales, TensorView fixed, TensorView row_exceptions) {
  CheckCsfScales(scales, fixed);
  CHECK_INPUT_AND_TYPE(row_exceptions, dl_int32);
  const int64_t num_rows = scales.size(0) * scales.size(1);
  TVM_FFI_ICHECK_EQ(row_exceptions.numel(), num_rows);
  if (num_rows == 0) return;
  CsfEncodeRowsKernel<<<NumBlocks(num_rows), kWarps * 32, 0, get_stream(scales.device())>>>(
      static_cast<const uint8_t*>(scales.data_ptr()), static_cast<uint8_t*>(fixed.data_ptr()),
      static_cast<int32_t*>(row_exceptions.data_ptr()), num_rows, scales.size(2));
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess) << "csf_encode_rows failed: " << cudaGetErrorString(status);
}

void csf_encode_exceptions(TensorView scales, TensorView fixed, TensorView row_offsets,
                           TensorView exceptions) {
  CheckCsfScales(scales, fixed);
  CHECK_INPUT_AND_TYPE(row_offsets, dl_int64);
  CHECK_INPUT_AND_TYPE(exceptions, dl_uint32);
  const int64_t num_rows = scales.size(0) * scales.size(1);
  TVM_FFI_ICHECK_EQ(row_offsets.numel(), num_rows + 1);
  if (exceptions.numel() == 0) return;
  CsfEncodeExceptionsKernel<<<NumBlocks(num_rows), kWarps * 32, 0, get_stream(scales.device())>>>(
      static_cast<const uint8_t*>(scales.data_ptr()), static_cast<const uint8_t*>(fixed.data_ptr()),
      static_cast<const int64_t*>(row_offsets.data_ptr()),
      static_cast<uint32_t*>(exceptions.data_ptr()), num_rows, scales.size(1), scales.size(2));
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "csf_encode_exceptions failed: " << cudaGetErrorString(status);
}
