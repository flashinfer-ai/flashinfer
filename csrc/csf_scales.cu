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

using tvm::ffi::Optional;

namespace {

constexpr int kWarps = 8;
constexpr int kSlabRows = 16;
// Expansion works on the 128-row tiles of the 128x4 layout, 8 slabs each.
constexpr int kTileRows = 128;
constexpr int kTileSlabs = kTileRows / kSlabRows;
static_assert(kTileSlabs == kWarps, "expansion takes one warp per slab of a tile");
// 16-byte loads each lane has in flight while staging a slab.
constexpr int kStageLoads = 4;
// SfLayout values.
constexpr int64_t kLayout128x4 = 0;
constexpr int64_t kLayoutLinear = 2;

__host__ __device__ __forceinline__ int64_t SlabBytes(int cols) {
  return kSlabRows * (1 + cols / 2);
}

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

enum class Dest { kSwizzled128x4, kLinear };

struct CsfExpandParams {
  const uint8_t* fixed;
  const uint32_t* exceptions;
  const int32_t* exc_offsets;
  // Routed expert ids, or null to expand every expert.
  const int32_t* experts;
  // The routed id of expert 0.
  int expert_offset;
  int num_experts;
  int rows, cols;
  // Columns of the destination: the padded columns for 128x4; at least `cols`.
  int out_cols;
  // Destination slab d comes from source slab (d + rotate_slabs) % (rows / 16).
  int rotate_slabs;
  // Shared memory words per tile row, see TileWord.
  int tile_words;
};

// Word `word` (columns 4 * word ..) of tile row `row`. Rows are an odd number of words apart and
// their words XOR-swizzled within groups of four by (row / 32) % 4, so the 128x4 writer (rows
// m + 32 * i of one word) reads without bank conflicts.
__device__ __forceinline__ int TileWord(int row, int word, int tile_words) {
  return row * tile_words + (word ^ (row >> 5 & 3));
}

__device__ __forceinline__ uint8_t* TileByte(uint32_t* tile, int row, int col, int tile_words) {
  return reinterpret_cast<uint8_t*>(tile + TileWord(row, col / 4, tile_words)) + col % 4;
}

// One block per (expert or routing slot, 128-row tile of the destination), warp k taking the
// tile's slab k: the warp stages the slab in shared memory, decodes its rows into the tile kUnit
// scales at a time and overwrites the exceptions it fetched meanwhile. The block then writes the
// tile out.
template <Dest kDest, int kUnit>
__global__ void CsfExpandKernel(CsfExpandParams p, uint8_t* __restrict__ out) {
  extern __shared__ uint4 smem[];
  int expert = blockIdx.x;
  if (p.experts != nullptr) {
    // Each routed expert is expanded once, by the first slot naming it. The earlier slots load
    // alongside this one.
    const int id = p.experts[blockIdx.x];
    bool skip = false;
    for (int i = threadIdx.x; i < static_cast<int>(blockIdx.x); i += blockDim.x) {
      skip |= p.experts[i] == id;
    }
    // Compared first, as id - expert_offset can overflow.
    expert = id < p.expert_offset ? -1 : id - p.expert_offset;
    if (__syncthreads_or(skip || expert < 0 || expert >= p.num_experts)) return;
  }
  const int slabs = p.rows / kSlabRows, cols = p.cols, tile_words = p.tile_words;
  // An expert's slabs take rows * (1 + cols / 2) <= 2^24 bytes: 32-bit offsets.
  const int slab_bytes = static_cast<int>(SlabBytes(cols));
  const int first_slab = blockIdx.y * kTileSlabs;
  const int tile_slabs = min(kTileSlabs, slabs - first_slab);
  const int tile_rows = tile_slabs * kSlabRows;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  uint32_t* tile = reinterpret_cast<uint32_t*>(smem);
  // The staged slabs follow the tile's 512 * tile_words bytes, 16-byte aligned.
  uint8_t* staged = reinterpret_cast<uint8_t*>(tile + kTileRows * tile_words);
  if (warp < tile_slabs) {
    int s = first_slab + warp + p.rotate_slabs;
    if (s >= slabs) s -= slabs;
    const int64_t slab = static_cast<int64_t>(expert) * slabs + s;
    // Lane 0 reads where the slab's exceptions start, lane 1 where they end.
    const int bound = p.exc_offsets[slab + (lane & 1)];
    uint8_t* bases = staged + warp * slab_bytes;
    {
      const uint4* src = reinterpret_cast<const uint4*>(p.fixed + slab * slab_bytes);
      uint4* dst = reinterpret_cast<uint4*>(bases);
      for (int i = lane; i < slab_bytes / 16; i += 32 * kStageLoads) {
        uint4 v[kStageLoads];
#pragma unroll
        for (int j = 0; j < kStageLoads; ++j) {
          if (i + 32 * j < slab_bytes / 16) v[j] = src[i + 32 * j];
        }
#pragma unroll
        for (int j = 0; j < kStageLoads; ++j) {
          if (i + 32 * j < slab_bytes / 16) dst[i + 32 * j] = v[j];
        }
      }
    }
    const int start = __shfl_sync(0xffffffffu, bound, 0), end = __shfl_sync(0xffffffffu, bound, 1);
    // The slab's first 32 exceptions load while it decodes.
    const uint32_t first_exception = start + lane < end ? p.exceptions[start + lane] : 0;
    // Units of two scales leave the top half of a row's last word to the zeros.
    if (cols % 4 != 0 && lane < kSlabRows) {
      tile[TileWord(warp * kSlabRows + lane, cols / 4, tile_words)] = 0;
    }
    __syncwarp();
    // Unit u of row r: kUnit scales from column kUnit * u, kUnit / 2 bytes of codes.
    const int row_units = cols / kUnit;
    const int step_rows = 32 / row_units, step_units = 32 % row_units;
    int r = lane / row_units, u = lane % row_units;
    for (int i = lane; i < kSlabRows * row_units; i += 32) {
      const uint8_t* src = bases + kSlabRows + r * (cols / 2) + u * (kUnit / 2);
      const uint32_t base = bases[r] * 0x01010101u;
      const int row = warp * kSlabRows + r;
      if constexpr (kUnit == 8) {
        const uint32_t codes = *reinterpret_cast<const uint32_t*>(src);
        const uint32_t lo = codes & 0x0f0f0f0fu, hi = codes >> 4 & 0x0f0f0f0fu;
        tile[TileWord(row, 2 * u, tile_words)] = __byte_perm(lo, hi, 0x5140) + base;
        tile[TileWord(row, 2 * u + 1, tile_words)] = __byte_perm(lo, hi, 0x7362) + base;
      } else if constexpr (kUnit == 4) {
        const uint32_t codes = *reinterpret_cast<const uint16_t*>(src);
        tile[TileWord(row, u, tile_words)] =
            __byte_perm(codes & 0x0f0fu, codes >> 4 & 0x0f0fu, 0x5140) + base;
      } else {
        const uint32_t codes = *src;
        reinterpret_cast<uint16_t*>(tile + TileWord(row, u / 2, tile_words))[u % 2] =
            ((codes & 0xfu) | (codes >> 4) << 8) + (base & 0xffffu);
      }
      r += step_rows;
      u += step_units;
      if (u >= row_units) {
        u -= row_units;
        ++r;
      }
    }
    __syncwarp();
    // Exceptions overwrite their decoded bytes.
    const int first = s * kSlabRows * cols;
    auto overwrite = [&](uint32_t e) {
      const int pos = static_cast<int>(e & 0xffffff) - first, slab_row = pos / cols;
      *TileByte(tile, warp * kSlabRows + slab_row, pos - slab_row * cols, tile_words) = e >> 24;
    };
    if (start + lane < end) overwrite(first_exception);
    for (int x = start + 32 + lane; x < end; x += 32) overwrite(p.exceptions[x]);
  }
  __syncthreads();

  if constexpr (kDest == Dest::kSwizzled128x4) {
    // The tile's out_cols / 4 blocks of 512 bytes are contiguous; 16-byte line (word, m) holds
    // rows m + 32 * i of columns 4 * word .. 4 * word + 3.
    const int64_t padded_rows = static_cast<int64_t>(gridDim.y) * kTileRows;
    uint4* dst = reinterpret_cast<uint4*>(
        out + (expert * padded_rows + static_cast<int64_t>(blockIdx.y) * kTileRows) * p.out_cols);
    for (int l = threadIdx.x; l < p.out_cols / 4 * 32; l += blockDim.x) {
      const int word = l / 32, m = l % 32;
      uint32_t v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = i * 32 + m;
        v[i] = row < tile_rows ? tile[TileWord(row, word, tile_words)] : 0;
      }
      dst[l] = make_uint4(v[0], v[1], v[2], v[3]);
    }
  } else {
    uint8_t* dst = out + (static_cast<int64_t>(expert) * p.rows + first_slab * kSlabRows) * cols;
    for (int i = threadIdx.x; i < tile_rows * cols; i += blockDim.x) {
      const int r = i / cols;
      dst[i] = *TileByte(tile, r, i - r * cols, tile_words);
    }
  }
}

// An odd number of words per tile row, past the row's last group of four that the swizzle
// permutes within.
int TileWords(int64_t cols) { return static_cast<int>((cols + 15) / 16 * 4 + 1); }

// The tile and the staged slabs, about 192 bytes per column.
int64_t ExpandSmemBytes(int64_t cols) {
  return kTileRows * 4 * TileWords(cols) + kTileSlabs * SlabBytes(static_cast<int>(cols));
}

template <Dest kDest, int kUnit>
void LaunchCsfExpandUnit(const CsfExpandParams& p, int64_t num_items, uint8_t* out,
                         cudaStream_t stream) {
  const int64_t smem = ExpandSmemBytes(p.cols);
  auto kernel = CsfExpandKernel<kDest, kUnit>;
  cudaError_t status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(smem));
  // Clear the error, or the next launch check reports it.
  if (status != cudaSuccess) cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "csf_expand needs " << smem << " bytes of shared memory: " << cudaGetErrorString(status);
  const dim3 grid(num_items, (p.rows + kTileRows - 1) / kTileRows);
  kernel<<<grid, kWarps * 32, smem, stream>>>(p, out);
  status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess) << "csf_expand failed: " << cudaGetErrorString(status);
}

// Decodes the widest unit of scales that tiles a row.
template <Dest kDest>
void LaunchCsfExpand(const CsfExpandParams& p, int64_t num_items, uint8_t* out,
                     cudaStream_t stream) {
  if (p.cols % 8 == 0) {
    LaunchCsfExpandUnit<kDest, 8>(p, num_items, out, stream);
  } else if (p.cols % 4 == 0) {
    LaunchCsfExpandUnit<kDest, 4>(p, num_items, out, stream);
  } else {
    LaunchCsfExpandUnit<kDest, 2>(p, num_items, out, stream);
  }
}

void CheckCsfScales(TensorView scales, TensorView fixed) {
  CHECK_INPUT_AND_TYPE(scales, dl_uint8);
  CHECK_INPUT_AND_TYPE(fixed, dl_uint8);
  CHECK_DIM(3, scales);
  CHECK_DIM(3, fixed);
  CHECK_DEVICE(fixed, scales);
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
  CHECK_DEVICE(row_exceptions, scales);
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
  CHECK_DEVICE(row_offsets, scales);
  CHECK_DEVICE(exceptions, scales);
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

void csf_expand(TensorView fixed, TensorView exceptions, TensorView exc_offsets,
                Optional<TensorView> token_selected_experts, TensorView out, int64_t layout,
                bool swap_row_halves, int64_t local_expert_offset) {
  CHECK_INPUT_AND_TYPE(fixed, dl_uint8);
  CHECK_INPUT_AND_TYPE(exceptions, dl_uint32);
  CHECK_INPUT_AND_TYPE(exc_offsets, dl_int32);
  CHECK_INPUT_AND_TYPE(out, dl_uint8);
  CHECK_DIM(3, fixed);
  CHECK_DIM(3, out);
  CHECK_DEVICE(exceptions, fixed);
  CHECK_DEVICE(exc_offsets, fixed);
  CHECK_DEVICE(out, fixed);
  const int64_t num_experts = fixed.size(0), slabs = fixed.size(1);
  const int64_t rows = slabs * kSlabRows, cols = (fixed.size(2) / kSlabRows - 1) * 2;
  TVM_FFI_ICHECK(fixed.size(2) % kSlabRows == 0 && cols > 0 && rows * cols <= 1 << 24)
      << "fixed must be [num_matrices, rows / 16, 16 * (1 + cols / 2)]";
  TVM_FFI_ICHECK_EQ(exc_offsets.numel(), num_experts * slabs + 1);
  TVM_FFI_ICHECK(!swap_row_halves || slabs % 2 == 0)
      << "swapping row halves needs rows % 32 == 0, got " << rows;
  TVM_FFI_ICHECK_EQ(out.size(0), num_experts);
  Dest dest;
  int64_t out_cols = cols;
  if (layout == kLayout128x4) {
    out_cols = (cols + 3) / 4 * 4;
    TVM_FFI_ICHECK(out.size(1) == (rows + kTileRows - 1) / kTileRows * kTileRows &&
                   out.size(2) == out_cols)
        << "out must be [num_matrices, round_up(rows, 128), round_up(cols, 4)]";
    dest = Dest::kSwizzled128x4;
  } else {
    TVM_FFI_ICHECK(layout == kLayoutLinear) << "unsupported layout " << layout;
    TVM_FFI_ICHECK(out.size(1) == rows && out.size(2) == cols)
        << "out must be [num_matrices, rows, cols]";
    dest = Dest::kLinear;
  }
  // Slabs load 16 bytes at a time, and the 128x4 tiles store 16.
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(fixed.data_ptr()) % 16 == 0)
      << "fixed must be 16-byte aligned";
  TVM_FFI_ICHECK(dest == Dest::kLinear || reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0)
      << "out must be 16-byte aligned";
  CsfExpandParams p{};
  p.fixed = static_cast<const uint8_t*>(fixed.data_ptr());
  p.exceptions = static_cast<const uint32_t*>(exceptions.data_ptr());
  p.exc_offsets = static_cast<const int32_t*>(exc_offsets.data_ptr());
  p.num_experts = static_cast<int>(num_experts);
  p.rows = static_cast<int>(rows);
  p.cols = static_cast<int>(cols);
  p.out_cols = static_cast<int>(out_cols);
  p.rotate_slabs = swap_row_halves ? static_cast<int>(slabs / 2) : 0;
  p.tile_words = TileWords(cols);
  int64_t num_items = num_experts;
  if (token_selected_experts.has_value()) {
    TensorView experts = token_selected_experts.value();
    CHECK_INPUT_AND_TYPE(experts, dl_int32);
    CHECK_DEVICE(experts, fixed);
    TVM_FFI_ICHECK(local_expert_offset >= 0 && local_expert_offset <= INT32_MAX)
        << "local_expert_offset must fit in int32, got " << local_expert_offset;
    p.experts = static_cast<const int32_t*>(experts.data_ptr());
    p.expert_offset = static_cast<int>(local_expert_offset);
    num_items = experts.numel();
  }
  if (num_items == 0 || num_experts == 0 || rows == 0) return;
  int max_smem;
  TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                           fixed.device().device_id),
                    cudaSuccess);
  if (ExpandSmemBytes(cols) > max_smem) {
    int64_t max_cols = 0;
    while (ExpandSmemBytes(max_cols + 2) <= max_smem) max_cols += 2;
    TVM_FFI_THROW(ValueError) << "csf_expand supports up to " << max_cols
                              << " columns on this device, got " << cols;
  }
  ffi::CUDADeviceGuard device_guard(fixed.device().device_id);
  const cudaStream_t stream = get_stream(fixed.device());
  uint8_t* dst = static_cast<uint8_t*>(out.data_ptr());
  if (dest == Dest::kSwizzled128x4) {
    LaunchCsfExpand<Dest::kSwizzled128x4>(p, num_items, dst, stream);
  } else {
    LaunchCsfExpand<Dest::kLinear>(p, num_items, dst, stream);
  }
}
