// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gpu_iface/backend/hip/mma_hip.h"
#include "gpu_iface/gpu_runtime_compat.hpp"

namespace flashinfer::gpu_iface::debug_utils::hip {

enum class MatrixLayout { A, B };

/// @brief Initializes a 2D LDS array with lexicographical values (0, 1, 2, ...).
/// @param lds_array Pointer to the shared memory array.
/// @param dimY The height of the 2D array.
/// @param dimX The width of the 2D array.
__device__ void lexicographic_init_lds_array(half* lds_array, uint32_t dimY, uint32_t dimX) {
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int y = 0; y < dimY; ++y) {
      for (int x = 0; x < dimX; ++x) {
        lds_array[y * dimX + x] = __half(y * dimX + x);
      }
    }
  }
  __syncthreads();
}

/// @brief Loads a 16x16 tile from LDS into registers using the A-matrix layout pattern.
/// @details Each thread `T_(16*c + r)` loads a 1x4 horizontal fragment from `LDS[r, 4*c : 4*c+3]`.
/// @tparam T The data type of the LDS array, must be `__half`.
/// @param lds_array Pointer to the shared memory array.
/// @param R Pointer to the thread's registers (uint32_t[2]).
/// @param dimX The width of the LDS array.
template <typename T>
__device__ void load_amatrix_layout(T* lds_array, uint32_t* R, uint32_t dimX) {
  static_assert(std::is_same_v<T, __half>, "Only supported for __half types");
  const int lane_id = threadIdx.x % 64;
  const int row = lane_id % 16;
  const int col_start = (lane_id / 16) * 4;

  auto offset = lds_array + row * dimX + col_start;
  mma_impl::hip::load_fragment(R, offset);
}

/// @brief Loads a 16x16 tile from LDS into registers using the B-matrix layout pattern.
/// @details Uses an efficient load-and-transpose strategy. A 4x4 block of threads loads a
///          contiguous 4x4 tile from LDS and then performs an in-register transpose,
///          resulting in each thread holding a column fragment.
/// @tparam T The data type of the LDS array, must be `__half`.
/// @param arr Pointer to the shared memory array.
/// @param R Pointer to the thread's registers (uint32_t[2]).
/// @param dimY The height of the LDS array.
template <typename T>
__device__ void load_bmatrix_layout(T* arr, uint32_t* R, uint32_t dimY) {
  static_assert(std::is_same_v<T, __half>, "Only supported for __half types");
  const int lane_id = threadIdx.x % 64;
  int b_idx = ((lane_id % 4) + 4 * (lane_id / 16)) * dimY + ((lane_id % 16) / 4) * 4;
  mma_impl::hip::load_quad_transposed_fragment<__half>(R, &arr[b_idx]);
}

/// @brief Prints the four `half` values held in a thread's registers.
/// @tparam T The data type to interpret the registers as, must be `__half`.
/// @param R Pointer to the thread's registers (uint32_t[2]).
template <typename T>
__device__ void print_register(uint32_t* R) {
  static_assert(std::is_same_v<T, __half>, "Only supported for __half types");
  auto values = reinterpret_cast<__half*>(R);
  printf("[%5.1f %5.1f %5.1f %5.1f]\n", __half2float(values[0]), __half2float(values[1]),
         __half2float(values[2]), __half2float(values[3]));
}

/// @brief Prints a 2D LDS array to the console from a single thread.
/// @tparam T The data type of the LDS array, must be `__half`.
/// @param lds_array Pointer to the shared memory array.
/// @param dimY The height of the 2D array.
/// @param dimX The width of the 2D array.
template <typename T>
__device__ void print_lds_array(T* lds_array, uint32_t dimY, uint32_t dimX) {
  static_assert(std::is_same_v<T, __half>, "Only supported for __half types");
  if (threadIdx.x == 0) {
    printf("LDS Array (%dx%d):\n", dimX, dimY);
    for (int y = 0; y < dimY; ++y) {
      for (int x = 0; x < dimX; ++x) {
        printf("%5.1f ", __half2float(lds_array[y * dimX + x]));
      }
      printf("\n");
    }
    printf("\n");
  }
  __syncthreads();
}

/// @brief Writes the 4 `half` values from each thread's registers back to LDS.
/// @details This function is the inverse of the `load_*_layout` functions. It materializes
///          the in-register matrix layout into shared memory.
///
///          A-Layout Pattern:
///          Each thread `T_(16*c + r)` writes its 4 values to `LDS[r, 4*c : 4*c+3]`.
///          This reconstructs the standard row-major matrix.
///
///          B-Layout Pattern:
///          Each thread `T_(16*br + 4*bc + ti)` (where br=block_row, bc=block_col,
///          ti=thread_in_block) writes its 4 values to `LDS[4*br + ti, 4*bc : 4*bc+3]`. This
///          creates a block-transposed matrix in shared memory.
/// @tparam T The data type of the LDS array, must be `__half`.
/// @param R Pointer to the thread's registers (uint32_t[2]).
/// @param lds_array Pointer to the shared memory array.
/// @param dimY The height of the LDS array.
/// @param dimX The width of the LDS array.
/// @param layout The target memory layout (A or B) to use for writing.
template <typename T>
__device__ void write_matrix_frag_to_lds(const uint32_t* R, T* lds_array, uint32_t dimY,
                                         uint32_t dimX, MatrixLayout layout) {
  static_assert(std::is_same_v<T, __half>, "Only supported for __half types");

  const int lane_id = threadIdx.x % 64;
  const T* values = reinterpret_cast<const T*>(R);
  int row, col_start;

  if (layout == MatrixLayout::A) {
    // A-matrix layout: each thread owns a 1x4 strip of a row
    row = lane_id % 16;
    col_start = (lane_id / 16) * 4;
  } else {  // MatrixLayout::B
    // B-matrix layout: each thread owns a 1x4 strip of a column block
    const uint32_t block_row = (lane_id % 16) / 4;
    const uint32_t block_col = (lane_id / 16);
    const uint32_t thread_in_block = lane_id % 4;
    row = block_col * 4 + thread_in_block;
    col_start = block_row * 4;
  }

  half* offset = lds_array + row * dimX + col_start;
  offset[0] = values[0];
  offset[1] = values[1];
  offset[2] = values[2];
  offset[3] = values[3];
}

}  // namespace flashinfer::gpu_iface::debug_utils::hip
