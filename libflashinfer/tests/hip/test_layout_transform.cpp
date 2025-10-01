// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#include <gtest/gtest.h>
#include <stdio.h>

#include "gpu_iface/backend/hip/mma_debug_utils_hip.h"
#include "gpu_iface/backend/hip/mma_hip.h"
#include "gpu_iface/gpu_runtime_compat.hpp"

namespace {

using namespace flashinfer::gpu_iface::debug_utils::hip;

/// Kernel to test the result of load_amatrix_layout
__global__ void get_a_layout_fragments_kernel(half* output) {
  uint32_t registers[2];
  __shared__ half lds_array[16 * 16];

  lexicographic_init_lds_array(lds_array, 16, 16);
  load_amatrix_layout<__half>(lds_array, registers, 16);

  const __half* values = reinterpret_cast<const __half*>(registers);
  int offset = threadIdx.x * 4;
  output[offset + 0] = values[0];
  output[offset + 1] = values[1];
  output[offset + 2] = values[2];
  output[offset + 3] = values[3];
}

/// Kernel to test the result of load_bmatrix_layout
__global__ void get_b_layout_fragments_kernel(half* output) {
  uint32_t registers[2];
  __shared__ half lds_array[16 * 16];

  // 1. Init LDS with 0..255
  lexicographic_init_lds_array(lds_array, 16, 16);

  // 2. Load registers using B-layout pattern
  load_bmatrix_layout<__half>(lds_array, registers, 16);

  // 3. Write register contents to global memory for validation
  const __half* values = reinterpret_cast<const __half*>(registers);
  int offset = threadIdx.x * 4;
  output[offset + 0] = values[0];
  output[offset + 1] = values[1];
  output[offset + 2] = values[2];
  output[offset + 3] = values[3];
}

/// Kernel to test the full B -> A transformation
__global__ void get_b_to_a_transform_fragments_kernel(half* output) {
  uint32_t registers[2];
  __shared__ half lds_array[16 * 16];

  // 1. Init and load B-layout into registers
  lexicographic_init_lds_array(lds_array, 16, 16);
  load_bmatrix_layout<__half>(lds_array, registers, 16);

  // 2. Apply the B -> A transformation
  flashinfer::gpu_iface::mma_impl::hip::transpose_intra_quad_fragments(registers);
  flashinfer::gpu_iface::mma_impl::hip::transpose_inter_quad_fragments(registers);

  // 3. Write final register contents to global memory
  const __half* values = reinterpret_cast<const __half*>(registers);
  int offset = threadIdx.x * 4;
  output[offset + 0] = values[0];
  output[offset + 1] = values[1];
  output[offset + 2] = values[2];
  output[offset + 3] = values[3];
}

/// Kernel to test the full A -> B transformation
__global__ void get_a_to_b_transform_fragments_kernel(half* output) {
  uint32_t registers[2];
  __shared__ half lds_array[16 * 16];

  // 1. Init and load A-layout into registers
  lexicographic_init_lds_array(lds_array, 16, 16);
  load_amatrix_layout<__half>(lds_array, registers, 16);

  // 2. Apply the A -> B transformation
  flashinfer::gpu_iface::mma_impl::hip::transpose_intra_quad_fragments(registers);
  flashinfer::gpu_iface::mma_impl::hip::transpose_inter_quad_fragments(registers);

  // 3. Write final register contents to global memory
  const __half* values = reinterpret_cast<const __half*>(registers);
  int offset = threadIdx.x * 4;
  output[offset + 0] = values[0];
  output[offset + 1] = values[1];
  output[offset + 2] = values[2];
  output[offset + 3] = values[3];
}

}  // namespace

class LayoutTransformTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Allocate 256 * sizeof(half) for output
    FI_GPU_CALL(hipMalloc(&d_output, 256 * sizeof(half)));
    h_output.resize(256);
  }

  void TearDown() override { FI_GPU_CALL(hipFree(d_output)); }

  half* d_output;
  std::vector<half> h_output;
};

TEST_F(LayoutTransformTest, LoadALayoutIsCorrect) {
  get_a_layout_fragments_kernel<<<1, 64>>>(d_output);
  FI_GPU_CALL(hipMemcpy(h_output.data(), d_output, 256 * sizeof(half), hipMemcpyDeviceToHost));

  // On CPU, compute the expected A-layout fragments
  for (int lane_id = 0; lane_id < 64; ++lane_id) {
    // A-layout: T_(16*c+r) holds row r, columns 4c to 4c+3
    int row = lane_id % 16;
    int col_start = (lane_id / 16) * 4;
    for (int i = 0; i < 4; ++i) {
      float expected_val = (float)(row * 16 + col_start + i);
      float gpu_val = __half2float(h_output[lane_id * 4 + i]);
      EXPECT_EQ(gpu_val, expected_val) << "Mismatch at lane " << lane_id << ", element " << i;
    }
  }
}

TEST_F(LayoutTransformTest, LoadBLayoutIsCorrect) {
  // Launch kernel to get B-layout fragments
  get_b_layout_fragments_kernel<<<1, 64>>>(d_output);
  FI_GPU_CALL(hipMemcpy(h_output.data(), d_output, 256 * sizeof(half), hipMemcpyDeviceToHost));

  // On CPU, compute the expected fragments based on the correct CDNA layout
  for (int lane_id = 0; lane_id < 64; ++lane_id) {
    // Correct mapping from lane_id to block coordinates
    int block_row = lane_id / 16;
    int block_col = (lane_id % 16) / 4;
    int thread_in_block = lane_id % 4;

    // The B-Layout means each thread holds a column fragment.
    // The fragment's column index is determined by the thread's block_col.
    // The fragment's starting row is determined by the thread's block_row.
    // The element within the fragment is determined by the thread_in_block.
    //
    // Example 1, T0 (lane_id=0): br=0, bc=0, tib=0. It holds column 0, elements 0-3.
    // Expected: [M(0,0), M(1,0), M(2,0), M(3,0)] -> [0, 16, 32, 48]
    //
    // Example 2, T17 (lane_id=17): br=1, bc=0, tib=1. It holds column 1, elements 4-7.
    // Expected: [M(4,1), M(5,1), M(6,1), M(7,1)] -> [65, 81, 97, 113]
    for (int i = 0; i < 4; ++i) {
      float expected_val = (float)((block_col * 4 + i) * 16 + (block_row * 4 + thread_in_block));

      // Let's use the original correct logic with the corrected variable names.
      int orig_matrix_col = block_col * 4 + i;
      int orig_matrix_row = block_row * 4 + thread_in_block;
      expected_val = (float)(orig_matrix_row * 16 + orig_matrix_col);

      // The B-layout fragment for a thread is a column segment.
      // T0 gets column 0, elements 0-3 -> [0, 16, 32, 48]
      // T1 gets column 1, elements 0-3 -> [1, 17, 33, 49]
      // T16 gets column 0, elements 4-7 -> [64, 80, 96, 112]
      // T17 gets column 1, elements 4-7 -> [65, 81, 97, 113]
      int frag_col_idx = block_col * 4 + thread_in_block;
      int frag_row_start = block_row * 4;

      expected_val = (frag_row_start + i) * 16 + frag_col_idx;

      float gpu_val = __half2float(h_output[lane_id * 4 + i]);
      EXPECT_EQ(gpu_val, expected_val) << "Mismatch at lane " << lane_id << ", element " << i;
    }
  }
}

TEST_F(LayoutTransformTest, TransformAtoBIsCorrect) {
  get_a_to_b_transform_fragments_kernel<<<1, 64>>>(d_output);
  FI_GPU_CALL(hipMemcpy(h_output.data(), d_output, 256 * sizeof(half), hipMemcpyDeviceToHost));

  // The expected result is the B-layout fragment, same as the LoadBLayoutIsCorrect test
  for (int lane_id = 0; lane_id < 64; ++lane_id) {
    int block_row = lane_id / 16;
    int block_col = (lane_id % 16) / 4;
    int thread_in_block = lane_id % 4;
    int frag_col_idx = block_col * 4 + thread_in_block;
    int frag_row_start = block_row * 4;

    for (int i = 0; i < 4; ++i) {
      float expected_val = (float)((frag_row_start + i) * 16 + frag_col_idx);
      float gpu_val = __half2float(h_output[lane_id * 4 + i]);
      EXPECT_EQ(gpu_val, expected_val) << "Mismatch at lane " << lane_id << ", element " << i;
    }
  }
}

TEST_F(LayoutTransformTest, TransformBtoAIsCorrect) {
  // Launch kernel to get transformed fragments
  get_b_to_a_transform_fragments_kernel<<<1, 64>>>(d_output);
  FI_GPU_CALL(hipMemcpy(h_output.data(), d_output, 256 * sizeof(half), hipMemcpyDeviceToHost));

  // On CPU, compute the expected A-layout fragments
  for (int lane_id = 0; lane_id < 64; ++lane_id) {
    // T0 gets {0,1,2,3}, T1 gets {16,17,18,19}, etc.
    int row = lane_id % 16;
    int col_start = (lane_id / 16) * 4;

    for (int i = 0; i < 4; ++i) {
      float expected_val = (float)(row * 16 + col_start + i);
      float gpu_val = __half2float(h_output[lane_id * 4 + i]);
      EXPECT_EQ(gpu_val, expected_val) << "Mismatch at lane " << lane_id << ", element " << i;
    }
  }
}
