/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "moe_gemm_mixed_utils.h"

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels {

/////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void interleave_fp4_weights_for_sm90_mixed_gemm_kernel(uint8_t* fp4_weight,
                                                                  uint8_t* fp4_weight_interleaved,
                                                                  int const rows, int const cols) {
  for (int block_id = blockIdx.x; block_id < rows / 2; block_id += gridDim.x) {
    for (int partition_id = threadIdx.y; partition_id < cols / 64; partition_id += blockDim.y) {
      int lane_id = threadIdx.x;
      int row_id = block_id / 8 * 16 + block_id % 8;

      int mma_id = lane_id / 4;
      int dst_row_id = row_id + (mma_id % 2) * 8;

      int interleaved_lane_id = lane_id / 8 * 16 + (lane_id % 4) * 4;

      int col_id = partition_id * 32 + mma_id * 8 + lane_id % 4;
      int dst_col_id = partition_id * 32 + interleaved_lane_id;

      int first_fp4_id = row_id * cols / 2 + col_id;
      int second_fp4_id = (row_id + 8) * cols / 2 + col_id;
      int third_fp4_id = first_fp4_id + 4;
      int fourth_fp4_id = second_fp4_id + 4;

      uint32_t fp4x8_raw = 0;
      uint8_t* fp4x2 = reinterpret_cast<uint8_t*>(&fp4x8_raw);
      fp4x2[0] = fp4_weight[first_fp4_id];
      fp4x2[1] = fp4_weight[second_fp4_id];
      fp4x2[2] = fp4_weight[third_fp4_id];
      fp4x2[3] = fp4_weight[fourth_fp4_id];

      uint32_t fp4x8_interleaved = 0;
      uint32_t mask;

      mask = 0b00000000000000000000000010000000;
      fp4x8_interleaved |= (fp4x8_raw & mask) << 24;
      mask = 0b00000000000000000000000001110000;
      fp4x8_interleaved |= (fp4x8_raw & mask) << 18;
      mask = 0b00000000000000000000000000001000;
      fp4x8_interleaved |= (fp4x8_raw & mask) << 12;
      mask = 0b00000000000000000000000000000111;
      fp4x8_interleaved |= (fp4x8_raw & mask) << 6;

      mask = 0b00000000000000001000000000000000;
      fp4x8_interleaved |= (fp4x8_raw & mask) << 13;
      mask = 0b00000000000000000111000000000000;
      fp4x8_interleaved |= (fp4x8_raw & mask) << 7;
      mask = 0b00000000000000000000100000000000;
      fp4x8_interleaved |= (fp4x8_raw & mask) << 1;
      mask = 0b00000000000000000000011100000000;
      fp4x8_interleaved |= (fp4x8_raw & mask) >> 5;

      mask = 0b00000000100000000000000000000000;
      fp4x8_interleaved |= (fp4x8_raw & mask) << 2;
      mask = 0b00000000011100000000000000000000;
      fp4x8_interleaved |= (fp4x8_raw & mask) >> 4;
      mask = 0b00000000000010000000000000000000;
      fp4x8_interleaved |= (fp4x8_raw & mask) >> 10;
      mask = 0b00000000000001110000000000000000;
      fp4x8_interleaved |= (fp4x8_raw & mask) >> 16;

      mask = 0b10000000000000000000000000000000;
      fp4x8_interleaved |= (fp4x8_raw & mask) >> 1;
      mask = 0b00010000000000000000000000000000;
      fp4x8_interleaved |= (fp4x8_raw & mask) << 1;
      mask = 0b01100000000000000000000000000000;
      fp4x8_interleaved |= (fp4x8_raw & mask) >> 3;
      mask = 0b00001000000000000000000000000000;
      fp4x8_interleaved |= (fp4x8_raw & mask) >> 13;
      mask = 0b00000001000000000000000000000000;
      fp4x8_interleaved |= (fp4x8_raw & mask) >> 11;
      mask = 0b00000110000000000000000000000000;
      fp4x8_interleaved |= (fp4x8_raw & mask) >> 15;

      int dst_id = dst_row_id * cols / 2 + dst_col_id;
      uint8_t* fp4x2_interleaved = reinterpret_cast<uint8_t*>(&fp4x8_interleaved);

      fp4_weight_interleaved[dst_id] = fp4x2_interleaved[0];
      fp4_weight_interleaved[dst_id + 1] = fp4x2_interleaved[1];
      fp4_weight_interleaved[dst_id + 2] = fp4x2_interleaved[2];
      fp4_weight_interleaved[dst_id + 3] = fp4x2_interleaved[3];
    }
  }
}

__global__ void interleave_int4_weights_for_sm90_mixed_gemm_kernel(uint8_t* int4_weight,
                                                                   uint8_t* int4_weight_interleaved,
                                                                   int const rows, int const cols) {
  uint16_t* uint16_ptr = reinterpret_cast<uint16_t*>(int4_weight);
  uint16_t* uint16_interleaved_ptr = reinterpret_cast<uint16_t*>(int4_weight_interleaved);

  for (int block_id = blockIdx.x; block_id < rows / 2; block_id += gridDim.x) {
    for (int partition_id = threadIdx.y; partition_id < cols / 64; partition_id += blockDim.y) {
      int lane_id = threadIdx.x;

      int row_id = block_id / 8 * 16 + block_id % 8;
      int dst_row_id = row_id + (lane_id % 8) / 4 * 8;

      int mma_id = lane_id / 8;
      int interleaved_lane_id = mma_id * 8 + lane_id % 4 * 2;

      int col_id = partition_id * 16 + lane_id;
      int dst_col_id = partition_id * 16 + interleaved_lane_id;

      int src_id_a = row_id * cols / 4 + col_id;
      int src_id_b = (row_id + 8) * cols / 4 + col_id;

      uint16_t int4x2_a = uint16_ptr[src_id_a];
      uint16_t int4x2_b = uint16_ptr[src_id_b];

      int dst_id = dst_row_id * cols / 4 + dst_col_id;

      uint16_interleaved_ptr[dst_id] = int4x2_a;
      uint16_interleaved_ptr[dst_id + 1] = int4x2_b;
    }
  }
}

__device__ __forceinline__ uint32_t preprocess_fp4x8_signs_for_fp8(uint32_t fp4x8) {
  uint32_t const em = fp4x8 & 0x77777777U;
  uint32_t const signs = ((fp4x8 & 0x00000008U) << 4U) | ((fp4x8 & 0x00000080U) << 8U) |
                         ((fp4x8 & 0x00000800U) << 12U) | ((fp4x8 & 0x00008000U) << 16U) |
                         ((fp4x8 & 0x00080000U) >> 16U) | ((fp4x8 & 0x00800000U) >> 12U) |
                         ((fp4x8 & 0x08000000U) >> 8U) | ((fp4x8 & 0x80000000U) >> 4U);
  return em | signs;
}

__global__ void interleave_fp4_fp8_weights_for_sm90_mixed_gemm_kernel(
    uint8_t* fp4_weight, uint8_t* fp4_weight_interleaved, int const rows, int const cols) {
  uint16_t* uint16_ptr = reinterpret_cast<uint16_t*>(fp4_weight);
  uint16_t* uint16_interleaved_ptr = reinterpret_cast<uint16_t*>(fp4_weight_interleaved);

  for (int block_id = blockIdx.x; block_id < rows / 2; block_id += gridDim.x) {
    for (int partition_id = threadIdx.y; partition_id < cols / 64; partition_id += blockDim.y) {
      int lane_id = threadIdx.x;

      int row_id = block_id / 8 * 16 + block_id % 8;
      int dst_row_id = row_id + (lane_id % 8) / 4 * 8;

      int mma_id = lane_id / 8;
      int interleaved_lane_id = mma_id * 8 + lane_id % 4 * 2;

      int col_id = partition_id * 16 + lane_id;
      int dst_col_id = partition_id * 16 + interleaved_lane_id;

      int src_id_a = row_id * cols / 4 + col_id;
      int src_id_b = (row_id + 8) * cols / 4 + col_id;

      uint16_t packed_4b_a = uint16_ptr[src_id_a];
      uint16_t packed_4b_b = uint16_ptr[src_id_b];

      uint32_t fp4x8 = uint32_t(packed_4b_a) | (uint32_t(packed_4b_b) << 16U);
      fp4x8 = preprocess_fp4x8_signs_for_fp8(fp4x8);
      packed_4b_a = uint16_t(fp4x8);
      packed_4b_b = uint16_t(fp4x8 >> 16U);

      int dst_id = dst_row_id * cols / 4 + dst_col_id;
      uint16_interleaved_ptr[dst_id] = packed_4b_a;
      uint16_interleaved_ptr[dst_id + 1] = packed_4b_b;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void interleave_fp4_weights_for_sm90_mixed_gemm(uint8_t* fp4_weight,
                                                uint8_t* fp4_weight_interleaved, int const rows,
                                                int const cols, cudaStream_t stream) {
  dim3 block(16, 32);
  interleave_fp4_weights_for_sm90_mixed_gemm_kernel<<<1024, block, 0, stream>>>(
      fp4_weight, fp4_weight_interleaved, rows, cols);
}

void interleave_fp4_fp8_weights_for_sm90_mixed_gemm(uint8_t* fp4_weight,
                                                    uint8_t* fp4_weight_interleaved, int const rows,
                                                    int const cols, cudaStream_t stream) {
  dim3 block(16, 32);
  interleave_fp4_fp8_weights_for_sm90_mixed_gemm_kernel<<<1024, block, 0, stream>>>(
      fp4_weight, fp4_weight_interleaved, rows, cols);
}

void interleave_int4_weights_for_sm90_mixed_gemm(uint8_t* int4_weight,
                                                 uint8_t* int4_weight_interleaved, int const rows,
                                                 int const cols, cudaStream_t stream) {
  dim3 block(16, 32);
  interleave_int4_weights_for_sm90_mixed_gemm_kernel<<<1024, block, 0, stream>>>(
      int4_weight, int4_weight_interleaved, rows, cols);
}

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace tensorrt_llm
