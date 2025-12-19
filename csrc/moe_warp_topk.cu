/*
 * Copyright (c) 2024-2025 by FlashInfer team.
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

#include <cooperative_groups.h>

#include "flashinfer/trtllm/fused_moe/RoutingKernelTopK.cuh"
#include "tvm_ffi_utils.h"

namespace cg = cooperative_groups;
using namespace moe::dev::routing;

// Kernel that uses warp-level reduction for top-k
// Each warp processes one row
// Supports N up to 512 (16 values per thread * 32 threads)
template <typename T, int MaxK, int VecSize>
__global__ void moe_warp_topk_kernel(const T* __restrict__ input,           // [num_rows, num_cols]
                                     T* __restrict__ output_values,         // [num_rows, k]
                                     int32_t* __restrict__ output_indices,  // [num_rows, k]
                                     int32_t num_rows, int32_t num_cols, int32_t k) {
  static_assert(VecSize <= 16, "VecSize must be <= 16");
  static_assert(MaxK < topk::WarpSize, "MaxK must be < 32");

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<topk::WarpSize>(block);

  // Each warp handles one row
  int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / topk::WarpSize;
  int32_t lane_id = threadIdx.x % topk::WarpSize;

  if (warp_id >= num_rows) return;

  const T* row_input = input + warp_id * num_cols;
  T* row_output_values = output_values + warp_id * k;
  int32_t* row_output_indices = output_indices + warp_id * k;

  // Each thread loads VecSize elements
  T values[VecSize];
  int32_t indices[VecSize];
  T min_value = T{-INFINITY};

#pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    int32_t col = i * topk::WarpSize + lane_id;
    if (col < num_cols) {
      values[i] = row_input[col];
      indices[i] = col;
    } else {
      values[i] = min_value;
      indices[i] = col;
    }
  }

  // Output arrays
  T topk_values[MaxK];
  int32_t topk_indices[MaxK];

  // Call warp-level top-k reduction
  topk::reduceTopK<MaxK, T, VecSize>(warp, topk_values, topk_indices, values, indices, min_value,
                                     k);

  // Write output (only first k threads write)
  if (lane_id < k) {
    row_output_values[lane_id] = topk_values[lane_id];
    row_output_indices[lane_id] = topk_indices[lane_id];
  }
}

// Dispatch based on num_cols to select appropriate VecSize
template <typename T, int MaxK>
void launch_moe_warp_topk(const T* input, T* output_values, int32_t* output_indices,
                          int32_t num_rows, int32_t num_cols, int32_t k, cudaStream_t stream) {
  // Each warp processes one row, use multiple warps per block
  constexpr int warps_per_block = 8;
  constexpr int threads_per_block = warps_per_block * topk::WarpSize;
  int num_blocks = (num_rows + warps_per_block - 1) / warps_per_block;

  // Select VecSize based on num_cols (rounded up to handle any size <= VecSize * 32)
  if (num_cols <= 32) {
    moe_warp_topk_kernel<T, MaxK, 1><<<num_blocks, threads_per_block, 0, stream>>>(
        input, output_values, output_indices, num_rows, num_cols, k);
  } else if (num_cols <= 64) {
    moe_warp_topk_kernel<T, MaxK, 2><<<num_blocks, threads_per_block, 0, stream>>>(
        input, output_values, output_indices, num_rows, num_cols, k);
  } else if (num_cols <= 128) {
    moe_warp_topk_kernel<T, MaxK, 4><<<num_blocks, threads_per_block, 0, stream>>>(
        input, output_values, output_indices, num_rows, num_cols, k);
  } else if (num_cols <= 256) {
    moe_warp_topk_kernel<T, MaxK, 8><<<num_blocks, threads_per_block, 0, stream>>>(
        input, output_values, output_indices, num_rows, num_cols, k);
  } else if (num_cols <= 512) {
    moe_warp_topk_kernel<T, MaxK, 16><<<num_blocks, threads_per_block, 0, stream>>>(
        input, output_values, output_indices, num_rows, num_cols, k);
  } else {
    throw std::runtime_error("moe_warp_topk: num_cols must be <= 512");
  }
}

// Dispatch based on k to select appropriate MaxK template parameter
template <typename T>
void dispatch_moe_warp_topk_k(const T* input, T* output_values, int32_t* output_indices,
                              int32_t num_rows, int32_t num_cols, int32_t k, cudaStream_t stream) {
  if (k <= 1) {
    launch_moe_warp_topk<T, 1>(input, output_values, output_indices, num_rows, num_cols, k, stream);
  } else if (k <= 2) {
    launch_moe_warp_topk<T, 2>(input, output_values, output_indices, num_rows, num_cols, k, stream);
  } else if (k <= 4) {
    launch_moe_warp_topk<T, 4>(input, output_values, output_indices, num_rows, num_cols, k, stream);
  } else if (k <= 8) {
    launch_moe_warp_topk<T, 8>(input, output_values, output_indices, num_rows, num_cols, k, stream);
  } else if (k <= 16) {
    launch_moe_warp_topk<T, 16>(input, output_values, output_indices, num_rows, num_cols, k,
                                stream);
  } else {
    throw std::runtime_error("moe_warp_topk: k must be <= 16");
  }
}

void moe_warp_topk(TensorView input, TensorView output_values, TensorView output_indices,
                   int64_t k) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_values);
  CHECK_INPUT(output_indices);
  CHECK_DIM(2, input);
  CHECK_DIM(2, output_values);
  CHECK_DIM(2, output_indices);

  int32_t num_rows = input.size(0);
  int32_t num_cols = input.size(1);

  cudaSetDevice(input.device().device_id);
  auto stream = get_stream(input.device());
  auto dtype = input.dtype();

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP32_FP16(dtype, c_type, [&] {
    dispatch_moe_warp_topk_k<c_type>(static_cast<c_type*>(input.data_ptr()),
                                     static_cast<c_type*>(output_values.data_ptr()),
                                     static_cast<int32_t*>(output_indices.data_ptr()), num_rows,
                                     num_cols, static_cast<int32_t>(k), stream);
    return true;
  });
}
