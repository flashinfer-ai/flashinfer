/*
 * Copyright (c) 2026 by FlashInfer team.
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
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer::fused_moe {

struct Sm120DirectFusedMoeParams {
  const __nv_bfloat16* hidden_states;
  const __nv_bfloat16* gemm1_weights;
  const __nv_bfloat16* gemm2_weights;
  const int32_t* topk_ids;
  const int32_t* expert_map;
  const float* topk_weights;
  __nv_bfloat16* intermediate;
  __nv_bfloat16* output;
  int32_t num_tokens;
  int32_t topk;
  int32_t num_local_experts;
  int32_t expert_map_items;
  int32_t hidden_size;
  int32_t intermediate_size;
  int32_t outputs_per_warp;
  int32_t num_threads;
};

namespace detail {

constexpr int kWarpSize = 32;
constexpr int kVectorElements = 8;
constexpr int kWarpVectorStride = kWarpSize * kVectorElements;

__device__ __forceinline__ float WarpReduceSum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float LoadBFloat16(const __nv_bfloat16* pointer) {
  return __bfloat162float(*pointer);
}

__device__ __forceinline__ float4 LoadBFloat16x4(const __nv_bfloat16* pointer) {
  const auto* packed = reinterpret_cast<const __nv_bfloat162*>(pointer);
  const float2 first = __bfloat1622float2(packed[0]);
  const float2 second = __bfloat1622float2(packed[1]);
  return make_float4(first.x, first.y, second.x, second.y);
}

struct Float8 {
  float4 first;
  float4 second;
};

__device__ __forceinline__ Float8 LoadBFloat16x8(const __nv_bfloat16* pointer) {
  return {LoadBFloat16x4(pointer), LoadBFloat16x4(pointer + 4)};
}

__device__ __forceinline__ int ResolveExpert(int global_expert, const int32_t* expert_map,
                                             int expert_map_items, int num_local_experts) {
  if (global_expert < 0) {
    return -1;
  }
  const int local_expert =
      expert_map_items == 0 ? global_expert
                            : (global_expert < expert_map_items ? expert_map[global_expert] : -1);
  return local_expert >= 0 && local_expert < num_local_experts ? local_expert : -1;
}

template <int kOutputs>
__global__ void GateUpSwiGLUKernel(const __nv_bfloat16* __restrict__ hidden_states,
                                   const __nv_bfloat16* __restrict__ gemm1_weights,
                                   const int32_t* __restrict__ topk_ids,
                                   const int32_t* __restrict__ expert_map,
                                   __nv_bfloat16* __restrict__ intermediate, int routed_rows,
                                   int topk, int num_local_experts, int expert_map_items,
                                   int hidden_size, int intermediate_size) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  auto* shared_hidden = reinterpret_cast<__nv_bfloat16*>(shared_storage);

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const int column_groups = (intermediate_size + kOutputs - 1) / kOutputs;
  const int blocks_per_row = (column_groups + warps - 1) / warps;
  const int row = blockIdx.x / blocks_per_row;
  const int column_group = (blockIdx.x - row * blocks_per_row) * warps + warp;
  if (row >= routed_rows) {
    return;
  }

  const int column = column_group * kOutputs;
  const int token = row / topk;
  const int expert = ResolveExpert(topk_ids[row], expert_map, expert_map_items, num_local_experts);
  if (expert < 0) {
    if (column_group < column_groups && lane == 0) {
#pragma unroll
      for (int output_index = 0; output_index < kOutputs; ++output_index) {
        const int output_column = column + output_index;
        if (output_column < intermediate_size) {
          intermediate[static_cast<int64_t>(row) * intermediate_size + output_column] =
              __float2bfloat16_rn(0.0f);
        }
      }
    }
    return;
  }

  const __nv_bfloat16* hidden_row = hidden_states + static_cast<int64_t>(token) * hidden_size;
  for (int index = threadIdx.x; index < hidden_size; index += blockDim.x) {
    shared_hidden[index] = hidden_row[index];
  }
  __syncthreads();
  if (column_group >= column_groups) {
    return;
  }

  // FlashInfer's unquantized MoE layout is [up || gate].
  const int64_t expert_stride = static_cast<int64_t>(2) * intermediate_size * hidden_size;
  const __nv_bfloat16* up_base = gemm1_weights + static_cast<int64_t>(expert) * expert_stride;
  const __nv_bfloat16* gate_base = up_base + static_cast<int64_t>(intermediate_size) * hidden_size;

  float gate[kOutputs] = {};
  float up[kOutputs] = {};
  const int hidden_vector_end = hidden_size - hidden_size % kVectorElements;
  for (int index = lane * kVectorElements; index < hidden_vector_end; index += kWarpVectorStride) {
    const Float8 activation = LoadBFloat16x8(shared_hidden + index);
#pragma unroll
    for (int output_index = 0; output_index < kOutputs; ++output_index) {
      const int output_column = column + output_index;
      if (output_column < intermediate_size) {
        const Float8 gate_weight =
            LoadBFloat16x8(gate_base + static_cast<int64_t>(output_column) * hidden_size + index);
        const Float8 up_weight =
            LoadBFloat16x8(up_base + static_cast<int64_t>(output_column) * hidden_size + index);
        gate[output_index] = fmaf(activation.first.x, gate_weight.first.x, gate[output_index]);
        gate[output_index] = fmaf(activation.first.y, gate_weight.first.y, gate[output_index]);
        gate[output_index] = fmaf(activation.first.z, gate_weight.first.z, gate[output_index]);
        gate[output_index] = fmaf(activation.first.w, gate_weight.first.w, gate[output_index]);
        gate[output_index] = fmaf(activation.second.x, gate_weight.second.x, gate[output_index]);
        gate[output_index] = fmaf(activation.second.y, gate_weight.second.y, gate[output_index]);
        gate[output_index] = fmaf(activation.second.z, gate_weight.second.z, gate[output_index]);
        gate[output_index] = fmaf(activation.second.w, gate_weight.second.w, gate[output_index]);
        up[output_index] = fmaf(activation.first.x, up_weight.first.x, up[output_index]);
        up[output_index] = fmaf(activation.first.y, up_weight.first.y, up[output_index]);
        up[output_index] = fmaf(activation.first.z, up_weight.first.z, up[output_index]);
        up[output_index] = fmaf(activation.first.w, up_weight.first.w, up[output_index]);
        up[output_index] = fmaf(activation.second.x, up_weight.second.x, up[output_index]);
        up[output_index] = fmaf(activation.second.y, up_weight.second.y, up[output_index]);
        up[output_index] = fmaf(activation.second.z, up_weight.second.z, up[output_index]);
        up[output_index] = fmaf(activation.second.w, up_weight.second.w, up[output_index]);
      }
    }
  }
  // Defensive tail handling; the launcher currently requires hidden_size % 8 == 0.
  for (int index = hidden_vector_end + lane; index < hidden_size; index += 32) {
    const float activation = LoadBFloat16(shared_hidden + index);
#pragma unroll
    for (int output_index = 0; output_index < kOutputs; ++output_index) {
      const int output_column = column + output_index;
      if (output_column < intermediate_size) {
        gate[output_index] = fmaf(
            activation,
            LoadBFloat16(gate_base + static_cast<int64_t>(output_column) * hidden_size + index),
            gate[output_index]);
        up[output_index] =
            fmaf(activation,
                 LoadBFloat16(up_base + static_cast<int64_t>(output_column) * hidden_size + index),
                 up[output_index]);
      }
    }
  }
#pragma unroll
  for (int output_index = 0; output_index < kOutputs; ++output_index) {
    gate[output_index] = WarpReduceSum(gate[output_index]);
    up[output_index] = WarpReduceSum(up[output_index]);
  }

  if (lane == 0) {
#pragma unroll
    for (int output_index = 0; output_index < kOutputs; ++output_index) {
      const int output_column = column + output_index;
      if (output_column < intermediate_size) {
        const float gate_value = gate[output_index];
        const float silu = gate_value / (1.0f + __expf(-gate_value));
        intermediate[static_cast<int64_t>(row) * intermediate_size + output_column] =
            __float2bfloat16_rn(silu * up[output_index]);
      }
    }
  }
}

template <int kOutputs>
__global__ void DownFusedTopKKernel(
    const __nv_bfloat16* __restrict__ intermediate, const __nv_bfloat16* __restrict__ gemm2_weights,
    const int32_t* __restrict__ topk_ids, const int32_t* __restrict__ expert_map,
    const float* __restrict__ topk_weights, __nv_bfloat16* __restrict__ output, int num_tokens,
    int topk, int num_local_experts, int expert_map_items, int hidden_size, int intermediate_size) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  auto* shared_intermediate = reinterpret_cast<__nv_bfloat16*>(shared_storage);

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const int column_groups = (hidden_size + kOutputs - 1) / kOutputs;
  const int blocks_per_token = (column_groups + warps - 1) / warps;
  const int token = blockIdx.x / blocks_per_token;
  const int column_group = (blockIdx.x - token * blocks_per_token) * warps + warp;
  if (token >= num_tokens) {
    return;
  }

  const int64_t token_intermediate_items = static_cast<int64_t>(topk) * intermediate_size;
  const __nv_bfloat16* token_intermediate =
      intermediate + static_cast<int64_t>(token) * topk * intermediate_size;
  for (int64_t index = threadIdx.x; index < token_intermediate_items; index += blockDim.x) {
    shared_intermediate[index] = token_intermediate[index];
  }
  __syncthreads();
  if (column_group >= column_groups) {
    return;
  }

  const int column = column_group * kOutputs;
  const int64_t expert_stride = static_cast<int64_t>(hidden_size) * intermediate_size;
  float combined[kOutputs] = {};
  for (int slot = 0; slot < topk; ++slot) {
    const int row = token * topk + slot;
    const int expert =
        ResolveExpert(topk_ids[row], expert_map, expert_map_items, num_local_experts);
    if (expert < 0) {
      continue;
    }
    const __nv_bfloat16* input_row =
        shared_intermediate + static_cast<int64_t>(slot) * intermediate_size;
    const __nv_bfloat16* weight_base = gemm2_weights + static_cast<int64_t>(expert) * expert_stride;
    float result[kOutputs] = {};
    const int intermediate_vector_end = intermediate_size - intermediate_size % kVectorElements;
    for (int index = lane * kVectorElements; index < intermediate_vector_end;
         index += kWarpVectorStride) {
      const Float8 activation = LoadBFloat16x8(input_row + index);
#pragma unroll
      for (int output_index = 0; output_index < kOutputs; ++output_index) {
        const int output_column = column + output_index;
        if (output_column < hidden_size) {
          const Float8 weight = LoadBFloat16x8(
              weight_base + static_cast<int64_t>(output_column) * intermediate_size + index);
          result[output_index] = fmaf(activation.first.x, weight.first.x, result[output_index]);
          result[output_index] = fmaf(activation.first.y, weight.first.y, result[output_index]);
          result[output_index] = fmaf(activation.first.z, weight.first.z, result[output_index]);
          result[output_index] = fmaf(activation.first.w, weight.first.w, result[output_index]);
          result[output_index] = fmaf(activation.second.x, weight.second.x, result[output_index]);
          result[output_index] = fmaf(activation.second.y, weight.second.y, result[output_index]);
          result[output_index] = fmaf(activation.second.z, weight.second.z, result[output_index]);
          result[output_index] = fmaf(activation.second.w, weight.second.w, result[output_index]);
        }
      }
    }
    // Defensive tail handling; the launcher currently requires intermediate_size % 8 == 0.
    for (int index = intermediate_vector_end + lane; index < intermediate_size; index += 32) {
      const float activation = LoadBFloat16(input_row + index);
#pragma unroll
      for (int output_index = 0; output_index < kOutputs; ++output_index) {
        const int output_column = column + output_index;
        if (output_column < hidden_size) {
          result[output_index] =
              fmaf(activation,
                   LoadBFloat16(weight_base +
                                static_cast<int64_t>(output_column) * intermediate_size + index),
                   result[output_index]);
        }
      }
    }
#pragma unroll
    for (int output_index = 0; output_index < kOutputs; ++output_index) {
      result[output_index] = WarpReduceSum(result[output_index]);
      if (lane == 0) {
        combined[output_index] += result[output_index] * topk_weights[row];
      }
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int output_index = 0; output_index < kOutputs; ++output_index) {
      const int output_column = column + output_index;
      if (output_column < hidden_size) {
        output[static_cast<int64_t>(token) * hidden_size + output_column] =
            __float2bfloat16_rn(combined[output_index]);
      }
    }
  }
}

template <int kOutputs>
inline cudaError_t LaunchWithOutputs(const Sm120DirectFusedMoeParams& params, cudaStream_t stream) {
  const int warps = params.num_threads / 32;
  const int routed_rows = params.num_tokens * params.topk;
  const int gate_groups = (params.intermediate_size + kOutputs - 1) / kOutputs;
  const int down_groups = (params.hidden_size + kOutputs - 1) / kOutputs;
  const int gate_blocks = routed_rows * ((gate_groups + warps - 1) / warps);
  const int down_blocks = params.num_tokens * ((down_groups + warps - 1) / warps);
  const size_t gate_shared_bytes = static_cast<size_t>(params.hidden_size) * sizeof(__nv_bfloat16);
  const size_t down_shared_bytes =
      static_cast<size_t>(params.topk) * params.intermediate_size * sizeof(__nv_bfloat16);

  GateUpSwiGLUKernel<kOutputs><<<gate_blocks, params.num_threads, gate_shared_bytes, stream>>>(
      params.hidden_states, params.gemm1_weights, params.topk_ids, params.expert_map,
      params.intermediate, routed_rows, params.topk, params.num_local_experts,
      params.expert_map_items, params.hidden_size, params.intermediate_size);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  DownFusedTopKKernel<kOutputs><<<down_blocks, params.num_threads, down_shared_bytes, stream>>>(
      params.intermediate, params.gemm2_weights, params.topk_ids, params.expert_map,
      params.topk_weights, params.output, params.num_tokens, params.topk, params.num_local_experts,
      params.expert_map_items, params.hidden_size, params.intermediate_size);
  return cudaGetLastError();
}

}  // namespace detail

inline cudaError_t LaunchSm120DirectFusedMoe(const Sm120DirectFusedMoeParams& params,
                                             cudaStream_t stream) {
  switch (params.outputs_per_warp) {
    case 1:
      return detail::LaunchWithOutputs<1>(params, stream);
    case 2:
      return detail::LaunchWithOutputs<2>(params, stream);
    case 4:
      return detail::LaunchWithOutputs<4>(params, stream);
    case 8:
      return detail::LaunchWithOutputs<8>(params, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace flashinfer::fused_moe
