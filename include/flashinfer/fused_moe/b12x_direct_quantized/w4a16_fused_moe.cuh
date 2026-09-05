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
#include <cuda_fp4.h>
#include <cuda_runtime.h>

#if defined(CUDART_VERSION) && CUDART_VERSION < 12090
#error "B12x Direct W4A16 requires CUDA 12.9 or newer"
#endif

#include <cstdint>

namespace flashinfer::fused_moe {

struct B12xDirectW4A16FusedMoeParams {
  const __nv_bfloat16* hidden_states;
  const uint8_t* gemm1_weights;
  const __nv_bfloat16* gemm1_scales;
  const uint8_t* gemm2_weights;
  const __nv_bfloat16* gemm2_scales;
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

namespace b12x_direct_w4a16_detail {

constexpr int kWarpSize = 32;
constexpr int kVectorElements = 8;
constexpr int kScaleVectorElements = 16;
constexpr int kWarpVectorStride = kWarpSize * kVectorElements;

struct Float8 {
  float4 first;
  float4 second;
};

__device__ __forceinline__ float WarpReduceSum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ Float8 LoadBFloat16x8(const __nv_bfloat16* pointer) {
  const auto* packed = reinterpret_cast<const __nv_bfloat162*>(pointer);
  const float2 first = __bfloat1622float2(packed[0]);
  const float2 second = __bfloat1622float2(packed[1]);
  const float2 third = __bfloat1622float2(packed[2]);
  const float2 fourth = __bfloat1622float2(packed[3]);
  return {make_float4(first.x, first.y, second.x, second.y),
          make_float4(third.x, third.y, fourth.x, fourth.y)};
}

__device__ __forceinline__ Float8 LoadE2M1x8(const uint8_t* pointer, float scale) {
  const uint32_t packed = *reinterpret_cast<const uint32_t*>(pointer);
  __nv_fp4x2_e2m1 pairs[4];
  pairs[0].__x = static_cast<__nv_fp4x2_storage_t>(packed);
  pairs[1].__x = static_cast<__nv_fp4x2_storage_t>(packed >> 8);
  pairs[2].__x = static_cast<__nv_fp4x2_storage_t>(packed >> 16);
  pairs[3].__x = static_cast<__nv_fp4x2_storage_t>(packed >> 24);
  const float2 first = static_cast<float2>(pairs[0]);
  const float2 second = static_cast<float2>(pairs[1]);
  const float2 third = static_cast<float2>(pairs[2]);
  const float2 fourth = static_cast<float2>(pairs[3]);
  return {make_float4(first.x * scale, first.y * scale, second.x * scale, second.y * scale),
          make_float4(third.x * scale, third.y * scale, fourth.x * scale, fourth.y * scale)};
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

__device__ __forceinline__ void Accumulate8(float activation0, float activation1, float activation2,
                                            float activation3, float activation4, float activation5,
                                            float activation6, float activation7,
                                            const Float8& weight, float& accumulator) {
  accumulator = fmaf(activation0, weight.first.x, accumulator);
  accumulator = fmaf(activation1, weight.first.y, accumulator);
  accumulator = fmaf(activation2, weight.first.z, accumulator);
  accumulator = fmaf(activation3, weight.first.w, accumulator);
  accumulator = fmaf(activation4, weight.second.x, accumulator);
  accumulator = fmaf(activation5, weight.second.y, accumulator);
  accumulator = fmaf(activation6, weight.second.z, accumulator);
  accumulator = fmaf(activation7, weight.second.w, accumulator);
}

template <int kOutputs>
__global__ void GateUpSwiGLUKernel(const __nv_bfloat16* __restrict__ hidden_states,
                                   const uint8_t* __restrict__ gemm1_weights,
                                   const __nv_bfloat16* __restrict__ gemm1_scales,
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

  const int packed_hidden = hidden_size / 2;
  const int scale_hidden = hidden_size / kScaleVectorElements;
  const int64_t weight_expert_stride = static_cast<int64_t>(2) * intermediate_size * packed_hidden;
  const int64_t scale_expert_stride = static_cast<int64_t>(2) * intermediate_size * scale_hidden;
  const uint8_t* up_weight = gemm1_weights + static_cast<int64_t>(expert) * weight_expert_stride;
  const uint8_t* gate_weight = up_weight + static_cast<int64_t>(intermediate_size) * packed_hidden;
  const __nv_bfloat16* up_scale = gemm1_scales + static_cast<int64_t>(expert) * scale_expert_stride;
  const __nv_bfloat16* gate_scale =
      up_scale + static_cast<int64_t>(intermediate_size) * scale_hidden;

  float gate[kOutputs] = {};
  float up[kOutputs] = {};
  for (int index = lane * kVectorElements; index < hidden_size; index += kWarpVectorStride) {
    const Float8 activation = LoadBFloat16x8(shared_hidden + index);
#pragma unroll
    for (int output_index = 0; output_index < kOutputs; ++output_index) {
      const int output_column = column + output_index;
      if (output_column < intermediate_size) {
        const int64_t packed_offset =
            static_cast<int64_t>(output_column) * packed_hidden + index / 2;
        const int64_t scale_offset =
            static_cast<int64_t>(output_column) * scale_hidden + index / kScaleVectorElements;
        const Float8 gate_values =
            LoadE2M1x8(gate_weight + packed_offset, __bfloat162float(gate_scale[scale_offset]));
        const Float8 up_values =
            LoadE2M1x8(up_weight + packed_offset, __bfloat162float(up_scale[scale_offset]));
        Accumulate8(activation.first.x, activation.first.y, activation.first.z, activation.first.w,
                    activation.second.x, activation.second.y, activation.second.z,
                    activation.second.w, gate_values, gate[output_index]);
        Accumulate8(activation.first.x, activation.first.y, activation.first.z, activation.first.w,
                    activation.second.x, activation.second.y, activation.second.z,
                    activation.second.w, up_values, up[output_index]);
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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <int kOutputs>
__global__ void DownFusedTopKKernel(
    const __nv_bfloat16* __restrict__ intermediate, const uint8_t* __restrict__ gemm2_weights,
    const __nv_bfloat16* __restrict__ gemm2_scales, const int32_t* __restrict__ topk_ids,
    const int32_t* __restrict__ expert_map, const float* __restrict__ topk_weights,
    __nv_bfloat16* __restrict__ output, int num_tokens, int topk, int num_local_experts,
    int expert_map_items, int hidden_size, int intermediate_size) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif
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
  const int packed_intermediate = intermediate_size / 2;
  const int scale_intermediate = intermediate_size / kScaleVectorElements;
  const int64_t weight_expert_stride = static_cast<int64_t>(hidden_size) * packed_intermediate;
  const int64_t scale_expert_stride = static_cast<int64_t>(hidden_size) * scale_intermediate;
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
    const uint8_t* weight = gemm2_weights + static_cast<int64_t>(expert) * weight_expert_stride;
    const __nv_bfloat16* scales = gemm2_scales + static_cast<int64_t>(expert) * scale_expert_stride;
    float result[kOutputs] = {};
    for (int index = lane * kVectorElements; index < intermediate_size;
         index += kWarpVectorStride) {
      const Float8 activation = LoadBFloat16x8(input_row + index);
#pragma unroll
      for (int output_index = 0; output_index < kOutputs; ++output_index) {
        const int output_column = column + output_index;
        if (output_column < hidden_size) {
          const int64_t packed_offset =
              static_cast<int64_t>(output_column) * packed_intermediate + index / 2;
          const int64_t scale_offset = static_cast<int64_t>(output_column) * scale_intermediate +
                                       index / kScaleVectorElements;
          const Float8 weight_values =
              LoadE2M1x8(weight + packed_offset, __bfloat162float(scales[scale_offset]));
          Accumulate8(activation.first.x, activation.first.y, activation.first.z,
                      activation.first.w, activation.second.x, activation.second.y,
                      activation.second.z, activation.second.w, weight_values,
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
inline cudaError_t LaunchWithOutputs(const B12xDirectW4A16FusedMoeParams& params,
                                     cudaStream_t stream) {
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
      params.hidden_states, params.gemm1_weights, params.gemm1_scales, params.topk_ids,
      params.expert_map, params.intermediate, routed_rows, params.topk, params.num_local_experts,
      params.expert_map_items, params.hidden_size, params.intermediate_size);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  cudaLaunchConfig_t config{};
  cudaLaunchAttribute attributes[1]{};
  config.gridDim = dim3(down_blocks);
  config.blockDim = dim3(params.num_threads);
  config.dynamicSmemBytes = down_shared_bytes;
  config.stream = stream;
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attributes;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(&config, &DownFusedTopKKernel<kOutputs>, params.intermediate,
                            params.gemm2_weights, params.gemm2_scales, params.topk_ids,
                            params.expert_map, params.topk_weights, params.output,
                            params.num_tokens, params.topk, params.num_local_experts,
                            params.expert_map_items, params.hidden_size, params.intermediate_size);
}

}  // namespace b12x_direct_w4a16_detail

inline cudaError_t LaunchB12xDirectW4A16FusedMoe(const B12xDirectW4A16FusedMoeParams& params,
                                                 cudaStream_t stream) {
  switch (params.outputs_per_warp) {
    case 1:
      return b12x_direct_w4a16_detail::LaunchWithOutputs<1>(params, stream);
    case 2:
      return b12x_direct_w4a16_detail::LaunchWithOutputs<2>(params, stream);
    case 4:
      return b12x_direct_w4a16_detail::LaunchWithOutputs<4>(params, stream);
    case 8:
      return b12x_direct_w4a16_detail::LaunchWithOutputs<8>(params, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace flashinfer::fused_moe
