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
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "flashinfer/fused_moe/b12x_direct_quantized/w4a16_fused_moe.cuh"

namespace flashinfer::fused_moe {

struct B12xDirectNVFP4FusedMoeParams {
  const uint8_t* gemm1_weights;
  const __nv_bfloat16* gemm1_scales;
  const uint8_t* gemm2_weights;
  const __nv_bfloat16* gemm2_scales;
  const int32_t* topk_ids;
  const int32_t* expert_map;
  const float* topk_weights;
  uint8_t* hidden_quantized;
  uint8_t* hidden_scales;
  uint8_t* intermediate_quantized;
  __nv_bfloat16* intermediate_scales;
  __nv_bfloat16* output;
  int32_t num_tokens;
  int32_t topk;
  int32_t num_local_experts;
  int32_t expert_map_items;
  int32_t hidden_size;
  int32_t intermediate_size;
  int32_t outputs_per_warp;
  int32_t num_threads;
  float hidden_global_decode_scale;
  float intermediate_global_encode_scale;
};

namespace b12x_direct_nvfp4_detail {

using b12x_direct_w4a16_detail::Float8;
using b12x_direct_w4a16_detail::LoadE2M1x8;
using b12x_direct_w4a16_detail::ResolveExpert;
using b12x_direct_w4a16_detail::WarpReduceSum;

constexpr int kWarpSize = 32;
constexpr int kVectorElements = 8;
constexpr int kScaleVectorElements = 16;
constexpr int kWarpVectorStride = kWarpSize * kVectorElements;

__device__ __forceinline__ float Dot8(const Float8& lhs, const Float8& rhs) {
  float value = lhs.first.x * rhs.first.x;
  value = fmaf(lhs.first.y, rhs.first.y, value);
  value = fmaf(lhs.first.z, rhs.first.z, value);
  value = fmaf(lhs.first.w, rhs.first.w, value);
  value = fmaf(lhs.second.x, rhs.second.x, value);
  value = fmaf(lhs.second.y, rhs.second.y, value);
  value = fmaf(lhs.second.z, rhs.second.z, value);
  value = fmaf(lhs.second.w, rhs.second.w, value);
  return value;
}

__device__ __forceinline__ float Dot8Packed(const uint8_t* lhs, const uint8_t* rhs) {
  const uint32_t lhs_packed = *reinterpret_cast<const uint32_t*>(lhs);
  const uint32_t rhs_packed = *reinterpret_cast<const uint32_t*>(rhs);
  float result;
  asm volatile(
      "{\n"
      ".reg .b8 a0, a1, a2, a3;\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      ".reg .b32 ah0, ah1, ah2, ah3;\n"
      ".reg .b32 bh0, bh1, bh2, bh3;\n"
      ".reg .f16x2 acc;\n"
      ".reg .b16 lo, hi;\n"
      ".reg .f32 flo, fhi;\n"
      "mov.b32 {a0, a1, a2, a3}, %1;\n"
      "mov.b32 {b0, b1, b2, b3}, %2;\n"
      "cvt.rn.f16x2.e2m1x2 ah0, a0;\n"
      "cvt.rn.f16x2.e2m1x2 ah1, a1;\n"
      "cvt.rn.f16x2.e2m1x2 ah2, a2;\n"
      "cvt.rn.f16x2.e2m1x2 ah3, a3;\n"
      "cvt.rn.f16x2.e2m1x2 bh0, b0;\n"
      "cvt.rn.f16x2.e2m1x2 bh1, b1;\n"
      "cvt.rn.f16x2.e2m1x2 bh2, b2;\n"
      "cvt.rn.f16x2.e2m1x2 bh3, b3;\n"
      "mov.b32 acc, 0;\n"
      "fma.rn.f16x2 acc, ah0, bh0, acc;\n"
      "fma.rn.f16x2 acc, ah1, bh1, acc;\n"
      "fma.rn.f16x2 acc, ah2, bh2, acc;\n"
      "fma.rn.f16x2 acc, ah3, bh3, acc;\n"
      "mov.b32 {lo, hi}, acc;\n"
      "cvt.f32.f16 flo, lo;\n"
      "cvt.f32.f16 fhi, hi;\n"
      "add.f32 %0, flo, fhi;\n"
      "}"
      : "=f"(result)
      : "r"(lhs_packed), "r"(rhs_packed));
  return result;
}

__device__ __forceinline__ float RoundE4M3(float value) {
  const __nv_fp8_e4m3 rounded(value);
  return static_cast<float>(rounded);
}

__device__ __forceinline__ float DecodeE4M3(uint8_t value) {
  __nv_fp8_e4m3 decoded;
  decoded.__x = value;
  return static_cast<float>(decoded);
}

template <int kOutputs>
__global__ void GateUpSwiGLUKernel(
    const uint8_t* __restrict__ hidden_states, const uint8_t* __restrict__ hidden_scales,
    const uint8_t* __restrict__ gemm1_weights, const __nv_bfloat16* __restrict__ gemm1_scales,
    const int32_t* __restrict__ topk_ids, const int32_t* __restrict__ expert_map,
    uint8_t* __restrict__ intermediate_quantized, __nv_bfloat16* __restrict__ intermediate_scales,
    int routed_rows, int topk, int num_local_experts, int expert_map_items, int hidden_size,
    int intermediate_size, float hidden_global_decode_scale,
    float intermediate_global_encode_scale) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif
  extern __shared__ __align__(16) float shared_outputs[];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const int outputs_per_block = warps * kOutputs;
  const int blocks_per_row = intermediate_size / outputs_per_block;
  const int row = blockIdx.x / blocks_per_row;
  const int block_column = (blockIdx.x - row * blocks_per_row) * outputs_per_block;
  if (row >= routed_rows) {
    return;
  }

  const int column = block_column + warp * kOutputs;
  const int token = row / topk;
  const int expert = ResolveExpert(topk_ids[row], expert_map, expert_map_items, num_local_experts);
  const int safe_expert = expert < 0 ? 0 : expert;

  const int packed_hidden = hidden_size / 2;
  const int scale_hidden = hidden_size / kScaleVectorElements;
  const uint8_t* hidden_row = hidden_states + static_cast<int64_t>(token) * packed_hidden;
  const uint8_t* hidden_scale_row = hidden_scales + static_cast<int64_t>(token) * scale_hidden;
  const int64_t weight_expert_stride = static_cast<int64_t>(2) * intermediate_size * packed_hidden;
  const int64_t scale_expert_stride = static_cast<int64_t>(2) * intermediate_size * scale_hidden;
  const uint8_t* up_weight =
      gemm1_weights + static_cast<int64_t>(safe_expert) * weight_expert_stride;
  const uint8_t* gate_weight = up_weight + static_cast<int64_t>(intermediate_size) * packed_hidden;
  const __nv_bfloat16* up_scale =
      gemm1_scales + static_cast<int64_t>(safe_expert) * scale_expert_stride;
  const __nv_bfloat16* gate_scale =
      up_scale + static_cast<int64_t>(intermediate_size) * scale_hidden;

  float gate[kOutputs] = {};
  float up[kOutputs] = {};
  if (expert >= 0) {
    for (int index = lane * kVectorElements; index < hidden_size; index += kWarpVectorStride) {
      const float activation_scale =
          DecodeE4M3(hidden_scale_row[index / kScaleVectorElements]) * hidden_global_decode_scale;
#pragma unroll
      for (int output_index = 0; output_index < kOutputs; ++output_index) {
        const int output_column = column + output_index;
        const int64_t packed_offset =
            static_cast<int64_t>(output_column) * packed_hidden + index / 2;
        const int64_t scale_offset =
            static_cast<int64_t>(output_column) * scale_hidden + index / kScaleVectorElements;
        gate[output_index] =
            fmaf(Dot8Packed(hidden_row + index / 2, gate_weight + packed_offset),
                 activation_scale * __bfloat162float(gate_scale[scale_offset]), gate[output_index]);
        up[output_index] =
            fmaf(Dot8Packed(hidden_row + index / 2, up_weight + packed_offset),
                 activation_scale * __bfloat162float(up_scale[scale_offset]), up[output_index]);
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
      const float gate_value = gate[output_index];
      const float silu = gate_value / (1.0f + __expf(-gate_value));
      shared_outputs[warp * kOutputs + output_index] = silu * up[output_index];
    }
  }
  __syncthreads();

  const int lane16 = threadIdx.x & 15;
  const int subgroup = (threadIdx.x & 31) >> 4;
  const int subgroup_in_block = threadIdx.x >> 4;
  const int subgroups_per_block = blockDim.x >> 4;
  const unsigned subgroup_mask = subgroup == 0 ? 0x0000ffffu : 0xffff0000u;
  const int scale_groups_per_block = outputs_per_block / kScaleVectorElements;
  for (int group = subgroup_in_block; group < scale_groups_per_block;
       group += subgroups_per_block) {
    const float value = shared_outputs[group * kScaleVectorElements + lane16];
    float amax = fabsf(value);
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(subgroup_mask, amax, offset));
    }
    float dequant_scale = 0.0f;
    if (lane16 == 0) {
      const float encoded_scale =
          RoundE4M3(intermediate_global_encode_scale * amax * (1.0f / 6.0f));
      dequant_scale = __bfloat162float(
          __float2bfloat16_rn(__fdividef(encoded_scale, intermediate_global_encode_scale)));
      intermediate_scales[static_cast<int64_t>(row) * (intermediate_size / 16) + block_column / 16 +
                          group] = __float2bfloat16_rn(dequant_scale);
    }
    dequant_scale = __shfl_sync(subgroup_mask, dequant_scale, 0, 16);
    const float inverse_scale = dequant_scale == 0.0f ? 0.0f : __fdividef(1.0f, dequant_scale);
    const float scaled = value * inverse_scale;
    const float paired = __shfl_down_sync(subgroup_mask, scaled, 1);
    if ((lane16 & 1) == 0) {
      intermediate_quantized[static_cast<int64_t>(row) * (intermediate_size / 2) +
                             block_column / 2 + group * 8 + lane16 / 2] =
          __nv_cvt_float2_to_fp4x2(make_float2(scaled, paired), __NV_E2M1, cudaRoundNearest);
    }
  }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <int kOutputs>
__global__ void DownFusedTopKKernel(
    const uint8_t* __restrict__ intermediate, const __nv_bfloat16* __restrict__ intermediate_scales,
    const uint8_t* __restrict__ gemm2_weights, const __nv_bfloat16* __restrict__ gemm2_scales,
    const int32_t* __restrict__ topk_ids, const int32_t* __restrict__ expert_map,
    const float* __restrict__ topk_weights, __nv_bfloat16* __restrict__ output, int num_tokens,
    int topk, int num_local_experts, int expert_map_items, int hidden_size, int intermediate_size) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const int column_groups = (hidden_size + kOutputs - 1) / kOutputs;
  const int blocks_per_token = (column_groups + warps - 1) / warps;
  const int token = blockIdx.x / blocks_per_token;
  const int column_group = (blockIdx.x - token * blocks_per_token) * warps + warp;
  if (token >= num_tokens || column_group >= column_groups) {
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
    const uint8_t* input_row = intermediate + static_cast<int64_t>(row) * packed_intermediate;
    const __nv_bfloat16* input_scales =
        intermediate_scales + static_cast<int64_t>(row) * scale_intermediate;
    const uint8_t* weight = gemm2_weights + static_cast<int64_t>(expert) * weight_expert_stride;
    const __nv_bfloat16* scales = gemm2_scales + static_cast<int64_t>(expert) * scale_expert_stride;
    float result[kOutputs] = {};
    for (int index = lane * kVectorElements; index < intermediate_size;
         index += kWarpVectorStride) {
      const float activation_scale = __bfloat162float(input_scales[index / kScaleVectorElements]);
#pragma unroll
      for (int output_index = 0; output_index < kOutputs; ++output_index) {
        const int output_column = column + output_index;
        if (output_column < hidden_size) {
          const int64_t packed_offset =
              static_cast<int64_t>(output_column) * packed_intermediate + index / 2;
          const int64_t scale_offset = static_cast<int64_t>(output_column) * scale_intermediate +
                                       index / kScaleVectorElements;
          result[output_index] =
              fmaf(Dot8Packed(input_row + index / 2, weight + packed_offset),
                   activation_scale * __bfloat162float(scales[scale_offset]), result[output_index]);
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
inline cudaError_t LaunchWithOutputs(const B12xDirectNVFP4FusedMoeParams& params,
                                     cudaStream_t stream, bool run_down) {
  const int warps = params.num_threads / 32;
  const int routed_rows = params.num_tokens * params.topk;
  const int gate_outputs_per_block = warps * kOutputs;
  const int down_groups = (params.hidden_size + kOutputs - 1) / kOutputs;
  if (gate_outputs_per_block < 16 || gate_outputs_per_block % 16 != 0 ||
      params.intermediate_size % gate_outputs_per_block != 0) {
    return cudaErrorInvalidConfiguration;
  }
  const int gate_blocks = routed_rows * (params.intermediate_size / gate_outputs_per_block);
  const int down_blocks = params.num_tokens * ((down_groups + warps - 1) / warps);

  cudaLaunchAttribute attributes[1]{};
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t gate_config{};
  gate_config.gridDim = dim3(gate_blocks);
  gate_config.blockDim = dim3(params.num_threads);
  gate_config.dynamicSmemBytes = gate_outputs_per_block * sizeof(float);
  gate_config.stream = stream;
  gate_config.attrs = attributes;
  gate_config.numAttrs = 1;
  cudaError_t status = cudaLaunchKernelEx(
      &gate_config, &GateUpSwiGLUKernel<kOutputs>, params.hidden_quantized, params.hidden_scales,
      params.gemm1_weights, params.gemm1_scales, params.topk_ids, params.expert_map,
      params.intermediate_quantized, params.intermediate_scales, routed_rows, params.topk,
      params.num_local_experts, params.expert_map_items, params.hidden_size,
      params.intermediate_size, params.hidden_global_decode_scale,
      params.intermediate_global_encode_scale);
  if (status != cudaSuccess) {
    return status;
  }
  if (!run_down) {
    return cudaSuccess;
  }
  cudaLaunchConfig_t down_config{};
  down_config.gridDim = dim3(down_blocks);
  down_config.blockDim = dim3(params.num_threads);
  down_config.dynamicSmemBytes = 0;
  down_config.stream = stream;
  down_config.attrs = attributes;
  down_config.numAttrs = 1;
  return cudaLaunchKernelEx(&down_config, &DownFusedTopKKernel<kOutputs>,
                            params.intermediate_quantized, params.intermediate_scales,
                            params.gemm2_weights, params.gemm2_scales, params.topk_ids,
                            params.expert_map, params.topk_weights, params.output,
                            params.num_tokens, params.topk, params.num_local_experts,
                            params.expert_map_items, params.hidden_size, params.intermediate_size);
}

}  // namespace b12x_direct_nvfp4_detail

inline cudaError_t LaunchB12xDirectNVFP4FusedMoe(const B12xDirectNVFP4FusedMoeParams& params,
                                                 cudaStream_t stream, bool run_down = true) {
  switch (params.outputs_per_warp) {
    case 1:
      return b12x_direct_nvfp4_detail::LaunchWithOutputs<1>(params, stream, run_down);
    case 2:
      return b12x_direct_nvfp4_detail::LaunchWithOutputs<2>(params, stream, run_down);
    case 4:
      return b12x_direct_nvfp4_detail::LaunchWithOutputs<4>(params, stream, run_down);
    case 8:
      return b12x_direct_nvfp4_detail::LaunchWithOutputs<8>(params, stream, run_down);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace flashinfer::fused_moe
