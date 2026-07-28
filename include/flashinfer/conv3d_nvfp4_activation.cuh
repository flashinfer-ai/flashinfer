/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef FLASHINFER_CONV3D_NVFP4_ACTIVATION_CUH_
#define FLASHINFER_CONV3D_NVFP4_ACTIVATION_CUH_

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer {
namespace conv3d_nvfp4 {

constexpr int kScaleVectorSize = 16;
constexpr int kActivationThreads = 256;

struct ActivationTileConfig {
  int spatial;
  int channels;
};

constexpr ActivationTileConfig kActivationTileConfigs[] = {
    {32, 64}, {16, 128}, {8, 256}, {32, 128}, {64, 64},
};
constexpr int kNumActivationTileVariants =
    sizeof(kActivationTileConfigs) / sizeof(kActivationTileConfigs[0]);

inline int activation_channel_tile(int tile_variant) {
  if (tile_variant < 0 || tile_variant >= kNumActivationTileVariants) {
    return 0;
  }
  return kActivationTileConfigs[tile_variant].channels;
}

__device__ __forceinline__ float reciprocal_approximate_ftz(float value) {
  float result;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ uint32_t fp32_vec_to_e2m1(float (&values)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t packed;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(packed)
      : "f"(values[0]), "f"(values[1]), "f"(values[2]), "f"(values[3]), "f"(values[4]),
        "f"(values[5]), "f"(values[6]), "f"(values[7]));
  return packed;
#else
  return 0;
#endif
}

template <int kSpatialTile, int kChannelTile, bool kMaterializeHalo>
__global__ __launch_bounds__(kActivationThreads, 4) void quantize_ncdhw_kernel(
    const __nv_bfloat16* input, const float* global_scale, uint8_t* packed_output,
    uint8_t* scale_output, int channels, int depth, int height, int width, int physical_depth,
    int physical_height, int physical_width, int pad_depth, int pad_height, int pad_width) {
  static_assert(kSpatialTile % 8 == 0);
  static_assert(kChannelTile % kScaleVectorSize == 0);
  static_assert((kSpatialTile * kChannelTile) % (kActivationThreads * 8) == 0);
  // The paired-lane shuffle below requires every quantization wave to be full.
  static_assert(((kSpatialTile * kChannelTile) / 8) % kActivationThreads == 0);

  __shared__ __align__(16) __nv_bfloat16 source_tile[kSpatialTile][kChannelTile];

  const int batch_idx = static_cast<int>(blockIdx.z);
  const int channel_base = static_cast<int>(blockIdx.y) * kChannelTile;
  const int physical_voxels = physical_depth * physical_height * physical_width;
  const int source_voxels = depth * height * width;
  const int voxel_base = static_cast<int>(blockIdx.x) * kSpatialTile;
  constexpr int kSpatialVectors = kSpatialTile / 8;
  constexpr int kLoadJobs = kChannelTile * kSpatialVectors;

  for (int load_job = static_cast<int>(threadIdx.x); load_job < kLoadJobs;
       load_job += kActivationThreads) {
    const int load_channel = load_job / kSpatialVectors;
    const int load_voxel_base = (load_job % kSpatialVectors) * 8;
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      const int tile_voxel = load_voxel_base + index;
      const int physical_voxel = voxel_base + tile_voxel;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (physical_voxel < physical_voxels) {
        int source_voxel = physical_voxel;
        bool valid = true;
        if constexpr (kMaterializeHalo) {
          int coordinate = physical_voxel;
          const int physical_w = coordinate % physical_width;
          coordinate /= physical_width;
          const int physical_h = coordinate % physical_height;
          const int physical_d = coordinate / physical_height;
          const int source_w = physical_w - pad_width;
          const int source_h = physical_h - pad_height;
          const int source_d = physical_d - pad_depth;
          valid = source_d >= 0 && source_d < depth && source_h >= 0 && source_h < height &&
                  source_w >= 0 && source_w < width;
          if (valid) {
            source_voxel = (source_d * height + source_h) * width + source_w;
          }
        }
        if (valid) {
          const int64_t offset =
              (static_cast<int64_t>(batch_idx) * channels + channel_base + load_channel) *
                  source_voxels +
              source_voxel;
          value = input[offset];
        }
      }
      source_tile[tile_voxel][load_channel] = value;
    }
  }
  __syncthreads();

  const float scale_multiplier = global_scale[0];
  constexpr int kScaleGroupsPerRow = kChannelTile / kScaleVectorSize;
  constexpr int kQuantJobs = kSpatialTile * kChannelTile / 8;
  uint32_t lane_id;
  asm("mov.u32 %0, %%laneid;" : "=r"(lane_id));
  const uint32_t pair_mask = 3u << (lane_id & ~1u);

  for (int quant_job = static_cast<int>(threadIdx.x); quant_job < kQuantJobs;
       quant_job += kActivationThreads) {
    const int quant_pair = quant_job / 2;
    const int tile_voxel = quant_pair / kScaleGroupsPerRow;
    const int scale_group = quant_pair % kScaleGroupsPerRow;
    const int half_group = quant_job & 1;
    const int channel_in_tile = scale_group * kScaleVectorSize + half_group * 8;
    const int physical_voxel = voxel_base + tile_voxel;

    if (physical_voxel < physical_voxels) {
      float values[8];
      float local_max = 0.0f;
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        const float value = __bfloat162float(source_tile[tile_voxel][channel_in_tile + index]);
        values[index] = value;
        local_max = fmaxf(local_max, fabsf(value));
      }
      const float peer_max = __shfl_xor_sync(pair_mask, local_max, 1);
      const float vector_max = fmaxf(local_max, peer_max);

      float scale_value = scale_multiplier * (vector_max * reciprocal_approximate_ftz(6.0f));
      __nv_fp8_e4m3 narrowed_scale = __nv_fp8_e4m3(scale_value);
      const uint8_t scale_code = narrowed_scale.__x;
      scale_value = static_cast<float>(narrowed_scale);
      const float output_scale =
          vector_max != 0.0f ? reciprocal_approximate_ftz(
                                   scale_value * reciprocal_approximate_ftz(scale_multiplier))
                             : 0.0f;

      if (half_group == 0) {
        const int64_t scale_offset =
            (static_cast<int64_t>(batch_idx) * physical_voxels + physical_voxel) *
                (channels / kScaleVectorSize) +
            channel_base / kScaleVectorSize + scale_group;
        scale_output[scale_offset] = scale_code;
      }

#pragma unroll
      for (int index = 0; index < 8; ++index) {
        values[index] *= output_scale;
      }
      const uint32_t packed = fp32_vec_to_e2m1(values);
      const int64_t packed_byte_offset =
          (static_cast<int64_t>(batch_idx) * physical_voxels + physical_voxel) * (channels / 2) +
          (channel_base + channel_in_tile) / 2;
      *reinterpret_cast<uint32_t*>(packed_output + packed_byte_offset) = packed;
    }
  }
}

template <int kSpatialTile, int kChannelTile>
inline void launch_quantize_ncdhw(bool materialize_halo, dim3 grid, cudaStream_t stream,
                                  const __nv_bfloat16* input, const float* global_scale,
                                  uint8_t* packed_output, uint8_t* scale_output, int channels,
                                  int depth, int height, int width, int physical_depth,
                                  int physical_height, int physical_width, int pad_depth,
                                  int pad_height, int pad_width) {
  const dim3 block(kActivationThreads);
  if (materialize_halo) {
    quantize_ncdhw_kernel<kSpatialTile, kChannelTile, true><<<grid, block, 0, stream>>>(
        input, global_scale, packed_output, scale_output, channels, depth, height, width,
        physical_depth, physical_height, physical_width, pad_depth, pad_height, pad_width);
  } else {
    quantize_ncdhw_kernel<kSpatialTile, kChannelTile, false><<<grid, block, 0, stream>>>(
        input, global_scale, packed_output, scale_output, channels, depth, height, width,
        physical_depth, physical_height, physical_width, 0, 0, 0);
  }
}

inline cudaError_t launch_activation_quantization(const __nv_bfloat16* input,
                                                  const float* global_scale, uint8_t* packed_output,
                                                  uint8_t* scale_output, int batch, int channels,
                                                  int depth, int height, int width, int pad_depth,
                                                  int pad_height, int pad_width, int tile_variant,
                                                  cudaStream_t stream) {
  const int physical_depth = depth + 2 * pad_depth;
  const int physical_height = height + 2 * pad_height;
  const int physical_width = width + 2 * pad_width;
  const int physical_voxels = physical_depth * physical_height * physical_width;
  const bool materialize_halo = pad_depth != 0 || pad_height != 0 || pad_width != 0;

  if (tile_variant < 0 || tile_variant >= kNumActivationTileVariants) {
    return cudaErrorInvalidValue;
  }
  const ActivationTileConfig tile = kActivationTileConfigs[tile_variant];
  if (channels <= 0 || channels % tile.channels != 0 || channels % kScaleVectorSize != 0) {
    return cudaErrorInvalidValue;
  }

  const dim3 grid((physical_voxels + tile.spatial - 1) / tile.spatial, channels / tile.channels,
                  batch);
  switch (tile_variant) {
    case 0:
      launch_quantize_ncdhw<kActivationTileConfigs[0].spatial, kActivationTileConfigs[0].channels>(
          materialize_halo, grid, stream, input, global_scale, packed_output, scale_output,
          channels, depth, height, width, physical_depth, physical_height, physical_width,
          pad_depth, pad_height, pad_width);
      break;
    case 1:
      launch_quantize_ncdhw<kActivationTileConfigs[1].spatial, kActivationTileConfigs[1].channels>(
          materialize_halo, grid, stream, input, global_scale, packed_output, scale_output,
          channels, depth, height, width, physical_depth, physical_height, physical_width,
          pad_depth, pad_height, pad_width);
      break;
    case 2:
      launch_quantize_ncdhw<kActivationTileConfigs[2].spatial, kActivationTileConfigs[2].channels>(
          materialize_halo, grid, stream, input, global_scale, packed_output, scale_output,
          channels, depth, height, width, physical_depth, physical_height, physical_width,
          pad_depth, pad_height, pad_width);
      break;
    case 3:
      launch_quantize_ncdhw<kActivationTileConfigs[3].spatial, kActivationTileConfigs[3].channels>(
          materialize_halo, grid, stream, input, global_scale, packed_output, scale_output,
          channels, depth, height, width, physical_depth, physical_height, physical_width,
          pad_depth, pad_height, pad_width);
      break;
    case 4:
      launch_quantize_ncdhw<kActivationTileConfigs[4].spatial, kActivationTileConfigs[4].channels>(
          materialize_halo, grid, stream, input, global_scale, packed_output, scale_output,
          channels, depth, height, width, physical_depth, physical_height, physical_width,
          pad_depth, pad_height, pad_width);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace conv3d_nvfp4
}  // namespace flashinfer

#endif  // FLASHINFER_CONV3D_NVFP4_ACTIVATION_CUH_
