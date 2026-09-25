/*
 * Copyright (c) 2024 by FlashInfer team.
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
#ifndef FLASHINFER_QUANTIZATION_CUH_
#define FLASHINFER_QUANTIZATION_CUH_
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cub/cub.cuh>

#include "utils.cuh"

namespace flashinfer {
namespace quantization {

enum class BitOrder { kBig = 0U, kLittle = 1U };

namespace detail {

template <typename DTypeOut>
__device__ __forceinline__ DTypeOut Quantize8Bit(float value) {
  return DTypeOut(value);
}

template <>
__device__ __forceinline__ int8_t Quantize8Bit<int8_t>(float value) {
  return static_cast<int8_t>(__float2int_rn(value));
}

// Round a positive, normal float up to the next power of two. The quantizer
// clamps the input to at least 1e-10 before calling this helper, so neither
// zero nor subnormal handling is needed here.
__device__ __forceinline__ float RoundUpToPowerOfTwo(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t exponent = (bits >> 23) & 0xffu;
  const uint32_t mantissa = bits & 0x7fffffu;
  return __uint_as_float((exponent + static_cast<uint32_t>(mantissa != 0)) << 23);
}

}  // namespace detail

// A 128-value group occupies only 256 bytes for FP16/BF16 input. Keeping it in
// registers avoids both the shared-memory round trip and the block-wide barrier
// between absmax reduction and quantization. Each half warp owns one group:
// 16 lanes x 8 values, with a 16-byte vector load and an 8-byte vector store
// per lane on the aligned path.
template <typename DTypeIn, typename DTypeOut, bool COLUMN_MAJOR, bool SCALE_UE8M0,
          bool ALIGNED_INPUT>
__global__ void PerTokenGroupQuant8BitRegisterKernel(const DTypeIn* __restrict__ input,
                                                     DTypeOut* __restrict__ output_q,
                                                     float* __restrict__ output_s,
                                                     int64_t num_groups, float eps, float min_8bit,
                                                     float max_8bit, int64_t groups_per_row,
                                                     int64_t scale_stride) {
  constexpr int kGroupSize = 128;
  constexpr int kThreadsPerGroup = 16;
  constexpr int kValuesPerThread = kGroupSize / kThreadsPerGroup;
  constexpr int kGroupsPerBlock = 256 / kThreadsPerGroup;
  static_assert(sizeof(DTypeIn) == 2, "register path requires 16-bit input");
  static_assert(sizeof(DTypeOut) == 1, "quantized output must be one byte");

  const int subgroup = threadIdx.x / kThreadsPerGroup;
  const int lane = threadIdx.x % kThreadsPerGroup;
  const int64_t group_idx = static_cast<int64_t>(blockIdx.x) * kGroupsPerBlock + subgroup;
  if (group_idx >= num_groups) {
    return;
  }

  const DTypeIn* group_input = input + group_idx * kGroupSize + lane * kValuesPerThread;
  alignas(16) DTypeIn values[kValuesPerThread];
  if constexpr (ALIGNED_INPUT) {
    *reinterpret_cast<uint4*>(values) = *reinterpret_cast<const uint4*>(group_input);
  } else {
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
      values[i] = group_input[i];
    }
  }

  float absmax = eps;
#pragma unroll
  for (int i = 0; i < kValuesPerThread; ++i) {
    absmax = fmaxf(absmax, fabsf(static_cast<float>(values[i])));
  }

  const unsigned subgroup_mask = 0xffffu << (threadIdx.x & 16);
#pragma unroll
  for (int offset = kThreadsPerGroup / 2; offset > 0; offset >>= 1) {
    absmax = fmaxf(absmax, __shfl_xor_sync(subgroup_mask, absmax, offset, kThreadsPerGroup));
  }

  float scale = absmax / max_8bit;
  if constexpr (SCALE_UE8M0) {
    scale = detail::RoundUpToPowerOfTwo(fmaxf(fabsf(scale), 1e-10f));
  }

  if (lane == 0) {
    if constexpr (COLUMN_MAJOR) {
      const int64_t token_idx = group_idx / groups_per_row;
      const int64_t group_in_row = group_idx % groups_per_row;
      output_s[group_in_row * scale_stride + token_idx] = scale;
    } else {
      output_s[group_idx] = scale;
    }
  }

  const float inv_scale = 1.0f / scale;
  alignas(8) DTypeOut quantized[kValuesPerThread];
#pragma unroll
  for (int i = 0; i < kValuesPerThread; ++i) {
    const float value = static_cast<float>(values[i]);
    // Multiplication is bit-equivalent to division only for power-of-two
    // scales. Keep division for ordinary FP32 scales to preserve boundaries.
    const float scaled = SCALE_UE8M0 ? value * inv_scale : __fdiv_rn(value, scale);
    const float clamped = fminf(fmaxf(scaled, min_8bit), max_8bit);
    quantized[i] = detail::Quantize8Bit<DTypeOut>(clamped);
  }

  DTypeOut* group_output = output_q + group_idx * kGroupSize + lane * kValuesPerThread;
  *reinterpret_cast<uint2*>(group_output) = *reinterpret_cast<const uint2*>(quantized);
}

template <typename DTypeIn, typename DTypeOut, bool COLUMN_MAJOR, bool SCALE_UE8M0,
          bool ALIGNED_INPUT>
cudaError_t LaunchPerTokenGroupQuant8BitRegister(const DTypeIn* input, DTypeOut* output_q,
                                                 float* output_s, int64_t num_groups, float eps,
                                                 float min_8bit, float max_8bit,
                                                 int64_t groups_per_row, int64_t scale_stride,
                                                 cudaStream_t stream) {
  if (num_groups == 0) {
    return cudaSuccess;
  }
  constexpr int kThreads = 256;
  constexpr int kGroupsPerBlock = 16;
  const dim3 blocks(ceil_div(num_groups, kGroupsPerBlock));
  PerTokenGroupQuant8BitRegisterKernel<DTypeIn, DTypeOut, COLUMN_MAJOR, SCALE_UE8M0, ALIGNED_INPUT>
      <<<blocks, kThreads, 0, stream>>>(input, output_q, output_s, num_groups, eps, min_8bit,
                                        max_8bit, groups_per_row, scale_stride);
  return cudaGetLastError();
}

template <typename DTypeIn, typename DTypeOut>
cudaError_t PerTokenGroupQuant8BitRegister(const DTypeIn* input, DTypeOut* output_q,
                                           float* output_s, int64_t num_groups, float eps,
                                           float min_8bit, float max_8bit, int64_t groups_per_row,
                                           int64_t scale_stride, bool column_major,
                                           bool scale_ue8m0, cudaStream_t stream) {
  const bool aligned_input = reinterpret_cast<uintptr_t>(input) % alignof(uint4) == 0;

#define FLASHINFER_LAUNCH_QUANT_REGISTER(COLUMN_MAJOR, SCALE_UE8M0, ALIGNED_INPUT)                \
  return LaunchPerTokenGroupQuant8BitRegister<DTypeIn, DTypeOut, COLUMN_MAJOR, SCALE_UE8M0,       \
                                              ALIGNED_INPUT>(input, output_q, output_s,           \
                                                             num_groups, eps, min_8bit, max_8bit, \
                                                             groups_per_row, scale_stride, stream)

  if (column_major) {
    if (scale_ue8m0) {
      if (aligned_input) {
        FLASHINFER_LAUNCH_QUANT_REGISTER(true, true, true);
      }
      FLASHINFER_LAUNCH_QUANT_REGISTER(true, true, false);
    }
    if (aligned_input) {
      FLASHINFER_LAUNCH_QUANT_REGISTER(true, false, true);
    }
    FLASHINFER_LAUNCH_QUANT_REGISTER(true, false, false);
  }
  if (scale_ue8m0) {
    if (aligned_input) {
      FLASHINFER_LAUNCH_QUANT_REGISTER(false, true, true);
    }
    FLASHINFER_LAUNCH_QUANT_REGISTER(false, true, false);
  }
  if (aligned_input) {
    FLASHINFER_LAUNCH_QUANT_REGISTER(false, false, true);
  }
  FLASHINFER_LAUNCH_QUANT_REGISTER(false, false, false);

#undef FLASHINFER_LAUNCH_QUANT_REGISTER
}

#define DISPATCH_BITORDER(bitorder, BITORDER, ...)   \
  if (bitorder == BitOrder::kBig) {                  \
    constexpr BitOrder BITORDER = BitOrder::kBig;    \
    __VA_ARGS__                                      \
  } else {                                           \
    constexpr BitOrder BITORDER = BitOrder::kLittle; \
    __VA_ARGS__                                      \
  }

template <BitOrder BITORDER>
__global__ void PackBitsKernel(bool* input, uint8_t* output, int64_t num_elements) {
  int64_t start_offset = static_cast<int64_t>(blockIdx.x) * blockDim.x * 8, tx = threadIdx.x;
  uint8_t ret = 0;
  bool input_vec[8];
  typedef cub::BlockLoad<bool, 256, 8, cub::BLOCK_LOAD_VECTORIZE> BlockLoad;
  __shared__ typename BlockLoad::TempStorage temp_storage;

  // This fix the INT32_T overflow issue, which is possible in DiT video models
  // where the kv_len could be 128K.
  // ref:
  // https://github.com/NVIDIA/cub/blob/0fc3c3701632a4be906765b73be20a9ad0da603d/cub/block/block_load.cuh#L711C13-L711C100
  int block_items_end =
      (num_elements - start_offset > INT32_MAX) ? INT32_MAX : num_elements - start_offset;
  BlockLoad(temp_storage).Load(input + start_offset, input_vec, block_items_end, /*default=*/0);

  if constexpr (BITORDER == BitOrder::kBig) {
    ret = (input_vec[0] << 7) | (input_vec[1] << 6) | (input_vec[2] << 5) | (input_vec[3] << 4) |
          (input_vec[4] << 3) | (input_vec[5] << 2) | (input_vec[6] << 1) | input_vec[7];
  } else {
    ret = (input_vec[7] << 7) | (input_vec[6] << 6) | (input_vec[5] << 5) | (input_vec[4] << 4) |
          (input_vec[3] << 3) | (input_vec[2] << 2) | (input_vec[1] << 1) | input_vec[0];
  }
  if (start_offset + tx * 8 < num_elements) output[start_offset / 8 + tx] = ret;
}

template <BitOrder BITORDER, typename IdType>
__global__ void SegmentPackBitsKernel(bool* input, uint8_t* output, IdType* input_indptr,
                                      IdType* output_indptr) {
  int64_t bx = blockIdx.x, tx = threadIdx.x;
  bool input_vec[8];
  typedef cub::BlockLoad<bool, 256, 8, cub::BLOCK_LOAD_VECTORIZE> BlockLoad;
  __shared__ typename BlockLoad::TempStorage temp_storage;
  int64_t num_elements = input_indptr[bx + 1] - input_indptr[bx];
  for (uint32_t start_offset = 0; start_offset < num_elements; start_offset += 8 * blockDim.x) {
    uint8_t ret = 0;
    BlockLoad(temp_storage)
        .Load(input + input_indptr[bx] + start_offset, input_vec, num_elements - start_offset,
              /*default=*/0);

    if constexpr (BITORDER == BitOrder::kBig) {
      ret = (input_vec[0] << 7) | (input_vec[1] << 6) | (input_vec[2] << 5) | (input_vec[3] << 4) |
            (input_vec[4] << 3) | (input_vec[5] << 2) | (input_vec[6] << 1) | input_vec[7];
    } else {
      ret = (input_vec[7] << 7) | (input_vec[6] << 6) | (input_vec[5] << 5) | (input_vec[4] << 4) |
            (input_vec[3] << 3) | (input_vec[2] << 2) | (input_vec[1] << 1) | input_vec[0];
    }
    if (start_offset + tx * 8 < num_elements)
      output[output_indptr[bx] + start_offset / 8 + tx] = ret;
  }
}

cudaError_t PackBits(bool* input, uint8_t* output, int64_t num_elements, BitOrder bitorder,
                     cudaStream_t stream) {
  DISPATCH_BITORDER(bitorder, BITORDER, {
    auto kernel = PackBitsKernel<BITORDER>;
    const dim3 nthrs(256);
    const dim3 nblks(ceil_div(num_elements, nthrs.x * 8));
    void* args[] = {&input, &output, &num_elements};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

template <typename IdType>
cudaError_t SegmentPackBits(bool* input, uint8_t* output, IdType* input_indptr,
                            IdType* output_indptr, uint32_t batch_size, BitOrder bitorder,
                            cudaStream_t stream) {
  DISPATCH_BITORDER(bitorder, BITORDER, {
    auto kernel = SegmentPackBitsKernel<BITORDER, IdType>;
    const dim3 nthrs(256);
    const dim3 nblks(batch_size);
    void* args[] = {&input, &output, &input_indptr, &output_indptr};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

}  // namespace quantization
}  // namespace flashinfer

#endif  // FLASHINFER_QUANTIZATION_CUH_
