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

// Quantize DeepSeek-V4 latent KV into the paged cache layout consumed by the
// SM120 NVFP4 sparse-MLA kernels. The E2M1 conversion intentionally uses the
// same FlashInfer PTX helper as the dense SM120 NVFP4 attention quantizer.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <flashinfer/math.cuh>

#include "tvm_ffi_utils.h"

namespace flashinfer::sparse_mla_sm120::nvfp4 {

constexpr int kDNope = 448;
constexpr int kDRope = 64;
constexpr int kDLatent = kDNope + kDRope;
constexpr int kSFVecSize = 16;
constexpr int kNumScaleGroups = kDNope / kSFVecSize;
constexpr int kPackedNopeBytes = kDNope / 2;
constexpr int kRopeBytes = kDRope * sizeof(__nv_bfloat16);
constexpr int kDataBytesPerToken = kPackedNopeBytes + kRopeBytes;
constexpr int kScaleBytesPerToken = 32;
constexpr int kBytesPerToken = kDataBytesPerToken + kScaleBytesPerToken;
constexpr int kThreadsPerToken = 32;

static_assert(kNumScaleGroups == 28);
static_assert(kPackedNopeBytes == 224);
static_assert(kDataBytesPerToken == 352);
static_assert(kBytesPerToken == 384);

template <typename T>
__device__ __forceinline__ float to_float(T value);

template <>
__device__ __forceinline__ float to_float<half>(half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ void quantize_nope_group(const T* input, uint8_t* packed_output,
                                                    uint8_t* scale_output) {
  float values[kSFVecSize];
  float amax = 0.0f;

#pragma unroll
  for (int i = 0; i < kSFVecSize; ++i) {
    const float value = to_float(input[i]);
    values[i] = value;
    amax = fmaxf(amax, fabsf(value));
  }

  __nv_fp8_e4m3 scale_fp8 = __nv_fp8_e4m3(amax / 6.0f);
  *scale_output = scale_fp8.__x;
  const float scale = static_cast<float>(scale_fp8);
  const float scale_inv = scale == 0.0f ? 0.0f : 1.0f / scale;

  float normalized[kSFVecSize];
#pragma unroll
  for (int i = 0; i < kSFVecSize; ++i) {
    normalized[i] = values[i] * scale_inv;
  }

  const uint2 packed = make_uint2(
      math::fp32_vec_to_e2m1(normalized[0], normalized[1], normalized[2], normalized[3],
                             normalized[4], normalized[5], normalized[6], normalized[7]),
      math::fp32_vec_to_e2m1(normalized[8], normalized[9], normalized[10], normalized[11],
                             normalized[12], normalized[13], normalized[14], normalized[15]));
  *reinterpret_cast<uint2*>(packed_output) = packed;
}

template <typename T>
__device__ __forceinline__ void quantize_token(const T* input, uint8_t* data_output,
                                               uint8_t* scale_output) {
  const int tid = threadIdx.x;
  if (tid < kNumScaleGroups) {
    quantize_nope_group(input + tid * kSFVecSize, data_output + tid * (kSFVecSize / 2),
                        scale_output + tid);
    return;
  }

  // Four threads copy 16 BF16/FP16 RoPE elements (32 bytes) each. The cache
  // stores the source bits unchanged; the attention ABI interprets them with
  // the same 16-bit dtype as the source model path.
  const int rope_lane = tid - kNumScaleGroups;
  const uint4* rope_input = reinterpret_cast<const uint4*>(input + kDNope + rope_lane * 16);
  uint4* rope_output = reinterpret_cast<uint4*>(data_output + kPackedNopeBytes + rope_lane * 32);
  rope_output[0] = rope_input[0];
  rope_output[1] = rope_input[1];

  if (tid == kNumScaleGroups) {
    *reinterpret_cast<uint32_t*>(scale_output + kNumScaleGroups) = 0;
  }
}

template <typename T>
__global__ void QuantizePackKernel(const T* input, uint8_t* cache, int num_pages, int page_size,
                                   size_t page_stride_bytes) {
  const int page_idx = blockIdx.x;
  const int entry_idx = blockIdx.y;
  if (page_idx >= num_pages || entry_idx >= page_size) return;

  const size_t token_idx = static_cast<size_t>(page_idx) * page_size + entry_idx;
  const T* token_input = input + token_idx * kDLatent;
  uint8_t* page = cache + static_cast<size_t>(page_idx) * page_stride_bytes;
  uint8_t* data_output = page + static_cast<size_t>(entry_idx) * kDataBytesPerToken;
  uint8_t* scale_output = page + static_cast<size_t>(page_size) * kDataBytesPerToken +
                          static_cast<size_t>(entry_idx) * kScaleBytesPerToken;
  quantize_token(token_input, data_output, scale_output);
}

template <typename T, typename IdType>
__global__ void QuantizeAppendKernel(const T* input, const IdType* slot_mapping, int num_tokens,
                                     uint8_t* cache, int num_pages, int page_size,
                                     size_t page_stride_bytes) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;

  const IdType slot = slot_mapping[token_idx];
  if (slot < 0 || static_cast<size_t>(slot) >= static_cast<size_t>(num_pages) * page_size) return;

  const size_t page_idx = static_cast<size_t>(slot) / page_size;
  const size_t entry_idx = static_cast<size_t>(slot) % page_size;
  const T* token_input = input + static_cast<size_t>(token_idx) * kDLatent;
  uint8_t* page = cache + page_idx * page_stride_bytes;
  uint8_t* data_output = page + entry_idx * kDataBytesPerToken;
  uint8_t* scale_output =
      page + static_cast<size_t>(page_size) * kDataBytesPerToken + entry_idx * kScaleBytesPerToken;
  quantize_token(token_input, data_output, scale_output);
}

namespace {

struct CacheShape {
  int num_pages;
  int page_size;
  size_t page_stride_bytes;
};

CacheShape parse_cache_shape(const TensorView& cache) {
  TVM_FFI_ICHECK(cache.ndim() == 3 || cache.ndim() == 4)
      << "cache must be [num_pages, page_size, 384], HND "
         "[num_pages, 1, page_size, 384], or NHD "
         "[num_pages, page_size, 1, 384]";
  TVM_FFI_ICHECK_EQ(cache.dtype(), dl_uint8) << "cache must have dtype uint8";
  TVM_FFI_ICHECK_EQ(cache.size(cache.ndim() - 1), kBytesPerToken)
      << "cache last dimension must be " << kBytesPerToken;
  int page_dim;
  if (cache.ndim() == 3) {
    page_dim = 1;
  } else if (cache.size(1) == 1) {
    page_dim = 2;
  } else {
    TVM_FFI_ICHECK_EQ(cache.size(2), 1)
        << "cache must have a singleton latent-head dimension at axis 1 or 2";
    page_dim = 1;
  }
  const int page_size = static_cast<int>(cache.size(page_dim));
  TVM_FFI_ICHECK_EQ(cache.stride(cache.ndim() - 1), 1) << "cache byte dimension must be contiguous";
  TVM_FFI_ICHECK_EQ(cache.stride(page_dim), kBytesPerToken)
      << "cache entries inside a page must have stride " << kBytesPerToken;
  TVM_FFI_ICHECK_GE(cache.stride(0), static_cast<int64_t>(page_size) * kBytesPerToken)
      << "cache page stride is smaller than its logical page payload";
  return {static_cast<int>(cache.size(0)), page_size, static_cast<size_t>(cache.stride(0))};
}

void check_input(const TensorView& input, int expected_tokens) {
  TVM_FFI_ICHECK(input.ndim() == 2 || input.ndim() == 3 || input.ndim() == 4)
      << "latent_kv must be 2D, 3D, or 4D";
  TVM_FFI_ICHECK_EQ(input.size(input.ndim() - 1), kDLatent)
      << "latent_kv last dimension must be " << kDLatent;
  TVM_FFI_ICHECK_EQ(input.dtype(), dl_bfloat16) << "DeepSeek-V4 latent_kv must have dtype bfloat16";
  TVM_FFI_ICHECK(input.IsContiguous()) << "latent_kv must be contiguous";

  int64_t num_tokens = 1;
  for (int i = 0; i + 1 < input.ndim(); ++i) num_tokens *= input.size(i);
  TVM_FFI_ICHECK_EQ(num_tokens, expected_tokens)
      << "latent_kv contains " << num_tokens << " rows, expected " << expected_tokens;
}

}  // namespace

void SparseMlaSm120NVFP4QuantizePack(TensorView latent_kv, TensorView cache) {
  CHECK_CUDA(latent_kv);
  CHECK_CUDA(cache);
  TVM_FFI_ICHECK_EQ(latent_kv.device().device_id, cache.device().device_id)
      << "latent_kv and cache must be on the same CUDA device";

  const CacheShape shape = parse_cache_shape(cache);
  check_input(latent_kv, shape.num_pages * shape.page_size);
  if (shape.num_pages == 0 || shape.page_size == 0) return;
  ffi::CUDADeviceGuard device_guard(latent_kv.device().device_id);
  cudaStream_t stream = get_stream(latent_kv.device());
  const dim3 grid(shape.num_pages, shape.page_size);
  const dim3 block(kThreadsPerToken);

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(latent_kv.dtype(), c_type, [&] {
    QuantizePackKernel<c_type><<<grid, block, 0, stream>>>(
        static_cast<const c_type*>(latent_kv.data_ptr()), static_cast<uint8_t*>(cache.data_ptr()),
        shape.num_pages, shape.page_size, shape.page_stride_bytes);
    return true;
  });
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "NVFP4 sparse-MLA full-page pack launch failed: " << cudaGetErrorString(status);
}

void SparseMlaSm120NVFP4QuantizeAppend(TensorView latent_kv, TensorView slot_mapping,
                                       TensorView cache) {
  CHECK_CUDA(latent_kv);
  CHECK_CUDA(slot_mapping);
  CHECK_CUDA(cache);
  TVM_FFI_ICHECK_EQ(latent_kv.device().device_id, cache.device().device_id)
      << "latent_kv and cache must be on the same CUDA device";
  TVM_FFI_ICHECK_EQ(slot_mapping.device().device_id, cache.device().device_id)
      << "slot_mapping and cache must be on the same CUDA device";
  TVM_FFI_ICHECK_EQ(slot_mapping.ndim(), 1) << "slot_mapping must be 1D";
  TVM_FFI_ICHECK(slot_mapping.dtype() == dl_int32 || slot_mapping.dtype() == dl_int64)
      << "slot_mapping must have dtype int32 or int64";
  TVM_FFI_ICHECK(slot_mapping.IsContiguous()) << "slot_mapping must be contiguous";

  const CacheShape shape = parse_cache_shape(cache);
  const int num_tokens = static_cast<int>(slot_mapping.size(0));
  check_input(latent_kv, num_tokens);
  if (num_tokens == 0) return;

  ffi::CUDADeviceGuard device_guard(latent_kv.device().device_id);
  cudaStream_t stream = get_stream(latent_kv.device());
  const dim3 grid(num_tokens);
  const dim3 block(kThreadsPerToken);

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(latent_kv.dtype(), c_type, [&] {
    if (slot_mapping.dtype() == dl_int32) {
      QuantizeAppendKernel<c_type, int32_t>
          <<<grid, block, 0, stream>>>(static_cast<const c_type*>(latent_kv.data_ptr()),
                                       static_cast<const int32_t*>(slot_mapping.data_ptr()),
                                       num_tokens, static_cast<uint8_t*>(cache.data_ptr()),
                                       shape.num_pages, shape.page_size, shape.page_stride_bytes);
    } else {
      QuantizeAppendKernel<c_type, int64_t>
          <<<grid, block, 0, stream>>>(static_cast<const c_type*>(latent_kv.data_ptr()),
                                       static_cast<const int64_t*>(slot_mapping.data_ptr()),
                                       num_tokens, static_cast<uint8_t*>(cache.data_ptr()),
                                       shape.num_pages, shape.page_size, shape.page_stride_bytes);
    }
    return true;
  });
  const cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "NVFP4 sparse-MLA append launch failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer::sparse_mla_sm120::nvfp4

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_nvfp4_quantize_pack,
                              flashinfer::sparse_mla_sm120::nvfp4::SparseMlaSm120NVFP4QuantizePack);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sparse_mla_sm120_nvfp4_quantize_append,
    flashinfer::sparse_mla_sm120::nvfp4::SparseMlaSm120NVFP4QuantizeAppend);
