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

// Quantize DeepSeek-V4.1 latent KV into the DSV4_1 FP8 paged layout (528
// B/token: 512 B E4M3 covering all 512 dims including RoPE + a 16 B footer of
// 16 UE8M0 scales, one per 32-wide group) read by the SM120 sparse-MLA kernels'
// main (SWA) cache path. The conversion mirrors the FlashMLA FP8 FOOTER torch
// trajectory bit for bit: scale = 2^ceil(log2(max(amax / 448, 1e-4))) with
// amax clamped at 1e-4 (so the smallest scale is 2^-13), values divided by the
// power-of-two scale (exact) and converted with round-to-nearest-even
// saturating E4M3. Non-finite inputs follow the torch trajectory as well: a
// NaN anywhere in the group gives a 0xFF scale byte and 0x7F (NaN) codes; an
// infinite group without NaN gives a 0xFF scale byte, 0x7F for the infinite
// elements and signed zero codes (0x00 / 0x80) for the finite ones.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <flashinfer/attention/sparse_mla_sm120/compute/nvfp4_quantization.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/dsv41_layout.cuh>

#include "../tvm_ffi_utils.h"

namespace flashinfer::sparse_mla_sm120 {

namespace {

using nvfp4::float_to_e4m3_byte;
using nvfp4::nvfp4_input_to_float;

constexpr int kThreadsPerToken = 32;                // two threads per 32-wide scale group
constexpr int kGroup = Dsv41Fp8Layout::QUANT_TILE;  // 32
constexpr int kHalfGroup = kGroup / 2;              // 16 values per thread
static_assert(kThreadsPerToken == 2 * Dsv41Fp8Layout::NUM_SCALES);
static_assert(kGroup == 32 && Dsv41Fp8Layout::NUM_SCALES == 16);
static_assert(Dsv41Fp8Layout::DATA_BYTES == 512 && Dsv41Fp8Layout::SCALE_BYTES == 16);

constexpr float kAmaxFloor = 1e-4f;
constexpr float kE4m3Max = 448.f;

struct PagedCacheInfo {
  int num_pages;
  int page_size;
  size_t page_stride_bytes;
};

PagedCacheInfo parse_dsv41_fp8_paged_layout(const TensorView& cache) {
  constexpr size_t BPT = Dsv41Fp8Layout::BYTES_PER_TOKEN;
  constexpr size_t VECTOR_ALIGNMENT = alignof(uint4);
  TVM_FFI_ICHECK_EQ(cache.dtype(), dl_uint8) << "DSV4_1 FP8 kv_cache must be uint8";
  TVM_FFI_ICHECK(cache.ndim() == 2 || cache.ndim() == 3 || cache.ndim() == 4)
      << "kv_cache must be 2D, 3D, HND, or NHD";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(cache.data_ptr()) % VECTOR_ALIGNMENT, 0)
      << "DSV4_1 FP8 kv_cache base pointer must be " << VECTOR_ALIGNMENT << "-byte aligned";
  TVM_FFI_ICHECK_EQ(static_cast<size_t>(cache.stride(0)) % VECTOR_ALIGNMENT, 0)
      << "DSV4_1 FP8 kv_cache page stride must be a multiple of " << VECTOR_ALIGNMENT << " bytes";
  if (cache.ndim() == 2) {
    const size_t page_bytes = static_cast<size_t>(cache.size(1));
    TVM_FFI_ICHECK_EQ(page_bytes % BPT, 0);
    TVM_FFI_ICHECK_EQ(cache.stride(1), 1) << "kv_cache byte dimension must be contiguous";
    TVM_FFI_ICHECK_GE(cache.stride(0), static_cast<int64_t>(page_bytes))
        << "kv_cache page stride is smaller than its logical payload";
    return {static_cast<int>(cache.size(0)), static_cast<int>(page_bytes / BPT),
            static_cast<size_t>(cache.stride(0))};
  }
  TVM_FFI_ICHECK_EQ(static_cast<size_t>(cache.size(cache.ndim() - 1)), BPT)
      << "DSV4_1 FP8 cache last dimension must be " << BPT;
  int page_dim;
  if (cache.ndim() == 3) {
    page_dim = 1;
  } else if (cache.size(1) == 1) {
    page_dim = 2;  // HND
  } else {
    TVM_FFI_ICHECK_EQ(cache.size(2), 1) << "4D kv_cache must be HND or NHD";
    page_dim = 1;  // NHD
  }
  TVM_FFI_ICHECK_EQ(cache.stride(cache.ndim() - 1), 1)
      << "kv_cache byte dimension must be contiguous";
  TVM_FFI_ICHECK_EQ(cache.stride(page_dim), static_cast<int64_t>(BPT))
      << "DSV4_1 FP8 footer rows must stay packed";
  TVM_FFI_ICHECK_GE(cache.stride(0), cache.size(page_dim) * static_cast<int64_t>(BPT))
      << "DSV4_1 FP8 page stride is smaller than its logical payload";
  return {static_cast<int>(cache.size(0)), static_cast<int>(cache.size(page_dim)),
          static_cast<size_t>(cache.stride(0))};
}

void check_latent(const TensorView& input, int64_t expected_tokens) {
  TVM_FFI_ICHECK(input.ndim() == 2 || input.ndim() == 3 || input.ndim() == 4)
      << "latent_kv must be 2D, 3D, or 4D";
  TVM_FFI_ICHECK_EQ(input.size(input.ndim() - 1), Dsv41Fp8Layout::D_NOPE)
      << "latent_kv last dimension must be " << Dsv41Fp8Layout::D_NOPE;
  TVM_FFI_ICHECK(input.dtype() == dl_bfloat16 || input.dtype() == dl_float16)
      << "latent_kv must have dtype bfloat16 or float16";
  TVM_FFI_ICHECK(input.IsContiguous()) << "latent_kv must be contiguous";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(input.data_ptr()) % alignof(uint4), 0)
      << "latent_kv base pointer must be 16-byte aligned";
  int64_t num_tokens = 1;
  for (int i = 0; i + 1 < input.ndim(); ++i) num_tokens *= input.size(i);
  TVM_FFI_ICHECK_EQ(num_tokens, expected_tokens)
      << "latent_kv contains " << num_tokens << " rows, expected " << expected_tokens;
}

// Power-of-two ceiling exponent of a positive normal fp32: k with 2^(k-1) < x <= 2^k.
__device__ __forceinline__ int pow2_ceil_exponent(float x) {
  const uint32_t bits = __float_as_uint(x);
  const int exponent = static_cast<int>(bits >> 23) - 127;
  return exponent + ((bits & 0x7FFFFFu) != 0 ? 1 : 0);
}

// Two threads (lane pair) quantize one 32-wide group: each converts 16 values
// and writes 16 E4M3 bytes; the even lane writes the group's UE8M0 scale byte.
template <typename T>
__device__ __forceinline__ void quantize_dsv41_half_group(const T* input, uint8_t* data_output,
                                                          uint8_t* scale_output, bool write_scale) {
  float values[kHalfGroup];
  bool has_nan = false;
  bool has_inf = false;
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < kHalfGroup; ++i) {
    values[i] = nvfp4_input_to_float(input[i]);
    has_nan |= isnan(values[i]);
    has_inf |= isinf(values[i]);
    amax = fmaxf(amax, fabsf(values[i]));
  }
  // Group-wide reduction over the lane pair.
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  const int flags = (has_nan ? 1 : 0) | (has_inf ? 2 : 0);
  const int group_flags = flags | __shfl_xor_sync(0xffffffffu, flags, 1);
  uint8_t codes[kHalfGroup];
  uint8_t scale_byte;
  if (group_flags & 1) {
    // torch: NaN amax -> NaN scale (exponent byte 0xFF) and NaN codes everywhere.
    scale_byte = 0xFF;
#pragma unroll
    for (int i = 0; i < kHalfGroup; ++i) codes[i] = 0x7F;
  } else if (group_flags & 2) {
    // torch: inf amax -> inf scale (0xFF); finite / inf = signed zero, inf / inf = NaN.
    scale_byte = 0xFF;
#pragma unroll
    for (int i = 0; i < kHalfGroup; ++i) {
      codes[i] = isinf(values[i]) ? 0x7F : (signbit(values[i]) ? 0x80 : 0x00);
    }
  } else {
    // scale = 2^ceil(log2(max(max(amax, 1e-4) / 448, 1e-4))); the division is the
    // only inexact step of the reference and must round to nearest (the module
    // builds with -use_fast_math).
    const float ratio = fmaxf(__fdiv_rn(fmaxf(amax, kAmaxFloor), kE4m3Max), kAmaxFloor);
    const int k = pow2_ceil_exponent(ratio);  // k in [-13, 120] for fp16 / bf16 inputs
    scale_byte = static_cast<uint8_t>(k + 127);
    const float inv_scale = __uint_as_float(static_cast<uint32_t>(127 - k) << 23);  // 2^-k
#pragma unroll
    for (int i = 0; i < kHalfGroup; ++i) {
      const float scaled = fminf(fmaxf(values[i] * inv_scale, -kE4m3Max), kE4m3Max);
      codes[i] = float_to_e4m3_byte(scaled);
    }
  }
  uint4 packed;
  packed.x = codes[0] | (codes[1] << 8) | (codes[2] << 16) | (codes[3] << 24);
  packed.y = codes[4] | (codes[5] << 8) | (codes[6] << 16) | (codes[7] << 24);
  packed.z = codes[8] | (codes[9] << 8) | (codes[10] << 16) | (codes[11] << 24);
  packed.w = codes[12] | (codes[13] << 8) | (codes[14] << 16) | (codes[15] << 24);
  *reinterpret_cast<uint4*>(data_output) = packed;
  if (write_scale) *scale_output = scale_byte;
}

template <typename T>
__device__ __forceinline__ void quantize_token(const T* input, uint8_t* data_output,
                                               uint8_t* scale_output) {
  const int tid = threadIdx.x;
  const int group = tid >> 1;
  const int half = tid & 1;
  const int offset = group * kGroup + half * kHalfGroup;
  quantize_dsv41_half_group(input + offset, data_output + offset, scale_output + group, half == 0);
}

__device__ __forceinline__ uint8_t* data_row(uint8_t* cache, size_t slot, int page_size,
                                             size_t page_stride_bytes) {
  const size_t page_idx = slot / page_size;
  const size_t entry_idx = slot % page_size;
  return cache + page_idx * page_stride_bytes + Dsv41Fp8Layout::data_offset(entry_idx);
}

__device__ __forceinline__ uint8_t* scale_row(uint8_t* cache, size_t slot, int page_size,
                                              size_t page_stride_bytes) {
  const size_t page_idx = slot / page_size;
  const size_t entry_idx = slot % page_size;
  return cache + page_idx * page_stride_bytes +
         Dsv41Fp8Layout::scale_offset(static_cast<size_t>(page_size), entry_idx);
}

template <typename T>
__global__ void QuantizePackKernel(const T* input, uint8_t* cache, int num_pages, int page_size,
                                   size_t page_stride_bytes) {
  const int page_idx = blockIdx.x;
  const int entry_idx = blockIdx.y;
  if (page_idx >= num_pages || entry_idx >= page_size) return;
  const size_t token_idx = static_cast<size_t>(page_idx) * page_size + entry_idx;
  quantize_token(input + token_idx * Dsv41Fp8Layout::D_NOPE,
                 data_row(cache, token_idx, page_size, page_stride_bytes),
                 scale_row(cache, token_idx, page_size, page_stride_bytes));
}

template <typename T, typename IdType>
__global__ void QuantizeAppendKernel(const T* input, const IdType* slot_mapping, int num_tokens,
                                     uint8_t* cache, int num_pages, int page_size,
                                     size_t page_stride_bytes) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;
  const IdType slot = slot_mapping[token_idx];
  if (slot < 0 || static_cast<size_t>(slot) >= static_cast<size_t>(num_pages) * page_size) return;

  // Duplicate slot mappings resolve deterministically: the lowest-index input
  // row wins, so a repeated slot cannot tear a cache record across blocks
  // (same warp-parallel scan as the V41_FP4 writer; the 528 B record has no
  // pad bytes to hold an owner scratch).
  bool has_earlier_duplicate = false;
  for (int prior = threadIdx.x; prior < token_idx; prior += kThreadsPerToken) {
    has_earlier_duplicate |= slot_mapping[prior] == slot;
  }
  if (__any_sync(0xffffffffu, has_earlier_duplicate)) return;

  quantize_token(input + static_cast<size_t>(token_idx) * Dsv41Fp8Layout::D_NOPE,
                 data_row(cache, static_cast<size_t>(slot), page_size, page_stride_bytes),
                 scale_row(cache, static_cast<size_t>(slot), page_size, page_stride_bytes));
}

}  // namespace

void SparseMlaSm120Dsv41Fp8QuantizePack(TensorView latent_kv, TensorView cache) {
  CHECK_CUDA(latent_kv);
  CHECK_CUDA(cache);
  TVM_FFI_ICHECK_EQ(latent_kv.device().device_id, cache.device().device_id)
      << "latent_kv and cache must be on the same CUDA device";

  const PagedCacheInfo shape = parse_dsv41_fp8_paged_layout(cache);
  check_latent(latent_kv, static_cast<int64_t>(shape.num_pages) * shape.page_size);
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
      << "DSV4_1 FP8 sparse-MLA full-page pack launch failed: " << cudaGetErrorString(status);
}

void SparseMlaSm120Dsv41Fp8QuantizeAppend(TensorView latent_kv, TensorView slot_mapping,
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

  const PagedCacheInfo shape = parse_dsv41_fp8_paged_layout(cache);
  const int num_tokens = static_cast<int>(slot_mapping.size(0));
  check_latent(latent_kv, num_tokens);
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
      << "DSV4_1 FP8 sparse-MLA append launch failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer::sparse_mla_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_dsv41_fp8_quantize_pack,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120Dsv41Fp8QuantizePack);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_dsv41_fp8_quantize_append,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120Dsv41Fp8QuantizeAppend);
