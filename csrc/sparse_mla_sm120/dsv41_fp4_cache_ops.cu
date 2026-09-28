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

// Quantize DeepSeek-V4.1 latent KV into the V41_FP4 paged layout (288
// B/token: 256 B packed E2M1 + 32 B E4M3-G16 scales, all 512 dims including
// RoPE) read by the SM120 sparse-MLA kernels' FP4-extra gather path. The
// conversion mirrors the FlashMLA tests/quant.py V41_FP4 trajectory bit for
// bit: scale = clamp(amax/6, 2^-9, 448) rounded to E4M3, values divided by
// the scale (fp32 RN) and packed with cvt.rn.satfinite.e2m1x2.f32. A NaN/Inf
// element poisons its group (E4M3 NaN scale, zero codes), matching FlashMLA's
// "keep the NaN in the scale" convention.

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

using nvfp4::e4m3_byte_to_float;
using nvfp4::float_to_e4m3_byte;

constexpr int kThreadsPerToken = 32;  // one thread per 16-wide scale group
static_assert(kThreadsPerToken == Dsv41Fp4Layout::NUM_SCALES);

struct PagedCacheInfo {
  int num_pages;
  int page_size;
  size_t page_stride_bytes;
};

PagedCacheInfo parse_dsv41_fp4_paged_layout(const TensorView& cache) {
  constexpr size_t BPT = Dsv41Fp4Layout::BYTES_PER_TOKEN;
  constexpr size_t VECTOR_ALIGNMENT = alignof(uint4);
  TVM_FFI_ICHECK_EQ(cache.dtype(), dl_uint8) << "V41_FP4 kv_cache must be uint8";
  TVM_FFI_ICHECK(cache.ndim() == 2 || cache.ndim() == 3 || cache.ndim() == 4)
      << "kv_cache must be 2D, 3D, HND, or NHD";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(cache.data_ptr()) % VECTOR_ALIGNMENT, 0)
      << "V41_FP4 kv_cache base pointer must be " << VECTOR_ALIGNMENT << "-byte aligned";
  TVM_FFI_ICHECK_EQ(static_cast<size_t>(cache.stride(0)) % VECTOR_ALIGNMENT, 0)
      << "V41_FP4 kv_cache page stride must be a multiple of " << VECTOR_ALIGNMENT << " bytes";
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
      << "V41_FP4 cache last dimension must be " << BPT;
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
      << "V41_FP4 footer rows must stay packed";
  TVM_FFI_ICHECK_GE(cache.stride(0), cache.size(page_dim) * static_cast<int64_t>(BPT))
      << "V41_FP4 page stride is smaller than its logical payload";
  return {static_cast<int>(cache.size(0)), static_cast<int>(cache.size(page_dim)),
          static_cast<size_t>(cache.stride(0))};
}

void check_latent(const TensorView& input, int64_t expected_tokens) {
  TVM_FFI_ICHECK(input.ndim() == 2 || input.ndim() == 3 || input.ndim() == 4)
      << "latent_kv must be 2D, 3D, or 4D";
  TVM_FFI_ICHECK_EQ(input.size(input.ndim() - 1), Dsv41Fp4Layout::DIMS)
      << "latent_kv last dimension must be " << Dsv41Fp4Layout::DIMS;
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

// One thread writes a 16-wide E2M1 group and its clamped E4M3 scale.
template <typename T>
__device__ __forceinline__ void quantize_dsv41_group16(const T* input, uint8_t* data_output,
                                                       uint8_t* scale_output) {
  float values[nvfp4::SF_VEC_SIZE];
  bool poisoned = false;
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < nvfp4::SF_VEC_SIZE; ++i) {
    values[i] = nvfp4::nvfp4_input_to_float(input[i]);
    poisoned |= !isfinite(values[i]);
    amax = fmaxf(amax, fabsf(values[i]));
  }
  if (poisoned) {
    // FlashMLA poison semantics: zero codes plus a NaN scale, so the reader
    // propagates 0 * NaN = NaN for the whole group.
    *reinterpret_cast<uint2*>(data_output) = make_uint2(0, 0);
    *scale_output = 0x7F;  // E4M3 NaN
    return;
  }
  // The stored (E4M3-rounded) scale is what the values are divided by.
  // __fdiv_rn: the module builds with -use_fast_math, and an approximate
  // division flips exact-tie roundings against the FlashMLA reference.
  const uint8_t scale_byte =
      float_to_e4m3_byte(fminf(fmaxf(__fdiv_rn(amax, 6.f), 0.001953125f /* 2^-9 */), 448.f));
  const float scale = e4m3_byte_to_float(scale_byte);
  float normalized[nvfp4::SF_VEC_SIZE];
#pragma unroll
  for (int i = 0; i < nvfp4::SF_VEC_SIZE; ++i) normalized[i] = __fdiv_rn(values[i], scale);
  *scale_output = scale_byte;
  *reinterpret_cast<uint2*>(data_output) = make_uint2(
      math::fp32_vec_to_e2m1(normalized[0], normalized[1], normalized[2], normalized[3],
                             normalized[4], normalized[5], normalized[6], normalized[7]),
      math::fp32_vec_to_e2m1(normalized[8], normalized[9], normalized[10], normalized[11],
                             normalized[12], normalized[13], normalized[14], normalized[15]));
}

template <typename T>
__device__ __forceinline__ void quantize_token(const T* input, uint8_t* data_output,
                                               uint8_t* scale_output) {
  const int tid = threadIdx.x;
  quantize_dsv41_group16(input + tid * nvfp4::SF_VEC_SIZE,
                         data_output + tid * nvfp4::FP4_PACKED_PER_GROUP, scale_output + tid);
}

__device__ __forceinline__ uint8_t* data_row(uint8_t* cache, size_t slot, int page_size,
                                             size_t page_stride_bytes) {
  const size_t page_idx = slot / page_size;
  const size_t entry_idx = slot % page_size;
  return cache + page_idx * page_stride_bytes + Dsv41Fp4Layout::data_offset(entry_idx);
}

__device__ __forceinline__ uint8_t* scale_row(uint8_t* cache, size_t slot, int page_size,
                                              size_t page_stride_bytes) {
  const size_t page_idx = slot / page_size;
  const size_t entry_idx = slot % page_size;
  return cache + page_idx * page_stride_bytes +
         Dsv41Fp4Layout::scale_offset(static_cast<size_t>(page_size), entry_idx);
}

template <typename T>
__global__ void QuantizePackKernel(const T* input, uint8_t* cache, int num_pages, int page_size,
                                   size_t page_stride_bytes) {
  const int page_idx = blockIdx.x;
  const int entry_idx = blockIdx.y;
  if (page_idx >= num_pages || entry_idx >= page_size) return;
  const size_t token_idx = static_cast<size_t>(page_idx) * page_size + entry_idx;
  quantize_token(input + token_idx * Dsv41Fp4Layout::DIMS,
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
  // row wins, so a repeated slot cannot tear a cache record across blocks.
  // The NVFP4 (384 B) writer keeps an owner scratch in its scale-row padding;
  // V41_FP4 has no pad bytes, so this warp-parallel scan replaces it (decode-
  // sized appends are short; prefill chunks stay warp-cheap at ~n/32 steps).
  bool has_earlier_duplicate = false;
  for (int prior = threadIdx.x; prior < token_idx; prior += kThreadsPerToken) {
    has_earlier_duplicate |= slot_mapping[prior] == slot;
  }
  if (__any_sync(0xffffffffu, has_earlier_duplicate)) return;

  quantize_token(input + static_cast<size_t>(token_idx) * Dsv41Fp4Layout::DIMS,
                 data_row(cache, static_cast<size_t>(slot), page_size, page_stride_bytes),
                 scale_row(cache, static_cast<size_t>(slot), page_size, page_stride_bytes));
}

}  // namespace

void SparseMlaSm120Dsv41Fp4QuantizePack(TensorView latent_kv, TensorView cache) {
  CHECK_CUDA(latent_kv);
  CHECK_CUDA(cache);
  TVM_FFI_ICHECK_EQ(latent_kv.device().device_id, cache.device().device_id)
      << "latent_kv and cache must be on the same CUDA device";

  const PagedCacheInfo shape = parse_dsv41_fp4_paged_layout(cache);
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
      << "V41_FP4 sparse-MLA full-page pack launch failed: " << cudaGetErrorString(status);
}

void SparseMlaSm120Dsv41Fp4QuantizeAppend(TensorView latent_kv, TensorView slot_mapping,
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

  const PagedCacheInfo shape = parse_dsv41_fp4_paged_layout(cache);
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
      << "V41_FP4 sparse-MLA append launch failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer::sparse_mla_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_dsv41_fp4_quantize_pack,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120Dsv41Fp4QuantizePack);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_dsv41_fp4_quantize_append,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120Dsv41Fp4QuantizeAppend);
