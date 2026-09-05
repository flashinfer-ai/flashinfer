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
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <flashinfer/attention/sparse_mla_sm120/common/nvfp4_quantization.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/nvfp4_cache_traits.cuh>

#include "sparse_mla_sm120_nvfp4_common.h"
#include "tvm_ffi_utils.h"

namespace flashinfer::sparse_mla_sm120::nvfp4 {

using Cache = NVFP4CacheTraits<ModelType::DSV4>;

constexpr int kDNope = Cache::D_NOPE;
constexpr int kDRope = Cache::D_ROPE;
constexpr int kDLatent = kDNope + kDRope;
constexpr int kNumScaleGroups = Cache::NUM_SCALES;
constexpr int kPackedNopeBytes = Cache::PACKED_NOPE_BYTES;
constexpr int kDataBytesPerToken = Cache::DATA_BYTES_PER_TOKEN;
constexpr int kScaleBytesPerToken = Cache::SCALE_BYTES_PER_TOKEN;
constexpr int kBytesPerToken = Cache::BYTES_PER_TOKEN;
constexpr int kThreadsPerToken = 32;

static_assert(kNumScaleGroups == 28);
static_assert(Cache::SCALE_GROUP_SIZE == SF_VEC_SIZE);
static_assert(kPackedNopeBytes == 224);
static_assert(kDataBytesPerToken == 352);
static_assert(kBytesPerToken == 384);

template <typename T>
__device__ __forceinline__ void quantize_token(const T* input, uint8_t* data_output,
                                               uint8_t* scale_output) {
  const int tid = threadIdx.x;
  if (tid < kNumScaleGroups) {
    quantize_group16_to_nvfp4(input + tid * SF_VEC_SIZE, data_output + tid * FP4_PACKED_PER_GROUP,
                              scale_output + tid);
    return;
  }

  // Four threads copy 16 BF16 RoPE elements (32 bytes) each. The cache
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

void check_input(const TensorView& input, int expected_tokens) {
  constexpr size_t VECTOR_ALIGNMENT = alignof(uint4);
  TVM_FFI_ICHECK(input.ndim() == 2 || input.ndim() == 3 || input.ndim() == 4)
      << "latent_kv must be 2D, 3D, or 4D";
  TVM_FFI_ICHECK_EQ(input.size(input.ndim() - 1), kDLatent)
      << "latent_kv last dimension must be " << kDLatent;
  TVM_FFI_ICHECK_EQ(input.dtype(), dl_bfloat16) << "DeepSeek-V4 latent_kv must have dtype bfloat16";
  TVM_FFI_ICHECK(input.IsContiguous()) << "latent_kv must be contiguous";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(input.data_ptr()) % VECTOR_ALIGNMENT, 0)
      << "latent_kv base pointer must be " << VECTOR_ALIGNMENT << "-byte aligned";

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

  const PagedLayout shape = parse_nvfp4_paged_layout(cache);
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

  const PagedLayout shape = parse_nvfp4_paged_layout(cache);
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
