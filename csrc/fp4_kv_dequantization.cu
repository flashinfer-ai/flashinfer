/*
 * Copyright (c) 2025 by FlashInfer team.
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

// FP4 KV cache dequantization kernels with linear (non-swizzled) block scale layout.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "flashinfer/utils.cuh"
#include "tvm_ffi_utils.h"

// Number of elements per block scale group
constexpr int NVFP4_BLOCK_SIZE = 16;

// E2M1 lookup table
__device__ __constant__ float E2M1_LUT[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,
                                              4.0f,  6.0f,  -0.0f, -0.5f, -1.0f, -1.5f,
                                              -2.0f, -3.0f, -4.0f, -6.0f};

// Dequantize 4 FP4 values (packed in uint16_t) to float2x2
__device__ __forceinline__ void dequant_fp4x4_to_float2x2(uint16_t packed_fp4, float scale_0,
                                                          float scale_1, float global_scale,
                                                          float2& out0, float2& out1) {
  uint8_t fp4_0 = (packed_fp4 >> 0) & 0xF;
  uint8_t fp4_1 = (packed_fp4 >> 4) & 0xF;
  uint8_t fp4_2 = (packed_fp4 >> 8) & 0xF;
  uint8_t fp4_3 = (packed_fp4 >> 12) & 0xF;

  out0.x = E2M1_LUT[fp4_0] * scale_0 * global_scale;
  out0.y = E2M1_LUT[fp4_1] * scale_0 * global_scale;
  out1.x = E2M1_LUT[fp4_2] * scale_1 * global_scale;
  out1.y = E2M1_LUT[fp4_3] * scale_1 * global_scale;
}

template <typename OutType, int BLOCK_SIZE = 128, int ELTS_PER_THREAD = 16>
__global__ void nvfp4_dequant_vectorized_kernel(const uint8_t* __restrict__ fp4_data,
                                                const uint8_t* __restrict__ block_scales,
                                                const float* __restrict__ global_scale_ptr,
                                                OutType* __restrict__ output, const int M,
                                                const int K) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (row >= M) return;

  extern __shared__ uint8_t smem[];
  float& global_scale = *reinterpret_cast<float*>(smem);
  // smem_scales starts after global_scale (aligned to 4 bytes)
  uint8_t* smem_scales = smem + sizeof(float);

  if (tid == 0) {
    global_scale = *global_scale_ptr;
  }

  const int K_scales = K / NVFP4_BLOCK_SIZE;
  const int K_packed = K / 2;

  for (int i = tid; i < K_scales; i += BLOCK_SIZE) {
    smem_scales[i] = block_scales[row * K_scales + i];
  }
  __syncthreads();

  constexpr int PACKED_PER_THREAD = ELTS_PER_THREAD / 2;
  const int elts_per_block = BLOCK_SIZE * ELTS_PER_THREAD;

  const uint8_t* row_fp4 = fp4_data + row * K_packed;
  OutType* row_output = output + row * K;

  for (int base_col = 0; base_col < K; base_col += elts_per_block) {
    const int col_start = base_col + tid * ELTS_PER_THREAD;

    if (col_start >= K) break;

#pragma unroll
    for (int i = 0; i < PACKED_PER_THREAD / 2; ++i) {
      const int col = col_start + i * 4;
      if (col + 3 >= K) break;

      const int packed_idx = col / 2;
      uint16_t packed_fp4 = *reinterpret_cast<const uint16_t*>(&row_fp4[packed_idx]);

      const int scale_idx_0 = col / NVFP4_BLOCK_SIZE;
      const int scale_idx_1 = (col + 2) / NVFP4_BLOCK_SIZE;

      const uint8_t scale_fp8_0 = smem_scales[scale_idx_0];
      const uint8_t scale_fp8_1 = smem_scales[scale_idx_1];

      const float scale_0 =
          static_cast<float>(*reinterpret_cast<const __nv_fp8_e4m3*>(&scale_fp8_0));
      const float scale_1 =
          static_cast<float>(*reinterpret_cast<const __nv_fp8_e4m3*>(&scale_fp8_1));

      float2 out0, out1;
      dequant_fp4x4_to_float2x2(packed_fp4, scale_0, scale_1, global_scale, out0, out1);

      if constexpr (std::is_same_v<OutType, __nv_bfloat16>) {
        __nv_bfloat162 bf16_0 = __float22bfloat162_rn(out0);
        __nv_bfloat162 bf16_1 = __float22bfloat162_rn(out1);

        *reinterpret_cast<__nv_bfloat162*>(&row_output[col]) = bf16_0;
        *reinterpret_cast<__nv_bfloat162*>(&row_output[col + 2]) = bf16_1;
      } else if constexpr (std::is_same_v<OutType, half>) {
        half2 h2_0 = __float22half2_rn(out0);
        half2 h2_1 = __float22half2_rn(out1);

        *reinterpret_cast<half2*>(&row_output[col]) = h2_0;
        *reinterpret_cast<half2*>(&row_output[col + 2]) = h2_1;
      }
    }
  }
}

template <typename OutType, typename IdType>
__global__ void nvfp4_paged_dequant_kernel(
    const uint8_t* __restrict__ paged_cache, const uint8_t* __restrict__ paged_scales,
    const IdType* __restrict__ block_tables, const int32_t* __restrict__ seq_lens,
    const float* __restrict__ global_scale_ptr, OutType* __restrict__ output, const int batch_size,
    const int max_seq_len, const int block_table_stride, const int num_pages, const int page_size,
    const int num_heads, const int head_dim, const int64_t cache_stride_page,
    const int64_t cache_stride_n, const int64_t cache_stride_h, const int64_t scale_stride_page,
    const int64_t scale_stride_n, const int64_t scale_stride_h) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  const int head = row % num_heads;
  const int token = (row / num_heads) % max_seq_len;
  const int batch = row / (num_heads * max_seq_len);

  if (batch >= batch_size || token >= seq_lens[batch]) return;

  const int page_offset = token / page_size;
  const int entry_idx = token - page_offset * page_size;
  const IdType page = block_tables[batch * block_table_stride + page_offset];
  if (page < 0 || page >= num_pages) return;

  const int packed_dim = head_dim / 2;
  const int scale_dim = head_dim / NVFP4_BLOCK_SIZE;

  const int64_t cache_base = static_cast<int64_t>(page) * cache_stride_page +
                             static_cast<int64_t>(entry_idx) * cache_stride_n +
                             static_cast<int64_t>(head) * cache_stride_h;
  const int64_t scale_base = static_cast<int64_t>(page) * scale_stride_page +
                             static_cast<int64_t>(entry_idx) * scale_stride_n +
                             static_cast<int64_t>(head) * scale_stride_h;

  const uint8_t* row_fp4 = paged_cache + cache_base;
  const uint8_t* row_scales = paged_scales + scale_base;
  OutType* row_output =
      output + ((static_cast<int64_t>(batch) * max_seq_len + token) * num_heads + head) * head_dim;
  const float global_scale = *global_scale_ptr;

  for (int packed_col = tid; packed_col < packed_dim; packed_col += blockDim.x) {
    const uint8_t packed_fp4 = row_fp4[packed_col];
    const int col = packed_col * 2;
    const int scale_idx = col / NVFP4_BLOCK_SIZE;
    const uint8_t scale_fp8 = row_scales[scale_idx];
    const float scale = static_cast<float>(*reinterpret_cast<const __nv_fp8_e4m3*>(&scale_fp8));

    const float out0 = E2M1_LUT[packed_fp4 & 0xF] * scale * global_scale;
    const float out1 = E2M1_LUT[(packed_fp4 >> 4) & 0xF] * scale * global_scale;

    if constexpr (std::is_same_v<OutType, __nv_bfloat16>) {
      *reinterpret_cast<__nv_bfloat162*>(&row_output[col]) =
          __float22bfloat162_rn(make_float2(out0, out1));
    } else if constexpr (std::is_same_v<OutType, half>) {
      *reinterpret_cast<half2*>(&row_output[col]) = __float22half2_rn(make_float2(out0, out1));
    }
  }
}

void nvfp4_kv_dequant(TensorView fp4_data, TensorView block_scales, TensorView global_scale,
                      TensorView output) {
  CHECK_INPUT(fp4_data);
  CHECK_INPUT(block_scales);
  CHECK_CUDA(global_scale);
  CHECK_INPUT(output);

  const int M = fp4_data.size(0);
  const int K = fp4_data.size(1) * 2;

  TVM_FFI_ICHECK(fp4_data.ndim() == 2) << "fp4_data must be 2D";
  TVM_FFI_ICHECK(block_scales.ndim() == 2) << "block_scales must be 2D";
  TVM_FFI_ICHECK(output.ndim() == 2) << "output must be 2D";
  TVM_FFI_ICHECK(output.size(0) == M) << "output row count mismatch";
  TVM_FFI_ICHECK(output.size(1) == K) << "output column count mismatch";
  TVM_FFI_ICHECK(block_scales.size(0) == M) << "block_scales row count mismatch";
  TVM_FFI_ICHECK(K % NVFP4_BLOCK_SIZE == 0)
      << "K dimension must be divisible by " << NVFP4_BLOCK_SIZE;
  TVM_FFI_ICHECK(block_scales.size(1) == K / NVFP4_BLOCK_SIZE)
      << "block_scales column count mismatch";
  TVM_FFI_ICHECK(block_scales.device().device_id == fp4_data.device().device_id)
      << "block_scales must be on the same device as fp4_data";
  TVM_FFI_ICHECK(global_scale.device().device_id == fp4_data.device().device_id)
      << "global_scale must be on the same device as fp4_data";
  TVM_FFI_ICHECK(output.device().device_id == fp4_data.device().device_id)
      << "output must be on the same device as fp4_data";

  ffi::CUDADeviceGuard device_guard(fp4_data.device().device_id);
  cudaStream_t stream = get_stream(fp4_data.device());

  const float* scale_ptr = static_cast<const float*>(global_scale.data_ptr());

  constexpr int BLOCK_SIZE = 128;
  dim3 grid(M);
  dim3 block(BLOCK_SIZE);

  constexpr int ELTS_PER_THREAD = 16;
  const size_t smem_size = sizeof(float) + static_cast<size_t>(K / NVFP4_BLOCK_SIZE);

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(output.dtype(), c_type, [&] {
    nvfp4_dequant_vectorized_kernel<c_type, BLOCK_SIZE, ELTS_PER_THREAD>
        <<<grid, block, smem_size, stream>>>(static_cast<const uint8_t*>(fp4_data.data_ptr()),
                                             static_cast<const uint8_t*>(block_scales.data_ptr()),
                                             scale_ptr, static_cast<c_type*>(output.data_ptr()), M,
                                             K);
    return true;
  });
}

void nvfp4_paged_kv_dequant(TensorView paged_k_cache, TensorView paged_v_cache, TensorView k_scales,
                            TensorView v_scales, TensorView block_tables, TensorView seq_lens,
                            TensorView k_global_scale, TensorView v_global_scale,
                            TensorView output_k, TensorView output_v, int64_t kv_layout) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(paged_k_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(paged_v_cache);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_scales);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_scales);
  CHECK_CUDA(block_tables);
  CHECK_INPUT(seq_lens);
  CHECK_CUDA(k_global_scale);
  CHECK_CUDA(v_global_scale);
  CHECK_CUDA(output_k);
  CHECK_CUDA(output_v);

  TVM_FFI_ICHECK(kv_layout == 0 || kv_layout == 1) << "kv_layout must be 0 (NHD) or 1 (HND)";
  TVM_FFI_ICHECK(paged_k_cache.ndim() == 4) << "paged_k_cache must be 4D";
  TVM_FFI_ICHECK(paged_v_cache.ndim() == 4) << "paged_v_cache must be 4D";
  TVM_FFI_ICHECK(k_scales.ndim() == 4) << "k_scales must be 4D";
  TVM_FFI_ICHECK(v_scales.ndim() == 4) << "v_scales must be 4D";
  TVM_FFI_ICHECK(block_tables.ndim() == 2) << "block_tables must be 2D";
  TVM_FFI_ICHECK(seq_lens.ndim() == 1) << "seq_lens must be 1D";
  TVM_FFI_ICHECK(output_k.ndim() == 4) << "output_k must be 4D";
  TVM_FFI_ICHECK(output_v.ndim() == 4) << "output_v must be 4D";

  TVM_FFI_ICHECK(paged_k_cache.dtype() == dl_uint8) << "paged_k_cache must have dtype uint8";
  TVM_FFI_ICHECK(paged_v_cache.dtype() == dl_uint8) << "paged_v_cache must have dtype uint8";
  TVM_FFI_ICHECK(k_scales.dtype() == dl_float8_e4m3fn || k_scales.dtype() == dl_uint8)
      << "k_scales must have dtype float8_e4m3fn or uint8";
  TVM_FFI_ICHECK(v_scales.dtype() == dl_float8_e4m3fn || v_scales.dtype() == dl_uint8)
      << "v_scales must have dtype float8_e4m3fn or uint8";
  TVM_FFI_ICHECK(seq_lens.dtype() == dl_int32) << "seq_lens must have dtype int32";
  TVM_FFI_ICHECK(k_global_scale.dtype() == dl_float32) << "k_global_scale must have dtype float32";
  TVM_FFI_ICHECK(v_global_scale.dtype() == dl_float32) << "v_global_scale must have dtype float32";
  TVM_FFI_ICHECK(k_global_scale.numel() == 1) << "k_global_scale must be a scalar tensor";
  TVM_FFI_ICHECK(v_global_scale.numel() == 1) << "v_global_scale must be a scalar tensor";
  TVM_FFI_ICHECK(output_k.dtype() == output_v.dtype())
      << "output_k and output_v must have the same dtype";

  const int batch_size = output_k.size(0);
  const int max_seq_len = output_k.size(1);
  const int num_heads = output_k.size(2);
  const int k_head_dim = output_k.size(3);
  const int v_head_dim = output_v.size(3);
  const int k_packed_dim = k_head_dim / 2;
  const int v_packed_dim = v_head_dim / 2;
  const int k_scale_dim = k_head_dim / NVFP4_BLOCK_SIZE;
  const int v_scale_dim = v_head_dim / NVFP4_BLOCK_SIZE;

  TVM_FFI_ICHECK(output_v.size(0) == batch_size) << "output_v batch size mismatch";
  TVM_FFI_ICHECK(output_v.size(1) == max_seq_len) << "output_v max_seq_len mismatch";
  TVM_FFI_ICHECK(output_v.size(2) == num_heads) << "output_v num_heads mismatch";
  TVM_FFI_ICHECK(block_tables.size(0) == batch_size) << "block_tables batch size mismatch";
  TVM_FFI_ICHECK(seq_lens.size(0) == batch_size) << "seq_lens batch size mismatch";
  TVM_FFI_ICHECK(k_head_dim > 0) << "output_k head_dim must be positive";
  TVM_FFI_ICHECK(v_head_dim > 0) << "output_v head_dim must be positive";
  TVM_FFI_ICHECK(k_head_dim % NVFP4_BLOCK_SIZE == 0)
      << "output_k head_dim must be divisible by " << NVFP4_BLOCK_SIZE;
  TVM_FFI_ICHECK(v_head_dim % NVFP4_BLOCK_SIZE == 0)
      << "output_v head_dim must be divisible by " << NVFP4_BLOCK_SIZE;

  const int num_pages = paged_k_cache.size(0);
  int page_size;
  int cache_num_heads;
  if (kv_layout == 0) {
    page_size = paged_k_cache.size(1);
    cache_num_heads = paged_k_cache.size(2);
    TVM_FFI_ICHECK(paged_k_cache.size(3) == k_packed_dim) << "paged_k_cache head_dim mismatch";
    TVM_FFI_ICHECK(paged_v_cache.size(3) == v_packed_dim) << "paged_v_cache head_dim mismatch";
    TVM_FFI_ICHECK(k_scales.size(3) == k_scale_dim) << "k_scales head_dim mismatch";
    TVM_FFI_ICHECK(v_scales.size(3) == v_scale_dim) << "v_scales head_dim mismatch";
  } else {
    cache_num_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
    TVM_FFI_ICHECK(paged_k_cache.size(3) == k_packed_dim) << "paged_k_cache head_dim mismatch";
    TVM_FFI_ICHECK(paged_v_cache.size(3) == v_packed_dim) << "paged_v_cache head_dim mismatch";
    TVM_FFI_ICHECK(k_scales.size(3) == k_scale_dim) << "k_scales head_dim mismatch";
    TVM_FFI_ICHECK(v_scales.size(3) == v_scale_dim) << "v_scales head_dim mismatch";
  }
  TVM_FFI_ICHECK(cache_num_heads == num_heads) << "cache num_heads mismatch";
  TVM_FFI_ICHECK(block_tables.size(1) * page_size >= max_seq_len)
      << "block_tables column count insufficient for max_seq_len";

  auto check_cache_shape = [&](TensorView data, TensorView scales, const int packed_dim,
                               const int scale_dim, const char* name) {
    TVM_FFI_ICHECK(data.size(0) == num_pages) << name << " page count mismatch";
    TVM_FFI_ICHECK(scales.size(0) == num_pages) << name << " scale page count mismatch";
    if (kv_layout == 0) {
      TVM_FFI_ICHECK(data.size(1) == page_size) << name << " page_size mismatch";
      TVM_FFI_ICHECK(data.size(2) == num_heads) << name << " num_heads mismatch";
      TVM_FFI_ICHECK(data.size(3) == packed_dim) << name << " packed_dim mismatch";
      TVM_FFI_ICHECK(scales.size(1) == page_size) << name << " scale page_size mismatch";
      TVM_FFI_ICHECK(scales.size(2) == num_heads) << name << " scale num_heads mismatch";
      TVM_FFI_ICHECK(scales.size(3) == scale_dim) << name << " scale_dim mismatch";
    } else {
      TVM_FFI_ICHECK(data.size(1) == num_heads) << name << " num_heads mismatch";
      TVM_FFI_ICHECK(data.size(2) == page_size) << name << " page_size mismatch";
      TVM_FFI_ICHECK(data.size(3) == packed_dim) << name << " packed_dim mismatch";
      TVM_FFI_ICHECK(scales.size(1) == num_heads) << name << " scale num_heads mismatch";
      TVM_FFI_ICHECK(scales.size(2) == page_size) << name << " scale page_size mismatch";
      TVM_FFI_ICHECK(scales.size(3) == scale_dim) << name << " scale_dim mismatch";
    }
  };
  check_cache_shape(paged_k_cache, k_scales, k_packed_dim, k_scale_dim, "K cache");
  check_cache_shape(paged_v_cache, v_scales, v_packed_dim, v_scale_dim, "V cache");

  CHECK_DEVICE(paged_k_cache, paged_v_cache);
  CHECK_DEVICE(paged_k_cache, k_scales);
  CHECK_DEVICE(paged_k_cache, v_scales);
  CHECK_DEVICE(paged_k_cache, block_tables);
  CHECK_DEVICE(paged_k_cache, seq_lens);
  CHECK_DEVICE(paged_k_cache, k_global_scale);
  CHECK_DEVICE(paged_k_cache, v_global_scale);
  CHECK_DEVICE(paged_k_cache, output_k);
  CHECK_DEVICE(paged_k_cache, output_v);

  auto k_strides = paged_k_cache.strides();
  auto v_strides = paged_v_cache.strides();
  auto k_scale_strides = k_scales.strides();
  auto v_scale_strides = v_scales.strides();
  const int64_t k_stride_page = k_strides[0];
  const int64_t k_stride_n = kv_layout == 1 ? k_strides[2] : k_strides[1];
  const int64_t k_stride_h = kv_layout == 1 ? k_strides[1] : k_strides[2];
  const int64_t v_stride_page = v_strides[0];
  const int64_t v_stride_n = kv_layout == 1 ? v_strides[2] : v_strides[1];
  const int64_t v_stride_h = kv_layout == 1 ? v_strides[1] : v_strides[2];
  const int64_t k_scale_stride_page = k_scale_strides[0];
  const int64_t k_scale_stride_n = kv_layout == 1 ? k_scale_strides[2] : k_scale_strides[1];
  const int64_t k_scale_stride_h = kv_layout == 1 ? k_scale_strides[1] : k_scale_strides[2];
  const int64_t v_scale_stride_page = v_scale_strides[0];
  const int64_t v_scale_stride_n = kv_layout == 1 ? v_scale_strides[2] : v_scale_strides[1];
  const int64_t v_scale_stride_h = kv_layout == 1 ? v_scale_strides[1] : v_scale_strides[2];

  ffi::CUDADeviceGuard device_guard(paged_k_cache.device().device_id);
  cudaStream_t stream = get_stream(paged_k_cache.device());

  if (batch_size == 0 || max_seq_len == 0 || num_heads == 0) {
    return;
  }

  CHECK_INPUT(block_tables);
  CHECK_INPUT(output_k);
  CHECK_INPUT(output_v);

  const int k_block_size = k_packed_dim < 128 ? k_packed_dim : 128;
  const int v_block_size = v_packed_dim < 128 ? v_packed_dim : 128;
  dim3 k_block(k_block_size);
  dim3 v_block(v_block_size);
  dim3 grid(batch_size * max_seq_len * num_heads);

  DISPATCH_DLPACK_IDTYPE_TO_CTYPE(block_tables.dtype(), id_type, [&] {
    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(output_k.dtype(), out_type, [&] {
      nvfp4_paged_dequant_kernel<out_type, id_type><<<grid, k_block, 0, stream>>>(
          static_cast<const uint8_t*>(paged_k_cache.data_ptr()),
          static_cast<const uint8_t*>(k_scales.data_ptr()),
          static_cast<const id_type*>(block_tables.data_ptr()),
          static_cast<const int32_t*>(seq_lens.data_ptr()),
          static_cast<const float*>(k_global_scale.data_ptr()),
          static_cast<out_type*>(output_k.data_ptr()), batch_size, max_seq_len,
          block_tables.size(1), num_pages, page_size, num_heads, k_head_dim, k_stride_page,
          k_stride_n, k_stride_h, k_scale_stride_page, k_scale_stride_n, k_scale_stride_h);
      FLASHINFER_CUDA_CHECK(cudaGetLastError());
      nvfp4_paged_dequant_kernel<out_type, id_type><<<grid, v_block, 0, stream>>>(
          static_cast<const uint8_t*>(paged_v_cache.data_ptr()),
          static_cast<const uint8_t*>(v_scales.data_ptr()),
          static_cast<const id_type*>(block_tables.data_ptr()),
          static_cast<const int32_t*>(seq_lens.data_ptr()),
          static_cast<const float*>(v_global_scale.data_ptr()),
          static_cast<out_type*>(output_v.data_ptr()), batch_size, max_seq_len,
          block_tables.size(1), num_pages, page_size, num_heads, v_head_dim, v_stride_page,
          v_stride_n, v_stride_h, v_scale_stride_page, v_scale_stride_n, v_scale_stride_h);
      FLASHINFER_CUDA_CHECK(cudaGetLastError());
      return true;
    });
    return true;
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_kv_dequant, nvfp4_kv_dequant);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_paged_kv_dequant, nvfp4_paged_kv_dequant);
