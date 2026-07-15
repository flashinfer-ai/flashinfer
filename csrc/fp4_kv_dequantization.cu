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

// Exact fp32 E2M1 value from a nibble, with no constant-memory table. Produces the
// same values as E2M1_LUT bit-for-bit: e in {1,2,3} -> 2^(e-1)*(1 + 0.5*m); e==0 -> 0.5*m.
__device__ __forceinline__ float e2m1_to_f32(uint32_t n) {
  uint32_t e = (n >> 1) & 0x3u;
  uint32_t m = n & 0x1u;
  float mag = (e == 0u) ? (0.5f * m) : (exp2f(static_cast<float>(static_cast<int>(e) - 1)) *
                                        (1.0f + 0.5f * m));
  return (n & 0x8u) ? -mag : mag;
}

// Blockwise dequant: one thread per 16-element NVFP4 block. Each thread does a single
// uint2 load (8 packed bytes = 16 nibbles), reads the one FP8 E4M3 block scale it needs,
// dequantizes in fp32 (multiply-then-round — numerically identical to the previous
// LUT-based kernel, which also multiplied in fp32 and rounded once), and writes 16 outputs
// as two 128-bit float4 stores. Grid-strided over all M*(K/16) blocks.
//
// This replaces the row-per-block mapping (grid(M), shared-memory scale staging) which is
// memory-latency bound and reaches only ~20% of peak bandwidth on large tensors; the
// blockwise mapping with wide vectorized loads/stores reaches ~65% (~3x faster) while
// producing byte-identical output. Block scales/data/output are all contiguous and K is a
// multiple of 16, so the flat block index addresses each cleanly.
template <typename OutType>
__global__ void nvfp4_dequant_blockwise_kernel(const uint2* __restrict__ fp4_data,
                                               const uint8_t* __restrict__ block_scales,
                                               const float* __restrict__ global_scale_ptr,
                                               OutType* __restrict__ output,
                                               const long num_blocks_total) {
  const float global_scale = __ldg(global_scale_ptr);
  const long stride = static_cast<long>(gridDim.x) * blockDim.x;
  for (long bidx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
       bidx < num_blocks_total; bidx += stride) {
    uint2 packed = fp4_data[bidx];
    __nv_fp8_e4m3 sc;
    sc.__x = __ldg(&block_scales[bidx]);
    const float scale = static_cast<float>(sc) * global_scale;

    OutType res[16];
    const uint32_t words[2] = {packed.x, packed.y};
#pragma unroll
    for (int w = 0; w < 2; ++w) {
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const uint32_t byte = (words[w] >> (b * 8)) & 0xFFu;
        float2 o;
        o.x = e2m1_to_f32(byte & 0xFu) * scale;
        o.y = e2m1_to_f32((byte >> 4) & 0xFu) * scale;
        if constexpr (std::is_same_v<OutType, __nv_bfloat16>) {
          *reinterpret_cast<__nv_bfloat162*>(&res[w * 8 + b * 2]) = __float22bfloat162_rn(o);
        } else {
          *reinterpret_cast<half2*>(&res[w * 8 + b * 2]) = __float22half2_rn(o);
        }
      }
    }
    float4* out_vec = reinterpret_cast<float4*>(output + bidx * 16);
    const float4* res_vec = reinterpret_cast<const float4*>(res);
    out_vec[0] = res_vec[0];
    out_vec[1] = res_vec[1];
  }
}

// Paged dequant, blockwise: one thread per 16-element NVFP4 block, grid-strided over all
// batch*seq*head*(head_dim/16) blocks. Each thread does the page lookup once, a uint2 load
// (8 packed bytes = 16 nibbles), reads its one FP8 scale, dequantizes in fp32 with the same
// multiply order as before — (e2m1 * block_scale) * global_scale, so output is byte-identical
// — and writes 16 outputs as two float4 stores. Invalid rows (token >= seq_len, or an
// out-of-range page) are skipped without writing, exactly as before. This replaces the
// one-block-per-(batch,token,head) mapping with scalar 1-byte loads, which is memory-latency
// bound; the blockwise mapping with wide transactions is ~2.7x faster on large caches.
template <typename OutType, typename IdType>
__global__ void nvfp4_paged_dequant_blockwise_kernel(
    const uint8_t* __restrict__ paged_cache, const uint8_t* __restrict__ paged_scales,
    const IdType* __restrict__ block_tables, const int32_t* __restrict__ seq_lens,
    const float* __restrict__ global_scale_ptr, OutType* __restrict__ output, const int batch_size,
    const int max_seq_len, const int block_table_stride, const int num_pages, const int page_size,
    const int num_heads, const int head_dim, const int64_t cache_stride_page,
    const int64_t cache_stride_n, const int64_t cache_stride_h, const int64_t scale_stride_page,
    const int64_t scale_stride_n, const int64_t scale_stride_h) {
  const float global_scale = __ldg(global_scale_ptr);
  const int blocks_per_row = head_dim / NVFP4_BLOCK_SIZE;
  const long total = static_cast<long>(batch_size) * max_seq_len * num_heads * blocks_per_row;
  const long stride = static_cast<long>(gridDim.x) * blockDim.x;
  for (long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    const int blk = idx % blocks_per_row;
    const long row = idx / blocks_per_row;
    const int head = row % num_heads;
    const int token = (row / num_heads) % max_seq_len;
    const int batch = row / (static_cast<long>(num_heads) * max_seq_len);
    if (token >= seq_lens[batch]) continue;
    const int page_offset = token / page_size;
    const int entry_idx = token - page_offset * page_size;
    const IdType page = block_tables[batch * block_table_stride + page_offset];
    if (page < 0 || page >= num_pages) continue;

    const int64_t cache_base = static_cast<int64_t>(page) * cache_stride_page +
                               static_cast<int64_t>(entry_idx) * cache_stride_n +
                               static_cast<int64_t>(head) * cache_stride_h;
    const int64_t scale_base = static_cast<int64_t>(page) * scale_stride_page +
                               static_cast<int64_t>(entry_idx) * scale_stride_n +
                               static_cast<int64_t>(head) * scale_stride_h;
    uint2 packed =
        *reinterpret_cast<const uint2*>(paged_cache + cache_base + static_cast<int64_t>(blk) * 8);
    __nv_fp8_e4m3 sc;
    sc.__x = __ldg(paged_scales + scale_base + blk);
    const float scale = static_cast<float>(sc);

    OutType res[16];
    const uint32_t words[2] = {packed.x, packed.y};
#pragma unroll
    for (int w = 0; w < 2; ++w) {
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const uint32_t byte = (words[w] >> (b * 8)) & 0xFFu;
        float2 o;
        o.x = e2m1_to_f32(byte & 0xFu) * scale * global_scale;
        o.y = e2m1_to_f32((byte >> 4) & 0xFu) * scale * global_scale;
        if constexpr (std::is_same_v<OutType, __nv_bfloat16>) {
          *reinterpret_cast<__nv_bfloat162*>(&res[w * 8 + b * 2]) = __float22bfloat162_rn(o);
        } else {
          *reinterpret_cast<half2*>(&res[w * 8 + b * 2]) = __float22half2_rn(o);
        }
      }
    }
    OutType* row_out = output +
                       ((static_cast<int64_t>(batch) * max_seq_len + token) * num_heads + head) *
                           head_dim +
                       static_cast<int64_t>(blk) * 16;
    float4* out_vec = reinterpret_cast<float4*>(row_out);
    const float4* res_vec = reinterpret_cast<const float4*>(res);
    out_vec[0] = res_vec[0];
    out_vec[1] = res_vec[1];
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

  // One thread per 16-element NVFP4 block, grid-strided. Wide vectorized loads/stores;
  // grid capped so the launch stays lean (the grid-stride loop covers the remainder).
  const long num_blocks_total = static_cast<long>(M) * (K / NVFP4_BLOCK_SIZE);
  constexpr int THREADS = 256;
  long blocks = (num_blocks_total + THREADS - 1) / THREADS;
  const long MAX_BLOCKS = 108L * 32;
  if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;
  if (blocks < 1) blocks = 1;

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(output.dtype(), c_type, [&] {
    nvfp4_dequant_blockwise_kernel<c_type><<<blocks, THREADS, 0, stream>>>(
        reinterpret_cast<const uint2*>(fp4_data.data_ptr()),
        static_cast<const uint8_t*>(block_scales.data_ptr()), scale_ptr,
        static_cast<c_type*>(output.data_ptr()), num_blocks_total);
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

  // One thread per 16-element NVFP4 block, grid-strided over all
  // batch*seq*head*(head_dim/16) blocks; grid capped so the launch stays lean.
  constexpr int THREADS = 256;
  const long MAX_BLOCKS = 108L * 32;
  auto grid_for = [&](int head_dim) {
    const long total =
        static_cast<long>(batch_size) * max_seq_len * num_heads * (head_dim / NVFP4_BLOCK_SIZE);
    long blocks = (total + THREADS - 1) / THREADS;
    if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;
    if (blocks < 1) blocks = 1;
    return static_cast<unsigned int>(blocks);
  };

  DISPATCH_DLPACK_IDTYPE_TO_CTYPE(block_tables.dtype(), id_type, [&] {
    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(output_k.dtype(), out_type, [&] {
      nvfp4_paged_dequant_blockwise_kernel<out_type, id_type>
          <<<grid_for(k_head_dim), THREADS, 0, stream>>>(
              static_cast<const uint8_t*>(paged_k_cache.data_ptr()),
              static_cast<const uint8_t*>(k_scales.data_ptr()),
              static_cast<const id_type*>(block_tables.data_ptr()),
              static_cast<const int32_t*>(seq_lens.data_ptr()),
              static_cast<const float*>(k_global_scale.data_ptr()),
              static_cast<out_type*>(output_k.data_ptr()), batch_size, max_seq_len,
              block_tables.size(1), num_pages, page_size, num_heads, k_head_dim, k_stride_page,
              k_stride_n, k_stride_h, k_scale_stride_page, k_scale_stride_n, k_scale_stride_h);
      FLASHINFER_CUDA_CHECK(cudaGetLastError());
      nvfp4_paged_dequant_blockwise_kernel<out_type, id_type>
          <<<grid_for(v_head_dim), THREADS, 0, stream>>>(
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
