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

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

namespace flashinfer::gpt_oss_ops {

// SM100 exposes a packed FP32-to-E4M3 conversion that writes two FP8 values at a
// time. The host wrapper checks the device capability before launch; the trap
// keeps accidental non-SM100 instantiations loud if this source is built for an
// older target.
__device__ __forceinline__ uint16_t cvt_e4m3x2(float hi, float lo) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint16_t out;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(out) : "f"(hi), "f"(lo));
  return out;
#else
  __trap();
  return 0;
#endif
}

constexpr int kPageSize = 16;
constexpr int kPageSizeLog2 = 4;

#define DISPATCH_WARPS_PER_CTA(total_items, WARPS_PER_CTA, ...) \
  do {                                                          \
    if ((total_items) <= 1) {                                   \
      constexpr int WARPS_PER_CTA = 1;                          \
      __VA_ARGS__                                               \
    } else if ((total_items) <= 2) {                            \
      constexpr int WARPS_PER_CTA = 2;                          \
      __VA_ARGS__                                               \
    } else if ((total_items) <= 4) {                            \
      constexpr int WARPS_PER_CTA = 4;                          \
      __VA_ARGS__                                               \
    } else {                                                    \
      constexpr int WARPS_PER_CTA = 8;                          \
      __VA_ARGS__                                               \
    }                                                           \
  } while (0)

template <int WARPS_PER_CTA>
__global__ __launch_bounds__(WARPS_PER_CTA * 32) void fp8_cache_scatter_kernel(
    const __nv_bfloat16* __restrict__ key, const __nv_bfloat16* __restrict__ value,
    uint8_t* __restrict__ key_cache, uint8_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping, const float* __restrict__ k_scale,
    const float* __restrict__ v_scale, int num_tokens, int num_heads, int64_t key_stride,
    int64_t key_head_stride, int64_t value_stride, int64_t value_head_stride,
    int64_t key_block_stride, int64_t key_page_stride, int64_t key_head_cache_stride,
    int64_t value_block_stride, int64_t value_page_stride, int64_t value_head_cache_stride,
    int64_t max_slots) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int item = static_cast<int>(blockIdx.x) * WARPS_PER_CTA + warp;
  const int total_items = num_tokens * num_heads;
  if (item >= total_items) {
    return;
  }

  // One warp owns one (token, head) row. Each lane handles two adjacent head
  // elements, covering the fixed 64-wide head dimension with coalesced 16-bit
  // loads from BF16 input and 16-bit packed stores into FP8 cache storage.
  const int token = item / num_heads;
  const int head = item - token * num_heads;
  const int64_t slot = __ldg(slot_mapping + token);
  // vLLM-style cache updates may use -1 entries for padded or skipped tokens.
  if (slot < 0 || slot >= max_slots) {
    return;
  }

  const __nv_bfloat162* __restrict__ k_src =
      reinterpret_cast<const __nv_bfloat162*>(key + token * key_stride + head * key_head_stride);
  const __nv_bfloat162* __restrict__ v_src = reinterpret_cast<const __nv_bfloat162*>(
      value + token * value_stride + head * value_head_stride);
  const __nv_bfloat162 kpair = __ldg(k_src + lane);
  const __nv_bfloat162 vpair = __ldg(v_src + lane);
  const float ks = __ldg(k_scale);
  const float vs = __ldg(v_scale);

  const float kinv = 1.0f / ks;
  const float vinv = 1.0f / vs;
  float2 key_float = __bfloat1622float2(kpair);
  float2 value_float = __bfloat1622float2(vpair);
  key_float.x *= kinv;
  key_float.y *= kinv;
  value_float.x *= vinv;
  value_float.y *= vinv;

  const uint16_t ko = cvt_e4m3x2(key_float.y, key_float.x);
  const uint16_t vo = cvt_e4m3x2(value_float.y, value_float.x);

  const int64_t block = slot >> kPageSizeLog2;
  const int64_t offset = slot & (kPageSize - 1);
  // Cache tensors are laid out as [num_blocks, 16, num_heads, 64]. Strides are
  // byte offsets because the cache may be passed either as uint8 storage or as
  // torch.float8_e4m3fn.
  const int64_t key_dst =
      block * key_block_stride + offset * key_page_stride + head * key_head_cache_stride;
  const int64_t value_dst =
      block * value_block_stride + offset * value_page_stride + head * value_head_cache_stride;
  reinterpret_cast<uint16_t*>(key_cache + key_dst)[lane] = ko;
  reinterpret_cast<uint16_t*>(value_cache + value_dst)[lane] = vo;
}

void fp8_cache_scatter_launcher(const void* key, const void* value, void* key_cache,
                                void* value_cache, const int64_t* slot_mapping,
                                const float* k_scale, const float* v_scale, int num_tokens,
                                int num_heads, int64_t key_stride, int64_t key_head_stride,
                                int64_t value_stride, int64_t value_head_stride,
                                int64_t key_block_stride, int64_t key_page_stride,
                                int64_t key_head_cache_stride, int64_t value_block_stride,
                                int64_t value_page_stride, int64_t value_head_cache_stride,
                                int64_t max_slots, cudaStream_t stream) {
  if (num_tokens <= 0 || num_heads <= 0) {
    return;
  }

  const __nv_bfloat16* k = reinterpret_cast<const __nv_bfloat16*>(key);
  const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(value);
  uint8_t* kc = reinterpret_cast<uint8_t*>(key_cache);
  uint8_t* vc = reinterpret_cast<uint8_t*>(value_cache);

  const int total_items = num_tokens * num_heads;
  // Small updates are common in decode. Match the number of active warps to the
  // amount of work for tiny batches, then use 8 warps per CTA once there is
  // enough independent (token, head) work to fill them.
  DISPATCH_WARPS_PER_CTA(total_items, kWarps, {
    const int grid = (total_items + kWarps - 1) / kWarps;
    fp8_cache_scatter_kernel<kWarps><<<grid, kWarps * 32, 0, stream>>>(
        k, v, kc, vc, slot_mapping, k_scale, v_scale, num_tokens, num_heads, key_stride,
        key_head_stride, value_stride, value_head_stride, key_block_stride, key_page_stride,
        key_head_cache_stride, value_block_stride, value_page_stride, value_head_cache_stride,
        max_slots);
  });
}

void reshape_and_cache_fp8(TensorView key, TensorView value, TensorView key_cache,
                           TensorView value_cache, TensorView slot_mapping, TensorView k_scale,
                           TensorView v_scale) {
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(key_cache);
  CHECK_CUDA(value_cache);
  CHECK_INPUT(slot_mapping);
  CHECK_INPUT(k_scale);
  CHECK_INPUT(v_scale);
  CHECK_DIM(3, key);
  CHECK_DIM(3, value);
  CHECK_DIM(4, key_cache);
  CHECK_DIM(4, value_cache);
  CHECK_DIM(1, slot_mapping);
  CHECK_LAST_DIM_CONTIGUOUS(key);
  CHECK_LAST_DIM_CONTIGUOUS(value);
  CHECK_LAST_DIM_CONTIGUOUS(key_cache);
  CHECK_LAST_DIM_CONTIGUOUS(value_cache);
  CHECK_DEVICE(value, key);
  CHECK_DEVICE(key_cache, key);
  CHECK_DEVICE(value_cache, key);
  CHECK_DEVICE(slot_mapping, key);
  CHECK_DEVICE(k_scale, key);
  CHECK_DEVICE(v_scale, key);
  TVM_FFI_ICHECK_EQ(key.dtype(), dl_bfloat16) << "key must be bfloat16";
  TVM_FFI_ICHECK_EQ(value.dtype(), dl_bfloat16) << "value must be bfloat16";
  TVM_FFI_ICHECK(key_cache.dtype() == dl_uint8 || key_cache.dtype() == dl_float8_e4m3fn)
      << "key_cache must be uint8 or float8_e4m3fn";
  TVM_FFI_ICHECK(value_cache.dtype() == dl_uint8 || value_cache.dtype() == dl_float8_e4m3fn)
      << "value_cache must be uint8 or float8_e4m3fn";
  TVM_FFI_ICHECK_EQ(slot_mapping.dtype(), dl_int64) << "slot_mapping must be int64";
  TVM_FFI_ICHECK_EQ(k_scale.dtype(), dl_float32) << "k_scale must be float32";
  TVM_FFI_ICHECK_EQ(v_scale.dtype(), dl_float32) << "v_scale must be float32";
  TVM_FFI_ICHECK_LE(k_scale.ndim(), 1) << "k_scale must be scalar or 1D";
  TVM_FFI_ICHECK_LE(v_scale.ndim(), 1) << "v_scale must be scalar or 1D";
  TVM_FFI_ICHECK(k_scale.ndim() == 0 || k_scale.size(0) == 1) << "k_scale must have one element";
  TVM_FFI_ICHECK(v_scale.ndim() == 0 || v_scale.size(0) == 1) << "v_scale must have one element";

  // GPT-OSS paged FP8 caches use a fixed 64-wide BF16 K/V row and a 16-token
  // FP8 page. Other page layouts, head dimensions, or cache dtypes should use
  // the framework's generic cache update path.
  const int64_t num_tokens = key.size(0);
  const int64_t num_heads = key.size(1);
  TVM_FFI_ICHECK_EQ(value.size(0), num_tokens) << "key/value token count mismatch";
  TVM_FFI_ICHECK_EQ(slot_mapping.size(0), num_tokens) << "slot_mapping token count mismatch";
  TVM_FFI_ICHECK_EQ(value.size(1), num_heads) << "key/value head count mismatch";
  TVM_FFI_ICHECK_EQ(key.size(2), 64) << "key head_dim must be 64";
  TVM_FFI_ICHECK_EQ(value.size(2), 64) << "value head_dim must be 64";
  TVM_FFI_ICHECK_EQ(key_cache.size(1), 16) << "key_cache block size must be 16";
  TVM_FFI_ICHECK_EQ(value_cache.size(1), 16) << "value_cache block size must be 16";
  TVM_FFI_ICHECK_EQ(key_cache.size(2), num_heads) << "key_cache head count mismatch";
  TVM_FFI_ICHECK_EQ(value_cache.size(2), num_heads) << "value_cache head count mismatch";
  TVM_FFI_ICHECK_EQ(key_cache.size(3), 64) << "key_cache head_dim must be 64";
  TVM_FFI_ICHECK_EQ(value_cache.size(3), 64) << "value_cache head_dim must be 64";
  TVM_FFI_ICHECK_GE(key.stride(1), 64) << "key head stride must be at least 64";
  TVM_FFI_ICHECK_GE(value.stride(1), 64) << "value head stride must be at least 64";
  TVM_FFI_ICHECK_EQ(key_cache.stride(1), 64) << "key_cache token stride must be 64";
  TVM_FFI_ICHECK_EQ(value_cache.stride(1), 64) << "value_cache token stride must be 64";
  TVM_FFI_ICHECK_GE(key_cache.stride(2), 64) << "key_cache head stride must be at least 64";
  TVM_FFI_ICHECK_GE(value_cache.stride(2), 64) << "value_cache head stride must be at least 64";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(key.data_ptr()) % alignof(__nv_bfloat162), 0)
      << "key data pointer must be aligned for bfloat162 loads";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(value.data_ptr()) % alignof(__nv_bfloat162), 0)
      << "value data pointer must be aligned for bfloat162 loads";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(key_cache.data_ptr()) % alignof(uint16_t), 0)
      << "key_cache data pointer must be aligned for uint16 stores";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(value_cache.data_ptr()) % alignof(uint16_t), 0)
      << "value_cache data pointer must be aligned for uint16 stores";
  TVM_FFI_ICHECK_EQ(key.stride(0) % 2, 0) << "key token stride must be even";
  TVM_FFI_ICHECK_EQ(key.stride(1) % 2, 0) << "key head stride must be even";
  TVM_FFI_ICHECK_EQ(value.stride(0) % 2, 0) << "value token stride must be even";
  TVM_FFI_ICHECK_EQ(value.stride(1) % 2, 0) << "value head stride must be even";
  TVM_FFI_ICHECK_EQ(key_cache.stride(0) % 2, 0) << "key_cache block stride must be even";
  TVM_FFI_ICHECK_EQ(key_cache.stride(2) % 2, 0) << "key_cache head stride must be even";
  TVM_FFI_ICHECK_EQ(value_cache.stride(0) % 2, 0) << "value_cache block stride must be even";
  TVM_FFI_ICHECK_EQ(value_cache.stride(2) % 2, 0) << "value_cache head stride must be even";

  ffi::CUDADeviceGuard device_guard(key.device().device_id);
  cudaDeviceProp prop;
  auto status = cudaGetDeviceProperties(&prop, key.device().device_id);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cudaGetDeviceProperties failed: " << cudaGetErrorString(status);
  TVM_FFI_ICHECK_GE(prop.major, 10) << "reshape_and_cache_fp8 requires SM100 or newer";
  const cudaStream_t stream = get_stream(key.device());
  const int64_t max_slots = key_cache.size(0) * key_cache.size(1);
  fp8_cache_scatter_launcher(
      key.data_ptr(), value.data_ptr(), key_cache.data_ptr(), value_cache.data_ptr(),
      static_cast<int64_t*>(slot_mapping.data_ptr()), static_cast<float*>(k_scale.data_ptr()),
      static_cast<float*>(v_scale.data_ptr()), static_cast<int>(num_tokens),
      static_cast<int>(num_heads), key.stride(0), key.stride(1), value.stride(0), value.stride(1),
      key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), value_cache.stride(0),
      value_cache.stride(1), value_cache.stride(2), max_slots, stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(reshape_and_cache_fp8,
                              flashinfer::gpt_oss_ops::reshape_and_cache_fp8);

#undef DISPATCH_WARPS_PER_CTA

}  // namespace flashinfer::gpt_oss_ops
