/*
 * Copyright (c) 2026 by NVIDIA Corporation.
 *
 * Adapted from SageAttention's fused low-precision Ulysses quantization
 * kernels: https://github.com/thu-ml/SageAttention
 * (numeric conversion inspired by CUTLASS; block reductions adapted from
 * vLLM / FasterTransformer).
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

// Low-precision Ulysses all-to-all quantization kernels (payload ABI v3,
// stats protocol 3 / ALIGN-128, "V2-G" global-grid form).
//
// Torch-free kernel header: raw pointers and scalar parameters only. Host
// launchers live in csrc/ulysses_lowp.cu; the kernel math below is a
// bit-exact port of the pinned SageAttention fork kernels -- element order,
// rounding, clamps, multiply/divide order and PTX conversions are
// load-bearing and must not be changed.

#ifndef FLASHINFER_COMM_ULYSSES_LOWP_CUH_
#define FLASHINFER_COMM_ULYSSES_LOWP_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace flashinfer {
namespace ulysses_lowp {

// Low-precision Ulysses payload ABI version (protocol 3 == ALIGN-128 chunks).
constexpr uint32_t kAbiVersion = 3;
// Fixed sequence-chunk length of the two-stage KSumVAmax reduction.
constexpr uint32_t KSUM_CHUNK_TOKENS = 256;

namespace detail {

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120400)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890))
#define FLASHINFER_ULYSSES_LOWP_FP8_CAST_ENABLED
#endif
#endif

#if defined(__CUDA_ARCH__)
#define FLASHINFER_ULYSSES_LOWP_RUNTIME_ASSERT(x) __brkpt()
#else
#include <assert.h>
#define FLASHINFER_ULYSSES_LOWP_RUNTIME_ASSERT(x) assert(0 && x)
#endif

__device__ __forceinline__ void floatx4_to_e4m3x4(uint32_t *dest, float *source0, float *source1)
{
#ifdef FLASHINFER_ULYSSES_LOWP_FP8_CAST_ENABLED
  asm volatile( \
      "{\n" \
      ".reg .b16 lo;\n" \
      ".reg .b16 hi;\n" \
      "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n" \
      "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n" \
      "mov.b32 %0, {lo, hi};\n" \
      "}" \
      : "=r"(dest[0]) : "f"(source0[0]), "f"(source0[1]), "f"(source1[0]), "f"(source1[1]));
#else
  FLASHINFER_ULYSSES_LOWP_RUNTIME_ASSERT("Unsupported CUDA architecture for FP8 CAST instruction");
#endif
}

__device__ __forceinline__ int8_t float_to_int8_rn(float x)
{
    uint32_t dst;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
    return reinterpret_cast<const int8_t&>(dst);
}

template<typename T>
__inline__ __device__ T warpReduceMax(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    return val;
}
/* Calculate the maximum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceMax(T val)
{
    static __shared__ T shared[32];
    int                 lane = threadIdx.x & 0x1f;  // in-warp idx
    int                 wid  = threadIdx.x >> 5;    // warp idx
    val = warpReduceMax(val);  // get maxx in each warp
    if (lane == 0)  // record in-warp maxx by warp Idx
        shared[wid] = val;
    __syncthreads();
    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);
    return val;
}

template <typename T>
__device__ __forceinline__ float convert_to_float(T val)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");

  if constexpr (std::is_same<T, half>::value)
  {
    return __half2float(val);
  }
  else if constexpr (std::is_same<T, nv_bfloat16>::value)
  {
    return __bfloat162float(val);
  }
}

template <typename T>
__device__ __forceinline__ T convert_from_float(float val)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");

  if constexpr (std::is_same<T, half>::value)
  {
    return __float2half_rn(val);
  }
  else if constexpr (std::is_same<T, nv_bfloat16>::value)
  {
    return __float2bfloat16_rn(val);
  }
}

}  // namespace detail

// ============================================================================
// V2-G global-grid (payload ABI v3)
//
// V2-G keeps ordinary Sage2's global 32/64-token quantization grids across
// rank boundaries.  Every token is quantized with the FINAL scale of its
// global group; boundary-group scales are max-merged across ranks before
// quantization.  Contract: SM120_ULYSSES_LOWP_A2A_V2G_ABI_CONTRACT.md.
// ============================================================================

namespace grid {

// slots(L,G) = ceil((L+G-1)/G): upper bound of groups a length-L interval can
// touch at any offset.  NOT the per-rank valid count.
__host__ __device__ __forceinline__ int64_t slots(int64_t L, int64_t G)
{
  return (L + 2 * G - 2) / G;
}

__host__ __device__ __forceinline__ int64_t group_first(int64_t rank, int64_t L, int64_t G)
{
  return (rank * L) / G;
}

__host__ __device__ __forceinline__ int64_t group_last(int64_t rank, int64_t L, int64_t G)
{
  return (rank * L + L - 1) / G;
}

struct ChunkSpec
{
  int64_t q_slots;
  int64_t k_slots;
  int64_t main_bytes;
  int64_t q_scale_bytes;
  int64_t k_scale_bytes;
  int64_t raw_chunk_bytes;
  int64_t chunk_bytes;
};

inline ChunkSpec chunk_spec(int64_t batch_size, int64_t local_sequence,
                            int64_t local_heads, int64_t head_dim)
{
  ChunkSpec spec;
  spec.q_slots = slots(local_sequence, 32);
  spec.k_slots = slots(local_sequence, 64);
  spec.main_bytes = batch_size * local_sequence * local_heads * head_dim;
  spec.q_scale_bytes = batch_size * local_heads * spec.q_slots *
                       static_cast<int64_t>(sizeof(float));
  spec.k_scale_bytes = batch_size * local_heads * spec.k_slots *
                       static_cast<int64_t>(sizeof(float));
  spec.raw_chunk_bytes = 3 * spec.main_bytes + spec.q_scale_bytes + spec.k_scale_bytes;
  spec.chunk_bytes = (spec.raw_chunk_bytes + 127) / 128 * 128;
  return spec;
}

}  // namespace grid

// Quantize canonical NHD V with an externally supplied per-channel scale.
// The output intentionally remains canonical uint8 FP8 bits for low-precision
// Ulysses communication; Sage's sequence permutation is a separate operation.
template <typename T>
__global__ void QuantVFP8WithScaleKernel(
    const T *__restrict__ input,
    const float *__restrict__ scale,
    int8_t *__restrict__ output,
    const uint64_t num_packs,
    const uint32_t num_tokens,
    const uint32_t num_heads,
    const uint32_t head_dim)
{
  constexpr uint32_t pack_size = 8;
  const uint64_t pack_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pack_id >= num_packs)
  {
    return;
  }

  const uint32_t packs_per_head = head_dim / pack_size;
  const uint32_t d_base = static_cast<uint32_t>(pack_id % packs_per_head) * pack_size;
  const uint64_t token_head_id = pack_id / packs_per_head;
  const uint32_t head_id = static_cast<uint32_t>(token_head_id % num_heads);
  const uint32_t batch_id = static_cast<uint32_t>(token_head_id / (static_cast<uint64_t>(num_tokens) * num_heads));
  const uint64_t element_offset = pack_id * pack_size;
  const uint64_t scale_offset =
      (static_cast<uint64_t>(batch_id) * num_heads + head_id) * head_dim + d_base;

  T x_val[pack_size];
  float scale_val[pack_size];
  float x_val_float[pack_size];
  uint32_t x_val_fp8[2];

  *reinterpret_cast<float4 *>(&x_val[0]) =
      *reinterpret_cast<const float4 *>(input + element_offset);
  *reinterpret_cast<float4 *>(&scale_val[0]) =
      *reinterpret_cast<const float4 *>(scale + scale_offset);
  *reinterpret_cast<float4 *>(&scale_val[4]) =
      *reinterpret_cast<const float4 *>(scale + scale_offset + 4);

#pragma unroll
  for (uint32_t i = 0; i < pack_size; ++i)
  {
    // Keep the pinned kernel's multiply-by-reciprocal behavior. In particular,
    // zero input with zero scale converts NaN to the same E4M3 bit pattern.
    // Reconstruct the source amax before taking the reciprocal. The amax is
    // selected from BF16/FP16 input values, so rounding the reconstructed
    // value back to T recovers information lost when amax / 2.25 was stored
    // as FP32. This matches the pinned quantizer's 2.25 / amax operation.
    const float amax_val = detail::convert_to_float(detail::convert_from_float<T>(scale_val[i] * 2.25f));
    x_val_float[i] = detail::convert_to_float(x_val[i]) * __fdividef(2.25f, amax_val);
  }

  detail::floatx4_to_e4m3x4(x_val_fp8, x_val_float, x_val_float + 2);
  detail::floatx4_to_e4m3x4(x_val_fp8 + 1, x_val_float + 4, x_val_float + 6);
  *reinterpret_cast<uint2 *>(output + element_offset) =
      *reinterpret_cast<uint2 *>(&x_val_fp8[0]);
}

// Quantize canonical NHD V directly into the destination-major V section of
// the V1 communication payload. This removes the canonical FP8 intermediate
// and the subsequent full-tensor pack read while preserving the pinned E4M3
// conversion exactly.
template <typename T>
__global__ void QuantVFP8WithScalePackKernel(
    const T *__restrict__ input,
    const float *__restrict__ scale,
    uint8_t *__restrict__ output,
    const uint64_t num_packs,
    const uint32_t num_tokens,
    const uint32_t num_heads,
    const uint32_t local_heads,
    const uint32_t head_dim,
    const uint32_t batch_size,
    const uint64_t main_bytes,
    const uint64_t chunk_bytes,
    const int64_t stride_batch,
    const int64_t stride_token,
    const int64_t stride_head)
{
  constexpr uint32_t pack_size = 8;
  const uint64_t pack_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pack_id >= num_packs)
  {
    return;
  }

  const uint32_t packs_per_head = head_dim / pack_size;
  uint64_t logical_id = pack_id;
  const uint32_t d_base = static_cast<uint32_t>(logical_id % packs_per_head) * pack_size;
  logical_id /= packs_per_head;
  const uint32_t local_head = static_cast<uint32_t>(logical_id % local_heads);
  logical_id /= local_heads;
  const uint32_t batch_id = static_cast<uint32_t>(logical_id % batch_size);
  const uint32_t token_id = static_cast<uint32_t>(logical_id / batch_size);
  const uint32_t destination = blockIdx.y;
  const uint32_t head_id = destination * local_heads + local_head;

  const uint64_t input_offset = static_cast<uint64_t>(batch_id) * stride_batch +
                                static_cast<uint64_t>(token_id) * stride_token +
                                static_cast<uint64_t>(head_id) * stride_head + d_base;
  const uint64_t scale_offset =
      (static_cast<uint64_t>(batch_id) * num_heads + head_id) * head_dim + d_base;
  const uint64_t output_offset =
      static_cast<uint64_t>(destination) * chunk_bytes + 2 * main_bytes +
      ((static_cast<uint64_t>(token_id) * batch_size + batch_id) * local_heads + local_head) *
          head_dim + d_base;

  T x_val[pack_size];
  float scale_val[pack_size];
  float x_val_float[pack_size];
  uint32_t x_val_fp8[2];

  *reinterpret_cast<float4 *>(&x_val[0]) =
      *reinterpret_cast<const float4 *>(input + input_offset);
  *reinterpret_cast<float4 *>(&scale_val[0]) =
      *reinterpret_cast<const float4 *>(scale + scale_offset);
  *reinterpret_cast<float4 *>(&scale_val[4]) =
      *reinterpret_cast<const float4 *>(scale + scale_offset + 4);

#pragma unroll
  for (uint32_t i = 0; i < pack_size; ++i)
  {
    const float amax_val = detail::convert_to_float(detail::convert_from_float<T>(scale_val[i] * 2.25f));
    x_val_float[i] = detail::convert_to_float(x_val[i]) * __fdividef(2.25f, amax_val);
  }

  detail::floatx4_to_e4m3x4(x_val_fp8, x_val_float, x_val_float + 2);
  detail::floatx4_to_e4m3x4(x_val_fp8 + 1, x_val_float + 4, x_val_float + 6);
  *reinterpret_cast<uint2 *>(output + output_offset) =
      *reinterpret_cast<uint2 *>(&x_val_fp8[0]);
}

// Fused local statistics for the V1 NHD path, two-stage sequence-parallel
// form (2026-09-01 launch fix): the original single kernel put only
// (num_heads x batch) blocks on the device -- 56 blocks against 110 SMs at
// H=56 -- reaching ~1/3 of memory bandwidth.  Stage 1 additionally splits
// the sequence into fixed CHUNK_TOKENS chunks (grid z), each block reducing
// its chunk with the original 2-lane structure; stage 2 combines the chunk
// partials in FIXED ascending chunk order, so results are deterministic
// (bit-identical run to run).  NOTE: the fp32 k_sum association differs from
// the pre-2026-09-01 single-pass kernel, shifting k_mean by ULPs -- this is
// the one deliberate bit change of the launch fix (anchor re-freeze
// documented in v2g/align128/ksum_fix/).  v_amax is max-reduced and stays
// byte-identical to the old kernel.
template <typename T, uint32_t head_dim, uint32_t CHUNK_TOKENS>
__global__ void KSumVAmaxPartialKernel(
    const T *__restrict__ k,
    const T *__restrict__ v,
    float *__restrict__ k_partial,
    float *__restrict__ v_partial,
    const uint32_t num_tokens,
    const uint32_t num_heads,
    const uint32_t num_chunks,
    const int64_t k_stride_batch,
    const int64_t k_stride_token,
    const int64_t k_stride_head,
    const int64_t v_stride_batch,
    const int64_t v_stride_token,
    const int64_t v_stride_head)
{
  static_assert(head_dim == 128);
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value);
  constexpr uint32_t token_lanes = 2;
  constexpr uint32_t num_threads = token_lanes * head_dim;
  const uint32_t thread_id = threadIdx.x;
  const uint32_t d_id = thread_id % head_dim;
  const uint32_t token_lane = thread_id / head_dim;
  const uint32_t head_id = blockIdx.x;
  const uint32_t batch_id = blockIdx.y;
  const uint32_t chunk_id = blockIdx.z;
  const uint32_t token_begin = chunk_id * CHUNK_TOKENS;
  const uint32_t token_end = min(token_begin + CHUNK_TOKENS, num_tokens);

  float local_sum = 0.0f;
  float local_amax = 0.0f;
  for (uint32_t token_id = token_begin + token_lane; token_id < token_end;
       token_id += token_lanes)
  {
    const uint64_t k_offset = static_cast<uint64_t>(batch_id) * k_stride_batch +
                              static_cast<uint64_t>(token_id) * k_stride_token +
                              static_cast<uint64_t>(head_id) * k_stride_head + d_id;
    const uint64_t v_offset = static_cast<uint64_t>(batch_id) * v_stride_batch +
                              static_cast<uint64_t>(token_id) * v_stride_token +
                              static_cast<uint64_t>(head_id) * v_stride_head + d_id;
    local_sum += detail::convert_to_float(k[k_offset]);
    local_amax = fmaxf(local_amax, fabsf(detail::convert_to_float(v[v_offset])));
  }

  __shared__ float shared_sum[num_threads];
  __shared__ float shared_amax[num_threads];
  shared_sum[thread_id] = local_sum;
  shared_amax[thread_id] = local_amax;
  __syncthreads();

  if (token_lane == 0)
  {
    const uint64_t out =
        (((static_cast<uint64_t>(batch_id) * num_heads + head_id) * num_chunks) +
         chunk_id) *
            head_dim +
        d_id;
    k_partial[out] = shared_sum[d_id] + shared_sum[head_dim + d_id];
    v_partial[out] = fmaxf(shared_amax[d_id], shared_amax[head_dim + d_id]);
  }
}

template <uint32_t head_dim>
__global__ void KSumVAmaxCombineKernel(
    const float *__restrict__ k_partial,
    const float *__restrict__ v_partial,
    float *__restrict__ k_sum,
    float *__restrict__ v_amax,
    const uint32_t num_heads,
    const uint32_t num_chunks)
{
  // One thread per (batch, head, d); FIXED ascending-chunk-order reduction
  // keeps the fp32 sum deterministic.
  const uint32_t d_id = threadIdx.x;
  const uint32_t head_id = blockIdx.x;
  const uint32_t batch_id = blockIdx.y;
  const uint64_t base =
      (static_cast<uint64_t>(batch_id) * num_heads + head_id) * num_chunks;
  float s = 0.0f;
  float m = 0.0f;
  for (uint32_t c = 0; c < num_chunks; ++c)
  {
    s += k_partial[(base + c) * head_dim + d_id];
    m = fmaxf(m, v_partial[(base + c) * head_dim + d_id]);
  }
  const uint64_t out =
      (static_cast<uint64_t>(batch_id) * num_heads + head_id) * head_dim + d_id;
  k_sum[out] = s;
  v_amax[out] = m;
}

// Per-touched-group partial amax on this rank's shard, on the GLOBAL grid.
// Reproduces the pinned QuantInt8Kernel amax semantics exactly: fp32 convert,
// optional fp32 subtract of the dtype-T mean, per-thread 1e-7 floor, blockmax.
// Tokens of the group owned by other ranks are simply absent (max over the
// local intersection); out-of-range threads contribute nothing.
template <uint32_t head_dim, uint32_t GROUP, bool sub_mean, typename T>
__global__ void GroupedAmaxKernel(
    const T *__restrict__ input,
    const T *__restrict__ mean,
    float *__restrict__ amax_out,
    const uint32_t local_sequence,
    const uint32_t global_offset,
    const uint32_t num_heads,
    const uint32_t slots_alloc,
    const uint32_t group_first,
    const int64_t stride_batch,
    const int64_t stride_token,
    const int64_t stride_head)
{
  static_assert(head_dim == 128);
  static_assert(GROUP == 32 || GROUP == 64);
  constexpr uint32_t pack_size = 8;
  constexpr uint32_t threads_per_token = head_dim / pack_size;
  const uint32_t slot = blockIdx.x;
  const uint32_t head_id = blockIdx.y;
  const uint32_t batch_id = blockIdx.z;
  const uint32_t thread_id = threadIdx.x;
  const uint32_t token_in_group = thread_id / threads_per_token;
  const uint32_t d_base = thread_id % threads_per_token * pack_size;
  const uint32_t group_id = group_first + slot;
  const uint32_t global_token = group_id * GROUP + token_in_group;
  const bool valid = global_token >= global_offset &&
                     global_token < global_offset + local_sequence;

  T x_val[pack_size];
  float x_val_float[pack_size];
  float amax_val = 0.0000001f;
  if (valid)
  {
    const uint32_t local_token = global_token - global_offset;
    const uint64_t input_offset = static_cast<uint64_t>(batch_id) * stride_batch +
                                  static_cast<uint64_t>(local_token) * stride_token +
                                  static_cast<uint64_t>(head_id) * stride_head + d_base;
    *reinterpret_cast<float4 *>(&x_val[0]) =
        *reinterpret_cast<const float4 *>(input + input_offset);
    if constexpr (sub_mean)
    {
      T mean_val[pack_size];
      const uint64_t mean_offset =
          (static_cast<uint64_t>(batch_id) * num_heads + head_id) * head_dim + d_base;
      *reinterpret_cast<float4 *>(&mean_val[0]) =
          *reinterpret_cast<const float4 *>(mean + mean_offset);
#pragma unroll
      for (uint32_t j = 0; j < pack_size; ++j)
      {
        x_val_float[j] = detail::convert_to_float(x_val[j]) - detail::convert_to_float(mean_val[j]);
      }
    }
    else
    {
#pragma unroll
      for (uint32_t j = 0; j < pack_size; ++j)
      {
        x_val_float[j] = detail::convert_to_float(x_val[j]);
      }
    }
#pragma unroll
    for (uint32_t j = 0; j < pack_size; ++j)
    {
      amax_val = fmaxf(amax_val, fabsf(x_val_float[j]));
    }
  }

  const float block_amax_val = detail::blockReduceMax(amax_val);
  if (thread_id == 0)
  {
    amax_out[(static_cast<uint64_t>(batch_id) * num_heads + head_id) * slots_alloc + slot] =
        block_amax_val;
  }
}

// Quantize this rank's shard with externally supplied FINAL per-global-group
// amax values and write directly into the destination-major payload.  The
// element math is byte-identical to the pinned QuantInt8Kernel:
//   scale_out = amax / 127.0f;  int8 = float_to_int8_rn(x * (127.0f / amax)).
// Boundary-group amax values arrive already max-merged and bit-identical on
// every participating rank; interior groups carry this rank's local blockmax.
//
// Callers must zero the scale-and-padding region [3*main_bytes, chunk_bytes)
// of every destination chunk before the first V2-G pack launch; the kernels
// write only the touched slots.
template <uint32_t head_dim, uint32_t GROUP, bool sub_mean, typename T>
__global__ void QuantInt8GroupScalePackKernel(
    const T *__restrict__ input,
    const T *__restrict__ mean,
    const float *__restrict__ amax_final,
    uint8_t *__restrict__ output,
    const uint32_t local_sequence,
    const uint32_t global_offset,
    const uint32_t num_heads,
    const uint32_t local_heads,
    const uint32_t batch_size,
    const uint64_t chunk_bytes,
    const uint64_t section_offset,
    const uint64_t scale_offset,
    const uint32_t slots_alloc,
    const uint32_t group_first,
    const int64_t stride_batch,
    const int64_t stride_token,
    const int64_t stride_head)
{
  static_assert(head_dim == 128);
  static_assert(GROUP == 32 || GROUP == 64);
  constexpr uint32_t pack_size = 8;
  constexpr uint32_t threads_per_token = head_dim / pack_size;
  const uint32_t slot = blockIdx.x;
  const uint32_t head_id = blockIdx.y;
  const uint32_t batch_id = blockIdx.z;
  const uint32_t thread_id = threadIdx.x;
  const uint32_t token_in_group = thread_id / threads_per_token;
  const uint32_t d_base = thread_id % threads_per_token * pack_size;
  const uint32_t group_id = group_first + slot;
  const uint32_t global_token = group_id * GROUP + token_in_group;
  const bool valid = global_token >= global_offset &&
                     global_token < global_offset + local_sequence;
  const uint32_t destination = head_id / local_heads;
  const uint32_t local_head = head_id % local_heads;

  const float amax_val =
      amax_final[(static_cast<uint64_t>(batch_id) * num_heads + head_id) * slots_alloc + slot];

  if (thread_id == 0)
  {
    float *scale_output = reinterpret_cast<float *>(
        output + static_cast<uint64_t>(destination) * chunk_bytes + scale_offset);
    scale_output[(static_cast<uint64_t>(batch_id) * local_heads + local_head) * slots_alloc +
                 slot] = amax_val / 127.0f;
  }

  if (!valid)
  {
    return;
  }

  const uint32_t local_token = global_token - global_offset;
  const uint64_t input_offset = static_cast<uint64_t>(batch_id) * stride_batch +
                                static_cast<uint64_t>(local_token) * stride_token +
                                static_cast<uint64_t>(head_id) * stride_head + d_base;
  T x_val[pack_size];
  float x_val_float[pack_size];
  *reinterpret_cast<float4 *>(&x_val[0]) =
      *reinterpret_cast<const float4 *>(input + input_offset);
  if constexpr (sub_mean)
  {
    T mean_val[pack_size];
    const uint64_t mean_offset =
        (static_cast<uint64_t>(batch_id) * num_heads + head_id) * head_dim + d_base;
    *reinterpret_cast<float4 *>(&mean_val[0]) =
        *reinterpret_cast<const float4 *>(mean + mean_offset);
#pragma unroll
    for (uint32_t j = 0; j < pack_size; ++j)
    {
      x_val_float[j] = detail::convert_to_float(x_val[j]) - detail::convert_to_float(mean_val[j]);
    }
  }
  else
  {
#pragma unroll
    for (uint32_t j = 0; j < pack_size; ++j)
    {
      x_val_float[j] = detail::convert_to_float(x_val[j]);
    }
  }

  const float reciprocal_scale = 127.0f / amax_val;
  char4 quantized[2];
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j)
  {
    quantized[j] = make_char4(
        detail::float_to_int8_rn(x_val_float[j * 4 + 0] * reciprocal_scale),
        detail::float_to_int8_rn(x_val_float[j * 4 + 1] * reciprocal_scale),
        detail::float_to_int8_rn(x_val_float[j * 4 + 2] * reciprocal_scale),
        detail::float_to_int8_rn(x_val_float[j * 4 + 3] * reciprocal_scale));
  }
  const uint64_t packed_offset =
      static_cast<uint64_t>(destination) * chunk_bytes + section_offset +
      ((static_cast<uint64_t>(local_token) * batch_size + batch_id) * local_heads + local_head) *
          head_dim +
      d_base;
  *reinterpret_cast<float2 *>(output + packed_offset) =
      *reinterpret_cast<float2 *>(&quantized[0]);
}

// V2-G receiver, ALIGN-128 (stats protocol 3): rebuild contiguous logical
// Q/K [B,S,h,128], globally packed V [B,128,h,S] (Sage 16-token permutation
// applied on the GLOBAL token index), and global-grid Q/K scale tensors.
// The host asserts local_sequence % 128 == 0, so every 64-token CTA tile and
// every Q/K scale slot has exactly ONE source chunk: the rebuild is a
// straight per-chunk copy -- no cross-chunk gather, no token-validity tail
// (padded_sequence == logical_sequence), no unused scale slots
// (scale_alloc == groups_total).
template <uint32_t head_dim, uint32_t CTA_SIZE>
__global__ void UnpackForSageKernel(
    const uint8_t *__restrict__ input,
    uint8_t *__restrict__ q,
    uint8_t *__restrict__ k,
    uint8_t *__restrict__ v,
    uint8_t *__restrict__ q_scale,
    uint8_t *__restrict__ k_scale,
    const uint64_t main_bytes,
    const uint64_t chunk_bytes,
    const uint32_t batch_size,
    const uint32_t local_sequence,
    const uint32_t logical_sequence,
    const uint32_t padded_sequence,
    const uint32_t q_slots_per_source,
    const uint32_t k_slots_per_source,
    const uint32_t q_scale_alloc,
    const uint32_t k_scale_alloc)
{
  static_assert(head_dim == 128 && CTA_SIZE == 64);
  constexpr uint32_t vector_size = 16;
  constexpr uint32_t vectors_per_token = head_dim / vector_size;
  constexpr uint32_t sequence_vectors = CTA_SIZE / vector_size;
  const uint32_t block_token_base = blockIdx.x * CTA_SIZE;
  const uint32_t local_head = blockIdx.y;
  const uint32_t batch_id = blockIdx.z;
  const uint32_t thread_id = threadIdx.x;
  const uint32_t local_heads = gridDim.y;

  const uint32_t token_in_block = thread_id / vectors_per_token;
  const uint32_t global_token = block_token_base + token_in_block;
  const uint32_t d_base = thread_id % vectors_per_token * vector_size;

  // Single source per tile: L % 128 == 0 makes every 64-token tile whole
  // within one source chunk, so the division hoists out of the token path.
  const uint32_t source = block_token_base / local_sequence;
  const uint32_t local_token = global_token - source * local_sequence;
  const uint64_t source_chunk = static_cast<uint64_t>(source) * chunk_bytes;
  const uint64_t source_element =
      ((static_cast<uint64_t>(local_token) * batch_size + batch_id) * local_heads +
       local_head) *
          head_dim +
      d_base;
  const uint4 q_value =
      *reinterpret_cast<const uint4 *>(input + source_chunk + source_element);
  const uint4 k_value =
      *reinterpret_cast<const uint4 *>(input + source_chunk + main_bytes + source_element);
  const uint4 v_value = *reinterpret_cast<const uint4 *>(
      input + source_chunk + 2 * main_bytes + source_element);

  const uint64_t logical_element =
      ((static_cast<uint64_t>(batch_id) * logical_sequence + global_token) * local_heads +
       local_head) *
          head_dim +
      d_base;
  *reinterpret_cast<uint4 *>(q + logical_element) = q_value;
  *reinterpret_cast<uint4 *>(k + logical_element) = k_value;

  // Sage private 16-token permutation on the global token index.  The block
  // base is 64-aligned, hence 16-aligned, so token_in_block mod 16 equals
  // global_token mod 16.
  const uint32_t token_mod_16 = token_in_block & 15;
  const uint32_t packed_row =
      (token_in_block & ~15U) +
      (token_mod_16 / 8) * 2 +
      ((token_mod_16 / 2) % 4) * 4 +
      token_mod_16 % 2;
  __shared__ uint8_t shared_load[CTA_SIZE][head_dim];
  __shared__ uint8_t shared_store[head_dim][CTA_SIZE];
  *reinterpret_cast<uint4 *>(&shared_load[packed_row][d_base]) = v_value;
  __syncthreads();
#pragma unroll
  for (uint32_t i = 0; i < vector_size; ++i)
  {
    shared_store[d_base + i][packed_row] = shared_load[packed_row][d_base + i];
  }
  __syncthreads();
  const uint32_t output_d = thread_id / sequence_vectors;
  const uint32_t output_token_base = thread_id % sequence_vectors * vector_size;
  const uint64_t v_output_offset =
      ((static_cast<uint64_t>(batch_id) * head_dim + output_d) * local_heads + local_head) *
          padded_sequence +
      block_token_base + output_token_base;
  *reinterpret_cast<uint4 *>(v + v_output_offset) =
      *reinterpret_cast<uint4 *>(&shared_store[output_d][output_token_base]);

  if (blockIdx.x != 0)
  {
    return;
  }

  // Global-grid scale reconstruction: straight per-source sections.  Under
  // ALIGN-128 every source owns exactly local_sequence/32 Q (resp. /64 K)
  // whole groups, scale_alloc == groups_total (no zero tail), and slot g
  // belongs to source g / groups_per_source at offset g % groups_per_source.
  // Each (b, local_head) pair is handled by its own CTA (blockIdx.y/z), so
  // every output slot has exactly one writer.
  uint32_t *q_scale_output = reinterpret_cast<uint32_t *>(q_scale);
  uint32_t *k_scale_output = reinterpret_cast<uint32_t *>(k_scale);
  const uint64_t q_scale_section = 3 * main_bytes;
  const uint64_t q_scale_chunk_bytes =
      static_cast<uint64_t>(batch_size) * local_heads * q_slots_per_source * sizeof(float);
  const uint64_t scale_head = static_cast<uint64_t>(batch_id) * local_heads + local_head;
  const uint32_t q_groups_per_source = local_sequence / 32;
  const uint32_t k_groups_per_source = local_sequence / 64;

  for (uint32_t g = thread_id; g < q_scale_alloc; g += blockDim.x)
  {
    const uint32_t owner = g / q_groups_per_source;
    const uint32_t owner_slot = g - owner * q_groups_per_source;
    const uint32_t *q_scale_input = reinterpret_cast<const uint32_t *>(
        input + static_cast<uint64_t>(owner) * chunk_bytes + q_scale_section);
    q_scale_output[scale_head * q_scale_alloc + g] =
        q_scale_input[scale_head * q_slots_per_source + owner_slot];
  }
  for (uint32_t g = thread_id; g < k_scale_alloc; g += blockDim.x)
  {
    const uint32_t owner = g / k_groups_per_source;
    const uint32_t owner_slot = g - owner * k_groups_per_source;
    const uint32_t *k_scale_input = reinterpret_cast<const uint32_t *>(
        input + static_cast<uint64_t>(owner) * chunk_bytes + q_scale_section +
        q_scale_chunk_bytes);
    k_scale_output[scale_head * k_scale_alloc + g] =
        k_scale_input[scale_head * k_slots_per_source + owner_slot];
  }
}

// V2-G receiver, UNALIGNED variant (boundary-stats protocol 2, 64-aligned
// GLOBAL packing): local_sequence carries no 128-alignment guarantee, so a
// 64-token CTA tile may span two source chunks and the tail of the global
// grid may be partial.  Every token computes its own source (global_token /
// local_sequence) behind a validity guard; the Sage 16-token permutation is
// applied on the global token index with zero padding for invalid rows, and
// the scale rebuild is owner-only with deterministic zeroing of the unused
// tail slots (scale_alloc may exceed groups_total here).  Bit-exact port of
// the SageAttention protocol-2 kernel (p2-upstream-prep @8a1d1f6).
template <uint32_t head_dim, uint32_t CTA_SIZE>
__global__ void UnpackForSageUnalignedKernel(
    const uint8_t *__restrict__ input,
    uint8_t *__restrict__ q,
    uint8_t *__restrict__ k,
    uint8_t *__restrict__ v,
    uint8_t *__restrict__ q_scale,
    uint8_t *__restrict__ k_scale,
    const uint64_t main_bytes,
    const uint64_t chunk_bytes,
    const uint32_t batch_size,
    const uint32_t local_sequence,
    const uint32_t logical_sequence,
    const uint32_t padded_sequence,
    const uint32_t q_slots_per_source,
    const uint32_t k_slots_per_source,
    const uint32_t q_scale_alloc,
    const uint32_t k_scale_alloc)
{
  static_assert(head_dim == 128 && CTA_SIZE == 64);
  constexpr uint32_t vector_size = 16;
  constexpr uint32_t vectors_per_token = head_dim / vector_size;
  constexpr uint32_t sequence_vectors = CTA_SIZE / vector_size;
  const uint32_t block_token_base = blockIdx.x * CTA_SIZE;
  const uint32_t local_head = blockIdx.y;
  const uint32_t batch_id = blockIdx.z;
  const uint32_t thread_id = threadIdx.x;
  const uint32_t local_heads = gridDim.y;

  const uint32_t token_in_block = thread_id / vectors_per_token;
  const uint32_t global_token = block_token_base + token_in_block;
  const bool token_valid = global_token < logical_sequence;
  const uint32_t d_base = thread_id % vectors_per_token * vector_size;

  uint4 q_value = make_uint4(0, 0, 0, 0);
  uint4 k_value = make_uint4(0, 0, 0, 0);
  uint4 v_value = make_uint4(0, 0, 0, 0);
  if (token_valid)
  {
    const uint32_t source = global_token / local_sequence;
    const uint32_t local_token = global_token - source * local_sequence;
    const uint64_t source_chunk = static_cast<uint64_t>(source) * chunk_bytes;
    const uint64_t source_element =
        ((static_cast<uint64_t>(local_token) * batch_size + batch_id) * local_heads +
         local_head) *
            head_dim +
        d_base;
    q_value = *reinterpret_cast<const uint4 *>(input + source_chunk + source_element);
    k_value =
        *reinterpret_cast<const uint4 *>(input + source_chunk + main_bytes + source_element);
    v_value = *reinterpret_cast<const uint4 *>(
        input + source_chunk + 2 * main_bytes + source_element);

    const uint64_t logical_element =
        ((static_cast<uint64_t>(batch_id) * logical_sequence + global_token) * local_heads +
         local_head) *
            head_dim +
        d_base;
    *reinterpret_cast<uint4 *>(q + logical_element) = q_value;
    *reinterpret_cast<uint4 *>(k + logical_element) = k_value;
  }

  // Sage private 16-token permutation on the global token index.  The block
  // base is 64-aligned, hence 16-aligned, so token_in_block mod 16 equals
  // global_token mod 16.
  const uint32_t token_mod_16 = token_in_block & 15;
  const uint32_t packed_row =
      (token_in_block & ~15U) +
      (token_mod_16 / 8) * 2 +
      ((token_mod_16 / 2) % 4) * 4 +
      token_mod_16 % 2;
  __shared__ uint8_t shared_load[CTA_SIZE][head_dim];
  __shared__ uint8_t shared_store[head_dim][CTA_SIZE];
  *reinterpret_cast<uint4 *>(&shared_load[packed_row][d_base]) = v_value;
  __syncthreads();
#pragma unroll
  for (uint32_t i = 0; i < vector_size; ++i)
  {
    shared_store[d_base + i][packed_row] = shared_load[packed_row][d_base + i];
  }
  __syncthreads();
  const uint32_t output_d = thread_id / sequence_vectors;
  const uint32_t output_token_base = thread_id % sequence_vectors * vector_size;
  const uint64_t v_output_offset =
      ((static_cast<uint64_t>(batch_id) * head_dim + output_d) * local_heads + local_head) *
          padded_sequence +
      block_token_base + output_token_base;
  *reinterpret_cast<uint4 *>(v + v_output_offset) =
      *reinterpret_cast<uint4 *>(&shared_store[output_d][output_token_base]);

  if (blockIdx.x != 0)
  {
    return;
  }

  // Owner-only global-grid scale reconstruction plus deterministic zeroing of
  // the unused Q/K-scale tail slots.  Each (b, local_head) pair is handled by
  // its own CTA (blockIdx.y/z), so every output slot has exactly one writer.
  uint32_t *q_scale_output = reinterpret_cast<uint32_t *>(q_scale);
  uint32_t *k_scale_output = reinterpret_cast<uint32_t *>(k_scale);
  const uint64_t q_scale_section = 3 * main_bytes;
  const uint64_t q_scale_chunk_bytes =
      static_cast<uint64_t>(batch_size) * local_heads * q_slots_per_source * sizeof(float);
  const uint64_t scale_head = static_cast<uint64_t>(batch_id) * local_heads + local_head;
  const uint32_t q_groups_total = (logical_sequence + 31) / 32;
  const uint32_t k_groups_total = (logical_sequence + 63) / 64;

  for (uint32_t g = thread_id; g < q_scale_alloc; g += blockDim.x)
  {
    uint32_t value = 0u;
    if (g < q_groups_total)
    {
      const uint32_t owner = (g * 32) / local_sequence;
      const uint32_t owner_first = (owner * local_sequence) / 32;
      const uint32_t owner_slot = g - owner_first;
      const uint32_t *q_scale_input = reinterpret_cast<const uint32_t *>(
          input + static_cast<uint64_t>(owner) * chunk_bytes + q_scale_section);
      value = q_scale_input[scale_head * q_slots_per_source + owner_slot];
    }
    q_scale_output[scale_head * q_scale_alloc + g] = value;
  }
  for (uint32_t g = thread_id; g < k_scale_alloc; g += blockDim.x)
  {
    uint32_t value = 0u;
    if (g < k_groups_total)
    {
      const uint32_t owner = (g * 64) / local_sequence;
      const uint32_t owner_first = (owner * local_sequence) / 64;
      const uint32_t owner_slot = g - owner_first;
      const uint32_t *k_scale_input = reinterpret_cast<const uint32_t *>(
          input + static_cast<uint64_t>(owner) * chunk_bytes + q_scale_section +
          q_scale_chunk_bytes);
      value = k_scale_input[scale_head * k_slots_per_source + owner_slot];
    }
    k_scale_output[scale_head * k_scale_alloc + g] = value;
  }
}

}  // namespace ulysses_lowp
}  // namespace flashinfer

#endif  // FLASHINFER_COMM_ULYSSES_LOWP_CUH_
