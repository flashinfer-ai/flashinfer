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
// Fused per-head QK RMSNorm + Rotary Position Embedding.
//
// Adapted from NVIDIA/TensorRT-LLM:
//   cpp/tensorrt_llm/kernels/fusedQKNormRopeKernel.cu
// One warp processes one (token, qk_head). Elements are held in registers,
// a warp-wide reduction computes the RMSNorm scale, and RoPE (with optional
// YaRN frequency correction) is applied before writing the result back in
// place. Relative to the upstream kernel, this version:
//   - takes separate q / k pointers with runtime strides (matches flashinfer's
//     3D ragged layout) instead of a packed 2D qkv buffer,
//   - is templated over DType so both __nv_bfloat16 and __half are supported.
#ifndef FLASHINFER_FUSED_QK_NORM_ROPE_CUH_
#define FLASHINFER_FUSED_QK_NORM_ROPE_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <type_traits>

#include "utils.cuh"

namespace flashinfer {

namespace fused_qk_norm_rope_impl {

// ------------------ Helpers ----------------------------------------

// Traits to convert between a 16-bit DType and its 2-wide packed form / float.
template <typename DType>
struct DTypeTraits;

template <>
struct DTypeTraits<__nv_bfloat16> {
  using pack2_t = __nv_bfloat162;
  __device__ __forceinline__ static float2 unpack(pack2_t v) { return __bfloat1622float2(v); }
  __device__ __forceinline__ static pack2_t pack(float2 v) { return __float22bfloat162_rn(v); }
  __device__ __forceinline__ static float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
};

template <>
struct DTypeTraits<__half> {
  using pack2_t = __half2;
  __device__ __forceinline__ static float2 unpack(pack2_t v) { return __half22float2(v); }
  __device__ __forceinline__ static pack2_t pack(float2 v) { return __float22half2_rn(v); }
  __device__ __forceinline__ static float to_float(__half x) { return __half2float(x); }
};

// Number of 32-bit words covered by a warp's register tile. Maps to uint,
// uint2 or uint4 to get coalesced 32/64/128-bit global memory transactions.
template <int kNumWords>
struct PackedVec;
template <>
struct PackedVec<1> {
  using type = uint32_t;
};
template <>
struct PackedVec<2> {
  using type = uint2;
};
template <>
struct PackedVec<4> {
  using type = uint4;
};

__device__ __forceinline__ float WarpReduceSum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, offset, 32);
  }
  return val;
}

// YaRN inverse-frequency ramp (from TRT-LLM).  When factor == 1 the YaRN
// branch is skipped and we fall back to the standard RoPE frequency.
__device__ __forceinline__ float ComputeFreqYarn(float base, int rotary_dim, int half_dim,
                                                 float factor, float low, float high) {
  float freq = powf(base, -2.0f * static_cast<float>(half_dim) / static_cast<float>(rotary_dim));
  if (factor != 1.0f) {
    float inv_freq_extrapolation = freq;
    float inv_freq_interpolation = freq / factor;
    float high_adj = high;
    // Prevent division-by-zero when low == high (TRT-LLM does this in-place;
    // we keep a local copy since CUDA function args are const).
    if (fabsf(low - high_adj) <= 1e-6f) {
      high_adj += 0.001f;
    }
    float linear_func = (static_cast<float>(half_dim) - low) / (high_adj - low);
    float ramp_func = fminf(fmaxf(linear_func, 0.0f), 1.0f);
    float inv_freq_extrapolation_factor = 1.0f - ramp_func;
    freq = inv_freq_interpolation * (1.0f - inv_freq_extrapolation_factor) +
           inv_freq_extrapolation * inv_freq_extrapolation_factor;
  }
  return freq;
}

}  // namespace fused_qk_norm_rope_impl

// ------------------ Kernel ------------------------------------------

/*!
 * \brief Fused per-head QK RMSNorm + Rotary Position Embedding kernel.
 *
 * \tparam DType Element dtype (__nv_bfloat16 or __half).
 * \tparam head_dim Dimension per head. Must be a multiple of 64.
 * \tparam interleave If true, RoPE operates on even/odd pairs (GPT-J style).
 *                    If false, RoPE operates on the first/second half (Neox / Llama style).
 *
 * Each warp processes one (token, qk_head) pair. The warp first loads the
 * entire head into registers (1 element per lane iteration, numElemsPerThread
 * iterations), performs an optional warp-level RMSNorm with the per-dim
 * weight, then applies RoPE with on-the-fly cos/sin derived from
 * ``position_ids`` and ``rope_theta``, and writes the result back in place.
 */
template <typename DType, int head_dim, bool interleave>
__global__ void FusedQKNormRopeKernel(DType* __restrict__ q,               // [nnz, H_q, D]
                                      DType* __restrict__ k,               // [nnz, H_k, D]
                                      DType const* __restrict__ q_weight,  // [D]
                                      DType const* __restrict__ k_weight,  // [D]
                                      int const* __restrict__ pos_ids,     // [nnz]
                                      int const num_q_heads, int const num_kv_heads,
                                      int const rotary_dim, float const eps, float const rope_theta,
                                      float const yarn_factor, float const yarn_low,
                                      float const yarn_high, float const yarn_attention_factor,
                                      int const num_tokens, bool const is_qk_norm,
                                      size_t const q_stride_n, size_t const q_stride_h,
                                      size_t const k_stride_n, size_t const k_stride_h) {
  using namespace fused_qk_norm_rope_impl;
  using Traits = DTypeTraits<DType>;
  using pack2_t = typename Traits::pack2_t;

  int const warps_per_block = blockDim.x / 32;
  int const warp_id = threadIdx.x / 32;
  int const lane_id = threadIdx.x % 32;

  int const global_warp_idx = blockIdx.x * warps_per_block + warp_id;
  int const total_qk_heads = num_q_heads + num_kv_heads;
  int const token_idx = global_warp_idx / total_qk_heads;
  int const local_head_idx = global_warp_idx % total_qk_heads;

  if (token_idx >= num_tokens) return;

  bool const is_q = local_head_idx < num_q_heads;
  int const head_idx = is_q ? local_head_idx : local_head_idx - num_q_heads;

  // head_dim must be divisible by 64 so each lane owns an even number of elements,
  // which is required for the packed (pair-wise) load and for the interleaved /
  // half-split RoPE variants to line up with the warp's 32 lanes.
  static_assert(head_dim % 64 == 0, "head_dim must be a multiple of 64");
  constexpr int numElemsPerThread = head_dim / 32;
  constexpr int elemSizeBytes = numElemsPerThread * sizeof(DType);
  static_assert(elemSizeBytes % 4 == 0,
                "numElemsPerThread * sizeof(DType) must be a multiple of 4");
  constexpr int vecSize = elemSizeBytes / 4;
  using vec_T = typename PackedVec<vecSize>::type;

  float elements[numElemsPerThread];

  // Select the tensor (q vs k) and compute this warp's offset into it.
  DType* head_ptr;
  size_t offset_warp;
  if (is_q) {
    head_ptr = q;
    offset_warp =
        static_cast<size_t>(token_idx) * q_stride_n + static_cast<size_t>(head_idx) * q_stride_h;
  } else {
    head_ptr = k;
    offset_warp =
        static_cast<size_t>(token_idx) * k_stride_n + static_cast<size_t>(head_idx) * k_stride_h;
  }
  size_t offset_thread = offset_warp + static_cast<size_t>(lane_id) * numElemsPerThread;

  // --- Load into registers and accumulate sum of squares ---
  float sum_of_squares = 0.0f;
  {
    vec_T vec = *reinterpret_cast<vec_T const*>(head_ptr + offset_thread);
#pragma unroll
    for (int i = 0; i < vecSize; i++) {
      pack2_t packed = *(reinterpret_cast<pack2_t*>(&vec) + i);
      float2 vals = Traits::unpack(packed);
      sum_of_squares += vals.x * vals.x;
      sum_of_squares += vals.y * vals.y;
      elements[2 * i] = vals.x;
      elements[2 * i + 1] = vals.y;
    }
  }

  // --- RMSNorm (optional) ---
  if (is_qk_norm) {
    sum_of_squares = WarpReduceSum(sum_of_squares);
    float rms_rcp = rsqrtf(sum_of_squares / static_cast<float>(head_dim) + eps);
#pragma unroll
    for (int i = 0; i < numElemsPerThread; i++) {
      int dim = lane_id * numElemsPerThread + i;
      float weight = is_q ? Traits::to_float(q_weight[dim]) : Traits::to_float(k_weight[dim]);
      elements[i] *= rms_rcp * weight;
    }
  }

  // --- RoPE ---
  float elements2[numElemsPerThread];
  float cos_vals[numElemsPerThread];
  float sin_vals[numElemsPerThread];
  float pos_id = static_cast<float>(pos_ids[token_idx]);

  if constexpr (interleave) {
    // GPT-J style: rotate even/odd pairs (elements[2i], elements[2i+1]).
#pragma unroll
    for (int i = 0; i < numElemsPerThread; i++) {
      elements2[i] = (i % 2 == 0) ? -elements[i + 1] : elements[i - 1];
      int dim_idx = lane_id * numElemsPerThread + i;
      int half_dim = dim_idx / 2;
      float freq =
          ComputeFreqYarn(rope_theta, rotary_dim, half_dim, yarn_factor, yarn_low, yarn_high);
      float theta = pos_id * freq;
      __sincosf(theta, &sin_vals[i], &cos_vals[i]);
    }
  } else {
    // Neox / Llama style: rotate (x[i], x[i + rotary_dim/2]).
    // Lanes in [0, pair_offset) exchange elements with lanes in
    // [pair_offset, 2*pair_offset) via __shfl_xor_sync.
    // pair_offset must be a power of 2 for this exchange to be correct; we
    // enforce that in the C++ launcher / Python API.
    //
    // __shfl_xor_sync with a full mask (0xffffffff) is itself a warp-wide
    // barrier, and this kernel only touches registers (no shared/global
    // memory) between the prior writes and the shuffle, so no explicit
    // __syncwarp() is needed -- see `QKRMSNormKernel` in flashinfer/norm.cuh
    // for the same convention.
    int const pair_offset = (rotary_dim / 2) / numElemsPerThread;
#pragma unroll
    for (int i = 0; i < numElemsPerThread; i++) {
      elements2[i] = __shfl_xor_sync(0xffffffff, elements[i], pair_offset);
      if (lane_id < pair_offset) {
        elements2[i] = -elements2[i];
      }
      int dim_idx = lane_id * numElemsPerThread + i;
      dim_idx = (dim_idx * 2) % rotary_dim;
      int half_dim = dim_idx / 2;
      float freq =
          ComputeFreqYarn(rope_theta, rotary_dim, half_dim, yarn_factor, yarn_low, yarn_high);
      float theta = pos_id * freq;
      __sincosf(theta, &sin_vals[i], &cos_vals[i]);
    }
  }

  bool const is_full_rope = (rotary_dim == head_dim);
  if (is_full_rope) {
#pragma unroll
    for (int i = 0; i < numElemsPerThread; i++) {
      elements[i] =
          (elements[i] * cos_vals[i] + elements2[i] * sin_vals[i]) * yarn_attention_factor;
    }
  } else {
#pragma unroll
    for (int i = 0; i < numElemsPerThread; i++) {
      int dim_idx = lane_id * numElemsPerThread + i;
      if (dim_idx < rotary_dim) {
        elements[i] =
            (elements[i] * cos_vals[i] + elements2[i] * sin_vals[i]) * yarn_attention_factor;
      }
    }
  }

  // --- Store ---
  {
    vec_T vec;
#pragma unroll
    for (int i = 0; i < vecSize; i++) {
      pack2_t packed = Traits::pack(make_float2(elements[2 * i], elements[2 * i + 1]));
      *(reinterpret_cast<pack2_t*>(&vec) + i) = packed;
    }
    *reinterpret_cast<vec_T*>(head_ptr + offset_thread) = vec;
  }
}

// ------------------ Host launcher -----------------------------------

/*!
 * \brief Host launcher for :c:func:`FusedQKNormRopeKernel`.
 *
 * Dispatches over ``head_dim`` in {64, 128, 256} and ``interleave``. Returns
 * ``cudaErrorInvalidValue`` for unsupported ``head_dim`` so the C++ binding
 * can surface a clean TVM-FFI error.
 */
template <typename DType>
cudaError_t FusedQKNormRopeLauncher(DType* q, DType* k, DType const* q_weight,
                                    DType const* k_weight, int const* pos_ids, int num_tokens,
                                    int num_q_heads, int num_kv_heads, int head_dim, int rotary_dim,
                                    float eps, float rope_theta, bool interleave, float yarn_factor,
                                    float yarn_low, float yarn_high, float yarn_attention_factor,
                                    bool is_qk_norm, size_t q_stride_n, size_t q_stride_h,
                                    size_t k_stride_n, size_t k_stride_h, cudaStream_t stream) {
  if (num_tokens == 0 || (num_q_heads + num_kv_heads) == 0) {
    return cudaSuccess;
  }

  constexpr int block_size = 256;
  int const warps_per_block = block_size / 32;
  int const total_qk_heads = num_q_heads + num_kv_heads;
  // Use int64_t for the intermediate so num_tokens * total_qk_heads cannot
  // silently overflow int32 (e.g., ~54M tokens with H_q+H_k = 40 already
  // approaches INT_MAX). CUDA's x-grid limit is still INT_MAX, so the
  // resulting grid_size must fit in int.
  int64_t const total_warps =
      static_cast<int64_t>(num_tokens) * static_cast<int64_t>(total_qk_heads);
  int64_t const grid_size_i64 = (total_warps + warps_per_block - 1) / warps_per_block;
  if (grid_size_i64 > static_cast<int64_t>(INT_MAX)) {
    return cudaErrorInvalidConfiguration;
  }
  int const grid_size = static_cast<int>(grid_size_i64);
  dim3 grid_dim(grid_size);
  dim3 block_dim(block_size);

#define LAUNCH_IMPL(HEAD_DIM_VAL)                                                                  \
  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {                                                    \
    FusedQKNormRopeKernel<DType, HEAD_DIM_VAL, INTERLEAVE><<<grid_dim, block_dim, 0, stream>>>(    \
        q, k, q_weight, k_weight, pos_ids, num_q_heads, num_kv_heads, rotary_dim, eps, rope_theta, \
        yarn_factor, yarn_low, yarn_high, yarn_attention_factor, num_tokens, is_qk_norm,           \
        q_stride_n, q_stride_h, k_stride_n, k_stride_h);                                           \
  })

  switch (head_dim) {
    case 64:
      LAUNCH_IMPL(64);
      break;
    case 128:
      LAUNCH_IMPL(128);
      break;
    case 256:
      LAUNCH_IMPL(256);
      break;
    default:
      return cudaErrorInvalidValue;
  }
#undef LAUNCH_IMPL

  return cudaGetLastError();
}

}  // namespace flashinfer

#endif  // FLASHINFER_FUSED_QK_NORM_ROPE_CUH_
