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

#include <algorithm>
#include <cstdint>
#include <limits>

#include "tvm_ffi_utils.h"

namespace flashinfer::mhc {
namespace {

static constexpr int kHc4 = 4;
static constexpr int kThreads = 256;
static constexpr int kVecWidth = 8;
static constexpr int kMaxGridDimY = 65535;
static constexpr int64_t kMaxTokenVecHidden = static_cast<int64_t>(kThreads) * 8 * kVecWidth;

struct Bf16x8 {
  uint4 raw;
};

struct Hc4Mix {
  float4 post;
  float4 from0;
  float4 from1;
  float4 from2;
  float4 from3;
};

__device__ inline Bf16x8 load_bf16x8(const __nv_bfloat16* ptr) {
  Bf16x8 v;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.raw.x), "=r"(v.raw.y), "=r"(v.raw.z), "=r"(v.raw.w)
               : "l"(ptr));
  return v;
}

__device__ inline void store_bf16x8(__nv_bfloat16* ptr, const Bf16x8& v) {
  asm volatile("st.global.v4.u32 [%4], {%0, %1, %2, %3};\n"
               :
               : "r"(v.raw.x), "r"(v.raw.y), "r"(v.raw.z), "r"(v.raw.w), "l"(ptr));
}

__device__ inline __nv_bfloat162 bf16x8_get_pair(const Bf16x8& v, int i) {
  const uint32_t bits = i == 0 ? v.raw.x : (i == 1 ? v.raw.y : (i == 2 ? v.raw.z : v.raw.w));
  __nv_bfloat162 out;
  *reinterpret_cast<uint32_t*>(&out) = bits;
  return out;
}

__device__ inline void bf16x8_set_pair(Bf16x8& v, int i, __nv_bfloat162 pair) {
  const uint32_t bits = *reinterpret_cast<uint32_t*>(&pair);
  if (i == 0) {
    v.raw.x = bits;
  } else if (i == 1) {
    v.raw.y = bits;
  } else if (i == 2) {
    v.raw.z = bits;
  } else {
    v.raw.w = bits;
  }
}

__device__ __forceinline__ Hc4Mix load_hc4_mix(const float* __restrict__ post_layer_mix,
                                               const float* __restrict__ comb_res_mix,
                                               int64_t token) {
  const size_t post_base = static_cast<size_t>(token) * kHc4;
  const size_t mix_base = static_cast<size_t>(token) * kHc4 * kHc4;

  Hc4Mix mix;
  mix.post = make_float4(post_layer_mix[post_base + 0], post_layer_mix[post_base + 1],
                         post_layer_mix[post_base + 2], post_layer_mix[post_base + 3]);
  mix.from0 = make_float4(comb_res_mix[mix_base + 0], comb_res_mix[mix_base + 1],
                          comb_res_mix[mix_base + 2], comb_res_mix[mix_base + 3]);
  mix.from1 = make_float4(comb_res_mix[mix_base + 4], comb_res_mix[mix_base + 5],
                          comb_res_mix[mix_base + 6], comb_res_mix[mix_base + 7]);
  mix.from2 = make_float4(comb_res_mix[mix_base + 8], comb_res_mix[mix_base + 9],
                          comb_res_mix[mix_base + 10], comb_res_mix[mix_base + 11]);
  mix.from3 = make_float4(comb_res_mix[mix_base + 12], comb_res_mix[mix_base + 13],
                          comb_res_mix[mix_base + 14], comb_res_mix[mix_base + 15]);
  return mix;
}

__device__ __forceinline__ Hc4Mix load_hc4_mix_warp(const float* __restrict__ post_layer_mix,
                                                    const float* __restrict__ comb_res_mix,
                                                    int64_t token) {
  const int lane = threadIdx.x & 31;
  float coeff = 0.0f;
  const size_t post_base = static_cast<size_t>(token) * kHc4;
  const size_t mix_base = static_cast<size_t>(token) * kHc4 * kHc4;
  if (lane < kHc4) {
    coeff = post_layer_mix[post_base + lane];
  } else if (lane < kHc4 + kHc4 * kHc4) {
    coeff = comb_res_mix[mix_base + lane - kHc4];
  }

  constexpr unsigned mask = 0xffffffffu;
  Hc4Mix mix;
  mix.post = make_float4(__shfl_sync(mask, coeff, 0), __shfl_sync(mask, coeff, 1),
                         __shfl_sync(mask, coeff, 2), __shfl_sync(mask, coeff, 3));
  mix.from0 = make_float4(__shfl_sync(mask, coeff, 4), __shfl_sync(mask, coeff, 5),
                          __shfl_sync(mask, coeff, 6), __shfl_sync(mask, coeff, 7));
  mix.from1 = make_float4(__shfl_sync(mask, coeff, 8), __shfl_sync(mask, coeff, 9),
                          __shfl_sync(mask, coeff, 10), __shfl_sync(mask, coeff, 11));
  mix.from2 = make_float4(__shfl_sync(mask, coeff, 12), __shfl_sync(mask, coeff, 13),
                          __shfl_sync(mask, coeff, 14), __shfl_sync(mask, coeff, 15));
  mix.from3 = make_float4(__shfl_sync(mask, coeff, 16), __shfl_sync(mask, coeff, 17),
                          __shfl_sync(mask, coeff, 18), __shfl_sync(mask, coeff, 19));
  return mix;
}

__device__ __forceinline__ float2 combine_pair(float2 x, float2 r0, float2 r1, float2 r2, float2 r3,
                                               float post, float mix0, float mix1, float mix2,
                                               float mix3) {
  float2 acc = make_float2(x.x * post, x.y * post);
  acc.x = fmaf(r0.x, mix0, acc.x);
  acc.y = fmaf(r0.y, mix0, acc.y);
  acc.x = fmaf(r1.x, mix1, acc.x);
  acc.y = fmaf(r1.y, mix1, acc.y);
  acc.x = fmaf(r2.x, mix2, acc.x);
  acc.y = fmaf(r2.y, mix2, acc.y);
  acc.x = fmaf(r3.x, mix3, acc.x);
  acc.y = fmaf(r3.y, mix3, acc.y);
  return acc;
}

template <int64_t StaticH = 0>
__device__ __forceinline__ void compute_vec8(const __nv_bfloat16* __restrict__ x,
                                             const __nv_bfloat16* __restrict__ residual,
                                             __nv_bfloat16* __restrict__ out, const Hc4Mix& mix,
                                             int64_t token, int64_t runtime_H, int64_t h) {
  const int64_t H = StaticH == 0 ? runtime_H : StaticH;
  const size_t token_h = static_cast<size_t>(token) * static_cast<size_t>(H);
  const size_t token_hc_h = token_h * kHc4;
  const size_t offset = static_cast<size_t>(h);

  const Bf16x8 x_bf = load_bf16x8(x + token_h + offset);
  const Bf16x8 r0_bf = load_bf16x8(residual + token_hc_h + 0 * static_cast<size_t>(H) + offset);
  const Bf16x8 r1_bf = load_bf16x8(residual + token_hc_h + 1 * static_cast<size_t>(H) + offset);
  const Bf16x8 r2_bf = load_bf16x8(residual + token_hc_h + 2 * static_cast<size_t>(H) + offset);
  const Bf16x8 r3_bf = load_bf16x8(residual + token_hc_h + 3 * static_cast<size_t>(H) + offset);

  Bf16x8 y0;
  Bf16x8 y1;
  Bf16x8 y2;
  Bf16x8 y3;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 xv = __bfloat1622float2(bf16x8_get_pair(x_bf, i));
    const float2 r0 = __bfloat1622float2(bf16x8_get_pair(r0_bf, i));
    const float2 r1 = __bfloat1622float2(bf16x8_get_pair(r1_bf, i));
    const float2 r2 = __bfloat1622float2(bf16x8_get_pair(r2_bf, i));
    const float2 r3 = __bfloat1622float2(bf16x8_get_pair(r3_bf, i));

    bf16x8_set_pair(y0, i,
                    __float22bfloat162_rn(combine_pair(xv, r0, r1, r2, r3, mix.post.x, mix.from0.x,
                                                       mix.from1.x, mix.from2.x, mix.from3.x)));
    bf16x8_set_pair(y1, i,
                    __float22bfloat162_rn(combine_pair(xv, r0, r1, r2, r3, mix.post.y, mix.from0.y,
                                                       mix.from1.y, mix.from2.y, mix.from3.y)));
    bf16x8_set_pair(y2, i,
                    __float22bfloat162_rn(combine_pair(xv, r0, r1, r2, r3, mix.post.z, mix.from0.z,
                                                       mix.from1.z, mix.from2.z, mix.from3.z)));
    bf16x8_set_pair(y3, i,
                    __float22bfloat162_rn(combine_pair(xv, r0, r1, r2, r3, mix.post.w, mix.from0.w,
                                                       mix.from1.w, mix.from2.w, mix.from3.w)));
  }

  store_bf16x8(out + token_hc_h + 0 * static_cast<size_t>(H) + offset, y0);
  store_bf16x8(out + token_hc_h + 1 * static_cast<size_t>(H) + offset, y1);
  store_bf16x8(out + token_hc_h + 2 * static_cast<size_t>(H) + offset, y2);
  store_bf16x8(out + token_hc_h + 3 * static_cast<size_t>(H) + offset, y3);
}

template <int64_t StaticH = 0>
__device__ __forceinline__ void compute_vec8_head_serial(const __nv_bfloat16* __restrict__ x,
                                                         const __nv_bfloat16* __restrict__ residual,
                                                         __nv_bfloat16* __restrict__ out,
                                                         const Hc4Mix& mix, int64_t token,
                                                         int64_t runtime_H, int64_t h) {
  const int64_t H = StaticH == 0 ? runtime_H : StaticH;
  const size_t token_h = static_cast<size_t>(token) * static_cast<size_t>(H);
  const size_t token_hc_h = token_h * kHc4;
  const size_t offset = static_cast<size_t>(h);

  const Bf16x8 x_bf = load_bf16x8(x + token_h + offset);
  const Bf16x8 r0_bf = load_bf16x8(residual + token_hc_h + 0 * static_cast<size_t>(H) + offset);
  const Bf16x8 r1_bf = load_bf16x8(residual + token_hc_h + 1 * static_cast<size_t>(H) + offset);
  const Bf16x8 r2_bf = load_bf16x8(residual + token_hc_h + 2 * static_cast<size_t>(H) + offset);
  const Bf16x8 r3_bf = load_bf16x8(residual + token_hc_h + 3 * static_cast<size_t>(H) + offset);

  Bf16x8 y;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 xv = __bfloat1622float2(bf16x8_get_pair(x_bf, i));
    const float2 r0 = __bfloat1622float2(bf16x8_get_pair(r0_bf, i));
    const float2 r1 = __bfloat1622float2(bf16x8_get_pair(r1_bf, i));
    const float2 r2 = __bfloat1622float2(bf16x8_get_pair(r2_bf, i));
    const float2 r3 = __bfloat1622float2(bf16x8_get_pair(r3_bf, i));
    bf16x8_set_pair(y, i,
                    __float22bfloat162_rn(combine_pair(xv, r0, r1, r2, r3, mix.post.x, mix.from0.x,
                                                       mix.from1.x, mix.from2.x, mix.from3.x)));
  }
  store_bf16x8(out + token_hc_h + 0 * static_cast<size_t>(H) + offset, y);

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 xv = __bfloat1622float2(bf16x8_get_pair(x_bf, i));
    const float2 r0 = __bfloat1622float2(bf16x8_get_pair(r0_bf, i));
    const float2 r1 = __bfloat1622float2(bf16x8_get_pair(r1_bf, i));
    const float2 r2 = __bfloat1622float2(bf16x8_get_pair(r2_bf, i));
    const float2 r3 = __bfloat1622float2(bf16x8_get_pair(r3_bf, i));
    bf16x8_set_pair(y, i,
                    __float22bfloat162_rn(combine_pair(xv, r0, r1, r2, r3, mix.post.y, mix.from0.y,
                                                       mix.from1.y, mix.from2.y, mix.from3.y)));
  }
  store_bf16x8(out + token_hc_h + 1 * static_cast<size_t>(H) + offset, y);

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 xv = __bfloat1622float2(bf16x8_get_pair(x_bf, i));
    const float2 r0 = __bfloat1622float2(bf16x8_get_pair(r0_bf, i));
    const float2 r1 = __bfloat1622float2(bf16x8_get_pair(r1_bf, i));
    const float2 r2 = __bfloat1622float2(bf16x8_get_pair(r2_bf, i));
    const float2 r3 = __bfloat1622float2(bf16x8_get_pair(r3_bf, i));
    bf16x8_set_pair(y, i,
                    __float22bfloat162_rn(combine_pair(xv, r0, r1, r2, r3, mix.post.z, mix.from0.z,
                                                       mix.from1.z, mix.from2.z, mix.from3.z)));
  }
  store_bf16x8(out + token_hc_h + 2 * static_cast<size_t>(H) + offset, y);

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 xv = __bfloat1622float2(bf16x8_get_pair(x_bf, i));
    const float2 r0 = __bfloat1622float2(bf16x8_get_pair(r0_bf, i));
    const float2 r1 = __bfloat1622float2(bf16x8_get_pair(r1_bf, i));
    const float2 r2 = __bfloat1622float2(bf16x8_get_pair(r2_bf, i));
    const float2 r3 = __bfloat1622float2(bf16x8_get_pair(r3_bf, i));
    bf16x8_set_pair(y, i,
                    __float22bfloat162_rn(combine_pair(xv, r0, r1, r2, r3, mix.post.w, mix.from0.w,
                                                       mix.from1.w, mix.from2.w, mix.from3.w)));
  }
  store_bf16x8(out + token_hc_h + 3 * static_cast<size_t>(H) + offset, y);
}

__device__ __forceinline__ void compute_scalar(const __nv_bfloat16* __restrict__ x,
                                               const __nv_bfloat16* __restrict__ residual,
                                               __nv_bfloat16* __restrict__ out, const Hc4Mix& mix,
                                               int64_t token, int64_t H, int64_t h) {
  const size_t token_h = static_cast<size_t>(token) * static_cast<size_t>(H);
  const size_t token_hc_h = token_h * kHc4;
  const size_t offset = static_cast<size_t>(h);

  const float xv = __bfloat162float(x[token_h + offset]);
  const float r0 = __bfloat162float(residual[token_hc_h + 0 * static_cast<size_t>(H) + offset]);
  const float r1 = __bfloat162float(residual[token_hc_h + 1 * static_cast<size_t>(H) + offset]);
  const float r2 = __bfloat162float(residual[token_hc_h + 2 * static_cast<size_t>(H) + offset]);
  const float r3 = __bfloat162float(residual[token_hc_h + 3 * static_cast<size_t>(H) + offset]);

  float acc0 = xv * mix.post.x;
  float acc1 = xv * mix.post.y;
  float acc2 = xv * mix.post.z;
  float acc3 = xv * mix.post.w;

  acc0 = fmaf(r0, mix.from0.x, acc0);
  acc1 = fmaf(r0, mix.from0.y, acc1);
  acc2 = fmaf(r0, mix.from0.z, acc2);
  acc3 = fmaf(r0, mix.from0.w, acc3);

  acc0 = fmaf(r1, mix.from1.x, acc0);
  acc1 = fmaf(r1, mix.from1.y, acc1);
  acc2 = fmaf(r1, mix.from1.z, acc2);
  acc3 = fmaf(r1, mix.from1.w, acc3);

  acc0 = fmaf(r2, mix.from2.x, acc0);
  acc1 = fmaf(r2, mix.from2.y, acc1);
  acc2 = fmaf(r2, mix.from2.z, acc2);
  acc3 = fmaf(r2, mix.from2.w, acc3);

  acc0 = fmaf(r3, mix.from3.x, acc0);
  acc1 = fmaf(r3, mix.from3.y, acc1);
  acc2 = fmaf(r3, mix.from3.z, acc2);
  acc3 = fmaf(r3, mix.from3.w, acc3);

  out[token_hc_h + 0 * static_cast<size_t>(H) + offset] = __float2bfloat16(acc0);
  out[token_hc_h + 1 * static_cast<size_t>(H) + offset] = __float2bfloat16(acc1);
  out[token_hc_h + 2 * static_cast<size_t>(H) + offset] = __float2bfloat16(acc2);
  out[token_hc_h + 3 * static_cast<size_t>(H) + offset] = __float2bfloat16(acc3);
}

template <int TokensPerBlock, int VecPasses, bool WarpCoeffLoad, int64_t StaticH = 0>
__global__ __launch_bounds__(kThreads, 2) void mhc_post_bf16_hc4_token_vec_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ post_layer_mix, const float* __restrict__ comb_res_mix,
    __nv_bfloat16* __restrict__ out, int64_t total_tokens, int64_t H) {
  static_assert(kThreads % TokensPerBlock == 0, "TokensPerBlock must divide kThreads");
  constexpr int threads_per_token = kThreads / TokensPerBlock;

  __shared__ Hc4Mix shared_mix[TokensPerBlock];

  const int tid = threadIdx.x;
  const int token_slot = tid / threads_per_token;
  const int token_tid = tid - token_slot * threads_per_token;
  const int64_t token = static_cast<int64_t>(blockIdx.x) * TokensPerBlock + token_slot;
  const bool valid_token = token < total_tokens;

  Hc4Mix mix;
  if constexpr (WarpCoeffLoad) {
    if (!valid_token) {
      return;
    }
    mix = load_hc4_mix_warp(post_layer_mix, comb_res_mix, token);
  } else {
    if (token_tid == 0 && valid_token) {
      shared_mix[token_slot] = load_hc4_mix(post_layer_mix, comb_res_mix, token);
    }
    __syncthreads();
    if (!valid_token) {
      return;
    }
    mix = shared_mix[token_slot];
  }

  constexpr int64_t step = static_cast<int64_t>(threads_per_token) * kVecWidth;
  int64_t h = static_cast<int64_t>(token_tid) * kVecWidth;
#pragma unroll
  for (int iter = 0; iter < VecPasses; ++iter) {
    if (StaticH != 0 && (static_cast<int64_t>(iter + 1) * step) <= StaticH) {
      compute_vec8<StaticH>(x, residual, out, mix, token, H, h);
    } else {
      if (h < H) {
        compute_vec8<StaticH>(x, residual, out, mix, token, H, h);
      }
    }
    h += step;
  }
}

template <int Threads, int TileElems, int64_t StaticH = 0>
__global__ __launch_bounds__(Threads, 2) void mhc_post_bf16_hc4_split_vec_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ post_layer_mix, const float* __restrict__ comb_res_mix,
    __nv_bfloat16* __restrict__ out, int64_t total_tokens, int64_t H) {
  const int64_t h =
      static_cast<int64_t>(blockIdx.x) * TileElems + static_cast<int64_t>(threadIdx.x) * kVecWidth;

  for (int64_t token = static_cast<int64_t>(blockIdx.y); token < total_tokens;
       token += static_cast<int64_t>(gridDim.y)) {
    const Hc4Mix mix = load_hc4_mix_warp(post_layer_mix, comb_res_mix, token);
    if constexpr (StaticH == 0) {
      if (h < H) {
        compute_vec8(x, residual, out, mix, token, H, h);
      }
    } else {
      compute_vec8<StaticH>(x, residual, out, mix, token, H, h);
    }
  }
}

template <int Threads, int TileElems, int TokenBlocks, int64_t StaticH>
__global__ __launch_bounds__(Threads, 2) void mhc_post_bf16_hc4_persistent_split_vec_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ post_layer_mix, const float* __restrict__ comb_res_mix,
    __nv_bfloat16* __restrict__ out, int64_t total_tokens, int64_t H) {
  static_assert(StaticH != 0, "persistent split path is only for static-H arms");
  static_assert((StaticH % TileElems) == 0, "static H must be exactly tiled");
  static_assert((TileElems % kVecWidth) == 0, "tile elements must be vec8 aligned");
  static_assert(Threads * kVecWidth == TileElems, "one CTA must cover one exact H tile");

  const int64_t h =
      static_cast<int64_t>(blockIdx.x) * TileElems + static_cast<int64_t>(threadIdx.x) * kVecWidth;
  for (int64_t token = static_cast<int64_t>(blockIdx.y); token < total_tokens;
       token += static_cast<int64_t>(gridDim.y)) {
    const Hc4Mix mix = load_hc4_mix_warp(post_layer_mix, comb_res_mix, token);
    if constexpr (StaticH == 4096 || StaticH == 7168) {
      compute_vec8_head_serial<StaticH>(x, residual, out, mix, token, H, h);
    } else {
      compute_vec8<StaticH>(x, residual, out, mix, token, H, h);
    }
  }
}

__global__ __launch_bounds__(kThreads, 2) void mhc_post_bf16_hc4_scalar_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ post_layer_mix, const float* __restrict__ comb_res_mix,
    __nv_bfloat16* __restrict__ out, int64_t total_tokens, int64_t H) {
  const size_t total = static_cast<size_t>(total_tokens) * static_cast<size_t>(H);
  const size_t stride = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

  for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; linear < total;
       linear += stride) {
    const int64_t h = static_cast<int64_t>(linear % static_cast<size_t>(H));
    const int64_t token = static_cast<int64_t>(linear / static_cast<size_t>(H));
    const Hc4Mix mix = load_hc4_mix(post_layer_mix, comb_res_mix, token);
    compute_scalar(x, residual, out, mix, token, H, h);
  }
}

template <int TokensPerBlock, int VecPasses, bool WarpCoeffLoad, int64_t StaticH = 0>
void launch_token_vec_kernel(const __nv_bfloat16* x, const __nv_bfloat16* residual,
                             const float* post_layer_mix, const float* comb_res_mix,
                             __nv_bfloat16* out, int64_t total_tokens, int64_t H,
                             cudaStream_t stream) {
  const int64_t blocks64 = (total_tokens + TokensPerBlock - 1) / TokensPerBlock;
  TVM_FFI_ICHECK(blocks64 <= std::numeric_limits<unsigned int>::max())
      << "too many token blocks for mhc_post";
  mhc_post_bf16_hc4_token_vec_kernel<TokensPerBlock, VecPasses, WarpCoeffLoad, StaticH>
      <<<static_cast<unsigned int>(blocks64), kThreads, 0, stream>>>(
          x, residual, post_layer_mix, comb_res_mix, out, total_tokens, H);
}

template <int Threads, int TileElems, int64_t StaticH = 0>
void launch_split_vec_kernel(const __nv_bfloat16* x, const __nv_bfloat16* residual,
                             const float* post_layer_mix, const float* comb_res_mix,
                             __nv_bfloat16* out, int64_t total_tokens, int64_t H,
                             cudaStream_t stream) {
  const int64_t blocks_h64 = (H + TileElems - 1) / TileElems;
  TVM_FFI_ICHECK(blocks_h64 <= std::numeric_limits<unsigned int>::max())
      << "too many hidden blocks for mhc_post";
  const int64_t token_blocks64 = std::min<int64_t>(total_tokens, kMaxGridDimY);
  dim3 grid(static_cast<unsigned int>(blocks_h64), static_cast<unsigned int>(token_blocks64));
  mhc_post_bf16_hc4_split_vec_kernel<Threads, TileElems, StaticH><<<grid, Threads, 0, stream>>>(
      x, residual, post_layer_mix, comb_res_mix, out, total_tokens, H);
}

template <int Threads, int TileElems, int TokenBlocks, int64_t StaticH>
void launch_persistent_split_vec_kernel(const __nv_bfloat16* x, const __nv_bfloat16* residual,
                                        const float* post_layer_mix, const float* comb_res_mix,
                                        __nv_bfloat16* out, int64_t total_tokens, int64_t H,
                                        cudaStream_t stream) {
  const int64_t blocks_h64 = (StaticH + TileElems - 1) / TileElems;
  TVM_FFI_ICHECK(blocks_h64 <= std::numeric_limits<unsigned int>::max())
      << "too many hidden blocks for mhc_post";
  const int64_t token_blocks64 = std::min<int64_t>(total_tokens, TokenBlocks);
  TVM_FFI_ICHECK(token_blocks64 <= std::numeric_limits<unsigned int>::max())
      << "too many token blocks for mhc_post";
  dim3 grid(static_cast<unsigned int>(blocks_h64), static_cast<unsigned int>(token_blocks64));
  mhc_post_bf16_hc4_persistent_split_vec_kernel<Threads, TileElems, TokenBlocks, StaticH>
      <<<grid, Threads, 0, stream>>>(x, residual, post_layer_mix, comb_res_mix, out, total_tokens,
                                     H);
}

template <int64_t StaticH = 0>
void launch_vec_plan(const __nv_bfloat16* x, const __nv_bfloat16* residual,
                     const float* post_layer_mix, const float* comb_res_mix, __nv_bfloat16* out,
                     int64_t total_tokens, int64_t H, cudaStream_t stream) {
  if constexpr (StaticH != 0) {
    static_assert((StaticH % kVecWidth) == 0, "static H must be vec8 aligned");
    if constexpr (StaticH == 1280) {
      launch_split_vec_kernel<160, 1280, StaticH>(x, residual, post_layer_mix, comb_res_mix, out,
                                                  total_tokens, H, stream);
    } else if constexpr (StaticH <= 1536) {
      launch_token_vec_kernel<4, 3, true, StaticH>(x, residual, post_layer_mix, comb_res_mix, out,
                                                   total_tokens, H, stream);
    } else if constexpr (StaticH <= 2048) {
      launch_token_vec_kernel<4, 4, true, StaticH>(x, residual, post_layer_mix, comb_res_mix, out,
                                                   total_tokens, H, stream);
    } else if constexpr (StaticH <= 4096) {
      launch_split_vec_kernel<160, 1280, StaticH>(x, residual, post_layer_mix, comb_res_mix, out,
                                                  total_tokens, H, stream);
    } else if constexpr (StaticH <= 8192) {
      launch_token_vec_kernel<1, 4, false, StaticH>(x, residual, post_layer_mix, comb_res_mix, out,
                                                    total_tokens, H, stream);
    } else if constexpr (StaticH <= kMaxTokenVecHidden) {
      launch_token_vec_kernel<1, 8, false, StaticH>(x, residual, post_layer_mix, comb_res_mix, out,
                                                    total_tokens, H, stream);
    } else {
      launch_split_vec_kernel<160, 1280>(x, residual, post_layer_mix, comb_res_mix, out,
                                         total_tokens, H, stream);
    }
  } else {
    if (H <= 1536) {
      launch_token_vec_kernel<4, 3, true>(x, residual, post_layer_mix, comb_res_mix, out,
                                          total_tokens, H, stream);
    } else if (H <= 2048) {
      launch_token_vec_kernel<4, 4, true>(x, residual, post_layer_mix, comb_res_mix, out,
                                          total_tokens, H, stream);
    } else if (H <= 4096) {
      launch_split_vec_kernel<160, 1280>(x, residual, post_layer_mix, comb_res_mix, out,
                                         total_tokens, H, stream);
    } else if (H <= 8192) {
      launch_token_vec_kernel<1, 4, false>(x, residual, post_layer_mix, comb_res_mix, out,
                                           total_tokens, H, stream);
    } else if (H <= kMaxTokenVecHidden) {
      launch_token_vec_kernel<1, 8, false>(x, residual, post_layer_mix, comb_res_mix, out,
                                           total_tokens, H, stream);
    } else {
      launch_split_vec_kernel<160, 1280>(x, residual, post_layer_mix, comb_res_mix, out,
                                         total_tokens, H, stream);
    }
  }
}

void launch_bf16_hc4(TensorView x, TensorView residual, TensorView post_layer_mix,
                     TensorView comb_res_mix, TensorView out, int64_t total_tokens, int64_t H,
                     cudaStream_t stream) {
  const auto* x_ptr = static_cast<const __nv_bfloat16*>(x.data_ptr());
  const auto* residual_ptr = static_cast<const __nv_bfloat16*>(residual.data_ptr());
  auto* out_ptr = static_cast<__nv_bfloat16*>(out.data_ptr());
  const float* post_ptr = static_cast<const float*>(post_layer_mix.data_ptr());
  const float* comb_ptr = static_cast<const float*>(comb_res_mix.data_ptr());

  if ((H % kVecWidth) == 0) {
    switch (H) {
      case 4096:
        launch_persistent_split_vec_kernel<128, 1024, 2048, 4096>(
            x_ptr, residual_ptr, post_ptr, comb_ptr, out_ptr, total_tokens, H, stream);
        break;
      case 7168:
        if (total_tokens >= 4096) {
          launch_persistent_split_vec_kernel<128, 1024, 2048, 7168>(
              x_ptr, residual_ptr, post_ptr, comb_ptr, out_ptr, total_tokens, H, stream);
        } else {
          launch_vec_plan<7168>(x_ptr, residual_ptr, post_ptr, comb_ptr, out_ptr, total_tokens, H,
                                stream);
        }
        break;
      default:
        launch_vec_plan(x_ptr, residual_ptr, post_ptr, comb_ptr, out_ptr, total_tokens, H, stream);
        break;
    }
  } else {
    const int64_t total = total_tokens * H;
    int blocks = static_cast<int>(std::min<int64_t>((total + kThreads - 1) / kThreads, 65535));
    blocks = std::max(blocks, 1);
    mhc_post_bf16_hc4_scalar_kernel<<<blocks, kThreads, 0, stream>>>(
        x_ptr, residual_ptr, post_ptr, comb_ptr, out_ptr, total_tokens, H);
  }

  cudaError_t status = cudaPeekAtLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "mhc_post kernel launch failed with error code " << cudaGetErrorString(status);
}

}  // namespace

void mhc_post(TensorView out, TensorView x, TensorView residual, TensorView post_layer_mix,
              TensorView comb_res_mix) {
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(x, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(residual, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(post_layer_mix, dl_float32);
  CHECK_INPUT_AND_TYPE(comb_res_mix, dl_float32);
  CHECK_DEVICE(out, x);
  CHECK_DEVICE(out, residual);
  CHECK_DEVICE(out, post_layer_mix);
  CHECK_DEVICE(out, comb_res_mix);
  CHECK_DIM(2, x);
  CHECK_DIM(3, residual);
  CHECK_DIM(2, post_layer_mix);
  CHECK_DIM(3, comb_res_mix);
  CHECK_DIM(3, out);

  const int64_t total_tokens = residual.size(0);
  const int64_t HC = residual.size(1);
  const int64_t H = residual.size(2);

  TVM_FFI_ICHECK_EQ(HC, kHc4) << "residual.shape[1] / HC must be 4";
  TVM_FFI_ICHECK_GT(H, 0) << "hidden size must be positive";
  TVM_FFI_ICHECK_EQ(x.size(0), total_tokens) << "x.shape[0] must match residual.shape[0]";
  TVM_FFI_ICHECK_EQ(x.size(1), H) << "x.shape[1] must match hidden size";
  TVM_FFI_ICHECK_EQ(post_layer_mix.size(0), total_tokens)
      << "post_layer_mix.shape[0] must match residual.shape[0]";
  TVM_FFI_ICHECK_EQ(post_layer_mix.size(1), kHc4) << "post_layer_mix.shape[1] must be 4";
  TVM_FFI_ICHECK_EQ(comb_res_mix.size(0), total_tokens)
      << "comb_res_mix.shape[0] must match residual.shape[0]";
  TVM_FFI_ICHECK_EQ(comb_res_mix.size(1), kHc4) << "comb_res_mix.shape[1] must be 4";
  TVM_FFI_ICHECK_EQ(comb_res_mix.size(2), kHc4) << "comb_res_mix.shape[2] must be 4";
  TVM_FFI_ICHECK_EQ(out.size(0), total_tokens) << "out.shape[0] must match residual.shape[0]";
  TVM_FFI_ICHECK_EQ(out.size(1), kHc4) << "out.shape[1] must be 4";
  TVM_FFI_ICHECK_EQ(out.size(2), H) << "out.shape[2] must match hidden size";

  if (total_tokens == 0) {
    return;
  }

  ffi::CUDADeviceGuard device_guard(x.device().device_id);
  auto stream = get_stream(x.device());
  launch_bf16_hc4(x, residual, post_layer_mix, comb_res_mix, out, total_tokens, H, stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mhc_post, flashinfer::mhc::mhc_post);

}  // namespace flashinfer::mhc
