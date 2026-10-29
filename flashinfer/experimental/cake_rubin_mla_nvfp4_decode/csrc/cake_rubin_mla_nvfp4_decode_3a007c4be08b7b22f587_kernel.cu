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

typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int int32_t;
typedef short int int16_t;
struct __align__(64) CakeTensorMap64 {
  uint64_t opaque[16];
};
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) {
  uint64_t opaque[16];
} CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
  int result;
  asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;" : "=r"(result) : "r"(x));
  return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_W_OFF 0
#define SMEM_SMEM_W_STAGE_BYTES 2048
#define SMEM_SMEM_W_STRIDE 2048
#define SMEM_TOTAL 2048
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float approx_rcp(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float max_noftz(float a, float b) {
  float c;
  asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}

extern "C" {

__global__
__launch_bounds__(256, LAUNCH_MIN_BLOCKS) void kernel_cake_rubin_mla_nvfp4_decode_3a007c4be08b7b22f587(
    __nv_bfloat16* __restrict__ partial_o, float* __restrict__ partial_lse,
    __nv_bfloat16* __restrict__ O, float* __restrict__ lse, int rows, int num_split, int has_lse) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
#if __CUDA_ARCH__ == 1000
  asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }"
               : "=r"(smem)
               : "l"(smem_raw));
  smem = make_warp_uniform(smem);
#else
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  float* smem_w = reinterpret_cast<float*>(smem_raw + 0);
  const int smem_w_addr = smem + 0;

  // === Task calls (declaration order) ===
  asm volatile("griddepcontrol.wait;" ::: "memory");
  int part = 0;
  int row = blockIdx.x * 8 + warp;
  int merged = 0;
  if (row < rows) {
    merged = partial_lse[row * num_split] == CUDART_INF_F;
  }
  if (row < rows && merged == 0) {
    int lane_0 = lane;
    int stat_base = row * num_split;
    int w_base = warp * 64;
    int d0 = part * 512 + lane_0 * 16;
    int last = num_split - 1;
    int min_1 = ((0) < (last) ? (0) : (last));
    float vec_load[16];
    {
      const uint4* _vptr_0 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + min_1) * 512 + d0);
      uint4 _vld_0[2];
#pragma unroll
      for (int _blk = 0; _blk < 2; _blk++) {
        _vld_0[_blk] = _vptr_0[_blk];
        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec_load[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
          (&vec_load[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
        }
      }
    }
    int min_2 = ((1) < (last) ? (1) : (last));
    float vec_load_1[16];
    {
      const uint4* _vptr_1 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + min_2) * 512 + d0);
      uint4 _vld_1[2];
#pragma unroll
      for (int _blk = 0; _blk < 2; _blk++) {
        _vld_1[_blk] = _vptr_1[_blk];
        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec_load_1[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
          (&vec_load_1[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
        }
      }
    }
    int min_3 = ((2) < (last) ? (2) : (last));
    float vec_load_2[16];
    {
      const uint4* _vptr_2 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + min_3) * 512 + d0);
      uint4 _vld_2[2];
#pragma unroll
      for (int _blk = 0; _blk < 2; _blk++) {
        _vld_2[_blk] = _vptr_2[_blk];
        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec_load_2[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
          (&vec_load_2[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
        }
      }
    }
    int min_4 = ((3) < (last) ? (3) : (last));
    float vec_load_3[16];
    {
      const uint4* _vptr_3 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + min_4) * 512 + d0);
      uint4 _vld_3[2];
#pragma unroll
      for (int _blk = 0; _blk < 2; _blk++) {
        _vld_3[_blk] = _vptr_3[_blk];
        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec_load_3[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
          (&vec_load_3[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
        }
      }
    }
    int min_5 = ((4) < (last) ? (4) : (last));
    float vec_load_4[16];
    {
      const uint4* _vptr_4 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + min_5) * 512 + d0);
      uint4 _vld_4[2];
#pragma unroll
      for (int _blk = 0; _blk < 2; _blk++) {
        _vld_4[_blk] = _vptr_4[_blk];
        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec_load_4[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
          (&vec_load_4[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
        }
      }
    }
    int min_6 = ((5) < (last) ? (5) : (last));
    float vec_load_5[16];
    {
      const uint4* _vptr_5 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + min_6) * 512 + d0);
      uint4 _vld_5[2];
#pragma unroll
      for (int _blk = 0; _blk < 2; _blk++) {
        _vld_5[_blk] = _vptr_5[_blk];
        uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec_load_5[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
          (&vec_load_5[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
        }
      }
    }
    int min_7 = ((6) < (last) ? (6) : (last));
    float vec_load_6[16];
    {
      const uint4* _vptr_6 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + min_7) * 512 + d0);
      uint4 _vld_6[2];
#pragma unroll
      for (int _blk = 0; _blk < 2; _blk++) {
        _vld_6[_blk] = _vptr_6[_blk];
        uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec_load_6[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
          (&vec_load_6[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
        }
      }
    }
    int min_8 = ((7) < (last) ? (7) : (last));
    float vec_load_7[16];
    {
      const uint4* _vptr_7 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + min_8) * 512 + d0);
      uint4 _vld_7[2];
#pragma unroll
      for (int _blk = 0; _blk < 2; _blk++) {
        _vld_7[_blk] = _vptr_7[_blk];
        uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec_load_7[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
          (&vec_load_7[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
        }
      }
    }
    float l0 = -CUDART_INF_F;
    float l1 = -CUDART_INF_F;
    if (lane_0 < num_split) {
      l0 = partial_lse[stat_base + lane_0];
    }
    if (lane_0 + 32 < num_split) {
      l1 = partial_lse[stat_base + lane_0 + 32];
    }
    float m = max_noftz(l0, l1);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      m = max_noftz(m, __shfl_xor_sync(0xFFFFFFFF, m, offset));
    float w0 = 0.0f;
    float w1 = 0.0f;
    if (l0 > -CUDART_INF_F) {
      float _exp2_8 = approx_exp2(l0 - m);
      w0 = _exp2_8;
    }
    if (l1 > -CUDART_INF_F) {
      float _exp2_9 = approx_exp2(l1 - m);
      w1 = _exp2_9;
    }
    smem_w[w_base + lane_0] = w0;
    smem_w[w_base + 32 + lane_0] = w1;
    float sum_w = w0 + w1;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      sum_w += __shfl_xor_sync(0xFFFFFFFF, sum_w, offset);
    __syncwarp();
    float inv = 0.0f;
    if (sum_w > 0.0f) {
      float _rcp_10 = approx_rcp(sum_w);
      inv = _rcp_10;
    }
    float acc[16];
    for (int e = 0; e < 16; e++) {
      acc[e] = 0.0f;
    }
    float wts[8];
    wts[0] = smem_w[w_base];
    acc[0] = acc[0] + ((wts[0] > 0.0f) ? wts[0] * vec_load[0] : 0.0f);
    acc[1] = acc[1] + ((wts[0] > 0.0f) ? wts[0] * vec_load[1] : 0.0f);
    acc[2] = acc[2] + ((wts[0] > 0.0f) ? wts[0] * vec_load[2] : 0.0f);
    acc[3] = acc[3] + ((wts[0] > 0.0f) ? wts[0] * vec_load[3] : 0.0f);
    acc[4] = acc[4] + ((wts[0] > 0.0f) ? wts[0] * vec_load[4] : 0.0f);
    acc[5] = acc[5] + ((wts[0] > 0.0f) ? wts[0] * vec_load[5] : 0.0f);
    acc[6] = acc[6] + ((wts[0] > 0.0f) ? wts[0] * vec_load[6] : 0.0f);
    acc[7] = acc[7] + ((wts[0] > 0.0f) ? wts[0] * vec_load[7] : 0.0f);
    acc[8] = acc[8] + ((wts[0] > 0.0f) ? wts[0] * vec_load[8] : 0.0f);
    acc[9] = acc[9] + ((wts[0] > 0.0f) ? wts[0] * vec_load[9] : 0.0f);
    acc[10] = acc[10] + ((wts[0] > 0.0f) ? wts[0] * vec_load[10] : 0.0f);
    acc[11] = acc[11] + ((wts[0] > 0.0f) ? wts[0] * vec_load[11] : 0.0f);
    acc[12] = acc[12] + ((wts[0] > 0.0f) ? wts[0] * vec_load[12] : 0.0f);
    acc[13] = acc[13] + ((wts[0] > 0.0f) ? wts[0] * vec_load[13] : 0.0f);
    acc[14] = acc[14] + ((wts[0] > 0.0f) ? wts[0] * vec_load[14] : 0.0f);
    acc[15] = acc[15] + ((wts[0] > 0.0f) ? wts[0] * vec_load[15] : 0.0f);
    wts[1] = smem_w[w_base + 1];
    acc[0] = acc[0] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[0] : 0.0f);
    acc[1] = acc[1] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[1] : 0.0f);
    acc[2] = acc[2] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[2] : 0.0f);
    acc[3] = acc[3] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[3] : 0.0f);
    acc[4] = acc[4] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[4] : 0.0f);
    acc[5] = acc[5] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[5] : 0.0f);
    acc[6] = acc[6] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[6] : 0.0f);
    acc[7] = acc[7] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[7] : 0.0f);
    acc[8] = acc[8] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[8] : 0.0f);
    acc[9] = acc[9] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[9] : 0.0f);
    acc[10] = acc[10] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[10] : 0.0f);
    acc[11] = acc[11] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[11] : 0.0f);
    acc[12] = acc[12] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[12] : 0.0f);
    acc[13] = acc[13] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[13] : 0.0f);
    acc[14] = acc[14] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[14] : 0.0f);
    acc[15] = acc[15] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[15] : 0.0f);
    wts[2] = smem_w[w_base + 2];
    acc[0] = acc[0] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[0] : 0.0f);
    acc[1] = acc[1] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[1] : 0.0f);
    acc[2] = acc[2] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[2] : 0.0f);
    acc[3] = acc[3] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[3] : 0.0f);
    acc[4] = acc[4] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[4] : 0.0f);
    acc[5] = acc[5] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[5] : 0.0f);
    acc[6] = acc[6] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[6] : 0.0f);
    acc[7] = acc[7] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[7] : 0.0f);
    acc[8] = acc[8] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[8] : 0.0f);
    acc[9] = acc[9] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[9] : 0.0f);
    acc[10] = acc[10] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[10] : 0.0f);
    acc[11] = acc[11] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[11] : 0.0f);
    acc[12] = acc[12] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[12] : 0.0f);
    acc[13] = acc[13] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[13] : 0.0f);
    acc[14] = acc[14] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[14] : 0.0f);
    acc[15] = acc[15] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[15] : 0.0f);
    wts[3] = smem_w[w_base + 3];
    acc[0] = acc[0] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[0] : 0.0f);
    acc[1] = acc[1] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[1] : 0.0f);
    acc[2] = acc[2] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[2] : 0.0f);
    acc[3] = acc[3] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[3] : 0.0f);
    acc[4] = acc[4] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[4] : 0.0f);
    acc[5] = acc[5] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[5] : 0.0f);
    acc[6] = acc[6] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[6] : 0.0f);
    acc[7] = acc[7] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[7] : 0.0f);
    acc[8] = acc[8] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[8] : 0.0f);
    acc[9] = acc[9] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[9] : 0.0f);
    acc[10] = acc[10] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[10] : 0.0f);
    acc[11] = acc[11] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[11] : 0.0f);
    acc[12] = acc[12] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[12] : 0.0f);
    acc[13] = acc[13] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[13] : 0.0f);
    acc[14] = acc[14] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[14] : 0.0f);
    acc[15] = acc[15] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[15] : 0.0f);
    wts[4] = smem_w[w_base + 4];
    acc[0] = acc[0] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[0] : 0.0f);
    acc[1] = acc[1] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[1] : 0.0f);
    acc[2] = acc[2] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[2] : 0.0f);
    acc[3] = acc[3] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[3] : 0.0f);
    acc[4] = acc[4] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[4] : 0.0f);
    acc[5] = acc[5] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[5] : 0.0f);
    acc[6] = acc[6] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[6] : 0.0f);
    acc[7] = acc[7] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[7] : 0.0f);
    acc[8] = acc[8] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[8] : 0.0f);
    acc[9] = acc[9] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[9] : 0.0f);
    acc[10] = acc[10] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[10] : 0.0f);
    acc[11] = acc[11] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[11] : 0.0f);
    acc[12] = acc[12] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[12] : 0.0f);
    acc[13] = acc[13] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[13] : 0.0f);
    acc[14] = acc[14] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[14] : 0.0f);
    acc[15] = acc[15] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[15] : 0.0f);
    wts[5] = smem_w[w_base + 5];
    acc[0] = acc[0] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[0] : 0.0f);
    acc[1] = acc[1] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[1] : 0.0f);
    acc[2] = acc[2] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[2] : 0.0f);
    acc[3] = acc[3] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[3] : 0.0f);
    acc[4] = acc[4] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[4] : 0.0f);
    acc[5] = acc[5] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[5] : 0.0f);
    acc[6] = acc[6] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[6] : 0.0f);
    acc[7] = acc[7] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[7] : 0.0f);
    acc[8] = acc[8] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[8] : 0.0f);
    acc[9] = acc[9] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[9] : 0.0f);
    acc[10] = acc[10] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[10] : 0.0f);
    acc[11] = acc[11] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[11] : 0.0f);
    acc[12] = acc[12] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[12] : 0.0f);
    acc[13] = acc[13] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[13] : 0.0f);
    acc[14] = acc[14] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[14] : 0.0f);
    acc[15] = acc[15] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[15] : 0.0f);
    wts[6] = smem_w[w_base + 6];
    acc[0] = acc[0] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[0] : 0.0f);
    acc[1] = acc[1] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[1] : 0.0f);
    acc[2] = acc[2] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[2] : 0.0f);
    acc[3] = acc[3] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[3] : 0.0f);
    acc[4] = acc[4] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[4] : 0.0f);
    acc[5] = acc[5] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[5] : 0.0f);
    acc[6] = acc[6] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[6] : 0.0f);
    acc[7] = acc[7] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[7] : 0.0f);
    acc[8] = acc[8] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[8] : 0.0f);
    acc[9] = acc[9] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[9] : 0.0f);
    acc[10] = acc[10] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[10] : 0.0f);
    acc[11] = acc[11] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[11] : 0.0f);
    acc[12] = acc[12] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[12] : 0.0f);
    acc[13] = acc[13] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[13] : 0.0f);
    acc[14] = acc[14] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[14] : 0.0f);
    acc[15] = acc[15] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[15] : 0.0f);
    wts[7] = smem_w[w_base + 7];
    acc[0] = acc[0] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[0] : 0.0f);
    acc[1] = acc[1] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[1] : 0.0f);
    acc[2] = acc[2] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[2] : 0.0f);
    acc[3] = acc[3] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[3] : 0.0f);
    acc[4] = acc[4] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[4] : 0.0f);
    acc[5] = acc[5] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[5] : 0.0f);
    acc[6] = acc[6] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[6] : 0.0f);
    acc[7] = acc[7] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[7] : 0.0f);
    acc[8] = acc[8] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[8] : 0.0f);
    acc[9] = acc[9] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[9] : 0.0f);
    acc[10] = acc[10] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[10] : 0.0f);
    acc[11] = acc[11] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[11] : 0.0f);
    acc[12] = acc[12] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[12] : 0.0f);
    acc[13] = acc[13] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[13] : 0.0f);
    acc[14] = acc[14] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[14] : 0.0f);
    acc[15] = acc[15] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[15] : 0.0f);
    int n_batches = (num_split + 8 - 1) / 8;
#pragma unroll 1
    for (int bt = 1; bt < n_batches; bt++) {
      int k0 = bt * 8;
      int min_1_0 = ((k0) < (last) ? (k0) : (last));
      float vec_load_0[16];
      {
        const uint4* _vptr_11 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + min_1_0) * 512 + d0);
        uint4 _vld_11[2];
#pragma unroll
        for (int _blk = 0; _blk < 2; _blk++) {
          _vld_11[_blk] = _vptr_11[_blk];
          uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec_load_0[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) << 16);
            (&vec_load_0[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) & 0xffff0000u);
          }
        }
      }
      int min_2_0 = ((k0 + 1) < (last) ? (k0 + 1) : (last));
      float vec_load_1_0[16];
      {
        const uint4* _vptr_12 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + min_2_0) * 512 + d0);
        uint4 _vld_12[2];
#pragma unroll
        for (int _blk = 0; _blk < 2; _blk++) {
          _vld_12[_blk] = _vptr_12[_blk];
          uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec_load_1_0[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
            (&vec_load_1_0[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
          }
        }
      }
      int min_3_0 = ((k0 + 2) < (last) ? (k0 + 2) : (last));
      float vec_load_2_0[16];
      {
        const uint4* _vptr_13 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + min_3_0) * 512 + d0);
        uint4 _vld_13[2];
#pragma unroll
        for (int _blk = 0; _blk < 2; _blk++) {
          _vld_13[_blk] = _vptr_13[_blk];
          uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec_load_2_0[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) << 16);
            (&vec_load_2_0[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) & 0xffff0000u);
          }
        }
      }
      int min_4_0 = ((k0 + 3) < (last) ? (k0 + 3) : (last));
      float vec_load_3_0[16];
      {
        const uint4* _vptr_14 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + min_4_0) * 512 + d0);
        uint4 _vld_14[2];
#pragma unroll
        for (int _blk = 0; _blk < 2; _blk++) {
          _vld_14[_blk] = _vptr_14[_blk];
          uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec_load_3_0[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) << 16);
            (&vec_load_3_0[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) & 0xffff0000u);
          }
        }
      }
      int min_5_0 = ((k0 + 4) < (last) ? (k0 + 4) : (last));
      float vec_load_4_0[16];
      {
        const uint4* _vptr_15 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + min_5_0) * 512 + d0);
        uint4 _vld_15[2];
#pragma unroll
        for (int _blk = 0; _blk < 2; _blk++) {
          _vld_15[_blk] = _vptr_15[_blk];
          uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec_load_4_0[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) << 16);
            (&vec_load_4_0[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) & 0xffff0000u);
          }
        }
      }
      int min_6_0 = ((k0 + 5) < (last) ? (k0 + 5) : (last));
      float vec_load_5_0[16];
      {
        const uint4* _vptr_16 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + min_6_0) * 512 + d0);
        uint4 _vld_16[2];
#pragma unroll
        for (int _blk = 0; _blk < 2; _blk++) {
          _vld_16[_blk] = _vptr_16[_blk];
          uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec_load_5_0[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
            (&vec_load_5_0[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
          }
        }
      }
      int min_7_0 = ((k0 + 6) < (last) ? (k0 + 6) : (last));
      float vec_load_6_0[16];
      {
        const uint4* _vptr_17 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + min_7_0) * 512 + d0);
        uint4 _vld_17[2];
#pragma unroll
        for (int _blk = 0; _blk < 2; _blk++) {
          _vld_17[_blk] = _vptr_17[_blk];
          uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec_load_6_0[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) << 16);
            (&vec_load_6_0[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) & 0xffff0000u);
          }
        }
      }
      int min_8_0 = ((k0 + 7) < (last) ? (k0 + 7) : (last));
      float vec_load_7_0[16];
      {
        const uint4* _vptr_18 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + min_8_0) * 512 + d0);
        uint4 _vld_18[2];
#pragma unroll
        for (int _blk = 0; _blk < 2; _blk++) {
          _vld_18[_blk] = _vptr_18[_blk];
          uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec_load_7_0[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
            (&vec_load_7_0[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
          }
        }
      }
      wts[0] = smem_w[w_base + k0];
      acc[0] = acc[0] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[3] : 0.0f);
      acc[4] = acc[4] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[4] : 0.0f);
      acc[5] = acc[5] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[5] : 0.0f);
      acc[6] = acc[6] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[6] : 0.0f);
      acc[7] = acc[7] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[7] : 0.0f);
      acc[8] = acc[8] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[8] : 0.0f);
      acc[9] = acc[9] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[9] : 0.0f);
      acc[10] = acc[10] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[10] : 0.0f);
      acc[11] = acc[11] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[11] : 0.0f);
      acc[12] = acc[12] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[12] : 0.0f);
      acc[13] = acc[13] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[13] : 0.0f);
      acc[14] = acc[14] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[14] : 0.0f);
      acc[15] = acc[15] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[15] : 0.0f);
      wts[1] = smem_w[w_base + k0 + 1];
      acc[0] = acc[0] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[3] : 0.0f);
      acc[4] = acc[4] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[4] : 0.0f);
      acc[5] = acc[5] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[5] : 0.0f);
      acc[6] = acc[6] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[6] : 0.0f);
      acc[7] = acc[7] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[7] : 0.0f);
      acc[8] = acc[8] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[8] : 0.0f);
      acc[9] = acc[9] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[9] : 0.0f);
      acc[10] = acc[10] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[10] : 0.0f);
      acc[11] = acc[11] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[11] : 0.0f);
      acc[12] = acc[12] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[12] : 0.0f);
      acc[13] = acc[13] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[13] : 0.0f);
      acc[14] = acc[14] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[14] : 0.0f);
      acc[15] = acc[15] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[15] : 0.0f);
      wts[2] = smem_w[w_base + k0 + 2];
      acc[0] = acc[0] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[3] : 0.0f);
      acc[4] = acc[4] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[4] : 0.0f);
      acc[5] = acc[5] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[5] : 0.0f);
      acc[6] = acc[6] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[6] : 0.0f);
      acc[7] = acc[7] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[7] : 0.0f);
      acc[8] = acc[8] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[8] : 0.0f);
      acc[9] = acc[9] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[9] : 0.0f);
      acc[10] = acc[10] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[10] : 0.0f);
      acc[11] = acc[11] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[11] : 0.0f);
      acc[12] = acc[12] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[12] : 0.0f);
      acc[13] = acc[13] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[13] : 0.0f);
      acc[14] = acc[14] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[14] : 0.0f);
      acc[15] = acc[15] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[15] : 0.0f);
      wts[3] = smem_w[w_base + k0 + 3];
      acc[0] = acc[0] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[3] : 0.0f);
      acc[4] = acc[4] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[4] : 0.0f);
      acc[5] = acc[5] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[5] : 0.0f);
      acc[6] = acc[6] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[6] : 0.0f);
      acc[7] = acc[7] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[7] : 0.0f);
      acc[8] = acc[8] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[8] : 0.0f);
      acc[9] = acc[9] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[9] : 0.0f);
      acc[10] = acc[10] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[10] : 0.0f);
      acc[11] = acc[11] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[11] : 0.0f);
      acc[12] = acc[12] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[12] : 0.0f);
      acc[13] = acc[13] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[13] : 0.0f);
      acc[14] = acc[14] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[14] : 0.0f);
      acc[15] = acc[15] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[15] : 0.0f);
      wts[4] = smem_w[w_base + k0 + 4];
      acc[0] = acc[0] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[3] : 0.0f);
      acc[4] = acc[4] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[4] : 0.0f);
      acc[5] = acc[5] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[5] : 0.0f);
      acc[6] = acc[6] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[6] : 0.0f);
      acc[7] = acc[7] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[7] : 0.0f);
      acc[8] = acc[8] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[8] : 0.0f);
      acc[9] = acc[9] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[9] : 0.0f);
      acc[10] = acc[10] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[10] : 0.0f);
      acc[11] = acc[11] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[11] : 0.0f);
      acc[12] = acc[12] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[12] : 0.0f);
      acc[13] = acc[13] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[13] : 0.0f);
      acc[14] = acc[14] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[14] : 0.0f);
      acc[15] = acc[15] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[15] : 0.0f);
      wts[5] = smem_w[w_base + k0 + 5];
      acc[0] = acc[0] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[3] : 0.0f);
      acc[4] = acc[4] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[4] : 0.0f);
      acc[5] = acc[5] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[5] : 0.0f);
      acc[6] = acc[6] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[6] : 0.0f);
      acc[7] = acc[7] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[7] : 0.0f);
      acc[8] = acc[8] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[8] : 0.0f);
      acc[9] = acc[9] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[9] : 0.0f);
      acc[10] = acc[10] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[10] : 0.0f);
      acc[11] = acc[11] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[11] : 0.0f);
      acc[12] = acc[12] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[12] : 0.0f);
      acc[13] = acc[13] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[13] : 0.0f);
      acc[14] = acc[14] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[14] : 0.0f);
      acc[15] = acc[15] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[15] : 0.0f);
      wts[6] = smem_w[w_base + k0 + 6];
      acc[0] = acc[0] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[3] : 0.0f);
      acc[4] = acc[4] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[4] : 0.0f);
      acc[5] = acc[5] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[5] : 0.0f);
      acc[6] = acc[6] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[6] : 0.0f);
      acc[7] = acc[7] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[7] : 0.0f);
      acc[8] = acc[8] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[8] : 0.0f);
      acc[9] = acc[9] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[9] : 0.0f);
      acc[10] = acc[10] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[10] : 0.0f);
      acc[11] = acc[11] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[11] : 0.0f);
      acc[12] = acc[12] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[12] : 0.0f);
      acc[13] = acc[13] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[13] : 0.0f);
      acc[14] = acc[14] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[14] : 0.0f);
      acc[15] = acc[15] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[15] : 0.0f);
      wts[7] = smem_w[w_base + k0 + 7];
      acc[0] = acc[0] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[3] : 0.0f);
      acc[4] = acc[4] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[4] : 0.0f);
      acc[5] = acc[5] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[5] : 0.0f);
      acc[6] = acc[6] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[6] : 0.0f);
      acc[7] = acc[7] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[7] : 0.0f);
      acc[8] = acc[8] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[8] : 0.0f);
      acc[9] = acc[9] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[9] : 0.0f);
      acc[10] = acc[10] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[10] : 0.0f);
      acc[11] = acc[11] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[11] : 0.0f);
      acc[12] = acc[12] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[12] : 0.0f);
      acc[13] = acc[13] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[13] : 0.0f);
      acc[14] = acc[14] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[14] : 0.0f);
      acc[15] = acc[15] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[15] : 0.0f);
    }
    for (int e_1 = 0; e_1 < 16; e_1++) {
      acc[e_1] = acc[e_1] * inv;
    }
    unsigned int _packed_0[8];
#pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
      __nv_bfloat162 _bf2 =
          __float22bfloat162_rn(make_float2(acc[_lp * 2 + 0], acc[_lp * 2 + 1 + 0]));
      _packed_0[_lp] = *(uint32_t*)&_bf2;
    }
#pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
      __nv_bfloat162 _bf2 =
          __float22bfloat162_rn(make_float2(acc[_lp * 2 + 2], acc[_lp * 2 + 1 + 2]));
      _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
    }
#pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
      __nv_bfloat162 _bf2 =
          __float22bfloat162_rn(make_float2(acc[_lp * 2 + 4], acc[_lp * 2 + 1 + 4]));
      _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
    }
#pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
      __nv_bfloat162 _bf2 =
          __float22bfloat162_rn(make_float2(acc[_lp * 2 + 6], acc[_lp * 2 + 1 + 6]));
      _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
    }
#pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
      __nv_bfloat162 _bf2 =
          __float22bfloat162_rn(make_float2(acc[_lp * 2 + 8], acc[_lp * 2 + 1 + 8]));
      _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
    }
#pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
      __nv_bfloat162 _bf2 =
          __float22bfloat162_rn(make_float2(acc[_lp * 2 + 10], acc[_lp * 2 + 1 + 10]));
      _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
    }
#pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
      __nv_bfloat162 _bf2 =
          __float22bfloat162_rn(make_float2(acc[_lp * 2 + 12], acc[_lp * 2 + 1 + 12]));
      _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
    }
#pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
      __nv_bfloat162 _bf2 =
          __float22bfloat162_rn(make_float2(acc[_lp * 2 + 14], acc[_lp * 2 + 1 + 14]));
      _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int4*>(O + (row * 512 + d0))[0] = reinterpret_cast<int4*>(_packed_0)[0];
    reinterpret_cast<int4*>(O + (row * 512 + d0 + 8))[0] =
        reinterpret_cast<int4*>(_packed_0 + (4))[0];
    if (has_lse != 0) {
      if (part == 0) {
        if (lane_0 == 0) {
          float lse_v = -CUDART_INF_F;
          if (sum_w > 0.0f) {
            float _log2_19;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_19) : "f"(sum_w));
            lse_v = (m + _log2_19) * 0.6931472f;
          }
          *(reinterpret_cast<float*>(lse + row) + (0)) = lse_v;
        }
      }
    }
  }
}

}  // extern "C"
