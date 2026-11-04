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
__launch_bounds__(256, LAUNCH_MIN_BLOCKS) void kernel_cake_rubin_mla_nvfp4_decode_ed7a88b839ebde5e8d2d(
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
  int part = warp % 4;
  int row = blockIdx.x * 2 + warp / 4;
  int merged = 0;
  if (row < rows) {
    merged = partial_lse[row * num_split] == CUDART_INF_F;
  }
  if (row < rows && merged == 0) {
    int lane_0 = lane;
    int stat_base = row * num_split;
    int w_base = warp * 64;
    int d0 = part * 128 + lane_0 * 4;
    int last = num_split - 1;
    int _mn_0 = ((0) < (last) ? (0) : (last));
    float po_0[4];
    {
      uint2 _vld_0;
      _vld_0 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_0) * 512 + d0);
      uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
        (&po_0[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
      }
    }
    int _mn_1 = ((1) < (last) ? (1) : (last));
    float po_1[4];
    {
      uint2 _vld_1;
      _vld_1 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_1) * 512 + d0);
      uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
        (&po_1[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
      }
    }
    int _mn_2 = ((2) < (last) ? (2) : (last));
    float po_2[4];
    {
      uint2 _vld_2;
      _vld_2 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_2) * 512 + d0);
      uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
        (&po_2[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
      }
    }
    int _mn_3 = ((3) < (last) ? (3) : (last));
    float po_3[4];
    {
      uint2 _vld_3;
      _vld_3 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_3) * 512 + d0);
      uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_3[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
        (&po_3[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
      }
    }
    int _mn_4 = ((4) < (last) ? (4) : (last));
    float po_4[4];
    {
      uint2 _vld_4;
      _vld_4 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_4) * 512 + d0);
      uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_4[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
        (&po_4[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
      }
    }
    int _mn_5 = ((5) < (last) ? (5) : (last));
    float po_5[4];
    {
      uint2 _vld_5;
      _vld_5 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_5) * 512 + d0);
      uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_5[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
        (&po_5[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
      }
    }
    int _mn_6 = ((6) < (last) ? (6) : (last));
    float po_6[4];
    {
      uint2 _vld_6;
      _vld_6 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_6) * 512 + d0);
      uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_6[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
        (&po_6[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
      }
    }
    int _mn_7 = ((7) < (last) ? (7) : (last));
    float po_7[4];
    {
      uint2 _vld_7;
      _vld_7 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_7) * 512 + d0);
      uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_7[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
        (&po_7[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
      }
    }
    int _mn_8 = ((8) < (last) ? (8) : (last));
    float po_8[4];
    {
      uint2 _vld_8;
      _vld_8 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_8) * 512 + d0);
      uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_8[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
        (&po_8[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
      }
    }
    int _mn_9 = ((9) < (last) ? (9) : (last));
    float po_9[4];
    {
      uint2 _vld_9;
      _vld_9 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_9) * 512 + d0);
      uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_9[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
        (&po_9[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
      }
    }
    int _mn_10 = ((10) < (last) ? (10) : (last));
    float po_10[4];
    {
      uint2 _vld_10;
      _vld_10 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_10) * 512 + d0);
      uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_10[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
        (&po_10[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
      }
    }
    int _mn_11 = ((11) < (last) ? (11) : (last));
    float po_11[4];
    {
      uint2 _vld_11;
      _vld_11 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_11) * 512 + d0);
      uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_11[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) << 16);
        (&po_11[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) & 0xffff0000u);
      }
    }
    int _mn_12 = ((12) < (last) ? (12) : (last));
    float po_12[4];
    {
      uint2 _vld_12;
      _vld_12 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_12) * 512 + d0);
      uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_12[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
        (&po_12[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
      }
    }
    int _mn_13 = ((13) < (last) ? (13) : (last));
    float po_13[4];
    {
      uint2 _vld_13;
      _vld_13 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_13) * 512 + d0);
      uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_13[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) << 16);
        (&po_13[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) & 0xffff0000u);
      }
    }
    int _mn_14 = ((14) < (last) ? (14) : (last));
    float po_14[4];
    {
      uint2 _vld_14;
      _vld_14 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_14) * 512 + d0);
      uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_14[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) << 16);
        (&po_14[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) & 0xffff0000u);
      }
    }
    int _mn_15 = ((15) < (last) ? (15) : (last));
    float po_15[4];
    {
      uint2 _vld_15;
      _vld_15 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_15) * 512 + d0);
      uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_15[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) << 16);
        (&po_15[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) & 0xffff0000u);
      }
    }
    int _mn_16 = ((16) < (last) ? (16) : (last));
    float po_16[4];
    {
      uint2 _vld_16;
      _vld_16 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_16) * 512 + d0);
      uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_16[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
        (&po_16[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
      }
    }
    int _mn_17 = ((17) < (last) ? (17) : (last));
    float po_17[4];
    {
      uint2 _vld_17;
      _vld_17 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_17) * 512 + d0);
      uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_17[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) << 16);
        (&po_17[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) & 0xffff0000u);
      }
    }
    int _mn_18 = ((18) < (last) ? (18) : (last));
    float po_18[4];
    {
      uint2 _vld_18;
      _vld_18 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_18) * 512 + d0);
      uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_18[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
        (&po_18[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
      }
    }
    int _mn_19 = ((19) < (last) ? (19) : (last));
    float po_19[4];
    {
      uint2 _vld_19;
      _vld_19 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_19) * 512 + d0);
      uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_19[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) << 16);
        (&po_19[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) & 0xffff0000u);
      }
    }
    int _mn_20 = ((20) < (last) ? (20) : (last));
    float po_20[4];
    {
      uint2 _vld_20;
      _vld_20 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_20) * 512 + d0);
      uint32_t* _vpairs_20 = reinterpret_cast<uint32_t*>(&_vld_20);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_20[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) << 16);
        (&po_20[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) & 0xffff0000u);
      }
    }
    int _mn_21 = ((21) < (last) ? (21) : (last));
    float po_21[4];
    {
      uint2 _vld_21;
      _vld_21 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_21) * 512 + d0);
      uint32_t* _vpairs_21 = reinterpret_cast<uint32_t*>(&_vld_21);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_21[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) << 16);
        (&po_21[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) & 0xffff0000u);
      }
    }
    int _mn_22 = ((22) < (last) ? (22) : (last));
    float po_22[4];
    {
      uint2 _vld_22;
      _vld_22 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_22) * 512 + d0);
      uint32_t* _vpairs_22 = reinterpret_cast<uint32_t*>(&_vld_22);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_22[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_22[_pair]) << 16);
        (&po_22[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_22[_pair]) & 0xffff0000u);
      }
    }
    int _mn_23 = ((23) < (last) ? (23) : (last));
    float po_23[4];
    {
      uint2 _vld_23;
      _vld_23 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + _mn_23) * 512 + d0);
      uint32_t* _vpairs_23 = reinterpret_cast<uint32_t*>(&_vld_23);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&po_23[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_23[_pair]) << 16);
        (&po_23[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_23[_pair]) & 0xffff0000u);
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
      float _exp2_24 = approx_exp2(l0 - m);
      w0 = _exp2_24;
    }
    if (l1 > -CUDART_INF_F) {
      float _exp2_25 = approx_exp2(l1 - m);
      w1 = _exp2_25;
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
      float _rcp_26 = approx_rcp(sum_w);
      inv = _rcp_26;
    }
    float acc[4];
    for (int e = 0; e < 4; e++) {
      acc[e] = 0.0f;
    }
    float wts[24];
    wts[0] = smem_w[w_base];
    acc[0] = acc[0] + ((wts[0] > 0.0f) ? wts[0] * po_0[0] : 0.0f);
    acc[1] = acc[1] + ((wts[0] > 0.0f) ? wts[0] * po_0[1] : 0.0f);
    acc[2] = acc[2] + ((wts[0] > 0.0f) ? wts[0] * po_0[2] : 0.0f);
    acc[3] = acc[3] + ((wts[0] > 0.0f) ? wts[0] * po_0[3] : 0.0f);
    wts[1] = smem_w[w_base + 1];
    acc[0] = acc[0] + ((wts[1] > 0.0f) ? wts[1] * po_1[0] : 0.0f);
    acc[1] = acc[1] + ((wts[1] > 0.0f) ? wts[1] * po_1[1] : 0.0f);
    acc[2] = acc[2] + ((wts[1] > 0.0f) ? wts[1] * po_1[2] : 0.0f);
    acc[3] = acc[3] + ((wts[1] > 0.0f) ? wts[1] * po_1[3] : 0.0f);
    wts[2] = smem_w[w_base + 2];
    acc[0] = acc[0] + ((wts[2] > 0.0f) ? wts[2] * po_2[0] : 0.0f);
    acc[1] = acc[1] + ((wts[2] > 0.0f) ? wts[2] * po_2[1] : 0.0f);
    acc[2] = acc[2] + ((wts[2] > 0.0f) ? wts[2] * po_2[2] : 0.0f);
    acc[3] = acc[3] + ((wts[2] > 0.0f) ? wts[2] * po_2[3] : 0.0f);
    wts[3] = smem_w[w_base + 3];
    acc[0] = acc[0] + ((wts[3] > 0.0f) ? wts[3] * po_3[0] : 0.0f);
    acc[1] = acc[1] + ((wts[3] > 0.0f) ? wts[3] * po_3[1] : 0.0f);
    acc[2] = acc[2] + ((wts[3] > 0.0f) ? wts[3] * po_3[2] : 0.0f);
    acc[3] = acc[3] + ((wts[3] > 0.0f) ? wts[3] * po_3[3] : 0.0f);
    wts[4] = smem_w[w_base + 4];
    acc[0] = acc[0] + ((wts[4] > 0.0f) ? wts[4] * po_4[0] : 0.0f);
    acc[1] = acc[1] + ((wts[4] > 0.0f) ? wts[4] * po_4[1] : 0.0f);
    acc[2] = acc[2] + ((wts[4] > 0.0f) ? wts[4] * po_4[2] : 0.0f);
    acc[3] = acc[3] + ((wts[4] > 0.0f) ? wts[4] * po_4[3] : 0.0f);
    wts[5] = smem_w[w_base + 5];
    acc[0] = acc[0] + ((wts[5] > 0.0f) ? wts[5] * po_5[0] : 0.0f);
    acc[1] = acc[1] + ((wts[5] > 0.0f) ? wts[5] * po_5[1] : 0.0f);
    acc[2] = acc[2] + ((wts[5] > 0.0f) ? wts[5] * po_5[2] : 0.0f);
    acc[3] = acc[3] + ((wts[5] > 0.0f) ? wts[5] * po_5[3] : 0.0f);
    wts[6] = smem_w[w_base + 6];
    acc[0] = acc[0] + ((wts[6] > 0.0f) ? wts[6] * po_6[0] : 0.0f);
    acc[1] = acc[1] + ((wts[6] > 0.0f) ? wts[6] * po_6[1] : 0.0f);
    acc[2] = acc[2] + ((wts[6] > 0.0f) ? wts[6] * po_6[2] : 0.0f);
    acc[3] = acc[3] + ((wts[6] > 0.0f) ? wts[6] * po_6[3] : 0.0f);
    wts[7] = smem_w[w_base + 7];
    acc[0] = acc[0] + ((wts[7] > 0.0f) ? wts[7] * po_7[0] : 0.0f);
    acc[1] = acc[1] + ((wts[7] > 0.0f) ? wts[7] * po_7[1] : 0.0f);
    acc[2] = acc[2] + ((wts[7] > 0.0f) ? wts[7] * po_7[2] : 0.0f);
    acc[3] = acc[3] + ((wts[7] > 0.0f) ? wts[7] * po_7[3] : 0.0f);
    wts[8] = smem_w[w_base + 8];
    acc[0] = acc[0] + ((wts[8] > 0.0f) ? wts[8] * po_8[0] : 0.0f);
    acc[1] = acc[1] + ((wts[8] > 0.0f) ? wts[8] * po_8[1] : 0.0f);
    acc[2] = acc[2] + ((wts[8] > 0.0f) ? wts[8] * po_8[2] : 0.0f);
    acc[3] = acc[3] + ((wts[8] > 0.0f) ? wts[8] * po_8[3] : 0.0f);
    wts[9] = smem_w[w_base + 9];
    acc[0] = acc[0] + ((wts[9] > 0.0f) ? wts[9] * po_9[0] : 0.0f);
    acc[1] = acc[1] + ((wts[9] > 0.0f) ? wts[9] * po_9[1] : 0.0f);
    acc[2] = acc[2] + ((wts[9] > 0.0f) ? wts[9] * po_9[2] : 0.0f);
    acc[3] = acc[3] + ((wts[9] > 0.0f) ? wts[9] * po_9[3] : 0.0f);
    wts[10] = smem_w[w_base + 10];
    acc[0] = acc[0] + ((wts[10] > 0.0f) ? wts[10] * po_10[0] : 0.0f);
    acc[1] = acc[1] + ((wts[10] > 0.0f) ? wts[10] * po_10[1] : 0.0f);
    acc[2] = acc[2] + ((wts[10] > 0.0f) ? wts[10] * po_10[2] : 0.0f);
    acc[3] = acc[3] + ((wts[10] > 0.0f) ? wts[10] * po_10[3] : 0.0f);
    wts[11] = smem_w[w_base + 11];
    acc[0] = acc[0] + ((wts[11] > 0.0f) ? wts[11] * po_11[0] : 0.0f);
    acc[1] = acc[1] + ((wts[11] > 0.0f) ? wts[11] * po_11[1] : 0.0f);
    acc[2] = acc[2] + ((wts[11] > 0.0f) ? wts[11] * po_11[2] : 0.0f);
    acc[3] = acc[3] + ((wts[11] > 0.0f) ? wts[11] * po_11[3] : 0.0f);
    wts[12] = smem_w[w_base + 12];
    acc[0] = acc[0] + ((wts[12] > 0.0f) ? wts[12] * po_12[0] : 0.0f);
    acc[1] = acc[1] + ((wts[12] > 0.0f) ? wts[12] * po_12[1] : 0.0f);
    acc[2] = acc[2] + ((wts[12] > 0.0f) ? wts[12] * po_12[2] : 0.0f);
    acc[3] = acc[3] + ((wts[12] > 0.0f) ? wts[12] * po_12[3] : 0.0f);
    wts[13] = smem_w[w_base + 13];
    acc[0] = acc[0] + ((wts[13] > 0.0f) ? wts[13] * po_13[0] : 0.0f);
    acc[1] = acc[1] + ((wts[13] > 0.0f) ? wts[13] * po_13[1] : 0.0f);
    acc[2] = acc[2] + ((wts[13] > 0.0f) ? wts[13] * po_13[2] : 0.0f);
    acc[3] = acc[3] + ((wts[13] > 0.0f) ? wts[13] * po_13[3] : 0.0f);
    wts[14] = smem_w[w_base + 14];
    acc[0] = acc[0] + ((wts[14] > 0.0f) ? wts[14] * po_14[0] : 0.0f);
    acc[1] = acc[1] + ((wts[14] > 0.0f) ? wts[14] * po_14[1] : 0.0f);
    acc[2] = acc[2] + ((wts[14] > 0.0f) ? wts[14] * po_14[2] : 0.0f);
    acc[3] = acc[3] + ((wts[14] > 0.0f) ? wts[14] * po_14[3] : 0.0f);
    wts[15] = smem_w[w_base + 15];
    acc[0] = acc[0] + ((wts[15] > 0.0f) ? wts[15] * po_15[0] : 0.0f);
    acc[1] = acc[1] + ((wts[15] > 0.0f) ? wts[15] * po_15[1] : 0.0f);
    acc[2] = acc[2] + ((wts[15] > 0.0f) ? wts[15] * po_15[2] : 0.0f);
    acc[3] = acc[3] + ((wts[15] > 0.0f) ? wts[15] * po_15[3] : 0.0f);
    wts[16] = smem_w[w_base + 16];
    acc[0] = acc[0] + ((wts[16] > 0.0f) ? wts[16] * po_16[0] : 0.0f);
    acc[1] = acc[1] + ((wts[16] > 0.0f) ? wts[16] * po_16[1] : 0.0f);
    acc[2] = acc[2] + ((wts[16] > 0.0f) ? wts[16] * po_16[2] : 0.0f);
    acc[3] = acc[3] + ((wts[16] > 0.0f) ? wts[16] * po_16[3] : 0.0f);
    wts[17] = smem_w[w_base + 17];
    acc[0] = acc[0] + ((wts[17] > 0.0f) ? wts[17] * po_17[0] : 0.0f);
    acc[1] = acc[1] + ((wts[17] > 0.0f) ? wts[17] * po_17[1] : 0.0f);
    acc[2] = acc[2] + ((wts[17] > 0.0f) ? wts[17] * po_17[2] : 0.0f);
    acc[3] = acc[3] + ((wts[17] > 0.0f) ? wts[17] * po_17[3] : 0.0f);
    wts[18] = smem_w[w_base + 18];
    acc[0] = acc[0] + ((wts[18] > 0.0f) ? wts[18] * po_18[0] : 0.0f);
    acc[1] = acc[1] + ((wts[18] > 0.0f) ? wts[18] * po_18[1] : 0.0f);
    acc[2] = acc[2] + ((wts[18] > 0.0f) ? wts[18] * po_18[2] : 0.0f);
    acc[3] = acc[3] + ((wts[18] > 0.0f) ? wts[18] * po_18[3] : 0.0f);
    wts[19] = smem_w[w_base + 19];
    acc[0] = acc[0] + ((wts[19] > 0.0f) ? wts[19] * po_19[0] : 0.0f);
    acc[1] = acc[1] + ((wts[19] > 0.0f) ? wts[19] * po_19[1] : 0.0f);
    acc[2] = acc[2] + ((wts[19] > 0.0f) ? wts[19] * po_19[2] : 0.0f);
    acc[3] = acc[3] + ((wts[19] > 0.0f) ? wts[19] * po_19[3] : 0.0f);
    wts[20] = smem_w[w_base + 20];
    acc[0] = acc[0] + ((wts[20] > 0.0f) ? wts[20] * po_20[0] : 0.0f);
    acc[1] = acc[1] + ((wts[20] > 0.0f) ? wts[20] * po_20[1] : 0.0f);
    acc[2] = acc[2] + ((wts[20] > 0.0f) ? wts[20] * po_20[2] : 0.0f);
    acc[3] = acc[3] + ((wts[20] > 0.0f) ? wts[20] * po_20[3] : 0.0f);
    wts[21] = smem_w[w_base + 21];
    acc[0] = acc[0] + ((wts[21] > 0.0f) ? wts[21] * po_21[0] : 0.0f);
    acc[1] = acc[1] + ((wts[21] > 0.0f) ? wts[21] * po_21[1] : 0.0f);
    acc[2] = acc[2] + ((wts[21] > 0.0f) ? wts[21] * po_21[2] : 0.0f);
    acc[3] = acc[3] + ((wts[21] > 0.0f) ? wts[21] * po_21[3] : 0.0f);
    wts[22] = smem_w[w_base + 22];
    acc[0] = acc[0] + ((wts[22] > 0.0f) ? wts[22] * po_22[0] : 0.0f);
    acc[1] = acc[1] + ((wts[22] > 0.0f) ? wts[22] * po_22[1] : 0.0f);
    acc[2] = acc[2] + ((wts[22] > 0.0f) ? wts[22] * po_22[2] : 0.0f);
    acc[3] = acc[3] + ((wts[22] > 0.0f) ? wts[22] * po_22[3] : 0.0f);
    wts[23] = smem_w[w_base + 23];
    acc[0] = acc[0] + ((wts[23] > 0.0f) ? wts[23] * po_23[0] : 0.0f);
    acc[1] = acc[1] + ((wts[23] > 0.0f) ? wts[23] * po_23[1] : 0.0f);
    acc[2] = acc[2] + ((wts[23] > 0.0f) ? wts[23] * po_23[2] : 0.0f);
    acc[3] = acc[3] + ((wts[23] > 0.0f) ? wts[23] * po_23[3] : 0.0f);
    for (int e_1 = 0; e_1 < 4; e_1++) {
      acc[e_1] = acc[e_1] * inv;
    }
    unsigned int _packed[2];
#pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
      __nv_bfloat162 _bf2 =
          __float22bfloat162_rn(make_float2(acc[_lp * 2 + 0], acc[_lp * 2 + 1 + 0]));
      _packed[_lp] = *(uint32_t*)&_bf2;
    }
#pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
      __nv_bfloat162 _bf2 =
          __float22bfloat162_rn(make_float2(acc[_lp * 2 + 2], acc[_lp * 2 + 1 + 2]));
      _packed[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(O + (row * 512 + d0))[0] = reinterpret_cast<int2*>(_packed)[0];
    if (has_lse != 0) {
      if (part == 0) {
        if (lane_0 == 0) {
          float lse_v = -CUDART_INF_F;
          if (sum_w > 0.0f) {
            float _log2_27;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_27) : "f"(sum_w));
            lse_v = (m + _log2_27) * 0.6931472f;
          }
          *(reinterpret_cast<float*>(lse + row) + (0)) = lse_v;
        }
      }
    }
  }
}

}  // extern "C"
