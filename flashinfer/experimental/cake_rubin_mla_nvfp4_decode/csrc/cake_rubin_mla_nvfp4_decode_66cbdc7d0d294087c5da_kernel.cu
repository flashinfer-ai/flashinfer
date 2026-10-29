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
__launch_bounds__(256, LAUNCH_MIN_BLOCKS) void kernel_cake_rubin_mla_nvfp4_decode_66cbdc7d0d294087c5da(
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
    int min_1 = ((0) < (last) ? (0) : (last));
    float vec_load[4];
    {
      uint2 _vld_0;
      _vld_0 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_1) * 512 + d0);
      uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
        (&vec_load[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
      }
    }
    int min_2 = ((1) < (last) ? (1) : (last));
    float vec_load_1[4];
    {
      uint2 _vld_1;
      _vld_1 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_2) * 512 + d0);
      uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_1[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
        (&vec_load_1[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
      }
    }
    int min_3 = ((2) < (last) ? (2) : (last));
    float vec_load_2[4];
    {
      uint2 _vld_2;
      _vld_2 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_3) * 512 + d0);
      uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_2[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
        (&vec_load_2[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
      }
    }
    int min_4 = ((3) < (last) ? (3) : (last));
    float vec_load_3[4];
    {
      uint2 _vld_3;
      _vld_3 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_4) * 512 + d0);
      uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_3[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
        (&vec_load_3[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
      }
    }
    int min_5 = ((4) < (last) ? (4) : (last));
    float vec_load_4[4];
    {
      uint2 _vld_4;
      _vld_4 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_5) * 512 + d0);
      uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_4[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
        (&vec_load_4[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
      }
    }
    int min_6 = ((5) < (last) ? (5) : (last));
    float vec_load_5[4];
    {
      uint2 _vld_5;
      _vld_5 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_6) * 512 + d0);
      uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_5[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
        (&vec_load_5[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
      }
    }
    int min_7 = ((6) < (last) ? (6) : (last));
    float vec_load_6[4];
    {
      uint2 _vld_6;
      _vld_6 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_7) * 512 + d0);
      uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_6[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
        (&vec_load_6[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
      }
    }
    int min_8 = ((7) < (last) ? (7) : (last));
    float vec_load_7[4];
    {
      uint2 _vld_7;
      _vld_7 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_8) * 512 + d0);
      uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_7[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
        (&vec_load_7[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
      }
    }
    int min_9 = ((8) < (last) ? (8) : (last));
    float vec_load_8[4];
    {
      uint2 _vld_8;
      _vld_8 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_9) * 512 + d0);
      uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_8[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
        (&vec_load_8[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
      }
    }
    int min_10 = ((9) < (last) ? (9) : (last));
    float vec_load_9[4];
    {
      uint2 _vld_9;
      _vld_9 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_10) * 512 + d0);
      uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_9[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
        (&vec_load_9[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
      }
    }
    int min_11 = ((10) < (last) ? (10) : (last));
    float vec_load_10[4];
    {
      uint2 _vld_10;
      _vld_10 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_11) * 512 + d0);
      uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_10[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
        (&vec_load_10[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
      }
    }
    int min_12 = ((11) < (last) ? (11) : (last));
    float vec_load_11[4];
    {
      uint2 _vld_11;
      _vld_11 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_12) * 512 + d0);
      uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_11[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) << 16);
        (&vec_load_11[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) & 0xffff0000u);
      }
    }
    int min_13 = ((12) < (last) ? (12) : (last));
    float vec_load_12[4];
    {
      uint2 _vld_12;
      _vld_12 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_13) * 512 + d0);
      uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_12[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
        (&vec_load_12[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
      }
    }
    int min_14 = ((13) < (last) ? (13) : (last));
    float vec_load_13[4];
    {
      uint2 _vld_13;
      _vld_13 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_14) * 512 + d0);
      uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_13[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) << 16);
        (&vec_load_13[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) & 0xffff0000u);
      }
    }
    int min_15 = ((14) < (last) ? (14) : (last));
    float vec_load_14[4];
    {
      uint2 _vld_14;
      _vld_14 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_15) * 512 + d0);
      uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_14[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) << 16);
        (&vec_load_14[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) & 0xffff0000u);
      }
    }
    int min_16 = ((15) < (last) ? (15) : (last));
    float vec_load_15[4];
    {
      uint2 _vld_15;
      _vld_15 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_16) * 512 + d0);
      uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_15[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) << 16);
        (&vec_load_15[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) & 0xffff0000u);
      }
    }
    int min_17 = ((16) < (last) ? (16) : (last));
    float vec_load_16[4];
    {
      uint2 _vld_16;
      _vld_16 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_17) * 512 + d0);
      uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_16[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
        (&vec_load_16[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
      }
    }
    int min_18 = ((17) < (last) ? (17) : (last));
    float vec_load_17[4];
    {
      uint2 _vld_17;
      _vld_17 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_18) * 512 + d0);
      uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_17[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) << 16);
        (&vec_load_17[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) & 0xffff0000u);
      }
    }
    int min_19 = ((18) < (last) ? (18) : (last));
    float vec_load_18[4];
    {
      uint2 _vld_18;
      _vld_18 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_19) * 512 + d0);
      uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_18[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
        (&vec_load_18[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
      }
    }
    int min_20 = ((19) < (last) ? (19) : (last));
    float vec_load_19[4];
    {
      uint2 _vld_19;
      _vld_19 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_20) * 512 + d0);
      uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_19[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) << 16);
        (&vec_load_19[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) & 0xffff0000u);
      }
    }
    int min_21 = ((20) < (last) ? (20) : (last));
    float vec_load_20[4];
    {
      uint2 _vld_20;
      _vld_20 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_21) * 512 + d0);
      uint32_t* _vpairs_20 = reinterpret_cast<uint32_t*>(&_vld_20);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_20[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) << 16);
        (&vec_load_20[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) & 0xffff0000u);
      }
    }
    int min_22 = ((21) < (last) ? (21) : (last));
    float vec_load_21[4];
    {
      uint2 _vld_21;
      _vld_21 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_22) * 512 + d0);
      uint32_t* _vpairs_21 = reinterpret_cast<uint32_t*>(&_vld_21);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_21[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) << 16);
        (&vec_load_21[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) & 0xffff0000u);
      }
    }
    int min_23 = ((22) < (last) ? (22) : (last));
    float vec_load_22[4];
    {
      uint2 _vld_22;
      _vld_22 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_23) * 512 + d0);
      uint32_t* _vpairs_22 = reinterpret_cast<uint32_t*>(&_vld_22);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_22[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_22[_pair]) << 16);
        (&vec_load_22[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_22[_pair]) & 0xffff0000u);
      }
    }
    int min_24 = ((23) < (last) ? (23) : (last));
    float vec_load_23[4];
    {
      uint2 _vld_23;
      _vld_23 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_24) * 512 + d0);
      uint32_t* _vpairs_23 = reinterpret_cast<uint32_t*>(&_vld_23);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_23[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_23[_pair]) << 16);
        (&vec_load_23[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_23[_pair]) & 0xffff0000u);
      }
    }
    int min_25 = ((24) < (last) ? (24) : (last));
    float vec_load_24[4];
    {
      uint2 _vld_24;
      _vld_24 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_25) * 512 + d0);
      uint32_t* _vpairs_24 = reinterpret_cast<uint32_t*>(&_vld_24);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_24[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) << 16);
        (&vec_load_24[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) & 0xffff0000u);
      }
    }
    int min_26 = ((25) < (last) ? (25) : (last));
    float vec_load_25[4];
    {
      uint2 _vld_25;
      _vld_25 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_26) * 512 + d0);
      uint32_t* _vpairs_25 = reinterpret_cast<uint32_t*>(&_vld_25);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_25[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) << 16);
        (&vec_load_25[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) & 0xffff0000u);
      }
    }
    int min_27 = ((26) < (last) ? (26) : (last));
    float vec_load_26[4];
    {
      uint2 _vld_26;
      _vld_26 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_27) * 512 + d0);
      uint32_t* _vpairs_26 = reinterpret_cast<uint32_t*>(&_vld_26);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_26[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_26[_pair]) << 16);
        (&vec_load_26[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_26[_pair]) & 0xffff0000u);
      }
    }
    int min_28 = ((27) < (last) ? (27) : (last));
    float vec_load_27[4];
    {
      uint2 _vld_27;
      _vld_27 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_28) * 512 + d0);
      uint32_t* _vpairs_27 = reinterpret_cast<uint32_t*>(&_vld_27);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_27[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_27[_pair]) << 16);
        (&vec_load_27[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_27[_pair]) & 0xffff0000u);
      }
    }
    int min_29 = ((28) < (last) ? (28) : (last));
    float vec_load_28[4];
    {
      uint2 _vld_28;
      _vld_28 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_29) * 512 + d0);
      uint32_t* _vpairs_28 = reinterpret_cast<uint32_t*>(&_vld_28);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_28[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_28[_pair]) << 16);
        (&vec_load_28[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_28[_pair]) & 0xffff0000u);
      }
    }
    int min_30 = ((29) < (last) ? (29) : (last));
    float vec_load_29[4];
    {
      uint2 _vld_29;
      _vld_29 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_30) * 512 + d0);
      uint32_t* _vpairs_29 = reinterpret_cast<uint32_t*>(&_vld_29);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_29[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_29[_pair]) << 16);
        (&vec_load_29[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_29[_pair]) & 0xffff0000u);
      }
    }
    int min_31 = ((30) < (last) ? (30) : (last));
    float vec_load_30[4];
    {
      uint2 _vld_30;
      _vld_30 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_31) * 512 + d0);
      uint32_t* _vpairs_30 = reinterpret_cast<uint32_t*>(&_vld_30);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_30[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) << 16);
        (&vec_load_30[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) & 0xffff0000u);
      }
    }
    int min_32 = ((31) < (last) ? (31) : (last));
    float vec_load_31[4];
    {
      uint2 _vld_31;
      _vld_31 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_32) * 512 + d0);
      uint32_t* _vpairs_31 = reinterpret_cast<uint32_t*>(&_vld_31);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&vec_load_31[0 + _pair * 2])[0] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) << 16);
        (&vec_load_31[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) & 0xffff0000u);
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
      float _exp2_32 = approx_exp2(l0 - m);
      w0 = _exp2_32;
    }
    if (l1 > -CUDART_INF_F) {
      float _exp2_33 = approx_exp2(l1 - m);
      w1 = _exp2_33;
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
      float _rcp_34 = approx_rcp(sum_w);
      inv = _rcp_34;
    }
    float acc[4];
    for (int e = 0; e < 4; e++) {
      acc[e] = 0.0f;
    }
    float wts[32];
    wts[0] = smem_w[w_base];
    acc[0] = acc[0] + ((wts[0] > 0.0f) ? wts[0] * vec_load[0] : 0.0f);
    acc[1] = acc[1] + ((wts[0] > 0.0f) ? wts[0] * vec_load[1] : 0.0f);
    acc[2] = acc[2] + ((wts[0] > 0.0f) ? wts[0] * vec_load[2] : 0.0f);
    acc[3] = acc[3] + ((wts[0] > 0.0f) ? wts[0] * vec_load[3] : 0.0f);
    wts[1] = smem_w[w_base + 1];
    acc[0] = acc[0] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[0] : 0.0f);
    acc[1] = acc[1] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[1] : 0.0f);
    acc[2] = acc[2] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[2] : 0.0f);
    acc[3] = acc[3] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1[3] : 0.0f);
    wts[2] = smem_w[w_base + 2];
    acc[0] = acc[0] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[0] : 0.0f);
    acc[1] = acc[1] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[1] : 0.0f);
    acc[2] = acc[2] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[2] : 0.0f);
    acc[3] = acc[3] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2[3] : 0.0f);
    wts[3] = smem_w[w_base + 3];
    acc[0] = acc[0] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[0] : 0.0f);
    acc[1] = acc[1] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[1] : 0.0f);
    acc[2] = acc[2] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[2] : 0.0f);
    acc[3] = acc[3] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3[3] : 0.0f);
    wts[4] = smem_w[w_base + 4];
    acc[0] = acc[0] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[0] : 0.0f);
    acc[1] = acc[1] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[1] : 0.0f);
    acc[2] = acc[2] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[2] : 0.0f);
    acc[3] = acc[3] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4[3] : 0.0f);
    wts[5] = smem_w[w_base + 5];
    acc[0] = acc[0] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[0] : 0.0f);
    acc[1] = acc[1] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[1] : 0.0f);
    acc[2] = acc[2] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[2] : 0.0f);
    acc[3] = acc[3] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5[3] : 0.0f);
    wts[6] = smem_w[w_base + 6];
    acc[0] = acc[0] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[0] : 0.0f);
    acc[1] = acc[1] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[1] : 0.0f);
    acc[2] = acc[2] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[2] : 0.0f);
    acc[3] = acc[3] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6[3] : 0.0f);
    wts[7] = smem_w[w_base + 7];
    acc[0] = acc[0] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[0] : 0.0f);
    acc[1] = acc[1] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[1] : 0.0f);
    acc[2] = acc[2] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[2] : 0.0f);
    acc[3] = acc[3] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7[3] : 0.0f);
    wts[8] = smem_w[w_base + 8];
    acc[0] = acc[0] + ((wts[8] > 0.0f) ? wts[8] * vec_load_8[0] : 0.0f);
    acc[1] = acc[1] + ((wts[8] > 0.0f) ? wts[8] * vec_load_8[1] : 0.0f);
    acc[2] = acc[2] + ((wts[8] > 0.0f) ? wts[8] * vec_load_8[2] : 0.0f);
    acc[3] = acc[3] + ((wts[8] > 0.0f) ? wts[8] * vec_load_8[3] : 0.0f);
    wts[9] = smem_w[w_base + 9];
    acc[0] = acc[0] + ((wts[9] > 0.0f) ? wts[9] * vec_load_9[0] : 0.0f);
    acc[1] = acc[1] + ((wts[9] > 0.0f) ? wts[9] * vec_load_9[1] : 0.0f);
    acc[2] = acc[2] + ((wts[9] > 0.0f) ? wts[9] * vec_load_9[2] : 0.0f);
    acc[3] = acc[3] + ((wts[9] > 0.0f) ? wts[9] * vec_load_9[3] : 0.0f);
    wts[10] = smem_w[w_base + 10];
    acc[0] = acc[0] + ((wts[10] > 0.0f) ? wts[10] * vec_load_10[0] : 0.0f);
    acc[1] = acc[1] + ((wts[10] > 0.0f) ? wts[10] * vec_load_10[1] : 0.0f);
    acc[2] = acc[2] + ((wts[10] > 0.0f) ? wts[10] * vec_load_10[2] : 0.0f);
    acc[3] = acc[3] + ((wts[10] > 0.0f) ? wts[10] * vec_load_10[3] : 0.0f);
    wts[11] = smem_w[w_base + 11];
    acc[0] = acc[0] + ((wts[11] > 0.0f) ? wts[11] * vec_load_11[0] : 0.0f);
    acc[1] = acc[1] + ((wts[11] > 0.0f) ? wts[11] * vec_load_11[1] : 0.0f);
    acc[2] = acc[2] + ((wts[11] > 0.0f) ? wts[11] * vec_load_11[2] : 0.0f);
    acc[3] = acc[3] + ((wts[11] > 0.0f) ? wts[11] * vec_load_11[3] : 0.0f);
    wts[12] = smem_w[w_base + 12];
    acc[0] = acc[0] + ((wts[12] > 0.0f) ? wts[12] * vec_load_12[0] : 0.0f);
    acc[1] = acc[1] + ((wts[12] > 0.0f) ? wts[12] * vec_load_12[1] : 0.0f);
    acc[2] = acc[2] + ((wts[12] > 0.0f) ? wts[12] * vec_load_12[2] : 0.0f);
    acc[3] = acc[3] + ((wts[12] > 0.0f) ? wts[12] * vec_load_12[3] : 0.0f);
    wts[13] = smem_w[w_base + 13];
    acc[0] = acc[0] + ((wts[13] > 0.0f) ? wts[13] * vec_load_13[0] : 0.0f);
    acc[1] = acc[1] + ((wts[13] > 0.0f) ? wts[13] * vec_load_13[1] : 0.0f);
    acc[2] = acc[2] + ((wts[13] > 0.0f) ? wts[13] * vec_load_13[2] : 0.0f);
    acc[3] = acc[3] + ((wts[13] > 0.0f) ? wts[13] * vec_load_13[3] : 0.0f);
    wts[14] = smem_w[w_base + 14];
    acc[0] = acc[0] + ((wts[14] > 0.0f) ? wts[14] * vec_load_14[0] : 0.0f);
    acc[1] = acc[1] + ((wts[14] > 0.0f) ? wts[14] * vec_load_14[1] : 0.0f);
    acc[2] = acc[2] + ((wts[14] > 0.0f) ? wts[14] * vec_load_14[2] : 0.0f);
    acc[3] = acc[3] + ((wts[14] > 0.0f) ? wts[14] * vec_load_14[3] : 0.0f);
    wts[15] = smem_w[w_base + 15];
    acc[0] = acc[0] + ((wts[15] > 0.0f) ? wts[15] * vec_load_15[0] : 0.0f);
    acc[1] = acc[1] + ((wts[15] > 0.0f) ? wts[15] * vec_load_15[1] : 0.0f);
    acc[2] = acc[2] + ((wts[15] > 0.0f) ? wts[15] * vec_load_15[2] : 0.0f);
    acc[3] = acc[3] + ((wts[15] > 0.0f) ? wts[15] * vec_load_15[3] : 0.0f);
    wts[16] = smem_w[w_base + 16];
    acc[0] = acc[0] + ((wts[16] > 0.0f) ? wts[16] * vec_load_16[0] : 0.0f);
    acc[1] = acc[1] + ((wts[16] > 0.0f) ? wts[16] * vec_load_16[1] : 0.0f);
    acc[2] = acc[2] + ((wts[16] > 0.0f) ? wts[16] * vec_load_16[2] : 0.0f);
    acc[3] = acc[3] + ((wts[16] > 0.0f) ? wts[16] * vec_load_16[3] : 0.0f);
    wts[17] = smem_w[w_base + 17];
    acc[0] = acc[0] + ((wts[17] > 0.0f) ? wts[17] * vec_load_17[0] : 0.0f);
    acc[1] = acc[1] + ((wts[17] > 0.0f) ? wts[17] * vec_load_17[1] : 0.0f);
    acc[2] = acc[2] + ((wts[17] > 0.0f) ? wts[17] * vec_load_17[2] : 0.0f);
    acc[3] = acc[3] + ((wts[17] > 0.0f) ? wts[17] * vec_load_17[3] : 0.0f);
    wts[18] = smem_w[w_base + 18];
    acc[0] = acc[0] + ((wts[18] > 0.0f) ? wts[18] * vec_load_18[0] : 0.0f);
    acc[1] = acc[1] + ((wts[18] > 0.0f) ? wts[18] * vec_load_18[1] : 0.0f);
    acc[2] = acc[2] + ((wts[18] > 0.0f) ? wts[18] * vec_load_18[2] : 0.0f);
    acc[3] = acc[3] + ((wts[18] > 0.0f) ? wts[18] * vec_load_18[3] : 0.0f);
    wts[19] = smem_w[w_base + 19];
    acc[0] = acc[0] + ((wts[19] > 0.0f) ? wts[19] * vec_load_19[0] : 0.0f);
    acc[1] = acc[1] + ((wts[19] > 0.0f) ? wts[19] * vec_load_19[1] : 0.0f);
    acc[2] = acc[2] + ((wts[19] > 0.0f) ? wts[19] * vec_load_19[2] : 0.0f);
    acc[3] = acc[3] + ((wts[19] > 0.0f) ? wts[19] * vec_load_19[3] : 0.0f);
    wts[20] = smem_w[w_base + 20];
    acc[0] = acc[0] + ((wts[20] > 0.0f) ? wts[20] * vec_load_20[0] : 0.0f);
    acc[1] = acc[1] + ((wts[20] > 0.0f) ? wts[20] * vec_load_20[1] : 0.0f);
    acc[2] = acc[2] + ((wts[20] > 0.0f) ? wts[20] * vec_load_20[2] : 0.0f);
    acc[3] = acc[3] + ((wts[20] > 0.0f) ? wts[20] * vec_load_20[3] : 0.0f);
    wts[21] = smem_w[w_base + 21];
    acc[0] = acc[0] + ((wts[21] > 0.0f) ? wts[21] * vec_load_21[0] : 0.0f);
    acc[1] = acc[1] + ((wts[21] > 0.0f) ? wts[21] * vec_load_21[1] : 0.0f);
    acc[2] = acc[2] + ((wts[21] > 0.0f) ? wts[21] * vec_load_21[2] : 0.0f);
    acc[3] = acc[3] + ((wts[21] > 0.0f) ? wts[21] * vec_load_21[3] : 0.0f);
    wts[22] = smem_w[w_base + 22];
    acc[0] = acc[0] + ((wts[22] > 0.0f) ? wts[22] * vec_load_22[0] : 0.0f);
    acc[1] = acc[1] + ((wts[22] > 0.0f) ? wts[22] * vec_load_22[1] : 0.0f);
    acc[2] = acc[2] + ((wts[22] > 0.0f) ? wts[22] * vec_load_22[2] : 0.0f);
    acc[3] = acc[3] + ((wts[22] > 0.0f) ? wts[22] * vec_load_22[3] : 0.0f);
    wts[23] = smem_w[w_base + 23];
    acc[0] = acc[0] + ((wts[23] > 0.0f) ? wts[23] * vec_load_23[0] : 0.0f);
    acc[1] = acc[1] + ((wts[23] > 0.0f) ? wts[23] * vec_load_23[1] : 0.0f);
    acc[2] = acc[2] + ((wts[23] > 0.0f) ? wts[23] * vec_load_23[2] : 0.0f);
    acc[3] = acc[3] + ((wts[23] > 0.0f) ? wts[23] * vec_load_23[3] : 0.0f);
    wts[24] = smem_w[w_base + 24];
    acc[0] = acc[0] + ((wts[24] > 0.0f) ? wts[24] * vec_load_24[0] : 0.0f);
    acc[1] = acc[1] + ((wts[24] > 0.0f) ? wts[24] * vec_load_24[1] : 0.0f);
    acc[2] = acc[2] + ((wts[24] > 0.0f) ? wts[24] * vec_load_24[2] : 0.0f);
    acc[3] = acc[3] + ((wts[24] > 0.0f) ? wts[24] * vec_load_24[3] : 0.0f);
    wts[25] = smem_w[w_base + 25];
    acc[0] = acc[0] + ((wts[25] > 0.0f) ? wts[25] * vec_load_25[0] : 0.0f);
    acc[1] = acc[1] + ((wts[25] > 0.0f) ? wts[25] * vec_load_25[1] : 0.0f);
    acc[2] = acc[2] + ((wts[25] > 0.0f) ? wts[25] * vec_load_25[2] : 0.0f);
    acc[3] = acc[3] + ((wts[25] > 0.0f) ? wts[25] * vec_load_25[3] : 0.0f);
    wts[26] = smem_w[w_base + 26];
    acc[0] = acc[0] + ((wts[26] > 0.0f) ? wts[26] * vec_load_26[0] : 0.0f);
    acc[1] = acc[1] + ((wts[26] > 0.0f) ? wts[26] * vec_load_26[1] : 0.0f);
    acc[2] = acc[2] + ((wts[26] > 0.0f) ? wts[26] * vec_load_26[2] : 0.0f);
    acc[3] = acc[3] + ((wts[26] > 0.0f) ? wts[26] * vec_load_26[3] : 0.0f);
    wts[27] = smem_w[w_base + 27];
    acc[0] = acc[0] + ((wts[27] > 0.0f) ? wts[27] * vec_load_27[0] : 0.0f);
    acc[1] = acc[1] + ((wts[27] > 0.0f) ? wts[27] * vec_load_27[1] : 0.0f);
    acc[2] = acc[2] + ((wts[27] > 0.0f) ? wts[27] * vec_load_27[2] : 0.0f);
    acc[3] = acc[3] + ((wts[27] > 0.0f) ? wts[27] * vec_load_27[3] : 0.0f);
    wts[28] = smem_w[w_base + 28];
    acc[0] = acc[0] + ((wts[28] > 0.0f) ? wts[28] * vec_load_28[0] : 0.0f);
    acc[1] = acc[1] + ((wts[28] > 0.0f) ? wts[28] * vec_load_28[1] : 0.0f);
    acc[2] = acc[2] + ((wts[28] > 0.0f) ? wts[28] * vec_load_28[2] : 0.0f);
    acc[3] = acc[3] + ((wts[28] > 0.0f) ? wts[28] * vec_load_28[3] : 0.0f);
    wts[29] = smem_w[w_base + 29];
    acc[0] = acc[0] + ((wts[29] > 0.0f) ? wts[29] * vec_load_29[0] : 0.0f);
    acc[1] = acc[1] + ((wts[29] > 0.0f) ? wts[29] * vec_load_29[1] : 0.0f);
    acc[2] = acc[2] + ((wts[29] > 0.0f) ? wts[29] * vec_load_29[2] : 0.0f);
    acc[3] = acc[3] + ((wts[29] > 0.0f) ? wts[29] * vec_load_29[3] : 0.0f);
    wts[30] = smem_w[w_base + 30];
    acc[0] = acc[0] + ((wts[30] > 0.0f) ? wts[30] * vec_load_30[0] : 0.0f);
    acc[1] = acc[1] + ((wts[30] > 0.0f) ? wts[30] * vec_load_30[1] : 0.0f);
    acc[2] = acc[2] + ((wts[30] > 0.0f) ? wts[30] * vec_load_30[2] : 0.0f);
    acc[3] = acc[3] + ((wts[30] > 0.0f) ? wts[30] * vec_load_30[3] : 0.0f);
    wts[31] = smem_w[w_base + 31];
    acc[0] = acc[0] + ((wts[31] > 0.0f) ? wts[31] * vec_load_31[0] : 0.0f);
    acc[1] = acc[1] + ((wts[31] > 0.0f) ? wts[31] * vec_load_31[1] : 0.0f);
    acc[2] = acc[2] + ((wts[31] > 0.0f) ? wts[31] * vec_load_31[2] : 0.0f);
    acc[3] = acc[3] + ((wts[31] > 0.0f) ? wts[31] * vec_load_31[3] : 0.0f);
    int n_batches = (num_split + 32 - 1) / 32;
#pragma unroll 1
    for (int bt = 1; bt < n_batches; bt++) {
      int k0 = bt * 32;
      int min_1_0 = ((k0) < (last) ? (k0) : (last));
      float vec_load_0[4];
      {
        uint2 _vld_35;
        _vld_35 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_1_0) * 512 + d0);
        uint32_t* _vpairs_35 = reinterpret_cast<uint32_t*>(&_vld_35);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_35[_pair]) << 16);
          (&vec_load_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_35[_pair]) & 0xffff0000u);
        }
      }
      int min_2_0 = ((k0 + 1) < (last) ? (k0 + 1) : (last));
      float vec_load_1_0[4];
      {
        uint2 _vld_36;
        _vld_36 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_2_0) * 512 + d0);
        uint32_t* _vpairs_36 = reinterpret_cast<uint32_t*>(&_vld_36);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_1_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) << 16);
          (&vec_load_1_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) & 0xffff0000u);
        }
      }
      int min_3_0 = ((k0 + 2) < (last) ? (k0 + 2) : (last));
      float vec_load_2_0[4];
      {
        uint2 _vld_37;
        _vld_37 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_3_0) * 512 + d0);
        uint32_t* _vpairs_37 = reinterpret_cast<uint32_t*>(&_vld_37);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_2_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_37[_pair]) << 16);
          (&vec_load_2_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_37[_pair]) & 0xffff0000u);
        }
      }
      int min_4_0 = ((k0 + 3) < (last) ? (k0 + 3) : (last));
      float vec_load_3_0[4];
      {
        uint2 _vld_38;
        _vld_38 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_4_0) * 512 + d0);
        uint32_t* _vpairs_38 = reinterpret_cast<uint32_t*>(&_vld_38);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_3_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_38[_pair]) << 16);
          (&vec_load_3_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_38[_pair]) & 0xffff0000u);
        }
      }
      int min_5_0 = ((k0 + 4) < (last) ? (k0 + 4) : (last));
      float vec_load_4_0[4];
      {
        uint2 _vld_39;
        _vld_39 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_5_0) * 512 + d0);
        uint32_t* _vpairs_39 = reinterpret_cast<uint32_t*>(&_vld_39);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_4_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_39[_pair]) << 16);
          (&vec_load_4_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_39[_pair]) & 0xffff0000u);
        }
      }
      int min_6_0 = ((k0 + 5) < (last) ? (k0 + 5) : (last));
      float vec_load_5_0[4];
      {
        uint2 _vld_40;
        _vld_40 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_6_0) * 512 + d0);
        uint32_t* _vpairs_40 = reinterpret_cast<uint32_t*>(&_vld_40);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_5_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_40[_pair]) << 16);
          (&vec_load_5_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_40[_pair]) & 0xffff0000u);
        }
      }
      int min_7_0 = ((k0 + 6) < (last) ? (k0 + 6) : (last));
      float vec_load_6_0[4];
      {
        uint2 _vld_41;
        _vld_41 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_7_0) * 512 + d0);
        uint32_t* _vpairs_41 = reinterpret_cast<uint32_t*>(&_vld_41);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_6_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) << 16);
          (&vec_load_6_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) & 0xffff0000u);
        }
      }
      int min_8_0 = ((k0 + 7) < (last) ? (k0 + 7) : (last));
      float vec_load_7_0[4];
      {
        uint2 _vld_42;
        _vld_42 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_8_0) * 512 + d0);
        uint32_t* _vpairs_42 = reinterpret_cast<uint32_t*>(&_vld_42);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_7_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) << 16);
          (&vec_load_7_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) & 0xffff0000u);
        }
      }
      int min_9_0 = ((k0 + 8) < (last) ? (k0 + 8) : (last));
      float vec_load_8_0[4];
      {
        uint2 _vld_43;
        _vld_43 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_9_0) * 512 + d0);
        uint32_t* _vpairs_43 = reinterpret_cast<uint32_t*>(&_vld_43);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_8_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) << 16);
          (&vec_load_8_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) & 0xffff0000u);
        }
      }
      int min_10_0 = ((k0 + 9) < (last) ? (k0 + 9) : (last));
      float vec_load_9_0[4];
      {
        uint2 _vld_44;
        _vld_44 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_10_0) * 512 + d0);
        uint32_t* _vpairs_44 = reinterpret_cast<uint32_t*>(&_vld_44);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_9_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_44[_pair]) << 16);
          (&vec_load_9_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_44[_pair]) & 0xffff0000u);
        }
      }
      int min_11_0 = ((k0 + 10) < (last) ? (k0 + 10) : (last));
      float vec_load_10_0[4];
      {
        uint2 _vld_45;
        _vld_45 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_11_0) * 512 + d0);
        uint32_t* _vpairs_45 = reinterpret_cast<uint32_t*>(&_vld_45);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_10_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_45[_pair]) << 16);
          (&vec_load_10_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_45[_pair]) & 0xffff0000u);
        }
      }
      int min_12_0 = ((k0 + 11) < (last) ? (k0 + 11) : (last));
      float vec_load_11_0[4];
      {
        uint2 _vld_46;
        _vld_46 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_12_0) * 512 + d0);
        uint32_t* _vpairs_46 = reinterpret_cast<uint32_t*>(&_vld_46);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_11_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_46[_pair]) << 16);
          (&vec_load_11_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_46[_pair]) & 0xffff0000u);
        }
      }
      int min_13_0 = ((k0 + 12) < (last) ? (k0 + 12) : (last));
      float vec_load_12_0[4];
      {
        uint2 _vld_47;
        _vld_47 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_13_0) * 512 + d0);
        uint32_t* _vpairs_47 = reinterpret_cast<uint32_t*>(&_vld_47);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_12_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_47[_pair]) << 16);
          (&vec_load_12_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_47[_pair]) & 0xffff0000u);
        }
      }
      int min_14_0 = ((k0 + 13) < (last) ? (k0 + 13) : (last));
      float vec_load_13_0[4];
      {
        uint2 _vld_48;
        _vld_48 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_14_0) * 512 + d0);
        uint32_t* _vpairs_48 = reinterpret_cast<uint32_t*>(&_vld_48);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_13_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_48[_pair]) << 16);
          (&vec_load_13_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_48[_pair]) & 0xffff0000u);
        }
      }
      int min_15_0 = ((k0 + 14) < (last) ? (k0 + 14) : (last));
      float vec_load_14_0[4];
      {
        uint2 _vld_49;
        _vld_49 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_15_0) * 512 + d0);
        uint32_t* _vpairs_49 = reinterpret_cast<uint32_t*>(&_vld_49);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_14_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_49[_pair]) << 16);
          (&vec_load_14_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_49[_pair]) & 0xffff0000u);
        }
      }
      int min_16_0 = ((k0 + 15) < (last) ? (k0 + 15) : (last));
      float vec_load_15_0[4];
      {
        uint2 _vld_50;
        _vld_50 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_16_0) * 512 + d0);
        uint32_t* _vpairs_50 = reinterpret_cast<uint32_t*>(&_vld_50);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_15_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_50[_pair]) << 16);
          (&vec_load_15_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_50[_pair]) & 0xffff0000u);
        }
      }
      int min_17_0 = ((k0 + 16) < (last) ? (k0 + 16) : (last));
      float vec_load_16_0[4];
      {
        uint2 _vld_51;
        _vld_51 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_17_0) * 512 + d0);
        uint32_t* _vpairs_51 = reinterpret_cast<uint32_t*>(&_vld_51);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_16_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_51[_pair]) << 16);
          (&vec_load_16_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_51[_pair]) & 0xffff0000u);
        }
      }
      int min_18_0 = ((k0 + 17) < (last) ? (k0 + 17) : (last));
      float vec_load_17_0[4];
      {
        uint2 _vld_52;
        _vld_52 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_18_0) * 512 + d0);
        uint32_t* _vpairs_52 = reinterpret_cast<uint32_t*>(&_vld_52);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_17_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_52[_pair]) << 16);
          (&vec_load_17_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_52[_pair]) & 0xffff0000u);
        }
      }
      int min_19_0 = ((k0 + 18) < (last) ? (k0 + 18) : (last));
      float vec_load_18_0[4];
      {
        uint2 _vld_53;
        _vld_53 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_19_0) * 512 + d0);
        uint32_t* _vpairs_53 = reinterpret_cast<uint32_t*>(&_vld_53);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_18_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_53[_pair]) << 16);
          (&vec_load_18_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_53[_pair]) & 0xffff0000u);
        }
      }
      int min_20_0 = ((k0 + 19) < (last) ? (k0 + 19) : (last));
      float vec_load_19_0[4];
      {
        uint2 _vld_54;
        _vld_54 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_20_0) * 512 + d0);
        uint32_t* _vpairs_54 = reinterpret_cast<uint32_t*>(&_vld_54);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_19_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_54[_pair]) << 16);
          (&vec_load_19_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_54[_pair]) & 0xffff0000u);
        }
      }
      int min_21_0 = ((k0 + 20) < (last) ? (k0 + 20) : (last));
      float vec_load_20_0[4];
      {
        uint2 _vld_55;
        _vld_55 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_21_0) * 512 + d0);
        uint32_t* _vpairs_55 = reinterpret_cast<uint32_t*>(&_vld_55);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_20_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_55[_pair]) << 16);
          (&vec_load_20_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_55[_pair]) & 0xffff0000u);
        }
      }
      int min_22_0 = ((k0 + 21) < (last) ? (k0 + 21) : (last));
      float vec_load_21_0[4];
      {
        uint2 _vld_56;
        _vld_56 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_22_0) * 512 + d0);
        uint32_t* _vpairs_56 = reinterpret_cast<uint32_t*>(&_vld_56);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_21_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_56[_pair]) << 16);
          (&vec_load_21_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_56[_pair]) & 0xffff0000u);
        }
      }
      int min_23_0 = ((k0 + 22) < (last) ? (k0 + 22) : (last));
      float vec_load_22_0[4];
      {
        uint2 _vld_57;
        _vld_57 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_23_0) * 512 + d0);
        uint32_t* _vpairs_57 = reinterpret_cast<uint32_t*>(&_vld_57);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_22_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_57[_pair]) << 16);
          (&vec_load_22_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_57[_pair]) & 0xffff0000u);
        }
      }
      int min_24_0 = ((k0 + 23) < (last) ? (k0 + 23) : (last));
      float vec_load_23_0[4];
      {
        uint2 _vld_58;
        _vld_58 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_24_0) * 512 + d0);
        uint32_t* _vpairs_58 = reinterpret_cast<uint32_t*>(&_vld_58);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_23_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_58[_pair]) << 16);
          (&vec_load_23_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_58[_pair]) & 0xffff0000u);
        }
      }
      int min_25_0 = ((k0 + 24) < (last) ? (k0 + 24) : (last));
      float vec_load_24_0[4];
      {
        uint2 _vld_59;
        _vld_59 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_25_0) * 512 + d0);
        uint32_t* _vpairs_59 = reinterpret_cast<uint32_t*>(&_vld_59);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_24_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_59[_pair]) << 16);
          (&vec_load_24_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_59[_pair]) & 0xffff0000u);
        }
      }
      int min_26_0 = ((k0 + 25) < (last) ? (k0 + 25) : (last));
      float vec_load_25_0[4];
      {
        uint2 _vld_60;
        _vld_60 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_26_0) * 512 + d0);
        uint32_t* _vpairs_60 = reinterpret_cast<uint32_t*>(&_vld_60);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_25_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_60[_pair]) << 16);
          (&vec_load_25_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_60[_pair]) & 0xffff0000u);
        }
      }
      int min_27_0 = ((k0 + 26) < (last) ? (k0 + 26) : (last));
      float vec_load_26_0[4];
      {
        uint2 _vld_61;
        _vld_61 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_27_0) * 512 + d0);
        uint32_t* _vpairs_61 = reinterpret_cast<uint32_t*>(&_vld_61);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_26_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_61[_pair]) << 16);
          (&vec_load_26_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_61[_pair]) & 0xffff0000u);
        }
      }
      int min_28_0 = ((k0 + 27) < (last) ? (k0 + 27) : (last));
      float vec_load_27_0[4];
      {
        uint2 _vld_62;
        _vld_62 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_28_0) * 512 + d0);
        uint32_t* _vpairs_62 = reinterpret_cast<uint32_t*>(&_vld_62);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_27_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_62[_pair]) << 16);
          (&vec_load_27_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_62[_pair]) & 0xffff0000u);
        }
      }
      int min_29_0 = ((k0 + 28) < (last) ? (k0 + 28) : (last));
      float vec_load_28_0[4];
      {
        uint2 _vld_63;
        _vld_63 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_29_0) * 512 + d0);
        uint32_t* _vpairs_63 = reinterpret_cast<uint32_t*>(&_vld_63);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_28_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_63[_pair]) << 16);
          (&vec_load_28_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_63[_pair]) & 0xffff0000u);
        }
      }
      int min_30_0 = ((k0 + 29) < (last) ? (k0 + 29) : (last));
      float vec_load_29_0[4];
      {
        uint2 _vld_64;
        _vld_64 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_30_0) * 512 + d0);
        uint32_t* _vpairs_64 = reinterpret_cast<uint32_t*>(&_vld_64);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_29_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) << 16);
          (&vec_load_29_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) & 0xffff0000u);
        }
      }
      int min_31_0 = ((k0 + 30) < (last) ? (k0 + 30) : (last));
      float vec_load_30_0[4];
      {
        uint2 _vld_65;
        _vld_65 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_31_0) * 512 + d0);
        uint32_t* _vpairs_65 = reinterpret_cast<uint32_t*>(&_vld_65);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_30_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_65[_pair]) << 16);
          (&vec_load_30_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_65[_pair]) & 0xffff0000u);
        }
      }
      int min_32_0 = ((k0 + 31) < (last) ? (k0 + 31) : (last));
      float vec_load_31_0[4];
      {
        uint2 _vld_66;
        _vld_66 = *reinterpret_cast<const uint2*>(partial_o + (stat_base + min_32_0) * 512 + d0);
        uint32_t* _vpairs_66 = reinterpret_cast<uint32_t*>(&_vld_66);
#pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
          (&vec_load_31_0[0 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_66[_pair]) << 16);
          (&vec_load_31_0[0 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_66[_pair]) & 0xffff0000u);
        }
      }
      wts[0] = smem_w[w_base + k0];
      acc[0] = acc[0] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[0] > 0.0f) ? wts[0] * vec_load_0[3] : 0.0f);
      wts[1] = smem_w[w_base + k0 + 1];
      acc[0] = acc[0] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[1] > 0.0f) ? wts[1] * vec_load_1_0[3] : 0.0f);
      wts[2] = smem_w[w_base + k0 + 2];
      acc[0] = acc[0] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[2] > 0.0f) ? wts[2] * vec_load_2_0[3] : 0.0f);
      wts[3] = smem_w[w_base + k0 + 3];
      acc[0] = acc[0] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[3] > 0.0f) ? wts[3] * vec_load_3_0[3] : 0.0f);
      wts[4] = smem_w[w_base + k0 + 4];
      acc[0] = acc[0] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[4] > 0.0f) ? wts[4] * vec_load_4_0[3] : 0.0f);
      wts[5] = smem_w[w_base + k0 + 5];
      acc[0] = acc[0] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[5] > 0.0f) ? wts[5] * vec_load_5_0[3] : 0.0f);
      wts[6] = smem_w[w_base + k0 + 6];
      acc[0] = acc[0] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[6] > 0.0f) ? wts[6] * vec_load_6_0[3] : 0.0f);
      wts[7] = smem_w[w_base + k0 + 7];
      acc[0] = acc[0] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[7] > 0.0f) ? wts[7] * vec_load_7_0[3] : 0.0f);
      wts[8] = smem_w[w_base + k0 + 8];
      acc[0] = acc[0] + ((wts[8] > 0.0f) ? wts[8] * vec_load_8_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[8] > 0.0f) ? wts[8] * vec_load_8_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[8] > 0.0f) ? wts[8] * vec_load_8_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[8] > 0.0f) ? wts[8] * vec_load_8_0[3] : 0.0f);
      wts[9] = smem_w[w_base + k0 + 9];
      acc[0] = acc[0] + ((wts[9] > 0.0f) ? wts[9] * vec_load_9_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[9] > 0.0f) ? wts[9] * vec_load_9_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[9] > 0.0f) ? wts[9] * vec_load_9_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[9] > 0.0f) ? wts[9] * vec_load_9_0[3] : 0.0f);
      wts[10] = smem_w[w_base + k0 + 10];
      acc[0] = acc[0] + ((wts[10] > 0.0f) ? wts[10] * vec_load_10_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[10] > 0.0f) ? wts[10] * vec_load_10_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[10] > 0.0f) ? wts[10] * vec_load_10_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[10] > 0.0f) ? wts[10] * vec_load_10_0[3] : 0.0f);
      wts[11] = smem_w[w_base + k0 + 11];
      acc[0] = acc[0] + ((wts[11] > 0.0f) ? wts[11] * vec_load_11_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[11] > 0.0f) ? wts[11] * vec_load_11_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[11] > 0.0f) ? wts[11] * vec_load_11_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[11] > 0.0f) ? wts[11] * vec_load_11_0[3] : 0.0f);
      wts[12] = smem_w[w_base + k0 + 12];
      acc[0] = acc[0] + ((wts[12] > 0.0f) ? wts[12] * vec_load_12_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[12] > 0.0f) ? wts[12] * vec_load_12_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[12] > 0.0f) ? wts[12] * vec_load_12_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[12] > 0.0f) ? wts[12] * vec_load_12_0[3] : 0.0f);
      wts[13] = smem_w[w_base + k0 + 13];
      acc[0] = acc[0] + ((wts[13] > 0.0f) ? wts[13] * vec_load_13_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[13] > 0.0f) ? wts[13] * vec_load_13_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[13] > 0.0f) ? wts[13] * vec_load_13_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[13] > 0.0f) ? wts[13] * vec_load_13_0[3] : 0.0f);
      wts[14] = smem_w[w_base + k0 + 14];
      acc[0] = acc[0] + ((wts[14] > 0.0f) ? wts[14] * vec_load_14_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[14] > 0.0f) ? wts[14] * vec_load_14_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[14] > 0.0f) ? wts[14] * vec_load_14_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[14] > 0.0f) ? wts[14] * vec_load_14_0[3] : 0.0f);
      wts[15] = smem_w[w_base + k0 + 15];
      acc[0] = acc[0] + ((wts[15] > 0.0f) ? wts[15] * vec_load_15_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[15] > 0.0f) ? wts[15] * vec_load_15_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[15] > 0.0f) ? wts[15] * vec_load_15_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[15] > 0.0f) ? wts[15] * vec_load_15_0[3] : 0.0f);
      wts[16] = smem_w[w_base + k0 + 16];
      acc[0] = acc[0] + ((wts[16] > 0.0f) ? wts[16] * vec_load_16_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[16] > 0.0f) ? wts[16] * vec_load_16_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[16] > 0.0f) ? wts[16] * vec_load_16_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[16] > 0.0f) ? wts[16] * vec_load_16_0[3] : 0.0f);
      wts[17] = smem_w[w_base + k0 + 17];
      acc[0] = acc[0] + ((wts[17] > 0.0f) ? wts[17] * vec_load_17_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[17] > 0.0f) ? wts[17] * vec_load_17_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[17] > 0.0f) ? wts[17] * vec_load_17_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[17] > 0.0f) ? wts[17] * vec_load_17_0[3] : 0.0f);
      wts[18] = smem_w[w_base + k0 + 18];
      acc[0] = acc[0] + ((wts[18] > 0.0f) ? wts[18] * vec_load_18_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[18] > 0.0f) ? wts[18] * vec_load_18_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[18] > 0.0f) ? wts[18] * vec_load_18_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[18] > 0.0f) ? wts[18] * vec_load_18_0[3] : 0.0f);
      wts[19] = smem_w[w_base + k0 + 19];
      acc[0] = acc[0] + ((wts[19] > 0.0f) ? wts[19] * vec_load_19_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[19] > 0.0f) ? wts[19] * vec_load_19_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[19] > 0.0f) ? wts[19] * vec_load_19_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[19] > 0.0f) ? wts[19] * vec_load_19_0[3] : 0.0f);
      wts[20] = smem_w[w_base + k0 + 20];
      acc[0] = acc[0] + ((wts[20] > 0.0f) ? wts[20] * vec_load_20_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[20] > 0.0f) ? wts[20] * vec_load_20_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[20] > 0.0f) ? wts[20] * vec_load_20_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[20] > 0.0f) ? wts[20] * vec_load_20_0[3] : 0.0f);
      wts[21] = smem_w[w_base + k0 + 21];
      acc[0] = acc[0] + ((wts[21] > 0.0f) ? wts[21] * vec_load_21_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[21] > 0.0f) ? wts[21] * vec_load_21_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[21] > 0.0f) ? wts[21] * vec_load_21_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[21] > 0.0f) ? wts[21] * vec_load_21_0[3] : 0.0f);
      wts[22] = smem_w[w_base + k0 + 22];
      acc[0] = acc[0] + ((wts[22] > 0.0f) ? wts[22] * vec_load_22_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[22] > 0.0f) ? wts[22] * vec_load_22_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[22] > 0.0f) ? wts[22] * vec_load_22_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[22] > 0.0f) ? wts[22] * vec_load_22_0[3] : 0.0f);
      wts[23] = smem_w[w_base + k0 + 23];
      acc[0] = acc[0] + ((wts[23] > 0.0f) ? wts[23] * vec_load_23_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[23] > 0.0f) ? wts[23] * vec_load_23_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[23] > 0.0f) ? wts[23] * vec_load_23_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[23] > 0.0f) ? wts[23] * vec_load_23_0[3] : 0.0f);
      wts[24] = smem_w[w_base + k0 + 24];
      acc[0] = acc[0] + ((wts[24] > 0.0f) ? wts[24] * vec_load_24_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[24] > 0.0f) ? wts[24] * vec_load_24_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[24] > 0.0f) ? wts[24] * vec_load_24_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[24] > 0.0f) ? wts[24] * vec_load_24_0[3] : 0.0f);
      wts[25] = smem_w[w_base + k0 + 25];
      acc[0] = acc[0] + ((wts[25] > 0.0f) ? wts[25] * vec_load_25_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[25] > 0.0f) ? wts[25] * vec_load_25_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[25] > 0.0f) ? wts[25] * vec_load_25_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[25] > 0.0f) ? wts[25] * vec_load_25_0[3] : 0.0f);
      wts[26] = smem_w[w_base + k0 + 26];
      acc[0] = acc[0] + ((wts[26] > 0.0f) ? wts[26] * vec_load_26_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[26] > 0.0f) ? wts[26] * vec_load_26_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[26] > 0.0f) ? wts[26] * vec_load_26_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[26] > 0.0f) ? wts[26] * vec_load_26_0[3] : 0.0f);
      wts[27] = smem_w[w_base + k0 + 27];
      acc[0] = acc[0] + ((wts[27] > 0.0f) ? wts[27] * vec_load_27_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[27] > 0.0f) ? wts[27] * vec_load_27_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[27] > 0.0f) ? wts[27] * vec_load_27_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[27] > 0.0f) ? wts[27] * vec_load_27_0[3] : 0.0f);
      wts[28] = smem_w[w_base + k0 + 28];
      acc[0] = acc[0] + ((wts[28] > 0.0f) ? wts[28] * vec_load_28_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[28] > 0.0f) ? wts[28] * vec_load_28_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[28] > 0.0f) ? wts[28] * vec_load_28_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[28] > 0.0f) ? wts[28] * vec_load_28_0[3] : 0.0f);
      wts[29] = smem_w[w_base + k0 + 29];
      acc[0] = acc[0] + ((wts[29] > 0.0f) ? wts[29] * vec_load_29_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[29] > 0.0f) ? wts[29] * vec_load_29_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[29] > 0.0f) ? wts[29] * vec_load_29_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[29] > 0.0f) ? wts[29] * vec_load_29_0[3] : 0.0f);
      wts[30] = smem_w[w_base + k0 + 30];
      acc[0] = acc[0] + ((wts[30] > 0.0f) ? wts[30] * vec_load_30_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[30] > 0.0f) ? wts[30] * vec_load_30_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[30] > 0.0f) ? wts[30] * vec_load_30_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[30] > 0.0f) ? wts[30] * vec_load_30_0[3] : 0.0f);
      wts[31] = smem_w[w_base + k0 + 31];
      acc[0] = acc[0] + ((wts[31] > 0.0f) ? wts[31] * vec_load_31_0[0] : 0.0f);
      acc[1] = acc[1] + ((wts[31] > 0.0f) ? wts[31] * vec_load_31_0[1] : 0.0f);
      acc[2] = acc[2] + ((wts[31] > 0.0f) ? wts[31] * vec_load_31_0[2] : 0.0f);
      acc[3] = acc[3] + ((wts[31] > 0.0f) ? wts[31] * vec_load_31_0[3] : 0.0f);
    }
    for (int e_1 = 0; e_1 < 4; e_1++) {
      acc[e_1] = acc[e_1] * inv;
    }
    unsigned int _packed_0[2];
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
    reinterpret_cast<int2*>(O + (row * 512 + d0))[0] = reinterpret_cast<int2*>(_packed_0)[0];
    if (has_lse != 0) {
      if (part == 0) {
        if (lane_0 == 0) {
          float lse_v = -CUDART_INF_F;
          if (sum_w > 0.0f) {
            float _log2_67;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_67) : "f"(sum_w));
            lse_v = (m + _log2_67) * 0.6931472f;
          }
          *(reinterpret_cast<float*>(lse + row) + (0)) = lse_v;
        }
      }
    }
  }
}

}  // extern "C"
