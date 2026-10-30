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
__launch_bounds__(256, LAUNCH_MIN_BLOCKS) void kernel_cake_rubin_mla_nvfp4_decode_24df5b75726c0c32ca0e(
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
  int part_7 = warp % 2;
  int row_7 = blockIdx.x * 4 + warp / 2;
  int lane_7 = lane;
  int stat_base_7 = row_7 * num_split;
  int w_base_7 = warp * 64;
  int d0_7 = part_7 * 256 + lane_7 * 8;
  int last_7 = num_split - 1;
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (row_7 < rows) {
    float l0 = -CUDART_INF_F;
    float l1 = -CUDART_INF_F;
    if (lane_7 < num_split) {
      l0 = partial_lse[stat_base_7 + lane_7];
    }
    if (lane_7 + 32 < num_split) {
      l1 = partial_lse[stat_base_7 + lane_7 + 32];
    }
    float vec[128];
    int _mn_0 = ((0) < (last_7) ? (0) : (last_7));
    {
      const uint4* _vptr_0 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_0) * 512 + d0_7);
      uint4 _vld_0[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_0[_blk] = _vptr_0[_blk];
        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
          (&vec[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
        }
      }
    }
    int _mn_1 = ((1) < (last_7) ? (1) : (last_7));
    {
      const uint4* _vptr_1 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_1) * 512 + d0_7);
      uint4 _vld_1[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_1[_blk] = _vptr_1[_blk];
        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec[8 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
          (&vec[8 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
        }
      }
    }
    int _mn_2 = ((2) < (last_7) ? (2) : (last_7));
    {
      const uint4* _vptr_2 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_2) * 512 + d0_7);
      uint4 _vld_2[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_2[_blk] = _vptr_2[_blk];
        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec[16 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
          (&vec[16 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
        }
      }
    }
    int _mn_3 = ((3) < (last_7) ? (3) : (last_7));
    {
      const uint4* _vptr_3 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_3) * 512 + d0_7);
      uint4 _vld_3[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_3[_blk] = _vptr_3[_blk];
        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec[24 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
          (&vec[24 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
        }
      }
    }
    int _mn_4 = ((4) < (last_7) ? (4) : (last_7));
    {
      const uint4* _vptr_4 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_4) * 512 + d0_7);
      uint4 _vld_4[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_4[_blk] = _vptr_4[_blk];
        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec[32 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
          (&vec[32 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
        }
      }
    }
    int _mn_5 = ((5) < (last_7) ? (5) : (last_7));
    {
      const uint4* _vptr_5 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_5) * 512 + d0_7);
      uint4 _vld_5[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_5[_blk] = _vptr_5[_blk];
        uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec[40 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
          (&vec[40 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
        }
      }
    }
    int _mn_6 = ((6) < (last_7) ? (6) : (last_7));
    {
      const uint4* _vptr_6 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_6) * 512 + d0_7);
      uint4 _vld_6[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_6[_blk] = _vptr_6[_blk];
        uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec[48 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
          (&vec[48 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
        }
      }
    }
    int _mn_7 = ((7) < (last_7) ? (7) : (last_7));
    {
      const uint4* _vptr_7 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_7) * 512 + d0_7);
      uint4 _vld_7[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_7[_blk] = _vptr_7[_blk];
        uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec[56 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
          (&vec[56 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
        }
      }
    }
    int _mn_8 = ((8) < (last_7) ? (8) : (last_7));
    {
      const uint4* _vptr_8 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_8) * 512 + d0_7);
      uint4 _vld_8[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_8[_blk] = _vptr_8[_blk];
        uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec[64 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
          (&vec[64 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
        }
      }
    }
    int _mn_9 = ((9) < (last_7) ? (9) : (last_7));
    {
      const uint4* _vptr_9 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_9) * 512 + d0_7);
      uint4 _vld_9[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_9[_blk] = _vptr_9[_blk];
        uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec[72 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
          (&vec[72 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
        }
      }
    }
    int _mn_10 = ((10) < (last_7) ? (10) : (last_7));
    {
      const uint4* _vptr_10 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_10) * 512 + d0_7);
      uint4 _vld_10[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_10[_blk] = _vptr_10[_blk];
        uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec[80 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
          (&vec[80 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
        }
      }
    }
    int _mn_11 = ((11) < (last_7) ? (11) : (last_7));
    {
      const uint4* _vptr_11 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_11) * 512 + d0_7);
      uint4 _vld_11[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_11[_blk] = _vptr_11[_blk];
        uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec[88 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) << 16);
          (&vec[88 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) & 0xffff0000u);
        }
      }
    }
    int _mn_12 = ((12) < (last_7) ? (12) : (last_7));
    {
      const uint4* _vptr_12 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_12) * 512 + d0_7);
      uint4 _vld_12[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_12[_blk] = _vptr_12[_blk];
        uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec[96 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
          (&vec[96 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
        }
      }
    }
    int _mn_13 = ((13) < (last_7) ? (13) : (last_7));
    {
      const uint4* _vptr_13 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_13) * 512 + d0_7);
      uint4 _vld_13[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_13[_blk] = _vptr_13[_blk];
        uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec[104 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) << 16);
          (&vec[104 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) & 0xffff0000u);
        }
      }
    }
    int _mn_14 = ((14) < (last_7) ? (14) : (last_7));
    {
      const uint4* _vptr_14 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_14) * 512 + d0_7);
      uint4 _vld_14[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_14[_blk] = _vptr_14[_blk];
        uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec[112 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) << 16);
          (&vec[112 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) & 0xffff0000u);
        }
      }
    }
    int _mn_15 = ((15) < (last_7) ? (15) : (last_7));
    {
      const uint4* _vptr_15 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_15) * 512 + d0_7);
      uint4 _vld_15[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_15[_blk] = _vptr_15[_blk];
        uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&vec[120 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) << 16);
          (&vec[120 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) & 0xffff0000u);
        }
      }
    }
    int merged = 0;
    float _shfl_sync_16 = __shfl_sync(0xFFFFFFFF, l0, 0);
    merged = _shfl_sync_16 == CUDART_INF_F;
    float m = max_noftz(l0, l1);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      m = max_noftz(m, __shfl_xor_sync(0xFFFFFFFF, m, offset));
    float w0 = 0.0f;
    float w1 = 0.0f;
    if (l0 > -CUDART_INF_F) {
      float _exp2_17 = approx_exp2(l0 - m);
      w0 = _exp2_17;
    }
    if (l1 > -CUDART_INF_F) {
      float _exp2_18 = approx_exp2(l1 - m);
      w1 = _exp2_18;
    }
    smem_w[w_base_7 + lane_7] = w0;
    smem_w[w_base_7 + 32 + lane_7] = w1;
    float sum_w = w0 + w1;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      sum_w += __shfl_xor_sync(0xFFFFFFFF, sum_w, offset);
    __syncwarp();
    float inv = 0.0f;
    if (sum_w > 0.0f) {
      float _rcp_19 = approx_rcp(sum_w);
      inv = _rcp_19;
    }
    float acc[8];
    for (int e = 0; e < 8; e++) {
      acc[e] = 0.0f;
    }
    float wts[16];
    wts[0] = smem_w[w_base_7 + 0];
    acc[0] = acc[0] + ((wts[0] > 0.0f) ? wts[0] * vec[0] : 0.0f);
    acc[1] = acc[1] + ((wts[0] > 0.0f) ? wts[0] * vec[1] : 0.0f);
    acc[2] = acc[2] + ((wts[0] > 0.0f) ? wts[0] * vec[2] : 0.0f);
    acc[3] = acc[3] + ((wts[0] > 0.0f) ? wts[0] * vec[3] : 0.0f);
    acc[4] = acc[4] + ((wts[0] > 0.0f) ? wts[0] * vec[4] : 0.0f);
    acc[5] = acc[5] + ((wts[0] > 0.0f) ? wts[0] * vec[5] : 0.0f);
    acc[6] = acc[6] + ((wts[0] > 0.0f) ? wts[0] * vec[6] : 0.0f);
    acc[7] = acc[7] + ((wts[0] > 0.0f) ? wts[0] * vec[7] : 0.0f);
    wts[1] = smem_w[w_base_7 + 1];
    acc[0] = acc[0] + ((wts[1] > 0.0f) ? wts[1] * vec[8] : 0.0f);
    acc[1] = acc[1] + ((wts[1] > 0.0f) ? wts[1] * vec[9] : 0.0f);
    acc[2] = acc[2] + ((wts[1] > 0.0f) ? wts[1] * vec[10] : 0.0f);
    acc[3] = acc[3] + ((wts[1] > 0.0f) ? wts[1] * vec[11] : 0.0f);
    acc[4] = acc[4] + ((wts[1] > 0.0f) ? wts[1] * vec[12] : 0.0f);
    acc[5] = acc[5] + ((wts[1] > 0.0f) ? wts[1] * vec[13] : 0.0f);
    acc[6] = acc[6] + ((wts[1] > 0.0f) ? wts[1] * vec[14] : 0.0f);
    acc[7] = acc[7] + ((wts[1] > 0.0f) ? wts[1] * vec[15] : 0.0f);
    wts[2] = smem_w[w_base_7 + 2];
    acc[0] = acc[0] + ((wts[2] > 0.0f) ? wts[2] * vec[16] : 0.0f);
    acc[1] = acc[1] + ((wts[2] > 0.0f) ? wts[2] * vec[17] : 0.0f);
    acc[2] = acc[2] + ((wts[2] > 0.0f) ? wts[2] * vec[18] : 0.0f);
    acc[3] = acc[3] + ((wts[2] > 0.0f) ? wts[2] * vec[19] : 0.0f);
    acc[4] = acc[4] + ((wts[2] > 0.0f) ? wts[2] * vec[20] : 0.0f);
    acc[5] = acc[5] + ((wts[2] > 0.0f) ? wts[2] * vec[21] : 0.0f);
    acc[6] = acc[6] + ((wts[2] > 0.0f) ? wts[2] * vec[22] : 0.0f);
    acc[7] = acc[7] + ((wts[2] > 0.0f) ? wts[2] * vec[23] : 0.0f);
    wts[3] = smem_w[w_base_7 + 3];
    acc[0] = acc[0] + ((wts[3] > 0.0f) ? wts[3] * vec[24] : 0.0f);
    acc[1] = acc[1] + ((wts[3] > 0.0f) ? wts[3] * vec[25] : 0.0f);
    acc[2] = acc[2] + ((wts[3] > 0.0f) ? wts[3] * vec[26] : 0.0f);
    acc[3] = acc[3] + ((wts[3] > 0.0f) ? wts[3] * vec[27] : 0.0f);
    acc[4] = acc[4] + ((wts[3] > 0.0f) ? wts[3] * vec[28] : 0.0f);
    acc[5] = acc[5] + ((wts[3] > 0.0f) ? wts[3] * vec[29] : 0.0f);
    acc[6] = acc[6] + ((wts[3] > 0.0f) ? wts[3] * vec[30] : 0.0f);
    acc[7] = acc[7] + ((wts[3] > 0.0f) ? wts[3] * vec[31] : 0.0f);
    wts[4] = smem_w[w_base_7 + 4];
    acc[0] = acc[0] + ((wts[4] > 0.0f) ? wts[4] * vec[32] : 0.0f);
    acc[1] = acc[1] + ((wts[4] > 0.0f) ? wts[4] * vec[33] : 0.0f);
    acc[2] = acc[2] + ((wts[4] > 0.0f) ? wts[4] * vec[34] : 0.0f);
    acc[3] = acc[3] + ((wts[4] > 0.0f) ? wts[4] * vec[35] : 0.0f);
    acc[4] = acc[4] + ((wts[4] > 0.0f) ? wts[4] * vec[36] : 0.0f);
    acc[5] = acc[5] + ((wts[4] > 0.0f) ? wts[4] * vec[37] : 0.0f);
    acc[6] = acc[6] + ((wts[4] > 0.0f) ? wts[4] * vec[38] : 0.0f);
    acc[7] = acc[7] + ((wts[4] > 0.0f) ? wts[4] * vec[39] : 0.0f);
    wts[5] = smem_w[w_base_7 + 5];
    acc[0] = acc[0] + ((wts[5] > 0.0f) ? wts[5] * vec[40] : 0.0f);
    acc[1] = acc[1] + ((wts[5] > 0.0f) ? wts[5] * vec[41] : 0.0f);
    acc[2] = acc[2] + ((wts[5] > 0.0f) ? wts[5] * vec[42] : 0.0f);
    acc[3] = acc[3] + ((wts[5] > 0.0f) ? wts[5] * vec[43] : 0.0f);
    acc[4] = acc[4] + ((wts[5] > 0.0f) ? wts[5] * vec[44] : 0.0f);
    acc[5] = acc[5] + ((wts[5] > 0.0f) ? wts[5] * vec[45] : 0.0f);
    acc[6] = acc[6] + ((wts[5] > 0.0f) ? wts[5] * vec[46] : 0.0f);
    acc[7] = acc[7] + ((wts[5] > 0.0f) ? wts[5] * vec[47] : 0.0f);
    wts[6] = smem_w[w_base_7 + 6];
    acc[0] = acc[0] + ((wts[6] > 0.0f) ? wts[6] * vec[48] : 0.0f);
    acc[1] = acc[1] + ((wts[6] > 0.0f) ? wts[6] * vec[49] : 0.0f);
    acc[2] = acc[2] + ((wts[6] > 0.0f) ? wts[6] * vec[50] : 0.0f);
    acc[3] = acc[3] + ((wts[6] > 0.0f) ? wts[6] * vec[51] : 0.0f);
    acc[4] = acc[4] + ((wts[6] > 0.0f) ? wts[6] * vec[52] : 0.0f);
    acc[5] = acc[5] + ((wts[6] > 0.0f) ? wts[6] * vec[53] : 0.0f);
    acc[6] = acc[6] + ((wts[6] > 0.0f) ? wts[6] * vec[54] : 0.0f);
    acc[7] = acc[7] + ((wts[6] > 0.0f) ? wts[6] * vec[55] : 0.0f);
    wts[7] = smem_w[w_base_7 + 7];
    acc[0] = acc[0] + ((wts[7] > 0.0f) ? wts[7] * vec[56] : 0.0f);
    acc[1] = acc[1] + ((wts[7] > 0.0f) ? wts[7] * vec[57] : 0.0f);
    acc[2] = acc[2] + ((wts[7] > 0.0f) ? wts[7] * vec[58] : 0.0f);
    acc[3] = acc[3] + ((wts[7] > 0.0f) ? wts[7] * vec[59] : 0.0f);
    acc[4] = acc[4] + ((wts[7] > 0.0f) ? wts[7] * vec[60] : 0.0f);
    acc[5] = acc[5] + ((wts[7] > 0.0f) ? wts[7] * vec[61] : 0.0f);
    acc[6] = acc[6] + ((wts[7] > 0.0f) ? wts[7] * vec[62] : 0.0f);
    acc[7] = acc[7] + ((wts[7] > 0.0f) ? wts[7] * vec[63] : 0.0f);
    wts[8] = smem_w[w_base_7 + 8];
    acc[0] = acc[0] + ((wts[8] > 0.0f) ? wts[8] * vec[64] : 0.0f);
    acc[1] = acc[1] + ((wts[8] > 0.0f) ? wts[8] * vec[65] : 0.0f);
    acc[2] = acc[2] + ((wts[8] > 0.0f) ? wts[8] * vec[66] : 0.0f);
    acc[3] = acc[3] + ((wts[8] > 0.0f) ? wts[8] * vec[67] : 0.0f);
    acc[4] = acc[4] + ((wts[8] > 0.0f) ? wts[8] * vec[68] : 0.0f);
    acc[5] = acc[5] + ((wts[8] > 0.0f) ? wts[8] * vec[69] : 0.0f);
    acc[6] = acc[6] + ((wts[8] > 0.0f) ? wts[8] * vec[70] : 0.0f);
    acc[7] = acc[7] + ((wts[8] > 0.0f) ? wts[8] * vec[71] : 0.0f);
    wts[9] = smem_w[w_base_7 + 9];
    acc[0] = acc[0] + ((wts[9] > 0.0f) ? wts[9] * vec[72] : 0.0f);
    acc[1] = acc[1] + ((wts[9] > 0.0f) ? wts[9] * vec[73] : 0.0f);
    acc[2] = acc[2] + ((wts[9] > 0.0f) ? wts[9] * vec[74] : 0.0f);
    acc[3] = acc[3] + ((wts[9] > 0.0f) ? wts[9] * vec[75] : 0.0f);
    acc[4] = acc[4] + ((wts[9] > 0.0f) ? wts[9] * vec[76] : 0.0f);
    acc[5] = acc[5] + ((wts[9] > 0.0f) ? wts[9] * vec[77] : 0.0f);
    acc[6] = acc[6] + ((wts[9] > 0.0f) ? wts[9] * vec[78] : 0.0f);
    acc[7] = acc[7] + ((wts[9] > 0.0f) ? wts[9] * vec[79] : 0.0f);
    wts[10] = smem_w[w_base_7 + 10];
    acc[0] = acc[0] + ((wts[10] > 0.0f) ? wts[10] * vec[80] : 0.0f);
    acc[1] = acc[1] + ((wts[10] > 0.0f) ? wts[10] * vec[81] : 0.0f);
    acc[2] = acc[2] + ((wts[10] > 0.0f) ? wts[10] * vec[82] : 0.0f);
    acc[3] = acc[3] + ((wts[10] > 0.0f) ? wts[10] * vec[83] : 0.0f);
    acc[4] = acc[4] + ((wts[10] > 0.0f) ? wts[10] * vec[84] : 0.0f);
    acc[5] = acc[5] + ((wts[10] > 0.0f) ? wts[10] * vec[85] : 0.0f);
    acc[6] = acc[6] + ((wts[10] > 0.0f) ? wts[10] * vec[86] : 0.0f);
    acc[7] = acc[7] + ((wts[10] > 0.0f) ? wts[10] * vec[87] : 0.0f);
    wts[11] = smem_w[w_base_7 + 11];
    acc[0] = acc[0] + ((wts[11] > 0.0f) ? wts[11] * vec[88] : 0.0f);
    acc[1] = acc[1] + ((wts[11] > 0.0f) ? wts[11] * vec[89] : 0.0f);
    acc[2] = acc[2] + ((wts[11] > 0.0f) ? wts[11] * vec[90] : 0.0f);
    acc[3] = acc[3] + ((wts[11] > 0.0f) ? wts[11] * vec[91] : 0.0f);
    acc[4] = acc[4] + ((wts[11] > 0.0f) ? wts[11] * vec[92] : 0.0f);
    acc[5] = acc[5] + ((wts[11] > 0.0f) ? wts[11] * vec[93] : 0.0f);
    acc[6] = acc[6] + ((wts[11] > 0.0f) ? wts[11] * vec[94] : 0.0f);
    acc[7] = acc[7] + ((wts[11] > 0.0f) ? wts[11] * vec[95] : 0.0f);
    wts[12] = smem_w[w_base_7 + 12];
    acc[0] = acc[0] + ((wts[12] > 0.0f) ? wts[12] * vec[96] : 0.0f);
    acc[1] = acc[1] + ((wts[12] > 0.0f) ? wts[12] * vec[97] : 0.0f);
    acc[2] = acc[2] + ((wts[12] > 0.0f) ? wts[12] * vec[98] : 0.0f);
    acc[3] = acc[3] + ((wts[12] > 0.0f) ? wts[12] * vec[99] : 0.0f);
    acc[4] = acc[4] + ((wts[12] > 0.0f) ? wts[12] * vec[100] : 0.0f);
    acc[5] = acc[5] + ((wts[12] > 0.0f) ? wts[12] * vec[101] : 0.0f);
    acc[6] = acc[6] + ((wts[12] > 0.0f) ? wts[12] * vec[102] : 0.0f);
    acc[7] = acc[7] + ((wts[12] > 0.0f) ? wts[12] * vec[103] : 0.0f);
    wts[13] = smem_w[w_base_7 + 13];
    acc[0] = acc[0] + ((wts[13] > 0.0f) ? wts[13] * vec[104] : 0.0f);
    acc[1] = acc[1] + ((wts[13] > 0.0f) ? wts[13] * vec[105] : 0.0f);
    acc[2] = acc[2] + ((wts[13] > 0.0f) ? wts[13] * vec[106] : 0.0f);
    acc[3] = acc[3] + ((wts[13] > 0.0f) ? wts[13] * vec[107] : 0.0f);
    acc[4] = acc[4] + ((wts[13] > 0.0f) ? wts[13] * vec[108] : 0.0f);
    acc[5] = acc[5] + ((wts[13] > 0.0f) ? wts[13] * vec[109] : 0.0f);
    acc[6] = acc[6] + ((wts[13] > 0.0f) ? wts[13] * vec[110] : 0.0f);
    acc[7] = acc[7] + ((wts[13] > 0.0f) ? wts[13] * vec[111] : 0.0f);
    wts[14] = smem_w[w_base_7 + 14];
    acc[0] = acc[0] + ((wts[14] > 0.0f) ? wts[14] * vec[112] : 0.0f);
    acc[1] = acc[1] + ((wts[14] > 0.0f) ? wts[14] * vec[113] : 0.0f);
    acc[2] = acc[2] + ((wts[14] > 0.0f) ? wts[14] * vec[114] : 0.0f);
    acc[3] = acc[3] + ((wts[14] > 0.0f) ? wts[14] * vec[115] : 0.0f);
    acc[4] = acc[4] + ((wts[14] > 0.0f) ? wts[14] * vec[116] : 0.0f);
    acc[5] = acc[5] + ((wts[14] > 0.0f) ? wts[14] * vec[117] : 0.0f);
    acc[6] = acc[6] + ((wts[14] > 0.0f) ? wts[14] * vec[118] : 0.0f);
    acc[7] = acc[7] + ((wts[14] > 0.0f) ? wts[14] * vec[119] : 0.0f);
    wts[15] = smem_w[w_base_7 + 15];
    acc[0] = acc[0] + ((wts[15] > 0.0f) ? wts[15] * vec[120] : 0.0f);
    acc[1] = acc[1] + ((wts[15] > 0.0f) ? wts[15] * vec[121] : 0.0f);
    acc[2] = acc[2] + ((wts[15] > 0.0f) ? wts[15] * vec[122] : 0.0f);
    acc[3] = acc[3] + ((wts[15] > 0.0f) ? wts[15] * vec[123] : 0.0f);
    acc[4] = acc[4] + ((wts[15] > 0.0f) ? wts[15] * vec[124] : 0.0f);
    acc[5] = acc[5] + ((wts[15] > 0.0f) ? wts[15] * vec[125] : 0.0f);
    acc[6] = acc[6] + ((wts[15] > 0.0f) ? wts[15] * vec[126] : 0.0f);
    acc[7] = acc[7] + ((wts[15] > 0.0f) ? wts[15] * vec[127] : 0.0f);
    int n_batches = (num_split + 16 - 1) / 16;
#pragma unroll 1
    for (int bt = 1; bt < n_batches; bt++) {
      int k0 = bt * 16;
      int _mn_16 = ((k0 + 0) < (last_7) ? (k0 + 0) : (last_7));
      {
        const uint4* _vptr_20 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_16) * 512 + d0_7);
        uint4 _vld_20[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_20[_blk] = _vptr_20[_blk];
          uint32_t* _vpairs_20 = reinterpret_cast<uint32_t*>(&_vld_20[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) << 16);
            (&vec[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) & 0xffff0000u);
          }
        }
      }
      int _mn_17 = ((k0 + 1) < (last_7) ? (k0 + 1) : (last_7));
      {
        const uint4* _vptr_21 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_17) * 512 + d0_7);
        uint4 _vld_21[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_21[_blk] = _vptr_21[_blk];
          uint32_t* _vpairs_21 = reinterpret_cast<uint32_t*>(&_vld_21[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec[8 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) << 16);
            (&vec[8 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) & 0xffff0000u);
          }
        }
      }
      int _mn_18 = ((k0 + 2) < (last_7) ? (k0 + 2) : (last_7));
      {
        const uint4* _vptr_22 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_18) * 512 + d0_7);
        uint4 _vld_22[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_22[_blk] = _vptr_22[_blk];
          uint32_t* _vpairs_22 = reinterpret_cast<uint32_t*>(&_vld_22[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec[16 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_22[_pair]) << 16);
            (&vec[16 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_22[_pair]) & 0xffff0000u);
          }
        }
      }
      int _mn_19 = ((k0 + 3) < (last_7) ? (k0 + 3) : (last_7));
      {
        const uint4* _vptr_23 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_19) * 512 + d0_7);
        uint4 _vld_23[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_23[_blk] = _vptr_23[_blk];
          uint32_t* _vpairs_23 = reinterpret_cast<uint32_t*>(&_vld_23[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec[24 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_23[_pair]) << 16);
            (&vec[24 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_23[_pair]) & 0xffff0000u);
          }
        }
      }
      int _mn_20 = ((k0 + 4) < (last_7) ? (k0 + 4) : (last_7));
      {
        const uint4* _vptr_24 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_20) * 512 + d0_7);
        uint4 _vld_24[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_24[_blk] = _vptr_24[_blk];
          uint32_t* _vpairs_24 = reinterpret_cast<uint32_t*>(&_vld_24[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec[32 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) << 16);
            (&vec[32 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) & 0xffff0000u);
          }
        }
      }
      int _mn_21 = ((k0 + 5) < (last_7) ? (k0 + 5) : (last_7));
      {
        const uint4* _vptr_25 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_21) * 512 + d0_7);
        uint4 _vld_25[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_25[_blk] = _vptr_25[_blk];
          uint32_t* _vpairs_25 = reinterpret_cast<uint32_t*>(&_vld_25[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec[40 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) << 16);
            (&vec[40 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) & 0xffff0000u);
          }
        }
      }
      int _mn_22 = ((k0 + 6) < (last_7) ? (k0 + 6) : (last_7));
      {
        const uint4* _vptr_26 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_22) * 512 + d0_7);
        uint4 _vld_26[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_26[_blk] = _vptr_26[_blk];
          uint32_t* _vpairs_26 = reinterpret_cast<uint32_t*>(&_vld_26[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec[48 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_26[_pair]) << 16);
            (&vec[48 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_26[_pair]) & 0xffff0000u);
          }
        }
      }
      int _mn_23 = ((k0 + 7) < (last_7) ? (k0 + 7) : (last_7));
      {
        const uint4* _vptr_27 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_23) * 512 + d0_7);
        uint4 _vld_27[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_27[_blk] = _vptr_27[_blk];
          uint32_t* _vpairs_27 = reinterpret_cast<uint32_t*>(&_vld_27[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec[56 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_27[_pair]) << 16);
            (&vec[56 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_27[_pair]) & 0xffff0000u);
          }
        }
      }
      int _mn_24 = ((k0 + 8) < (last_7) ? (k0 + 8) : (last_7));
      {
        const uint4* _vptr_28 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_24) * 512 + d0_7);
        uint4 _vld_28[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_28[_blk] = _vptr_28[_blk];
          uint32_t* _vpairs_28 = reinterpret_cast<uint32_t*>(&_vld_28[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec[64 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_28[_pair]) << 16);
            (&vec[64 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_28[_pair]) & 0xffff0000u);
          }
        }
      }
      int _mn_25 = ((k0 + 9) < (last_7) ? (k0 + 9) : (last_7));
      {
        const uint4* _vptr_29 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_25) * 512 + d0_7);
        uint4 _vld_29[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_29[_blk] = _vptr_29[_blk];
          uint32_t* _vpairs_29 = reinterpret_cast<uint32_t*>(&_vld_29[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec[72 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_29[_pair]) << 16);
            (&vec[72 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_29[_pair]) & 0xffff0000u);
          }
        }
      }
      int _mn_26 = ((k0 + 10) < (last_7) ? (k0 + 10) : (last_7));
      {
        const uint4* _vptr_30 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_26) * 512 + d0_7);
        uint4 _vld_30[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_30[_blk] = _vptr_30[_blk];
          uint32_t* _vpairs_30 = reinterpret_cast<uint32_t*>(&_vld_30[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec[80 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) << 16);
            (&vec[80 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) & 0xffff0000u);
          }
        }
      }
      int _mn_27 = ((k0 + 11) < (last_7) ? (k0 + 11) : (last_7));
      {
        const uint4* _vptr_31 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_27) * 512 + d0_7);
        uint4 _vld_31[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_31[_blk] = _vptr_31[_blk];
          uint32_t* _vpairs_31 = reinterpret_cast<uint32_t*>(&_vld_31[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec[88 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) << 16);
            (&vec[88 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) & 0xffff0000u);
          }
        }
      }
      int _mn_28 = ((k0 + 12) < (last_7) ? (k0 + 12) : (last_7));
      {
        const uint4* _vptr_32 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_28) * 512 + d0_7);
        uint4 _vld_32[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_32[_blk] = _vptr_32[_blk];
          uint32_t* _vpairs_32 = reinterpret_cast<uint32_t*>(&_vld_32[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec[96 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) << 16);
            (&vec[96 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) & 0xffff0000u);
          }
        }
      }
      int _mn_29 = ((k0 + 13) < (last_7) ? (k0 + 13) : (last_7));
      {
        const uint4* _vptr_33 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_29) * 512 + d0_7);
        uint4 _vld_33[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_33[_blk] = _vptr_33[_blk];
          uint32_t* _vpairs_33 = reinterpret_cast<uint32_t*>(&_vld_33[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec[104 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) << 16);
            (&vec[104 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) & 0xffff0000u);
          }
        }
      }
      int _mn_30 = ((k0 + 14) < (last_7) ? (k0 + 14) : (last_7));
      {
        const uint4* _vptr_34 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_30) * 512 + d0_7);
        uint4 _vld_34[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_34[_blk] = _vptr_34[_blk];
          uint32_t* _vpairs_34 = reinterpret_cast<uint32_t*>(&_vld_34[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec[112 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) << 16);
            (&vec[112 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) & 0xffff0000u);
          }
        }
      }
      int _mn_31 = ((k0 + 15) < (last_7) ? (k0 + 15) : (last_7));
      {
        const uint4* _vptr_35 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base_7 + _mn_31) * 512 + d0_7);
        uint4 _vld_35[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_35[_blk] = _vptr_35[_blk];
          uint32_t* _vpairs_35 = reinterpret_cast<uint32_t*>(&_vld_35[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&vec[120 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_35[_pair]) << 16);
            (&vec[120 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_35[_pair]) & 0xffff0000u);
          }
        }
      }
      wts[0] = smem_w[w_base_7 + (k0 + 0)];
      acc[0] = acc[0] + ((wts[0] > 0.0f) ? wts[0] * vec[0] : 0.0f);
      acc[1] = acc[1] + ((wts[0] > 0.0f) ? wts[0] * vec[1] : 0.0f);
      acc[2] = acc[2] + ((wts[0] > 0.0f) ? wts[0] * vec[2] : 0.0f);
      acc[3] = acc[3] + ((wts[0] > 0.0f) ? wts[0] * vec[3] : 0.0f);
      acc[4] = acc[4] + ((wts[0] > 0.0f) ? wts[0] * vec[4] : 0.0f);
      acc[5] = acc[5] + ((wts[0] > 0.0f) ? wts[0] * vec[5] : 0.0f);
      acc[6] = acc[6] + ((wts[0] > 0.0f) ? wts[0] * vec[6] : 0.0f);
      acc[7] = acc[7] + ((wts[0] > 0.0f) ? wts[0] * vec[7] : 0.0f);
      wts[1] = smem_w[w_base_7 + (k0 + 1)];
      acc[0] = acc[0] + ((wts[1] > 0.0f) ? wts[1] * vec[8] : 0.0f);
      acc[1] = acc[1] + ((wts[1] > 0.0f) ? wts[1] * vec[9] : 0.0f);
      acc[2] = acc[2] + ((wts[1] > 0.0f) ? wts[1] * vec[10] : 0.0f);
      acc[3] = acc[3] + ((wts[1] > 0.0f) ? wts[1] * vec[11] : 0.0f);
      acc[4] = acc[4] + ((wts[1] > 0.0f) ? wts[1] * vec[12] : 0.0f);
      acc[5] = acc[5] + ((wts[1] > 0.0f) ? wts[1] * vec[13] : 0.0f);
      acc[6] = acc[6] + ((wts[1] > 0.0f) ? wts[1] * vec[14] : 0.0f);
      acc[7] = acc[7] + ((wts[1] > 0.0f) ? wts[1] * vec[15] : 0.0f);
      wts[2] = smem_w[w_base_7 + (k0 + 2)];
      acc[0] = acc[0] + ((wts[2] > 0.0f) ? wts[2] * vec[16] : 0.0f);
      acc[1] = acc[1] + ((wts[2] > 0.0f) ? wts[2] * vec[17] : 0.0f);
      acc[2] = acc[2] + ((wts[2] > 0.0f) ? wts[2] * vec[18] : 0.0f);
      acc[3] = acc[3] + ((wts[2] > 0.0f) ? wts[2] * vec[19] : 0.0f);
      acc[4] = acc[4] + ((wts[2] > 0.0f) ? wts[2] * vec[20] : 0.0f);
      acc[5] = acc[5] + ((wts[2] > 0.0f) ? wts[2] * vec[21] : 0.0f);
      acc[6] = acc[6] + ((wts[2] > 0.0f) ? wts[2] * vec[22] : 0.0f);
      acc[7] = acc[7] + ((wts[2] > 0.0f) ? wts[2] * vec[23] : 0.0f);
      wts[3] = smem_w[w_base_7 + (k0 + 3)];
      acc[0] = acc[0] + ((wts[3] > 0.0f) ? wts[3] * vec[24] : 0.0f);
      acc[1] = acc[1] + ((wts[3] > 0.0f) ? wts[3] * vec[25] : 0.0f);
      acc[2] = acc[2] + ((wts[3] > 0.0f) ? wts[3] * vec[26] : 0.0f);
      acc[3] = acc[3] + ((wts[3] > 0.0f) ? wts[3] * vec[27] : 0.0f);
      acc[4] = acc[4] + ((wts[3] > 0.0f) ? wts[3] * vec[28] : 0.0f);
      acc[5] = acc[5] + ((wts[3] > 0.0f) ? wts[3] * vec[29] : 0.0f);
      acc[6] = acc[6] + ((wts[3] > 0.0f) ? wts[3] * vec[30] : 0.0f);
      acc[7] = acc[7] + ((wts[3] > 0.0f) ? wts[3] * vec[31] : 0.0f);
      wts[4] = smem_w[w_base_7 + (k0 + 4)];
      acc[0] = acc[0] + ((wts[4] > 0.0f) ? wts[4] * vec[32] : 0.0f);
      acc[1] = acc[1] + ((wts[4] > 0.0f) ? wts[4] * vec[33] : 0.0f);
      acc[2] = acc[2] + ((wts[4] > 0.0f) ? wts[4] * vec[34] : 0.0f);
      acc[3] = acc[3] + ((wts[4] > 0.0f) ? wts[4] * vec[35] : 0.0f);
      acc[4] = acc[4] + ((wts[4] > 0.0f) ? wts[4] * vec[36] : 0.0f);
      acc[5] = acc[5] + ((wts[4] > 0.0f) ? wts[4] * vec[37] : 0.0f);
      acc[6] = acc[6] + ((wts[4] > 0.0f) ? wts[4] * vec[38] : 0.0f);
      acc[7] = acc[7] + ((wts[4] > 0.0f) ? wts[4] * vec[39] : 0.0f);
      wts[5] = smem_w[w_base_7 + (k0 + 5)];
      acc[0] = acc[0] + ((wts[5] > 0.0f) ? wts[5] * vec[40] : 0.0f);
      acc[1] = acc[1] + ((wts[5] > 0.0f) ? wts[5] * vec[41] : 0.0f);
      acc[2] = acc[2] + ((wts[5] > 0.0f) ? wts[5] * vec[42] : 0.0f);
      acc[3] = acc[3] + ((wts[5] > 0.0f) ? wts[5] * vec[43] : 0.0f);
      acc[4] = acc[4] + ((wts[5] > 0.0f) ? wts[5] * vec[44] : 0.0f);
      acc[5] = acc[5] + ((wts[5] > 0.0f) ? wts[5] * vec[45] : 0.0f);
      acc[6] = acc[6] + ((wts[5] > 0.0f) ? wts[5] * vec[46] : 0.0f);
      acc[7] = acc[7] + ((wts[5] > 0.0f) ? wts[5] * vec[47] : 0.0f);
      wts[6] = smem_w[w_base_7 + (k0 + 6)];
      acc[0] = acc[0] + ((wts[6] > 0.0f) ? wts[6] * vec[48] : 0.0f);
      acc[1] = acc[1] + ((wts[6] > 0.0f) ? wts[6] * vec[49] : 0.0f);
      acc[2] = acc[2] + ((wts[6] > 0.0f) ? wts[6] * vec[50] : 0.0f);
      acc[3] = acc[3] + ((wts[6] > 0.0f) ? wts[6] * vec[51] : 0.0f);
      acc[4] = acc[4] + ((wts[6] > 0.0f) ? wts[6] * vec[52] : 0.0f);
      acc[5] = acc[5] + ((wts[6] > 0.0f) ? wts[6] * vec[53] : 0.0f);
      acc[6] = acc[6] + ((wts[6] > 0.0f) ? wts[6] * vec[54] : 0.0f);
      acc[7] = acc[7] + ((wts[6] > 0.0f) ? wts[6] * vec[55] : 0.0f);
      wts[7] = smem_w[w_base_7 + (k0 + 7)];
      acc[0] = acc[0] + ((wts[7] > 0.0f) ? wts[7] * vec[56] : 0.0f);
      acc[1] = acc[1] + ((wts[7] > 0.0f) ? wts[7] * vec[57] : 0.0f);
      acc[2] = acc[2] + ((wts[7] > 0.0f) ? wts[7] * vec[58] : 0.0f);
      acc[3] = acc[3] + ((wts[7] > 0.0f) ? wts[7] * vec[59] : 0.0f);
      acc[4] = acc[4] + ((wts[7] > 0.0f) ? wts[7] * vec[60] : 0.0f);
      acc[5] = acc[5] + ((wts[7] > 0.0f) ? wts[7] * vec[61] : 0.0f);
      acc[6] = acc[6] + ((wts[7] > 0.0f) ? wts[7] * vec[62] : 0.0f);
      acc[7] = acc[7] + ((wts[7] > 0.0f) ? wts[7] * vec[63] : 0.0f);
      wts[8] = smem_w[w_base_7 + (k0 + 8)];
      acc[0] = acc[0] + ((wts[8] > 0.0f) ? wts[8] * vec[64] : 0.0f);
      acc[1] = acc[1] + ((wts[8] > 0.0f) ? wts[8] * vec[65] : 0.0f);
      acc[2] = acc[2] + ((wts[8] > 0.0f) ? wts[8] * vec[66] : 0.0f);
      acc[3] = acc[3] + ((wts[8] > 0.0f) ? wts[8] * vec[67] : 0.0f);
      acc[4] = acc[4] + ((wts[8] > 0.0f) ? wts[8] * vec[68] : 0.0f);
      acc[5] = acc[5] + ((wts[8] > 0.0f) ? wts[8] * vec[69] : 0.0f);
      acc[6] = acc[6] + ((wts[8] > 0.0f) ? wts[8] * vec[70] : 0.0f);
      acc[7] = acc[7] + ((wts[8] > 0.0f) ? wts[8] * vec[71] : 0.0f);
      wts[9] = smem_w[w_base_7 + (k0 + 9)];
      acc[0] = acc[0] + ((wts[9] > 0.0f) ? wts[9] * vec[72] : 0.0f);
      acc[1] = acc[1] + ((wts[9] > 0.0f) ? wts[9] * vec[73] : 0.0f);
      acc[2] = acc[2] + ((wts[9] > 0.0f) ? wts[9] * vec[74] : 0.0f);
      acc[3] = acc[3] + ((wts[9] > 0.0f) ? wts[9] * vec[75] : 0.0f);
      acc[4] = acc[4] + ((wts[9] > 0.0f) ? wts[9] * vec[76] : 0.0f);
      acc[5] = acc[5] + ((wts[9] > 0.0f) ? wts[9] * vec[77] : 0.0f);
      acc[6] = acc[6] + ((wts[9] > 0.0f) ? wts[9] * vec[78] : 0.0f);
      acc[7] = acc[7] + ((wts[9] > 0.0f) ? wts[9] * vec[79] : 0.0f);
      wts[10] = smem_w[w_base_7 + (k0 + 10)];
      acc[0] = acc[0] + ((wts[10] > 0.0f) ? wts[10] * vec[80] : 0.0f);
      acc[1] = acc[1] + ((wts[10] > 0.0f) ? wts[10] * vec[81] : 0.0f);
      acc[2] = acc[2] + ((wts[10] > 0.0f) ? wts[10] * vec[82] : 0.0f);
      acc[3] = acc[3] + ((wts[10] > 0.0f) ? wts[10] * vec[83] : 0.0f);
      acc[4] = acc[4] + ((wts[10] > 0.0f) ? wts[10] * vec[84] : 0.0f);
      acc[5] = acc[5] + ((wts[10] > 0.0f) ? wts[10] * vec[85] : 0.0f);
      acc[6] = acc[6] + ((wts[10] > 0.0f) ? wts[10] * vec[86] : 0.0f);
      acc[7] = acc[7] + ((wts[10] > 0.0f) ? wts[10] * vec[87] : 0.0f);
      wts[11] = smem_w[w_base_7 + (k0 + 11)];
      acc[0] = acc[0] + ((wts[11] > 0.0f) ? wts[11] * vec[88] : 0.0f);
      acc[1] = acc[1] + ((wts[11] > 0.0f) ? wts[11] * vec[89] : 0.0f);
      acc[2] = acc[2] + ((wts[11] > 0.0f) ? wts[11] * vec[90] : 0.0f);
      acc[3] = acc[3] + ((wts[11] > 0.0f) ? wts[11] * vec[91] : 0.0f);
      acc[4] = acc[4] + ((wts[11] > 0.0f) ? wts[11] * vec[92] : 0.0f);
      acc[5] = acc[5] + ((wts[11] > 0.0f) ? wts[11] * vec[93] : 0.0f);
      acc[6] = acc[6] + ((wts[11] > 0.0f) ? wts[11] * vec[94] : 0.0f);
      acc[7] = acc[7] + ((wts[11] > 0.0f) ? wts[11] * vec[95] : 0.0f);
      wts[12] = smem_w[w_base_7 + (k0 + 12)];
      acc[0] = acc[0] + ((wts[12] > 0.0f) ? wts[12] * vec[96] : 0.0f);
      acc[1] = acc[1] + ((wts[12] > 0.0f) ? wts[12] * vec[97] : 0.0f);
      acc[2] = acc[2] + ((wts[12] > 0.0f) ? wts[12] * vec[98] : 0.0f);
      acc[3] = acc[3] + ((wts[12] > 0.0f) ? wts[12] * vec[99] : 0.0f);
      acc[4] = acc[4] + ((wts[12] > 0.0f) ? wts[12] * vec[100] : 0.0f);
      acc[5] = acc[5] + ((wts[12] > 0.0f) ? wts[12] * vec[101] : 0.0f);
      acc[6] = acc[6] + ((wts[12] > 0.0f) ? wts[12] * vec[102] : 0.0f);
      acc[7] = acc[7] + ((wts[12] > 0.0f) ? wts[12] * vec[103] : 0.0f);
      wts[13] = smem_w[w_base_7 + (k0 + 13)];
      acc[0] = acc[0] + ((wts[13] > 0.0f) ? wts[13] * vec[104] : 0.0f);
      acc[1] = acc[1] + ((wts[13] > 0.0f) ? wts[13] * vec[105] : 0.0f);
      acc[2] = acc[2] + ((wts[13] > 0.0f) ? wts[13] * vec[106] : 0.0f);
      acc[3] = acc[3] + ((wts[13] > 0.0f) ? wts[13] * vec[107] : 0.0f);
      acc[4] = acc[4] + ((wts[13] > 0.0f) ? wts[13] * vec[108] : 0.0f);
      acc[5] = acc[5] + ((wts[13] > 0.0f) ? wts[13] * vec[109] : 0.0f);
      acc[6] = acc[6] + ((wts[13] > 0.0f) ? wts[13] * vec[110] : 0.0f);
      acc[7] = acc[7] + ((wts[13] > 0.0f) ? wts[13] * vec[111] : 0.0f);
      wts[14] = smem_w[w_base_7 + (k0 + 14)];
      acc[0] = acc[0] + ((wts[14] > 0.0f) ? wts[14] * vec[112] : 0.0f);
      acc[1] = acc[1] + ((wts[14] > 0.0f) ? wts[14] * vec[113] : 0.0f);
      acc[2] = acc[2] + ((wts[14] > 0.0f) ? wts[14] * vec[114] : 0.0f);
      acc[3] = acc[3] + ((wts[14] > 0.0f) ? wts[14] * vec[115] : 0.0f);
      acc[4] = acc[4] + ((wts[14] > 0.0f) ? wts[14] * vec[116] : 0.0f);
      acc[5] = acc[5] + ((wts[14] > 0.0f) ? wts[14] * vec[117] : 0.0f);
      acc[6] = acc[6] + ((wts[14] > 0.0f) ? wts[14] * vec[118] : 0.0f);
      acc[7] = acc[7] + ((wts[14] > 0.0f) ? wts[14] * vec[119] : 0.0f);
      wts[15] = smem_w[w_base_7 + (k0 + 15)];
      acc[0] = acc[0] + ((wts[15] > 0.0f) ? wts[15] * vec[120] : 0.0f);
      acc[1] = acc[1] + ((wts[15] > 0.0f) ? wts[15] * vec[121] : 0.0f);
      acc[2] = acc[2] + ((wts[15] > 0.0f) ? wts[15] * vec[122] : 0.0f);
      acc[3] = acc[3] + ((wts[15] > 0.0f) ? wts[15] * vec[123] : 0.0f);
      acc[4] = acc[4] + ((wts[15] > 0.0f) ? wts[15] * vec[124] : 0.0f);
      acc[5] = acc[5] + ((wts[15] > 0.0f) ? wts[15] * vec[125] : 0.0f);
      acc[6] = acc[6] + ((wts[15] > 0.0f) ? wts[15] * vec[126] : 0.0f);
      acc[7] = acc[7] + ((wts[15] > 0.0f) ? wts[15] * vec[127] : 0.0f);
    }
    for (int e_1 = 0; e_1 < 8; e_1++) {
      acc[e_1] = acc[e_1] * inv;
    }
    unsigned int _packed[4];
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
#pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
      __nv_bfloat162 _bf2 =
          __float22bfloat162_rn(make_float2(acc[_lp * 2 + 4], acc[_lp * 2 + 1 + 4]));
      _packed[_lp + 2] = *(uint32_t*)&_bf2;
    }
#pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
      __nv_bfloat162 _bf2 =
          __float22bfloat162_rn(make_float2(acc[_lp * 2 + 6], acc[_lp * 2 + 1 + 6]));
      _packed[_lp + 3] = *(uint32_t*)&_bf2;
    }
    if (merged == 0) {
      reinterpret_cast<int4*>(O + (row_7 * 512 + d0_7))[0] = reinterpret_cast<int4*>(_packed)[0];
      if (has_lse != 0) {
        if (part_7 == 0) {
          if (lane_7 == 0) {
            float lse_v = -CUDART_INF_F;
            if (sum_w > 0.0f) {
              float _log2_36;
              asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_36) : "f"(sum_w));
              lse_v = (m + _log2_36) * 0.6931472f;
            }
            *(reinterpret_cast<float*>(lse + row_7) + (0)) = lse_v;
          }
        }
      }
    }
  }
}

}  // extern "C"
