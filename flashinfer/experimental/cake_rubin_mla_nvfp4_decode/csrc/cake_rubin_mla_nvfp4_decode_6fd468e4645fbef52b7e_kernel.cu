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
__launch_bounds__(256, LAUNCH_MIN_BLOCKS) void kernel_cake_rubin_mla_nvfp4_decode_6fd468e4645fbef52b7e(
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

  // === Task calls (dependency order) ===
  asm volatile("griddepcontrol.wait;" ::: "memory");
  int part = warp % 2;
  int row = blockIdx.x * 4 + warp / 2;
  int merged = 0;
  if (row < rows) {
    merged = partial_lse[row * num_split] == CUDART_INF_F;
  }
  if (row < rows && merged == 0) {
    int lane_0 = lane;
    int stat_base = row * num_split;
    int w_base = warp * 64;
    int d0 = part * 256 + lane_0 * 8;
    int last = num_split - 1;
    int _min_0 = ((0) < (last) ? (0) : (last));
    float _vec_load_0[8];
    {
      const uint4* _vptr_0 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_0) * 512 + d0);
      uint4 _vld_0[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_0[_blk] = _vptr_0[_blk];
        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
          (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
        }
      }
    }
    int _min_1 = ((1) < (last) ? (1) : (last));
    float _vec_load_1[8];
    {
      const uint4* _vptr_1 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_1) * 512 + d0);
      uint4 _vld_1[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_1[_blk] = _vptr_1[_blk];
        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
          (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
        }
      }
    }
    int _min_2 = ((2) < (last) ? (2) : (last));
    float _vec_load_2[8];
    {
      const uint4* _vptr_2 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_2) * 512 + d0);
      uint4 _vld_2[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_2[_blk] = _vptr_2[_blk];
        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
          (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
        }
      }
    }
    int _min_3 = ((3) < (last) ? (3) : (last));
    float _vec_load_3[8];
    {
      const uint4* _vptr_3 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_3) * 512 + d0);
      uint4 _vld_3[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_3[_blk] = _vptr_3[_blk];
        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
          (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
        }
      }
    }
    int _min_4 = ((4) < (last) ? (4) : (last));
    float _vec_load_4[8];
    {
      const uint4* _vptr_4 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_4) * 512 + d0);
      uint4 _vld_4[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_4[_blk] = _vptr_4[_blk];
        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
          (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
        }
      }
    }
    int _min_5 = ((5) < (last) ? (5) : (last));
    float _vec_load_5[8];
    {
      const uint4* _vptr_5 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_5) * 512 + d0);
      uint4 _vld_5[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_5[_blk] = _vptr_5[_blk];
        uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&_vec_load_5[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
          (&_vec_load_5[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
        }
      }
    }
    int _min_6 = ((6) < (last) ? (6) : (last));
    float _vec_load_6[8];
    {
      const uint4* _vptr_6 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_6) * 512 + d0);
      uint4 _vld_6[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_6[_blk] = _vptr_6[_blk];
        uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&_vec_load_6[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
          (&_vec_load_6[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
        }
      }
    }
    int _min_7 = ((7) < (last) ? (7) : (last));
    float _vec_load_7[8];
    {
      const uint4* _vptr_7 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_7) * 512 + d0);
      uint4 _vld_7[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_7[_blk] = _vptr_7[_blk];
        uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&_vec_load_7[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
          (&_vec_load_7[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
        }
      }
    }
    int _min_8 = ((8) < (last) ? (8) : (last));
    float _vec_load_8[8];
    {
      const uint4* _vptr_8 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_8) * 512 + d0);
      uint4 _vld_8[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_8[_blk] = _vptr_8[_blk];
        uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&_vec_load_8[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
          (&_vec_load_8[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
        }
      }
    }
    int _min_9 = ((9) < (last) ? (9) : (last));
    float _vec_load_9[8];
    {
      const uint4* _vptr_9 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_9) * 512 + d0);
      uint4 _vld_9[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_9[_blk] = _vptr_9[_blk];
        uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&_vec_load_9[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
          (&_vec_load_9[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
        }
      }
    }
    int _min_10 = ((10) < (last) ? (10) : (last));
    float _vec_load_10[8];
    {
      const uint4* _vptr_10 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_10) * 512 + d0);
      uint4 _vld_10[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_10[_blk] = _vptr_10[_blk];
        uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&_vec_load_10[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
          (&_vec_load_10[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
        }
      }
    }
    int _min_11 = ((11) < (last) ? (11) : (last));
    float _vec_load_11[8];
    {
      const uint4* _vptr_11 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_11) * 512 + d0);
      uint4 _vld_11[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_11[_blk] = _vptr_11[_blk];
        uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&_vec_load_11[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) << 16);
          (&_vec_load_11[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) & 0xffff0000u);
        }
      }
    }
    int _min_12 = ((12) < (last) ? (12) : (last));
    float _vec_load_12[8];
    {
      const uint4* _vptr_12 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_12) * 512 + d0);
      uint4 _vld_12[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_12[_blk] = _vptr_12[_blk];
        uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&_vec_load_12[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
          (&_vec_load_12[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
        }
      }
    }
    int _min_13 = ((13) < (last) ? (13) : (last));
    float _vec_load_13[8];
    {
      const uint4* _vptr_13 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_13) * 512 + d0);
      uint4 _vld_13[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_13[_blk] = _vptr_13[_blk];
        uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&_vec_load_13[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) << 16);
          (&_vec_load_13[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) & 0xffff0000u);
        }
      }
    }
    int _min_14 = ((14) < (last) ? (14) : (last));
    float _vec_load_14[8];
    {
      const uint4* _vptr_14 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_14) * 512 + d0);
      uint4 _vld_14[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_14[_blk] = _vptr_14[_blk];
        uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&_vec_load_14[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) << 16);
          (&_vec_load_14[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) & 0xffff0000u);
        }
      }
    }
    int _min_15 = ((15) < (last) ? (15) : (last));
    float _vec_load_15[8];
    {
      const uint4* _vptr_15 =
          reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_15) * 512 + d0);
      uint4 _vld_15[1];
#pragma unroll
      for (int _blk = 0; _blk < 1; _blk++) {
        _vld_15[_blk] = _vptr_15[_blk];
        uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
#pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
          (&_vec_load_15[0 + _blk * 8 + _pair * 2])[0] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) << 16);
          (&_vec_load_15[0 + _blk * 8 + _pair * 2])[1] =
              __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) & 0xffff0000u);
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
      float _exp2_16 = approx_exp2(l0 - m);
      w0 = _exp2_16;
    }
    if (l1 > -CUDART_INF_F) {
      float _exp2_17 = approx_exp2(l1 - m);
      w1 = _exp2_17;
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
      float _rcp_18 = approx_rcp(sum_w);
      inv = _rcp_18;
    }
    float acc[8];
    for (int e = 0; e < 8; e++) {
      acc[e] = 0.0f;
    }
    float wts[16];
    wts[0] = smem_w[w_base];
    acc[0] = acc[0] + ((wts[0] > 0.0f) ? wts[0] * _vec_load_0[0] : 0.0f);
    acc[1] = acc[1] + ((wts[0] > 0.0f) ? wts[0] * _vec_load_0[1] : 0.0f);
    acc[2] = acc[2] + ((wts[0] > 0.0f) ? wts[0] * _vec_load_0[2] : 0.0f);
    acc[3] = acc[3] + ((wts[0] > 0.0f) ? wts[0] * _vec_load_0[3] : 0.0f);
    acc[4] = acc[4] + ((wts[0] > 0.0f) ? wts[0] * _vec_load_0[4] : 0.0f);
    acc[5] = acc[5] + ((wts[0] > 0.0f) ? wts[0] * _vec_load_0[5] : 0.0f);
    acc[6] = acc[6] + ((wts[0] > 0.0f) ? wts[0] * _vec_load_0[6] : 0.0f);
    acc[7] = acc[7] + ((wts[0] > 0.0f) ? wts[0] * _vec_load_0[7] : 0.0f);
    wts[1] = smem_w[w_base + 1];
    acc[0] = acc[0] + ((wts[1] > 0.0f) ? wts[1] * _vec_load_1[0] : 0.0f);
    acc[1] = acc[1] + ((wts[1] > 0.0f) ? wts[1] * _vec_load_1[1] : 0.0f);
    acc[2] = acc[2] + ((wts[1] > 0.0f) ? wts[1] * _vec_load_1[2] : 0.0f);
    acc[3] = acc[3] + ((wts[1] > 0.0f) ? wts[1] * _vec_load_1[3] : 0.0f);
    acc[4] = acc[4] + ((wts[1] > 0.0f) ? wts[1] * _vec_load_1[4] : 0.0f);
    acc[5] = acc[5] + ((wts[1] > 0.0f) ? wts[1] * _vec_load_1[5] : 0.0f);
    acc[6] = acc[6] + ((wts[1] > 0.0f) ? wts[1] * _vec_load_1[6] : 0.0f);
    acc[7] = acc[7] + ((wts[1] > 0.0f) ? wts[1] * _vec_load_1[7] : 0.0f);
    wts[2] = smem_w[w_base + 2];
    acc[0] = acc[0] + ((wts[2] > 0.0f) ? wts[2] * _vec_load_2[0] : 0.0f);
    acc[1] = acc[1] + ((wts[2] > 0.0f) ? wts[2] * _vec_load_2[1] : 0.0f);
    acc[2] = acc[2] + ((wts[2] > 0.0f) ? wts[2] * _vec_load_2[2] : 0.0f);
    acc[3] = acc[3] + ((wts[2] > 0.0f) ? wts[2] * _vec_load_2[3] : 0.0f);
    acc[4] = acc[4] + ((wts[2] > 0.0f) ? wts[2] * _vec_load_2[4] : 0.0f);
    acc[5] = acc[5] + ((wts[2] > 0.0f) ? wts[2] * _vec_load_2[5] : 0.0f);
    acc[6] = acc[6] + ((wts[2] > 0.0f) ? wts[2] * _vec_load_2[6] : 0.0f);
    acc[7] = acc[7] + ((wts[2] > 0.0f) ? wts[2] * _vec_load_2[7] : 0.0f);
    wts[3] = smem_w[w_base + 3];
    acc[0] = acc[0] + ((wts[3] > 0.0f) ? wts[3] * _vec_load_3[0] : 0.0f);
    acc[1] = acc[1] + ((wts[3] > 0.0f) ? wts[3] * _vec_load_3[1] : 0.0f);
    acc[2] = acc[2] + ((wts[3] > 0.0f) ? wts[3] * _vec_load_3[2] : 0.0f);
    acc[3] = acc[3] + ((wts[3] > 0.0f) ? wts[3] * _vec_load_3[3] : 0.0f);
    acc[4] = acc[4] + ((wts[3] > 0.0f) ? wts[3] * _vec_load_3[4] : 0.0f);
    acc[5] = acc[5] + ((wts[3] > 0.0f) ? wts[3] * _vec_load_3[5] : 0.0f);
    acc[6] = acc[6] + ((wts[3] > 0.0f) ? wts[3] * _vec_load_3[6] : 0.0f);
    acc[7] = acc[7] + ((wts[3] > 0.0f) ? wts[3] * _vec_load_3[7] : 0.0f);
    wts[4] = smem_w[w_base + 4];
    acc[0] = acc[0] + ((wts[4] > 0.0f) ? wts[4] * _vec_load_4[0] : 0.0f);
    acc[1] = acc[1] + ((wts[4] > 0.0f) ? wts[4] * _vec_load_4[1] : 0.0f);
    acc[2] = acc[2] + ((wts[4] > 0.0f) ? wts[4] * _vec_load_4[2] : 0.0f);
    acc[3] = acc[3] + ((wts[4] > 0.0f) ? wts[4] * _vec_load_4[3] : 0.0f);
    acc[4] = acc[4] + ((wts[4] > 0.0f) ? wts[4] * _vec_load_4[4] : 0.0f);
    acc[5] = acc[5] + ((wts[4] > 0.0f) ? wts[4] * _vec_load_4[5] : 0.0f);
    acc[6] = acc[6] + ((wts[4] > 0.0f) ? wts[4] * _vec_load_4[6] : 0.0f);
    acc[7] = acc[7] + ((wts[4] > 0.0f) ? wts[4] * _vec_load_4[7] : 0.0f);
    wts[5] = smem_w[w_base + 5];
    acc[0] = acc[0] + ((wts[5] > 0.0f) ? wts[5] * _vec_load_5[0] : 0.0f);
    acc[1] = acc[1] + ((wts[5] > 0.0f) ? wts[5] * _vec_load_5[1] : 0.0f);
    acc[2] = acc[2] + ((wts[5] > 0.0f) ? wts[5] * _vec_load_5[2] : 0.0f);
    acc[3] = acc[3] + ((wts[5] > 0.0f) ? wts[5] * _vec_load_5[3] : 0.0f);
    acc[4] = acc[4] + ((wts[5] > 0.0f) ? wts[5] * _vec_load_5[4] : 0.0f);
    acc[5] = acc[5] + ((wts[5] > 0.0f) ? wts[5] * _vec_load_5[5] : 0.0f);
    acc[6] = acc[6] + ((wts[5] > 0.0f) ? wts[5] * _vec_load_5[6] : 0.0f);
    acc[7] = acc[7] + ((wts[5] > 0.0f) ? wts[5] * _vec_load_5[7] : 0.0f);
    wts[6] = smem_w[w_base + 6];
    acc[0] = acc[0] + ((wts[6] > 0.0f) ? wts[6] * _vec_load_6[0] : 0.0f);
    acc[1] = acc[1] + ((wts[6] > 0.0f) ? wts[6] * _vec_load_6[1] : 0.0f);
    acc[2] = acc[2] + ((wts[6] > 0.0f) ? wts[6] * _vec_load_6[2] : 0.0f);
    acc[3] = acc[3] + ((wts[6] > 0.0f) ? wts[6] * _vec_load_6[3] : 0.0f);
    acc[4] = acc[4] + ((wts[6] > 0.0f) ? wts[6] * _vec_load_6[4] : 0.0f);
    acc[5] = acc[5] + ((wts[6] > 0.0f) ? wts[6] * _vec_load_6[5] : 0.0f);
    acc[6] = acc[6] + ((wts[6] > 0.0f) ? wts[6] * _vec_load_6[6] : 0.0f);
    acc[7] = acc[7] + ((wts[6] > 0.0f) ? wts[6] * _vec_load_6[7] : 0.0f);
    wts[7] = smem_w[w_base + 7];
    acc[0] = acc[0] + ((wts[7] > 0.0f) ? wts[7] * _vec_load_7[0] : 0.0f);
    acc[1] = acc[1] + ((wts[7] > 0.0f) ? wts[7] * _vec_load_7[1] : 0.0f);
    acc[2] = acc[2] + ((wts[7] > 0.0f) ? wts[7] * _vec_load_7[2] : 0.0f);
    acc[3] = acc[3] + ((wts[7] > 0.0f) ? wts[7] * _vec_load_7[3] : 0.0f);
    acc[4] = acc[4] + ((wts[7] > 0.0f) ? wts[7] * _vec_load_7[4] : 0.0f);
    acc[5] = acc[5] + ((wts[7] > 0.0f) ? wts[7] * _vec_load_7[5] : 0.0f);
    acc[6] = acc[6] + ((wts[7] > 0.0f) ? wts[7] * _vec_load_7[6] : 0.0f);
    acc[7] = acc[7] + ((wts[7] > 0.0f) ? wts[7] * _vec_load_7[7] : 0.0f);
    wts[8] = smem_w[w_base + 8];
    acc[0] = acc[0] + ((wts[8] > 0.0f) ? wts[8] * _vec_load_8[0] : 0.0f);
    acc[1] = acc[1] + ((wts[8] > 0.0f) ? wts[8] * _vec_load_8[1] : 0.0f);
    acc[2] = acc[2] + ((wts[8] > 0.0f) ? wts[8] * _vec_load_8[2] : 0.0f);
    acc[3] = acc[3] + ((wts[8] > 0.0f) ? wts[8] * _vec_load_8[3] : 0.0f);
    acc[4] = acc[4] + ((wts[8] > 0.0f) ? wts[8] * _vec_load_8[4] : 0.0f);
    acc[5] = acc[5] + ((wts[8] > 0.0f) ? wts[8] * _vec_load_8[5] : 0.0f);
    acc[6] = acc[6] + ((wts[8] > 0.0f) ? wts[8] * _vec_load_8[6] : 0.0f);
    acc[7] = acc[7] + ((wts[8] > 0.0f) ? wts[8] * _vec_load_8[7] : 0.0f);
    wts[9] = smem_w[w_base + 9];
    acc[0] = acc[0] + ((wts[9] > 0.0f) ? wts[9] * _vec_load_9[0] : 0.0f);
    acc[1] = acc[1] + ((wts[9] > 0.0f) ? wts[9] * _vec_load_9[1] : 0.0f);
    acc[2] = acc[2] + ((wts[9] > 0.0f) ? wts[9] * _vec_load_9[2] : 0.0f);
    acc[3] = acc[3] + ((wts[9] > 0.0f) ? wts[9] * _vec_load_9[3] : 0.0f);
    acc[4] = acc[4] + ((wts[9] > 0.0f) ? wts[9] * _vec_load_9[4] : 0.0f);
    acc[5] = acc[5] + ((wts[9] > 0.0f) ? wts[9] * _vec_load_9[5] : 0.0f);
    acc[6] = acc[6] + ((wts[9] > 0.0f) ? wts[9] * _vec_load_9[6] : 0.0f);
    acc[7] = acc[7] + ((wts[9] > 0.0f) ? wts[9] * _vec_load_9[7] : 0.0f);
    wts[10] = smem_w[w_base + 10];
    acc[0] = acc[0] + ((wts[10] > 0.0f) ? wts[10] * _vec_load_10[0] : 0.0f);
    acc[1] = acc[1] + ((wts[10] > 0.0f) ? wts[10] * _vec_load_10[1] : 0.0f);
    acc[2] = acc[2] + ((wts[10] > 0.0f) ? wts[10] * _vec_load_10[2] : 0.0f);
    acc[3] = acc[3] + ((wts[10] > 0.0f) ? wts[10] * _vec_load_10[3] : 0.0f);
    acc[4] = acc[4] + ((wts[10] > 0.0f) ? wts[10] * _vec_load_10[4] : 0.0f);
    acc[5] = acc[5] + ((wts[10] > 0.0f) ? wts[10] * _vec_load_10[5] : 0.0f);
    acc[6] = acc[6] + ((wts[10] > 0.0f) ? wts[10] * _vec_load_10[6] : 0.0f);
    acc[7] = acc[7] + ((wts[10] > 0.0f) ? wts[10] * _vec_load_10[7] : 0.0f);
    wts[11] = smem_w[w_base + 11];
    acc[0] = acc[0] + ((wts[11] > 0.0f) ? wts[11] * _vec_load_11[0] : 0.0f);
    acc[1] = acc[1] + ((wts[11] > 0.0f) ? wts[11] * _vec_load_11[1] : 0.0f);
    acc[2] = acc[2] + ((wts[11] > 0.0f) ? wts[11] * _vec_load_11[2] : 0.0f);
    acc[3] = acc[3] + ((wts[11] > 0.0f) ? wts[11] * _vec_load_11[3] : 0.0f);
    acc[4] = acc[4] + ((wts[11] > 0.0f) ? wts[11] * _vec_load_11[4] : 0.0f);
    acc[5] = acc[5] + ((wts[11] > 0.0f) ? wts[11] * _vec_load_11[5] : 0.0f);
    acc[6] = acc[6] + ((wts[11] > 0.0f) ? wts[11] * _vec_load_11[6] : 0.0f);
    acc[7] = acc[7] + ((wts[11] > 0.0f) ? wts[11] * _vec_load_11[7] : 0.0f);
    wts[12] = smem_w[w_base + 12];
    acc[0] = acc[0] + ((wts[12] > 0.0f) ? wts[12] * _vec_load_12[0] : 0.0f);
    acc[1] = acc[1] + ((wts[12] > 0.0f) ? wts[12] * _vec_load_12[1] : 0.0f);
    acc[2] = acc[2] + ((wts[12] > 0.0f) ? wts[12] * _vec_load_12[2] : 0.0f);
    acc[3] = acc[3] + ((wts[12] > 0.0f) ? wts[12] * _vec_load_12[3] : 0.0f);
    acc[4] = acc[4] + ((wts[12] > 0.0f) ? wts[12] * _vec_load_12[4] : 0.0f);
    acc[5] = acc[5] + ((wts[12] > 0.0f) ? wts[12] * _vec_load_12[5] : 0.0f);
    acc[6] = acc[6] + ((wts[12] > 0.0f) ? wts[12] * _vec_load_12[6] : 0.0f);
    acc[7] = acc[7] + ((wts[12] > 0.0f) ? wts[12] * _vec_load_12[7] : 0.0f);
    wts[13] = smem_w[w_base + 13];
    acc[0] = acc[0] + ((wts[13] > 0.0f) ? wts[13] * _vec_load_13[0] : 0.0f);
    acc[1] = acc[1] + ((wts[13] > 0.0f) ? wts[13] * _vec_load_13[1] : 0.0f);
    acc[2] = acc[2] + ((wts[13] > 0.0f) ? wts[13] * _vec_load_13[2] : 0.0f);
    acc[3] = acc[3] + ((wts[13] > 0.0f) ? wts[13] * _vec_load_13[3] : 0.0f);
    acc[4] = acc[4] + ((wts[13] > 0.0f) ? wts[13] * _vec_load_13[4] : 0.0f);
    acc[5] = acc[5] + ((wts[13] > 0.0f) ? wts[13] * _vec_load_13[5] : 0.0f);
    acc[6] = acc[6] + ((wts[13] > 0.0f) ? wts[13] * _vec_load_13[6] : 0.0f);
    acc[7] = acc[7] + ((wts[13] > 0.0f) ? wts[13] * _vec_load_13[7] : 0.0f);
    wts[14] = smem_w[w_base + 14];
    acc[0] = acc[0] + ((wts[14] > 0.0f) ? wts[14] * _vec_load_14[0] : 0.0f);
    acc[1] = acc[1] + ((wts[14] > 0.0f) ? wts[14] * _vec_load_14[1] : 0.0f);
    acc[2] = acc[2] + ((wts[14] > 0.0f) ? wts[14] * _vec_load_14[2] : 0.0f);
    acc[3] = acc[3] + ((wts[14] > 0.0f) ? wts[14] * _vec_load_14[3] : 0.0f);
    acc[4] = acc[4] + ((wts[14] > 0.0f) ? wts[14] * _vec_load_14[4] : 0.0f);
    acc[5] = acc[5] + ((wts[14] > 0.0f) ? wts[14] * _vec_load_14[5] : 0.0f);
    acc[6] = acc[6] + ((wts[14] > 0.0f) ? wts[14] * _vec_load_14[6] : 0.0f);
    acc[7] = acc[7] + ((wts[14] > 0.0f) ? wts[14] * _vec_load_14[7] : 0.0f);
    wts[15] = smem_w[w_base + 15];
    acc[0] = acc[0] + ((wts[15] > 0.0f) ? wts[15] * _vec_load_15[0] : 0.0f);
    acc[1] = acc[1] + ((wts[15] > 0.0f) ? wts[15] * _vec_load_15[1] : 0.0f);
    acc[2] = acc[2] + ((wts[15] > 0.0f) ? wts[15] * _vec_load_15[2] : 0.0f);
    acc[3] = acc[3] + ((wts[15] > 0.0f) ? wts[15] * _vec_load_15[3] : 0.0f);
    acc[4] = acc[4] + ((wts[15] > 0.0f) ? wts[15] * _vec_load_15[4] : 0.0f);
    acc[5] = acc[5] + ((wts[15] > 0.0f) ? wts[15] * _vec_load_15[5] : 0.0f);
    acc[6] = acc[6] + ((wts[15] > 0.0f) ? wts[15] * _vec_load_15[6] : 0.0f);
    acc[7] = acc[7] + ((wts[15] > 0.0f) ? wts[15] * _vec_load_15[7] : 0.0f);
    int n_batches = (num_split + 16 - 1) / 16;
#pragma unroll 1
    for (int bt = 1; bt < n_batches; bt++) {
      int k0 = bt * 16;
      int _min_16 = ((k0) < (last) ? (k0) : (last));
      float _vec_load_16[8];
      {
        const uint4* _vptr_19 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_16) * 512 + d0);
        uint4 _vld_19[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_19[_blk] = _vptr_19[_blk];
          uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&_vec_load_16[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) << 16);
            (&_vec_load_16[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) & 0xffff0000u);
          }
        }
      }
      int _min_17 = ((k0 + 1) < (last) ? (k0 + 1) : (last));
      float _vec_load_17[8];
      {
        const uint4* _vptr_20 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_17) * 512 + d0);
        uint4 _vld_20[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_20[_blk] = _vptr_20[_blk];
          uint32_t* _vpairs_20 = reinterpret_cast<uint32_t*>(&_vld_20[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&_vec_load_17[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) << 16);
            (&_vec_load_17[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) & 0xffff0000u);
          }
        }
      }
      int _min_18 = ((k0 + 2) < (last) ? (k0 + 2) : (last));
      float _vec_load_18[8];
      {
        const uint4* _vptr_21 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_18) * 512 + d0);
        uint4 _vld_21[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_21[_blk] = _vptr_21[_blk];
          uint32_t* _vpairs_21 = reinterpret_cast<uint32_t*>(&_vld_21[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&_vec_load_18[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) << 16);
            (&_vec_load_18[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) & 0xffff0000u);
          }
        }
      }
      int _min_19 = ((k0 + 3) < (last) ? (k0 + 3) : (last));
      float _vec_load_19[8];
      {
        const uint4* _vptr_22 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_19) * 512 + d0);
        uint4 _vld_22[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_22[_blk] = _vptr_22[_blk];
          uint32_t* _vpairs_22 = reinterpret_cast<uint32_t*>(&_vld_22[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&_vec_load_19[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_22[_pair]) << 16);
            (&_vec_load_19[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_22[_pair]) & 0xffff0000u);
          }
        }
      }
      int _min_20 = ((k0 + 4) < (last) ? (k0 + 4) : (last));
      float _vec_load_20[8];
      {
        const uint4* _vptr_23 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_20) * 512 + d0);
        uint4 _vld_23[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_23[_blk] = _vptr_23[_blk];
          uint32_t* _vpairs_23 = reinterpret_cast<uint32_t*>(&_vld_23[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&_vec_load_20[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_23[_pair]) << 16);
            (&_vec_load_20[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_23[_pair]) & 0xffff0000u);
          }
        }
      }
      int _min_21 = ((k0 + 5) < (last) ? (k0 + 5) : (last));
      float _vec_load_21[8];
      {
        const uint4* _vptr_24 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_21) * 512 + d0);
        uint4 _vld_24[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_24[_blk] = _vptr_24[_blk];
          uint32_t* _vpairs_24 = reinterpret_cast<uint32_t*>(&_vld_24[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&_vec_load_21[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) << 16);
            (&_vec_load_21[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) & 0xffff0000u);
          }
        }
      }
      int _min_22 = ((k0 + 6) < (last) ? (k0 + 6) : (last));
      float _vec_load_22[8];
      {
        const uint4* _vptr_25 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_22) * 512 + d0);
        uint4 _vld_25[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_25[_blk] = _vptr_25[_blk];
          uint32_t* _vpairs_25 = reinterpret_cast<uint32_t*>(&_vld_25[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&_vec_load_22[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) << 16);
            (&_vec_load_22[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) & 0xffff0000u);
          }
        }
      }
      int _min_23 = ((k0 + 7) < (last) ? (k0 + 7) : (last));
      float _vec_load_23[8];
      {
        const uint4* _vptr_26 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_23) * 512 + d0);
        uint4 _vld_26[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_26[_blk] = _vptr_26[_blk];
          uint32_t* _vpairs_26 = reinterpret_cast<uint32_t*>(&_vld_26[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&_vec_load_23[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_26[_pair]) << 16);
            (&_vec_load_23[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_26[_pair]) & 0xffff0000u);
          }
        }
      }
      int _min_24 = ((k0 + 8) < (last) ? (k0 + 8) : (last));
      float _vec_load_24[8];
      {
        const uint4* _vptr_27 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_24) * 512 + d0);
        uint4 _vld_27[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_27[_blk] = _vptr_27[_blk];
          uint32_t* _vpairs_27 = reinterpret_cast<uint32_t*>(&_vld_27[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&_vec_load_24[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_27[_pair]) << 16);
            (&_vec_load_24[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_27[_pair]) & 0xffff0000u);
          }
        }
      }
      int _min_25 = ((k0 + 9) < (last) ? (k0 + 9) : (last));
      float _vec_load_25[8];
      {
        const uint4* _vptr_28 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_25) * 512 + d0);
        uint4 _vld_28[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_28[_blk] = _vptr_28[_blk];
          uint32_t* _vpairs_28 = reinterpret_cast<uint32_t*>(&_vld_28[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&_vec_load_25[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_28[_pair]) << 16);
            (&_vec_load_25[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_28[_pair]) & 0xffff0000u);
          }
        }
      }
      int _min_26 = ((k0 + 10) < (last) ? (k0 + 10) : (last));
      float _vec_load_26[8];
      {
        const uint4* _vptr_29 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_26) * 512 + d0);
        uint4 _vld_29[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_29[_blk] = _vptr_29[_blk];
          uint32_t* _vpairs_29 = reinterpret_cast<uint32_t*>(&_vld_29[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&_vec_load_26[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_29[_pair]) << 16);
            (&_vec_load_26[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_29[_pair]) & 0xffff0000u);
          }
        }
      }
      int _min_27 = ((k0 + 11) < (last) ? (k0 + 11) : (last));
      float _vec_load_27[8];
      {
        const uint4* _vptr_30 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_27) * 512 + d0);
        uint4 _vld_30[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_30[_blk] = _vptr_30[_blk];
          uint32_t* _vpairs_30 = reinterpret_cast<uint32_t*>(&_vld_30[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&_vec_load_27[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) << 16);
            (&_vec_load_27[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_30[_pair]) & 0xffff0000u);
          }
        }
      }
      int _min_28 = ((k0 + 12) < (last) ? (k0 + 12) : (last));
      float _vec_load_28[8];
      {
        const uint4* _vptr_31 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_28) * 512 + d0);
        uint4 _vld_31[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_31[_blk] = _vptr_31[_blk];
          uint32_t* _vpairs_31 = reinterpret_cast<uint32_t*>(&_vld_31[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&_vec_load_28[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) << 16);
            (&_vec_load_28[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_31[_pair]) & 0xffff0000u);
          }
        }
      }
      int _min_29 = ((k0 + 13) < (last) ? (k0 + 13) : (last));
      float _vec_load_29[8];
      {
        const uint4* _vptr_32 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_29) * 512 + d0);
        uint4 _vld_32[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_32[_blk] = _vptr_32[_blk];
          uint32_t* _vpairs_32 = reinterpret_cast<uint32_t*>(&_vld_32[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&_vec_load_29[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) << 16);
            (&_vec_load_29[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) & 0xffff0000u);
          }
        }
      }
      int _min_30 = ((k0 + 14) < (last) ? (k0 + 14) : (last));
      float _vec_load_30[8];
      {
        const uint4* _vptr_33 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_30) * 512 + d0);
        uint4 _vld_33[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_33[_blk] = _vptr_33[_blk];
          uint32_t* _vpairs_33 = reinterpret_cast<uint32_t*>(&_vld_33[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&_vec_load_30[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) << 16);
            (&_vec_load_30[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) & 0xffff0000u);
          }
        }
      }
      int _min_31 = ((k0 + 15) < (last) ? (k0 + 15) : (last));
      float _vec_load_31[8];
      {
        const uint4* _vptr_34 =
            reinterpret_cast<const uint4*>(partial_o + (stat_base + _min_31) * 512 + d0);
        uint4 _vld_34[1];
#pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
          _vld_34[_blk] = _vptr_34[_blk];
          uint32_t* _vpairs_34 = reinterpret_cast<uint32_t*>(&_vld_34[_blk]);
#pragma unroll
          for (int _pair = 0; _pair < 4; _pair++) {
            (&_vec_load_31[0 + _blk * 8 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) << 16);
            (&_vec_load_31[0 + _blk * 8 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) & 0xffff0000u);
          }
        }
      }
      wts[0] = smem_w[w_base + k0];
      acc[0] = acc[0] + ((wts[0] > 0.0f) ? wts[0] * _vec_load_16[0] : 0.0f);
      acc[1] = acc[1] + ((wts[0] > 0.0f) ? wts[0] * _vec_load_16[1] : 0.0f);
      acc[2] = acc[2] + ((wts[0] > 0.0f) ? wts[0] * _vec_load_16[2] : 0.0f);
      acc[3] = acc[3] + ((wts[0] > 0.0f) ? wts[0] * _vec_load_16[3] : 0.0f);
      acc[4] = acc[4] + ((wts[0] > 0.0f) ? wts[0] * _vec_load_16[4] : 0.0f);
      acc[5] = acc[5] + ((wts[0] > 0.0f) ? wts[0] * _vec_load_16[5] : 0.0f);
      acc[6] = acc[6] + ((wts[0] > 0.0f) ? wts[0] * _vec_load_16[6] : 0.0f);
      acc[7] = acc[7] + ((wts[0] > 0.0f) ? wts[0] * _vec_load_16[7] : 0.0f);
      wts[1] = smem_w[w_base + k0 + 1];
      acc[0] = acc[0] + ((wts[1] > 0.0f) ? wts[1] * _vec_load_17[0] : 0.0f);
      acc[1] = acc[1] + ((wts[1] > 0.0f) ? wts[1] * _vec_load_17[1] : 0.0f);
      acc[2] = acc[2] + ((wts[1] > 0.0f) ? wts[1] * _vec_load_17[2] : 0.0f);
      acc[3] = acc[3] + ((wts[1] > 0.0f) ? wts[1] * _vec_load_17[3] : 0.0f);
      acc[4] = acc[4] + ((wts[1] > 0.0f) ? wts[1] * _vec_load_17[4] : 0.0f);
      acc[5] = acc[5] + ((wts[1] > 0.0f) ? wts[1] * _vec_load_17[5] : 0.0f);
      acc[6] = acc[6] + ((wts[1] > 0.0f) ? wts[1] * _vec_load_17[6] : 0.0f);
      acc[7] = acc[7] + ((wts[1] > 0.0f) ? wts[1] * _vec_load_17[7] : 0.0f);
      wts[2] = smem_w[w_base + k0 + 2];
      acc[0] = acc[0] + ((wts[2] > 0.0f) ? wts[2] * _vec_load_18[0] : 0.0f);
      acc[1] = acc[1] + ((wts[2] > 0.0f) ? wts[2] * _vec_load_18[1] : 0.0f);
      acc[2] = acc[2] + ((wts[2] > 0.0f) ? wts[2] * _vec_load_18[2] : 0.0f);
      acc[3] = acc[3] + ((wts[2] > 0.0f) ? wts[2] * _vec_load_18[3] : 0.0f);
      acc[4] = acc[4] + ((wts[2] > 0.0f) ? wts[2] * _vec_load_18[4] : 0.0f);
      acc[5] = acc[5] + ((wts[2] > 0.0f) ? wts[2] * _vec_load_18[5] : 0.0f);
      acc[6] = acc[6] + ((wts[2] > 0.0f) ? wts[2] * _vec_load_18[6] : 0.0f);
      acc[7] = acc[7] + ((wts[2] > 0.0f) ? wts[2] * _vec_load_18[7] : 0.0f);
      wts[3] = smem_w[w_base + k0 + 3];
      acc[0] = acc[0] + ((wts[3] > 0.0f) ? wts[3] * _vec_load_19[0] : 0.0f);
      acc[1] = acc[1] + ((wts[3] > 0.0f) ? wts[3] * _vec_load_19[1] : 0.0f);
      acc[2] = acc[2] + ((wts[3] > 0.0f) ? wts[3] * _vec_load_19[2] : 0.0f);
      acc[3] = acc[3] + ((wts[3] > 0.0f) ? wts[3] * _vec_load_19[3] : 0.0f);
      acc[4] = acc[4] + ((wts[3] > 0.0f) ? wts[3] * _vec_load_19[4] : 0.0f);
      acc[5] = acc[5] + ((wts[3] > 0.0f) ? wts[3] * _vec_load_19[5] : 0.0f);
      acc[6] = acc[6] + ((wts[3] > 0.0f) ? wts[3] * _vec_load_19[6] : 0.0f);
      acc[7] = acc[7] + ((wts[3] > 0.0f) ? wts[3] * _vec_load_19[7] : 0.0f);
      wts[4] = smem_w[w_base + k0 + 4];
      acc[0] = acc[0] + ((wts[4] > 0.0f) ? wts[4] * _vec_load_20[0] : 0.0f);
      acc[1] = acc[1] + ((wts[4] > 0.0f) ? wts[4] * _vec_load_20[1] : 0.0f);
      acc[2] = acc[2] + ((wts[4] > 0.0f) ? wts[4] * _vec_load_20[2] : 0.0f);
      acc[3] = acc[3] + ((wts[4] > 0.0f) ? wts[4] * _vec_load_20[3] : 0.0f);
      acc[4] = acc[4] + ((wts[4] > 0.0f) ? wts[4] * _vec_load_20[4] : 0.0f);
      acc[5] = acc[5] + ((wts[4] > 0.0f) ? wts[4] * _vec_load_20[5] : 0.0f);
      acc[6] = acc[6] + ((wts[4] > 0.0f) ? wts[4] * _vec_load_20[6] : 0.0f);
      acc[7] = acc[7] + ((wts[4] > 0.0f) ? wts[4] * _vec_load_20[7] : 0.0f);
      wts[5] = smem_w[w_base + k0 + 5];
      acc[0] = acc[0] + ((wts[5] > 0.0f) ? wts[5] * _vec_load_21[0] : 0.0f);
      acc[1] = acc[1] + ((wts[5] > 0.0f) ? wts[5] * _vec_load_21[1] : 0.0f);
      acc[2] = acc[2] + ((wts[5] > 0.0f) ? wts[5] * _vec_load_21[2] : 0.0f);
      acc[3] = acc[3] + ((wts[5] > 0.0f) ? wts[5] * _vec_load_21[3] : 0.0f);
      acc[4] = acc[4] + ((wts[5] > 0.0f) ? wts[5] * _vec_load_21[4] : 0.0f);
      acc[5] = acc[5] + ((wts[5] > 0.0f) ? wts[5] * _vec_load_21[5] : 0.0f);
      acc[6] = acc[6] + ((wts[5] > 0.0f) ? wts[5] * _vec_load_21[6] : 0.0f);
      acc[7] = acc[7] + ((wts[5] > 0.0f) ? wts[5] * _vec_load_21[7] : 0.0f);
      wts[6] = smem_w[w_base + k0 + 6];
      acc[0] = acc[0] + ((wts[6] > 0.0f) ? wts[6] * _vec_load_22[0] : 0.0f);
      acc[1] = acc[1] + ((wts[6] > 0.0f) ? wts[6] * _vec_load_22[1] : 0.0f);
      acc[2] = acc[2] + ((wts[6] > 0.0f) ? wts[6] * _vec_load_22[2] : 0.0f);
      acc[3] = acc[3] + ((wts[6] > 0.0f) ? wts[6] * _vec_load_22[3] : 0.0f);
      acc[4] = acc[4] + ((wts[6] > 0.0f) ? wts[6] * _vec_load_22[4] : 0.0f);
      acc[5] = acc[5] + ((wts[6] > 0.0f) ? wts[6] * _vec_load_22[5] : 0.0f);
      acc[6] = acc[6] + ((wts[6] > 0.0f) ? wts[6] * _vec_load_22[6] : 0.0f);
      acc[7] = acc[7] + ((wts[6] > 0.0f) ? wts[6] * _vec_load_22[7] : 0.0f);
      wts[7] = smem_w[w_base + k0 + 7];
      acc[0] = acc[0] + ((wts[7] > 0.0f) ? wts[7] * _vec_load_23[0] : 0.0f);
      acc[1] = acc[1] + ((wts[7] > 0.0f) ? wts[7] * _vec_load_23[1] : 0.0f);
      acc[2] = acc[2] + ((wts[7] > 0.0f) ? wts[7] * _vec_load_23[2] : 0.0f);
      acc[3] = acc[3] + ((wts[7] > 0.0f) ? wts[7] * _vec_load_23[3] : 0.0f);
      acc[4] = acc[4] + ((wts[7] > 0.0f) ? wts[7] * _vec_load_23[4] : 0.0f);
      acc[5] = acc[5] + ((wts[7] > 0.0f) ? wts[7] * _vec_load_23[5] : 0.0f);
      acc[6] = acc[6] + ((wts[7] > 0.0f) ? wts[7] * _vec_load_23[6] : 0.0f);
      acc[7] = acc[7] + ((wts[7] > 0.0f) ? wts[7] * _vec_load_23[7] : 0.0f);
      wts[8] = smem_w[w_base + k0 + 8];
      acc[0] = acc[0] + ((wts[8] > 0.0f) ? wts[8] * _vec_load_24[0] : 0.0f);
      acc[1] = acc[1] + ((wts[8] > 0.0f) ? wts[8] * _vec_load_24[1] : 0.0f);
      acc[2] = acc[2] + ((wts[8] > 0.0f) ? wts[8] * _vec_load_24[2] : 0.0f);
      acc[3] = acc[3] + ((wts[8] > 0.0f) ? wts[8] * _vec_load_24[3] : 0.0f);
      acc[4] = acc[4] + ((wts[8] > 0.0f) ? wts[8] * _vec_load_24[4] : 0.0f);
      acc[5] = acc[5] + ((wts[8] > 0.0f) ? wts[8] * _vec_load_24[5] : 0.0f);
      acc[6] = acc[6] + ((wts[8] > 0.0f) ? wts[8] * _vec_load_24[6] : 0.0f);
      acc[7] = acc[7] + ((wts[8] > 0.0f) ? wts[8] * _vec_load_24[7] : 0.0f);
      wts[9] = smem_w[w_base + k0 + 9];
      acc[0] = acc[0] + ((wts[9] > 0.0f) ? wts[9] * _vec_load_25[0] : 0.0f);
      acc[1] = acc[1] + ((wts[9] > 0.0f) ? wts[9] * _vec_load_25[1] : 0.0f);
      acc[2] = acc[2] + ((wts[9] > 0.0f) ? wts[9] * _vec_load_25[2] : 0.0f);
      acc[3] = acc[3] + ((wts[9] > 0.0f) ? wts[9] * _vec_load_25[3] : 0.0f);
      acc[4] = acc[4] + ((wts[9] > 0.0f) ? wts[9] * _vec_load_25[4] : 0.0f);
      acc[5] = acc[5] + ((wts[9] > 0.0f) ? wts[9] * _vec_load_25[5] : 0.0f);
      acc[6] = acc[6] + ((wts[9] > 0.0f) ? wts[9] * _vec_load_25[6] : 0.0f);
      acc[7] = acc[7] + ((wts[9] > 0.0f) ? wts[9] * _vec_load_25[7] : 0.0f);
      wts[10] = smem_w[w_base + k0 + 10];
      acc[0] = acc[0] + ((wts[10] > 0.0f) ? wts[10] * _vec_load_26[0] : 0.0f);
      acc[1] = acc[1] + ((wts[10] > 0.0f) ? wts[10] * _vec_load_26[1] : 0.0f);
      acc[2] = acc[2] + ((wts[10] > 0.0f) ? wts[10] * _vec_load_26[2] : 0.0f);
      acc[3] = acc[3] + ((wts[10] > 0.0f) ? wts[10] * _vec_load_26[3] : 0.0f);
      acc[4] = acc[4] + ((wts[10] > 0.0f) ? wts[10] * _vec_load_26[4] : 0.0f);
      acc[5] = acc[5] + ((wts[10] > 0.0f) ? wts[10] * _vec_load_26[5] : 0.0f);
      acc[6] = acc[6] + ((wts[10] > 0.0f) ? wts[10] * _vec_load_26[6] : 0.0f);
      acc[7] = acc[7] + ((wts[10] > 0.0f) ? wts[10] * _vec_load_26[7] : 0.0f);
      wts[11] = smem_w[w_base + k0 + 11];
      acc[0] = acc[0] + ((wts[11] > 0.0f) ? wts[11] * _vec_load_27[0] : 0.0f);
      acc[1] = acc[1] + ((wts[11] > 0.0f) ? wts[11] * _vec_load_27[1] : 0.0f);
      acc[2] = acc[2] + ((wts[11] > 0.0f) ? wts[11] * _vec_load_27[2] : 0.0f);
      acc[3] = acc[3] + ((wts[11] > 0.0f) ? wts[11] * _vec_load_27[3] : 0.0f);
      acc[4] = acc[4] + ((wts[11] > 0.0f) ? wts[11] * _vec_load_27[4] : 0.0f);
      acc[5] = acc[5] + ((wts[11] > 0.0f) ? wts[11] * _vec_load_27[5] : 0.0f);
      acc[6] = acc[6] + ((wts[11] > 0.0f) ? wts[11] * _vec_load_27[6] : 0.0f);
      acc[7] = acc[7] + ((wts[11] > 0.0f) ? wts[11] * _vec_load_27[7] : 0.0f);
      wts[12] = smem_w[w_base + k0 + 12];
      acc[0] = acc[0] + ((wts[12] > 0.0f) ? wts[12] * _vec_load_28[0] : 0.0f);
      acc[1] = acc[1] + ((wts[12] > 0.0f) ? wts[12] * _vec_load_28[1] : 0.0f);
      acc[2] = acc[2] + ((wts[12] > 0.0f) ? wts[12] * _vec_load_28[2] : 0.0f);
      acc[3] = acc[3] + ((wts[12] > 0.0f) ? wts[12] * _vec_load_28[3] : 0.0f);
      acc[4] = acc[4] + ((wts[12] > 0.0f) ? wts[12] * _vec_load_28[4] : 0.0f);
      acc[5] = acc[5] + ((wts[12] > 0.0f) ? wts[12] * _vec_load_28[5] : 0.0f);
      acc[6] = acc[6] + ((wts[12] > 0.0f) ? wts[12] * _vec_load_28[6] : 0.0f);
      acc[7] = acc[7] + ((wts[12] > 0.0f) ? wts[12] * _vec_load_28[7] : 0.0f);
      wts[13] = smem_w[w_base + k0 + 13];
      acc[0] = acc[0] + ((wts[13] > 0.0f) ? wts[13] * _vec_load_29[0] : 0.0f);
      acc[1] = acc[1] + ((wts[13] > 0.0f) ? wts[13] * _vec_load_29[1] : 0.0f);
      acc[2] = acc[2] + ((wts[13] > 0.0f) ? wts[13] * _vec_load_29[2] : 0.0f);
      acc[3] = acc[3] + ((wts[13] > 0.0f) ? wts[13] * _vec_load_29[3] : 0.0f);
      acc[4] = acc[4] + ((wts[13] > 0.0f) ? wts[13] * _vec_load_29[4] : 0.0f);
      acc[5] = acc[5] + ((wts[13] > 0.0f) ? wts[13] * _vec_load_29[5] : 0.0f);
      acc[6] = acc[6] + ((wts[13] > 0.0f) ? wts[13] * _vec_load_29[6] : 0.0f);
      acc[7] = acc[7] + ((wts[13] > 0.0f) ? wts[13] * _vec_load_29[7] : 0.0f);
      wts[14] = smem_w[w_base + k0 + 14];
      acc[0] = acc[0] + ((wts[14] > 0.0f) ? wts[14] * _vec_load_30[0] : 0.0f);
      acc[1] = acc[1] + ((wts[14] > 0.0f) ? wts[14] * _vec_load_30[1] : 0.0f);
      acc[2] = acc[2] + ((wts[14] > 0.0f) ? wts[14] * _vec_load_30[2] : 0.0f);
      acc[3] = acc[3] + ((wts[14] > 0.0f) ? wts[14] * _vec_load_30[3] : 0.0f);
      acc[4] = acc[4] + ((wts[14] > 0.0f) ? wts[14] * _vec_load_30[4] : 0.0f);
      acc[5] = acc[5] + ((wts[14] > 0.0f) ? wts[14] * _vec_load_30[5] : 0.0f);
      acc[6] = acc[6] + ((wts[14] > 0.0f) ? wts[14] * _vec_load_30[6] : 0.0f);
      acc[7] = acc[7] + ((wts[14] > 0.0f) ? wts[14] * _vec_load_30[7] : 0.0f);
      wts[15] = smem_w[w_base + k0 + 15];
      acc[0] = acc[0] + ((wts[15] > 0.0f) ? wts[15] * _vec_load_31[0] : 0.0f);
      acc[1] = acc[1] + ((wts[15] > 0.0f) ? wts[15] * _vec_load_31[1] : 0.0f);
      acc[2] = acc[2] + ((wts[15] > 0.0f) ? wts[15] * _vec_load_31[2] : 0.0f);
      acc[3] = acc[3] + ((wts[15] > 0.0f) ? wts[15] * _vec_load_31[3] : 0.0f);
      acc[4] = acc[4] + ((wts[15] > 0.0f) ? wts[15] * _vec_load_31[4] : 0.0f);
      acc[5] = acc[5] + ((wts[15] > 0.0f) ? wts[15] * _vec_load_31[5] : 0.0f);
      acc[6] = acc[6] + ((wts[15] > 0.0f) ? wts[15] * _vec_load_31[6] : 0.0f);
      acc[7] = acc[7] + ((wts[15] > 0.0f) ? wts[15] * _vec_load_31[7] : 0.0f);
    }
    for (int e_1 = 0; e_1 < 8; e_1++) {
      acc[e_1] = acc[e_1] * inv;
    }
    {
      __nv_bfloat162 _pk[4];
      _pk[0] = __floats2bfloat162_rn(acc[0 + 0], acc[0 + 1]);
      _pk[1] = __floats2bfloat162_rn(acc[0 + 2], acc[0 + 3]);
      _pk[2] = __floats2bfloat162_rn(acc[0 + 4], acc[0 + 5]);
      _pk[3] = __floats2bfloat162_rn(acc[0 + 6], acc[0 + 7]);
      *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (row * 512 + d0)))[0]) =
          *reinterpret_cast<uint4*>(&_pk[0]);
    }
    if (has_lse != 0) {
      if (part == 0) {
        if (lane_0 == 0) {
          float lse_v = -CUDART_INF_F;
          if (sum_w > 0.0f) {
            float _log2_35;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_35) : "f"(sum_w));
            lse_v = (m + _log2_35) * 0.6931471805599453f;
          }
          *(reinterpret_cast<float*>(lse + row) + (0)) = lse_v;
        }
      }
    }
  }
}

}  // extern "C"
