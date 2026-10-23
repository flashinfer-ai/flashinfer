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
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

__device__ __forceinline__ float max_noftz(float a, float b) {
  float c;
  asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
  return c;
}

extern "C" {

__global__
__launch_bounds__(128, LAUNCH_MIN_BLOCKS) void kernel_cake_rubin_mla_nvfp4_decode_302684a62be285e34842(
    __nv_bfloat16* __restrict__ q, unsigned int* __restrict__ q_nope, uint8_t* __restrict__ q_sf,
    unsigned int* __restrict__ q_rope, float* __restrict__ q_scale, int rows, float c_nope,
    float c_rope, float kpe_scale, float ckv_scale) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // === Task calls (declaration order) ===
  int row = blockIdx.x * 4 + warp;
  if (row < rows) {
    int lane_0 = lane;
    int base = row * 576;
    float v[16];
    float _vec_load_0[8];
    {
      const uint4* _vptr_0 = reinterpret_cast<const uint4*>(q + base + lane_0 * 16);
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
    float _vec_load_1[8];
    {
      const uint4* _vptr_1 = reinterpret_cast<const uint4*>(q + base + lane_0 * 16 + 8);
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
    for (int j = 0; j < 8; j++) {
      v[j] = _vec_load_0[j];
      v[8 + j] = _vec_load_1[j];
    }
    float _vec_load_2[2];
    {
      uint32_t _bf16x2_bits_2;
      _bf16x2_bits_2 = *reinterpret_cast<const uint32_t*>(q + base + 512 + lane_0 * 2);
      (&_vec_load_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_2) << 16);
      (&_vec_load_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_2) & 0xffff0000u);
    }
    float amax_b = 0.0f;
    for (int j_1 = 0; j_1 < 16; j_1++) {
      float _abs_3;
      asm("abs.f32 %0, %1;" : "=f"(_abs_3) : "f"(v[j_1]));
      amax_b = max_noftz(amax_b, _abs_3);
    }
    float _abs_4;
    asm("abs.f32 %0, %1;" : "=f"(_abs_4) : "f"(_vec_load_2[0]));
    float _abs_5;
    asm("abs.f32 %0, %1;" : "=f"(_abs_5) : "f"(_vec_load_2[1]));
    float amax_r_l;
    asm("max.f32 %0, %1, %2;" : "=f"(amax_r_l) : "f"(_abs_4), "f"(_abs_5));
    float amax_n = amax_b;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      amax_n = max_noftz(amax_n, __shfl_xor_sync(0xFFFFFFFF, amax_n, offset));
    float amax_r = amax_r_l;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      amax_r = max_noftz(amax_r, __shfl_xor_sync(0xFFFFFFFF, amax_r, offset));
    float qs;
    asm("max.f32 %0, %1, %2;" : "=f"(qs) : "f"(amax_n * c_nope), "f"(amax_r * c_rope));
    if (qs <= 0.0f) {
      qs = 1.0f;
    }
    float inv6 = __frcp_rn(6.0f * qs);
    float sf_val = 0.0f;
    if (amax_b > 0.0f) {
      sf_val = amax_b * inv6;
    }
    uint16_t _cvt_satfinite_e4m3x2_f32_6;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
        : "=h"(_cvt_satfinite_e4m3x2_f32_6)
        : "f"(0.0f), "f"(sf_val));
    unsigned int sf_pair = (unsigned int)_cvt_satfinite_e4m3x2_f32_6;
    unsigned int sf_byte = sf_pair & 255;
    uint16_t _e4m3x2_decode_7 = (uint16_t)(sf_pair & 0xFFu);
    uint32_t _f16x2_decode_7;
    float sf_f;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_7) : "h"(_e4m3x2_decode_7));
    uint16_t _f16_decode_7 = (uint16_t)_f16x2_decode_7;
    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_f) : "h"(_f16_decode_7));
    float out_scale = 0.0f;
    if (sf_f > 0.0f) {
      float _rcp_8 = __frcp_rn(sf_f * qs);
      out_scale = _rcp_8;
    }
    for (int j_2 = 0; j_2 < 16; j_2++) {
      v[j_2] = v[j_2] * out_scale;
    }
    unsigned int w0 = 0;
    unsigned int w1 = 0;
    for (int j_3 = 0; j_3 < 4; j_3++) {
      uint32_t _fp4_e2m1_pack_pair_9;
      asm("{\n"
          ".reg .b8 byte0;\n"
          "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
          "mov.b32 %0, {byte0, 0, 0, 0};\n"
          "}\n"
          : "=r"(_fp4_e2m1_pack_pair_9)
          : "f"(v[2 * j_3]), "f"(v[2 * j_3 + 1]));
      w0 = w0 | _fp4_e2m1_pack_pair_9 << (unsigned int)(8 * j_3);
      uint32_t _fp4_e2m1_pack_pair_10;
      asm("{\n"
          ".reg .b8 byte0;\n"
          "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
          "mov.b32 %0, {byte0, 0, 0, 0};\n"
          "}\n"
          : "=r"(_fp4_e2m1_pack_pair_10)
          : "f"(v[8 + 2 * j_3]), "f"(v[8 + 2 * j_3 + 1]));
      w1 = w1 | _fp4_e2m1_pack_pair_10 << (unsigned int)(8 * j_3);
    }
    *(reinterpret_cast<unsigned int*>(q_nope + (row * 64 + lane_0 * 2)) + (0)) = w0;
    *(reinterpret_cast<unsigned int*>(q_nope + (row * 64 + lane_0 * 2 + 1)) + (0)) = w1;
    *(reinterpret_cast<unsigned char*>(q_sf + (row * 32 + lane_0)) + (0)) =
        (unsigned char)(sf_byte);
    float _rcp_11 = __frcp_rn(qs * ckv_scale);
    float rope_mul = kpe_scale * _rcp_11;
    uint16_t _cvt_satfinite_e4m3x2_f32_12;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
        : "=h"(_cvt_satfinite_e4m3x2_f32_12)
        : "f"(_vec_load_2[1] * rope_mul), "f"(_vec_load_2[0] * rope_mul));
    unsigned int rpair = (unsigned int)_cvt_satfinite_e4m3x2_f32_12;
    unsigned int rw_hi = __shfl_down_sync(0xFFFFFFFF, rpair, 1, 32);
    if ((lane_0 & 1) == 0) {
      *(reinterpret_cast<unsigned int*>(q_rope + (row * 16 + (lane_0 >> 1))) + (0)) =
          rpair | rw_hi << 16;
    }
    if (lane_0 == 0) {
      *(reinterpret_cast<float*>(q_scale + row) + (0)) = qs;
    }
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

}  // extern "C"
