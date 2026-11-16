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
struct CakeFastDivmod {
  int32_t divisor;
  uint32_t multiplier;
  uint32_t shift_right;
};
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

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
#define SMEM_SK_OFF 0
#define SMEM_SK_STAGE_BYTES 24576
#define SMEM_SK_STRIDE 24576
#define SMEM_SQ_OFF 24576
#define SMEM_SQ_STAGE_BYTES 8192
#define SMEM_SQ_STRIDE 8192
#define SMEM_SV_OFF 32768
#define SMEM_SV_STAGE_BYTES 12288
#define SMEM_SV_STRIDE 12288
#define SMEM_SALPHA_OFF 45056
#define SMEM_SALPHA_STAGE_BYTES 64
#define SMEM_SALPHA_STRIDE 64
#define SMEM_SBETA_OFF 45120
#define SMEM_SBETA_STAGE_BYTES 192
#define SMEM_SBETA_STRIDE 192
#define SMEM_SSSM_OFF 45312
#define SMEM_SSSM_STAGE_BYTES 64
#define SMEM_SSSM_STRIDE 64
#define SMEM_TOTAL 45440
#define THREADS 64
#define LAUNCH_MIN_BLOCKS 4

#include <math_constants.h>

__device__ __forceinline__ float approx_rcp(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

#if __CUDA_ARCH__ == 900
#else

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
  float2 r;
  asm("mul.rn.ftz.f32x2 %0, %1, %2;"
      : "=l"(*(unsigned long long*)&r)
      : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
  return r;
}

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
  float2 r;
  asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
      : "=l"(*(unsigned long long*)&r)
      : "l"(*(const unsigned long long*)&a), "l"(*(const unsigned long long*)&b),
        "l"(*(const unsigned long long*)&c));
  return r;
}

#endif
extern "C" {

__global__ __launch_bounds__(64, LAUNCH_MIN_BLOCKS) void kernel_cake_gdp_585a5ea4cbdd07f1b2a8(
    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v,
    float* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a,
    float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out,
    int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices,
    int* __restrict__ ssm_state_indices, CakeFastDivmod heads, int num_tokens, float scale,
    int disable_state_update, int use_ssm_state_indices, long long state_size_p0,
    long long state_stride_p1, long long state_stride_p2, long long state_stride_p0,
    long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long v_stride_p0,
    long long v_stride_p1, long long v_stride_p2, long long b_stride_p0, long long b_stride_p1,
    long long b_stride_p2, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2,
    long long a_stride_p0, long long a_stride_p1, long long a_stride_p2,
    long long ssm_state_indices_stride_p0, long long ssm_state_indices_stride_p1,
    long long out_stride_p0, long long out_stride_p1, long long out_stride_p2) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
#if __CUDA_ARCH__ == 900
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#else
#if __CUDA_ARCH__ == 1000
  asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }"
               : "=r"(smem)
               : "l"(smem_raw));
  smem = make_warp_uniform(smem);
#else
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif
#endif

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  float* sK = reinterpret_cast<float*>(smem_raw + 0);
  const int sK_addr = smem + 0;
  float* sQ = reinterpret_cast<float*>(smem_raw + 24576);
  const int sQ_addr = smem + 24576;
  float* sV = reinterpret_cast<float*>(smem_raw + 32768);
  const int sV_addr = smem + 32768;
  float* sAlpha = reinterpret_cast<float*>(smem_raw + 45056);
  const int sAlpha_addr = smem + 45056;
  float* sBeta = reinterpret_cast<float*>(smem_raw + 45120);
  const int sBeta_addr = smem + 45120;
  int* sSsm = reinterpret_cast<int*>(smem_raw + 45312);
  const int sSsm_addr = smem + 45312;

  // === Task calls (declaration order) ===
  uint32_t _fast_div_q_0 =
      (heads.divisor != 1)
          ? (__umulhi((uint32_t)((unsigned int)blockIdx.x), heads.multiplier) >> heads.shift_right)
          : (uint32_t)((unsigned int)blockIdx.x);
  uint32_t _fast_div_r_0 =
      (uint32_t)((unsigned int)blockIdx.x) - _fast_div_q_0 * (uint32_t)(heads.divisor);
  int n = (int)_fast_div_q_0;
  int h = (int)_fast_div_r_0;
  int tid_0 = tid;
  int lane_local = lane;
  int warp_local = warp;
  int u = tid_0 - tid_0 / 4 * 4;
  int g = tid_0 / 4;
  int row0 = g * 4;
  unsigned int lane_byte = (unsigned int)(u * 16);
  unsigned int group_byte = (unsigned int)(g * 16);
  long long lane_elem = (long long)(u * 4);
  long long state_pool_size = state_size_p0;
  int read_state_slot_raw = initial_state_indices[n];
  int read_state_slot_valid =
      ((read_state_slot_raw >= 0 && state_pool_size > (long long)read_state_slot_raw) ? 1 : 0);
  int write_state_slot_raw = output_state_indices[n];
  int write_state_slot_valid =
      ((write_state_slot_raw >= 0 && state_pool_size > (long long)write_state_slot_raw) ? 1 : 0);
  long long head_rows_off =
      (long long)h * state_stride_p1 + (long long)row0 * state_stride_p2 + lane_elem;
  long long read_rows_base = (long long)read_state_slot_raw * state_stride_p0 + head_rows_off;
  long long write_rows_base = (long long)write_state_slot_raw * state_stride_p0 + head_rows_off;
  int k_col0 = lane_local * 4;
  int steps = num_tokens * 3;
  if (read_state_slot_valid != 0) {
    float S[128];
#if __CUDA_ARCH__ == 900
#else
    float2 S2[64];
#endif
    float k_regs[32];
#if __CUDA_ARCH__ == 900
#else
    float2 k2[16];
#endif
    float v_regs[4];
    float o4[4];
    float r_q[4];
    float r_k[4];
    float r_h[4];
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base);
      S[0 + 0] = _v4.x;
      S[0 + 1] = _v4.y;
      S[0 + 2] = _v4.z;
      S[0 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 16);
      S[4 + 0] = _v4.x;
      S[4 + 1] = _v4.y;
      S[4 + 2] = _v4.z;
      S[4 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 32);
      S[8 + 0] = _v4.x;
      S[8 + 1] = _v4.y;
      S[8 + 2] = _v4.z;
      S[8 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 48);
      S[12 + 0] = _v4.x;
      S[12 + 1] = _v4.y;
      S[12 + 2] = _v4.z;
      S[12 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 64);
      S[16 + 0] = _v4.x;
      S[16 + 1] = _v4.y;
      S[16 + 2] = _v4.z;
      S[16 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 80);
      S[20 + 0] = _v4.x;
      S[20 + 1] = _v4.y;
      S[20 + 2] = _v4.z;
      S[20 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 96);
      S[24 + 0] = _v4.x;
      S[24 + 1] = _v4.y;
      S[24 + 2] = _v4.z;
      S[24 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 112);
      S[28 + 0] = _v4.x;
      S[28 + 1] = _v4.y;
      S[28 + 2] = _v4.z;
      S[28 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + state_stride_p2);
      S[32 + 0] = _v4.x;
      S[32 + 1] = _v4.y;
      S[32 + 2] = _v4.z;
      S[32 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + state_stride_p2 + 16);
      S[36 + 0] = _v4.x;
      S[36 + 1] = _v4.y;
      S[36 + 2] = _v4.z;
      S[36 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + state_stride_p2 + 32);
      S[40 + 0] = _v4.x;
      S[40 + 1] = _v4.y;
      S[40 + 2] = _v4.z;
      S[40 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + state_stride_p2 + 48);
      S[44 + 0] = _v4.x;
      S[44 + 1] = _v4.y;
      S[44 + 2] = _v4.z;
      S[44 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + state_stride_p2 + 64);
      S[48 + 0] = _v4.x;
      S[48 + 1] = _v4.y;
      S[48 + 2] = _v4.z;
      S[48 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + state_stride_p2 + 80);
      S[52 + 0] = _v4.x;
      S[52 + 1] = _v4.y;
      S[52 + 2] = _v4.z;
      S[52 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + state_stride_p2 + 96);
      S[56 + 0] = _v4.x;
      S[56 + 1] = _v4.y;
      S[56 + 2] = _v4.z;
      S[56 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + state_stride_p2 + 112);
      S[60 + 0] = _v4.x;
      S[60 + 1] = _v4.y;
      S[60 + 2] = _v4.z;
      S[60 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 2 * state_stride_p2);
      S[64 + 0] = _v4.x;
      S[64 + 1] = _v4.y;
      S[64 + 2] = _v4.z;
      S[64 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 2 * state_stride_p2 + 16);
      S[68 + 0] = _v4.x;
      S[68 + 1] = _v4.y;
      S[68 + 2] = _v4.z;
      S[68 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 2 * state_stride_p2 + 32);
      S[72 + 0] = _v4.x;
      S[72 + 1] = _v4.y;
      S[72 + 2] = _v4.z;
      S[72 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 2 * state_stride_p2 + 48);
      S[76 + 0] = _v4.x;
      S[76 + 1] = _v4.y;
      S[76 + 2] = _v4.z;
      S[76 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 2 * state_stride_p2 + 64);
      S[80 + 0] = _v4.x;
      S[80 + 1] = _v4.y;
      S[80 + 2] = _v4.z;
      S[80 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 2 * state_stride_p2 + 80);
      S[84 + 0] = _v4.x;
      S[84 + 1] = _v4.y;
      S[84 + 2] = _v4.z;
      S[84 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 2 * state_stride_p2 + 96);
      S[88 + 0] = _v4.x;
      S[88 + 1] = _v4.y;
      S[88 + 2] = _v4.z;
      S[88 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 2 * state_stride_p2 + 112);
      S[92 + 0] = _v4.x;
      S[92 + 1] = _v4.y;
      S[92 + 2] = _v4.z;
      S[92 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 3 * state_stride_p2);
      S[96 + 0] = _v4.x;
      S[96 + 1] = _v4.y;
      S[96 + 2] = _v4.z;
      S[96 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 3 * state_stride_p2 + 16);
      S[100 + 0] = _v4.x;
      S[100 + 1] = _v4.y;
      S[100 + 2] = _v4.z;
      S[100 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 3 * state_stride_p2 + 32);
      S[104 + 0] = _v4.x;
      S[104 + 1] = _v4.y;
      S[104 + 2] = _v4.z;
      S[104 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 3 * state_stride_p2 + 48);
      S[108 + 0] = _v4.x;
      S[108 + 1] = _v4.y;
      S[108 + 2] = _v4.z;
      S[108 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 3 * state_stride_p2 + 64);
      S[112 + 0] = _v4.x;
      S[112 + 1] = _v4.y;
      S[112 + 2] = _v4.z;
      S[112 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 3 * state_stride_p2 + 80);
      S[116 + 0] = _v4.x;
      S[116 + 1] = _v4.y;
      S[116 + 2] = _v4.z;
      S[116 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 3 * state_stride_p2 + 96);
      S[120 + 0] = _v4.x;
      S[120 + 1] = _v4.y;
      S[120 + 2] = _v4.z;
      S[120 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 3 * state_stride_p2 + 112);
      S[124 + 0] = _v4.x;
      S[124 + 1] = _v4.y;
      S[124 + 2] = _v4.z;
      S[124 + 3] = _v4.w;
    }
#pragma unroll 2
    for (int mm = 0; mm < 24; mm++) {
      int m = mm * 2 + warp_local;
      if (m < steps) {
        long long k_index =
            (long long)n * k_stride_p0 + (long long)m * k_stride_p1 + (long long)h * k_stride_p2;
        {
          uint2 _vld_32;
          _vld_32 = *reinterpret_cast<const uint2*>(k + k_index + (long long)k_col0);
          uint32_t* _vpairs_32 = reinterpret_cast<uint32_t*>(&_vld_32);
#pragma unroll
          for (int _pair = 0; _pair < 2; _pair++) {
            (&r_k[0 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) << 16);
            (&r_k[0 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) & 0xffff0000u);
          }
        }
#if __CUDA_ARCH__ == 900
        float k_pair0_x = r_k[0];
        float k_pair0_y = r_k[1];
        float k_pair1_x = r_k[2];
        float k_pair1_y = r_k[3];
        float _fma_0;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_0)
            : "f"(k_pair0_x), "f"(k_pair0_x), "f"(0.0f));
        float _fma_1;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1)
            : "f"(k_pair0_y), "f"(k_pair0_y), "f"(0.0f));
        float sum_k_pair_x = _fma_0;
        float sum_k_pair_y = _fma_1;
        float _fma_2;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_2)
            : "f"(k_pair1_x), "f"(k_pair1_x), "f"(sum_k_pair_x));
        float _fma_3;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_3)
            : "f"(k_pair1_y), "f"(k_pair1_y), "f"(sum_k_pair_y));
        sum_k_pair_x = _fma_2;
        sum_k_pair_y = _fma_3;
        float sum_k = sum_k_pair_x + sum_k_pair_y;
#else
        float2 k_pair0 = make_float2(r_k[0], r_k[1]);
        float2 k_pair1 = make_float2(r_k[2], r_k[3]);
        float2 sum_k_pair = fma_f32x2_rn_ftz(k_pair0, k_pair0, make_float2(0.0f, 0.0f));
        sum_k_pair = fma_f32x2_rn_ftz(k_pair1, k_pair1, sum_k_pair);
        float sum_k = sum_k_pair.x + sum_k_pair.y;
#endif
        float _warp_reduce_33 = sum_k;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
          _warp_reduce_33 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_33, offset);
        sum_k = _warp_reduce_33;
        float k_norm = rsqrtf(sum_k + 1e-06f);
#if __CUDA_ARCH__ == 900
        float k_norm_pair_x = k_norm;
        float k_norm_pair_y = k_norm;
        float _mul_0;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(k_pair0_x), "f"(k_norm_pair_x));
        float _mul_1;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(k_pair0_y), "f"(k_norm_pair_y));
        k_pair0_x = _mul_0;
        k_pair0_y = _mul_1;
        float _mul_2;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(k_pair1_x), "f"(k_norm_pair_x));
        float _mul_3;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(k_pair1_y), "f"(k_norm_pair_y));
        k_pair1_x = _mul_2;
        k_pair1_y = _mul_3;
        sK[m * 128 + k_col0] = k_pair0_x;
        sK[m * 128 + k_col0 + 1] = k_pair0_y;
        sK[m * 128 + k_col0 + 2] = k_pair1_x;
        sK[m * 128 + k_col0 + 3] = k_pair1_y;
#else
        float2 k_norm_pair = make_float2(k_norm, k_norm);
        k_pair0 = mul_f32x2(k_pair0, k_norm_pair);
        k_pair1 = mul_f32x2(k_pair1, k_norm_pair);
        sK[m * 128 + k_col0] = k_pair0.x;
        sK[m * 128 + k_col0 + 1] = k_pair0.y;
        sK[m * 128 + k_col0 + 2] = k_pair1.x;
        sK[m * 128 + k_col0 + 3] = k_pair1.y;
#endif
        long long v_index =
            (long long)n * v_stride_p0 + (long long)m * v_stride_p1 + (long long)h * v_stride_p2;
        int v_col = lane_local * 2;
        sV[m * 64 + v_col] = (float)v[v_index + (long long)v_col];
        sV[m * 64 + v_col + 1] = (float)v[v_index + (long long)v_col + 1];
        if (lane_local == 0) {
          long long b_index =
              (long long)n * b_stride_p0 + (long long)m * b_stride_p1 + (long long)h * b_stride_p2;
          float _exp_35 = __expf(-(float)b[b_index]);
          float _rcp_34 = approx_rcp(1.0f + _exp_35);
          sBeta[m] = _rcp_34;
        }
      }
    }
#pragma unroll 2
    for (int tt = 0; tt < 8; tt++) {
      int t = tt * 2 + warp_local;
      if (t < num_tokens) {
        long long q_index =
            (long long)n * q_stride_p0 + (long long)t * q_stride_p1 + (long long)h * q_stride_p2;
        {
          uint2 _vld_36;
          _vld_36 = *reinterpret_cast<const uint2*>(q + q_index + (long long)k_col0);
          uint32_t* _vpairs_36 = reinterpret_cast<uint32_t*>(&_vld_36);
#pragma unroll
          for (int _pair = 0; _pair < 2; _pair++) {
            (&r_q[0 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) << 16);
            (&r_q[0 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) & 0xffff0000u);
          }
        }
#if __CUDA_ARCH__ == 900
        float q_raw_pair0_x = r_q[0];
        float q_raw_pair0_y = r_q[1];
        float q_raw_pair1_x = r_q[2];
        float q_raw_pair1_y = r_q[3];
#else
        float2 q_raw_pair0 = make_float2(r_q[0], r_q[1]);
        float2 q_raw_pair1 = make_float2(r_q[2], r_q[3]);
#endif
        float q_norm = scale;
#if __CUDA_ARCH__ == 900
        float _fma_4;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_4)
            : "f"(q_raw_pair0_x), "f"(q_raw_pair0_x), "f"(0.0f));
        float _fma_5;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_5)
            : "f"(q_raw_pair0_y), "f"(q_raw_pair0_y), "f"(0.0f));
        float sum_q_pair_x = _fma_4;
        float sum_q_pair_y = _fma_5;
        float _fma_6;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_6)
            : "f"(q_raw_pair1_x), "f"(q_raw_pair1_x), "f"(sum_q_pair_x));
        float _fma_7;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_7)
            : "f"(q_raw_pair1_y), "f"(q_raw_pair1_y), "f"(sum_q_pair_y));
        sum_q_pair_x = _fma_6;
        sum_q_pair_y = _fma_7;
        float sum_q = sum_q_pair_x + sum_q_pair_y;
#else
        float2 sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair0, q_raw_pair0, make_float2(0.0f, 0.0f));
        sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair1, q_raw_pair1, sum_q_pair);
        float sum_q = sum_q_pair.x + sum_q_pair.y;
#endif
        float _warp_reduce_37 = sum_q;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
          _warp_reduce_37 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_37, offset);
        sum_q = _warp_reduce_37;
        float _rsqrt_38 = rsqrtf(sum_q + 1e-06f);
        q_norm = _rsqrt_38 * scale;
#if __CUDA_ARCH__ == 900
        float q_norm_pair_x = q_norm;
        float q_norm_pair_y = q_norm;
        float _mul_4;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(q_raw_pair0_x), "f"(q_norm_pair_x));
        float _mul_5;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(q_raw_pair0_y), "f"(q_norm_pair_y));
        float q_pair0_x = _mul_4;
        float q_pair0_y = _mul_5;
        float _mul_6;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(q_raw_pair1_x), "f"(q_norm_pair_x));
        float _mul_7;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(q_raw_pair1_y), "f"(q_norm_pair_y));
        float q_pair1_x = _mul_6;
        float q_pair1_y = _mul_7;
        sQ[t * 128 + k_col0] = q_pair0_x;
        sQ[t * 128 + k_col0 + 1] = q_pair0_y;
        sQ[t * 128 + k_col0 + 2] = q_pair1_x;
        sQ[t * 128 + k_col0 + 3] = q_pair1_y;
#else
        float2 q_norm_pair = make_float2(q_norm, q_norm);
        float2 q_pair0;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long*)&q_pair0)
            : "l"(*(const unsigned long long*)&q_raw_pair0),
              "l"(*(const unsigned long long*)&q_norm_pair));
        float2 q_pair1;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;"
            : "=l"(*(unsigned long long*)&q_pair1)
            : "l"(*(const unsigned long long*)&q_raw_pair1),
              "l"(*(const unsigned long long*)&q_norm_pair));
        sQ[t * 128 + k_col0] = q_pair0.x;
        sQ[t * 128 + k_col0 + 1] = q_pair0.y;
        sQ[t * 128 + k_col0 + 2] = q_pair1.x;
        sQ[t * 128 + k_col0 + 3] = q_pair1.y;
#endif
        if (lane_local == 0) {
          long long gate_index =
              (long long)n * a_stride_p0 + (long long)t * a_stride_p1 + (long long)h * a_stride_p2;
          float x = (float)a[gate_index] + dt_bias[h];
          float softplus_x = x;
          if (x <= 20.0f) {
            float _exp_40 = __expf(x);
            float _log2_39;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_39) : "f"(1.0f + _exp_40));
            softplus_x = _log2_39 * 0.6931472f;
          }
          float _exp_41 = __expf(A_log[h]);
          float g_log = (-_exp_41) * softplus_x;
          float _exp_42 = __expf(g_log);
          sAlpha[t] = _exp_42;
          if (use_ssm_state_indices != 0) {
            sSsm[t] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 +
                                        (long long)t * ssm_state_indices_stride_p1];
          }
          if (use_ssm_state_indices == 0) {
            sSsm[t] = -1;
          }
        }
      }
    }
    __syncthreads();
#if __CUDA_ARCH__ == 900
#else
    S2[0] = make_float2(S[0], S[1]);
    S2[1] = make_float2(S[2], S[3]);
    S2[2] = make_float2(S[4], S[5]);
    S2[3] = make_float2(S[6], S[7]);
    S2[4] = make_float2(S[8], S[9]);
    S2[5] = make_float2(S[10], S[11]);
    S2[6] = make_float2(S[12], S[13]);
    S2[7] = make_float2(S[14], S[15]);
    S2[8] = make_float2(S[16], S[17]);
    S2[9] = make_float2(S[18], S[19]);
    S2[10] = make_float2(S[20], S[21]);
    S2[11] = make_float2(S[22], S[23]);
    S2[12] = make_float2(S[24], S[25]);
    S2[13] = make_float2(S[26], S[27]);
    S2[14] = make_float2(S[28], S[29]);
    S2[15] = make_float2(S[30], S[31]);
    S2[16] = make_float2(S[32], S[33]);
    S2[17] = make_float2(S[34], S[35]);
    S2[18] = make_float2(S[36], S[37]);
    S2[19] = make_float2(S[38], S[39]);
    S2[20] = make_float2(S[40], S[41]);
    S2[21] = make_float2(S[42], S[43]);
    S2[22] = make_float2(S[44], S[45]);
    S2[23] = make_float2(S[46], S[47]);
    S2[24] = make_float2(S[48], S[49]);
    S2[25] = make_float2(S[50], S[51]);
    S2[26] = make_float2(S[52], S[53]);
    S2[27] = make_float2(S[54], S[55]);
    S2[28] = make_float2(S[56], S[57]);
    S2[29] = make_float2(S[58], S[59]);
    S2[30] = make_float2(S[60], S[61]);
    S2[31] = make_float2(S[62], S[63]);
    S2[32] = make_float2(S[64], S[65]);
    S2[33] = make_float2(S[66], S[67]);
    S2[34] = make_float2(S[68], S[69]);
    S2[35] = make_float2(S[70], S[71]);
    S2[36] = make_float2(S[72], S[73]);
    S2[37] = make_float2(S[74], S[75]);
    S2[38] = make_float2(S[76], S[77]);
    S2[39] = make_float2(S[78], S[79]);
    S2[40] = make_float2(S[80], S[81]);
    S2[41] = make_float2(S[82], S[83]);
    S2[42] = make_float2(S[84], S[85]);
    S2[43] = make_float2(S[86], S[87]);
    S2[44] = make_float2(S[88], S[89]);
    S2[45] = make_float2(S[90], S[91]);
    S2[46] = make_float2(S[92], S[93]);
    S2[47] = make_float2(S[94], S[95]);
    S2[48] = make_float2(S[96], S[97]);
    S2[49] = make_float2(S[98], S[99]);
    S2[50] = make_float2(S[100], S[101]);
    S2[51] = make_float2(S[102], S[103]);
    S2[52] = make_float2(S[104], S[105]);
    S2[53] = make_float2(S[106], S[107]);
    S2[54] = make_float2(S[108], S[109]);
    S2[55] = make_float2(S[110], S[111]);
    S2[56] = make_float2(S[112], S[113]);
    S2[57] = make_float2(S[114], S[115]);
    S2[58] = make_float2(S[116], S[117]);
    S2[59] = make_float2(S[118], S[119]);
    S2[60] = make_float2(S[120], S[121]);
    S2[61] = make_float2(S[122], S[123]);
    S2[62] = make_float2(S[124], S[125]);
    S2[63] = make_float2(S[126], S[127]);
#endif
    if (use_ssm_state_indices != 0) {
#pragma unroll 1
      for (int t_1 = 0; t_1 < num_tokens; t_1++) {
        float alphas = sAlpha[t_1];
#if __CUDA_ARCH__ == 900
        float alpha2s_x = alphas;
        float alpha2s_y = alphas;
        float _mul_8;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(S[0]), "f"(alpha2s_x));
        float _mul_9;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(S[1]), "f"(alpha2s_y));
        S[0] = _mul_8;
        S[1] = _mul_9;
        float _mul_10;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(S[2]), "f"(alpha2s_x));
        float _mul_11;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(S[3]), "f"(alpha2s_y));
        S[2] = _mul_10;
        S[3] = _mul_11;
        float _mul_12;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(S[4]), "f"(alpha2s_x));
        float _mul_13;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(S[5]), "f"(alpha2s_y));
        S[4] = _mul_12;
        S[5] = _mul_13;
        float _mul_14;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(S[6]), "f"(alpha2s_x));
        float _mul_15;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(S[7]), "f"(alpha2s_y));
        S[6] = _mul_14;
        S[7] = _mul_15;
        float _mul_16;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(S[8]), "f"(alpha2s_x));
        float _mul_17;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(S[9]), "f"(alpha2s_y));
        S[8] = _mul_16;
        S[9] = _mul_17;
        float _mul_18;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(S[10]), "f"(alpha2s_x));
        float _mul_19;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(S[11]), "f"(alpha2s_y));
        S[10] = _mul_18;
        S[11] = _mul_19;
        float _mul_20;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(S[12]), "f"(alpha2s_x));
        float _mul_21;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(S[13]), "f"(alpha2s_y));
        S[12] = _mul_20;
        S[13] = _mul_21;
        float _mul_22;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(S[14]), "f"(alpha2s_x));
        float _mul_23;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(S[15]), "f"(alpha2s_y));
        S[14] = _mul_22;
        S[15] = _mul_23;
        float _mul_24;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_24) : "f"(S[16]), "f"(alpha2s_x));
        float _mul_25;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_25) : "f"(S[17]), "f"(alpha2s_y));
        S[16] = _mul_24;
        S[17] = _mul_25;
        float _mul_26;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_26) : "f"(S[18]), "f"(alpha2s_x));
        float _mul_27;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_27) : "f"(S[19]), "f"(alpha2s_y));
        S[18] = _mul_26;
        S[19] = _mul_27;
        float _mul_28;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_28) : "f"(S[20]), "f"(alpha2s_x));
        float _mul_29;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_29) : "f"(S[21]), "f"(alpha2s_y));
        S[20] = _mul_28;
        S[21] = _mul_29;
        float _mul_30;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_30) : "f"(S[22]), "f"(alpha2s_x));
        float _mul_31;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_31) : "f"(S[23]), "f"(alpha2s_y));
        S[22] = _mul_30;
        S[23] = _mul_31;
        float _mul_32;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_32) : "f"(S[24]), "f"(alpha2s_x));
        float _mul_33;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_33) : "f"(S[25]), "f"(alpha2s_y));
        S[24] = _mul_32;
        S[25] = _mul_33;
        float _mul_34;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_34) : "f"(S[26]), "f"(alpha2s_x));
        float _mul_35;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_35) : "f"(S[27]), "f"(alpha2s_y));
        S[26] = _mul_34;
        S[27] = _mul_35;
        float _mul_36;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_36) : "f"(S[28]), "f"(alpha2s_x));
        float _mul_37;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_37) : "f"(S[29]), "f"(alpha2s_y));
        S[28] = _mul_36;
        S[29] = _mul_37;
        float _mul_38;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_38) : "f"(S[30]), "f"(alpha2s_x));
        float _mul_39;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(S[31]), "f"(alpha2s_y));
        S[30] = _mul_38;
        S[31] = _mul_39;
        float _mul_40;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_40) : "f"(S[32]), "f"(alpha2s_x));
        float _mul_41;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_41) : "f"(S[33]), "f"(alpha2s_y));
        S[32] = _mul_40;
        S[33] = _mul_41;
        float _mul_42;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_42) : "f"(S[34]), "f"(alpha2s_x));
        float _mul_43;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_43) : "f"(S[35]), "f"(alpha2s_y));
        S[34] = _mul_42;
        S[35] = _mul_43;
        float _mul_44;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_44) : "f"(S[36]), "f"(alpha2s_x));
        float _mul_45;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_45) : "f"(S[37]), "f"(alpha2s_y));
        S[36] = _mul_44;
        S[37] = _mul_45;
        float _mul_46;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_46) : "f"(S[38]), "f"(alpha2s_x));
        float _mul_47;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_47) : "f"(S[39]), "f"(alpha2s_y));
        S[38] = _mul_46;
        S[39] = _mul_47;
        float _mul_48;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_48) : "f"(S[40]), "f"(alpha2s_x));
        float _mul_49;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_49) : "f"(S[41]), "f"(alpha2s_y));
        S[40] = _mul_48;
        S[41] = _mul_49;
        float _mul_50;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_50) : "f"(S[42]), "f"(alpha2s_x));
        float _mul_51;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_51) : "f"(S[43]), "f"(alpha2s_y));
        S[42] = _mul_50;
        S[43] = _mul_51;
        float _mul_52;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_52) : "f"(S[44]), "f"(alpha2s_x));
        float _mul_53;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_53) : "f"(S[45]), "f"(alpha2s_y));
        S[44] = _mul_52;
        S[45] = _mul_53;
        float _mul_54;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_54) : "f"(S[46]), "f"(alpha2s_x));
        float _mul_55;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_55) : "f"(S[47]), "f"(alpha2s_y));
        S[46] = _mul_54;
        S[47] = _mul_55;
        float _mul_56;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_56) : "f"(S[48]), "f"(alpha2s_x));
        float _mul_57;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_57) : "f"(S[49]), "f"(alpha2s_y));
        S[48] = _mul_56;
        S[49] = _mul_57;
        float _mul_58;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_58) : "f"(S[50]), "f"(alpha2s_x));
        float _mul_59;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_59) : "f"(S[51]), "f"(alpha2s_y));
        S[50] = _mul_58;
        S[51] = _mul_59;
        float _mul_60;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_60) : "f"(S[52]), "f"(alpha2s_x));
        float _mul_61;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_61) : "f"(S[53]), "f"(alpha2s_y));
        S[52] = _mul_60;
        S[53] = _mul_61;
        float _mul_62;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_62) : "f"(S[54]), "f"(alpha2s_x));
        float _mul_63;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_63) : "f"(S[55]), "f"(alpha2s_y));
        S[54] = _mul_62;
        S[55] = _mul_63;
        float _mul_64;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_64) : "f"(S[56]), "f"(alpha2s_x));
        float _mul_65;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_65) : "f"(S[57]), "f"(alpha2s_y));
        S[56] = _mul_64;
        S[57] = _mul_65;
        float _mul_66;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_66) : "f"(S[58]), "f"(alpha2s_x));
        float _mul_67;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_67) : "f"(S[59]), "f"(alpha2s_y));
        S[58] = _mul_66;
        S[59] = _mul_67;
        float _mul_68;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_68) : "f"(S[60]), "f"(alpha2s_x));
        float _mul_69;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_69) : "f"(S[61]), "f"(alpha2s_y));
        S[60] = _mul_68;
        S[61] = _mul_69;
        float _mul_70;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_70) : "f"(S[62]), "f"(alpha2s_x));
        float _mul_71;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_71) : "f"(S[63]), "f"(alpha2s_y));
        S[62] = _mul_70;
        S[63] = _mul_71;
        float _mul_72;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_72) : "f"(S[64]), "f"(alpha2s_x));
        float _mul_73;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_73) : "f"(S[65]), "f"(alpha2s_y));
        S[64] = _mul_72;
        S[65] = _mul_73;
        float _mul_74;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_74) : "f"(S[66]), "f"(alpha2s_x));
        float _mul_75;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_75) : "f"(S[67]), "f"(alpha2s_y));
        S[66] = _mul_74;
        S[67] = _mul_75;
        float _mul_76;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_76) : "f"(S[68]), "f"(alpha2s_x));
        float _mul_77;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_77) : "f"(S[69]), "f"(alpha2s_y));
        S[68] = _mul_76;
        S[69] = _mul_77;
        float _mul_78;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_78) : "f"(S[70]), "f"(alpha2s_x));
        float _mul_79;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_79) : "f"(S[71]), "f"(alpha2s_y));
        S[70] = _mul_78;
        S[71] = _mul_79;
        float _mul_80;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_80) : "f"(S[72]), "f"(alpha2s_x));
        float _mul_81;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_81) : "f"(S[73]), "f"(alpha2s_y));
        S[72] = _mul_80;
        S[73] = _mul_81;
        float _mul_82;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_82) : "f"(S[74]), "f"(alpha2s_x));
        float _mul_83;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_83) : "f"(S[75]), "f"(alpha2s_y));
        S[74] = _mul_82;
        S[75] = _mul_83;
        float _mul_84;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_84) : "f"(S[76]), "f"(alpha2s_x));
        float _mul_85;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_85) : "f"(S[77]), "f"(alpha2s_y));
        S[76] = _mul_84;
        S[77] = _mul_85;
        float _mul_86;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_86) : "f"(S[78]), "f"(alpha2s_x));
        float _mul_87;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_87) : "f"(S[79]), "f"(alpha2s_y));
        S[78] = _mul_86;
        S[79] = _mul_87;
        float _mul_88;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_88) : "f"(S[80]), "f"(alpha2s_x));
        float _mul_89;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_89) : "f"(S[81]), "f"(alpha2s_y));
        S[80] = _mul_88;
        S[81] = _mul_89;
        float _mul_90;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_90) : "f"(S[82]), "f"(alpha2s_x));
        float _mul_91;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_91) : "f"(S[83]), "f"(alpha2s_y));
        S[82] = _mul_90;
        S[83] = _mul_91;
        float _mul_92;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_92) : "f"(S[84]), "f"(alpha2s_x));
        float _mul_93;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_93) : "f"(S[85]), "f"(alpha2s_y));
        S[84] = _mul_92;
        S[85] = _mul_93;
        float _mul_94;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_94) : "f"(S[86]), "f"(alpha2s_x));
        float _mul_95;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_95) : "f"(S[87]), "f"(alpha2s_y));
        S[86] = _mul_94;
        S[87] = _mul_95;
        float _mul_96;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_96) : "f"(S[88]), "f"(alpha2s_x));
        float _mul_97;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_97) : "f"(S[89]), "f"(alpha2s_y));
        S[88] = _mul_96;
        S[89] = _mul_97;
        float _mul_98;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_98) : "f"(S[90]), "f"(alpha2s_x));
        float _mul_99;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_99) : "f"(S[91]), "f"(alpha2s_y));
        S[90] = _mul_98;
        S[91] = _mul_99;
        float _mul_100;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_100) : "f"(S[92]), "f"(alpha2s_x));
        float _mul_101;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_101) : "f"(S[93]), "f"(alpha2s_y));
        S[92] = _mul_100;
        S[93] = _mul_101;
        float _mul_102;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_102) : "f"(S[94]), "f"(alpha2s_x));
        float _mul_103;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_103) : "f"(S[95]), "f"(alpha2s_y));
        S[94] = _mul_102;
        S[95] = _mul_103;
        float _mul_104;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_104) : "f"(S[96]), "f"(alpha2s_x));
        float _mul_105;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_105) : "f"(S[97]), "f"(alpha2s_y));
        S[96] = _mul_104;
        S[97] = _mul_105;
        float _mul_106;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_106) : "f"(S[98]), "f"(alpha2s_x));
        float _mul_107;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_107) : "f"(S[99]), "f"(alpha2s_y));
        S[98] = _mul_106;
        S[99] = _mul_107;
        float _mul_108;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_108) : "f"(S[100]), "f"(alpha2s_x));
        float _mul_109;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_109) : "f"(S[101]), "f"(alpha2s_y));
        S[100] = _mul_108;
        S[101] = _mul_109;
        float _mul_110;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_110) : "f"(S[102]), "f"(alpha2s_x));
        float _mul_111;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_111) : "f"(S[103]), "f"(alpha2s_y));
        S[102] = _mul_110;
        S[103] = _mul_111;
        float _mul_112;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_112) : "f"(S[104]), "f"(alpha2s_x));
        float _mul_113;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_113) : "f"(S[105]), "f"(alpha2s_y));
        S[104] = _mul_112;
        S[105] = _mul_113;
        float _mul_114;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_114) : "f"(S[106]), "f"(alpha2s_x));
        float _mul_115;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_115) : "f"(S[107]), "f"(alpha2s_y));
        S[106] = _mul_114;
        S[107] = _mul_115;
        float _mul_116;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_116) : "f"(S[108]), "f"(alpha2s_x));
        float _mul_117;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_117) : "f"(S[109]), "f"(alpha2s_y));
        S[108] = _mul_116;
        S[109] = _mul_117;
        float _mul_118;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_118) : "f"(S[110]), "f"(alpha2s_x));
        float _mul_119;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_119) : "f"(S[111]), "f"(alpha2s_y));
        S[110] = _mul_118;
        S[111] = _mul_119;
        float _mul_120;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_120) : "f"(S[112]), "f"(alpha2s_x));
        float _mul_121;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_121) : "f"(S[113]), "f"(alpha2s_y));
        S[112] = _mul_120;
        S[113] = _mul_121;
        float _mul_122;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_122) : "f"(S[114]), "f"(alpha2s_x));
        float _mul_123;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_123) : "f"(S[115]), "f"(alpha2s_y));
        S[114] = _mul_122;
        S[115] = _mul_123;
        float _mul_124;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_124) : "f"(S[116]), "f"(alpha2s_x));
        float _mul_125;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_125) : "f"(S[117]), "f"(alpha2s_y));
        S[116] = _mul_124;
        S[117] = _mul_125;
        float _mul_126;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_126) : "f"(S[118]), "f"(alpha2s_x));
        float _mul_127;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_127) : "f"(S[119]), "f"(alpha2s_y));
        S[118] = _mul_126;
        S[119] = _mul_127;
        float _mul_128;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_128) : "f"(S[120]), "f"(alpha2s_x));
        float _mul_129;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_129) : "f"(S[121]), "f"(alpha2s_y));
        S[120] = _mul_128;
        S[121] = _mul_129;
        float _mul_130;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_130) : "f"(S[122]), "f"(alpha2s_x));
        float _mul_131;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_131) : "f"(S[123]), "f"(alpha2s_y));
        S[122] = _mul_130;
        S[123] = _mul_131;
        float _mul_132;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_132) : "f"(S[124]), "f"(alpha2s_x));
        float _mul_133;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_133) : "f"(S[125]), "f"(alpha2s_y));
        S[124] = _mul_132;
        S[125] = _mul_133;
        float _mul_134;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_134) : "f"(S[126]), "f"(alpha2s_x));
        float _mul_135;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_135) : "f"(S[127]), "f"(alpha2s_y));
        S[126] = _mul_134;
        S[127] = _mul_135;
#else
        float2 alpha2s = make_float2(alphas, alphas);
        S2[0] = mul_f32x2(S2[0], alpha2s);
        S2[1] = mul_f32x2(S2[1], alpha2s);
        S2[2] = mul_f32x2(S2[2], alpha2s);
        S2[3] = mul_f32x2(S2[3], alpha2s);
        S2[4] = mul_f32x2(S2[4], alpha2s);
        S2[5] = mul_f32x2(S2[5], alpha2s);
        S2[6] = mul_f32x2(S2[6], alpha2s);
        S2[7] = mul_f32x2(S2[7], alpha2s);
        S2[8] = mul_f32x2(S2[8], alpha2s);
        S2[9] = mul_f32x2(S2[9], alpha2s);
        S2[10] = mul_f32x2(S2[10], alpha2s);
        S2[11] = mul_f32x2(S2[11], alpha2s);
        S2[12] = mul_f32x2(S2[12], alpha2s);
        S2[13] = mul_f32x2(S2[13], alpha2s);
        S2[14] = mul_f32x2(S2[14], alpha2s);
        S2[15] = mul_f32x2(S2[15], alpha2s);
        S2[16] = mul_f32x2(S2[16], alpha2s);
        S2[17] = mul_f32x2(S2[17], alpha2s);
        S2[18] = mul_f32x2(S2[18], alpha2s);
        S2[19] = mul_f32x2(S2[19], alpha2s);
        S2[20] = mul_f32x2(S2[20], alpha2s);
        S2[21] = mul_f32x2(S2[21], alpha2s);
        S2[22] = mul_f32x2(S2[22], alpha2s);
        S2[23] = mul_f32x2(S2[23], alpha2s);
        S2[24] = mul_f32x2(S2[24], alpha2s);
        S2[25] = mul_f32x2(S2[25], alpha2s);
        S2[26] = mul_f32x2(S2[26], alpha2s);
        S2[27] = mul_f32x2(S2[27], alpha2s);
        S2[28] = mul_f32x2(S2[28], alpha2s);
        S2[29] = mul_f32x2(S2[29], alpha2s);
        S2[30] = mul_f32x2(S2[30], alpha2s);
        S2[31] = mul_f32x2(S2[31], alpha2s);
        S2[32] = mul_f32x2(S2[32], alpha2s);
        S2[33] = mul_f32x2(S2[33], alpha2s);
        S2[34] = mul_f32x2(S2[34], alpha2s);
        S2[35] = mul_f32x2(S2[35], alpha2s);
        S2[36] = mul_f32x2(S2[36], alpha2s);
        S2[37] = mul_f32x2(S2[37], alpha2s);
        S2[38] = mul_f32x2(S2[38], alpha2s);
        S2[39] = mul_f32x2(S2[39], alpha2s);
        S2[40] = mul_f32x2(S2[40], alpha2s);
        S2[41] = mul_f32x2(S2[41], alpha2s);
        S2[42] = mul_f32x2(S2[42], alpha2s);
        S2[43] = mul_f32x2(S2[43], alpha2s);
        S2[44] = mul_f32x2(S2[44], alpha2s);
        S2[45] = mul_f32x2(S2[45], alpha2s);
        S2[46] = mul_f32x2(S2[46], alpha2s);
        S2[47] = mul_f32x2(S2[47], alpha2s);
        S2[48] = mul_f32x2(S2[48], alpha2s);
        S2[49] = mul_f32x2(S2[49], alpha2s);
        S2[50] = mul_f32x2(S2[50], alpha2s);
        S2[51] = mul_f32x2(S2[51], alpha2s);
        S2[52] = mul_f32x2(S2[52], alpha2s);
        S2[53] = mul_f32x2(S2[53], alpha2s);
        S2[54] = mul_f32x2(S2[54], alpha2s);
        S2[55] = mul_f32x2(S2[55], alpha2s);
        S2[56] = mul_f32x2(S2[56], alpha2s);
        S2[57] = mul_f32x2(S2[57], alpha2s);
        S2[58] = mul_f32x2(S2[58], alpha2s);
        S2[59] = mul_f32x2(S2[59], alpha2s);
        S2[60] = mul_f32x2(S2[60], alpha2s);
        S2[61] = mul_f32x2(S2[61], alpha2s);
        S2[62] = mul_f32x2(S2[62], alpha2s);
        S2[63] = mul_f32x2(S2[63], alpha2s);
#endif
        int ms0 = t_1 * 3;
        unsigned int kbases0 = sK_addr + (unsigned int)(ms0 * 512) + lane_byte;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                     : "r"(kbases0));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                     : "r"(kbases0 + 64));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                     : "r"(kbases0 + 128));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                     : "r"(kbases0 + 192));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[16])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 3]))
                     : "r"(kbases0 + 256));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[20])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 3]))
                     : "r"(kbases0 + 320));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[24])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 3]))
                     : "r"(kbases0 + 384));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[28])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 3]))
                     : "r"(kbases0 + 448));
#if __CUDA_ARCH__ == 900
#else
        k2[0] = make_float2(k_regs[0], k_regs[1]);
        k2[1] = make_float2(k_regs[2], k_regs[3]);
        k2[2] = make_float2(k_regs[4], k_regs[5]);
        k2[3] = make_float2(k_regs[6], k_regs[7]);
        k2[4] = make_float2(k_regs[8], k_regs[9]);
        k2[5] = make_float2(k_regs[10], k_regs[11]);
        k2[6] = make_float2(k_regs[12], k_regs[13]);
        k2[7] = make_float2(k_regs[14], k_regs[15]);
        k2[8] = make_float2(k_regs[16], k_regs[17]);
        k2[9] = make_float2(k_regs[18], k_regs[19]);
        k2[10] = make_float2(k_regs[20], k_regs[21]);
        k2[11] = make_float2(k_regs[22], k_regs[23]);
        k2[12] = make_float2(k_regs[24], k_regs[25]);
        k2[13] = make_float2(k_regs[26], k_regs[27]);
        k2[14] = make_float2(k_regs[28], k_regs[29]);
        k2[15] = make_float2(k_regs[30], k_regs[31]);
#endif
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                     : "r"(sV_addr + (unsigned int)(ms0 * 256) + group_byte));
        float betas0 = sBeta[ms0];
#if __CUDA_ARCH__ == 900
        float accs0_0_0_x = 0.0f;
        float accs0_0_0_y = 0.0f;
        float accs0_0_1_x = 0.0f;
        float accs0_0_1_y = 0.0f;
        float _fma_8;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_8)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accs0_0_0_x));
        float _fma_9;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_9)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accs0_0_0_y));
        accs0_0_0_x = _fma_8;
        accs0_0_0_y = _fma_9;
        float _fma_10;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_10)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accs0_0_1_x));
        float _fma_11;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_11)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accs0_0_1_y));
        accs0_0_1_x = _fma_10;
        accs0_0_1_y = _fma_11;
        float _fma_12;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_12)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accs0_0_0_x));
        float _fma_13;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_13)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accs0_0_0_y));
        accs0_0_0_x = _fma_12;
        accs0_0_0_y = _fma_13;
        float _fma_14;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_14)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accs0_0_1_x));
        float _fma_15;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_15)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accs0_0_1_y));
        accs0_0_1_x = _fma_14;
        accs0_0_1_y = _fma_15;
        float _fma_16;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_16)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accs0_0_0_x));
        float _fma_17;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_17)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accs0_0_0_y));
        accs0_0_0_x = _fma_16;
        accs0_0_0_y = _fma_17;
        float _fma_18;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_18)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accs0_0_1_x));
        float _fma_19;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_19)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accs0_0_1_y));
        accs0_0_1_x = _fma_18;
        accs0_0_1_y = _fma_19;
        float _fma_20;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_20)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accs0_0_0_x));
        float _fma_21;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_21)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accs0_0_0_y));
        accs0_0_0_x = _fma_20;
        accs0_0_0_y = _fma_21;
        float _fma_22;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_22)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accs0_0_1_x));
        float _fma_23;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_23)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accs0_0_1_y));
        accs0_0_1_x = _fma_22;
        accs0_0_1_y = _fma_23;
        float _fma_24;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_24)
            : "f"(S[16]), "f"(k_regs[16]), "f"(accs0_0_0_x));
        float _fma_25;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_25)
            : "f"(S[17]), "f"(k_regs[17]), "f"(accs0_0_0_y));
        accs0_0_0_x = _fma_24;
        accs0_0_0_y = _fma_25;
        float _fma_26;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_26)
            : "f"(S[18]), "f"(k_regs[18]), "f"(accs0_0_1_x));
        float _fma_27;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_27)
            : "f"(S[19]), "f"(k_regs[19]), "f"(accs0_0_1_y));
        accs0_0_1_x = _fma_26;
        accs0_0_1_y = _fma_27;
        float _fma_28;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_28)
            : "f"(S[20]), "f"(k_regs[20]), "f"(accs0_0_0_x));
        float _fma_29;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_29)
            : "f"(S[21]), "f"(k_regs[21]), "f"(accs0_0_0_y));
        accs0_0_0_x = _fma_28;
        accs0_0_0_y = _fma_29;
        float _fma_30;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_30)
            : "f"(S[22]), "f"(k_regs[22]), "f"(accs0_0_1_x));
        float _fma_31;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_31)
            : "f"(S[23]), "f"(k_regs[23]), "f"(accs0_0_1_y));
        accs0_0_1_x = _fma_30;
        accs0_0_1_y = _fma_31;
        float _fma_32;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_32)
            : "f"(S[24]), "f"(k_regs[24]), "f"(accs0_0_0_x));
        float _fma_33;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_33)
            : "f"(S[25]), "f"(k_regs[25]), "f"(accs0_0_0_y));
        accs0_0_0_x = _fma_32;
        accs0_0_0_y = _fma_33;
        float _fma_34;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_34)
            : "f"(S[26]), "f"(k_regs[26]), "f"(accs0_0_1_x));
        float _fma_35;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_35)
            : "f"(S[27]), "f"(k_regs[27]), "f"(accs0_0_1_y));
        accs0_0_1_x = _fma_34;
        accs0_0_1_y = _fma_35;
        float _fma_36;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_36)
            : "f"(S[28]), "f"(k_regs[28]), "f"(accs0_0_0_x));
        float _fma_37;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_37)
            : "f"(S[29]), "f"(k_regs[29]), "f"(accs0_0_0_y));
        accs0_0_0_x = _fma_36;
        accs0_0_0_y = _fma_37;
        float _fma_38;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_38)
            : "f"(S[30]), "f"(k_regs[30]), "f"(accs0_0_1_x));
        float _fma_39;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_39)
            : "f"(S[31]), "f"(k_regs[31]), "f"(accs0_0_1_y));
        accs0_0_1_x = _fma_38;
        accs0_0_1_y = _fma_39;
        float _add_43;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(accs0_0_0_x), "f"(accs0_0_1_x));
        float _add_44;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(accs0_0_0_y), "f"(accs0_0_1_y));
        float dots0_0 = _add_43 + _add_44;
        float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 1);
        dots0_0 = dots0_0 + _shfl_xor_sync_45;
        float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 2);
        dots0_0 = dots0_0 + _shfl_xor_sync_46;
#else
        float2 accs0_0_0 = make_float2(0.0f, 0.0f);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accs0_0_0);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[1], k2[1], accs0_0_0);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accs0_0_0);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[3], k2[3], accs0_0_0);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accs0_0_0);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[5], k2[5], accs0_0_0);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accs0_0_0);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[7], k2[7], accs0_0_0);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[8], k2[8], accs0_0_0);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[9], k2[9], accs0_0_0);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[10], k2[10], accs0_0_0);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[11], k2[11], accs0_0_0);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[12], k2[12], accs0_0_0);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[13], k2[13], accs0_0_0);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[14], k2[14], accs0_0_0);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[15], k2[15], accs0_0_0);
        float dots0_0 = accs0_0_0.x + accs0_0_0.y;
        float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 1);
        dots0_0 = dots0_0 + _shfl_xor_sync_43;
        float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 2);
        dots0_0 = dots0_0 + _shfl_xor_sync_44;
#endif
        float vns0_0 = (v_regs[0] - dots0_0) * betas0;
#if __CUDA_ARCH__ == 900
        float vn2s0_0_x = vns0_0;
        float vn2s0_0_y = vns0_0;
        float _fma_40;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_40)
            : "f"(k_regs[0]), "f"(vn2s0_0_x), "f"(S[0]));
        float _fma_41;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_41)
            : "f"(k_regs[1]), "f"(vn2s0_0_y), "f"(S[1]));
        S[0] = _fma_40;
        S[1] = _fma_41;
        float _fma_42;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_42)
            : "f"(k_regs[2]), "f"(vn2s0_0_x), "f"(S[2]));
        float _fma_43;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_43)
            : "f"(k_regs[3]), "f"(vn2s0_0_y), "f"(S[3]));
        S[2] = _fma_42;
        S[3] = _fma_43;
        float _fma_44;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_44)
            : "f"(k_regs[4]), "f"(vn2s0_0_x), "f"(S[4]));
        float _fma_45;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_45)
            : "f"(k_regs[5]), "f"(vn2s0_0_y), "f"(S[5]));
        S[4] = _fma_44;
        S[5] = _fma_45;
        float _fma_46;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_46)
            : "f"(k_regs[6]), "f"(vn2s0_0_x), "f"(S[6]));
        float _fma_47;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_47)
            : "f"(k_regs[7]), "f"(vn2s0_0_y), "f"(S[7]));
        S[6] = _fma_46;
        S[7] = _fma_47;
        float _fma_48;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_48)
            : "f"(k_regs[8]), "f"(vn2s0_0_x), "f"(S[8]));
        float _fma_49;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_49)
            : "f"(k_regs[9]), "f"(vn2s0_0_y), "f"(S[9]));
        S[8] = _fma_48;
        S[9] = _fma_49;
        float _fma_50;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_50)
            : "f"(k_regs[10]), "f"(vn2s0_0_x), "f"(S[10]));
        float _fma_51;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_51)
            : "f"(k_regs[11]), "f"(vn2s0_0_y), "f"(S[11]));
        S[10] = _fma_50;
        S[11] = _fma_51;
        float _fma_52;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_52)
            : "f"(k_regs[12]), "f"(vn2s0_0_x), "f"(S[12]));
        float _fma_53;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_53)
            : "f"(k_regs[13]), "f"(vn2s0_0_y), "f"(S[13]));
        S[12] = _fma_52;
        S[13] = _fma_53;
        float _fma_54;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_54)
            : "f"(k_regs[14]), "f"(vn2s0_0_x), "f"(S[14]));
        float _fma_55;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_55)
            : "f"(k_regs[15]), "f"(vn2s0_0_y), "f"(S[15]));
        S[14] = _fma_54;
        S[15] = _fma_55;
        float _fma_56;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_56)
            : "f"(k_regs[16]), "f"(vn2s0_0_x), "f"(S[16]));
        float _fma_57;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_57)
            : "f"(k_regs[17]), "f"(vn2s0_0_y), "f"(S[17]));
        S[16] = _fma_56;
        S[17] = _fma_57;
        float _fma_58;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_58)
            : "f"(k_regs[18]), "f"(vn2s0_0_x), "f"(S[18]));
        float _fma_59;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_59)
            : "f"(k_regs[19]), "f"(vn2s0_0_y), "f"(S[19]));
        S[18] = _fma_58;
        S[19] = _fma_59;
        float _fma_60;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_60)
            : "f"(k_regs[20]), "f"(vn2s0_0_x), "f"(S[20]));
        float _fma_61;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_61)
            : "f"(k_regs[21]), "f"(vn2s0_0_y), "f"(S[21]));
        S[20] = _fma_60;
        S[21] = _fma_61;
        float _fma_62;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_62)
            : "f"(k_regs[22]), "f"(vn2s0_0_x), "f"(S[22]));
        float _fma_63;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_63)
            : "f"(k_regs[23]), "f"(vn2s0_0_y), "f"(S[23]));
        S[22] = _fma_62;
        S[23] = _fma_63;
        float _fma_64;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_64)
            : "f"(k_regs[24]), "f"(vn2s0_0_x), "f"(S[24]));
        float _fma_65;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_65)
            : "f"(k_regs[25]), "f"(vn2s0_0_y), "f"(S[25]));
        S[24] = _fma_64;
        S[25] = _fma_65;
        float _fma_66;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_66)
            : "f"(k_regs[26]), "f"(vn2s0_0_x), "f"(S[26]));
        float _fma_67;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_67)
            : "f"(k_regs[27]), "f"(vn2s0_0_y), "f"(S[27]));
        S[26] = _fma_66;
        S[27] = _fma_67;
        float _fma_68;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_68)
            : "f"(k_regs[28]), "f"(vn2s0_0_x), "f"(S[28]));
        float _fma_69;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_69)
            : "f"(k_regs[29]), "f"(vn2s0_0_y), "f"(S[29]));
        S[28] = _fma_68;
        S[29] = _fma_69;
        float _fma_70;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_70)
            : "f"(k_regs[30]), "f"(vn2s0_0_x), "f"(S[30]));
        float _fma_71;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_71)
            : "f"(k_regs[31]), "f"(vn2s0_0_y), "f"(S[31]));
        S[30] = _fma_70;
        S[31] = _fma_71;
        float accs0_1_0_x = 0.0f;
        float accs0_1_0_y = 0.0f;
        float accs0_1_1_x = 0.0f;
        float accs0_1_1_y = 0.0f;
        float _fma_72;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_72)
            : "f"(S[32]), "f"(k_regs[0]), "f"(accs0_1_0_x));
        float _fma_73;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_73)
            : "f"(S[33]), "f"(k_regs[1]), "f"(accs0_1_0_y));
        accs0_1_0_x = _fma_72;
        accs0_1_0_y = _fma_73;
        float _fma_74;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_74)
            : "f"(S[34]), "f"(k_regs[2]), "f"(accs0_1_1_x));
        float _fma_75;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_75)
            : "f"(S[35]), "f"(k_regs[3]), "f"(accs0_1_1_y));
        accs0_1_1_x = _fma_74;
        accs0_1_1_y = _fma_75;
        float _fma_76;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_76)
            : "f"(S[36]), "f"(k_regs[4]), "f"(accs0_1_0_x));
        float _fma_77;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_77)
            : "f"(S[37]), "f"(k_regs[5]), "f"(accs0_1_0_y));
        accs0_1_0_x = _fma_76;
        accs0_1_0_y = _fma_77;
        float _fma_78;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_78)
            : "f"(S[38]), "f"(k_regs[6]), "f"(accs0_1_1_x));
        float _fma_79;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_79)
            : "f"(S[39]), "f"(k_regs[7]), "f"(accs0_1_1_y));
        accs0_1_1_x = _fma_78;
        accs0_1_1_y = _fma_79;
        float _fma_80;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_80)
            : "f"(S[40]), "f"(k_regs[8]), "f"(accs0_1_0_x));
        float _fma_81;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_81)
            : "f"(S[41]), "f"(k_regs[9]), "f"(accs0_1_0_y));
        accs0_1_0_x = _fma_80;
        accs0_1_0_y = _fma_81;
        float _fma_82;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_82)
            : "f"(S[42]), "f"(k_regs[10]), "f"(accs0_1_1_x));
        float _fma_83;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_83)
            : "f"(S[43]), "f"(k_regs[11]), "f"(accs0_1_1_y));
        accs0_1_1_x = _fma_82;
        accs0_1_1_y = _fma_83;
        float _fma_84;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_84)
            : "f"(S[44]), "f"(k_regs[12]), "f"(accs0_1_0_x));
        float _fma_85;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_85)
            : "f"(S[45]), "f"(k_regs[13]), "f"(accs0_1_0_y));
        accs0_1_0_x = _fma_84;
        accs0_1_0_y = _fma_85;
        float _fma_86;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_86)
            : "f"(S[46]), "f"(k_regs[14]), "f"(accs0_1_1_x));
        float _fma_87;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_87)
            : "f"(S[47]), "f"(k_regs[15]), "f"(accs0_1_1_y));
        accs0_1_1_x = _fma_86;
        accs0_1_1_y = _fma_87;
        float _fma_88;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_88)
            : "f"(S[48]), "f"(k_regs[16]), "f"(accs0_1_0_x));
        float _fma_89;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_89)
            : "f"(S[49]), "f"(k_regs[17]), "f"(accs0_1_0_y));
        accs0_1_0_x = _fma_88;
        accs0_1_0_y = _fma_89;
        float _fma_90;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_90)
            : "f"(S[50]), "f"(k_regs[18]), "f"(accs0_1_1_x));
        float _fma_91;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_91)
            : "f"(S[51]), "f"(k_regs[19]), "f"(accs0_1_1_y));
        accs0_1_1_x = _fma_90;
        accs0_1_1_y = _fma_91;
        float _fma_92;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_92)
            : "f"(S[52]), "f"(k_regs[20]), "f"(accs0_1_0_x));
        float _fma_93;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_93)
            : "f"(S[53]), "f"(k_regs[21]), "f"(accs0_1_0_y));
        accs0_1_0_x = _fma_92;
        accs0_1_0_y = _fma_93;
        float _fma_94;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_94)
            : "f"(S[54]), "f"(k_regs[22]), "f"(accs0_1_1_x));
        float _fma_95;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_95)
            : "f"(S[55]), "f"(k_regs[23]), "f"(accs0_1_1_y));
        accs0_1_1_x = _fma_94;
        accs0_1_1_y = _fma_95;
        float _fma_96;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_96)
            : "f"(S[56]), "f"(k_regs[24]), "f"(accs0_1_0_x));
        float _fma_97;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_97)
            : "f"(S[57]), "f"(k_regs[25]), "f"(accs0_1_0_y));
        accs0_1_0_x = _fma_96;
        accs0_1_0_y = _fma_97;
        float _fma_98;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_98)
            : "f"(S[58]), "f"(k_regs[26]), "f"(accs0_1_1_x));
        float _fma_99;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_99)
            : "f"(S[59]), "f"(k_regs[27]), "f"(accs0_1_1_y));
        accs0_1_1_x = _fma_98;
        accs0_1_1_y = _fma_99;
        float _fma_100;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_100)
            : "f"(S[60]), "f"(k_regs[28]), "f"(accs0_1_0_x));
        float _fma_101;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_101)
            : "f"(S[61]), "f"(k_regs[29]), "f"(accs0_1_0_y));
        accs0_1_0_x = _fma_100;
        accs0_1_0_y = _fma_101;
        float _fma_102;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_102)
            : "f"(S[62]), "f"(k_regs[30]), "f"(accs0_1_1_x));
        float _fma_103;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_103)
            : "f"(S[63]), "f"(k_regs[31]), "f"(accs0_1_1_y));
        accs0_1_1_x = _fma_102;
        accs0_1_1_y = _fma_103;
        float _add_47;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(accs0_1_0_x), "f"(accs0_1_1_x));
        float _add_48;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(accs0_1_0_y), "f"(accs0_1_1_y));
        float dots0_1 = _add_47 + _add_48;
        float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 1);
        dots0_1 = dots0_1 + _shfl_xor_sync_49;
        float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 2);
        dots0_1 = dots0_1 + _shfl_xor_sync_50;
#else
        float2 vn2s0_0 = make_float2(vns0_0, vns0_0);
        S2[0] = fma_f32x2_rn_ftz(k2[0], vn2s0_0, S2[0]);
        S2[1] = fma_f32x2_rn_ftz(k2[1], vn2s0_0, S2[1]);
        S2[2] = fma_f32x2_rn_ftz(k2[2], vn2s0_0, S2[2]);
        S2[3] = fma_f32x2_rn_ftz(k2[3], vn2s0_0, S2[3]);
        S2[4] = fma_f32x2_rn_ftz(k2[4], vn2s0_0, S2[4]);
        S2[5] = fma_f32x2_rn_ftz(k2[5], vn2s0_0, S2[5]);
        S2[6] = fma_f32x2_rn_ftz(k2[6], vn2s0_0, S2[6]);
        S2[7] = fma_f32x2_rn_ftz(k2[7], vn2s0_0, S2[7]);
        S2[8] = fma_f32x2_rn_ftz(k2[8], vn2s0_0, S2[8]);
        S2[9] = fma_f32x2_rn_ftz(k2[9], vn2s0_0, S2[9]);
        S2[10] = fma_f32x2_rn_ftz(k2[10], vn2s0_0, S2[10]);
        S2[11] = fma_f32x2_rn_ftz(k2[11], vn2s0_0, S2[11]);
        S2[12] = fma_f32x2_rn_ftz(k2[12], vn2s0_0, S2[12]);
        S2[13] = fma_f32x2_rn_ftz(k2[13], vn2s0_0, S2[13]);
        S2[14] = fma_f32x2_rn_ftz(k2[14], vn2s0_0, S2[14]);
        S2[15] = fma_f32x2_rn_ftz(k2[15], vn2s0_0, S2[15]);
        float2 accs0_1_0 = make_float2(0.0f, 0.0f);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[24], k2[8], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[25], k2[9], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[26], k2[10], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[27], k2[11], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[28], k2[12], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[29], k2[13], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[30], k2[14], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[31], k2[15], accs0_1_0);
        float dots0_1 = accs0_1_0.x + accs0_1_0.y;
        float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 1);
        dots0_1 = dots0_1 + _shfl_xor_sync_45;
        float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 2);
        dots0_1 = dots0_1 + _shfl_xor_sync_46;
#endif
        float vns0_1 = (v_regs[1] - dots0_1) * betas0;
#if __CUDA_ARCH__ == 900
        float vn2s0_1_x = vns0_1;
        float vn2s0_1_y = vns0_1;
        float _fma_104;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_104)
            : "f"(k_regs[0]), "f"(vn2s0_1_x), "f"(S[32]));
        float _fma_105;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_105)
            : "f"(k_regs[1]), "f"(vn2s0_1_y), "f"(S[33]));
        S[32] = _fma_104;
        S[33] = _fma_105;
        float _fma_106;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_106)
            : "f"(k_regs[2]), "f"(vn2s0_1_x), "f"(S[34]));
        float _fma_107;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_107)
            : "f"(k_regs[3]), "f"(vn2s0_1_y), "f"(S[35]));
        S[34] = _fma_106;
        S[35] = _fma_107;
        float _fma_108;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_108)
            : "f"(k_regs[4]), "f"(vn2s0_1_x), "f"(S[36]));
        float _fma_109;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_109)
            : "f"(k_regs[5]), "f"(vn2s0_1_y), "f"(S[37]));
        S[36] = _fma_108;
        S[37] = _fma_109;
        float _fma_110;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_110)
            : "f"(k_regs[6]), "f"(vn2s0_1_x), "f"(S[38]));
        float _fma_111;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_111)
            : "f"(k_regs[7]), "f"(vn2s0_1_y), "f"(S[39]));
        S[38] = _fma_110;
        S[39] = _fma_111;
        float _fma_112;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_112)
            : "f"(k_regs[8]), "f"(vn2s0_1_x), "f"(S[40]));
        float _fma_113;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_113)
            : "f"(k_regs[9]), "f"(vn2s0_1_y), "f"(S[41]));
        S[40] = _fma_112;
        S[41] = _fma_113;
        float _fma_114;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_114)
            : "f"(k_regs[10]), "f"(vn2s0_1_x), "f"(S[42]));
        float _fma_115;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_115)
            : "f"(k_regs[11]), "f"(vn2s0_1_y), "f"(S[43]));
        S[42] = _fma_114;
        S[43] = _fma_115;
        float _fma_116;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_116)
            : "f"(k_regs[12]), "f"(vn2s0_1_x), "f"(S[44]));
        float _fma_117;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_117)
            : "f"(k_regs[13]), "f"(vn2s0_1_y), "f"(S[45]));
        S[44] = _fma_116;
        S[45] = _fma_117;
        float _fma_118;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_118)
            : "f"(k_regs[14]), "f"(vn2s0_1_x), "f"(S[46]));
        float _fma_119;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_119)
            : "f"(k_regs[15]), "f"(vn2s0_1_y), "f"(S[47]));
        S[46] = _fma_118;
        S[47] = _fma_119;
        float _fma_120;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_120)
            : "f"(k_regs[16]), "f"(vn2s0_1_x), "f"(S[48]));
        float _fma_121;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_121)
            : "f"(k_regs[17]), "f"(vn2s0_1_y), "f"(S[49]));
        S[48] = _fma_120;
        S[49] = _fma_121;
        float _fma_122;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_122)
            : "f"(k_regs[18]), "f"(vn2s0_1_x), "f"(S[50]));
        float _fma_123;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_123)
            : "f"(k_regs[19]), "f"(vn2s0_1_y), "f"(S[51]));
        S[50] = _fma_122;
        S[51] = _fma_123;
        float _fma_124;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_124)
            : "f"(k_regs[20]), "f"(vn2s0_1_x), "f"(S[52]));
        float _fma_125;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_125)
            : "f"(k_regs[21]), "f"(vn2s0_1_y), "f"(S[53]));
        S[52] = _fma_124;
        S[53] = _fma_125;
        float _fma_126;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_126)
            : "f"(k_regs[22]), "f"(vn2s0_1_x), "f"(S[54]));
        float _fma_127;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_127)
            : "f"(k_regs[23]), "f"(vn2s0_1_y), "f"(S[55]));
        S[54] = _fma_126;
        S[55] = _fma_127;
        float _fma_128;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_128)
            : "f"(k_regs[24]), "f"(vn2s0_1_x), "f"(S[56]));
        float _fma_129;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_129)
            : "f"(k_regs[25]), "f"(vn2s0_1_y), "f"(S[57]));
        S[56] = _fma_128;
        S[57] = _fma_129;
        float _fma_130;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_130)
            : "f"(k_regs[26]), "f"(vn2s0_1_x), "f"(S[58]));
        float _fma_131;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_131)
            : "f"(k_regs[27]), "f"(vn2s0_1_y), "f"(S[59]));
        S[58] = _fma_130;
        S[59] = _fma_131;
        float _fma_132;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_132)
            : "f"(k_regs[28]), "f"(vn2s0_1_x), "f"(S[60]));
        float _fma_133;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_133)
            : "f"(k_regs[29]), "f"(vn2s0_1_y), "f"(S[61]));
        S[60] = _fma_132;
        S[61] = _fma_133;
        float _fma_134;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_134)
            : "f"(k_regs[30]), "f"(vn2s0_1_x), "f"(S[62]));
        float _fma_135;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_135)
            : "f"(k_regs[31]), "f"(vn2s0_1_y), "f"(S[63]));
        S[62] = _fma_134;
        S[63] = _fma_135;
        float accs0_2_0_x = 0.0f;
        float accs0_2_0_y = 0.0f;
        float accs0_2_1_x = 0.0f;
        float accs0_2_1_y = 0.0f;
        float _fma_136;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_136)
            : "f"(S[64]), "f"(k_regs[0]), "f"(accs0_2_0_x));
        float _fma_137;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_137)
            : "f"(S[65]), "f"(k_regs[1]), "f"(accs0_2_0_y));
        accs0_2_0_x = _fma_136;
        accs0_2_0_y = _fma_137;
        float _fma_138;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_138)
            : "f"(S[66]), "f"(k_regs[2]), "f"(accs0_2_1_x));
        float _fma_139;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_139)
            : "f"(S[67]), "f"(k_regs[3]), "f"(accs0_2_1_y));
        accs0_2_1_x = _fma_138;
        accs0_2_1_y = _fma_139;
        float _fma_140;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_140)
            : "f"(S[68]), "f"(k_regs[4]), "f"(accs0_2_0_x));
        float _fma_141;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_141)
            : "f"(S[69]), "f"(k_regs[5]), "f"(accs0_2_0_y));
        accs0_2_0_x = _fma_140;
        accs0_2_0_y = _fma_141;
        float _fma_142;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_142)
            : "f"(S[70]), "f"(k_regs[6]), "f"(accs0_2_1_x));
        float _fma_143;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_143)
            : "f"(S[71]), "f"(k_regs[7]), "f"(accs0_2_1_y));
        accs0_2_1_x = _fma_142;
        accs0_2_1_y = _fma_143;
        float _fma_144;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_144)
            : "f"(S[72]), "f"(k_regs[8]), "f"(accs0_2_0_x));
        float _fma_145;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_145)
            : "f"(S[73]), "f"(k_regs[9]), "f"(accs0_2_0_y));
        accs0_2_0_x = _fma_144;
        accs0_2_0_y = _fma_145;
        float _fma_146;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_146)
            : "f"(S[74]), "f"(k_regs[10]), "f"(accs0_2_1_x));
        float _fma_147;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_147)
            : "f"(S[75]), "f"(k_regs[11]), "f"(accs0_2_1_y));
        accs0_2_1_x = _fma_146;
        accs0_2_1_y = _fma_147;
        float _fma_148;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_148)
            : "f"(S[76]), "f"(k_regs[12]), "f"(accs0_2_0_x));
        float _fma_149;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_149)
            : "f"(S[77]), "f"(k_regs[13]), "f"(accs0_2_0_y));
        accs0_2_0_x = _fma_148;
        accs0_2_0_y = _fma_149;
        float _fma_150;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_150)
            : "f"(S[78]), "f"(k_regs[14]), "f"(accs0_2_1_x));
        float _fma_151;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_151)
            : "f"(S[79]), "f"(k_regs[15]), "f"(accs0_2_1_y));
        accs0_2_1_x = _fma_150;
        accs0_2_1_y = _fma_151;
        float _fma_152;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_152)
            : "f"(S[80]), "f"(k_regs[16]), "f"(accs0_2_0_x));
        float _fma_153;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_153)
            : "f"(S[81]), "f"(k_regs[17]), "f"(accs0_2_0_y));
        accs0_2_0_x = _fma_152;
        accs0_2_0_y = _fma_153;
        float _fma_154;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_154)
            : "f"(S[82]), "f"(k_regs[18]), "f"(accs0_2_1_x));
        float _fma_155;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_155)
            : "f"(S[83]), "f"(k_regs[19]), "f"(accs0_2_1_y));
        accs0_2_1_x = _fma_154;
        accs0_2_1_y = _fma_155;
        float _fma_156;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_156)
            : "f"(S[84]), "f"(k_regs[20]), "f"(accs0_2_0_x));
        float _fma_157;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_157)
            : "f"(S[85]), "f"(k_regs[21]), "f"(accs0_2_0_y));
        accs0_2_0_x = _fma_156;
        accs0_2_0_y = _fma_157;
        float _fma_158;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_158)
            : "f"(S[86]), "f"(k_regs[22]), "f"(accs0_2_1_x));
        float _fma_159;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_159)
            : "f"(S[87]), "f"(k_regs[23]), "f"(accs0_2_1_y));
        accs0_2_1_x = _fma_158;
        accs0_2_1_y = _fma_159;
        float _fma_160;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_160)
            : "f"(S[88]), "f"(k_regs[24]), "f"(accs0_2_0_x));
        float _fma_161;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_161)
            : "f"(S[89]), "f"(k_regs[25]), "f"(accs0_2_0_y));
        accs0_2_0_x = _fma_160;
        accs0_2_0_y = _fma_161;
        float _fma_162;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_162)
            : "f"(S[90]), "f"(k_regs[26]), "f"(accs0_2_1_x));
        float _fma_163;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_163)
            : "f"(S[91]), "f"(k_regs[27]), "f"(accs0_2_1_y));
        accs0_2_1_x = _fma_162;
        accs0_2_1_y = _fma_163;
        float _fma_164;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_164)
            : "f"(S[92]), "f"(k_regs[28]), "f"(accs0_2_0_x));
        float _fma_165;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_165)
            : "f"(S[93]), "f"(k_regs[29]), "f"(accs0_2_0_y));
        accs0_2_0_x = _fma_164;
        accs0_2_0_y = _fma_165;
        float _fma_166;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_166)
            : "f"(S[94]), "f"(k_regs[30]), "f"(accs0_2_1_x));
        float _fma_167;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_167)
            : "f"(S[95]), "f"(k_regs[31]), "f"(accs0_2_1_y));
        accs0_2_1_x = _fma_166;
        accs0_2_1_y = _fma_167;
        float _add_51;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(accs0_2_0_x), "f"(accs0_2_1_x));
        float _add_52;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(accs0_2_0_y), "f"(accs0_2_1_y));
        float dots0_2 = _add_51 + _add_52;
        float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, dots0_2, 1);
        dots0_2 = dots0_2 + _shfl_xor_sync_53;
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, dots0_2, 2);
        dots0_2 = dots0_2 + _shfl_xor_sync_54;
#else
        float2 vn2s0_1 = make_float2(vns0_1, vns0_1);
        S2[16] = fma_f32x2_rn_ftz(k2[0], vn2s0_1, S2[16]);
        S2[17] = fma_f32x2_rn_ftz(k2[1], vn2s0_1, S2[17]);
        S2[18] = fma_f32x2_rn_ftz(k2[2], vn2s0_1, S2[18]);
        S2[19] = fma_f32x2_rn_ftz(k2[3], vn2s0_1, S2[19]);
        S2[20] = fma_f32x2_rn_ftz(k2[4], vn2s0_1, S2[20]);
        S2[21] = fma_f32x2_rn_ftz(k2[5], vn2s0_1, S2[21]);
        S2[22] = fma_f32x2_rn_ftz(k2[6], vn2s0_1, S2[22]);
        S2[23] = fma_f32x2_rn_ftz(k2[7], vn2s0_1, S2[23]);
        S2[24] = fma_f32x2_rn_ftz(k2[8], vn2s0_1, S2[24]);
        S2[25] = fma_f32x2_rn_ftz(k2[9], vn2s0_1, S2[25]);
        S2[26] = fma_f32x2_rn_ftz(k2[10], vn2s0_1, S2[26]);
        S2[27] = fma_f32x2_rn_ftz(k2[11], vn2s0_1, S2[27]);
        S2[28] = fma_f32x2_rn_ftz(k2[12], vn2s0_1, S2[28]);
        S2[29] = fma_f32x2_rn_ftz(k2[13], vn2s0_1, S2[29]);
        S2[30] = fma_f32x2_rn_ftz(k2[14], vn2s0_1, S2[30]);
        S2[31] = fma_f32x2_rn_ftz(k2[15], vn2s0_1, S2[31]);
        float2 accs0_2_0 = make_float2(0.0f, 0.0f);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[32], k2[0], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[33], k2[1], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[34], k2[2], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[35], k2[3], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[36], k2[4], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[37], k2[5], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[38], k2[6], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[39], k2[7], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[40], k2[8], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[41], k2[9], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[42], k2[10], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[43], k2[11], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[44], k2[12], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[45], k2[13], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[46], k2[14], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[47], k2[15], accs0_2_0);
        float dots0_2 = accs0_2_0.x + accs0_2_0.y;
        float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, dots0_2, 1);
        dots0_2 = dots0_2 + _shfl_xor_sync_47;
        float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, dots0_2, 2);
        dots0_2 = dots0_2 + _shfl_xor_sync_48;
#endif
        float vns0_2 = (v_regs[2] - dots0_2) * betas0;
#if __CUDA_ARCH__ == 900
        float vn2s0_2_x = vns0_2;
        float vn2s0_2_y = vns0_2;
        float _fma_168;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_168)
            : "f"(k_regs[0]), "f"(vn2s0_2_x), "f"(S[64]));
        float _fma_169;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_169)
            : "f"(k_regs[1]), "f"(vn2s0_2_y), "f"(S[65]));
        S[64] = _fma_168;
        S[65] = _fma_169;
        float _fma_170;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_170)
            : "f"(k_regs[2]), "f"(vn2s0_2_x), "f"(S[66]));
        float _fma_171;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_171)
            : "f"(k_regs[3]), "f"(vn2s0_2_y), "f"(S[67]));
        S[66] = _fma_170;
        S[67] = _fma_171;
        float _fma_172;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_172)
            : "f"(k_regs[4]), "f"(vn2s0_2_x), "f"(S[68]));
        float _fma_173;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_173)
            : "f"(k_regs[5]), "f"(vn2s0_2_y), "f"(S[69]));
        S[68] = _fma_172;
        S[69] = _fma_173;
        float _fma_174;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_174)
            : "f"(k_regs[6]), "f"(vn2s0_2_x), "f"(S[70]));
        float _fma_175;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_175)
            : "f"(k_regs[7]), "f"(vn2s0_2_y), "f"(S[71]));
        S[70] = _fma_174;
        S[71] = _fma_175;
        float _fma_176;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_176)
            : "f"(k_regs[8]), "f"(vn2s0_2_x), "f"(S[72]));
        float _fma_177;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_177)
            : "f"(k_regs[9]), "f"(vn2s0_2_y), "f"(S[73]));
        S[72] = _fma_176;
        S[73] = _fma_177;
        float _fma_178;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_178)
            : "f"(k_regs[10]), "f"(vn2s0_2_x), "f"(S[74]));
        float _fma_179;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_179)
            : "f"(k_regs[11]), "f"(vn2s0_2_y), "f"(S[75]));
        S[74] = _fma_178;
        S[75] = _fma_179;
        float _fma_180;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_180)
            : "f"(k_regs[12]), "f"(vn2s0_2_x), "f"(S[76]));
        float _fma_181;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_181)
            : "f"(k_regs[13]), "f"(vn2s0_2_y), "f"(S[77]));
        S[76] = _fma_180;
        S[77] = _fma_181;
        float _fma_182;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_182)
            : "f"(k_regs[14]), "f"(vn2s0_2_x), "f"(S[78]));
        float _fma_183;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_183)
            : "f"(k_regs[15]), "f"(vn2s0_2_y), "f"(S[79]));
        S[78] = _fma_182;
        S[79] = _fma_183;
        float _fma_184;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_184)
            : "f"(k_regs[16]), "f"(vn2s0_2_x), "f"(S[80]));
        float _fma_185;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_185)
            : "f"(k_regs[17]), "f"(vn2s0_2_y), "f"(S[81]));
        S[80] = _fma_184;
        S[81] = _fma_185;
        float _fma_186;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_186)
            : "f"(k_regs[18]), "f"(vn2s0_2_x), "f"(S[82]));
        float _fma_187;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_187)
            : "f"(k_regs[19]), "f"(vn2s0_2_y), "f"(S[83]));
        S[82] = _fma_186;
        S[83] = _fma_187;
        float _fma_188;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_188)
            : "f"(k_regs[20]), "f"(vn2s0_2_x), "f"(S[84]));
        float _fma_189;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_189)
            : "f"(k_regs[21]), "f"(vn2s0_2_y), "f"(S[85]));
        S[84] = _fma_188;
        S[85] = _fma_189;
        float _fma_190;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_190)
            : "f"(k_regs[22]), "f"(vn2s0_2_x), "f"(S[86]));
        float _fma_191;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_191)
            : "f"(k_regs[23]), "f"(vn2s0_2_y), "f"(S[87]));
        S[86] = _fma_190;
        S[87] = _fma_191;
        float _fma_192;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_192)
            : "f"(k_regs[24]), "f"(vn2s0_2_x), "f"(S[88]));
        float _fma_193;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_193)
            : "f"(k_regs[25]), "f"(vn2s0_2_y), "f"(S[89]));
        S[88] = _fma_192;
        S[89] = _fma_193;
        float _fma_194;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_194)
            : "f"(k_regs[26]), "f"(vn2s0_2_x), "f"(S[90]));
        float _fma_195;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_195)
            : "f"(k_regs[27]), "f"(vn2s0_2_y), "f"(S[91]));
        S[90] = _fma_194;
        S[91] = _fma_195;
        float _fma_196;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_196)
            : "f"(k_regs[28]), "f"(vn2s0_2_x), "f"(S[92]));
        float _fma_197;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_197)
            : "f"(k_regs[29]), "f"(vn2s0_2_y), "f"(S[93]));
        S[92] = _fma_196;
        S[93] = _fma_197;
        float _fma_198;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_198)
            : "f"(k_regs[30]), "f"(vn2s0_2_x), "f"(S[94]));
        float _fma_199;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_199)
            : "f"(k_regs[31]), "f"(vn2s0_2_y), "f"(S[95]));
        S[94] = _fma_198;
        S[95] = _fma_199;
        float accs0_3_0_x = 0.0f;
        float accs0_3_0_y = 0.0f;
        float accs0_3_1_x = 0.0f;
        float accs0_3_1_y = 0.0f;
        float _fma_200;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_200)
            : "f"(S[96]), "f"(k_regs[0]), "f"(accs0_3_0_x));
        float _fma_201;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_201)
            : "f"(S[97]), "f"(k_regs[1]), "f"(accs0_3_0_y));
        accs0_3_0_x = _fma_200;
        accs0_3_0_y = _fma_201;
        float _fma_202;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_202)
            : "f"(S[98]), "f"(k_regs[2]), "f"(accs0_3_1_x));
        float _fma_203;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_203)
            : "f"(S[99]), "f"(k_regs[3]), "f"(accs0_3_1_y));
        accs0_3_1_x = _fma_202;
        accs0_3_1_y = _fma_203;
        float _fma_204;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_204)
            : "f"(S[100]), "f"(k_regs[4]), "f"(accs0_3_0_x));
        float _fma_205;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_205)
            : "f"(S[101]), "f"(k_regs[5]), "f"(accs0_3_0_y));
        accs0_3_0_x = _fma_204;
        accs0_3_0_y = _fma_205;
        float _fma_206;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_206)
            : "f"(S[102]), "f"(k_regs[6]), "f"(accs0_3_1_x));
        float _fma_207;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_207)
            : "f"(S[103]), "f"(k_regs[7]), "f"(accs0_3_1_y));
        accs0_3_1_x = _fma_206;
        accs0_3_1_y = _fma_207;
        float _fma_208;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_208)
            : "f"(S[104]), "f"(k_regs[8]), "f"(accs0_3_0_x));
        float _fma_209;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_209)
            : "f"(S[105]), "f"(k_regs[9]), "f"(accs0_3_0_y));
        accs0_3_0_x = _fma_208;
        accs0_3_0_y = _fma_209;
        float _fma_210;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_210)
            : "f"(S[106]), "f"(k_regs[10]), "f"(accs0_3_1_x));
        float _fma_211;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_211)
            : "f"(S[107]), "f"(k_regs[11]), "f"(accs0_3_1_y));
        accs0_3_1_x = _fma_210;
        accs0_3_1_y = _fma_211;
        float _fma_212;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_212)
            : "f"(S[108]), "f"(k_regs[12]), "f"(accs0_3_0_x));
        float _fma_213;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_213)
            : "f"(S[109]), "f"(k_regs[13]), "f"(accs0_3_0_y));
        accs0_3_0_x = _fma_212;
        accs0_3_0_y = _fma_213;
        float _fma_214;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_214)
            : "f"(S[110]), "f"(k_regs[14]), "f"(accs0_3_1_x));
        float _fma_215;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_215)
            : "f"(S[111]), "f"(k_regs[15]), "f"(accs0_3_1_y));
        accs0_3_1_x = _fma_214;
        accs0_3_1_y = _fma_215;
        float _fma_216;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_216)
            : "f"(S[112]), "f"(k_regs[16]), "f"(accs0_3_0_x));
        float _fma_217;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_217)
            : "f"(S[113]), "f"(k_regs[17]), "f"(accs0_3_0_y));
        accs0_3_0_x = _fma_216;
        accs0_3_0_y = _fma_217;
        float _fma_218;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_218)
            : "f"(S[114]), "f"(k_regs[18]), "f"(accs0_3_1_x));
        float _fma_219;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_219)
            : "f"(S[115]), "f"(k_regs[19]), "f"(accs0_3_1_y));
        accs0_3_1_x = _fma_218;
        accs0_3_1_y = _fma_219;
        float _fma_220;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_220)
            : "f"(S[116]), "f"(k_regs[20]), "f"(accs0_3_0_x));
        float _fma_221;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_221)
            : "f"(S[117]), "f"(k_regs[21]), "f"(accs0_3_0_y));
        accs0_3_0_x = _fma_220;
        accs0_3_0_y = _fma_221;
        float _fma_222;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_222)
            : "f"(S[118]), "f"(k_regs[22]), "f"(accs0_3_1_x));
        float _fma_223;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_223)
            : "f"(S[119]), "f"(k_regs[23]), "f"(accs0_3_1_y));
        accs0_3_1_x = _fma_222;
        accs0_3_1_y = _fma_223;
        float _fma_224;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_224)
            : "f"(S[120]), "f"(k_regs[24]), "f"(accs0_3_0_x));
        float _fma_225;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_225)
            : "f"(S[121]), "f"(k_regs[25]), "f"(accs0_3_0_y));
        accs0_3_0_x = _fma_224;
        accs0_3_0_y = _fma_225;
        float _fma_226;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_226)
            : "f"(S[122]), "f"(k_regs[26]), "f"(accs0_3_1_x));
        float _fma_227;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_227)
            : "f"(S[123]), "f"(k_regs[27]), "f"(accs0_3_1_y));
        accs0_3_1_x = _fma_226;
        accs0_3_1_y = _fma_227;
        float _fma_228;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_228)
            : "f"(S[124]), "f"(k_regs[28]), "f"(accs0_3_0_x));
        float _fma_229;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_229)
            : "f"(S[125]), "f"(k_regs[29]), "f"(accs0_3_0_y));
        accs0_3_0_x = _fma_228;
        accs0_3_0_y = _fma_229;
        float _fma_230;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_230)
            : "f"(S[126]), "f"(k_regs[30]), "f"(accs0_3_1_x));
        float _fma_231;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_231)
            : "f"(S[127]), "f"(k_regs[31]), "f"(accs0_3_1_y));
        accs0_3_1_x = _fma_230;
        accs0_3_1_y = _fma_231;
        float _add_55;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(accs0_3_0_x), "f"(accs0_3_1_x));
        float _add_56;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(accs0_3_0_y), "f"(accs0_3_1_y));
        float dots0_3 = _add_55 + _add_56;
        float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, dots0_3, 1);
        dots0_3 = dots0_3 + _shfl_xor_sync_57;
        float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, dots0_3, 2);
        dots0_3 = dots0_3 + _shfl_xor_sync_58;
#else
        float2 vn2s0_2 = make_float2(vns0_2, vns0_2);
        S2[32] = fma_f32x2_rn_ftz(k2[0], vn2s0_2, S2[32]);
        S2[33] = fma_f32x2_rn_ftz(k2[1], vn2s0_2, S2[33]);
        S2[34] = fma_f32x2_rn_ftz(k2[2], vn2s0_2, S2[34]);
        S2[35] = fma_f32x2_rn_ftz(k2[3], vn2s0_2, S2[35]);
        S2[36] = fma_f32x2_rn_ftz(k2[4], vn2s0_2, S2[36]);
        S2[37] = fma_f32x2_rn_ftz(k2[5], vn2s0_2, S2[37]);
        S2[38] = fma_f32x2_rn_ftz(k2[6], vn2s0_2, S2[38]);
        S2[39] = fma_f32x2_rn_ftz(k2[7], vn2s0_2, S2[39]);
        S2[40] = fma_f32x2_rn_ftz(k2[8], vn2s0_2, S2[40]);
        S2[41] = fma_f32x2_rn_ftz(k2[9], vn2s0_2, S2[41]);
        S2[42] = fma_f32x2_rn_ftz(k2[10], vn2s0_2, S2[42]);
        S2[43] = fma_f32x2_rn_ftz(k2[11], vn2s0_2, S2[43]);
        S2[44] = fma_f32x2_rn_ftz(k2[12], vn2s0_2, S2[44]);
        S2[45] = fma_f32x2_rn_ftz(k2[13], vn2s0_2, S2[45]);
        S2[46] = fma_f32x2_rn_ftz(k2[14], vn2s0_2, S2[46]);
        S2[47] = fma_f32x2_rn_ftz(k2[15], vn2s0_2, S2[47]);
        float2 accs0_3_0 = make_float2(0.0f, 0.0f);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[48], k2[0], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[49], k2[1], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[50], k2[2], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[51], k2[3], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[52], k2[4], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[53], k2[5], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[54], k2[6], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[55], k2[7], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[56], k2[8], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[57], k2[9], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[58], k2[10], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[59], k2[11], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[60], k2[12], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[61], k2[13], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[62], k2[14], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[63], k2[15], accs0_3_0);
        float dots0_3 = accs0_3_0.x + accs0_3_0.y;
        float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, dots0_3, 1);
        dots0_3 = dots0_3 + _shfl_xor_sync_49;
        float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, dots0_3, 2);
        dots0_3 = dots0_3 + _shfl_xor_sync_50;
#endif
        float vns0_3 = (v_regs[3] - dots0_3) * betas0;
#if __CUDA_ARCH__ == 900
        float vn2s0_3_x = vns0_3;
        float vn2s0_3_y = vns0_3;
        float _fma_232;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_232)
            : "f"(k_regs[0]), "f"(vn2s0_3_x), "f"(S[96]));
        float _fma_233;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_233)
            : "f"(k_regs[1]), "f"(vn2s0_3_y), "f"(S[97]));
        S[96] = _fma_232;
        S[97] = _fma_233;
        float _fma_234;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_234)
            : "f"(k_regs[2]), "f"(vn2s0_3_x), "f"(S[98]));
        float _fma_235;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_235)
            : "f"(k_regs[3]), "f"(vn2s0_3_y), "f"(S[99]));
        S[98] = _fma_234;
        S[99] = _fma_235;
        float _fma_236;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_236)
            : "f"(k_regs[4]), "f"(vn2s0_3_x), "f"(S[100]));
        float _fma_237;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_237)
            : "f"(k_regs[5]), "f"(vn2s0_3_y), "f"(S[101]));
        S[100] = _fma_236;
        S[101] = _fma_237;
        float _fma_238;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_238)
            : "f"(k_regs[6]), "f"(vn2s0_3_x), "f"(S[102]));
        float _fma_239;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_239)
            : "f"(k_regs[7]), "f"(vn2s0_3_y), "f"(S[103]));
        S[102] = _fma_238;
        S[103] = _fma_239;
        float _fma_240;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_240)
            : "f"(k_regs[8]), "f"(vn2s0_3_x), "f"(S[104]));
        float _fma_241;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_241)
            : "f"(k_regs[9]), "f"(vn2s0_3_y), "f"(S[105]));
        S[104] = _fma_240;
        S[105] = _fma_241;
        float _fma_242;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_242)
            : "f"(k_regs[10]), "f"(vn2s0_3_x), "f"(S[106]));
        float _fma_243;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_243)
            : "f"(k_regs[11]), "f"(vn2s0_3_y), "f"(S[107]));
        S[106] = _fma_242;
        S[107] = _fma_243;
        float _fma_244;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_244)
            : "f"(k_regs[12]), "f"(vn2s0_3_x), "f"(S[108]));
        float _fma_245;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_245)
            : "f"(k_regs[13]), "f"(vn2s0_3_y), "f"(S[109]));
        S[108] = _fma_244;
        S[109] = _fma_245;
        float _fma_246;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_246)
            : "f"(k_regs[14]), "f"(vn2s0_3_x), "f"(S[110]));
        float _fma_247;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_247)
            : "f"(k_regs[15]), "f"(vn2s0_3_y), "f"(S[111]));
        S[110] = _fma_246;
        S[111] = _fma_247;
        float _fma_248;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_248)
            : "f"(k_regs[16]), "f"(vn2s0_3_x), "f"(S[112]));
        float _fma_249;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_249)
            : "f"(k_regs[17]), "f"(vn2s0_3_y), "f"(S[113]));
        S[112] = _fma_248;
        S[113] = _fma_249;
        float _fma_250;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_250)
            : "f"(k_regs[18]), "f"(vn2s0_3_x), "f"(S[114]));
        float _fma_251;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_251)
            : "f"(k_regs[19]), "f"(vn2s0_3_y), "f"(S[115]));
        S[114] = _fma_250;
        S[115] = _fma_251;
        float _fma_252;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_252)
            : "f"(k_regs[20]), "f"(vn2s0_3_x), "f"(S[116]));
        float _fma_253;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_253)
            : "f"(k_regs[21]), "f"(vn2s0_3_y), "f"(S[117]));
        S[116] = _fma_252;
        S[117] = _fma_253;
        float _fma_254;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_254)
            : "f"(k_regs[22]), "f"(vn2s0_3_x), "f"(S[118]));
        float _fma_255;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_255)
            : "f"(k_regs[23]), "f"(vn2s0_3_y), "f"(S[119]));
        S[118] = _fma_254;
        S[119] = _fma_255;
        float _fma_256;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_256)
            : "f"(k_regs[24]), "f"(vn2s0_3_x), "f"(S[120]));
        float _fma_257;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_257)
            : "f"(k_regs[25]), "f"(vn2s0_3_y), "f"(S[121]));
        S[120] = _fma_256;
        S[121] = _fma_257;
        float _fma_258;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_258)
            : "f"(k_regs[26]), "f"(vn2s0_3_x), "f"(S[122]));
        float _fma_259;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_259)
            : "f"(k_regs[27]), "f"(vn2s0_3_y), "f"(S[123]));
        S[122] = _fma_258;
        S[123] = _fma_259;
        float _fma_260;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_260)
            : "f"(k_regs[28]), "f"(vn2s0_3_x), "f"(S[124]));
        float _fma_261;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_261)
            : "f"(k_regs[29]), "f"(vn2s0_3_y), "f"(S[125]));
        S[124] = _fma_260;
        S[125] = _fma_261;
        float _fma_262;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_262)
            : "f"(k_regs[30]), "f"(vn2s0_3_x), "f"(S[126]));
        float _fma_263;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_263)
            : "f"(k_regs[31]), "f"(vn2s0_3_y), "f"(S[127]));
        S[126] = _fma_262;
        S[127] = _fma_263;
#else
        float2 vn2s0_3 = make_float2(vns0_3, vns0_3);
        S2[48] = fma_f32x2_rn_ftz(k2[0], vn2s0_3, S2[48]);
        S2[49] = fma_f32x2_rn_ftz(k2[1], vn2s0_3, S2[49]);
        S2[50] = fma_f32x2_rn_ftz(k2[2], vn2s0_3, S2[50]);
        S2[51] = fma_f32x2_rn_ftz(k2[3], vn2s0_3, S2[51]);
        S2[52] = fma_f32x2_rn_ftz(k2[4], vn2s0_3, S2[52]);
        S2[53] = fma_f32x2_rn_ftz(k2[5], vn2s0_3, S2[53]);
        S2[54] = fma_f32x2_rn_ftz(k2[6], vn2s0_3, S2[54]);
        S2[55] = fma_f32x2_rn_ftz(k2[7], vn2s0_3, S2[55]);
        S2[56] = fma_f32x2_rn_ftz(k2[8], vn2s0_3, S2[56]);
        S2[57] = fma_f32x2_rn_ftz(k2[9], vn2s0_3, S2[57]);
        S2[58] = fma_f32x2_rn_ftz(k2[10], vn2s0_3, S2[58]);
        S2[59] = fma_f32x2_rn_ftz(k2[11], vn2s0_3, S2[59]);
        S2[60] = fma_f32x2_rn_ftz(k2[12], vn2s0_3, S2[60]);
        S2[61] = fma_f32x2_rn_ftz(k2[13], vn2s0_3, S2[61]);
        S2[62] = fma_f32x2_rn_ftz(k2[14], vn2s0_3, S2[62]);
        S2[63] = fma_f32x2_rn_ftz(k2[15], vn2s0_3, S2[63]);
#endif
        int ms1 = t_1 * 3 + 1;
        unsigned int kbases1 = sK_addr + (unsigned int)(ms1 * 512) + lane_byte;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                     : "r"(kbases1));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                     : "r"(kbases1 + 64));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                     : "r"(kbases1 + 128));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                     : "r"(kbases1 + 192));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[16])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 3]))
                     : "r"(kbases1 + 256));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[20])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 3]))
                     : "r"(kbases1 + 320));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[24])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 3]))
                     : "r"(kbases1 + 384));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[28])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 3]))
                     : "r"(kbases1 + 448));
#if __CUDA_ARCH__ == 900
#else
        k2[0] = make_float2(k_regs[0], k_regs[1]);
        k2[1] = make_float2(k_regs[2], k_regs[3]);
        k2[2] = make_float2(k_regs[4], k_regs[5]);
        k2[3] = make_float2(k_regs[6], k_regs[7]);
        k2[4] = make_float2(k_regs[8], k_regs[9]);
        k2[5] = make_float2(k_regs[10], k_regs[11]);
        k2[6] = make_float2(k_regs[12], k_regs[13]);
        k2[7] = make_float2(k_regs[14], k_regs[15]);
        k2[8] = make_float2(k_regs[16], k_regs[17]);
        k2[9] = make_float2(k_regs[18], k_regs[19]);
        k2[10] = make_float2(k_regs[20], k_regs[21]);
        k2[11] = make_float2(k_regs[22], k_regs[23]);
        k2[12] = make_float2(k_regs[24], k_regs[25]);
        k2[13] = make_float2(k_regs[26], k_regs[27]);
        k2[14] = make_float2(k_regs[28], k_regs[29]);
        k2[15] = make_float2(k_regs[30], k_regs[31]);
#endif
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                     : "r"(sV_addr + (unsigned int)(ms1 * 256) + group_byte));
        float betas1 = sBeta[ms1];
#if __CUDA_ARCH__ == 900
        float accs1_0_0_x = 0.0f;
        float accs1_0_0_y = 0.0f;
        float accs1_0_1_x = 0.0f;
        float accs1_0_1_y = 0.0f;
        float _fma_264;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_264)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accs1_0_0_x));
        float _fma_265;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_265)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accs1_0_0_y));
        accs1_0_0_x = _fma_264;
        accs1_0_0_y = _fma_265;
        float _fma_266;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_266)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accs1_0_1_x));
        float _fma_267;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_267)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accs1_0_1_y));
        accs1_0_1_x = _fma_266;
        accs1_0_1_y = _fma_267;
        float _fma_268;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_268)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accs1_0_0_x));
        float _fma_269;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_269)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accs1_0_0_y));
        accs1_0_0_x = _fma_268;
        accs1_0_0_y = _fma_269;
        float _fma_270;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_270)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accs1_0_1_x));
        float _fma_271;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_271)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accs1_0_1_y));
        accs1_0_1_x = _fma_270;
        accs1_0_1_y = _fma_271;
        float _fma_272;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_272)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accs1_0_0_x));
        float _fma_273;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_273)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accs1_0_0_y));
        accs1_0_0_x = _fma_272;
        accs1_0_0_y = _fma_273;
        float _fma_274;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_274)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accs1_0_1_x));
        float _fma_275;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_275)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accs1_0_1_y));
        accs1_0_1_x = _fma_274;
        accs1_0_1_y = _fma_275;
        float _fma_276;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_276)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accs1_0_0_x));
        float _fma_277;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_277)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accs1_0_0_y));
        accs1_0_0_x = _fma_276;
        accs1_0_0_y = _fma_277;
        float _fma_278;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_278)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accs1_0_1_x));
        float _fma_279;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_279)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accs1_0_1_y));
        accs1_0_1_x = _fma_278;
        accs1_0_1_y = _fma_279;
        float _fma_280;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_280)
            : "f"(S[16]), "f"(k_regs[16]), "f"(accs1_0_0_x));
        float _fma_281;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_281)
            : "f"(S[17]), "f"(k_regs[17]), "f"(accs1_0_0_y));
        accs1_0_0_x = _fma_280;
        accs1_0_0_y = _fma_281;
        float _fma_282;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_282)
            : "f"(S[18]), "f"(k_regs[18]), "f"(accs1_0_1_x));
        float _fma_283;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_283)
            : "f"(S[19]), "f"(k_regs[19]), "f"(accs1_0_1_y));
        accs1_0_1_x = _fma_282;
        accs1_0_1_y = _fma_283;
        float _fma_284;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_284)
            : "f"(S[20]), "f"(k_regs[20]), "f"(accs1_0_0_x));
        float _fma_285;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_285)
            : "f"(S[21]), "f"(k_regs[21]), "f"(accs1_0_0_y));
        accs1_0_0_x = _fma_284;
        accs1_0_0_y = _fma_285;
        float _fma_286;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_286)
            : "f"(S[22]), "f"(k_regs[22]), "f"(accs1_0_1_x));
        float _fma_287;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_287)
            : "f"(S[23]), "f"(k_regs[23]), "f"(accs1_0_1_y));
        accs1_0_1_x = _fma_286;
        accs1_0_1_y = _fma_287;
        float _fma_288;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_288)
            : "f"(S[24]), "f"(k_regs[24]), "f"(accs1_0_0_x));
        float _fma_289;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_289)
            : "f"(S[25]), "f"(k_regs[25]), "f"(accs1_0_0_y));
        accs1_0_0_x = _fma_288;
        accs1_0_0_y = _fma_289;
        float _fma_290;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_290)
            : "f"(S[26]), "f"(k_regs[26]), "f"(accs1_0_1_x));
        float _fma_291;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_291)
            : "f"(S[27]), "f"(k_regs[27]), "f"(accs1_0_1_y));
        accs1_0_1_x = _fma_290;
        accs1_0_1_y = _fma_291;
        float _fma_292;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_292)
            : "f"(S[28]), "f"(k_regs[28]), "f"(accs1_0_0_x));
        float _fma_293;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_293)
            : "f"(S[29]), "f"(k_regs[29]), "f"(accs1_0_0_y));
        accs1_0_0_x = _fma_292;
        accs1_0_0_y = _fma_293;
        float _fma_294;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_294)
            : "f"(S[30]), "f"(k_regs[30]), "f"(accs1_0_1_x));
        float _fma_295;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_295)
            : "f"(S[31]), "f"(k_regs[31]), "f"(accs1_0_1_y));
        accs1_0_1_x = _fma_294;
        accs1_0_1_y = _fma_295;
        float _add_59;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(accs1_0_0_x), "f"(accs1_0_1_x));
        float _add_60;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(accs1_0_0_y), "f"(accs1_0_1_y));
        float dots1_0 = _add_59 + _add_60;
        float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 1);
        dots1_0 = dots1_0 + _shfl_xor_sync_61;
        float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 2);
        dots1_0 = dots1_0 + _shfl_xor_sync_62;
#else
        float2 accs1_0_0 = make_float2(0.0f, 0.0f);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accs1_0_0);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[1], k2[1], accs1_0_0);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accs1_0_0);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[3], k2[3], accs1_0_0);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accs1_0_0);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[5], k2[5], accs1_0_0);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accs1_0_0);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[7], k2[7], accs1_0_0);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[8], k2[8], accs1_0_0);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[9], k2[9], accs1_0_0);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[10], k2[10], accs1_0_0);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[11], k2[11], accs1_0_0);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[12], k2[12], accs1_0_0);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[13], k2[13], accs1_0_0);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[14], k2[14], accs1_0_0);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[15], k2[15], accs1_0_0);
        float dots1_0 = accs1_0_0.x + accs1_0_0.y;
        float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 1);
        dots1_0 = dots1_0 + _shfl_xor_sync_51;
        float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 2);
        dots1_0 = dots1_0 + _shfl_xor_sync_52;
#endif
        float vns1_0 = (v_regs[0] - dots1_0) * betas1;
#if __CUDA_ARCH__ == 900
        float vn2s1_0_x = vns1_0;
        float vn2s1_0_y = vns1_0;
        float _fma_296;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_296)
            : "f"(k_regs[0]), "f"(vn2s1_0_x), "f"(S[0]));
        float _fma_297;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_297)
            : "f"(k_regs[1]), "f"(vn2s1_0_y), "f"(S[1]));
        S[0] = _fma_296;
        S[1] = _fma_297;
        float _fma_298;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_298)
            : "f"(k_regs[2]), "f"(vn2s1_0_x), "f"(S[2]));
        float _fma_299;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_299)
            : "f"(k_regs[3]), "f"(vn2s1_0_y), "f"(S[3]));
        S[2] = _fma_298;
        S[3] = _fma_299;
        float _fma_300;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_300)
            : "f"(k_regs[4]), "f"(vn2s1_0_x), "f"(S[4]));
        float _fma_301;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_301)
            : "f"(k_regs[5]), "f"(vn2s1_0_y), "f"(S[5]));
        S[4] = _fma_300;
        S[5] = _fma_301;
        float _fma_302;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_302)
            : "f"(k_regs[6]), "f"(vn2s1_0_x), "f"(S[6]));
        float _fma_303;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_303)
            : "f"(k_regs[7]), "f"(vn2s1_0_y), "f"(S[7]));
        S[6] = _fma_302;
        S[7] = _fma_303;
        float _fma_304;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_304)
            : "f"(k_regs[8]), "f"(vn2s1_0_x), "f"(S[8]));
        float _fma_305;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_305)
            : "f"(k_regs[9]), "f"(vn2s1_0_y), "f"(S[9]));
        S[8] = _fma_304;
        S[9] = _fma_305;
        float _fma_306;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_306)
            : "f"(k_regs[10]), "f"(vn2s1_0_x), "f"(S[10]));
        float _fma_307;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_307)
            : "f"(k_regs[11]), "f"(vn2s1_0_y), "f"(S[11]));
        S[10] = _fma_306;
        S[11] = _fma_307;
        float _fma_308;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_308)
            : "f"(k_regs[12]), "f"(vn2s1_0_x), "f"(S[12]));
        float _fma_309;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_309)
            : "f"(k_regs[13]), "f"(vn2s1_0_y), "f"(S[13]));
        S[12] = _fma_308;
        S[13] = _fma_309;
        float _fma_310;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_310)
            : "f"(k_regs[14]), "f"(vn2s1_0_x), "f"(S[14]));
        float _fma_311;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_311)
            : "f"(k_regs[15]), "f"(vn2s1_0_y), "f"(S[15]));
        S[14] = _fma_310;
        S[15] = _fma_311;
        float _fma_312;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_312)
            : "f"(k_regs[16]), "f"(vn2s1_0_x), "f"(S[16]));
        float _fma_313;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_313)
            : "f"(k_regs[17]), "f"(vn2s1_0_y), "f"(S[17]));
        S[16] = _fma_312;
        S[17] = _fma_313;
        float _fma_314;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_314)
            : "f"(k_regs[18]), "f"(vn2s1_0_x), "f"(S[18]));
        float _fma_315;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_315)
            : "f"(k_regs[19]), "f"(vn2s1_0_y), "f"(S[19]));
        S[18] = _fma_314;
        S[19] = _fma_315;
        float _fma_316;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_316)
            : "f"(k_regs[20]), "f"(vn2s1_0_x), "f"(S[20]));
        float _fma_317;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_317)
            : "f"(k_regs[21]), "f"(vn2s1_0_y), "f"(S[21]));
        S[20] = _fma_316;
        S[21] = _fma_317;
        float _fma_318;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_318)
            : "f"(k_regs[22]), "f"(vn2s1_0_x), "f"(S[22]));
        float _fma_319;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_319)
            : "f"(k_regs[23]), "f"(vn2s1_0_y), "f"(S[23]));
        S[22] = _fma_318;
        S[23] = _fma_319;
        float _fma_320;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_320)
            : "f"(k_regs[24]), "f"(vn2s1_0_x), "f"(S[24]));
        float _fma_321;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_321)
            : "f"(k_regs[25]), "f"(vn2s1_0_y), "f"(S[25]));
        S[24] = _fma_320;
        S[25] = _fma_321;
        float _fma_322;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_322)
            : "f"(k_regs[26]), "f"(vn2s1_0_x), "f"(S[26]));
        float _fma_323;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_323)
            : "f"(k_regs[27]), "f"(vn2s1_0_y), "f"(S[27]));
        S[26] = _fma_322;
        S[27] = _fma_323;
        float _fma_324;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_324)
            : "f"(k_regs[28]), "f"(vn2s1_0_x), "f"(S[28]));
        float _fma_325;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_325)
            : "f"(k_regs[29]), "f"(vn2s1_0_y), "f"(S[29]));
        S[28] = _fma_324;
        S[29] = _fma_325;
        float _fma_326;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_326)
            : "f"(k_regs[30]), "f"(vn2s1_0_x), "f"(S[30]));
        float _fma_327;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_327)
            : "f"(k_regs[31]), "f"(vn2s1_0_y), "f"(S[31]));
        S[30] = _fma_326;
        S[31] = _fma_327;
        float accs1_1_0_x = 0.0f;
        float accs1_1_0_y = 0.0f;
        float accs1_1_1_x = 0.0f;
        float accs1_1_1_y = 0.0f;
        float _fma_328;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_328)
            : "f"(S[32]), "f"(k_regs[0]), "f"(accs1_1_0_x));
        float _fma_329;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_329)
            : "f"(S[33]), "f"(k_regs[1]), "f"(accs1_1_0_y));
        accs1_1_0_x = _fma_328;
        accs1_1_0_y = _fma_329;
        float _fma_330;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_330)
            : "f"(S[34]), "f"(k_regs[2]), "f"(accs1_1_1_x));
        float _fma_331;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_331)
            : "f"(S[35]), "f"(k_regs[3]), "f"(accs1_1_1_y));
        accs1_1_1_x = _fma_330;
        accs1_1_1_y = _fma_331;
        float _fma_332;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_332)
            : "f"(S[36]), "f"(k_regs[4]), "f"(accs1_1_0_x));
        float _fma_333;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_333)
            : "f"(S[37]), "f"(k_regs[5]), "f"(accs1_1_0_y));
        accs1_1_0_x = _fma_332;
        accs1_1_0_y = _fma_333;
        float _fma_334;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_334)
            : "f"(S[38]), "f"(k_regs[6]), "f"(accs1_1_1_x));
        float _fma_335;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_335)
            : "f"(S[39]), "f"(k_regs[7]), "f"(accs1_1_1_y));
        accs1_1_1_x = _fma_334;
        accs1_1_1_y = _fma_335;
        float _fma_336;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_336)
            : "f"(S[40]), "f"(k_regs[8]), "f"(accs1_1_0_x));
        float _fma_337;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_337)
            : "f"(S[41]), "f"(k_regs[9]), "f"(accs1_1_0_y));
        accs1_1_0_x = _fma_336;
        accs1_1_0_y = _fma_337;
        float _fma_338;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_338)
            : "f"(S[42]), "f"(k_regs[10]), "f"(accs1_1_1_x));
        float _fma_339;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_339)
            : "f"(S[43]), "f"(k_regs[11]), "f"(accs1_1_1_y));
        accs1_1_1_x = _fma_338;
        accs1_1_1_y = _fma_339;
        float _fma_340;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_340)
            : "f"(S[44]), "f"(k_regs[12]), "f"(accs1_1_0_x));
        float _fma_341;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_341)
            : "f"(S[45]), "f"(k_regs[13]), "f"(accs1_1_0_y));
        accs1_1_0_x = _fma_340;
        accs1_1_0_y = _fma_341;
        float _fma_342;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_342)
            : "f"(S[46]), "f"(k_regs[14]), "f"(accs1_1_1_x));
        float _fma_343;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_343)
            : "f"(S[47]), "f"(k_regs[15]), "f"(accs1_1_1_y));
        accs1_1_1_x = _fma_342;
        accs1_1_1_y = _fma_343;
        float _fma_344;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_344)
            : "f"(S[48]), "f"(k_regs[16]), "f"(accs1_1_0_x));
        float _fma_345;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_345)
            : "f"(S[49]), "f"(k_regs[17]), "f"(accs1_1_0_y));
        accs1_1_0_x = _fma_344;
        accs1_1_0_y = _fma_345;
        float _fma_346;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_346)
            : "f"(S[50]), "f"(k_regs[18]), "f"(accs1_1_1_x));
        float _fma_347;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_347)
            : "f"(S[51]), "f"(k_regs[19]), "f"(accs1_1_1_y));
        accs1_1_1_x = _fma_346;
        accs1_1_1_y = _fma_347;
        float _fma_348;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_348)
            : "f"(S[52]), "f"(k_regs[20]), "f"(accs1_1_0_x));
        float _fma_349;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_349)
            : "f"(S[53]), "f"(k_regs[21]), "f"(accs1_1_0_y));
        accs1_1_0_x = _fma_348;
        accs1_1_0_y = _fma_349;
        float _fma_350;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_350)
            : "f"(S[54]), "f"(k_regs[22]), "f"(accs1_1_1_x));
        float _fma_351;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_351)
            : "f"(S[55]), "f"(k_regs[23]), "f"(accs1_1_1_y));
        accs1_1_1_x = _fma_350;
        accs1_1_1_y = _fma_351;
        float _fma_352;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_352)
            : "f"(S[56]), "f"(k_regs[24]), "f"(accs1_1_0_x));
        float _fma_353;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_353)
            : "f"(S[57]), "f"(k_regs[25]), "f"(accs1_1_0_y));
        accs1_1_0_x = _fma_352;
        accs1_1_0_y = _fma_353;
        float _fma_354;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_354)
            : "f"(S[58]), "f"(k_regs[26]), "f"(accs1_1_1_x));
        float _fma_355;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_355)
            : "f"(S[59]), "f"(k_regs[27]), "f"(accs1_1_1_y));
        accs1_1_1_x = _fma_354;
        accs1_1_1_y = _fma_355;
        float _fma_356;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_356)
            : "f"(S[60]), "f"(k_regs[28]), "f"(accs1_1_0_x));
        float _fma_357;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_357)
            : "f"(S[61]), "f"(k_regs[29]), "f"(accs1_1_0_y));
        accs1_1_0_x = _fma_356;
        accs1_1_0_y = _fma_357;
        float _fma_358;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_358)
            : "f"(S[62]), "f"(k_regs[30]), "f"(accs1_1_1_x));
        float _fma_359;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_359)
            : "f"(S[63]), "f"(k_regs[31]), "f"(accs1_1_1_y));
        accs1_1_1_x = _fma_358;
        accs1_1_1_y = _fma_359;
        float _add_63;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_63) : "f"(accs1_1_0_x), "f"(accs1_1_1_x));
        float _add_64;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(accs1_1_0_y), "f"(accs1_1_1_y));
        float dots1_1 = _add_63 + _add_64;
        float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 1);
        dots1_1 = dots1_1 + _shfl_xor_sync_65;
        float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 2);
        dots1_1 = dots1_1 + _shfl_xor_sync_66;
#else
        float2 vn2s1_0 = make_float2(vns1_0, vns1_0);
        S2[0] = fma_f32x2_rn_ftz(k2[0], vn2s1_0, S2[0]);
        S2[1] = fma_f32x2_rn_ftz(k2[1], vn2s1_0, S2[1]);
        S2[2] = fma_f32x2_rn_ftz(k2[2], vn2s1_0, S2[2]);
        S2[3] = fma_f32x2_rn_ftz(k2[3], vn2s1_0, S2[3]);
        S2[4] = fma_f32x2_rn_ftz(k2[4], vn2s1_0, S2[4]);
        S2[5] = fma_f32x2_rn_ftz(k2[5], vn2s1_0, S2[5]);
        S2[6] = fma_f32x2_rn_ftz(k2[6], vn2s1_0, S2[6]);
        S2[7] = fma_f32x2_rn_ftz(k2[7], vn2s1_0, S2[7]);
        S2[8] = fma_f32x2_rn_ftz(k2[8], vn2s1_0, S2[8]);
        S2[9] = fma_f32x2_rn_ftz(k2[9], vn2s1_0, S2[9]);
        S2[10] = fma_f32x2_rn_ftz(k2[10], vn2s1_0, S2[10]);
        S2[11] = fma_f32x2_rn_ftz(k2[11], vn2s1_0, S2[11]);
        S2[12] = fma_f32x2_rn_ftz(k2[12], vn2s1_0, S2[12]);
        S2[13] = fma_f32x2_rn_ftz(k2[13], vn2s1_0, S2[13]);
        S2[14] = fma_f32x2_rn_ftz(k2[14], vn2s1_0, S2[14]);
        S2[15] = fma_f32x2_rn_ftz(k2[15], vn2s1_0, S2[15]);
        float2 accs1_1_0 = make_float2(0.0f, 0.0f);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[24], k2[8], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[25], k2[9], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[26], k2[10], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[27], k2[11], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[28], k2[12], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[29], k2[13], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[30], k2[14], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[31], k2[15], accs1_1_0);
        float dots1_1 = accs1_1_0.x + accs1_1_0.y;
        float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 1);
        dots1_1 = dots1_1 + _shfl_xor_sync_53;
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 2);
        dots1_1 = dots1_1 + _shfl_xor_sync_54;
#endif
        float vns1_1 = (v_regs[1] - dots1_1) * betas1;
#if __CUDA_ARCH__ == 900
        float vn2s1_1_x = vns1_1;
        float vn2s1_1_y = vns1_1;
        float _fma_360;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_360)
            : "f"(k_regs[0]), "f"(vn2s1_1_x), "f"(S[32]));
        float _fma_361;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_361)
            : "f"(k_regs[1]), "f"(vn2s1_1_y), "f"(S[33]));
        S[32] = _fma_360;
        S[33] = _fma_361;
        float _fma_362;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_362)
            : "f"(k_regs[2]), "f"(vn2s1_1_x), "f"(S[34]));
        float _fma_363;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_363)
            : "f"(k_regs[3]), "f"(vn2s1_1_y), "f"(S[35]));
        S[34] = _fma_362;
        S[35] = _fma_363;
        float _fma_364;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_364)
            : "f"(k_regs[4]), "f"(vn2s1_1_x), "f"(S[36]));
        float _fma_365;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_365)
            : "f"(k_regs[5]), "f"(vn2s1_1_y), "f"(S[37]));
        S[36] = _fma_364;
        S[37] = _fma_365;
        float _fma_366;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_366)
            : "f"(k_regs[6]), "f"(vn2s1_1_x), "f"(S[38]));
        float _fma_367;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_367)
            : "f"(k_regs[7]), "f"(vn2s1_1_y), "f"(S[39]));
        S[38] = _fma_366;
        S[39] = _fma_367;
        float _fma_368;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_368)
            : "f"(k_regs[8]), "f"(vn2s1_1_x), "f"(S[40]));
        float _fma_369;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_369)
            : "f"(k_regs[9]), "f"(vn2s1_1_y), "f"(S[41]));
        S[40] = _fma_368;
        S[41] = _fma_369;
        float _fma_370;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_370)
            : "f"(k_regs[10]), "f"(vn2s1_1_x), "f"(S[42]));
        float _fma_371;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_371)
            : "f"(k_regs[11]), "f"(vn2s1_1_y), "f"(S[43]));
        S[42] = _fma_370;
        S[43] = _fma_371;
        float _fma_372;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_372)
            : "f"(k_regs[12]), "f"(vn2s1_1_x), "f"(S[44]));
        float _fma_373;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_373)
            : "f"(k_regs[13]), "f"(vn2s1_1_y), "f"(S[45]));
        S[44] = _fma_372;
        S[45] = _fma_373;
        float _fma_374;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_374)
            : "f"(k_regs[14]), "f"(vn2s1_1_x), "f"(S[46]));
        float _fma_375;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_375)
            : "f"(k_regs[15]), "f"(vn2s1_1_y), "f"(S[47]));
        S[46] = _fma_374;
        S[47] = _fma_375;
        float _fma_376;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_376)
            : "f"(k_regs[16]), "f"(vn2s1_1_x), "f"(S[48]));
        float _fma_377;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_377)
            : "f"(k_regs[17]), "f"(vn2s1_1_y), "f"(S[49]));
        S[48] = _fma_376;
        S[49] = _fma_377;
        float _fma_378;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_378)
            : "f"(k_regs[18]), "f"(vn2s1_1_x), "f"(S[50]));
        float _fma_379;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_379)
            : "f"(k_regs[19]), "f"(vn2s1_1_y), "f"(S[51]));
        S[50] = _fma_378;
        S[51] = _fma_379;
        float _fma_380;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_380)
            : "f"(k_regs[20]), "f"(vn2s1_1_x), "f"(S[52]));
        float _fma_381;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_381)
            : "f"(k_regs[21]), "f"(vn2s1_1_y), "f"(S[53]));
        S[52] = _fma_380;
        S[53] = _fma_381;
        float _fma_382;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_382)
            : "f"(k_regs[22]), "f"(vn2s1_1_x), "f"(S[54]));
        float _fma_383;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_383)
            : "f"(k_regs[23]), "f"(vn2s1_1_y), "f"(S[55]));
        S[54] = _fma_382;
        S[55] = _fma_383;
        float _fma_384;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_384)
            : "f"(k_regs[24]), "f"(vn2s1_1_x), "f"(S[56]));
        float _fma_385;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_385)
            : "f"(k_regs[25]), "f"(vn2s1_1_y), "f"(S[57]));
        S[56] = _fma_384;
        S[57] = _fma_385;
        float _fma_386;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_386)
            : "f"(k_regs[26]), "f"(vn2s1_1_x), "f"(S[58]));
        float _fma_387;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_387)
            : "f"(k_regs[27]), "f"(vn2s1_1_y), "f"(S[59]));
        S[58] = _fma_386;
        S[59] = _fma_387;
        float _fma_388;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_388)
            : "f"(k_regs[28]), "f"(vn2s1_1_x), "f"(S[60]));
        float _fma_389;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_389)
            : "f"(k_regs[29]), "f"(vn2s1_1_y), "f"(S[61]));
        S[60] = _fma_388;
        S[61] = _fma_389;
        float _fma_390;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_390)
            : "f"(k_regs[30]), "f"(vn2s1_1_x), "f"(S[62]));
        float _fma_391;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_391)
            : "f"(k_regs[31]), "f"(vn2s1_1_y), "f"(S[63]));
        S[62] = _fma_390;
        S[63] = _fma_391;
        float accs1_2_0_x = 0.0f;
        float accs1_2_0_y = 0.0f;
        float accs1_2_1_x = 0.0f;
        float accs1_2_1_y = 0.0f;
        float _fma_392;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_392)
            : "f"(S[64]), "f"(k_regs[0]), "f"(accs1_2_0_x));
        float _fma_393;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_393)
            : "f"(S[65]), "f"(k_regs[1]), "f"(accs1_2_0_y));
        accs1_2_0_x = _fma_392;
        accs1_2_0_y = _fma_393;
        float _fma_394;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_394)
            : "f"(S[66]), "f"(k_regs[2]), "f"(accs1_2_1_x));
        float _fma_395;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_395)
            : "f"(S[67]), "f"(k_regs[3]), "f"(accs1_2_1_y));
        accs1_2_1_x = _fma_394;
        accs1_2_1_y = _fma_395;
        float _fma_396;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_396)
            : "f"(S[68]), "f"(k_regs[4]), "f"(accs1_2_0_x));
        float _fma_397;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_397)
            : "f"(S[69]), "f"(k_regs[5]), "f"(accs1_2_0_y));
        accs1_2_0_x = _fma_396;
        accs1_2_0_y = _fma_397;
        float _fma_398;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_398)
            : "f"(S[70]), "f"(k_regs[6]), "f"(accs1_2_1_x));
        float _fma_399;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_399)
            : "f"(S[71]), "f"(k_regs[7]), "f"(accs1_2_1_y));
        accs1_2_1_x = _fma_398;
        accs1_2_1_y = _fma_399;
        float _fma_400;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_400)
            : "f"(S[72]), "f"(k_regs[8]), "f"(accs1_2_0_x));
        float _fma_401;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_401)
            : "f"(S[73]), "f"(k_regs[9]), "f"(accs1_2_0_y));
        accs1_2_0_x = _fma_400;
        accs1_2_0_y = _fma_401;
        float _fma_402;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_402)
            : "f"(S[74]), "f"(k_regs[10]), "f"(accs1_2_1_x));
        float _fma_403;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_403)
            : "f"(S[75]), "f"(k_regs[11]), "f"(accs1_2_1_y));
        accs1_2_1_x = _fma_402;
        accs1_2_1_y = _fma_403;
        float _fma_404;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_404)
            : "f"(S[76]), "f"(k_regs[12]), "f"(accs1_2_0_x));
        float _fma_405;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_405)
            : "f"(S[77]), "f"(k_regs[13]), "f"(accs1_2_0_y));
        accs1_2_0_x = _fma_404;
        accs1_2_0_y = _fma_405;
        float _fma_406;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_406)
            : "f"(S[78]), "f"(k_regs[14]), "f"(accs1_2_1_x));
        float _fma_407;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_407)
            : "f"(S[79]), "f"(k_regs[15]), "f"(accs1_2_1_y));
        accs1_2_1_x = _fma_406;
        accs1_2_1_y = _fma_407;
        float _fma_408;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_408)
            : "f"(S[80]), "f"(k_regs[16]), "f"(accs1_2_0_x));
        float _fma_409;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_409)
            : "f"(S[81]), "f"(k_regs[17]), "f"(accs1_2_0_y));
        accs1_2_0_x = _fma_408;
        accs1_2_0_y = _fma_409;
        float _fma_410;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_410)
            : "f"(S[82]), "f"(k_regs[18]), "f"(accs1_2_1_x));
        float _fma_411;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_411)
            : "f"(S[83]), "f"(k_regs[19]), "f"(accs1_2_1_y));
        accs1_2_1_x = _fma_410;
        accs1_2_1_y = _fma_411;
        float _fma_412;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_412)
            : "f"(S[84]), "f"(k_regs[20]), "f"(accs1_2_0_x));
        float _fma_413;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_413)
            : "f"(S[85]), "f"(k_regs[21]), "f"(accs1_2_0_y));
        accs1_2_0_x = _fma_412;
        accs1_2_0_y = _fma_413;
        float _fma_414;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_414)
            : "f"(S[86]), "f"(k_regs[22]), "f"(accs1_2_1_x));
        float _fma_415;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_415)
            : "f"(S[87]), "f"(k_regs[23]), "f"(accs1_2_1_y));
        accs1_2_1_x = _fma_414;
        accs1_2_1_y = _fma_415;
        float _fma_416;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_416)
            : "f"(S[88]), "f"(k_regs[24]), "f"(accs1_2_0_x));
        float _fma_417;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_417)
            : "f"(S[89]), "f"(k_regs[25]), "f"(accs1_2_0_y));
        accs1_2_0_x = _fma_416;
        accs1_2_0_y = _fma_417;
        float _fma_418;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_418)
            : "f"(S[90]), "f"(k_regs[26]), "f"(accs1_2_1_x));
        float _fma_419;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_419)
            : "f"(S[91]), "f"(k_regs[27]), "f"(accs1_2_1_y));
        accs1_2_1_x = _fma_418;
        accs1_2_1_y = _fma_419;
        float _fma_420;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_420)
            : "f"(S[92]), "f"(k_regs[28]), "f"(accs1_2_0_x));
        float _fma_421;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_421)
            : "f"(S[93]), "f"(k_regs[29]), "f"(accs1_2_0_y));
        accs1_2_0_x = _fma_420;
        accs1_2_0_y = _fma_421;
        float _fma_422;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_422)
            : "f"(S[94]), "f"(k_regs[30]), "f"(accs1_2_1_x));
        float _fma_423;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_423)
            : "f"(S[95]), "f"(k_regs[31]), "f"(accs1_2_1_y));
        accs1_2_1_x = _fma_422;
        accs1_2_1_y = _fma_423;
        float _add_67;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(accs1_2_0_x), "f"(accs1_2_1_x));
        float _add_68;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(accs1_2_0_y), "f"(accs1_2_1_y));
        float dots1_2 = _add_67 + _add_68;
        float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, dots1_2, 1);
        dots1_2 = dots1_2 + _shfl_xor_sync_69;
        float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, dots1_2, 2);
        dots1_2 = dots1_2 + _shfl_xor_sync_70;
#else
        float2 vn2s1_1 = make_float2(vns1_1, vns1_1);
        S2[16] = fma_f32x2_rn_ftz(k2[0], vn2s1_1, S2[16]);
        S2[17] = fma_f32x2_rn_ftz(k2[1], vn2s1_1, S2[17]);
        S2[18] = fma_f32x2_rn_ftz(k2[2], vn2s1_1, S2[18]);
        S2[19] = fma_f32x2_rn_ftz(k2[3], vn2s1_1, S2[19]);
        S2[20] = fma_f32x2_rn_ftz(k2[4], vn2s1_1, S2[20]);
        S2[21] = fma_f32x2_rn_ftz(k2[5], vn2s1_1, S2[21]);
        S2[22] = fma_f32x2_rn_ftz(k2[6], vn2s1_1, S2[22]);
        S2[23] = fma_f32x2_rn_ftz(k2[7], vn2s1_1, S2[23]);
        S2[24] = fma_f32x2_rn_ftz(k2[8], vn2s1_1, S2[24]);
        S2[25] = fma_f32x2_rn_ftz(k2[9], vn2s1_1, S2[25]);
        S2[26] = fma_f32x2_rn_ftz(k2[10], vn2s1_1, S2[26]);
        S2[27] = fma_f32x2_rn_ftz(k2[11], vn2s1_1, S2[27]);
        S2[28] = fma_f32x2_rn_ftz(k2[12], vn2s1_1, S2[28]);
        S2[29] = fma_f32x2_rn_ftz(k2[13], vn2s1_1, S2[29]);
        S2[30] = fma_f32x2_rn_ftz(k2[14], vn2s1_1, S2[30]);
        S2[31] = fma_f32x2_rn_ftz(k2[15], vn2s1_1, S2[31]);
        float2 accs1_2_0 = make_float2(0.0f, 0.0f);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[32], k2[0], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[33], k2[1], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[34], k2[2], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[35], k2[3], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[36], k2[4], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[37], k2[5], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[38], k2[6], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[39], k2[7], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[40], k2[8], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[41], k2[9], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[42], k2[10], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[43], k2[11], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[44], k2[12], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[45], k2[13], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[46], k2[14], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[47], k2[15], accs1_2_0);
        float dots1_2 = accs1_2_0.x + accs1_2_0.y;
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, dots1_2, 1);
        dots1_2 = dots1_2 + _shfl_xor_sync_55;
        float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, dots1_2, 2);
        dots1_2 = dots1_2 + _shfl_xor_sync_56;
#endif
        float vns1_2 = (v_regs[2] - dots1_2) * betas1;
#if __CUDA_ARCH__ == 900
        float vn2s1_2_x = vns1_2;
        float vn2s1_2_y = vns1_2;
        float _fma_424;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_424)
            : "f"(k_regs[0]), "f"(vn2s1_2_x), "f"(S[64]));
        float _fma_425;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_425)
            : "f"(k_regs[1]), "f"(vn2s1_2_y), "f"(S[65]));
        S[64] = _fma_424;
        S[65] = _fma_425;
        float _fma_426;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_426)
            : "f"(k_regs[2]), "f"(vn2s1_2_x), "f"(S[66]));
        float _fma_427;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_427)
            : "f"(k_regs[3]), "f"(vn2s1_2_y), "f"(S[67]));
        S[66] = _fma_426;
        S[67] = _fma_427;
        float _fma_428;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_428)
            : "f"(k_regs[4]), "f"(vn2s1_2_x), "f"(S[68]));
        float _fma_429;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_429)
            : "f"(k_regs[5]), "f"(vn2s1_2_y), "f"(S[69]));
        S[68] = _fma_428;
        S[69] = _fma_429;
        float _fma_430;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_430)
            : "f"(k_regs[6]), "f"(vn2s1_2_x), "f"(S[70]));
        float _fma_431;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_431)
            : "f"(k_regs[7]), "f"(vn2s1_2_y), "f"(S[71]));
        S[70] = _fma_430;
        S[71] = _fma_431;
        float _fma_432;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_432)
            : "f"(k_regs[8]), "f"(vn2s1_2_x), "f"(S[72]));
        float _fma_433;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_433)
            : "f"(k_regs[9]), "f"(vn2s1_2_y), "f"(S[73]));
        S[72] = _fma_432;
        S[73] = _fma_433;
        float _fma_434;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_434)
            : "f"(k_regs[10]), "f"(vn2s1_2_x), "f"(S[74]));
        float _fma_435;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_435)
            : "f"(k_regs[11]), "f"(vn2s1_2_y), "f"(S[75]));
        S[74] = _fma_434;
        S[75] = _fma_435;
        float _fma_436;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_436)
            : "f"(k_regs[12]), "f"(vn2s1_2_x), "f"(S[76]));
        float _fma_437;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_437)
            : "f"(k_regs[13]), "f"(vn2s1_2_y), "f"(S[77]));
        S[76] = _fma_436;
        S[77] = _fma_437;
        float _fma_438;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_438)
            : "f"(k_regs[14]), "f"(vn2s1_2_x), "f"(S[78]));
        float _fma_439;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_439)
            : "f"(k_regs[15]), "f"(vn2s1_2_y), "f"(S[79]));
        S[78] = _fma_438;
        S[79] = _fma_439;
        float _fma_440;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_440)
            : "f"(k_regs[16]), "f"(vn2s1_2_x), "f"(S[80]));
        float _fma_441;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_441)
            : "f"(k_regs[17]), "f"(vn2s1_2_y), "f"(S[81]));
        S[80] = _fma_440;
        S[81] = _fma_441;
        float _fma_442;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_442)
            : "f"(k_regs[18]), "f"(vn2s1_2_x), "f"(S[82]));
        float _fma_443;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_443)
            : "f"(k_regs[19]), "f"(vn2s1_2_y), "f"(S[83]));
        S[82] = _fma_442;
        S[83] = _fma_443;
        float _fma_444;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_444)
            : "f"(k_regs[20]), "f"(vn2s1_2_x), "f"(S[84]));
        float _fma_445;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_445)
            : "f"(k_regs[21]), "f"(vn2s1_2_y), "f"(S[85]));
        S[84] = _fma_444;
        S[85] = _fma_445;
        float _fma_446;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_446)
            : "f"(k_regs[22]), "f"(vn2s1_2_x), "f"(S[86]));
        float _fma_447;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_447)
            : "f"(k_regs[23]), "f"(vn2s1_2_y), "f"(S[87]));
        S[86] = _fma_446;
        S[87] = _fma_447;
        float _fma_448;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_448)
            : "f"(k_regs[24]), "f"(vn2s1_2_x), "f"(S[88]));
        float _fma_449;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_449)
            : "f"(k_regs[25]), "f"(vn2s1_2_y), "f"(S[89]));
        S[88] = _fma_448;
        S[89] = _fma_449;
        float _fma_450;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_450)
            : "f"(k_regs[26]), "f"(vn2s1_2_x), "f"(S[90]));
        float _fma_451;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_451)
            : "f"(k_regs[27]), "f"(vn2s1_2_y), "f"(S[91]));
        S[90] = _fma_450;
        S[91] = _fma_451;
        float _fma_452;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_452)
            : "f"(k_regs[28]), "f"(vn2s1_2_x), "f"(S[92]));
        float _fma_453;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_453)
            : "f"(k_regs[29]), "f"(vn2s1_2_y), "f"(S[93]));
        S[92] = _fma_452;
        S[93] = _fma_453;
        float _fma_454;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_454)
            : "f"(k_regs[30]), "f"(vn2s1_2_x), "f"(S[94]));
        float _fma_455;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_455)
            : "f"(k_regs[31]), "f"(vn2s1_2_y), "f"(S[95]));
        S[94] = _fma_454;
        S[95] = _fma_455;
        float accs1_3_0_x = 0.0f;
        float accs1_3_0_y = 0.0f;
        float accs1_3_1_x = 0.0f;
        float accs1_3_1_y = 0.0f;
        float _fma_456;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_456)
            : "f"(S[96]), "f"(k_regs[0]), "f"(accs1_3_0_x));
        float _fma_457;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_457)
            : "f"(S[97]), "f"(k_regs[1]), "f"(accs1_3_0_y));
        accs1_3_0_x = _fma_456;
        accs1_3_0_y = _fma_457;
        float _fma_458;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_458)
            : "f"(S[98]), "f"(k_regs[2]), "f"(accs1_3_1_x));
        float _fma_459;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_459)
            : "f"(S[99]), "f"(k_regs[3]), "f"(accs1_3_1_y));
        accs1_3_1_x = _fma_458;
        accs1_3_1_y = _fma_459;
        float _fma_460;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_460)
            : "f"(S[100]), "f"(k_regs[4]), "f"(accs1_3_0_x));
        float _fma_461;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_461)
            : "f"(S[101]), "f"(k_regs[5]), "f"(accs1_3_0_y));
        accs1_3_0_x = _fma_460;
        accs1_3_0_y = _fma_461;
        float _fma_462;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_462)
            : "f"(S[102]), "f"(k_regs[6]), "f"(accs1_3_1_x));
        float _fma_463;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_463)
            : "f"(S[103]), "f"(k_regs[7]), "f"(accs1_3_1_y));
        accs1_3_1_x = _fma_462;
        accs1_3_1_y = _fma_463;
        float _fma_464;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_464)
            : "f"(S[104]), "f"(k_regs[8]), "f"(accs1_3_0_x));
        float _fma_465;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_465)
            : "f"(S[105]), "f"(k_regs[9]), "f"(accs1_3_0_y));
        accs1_3_0_x = _fma_464;
        accs1_3_0_y = _fma_465;
        float _fma_466;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_466)
            : "f"(S[106]), "f"(k_regs[10]), "f"(accs1_3_1_x));
        float _fma_467;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_467)
            : "f"(S[107]), "f"(k_regs[11]), "f"(accs1_3_1_y));
        accs1_3_1_x = _fma_466;
        accs1_3_1_y = _fma_467;
        float _fma_468;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_468)
            : "f"(S[108]), "f"(k_regs[12]), "f"(accs1_3_0_x));
        float _fma_469;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_469)
            : "f"(S[109]), "f"(k_regs[13]), "f"(accs1_3_0_y));
        accs1_3_0_x = _fma_468;
        accs1_3_0_y = _fma_469;
        float _fma_470;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_470)
            : "f"(S[110]), "f"(k_regs[14]), "f"(accs1_3_1_x));
        float _fma_471;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_471)
            : "f"(S[111]), "f"(k_regs[15]), "f"(accs1_3_1_y));
        accs1_3_1_x = _fma_470;
        accs1_3_1_y = _fma_471;
        float _fma_472;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_472)
            : "f"(S[112]), "f"(k_regs[16]), "f"(accs1_3_0_x));
        float _fma_473;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_473)
            : "f"(S[113]), "f"(k_regs[17]), "f"(accs1_3_0_y));
        accs1_3_0_x = _fma_472;
        accs1_3_0_y = _fma_473;
        float _fma_474;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_474)
            : "f"(S[114]), "f"(k_regs[18]), "f"(accs1_3_1_x));
        float _fma_475;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_475)
            : "f"(S[115]), "f"(k_regs[19]), "f"(accs1_3_1_y));
        accs1_3_1_x = _fma_474;
        accs1_3_1_y = _fma_475;
        float _fma_476;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_476)
            : "f"(S[116]), "f"(k_regs[20]), "f"(accs1_3_0_x));
        float _fma_477;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_477)
            : "f"(S[117]), "f"(k_regs[21]), "f"(accs1_3_0_y));
        accs1_3_0_x = _fma_476;
        accs1_3_0_y = _fma_477;
        float _fma_478;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_478)
            : "f"(S[118]), "f"(k_regs[22]), "f"(accs1_3_1_x));
        float _fma_479;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_479)
            : "f"(S[119]), "f"(k_regs[23]), "f"(accs1_3_1_y));
        accs1_3_1_x = _fma_478;
        accs1_3_1_y = _fma_479;
        float _fma_480;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_480)
            : "f"(S[120]), "f"(k_regs[24]), "f"(accs1_3_0_x));
        float _fma_481;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_481)
            : "f"(S[121]), "f"(k_regs[25]), "f"(accs1_3_0_y));
        accs1_3_0_x = _fma_480;
        accs1_3_0_y = _fma_481;
        float _fma_482;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_482)
            : "f"(S[122]), "f"(k_regs[26]), "f"(accs1_3_1_x));
        float _fma_483;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_483)
            : "f"(S[123]), "f"(k_regs[27]), "f"(accs1_3_1_y));
        accs1_3_1_x = _fma_482;
        accs1_3_1_y = _fma_483;
        float _fma_484;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_484)
            : "f"(S[124]), "f"(k_regs[28]), "f"(accs1_3_0_x));
        float _fma_485;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_485)
            : "f"(S[125]), "f"(k_regs[29]), "f"(accs1_3_0_y));
        accs1_3_0_x = _fma_484;
        accs1_3_0_y = _fma_485;
        float _fma_486;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_486)
            : "f"(S[126]), "f"(k_regs[30]), "f"(accs1_3_1_x));
        float _fma_487;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_487)
            : "f"(S[127]), "f"(k_regs[31]), "f"(accs1_3_1_y));
        accs1_3_1_x = _fma_486;
        accs1_3_1_y = _fma_487;
        float _add_71;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(accs1_3_0_x), "f"(accs1_3_1_x));
        float _add_72;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(accs1_3_0_y), "f"(accs1_3_1_y));
        float dots1_3 = _add_71 + _add_72;
        float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, dots1_3, 1);
        dots1_3 = dots1_3 + _shfl_xor_sync_73;
        float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, dots1_3, 2);
        dots1_3 = dots1_3 + _shfl_xor_sync_74;
#else
        float2 vn2s1_2 = make_float2(vns1_2, vns1_2);
        S2[32] = fma_f32x2_rn_ftz(k2[0], vn2s1_2, S2[32]);
        S2[33] = fma_f32x2_rn_ftz(k2[1], vn2s1_2, S2[33]);
        S2[34] = fma_f32x2_rn_ftz(k2[2], vn2s1_2, S2[34]);
        S2[35] = fma_f32x2_rn_ftz(k2[3], vn2s1_2, S2[35]);
        S2[36] = fma_f32x2_rn_ftz(k2[4], vn2s1_2, S2[36]);
        S2[37] = fma_f32x2_rn_ftz(k2[5], vn2s1_2, S2[37]);
        S2[38] = fma_f32x2_rn_ftz(k2[6], vn2s1_2, S2[38]);
        S2[39] = fma_f32x2_rn_ftz(k2[7], vn2s1_2, S2[39]);
        S2[40] = fma_f32x2_rn_ftz(k2[8], vn2s1_2, S2[40]);
        S2[41] = fma_f32x2_rn_ftz(k2[9], vn2s1_2, S2[41]);
        S2[42] = fma_f32x2_rn_ftz(k2[10], vn2s1_2, S2[42]);
        S2[43] = fma_f32x2_rn_ftz(k2[11], vn2s1_2, S2[43]);
        S2[44] = fma_f32x2_rn_ftz(k2[12], vn2s1_2, S2[44]);
        S2[45] = fma_f32x2_rn_ftz(k2[13], vn2s1_2, S2[45]);
        S2[46] = fma_f32x2_rn_ftz(k2[14], vn2s1_2, S2[46]);
        S2[47] = fma_f32x2_rn_ftz(k2[15], vn2s1_2, S2[47]);
        float2 accs1_3_0 = make_float2(0.0f, 0.0f);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[48], k2[0], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[49], k2[1], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[50], k2[2], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[51], k2[3], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[52], k2[4], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[53], k2[5], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[54], k2[6], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[55], k2[7], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[56], k2[8], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[57], k2[9], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[58], k2[10], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[59], k2[11], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[60], k2[12], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[61], k2[13], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[62], k2[14], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[63], k2[15], accs1_3_0);
        float dots1_3 = accs1_3_0.x + accs1_3_0.y;
        float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, dots1_3, 1);
        dots1_3 = dots1_3 + _shfl_xor_sync_57;
        float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, dots1_3, 2);
        dots1_3 = dots1_3 + _shfl_xor_sync_58;
#endif
        float vns1_3 = (v_regs[3] - dots1_3) * betas1;
#if __CUDA_ARCH__ == 900
        float vn2s1_3_x = vns1_3;
        float vn2s1_3_y = vns1_3;
        float _fma_488;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_488)
            : "f"(k_regs[0]), "f"(vn2s1_3_x), "f"(S[96]));
        float _fma_489;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_489)
            : "f"(k_regs[1]), "f"(vn2s1_3_y), "f"(S[97]));
        S[96] = _fma_488;
        S[97] = _fma_489;
        float _fma_490;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_490)
            : "f"(k_regs[2]), "f"(vn2s1_3_x), "f"(S[98]));
        float _fma_491;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_491)
            : "f"(k_regs[3]), "f"(vn2s1_3_y), "f"(S[99]));
        S[98] = _fma_490;
        S[99] = _fma_491;
        float _fma_492;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_492)
            : "f"(k_regs[4]), "f"(vn2s1_3_x), "f"(S[100]));
        float _fma_493;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_493)
            : "f"(k_regs[5]), "f"(vn2s1_3_y), "f"(S[101]));
        S[100] = _fma_492;
        S[101] = _fma_493;
        float _fma_494;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_494)
            : "f"(k_regs[6]), "f"(vn2s1_3_x), "f"(S[102]));
        float _fma_495;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_495)
            : "f"(k_regs[7]), "f"(vn2s1_3_y), "f"(S[103]));
        S[102] = _fma_494;
        S[103] = _fma_495;
        float _fma_496;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_496)
            : "f"(k_regs[8]), "f"(vn2s1_3_x), "f"(S[104]));
        float _fma_497;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_497)
            : "f"(k_regs[9]), "f"(vn2s1_3_y), "f"(S[105]));
        S[104] = _fma_496;
        S[105] = _fma_497;
        float _fma_498;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_498)
            : "f"(k_regs[10]), "f"(vn2s1_3_x), "f"(S[106]));
        float _fma_499;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_499)
            : "f"(k_regs[11]), "f"(vn2s1_3_y), "f"(S[107]));
        S[106] = _fma_498;
        S[107] = _fma_499;
        float _fma_500;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_500)
            : "f"(k_regs[12]), "f"(vn2s1_3_x), "f"(S[108]));
        float _fma_501;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_501)
            : "f"(k_regs[13]), "f"(vn2s1_3_y), "f"(S[109]));
        S[108] = _fma_500;
        S[109] = _fma_501;
        float _fma_502;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_502)
            : "f"(k_regs[14]), "f"(vn2s1_3_x), "f"(S[110]));
        float _fma_503;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_503)
            : "f"(k_regs[15]), "f"(vn2s1_3_y), "f"(S[111]));
        S[110] = _fma_502;
        S[111] = _fma_503;
        float _fma_504;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_504)
            : "f"(k_regs[16]), "f"(vn2s1_3_x), "f"(S[112]));
        float _fma_505;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_505)
            : "f"(k_regs[17]), "f"(vn2s1_3_y), "f"(S[113]));
        S[112] = _fma_504;
        S[113] = _fma_505;
        float _fma_506;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_506)
            : "f"(k_regs[18]), "f"(vn2s1_3_x), "f"(S[114]));
        float _fma_507;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_507)
            : "f"(k_regs[19]), "f"(vn2s1_3_y), "f"(S[115]));
        S[114] = _fma_506;
        S[115] = _fma_507;
        float _fma_508;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_508)
            : "f"(k_regs[20]), "f"(vn2s1_3_x), "f"(S[116]));
        float _fma_509;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_509)
            : "f"(k_regs[21]), "f"(vn2s1_3_y), "f"(S[117]));
        S[116] = _fma_508;
        S[117] = _fma_509;
        float _fma_510;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_510)
            : "f"(k_regs[22]), "f"(vn2s1_3_x), "f"(S[118]));
        float _fma_511;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_511)
            : "f"(k_regs[23]), "f"(vn2s1_3_y), "f"(S[119]));
        S[118] = _fma_510;
        S[119] = _fma_511;
        float _fma_512;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_512)
            : "f"(k_regs[24]), "f"(vn2s1_3_x), "f"(S[120]));
        float _fma_513;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_513)
            : "f"(k_regs[25]), "f"(vn2s1_3_y), "f"(S[121]));
        S[120] = _fma_512;
        S[121] = _fma_513;
        float _fma_514;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_514)
            : "f"(k_regs[26]), "f"(vn2s1_3_x), "f"(S[122]));
        float _fma_515;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_515)
            : "f"(k_regs[27]), "f"(vn2s1_3_y), "f"(S[123]));
        S[122] = _fma_514;
        S[123] = _fma_515;
        float _fma_516;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_516)
            : "f"(k_regs[28]), "f"(vn2s1_3_x), "f"(S[124]));
        float _fma_517;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_517)
            : "f"(k_regs[29]), "f"(vn2s1_3_y), "f"(S[125]));
        S[124] = _fma_516;
        S[125] = _fma_517;
        float _fma_518;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_518)
            : "f"(k_regs[30]), "f"(vn2s1_3_x), "f"(S[126]));
        float _fma_519;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_519)
            : "f"(k_regs[31]), "f"(vn2s1_3_y), "f"(S[127]));
        S[126] = _fma_518;
        S[127] = _fma_519;
#else
        float2 vn2s1_3 = make_float2(vns1_3, vns1_3);
        S2[48] = fma_f32x2_rn_ftz(k2[0], vn2s1_3, S2[48]);
        S2[49] = fma_f32x2_rn_ftz(k2[1], vn2s1_3, S2[49]);
        S2[50] = fma_f32x2_rn_ftz(k2[2], vn2s1_3, S2[50]);
        S2[51] = fma_f32x2_rn_ftz(k2[3], vn2s1_3, S2[51]);
        S2[52] = fma_f32x2_rn_ftz(k2[4], vn2s1_3, S2[52]);
        S2[53] = fma_f32x2_rn_ftz(k2[5], vn2s1_3, S2[53]);
        S2[54] = fma_f32x2_rn_ftz(k2[6], vn2s1_3, S2[54]);
        S2[55] = fma_f32x2_rn_ftz(k2[7], vn2s1_3, S2[55]);
        S2[56] = fma_f32x2_rn_ftz(k2[8], vn2s1_3, S2[56]);
        S2[57] = fma_f32x2_rn_ftz(k2[9], vn2s1_3, S2[57]);
        S2[58] = fma_f32x2_rn_ftz(k2[10], vn2s1_3, S2[58]);
        S2[59] = fma_f32x2_rn_ftz(k2[11], vn2s1_3, S2[59]);
        S2[60] = fma_f32x2_rn_ftz(k2[12], vn2s1_3, S2[60]);
        S2[61] = fma_f32x2_rn_ftz(k2[13], vn2s1_3, S2[61]);
        S2[62] = fma_f32x2_rn_ftz(k2[14], vn2s1_3, S2[62]);
        S2[63] = fma_f32x2_rn_ftz(k2[15], vn2s1_3, S2[63]);
#endif
        int ms2 = t_1 * 3 + 2;
        unsigned int kbases2 = sK_addr + (unsigned int)(ms2 * 512) + lane_byte;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                     : "r"(kbases2));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                     : "r"(kbases2 + 64));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                     : "r"(kbases2 + 128));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                     : "r"(kbases2 + 192));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[16])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 3]))
                     : "r"(kbases2 + 256));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[20])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 3]))
                     : "r"(kbases2 + 320));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[24])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 3]))
                     : "r"(kbases2 + 384));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[28])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 3]))
                     : "r"(kbases2 + 448));
#if __CUDA_ARCH__ == 900
#else
        k2[0] = make_float2(k_regs[0], k_regs[1]);
        k2[1] = make_float2(k_regs[2], k_regs[3]);
        k2[2] = make_float2(k_regs[4], k_regs[5]);
        k2[3] = make_float2(k_regs[6], k_regs[7]);
        k2[4] = make_float2(k_regs[8], k_regs[9]);
        k2[5] = make_float2(k_regs[10], k_regs[11]);
        k2[6] = make_float2(k_regs[12], k_regs[13]);
        k2[7] = make_float2(k_regs[14], k_regs[15]);
        k2[8] = make_float2(k_regs[16], k_regs[17]);
        k2[9] = make_float2(k_regs[18], k_regs[19]);
        k2[10] = make_float2(k_regs[20], k_regs[21]);
        k2[11] = make_float2(k_regs[22], k_regs[23]);
        k2[12] = make_float2(k_regs[24], k_regs[25]);
        k2[13] = make_float2(k_regs[26], k_regs[27]);
        k2[14] = make_float2(k_regs[28], k_regs[29]);
        k2[15] = make_float2(k_regs[30], k_regs[31]);
#endif
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                     : "r"(sV_addr + (unsigned int)(ms2 * 256) + group_byte));
        float betas2 = sBeta[ms2];
#if __CUDA_ARCH__ == 900
        float accs2_0_0_x = 0.0f;
        float accs2_0_0_y = 0.0f;
        float accs2_0_1_x = 0.0f;
        float accs2_0_1_y = 0.0f;
        float _fma_520;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_520)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accs2_0_0_x));
        float _fma_521;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_521)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accs2_0_0_y));
        accs2_0_0_x = _fma_520;
        accs2_0_0_y = _fma_521;
        float _fma_522;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_522)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accs2_0_1_x));
        float _fma_523;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_523)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accs2_0_1_y));
        accs2_0_1_x = _fma_522;
        accs2_0_1_y = _fma_523;
        float _fma_524;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_524)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accs2_0_0_x));
        float _fma_525;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_525)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accs2_0_0_y));
        accs2_0_0_x = _fma_524;
        accs2_0_0_y = _fma_525;
        float _fma_526;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_526)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accs2_0_1_x));
        float _fma_527;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_527)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accs2_0_1_y));
        accs2_0_1_x = _fma_526;
        accs2_0_1_y = _fma_527;
        float _fma_528;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_528)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accs2_0_0_x));
        float _fma_529;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_529)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accs2_0_0_y));
        accs2_0_0_x = _fma_528;
        accs2_0_0_y = _fma_529;
        float _fma_530;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_530)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accs2_0_1_x));
        float _fma_531;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_531)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accs2_0_1_y));
        accs2_0_1_x = _fma_530;
        accs2_0_1_y = _fma_531;
        float _fma_532;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_532)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accs2_0_0_x));
        float _fma_533;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_533)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accs2_0_0_y));
        accs2_0_0_x = _fma_532;
        accs2_0_0_y = _fma_533;
        float _fma_534;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_534)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accs2_0_1_x));
        float _fma_535;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_535)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accs2_0_1_y));
        accs2_0_1_x = _fma_534;
        accs2_0_1_y = _fma_535;
        float _fma_536;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_536)
            : "f"(S[16]), "f"(k_regs[16]), "f"(accs2_0_0_x));
        float _fma_537;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_537)
            : "f"(S[17]), "f"(k_regs[17]), "f"(accs2_0_0_y));
        accs2_0_0_x = _fma_536;
        accs2_0_0_y = _fma_537;
        float _fma_538;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_538)
            : "f"(S[18]), "f"(k_regs[18]), "f"(accs2_0_1_x));
        float _fma_539;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_539)
            : "f"(S[19]), "f"(k_regs[19]), "f"(accs2_0_1_y));
        accs2_0_1_x = _fma_538;
        accs2_0_1_y = _fma_539;
        float _fma_540;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_540)
            : "f"(S[20]), "f"(k_regs[20]), "f"(accs2_0_0_x));
        float _fma_541;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_541)
            : "f"(S[21]), "f"(k_regs[21]), "f"(accs2_0_0_y));
        accs2_0_0_x = _fma_540;
        accs2_0_0_y = _fma_541;
        float _fma_542;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_542)
            : "f"(S[22]), "f"(k_regs[22]), "f"(accs2_0_1_x));
        float _fma_543;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_543)
            : "f"(S[23]), "f"(k_regs[23]), "f"(accs2_0_1_y));
        accs2_0_1_x = _fma_542;
        accs2_0_1_y = _fma_543;
        float _fma_544;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_544)
            : "f"(S[24]), "f"(k_regs[24]), "f"(accs2_0_0_x));
        float _fma_545;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_545)
            : "f"(S[25]), "f"(k_regs[25]), "f"(accs2_0_0_y));
        accs2_0_0_x = _fma_544;
        accs2_0_0_y = _fma_545;
        float _fma_546;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_546)
            : "f"(S[26]), "f"(k_regs[26]), "f"(accs2_0_1_x));
        float _fma_547;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_547)
            : "f"(S[27]), "f"(k_regs[27]), "f"(accs2_0_1_y));
        accs2_0_1_x = _fma_546;
        accs2_0_1_y = _fma_547;
        float _fma_548;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_548)
            : "f"(S[28]), "f"(k_regs[28]), "f"(accs2_0_0_x));
        float _fma_549;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_549)
            : "f"(S[29]), "f"(k_regs[29]), "f"(accs2_0_0_y));
        accs2_0_0_x = _fma_548;
        accs2_0_0_y = _fma_549;
        float _fma_550;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_550)
            : "f"(S[30]), "f"(k_regs[30]), "f"(accs2_0_1_x));
        float _fma_551;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_551)
            : "f"(S[31]), "f"(k_regs[31]), "f"(accs2_0_1_y));
        accs2_0_1_x = _fma_550;
        accs2_0_1_y = _fma_551;
        float _add_75;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(accs2_0_0_x), "f"(accs2_0_1_x));
        float _add_76;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(accs2_0_0_y), "f"(accs2_0_1_y));
        float dots2_0 = _add_75 + _add_76;
        float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 1);
        dots2_0 = dots2_0 + _shfl_xor_sync_77;
        float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 2);
        dots2_0 = dots2_0 + _shfl_xor_sync_78;
#else
        float2 accs2_0_0 = make_float2(0.0f, 0.0f);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accs2_0_0);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[1], k2[1], accs2_0_0);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accs2_0_0);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[3], k2[3], accs2_0_0);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accs2_0_0);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[5], k2[5], accs2_0_0);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accs2_0_0);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[7], k2[7], accs2_0_0);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[8], k2[8], accs2_0_0);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[9], k2[9], accs2_0_0);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[10], k2[10], accs2_0_0);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[11], k2[11], accs2_0_0);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[12], k2[12], accs2_0_0);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[13], k2[13], accs2_0_0);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[14], k2[14], accs2_0_0);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[15], k2[15], accs2_0_0);
        float dots2_0 = accs2_0_0.x + accs2_0_0.y;
        float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 1);
        dots2_0 = dots2_0 + _shfl_xor_sync_59;
        float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 2);
        dots2_0 = dots2_0 + _shfl_xor_sync_60;
#endif
        float vns2_0 = (v_regs[0] - dots2_0) * betas2;
#if __CUDA_ARCH__ == 900
        float vn2s2_0_x = vns2_0;
        float vn2s2_0_y = vns2_0;
        float _fma_552;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_552)
            : "f"(k_regs[0]), "f"(vn2s2_0_x), "f"(S[0]));
        float _fma_553;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_553)
            : "f"(k_regs[1]), "f"(vn2s2_0_y), "f"(S[1]));
        S[0] = _fma_552;
        S[1] = _fma_553;
        float _fma_554;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_554)
            : "f"(k_regs[2]), "f"(vn2s2_0_x), "f"(S[2]));
        float _fma_555;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_555)
            : "f"(k_regs[3]), "f"(vn2s2_0_y), "f"(S[3]));
        S[2] = _fma_554;
        S[3] = _fma_555;
        float _fma_556;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_556)
            : "f"(k_regs[4]), "f"(vn2s2_0_x), "f"(S[4]));
        float _fma_557;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_557)
            : "f"(k_regs[5]), "f"(vn2s2_0_y), "f"(S[5]));
        S[4] = _fma_556;
        S[5] = _fma_557;
        float _fma_558;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_558)
            : "f"(k_regs[6]), "f"(vn2s2_0_x), "f"(S[6]));
        float _fma_559;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_559)
            : "f"(k_regs[7]), "f"(vn2s2_0_y), "f"(S[7]));
        S[6] = _fma_558;
        S[7] = _fma_559;
        float _fma_560;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_560)
            : "f"(k_regs[8]), "f"(vn2s2_0_x), "f"(S[8]));
        float _fma_561;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_561)
            : "f"(k_regs[9]), "f"(vn2s2_0_y), "f"(S[9]));
        S[8] = _fma_560;
        S[9] = _fma_561;
        float _fma_562;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_562)
            : "f"(k_regs[10]), "f"(vn2s2_0_x), "f"(S[10]));
        float _fma_563;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_563)
            : "f"(k_regs[11]), "f"(vn2s2_0_y), "f"(S[11]));
        S[10] = _fma_562;
        S[11] = _fma_563;
        float _fma_564;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_564)
            : "f"(k_regs[12]), "f"(vn2s2_0_x), "f"(S[12]));
        float _fma_565;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_565)
            : "f"(k_regs[13]), "f"(vn2s2_0_y), "f"(S[13]));
        S[12] = _fma_564;
        S[13] = _fma_565;
        float _fma_566;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_566)
            : "f"(k_regs[14]), "f"(vn2s2_0_x), "f"(S[14]));
        float _fma_567;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_567)
            : "f"(k_regs[15]), "f"(vn2s2_0_y), "f"(S[15]));
        S[14] = _fma_566;
        S[15] = _fma_567;
        float _fma_568;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_568)
            : "f"(k_regs[16]), "f"(vn2s2_0_x), "f"(S[16]));
        float _fma_569;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_569)
            : "f"(k_regs[17]), "f"(vn2s2_0_y), "f"(S[17]));
        S[16] = _fma_568;
        S[17] = _fma_569;
        float _fma_570;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_570)
            : "f"(k_regs[18]), "f"(vn2s2_0_x), "f"(S[18]));
        float _fma_571;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_571)
            : "f"(k_regs[19]), "f"(vn2s2_0_y), "f"(S[19]));
        S[18] = _fma_570;
        S[19] = _fma_571;
        float _fma_572;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_572)
            : "f"(k_regs[20]), "f"(vn2s2_0_x), "f"(S[20]));
        float _fma_573;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_573)
            : "f"(k_regs[21]), "f"(vn2s2_0_y), "f"(S[21]));
        S[20] = _fma_572;
        S[21] = _fma_573;
        float _fma_574;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_574)
            : "f"(k_regs[22]), "f"(vn2s2_0_x), "f"(S[22]));
        float _fma_575;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_575)
            : "f"(k_regs[23]), "f"(vn2s2_0_y), "f"(S[23]));
        S[22] = _fma_574;
        S[23] = _fma_575;
        float _fma_576;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_576)
            : "f"(k_regs[24]), "f"(vn2s2_0_x), "f"(S[24]));
        float _fma_577;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_577)
            : "f"(k_regs[25]), "f"(vn2s2_0_y), "f"(S[25]));
        S[24] = _fma_576;
        S[25] = _fma_577;
        float _fma_578;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_578)
            : "f"(k_regs[26]), "f"(vn2s2_0_x), "f"(S[26]));
        float _fma_579;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_579)
            : "f"(k_regs[27]), "f"(vn2s2_0_y), "f"(S[27]));
        S[26] = _fma_578;
        S[27] = _fma_579;
        float _fma_580;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_580)
            : "f"(k_regs[28]), "f"(vn2s2_0_x), "f"(S[28]));
        float _fma_581;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_581)
            : "f"(k_regs[29]), "f"(vn2s2_0_y), "f"(S[29]));
        S[28] = _fma_580;
        S[29] = _fma_581;
        float _fma_582;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_582)
            : "f"(k_regs[30]), "f"(vn2s2_0_x), "f"(S[30]));
        float _fma_583;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_583)
            : "f"(k_regs[31]), "f"(vn2s2_0_y), "f"(S[31]));
        S[30] = _fma_582;
        S[31] = _fma_583;
        float accs2_1_0_x = 0.0f;
        float accs2_1_0_y = 0.0f;
        float accs2_1_1_x = 0.0f;
        float accs2_1_1_y = 0.0f;
        float _fma_584;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_584)
            : "f"(S[32]), "f"(k_regs[0]), "f"(accs2_1_0_x));
        float _fma_585;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_585)
            : "f"(S[33]), "f"(k_regs[1]), "f"(accs2_1_0_y));
        accs2_1_0_x = _fma_584;
        accs2_1_0_y = _fma_585;
        float _fma_586;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_586)
            : "f"(S[34]), "f"(k_regs[2]), "f"(accs2_1_1_x));
        float _fma_587;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_587)
            : "f"(S[35]), "f"(k_regs[3]), "f"(accs2_1_1_y));
        accs2_1_1_x = _fma_586;
        accs2_1_1_y = _fma_587;
        float _fma_588;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_588)
            : "f"(S[36]), "f"(k_regs[4]), "f"(accs2_1_0_x));
        float _fma_589;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_589)
            : "f"(S[37]), "f"(k_regs[5]), "f"(accs2_1_0_y));
        accs2_1_0_x = _fma_588;
        accs2_1_0_y = _fma_589;
        float _fma_590;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_590)
            : "f"(S[38]), "f"(k_regs[6]), "f"(accs2_1_1_x));
        float _fma_591;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_591)
            : "f"(S[39]), "f"(k_regs[7]), "f"(accs2_1_1_y));
        accs2_1_1_x = _fma_590;
        accs2_1_1_y = _fma_591;
        float _fma_592;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_592)
            : "f"(S[40]), "f"(k_regs[8]), "f"(accs2_1_0_x));
        float _fma_593;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_593)
            : "f"(S[41]), "f"(k_regs[9]), "f"(accs2_1_0_y));
        accs2_1_0_x = _fma_592;
        accs2_1_0_y = _fma_593;
        float _fma_594;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_594)
            : "f"(S[42]), "f"(k_regs[10]), "f"(accs2_1_1_x));
        float _fma_595;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_595)
            : "f"(S[43]), "f"(k_regs[11]), "f"(accs2_1_1_y));
        accs2_1_1_x = _fma_594;
        accs2_1_1_y = _fma_595;
        float _fma_596;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_596)
            : "f"(S[44]), "f"(k_regs[12]), "f"(accs2_1_0_x));
        float _fma_597;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_597)
            : "f"(S[45]), "f"(k_regs[13]), "f"(accs2_1_0_y));
        accs2_1_0_x = _fma_596;
        accs2_1_0_y = _fma_597;
        float _fma_598;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_598)
            : "f"(S[46]), "f"(k_regs[14]), "f"(accs2_1_1_x));
        float _fma_599;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_599)
            : "f"(S[47]), "f"(k_regs[15]), "f"(accs2_1_1_y));
        accs2_1_1_x = _fma_598;
        accs2_1_1_y = _fma_599;
        float _fma_600;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_600)
            : "f"(S[48]), "f"(k_regs[16]), "f"(accs2_1_0_x));
        float _fma_601;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_601)
            : "f"(S[49]), "f"(k_regs[17]), "f"(accs2_1_0_y));
        accs2_1_0_x = _fma_600;
        accs2_1_0_y = _fma_601;
        float _fma_602;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_602)
            : "f"(S[50]), "f"(k_regs[18]), "f"(accs2_1_1_x));
        float _fma_603;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_603)
            : "f"(S[51]), "f"(k_regs[19]), "f"(accs2_1_1_y));
        accs2_1_1_x = _fma_602;
        accs2_1_1_y = _fma_603;
        float _fma_604;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_604)
            : "f"(S[52]), "f"(k_regs[20]), "f"(accs2_1_0_x));
        float _fma_605;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_605)
            : "f"(S[53]), "f"(k_regs[21]), "f"(accs2_1_0_y));
        accs2_1_0_x = _fma_604;
        accs2_1_0_y = _fma_605;
        float _fma_606;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_606)
            : "f"(S[54]), "f"(k_regs[22]), "f"(accs2_1_1_x));
        float _fma_607;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_607)
            : "f"(S[55]), "f"(k_regs[23]), "f"(accs2_1_1_y));
        accs2_1_1_x = _fma_606;
        accs2_1_1_y = _fma_607;
        float _fma_608;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_608)
            : "f"(S[56]), "f"(k_regs[24]), "f"(accs2_1_0_x));
        float _fma_609;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_609)
            : "f"(S[57]), "f"(k_regs[25]), "f"(accs2_1_0_y));
        accs2_1_0_x = _fma_608;
        accs2_1_0_y = _fma_609;
        float _fma_610;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_610)
            : "f"(S[58]), "f"(k_regs[26]), "f"(accs2_1_1_x));
        float _fma_611;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_611)
            : "f"(S[59]), "f"(k_regs[27]), "f"(accs2_1_1_y));
        accs2_1_1_x = _fma_610;
        accs2_1_1_y = _fma_611;
        float _fma_612;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_612)
            : "f"(S[60]), "f"(k_regs[28]), "f"(accs2_1_0_x));
        float _fma_613;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_613)
            : "f"(S[61]), "f"(k_regs[29]), "f"(accs2_1_0_y));
        accs2_1_0_x = _fma_612;
        accs2_1_0_y = _fma_613;
        float _fma_614;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_614)
            : "f"(S[62]), "f"(k_regs[30]), "f"(accs2_1_1_x));
        float _fma_615;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_615)
            : "f"(S[63]), "f"(k_regs[31]), "f"(accs2_1_1_y));
        accs2_1_1_x = _fma_614;
        accs2_1_1_y = _fma_615;
        float _add_79;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(accs2_1_0_x), "f"(accs2_1_1_x));
        float _add_80;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(accs2_1_0_y), "f"(accs2_1_1_y));
        float dots2_1 = _add_79 + _add_80;
        float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 1);
        dots2_1 = dots2_1 + _shfl_xor_sync_81;
        float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 2);
        dots2_1 = dots2_1 + _shfl_xor_sync_82;
#else
        float2 vn2s2_0 = make_float2(vns2_0, vns2_0);
        S2[0] = fma_f32x2_rn_ftz(k2[0], vn2s2_0, S2[0]);
        S2[1] = fma_f32x2_rn_ftz(k2[1], vn2s2_0, S2[1]);
        S2[2] = fma_f32x2_rn_ftz(k2[2], vn2s2_0, S2[2]);
        S2[3] = fma_f32x2_rn_ftz(k2[3], vn2s2_0, S2[3]);
        S2[4] = fma_f32x2_rn_ftz(k2[4], vn2s2_0, S2[4]);
        S2[5] = fma_f32x2_rn_ftz(k2[5], vn2s2_0, S2[5]);
        S2[6] = fma_f32x2_rn_ftz(k2[6], vn2s2_0, S2[6]);
        S2[7] = fma_f32x2_rn_ftz(k2[7], vn2s2_0, S2[7]);
        S2[8] = fma_f32x2_rn_ftz(k2[8], vn2s2_0, S2[8]);
        S2[9] = fma_f32x2_rn_ftz(k2[9], vn2s2_0, S2[9]);
        S2[10] = fma_f32x2_rn_ftz(k2[10], vn2s2_0, S2[10]);
        S2[11] = fma_f32x2_rn_ftz(k2[11], vn2s2_0, S2[11]);
        S2[12] = fma_f32x2_rn_ftz(k2[12], vn2s2_0, S2[12]);
        S2[13] = fma_f32x2_rn_ftz(k2[13], vn2s2_0, S2[13]);
        S2[14] = fma_f32x2_rn_ftz(k2[14], vn2s2_0, S2[14]);
        S2[15] = fma_f32x2_rn_ftz(k2[15], vn2s2_0, S2[15]);
        float2 accs2_1_0 = make_float2(0.0f, 0.0f);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[24], k2[8], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[25], k2[9], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[26], k2[10], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[27], k2[11], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[28], k2[12], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[29], k2[13], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[30], k2[14], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[31], k2[15], accs2_1_0);
        float dots2_1 = accs2_1_0.x + accs2_1_0.y;
        float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 1);
        dots2_1 = dots2_1 + _shfl_xor_sync_61;
        float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 2);
        dots2_1 = dots2_1 + _shfl_xor_sync_62;
#endif
        float vns2_1 = (v_regs[1] - dots2_1) * betas2;
#if __CUDA_ARCH__ == 900
        float vn2s2_1_x = vns2_1;
        float vn2s2_1_y = vns2_1;
        float _fma_616;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_616)
            : "f"(k_regs[0]), "f"(vn2s2_1_x), "f"(S[32]));
        float _fma_617;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_617)
            : "f"(k_regs[1]), "f"(vn2s2_1_y), "f"(S[33]));
        S[32] = _fma_616;
        S[33] = _fma_617;
        float _fma_618;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_618)
            : "f"(k_regs[2]), "f"(vn2s2_1_x), "f"(S[34]));
        float _fma_619;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_619)
            : "f"(k_regs[3]), "f"(vn2s2_1_y), "f"(S[35]));
        S[34] = _fma_618;
        S[35] = _fma_619;
        float _fma_620;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_620)
            : "f"(k_regs[4]), "f"(vn2s2_1_x), "f"(S[36]));
        float _fma_621;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_621)
            : "f"(k_regs[5]), "f"(vn2s2_1_y), "f"(S[37]));
        S[36] = _fma_620;
        S[37] = _fma_621;
        float _fma_622;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_622)
            : "f"(k_regs[6]), "f"(vn2s2_1_x), "f"(S[38]));
        float _fma_623;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_623)
            : "f"(k_regs[7]), "f"(vn2s2_1_y), "f"(S[39]));
        S[38] = _fma_622;
        S[39] = _fma_623;
        float _fma_624;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_624)
            : "f"(k_regs[8]), "f"(vn2s2_1_x), "f"(S[40]));
        float _fma_625;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_625)
            : "f"(k_regs[9]), "f"(vn2s2_1_y), "f"(S[41]));
        S[40] = _fma_624;
        S[41] = _fma_625;
        float _fma_626;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_626)
            : "f"(k_regs[10]), "f"(vn2s2_1_x), "f"(S[42]));
        float _fma_627;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_627)
            : "f"(k_regs[11]), "f"(vn2s2_1_y), "f"(S[43]));
        S[42] = _fma_626;
        S[43] = _fma_627;
        float _fma_628;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_628)
            : "f"(k_regs[12]), "f"(vn2s2_1_x), "f"(S[44]));
        float _fma_629;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_629)
            : "f"(k_regs[13]), "f"(vn2s2_1_y), "f"(S[45]));
        S[44] = _fma_628;
        S[45] = _fma_629;
        float _fma_630;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_630)
            : "f"(k_regs[14]), "f"(vn2s2_1_x), "f"(S[46]));
        float _fma_631;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_631)
            : "f"(k_regs[15]), "f"(vn2s2_1_y), "f"(S[47]));
        S[46] = _fma_630;
        S[47] = _fma_631;
        float _fma_632;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_632)
            : "f"(k_regs[16]), "f"(vn2s2_1_x), "f"(S[48]));
        float _fma_633;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_633)
            : "f"(k_regs[17]), "f"(vn2s2_1_y), "f"(S[49]));
        S[48] = _fma_632;
        S[49] = _fma_633;
        float _fma_634;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_634)
            : "f"(k_regs[18]), "f"(vn2s2_1_x), "f"(S[50]));
        float _fma_635;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_635)
            : "f"(k_regs[19]), "f"(vn2s2_1_y), "f"(S[51]));
        S[50] = _fma_634;
        S[51] = _fma_635;
        float _fma_636;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_636)
            : "f"(k_regs[20]), "f"(vn2s2_1_x), "f"(S[52]));
        float _fma_637;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_637)
            : "f"(k_regs[21]), "f"(vn2s2_1_y), "f"(S[53]));
        S[52] = _fma_636;
        S[53] = _fma_637;
        float _fma_638;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_638)
            : "f"(k_regs[22]), "f"(vn2s2_1_x), "f"(S[54]));
        float _fma_639;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_639)
            : "f"(k_regs[23]), "f"(vn2s2_1_y), "f"(S[55]));
        S[54] = _fma_638;
        S[55] = _fma_639;
        float _fma_640;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_640)
            : "f"(k_regs[24]), "f"(vn2s2_1_x), "f"(S[56]));
        float _fma_641;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_641)
            : "f"(k_regs[25]), "f"(vn2s2_1_y), "f"(S[57]));
        S[56] = _fma_640;
        S[57] = _fma_641;
        float _fma_642;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_642)
            : "f"(k_regs[26]), "f"(vn2s2_1_x), "f"(S[58]));
        float _fma_643;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_643)
            : "f"(k_regs[27]), "f"(vn2s2_1_y), "f"(S[59]));
        S[58] = _fma_642;
        S[59] = _fma_643;
        float _fma_644;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_644)
            : "f"(k_regs[28]), "f"(vn2s2_1_x), "f"(S[60]));
        float _fma_645;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_645)
            : "f"(k_regs[29]), "f"(vn2s2_1_y), "f"(S[61]));
        S[60] = _fma_644;
        S[61] = _fma_645;
        float _fma_646;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_646)
            : "f"(k_regs[30]), "f"(vn2s2_1_x), "f"(S[62]));
        float _fma_647;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_647)
            : "f"(k_regs[31]), "f"(vn2s2_1_y), "f"(S[63]));
        S[62] = _fma_646;
        S[63] = _fma_647;
        float accs2_2_0_x = 0.0f;
        float accs2_2_0_y = 0.0f;
        float accs2_2_1_x = 0.0f;
        float accs2_2_1_y = 0.0f;
        float _fma_648;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_648)
            : "f"(S[64]), "f"(k_regs[0]), "f"(accs2_2_0_x));
        float _fma_649;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_649)
            : "f"(S[65]), "f"(k_regs[1]), "f"(accs2_2_0_y));
        accs2_2_0_x = _fma_648;
        accs2_2_0_y = _fma_649;
        float _fma_650;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_650)
            : "f"(S[66]), "f"(k_regs[2]), "f"(accs2_2_1_x));
        float _fma_651;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_651)
            : "f"(S[67]), "f"(k_regs[3]), "f"(accs2_2_1_y));
        accs2_2_1_x = _fma_650;
        accs2_2_1_y = _fma_651;
        float _fma_652;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_652)
            : "f"(S[68]), "f"(k_regs[4]), "f"(accs2_2_0_x));
        float _fma_653;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_653)
            : "f"(S[69]), "f"(k_regs[5]), "f"(accs2_2_0_y));
        accs2_2_0_x = _fma_652;
        accs2_2_0_y = _fma_653;
        float _fma_654;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_654)
            : "f"(S[70]), "f"(k_regs[6]), "f"(accs2_2_1_x));
        float _fma_655;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_655)
            : "f"(S[71]), "f"(k_regs[7]), "f"(accs2_2_1_y));
        accs2_2_1_x = _fma_654;
        accs2_2_1_y = _fma_655;
        float _fma_656;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_656)
            : "f"(S[72]), "f"(k_regs[8]), "f"(accs2_2_0_x));
        float _fma_657;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_657)
            : "f"(S[73]), "f"(k_regs[9]), "f"(accs2_2_0_y));
        accs2_2_0_x = _fma_656;
        accs2_2_0_y = _fma_657;
        float _fma_658;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_658)
            : "f"(S[74]), "f"(k_regs[10]), "f"(accs2_2_1_x));
        float _fma_659;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_659)
            : "f"(S[75]), "f"(k_regs[11]), "f"(accs2_2_1_y));
        accs2_2_1_x = _fma_658;
        accs2_2_1_y = _fma_659;
        float _fma_660;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_660)
            : "f"(S[76]), "f"(k_regs[12]), "f"(accs2_2_0_x));
        float _fma_661;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_661)
            : "f"(S[77]), "f"(k_regs[13]), "f"(accs2_2_0_y));
        accs2_2_0_x = _fma_660;
        accs2_2_0_y = _fma_661;
        float _fma_662;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_662)
            : "f"(S[78]), "f"(k_regs[14]), "f"(accs2_2_1_x));
        float _fma_663;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_663)
            : "f"(S[79]), "f"(k_regs[15]), "f"(accs2_2_1_y));
        accs2_2_1_x = _fma_662;
        accs2_2_1_y = _fma_663;
        float _fma_664;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_664)
            : "f"(S[80]), "f"(k_regs[16]), "f"(accs2_2_0_x));
        float _fma_665;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_665)
            : "f"(S[81]), "f"(k_regs[17]), "f"(accs2_2_0_y));
        accs2_2_0_x = _fma_664;
        accs2_2_0_y = _fma_665;
        float _fma_666;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_666)
            : "f"(S[82]), "f"(k_regs[18]), "f"(accs2_2_1_x));
        float _fma_667;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_667)
            : "f"(S[83]), "f"(k_regs[19]), "f"(accs2_2_1_y));
        accs2_2_1_x = _fma_666;
        accs2_2_1_y = _fma_667;
        float _fma_668;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_668)
            : "f"(S[84]), "f"(k_regs[20]), "f"(accs2_2_0_x));
        float _fma_669;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_669)
            : "f"(S[85]), "f"(k_regs[21]), "f"(accs2_2_0_y));
        accs2_2_0_x = _fma_668;
        accs2_2_0_y = _fma_669;
        float _fma_670;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_670)
            : "f"(S[86]), "f"(k_regs[22]), "f"(accs2_2_1_x));
        float _fma_671;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_671)
            : "f"(S[87]), "f"(k_regs[23]), "f"(accs2_2_1_y));
        accs2_2_1_x = _fma_670;
        accs2_2_1_y = _fma_671;
        float _fma_672;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_672)
            : "f"(S[88]), "f"(k_regs[24]), "f"(accs2_2_0_x));
        float _fma_673;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_673)
            : "f"(S[89]), "f"(k_regs[25]), "f"(accs2_2_0_y));
        accs2_2_0_x = _fma_672;
        accs2_2_0_y = _fma_673;
        float _fma_674;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_674)
            : "f"(S[90]), "f"(k_regs[26]), "f"(accs2_2_1_x));
        float _fma_675;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_675)
            : "f"(S[91]), "f"(k_regs[27]), "f"(accs2_2_1_y));
        accs2_2_1_x = _fma_674;
        accs2_2_1_y = _fma_675;
        float _fma_676;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_676)
            : "f"(S[92]), "f"(k_regs[28]), "f"(accs2_2_0_x));
        float _fma_677;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_677)
            : "f"(S[93]), "f"(k_regs[29]), "f"(accs2_2_0_y));
        accs2_2_0_x = _fma_676;
        accs2_2_0_y = _fma_677;
        float _fma_678;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_678)
            : "f"(S[94]), "f"(k_regs[30]), "f"(accs2_2_1_x));
        float _fma_679;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_679)
            : "f"(S[95]), "f"(k_regs[31]), "f"(accs2_2_1_y));
        accs2_2_1_x = _fma_678;
        accs2_2_1_y = _fma_679;
        float _add_83;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(accs2_2_0_x), "f"(accs2_2_1_x));
        float _add_84;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(accs2_2_0_y), "f"(accs2_2_1_y));
        float dots2_2 = _add_83 + _add_84;
        float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, dots2_2, 1);
        dots2_2 = dots2_2 + _shfl_xor_sync_85;
        float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, dots2_2, 2);
        dots2_2 = dots2_2 + _shfl_xor_sync_86;
#else
        float2 vn2s2_1 = make_float2(vns2_1, vns2_1);
        S2[16] = fma_f32x2_rn_ftz(k2[0], vn2s2_1, S2[16]);
        S2[17] = fma_f32x2_rn_ftz(k2[1], vn2s2_1, S2[17]);
        S2[18] = fma_f32x2_rn_ftz(k2[2], vn2s2_1, S2[18]);
        S2[19] = fma_f32x2_rn_ftz(k2[3], vn2s2_1, S2[19]);
        S2[20] = fma_f32x2_rn_ftz(k2[4], vn2s2_1, S2[20]);
        S2[21] = fma_f32x2_rn_ftz(k2[5], vn2s2_1, S2[21]);
        S2[22] = fma_f32x2_rn_ftz(k2[6], vn2s2_1, S2[22]);
        S2[23] = fma_f32x2_rn_ftz(k2[7], vn2s2_1, S2[23]);
        S2[24] = fma_f32x2_rn_ftz(k2[8], vn2s2_1, S2[24]);
        S2[25] = fma_f32x2_rn_ftz(k2[9], vn2s2_1, S2[25]);
        S2[26] = fma_f32x2_rn_ftz(k2[10], vn2s2_1, S2[26]);
        S2[27] = fma_f32x2_rn_ftz(k2[11], vn2s2_1, S2[27]);
        S2[28] = fma_f32x2_rn_ftz(k2[12], vn2s2_1, S2[28]);
        S2[29] = fma_f32x2_rn_ftz(k2[13], vn2s2_1, S2[29]);
        S2[30] = fma_f32x2_rn_ftz(k2[14], vn2s2_1, S2[30]);
        S2[31] = fma_f32x2_rn_ftz(k2[15], vn2s2_1, S2[31]);
        float2 accs2_2_0 = make_float2(0.0f, 0.0f);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[32], k2[0], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[33], k2[1], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[34], k2[2], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[35], k2[3], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[36], k2[4], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[37], k2[5], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[38], k2[6], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[39], k2[7], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[40], k2[8], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[41], k2[9], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[42], k2[10], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[43], k2[11], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[44], k2[12], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[45], k2[13], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[46], k2[14], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[47], k2[15], accs2_2_0);
        float dots2_2 = accs2_2_0.x + accs2_2_0.y;
        float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, dots2_2, 1);
        dots2_2 = dots2_2 + _shfl_xor_sync_63;
        float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, dots2_2, 2);
        dots2_2 = dots2_2 + _shfl_xor_sync_64;
#endif
        float vns2_2 = (v_regs[2] - dots2_2) * betas2;
#if __CUDA_ARCH__ == 900
        float vn2s2_2_x = vns2_2;
        float vn2s2_2_y = vns2_2;
        float _fma_680;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_680)
            : "f"(k_regs[0]), "f"(vn2s2_2_x), "f"(S[64]));
        float _fma_681;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_681)
            : "f"(k_regs[1]), "f"(vn2s2_2_y), "f"(S[65]));
        S[64] = _fma_680;
        S[65] = _fma_681;
        float _fma_682;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_682)
            : "f"(k_regs[2]), "f"(vn2s2_2_x), "f"(S[66]));
        float _fma_683;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_683)
            : "f"(k_regs[3]), "f"(vn2s2_2_y), "f"(S[67]));
        S[66] = _fma_682;
        S[67] = _fma_683;
        float _fma_684;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_684)
            : "f"(k_regs[4]), "f"(vn2s2_2_x), "f"(S[68]));
        float _fma_685;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_685)
            : "f"(k_regs[5]), "f"(vn2s2_2_y), "f"(S[69]));
        S[68] = _fma_684;
        S[69] = _fma_685;
        float _fma_686;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_686)
            : "f"(k_regs[6]), "f"(vn2s2_2_x), "f"(S[70]));
        float _fma_687;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_687)
            : "f"(k_regs[7]), "f"(vn2s2_2_y), "f"(S[71]));
        S[70] = _fma_686;
        S[71] = _fma_687;
        float _fma_688;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_688)
            : "f"(k_regs[8]), "f"(vn2s2_2_x), "f"(S[72]));
        float _fma_689;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_689)
            : "f"(k_regs[9]), "f"(vn2s2_2_y), "f"(S[73]));
        S[72] = _fma_688;
        S[73] = _fma_689;
        float _fma_690;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_690)
            : "f"(k_regs[10]), "f"(vn2s2_2_x), "f"(S[74]));
        float _fma_691;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_691)
            : "f"(k_regs[11]), "f"(vn2s2_2_y), "f"(S[75]));
        S[74] = _fma_690;
        S[75] = _fma_691;
        float _fma_692;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_692)
            : "f"(k_regs[12]), "f"(vn2s2_2_x), "f"(S[76]));
        float _fma_693;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_693)
            : "f"(k_regs[13]), "f"(vn2s2_2_y), "f"(S[77]));
        S[76] = _fma_692;
        S[77] = _fma_693;
        float _fma_694;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_694)
            : "f"(k_regs[14]), "f"(vn2s2_2_x), "f"(S[78]));
        float _fma_695;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_695)
            : "f"(k_regs[15]), "f"(vn2s2_2_y), "f"(S[79]));
        S[78] = _fma_694;
        S[79] = _fma_695;
        float _fma_696;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_696)
            : "f"(k_regs[16]), "f"(vn2s2_2_x), "f"(S[80]));
        float _fma_697;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_697)
            : "f"(k_regs[17]), "f"(vn2s2_2_y), "f"(S[81]));
        S[80] = _fma_696;
        S[81] = _fma_697;
        float _fma_698;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_698)
            : "f"(k_regs[18]), "f"(vn2s2_2_x), "f"(S[82]));
        float _fma_699;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_699)
            : "f"(k_regs[19]), "f"(vn2s2_2_y), "f"(S[83]));
        S[82] = _fma_698;
        S[83] = _fma_699;
        float _fma_700;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_700)
            : "f"(k_regs[20]), "f"(vn2s2_2_x), "f"(S[84]));
        float _fma_701;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_701)
            : "f"(k_regs[21]), "f"(vn2s2_2_y), "f"(S[85]));
        S[84] = _fma_700;
        S[85] = _fma_701;
        float _fma_702;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_702)
            : "f"(k_regs[22]), "f"(vn2s2_2_x), "f"(S[86]));
        float _fma_703;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_703)
            : "f"(k_regs[23]), "f"(vn2s2_2_y), "f"(S[87]));
        S[86] = _fma_702;
        S[87] = _fma_703;
        float _fma_704;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_704)
            : "f"(k_regs[24]), "f"(vn2s2_2_x), "f"(S[88]));
        float _fma_705;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_705)
            : "f"(k_regs[25]), "f"(vn2s2_2_y), "f"(S[89]));
        S[88] = _fma_704;
        S[89] = _fma_705;
        float _fma_706;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_706)
            : "f"(k_regs[26]), "f"(vn2s2_2_x), "f"(S[90]));
        float _fma_707;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_707)
            : "f"(k_regs[27]), "f"(vn2s2_2_y), "f"(S[91]));
        S[90] = _fma_706;
        S[91] = _fma_707;
        float _fma_708;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_708)
            : "f"(k_regs[28]), "f"(vn2s2_2_x), "f"(S[92]));
        float _fma_709;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_709)
            : "f"(k_regs[29]), "f"(vn2s2_2_y), "f"(S[93]));
        S[92] = _fma_708;
        S[93] = _fma_709;
        float _fma_710;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_710)
            : "f"(k_regs[30]), "f"(vn2s2_2_x), "f"(S[94]));
        float _fma_711;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_711)
            : "f"(k_regs[31]), "f"(vn2s2_2_y), "f"(S[95]));
        S[94] = _fma_710;
        S[95] = _fma_711;
        float accs2_3_0_x = 0.0f;
        float accs2_3_0_y = 0.0f;
        float accs2_3_1_x = 0.0f;
        float accs2_3_1_y = 0.0f;
        float _fma_712;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_712)
            : "f"(S[96]), "f"(k_regs[0]), "f"(accs2_3_0_x));
        float _fma_713;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_713)
            : "f"(S[97]), "f"(k_regs[1]), "f"(accs2_3_0_y));
        accs2_3_0_x = _fma_712;
        accs2_3_0_y = _fma_713;
        float _fma_714;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_714)
            : "f"(S[98]), "f"(k_regs[2]), "f"(accs2_3_1_x));
        float _fma_715;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_715)
            : "f"(S[99]), "f"(k_regs[3]), "f"(accs2_3_1_y));
        accs2_3_1_x = _fma_714;
        accs2_3_1_y = _fma_715;
        float _fma_716;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_716)
            : "f"(S[100]), "f"(k_regs[4]), "f"(accs2_3_0_x));
        float _fma_717;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_717)
            : "f"(S[101]), "f"(k_regs[5]), "f"(accs2_3_0_y));
        accs2_3_0_x = _fma_716;
        accs2_3_0_y = _fma_717;
        float _fma_718;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_718)
            : "f"(S[102]), "f"(k_regs[6]), "f"(accs2_3_1_x));
        float _fma_719;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_719)
            : "f"(S[103]), "f"(k_regs[7]), "f"(accs2_3_1_y));
        accs2_3_1_x = _fma_718;
        accs2_3_1_y = _fma_719;
        float _fma_720;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_720)
            : "f"(S[104]), "f"(k_regs[8]), "f"(accs2_3_0_x));
        float _fma_721;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_721)
            : "f"(S[105]), "f"(k_regs[9]), "f"(accs2_3_0_y));
        accs2_3_0_x = _fma_720;
        accs2_3_0_y = _fma_721;
        float _fma_722;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_722)
            : "f"(S[106]), "f"(k_regs[10]), "f"(accs2_3_1_x));
        float _fma_723;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_723)
            : "f"(S[107]), "f"(k_regs[11]), "f"(accs2_3_1_y));
        accs2_3_1_x = _fma_722;
        accs2_3_1_y = _fma_723;
        float _fma_724;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_724)
            : "f"(S[108]), "f"(k_regs[12]), "f"(accs2_3_0_x));
        float _fma_725;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_725)
            : "f"(S[109]), "f"(k_regs[13]), "f"(accs2_3_0_y));
        accs2_3_0_x = _fma_724;
        accs2_3_0_y = _fma_725;
        float _fma_726;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_726)
            : "f"(S[110]), "f"(k_regs[14]), "f"(accs2_3_1_x));
        float _fma_727;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_727)
            : "f"(S[111]), "f"(k_regs[15]), "f"(accs2_3_1_y));
        accs2_3_1_x = _fma_726;
        accs2_3_1_y = _fma_727;
        float _fma_728;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_728)
            : "f"(S[112]), "f"(k_regs[16]), "f"(accs2_3_0_x));
        float _fma_729;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_729)
            : "f"(S[113]), "f"(k_regs[17]), "f"(accs2_3_0_y));
        accs2_3_0_x = _fma_728;
        accs2_3_0_y = _fma_729;
        float _fma_730;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_730)
            : "f"(S[114]), "f"(k_regs[18]), "f"(accs2_3_1_x));
        float _fma_731;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_731)
            : "f"(S[115]), "f"(k_regs[19]), "f"(accs2_3_1_y));
        accs2_3_1_x = _fma_730;
        accs2_3_1_y = _fma_731;
        float _fma_732;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_732)
            : "f"(S[116]), "f"(k_regs[20]), "f"(accs2_3_0_x));
        float _fma_733;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_733)
            : "f"(S[117]), "f"(k_regs[21]), "f"(accs2_3_0_y));
        accs2_3_0_x = _fma_732;
        accs2_3_0_y = _fma_733;
        float _fma_734;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_734)
            : "f"(S[118]), "f"(k_regs[22]), "f"(accs2_3_1_x));
        float _fma_735;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_735)
            : "f"(S[119]), "f"(k_regs[23]), "f"(accs2_3_1_y));
        accs2_3_1_x = _fma_734;
        accs2_3_1_y = _fma_735;
        float _fma_736;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_736)
            : "f"(S[120]), "f"(k_regs[24]), "f"(accs2_3_0_x));
        float _fma_737;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_737)
            : "f"(S[121]), "f"(k_regs[25]), "f"(accs2_3_0_y));
        accs2_3_0_x = _fma_736;
        accs2_3_0_y = _fma_737;
        float _fma_738;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_738)
            : "f"(S[122]), "f"(k_regs[26]), "f"(accs2_3_1_x));
        float _fma_739;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_739)
            : "f"(S[123]), "f"(k_regs[27]), "f"(accs2_3_1_y));
        accs2_3_1_x = _fma_738;
        accs2_3_1_y = _fma_739;
        float _fma_740;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_740)
            : "f"(S[124]), "f"(k_regs[28]), "f"(accs2_3_0_x));
        float _fma_741;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_741)
            : "f"(S[125]), "f"(k_regs[29]), "f"(accs2_3_0_y));
        accs2_3_0_x = _fma_740;
        accs2_3_0_y = _fma_741;
        float _fma_742;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_742)
            : "f"(S[126]), "f"(k_regs[30]), "f"(accs2_3_1_x));
        float _fma_743;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_743)
            : "f"(S[127]), "f"(k_regs[31]), "f"(accs2_3_1_y));
        accs2_3_1_x = _fma_742;
        accs2_3_1_y = _fma_743;
        float _add_87;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(accs2_3_0_x), "f"(accs2_3_1_x));
        float _add_88;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(accs2_3_0_y), "f"(accs2_3_1_y));
        float dots2_3 = _add_87 + _add_88;
        float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, dots2_3, 1);
        dots2_3 = dots2_3 + _shfl_xor_sync_89;
        float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, dots2_3, 2);
        dots2_3 = dots2_3 + _shfl_xor_sync_90;
#else
        float2 vn2s2_2 = make_float2(vns2_2, vns2_2);
        S2[32] = fma_f32x2_rn_ftz(k2[0], vn2s2_2, S2[32]);
        S2[33] = fma_f32x2_rn_ftz(k2[1], vn2s2_2, S2[33]);
        S2[34] = fma_f32x2_rn_ftz(k2[2], vn2s2_2, S2[34]);
        S2[35] = fma_f32x2_rn_ftz(k2[3], vn2s2_2, S2[35]);
        S2[36] = fma_f32x2_rn_ftz(k2[4], vn2s2_2, S2[36]);
        S2[37] = fma_f32x2_rn_ftz(k2[5], vn2s2_2, S2[37]);
        S2[38] = fma_f32x2_rn_ftz(k2[6], vn2s2_2, S2[38]);
        S2[39] = fma_f32x2_rn_ftz(k2[7], vn2s2_2, S2[39]);
        S2[40] = fma_f32x2_rn_ftz(k2[8], vn2s2_2, S2[40]);
        S2[41] = fma_f32x2_rn_ftz(k2[9], vn2s2_2, S2[41]);
        S2[42] = fma_f32x2_rn_ftz(k2[10], vn2s2_2, S2[42]);
        S2[43] = fma_f32x2_rn_ftz(k2[11], vn2s2_2, S2[43]);
        S2[44] = fma_f32x2_rn_ftz(k2[12], vn2s2_2, S2[44]);
        S2[45] = fma_f32x2_rn_ftz(k2[13], vn2s2_2, S2[45]);
        S2[46] = fma_f32x2_rn_ftz(k2[14], vn2s2_2, S2[46]);
        S2[47] = fma_f32x2_rn_ftz(k2[15], vn2s2_2, S2[47]);
        float2 accs2_3_0 = make_float2(0.0f, 0.0f);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[48], k2[0], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[49], k2[1], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[50], k2[2], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[51], k2[3], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[52], k2[4], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[53], k2[5], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[54], k2[6], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[55], k2[7], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[56], k2[8], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[57], k2[9], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[58], k2[10], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[59], k2[11], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[60], k2[12], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[61], k2[13], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[62], k2[14], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[63], k2[15], accs2_3_0);
        float dots2_3 = accs2_3_0.x + accs2_3_0.y;
        float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, dots2_3, 1);
        dots2_3 = dots2_3 + _shfl_xor_sync_65;
        float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, dots2_3, 2);
        dots2_3 = dots2_3 + _shfl_xor_sync_66;
#endif
        float vns2_3 = (v_regs[3] - dots2_3) * betas2;
#if __CUDA_ARCH__ == 900
        float vn2s2_3_x = vns2_3;
        float vn2s2_3_y = vns2_3;
        float _fma_744;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_744)
            : "f"(k_regs[0]), "f"(vn2s2_3_x), "f"(S[96]));
        float _fma_745;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_745)
            : "f"(k_regs[1]), "f"(vn2s2_3_y), "f"(S[97]));
        S[96] = _fma_744;
        S[97] = _fma_745;
        float _fma_746;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_746)
            : "f"(k_regs[2]), "f"(vn2s2_3_x), "f"(S[98]));
        float _fma_747;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_747)
            : "f"(k_regs[3]), "f"(vn2s2_3_y), "f"(S[99]));
        S[98] = _fma_746;
        S[99] = _fma_747;
        float _fma_748;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_748)
            : "f"(k_regs[4]), "f"(vn2s2_3_x), "f"(S[100]));
        float _fma_749;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_749)
            : "f"(k_regs[5]), "f"(vn2s2_3_y), "f"(S[101]));
        S[100] = _fma_748;
        S[101] = _fma_749;
        float _fma_750;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_750)
            : "f"(k_regs[6]), "f"(vn2s2_3_x), "f"(S[102]));
        float _fma_751;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_751)
            : "f"(k_regs[7]), "f"(vn2s2_3_y), "f"(S[103]));
        S[102] = _fma_750;
        S[103] = _fma_751;
        float _fma_752;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_752)
            : "f"(k_regs[8]), "f"(vn2s2_3_x), "f"(S[104]));
        float _fma_753;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_753)
            : "f"(k_regs[9]), "f"(vn2s2_3_y), "f"(S[105]));
        S[104] = _fma_752;
        S[105] = _fma_753;
        float _fma_754;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_754)
            : "f"(k_regs[10]), "f"(vn2s2_3_x), "f"(S[106]));
        float _fma_755;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_755)
            : "f"(k_regs[11]), "f"(vn2s2_3_y), "f"(S[107]));
        S[106] = _fma_754;
        S[107] = _fma_755;
        float _fma_756;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_756)
            : "f"(k_regs[12]), "f"(vn2s2_3_x), "f"(S[108]));
        float _fma_757;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_757)
            : "f"(k_regs[13]), "f"(vn2s2_3_y), "f"(S[109]));
        S[108] = _fma_756;
        S[109] = _fma_757;
        float _fma_758;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_758)
            : "f"(k_regs[14]), "f"(vn2s2_3_x), "f"(S[110]));
        float _fma_759;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_759)
            : "f"(k_regs[15]), "f"(vn2s2_3_y), "f"(S[111]));
        S[110] = _fma_758;
        S[111] = _fma_759;
        float _fma_760;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_760)
            : "f"(k_regs[16]), "f"(vn2s2_3_x), "f"(S[112]));
        float _fma_761;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_761)
            : "f"(k_regs[17]), "f"(vn2s2_3_y), "f"(S[113]));
        S[112] = _fma_760;
        S[113] = _fma_761;
        float _fma_762;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_762)
            : "f"(k_regs[18]), "f"(vn2s2_3_x), "f"(S[114]));
        float _fma_763;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_763)
            : "f"(k_regs[19]), "f"(vn2s2_3_y), "f"(S[115]));
        S[114] = _fma_762;
        S[115] = _fma_763;
        float _fma_764;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_764)
            : "f"(k_regs[20]), "f"(vn2s2_3_x), "f"(S[116]));
        float _fma_765;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_765)
            : "f"(k_regs[21]), "f"(vn2s2_3_y), "f"(S[117]));
        S[116] = _fma_764;
        S[117] = _fma_765;
        float _fma_766;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_766)
            : "f"(k_regs[22]), "f"(vn2s2_3_x), "f"(S[118]));
        float _fma_767;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_767)
            : "f"(k_regs[23]), "f"(vn2s2_3_y), "f"(S[119]));
        S[118] = _fma_766;
        S[119] = _fma_767;
        float _fma_768;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_768)
            : "f"(k_regs[24]), "f"(vn2s2_3_x), "f"(S[120]));
        float _fma_769;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_769)
            : "f"(k_regs[25]), "f"(vn2s2_3_y), "f"(S[121]));
        S[120] = _fma_768;
        S[121] = _fma_769;
        float _fma_770;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_770)
            : "f"(k_regs[26]), "f"(vn2s2_3_x), "f"(S[122]));
        float _fma_771;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_771)
            : "f"(k_regs[27]), "f"(vn2s2_3_y), "f"(S[123]));
        S[122] = _fma_770;
        S[123] = _fma_771;
        float _fma_772;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_772)
            : "f"(k_regs[28]), "f"(vn2s2_3_x), "f"(S[124]));
        float _fma_773;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_773)
            : "f"(k_regs[29]), "f"(vn2s2_3_y), "f"(S[125]));
        S[124] = _fma_772;
        S[125] = _fma_773;
        float _fma_774;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_774)
            : "f"(k_regs[30]), "f"(vn2s2_3_x), "f"(S[126]));
        float _fma_775;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_775)
            : "f"(k_regs[31]), "f"(vn2s2_3_y), "f"(S[127]));
        S[126] = _fma_774;
        S[127] = _fma_775;
#else
        float2 vn2s2_3 = make_float2(vns2_3, vns2_3);
        S2[48] = fma_f32x2_rn_ftz(k2[0], vn2s2_3, S2[48]);
        S2[49] = fma_f32x2_rn_ftz(k2[1], vn2s2_3, S2[49]);
        S2[50] = fma_f32x2_rn_ftz(k2[2], vn2s2_3, S2[50]);
        S2[51] = fma_f32x2_rn_ftz(k2[3], vn2s2_3, S2[51]);
        S2[52] = fma_f32x2_rn_ftz(k2[4], vn2s2_3, S2[52]);
        S2[53] = fma_f32x2_rn_ftz(k2[5], vn2s2_3, S2[53]);
        S2[54] = fma_f32x2_rn_ftz(k2[6], vn2s2_3, S2[54]);
        S2[55] = fma_f32x2_rn_ftz(k2[7], vn2s2_3, S2[55]);
        S2[56] = fma_f32x2_rn_ftz(k2[8], vn2s2_3, S2[56]);
        S2[57] = fma_f32x2_rn_ftz(k2[9], vn2s2_3, S2[57]);
        S2[58] = fma_f32x2_rn_ftz(k2[10], vn2s2_3, S2[58]);
        S2[59] = fma_f32x2_rn_ftz(k2[11], vn2s2_3, S2[59]);
        S2[60] = fma_f32x2_rn_ftz(k2[12], vn2s2_3, S2[60]);
        S2[61] = fma_f32x2_rn_ftz(k2[13], vn2s2_3, S2[61]);
        S2[62] = fma_f32x2_rn_ftz(k2[14], vn2s2_3, S2[62]);
        S2[63] = fma_f32x2_rn_ftz(k2[15], vn2s2_3, S2[63]);
#endif
        unsigned int qbases = sQ_addr + (unsigned int)(t_1 * 512) + lane_byte;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                     : "r"(qbases));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                     : "r"(qbases + 64));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                     : "r"(qbases + 128));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                     : "r"(qbases + 192));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[16])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 3]))
                     : "r"(qbases + 256));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[20])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 3]))
                     : "r"(qbases + 320));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[24])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 3]))
                     : "r"(qbases + 384));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[28])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 3]))
                     : "r"(qbases + 448));
#if __CUDA_ARCH__ == 900
        float accsq_0_0_x = 0.0f;
        float accsq_0_0_y = 0.0f;
        float accsq_0_1_x = 0.0f;
        float accsq_0_1_y = 0.0f;
        float _fma_776;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_776)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accsq_0_0_x));
        float _fma_777;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_777)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accsq_0_0_y));
        accsq_0_0_x = _fma_776;
        accsq_0_0_y = _fma_777;
        float _fma_778;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_778)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accsq_0_1_x));
        float _fma_779;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_779)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accsq_0_1_y));
        accsq_0_1_x = _fma_778;
        accsq_0_1_y = _fma_779;
        float _fma_780;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_780)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accsq_0_0_x));
        float _fma_781;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_781)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accsq_0_0_y));
        accsq_0_0_x = _fma_780;
        accsq_0_0_y = _fma_781;
        float _fma_782;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_782)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accsq_0_1_x));
        float _fma_783;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_783)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accsq_0_1_y));
        accsq_0_1_x = _fma_782;
        accsq_0_1_y = _fma_783;
        float _fma_784;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_784)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accsq_0_0_x));
        float _fma_785;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_785)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accsq_0_0_y));
        accsq_0_0_x = _fma_784;
        accsq_0_0_y = _fma_785;
        float _fma_786;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_786)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accsq_0_1_x));
        float _fma_787;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_787)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accsq_0_1_y));
        accsq_0_1_x = _fma_786;
        accsq_0_1_y = _fma_787;
        float _fma_788;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_788)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accsq_0_0_x));
        float _fma_789;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_789)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accsq_0_0_y));
        accsq_0_0_x = _fma_788;
        accsq_0_0_y = _fma_789;
        float _fma_790;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_790)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accsq_0_1_x));
        float _fma_791;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_791)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accsq_0_1_y));
        accsq_0_1_x = _fma_790;
        accsq_0_1_y = _fma_791;
        float _fma_792;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_792)
            : "f"(S[16]), "f"(k_regs[16]), "f"(accsq_0_0_x));
        float _fma_793;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_793)
            : "f"(S[17]), "f"(k_regs[17]), "f"(accsq_0_0_y));
        accsq_0_0_x = _fma_792;
        accsq_0_0_y = _fma_793;
        float _fma_794;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_794)
            : "f"(S[18]), "f"(k_regs[18]), "f"(accsq_0_1_x));
        float _fma_795;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_795)
            : "f"(S[19]), "f"(k_regs[19]), "f"(accsq_0_1_y));
        accsq_0_1_x = _fma_794;
        accsq_0_1_y = _fma_795;
        float _fma_796;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_796)
            : "f"(S[20]), "f"(k_regs[20]), "f"(accsq_0_0_x));
        float _fma_797;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_797)
            : "f"(S[21]), "f"(k_regs[21]), "f"(accsq_0_0_y));
        accsq_0_0_x = _fma_796;
        accsq_0_0_y = _fma_797;
        float _fma_798;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_798)
            : "f"(S[22]), "f"(k_regs[22]), "f"(accsq_0_1_x));
        float _fma_799;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_799)
            : "f"(S[23]), "f"(k_regs[23]), "f"(accsq_0_1_y));
        accsq_0_1_x = _fma_798;
        accsq_0_1_y = _fma_799;
        float _fma_800;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_800)
            : "f"(S[24]), "f"(k_regs[24]), "f"(accsq_0_0_x));
        float _fma_801;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_801)
            : "f"(S[25]), "f"(k_regs[25]), "f"(accsq_0_0_y));
        accsq_0_0_x = _fma_800;
        accsq_0_0_y = _fma_801;
        float _fma_802;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_802)
            : "f"(S[26]), "f"(k_regs[26]), "f"(accsq_0_1_x));
        float _fma_803;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_803)
            : "f"(S[27]), "f"(k_regs[27]), "f"(accsq_0_1_y));
        accsq_0_1_x = _fma_802;
        accsq_0_1_y = _fma_803;
        float _fma_804;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_804)
            : "f"(S[28]), "f"(k_regs[28]), "f"(accsq_0_0_x));
        float _fma_805;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_805)
            : "f"(S[29]), "f"(k_regs[29]), "f"(accsq_0_0_y));
        accsq_0_0_x = _fma_804;
        accsq_0_0_y = _fma_805;
        float _fma_806;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_806)
            : "f"(S[30]), "f"(k_regs[30]), "f"(accsq_0_1_x));
        float _fma_807;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_807)
            : "f"(S[31]), "f"(k_regs[31]), "f"(accsq_0_1_y));
        accsq_0_1_x = _fma_806;
        accsq_0_1_y = _fma_807;
        float _add_91;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(accsq_0_0_x), "f"(accsq_0_1_x));
        float _add_92;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(accsq_0_0_y), "f"(accsq_0_1_y));
        float dotsq_0 = _add_91 + _add_92;
        float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 1);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_93;
        float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 2);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_94;
#else
        k2[0] = make_float2(k_regs[0], k_regs[1]);
        k2[1] = make_float2(k_regs[2], k_regs[3]);
        k2[2] = make_float2(k_regs[4], k_regs[5]);
        k2[3] = make_float2(k_regs[6], k_regs[7]);
        k2[4] = make_float2(k_regs[8], k_regs[9]);
        k2[5] = make_float2(k_regs[10], k_regs[11]);
        k2[6] = make_float2(k_regs[12], k_regs[13]);
        k2[7] = make_float2(k_regs[14], k_regs[15]);
        k2[8] = make_float2(k_regs[16], k_regs[17]);
        k2[9] = make_float2(k_regs[18], k_regs[19]);
        k2[10] = make_float2(k_regs[20], k_regs[21]);
        k2[11] = make_float2(k_regs[22], k_regs[23]);
        k2[12] = make_float2(k_regs[24], k_regs[25]);
        k2[13] = make_float2(k_regs[26], k_regs[27]);
        k2[14] = make_float2(k_regs[28], k_regs[29]);
        k2[15] = make_float2(k_regs[30], k_regs[31]);
        float2 accsq_0_0 = make_float2(0.0f, 0.0f);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[1], k2[1], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[3], k2[3], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[5], k2[5], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[7], k2[7], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[8], k2[8], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[9], k2[9], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[10], k2[10], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[11], k2[11], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[12], k2[12], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[13], k2[13], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[14], k2[14], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[15], k2[15], accsq_0_0);
        float dotsq_0 = accsq_0_0.x + accsq_0_0.y;
        float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 1);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_67;
        float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 2);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_68;
#endif
        o4[0] = dotsq_0;
#if __CUDA_ARCH__ == 900
        float accsq_1_0_x = 0.0f;
        float accsq_1_0_y = 0.0f;
        float accsq_1_1_x = 0.0f;
        float accsq_1_1_y = 0.0f;
        float _fma_808;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_808)
            : "f"(S[32]), "f"(k_regs[0]), "f"(accsq_1_0_x));
        float _fma_809;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_809)
            : "f"(S[33]), "f"(k_regs[1]), "f"(accsq_1_0_y));
        accsq_1_0_x = _fma_808;
        accsq_1_0_y = _fma_809;
        float _fma_810;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_810)
            : "f"(S[34]), "f"(k_regs[2]), "f"(accsq_1_1_x));
        float _fma_811;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_811)
            : "f"(S[35]), "f"(k_regs[3]), "f"(accsq_1_1_y));
        accsq_1_1_x = _fma_810;
        accsq_1_1_y = _fma_811;
        float _fma_812;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_812)
            : "f"(S[36]), "f"(k_regs[4]), "f"(accsq_1_0_x));
        float _fma_813;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_813)
            : "f"(S[37]), "f"(k_regs[5]), "f"(accsq_1_0_y));
        accsq_1_0_x = _fma_812;
        accsq_1_0_y = _fma_813;
        float _fma_814;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_814)
            : "f"(S[38]), "f"(k_regs[6]), "f"(accsq_1_1_x));
        float _fma_815;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_815)
            : "f"(S[39]), "f"(k_regs[7]), "f"(accsq_1_1_y));
        accsq_1_1_x = _fma_814;
        accsq_1_1_y = _fma_815;
        float _fma_816;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_816)
            : "f"(S[40]), "f"(k_regs[8]), "f"(accsq_1_0_x));
        float _fma_817;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_817)
            : "f"(S[41]), "f"(k_regs[9]), "f"(accsq_1_0_y));
        accsq_1_0_x = _fma_816;
        accsq_1_0_y = _fma_817;
        float _fma_818;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_818)
            : "f"(S[42]), "f"(k_regs[10]), "f"(accsq_1_1_x));
        float _fma_819;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_819)
            : "f"(S[43]), "f"(k_regs[11]), "f"(accsq_1_1_y));
        accsq_1_1_x = _fma_818;
        accsq_1_1_y = _fma_819;
        float _fma_820;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_820)
            : "f"(S[44]), "f"(k_regs[12]), "f"(accsq_1_0_x));
        float _fma_821;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_821)
            : "f"(S[45]), "f"(k_regs[13]), "f"(accsq_1_0_y));
        accsq_1_0_x = _fma_820;
        accsq_1_0_y = _fma_821;
        float _fma_822;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_822)
            : "f"(S[46]), "f"(k_regs[14]), "f"(accsq_1_1_x));
        float _fma_823;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_823)
            : "f"(S[47]), "f"(k_regs[15]), "f"(accsq_1_1_y));
        accsq_1_1_x = _fma_822;
        accsq_1_1_y = _fma_823;
        float _fma_824;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_824)
            : "f"(S[48]), "f"(k_regs[16]), "f"(accsq_1_0_x));
        float _fma_825;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_825)
            : "f"(S[49]), "f"(k_regs[17]), "f"(accsq_1_0_y));
        accsq_1_0_x = _fma_824;
        accsq_1_0_y = _fma_825;
        float _fma_826;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_826)
            : "f"(S[50]), "f"(k_regs[18]), "f"(accsq_1_1_x));
        float _fma_827;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_827)
            : "f"(S[51]), "f"(k_regs[19]), "f"(accsq_1_1_y));
        accsq_1_1_x = _fma_826;
        accsq_1_1_y = _fma_827;
        float _fma_828;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_828)
            : "f"(S[52]), "f"(k_regs[20]), "f"(accsq_1_0_x));
        float _fma_829;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_829)
            : "f"(S[53]), "f"(k_regs[21]), "f"(accsq_1_0_y));
        accsq_1_0_x = _fma_828;
        accsq_1_0_y = _fma_829;
        float _fma_830;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_830)
            : "f"(S[54]), "f"(k_regs[22]), "f"(accsq_1_1_x));
        float _fma_831;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_831)
            : "f"(S[55]), "f"(k_regs[23]), "f"(accsq_1_1_y));
        accsq_1_1_x = _fma_830;
        accsq_1_1_y = _fma_831;
        float _fma_832;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_832)
            : "f"(S[56]), "f"(k_regs[24]), "f"(accsq_1_0_x));
        float _fma_833;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_833)
            : "f"(S[57]), "f"(k_regs[25]), "f"(accsq_1_0_y));
        accsq_1_0_x = _fma_832;
        accsq_1_0_y = _fma_833;
        float _fma_834;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_834)
            : "f"(S[58]), "f"(k_regs[26]), "f"(accsq_1_1_x));
        float _fma_835;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_835)
            : "f"(S[59]), "f"(k_regs[27]), "f"(accsq_1_1_y));
        accsq_1_1_x = _fma_834;
        accsq_1_1_y = _fma_835;
        float _fma_836;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_836)
            : "f"(S[60]), "f"(k_regs[28]), "f"(accsq_1_0_x));
        float _fma_837;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_837)
            : "f"(S[61]), "f"(k_regs[29]), "f"(accsq_1_0_y));
        accsq_1_0_x = _fma_836;
        accsq_1_0_y = _fma_837;
        float _fma_838;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_838)
            : "f"(S[62]), "f"(k_regs[30]), "f"(accsq_1_1_x));
        float _fma_839;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_839)
            : "f"(S[63]), "f"(k_regs[31]), "f"(accsq_1_1_y));
        accsq_1_1_x = _fma_838;
        accsq_1_1_y = _fma_839;
        float _add_95;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_95) : "f"(accsq_1_0_x), "f"(accsq_1_1_x));
        float _add_96;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(accsq_1_0_y), "f"(accsq_1_1_y));
        float dotsq_1 = _add_95 + _add_96;
        float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 1);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_97;
        float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 2);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_98;
#else
        float2 accsq_1_0 = make_float2(0.0f, 0.0f);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[24], k2[8], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[25], k2[9], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[26], k2[10], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[27], k2[11], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[28], k2[12], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[29], k2[13], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[30], k2[14], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[31], k2[15], accsq_1_0);
        float dotsq_1 = accsq_1_0.x + accsq_1_0.y;
        float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 1);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_69;
        float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 2);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_70;
#endif
        o4[1] = dotsq_1;
#if __CUDA_ARCH__ == 900
        float accsq_2_0_x = 0.0f;
        float accsq_2_0_y = 0.0f;
        float accsq_2_1_x = 0.0f;
        float accsq_2_1_y = 0.0f;
        float _fma_840;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_840)
            : "f"(S[64]), "f"(k_regs[0]), "f"(accsq_2_0_x));
        float _fma_841;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_841)
            : "f"(S[65]), "f"(k_regs[1]), "f"(accsq_2_0_y));
        accsq_2_0_x = _fma_840;
        accsq_2_0_y = _fma_841;
        float _fma_842;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_842)
            : "f"(S[66]), "f"(k_regs[2]), "f"(accsq_2_1_x));
        float _fma_843;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_843)
            : "f"(S[67]), "f"(k_regs[3]), "f"(accsq_2_1_y));
        accsq_2_1_x = _fma_842;
        accsq_2_1_y = _fma_843;
        float _fma_844;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_844)
            : "f"(S[68]), "f"(k_regs[4]), "f"(accsq_2_0_x));
        float _fma_845;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_845)
            : "f"(S[69]), "f"(k_regs[5]), "f"(accsq_2_0_y));
        accsq_2_0_x = _fma_844;
        accsq_2_0_y = _fma_845;
        float _fma_846;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_846)
            : "f"(S[70]), "f"(k_regs[6]), "f"(accsq_2_1_x));
        float _fma_847;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_847)
            : "f"(S[71]), "f"(k_regs[7]), "f"(accsq_2_1_y));
        accsq_2_1_x = _fma_846;
        accsq_2_1_y = _fma_847;
        float _fma_848;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_848)
            : "f"(S[72]), "f"(k_regs[8]), "f"(accsq_2_0_x));
        float _fma_849;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_849)
            : "f"(S[73]), "f"(k_regs[9]), "f"(accsq_2_0_y));
        accsq_2_0_x = _fma_848;
        accsq_2_0_y = _fma_849;
        float _fma_850;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_850)
            : "f"(S[74]), "f"(k_regs[10]), "f"(accsq_2_1_x));
        float _fma_851;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_851)
            : "f"(S[75]), "f"(k_regs[11]), "f"(accsq_2_1_y));
        accsq_2_1_x = _fma_850;
        accsq_2_1_y = _fma_851;
        float _fma_852;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_852)
            : "f"(S[76]), "f"(k_regs[12]), "f"(accsq_2_0_x));
        float _fma_853;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_853)
            : "f"(S[77]), "f"(k_regs[13]), "f"(accsq_2_0_y));
        accsq_2_0_x = _fma_852;
        accsq_2_0_y = _fma_853;
        float _fma_854;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_854)
            : "f"(S[78]), "f"(k_regs[14]), "f"(accsq_2_1_x));
        float _fma_855;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_855)
            : "f"(S[79]), "f"(k_regs[15]), "f"(accsq_2_1_y));
        accsq_2_1_x = _fma_854;
        accsq_2_1_y = _fma_855;
        float _fma_856;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_856)
            : "f"(S[80]), "f"(k_regs[16]), "f"(accsq_2_0_x));
        float _fma_857;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_857)
            : "f"(S[81]), "f"(k_regs[17]), "f"(accsq_2_0_y));
        accsq_2_0_x = _fma_856;
        accsq_2_0_y = _fma_857;
        float _fma_858;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_858)
            : "f"(S[82]), "f"(k_regs[18]), "f"(accsq_2_1_x));
        float _fma_859;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_859)
            : "f"(S[83]), "f"(k_regs[19]), "f"(accsq_2_1_y));
        accsq_2_1_x = _fma_858;
        accsq_2_1_y = _fma_859;
        float _fma_860;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_860)
            : "f"(S[84]), "f"(k_regs[20]), "f"(accsq_2_0_x));
        float _fma_861;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_861)
            : "f"(S[85]), "f"(k_regs[21]), "f"(accsq_2_0_y));
        accsq_2_0_x = _fma_860;
        accsq_2_0_y = _fma_861;
        float _fma_862;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_862)
            : "f"(S[86]), "f"(k_regs[22]), "f"(accsq_2_1_x));
        float _fma_863;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_863)
            : "f"(S[87]), "f"(k_regs[23]), "f"(accsq_2_1_y));
        accsq_2_1_x = _fma_862;
        accsq_2_1_y = _fma_863;
        float _fma_864;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_864)
            : "f"(S[88]), "f"(k_regs[24]), "f"(accsq_2_0_x));
        float _fma_865;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_865)
            : "f"(S[89]), "f"(k_regs[25]), "f"(accsq_2_0_y));
        accsq_2_0_x = _fma_864;
        accsq_2_0_y = _fma_865;
        float _fma_866;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_866)
            : "f"(S[90]), "f"(k_regs[26]), "f"(accsq_2_1_x));
        float _fma_867;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_867)
            : "f"(S[91]), "f"(k_regs[27]), "f"(accsq_2_1_y));
        accsq_2_1_x = _fma_866;
        accsq_2_1_y = _fma_867;
        float _fma_868;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_868)
            : "f"(S[92]), "f"(k_regs[28]), "f"(accsq_2_0_x));
        float _fma_869;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_869)
            : "f"(S[93]), "f"(k_regs[29]), "f"(accsq_2_0_y));
        accsq_2_0_x = _fma_868;
        accsq_2_0_y = _fma_869;
        float _fma_870;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_870)
            : "f"(S[94]), "f"(k_regs[30]), "f"(accsq_2_1_x));
        float _fma_871;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_871)
            : "f"(S[95]), "f"(k_regs[31]), "f"(accsq_2_1_y));
        accsq_2_1_x = _fma_870;
        accsq_2_1_y = _fma_871;
        float _add_99;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_99) : "f"(accsq_2_0_x), "f"(accsq_2_1_x));
        float _add_100;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(accsq_2_0_y), "f"(accsq_2_1_y));
        float dotsq_2 = _add_99 + _add_100;
        float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, dotsq_2, 1);
        dotsq_2 = dotsq_2 + _shfl_xor_sync_101;
        float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, dotsq_2, 2);
        dotsq_2 = dotsq_2 + _shfl_xor_sync_102;
#else
        float2 accsq_2_0 = make_float2(0.0f, 0.0f);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[32], k2[0], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[33], k2[1], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[34], k2[2], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[35], k2[3], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[36], k2[4], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[37], k2[5], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[38], k2[6], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[39], k2[7], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[40], k2[8], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[41], k2[9], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[42], k2[10], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[43], k2[11], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[44], k2[12], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[45], k2[13], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[46], k2[14], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[47], k2[15], accsq_2_0);
        float dotsq_2 = accsq_2_0.x + accsq_2_0.y;
        float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, dotsq_2, 1);
        dotsq_2 = dotsq_2 + _shfl_xor_sync_71;
        float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, dotsq_2, 2);
        dotsq_2 = dotsq_2 + _shfl_xor_sync_72;
#endif
        o4[2] = dotsq_2;
#if __CUDA_ARCH__ == 900
        float accsq_3_0_x = 0.0f;
        float accsq_3_0_y = 0.0f;
        float accsq_3_1_x = 0.0f;
        float accsq_3_1_y = 0.0f;
        float _fma_872;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_872)
            : "f"(S[96]), "f"(k_regs[0]), "f"(accsq_3_0_x));
        float _fma_873;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_873)
            : "f"(S[97]), "f"(k_regs[1]), "f"(accsq_3_0_y));
        accsq_3_0_x = _fma_872;
        accsq_3_0_y = _fma_873;
        float _fma_874;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_874)
            : "f"(S[98]), "f"(k_regs[2]), "f"(accsq_3_1_x));
        float _fma_875;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_875)
            : "f"(S[99]), "f"(k_regs[3]), "f"(accsq_3_1_y));
        accsq_3_1_x = _fma_874;
        accsq_3_1_y = _fma_875;
        float _fma_876;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_876)
            : "f"(S[100]), "f"(k_regs[4]), "f"(accsq_3_0_x));
        float _fma_877;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_877)
            : "f"(S[101]), "f"(k_regs[5]), "f"(accsq_3_0_y));
        accsq_3_0_x = _fma_876;
        accsq_3_0_y = _fma_877;
        float _fma_878;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_878)
            : "f"(S[102]), "f"(k_regs[6]), "f"(accsq_3_1_x));
        float _fma_879;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_879)
            : "f"(S[103]), "f"(k_regs[7]), "f"(accsq_3_1_y));
        accsq_3_1_x = _fma_878;
        accsq_3_1_y = _fma_879;
        float _fma_880;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_880)
            : "f"(S[104]), "f"(k_regs[8]), "f"(accsq_3_0_x));
        float _fma_881;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_881)
            : "f"(S[105]), "f"(k_regs[9]), "f"(accsq_3_0_y));
        accsq_3_0_x = _fma_880;
        accsq_3_0_y = _fma_881;
        float _fma_882;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_882)
            : "f"(S[106]), "f"(k_regs[10]), "f"(accsq_3_1_x));
        float _fma_883;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_883)
            : "f"(S[107]), "f"(k_regs[11]), "f"(accsq_3_1_y));
        accsq_3_1_x = _fma_882;
        accsq_3_1_y = _fma_883;
        float _fma_884;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_884)
            : "f"(S[108]), "f"(k_regs[12]), "f"(accsq_3_0_x));
        float _fma_885;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_885)
            : "f"(S[109]), "f"(k_regs[13]), "f"(accsq_3_0_y));
        accsq_3_0_x = _fma_884;
        accsq_3_0_y = _fma_885;
        float _fma_886;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_886)
            : "f"(S[110]), "f"(k_regs[14]), "f"(accsq_3_1_x));
        float _fma_887;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_887)
            : "f"(S[111]), "f"(k_regs[15]), "f"(accsq_3_1_y));
        accsq_3_1_x = _fma_886;
        accsq_3_1_y = _fma_887;
        float _fma_888;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_888)
            : "f"(S[112]), "f"(k_regs[16]), "f"(accsq_3_0_x));
        float _fma_889;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_889)
            : "f"(S[113]), "f"(k_regs[17]), "f"(accsq_3_0_y));
        accsq_3_0_x = _fma_888;
        accsq_3_0_y = _fma_889;
        float _fma_890;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_890)
            : "f"(S[114]), "f"(k_regs[18]), "f"(accsq_3_1_x));
        float _fma_891;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_891)
            : "f"(S[115]), "f"(k_regs[19]), "f"(accsq_3_1_y));
        accsq_3_1_x = _fma_890;
        accsq_3_1_y = _fma_891;
        float _fma_892;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_892)
            : "f"(S[116]), "f"(k_regs[20]), "f"(accsq_3_0_x));
        float _fma_893;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_893)
            : "f"(S[117]), "f"(k_regs[21]), "f"(accsq_3_0_y));
        accsq_3_0_x = _fma_892;
        accsq_3_0_y = _fma_893;
        float _fma_894;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_894)
            : "f"(S[118]), "f"(k_regs[22]), "f"(accsq_3_1_x));
        float _fma_895;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_895)
            : "f"(S[119]), "f"(k_regs[23]), "f"(accsq_3_1_y));
        accsq_3_1_x = _fma_894;
        accsq_3_1_y = _fma_895;
        float _fma_896;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_896)
            : "f"(S[120]), "f"(k_regs[24]), "f"(accsq_3_0_x));
        float _fma_897;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_897)
            : "f"(S[121]), "f"(k_regs[25]), "f"(accsq_3_0_y));
        accsq_3_0_x = _fma_896;
        accsq_3_0_y = _fma_897;
        float _fma_898;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_898)
            : "f"(S[122]), "f"(k_regs[26]), "f"(accsq_3_1_x));
        float _fma_899;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_899)
            : "f"(S[123]), "f"(k_regs[27]), "f"(accsq_3_1_y));
        accsq_3_1_x = _fma_898;
        accsq_3_1_y = _fma_899;
        float _fma_900;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_900)
            : "f"(S[124]), "f"(k_regs[28]), "f"(accsq_3_0_x));
        float _fma_901;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_901)
            : "f"(S[125]), "f"(k_regs[29]), "f"(accsq_3_0_y));
        accsq_3_0_x = _fma_900;
        accsq_3_0_y = _fma_901;
        float _fma_902;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_902)
            : "f"(S[126]), "f"(k_regs[30]), "f"(accsq_3_1_x));
        float _fma_903;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_903)
            : "f"(S[127]), "f"(k_regs[31]), "f"(accsq_3_1_y));
        accsq_3_1_x = _fma_902;
        accsq_3_1_y = _fma_903;
        float _add_103;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(accsq_3_0_x), "f"(accsq_3_1_x));
        float _add_104;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(accsq_3_0_y), "f"(accsq_3_1_y));
        float dotsq_3 = _add_103 + _add_104;
        float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, dotsq_3, 1);
        dotsq_3 = dotsq_3 + _shfl_xor_sync_105;
        float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, dotsq_3, 2);
        dotsq_3 = dotsq_3 + _shfl_xor_sync_106;
#else
        float2 accsq_3_0 = make_float2(0.0f, 0.0f);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[48], k2[0], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[49], k2[1], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[50], k2[2], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[51], k2[3], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[52], k2[4], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[53], k2[5], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[54], k2[6], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[55], k2[7], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[56], k2[8], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[57], k2[9], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[58], k2[10], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[59], k2[11], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[60], k2[12], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[61], k2[13], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[62], k2[14], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[63], k2[15], accsq_3_0);
        float dotsq_3 = accsq_3_0.x + accsq_3_0.y;
        float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, dotsq_3, 1);
        dotsq_3 = dotsq_3 + _shfl_xor_sync_73;
        float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, dotsq_3, 2);
        dotsq_3 = dotsq_3 + _shfl_xor_sync_74;
#endif
        o4[3] = dotsq_3;
        if (u == 0) {
          long long out_index = (long long)n * out_stride_p0 + (long long)t_1 * out_stride_p1 +
                                (long long)h * out_stride_p2 + (long long)row0;
          {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(o4[0 + 0], o4[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(o4[0 + 2], o4[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[out_index]) = _pk2;
          }
        }
        int ssm_slot = sSsm[t_1];
        int snap_valid =
            ((use_ssm_state_indices != 0 && ssm_slot >= 0 && state_pool_size > (long long)ssm_slot)
                 ? 1
                 : 0);
        int safe_slot = ((snap_valid != 0) ? ssm_slot : 0);
        long long snap_base = (long long)safe_slot * state_stride_p0 + head_rows_off;
#if __CUDA_ARCH__ == 900
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base])),
                     "f"(S[0]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 1])),
                     "f"(S[1]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2])),
                     "f"(S[2]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 + 1])),
                     "f"(S[3]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 16])),
                     "f"(S[4]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 16 + 1])),
                     "f"(S[5]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 18])),
                     "f"(S[6]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 18 + 1])),
                     "f"(S[7]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 32])),
                     "f"(S[8]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 32 + 1])),
                     "f"(S[9]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 34])),
                     "f"(S[10]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 34 + 1])),
                     "f"(S[11]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 48])),
                     "f"(S[12]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 48 + 1])),
                     "f"(S[13]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 50])),
                     "f"(S[14]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 50 + 1])),
                     "f"(S[15]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 64])),
                     "f"(S[16]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 64 + 1])),
                     "f"(S[17]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 66])),
                     "f"(S[18]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 66 + 1])),
                     "f"(S[19]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 80])),
                     "f"(S[20]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 80 + 1])),
                     "f"(S[21]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 82])),
                     "f"(S[22]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 82 + 1])),
                     "f"(S[23]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 96])),
                     "f"(S[24]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 96 + 1])),
                     "f"(S[25]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 98])),
                     "f"(S[26]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 98 + 1])),
                     "f"(S[27]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 112])),
                     "f"(S[28]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 112 + 1])),
                     "f"(S[29]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 114])),
                     "f"(S[30]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 114 + 1])),
                     "f"(S[31]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2])),
                     "f"(S[32]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 1])),
                     "f"(S[33]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 2])),
                     "f"(S[34]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 2 + 1])),
                     "f"(S[35]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 16])),
                     "f"(S[36]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 16 + 1])),
                     "f"(S[37]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 18])),
                     "f"(S[38]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 18 + 1])),
                     "f"(S[39]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 32])),
                     "f"(S[40]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 32 + 1])),
                     "f"(S[41]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 34])),
                     "f"(S[42]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 34 + 1])),
                     "f"(S[43]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 48])),
                     "f"(S[44]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 48 + 1])),
                     "f"(S[45]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 50])),
                     "f"(S[46]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 50 + 1])),
                     "f"(S[47]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 64])),
                     "f"(S[48]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 64 + 1])),
                     "f"(S[49]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 66])),
                     "f"(S[50]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 66 + 1])),
                     "f"(S[51]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 80])),
                     "f"(S[52]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 80 + 1])),
                     "f"(S[53]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 82])),
                     "f"(S[54]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 82 + 1])),
                     "f"(S[55]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 96])),
                     "f"(S[56]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 96 + 1])),
                     "f"(S[57]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 98])),
                     "f"(S[58]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 98 + 1])),
                     "f"(S[59]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 112])),
                     "f"(S[60]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 112 + 1])),
                     "f"(S[61]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 114])),
                     "f"(S[62]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 114 + 1])),
                     "f"(S[63]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2])),
                     "f"(S[64]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 1])),
                     "f"(S[65]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 2])),
                     "f"(S[66]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 2 + 1])),
            "f"(S[67]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 16])),
                     "f"(S[68]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 16 + 1])),
            "f"(S[69]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 18])),
                     "f"(S[70]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 18 + 1])),
            "f"(S[71]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 32])),
                     "f"(S[72]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 32 + 1])),
            "f"(S[73]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 34])),
                     "f"(S[74]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 34 + 1])),
            "f"(S[75]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 48])),
                     "f"(S[76]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 48 + 1])),
            "f"(S[77]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 50])),
                     "f"(S[78]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 50 + 1])),
            "f"(S[79]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 64])),
                     "f"(S[80]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 64 + 1])),
            "f"(S[81]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 66])),
                     "f"(S[82]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 66 + 1])),
            "f"(S[83]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 80])),
                     "f"(S[84]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 80 + 1])),
            "f"(S[85]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 82])),
                     "f"(S[86]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 82 + 1])),
            "f"(S[87]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 96])),
                     "f"(S[88]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 96 + 1])),
            "f"(S[89]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 98])),
                     "f"(S[90]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 98 + 1])),
            "f"(S[91]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 112])),
                     "f"(S[92]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 112 + 1])),
            "f"(S[93]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 114])),
                     "f"(S[94]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 114 + 1])),
            "f"(S[95]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2])),
                     "f"(S[96]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 1])),
                     "f"(S[97]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 2])),
                     "f"(S[98]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 2 + 1])),
            "f"(S[99]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 16])),
                     "f"(S[100]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 16 + 1])),
            "f"(S[101]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 18])),
                     "f"(S[102]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 18 + 1])),
            "f"(S[103]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 32])),
                     "f"(S[104]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 32 + 1])),
            "f"(S[105]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 34])),
                     "f"(S[106]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 34 + 1])),
            "f"(S[107]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 48])),
                     "f"(S[108]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 48 + 1])),
            "f"(S[109]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 50])),
                     "f"(S[110]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 50 + 1])),
            "f"(S[111]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 64])),
                     "f"(S[112]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 64 + 1])),
            "f"(S[113]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 66])),
                     "f"(S[114]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 66 + 1])),
            "f"(S[115]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 80])),
                     "f"(S[116]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 80 + 1])),
            "f"(S[117]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 82])),
                     "f"(S[118]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 82 + 1])),
            "f"(S[119]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 96])),
                     "f"(S[120]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 96 + 1])),
            "f"(S[121]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 98])),
                     "f"(S[122]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 98 + 1])),
            "f"(S[123]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 112])),
                     "f"(S[124]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 112 + 1])),
            "f"(S[125]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 114])),
                     "f"(S[126]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 114 + 1])),
            "f"(S[127]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
#else
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base])),
                     "f"(S2[0].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 1])),
                     "f"(S2[0].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2])),
                     "f"(S2[1].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 + 1])),
                     "f"(S2[1].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 16])),
                     "f"(S2[2].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 16 + 1])),
                     "f"(S2[2].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 18])),
                     "f"(S2[3].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 18 + 1])),
                     "f"(S2[3].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 32])),
                     "f"(S2[4].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 32 + 1])),
                     "f"(S2[4].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 34])),
                     "f"(S2[5].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 34 + 1])),
                     "f"(S2[5].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 48])),
                     "f"(S2[6].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 48 + 1])),
                     "f"(S2[6].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 50])),
                     "f"(S2[7].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 50 + 1])),
                     "f"(S2[7].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 64])),
                     "f"(S2[8].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 64 + 1])),
                     "f"(S2[8].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 66])),
                     "f"(S2[9].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 66 + 1])),
                     "f"(S2[9].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 80])),
                     "f"(S2[10].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 80 + 1])),
                     "f"(S2[10].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 82])),
                     "f"(S2[11].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 82 + 1])),
                     "f"(S2[11].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 96])),
                     "f"(S2[12].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 96 + 1])),
                     "f"(S2[12].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 98])),
                     "f"(S2[13].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 98 + 1])),
                     "f"(S2[13].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 112])),
                     "f"(S2[14].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 112 + 1])),
                     "f"(S2[14].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 114])),
                     "f"(S2[15].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 114 + 1])),
                     "f"(S2[15].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2])),
                     "f"(S2[16].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 1])),
                     "f"(S2[16].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 2])),
                     "f"(S2[17].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 2 + 1])),
                     "f"(S2[17].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 16])),
                     "f"(S2[18].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 16 + 1])),
                     "f"(S2[18].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 18])),
                     "f"(S2[19].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 18 + 1])),
                     "f"(S2[19].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 32])),
                     "f"(S2[20].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 32 + 1])),
                     "f"(S2[20].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 34])),
                     "f"(S2[21].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 34 + 1])),
                     "f"(S2[21].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 48])),
                     "f"(S2[22].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 48 + 1])),
                     "f"(S2[22].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 50])),
                     "f"(S2[23].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 50 + 1])),
                     "f"(S2[23].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 64])),
                     "f"(S2[24].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 64 + 1])),
                     "f"(S2[24].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 66])),
                     "f"(S2[25].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 66 + 1])),
                     "f"(S2[25].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 80])),
                     "f"(S2[26].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 80 + 1])),
                     "f"(S2[26].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 82])),
                     "f"(S2[27].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 82 + 1])),
                     "f"(S2[27].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 96])),
                     "f"(S2[28].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 96 + 1])),
                     "f"(S2[28].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 98])),
                     "f"(S2[29].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 98 + 1])),
                     "f"(S2[29].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 112])),
                     "f"(S2[30].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 112 + 1])),
                     "f"(S2[30].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 114])),
                     "f"(S2[31].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 114 + 1])),
                     "f"(S2[31].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2])),
                     "f"(S2[32].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 1])),
                     "f"(S2[32].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 2])),
                     "f"(S2[33].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 2 + 1])),
            "f"(S2[33].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 16])),
                     "f"(S2[34].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 16 + 1])),
            "f"(S2[34].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 18])),
                     "f"(S2[35].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 18 + 1])),
            "f"(S2[35].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 32])),
                     "f"(S2[36].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 32 + 1])),
            "f"(S2[36].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 34])),
                     "f"(S2[37].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 34 + 1])),
            "f"(S2[37].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 48])),
                     "f"(S2[38].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 48 + 1])),
            "f"(S2[38].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 50])),
                     "f"(S2[39].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 50 + 1])),
            "f"(S2[39].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 64])),
                     "f"(S2[40].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 64 + 1])),
            "f"(S2[40].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 66])),
                     "f"(S2[41].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 66 + 1])),
            "f"(S2[41].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 80])),
                     "f"(S2[42].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 80 + 1])),
            "f"(S2[42].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 82])),
                     "f"(S2[43].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 82 + 1])),
            "f"(S2[43].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 96])),
                     "f"(S2[44].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 96 + 1])),
            "f"(S2[44].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 98])),
                     "f"(S2[45].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 98 + 1])),
            "f"(S2[45].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 112])),
                     "f"(S2[46].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 112 + 1])),
            "f"(S2[46].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 114])),
                     "f"(S2[47].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 114 + 1])),
            "f"(S2[47].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2])),
                     "f"(S2[48].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 1])),
                     "f"(S2[48].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 2])),
                     "f"(S2[49].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 2 + 1])),
            "f"(S2[49].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 16])),
                     "f"(S2[50].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 16 + 1])),
            "f"(S2[50].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 18])),
                     "f"(S2[51].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 18 + 1])),
            "f"(S2[51].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 32])),
                     "f"(S2[52].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 32 + 1])),
            "f"(S2[52].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 34])),
                     "f"(S2[53].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 34 + 1])),
            "f"(S2[53].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 48])),
                     "f"(S2[54].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 48 + 1])),
            "f"(S2[54].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 50])),
                     "f"(S2[55].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 50 + 1])),
            "f"(S2[55].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 64])),
                     "f"(S2[56].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 64 + 1])),
            "f"(S2[56].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 66])),
                     "f"(S2[57].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 66 + 1])),
            "f"(S2[57].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 80])),
                     "f"(S2[58].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 80 + 1])),
            "f"(S2[58].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 82])),
                     "f"(S2[59].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 82 + 1])),
            "f"(S2[59].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 96])),
                     "f"(S2[60].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 96 + 1])),
            "f"(S2[60].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 98])),
                     "f"(S2[61].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 98 + 1])),
            "f"(S2[61].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 112])),
                     "f"(S2[62].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 112 + 1])),
            "f"(S2[62].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 114])),
                     "f"(S2[63].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 114 + 1])),
            "f"(S2[63].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
#endif
      }
    }
    if (use_ssm_state_indices == 0) {
#pragma unroll 1
      for (int t_2 = 0; t_2 < num_tokens; t_2++) {
        float alphap = sAlpha[t_2];
#if __CUDA_ARCH__ == 900
        float alpha2p_x = alphap;
        float alpha2p_y = alphap;
        float _mul_136;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_136) : "f"(S[0]), "f"(alpha2p_x));
        float _mul_137;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_137) : "f"(S[1]), "f"(alpha2p_y));
        S[0] = _mul_136;
        S[1] = _mul_137;
        float _mul_138;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_138) : "f"(S[2]), "f"(alpha2p_x));
        float _mul_139;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_139) : "f"(S[3]), "f"(alpha2p_y));
        S[2] = _mul_138;
        S[3] = _mul_139;
        float _mul_140;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_140) : "f"(S[4]), "f"(alpha2p_x));
        float _mul_141;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_141) : "f"(S[5]), "f"(alpha2p_y));
        S[4] = _mul_140;
        S[5] = _mul_141;
        float _mul_142;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_142) : "f"(S[6]), "f"(alpha2p_x));
        float _mul_143;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_143) : "f"(S[7]), "f"(alpha2p_y));
        S[6] = _mul_142;
        S[7] = _mul_143;
        float _mul_144;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_144) : "f"(S[8]), "f"(alpha2p_x));
        float _mul_145;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_145) : "f"(S[9]), "f"(alpha2p_y));
        S[8] = _mul_144;
        S[9] = _mul_145;
        float _mul_146;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_146) : "f"(S[10]), "f"(alpha2p_x));
        float _mul_147;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_147) : "f"(S[11]), "f"(alpha2p_y));
        S[10] = _mul_146;
        S[11] = _mul_147;
        float _mul_148;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_148) : "f"(S[12]), "f"(alpha2p_x));
        float _mul_149;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_149) : "f"(S[13]), "f"(alpha2p_y));
        S[12] = _mul_148;
        S[13] = _mul_149;
        float _mul_150;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_150) : "f"(S[14]), "f"(alpha2p_x));
        float _mul_151;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_151) : "f"(S[15]), "f"(alpha2p_y));
        S[14] = _mul_150;
        S[15] = _mul_151;
        float _mul_152;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_152) : "f"(S[16]), "f"(alpha2p_x));
        float _mul_153;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_153) : "f"(S[17]), "f"(alpha2p_y));
        S[16] = _mul_152;
        S[17] = _mul_153;
        float _mul_154;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_154) : "f"(S[18]), "f"(alpha2p_x));
        float _mul_155;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_155) : "f"(S[19]), "f"(alpha2p_y));
        S[18] = _mul_154;
        S[19] = _mul_155;
        float _mul_156;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_156) : "f"(S[20]), "f"(alpha2p_x));
        float _mul_157;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_157) : "f"(S[21]), "f"(alpha2p_y));
        S[20] = _mul_156;
        S[21] = _mul_157;
        float _mul_158;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_158) : "f"(S[22]), "f"(alpha2p_x));
        float _mul_159;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_159) : "f"(S[23]), "f"(alpha2p_y));
        S[22] = _mul_158;
        S[23] = _mul_159;
        float _mul_160;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_160) : "f"(S[24]), "f"(alpha2p_x));
        float _mul_161;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_161) : "f"(S[25]), "f"(alpha2p_y));
        S[24] = _mul_160;
        S[25] = _mul_161;
        float _mul_162;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_162) : "f"(S[26]), "f"(alpha2p_x));
        float _mul_163;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_163) : "f"(S[27]), "f"(alpha2p_y));
        S[26] = _mul_162;
        S[27] = _mul_163;
        float _mul_164;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_164) : "f"(S[28]), "f"(alpha2p_x));
        float _mul_165;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_165) : "f"(S[29]), "f"(alpha2p_y));
        S[28] = _mul_164;
        S[29] = _mul_165;
        float _mul_166;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_166) : "f"(S[30]), "f"(alpha2p_x));
        float _mul_167;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_167) : "f"(S[31]), "f"(alpha2p_y));
        S[30] = _mul_166;
        S[31] = _mul_167;
        float _mul_168;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_168) : "f"(S[32]), "f"(alpha2p_x));
        float _mul_169;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_169) : "f"(S[33]), "f"(alpha2p_y));
        S[32] = _mul_168;
        S[33] = _mul_169;
        float _mul_170;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_170) : "f"(S[34]), "f"(alpha2p_x));
        float _mul_171;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_171) : "f"(S[35]), "f"(alpha2p_y));
        S[34] = _mul_170;
        S[35] = _mul_171;
        float _mul_172;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_172) : "f"(S[36]), "f"(alpha2p_x));
        float _mul_173;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_173) : "f"(S[37]), "f"(alpha2p_y));
        S[36] = _mul_172;
        S[37] = _mul_173;
        float _mul_174;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_174) : "f"(S[38]), "f"(alpha2p_x));
        float _mul_175;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_175) : "f"(S[39]), "f"(alpha2p_y));
        S[38] = _mul_174;
        S[39] = _mul_175;
        float _mul_176;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_176) : "f"(S[40]), "f"(alpha2p_x));
        float _mul_177;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_177) : "f"(S[41]), "f"(alpha2p_y));
        S[40] = _mul_176;
        S[41] = _mul_177;
        float _mul_178;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_178) : "f"(S[42]), "f"(alpha2p_x));
        float _mul_179;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_179) : "f"(S[43]), "f"(alpha2p_y));
        S[42] = _mul_178;
        S[43] = _mul_179;
        float _mul_180;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_180) : "f"(S[44]), "f"(alpha2p_x));
        float _mul_181;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_181) : "f"(S[45]), "f"(alpha2p_y));
        S[44] = _mul_180;
        S[45] = _mul_181;
        float _mul_182;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_182) : "f"(S[46]), "f"(alpha2p_x));
        float _mul_183;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_183) : "f"(S[47]), "f"(alpha2p_y));
        S[46] = _mul_182;
        S[47] = _mul_183;
        float _mul_184;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_184) : "f"(S[48]), "f"(alpha2p_x));
        float _mul_185;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_185) : "f"(S[49]), "f"(alpha2p_y));
        S[48] = _mul_184;
        S[49] = _mul_185;
        float _mul_186;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_186) : "f"(S[50]), "f"(alpha2p_x));
        float _mul_187;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_187) : "f"(S[51]), "f"(alpha2p_y));
        S[50] = _mul_186;
        S[51] = _mul_187;
        float _mul_188;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_188) : "f"(S[52]), "f"(alpha2p_x));
        float _mul_189;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_189) : "f"(S[53]), "f"(alpha2p_y));
        S[52] = _mul_188;
        S[53] = _mul_189;
        float _mul_190;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_190) : "f"(S[54]), "f"(alpha2p_x));
        float _mul_191;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_191) : "f"(S[55]), "f"(alpha2p_y));
        S[54] = _mul_190;
        S[55] = _mul_191;
        float _mul_192;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_192) : "f"(S[56]), "f"(alpha2p_x));
        float _mul_193;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_193) : "f"(S[57]), "f"(alpha2p_y));
        S[56] = _mul_192;
        S[57] = _mul_193;
        float _mul_194;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_194) : "f"(S[58]), "f"(alpha2p_x));
        float _mul_195;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_195) : "f"(S[59]), "f"(alpha2p_y));
        S[58] = _mul_194;
        S[59] = _mul_195;
        float _mul_196;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_196) : "f"(S[60]), "f"(alpha2p_x));
        float _mul_197;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_197) : "f"(S[61]), "f"(alpha2p_y));
        S[60] = _mul_196;
        S[61] = _mul_197;
        float _mul_198;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_198) : "f"(S[62]), "f"(alpha2p_x));
        float _mul_199;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_199) : "f"(S[63]), "f"(alpha2p_y));
        S[62] = _mul_198;
        S[63] = _mul_199;
        float _mul_200;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_200) : "f"(S[64]), "f"(alpha2p_x));
        float _mul_201;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_201) : "f"(S[65]), "f"(alpha2p_y));
        S[64] = _mul_200;
        S[65] = _mul_201;
        float _mul_202;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_202) : "f"(S[66]), "f"(alpha2p_x));
        float _mul_203;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_203) : "f"(S[67]), "f"(alpha2p_y));
        S[66] = _mul_202;
        S[67] = _mul_203;
        float _mul_204;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_204) : "f"(S[68]), "f"(alpha2p_x));
        float _mul_205;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_205) : "f"(S[69]), "f"(alpha2p_y));
        S[68] = _mul_204;
        S[69] = _mul_205;
        float _mul_206;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_206) : "f"(S[70]), "f"(alpha2p_x));
        float _mul_207;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_207) : "f"(S[71]), "f"(alpha2p_y));
        S[70] = _mul_206;
        S[71] = _mul_207;
        float _mul_208;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_208) : "f"(S[72]), "f"(alpha2p_x));
        float _mul_209;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_209) : "f"(S[73]), "f"(alpha2p_y));
        S[72] = _mul_208;
        S[73] = _mul_209;
        float _mul_210;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_210) : "f"(S[74]), "f"(alpha2p_x));
        float _mul_211;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_211) : "f"(S[75]), "f"(alpha2p_y));
        S[74] = _mul_210;
        S[75] = _mul_211;
        float _mul_212;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_212) : "f"(S[76]), "f"(alpha2p_x));
        float _mul_213;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_213) : "f"(S[77]), "f"(alpha2p_y));
        S[76] = _mul_212;
        S[77] = _mul_213;
        float _mul_214;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_214) : "f"(S[78]), "f"(alpha2p_x));
        float _mul_215;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_215) : "f"(S[79]), "f"(alpha2p_y));
        S[78] = _mul_214;
        S[79] = _mul_215;
        float _mul_216;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_216) : "f"(S[80]), "f"(alpha2p_x));
        float _mul_217;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_217) : "f"(S[81]), "f"(alpha2p_y));
        S[80] = _mul_216;
        S[81] = _mul_217;
        float _mul_218;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_218) : "f"(S[82]), "f"(alpha2p_x));
        float _mul_219;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_219) : "f"(S[83]), "f"(alpha2p_y));
        S[82] = _mul_218;
        S[83] = _mul_219;
        float _mul_220;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_220) : "f"(S[84]), "f"(alpha2p_x));
        float _mul_221;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_221) : "f"(S[85]), "f"(alpha2p_y));
        S[84] = _mul_220;
        S[85] = _mul_221;
        float _mul_222;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_222) : "f"(S[86]), "f"(alpha2p_x));
        float _mul_223;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_223) : "f"(S[87]), "f"(alpha2p_y));
        S[86] = _mul_222;
        S[87] = _mul_223;
        float _mul_224;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_224) : "f"(S[88]), "f"(alpha2p_x));
        float _mul_225;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_225) : "f"(S[89]), "f"(alpha2p_y));
        S[88] = _mul_224;
        S[89] = _mul_225;
        float _mul_226;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_226) : "f"(S[90]), "f"(alpha2p_x));
        float _mul_227;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_227) : "f"(S[91]), "f"(alpha2p_y));
        S[90] = _mul_226;
        S[91] = _mul_227;
        float _mul_228;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_228) : "f"(S[92]), "f"(alpha2p_x));
        float _mul_229;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_229) : "f"(S[93]), "f"(alpha2p_y));
        S[92] = _mul_228;
        S[93] = _mul_229;
        float _mul_230;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_230) : "f"(S[94]), "f"(alpha2p_x));
        float _mul_231;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_231) : "f"(S[95]), "f"(alpha2p_y));
        S[94] = _mul_230;
        S[95] = _mul_231;
        float _mul_232;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_232) : "f"(S[96]), "f"(alpha2p_x));
        float _mul_233;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_233) : "f"(S[97]), "f"(alpha2p_y));
        S[96] = _mul_232;
        S[97] = _mul_233;
        float _mul_234;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_234) : "f"(S[98]), "f"(alpha2p_x));
        float _mul_235;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_235) : "f"(S[99]), "f"(alpha2p_y));
        S[98] = _mul_234;
        S[99] = _mul_235;
        float _mul_236;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_236) : "f"(S[100]), "f"(alpha2p_x));
        float _mul_237;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_237) : "f"(S[101]), "f"(alpha2p_y));
        S[100] = _mul_236;
        S[101] = _mul_237;
        float _mul_238;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_238) : "f"(S[102]), "f"(alpha2p_x));
        float _mul_239;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_239) : "f"(S[103]), "f"(alpha2p_y));
        S[102] = _mul_238;
        S[103] = _mul_239;
        float _mul_240;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_240) : "f"(S[104]), "f"(alpha2p_x));
        float _mul_241;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_241) : "f"(S[105]), "f"(alpha2p_y));
        S[104] = _mul_240;
        S[105] = _mul_241;
        float _mul_242;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_242) : "f"(S[106]), "f"(alpha2p_x));
        float _mul_243;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_243) : "f"(S[107]), "f"(alpha2p_y));
        S[106] = _mul_242;
        S[107] = _mul_243;
        float _mul_244;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_244) : "f"(S[108]), "f"(alpha2p_x));
        float _mul_245;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_245) : "f"(S[109]), "f"(alpha2p_y));
        S[108] = _mul_244;
        S[109] = _mul_245;
        float _mul_246;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_246) : "f"(S[110]), "f"(alpha2p_x));
        float _mul_247;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_247) : "f"(S[111]), "f"(alpha2p_y));
        S[110] = _mul_246;
        S[111] = _mul_247;
        float _mul_248;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_248) : "f"(S[112]), "f"(alpha2p_x));
        float _mul_249;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_249) : "f"(S[113]), "f"(alpha2p_y));
        S[112] = _mul_248;
        S[113] = _mul_249;
        float _mul_250;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_250) : "f"(S[114]), "f"(alpha2p_x));
        float _mul_251;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_251) : "f"(S[115]), "f"(alpha2p_y));
        S[114] = _mul_250;
        S[115] = _mul_251;
        float _mul_252;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_252) : "f"(S[116]), "f"(alpha2p_x));
        float _mul_253;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_253) : "f"(S[117]), "f"(alpha2p_y));
        S[116] = _mul_252;
        S[117] = _mul_253;
        float _mul_254;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_254) : "f"(S[118]), "f"(alpha2p_x));
        float _mul_255;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_255) : "f"(S[119]), "f"(alpha2p_y));
        S[118] = _mul_254;
        S[119] = _mul_255;
        float _mul_256;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_256) : "f"(S[120]), "f"(alpha2p_x));
        float _mul_257;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_257) : "f"(S[121]), "f"(alpha2p_y));
        S[120] = _mul_256;
        S[121] = _mul_257;
        float _mul_258;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_258) : "f"(S[122]), "f"(alpha2p_x));
        float _mul_259;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_259) : "f"(S[123]), "f"(alpha2p_y));
        S[122] = _mul_258;
        S[123] = _mul_259;
        float _mul_260;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_260) : "f"(S[124]), "f"(alpha2p_x));
        float _mul_261;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_261) : "f"(S[125]), "f"(alpha2p_y));
        S[124] = _mul_260;
        S[125] = _mul_261;
        float _mul_262;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_262) : "f"(S[126]), "f"(alpha2p_x));
        float _mul_263;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_263) : "f"(S[127]), "f"(alpha2p_y));
        S[126] = _mul_262;
        S[127] = _mul_263;
#else
        float2 alpha2p = make_float2(alphap, alphap);
        S2[0] = mul_f32x2(S2[0], alpha2p);
        S2[1] = mul_f32x2(S2[1], alpha2p);
        S2[2] = mul_f32x2(S2[2], alpha2p);
        S2[3] = mul_f32x2(S2[3], alpha2p);
        S2[4] = mul_f32x2(S2[4], alpha2p);
        S2[5] = mul_f32x2(S2[5], alpha2p);
        S2[6] = mul_f32x2(S2[6], alpha2p);
        S2[7] = mul_f32x2(S2[7], alpha2p);
        S2[8] = mul_f32x2(S2[8], alpha2p);
        S2[9] = mul_f32x2(S2[9], alpha2p);
        S2[10] = mul_f32x2(S2[10], alpha2p);
        S2[11] = mul_f32x2(S2[11], alpha2p);
        S2[12] = mul_f32x2(S2[12], alpha2p);
        S2[13] = mul_f32x2(S2[13], alpha2p);
        S2[14] = mul_f32x2(S2[14], alpha2p);
        S2[15] = mul_f32x2(S2[15], alpha2p);
        S2[16] = mul_f32x2(S2[16], alpha2p);
        S2[17] = mul_f32x2(S2[17], alpha2p);
        S2[18] = mul_f32x2(S2[18], alpha2p);
        S2[19] = mul_f32x2(S2[19], alpha2p);
        S2[20] = mul_f32x2(S2[20], alpha2p);
        S2[21] = mul_f32x2(S2[21], alpha2p);
        S2[22] = mul_f32x2(S2[22], alpha2p);
        S2[23] = mul_f32x2(S2[23], alpha2p);
        S2[24] = mul_f32x2(S2[24], alpha2p);
        S2[25] = mul_f32x2(S2[25], alpha2p);
        S2[26] = mul_f32x2(S2[26], alpha2p);
        S2[27] = mul_f32x2(S2[27], alpha2p);
        S2[28] = mul_f32x2(S2[28], alpha2p);
        S2[29] = mul_f32x2(S2[29], alpha2p);
        S2[30] = mul_f32x2(S2[30], alpha2p);
        S2[31] = mul_f32x2(S2[31], alpha2p);
        S2[32] = mul_f32x2(S2[32], alpha2p);
        S2[33] = mul_f32x2(S2[33], alpha2p);
        S2[34] = mul_f32x2(S2[34], alpha2p);
        S2[35] = mul_f32x2(S2[35], alpha2p);
        S2[36] = mul_f32x2(S2[36], alpha2p);
        S2[37] = mul_f32x2(S2[37], alpha2p);
        S2[38] = mul_f32x2(S2[38], alpha2p);
        S2[39] = mul_f32x2(S2[39], alpha2p);
        S2[40] = mul_f32x2(S2[40], alpha2p);
        S2[41] = mul_f32x2(S2[41], alpha2p);
        S2[42] = mul_f32x2(S2[42], alpha2p);
        S2[43] = mul_f32x2(S2[43], alpha2p);
        S2[44] = mul_f32x2(S2[44], alpha2p);
        S2[45] = mul_f32x2(S2[45], alpha2p);
        S2[46] = mul_f32x2(S2[46], alpha2p);
        S2[47] = mul_f32x2(S2[47], alpha2p);
        S2[48] = mul_f32x2(S2[48], alpha2p);
        S2[49] = mul_f32x2(S2[49], alpha2p);
        S2[50] = mul_f32x2(S2[50], alpha2p);
        S2[51] = mul_f32x2(S2[51], alpha2p);
        S2[52] = mul_f32x2(S2[52], alpha2p);
        S2[53] = mul_f32x2(S2[53], alpha2p);
        S2[54] = mul_f32x2(S2[54], alpha2p);
        S2[55] = mul_f32x2(S2[55], alpha2p);
        S2[56] = mul_f32x2(S2[56], alpha2p);
        S2[57] = mul_f32x2(S2[57], alpha2p);
        S2[58] = mul_f32x2(S2[58], alpha2p);
        S2[59] = mul_f32x2(S2[59], alpha2p);
        S2[60] = mul_f32x2(S2[60], alpha2p);
        S2[61] = mul_f32x2(S2[61], alpha2p);
        S2[62] = mul_f32x2(S2[62], alpha2p);
        S2[63] = mul_f32x2(S2[63], alpha2p);
#endif
        int mp0 = t_2 * 3;
        unsigned int kbasep0 = sK_addr + (unsigned int)(mp0 * 512) + lane_byte;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                     : "r"(kbasep0));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                     : "r"(kbasep0 + 64));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                     : "r"(kbasep0 + 128));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                     : "r"(kbasep0 + 192));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[16])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 3]))
                     : "r"(kbasep0 + 256));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[20])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 3]))
                     : "r"(kbasep0 + 320));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[24])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 3]))
                     : "r"(kbasep0 + 384));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[28])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 3]))
                     : "r"(kbasep0 + 448));
#if __CUDA_ARCH__ == 900
#else
        k2[0] = make_float2(k_regs[0], k_regs[1]);
        k2[1] = make_float2(k_regs[2], k_regs[3]);
        k2[2] = make_float2(k_regs[4], k_regs[5]);
        k2[3] = make_float2(k_regs[6], k_regs[7]);
        k2[4] = make_float2(k_regs[8], k_regs[9]);
        k2[5] = make_float2(k_regs[10], k_regs[11]);
        k2[6] = make_float2(k_regs[12], k_regs[13]);
        k2[7] = make_float2(k_regs[14], k_regs[15]);
        k2[8] = make_float2(k_regs[16], k_regs[17]);
        k2[9] = make_float2(k_regs[18], k_regs[19]);
        k2[10] = make_float2(k_regs[20], k_regs[21]);
        k2[11] = make_float2(k_regs[22], k_regs[23]);
        k2[12] = make_float2(k_regs[24], k_regs[25]);
        k2[13] = make_float2(k_regs[26], k_regs[27]);
        k2[14] = make_float2(k_regs[28], k_regs[29]);
        k2[15] = make_float2(k_regs[30], k_regs[31]);
#endif
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                     : "r"(sV_addr + (unsigned int)(mp0 * 256) + group_byte));
        float betap0 = sBeta[mp0];
#if __CUDA_ARCH__ == 900
        float accp0_0_0_x = 0.0f;
        float accp0_0_0_y = 0.0f;
        float accp0_0_1_x = 0.0f;
        float accp0_0_1_y = 0.0f;
        float _fma_904;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_904)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accp0_0_0_x));
        float _fma_905;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_905)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accp0_0_0_y));
        accp0_0_0_x = _fma_904;
        accp0_0_0_y = _fma_905;
        float _fma_906;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_906)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accp0_0_1_x));
        float _fma_907;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_907)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accp0_0_1_y));
        accp0_0_1_x = _fma_906;
        accp0_0_1_y = _fma_907;
        float _fma_908;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_908)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accp0_0_0_x));
        float _fma_909;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_909)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accp0_0_0_y));
        accp0_0_0_x = _fma_908;
        accp0_0_0_y = _fma_909;
        float _fma_910;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_910)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accp0_0_1_x));
        float _fma_911;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_911)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accp0_0_1_y));
        accp0_0_1_x = _fma_910;
        accp0_0_1_y = _fma_911;
        float _fma_912;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_912)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accp0_0_0_x));
        float _fma_913;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_913)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accp0_0_0_y));
        accp0_0_0_x = _fma_912;
        accp0_0_0_y = _fma_913;
        float _fma_914;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_914)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accp0_0_1_x));
        float _fma_915;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_915)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accp0_0_1_y));
        accp0_0_1_x = _fma_914;
        accp0_0_1_y = _fma_915;
        float _fma_916;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_916)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accp0_0_0_x));
        float _fma_917;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_917)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accp0_0_0_y));
        accp0_0_0_x = _fma_916;
        accp0_0_0_y = _fma_917;
        float _fma_918;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_918)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accp0_0_1_x));
        float _fma_919;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_919)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accp0_0_1_y));
        accp0_0_1_x = _fma_918;
        accp0_0_1_y = _fma_919;
        float _fma_920;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_920)
            : "f"(S[16]), "f"(k_regs[16]), "f"(accp0_0_0_x));
        float _fma_921;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_921)
            : "f"(S[17]), "f"(k_regs[17]), "f"(accp0_0_0_y));
        accp0_0_0_x = _fma_920;
        accp0_0_0_y = _fma_921;
        float _fma_922;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_922)
            : "f"(S[18]), "f"(k_regs[18]), "f"(accp0_0_1_x));
        float _fma_923;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_923)
            : "f"(S[19]), "f"(k_regs[19]), "f"(accp0_0_1_y));
        accp0_0_1_x = _fma_922;
        accp0_0_1_y = _fma_923;
        float _fma_924;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_924)
            : "f"(S[20]), "f"(k_regs[20]), "f"(accp0_0_0_x));
        float _fma_925;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_925)
            : "f"(S[21]), "f"(k_regs[21]), "f"(accp0_0_0_y));
        accp0_0_0_x = _fma_924;
        accp0_0_0_y = _fma_925;
        float _fma_926;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_926)
            : "f"(S[22]), "f"(k_regs[22]), "f"(accp0_0_1_x));
        float _fma_927;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_927)
            : "f"(S[23]), "f"(k_regs[23]), "f"(accp0_0_1_y));
        accp0_0_1_x = _fma_926;
        accp0_0_1_y = _fma_927;
        float _fma_928;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_928)
            : "f"(S[24]), "f"(k_regs[24]), "f"(accp0_0_0_x));
        float _fma_929;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_929)
            : "f"(S[25]), "f"(k_regs[25]), "f"(accp0_0_0_y));
        accp0_0_0_x = _fma_928;
        accp0_0_0_y = _fma_929;
        float _fma_930;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_930)
            : "f"(S[26]), "f"(k_regs[26]), "f"(accp0_0_1_x));
        float _fma_931;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_931)
            : "f"(S[27]), "f"(k_regs[27]), "f"(accp0_0_1_y));
        accp0_0_1_x = _fma_930;
        accp0_0_1_y = _fma_931;
        float _fma_932;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_932)
            : "f"(S[28]), "f"(k_regs[28]), "f"(accp0_0_0_x));
        float _fma_933;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_933)
            : "f"(S[29]), "f"(k_regs[29]), "f"(accp0_0_0_y));
        accp0_0_0_x = _fma_932;
        accp0_0_0_y = _fma_933;
        float _fma_934;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_934)
            : "f"(S[30]), "f"(k_regs[30]), "f"(accp0_0_1_x));
        float _fma_935;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_935)
            : "f"(S[31]), "f"(k_regs[31]), "f"(accp0_0_1_y));
        accp0_0_1_x = _fma_934;
        accp0_0_1_y = _fma_935;
        float _add_107;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_107) : "f"(accp0_0_0_x), "f"(accp0_0_1_x));
        float _add_108;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(accp0_0_0_y), "f"(accp0_0_1_y));
        float dotp0_0 = _add_107 + _add_108;
        float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 1);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_109;
        float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 2);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_110;
#else
        float2 accp0_0_0 = make_float2(0.0f, 0.0f);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accp0_0_0);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[1], k2[1], accp0_0_0);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accp0_0_0);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[3], k2[3], accp0_0_0);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accp0_0_0);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[5], k2[5], accp0_0_0);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accp0_0_0);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[7], k2[7], accp0_0_0);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[8], k2[8], accp0_0_0);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[9], k2[9], accp0_0_0);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[10], k2[10], accp0_0_0);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[11], k2[11], accp0_0_0);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[12], k2[12], accp0_0_0);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[13], k2[13], accp0_0_0);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[14], k2[14], accp0_0_0);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[15], k2[15], accp0_0_0);
        float dotp0_0 = accp0_0_0.x + accp0_0_0.y;
        float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 1);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_75;
        float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 2);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_76;
#endif
        float vnp0_0 = (v_regs[0] - dotp0_0) * betap0;
#if __CUDA_ARCH__ == 900
        float vn2p0_0_x = vnp0_0;
        float vn2p0_0_y = vnp0_0;
        float _fma_936;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_936)
            : "f"(k_regs[0]), "f"(vn2p0_0_x), "f"(S[0]));
        float _fma_937;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_937)
            : "f"(k_regs[1]), "f"(vn2p0_0_y), "f"(S[1]));
        S[0] = _fma_936;
        S[1] = _fma_937;
        float _fma_938;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_938)
            : "f"(k_regs[2]), "f"(vn2p0_0_x), "f"(S[2]));
        float _fma_939;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_939)
            : "f"(k_regs[3]), "f"(vn2p0_0_y), "f"(S[3]));
        S[2] = _fma_938;
        S[3] = _fma_939;
        float _fma_940;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_940)
            : "f"(k_regs[4]), "f"(vn2p0_0_x), "f"(S[4]));
        float _fma_941;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_941)
            : "f"(k_regs[5]), "f"(vn2p0_0_y), "f"(S[5]));
        S[4] = _fma_940;
        S[5] = _fma_941;
        float _fma_942;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_942)
            : "f"(k_regs[6]), "f"(vn2p0_0_x), "f"(S[6]));
        float _fma_943;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_943)
            : "f"(k_regs[7]), "f"(vn2p0_0_y), "f"(S[7]));
        S[6] = _fma_942;
        S[7] = _fma_943;
        float _fma_944;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_944)
            : "f"(k_regs[8]), "f"(vn2p0_0_x), "f"(S[8]));
        float _fma_945;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_945)
            : "f"(k_regs[9]), "f"(vn2p0_0_y), "f"(S[9]));
        S[8] = _fma_944;
        S[9] = _fma_945;
        float _fma_946;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_946)
            : "f"(k_regs[10]), "f"(vn2p0_0_x), "f"(S[10]));
        float _fma_947;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_947)
            : "f"(k_regs[11]), "f"(vn2p0_0_y), "f"(S[11]));
        S[10] = _fma_946;
        S[11] = _fma_947;
        float _fma_948;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_948)
            : "f"(k_regs[12]), "f"(vn2p0_0_x), "f"(S[12]));
        float _fma_949;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_949)
            : "f"(k_regs[13]), "f"(vn2p0_0_y), "f"(S[13]));
        S[12] = _fma_948;
        S[13] = _fma_949;
        float _fma_950;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_950)
            : "f"(k_regs[14]), "f"(vn2p0_0_x), "f"(S[14]));
        float _fma_951;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_951)
            : "f"(k_regs[15]), "f"(vn2p0_0_y), "f"(S[15]));
        S[14] = _fma_950;
        S[15] = _fma_951;
        float _fma_952;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_952)
            : "f"(k_regs[16]), "f"(vn2p0_0_x), "f"(S[16]));
        float _fma_953;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_953)
            : "f"(k_regs[17]), "f"(vn2p0_0_y), "f"(S[17]));
        S[16] = _fma_952;
        S[17] = _fma_953;
        float _fma_954;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_954)
            : "f"(k_regs[18]), "f"(vn2p0_0_x), "f"(S[18]));
        float _fma_955;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_955)
            : "f"(k_regs[19]), "f"(vn2p0_0_y), "f"(S[19]));
        S[18] = _fma_954;
        S[19] = _fma_955;
        float _fma_956;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_956)
            : "f"(k_regs[20]), "f"(vn2p0_0_x), "f"(S[20]));
        float _fma_957;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_957)
            : "f"(k_regs[21]), "f"(vn2p0_0_y), "f"(S[21]));
        S[20] = _fma_956;
        S[21] = _fma_957;
        float _fma_958;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_958)
            : "f"(k_regs[22]), "f"(vn2p0_0_x), "f"(S[22]));
        float _fma_959;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_959)
            : "f"(k_regs[23]), "f"(vn2p0_0_y), "f"(S[23]));
        S[22] = _fma_958;
        S[23] = _fma_959;
        float _fma_960;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_960)
            : "f"(k_regs[24]), "f"(vn2p0_0_x), "f"(S[24]));
        float _fma_961;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_961)
            : "f"(k_regs[25]), "f"(vn2p0_0_y), "f"(S[25]));
        S[24] = _fma_960;
        S[25] = _fma_961;
        float _fma_962;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_962)
            : "f"(k_regs[26]), "f"(vn2p0_0_x), "f"(S[26]));
        float _fma_963;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_963)
            : "f"(k_regs[27]), "f"(vn2p0_0_y), "f"(S[27]));
        S[26] = _fma_962;
        S[27] = _fma_963;
        float _fma_964;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_964)
            : "f"(k_regs[28]), "f"(vn2p0_0_x), "f"(S[28]));
        float _fma_965;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_965)
            : "f"(k_regs[29]), "f"(vn2p0_0_y), "f"(S[29]));
        S[28] = _fma_964;
        S[29] = _fma_965;
        float _fma_966;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_966)
            : "f"(k_regs[30]), "f"(vn2p0_0_x), "f"(S[30]));
        float _fma_967;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_967)
            : "f"(k_regs[31]), "f"(vn2p0_0_y), "f"(S[31]));
        S[30] = _fma_966;
        S[31] = _fma_967;
        float accp0_1_0_x = 0.0f;
        float accp0_1_0_y = 0.0f;
        float accp0_1_1_x = 0.0f;
        float accp0_1_1_y = 0.0f;
        float _fma_968;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_968)
            : "f"(S[32]), "f"(k_regs[0]), "f"(accp0_1_0_x));
        float _fma_969;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_969)
            : "f"(S[33]), "f"(k_regs[1]), "f"(accp0_1_0_y));
        accp0_1_0_x = _fma_968;
        accp0_1_0_y = _fma_969;
        float _fma_970;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_970)
            : "f"(S[34]), "f"(k_regs[2]), "f"(accp0_1_1_x));
        float _fma_971;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_971)
            : "f"(S[35]), "f"(k_regs[3]), "f"(accp0_1_1_y));
        accp0_1_1_x = _fma_970;
        accp0_1_1_y = _fma_971;
        float _fma_972;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_972)
            : "f"(S[36]), "f"(k_regs[4]), "f"(accp0_1_0_x));
        float _fma_973;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_973)
            : "f"(S[37]), "f"(k_regs[5]), "f"(accp0_1_0_y));
        accp0_1_0_x = _fma_972;
        accp0_1_0_y = _fma_973;
        float _fma_974;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_974)
            : "f"(S[38]), "f"(k_regs[6]), "f"(accp0_1_1_x));
        float _fma_975;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_975)
            : "f"(S[39]), "f"(k_regs[7]), "f"(accp0_1_1_y));
        accp0_1_1_x = _fma_974;
        accp0_1_1_y = _fma_975;
        float _fma_976;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_976)
            : "f"(S[40]), "f"(k_regs[8]), "f"(accp0_1_0_x));
        float _fma_977;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_977)
            : "f"(S[41]), "f"(k_regs[9]), "f"(accp0_1_0_y));
        accp0_1_0_x = _fma_976;
        accp0_1_0_y = _fma_977;
        float _fma_978;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_978)
            : "f"(S[42]), "f"(k_regs[10]), "f"(accp0_1_1_x));
        float _fma_979;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_979)
            : "f"(S[43]), "f"(k_regs[11]), "f"(accp0_1_1_y));
        accp0_1_1_x = _fma_978;
        accp0_1_1_y = _fma_979;
        float _fma_980;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_980)
            : "f"(S[44]), "f"(k_regs[12]), "f"(accp0_1_0_x));
        float _fma_981;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_981)
            : "f"(S[45]), "f"(k_regs[13]), "f"(accp0_1_0_y));
        accp0_1_0_x = _fma_980;
        accp0_1_0_y = _fma_981;
        float _fma_982;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_982)
            : "f"(S[46]), "f"(k_regs[14]), "f"(accp0_1_1_x));
        float _fma_983;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_983)
            : "f"(S[47]), "f"(k_regs[15]), "f"(accp0_1_1_y));
        accp0_1_1_x = _fma_982;
        accp0_1_1_y = _fma_983;
        float _fma_984;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_984)
            : "f"(S[48]), "f"(k_regs[16]), "f"(accp0_1_0_x));
        float _fma_985;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_985)
            : "f"(S[49]), "f"(k_regs[17]), "f"(accp0_1_0_y));
        accp0_1_0_x = _fma_984;
        accp0_1_0_y = _fma_985;
        float _fma_986;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_986)
            : "f"(S[50]), "f"(k_regs[18]), "f"(accp0_1_1_x));
        float _fma_987;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_987)
            : "f"(S[51]), "f"(k_regs[19]), "f"(accp0_1_1_y));
        accp0_1_1_x = _fma_986;
        accp0_1_1_y = _fma_987;
        float _fma_988;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_988)
            : "f"(S[52]), "f"(k_regs[20]), "f"(accp0_1_0_x));
        float _fma_989;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_989)
            : "f"(S[53]), "f"(k_regs[21]), "f"(accp0_1_0_y));
        accp0_1_0_x = _fma_988;
        accp0_1_0_y = _fma_989;
        float _fma_990;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_990)
            : "f"(S[54]), "f"(k_regs[22]), "f"(accp0_1_1_x));
        float _fma_991;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_991)
            : "f"(S[55]), "f"(k_regs[23]), "f"(accp0_1_1_y));
        accp0_1_1_x = _fma_990;
        accp0_1_1_y = _fma_991;
        float _fma_992;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_992)
            : "f"(S[56]), "f"(k_regs[24]), "f"(accp0_1_0_x));
        float _fma_993;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_993)
            : "f"(S[57]), "f"(k_regs[25]), "f"(accp0_1_0_y));
        accp0_1_0_x = _fma_992;
        accp0_1_0_y = _fma_993;
        float _fma_994;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_994)
            : "f"(S[58]), "f"(k_regs[26]), "f"(accp0_1_1_x));
        float _fma_995;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_995)
            : "f"(S[59]), "f"(k_regs[27]), "f"(accp0_1_1_y));
        accp0_1_1_x = _fma_994;
        accp0_1_1_y = _fma_995;
        float _fma_996;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_996)
            : "f"(S[60]), "f"(k_regs[28]), "f"(accp0_1_0_x));
        float _fma_997;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_997)
            : "f"(S[61]), "f"(k_regs[29]), "f"(accp0_1_0_y));
        accp0_1_0_x = _fma_996;
        accp0_1_0_y = _fma_997;
        float _fma_998;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_998)
            : "f"(S[62]), "f"(k_regs[30]), "f"(accp0_1_1_x));
        float _fma_999;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_999)
            : "f"(S[63]), "f"(k_regs[31]), "f"(accp0_1_1_y));
        accp0_1_1_x = _fma_998;
        accp0_1_1_y = _fma_999;
        float _add_111;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_111) : "f"(accp0_1_0_x), "f"(accp0_1_1_x));
        float _add_112;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(accp0_1_0_y), "f"(accp0_1_1_y));
        float dotp0_1 = _add_111 + _add_112;
        float _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 1);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_113;
        float _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 2);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_114;
#else
        float2 vn2p0_0 = make_float2(vnp0_0, vnp0_0);
        S2[0] = fma_f32x2_rn_ftz(k2[0], vn2p0_0, S2[0]);
        S2[1] = fma_f32x2_rn_ftz(k2[1], vn2p0_0, S2[1]);
        S2[2] = fma_f32x2_rn_ftz(k2[2], vn2p0_0, S2[2]);
        S2[3] = fma_f32x2_rn_ftz(k2[3], vn2p0_0, S2[3]);
        S2[4] = fma_f32x2_rn_ftz(k2[4], vn2p0_0, S2[4]);
        S2[5] = fma_f32x2_rn_ftz(k2[5], vn2p0_0, S2[5]);
        S2[6] = fma_f32x2_rn_ftz(k2[6], vn2p0_0, S2[6]);
        S2[7] = fma_f32x2_rn_ftz(k2[7], vn2p0_0, S2[7]);
        S2[8] = fma_f32x2_rn_ftz(k2[8], vn2p0_0, S2[8]);
        S2[9] = fma_f32x2_rn_ftz(k2[9], vn2p0_0, S2[9]);
        S2[10] = fma_f32x2_rn_ftz(k2[10], vn2p0_0, S2[10]);
        S2[11] = fma_f32x2_rn_ftz(k2[11], vn2p0_0, S2[11]);
        S2[12] = fma_f32x2_rn_ftz(k2[12], vn2p0_0, S2[12]);
        S2[13] = fma_f32x2_rn_ftz(k2[13], vn2p0_0, S2[13]);
        S2[14] = fma_f32x2_rn_ftz(k2[14], vn2p0_0, S2[14]);
        S2[15] = fma_f32x2_rn_ftz(k2[15], vn2p0_0, S2[15]);
        float2 accp0_1_0 = make_float2(0.0f, 0.0f);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[24], k2[8], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[25], k2[9], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[26], k2[10], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[27], k2[11], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[28], k2[12], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[29], k2[13], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[30], k2[14], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[31], k2[15], accp0_1_0);
        float dotp0_1 = accp0_1_0.x + accp0_1_0.y;
        float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 1);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_77;
        float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 2);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_78;
#endif
        float vnp0_1 = (v_regs[1] - dotp0_1) * betap0;
#if __CUDA_ARCH__ == 900
        float vn2p0_1_x = vnp0_1;
        float vn2p0_1_y = vnp0_1;
        float _fma_1000;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1000)
            : "f"(k_regs[0]), "f"(vn2p0_1_x), "f"(S[32]));
        float _fma_1001;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1001)
            : "f"(k_regs[1]), "f"(vn2p0_1_y), "f"(S[33]));
        S[32] = _fma_1000;
        S[33] = _fma_1001;
        float _fma_1002;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1002)
            : "f"(k_regs[2]), "f"(vn2p0_1_x), "f"(S[34]));
        float _fma_1003;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1003)
            : "f"(k_regs[3]), "f"(vn2p0_1_y), "f"(S[35]));
        S[34] = _fma_1002;
        S[35] = _fma_1003;
        float _fma_1004;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1004)
            : "f"(k_regs[4]), "f"(vn2p0_1_x), "f"(S[36]));
        float _fma_1005;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1005)
            : "f"(k_regs[5]), "f"(vn2p0_1_y), "f"(S[37]));
        S[36] = _fma_1004;
        S[37] = _fma_1005;
        float _fma_1006;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1006)
            : "f"(k_regs[6]), "f"(vn2p0_1_x), "f"(S[38]));
        float _fma_1007;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1007)
            : "f"(k_regs[7]), "f"(vn2p0_1_y), "f"(S[39]));
        S[38] = _fma_1006;
        S[39] = _fma_1007;
        float _fma_1008;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1008)
            : "f"(k_regs[8]), "f"(vn2p0_1_x), "f"(S[40]));
        float _fma_1009;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1009)
            : "f"(k_regs[9]), "f"(vn2p0_1_y), "f"(S[41]));
        S[40] = _fma_1008;
        S[41] = _fma_1009;
        float _fma_1010;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1010)
            : "f"(k_regs[10]), "f"(vn2p0_1_x), "f"(S[42]));
        float _fma_1011;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1011)
            : "f"(k_regs[11]), "f"(vn2p0_1_y), "f"(S[43]));
        S[42] = _fma_1010;
        S[43] = _fma_1011;
        float _fma_1012;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1012)
            : "f"(k_regs[12]), "f"(vn2p0_1_x), "f"(S[44]));
        float _fma_1013;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1013)
            : "f"(k_regs[13]), "f"(vn2p0_1_y), "f"(S[45]));
        S[44] = _fma_1012;
        S[45] = _fma_1013;
        float _fma_1014;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1014)
            : "f"(k_regs[14]), "f"(vn2p0_1_x), "f"(S[46]));
        float _fma_1015;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1015)
            : "f"(k_regs[15]), "f"(vn2p0_1_y), "f"(S[47]));
        S[46] = _fma_1014;
        S[47] = _fma_1015;
        float _fma_1016;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1016)
            : "f"(k_regs[16]), "f"(vn2p0_1_x), "f"(S[48]));
        float _fma_1017;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1017)
            : "f"(k_regs[17]), "f"(vn2p0_1_y), "f"(S[49]));
        S[48] = _fma_1016;
        S[49] = _fma_1017;
        float _fma_1018;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1018)
            : "f"(k_regs[18]), "f"(vn2p0_1_x), "f"(S[50]));
        float _fma_1019;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1019)
            : "f"(k_regs[19]), "f"(vn2p0_1_y), "f"(S[51]));
        S[50] = _fma_1018;
        S[51] = _fma_1019;
        float _fma_1020;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1020)
            : "f"(k_regs[20]), "f"(vn2p0_1_x), "f"(S[52]));
        float _fma_1021;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1021)
            : "f"(k_regs[21]), "f"(vn2p0_1_y), "f"(S[53]));
        S[52] = _fma_1020;
        S[53] = _fma_1021;
        float _fma_1022;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1022)
            : "f"(k_regs[22]), "f"(vn2p0_1_x), "f"(S[54]));
        float _fma_1023;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1023)
            : "f"(k_regs[23]), "f"(vn2p0_1_y), "f"(S[55]));
        S[54] = _fma_1022;
        S[55] = _fma_1023;
        float _fma_1024;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1024)
            : "f"(k_regs[24]), "f"(vn2p0_1_x), "f"(S[56]));
        float _fma_1025;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1025)
            : "f"(k_regs[25]), "f"(vn2p0_1_y), "f"(S[57]));
        S[56] = _fma_1024;
        S[57] = _fma_1025;
        float _fma_1026;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1026)
            : "f"(k_regs[26]), "f"(vn2p0_1_x), "f"(S[58]));
        float _fma_1027;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1027)
            : "f"(k_regs[27]), "f"(vn2p0_1_y), "f"(S[59]));
        S[58] = _fma_1026;
        S[59] = _fma_1027;
        float _fma_1028;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1028)
            : "f"(k_regs[28]), "f"(vn2p0_1_x), "f"(S[60]));
        float _fma_1029;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1029)
            : "f"(k_regs[29]), "f"(vn2p0_1_y), "f"(S[61]));
        S[60] = _fma_1028;
        S[61] = _fma_1029;
        float _fma_1030;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1030)
            : "f"(k_regs[30]), "f"(vn2p0_1_x), "f"(S[62]));
        float _fma_1031;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1031)
            : "f"(k_regs[31]), "f"(vn2p0_1_y), "f"(S[63]));
        S[62] = _fma_1030;
        S[63] = _fma_1031;
        float accp0_2_0_x = 0.0f;
        float accp0_2_0_y = 0.0f;
        float accp0_2_1_x = 0.0f;
        float accp0_2_1_y = 0.0f;
        float _fma_1032;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1032)
            : "f"(S[64]), "f"(k_regs[0]), "f"(accp0_2_0_x));
        float _fma_1033;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1033)
            : "f"(S[65]), "f"(k_regs[1]), "f"(accp0_2_0_y));
        accp0_2_0_x = _fma_1032;
        accp0_2_0_y = _fma_1033;
        float _fma_1034;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1034)
            : "f"(S[66]), "f"(k_regs[2]), "f"(accp0_2_1_x));
        float _fma_1035;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1035)
            : "f"(S[67]), "f"(k_regs[3]), "f"(accp0_2_1_y));
        accp0_2_1_x = _fma_1034;
        accp0_2_1_y = _fma_1035;
        float _fma_1036;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1036)
            : "f"(S[68]), "f"(k_regs[4]), "f"(accp0_2_0_x));
        float _fma_1037;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1037)
            : "f"(S[69]), "f"(k_regs[5]), "f"(accp0_2_0_y));
        accp0_2_0_x = _fma_1036;
        accp0_2_0_y = _fma_1037;
        float _fma_1038;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1038)
            : "f"(S[70]), "f"(k_regs[6]), "f"(accp0_2_1_x));
        float _fma_1039;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1039)
            : "f"(S[71]), "f"(k_regs[7]), "f"(accp0_2_1_y));
        accp0_2_1_x = _fma_1038;
        accp0_2_1_y = _fma_1039;
        float _fma_1040;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1040)
            : "f"(S[72]), "f"(k_regs[8]), "f"(accp0_2_0_x));
        float _fma_1041;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1041)
            : "f"(S[73]), "f"(k_regs[9]), "f"(accp0_2_0_y));
        accp0_2_0_x = _fma_1040;
        accp0_2_0_y = _fma_1041;
        float _fma_1042;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1042)
            : "f"(S[74]), "f"(k_regs[10]), "f"(accp0_2_1_x));
        float _fma_1043;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1043)
            : "f"(S[75]), "f"(k_regs[11]), "f"(accp0_2_1_y));
        accp0_2_1_x = _fma_1042;
        accp0_2_1_y = _fma_1043;
        float _fma_1044;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1044)
            : "f"(S[76]), "f"(k_regs[12]), "f"(accp0_2_0_x));
        float _fma_1045;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1045)
            : "f"(S[77]), "f"(k_regs[13]), "f"(accp0_2_0_y));
        accp0_2_0_x = _fma_1044;
        accp0_2_0_y = _fma_1045;
        float _fma_1046;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1046)
            : "f"(S[78]), "f"(k_regs[14]), "f"(accp0_2_1_x));
        float _fma_1047;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1047)
            : "f"(S[79]), "f"(k_regs[15]), "f"(accp0_2_1_y));
        accp0_2_1_x = _fma_1046;
        accp0_2_1_y = _fma_1047;
        float _fma_1048;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1048)
            : "f"(S[80]), "f"(k_regs[16]), "f"(accp0_2_0_x));
        float _fma_1049;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1049)
            : "f"(S[81]), "f"(k_regs[17]), "f"(accp0_2_0_y));
        accp0_2_0_x = _fma_1048;
        accp0_2_0_y = _fma_1049;
        float _fma_1050;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1050)
            : "f"(S[82]), "f"(k_regs[18]), "f"(accp0_2_1_x));
        float _fma_1051;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1051)
            : "f"(S[83]), "f"(k_regs[19]), "f"(accp0_2_1_y));
        accp0_2_1_x = _fma_1050;
        accp0_2_1_y = _fma_1051;
        float _fma_1052;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1052)
            : "f"(S[84]), "f"(k_regs[20]), "f"(accp0_2_0_x));
        float _fma_1053;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1053)
            : "f"(S[85]), "f"(k_regs[21]), "f"(accp0_2_0_y));
        accp0_2_0_x = _fma_1052;
        accp0_2_0_y = _fma_1053;
        float _fma_1054;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1054)
            : "f"(S[86]), "f"(k_regs[22]), "f"(accp0_2_1_x));
        float _fma_1055;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1055)
            : "f"(S[87]), "f"(k_regs[23]), "f"(accp0_2_1_y));
        accp0_2_1_x = _fma_1054;
        accp0_2_1_y = _fma_1055;
        float _fma_1056;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1056)
            : "f"(S[88]), "f"(k_regs[24]), "f"(accp0_2_0_x));
        float _fma_1057;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1057)
            : "f"(S[89]), "f"(k_regs[25]), "f"(accp0_2_0_y));
        accp0_2_0_x = _fma_1056;
        accp0_2_0_y = _fma_1057;
        float _fma_1058;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1058)
            : "f"(S[90]), "f"(k_regs[26]), "f"(accp0_2_1_x));
        float _fma_1059;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1059)
            : "f"(S[91]), "f"(k_regs[27]), "f"(accp0_2_1_y));
        accp0_2_1_x = _fma_1058;
        accp0_2_1_y = _fma_1059;
        float _fma_1060;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1060)
            : "f"(S[92]), "f"(k_regs[28]), "f"(accp0_2_0_x));
        float _fma_1061;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1061)
            : "f"(S[93]), "f"(k_regs[29]), "f"(accp0_2_0_y));
        accp0_2_0_x = _fma_1060;
        accp0_2_0_y = _fma_1061;
        float _fma_1062;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1062)
            : "f"(S[94]), "f"(k_regs[30]), "f"(accp0_2_1_x));
        float _fma_1063;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1063)
            : "f"(S[95]), "f"(k_regs[31]), "f"(accp0_2_1_y));
        accp0_2_1_x = _fma_1062;
        accp0_2_1_y = _fma_1063;
        float _add_115;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_115) : "f"(accp0_2_0_x), "f"(accp0_2_1_x));
        float _add_116;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(accp0_2_0_y), "f"(accp0_2_1_y));
        float dotp0_2 = _add_115 + _add_116;
        float _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, dotp0_2, 1);
        dotp0_2 = dotp0_2 + _shfl_xor_sync_117;
        float _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, dotp0_2, 2);
        dotp0_2 = dotp0_2 + _shfl_xor_sync_118;
#else
        float2 vn2p0_1 = make_float2(vnp0_1, vnp0_1);
        S2[16] = fma_f32x2_rn_ftz(k2[0], vn2p0_1, S2[16]);
        S2[17] = fma_f32x2_rn_ftz(k2[1], vn2p0_1, S2[17]);
        S2[18] = fma_f32x2_rn_ftz(k2[2], vn2p0_1, S2[18]);
        S2[19] = fma_f32x2_rn_ftz(k2[3], vn2p0_1, S2[19]);
        S2[20] = fma_f32x2_rn_ftz(k2[4], vn2p0_1, S2[20]);
        S2[21] = fma_f32x2_rn_ftz(k2[5], vn2p0_1, S2[21]);
        S2[22] = fma_f32x2_rn_ftz(k2[6], vn2p0_1, S2[22]);
        S2[23] = fma_f32x2_rn_ftz(k2[7], vn2p0_1, S2[23]);
        S2[24] = fma_f32x2_rn_ftz(k2[8], vn2p0_1, S2[24]);
        S2[25] = fma_f32x2_rn_ftz(k2[9], vn2p0_1, S2[25]);
        S2[26] = fma_f32x2_rn_ftz(k2[10], vn2p0_1, S2[26]);
        S2[27] = fma_f32x2_rn_ftz(k2[11], vn2p0_1, S2[27]);
        S2[28] = fma_f32x2_rn_ftz(k2[12], vn2p0_1, S2[28]);
        S2[29] = fma_f32x2_rn_ftz(k2[13], vn2p0_1, S2[29]);
        S2[30] = fma_f32x2_rn_ftz(k2[14], vn2p0_1, S2[30]);
        S2[31] = fma_f32x2_rn_ftz(k2[15], vn2p0_1, S2[31]);
        float2 accp0_2_0 = make_float2(0.0f, 0.0f);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[32], k2[0], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[33], k2[1], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[34], k2[2], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[35], k2[3], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[36], k2[4], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[37], k2[5], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[38], k2[6], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[39], k2[7], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[40], k2[8], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[41], k2[9], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[42], k2[10], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[43], k2[11], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[44], k2[12], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[45], k2[13], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[46], k2[14], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[47], k2[15], accp0_2_0);
        float dotp0_2 = accp0_2_0.x + accp0_2_0.y;
        float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, dotp0_2, 1);
        dotp0_2 = dotp0_2 + _shfl_xor_sync_79;
        float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, dotp0_2, 2);
        dotp0_2 = dotp0_2 + _shfl_xor_sync_80;
#endif
        float vnp0_2 = (v_regs[2] - dotp0_2) * betap0;
#if __CUDA_ARCH__ == 900
        float vn2p0_2_x = vnp0_2;
        float vn2p0_2_y = vnp0_2;
        float _fma_1064;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1064)
            : "f"(k_regs[0]), "f"(vn2p0_2_x), "f"(S[64]));
        float _fma_1065;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1065)
            : "f"(k_regs[1]), "f"(vn2p0_2_y), "f"(S[65]));
        S[64] = _fma_1064;
        S[65] = _fma_1065;
        float _fma_1066;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1066)
            : "f"(k_regs[2]), "f"(vn2p0_2_x), "f"(S[66]));
        float _fma_1067;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1067)
            : "f"(k_regs[3]), "f"(vn2p0_2_y), "f"(S[67]));
        S[66] = _fma_1066;
        S[67] = _fma_1067;
        float _fma_1068;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1068)
            : "f"(k_regs[4]), "f"(vn2p0_2_x), "f"(S[68]));
        float _fma_1069;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1069)
            : "f"(k_regs[5]), "f"(vn2p0_2_y), "f"(S[69]));
        S[68] = _fma_1068;
        S[69] = _fma_1069;
        float _fma_1070;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1070)
            : "f"(k_regs[6]), "f"(vn2p0_2_x), "f"(S[70]));
        float _fma_1071;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1071)
            : "f"(k_regs[7]), "f"(vn2p0_2_y), "f"(S[71]));
        S[70] = _fma_1070;
        S[71] = _fma_1071;
        float _fma_1072;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1072)
            : "f"(k_regs[8]), "f"(vn2p0_2_x), "f"(S[72]));
        float _fma_1073;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1073)
            : "f"(k_regs[9]), "f"(vn2p0_2_y), "f"(S[73]));
        S[72] = _fma_1072;
        S[73] = _fma_1073;
        float _fma_1074;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1074)
            : "f"(k_regs[10]), "f"(vn2p0_2_x), "f"(S[74]));
        float _fma_1075;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1075)
            : "f"(k_regs[11]), "f"(vn2p0_2_y), "f"(S[75]));
        S[74] = _fma_1074;
        S[75] = _fma_1075;
        float _fma_1076;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1076)
            : "f"(k_regs[12]), "f"(vn2p0_2_x), "f"(S[76]));
        float _fma_1077;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1077)
            : "f"(k_regs[13]), "f"(vn2p0_2_y), "f"(S[77]));
        S[76] = _fma_1076;
        S[77] = _fma_1077;
        float _fma_1078;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1078)
            : "f"(k_regs[14]), "f"(vn2p0_2_x), "f"(S[78]));
        float _fma_1079;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1079)
            : "f"(k_regs[15]), "f"(vn2p0_2_y), "f"(S[79]));
        S[78] = _fma_1078;
        S[79] = _fma_1079;
        float _fma_1080;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1080)
            : "f"(k_regs[16]), "f"(vn2p0_2_x), "f"(S[80]));
        float _fma_1081;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1081)
            : "f"(k_regs[17]), "f"(vn2p0_2_y), "f"(S[81]));
        S[80] = _fma_1080;
        S[81] = _fma_1081;
        float _fma_1082;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1082)
            : "f"(k_regs[18]), "f"(vn2p0_2_x), "f"(S[82]));
        float _fma_1083;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1083)
            : "f"(k_regs[19]), "f"(vn2p0_2_y), "f"(S[83]));
        S[82] = _fma_1082;
        S[83] = _fma_1083;
        float _fma_1084;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1084)
            : "f"(k_regs[20]), "f"(vn2p0_2_x), "f"(S[84]));
        float _fma_1085;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1085)
            : "f"(k_regs[21]), "f"(vn2p0_2_y), "f"(S[85]));
        S[84] = _fma_1084;
        S[85] = _fma_1085;
        float _fma_1086;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1086)
            : "f"(k_regs[22]), "f"(vn2p0_2_x), "f"(S[86]));
        float _fma_1087;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1087)
            : "f"(k_regs[23]), "f"(vn2p0_2_y), "f"(S[87]));
        S[86] = _fma_1086;
        S[87] = _fma_1087;
        float _fma_1088;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1088)
            : "f"(k_regs[24]), "f"(vn2p0_2_x), "f"(S[88]));
        float _fma_1089;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1089)
            : "f"(k_regs[25]), "f"(vn2p0_2_y), "f"(S[89]));
        S[88] = _fma_1088;
        S[89] = _fma_1089;
        float _fma_1090;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1090)
            : "f"(k_regs[26]), "f"(vn2p0_2_x), "f"(S[90]));
        float _fma_1091;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1091)
            : "f"(k_regs[27]), "f"(vn2p0_2_y), "f"(S[91]));
        S[90] = _fma_1090;
        S[91] = _fma_1091;
        float _fma_1092;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1092)
            : "f"(k_regs[28]), "f"(vn2p0_2_x), "f"(S[92]));
        float _fma_1093;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1093)
            : "f"(k_regs[29]), "f"(vn2p0_2_y), "f"(S[93]));
        S[92] = _fma_1092;
        S[93] = _fma_1093;
        float _fma_1094;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1094)
            : "f"(k_regs[30]), "f"(vn2p0_2_x), "f"(S[94]));
        float _fma_1095;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1095)
            : "f"(k_regs[31]), "f"(vn2p0_2_y), "f"(S[95]));
        S[94] = _fma_1094;
        S[95] = _fma_1095;
        float accp0_3_0_x = 0.0f;
        float accp0_3_0_y = 0.0f;
        float accp0_3_1_x = 0.0f;
        float accp0_3_1_y = 0.0f;
        float _fma_1096;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1096)
            : "f"(S[96]), "f"(k_regs[0]), "f"(accp0_3_0_x));
        float _fma_1097;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1097)
            : "f"(S[97]), "f"(k_regs[1]), "f"(accp0_3_0_y));
        accp0_3_0_x = _fma_1096;
        accp0_3_0_y = _fma_1097;
        float _fma_1098;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1098)
            : "f"(S[98]), "f"(k_regs[2]), "f"(accp0_3_1_x));
        float _fma_1099;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1099)
            : "f"(S[99]), "f"(k_regs[3]), "f"(accp0_3_1_y));
        accp0_3_1_x = _fma_1098;
        accp0_3_1_y = _fma_1099;
        float _fma_1100;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1100)
            : "f"(S[100]), "f"(k_regs[4]), "f"(accp0_3_0_x));
        float _fma_1101;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1101)
            : "f"(S[101]), "f"(k_regs[5]), "f"(accp0_3_0_y));
        accp0_3_0_x = _fma_1100;
        accp0_3_0_y = _fma_1101;
        float _fma_1102;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1102)
            : "f"(S[102]), "f"(k_regs[6]), "f"(accp0_3_1_x));
        float _fma_1103;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1103)
            : "f"(S[103]), "f"(k_regs[7]), "f"(accp0_3_1_y));
        accp0_3_1_x = _fma_1102;
        accp0_3_1_y = _fma_1103;
        float _fma_1104;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1104)
            : "f"(S[104]), "f"(k_regs[8]), "f"(accp0_3_0_x));
        float _fma_1105;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1105)
            : "f"(S[105]), "f"(k_regs[9]), "f"(accp0_3_0_y));
        accp0_3_0_x = _fma_1104;
        accp0_3_0_y = _fma_1105;
        float _fma_1106;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1106)
            : "f"(S[106]), "f"(k_regs[10]), "f"(accp0_3_1_x));
        float _fma_1107;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1107)
            : "f"(S[107]), "f"(k_regs[11]), "f"(accp0_3_1_y));
        accp0_3_1_x = _fma_1106;
        accp0_3_1_y = _fma_1107;
        float _fma_1108;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1108)
            : "f"(S[108]), "f"(k_regs[12]), "f"(accp0_3_0_x));
        float _fma_1109;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1109)
            : "f"(S[109]), "f"(k_regs[13]), "f"(accp0_3_0_y));
        accp0_3_0_x = _fma_1108;
        accp0_3_0_y = _fma_1109;
        float _fma_1110;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1110)
            : "f"(S[110]), "f"(k_regs[14]), "f"(accp0_3_1_x));
        float _fma_1111;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1111)
            : "f"(S[111]), "f"(k_regs[15]), "f"(accp0_3_1_y));
        accp0_3_1_x = _fma_1110;
        accp0_3_1_y = _fma_1111;
        float _fma_1112;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1112)
            : "f"(S[112]), "f"(k_regs[16]), "f"(accp0_3_0_x));
        float _fma_1113;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1113)
            : "f"(S[113]), "f"(k_regs[17]), "f"(accp0_3_0_y));
        accp0_3_0_x = _fma_1112;
        accp0_3_0_y = _fma_1113;
        float _fma_1114;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1114)
            : "f"(S[114]), "f"(k_regs[18]), "f"(accp0_3_1_x));
        float _fma_1115;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1115)
            : "f"(S[115]), "f"(k_regs[19]), "f"(accp0_3_1_y));
        accp0_3_1_x = _fma_1114;
        accp0_3_1_y = _fma_1115;
        float _fma_1116;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1116)
            : "f"(S[116]), "f"(k_regs[20]), "f"(accp0_3_0_x));
        float _fma_1117;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1117)
            : "f"(S[117]), "f"(k_regs[21]), "f"(accp0_3_0_y));
        accp0_3_0_x = _fma_1116;
        accp0_3_0_y = _fma_1117;
        float _fma_1118;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1118)
            : "f"(S[118]), "f"(k_regs[22]), "f"(accp0_3_1_x));
        float _fma_1119;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1119)
            : "f"(S[119]), "f"(k_regs[23]), "f"(accp0_3_1_y));
        accp0_3_1_x = _fma_1118;
        accp0_3_1_y = _fma_1119;
        float _fma_1120;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1120)
            : "f"(S[120]), "f"(k_regs[24]), "f"(accp0_3_0_x));
        float _fma_1121;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1121)
            : "f"(S[121]), "f"(k_regs[25]), "f"(accp0_3_0_y));
        accp0_3_0_x = _fma_1120;
        accp0_3_0_y = _fma_1121;
        float _fma_1122;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1122)
            : "f"(S[122]), "f"(k_regs[26]), "f"(accp0_3_1_x));
        float _fma_1123;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1123)
            : "f"(S[123]), "f"(k_regs[27]), "f"(accp0_3_1_y));
        accp0_3_1_x = _fma_1122;
        accp0_3_1_y = _fma_1123;
        float _fma_1124;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1124)
            : "f"(S[124]), "f"(k_regs[28]), "f"(accp0_3_0_x));
        float _fma_1125;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1125)
            : "f"(S[125]), "f"(k_regs[29]), "f"(accp0_3_0_y));
        accp0_3_0_x = _fma_1124;
        accp0_3_0_y = _fma_1125;
        float _fma_1126;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1126)
            : "f"(S[126]), "f"(k_regs[30]), "f"(accp0_3_1_x));
        float _fma_1127;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1127)
            : "f"(S[127]), "f"(k_regs[31]), "f"(accp0_3_1_y));
        accp0_3_1_x = _fma_1126;
        accp0_3_1_y = _fma_1127;
        float _add_119;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_119) : "f"(accp0_3_0_x), "f"(accp0_3_1_x));
        float _add_120;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(accp0_3_0_y), "f"(accp0_3_1_y));
        float dotp0_3 = _add_119 + _add_120;
        float _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, dotp0_3, 1);
        dotp0_3 = dotp0_3 + _shfl_xor_sync_121;
        float _shfl_xor_sync_122 = __shfl_xor_sync(0xFFFFFFFF, dotp0_3, 2);
        dotp0_3 = dotp0_3 + _shfl_xor_sync_122;
#else
        float2 vn2p0_2 = make_float2(vnp0_2, vnp0_2);
        S2[32] = fma_f32x2_rn_ftz(k2[0], vn2p0_2, S2[32]);
        S2[33] = fma_f32x2_rn_ftz(k2[1], vn2p0_2, S2[33]);
        S2[34] = fma_f32x2_rn_ftz(k2[2], vn2p0_2, S2[34]);
        S2[35] = fma_f32x2_rn_ftz(k2[3], vn2p0_2, S2[35]);
        S2[36] = fma_f32x2_rn_ftz(k2[4], vn2p0_2, S2[36]);
        S2[37] = fma_f32x2_rn_ftz(k2[5], vn2p0_2, S2[37]);
        S2[38] = fma_f32x2_rn_ftz(k2[6], vn2p0_2, S2[38]);
        S2[39] = fma_f32x2_rn_ftz(k2[7], vn2p0_2, S2[39]);
        S2[40] = fma_f32x2_rn_ftz(k2[8], vn2p0_2, S2[40]);
        S2[41] = fma_f32x2_rn_ftz(k2[9], vn2p0_2, S2[41]);
        S2[42] = fma_f32x2_rn_ftz(k2[10], vn2p0_2, S2[42]);
        S2[43] = fma_f32x2_rn_ftz(k2[11], vn2p0_2, S2[43]);
        S2[44] = fma_f32x2_rn_ftz(k2[12], vn2p0_2, S2[44]);
        S2[45] = fma_f32x2_rn_ftz(k2[13], vn2p0_2, S2[45]);
        S2[46] = fma_f32x2_rn_ftz(k2[14], vn2p0_2, S2[46]);
        S2[47] = fma_f32x2_rn_ftz(k2[15], vn2p0_2, S2[47]);
        float2 accp0_3_0 = make_float2(0.0f, 0.0f);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[48], k2[0], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[49], k2[1], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[50], k2[2], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[51], k2[3], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[52], k2[4], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[53], k2[5], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[54], k2[6], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[55], k2[7], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[56], k2[8], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[57], k2[9], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[58], k2[10], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[59], k2[11], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[60], k2[12], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[61], k2[13], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[62], k2[14], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[63], k2[15], accp0_3_0);
        float dotp0_3 = accp0_3_0.x + accp0_3_0.y;
        float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, dotp0_3, 1);
        dotp0_3 = dotp0_3 + _shfl_xor_sync_81;
        float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, dotp0_3, 2);
        dotp0_3 = dotp0_3 + _shfl_xor_sync_82;
#endif
        float vnp0_3 = (v_regs[3] - dotp0_3) * betap0;
#if __CUDA_ARCH__ == 900
        float vn2p0_3_x = vnp0_3;
        float vn2p0_3_y = vnp0_3;
        float _fma_1128;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1128)
            : "f"(k_regs[0]), "f"(vn2p0_3_x), "f"(S[96]));
        float _fma_1129;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1129)
            : "f"(k_regs[1]), "f"(vn2p0_3_y), "f"(S[97]));
        S[96] = _fma_1128;
        S[97] = _fma_1129;
        float _fma_1130;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1130)
            : "f"(k_regs[2]), "f"(vn2p0_3_x), "f"(S[98]));
        float _fma_1131;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1131)
            : "f"(k_regs[3]), "f"(vn2p0_3_y), "f"(S[99]));
        S[98] = _fma_1130;
        S[99] = _fma_1131;
        float _fma_1132;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1132)
            : "f"(k_regs[4]), "f"(vn2p0_3_x), "f"(S[100]));
        float _fma_1133;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1133)
            : "f"(k_regs[5]), "f"(vn2p0_3_y), "f"(S[101]));
        S[100] = _fma_1132;
        S[101] = _fma_1133;
        float _fma_1134;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1134)
            : "f"(k_regs[6]), "f"(vn2p0_3_x), "f"(S[102]));
        float _fma_1135;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1135)
            : "f"(k_regs[7]), "f"(vn2p0_3_y), "f"(S[103]));
        S[102] = _fma_1134;
        S[103] = _fma_1135;
        float _fma_1136;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1136)
            : "f"(k_regs[8]), "f"(vn2p0_3_x), "f"(S[104]));
        float _fma_1137;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1137)
            : "f"(k_regs[9]), "f"(vn2p0_3_y), "f"(S[105]));
        S[104] = _fma_1136;
        S[105] = _fma_1137;
        float _fma_1138;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1138)
            : "f"(k_regs[10]), "f"(vn2p0_3_x), "f"(S[106]));
        float _fma_1139;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1139)
            : "f"(k_regs[11]), "f"(vn2p0_3_y), "f"(S[107]));
        S[106] = _fma_1138;
        S[107] = _fma_1139;
        float _fma_1140;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1140)
            : "f"(k_regs[12]), "f"(vn2p0_3_x), "f"(S[108]));
        float _fma_1141;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1141)
            : "f"(k_regs[13]), "f"(vn2p0_3_y), "f"(S[109]));
        S[108] = _fma_1140;
        S[109] = _fma_1141;
        float _fma_1142;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1142)
            : "f"(k_regs[14]), "f"(vn2p0_3_x), "f"(S[110]));
        float _fma_1143;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1143)
            : "f"(k_regs[15]), "f"(vn2p0_3_y), "f"(S[111]));
        S[110] = _fma_1142;
        S[111] = _fma_1143;
        float _fma_1144;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1144)
            : "f"(k_regs[16]), "f"(vn2p0_3_x), "f"(S[112]));
        float _fma_1145;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1145)
            : "f"(k_regs[17]), "f"(vn2p0_3_y), "f"(S[113]));
        S[112] = _fma_1144;
        S[113] = _fma_1145;
        float _fma_1146;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1146)
            : "f"(k_regs[18]), "f"(vn2p0_3_x), "f"(S[114]));
        float _fma_1147;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1147)
            : "f"(k_regs[19]), "f"(vn2p0_3_y), "f"(S[115]));
        S[114] = _fma_1146;
        S[115] = _fma_1147;
        float _fma_1148;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1148)
            : "f"(k_regs[20]), "f"(vn2p0_3_x), "f"(S[116]));
        float _fma_1149;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1149)
            : "f"(k_regs[21]), "f"(vn2p0_3_y), "f"(S[117]));
        S[116] = _fma_1148;
        S[117] = _fma_1149;
        float _fma_1150;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1150)
            : "f"(k_regs[22]), "f"(vn2p0_3_x), "f"(S[118]));
        float _fma_1151;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1151)
            : "f"(k_regs[23]), "f"(vn2p0_3_y), "f"(S[119]));
        S[118] = _fma_1150;
        S[119] = _fma_1151;
        float _fma_1152;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1152)
            : "f"(k_regs[24]), "f"(vn2p0_3_x), "f"(S[120]));
        float _fma_1153;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1153)
            : "f"(k_regs[25]), "f"(vn2p0_3_y), "f"(S[121]));
        S[120] = _fma_1152;
        S[121] = _fma_1153;
        float _fma_1154;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1154)
            : "f"(k_regs[26]), "f"(vn2p0_3_x), "f"(S[122]));
        float _fma_1155;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1155)
            : "f"(k_regs[27]), "f"(vn2p0_3_y), "f"(S[123]));
        S[122] = _fma_1154;
        S[123] = _fma_1155;
        float _fma_1156;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1156)
            : "f"(k_regs[28]), "f"(vn2p0_3_x), "f"(S[124]));
        float _fma_1157;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1157)
            : "f"(k_regs[29]), "f"(vn2p0_3_y), "f"(S[125]));
        S[124] = _fma_1156;
        S[125] = _fma_1157;
        float _fma_1158;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1158)
            : "f"(k_regs[30]), "f"(vn2p0_3_x), "f"(S[126]));
        float _fma_1159;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1159)
            : "f"(k_regs[31]), "f"(vn2p0_3_y), "f"(S[127]));
        S[126] = _fma_1158;
        S[127] = _fma_1159;
#else
        float2 vn2p0_3 = make_float2(vnp0_3, vnp0_3);
        S2[48] = fma_f32x2_rn_ftz(k2[0], vn2p0_3, S2[48]);
        S2[49] = fma_f32x2_rn_ftz(k2[1], vn2p0_3, S2[49]);
        S2[50] = fma_f32x2_rn_ftz(k2[2], vn2p0_3, S2[50]);
        S2[51] = fma_f32x2_rn_ftz(k2[3], vn2p0_3, S2[51]);
        S2[52] = fma_f32x2_rn_ftz(k2[4], vn2p0_3, S2[52]);
        S2[53] = fma_f32x2_rn_ftz(k2[5], vn2p0_3, S2[53]);
        S2[54] = fma_f32x2_rn_ftz(k2[6], vn2p0_3, S2[54]);
        S2[55] = fma_f32x2_rn_ftz(k2[7], vn2p0_3, S2[55]);
        S2[56] = fma_f32x2_rn_ftz(k2[8], vn2p0_3, S2[56]);
        S2[57] = fma_f32x2_rn_ftz(k2[9], vn2p0_3, S2[57]);
        S2[58] = fma_f32x2_rn_ftz(k2[10], vn2p0_3, S2[58]);
        S2[59] = fma_f32x2_rn_ftz(k2[11], vn2p0_3, S2[59]);
        S2[60] = fma_f32x2_rn_ftz(k2[12], vn2p0_3, S2[60]);
        S2[61] = fma_f32x2_rn_ftz(k2[13], vn2p0_3, S2[61]);
        S2[62] = fma_f32x2_rn_ftz(k2[14], vn2p0_3, S2[62]);
        S2[63] = fma_f32x2_rn_ftz(k2[15], vn2p0_3, S2[63]);
#endif
        int mp1 = t_2 * 3 + 1;
        unsigned int kbasep1 = sK_addr + (unsigned int)(mp1 * 512) + lane_byte;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                     : "r"(kbasep1));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                     : "r"(kbasep1 + 64));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                     : "r"(kbasep1 + 128));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                     : "r"(kbasep1 + 192));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[16])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 3]))
                     : "r"(kbasep1 + 256));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[20])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 3]))
                     : "r"(kbasep1 + 320));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[24])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 3]))
                     : "r"(kbasep1 + 384));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[28])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 3]))
                     : "r"(kbasep1 + 448));
#if __CUDA_ARCH__ == 900
#else
        k2[0] = make_float2(k_regs[0], k_regs[1]);
        k2[1] = make_float2(k_regs[2], k_regs[3]);
        k2[2] = make_float2(k_regs[4], k_regs[5]);
        k2[3] = make_float2(k_regs[6], k_regs[7]);
        k2[4] = make_float2(k_regs[8], k_regs[9]);
        k2[5] = make_float2(k_regs[10], k_regs[11]);
        k2[6] = make_float2(k_regs[12], k_regs[13]);
        k2[7] = make_float2(k_regs[14], k_regs[15]);
        k2[8] = make_float2(k_regs[16], k_regs[17]);
        k2[9] = make_float2(k_regs[18], k_regs[19]);
        k2[10] = make_float2(k_regs[20], k_regs[21]);
        k2[11] = make_float2(k_regs[22], k_regs[23]);
        k2[12] = make_float2(k_regs[24], k_regs[25]);
        k2[13] = make_float2(k_regs[26], k_regs[27]);
        k2[14] = make_float2(k_regs[28], k_regs[29]);
        k2[15] = make_float2(k_regs[30], k_regs[31]);
#endif
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                     : "r"(sV_addr + (unsigned int)(mp1 * 256) + group_byte));
        float betap1 = sBeta[mp1];
#if __CUDA_ARCH__ == 900
        float accp1_0_0_x = 0.0f;
        float accp1_0_0_y = 0.0f;
        float accp1_0_1_x = 0.0f;
        float accp1_0_1_y = 0.0f;
        float _fma_1160;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1160)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accp1_0_0_x));
        float _fma_1161;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1161)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accp1_0_0_y));
        accp1_0_0_x = _fma_1160;
        accp1_0_0_y = _fma_1161;
        float _fma_1162;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1162)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accp1_0_1_x));
        float _fma_1163;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1163)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accp1_0_1_y));
        accp1_0_1_x = _fma_1162;
        accp1_0_1_y = _fma_1163;
        float _fma_1164;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1164)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accp1_0_0_x));
        float _fma_1165;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1165)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accp1_0_0_y));
        accp1_0_0_x = _fma_1164;
        accp1_0_0_y = _fma_1165;
        float _fma_1166;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1166)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accp1_0_1_x));
        float _fma_1167;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1167)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accp1_0_1_y));
        accp1_0_1_x = _fma_1166;
        accp1_0_1_y = _fma_1167;
        float _fma_1168;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1168)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accp1_0_0_x));
        float _fma_1169;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1169)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accp1_0_0_y));
        accp1_0_0_x = _fma_1168;
        accp1_0_0_y = _fma_1169;
        float _fma_1170;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1170)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accp1_0_1_x));
        float _fma_1171;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1171)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accp1_0_1_y));
        accp1_0_1_x = _fma_1170;
        accp1_0_1_y = _fma_1171;
        float _fma_1172;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1172)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accp1_0_0_x));
        float _fma_1173;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1173)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accp1_0_0_y));
        accp1_0_0_x = _fma_1172;
        accp1_0_0_y = _fma_1173;
        float _fma_1174;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1174)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accp1_0_1_x));
        float _fma_1175;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1175)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accp1_0_1_y));
        accp1_0_1_x = _fma_1174;
        accp1_0_1_y = _fma_1175;
        float _fma_1176;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1176)
            : "f"(S[16]), "f"(k_regs[16]), "f"(accp1_0_0_x));
        float _fma_1177;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1177)
            : "f"(S[17]), "f"(k_regs[17]), "f"(accp1_0_0_y));
        accp1_0_0_x = _fma_1176;
        accp1_0_0_y = _fma_1177;
        float _fma_1178;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1178)
            : "f"(S[18]), "f"(k_regs[18]), "f"(accp1_0_1_x));
        float _fma_1179;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1179)
            : "f"(S[19]), "f"(k_regs[19]), "f"(accp1_0_1_y));
        accp1_0_1_x = _fma_1178;
        accp1_0_1_y = _fma_1179;
        float _fma_1180;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1180)
            : "f"(S[20]), "f"(k_regs[20]), "f"(accp1_0_0_x));
        float _fma_1181;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1181)
            : "f"(S[21]), "f"(k_regs[21]), "f"(accp1_0_0_y));
        accp1_0_0_x = _fma_1180;
        accp1_0_0_y = _fma_1181;
        float _fma_1182;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1182)
            : "f"(S[22]), "f"(k_regs[22]), "f"(accp1_0_1_x));
        float _fma_1183;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1183)
            : "f"(S[23]), "f"(k_regs[23]), "f"(accp1_0_1_y));
        accp1_0_1_x = _fma_1182;
        accp1_0_1_y = _fma_1183;
        float _fma_1184;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1184)
            : "f"(S[24]), "f"(k_regs[24]), "f"(accp1_0_0_x));
        float _fma_1185;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1185)
            : "f"(S[25]), "f"(k_regs[25]), "f"(accp1_0_0_y));
        accp1_0_0_x = _fma_1184;
        accp1_0_0_y = _fma_1185;
        float _fma_1186;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1186)
            : "f"(S[26]), "f"(k_regs[26]), "f"(accp1_0_1_x));
        float _fma_1187;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1187)
            : "f"(S[27]), "f"(k_regs[27]), "f"(accp1_0_1_y));
        accp1_0_1_x = _fma_1186;
        accp1_0_1_y = _fma_1187;
        float _fma_1188;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1188)
            : "f"(S[28]), "f"(k_regs[28]), "f"(accp1_0_0_x));
        float _fma_1189;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1189)
            : "f"(S[29]), "f"(k_regs[29]), "f"(accp1_0_0_y));
        accp1_0_0_x = _fma_1188;
        accp1_0_0_y = _fma_1189;
        float _fma_1190;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1190)
            : "f"(S[30]), "f"(k_regs[30]), "f"(accp1_0_1_x));
        float _fma_1191;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1191)
            : "f"(S[31]), "f"(k_regs[31]), "f"(accp1_0_1_y));
        accp1_0_1_x = _fma_1190;
        accp1_0_1_y = _fma_1191;
        float _add_123;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_123) : "f"(accp1_0_0_x), "f"(accp1_0_1_x));
        float _add_124;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_124) : "f"(accp1_0_0_y), "f"(accp1_0_1_y));
        float dotp1_0 = _add_123 + _add_124;
        float _shfl_xor_sync_125 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 1);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_125;
        float _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 2);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_126;
#else
        float2 accp1_0_0 = make_float2(0.0f, 0.0f);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accp1_0_0);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[1], k2[1], accp1_0_0);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accp1_0_0);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[3], k2[3], accp1_0_0);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accp1_0_0);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[5], k2[5], accp1_0_0);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accp1_0_0);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[7], k2[7], accp1_0_0);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[8], k2[8], accp1_0_0);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[9], k2[9], accp1_0_0);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[10], k2[10], accp1_0_0);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[11], k2[11], accp1_0_0);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[12], k2[12], accp1_0_0);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[13], k2[13], accp1_0_0);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[14], k2[14], accp1_0_0);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[15], k2[15], accp1_0_0);
        float dotp1_0 = accp1_0_0.x + accp1_0_0.y;
        float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 1);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_83;
        float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 2);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_84;
#endif
        float vnp1_0 = (v_regs[0] - dotp1_0) * betap1;
#if __CUDA_ARCH__ == 900
        float vn2p1_0_x = vnp1_0;
        float vn2p1_0_y = vnp1_0;
        float _fma_1192;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1192)
            : "f"(k_regs[0]), "f"(vn2p1_0_x), "f"(S[0]));
        float _fma_1193;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1193)
            : "f"(k_regs[1]), "f"(vn2p1_0_y), "f"(S[1]));
        S[0] = _fma_1192;
        S[1] = _fma_1193;
        float _fma_1194;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1194)
            : "f"(k_regs[2]), "f"(vn2p1_0_x), "f"(S[2]));
        float _fma_1195;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1195)
            : "f"(k_regs[3]), "f"(vn2p1_0_y), "f"(S[3]));
        S[2] = _fma_1194;
        S[3] = _fma_1195;
        float _fma_1196;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1196)
            : "f"(k_regs[4]), "f"(vn2p1_0_x), "f"(S[4]));
        float _fma_1197;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1197)
            : "f"(k_regs[5]), "f"(vn2p1_0_y), "f"(S[5]));
        S[4] = _fma_1196;
        S[5] = _fma_1197;
        float _fma_1198;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1198)
            : "f"(k_regs[6]), "f"(vn2p1_0_x), "f"(S[6]));
        float _fma_1199;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1199)
            : "f"(k_regs[7]), "f"(vn2p1_0_y), "f"(S[7]));
        S[6] = _fma_1198;
        S[7] = _fma_1199;
        float _fma_1200;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1200)
            : "f"(k_regs[8]), "f"(vn2p1_0_x), "f"(S[8]));
        float _fma_1201;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1201)
            : "f"(k_regs[9]), "f"(vn2p1_0_y), "f"(S[9]));
        S[8] = _fma_1200;
        S[9] = _fma_1201;
        float _fma_1202;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1202)
            : "f"(k_regs[10]), "f"(vn2p1_0_x), "f"(S[10]));
        float _fma_1203;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1203)
            : "f"(k_regs[11]), "f"(vn2p1_0_y), "f"(S[11]));
        S[10] = _fma_1202;
        S[11] = _fma_1203;
        float _fma_1204;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1204)
            : "f"(k_regs[12]), "f"(vn2p1_0_x), "f"(S[12]));
        float _fma_1205;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1205)
            : "f"(k_regs[13]), "f"(vn2p1_0_y), "f"(S[13]));
        S[12] = _fma_1204;
        S[13] = _fma_1205;
        float _fma_1206;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1206)
            : "f"(k_regs[14]), "f"(vn2p1_0_x), "f"(S[14]));
        float _fma_1207;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1207)
            : "f"(k_regs[15]), "f"(vn2p1_0_y), "f"(S[15]));
        S[14] = _fma_1206;
        S[15] = _fma_1207;
        float _fma_1208;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1208)
            : "f"(k_regs[16]), "f"(vn2p1_0_x), "f"(S[16]));
        float _fma_1209;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1209)
            : "f"(k_regs[17]), "f"(vn2p1_0_y), "f"(S[17]));
        S[16] = _fma_1208;
        S[17] = _fma_1209;
        float _fma_1210;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1210)
            : "f"(k_regs[18]), "f"(vn2p1_0_x), "f"(S[18]));
        float _fma_1211;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1211)
            : "f"(k_regs[19]), "f"(vn2p1_0_y), "f"(S[19]));
        S[18] = _fma_1210;
        S[19] = _fma_1211;
        float _fma_1212;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1212)
            : "f"(k_regs[20]), "f"(vn2p1_0_x), "f"(S[20]));
        float _fma_1213;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1213)
            : "f"(k_regs[21]), "f"(vn2p1_0_y), "f"(S[21]));
        S[20] = _fma_1212;
        S[21] = _fma_1213;
        float _fma_1214;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1214)
            : "f"(k_regs[22]), "f"(vn2p1_0_x), "f"(S[22]));
        float _fma_1215;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1215)
            : "f"(k_regs[23]), "f"(vn2p1_0_y), "f"(S[23]));
        S[22] = _fma_1214;
        S[23] = _fma_1215;
        float _fma_1216;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1216)
            : "f"(k_regs[24]), "f"(vn2p1_0_x), "f"(S[24]));
        float _fma_1217;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1217)
            : "f"(k_regs[25]), "f"(vn2p1_0_y), "f"(S[25]));
        S[24] = _fma_1216;
        S[25] = _fma_1217;
        float _fma_1218;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1218)
            : "f"(k_regs[26]), "f"(vn2p1_0_x), "f"(S[26]));
        float _fma_1219;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1219)
            : "f"(k_regs[27]), "f"(vn2p1_0_y), "f"(S[27]));
        S[26] = _fma_1218;
        S[27] = _fma_1219;
        float _fma_1220;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1220)
            : "f"(k_regs[28]), "f"(vn2p1_0_x), "f"(S[28]));
        float _fma_1221;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1221)
            : "f"(k_regs[29]), "f"(vn2p1_0_y), "f"(S[29]));
        S[28] = _fma_1220;
        S[29] = _fma_1221;
        float _fma_1222;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1222)
            : "f"(k_regs[30]), "f"(vn2p1_0_x), "f"(S[30]));
        float _fma_1223;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1223)
            : "f"(k_regs[31]), "f"(vn2p1_0_y), "f"(S[31]));
        S[30] = _fma_1222;
        S[31] = _fma_1223;
        float accp1_1_0_x = 0.0f;
        float accp1_1_0_y = 0.0f;
        float accp1_1_1_x = 0.0f;
        float accp1_1_1_y = 0.0f;
        float _fma_1224;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1224)
            : "f"(S[32]), "f"(k_regs[0]), "f"(accp1_1_0_x));
        float _fma_1225;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1225)
            : "f"(S[33]), "f"(k_regs[1]), "f"(accp1_1_0_y));
        accp1_1_0_x = _fma_1224;
        accp1_1_0_y = _fma_1225;
        float _fma_1226;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1226)
            : "f"(S[34]), "f"(k_regs[2]), "f"(accp1_1_1_x));
        float _fma_1227;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1227)
            : "f"(S[35]), "f"(k_regs[3]), "f"(accp1_1_1_y));
        accp1_1_1_x = _fma_1226;
        accp1_1_1_y = _fma_1227;
        float _fma_1228;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1228)
            : "f"(S[36]), "f"(k_regs[4]), "f"(accp1_1_0_x));
        float _fma_1229;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1229)
            : "f"(S[37]), "f"(k_regs[5]), "f"(accp1_1_0_y));
        accp1_1_0_x = _fma_1228;
        accp1_1_0_y = _fma_1229;
        float _fma_1230;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1230)
            : "f"(S[38]), "f"(k_regs[6]), "f"(accp1_1_1_x));
        float _fma_1231;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1231)
            : "f"(S[39]), "f"(k_regs[7]), "f"(accp1_1_1_y));
        accp1_1_1_x = _fma_1230;
        accp1_1_1_y = _fma_1231;
        float _fma_1232;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1232)
            : "f"(S[40]), "f"(k_regs[8]), "f"(accp1_1_0_x));
        float _fma_1233;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1233)
            : "f"(S[41]), "f"(k_regs[9]), "f"(accp1_1_0_y));
        accp1_1_0_x = _fma_1232;
        accp1_1_0_y = _fma_1233;
        float _fma_1234;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1234)
            : "f"(S[42]), "f"(k_regs[10]), "f"(accp1_1_1_x));
        float _fma_1235;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1235)
            : "f"(S[43]), "f"(k_regs[11]), "f"(accp1_1_1_y));
        accp1_1_1_x = _fma_1234;
        accp1_1_1_y = _fma_1235;
        float _fma_1236;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1236)
            : "f"(S[44]), "f"(k_regs[12]), "f"(accp1_1_0_x));
        float _fma_1237;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1237)
            : "f"(S[45]), "f"(k_regs[13]), "f"(accp1_1_0_y));
        accp1_1_0_x = _fma_1236;
        accp1_1_0_y = _fma_1237;
        float _fma_1238;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1238)
            : "f"(S[46]), "f"(k_regs[14]), "f"(accp1_1_1_x));
        float _fma_1239;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1239)
            : "f"(S[47]), "f"(k_regs[15]), "f"(accp1_1_1_y));
        accp1_1_1_x = _fma_1238;
        accp1_1_1_y = _fma_1239;
        float _fma_1240;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1240)
            : "f"(S[48]), "f"(k_regs[16]), "f"(accp1_1_0_x));
        float _fma_1241;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1241)
            : "f"(S[49]), "f"(k_regs[17]), "f"(accp1_1_0_y));
        accp1_1_0_x = _fma_1240;
        accp1_1_0_y = _fma_1241;
        float _fma_1242;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1242)
            : "f"(S[50]), "f"(k_regs[18]), "f"(accp1_1_1_x));
        float _fma_1243;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1243)
            : "f"(S[51]), "f"(k_regs[19]), "f"(accp1_1_1_y));
        accp1_1_1_x = _fma_1242;
        accp1_1_1_y = _fma_1243;
        float _fma_1244;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1244)
            : "f"(S[52]), "f"(k_regs[20]), "f"(accp1_1_0_x));
        float _fma_1245;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1245)
            : "f"(S[53]), "f"(k_regs[21]), "f"(accp1_1_0_y));
        accp1_1_0_x = _fma_1244;
        accp1_1_0_y = _fma_1245;
        float _fma_1246;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1246)
            : "f"(S[54]), "f"(k_regs[22]), "f"(accp1_1_1_x));
        float _fma_1247;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1247)
            : "f"(S[55]), "f"(k_regs[23]), "f"(accp1_1_1_y));
        accp1_1_1_x = _fma_1246;
        accp1_1_1_y = _fma_1247;
        float _fma_1248;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1248)
            : "f"(S[56]), "f"(k_regs[24]), "f"(accp1_1_0_x));
        float _fma_1249;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1249)
            : "f"(S[57]), "f"(k_regs[25]), "f"(accp1_1_0_y));
        accp1_1_0_x = _fma_1248;
        accp1_1_0_y = _fma_1249;
        float _fma_1250;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1250)
            : "f"(S[58]), "f"(k_regs[26]), "f"(accp1_1_1_x));
        float _fma_1251;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1251)
            : "f"(S[59]), "f"(k_regs[27]), "f"(accp1_1_1_y));
        accp1_1_1_x = _fma_1250;
        accp1_1_1_y = _fma_1251;
        float _fma_1252;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1252)
            : "f"(S[60]), "f"(k_regs[28]), "f"(accp1_1_0_x));
        float _fma_1253;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1253)
            : "f"(S[61]), "f"(k_regs[29]), "f"(accp1_1_0_y));
        accp1_1_0_x = _fma_1252;
        accp1_1_0_y = _fma_1253;
        float _fma_1254;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1254)
            : "f"(S[62]), "f"(k_regs[30]), "f"(accp1_1_1_x));
        float _fma_1255;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1255)
            : "f"(S[63]), "f"(k_regs[31]), "f"(accp1_1_1_y));
        accp1_1_1_x = _fma_1254;
        accp1_1_1_y = _fma_1255;
        float _add_127;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(accp1_1_0_x), "f"(accp1_1_1_x));
        float _add_128;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(accp1_1_0_y), "f"(accp1_1_1_y));
        float dotp1_1 = _add_127 + _add_128;
        float _shfl_xor_sync_129 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 1);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_129;
        float _shfl_xor_sync_130 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 2);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_130;
#else
        float2 vn2p1_0 = make_float2(vnp1_0, vnp1_0);
        S2[0] = fma_f32x2_rn_ftz(k2[0], vn2p1_0, S2[0]);
        S2[1] = fma_f32x2_rn_ftz(k2[1], vn2p1_0, S2[1]);
        S2[2] = fma_f32x2_rn_ftz(k2[2], vn2p1_0, S2[2]);
        S2[3] = fma_f32x2_rn_ftz(k2[3], vn2p1_0, S2[3]);
        S2[4] = fma_f32x2_rn_ftz(k2[4], vn2p1_0, S2[4]);
        S2[5] = fma_f32x2_rn_ftz(k2[5], vn2p1_0, S2[5]);
        S2[6] = fma_f32x2_rn_ftz(k2[6], vn2p1_0, S2[6]);
        S2[7] = fma_f32x2_rn_ftz(k2[7], vn2p1_0, S2[7]);
        S2[8] = fma_f32x2_rn_ftz(k2[8], vn2p1_0, S2[8]);
        S2[9] = fma_f32x2_rn_ftz(k2[9], vn2p1_0, S2[9]);
        S2[10] = fma_f32x2_rn_ftz(k2[10], vn2p1_0, S2[10]);
        S2[11] = fma_f32x2_rn_ftz(k2[11], vn2p1_0, S2[11]);
        S2[12] = fma_f32x2_rn_ftz(k2[12], vn2p1_0, S2[12]);
        S2[13] = fma_f32x2_rn_ftz(k2[13], vn2p1_0, S2[13]);
        S2[14] = fma_f32x2_rn_ftz(k2[14], vn2p1_0, S2[14]);
        S2[15] = fma_f32x2_rn_ftz(k2[15], vn2p1_0, S2[15]);
        float2 accp1_1_0 = make_float2(0.0f, 0.0f);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[24], k2[8], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[25], k2[9], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[26], k2[10], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[27], k2[11], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[28], k2[12], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[29], k2[13], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[30], k2[14], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[31], k2[15], accp1_1_0);
        float dotp1_1 = accp1_1_0.x + accp1_1_0.y;
        float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 1);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_85;
        float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 2);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_86;
#endif
        float vnp1_1 = (v_regs[1] - dotp1_1) * betap1;
#if __CUDA_ARCH__ == 900
        float vn2p1_1_x = vnp1_1;
        float vn2p1_1_y = vnp1_1;
        float _fma_1256;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1256)
            : "f"(k_regs[0]), "f"(vn2p1_1_x), "f"(S[32]));
        float _fma_1257;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1257)
            : "f"(k_regs[1]), "f"(vn2p1_1_y), "f"(S[33]));
        S[32] = _fma_1256;
        S[33] = _fma_1257;
        float _fma_1258;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1258)
            : "f"(k_regs[2]), "f"(vn2p1_1_x), "f"(S[34]));
        float _fma_1259;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1259)
            : "f"(k_regs[3]), "f"(vn2p1_1_y), "f"(S[35]));
        S[34] = _fma_1258;
        S[35] = _fma_1259;
        float _fma_1260;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1260)
            : "f"(k_regs[4]), "f"(vn2p1_1_x), "f"(S[36]));
        float _fma_1261;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1261)
            : "f"(k_regs[5]), "f"(vn2p1_1_y), "f"(S[37]));
        S[36] = _fma_1260;
        S[37] = _fma_1261;
        float _fma_1262;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1262)
            : "f"(k_regs[6]), "f"(vn2p1_1_x), "f"(S[38]));
        float _fma_1263;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1263)
            : "f"(k_regs[7]), "f"(vn2p1_1_y), "f"(S[39]));
        S[38] = _fma_1262;
        S[39] = _fma_1263;
        float _fma_1264;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1264)
            : "f"(k_regs[8]), "f"(vn2p1_1_x), "f"(S[40]));
        float _fma_1265;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1265)
            : "f"(k_regs[9]), "f"(vn2p1_1_y), "f"(S[41]));
        S[40] = _fma_1264;
        S[41] = _fma_1265;
        float _fma_1266;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1266)
            : "f"(k_regs[10]), "f"(vn2p1_1_x), "f"(S[42]));
        float _fma_1267;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1267)
            : "f"(k_regs[11]), "f"(vn2p1_1_y), "f"(S[43]));
        S[42] = _fma_1266;
        S[43] = _fma_1267;
        float _fma_1268;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1268)
            : "f"(k_regs[12]), "f"(vn2p1_1_x), "f"(S[44]));
        float _fma_1269;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1269)
            : "f"(k_regs[13]), "f"(vn2p1_1_y), "f"(S[45]));
        S[44] = _fma_1268;
        S[45] = _fma_1269;
        float _fma_1270;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1270)
            : "f"(k_regs[14]), "f"(vn2p1_1_x), "f"(S[46]));
        float _fma_1271;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1271)
            : "f"(k_regs[15]), "f"(vn2p1_1_y), "f"(S[47]));
        S[46] = _fma_1270;
        S[47] = _fma_1271;
        float _fma_1272;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1272)
            : "f"(k_regs[16]), "f"(vn2p1_1_x), "f"(S[48]));
        float _fma_1273;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1273)
            : "f"(k_regs[17]), "f"(vn2p1_1_y), "f"(S[49]));
        S[48] = _fma_1272;
        S[49] = _fma_1273;
        float _fma_1274;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1274)
            : "f"(k_regs[18]), "f"(vn2p1_1_x), "f"(S[50]));
        float _fma_1275;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1275)
            : "f"(k_regs[19]), "f"(vn2p1_1_y), "f"(S[51]));
        S[50] = _fma_1274;
        S[51] = _fma_1275;
        float _fma_1276;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1276)
            : "f"(k_regs[20]), "f"(vn2p1_1_x), "f"(S[52]));
        float _fma_1277;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1277)
            : "f"(k_regs[21]), "f"(vn2p1_1_y), "f"(S[53]));
        S[52] = _fma_1276;
        S[53] = _fma_1277;
        float _fma_1278;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1278)
            : "f"(k_regs[22]), "f"(vn2p1_1_x), "f"(S[54]));
        float _fma_1279;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1279)
            : "f"(k_regs[23]), "f"(vn2p1_1_y), "f"(S[55]));
        S[54] = _fma_1278;
        S[55] = _fma_1279;
        float _fma_1280;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1280)
            : "f"(k_regs[24]), "f"(vn2p1_1_x), "f"(S[56]));
        float _fma_1281;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1281)
            : "f"(k_regs[25]), "f"(vn2p1_1_y), "f"(S[57]));
        S[56] = _fma_1280;
        S[57] = _fma_1281;
        float _fma_1282;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1282)
            : "f"(k_regs[26]), "f"(vn2p1_1_x), "f"(S[58]));
        float _fma_1283;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1283)
            : "f"(k_regs[27]), "f"(vn2p1_1_y), "f"(S[59]));
        S[58] = _fma_1282;
        S[59] = _fma_1283;
        float _fma_1284;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1284)
            : "f"(k_regs[28]), "f"(vn2p1_1_x), "f"(S[60]));
        float _fma_1285;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1285)
            : "f"(k_regs[29]), "f"(vn2p1_1_y), "f"(S[61]));
        S[60] = _fma_1284;
        S[61] = _fma_1285;
        float _fma_1286;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1286)
            : "f"(k_regs[30]), "f"(vn2p1_1_x), "f"(S[62]));
        float _fma_1287;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1287)
            : "f"(k_regs[31]), "f"(vn2p1_1_y), "f"(S[63]));
        S[62] = _fma_1286;
        S[63] = _fma_1287;
        float accp1_2_0_x = 0.0f;
        float accp1_2_0_y = 0.0f;
        float accp1_2_1_x = 0.0f;
        float accp1_2_1_y = 0.0f;
        float _fma_1288;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1288)
            : "f"(S[64]), "f"(k_regs[0]), "f"(accp1_2_0_x));
        float _fma_1289;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1289)
            : "f"(S[65]), "f"(k_regs[1]), "f"(accp1_2_0_y));
        accp1_2_0_x = _fma_1288;
        accp1_2_0_y = _fma_1289;
        float _fma_1290;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1290)
            : "f"(S[66]), "f"(k_regs[2]), "f"(accp1_2_1_x));
        float _fma_1291;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1291)
            : "f"(S[67]), "f"(k_regs[3]), "f"(accp1_2_1_y));
        accp1_2_1_x = _fma_1290;
        accp1_2_1_y = _fma_1291;
        float _fma_1292;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1292)
            : "f"(S[68]), "f"(k_regs[4]), "f"(accp1_2_0_x));
        float _fma_1293;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1293)
            : "f"(S[69]), "f"(k_regs[5]), "f"(accp1_2_0_y));
        accp1_2_0_x = _fma_1292;
        accp1_2_0_y = _fma_1293;
        float _fma_1294;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1294)
            : "f"(S[70]), "f"(k_regs[6]), "f"(accp1_2_1_x));
        float _fma_1295;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1295)
            : "f"(S[71]), "f"(k_regs[7]), "f"(accp1_2_1_y));
        accp1_2_1_x = _fma_1294;
        accp1_2_1_y = _fma_1295;
        float _fma_1296;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1296)
            : "f"(S[72]), "f"(k_regs[8]), "f"(accp1_2_0_x));
        float _fma_1297;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1297)
            : "f"(S[73]), "f"(k_regs[9]), "f"(accp1_2_0_y));
        accp1_2_0_x = _fma_1296;
        accp1_2_0_y = _fma_1297;
        float _fma_1298;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1298)
            : "f"(S[74]), "f"(k_regs[10]), "f"(accp1_2_1_x));
        float _fma_1299;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1299)
            : "f"(S[75]), "f"(k_regs[11]), "f"(accp1_2_1_y));
        accp1_2_1_x = _fma_1298;
        accp1_2_1_y = _fma_1299;
        float _fma_1300;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1300)
            : "f"(S[76]), "f"(k_regs[12]), "f"(accp1_2_0_x));
        float _fma_1301;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1301)
            : "f"(S[77]), "f"(k_regs[13]), "f"(accp1_2_0_y));
        accp1_2_0_x = _fma_1300;
        accp1_2_0_y = _fma_1301;
        float _fma_1302;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1302)
            : "f"(S[78]), "f"(k_regs[14]), "f"(accp1_2_1_x));
        float _fma_1303;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1303)
            : "f"(S[79]), "f"(k_regs[15]), "f"(accp1_2_1_y));
        accp1_2_1_x = _fma_1302;
        accp1_2_1_y = _fma_1303;
        float _fma_1304;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1304)
            : "f"(S[80]), "f"(k_regs[16]), "f"(accp1_2_0_x));
        float _fma_1305;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1305)
            : "f"(S[81]), "f"(k_regs[17]), "f"(accp1_2_0_y));
        accp1_2_0_x = _fma_1304;
        accp1_2_0_y = _fma_1305;
        float _fma_1306;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1306)
            : "f"(S[82]), "f"(k_regs[18]), "f"(accp1_2_1_x));
        float _fma_1307;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1307)
            : "f"(S[83]), "f"(k_regs[19]), "f"(accp1_2_1_y));
        accp1_2_1_x = _fma_1306;
        accp1_2_1_y = _fma_1307;
        float _fma_1308;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1308)
            : "f"(S[84]), "f"(k_regs[20]), "f"(accp1_2_0_x));
        float _fma_1309;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1309)
            : "f"(S[85]), "f"(k_regs[21]), "f"(accp1_2_0_y));
        accp1_2_0_x = _fma_1308;
        accp1_2_0_y = _fma_1309;
        float _fma_1310;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1310)
            : "f"(S[86]), "f"(k_regs[22]), "f"(accp1_2_1_x));
        float _fma_1311;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1311)
            : "f"(S[87]), "f"(k_regs[23]), "f"(accp1_2_1_y));
        accp1_2_1_x = _fma_1310;
        accp1_2_1_y = _fma_1311;
        float _fma_1312;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1312)
            : "f"(S[88]), "f"(k_regs[24]), "f"(accp1_2_0_x));
        float _fma_1313;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1313)
            : "f"(S[89]), "f"(k_regs[25]), "f"(accp1_2_0_y));
        accp1_2_0_x = _fma_1312;
        accp1_2_0_y = _fma_1313;
        float _fma_1314;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1314)
            : "f"(S[90]), "f"(k_regs[26]), "f"(accp1_2_1_x));
        float _fma_1315;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1315)
            : "f"(S[91]), "f"(k_regs[27]), "f"(accp1_2_1_y));
        accp1_2_1_x = _fma_1314;
        accp1_2_1_y = _fma_1315;
        float _fma_1316;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1316)
            : "f"(S[92]), "f"(k_regs[28]), "f"(accp1_2_0_x));
        float _fma_1317;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1317)
            : "f"(S[93]), "f"(k_regs[29]), "f"(accp1_2_0_y));
        accp1_2_0_x = _fma_1316;
        accp1_2_0_y = _fma_1317;
        float _fma_1318;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1318)
            : "f"(S[94]), "f"(k_regs[30]), "f"(accp1_2_1_x));
        float _fma_1319;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1319)
            : "f"(S[95]), "f"(k_regs[31]), "f"(accp1_2_1_y));
        accp1_2_1_x = _fma_1318;
        accp1_2_1_y = _fma_1319;
        float _add_131;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_131) : "f"(accp1_2_0_x), "f"(accp1_2_1_x));
        float _add_132;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(accp1_2_0_y), "f"(accp1_2_1_y));
        float dotp1_2 = _add_131 + _add_132;
        float _shfl_xor_sync_133 = __shfl_xor_sync(0xFFFFFFFF, dotp1_2, 1);
        dotp1_2 = dotp1_2 + _shfl_xor_sync_133;
        float _shfl_xor_sync_134 = __shfl_xor_sync(0xFFFFFFFF, dotp1_2, 2);
        dotp1_2 = dotp1_2 + _shfl_xor_sync_134;
#else
        float2 vn2p1_1 = make_float2(vnp1_1, vnp1_1);
        S2[16] = fma_f32x2_rn_ftz(k2[0], vn2p1_1, S2[16]);
        S2[17] = fma_f32x2_rn_ftz(k2[1], vn2p1_1, S2[17]);
        S2[18] = fma_f32x2_rn_ftz(k2[2], vn2p1_1, S2[18]);
        S2[19] = fma_f32x2_rn_ftz(k2[3], vn2p1_1, S2[19]);
        S2[20] = fma_f32x2_rn_ftz(k2[4], vn2p1_1, S2[20]);
        S2[21] = fma_f32x2_rn_ftz(k2[5], vn2p1_1, S2[21]);
        S2[22] = fma_f32x2_rn_ftz(k2[6], vn2p1_1, S2[22]);
        S2[23] = fma_f32x2_rn_ftz(k2[7], vn2p1_1, S2[23]);
        S2[24] = fma_f32x2_rn_ftz(k2[8], vn2p1_1, S2[24]);
        S2[25] = fma_f32x2_rn_ftz(k2[9], vn2p1_1, S2[25]);
        S2[26] = fma_f32x2_rn_ftz(k2[10], vn2p1_1, S2[26]);
        S2[27] = fma_f32x2_rn_ftz(k2[11], vn2p1_1, S2[27]);
        S2[28] = fma_f32x2_rn_ftz(k2[12], vn2p1_1, S2[28]);
        S2[29] = fma_f32x2_rn_ftz(k2[13], vn2p1_1, S2[29]);
        S2[30] = fma_f32x2_rn_ftz(k2[14], vn2p1_1, S2[30]);
        S2[31] = fma_f32x2_rn_ftz(k2[15], vn2p1_1, S2[31]);
        float2 accp1_2_0 = make_float2(0.0f, 0.0f);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[32], k2[0], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[33], k2[1], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[34], k2[2], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[35], k2[3], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[36], k2[4], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[37], k2[5], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[38], k2[6], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[39], k2[7], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[40], k2[8], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[41], k2[9], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[42], k2[10], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[43], k2[11], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[44], k2[12], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[45], k2[13], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[46], k2[14], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[47], k2[15], accp1_2_0);
        float dotp1_2 = accp1_2_0.x + accp1_2_0.y;
        float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, dotp1_2, 1);
        dotp1_2 = dotp1_2 + _shfl_xor_sync_87;
        float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, dotp1_2, 2);
        dotp1_2 = dotp1_2 + _shfl_xor_sync_88;
#endif
        float vnp1_2 = (v_regs[2] - dotp1_2) * betap1;
#if __CUDA_ARCH__ == 900
        float vn2p1_2_x = vnp1_2;
        float vn2p1_2_y = vnp1_2;
        float _fma_1320;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1320)
            : "f"(k_regs[0]), "f"(vn2p1_2_x), "f"(S[64]));
        float _fma_1321;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1321)
            : "f"(k_regs[1]), "f"(vn2p1_2_y), "f"(S[65]));
        S[64] = _fma_1320;
        S[65] = _fma_1321;
        float _fma_1322;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1322)
            : "f"(k_regs[2]), "f"(vn2p1_2_x), "f"(S[66]));
        float _fma_1323;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1323)
            : "f"(k_regs[3]), "f"(vn2p1_2_y), "f"(S[67]));
        S[66] = _fma_1322;
        S[67] = _fma_1323;
        float _fma_1324;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1324)
            : "f"(k_regs[4]), "f"(vn2p1_2_x), "f"(S[68]));
        float _fma_1325;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1325)
            : "f"(k_regs[5]), "f"(vn2p1_2_y), "f"(S[69]));
        S[68] = _fma_1324;
        S[69] = _fma_1325;
        float _fma_1326;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1326)
            : "f"(k_regs[6]), "f"(vn2p1_2_x), "f"(S[70]));
        float _fma_1327;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1327)
            : "f"(k_regs[7]), "f"(vn2p1_2_y), "f"(S[71]));
        S[70] = _fma_1326;
        S[71] = _fma_1327;
        float _fma_1328;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1328)
            : "f"(k_regs[8]), "f"(vn2p1_2_x), "f"(S[72]));
        float _fma_1329;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1329)
            : "f"(k_regs[9]), "f"(vn2p1_2_y), "f"(S[73]));
        S[72] = _fma_1328;
        S[73] = _fma_1329;
        float _fma_1330;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1330)
            : "f"(k_regs[10]), "f"(vn2p1_2_x), "f"(S[74]));
        float _fma_1331;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1331)
            : "f"(k_regs[11]), "f"(vn2p1_2_y), "f"(S[75]));
        S[74] = _fma_1330;
        S[75] = _fma_1331;
        float _fma_1332;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1332)
            : "f"(k_regs[12]), "f"(vn2p1_2_x), "f"(S[76]));
        float _fma_1333;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1333)
            : "f"(k_regs[13]), "f"(vn2p1_2_y), "f"(S[77]));
        S[76] = _fma_1332;
        S[77] = _fma_1333;
        float _fma_1334;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1334)
            : "f"(k_regs[14]), "f"(vn2p1_2_x), "f"(S[78]));
        float _fma_1335;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1335)
            : "f"(k_regs[15]), "f"(vn2p1_2_y), "f"(S[79]));
        S[78] = _fma_1334;
        S[79] = _fma_1335;
        float _fma_1336;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1336)
            : "f"(k_regs[16]), "f"(vn2p1_2_x), "f"(S[80]));
        float _fma_1337;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1337)
            : "f"(k_regs[17]), "f"(vn2p1_2_y), "f"(S[81]));
        S[80] = _fma_1336;
        S[81] = _fma_1337;
        float _fma_1338;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1338)
            : "f"(k_regs[18]), "f"(vn2p1_2_x), "f"(S[82]));
        float _fma_1339;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1339)
            : "f"(k_regs[19]), "f"(vn2p1_2_y), "f"(S[83]));
        S[82] = _fma_1338;
        S[83] = _fma_1339;
        float _fma_1340;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1340)
            : "f"(k_regs[20]), "f"(vn2p1_2_x), "f"(S[84]));
        float _fma_1341;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1341)
            : "f"(k_regs[21]), "f"(vn2p1_2_y), "f"(S[85]));
        S[84] = _fma_1340;
        S[85] = _fma_1341;
        float _fma_1342;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1342)
            : "f"(k_regs[22]), "f"(vn2p1_2_x), "f"(S[86]));
        float _fma_1343;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1343)
            : "f"(k_regs[23]), "f"(vn2p1_2_y), "f"(S[87]));
        S[86] = _fma_1342;
        S[87] = _fma_1343;
        float _fma_1344;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1344)
            : "f"(k_regs[24]), "f"(vn2p1_2_x), "f"(S[88]));
        float _fma_1345;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1345)
            : "f"(k_regs[25]), "f"(vn2p1_2_y), "f"(S[89]));
        S[88] = _fma_1344;
        S[89] = _fma_1345;
        float _fma_1346;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1346)
            : "f"(k_regs[26]), "f"(vn2p1_2_x), "f"(S[90]));
        float _fma_1347;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1347)
            : "f"(k_regs[27]), "f"(vn2p1_2_y), "f"(S[91]));
        S[90] = _fma_1346;
        S[91] = _fma_1347;
        float _fma_1348;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1348)
            : "f"(k_regs[28]), "f"(vn2p1_2_x), "f"(S[92]));
        float _fma_1349;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1349)
            : "f"(k_regs[29]), "f"(vn2p1_2_y), "f"(S[93]));
        S[92] = _fma_1348;
        S[93] = _fma_1349;
        float _fma_1350;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1350)
            : "f"(k_regs[30]), "f"(vn2p1_2_x), "f"(S[94]));
        float _fma_1351;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1351)
            : "f"(k_regs[31]), "f"(vn2p1_2_y), "f"(S[95]));
        S[94] = _fma_1350;
        S[95] = _fma_1351;
        float accp1_3_0_x = 0.0f;
        float accp1_3_0_y = 0.0f;
        float accp1_3_1_x = 0.0f;
        float accp1_3_1_y = 0.0f;
        float _fma_1352;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1352)
            : "f"(S[96]), "f"(k_regs[0]), "f"(accp1_3_0_x));
        float _fma_1353;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1353)
            : "f"(S[97]), "f"(k_regs[1]), "f"(accp1_3_0_y));
        accp1_3_0_x = _fma_1352;
        accp1_3_0_y = _fma_1353;
        float _fma_1354;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1354)
            : "f"(S[98]), "f"(k_regs[2]), "f"(accp1_3_1_x));
        float _fma_1355;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1355)
            : "f"(S[99]), "f"(k_regs[3]), "f"(accp1_3_1_y));
        accp1_3_1_x = _fma_1354;
        accp1_3_1_y = _fma_1355;
        float _fma_1356;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1356)
            : "f"(S[100]), "f"(k_regs[4]), "f"(accp1_3_0_x));
        float _fma_1357;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1357)
            : "f"(S[101]), "f"(k_regs[5]), "f"(accp1_3_0_y));
        accp1_3_0_x = _fma_1356;
        accp1_3_0_y = _fma_1357;
        float _fma_1358;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1358)
            : "f"(S[102]), "f"(k_regs[6]), "f"(accp1_3_1_x));
        float _fma_1359;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1359)
            : "f"(S[103]), "f"(k_regs[7]), "f"(accp1_3_1_y));
        accp1_3_1_x = _fma_1358;
        accp1_3_1_y = _fma_1359;
        float _fma_1360;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1360)
            : "f"(S[104]), "f"(k_regs[8]), "f"(accp1_3_0_x));
        float _fma_1361;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1361)
            : "f"(S[105]), "f"(k_regs[9]), "f"(accp1_3_0_y));
        accp1_3_0_x = _fma_1360;
        accp1_3_0_y = _fma_1361;
        float _fma_1362;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1362)
            : "f"(S[106]), "f"(k_regs[10]), "f"(accp1_3_1_x));
        float _fma_1363;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1363)
            : "f"(S[107]), "f"(k_regs[11]), "f"(accp1_3_1_y));
        accp1_3_1_x = _fma_1362;
        accp1_3_1_y = _fma_1363;
        float _fma_1364;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1364)
            : "f"(S[108]), "f"(k_regs[12]), "f"(accp1_3_0_x));
        float _fma_1365;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1365)
            : "f"(S[109]), "f"(k_regs[13]), "f"(accp1_3_0_y));
        accp1_3_0_x = _fma_1364;
        accp1_3_0_y = _fma_1365;
        float _fma_1366;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1366)
            : "f"(S[110]), "f"(k_regs[14]), "f"(accp1_3_1_x));
        float _fma_1367;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1367)
            : "f"(S[111]), "f"(k_regs[15]), "f"(accp1_3_1_y));
        accp1_3_1_x = _fma_1366;
        accp1_3_1_y = _fma_1367;
        float _fma_1368;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1368)
            : "f"(S[112]), "f"(k_regs[16]), "f"(accp1_3_0_x));
        float _fma_1369;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1369)
            : "f"(S[113]), "f"(k_regs[17]), "f"(accp1_3_0_y));
        accp1_3_0_x = _fma_1368;
        accp1_3_0_y = _fma_1369;
        float _fma_1370;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1370)
            : "f"(S[114]), "f"(k_regs[18]), "f"(accp1_3_1_x));
        float _fma_1371;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1371)
            : "f"(S[115]), "f"(k_regs[19]), "f"(accp1_3_1_y));
        accp1_3_1_x = _fma_1370;
        accp1_3_1_y = _fma_1371;
        float _fma_1372;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1372)
            : "f"(S[116]), "f"(k_regs[20]), "f"(accp1_3_0_x));
        float _fma_1373;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1373)
            : "f"(S[117]), "f"(k_regs[21]), "f"(accp1_3_0_y));
        accp1_3_0_x = _fma_1372;
        accp1_3_0_y = _fma_1373;
        float _fma_1374;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1374)
            : "f"(S[118]), "f"(k_regs[22]), "f"(accp1_3_1_x));
        float _fma_1375;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1375)
            : "f"(S[119]), "f"(k_regs[23]), "f"(accp1_3_1_y));
        accp1_3_1_x = _fma_1374;
        accp1_3_1_y = _fma_1375;
        float _fma_1376;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1376)
            : "f"(S[120]), "f"(k_regs[24]), "f"(accp1_3_0_x));
        float _fma_1377;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1377)
            : "f"(S[121]), "f"(k_regs[25]), "f"(accp1_3_0_y));
        accp1_3_0_x = _fma_1376;
        accp1_3_0_y = _fma_1377;
        float _fma_1378;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1378)
            : "f"(S[122]), "f"(k_regs[26]), "f"(accp1_3_1_x));
        float _fma_1379;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1379)
            : "f"(S[123]), "f"(k_regs[27]), "f"(accp1_3_1_y));
        accp1_3_1_x = _fma_1378;
        accp1_3_1_y = _fma_1379;
        float _fma_1380;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1380)
            : "f"(S[124]), "f"(k_regs[28]), "f"(accp1_3_0_x));
        float _fma_1381;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1381)
            : "f"(S[125]), "f"(k_regs[29]), "f"(accp1_3_0_y));
        accp1_3_0_x = _fma_1380;
        accp1_3_0_y = _fma_1381;
        float _fma_1382;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1382)
            : "f"(S[126]), "f"(k_regs[30]), "f"(accp1_3_1_x));
        float _fma_1383;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1383)
            : "f"(S[127]), "f"(k_regs[31]), "f"(accp1_3_1_y));
        accp1_3_1_x = _fma_1382;
        accp1_3_1_y = _fma_1383;
        float _add_135;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_135) : "f"(accp1_3_0_x), "f"(accp1_3_1_x));
        float _add_136;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_136) : "f"(accp1_3_0_y), "f"(accp1_3_1_y));
        float dotp1_3 = _add_135 + _add_136;
        float _shfl_xor_sync_137 = __shfl_xor_sync(0xFFFFFFFF, dotp1_3, 1);
        dotp1_3 = dotp1_3 + _shfl_xor_sync_137;
        float _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, dotp1_3, 2);
        dotp1_3 = dotp1_3 + _shfl_xor_sync_138;
#else
        float2 vn2p1_2 = make_float2(vnp1_2, vnp1_2);
        S2[32] = fma_f32x2_rn_ftz(k2[0], vn2p1_2, S2[32]);
        S2[33] = fma_f32x2_rn_ftz(k2[1], vn2p1_2, S2[33]);
        S2[34] = fma_f32x2_rn_ftz(k2[2], vn2p1_2, S2[34]);
        S2[35] = fma_f32x2_rn_ftz(k2[3], vn2p1_2, S2[35]);
        S2[36] = fma_f32x2_rn_ftz(k2[4], vn2p1_2, S2[36]);
        S2[37] = fma_f32x2_rn_ftz(k2[5], vn2p1_2, S2[37]);
        S2[38] = fma_f32x2_rn_ftz(k2[6], vn2p1_2, S2[38]);
        S2[39] = fma_f32x2_rn_ftz(k2[7], vn2p1_2, S2[39]);
        S2[40] = fma_f32x2_rn_ftz(k2[8], vn2p1_2, S2[40]);
        S2[41] = fma_f32x2_rn_ftz(k2[9], vn2p1_2, S2[41]);
        S2[42] = fma_f32x2_rn_ftz(k2[10], vn2p1_2, S2[42]);
        S2[43] = fma_f32x2_rn_ftz(k2[11], vn2p1_2, S2[43]);
        S2[44] = fma_f32x2_rn_ftz(k2[12], vn2p1_2, S2[44]);
        S2[45] = fma_f32x2_rn_ftz(k2[13], vn2p1_2, S2[45]);
        S2[46] = fma_f32x2_rn_ftz(k2[14], vn2p1_2, S2[46]);
        S2[47] = fma_f32x2_rn_ftz(k2[15], vn2p1_2, S2[47]);
        float2 accp1_3_0 = make_float2(0.0f, 0.0f);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[48], k2[0], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[49], k2[1], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[50], k2[2], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[51], k2[3], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[52], k2[4], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[53], k2[5], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[54], k2[6], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[55], k2[7], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[56], k2[8], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[57], k2[9], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[58], k2[10], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[59], k2[11], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[60], k2[12], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[61], k2[13], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[62], k2[14], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[63], k2[15], accp1_3_0);
        float dotp1_3 = accp1_3_0.x + accp1_3_0.y;
        float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, dotp1_3, 1);
        dotp1_3 = dotp1_3 + _shfl_xor_sync_89;
        float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, dotp1_3, 2);
        dotp1_3 = dotp1_3 + _shfl_xor_sync_90;
#endif
        float vnp1_3 = (v_regs[3] - dotp1_3) * betap1;
#if __CUDA_ARCH__ == 900
        float vn2p1_3_x = vnp1_3;
        float vn2p1_3_y = vnp1_3;
        float _fma_1384;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1384)
            : "f"(k_regs[0]), "f"(vn2p1_3_x), "f"(S[96]));
        float _fma_1385;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1385)
            : "f"(k_regs[1]), "f"(vn2p1_3_y), "f"(S[97]));
        S[96] = _fma_1384;
        S[97] = _fma_1385;
        float _fma_1386;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1386)
            : "f"(k_regs[2]), "f"(vn2p1_3_x), "f"(S[98]));
        float _fma_1387;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1387)
            : "f"(k_regs[3]), "f"(vn2p1_3_y), "f"(S[99]));
        S[98] = _fma_1386;
        S[99] = _fma_1387;
        float _fma_1388;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1388)
            : "f"(k_regs[4]), "f"(vn2p1_3_x), "f"(S[100]));
        float _fma_1389;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1389)
            : "f"(k_regs[5]), "f"(vn2p1_3_y), "f"(S[101]));
        S[100] = _fma_1388;
        S[101] = _fma_1389;
        float _fma_1390;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1390)
            : "f"(k_regs[6]), "f"(vn2p1_3_x), "f"(S[102]));
        float _fma_1391;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1391)
            : "f"(k_regs[7]), "f"(vn2p1_3_y), "f"(S[103]));
        S[102] = _fma_1390;
        S[103] = _fma_1391;
        float _fma_1392;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1392)
            : "f"(k_regs[8]), "f"(vn2p1_3_x), "f"(S[104]));
        float _fma_1393;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1393)
            : "f"(k_regs[9]), "f"(vn2p1_3_y), "f"(S[105]));
        S[104] = _fma_1392;
        S[105] = _fma_1393;
        float _fma_1394;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1394)
            : "f"(k_regs[10]), "f"(vn2p1_3_x), "f"(S[106]));
        float _fma_1395;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1395)
            : "f"(k_regs[11]), "f"(vn2p1_3_y), "f"(S[107]));
        S[106] = _fma_1394;
        S[107] = _fma_1395;
        float _fma_1396;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1396)
            : "f"(k_regs[12]), "f"(vn2p1_3_x), "f"(S[108]));
        float _fma_1397;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1397)
            : "f"(k_regs[13]), "f"(vn2p1_3_y), "f"(S[109]));
        S[108] = _fma_1396;
        S[109] = _fma_1397;
        float _fma_1398;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1398)
            : "f"(k_regs[14]), "f"(vn2p1_3_x), "f"(S[110]));
        float _fma_1399;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1399)
            : "f"(k_regs[15]), "f"(vn2p1_3_y), "f"(S[111]));
        S[110] = _fma_1398;
        S[111] = _fma_1399;
        float _fma_1400;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1400)
            : "f"(k_regs[16]), "f"(vn2p1_3_x), "f"(S[112]));
        float _fma_1401;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1401)
            : "f"(k_regs[17]), "f"(vn2p1_3_y), "f"(S[113]));
        S[112] = _fma_1400;
        S[113] = _fma_1401;
        float _fma_1402;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1402)
            : "f"(k_regs[18]), "f"(vn2p1_3_x), "f"(S[114]));
        float _fma_1403;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1403)
            : "f"(k_regs[19]), "f"(vn2p1_3_y), "f"(S[115]));
        S[114] = _fma_1402;
        S[115] = _fma_1403;
        float _fma_1404;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1404)
            : "f"(k_regs[20]), "f"(vn2p1_3_x), "f"(S[116]));
        float _fma_1405;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1405)
            : "f"(k_regs[21]), "f"(vn2p1_3_y), "f"(S[117]));
        S[116] = _fma_1404;
        S[117] = _fma_1405;
        float _fma_1406;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1406)
            : "f"(k_regs[22]), "f"(vn2p1_3_x), "f"(S[118]));
        float _fma_1407;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1407)
            : "f"(k_regs[23]), "f"(vn2p1_3_y), "f"(S[119]));
        S[118] = _fma_1406;
        S[119] = _fma_1407;
        float _fma_1408;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1408)
            : "f"(k_regs[24]), "f"(vn2p1_3_x), "f"(S[120]));
        float _fma_1409;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1409)
            : "f"(k_regs[25]), "f"(vn2p1_3_y), "f"(S[121]));
        S[120] = _fma_1408;
        S[121] = _fma_1409;
        float _fma_1410;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1410)
            : "f"(k_regs[26]), "f"(vn2p1_3_x), "f"(S[122]));
        float _fma_1411;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1411)
            : "f"(k_regs[27]), "f"(vn2p1_3_y), "f"(S[123]));
        S[122] = _fma_1410;
        S[123] = _fma_1411;
        float _fma_1412;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1412)
            : "f"(k_regs[28]), "f"(vn2p1_3_x), "f"(S[124]));
        float _fma_1413;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1413)
            : "f"(k_regs[29]), "f"(vn2p1_3_y), "f"(S[125]));
        S[124] = _fma_1412;
        S[125] = _fma_1413;
        float _fma_1414;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1414)
            : "f"(k_regs[30]), "f"(vn2p1_3_x), "f"(S[126]));
        float _fma_1415;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1415)
            : "f"(k_regs[31]), "f"(vn2p1_3_y), "f"(S[127]));
        S[126] = _fma_1414;
        S[127] = _fma_1415;
#else
        float2 vn2p1_3 = make_float2(vnp1_3, vnp1_3);
        S2[48] = fma_f32x2_rn_ftz(k2[0], vn2p1_3, S2[48]);
        S2[49] = fma_f32x2_rn_ftz(k2[1], vn2p1_3, S2[49]);
        S2[50] = fma_f32x2_rn_ftz(k2[2], vn2p1_3, S2[50]);
        S2[51] = fma_f32x2_rn_ftz(k2[3], vn2p1_3, S2[51]);
        S2[52] = fma_f32x2_rn_ftz(k2[4], vn2p1_3, S2[52]);
        S2[53] = fma_f32x2_rn_ftz(k2[5], vn2p1_3, S2[53]);
        S2[54] = fma_f32x2_rn_ftz(k2[6], vn2p1_3, S2[54]);
        S2[55] = fma_f32x2_rn_ftz(k2[7], vn2p1_3, S2[55]);
        S2[56] = fma_f32x2_rn_ftz(k2[8], vn2p1_3, S2[56]);
        S2[57] = fma_f32x2_rn_ftz(k2[9], vn2p1_3, S2[57]);
        S2[58] = fma_f32x2_rn_ftz(k2[10], vn2p1_3, S2[58]);
        S2[59] = fma_f32x2_rn_ftz(k2[11], vn2p1_3, S2[59]);
        S2[60] = fma_f32x2_rn_ftz(k2[12], vn2p1_3, S2[60]);
        S2[61] = fma_f32x2_rn_ftz(k2[13], vn2p1_3, S2[61]);
        S2[62] = fma_f32x2_rn_ftz(k2[14], vn2p1_3, S2[62]);
        S2[63] = fma_f32x2_rn_ftz(k2[15], vn2p1_3, S2[63]);
#endif
        int mp2 = t_2 * 3 + 2;
        unsigned int kbasep2 = sK_addr + (unsigned int)(mp2 * 512) + lane_byte;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                     : "r"(kbasep2));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                     : "r"(kbasep2 + 64));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                     : "r"(kbasep2 + 128));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                     : "r"(kbasep2 + 192));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[16])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 3]))
                     : "r"(kbasep2 + 256));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[20])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 3]))
                     : "r"(kbasep2 + 320));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[24])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 3]))
                     : "r"(kbasep2 + 384));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[28])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 3]))
                     : "r"(kbasep2 + 448));
#if __CUDA_ARCH__ == 900
#else
        k2[0] = make_float2(k_regs[0], k_regs[1]);
        k2[1] = make_float2(k_regs[2], k_regs[3]);
        k2[2] = make_float2(k_regs[4], k_regs[5]);
        k2[3] = make_float2(k_regs[6], k_regs[7]);
        k2[4] = make_float2(k_regs[8], k_regs[9]);
        k2[5] = make_float2(k_regs[10], k_regs[11]);
        k2[6] = make_float2(k_regs[12], k_regs[13]);
        k2[7] = make_float2(k_regs[14], k_regs[15]);
        k2[8] = make_float2(k_regs[16], k_regs[17]);
        k2[9] = make_float2(k_regs[18], k_regs[19]);
        k2[10] = make_float2(k_regs[20], k_regs[21]);
        k2[11] = make_float2(k_regs[22], k_regs[23]);
        k2[12] = make_float2(k_regs[24], k_regs[25]);
        k2[13] = make_float2(k_regs[26], k_regs[27]);
        k2[14] = make_float2(k_regs[28], k_regs[29]);
        k2[15] = make_float2(k_regs[30], k_regs[31]);
#endif
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                     : "r"(sV_addr + (unsigned int)(mp2 * 256) + group_byte));
        float betap2 = sBeta[mp2];
#if __CUDA_ARCH__ == 900
        float accp2_0_0_x = 0.0f;
        float accp2_0_0_y = 0.0f;
        float accp2_0_1_x = 0.0f;
        float accp2_0_1_y = 0.0f;
        float _fma_1416;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1416)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accp2_0_0_x));
        float _fma_1417;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1417)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accp2_0_0_y));
        accp2_0_0_x = _fma_1416;
        accp2_0_0_y = _fma_1417;
        float _fma_1418;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1418)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accp2_0_1_x));
        float _fma_1419;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1419)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accp2_0_1_y));
        accp2_0_1_x = _fma_1418;
        accp2_0_1_y = _fma_1419;
        float _fma_1420;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1420)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accp2_0_0_x));
        float _fma_1421;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1421)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accp2_0_0_y));
        accp2_0_0_x = _fma_1420;
        accp2_0_0_y = _fma_1421;
        float _fma_1422;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1422)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accp2_0_1_x));
        float _fma_1423;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1423)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accp2_0_1_y));
        accp2_0_1_x = _fma_1422;
        accp2_0_1_y = _fma_1423;
        float _fma_1424;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1424)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accp2_0_0_x));
        float _fma_1425;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1425)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accp2_0_0_y));
        accp2_0_0_x = _fma_1424;
        accp2_0_0_y = _fma_1425;
        float _fma_1426;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1426)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accp2_0_1_x));
        float _fma_1427;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1427)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accp2_0_1_y));
        accp2_0_1_x = _fma_1426;
        accp2_0_1_y = _fma_1427;
        float _fma_1428;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1428)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accp2_0_0_x));
        float _fma_1429;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1429)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accp2_0_0_y));
        accp2_0_0_x = _fma_1428;
        accp2_0_0_y = _fma_1429;
        float _fma_1430;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1430)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accp2_0_1_x));
        float _fma_1431;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1431)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accp2_0_1_y));
        accp2_0_1_x = _fma_1430;
        accp2_0_1_y = _fma_1431;
        float _fma_1432;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1432)
            : "f"(S[16]), "f"(k_regs[16]), "f"(accp2_0_0_x));
        float _fma_1433;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1433)
            : "f"(S[17]), "f"(k_regs[17]), "f"(accp2_0_0_y));
        accp2_0_0_x = _fma_1432;
        accp2_0_0_y = _fma_1433;
        float _fma_1434;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1434)
            : "f"(S[18]), "f"(k_regs[18]), "f"(accp2_0_1_x));
        float _fma_1435;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1435)
            : "f"(S[19]), "f"(k_regs[19]), "f"(accp2_0_1_y));
        accp2_0_1_x = _fma_1434;
        accp2_0_1_y = _fma_1435;
        float _fma_1436;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1436)
            : "f"(S[20]), "f"(k_regs[20]), "f"(accp2_0_0_x));
        float _fma_1437;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1437)
            : "f"(S[21]), "f"(k_regs[21]), "f"(accp2_0_0_y));
        accp2_0_0_x = _fma_1436;
        accp2_0_0_y = _fma_1437;
        float _fma_1438;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1438)
            : "f"(S[22]), "f"(k_regs[22]), "f"(accp2_0_1_x));
        float _fma_1439;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1439)
            : "f"(S[23]), "f"(k_regs[23]), "f"(accp2_0_1_y));
        accp2_0_1_x = _fma_1438;
        accp2_0_1_y = _fma_1439;
        float _fma_1440;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1440)
            : "f"(S[24]), "f"(k_regs[24]), "f"(accp2_0_0_x));
        float _fma_1441;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1441)
            : "f"(S[25]), "f"(k_regs[25]), "f"(accp2_0_0_y));
        accp2_0_0_x = _fma_1440;
        accp2_0_0_y = _fma_1441;
        float _fma_1442;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1442)
            : "f"(S[26]), "f"(k_regs[26]), "f"(accp2_0_1_x));
        float _fma_1443;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1443)
            : "f"(S[27]), "f"(k_regs[27]), "f"(accp2_0_1_y));
        accp2_0_1_x = _fma_1442;
        accp2_0_1_y = _fma_1443;
        float _fma_1444;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1444)
            : "f"(S[28]), "f"(k_regs[28]), "f"(accp2_0_0_x));
        float _fma_1445;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1445)
            : "f"(S[29]), "f"(k_regs[29]), "f"(accp2_0_0_y));
        accp2_0_0_x = _fma_1444;
        accp2_0_0_y = _fma_1445;
        float _fma_1446;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1446)
            : "f"(S[30]), "f"(k_regs[30]), "f"(accp2_0_1_x));
        float _fma_1447;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1447)
            : "f"(S[31]), "f"(k_regs[31]), "f"(accp2_0_1_y));
        accp2_0_1_x = _fma_1446;
        accp2_0_1_y = _fma_1447;
        float _add_139;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_139) : "f"(accp2_0_0_x), "f"(accp2_0_1_x));
        float _add_140;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_140) : "f"(accp2_0_0_y), "f"(accp2_0_1_y));
        float dotp2_0 = _add_139 + _add_140;
        float _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 1);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_141;
        float _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 2);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_142;
#else
        float2 accp2_0_0 = make_float2(0.0f, 0.0f);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accp2_0_0);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[1], k2[1], accp2_0_0);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accp2_0_0);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[3], k2[3], accp2_0_0);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accp2_0_0);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[5], k2[5], accp2_0_0);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accp2_0_0);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[7], k2[7], accp2_0_0);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[8], k2[8], accp2_0_0);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[9], k2[9], accp2_0_0);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[10], k2[10], accp2_0_0);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[11], k2[11], accp2_0_0);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[12], k2[12], accp2_0_0);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[13], k2[13], accp2_0_0);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[14], k2[14], accp2_0_0);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[15], k2[15], accp2_0_0);
        float dotp2_0 = accp2_0_0.x + accp2_0_0.y;
        float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 1);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_91;
        float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 2);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_92;
#endif
        float vnp2_0 = (v_regs[0] - dotp2_0) * betap2;
#if __CUDA_ARCH__ == 900
        float vn2p2_0_x = vnp2_0;
        float vn2p2_0_y = vnp2_0;
        float _fma_1448;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1448)
            : "f"(k_regs[0]), "f"(vn2p2_0_x), "f"(S[0]));
        float _fma_1449;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1449)
            : "f"(k_regs[1]), "f"(vn2p2_0_y), "f"(S[1]));
        S[0] = _fma_1448;
        S[1] = _fma_1449;
        float _fma_1450;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1450)
            : "f"(k_regs[2]), "f"(vn2p2_0_x), "f"(S[2]));
        float _fma_1451;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1451)
            : "f"(k_regs[3]), "f"(vn2p2_0_y), "f"(S[3]));
        S[2] = _fma_1450;
        S[3] = _fma_1451;
        float _fma_1452;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1452)
            : "f"(k_regs[4]), "f"(vn2p2_0_x), "f"(S[4]));
        float _fma_1453;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1453)
            : "f"(k_regs[5]), "f"(vn2p2_0_y), "f"(S[5]));
        S[4] = _fma_1452;
        S[5] = _fma_1453;
        float _fma_1454;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1454)
            : "f"(k_regs[6]), "f"(vn2p2_0_x), "f"(S[6]));
        float _fma_1455;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1455)
            : "f"(k_regs[7]), "f"(vn2p2_0_y), "f"(S[7]));
        S[6] = _fma_1454;
        S[7] = _fma_1455;
        float _fma_1456;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1456)
            : "f"(k_regs[8]), "f"(vn2p2_0_x), "f"(S[8]));
        float _fma_1457;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1457)
            : "f"(k_regs[9]), "f"(vn2p2_0_y), "f"(S[9]));
        S[8] = _fma_1456;
        S[9] = _fma_1457;
        float _fma_1458;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1458)
            : "f"(k_regs[10]), "f"(vn2p2_0_x), "f"(S[10]));
        float _fma_1459;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1459)
            : "f"(k_regs[11]), "f"(vn2p2_0_y), "f"(S[11]));
        S[10] = _fma_1458;
        S[11] = _fma_1459;
        float _fma_1460;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1460)
            : "f"(k_regs[12]), "f"(vn2p2_0_x), "f"(S[12]));
        float _fma_1461;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1461)
            : "f"(k_regs[13]), "f"(vn2p2_0_y), "f"(S[13]));
        S[12] = _fma_1460;
        S[13] = _fma_1461;
        float _fma_1462;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1462)
            : "f"(k_regs[14]), "f"(vn2p2_0_x), "f"(S[14]));
        float _fma_1463;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1463)
            : "f"(k_regs[15]), "f"(vn2p2_0_y), "f"(S[15]));
        S[14] = _fma_1462;
        S[15] = _fma_1463;
        float _fma_1464;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1464)
            : "f"(k_regs[16]), "f"(vn2p2_0_x), "f"(S[16]));
        float _fma_1465;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1465)
            : "f"(k_regs[17]), "f"(vn2p2_0_y), "f"(S[17]));
        S[16] = _fma_1464;
        S[17] = _fma_1465;
        float _fma_1466;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1466)
            : "f"(k_regs[18]), "f"(vn2p2_0_x), "f"(S[18]));
        float _fma_1467;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1467)
            : "f"(k_regs[19]), "f"(vn2p2_0_y), "f"(S[19]));
        S[18] = _fma_1466;
        S[19] = _fma_1467;
        float _fma_1468;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1468)
            : "f"(k_regs[20]), "f"(vn2p2_0_x), "f"(S[20]));
        float _fma_1469;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1469)
            : "f"(k_regs[21]), "f"(vn2p2_0_y), "f"(S[21]));
        S[20] = _fma_1468;
        S[21] = _fma_1469;
        float _fma_1470;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1470)
            : "f"(k_regs[22]), "f"(vn2p2_0_x), "f"(S[22]));
        float _fma_1471;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1471)
            : "f"(k_regs[23]), "f"(vn2p2_0_y), "f"(S[23]));
        S[22] = _fma_1470;
        S[23] = _fma_1471;
        float _fma_1472;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1472)
            : "f"(k_regs[24]), "f"(vn2p2_0_x), "f"(S[24]));
        float _fma_1473;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1473)
            : "f"(k_regs[25]), "f"(vn2p2_0_y), "f"(S[25]));
        S[24] = _fma_1472;
        S[25] = _fma_1473;
        float _fma_1474;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1474)
            : "f"(k_regs[26]), "f"(vn2p2_0_x), "f"(S[26]));
        float _fma_1475;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1475)
            : "f"(k_regs[27]), "f"(vn2p2_0_y), "f"(S[27]));
        S[26] = _fma_1474;
        S[27] = _fma_1475;
        float _fma_1476;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1476)
            : "f"(k_regs[28]), "f"(vn2p2_0_x), "f"(S[28]));
        float _fma_1477;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1477)
            : "f"(k_regs[29]), "f"(vn2p2_0_y), "f"(S[29]));
        S[28] = _fma_1476;
        S[29] = _fma_1477;
        float _fma_1478;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1478)
            : "f"(k_regs[30]), "f"(vn2p2_0_x), "f"(S[30]));
        float _fma_1479;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1479)
            : "f"(k_regs[31]), "f"(vn2p2_0_y), "f"(S[31]));
        S[30] = _fma_1478;
        S[31] = _fma_1479;
        float accp2_1_0_x = 0.0f;
        float accp2_1_0_y = 0.0f;
        float accp2_1_1_x = 0.0f;
        float accp2_1_1_y = 0.0f;
        float _fma_1480;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1480)
            : "f"(S[32]), "f"(k_regs[0]), "f"(accp2_1_0_x));
        float _fma_1481;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1481)
            : "f"(S[33]), "f"(k_regs[1]), "f"(accp2_1_0_y));
        accp2_1_0_x = _fma_1480;
        accp2_1_0_y = _fma_1481;
        float _fma_1482;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1482)
            : "f"(S[34]), "f"(k_regs[2]), "f"(accp2_1_1_x));
        float _fma_1483;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1483)
            : "f"(S[35]), "f"(k_regs[3]), "f"(accp2_1_1_y));
        accp2_1_1_x = _fma_1482;
        accp2_1_1_y = _fma_1483;
        float _fma_1484;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1484)
            : "f"(S[36]), "f"(k_regs[4]), "f"(accp2_1_0_x));
        float _fma_1485;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1485)
            : "f"(S[37]), "f"(k_regs[5]), "f"(accp2_1_0_y));
        accp2_1_0_x = _fma_1484;
        accp2_1_0_y = _fma_1485;
        float _fma_1486;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1486)
            : "f"(S[38]), "f"(k_regs[6]), "f"(accp2_1_1_x));
        float _fma_1487;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1487)
            : "f"(S[39]), "f"(k_regs[7]), "f"(accp2_1_1_y));
        accp2_1_1_x = _fma_1486;
        accp2_1_1_y = _fma_1487;
        float _fma_1488;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1488)
            : "f"(S[40]), "f"(k_regs[8]), "f"(accp2_1_0_x));
        float _fma_1489;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1489)
            : "f"(S[41]), "f"(k_regs[9]), "f"(accp2_1_0_y));
        accp2_1_0_x = _fma_1488;
        accp2_1_0_y = _fma_1489;
        float _fma_1490;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1490)
            : "f"(S[42]), "f"(k_regs[10]), "f"(accp2_1_1_x));
        float _fma_1491;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1491)
            : "f"(S[43]), "f"(k_regs[11]), "f"(accp2_1_1_y));
        accp2_1_1_x = _fma_1490;
        accp2_1_1_y = _fma_1491;
        float _fma_1492;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1492)
            : "f"(S[44]), "f"(k_regs[12]), "f"(accp2_1_0_x));
        float _fma_1493;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1493)
            : "f"(S[45]), "f"(k_regs[13]), "f"(accp2_1_0_y));
        accp2_1_0_x = _fma_1492;
        accp2_1_0_y = _fma_1493;
        float _fma_1494;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1494)
            : "f"(S[46]), "f"(k_regs[14]), "f"(accp2_1_1_x));
        float _fma_1495;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1495)
            : "f"(S[47]), "f"(k_regs[15]), "f"(accp2_1_1_y));
        accp2_1_1_x = _fma_1494;
        accp2_1_1_y = _fma_1495;
        float _fma_1496;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1496)
            : "f"(S[48]), "f"(k_regs[16]), "f"(accp2_1_0_x));
        float _fma_1497;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1497)
            : "f"(S[49]), "f"(k_regs[17]), "f"(accp2_1_0_y));
        accp2_1_0_x = _fma_1496;
        accp2_1_0_y = _fma_1497;
        float _fma_1498;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1498)
            : "f"(S[50]), "f"(k_regs[18]), "f"(accp2_1_1_x));
        float _fma_1499;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1499)
            : "f"(S[51]), "f"(k_regs[19]), "f"(accp2_1_1_y));
        accp2_1_1_x = _fma_1498;
        accp2_1_1_y = _fma_1499;
        float _fma_1500;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1500)
            : "f"(S[52]), "f"(k_regs[20]), "f"(accp2_1_0_x));
        float _fma_1501;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1501)
            : "f"(S[53]), "f"(k_regs[21]), "f"(accp2_1_0_y));
        accp2_1_0_x = _fma_1500;
        accp2_1_0_y = _fma_1501;
        float _fma_1502;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1502)
            : "f"(S[54]), "f"(k_regs[22]), "f"(accp2_1_1_x));
        float _fma_1503;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1503)
            : "f"(S[55]), "f"(k_regs[23]), "f"(accp2_1_1_y));
        accp2_1_1_x = _fma_1502;
        accp2_1_1_y = _fma_1503;
        float _fma_1504;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1504)
            : "f"(S[56]), "f"(k_regs[24]), "f"(accp2_1_0_x));
        float _fma_1505;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1505)
            : "f"(S[57]), "f"(k_regs[25]), "f"(accp2_1_0_y));
        accp2_1_0_x = _fma_1504;
        accp2_1_0_y = _fma_1505;
        float _fma_1506;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1506)
            : "f"(S[58]), "f"(k_regs[26]), "f"(accp2_1_1_x));
        float _fma_1507;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1507)
            : "f"(S[59]), "f"(k_regs[27]), "f"(accp2_1_1_y));
        accp2_1_1_x = _fma_1506;
        accp2_1_1_y = _fma_1507;
        float _fma_1508;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1508)
            : "f"(S[60]), "f"(k_regs[28]), "f"(accp2_1_0_x));
        float _fma_1509;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1509)
            : "f"(S[61]), "f"(k_regs[29]), "f"(accp2_1_0_y));
        accp2_1_0_x = _fma_1508;
        accp2_1_0_y = _fma_1509;
        float _fma_1510;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1510)
            : "f"(S[62]), "f"(k_regs[30]), "f"(accp2_1_1_x));
        float _fma_1511;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1511)
            : "f"(S[63]), "f"(k_regs[31]), "f"(accp2_1_1_y));
        accp2_1_1_x = _fma_1510;
        accp2_1_1_y = _fma_1511;
        float _add_143;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_143) : "f"(accp2_1_0_x), "f"(accp2_1_1_x));
        float _add_144;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_144) : "f"(accp2_1_0_y), "f"(accp2_1_1_y));
        float dotp2_1 = _add_143 + _add_144;
        float _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 1);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_145;
        float _shfl_xor_sync_146 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 2);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_146;
#else
        float2 vn2p2_0 = make_float2(vnp2_0, vnp2_0);
        S2[0] = fma_f32x2_rn_ftz(k2[0], vn2p2_0, S2[0]);
        S2[1] = fma_f32x2_rn_ftz(k2[1], vn2p2_0, S2[1]);
        S2[2] = fma_f32x2_rn_ftz(k2[2], vn2p2_0, S2[2]);
        S2[3] = fma_f32x2_rn_ftz(k2[3], vn2p2_0, S2[3]);
        S2[4] = fma_f32x2_rn_ftz(k2[4], vn2p2_0, S2[4]);
        S2[5] = fma_f32x2_rn_ftz(k2[5], vn2p2_0, S2[5]);
        S2[6] = fma_f32x2_rn_ftz(k2[6], vn2p2_0, S2[6]);
        S2[7] = fma_f32x2_rn_ftz(k2[7], vn2p2_0, S2[7]);
        S2[8] = fma_f32x2_rn_ftz(k2[8], vn2p2_0, S2[8]);
        S2[9] = fma_f32x2_rn_ftz(k2[9], vn2p2_0, S2[9]);
        S2[10] = fma_f32x2_rn_ftz(k2[10], vn2p2_0, S2[10]);
        S2[11] = fma_f32x2_rn_ftz(k2[11], vn2p2_0, S2[11]);
        S2[12] = fma_f32x2_rn_ftz(k2[12], vn2p2_0, S2[12]);
        S2[13] = fma_f32x2_rn_ftz(k2[13], vn2p2_0, S2[13]);
        S2[14] = fma_f32x2_rn_ftz(k2[14], vn2p2_0, S2[14]);
        S2[15] = fma_f32x2_rn_ftz(k2[15], vn2p2_0, S2[15]);
        float2 accp2_1_0 = make_float2(0.0f, 0.0f);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[24], k2[8], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[25], k2[9], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[26], k2[10], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[27], k2[11], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[28], k2[12], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[29], k2[13], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[30], k2[14], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[31], k2[15], accp2_1_0);
        float dotp2_1 = accp2_1_0.x + accp2_1_0.y;
        float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 1);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_93;
        float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 2);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_94;
#endif
        float vnp2_1 = (v_regs[1] - dotp2_1) * betap2;
#if __CUDA_ARCH__ == 900
        float vn2p2_1_x = vnp2_1;
        float vn2p2_1_y = vnp2_1;
        float _fma_1512;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1512)
            : "f"(k_regs[0]), "f"(vn2p2_1_x), "f"(S[32]));
        float _fma_1513;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1513)
            : "f"(k_regs[1]), "f"(vn2p2_1_y), "f"(S[33]));
        S[32] = _fma_1512;
        S[33] = _fma_1513;
        float _fma_1514;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1514)
            : "f"(k_regs[2]), "f"(vn2p2_1_x), "f"(S[34]));
        float _fma_1515;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1515)
            : "f"(k_regs[3]), "f"(vn2p2_1_y), "f"(S[35]));
        S[34] = _fma_1514;
        S[35] = _fma_1515;
        float _fma_1516;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1516)
            : "f"(k_regs[4]), "f"(vn2p2_1_x), "f"(S[36]));
        float _fma_1517;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1517)
            : "f"(k_regs[5]), "f"(vn2p2_1_y), "f"(S[37]));
        S[36] = _fma_1516;
        S[37] = _fma_1517;
        float _fma_1518;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1518)
            : "f"(k_regs[6]), "f"(vn2p2_1_x), "f"(S[38]));
        float _fma_1519;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1519)
            : "f"(k_regs[7]), "f"(vn2p2_1_y), "f"(S[39]));
        S[38] = _fma_1518;
        S[39] = _fma_1519;
        float _fma_1520;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1520)
            : "f"(k_regs[8]), "f"(vn2p2_1_x), "f"(S[40]));
        float _fma_1521;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1521)
            : "f"(k_regs[9]), "f"(vn2p2_1_y), "f"(S[41]));
        S[40] = _fma_1520;
        S[41] = _fma_1521;
        float _fma_1522;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1522)
            : "f"(k_regs[10]), "f"(vn2p2_1_x), "f"(S[42]));
        float _fma_1523;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1523)
            : "f"(k_regs[11]), "f"(vn2p2_1_y), "f"(S[43]));
        S[42] = _fma_1522;
        S[43] = _fma_1523;
        float _fma_1524;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1524)
            : "f"(k_regs[12]), "f"(vn2p2_1_x), "f"(S[44]));
        float _fma_1525;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1525)
            : "f"(k_regs[13]), "f"(vn2p2_1_y), "f"(S[45]));
        S[44] = _fma_1524;
        S[45] = _fma_1525;
        float _fma_1526;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1526)
            : "f"(k_regs[14]), "f"(vn2p2_1_x), "f"(S[46]));
        float _fma_1527;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1527)
            : "f"(k_regs[15]), "f"(vn2p2_1_y), "f"(S[47]));
        S[46] = _fma_1526;
        S[47] = _fma_1527;
        float _fma_1528;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1528)
            : "f"(k_regs[16]), "f"(vn2p2_1_x), "f"(S[48]));
        float _fma_1529;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1529)
            : "f"(k_regs[17]), "f"(vn2p2_1_y), "f"(S[49]));
        S[48] = _fma_1528;
        S[49] = _fma_1529;
        float _fma_1530;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1530)
            : "f"(k_regs[18]), "f"(vn2p2_1_x), "f"(S[50]));
        float _fma_1531;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1531)
            : "f"(k_regs[19]), "f"(vn2p2_1_y), "f"(S[51]));
        S[50] = _fma_1530;
        S[51] = _fma_1531;
        float _fma_1532;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1532)
            : "f"(k_regs[20]), "f"(vn2p2_1_x), "f"(S[52]));
        float _fma_1533;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1533)
            : "f"(k_regs[21]), "f"(vn2p2_1_y), "f"(S[53]));
        S[52] = _fma_1532;
        S[53] = _fma_1533;
        float _fma_1534;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1534)
            : "f"(k_regs[22]), "f"(vn2p2_1_x), "f"(S[54]));
        float _fma_1535;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1535)
            : "f"(k_regs[23]), "f"(vn2p2_1_y), "f"(S[55]));
        S[54] = _fma_1534;
        S[55] = _fma_1535;
        float _fma_1536;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1536)
            : "f"(k_regs[24]), "f"(vn2p2_1_x), "f"(S[56]));
        float _fma_1537;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1537)
            : "f"(k_regs[25]), "f"(vn2p2_1_y), "f"(S[57]));
        S[56] = _fma_1536;
        S[57] = _fma_1537;
        float _fma_1538;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1538)
            : "f"(k_regs[26]), "f"(vn2p2_1_x), "f"(S[58]));
        float _fma_1539;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1539)
            : "f"(k_regs[27]), "f"(vn2p2_1_y), "f"(S[59]));
        S[58] = _fma_1538;
        S[59] = _fma_1539;
        float _fma_1540;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1540)
            : "f"(k_regs[28]), "f"(vn2p2_1_x), "f"(S[60]));
        float _fma_1541;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1541)
            : "f"(k_regs[29]), "f"(vn2p2_1_y), "f"(S[61]));
        S[60] = _fma_1540;
        S[61] = _fma_1541;
        float _fma_1542;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1542)
            : "f"(k_regs[30]), "f"(vn2p2_1_x), "f"(S[62]));
        float _fma_1543;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1543)
            : "f"(k_regs[31]), "f"(vn2p2_1_y), "f"(S[63]));
        S[62] = _fma_1542;
        S[63] = _fma_1543;
        float accp2_2_0_x = 0.0f;
        float accp2_2_0_y = 0.0f;
        float accp2_2_1_x = 0.0f;
        float accp2_2_1_y = 0.0f;
        float _fma_1544;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1544)
            : "f"(S[64]), "f"(k_regs[0]), "f"(accp2_2_0_x));
        float _fma_1545;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1545)
            : "f"(S[65]), "f"(k_regs[1]), "f"(accp2_2_0_y));
        accp2_2_0_x = _fma_1544;
        accp2_2_0_y = _fma_1545;
        float _fma_1546;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1546)
            : "f"(S[66]), "f"(k_regs[2]), "f"(accp2_2_1_x));
        float _fma_1547;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1547)
            : "f"(S[67]), "f"(k_regs[3]), "f"(accp2_2_1_y));
        accp2_2_1_x = _fma_1546;
        accp2_2_1_y = _fma_1547;
        float _fma_1548;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1548)
            : "f"(S[68]), "f"(k_regs[4]), "f"(accp2_2_0_x));
        float _fma_1549;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1549)
            : "f"(S[69]), "f"(k_regs[5]), "f"(accp2_2_0_y));
        accp2_2_0_x = _fma_1548;
        accp2_2_0_y = _fma_1549;
        float _fma_1550;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1550)
            : "f"(S[70]), "f"(k_regs[6]), "f"(accp2_2_1_x));
        float _fma_1551;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1551)
            : "f"(S[71]), "f"(k_regs[7]), "f"(accp2_2_1_y));
        accp2_2_1_x = _fma_1550;
        accp2_2_1_y = _fma_1551;
        float _fma_1552;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1552)
            : "f"(S[72]), "f"(k_regs[8]), "f"(accp2_2_0_x));
        float _fma_1553;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1553)
            : "f"(S[73]), "f"(k_regs[9]), "f"(accp2_2_0_y));
        accp2_2_0_x = _fma_1552;
        accp2_2_0_y = _fma_1553;
        float _fma_1554;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1554)
            : "f"(S[74]), "f"(k_regs[10]), "f"(accp2_2_1_x));
        float _fma_1555;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1555)
            : "f"(S[75]), "f"(k_regs[11]), "f"(accp2_2_1_y));
        accp2_2_1_x = _fma_1554;
        accp2_2_1_y = _fma_1555;
        float _fma_1556;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1556)
            : "f"(S[76]), "f"(k_regs[12]), "f"(accp2_2_0_x));
        float _fma_1557;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1557)
            : "f"(S[77]), "f"(k_regs[13]), "f"(accp2_2_0_y));
        accp2_2_0_x = _fma_1556;
        accp2_2_0_y = _fma_1557;
        float _fma_1558;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1558)
            : "f"(S[78]), "f"(k_regs[14]), "f"(accp2_2_1_x));
        float _fma_1559;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1559)
            : "f"(S[79]), "f"(k_regs[15]), "f"(accp2_2_1_y));
        accp2_2_1_x = _fma_1558;
        accp2_2_1_y = _fma_1559;
        float _fma_1560;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1560)
            : "f"(S[80]), "f"(k_regs[16]), "f"(accp2_2_0_x));
        float _fma_1561;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1561)
            : "f"(S[81]), "f"(k_regs[17]), "f"(accp2_2_0_y));
        accp2_2_0_x = _fma_1560;
        accp2_2_0_y = _fma_1561;
        float _fma_1562;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1562)
            : "f"(S[82]), "f"(k_regs[18]), "f"(accp2_2_1_x));
        float _fma_1563;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1563)
            : "f"(S[83]), "f"(k_regs[19]), "f"(accp2_2_1_y));
        accp2_2_1_x = _fma_1562;
        accp2_2_1_y = _fma_1563;
        float _fma_1564;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1564)
            : "f"(S[84]), "f"(k_regs[20]), "f"(accp2_2_0_x));
        float _fma_1565;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1565)
            : "f"(S[85]), "f"(k_regs[21]), "f"(accp2_2_0_y));
        accp2_2_0_x = _fma_1564;
        accp2_2_0_y = _fma_1565;
        float _fma_1566;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1566)
            : "f"(S[86]), "f"(k_regs[22]), "f"(accp2_2_1_x));
        float _fma_1567;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1567)
            : "f"(S[87]), "f"(k_regs[23]), "f"(accp2_2_1_y));
        accp2_2_1_x = _fma_1566;
        accp2_2_1_y = _fma_1567;
        float _fma_1568;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1568)
            : "f"(S[88]), "f"(k_regs[24]), "f"(accp2_2_0_x));
        float _fma_1569;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1569)
            : "f"(S[89]), "f"(k_regs[25]), "f"(accp2_2_0_y));
        accp2_2_0_x = _fma_1568;
        accp2_2_0_y = _fma_1569;
        float _fma_1570;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1570)
            : "f"(S[90]), "f"(k_regs[26]), "f"(accp2_2_1_x));
        float _fma_1571;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1571)
            : "f"(S[91]), "f"(k_regs[27]), "f"(accp2_2_1_y));
        accp2_2_1_x = _fma_1570;
        accp2_2_1_y = _fma_1571;
        float _fma_1572;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1572)
            : "f"(S[92]), "f"(k_regs[28]), "f"(accp2_2_0_x));
        float _fma_1573;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1573)
            : "f"(S[93]), "f"(k_regs[29]), "f"(accp2_2_0_y));
        accp2_2_0_x = _fma_1572;
        accp2_2_0_y = _fma_1573;
        float _fma_1574;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1574)
            : "f"(S[94]), "f"(k_regs[30]), "f"(accp2_2_1_x));
        float _fma_1575;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1575)
            : "f"(S[95]), "f"(k_regs[31]), "f"(accp2_2_1_y));
        accp2_2_1_x = _fma_1574;
        accp2_2_1_y = _fma_1575;
        float _add_147;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_147) : "f"(accp2_2_0_x), "f"(accp2_2_1_x));
        float _add_148;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_148) : "f"(accp2_2_0_y), "f"(accp2_2_1_y));
        float dotp2_2 = _add_147 + _add_148;
        float _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, dotp2_2, 1);
        dotp2_2 = dotp2_2 + _shfl_xor_sync_149;
        float _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, dotp2_2, 2);
        dotp2_2 = dotp2_2 + _shfl_xor_sync_150;
#else
        float2 vn2p2_1 = make_float2(vnp2_1, vnp2_1);
        S2[16] = fma_f32x2_rn_ftz(k2[0], vn2p2_1, S2[16]);
        S2[17] = fma_f32x2_rn_ftz(k2[1], vn2p2_1, S2[17]);
        S2[18] = fma_f32x2_rn_ftz(k2[2], vn2p2_1, S2[18]);
        S2[19] = fma_f32x2_rn_ftz(k2[3], vn2p2_1, S2[19]);
        S2[20] = fma_f32x2_rn_ftz(k2[4], vn2p2_1, S2[20]);
        S2[21] = fma_f32x2_rn_ftz(k2[5], vn2p2_1, S2[21]);
        S2[22] = fma_f32x2_rn_ftz(k2[6], vn2p2_1, S2[22]);
        S2[23] = fma_f32x2_rn_ftz(k2[7], vn2p2_1, S2[23]);
        S2[24] = fma_f32x2_rn_ftz(k2[8], vn2p2_1, S2[24]);
        S2[25] = fma_f32x2_rn_ftz(k2[9], vn2p2_1, S2[25]);
        S2[26] = fma_f32x2_rn_ftz(k2[10], vn2p2_1, S2[26]);
        S2[27] = fma_f32x2_rn_ftz(k2[11], vn2p2_1, S2[27]);
        S2[28] = fma_f32x2_rn_ftz(k2[12], vn2p2_1, S2[28]);
        S2[29] = fma_f32x2_rn_ftz(k2[13], vn2p2_1, S2[29]);
        S2[30] = fma_f32x2_rn_ftz(k2[14], vn2p2_1, S2[30]);
        S2[31] = fma_f32x2_rn_ftz(k2[15], vn2p2_1, S2[31]);
        float2 accp2_2_0 = make_float2(0.0f, 0.0f);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[32], k2[0], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[33], k2[1], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[34], k2[2], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[35], k2[3], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[36], k2[4], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[37], k2[5], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[38], k2[6], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[39], k2[7], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[40], k2[8], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[41], k2[9], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[42], k2[10], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[43], k2[11], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[44], k2[12], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[45], k2[13], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[46], k2[14], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[47], k2[15], accp2_2_0);
        float dotp2_2 = accp2_2_0.x + accp2_2_0.y;
        float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, dotp2_2, 1);
        dotp2_2 = dotp2_2 + _shfl_xor_sync_95;
        float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, dotp2_2, 2);
        dotp2_2 = dotp2_2 + _shfl_xor_sync_96;
#endif
        float vnp2_2 = (v_regs[2] - dotp2_2) * betap2;
#if __CUDA_ARCH__ == 900
        float vn2p2_2_x = vnp2_2;
        float vn2p2_2_y = vnp2_2;
        float _fma_1576;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1576)
            : "f"(k_regs[0]), "f"(vn2p2_2_x), "f"(S[64]));
        float _fma_1577;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1577)
            : "f"(k_regs[1]), "f"(vn2p2_2_y), "f"(S[65]));
        S[64] = _fma_1576;
        S[65] = _fma_1577;
        float _fma_1578;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1578)
            : "f"(k_regs[2]), "f"(vn2p2_2_x), "f"(S[66]));
        float _fma_1579;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1579)
            : "f"(k_regs[3]), "f"(vn2p2_2_y), "f"(S[67]));
        S[66] = _fma_1578;
        S[67] = _fma_1579;
        float _fma_1580;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1580)
            : "f"(k_regs[4]), "f"(vn2p2_2_x), "f"(S[68]));
        float _fma_1581;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1581)
            : "f"(k_regs[5]), "f"(vn2p2_2_y), "f"(S[69]));
        S[68] = _fma_1580;
        S[69] = _fma_1581;
        float _fma_1582;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1582)
            : "f"(k_regs[6]), "f"(vn2p2_2_x), "f"(S[70]));
        float _fma_1583;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1583)
            : "f"(k_regs[7]), "f"(vn2p2_2_y), "f"(S[71]));
        S[70] = _fma_1582;
        S[71] = _fma_1583;
        float _fma_1584;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1584)
            : "f"(k_regs[8]), "f"(vn2p2_2_x), "f"(S[72]));
        float _fma_1585;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1585)
            : "f"(k_regs[9]), "f"(vn2p2_2_y), "f"(S[73]));
        S[72] = _fma_1584;
        S[73] = _fma_1585;
        float _fma_1586;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1586)
            : "f"(k_regs[10]), "f"(vn2p2_2_x), "f"(S[74]));
        float _fma_1587;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1587)
            : "f"(k_regs[11]), "f"(vn2p2_2_y), "f"(S[75]));
        S[74] = _fma_1586;
        S[75] = _fma_1587;
        float _fma_1588;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1588)
            : "f"(k_regs[12]), "f"(vn2p2_2_x), "f"(S[76]));
        float _fma_1589;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1589)
            : "f"(k_regs[13]), "f"(vn2p2_2_y), "f"(S[77]));
        S[76] = _fma_1588;
        S[77] = _fma_1589;
        float _fma_1590;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1590)
            : "f"(k_regs[14]), "f"(vn2p2_2_x), "f"(S[78]));
        float _fma_1591;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1591)
            : "f"(k_regs[15]), "f"(vn2p2_2_y), "f"(S[79]));
        S[78] = _fma_1590;
        S[79] = _fma_1591;
        float _fma_1592;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1592)
            : "f"(k_regs[16]), "f"(vn2p2_2_x), "f"(S[80]));
        float _fma_1593;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1593)
            : "f"(k_regs[17]), "f"(vn2p2_2_y), "f"(S[81]));
        S[80] = _fma_1592;
        S[81] = _fma_1593;
        float _fma_1594;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1594)
            : "f"(k_regs[18]), "f"(vn2p2_2_x), "f"(S[82]));
        float _fma_1595;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1595)
            : "f"(k_regs[19]), "f"(vn2p2_2_y), "f"(S[83]));
        S[82] = _fma_1594;
        S[83] = _fma_1595;
        float _fma_1596;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1596)
            : "f"(k_regs[20]), "f"(vn2p2_2_x), "f"(S[84]));
        float _fma_1597;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1597)
            : "f"(k_regs[21]), "f"(vn2p2_2_y), "f"(S[85]));
        S[84] = _fma_1596;
        S[85] = _fma_1597;
        float _fma_1598;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1598)
            : "f"(k_regs[22]), "f"(vn2p2_2_x), "f"(S[86]));
        float _fma_1599;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1599)
            : "f"(k_regs[23]), "f"(vn2p2_2_y), "f"(S[87]));
        S[86] = _fma_1598;
        S[87] = _fma_1599;
        float _fma_1600;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1600)
            : "f"(k_regs[24]), "f"(vn2p2_2_x), "f"(S[88]));
        float _fma_1601;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1601)
            : "f"(k_regs[25]), "f"(vn2p2_2_y), "f"(S[89]));
        S[88] = _fma_1600;
        S[89] = _fma_1601;
        float _fma_1602;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1602)
            : "f"(k_regs[26]), "f"(vn2p2_2_x), "f"(S[90]));
        float _fma_1603;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1603)
            : "f"(k_regs[27]), "f"(vn2p2_2_y), "f"(S[91]));
        S[90] = _fma_1602;
        S[91] = _fma_1603;
        float _fma_1604;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1604)
            : "f"(k_regs[28]), "f"(vn2p2_2_x), "f"(S[92]));
        float _fma_1605;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1605)
            : "f"(k_regs[29]), "f"(vn2p2_2_y), "f"(S[93]));
        S[92] = _fma_1604;
        S[93] = _fma_1605;
        float _fma_1606;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1606)
            : "f"(k_regs[30]), "f"(vn2p2_2_x), "f"(S[94]));
        float _fma_1607;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1607)
            : "f"(k_regs[31]), "f"(vn2p2_2_y), "f"(S[95]));
        S[94] = _fma_1606;
        S[95] = _fma_1607;
        float accp2_3_0_x = 0.0f;
        float accp2_3_0_y = 0.0f;
        float accp2_3_1_x = 0.0f;
        float accp2_3_1_y = 0.0f;
        float _fma_1608;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1608)
            : "f"(S[96]), "f"(k_regs[0]), "f"(accp2_3_0_x));
        float _fma_1609;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1609)
            : "f"(S[97]), "f"(k_regs[1]), "f"(accp2_3_0_y));
        accp2_3_0_x = _fma_1608;
        accp2_3_0_y = _fma_1609;
        float _fma_1610;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1610)
            : "f"(S[98]), "f"(k_regs[2]), "f"(accp2_3_1_x));
        float _fma_1611;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1611)
            : "f"(S[99]), "f"(k_regs[3]), "f"(accp2_3_1_y));
        accp2_3_1_x = _fma_1610;
        accp2_3_1_y = _fma_1611;
        float _fma_1612;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1612)
            : "f"(S[100]), "f"(k_regs[4]), "f"(accp2_3_0_x));
        float _fma_1613;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1613)
            : "f"(S[101]), "f"(k_regs[5]), "f"(accp2_3_0_y));
        accp2_3_0_x = _fma_1612;
        accp2_3_0_y = _fma_1613;
        float _fma_1614;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1614)
            : "f"(S[102]), "f"(k_regs[6]), "f"(accp2_3_1_x));
        float _fma_1615;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1615)
            : "f"(S[103]), "f"(k_regs[7]), "f"(accp2_3_1_y));
        accp2_3_1_x = _fma_1614;
        accp2_3_1_y = _fma_1615;
        float _fma_1616;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1616)
            : "f"(S[104]), "f"(k_regs[8]), "f"(accp2_3_0_x));
        float _fma_1617;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1617)
            : "f"(S[105]), "f"(k_regs[9]), "f"(accp2_3_0_y));
        accp2_3_0_x = _fma_1616;
        accp2_3_0_y = _fma_1617;
        float _fma_1618;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1618)
            : "f"(S[106]), "f"(k_regs[10]), "f"(accp2_3_1_x));
        float _fma_1619;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1619)
            : "f"(S[107]), "f"(k_regs[11]), "f"(accp2_3_1_y));
        accp2_3_1_x = _fma_1618;
        accp2_3_1_y = _fma_1619;
        float _fma_1620;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1620)
            : "f"(S[108]), "f"(k_regs[12]), "f"(accp2_3_0_x));
        float _fma_1621;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1621)
            : "f"(S[109]), "f"(k_regs[13]), "f"(accp2_3_0_y));
        accp2_3_0_x = _fma_1620;
        accp2_3_0_y = _fma_1621;
        float _fma_1622;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1622)
            : "f"(S[110]), "f"(k_regs[14]), "f"(accp2_3_1_x));
        float _fma_1623;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1623)
            : "f"(S[111]), "f"(k_regs[15]), "f"(accp2_3_1_y));
        accp2_3_1_x = _fma_1622;
        accp2_3_1_y = _fma_1623;
        float _fma_1624;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1624)
            : "f"(S[112]), "f"(k_regs[16]), "f"(accp2_3_0_x));
        float _fma_1625;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1625)
            : "f"(S[113]), "f"(k_regs[17]), "f"(accp2_3_0_y));
        accp2_3_0_x = _fma_1624;
        accp2_3_0_y = _fma_1625;
        float _fma_1626;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1626)
            : "f"(S[114]), "f"(k_regs[18]), "f"(accp2_3_1_x));
        float _fma_1627;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1627)
            : "f"(S[115]), "f"(k_regs[19]), "f"(accp2_3_1_y));
        accp2_3_1_x = _fma_1626;
        accp2_3_1_y = _fma_1627;
        float _fma_1628;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1628)
            : "f"(S[116]), "f"(k_regs[20]), "f"(accp2_3_0_x));
        float _fma_1629;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1629)
            : "f"(S[117]), "f"(k_regs[21]), "f"(accp2_3_0_y));
        accp2_3_0_x = _fma_1628;
        accp2_3_0_y = _fma_1629;
        float _fma_1630;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1630)
            : "f"(S[118]), "f"(k_regs[22]), "f"(accp2_3_1_x));
        float _fma_1631;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1631)
            : "f"(S[119]), "f"(k_regs[23]), "f"(accp2_3_1_y));
        accp2_3_1_x = _fma_1630;
        accp2_3_1_y = _fma_1631;
        float _fma_1632;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1632)
            : "f"(S[120]), "f"(k_regs[24]), "f"(accp2_3_0_x));
        float _fma_1633;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1633)
            : "f"(S[121]), "f"(k_regs[25]), "f"(accp2_3_0_y));
        accp2_3_0_x = _fma_1632;
        accp2_3_0_y = _fma_1633;
        float _fma_1634;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1634)
            : "f"(S[122]), "f"(k_regs[26]), "f"(accp2_3_1_x));
        float _fma_1635;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1635)
            : "f"(S[123]), "f"(k_regs[27]), "f"(accp2_3_1_y));
        accp2_3_1_x = _fma_1634;
        accp2_3_1_y = _fma_1635;
        float _fma_1636;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1636)
            : "f"(S[124]), "f"(k_regs[28]), "f"(accp2_3_0_x));
        float _fma_1637;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1637)
            : "f"(S[125]), "f"(k_regs[29]), "f"(accp2_3_0_y));
        accp2_3_0_x = _fma_1636;
        accp2_3_0_y = _fma_1637;
        float _fma_1638;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1638)
            : "f"(S[126]), "f"(k_regs[30]), "f"(accp2_3_1_x));
        float _fma_1639;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1639)
            : "f"(S[127]), "f"(k_regs[31]), "f"(accp2_3_1_y));
        accp2_3_1_x = _fma_1638;
        accp2_3_1_y = _fma_1639;
        float _add_151;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_151) : "f"(accp2_3_0_x), "f"(accp2_3_1_x));
        float _add_152;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_152) : "f"(accp2_3_0_y), "f"(accp2_3_1_y));
        float dotp2_3 = _add_151 + _add_152;
        float _shfl_xor_sync_153 = __shfl_xor_sync(0xFFFFFFFF, dotp2_3, 1);
        dotp2_3 = dotp2_3 + _shfl_xor_sync_153;
        float _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, dotp2_3, 2);
        dotp2_3 = dotp2_3 + _shfl_xor_sync_154;
#else
        float2 vn2p2_2 = make_float2(vnp2_2, vnp2_2);
        S2[32] = fma_f32x2_rn_ftz(k2[0], vn2p2_2, S2[32]);
        S2[33] = fma_f32x2_rn_ftz(k2[1], vn2p2_2, S2[33]);
        S2[34] = fma_f32x2_rn_ftz(k2[2], vn2p2_2, S2[34]);
        S2[35] = fma_f32x2_rn_ftz(k2[3], vn2p2_2, S2[35]);
        S2[36] = fma_f32x2_rn_ftz(k2[4], vn2p2_2, S2[36]);
        S2[37] = fma_f32x2_rn_ftz(k2[5], vn2p2_2, S2[37]);
        S2[38] = fma_f32x2_rn_ftz(k2[6], vn2p2_2, S2[38]);
        S2[39] = fma_f32x2_rn_ftz(k2[7], vn2p2_2, S2[39]);
        S2[40] = fma_f32x2_rn_ftz(k2[8], vn2p2_2, S2[40]);
        S2[41] = fma_f32x2_rn_ftz(k2[9], vn2p2_2, S2[41]);
        S2[42] = fma_f32x2_rn_ftz(k2[10], vn2p2_2, S2[42]);
        S2[43] = fma_f32x2_rn_ftz(k2[11], vn2p2_2, S2[43]);
        S2[44] = fma_f32x2_rn_ftz(k2[12], vn2p2_2, S2[44]);
        S2[45] = fma_f32x2_rn_ftz(k2[13], vn2p2_2, S2[45]);
        S2[46] = fma_f32x2_rn_ftz(k2[14], vn2p2_2, S2[46]);
        S2[47] = fma_f32x2_rn_ftz(k2[15], vn2p2_2, S2[47]);
        float2 accp2_3_0 = make_float2(0.0f, 0.0f);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[48], k2[0], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[49], k2[1], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[50], k2[2], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[51], k2[3], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[52], k2[4], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[53], k2[5], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[54], k2[6], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[55], k2[7], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[56], k2[8], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[57], k2[9], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[58], k2[10], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[59], k2[11], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[60], k2[12], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[61], k2[13], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[62], k2[14], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[63], k2[15], accp2_3_0);
        float dotp2_3 = accp2_3_0.x + accp2_3_0.y;
        float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, dotp2_3, 1);
        dotp2_3 = dotp2_3 + _shfl_xor_sync_97;
        float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, dotp2_3, 2);
        dotp2_3 = dotp2_3 + _shfl_xor_sync_98;
#endif
        float vnp2_3 = (v_regs[3] - dotp2_3) * betap2;
#if __CUDA_ARCH__ == 900
        float vn2p2_3_x = vnp2_3;
        float vn2p2_3_y = vnp2_3;
        float _fma_1640;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1640)
            : "f"(k_regs[0]), "f"(vn2p2_3_x), "f"(S[96]));
        float _fma_1641;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1641)
            : "f"(k_regs[1]), "f"(vn2p2_3_y), "f"(S[97]));
        S[96] = _fma_1640;
        S[97] = _fma_1641;
        float _fma_1642;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1642)
            : "f"(k_regs[2]), "f"(vn2p2_3_x), "f"(S[98]));
        float _fma_1643;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1643)
            : "f"(k_regs[3]), "f"(vn2p2_3_y), "f"(S[99]));
        S[98] = _fma_1642;
        S[99] = _fma_1643;
        float _fma_1644;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1644)
            : "f"(k_regs[4]), "f"(vn2p2_3_x), "f"(S[100]));
        float _fma_1645;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1645)
            : "f"(k_regs[5]), "f"(vn2p2_3_y), "f"(S[101]));
        S[100] = _fma_1644;
        S[101] = _fma_1645;
        float _fma_1646;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1646)
            : "f"(k_regs[6]), "f"(vn2p2_3_x), "f"(S[102]));
        float _fma_1647;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1647)
            : "f"(k_regs[7]), "f"(vn2p2_3_y), "f"(S[103]));
        S[102] = _fma_1646;
        S[103] = _fma_1647;
        float _fma_1648;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1648)
            : "f"(k_regs[8]), "f"(vn2p2_3_x), "f"(S[104]));
        float _fma_1649;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1649)
            : "f"(k_regs[9]), "f"(vn2p2_3_y), "f"(S[105]));
        S[104] = _fma_1648;
        S[105] = _fma_1649;
        float _fma_1650;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1650)
            : "f"(k_regs[10]), "f"(vn2p2_3_x), "f"(S[106]));
        float _fma_1651;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1651)
            : "f"(k_regs[11]), "f"(vn2p2_3_y), "f"(S[107]));
        S[106] = _fma_1650;
        S[107] = _fma_1651;
        float _fma_1652;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1652)
            : "f"(k_regs[12]), "f"(vn2p2_3_x), "f"(S[108]));
        float _fma_1653;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1653)
            : "f"(k_regs[13]), "f"(vn2p2_3_y), "f"(S[109]));
        S[108] = _fma_1652;
        S[109] = _fma_1653;
        float _fma_1654;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1654)
            : "f"(k_regs[14]), "f"(vn2p2_3_x), "f"(S[110]));
        float _fma_1655;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1655)
            : "f"(k_regs[15]), "f"(vn2p2_3_y), "f"(S[111]));
        S[110] = _fma_1654;
        S[111] = _fma_1655;
        float _fma_1656;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1656)
            : "f"(k_regs[16]), "f"(vn2p2_3_x), "f"(S[112]));
        float _fma_1657;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1657)
            : "f"(k_regs[17]), "f"(vn2p2_3_y), "f"(S[113]));
        S[112] = _fma_1656;
        S[113] = _fma_1657;
        float _fma_1658;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1658)
            : "f"(k_regs[18]), "f"(vn2p2_3_x), "f"(S[114]));
        float _fma_1659;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1659)
            : "f"(k_regs[19]), "f"(vn2p2_3_y), "f"(S[115]));
        S[114] = _fma_1658;
        S[115] = _fma_1659;
        float _fma_1660;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1660)
            : "f"(k_regs[20]), "f"(vn2p2_3_x), "f"(S[116]));
        float _fma_1661;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1661)
            : "f"(k_regs[21]), "f"(vn2p2_3_y), "f"(S[117]));
        S[116] = _fma_1660;
        S[117] = _fma_1661;
        float _fma_1662;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1662)
            : "f"(k_regs[22]), "f"(vn2p2_3_x), "f"(S[118]));
        float _fma_1663;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1663)
            : "f"(k_regs[23]), "f"(vn2p2_3_y), "f"(S[119]));
        S[118] = _fma_1662;
        S[119] = _fma_1663;
        float _fma_1664;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1664)
            : "f"(k_regs[24]), "f"(vn2p2_3_x), "f"(S[120]));
        float _fma_1665;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1665)
            : "f"(k_regs[25]), "f"(vn2p2_3_y), "f"(S[121]));
        S[120] = _fma_1664;
        S[121] = _fma_1665;
        float _fma_1666;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1666)
            : "f"(k_regs[26]), "f"(vn2p2_3_x), "f"(S[122]));
        float _fma_1667;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1667)
            : "f"(k_regs[27]), "f"(vn2p2_3_y), "f"(S[123]));
        S[122] = _fma_1666;
        S[123] = _fma_1667;
        float _fma_1668;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1668)
            : "f"(k_regs[28]), "f"(vn2p2_3_x), "f"(S[124]));
        float _fma_1669;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1669)
            : "f"(k_regs[29]), "f"(vn2p2_3_y), "f"(S[125]));
        S[124] = _fma_1668;
        S[125] = _fma_1669;
        float _fma_1670;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1670)
            : "f"(k_regs[30]), "f"(vn2p2_3_x), "f"(S[126]));
        float _fma_1671;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1671)
            : "f"(k_regs[31]), "f"(vn2p2_3_y), "f"(S[127]));
        S[126] = _fma_1670;
        S[127] = _fma_1671;
#else
        float2 vn2p2_3 = make_float2(vnp2_3, vnp2_3);
        S2[48] = fma_f32x2_rn_ftz(k2[0], vn2p2_3, S2[48]);
        S2[49] = fma_f32x2_rn_ftz(k2[1], vn2p2_3, S2[49]);
        S2[50] = fma_f32x2_rn_ftz(k2[2], vn2p2_3, S2[50]);
        S2[51] = fma_f32x2_rn_ftz(k2[3], vn2p2_3, S2[51]);
        S2[52] = fma_f32x2_rn_ftz(k2[4], vn2p2_3, S2[52]);
        S2[53] = fma_f32x2_rn_ftz(k2[5], vn2p2_3, S2[53]);
        S2[54] = fma_f32x2_rn_ftz(k2[6], vn2p2_3, S2[54]);
        S2[55] = fma_f32x2_rn_ftz(k2[7], vn2p2_3, S2[55]);
        S2[56] = fma_f32x2_rn_ftz(k2[8], vn2p2_3, S2[56]);
        S2[57] = fma_f32x2_rn_ftz(k2[9], vn2p2_3, S2[57]);
        S2[58] = fma_f32x2_rn_ftz(k2[10], vn2p2_3, S2[58]);
        S2[59] = fma_f32x2_rn_ftz(k2[11], vn2p2_3, S2[59]);
        S2[60] = fma_f32x2_rn_ftz(k2[12], vn2p2_3, S2[60]);
        S2[61] = fma_f32x2_rn_ftz(k2[13], vn2p2_3, S2[61]);
        S2[62] = fma_f32x2_rn_ftz(k2[14], vn2p2_3, S2[62]);
        S2[63] = fma_f32x2_rn_ftz(k2[15], vn2p2_3, S2[63]);
#endif
        unsigned int qbasep = sQ_addr + (unsigned int)(t_2 * 512) + lane_byte;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                     : "r"(qbasep));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                     : "r"(qbasep + 64));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                     : "r"(qbasep + 128));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                     : "r"(qbasep + 192));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[16])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 3]))
                     : "r"(qbasep + 256));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[20])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 3]))
                     : "r"(qbasep + 320));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[24])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 3]))
                     : "r"(qbasep + 384));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[28])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 3]))
                     : "r"(qbasep + 448));
#if __CUDA_ARCH__ == 900
        float accpq_0_0_x = 0.0f;
        float accpq_0_0_y = 0.0f;
        float accpq_0_1_x = 0.0f;
        float accpq_0_1_y = 0.0f;
        float _fma_1672;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1672)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accpq_0_0_x));
        float _fma_1673;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1673)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accpq_0_0_y));
        accpq_0_0_x = _fma_1672;
        accpq_0_0_y = _fma_1673;
        float _fma_1674;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1674)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accpq_0_1_x));
        float _fma_1675;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1675)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accpq_0_1_y));
        accpq_0_1_x = _fma_1674;
        accpq_0_1_y = _fma_1675;
        float _fma_1676;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1676)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accpq_0_0_x));
        float _fma_1677;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1677)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accpq_0_0_y));
        accpq_0_0_x = _fma_1676;
        accpq_0_0_y = _fma_1677;
        float _fma_1678;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1678)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accpq_0_1_x));
        float _fma_1679;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1679)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accpq_0_1_y));
        accpq_0_1_x = _fma_1678;
        accpq_0_1_y = _fma_1679;
        float _fma_1680;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1680)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accpq_0_0_x));
        float _fma_1681;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1681)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accpq_0_0_y));
        accpq_0_0_x = _fma_1680;
        accpq_0_0_y = _fma_1681;
        float _fma_1682;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1682)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accpq_0_1_x));
        float _fma_1683;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1683)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accpq_0_1_y));
        accpq_0_1_x = _fma_1682;
        accpq_0_1_y = _fma_1683;
        float _fma_1684;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1684)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accpq_0_0_x));
        float _fma_1685;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1685)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accpq_0_0_y));
        accpq_0_0_x = _fma_1684;
        accpq_0_0_y = _fma_1685;
        float _fma_1686;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1686)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accpq_0_1_x));
        float _fma_1687;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1687)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accpq_0_1_y));
        accpq_0_1_x = _fma_1686;
        accpq_0_1_y = _fma_1687;
        float _fma_1688;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1688)
            : "f"(S[16]), "f"(k_regs[16]), "f"(accpq_0_0_x));
        float _fma_1689;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1689)
            : "f"(S[17]), "f"(k_regs[17]), "f"(accpq_0_0_y));
        accpq_0_0_x = _fma_1688;
        accpq_0_0_y = _fma_1689;
        float _fma_1690;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1690)
            : "f"(S[18]), "f"(k_regs[18]), "f"(accpq_0_1_x));
        float _fma_1691;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1691)
            : "f"(S[19]), "f"(k_regs[19]), "f"(accpq_0_1_y));
        accpq_0_1_x = _fma_1690;
        accpq_0_1_y = _fma_1691;
        float _fma_1692;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1692)
            : "f"(S[20]), "f"(k_regs[20]), "f"(accpq_0_0_x));
        float _fma_1693;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1693)
            : "f"(S[21]), "f"(k_regs[21]), "f"(accpq_0_0_y));
        accpq_0_0_x = _fma_1692;
        accpq_0_0_y = _fma_1693;
        float _fma_1694;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1694)
            : "f"(S[22]), "f"(k_regs[22]), "f"(accpq_0_1_x));
        float _fma_1695;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1695)
            : "f"(S[23]), "f"(k_regs[23]), "f"(accpq_0_1_y));
        accpq_0_1_x = _fma_1694;
        accpq_0_1_y = _fma_1695;
        float _fma_1696;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1696)
            : "f"(S[24]), "f"(k_regs[24]), "f"(accpq_0_0_x));
        float _fma_1697;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1697)
            : "f"(S[25]), "f"(k_regs[25]), "f"(accpq_0_0_y));
        accpq_0_0_x = _fma_1696;
        accpq_0_0_y = _fma_1697;
        float _fma_1698;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1698)
            : "f"(S[26]), "f"(k_regs[26]), "f"(accpq_0_1_x));
        float _fma_1699;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1699)
            : "f"(S[27]), "f"(k_regs[27]), "f"(accpq_0_1_y));
        accpq_0_1_x = _fma_1698;
        accpq_0_1_y = _fma_1699;
        float _fma_1700;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1700)
            : "f"(S[28]), "f"(k_regs[28]), "f"(accpq_0_0_x));
        float _fma_1701;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1701)
            : "f"(S[29]), "f"(k_regs[29]), "f"(accpq_0_0_y));
        accpq_0_0_x = _fma_1700;
        accpq_0_0_y = _fma_1701;
        float _fma_1702;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1702)
            : "f"(S[30]), "f"(k_regs[30]), "f"(accpq_0_1_x));
        float _fma_1703;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1703)
            : "f"(S[31]), "f"(k_regs[31]), "f"(accpq_0_1_y));
        accpq_0_1_x = _fma_1702;
        accpq_0_1_y = _fma_1703;
        float _add_155;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_155) : "f"(accpq_0_0_x), "f"(accpq_0_1_x));
        float _add_156;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_156) : "f"(accpq_0_0_y), "f"(accpq_0_1_y));
        float dotpq_0 = _add_155 + _add_156;
        float _shfl_xor_sync_157 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 1);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_157;
        float _shfl_xor_sync_158 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 2);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_158;
#else
        k2[0] = make_float2(k_regs[0], k_regs[1]);
        k2[1] = make_float2(k_regs[2], k_regs[3]);
        k2[2] = make_float2(k_regs[4], k_regs[5]);
        k2[3] = make_float2(k_regs[6], k_regs[7]);
        k2[4] = make_float2(k_regs[8], k_regs[9]);
        k2[5] = make_float2(k_regs[10], k_regs[11]);
        k2[6] = make_float2(k_regs[12], k_regs[13]);
        k2[7] = make_float2(k_regs[14], k_regs[15]);
        k2[8] = make_float2(k_regs[16], k_regs[17]);
        k2[9] = make_float2(k_regs[18], k_regs[19]);
        k2[10] = make_float2(k_regs[20], k_regs[21]);
        k2[11] = make_float2(k_regs[22], k_regs[23]);
        k2[12] = make_float2(k_regs[24], k_regs[25]);
        k2[13] = make_float2(k_regs[26], k_regs[27]);
        k2[14] = make_float2(k_regs[28], k_regs[29]);
        k2[15] = make_float2(k_regs[30], k_regs[31]);
        float2 accpq_0_0 = make_float2(0.0f, 0.0f);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[1], k2[1], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[3], k2[3], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[5], k2[5], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[7], k2[7], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[8], k2[8], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[9], k2[9], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[10], k2[10], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[11], k2[11], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[12], k2[12], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[13], k2[13], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[14], k2[14], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[15], k2[15], accpq_0_0);
        float dotpq_0 = accpq_0_0.x + accpq_0_0.y;
        float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 1);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_99;
        float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 2);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_100;
#endif
        o4[0] = dotpq_0;
#if __CUDA_ARCH__ == 900
        float accpq_1_0_x = 0.0f;
        float accpq_1_0_y = 0.0f;
        float accpq_1_1_x = 0.0f;
        float accpq_1_1_y = 0.0f;
        float _fma_1704;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1704)
            : "f"(S[32]), "f"(k_regs[0]), "f"(accpq_1_0_x));
        float _fma_1705;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1705)
            : "f"(S[33]), "f"(k_regs[1]), "f"(accpq_1_0_y));
        accpq_1_0_x = _fma_1704;
        accpq_1_0_y = _fma_1705;
        float _fma_1706;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1706)
            : "f"(S[34]), "f"(k_regs[2]), "f"(accpq_1_1_x));
        float _fma_1707;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1707)
            : "f"(S[35]), "f"(k_regs[3]), "f"(accpq_1_1_y));
        accpq_1_1_x = _fma_1706;
        accpq_1_1_y = _fma_1707;
        float _fma_1708;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1708)
            : "f"(S[36]), "f"(k_regs[4]), "f"(accpq_1_0_x));
        float _fma_1709;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1709)
            : "f"(S[37]), "f"(k_regs[5]), "f"(accpq_1_0_y));
        accpq_1_0_x = _fma_1708;
        accpq_1_0_y = _fma_1709;
        float _fma_1710;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1710)
            : "f"(S[38]), "f"(k_regs[6]), "f"(accpq_1_1_x));
        float _fma_1711;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1711)
            : "f"(S[39]), "f"(k_regs[7]), "f"(accpq_1_1_y));
        accpq_1_1_x = _fma_1710;
        accpq_1_1_y = _fma_1711;
        float _fma_1712;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1712)
            : "f"(S[40]), "f"(k_regs[8]), "f"(accpq_1_0_x));
        float _fma_1713;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1713)
            : "f"(S[41]), "f"(k_regs[9]), "f"(accpq_1_0_y));
        accpq_1_0_x = _fma_1712;
        accpq_1_0_y = _fma_1713;
        float _fma_1714;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1714)
            : "f"(S[42]), "f"(k_regs[10]), "f"(accpq_1_1_x));
        float _fma_1715;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1715)
            : "f"(S[43]), "f"(k_regs[11]), "f"(accpq_1_1_y));
        accpq_1_1_x = _fma_1714;
        accpq_1_1_y = _fma_1715;
        float _fma_1716;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1716)
            : "f"(S[44]), "f"(k_regs[12]), "f"(accpq_1_0_x));
        float _fma_1717;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1717)
            : "f"(S[45]), "f"(k_regs[13]), "f"(accpq_1_0_y));
        accpq_1_0_x = _fma_1716;
        accpq_1_0_y = _fma_1717;
        float _fma_1718;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1718)
            : "f"(S[46]), "f"(k_regs[14]), "f"(accpq_1_1_x));
        float _fma_1719;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1719)
            : "f"(S[47]), "f"(k_regs[15]), "f"(accpq_1_1_y));
        accpq_1_1_x = _fma_1718;
        accpq_1_1_y = _fma_1719;
        float _fma_1720;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1720)
            : "f"(S[48]), "f"(k_regs[16]), "f"(accpq_1_0_x));
        float _fma_1721;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1721)
            : "f"(S[49]), "f"(k_regs[17]), "f"(accpq_1_0_y));
        accpq_1_0_x = _fma_1720;
        accpq_1_0_y = _fma_1721;
        float _fma_1722;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1722)
            : "f"(S[50]), "f"(k_regs[18]), "f"(accpq_1_1_x));
        float _fma_1723;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1723)
            : "f"(S[51]), "f"(k_regs[19]), "f"(accpq_1_1_y));
        accpq_1_1_x = _fma_1722;
        accpq_1_1_y = _fma_1723;
        float _fma_1724;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1724)
            : "f"(S[52]), "f"(k_regs[20]), "f"(accpq_1_0_x));
        float _fma_1725;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1725)
            : "f"(S[53]), "f"(k_regs[21]), "f"(accpq_1_0_y));
        accpq_1_0_x = _fma_1724;
        accpq_1_0_y = _fma_1725;
        float _fma_1726;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1726)
            : "f"(S[54]), "f"(k_regs[22]), "f"(accpq_1_1_x));
        float _fma_1727;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1727)
            : "f"(S[55]), "f"(k_regs[23]), "f"(accpq_1_1_y));
        accpq_1_1_x = _fma_1726;
        accpq_1_1_y = _fma_1727;
        float _fma_1728;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1728)
            : "f"(S[56]), "f"(k_regs[24]), "f"(accpq_1_0_x));
        float _fma_1729;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1729)
            : "f"(S[57]), "f"(k_regs[25]), "f"(accpq_1_0_y));
        accpq_1_0_x = _fma_1728;
        accpq_1_0_y = _fma_1729;
        float _fma_1730;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1730)
            : "f"(S[58]), "f"(k_regs[26]), "f"(accpq_1_1_x));
        float _fma_1731;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1731)
            : "f"(S[59]), "f"(k_regs[27]), "f"(accpq_1_1_y));
        accpq_1_1_x = _fma_1730;
        accpq_1_1_y = _fma_1731;
        float _fma_1732;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1732)
            : "f"(S[60]), "f"(k_regs[28]), "f"(accpq_1_0_x));
        float _fma_1733;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1733)
            : "f"(S[61]), "f"(k_regs[29]), "f"(accpq_1_0_y));
        accpq_1_0_x = _fma_1732;
        accpq_1_0_y = _fma_1733;
        float _fma_1734;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1734)
            : "f"(S[62]), "f"(k_regs[30]), "f"(accpq_1_1_x));
        float _fma_1735;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1735)
            : "f"(S[63]), "f"(k_regs[31]), "f"(accpq_1_1_y));
        accpq_1_1_x = _fma_1734;
        accpq_1_1_y = _fma_1735;
        float _add_159;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_159) : "f"(accpq_1_0_x), "f"(accpq_1_1_x));
        float _add_160;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_160) : "f"(accpq_1_0_y), "f"(accpq_1_1_y));
        float dotpq_1 = _add_159 + _add_160;
        float _shfl_xor_sync_161 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 1);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_161;
        float _shfl_xor_sync_162 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 2);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_162;
#else
        float2 accpq_1_0 = make_float2(0.0f, 0.0f);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[24], k2[8], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[25], k2[9], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[26], k2[10], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[27], k2[11], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[28], k2[12], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[29], k2[13], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[30], k2[14], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[31], k2[15], accpq_1_0);
        float dotpq_1 = accpq_1_0.x + accpq_1_0.y;
        float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 1);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_101;
        float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 2);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_102;
#endif
        o4[1] = dotpq_1;
#if __CUDA_ARCH__ == 900
        float accpq_2_0_x = 0.0f;
        float accpq_2_0_y = 0.0f;
        float accpq_2_1_x = 0.0f;
        float accpq_2_1_y = 0.0f;
        float _fma_1736;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1736)
            : "f"(S[64]), "f"(k_regs[0]), "f"(accpq_2_0_x));
        float _fma_1737;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1737)
            : "f"(S[65]), "f"(k_regs[1]), "f"(accpq_2_0_y));
        accpq_2_0_x = _fma_1736;
        accpq_2_0_y = _fma_1737;
        float _fma_1738;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1738)
            : "f"(S[66]), "f"(k_regs[2]), "f"(accpq_2_1_x));
        float _fma_1739;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1739)
            : "f"(S[67]), "f"(k_regs[3]), "f"(accpq_2_1_y));
        accpq_2_1_x = _fma_1738;
        accpq_2_1_y = _fma_1739;
        float _fma_1740;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1740)
            : "f"(S[68]), "f"(k_regs[4]), "f"(accpq_2_0_x));
        float _fma_1741;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1741)
            : "f"(S[69]), "f"(k_regs[5]), "f"(accpq_2_0_y));
        accpq_2_0_x = _fma_1740;
        accpq_2_0_y = _fma_1741;
        float _fma_1742;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1742)
            : "f"(S[70]), "f"(k_regs[6]), "f"(accpq_2_1_x));
        float _fma_1743;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1743)
            : "f"(S[71]), "f"(k_regs[7]), "f"(accpq_2_1_y));
        accpq_2_1_x = _fma_1742;
        accpq_2_1_y = _fma_1743;
        float _fma_1744;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1744)
            : "f"(S[72]), "f"(k_regs[8]), "f"(accpq_2_0_x));
        float _fma_1745;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1745)
            : "f"(S[73]), "f"(k_regs[9]), "f"(accpq_2_0_y));
        accpq_2_0_x = _fma_1744;
        accpq_2_0_y = _fma_1745;
        float _fma_1746;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1746)
            : "f"(S[74]), "f"(k_regs[10]), "f"(accpq_2_1_x));
        float _fma_1747;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1747)
            : "f"(S[75]), "f"(k_regs[11]), "f"(accpq_2_1_y));
        accpq_2_1_x = _fma_1746;
        accpq_2_1_y = _fma_1747;
        float _fma_1748;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1748)
            : "f"(S[76]), "f"(k_regs[12]), "f"(accpq_2_0_x));
        float _fma_1749;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1749)
            : "f"(S[77]), "f"(k_regs[13]), "f"(accpq_2_0_y));
        accpq_2_0_x = _fma_1748;
        accpq_2_0_y = _fma_1749;
        float _fma_1750;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1750)
            : "f"(S[78]), "f"(k_regs[14]), "f"(accpq_2_1_x));
        float _fma_1751;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1751)
            : "f"(S[79]), "f"(k_regs[15]), "f"(accpq_2_1_y));
        accpq_2_1_x = _fma_1750;
        accpq_2_1_y = _fma_1751;
        float _fma_1752;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1752)
            : "f"(S[80]), "f"(k_regs[16]), "f"(accpq_2_0_x));
        float _fma_1753;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1753)
            : "f"(S[81]), "f"(k_regs[17]), "f"(accpq_2_0_y));
        accpq_2_0_x = _fma_1752;
        accpq_2_0_y = _fma_1753;
        float _fma_1754;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1754)
            : "f"(S[82]), "f"(k_regs[18]), "f"(accpq_2_1_x));
        float _fma_1755;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1755)
            : "f"(S[83]), "f"(k_regs[19]), "f"(accpq_2_1_y));
        accpq_2_1_x = _fma_1754;
        accpq_2_1_y = _fma_1755;
        float _fma_1756;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1756)
            : "f"(S[84]), "f"(k_regs[20]), "f"(accpq_2_0_x));
        float _fma_1757;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1757)
            : "f"(S[85]), "f"(k_regs[21]), "f"(accpq_2_0_y));
        accpq_2_0_x = _fma_1756;
        accpq_2_0_y = _fma_1757;
        float _fma_1758;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1758)
            : "f"(S[86]), "f"(k_regs[22]), "f"(accpq_2_1_x));
        float _fma_1759;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1759)
            : "f"(S[87]), "f"(k_regs[23]), "f"(accpq_2_1_y));
        accpq_2_1_x = _fma_1758;
        accpq_2_1_y = _fma_1759;
        float _fma_1760;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1760)
            : "f"(S[88]), "f"(k_regs[24]), "f"(accpq_2_0_x));
        float _fma_1761;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1761)
            : "f"(S[89]), "f"(k_regs[25]), "f"(accpq_2_0_y));
        accpq_2_0_x = _fma_1760;
        accpq_2_0_y = _fma_1761;
        float _fma_1762;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1762)
            : "f"(S[90]), "f"(k_regs[26]), "f"(accpq_2_1_x));
        float _fma_1763;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1763)
            : "f"(S[91]), "f"(k_regs[27]), "f"(accpq_2_1_y));
        accpq_2_1_x = _fma_1762;
        accpq_2_1_y = _fma_1763;
        float _fma_1764;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1764)
            : "f"(S[92]), "f"(k_regs[28]), "f"(accpq_2_0_x));
        float _fma_1765;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1765)
            : "f"(S[93]), "f"(k_regs[29]), "f"(accpq_2_0_y));
        accpq_2_0_x = _fma_1764;
        accpq_2_0_y = _fma_1765;
        float _fma_1766;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1766)
            : "f"(S[94]), "f"(k_regs[30]), "f"(accpq_2_1_x));
        float _fma_1767;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1767)
            : "f"(S[95]), "f"(k_regs[31]), "f"(accpq_2_1_y));
        accpq_2_1_x = _fma_1766;
        accpq_2_1_y = _fma_1767;
        float _add_163;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_163) : "f"(accpq_2_0_x), "f"(accpq_2_1_x));
        float _add_164;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_164) : "f"(accpq_2_0_y), "f"(accpq_2_1_y));
        float dotpq_2 = _add_163 + _add_164;
        float _shfl_xor_sync_165 = __shfl_xor_sync(0xFFFFFFFF, dotpq_2, 1);
        dotpq_2 = dotpq_2 + _shfl_xor_sync_165;
        float _shfl_xor_sync_166 = __shfl_xor_sync(0xFFFFFFFF, dotpq_2, 2);
        dotpq_2 = dotpq_2 + _shfl_xor_sync_166;
#else
        float2 accpq_2_0 = make_float2(0.0f, 0.0f);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[32], k2[0], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[33], k2[1], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[34], k2[2], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[35], k2[3], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[36], k2[4], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[37], k2[5], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[38], k2[6], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[39], k2[7], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[40], k2[8], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[41], k2[9], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[42], k2[10], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[43], k2[11], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[44], k2[12], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[45], k2[13], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[46], k2[14], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[47], k2[15], accpq_2_0);
        float dotpq_2 = accpq_2_0.x + accpq_2_0.y;
        float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, dotpq_2, 1);
        dotpq_2 = dotpq_2 + _shfl_xor_sync_103;
        float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, dotpq_2, 2);
        dotpq_2 = dotpq_2 + _shfl_xor_sync_104;
#endif
        o4[2] = dotpq_2;
#if __CUDA_ARCH__ == 900
        float accpq_3_0_x = 0.0f;
        float accpq_3_0_y = 0.0f;
        float accpq_3_1_x = 0.0f;
        float accpq_3_1_y = 0.0f;
        float _fma_1768;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1768)
            : "f"(S[96]), "f"(k_regs[0]), "f"(accpq_3_0_x));
        float _fma_1769;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1769)
            : "f"(S[97]), "f"(k_regs[1]), "f"(accpq_3_0_y));
        accpq_3_0_x = _fma_1768;
        accpq_3_0_y = _fma_1769;
        float _fma_1770;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1770)
            : "f"(S[98]), "f"(k_regs[2]), "f"(accpq_3_1_x));
        float _fma_1771;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1771)
            : "f"(S[99]), "f"(k_regs[3]), "f"(accpq_3_1_y));
        accpq_3_1_x = _fma_1770;
        accpq_3_1_y = _fma_1771;
        float _fma_1772;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1772)
            : "f"(S[100]), "f"(k_regs[4]), "f"(accpq_3_0_x));
        float _fma_1773;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1773)
            : "f"(S[101]), "f"(k_regs[5]), "f"(accpq_3_0_y));
        accpq_3_0_x = _fma_1772;
        accpq_3_0_y = _fma_1773;
        float _fma_1774;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1774)
            : "f"(S[102]), "f"(k_regs[6]), "f"(accpq_3_1_x));
        float _fma_1775;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1775)
            : "f"(S[103]), "f"(k_regs[7]), "f"(accpq_3_1_y));
        accpq_3_1_x = _fma_1774;
        accpq_3_1_y = _fma_1775;
        float _fma_1776;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1776)
            : "f"(S[104]), "f"(k_regs[8]), "f"(accpq_3_0_x));
        float _fma_1777;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1777)
            : "f"(S[105]), "f"(k_regs[9]), "f"(accpq_3_0_y));
        accpq_3_0_x = _fma_1776;
        accpq_3_0_y = _fma_1777;
        float _fma_1778;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1778)
            : "f"(S[106]), "f"(k_regs[10]), "f"(accpq_3_1_x));
        float _fma_1779;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1779)
            : "f"(S[107]), "f"(k_regs[11]), "f"(accpq_3_1_y));
        accpq_3_1_x = _fma_1778;
        accpq_3_1_y = _fma_1779;
        float _fma_1780;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1780)
            : "f"(S[108]), "f"(k_regs[12]), "f"(accpq_3_0_x));
        float _fma_1781;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1781)
            : "f"(S[109]), "f"(k_regs[13]), "f"(accpq_3_0_y));
        accpq_3_0_x = _fma_1780;
        accpq_3_0_y = _fma_1781;
        float _fma_1782;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1782)
            : "f"(S[110]), "f"(k_regs[14]), "f"(accpq_3_1_x));
        float _fma_1783;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1783)
            : "f"(S[111]), "f"(k_regs[15]), "f"(accpq_3_1_y));
        accpq_3_1_x = _fma_1782;
        accpq_3_1_y = _fma_1783;
        float _fma_1784;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1784)
            : "f"(S[112]), "f"(k_regs[16]), "f"(accpq_3_0_x));
        float _fma_1785;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1785)
            : "f"(S[113]), "f"(k_regs[17]), "f"(accpq_3_0_y));
        accpq_3_0_x = _fma_1784;
        accpq_3_0_y = _fma_1785;
        float _fma_1786;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1786)
            : "f"(S[114]), "f"(k_regs[18]), "f"(accpq_3_1_x));
        float _fma_1787;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1787)
            : "f"(S[115]), "f"(k_regs[19]), "f"(accpq_3_1_y));
        accpq_3_1_x = _fma_1786;
        accpq_3_1_y = _fma_1787;
        float _fma_1788;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1788)
            : "f"(S[116]), "f"(k_regs[20]), "f"(accpq_3_0_x));
        float _fma_1789;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1789)
            : "f"(S[117]), "f"(k_regs[21]), "f"(accpq_3_0_y));
        accpq_3_0_x = _fma_1788;
        accpq_3_0_y = _fma_1789;
        float _fma_1790;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1790)
            : "f"(S[118]), "f"(k_regs[22]), "f"(accpq_3_1_x));
        float _fma_1791;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1791)
            : "f"(S[119]), "f"(k_regs[23]), "f"(accpq_3_1_y));
        accpq_3_1_x = _fma_1790;
        accpq_3_1_y = _fma_1791;
        float _fma_1792;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1792)
            : "f"(S[120]), "f"(k_regs[24]), "f"(accpq_3_0_x));
        float _fma_1793;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1793)
            : "f"(S[121]), "f"(k_regs[25]), "f"(accpq_3_0_y));
        accpq_3_0_x = _fma_1792;
        accpq_3_0_y = _fma_1793;
        float _fma_1794;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1794)
            : "f"(S[122]), "f"(k_regs[26]), "f"(accpq_3_1_x));
        float _fma_1795;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1795)
            : "f"(S[123]), "f"(k_regs[27]), "f"(accpq_3_1_y));
        accpq_3_1_x = _fma_1794;
        accpq_3_1_y = _fma_1795;
        float _fma_1796;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1796)
            : "f"(S[124]), "f"(k_regs[28]), "f"(accpq_3_0_x));
        float _fma_1797;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1797)
            : "f"(S[125]), "f"(k_regs[29]), "f"(accpq_3_0_y));
        accpq_3_0_x = _fma_1796;
        accpq_3_0_y = _fma_1797;
        float _fma_1798;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1798)
            : "f"(S[126]), "f"(k_regs[30]), "f"(accpq_3_1_x));
        float _fma_1799;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_1799)
            : "f"(S[127]), "f"(k_regs[31]), "f"(accpq_3_1_y));
        accpq_3_1_x = _fma_1798;
        accpq_3_1_y = _fma_1799;
        float _add_167;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_167) : "f"(accpq_3_0_x), "f"(accpq_3_1_x));
        float _add_168;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_168) : "f"(accpq_3_0_y), "f"(accpq_3_1_y));
        float dotpq_3 = _add_167 + _add_168;
        float _shfl_xor_sync_169 = __shfl_xor_sync(0xFFFFFFFF, dotpq_3, 1);
        dotpq_3 = dotpq_3 + _shfl_xor_sync_169;
        float _shfl_xor_sync_170 = __shfl_xor_sync(0xFFFFFFFF, dotpq_3, 2);
        dotpq_3 = dotpq_3 + _shfl_xor_sync_170;
#else
        float2 accpq_3_0 = make_float2(0.0f, 0.0f);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[48], k2[0], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[49], k2[1], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[50], k2[2], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[51], k2[3], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[52], k2[4], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[53], k2[5], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[54], k2[6], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[55], k2[7], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[56], k2[8], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[57], k2[9], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[58], k2[10], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[59], k2[11], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[60], k2[12], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[61], k2[13], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[62], k2[14], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[63], k2[15], accpq_3_0);
        float dotpq_3 = accpq_3_0.x + accpq_3_0.y;
        float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, dotpq_3, 1);
        dotpq_3 = dotpq_3 + _shfl_xor_sync_105;
        float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, dotpq_3, 2);
        dotpq_3 = dotpq_3 + _shfl_xor_sync_106;
#endif
        o4[3] = dotpq_3;
        if (u == 0) {
          long long out_index_1 = (long long)n * out_stride_p0 + (long long)t_2 * out_stride_p1 +
                                  (long long)h * out_stride_p2 + (long long)row0;
          {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(o4[0 + 0], o4[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(o4[0 + 2], o4[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[out_index_1]) = _pk2;
          }
        }
      }
    }
    if (write_state_slot_valid != 0) {
      if (disable_state_update == 0) {
        if (use_ssm_state_indices == 0) {
#if __CUDA_ARCH__ == 900
          r_h[0] = S[0];
          r_h[1] = S[1];
          r_h[2] = S[2];
          r_h[3] = S[3];
#else
          r_h[0] = S2[0].x;
          r_h[1] = S2[0].y;
          r_h[2] = S2[1].x;
          r_h[3] = S2[1].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[4];
          r_h[1] = S[5];
          r_h[2] = S[6];
          r_h[3] = S[7];
#else
          r_h[0] = S2[2].x;
          r_h[1] = S2[2].y;
          r_h[2] = S2[3].x;
          r_h[3] = S2[3].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 16) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[8];
          r_h[1] = S[9];
          r_h[2] = S[10];
          r_h[3] = S[11];
#else
          r_h[0] = S2[4].x;
          r_h[1] = S2[4].y;
          r_h[2] = S2[5].x;
          r_h[3] = S2[5].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 32) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[12];
          r_h[1] = S[13];
          r_h[2] = S[14];
          r_h[3] = S[15];
#else
          r_h[0] = S2[6].x;
          r_h[1] = S2[6].y;
          r_h[2] = S2[7].x;
          r_h[3] = S2[7].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 48) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[16];
          r_h[1] = S[17];
          r_h[2] = S[18];
          r_h[3] = S[19];
#else
          r_h[0] = S2[8].x;
          r_h[1] = S2[8].y;
          r_h[2] = S2[9].x;
          r_h[3] = S2[9].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 64) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[20];
          r_h[1] = S[21];
          r_h[2] = S[22];
          r_h[3] = S[23];
#else
          r_h[0] = S2[10].x;
          r_h[1] = S2[10].y;
          r_h[2] = S2[11].x;
          r_h[3] = S2[11].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 80) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[24];
          r_h[1] = S[25];
          r_h[2] = S[26];
          r_h[3] = S[27];
#else
          r_h[0] = S2[12].x;
          r_h[1] = S2[12].y;
          r_h[2] = S2[13].x;
          r_h[3] = S2[13].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 96) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[28];
          r_h[1] = S[29];
          r_h[2] = S[30];
          r_h[3] = S[31];
#else
          r_h[0] = S2[14].x;
          r_h[1] = S2[14].y;
          r_h[2] = S2[15].x;
          r_h[3] = S2[15].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 112) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[32];
          r_h[1] = S[33];
          r_h[2] = S[34];
          r_h[3] = S[35];
#else
          r_h[0] = S2[16].x;
          r_h[1] = S2[16].y;
          r_h[2] = S2[17].x;
          r_h[3] = S2[17].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[36];
          r_h[1] = S[37];
          r_h[2] = S[38];
          r_h[3] = S[39];
#else
          r_h[0] = S2[18].x;
          r_h[1] = S2[18].y;
          r_h[2] = S2[19].x;
          r_h[3] = S2[19].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 16) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[40];
          r_h[1] = S[41];
          r_h[2] = S[42];
          r_h[3] = S[43];
#else
          r_h[0] = S2[20].x;
          r_h[1] = S2[20].y;
          r_h[2] = S2[21].x;
          r_h[3] = S2[21].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 32) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[44];
          r_h[1] = S[45];
          r_h[2] = S[46];
          r_h[3] = S[47];
#else
          r_h[0] = S2[22].x;
          r_h[1] = S2[22].y;
          r_h[2] = S2[23].x;
          r_h[3] = S2[23].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 48) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[48];
          r_h[1] = S[49];
          r_h[2] = S[50];
          r_h[3] = S[51];
#else
          r_h[0] = S2[24].x;
          r_h[1] = S2[24].y;
          r_h[2] = S2[25].x;
          r_h[3] = S2[25].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 64) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[52];
          r_h[1] = S[53];
          r_h[2] = S[54];
          r_h[3] = S[55];
#else
          r_h[0] = S2[26].x;
          r_h[1] = S2[26].y;
          r_h[2] = S2[27].x;
          r_h[3] = S2[27].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 80) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[56];
          r_h[1] = S[57];
          r_h[2] = S[58];
          r_h[3] = S[59];
#else
          r_h[0] = S2[28].x;
          r_h[1] = S2[28].y;
          r_h[2] = S2[29].x;
          r_h[3] = S2[29].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 96) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[60];
          r_h[1] = S[61];
          r_h[2] = S[62];
          r_h[3] = S[63];
#else
          r_h[0] = S2[30].x;
          r_h[1] = S2[30].y;
          r_h[2] = S2[31].x;
          r_h[3] = S2[31].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 112) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[64];
          r_h[1] = S[65];
          r_h[2] = S[66];
          r_h[3] = S[67];
#else
          r_h[0] = S2[32].x;
          r_h[1] = S2[32].y;
          r_h[2] = S2[33].x;
          r_h[3] = S2[33].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[68];
          r_h[1] = S[69];
          r_h[2] = S[70];
          r_h[3] = S[71];
#else
          r_h[0] = S2[34].x;
          r_h[1] = S2[34].y;
          r_h[2] = S2[35].x;
          r_h[3] = S2[35].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2 + 16) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[72];
          r_h[1] = S[73];
          r_h[2] = S[74];
          r_h[3] = S[75];
#else
          r_h[0] = S2[36].x;
          r_h[1] = S2[36].y;
          r_h[2] = S2[37].x;
          r_h[3] = S2[37].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2 + 32) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[76];
          r_h[1] = S[77];
          r_h[2] = S[78];
          r_h[3] = S[79];
#else
          r_h[0] = S2[38].x;
          r_h[1] = S2[38].y;
          r_h[2] = S2[39].x;
          r_h[3] = S2[39].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2 + 48) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[80];
          r_h[1] = S[81];
          r_h[2] = S[82];
          r_h[3] = S[83];
#else
          r_h[0] = S2[40].x;
          r_h[1] = S2[40].y;
          r_h[2] = S2[41].x;
          r_h[3] = S2[41].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2 + 64) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[84];
          r_h[1] = S[85];
          r_h[2] = S[86];
          r_h[3] = S[87];
#else
          r_h[0] = S2[42].x;
          r_h[1] = S2[42].y;
          r_h[2] = S2[43].x;
          r_h[3] = S2[43].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2 + 80) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[88];
          r_h[1] = S[89];
          r_h[2] = S[90];
          r_h[3] = S[91];
#else
          r_h[0] = S2[44].x;
          r_h[1] = S2[44].y;
          r_h[2] = S2[45].x;
          r_h[3] = S2[45].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2 + 96) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[92];
          r_h[1] = S[93];
          r_h[2] = S[94];
          r_h[3] = S[95];
#else
          r_h[0] = S2[46].x;
          r_h[1] = S2[46].y;
          r_h[2] = S2[47].x;
          r_h[3] = S2[47].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2 + 112) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[96];
          r_h[1] = S[97];
          r_h[2] = S[98];
          r_h[3] = S[99];
#else
          r_h[0] = S2[48].x;
          r_h[1] = S2[48].y;
          r_h[2] = S2[49].x;
          r_h[3] = S2[49].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 3 * state_stride_p2) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[100];
          r_h[1] = S[101];
          r_h[2] = S[102];
          r_h[3] = S[103];
#else
          r_h[0] = S2[50].x;
          r_h[1] = S2[50].y;
          r_h[2] = S2[51].x;
          r_h[3] = S2[51].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 3 * state_stride_p2 + 16) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[104];
          r_h[1] = S[105];
          r_h[2] = S[106];
          r_h[3] = S[107];
#else
          r_h[0] = S2[52].x;
          r_h[1] = S2[52].y;
          r_h[2] = S2[53].x;
          r_h[3] = S2[53].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 3 * state_stride_p2 + 32) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[108];
          r_h[1] = S[109];
          r_h[2] = S[110];
          r_h[3] = S[111];
#else
          r_h[0] = S2[54].x;
          r_h[1] = S2[54].y;
          r_h[2] = S2[55].x;
          r_h[3] = S2[55].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 3 * state_stride_p2 + 48) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[112];
          r_h[1] = S[113];
          r_h[2] = S[114];
          r_h[3] = S[115];
#else
          r_h[0] = S2[56].x;
          r_h[1] = S2[56].y;
          r_h[2] = S2[57].x;
          r_h[3] = S2[57].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 3 * state_stride_p2 + 64) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[116];
          r_h[1] = S[117];
          r_h[2] = S[118];
          r_h[3] = S[119];
#else
          r_h[0] = S2[58].x;
          r_h[1] = S2[58].y;
          r_h[2] = S2[59].x;
          r_h[3] = S2[59].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 3 * state_stride_p2 + 80) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[120];
          r_h[1] = S[121];
          r_h[2] = S[122];
          r_h[3] = S[123];
#else
          r_h[0] = S2[60].x;
          r_h[1] = S2[60].y;
          r_h[2] = S2[61].x;
          r_h[3] = S2[61].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 3 * state_stride_p2 + 96) = _v4;
          }
#if __CUDA_ARCH__ == 900
          r_h[0] = S[124];
          r_h[1] = S[125];
          r_h[2] = S[126];
          r_h[3] = S[127];
#else
          r_h[0] = S2[62].x;
          r_h[1] = S2[62].y;
          r_h[2] = S2[63].x;
          r_h[3] = S2[63].y;
#endif
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 3 * state_stride_p2 + 112) = _v4;
          }
        }
      }
    }
  }
}

}  // extern "C"
