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

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
  float2 r;
  asm("add.rn.ftz.f32x2 %0, %1, %2;"
      : "=l"(*(unsigned long long*)&r)
      : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
  return r;
}

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

  // === Task calls (dependency order) ===
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
    float2 S2[64];
    float k_regs[32];
    float2 k2[16];
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
        float2 k_pair0 = make_float2(r_k[0], r_k[1]);
        float2 k_pair1 = make_float2(r_k[2], r_k[3]);
        float2 sum_k_pair = fma_f32x2_rn_ftz(k_pair0, k_pair0, make_float2(0.0f, 0.0f));
        sum_k_pair = fma_f32x2_rn_ftz(k_pair1, k_pair1, sum_k_pair);
        float sum_k = sum_k_pair.x + sum_k_pair.y;
        float _warp_reduce_33 = sum_k;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
          _warp_reduce_33 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_33, offset);
        sum_k = _warp_reduce_33;
        float k_norm = rsqrtf(sum_k + 1e-06f);
        float2 k_norm_pair = make_float2(k_norm, k_norm);
        k_pair0 = mul_f32x2(k_pair0, k_norm_pair);
        k_pair1 = mul_f32x2(k_pair1, k_norm_pair);
        sK[m * 128 + k_col0] = k_pair0.x;
        sK[m * 128 + k_col0 + 1] = k_pair0.y;
        sK[m * 128 + k_col0 + 2] = k_pair1.x;
        sK[m * 128 + k_col0 + 3] = k_pair1.y;
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
        float2 q_raw_pair0 = make_float2(r_q[0], r_q[1]);
        float2 q_raw_pair1 = make_float2(r_q[2], r_q[3]);
        float q_norm = scale;
        float2 sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair0, q_raw_pair0, make_float2(0.0f, 0.0f));
        sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair1, q_raw_pair1, sum_q_pair);
        float sum_q = sum_q_pair.x + sum_q_pair.y;
        float _warp_reduce_37 = sum_q;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
          _warp_reduce_37 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_37, offset);
        sum_q = _warp_reduce_37;
        float _rsqrt_38 = rsqrtf(sum_q + 1e-06f);
        q_norm = _rsqrt_38 * scale;
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
        if (lane_local == 0) {
          long long gate_index =
              (long long)n * a_stride_p0 + (long long)t * a_stride_p1 + (long long)h * a_stride_p2;
          float x = (float)a[gate_index] + dt_bias[h];
          float softplus_x = x;
          if (x <= 20.0f) {
            float _exp_40 = __expf(x);
            float _log2_39;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_39) : "f"(1.0f + _exp_40));
            softplus_x = _log2_39 * 0.6931471805599453f;
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
    if (use_ssm_state_indices != 0) {
#pragma unroll 1
      for (int t_1 = 0; t_1 < num_tokens; t_1++) {
        float alphas = sAlpha[t_1];
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
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                     : "r"(sV_addr + (unsigned int)(ms0 * 256) + group_byte));
        float betas0 = sBeta[ms0];
        float2 accs0_0_0 = make_float2(0.0f, 0.0f);
        float2 accs0_0_1 = make_float2(0.0f, 0.0f);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accs0_0_0);
        accs0_0_1 = fma_f32x2_rn_ftz(S2[1], k2[1], accs0_0_1);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accs0_0_0);
        accs0_0_1 = fma_f32x2_rn_ftz(S2[3], k2[3], accs0_0_1);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accs0_0_0);
        accs0_0_1 = fma_f32x2_rn_ftz(S2[5], k2[5], accs0_0_1);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accs0_0_0);
        accs0_0_1 = fma_f32x2_rn_ftz(S2[7], k2[7], accs0_0_1);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[8], k2[8], accs0_0_0);
        accs0_0_1 = fma_f32x2_rn_ftz(S2[9], k2[9], accs0_0_1);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[10], k2[10], accs0_0_0);
        accs0_0_1 = fma_f32x2_rn_ftz(S2[11], k2[11], accs0_0_1);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[12], k2[12], accs0_0_0);
        accs0_0_1 = fma_f32x2_rn_ftz(S2[13], k2[13], accs0_0_1);
        accs0_0_0 = fma_f32x2_rn_ftz(S2[14], k2[14], accs0_0_0);
        accs0_0_1 = fma_f32x2_rn_ftz(S2[15], k2[15], accs0_0_1);
        float2 _add_0 = add_f32x2(accs0_0_0, accs0_0_1);
        float dots0_0 = _add_0.x + _add_0.y;
        float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 1);
        dots0_0 = dots0_0 + _shfl_xor_sync_43;
        float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 2);
        dots0_0 = dots0_0 + _shfl_xor_sync_44;
        float vns0_0 = (v_regs[0] - dots0_0) * betas0;
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
        float2 accs0_1_1 = make_float2(0.0f, 0.0f);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accs0_1_0);
        accs0_1_1 = fma_f32x2_rn_ftz(S2[17], k2[1], accs0_1_1);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accs0_1_0);
        accs0_1_1 = fma_f32x2_rn_ftz(S2[19], k2[3], accs0_1_1);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accs0_1_0);
        accs0_1_1 = fma_f32x2_rn_ftz(S2[21], k2[5], accs0_1_1);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accs0_1_0);
        accs0_1_1 = fma_f32x2_rn_ftz(S2[23], k2[7], accs0_1_1);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[24], k2[8], accs0_1_0);
        accs0_1_1 = fma_f32x2_rn_ftz(S2[25], k2[9], accs0_1_1);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[26], k2[10], accs0_1_0);
        accs0_1_1 = fma_f32x2_rn_ftz(S2[27], k2[11], accs0_1_1);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[28], k2[12], accs0_1_0);
        accs0_1_1 = fma_f32x2_rn_ftz(S2[29], k2[13], accs0_1_1);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[30], k2[14], accs0_1_0);
        accs0_1_1 = fma_f32x2_rn_ftz(S2[31], k2[15], accs0_1_1);
        float2 _add_1 = add_f32x2(accs0_1_0, accs0_1_1);
        float dots0_1 = _add_1.x + _add_1.y;
        float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 1);
        dots0_1 = dots0_1 + _shfl_xor_sync_45;
        float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 2);
        dots0_1 = dots0_1 + _shfl_xor_sync_46;
        float vns0_1 = (v_regs[1] - dots0_1) * betas0;
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
        float2 accs0_2_1 = make_float2(0.0f, 0.0f);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[32], k2[0], accs0_2_0);
        accs0_2_1 = fma_f32x2_rn_ftz(S2[33], k2[1], accs0_2_1);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[34], k2[2], accs0_2_0);
        accs0_2_1 = fma_f32x2_rn_ftz(S2[35], k2[3], accs0_2_1);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[36], k2[4], accs0_2_0);
        accs0_2_1 = fma_f32x2_rn_ftz(S2[37], k2[5], accs0_2_1);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[38], k2[6], accs0_2_0);
        accs0_2_1 = fma_f32x2_rn_ftz(S2[39], k2[7], accs0_2_1);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[40], k2[8], accs0_2_0);
        accs0_2_1 = fma_f32x2_rn_ftz(S2[41], k2[9], accs0_2_1);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[42], k2[10], accs0_2_0);
        accs0_2_1 = fma_f32x2_rn_ftz(S2[43], k2[11], accs0_2_1);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[44], k2[12], accs0_2_0);
        accs0_2_1 = fma_f32x2_rn_ftz(S2[45], k2[13], accs0_2_1);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[46], k2[14], accs0_2_0);
        accs0_2_1 = fma_f32x2_rn_ftz(S2[47], k2[15], accs0_2_1);
        float2 _add_2 = add_f32x2(accs0_2_0, accs0_2_1);
        float dots0_2 = _add_2.x + _add_2.y;
        float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, dots0_2, 1);
        dots0_2 = dots0_2 + _shfl_xor_sync_47;
        float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, dots0_2, 2);
        dots0_2 = dots0_2 + _shfl_xor_sync_48;
        float vns0_2 = (v_regs[2] - dots0_2) * betas0;
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
        float2 accs0_3_1 = make_float2(0.0f, 0.0f);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[48], k2[0], accs0_3_0);
        accs0_3_1 = fma_f32x2_rn_ftz(S2[49], k2[1], accs0_3_1);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[50], k2[2], accs0_3_0);
        accs0_3_1 = fma_f32x2_rn_ftz(S2[51], k2[3], accs0_3_1);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[52], k2[4], accs0_3_0);
        accs0_3_1 = fma_f32x2_rn_ftz(S2[53], k2[5], accs0_3_1);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[54], k2[6], accs0_3_0);
        accs0_3_1 = fma_f32x2_rn_ftz(S2[55], k2[7], accs0_3_1);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[56], k2[8], accs0_3_0);
        accs0_3_1 = fma_f32x2_rn_ftz(S2[57], k2[9], accs0_3_1);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[58], k2[10], accs0_3_0);
        accs0_3_1 = fma_f32x2_rn_ftz(S2[59], k2[11], accs0_3_1);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[60], k2[12], accs0_3_0);
        accs0_3_1 = fma_f32x2_rn_ftz(S2[61], k2[13], accs0_3_1);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[62], k2[14], accs0_3_0);
        accs0_3_1 = fma_f32x2_rn_ftz(S2[63], k2[15], accs0_3_1);
        float2 _add_3 = add_f32x2(accs0_3_0, accs0_3_1);
        float dots0_3 = _add_3.x + _add_3.y;
        float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, dots0_3, 1);
        dots0_3 = dots0_3 + _shfl_xor_sync_49;
        float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, dots0_3, 2);
        dots0_3 = dots0_3 + _shfl_xor_sync_50;
        float vns0_3 = (v_regs[3] - dots0_3) * betas0;
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
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                     : "r"(sV_addr + (unsigned int)(ms1 * 256) + group_byte));
        float betas1 = sBeta[ms1];
        float2 accs1_0_0 = make_float2(0.0f, 0.0f);
        float2 accs1_0_1 = make_float2(0.0f, 0.0f);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accs1_0_0);
        accs1_0_1 = fma_f32x2_rn_ftz(S2[1], k2[1], accs1_0_1);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accs1_0_0);
        accs1_0_1 = fma_f32x2_rn_ftz(S2[3], k2[3], accs1_0_1);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accs1_0_0);
        accs1_0_1 = fma_f32x2_rn_ftz(S2[5], k2[5], accs1_0_1);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accs1_0_0);
        accs1_0_1 = fma_f32x2_rn_ftz(S2[7], k2[7], accs1_0_1);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[8], k2[8], accs1_0_0);
        accs1_0_1 = fma_f32x2_rn_ftz(S2[9], k2[9], accs1_0_1);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[10], k2[10], accs1_0_0);
        accs1_0_1 = fma_f32x2_rn_ftz(S2[11], k2[11], accs1_0_1);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[12], k2[12], accs1_0_0);
        accs1_0_1 = fma_f32x2_rn_ftz(S2[13], k2[13], accs1_0_1);
        accs1_0_0 = fma_f32x2_rn_ftz(S2[14], k2[14], accs1_0_0);
        accs1_0_1 = fma_f32x2_rn_ftz(S2[15], k2[15], accs1_0_1);
        float2 _add_4 = add_f32x2(accs1_0_0, accs1_0_1);
        float dots1_0 = _add_4.x + _add_4.y;
        float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 1);
        dots1_0 = dots1_0 + _shfl_xor_sync_51;
        float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 2);
        dots1_0 = dots1_0 + _shfl_xor_sync_52;
        float vns1_0 = (v_regs[0] - dots1_0) * betas1;
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
        float2 accs1_1_1 = make_float2(0.0f, 0.0f);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accs1_1_0);
        accs1_1_1 = fma_f32x2_rn_ftz(S2[17], k2[1], accs1_1_1);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accs1_1_0);
        accs1_1_1 = fma_f32x2_rn_ftz(S2[19], k2[3], accs1_1_1);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accs1_1_0);
        accs1_1_1 = fma_f32x2_rn_ftz(S2[21], k2[5], accs1_1_1);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accs1_1_0);
        accs1_1_1 = fma_f32x2_rn_ftz(S2[23], k2[7], accs1_1_1);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[24], k2[8], accs1_1_0);
        accs1_1_1 = fma_f32x2_rn_ftz(S2[25], k2[9], accs1_1_1);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[26], k2[10], accs1_1_0);
        accs1_1_1 = fma_f32x2_rn_ftz(S2[27], k2[11], accs1_1_1);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[28], k2[12], accs1_1_0);
        accs1_1_1 = fma_f32x2_rn_ftz(S2[29], k2[13], accs1_1_1);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[30], k2[14], accs1_1_0);
        accs1_1_1 = fma_f32x2_rn_ftz(S2[31], k2[15], accs1_1_1);
        float2 _add_5 = add_f32x2(accs1_1_0, accs1_1_1);
        float dots1_1 = _add_5.x + _add_5.y;
        float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 1);
        dots1_1 = dots1_1 + _shfl_xor_sync_53;
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 2);
        dots1_1 = dots1_1 + _shfl_xor_sync_54;
        float vns1_1 = (v_regs[1] - dots1_1) * betas1;
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
        float2 accs1_2_1 = make_float2(0.0f, 0.0f);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[32], k2[0], accs1_2_0);
        accs1_2_1 = fma_f32x2_rn_ftz(S2[33], k2[1], accs1_2_1);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[34], k2[2], accs1_2_0);
        accs1_2_1 = fma_f32x2_rn_ftz(S2[35], k2[3], accs1_2_1);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[36], k2[4], accs1_2_0);
        accs1_2_1 = fma_f32x2_rn_ftz(S2[37], k2[5], accs1_2_1);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[38], k2[6], accs1_2_0);
        accs1_2_1 = fma_f32x2_rn_ftz(S2[39], k2[7], accs1_2_1);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[40], k2[8], accs1_2_0);
        accs1_2_1 = fma_f32x2_rn_ftz(S2[41], k2[9], accs1_2_1);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[42], k2[10], accs1_2_0);
        accs1_2_1 = fma_f32x2_rn_ftz(S2[43], k2[11], accs1_2_1);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[44], k2[12], accs1_2_0);
        accs1_2_1 = fma_f32x2_rn_ftz(S2[45], k2[13], accs1_2_1);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[46], k2[14], accs1_2_0);
        accs1_2_1 = fma_f32x2_rn_ftz(S2[47], k2[15], accs1_2_1);
        float2 _add_6 = add_f32x2(accs1_2_0, accs1_2_1);
        float dots1_2 = _add_6.x + _add_6.y;
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, dots1_2, 1);
        dots1_2 = dots1_2 + _shfl_xor_sync_55;
        float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, dots1_2, 2);
        dots1_2 = dots1_2 + _shfl_xor_sync_56;
        float vns1_2 = (v_regs[2] - dots1_2) * betas1;
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
        float2 accs1_3_1 = make_float2(0.0f, 0.0f);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[48], k2[0], accs1_3_0);
        accs1_3_1 = fma_f32x2_rn_ftz(S2[49], k2[1], accs1_3_1);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[50], k2[2], accs1_3_0);
        accs1_3_1 = fma_f32x2_rn_ftz(S2[51], k2[3], accs1_3_1);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[52], k2[4], accs1_3_0);
        accs1_3_1 = fma_f32x2_rn_ftz(S2[53], k2[5], accs1_3_1);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[54], k2[6], accs1_3_0);
        accs1_3_1 = fma_f32x2_rn_ftz(S2[55], k2[7], accs1_3_1);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[56], k2[8], accs1_3_0);
        accs1_3_1 = fma_f32x2_rn_ftz(S2[57], k2[9], accs1_3_1);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[58], k2[10], accs1_3_0);
        accs1_3_1 = fma_f32x2_rn_ftz(S2[59], k2[11], accs1_3_1);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[60], k2[12], accs1_3_0);
        accs1_3_1 = fma_f32x2_rn_ftz(S2[61], k2[13], accs1_3_1);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[62], k2[14], accs1_3_0);
        accs1_3_1 = fma_f32x2_rn_ftz(S2[63], k2[15], accs1_3_1);
        float2 _add_7 = add_f32x2(accs1_3_0, accs1_3_1);
        float dots1_3 = _add_7.x + _add_7.y;
        float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, dots1_3, 1);
        dots1_3 = dots1_3 + _shfl_xor_sync_57;
        float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, dots1_3, 2);
        dots1_3 = dots1_3 + _shfl_xor_sync_58;
        float vns1_3 = (v_regs[3] - dots1_3) * betas1;
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
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                     : "r"(sV_addr + (unsigned int)(ms2 * 256) + group_byte));
        float betas2 = sBeta[ms2];
        float2 accs2_0_0 = make_float2(0.0f, 0.0f);
        float2 accs2_0_1 = make_float2(0.0f, 0.0f);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accs2_0_0);
        accs2_0_1 = fma_f32x2_rn_ftz(S2[1], k2[1], accs2_0_1);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accs2_0_0);
        accs2_0_1 = fma_f32x2_rn_ftz(S2[3], k2[3], accs2_0_1);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accs2_0_0);
        accs2_0_1 = fma_f32x2_rn_ftz(S2[5], k2[5], accs2_0_1);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accs2_0_0);
        accs2_0_1 = fma_f32x2_rn_ftz(S2[7], k2[7], accs2_0_1);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[8], k2[8], accs2_0_0);
        accs2_0_1 = fma_f32x2_rn_ftz(S2[9], k2[9], accs2_0_1);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[10], k2[10], accs2_0_0);
        accs2_0_1 = fma_f32x2_rn_ftz(S2[11], k2[11], accs2_0_1);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[12], k2[12], accs2_0_0);
        accs2_0_1 = fma_f32x2_rn_ftz(S2[13], k2[13], accs2_0_1);
        accs2_0_0 = fma_f32x2_rn_ftz(S2[14], k2[14], accs2_0_0);
        accs2_0_1 = fma_f32x2_rn_ftz(S2[15], k2[15], accs2_0_1);
        float2 _add_8 = add_f32x2(accs2_0_0, accs2_0_1);
        float dots2_0 = _add_8.x + _add_8.y;
        float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 1);
        dots2_0 = dots2_0 + _shfl_xor_sync_59;
        float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 2);
        dots2_0 = dots2_0 + _shfl_xor_sync_60;
        float vns2_0 = (v_regs[0] - dots2_0) * betas2;
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
        float2 accs2_1_1 = make_float2(0.0f, 0.0f);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accs2_1_0);
        accs2_1_1 = fma_f32x2_rn_ftz(S2[17], k2[1], accs2_1_1);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accs2_1_0);
        accs2_1_1 = fma_f32x2_rn_ftz(S2[19], k2[3], accs2_1_1);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accs2_1_0);
        accs2_1_1 = fma_f32x2_rn_ftz(S2[21], k2[5], accs2_1_1);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accs2_1_0);
        accs2_1_1 = fma_f32x2_rn_ftz(S2[23], k2[7], accs2_1_1);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[24], k2[8], accs2_1_0);
        accs2_1_1 = fma_f32x2_rn_ftz(S2[25], k2[9], accs2_1_1);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[26], k2[10], accs2_1_0);
        accs2_1_1 = fma_f32x2_rn_ftz(S2[27], k2[11], accs2_1_1);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[28], k2[12], accs2_1_0);
        accs2_1_1 = fma_f32x2_rn_ftz(S2[29], k2[13], accs2_1_1);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[30], k2[14], accs2_1_0);
        accs2_1_1 = fma_f32x2_rn_ftz(S2[31], k2[15], accs2_1_1);
        float2 _add_9 = add_f32x2(accs2_1_0, accs2_1_1);
        float dots2_1 = _add_9.x + _add_9.y;
        float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 1);
        dots2_1 = dots2_1 + _shfl_xor_sync_61;
        float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 2);
        dots2_1 = dots2_1 + _shfl_xor_sync_62;
        float vns2_1 = (v_regs[1] - dots2_1) * betas2;
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
        float2 accs2_2_1 = make_float2(0.0f, 0.0f);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[32], k2[0], accs2_2_0);
        accs2_2_1 = fma_f32x2_rn_ftz(S2[33], k2[1], accs2_2_1);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[34], k2[2], accs2_2_0);
        accs2_2_1 = fma_f32x2_rn_ftz(S2[35], k2[3], accs2_2_1);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[36], k2[4], accs2_2_0);
        accs2_2_1 = fma_f32x2_rn_ftz(S2[37], k2[5], accs2_2_1);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[38], k2[6], accs2_2_0);
        accs2_2_1 = fma_f32x2_rn_ftz(S2[39], k2[7], accs2_2_1);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[40], k2[8], accs2_2_0);
        accs2_2_1 = fma_f32x2_rn_ftz(S2[41], k2[9], accs2_2_1);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[42], k2[10], accs2_2_0);
        accs2_2_1 = fma_f32x2_rn_ftz(S2[43], k2[11], accs2_2_1);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[44], k2[12], accs2_2_0);
        accs2_2_1 = fma_f32x2_rn_ftz(S2[45], k2[13], accs2_2_1);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[46], k2[14], accs2_2_0);
        accs2_2_1 = fma_f32x2_rn_ftz(S2[47], k2[15], accs2_2_1);
        float2 _add_10 = add_f32x2(accs2_2_0, accs2_2_1);
        float dots2_2 = _add_10.x + _add_10.y;
        float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, dots2_2, 1);
        dots2_2 = dots2_2 + _shfl_xor_sync_63;
        float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, dots2_2, 2);
        dots2_2 = dots2_2 + _shfl_xor_sync_64;
        float vns2_2 = (v_regs[2] - dots2_2) * betas2;
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
        float2 accs2_3_1 = make_float2(0.0f, 0.0f);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[48], k2[0], accs2_3_0);
        accs2_3_1 = fma_f32x2_rn_ftz(S2[49], k2[1], accs2_3_1);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[50], k2[2], accs2_3_0);
        accs2_3_1 = fma_f32x2_rn_ftz(S2[51], k2[3], accs2_3_1);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[52], k2[4], accs2_3_0);
        accs2_3_1 = fma_f32x2_rn_ftz(S2[53], k2[5], accs2_3_1);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[54], k2[6], accs2_3_0);
        accs2_3_1 = fma_f32x2_rn_ftz(S2[55], k2[7], accs2_3_1);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[56], k2[8], accs2_3_0);
        accs2_3_1 = fma_f32x2_rn_ftz(S2[57], k2[9], accs2_3_1);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[58], k2[10], accs2_3_0);
        accs2_3_1 = fma_f32x2_rn_ftz(S2[59], k2[11], accs2_3_1);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[60], k2[12], accs2_3_0);
        accs2_3_1 = fma_f32x2_rn_ftz(S2[61], k2[13], accs2_3_1);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[62], k2[14], accs2_3_0);
        accs2_3_1 = fma_f32x2_rn_ftz(S2[63], k2[15], accs2_3_1);
        float2 _add_11 = add_f32x2(accs2_3_0, accs2_3_1);
        float dots2_3 = _add_11.x + _add_11.y;
        float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, dots2_3, 1);
        dots2_3 = dots2_3 + _shfl_xor_sync_65;
        float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, dots2_3, 2);
        dots2_3 = dots2_3 + _shfl_xor_sync_66;
        float vns2_3 = (v_regs[3] - dots2_3) * betas2;
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
        float2 accsq_0_1 = make_float2(0.0f, 0.0f);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accsq_0_0);
        accsq_0_1 = fma_f32x2_rn_ftz(S2[1], k2[1], accsq_0_1);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accsq_0_0);
        accsq_0_1 = fma_f32x2_rn_ftz(S2[3], k2[3], accsq_0_1);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accsq_0_0);
        accsq_0_1 = fma_f32x2_rn_ftz(S2[5], k2[5], accsq_0_1);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accsq_0_0);
        accsq_0_1 = fma_f32x2_rn_ftz(S2[7], k2[7], accsq_0_1);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[8], k2[8], accsq_0_0);
        accsq_0_1 = fma_f32x2_rn_ftz(S2[9], k2[9], accsq_0_1);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[10], k2[10], accsq_0_0);
        accsq_0_1 = fma_f32x2_rn_ftz(S2[11], k2[11], accsq_0_1);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[12], k2[12], accsq_0_0);
        accsq_0_1 = fma_f32x2_rn_ftz(S2[13], k2[13], accsq_0_1);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[14], k2[14], accsq_0_0);
        accsq_0_1 = fma_f32x2_rn_ftz(S2[15], k2[15], accsq_0_1);
        float2 _add_12 = add_f32x2(accsq_0_0, accsq_0_1);
        float dotsq_0 = _add_12.x + _add_12.y;
        float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 1);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_67;
        float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 2);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_68;
        o4[0] = dotsq_0;
        float2 accsq_1_0 = make_float2(0.0f, 0.0f);
        float2 accsq_1_1 = make_float2(0.0f, 0.0f);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accsq_1_0);
        accsq_1_1 = fma_f32x2_rn_ftz(S2[17], k2[1], accsq_1_1);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accsq_1_0);
        accsq_1_1 = fma_f32x2_rn_ftz(S2[19], k2[3], accsq_1_1);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accsq_1_0);
        accsq_1_1 = fma_f32x2_rn_ftz(S2[21], k2[5], accsq_1_1);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accsq_1_0);
        accsq_1_1 = fma_f32x2_rn_ftz(S2[23], k2[7], accsq_1_1);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[24], k2[8], accsq_1_0);
        accsq_1_1 = fma_f32x2_rn_ftz(S2[25], k2[9], accsq_1_1);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[26], k2[10], accsq_1_0);
        accsq_1_1 = fma_f32x2_rn_ftz(S2[27], k2[11], accsq_1_1);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[28], k2[12], accsq_1_0);
        accsq_1_1 = fma_f32x2_rn_ftz(S2[29], k2[13], accsq_1_1);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[30], k2[14], accsq_1_0);
        accsq_1_1 = fma_f32x2_rn_ftz(S2[31], k2[15], accsq_1_1);
        float2 _add_13 = add_f32x2(accsq_1_0, accsq_1_1);
        float dotsq_1 = _add_13.x + _add_13.y;
        float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 1);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_69;
        float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 2);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_70;
        o4[1] = dotsq_1;
        float2 accsq_2_0 = make_float2(0.0f, 0.0f);
        float2 accsq_2_1 = make_float2(0.0f, 0.0f);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[32], k2[0], accsq_2_0);
        accsq_2_1 = fma_f32x2_rn_ftz(S2[33], k2[1], accsq_2_1);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[34], k2[2], accsq_2_0);
        accsq_2_1 = fma_f32x2_rn_ftz(S2[35], k2[3], accsq_2_1);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[36], k2[4], accsq_2_0);
        accsq_2_1 = fma_f32x2_rn_ftz(S2[37], k2[5], accsq_2_1);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[38], k2[6], accsq_2_0);
        accsq_2_1 = fma_f32x2_rn_ftz(S2[39], k2[7], accsq_2_1);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[40], k2[8], accsq_2_0);
        accsq_2_1 = fma_f32x2_rn_ftz(S2[41], k2[9], accsq_2_1);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[42], k2[10], accsq_2_0);
        accsq_2_1 = fma_f32x2_rn_ftz(S2[43], k2[11], accsq_2_1);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[44], k2[12], accsq_2_0);
        accsq_2_1 = fma_f32x2_rn_ftz(S2[45], k2[13], accsq_2_1);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[46], k2[14], accsq_2_0);
        accsq_2_1 = fma_f32x2_rn_ftz(S2[47], k2[15], accsq_2_1);
        float2 _add_14 = add_f32x2(accsq_2_0, accsq_2_1);
        float dotsq_2 = _add_14.x + _add_14.y;
        float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, dotsq_2, 1);
        dotsq_2 = dotsq_2 + _shfl_xor_sync_71;
        float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, dotsq_2, 2);
        dotsq_2 = dotsq_2 + _shfl_xor_sync_72;
        o4[2] = dotsq_2;
        float2 accsq_3_0 = make_float2(0.0f, 0.0f);
        float2 accsq_3_1 = make_float2(0.0f, 0.0f);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[48], k2[0], accsq_3_0);
        accsq_3_1 = fma_f32x2_rn_ftz(S2[49], k2[1], accsq_3_1);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[50], k2[2], accsq_3_0);
        accsq_3_1 = fma_f32x2_rn_ftz(S2[51], k2[3], accsq_3_1);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[52], k2[4], accsq_3_0);
        accsq_3_1 = fma_f32x2_rn_ftz(S2[53], k2[5], accsq_3_1);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[54], k2[6], accsq_3_0);
        accsq_3_1 = fma_f32x2_rn_ftz(S2[55], k2[7], accsq_3_1);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[56], k2[8], accsq_3_0);
        accsq_3_1 = fma_f32x2_rn_ftz(S2[57], k2[9], accsq_3_1);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[58], k2[10], accsq_3_0);
        accsq_3_1 = fma_f32x2_rn_ftz(S2[59], k2[11], accsq_3_1);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[60], k2[12], accsq_3_0);
        accsq_3_1 = fma_f32x2_rn_ftz(S2[61], k2[13], accsq_3_1);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[62], k2[14], accsq_3_0);
        accsq_3_1 = fma_f32x2_rn_ftz(S2[63], k2[15], accsq_3_1);
        float2 _add_15 = add_f32x2(accsq_3_0, accsq_3_1);
        float dotsq_3 = _add_15.x + _add_15.y;
        float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, dotsq_3, 1);
        dotsq_3 = dotsq_3 + _shfl_xor_sync_73;
        float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, dotsq_3, 2);
        dotsq_3 = dotsq_3 + _shfl_xor_sync_74;
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
      }
    }
    if (use_ssm_state_indices == 0) {
#pragma unroll 1
      for (int t_2 = 0; t_2 < num_tokens; t_2++) {
        float alphap = sAlpha[t_2];
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
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                     : "r"(sV_addr + (unsigned int)(mp0 * 256) + group_byte));
        float betap0 = sBeta[mp0];
        float2 accp0_0_0 = make_float2(0.0f, 0.0f);
        float2 accp0_0_1 = make_float2(0.0f, 0.0f);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accp0_0_0);
        accp0_0_1 = fma_f32x2_rn_ftz(S2[1], k2[1], accp0_0_1);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accp0_0_0);
        accp0_0_1 = fma_f32x2_rn_ftz(S2[3], k2[3], accp0_0_1);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accp0_0_0);
        accp0_0_1 = fma_f32x2_rn_ftz(S2[5], k2[5], accp0_0_1);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accp0_0_0);
        accp0_0_1 = fma_f32x2_rn_ftz(S2[7], k2[7], accp0_0_1);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[8], k2[8], accp0_0_0);
        accp0_0_1 = fma_f32x2_rn_ftz(S2[9], k2[9], accp0_0_1);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[10], k2[10], accp0_0_0);
        accp0_0_1 = fma_f32x2_rn_ftz(S2[11], k2[11], accp0_0_1);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[12], k2[12], accp0_0_0);
        accp0_0_1 = fma_f32x2_rn_ftz(S2[13], k2[13], accp0_0_1);
        accp0_0_0 = fma_f32x2_rn_ftz(S2[14], k2[14], accp0_0_0);
        accp0_0_1 = fma_f32x2_rn_ftz(S2[15], k2[15], accp0_0_1);
        float2 _add_16 = add_f32x2(accp0_0_0, accp0_0_1);
        float dotp0_0 = _add_16.x + _add_16.y;
        float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 1);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_75;
        float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 2);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_76;
        float vnp0_0 = (v_regs[0] - dotp0_0) * betap0;
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
        float2 accp0_1_1 = make_float2(0.0f, 0.0f);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accp0_1_0);
        accp0_1_1 = fma_f32x2_rn_ftz(S2[17], k2[1], accp0_1_1);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accp0_1_0);
        accp0_1_1 = fma_f32x2_rn_ftz(S2[19], k2[3], accp0_1_1);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accp0_1_0);
        accp0_1_1 = fma_f32x2_rn_ftz(S2[21], k2[5], accp0_1_1);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accp0_1_0);
        accp0_1_1 = fma_f32x2_rn_ftz(S2[23], k2[7], accp0_1_1);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[24], k2[8], accp0_1_0);
        accp0_1_1 = fma_f32x2_rn_ftz(S2[25], k2[9], accp0_1_1);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[26], k2[10], accp0_1_0);
        accp0_1_1 = fma_f32x2_rn_ftz(S2[27], k2[11], accp0_1_1);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[28], k2[12], accp0_1_0);
        accp0_1_1 = fma_f32x2_rn_ftz(S2[29], k2[13], accp0_1_1);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[30], k2[14], accp0_1_0);
        accp0_1_1 = fma_f32x2_rn_ftz(S2[31], k2[15], accp0_1_1);
        float2 _add_17 = add_f32x2(accp0_1_0, accp0_1_1);
        float dotp0_1 = _add_17.x + _add_17.y;
        float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 1);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_77;
        float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 2);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_78;
        float vnp0_1 = (v_regs[1] - dotp0_1) * betap0;
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
        float2 accp0_2_1 = make_float2(0.0f, 0.0f);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[32], k2[0], accp0_2_0);
        accp0_2_1 = fma_f32x2_rn_ftz(S2[33], k2[1], accp0_2_1);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[34], k2[2], accp0_2_0);
        accp0_2_1 = fma_f32x2_rn_ftz(S2[35], k2[3], accp0_2_1);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[36], k2[4], accp0_2_0);
        accp0_2_1 = fma_f32x2_rn_ftz(S2[37], k2[5], accp0_2_1);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[38], k2[6], accp0_2_0);
        accp0_2_1 = fma_f32x2_rn_ftz(S2[39], k2[7], accp0_2_1);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[40], k2[8], accp0_2_0);
        accp0_2_1 = fma_f32x2_rn_ftz(S2[41], k2[9], accp0_2_1);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[42], k2[10], accp0_2_0);
        accp0_2_1 = fma_f32x2_rn_ftz(S2[43], k2[11], accp0_2_1);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[44], k2[12], accp0_2_0);
        accp0_2_1 = fma_f32x2_rn_ftz(S2[45], k2[13], accp0_2_1);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[46], k2[14], accp0_2_0);
        accp0_2_1 = fma_f32x2_rn_ftz(S2[47], k2[15], accp0_2_1);
        float2 _add_18 = add_f32x2(accp0_2_0, accp0_2_1);
        float dotp0_2 = _add_18.x + _add_18.y;
        float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, dotp0_2, 1);
        dotp0_2 = dotp0_2 + _shfl_xor_sync_79;
        float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, dotp0_2, 2);
        dotp0_2 = dotp0_2 + _shfl_xor_sync_80;
        float vnp0_2 = (v_regs[2] - dotp0_2) * betap0;
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
        float2 accp0_3_1 = make_float2(0.0f, 0.0f);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[48], k2[0], accp0_3_0);
        accp0_3_1 = fma_f32x2_rn_ftz(S2[49], k2[1], accp0_3_1);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[50], k2[2], accp0_3_0);
        accp0_3_1 = fma_f32x2_rn_ftz(S2[51], k2[3], accp0_3_1);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[52], k2[4], accp0_3_0);
        accp0_3_1 = fma_f32x2_rn_ftz(S2[53], k2[5], accp0_3_1);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[54], k2[6], accp0_3_0);
        accp0_3_1 = fma_f32x2_rn_ftz(S2[55], k2[7], accp0_3_1);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[56], k2[8], accp0_3_0);
        accp0_3_1 = fma_f32x2_rn_ftz(S2[57], k2[9], accp0_3_1);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[58], k2[10], accp0_3_0);
        accp0_3_1 = fma_f32x2_rn_ftz(S2[59], k2[11], accp0_3_1);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[60], k2[12], accp0_3_0);
        accp0_3_1 = fma_f32x2_rn_ftz(S2[61], k2[13], accp0_3_1);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[62], k2[14], accp0_3_0);
        accp0_3_1 = fma_f32x2_rn_ftz(S2[63], k2[15], accp0_3_1);
        float2 _add_19 = add_f32x2(accp0_3_0, accp0_3_1);
        float dotp0_3 = _add_19.x + _add_19.y;
        float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, dotp0_3, 1);
        dotp0_3 = dotp0_3 + _shfl_xor_sync_81;
        float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, dotp0_3, 2);
        dotp0_3 = dotp0_3 + _shfl_xor_sync_82;
        float vnp0_3 = (v_regs[3] - dotp0_3) * betap0;
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
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                     : "r"(sV_addr + (unsigned int)(mp1 * 256) + group_byte));
        float betap1 = sBeta[mp1];
        float2 accp1_0_0 = make_float2(0.0f, 0.0f);
        float2 accp1_0_1 = make_float2(0.0f, 0.0f);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accp1_0_0);
        accp1_0_1 = fma_f32x2_rn_ftz(S2[1], k2[1], accp1_0_1);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accp1_0_0);
        accp1_0_1 = fma_f32x2_rn_ftz(S2[3], k2[3], accp1_0_1);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accp1_0_0);
        accp1_0_1 = fma_f32x2_rn_ftz(S2[5], k2[5], accp1_0_1);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accp1_0_0);
        accp1_0_1 = fma_f32x2_rn_ftz(S2[7], k2[7], accp1_0_1);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[8], k2[8], accp1_0_0);
        accp1_0_1 = fma_f32x2_rn_ftz(S2[9], k2[9], accp1_0_1);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[10], k2[10], accp1_0_0);
        accp1_0_1 = fma_f32x2_rn_ftz(S2[11], k2[11], accp1_0_1);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[12], k2[12], accp1_0_0);
        accp1_0_1 = fma_f32x2_rn_ftz(S2[13], k2[13], accp1_0_1);
        accp1_0_0 = fma_f32x2_rn_ftz(S2[14], k2[14], accp1_0_0);
        accp1_0_1 = fma_f32x2_rn_ftz(S2[15], k2[15], accp1_0_1);
        float2 _add_20 = add_f32x2(accp1_0_0, accp1_0_1);
        float dotp1_0 = _add_20.x + _add_20.y;
        float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 1);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_83;
        float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 2);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_84;
        float vnp1_0 = (v_regs[0] - dotp1_0) * betap1;
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
        float2 accp1_1_1 = make_float2(0.0f, 0.0f);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accp1_1_0);
        accp1_1_1 = fma_f32x2_rn_ftz(S2[17], k2[1], accp1_1_1);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accp1_1_0);
        accp1_1_1 = fma_f32x2_rn_ftz(S2[19], k2[3], accp1_1_1);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accp1_1_0);
        accp1_1_1 = fma_f32x2_rn_ftz(S2[21], k2[5], accp1_1_1);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accp1_1_0);
        accp1_1_1 = fma_f32x2_rn_ftz(S2[23], k2[7], accp1_1_1);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[24], k2[8], accp1_1_0);
        accp1_1_1 = fma_f32x2_rn_ftz(S2[25], k2[9], accp1_1_1);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[26], k2[10], accp1_1_0);
        accp1_1_1 = fma_f32x2_rn_ftz(S2[27], k2[11], accp1_1_1);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[28], k2[12], accp1_1_0);
        accp1_1_1 = fma_f32x2_rn_ftz(S2[29], k2[13], accp1_1_1);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[30], k2[14], accp1_1_0);
        accp1_1_1 = fma_f32x2_rn_ftz(S2[31], k2[15], accp1_1_1);
        float2 _add_21 = add_f32x2(accp1_1_0, accp1_1_1);
        float dotp1_1 = _add_21.x + _add_21.y;
        float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 1);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_85;
        float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 2);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_86;
        float vnp1_1 = (v_regs[1] - dotp1_1) * betap1;
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
        float2 accp1_2_1 = make_float2(0.0f, 0.0f);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[32], k2[0], accp1_2_0);
        accp1_2_1 = fma_f32x2_rn_ftz(S2[33], k2[1], accp1_2_1);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[34], k2[2], accp1_2_0);
        accp1_2_1 = fma_f32x2_rn_ftz(S2[35], k2[3], accp1_2_1);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[36], k2[4], accp1_2_0);
        accp1_2_1 = fma_f32x2_rn_ftz(S2[37], k2[5], accp1_2_1);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[38], k2[6], accp1_2_0);
        accp1_2_1 = fma_f32x2_rn_ftz(S2[39], k2[7], accp1_2_1);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[40], k2[8], accp1_2_0);
        accp1_2_1 = fma_f32x2_rn_ftz(S2[41], k2[9], accp1_2_1);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[42], k2[10], accp1_2_0);
        accp1_2_1 = fma_f32x2_rn_ftz(S2[43], k2[11], accp1_2_1);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[44], k2[12], accp1_2_0);
        accp1_2_1 = fma_f32x2_rn_ftz(S2[45], k2[13], accp1_2_1);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[46], k2[14], accp1_2_0);
        accp1_2_1 = fma_f32x2_rn_ftz(S2[47], k2[15], accp1_2_1);
        float2 _add_22 = add_f32x2(accp1_2_0, accp1_2_1);
        float dotp1_2 = _add_22.x + _add_22.y;
        float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, dotp1_2, 1);
        dotp1_2 = dotp1_2 + _shfl_xor_sync_87;
        float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, dotp1_2, 2);
        dotp1_2 = dotp1_2 + _shfl_xor_sync_88;
        float vnp1_2 = (v_regs[2] - dotp1_2) * betap1;
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
        float2 accp1_3_1 = make_float2(0.0f, 0.0f);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[48], k2[0], accp1_3_0);
        accp1_3_1 = fma_f32x2_rn_ftz(S2[49], k2[1], accp1_3_1);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[50], k2[2], accp1_3_0);
        accp1_3_1 = fma_f32x2_rn_ftz(S2[51], k2[3], accp1_3_1);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[52], k2[4], accp1_3_0);
        accp1_3_1 = fma_f32x2_rn_ftz(S2[53], k2[5], accp1_3_1);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[54], k2[6], accp1_3_0);
        accp1_3_1 = fma_f32x2_rn_ftz(S2[55], k2[7], accp1_3_1);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[56], k2[8], accp1_3_0);
        accp1_3_1 = fma_f32x2_rn_ftz(S2[57], k2[9], accp1_3_1);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[58], k2[10], accp1_3_0);
        accp1_3_1 = fma_f32x2_rn_ftz(S2[59], k2[11], accp1_3_1);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[60], k2[12], accp1_3_0);
        accp1_3_1 = fma_f32x2_rn_ftz(S2[61], k2[13], accp1_3_1);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[62], k2[14], accp1_3_0);
        accp1_3_1 = fma_f32x2_rn_ftz(S2[63], k2[15], accp1_3_1);
        float2 _add_23 = add_f32x2(accp1_3_0, accp1_3_1);
        float dotp1_3 = _add_23.x + _add_23.y;
        float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, dotp1_3, 1);
        dotp1_3 = dotp1_3 + _shfl_xor_sync_89;
        float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, dotp1_3, 2);
        dotp1_3 = dotp1_3 + _shfl_xor_sync_90;
        float vnp1_3 = (v_regs[3] - dotp1_3) * betap1;
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
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                     : "r"(sV_addr + (unsigned int)(mp2 * 256) + group_byte));
        float betap2 = sBeta[mp2];
        float2 accp2_0_0 = make_float2(0.0f, 0.0f);
        float2 accp2_0_1 = make_float2(0.0f, 0.0f);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accp2_0_0);
        accp2_0_1 = fma_f32x2_rn_ftz(S2[1], k2[1], accp2_0_1);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accp2_0_0);
        accp2_0_1 = fma_f32x2_rn_ftz(S2[3], k2[3], accp2_0_1);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accp2_0_0);
        accp2_0_1 = fma_f32x2_rn_ftz(S2[5], k2[5], accp2_0_1);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accp2_0_0);
        accp2_0_1 = fma_f32x2_rn_ftz(S2[7], k2[7], accp2_0_1);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[8], k2[8], accp2_0_0);
        accp2_0_1 = fma_f32x2_rn_ftz(S2[9], k2[9], accp2_0_1);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[10], k2[10], accp2_0_0);
        accp2_0_1 = fma_f32x2_rn_ftz(S2[11], k2[11], accp2_0_1);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[12], k2[12], accp2_0_0);
        accp2_0_1 = fma_f32x2_rn_ftz(S2[13], k2[13], accp2_0_1);
        accp2_0_0 = fma_f32x2_rn_ftz(S2[14], k2[14], accp2_0_0);
        accp2_0_1 = fma_f32x2_rn_ftz(S2[15], k2[15], accp2_0_1);
        float2 _add_24 = add_f32x2(accp2_0_0, accp2_0_1);
        float dotp2_0 = _add_24.x + _add_24.y;
        float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 1);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_91;
        float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 2);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_92;
        float vnp2_0 = (v_regs[0] - dotp2_0) * betap2;
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
        float2 accp2_1_1 = make_float2(0.0f, 0.0f);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accp2_1_0);
        accp2_1_1 = fma_f32x2_rn_ftz(S2[17], k2[1], accp2_1_1);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accp2_1_0);
        accp2_1_1 = fma_f32x2_rn_ftz(S2[19], k2[3], accp2_1_1);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accp2_1_0);
        accp2_1_1 = fma_f32x2_rn_ftz(S2[21], k2[5], accp2_1_1);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accp2_1_0);
        accp2_1_1 = fma_f32x2_rn_ftz(S2[23], k2[7], accp2_1_1);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[24], k2[8], accp2_1_0);
        accp2_1_1 = fma_f32x2_rn_ftz(S2[25], k2[9], accp2_1_1);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[26], k2[10], accp2_1_0);
        accp2_1_1 = fma_f32x2_rn_ftz(S2[27], k2[11], accp2_1_1);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[28], k2[12], accp2_1_0);
        accp2_1_1 = fma_f32x2_rn_ftz(S2[29], k2[13], accp2_1_1);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[30], k2[14], accp2_1_0);
        accp2_1_1 = fma_f32x2_rn_ftz(S2[31], k2[15], accp2_1_1);
        float2 _add_25 = add_f32x2(accp2_1_0, accp2_1_1);
        float dotp2_1 = _add_25.x + _add_25.y;
        float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 1);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_93;
        float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 2);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_94;
        float vnp2_1 = (v_regs[1] - dotp2_1) * betap2;
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
        float2 accp2_2_1 = make_float2(0.0f, 0.0f);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[32], k2[0], accp2_2_0);
        accp2_2_1 = fma_f32x2_rn_ftz(S2[33], k2[1], accp2_2_1);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[34], k2[2], accp2_2_0);
        accp2_2_1 = fma_f32x2_rn_ftz(S2[35], k2[3], accp2_2_1);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[36], k2[4], accp2_2_0);
        accp2_2_1 = fma_f32x2_rn_ftz(S2[37], k2[5], accp2_2_1);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[38], k2[6], accp2_2_0);
        accp2_2_1 = fma_f32x2_rn_ftz(S2[39], k2[7], accp2_2_1);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[40], k2[8], accp2_2_0);
        accp2_2_1 = fma_f32x2_rn_ftz(S2[41], k2[9], accp2_2_1);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[42], k2[10], accp2_2_0);
        accp2_2_1 = fma_f32x2_rn_ftz(S2[43], k2[11], accp2_2_1);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[44], k2[12], accp2_2_0);
        accp2_2_1 = fma_f32x2_rn_ftz(S2[45], k2[13], accp2_2_1);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[46], k2[14], accp2_2_0);
        accp2_2_1 = fma_f32x2_rn_ftz(S2[47], k2[15], accp2_2_1);
        float2 _add_26 = add_f32x2(accp2_2_0, accp2_2_1);
        float dotp2_2 = _add_26.x + _add_26.y;
        float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, dotp2_2, 1);
        dotp2_2 = dotp2_2 + _shfl_xor_sync_95;
        float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, dotp2_2, 2);
        dotp2_2 = dotp2_2 + _shfl_xor_sync_96;
        float vnp2_2 = (v_regs[2] - dotp2_2) * betap2;
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
        float2 accp2_3_1 = make_float2(0.0f, 0.0f);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[48], k2[0], accp2_3_0);
        accp2_3_1 = fma_f32x2_rn_ftz(S2[49], k2[1], accp2_3_1);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[50], k2[2], accp2_3_0);
        accp2_3_1 = fma_f32x2_rn_ftz(S2[51], k2[3], accp2_3_1);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[52], k2[4], accp2_3_0);
        accp2_3_1 = fma_f32x2_rn_ftz(S2[53], k2[5], accp2_3_1);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[54], k2[6], accp2_3_0);
        accp2_3_1 = fma_f32x2_rn_ftz(S2[55], k2[7], accp2_3_1);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[56], k2[8], accp2_3_0);
        accp2_3_1 = fma_f32x2_rn_ftz(S2[57], k2[9], accp2_3_1);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[58], k2[10], accp2_3_0);
        accp2_3_1 = fma_f32x2_rn_ftz(S2[59], k2[11], accp2_3_1);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[60], k2[12], accp2_3_0);
        accp2_3_1 = fma_f32x2_rn_ftz(S2[61], k2[13], accp2_3_1);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[62], k2[14], accp2_3_0);
        accp2_3_1 = fma_f32x2_rn_ftz(S2[63], k2[15], accp2_3_1);
        float2 _add_27 = add_f32x2(accp2_3_0, accp2_3_1);
        float dotp2_3 = _add_27.x + _add_27.y;
        float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, dotp2_3, 1);
        dotp2_3 = dotp2_3 + _shfl_xor_sync_97;
        float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, dotp2_3, 2);
        dotp2_3 = dotp2_3 + _shfl_xor_sync_98;
        float vnp2_3 = (v_regs[3] - dotp2_3) * betap2;
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
        float2 accpq_0_1 = make_float2(0.0f, 0.0f);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accpq_0_0);
        accpq_0_1 = fma_f32x2_rn_ftz(S2[1], k2[1], accpq_0_1);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accpq_0_0);
        accpq_0_1 = fma_f32x2_rn_ftz(S2[3], k2[3], accpq_0_1);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accpq_0_0);
        accpq_0_1 = fma_f32x2_rn_ftz(S2[5], k2[5], accpq_0_1);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accpq_0_0);
        accpq_0_1 = fma_f32x2_rn_ftz(S2[7], k2[7], accpq_0_1);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[8], k2[8], accpq_0_0);
        accpq_0_1 = fma_f32x2_rn_ftz(S2[9], k2[9], accpq_0_1);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[10], k2[10], accpq_0_0);
        accpq_0_1 = fma_f32x2_rn_ftz(S2[11], k2[11], accpq_0_1);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[12], k2[12], accpq_0_0);
        accpq_0_1 = fma_f32x2_rn_ftz(S2[13], k2[13], accpq_0_1);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[14], k2[14], accpq_0_0);
        accpq_0_1 = fma_f32x2_rn_ftz(S2[15], k2[15], accpq_0_1);
        float2 _add_28 = add_f32x2(accpq_0_0, accpq_0_1);
        float dotpq_0 = _add_28.x + _add_28.y;
        float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 1);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_99;
        float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 2);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_100;
        o4[0] = dotpq_0;
        float2 accpq_1_0 = make_float2(0.0f, 0.0f);
        float2 accpq_1_1 = make_float2(0.0f, 0.0f);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accpq_1_0);
        accpq_1_1 = fma_f32x2_rn_ftz(S2[17], k2[1], accpq_1_1);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accpq_1_0);
        accpq_1_1 = fma_f32x2_rn_ftz(S2[19], k2[3], accpq_1_1);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accpq_1_0);
        accpq_1_1 = fma_f32x2_rn_ftz(S2[21], k2[5], accpq_1_1);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accpq_1_0);
        accpq_1_1 = fma_f32x2_rn_ftz(S2[23], k2[7], accpq_1_1);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[24], k2[8], accpq_1_0);
        accpq_1_1 = fma_f32x2_rn_ftz(S2[25], k2[9], accpq_1_1);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[26], k2[10], accpq_1_0);
        accpq_1_1 = fma_f32x2_rn_ftz(S2[27], k2[11], accpq_1_1);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[28], k2[12], accpq_1_0);
        accpq_1_1 = fma_f32x2_rn_ftz(S2[29], k2[13], accpq_1_1);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[30], k2[14], accpq_1_0);
        accpq_1_1 = fma_f32x2_rn_ftz(S2[31], k2[15], accpq_1_1);
        float2 _add_29 = add_f32x2(accpq_1_0, accpq_1_1);
        float dotpq_1 = _add_29.x + _add_29.y;
        float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 1);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_101;
        float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 2);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_102;
        o4[1] = dotpq_1;
        float2 accpq_2_0 = make_float2(0.0f, 0.0f);
        float2 accpq_2_1 = make_float2(0.0f, 0.0f);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[32], k2[0], accpq_2_0);
        accpq_2_1 = fma_f32x2_rn_ftz(S2[33], k2[1], accpq_2_1);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[34], k2[2], accpq_2_0);
        accpq_2_1 = fma_f32x2_rn_ftz(S2[35], k2[3], accpq_2_1);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[36], k2[4], accpq_2_0);
        accpq_2_1 = fma_f32x2_rn_ftz(S2[37], k2[5], accpq_2_1);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[38], k2[6], accpq_2_0);
        accpq_2_1 = fma_f32x2_rn_ftz(S2[39], k2[7], accpq_2_1);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[40], k2[8], accpq_2_0);
        accpq_2_1 = fma_f32x2_rn_ftz(S2[41], k2[9], accpq_2_1);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[42], k2[10], accpq_2_0);
        accpq_2_1 = fma_f32x2_rn_ftz(S2[43], k2[11], accpq_2_1);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[44], k2[12], accpq_2_0);
        accpq_2_1 = fma_f32x2_rn_ftz(S2[45], k2[13], accpq_2_1);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[46], k2[14], accpq_2_0);
        accpq_2_1 = fma_f32x2_rn_ftz(S2[47], k2[15], accpq_2_1);
        float2 _add_30 = add_f32x2(accpq_2_0, accpq_2_1);
        float dotpq_2 = _add_30.x + _add_30.y;
        float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, dotpq_2, 1);
        dotpq_2 = dotpq_2 + _shfl_xor_sync_103;
        float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, dotpq_2, 2);
        dotpq_2 = dotpq_2 + _shfl_xor_sync_104;
        o4[2] = dotpq_2;
        float2 accpq_3_0 = make_float2(0.0f, 0.0f);
        float2 accpq_3_1 = make_float2(0.0f, 0.0f);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[48], k2[0], accpq_3_0);
        accpq_3_1 = fma_f32x2_rn_ftz(S2[49], k2[1], accpq_3_1);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[50], k2[2], accpq_3_0);
        accpq_3_1 = fma_f32x2_rn_ftz(S2[51], k2[3], accpq_3_1);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[52], k2[4], accpq_3_0);
        accpq_3_1 = fma_f32x2_rn_ftz(S2[53], k2[5], accpq_3_1);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[54], k2[6], accpq_3_0);
        accpq_3_1 = fma_f32x2_rn_ftz(S2[55], k2[7], accpq_3_1);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[56], k2[8], accpq_3_0);
        accpq_3_1 = fma_f32x2_rn_ftz(S2[57], k2[9], accpq_3_1);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[58], k2[10], accpq_3_0);
        accpq_3_1 = fma_f32x2_rn_ftz(S2[59], k2[11], accpq_3_1);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[60], k2[12], accpq_3_0);
        accpq_3_1 = fma_f32x2_rn_ftz(S2[61], k2[13], accpq_3_1);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[62], k2[14], accpq_3_0);
        accpq_3_1 = fma_f32x2_rn_ftz(S2[63], k2[15], accpq_3_1);
        float2 _add_31 = add_f32x2(accpq_3_0, accpq_3_1);
        float dotpq_3 = _add_31.x + _add_31.y;
        float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, dotpq_3, 1);
        dotpq_3 = dotpq_3 + _shfl_xor_sync_105;
        float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, dotpq_3, 2);
        dotpq_3 = dotpq_3 + _shfl_xor_sync_106;
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
          r_h[0] = S2[0].x;
          r_h[1] = S2[0].y;
          r_h[2] = S2[1].x;
          r_h[3] = S2[1].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base) = _v4;
          }
          r_h[0] = S2[2].x;
          r_h[1] = S2[2].y;
          r_h[2] = S2[3].x;
          r_h[3] = S2[3].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 16) = _v4;
          }
          r_h[0] = S2[4].x;
          r_h[1] = S2[4].y;
          r_h[2] = S2[5].x;
          r_h[3] = S2[5].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 32) = _v4;
          }
          r_h[0] = S2[6].x;
          r_h[1] = S2[6].y;
          r_h[2] = S2[7].x;
          r_h[3] = S2[7].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 48) = _v4;
          }
          r_h[0] = S2[8].x;
          r_h[1] = S2[8].y;
          r_h[2] = S2[9].x;
          r_h[3] = S2[9].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 64) = _v4;
          }
          r_h[0] = S2[10].x;
          r_h[1] = S2[10].y;
          r_h[2] = S2[11].x;
          r_h[3] = S2[11].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 80) = _v4;
          }
          r_h[0] = S2[12].x;
          r_h[1] = S2[12].y;
          r_h[2] = S2[13].x;
          r_h[3] = S2[13].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 96) = _v4;
          }
          r_h[0] = S2[14].x;
          r_h[1] = S2[14].y;
          r_h[2] = S2[15].x;
          r_h[3] = S2[15].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 112) = _v4;
          }
          r_h[0] = S2[16].x;
          r_h[1] = S2[16].y;
          r_h[2] = S2[17].x;
          r_h[3] = S2[17].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2) = _v4;
          }
          r_h[0] = S2[18].x;
          r_h[1] = S2[18].y;
          r_h[2] = S2[19].x;
          r_h[3] = S2[19].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 16) = _v4;
          }
          r_h[0] = S2[20].x;
          r_h[1] = S2[20].y;
          r_h[2] = S2[21].x;
          r_h[3] = S2[21].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 32) = _v4;
          }
          r_h[0] = S2[22].x;
          r_h[1] = S2[22].y;
          r_h[2] = S2[23].x;
          r_h[3] = S2[23].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 48) = _v4;
          }
          r_h[0] = S2[24].x;
          r_h[1] = S2[24].y;
          r_h[2] = S2[25].x;
          r_h[3] = S2[25].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 64) = _v4;
          }
          r_h[0] = S2[26].x;
          r_h[1] = S2[26].y;
          r_h[2] = S2[27].x;
          r_h[3] = S2[27].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 80) = _v4;
          }
          r_h[0] = S2[28].x;
          r_h[1] = S2[28].y;
          r_h[2] = S2[29].x;
          r_h[3] = S2[29].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 96) = _v4;
          }
          r_h[0] = S2[30].x;
          r_h[1] = S2[30].y;
          r_h[2] = S2[31].x;
          r_h[3] = S2[31].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 112) = _v4;
          }
          r_h[0] = S2[32].x;
          r_h[1] = S2[32].y;
          r_h[2] = S2[33].x;
          r_h[3] = S2[33].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2) = _v4;
          }
          r_h[0] = S2[34].x;
          r_h[1] = S2[34].y;
          r_h[2] = S2[35].x;
          r_h[3] = S2[35].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2 + 16) = _v4;
          }
          r_h[0] = S2[36].x;
          r_h[1] = S2[36].y;
          r_h[2] = S2[37].x;
          r_h[3] = S2[37].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2 + 32) = _v4;
          }
          r_h[0] = S2[38].x;
          r_h[1] = S2[38].y;
          r_h[2] = S2[39].x;
          r_h[3] = S2[39].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2 + 48) = _v4;
          }
          r_h[0] = S2[40].x;
          r_h[1] = S2[40].y;
          r_h[2] = S2[41].x;
          r_h[3] = S2[41].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2 + 64) = _v4;
          }
          r_h[0] = S2[42].x;
          r_h[1] = S2[42].y;
          r_h[2] = S2[43].x;
          r_h[3] = S2[43].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2 + 80) = _v4;
          }
          r_h[0] = S2[44].x;
          r_h[1] = S2[44].y;
          r_h[2] = S2[45].x;
          r_h[3] = S2[45].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2 + 96) = _v4;
          }
          r_h[0] = S2[46].x;
          r_h[1] = S2[46].y;
          r_h[2] = S2[47].x;
          r_h[3] = S2[47].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2 + 112) = _v4;
          }
          r_h[0] = S2[48].x;
          r_h[1] = S2[48].y;
          r_h[2] = S2[49].x;
          r_h[3] = S2[49].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 3 * state_stride_p2) = _v4;
          }
          r_h[0] = S2[50].x;
          r_h[1] = S2[50].y;
          r_h[2] = S2[51].x;
          r_h[3] = S2[51].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 3 * state_stride_p2 + 16) = _v4;
          }
          r_h[0] = S2[52].x;
          r_h[1] = S2[52].y;
          r_h[2] = S2[53].x;
          r_h[3] = S2[53].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 3 * state_stride_p2 + 32) = _v4;
          }
          r_h[0] = S2[54].x;
          r_h[1] = S2[54].y;
          r_h[2] = S2[55].x;
          r_h[3] = S2[55].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 3 * state_stride_p2 + 48) = _v4;
          }
          r_h[0] = S2[56].x;
          r_h[1] = S2[56].y;
          r_h[2] = S2[57].x;
          r_h[3] = S2[57].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 3 * state_stride_p2 + 64) = _v4;
          }
          r_h[0] = S2[58].x;
          r_h[1] = S2[58].y;
          r_h[2] = S2[59].x;
          r_h[3] = S2[59].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 3 * state_stride_p2 + 80) = _v4;
          }
          r_h[0] = S2[60].x;
          r_h[1] = S2[60].y;
          r_h[2] = S2[61].x;
          r_h[3] = S2[61].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 3 * state_stride_p2 + 96) = _v4;
          }
          r_h[0] = S2[62].x;
          r_h[1] = S2[62].y;
          r_h[2] = S2[63].x;
          r_h[3] = S2[63].y;
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
