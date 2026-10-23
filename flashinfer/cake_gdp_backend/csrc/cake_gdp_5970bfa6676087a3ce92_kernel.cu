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
#define SMEM_SK_STAGE_BYTES 12288
#define SMEM_SK_STRIDE 12288
#define SMEM_SQ_OFF 12288
#define SMEM_SQ_STAGE_BYTES 4096
#define SMEM_SQ_STRIDE 4096
#define SMEM_SV_OFF 16384
#define SMEM_SV_STAGE_BYTES 6144
#define SMEM_SV_STRIDE 6144
#define SMEM_SALPHA_OFF 22528
#define SMEM_SALPHA_STAGE_BYTES 32
#define SMEM_SALPHA_STRIDE 32
#define SMEM_SBETA_OFF 22560
#define SMEM_SBETA_STAGE_BYTES 96
#define SMEM_SBETA_STRIDE 96
#define SMEM_SSSM_OFF 22656
#define SMEM_SSSM_STAGE_BYTES 32
#define SMEM_SSSM_STRIDE 32
#define SMEM_TOTAL 22784
#define THREADS 128
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

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void kernel_cake_gdp_5970bfa6676087a3ce92(
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
  float* sQ = reinterpret_cast<float*>(smem_raw + 12288);
  const int sQ_addr = smem + 12288;
  float* sV = reinterpret_cast<float*>(smem_raw + 16384);
  const int sV_addr = smem + 16384;
  float* sAlpha = reinterpret_cast<float*>(smem_raw + 22528);
  const int sAlpha_addr = smem + 22528;
  float* sBeta = reinterpret_cast<float*>(smem_raw + 22560);
  const int sBeta_addr = smem + 22560;
  int* sSsm = reinterpret_cast<int*>(smem_raw + 22656);
  const int sSsm_addr = smem + 22656;

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
  int u = tid_0 - tid_0 / 8 * 8;
  int g = tid_0 / 8;
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
  int steps = 24;
  int bm0 = warp_local;
  float bk0[4];
  float bv0[2];
  float bb0[1];
  long long bki0 =
      (long long)n * k_stride_p0 + (long long)bm0 * k_stride_p1 + (long long)h * k_stride_p2;
  {
    uint2 _vld_0;
    _vld_0 = *reinterpret_cast<const uint2*>(k + bki0 + (long long)k_col0);
    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
#pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
      (&bk0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
      (&bk0[0 + _pair * 2])[1] =
          __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
    }
  }
  long long bvi0 =
      (long long)n * v_stride_p0 + (long long)bm0 * v_stride_p1 + (long long)h * v_stride_p2;
  int bvc0 = lane_local * 2;
  bv0[0] = (float)v[bvi0 + (long long)bvc0];
  bv0[1] = (float)v[bvi0 + (long long)bvc0 + 1];
  long long bbi0 =
      (long long)n * b_stride_p0 + (long long)bm0 * b_stride_p1 + (long long)h * b_stride_p2;
  bb0[0] = (float)b[bbi0];
  int bm1 = 4 + warp_local;
  float bk1[4];
  float bv1[2];
  float bb1[1];
  long long bki1 =
      (long long)n * k_stride_p0 + (long long)bm1 * k_stride_p1 + (long long)h * k_stride_p2;
  {
    uint2 _vld_1;
    _vld_1 = *reinterpret_cast<const uint2*>(k + bki1 + (long long)k_col0);
    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
#pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
      (&bk1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
      (&bk1[0 + _pair * 2])[1] =
          __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
    }
  }
  long long bvi1 =
      (long long)n * v_stride_p0 + (long long)bm1 * v_stride_p1 + (long long)h * v_stride_p2;
  int bvc1 = lane_local * 2;
  bv1[0] = (float)v[bvi1 + (long long)bvc1];
  bv1[1] = (float)v[bvi1 + (long long)bvc1 + 1];
  long long bbi1 =
      (long long)n * b_stride_p0 + (long long)bm1 * b_stride_p1 + (long long)h * b_stride_p2;
  bb1[0] = (float)b[bbi1];
  int bm2 = 8 + warp_local;
  float bk2[4];
  float bv2[2];
  float bb2[1];
  long long bki2 =
      (long long)n * k_stride_p0 + (long long)bm2 * k_stride_p1 + (long long)h * k_stride_p2;
  {
    uint2 _vld_2;
    _vld_2 = *reinterpret_cast<const uint2*>(k + bki2 + (long long)k_col0);
    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
#pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
      (&bk2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
      (&bk2[0 + _pair * 2])[1] =
          __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
    }
  }
  long long bvi2 =
      (long long)n * v_stride_p0 + (long long)bm2 * v_stride_p1 + (long long)h * v_stride_p2;
  int bvc2 = lane_local * 2;
  bv2[0] = (float)v[bvi2 + (long long)bvc2];
  bv2[1] = (float)v[bvi2 + (long long)bvc2 + 1];
  long long bbi2 =
      (long long)n * b_stride_p0 + (long long)bm2 * b_stride_p1 + (long long)h * b_stride_p2;
  bb2[0] = (float)b[bbi2];
  int bm3 = 12 + warp_local;
  float bk3[4];
  float bv3[2];
  float bb3[1];
  long long bki3 =
      (long long)n * k_stride_p0 + (long long)bm3 * k_stride_p1 + (long long)h * k_stride_p2;
  {
    uint2 _vld_3;
    _vld_3 = *reinterpret_cast<const uint2*>(k + bki3 + (long long)k_col0);
    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
#pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
      (&bk3[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
      (&bk3[0 + _pair * 2])[1] =
          __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
    }
  }
  long long bvi3 =
      (long long)n * v_stride_p0 + (long long)bm3 * v_stride_p1 + (long long)h * v_stride_p2;
  int bvc3 = lane_local * 2;
  bv3[0] = (float)v[bvi3 + (long long)bvc3];
  bv3[1] = (float)v[bvi3 + (long long)bvc3 + 1];
  long long bbi3 =
      (long long)n * b_stride_p0 + (long long)bm3 * b_stride_p1 + (long long)h * b_stride_p2;
  bb3[0] = (float)b[bbi3];
  if (read_state_slot_valid != 0) {
    float S[64];
#if __CUDA_ARCH__ == 900
#else
    float2 S2[32];
#endif
    float k_regs[16];
#if __CUDA_ARCH__ == 900
#else
    float2 k2[8];
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
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 32);
      S[4 + 0] = _v4.x;
      S[4 + 1] = _v4.y;
      S[4 + 2] = _v4.z;
      S[4 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 64);
      S[8 + 0] = _v4.x;
      S[8 + 1] = _v4.y;
      S[8 + 2] = _v4.z;
      S[8 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 96);
      S[12 + 0] = _v4.x;
      S[12 + 1] = _v4.y;
      S[12 + 2] = _v4.z;
      S[12 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + state_stride_p2);
      S[16 + 0] = _v4.x;
      S[16 + 1] = _v4.y;
      S[16 + 2] = _v4.z;
      S[16 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + state_stride_p2 + 32);
      S[20 + 0] = _v4.x;
      S[20 + 1] = _v4.y;
      S[20 + 2] = _v4.z;
      S[20 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + state_stride_p2 + 64);
      S[24 + 0] = _v4.x;
      S[24 + 1] = _v4.y;
      S[24 + 2] = _v4.z;
      S[24 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + state_stride_p2 + 96);
      S[28 + 0] = _v4.x;
      S[28 + 1] = _v4.y;
      S[28 + 2] = _v4.z;
      S[28 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 2 * state_stride_p2);
      S[32 + 0] = _v4.x;
      S[32 + 1] = _v4.y;
      S[32 + 2] = _v4.z;
      S[32 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 2 * state_stride_p2 + 32);
      S[36 + 0] = _v4.x;
      S[36 + 1] = _v4.y;
      S[36 + 2] = _v4.z;
      S[36 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 2 * state_stride_p2 + 64);
      S[40 + 0] = _v4.x;
      S[40 + 1] = _v4.y;
      S[40 + 2] = _v4.z;
      S[40 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 2 * state_stride_p2 + 96);
      S[44 + 0] = _v4.x;
      S[44 + 1] = _v4.y;
      S[44 + 2] = _v4.z;
      S[44 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 3 * state_stride_p2);
      S[48 + 0] = _v4.x;
      S[48 + 1] = _v4.y;
      S[48 + 2] = _v4.z;
      S[48 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 3 * state_stride_p2 + 32);
      S[52 + 0] = _v4.x;
      S[52 + 1] = _v4.y;
      S[52 + 2] = _v4.z;
      S[52 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 3 * state_stride_p2 + 64);
      S[56 + 0] = _v4.x;
      S[56 + 1] = _v4.y;
      S[56 + 2] = _v4.z;
      S[56 + 3] = _v4.w;
    }
    {
      float4 _v4 =
          *reinterpret_cast<const float4*>(state + read_rows_base + 3 * state_stride_p2 + 96);
      S[60 + 0] = _v4.x;
      S[60 + 1] = _v4.y;
      S[60 + 2] = _v4.z;
      S[60 + 3] = _v4.w;
    }
#if __CUDA_ARCH__ == 900
    float k_pair0_x = bk0[0];
    float k_pair0_y = bk0[1];
    float k_pair1_x = bk0[2];
    float k_pair1_y = bk0[3];
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
    float2 k_pair0 = make_float2(bk0[0], bk0[1]);
    float2 k_pair1 = make_float2(bk0[2], bk0[3]);
    float2 sum_k_pair = fma_f32x2_rn_ftz(k_pair0, k_pair0, make_float2(0.0f, 0.0f));
    sum_k_pair = fma_f32x2_rn_ftz(k_pair1, k_pair1, sum_k_pair);
    float sum_k = sum_k_pair.x + sum_k_pair.y;
#endif
    float _warp_reduce_20 = sum_k;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_20 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_20, offset);
    float k_norm = rsqrtf(_warp_reduce_20 + 1e-06f);
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
    sK[bm0 * 128 + k_col0] = k_pair0_x;
    sK[bm0 * 128 + k_col0 + 1] = k_pair0_y;
    sK[bm0 * 128 + k_col0 + 2] = k_pair1_x;
    sK[bm0 * 128 + k_col0 + 3] = k_pair1_y;
#else
    float2 k_norm_pair = make_float2(k_norm, k_norm);
    k_pair0 = mul_f32x2(k_pair0, k_norm_pair);
    k_pair1 = mul_f32x2(k_pair1, k_norm_pair);
    sK[bm0 * 128 + k_col0] = k_pair0.x;
    sK[bm0 * 128 + k_col0 + 1] = k_pair0.y;
    sK[bm0 * 128 + k_col0 + 2] = k_pair1.x;
    sK[bm0 * 128 + k_col0 + 3] = k_pair1.y;
#endif
    int v_col = lane_local * 2;
    sV[bm0 * 64 + v_col] = bv0[0];
    sV[bm0 * 64 + v_col + 1] = bv0[1];
    if (lane_local == 0) {
      float _exp_22 = __expf(-bb0[0]);
      float _rcp_21 = approx_rcp(1.0f + _exp_22);
      sBeta[bm0] = _rcp_21;
    }
#if __CUDA_ARCH__ == 900
    float k_pair0_x_0 = bk1[0];
    float k_pair0_y_1 = bk1[1];
    float k_pair1_x_2 = bk1[2];
    float k_pair1_y_3 = bk1[3];
    float _fma_4;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_4)
        : "f"(k_pair0_x_0), "f"(k_pair0_x_0), "f"(0.0f));
    float _fma_5;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_5)
        : "f"(k_pair0_y_1), "f"(k_pair0_y_1), "f"(0.0f));
    float sum_k_pair_x_4 = _fma_4;
    float sum_k_pair_y_5 = _fma_5;
    float _fma_6;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_6)
        : "f"(k_pair1_x_2), "f"(k_pair1_x_2), "f"(sum_k_pair_x_4));
    float _fma_7;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_7)
        : "f"(k_pair1_y_3), "f"(k_pair1_y_3), "f"(sum_k_pair_y_5));
    sum_k_pair_x_4 = _fma_6;
    sum_k_pair_y_5 = _fma_7;
    float sum_k_6 = sum_k_pair_x_4 + sum_k_pair_y_5;
    float _warp_reduce_23 = sum_k_6;
#else
    float2 k_pair0_0 = make_float2(bk1[0], bk1[1]);
    float2 k_pair1_1 = make_float2(bk1[2], bk1[3]);
    float2 sum_k_pair_2 = fma_f32x2_rn_ftz(k_pair0_0, k_pair0_0, make_float2(0.0f, 0.0f));
    sum_k_pair_2 = fma_f32x2_rn_ftz(k_pair1_1, k_pair1_1, sum_k_pair_2);
    float sum_k_3 = sum_k_pair_2.x + sum_k_pair_2.y;
    float _warp_reduce_23 = sum_k_3;
#endif
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_23 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_23, offset);
#if __CUDA_ARCH__ == 900
    float k_norm_7 = rsqrtf(_warp_reduce_23 + 1e-06f);
    float k_norm_pair_x_8 = k_norm_7;
    float k_norm_pair_y_9 = k_norm_7;
    float _mul_4;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(k_pair0_x_0), "f"(k_norm_pair_x_8));
    float _mul_5;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(k_pair0_y_1), "f"(k_norm_pair_y_9));
    k_pair0_x_0 = _mul_4;
    k_pair0_y_1 = _mul_5;
    float _mul_6;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(k_pair1_x_2), "f"(k_norm_pair_x_8));
    float _mul_7;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(k_pair1_y_3), "f"(k_norm_pair_y_9));
    k_pair1_x_2 = _mul_6;
    k_pair1_y_3 = _mul_7;
    sK[bm1 * 128 + k_col0] = k_pair0_x_0;
    sK[bm1 * 128 + k_col0 + 1] = k_pair0_y_1;
    sK[bm1 * 128 + k_col0 + 2] = k_pair1_x_2;
    sK[bm1 * 128 + k_col0 + 3] = k_pair1_y_3;
    int v_col_10 = lane_local * 2;
    sV[bm1 * 64 + v_col_10] = bv1[0];
    sV[bm1 * 64 + v_col_10 + 1] = bv1[1];
#else
    float k_norm_4 = rsqrtf(_warp_reduce_23 + 1e-06f);
    float2 k_norm_pair_5 = make_float2(k_norm_4, k_norm_4);
    k_pair0_0 = mul_f32x2(k_pair0_0, k_norm_pair_5);
    k_pair1_1 = mul_f32x2(k_pair1_1, k_norm_pair_5);
    sK[bm1 * 128 + k_col0] = k_pair0_0.x;
    sK[bm1 * 128 + k_col0 + 1] = k_pair0_0.y;
    sK[bm1 * 128 + k_col0 + 2] = k_pair1_1.x;
    sK[bm1 * 128 + k_col0 + 3] = k_pair1_1.y;
    int v_col_6 = lane_local * 2;
    sV[bm1 * 64 + v_col_6] = bv1[0];
    sV[bm1 * 64 + v_col_6 + 1] = bv1[1];
#endif
    if (lane_local == 0) {
      float _exp_25 = __expf(-bb1[0]);
      float _rcp_24 = approx_rcp(1.0f + _exp_25);
      sBeta[bm1] = _rcp_24;
    }
#if __CUDA_ARCH__ == 900
    float k_pair0_x_11 = bk2[0];
    float k_pair0_y_12 = bk2[1];
    float k_pair1_x_13 = bk2[2];
    float k_pair1_y_14 = bk2[3];
    float _fma_8;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_8)
        : "f"(k_pair0_x_11), "f"(k_pair0_x_11), "f"(0.0f));
    float _fma_9;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_9)
        : "f"(k_pair0_y_12), "f"(k_pair0_y_12), "f"(0.0f));
    float sum_k_pair_x_15 = _fma_8;
    float sum_k_pair_y_16 = _fma_9;
    float _fma_10;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_10)
        : "f"(k_pair1_x_13), "f"(k_pair1_x_13), "f"(sum_k_pair_x_15));
    float _fma_11;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_11)
        : "f"(k_pair1_y_14), "f"(k_pair1_y_14), "f"(sum_k_pair_y_16));
    sum_k_pair_x_15 = _fma_10;
    sum_k_pair_y_16 = _fma_11;
    float sum_k_17 = sum_k_pair_x_15 + sum_k_pair_y_16;
    float _warp_reduce_26 = sum_k_17;
#else
    float2 k_pair0_7 = make_float2(bk2[0], bk2[1]);
    float2 k_pair1_8 = make_float2(bk2[2], bk2[3]);
    float2 sum_k_pair_9 = fma_f32x2_rn_ftz(k_pair0_7, k_pair0_7, make_float2(0.0f, 0.0f));
    sum_k_pair_9 = fma_f32x2_rn_ftz(k_pair1_8, k_pair1_8, sum_k_pair_9);
    float sum_k_10 = sum_k_pair_9.x + sum_k_pair_9.y;
    float _warp_reduce_26 = sum_k_10;
#endif
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_26 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_26, offset);
#if __CUDA_ARCH__ == 900
    float k_norm_18 = rsqrtf(_warp_reduce_26 + 1e-06f);
    float k_norm_pair_x_19 = k_norm_18;
    float k_norm_pair_y_20 = k_norm_18;
    float _mul_8;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(k_pair0_x_11), "f"(k_norm_pair_x_19));
    float _mul_9;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(k_pair0_y_12), "f"(k_norm_pair_y_20));
    k_pair0_x_11 = _mul_8;
    k_pair0_y_12 = _mul_9;
    float _mul_10;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(k_pair1_x_13), "f"(k_norm_pair_x_19));
    float _mul_11;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(k_pair1_y_14), "f"(k_norm_pair_y_20));
    k_pair1_x_13 = _mul_10;
    k_pair1_y_14 = _mul_11;
    sK[bm2 * 128 + k_col0] = k_pair0_x_11;
    sK[bm2 * 128 + k_col0 + 1] = k_pair0_y_12;
    sK[bm2 * 128 + k_col0 + 2] = k_pair1_x_13;
    sK[bm2 * 128 + k_col0 + 3] = k_pair1_y_14;
    int v_col_21 = lane_local * 2;
    sV[bm2 * 64 + v_col_21] = bv2[0];
    sV[bm2 * 64 + v_col_21 + 1] = bv2[1];
#else
    float k_norm_11 = rsqrtf(_warp_reduce_26 + 1e-06f);
    float2 k_norm_pair_12 = make_float2(k_norm_11, k_norm_11);
    k_pair0_7 = mul_f32x2(k_pair0_7, k_norm_pair_12);
    k_pair1_8 = mul_f32x2(k_pair1_8, k_norm_pair_12);
    sK[bm2 * 128 + k_col0] = k_pair0_7.x;
    sK[bm2 * 128 + k_col0 + 1] = k_pair0_7.y;
    sK[bm2 * 128 + k_col0 + 2] = k_pair1_8.x;
    sK[bm2 * 128 + k_col0 + 3] = k_pair1_8.y;
    int v_col_13 = lane_local * 2;
    sV[bm2 * 64 + v_col_13] = bv2[0];
    sV[bm2 * 64 + v_col_13 + 1] = bv2[1];
#endif
    if (lane_local == 0) {
      float _exp_28 = __expf(-bb2[0]);
      float _rcp_27 = approx_rcp(1.0f + _exp_28);
      sBeta[bm2] = _rcp_27;
    }
#if __CUDA_ARCH__ == 900
    float k_pair0_x_22 = bk3[0];
    float k_pair0_y_23 = bk3[1];
    float k_pair1_x_24 = bk3[2];
    float k_pair1_y_25 = bk3[3];
    float _fma_12;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_12)
        : "f"(k_pair0_x_22), "f"(k_pair0_x_22), "f"(0.0f));
    float _fma_13;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_13)
        : "f"(k_pair0_y_23), "f"(k_pair0_y_23), "f"(0.0f));
    float sum_k_pair_x_26 = _fma_12;
    float sum_k_pair_y_27 = _fma_13;
    float _fma_14;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_14)
        : "f"(k_pair1_x_24), "f"(k_pair1_x_24), "f"(sum_k_pair_x_26));
    float _fma_15;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_15)
        : "f"(k_pair1_y_25), "f"(k_pair1_y_25), "f"(sum_k_pair_y_27));
    sum_k_pair_x_26 = _fma_14;
    sum_k_pair_y_27 = _fma_15;
    float sum_k_28 = sum_k_pair_x_26 + sum_k_pair_y_27;
    float _warp_reduce_29 = sum_k_28;
#else
    float2 k_pair0_14 = make_float2(bk3[0], bk3[1]);
    float2 k_pair1_15 = make_float2(bk3[2], bk3[3]);
    float2 sum_k_pair_16 = fma_f32x2_rn_ftz(k_pair0_14, k_pair0_14, make_float2(0.0f, 0.0f));
    sum_k_pair_16 = fma_f32x2_rn_ftz(k_pair1_15, k_pair1_15, sum_k_pair_16);
    float sum_k_17 = sum_k_pair_16.x + sum_k_pair_16.y;
    float _warp_reduce_29 = sum_k_17;
#endif
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_29 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_29, offset);
#if __CUDA_ARCH__ == 900
    float k_norm_29 = rsqrtf(_warp_reduce_29 + 1e-06f);
    float k_norm_pair_x_30 = k_norm_29;
    float k_norm_pair_y_31 = k_norm_29;
    float _mul_12;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(k_pair0_x_22), "f"(k_norm_pair_x_30));
    float _mul_13;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(k_pair0_y_23), "f"(k_norm_pair_y_31));
    k_pair0_x_22 = _mul_12;
    k_pair0_y_23 = _mul_13;
    float _mul_14;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(k_pair1_x_24), "f"(k_norm_pair_x_30));
    float _mul_15;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(k_pair1_y_25), "f"(k_norm_pair_y_31));
    k_pair1_x_24 = _mul_14;
    k_pair1_y_25 = _mul_15;
    sK[bm3 * 128 + k_col0] = k_pair0_x_22;
    sK[bm3 * 128 + k_col0 + 1] = k_pair0_y_23;
    sK[bm3 * 128 + k_col0 + 2] = k_pair1_x_24;
    sK[bm3 * 128 + k_col0 + 3] = k_pair1_y_25;
    int v_col_32 = lane_local * 2;
    sV[bm3 * 64 + v_col_32] = bv3[0];
    sV[bm3 * 64 + v_col_32 + 1] = bv3[1];
#else
    float k_norm_18 = rsqrtf(_warp_reduce_29 + 1e-06f);
    float2 k_norm_pair_19 = make_float2(k_norm_18, k_norm_18);
    k_pair0_14 = mul_f32x2(k_pair0_14, k_norm_pair_19);
    k_pair1_15 = mul_f32x2(k_pair1_15, k_norm_pair_19);
    sK[bm3 * 128 + k_col0] = k_pair0_14.x;
    sK[bm3 * 128 + k_col0 + 1] = k_pair0_14.y;
    sK[bm3 * 128 + k_col0 + 2] = k_pair1_15.x;
    sK[bm3 * 128 + k_col0 + 3] = k_pair1_15.y;
    int v_col_20 = lane_local * 2;
    sV[bm3 * 64 + v_col_20] = bv3[0];
    sV[bm3 * 64 + v_col_20 + 1] = bv3[1];
#endif
    if (lane_local == 0) {
      float _exp_31 = __expf(-bb3[0]);
      float _rcp_30 = approx_rcp(1.0f + _exp_31);
      sBeta[bm3] = _rcp_30;
    }
    int bm4 = 16 + warp_local;
    float bk4[4];
    float bv4[2];
    float bb4[1];
    long long bki4 =
        (long long)n * k_stride_p0 + (long long)bm4 * k_stride_p1 + (long long)h * k_stride_p2;
    {
      uint2 _vld_32;
      _vld_32 = *reinterpret_cast<const uint2*>(k + bki4 + (long long)k_col0);
      uint32_t* _vpairs_32 = reinterpret_cast<uint32_t*>(&_vld_32);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&bk4[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) << 16);
        (&bk4[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) & 0xffff0000u);
      }
    }
    long long bvi4 =
        (long long)n * v_stride_p0 + (long long)bm4 * v_stride_p1 + (long long)h * v_stride_p2;
    int bvc4 = lane_local * 2;
    bv4[0] = (float)v[bvi4 + (long long)bvc4];
    bv4[1] = (float)v[bvi4 + (long long)bvc4 + 1];
    long long bbi4 =
        (long long)n * b_stride_p0 + (long long)bm4 * b_stride_p1 + (long long)h * b_stride_p2;
    bb4[0] = (float)b[bbi4];
    int bm5 = 20 + warp_local;
    float bk5[4];
    float bv5[2];
    float bb5[1];
    long long bki5 =
        (long long)n * k_stride_p0 + (long long)bm5 * k_stride_p1 + (long long)h * k_stride_p2;
    {
      uint2 _vld_33;
      _vld_33 = *reinterpret_cast<const uint2*>(k + bki5 + (long long)k_col0);
      uint32_t* _vpairs_33 = reinterpret_cast<uint32_t*>(&_vld_33);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&bk5[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) << 16);
        (&bk5[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) & 0xffff0000u);
      }
    }
    long long bvi5 =
        (long long)n * v_stride_p0 + (long long)bm5 * v_stride_p1 + (long long)h * v_stride_p2;
    int bvc5 = lane_local * 2;
    bv5[0] = (float)v[bvi5 + (long long)bvc5];
    bv5[1] = (float)v[bvi5 + (long long)bvc5 + 1];
    long long bbi5 =
        (long long)n * b_stride_p0 + (long long)bm5 * b_stride_p1 + (long long)h * b_stride_p2;
    bb5[0] = (float)b[bbi5];
    int bt0 = warp_local;
    float bq0[4];
    float ba0[3];
    long long bqi0 =
        (long long)n * q_stride_p0 + (long long)bt0 * q_stride_p1 + (long long)h * q_stride_p2;
    {
      uint2 _vld_34;
      _vld_34 = *reinterpret_cast<const uint2*>(q + bqi0 + (long long)k_col0);
      uint32_t* _vpairs_34 = reinterpret_cast<uint32_t*>(&_vld_34);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&bq0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) << 16);
        (&bq0[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) & 0xffff0000u);
      }
    }
    long long bgi0 =
        (long long)n * a_stride_p0 + (long long)bt0 * a_stride_p1 + (long long)h * a_stride_p2;
    ba0[0] = (float)a[bgi0];
    ba0[1] = dt_bias[h];
    ba0[2] = A_log[h];
    int bt1 = 4 + warp_local;
    float bq1[4];
    float ba1[3];
    long long bqi1 =
        (long long)n * q_stride_p0 + (long long)bt1 * q_stride_p1 + (long long)h * q_stride_p2;
    {
      uint2 _vld_35;
      _vld_35 = *reinterpret_cast<const uint2*>(q + bqi1 + (long long)k_col0);
      uint32_t* _vpairs_35 = reinterpret_cast<uint32_t*>(&_vld_35);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&bq1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_35[_pair]) << 16);
        (&bq1[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_35[_pair]) & 0xffff0000u);
      }
    }
    long long bgi1 =
        (long long)n * a_stride_p0 + (long long)bt1 * a_stride_p1 + (long long)h * a_stride_p2;
    ba1[0] = (float)a[bgi1];
    ba1[1] = dt_bias[h];
    ba1[2] = A_log[h];
#if __CUDA_ARCH__ == 900
    float k_pair0_x_33 = bk4[0];
    float k_pair0_y_34 = bk4[1];
    float k_pair1_x_35 = bk4[2];
    float k_pair1_y_36 = bk4[3];
    float _fma_16;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_16)
        : "f"(k_pair0_x_33), "f"(k_pair0_x_33), "f"(0.0f));
    float _fma_17;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_17)
        : "f"(k_pair0_y_34), "f"(k_pair0_y_34), "f"(0.0f));
    float sum_k_pair_x_37 = _fma_16;
    float sum_k_pair_y_38 = _fma_17;
    float _fma_18;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_18)
        : "f"(k_pair1_x_35), "f"(k_pair1_x_35), "f"(sum_k_pair_x_37));
    float _fma_19;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_19)
        : "f"(k_pair1_y_36), "f"(k_pair1_y_36), "f"(sum_k_pair_y_38));
    sum_k_pair_x_37 = _fma_18;
    sum_k_pair_y_38 = _fma_19;
    float sum_k_39 = sum_k_pair_x_37 + sum_k_pair_y_38;
    float _warp_reduce_36 = sum_k_39;
#else
    float2 k_pair0_21 = make_float2(bk4[0], bk4[1]);
    float2 k_pair1_22 = make_float2(bk4[2], bk4[3]);
    float2 sum_k_pair_23 = fma_f32x2_rn_ftz(k_pair0_21, k_pair0_21, make_float2(0.0f, 0.0f));
    sum_k_pair_23 = fma_f32x2_rn_ftz(k_pair1_22, k_pair1_22, sum_k_pair_23);
    float sum_k_24 = sum_k_pair_23.x + sum_k_pair_23.y;
    float _warp_reduce_36 = sum_k_24;
#endif
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_36 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_36, offset);
#if __CUDA_ARCH__ == 900
    float k_norm_40 = rsqrtf(_warp_reduce_36 + 1e-06f);
    float k_norm_pair_x_41 = k_norm_40;
    float k_norm_pair_y_42 = k_norm_40;
    float _mul_16;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(k_pair0_x_33), "f"(k_norm_pair_x_41));
    float _mul_17;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(k_pair0_y_34), "f"(k_norm_pair_y_42));
    k_pair0_x_33 = _mul_16;
    k_pair0_y_34 = _mul_17;
    float _mul_18;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(k_pair1_x_35), "f"(k_norm_pair_x_41));
    float _mul_19;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(k_pair1_y_36), "f"(k_norm_pair_y_42));
    k_pair1_x_35 = _mul_18;
    k_pair1_y_36 = _mul_19;
    sK[bm4 * 128 + k_col0] = k_pair0_x_33;
    sK[bm4 * 128 + k_col0 + 1] = k_pair0_y_34;
    sK[bm4 * 128 + k_col0 + 2] = k_pair1_x_35;
    sK[bm4 * 128 + k_col0 + 3] = k_pair1_y_36;
    int v_col_43 = lane_local * 2;
    sV[bm4 * 64 + v_col_43] = bv4[0];
    sV[bm4 * 64 + v_col_43 + 1] = bv4[1];
#else
    float k_norm_25 = rsqrtf(_warp_reduce_36 + 1e-06f);
    float2 k_norm_pair_26 = make_float2(k_norm_25, k_norm_25);
    k_pair0_21 = mul_f32x2(k_pair0_21, k_norm_pair_26);
    k_pair1_22 = mul_f32x2(k_pair1_22, k_norm_pair_26);
    sK[bm4 * 128 + k_col0] = k_pair0_21.x;
    sK[bm4 * 128 + k_col0 + 1] = k_pair0_21.y;
    sK[bm4 * 128 + k_col0 + 2] = k_pair1_22.x;
    sK[bm4 * 128 + k_col0 + 3] = k_pair1_22.y;
    int v_col_27 = lane_local * 2;
    sV[bm4 * 64 + v_col_27] = bv4[0];
    sV[bm4 * 64 + v_col_27 + 1] = bv4[1];
#endif
    if (lane_local == 0) {
      float _exp_38 = __expf(-bb4[0]);
      float _rcp_37 = approx_rcp(1.0f + _exp_38);
      sBeta[bm4] = _rcp_37;
    }
#if __CUDA_ARCH__ == 900
    float k_pair0_x_44 = bk5[0];
    float k_pair0_y_45 = bk5[1];
    float k_pair1_x_46 = bk5[2];
    float k_pair1_y_47 = bk5[3];
    float _fma_20;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_20)
        : "f"(k_pair0_x_44), "f"(k_pair0_x_44), "f"(0.0f));
    float _fma_21;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_21)
        : "f"(k_pair0_y_45), "f"(k_pair0_y_45), "f"(0.0f));
    float sum_k_pair_x_48 = _fma_20;
    float sum_k_pair_y_49 = _fma_21;
    float _fma_22;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_22)
        : "f"(k_pair1_x_46), "f"(k_pair1_x_46), "f"(sum_k_pair_x_48));
    float _fma_23;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_23)
        : "f"(k_pair1_y_47), "f"(k_pair1_y_47), "f"(sum_k_pair_y_49));
    sum_k_pair_x_48 = _fma_22;
    sum_k_pair_y_49 = _fma_23;
    float sum_k_50 = sum_k_pair_x_48 + sum_k_pair_y_49;
    float _warp_reduce_39 = sum_k_50;
#else
    float2 k_pair0_28 = make_float2(bk5[0], bk5[1]);
    float2 k_pair1_29 = make_float2(bk5[2], bk5[3]);
    float2 sum_k_pair_30 = fma_f32x2_rn_ftz(k_pair0_28, k_pair0_28, make_float2(0.0f, 0.0f));
    sum_k_pair_30 = fma_f32x2_rn_ftz(k_pair1_29, k_pair1_29, sum_k_pair_30);
    float sum_k_31 = sum_k_pair_30.x + sum_k_pair_30.y;
    float _warp_reduce_39 = sum_k_31;
#endif
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_39 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_39, offset);
#if __CUDA_ARCH__ == 900
    float k_norm_51 = rsqrtf(_warp_reduce_39 + 1e-06f);
    float k_norm_pair_x_52 = k_norm_51;
    float k_norm_pair_y_53 = k_norm_51;
    float _mul_20;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(k_pair0_x_44), "f"(k_norm_pair_x_52));
    float _mul_21;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(k_pair0_y_45), "f"(k_norm_pair_y_53));
    k_pair0_x_44 = _mul_20;
    k_pair0_y_45 = _mul_21;
    float _mul_22;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(k_pair1_x_46), "f"(k_norm_pair_x_52));
    float _mul_23;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(k_pair1_y_47), "f"(k_norm_pair_y_53));
    k_pair1_x_46 = _mul_22;
    k_pair1_y_47 = _mul_23;
    sK[bm5 * 128 + k_col0] = k_pair0_x_44;
    sK[bm5 * 128 + k_col0 + 1] = k_pair0_y_45;
    sK[bm5 * 128 + k_col0 + 2] = k_pair1_x_46;
    sK[bm5 * 128 + k_col0 + 3] = k_pair1_y_47;
    int v_col_54 = lane_local * 2;
    sV[bm5 * 64 + v_col_54] = bv5[0];
    sV[bm5 * 64 + v_col_54 + 1] = bv5[1];
#else
    float k_norm_32 = rsqrtf(_warp_reduce_39 + 1e-06f);
    float2 k_norm_pair_33 = make_float2(k_norm_32, k_norm_32);
    k_pair0_28 = mul_f32x2(k_pair0_28, k_norm_pair_33);
    k_pair1_29 = mul_f32x2(k_pair1_29, k_norm_pair_33);
    sK[bm5 * 128 + k_col0] = k_pair0_28.x;
    sK[bm5 * 128 + k_col0 + 1] = k_pair0_28.y;
    sK[bm5 * 128 + k_col0 + 2] = k_pair1_29.x;
    sK[bm5 * 128 + k_col0 + 3] = k_pair1_29.y;
    int v_col_34 = lane_local * 2;
    sV[bm5 * 64 + v_col_34] = bv5[0];
    sV[bm5 * 64 + v_col_34 + 1] = bv5[1];
#endif
    if (lane_local == 0) {
      float _exp_41 = __expf(-bb5[0]);
      float _rcp_40 = approx_rcp(1.0f + _exp_41);
      sBeta[bm5] = _rcp_40;
    }
#if __CUDA_ARCH__ == 900
    float q_raw_pair0_x = bq0[0];
    float q_raw_pair0_y = bq0[1];
    float q_raw_pair1_x = bq0[2];
    float q_raw_pair1_y = bq0[3];
#else
    float2 q_raw_pair0 = make_float2(bq0[0], bq0[1]);
    float2 q_raw_pair1 = make_float2(bq0[2], bq0[3]);
#endif
    float q_norm = scale;
#if __CUDA_ARCH__ == 900
    float _fma_24;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_24)
        : "f"(q_raw_pair0_x), "f"(q_raw_pair0_x), "f"(0.0f));
    float _fma_25;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_25)
        : "f"(q_raw_pair0_y), "f"(q_raw_pair0_y), "f"(0.0f));
    float sum_q_pair_x = _fma_24;
    float sum_q_pair_y = _fma_25;
    float _fma_26;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_26)
        : "f"(q_raw_pair1_x), "f"(q_raw_pair1_x), "f"(sum_q_pair_x));
    float _fma_27;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_27)
        : "f"(q_raw_pair1_y), "f"(q_raw_pair1_y), "f"(sum_q_pair_y));
    sum_q_pair_x = _fma_26;
    sum_q_pair_y = _fma_27;
    float sum_q = sum_q_pair_x + sum_q_pair_y;
#else
    float2 sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair0, q_raw_pair0, make_float2(0.0f, 0.0f));
    sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair1, q_raw_pair1, sum_q_pair);
    float sum_q = sum_q_pair.x + sum_q_pair.y;
#endif
    float _warp_reduce_42 = sum_q;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_42 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_42, offset);
    float _rsqrt_0 = rsqrtf(_warp_reduce_42 + 1e-06f);
#if __CUDA_ARCH__ == 900
    float q_norm_pair_x = _rsqrt_0 * scale;
    float q_norm_pair_y = _rsqrt_0 * scale;
    float _mul_24;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_24) : "f"(q_raw_pair0_x), "f"(q_norm_pair_x));
    float _mul_25;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_25) : "f"(q_raw_pair0_y), "f"(q_norm_pair_y));
    float q_pair0_x = _mul_24;
    float q_pair0_y = _mul_25;
    float _mul_26;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_26) : "f"(q_raw_pair1_x), "f"(q_norm_pair_x));
    float _mul_27;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_27) : "f"(q_raw_pair1_y), "f"(q_norm_pair_y));
    float q_pair1_x = _mul_26;
    float q_pair1_y = _mul_27;
    sQ[bt0 * 128 + k_col0] = q_pair0_x;
    sQ[bt0 * 128 + k_col0 + 1] = q_pair0_y;
    sQ[bt0 * 128 + k_col0 + 2] = q_pair1_x;
    sQ[bt0 * 128 + k_col0 + 3] = q_pair1_y;
#else
    float2 q_norm_pair = make_float2(_rsqrt_0 * scale, _rsqrt_0 * scale);
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
    sQ[bt0 * 128 + k_col0] = q_pair0.x;
    sQ[bt0 * 128 + k_col0 + 1] = q_pair0.y;
    sQ[bt0 * 128 + k_col0 + 2] = q_pair1.x;
    sQ[bt0 * 128 + k_col0 + 3] = q_pair1.y;
#endif
    if (lane_local == 0) {
      float x = ba0[0] + ba0[1];
      float softplus_x = x;
      if (x <= 20.0f) {
        float _exp_44 = __expf(x);
        float _log2_43;
        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_43) : "f"(1.0f + _exp_44));
        softplus_x = _log2_43 * 0.6931472f;
      }
      float _exp_45 = __expf(ba0[2]);
      float g_log = (-_exp_45) * softplus_x;
      float _exp_46 = __expf(g_log);
      sAlpha[bt0] = _exp_46;
      if (use_ssm_state_indices != 0) {
        sSsm[bt0] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 +
                                      (long long)bt0 * ssm_state_indices_stride_p1];
      }
      if (use_ssm_state_indices == 0) {
        sSsm[bt0] = -1;
      }
    }
#if __CUDA_ARCH__ == 900
    float q_raw_pair0_x_55 = bq1[0];
    float q_raw_pair0_y_56 = bq1[1];
    float q_raw_pair1_x_57 = bq1[2];
    float q_raw_pair1_y_58 = bq1[3];
    float q_norm_59 = scale;
    float _fma_28;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_28)
        : "f"(q_raw_pair0_x_55), "f"(q_raw_pair0_x_55), "f"(0.0f));
    float _fma_29;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_29)
        : "f"(q_raw_pair0_y_56), "f"(q_raw_pair0_y_56), "f"(0.0f));
    float sum_q_pair_x_60 = _fma_28;
    float sum_q_pair_y_61 = _fma_29;
    float _fma_30;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_30)
        : "f"(q_raw_pair1_x_57), "f"(q_raw_pair1_x_57), "f"(sum_q_pair_x_60));
    float _fma_31;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_31)
        : "f"(q_raw_pair1_y_58), "f"(q_raw_pair1_y_58), "f"(sum_q_pair_y_61));
    sum_q_pair_x_60 = _fma_30;
    sum_q_pair_y_61 = _fma_31;
    float sum_q_62 = sum_q_pair_x_60 + sum_q_pair_y_61;
    float _warp_reduce_47 = sum_q_62;
#else
    float2 q_raw_pair0_35 = make_float2(bq1[0], bq1[1]);
    float2 q_raw_pair1_36 = make_float2(bq1[2], bq1[3]);
    float q_norm_37 = scale;
    float2 sum_q_pair_38 =
        fma_f32x2_rn_ftz(q_raw_pair0_35, q_raw_pair0_35, make_float2(0.0f, 0.0f));
    sum_q_pair_38 = fma_f32x2_rn_ftz(q_raw_pair1_36, q_raw_pair1_36, sum_q_pair_38);
    float sum_q_39 = sum_q_pair_38.x + sum_q_pair_38.y;
    float _warp_reduce_47 = sum_q_39;
#endif
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_47 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_47, offset);
    float _rsqrt_1 = rsqrtf(_warp_reduce_47 + 1e-06f);
#if __CUDA_ARCH__ == 900
    float q_norm_pair_x_63 = _rsqrt_1 * scale;
    float q_norm_pair_y_64 = _rsqrt_1 * scale;
    float _mul_28;
    asm("mul.rn.ftz.f32 %0, %1, %2;"
        : "=f"(_mul_28)
        : "f"(q_raw_pair0_x_55), "f"(q_norm_pair_x_63));
    float _mul_29;
    asm("mul.rn.ftz.f32 %0, %1, %2;"
        : "=f"(_mul_29)
        : "f"(q_raw_pair0_y_56), "f"(q_norm_pair_y_64));
    float q_pair0_x_65 = _mul_28;
    float q_pair0_y_66 = _mul_29;
    float _mul_30;
    asm("mul.rn.ftz.f32 %0, %1, %2;"
        : "=f"(_mul_30)
        : "f"(q_raw_pair1_x_57), "f"(q_norm_pair_x_63));
    float _mul_31;
    asm("mul.rn.ftz.f32 %0, %1, %2;"
        : "=f"(_mul_31)
        : "f"(q_raw_pair1_y_58), "f"(q_norm_pair_y_64));
    float q_pair1_x_67 = _mul_30;
    float q_pair1_y_68 = _mul_31;
    sQ[bt1 * 128 + k_col0] = q_pair0_x_65;
    sQ[bt1 * 128 + k_col0 + 1] = q_pair0_y_66;
    sQ[bt1 * 128 + k_col0 + 2] = q_pair1_x_67;
    sQ[bt1 * 128 + k_col0 + 3] = q_pair1_y_68;
#else
    float2 q_norm_pair_40 = make_float2(_rsqrt_1 * scale, _rsqrt_1 * scale);
    float2 q_pair0_41;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&q_pair0_41)
        : "l"(*(const unsigned long long*)&q_raw_pair0_35),
          "l"(*(const unsigned long long*)&q_norm_pair_40));
    float2 q_pair1_42;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&q_pair1_42)
        : "l"(*(const unsigned long long*)&q_raw_pair1_36),
          "l"(*(const unsigned long long*)&q_norm_pair_40));
    sQ[bt1 * 128 + k_col0] = q_pair0_41.x;
    sQ[bt1 * 128 + k_col0 + 1] = q_pair0_41.y;
    sQ[bt1 * 128 + k_col0 + 2] = q_pair1_42.x;
    sQ[bt1 * 128 + k_col0 + 3] = q_pair1_42.y;
#endif
    if (lane_local == 0) {
      float x_1 = ba1[0] + ba1[1];
      float softplus_x_1 = x_1;
      if (x_1 <= 20.0f) {
        float _exp_49 = __expf(x_1);
        float _log2_48;
        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_48) : "f"(1.0f + _exp_49));
        softplus_x_1 = _log2_48 * 0.6931472f;
      }
      float _exp_50 = __expf(ba1[2]);
      float g_log_1 = (-_exp_50) * softplus_x_1;
      float _exp_51 = __expf(g_log_1);
      sAlpha[bt1] = _exp_51;
      if (use_ssm_state_indices != 0) {
        sSsm[bt1] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 +
                                      (long long)bt1 * ssm_state_indices_stride_p1];
      }
      if (use_ssm_state_indices == 0) {
        sSsm[bt1] = -1;
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
#endif
    if (use_ssm_state_indices != 0) {
#pragma unroll 1
      for (int t = 0; t < 8; t++) {
        float alphas = sAlpha[t];
#if __CUDA_ARCH__ == 900
        float alpha2s_x = alphas;
        float alpha2s_y = alphas;
        float _mul_32;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_32) : "f"(S[0]), "f"(alpha2s_x));
        float _mul_33;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_33) : "f"(S[1]), "f"(alpha2s_y));
        S[0] = _mul_32;
        S[1] = _mul_33;
        float _mul_34;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_34) : "f"(S[2]), "f"(alpha2s_x));
        float _mul_35;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_35) : "f"(S[3]), "f"(alpha2s_y));
        S[2] = _mul_34;
        S[3] = _mul_35;
        float _mul_36;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_36) : "f"(S[4]), "f"(alpha2s_x));
        float _mul_37;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_37) : "f"(S[5]), "f"(alpha2s_y));
        S[4] = _mul_36;
        S[5] = _mul_37;
        float _mul_38;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_38) : "f"(S[6]), "f"(alpha2s_x));
        float _mul_39;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(S[7]), "f"(alpha2s_y));
        S[6] = _mul_38;
        S[7] = _mul_39;
        float _mul_40;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_40) : "f"(S[8]), "f"(alpha2s_x));
        float _mul_41;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_41) : "f"(S[9]), "f"(alpha2s_y));
        S[8] = _mul_40;
        S[9] = _mul_41;
        float _mul_42;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_42) : "f"(S[10]), "f"(alpha2s_x));
        float _mul_43;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_43) : "f"(S[11]), "f"(alpha2s_y));
        S[10] = _mul_42;
        S[11] = _mul_43;
        float _mul_44;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_44) : "f"(S[12]), "f"(alpha2s_x));
        float _mul_45;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_45) : "f"(S[13]), "f"(alpha2s_y));
        S[12] = _mul_44;
        S[13] = _mul_45;
        float _mul_46;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_46) : "f"(S[14]), "f"(alpha2s_x));
        float _mul_47;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_47) : "f"(S[15]), "f"(alpha2s_y));
        S[14] = _mul_46;
        S[15] = _mul_47;
        float _mul_48;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_48) : "f"(S[16]), "f"(alpha2s_x));
        float _mul_49;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_49) : "f"(S[17]), "f"(alpha2s_y));
        S[16] = _mul_48;
        S[17] = _mul_49;
        float _mul_50;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_50) : "f"(S[18]), "f"(alpha2s_x));
        float _mul_51;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_51) : "f"(S[19]), "f"(alpha2s_y));
        S[18] = _mul_50;
        S[19] = _mul_51;
        float _mul_52;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_52) : "f"(S[20]), "f"(alpha2s_x));
        float _mul_53;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_53) : "f"(S[21]), "f"(alpha2s_y));
        S[20] = _mul_52;
        S[21] = _mul_53;
        float _mul_54;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_54) : "f"(S[22]), "f"(alpha2s_x));
        float _mul_55;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_55) : "f"(S[23]), "f"(alpha2s_y));
        S[22] = _mul_54;
        S[23] = _mul_55;
        float _mul_56;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_56) : "f"(S[24]), "f"(alpha2s_x));
        float _mul_57;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_57) : "f"(S[25]), "f"(alpha2s_y));
        S[24] = _mul_56;
        S[25] = _mul_57;
        float _mul_58;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_58) : "f"(S[26]), "f"(alpha2s_x));
        float _mul_59;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_59) : "f"(S[27]), "f"(alpha2s_y));
        S[26] = _mul_58;
        S[27] = _mul_59;
        float _mul_60;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_60) : "f"(S[28]), "f"(alpha2s_x));
        float _mul_61;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_61) : "f"(S[29]), "f"(alpha2s_y));
        S[28] = _mul_60;
        S[29] = _mul_61;
        float _mul_62;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_62) : "f"(S[30]), "f"(alpha2s_x));
        float _mul_63;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_63) : "f"(S[31]), "f"(alpha2s_y));
        S[30] = _mul_62;
        S[31] = _mul_63;
        float _mul_64;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_64) : "f"(S[32]), "f"(alpha2s_x));
        float _mul_65;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_65) : "f"(S[33]), "f"(alpha2s_y));
        S[32] = _mul_64;
        S[33] = _mul_65;
        float _mul_66;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_66) : "f"(S[34]), "f"(alpha2s_x));
        float _mul_67;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_67) : "f"(S[35]), "f"(alpha2s_y));
        S[34] = _mul_66;
        S[35] = _mul_67;
        float _mul_68;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_68) : "f"(S[36]), "f"(alpha2s_x));
        float _mul_69;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_69) : "f"(S[37]), "f"(alpha2s_y));
        S[36] = _mul_68;
        S[37] = _mul_69;
        float _mul_70;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_70) : "f"(S[38]), "f"(alpha2s_x));
        float _mul_71;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_71) : "f"(S[39]), "f"(alpha2s_y));
        S[38] = _mul_70;
        S[39] = _mul_71;
        float _mul_72;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_72) : "f"(S[40]), "f"(alpha2s_x));
        float _mul_73;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_73) : "f"(S[41]), "f"(alpha2s_y));
        S[40] = _mul_72;
        S[41] = _mul_73;
        float _mul_74;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_74) : "f"(S[42]), "f"(alpha2s_x));
        float _mul_75;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_75) : "f"(S[43]), "f"(alpha2s_y));
        S[42] = _mul_74;
        S[43] = _mul_75;
        float _mul_76;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_76) : "f"(S[44]), "f"(alpha2s_x));
        float _mul_77;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_77) : "f"(S[45]), "f"(alpha2s_y));
        S[44] = _mul_76;
        S[45] = _mul_77;
        float _mul_78;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_78) : "f"(S[46]), "f"(alpha2s_x));
        float _mul_79;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_79) : "f"(S[47]), "f"(alpha2s_y));
        S[46] = _mul_78;
        S[47] = _mul_79;
        float _mul_80;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_80) : "f"(S[48]), "f"(alpha2s_x));
        float _mul_81;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_81) : "f"(S[49]), "f"(alpha2s_y));
        S[48] = _mul_80;
        S[49] = _mul_81;
        float _mul_82;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_82) : "f"(S[50]), "f"(alpha2s_x));
        float _mul_83;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_83) : "f"(S[51]), "f"(alpha2s_y));
        S[50] = _mul_82;
        S[51] = _mul_83;
        float _mul_84;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_84) : "f"(S[52]), "f"(alpha2s_x));
        float _mul_85;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_85) : "f"(S[53]), "f"(alpha2s_y));
        S[52] = _mul_84;
        S[53] = _mul_85;
        float _mul_86;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_86) : "f"(S[54]), "f"(alpha2s_x));
        float _mul_87;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_87) : "f"(S[55]), "f"(alpha2s_y));
        S[54] = _mul_86;
        S[55] = _mul_87;
        float _mul_88;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_88) : "f"(S[56]), "f"(alpha2s_x));
        float _mul_89;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_89) : "f"(S[57]), "f"(alpha2s_y));
        S[56] = _mul_88;
        S[57] = _mul_89;
        float _mul_90;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_90) : "f"(S[58]), "f"(alpha2s_x));
        float _mul_91;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_91) : "f"(S[59]), "f"(alpha2s_y));
        S[58] = _mul_90;
        S[59] = _mul_91;
        float _mul_92;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_92) : "f"(S[60]), "f"(alpha2s_x));
        float _mul_93;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_93) : "f"(S[61]), "f"(alpha2s_y));
        S[60] = _mul_92;
        S[61] = _mul_93;
        float _mul_94;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_94) : "f"(S[62]), "f"(alpha2s_x));
        float _mul_95;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_95) : "f"(S[63]), "f"(alpha2s_y));
        S[62] = _mul_94;
        S[63] = _mul_95;
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
#endif
        int ms0 = t * 3;
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
                     : "r"(kbases0 + 128));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                     : "r"(kbases0 + 256));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                     : "r"(kbases0 + 384));
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
        float _fma_32;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_32)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accs0_0_0_x));
        float _fma_33;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_33)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accs0_0_0_y));
        accs0_0_0_x = _fma_32;
        accs0_0_0_y = _fma_33;
        float _fma_34;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_34)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accs0_0_1_x));
        float _fma_35;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_35)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accs0_0_1_y));
        accs0_0_1_x = _fma_34;
        accs0_0_1_y = _fma_35;
        float _fma_36;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_36)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accs0_0_0_x));
        float _fma_37;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_37)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accs0_0_0_y));
        accs0_0_0_x = _fma_36;
        accs0_0_0_y = _fma_37;
        float _fma_38;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_38)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accs0_0_1_x));
        float _fma_39;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_39)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accs0_0_1_y));
        accs0_0_1_x = _fma_38;
        accs0_0_1_y = _fma_39;
        float _fma_40;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_40)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accs0_0_0_x));
        float _fma_41;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_41)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accs0_0_0_y));
        accs0_0_0_x = _fma_40;
        accs0_0_0_y = _fma_41;
        float _fma_42;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_42)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accs0_0_1_x));
        float _fma_43;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_43)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accs0_0_1_y));
        accs0_0_1_x = _fma_42;
        accs0_0_1_y = _fma_43;
        float _fma_44;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_44)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accs0_0_0_x));
        float _fma_45;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_45)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accs0_0_0_y));
        accs0_0_0_x = _fma_44;
        accs0_0_0_y = _fma_45;
        float _fma_46;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_46)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accs0_0_1_x));
        float _fma_47;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_47)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accs0_0_1_y));
        accs0_0_1_x = _fma_46;
        accs0_0_1_y = _fma_47;
        float _add_52;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(accs0_0_0_x), "f"(accs0_0_1_x));
        float _add_53;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(accs0_0_0_y), "f"(accs0_0_1_y));
        float dots0_0 = _add_52 + _add_53;
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 1);
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
        float dots0_0 = accs0_0_0.x + accs0_0_0.y;
        float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 1);
        dots0_0 = dots0_0 + _shfl_xor_sync_52;
        float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 2);
        dots0_0 = dots0_0 + _shfl_xor_sync_53;
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 4);
#endif
        dots0_0 = dots0_0 + _shfl_xor_sync_54;
#if __CUDA_ARCH__ == 900
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 2);
        dots0_0 = dots0_0 + _shfl_xor_sync_55;
        float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 4);
        dots0_0 = dots0_0 + _shfl_xor_sync_56;
#endif
        float vns0_0 = (v_regs[0] - dots0_0) * betas0;
#if __CUDA_ARCH__ == 900
        float vn2s0_0_x = vns0_0;
        float vn2s0_0_y = vns0_0;
        float _fma_48;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_48)
            : "f"(k_regs[0]), "f"(vn2s0_0_x), "f"(S[0]));
        float _fma_49;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_49)
            : "f"(k_regs[1]), "f"(vn2s0_0_y), "f"(S[1]));
        S[0] = _fma_48;
        S[1] = _fma_49;
        float _fma_50;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_50)
            : "f"(k_regs[2]), "f"(vn2s0_0_x), "f"(S[2]));
        float _fma_51;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_51)
            : "f"(k_regs[3]), "f"(vn2s0_0_y), "f"(S[3]));
        S[2] = _fma_50;
        S[3] = _fma_51;
        float _fma_52;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_52)
            : "f"(k_regs[4]), "f"(vn2s0_0_x), "f"(S[4]));
        float _fma_53;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_53)
            : "f"(k_regs[5]), "f"(vn2s0_0_y), "f"(S[5]));
        S[4] = _fma_52;
        S[5] = _fma_53;
        float _fma_54;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_54)
            : "f"(k_regs[6]), "f"(vn2s0_0_x), "f"(S[6]));
        float _fma_55;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_55)
            : "f"(k_regs[7]), "f"(vn2s0_0_y), "f"(S[7]));
        S[6] = _fma_54;
        S[7] = _fma_55;
        float _fma_56;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_56)
            : "f"(k_regs[8]), "f"(vn2s0_0_x), "f"(S[8]));
        float _fma_57;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_57)
            : "f"(k_regs[9]), "f"(vn2s0_0_y), "f"(S[9]));
        S[8] = _fma_56;
        S[9] = _fma_57;
        float _fma_58;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_58)
            : "f"(k_regs[10]), "f"(vn2s0_0_x), "f"(S[10]));
        float _fma_59;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_59)
            : "f"(k_regs[11]), "f"(vn2s0_0_y), "f"(S[11]));
        S[10] = _fma_58;
        S[11] = _fma_59;
        float _fma_60;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_60)
            : "f"(k_regs[12]), "f"(vn2s0_0_x), "f"(S[12]));
        float _fma_61;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_61)
            : "f"(k_regs[13]), "f"(vn2s0_0_y), "f"(S[13]));
        S[12] = _fma_60;
        S[13] = _fma_61;
        float _fma_62;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_62)
            : "f"(k_regs[14]), "f"(vn2s0_0_x), "f"(S[14]));
        float _fma_63;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_63)
            : "f"(k_regs[15]), "f"(vn2s0_0_y), "f"(S[15]));
        S[14] = _fma_62;
        S[15] = _fma_63;
        float accs0_1_0_x = 0.0f;
        float accs0_1_0_y = 0.0f;
        float accs0_1_1_x = 0.0f;
        float accs0_1_1_y = 0.0f;
        float _fma_64;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_64)
            : "f"(S[16]), "f"(k_regs[0]), "f"(accs0_1_0_x));
        float _fma_65;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_65)
            : "f"(S[17]), "f"(k_regs[1]), "f"(accs0_1_0_y));
        accs0_1_0_x = _fma_64;
        accs0_1_0_y = _fma_65;
        float _fma_66;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_66)
            : "f"(S[18]), "f"(k_regs[2]), "f"(accs0_1_1_x));
        float _fma_67;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_67)
            : "f"(S[19]), "f"(k_regs[3]), "f"(accs0_1_1_y));
        accs0_1_1_x = _fma_66;
        accs0_1_1_y = _fma_67;
        float _fma_68;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_68)
            : "f"(S[20]), "f"(k_regs[4]), "f"(accs0_1_0_x));
        float _fma_69;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_69)
            : "f"(S[21]), "f"(k_regs[5]), "f"(accs0_1_0_y));
        accs0_1_0_x = _fma_68;
        accs0_1_0_y = _fma_69;
        float _fma_70;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_70)
            : "f"(S[22]), "f"(k_regs[6]), "f"(accs0_1_1_x));
        float _fma_71;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_71)
            : "f"(S[23]), "f"(k_regs[7]), "f"(accs0_1_1_y));
        accs0_1_1_x = _fma_70;
        accs0_1_1_y = _fma_71;
        float _fma_72;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_72)
            : "f"(S[24]), "f"(k_regs[8]), "f"(accs0_1_0_x));
        float _fma_73;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_73)
            : "f"(S[25]), "f"(k_regs[9]), "f"(accs0_1_0_y));
        accs0_1_0_x = _fma_72;
        accs0_1_0_y = _fma_73;
        float _fma_74;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_74)
            : "f"(S[26]), "f"(k_regs[10]), "f"(accs0_1_1_x));
        float _fma_75;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_75)
            : "f"(S[27]), "f"(k_regs[11]), "f"(accs0_1_1_y));
        accs0_1_1_x = _fma_74;
        accs0_1_1_y = _fma_75;
        float _fma_76;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_76)
            : "f"(S[28]), "f"(k_regs[12]), "f"(accs0_1_0_x));
        float _fma_77;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_77)
            : "f"(S[29]), "f"(k_regs[13]), "f"(accs0_1_0_y));
        accs0_1_0_x = _fma_76;
        accs0_1_0_y = _fma_77;
        float _fma_78;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_78)
            : "f"(S[30]), "f"(k_regs[14]), "f"(accs0_1_1_x));
        float _fma_79;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_79)
            : "f"(S[31]), "f"(k_regs[15]), "f"(accs0_1_1_y));
        accs0_1_1_x = _fma_78;
        accs0_1_1_y = _fma_79;
        float _add_57;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(accs0_1_0_x), "f"(accs0_1_1_x));
        float _add_58;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(accs0_1_0_y), "f"(accs0_1_1_y));
        float dots0_1 = _add_57 + _add_58;
        float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 1);
        dots0_1 = dots0_1 + _shfl_xor_sync_59;
        float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 2);
        dots0_1 = dots0_1 + _shfl_xor_sync_60;
        float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 4);
        dots0_1 = dots0_1 + _shfl_xor_sync_61;
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
        float2 accs0_1_0 = make_float2(0.0f, 0.0f);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[9], k2[1], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[11], k2[3], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[13], k2[5], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accs0_1_0);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[15], k2[7], accs0_1_0);
        float dots0_1 = accs0_1_0.x + accs0_1_0.y;
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 1);
        dots0_1 = dots0_1 + _shfl_xor_sync_55;
        float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 2);
        dots0_1 = dots0_1 + _shfl_xor_sync_56;
        float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 4);
        dots0_1 = dots0_1 + _shfl_xor_sync_57;
#endif
        float vns0_1 = (v_regs[1] - dots0_1) * betas0;
#if __CUDA_ARCH__ == 900
        float vn2s0_1_x = vns0_1;
        float vn2s0_1_y = vns0_1;
        float _fma_80;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_80)
            : "f"(k_regs[0]), "f"(vn2s0_1_x), "f"(S[16]));
        float _fma_81;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_81)
            : "f"(k_regs[1]), "f"(vn2s0_1_y), "f"(S[17]));
        S[16] = _fma_80;
        S[17] = _fma_81;
        float _fma_82;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_82)
            : "f"(k_regs[2]), "f"(vn2s0_1_x), "f"(S[18]));
        float _fma_83;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_83)
            : "f"(k_regs[3]), "f"(vn2s0_1_y), "f"(S[19]));
        S[18] = _fma_82;
        S[19] = _fma_83;
        float _fma_84;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_84)
            : "f"(k_regs[4]), "f"(vn2s0_1_x), "f"(S[20]));
        float _fma_85;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_85)
            : "f"(k_regs[5]), "f"(vn2s0_1_y), "f"(S[21]));
        S[20] = _fma_84;
        S[21] = _fma_85;
        float _fma_86;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_86)
            : "f"(k_regs[6]), "f"(vn2s0_1_x), "f"(S[22]));
        float _fma_87;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_87)
            : "f"(k_regs[7]), "f"(vn2s0_1_y), "f"(S[23]));
        S[22] = _fma_86;
        S[23] = _fma_87;
        float _fma_88;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_88)
            : "f"(k_regs[8]), "f"(vn2s0_1_x), "f"(S[24]));
        float _fma_89;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_89)
            : "f"(k_regs[9]), "f"(vn2s0_1_y), "f"(S[25]));
        S[24] = _fma_88;
        S[25] = _fma_89;
        float _fma_90;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_90)
            : "f"(k_regs[10]), "f"(vn2s0_1_x), "f"(S[26]));
        float _fma_91;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_91)
            : "f"(k_regs[11]), "f"(vn2s0_1_y), "f"(S[27]));
        S[26] = _fma_90;
        S[27] = _fma_91;
        float _fma_92;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_92)
            : "f"(k_regs[12]), "f"(vn2s0_1_x), "f"(S[28]));
        float _fma_93;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_93)
            : "f"(k_regs[13]), "f"(vn2s0_1_y), "f"(S[29]));
        S[28] = _fma_92;
        S[29] = _fma_93;
        float _fma_94;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_94)
            : "f"(k_regs[14]), "f"(vn2s0_1_x), "f"(S[30]));
        float _fma_95;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_95)
            : "f"(k_regs[15]), "f"(vn2s0_1_y), "f"(S[31]));
        S[30] = _fma_94;
        S[31] = _fma_95;
        float accs0_2_0_x = 0.0f;
        float accs0_2_0_y = 0.0f;
        float accs0_2_1_x = 0.0f;
        float accs0_2_1_y = 0.0f;
        float _fma_96;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_96)
            : "f"(S[32]), "f"(k_regs[0]), "f"(accs0_2_0_x));
        float _fma_97;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_97)
            : "f"(S[33]), "f"(k_regs[1]), "f"(accs0_2_0_y));
        accs0_2_0_x = _fma_96;
        accs0_2_0_y = _fma_97;
        float _fma_98;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_98)
            : "f"(S[34]), "f"(k_regs[2]), "f"(accs0_2_1_x));
        float _fma_99;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_99)
            : "f"(S[35]), "f"(k_regs[3]), "f"(accs0_2_1_y));
        accs0_2_1_x = _fma_98;
        accs0_2_1_y = _fma_99;
        float _fma_100;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_100)
            : "f"(S[36]), "f"(k_regs[4]), "f"(accs0_2_0_x));
        float _fma_101;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_101)
            : "f"(S[37]), "f"(k_regs[5]), "f"(accs0_2_0_y));
        accs0_2_0_x = _fma_100;
        accs0_2_0_y = _fma_101;
        float _fma_102;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_102)
            : "f"(S[38]), "f"(k_regs[6]), "f"(accs0_2_1_x));
        float _fma_103;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_103)
            : "f"(S[39]), "f"(k_regs[7]), "f"(accs0_2_1_y));
        accs0_2_1_x = _fma_102;
        accs0_2_1_y = _fma_103;
        float _fma_104;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_104)
            : "f"(S[40]), "f"(k_regs[8]), "f"(accs0_2_0_x));
        float _fma_105;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_105)
            : "f"(S[41]), "f"(k_regs[9]), "f"(accs0_2_0_y));
        accs0_2_0_x = _fma_104;
        accs0_2_0_y = _fma_105;
        float _fma_106;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_106)
            : "f"(S[42]), "f"(k_regs[10]), "f"(accs0_2_1_x));
        float _fma_107;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_107)
            : "f"(S[43]), "f"(k_regs[11]), "f"(accs0_2_1_y));
        accs0_2_1_x = _fma_106;
        accs0_2_1_y = _fma_107;
        float _fma_108;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_108)
            : "f"(S[44]), "f"(k_regs[12]), "f"(accs0_2_0_x));
        float _fma_109;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_109)
            : "f"(S[45]), "f"(k_regs[13]), "f"(accs0_2_0_y));
        accs0_2_0_x = _fma_108;
        accs0_2_0_y = _fma_109;
        float _fma_110;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_110)
            : "f"(S[46]), "f"(k_regs[14]), "f"(accs0_2_1_x));
        float _fma_111;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_111)
            : "f"(S[47]), "f"(k_regs[15]), "f"(accs0_2_1_y));
        accs0_2_1_x = _fma_110;
        accs0_2_1_y = _fma_111;
        float _add_62;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(accs0_2_0_x), "f"(accs0_2_1_x));
        float _add_63;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_63) : "f"(accs0_2_0_y), "f"(accs0_2_1_y));
        float dots0_2 = _add_62 + _add_63;
        float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, dots0_2, 1);
        dots0_2 = dots0_2 + _shfl_xor_sync_64;
        float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, dots0_2, 2);
        dots0_2 = dots0_2 + _shfl_xor_sync_65;
        float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, dots0_2, 4);
        dots0_2 = dots0_2 + _shfl_xor_sync_66;
#else
        float2 vn2s0_1 = make_float2(vns0_1, vns0_1);
        S2[8] = fma_f32x2_rn_ftz(k2[0], vn2s0_1, S2[8]);
        S2[9] = fma_f32x2_rn_ftz(k2[1], vn2s0_1, S2[9]);
        S2[10] = fma_f32x2_rn_ftz(k2[2], vn2s0_1, S2[10]);
        S2[11] = fma_f32x2_rn_ftz(k2[3], vn2s0_1, S2[11]);
        S2[12] = fma_f32x2_rn_ftz(k2[4], vn2s0_1, S2[12]);
        S2[13] = fma_f32x2_rn_ftz(k2[5], vn2s0_1, S2[13]);
        S2[14] = fma_f32x2_rn_ftz(k2[6], vn2s0_1, S2[14]);
        S2[15] = fma_f32x2_rn_ftz(k2[7], vn2s0_1, S2[15]);
        float2 accs0_2_0 = make_float2(0.0f, 0.0f);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accs0_2_0);
        accs0_2_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accs0_2_0);
        float dots0_2 = accs0_2_0.x + accs0_2_0.y;
        float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, dots0_2, 1);
        dots0_2 = dots0_2 + _shfl_xor_sync_58;
        float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, dots0_2, 2);
        dots0_2 = dots0_2 + _shfl_xor_sync_59;
        float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, dots0_2, 4);
        dots0_2 = dots0_2 + _shfl_xor_sync_60;
#endif
        float vns0_2 = (v_regs[2] - dots0_2) * betas0;
#if __CUDA_ARCH__ == 900
        float vn2s0_2_x = vns0_2;
        float vn2s0_2_y = vns0_2;
        float _fma_112;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_112)
            : "f"(k_regs[0]), "f"(vn2s0_2_x), "f"(S[32]));
        float _fma_113;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_113)
            : "f"(k_regs[1]), "f"(vn2s0_2_y), "f"(S[33]));
        S[32] = _fma_112;
        S[33] = _fma_113;
        float _fma_114;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_114)
            : "f"(k_regs[2]), "f"(vn2s0_2_x), "f"(S[34]));
        float _fma_115;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_115)
            : "f"(k_regs[3]), "f"(vn2s0_2_y), "f"(S[35]));
        S[34] = _fma_114;
        S[35] = _fma_115;
        float _fma_116;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_116)
            : "f"(k_regs[4]), "f"(vn2s0_2_x), "f"(S[36]));
        float _fma_117;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_117)
            : "f"(k_regs[5]), "f"(vn2s0_2_y), "f"(S[37]));
        S[36] = _fma_116;
        S[37] = _fma_117;
        float _fma_118;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_118)
            : "f"(k_regs[6]), "f"(vn2s0_2_x), "f"(S[38]));
        float _fma_119;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_119)
            : "f"(k_regs[7]), "f"(vn2s0_2_y), "f"(S[39]));
        S[38] = _fma_118;
        S[39] = _fma_119;
        float _fma_120;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_120)
            : "f"(k_regs[8]), "f"(vn2s0_2_x), "f"(S[40]));
        float _fma_121;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_121)
            : "f"(k_regs[9]), "f"(vn2s0_2_y), "f"(S[41]));
        S[40] = _fma_120;
        S[41] = _fma_121;
        float _fma_122;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_122)
            : "f"(k_regs[10]), "f"(vn2s0_2_x), "f"(S[42]));
        float _fma_123;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_123)
            : "f"(k_regs[11]), "f"(vn2s0_2_y), "f"(S[43]));
        S[42] = _fma_122;
        S[43] = _fma_123;
        float _fma_124;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_124)
            : "f"(k_regs[12]), "f"(vn2s0_2_x), "f"(S[44]));
        float _fma_125;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_125)
            : "f"(k_regs[13]), "f"(vn2s0_2_y), "f"(S[45]));
        S[44] = _fma_124;
        S[45] = _fma_125;
        float _fma_126;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_126)
            : "f"(k_regs[14]), "f"(vn2s0_2_x), "f"(S[46]));
        float _fma_127;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_127)
            : "f"(k_regs[15]), "f"(vn2s0_2_y), "f"(S[47]));
        S[46] = _fma_126;
        S[47] = _fma_127;
        float accs0_3_0_x = 0.0f;
        float accs0_3_0_y = 0.0f;
        float accs0_3_1_x = 0.0f;
        float accs0_3_1_y = 0.0f;
        float _fma_128;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_128)
            : "f"(S[48]), "f"(k_regs[0]), "f"(accs0_3_0_x));
        float _fma_129;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_129)
            : "f"(S[49]), "f"(k_regs[1]), "f"(accs0_3_0_y));
        accs0_3_0_x = _fma_128;
        accs0_3_0_y = _fma_129;
        float _fma_130;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_130)
            : "f"(S[50]), "f"(k_regs[2]), "f"(accs0_3_1_x));
        float _fma_131;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_131)
            : "f"(S[51]), "f"(k_regs[3]), "f"(accs0_3_1_y));
        accs0_3_1_x = _fma_130;
        accs0_3_1_y = _fma_131;
        float _fma_132;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_132)
            : "f"(S[52]), "f"(k_regs[4]), "f"(accs0_3_0_x));
        float _fma_133;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_133)
            : "f"(S[53]), "f"(k_regs[5]), "f"(accs0_3_0_y));
        accs0_3_0_x = _fma_132;
        accs0_3_0_y = _fma_133;
        float _fma_134;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_134)
            : "f"(S[54]), "f"(k_regs[6]), "f"(accs0_3_1_x));
        float _fma_135;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_135)
            : "f"(S[55]), "f"(k_regs[7]), "f"(accs0_3_1_y));
        accs0_3_1_x = _fma_134;
        accs0_3_1_y = _fma_135;
        float _fma_136;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_136)
            : "f"(S[56]), "f"(k_regs[8]), "f"(accs0_3_0_x));
        float _fma_137;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_137)
            : "f"(S[57]), "f"(k_regs[9]), "f"(accs0_3_0_y));
        accs0_3_0_x = _fma_136;
        accs0_3_0_y = _fma_137;
        float _fma_138;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_138)
            : "f"(S[58]), "f"(k_regs[10]), "f"(accs0_3_1_x));
        float _fma_139;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_139)
            : "f"(S[59]), "f"(k_regs[11]), "f"(accs0_3_1_y));
        accs0_3_1_x = _fma_138;
        accs0_3_1_y = _fma_139;
        float _fma_140;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_140)
            : "f"(S[60]), "f"(k_regs[12]), "f"(accs0_3_0_x));
        float _fma_141;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_141)
            : "f"(S[61]), "f"(k_regs[13]), "f"(accs0_3_0_y));
        accs0_3_0_x = _fma_140;
        accs0_3_0_y = _fma_141;
        float _fma_142;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_142)
            : "f"(S[62]), "f"(k_regs[14]), "f"(accs0_3_1_x));
        float _fma_143;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_143)
            : "f"(S[63]), "f"(k_regs[15]), "f"(accs0_3_1_y));
        accs0_3_1_x = _fma_142;
        accs0_3_1_y = _fma_143;
        float _add_67;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(accs0_3_0_x), "f"(accs0_3_1_x));
        float _add_68;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(accs0_3_0_y), "f"(accs0_3_1_y));
        float dots0_3 = _add_67 + _add_68;
        float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, dots0_3, 1);
        dots0_3 = dots0_3 + _shfl_xor_sync_69;
        float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, dots0_3, 2);
        dots0_3 = dots0_3 + _shfl_xor_sync_70;
        float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, dots0_3, 4);
        dots0_3 = dots0_3 + _shfl_xor_sync_71;
#else
        float2 vn2s0_2 = make_float2(vns0_2, vns0_2);
        S2[16] = fma_f32x2_rn_ftz(k2[0], vn2s0_2, S2[16]);
        S2[17] = fma_f32x2_rn_ftz(k2[1], vn2s0_2, S2[17]);
        S2[18] = fma_f32x2_rn_ftz(k2[2], vn2s0_2, S2[18]);
        S2[19] = fma_f32x2_rn_ftz(k2[3], vn2s0_2, S2[19]);
        S2[20] = fma_f32x2_rn_ftz(k2[4], vn2s0_2, S2[20]);
        S2[21] = fma_f32x2_rn_ftz(k2[5], vn2s0_2, S2[21]);
        S2[22] = fma_f32x2_rn_ftz(k2[6], vn2s0_2, S2[22]);
        S2[23] = fma_f32x2_rn_ftz(k2[7], vn2s0_2, S2[23]);
        float2 accs0_3_0 = make_float2(0.0f, 0.0f);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[24], k2[0], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[25], k2[1], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[26], k2[2], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[27], k2[3], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[28], k2[4], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[29], k2[5], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[30], k2[6], accs0_3_0);
        accs0_3_0 = fma_f32x2_rn_ftz(S2[31], k2[7], accs0_3_0);
        float dots0_3 = accs0_3_0.x + accs0_3_0.y;
        float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, dots0_3, 1);
        dots0_3 = dots0_3 + _shfl_xor_sync_61;
        float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, dots0_3, 2);
        dots0_3 = dots0_3 + _shfl_xor_sync_62;
        float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, dots0_3, 4);
        dots0_3 = dots0_3 + _shfl_xor_sync_63;
#endif
        float vns0_3 = (v_regs[3] - dots0_3) * betas0;
#if __CUDA_ARCH__ == 900
        float vn2s0_3_x = vns0_3;
        float vn2s0_3_y = vns0_3;
        float _fma_144;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_144)
            : "f"(k_regs[0]), "f"(vn2s0_3_x), "f"(S[48]));
        float _fma_145;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_145)
            : "f"(k_regs[1]), "f"(vn2s0_3_y), "f"(S[49]));
        S[48] = _fma_144;
        S[49] = _fma_145;
        float _fma_146;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_146)
            : "f"(k_regs[2]), "f"(vn2s0_3_x), "f"(S[50]));
        float _fma_147;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_147)
            : "f"(k_regs[3]), "f"(vn2s0_3_y), "f"(S[51]));
        S[50] = _fma_146;
        S[51] = _fma_147;
        float _fma_148;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_148)
            : "f"(k_regs[4]), "f"(vn2s0_3_x), "f"(S[52]));
        float _fma_149;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_149)
            : "f"(k_regs[5]), "f"(vn2s0_3_y), "f"(S[53]));
        S[52] = _fma_148;
        S[53] = _fma_149;
        float _fma_150;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_150)
            : "f"(k_regs[6]), "f"(vn2s0_3_x), "f"(S[54]));
        float _fma_151;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_151)
            : "f"(k_regs[7]), "f"(vn2s0_3_y), "f"(S[55]));
        S[54] = _fma_150;
        S[55] = _fma_151;
        float _fma_152;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_152)
            : "f"(k_regs[8]), "f"(vn2s0_3_x), "f"(S[56]));
        float _fma_153;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_153)
            : "f"(k_regs[9]), "f"(vn2s0_3_y), "f"(S[57]));
        S[56] = _fma_152;
        S[57] = _fma_153;
        float _fma_154;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_154)
            : "f"(k_regs[10]), "f"(vn2s0_3_x), "f"(S[58]));
        float _fma_155;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_155)
            : "f"(k_regs[11]), "f"(vn2s0_3_y), "f"(S[59]));
        S[58] = _fma_154;
        S[59] = _fma_155;
        float _fma_156;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_156)
            : "f"(k_regs[12]), "f"(vn2s0_3_x), "f"(S[60]));
        float _fma_157;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_157)
            : "f"(k_regs[13]), "f"(vn2s0_3_y), "f"(S[61]));
        S[60] = _fma_156;
        S[61] = _fma_157;
        float _fma_158;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_158)
            : "f"(k_regs[14]), "f"(vn2s0_3_x), "f"(S[62]));
        float _fma_159;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_159)
            : "f"(k_regs[15]), "f"(vn2s0_3_y), "f"(S[63]));
        S[62] = _fma_158;
        S[63] = _fma_159;
#else
        float2 vn2s0_3 = make_float2(vns0_3, vns0_3);
        S2[24] = fma_f32x2_rn_ftz(k2[0], vn2s0_3, S2[24]);
        S2[25] = fma_f32x2_rn_ftz(k2[1], vn2s0_3, S2[25]);
        S2[26] = fma_f32x2_rn_ftz(k2[2], vn2s0_3, S2[26]);
        S2[27] = fma_f32x2_rn_ftz(k2[3], vn2s0_3, S2[27]);
        S2[28] = fma_f32x2_rn_ftz(k2[4], vn2s0_3, S2[28]);
        S2[29] = fma_f32x2_rn_ftz(k2[5], vn2s0_3, S2[29]);
        S2[30] = fma_f32x2_rn_ftz(k2[6], vn2s0_3, S2[30]);
        S2[31] = fma_f32x2_rn_ftz(k2[7], vn2s0_3, S2[31]);
#endif
        int ms1 = t * 3 + 1;
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
                     : "r"(kbases1 + 128));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                     : "r"(kbases1 + 256));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                     : "r"(kbases1 + 384));
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
        float _fma_160;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_160)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accs1_0_0_x));
        float _fma_161;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_161)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accs1_0_0_y));
        accs1_0_0_x = _fma_160;
        accs1_0_0_y = _fma_161;
        float _fma_162;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_162)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accs1_0_1_x));
        float _fma_163;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_163)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accs1_0_1_y));
        accs1_0_1_x = _fma_162;
        accs1_0_1_y = _fma_163;
        float _fma_164;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_164)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accs1_0_0_x));
        float _fma_165;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_165)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accs1_0_0_y));
        accs1_0_0_x = _fma_164;
        accs1_0_0_y = _fma_165;
        float _fma_166;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_166)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accs1_0_1_x));
        float _fma_167;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_167)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accs1_0_1_y));
        accs1_0_1_x = _fma_166;
        accs1_0_1_y = _fma_167;
        float _fma_168;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_168)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accs1_0_0_x));
        float _fma_169;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_169)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accs1_0_0_y));
        accs1_0_0_x = _fma_168;
        accs1_0_0_y = _fma_169;
        float _fma_170;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_170)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accs1_0_1_x));
        float _fma_171;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_171)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accs1_0_1_y));
        accs1_0_1_x = _fma_170;
        accs1_0_1_y = _fma_171;
        float _fma_172;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_172)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accs1_0_0_x));
        float _fma_173;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_173)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accs1_0_0_y));
        accs1_0_0_x = _fma_172;
        accs1_0_0_y = _fma_173;
        float _fma_174;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_174)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accs1_0_1_x));
        float _fma_175;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_175)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accs1_0_1_y));
        accs1_0_1_x = _fma_174;
        accs1_0_1_y = _fma_175;
        float _add_72;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(accs1_0_0_x), "f"(accs1_0_1_x));
        float _add_73;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(accs1_0_0_y), "f"(accs1_0_1_y));
        float dots1_0 = _add_72 + _add_73;
        float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 1);
        dots1_0 = dots1_0 + _shfl_xor_sync_74;
        float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 2);
        dots1_0 = dots1_0 + _shfl_xor_sync_75;
        float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 4);
        dots1_0 = dots1_0 + _shfl_xor_sync_76;
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
        float dots1_0 = accs1_0_0.x + accs1_0_0.y;
        float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 1);
        dots1_0 = dots1_0 + _shfl_xor_sync_64;
        float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 2);
        dots1_0 = dots1_0 + _shfl_xor_sync_65;
        float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 4);
        dots1_0 = dots1_0 + _shfl_xor_sync_66;
#endif
        float vns1_0 = (v_regs[0] - dots1_0) * betas1;
#if __CUDA_ARCH__ == 900
        float vn2s1_0_x = vns1_0;
        float vn2s1_0_y = vns1_0;
        float _fma_176;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_176)
            : "f"(k_regs[0]), "f"(vn2s1_0_x), "f"(S[0]));
        float _fma_177;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_177)
            : "f"(k_regs[1]), "f"(vn2s1_0_y), "f"(S[1]));
        S[0] = _fma_176;
        S[1] = _fma_177;
        float _fma_178;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_178)
            : "f"(k_regs[2]), "f"(vn2s1_0_x), "f"(S[2]));
        float _fma_179;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_179)
            : "f"(k_regs[3]), "f"(vn2s1_0_y), "f"(S[3]));
        S[2] = _fma_178;
        S[3] = _fma_179;
        float _fma_180;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_180)
            : "f"(k_regs[4]), "f"(vn2s1_0_x), "f"(S[4]));
        float _fma_181;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_181)
            : "f"(k_regs[5]), "f"(vn2s1_0_y), "f"(S[5]));
        S[4] = _fma_180;
        S[5] = _fma_181;
        float _fma_182;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_182)
            : "f"(k_regs[6]), "f"(vn2s1_0_x), "f"(S[6]));
        float _fma_183;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_183)
            : "f"(k_regs[7]), "f"(vn2s1_0_y), "f"(S[7]));
        S[6] = _fma_182;
        S[7] = _fma_183;
        float _fma_184;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_184)
            : "f"(k_regs[8]), "f"(vn2s1_0_x), "f"(S[8]));
        float _fma_185;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_185)
            : "f"(k_regs[9]), "f"(vn2s1_0_y), "f"(S[9]));
        S[8] = _fma_184;
        S[9] = _fma_185;
        float _fma_186;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_186)
            : "f"(k_regs[10]), "f"(vn2s1_0_x), "f"(S[10]));
        float _fma_187;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_187)
            : "f"(k_regs[11]), "f"(vn2s1_0_y), "f"(S[11]));
        S[10] = _fma_186;
        S[11] = _fma_187;
        float _fma_188;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_188)
            : "f"(k_regs[12]), "f"(vn2s1_0_x), "f"(S[12]));
        float _fma_189;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_189)
            : "f"(k_regs[13]), "f"(vn2s1_0_y), "f"(S[13]));
        S[12] = _fma_188;
        S[13] = _fma_189;
        float _fma_190;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_190)
            : "f"(k_regs[14]), "f"(vn2s1_0_x), "f"(S[14]));
        float _fma_191;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_191)
            : "f"(k_regs[15]), "f"(vn2s1_0_y), "f"(S[15]));
        S[14] = _fma_190;
        S[15] = _fma_191;
        float accs1_1_0_x = 0.0f;
        float accs1_1_0_y = 0.0f;
        float accs1_1_1_x = 0.0f;
        float accs1_1_1_y = 0.0f;
        float _fma_192;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_192)
            : "f"(S[16]), "f"(k_regs[0]), "f"(accs1_1_0_x));
        float _fma_193;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_193)
            : "f"(S[17]), "f"(k_regs[1]), "f"(accs1_1_0_y));
        accs1_1_0_x = _fma_192;
        accs1_1_0_y = _fma_193;
        float _fma_194;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_194)
            : "f"(S[18]), "f"(k_regs[2]), "f"(accs1_1_1_x));
        float _fma_195;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_195)
            : "f"(S[19]), "f"(k_regs[3]), "f"(accs1_1_1_y));
        accs1_1_1_x = _fma_194;
        accs1_1_1_y = _fma_195;
        float _fma_196;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_196)
            : "f"(S[20]), "f"(k_regs[4]), "f"(accs1_1_0_x));
        float _fma_197;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_197)
            : "f"(S[21]), "f"(k_regs[5]), "f"(accs1_1_0_y));
        accs1_1_0_x = _fma_196;
        accs1_1_0_y = _fma_197;
        float _fma_198;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_198)
            : "f"(S[22]), "f"(k_regs[6]), "f"(accs1_1_1_x));
        float _fma_199;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_199)
            : "f"(S[23]), "f"(k_regs[7]), "f"(accs1_1_1_y));
        accs1_1_1_x = _fma_198;
        accs1_1_1_y = _fma_199;
        float _fma_200;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_200)
            : "f"(S[24]), "f"(k_regs[8]), "f"(accs1_1_0_x));
        float _fma_201;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_201)
            : "f"(S[25]), "f"(k_regs[9]), "f"(accs1_1_0_y));
        accs1_1_0_x = _fma_200;
        accs1_1_0_y = _fma_201;
        float _fma_202;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_202)
            : "f"(S[26]), "f"(k_regs[10]), "f"(accs1_1_1_x));
        float _fma_203;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_203)
            : "f"(S[27]), "f"(k_regs[11]), "f"(accs1_1_1_y));
        accs1_1_1_x = _fma_202;
        accs1_1_1_y = _fma_203;
        float _fma_204;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_204)
            : "f"(S[28]), "f"(k_regs[12]), "f"(accs1_1_0_x));
        float _fma_205;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_205)
            : "f"(S[29]), "f"(k_regs[13]), "f"(accs1_1_0_y));
        accs1_1_0_x = _fma_204;
        accs1_1_0_y = _fma_205;
        float _fma_206;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_206)
            : "f"(S[30]), "f"(k_regs[14]), "f"(accs1_1_1_x));
        float _fma_207;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_207)
            : "f"(S[31]), "f"(k_regs[15]), "f"(accs1_1_1_y));
        accs1_1_1_x = _fma_206;
        accs1_1_1_y = _fma_207;
        float _add_77;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(accs1_1_0_x), "f"(accs1_1_1_x));
        float _add_78;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(accs1_1_0_y), "f"(accs1_1_1_y));
        float dots1_1 = _add_77 + _add_78;
        float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 1);
        dots1_1 = dots1_1 + _shfl_xor_sync_79;
        float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 2);
        dots1_1 = dots1_1 + _shfl_xor_sync_80;
        float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 4);
        dots1_1 = dots1_1 + _shfl_xor_sync_81;
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
        float2 accs1_1_0 = make_float2(0.0f, 0.0f);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[9], k2[1], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[11], k2[3], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[13], k2[5], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accs1_1_0);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[15], k2[7], accs1_1_0);
        float dots1_1 = accs1_1_0.x + accs1_1_0.y;
        float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 1);
        dots1_1 = dots1_1 + _shfl_xor_sync_67;
        float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 2);
        dots1_1 = dots1_1 + _shfl_xor_sync_68;
        float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 4);
        dots1_1 = dots1_1 + _shfl_xor_sync_69;
#endif
        float vns1_1 = (v_regs[1] - dots1_1) * betas1;
#if __CUDA_ARCH__ == 900
        float vn2s1_1_x = vns1_1;
        float vn2s1_1_y = vns1_1;
        float _fma_208;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_208)
            : "f"(k_regs[0]), "f"(vn2s1_1_x), "f"(S[16]));
        float _fma_209;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_209)
            : "f"(k_regs[1]), "f"(vn2s1_1_y), "f"(S[17]));
        S[16] = _fma_208;
        S[17] = _fma_209;
        float _fma_210;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_210)
            : "f"(k_regs[2]), "f"(vn2s1_1_x), "f"(S[18]));
        float _fma_211;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_211)
            : "f"(k_regs[3]), "f"(vn2s1_1_y), "f"(S[19]));
        S[18] = _fma_210;
        S[19] = _fma_211;
        float _fma_212;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_212)
            : "f"(k_regs[4]), "f"(vn2s1_1_x), "f"(S[20]));
        float _fma_213;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_213)
            : "f"(k_regs[5]), "f"(vn2s1_1_y), "f"(S[21]));
        S[20] = _fma_212;
        S[21] = _fma_213;
        float _fma_214;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_214)
            : "f"(k_regs[6]), "f"(vn2s1_1_x), "f"(S[22]));
        float _fma_215;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_215)
            : "f"(k_regs[7]), "f"(vn2s1_1_y), "f"(S[23]));
        S[22] = _fma_214;
        S[23] = _fma_215;
        float _fma_216;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_216)
            : "f"(k_regs[8]), "f"(vn2s1_1_x), "f"(S[24]));
        float _fma_217;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_217)
            : "f"(k_regs[9]), "f"(vn2s1_1_y), "f"(S[25]));
        S[24] = _fma_216;
        S[25] = _fma_217;
        float _fma_218;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_218)
            : "f"(k_regs[10]), "f"(vn2s1_1_x), "f"(S[26]));
        float _fma_219;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_219)
            : "f"(k_regs[11]), "f"(vn2s1_1_y), "f"(S[27]));
        S[26] = _fma_218;
        S[27] = _fma_219;
        float _fma_220;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_220)
            : "f"(k_regs[12]), "f"(vn2s1_1_x), "f"(S[28]));
        float _fma_221;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_221)
            : "f"(k_regs[13]), "f"(vn2s1_1_y), "f"(S[29]));
        S[28] = _fma_220;
        S[29] = _fma_221;
        float _fma_222;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_222)
            : "f"(k_regs[14]), "f"(vn2s1_1_x), "f"(S[30]));
        float _fma_223;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_223)
            : "f"(k_regs[15]), "f"(vn2s1_1_y), "f"(S[31]));
        S[30] = _fma_222;
        S[31] = _fma_223;
        float accs1_2_0_x = 0.0f;
        float accs1_2_0_y = 0.0f;
        float accs1_2_1_x = 0.0f;
        float accs1_2_1_y = 0.0f;
        float _fma_224;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_224)
            : "f"(S[32]), "f"(k_regs[0]), "f"(accs1_2_0_x));
        float _fma_225;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_225)
            : "f"(S[33]), "f"(k_regs[1]), "f"(accs1_2_0_y));
        accs1_2_0_x = _fma_224;
        accs1_2_0_y = _fma_225;
        float _fma_226;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_226)
            : "f"(S[34]), "f"(k_regs[2]), "f"(accs1_2_1_x));
        float _fma_227;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_227)
            : "f"(S[35]), "f"(k_regs[3]), "f"(accs1_2_1_y));
        accs1_2_1_x = _fma_226;
        accs1_2_1_y = _fma_227;
        float _fma_228;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_228)
            : "f"(S[36]), "f"(k_regs[4]), "f"(accs1_2_0_x));
        float _fma_229;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_229)
            : "f"(S[37]), "f"(k_regs[5]), "f"(accs1_2_0_y));
        accs1_2_0_x = _fma_228;
        accs1_2_0_y = _fma_229;
        float _fma_230;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_230)
            : "f"(S[38]), "f"(k_regs[6]), "f"(accs1_2_1_x));
        float _fma_231;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_231)
            : "f"(S[39]), "f"(k_regs[7]), "f"(accs1_2_1_y));
        accs1_2_1_x = _fma_230;
        accs1_2_1_y = _fma_231;
        float _fma_232;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_232)
            : "f"(S[40]), "f"(k_regs[8]), "f"(accs1_2_0_x));
        float _fma_233;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_233)
            : "f"(S[41]), "f"(k_regs[9]), "f"(accs1_2_0_y));
        accs1_2_0_x = _fma_232;
        accs1_2_0_y = _fma_233;
        float _fma_234;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_234)
            : "f"(S[42]), "f"(k_regs[10]), "f"(accs1_2_1_x));
        float _fma_235;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_235)
            : "f"(S[43]), "f"(k_regs[11]), "f"(accs1_2_1_y));
        accs1_2_1_x = _fma_234;
        accs1_2_1_y = _fma_235;
        float _fma_236;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_236)
            : "f"(S[44]), "f"(k_regs[12]), "f"(accs1_2_0_x));
        float _fma_237;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_237)
            : "f"(S[45]), "f"(k_regs[13]), "f"(accs1_2_0_y));
        accs1_2_0_x = _fma_236;
        accs1_2_0_y = _fma_237;
        float _fma_238;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_238)
            : "f"(S[46]), "f"(k_regs[14]), "f"(accs1_2_1_x));
        float _fma_239;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_239)
            : "f"(S[47]), "f"(k_regs[15]), "f"(accs1_2_1_y));
        accs1_2_1_x = _fma_238;
        accs1_2_1_y = _fma_239;
        float _add_82;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(accs1_2_0_x), "f"(accs1_2_1_x));
        float _add_83;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(accs1_2_0_y), "f"(accs1_2_1_y));
        float dots1_2 = _add_82 + _add_83;
        float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, dots1_2, 1);
        dots1_2 = dots1_2 + _shfl_xor_sync_84;
        float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, dots1_2, 2);
        dots1_2 = dots1_2 + _shfl_xor_sync_85;
        float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, dots1_2, 4);
        dots1_2 = dots1_2 + _shfl_xor_sync_86;
#else
        float2 vn2s1_1 = make_float2(vns1_1, vns1_1);
        S2[8] = fma_f32x2_rn_ftz(k2[0], vn2s1_1, S2[8]);
        S2[9] = fma_f32x2_rn_ftz(k2[1], vn2s1_1, S2[9]);
        S2[10] = fma_f32x2_rn_ftz(k2[2], vn2s1_1, S2[10]);
        S2[11] = fma_f32x2_rn_ftz(k2[3], vn2s1_1, S2[11]);
        S2[12] = fma_f32x2_rn_ftz(k2[4], vn2s1_1, S2[12]);
        S2[13] = fma_f32x2_rn_ftz(k2[5], vn2s1_1, S2[13]);
        S2[14] = fma_f32x2_rn_ftz(k2[6], vn2s1_1, S2[14]);
        S2[15] = fma_f32x2_rn_ftz(k2[7], vn2s1_1, S2[15]);
        float2 accs1_2_0 = make_float2(0.0f, 0.0f);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accs1_2_0);
        accs1_2_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accs1_2_0);
        float dots1_2 = accs1_2_0.x + accs1_2_0.y;
        float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, dots1_2, 1);
        dots1_2 = dots1_2 + _shfl_xor_sync_70;
        float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, dots1_2, 2);
        dots1_2 = dots1_2 + _shfl_xor_sync_71;
        float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, dots1_2, 4);
        dots1_2 = dots1_2 + _shfl_xor_sync_72;
#endif
        float vns1_2 = (v_regs[2] - dots1_2) * betas1;
#if __CUDA_ARCH__ == 900
        float vn2s1_2_x = vns1_2;
        float vn2s1_2_y = vns1_2;
        float _fma_240;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_240)
            : "f"(k_regs[0]), "f"(vn2s1_2_x), "f"(S[32]));
        float _fma_241;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_241)
            : "f"(k_regs[1]), "f"(vn2s1_2_y), "f"(S[33]));
        S[32] = _fma_240;
        S[33] = _fma_241;
        float _fma_242;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_242)
            : "f"(k_regs[2]), "f"(vn2s1_2_x), "f"(S[34]));
        float _fma_243;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_243)
            : "f"(k_regs[3]), "f"(vn2s1_2_y), "f"(S[35]));
        S[34] = _fma_242;
        S[35] = _fma_243;
        float _fma_244;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_244)
            : "f"(k_regs[4]), "f"(vn2s1_2_x), "f"(S[36]));
        float _fma_245;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_245)
            : "f"(k_regs[5]), "f"(vn2s1_2_y), "f"(S[37]));
        S[36] = _fma_244;
        S[37] = _fma_245;
        float _fma_246;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_246)
            : "f"(k_regs[6]), "f"(vn2s1_2_x), "f"(S[38]));
        float _fma_247;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_247)
            : "f"(k_regs[7]), "f"(vn2s1_2_y), "f"(S[39]));
        S[38] = _fma_246;
        S[39] = _fma_247;
        float _fma_248;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_248)
            : "f"(k_regs[8]), "f"(vn2s1_2_x), "f"(S[40]));
        float _fma_249;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_249)
            : "f"(k_regs[9]), "f"(vn2s1_2_y), "f"(S[41]));
        S[40] = _fma_248;
        S[41] = _fma_249;
        float _fma_250;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_250)
            : "f"(k_regs[10]), "f"(vn2s1_2_x), "f"(S[42]));
        float _fma_251;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_251)
            : "f"(k_regs[11]), "f"(vn2s1_2_y), "f"(S[43]));
        S[42] = _fma_250;
        S[43] = _fma_251;
        float _fma_252;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_252)
            : "f"(k_regs[12]), "f"(vn2s1_2_x), "f"(S[44]));
        float _fma_253;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_253)
            : "f"(k_regs[13]), "f"(vn2s1_2_y), "f"(S[45]));
        S[44] = _fma_252;
        S[45] = _fma_253;
        float _fma_254;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_254)
            : "f"(k_regs[14]), "f"(vn2s1_2_x), "f"(S[46]));
        float _fma_255;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_255)
            : "f"(k_regs[15]), "f"(vn2s1_2_y), "f"(S[47]));
        S[46] = _fma_254;
        S[47] = _fma_255;
        float accs1_3_0_x = 0.0f;
        float accs1_3_0_y = 0.0f;
        float accs1_3_1_x = 0.0f;
        float accs1_3_1_y = 0.0f;
        float _fma_256;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_256)
            : "f"(S[48]), "f"(k_regs[0]), "f"(accs1_3_0_x));
        float _fma_257;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_257)
            : "f"(S[49]), "f"(k_regs[1]), "f"(accs1_3_0_y));
        accs1_3_0_x = _fma_256;
        accs1_3_0_y = _fma_257;
        float _fma_258;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_258)
            : "f"(S[50]), "f"(k_regs[2]), "f"(accs1_3_1_x));
        float _fma_259;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_259)
            : "f"(S[51]), "f"(k_regs[3]), "f"(accs1_3_1_y));
        accs1_3_1_x = _fma_258;
        accs1_3_1_y = _fma_259;
        float _fma_260;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_260)
            : "f"(S[52]), "f"(k_regs[4]), "f"(accs1_3_0_x));
        float _fma_261;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_261)
            : "f"(S[53]), "f"(k_regs[5]), "f"(accs1_3_0_y));
        accs1_3_0_x = _fma_260;
        accs1_3_0_y = _fma_261;
        float _fma_262;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_262)
            : "f"(S[54]), "f"(k_regs[6]), "f"(accs1_3_1_x));
        float _fma_263;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_263)
            : "f"(S[55]), "f"(k_regs[7]), "f"(accs1_3_1_y));
        accs1_3_1_x = _fma_262;
        accs1_3_1_y = _fma_263;
        float _fma_264;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_264)
            : "f"(S[56]), "f"(k_regs[8]), "f"(accs1_3_0_x));
        float _fma_265;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_265)
            : "f"(S[57]), "f"(k_regs[9]), "f"(accs1_3_0_y));
        accs1_3_0_x = _fma_264;
        accs1_3_0_y = _fma_265;
        float _fma_266;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_266)
            : "f"(S[58]), "f"(k_regs[10]), "f"(accs1_3_1_x));
        float _fma_267;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_267)
            : "f"(S[59]), "f"(k_regs[11]), "f"(accs1_3_1_y));
        accs1_3_1_x = _fma_266;
        accs1_3_1_y = _fma_267;
        float _fma_268;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_268)
            : "f"(S[60]), "f"(k_regs[12]), "f"(accs1_3_0_x));
        float _fma_269;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_269)
            : "f"(S[61]), "f"(k_regs[13]), "f"(accs1_3_0_y));
        accs1_3_0_x = _fma_268;
        accs1_3_0_y = _fma_269;
        float _fma_270;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_270)
            : "f"(S[62]), "f"(k_regs[14]), "f"(accs1_3_1_x));
        float _fma_271;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_271)
            : "f"(S[63]), "f"(k_regs[15]), "f"(accs1_3_1_y));
        accs1_3_1_x = _fma_270;
        accs1_3_1_y = _fma_271;
        float _add_87;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(accs1_3_0_x), "f"(accs1_3_1_x));
        float _add_88;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(accs1_3_0_y), "f"(accs1_3_1_y));
        float dots1_3 = _add_87 + _add_88;
        float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, dots1_3, 1);
        dots1_3 = dots1_3 + _shfl_xor_sync_89;
        float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, dots1_3, 2);
        dots1_3 = dots1_3 + _shfl_xor_sync_90;
        float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, dots1_3, 4);
        dots1_3 = dots1_3 + _shfl_xor_sync_91;
#else
        float2 vn2s1_2 = make_float2(vns1_2, vns1_2);
        S2[16] = fma_f32x2_rn_ftz(k2[0], vn2s1_2, S2[16]);
        S2[17] = fma_f32x2_rn_ftz(k2[1], vn2s1_2, S2[17]);
        S2[18] = fma_f32x2_rn_ftz(k2[2], vn2s1_2, S2[18]);
        S2[19] = fma_f32x2_rn_ftz(k2[3], vn2s1_2, S2[19]);
        S2[20] = fma_f32x2_rn_ftz(k2[4], vn2s1_2, S2[20]);
        S2[21] = fma_f32x2_rn_ftz(k2[5], vn2s1_2, S2[21]);
        S2[22] = fma_f32x2_rn_ftz(k2[6], vn2s1_2, S2[22]);
        S2[23] = fma_f32x2_rn_ftz(k2[7], vn2s1_2, S2[23]);
        float2 accs1_3_0 = make_float2(0.0f, 0.0f);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[24], k2[0], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[25], k2[1], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[26], k2[2], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[27], k2[3], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[28], k2[4], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[29], k2[5], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[30], k2[6], accs1_3_0);
        accs1_3_0 = fma_f32x2_rn_ftz(S2[31], k2[7], accs1_3_0);
        float dots1_3 = accs1_3_0.x + accs1_3_0.y;
        float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, dots1_3, 1);
        dots1_3 = dots1_3 + _shfl_xor_sync_73;
        float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, dots1_3, 2);
        dots1_3 = dots1_3 + _shfl_xor_sync_74;
        float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, dots1_3, 4);
        dots1_3 = dots1_3 + _shfl_xor_sync_75;
#endif
        float vns1_3 = (v_regs[3] - dots1_3) * betas1;
#if __CUDA_ARCH__ == 900
        float vn2s1_3_x = vns1_3;
        float vn2s1_3_y = vns1_3;
        float _fma_272;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_272)
            : "f"(k_regs[0]), "f"(vn2s1_3_x), "f"(S[48]));
        float _fma_273;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_273)
            : "f"(k_regs[1]), "f"(vn2s1_3_y), "f"(S[49]));
        S[48] = _fma_272;
        S[49] = _fma_273;
        float _fma_274;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_274)
            : "f"(k_regs[2]), "f"(vn2s1_3_x), "f"(S[50]));
        float _fma_275;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_275)
            : "f"(k_regs[3]), "f"(vn2s1_3_y), "f"(S[51]));
        S[50] = _fma_274;
        S[51] = _fma_275;
        float _fma_276;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_276)
            : "f"(k_regs[4]), "f"(vn2s1_3_x), "f"(S[52]));
        float _fma_277;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_277)
            : "f"(k_regs[5]), "f"(vn2s1_3_y), "f"(S[53]));
        S[52] = _fma_276;
        S[53] = _fma_277;
        float _fma_278;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_278)
            : "f"(k_regs[6]), "f"(vn2s1_3_x), "f"(S[54]));
        float _fma_279;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_279)
            : "f"(k_regs[7]), "f"(vn2s1_3_y), "f"(S[55]));
        S[54] = _fma_278;
        S[55] = _fma_279;
        float _fma_280;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_280)
            : "f"(k_regs[8]), "f"(vn2s1_3_x), "f"(S[56]));
        float _fma_281;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_281)
            : "f"(k_regs[9]), "f"(vn2s1_3_y), "f"(S[57]));
        S[56] = _fma_280;
        S[57] = _fma_281;
        float _fma_282;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_282)
            : "f"(k_regs[10]), "f"(vn2s1_3_x), "f"(S[58]));
        float _fma_283;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_283)
            : "f"(k_regs[11]), "f"(vn2s1_3_y), "f"(S[59]));
        S[58] = _fma_282;
        S[59] = _fma_283;
        float _fma_284;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_284)
            : "f"(k_regs[12]), "f"(vn2s1_3_x), "f"(S[60]));
        float _fma_285;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_285)
            : "f"(k_regs[13]), "f"(vn2s1_3_y), "f"(S[61]));
        S[60] = _fma_284;
        S[61] = _fma_285;
        float _fma_286;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_286)
            : "f"(k_regs[14]), "f"(vn2s1_3_x), "f"(S[62]));
        float _fma_287;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_287)
            : "f"(k_regs[15]), "f"(vn2s1_3_y), "f"(S[63]));
        S[62] = _fma_286;
        S[63] = _fma_287;
#else
        float2 vn2s1_3 = make_float2(vns1_3, vns1_3);
        S2[24] = fma_f32x2_rn_ftz(k2[0], vn2s1_3, S2[24]);
        S2[25] = fma_f32x2_rn_ftz(k2[1], vn2s1_3, S2[25]);
        S2[26] = fma_f32x2_rn_ftz(k2[2], vn2s1_3, S2[26]);
        S2[27] = fma_f32x2_rn_ftz(k2[3], vn2s1_3, S2[27]);
        S2[28] = fma_f32x2_rn_ftz(k2[4], vn2s1_3, S2[28]);
        S2[29] = fma_f32x2_rn_ftz(k2[5], vn2s1_3, S2[29]);
        S2[30] = fma_f32x2_rn_ftz(k2[6], vn2s1_3, S2[30]);
        S2[31] = fma_f32x2_rn_ftz(k2[7], vn2s1_3, S2[31]);
#endif
        int ms2 = t * 3 + 2;
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
                     : "r"(kbases2 + 128));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                     : "r"(kbases2 + 256));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                     : "r"(kbases2 + 384));
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
        float _fma_288;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_288)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accs2_0_0_x));
        float _fma_289;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_289)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accs2_0_0_y));
        accs2_0_0_x = _fma_288;
        accs2_0_0_y = _fma_289;
        float _fma_290;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_290)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accs2_0_1_x));
        float _fma_291;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_291)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accs2_0_1_y));
        accs2_0_1_x = _fma_290;
        accs2_0_1_y = _fma_291;
        float _fma_292;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_292)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accs2_0_0_x));
        float _fma_293;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_293)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accs2_0_0_y));
        accs2_0_0_x = _fma_292;
        accs2_0_0_y = _fma_293;
        float _fma_294;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_294)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accs2_0_1_x));
        float _fma_295;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_295)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accs2_0_1_y));
        accs2_0_1_x = _fma_294;
        accs2_0_1_y = _fma_295;
        float _fma_296;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_296)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accs2_0_0_x));
        float _fma_297;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_297)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accs2_0_0_y));
        accs2_0_0_x = _fma_296;
        accs2_0_0_y = _fma_297;
        float _fma_298;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_298)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accs2_0_1_x));
        float _fma_299;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_299)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accs2_0_1_y));
        accs2_0_1_x = _fma_298;
        accs2_0_1_y = _fma_299;
        float _fma_300;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_300)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accs2_0_0_x));
        float _fma_301;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_301)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accs2_0_0_y));
        accs2_0_0_x = _fma_300;
        accs2_0_0_y = _fma_301;
        float _fma_302;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_302)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accs2_0_1_x));
        float _fma_303;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_303)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accs2_0_1_y));
        accs2_0_1_x = _fma_302;
        accs2_0_1_y = _fma_303;
        float _add_92;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(accs2_0_0_x), "f"(accs2_0_1_x));
        float _add_93;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(accs2_0_0_y), "f"(accs2_0_1_y));
        float dots2_0 = _add_92 + _add_93;
        float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 1);
        dots2_0 = dots2_0 + _shfl_xor_sync_94;
        float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 2);
        dots2_0 = dots2_0 + _shfl_xor_sync_95;
        float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 4);
        dots2_0 = dots2_0 + _shfl_xor_sync_96;
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
        float dots2_0 = accs2_0_0.x + accs2_0_0.y;
        float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 1);
        dots2_0 = dots2_0 + _shfl_xor_sync_76;
        float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 2);
        dots2_0 = dots2_0 + _shfl_xor_sync_77;
        float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 4);
        dots2_0 = dots2_0 + _shfl_xor_sync_78;
#endif
        float vns2_0 = (v_regs[0] - dots2_0) * betas2;
#if __CUDA_ARCH__ == 900
        float vn2s2_0_x = vns2_0;
        float vn2s2_0_y = vns2_0;
        float _fma_304;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_304)
            : "f"(k_regs[0]), "f"(vn2s2_0_x), "f"(S[0]));
        float _fma_305;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_305)
            : "f"(k_regs[1]), "f"(vn2s2_0_y), "f"(S[1]));
        S[0] = _fma_304;
        S[1] = _fma_305;
        float _fma_306;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_306)
            : "f"(k_regs[2]), "f"(vn2s2_0_x), "f"(S[2]));
        float _fma_307;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_307)
            : "f"(k_regs[3]), "f"(vn2s2_0_y), "f"(S[3]));
        S[2] = _fma_306;
        S[3] = _fma_307;
        float _fma_308;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_308)
            : "f"(k_regs[4]), "f"(vn2s2_0_x), "f"(S[4]));
        float _fma_309;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_309)
            : "f"(k_regs[5]), "f"(vn2s2_0_y), "f"(S[5]));
        S[4] = _fma_308;
        S[5] = _fma_309;
        float _fma_310;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_310)
            : "f"(k_regs[6]), "f"(vn2s2_0_x), "f"(S[6]));
        float _fma_311;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_311)
            : "f"(k_regs[7]), "f"(vn2s2_0_y), "f"(S[7]));
        S[6] = _fma_310;
        S[7] = _fma_311;
        float _fma_312;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_312)
            : "f"(k_regs[8]), "f"(vn2s2_0_x), "f"(S[8]));
        float _fma_313;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_313)
            : "f"(k_regs[9]), "f"(vn2s2_0_y), "f"(S[9]));
        S[8] = _fma_312;
        S[9] = _fma_313;
        float _fma_314;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_314)
            : "f"(k_regs[10]), "f"(vn2s2_0_x), "f"(S[10]));
        float _fma_315;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_315)
            : "f"(k_regs[11]), "f"(vn2s2_0_y), "f"(S[11]));
        S[10] = _fma_314;
        S[11] = _fma_315;
        float _fma_316;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_316)
            : "f"(k_regs[12]), "f"(vn2s2_0_x), "f"(S[12]));
        float _fma_317;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_317)
            : "f"(k_regs[13]), "f"(vn2s2_0_y), "f"(S[13]));
        S[12] = _fma_316;
        S[13] = _fma_317;
        float _fma_318;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_318)
            : "f"(k_regs[14]), "f"(vn2s2_0_x), "f"(S[14]));
        float _fma_319;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_319)
            : "f"(k_regs[15]), "f"(vn2s2_0_y), "f"(S[15]));
        S[14] = _fma_318;
        S[15] = _fma_319;
        float accs2_1_0_x = 0.0f;
        float accs2_1_0_y = 0.0f;
        float accs2_1_1_x = 0.0f;
        float accs2_1_1_y = 0.0f;
        float _fma_320;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_320)
            : "f"(S[16]), "f"(k_regs[0]), "f"(accs2_1_0_x));
        float _fma_321;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_321)
            : "f"(S[17]), "f"(k_regs[1]), "f"(accs2_1_0_y));
        accs2_1_0_x = _fma_320;
        accs2_1_0_y = _fma_321;
        float _fma_322;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_322)
            : "f"(S[18]), "f"(k_regs[2]), "f"(accs2_1_1_x));
        float _fma_323;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_323)
            : "f"(S[19]), "f"(k_regs[3]), "f"(accs2_1_1_y));
        accs2_1_1_x = _fma_322;
        accs2_1_1_y = _fma_323;
        float _fma_324;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_324)
            : "f"(S[20]), "f"(k_regs[4]), "f"(accs2_1_0_x));
        float _fma_325;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_325)
            : "f"(S[21]), "f"(k_regs[5]), "f"(accs2_1_0_y));
        accs2_1_0_x = _fma_324;
        accs2_1_0_y = _fma_325;
        float _fma_326;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_326)
            : "f"(S[22]), "f"(k_regs[6]), "f"(accs2_1_1_x));
        float _fma_327;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_327)
            : "f"(S[23]), "f"(k_regs[7]), "f"(accs2_1_1_y));
        accs2_1_1_x = _fma_326;
        accs2_1_1_y = _fma_327;
        float _fma_328;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_328)
            : "f"(S[24]), "f"(k_regs[8]), "f"(accs2_1_0_x));
        float _fma_329;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_329)
            : "f"(S[25]), "f"(k_regs[9]), "f"(accs2_1_0_y));
        accs2_1_0_x = _fma_328;
        accs2_1_0_y = _fma_329;
        float _fma_330;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_330)
            : "f"(S[26]), "f"(k_regs[10]), "f"(accs2_1_1_x));
        float _fma_331;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_331)
            : "f"(S[27]), "f"(k_regs[11]), "f"(accs2_1_1_y));
        accs2_1_1_x = _fma_330;
        accs2_1_1_y = _fma_331;
        float _fma_332;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_332)
            : "f"(S[28]), "f"(k_regs[12]), "f"(accs2_1_0_x));
        float _fma_333;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_333)
            : "f"(S[29]), "f"(k_regs[13]), "f"(accs2_1_0_y));
        accs2_1_0_x = _fma_332;
        accs2_1_0_y = _fma_333;
        float _fma_334;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_334)
            : "f"(S[30]), "f"(k_regs[14]), "f"(accs2_1_1_x));
        float _fma_335;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_335)
            : "f"(S[31]), "f"(k_regs[15]), "f"(accs2_1_1_y));
        accs2_1_1_x = _fma_334;
        accs2_1_1_y = _fma_335;
        float _add_97;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(accs2_1_0_x), "f"(accs2_1_1_x));
        float _add_98;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_98) : "f"(accs2_1_0_y), "f"(accs2_1_1_y));
        float dots2_1 = _add_97 + _add_98;
        float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 1);
        dots2_1 = dots2_1 + _shfl_xor_sync_99;
        float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 2);
        dots2_1 = dots2_1 + _shfl_xor_sync_100;
        float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 4);
        dots2_1 = dots2_1 + _shfl_xor_sync_101;
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
        float2 accs2_1_0 = make_float2(0.0f, 0.0f);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[9], k2[1], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[11], k2[3], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[13], k2[5], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accs2_1_0);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[15], k2[7], accs2_1_0);
        float dots2_1 = accs2_1_0.x + accs2_1_0.y;
        float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 1);
        dots2_1 = dots2_1 + _shfl_xor_sync_79;
        float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 2);
        dots2_1 = dots2_1 + _shfl_xor_sync_80;
        float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 4);
        dots2_1 = dots2_1 + _shfl_xor_sync_81;
#endif
        float vns2_1 = (v_regs[1] - dots2_1) * betas2;
#if __CUDA_ARCH__ == 900
        float vn2s2_1_x = vns2_1;
        float vn2s2_1_y = vns2_1;
        float _fma_336;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_336)
            : "f"(k_regs[0]), "f"(vn2s2_1_x), "f"(S[16]));
        float _fma_337;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_337)
            : "f"(k_regs[1]), "f"(vn2s2_1_y), "f"(S[17]));
        S[16] = _fma_336;
        S[17] = _fma_337;
        float _fma_338;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_338)
            : "f"(k_regs[2]), "f"(vn2s2_1_x), "f"(S[18]));
        float _fma_339;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_339)
            : "f"(k_regs[3]), "f"(vn2s2_1_y), "f"(S[19]));
        S[18] = _fma_338;
        S[19] = _fma_339;
        float _fma_340;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_340)
            : "f"(k_regs[4]), "f"(vn2s2_1_x), "f"(S[20]));
        float _fma_341;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_341)
            : "f"(k_regs[5]), "f"(vn2s2_1_y), "f"(S[21]));
        S[20] = _fma_340;
        S[21] = _fma_341;
        float _fma_342;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_342)
            : "f"(k_regs[6]), "f"(vn2s2_1_x), "f"(S[22]));
        float _fma_343;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_343)
            : "f"(k_regs[7]), "f"(vn2s2_1_y), "f"(S[23]));
        S[22] = _fma_342;
        S[23] = _fma_343;
        float _fma_344;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_344)
            : "f"(k_regs[8]), "f"(vn2s2_1_x), "f"(S[24]));
        float _fma_345;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_345)
            : "f"(k_regs[9]), "f"(vn2s2_1_y), "f"(S[25]));
        S[24] = _fma_344;
        S[25] = _fma_345;
        float _fma_346;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_346)
            : "f"(k_regs[10]), "f"(vn2s2_1_x), "f"(S[26]));
        float _fma_347;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_347)
            : "f"(k_regs[11]), "f"(vn2s2_1_y), "f"(S[27]));
        S[26] = _fma_346;
        S[27] = _fma_347;
        float _fma_348;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_348)
            : "f"(k_regs[12]), "f"(vn2s2_1_x), "f"(S[28]));
        float _fma_349;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_349)
            : "f"(k_regs[13]), "f"(vn2s2_1_y), "f"(S[29]));
        S[28] = _fma_348;
        S[29] = _fma_349;
        float _fma_350;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_350)
            : "f"(k_regs[14]), "f"(vn2s2_1_x), "f"(S[30]));
        float _fma_351;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_351)
            : "f"(k_regs[15]), "f"(vn2s2_1_y), "f"(S[31]));
        S[30] = _fma_350;
        S[31] = _fma_351;
        float accs2_2_0_x = 0.0f;
        float accs2_2_0_y = 0.0f;
        float accs2_2_1_x = 0.0f;
        float accs2_2_1_y = 0.0f;
        float _fma_352;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_352)
            : "f"(S[32]), "f"(k_regs[0]), "f"(accs2_2_0_x));
        float _fma_353;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_353)
            : "f"(S[33]), "f"(k_regs[1]), "f"(accs2_2_0_y));
        accs2_2_0_x = _fma_352;
        accs2_2_0_y = _fma_353;
        float _fma_354;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_354)
            : "f"(S[34]), "f"(k_regs[2]), "f"(accs2_2_1_x));
        float _fma_355;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_355)
            : "f"(S[35]), "f"(k_regs[3]), "f"(accs2_2_1_y));
        accs2_2_1_x = _fma_354;
        accs2_2_1_y = _fma_355;
        float _fma_356;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_356)
            : "f"(S[36]), "f"(k_regs[4]), "f"(accs2_2_0_x));
        float _fma_357;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_357)
            : "f"(S[37]), "f"(k_regs[5]), "f"(accs2_2_0_y));
        accs2_2_0_x = _fma_356;
        accs2_2_0_y = _fma_357;
        float _fma_358;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_358)
            : "f"(S[38]), "f"(k_regs[6]), "f"(accs2_2_1_x));
        float _fma_359;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_359)
            : "f"(S[39]), "f"(k_regs[7]), "f"(accs2_2_1_y));
        accs2_2_1_x = _fma_358;
        accs2_2_1_y = _fma_359;
        float _fma_360;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_360)
            : "f"(S[40]), "f"(k_regs[8]), "f"(accs2_2_0_x));
        float _fma_361;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_361)
            : "f"(S[41]), "f"(k_regs[9]), "f"(accs2_2_0_y));
        accs2_2_0_x = _fma_360;
        accs2_2_0_y = _fma_361;
        float _fma_362;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_362)
            : "f"(S[42]), "f"(k_regs[10]), "f"(accs2_2_1_x));
        float _fma_363;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_363)
            : "f"(S[43]), "f"(k_regs[11]), "f"(accs2_2_1_y));
        accs2_2_1_x = _fma_362;
        accs2_2_1_y = _fma_363;
        float _fma_364;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_364)
            : "f"(S[44]), "f"(k_regs[12]), "f"(accs2_2_0_x));
        float _fma_365;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_365)
            : "f"(S[45]), "f"(k_regs[13]), "f"(accs2_2_0_y));
        accs2_2_0_x = _fma_364;
        accs2_2_0_y = _fma_365;
        float _fma_366;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_366)
            : "f"(S[46]), "f"(k_regs[14]), "f"(accs2_2_1_x));
        float _fma_367;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_367)
            : "f"(S[47]), "f"(k_regs[15]), "f"(accs2_2_1_y));
        accs2_2_1_x = _fma_366;
        accs2_2_1_y = _fma_367;
        float _add_102;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(accs2_2_0_x), "f"(accs2_2_1_x));
        float _add_103;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(accs2_2_0_y), "f"(accs2_2_1_y));
        float dots2_2 = _add_102 + _add_103;
        float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, dots2_2, 1);
        dots2_2 = dots2_2 + _shfl_xor_sync_104;
        float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, dots2_2, 2);
        dots2_2 = dots2_2 + _shfl_xor_sync_105;
        float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, dots2_2, 4);
        dots2_2 = dots2_2 + _shfl_xor_sync_106;
#else
        float2 vn2s2_1 = make_float2(vns2_1, vns2_1);
        S2[8] = fma_f32x2_rn_ftz(k2[0], vn2s2_1, S2[8]);
        S2[9] = fma_f32x2_rn_ftz(k2[1], vn2s2_1, S2[9]);
        S2[10] = fma_f32x2_rn_ftz(k2[2], vn2s2_1, S2[10]);
        S2[11] = fma_f32x2_rn_ftz(k2[3], vn2s2_1, S2[11]);
        S2[12] = fma_f32x2_rn_ftz(k2[4], vn2s2_1, S2[12]);
        S2[13] = fma_f32x2_rn_ftz(k2[5], vn2s2_1, S2[13]);
        S2[14] = fma_f32x2_rn_ftz(k2[6], vn2s2_1, S2[14]);
        S2[15] = fma_f32x2_rn_ftz(k2[7], vn2s2_1, S2[15]);
        float2 accs2_2_0 = make_float2(0.0f, 0.0f);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accs2_2_0);
        accs2_2_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accs2_2_0);
        float dots2_2 = accs2_2_0.x + accs2_2_0.y;
        float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, dots2_2, 1);
        dots2_2 = dots2_2 + _shfl_xor_sync_82;
        float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, dots2_2, 2);
        dots2_2 = dots2_2 + _shfl_xor_sync_83;
        float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, dots2_2, 4);
        dots2_2 = dots2_2 + _shfl_xor_sync_84;
#endif
        float vns2_2 = (v_regs[2] - dots2_2) * betas2;
#if __CUDA_ARCH__ == 900
        float vn2s2_2_x = vns2_2;
        float vn2s2_2_y = vns2_2;
        float _fma_368;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_368)
            : "f"(k_regs[0]), "f"(vn2s2_2_x), "f"(S[32]));
        float _fma_369;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_369)
            : "f"(k_regs[1]), "f"(vn2s2_2_y), "f"(S[33]));
        S[32] = _fma_368;
        S[33] = _fma_369;
        float _fma_370;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_370)
            : "f"(k_regs[2]), "f"(vn2s2_2_x), "f"(S[34]));
        float _fma_371;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_371)
            : "f"(k_regs[3]), "f"(vn2s2_2_y), "f"(S[35]));
        S[34] = _fma_370;
        S[35] = _fma_371;
        float _fma_372;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_372)
            : "f"(k_regs[4]), "f"(vn2s2_2_x), "f"(S[36]));
        float _fma_373;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_373)
            : "f"(k_regs[5]), "f"(vn2s2_2_y), "f"(S[37]));
        S[36] = _fma_372;
        S[37] = _fma_373;
        float _fma_374;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_374)
            : "f"(k_regs[6]), "f"(vn2s2_2_x), "f"(S[38]));
        float _fma_375;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_375)
            : "f"(k_regs[7]), "f"(vn2s2_2_y), "f"(S[39]));
        S[38] = _fma_374;
        S[39] = _fma_375;
        float _fma_376;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_376)
            : "f"(k_regs[8]), "f"(vn2s2_2_x), "f"(S[40]));
        float _fma_377;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_377)
            : "f"(k_regs[9]), "f"(vn2s2_2_y), "f"(S[41]));
        S[40] = _fma_376;
        S[41] = _fma_377;
        float _fma_378;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_378)
            : "f"(k_regs[10]), "f"(vn2s2_2_x), "f"(S[42]));
        float _fma_379;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_379)
            : "f"(k_regs[11]), "f"(vn2s2_2_y), "f"(S[43]));
        S[42] = _fma_378;
        S[43] = _fma_379;
        float _fma_380;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_380)
            : "f"(k_regs[12]), "f"(vn2s2_2_x), "f"(S[44]));
        float _fma_381;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_381)
            : "f"(k_regs[13]), "f"(vn2s2_2_y), "f"(S[45]));
        S[44] = _fma_380;
        S[45] = _fma_381;
        float _fma_382;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_382)
            : "f"(k_regs[14]), "f"(vn2s2_2_x), "f"(S[46]));
        float _fma_383;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_383)
            : "f"(k_regs[15]), "f"(vn2s2_2_y), "f"(S[47]));
        S[46] = _fma_382;
        S[47] = _fma_383;
        float accs2_3_0_x = 0.0f;
        float accs2_3_0_y = 0.0f;
        float accs2_3_1_x = 0.0f;
        float accs2_3_1_y = 0.0f;
        float _fma_384;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_384)
            : "f"(S[48]), "f"(k_regs[0]), "f"(accs2_3_0_x));
        float _fma_385;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_385)
            : "f"(S[49]), "f"(k_regs[1]), "f"(accs2_3_0_y));
        accs2_3_0_x = _fma_384;
        accs2_3_0_y = _fma_385;
        float _fma_386;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_386)
            : "f"(S[50]), "f"(k_regs[2]), "f"(accs2_3_1_x));
        float _fma_387;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_387)
            : "f"(S[51]), "f"(k_regs[3]), "f"(accs2_3_1_y));
        accs2_3_1_x = _fma_386;
        accs2_3_1_y = _fma_387;
        float _fma_388;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_388)
            : "f"(S[52]), "f"(k_regs[4]), "f"(accs2_3_0_x));
        float _fma_389;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_389)
            : "f"(S[53]), "f"(k_regs[5]), "f"(accs2_3_0_y));
        accs2_3_0_x = _fma_388;
        accs2_3_0_y = _fma_389;
        float _fma_390;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_390)
            : "f"(S[54]), "f"(k_regs[6]), "f"(accs2_3_1_x));
        float _fma_391;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_391)
            : "f"(S[55]), "f"(k_regs[7]), "f"(accs2_3_1_y));
        accs2_3_1_x = _fma_390;
        accs2_3_1_y = _fma_391;
        float _fma_392;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_392)
            : "f"(S[56]), "f"(k_regs[8]), "f"(accs2_3_0_x));
        float _fma_393;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_393)
            : "f"(S[57]), "f"(k_regs[9]), "f"(accs2_3_0_y));
        accs2_3_0_x = _fma_392;
        accs2_3_0_y = _fma_393;
        float _fma_394;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_394)
            : "f"(S[58]), "f"(k_regs[10]), "f"(accs2_3_1_x));
        float _fma_395;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_395)
            : "f"(S[59]), "f"(k_regs[11]), "f"(accs2_3_1_y));
        accs2_3_1_x = _fma_394;
        accs2_3_1_y = _fma_395;
        float _fma_396;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_396)
            : "f"(S[60]), "f"(k_regs[12]), "f"(accs2_3_0_x));
        float _fma_397;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_397)
            : "f"(S[61]), "f"(k_regs[13]), "f"(accs2_3_0_y));
        accs2_3_0_x = _fma_396;
        accs2_3_0_y = _fma_397;
        float _fma_398;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_398)
            : "f"(S[62]), "f"(k_regs[14]), "f"(accs2_3_1_x));
        float _fma_399;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_399)
            : "f"(S[63]), "f"(k_regs[15]), "f"(accs2_3_1_y));
        accs2_3_1_x = _fma_398;
        accs2_3_1_y = _fma_399;
        float _add_107;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_107) : "f"(accs2_3_0_x), "f"(accs2_3_1_x));
        float _add_108;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(accs2_3_0_y), "f"(accs2_3_1_y));
        float dots2_3 = _add_107 + _add_108;
        float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, dots2_3, 1);
        dots2_3 = dots2_3 + _shfl_xor_sync_109;
        float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, dots2_3, 2);
        dots2_3 = dots2_3 + _shfl_xor_sync_110;
        float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, dots2_3, 4);
        dots2_3 = dots2_3 + _shfl_xor_sync_111;
#else
        float2 vn2s2_2 = make_float2(vns2_2, vns2_2);
        S2[16] = fma_f32x2_rn_ftz(k2[0], vn2s2_2, S2[16]);
        S2[17] = fma_f32x2_rn_ftz(k2[1], vn2s2_2, S2[17]);
        S2[18] = fma_f32x2_rn_ftz(k2[2], vn2s2_2, S2[18]);
        S2[19] = fma_f32x2_rn_ftz(k2[3], vn2s2_2, S2[19]);
        S2[20] = fma_f32x2_rn_ftz(k2[4], vn2s2_2, S2[20]);
        S2[21] = fma_f32x2_rn_ftz(k2[5], vn2s2_2, S2[21]);
        S2[22] = fma_f32x2_rn_ftz(k2[6], vn2s2_2, S2[22]);
        S2[23] = fma_f32x2_rn_ftz(k2[7], vn2s2_2, S2[23]);
        float2 accs2_3_0 = make_float2(0.0f, 0.0f);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[24], k2[0], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[25], k2[1], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[26], k2[2], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[27], k2[3], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[28], k2[4], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[29], k2[5], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[30], k2[6], accs2_3_0);
        accs2_3_0 = fma_f32x2_rn_ftz(S2[31], k2[7], accs2_3_0);
        float dots2_3 = accs2_3_0.x + accs2_3_0.y;
        float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, dots2_3, 1);
        dots2_3 = dots2_3 + _shfl_xor_sync_85;
        float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, dots2_3, 2);
        dots2_3 = dots2_3 + _shfl_xor_sync_86;
        float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, dots2_3, 4);
        dots2_3 = dots2_3 + _shfl_xor_sync_87;
#endif
        float vns2_3 = (v_regs[3] - dots2_3) * betas2;
#if __CUDA_ARCH__ == 900
        float vn2s2_3_x = vns2_3;
        float vn2s2_3_y = vns2_3;
        float _fma_400;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_400)
            : "f"(k_regs[0]), "f"(vn2s2_3_x), "f"(S[48]));
        float _fma_401;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_401)
            : "f"(k_regs[1]), "f"(vn2s2_3_y), "f"(S[49]));
        S[48] = _fma_400;
        S[49] = _fma_401;
        float _fma_402;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_402)
            : "f"(k_regs[2]), "f"(vn2s2_3_x), "f"(S[50]));
        float _fma_403;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_403)
            : "f"(k_regs[3]), "f"(vn2s2_3_y), "f"(S[51]));
        S[50] = _fma_402;
        S[51] = _fma_403;
        float _fma_404;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_404)
            : "f"(k_regs[4]), "f"(vn2s2_3_x), "f"(S[52]));
        float _fma_405;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_405)
            : "f"(k_regs[5]), "f"(vn2s2_3_y), "f"(S[53]));
        S[52] = _fma_404;
        S[53] = _fma_405;
        float _fma_406;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_406)
            : "f"(k_regs[6]), "f"(vn2s2_3_x), "f"(S[54]));
        float _fma_407;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_407)
            : "f"(k_regs[7]), "f"(vn2s2_3_y), "f"(S[55]));
        S[54] = _fma_406;
        S[55] = _fma_407;
        float _fma_408;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_408)
            : "f"(k_regs[8]), "f"(vn2s2_3_x), "f"(S[56]));
        float _fma_409;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_409)
            : "f"(k_regs[9]), "f"(vn2s2_3_y), "f"(S[57]));
        S[56] = _fma_408;
        S[57] = _fma_409;
        float _fma_410;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_410)
            : "f"(k_regs[10]), "f"(vn2s2_3_x), "f"(S[58]));
        float _fma_411;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_411)
            : "f"(k_regs[11]), "f"(vn2s2_3_y), "f"(S[59]));
        S[58] = _fma_410;
        S[59] = _fma_411;
        float _fma_412;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_412)
            : "f"(k_regs[12]), "f"(vn2s2_3_x), "f"(S[60]));
        float _fma_413;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_413)
            : "f"(k_regs[13]), "f"(vn2s2_3_y), "f"(S[61]));
        S[60] = _fma_412;
        S[61] = _fma_413;
        float _fma_414;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_414)
            : "f"(k_regs[14]), "f"(vn2s2_3_x), "f"(S[62]));
        float _fma_415;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_415)
            : "f"(k_regs[15]), "f"(vn2s2_3_y), "f"(S[63]));
        S[62] = _fma_414;
        S[63] = _fma_415;
#else
        float2 vn2s2_3 = make_float2(vns2_3, vns2_3);
        S2[24] = fma_f32x2_rn_ftz(k2[0], vn2s2_3, S2[24]);
        S2[25] = fma_f32x2_rn_ftz(k2[1], vn2s2_3, S2[25]);
        S2[26] = fma_f32x2_rn_ftz(k2[2], vn2s2_3, S2[26]);
        S2[27] = fma_f32x2_rn_ftz(k2[3], vn2s2_3, S2[27]);
        S2[28] = fma_f32x2_rn_ftz(k2[4], vn2s2_3, S2[28]);
        S2[29] = fma_f32x2_rn_ftz(k2[5], vn2s2_3, S2[29]);
        S2[30] = fma_f32x2_rn_ftz(k2[6], vn2s2_3, S2[30]);
        S2[31] = fma_f32x2_rn_ftz(k2[7], vn2s2_3, S2[31]);
#endif
        unsigned int qbases = sQ_addr + (unsigned int)(t * 512) + lane_byte;
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
                     : "r"(qbases + 128));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                     : "r"(qbases + 256));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                     : "r"(qbases + 384));
#if __CUDA_ARCH__ == 900
        float accsq_0_0_x = 0.0f;
        float accsq_0_0_y = 0.0f;
        float accsq_0_1_x = 0.0f;
        float accsq_0_1_y = 0.0f;
        float _fma_416;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_416)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accsq_0_0_x));
        float _fma_417;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_417)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accsq_0_0_y));
        accsq_0_0_x = _fma_416;
        accsq_0_0_y = _fma_417;
        float _fma_418;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_418)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accsq_0_1_x));
        float _fma_419;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_419)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accsq_0_1_y));
        accsq_0_1_x = _fma_418;
        accsq_0_1_y = _fma_419;
        float _fma_420;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_420)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accsq_0_0_x));
        float _fma_421;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_421)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accsq_0_0_y));
        accsq_0_0_x = _fma_420;
        accsq_0_0_y = _fma_421;
        float _fma_422;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_422)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accsq_0_1_x));
        float _fma_423;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_423)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accsq_0_1_y));
        accsq_0_1_x = _fma_422;
        accsq_0_1_y = _fma_423;
        float _fma_424;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_424)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accsq_0_0_x));
        float _fma_425;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_425)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accsq_0_0_y));
        accsq_0_0_x = _fma_424;
        accsq_0_0_y = _fma_425;
        float _fma_426;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_426)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accsq_0_1_x));
        float _fma_427;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_427)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accsq_0_1_y));
        accsq_0_1_x = _fma_426;
        accsq_0_1_y = _fma_427;
        float _fma_428;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_428)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accsq_0_0_x));
        float _fma_429;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_429)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accsq_0_0_y));
        accsq_0_0_x = _fma_428;
        accsq_0_0_y = _fma_429;
        float _fma_430;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_430)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accsq_0_1_x));
        float _fma_431;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_431)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accsq_0_1_y));
        accsq_0_1_x = _fma_430;
        accsq_0_1_y = _fma_431;
        float _add_112;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(accsq_0_0_x), "f"(accsq_0_1_x));
        float _add_113;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(accsq_0_0_y), "f"(accsq_0_1_y));
        float dotsq_0 = _add_112 + _add_113;
        float _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 1);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_114;
        float _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 2);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_115;
        float _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 4);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_116;
#else
        k2[0] = make_float2(k_regs[0], k_regs[1]);
        k2[1] = make_float2(k_regs[2], k_regs[3]);
        k2[2] = make_float2(k_regs[4], k_regs[5]);
        k2[3] = make_float2(k_regs[6], k_regs[7]);
        k2[4] = make_float2(k_regs[8], k_regs[9]);
        k2[5] = make_float2(k_regs[10], k_regs[11]);
        k2[6] = make_float2(k_regs[12], k_regs[13]);
        k2[7] = make_float2(k_regs[14], k_regs[15]);
        float2 accsq_0_0 = make_float2(0.0f, 0.0f);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[1], k2[1], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[3], k2[3], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[5], k2[5], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accsq_0_0);
        accsq_0_0 = fma_f32x2_rn_ftz(S2[7], k2[7], accsq_0_0);
        float dotsq_0 = accsq_0_0.x + accsq_0_0.y;
        float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 1);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_88;
        float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 2);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_89;
        float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 4);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_90;
#endif
        o4[0] = dotsq_0;
#if __CUDA_ARCH__ == 900
        float accsq_1_0_x = 0.0f;
        float accsq_1_0_y = 0.0f;
        float accsq_1_1_x = 0.0f;
        float accsq_1_1_y = 0.0f;
        float _fma_432;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_432)
            : "f"(S[16]), "f"(k_regs[0]), "f"(accsq_1_0_x));
        float _fma_433;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_433)
            : "f"(S[17]), "f"(k_regs[1]), "f"(accsq_1_0_y));
        accsq_1_0_x = _fma_432;
        accsq_1_0_y = _fma_433;
        float _fma_434;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_434)
            : "f"(S[18]), "f"(k_regs[2]), "f"(accsq_1_1_x));
        float _fma_435;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_435)
            : "f"(S[19]), "f"(k_regs[3]), "f"(accsq_1_1_y));
        accsq_1_1_x = _fma_434;
        accsq_1_1_y = _fma_435;
        float _fma_436;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_436)
            : "f"(S[20]), "f"(k_regs[4]), "f"(accsq_1_0_x));
        float _fma_437;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_437)
            : "f"(S[21]), "f"(k_regs[5]), "f"(accsq_1_0_y));
        accsq_1_0_x = _fma_436;
        accsq_1_0_y = _fma_437;
        float _fma_438;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_438)
            : "f"(S[22]), "f"(k_regs[6]), "f"(accsq_1_1_x));
        float _fma_439;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_439)
            : "f"(S[23]), "f"(k_regs[7]), "f"(accsq_1_1_y));
        accsq_1_1_x = _fma_438;
        accsq_1_1_y = _fma_439;
        float _fma_440;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_440)
            : "f"(S[24]), "f"(k_regs[8]), "f"(accsq_1_0_x));
        float _fma_441;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_441)
            : "f"(S[25]), "f"(k_regs[9]), "f"(accsq_1_0_y));
        accsq_1_0_x = _fma_440;
        accsq_1_0_y = _fma_441;
        float _fma_442;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_442)
            : "f"(S[26]), "f"(k_regs[10]), "f"(accsq_1_1_x));
        float _fma_443;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_443)
            : "f"(S[27]), "f"(k_regs[11]), "f"(accsq_1_1_y));
        accsq_1_1_x = _fma_442;
        accsq_1_1_y = _fma_443;
        float _fma_444;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_444)
            : "f"(S[28]), "f"(k_regs[12]), "f"(accsq_1_0_x));
        float _fma_445;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_445)
            : "f"(S[29]), "f"(k_regs[13]), "f"(accsq_1_0_y));
        accsq_1_0_x = _fma_444;
        accsq_1_0_y = _fma_445;
        float _fma_446;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_446)
            : "f"(S[30]), "f"(k_regs[14]), "f"(accsq_1_1_x));
        float _fma_447;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_447)
            : "f"(S[31]), "f"(k_regs[15]), "f"(accsq_1_1_y));
        accsq_1_1_x = _fma_446;
        accsq_1_1_y = _fma_447;
        float _add_117;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_117) : "f"(accsq_1_0_x), "f"(accsq_1_1_x));
        float _add_118;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_118) : "f"(accsq_1_0_y), "f"(accsq_1_1_y));
        float dotsq_1 = _add_117 + _add_118;
        float _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 1);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_119;
        float _shfl_xor_sync_120 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 2);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_120;
        float _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 4);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_121;
#else
        float2 accsq_1_0 = make_float2(0.0f, 0.0f);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[9], k2[1], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[11], k2[3], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[13], k2[5], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accsq_1_0);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[15], k2[7], accsq_1_0);
        float dotsq_1 = accsq_1_0.x + accsq_1_0.y;
        float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 1);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_91;
        float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 2);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_92;
        float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 4);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_93;
#endif
        o4[1] = dotsq_1;
#if __CUDA_ARCH__ == 900
        float accsq_2_0_x = 0.0f;
        float accsq_2_0_y = 0.0f;
        float accsq_2_1_x = 0.0f;
        float accsq_2_1_y = 0.0f;
        float _fma_448;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_448)
            : "f"(S[32]), "f"(k_regs[0]), "f"(accsq_2_0_x));
        float _fma_449;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_449)
            : "f"(S[33]), "f"(k_regs[1]), "f"(accsq_2_0_y));
        accsq_2_0_x = _fma_448;
        accsq_2_0_y = _fma_449;
        float _fma_450;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_450)
            : "f"(S[34]), "f"(k_regs[2]), "f"(accsq_2_1_x));
        float _fma_451;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_451)
            : "f"(S[35]), "f"(k_regs[3]), "f"(accsq_2_1_y));
        accsq_2_1_x = _fma_450;
        accsq_2_1_y = _fma_451;
        float _fma_452;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_452)
            : "f"(S[36]), "f"(k_regs[4]), "f"(accsq_2_0_x));
        float _fma_453;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_453)
            : "f"(S[37]), "f"(k_regs[5]), "f"(accsq_2_0_y));
        accsq_2_0_x = _fma_452;
        accsq_2_0_y = _fma_453;
        float _fma_454;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_454)
            : "f"(S[38]), "f"(k_regs[6]), "f"(accsq_2_1_x));
        float _fma_455;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_455)
            : "f"(S[39]), "f"(k_regs[7]), "f"(accsq_2_1_y));
        accsq_2_1_x = _fma_454;
        accsq_2_1_y = _fma_455;
        float _fma_456;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_456)
            : "f"(S[40]), "f"(k_regs[8]), "f"(accsq_2_0_x));
        float _fma_457;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_457)
            : "f"(S[41]), "f"(k_regs[9]), "f"(accsq_2_0_y));
        accsq_2_0_x = _fma_456;
        accsq_2_0_y = _fma_457;
        float _fma_458;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_458)
            : "f"(S[42]), "f"(k_regs[10]), "f"(accsq_2_1_x));
        float _fma_459;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_459)
            : "f"(S[43]), "f"(k_regs[11]), "f"(accsq_2_1_y));
        accsq_2_1_x = _fma_458;
        accsq_2_1_y = _fma_459;
        float _fma_460;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_460)
            : "f"(S[44]), "f"(k_regs[12]), "f"(accsq_2_0_x));
        float _fma_461;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_461)
            : "f"(S[45]), "f"(k_regs[13]), "f"(accsq_2_0_y));
        accsq_2_0_x = _fma_460;
        accsq_2_0_y = _fma_461;
        float _fma_462;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_462)
            : "f"(S[46]), "f"(k_regs[14]), "f"(accsq_2_1_x));
        float _fma_463;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_463)
            : "f"(S[47]), "f"(k_regs[15]), "f"(accsq_2_1_y));
        accsq_2_1_x = _fma_462;
        accsq_2_1_y = _fma_463;
        float _add_122;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(accsq_2_0_x), "f"(accsq_2_1_x));
        float _add_123;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_123) : "f"(accsq_2_0_y), "f"(accsq_2_1_y));
        float dotsq_2 = _add_122 + _add_123;
        float _shfl_xor_sync_124 = __shfl_xor_sync(0xFFFFFFFF, dotsq_2, 1);
        dotsq_2 = dotsq_2 + _shfl_xor_sync_124;
        float _shfl_xor_sync_125 = __shfl_xor_sync(0xFFFFFFFF, dotsq_2, 2);
        dotsq_2 = dotsq_2 + _shfl_xor_sync_125;
        float _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, dotsq_2, 4);
        dotsq_2 = dotsq_2 + _shfl_xor_sync_126;
#else
        float2 accsq_2_0 = make_float2(0.0f, 0.0f);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accsq_2_0);
        accsq_2_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accsq_2_0);
        float dotsq_2 = accsq_2_0.x + accsq_2_0.y;
        float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, dotsq_2, 1);
        dotsq_2 = dotsq_2 + _shfl_xor_sync_94;
        float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, dotsq_2, 2);
        dotsq_2 = dotsq_2 + _shfl_xor_sync_95;
        float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, dotsq_2, 4);
        dotsq_2 = dotsq_2 + _shfl_xor_sync_96;
#endif
        o4[2] = dotsq_2;
#if __CUDA_ARCH__ == 900
        float accsq_3_0_x = 0.0f;
        float accsq_3_0_y = 0.0f;
        float accsq_3_1_x = 0.0f;
        float accsq_3_1_y = 0.0f;
        float _fma_464;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_464)
            : "f"(S[48]), "f"(k_regs[0]), "f"(accsq_3_0_x));
        float _fma_465;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_465)
            : "f"(S[49]), "f"(k_regs[1]), "f"(accsq_3_0_y));
        accsq_3_0_x = _fma_464;
        accsq_3_0_y = _fma_465;
        float _fma_466;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_466)
            : "f"(S[50]), "f"(k_regs[2]), "f"(accsq_3_1_x));
        float _fma_467;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_467)
            : "f"(S[51]), "f"(k_regs[3]), "f"(accsq_3_1_y));
        accsq_3_1_x = _fma_466;
        accsq_3_1_y = _fma_467;
        float _fma_468;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_468)
            : "f"(S[52]), "f"(k_regs[4]), "f"(accsq_3_0_x));
        float _fma_469;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_469)
            : "f"(S[53]), "f"(k_regs[5]), "f"(accsq_3_0_y));
        accsq_3_0_x = _fma_468;
        accsq_3_0_y = _fma_469;
        float _fma_470;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_470)
            : "f"(S[54]), "f"(k_regs[6]), "f"(accsq_3_1_x));
        float _fma_471;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_471)
            : "f"(S[55]), "f"(k_regs[7]), "f"(accsq_3_1_y));
        accsq_3_1_x = _fma_470;
        accsq_3_1_y = _fma_471;
        float _fma_472;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_472)
            : "f"(S[56]), "f"(k_regs[8]), "f"(accsq_3_0_x));
        float _fma_473;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_473)
            : "f"(S[57]), "f"(k_regs[9]), "f"(accsq_3_0_y));
        accsq_3_0_x = _fma_472;
        accsq_3_0_y = _fma_473;
        float _fma_474;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_474)
            : "f"(S[58]), "f"(k_regs[10]), "f"(accsq_3_1_x));
        float _fma_475;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_475)
            : "f"(S[59]), "f"(k_regs[11]), "f"(accsq_3_1_y));
        accsq_3_1_x = _fma_474;
        accsq_3_1_y = _fma_475;
        float _fma_476;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_476)
            : "f"(S[60]), "f"(k_regs[12]), "f"(accsq_3_0_x));
        float _fma_477;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_477)
            : "f"(S[61]), "f"(k_regs[13]), "f"(accsq_3_0_y));
        accsq_3_0_x = _fma_476;
        accsq_3_0_y = _fma_477;
        float _fma_478;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_478)
            : "f"(S[62]), "f"(k_regs[14]), "f"(accsq_3_1_x));
        float _fma_479;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_479)
            : "f"(S[63]), "f"(k_regs[15]), "f"(accsq_3_1_y));
        accsq_3_1_x = _fma_478;
        accsq_3_1_y = _fma_479;
        float _add_127;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(accsq_3_0_x), "f"(accsq_3_1_x));
        float _add_128;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(accsq_3_0_y), "f"(accsq_3_1_y));
        float dotsq_3 = _add_127 + _add_128;
        float _shfl_xor_sync_129 = __shfl_xor_sync(0xFFFFFFFF, dotsq_3, 1);
        dotsq_3 = dotsq_3 + _shfl_xor_sync_129;
        float _shfl_xor_sync_130 = __shfl_xor_sync(0xFFFFFFFF, dotsq_3, 2);
        dotsq_3 = dotsq_3 + _shfl_xor_sync_130;
        float _shfl_xor_sync_131 = __shfl_xor_sync(0xFFFFFFFF, dotsq_3, 4);
        dotsq_3 = dotsq_3 + _shfl_xor_sync_131;
#else
        float2 accsq_3_0 = make_float2(0.0f, 0.0f);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[24], k2[0], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[25], k2[1], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[26], k2[2], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[27], k2[3], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[28], k2[4], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[29], k2[5], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[30], k2[6], accsq_3_0);
        accsq_3_0 = fma_f32x2_rn_ftz(S2[31], k2[7], accsq_3_0);
        float dotsq_3 = accsq_3_0.x + accsq_3_0.y;
        float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, dotsq_3, 1);
        dotsq_3 = dotsq_3 + _shfl_xor_sync_97;
        float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, dotsq_3, 2);
        dotsq_3 = dotsq_3 + _shfl_xor_sync_98;
        float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, dotsq_3, 4);
        dotsq_3 = dotsq_3 + _shfl_xor_sync_99;
#endif
        o4[3] = dotsq_3;
        if (u == 0) {
          long long out_index = (long long)n * out_stride_p0 + (long long)t * out_stride_p1 +
                                (long long)h * out_stride_p2 + (long long)row0;
          {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(o4[0 + 0], o4[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(o4[0 + 2], o4[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[out_index]) = _pk2;
          }
        }
        int ssm_slot = sSsm[t];
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
                         reinterpret_cast<uint64_t>(&state[snap_base + 32])),
                     "f"(S[4]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 32 + 1])),
                     "f"(S[5]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 34])),
                     "f"(S[6]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 34 + 1])),
                     "f"(S[7]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 64])),
                     "f"(S[8]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 64 + 1])),
                     "f"(S[9]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 66])),
                     "f"(S[10]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 66 + 1])),
                     "f"(S[11]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 96])),
                     "f"(S[12]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 96 + 1])),
                     "f"(S[13]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 98])),
                     "f"(S[14]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 98 + 1])),
                     "f"(S[15]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2])),
                     "f"(S[16]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 1])),
                     "f"(S[17]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 2])),
                     "f"(S[18]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 2 + 1])),
                     "f"(S[19]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 32])),
                     "f"(S[20]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 32 + 1])),
                     "f"(S[21]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 34])),
                     "f"(S[22]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 34 + 1])),
                     "f"(S[23]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 64])),
                     "f"(S[24]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 64 + 1])),
                     "f"(S[25]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 66])),
                     "f"(S[26]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 66 + 1])),
                     "f"(S[27]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 96])),
                     "f"(S[28]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 96 + 1])),
                     "f"(S[29]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 98])),
                     "f"(S[30]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 98 + 1])),
                     "f"(S[31]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2])),
                     "f"(S[32]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 1])),
                     "f"(S[33]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 2])),
                     "f"(S[34]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 2 + 1])),
            "f"(S[35]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 32])),
                     "f"(S[36]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 32 + 1])),
            "f"(S[37]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 34])),
                     "f"(S[38]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 34 + 1])),
            "f"(S[39]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 64])),
                     "f"(S[40]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 64 + 1])),
            "f"(S[41]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 66])),
                     "f"(S[42]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 66 + 1])),
            "f"(S[43]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 96])),
                     "f"(S[44]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 96 + 1])),
            "f"(S[45]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 98])),
                     "f"(S[46]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 98 + 1])),
            "f"(S[47]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2])),
                     "f"(S[48]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 1])),
                     "f"(S[49]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 2])),
                     "f"(S[50]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 2 + 1])),
            "f"(S[51]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 32])),
                     "f"(S[52]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 32 + 1])),
            "f"(S[53]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 34])),
                     "f"(S[54]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 34 + 1])),
            "f"(S[55]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 64])),
                     "f"(S[56]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 64 + 1])),
            "f"(S[57]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 66])),
                     "f"(S[58]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 66 + 1])),
            "f"(S[59]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 96])),
                     "f"(S[60]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 96 + 1])),
            "f"(S[61]), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 98])),
                     "f"(S[62]), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 98 + 1])),
            "f"(S[63]), "r"((unsigned int)(snap_valid != 0))
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
                         reinterpret_cast<uint64_t>(&state[snap_base + 32])),
                     "f"(S2[2].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 32 + 1])),
                     "f"(S2[2].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 34])),
                     "f"(S2[3].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 34 + 1])),
                     "f"(S2[3].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 64])),
                     "f"(S2[4].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 64 + 1])),
                     "f"(S2[4].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 66])),
                     "f"(S2[5].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 66 + 1])),
                     "f"(S2[5].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 96])),
                     "f"(S2[6].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 96 + 1])),
                     "f"(S2[6].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 98])),
                     "f"(S2[7].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 98 + 1])),
                     "f"(S2[7].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2])),
                     "f"(S2[8].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 1])),
                     "f"(S2[8].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 2])),
                     "f"(S2[9].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 2 + 1])),
                     "f"(S2[9].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 32])),
                     "f"(S2[10].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 32 + 1])),
                     "f"(S2[10].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 34])),
                     "f"(S2[11].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 34 + 1])),
                     "f"(S2[11].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 64])),
                     "f"(S2[12].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 64 + 1])),
                     "f"(S2[12].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 66])),
                     "f"(S2[13].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 66 + 1])),
                     "f"(S2[13].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 96])),
                     "f"(S2[14].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 96 + 1])),
                     "f"(S2[14].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 98])),
                     "f"(S2[15].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 98 + 1])),
                     "f"(S2[15].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2])),
                     "f"(S2[16].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 1])),
                     "f"(S2[16].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 2])),
                     "f"(S2[17].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 2 + 1])),
            "f"(S2[17].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 32])),
                     "f"(S2[18].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 32 + 1])),
            "f"(S2[18].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 34])),
                     "f"(S2[19].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 34 + 1])),
            "f"(S2[19].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 64])),
                     "f"(S2[20].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 64 + 1])),
            "f"(S2[20].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 66])),
                     "f"(S2[21].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 66 + 1])),
            "f"(S2[21].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 96])),
                     "f"(S2[22].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 96 + 1])),
            "f"(S2[22].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 98])),
                     "f"(S2[23].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 2 * state_stride_p2 + 98 + 1])),
            "f"(S2[23].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2])),
                     "f"(S2[24].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 1])),
                     "f"(S2[24].y), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 2])),
                     "f"(S2[25].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 2 + 1])),
            "f"(S2[25].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 32])),
                     "f"(S2[26].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 32 + 1])),
            "f"(S2[26].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 34])),
                     "f"(S2[27].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 34 + 1])),
            "f"(S2[27].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 64])),
                     "f"(S2[28].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 64 + 1])),
            "f"(S2[28].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 66])),
                     "f"(S2[29].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 66 + 1])),
            "f"(S2[29].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 96])),
                     "f"(S2[30].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 96 + 1])),
            "f"(S2[30].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                         reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 98])),
                     "f"(S2[31].x), "r"((unsigned int)(snap_valid != 0))
                     : "memory");
        asm volatile(
            "{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" ::"l"(
                reinterpret_cast<uint64_t>(&state[snap_base + 3 * state_stride_p2 + 98 + 1])),
            "f"(S2[31].y), "r"((unsigned int)(snap_valid != 0))
            : "memory");
#endif
      }
    }
    if (use_ssm_state_indices == 0) {
#pragma unroll 1
      for (int t_1 = 0; t_1 < 8; t_1++) {
        float alphap = sAlpha[t_1];
#if __CUDA_ARCH__ == 900
        float alpha2p_x = alphap;
        float alpha2p_y = alphap;
        float _mul_96;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_96) : "f"(S[0]), "f"(alpha2p_x));
        float _mul_97;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_97) : "f"(S[1]), "f"(alpha2p_y));
        S[0] = _mul_96;
        S[1] = _mul_97;
        float _mul_98;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_98) : "f"(S[2]), "f"(alpha2p_x));
        float _mul_99;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_99) : "f"(S[3]), "f"(alpha2p_y));
        S[2] = _mul_98;
        S[3] = _mul_99;
        float _mul_100;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_100) : "f"(S[4]), "f"(alpha2p_x));
        float _mul_101;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_101) : "f"(S[5]), "f"(alpha2p_y));
        S[4] = _mul_100;
        S[5] = _mul_101;
        float _mul_102;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_102) : "f"(S[6]), "f"(alpha2p_x));
        float _mul_103;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_103) : "f"(S[7]), "f"(alpha2p_y));
        S[6] = _mul_102;
        S[7] = _mul_103;
        float _mul_104;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_104) : "f"(S[8]), "f"(alpha2p_x));
        float _mul_105;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_105) : "f"(S[9]), "f"(alpha2p_y));
        S[8] = _mul_104;
        S[9] = _mul_105;
        float _mul_106;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_106) : "f"(S[10]), "f"(alpha2p_x));
        float _mul_107;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_107) : "f"(S[11]), "f"(alpha2p_y));
        S[10] = _mul_106;
        S[11] = _mul_107;
        float _mul_108;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_108) : "f"(S[12]), "f"(alpha2p_x));
        float _mul_109;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_109) : "f"(S[13]), "f"(alpha2p_y));
        S[12] = _mul_108;
        S[13] = _mul_109;
        float _mul_110;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_110) : "f"(S[14]), "f"(alpha2p_x));
        float _mul_111;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_111) : "f"(S[15]), "f"(alpha2p_y));
        S[14] = _mul_110;
        S[15] = _mul_111;
        float _mul_112;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_112) : "f"(S[16]), "f"(alpha2p_x));
        float _mul_113;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_113) : "f"(S[17]), "f"(alpha2p_y));
        S[16] = _mul_112;
        S[17] = _mul_113;
        float _mul_114;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_114) : "f"(S[18]), "f"(alpha2p_x));
        float _mul_115;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_115) : "f"(S[19]), "f"(alpha2p_y));
        S[18] = _mul_114;
        S[19] = _mul_115;
        float _mul_116;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_116) : "f"(S[20]), "f"(alpha2p_x));
        float _mul_117;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_117) : "f"(S[21]), "f"(alpha2p_y));
        S[20] = _mul_116;
        S[21] = _mul_117;
        float _mul_118;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_118) : "f"(S[22]), "f"(alpha2p_x));
        float _mul_119;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_119) : "f"(S[23]), "f"(alpha2p_y));
        S[22] = _mul_118;
        S[23] = _mul_119;
        float _mul_120;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_120) : "f"(S[24]), "f"(alpha2p_x));
        float _mul_121;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_121) : "f"(S[25]), "f"(alpha2p_y));
        S[24] = _mul_120;
        S[25] = _mul_121;
        float _mul_122;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_122) : "f"(S[26]), "f"(alpha2p_x));
        float _mul_123;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_123) : "f"(S[27]), "f"(alpha2p_y));
        S[26] = _mul_122;
        S[27] = _mul_123;
        float _mul_124;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_124) : "f"(S[28]), "f"(alpha2p_x));
        float _mul_125;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_125) : "f"(S[29]), "f"(alpha2p_y));
        S[28] = _mul_124;
        S[29] = _mul_125;
        float _mul_126;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_126) : "f"(S[30]), "f"(alpha2p_x));
        float _mul_127;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_127) : "f"(S[31]), "f"(alpha2p_y));
        S[30] = _mul_126;
        S[31] = _mul_127;
        float _mul_128;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_128) : "f"(S[32]), "f"(alpha2p_x));
        float _mul_129;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_129) : "f"(S[33]), "f"(alpha2p_y));
        S[32] = _mul_128;
        S[33] = _mul_129;
        float _mul_130;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_130) : "f"(S[34]), "f"(alpha2p_x));
        float _mul_131;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_131) : "f"(S[35]), "f"(alpha2p_y));
        S[34] = _mul_130;
        S[35] = _mul_131;
        float _mul_132;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_132) : "f"(S[36]), "f"(alpha2p_x));
        float _mul_133;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_133) : "f"(S[37]), "f"(alpha2p_y));
        S[36] = _mul_132;
        S[37] = _mul_133;
        float _mul_134;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_134) : "f"(S[38]), "f"(alpha2p_x));
        float _mul_135;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_135) : "f"(S[39]), "f"(alpha2p_y));
        S[38] = _mul_134;
        S[39] = _mul_135;
        float _mul_136;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_136) : "f"(S[40]), "f"(alpha2p_x));
        float _mul_137;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_137) : "f"(S[41]), "f"(alpha2p_y));
        S[40] = _mul_136;
        S[41] = _mul_137;
        float _mul_138;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_138) : "f"(S[42]), "f"(alpha2p_x));
        float _mul_139;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_139) : "f"(S[43]), "f"(alpha2p_y));
        S[42] = _mul_138;
        S[43] = _mul_139;
        float _mul_140;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_140) : "f"(S[44]), "f"(alpha2p_x));
        float _mul_141;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_141) : "f"(S[45]), "f"(alpha2p_y));
        S[44] = _mul_140;
        S[45] = _mul_141;
        float _mul_142;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_142) : "f"(S[46]), "f"(alpha2p_x));
        float _mul_143;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_143) : "f"(S[47]), "f"(alpha2p_y));
        S[46] = _mul_142;
        S[47] = _mul_143;
        float _mul_144;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_144) : "f"(S[48]), "f"(alpha2p_x));
        float _mul_145;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_145) : "f"(S[49]), "f"(alpha2p_y));
        S[48] = _mul_144;
        S[49] = _mul_145;
        float _mul_146;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_146) : "f"(S[50]), "f"(alpha2p_x));
        float _mul_147;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_147) : "f"(S[51]), "f"(alpha2p_y));
        S[50] = _mul_146;
        S[51] = _mul_147;
        float _mul_148;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_148) : "f"(S[52]), "f"(alpha2p_x));
        float _mul_149;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_149) : "f"(S[53]), "f"(alpha2p_y));
        S[52] = _mul_148;
        S[53] = _mul_149;
        float _mul_150;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_150) : "f"(S[54]), "f"(alpha2p_x));
        float _mul_151;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_151) : "f"(S[55]), "f"(alpha2p_y));
        S[54] = _mul_150;
        S[55] = _mul_151;
        float _mul_152;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_152) : "f"(S[56]), "f"(alpha2p_x));
        float _mul_153;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_153) : "f"(S[57]), "f"(alpha2p_y));
        S[56] = _mul_152;
        S[57] = _mul_153;
        float _mul_154;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_154) : "f"(S[58]), "f"(alpha2p_x));
        float _mul_155;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_155) : "f"(S[59]), "f"(alpha2p_y));
        S[58] = _mul_154;
        S[59] = _mul_155;
        float _mul_156;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_156) : "f"(S[60]), "f"(alpha2p_x));
        float _mul_157;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_157) : "f"(S[61]), "f"(alpha2p_y));
        S[60] = _mul_156;
        S[61] = _mul_157;
        float _mul_158;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_158) : "f"(S[62]), "f"(alpha2p_x));
        float _mul_159;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_159) : "f"(S[63]), "f"(alpha2p_y));
        S[62] = _mul_158;
        S[63] = _mul_159;
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
#endif
        int mp0 = t_1 * 3;
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
                     : "r"(kbasep0 + 128));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                     : "r"(kbasep0 + 256));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                     : "r"(kbasep0 + 384));
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
        float _fma_480;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_480)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accp0_0_0_x));
        float _fma_481;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_481)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accp0_0_0_y));
        accp0_0_0_x = _fma_480;
        accp0_0_0_y = _fma_481;
        float _fma_482;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_482)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accp0_0_1_x));
        float _fma_483;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_483)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accp0_0_1_y));
        accp0_0_1_x = _fma_482;
        accp0_0_1_y = _fma_483;
        float _fma_484;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_484)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accp0_0_0_x));
        float _fma_485;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_485)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accp0_0_0_y));
        accp0_0_0_x = _fma_484;
        accp0_0_0_y = _fma_485;
        float _fma_486;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_486)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accp0_0_1_x));
        float _fma_487;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_487)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accp0_0_1_y));
        accp0_0_1_x = _fma_486;
        accp0_0_1_y = _fma_487;
        float _fma_488;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_488)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accp0_0_0_x));
        float _fma_489;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_489)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accp0_0_0_y));
        accp0_0_0_x = _fma_488;
        accp0_0_0_y = _fma_489;
        float _fma_490;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_490)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accp0_0_1_x));
        float _fma_491;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_491)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accp0_0_1_y));
        accp0_0_1_x = _fma_490;
        accp0_0_1_y = _fma_491;
        float _fma_492;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_492)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accp0_0_0_x));
        float _fma_493;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_493)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accp0_0_0_y));
        accp0_0_0_x = _fma_492;
        accp0_0_0_y = _fma_493;
        float _fma_494;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_494)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accp0_0_1_x));
        float _fma_495;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_495)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accp0_0_1_y));
        accp0_0_1_x = _fma_494;
        accp0_0_1_y = _fma_495;
        float _add_132;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_132) : "f"(accp0_0_0_x), "f"(accp0_0_1_x));
        float _add_133;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_133) : "f"(accp0_0_0_y), "f"(accp0_0_1_y));
        float dotp0_0 = _add_132 + _add_133;
        float _shfl_xor_sync_134 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 1);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_134;
        float _shfl_xor_sync_135 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 2);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_135;
        float _shfl_xor_sync_136 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 4);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_136;
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
        float dotp0_0 = accp0_0_0.x + accp0_0_0.y;
        float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 1);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_100;
        float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 2);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_101;
        float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 4);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_102;
#endif
        float vnp0_0 = (v_regs[0] - dotp0_0) * betap0;
#if __CUDA_ARCH__ == 900
        float vn2p0_0_x = vnp0_0;
        float vn2p0_0_y = vnp0_0;
        float _fma_496;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_496)
            : "f"(k_regs[0]), "f"(vn2p0_0_x), "f"(S[0]));
        float _fma_497;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_497)
            : "f"(k_regs[1]), "f"(vn2p0_0_y), "f"(S[1]));
        S[0] = _fma_496;
        S[1] = _fma_497;
        float _fma_498;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_498)
            : "f"(k_regs[2]), "f"(vn2p0_0_x), "f"(S[2]));
        float _fma_499;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_499)
            : "f"(k_regs[3]), "f"(vn2p0_0_y), "f"(S[3]));
        S[2] = _fma_498;
        S[3] = _fma_499;
        float _fma_500;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_500)
            : "f"(k_regs[4]), "f"(vn2p0_0_x), "f"(S[4]));
        float _fma_501;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_501)
            : "f"(k_regs[5]), "f"(vn2p0_0_y), "f"(S[5]));
        S[4] = _fma_500;
        S[5] = _fma_501;
        float _fma_502;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_502)
            : "f"(k_regs[6]), "f"(vn2p0_0_x), "f"(S[6]));
        float _fma_503;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_503)
            : "f"(k_regs[7]), "f"(vn2p0_0_y), "f"(S[7]));
        S[6] = _fma_502;
        S[7] = _fma_503;
        float _fma_504;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_504)
            : "f"(k_regs[8]), "f"(vn2p0_0_x), "f"(S[8]));
        float _fma_505;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_505)
            : "f"(k_regs[9]), "f"(vn2p0_0_y), "f"(S[9]));
        S[8] = _fma_504;
        S[9] = _fma_505;
        float _fma_506;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_506)
            : "f"(k_regs[10]), "f"(vn2p0_0_x), "f"(S[10]));
        float _fma_507;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_507)
            : "f"(k_regs[11]), "f"(vn2p0_0_y), "f"(S[11]));
        S[10] = _fma_506;
        S[11] = _fma_507;
        float _fma_508;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_508)
            : "f"(k_regs[12]), "f"(vn2p0_0_x), "f"(S[12]));
        float _fma_509;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_509)
            : "f"(k_regs[13]), "f"(vn2p0_0_y), "f"(S[13]));
        S[12] = _fma_508;
        S[13] = _fma_509;
        float _fma_510;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_510)
            : "f"(k_regs[14]), "f"(vn2p0_0_x), "f"(S[14]));
        float _fma_511;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_511)
            : "f"(k_regs[15]), "f"(vn2p0_0_y), "f"(S[15]));
        S[14] = _fma_510;
        S[15] = _fma_511;
        float accp0_1_0_x = 0.0f;
        float accp0_1_0_y = 0.0f;
        float accp0_1_1_x = 0.0f;
        float accp0_1_1_y = 0.0f;
        float _fma_512;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_512)
            : "f"(S[16]), "f"(k_regs[0]), "f"(accp0_1_0_x));
        float _fma_513;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_513)
            : "f"(S[17]), "f"(k_regs[1]), "f"(accp0_1_0_y));
        accp0_1_0_x = _fma_512;
        accp0_1_0_y = _fma_513;
        float _fma_514;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_514)
            : "f"(S[18]), "f"(k_regs[2]), "f"(accp0_1_1_x));
        float _fma_515;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_515)
            : "f"(S[19]), "f"(k_regs[3]), "f"(accp0_1_1_y));
        accp0_1_1_x = _fma_514;
        accp0_1_1_y = _fma_515;
        float _fma_516;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_516)
            : "f"(S[20]), "f"(k_regs[4]), "f"(accp0_1_0_x));
        float _fma_517;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_517)
            : "f"(S[21]), "f"(k_regs[5]), "f"(accp0_1_0_y));
        accp0_1_0_x = _fma_516;
        accp0_1_0_y = _fma_517;
        float _fma_518;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_518)
            : "f"(S[22]), "f"(k_regs[6]), "f"(accp0_1_1_x));
        float _fma_519;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_519)
            : "f"(S[23]), "f"(k_regs[7]), "f"(accp0_1_1_y));
        accp0_1_1_x = _fma_518;
        accp0_1_1_y = _fma_519;
        float _fma_520;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_520)
            : "f"(S[24]), "f"(k_regs[8]), "f"(accp0_1_0_x));
        float _fma_521;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_521)
            : "f"(S[25]), "f"(k_regs[9]), "f"(accp0_1_0_y));
        accp0_1_0_x = _fma_520;
        accp0_1_0_y = _fma_521;
        float _fma_522;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_522)
            : "f"(S[26]), "f"(k_regs[10]), "f"(accp0_1_1_x));
        float _fma_523;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_523)
            : "f"(S[27]), "f"(k_regs[11]), "f"(accp0_1_1_y));
        accp0_1_1_x = _fma_522;
        accp0_1_1_y = _fma_523;
        float _fma_524;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_524)
            : "f"(S[28]), "f"(k_regs[12]), "f"(accp0_1_0_x));
        float _fma_525;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_525)
            : "f"(S[29]), "f"(k_regs[13]), "f"(accp0_1_0_y));
        accp0_1_0_x = _fma_524;
        accp0_1_0_y = _fma_525;
        float _fma_526;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_526)
            : "f"(S[30]), "f"(k_regs[14]), "f"(accp0_1_1_x));
        float _fma_527;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_527)
            : "f"(S[31]), "f"(k_regs[15]), "f"(accp0_1_1_y));
        accp0_1_1_x = _fma_526;
        accp0_1_1_y = _fma_527;
        float _add_137;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_137) : "f"(accp0_1_0_x), "f"(accp0_1_1_x));
        float _add_138;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_138) : "f"(accp0_1_0_y), "f"(accp0_1_1_y));
        float dotp0_1 = _add_137 + _add_138;
        float _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 1);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_139;
        float _shfl_xor_sync_140 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 2);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_140;
        float _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 4);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_141;
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
        float2 accp0_1_0 = make_float2(0.0f, 0.0f);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[9], k2[1], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[11], k2[3], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[13], k2[5], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accp0_1_0);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[15], k2[7], accp0_1_0);
        float dotp0_1 = accp0_1_0.x + accp0_1_0.y;
        float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 1);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_103;
        float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 2);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_104;
        float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 4);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_105;
#endif
        float vnp0_1 = (v_regs[1] - dotp0_1) * betap0;
#if __CUDA_ARCH__ == 900
        float vn2p0_1_x = vnp0_1;
        float vn2p0_1_y = vnp0_1;
        float _fma_528;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_528)
            : "f"(k_regs[0]), "f"(vn2p0_1_x), "f"(S[16]));
        float _fma_529;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_529)
            : "f"(k_regs[1]), "f"(vn2p0_1_y), "f"(S[17]));
        S[16] = _fma_528;
        S[17] = _fma_529;
        float _fma_530;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_530)
            : "f"(k_regs[2]), "f"(vn2p0_1_x), "f"(S[18]));
        float _fma_531;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_531)
            : "f"(k_regs[3]), "f"(vn2p0_1_y), "f"(S[19]));
        S[18] = _fma_530;
        S[19] = _fma_531;
        float _fma_532;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_532)
            : "f"(k_regs[4]), "f"(vn2p0_1_x), "f"(S[20]));
        float _fma_533;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_533)
            : "f"(k_regs[5]), "f"(vn2p0_1_y), "f"(S[21]));
        S[20] = _fma_532;
        S[21] = _fma_533;
        float _fma_534;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_534)
            : "f"(k_regs[6]), "f"(vn2p0_1_x), "f"(S[22]));
        float _fma_535;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_535)
            : "f"(k_regs[7]), "f"(vn2p0_1_y), "f"(S[23]));
        S[22] = _fma_534;
        S[23] = _fma_535;
        float _fma_536;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_536)
            : "f"(k_regs[8]), "f"(vn2p0_1_x), "f"(S[24]));
        float _fma_537;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_537)
            : "f"(k_regs[9]), "f"(vn2p0_1_y), "f"(S[25]));
        S[24] = _fma_536;
        S[25] = _fma_537;
        float _fma_538;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_538)
            : "f"(k_regs[10]), "f"(vn2p0_1_x), "f"(S[26]));
        float _fma_539;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_539)
            : "f"(k_regs[11]), "f"(vn2p0_1_y), "f"(S[27]));
        S[26] = _fma_538;
        S[27] = _fma_539;
        float _fma_540;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_540)
            : "f"(k_regs[12]), "f"(vn2p0_1_x), "f"(S[28]));
        float _fma_541;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_541)
            : "f"(k_regs[13]), "f"(vn2p0_1_y), "f"(S[29]));
        S[28] = _fma_540;
        S[29] = _fma_541;
        float _fma_542;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_542)
            : "f"(k_regs[14]), "f"(vn2p0_1_x), "f"(S[30]));
        float _fma_543;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_543)
            : "f"(k_regs[15]), "f"(vn2p0_1_y), "f"(S[31]));
        S[30] = _fma_542;
        S[31] = _fma_543;
        float accp0_2_0_x = 0.0f;
        float accp0_2_0_y = 0.0f;
        float accp0_2_1_x = 0.0f;
        float accp0_2_1_y = 0.0f;
        float _fma_544;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_544)
            : "f"(S[32]), "f"(k_regs[0]), "f"(accp0_2_0_x));
        float _fma_545;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_545)
            : "f"(S[33]), "f"(k_regs[1]), "f"(accp0_2_0_y));
        accp0_2_0_x = _fma_544;
        accp0_2_0_y = _fma_545;
        float _fma_546;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_546)
            : "f"(S[34]), "f"(k_regs[2]), "f"(accp0_2_1_x));
        float _fma_547;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_547)
            : "f"(S[35]), "f"(k_regs[3]), "f"(accp0_2_1_y));
        accp0_2_1_x = _fma_546;
        accp0_2_1_y = _fma_547;
        float _fma_548;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_548)
            : "f"(S[36]), "f"(k_regs[4]), "f"(accp0_2_0_x));
        float _fma_549;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_549)
            : "f"(S[37]), "f"(k_regs[5]), "f"(accp0_2_0_y));
        accp0_2_0_x = _fma_548;
        accp0_2_0_y = _fma_549;
        float _fma_550;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_550)
            : "f"(S[38]), "f"(k_regs[6]), "f"(accp0_2_1_x));
        float _fma_551;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_551)
            : "f"(S[39]), "f"(k_regs[7]), "f"(accp0_2_1_y));
        accp0_2_1_x = _fma_550;
        accp0_2_1_y = _fma_551;
        float _fma_552;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_552)
            : "f"(S[40]), "f"(k_regs[8]), "f"(accp0_2_0_x));
        float _fma_553;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_553)
            : "f"(S[41]), "f"(k_regs[9]), "f"(accp0_2_0_y));
        accp0_2_0_x = _fma_552;
        accp0_2_0_y = _fma_553;
        float _fma_554;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_554)
            : "f"(S[42]), "f"(k_regs[10]), "f"(accp0_2_1_x));
        float _fma_555;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_555)
            : "f"(S[43]), "f"(k_regs[11]), "f"(accp0_2_1_y));
        accp0_2_1_x = _fma_554;
        accp0_2_1_y = _fma_555;
        float _fma_556;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_556)
            : "f"(S[44]), "f"(k_regs[12]), "f"(accp0_2_0_x));
        float _fma_557;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_557)
            : "f"(S[45]), "f"(k_regs[13]), "f"(accp0_2_0_y));
        accp0_2_0_x = _fma_556;
        accp0_2_0_y = _fma_557;
        float _fma_558;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_558)
            : "f"(S[46]), "f"(k_regs[14]), "f"(accp0_2_1_x));
        float _fma_559;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_559)
            : "f"(S[47]), "f"(k_regs[15]), "f"(accp0_2_1_y));
        accp0_2_1_x = _fma_558;
        accp0_2_1_y = _fma_559;
        float _add_142;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_142) : "f"(accp0_2_0_x), "f"(accp0_2_1_x));
        float _add_143;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_143) : "f"(accp0_2_0_y), "f"(accp0_2_1_y));
        float dotp0_2 = _add_142 + _add_143;
        float _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, dotp0_2, 1);
        dotp0_2 = dotp0_2 + _shfl_xor_sync_144;
        float _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, dotp0_2, 2);
        dotp0_2 = dotp0_2 + _shfl_xor_sync_145;
        float _shfl_xor_sync_146 = __shfl_xor_sync(0xFFFFFFFF, dotp0_2, 4);
        dotp0_2 = dotp0_2 + _shfl_xor_sync_146;
#else
        float2 vn2p0_1 = make_float2(vnp0_1, vnp0_1);
        S2[8] = fma_f32x2_rn_ftz(k2[0], vn2p0_1, S2[8]);
        S2[9] = fma_f32x2_rn_ftz(k2[1], vn2p0_1, S2[9]);
        S2[10] = fma_f32x2_rn_ftz(k2[2], vn2p0_1, S2[10]);
        S2[11] = fma_f32x2_rn_ftz(k2[3], vn2p0_1, S2[11]);
        S2[12] = fma_f32x2_rn_ftz(k2[4], vn2p0_1, S2[12]);
        S2[13] = fma_f32x2_rn_ftz(k2[5], vn2p0_1, S2[13]);
        S2[14] = fma_f32x2_rn_ftz(k2[6], vn2p0_1, S2[14]);
        S2[15] = fma_f32x2_rn_ftz(k2[7], vn2p0_1, S2[15]);
        float2 accp0_2_0 = make_float2(0.0f, 0.0f);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accp0_2_0);
        accp0_2_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accp0_2_0);
        float dotp0_2 = accp0_2_0.x + accp0_2_0.y;
        float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, dotp0_2, 1);
        dotp0_2 = dotp0_2 + _shfl_xor_sync_106;
        float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, dotp0_2, 2);
        dotp0_2 = dotp0_2 + _shfl_xor_sync_107;
        float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, dotp0_2, 4);
        dotp0_2 = dotp0_2 + _shfl_xor_sync_108;
#endif
        float vnp0_2 = (v_regs[2] - dotp0_2) * betap0;
#if __CUDA_ARCH__ == 900
        float vn2p0_2_x = vnp0_2;
        float vn2p0_2_y = vnp0_2;
        float _fma_560;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_560)
            : "f"(k_regs[0]), "f"(vn2p0_2_x), "f"(S[32]));
        float _fma_561;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_561)
            : "f"(k_regs[1]), "f"(vn2p0_2_y), "f"(S[33]));
        S[32] = _fma_560;
        S[33] = _fma_561;
        float _fma_562;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_562)
            : "f"(k_regs[2]), "f"(vn2p0_2_x), "f"(S[34]));
        float _fma_563;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_563)
            : "f"(k_regs[3]), "f"(vn2p0_2_y), "f"(S[35]));
        S[34] = _fma_562;
        S[35] = _fma_563;
        float _fma_564;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_564)
            : "f"(k_regs[4]), "f"(vn2p0_2_x), "f"(S[36]));
        float _fma_565;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_565)
            : "f"(k_regs[5]), "f"(vn2p0_2_y), "f"(S[37]));
        S[36] = _fma_564;
        S[37] = _fma_565;
        float _fma_566;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_566)
            : "f"(k_regs[6]), "f"(vn2p0_2_x), "f"(S[38]));
        float _fma_567;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_567)
            : "f"(k_regs[7]), "f"(vn2p0_2_y), "f"(S[39]));
        S[38] = _fma_566;
        S[39] = _fma_567;
        float _fma_568;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_568)
            : "f"(k_regs[8]), "f"(vn2p0_2_x), "f"(S[40]));
        float _fma_569;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_569)
            : "f"(k_regs[9]), "f"(vn2p0_2_y), "f"(S[41]));
        S[40] = _fma_568;
        S[41] = _fma_569;
        float _fma_570;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_570)
            : "f"(k_regs[10]), "f"(vn2p0_2_x), "f"(S[42]));
        float _fma_571;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_571)
            : "f"(k_regs[11]), "f"(vn2p0_2_y), "f"(S[43]));
        S[42] = _fma_570;
        S[43] = _fma_571;
        float _fma_572;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_572)
            : "f"(k_regs[12]), "f"(vn2p0_2_x), "f"(S[44]));
        float _fma_573;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_573)
            : "f"(k_regs[13]), "f"(vn2p0_2_y), "f"(S[45]));
        S[44] = _fma_572;
        S[45] = _fma_573;
        float _fma_574;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_574)
            : "f"(k_regs[14]), "f"(vn2p0_2_x), "f"(S[46]));
        float _fma_575;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_575)
            : "f"(k_regs[15]), "f"(vn2p0_2_y), "f"(S[47]));
        S[46] = _fma_574;
        S[47] = _fma_575;
        float accp0_3_0_x = 0.0f;
        float accp0_3_0_y = 0.0f;
        float accp0_3_1_x = 0.0f;
        float accp0_3_1_y = 0.0f;
        float _fma_576;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_576)
            : "f"(S[48]), "f"(k_regs[0]), "f"(accp0_3_0_x));
        float _fma_577;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_577)
            : "f"(S[49]), "f"(k_regs[1]), "f"(accp0_3_0_y));
        accp0_3_0_x = _fma_576;
        accp0_3_0_y = _fma_577;
        float _fma_578;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_578)
            : "f"(S[50]), "f"(k_regs[2]), "f"(accp0_3_1_x));
        float _fma_579;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_579)
            : "f"(S[51]), "f"(k_regs[3]), "f"(accp0_3_1_y));
        accp0_3_1_x = _fma_578;
        accp0_3_1_y = _fma_579;
        float _fma_580;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_580)
            : "f"(S[52]), "f"(k_regs[4]), "f"(accp0_3_0_x));
        float _fma_581;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_581)
            : "f"(S[53]), "f"(k_regs[5]), "f"(accp0_3_0_y));
        accp0_3_0_x = _fma_580;
        accp0_3_0_y = _fma_581;
        float _fma_582;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_582)
            : "f"(S[54]), "f"(k_regs[6]), "f"(accp0_3_1_x));
        float _fma_583;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_583)
            : "f"(S[55]), "f"(k_regs[7]), "f"(accp0_3_1_y));
        accp0_3_1_x = _fma_582;
        accp0_3_1_y = _fma_583;
        float _fma_584;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_584)
            : "f"(S[56]), "f"(k_regs[8]), "f"(accp0_3_0_x));
        float _fma_585;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_585)
            : "f"(S[57]), "f"(k_regs[9]), "f"(accp0_3_0_y));
        accp0_3_0_x = _fma_584;
        accp0_3_0_y = _fma_585;
        float _fma_586;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_586)
            : "f"(S[58]), "f"(k_regs[10]), "f"(accp0_3_1_x));
        float _fma_587;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_587)
            : "f"(S[59]), "f"(k_regs[11]), "f"(accp0_3_1_y));
        accp0_3_1_x = _fma_586;
        accp0_3_1_y = _fma_587;
        float _fma_588;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_588)
            : "f"(S[60]), "f"(k_regs[12]), "f"(accp0_3_0_x));
        float _fma_589;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_589)
            : "f"(S[61]), "f"(k_regs[13]), "f"(accp0_3_0_y));
        accp0_3_0_x = _fma_588;
        accp0_3_0_y = _fma_589;
        float _fma_590;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_590)
            : "f"(S[62]), "f"(k_regs[14]), "f"(accp0_3_1_x));
        float _fma_591;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_591)
            : "f"(S[63]), "f"(k_regs[15]), "f"(accp0_3_1_y));
        accp0_3_1_x = _fma_590;
        accp0_3_1_y = _fma_591;
        float _add_147;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_147) : "f"(accp0_3_0_x), "f"(accp0_3_1_x));
        float _add_148;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_148) : "f"(accp0_3_0_y), "f"(accp0_3_1_y));
        float dotp0_3 = _add_147 + _add_148;
        float _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, dotp0_3, 1);
        dotp0_3 = dotp0_3 + _shfl_xor_sync_149;
        float _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, dotp0_3, 2);
        dotp0_3 = dotp0_3 + _shfl_xor_sync_150;
        float _shfl_xor_sync_151 = __shfl_xor_sync(0xFFFFFFFF, dotp0_3, 4);
        dotp0_3 = dotp0_3 + _shfl_xor_sync_151;
#else
        float2 vn2p0_2 = make_float2(vnp0_2, vnp0_2);
        S2[16] = fma_f32x2_rn_ftz(k2[0], vn2p0_2, S2[16]);
        S2[17] = fma_f32x2_rn_ftz(k2[1], vn2p0_2, S2[17]);
        S2[18] = fma_f32x2_rn_ftz(k2[2], vn2p0_2, S2[18]);
        S2[19] = fma_f32x2_rn_ftz(k2[3], vn2p0_2, S2[19]);
        S2[20] = fma_f32x2_rn_ftz(k2[4], vn2p0_2, S2[20]);
        S2[21] = fma_f32x2_rn_ftz(k2[5], vn2p0_2, S2[21]);
        S2[22] = fma_f32x2_rn_ftz(k2[6], vn2p0_2, S2[22]);
        S2[23] = fma_f32x2_rn_ftz(k2[7], vn2p0_2, S2[23]);
        float2 accp0_3_0 = make_float2(0.0f, 0.0f);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[24], k2[0], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[25], k2[1], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[26], k2[2], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[27], k2[3], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[28], k2[4], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[29], k2[5], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[30], k2[6], accp0_3_0);
        accp0_3_0 = fma_f32x2_rn_ftz(S2[31], k2[7], accp0_3_0);
        float dotp0_3 = accp0_3_0.x + accp0_3_0.y;
        float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, dotp0_3, 1);
        dotp0_3 = dotp0_3 + _shfl_xor_sync_109;
        float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, dotp0_3, 2);
        dotp0_3 = dotp0_3 + _shfl_xor_sync_110;
        float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, dotp0_3, 4);
        dotp0_3 = dotp0_3 + _shfl_xor_sync_111;
#endif
        float vnp0_3 = (v_regs[3] - dotp0_3) * betap0;
#if __CUDA_ARCH__ == 900
        float vn2p0_3_x = vnp0_3;
        float vn2p0_3_y = vnp0_3;
        float _fma_592;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_592)
            : "f"(k_regs[0]), "f"(vn2p0_3_x), "f"(S[48]));
        float _fma_593;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_593)
            : "f"(k_regs[1]), "f"(vn2p0_3_y), "f"(S[49]));
        S[48] = _fma_592;
        S[49] = _fma_593;
        float _fma_594;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_594)
            : "f"(k_regs[2]), "f"(vn2p0_3_x), "f"(S[50]));
        float _fma_595;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_595)
            : "f"(k_regs[3]), "f"(vn2p0_3_y), "f"(S[51]));
        S[50] = _fma_594;
        S[51] = _fma_595;
        float _fma_596;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_596)
            : "f"(k_regs[4]), "f"(vn2p0_3_x), "f"(S[52]));
        float _fma_597;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_597)
            : "f"(k_regs[5]), "f"(vn2p0_3_y), "f"(S[53]));
        S[52] = _fma_596;
        S[53] = _fma_597;
        float _fma_598;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_598)
            : "f"(k_regs[6]), "f"(vn2p0_3_x), "f"(S[54]));
        float _fma_599;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_599)
            : "f"(k_regs[7]), "f"(vn2p0_3_y), "f"(S[55]));
        S[54] = _fma_598;
        S[55] = _fma_599;
        float _fma_600;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_600)
            : "f"(k_regs[8]), "f"(vn2p0_3_x), "f"(S[56]));
        float _fma_601;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_601)
            : "f"(k_regs[9]), "f"(vn2p0_3_y), "f"(S[57]));
        S[56] = _fma_600;
        S[57] = _fma_601;
        float _fma_602;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_602)
            : "f"(k_regs[10]), "f"(vn2p0_3_x), "f"(S[58]));
        float _fma_603;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_603)
            : "f"(k_regs[11]), "f"(vn2p0_3_y), "f"(S[59]));
        S[58] = _fma_602;
        S[59] = _fma_603;
        float _fma_604;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_604)
            : "f"(k_regs[12]), "f"(vn2p0_3_x), "f"(S[60]));
        float _fma_605;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_605)
            : "f"(k_regs[13]), "f"(vn2p0_3_y), "f"(S[61]));
        S[60] = _fma_604;
        S[61] = _fma_605;
        float _fma_606;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_606)
            : "f"(k_regs[14]), "f"(vn2p0_3_x), "f"(S[62]));
        float _fma_607;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_607)
            : "f"(k_regs[15]), "f"(vn2p0_3_y), "f"(S[63]));
        S[62] = _fma_606;
        S[63] = _fma_607;
#else
        float2 vn2p0_3 = make_float2(vnp0_3, vnp0_3);
        S2[24] = fma_f32x2_rn_ftz(k2[0], vn2p0_3, S2[24]);
        S2[25] = fma_f32x2_rn_ftz(k2[1], vn2p0_3, S2[25]);
        S2[26] = fma_f32x2_rn_ftz(k2[2], vn2p0_3, S2[26]);
        S2[27] = fma_f32x2_rn_ftz(k2[3], vn2p0_3, S2[27]);
        S2[28] = fma_f32x2_rn_ftz(k2[4], vn2p0_3, S2[28]);
        S2[29] = fma_f32x2_rn_ftz(k2[5], vn2p0_3, S2[29]);
        S2[30] = fma_f32x2_rn_ftz(k2[6], vn2p0_3, S2[30]);
        S2[31] = fma_f32x2_rn_ftz(k2[7], vn2p0_3, S2[31]);
#endif
        int mp1 = t_1 * 3 + 1;
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
                     : "r"(kbasep1 + 128));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                     : "r"(kbasep1 + 256));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                     : "r"(kbasep1 + 384));
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
        float _fma_608;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_608)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accp1_0_0_x));
        float _fma_609;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_609)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accp1_0_0_y));
        accp1_0_0_x = _fma_608;
        accp1_0_0_y = _fma_609;
        float _fma_610;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_610)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accp1_0_1_x));
        float _fma_611;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_611)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accp1_0_1_y));
        accp1_0_1_x = _fma_610;
        accp1_0_1_y = _fma_611;
        float _fma_612;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_612)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accp1_0_0_x));
        float _fma_613;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_613)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accp1_0_0_y));
        accp1_0_0_x = _fma_612;
        accp1_0_0_y = _fma_613;
        float _fma_614;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_614)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accp1_0_1_x));
        float _fma_615;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_615)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accp1_0_1_y));
        accp1_0_1_x = _fma_614;
        accp1_0_1_y = _fma_615;
        float _fma_616;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_616)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accp1_0_0_x));
        float _fma_617;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_617)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accp1_0_0_y));
        accp1_0_0_x = _fma_616;
        accp1_0_0_y = _fma_617;
        float _fma_618;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_618)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accp1_0_1_x));
        float _fma_619;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_619)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accp1_0_1_y));
        accp1_0_1_x = _fma_618;
        accp1_0_1_y = _fma_619;
        float _fma_620;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_620)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accp1_0_0_x));
        float _fma_621;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_621)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accp1_0_0_y));
        accp1_0_0_x = _fma_620;
        accp1_0_0_y = _fma_621;
        float _fma_622;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_622)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accp1_0_1_x));
        float _fma_623;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_623)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accp1_0_1_y));
        accp1_0_1_x = _fma_622;
        accp1_0_1_y = _fma_623;
        float _add_152;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_152) : "f"(accp1_0_0_x), "f"(accp1_0_1_x));
        float _add_153;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_153) : "f"(accp1_0_0_y), "f"(accp1_0_1_y));
        float dotp1_0 = _add_152 + _add_153;
        float _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 1);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_154;
        float _shfl_xor_sync_155 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 2);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_155;
        float _shfl_xor_sync_156 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 4);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_156;
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
        float dotp1_0 = accp1_0_0.x + accp1_0_0.y;
        float _shfl_xor_sync_112 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 1);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_112;
        float _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 2);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_113;
        float _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 4);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_114;
#endif
        float vnp1_0 = (v_regs[0] - dotp1_0) * betap1;
#if __CUDA_ARCH__ == 900
        float vn2p1_0_x = vnp1_0;
        float vn2p1_0_y = vnp1_0;
        float _fma_624;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_624)
            : "f"(k_regs[0]), "f"(vn2p1_0_x), "f"(S[0]));
        float _fma_625;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_625)
            : "f"(k_regs[1]), "f"(vn2p1_0_y), "f"(S[1]));
        S[0] = _fma_624;
        S[1] = _fma_625;
        float _fma_626;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_626)
            : "f"(k_regs[2]), "f"(vn2p1_0_x), "f"(S[2]));
        float _fma_627;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_627)
            : "f"(k_regs[3]), "f"(vn2p1_0_y), "f"(S[3]));
        S[2] = _fma_626;
        S[3] = _fma_627;
        float _fma_628;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_628)
            : "f"(k_regs[4]), "f"(vn2p1_0_x), "f"(S[4]));
        float _fma_629;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_629)
            : "f"(k_regs[5]), "f"(vn2p1_0_y), "f"(S[5]));
        S[4] = _fma_628;
        S[5] = _fma_629;
        float _fma_630;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_630)
            : "f"(k_regs[6]), "f"(vn2p1_0_x), "f"(S[6]));
        float _fma_631;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_631)
            : "f"(k_regs[7]), "f"(vn2p1_0_y), "f"(S[7]));
        S[6] = _fma_630;
        S[7] = _fma_631;
        float _fma_632;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_632)
            : "f"(k_regs[8]), "f"(vn2p1_0_x), "f"(S[8]));
        float _fma_633;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_633)
            : "f"(k_regs[9]), "f"(vn2p1_0_y), "f"(S[9]));
        S[8] = _fma_632;
        S[9] = _fma_633;
        float _fma_634;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_634)
            : "f"(k_regs[10]), "f"(vn2p1_0_x), "f"(S[10]));
        float _fma_635;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_635)
            : "f"(k_regs[11]), "f"(vn2p1_0_y), "f"(S[11]));
        S[10] = _fma_634;
        S[11] = _fma_635;
        float _fma_636;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_636)
            : "f"(k_regs[12]), "f"(vn2p1_0_x), "f"(S[12]));
        float _fma_637;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_637)
            : "f"(k_regs[13]), "f"(vn2p1_0_y), "f"(S[13]));
        S[12] = _fma_636;
        S[13] = _fma_637;
        float _fma_638;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_638)
            : "f"(k_regs[14]), "f"(vn2p1_0_x), "f"(S[14]));
        float _fma_639;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_639)
            : "f"(k_regs[15]), "f"(vn2p1_0_y), "f"(S[15]));
        S[14] = _fma_638;
        S[15] = _fma_639;
        float accp1_1_0_x = 0.0f;
        float accp1_1_0_y = 0.0f;
        float accp1_1_1_x = 0.0f;
        float accp1_1_1_y = 0.0f;
        float _fma_640;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_640)
            : "f"(S[16]), "f"(k_regs[0]), "f"(accp1_1_0_x));
        float _fma_641;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_641)
            : "f"(S[17]), "f"(k_regs[1]), "f"(accp1_1_0_y));
        accp1_1_0_x = _fma_640;
        accp1_1_0_y = _fma_641;
        float _fma_642;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_642)
            : "f"(S[18]), "f"(k_regs[2]), "f"(accp1_1_1_x));
        float _fma_643;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_643)
            : "f"(S[19]), "f"(k_regs[3]), "f"(accp1_1_1_y));
        accp1_1_1_x = _fma_642;
        accp1_1_1_y = _fma_643;
        float _fma_644;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_644)
            : "f"(S[20]), "f"(k_regs[4]), "f"(accp1_1_0_x));
        float _fma_645;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_645)
            : "f"(S[21]), "f"(k_regs[5]), "f"(accp1_1_0_y));
        accp1_1_0_x = _fma_644;
        accp1_1_0_y = _fma_645;
        float _fma_646;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_646)
            : "f"(S[22]), "f"(k_regs[6]), "f"(accp1_1_1_x));
        float _fma_647;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_647)
            : "f"(S[23]), "f"(k_regs[7]), "f"(accp1_1_1_y));
        accp1_1_1_x = _fma_646;
        accp1_1_1_y = _fma_647;
        float _fma_648;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_648)
            : "f"(S[24]), "f"(k_regs[8]), "f"(accp1_1_0_x));
        float _fma_649;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_649)
            : "f"(S[25]), "f"(k_regs[9]), "f"(accp1_1_0_y));
        accp1_1_0_x = _fma_648;
        accp1_1_0_y = _fma_649;
        float _fma_650;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_650)
            : "f"(S[26]), "f"(k_regs[10]), "f"(accp1_1_1_x));
        float _fma_651;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_651)
            : "f"(S[27]), "f"(k_regs[11]), "f"(accp1_1_1_y));
        accp1_1_1_x = _fma_650;
        accp1_1_1_y = _fma_651;
        float _fma_652;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_652)
            : "f"(S[28]), "f"(k_regs[12]), "f"(accp1_1_0_x));
        float _fma_653;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_653)
            : "f"(S[29]), "f"(k_regs[13]), "f"(accp1_1_0_y));
        accp1_1_0_x = _fma_652;
        accp1_1_0_y = _fma_653;
        float _fma_654;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_654)
            : "f"(S[30]), "f"(k_regs[14]), "f"(accp1_1_1_x));
        float _fma_655;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_655)
            : "f"(S[31]), "f"(k_regs[15]), "f"(accp1_1_1_y));
        accp1_1_1_x = _fma_654;
        accp1_1_1_y = _fma_655;
        float _add_157;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_157) : "f"(accp1_1_0_x), "f"(accp1_1_1_x));
        float _add_158;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_158) : "f"(accp1_1_0_y), "f"(accp1_1_1_y));
        float dotp1_1 = _add_157 + _add_158;
        float _shfl_xor_sync_159 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 1);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_159;
        float _shfl_xor_sync_160 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 2);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_160;
        float _shfl_xor_sync_161 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 4);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_161;
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
        float2 accp1_1_0 = make_float2(0.0f, 0.0f);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[9], k2[1], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[11], k2[3], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[13], k2[5], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accp1_1_0);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[15], k2[7], accp1_1_0);
        float dotp1_1 = accp1_1_0.x + accp1_1_0.y;
        float _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 1);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_115;
        float _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 2);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_116;
        float _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 4);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_117;
#endif
        float vnp1_1 = (v_regs[1] - dotp1_1) * betap1;
#if __CUDA_ARCH__ == 900
        float vn2p1_1_x = vnp1_1;
        float vn2p1_1_y = vnp1_1;
        float _fma_656;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_656)
            : "f"(k_regs[0]), "f"(vn2p1_1_x), "f"(S[16]));
        float _fma_657;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_657)
            : "f"(k_regs[1]), "f"(vn2p1_1_y), "f"(S[17]));
        S[16] = _fma_656;
        S[17] = _fma_657;
        float _fma_658;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_658)
            : "f"(k_regs[2]), "f"(vn2p1_1_x), "f"(S[18]));
        float _fma_659;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_659)
            : "f"(k_regs[3]), "f"(vn2p1_1_y), "f"(S[19]));
        S[18] = _fma_658;
        S[19] = _fma_659;
        float _fma_660;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_660)
            : "f"(k_regs[4]), "f"(vn2p1_1_x), "f"(S[20]));
        float _fma_661;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_661)
            : "f"(k_regs[5]), "f"(vn2p1_1_y), "f"(S[21]));
        S[20] = _fma_660;
        S[21] = _fma_661;
        float _fma_662;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_662)
            : "f"(k_regs[6]), "f"(vn2p1_1_x), "f"(S[22]));
        float _fma_663;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_663)
            : "f"(k_regs[7]), "f"(vn2p1_1_y), "f"(S[23]));
        S[22] = _fma_662;
        S[23] = _fma_663;
        float _fma_664;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_664)
            : "f"(k_regs[8]), "f"(vn2p1_1_x), "f"(S[24]));
        float _fma_665;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_665)
            : "f"(k_regs[9]), "f"(vn2p1_1_y), "f"(S[25]));
        S[24] = _fma_664;
        S[25] = _fma_665;
        float _fma_666;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_666)
            : "f"(k_regs[10]), "f"(vn2p1_1_x), "f"(S[26]));
        float _fma_667;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_667)
            : "f"(k_regs[11]), "f"(vn2p1_1_y), "f"(S[27]));
        S[26] = _fma_666;
        S[27] = _fma_667;
        float _fma_668;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_668)
            : "f"(k_regs[12]), "f"(vn2p1_1_x), "f"(S[28]));
        float _fma_669;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_669)
            : "f"(k_regs[13]), "f"(vn2p1_1_y), "f"(S[29]));
        S[28] = _fma_668;
        S[29] = _fma_669;
        float _fma_670;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_670)
            : "f"(k_regs[14]), "f"(vn2p1_1_x), "f"(S[30]));
        float _fma_671;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_671)
            : "f"(k_regs[15]), "f"(vn2p1_1_y), "f"(S[31]));
        S[30] = _fma_670;
        S[31] = _fma_671;
        float accp1_2_0_x = 0.0f;
        float accp1_2_0_y = 0.0f;
        float accp1_2_1_x = 0.0f;
        float accp1_2_1_y = 0.0f;
        float _fma_672;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_672)
            : "f"(S[32]), "f"(k_regs[0]), "f"(accp1_2_0_x));
        float _fma_673;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_673)
            : "f"(S[33]), "f"(k_regs[1]), "f"(accp1_2_0_y));
        accp1_2_0_x = _fma_672;
        accp1_2_0_y = _fma_673;
        float _fma_674;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_674)
            : "f"(S[34]), "f"(k_regs[2]), "f"(accp1_2_1_x));
        float _fma_675;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_675)
            : "f"(S[35]), "f"(k_regs[3]), "f"(accp1_2_1_y));
        accp1_2_1_x = _fma_674;
        accp1_2_1_y = _fma_675;
        float _fma_676;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_676)
            : "f"(S[36]), "f"(k_regs[4]), "f"(accp1_2_0_x));
        float _fma_677;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_677)
            : "f"(S[37]), "f"(k_regs[5]), "f"(accp1_2_0_y));
        accp1_2_0_x = _fma_676;
        accp1_2_0_y = _fma_677;
        float _fma_678;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_678)
            : "f"(S[38]), "f"(k_regs[6]), "f"(accp1_2_1_x));
        float _fma_679;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_679)
            : "f"(S[39]), "f"(k_regs[7]), "f"(accp1_2_1_y));
        accp1_2_1_x = _fma_678;
        accp1_2_1_y = _fma_679;
        float _fma_680;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_680)
            : "f"(S[40]), "f"(k_regs[8]), "f"(accp1_2_0_x));
        float _fma_681;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_681)
            : "f"(S[41]), "f"(k_regs[9]), "f"(accp1_2_0_y));
        accp1_2_0_x = _fma_680;
        accp1_2_0_y = _fma_681;
        float _fma_682;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_682)
            : "f"(S[42]), "f"(k_regs[10]), "f"(accp1_2_1_x));
        float _fma_683;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_683)
            : "f"(S[43]), "f"(k_regs[11]), "f"(accp1_2_1_y));
        accp1_2_1_x = _fma_682;
        accp1_2_1_y = _fma_683;
        float _fma_684;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_684)
            : "f"(S[44]), "f"(k_regs[12]), "f"(accp1_2_0_x));
        float _fma_685;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_685)
            : "f"(S[45]), "f"(k_regs[13]), "f"(accp1_2_0_y));
        accp1_2_0_x = _fma_684;
        accp1_2_0_y = _fma_685;
        float _fma_686;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_686)
            : "f"(S[46]), "f"(k_regs[14]), "f"(accp1_2_1_x));
        float _fma_687;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_687)
            : "f"(S[47]), "f"(k_regs[15]), "f"(accp1_2_1_y));
        accp1_2_1_x = _fma_686;
        accp1_2_1_y = _fma_687;
        float _add_162;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_162) : "f"(accp1_2_0_x), "f"(accp1_2_1_x));
        float _add_163;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_163) : "f"(accp1_2_0_y), "f"(accp1_2_1_y));
        float dotp1_2 = _add_162 + _add_163;
        float _shfl_xor_sync_164 = __shfl_xor_sync(0xFFFFFFFF, dotp1_2, 1);
        dotp1_2 = dotp1_2 + _shfl_xor_sync_164;
        float _shfl_xor_sync_165 = __shfl_xor_sync(0xFFFFFFFF, dotp1_2, 2);
        dotp1_2 = dotp1_2 + _shfl_xor_sync_165;
        float _shfl_xor_sync_166 = __shfl_xor_sync(0xFFFFFFFF, dotp1_2, 4);
        dotp1_2 = dotp1_2 + _shfl_xor_sync_166;
#else
        float2 vn2p1_1 = make_float2(vnp1_1, vnp1_1);
        S2[8] = fma_f32x2_rn_ftz(k2[0], vn2p1_1, S2[8]);
        S2[9] = fma_f32x2_rn_ftz(k2[1], vn2p1_1, S2[9]);
        S2[10] = fma_f32x2_rn_ftz(k2[2], vn2p1_1, S2[10]);
        S2[11] = fma_f32x2_rn_ftz(k2[3], vn2p1_1, S2[11]);
        S2[12] = fma_f32x2_rn_ftz(k2[4], vn2p1_1, S2[12]);
        S2[13] = fma_f32x2_rn_ftz(k2[5], vn2p1_1, S2[13]);
        S2[14] = fma_f32x2_rn_ftz(k2[6], vn2p1_1, S2[14]);
        S2[15] = fma_f32x2_rn_ftz(k2[7], vn2p1_1, S2[15]);
        float2 accp1_2_0 = make_float2(0.0f, 0.0f);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accp1_2_0);
        accp1_2_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accp1_2_0);
        float dotp1_2 = accp1_2_0.x + accp1_2_0.y;
        float _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, dotp1_2, 1);
        dotp1_2 = dotp1_2 + _shfl_xor_sync_118;
        float _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, dotp1_2, 2);
        dotp1_2 = dotp1_2 + _shfl_xor_sync_119;
        float _shfl_xor_sync_120 = __shfl_xor_sync(0xFFFFFFFF, dotp1_2, 4);
        dotp1_2 = dotp1_2 + _shfl_xor_sync_120;
#endif
        float vnp1_2 = (v_regs[2] - dotp1_2) * betap1;
#if __CUDA_ARCH__ == 900
        float vn2p1_2_x = vnp1_2;
        float vn2p1_2_y = vnp1_2;
        float _fma_688;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_688)
            : "f"(k_regs[0]), "f"(vn2p1_2_x), "f"(S[32]));
        float _fma_689;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_689)
            : "f"(k_regs[1]), "f"(vn2p1_2_y), "f"(S[33]));
        S[32] = _fma_688;
        S[33] = _fma_689;
        float _fma_690;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_690)
            : "f"(k_regs[2]), "f"(vn2p1_2_x), "f"(S[34]));
        float _fma_691;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_691)
            : "f"(k_regs[3]), "f"(vn2p1_2_y), "f"(S[35]));
        S[34] = _fma_690;
        S[35] = _fma_691;
        float _fma_692;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_692)
            : "f"(k_regs[4]), "f"(vn2p1_2_x), "f"(S[36]));
        float _fma_693;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_693)
            : "f"(k_regs[5]), "f"(vn2p1_2_y), "f"(S[37]));
        S[36] = _fma_692;
        S[37] = _fma_693;
        float _fma_694;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_694)
            : "f"(k_regs[6]), "f"(vn2p1_2_x), "f"(S[38]));
        float _fma_695;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_695)
            : "f"(k_regs[7]), "f"(vn2p1_2_y), "f"(S[39]));
        S[38] = _fma_694;
        S[39] = _fma_695;
        float _fma_696;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_696)
            : "f"(k_regs[8]), "f"(vn2p1_2_x), "f"(S[40]));
        float _fma_697;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_697)
            : "f"(k_regs[9]), "f"(vn2p1_2_y), "f"(S[41]));
        S[40] = _fma_696;
        S[41] = _fma_697;
        float _fma_698;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_698)
            : "f"(k_regs[10]), "f"(vn2p1_2_x), "f"(S[42]));
        float _fma_699;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_699)
            : "f"(k_regs[11]), "f"(vn2p1_2_y), "f"(S[43]));
        S[42] = _fma_698;
        S[43] = _fma_699;
        float _fma_700;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_700)
            : "f"(k_regs[12]), "f"(vn2p1_2_x), "f"(S[44]));
        float _fma_701;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_701)
            : "f"(k_regs[13]), "f"(vn2p1_2_y), "f"(S[45]));
        S[44] = _fma_700;
        S[45] = _fma_701;
        float _fma_702;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_702)
            : "f"(k_regs[14]), "f"(vn2p1_2_x), "f"(S[46]));
        float _fma_703;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_703)
            : "f"(k_regs[15]), "f"(vn2p1_2_y), "f"(S[47]));
        S[46] = _fma_702;
        S[47] = _fma_703;
        float accp1_3_0_x = 0.0f;
        float accp1_3_0_y = 0.0f;
        float accp1_3_1_x = 0.0f;
        float accp1_3_1_y = 0.0f;
        float _fma_704;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_704)
            : "f"(S[48]), "f"(k_regs[0]), "f"(accp1_3_0_x));
        float _fma_705;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_705)
            : "f"(S[49]), "f"(k_regs[1]), "f"(accp1_3_0_y));
        accp1_3_0_x = _fma_704;
        accp1_3_0_y = _fma_705;
        float _fma_706;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_706)
            : "f"(S[50]), "f"(k_regs[2]), "f"(accp1_3_1_x));
        float _fma_707;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_707)
            : "f"(S[51]), "f"(k_regs[3]), "f"(accp1_3_1_y));
        accp1_3_1_x = _fma_706;
        accp1_3_1_y = _fma_707;
        float _fma_708;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_708)
            : "f"(S[52]), "f"(k_regs[4]), "f"(accp1_3_0_x));
        float _fma_709;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_709)
            : "f"(S[53]), "f"(k_regs[5]), "f"(accp1_3_0_y));
        accp1_3_0_x = _fma_708;
        accp1_3_0_y = _fma_709;
        float _fma_710;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_710)
            : "f"(S[54]), "f"(k_regs[6]), "f"(accp1_3_1_x));
        float _fma_711;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_711)
            : "f"(S[55]), "f"(k_regs[7]), "f"(accp1_3_1_y));
        accp1_3_1_x = _fma_710;
        accp1_3_1_y = _fma_711;
        float _fma_712;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_712)
            : "f"(S[56]), "f"(k_regs[8]), "f"(accp1_3_0_x));
        float _fma_713;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_713)
            : "f"(S[57]), "f"(k_regs[9]), "f"(accp1_3_0_y));
        accp1_3_0_x = _fma_712;
        accp1_3_0_y = _fma_713;
        float _fma_714;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_714)
            : "f"(S[58]), "f"(k_regs[10]), "f"(accp1_3_1_x));
        float _fma_715;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_715)
            : "f"(S[59]), "f"(k_regs[11]), "f"(accp1_3_1_y));
        accp1_3_1_x = _fma_714;
        accp1_3_1_y = _fma_715;
        float _fma_716;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_716)
            : "f"(S[60]), "f"(k_regs[12]), "f"(accp1_3_0_x));
        float _fma_717;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_717)
            : "f"(S[61]), "f"(k_regs[13]), "f"(accp1_3_0_y));
        accp1_3_0_x = _fma_716;
        accp1_3_0_y = _fma_717;
        float _fma_718;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_718)
            : "f"(S[62]), "f"(k_regs[14]), "f"(accp1_3_1_x));
        float _fma_719;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_719)
            : "f"(S[63]), "f"(k_regs[15]), "f"(accp1_3_1_y));
        accp1_3_1_x = _fma_718;
        accp1_3_1_y = _fma_719;
        float _add_167;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_167) : "f"(accp1_3_0_x), "f"(accp1_3_1_x));
        float _add_168;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_168) : "f"(accp1_3_0_y), "f"(accp1_3_1_y));
        float dotp1_3 = _add_167 + _add_168;
        float _shfl_xor_sync_169 = __shfl_xor_sync(0xFFFFFFFF, dotp1_3, 1);
        dotp1_3 = dotp1_3 + _shfl_xor_sync_169;
        float _shfl_xor_sync_170 = __shfl_xor_sync(0xFFFFFFFF, dotp1_3, 2);
        dotp1_3 = dotp1_3 + _shfl_xor_sync_170;
        float _shfl_xor_sync_171 = __shfl_xor_sync(0xFFFFFFFF, dotp1_3, 4);
        dotp1_3 = dotp1_3 + _shfl_xor_sync_171;
#else
        float2 vn2p1_2 = make_float2(vnp1_2, vnp1_2);
        S2[16] = fma_f32x2_rn_ftz(k2[0], vn2p1_2, S2[16]);
        S2[17] = fma_f32x2_rn_ftz(k2[1], vn2p1_2, S2[17]);
        S2[18] = fma_f32x2_rn_ftz(k2[2], vn2p1_2, S2[18]);
        S2[19] = fma_f32x2_rn_ftz(k2[3], vn2p1_2, S2[19]);
        S2[20] = fma_f32x2_rn_ftz(k2[4], vn2p1_2, S2[20]);
        S2[21] = fma_f32x2_rn_ftz(k2[5], vn2p1_2, S2[21]);
        S2[22] = fma_f32x2_rn_ftz(k2[6], vn2p1_2, S2[22]);
        S2[23] = fma_f32x2_rn_ftz(k2[7], vn2p1_2, S2[23]);
        float2 accp1_3_0 = make_float2(0.0f, 0.0f);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[24], k2[0], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[25], k2[1], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[26], k2[2], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[27], k2[3], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[28], k2[4], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[29], k2[5], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[30], k2[6], accp1_3_0);
        accp1_3_0 = fma_f32x2_rn_ftz(S2[31], k2[7], accp1_3_0);
        float dotp1_3 = accp1_3_0.x + accp1_3_0.y;
        float _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, dotp1_3, 1);
        dotp1_3 = dotp1_3 + _shfl_xor_sync_121;
        float _shfl_xor_sync_122 = __shfl_xor_sync(0xFFFFFFFF, dotp1_3, 2);
        dotp1_3 = dotp1_3 + _shfl_xor_sync_122;
        float _shfl_xor_sync_123 = __shfl_xor_sync(0xFFFFFFFF, dotp1_3, 4);
        dotp1_3 = dotp1_3 + _shfl_xor_sync_123;
#endif
        float vnp1_3 = (v_regs[3] - dotp1_3) * betap1;
#if __CUDA_ARCH__ == 900
        float vn2p1_3_x = vnp1_3;
        float vn2p1_3_y = vnp1_3;
        float _fma_720;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_720)
            : "f"(k_regs[0]), "f"(vn2p1_3_x), "f"(S[48]));
        float _fma_721;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_721)
            : "f"(k_regs[1]), "f"(vn2p1_3_y), "f"(S[49]));
        S[48] = _fma_720;
        S[49] = _fma_721;
        float _fma_722;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_722)
            : "f"(k_regs[2]), "f"(vn2p1_3_x), "f"(S[50]));
        float _fma_723;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_723)
            : "f"(k_regs[3]), "f"(vn2p1_3_y), "f"(S[51]));
        S[50] = _fma_722;
        S[51] = _fma_723;
        float _fma_724;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_724)
            : "f"(k_regs[4]), "f"(vn2p1_3_x), "f"(S[52]));
        float _fma_725;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_725)
            : "f"(k_regs[5]), "f"(vn2p1_3_y), "f"(S[53]));
        S[52] = _fma_724;
        S[53] = _fma_725;
        float _fma_726;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_726)
            : "f"(k_regs[6]), "f"(vn2p1_3_x), "f"(S[54]));
        float _fma_727;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_727)
            : "f"(k_regs[7]), "f"(vn2p1_3_y), "f"(S[55]));
        S[54] = _fma_726;
        S[55] = _fma_727;
        float _fma_728;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_728)
            : "f"(k_regs[8]), "f"(vn2p1_3_x), "f"(S[56]));
        float _fma_729;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_729)
            : "f"(k_regs[9]), "f"(vn2p1_3_y), "f"(S[57]));
        S[56] = _fma_728;
        S[57] = _fma_729;
        float _fma_730;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_730)
            : "f"(k_regs[10]), "f"(vn2p1_3_x), "f"(S[58]));
        float _fma_731;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_731)
            : "f"(k_regs[11]), "f"(vn2p1_3_y), "f"(S[59]));
        S[58] = _fma_730;
        S[59] = _fma_731;
        float _fma_732;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_732)
            : "f"(k_regs[12]), "f"(vn2p1_3_x), "f"(S[60]));
        float _fma_733;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_733)
            : "f"(k_regs[13]), "f"(vn2p1_3_y), "f"(S[61]));
        S[60] = _fma_732;
        S[61] = _fma_733;
        float _fma_734;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_734)
            : "f"(k_regs[14]), "f"(vn2p1_3_x), "f"(S[62]));
        float _fma_735;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_735)
            : "f"(k_regs[15]), "f"(vn2p1_3_y), "f"(S[63]));
        S[62] = _fma_734;
        S[63] = _fma_735;
#else
        float2 vn2p1_3 = make_float2(vnp1_3, vnp1_3);
        S2[24] = fma_f32x2_rn_ftz(k2[0], vn2p1_3, S2[24]);
        S2[25] = fma_f32x2_rn_ftz(k2[1], vn2p1_3, S2[25]);
        S2[26] = fma_f32x2_rn_ftz(k2[2], vn2p1_3, S2[26]);
        S2[27] = fma_f32x2_rn_ftz(k2[3], vn2p1_3, S2[27]);
        S2[28] = fma_f32x2_rn_ftz(k2[4], vn2p1_3, S2[28]);
        S2[29] = fma_f32x2_rn_ftz(k2[5], vn2p1_3, S2[29]);
        S2[30] = fma_f32x2_rn_ftz(k2[6], vn2p1_3, S2[30]);
        S2[31] = fma_f32x2_rn_ftz(k2[7], vn2p1_3, S2[31]);
#endif
        int mp2 = t_1 * 3 + 2;
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
                     : "r"(kbasep2 + 128));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                     : "r"(kbasep2 + 256));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                     : "r"(kbasep2 + 384));
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
        float _fma_736;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_736)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accp2_0_0_x));
        float _fma_737;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_737)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accp2_0_0_y));
        accp2_0_0_x = _fma_736;
        accp2_0_0_y = _fma_737;
        float _fma_738;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_738)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accp2_0_1_x));
        float _fma_739;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_739)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accp2_0_1_y));
        accp2_0_1_x = _fma_738;
        accp2_0_1_y = _fma_739;
        float _fma_740;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_740)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accp2_0_0_x));
        float _fma_741;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_741)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accp2_0_0_y));
        accp2_0_0_x = _fma_740;
        accp2_0_0_y = _fma_741;
        float _fma_742;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_742)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accp2_0_1_x));
        float _fma_743;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_743)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accp2_0_1_y));
        accp2_0_1_x = _fma_742;
        accp2_0_1_y = _fma_743;
        float _fma_744;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_744)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accp2_0_0_x));
        float _fma_745;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_745)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accp2_0_0_y));
        accp2_0_0_x = _fma_744;
        accp2_0_0_y = _fma_745;
        float _fma_746;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_746)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accp2_0_1_x));
        float _fma_747;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_747)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accp2_0_1_y));
        accp2_0_1_x = _fma_746;
        accp2_0_1_y = _fma_747;
        float _fma_748;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_748)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accp2_0_0_x));
        float _fma_749;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_749)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accp2_0_0_y));
        accp2_0_0_x = _fma_748;
        accp2_0_0_y = _fma_749;
        float _fma_750;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_750)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accp2_0_1_x));
        float _fma_751;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_751)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accp2_0_1_y));
        accp2_0_1_x = _fma_750;
        accp2_0_1_y = _fma_751;
        float _add_172;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_172) : "f"(accp2_0_0_x), "f"(accp2_0_1_x));
        float _add_173;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_173) : "f"(accp2_0_0_y), "f"(accp2_0_1_y));
        float dotp2_0 = _add_172 + _add_173;
        float _shfl_xor_sync_174 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 1);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_174;
        float _shfl_xor_sync_175 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 2);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_175;
        float _shfl_xor_sync_176 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 4);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_176;
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
        float dotp2_0 = accp2_0_0.x + accp2_0_0.y;
        float _shfl_xor_sync_124 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 1);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_124;
        float _shfl_xor_sync_125 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 2);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_125;
        float _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 4);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_126;
#endif
        float vnp2_0 = (v_regs[0] - dotp2_0) * betap2;
#if __CUDA_ARCH__ == 900
        float vn2p2_0_x = vnp2_0;
        float vn2p2_0_y = vnp2_0;
        float _fma_752;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_752)
            : "f"(k_regs[0]), "f"(vn2p2_0_x), "f"(S[0]));
        float _fma_753;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_753)
            : "f"(k_regs[1]), "f"(vn2p2_0_y), "f"(S[1]));
        S[0] = _fma_752;
        S[1] = _fma_753;
        float _fma_754;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_754)
            : "f"(k_regs[2]), "f"(vn2p2_0_x), "f"(S[2]));
        float _fma_755;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_755)
            : "f"(k_regs[3]), "f"(vn2p2_0_y), "f"(S[3]));
        S[2] = _fma_754;
        S[3] = _fma_755;
        float _fma_756;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_756)
            : "f"(k_regs[4]), "f"(vn2p2_0_x), "f"(S[4]));
        float _fma_757;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_757)
            : "f"(k_regs[5]), "f"(vn2p2_0_y), "f"(S[5]));
        S[4] = _fma_756;
        S[5] = _fma_757;
        float _fma_758;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_758)
            : "f"(k_regs[6]), "f"(vn2p2_0_x), "f"(S[6]));
        float _fma_759;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_759)
            : "f"(k_regs[7]), "f"(vn2p2_0_y), "f"(S[7]));
        S[6] = _fma_758;
        S[7] = _fma_759;
        float _fma_760;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_760)
            : "f"(k_regs[8]), "f"(vn2p2_0_x), "f"(S[8]));
        float _fma_761;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_761)
            : "f"(k_regs[9]), "f"(vn2p2_0_y), "f"(S[9]));
        S[8] = _fma_760;
        S[9] = _fma_761;
        float _fma_762;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_762)
            : "f"(k_regs[10]), "f"(vn2p2_0_x), "f"(S[10]));
        float _fma_763;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_763)
            : "f"(k_regs[11]), "f"(vn2p2_0_y), "f"(S[11]));
        S[10] = _fma_762;
        S[11] = _fma_763;
        float _fma_764;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_764)
            : "f"(k_regs[12]), "f"(vn2p2_0_x), "f"(S[12]));
        float _fma_765;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_765)
            : "f"(k_regs[13]), "f"(vn2p2_0_y), "f"(S[13]));
        S[12] = _fma_764;
        S[13] = _fma_765;
        float _fma_766;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_766)
            : "f"(k_regs[14]), "f"(vn2p2_0_x), "f"(S[14]));
        float _fma_767;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_767)
            : "f"(k_regs[15]), "f"(vn2p2_0_y), "f"(S[15]));
        S[14] = _fma_766;
        S[15] = _fma_767;
        float accp2_1_0_x = 0.0f;
        float accp2_1_0_y = 0.0f;
        float accp2_1_1_x = 0.0f;
        float accp2_1_1_y = 0.0f;
        float _fma_768;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_768)
            : "f"(S[16]), "f"(k_regs[0]), "f"(accp2_1_0_x));
        float _fma_769;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_769)
            : "f"(S[17]), "f"(k_regs[1]), "f"(accp2_1_0_y));
        accp2_1_0_x = _fma_768;
        accp2_1_0_y = _fma_769;
        float _fma_770;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_770)
            : "f"(S[18]), "f"(k_regs[2]), "f"(accp2_1_1_x));
        float _fma_771;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_771)
            : "f"(S[19]), "f"(k_regs[3]), "f"(accp2_1_1_y));
        accp2_1_1_x = _fma_770;
        accp2_1_1_y = _fma_771;
        float _fma_772;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_772)
            : "f"(S[20]), "f"(k_regs[4]), "f"(accp2_1_0_x));
        float _fma_773;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_773)
            : "f"(S[21]), "f"(k_regs[5]), "f"(accp2_1_0_y));
        accp2_1_0_x = _fma_772;
        accp2_1_0_y = _fma_773;
        float _fma_774;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_774)
            : "f"(S[22]), "f"(k_regs[6]), "f"(accp2_1_1_x));
        float _fma_775;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_775)
            : "f"(S[23]), "f"(k_regs[7]), "f"(accp2_1_1_y));
        accp2_1_1_x = _fma_774;
        accp2_1_1_y = _fma_775;
        float _fma_776;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_776)
            : "f"(S[24]), "f"(k_regs[8]), "f"(accp2_1_0_x));
        float _fma_777;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_777)
            : "f"(S[25]), "f"(k_regs[9]), "f"(accp2_1_0_y));
        accp2_1_0_x = _fma_776;
        accp2_1_0_y = _fma_777;
        float _fma_778;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_778)
            : "f"(S[26]), "f"(k_regs[10]), "f"(accp2_1_1_x));
        float _fma_779;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_779)
            : "f"(S[27]), "f"(k_regs[11]), "f"(accp2_1_1_y));
        accp2_1_1_x = _fma_778;
        accp2_1_1_y = _fma_779;
        float _fma_780;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_780)
            : "f"(S[28]), "f"(k_regs[12]), "f"(accp2_1_0_x));
        float _fma_781;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_781)
            : "f"(S[29]), "f"(k_regs[13]), "f"(accp2_1_0_y));
        accp2_1_0_x = _fma_780;
        accp2_1_0_y = _fma_781;
        float _fma_782;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_782)
            : "f"(S[30]), "f"(k_regs[14]), "f"(accp2_1_1_x));
        float _fma_783;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_783)
            : "f"(S[31]), "f"(k_regs[15]), "f"(accp2_1_1_y));
        accp2_1_1_x = _fma_782;
        accp2_1_1_y = _fma_783;
        float _add_177;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_177) : "f"(accp2_1_0_x), "f"(accp2_1_1_x));
        float _add_178;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_178) : "f"(accp2_1_0_y), "f"(accp2_1_1_y));
        float dotp2_1 = _add_177 + _add_178;
        float _shfl_xor_sync_179 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 1);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_179;
        float _shfl_xor_sync_180 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 2);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_180;
        float _shfl_xor_sync_181 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 4);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_181;
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
        float2 accp2_1_0 = make_float2(0.0f, 0.0f);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[9], k2[1], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[11], k2[3], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[13], k2[5], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accp2_1_0);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[15], k2[7], accp2_1_0);
        float dotp2_1 = accp2_1_0.x + accp2_1_0.y;
        float _shfl_xor_sync_127 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 1);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_127;
        float _shfl_xor_sync_128 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 2);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_128;
        float _shfl_xor_sync_129 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 4);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_129;
#endif
        float vnp2_1 = (v_regs[1] - dotp2_1) * betap2;
#if __CUDA_ARCH__ == 900
        float vn2p2_1_x = vnp2_1;
        float vn2p2_1_y = vnp2_1;
        float _fma_784;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_784)
            : "f"(k_regs[0]), "f"(vn2p2_1_x), "f"(S[16]));
        float _fma_785;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_785)
            : "f"(k_regs[1]), "f"(vn2p2_1_y), "f"(S[17]));
        S[16] = _fma_784;
        S[17] = _fma_785;
        float _fma_786;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_786)
            : "f"(k_regs[2]), "f"(vn2p2_1_x), "f"(S[18]));
        float _fma_787;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_787)
            : "f"(k_regs[3]), "f"(vn2p2_1_y), "f"(S[19]));
        S[18] = _fma_786;
        S[19] = _fma_787;
        float _fma_788;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_788)
            : "f"(k_regs[4]), "f"(vn2p2_1_x), "f"(S[20]));
        float _fma_789;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_789)
            : "f"(k_regs[5]), "f"(vn2p2_1_y), "f"(S[21]));
        S[20] = _fma_788;
        S[21] = _fma_789;
        float _fma_790;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_790)
            : "f"(k_regs[6]), "f"(vn2p2_1_x), "f"(S[22]));
        float _fma_791;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_791)
            : "f"(k_regs[7]), "f"(vn2p2_1_y), "f"(S[23]));
        S[22] = _fma_790;
        S[23] = _fma_791;
        float _fma_792;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_792)
            : "f"(k_regs[8]), "f"(vn2p2_1_x), "f"(S[24]));
        float _fma_793;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_793)
            : "f"(k_regs[9]), "f"(vn2p2_1_y), "f"(S[25]));
        S[24] = _fma_792;
        S[25] = _fma_793;
        float _fma_794;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_794)
            : "f"(k_regs[10]), "f"(vn2p2_1_x), "f"(S[26]));
        float _fma_795;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_795)
            : "f"(k_regs[11]), "f"(vn2p2_1_y), "f"(S[27]));
        S[26] = _fma_794;
        S[27] = _fma_795;
        float _fma_796;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_796)
            : "f"(k_regs[12]), "f"(vn2p2_1_x), "f"(S[28]));
        float _fma_797;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_797)
            : "f"(k_regs[13]), "f"(vn2p2_1_y), "f"(S[29]));
        S[28] = _fma_796;
        S[29] = _fma_797;
        float _fma_798;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_798)
            : "f"(k_regs[14]), "f"(vn2p2_1_x), "f"(S[30]));
        float _fma_799;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_799)
            : "f"(k_regs[15]), "f"(vn2p2_1_y), "f"(S[31]));
        S[30] = _fma_798;
        S[31] = _fma_799;
        float accp2_2_0_x = 0.0f;
        float accp2_2_0_y = 0.0f;
        float accp2_2_1_x = 0.0f;
        float accp2_2_1_y = 0.0f;
        float _fma_800;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_800)
            : "f"(S[32]), "f"(k_regs[0]), "f"(accp2_2_0_x));
        float _fma_801;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_801)
            : "f"(S[33]), "f"(k_regs[1]), "f"(accp2_2_0_y));
        accp2_2_0_x = _fma_800;
        accp2_2_0_y = _fma_801;
        float _fma_802;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_802)
            : "f"(S[34]), "f"(k_regs[2]), "f"(accp2_2_1_x));
        float _fma_803;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_803)
            : "f"(S[35]), "f"(k_regs[3]), "f"(accp2_2_1_y));
        accp2_2_1_x = _fma_802;
        accp2_2_1_y = _fma_803;
        float _fma_804;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_804)
            : "f"(S[36]), "f"(k_regs[4]), "f"(accp2_2_0_x));
        float _fma_805;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_805)
            : "f"(S[37]), "f"(k_regs[5]), "f"(accp2_2_0_y));
        accp2_2_0_x = _fma_804;
        accp2_2_0_y = _fma_805;
        float _fma_806;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_806)
            : "f"(S[38]), "f"(k_regs[6]), "f"(accp2_2_1_x));
        float _fma_807;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_807)
            : "f"(S[39]), "f"(k_regs[7]), "f"(accp2_2_1_y));
        accp2_2_1_x = _fma_806;
        accp2_2_1_y = _fma_807;
        float _fma_808;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_808)
            : "f"(S[40]), "f"(k_regs[8]), "f"(accp2_2_0_x));
        float _fma_809;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_809)
            : "f"(S[41]), "f"(k_regs[9]), "f"(accp2_2_0_y));
        accp2_2_0_x = _fma_808;
        accp2_2_0_y = _fma_809;
        float _fma_810;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_810)
            : "f"(S[42]), "f"(k_regs[10]), "f"(accp2_2_1_x));
        float _fma_811;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_811)
            : "f"(S[43]), "f"(k_regs[11]), "f"(accp2_2_1_y));
        accp2_2_1_x = _fma_810;
        accp2_2_1_y = _fma_811;
        float _fma_812;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_812)
            : "f"(S[44]), "f"(k_regs[12]), "f"(accp2_2_0_x));
        float _fma_813;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_813)
            : "f"(S[45]), "f"(k_regs[13]), "f"(accp2_2_0_y));
        accp2_2_0_x = _fma_812;
        accp2_2_0_y = _fma_813;
        float _fma_814;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_814)
            : "f"(S[46]), "f"(k_regs[14]), "f"(accp2_2_1_x));
        float _fma_815;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_815)
            : "f"(S[47]), "f"(k_regs[15]), "f"(accp2_2_1_y));
        accp2_2_1_x = _fma_814;
        accp2_2_1_y = _fma_815;
        float _add_182;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_182) : "f"(accp2_2_0_x), "f"(accp2_2_1_x));
        float _add_183;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_183) : "f"(accp2_2_0_y), "f"(accp2_2_1_y));
        float dotp2_2 = _add_182 + _add_183;
        float _shfl_xor_sync_184 = __shfl_xor_sync(0xFFFFFFFF, dotp2_2, 1);
        dotp2_2 = dotp2_2 + _shfl_xor_sync_184;
        float _shfl_xor_sync_185 = __shfl_xor_sync(0xFFFFFFFF, dotp2_2, 2);
        dotp2_2 = dotp2_2 + _shfl_xor_sync_185;
        float _shfl_xor_sync_186 = __shfl_xor_sync(0xFFFFFFFF, dotp2_2, 4);
        dotp2_2 = dotp2_2 + _shfl_xor_sync_186;
#else
        float2 vn2p2_1 = make_float2(vnp2_1, vnp2_1);
        S2[8] = fma_f32x2_rn_ftz(k2[0], vn2p2_1, S2[8]);
        S2[9] = fma_f32x2_rn_ftz(k2[1], vn2p2_1, S2[9]);
        S2[10] = fma_f32x2_rn_ftz(k2[2], vn2p2_1, S2[10]);
        S2[11] = fma_f32x2_rn_ftz(k2[3], vn2p2_1, S2[11]);
        S2[12] = fma_f32x2_rn_ftz(k2[4], vn2p2_1, S2[12]);
        S2[13] = fma_f32x2_rn_ftz(k2[5], vn2p2_1, S2[13]);
        S2[14] = fma_f32x2_rn_ftz(k2[6], vn2p2_1, S2[14]);
        S2[15] = fma_f32x2_rn_ftz(k2[7], vn2p2_1, S2[15]);
        float2 accp2_2_0 = make_float2(0.0f, 0.0f);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accp2_2_0);
        accp2_2_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accp2_2_0);
        float dotp2_2 = accp2_2_0.x + accp2_2_0.y;
        float _shfl_xor_sync_130 = __shfl_xor_sync(0xFFFFFFFF, dotp2_2, 1);
        dotp2_2 = dotp2_2 + _shfl_xor_sync_130;
        float _shfl_xor_sync_131 = __shfl_xor_sync(0xFFFFFFFF, dotp2_2, 2);
        dotp2_2 = dotp2_2 + _shfl_xor_sync_131;
        float _shfl_xor_sync_132 = __shfl_xor_sync(0xFFFFFFFF, dotp2_2, 4);
        dotp2_2 = dotp2_2 + _shfl_xor_sync_132;
#endif
        float vnp2_2 = (v_regs[2] - dotp2_2) * betap2;
#if __CUDA_ARCH__ == 900
        float vn2p2_2_x = vnp2_2;
        float vn2p2_2_y = vnp2_2;
        float _fma_816;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_816)
            : "f"(k_regs[0]), "f"(vn2p2_2_x), "f"(S[32]));
        float _fma_817;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_817)
            : "f"(k_regs[1]), "f"(vn2p2_2_y), "f"(S[33]));
        S[32] = _fma_816;
        S[33] = _fma_817;
        float _fma_818;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_818)
            : "f"(k_regs[2]), "f"(vn2p2_2_x), "f"(S[34]));
        float _fma_819;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_819)
            : "f"(k_regs[3]), "f"(vn2p2_2_y), "f"(S[35]));
        S[34] = _fma_818;
        S[35] = _fma_819;
        float _fma_820;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_820)
            : "f"(k_regs[4]), "f"(vn2p2_2_x), "f"(S[36]));
        float _fma_821;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_821)
            : "f"(k_regs[5]), "f"(vn2p2_2_y), "f"(S[37]));
        S[36] = _fma_820;
        S[37] = _fma_821;
        float _fma_822;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_822)
            : "f"(k_regs[6]), "f"(vn2p2_2_x), "f"(S[38]));
        float _fma_823;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_823)
            : "f"(k_regs[7]), "f"(vn2p2_2_y), "f"(S[39]));
        S[38] = _fma_822;
        S[39] = _fma_823;
        float _fma_824;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_824)
            : "f"(k_regs[8]), "f"(vn2p2_2_x), "f"(S[40]));
        float _fma_825;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_825)
            : "f"(k_regs[9]), "f"(vn2p2_2_y), "f"(S[41]));
        S[40] = _fma_824;
        S[41] = _fma_825;
        float _fma_826;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_826)
            : "f"(k_regs[10]), "f"(vn2p2_2_x), "f"(S[42]));
        float _fma_827;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_827)
            : "f"(k_regs[11]), "f"(vn2p2_2_y), "f"(S[43]));
        S[42] = _fma_826;
        S[43] = _fma_827;
        float _fma_828;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_828)
            : "f"(k_regs[12]), "f"(vn2p2_2_x), "f"(S[44]));
        float _fma_829;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_829)
            : "f"(k_regs[13]), "f"(vn2p2_2_y), "f"(S[45]));
        S[44] = _fma_828;
        S[45] = _fma_829;
        float _fma_830;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_830)
            : "f"(k_regs[14]), "f"(vn2p2_2_x), "f"(S[46]));
        float _fma_831;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_831)
            : "f"(k_regs[15]), "f"(vn2p2_2_y), "f"(S[47]));
        S[46] = _fma_830;
        S[47] = _fma_831;
        float accp2_3_0_x = 0.0f;
        float accp2_3_0_y = 0.0f;
        float accp2_3_1_x = 0.0f;
        float accp2_3_1_y = 0.0f;
        float _fma_832;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_832)
            : "f"(S[48]), "f"(k_regs[0]), "f"(accp2_3_0_x));
        float _fma_833;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_833)
            : "f"(S[49]), "f"(k_regs[1]), "f"(accp2_3_0_y));
        accp2_3_0_x = _fma_832;
        accp2_3_0_y = _fma_833;
        float _fma_834;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_834)
            : "f"(S[50]), "f"(k_regs[2]), "f"(accp2_3_1_x));
        float _fma_835;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_835)
            : "f"(S[51]), "f"(k_regs[3]), "f"(accp2_3_1_y));
        accp2_3_1_x = _fma_834;
        accp2_3_1_y = _fma_835;
        float _fma_836;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_836)
            : "f"(S[52]), "f"(k_regs[4]), "f"(accp2_3_0_x));
        float _fma_837;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_837)
            : "f"(S[53]), "f"(k_regs[5]), "f"(accp2_3_0_y));
        accp2_3_0_x = _fma_836;
        accp2_3_0_y = _fma_837;
        float _fma_838;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_838)
            : "f"(S[54]), "f"(k_regs[6]), "f"(accp2_3_1_x));
        float _fma_839;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_839)
            : "f"(S[55]), "f"(k_regs[7]), "f"(accp2_3_1_y));
        accp2_3_1_x = _fma_838;
        accp2_3_1_y = _fma_839;
        float _fma_840;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_840)
            : "f"(S[56]), "f"(k_regs[8]), "f"(accp2_3_0_x));
        float _fma_841;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_841)
            : "f"(S[57]), "f"(k_regs[9]), "f"(accp2_3_0_y));
        accp2_3_0_x = _fma_840;
        accp2_3_0_y = _fma_841;
        float _fma_842;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_842)
            : "f"(S[58]), "f"(k_regs[10]), "f"(accp2_3_1_x));
        float _fma_843;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_843)
            : "f"(S[59]), "f"(k_regs[11]), "f"(accp2_3_1_y));
        accp2_3_1_x = _fma_842;
        accp2_3_1_y = _fma_843;
        float _fma_844;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_844)
            : "f"(S[60]), "f"(k_regs[12]), "f"(accp2_3_0_x));
        float _fma_845;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_845)
            : "f"(S[61]), "f"(k_regs[13]), "f"(accp2_3_0_y));
        accp2_3_0_x = _fma_844;
        accp2_3_0_y = _fma_845;
        float _fma_846;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_846)
            : "f"(S[62]), "f"(k_regs[14]), "f"(accp2_3_1_x));
        float _fma_847;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_847)
            : "f"(S[63]), "f"(k_regs[15]), "f"(accp2_3_1_y));
        accp2_3_1_x = _fma_846;
        accp2_3_1_y = _fma_847;
        float _add_187;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_187) : "f"(accp2_3_0_x), "f"(accp2_3_1_x));
        float _add_188;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_188) : "f"(accp2_3_0_y), "f"(accp2_3_1_y));
        float dotp2_3 = _add_187 + _add_188;
        float _shfl_xor_sync_189 = __shfl_xor_sync(0xFFFFFFFF, dotp2_3, 1);
        dotp2_3 = dotp2_3 + _shfl_xor_sync_189;
        float _shfl_xor_sync_190 = __shfl_xor_sync(0xFFFFFFFF, dotp2_3, 2);
        dotp2_3 = dotp2_3 + _shfl_xor_sync_190;
        float _shfl_xor_sync_191 = __shfl_xor_sync(0xFFFFFFFF, dotp2_3, 4);
        dotp2_3 = dotp2_3 + _shfl_xor_sync_191;
#else
        float2 vn2p2_2 = make_float2(vnp2_2, vnp2_2);
        S2[16] = fma_f32x2_rn_ftz(k2[0], vn2p2_2, S2[16]);
        S2[17] = fma_f32x2_rn_ftz(k2[1], vn2p2_2, S2[17]);
        S2[18] = fma_f32x2_rn_ftz(k2[2], vn2p2_2, S2[18]);
        S2[19] = fma_f32x2_rn_ftz(k2[3], vn2p2_2, S2[19]);
        S2[20] = fma_f32x2_rn_ftz(k2[4], vn2p2_2, S2[20]);
        S2[21] = fma_f32x2_rn_ftz(k2[5], vn2p2_2, S2[21]);
        S2[22] = fma_f32x2_rn_ftz(k2[6], vn2p2_2, S2[22]);
        S2[23] = fma_f32x2_rn_ftz(k2[7], vn2p2_2, S2[23]);
        float2 accp2_3_0 = make_float2(0.0f, 0.0f);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[24], k2[0], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[25], k2[1], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[26], k2[2], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[27], k2[3], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[28], k2[4], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[29], k2[5], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[30], k2[6], accp2_3_0);
        accp2_3_0 = fma_f32x2_rn_ftz(S2[31], k2[7], accp2_3_0);
        float dotp2_3 = accp2_3_0.x + accp2_3_0.y;
        float _shfl_xor_sync_133 = __shfl_xor_sync(0xFFFFFFFF, dotp2_3, 1);
        dotp2_3 = dotp2_3 + _shfl_xor_sync_133;
        float _shfl_xor_sync_134 = __shfl_xor_sync(0xFFFFFFFF, dotp2_3, 2);
        dotp2_3 = dotp2_3 + _shfl_xor_sync_134;
        float _shfl_xor_sync_135 = __shfl_xor_sync(0xFFFFFFFF, dotp2_3, 4);
        dotp2_3 = dotp2_3 + _shfl_xor_sync_135;
#endif
        float vnp2_3 = (v_regs[3] - dotp2_3) * betap2;
#if __CUDA_ARCH__ == 900
        float vn2p2_3_x = vnp2_3;
        float vn2p2_3_y = vnp2_3;
        float _fma_848;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_848)
            : "f"(k_regs[0]), "f"(vn2p2_3_x), "f"(S[48]));
        float _fma_849;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_849)
            : "f"(k_regs[1]), "f"(vn2p2_3_y), "f"(S[49]));
        S[48] = _fma_848;
        S[49] = _fma_849;
        float _fma_850;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_850)
            : "f"(k_regs[2]), "f"(vn2p2_3_x), "f"(S[50]));
        float _fma_851;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_851)
            : "f"(k_regs[3]), "f"(vn2p2_3_y), "f"(S[51]));
        S[50] = _fma_850;
        S[51] = _fma_851;
        float _fma_852;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_852)
            : "f"(k_regs[4]), "f"(vn2p2_3_x), "f"(S[52]));
        float _fma_853;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_853)
            : "f"(k_regs[5]), "f"(vn2p2_3_y), "f"(S[53]));
        S[52] = _fma_852;
        S[53] = _fma_853;
        float _fma_854;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_854)
            : "f"(k_regs[6]), "f"(vn2p2_3_x), "f"(S[54]));
        float _fma_855;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_855)
            : "f"(k_regs[7]), "f"(vn2p2_3_y), "f"(S[55]));
        S[54] = _fma_854;
        S[55] = _fma_855;
        float _fma_856;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_856)
            : "f"(k_regs[8]), "f"(vn2p2_3_x), "f"(S[56]));
        float _fma_857;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_857)
            : "f"(k_regs[9]), "f"(vn2p2_3_y), "f"(S[57]));
        S[56] = _fma_856;
        S[57] = _fma_857;
        float _fma_858;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_858)
            : "f"(k_regs[10]), "f"(vn2p2_3_x), "f"(S[58]));
        float _fma_859;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_859)
            : "f"(k_regs[11]), "f"(vn2p2_3_y), "f"(S[59]));
        S[58] = _fma_858;
        S[59] = _fma_859;
        float _fma_860;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_860)
            : "f"(k_regs[12]), "f"(vn2p2_3_x), "f"(S[60]));
        float _fma_861;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_861)
            : "f"(k_regs[13]), "f"(vn2p2_3_y), "f"(S[61]));
        S[60] = _fma_860;
        S[61] = _fma_861;
        float _fma_862;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_862)
            : "f"(k_regs[14]), "f"(vn2p2_3_x), "f"(S[62]));
        float _fma_863;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_863)
            : "f"(k_regs[15]), "f"(vn2p2_3_y), "f"(S[63]));
        S[62] = _fma_862;
        S[63] = _fma_863;
#else
        float2 vn2p2_3 = make_float2(vnp2_3, vnp2_3);
        S2[24] = fma_f32x2_rn_ftz(k2[0], vn2p2_3, S2[24]);
        S2[25] = fma_f32x2_rn_ftz(k2[1], vn2p2_3, S2[25]);
        S2[26] = fma_f32x2_rn_ftz(k2[2], vn2p2_3, S2[26]);
        S2[27] = fma_f32x2_rn_ftz(k2[3], vn2p2_3, S2[27]);
        S2[28] = fma_f32x2_rn_ftz(k2[4], vn2p2_3, S2[28]);
        S2[29] = fma_f32x2_rn_ftz(k2[5], vn2p2_3, S2[29]);
        S2[30] = fma_f32x2_rn_ftz(k2[6], vn2p2_3, S2[30]);
        S2[31] = fma_f32x2_rn_ftz(k2[7], vn2p2_3, S2[31]);
#endif
        unsigned int qbasep = sQ_addr + (unsigned int)(t_1 * 512) + lane_byte;
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
                     : "r"(qbasep + 128));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                     : "r"(qbasep + 256));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                       "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                     : "r"(qbasep + 384));
#if __CUDA_ARCH__ == 900
        float accpq_0_0_x = 0.0f;
        float accpq_0_0_y = 0.0f;
        float accpq_0_1_x = 0.0f;
        float accpq_0_1_y = 0.0f;
        float _fma_864;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_864)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accpq_0_0_x));
        float _fma_865;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_865)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accpq_0_0_y));
        accpq_0_0_x = _fma_864;
        accpq_0_0_y = _fma_865;
        float _fma_866;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_866)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accpq_0_1_x));
        float _fma_867;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_867)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accpq_0_1_y));
        accpq_0_1_x = _fma_866;
        accpq_0_1_y = _fma_867;
        float _fma_868;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_868)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accpq_0_0_x));
        float _fma_869;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_869)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accpq_0_0_y));
        accpq_0_0_x = _fma_868;
        accpq_0_0_y = _fma_869;
        float _fma_870;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_870)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accpq_0_1_x));
        float _fma_871;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_871)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accpq_0_1_y));
        accpq_0_1_x = _fma_870;
        accpq_0_1_y = _fma_871;
        float _fma_872;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_872)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accpq_0_0_x));
        float _fma_873;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_873)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accpq_0_0_y));
        accpq_0_0_x = _fma_872;
        accpq_0_0_y = _fma_873;
        float _fma_874;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_874)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accpq_0_1_x));
        float _fma_875;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_875)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accpq_0_1_y));
        accpq_0_1_x = _fma_874;
        accpq_0_1_y = _fma_875;
        float _fma_876;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_876)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accpq_0_0_x));
        float _fma_877;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_877)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accpq_0_0_y));
        accpq_0_0_x = _fma_876;
        accpq_0_0_y = _fma_877;
        float _fma_878;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_878)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accpq_0_1_x));
        float _fma_879;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_879)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accpq_0_1_y));
        accpq_0_1_x = _fma_878;
        accpq_0_1_y = _fma_879;
        float _add_192;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_192) : "f"(accpq_0_0_x), "f"(accpq_0_1_x));
        float _add_193;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_193) : "f"(accpq_0_0_y), "f"(accpq_0_1_y));
        float dotpq_0 = _add_192 + _add_193;
        float _shfl_xor_sync_194 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 1);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_194;
        float _shfl_xor_sync_195 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 2);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_195;
        float _shfl_xor_sync_196 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 4);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_196;
#else
        k2[0] = make_float2(k_regs[0], k_regs[1]);
        k2[1] = make_float2(k_regs[2], k_regs[3]);
        k2[2] = make_float2(k_regs[4], k_regs[5]);
        k2[3] = make_float2(k_regs[6], k_regs[7]);
        k2[4] = make_float2(k_regs[8], k_regs[9]);
        k2[5] = make_float2(k_regs[10], k_regs[11]);
        k2[6] = make_float2(k_regs[12], k_regs[13]);
        k2[7] = make_float2(k_regs[14], k_regs[15]);
        float2 accpq_0_0 = make_float2(0.0f, 0.0f);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[1], k2[1], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[3], k2[3], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[5], k2[5], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accpq_0_0);
        accpq_0_0 = fma_f32x2_rn_ftz(S2[7], k2[7], accpq_0_0);
        float dotpq_0 = accpq_0_0.x + accpq_0_0.y;
        float _shfl_xor_sync_136 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 1);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_136;
        float _shfl_xor_sync_137 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 2);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_137;
        float _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 4);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_138;
#endif
        o4[0] = dotpq_0;
#if __CUDA_ARCH__ == 900
        float accpq_1_0_x = 0.0f;
        float accpq_1_0_y = 0.0f;
        float accpq_1_1_x = 0.0f;
        float accpq_1_1_y = 0.0f;
        float _fma_880;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_880)
            : "f"(S[16]), "f"(k_regs[0]), "f"(accpq_1_0_x));
        float _fma_881;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_881)
            : "f"(S[17]), "f"(k_regs[1]), "f"(accpq_1_0_y));
        accpq_1_0_x = _fma_880;
        accpq_1_0_y = _fma_881;
        float _fma_882;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_882)
            : "f"(S[18]), "f"(k_regs[2]), "f"(accpq_1_1_x));
        float _fma_883;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_883)
            : "f"(S[19]), "f"(k_regs[3]), "f"(accpq_1_1_y));
        accpq_1_1_x = _fma_882;
        accpq_1_1_y = _fma_883;
        float _fma_884;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_884)
            : "f"(S[20]), "f"(k_regs[4]), "f"(accpq_1_0_x));
        float _fma_885;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_885)
            : "f"(S[21]), "f"(k_regs[5]), "f"(accpq_1_0_y));
        accpq_1_0_x = _fma_884;
        accpq_1_0_y = _fma_885;
        float _fma_886;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_886)
            : "f"(S[22]), "f"(k_regs[6]), "f"(accpq_1_1_x));
        float _fma_887;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_887)
            : "f"(S[23]), "f"(k_regs[7]), "f"(accpq_1_1_y));
        accpq_1_1_x = _fma_886;
        accpq_1_1_y = _fma_887;
        float _fma_888;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_888)
            : "f"(S[24]), "f"(k_regs[8]), "f"(accpq_1_0_x));
        float _fma_889;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_889)
            : "f"(S[25]), "f"(k_regs[9]), "f"(accpq_1_0_y));
        accpq_1_0_x = _fma_888;
        accpq_1_0_y = _fma_889;
        float _fma_890;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_890)
            : "f"(S[26]), "f"(k_regs[10]), "f"(accpq_1_1_x));
        float _fma_891;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_891)
            : "f"(S[27]), "f"(k_regs[11]), "f"(accpq_1_1_y));
        accpq_1_1_x = _fma_890;
        accpq_1_1_y = _fma_891;
        float _fma_892;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_892)
            : "f"(S[28]), "f"(k_regs[12]), "f"(accpq_1_0_x));
        float _fma_893;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_893)
            : "f"(S[29]), "f"(k_regs[13]), "f"(accpq_1_0_y));
        accpq_1_0_x = _fma_892;
        accpq_1_0_y = _fma_893;
        float _fma_894;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_894)
            : "f"(S[30]), "f"(k_regs[14]), "f"(accpq_1_1_x));
        float _fma_895;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_895)
            : "f"(S[31]), "f"(k_regs[15]), "f"(accpq_1_1_y));
        accpq_1_1_x = _fma_894;
        accpq_1_1_y = _fma_895;
        float _add_197;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_197) : "f"(accpq_1_0_x), "f"(accpq_1_1_x));
        float _add_198;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_198) : "f"(accpq_1_0_y), "f"(accpq_1_1_y));
        float dotpq_1 = _add_197 + _add_198;
        float _shfl_xor_sync_199 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 1);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_199;
        float _shfl_xor_sync_200 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 2);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_200;
        float _shfl_xor_sync_201 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 4);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_201;
#else
        float2 accpq_1_0 = make_float2(0.0f, 0.0f);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[9], k2[1], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[11], k2[3], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[13], k2[5], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accpq_1_0);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[15], k2[7], accpq_1_0);
        float dotpq_1 = accpq_1_0.x + accpq_1_0.y;
        float _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 1);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_139;
        float _shfl_xor_sync_140 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 2);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_140;
        float _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 4);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_141;
#endif
        o4[1] = dotpq_1;
#if __CUDA_ARCH__ == 900
        float accpq_2_0_x = 0.0f;
        float accpq_2_0_y = 0.0f;
        float accpq_2_1_x = 0.0f;
        float accpq_2_1_y = 0.0f;
        float _fma_896;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_896)
            : "f"(S[32]), "f"(k_regs[0]), "f"(accpq_2_0_x));
        float _fma_897;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_897)
            : "f"(S[33]), "f"(k_regs[1]), "f"(accpq_2_0_y));
        accpq_2_0_x = _fma_896;
        accpq_2_0_y = _fma_897;
        float _fma_898;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_898)
            : "f"(S[34]), "f"(k_regs[2]), "f"(accpq_2_1_x));
        float _fma_899;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_899)
            : "f"(S[35]), "f"(k_regs[3]), "f"(accpq_2_1_y));
        accpq_2_1_x = _fma_898;
        accpq_2_1_y = _fma_899;
        float _fma_900;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_900)
            : "f"(S[36]), "f"(k_regs[4]), "f"(accpq_2_0_x));
        float _fma_901;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_901)
            : "f"(S[37]), "f"(k_regs[5]), "f"(accpq_2_0_y));
        accpq_2_0_x = _fma_900;
        accpq_2_0_y = _fma_901;
        float _fma_902;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_902)
            : "f"(S[38]), "f"(k_regs[6]), "f"(accpq_2_1_x));
        float _fma_903;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_903)
            : "f"(S[39]), "f"(k_regs[7]), "f"(accpq_2_1_y));
        accpq_2_1_x = _fma_902;
        accpq_2_1_y = _fma_903;
        float _fma_904;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_904)
            : "f"(S[40]), "f"(k_regs[8]), "f"(accpq_2_0_x));
        float _fma_905;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_905)
            : "f"(S[41]), "f"(k_regs[9]), "f"(accpq_2_0_y));
        accpq_2_0_x = _fma_904;
        accpq_2_0_y = _fma_905;
        float _fma_906;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_906)
            : "f"(S[42]), "f"(k_regs[10]), "f"(accpq_2_1_x));
        float _fma_907;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_907)
            : "f"(S[43]), "f"(k_regs[11]), "f"(accpq_2_1_y));
        accpq_2_1_x = _fma_906;
        accpq_2_1_y = _fma_907;
        float _fma_908;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_908)
            : "f"(S[44]), "f"(k_regs[12]), "f"(accpq_2_0_x));
        float _fma_909;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_909)
            : "f"(S[45]), "f"(k_regs[13]), "f"(accpq_2_0_y));
        accpq_2_0_x = _fma_908;
        accpq_2_0_y = _fma_909;
        float _fma_910;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_910)
            : "f"(S[46]), "f"(k_regs[14]), "f"(accpq_2_1_x));
        float _fma_911;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_911)
            : "f"(S[47]), "f"(k_regs[15]), "f"(accpq_2_1_y));
        accpq_2_1_x = _fma_910;
        accpq_2_1_y = _fma_911;
        float _add_202;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_202) : "f"(accpq_2_0_x), "f"(accpq_2_1_x));
        float _add_203;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_203) : "f"(accpq_2_0_y), "f"(accpq_2_1_y));
        float dotpq_2 = _add_202 + _add_203;
        float _shfl_xor_sync_204 = __shfl_xor_sync(0xFFFFFFFF, dotpq_2, 1);
        dotpq_2 = dotpq_2 + _shfl_xor_sync_204;
        float _shfl_xor_sync_205 = __shfl_xor_sync(0xFFFFFFFF, dotpq_2, 2);
        dotpq_2 = dotpq_2 + _shfl_xor_sync_205;
        float _shfl_xor_sync_206 = __shfl_xor_sync(0xFFFFFFFF, dotpq_2, 4);
        dotpq_2 = dotpq_2 + _shfl_xor_sync_206;
#else
        float2 accpq_2_0 = make_float2(0.0f, 0.0f);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accpq_2_0);
        accpq_2_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accpq_2_0);
        float dotpq_2 = accpq_2_0.x + accpq_2_0.y;
        float _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, dotpq_2, 1);
        dotpq_2 = dotpq_2 + _shfl_xor_sync_142;
        float _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, dotpq_2, 2);
        dotpq_2 = dotpq_2 + _shfl_xor_sync_143;
        float _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, dotpq_2, 4);
        dotpq_2 = dotpq_2 + _shfl_xor_sync_144;
#endif
        o4[2] = dotpq_2;
#if __CUDA_ARCH__ == 900
        float accpq_3_0_x = 0.0f;
        float accpq_3_0_y = 0.0f;
        float accpq_3_1_x = 0.0f;
        float accpq_3_1_y = 0.0f;
        float _fma_912;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_912)
            : "f"(S[48]), "f"(k_regs[0]), "f"(accpq_3_0_x));
        float _fma_913;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_913)
            : "f"(S[49]), "f"(k_regs[1]), "f"(accpq_3_0_y));
        accpq_3_0_x = _fma_912;
        accpq_3_0_y = _fma_913;
        float _fma_914;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_914)
            : "f"(S[50]), "f"(k_regs[2]), "f"(accpq_3_1_x));
        float _fma_915;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_915)
            : "f"(S[51]), "f"(k_regs[3]), "f"(accpq_3_1_y));
        accpq_3_1_x = _fma_914;
        accpq_3_1_y = _fma_915;
        float _fma_916;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_916)
            : "f"(S[52]), "f"(k_regs[4]), "f"(accpq_3_0_x));
        float _fma_917;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_917)
            : "f"(S[53]), "f"(k_regs[5]), "f"(accpq_3_0_y));
        accpq_3_0_x = _fma_916;
        accpq_3_0_y = _fma_917;
        float _fma_918;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_918)
            : "f"(S[54]), "f"(k_regs[6]), "f"(accpq_3_1_x));
        float _fma_919;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_919)
            : "f"(S[55]), "f"(k_regs[7]), "f"(accpq_3_1_y));
        accpq_3_1_x = _fma_918;
        accpq_3_1_y = _fma_919;
        float _fma_920;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_920)
            : "f"(S[56]), "f"(k_regs[8]), "f"(accpq_3_0_x));
        float _fma_921;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_921)
            : "f"(S[57]), "f"(k_regs[9]), "f"(accpq_3_0_y));
        accpq_3_0_x = _fma_920;
        accpq_3_0_y = _fma_921;
        float _fma_922;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_922)
            : "f"(S[58]), "f"(k_regs[10]), "f"(accpq_3_1_x));
        float _fma_923;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_923)
            : "f"(S[59]), "f"(k_regs[11]), "f"(accpq_3_1_y));
        accpq_3_1_x = _fma_922;
        accpq_3_1_y = _fma_923;
        float _fma_924;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_924)
            : "f"(S[60]), "f"(k_regs[12]), "f"(accpq_3_0_x));
        float _fma_925;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_925)
            : "f"(S[61]), "f"(k_regs[13]), "f"(accpq_3_0_y));
        accpq_3_0_x = _fma_924;
        accpq_3_0_y = _fma_925;
        float _fma_926;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_926)
            : "f"(S[62]), "f"(k_regs[14]), "f"(accpq_3_1_x));
        float _fma_927;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_927)
            : "f"(S[63]), "f"(k_regs[15]), "f"(accpq_3_1_y));
        accpq_3_1_x = _fma_926;
        accpq_3_1_y = _fma_927;
        float _add_207;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_207) : "f"(accpq_3_0_x), "f"(accpq_3_1_x));
        float _add_208;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_208) : "f"(accpq_3_0_y), "f"(accpq_3_1_y));
        float dotpq_3 = _add_207 + _add_208;
        float _shfl_xor_sync_209 = __shfl_xor_sync(0xFFFFFFFF, dotpq_3, 1);
        dotpq_3 = dotpq_3 + _shfl_xor_sync_209;
        float _shfl_xor_sync_210 = __shfl_xor_sync(0xFFFFFFFF, dotpq_3, 2);
        dotpq_3 = dotpq_3 + _shfl_xor_sync_210;
        float _shfl_xor_sync_211 = __shfl_xor_sync(0xFFFFFFFF, dotpq_3, 4);
        dotpq_3 = dotpq_3 + _shfl_xor_sync_211;
#else
        float2 accpq_3_0 = make_float2(0.0f, 0.0f);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[24], k2[0], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[25], k2[1], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[26], k2[2], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[27], k2[3], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[28], k2[4], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[29], k2[5], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[30], k2[6], accpq_3_0);
        accpq_3_0 = fma_f32x2_rn_ftz(S2[31], k2[7], accpq_3_0);
        float dotpq_3 = accpq_3_0.x + accpq_3_0.y;
        float _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, dotpq_3, 1);
        dotpq_3 = dotpq_3 + _shfl_xor_sync_145;
        float _shfl_xor_sync_146 = __shfl_xor_sync(0xFFFFFFFF, dotpq_3, 2);
        dotpq_3 = dotpq_3 + _shfl_xor_sync_146;
        float _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, dotpq_3, 4);
        dotpq_3 = dotpq_3 + _shfl_xor_sync_147;
#endif
        o4[3] = dotpq_3;
        if (u == 0) {
          long long out_index_1 = (long long)n * out_stride_p0 + (long long)t_1 * out_stride_p1 +
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
            *reinterpret_cast<float4*>(state + write_rows_base + 32) = _v4;
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
            *reinterpret_cast<float4*>(state + write_rows_base + 64) = _v4;
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
            *reinterpret_cast<float4*>(state + write_rows_base + 96) = _v4;
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
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2) = _v4;
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
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 32) = _v4;
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
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 64) = _v4;
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
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 96) = _v4;
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
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2) = _v4;
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
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2 + 32) = _v4;
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
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2 + 64) = _v4;
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
            *reinterpret_cast<float4*>(state + write_rows_base + 2 * state_stride_p2 + 96) = _v4;
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
            *reinterpret_cast<float4*>(state + write_rows_base + 3 * state_stride_p2) = _v4;
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
            *reinterpret_cast<float4*>(state + write_rows_base + 3 * state_stride_p2 + 32) = _v4;
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
            *reinterpret_cast<float4*>(state + write_rows_base + 3 * state_stride_p2 + 64) = _v4;
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
            *reinterpret_cast<float4*>(state + write_rows_base + 3 * state_stride_p2 + 96) = _v4;
          }
        }
      }
    }
  }
}

}  // extern "C"
