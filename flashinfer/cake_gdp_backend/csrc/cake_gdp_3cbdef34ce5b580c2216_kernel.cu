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
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 1

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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void kernel_cake_gdp_3cbdef34ce5b580c2216(
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
  int u = tid_0 - tid_0 / 8 * 8;
  int g = tid_0 / 8;
  int row0 = g * 2;
  unsigned int lane_byte = (unsigned int)(u * 16);
  unsigned int group_byte = (unsigned int)(g * 8);
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
  if (read_state_slot_valid != 0) {
    float S[32];
    float2 S2[16];
    float k_regs[16];
    float2 k2[8];
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
#pragma unroll
    for (int mm = 0; mm < 3; mm++) {
      int m = mm * 8 + warp_local;
      if ((mm + 1) * 8 <= 24) {
        long long k_index =
            (long long)n * k_stride_p0 + (long long)m * k_stride_p1 + (long long)h * k_stride_p2;
        {
          uint2 _vld_8;
          _vld_8 = *reinterpret_cast<const uint2*>(k + k_index + (long long)k_col0);
          uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8);
#pragma unroll
          for (int _pair = 0; _pair < 2; _pair++) {
            (&r_k[0 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
            (&r_k[0 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
          }
        }
        float2 k_pair0 = make_float2(r_k[0], r_k[1]);
        float2 k_pair1 = make_float2(r_k[2], r_k[3]);
        float2 sum_k_pair = fma_f32x2_rn_ftz(k_pair0, k_pair0, make_float2(0.0f, 0.0f));
        sum_k_pair = fma_f32x2_rn_ftz(k_pair1, k_pair1, sum_k_pair);
        float sum_k = sum_k_pair.x + sum_k_pair.y;
        float _warp_reduce_9 = sum_k;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
          _warp_reduce_9 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_9, offset);
        sum_k = _warp_reduce_9;
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
          float _exp_11 = __expf(-(float)b[b_index]);
          float _rcp_10 = approx_rcp(1.0f + _exp_11);
          sBeta[m] = _rcp_10;
        }
      }
      if ((mm + 1) * 8 > 24) {
        if (m < steps) {
          long long k_index_1 =
              (long long)n * k_stride_p0 + (long long)m * k_stride_p1 + (long long)h * k_stride_p2;
          {
            uint2 _vld_12;
            _vld_12 = *reinterpret_cast<const uint2*>(k + k_index_1 + (long long)k_col0);
            uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12);
#pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
              (&r_k[0 + _pair * 2])[0] =
                  __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
              (&r_k[0 + _pair * 2])[1] =
                  __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
            }
          }
          float2 k_pair0_1 = make_float2(r_k[0], r_k[1]);
          float2 k_pair1_1 = make_float2(r_k[2], r_k[3]);
          float2 sum_k_pair_1 = fma_f32x2_rn_ftz(k_pair0_1, k_pair0_1, make_float2(0.0f, 0.0f));
          sum_k_pair_1 = fma_f32x2_rn_ftz(k_pair1_1, k_pair1_1, sum_k_pair_1);
          float sum_k_1 = sum_k_pair_1.x + sum_k_pair_1.y;
          float _warp_reduce_13 = sum_k_1;
#pragma unroll
          for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_13 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_13, offset);
          sum_k_1 = _warp_reduce_13;
          float k_norm_1 = rsqrtf(sum_k_1 + 1e-06f);
          float2 k_norm_pair_1 = make_float2(k_norm_1, k_norm_1);
          k_pair0_1 = mul_f32x2(k_pair0_1, k_norm_pair_1);
          k_pair1_1 = mul_f32x2(k_pair1_1, k_norm_pair_1);
          sK[m * 128 + k_col0] = k_pair0_1.x;
          sK[m * 128 + k_col0 + 1] = k_pair0_1.y;
          sK[m * 128 + k_col0 + 2] = k_pair1_1.x;
          sK[m * 128 + k_col0 + 3] = k_pair1_1.y;
          long long v_index_1 =
              (long long)n * v_stride_p0 + (long long)m * v_stride_p1 + (long long)h * v_stride_p2;
          int v_col_1 = lane_local * 2;
          sV[m * 64 + v_col_1] = (float)v[v_index_1 + (long long)v_col_1];
          sV[m * 64 + v_col_1 + 1] = (float)v[v_index_1 + (long long)v_col_1 + 1];
          if (lane_local == 0) {
            long long b_index_1 = (long long)n * b_stride_p0 + (long long)m * b_stride_p1 +
                                  (long long)h * b_stride_p2;
            float _exp_15 = __expf(-(float)b[b_index_1]);
            float _rcp_14 = approx_rcp(1.0f + _exp_15);
            sBeta[m] = _rcp_14;
          }
        }
      }
    }
#pragma unroll
    for (int tt = 0; tt < 1; tt++) {
      int t = tt * 8 + warp_local;
      if ((tt + 1) * 8 <= 8) {
        long long q_index =
            (long long)n * q_stride_p0 + (long long)t * q_stride_p1 + (long long)h * q_stride_p2;
        {
          uint2 _vld_16;
          _vld_16 = *reinterpret_cast<const uint2*>(q + q_index + (long long)k_col0);
          uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16);
#pragma unroll
          for (int _pair = 0; _pair < 2; _pair++) {
            (&r_q[0 + _pair * 2])[0] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
            (&r_q[0 + _pair * 2])[1] =
                __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
          }
        }
        float2 q_raw_pair0 = make_float2(r_q[0], r_q[1]);
        float2 q_raw_pair1 = make_float2(r_q[2], r_q[3]);
        float q_norm = scale;
        float2 sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair0, q_raw_pair0, make_float2(0.0f, 0.0f));
        sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair1, q_raw_pair1, sum_q_pair);
        float sum_q = sum_q_pair.x + sum_q_pair.y;
        float _warp_reduce_17 = sum_q;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
          _warp_reduce_17 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_17, offset);
        sum_q = _warp_reduce_17;
        float _rsqrt_18 = rsqrtf(sum_q + 1e-06f);
        q_norm = _rsqrt_18 * scale;
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
            float _exp_20 = __expf(x);
            float _log2_19;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_19) : "f"(1.0f + _exp_20));
            softplus_x = _log2_19 * 0.6931471805599453f;
          }
          float _exp_21 = __expf(A_log[h]);
          float g_log = (-_exp_21) * softplus_x;
          float _exp_22 = __expf(g_log);
          sAlpha[t] = _exp_22;
          if (use_ssm_state_indices != 0) {
            sSsm[t] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 +
                                        (long long)t * ssm_state_indices_stride_p1];
          }
          if (use_ssm_state_indices == 0) {
            sSsm[t] = -1;
          }
        }
      }
      if ((tt + 1) * 8 > 8) {
        if (t < num_tokens) {
          long long q_index_1 =
              (long long)n * q_stride_p0 + (long long)t * q_stride_p1 + (long long)h * q_stride_p2;
          {
            uint2 _vld_23;
            _vld_23 = *reinterpret_cast<const uint2*>(q + q_index_1 + (long long)k_col0);
            uint32_t* _vpairs_23 = reinterpret_cast<uint32_t*>(&_vld_23);
#pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
              (&r_q[0 + _pair * 2])[0] =
                  __uint_as_float(static_cast<uint32_t>(_vpairs_23[_pair]) << 16);
              (&r_q[0 + _pair * 2])[1] =
                  __uint_as_float(static_cast<uint32_t>(_vpairs_23[_pair]) & 0xffff0000u);
            }
          }
          float2 q_raw_pair0_1 = make_float2(r_q[0], r_q[1]);
          float2 q_raw_pair1_1 = make_float2(r_q[2], r_q[3]);
          float q_norm_1 = scale;
          float2 sum_q_pair_1 =
              fma_f32x2_rn_ftz(q_raw_pair0_1, q_raw_pair0_1, make_float2(0.0f, 0.0f));
          sum_q_pair_1 = fma_f32x2_rn_ftz(q_raw_pair1_1, q_raw_pair1_1, sum_q_pair_1);
          float sum_q_1 = sum_q_pair_1.x + sum_q_pair_1.y;
          float _warp_reduce_24 = sum_q_1;
#pragma unroll
          for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_24 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_24, offset);
          sum_q_1 = _warp_reduce_24;
          float _rsqrt_25 = rsqrtf(sum_q_1 + 1e-06f);
          q_norm_1 = _rsqrt_25 * scale;
          float2 q_norm_pair_1 = make_float2(q_norm_1, q_norm_1);
          float2 q_pair0_1;
          asm("mul.rn.ftz.f32x2 %0, %1, %2;"
              : "=l"(*(unsigned long long*)&q_pair0_1)
              : "l"(*(const unsigned long long*)&q_raw_pair0_1),
                "l"(*(const unsigned long long*)&q_norm_pair_1));
          float2 q_pair1_1;
          asm("mul.rn.ftz.f32x2 %0, %1, %2;"
              : "=l"(*(unsigned long long*)&q_pair1_1)
              : "l"(*(const unsigned long long*)&q_raw_pair1_1),
                "l"(*(const unsigned long long*)&q_norm_pair_1));
          sQ[t * 128 + k_col0] = q_pair0_1.x;
          sQ[t * 128 + k_col0 + 1] = q_pair0_1.y;
          sQ[t * 128 + k_col0 + 2] = q_pair1_1.x;
          sQ[t * 128 + k_col0 + 3] = q_pair1_1.y;
          if (lane_local == 0) {
            long long gate_index_1 = (long long)n * a_stride_p0 + (long long)t * a_stride_p1 +
                                     (long long)h * a_stride_p2;
            float x_1 = (float)a[gate_index_1] + dt_bias[h];
            float softplus_x_1 = x_1;
            if (x_1 <= 20.0f) {
              float _exp_27 = __expf(x_1);
              float _log2_26;
              asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_26) : "f"(1.0f + _exp_27));
              softplus_x_1 = _log2_26 * 0.6931471805599453f;
            }
            float _exp_28 = __expf(A_log[h]);
            float g_log_1 = (-_exp_28) * softplus_x_1;
            float _exp_29 = __expf(g_log_1);
            sAlpha[t] = _exp_29;
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
    if (use_ssm_state_indices != 0) {
#pragma unroll 1
      for (int t_1 = 0; t_1 < 8; t_1++) {
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
        k2[0] = make_float2(k_regs[0], k_regs[1]);
        k2[1] = make_float2(k_regs[2], k_regs[3]);
        k2[2] = make_float2(k_regs[4], k_regs[5]);
        k2[3] = make_float2(k_regs[6], k_regs[7]);
        k2[4] = make_float2(k_regs[8], k_regs[9]);
        k2[5] = make_float2(k_regs[10], k_regs[11]);
        k2[6] = make_float2(k_regs[12], k_regs[13]);
        k2[7] = make_float2(k_regs[14], k_regs[15]);
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
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
        float2 _add_0 = add_f32x2(accs0_0_0, accs0_0_1);
        float dots0_0 = _add_0.x + _add_0.y;
        float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 1);
        dots0_0 = dots0_0 + _shfl_xor_sync_30;
        float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 2);
        dots0_0 = dots0_0 + _shfl_xor_sync_31;
        float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 4);
        dots0_0 = dots0_0 + _shfl_xor_sync_32;
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
        float2 accs0_1_0 = make_float2(0.0f, 0.0f);
        float2 accs0_1_1 = make_float2(0.0f, 0.0f);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accs0_1_0);
        accs0_1_1 = fma_f32x2_rn_ftz(S2[9], k2[1], accs0_1_1);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accs0_1_0);
        accs0_1_1 = fma_f32x2_rn_ftz(S2[11], k2[3], accs0_1_1);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accs0_1_0);
        accs0_1_1 = fma_f32x2_rn_ftz(S2[13], k2[5], accs0_1_1);
        accs0_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accs0_1_0);
        accs0_1_1 = fma_f32x2_rn_ftz(S2[15], k2[7], accs0_1_1);
        float2 _add_1 = add_f32x2(accs0_1_0, accs0_1_1);
        float dots0_1 = _add_1.x + _add_1.y;
        float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 1);
        dots0_1 = dots0_1 + _shfl_xor_sync_33;
        float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 2);
        dots0_1 = dots0_1 + _shfl_xor_sync_34;
        float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 4);
        dots0_1 = dots0_1 + _shfl_xor_sync_35;
        float vns0_1 = (v_regs[1] - dots0_1) * betas0;
        float2 vn2s0_1 = make_float2(vns0_1, vns0_1);
        S2[8] = fma_f32x2_rn_ftz(k2[0], vn2s0_1, S2[8]);
        S2[9] = fma_f32x2_rn_ftz(k2[1], vn2s0_1, S2[9]);
        S2[10] = fma_f32x2_rn_ftz(k2[2], vn2s0_1, S2[10]);
        S2[11] = fma_f32x2_rn_ftz(k2[3], vn2s0_1, S2[11]);
        S2[12] = fma_f32x2_rn_ftz(k2[4], vn2s0_1, S2[12]);
        S2[13] = fma_f32x2_rn_ftz(k2[5], vn2s0_1, S2[13]);
        S2[14] = fma_f32x2_rn_ftz(k2[6], vn2s0_1, S2[14]);
        S2[15] = fma_f32x2_rn_ftz(k2[7], vn2s0_1, S2[15]);
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
        k2[0] = make_float2(k_regs[0], k_regs[1]);
        k2[1] = make_float2(k_regs[2], k_regs[3]);
        k2[2] = make_float2(k_regs[4], k_regs[5]);
        k2[3] = make_float2(k_regs[6], k_regs[7]);
        k2[4] = make_float2(k_regs[8], k_regs[9]);
        k2[5] = make_float2(k_regs[10], k_regs[11]);
        k2[6] = make_float2(k_regs[12], k_regs[13]);
        k2[7] = make_float2(k_regs[14], k_regs[15]);
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
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
        float2 _add_2 = add_f32x2(accs1_0_0, accs1_0_1);
        float dots1_0 = _add_2.x + _add_2.y;
        float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 1);
        dots1_0 = dots1_0 + _shfl_xor_sync_36;
        float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 2);
        dots1_0 = dots1_0 + _shfl_xor_sync_37;
        float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 4);
        dots1_0 = dots1_0 + _shfl_xor_sync_38;
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
        float2 accs1_1_0 = make_float2(0.0f, 0.0f);
        float2 accs1_1_1 = make_float2(0.0f, 0.0f);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accs1_1_0);
        accs1_1_1 = fma_f32x2_rn_ftz(S2[9], k2[1], accs1_1_1);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accs1_1_0);
        accs1_1_1 = fma_f32x2_rn_ftz(S2[11], k2[3], accs1_1_1);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accs1_1_0);
        accs1_1_1 = fma_f32x2_rn_ftz(S2[13], k2[5], accs1_1_1);
        accs1_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accs1_1_0);
        accs1_1_1 = fma_f32x2_rn_ftz(S2[15], k2[7], accs1_1_1);
        float2 _add_3 = add_f32x2(accs1_1_0, accs1_1_1);
        float dots1_1 = _add_3.x + _add_3.y;
        float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 1);
        dots1_1 = dots1_1 + _shfl_xor_sync_39;
        float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 2);
        dots1_1 = dots1_1 + _shfl_xor_sync_40;
        float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 4);
        dots1_1 = dots1_1 + _shfl_xor_sync_41;
        float vns1_1 = (v_regs[1] - dots1_1) * betas1;
        float2 vn2s1_1 = make_float2(vns1_1, vns1_1);
        S2[8] = fma_f32x2_rn_ftz(k2[0], vn2s1_1, S2[8]);
        S2[9] = fma_f32x2_rn_ftz(k2[1], vn2s1_1, S2[9]);
        S2[10] = fma_f32x2_rn_ftz(k2[2], vn2s1_1, S2[10]);
        S2[11] = fma_f32x2_rn_ftz(k2[3], vn2s1_1, S2[11]);
        S2[12] = fma_f32x2_rn_ftz(k2[4], vn2s1_1, S2[12]);
        S2[13] = fma_f32x2_rn_ftz(k2[5], vn2s1_1, S2[13]);
        S2[14] = fma_f32x2_rn_ftz(k2[6], vn2s1_1, S2[14]);
        S2[15] = fma_f32x2_rn_ftz(k2[7], vn2s1_1, S2[15]);
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
        k2[0] = make_float2(k_regs[0], k_regs[1]);
        k2[1] = make_float2(k_regs[2], k_regs[3]);
        k2[2] = make_float2(k_regs[4], k_regs[5]);
        k2[3] = make_float2(k_regs[6], k_regs[7]);
        k2[4] = make_float2(k_regs[8], k_regs[9]);
        k2[5] = make_float2(k_regs[10], k_regs[11]);
        k2[6] = make_float2(k_regs[12], k_regs[13]);
        k2[7] = make_float2(k_regs[14], k_regs[15]);
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
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
        float2 _add_4 = add_f32x2(accs2_0_0, accs2_0_1);
        float dots2_0 = _add_4.x + _add_4.y;
        float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 1);
        dots2_0 = dots2_0 + _shfl_xor_sync_42;
        float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 2);
        dots2_0 = dots2_0 + _shfl_xor_sync_43;
        float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 4);
        dots2_0 = dots2_0 + _shfl_xor_sync_44;
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
        float2 accs2_1_0 = make_float2(0.0f, 0.0f);
        float2 accs2_1_1 = make_float2(0.0f, 0.0f);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accs2_1_0);
        accs2_1_1 = fma_f32x2_rn_ftz(S2[9], k2[1], accs2_1_1);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accs2_1_0);
        accs2_1_1 = fma_f32x2_rn_ftz(S2[11], k2[3], accs2_1_1);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accs2_1_0);
        accs2_1_1 = fma_f32x2_rn_ftz(S2[13], k2[5], accs2_1_1);
        accs2_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accs2_1_0);
        accs2_1_1 = fma_f32x2_rn_ftz(S2[15], k2[7], accs2_1_1);
        float2 _add_5 = add_f32x2(accs2_1_0, accs2_1_1);
        float dots2_1 = _add_5.x + _add_5.y;
        float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 1);
        dots2_1 = dots2_1 + _shfl_xor_sync_45;
        float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 2);
        dots2_1 = dots2_1 + _shfl_xor_sync_46;
        float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 4);
        dots2_1 = dots2_1 + _shfl_xor_sync_47;
        float vns2_1 = (v_regs[1] - dots2_1) * betas2;
        float2 vn2s2_1 = make_float2(vns2_1, vns2_1);
        S2[8] = fma_f32x2_rn_ftz(k2[0], vn2s2_1, S2[8]);
        S2[9] = fma_f32x2_rn_ftz(k2[1], vn2s2_1, S2[9]);
        S2[10] = fma_f32x2_rn_ftz(k2[2], vn2s2_1, S2[10]);
        S2[11] = fma_f32x2_rn_ftz(k2[3], vn2s2_1, S2[11]);
        S2[12] = fma_f32x2_rn_ftz(k2[4], vn2s2_1, S2[12]);
        S2[13] = fma_f32x2_rn_ftz(k2[5], vn2s2_1, S2[13]);
        S2[14] = fma_f32x2_rn_ftz(k2[6], vn2s2_1, S2[14]);
        S2[15] = fma_f32x2_rn_ftz(k2[7], vn2s2_1, S2[15]);
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
        k2[0] = make_float2(k_regs[0], k_regs[1]);
        k2[1] = make_float2(k_regs[2], k_regs[3]);
        k2[2] = make_float2(k_regs[4], k_regs[5]);
        k2[3] = make_float2(k_regs[6], k_regs[7]);
        k2[4] = make_float2(k_regs[8], k_regs[9]);
        k2[5] = make_float2(k_regs[10], k_regs[11]);
        k2[6] = make_float2(k_regs[12], k_regs[13]);
        k2[7] = make_float2(k_regs[14], k_regs[15]);
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
        float2 _add_6 = add_f32x2(accsq_0_0, accsq_0_1);
        float dotsq_0 = _add_6.x + _add_6.y;
        float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 1);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_48;
        float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 2);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_49;
        float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 4);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_50;
        o4[0] = dotsq_0;
        float2 accsq_1_0 = make_float2(0.0f, 0.0f);
        float2 accsq_1_1 = make_float2(0.0f, 0.0f);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accsq_1_0);
        accsq_1_1 = fma_f32x2_rn_ftz(S2[9], k2[1], accsq_1_1);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accsq_1_0);
        accsq_1_1 = fma_f32x2_rn_ftz(S2[11], k2[3], accsq_1_1);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accsq_1_0);
        accsq_1_1 = fma_f32x2_rn_ftz(S2[13], k2[5], accsq_1_1);
        accsq_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accsq_1_0);
        accsq_1_1 = fma_f32x2_rn_ftz(S2[15], k2[7], accsq_1_1);
        float2 _add_7 = add_f32x2(accsq_1_0, accsq_1_1);
        float dotsq_1 = _add_7.x + _add_7.y;
        float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 1);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_51;
        float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 2);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_52;
        float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 4);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_53;
        o4[1] = dotsq_1;
        if (u == 0) {
          long long out_index = (long long)n * out_stride_p0 + (long long)t_1 * out_stride_p1 +
                                (long long)h * out_stride_p2 + (long long)row0;
          {
            __nv_bfloat162 _pk = __floats2bfloat162_rn(o4[0 + 0], o4[0 + 1]);
            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out))[out_index]) = _pk;
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
      }
    }
    if (use_ssm_state_indices == 0) {
#pragma unroll 1
      for (int t_2 = 0; t_2 < 8; t_2++) {
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
        k2[0] = make_float2(k_regs[0], k_regs[1]);
        k2[1] = make_float2(k_regs[2], k_regs[3]);
        k2[2] = make_float2(k_regs[4], k_regs[5]);
        k2[3] = make_float2(k_regs[6], k_regs[7]);
        k2[4] = make_float2(k_regs[8], k_regs[9]);
        k2[5] = make_float2(k_regs[10], k_regs[11]);
        k2[6] = make_float2(k_regs[12], k_regs[13]);
        k2[7] = make_float2(k_regs[14], k_regs[15]);
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
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
        float2 _add_8 = add_f32x2(accp0_0_0, accp0_0_1);
        float dotp0_0 = _add_8.x + _add_8.y;
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 1);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_54;
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 2);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_55;
        float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 4);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_56;
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
        float2 accp0_1_0 = make_float2(0.0f, 0.0f);
        float2 accp0_1_1 = make_float2(0.0f, 0.0f);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accp0_1_0);
        accp0_1_1 = fma_f32x2_rn_ftz(S2[9], k2[1], accp0_1_1);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accp0_1_0);
        accp0_1_1 = fma_f32x2_rn_ftz(S2[11], k2[3], accp0_1_1);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accp0_1_0);
        accp0_1_1 = fma_f32x2_rn_ftz(S2[13], k2[5], accp0_1_1);
        accp0_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accp0_1_0);
        accp0_1_1 = fma_f32x2_rn_ftz(S2[15], k2[7], accp0_1_1);
        float2 _add_9 = add_f32x2(accp0_1_0, accp0_1_1);
        float dotp0_1 = _add_9.x + _add_9.y;
        float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 1);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_57;
        float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 2);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_58;
        float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 4);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_59;
        float vnp0_1 = (v_regs[1] - dotp0_1) * betap0;
        float2 vn2p0_1 = make_float2(vnp0_1, vnp0_1);
        S2[8] = fma_f32x2_rn_ftz(k2[0], vn2p0_1, S2[8]);
        S2[9] = fma_f32x2_rn_ftz(k2[1], vn2p0_1, S2[9]);
        S2[10] = fma_f32x2_rn_ftz(k2[2], vn2p0_1, S2[10]);
        S2[11] = fma_f32x2_rn_ftz(k2[3], vn2p0_1, S2[11]);
        S2[12] = fma_f32x2_rn_ftz(k2[4], vn2p0_1, S2[12]);
        S2[13] = fma_f32x2_rn_ftz(k2[5], vn2p0_1, S2[13]);
        S2[14] = fma_f32x2_rn_ftz(k2[6], vn2p0_1, S2[14]);
        S2[15] = fma_f32x2_rn_ftz(k2[7], vn2p0_1, S2[15]);
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
        k2[0] = make_float2(k_regs[0], k_regs[1]);
        k2[1] = make_float2(k_regs[2], k_regs[3]);
        k2[2] = make_float2(k_regs[4], k_regs[5]);
        k2[3] = make_float2(k_regs[6], k_regs[7]);
        k2[4] = make_float2(k_regs[8], k_regs[9]);
        k2[5] = make_float2(k_regs[10], k_regs[11]);
        k2[6] = make_float2(k_regs[12], k_regs[13]);
        k2[7] = make_float2(k_regs[14], k_regs[15]);
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
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
        float2 _add_10 = add_f32x2(accp1_0_0, accp1_0_1);
        float dotp1_0 = _add_10.x + _add_10.y;
        float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 1);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_60;
        float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 2);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_61;
        float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 4);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_62;
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
        float2 accp1_1_0 = make_float2(0.0f, 0.0f);
        float2 accp1_1_1 = make_float2(0.0f, 0.0f);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accp1_1_0);
        accp1_1_1 = fma_f32x2_rn_ftz(S2[9], k2[1], accp1_1_1);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accp1_1_0);
        accp1_1_1 = fma_f32x2_rn_ftz(S2[11], k2[3], accp1_1_1);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accp1_1_0);
        accp1_1_1 = fma_f32x2_rn_ftz(S2[13], k2[5], accp1_1_1);
        accp1_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accp1_1_0);
        accp1_1_1 = fma_f32x2_rn_ftz(S2[15], k2[7], accp1_1_1);
        float2 _add_11 = add_f32x2(accp1_1_0, accp1_1_1);
        float dotp1_1 = _add_11.x + _add_11.y;
        float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 1);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_63;
        float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 2);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_64;
        float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 4);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_65;
        float vnp1_1 = (v_regs[1] - dotp1_1) * betap1;
        float2 vn2p1_1 = make_float2(vnp1_1, vnp1_1);
        S2[8] = fma_f32x2_rn_ftz(k2[0], vn2p1_1, S2[8]);
        S2[9] = fma_f32x2_rn_ftz(k2[1], vn2p1_1, S2[9]);
        S2[10] = fma_f32x2_rn_ftz(k2[2], vn2p1_1, S2[10]);
        S2[11] = fma_f32x2_rn_ftz(k2[3], vn2p1_1, S2[11]);
        S2[12] = fma_f32x2_rn_ftz(k2[4], vn2p1_1, S2[12]);
        S2[13] = fma_f32x2_rn_ftz(k2[5], vn2p1_1, S2[13]);
        S2[14] = fma_f32x2_rn_ftz(k2[6], vn2p1_1, S2[14]);
        S2[15] = fma_f32x2_rn_ftz(k2[7], vn2p1_1, S2[15]);
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
        k2[0] = make_float2(k_regs[0], k_regs[1]);
        k2[1] = make_float2(k_regs[2], k_regs[3]);
        k2[2] = make_float2(k_regs[4], k_regs[5]);
        k2[3] = make_float2(k_regs[6], k_regs[7]);
        k2[4] = make_float2(k_regs[8], k_regs[9]);
        k2[5] = make_float2(k_regs[10], k_regs[11]);
        k2[6] = make_float2(k_regs[12], k_regs[13]);
        k2[7] = make_float2(k_regs[14], k_regs[15]);
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
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
        float2 _add_12 = add_f32x2(accp2_0_0, accp2_0_1);
        float dotp2_0 = _add_12.x + _add_12.y;
        float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 1);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_66;
        float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 2);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_67;
        float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 4);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_68;
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
        float2 accp2_1_0 = make_float2(0.0f, 0.0f);
        float2 accp2_1_1 = make_float2(0.0f, 0.0f);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accp2_1_0);
        accp2_1_1 = fma_f32x2_rn_ftz(S2[9], k2[1], accp2_1_1);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accp2_1_0);
        accp2_1_1 = fma_f32x2_rn_ftz(S2[11], k2[3], accp2_1_1);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accp2_1_0);
        accp2_1_1 = fma_f32x2_rn_ftz(S2[13], k2[5], accp2_1_1);
        accp2_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accp2_1_0);
        accp2_1_1 = fma_f32x2_rn_ftz(S2[15], k2[7], accp2_1_1);
        float2 _add_13 = add_f32x2(accp2_1_0, accp2_1_1);
        float dotp2_1 = _add_13.x + _add_13.y;
        float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 1);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_69;
        float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 2);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_70;
        float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 4);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_71;
        float vnp2_1 = (v_regs[1] - dotp2_1) * betap2;
        float2 vn2p2_1 = make_float2(vnp2_1, vnp2_1);
        S2[8] = fma_f32x2_rn_ftz(k2[0], vn2p2_1, S2[8]);
        S2[9] = fma_f32x2_rn_ftz(k2[1], vn2p2_1, S2[9]);
        S2[10] = fma_f32x2_rn_ftz(k2[2], vn2p2_1, S2[10]);
        S2[11] = fma_f32x2_rn_ftz(k2[3], vn2p2_1, S2[11]);
        S2[12] = fma_f32x2_rn_ftz(k2[4], vn2p2_1, S2[12]);
        S2[13] = fma_f32x2_rn_ftz(k2[5], vn2p2_1, S2[13]);
        S2[14] = fma_f32x2_rn_ftz(k2[6], vn2p2_1, S2[14]);
        S2[15] = fma_f32x2_rn_ftz(k2[7], vn2p2_1, S2[15]);
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
        k2[0] = make_float2(k_regs[0], k_regs[1]);
        k2[1] = make_float2(k_regs[2], k_regs[3]);
        k2[2] = make_float2(k_regs[4], k_regs[5]);
        k2[3] = make_float2(k_regs[6], k_regs[7]);
        k2[4] = make_float2(k_regs[8], k_regs[9]);
        k2[5] = make_float2(k_regs[10], k_regs[11]);
        k2[6] = make_float2(k_regs[12], k_regs[13]);
        k2[7] = make_float2(k_regs[14], k_regs[15]);
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
        float2 _add_14 = add_f32x2(accpq_0_0, accpq_0_1);
        float dotpq_0 = _add_14.x + _add_14.y;
        float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 1);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_72;
        float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 2);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_73;
        float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 4);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_74;
        o4[0] = dotpq_0;
        float2 accpq_1_0 = make_float2(0.0f, 0.0f);
        float2 accpq_1_1 = make_float2(0.0f, 0.0f);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accpq_1_0);
        accpq_1_1 = fma_f32x2_rn_ftz(S2[9], k2[1], accpq_1_1);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accpq_1_0);
        accpq_1_1 = fma_f32x2_rn_ftz(S2[11], k2[3], accpq_1_1);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accpq_1_0);
        accpq_1_1 = fma_f32x2_rn_ftz(S2[13], k2[5], accpq_1_1);
        accpq_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accpq_1_0);
        accpq_1_1 = fma_f32x2_rn_ftz(S2[15], k2[7], accpq_1_1);
        float2 _add_15 = add_f32x2(accpq_1_0, accpq_1_1);
        float dotpq_1 = _add_15.x + _add_15.y;
        float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 1);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_75;
        float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 2);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_76;
        float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 4);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_77;
        o4[1] = dotpq_1;
        if (u == 0) {
          long long out_index_1 = (long long)n * out_stride_p0 + (long long)t_2 * out_stride_p1 +
                                  (long long)h * out_stride_p2 + (long long)row0;
          {
            __nv_bfloat162 _pk = __floats2bfloat162_rn(o4[0 + 0], o4[0 + 1]);
            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out))[out_index_1]) = _pk;
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
            *reinterpret_cast<float4*>(state + write_rows_base + 32) = _v4;
          }
          r_h[0] = S2[4].x;
          r_h[1] = S2[4].y;
          r_h[2] = S2[5].x;
          r_h[3] = S2[5].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 64) = _v4;
          }
          r_h[0] = S2[6].x;
          r_h[1] = S2[6].y;
          r_h[2] = S2[7].x;
          r_h[3] = S2[7].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + 96) = _v4;
          }
          r_h[0] = S2[8].x;
          r_h[1] = S2[8].y;
          r_h[2] = S2[9].x;
          r_h[3] = S2[9].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2) = _v4;
          }
          r_h[0] = S2[10].x;
          r_h[1] = S2[10].y;
          r_h[2] = S2[11].x;
          r_h[3] = S2[11].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 32) = _v4;
          }
          r_h[0] = S2[12].x;
          r_h[1] = S2[12].y;
          r_h[2] = S2[13].x;
          r_h[3] = S2[13].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 64) = _v4;
          }
          r_h[0] = S2[14].x;
          r_h[1] = S2[14].y;
          r_h[2] = S2[15].x;
          r_h[3] = S2[15].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 96) = _v4;
          }
        }
      }
    }
  }
}

}  // extern "C"
