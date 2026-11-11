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
#define SMEM_SQ_OFF 0
#define SMEM_SQ_STAGE_BYTES 512
#define SMEM_SQ_STRIDE 512
#define SMEM_SK_OFF 512
#define SMEM_SK_STAGE_BYTES 1536
#define SMEM_SK_STRIDE 1536
#define SMEM_SV_OFF 2048
#define SMEM_SV_STAGE_BYTES 96
#define SMEM_SV_STRIDE 96
#define SMEM_SALPHA_OFF 2144
#define SMEM_SALPHA_STAGE_BYTES 4
#define SMEM_SALPHA_STRIDE 4
#define SMEM_SBETA_OFF 2148
#define SMEM_SBETA_STAGE_BYTES 12
#define SMEM_SBETA_STRIDE 12
#define SMEM_TOTAL 2304
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ float approx_rcp(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
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

__global__ __launch_bounds__(128) void kernel_cake_gdp_f2f0c7abbf10082a6996(
    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v,
    float* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a,
    float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out,
    int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices,
    int* __restrict__ ssm_state_indices, CakeFastDivmod heads, int num_tokens, float scale,
    int disable_state_update, int use_ssm_state_indices, long long state_size_p0,
    long long state_stride_p0, long long state_stride_p1, long long k_stride_p0,
    long long k_stride_p1, long long k_stride_p2, long long v_stride_p0, long long v_stride_p1,
    long long v_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2,
    long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long a_stride_p0,
    long long a_stride_p1, long long a_stride_p2, long long state_stride_p2,
    long long out_stride_p0, long long out_stride_p1, long long out_stride_p2,
    long long ssm_state_indices_stride_p0, long long ssm_state_indices_stride_p1) {
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
  float* sQ = reinterpret_cast<float*>(smem_raw + 0);
  const int sQ_addr = smem + 0;
  float* sK = reinterpret_cast<float*>(smem_raw + 512);
  const int sK_addr = smem + 512;
  float* sV = reinterpret_cast<float*>(smem_raw + 2048);
  const int sV_addr = smem + 2048;
  float* sAlpha = reinterpret_cast<float*>(smem_raw + 2144);
  const int sAlpha_addr = smem + 2144;
  float* sBeta = reinterpret_cast<float*>(smem_raw + 2148);
  const int sBeta_addr = smem + 2148;

  // === Task calls (declaration order) ===
  int linear_block = blockIdx.x;
  int state_head = linear_block / 8;
  int split = linear_block - state_head * 8;
  uint32_t _fast_div_q_0 =
      (heads.divisor != 1)
          ? (__umulhi((uint32_t)((unsigned int)state_head), heads.multiplier) >> heads.shift_right)
          : (uint32_t)((unsigned int)state_head);
  uint32_t _fast_div_r_0 =
      (uint32_t)((unsigned int)state_head) - _fast_div_q_0 * (uint32_t)(heads.divisor);
  int n = (int)_fast_div_q_0;
  int h = (int)_fast_div_r_0;
  int lane_local = lane;
  int warp_local = warp;
  long long state_pool_size = state_size_p0;
  int read_state_slot_raw = initial_state_indices[n];
  int read_state_slot_valid =
      ((read_state_slot_raw >= 0 && state_pool_size > (long long)read_state_slot_raw) ? 1 : 0);
  int write_state_slot_raw = output_state_indices[n];
  int write_state_slot_valid =
      ((write_state_slot_raw >= 0 && state_pool_size > (long long)write_state_slot_raw) ? 1 : 0);
  long long read_state_head_base =
      (long long)read_state_slot_raw * state_stride_p0 + (long long)h * state_stride_p1;
  long long write_state_head_base =
      (long long)write_state_slot_raw * state_stride_p0 + (long long)h * state_stride_p1;
  int split_v_base = split * 8;
  int k_col0 = lane_local * 4;
  int steps = 3;
  int pre_m = ((warp_local) < (steps - 1) ? (warp_local) : (steps - 1));
  int pre_q_shift = warp_local + 1;
  int pre_q_first = ((pre_q_shift < 4) ? pre_q_shift : pre_q_shift - 4);
  int pre_t = ((pre_q_first) < (num_tokens - 1) ? (pre_q_first) : (num_tokens - 1));
  float pre_k[4];
  float pre_q[4];
  float pre_s[5];
  long long pre_k_index =
      (long long)n * k_stride_p0 + (long long)pre_m * k_stride_p1 + (long long)h * k_stride_p2;
  {
    uint2 _vld_0;
    _vld_0 = *reinterpret_cast<const uint2*>(k + pre_k_index + (long long)k_col0);
    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
#pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
      (&pre_k[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
      (&pre_k[0 + _pair * 2])[1] =
          __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
    }
  }
  int pre_v_lane = ((lane_local < 8) ? lane_local : 7);
  long long pre_v_index =
      (long long)n * v_stride_p0 + (long long)pre_m * v_stride_p1 + (long long)h * v_stride_p2;
  pre_s[0] = (float)v[pre_v_index + (long long)split_v_base + (long long)pre_v_lane];
  long long pre_b_index =
      (long long)n * b_stride_p0 + (long long)pre_m * b_stride_p1 + (long long)h * b_stride_p2;
  pre_s[1] = (float)b[pre_b_index];
  long long pre_q_index =
      (long long)n * q_stride_p0 + (long long)pre_t * q_stride_p1 + (long long)h * q_stride_p2;
  {
    uint2 _vld_1;
    _vld_1 = *reinterpret_cast<const uint2*>(q + pre_q_index + (long long)k_col0);
    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
#pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
      (&pre_q[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
      (&pre_q[0 + _pair * 2])[1] =
          __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
    }
  }
  long long pre_gate_index =
      (long long)n * a_stride_p0 + (long long)pre_t * a_stride_p1 + (long long)h * a_stride_p2;
  pre_s[2] = (float)a[pre_gate_index];
  pre_s[3] = dt_bias[h];
  pre_s[4] = A_log[h];
  if (read_state_slot_valid != 0) {
    float r_q[4];
    float r_k[4];
    float r_h[4];
    int row_own = lane_local >> 4 & 1;
    int row_a = warp_local + row_own * 4;
    int row_b = warp_local + 4 - row_own * 4;
    int v_row_a = split_v_base + row_a;
    int v_row_b = split_v_base + row_b;
    float raw_a[4];
    float raw_b[4];
    long long raw_row_a = read_state_head_base + (long long)v_row_a * state_stride_p2;
    long long raw_row_b = read_state_head_base + (long long)v_row_b * state_stride_p2;
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + raw_row_a + (long long)k_col0);
      raw_a[0 + 0] = _v4.x;
      raw_a[0 + 1] = _v4.y;
      raw_a[0 + 2] = _v4.z;
      raw_a[0 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + raw_row_b + (long long)k_col0);
      raw_b[0 + 0] = _v4.x;
      raw_b[0 + 1] = _v4.y;
      raw_b[0 + 2] = _v4.z;
      raw_b[0 + 3] = _v4.w;
    }
    if (warp_local < steps) {
      float2 k_pair0 = make_float2(pre_k[0], pre_k[1]);
      float2 k_pair1 = make_float2(pre_k[2], pre_k[3]);
      float2 sum_k_pair = fma_f32x2_rn_ftz(k_pair0, k_pair0, make_float2(0.0f, 0.0f));
      sum_k_pair = fma_f32x2_rn_ftz(k_pair1, k_pair1, sum_k_pair);
      float sum_k = sum_k_pair.x + sum_k_pair.y;
      float _warp_reduce_4 = sum_k;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_4 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_4, offset);
      sum_k = _warp_reduce_4;
      float k_norm = rsqrtf(sum_k + 1e-06f);
      float2 k_norm_pair = make_float2(k_norm, k_norm);
      k_pair0 = mul_f32x2(k_pair0, k_norm_pair);
      k_pair1 = mul_f32x2(k_pair1, k_norm_pair);
      sK[pre_m * 128 + k_col0] = k_pair0.x;
      sK[pre_m * 128 + k_col0 + 1] = k_pair0.y;
      sK[pre_m * 128 + k_col0 + 2] = k_pair1.x;
      sK[pre_m * 128 + k_col0 + 3] = k_pair1.y;
      if (lane_local < 8) {
        sV[pre_m * 8 + lane_local] = pre_s[0];
      }
      if (lane_local == 0) {
        float _exp_6 = __expf(-pre_s[1]);
        float _rcp_5 = approx_rcp(1.0f + _exp_6);
        sBeta[pre_m] = _rcp_5;
      }
    }
    if (pre_q_first < num_tokens) {
      float2 q_raw_pair0 = make_float2(pre_q[0], pre_q[1]);
      float2 q_raw_pair1 = make_float2(pre_q[2], pre_q[3]);
      float q_norm = scale;
      float2 sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair0, q_raw_pair0, make_float2(0.0f, 0.0f));
      sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair1, q_raw_pair1, sum_q_pair);
      float sum_q = sum_q_pair.x + sum_q_pair.y;
      float _warp_reduce_7 = sum_q;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_7 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_7, offset);
      sum_q = _warp_reduce_7;
      float _rsqrt_8 = rsqrtf(sum_q + 1e-06f);
      q_norm = _rsqrt_8 * scale;
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
      sQ[pre_t * 128 + k_col0] = q_pair0.x;
      sQ[pre_t * 128 + k_col0 + 1] = q_pair0.y;
      sQ[pre_t * 128 + k_col0 + 2] = q_pair1.x;
      sQ[pre_t * 128 + k_col0 + 3] = q_pair1.y;
      if (lane_local == 0) {
        float x = pre_s[2] + pre_s[3];
        float softplus_x = x;
        if (x <= 20.0f) {
          float _exp_10 = __expf(x);
          float _log2_9;
          asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_9) : "f"(1.0f + _exp_10));
          softplus_x = _log2_9 * 0.6931472f;
        }
        float _exp_11 = __expf(pre_s[4]);
        float g_log = (-_exp_11) * softplus_x;
        float _exp_12 = __expf(g_log);
        sAlpha[pre_t] = _exp_12;
      }
    }
    __syncthreads();
    float2 ha_pair0 = make_float2(raw_a[0], raw_a[1]);
    float2 ha_pair1 = make_float2(raw_a[2], raw_a[3]);
    float2 hb_pair0 = make_float2(raw_b[0], raw_b[1]);
    float2 hb_pair1 = make_float2(raw_b[2], raw_b[3]);
#pragma unroll
    for (int t = 0; t < 1; t++) {
      float decay_val = sAlpha[t];
      float2 decay_pair = make_float2(decay_val, decay_val);
      ha_pair0 = mul_f32x2(ha_pair0, decay_pair);
      ha_pair1 = mul_f32x2(ha_pair1, decay_pair);
      hb_pair0 = mul_f32x2(hb_pair0, decay_pair);
      hb_pair1 = mul_f32x2(hb_pair1, decay_pair);
#pragma unroll
      for (int j = 0; j < 3; j++) {
        int m = t * 3 + j;
        float2 k_pair0_1 = make_float2(sK[m * 128 + k_col0], sK[m * 128 + k_col0 + 1]);
        float2 k_pair1_1 = make_float2(sK[m * 128 + k_col0 + 2], sK[m * 128 + k_col0 + 3]);
        float2 sum_a_pair = fma_f32x2_rn_ftz(ha_pair0, k_pair0_1, make_float2(0.0f, 0.0f));
        sum_a_pair = fma_f32x2_rn_ftz(ha_pair1, k_pair1_1, sum_a_pair);
        float2 sum_b_pair = fma_f32x2_rn_ftz(hb_pair0, k_pair0_1, make_float2(0.0f, 0.0f));
        sum_b_pair = fma_f32x2_rn_ftz(hb_pair1, k_pair1_1, sum_b_pair);
        float sum_a = sum_a_pair.x + sum_a_pair.y;
        float sum_b = sum_b_pair.x + sum_b_pair.y;
        float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, sum_b, 16);
        sum_a = sum_a + _shfl_xor_sync_13;
        float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, sum_a, 8);
        sum_a = sum_a + _shfl_xor_sync_14;
        float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, sum_a, 4);
        sum_a = sum_a + _shfl_xor_sync_15;
        float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, sum_a, 2);
        sum_a = sum_a + _shfl_xor_sync_16;
        float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, sum_a, 1);
        sum_a = sum_a + _shfl_xor_sync_17;
        sum_a = sum_a;
        float v_a = sV[m * 8 + row_a];
        float beta_val = sBeta[m];
        float vn_a = (v_a - sum_a) * beta_val;
        float vn_b = __shfl_xor_sync(0xFFFFFFFF, vn_a, 16);
        float2 vn_a_pair = make_float2(vn_a, vn_a);
        float2 vn_b_pair = make_float2(vn_b, vn_b);
        ha_pair0 = fma_f32x2_rn_ftz(k_pair0_1, vn_a_pair, ha_pair0);
        ha_pair1 = fma_f32x2_rn_ftz(k_pair1_1, vn_a_pair, ha_pair1);
        hb_pair0 = fma_f32x2_rn_ftz(k_pair0_1, vn_b_pair, hb_pair0);
        hb_pair1 = fma_f32x2_rn_ftz(k_pair1_1, vn_b_pair, hb_pair1);
      }
      float2 q_pair0_1 = make_float2(sQ[t * 128 + k_col0], sQ[t * 128 + k_col0 + 1]);
      float2 q_pair1_1 = make_float2(sQ[t * 128 + k_col0 + 2], sQ[t * 128 + k_col0 + 3]);
      float2 sum_hq_a_pair = fma_f32x2_rn_ftz(ha_pair0, q_pair0_1, make_float2(0.0f, 0.0f));
      sum_hq_a_pair = fma_f32x2_rn_ftz(ha_pair1, q_pair1_1, sum_hq_a_pair);
      float2 sum_hq_b_pair = fma_f32x2_rn_ftz(hb_pair0, q_pair0_1, make_float2(0.0f, 0.0f));
      sum_hq_b_pair = fma_f32x2_rn_ftz(hb_pair1, q_pair1_1, sum_hq_b_pair);
      float sum_hq_a = sum_hq_a_pair.x + sum_hq_a_pair.y;
      float sum_hq_b = sum_hq_b_pair.x + sum_hq_b_pair.y;
      float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_b, 16);
      sum_hq_a = sum_hq_a + _shfl_xor_sync_18;
      float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_a, 8);
      sum_hq_a = sum_hq_a + _shfl_xor_sync_19;
      float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_a, 4);
      sum_hq_a = sum_hq_a + _shfl_xor_sync_20;
      float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_a, 2);
      sum_hq_a = sum_hq_a + _shfl_xor_sync_21;
      float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_a, 1);
      sum_hq_a = sum_hq_a + _shfl_xor_sync_22;
      sum_hq_a = sum_hq_a;
      if ((lane_local & 15) == 0) {
        long long out_index = (long long)n * out_stride_p0 + (long long)t * out_stride_p1 +
                              (long long)h * out_stride_p2;
        out[out_index + (long long)v_row_a] = sum_hq_a;
      }
      if (use_ssm_state_indices != 0) {
        int ssm_slot = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 +
                                         (long long)t * ssm_state_indices_stride_p1];
        if (ssm_slot >= 0 && state_pool_size > (long long)ssm_slot) {
          long long snap_base = (long long)ssm_slot * state_stride_p0 +
                                (long long)h * state_stride_p1 + (long long)k_col0;
          r_h[0] = ha_pair0.x;
          r_h[1] = ha_pair0.y;
          r_h[2] = ha_pair1.x;
          r_h[3] = ha_pair1.y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base + (long long)v_row_a * state_stride_p2) =
                _v4;
          }
          r_h[0] = hb_pair0.x;
          r_h[1] = hb_pair0.y;
          r_h[2] = hb_pair1.x;
          r_h[3] = hb_pair1.y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base + (long long)v_row_b * state_stride_p2) =
                _v4;
          }
        }
      }
    }
    if (write_state_slot_valid != 0) {
      if (disable_state_update == 0) {
        if (use_ssm_state_indices == 0) {
          r_h[0] = ha_pair0.x;
          r_h[1] = ha_pair0.y;
          r_h[2] = ha_pair1.x;
          r_h[3] = ha_pair1.y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_state_head_base +
                                       (long long)v_row_a * state_stride_p2 + (long long)k_col0) =
                _v4;
          }
          r_h[0] = hb_pair0.x;
          r_h[1] = hb_pair0.y;
          r_h[2] = hb_pair1.x;
          r_h[3] = hb_pair1.y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + write_state_head_base +
                                       (long long)v_row_b * state_stride_p2 + (long long)k_col0) =
                _v4;
          }
        }
      }
    }
  }
}

}  // extern "C"
