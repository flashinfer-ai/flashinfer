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
#define SMEM_SK_STAGE_BYTES 1536
#define SMEM_SK_STRIDE 1536
#define SMEM_SQ_OFF 1536
#define SMEM_SQ_STAGE_BYTES 512
#define SMEM_SQ_STRIDE 512
#define SMEM_SV_OFF 2048
#define SMEM_SV_STAGE_BYTES 768
#define SMEM_SV_STRIDE 768
#define SMEM_SALPHA_OFF 2816
#define SMEM_SALPHA_STAGE_BYTES 4
#define SMEM_SALPHA_STRIDE 4
#define SMEM_SBETA_OFF 2820
#define SMEM_SBETA_STAGE_BYTES 12
#define SMEM_SBETA_STRIDE 12
#define SMEM_SSSM_OFF 2832
#define SMEM_SSSM_STAGE_BYTES 4
#define SMEM_SSSM_STRIDE 4
#define SMEM_TOTAL 2944
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 5

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

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void kernel_cake_gdp_4d40c98af90453032200(
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
  float* sQ = reinterpret_cast<float*>(smem_raw + 1536);
  const int sQ_addr = smem + 1536;
  float* sV = reinterpret_cast<float*>(smem_raw + 2048);
  const int sV_addr = smem + 2048;
  float* sAlpha = reinterpret_cast<float*>(smem_raw + 2816);
  const int sAlpha_addr = smem + 2816;
  float* sBeta = reinterpret_cast<float*>(smem_raw + 2820);
  const int sBeta_addr = smem + 2820;
  int* sSsm = reinterpret_cast<int*>(smem_raw + 2832);
  const int sSsm_addr = smem + 2832;

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
  int q_warp_local = (warp_local + 1) % 4;
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
  int steps = 3;
  float r_q[4];
  float r_k[4];
  unsigned int w_k[2];
  unsigned int w_v[2];
  unsigned int w_b[1];
  unsigned int w_q[2];
  unsigned int w_a[1];
  float r_a[3];
  int m = warp_local;
  if (m < 3) {
    long long dk_index =
        (long long)n * k_stride_p0 + (long long)m * k_stride_p1 + (long long)h * k_stride_p2;
    {
      uint2 _vld_0;
      _vld_0 = *reinterpret_cast<const uint2*>(k + dk_index + (long long)k_col0);
      uint2* _vdst_0 = reinterpret_cast<uint2*>(&w_k[0]);
      *_vdst_0 = _vld_0;
    }
    long long dv_index =
        (long long)n * v_stride_p0 + (long long)m * v_stride_p1 + (long long)h * v_stride_p2;
    int dv_col = lane_local * 2;
    {
      uint32_t _bf16_bits_1;
      _bf16_bits_1 = *reinterpret_cast<const uint16_t*>(v + dv_index + (long long)dv_col);
      w_v[0] = _bf16_bits_1;
    }
    {
      uint32_t _bf16_bits_2;
      _bf16_bits_2 = *reinterpret_cast<const uint16_t*>(v + dv_index + (long long)dv_col + 1);
      w_v[1] = _bf16_bits_2;
    }
    long long db_index =
        (long long)n * b_stride_p0 + (long long)m * b_stride_p1 + (long long)h * b_stride_p2;
    {
      uint32_t _bf16_bits_3;
      _bf16_bits_3 = *reinterpret_cast<const uint16_t*>(b + db_index);
      w_b[0] = _bf16_bits_3;
    }
  }
  if (q_warp_local == 0) {
    int t0 = 0;
    long long dq_index =
        (long long)n * q_stride_p0 + (long long)t0 * q_stride_p1 + (long long)h * q_stride_p2;
    {
      uint2 _vld_4;
      _vld_4 = *reinterpret_cast<const uint2*>(q + dq_index + (long long)k_col0);
      uint2* _vdst_4 = reinterpret_cast<uint2*>(&w_q[0]);
      *_vdst_4 = _vld_4;
    }
    long long dg_index =
        (long long)n * a_stride_p0 + (long long)t0 * a_stride_p1 + (long long)h * a_stride_p2;
    {
      uint32_t _bf16_bits_5;
      _bf16_bits_5 = *reinterpret_cast<const uint16_t*>(a + dg_index);
      w_a[0] = _bf16_bits_5;
    }
    r_a[1] = dt_bias[h];
    r_a[2] = A_log[h];
  }
  if (read_state_slot_valid != 0) {
    float S[64];
    float2 S2[32];
    float k_regs[16];
    float2 k2[8];
    float v_regs[4];
    float o4[4];
    float r_v[2];
    float r_b[1];
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
    int m_0 = warp_local;
    if (m_0 < steps) {
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&r_k[_pair * 2])[0]), "=f"((&r_k[_pair * 2])[1])
            : "r"(w_k[_pair]));
      }
      float dpb_dv[4];
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&dpb_dv[_pair * 2])[0]), "=f"((&dpb_dv[_pair * 2])[1])
            : "r"(w_v[_pair]));
      }
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&dpb_dv[2 + _pair * 2])[0]), "=f"((&dpb_dv[2 + _pair * 2])[1])
            : "r"(w_v[1 + _pair]));
      }
      float dpb_db[2];
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&dpb_db[_pair * 2])[0]), "=f"((&dpb_db[_pair * 2])[1])
            : "r"(w_b[_pair]));
      }
      r_v[0] = dpb_dv[0];
      r_v[1] = dpb_dv[2];
      r_b[0] = dpb_db[0];
      float2 k_pair0 = make_float2(r_k[0], r_k[1]);
      float2 k_pair1 = make_float2(r_k[2], r_k[3]);
      float2 sum_k_pair = fma_f32x2_rn_ftz(k_pair0, k_pair0, make_float2(0.0f, 0.0f));
      sum_k_pair = fma_f32x2_rn_ftz(k_pair1, k_pair1, sum_k_pair);
      float sum_k = sum_k_pair.x + sum_k_pair.y;
      float _warp_reduce_22 = sum_k;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_22 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_22, offset);
      float k_norm = rsqrtf(_warp_reduce_22 + 1e-06f);
      float2 k_norm_pair = make_float2(k_norm, k_norm);
      k_pair0 = mul_f32x2(k_pair0, k_norm_pair);
      k_pair1 = mul_f32x2(k_pair1, k_norm_pair);
      sK[m_0 * 128 + k_col0] = k_pair0.x;
      sK[m_0 * 128 + k_col0 + 1] = k_pair0.y;
      sK[m_0 * 128 + k_col0 + 2] = k_pair1.x;
      sK[m_0 * 128 + k_col0 + 3] = k_pair1.y;
      int v_col = lane_local * 2;
      sV[m_0 * 64 + v_col] = r_v[0];
      sV[m_0 * 64 + v_col + 1] = r_v[1];
      if (lane_local == 0) {
        float _exp_24 = __expf(-r_b[0]);
        float _rcp_23 = approx_rcp(1.0f + _exp_24);
        sBeta[m_0] = _rcp_23;
      }
    }
    if (q_warp_local == 0) {
      int t0_1 = 0;
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&r_q[_pair * 2])[0]), "=f"((&r_q[_pair * 2])[1])
            : "r"(w_q[_pair]));
      }
      float dpb_da[2];
#pragma unroll
      for (int _pair = 0; _pair < 1; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&dpb_da[_pair * 2])[0]), "=f"((&dpb_da[_pair * 2])[1])
            : "r"(w_a[_pair]));
      }
      r_a[0] = dpb_da[0];
      float2 q_raw_pair0 = make_float2(r_q[0], r_q[1]);
      float2 q_raw_pair1 = make_float2(r_q[2], r_q[3]);
      float q_norm = scale;
      float2 sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair0, q_raw_pair0, make_float2(0.0f, 0.0f));
      sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair1, q_raw_pair1, sum_q_pair);
      float sum_q = sum_q_pair.x + sum_q_pair.y;
      float _warp_reduce_25 = sum_q;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_25 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_25, offset);
      float _rsqrt_0 = rsqrtf(_warp_reduce_25 + 1e-06f);
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
      sQ[t0_1 * 128 + k_col0] = q_pair0.x;
      sQ[t0_1 * 128 + k_col0 + 1] = q_pair0.y;
      sQ[t0_1 * 128 + k_col0 + 2] = q_pair1.x;
      sQ[t0_1 * 128 + k_col0 + 3] = q_pair1.y;
      float x = r_a[0] + r_a[1];
      float softplus_x = x;
      if (x <= 20.0f) {
        float _exp_27 = __expf(x);
        float _log2_26;
        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_26) : "f"(1.0f + _exp_27));
        softplus_x = _log2_26 * 0.6931472f;
      }
      float a_log = r_a[2];
      {
        float _scope_arg_0 = a_log;
        asm volatile(
            "{\n.reg .f32 _cake_f0;\nmov.f32 _cake_f0, %0;\nmov.f32 _cake_f0, _cake_f0;\nmov.f32 "
            "%0, _cake_f0;\n}"
            : "+f"(_scope_arg_0)
            :);
        a_log = _scope_arg_0;
      }
      float _exp_28 = __expf(a_log);
      float g_log = (-_exp_28) * softplus_x;
      float alpha = __expf(g_log);
      if (lane_local == 0) {
        sAlpha[t0_1] = alpha;
        if (use_ssm_state_indices != 0) {
          sSsm[t0_1] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 +
                                         (long long)t0_1 * ssm_state_indices_stride_p1];
        }
        if (use_ssm_state_indices == 0) {
          sSsm[t0_1] = -1;
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
#pragma unroll 1
    for (int t = 0; t < 1; t++) {
      float alphap = sAlpha[t];
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
      int mp0 = t * 3;
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
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                   : "r"(sV_addr + (unsigned int)(mp0 * 256) + group_byte));
      float betap0 = sBeta[mp0];
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
      float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 1);
      dotp0_0 = dotp0_0 + _shfl_xor_sync_29;
      float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 2);
      dotp0_0 = dotp0_0 + _shfl_xor_sync_30;
      float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 4);
      dotp0_0 = dotp0_0 + _shfl_xor_sync_31;
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
      accp0_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accp0_1_0);
      accp0_1_0 = fma_f32x2_rn_ftz(S2[9], k2[1], accp0_1_0);
      accp0_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accp0_1_0);
      accp0_1_0 = fma_f32x2_rn_ftz(S2[11], k2[3], accp0_1_0);
      accp0_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accp0_1_0);
      accp0_1_0 = fma_f32x2_rn_ftz(S2[13], k2[5], accp0_1_0);
      accp0_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accp0_1_0);
      accp0_1_0 = fma_f32x2_rn_ftz(S2[15], k2[7], accp0_1_0);
      float dotp0_1 = accp0_1_0.x + accp0_1_0.y;
      float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 1);
      dotp0_1 = dotp0_1 + _shfl_xor_sync_32;
      float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 2);
      dotp0_1 = dotp0_1 + _shfl_xor_sync_33;
      float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 4);
      dotp0_1 = dotp0_1 + _shfl_xor_sync_34;
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
      float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, dotp0_2, 1);
      dotp0_2 = dotp0_2 + _shfl_xor_sync_35;
      float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, dotp0_2, 2);
      dotp0_2 = dotp0_2 + _shfl_xor_sync_36;
      float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, dotp0_2, 4);
      dotp0_2 = dotp0_2 + _shfl_xor_sync_37;
      float vnp0_2 = (v_regs[2] - dotp0_2) * betap0;
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
      float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, dotp0_3, 1);
      dotp0_3 = dotp0_3 + _shfl_xor_sync_38;
      float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, dotp0_3, 2);
      dotp0_3 = dotp0_3 + _shfl_xor_sync_39;
      float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, dotp0_3, 4);
      dotp0_3 = dotp0_3 + _shfl_xor_sync_40;
      float vnp0_3 = (v_regs[3] - dotp0_3) * betap0;
      float2 vn2p0_3 = make_float2(vnp0_3, vnp0_3);
      S2[24] = fma_f32x2_rn_ftz(k2[0], vn2p0_3, S2[24]);
      S2[25] = fma_f32x2_rn_ftz(k2[1], vn2p0_3, S2[25]);
      S2[26] = fma_f32x2_rn_ftz(k2[2], vn2p0_3, S2[26]);
      S2[27] = fma_f32x2_rn_ftz(k2[3], vn2p0_3, S2[27]);
      S2[28] = fma_f32x2_rn_ftz(k2[4], vn2p0_3, S2[28]);
      S2[29] = fma_f32x2_rn_ftz(k2[5], vn2p0_3, S2[29]);
      S2[30] = fma_f32x2_rn_ftz(k2[6], vn2p0_3, S2[30]);
      S2[31] = fma_f32x2_rn_ftz(k2[7], vn2p0_3, S2[31]);
      int mp1 = t * 3 + 1;
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
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                   : "r"(sV_addr + (unsigned int)(mp1 * 256) + group_byte));
      float betap1 = sBeta[mp1];
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
      float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 1);
      dotp1_0 = dotp1_0 + _shfl_xor_sync_41;
      float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 2);
      dotp1_0 = dotp1_0 + _shfl_xor_sync_42;
      float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 4);
      dotp1_0 = dotp1_0 + _shfl_xor_sync_43;
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
      accp1_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accp1_1_0);
      accp1_1_0 = fma_f32x2_rn_ftz(S2[9], k2[1], accp1_1_0);
      accp1_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accp1_1_0);
      accp1_1_0 = fma_f32x2_rn_ftz(S2[11], k2[3], accp1_1_0);
      accp1_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accp1_1_0);
      accp1_1_0 = fma_f32x2_rn_ftz(S2[13], k2[5], accp1_1_0);
      accp1_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accp1_1_0);
      accp1_1_0 = fma_f32x2_rn_ftz(S2[15], k2[7], accp1_1_0);
      float dotp1_1 = accp1_1_0.x + accp1_1_0.y;
      float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 1);
      dotp1_1 = dotp1_1 + _shfl_xor_sync_44;
      float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 2);
      dotp1_1 = dotp1_1 + _shfl_xor_sync_45;
      float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 4);
      dotp1_1 = dotp1_1 + _shfl_xor_sync_46;
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
      float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, dotp1_2, 1);
      dotp1_2 = dotp1_2 + _shfl_xor_sync_47;
      float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, dotp1_2, 2);
      dotp1_2 = dotp1_2 + _shfl_xor_sync_48;
      float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, dotp1_2, 4);
      dotp1_2 = dotp1_2 + _shfl_xor_sync_49;
      float vnp1_2 = (v_regs[2] - dotp1_2) * betap1;
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
      float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, dotp1_3, 1);
      dotp1_3 = dotp1_3 + _shfl_xor_sync_50;
      float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, dotp1_3, 2);
      dotp1_3 = dotp1_3 + _shfl_xor_sync_51;
      float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, dotp1_3, 4);
      dotp1_3 = dotp1_3 + _shfl_xor_sync_52;
      float vnp1_3 = (v_regs[3] - dotp1_3) * betap1;
      float2 vn2p1_3 = make_float2(vnp1_3, vnp1_3);
      S2[24] = fma_f32x2_rn_ftz(k2[0], vn2p1_3, S2[24]);
      S2[25] = fma_f32x2_rn_ftz(k2[1], vn2p1_3, S2[25]);
      S2[26] = fma_f32x2_rn_ftz(k2[2], vn2p1_3, S2[26]);
      S2[27] = fma_f32x2_rn_ftz(k2[3], vn2p1_3, S2[27]);
      S2[28] = fma_f32x2_rn_ftz(k2[4], vn2p1_3, S2[28]);
      S2[29] = fma_f32x2_rn_ftz(k2[5], vn2p1_3, S2[29]);
      S2[30] = fma_f32x2_rn_ftz(k2[6], vn2p1_3, S2[30]);
      S2[31] = fma_f32x2_rn_ftz(k2[7], vn2p1_3, S2[31]);
      int mp2 = t * 3 + 2;
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
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                   : "r"(sV_addr + (unsigned int)(mp2 * 256) + group_byte));
      float betap2 = sBeta[mp2];
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
      float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 1);
      dotp2_0 = dotp2_0 + _shfl_xor_sync_53;
      float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 2);
      dotp2_0 = dotp2_0 + _shfl_xor_sync_54;
      float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 4);
      dotp2_0 = dotp2_0 + _shfl_xor_sync_55;
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
      accp2_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accp2_1_0);
      accp2_1_0 = fma_f32x2_rn_ftz(S2[9], k2[1], accp2_1_0);
      accp2_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accp2_1_0);
      accp2_1_0 = fma_f32x2_rn_ftz(S2[11], k2[3], accp2_1_0);
      accp2_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accp2_1_0);
      accp2_1_0 = fma_f32x2_rn_ftz(S2[13], k2[5], accp2_1_0);
      accp2_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accp2_1_0);
      accp2_1_0 = fma_f32x2_rn_ftz(S2[15], k2[7], accp2_1_0);
      float dotp2_1 = accp2_1_0.x + accp2_1_0.y;
      float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 1);
      dotp2_1 = dotp2_1 + _shfl_xor_sync_56;
      float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 2);
      dotp2_1 = dotp2_1 + _shfl_xor_sync_57;
      float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 4);
      dotp2_1 = dotp2_1 + _shfl_xor_sync_58;
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
      float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, dotp2_2, 1);
      dotp2_2 = dotp2_2 + _shfl_xor_sync_59;
      float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, dotp2_2, 2);
      dotp2_2 = dotp2_2 + _shfl_xor_sync_60;
      float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, dotp2_2, 4);
      dotp2_2 = dotp2_2 + _shfl_xor_sync_61;
      float vnp2_2 = (v_regs[2] - dotp2_2) * betap2;
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
      float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, dotp2_3, 1);
      dotp2_3 = dotp2_3 + _shfl_xor_sync_62;
      float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, dotp2_3, 2);
      dotp2_3 = dotp2_3 + _shfl_xor_sync_63;
      float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, dotp2_3, 4);
      dotp2_3 = dotp2_3 + _shfl_xor_sync_64;
      float vnp2_3 = (v_regs[3] - dotp2_3) * betap2;
      float2 vn2p2_3 = make_float2(vnp2_3, vnp2_3);
      S2[24] = fma_f32x2_rn_ftz(k2[0], vn2p2_3, S2[24]);
      S2[25] = fma_f32x2_rn_ftz(k2[1], vn2p2_3, S2[25]);
      S2[26] = fma_f32x2_rn_ftz(k2[2], vn2p2_3, S2[26]);
      S2[27] = fma_f32x2_rn_ftz(k2[3], vn2p2_3, S2[27]);
      S2[28] = fma_f32x2_rn_ftz(k2[4], vn2p2_3, S2[28]);
      S2[29] = fma_f32x2_rn_ftz(k2[5], vn2p2_3, S2[29]);
      S2[30] = fma_f32x2_rn_ftz(k2[6], vn2p2_3, S2[30]);
      S2[31] = fma_f32x2_rn_ftz(k2[7], vn2p2_3, S2[31]);
      unsigned int qbasep = sQ_addr + (unsigned int)(t * 512) + lane_byte;
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
      accpq_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accpq_0_0);
      accpq_0_0 = fma_f32x2_rn_ftz(S2[1], k2[1], accpq_0_0);
      accpq_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accpq_0_0);
      accpq_0_0 = fma_f32x2_rn_ftz(S2[3], k2[3], accpq_0_0);
      accpq_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accpq_0_0);
      accpq_0_0 = fma_f32x2_rn_ftz(S2[5], k2[5], accpq_0_0);
      accpq_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accpq_0_0);
      accpq_0_0 = fma_f32x2_rn_ftz(S2[7], k2[7], accpq_0_0);
      float dotpq_0 = accpq_0_0.x + accpq_0_0.y;
      float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 1);
      dotpq_0 = dotpq_0 + _shfl_xor_sync_65;
      float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 2);
      dotpq_0 = dotpq_0 + _shfl_xor_sync_66;
      float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 4);
      dotpq_0 = dotpq_0 + _shfl_xor_sync_67;
      o4[0] = dotpq_0;
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
      float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 1);
      dotpq_1 = dotpq_1 + _shfl_xor_sync_68;
      float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 2);
      dotpq_1 = dotpq_1 + _shfl_xor_sync_69;
      float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 4);
      dotpq_1 = dotpq_1 + _shfl_xor_sync_70;
      o4[1] = dotpq_1;
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
      float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, dotpq_2, 1);
      dotpq_2 = dotpq_2 + _shfl_xor_sync_71;
      float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, dotpq_2, 2);
      dotpq_2 = dotpq_2 + _shfl_xor_sync_72;
      float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, dotpq_2, 4);
      dotpq_2 = dotpq_2 + _shfl_xor_sync_73;
      o4[2] = dotpq_2;
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
      float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, dotpq_3, 1);
      dotpq_3 = dotpq_3 + _shfl_xor_sync_74;
      float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, dotpq_3, 2);
      dotpq_3 = dotpq_3 + _shfl_xor_sync_75;
      float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, dotpq_3, 4);
      dotpq_3 = dotpq_3 + _shfl_xor_sync_76;
      o4[3] = dotpq_3;
      if (u == 0) {
        long long out_index = (long long)n * out_stride_p0 + (long long)t * out_stride_p1 +
                              (long long)h * out_stride_p2 + (long long)row0;
        unsigned int _packed_0[2];
#pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
          __nv_bfloat162 _bf2 =
              __float22bfloat162_rn(make_float2(o4[_lp * 2 + 0], o4[_lp * 2 + 1 + 0]));
          _packed_0[_lp] = *(uint32_t*)&_bf2;
        }
#pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
          __nv_bfloat162 _bf2 =
              __float22bfloat162_rn(make_float2(o4[_lp * 2 + 2], o4[_lp * 2 + 1 + 2]));
          _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(out + (out_index))[0] = reinterpret_cast<int2*>(_packed_0)[0];
      }
    }
    int ssm_slot0 = sSsm[0];
    int final_slot = ((use_ssm_state_indices != 0) ? ssm_slot0 : write_state_slot_raw);
    int final_valid = ((final_slot >= 0 && state_pool_size > (long long)final_slot &&
                        (use_ssm_state_indices != 0 || disable_state_update == 0))
                           ? 1
                           : 0);
    if (final_valid != 0) {
      long long final_base = (long long)final_slot * state_stride_p0 + head_rows_off;
      r_h[0] = S2[0].x;
      r_h[1] = S2[0].y;
      r_h[2] = S2[1].x;
      r_h[3] = S2[1].y;
      {
        float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
        *reinterpret_cast<float4*>(state + final_base) = _v4;
      }
      r_h[0] = S2[2].x;
      r_h[1] = S2[2].y;
      r_h[2] = S2[3].x;
      r_h[3] = S2[3].y;
      {
        float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
        *reinterpret_cast<float4*>(state + final_base + 32) = _v4;
      }
      r_h[0] = S2[4].x;
      r_h[1] = S2[4].y;
      r_h[2] = S2[5].x;
      r_h[3] = S2[5].y;
      {
        float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
        *reinterpret_cast<float4*>(state + final_base + 64) = _v4;
      }
      r_h[0] = S2[6].x;
      r_h[1] = S2[6].y;
      r_h[2] = S2[7].x;
      r_h[3] = S2[7].y;
      {
        float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
        *reinterpret_cast<float4*>(state + final_base + 96) = _v4;
      }
      r_h[0] = S2[8].x;
      r_h[1] = S2[8].y;
      r_h[2] = S2[9].x;
      r_h[3] = S2[9].y;
      {
        float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
        *reinterpret_cast<float4*>(state + final_base + state_stride_p2) = _v4;
      }
      r_h[0] = S2[10].x;
      r_h[1] = S2[10].y;
      r_h[2] = S2[11].x;
      r_h[3] = S2[11].y;
      {
        float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
        *reinterpret_cast<float4*>(state + final_base + state_stride_p2 + 32) = _v4;
      }
      r_h[0] = S2[12].x;
      r_h[1] = S2[12].y;
      r_h[2] = S2[13].x;
      r_h[3] = S2[13].y;
      {
        float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
        *reinterpret_cast<float4*>(state + final_base + state_stride_p2 + 64) = _v4;
      }
      r_h[0] = S2[14].x;
      r_h[1] = S2[14].y;
      r_h[2] = S2[15].x;
      r_h[3] = S2[15].y;
      {
        float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
        *reinterpret_cast<float4*>(state + final_base + state_stride_p2 + 96) = _v4;
      }
      r_h[0] = S2[16].x;
      r_h[1] = S2[16].y;
      r_h[2] = S2[17].x;
      r_h[3] = S2[17].y;
      {
        float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
        *reinterpret_cast<float4*>(state + final_base + 2 * state_stride_p2) = _v4;
      }
      r_h[0] = S2[18].x;
      r_h[1] = S2[18].y;
      r_h[2] = S2[19].x;
      r_h[3] = S2[19].y;
      {
        float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
        *reinterpret_cast<float4*>(state + final_base + 2 * state_stride_p2 + 32) = _v4;
      }
      r_h[0] = S2[20].x;
      r_h[1] = S2[20].y;
      r_h[2] = S2[21].x;
      r_h[3] = S2[21].y;
      {
        float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
        *reinterpret_cast<float4*>(state + final_base + 2 * state_stride_p2 + 64) = _v4;
      }
      r_h[0] = S2[22].x;
      r_h[1] = S2[22].y;
      r_h[2] = S2[23].x;
      r_h[3] = S2[23].y;
      {
        float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
        *reinterpret_cast<float4*>(state + final_base + 2 * state_stride_p2 + 96) = _v4;
      }
      r_h[0] = S2[24].x;
      r_h[1] = S2[24].y;
      r_h[2] = S2[25].x;
      r_h[3] = S2[25].y;
      {
        float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
        *reinterpret_cast<float4*>(state + final_base + 3 * state_stride_p2) = _v4;
      }
      r_h[0] = S2[26].x;
      r_h[1] = S2[26].y;
      r_h[2] = S2[27].x;
      r_h[3] = S2[27].y;
      {
        float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
        *reinterpret_cast<float4*>(state + final_base + 3 * state_stride_p2 + 32) = _v4;
      }
      r_h[0] = S2[28].x;
      r_h[1] = S2[28].y;
      r_h[2] = S2[29].x;
      r_h[3] = S2[29].y;
      {
        float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
        *reinterpret_cast<float4*>(state + final_base + 3 * state_stride_p2 + 64) = _v4;
      }
      r_h[0] = S2[30].x;
      r_h[1] = S2[30].y;
      r_h[2] = S2[31].x;
      r_h[3] = S2[31].y;
      {
        float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
        *reinterpret_cast<float4*>(state + final_base + 3 * state_stride_p2 + 96) = _v4;
      }
    }
  }
}

}  // extern "C"
