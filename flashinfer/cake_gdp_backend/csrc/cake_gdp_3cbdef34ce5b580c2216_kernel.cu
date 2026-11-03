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
#if __CUDA_ARCH__ == 900
#else
    float2 S2[16];
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
#if __CUDA_ARCH__ == 1000
    float r_v[2];
    float r_b[1];
    unsigned int w_k[2];
    unsigned int w_v[2];
    unsigned int w_b[1];
    unsigned int w_q[2];
    unsigned int w_a[1];
    float r_a[3];
#endif
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
#if __CUDA_ARCH__ == 900
    int bm0 = warp_local;
    float bk0[4];
    float bv0[2];
    float bb0[1];
    long long bki0 =
        (long long)n * k_stride_p0 + (long long)bm0 * k_stride_p1 + (long long)h * k_stride_p2;
    {
      uint2 _vld_8;
      _vld_8 = *reinterpret_cast<const uint2*>(k + bki0 + (long long)k_col0);
      uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&bk0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
        (&bk0[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
#elif __CUDA_ARCH__ == 1000
#pragma unroll
    for (int mm = 0; mm < 3; mm++) {
      int m = mm * 8 + warp_local;
      if ((mm + 1) * 8 <= 24) {
        long long dk_index =
            (long long)n * k_stride_p0 + (long long)m * k_stride_p1 + (long long)h * k_stride_p2;
        {
          uint2 _vld_8;
          _vld_8 = *reinterpret_cast<const uint2*>(k + dk_index + (long long)k_col0);
          uint2* _vdst_8 = reinterpret_cast<uint2*>(&w_k[0]);
          *_vdst_8 = _vld_8;
        }
        long long dv_index =
            (long long)n * v_stride_p0 + (long long)m * v_stride_p1 + (long long)h * v_stride_p2;
        int dv_col = lane_local * 2;
        {
          uint32_t _bf16_bits_9;
          _bf16_bits_9 = *reinterpret_cast<const uint16_t*>(v + dv_index + (long long)dv_col);
          w_v[0] = _bf16_bits_9;
        }
        {
          uint32_t _bf16_bits_10;
          _bf16_bits_10 = *reinterpret_cast<const uint16_t*>(v + dv_index + (long long)dv_col + 1);
          w_v[1] = _bf16_bits_10;
        }
        long long db_index =
            (long long)n * b_stride_p0 + (long long)m * b_stride_p1 + (long long)h * b_stride_p2;
        {
          uint32_t _bf16_bits_11;
          _bf16_bits_11 = *reinterpret_cast<const uint16_t*>(b + db_index);
          w_b[0] = _bf16_bits_11;
        }
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
        float _warp_reduce_12 = sum_k;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
          _warp_reduce_12 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_12, offset);
        float k_norm = rsqrtf(_warp_reduce_12 + 1e-06f);
        float2 k_norm_pair = make_float2(k_norm, k_norm);
        k_pair0 = mul_f32x2(k_pair0, k_norm_pair);
        k_pair1 = mul_f32x2(k_pair1, k_norm_pair);
        sK[m * 128 + k_col0] = k_pair0.x;
        sK[m * 128 + k_col0 + 1] = k_pair0.y;
        sK[m * 128 + k_col0 + 2] = k_pair1.x;
        sK[m * 128 + k_col0 + 3] = k_pair1.y;
        int v_col = lane_local * 2;
        sV[m * 64 + v_col] = r_v[0];
        sV[m * 64 + v_col + 1] = r_v[1];
        if (lane_local == 0) {
          float _exp_14 = __expf(-r_b[0]);
          float _rcp_13 = approx_rcp(1.0f + _exp_14);
          sBeta[m] = _rcp_13;
        }
      }
      if ((mm + 1) * 8 > 24) {
        if (m < steps) {
          long long dk_index_1 =
              (long long)n * k_stride_p0 + (long long)m * k_stride_p1 + (long long)h * k_stride_p2;
          {
            uint2 _vld_15;
            _vld_15 = *reinterpret_cast<const uint2*>(k + dk_index_1 + (long long)k_col0);
            uint2* _vdst_15 = reinterpret_cast<uint2*>(&w_k[0]);
            *_vdst_15 = _vld_15;
          }
          long long dv_index_1 =
              (long long)n * v_stride_p0 + (long long)m * v_stride_p1 + (long long)h * v_stride_p2;
          int dv_col_1 = lane_local * 2;
          {
            uint32_t _bf16_bits_16;
            _bf16_bits_16 =
                *reinterpret_cast<const uint16_t*>(v + dv_index_1 + (long long)dv_col_1);
            w_v[0] = _bf16_bits_16;
          }
          {
            uint32_t _bf16_bits_17;
            _bf16_bits_17 =
                *reinterpret_cast<const uint16_t*>(v + dv_index_1 + (long long)dv_col_1 + 1);
            w_v[1] = _bf16_bits_17;
          }
          long long db_index_1 =
              (long long)n * b_stride_p0 + (long long)m * b_stride_p1 + (long long)h * b_stride_p2;
          {
            uint32_t _bf16_bits_18;
            _bf16_bits_18 = *reinterpret_cast<const uint16_t*>(b + db_index_1);
            w_b[0] = _bf16_bits_18;
          }
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
          float dpb_dv_1[4];
#pragma unroll
          for (int _pair = 0; _pair < 1; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&dpb_dv_1[_pair * 2])[0]), "=f"((&dpb_dv_1[_pair * 2])[1])
                : "r"(w_v[_pair]));
          }
#pragma unroll
          for (int _pair = 0; _pair < 1; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&dpb_dv_1[2 + _pair * 2])[0]), "=f"((&dpb_dv_1[2 + _pair * 2])[1])
                : "r"(w_v[1 + _pair]));
          }
          float dpb_db_1[2];
#pragma unroll
          for (int _pair = 0; _pair < 1; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&dpb_db_1[_pair * 2])[0]), "=f"((&dpb_db_1[_pair * 2])[1])
                : "r"(w_b[_pair]));
          }
          r_v[0] = dpb_dv_1[0];
          r_v[1] = dpb_dv_1[2];
          r_b[0] = dpb_db_1[0];
          float2 k_pair0_1 = make_float2(r_k[0], r_k[1]);
          float2 k_pair1_1 = make_float2(r_k[2], r_k[3]);
          float2 sum_k_pair_1 = fma_f32x2_rn_ftz(k_pair0_1, k_pair0_1, make_float2(0.0f, 0.0f));
          sum_k_pair_1 = fma_f32x2_rn_ftz(k_pair1_1, k_pair1_1, sum_k_pair_1);
          float sum_k_1 = sum_k_pair_1.x + sum_k_pair_1.y;
          float _warp_reduce_19 = sum_k_1;
#pragma unroll
          for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_19 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_19, offset);
          float k_norm_1 = rsqrtf(_warp_reduce_19 + 1e-06f);
          float2 k_norm_pair_1 = make_float2(k_norm_1, k_norm_1);
          k_pair0_1 = mul_f32x2(k_pair0_1, k_norm_pair_1);
          k_pair1_1 = mul_f32x2(k_pair1_1, k_norm_pair_1);
          sK[m * 128 + k_col0] = k_pair0_1.x;
          sK[m * 128 + k_col0 + 1] = k_pair0_1.y;
          sK[m * 128 + k_col0 + 2] = k_pair1_1.x;
          sK[m * 128 + k_col0 + 3] = k_pair1_1.y;
          int v_col_1 = lane_local * 2;
          sV[m * 64 + v_col_1] = r_v[0];
          sV[m * 64 + v_col_1 + 1] = r_v[1];
          if (lane_local == 0) {
            float _exp_21 = __expf(-r_b[0]);
            float _rcp_20 = approx_rcp(1.0f + _exp_21);
            sBeta[m] = _rcp_20;
          }
        }
#else
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
#endif
      }
    }
#if __CUDA_ARCH__ == 900
    long long bvi0 =
        (long long)n * v_stride_p0 + (long long)bm0 * v_stride_p1 + (long long)h * v_stride_p2;
    int bvc0 = lane_local * 2;
    bv0[0] = (float)v[bvi0 + (long long)bvc0];
    bv0[1] = (float)v[bvi0 + (long long)bvc0 + 1];
    long long bbi0 =
        (long long)n * b_stride_p0 + (long long)bm0 * b_stride_p1 + (long long)h * b_stride_p2;
    bb0[0] = (float)b[bbi0];
    int bm1 = 8 + warp_local;
    float bk1[4];
    float bv1[2];
    float bb1[1];
    long long bki1 =
        (long long)n * k_stride_p0 + (long long)bm1 * k_stride_p1 + (long long)h * k_stride_p2;
    {
      uint2 _vld_9;
      _vld_9 = *reinterpret_cast<const uint2*>(k + bki1 + (long long)k_col0);
      uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&bk1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
        (&bk1[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
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
    int bm2 = 16 + warp_local;
    float bk2[4];
    float bv2[2];
    float bb2[1];
    long long bki2 =
        (long long)n * k_stride_p0 + (long long)bm2 * k_stride_p1 + (long long)h * k_stride_p2;
    {
      uint2 _vld_10;
      _vld_10 = *reinterpret_cast<const uint2*>(k + bki2 + (long long)k_col0);
      uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&bk2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
        (&bk2[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
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
    int bt0 = warp_local;
    float bq0[4];
    float ba0[3];
    long long bqi0 =
        (long long)n * q_stride_p0 + (long long)bt0 * q_stride_p1 + (long long)h * q_stride_p2;
    {
      uint2 _vld_11;
      _vld_11 = *reinterpret_cast<const uint2*>(q + bqi0 + (long long)k_col0);
      uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11);
#pragma unroll
      for (int _pair = 0; _pair < 2; _pair++) {
        (&bq0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) << 16);
        (&bq0[0 + _pair * 2])[1] =
            __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) & 0xffff0000u);
      }
    }
    long long bgi0 =
        (long long)n * a_stride_p0 + (long long)bt0 * a_stride_p1 + (long long)h * a_stride_p2;
    ba0[0] = (float)a[bgi0];
    ba0[1] = dt_bias[h];
    ba0[2] = A_log[h];
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
    float _warp_reduce_12 = sum_k;
#endif
#pragma unroll
#if __CUDA_ARCH__ == 900
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_12 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_12, offset);
    float k_norm = rsqrtf(_warp_reduce_12 + 1e-06f);
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
    int v_col = lane_local * 2;
    sV[bm0 * 64 + v_col] = bv0[0];
    sV[bm0 * 64 + v_col + 1] = bv0[1];
    if (lane_local == 0) {
      float _exp_14 = __expf(-bb0[0]);
      float _rcp_13 = approx_rcp(1.0f + _exp_14);
      sBeta[bm0] = _rcp_13;
    }
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
    float _warp_reduce_15 = sum_k_6;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_15 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_15, offset);
    float k_norm_7 = rsqrtf(_warp_reduce_15 + 1e-06f);
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
    if (lane_local == 0) {
      float _exp_17 = __expf(-bb1[0]);
      float _rcp_16 = approx_rcp(1.0f + _exp_17);
      sBeta[bm1] = _rcp_16;
    }
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
    float _warp_reduce_18 = sum_k_17;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_18 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_18, offset);
    float k_norm_18 = rsqrtf(_warp_reduce_18 + 1e-06f);
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
    if (lane_local == 0) {
      float _exp_20 = __expf(-bb2[0]);
      float _rcp_19 = approx_rcp(1.0f + _exp_20);
      sBeta[bm2] = _rcp_19;
    }
    float q_raw_pair0_x = bq0[0];
    float q_raw_pair0_y = bq0[1];
    float q_raw_pair1_x = bq0[2];
    float q_raw_pair1_y = bq0[3];
    float q_norm = scale;
    float _fma_12;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_12)
        : "f"(q_raw_pair0_x), "f"(q_raw_pair0_x), "f"(0.0f));
    float _fma_13;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_13)
        : "f"(q_raw_pair0_y), "f"(q_raw_pair0_y), "f"(0.0f));
    float sum_q_pair_x = _fma_12;
    float sum_q_pair_y = _fma_13;
    float _fma_14;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_14)
        : "f"(q_raw_pair1_x), "f"(q_raw_pair1_x), "f"(sum_q_pair_x));
    float _fma_15;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
        : "=f"(_fma_15)
        : "f"(q_raw_pair1_y), "f"(q_raw_pair1_y), "f"(sum_q_pair_y));
    sum_q_pair_x = _fma_14;
    sum_q_pair_y = _fma_15;
    float sum_q = sum_q_pair_x + sum_q_pair_y;
    float _warp_reduce_21 = sum_q;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_21 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_21, offset);
    float _rsqrt_0 = rsqrtf(_warp_reduce_21 + 1e-06f);
    float q_norm_pair_x = _rsqrt_0 * scale;
    float q_norm_pair_y = _rsqrt_0 * scale;
    float _mul_12;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(q_raw_pair0_x), "f"(q_norm_pair_x));
    float _mul_13;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(q_raw_pair0_y), "f"(q_norm_pair_y));
    float q_pair0_x = _mul_12;
    float q_pair0_y = _mul_13;
    float _mul_14;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(q_raw_pair1_x), "f"(q_norm_pair_x));
    float _mul_15;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(q_raw_pair1_y), "f"(q_norm_pair_y));
    float q_pair1_x = _mul_14;
    float q_pair1_y = _mul_15;
    sQ[bt0 * 128 + k_col0] = q_pair0_x;
    sQ[bt0 * 128 + k_col0 + 1] = q_pair0_y;
    sQ[bt0 * 128 + k_col0 + 2] = q_pair1_x;
    sQ[bt0 * 128 + k_col0 + 3] = q_pair1_y;
    if (lane_local == 0) {
      float x = ba0[0] + ba0[1];
      float softplus_x = x;
      if (x <= 20.0f) {
        float _exp_23 = __expf(x);
        float _log2_22;
        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_22) : "f"(1.0f + _exp_23));
        softplus_x = _log2_22 * 0.6931472f;
#elif __CUDA_ARCH__ == 1000
    for (int tt = 0; tt < 1; tt++) {
      int t = tt * 8 + warp_local;
      if ((tt + 1) * 8 <= 8) {
        long long dq_index =
            (long long)n * q_stride_p0 + (long long)t * q_stride_p1 + (long long)h * q_stride_p2;
        {
          uint2 _vld_22;
          _vld_22 = *reinterpret_cast<const uint2*>(q + dq_index + (long long)k_col0);
          uint2* _vdst_22 = reinterpret_cast<uint2*>(&w_q[0]);
          *_vdst_22 = _vld_22;
        }
        long long dg_index =
            (long long)n * a_stride_p0 + (long long)t * a_stride_p1 + (long long)h * a_stride_p2;
        {
          uint32_t _bf16_bits_23;
          _bf16_bits_23 = *reinterpret_cast<const uint16_t*>(a + dg_index);
          w_a[0] = _bf16_bits_23;
        }
        r_a[1] = dt_bias[h];
        r_a[2] = A_log[h];
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
        float _warp_reduce_24 = sum_q;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
          _warp_reduce_24 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_24, offset);
        float _rsqrt_0 = rsqrtf(_warp_reduce_24 + 1e-06f);
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
        sQ[t * 128 + k_col0] = q_pair0.x;
        sQ[t * 128 + k_col0 + 1] = q_pair0.y;
        sQ[t * 128 + k_col0 + 2] = q_pair1.x;
        sQ[t * 128 + k_col0 + 3] = q_pair1.y;
        float x = r_a[0] + r_a[1];
        float softplus_x = x;
        if (x <= 20.0f) {
          float _exp_26 = __expf(x);
          float _log2_25;
          asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_25) : "f"(1.0f + _exp_26));
          softplus_x = _log2_25 * 0.6931472f;
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
        float _exp_27 = __expf(a_log);
        float g_log = (-_exp_27) * softplus_x;
        float alpha = __expf(g_log);
        if (lane_local == 0) {
          sAlpha[t] = alpha;
          if (use_ssm_state_indices != 0) {
            sSsm[t] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 +
                                        (long long)t * ssm_state_indices_stride_p1];
          }
          if (use_ssm_state_indices == 0) {
            sSsm[t] = -1;
          }
        }
#else
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
            softplus_x = _log2_19 * 0.6931472f;
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
#endif
      }
#if __CUDA_ARCH__ == 900
      float _exp_24 = __expf(ba0[2]);
      float g_log = (-_exp_24) * softplus_x;
      float _exp_25 = __expf(g_log);
      sAlpha[bt0] = _exp_25;
      if (use_ssm_state_indices != 0) {
        sSsm[bt0] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 +
                                      (long long)bt0 * ssm_state_indices_stride_p1];
      }
      if (use_ssm_state_indices == 0) {
        sSsm[bt0] = -1;
#elif __CUDA_ARCH__ == 1000
      if ((tt + 1) * 8 > 8) {
        if (t < num_tokens) {
          long long dq_index_1 =
              (long long)n * q_stride_p0 + (long long)t * q_stride_p1 + (long long)h * q_stride_p2;
          {
            uint2 _vld_28;
            _vld_28 = *reinterpret_cast<const uint2*>(q + dq_index_1 + (long long)k_col0);
            uint2* _vdst_28 = reinterpret_cast<uint2*>(&w_q[0]);
            *_vdst_28 = _vld_28;
          }
          long long dg_index_1 =
              (long long)n * a_stride_p0 + (long long)t * a_stride_p1 + (long long)h * a_stride_p2;
          {
            uint32_t _bf16_bits_29;
            _bf16_bits_29 = *reinterpret_cast<const uint16_t*>(a + dg_index_1);
            w_a[0] = _bf16_bits_29;
          }
          r_a[1] = dt_bias[h];
          r_a[2] = A_log[h];
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
          float dpb_da_1[2];
#pragma unroll
          for (int _pair = 0; _pair < 1; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&dpb_da_1[_pair * 2])[0]), "=f"((&dpb_da_1[_pair * 2])[1])
                : "r"(w_a[_pair]));
          }
          r_a[0] = dpb_da_1[0];
          float2 q_raw_pair0_1 = make_float2(r_q[0], r_q[1]);
          float2 q_raw_pair1_1 = make_float2(r_q[2], r_q[3]);
          float q_norm_1 = scale;
          float2 sum_q_pair_1 =
              fma_f32x2_rn_ftz(q_raw_pair0_1, q_raw_pair0_1, make_float2(0.0f, 0.0f));
          sum_q_pair_1 = fma_f32x2_rn_ftz(q_raw_pair1_1, q_raw_pair1_1, sum_q_pair_1);
          float sum_q_1 = sum_q_pair_1.x + sum_q_pair_1.y;
          float _warp_reduce_30 = sum_q_1;
#pragma unroll
          for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_30 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_30, offset);
          float _rsqrt_1 = rsqrtf(_warp_reduce_30 + 1e-06f);
          float2 q_norm_pair_1 = make_float2(_rsqrt_1 * scale, _rsqrt_1 * scale);
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
          float x_1 = r_a[0] + r_a[1];
          float softplus_x_1 = x_1;
          if (x_1 <= 20.0f) {
            float _exp_32 = __expf(x_1);
            float _log2_31;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_31) : "f"(1.0f + _exp_32));
            softplus_x_1 = _log2_31 * 0.6931472f;
          }
          float a_log_1 = r_a[2];
          {
            float _scope_arg_0_1 = a_log_1;
            asm volatile(
                "{\n.reg .f32 _cake_f0;\nmov.f32 _cake_f0, %0;\nmov.f32 _cake_f0, "
                "_cake_f0;\nmov.f32 %0, _cake_f0;\n}"
                : "+f"(_scope_arg_0_1)
                :);
            a_log_1 = _scope_arg_0_1;
          }
          float _exp_33 = __expf(a_log_1);
          float g_log_1 = (-_exp_33) * softplus_x_1;
          float alpha_1 = __expf(g_log_1);
          if (lane_local == 0) {
            sAlpha[t] = alpha_1;
            if (use_ssm_state_indices != 0) {
              sSsm[t] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 +
                                          (long long)t * ssm_state_indices_stride_p1];
            }
            if (use_ssm_state_indices == 0) {
              sSsm[t] = -1;
            }
          }
        }
#else
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
              softplus_x_1 = _log2_26 * 0.6931472f;
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
#endif
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
#endif
    if (use_ssm_state_indices != 0) {
#pragma unroll 1
#if __CUDA_ARCH__ == 900
      for (int t = 0; t < 8; t++) {
        float alphas = sAlpha[t];
        float alpha2s_x = alphas;
        float alpha2s_y = alphas;
        float _mul_16;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(S[0]), "f"(alpha2s_x));
        float _mul_17;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(S[1]), "f"(alpha2s_y));
        S[0] = _mul_16;
        S[1] = _mul_17;
        float _mul_18;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(S[2]), "f"(alpha2s_x));
        float _mul_19;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(S[3]), "f"(alpha2s_y));
        S[2] = _mul_18;
        S[3] = _mul_19;
        float _mul_20;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(S[4]), "f"(alpha2s_x));
        float _mul_21;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(S[5]), "f"(alpha2s_y));
        S[4] = _mul_20;
        S[5] = _mul_21;
        float _mul_22;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(S[6]), "f"(alpha2s_x));
        float _mul_23;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(S[7]), "f"(alpha2s_y));
        S[6] = _mul_22;
        S[7] = _mul_23;
        float _mul_24;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_24) : "f"(S[8]), "f"(alpha2s_x));
        float _mul_25;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_25) : "f"(S[9]), "f"(alpha2s_y));
        S[8] = _mul_24;
        S[9] = _mul_25;
        float _mul_26;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_26) : "f"(S[10]), "f"(alpha2s_x));
        float _mul_27;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_27) : "f"(S[11]), "f"(alpha2s_y));
        S[10] = _mul_26;
        S[11] = _mul_27;
        float _mul_28;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_28) : "f"(S[12]), "f"(alpha2s_x));
        float _mul_29;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_29) : "f"(S[13]), "f"(alpha2s_y));
        S[12] = _mul_28;
        S[13] = _mul_29;
        float _mul_30;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_30) : "f"(S[14]), "f"(alpha2s_x));
        float _mul_31;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_31) : "f"(S[15]), "f"(alpha2s_y));
        S[14] = _mul_30;
        S[15] = _mul_31;
        float _mul_32;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_32) : "f"(S[16]), "f"(alpha2s_x));
        float _mul_33;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_33) : "f"(S[17]), "f"(alpha2s_y));
        S[16] = _mul_32;
        S[17] = _mul_33;
        float _mul_34;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_34) : "f"(S[18]), "f"(alpha2s_x));
        float _mul_35;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_35) : "f"(S[19]), "f"(alpha2s_y));
        S[18] = _mul_34;
        S[19] = _mul_35;
        float _mul_36;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_36) : "f"(S[20]), "f"(alpha2s_x));
        float _mul_37;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_37) : "f"(S[21]), "f"(alpha2s_y));
        S[20] = _mul_36;
        S[21] = _mul_37;
        float _mul_38;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_38) : "f"(S[22]), "f"(alpha2s_x));
        float _mul_39;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(S[23]), "f"(alpha2s_y));
        S[22] = _mul_38;
        S[23] = _mul_39;
        float _mul_40;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_40) : "f"(S[24]), "f"(alpha2s_x));
        float _mul_41;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_41) : "f"(S[25]), "f"(alpha2s_y));
        S[24] = _mul_40;
        S[25] = _mul_41;
        float _mul_42;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_42) : "f"(S[26]), "f"(alpha2s_x));
        float _mul_43;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_43) : "f"(S[27]), "f"(alpha2s_y));
        S[26] = _mul_42;
        S[27] = _mul_43;
        float _mul_44;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_44) : "f"(S[28]), "f"(alpha2s_x));
        float _mul_45;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_45) : "f"(S[29]), "f"(alpha2s_y));
        S[28] = _mul_44;
        S[29] = _mul_45;
        float _mul_46;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_46) : "f"(S[30]), "f"(alpha2s_x));
        float _mul_47;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_47) : "f"(S[31]), "f"(alpha2s_y));
        S[30] = _mul_46;
        S[31] = _mul_47;
        int ms0 = t * 3;
#else
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
#endif
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
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                     : "r"(sV_addr + (unsigned int)(ms0 * 256) + group_byte));
        float betas0 = sBeta[ms0];
#if __CUDA_ARCH__ == 900
        float accs0_0_0_x = 0.0f;
        float accs0_0_0_y = 0.0f;
        float accs0_0_1_x = 0.0f;
        float accs0_0_1_y = 0.0f;
        float _fma_16;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_16)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accs0_0_0_x));
        float _fma_17;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_17)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accs0_0_0_y));
        accs0_0_0_x = _fma_16;
        accs0_0_0_y = _fma_17;
        float _fma_18;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_18)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accs0_0_1_x));
        float _fma_19;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_19)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accs0_0_1_y));
        accs0_0_1_x = _fma_18;
        accs0_0_1_y = _fma_19;
        float _fma_20;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_20)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accs0_0_0_x));
        float _fma_21;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_21)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accs0_0_0_y));
        accs0_0_0_x = _fma_20;
        accs0_0_0_y = _fma_21;
        float _fma_22;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_22)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accs0_0_1_x));
        float _fma_23;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_23)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accs0_0_1_y));
        accs0_0_1_x = _fma_22;
        accs0_0_1_y = _fma_23;
        float _fma_24;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_24)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accs0_0_0_x));
        float _fma_25;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_25)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accs0_0_0_y));
        accs0_0_0_x = _fma_24;
        accs0_0_0_y = _fma_25;
        float _fma_26;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_26)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accs0_0_1_x));
        float _fma_27;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_27)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accs0_0_1_y));
        accs0_0_1_x = _fma_26;
        accs0_0_1_y = _fma_27;
        float _fma_28;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_28)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accs0_0_0_x));
        float _fma_29;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_29)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accs0_0_0_y));
        accs0_0_0_x = _fma_28;
        accs0_0_0_y = _fma_29;
        float _fma_30;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_30)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accs0_0_1_x));
        float _fma_31;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_31)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accs0_0_1_y));
        accs0_0_1_x = _fma_30;
        accs0_0_1_y = _fma_31;
        float _add_26;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(accs0_0_0_x), "f"(accs0_0_1_x));
        float _add_27;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(accs0_0_0_y), "f"(accs0_0_1_y));
        float dots0_0 = _add_26 + _add_27;
        float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 1);
        dots0_0 = dots0_0 + _shfl_xor_sync_28;
        float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 2);
        dots0_0 = dots0_0 + _shfl_xor_sync_29;
        float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 4);
        dots0_0 = dots0_0 + _shfl_xor_sync_30;
#elif __CUDA_ARCH__ == 1000
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
        float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 1);
        dots0_0 = dots0_0 + _shfl_xor_sync_34;
        float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 2);
        dots0_0 = dots0_0 + _shfl_xor_sync_35;
        float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 4);
        dots0_0 = dots0_0 + _shfl_xor_sync_36;
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
      float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 1);
      dots0_0 = dots0_0 + _shfl_xor_sync_30;
      float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 2);
      dots0_0 = dots0_0 + _shfl_xor_sync_31;
      float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 4);
      dots0_0 = dots0_0 + _shfl_xor_sync_32;
#endif
        float vns0_0 = (v_regs[0] - dots0_0) * betas0;
#if __CUDA_ARCH__ == 900
        float vn2s0_0_x = vns0_0;
        float vn2s0_0_y = vns0_0;
        float _fma_32;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_32)
            : "f"(k_regs[0]), "f"(vn2s0_0_x), "f"(S[0]));
        float _fma_33;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_33)
            : "f"(k_regs[1]), "f"(vn2s0_0_y), "f"(S[1]));
        S[0] = _fma_32;
        S[1] = _fma_33;
        float _fma_34;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_34)
            : "f"(k_regs[2]), "f"(vn2s0_0_x), "f"(S[2]));
        float _fma_35;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_35)
            : "f"(k_regs[3]), "f"(vn2s0_0_y), "f"(S[3]));
        S[2] = _fma_34;
        S[3] = _fma_35;
        float _fma_36;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_36)
            : "f"(k_regs[4]), "f"(vn2s0_0_x), "f"(S[4]));
        float _fma_37;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_37)
            : "f"(k_regs[5]), "f"(vn2s0_0_y), "f"(S[5]));
        S[4] = _fma_36;
        S[5] = _fma_37;
        float _fma_38;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_38)
            : "f"(k_regs[6]), "f"(vn2s0_0_x), "f"(S[6]));
        float _fma_39;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_39)
            : "f"(k_regs[7]), "f"(vn2s0_0_y), "f"(S[7]));
        S[6] = _fma_38;
        S[7] = _fma_39;
        float _fma_40;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_40)
            : "f"(k_regs[8]), "f"(vn2s0_0_x), "f"(S[8]));
        float _fma_41;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_41)
            : "f"(k_regs[9]), "f"(vn2s0_0_y), "f"(S[9]));
        S[8] = _fma_40;
        S[9] = _fma_41;
        float _fma_42;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_42)
            : "f"(k_regs[10]), "f"(vn2s0_0_x), "f"(S[10]));
        float _fma_43;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_43)
            : "f"(k_regs[11]), "f"(vn2s0_0_y), "f"(S[11]));
        S[10] = _fma_42;
        S[11] = _fma_43;
        float _fma_44;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_44)
            : "f"(k_regs[12]), "f"(vn2s0_0_x), "f"(S[12]));
        float _fma_45;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_45)
            : "f"(k_regs[13]), "f"(vn2s0_0_y), "f"(S[13]));
        S[12] = _fma_44;
        S[13] = _fma_45;
        float _fma_46;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_46)
            : "f"(k_regs[14]), "f"(vn2s0_0_x), "f"(S[14]));
        float _fma_47;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_47)
            : "f"(k_regs[15]), "f"(vn2s0_0_y), "f"(S[15]));
        S[14] = _fma_46;
        S[15] = _fma_47;
        float accs0_1_0_x = 0.0f;
        float accs0_1_0_y = 0.0f;
        float accs0_1_1_x = 0.0f;
        float accs0_1_1_y = 0.0f;
        float _fma_48;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_48)
            : "f"(S[16]), "f"(k_regs[0]), "f"(accs0_1_0_x));
        float _fma_49;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_49)
            : "f"(S[17]), "f"(k_regs[1]), "f"(accs0_1_0_y));
        accs0_1_0_x = _fma_48;
        accs0_1_0_y = _fma_49;
        float _fma_50;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_50)
            : "f"(S[18]), "f"(k_regs[2]), "f"(accs0_1_1_x));
        float _fma_51;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_51)
            : "f"(S[19]), "f"(k_regs[3]), "f"(accs0_1_1_y));
        accs0_1_1_x = _fma_50;
        accs0_1_1_y = _fma_51;
        float _fma_52;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_52)
            : "f"(S[20]), "f"(k_regs[4]), "f"(accs0_1_0_x));
        float _fma_53;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_53)
            : "f"(S[21]), "f"(k_regs[5]), "f"(accs0_1_0_y));
        accs0_1_0_x = _fma_52;
        accs0_1_0_y = _fma_53;
        float _fma_54;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_54)
            : "f"(S[22]), "f"(k_regs[6]), "f"(accs0_1_1_x));
        float _fma_55;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_55)
            : "f"(S[23]), "f"(k_regs[7]), "f"(accs0_1_1_y));
        accs0_1_1_x = _fma_54;
        accs0_1_1_y = _fma_55;
        float _fma_56;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_56)
            : "f"(S[24]), "f"(k_regs[8]), "f"(accs0_1_0_x));
        float _fma_57;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_57)
            : "f"(S[25]), "f"(k_regs[9]), "f"(accs0_1_0_y));
        accs0_1_0_x = _fma_56;
        accs0_1_0_y = _fma_57;
        float _fma_58;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_58)
            : "f"(S[26]), "f"(k_regs[10]), "f"(accs0_1_1_x));
        float _fma_59;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_59)
            : "f"(S[27]), "f"(k_regs[11]), "f"(accs0_1_1_y));
        accs0_1_1_x = _fma_58;
        accs0_1_1_y = _fma_59;
        float _fma_60;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_60)
            : "f"(S[28]), "f"(k_regs[12]), "f"(accs0_1_0_x));
        float _fma_61;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_61)
            : "f"(S[29]), "f"(k_regs[13]), "f"(accs0_1_0_y));
        accs0_1_0_x = _fma_60;
        accs0_1_0_y = _fma_61;
        float _fma_62;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_62)
            : "f"(S[30]), "f"(k_regs[14]), "f"(accs0_1_1_x));
        float _fma_63;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_63)
            : "f"(S[31]), "f"(k_regs[15]), "f"(accs0_1_1_y));
        accs0_1_1_x = _fma_62;
        accs0_1_1_y = _fma_63;
        float _add_31;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(accs0_1_0_x), "f"(accs0_1_1_x));
        float _add_32;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(accs0_1_0_y), "f"(accs0_1_1_y));
        float dots0_1 = _add_31 + _add_32;
        float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 1);
        dots0_1 = dots0_1 + _shfl_xor_sync_33;
        float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 2);
        dots0_1 = dots0_1 + _shfl_xor_sync_34;
        float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 4);
        dots0_1 = dots0_1 + _shfl_xor_sync_35;
#elif __CUDA_ARCH__ == 1000
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
        float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 1);
        dots0_1 = dots0_1 + _shfl_xor_sync_37;
        float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 2);
        dots0_1 = dots0_1 + _shfl_xor_sync_38;
        float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 4);
        dots0_1 = dots0_1 + _shfl_xor_sync_39;
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
      float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 1);
      dots0_1 = dots0_1 + _shfl_xor_sync_33;
      float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 2);
      dots0_1 = dots0_1 + _shfl_xor_sync_34;
      float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 4);
      dots0_1 = dots0_1 + _shfl_xor_sync_35;
#endif
        float vns0_1 = (v_regs[1] - dots0_1) * betas0;
#if __CUDA_ARCH__ == 900
        float vn2s0_1_x = vns0_1;
        float vn2s0_1_y = vns0_1;
        float _fma_64;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_64)
            : "f"(k_regs[0]), "f"(vn2s0_1_x), "f"(S[16]));
        float _fma_65;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_65)
            : "f"(k_regs[1]), "f"(vn2s0_1_y), "f"(S[17]));
        S[16] = _fma_64;
        S[17] = _fma_65;
        float _fma_66;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_66)
            : "f"(k_regs[2]), "f"(vn2s0_1_x), "f"(S[18]));
        float _fma_67;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_67)
            : "f"(k_regs[3]), "f"(vn2s0_1_y), "f"(S[19]));
        S[18] = _fma_66;
        S[19] = _fma_67;
        float _fma_68;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_68)
            : "f"(k_regs[4]), "f"(vn2s0_1_x), "f"(S[20]));
        float _fma_69;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_69)
            : "f"(k_regs[5]), "f"(vn2s0_1_y), "f"(S[21]));
        S[20] = _fma_68;
        S[21] = _fma_69;
        float _fma_70;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_70)
            : "f"(k_regs[6]), "f"(vn2s0_1_x), "f"(S[22]));
        float _fma_71;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_71)
            : "f"(k_regs[7]), "f"(vn2s0_1_y), "f"(S[23]));
        S[22] = _fma_70;
        S[23] = _fma_71;
        float _fma_72;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_72)
            : "f"(k_regs[8]), "f"(vn2s0_1_x), "f"(S[24]));
        float _fma_73;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_73)
            : "f"(k_regs[9]), "f"(vn2s0_1_y), "f"(S[25]));
        S[24] = _fma_72;
        S[25] = _fma_73;
        float _fma_74;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_74)
            : "f"(k_regs[10]), "f"(vn2s0_1_x), "f"(S[26]));
        float _fma_75;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_75)
            : "f"(k_regs[11]), "f"(vn2s0_1_y), "f"(S[27]));
        S[26] = _fma_74;
        S[27] = _fma_75;
        float _fma_76;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_76)
            : "f"(k_regs[12]), "f"(vn2s0_1_x), "f"(S[28]));
        float _fma_77;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_77)
            : "f"(k_regs[13]), "f"(vn2s0_1_y), "f"(S[29]));
        S[28] = _fma_76;
        S[29] = _fma_77;
        float _fma_78;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_78)
            : "f"(k_regs[14]), "f"(vn2s0_1_x), "f"(S[30]));
        float _fma_79;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_79)
            : "f"(k_regs[15]), "f"(vn2s0_1_y), "f"(S[31]));
        S[30] = _fma_78;
        S[31] = _fma_79;
        int ms1 = t * 3 + 1;
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
        int ms1 = t_1 * 3 + 1;
#endif
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
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                     : "r"(sV_addr + (unsigned int)(ms1 * 256) + group_byte));
        float betas1 = sBeta[ms1];
#if __CUDA_ARCH__ == 900
        float accs1_0_0_x = 0.0f;
        float accs1_0_0_y = 0.0f;
        float accs1_0_1_x = 0.0f;
        float accs1_0_1_y = 0.0f;
        float _fma_80;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_80)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accs1_0_0_x));
        float _fma_81;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_81)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accs1_0_0_y));
        accs1_0_0_x = _fma_80;
        accs1_0_0_y = _fma_81;
        float _fma_82;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_82)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accs1_0_1_x));
        float _fma_83;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_83)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accs1_0_1_y));
        accs1_0_1_x = _fma_82;
        accs1_0_1_y = _fma_83;
        float _fma_84;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_84)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accs1_0_0_x));
        float _fma_85;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_85)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accs1_0_0_y));
        accs1_0_0_x = _fma_84;
        accs1_0_0_y = _fma_85;
        float _fma_86;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_86)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accs1_0_1_x));
        float _fma_87;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_87)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accs1_0_1_y));
        accs1_0_1_x = _fma_86;
        accs1_0_1_y = _fma_87;
        float _fma_88;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_88)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accs1_0_0_x));
        float _fma_89;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_89)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accs1_0_0_y));
        accs1_0_0_x = _fma_88;
        accs1_0_0_y = _fma_89;
        float _fma_90;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_90)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accs1_0_1_x));
        float _fma_91;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_91)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accs1_0_1_y));
        accs1_0_1_x = _fma_90;
        accs1_0_1_y = _fma_91;
        float _fma_92;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_92)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accs1_0_0_x));
        float _fma_93;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_93)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accs1_0_0_y));
        accs1_0_0_x = _fma_92;
        accs1_0_0_y = _fma_93;
        float _fma_94;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_94)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accs1_0_1_x));
        float _fma_95;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_95)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accs1_0_1_y));
        accs1_0_1_x = _fma_94;
        accs1_0_1_y = _fma_95;
        float _add_36;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(accs1_0_0_x), "f"(accs1_0_1_x));
        float _add_37;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(accs1_0_0_y), "f"(accs1_0_1_y));
        float dots1_0 = _add_36 + _add_37;
        float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 1);
        dots1_0 = dots1_0 + _shfl_xor_sync_38;
        float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 2);
        dots1_0 = dots1_0 + _shfl_xor_sync_39;
        float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 4);
        dots1_0 = dots1_0 + _shfl_xor_sync_40;
#elif __CUDA_ARCH__ == 1000
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
        float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 1);
        dots1_0 = dots1_0 + _shfl_xor_sync_40;
        float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 2);
        dots1_0 = dots1_0 + _shfl_xor_sync_41;
        float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 4);
        dots1_0 = dots1_0 + _shfl_xor_sync_42;
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
      float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 1);
      dots1_0 = dots1_0 + _shfl_xor_sync_36;
      float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 2);
      dots1_0 = dots1_0 + _shfl_xor_sync_37;
      float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 4);
      dots1_0 = dots1_0 + _shfl_xor_sync_38;
#endif
        float vns1_0 = (v_regs[0] - dots1_0) * betas1;
#if __CUDA_ARCH__ == 900
        float vn2s1_0_x = vns1_0;
        float vn2s1_0_y = vns1_0;
        float _fma_96;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_96)
            : "f"(k_regs[0]), "f"(vn2s1_0_x), "f"(S[0]));
        float _fma_97;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_97)
            : "f"(k_regs[1]), "f"(vn2s1_0_y), "f"(S[1]));
        S[0] = _fma_96;
        S[1] = _fma_97;
        float _fma_98;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_98)
            : "f"(k_regs[2]), "f"(vn2s1_0_x), "f"(S[2]));
        float _fma_99;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_99)
            : "f"(k_regs[3]), "f"(vn2s1_0_y), "f"(S[3]));
        S[2] = _fma_98;
        S[3] = _fma_99;
        float _fma_100;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_100)
            : "f"(k_regs[4]), "f"(vn2s1_0_x), "f"(S[4]));
        float _fma_101;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_101)
            : "f"(k_regs[5]), "f"(vn2s1_0_y), "f"(S[5]));
        S[4] = _fma_100;
        S[5] = _fma_101;
        float _fma_102;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_102)
            : "f"(k_regs[6]), "f"(vn2s1_0_x), "f"(S[6]));
        float _fma_103;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_103)
            : "f"(k_regs[7]), "f"(vn2s1_0_y), "f"(S[7]));
        S[6] = _fma_102;
        S[7] = _fma_103;
        float _fma_104;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_104)
            : "f"(k_regs[8]), "f"(vn2s1_0_x), "f"(S[8]));
        float _fma_105;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_105)
            : "f"(k_regs[9]), "f"(vn2s1_0_y), "f"(S[9]));
        S[8] = _fma_104;
        S[9] = _fma_105;
        float _fma_106;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_106)
            : "f"(k_regs[10]), "f"(vn2s1_0_x), "f"(S[10]));
        float _fma_107;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_107)
            : "f"(k_regs[11]), "f"(vn2s1_0_y), "f"(S[11]));
        S[10] = _fma_106;
        S[11] = _fma_107;
        float _fma_108;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_108)
            : "f"(k_regs[12]), "f"(vn2s1_0_x), "f"(S[12]));
        float _fma_109;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_109)
            : "f"(k_regs[13]), "f"(vn2s1_0_y), "f"(S[13]));
        S[12] = _fma_108;
        S[13] = _fma_109;
        float _fma_110;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_110)
            : "f"(k_regs[14]), "f"(vn2s1_0_x), "f"(S[14]));
        float _fma_111;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_111)
            : "f"(k_regs[15]), "f"(vn2s1_0_y), "f"(S[15]));
        S[14] = _fma_110;
        S[15] = _fma_111;
        float accs1_1_0_x = 0.0f;
        float accs1_1_0_y = 0.0f;
        float accs1_1_1_x = 0.0f;
        float accs1_1_1_y = 0.0f;
        float _fma_112;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_112)
            : "f"(S[16]), "f"(k_regs[0]), "f"(accs1_1_0_x));
        float _fma_113;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_113)
            : "f"(S[17]), "f"(k_regs[1]), "f"(accs1_1_0_y));
        accs1_1_0_x = _fma_112;
        accs1_1_0_y = _fma_113;
        float _fma_114;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_114)
            : "f"(S[18]), "f"(k_regs[2]), "f"(accs1_1_1_x));
        float _fma_115;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_115)
            : "f"(S[19]), "f"(k_regs[3]), "f"(accs1_1_1_y));
        accs1_1_1_x = _fma_114;
        accs1_1_1_y = _fma_115;
        float _fma_116;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_116)
            : "f"(S[20]), "f"(k_regs[4]), "f"(accs1_1_0_x));
        float _fma_117;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_117)
            : "f"(S[21]), "f"(k_regs[5]), "f"(accs1_1_0_y));
        accs1_1_0_x = _fma_116;
        accs1_1_0_y = _fma_117;
        float _fma_118;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_118)
            : "f"(S[22]), "f"(k_regs[6]), "f"(accs1_1_1_x));
        float _fma_119;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_119)
            : "f"(S[23]), "f"(k_regs[7]), "f"(accs1_1_1_y));
        accs1_1_1_x = _fma_118;
        accs1_1_1_y = _fma_119;
        float _fma_120;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_120)
            : "f"(S[24]), "f"(k_regs[8]), "f"(accs1_1_0_x));
        float _fma_121;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_121)
            : "f"(S[25]), "f"(k_regs[9]), "f"(accs1_1_0_y));
        accs1_1_0_x = _fma_120;
        accs1_1_0_y = _fma_121;
        float _fma_122;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_122)
            : "f"(S[26]), "f"(k_regs[10]), "f"(accs1_1_1_x));
        float _fma_123;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_123)
            : "f"(S[27]), "f"(k_regs[11]), "f"(accs1_1_1_y));
        accs1_1_1_x = _fma_122;
        accs1_1_1_y = _fma_123;
        float _fma_124;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_124)
            : "f"(S[28]), "f"(k_regs[12]), "f"(accs1_1_0_x));
        float _fma_125;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_125)
            : "f"(S[29]), "f"(k_regs[13]), "f"(accs1_1_0_y));
        accs1_1_0_x = _fma_124;
        accs1_1_0_y = _fma_125;
        float _fma_126;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_126)
            : "f"(S[30]), "f"(k_regs[14]), "f"(accs1_1_1_x));
        float _fma_127;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_127)
            : "f"(S[31]), "f"(k_regs[15]), "f"(accs1_1_1_y));
        accs1_1_1_x = _fma_126;
        accs1_1_1_y = _fma_127;
        float _add_41;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(accs1_1_0_x), "f"(accs1_1_1_x));
        float _add_42;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(accs1_1_0_y), "f"(accs1_1_1_y));
        float dots1_1 = _add_41 + _add_42;
        float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 1);
        dots1_1 = dots1_1 + _shfl_xor_sync_43;
        float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 2);
        dots1_1 = dots1_1 + _shfl_xor_sync_44;
        float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 4);
        dots1_1 = dots1_1 + _shfl_xor_sync_45;
#elif __CUDA_ARCH__ == 1000
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
        float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 1);
        dots1_1 = dots1_1 + _shfl_xor_sync_43;
        float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 2);
        dots1_1 = dots1_1 + _shfl_xor_sync_44;
        float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 4);
        dots1_1 = dots1_1 + _shfl_xor_sync_45;
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
      float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 1);
      dots1_1 = dots1_1 + _shfl_xor_sync_39;
      float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 2);
      dots1_1 = dots1_1 + _shfl_xor_sync_40;
      float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 4);
      dots1_1 = dots1_1 + _shfl_xor_sync_41;
#endif
        float vns1_1 = (v_regs[1] - dots1_1) * betas1;
#if __CUDA_ARCH__ == 900
        float vn2s1_1_x = vns1_1;
        float vn2s1_1_y = vns1_1;
        float _fma_128;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_128)
            : "f"(k_regs[0]), "f"(vn2s1_1_x), "f"(S[16]));
        float _fma_129;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_129)
            : "f"(k_regs[1]), "f"(vn2s1_1_y), "f"(S[17]));
        S[16] = _fma_128;
        S[17] = _fma_129;
        float _fma_130;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_130)
            : "f"(k_regs[2]), "f"(vn2s1_1_x), "f"(S[18]));
        float _fma_131;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_131)
            : "f"(k_regs[3]), "f"(vn2s1_1_y), "f"(S[19]));
        S[18] = _fma_130;
        S[19] = _fma_131;
        float _fma_132;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_132)
            : "f"(k_regs[4]), "f"(vn2s1_1_x), "f"(S[20]));
        float _fma_133;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_133)
            : "f"(k_regs[5]), "f"(vn2s1_1_y), "f"(S[21]));
        S[20] = _fma_132;
        S[21] = _fma_133;
        float _fma_134;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_134)
            : "f"(k_regs[6]), "f"(vn2s1_1_x), "f"(S[22]));
        float _fma_135;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_135)
            : "f"(k_regs[7]), "f"(vn2s1_1_y), "f"(S[23]));
        S[22] = _fma_134;
        S[23] = _fma_135;
        float _fma_136;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_136)
            : "f"(k_regs[8]), "f"(vn2s1_1_x), "f"(S[24]));
        float _fma_137;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_137)
            : "f"(k_regs[9]), "f"(vn2s1_1_y), "f"(S[25]));
        S[24] = _fma_136;
        S[25] = _fma_137;
        float _fma_138;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_138)
            : "f"(k_regs[10]), "f"(vn2s1_1_x), "f"(S[26]));
        float _fma_139;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_139)
            : "f"(k_regs[11]), "f"(vn2s1_1_y), "f"(S[27]));
        S[26] = _fma_138;
        S[27] = _fma_139;
        float _fma_140;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_140)
            : "f"(k_regs[12]), "f"(vn2s1_1_x), "f"(S[28]));
        float _fma_141;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_141)
            : "f"(k_regs[13]), "f"(vn2s1_1_y), "f"(S[29]));
        S[28] = _fma_140;
        S[29] = _fma_141;
        float _fma_142;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_142)
            : "f"(k_regs[14]), "f"(vn2s1_1_x), "f"(S[30]));
        float _fma_143;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_143)
            : "f"(k_regs[15]), "f"(vn2s1_1_y), "f"(S[31]));
        S[30] = _fma_142;
        S[31] = _fma_143;
        int ms2 = t * 3 + 2;
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
        int ms2 = t_1 * 3 + 2;
#endif
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
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                     : "r"(sV_addr + (unsigned int)(ms2 * 256) + group_byte));
        float betas2 = sBeta[ms2];
#if __CUDA_ARCH__ == 900
        float accs2_0_0_x = 0.0f;
        float accs2_0_0_y = 0.0f;
        float accs2_0_1_x = 0.0f;
        float accs2_0_1_y = 0.0f;
        float _fma_144;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_144)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accs2_0_0_x));
        float _fma_145;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_145)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accs2_0_0_y));
        accs2_0_0_x = _fma_144;
        accs2_0_0_y = _fma_145;
        float _fma_146;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_146)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accs2_0_1_x));
        float _fma_147;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_147)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accs2_0_1_y));
        accs2_0_1_x = _fma_146;
        accs2_0_1_y = _fma_147;
        float _fma_148;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_148)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accs2_0_0_x));
        float _fma_149;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_149)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accs2_0_0_y));
        accs2_0_0_x = _fma_148;
        accs2_0_0_y = _fma_149;
        float _fma_150;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_150)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accs2_0_1_x));
        float _fma_151;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_151)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accs2_0_1_y));
        accs2_0_1_x = _fma_150;
        accs2_0_1_y = _fma_151;
        float _fma_152;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_152)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accs2_0_0_x));
        float _fma_153;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_153)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accs2_0_0_y));
        accs2_0_0_x = _fma_152;
        accs2_0_0_y = _fma_153;
        float _fma_154;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_154)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accs2_0_1_x));
        float _fma_155;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_155)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accs2_0_1_y));
        accs2_0_1_x = _fma_154;
        accs2_0_1_y = _fma_155;
        float _fma_156;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_156)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accs2_0_0_x));
        float _fma_157;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_157)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accs2_0_0_y));
        accs2_0_0_x = _fma_156;
        accs2_0_0_y = _fma_157;
        float _fma_158;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_158)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accs2_0_1_x));
        float _fma_159;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_159)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accs2_0_1_y));
        accs2_0_1_x = _fma_158;
        accs2_0_1_y = _fma_159;
        float _add_46;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(accs2_0_0_x), "f"(accs2_0_1_x));
        float _add_47;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(accs2_0_0_y), "f"(accs2_0_1_y));
        float dots2_0 = _add_46 + _add_47;
        float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 1);
        dots2_0 = dots2_0 + _shfl_xor_sync_48;
        float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 2);
        dots2_0 = dots2_0 + _shfl_xor_sync_49;
        float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 4);
        dots2_0 = dots2_0 + _shfl_xor_sync_50;
#elif __CUDA_ARCH__ == 1000
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
        float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 1);
        dots2_0 = dots2_0 + _shfl_xor_sync_46;
        float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 2);
        dots2_0 = dots2_0 + _shfl_xor_sync_47;
        float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 4);
        dots2_0 = dots2_0 + _shfl_xor_sync_48;
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
      float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 1);
      dots2_0 = dots2_0 + _shfl_xor_sync_42;
      float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 2);
      dots2_0 = dots2_0 + _shfl_xor_sync_43;
      float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 4);
      dots2_0 = dots2_0 + _shfl_xor_sync_44;
#endif
        float vns2_0 = (v_regs[0] - dots2_0) * betas2;
#if __CUDA_ARCH__ == 900
        float vn2s2_0_x = vns2_0;
        float vn2s2_0_y = vns2_0;
        float _fma_160;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_160)
            : "f"(k_regs[0]), "f"(vn2s2_0_x), "f"(S[0]));
        float _fma_161;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_161)
            : "f"(k_regs[1]), "f"(vn2s2_0_y), "f"(S[1]));
        S[0] = _fma_160;
        S[1] = _fma_161;
        float _fma_162;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_162)
            : "f"(k_regs[2]), "f"(vn2s2_0_x), "f"(S[2]));
        float _fma_163;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_163)
            : "f"(k_regs[3]), "f"(vn2s2_0_y), "f"(S[3]));
        S[2] = _fma_162;
        S[3] = _fma_163;
        float _fma_164;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_164)
            : "f"(k_regs[4]), "f"(vn2s2_0_x), "f"(S[4]));
        float _fma_165;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_165)
            : "f"(k_regs[5]), "f"(vn2s2_0_y), "f"(S[5]));
        S[4] = _fma_164;
        S[5] = _fma_165;
        float _fma_166;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_166)
            : "f"(k_regs[6]), "f"(vn2s2_0_x), "f"(S[6]));
        float _fma_167;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_167)
            : "f"(k_regs[7]), "f"(vn2s2_0_y), "f"(S[7]));
        S[6] = _fma_166;
        S[7] = _fma_167;
        float _fma_168;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_168)
            : "f"(k_regs[8]), "f"(vn2s2_0_x), "f"(S[8]));
        float _fma_169;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_169)
            : "f"(k_regs[9]), "f"(vn2s2_0_y), "f"(S[9]));
        S[8] = _fma_168;
        S[9] = _fma_169;
        float _fma_170;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_170)
            : "f"(k_regs[10]), "f"(vn2s2_0_x), "f"(S[10]));
        float _fma_171;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_171)
            : "f"(k_regs[11]), "f"(vn2s2_0_y), "f"(S[11]));
        S[10] = _fma_170;
        S[11] = _fma_171;
        float _fma_172;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_172)
            : "f"(k_regs[12]), "f"(vn2s2_0_x), "f"(S[12]));
        float _fma_173;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_173)
            : "f"(k_regs[13]), "f"(vn2s2_0_y), "f"(S[13]));
        S[12] = _fma_172;
        S[13] = _fma_173;
        float _fma_174;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_174)
            : "f"(k_regs[14]), "f"(vn2s2_0_x), "f"(S[14]));
        float _fma_175;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_175)
            : "f"(k_regs[15]), "f"(vn2s2_0_y), "f"(S[15]));
        S[14] = _fma_174;
        S[15] = _fma_175;
        float accs2_1_0_x = 0.0f;
        float accs2_1_0_y = 0.0f;
        float accs2_1_1_x = 0.0f;
        float accs2_1_1_y = 0.0f;
        float _fma_176;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_176)
            : "f"(S[16]), "f"(k_regs[0]), "f"(accs2_1_0_x));
        float _fma_177;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_177)
            : "f"(S[17]), "f"(k_regs[1]), "f"(accs2_1_0_y));
        accs2_1_0_x = _fma_176;
        accs2_1_0_y = _fma_177;
        float _fma_178;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_178)
            : "f"(S[18]), "f"(k_regs[2]), "f"(accs2_1_1_x));
        float _fma_179;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_179)
            : "f"(S[19]), "f"(k_regs[3]), "f"(accs2_1_1_y));
        accs2_1_1_x = _fma_178;
        accs2_1_1_y = _fma_179;
        float _fma_180;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_180)
            : "f"(S[20]), "f"(k_regs[4]), "f"(accs2_1_0_x));
        float _fma_181;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_181)
            : "f"(S[21]), "f"(k_regs[5]), "f"(accs2_1_0_y));
        accs2_1_0_x = _fma_180;
        accs2_1_0_y = _fma_181;
        float _fma_182;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_182)
            : "f"(S[22]), "f"(k_regs[6]), "f"(accs2_1_1_x));
        float _fma_183;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_183)
            : "f"(S[23]), "f"(k_regs[7]), "f"(accs2_1_1_y));
        accs2_1_1_x = _fma_182;
        accs2_1_1_y = _fma_183;
        float _fma_184;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_184)
            : "f"(S[24]), "f"(k_regs[8]), "f"(accs2_1_0_x));
        float _fma_185;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_185)
            : "f"(S[25]), "f"(k_regs[9]), "f"(accs2_1_0_y));
        accs2_1_0_x = _fma_184;
        accs2_1_0_y = _fma_185;
        float _fma_186;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_186)
            : "f"(S[26]), "f"(k_regs[10]), "f"(accs2_1_1_x));
        float _fma_187;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_187)
            : "f"(S[27]), "f"(k_regs[11]), "f"(accs2_1_1_y));
        accs2_1_1_x = _fma_186;
        accs2_1_1_y = _fma_187;
        float _fma_188;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_188)
            : "f"(S[28]), "f"(k_regs[12]), "f"(accs2_1_0_x));
        float _fma_189;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_189)
            : "f"(S[29]), "f"(k_regs[13]), "f"(accs2_1_0_y));
        accs2_1_0_x = _fma_188;
        accs2_1_0_y = _fma_189;
        float _fma_190;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_190)
            : "f"(S[30]), "f"(k_regs[14]), "f"(accs2_1_1_x));
        float _fma_191;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_191)
            : "f"(S[31]), "f"(k_regs[15]), "f"(accs2_1_1_y));
        accs2_1_1_x = _fma_190;
        accs2_1_1_y = _fma_191;
        float _add_51;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(accs2_1_0_x), "f"(accs2_1_1_x));
        float _add_52;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(accs2_1_0_y), "f"(accs2_1_1_y));
        float dots2_1 = _add_51 + _add_52;
        float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 1);
        dots2_1 = dots2_1 + _shfl_xor_sync_53;
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 2);
        dots2_1 = dots2_1 + _shfl_xor_sync_54;
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 4);
        dots2_1 = dots2_1 + _shfl_xor_sync_55;
#elif __CUDA_ARCH__ == 1000
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
        float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 1);
        dots2_1 = dots2_1 + _shfl_xor_sync_49;
        float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 2);
        dots2_1 = dots2_1 + _shfl_xor_sync_50;
        float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 4);
        dots2_1 = dots2_1 + _shfl_xor_sync_51;
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
      float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 1);
      dots2_1 = dots2_1 + _shfl_xor_sync_45;
      float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 2);
      dots2_1 = dots2_1 + _shfl_xor_sync_46;
      float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 4);
      dots2_1 = dots2_1 + _shfl_xor_sync_47;
#endif
        float vns2_1 = (v_regs[1] - dots2_1) * betas2;
#if __CUDA_ARCH__ == 900
        float vn2s2_1_x = vns2_1;
        float vn2s2_1_y = vns2_1;
        float _fma_192;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_192)
            : "f"(k_regs[0]), "f"(vn2s2_1_x), "f"(S[16]));
        float _fma_193;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_193)
            : "f"(k_regs[1]), "f"(vn2s2_1_y), "f"(S[17]));
        S[16] = _fma_192;
        S[17] = _fma_193;
        float _fma_194;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_194)
            : "f"(k_regs[2]), "f"(vn2s2_1_x), "f"(S[18]));
        float _fma_195;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_195)
            : "f"(k_regs[3]), "f"(vn2s2_1_y), "f"(S[19]));
        S[18] = _fma_194;
        S[19] = _fma_195;
        float _fma_196;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_196)
            : "f"(k_regs[4]), "f"(vn2s2_1_x), "f"(S[20]));
        float _fma_197;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_197)
            : "f"(k_regs[5]), "f"(vn2s2_1_y), "f"(S[21]));
        S[20] = _fma_196;
        S[21] = _fma_197;
        float _fma_198;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_198)
            : "f"(k_regs[6]), "f"(vn2s2_1_x), "f"(S[22]));
        float _fma_199;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_199)
            : "f"(k_regs[7]), "f"(vn2s2_1_y), "f"(S[23]));
        S[22] = _fma_198;
        S[23] = _fma_199;
        float _fma_200;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_200)
            : "f"(k_regs[8]), "f"(vn2s2_1_x), "f"(S[24]));
        float _fma_201;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_201)
            : "f"(k_regs[9]), "f"(vn2s2_1_y), "f"(S[25]));
        S[24] = _fma_200;
        S[25] = _fma_201;
        float _fma_202;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_202)
            : "f"(k_regs[10]), "f"(vn2s2_1_x), "f"(S[26]));
        float _fma_203;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_203)
            : "f"(k_regs[11]), "f"(vn2s2_1_y), "f"(S[27]));
        S[26] = _fma_202;
        S[27] = _fma_203;
        float _fma_204;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_204)
            : "f"(k_regs[12]), "f"(vn2s2_1_x), "f"(S[28]));
        float _fma_205;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_205)
            : "f"(k_regs[13]), "f"(vn2s2_1_y), "f"(S[29]));
        S[28] = _fma_204;
        S[29] = _fma_205;
        float _fma_206;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_206)
            : "f"(k_regs[14]), "f"(vn2s2_1_x), "f"(S[30]));
        float _fma_207;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_207)
            : "f"(k_regs[15]), "f"(vn2s2_1_y), "f"(S[31]));
        S[30] = _fma_206;
        S[31] = _fma_207;
        unsigned int qbases = sQ_addr + (unsigned int)(t * 512) + lane_byte;
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
        unsigned int qbases = sQ_addr + (unsigned int)(t_1 * 512) + lane_byte;
#endif
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
        float _fma_208;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_208)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accsq_0_0_x));
        float _fma_209;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_209)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accsq_0_0_y));
        accsq_0_0_x = _fma_208;
        accsq_0_0_y = _fma_209;
        float _fma_210;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_210)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accsq_0_1_x));
        float _fma_211;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_211)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accsq_0_1_y));
        accsq_0_1_x = _fma_210;
        accsq_0_1_y = _fma_211;
        float _fma_212;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_212)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accsq_0_0_x));
        float _fma_213;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_213)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accsq_0_0_y));
        accsq_0_0_x = _fma_212;
        accsq_0_0_y = _fma_213;
        float _fma_214;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_214)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accsq_0_1_x));
        float _fma_215;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_215)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accsq_0_1_y));
        accsq_0_1_x = _fma_214;
        accsq_0_1_y = _fma_215;
        float _fma_216;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_216)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accsq_0_0_x));
        float _fma_217;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_217)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accsq_0_0_y));
        accsq_0_0_x = _fma_216;
        accsq_0_0_y = _fma_217;
        float _fma_218;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_218)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accsq_0_1_x));
        float _fma_219;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_219)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accsq_0_1_y));
        accsq_0_1_x = _fma_218;
        accsq_0_1_y = _fma_219;
        float _fma_220;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_220)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accsq_0_0_x));
        float _fma_221;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_221)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accsq_0_0_y));
        accsq_0_0_x = _fma_220;
        accsq_0_0_y = _fma_221;
        float _fma_222;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_222)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accsq_0_1_x));
        float _fma_223;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_223)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accsq_0_1_y));
        accsq_0_1_x = _fma_222;
        accsq_0_1_y = _fma_223;
        float _add_56;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(accsq_0_0_x), "f"(accsq_0_1_x));
        float _add_57;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(accsq_0_0_y), "f"(accsq_0_1_y));
        float dotsq_0 = _add_56 + _add_57;
        float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 1);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_58;
        float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 2);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_59;
        float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 4);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_60;
#elif __CUDA_ARCH__ == 1000
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
        float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 1);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_52;
        float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 2);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_53;
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 4);
        dotsq_0 = dotsq_0 + _shfl_xor_sync_54;
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
      float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 1);
      dotsq_0 = dotsq_0 + _shfl_xor_sync_48;
      float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 2);
      dotsq_0 = dotsq_0 + _shfl_xor_sync_49;
      float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 4);
      dotsq_0 = dotsq_0 + _shfl_xor_sync_50;
#endif
        o4[0] = dotsq_0;
#if __CUDA_ARCH__ == 900
        float accsq_1_0_x = 0.0f;
        float accsq_1_0_y = 0.0f;
        float accsq_1_1_x = 0.0f;
        float accsq_1_1_y = 0.0f;
        float _fma_224;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_224)
            : "f"(S[16]), "f"(k_regs[0]), "f"(accsq_1_0_x));
        float _fma_225;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_225)
            : "f"(S[17]), "f"(k_regs[1]), "f"(accsq_1_0_y));
        accsq_1_0_x = _fma_224;
        accsq_1_0_y = _fma_225;
        float _fma_226;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_226)
            : "f"(S[18]), "f"(k_regs[2]), "f"(accsq_1_1_x));
        float _fma_227;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_227)
            : "f"(S[19]), "f"(k_regs[3]), "f"(accsq_1_1_y));
        accsq_1_1_x = _fma_226;
        accsq_1_1_y = _fma_227;
        float _fma_228;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_228)
            : "f"(S[20]), "f"(k_regs[4]), "f"(accsq_1_0_x));
        float _fma_229;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_229)
            : "f"(S[21]), "f"(k_regs[5]), "f"(accsq_1_0_y));
        accsq_1_0_x = _fma_228;
        accsq_1_0_y = _fma_229;
        float _fma_230;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_230)
            : "f"(S[22]), "f"(k_regs[6]), "f"(accsq_1_1_x));
        float _fma_231;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_231)
            : "f"(S[23]), "f"(k_regs[7]), "f"(accsq_1_1_y));
        accsq_1_1_x = _fma_230;
        accsq_1_1_y = _fma_231;
        float _fma_232;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_232)
            : "f"(S[24]), "f"(k_regs[8]), "f"(accsq_1_0_x));
        float _fma_233;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_233)
            : "f"(S[25]), "f"(k_regs[9]), "f"(accsq_1_0_y));
        accsq_1_0_x = _fma_232;
        accsq_1_0_y = _fma_233;
        float _fma_234;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_234)
            : "f"(S[26]), "f"(k_regs[10]), "f"(accsq_1_1_x));
        float _fma_235;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_235)
            : "f"(S[27]), "f"(k_regs[11]), "f"(accsq_1_1_y));
        accsq_1_1_x = _fma_234;
        accsq_1_1_y = _fma_235;
        float _fma_236;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_236)
            : "f"(S[28]), "f"(k_regs[12]), "f"(accsq_1_0_x));
        float _fma_237;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_237)
            : "f"(S[29]), "f"(k_regs[13]), "f"(accsq_1_0_y));
        accsq_1_0_x = _fma_236;
        accsq_1_0_y = _fma_237;
        float _fma_238;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_238)
            : "f"(S[30]), "f"(k_regs[14]), "f"(accsq_1_1_x));
        float _fma_239;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_239)
            : "f"(S[31]), "f"(k_regs[15]), "f"(accsq_1_1_y));
        accsq_1_1_x = _fma_238;
        accsq_1_1_y = _fma_239;
        float _add_61;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(accsq_1_0_x), "f"(accsq_1_1_x));
        float _add_62;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(accsq_1_0_y), "f"(accsq_1_1_y));
        float dotsq_1 = _add_61 + _add_62;
        float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 1);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_63;
        float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 2);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_64;
        float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 4);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_65;
#elif __CUDA_ARCH__ == 1000
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
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 1);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_55;
        float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 2);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_56;
        float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 4);
        dotsq_1 = dotsq_1 + _shfl_xor_sync_57;
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
      float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 1);
      dotsq_1 = dotsq_1 + _shfl_xor_sync_51;
      float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 2);
      dotsq_1 = dotsq_1 + _shfl_xor_sync_52;
      float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 4);
      dotsq_1 = dotsq_1 + _shfl_xor_sync_53;
#endif
        o4[1] = dotsq_1;
        if (u == 0) {
#if __CUDA_ARCH__ == 900
          long long out_index = (long long)n * out_stride_p0 + (long long)t * out_stride_p1 +
                                (long long)h * out_stride_p2 + (long long)row0;
#else
          long long out_index = (long long)n * out_stride_p0 + (long long)t_1 * out_stride_p1 +
                                (long long)h * out_stride_p2 + (long long)row0;
#endif
          unsigned int _packed_0[1];
#pragma unroll
          for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 =
                __float22bfloat162_rn(make_float2(o4[_lp * 2 + 0], o4[_lp * 2 + 1 + 0]));
            _packed_0[_lp] = *(uint32_t*)&_bf2;
          }
          reinterpret_cast<unsigned*>(out + (out_index))[0] =
              reinterpret_cast<unsigned*>(_packed_0)[0];
        }
#if __CUDA_ARCH__ == 900
        int ssm_slot = sSsm[t];
#else
        int ssm_slot = sSsm[t_1];
#endif
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
#endif
      }
    }
    if (use_ssm_state_indices == 0) {
#pragma unroll 1
#if __CUDA_ARCH__ == 900
      for (int t_1 = 0; t_1 < 8; t_1++) {
        float alphap = sAlpha[t_1];
        float alpha2p_x = alphap;
        float alpha2p_y = alphap;
        float _mul_48;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_48) : "f"(S[0]), "f"(alpha2p_x));
        float _mul_49;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_49) : "f"(S[1]), "f"(alpha2p_y));
        S[0] = _mul_48;
        S[1] = _mul_49;
        float _mul_50;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_50) : "f"(S[2]), "f"(alpha2p_x));
        float _mul_51;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_51) : "f"(S[3]), "f"(alpha2p_y));
        S[2] = _mul_50;
        S[3] = _mul_51;
        float _mul_52;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_52) : "f"(S[4]), "f"(alpha2p_x));
        float _mul_53;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_53) : "f"(S[5]), "f"(alpha2p_y));
        S[4] = _mul_52;
        S[5] = _mul_53;
        float _mul_54;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_54) : "f"(S[6]), "f"(alpha2p_x));
        float _mul_55;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_55) : "f"(S[7]), "f"(alpha2p_y));
        S[6] = _mul_54;
        S[7] = _mul_55;
        float _mul_56;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_56) : "f"(S[8]), "f"(alpha2p_x));
        float _mul_57;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_57) : "f"(S[9]), "f"(alpha2p_y));
        S[8] = _mul_56;
        S[9] = _mul_57;
        float _mul_58;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_58) : "f"(S[10]), "f"(alpha2p_x));
        float _mul_59;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_59) : "f"(S[11]), "f"(alpha2p_y));
        S[10] = _mul_58;
        S[11] = _mul_59;
        float _mul_60;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_60) : "f"(S[12]), "f"(alpha2p_x));
        float _mul_61;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_61) : "f"(S[13]), "f"(alpha2p_y));
        S[12] = _mul_60;
        S[13] = _mul_61;
        float _mul_62;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_62) : "f"(S[14]), "f"(alpha2p_x));
        float _mul_63;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_63) : "f"(S[15]), "f"(alpha2p_y));
        S[14] = _mul_62;
        S[15] = _mul_63;
        float _mul_64;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_64) : "f"(S[16]), "f"(alpha2p_x));
        float _mul_65;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_65) : "f"(S[17]), "f"(alpha2p_y));
        S[16] = _mul_64;
        S[17] = _mul_65;
        float _mul_66;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_66) : "f"(S[18]), "f"(alpha2p_x));
        float _mul_67;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_67) : "f"(S[19]), "f"(alpha2p_y));
        S[18] = _mul_66;
        S[19] = _mul_67;
        float _mul_68;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_68) : "f"(S[20]), "f"(alpha2p_x));
        float _mul_69;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_69) : "f"(S[21]), "f"(alpha2p_y));
        S[20] = _mul_68;
        S[21] = _mul_69;
        float _mul_70;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_70) : "f"(S[22]), "f"(alpha2p_x));
        float _mul_71;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_71) : "f"(S[23]), "f"(alpha2p_y));
        S[22] = _mul_70;
        S[23] = _mul_71;
        float _mul_72;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_72) : "f"(S[24]), "f"(alpha2p_x));
        float _mul_73;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_73) : "f"(S[25]), "f"(alpha2p_y));
        S[24] = _mul_72;
        S[25] = _mul_73;
        float _mul_74;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_74) : "f"(S[26]), "f"(alpha2p_x));
        float _mul_75;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_75) : "f"(S[27]), "f"(alpha2p_y));
        S[26] = _mul_74;
        S[27] = _mul_75;
        float _mul_76;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_76) : "f"(S[28]), "f"(alpha2p_x));
        float _mul_77;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_77) : "f"(S[29]), "f"(alpha2p_y));
        S[28] = _mul_76;
        S[29] = _mul_77;
        float _mul_78;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_78) : "f"(S[30]), "f"(alpha2p_x));
        float _mul_79;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_79) : "f"(S[31]), "f"(alpha2p_y));
        S[30] = _mul_78;
        S[31] = _mul_79;
        int mp0 = t_1 * 3;
#else
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
#endif
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
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                     : "r"(sV_addr + (unsigned int)(mp0 * 256) + group_byte));
        float betap0 = sBeta[mp0];
#if __CUDA_ARCH__ == 900
        float accp0_0_0_x = 0.0f;
        float accp0_0_0_y = 0.0f;
        float accp0_0_1_x = 0.0f;
        float accp0_0_1_y = 0.0f;
        float _fma_240;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_240)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accp0_0_0_x));
        float _fma_241;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_241)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accp0_0_0_y));
        accp0_0_0_x = _fma_240;
        accp0_0_0_y = _fma_241;
        float _fma_242;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_242)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accp0_0_1_x));
        float _fma_243;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_243)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accp0_0_1_y));
        accp0_0_1_x = _fma_242;
        accp0_0_1_y = _fma_243;
        float _fma_244;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_244)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accp0_0_0_x));
        float _fma_245;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_245)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accp0_0_0_y));
        accp0_0_0_x = _fma_244;
        accp0_0_0_y = _fma_245;
        float _fma_246;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_246)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accp0_0_1_x));
        float _fma_247;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_247)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accp0_0_1_y));
        accp0_0_1_x = _fma_246;
        accp0_0_1_y = _fma_247;
        float _fma_248;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_248)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accp0_0_0_x));
        float _fma_249;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_249)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accp0_0_0_y));
        accp0_0_0_x = _fma_248;
        accp0_0_0_y = _fma_249;
        float _fma_250;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_250)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accp0_0_1_x));
        float _fma_251;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_251)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accp0_0_1_y));
        accp0_0_1_x = _fma_250;
        accp0_0_1_y = _fma_251;
        float _fma_252;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_252)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accp0_0_0_x));
        float _fma_253;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_253)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accp0_0_0_y));
        accp0_0_0_x = _fma_252;
        accp0_0_0_y = _fma_253;
        float _fma_254;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_254)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accp0_0_1_x));
        float _fma_255;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_255)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accp0_0_1_y));
        accp0_0_1_x = _fma_254;
        accp0_0_1_y = _fma_255;
        float _add_66;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(accp0_0_0_x), "f"(accp0_0_1_x));
        float _add_67;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(accp0_0_0_y), "f"(accp0_0_1_y));
        float dotp0_0 = _add_66 + _add_67;
        float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 1);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_68;
        float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 2);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_69;
        float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 4);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_70;
#elif __CUDA_ARCH__ == 1000
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
        float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 1);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_58;
        float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 2);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_59;
        float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 4);
        dotp0_0 = dotp0_0 + _shfl_xor_sync_60;
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
    float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 1);
    dotp0_0 = dotp0_0 + _shfl_xor_sync_54;
    float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 2);
    dotp0_0 = dotp0_0 + _shfl_xor_sync_55;
    float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, dotp0_0, 4);
    dotp0_0 = dotp0_0 + _shfl_xor_sync_56;
#endif
        float vnp0_0 = (v_regs[0] - dotp0_0) * betap0;
#if __CUDA_ARCH__ == 900
        float vn2p0_0_x = vnp0_0;
        float vn2p0_0_y = vnp0_0;
        float _fma_256;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_256)
            : "f"(k_regs[0]), "f"(vn2p0_0_x), "f"(S[0]));
        float _fma_257;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_257)
            : "f"(k_regs[1]), "f"(vn2p0_0_y), "f"(S[1]));
        S[0] = _fma_256;
        S[1] = _fma_257;
        float _fma_258;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_258)
            : "f"(k_regs[2]), "f"(vn2p0_0_x), "f"(S[2]));
        float _fma_259;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_259)
            : "f"(k_regs[3]), "f"(vn2p0_0_y), "f"(S[3]));
        S[2] = _fma_258;
        S[3] = _fma_259;
        float _fma_260;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_260)
            : "f"(k_regs[4]), "f"(vn2p0_0_x), "f"(S[4]));
        float _fma_261;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_261)
            : "f"(k_regs[5]), "f"(vn2p0_0_y), "f"(S[5]));
        S[4] = _fma_260;
        S[5] = _fma_261;
        float _fma_262;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_262)
            : "f"(k_regs[6]), "f"(vn2p0_0_x), "f"(S[6]));
        float _fma_263;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_263)
            : "f"(k_regs[7]), "f"(vn2p0_0_y), "f"(S[7]));
        S[6] = _fma_262;
        S[7] = _fma_263;
        float _fma_264;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_264)
            : "f"(k_regs[8]), "f"(vn2p0_0_x), "f"(S[8]));
        float _fma_265;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_265)
            : "f"(k_regs[9]), "f"(vn2p0_0_y), "f"(S[9]));
        S[8] = _fma_264;
        S[9] = _fma_265;
        float _fma_266;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_266)
            : "f"(k_regs[10]), "f"(vn2p0_0_x), "f"(S[10]));
        float _fma_267;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_267)
            : "f"(k_regs[11]), "f"(vn2p0_0_y), "f"(S[11]));
        S[10] = _fma_266;
        S[11] = _fma_267;
        float _fma_268;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_268)
            : "f"(k_regs[12]), "f"(vn2p0_0_x), "f"(S[12]));
        float _fma_269;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_269)
            : "f"(k_regs[13]), "f"(vn2p0_0_y), "f"(S[13]));
        S[12] = _fma_268;
        S[13] = _fma_269;
        float _fma_270;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_270)
            : "f"(k_regs[14]), "f"(vn2p0_0_x), "f"(S[14]));
        float _fma_271;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_271)
            : "f"(k_regs[15]), "f"(vn2p0_0_y), "f"(S[15]));
        S[14] = _fma_270;
        S[15] = _fma_271;
        float accp0_1_0_x = 0.0f;
        float accp0_1_0_y = 0.0f;
        float accp0_1_1_x = 0.0f;
        float accp0_1_1_y = 0.0f;
        float _fma_272;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_272)
            : "f"(S[16]), "f"(k_regs[0]), "f"(accp0_1_0_x));
        float _fma_273;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_273)
            : "f"(S[17]), "f"(k_regs[1]), "f"(accp0_1_0_y));
        accp0_1_0_x = _fma_272;
        accp0_1_0_y = _fma_273;
        float _fma_274;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_274)
            : "f"(S[18]), "f"(k_regs[2]), "f"(accp0_1_1_x));
        float _fma_275;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_275)
            : "f"(S[19]), "f"(k_regs[3]), "f"(accp0_1_1_y));
        accp0_1_1_x = _fma_274;
        accp0_1_1_y = _fma_275;
        float _fma_276;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_276)
            : "f"(S[20]), "f"(k_regs[4]), "f"(accp0_1_0_x));
        float _fma_277;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_277)
            : "f"(S[21]), "f"(k_regs[5]), "f"(accp0_1_0_y));
        accp0_1_0_x = _fma_276;
        accp0_1_0_y = _fma_277;
        float _fma_278;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_278)
            : "f"(S[22]), "f"(k_regs[6]), "f"(accp0_1_1_x));
        float _fma_279;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_279)
            : "f"(S[23]), "f"(k_regs[7]), "f"(accp0_1_1_y));
        accp0_1_1_x = _fma_278;
        accp0_1_1_y = _fma_279;
        float _fma_280;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_280)
            : "f"(S[24]), "f"(k_regs[8]), "f"(accp0_1_0_x));
        float _fma_281;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_281)
            : "f"(S[25]), "f"(k_regs[9]), "f"(accp0_1_0_y));
        accp0_1_0_x = _fma_280;
        accp0_1_0_y = _fma_281;
        float _fma_282;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_282)
            : "f"(S[26]), "f"(k_regs[10]), "f"(accp0_1_1_x));
        float _fma_283;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_283)
            : "f"(S[27]), "f"(k_regs[11]), "f"(accp0_1_1_y));
        accp0_1_1_x = _fma_282;
        accp0_1_1_y = _fma_283;
        float _fma_284;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_284)
            : "f"(S[28]), "f"(k_regs[12]), "f"(accp0_1_0_x));
        float _fma_285;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_285)
            : "f"(S[29]), "f"(k_regs[13]), "f"(accp0_1_0_y));
        accp0_1_0_x = _fma_284;
        accp0_1_0_y = _fma_285;
        float _fma_286;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_286)
            : "f"(S[30]), "f"(k_regs[14]), "f"(accp0_1_1_x));
        float _fma_287;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_287)
            : "f"(S[31]), "f"(k_regs[15]), "f"(accp0_1_1_y));
        accp0_1_1_x = _fma_286;
        accp0_1_1_y = _fma_287;
        float _add_71;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(accp0_1_0_x), "f"(accp0_1_1_x));
        float _add_72;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(accp0_1_0_y), "f"(accp0_1_1_y));
        float dotp0_1 = _add_71 + _add_72;
        float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 1);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_73;
        float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 2);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_74;
        float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 4);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_75;
#elif __CUDA_ARCH__ == 1000
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
        float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 1);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_61;
        float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 2);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_62;
        float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 4);
        dotp0_1 = dotp0_1 + _shfl_xor_sync_63;
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
    float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 1);
    dotp0_1 = dotp0_1 + _shfl_xor_sync_57;
    float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 2);
    dotp0_1 = dotp0_1 + _shfl_xor_sync_58;
    float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, dotp0_1, 4);
    dotp0_1 = dotp0_1 + _shfl_xor_sync_59;
#endif
        float vnp0_1 = (v_regs[1] - dotp0_1) * betap0;
#if __CUDA_ARCH__ == 900
        float vn2p0_1_x = vnp0_1;
        float vn2p0_1_y = vnp0_1;
        float _fma_288;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_288)
            : "f"(k_regs[0]), "f"(vn2p0_1_x), "f"(S[16]));
        float _fma_289;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_289)
            : "f"(k_regs[1]), "f"(vn2p0_1_y), "f"(S[17]));
        S[16] = _fma_288;
        S[17] = _fma_289;
        float _fma_290;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_290)
            : "f"(k_regs[2]), "f"(vn2p0_1_x), "f"(S[18]));
        float _fma_291;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_291)
            : "f"(k_regs[3]), "f"(vn2p0_1_y), "f"(S[19]));
        S[18] = _fma_290;
        S[19] = _fma_291;
        float _fma_292;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_292)
            : "f"(k_regs[4]), "f"(vn2p0_1_x), "f"(S[20]));
        float _fma_293;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_293)
            : "f"(k_regs[5]), "f"(vn2p0_1_y), "f"(S[21]));
        S[20] = _fma_292;
        S[21] = _fma_293;
        float _fma_294;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_294)
            : "f"(k_regs[6]), "f"(vn2p0_1_x), "f"(S[22]));
        float _fma_295;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_295)
            : "f"(k_regs[7]), "f"(vn2p0_1_y), "f"(S[23]));
        S[22] = _fma_294;
        S[23] = _fma_295;
        float _fma_296;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_296)
            : "f"(k_regs[8]), "f"(vn2p0_1_x), "f"(S[24]));
        float _fma_297;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_297)
            : "f"(k_regs[9]), "f"(vn2p0_1_y), "f"(S[25]));
        S[24] = _fma_296;
        S[25] = _fma_297;
        float _fma_298;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_298)
            : "f"(k_regs[10]), "f"(vn2p0_1_x), "f"(S[26]));
        float _fma_299;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_299)
            : "f"(k_regs[11]), "f"(vn2p0_1_y), "f"(S[27]));
        S[26] = _fma_298;
        S[27] = _fma_299;
        float _fma_300;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_300)
            : "f"(k_regs[12]), "f"(vn2p0_1_x), "f"(S[28]));
        float _fma_301;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_301)
            : "f"(k_regs[13]), "f"(vn2p0_1_y), "f"(S[29]));
        S[28] = _fma_300;
        S[29] = _fma_301;
        float _fma_302;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_302)
            : "f"(k_regs[14]), "f"(vn2p0_1_x), "f"(S[30]));
        float _fma_303;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_303)
            : "f"(k_regs[15]), "f"(vn2p0_1_y), "f"(S[31]));
        S[30] = _fma_302;
        S[31] = _fma_303;
        int mp1 = t_1 * 3 + 1;
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
        int mp1 = t_2 * 3 + 1;
#endif
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
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                     : "r"(sV_addr + (unsigned int)(mp1 * 256) + group_byte));
        float betap1 = sBeta[mp1];
#if __CUDA_ARCH__ == 900
        float accp1_0_0_x = 0.0f;
        float accp1_0_0_y = 0.0f;
        float accp1_0_1_x = 0.0f;
        float accp1_0_1_y = 0.0f;
        float _fma_304;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_304)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accp1_0_0_x));
        float _fma_305;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_305)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accp1_0_0_y));
        accp1_0_0_x = _fma_304;
        accp1_0_0_y = _fma_305;
        float _fma_306;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_306)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accp1_0_1_x));
        float _fma_307;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_307)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accp1_0_1_y));
        accp1_0_1_x = _fma_306;
        accp1_0_1_y = _fma_307;
        float _fma_308;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_308)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accp1_0_0_x));
        float _fma_309;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_309)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accp1_0_0_y));
        accp1_0_0_x = _fma_308;
        accp1_0_0_y = _fma_309;
        float _fma_310;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_310)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accp1_0_1_x));
        float _fma_311;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_311)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accp1_0_1_y));
        accp1_0_1_x = _fma_310;
        accp1_0_1_y = _fma_311;
        float _fma_312;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_312)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accp1_0_0_x));
        float _fma_313;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_313)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accp1_0_0_y));
        accp1_0_0_x = _fma_312;
        accp1_0_0_y = _fma_313;
        float _fma_314;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_314)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accp1_0_1_x));
        float _fma_315;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_315)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accp1_0_1_y));
        accp1_0_1_x = _fma_314;
        accp1_0_1_y = _fma_315;
        float _fma_316;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_316)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accp1_0_0_x));
        float _fma_317;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_317)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accp1_0_0_y));
        accp1_0_0_x = _fma_316;
        accp1_0_0_y = _fma_317;
        float _fma_318;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_318)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accp1_0_1_x));
        float _fma_319;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_319)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accp1_0_1_y));
        accp1_0_1_x = _fma_318;
        accp1_0_1_y = _fma_319;
        float _add_76;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(accp1_0_0_x), "f"(accp1_0_1_x));
        float _add_77;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(accp1_0_0_y), "f"(accp1_0_1_y));
        float dotp1_0 = _add_76 + _add_77;
        float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 1);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_78;
        float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 2);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_79;
        float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 4);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_80;
#elif __CUDA_ARCH__ == 1000
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
        float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 1);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_64;
        float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 2);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_65;
        float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 4);
        dotp1_0 = dotp1_0 + _shfl_xor_sync_66;
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
    float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 1);
    dotp1_0 = dotp1_0 + _shfl_xor_sync_60;
    float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 2);
    dotp1_0 = dotp1_0 + _shfl_xor_sync_61;
    float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, dotp1_0, 4);
    dotp1_0 = dotp1_0 + _shfl_xor_sync_62;
#endif
        float vnp1_0 = (v_regs[0] - dotp1_0) * betap1;
#if __CUDA_ARCH__ == 900
        float vn2p1_0_x = vnp1_0;
        float vn2p1_0_y = vnp1_0;
        float _fma_320;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_320)
            : "f"(k_regs[0]), "f"(vn2p1_0_x), "f"(S[0]));
        float _fma_321;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_321)
            : "f"(k_regs[1]), "f"(vn2p1_0_y), "f"(S[1]));
        S[0] = _fma_320;
        S[1] = _fma_321;
        float _fma_322;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_322)
            : "f"(k_regs[2]), "f"(vn2p1_0_x), "f"(S[2]));
        float _fma_323;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_323)
            : "f"(k_regs[3]), "f"(vn2p1_0_y), "f"(S[3]));
        S[2] = _fma_322;
        S[3] = _fma_323;
        float _fma_324;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_324)
            : "f"(k_regs[4]), "f"(vn2p1_0_x), "f"(S[4]));
        float _fma_325;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_325)
            : "f"(k_regs[5]), "f"(vn2p1_0_y), "f"(S[5]));
        S[4] = _fma_324;
        S[5] = _fma_325;
        float _fma_326;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_326)
            : "f"(k_regs[6]), "f"(vn2p1_0_x), "f"(S[6]));
        float _fma_327;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_327)
            : "f"(k_regs[7]), "f"(vn2p1_0_y), "f"(S[7]));
        S[6] = _fma_326;
        S[7] = _fma_327;
        float _fma_328;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_328)
            : "f"(k_regs[8]), "f"(vn2p1_0_x), "f"(S[8]));
        float _fma_329;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_329)
            : "f"(k_regs[9]), "f"(vn2p1_0_y), "f"(S[9]));
        S[8] = _fma_328;
        S[9] = _fma_329;
        float _fma_330;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_330)
            : "f"(k_regs[10]), "f"(vn2p1_0_x), "f"(S[10]));
        float _fma_331;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_331)
            : "f"(k_regs[11]), "f"(vn2p1_0_y), "f"(S[11]));
        S[10] = _fma_330;
        S[11] = _fma_331;
        float _fma_332;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_332)
            : "f"(k_regs[12]), "f"(vn2p1_0_x), "f"(S[12]));
        float _fma_333;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_333)
            : "f"(k_regs[13]), "f"(vn2p1_0_y), "f"(S[13]));
        S[12] = _fma_332;
        S[13] = _fma_333;
        float _fma_334;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_334)
            : "f"(k_regs[14]), "f"(vn2p1_0_x), "f"(S[14]));
        float _fma_335;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_335)
            : "f"(k_regs[15]), "f"(vn2p1_0_y), "f"(S[15]));
        S[14] = _fma_334;
        S[15] = _fma_335;
        float accp1_1_0_x = 0.0f;
        float accp1_1_0_y = 0.0f;
        float accp1_1_1_x = 0.0f;
        float accp1_1_1_y = 0.0f;
        float _fma_336;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_336)
            : "f"(S[16]), "f"(k_regs[0]), "f"(accp1_1_0_x));
        float _fma_337;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_337)
            : "f"(S[17]), "f"(k_regs[1]), "f"(accp1_1_0_y));
        accp1_1_0_x = _fma_336;
        accp1_1_0_y = _fma_337;
        float _fma_338;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_338)
            : "f"(S[18]), "f"(k_regs[2]), "f"(accp1_1_1_x));
        float _fma_339;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_339)
            : "f"(S[19]), "f"(k_regs[3]), "f"(accp1_1_1_y));
        accp1_1_1_x = _fma_338;
        accp1_1_1_y = _fma_339;
        float _fma_340;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_340)
            : "f"(S[20]), "f"(k_regs[4]), "f"(accp1_1_0_x));
        float _fma_341;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_341)
            : "f"(S[21]), "f"(k_regs[5]), "f"(accp1_1_0_y));
        accp1_1_0_x = _fma_340;
        accp1_1_0_y = _fma_341;
        float _fma_342;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_342)
            : "f"(S[22]), "f"(k_regs[6]), "f"(accp1_1_1_x));
        float _fma_343;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_343)
            : "f"(S[23]), "f"(k_regs[7]), "f"(accp1_1_1_y));
        accp1_1_1_x = _fma_342;
        accp1_1_1_y = _fma_343;
        float _fma_344;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_344)
            : "f"(S[24]), "f"(k_regs[8]), "f"(accp1_1_0_x));
        float _fma_345;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_345)
            : "f"(S[25]), "f"(k_regs[9]), "f"(accp1_1_0_y));
        accp1_1_0_x = _fma_344;
        accp1_1_0_y = _fma_345;
        float _fma_346;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_346)
            : "f"(S[26]), "f"(k_regs[10]), "f"(accp1_1_1_x));
        float _fma_347;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_347)
            : "f"(S[27]), "f"(k_regs[11]), "f"(accp1_1_1_y));
        accp1_1_1_x = _fma_346;
        accp1_1_1_y = _fma_347;
        float _fma_348;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_348)
            : "f"(S[28]), "f"(k_regs[12]), "f"(accp1_1_0_x));
        float _fma_349;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_349)
            : "f"(S[29]), "f"(k_regs[13]), "f"(accp1_1_0_y));
        accp1_1_0_x = _fma_348;
        accp1_1_0_y = _fma_349;
        float _fma_350;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_350)
            : "f"(S[30]), "f"(k_regs[14]), "f"(accp1_1_1_x));
        float _fma_351;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_351)
            : "f"(S[31]), "f"(k_regs[15]), "f"(accp1_1_1_y));
        accp1_1_1_x = _fma_350;
        accp1_1_1_y = _fma_351;
        float _add_81;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(accp1_1_0_x), "f"(accp1_1_1_x));
        float _add_82;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(accp1_1_0_y), "f"(accp1_1_1_y));
        float dotp1_1 = _add_81 + _add_82;
        float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 1);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_83;
        float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 2);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_84;
        float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 4);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_85;
#elif __CUDA_ARCH__ == 1000
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
        float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 1);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_67;
        float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 2);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_68;
        float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 4);
        dotp1_1 = dotp1_1 + _shfl_xor_sync_69;
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
    float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 1);
    dotp1_1 = dotp1_1 + _shfl_xor_sync_63;
    float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 2);
    dotp1_1 = dotp1_1 + _shfl_xor_sync_64;
    float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, dotp1_1, 4);
    dotp1_1 = dotp1_1 + _shfl_xor_sync_65;
#endif
        float vnp1_1 = (v_regs[1] - dotp1_1) * betap1;
#if __CUDA_ARCH__ == 900
        float vn2p1_1_x = vnp1_1;
        float vn2p1_1_y = vnp1_1;
        float _fma_352;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_352)
            : "f"(k_regs[0]), "f"(vn2p1_1_x), "f"(S[16]));
        float _fma_353;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_353)
            : "f"(k_regs[1]), "f"(vn2p1_1_y), "f"(S[17]));
        S[16] = _fma_352;
        S[17] = _fma_353;
        float _fma_354;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_354)
            : "f"(k_regs[2]), "f"(vn2p1_1_x), "f"(S[18]));
        float _fma_355;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_355)
            : "f"(k_regs[3]), "f"(vn2p1_1_y), "f"(S[19]));
        S[18] = _fma_354;
        S[19] = _fma_355;
        float _fma_356;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_356)
            : "f"(k_regs[4]), "f"(vn2p1_1_x), "f"(S[20]));
        float _fma_357;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_357)
            : "f"(k_regs[5]), "f"(vn2p1_1_y), "f"(S[21]));
        S[20] = _fma_356;
        S[21] = _fma_357;
        float _fma_358;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_358)
            : "f"(k_regs[6]), "f"(vn2p1_1_x), "f"(S[22]));
        float _fma_359;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_359)
            : "f"(k_regs[7]), "f"(vn2p1_1_y), "f"(S[23]));
        S[22] = _fma_358;
        S[23] = _fma_359;
        float _fma_360;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_360)
            : "f"(k_regs[8]), "f"(vn2p1_1_x), "f"(S[24]));
        float _fma_361;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_361)
            : "f"(k_regs[9]), "f"(vn2p1_1_y), "f"(S[25]));
        S[24] = _fma_360;
        S[25] = _fma_361;
        float _fma_362;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_362)
            : "f"(k_regs[10]), "f"(vn2p1_1_x), "f"(S[26]));
        float _fma_363;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_363)
            : "f"(k_regs[11]), "f"(vn2p1_1_y), "f"(S[27]));
        S[26] = _fma_362;
        S[27] = _fma_363;
        float _fma_364;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_364)
            : "f"(k_regs[12]), "f"(vn2p1_1_x), "f"(S[28]));
        float _fma_365;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_365)
            : "f"(k_regs[13]), "f"(vn2p1_1_y), "f"(S[29]));
        S[28] = _fma_364;
        S[29] = _fma_365;
        float _fma_366;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_366)
            : "f"(k_regs[14]), "f"(vn2p1_1_x), "f"(S[30]));
        float _fma_367;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_367)
            : "f"(k_regs[15]), "f"(vn2p1_1_y), "f"(S[31]));
        S[30] = _fma_366;
        S[31] = _fma_367;
        int mp2 = t_1 * 3 + 2;
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
        int mp2 = t_2 * 3 + 2;
#endif
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
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                     : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])),
                       "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                     : "r"(sV_addr + (unsigned int)(mp2 * 256) + group_byte));
        float betap2 = sBeta[mp2];
#if __CUDA_ARCH__ == 900
        float accp2_0_0_x = 0.0f;
        float accp2_0_0_y = 0.0f;
        float accp2_0_1_x = 0.0f;
        float accp2_0_1_y = 0.0f;
        float _fma_368;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_368)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accp2_0_0_x));
        float _fma_369;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_369)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accp2_0_0_y));
        accp2_0_0_x = _fma_368;
        accp2_0_0_y = _fma_369;
        float _fma_370;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_370)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accp2_0_1_x));
        float _fma_371;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_371)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accp2_0_1_y));
        accp2_0_1_x = _fma_370;
        accp2_0_1_y = _fma_371;
        float _fma_372;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_372)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accp2_0_0_x));
        float _fma_373;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_373)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accp2_0_0_y));
        accp2_0_0_x = _fma_372;
        accp2_0_0_y = _fma_373;
        float _fma_374;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_374)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accp2_0_1_x));
        float _fma_375;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_375)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accp2_0_1_y));
        accp2_0_1_x = _fma_374;
        accp2_0_1_y = _fma_375;
        float _fma_376;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_376)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accp2_0_0_x));
        float _fma_377;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_377)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accp2_0_0_y));
        accp2_0_0_x = _fma_376;
        accp2_0_0_y = _fma_377;
        float _fma_378;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_378)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accp2_0_1_x));
        float _fma_379;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_379)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accp2_0_1_y));
        accp2_0_1_x = _fma_378;
        accp2_0_1_y = _fma_379;
        float _fma_380;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_380)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accp2_0_0_x));
        float _fma_381;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_381)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accp2_0_0_y));
        accp2_0_0_x = _fma_380;
        accp2_0_0_y = _fma_381;
        float _fma_382;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_382)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accp2_0_1_x));
        float _fma_383;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_383)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accp2_0_1_y));
        accp2_0_1_x = _fma_382;
        accp2_0_1_y = _fma_383;
        float _add_86;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(accp2_0_0_x), "f"(accp2_0_1_x));
        float _add_87;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(accp2_0_0_y), "f"(accp2_0_1_y));
        float dotp2_0 = _add_86 + _add_87;
        float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 1);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_88;
        float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 2);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_89;
        float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 4);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_90;
#elif __CUDA_ARCH__ == 1000
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
        float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 1);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_70;
        float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 2);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_71;
        float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 4);
        dotp2_0 = dotp2_0 + _shfl_xor_sync_72;
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
    float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 1);
    dotp2_0 = dotp2_0 + _shfl_xor_sync_66;
    float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 2);
    dotp2_0 = dotp2_0 + _shfl_xor_sync_67;
    float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, dotp2_0, 4);
    dotp2_0 = dotp2_0 + _shfl_xor_sync_68;
#endif
        float vnp2_0 = (v_regs[0] - dotp2_0) * betap2;
#if __CUDA_ARCH__ == 900
        float vn2p2_0_x = vnp2_0;
        float vn2p2_0_y = vnp2_0;
        float _fma_384;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_384)
            : "f"(k_regs[0]), "f"(vn2p2_0_x), "f"(S[0]));
        float _fma_385;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_385)
            : "f"(k_regs[1]), "f"(vn2p2_0_y), "f"(S[1]));
        S[0] = _fma_384;
        S[1] = _fma_385;
        float _fma_386;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_386)
            : "f"(k_regs[2]), "f"(vn2p2_0_x), "f"(S[2]));
        float _fma_387;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_387)
            : "f"(k_regs[3]), "f"(vn2p2_0_y), "f"(S[3]));
        S[2] = _fma_386;
        S[3] = _fma_387;
        float _fma_388;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_388)
            : "f"(k_regs[4]), "f"(vn2p2_0_x), "f"(S[4]));
        float _fma_389;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_389)
            : "f"(k_regs[5]), "f"(vn2p2_0_y), "f"(S[5]));
        S[4] = _fma_388;
        S[5] = _fma_389;
        float _fma_390;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_390)
            : "f"(k_regs[6]), "f"(vn2p2_0_x), "f"(S[6]));
        float _fma_391;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_391)
            : "f"(k_regs[7]), "f"(vn2p2_0_y), "f"(S[7]));
        S[6] = _fma_390;
        S[7] = _fma_391;
        float _fma_392;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_392)
            : "f"(k_regs[8]), "f"(vn2p2_0_x), "f"(S[8]));
        float _fma_393;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_393)
            : "f"(k_regs[9]), "f"(vn2p2_0_y), "f"(S[9]));
        S[8] = _fma_392;
        S[9] = _fma_393;
        float _fma_394;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_394)
            : "f"(k_regs[10]), "f"(vn2p2_0_x), "f"(S[10]));
        float _fma_395;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_395)
            : "f"(k_regs[11]), "f"(vn2p2_0_y), "f"(S[11]));
        S[10] = _fma_394;
        S[11] = _fma_395;
        float _fma_396;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_396)
            : "f"(k_regs[12]), "f"(vn2p2_0_x), "f"(S[12]));
        float _fma_397;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_397)
            : "f"(k_regs[13]), "f"(vn2p2_0_y), "f"(S[13]));
        S[12] = _fma_396;
        S[13] = _fma_397;
        float _fma_398;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_398)
            : "f"(k_regs[14]), "f"(vn2p2_0_x), "f"(S[14]));
        float _fma_399;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_399)
            : "f"(k_regs[15]), "f"(vn2p2_0_y), "f"(S[15]));
        S[14] = _fma_398;
        S[15] = _fma_399;
        float accp2_1_0_x = 0.0f;
        float accp2_1_0_y = 0.0f;
        float accp2_1_1_x = 0.0f;
        float accp2_1_1_y = 0.0f;
        float _fma_400;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_400)
            : "f"(S[16]), "f"(k_regs[0]), "f"(accp2_1_0_x));
        float _fma_401;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_401)
            : "f"(S[17]), "f"(k_regs[1]), "f"(accp2_1_0_y));
        accp2_1_0_x = _fma_400;
        accp2_1_0_y = _fma_401;
        float _fma_402;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_402)
            : "f"(S[18]), "f"(k_regs[2]), "f"(accp2_1_1_x));
        float _fma_403;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_403)
            : "f"(S[19]), "f"(k_regs[3]), "f"(accp2_1_1_y));
        accp2_1_1_x = _fma_402;
        accp2_1_1_y = _fma_403;
        float _fma_404;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_404)
            : "f"(S[20]), "f"(k_regs[4]), "f"(accp2_1_0_x));
        float _fma_405;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_405)
            : "f"(S[21]), "f"(k_regs[5]), "f"(accp2_1_0_y));
        accp2_1_0_x = _fma_404;
        accp2_1_0_y = _fma_405;
        float _fma_406;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_406)
            : "f"(S[22]), "f"(k_regs[6]), "f"(accp2_1_1_x));
        float _fma_407;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_407)
            : "f"(S[23]), "f"(k_regs[7]), "f"(accp2_1_1_y));
        accp2_1_1_x = _fma_406;
        accp2_1_1_y = _fma_407;
        float _fma_408;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_408)
            : "f"(S[24]), "f"(k_regs[8]), "f"(accp2_1_0_x));
        float _fma_409;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_409)
            : "f"(S[25]), "f"(k_regs[9]), "f"(accp2_1_0_y));
        accp2_1_0_x = _fma_408;
        accp2_1_0_y = _fma_409;
        float _fma_410;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_410)
            : "f"(S[26]), "f"(k_regs[10]), "f"(accp2_1_1_x));
        float _fma_411;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_411)
            : "f"(S[27]), "f"(k_regs[11]), "f"(accp2_1_1_y));
        accp2_1_1_x = _fma_410;
        accp2_1_1_y = _fma_411;
        float _fma_412;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_412)
            : "f"(S[28]), "f"(k_regs[12]), "f"(accp2_1_0_x));
        float _fma_413;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_413)
            : "f"(S[29]), "f"(k_regs[13]), "f"(accp2_1_0_y));
        accp2_1_0_x = _fma_412;
        accp2_1_0_y = _fma_413;
        float _fma_414;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_414)
            : "f"(S[30]), "f"(k_regs[14]), "f"(accp2_1_1_x));
        float _fma_415;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_415)
            : "f"(S[31]), "f"(k_regs[15]), "f"(accp2_1_1_y));
        accp2_1_1_x = _fma_414;
        accp2_1_1_y = _fma_415;
        float _add_91;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(accp2_1_0_x), "f"(accp2_1_1_x));
        float _add_92;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(accp2_1_0_y), "f"(accp2_1_1_y));
        float dotp2_1 = _add_91 + _add_92;
        float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 1);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_93;
        float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 2);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_94;
        float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 4);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_95;
#elif __CUDA_ARCH__ == 1000
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
        float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 1);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_73;
        float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 2);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_74;
        float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 4);
        dotp2_1 = dotp2_1 + _shfl_xor_sync_75;
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
    float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 1);
    dotp2_1 = dotp2_1 + _shfl_xor_sync_69;
    float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 2);
    dotp2_1 = dotp2_1 + _shfl_xor_sync_70;
    float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, dotp2_1, 4);
    dotp2_1 = dotp2_1 + _shfl_xor_sync_71;
#endif
        float vnp2_1 = (v_regs[1] - dotp2_1) * betap2;
#if __CUDA_ARCH__ == 900
        float vn2p2_1_x = vnp2_1;
        float vn2p2_1_y = vnp2_1;
        float _fma_416;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_416)
            : "f"(k_regs[0]), "f"(vn2p2_1_x), "f"(S[16]));
        float _fma_417;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_417)
            : "f"(k_regs[1]), "f"(vn2p2_1_y), "f"(S[17]));
        S[16] = _fma_416;
        S[17] = _fma_417;
        float _fma_418;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_418)
            : "f"(k_regs[2]), "f"(vn2p2_1_x), "f"(S[18]));
        float _fma_419;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_419)
            : "f"(k_regs[3]), "f"(vn2p2_1_y), "f"(S[19]));
        S[18] = _fma_418;
        S[19] = _fma_419;
        float _fma_420;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_420)
            : "f"(k_regs[4]), "f"(vn2p2_1_x), "f"(S[20]));
        float _fma_421;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_421)
            : "f"(k_regs[5]), "f"(vn2p2_1_y), "f"(S[21]));
        S[20] = _fma_420;
        S[21] = _fma_421;
        float _fma_422;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_422)
            : "f"(k_regs[6]), "f"(vn2p2_1_x), "f"(S[22]));
        float _fma_423;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_423)
            : "f"(k_regs[7]), "f"(vn2p2_1_y), "f"(S[23]));
        S[22] = _fma_422;
        S[23] = _fma_423;
        float _fma_424;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_424)
            : "f"(k_regs[8]), "f"(vn2p2_1_x), "f"(S[24]));
        float _fma_425;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_425)
            : "f"(k_regs[9]), "f"(vn2p2_1_y), "f"(S[25]));
        S[24] = _fma_424;
        S[25] = _fma_425;
        float _fma_426;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_426)
            : "f"(k_regs[10]), "f"(vn2p2_1_x), "f"(S[26]));
        float _fma_427;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_427)
            : "f"(k_regs[11]), "f"(vn2p2_1_y), "f"(S[27]));
        S[26] = _fma_426;
        S[27] = _fma_427;
        float _fma_428;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_428)
            : "f"(k_regs[12]), "f"(vn2p2_1_x), "f"(S[28]));
        float _fma_429;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_429)
            : "f"(k_regs[13]), "f"(vn2p2_1_y), "f"(S[29]));
        S[28] = _fma_428;
        S[29] = _fma_429;
        float _fma_430;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_430)
            : "f"(k_regs[14]), "f"(vn2p2_1_x), "f"(S[30]));
        float _fma_431;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_431)
            : "f"(k_regs[15]), "f"(vn2p2_1_y), "f"(S[31]));
        S[30] = _fma_430;
        S[31] = _fma_431;
        unsigned int qbasep = sQ_addr + (unsigned int)(t_1 * 512) + lane_byte;
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
        unsigned int qbasep = sQ_addr + (unsigned int)(t_2 * 512) + lane_byte;
#endif
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
        float _fma_432;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_432)
            : "f"(S[0]), "f"(k_regs[0]), "f"(accpq_0_0_x));
        float _fma_433;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_433)
            : "f"(S[1]), "f"(k_regs[1]), "f"(accpq_0_0_y));
        accpq_0_0_x = _fma_432;
        accpq_0_0_y = _fma_433;
        float _fma_434;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_434)
            : "f"(S[2]), "f"(k_regs[2]), "f"(accpq_0_1_x));
        float _fma_435;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_435)
            : "f"(S[3]), "f"(k_regs[3]), "f"(accpq_0_1_y));
        accpq_0_1_x = _fma_434;
        accpq_0_1_y = _fma_435;
        float _fma_436;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_436)
            : "f"(S[4]), "f"(k_regs[4]), "f"(accpq_0_0_x));
        float _fma_437;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_437)
            : "f"(S[5]), "f"(k_regs[5]), "f"(accpq_0_0_y));
        accpq_0_0_x = _fma_436;
        accpq_0_0_y = _fma_437;
        float _fma_438;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_438)
            : "f"(S[6]), "f"(k_regs[6]), "f"(accpq_0_1_x));
        float _fma_439;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_439)
            : "f"(S[7]), "f"(k_regs[7]), "f"(accpq_0_1_y));
        accpq_0_1_x = _fma_438;
        accpq_0_1_y = _fma_439;
        float _fma_440;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_440)
            : "f"(S[8]), "f"(k_regs[8]), "f"(accpq_0_0_x));
        float _fma_441;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_441)
            : "f"(S[9]), "f"(k_regs[9]), "f"(accpq_0_0_y));
        accpq_0_0_x = _fma_440;
        accpq_0_0_y = _fma_441;
        float _fma_442;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_442)
            : "f"(S[10]), "f"(k_regs[10]), "f"(accpq_0_1_x));
        float _fma_443;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_443)
            : "f"(S[11]), "f"(k_regs[11]), "f"(accpq_0_1_y));
        accpq_0_1_x = _fma_442;
        accpq_0_1_y = _fma_443;
        float _fma_444;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_444)
            : "f"(S[12]), "f"(k_regs[12]), "f"(accpq_0_0_x));
        float _fma_445;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_445)
            : "f"(S[13]), "f"(k_regs[13]), "f"(accpq_0_0_y));
        accpq_0_0_x = _fma_444;
        accpq_0_0_y = _fma_445;
        float _fma_446;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_446)
            : "f"(S[14]), "f"(k_regs[14]), "f"(accpq_0_1_x));
        float _fma_447;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_447)
            : "f"(S[15]), "f"(k_regs[15]), "f"(accpq_0_1_y));
        accpq_0_1_x = _fma_446;
        accpq_0_1_y = _fma_447;
        float _add_96;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(accpq_0_0_x), "f"(accpq_0_1_x));
        float _add_97;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(accpq_0_0_y), "f"(accpq_0_1_y));
        float dotpq_0 = _add_96 + _add_97;
        float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 1);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_98;
        float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 2);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_99;
        float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 4);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_100;
#elif __CUDA_ARCH__ == 1000
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
        float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 1);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_76;
        float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 2);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_77;
        float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 4);
        dotpq_0 = dotpq_0 + _shfl_xor_sync_78;
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
    float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 1);
    dotpq_0 = dotpq_0 + _shfl_xor_sync_72;
    float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 2);
    dotpq_0 = dotpq_0 + _shfl_xor_sync_73;
    float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, dotpq_0, 4);
    dotpq_0 = dotpq_0 + _shfl_xor_sync_74;
#endif
        o4[0] = dotpq_0;
#if __CUDA_ARCH__ == 900
        float accpq_1_0_x = 0.0f;
        float accpq_1_0_y = 0.0f;
        float accpq_1_1_x = 0.0f;
        float accpq_1_1_y = 0.0f;
        float _fma_448;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_448)
            : "f"(S[16]), "f"(k_regs[0]), "f"(accpq_1_0_x));
        float _fma_449;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_449)
            : "f"(S[17]), "f"(k_regs[1]), "f"(accpq_1_0_y));
        accpq_1_0_x = _fma_448;
        accpq_1_0_y = _fma_449;
        float _fma_450;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_450)
            : "f"(S[18]), "f"(k_regs[2]), "f"(accpq_1_1_x));
        float _fma_451;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_451)
            : "f"(S[19]), "f"(k_regs[3]), "f"(accpq_1_1_y));
        accpq_1_1_x = _fma_450;
        accpq_1_1_y = _fma_451;
        float _fma_452;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_452)
            : "f"(S[20]), "f"(k_regs[4]), "f"(accpq_1_0_x));
        float _fma_453;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_453)
            : "f"(S[21]), "f"(k_regs[5]), "f"(accpq_1_0_y));
        accpq_1_0_x = _fma_452;
        accpq_1_0_y = _fma_453;
        float _fma_454;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_454)
            : "f"(S[22]), "f"(k_regs[6]), "f"(accpq_1_1_x));
        float _fma_455;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_455)
            : "f"(S[23]), "f"(k_regs[7]), "f"(accpq_1_1_y));
        accpq_1_1_x = _fma_454;
        accpq_1_1_y = _fma_455;
        float _fma_456;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_456)
            : "f"(S[24]), "f"(k_regs[8]), "f"(accpq_1_0_x));
        float _fma_457;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_457)
            : "f"(S[25]), "f"(k_regs[9]), "f"(accpq_1_0_y));
        accpq_1_0_x = _fma_456;
        accpq_1_0_y = _fma_457;
        float _fma_458;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_458)
            : "f"(S[26]), "f"(k_regs[10]), "f"(accpq_1_1_x));
        float _fma_459;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_459)
            : "f"(S[27]), "f"(k_regs[11]), "f"(accpq_1_1_y));
        accpq_1_1_x = _fma_458;
        accpq_1_1_y = _fma_459;
        float _fma_460;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_460)
            : "f"(S[28]), "f"(k_regs[12]), "f"(accpq_1_0_x));
        float _fma_461;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_461)
            : "f"(S[29]), "f"(k_regs[13]), "f"(accpq_1_0_y));
        accpq_1_0_x = _fma_460;
        accpq_1_0_y = _fma_461;
        float _fma_462;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_462)
            : "f"(S[30]), "f"(k_regs[14]), "f"(accpq_1_1_x));
        float _fma_463;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_463)
            : "f"(S[31]), "f"(k_regs[15]), "f"(accpq_1_1_y));
        accpq_1_1_x = _fma_462;
        accpq_1_1_y = _fma_463;
        float _add_101;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_101) : "f"(accpq_1_0_x), "f"(accpq_1_1_x));
        float _add_102;
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(accpq_1_0_y), "f"(accpq_1_1_y));
        float dotpq_1 = _add_101 + _add_102;
        float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 1);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_103;
        float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 2);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_104;
        float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 4);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_105;
#elif __CUDA_ARCH__ == 1000
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
        float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 1);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_79;
        float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 2);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_80;
        float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 4);
        dotpq_1 = dotpq_1 + _shfl_xor_sync_81;
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
    float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 1);
    dotpq_1 = dotpq_1 + _shfl_xor_sync_75;
    float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 2);
    dotpq_1 = dotpq_1 + _shfl_xor_sync_76;
    float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, dotpq_1, 4);
    dotpq_1 = dotpq_1 + _shfl_xor_sync_77;
#endif
        o4[1] = dotpq_1;
        if (u == 0) {
#if __CUDA_ARCH__ == 900
          long long out_index_1 = (long long)n * out_stride_p0 + (long long)t_1 * out_stride_p1 +
                                  (long long)h * out_stride_p2 + (long long)row0;
#else
          long long out_index_1 = (long long)n * out_stride_p0 + (long long)t_2 * out_stride_p1 +
                                  (long long)h * out_stride_p2 + (long long)row0;
#endif
          unsigned int _packed_1[1];
#pragma unroll
          for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 =
                __float22bfloat162_rn(make_float2(o4[_lp * 2 + 0], o4[_lp * 2 + 1 + 0]));
            _packed_1[_lp] = *(uint32_t*)&_bf2;
          }
          reinterpret_cast<unsigned*>(out + (out_index_1))[0] =
              reinterpret_cast<unsigned*>(_packed_1)[0];
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
        }
      }
    }
  }
}

}  // extern "C"
