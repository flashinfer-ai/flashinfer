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
#define SMEM_SSTATE_OFF 0
#define SMEM_SSTATE_STAGE_BYTES 4096
#define SMEM_SSTATE_STRIDE 4096
#define SMEM_SQ_OFF 4096
#define SMEM_SQ_STAGE_BYTES 512
#define SMEM_SQ_STRIDE 512
#define SMEM_SK_OFF 4608
#define SMEM_SK_STAGE_BYTES 1536
#define SMEM_SK_STRIDE 1536
#define SMEM_SV_OFF 6144
#define SMEM_SV_STAGE_BYTES 96
#define SMEM_SV_STRIDE 96
#define SMEM_SALPHA_OFF 6240
#define SMEM_SALPHA_STAGE_BYTES 4
#define SMEM_SALPHA_STRIDE 4
#define SMEM_SBETA_OFF 6244
#define SMEM_SBETA_STAGE_BYTES 12
#define SMEM_SBETA_STRIDE 12
#define SMEM_TOTAL 6400
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ float approx_rcp(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

extern "C" {

__global__ __launch_bounds__(128) void kernel_cake_gdp_068a895d0c8fa77ed2f9(
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
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  const int cta_rank = 0;

  // Kernel setup ops
  float* sState = reinterpret_cast<float*>(smem_raw + 0);
  const int sState_addr = smem + 0;
  float* sQ = reinterpret_cast<float*>(smem_raw + 4096);
  const int sQ_addr = smem + 4096;
  float* sK = reinterpret_cast<float*>(smem_raw + 4608);
  const int sK_addr = smem + 4608;
  float* sV = reinterpret_cast<float*>(smem_raw + 6144);
  const int sV_addr = smem + 6144;
  float* sAlpha = reinterpret_cast<float*>(smem_raw + 6240);
  const int sAlpha_addr = smem + 6240;
  float* sBeta = reinterpret_cast<float*>(smem_raw + 6244);
  const int sBeta_addr = smem + 6244;

  // === Task calls (dependency order) ===
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
#pragma unroll
    for (int copy_iter = 0; copy_iter < 2; copy_iter++) {
      int copy_seg = copy_iter * 128 + tid;
      int copy_row = copy_seg / 32;
      int copy_k_vec = copy_seg - copy_row * 32;
      int copy_v_row = split_v_base + copy_row;
      int copy_k_base = copy_k_vec * 4;
      int copy_dst = sState_addr + (unsigned int)((copy_row * 128 + copy_k_base) * 4);
      asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(copy_dst),
                   "l"(state + (read_state_head_base + (long long)copy_v_row * state_stride_p2 +
                                (long long)copy_k_base)));
    }
    asm volatile("cp.async.commit_group;");
    if (warp_local < steps) {
      float k_pair0_x = pre_k[0];
      float k_pair0_y = pre_k[1];
      float k_pair1_x = pre_k[2];
      float k_pair1_y = pre_k[3];
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
      float _warp_reduce_2 = sum_k;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
      sum_k = _warp_reduce_2;
      float k_norm = rsqrtf(sum_k + 1e-06f);
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
      sK[pre_m * 128 + k_col0] = k_pair0_x;
      sK[pre_m * 128 + k_col0 + 1] = k_pair0_y;
      sK[pre_m * 128 + k_col0 + 2] = k_pair1_x;
      sK[pre_m * 128 + k_col0 + 3] = k_pair1_y;
      if (lane_local < 8) {
        sV[pre_m * 8 + lane_local] = pre_s[0];
      }
      if (lane_local == 0) {
        float _exp_4 = __expf(-pre_s[1]);
        float _rcp_3 = approx_rcp(1.0f + _exp_4);
        sBeta[pre_m] = _rcp_3;
      }
    }
    if (pre_q_first < num_tokens) {
      float q_raw_pair0_x = pre_q[0];
      float q_raw_pair0_y = pre_q[1];
      float q_raw_pair1_x = pre_q[2];
      float q_raw_pair1_y = pre_q[3];
      float q_norm = scale;
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
      float _warp_reduce_5 = sum_q;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_5 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_5, offset);
      sum_q = _warp_reduce_5;
      float _rsqrt_6 = rsqrtf(sum_q + 1e-06f);
      q_norm = _rsqrt_6 * scale;
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
      sQ[pre_t * 128 + k_col0] = q_pair0_x;
      sQ[pre_t * 128 + k_col0 + 1] = q_pair0_y;
      sQ[pre_t * 128 + k_col0 + 2] = q_pair1_x;
      sQ[pre_t * 128 + k_col0 + 3] = q_pair1_y;
      if (lane_local == 0) {
        float x = pre_s[2] + pre_s[3];
        float softplus_x = x;
        if (x <= 20.0f) {
          float _exp_8 = __expf(x);
          float _log2_7;
          asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_7) : "f"(1.0f + _exp_8));
          softplus_x = _log2_7 * 0.6931471805599453f;
        }
        float _exp_9 = __expf(pre_s[4]);
        float g_log = (-_exp_9) * softplus_x;
        float _exp_10 = __expf(g_log);
        sAlpha[pre_t] = _exp_10;
      }
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
#pragma unroll
    for (int row_group = 0; row_group < 8; row_group += 4) {
      int local_row = row_group + warp_local;
      int v_row = split_v_base + local_row;
      float h_pair0[2];
      h_pair0[0] = sState[local_row * 128 + k_col0];
      h_pair0[1] = sState[local_row * 128 + k_col0 + 1];
      float h_pair1[2];
      h_pair1[0] = sState[local_row * 128 + k_col0 + 2];
      h_pair1[1] = sState[local_row * 128 + k_col0 + 3];
#pragma unroll
      for (int t = 0; t < 1; t++) {
        float decay_val = sAlpha[t];
        float decay_pair_x = decay_val;
        float decay_pair_y = decay_val;
        float _mul_8;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(h_pair0[0]), "f"(decay_pair_x));
        float _mul_9;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(h_pair0[1]), "f"(decay_pair_y));
        h_pair0[0] = _mul_8;
        h_pair0[1] = _mul_9;
        float _mul_10;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(h_pair1[0]), "f"(decay_pair_x));
        float _mul_11;
        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(h_pair1[1]), "f"(decay_pair_y));
        h_pair1[0] = _mul_10;
        h_pair1[1] = _mul_11;
#pragma unroll
        for (int j = 0; j < 3; j++) {
          int m = t * 3 + j;
          float k_pair0_x_1 = sK[m * 128 + k_col0];
          float k_pair0_y_1 = sK[m * 128 + k_col0 + 1];
          float k_pair1_x_1 = sK[m * 128 + k_col0 + 2];
          float k_pair1_y_1 = sK[m * 128 + k_col0 + 3];
          float _fma_8;
          asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
              : "=f"(_fma_8)
              : "f"(h_pair0[0]), "f"(k_pair0_x_1), "f"(0.0f));
          float _fma_9;
          asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
              : "=f"(_fma_9)
              : "f"(h_pair0[1]), "f"(k_pair0_y_1), "f"(0.0f));
          float sum_hk_pair_x = _fma_8;
          float sum_hk_pair_y = _fma_9;
          float _fma_10;
          asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
              : "=f"(_fma_10)
              : "f"(h_pair1[0]), "f"(k_pair1_x_1), "f"(sum_hk_pair_x));
          float _fma_11;
          asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
              : "=f"(_fma_11)
              : "f"(h_pair1[1]), "f"(k_pair1_y_1), "f"(sum_hk_pair_y));
          sum_hk_pair_x = _fma_10;
          sum_hk_pair_y = _fma_11;
          float sum_hk = sum_hk_pair_x + sum_hk_pair_y;
          float _warp_reduce_11 = sum_hk;
#pragma unroll
          for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_11 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_11, offset);
          sum_hk = _warp_reduce_11;
          float v_val = sV[m * 8 + local_row];
          float beta_val = sBeta[m];
          float v_new = (v_val - sum_hk) * beta_val;
          float v_new_pair_x = v_new;
          float v_new_pair_y = v_new;
          float _fma_12;
          asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
              : "=f"(_fma_12)
              : "f"(k_pair0_x_1), "f"(v_new_pair_x), "f"(h_pair0[0]));
          float _fma_13;
          asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
              : "=f"(_fma_13)
              : "f"(k_pair0_y_1), "f"(v_new_pair_y), "f"(h_pair0[1]));
          h_pair0[0] = _fma_12;
          h_pair0[1] = _fma_13;
          float _fma_14;
          asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
              : "=f"(_fma_14)
              : "f"(k_pair1_x_1), "f"(v_new_pair_x), "f"(h_pair1[0]));
          float _fma_15;
          asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
              : "=f"(_fma_15)
              : "f"(k_pair1_y_1), "f"(v_new_pair_y), "f"(h_pair1[1]));
          h_pair1[0] = _fma_14;
          h_pair1[1] = _fma_15;
        }
        float q_pair0_x_1 = sQ[t * 128 + k_col0];
        float q_pair0_y_1 = sQ[t * 128 + k_col0 + 1];
        float q_pair1_x_1 = sQ[t * 128 + k_col0 + 2];
        float q_pair1_y_1 = sQ[t * 128 + k_col0 + 3];
        float _fma_16;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_16)
            : "f"(h_pair0[0]), "f"(q_pair0_x_1), "f"(0.0f));
        float _fma_17;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_17)
            : "f"(h_pair0[1]), "f"(q_pair0_y_1), "f"(0.0f));
        float sum_hq_pair_x = _fma_16;
        float sum_hq_pair_y = _fma_17;
        float _fma_18;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_18)
            : "f"(h_pair1[0]), "f"(q_pair1_x_1), "f"(sum_hq_pair_x));
        float _fma_19;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;"
            : "=f"(_fma_19)
            : "f"(h_pair1[1]), "f"(q_pair1_y_1), "f"(sum_hq_pair_y));
        sum_hq_pair_x = _fma_18;
        sum_hq_pair_y = _fma_19;
        float sum_hq = sum_hq_pair_x + sum_hq_pair_y;
        float _warp_reduce_12 = sum_hq;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
          _warp_reduce_12 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_12, offset);
        sum_hq = _warp_reduce_12;
        if (lane_local == 0) {
          long long out_index = (long long)n * out_stride_p0 + (long long)t * out_stride_p1 +
                                (long long)h * out_stride_p2;
          out[out_index + (long long)v_row] = sum_hq;
        }
        if (use_ssm_state_indices != 0) {
          int ssm_slot = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 +
                                           (long long)t * ssm_state_indices_stride_p1];
          if (ssm_slot >= 0 && state_pool_size > (long long)ssm_slot) {
            r_h[0] = h_pair0[0];
            r_h[1] = h_pair0[1];
            r_h[2] = h_pair1[0];
            r_h[3] = h_pair1[1];
            {
              float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
              *reinterpret_cast<float4*>(
                  state + (long long)ssm_slot * state_stride_p0 + (long long)h * state_stride_p1 +
                  (long long)v_row * state_stride_p2 + (long long)k_col0) = _v4;
            }
          }
        }
      }
      if (write_state_slot_valid != 0) {
        if (disable_state_update == 0) {
          if (use_ssm_state_indices == 0) {
            r_h[0] = h_pair0[0];
            r_h[1] = h_pair0[1];
            r_h[2] = h_pair1[0];
            r_h[3] = h_pair1[1];
            {
              float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
              *reinterpret_cast<float4*>(state + write_state_head_base +
                                         (long long)v_row * state_stride_p2 + (long long)k_col0) =
                  _v4;
            }
          }
        }
      }
    }
  }
}

}  // extern "C"
