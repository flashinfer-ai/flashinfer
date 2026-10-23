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
#define SMEM_SK1_OFF 2848
#define SMEM_SK1_STAGE_BYTES 1536
#define SMEM_SK1_STRIDE 1536
#define SMEM_SQ1_OFF 4384
#define SMEM_SQ1_STAGE_BYTES 512
#define SMEM_SQ1_STRIDE 512
#define SMEM_SV1_OFF 4896
#define SMEM_SV1_STAGE_BYTES 768
#define SMEM_SV1_STRIDE 768
#define SMEM_SALPHA1_OFF 5664
#define SMEM_SALPHA1_STAGE_BYTES 4
#define SMEM_SALPHA1_STRIDE 4
#define SMEM_SBETA1_OFF 5668
#define SMEM_SBETA1_STAGE_BYTES 12
#define SMEM_SBETA1_STRIDE 12
#define SMEM_SSSM1_OFF 5680
#define SMEM_SSSM1_STAGE_BYTES 4
#define SMEM_SSSM1_STRIDE 4
#define SMEM_SSTAGE_OFF 5696
#define SMEM_SSTAGE_STAGE_BYTES 32768
#define SMEM_SSTAGE_STRIDE 32768
#define SMEM_TOTAL 38528
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 4

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

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void kernel_cake_gdp_266d24496704b9063e28(
    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v,
    float* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a,
    float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out,
    int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices,
    int* __restrict__ ssm_state_indices, CakeFastDivmod heads, int num_tokens, float scale,
    int disable_state_update, int use_ssm_state_indices, long long state_size_p0,
    long long q_size_p0, long long state_stride_p1, long long state_stride_p2,
    long long state_stride_p0, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2,
    long long v_stride_p0, long long v_stride_p1, long long v_stride_p2, long long b_stride_p0,
    long long b_stride_p1, long long b_stride_p2, long long q_stride_p0, long long q_stride_p1,
    long long q_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2,
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
  float* sK1 = reinterpret_cast<float*>(smem_raw + 2848);
  const int sK1_addr = smem + 2848;
  float* sQ1 = reinterpret_cast<float*>(smem_raw + 4384);
  const int sQ1_addr = smem + 4384;
  float* sV1 = reinterpret_cast<float*>(smem_raw + 4896);
  const int sV1_addr = smem + 4896;
  float* sAlpha1 = reinterpret_cast<float*>(smem_raw + 5664);
  const int sAlpha1_addr = smem + 5664;
  float* sBeta1 = reinterpret_cast<float*>(smem_raw + 5668);
  const int sBeta1_addr = smem + 5668;
  int* sSsm1 = reinterpret_cast<int*>(smem_raw + 5680);
  const int sSsm1_addr = smem + 5680;
  float* sStage = reinterpret_cast<float*>(smem_raw + 5696);
  const int sStage_addr = smem + 5696;

  // === Task calls (declaration order) ===
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
  int k_col0 = lane_local * 4;
  int steps = 3;
  unsigned int blk = (unsigned int)blockIdx.x;
  unsigned int nblk = (unsigned int)gridDim.x;
  long long batch_rows = q_size_p0;
  uint32_t _fast_div_q_0 = (heads.divisor != 1)
                               ? (__umulhi((uint32_t)(blk), heads.multiplier) >> heads.shift_right)
                               : (uint32_t)(blk);
  uint32_t _fast_div_r_0 = (uint32_t)(blk)-_fast_div_q_0 * (uint32_t)(heads.divisor);
  int nA = (int)_fast_div_q_0;
  int hA = (int)_fast_div_r_0;
  uint32_t _fast_div_q_1 =
      (heads.divisor != 1)
          ? (__umulhi((uint32_t)(blk + nblk), heads.multiplier) >> heads.shift_right)
          : (uint32_t)(blk + nblk);
  uint32_t _fast_div_r_1 = (uint32_t)(blk + nblk) - _fast_div_q_1 * (uint32_t)(heads.divisor);
  int nB_raw = (int)_fast_div_q_1;
  int hB = (int)_fast_div_r_1;
  int b_in_batch = ((batch_rows > (long long)nB_raw) ? 1 : 0);
  int nB = ((b_in_batch != 0) ? nB_raw : 0);
  int readA_raw = initial_state_indices[nA];
  int readA_valid = ((readA_raw >= 0 && state_pool_size > (long long)readA_raw) ? 1 : 0);
  int writeA_raw = output_state_indices[nA];
  int writeA_valid = ((writeA_raw >= 0 && state_pool_size > (long long)writeA_raw) ? 1 : 0);
  long long rowsA_off =
      (long long)hA * state_stride_p1 + (long long)row0 * state_stride_p2 + lane_elem;
  long long readA_base = (long long)readA_raw * state_stride_p0 + rowsA_off;
  long long writeA_base = (long long)writeA_raw * state_stride_p0 + rowsA_off;
  int readB_raw = initial_state_indices[nB];
  int readB_valid =
      ((b_in_batch != 0 && readB_raw >= 0 && state_pool_size > (long long)readB_raw) ? 1 : 0);
  int writeB_raw = output_state_indices[nB];
  int writeB_valid = ((writeB_raw >= 0 && state_pool_size > (long long)writeB_raw) ? 1 : 0);
  long long rowsB_off =
      (long long)hB * state_stride_p1 + (long long)row0 * state_stride_p2 + lane_elem;
  long long readB_base = (long long)readB_raw * state_stride_p0 + rowsB_off;
  long long writeB_base = (long long)writeB_raw * state_stride_p0 + rowsB_off;
  float S[64];
  float2 S2[32];
  float k_regs[16];
  float2 k2[8];
  float v_regs[4];
  float o4[4];
  float r_h[4];
  if (readA_valid != 0) {
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + readA_base);
      S[0 + 0] = _v4.x;
      S[0 + 1] = _v4.y;
      S[0 + 2] = _v4.z;
      S[0 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + readA_base + 32);
      S[4 + 0] = _v4.x;
      S[4 + 1] = _v4.y;
      S[4 + 2] = _v4.z;
      S[4 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + readA_base + 64);
      S[8 + 0] = _v4.x;
      S[8 + 1] = _v4.y;
      S[8 + 2] = _v4.z;
      S[8 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + readA_base + 96);
      S[12 + 0] = _v4.x;
      S[12 + 1] = _v4.y;
      S[12 + 2] = _v4.z;
      S[12 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + readA_base + state_stride_p2);
      S[16 + 0] = _v4.x;
      S[16 + 1] = _v4.y;
      S[16 + 2] = _v4.z;
      S[16 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + readA_base + state_stride_p2 + 32);
      S[20 + 0] = _v4.x;
      S[20 + 1] = _v4.y;
      S[20 + 2] = _v4.z;
      S[20 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + readA_base + state_stride_p2 + 64);
      S[24 + 0] = _v4.x;
      S[24 + 1] = _v4.y;
      S[24 + 2] = _v4.z;
      S[24 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + readA_base + state_stride_p2 + 96);
      S[28 + 0] = _v4.x;
      S[28 + 1] = _v4.y;
      S[28 + 2] = _v4.z;
      S[28 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + readA_base + 2 * state_stride_p2);
      S[32 + 0] = _v4.x;
      S[32 + 1] = _v4.y;
      S[32 + 2] = _v4.z;
      S[32 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + readA_base + 2 * state_stride_p2 + 32);
      S[36 + 0] = _v4.x;
      S[36 + 1] = _v4.y;
      S[36 + 2] = _v4.z;
      S[36 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + readA_base + 2 * state_stride_p2 + 64);
      S[40 + 0] = _v4.x;
      S[40 + 1] = _v4.y;
      S[40 + 2] = _v4.z;
      S[40 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + readA_base + 2 * state_stride_p2 + 96);
      S[44 + 0] = _v4.x;
      S[44 + 1] = _v4.y;
      S[44 + 2] = _v4.z;
      S[44 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + readA_base + 3 * state_stride_p2);
      S[48 + 0] = _v4.x;
      S[48 + 1] = _v4.y;
      S[48 + 2] = _v4.z;
      S[48 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + readA_base + 3 * state_stride_p2 + 32);
      S[52 + 0] = _v4.x;
      S[52 + 1] = _v4.y;
      S[52 + 2] = _v4.z;
      S[52 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + readA_base + 3 * state_stride_p2 + 64);
      S[56 + 0] = _v4.x;
      S[56 + 1] = _v4.y;
      S[56 + 2] = _v4.z;
      S[56 + 3] = _v4.w;
    }
    {
      float4 _v4 = *reinterpret_cast<const float4*>(state + readA_base + 3 * state_stride_p2 + 96);
      S[60 + 0] = _v4.x;
      S[60 + 1] = _v4.y;
      S[60 + 2] = _v4.z;
      S[60 + 3] = _v4.w;
    }
  }
  if (readB_valid != 0) {
    unsigned int stage_dst0 = sStage_addr + (unsigned int)(tid_0 * 16);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(stage_dst0),
                 "l"(state + readB_base));
    unsigned int stage_dst1 = sStage_addr + (unsigned int)((128 + tid_0) * 16);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(stage_dst1),
                 "l"(state + (readB_base + 32)));
    unsigned int stage_dst2 = sStage_addr + (unsigned int)((256 + tid_0) * 16);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(stage_dst2),
                 "l"(state + (readB_base + 64)));
    unsigned int stage_dst3 = sStage_addr + (unsigned int)((384 + tid_0) * 16);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(stage_dst3),
                 "l"(state + (readB_base + 96)));
    unsigned int stage_dst4 = sStage_addr + (unsigned int)((512 + tid_0) * 16);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(stage_dst4),
                 "l"(state + (readB_base + state_stride_p2)));
    unsigned int stage_dst5 = sStage_addr + (unsigned int)((640 + tid_0) * 16);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(stage_dst5),
                 "l"(state + (readB_base + state_stride_p2 + 32)));
    unsigned int stage_dst6 = sStage_addr + (unsigned int)((768 + tid_0) * 16);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(stage_dst6),
                 "l"(state + (readB_base + state_stride_p2 + 64)));
    unsigned int stage_dst7 = sStage_addr + (unsigned int)((896 + tid_0) * 16);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(stage_dst7),
                 "l"(state + (readB_base + state_stride_p2 + 96)));
    unsigned int stage_dst8 = sStage_addr + (unsigned int)((1024 + tid_0) * 16);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(stage_dst8),
                 "l"(state + (readB_base + 2 * state_stride_p2)));
    unsigned int stage_dst9 = sStage_addr + (unsigned int)((1152 + tid_0) * 16);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(stage_dst9),
                 "l"(state + (readB_base + 2 * state_stride_p2 + 32)));
    unsigned int stage_dst10 = sStage_addr + (unsigned int)((1280 + tid_0) * 16);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(stage_dst10),
                 "l"(state + (readB_base + 2 * state_stride_p2 + 64)));
    unsigned int stage_dst11 = sStage_addr + (unsigned int)((1408 + tid_0) * 16);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(stage_dst11),
                 "l"(state + (readB_base + 2 * state_stride_p2 + 96)));
    unsigned int stage_dst12 = sStage_addr + (unsigned int)((1536 + tid_0) * 16);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(stage_dst12),
                 "l"(state + (readB_base + 3 * state_stride_p2)));
    unsigned int stage_dst13 = sStage_addr + (unsigned int)((1664 + tid_0) * 16);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(stage_dst13),
                 "l"(state + (readB_base + 3 * state_stride_p2 + 32)));
    unsigned int stage_dst14 = sStage_addr + (unsigned int)((1792 + tid_0) * 16);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(stage_dst14),
                 "l"(state + (readB_base + 3 * state_stride_p2 + 64)));
    unsigned int stage_dst15 = sStage_addr + (unsigned int)((1920 + tid_0) * 16);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;" ::"r"(stage_dst15),
                 "l"(state + (readB_base + 3 * state_stride_p2 + 96)));
    asm volatile("cp.async.commit_group;");
  }
  int xm0 = warp_local;
  int xml0 = ((xm0) < (steps - 1) ? (xm0) : (steps - 1));
  float bkxa0[4];
  float bvxa0[2];
  float bbxa0[1];
  long long bkixa0 =
      (long long)nA * k_stride_p0 + (long long)xml0 * k_stride_p1 + (long long)hA * k_stride_p2;
  {
    uint2 _vld_16;
    _vld_16 = *reinterpret_cast<const uint2*>(k + bkixa0 + (long long)k_col0);
    uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16);
#pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
      (&bkxa0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
      (&bkxa0[0 + _pair * 2])[1] =
          __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
    }
  }
  long long bvixa0 =
      (long long)nA * v_stride_p0 + (long long)xml0 * v_stride_p1 + (long long)hA * v_stride_p2;
  int bvcxa0 = lane_local * 2;
  bvxa0[0] = (float)v[bvixa0 + (long long)bvcxa0];
  bvxa0[1] = (float)v[bvixa0 + (long long)bvcxa0 + 1];
  long long bbixa0 =
      (long long)nA * b_stride_p0 + (long long)xml0 * b_stride_p1 + (long long)hA * b_stride_p2;
  bbxa0[0] = (float)b[bbixa0];
  float bkxb0[4];
  float bvxb0[2];
  float bbxb0[1];
  long long bkixb0 =
      (long long)nB * k_stride_p0 + (long long)xml0 * k_stride_p1 + (long long)hB * k_stride_p2;
  {
    uint2 _vld_17;
    _vld_17 = *reinterpret_cast<const uint2*>(k + bkixb0 + (long long)k_col0);
    uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17);
#pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
      (&bkxb0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) << 16);
      (&bkxb0[0 + _pair * 2])[1] =
          __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) & 0xffff0000u);
    }
  }
  long long bvixb0 =
      (long long)nB * v_stride_p0 + (long long)xml0 * v_stride_p1 + (long long)hB * v_stride_p2;
  int bvcxb0 = lane_local * 2;
  bvxb0[0] = (float)v[bvixb0 + (long long)bvcxb0];
  bvxb0[1] = (float)v[bvixb0 + (long long)bvcxb0 + 1];
  long long bbixb0 =
      (long long)nB * b_stride_p0 + (long long)xml0 * b_stride_p1 + (long long)hB * b_stride_p2;
  bbxb0[0] = (float)b[bbixb0];
  if (xm0 < steps) {
    float2 k_pair0 = make_float2(bkxa0[0], bkxa0[1]);
    float2 k_pair1 = make_float2(bkxa0[2], bkxa0[3]);
    float2 sum_k_pair = fma_f32x2_rn_ftz(k_pair0, k_pair0, make_float2(0.0f, 0.0f));
    sum_k_pair = fma_f32x2_rn_ftz(k_pair1, k_pair1, sum_k_pair);
    float sum_k = sum_k_pair.x + sum_k_pair.y;
    float _warp_reduce_18 = sum_k;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_18 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_18, offset);
    float k_norm = rsqrtf(_warp_reduce_18 + 1e-06f);
    float2 k_norm_pair = make_float2(k_norm, k_norm);
    k_pair0 = mul_f32x2(k_pair0, k_norm_pair);
    k_pair1 = mul_f32x2(k_pair1, k_norm_pair);
    sK[xm0 * 128 + k_col0] = k_pair0.x;
    sK[xm0 * 128 + k_col0 + 1] = k_pair0.y;
    sK[xm0 * 128 + k_col0 + 2] = k_pair1.x;
    sK[xm0 * 128 + k_col0 + 3] = k_pair1.y;
    int v_col = lane_local * 2;
    sV[xm0 * 64 + v_col] = bvxa0[0];
    sV[xm0 * 64 + v_col + 1] = bvxa0[1];
    if (lane_local == 0) {
      float _exp_20 = __expf(-bbxa0[0]);
      float _rcp_19 = approx_rcp(1.0f + _exp_20);
      sBeta[xm0] = _rcp_19;
    }
  }
  if (xm0 < steps) {
    float2 k_pair0_1 = make_float2(bkxb0[0], bkxb0[1]);
    float2 k_pair1_1 = make_float2(bkxb0[2], bkxb0[3]);
    float2 sum_k_pair_1 = fma_f32x2_rn_ftz(k_pair0_1, k_pair0_1, make_float2(0.0f, 0.0f));
    sum_k_pair_1 = fma_f32x2_rn_ftz(k_pair1_1, k_pair1_1, sum_k_pair_1);
    float sum_k_1 = sum_k_pair_1.x + sum_k_pair_1.y;
    float _warp_reduce_21 = sum_k_1;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_21 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_21, offset);
    float k_norm_1 = rsqrtf(_warp_reduce_21 + 1e-06f);
    float2 k_norm_pair_1 = make_float2(k_norm_1, k_norm_1);
    k_pair0_1 = mul_f32x2(k_pair0_1, k_norm_pair_1);
    k_pair1_1 = mul_f32x2(k_pair1_1, k_norm_pair_1);
    sK1[xm0 * 128 + k_col0] = k_pair0_1.x;
    sK1[xm0 * 128 + k_col0 + 1] = k_pair0_1.y;
    sK1[xm0 * 128 + k_col0 + 2] = k_pair1_1.x;
    sK1[xm0 * 128 + k_col0 + 3] = k_pair1_1.y;
    int v_col_1 = lane_local * 2;
    sV1[xm0 * 64 + v_col_1] = bvxb0[0];
    sV1[xm0 * 64 + v_col_1 + 1] = bvxb0[1];
    if (lane_local == 0) {
      float _exp_23 = __expf(-bbxb0[0]);
      float _rcp_22 = approx_rcp(1.0f + _exp_23);
      sBeta1[xm0] = _rcp_22;
    }
  }
  int xt0 = warp_local;
  int xtl0 = ((xt0) < (num_tokens - 1) ? (xt0) : (num_tokens - 1));
  float bqxa0[4];
  float baxa0[3];
  long long bqixa0 =
      (long long)nA * q_stride_p0 + (long long)xtl0 * q_stride_p1 + (long long)hA * q_stride_p2;
  {
    uint2 _vld_24;
    _vld_24 = *reinterpret_cast<const uint2*>(q + bqixa0 + (long long)k_col0);
    uint32_t* _vpairs_24 = reinterpret_cast<uint32_t*>(&_vld_24);
#pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
      (&bqxa0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) << 16);
      (&bqxa0[0 + _pair * 2])[1] =
          __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) & 0xffff0000u);
    }
  }
  long long bgixa0 =
      (long long)nA * a_stride_p0 + (long long)xtl0 * a_stride_p1 + (long long)hA * a_stride_p2;
  baxa0[0] = (float)a[bgixa0];
  baxa0[1] = dt_bias[hA];
  baxa0[2] = A_log[hA];
  float bqxb0[4];
  float baxb0[3];
  long long bqixb0 =
      (long long)nB * q_stride_p0 + (long long)xtl0 * q_stride_p1 + (long long)hB * q_stride_p2;
  {
    uint2 _vld_25;
    _vld_25 = *reinterpret_cast<const uint2*>(q + bqixb0 + (long long)k_col0);
    uint32_t* _vpairs_25 = reinterpret_cast<uint32_t*>(&_vld_25);
#pragma unroll
    for (int _pair = 0; _pair < 2; _pair++) {
      (&bqxb0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) << 16);
      (&bqxb0[0 + _pair * 2])[1] =
          __uint_as_float(static_cast<uint32_t>(_vpairs_25[_pair]) & 0xffff0000u);
    }
  }
  long long bgixb0 =
      (long long)nB * a_stride_p0 + (long long)xtl0 * a_stride_p1 + (long long)hB * a_stride_p2;
  baxb0[0] = (float)a[bgixb0];
  baxb0[1] = dt_bias[hB];
  baxb0[2] = A_log[hB];
  if (xt0 < num_tokens) {
    float2 q_raw_pair0 = make_float2(bqxa0[0], bqxa0[1]);
    float2 q_raw_pair1 = make_float2(bqxa0[2], bqxa0[3]);
    float q_norm = scale;
    float2 sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair0, q_raw_pair0, make_float2(0.0f, 0.0f));
    sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair1, q_raw_pair1, sum_q_pair);
    float sum_q = sum_q_pair.x + sum_q_pair.y;
    float _warp_reduce_26 = sum_q;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_26 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_26, offset);
    float _rsqrt_0 = rsqrtf(_warp_reduce_26 + 1e-06f);
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
    sQ[xt0 * 128 + k_col0] = q_pair0.x;
    sQ[xt0 * 128 + k_col0 + 1] = q_pair0.y;
    sQ[xt0 * 128 + k_col0 + 2] = q_pair1.x;
    sQ[xt0 * 128 + k_col0 + 3] = q_pair1.y;
    if (lane_local == 0) {
      float x = baxa0[0] + baxa0[1];
      float softplus_x = x;
      if (x <= 20.0f) {
        float _exp_28 = __expf(x);
        float _log2_27;
        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_27) : "f"(1.0f + _exp_28));
        softplus_x = _log2_27 * 0.6931472f;
      }
      float _exp_29 = __expf(baxa0[2]);
      float g_log = (-_exp_29) * softplus_x;
      float _exp_30 = __expf(g_log);
      sAlpha[xt0] = _exp_30;
      if (use_ssm_state_indices != 0) {
        sSsm[xt0] = ssm_state_indices[(long long)nA * ssm_state_indices_stride_p0 +
                                      (long long)xt0 * ssm_state_indices_stride_p1];
      }
      if (use_ssm_state_indices == 0) {
        sSsm[xt0] = -1;
      }
    }
  }
  if (xt0 < num_tokens) {
    float2 q_raw_pair0_1 = make_float2(bqxb0[0], bqxb0[1]);
    float2 q_raw_pair1_1 = make_float2(bqxb0[2], bqxb0[3]);
    float q_norm_1 = scale;
    float2 sum_q_pair_1 = fma_f32x2_rn_ftz(q_raw_pair0_1, q_raw_pair0_1, make_float2(0.0f, 0.0f));
    sum_q_pair_1 = fma_f32x2_rn_ftz(q_raw_pair1_1, q_raw_pair1_1, sum_q_pair_1);
    float sum_q_1 = sum_q_pair_1.x + sum_q_pair_1.y;
    float _warp_reduce_31 = sum_q_1;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      _warp_reduce_31 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_31, offset);
    float _rsqrt_1 = rsqrtf(_warp_reduce_31 + 1e-06f);
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
    sQ1[xt0 * 128 + k_col0] = q_pair0_1.x;
    sQ1[xt0 * 128 + k_col0 + 1] = q_pair0_1.y;
    sQ1[xt0 * 128 + k_col0 + 2] = q_pair1_1.x;
    sQ1[xt0 * 128 + k_col0 + 3] = q_pair1_1.y;
    if (lane_local == 0) {
      float x_1 = baxb0[0] + baxb0[1];
      float softplus_x_1 = x_1;
      if (x_1 <= 20.0f) {
        float _exp_33 = __expf(x_1);
        float _log2_32;
        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_32) : "f"(1.0f + _exp_33));
        softplus_x_1 = _log2_32 * 0.6931472f;
      }
      float _exp_34 = __expf(baxb0[2]);
      float g_log_1 = (-_exp_34) * softplus_x_1;
      float _exp_35 = __expf(g_log_1);
      sAlpha1[xt0] = _exp_35;
      if (use_ssm_state_indices != 0) {
        sSsm1[xt0] = ssm_state_indices[(long long)nB * ssm_state_indices_stride_p0 +
                                       (long long)xt0 * ssm_state_indices_stride_p1];
      }
      if (use_ssm_state_indices == 0) {
        sSsm1[xt0] = -1;
      }
    }
  }
  __syncthreads();
  if (readA_valid != 0) {
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
      float alphas = sAlpha[t];
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
                   : "r"(sV_addr + (unsigned int)(ms0 * 256) + group_byte));
      float betas0 = sBeta[ms0];
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
      float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 1);
      dots0_0 = dots0_0 + _shfl_xor_sync_36;
      float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 2);
      dots0_0 = dots0_0 + _shfl_xor_sync_37;
      float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, dots0_0, 4);
      dots0_0 = dots0_0 + _shfl_xor_sync_38;
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
      accs0_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accs0_1_0);
      accs0_1_0 = fma_f32x2_rn_ftz(S2[9], k2[1], accs0_1_0);
      accs0_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accs0_1_0);
      accs0_1_0 = fma_f32x2_rn_ftz(S2[11], k2[3], accs0_1_0);
      accs0_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accs0_1_0);
      accs0_1_0 = fma_f32x2_rn_ftz(S2[13], k2[5], accs0_1_0);
      accs0_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accs0_1_0);
      accs0_1_0 = fma_f32x2_rn_ftz(S2[15], k2[7], accs0_1_0);
      float dots0_1 = accs0_1_0.x + accs0_1_0.y;
      float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 1);
      dots0_1 = dots0_1 + _shfl_xor_sync_39;
      float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 2);
      dots0_1 = dots0_1 + _shfl_xor_sync_40;
      float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, dots0_1, 4);
      dots0_1 = dots0_1 + _shfl_xor_sync_41;
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
      float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, dots0_2, 1);
      dots0_2 = dots0_2 + _shfl_xor_sync_42;
      float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, dots0_2, 2);
      dots0_2 = dots0_2 + _shfl_xor_sync_43;
      float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, dots0_2, 4);
      dots0_2 = dots0_2 + _shfl_xor_sync_44;
      float vns0_2 = (v_regs[2] - dots0_2) * betas0;
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
      float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, dots0_3, 1);
      dots0_3 = dots0_3 + _shfl_xor_sync_45;
      float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, dots0_3, 2);
      dots0_3 = dots0_3 + _shfl_xor_sync_46;
      float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, dots0_3, 4);
      dots0_3 = dots0_3 + _shfl_xor_sync_47;
      float vns0_3 = (v_regs[3] - dots0_3) * betas0;
      float2 vn2s0_3 = make_float2(vns0_3, vns0_3);
      S2[24] = fma_f32x2_rn_ftz(k2[0], vn2s0_3, S2[24]);
      S2[25] = fma_f32x2_rn_ftz(k2[1], vn2s0_3, S2[25]);
      S2[26] = fma_f32x2_rn_ftz(k2[2], vn2s0_3, S2[26]);
      S2[27] = fma_f32x2_rn_ftz(k2[3], vn2s0_3, S2[27]);
      S2[28] = fma_f32x2_rn_ftz(k2[4], vn2s0_3, S2[28]);
      S2[29] = fma_f32x2_rn_ftz(k2[5], vn2s0_3, S2[29]);
      S2[30] = fma_f32x2_rn_ftz(k2[6], vn2s0_3, S2[30]);
      S2[31] = fma_f32x2_rn_ftz(k2[7], vn2s0_3, S2[31]);
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
                   : "r"(sV_addr + (unsigned int)(ms1 * 256) + group_byte));
      float betas1 = sBeta[ms1];
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
      float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 1);
      dots1_0 = dots1_0 + _shfl_xor_sync_48;
      float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 2);
      dots1_0 = dots1_0 + _shfl_xor_sync_49;
      float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, dots1_0, 4);
      dots1_0 = dots1_0 + _shfl_xor_sync_50;
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
      accs1_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accs1_1_0);
      accs1_1_0 = fma_f32x2_rn_ftz(S2[9], k2[1], accs1_1_0);
      accs1_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accs1_1_0);
      accs1_1_0 = fma_f32x2_rn_ftz(S2[11], k2[3], accs1_1_0);
      accs1_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accs1_1_0);
      accs1_1_0 = fma_f32x2_rn_ftz(S2[13], k2[5], accs1_1_0);
      accs1_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accs1_1_0);
      accs1_1_0 = fma_f32x2_rn_ftz(S2[15], k2[7], accs1_1_0);
      float dots1_1 = accs1_1_0.x + accs1_1_0.y;
      float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 1);
      dots1_1 = dots1_1 + _shfl_xor_sync_51;
      float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 2);
      dots1_1 = dots1_1 + _shfl_xor_sync_52;
      float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, dots1_1, 4);
      dots1_1 = dots1_1 + _shfl_xor_sync_53;
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
      float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, dots1_2, 1);
      dots1_2 = dots1_2 + _shfl_xor_sync_54;
      float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, dots1_2, 2);
      dots1_2 = dots1_2 + _shfl_xor_sync_55;
      float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, dots1_2, 4);
      dots1_2 = dots1_2 + _shfl_xor_sync_56;
      float vns1_2 = (v_regs[2] - dots1_2) * betas1;
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
      float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, dots1_3, 1);
      dots1_3 = dots1_3 + _shfl_xor_sync_57;
      float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, dots1_3, 2);
      dots1_3 = dots1_3 + _shfl_xor_sync_58;
      float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, dots1_3, 4);
      dots1_3 = dots1_3 + _shfl_xor_sync_59;
      float vns1_3 = (v_regs[3] - dots1_3) * betas1;
      float2 vn2s1_3 = make_float2(vns1_3, vns1_3);
      S2[24] = fma_f32x2_rn_ftz(k2[0], vn2s1_3, S2[24]);
      S2[25] = fma_f32x2_rn_ftz(k2[1], vn2s1_3, S2[25]);
      S2[26] = fma_f32x2_rn_ftz(k2[2], vn2s1_3, S2[26]);
      S2[27] = fma_f32x2_rn_ftz(k2[3], vn2s1_3, S2[27]);
      S2[28] = fma_f32x2_rn_ftz(k2[4], vn2s1_3, S2[28]);
      S2[29] = fma_f32x2_rn_ftz(k2[5], vn2s1_3, S2[29]);
      S2[30] = fma_f32x2_rn_ftz(k2[6], vn2s1_3, S2[30]);
      S2[31] = fma_f32x2_rn_ftz(k2[7], vn2s1_3, S2[31]);
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
                   : "r"(sV_addr + (unsigned int)(ms2 * 256) + group_byte));
      float betas2 = sBeta[ms2];
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
      float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 1);
      dots2_0 = dots2_0 + _shfl_xor_sync_60;
      float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 2);
      dots2_0 = dots2_0 + _shfl_xor_sync_61;
      float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, dots2_0, 4);
      dots2_0 = dots2_0 + _shfl_xor_sync_62;
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
      accs2_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accs2_1_0);
      accs2_1_0 = fma_f32x2_rn_ftz(S2[9], k2[1], accs2_1_0);
      accs2_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accs2_1_0);
      accs2_1_0 = fma_f32x2_rn_ftz(S2[11], k2[3], accs2_1_0);
      accs2_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accs2_1_0);
      accs2_1_0 = fma_f32x2_rn_ftz(S2[13], k2[5], accs2_1_0);
      accs2_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accs2_1_0);
      accs2_1_0 = fma_f32x2_rn_ftz(S2[15], k2[7], accs2_1_0);
      float dots2_1 = accs2_1_0.x + accs2_1_0.y;
      float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 1);
      dots2_1 = dots2_1 + _shfl_xor_sync_63;
      float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 2);
      dots2_1 = dots2_1 + _shfl_xor_sync_64;
      float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, dots2_1, 4);
      dots2_1 = dots2_1 + _shfl_xor_sync_65;
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
      float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, dots2_2, 1);
      dots2_2 = dots2_2 + _shfl_xor_sync_66;
      float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, dots2_2, 2);
      dots2_2 = dots2_2 + _shfl_xor_sync_67;
      float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, dots2_2, 4);
      dots2_2 = dots2_2 + _shfl_xor_sync_68;
      float vns2_2 = (v_regs[2] - dots2_2) * betas2;
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
      float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, dots2_3, 1);
      dots2_3 = dots2_3 + _shfl_xor_sync_69;
      float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, dots2_3, 2);
      dots2_3 = dots2_3 + _shfl_xor_sync_70;
      float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, dots2_3, 4);
      dots2_3 = dots2_3 + _shfl_xor_sync_71;
      float vns2_3 = (v_regs[3] - dots2_3) * betas2;
      float2 vn2s2_3 = make_float2(vns2_3, vns2_3);
      S2[24] = fma_f32x2_rn_ftz(k2[0], vn2s2_3, S2[24]);
      S2[25] = fma_f32x2_rn_ftz(k2[1], vn2s2_3, S2[25]);
      S2[26] = fma_f32x2_rn_ftz(k2[2], vn2s2_3, S2[26]);
      S2[27] = fma_f32x2_rn_ftz(k2[3], vn2s2_3, S2[27]);
      S2[28] = fma_f32x2_rn_ftz(k2[4], vn2s2_3, S2[28]);
      S2[29] = fma_f32x2_rn_ftz(k2[5], vn2s2_3, S2[29]);
      S2[30] = fma_f32x2_rn_ftz(k2[6], vn2s2_3, S2[30]);
      S2[31] = fma_f32x2_rn_ftz(k2[7], vn2s2_3, S2[31]);
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
      float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 1);
      dotsq_0 = dotsq_0 + _shfl_xor_sync_72;
      float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 2);
      dotsq_0 = dotsq_0 + _shfl_xor_sync_73;
      float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, dotsq_0, 4);
      dotsq_0 = dotsq_0 + _shfl_xor_sync_74;
      o4[0] = dotsq_0;
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
      float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 1);
      dotsq_1 = dotsq_1 + _shfl_xor_sync_75;
      float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 2);
      dotsq_1 = dotsq_1 + _shfl_xor_sync_76;
      float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, dotsq_1, 4);
      dotsq_1 = dotsq_1 + _shfl_xor_sync_77;
      o4[1] = dotsq_1;
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
      float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, dotsq_2, 1);
      dotsq_2 = dotsq_2 + _shfl_xor_sync_78;
      float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, dotsq_2, 2);
      dotsq_2 = dotsq_2 + _shfl_xor_sync_79;
      float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, dotsq_2, 4);
      dotsq_2 = dotsq_2 + _shfl_xor_sync_80;
      o4[2] = dotsq_2;
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
      float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, dotsq_3, 1);
      dotsq_3 = dotsq_3 + _shfl_xor_sync_81;
      float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, dotsq_3, 2);
      dotsq_3 = dotsq_3 + _shfl_xor_sync_82;
      float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, dotsq_3, 4);
      dotsq_3 = dotsq_3 + _shfl_xor_sync_83;
      o4[3] = dotsq_3;
      if (u == 0) {
        long long out_index = (long long)nA * out_stride_p0 + (long long)t * out_stride_p1 +
                              (long long)hA * out_stride_p2 + (long long)row0;
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
      if (use_ssm_state_indices != 0) {
        int ssm_slot = sSsm[t];
        if (ssm_slot >= 0 && state_pool_size > (long long)ssm_slot) {
          long long snap_base = (long long)ssm_slot * state_stride_p0 + rowsA_off;
          r_h[0] = S2[0].x;
          r_h[1] = S2[0].y;
          r_h[2] = S2[1].x;
          r_h[3] = S2[1].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base) = _v4;
          }
          r_h[0] = S2[2].x;
          r_h[1] = S2[2].y;
          r_h[2] = S2[3].x;
          r_h[3] = S2[3].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base + 32) = _v4;
          }
          r_h[0] = S2[4].x;
          r_h[1] = S2[4].y;
          r_h[2] = S2[5].x;
          r_h[3] = S2[5].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base + 64) = _v4;
          }
          r_h[0] = S2[6].x;
          r_h[1] = S2[6].y;
          r_h[2] = S2[7].x;
          r_h[3] = S2[7].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base + 96) = _v4;
          }
          r_h[0] = S2[8].x;
          r_h[1] = S2[8].y;
          r_h[2] = S2[9].x;
          r_h[3] = S2[9].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base + state_stride_p2) = _v4;
          }
          r_h[0] = S2[10].x;
          r_h[1] = S2[10].y;
          r_h[2] = S2[11].x;
          r_h[3] = S2[11].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base + state_stride_p2 + 32) = _v4;
          }
          r_h[0] = S2[12].x;
          r_h[1] = S2[12].y;
          r_h[2] = S2[13].x;
          r_h[3] = S2[13].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base + state_stride_p2 + 64) = _v4;
          }
          r_h[0] = S2[14].x;
          r_h[1] = S2[14].y;
          r_h[2] = S2[15].x;
          r_h[3] = S2[15].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base + state_stride_p2 + 96) = _v4;
          }
          r_h[0] = S2[16].x;
          r_h[1] = S2[16].y;
          r_h[2] = S2[17].x;
          r_h[3] = S2[17].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base + 2 * state_stride_p2) = _v4;
          }
          r_h[0] = S2[18].x;
          r_h[1] = S2[18].y;
          r_h[2] = S2[19].x;
          r_h[3] = S2[19].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base + 2 * state_stride_p2 + 32) = _v4;
          }
          r_h[0] = S2[20].x;
          r_h[1] = S2[20].y;
          r_h[2] = S2[21].x;
          r_h[3] = S2[21].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base + 2 * state_stride_p2 + 64) = _v4;
          }
          r_h[0] = S2[22].x;
          r_h[1] = S2[22].y;
          r_h[2] = S2[23].x;
          r_h[3] = S2[23].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base + 2 * state_stride_p2 + 96) = _v4;
          }
          r_h[0] = S2[24].x;
          r_h[1] = S2[24].y;
          r_h[2] = S2[25].x;
          r_h[3] = S2[25].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base + 3 * state_stride_p2) = _v4;
          }
          r_h[0] = S2[26].x;
          r_h[1] = S2[26].y;
          r_h[2] = S2[27].x;
          r_h[3] = S2[27].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base + 3 * state_stride_p2 + 32) = _v4;
          }
          r_h[0] = S2[28].x;
          r_h[1] = S2[28].y;
          r_h[2] = S2[29].x;
          r_h[3] = S2[29].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base + 3 * state_stride_p2 + 64) = _v4;
          }
          r_h[0] = S2[30].x;
          r_h[1] = S2[30].y;
          r_h[2] = S2[31].x;
          r_h[3] = S2[31].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base + 3 * state_stride_p2 + 96) = _v4;
          }
        }
      }
    }
    if (writeA_valid != 0) {
      if (disable_state_update == 0) {
        if (use_ssm_state_indices == 0) {
          r_h[0] = S2[0].x;
          r_h[1] = S2[0].y;
          r_h[2] = S2[1].x;
          r_h[3] = S2[1].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeA_base) = _v4;
          }
          r_h[0] = S2[2].x;
          r_h[1] = S2[2].y;
          r_h[2] = S2[3].x;
          r_h[3] = S2[3].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeA_base + 32) = _v4;
          }
          r_h[0] = S2[4].x;
          r_h[1] = S2[4].y;
          r_h[2] = S2[5].x;
          r_h[3] = S2[5].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeA_base + 64) = _v4;
          }
          r_h[0] = S2[6].x;
          r_h[1] = S2[6].y;
          r_h[2] = S2[7].x;
          r_h[3] = S2[7].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeA_base + 96) = _v4;
          }
          r_h[0] = S2[8].x;
          r_h[1] = S2[8].y;
          r_h[2] = S2[9].x;
          r_h[3] = S2[9].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeA_base + state_stride_p2) = _v4;
          }
          r_h[0] = S2[10].x;
          r_h[1] = S2[10].y;
          r_h[2] = S2[11].x;
          r_h[3] = S2[11].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeA_base + state_stride_p2 + 32) = _v4;
          }
          r_h[0] = S2[12].x;
          r_h[1] = S2[12].y;
          r_h[2] = S2[13].x;
          r_h[3] = S2[13].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeA_base + state_stride_p2 + 64) = _v4;
          }
          r_h[0] = S2[14].x;
          r_h[1] = S2[14].y;
          r_h[2] = S2[15].x;
          r_h[3] = S2[15].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeA_base + state_stride_p2 + 96) = _v4;
          }
          r_h[0] = S2[16].x;
          r_h[1] = S2[16].y;
          r_h[2] = S2[17].x;
          r_h[3] = S2[17].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeA_base + 2 * state_stride_p2) = _v4;
          }
          r_h[0] = S2[18].x;
          r_h[1] = S2[18].y;
          r_h[2] = S2[19].x;
          r_h[3] = S2[19].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeA_base + 2 * state_stride_p2 + 32) = _v4;
          }
          r_h[0] = S2[20].x;
          r_h[1] = S2[20].y;
          r_h[2] = S2[21].x;
          r_h[3] = S2[21].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeA_base + 2 * state_stride_p2 + 64) = _v4;
          }
          r_h[0] = S2[22].x;
          r_h[1] = S2[22].y;
          r_h[2] = S2[23].x;
          r_h[3] = S2[23].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeA_base + 2 * state_stride_p2 + 96) = _v4;
          }
          r_h[0] = S2[24].x;
          r_h[1] = S2[24].y;
          r_h[2] = S2[25].x;
          r_h[3] = S2[25].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeA_base + 3 * state_stride_p2) = _v4;
          }
          r_h[0] = S2[26].x;
          r_h[1] = S2[26].y;
          r_h[2] = S2[27].x;
          r_h[3] = S2[27].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeA_base + 3 * state_stride_p2 + 32) = _v4;
          }
          r_h[0] = S2[28].x;
          r_h[1] = S2[28].y;
          r_h[2] = S2[29].x;
          r_h[3] = S2[29].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeA_base + 3 * state_stride_p2 + 64) = _v4;
          }
          r_h[0] = S2[30].x;
          r_h[1] = S2[30].y;
          r_h[2] = S2[31].x;
          r_h[3] = S2[31].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeA_base + 3 * state_stride_p2 + 96) = _v4;
          }
        }
      }
    }
  }
  if (readB_valid != 0) {
    asm volatile("cp.async.wait_group 0;");
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&S[0])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(0) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(0) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(0) + 3]))
                 : "r"(sStage_addr + (unsigned int)(tid_0 * 16)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&S[4])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(4) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(4) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(4) + 3]))
                 : "r"(sStage_addr + (unsigned int)((128 + tid_0) * 16)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&S[8])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(8) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(8) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(8) + 3]))
                 : "r"(sStage_addr + (unsigned int)((256 + tid_0) * 16)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&S[12])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(12) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(12) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(12) + 3]))
                 : "r"(sStage_addr + (unsigned int)((384 + tid_0) * 16)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&S[16])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(16) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(16) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(16) + 3]))
                 : "r"(sStage_addr + (unsigned int)((512 + tid_0) * 16)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&S[20])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(20) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(20) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(20) + 3]))
                 : "r"(sStage_addr + (unsigned int)((640 + tid_0) * 16)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&S[24])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(24) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(24) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(24) + 3]))
                 : "r"(sStage_addr + (unsigned int)((768 + tid_0) * 16)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&S[28])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(28) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(28) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(28) + 3]))
                 : "r"(sStage_addr + (unsigned int)((896 + tid_0) * 16)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&S[32])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(32) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(32) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(32) + 3]))
                 : "r"(sStage_addr + (unsigned int)((1024 + tid_0) * 16)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&S[36])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(36) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(36) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(36) + 3]))
                 : "r"(sStage_addr + (unsigned int)((1152 + tid_0) * 16)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&S[40])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(40) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(40) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(40) + 3]))
                 : "r"(sStage_addr + (unsigned int)((1280 + tid_0) * 16)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&S[44])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(44) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(44) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(44) + 3]))
                 : "r"(sStage_addr + (unsigned int)((1408 + tid_0) * 16)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&S[48])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(48) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(48) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(48) + 3]))
                 : "r"(sStage_addr + (unsigned int)((1536 + tid_0) * 16)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&S[52])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(52) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(52) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(52) + 3]))
                 : "r"(sStage_addr + (unsigned int)((1664 + tid_0) * 16)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&S[56])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(56) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(56) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(56) + 3]))
                 : "r"(sStage_addr + (unsigned int)((1792 + tid_0) * 16)));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*reinterpret_cast<uint32_t*>(&S[60])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(60) + 1])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(60) + 2])),
                   "=r"(*reinterpret_cast<uint32_t*>(&S[(60) + 3]))
                 : "r"(sStage_addr + (unsigned int)((1920 + tid_0) * 16)));
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
    for (int t_1 = 0; t_1 < 1; t_1++) {
      float alphaxb = sAlpha1[t_1];
      float2 alpha2xb = make_float2(alphaxb, alphaxb);
      S2[0] = mul_f32x2(S2[0], alpha2xb);
      S2[1] = mul_f32x2(S2[1], alpha2xb);
      S2[2] = mul_f32x2(S2[2], alpha2xb);
      S2[3] = mul_f32x2(S2[3], alpha2xb);
      S2[4] = mul_f32x2(S2[4], alpha2xb);
      S2[5] = mul_f32x2(S2[5], alpha2xb);
      S2[6] = mul_f32x2(S2[6], alpha2xb);
      S2[7] = mul_f32x2(S2[7], alpha2xb);
      S2[8] = mul_f32x2(S2[8], alpha2xb);
      S2[9] = mul_f32x2(S2[9], alpha2xb);
      S2[10] = mul_f32x2(S2[10], alpha2xb);
      S2[11] = mul_f32x2(S2[11], alpha2xb);
      S2[12] = mul_f32x2(S2[12], alpha2xb);
      S2[13] = mul_f32x2(S2[13], alpha2xb);
      S2[14] = mul_f32x2(S2[14], alpha2xb);
      S2[15] = mul_f32x2(S2[15], alpha2xb);
      S2[16] = mul_f32x2(S2[16], alpha2xb);
      S2[17] = mul_f32x2(S2[17], alpha2xb);
      S2[18] = mul_f32x2(S2[18], alpha2xb);
      S2[19] = mul_f32x2(S2[19], alpha2xb);
      S2[20] = mul_f32x2(S2[20], alpha2xb);
      S2[21] = mul_f32x2(S2[21], alpha2xb);
      S2[22] = mul_f32x2(S2[22], alpha2xb);
      S2[23] = mul_f32x2(S2[23], alpha2xb);
      S2[24] = mul_f32x2(S2[24], alpha2xb);
      S2[25] = mul_f32x2(S2[25], alpha2xb);
      S2[26] = mul_f32x2(S2[26], alpha2xb);
      S2[27] = mul_f32x2(S2[27], alpha2xb);
      S2[28] = mul_f32x2(S2[28], alpha2xb);
      S2[29] = mul_f32x2(S2[29], alpha2xb);
      S2[30] = mul_f32x2(S2[30], alpha2xb);
      S2[31] = mul_f32x2(S2[31], alpha2xb);
      int mxb0 = t_1 * 3;
      unsigned int kbasexb0 = sK1_addr + (unsigned int)(mxb0 * 512) + lane_byte;
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                   : "r"(kbasexb0));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                   : "r"(kbasexb0 + 128));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                   : "r"(kbasexb0 + 256));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                   : "r"(kbasexb0 + 384));
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
                   : "r"(sV1_addr + (unsigned int)(mxb0 * 256) + group_byte));
      float betaxb0 = sBeta1[mxb0];
      float2 accxb0_0_0 = make_float2(0.0f, 0.0f);
      accxb0_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accxb0_0_0);
      accxb0_0_0 = fma_f32x2_rn_ftz(S2[1], k2[1], accxb0_0_0);
      accxb0_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accxb0_0_0);
      accxb0_0_0 = fma_f32x2_rn_ftz(S2[3], k2[3], accxb0_0_0);
      accxb0_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accxb0_0_0);
      accxb0_0_0 = fma_f32x2_rn_ftz(S2[5], k2[5], accxb0_0_0);
      accxb0_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accxb0_0_0);
      accxb0_0_0 = fma_f32x2_rn_ftz(S2[7], k2[7], accxb0_0_0);
      float dotxb0_0 = accxb0_0_0.x + accxb0_0_0.y;
      float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, dotxb0_0, 1);
      dotxb0_0 = dotxb0_0 + _shfl_xor_sync_84;
      float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, dotxb0_0, 2);
      dotxb0_0 = dotxb0_0 + _shfl_xor_sync_85;
      float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, dotxb0_0, 4);
      dotxb0_0 = dotxb0_0 + _shfl_xor_sync_86;
      float vnxb0_0 = (v_regs[0] - dotxb0_0) * betaxb0;
      float2 vn2xb0_0 = make_float2(vnxb0_0, vnxb0_0);
      S2[0] = fma_f32x2_rn_ftz(k2[0], vn2xb0_0, S2[0]);
      S2[1] = fma_f32x2_rn_ftz(k2[1], vn2xb0_0, S2[1]);
      S2[2] = fma_f32x2_rn_ftz(k2[2], vn2xb0_0, S2[2]);
      S2[3] = fma_f32x2_rn_ftz(k2[3], vn2xb0_0, S2[3]);
      S2[4] = fma_f32x2_rn_ftz(k2[4], vn2xb0_0, S2[4]);
      S2[5] = fma_f32x2_rn_ftz(k2[5], vn2xb0_0, S2[5]);
      S2[6] = fma_f32x2_rn_ftz(k2[6], vn2xb0_0, S2[6]);
      S2[7] = fma_f32x2_rn_ftz(k2[7], vn2xb0_0, S2[7]);
      float2 accxb0_1_0 = make_float2(0.0f, 0.0f);
      accxb0_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accxb0_1_0);
      accxb0_1_0 = fma_f32x2_rn_ftz(S2[9], k2[1], accxb0_1_0);
      accxb0_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accxb0_1_0);
      accxb0_1_0 = fma_f32x2_rn_ftz(S2[11], k2[3], accxb0_1_0);
      accxb0_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accxb0_1_0);
      accxb0_1_0 = fma_f32x2_rn_ftz(S2[13], k2[5], accxb0_1_0);
      accxb0_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accxb0_1_0);
      accxb0_1_0 = fma_f32x2_rn_ftz(S2[15], k2[7], accxb0_1_0);
      float dotxb0_1 = accxb0_1_0.x + accxb0_1_0.y;
      float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, dotxb0_1, 1);
      dotxb0_1 = dotxb0_1 + _shfl_xor_sync_87;
      float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, dotxb0_1, 2);
      dotxb0_1 = dotxb0_1 + _shfl_xor_sync_88;
      float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, dotxb0_1, 4);
      dotxb0_1 = dotxb0_1 + _shfl_xor_sync_89;
      float vnxb0_1 = (v_regs[1] - dotxb0_1) * betaxb0;
      float2 vn2xb0_1 = make_float2(vnxb0_1, vnxb0_1);
      S2[8] = fma_f32x2_rn_ftz(k2[0], vn2xb0_1, S2[8]);
      S2[9] = fma_f32x2_rn_ftz(k2[1], vn2xb0_1, S2[9]);
      S2[10] = fma_f32x2_rn_ftz(k2[2], vn2xb0_1, S2[10]);
      S2[11] = fma_f32x2_rn_ftz(k2[3], vn2xb0_1, S2[11]);
      S2[12] = fma_f32x2_rn_ftz(k2[4], vn2xb0_1, S2[12]);
      S2[13] = fma_f32x2_rn_ftz(k2[5], vn2xb0_1, S2[13]);
      S2[14] = fma_f32x2_rn_ftz(k2[6], vn2xb0_1, S2[14]);
      S2[15] = fma_f32x2_rn_ftz(k2[7], vn2xb0_1, S2[15]);
      float2 accxb0_2_0 = make_float2(0.0f, 0.0f);
      accxb0_2_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accxb0_2_0);
      accxb0_2_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accxb0_2_0);
      accxb0_2_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accxb0_2_0);
      accxb0_2_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accxb0_2_0);
      accxb0_2_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accxb0_2_0);
      accxb0_2_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accxb0_2_0);
      accxb0_2_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accxb0_2_0);
      accxb0_2_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accxb0_2_0);
      float dotxb0_2 = accxb0_2_0.x + accxb0_2_0.y;
      float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, dotxb0_2, 1);
      dotxb0_2 = dotxb0_2 + _shfl_xor_sync_90;
      float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, dotxb0_2, 2);
      dotxb0_2 = dotxb0_2 + _shfl_xor_sync_91;
      float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, dotxb0_2, 4);
      dotxb0_2 = dotxb0_2 + _shfl_xor_sync_92;
      float vnxb0_2 = (v_regs[2] - dotxb0_2) * betaxb0;
      float2 vn2xb0_2 = make_float2(vnxb0_2, vnxb0_2);
      S2[16] = fma_f32x2_rn_ftz(k2[0], vn2xb0_2, S2[16]);
      S2[17] = fma_f32x2_rn_ftz(k2[1], vn2xb0_2, S2[17]);
      S2[18] = fma_f32x2_rn_ftz(k2[2], vn2xb0_2, S2[18]);
      S2[19] = fma_f32x2_rn_ftz(k2[3], vn2xb0_2, S2[19]);
      S2[20] = fma_f32x2_rn_ftz(k2[4], vn2xb0_2, S2[20]);
      S2[21] = fma_f32x2_rn_ftz(k2[5], vn2xb0_2, S2[21]);
      S2[22] = fma_f32x2_rn_ftz(k2[6], vn2xb0_2, S2[22]);
      S2[23] = fma_f32x2_rn_ftz(k2[7], vn2xb0_2, S2[23]);
      float2 accxb0_3_0 = make_float2(0.0f, 0.0f);
      accxb0_3_0 = fma_f32x2_rn_ftz(S2[24], k2[0], accxb0_3_0);
      accxb0_3_0 = fma_f32x2_rn_ftz(S2[25], k2[1], accxb0_3_0);
      accxb0_3_0 = fma_f32x2_rn_ftz(S2[26], k2[2], accxb0_3_0);
      accxb0_3_0 = fma_f32x2_rn_ftz(S2[27], k2[3], accxb0_3_0);
      accxb0_3_0 = fma_f32x2_rn_ftz(S2[28], k2[4], accxb0_3_0);
      accxb0_3_0 = fma_f32x2_rn_ftz(S2[29], k2[5], accxb0_3_0);
      accxb0_3_0 = fma_f32x2_rn_ftz(S2[30], k2[6], accxb0_3_0);
      accxb0_3_0 = fma_f32x2_rn_ftz(S2[31], k2[7], accxb0_3_0);
      float dotxb0_3 = accxb0_3_0.x + accxb0_3_0.y;
      float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, dotxb0_3, 1);
      dotxb0_3 = dotxb0_3 + _shfl_xor_sync_93;
      float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, dotxb0_3, 2);
      dotxb0_3 = dotxb0_3 + _shfl_xor_sync_94;
      float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, dotxb0_3, 4);
      dotxb0_3 = dotxb0_3 + _shfl_xor_sync_95;
      float vnxb0_3 = (v_regs[3] - dotxb0_3) * betaxb0;
      float2 vn2xb0_3 = make_float2(vnxb0_3, vnxb0_3);
      S2[24] = fma_f32x2_rn_ftz(k2[0], vn2xb0_3, S2[24]);
      S2[25] = fma_f32x2_rn_ftz(k2[1], vn2xb0_3, S2[25]);
      S2[26] = fma_f32x2_rn_ftz(k2[2], vn2xb0_3, S2[26]);
      S2[27] = fma_f32x2_rn_ftz(k2[3], vn2xb0_3, S2[27]);
      S2[28] = fma_f32x2_rn_ftz(k2[4], vn2xb0_3, S2[28]);
      S2[29] = fma_f32x2_rn_ftz(k2[5], vn2xb0_3, S2[29]);
      S2[30] = fma_f32x2_rn_ftz(k2[6], vn2xb0_3, S2[30]);
      S2[31] = fma_f32x2_rn_ftz(k2[7], vn2xb0_3, S2[31]);
      int mxb1 = t_1 * 3 + 1;
      unsigned int kbasexb1 = sK1_addr + (unsigned int)(mxb1 * 512) + lane_byte;
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                   : "r"(kbasexb1));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                   : "r"(kbasexb1 + 128));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                   : "r"(kbasexb1 + 256));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                   : "r"(kbasexb1 + 384));
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
                   : "r"(sV1_addr + (unsigned int)(mxb1 * 256) + group_byte));
      float betaxb1 = sBeta1[mxb1];
      float2 accxb1_0_0 = make_float2(0.0f, 0.0f);
      accxb1_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accxb1_0_0);
      accxb1_0_0 = fma_f32x2_rn_ftz(S2[1], k2[1], accxb1_0_0);
      accxb1_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accxb1_0_0);
      accxb1_0_0 = fma_f32x2_rn_ftz(S2[3], k2[3], accxb1_0_0);
      accxb1_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accxb1_0_0);
      accxb1_0_0 = fma_f32x2_rn_ftz(S2[5], k2[5], accxb1_0_0);
      accxb1_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accxb1_0_0);
      accxb1_0_0 = fma_f32x2_rn_ftz(S2[7], k2[7], accxb1_0_0);
      float dotxb1_0 = accxb1_0_0.x + accxb1_0_0.y;
      float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, dotxb1_0, 1);
      dotxb1_0 = dotxb1_0 + _shfl_xor_sync_96;
      float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, dotxb1_0, 2);
      dotxb1_0 = dotxb1_0 + _shfl_xor_sync_97;
      float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, dotxb1_0, 4);
      dotxb1_0 = dotxb1_0 + _shfl_xor_sync_98;
      float vnxb1_0 = (v_regs[0] - dotxb1_0) * betaxb1;
      float2 vn2xb1_0 = make_float2(vnxb1_0, vnxb1_0);
      S2[0] = fma_f32x2_rn_ftz(k2[0], vn2xb1_0, S2[0]);
      S2[1] = fma_f32x2_rn_ftz(k2[1], vn2xb1_0, S2[1]);
      S2[2] = fma_f32x2_rn_ftz(k2[2], vn2xb1_0, S2[2]);
      S2[3] = fma_f32x2_rn_ftz(k2[3], vn2xb1_0, S2[3]);
      S2[4] = fma_f32x2_rn_ftz(k2[4], vn2xb1_0, S2[4]);
      S2[5] = fma_f32x2_rn_ftz(k2[5], vn2xb1_0, S2[5]);
      S2[6] = fma_f32x2_rn_ftz(k2[6], vn2xb1_0, S2[6]);
      S2[7] = fma_f32x2_rn_ftz(k2[7], vn2xb1_0, S2[7]);
      float2 accxb1_1_0 = make_float2(0.0f, 0.0f);
      accxb1_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accxb1_1_0);
      accxb1_1_0 = fma_f32x2_rn_ftz(S2[9], k2[1], accxb1_1_0);
      accxb1_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accxb1_1_0);
      accxb1_1_0 = fma_f32x2_rn_ftz(S2[11], k2[3], accxb1_1_0);
      accxb1_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accxb1_1_0);
      accxb1_1_0 = fma_f32x2_rn_ftz(S2[13], k2[5], accxb1_1_0);
      accxb1_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accxb1_1_0);
      accxb1_1_0 = fma_f32x2_rn_ftz(S2[15], k2[7], accxb1_1_0);
      float dotxb1_1 = accxb1_1_0.x + accxb1_1_0.y;
      float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, dotxb1_1, 1);
      dotxb1_1 = dotxb1_1 + _shfl_xor_sync_99;
      float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, dotxb1_1, 2);
      dotxb1_1 = dotxb1_1 + _shfl_xor_sync_100;
      float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, dotxb1_1, 4);
      dotxb1_1 = dotxb1_1 + _shfl_xor_sync_101;
      float vnxb1_1 = (v_regs[1] - dotxb1_1) * betaxb1;
      float2 vn2xb1_1 = make_float2(vnxb1_1, vnxb1_1);
      S2[8] = fma_f32x2_rn_ftz(k2[0], vn2xb1_1, S2[8]);
      S2[9] = fma_f32x2_rn_ftz(k2[1], vn2xb1_1, S2[9]);
      S2[10] = fma_f32x2_rn_ftz(k2[2], vn2xb1_1, S2[10]);
      S2[11] = fma_f32x2_rn_ftz(k2[3], vn2xb1_1, S2[11]);
      S2[12] = fma_f32x2_rn_ftz(k2[4], vn2xb1_1, S2[12]);
      S2[13] = fma_f32x2_rn_ftz(k2[5], vn2xb1_1, S2[13]);
      S2[14] = fma_f32x2_rn_ftz(k2[6], vn2xb1_1, S2[14]);
      S2[15] = fma_f32x2_rn_ftz(k2[7], vn2xb1_1, S2[15]);
      float2 accxb1_2_0 = make_float2(0.0f, 0.0f);
      accxb1_2_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accxb1_2_0);
      accxb1_2_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accxb1_2_0);
      accxb1_2_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accxb1_2_0);
      accxb1_2_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accxb1_2_0);
      accxb1_2_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accxb1_2_0);
      accxb1_2_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accxb1_2_0);
      accxb1_2_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accxb1_2_0);
      accxb1_2_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accxb1_2_0);
      float dotxb1_2 = accxb1_2_0.x + accxb1_2_0.y;
      float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, dotxb1_2, 1);
      dotxb1_2 = dotxb1_2 + _shfl_xor_sync_102;
      float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, dotxb1_2, 2);
      dotxb1_2 = dotxb1_2 + _shfl_xor_sync_103;
      float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, dotxb1_2, 4);
      dotxb1_2 = dotxb1_2 + _shfl_xor_sync_104;
      float vnxb1_2 = (v_regs[2] - dotxb1_2) * betaxb1;
      float2 vn2xb1_2 = make_float2(vnxb1_2, vnxb1_2);
      S2[16] = fma_f32x2_rn_ftz(k2[0], vn2xb1_2, S2[16]);
      S2[17] = fma_f32x2_rn_ftz(k2[1], vn2xb1_2, S2[17]);
      S2[18] = fma_f32x2_rn_ftz(k2[2], vn2xb1_2, S2[18]);
      S2[19] = fma_f32x2_rn_ftz(k2[3], vn2xb1_2, S2[19]);
      S2[20] = fma_f32x2_rn_ftz(k2[4], vn2xb1_2, S2[20]);
      S2[21] = fma_f32x2_rn_ftz(k2[5], vn2xb1_2, S2[21]);
      S2[22] = fma_f32x2_rn_ftz(k2[6], vn2xb1_2, S2[22]);
      S2[23] = fma_f32x2_rn_ftz(k2[7], vn2xb1_2, S2[23]);
      float2 accxb1_3_0 = make_float2(0.0f, 0.0f);
      accxb1_3_0 = fma_f32x2_rn_ftz(S2[24], k2[0], accxb1_3_0);
      accxb1_3_0 = fma_f32x2_rn_ftz(S2[25], k2[1], accxb1_3_0);
      accxb1_3_0 = fma_f32x2_rn_ftz(S2[26], k2[2], accxb1_3_0);
      accxb1_3_0 = fma_f32x2_rn_ftz(S2[27], k2[3], accxb1_3_0);
      accxb1_3_0 = fma_f32x2_rn_ftz(S2[28], k2[4], accxb1_3_0);
      accxb1_3_0 = fma_f32x2_rn_ftz(S2[29], k2[5], accxb1_3_0);
      accxb1_3_0 = fma_f32x2_rn_ftz(S2[30], k2[6], accxb1_3_0);
      accxb1_3_0 = fma_f32x2_rn_ftz(S2[31], k2[7], accxb1_3_0);
      float dotxb1_3 = accxb1_3_0.x + accxb1_3_0.y;
      float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, dotxb1_3, 1);
      dotxb1_3 = dotxb1_3 + _shfl_xor_sync_105;
      float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, dotxb1_3, 2);
      dotxb1_3 = dotxb1_3 + _shfl_xor_sync_106;
      float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, dotxb1_3, 4);
      dotxb1_3 = dotxb1_3 + _shfl_xor_sync_107;
      float vnxb1_3 = (v_regs[3] - dotxb1_3) * betaxb1;
      float2 vn2xb1_3 = make_float2(vnxb1_3, vnxb1_3);
      S2[24] = fma_f32x2_rn_ftz(k2[0], vn2xb1_3, S2[24]);
      S2[25] = fma_f32x2_rn_ftz(k2[1], vn2xb1_3, S2[25]);
      S2[26] = fma_f32x2_rn_ftz(k2[2], vn2xb1_3, S2[26]);
      S2[27] = fma_f32x2_rn_ftz(k2[3], vn2xb1_3, S2[27]);
      S2[28] = fma_f32x2_rn_ftz(k2[4], vn2xb1_3, S2[28]);
      S2[29] = fma_f32x2_rn_ftz(k2[5], vn2xb1_3, S2[29]);
      S2[30] = fma_f32x2_rn_ftz(k2[6], vn2xb1_3, S2[30]);
      S2[31] = fma_f32x2_rn_ftz(k2[7], vn2xb1_3, S2[31]);
      int mxb2 = t_1 * 3 + 2;
      unsigned int kbasexb2 = sK1_addr + (unsigned int)(mxb2 * 512) + lane_byte;
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                   : "r"(kbasexb2));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                   : "r"(kbasexb2 + 128));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                   : "r"(kbasexb2 + 256));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                   : "r"(kbasexb2 + 384));
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
                   : "r"(sV1_addr + (unsigned int)(mxb2 * 256) + group_byte));
      float betaxb2 = sBeta1[mxb2];
      float2 accxb2_0_0 = make_float2(0.0f, 0.0f);
      accxb2_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accxb2_0_0);
      accxb2_0_0 = fma_f32x2_rn_ftz(S2[1], k2[1], accxb2_0_0);
      accxb2_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accxb2_0_0);
      accxb2_0_0 = fma_f32x2_rn_ftz(S2[3], k2[3], accxb2_0_0);
      accxb2_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accxb2_0_0);
      accxb2_0_0 = fma_f32x2_rn_ftz(S2[5], k2[5], accxb2_0_0);
      accxb2_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accxb2_0_0);
      accxb2_0_0 = fma_f32x2_rn_ftz(S2[7], k2[7], accxb2_0_0);
      float dotxb2_0 = accxb2_0_0.x + accxb2_0_0.y;
      float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, dotxb2_0, 1);
      dotxb2_0 = dotxb2_0 + _shfl_xor_sync_108;
      float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, dotxb2_0, 2);
      dotxb2_0 = dotxb2_0 + _shfl_xor_sync_109;
      float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, dotxb2_0, 4);
      dotxb2_0 = dotxb2_0 + _shfl_xor_sync_110;
      float vnxb2_0 = (v_regs[0] - dotxb2_0) * betaxb2;
      float2 vn2xb2_0 = make_float2(vnxb2_0, vnxb2_0);
      S2[0] = fma_f32x2_rn_ftz(k2[0], vn2xb2_0, S2[0]);
      S2[1] = fma_f32x2_rn_ftz(k2[1], vn2xb2_0, S2[1]);
      S2[2] = fma_f32x2_rn_ftz(k2[2], vn2xb2_0, S2[2]);
      S2[3] = fma_f32x2_rn_ftz(k2[3], vn2xb2_0, S2[3]);
      S2[4] = fma_f32x2_rn_ftz(k2[4], vn2xb2_0, S2[4]);
      S2[5] = fma_f32x2_rn_ftz(k2[5], vn2xb2_0, S2[5]);
      S2[6] = fma_f32x2_rn_ftz(k2[6], vn2xb2_0, S2[6]);
      S2[7] = fma_f32x2_rn_ftz(k2[7], vn2xb2_0, S2[7]);
      float2 accxb2_1_0 = make_float2(0.0f, 0.0f);
      accxb2_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accxb2_1_0);
      accxb2_1_0 = fma_f32x2_rn_ftz(S2[9], k2[1], accxb2_1_0);
      accxb2_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accxb2_1_0);
      accxb2_1_0 = fma_f32x2_rn_ftz(S2[11], k2[3], accxb2_1_0);
      accxb2_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accxb2_1_0);
      accxb2_1_0 = fma_f32x2_rn_ftz(S2[13], k2[5], accxb2_1_0);
      accxb2_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accxb2_1_0);
      accxb2_1_0 = fma_f32x2_rn_ftz(S2[15], k2[7], accxb2_1_0);
      float dotxb2_1 = accxb2_1_0.x + accxb2_1_0.y;
      float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, dotxb2_1, 1);
      dotxb2_1 = dotxb2_1 + _shfl_xor_sync_111;
      float _shfl_xor_sync_112 = __shfl_xor_sync(0xFFFFFFFF, dotxb2_1, 2);
      dotxb2_1 = dotxb2_1 + _shfl_xor_sync_112;
      float _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, dotxb2_1, 4);
      dotxb2_1 = dotxb2_1 + _shfl_xor_sync_113;
      float vnxb2_1 = (v_regs[1] - dotxb2_1) * betaxb2;
      float2 vn2xb2_1 = make_float2(vnxb2_1, vnxb2_1);
      S2[8] = fma_f32x2_rn_ftz(k2[0], vn2xb2_1, S2[8]);
      S2[9] = fma_f32x2_rn_ftz(k2[1], vn2xb2_1, S2[9]);
      S2[10] = fma_f32x2_rn_ftz(k2[2], vn2xb2_1, S2[10]);
      S2[11] = fma_f32x2_rn_ftz(k2[3], vn2xb2_1, S2[11]);
      S2[12] = fma_f32x2_rn_ftz(k2[4], vn2xb2_1, S2[12]);
      S2[13] = fma_f32x2_rn_ftz(k2[5], vn2xb2_1, S2[13]);
      S2[14] = fma_f32x2_rn_ftz(k2[6], vn2xb2_1, S2[14]);
      S2[15] = fma_f32x2_rn_ftz(k2[7], vn2xb2_1, S2[15]);
      float2 accxb2_2_0 = make_float2(0.0f, 0.0f);
      accxb2_2_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accxb2_2_0);
      accxb2_2_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accxb2_2_0);
      accxb2_2_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accxb2_2_0);
      accxb2_2_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accxb2_2_0);
      accxb2_2_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accxb2_2_0);
      accxb2_2_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accxb2_2_0);
      accxb2_2_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accxb2_2_0);
      accxb2_2_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accxb2_2_0);
      float dotxb2_2 = accxb2_2_0.x + accxb2_2_0.y;
      float _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, dotxb2_2, 1);
      dotxb2_2 = dotxb2_2 + _shfl_xor_sync_114;
      float _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, dotxb2_2, 2);
      dotxb2_2 = dotxb2_2 + _shfl_xor_sync_115;
      float _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, dotxb2_2, 4);
      dotxb2_2 = dotxb2_2 + _shfl_xor_sync_116;
      float vnxb2_2 = (v_regs[2] - dotxb2_2) * betaxb2;
      float2 vn2xb2_2 = make_float2(vnxb2_2, vnxb2_2);
      S2[16] = fma_f32x2_rn_ftz(k2[0], vn2xb2_2, S2[16]);
      S2[17] = fma_f32x2_rn_ftz(k2[1], vn2xb2_2, S2[17]);
      S2[18] = fma_f32x2_rn_ftz(k2[2], vn2xb2_2, S2[18]);
      S2[19] = fma_f32x2_rn_ftz(k2[3], vn2xb2_2, S2[19]);
      S2[20] = fma_f32x2_rn_ftz(k2[4], vn2xb2_2, S2[20]);
      S2[21] = fma_f32x2_rn_ftz(k2[5], vn2xb2_2, S2[21]);
      S2[22] = fma_f32x2_rn_ftz(k2[6], vn2xb2_2, S2[22]);
      S2[23] = fma_f32x2_rn_ftz(k2[7], vn2xb2_2, S2[23]);
      float2 accxb2_3_0 = make_float2(0.0f, 0.0f);
      accxb2_3_0 = fma_f32x2_rn_ftz(S2[24], k2[0], accxb2_3_0);
      accxb2_3_0 = fma_f32x2_rn_ftz(S2[25], k2[1], accxb2_3_0);
      accxb2_3_0 = fma_f32x2_rn_ftz(S2[26], k2[2], accxb2_3_0);
      accxb2_3_0 = fma_f32x2_rn_ftz(S2[27], k2[3], accxb2_3_0);
      accxb2_3_0 = fma_f32x2_rn_ftz(S2[28], k2[4], accxb2_3_0);
      accxb2_3_0 = fma_f32x2_rn_ftz(S2[29], k2[5], accxb2_3_0);
      accxb2_3_0 = fma_f32x2_rn_ftz(S2[30], k2[6], accxb2_3_0);
      accxb2_3_0 = fma_f32x2_rn_ftz(S2[31], k2[7], accxb2_3_0);
      float dotxb2_3 = accxb2_3_0.x + accxb2_3_0.y;
      float _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, dotxb2_3, 1);
      dotxb2_3 = dotxb2_3 + _shfl_xor_sync_117;
      float _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, dotxb2_3, 2);
      dotxb2_3 = dotxb2_3 + _shfl_xor_sync_118;
      float _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, dotxb2_3, 4);
      dotxb2_3 = dotxb2_3 + _shfl_xor_sync_119;
      float vnxb2_3 = (v_regs[3] - dotxb2_3) * betaxb2;
      float2 vn2xb2_3 = make_float2(vnxb2_3, vnxb2_3);
      S2[24] = fma_f32x2_rn_ftz(k2[0], vn2xb2_3, S2[24]);
      S2[25] = fma_f32x2_rn_ftz(k2[1], vn2xb2_3, S2[25]);
      S2[26] = fma_f32x2_rn_ftz(k2[2], vn2xb2_3, S2[26]);
      S2[27] = fma_f32x2_rn_ftz(k2[3], vn2xb2_3, S2[27]);
      S2[28] = fma_f32x2_rn_ftz(k2[4], vn2xb2_3, S2[28]);
      S2[29] = fma_f32x2_rn_ftz(k2[5], vn2xb2_3, S2[29]);
      S2[30] = fma_f32x2_rn_ftz(k2[6], vn2xb2_3, S2[30]);
      S2[31] = fma_f32x2_rn_ftz(k2[7], vn2xb2_3, S2[31]);
      unsigned int qbasexb = sQ1_addr + (unsigned int)(t_1 * 512) + lane_byte;
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                   : "r"(qbasexb));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                   : "r"(qbasexb + 128));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                   : "r"(qbasexb + 256));
      asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])),
                     "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                   : "r"(qbasexb + 384));
      k2[0] = make_float2(k_regs[0], k_regs[1]);
      k2[1] = make_float2(k_regs[2], k_regs[3]);
      k2[2] = make_float2(k_regs[4], k_regs[5]);
      k2[3] = make_float2(k_regs[6], k_regs[7]);
      k2[4] = make_float2(k_regs[8], k_regs[9]);
      k2[5] = make_float2(k_regs[10], k_regs[11]);
      k2[6] = make_float2(k_regs[12], k_regs[13]);
      k2[7] = make_float2(k_regs[14], k_regs[15]);
      float2 accxbq_0_0 = make_float2(0.0f, 0.0f);
      accxbq_0_0 = fma_f32x2_rn_ftz(S2[0], k2[0], accxbq_0_0);
      accxbq_0_0 = fma_f32x2_rn_ftz(S2[1], k2[1], accxbq_0_0);
      accxbq_0_0 = fma_f32x2_rn_ftz(S2[2], k2[2], accxbq_0_0);
      accxbq_0_0 = fma_f32x2_rn_ftz(S2[3], k2[3], accxbq_0_0);
      accxbq_0_0 = fma_f32x2_rn_ftz(S2[4], k2[4], accxbq_0_0);
      accxbq_0_0 = fma_f32x2_rn_ftz(S2[5], k2[5], accxbq_0_0);
      accxbq_0_0 = fma_f32x2_rn_ftz(S2[6], k2[6], accxbq_0_0);
      accxbq_0_0 = fma_f32x2_rn_ftz(S2[7], k2[7], accxbq_0_0);
      float dotxbq_0 = accxbq_0_0.x + accxbq_0_0.y;
      float _shfl_xor_sync_120 = __shfl_xor_sync(0xFFFFFFFF, dotxbq_0, 1);
      dotxbq_0 = dotxbq_0 + _shfl_xor_sync_120;
      float _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, dotxbq_0, 2);
      dotxbq_0 = dotxbq_0 + _shfl_xor_sync_121;
      float _shfl_xor_sync_122 = __shfl_xor_sync(0xFFFFFFFF, dotxbq_0, 4);
      dotxbq_0 = dotxbq_0 + _shfl_xor_sync_122;
      o4[0] = dotxbq_0;
      float2 accxbq_1_0 = make_float2(0.0f, 0.0f);
      accxbq_1_0 = fma_f32x2_rn_ftz(S2[8], k2[0], accxbq_1_0);
      accxbq_1_0 = fma_f32x2_rn_ftz(S2[9], k2[1], accxbq_1_0);
      accxbq_1_0 = fma_f32x2_rn_ftz(S2[10], k2[2], accxbq_1_0);
      accxbq_1_0 = fma_f32x2_rn_ftz(S2[11], k2[3], accxbq_1_0);
      accxbq_1_0 = fma_f32x2_rn_ftz(S2[12], k2[4], accxbq_1_0);
      accxbq_1_0 = fma_f32x2_rn_ftz(S2[13], k2[5], accxbq_1_0);
      accxbq_1_0 = fma_f32x2_rn_ftz(S2[14], k2[6], accxbq_1_0);
      accxbq_1_0 = fma_f32x2_rn_ftz(S2[15], k2[7], accxbq_1_0);
      float dotxbq_1 = accxbq_1_0.x + accxbq_1_0.y;
      float _shfl_xor_sync_123 = __shfl_xor_sync(0xFFFFFFFF, dotxbq_1, 1);
      dotxbq_1 = dotxbq_1 + _shfl_xor_sync_123;
      float _shfl_xor_sync_124 = __shfl_xor_sync(0xFFFFFFFF, dotxbq_1, 2);
      dotxbq_1 = dotxbq_1 + _shfl_xor_sync_124;
      float _shfl_xor_sync_125 = __shfl_xor_sync(0xFFFFFFFF, dotxbq_1, 4);
      dotxbq_1 = dotxbq_1 + _shfl_xor_sync_125;
      o4[1] = dotxbq_1;
      float2 accxbq_2_0 = make_float2(0.0f, 0.0f);
      accxbq_2_0 = fma_f32x2_rn_ftz(S2[16], k2[0], accxbq_2_0);
      accxbq_2_0 = fma_f32x2_rn_ftz(S2[17], k2[1], accxbq_2_0);
      accxbq_2_0 = fma_f32x2_rn_ftz(S2[18], k2[2], accxbq_2_0);
      accxbq_2_0 = fma_f32x2_rn_ftz(S2[19], k2[3], accxbq_2_0);
      accxbq_2_0 = fma_f32x2_rn_ftz(S2[20], k2[4], accxbq_2_0);
      accxbq_2_0 = fma_f32x2_rn_ftz(S2[21], k2[5], accxbq_2_0);
      accxbq_2_0 = fma_f32x2_rn_ftz(S2[22], k2[6], accxbq_2_0);
      accxbq_2_0 = fma_f32x2_rn_ftz(S2[23], k2[7], accxbq_2_0);
      float dotxbq_2 = accxbq_2_0.x + accxbq_2_0.y;
      float _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, dotxbq_2, 1);
      dotxbq_2 = dotxbq_2 + _shfl_xor_sync_126;
      float _shfl_xor_sync_127 = __shfl_xor_sync(0xFFFFFFFF, dotxbq_2, 2);
      dotxbq_2 = dotxbq_2 + _shfl_xor_sync_127;
      float _shfl_xor_sync_128 = __shfl_xor_sync(0xFFFFFFFF, dotxbq_2, 4);
      dotxbq_2 = dotxbq_2 + _shfl_xor_sync_128;
      o4[2] = dotxbq_2;
      float2 accxbq_3_0 = make_float2(0.0f, 0.0f);
      accxbq_3_0 = fma_f32x2_rn_ftz(S2[24], k2[0], accxbq_3_0);
      accxbq_3_0 = fma_f32x2_rn_ftz(S2[25], k2[1], accxbq_3_0);
      accxbq_3_0 = fma_f32x2_rn_ftz(S2[26], k2[2], accxbq_3_0);
      accxbq_3_0 = fma_f32x2_rn_ftz(S2[27], k2[3], accxbq_3_0);
      accxbq_3_0 = fma_f32x2_rn_ftz(S2[28], k2[4], accxbq_3_0);
      accxbq_3_0 = fma_f32x2_rn_ftz(S2[29], k2[5], accxbq_3_0);
      accxbq_3_0 = fma_f32x2_rn_ftz(S2[30], k2[6], accxbq_3_0);
      accxbq_3_0 = fma_f32x2_rn_ftz(S2[31], k2[7], accxbq_3_0);
      float dotxbq_3 = accxbq_3_0.x + accxbq_3_0.y;
      float _shfl_xor_sync_129 = __shfl_xor_sync(0xFFFFFFFF, dotxbq_3, 1);
      dotxbq_3 = dotxbq_3 + _shfl_xor_sync_129;
      float _shfl_xor_sync_130 = __shfl_xor_sync(0xFFFFFFFF, dotxbq_3, 2);
      dotxbq_3 = dotxbq_3 + _shfl_xor_sync_130;
      float _shfl_xor_sync_131 = __shfl_xor_sync(0xFFFFFFFF, dotxbq_3, 4);
      dotxbq_3 = dotxbq_3 + _shfl_xor_sync_131;
      o4[3] = dotxbq_3;
      if (u == 0) {
        long long out_index_1 = (long long)nB * out_stride_p0 + (long long)t_1 * out_stride_p1 +
                                (long long)hB * out_stride_p2 + (long long)row0;
        unsigned int _packed_1[2];
#pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
          __nv_bfloat162 _bf2 =
              __float22bfloat162_rn(make_float2(o4[_lp * 2 + 0], o4[_lp * 2 + 1 + 0]));
          _packed_1[_lp] = *(uint32_t*)&_bf2;
        }
#pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
          __nv_bfloat162 _bf2 =
              __float22bfloat162_rn(make_float2(o4[_lp * 2 + 2], o4[_lp * 2 + 1 + 2]));
          _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(out + (out_index_1))[0] = reinterpret_cast<int2*>(_packed_1)[0];
      }
      if (use_ssm_state_indices != 0) {
        int ssm_slot_1 = sSsm1[t_1];
        if (ssm_slot_1 >= 0 && state_pool_size > (long long)ssm_slot_1) {
          long long snap_base_1 = (long long)ssm_slot_1 * state_stride_p0 + rowsB_off;
          r_h[0] = S2[0].x;
          r_h[1] = S2[0].y;
          r_h[2] = S2[1].x;
          r_h[3] = S2[1].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base_1) = _v4;
          }
          r_h[0] = S2[2].x;
          r_h[1] = S2[2].y;
          r_h[2] = S2[3].x;
          r_h[3] = S2[3].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base_1 + 32) = _v4;
          }
          r_h[0] = S2[4].x;
          r_h[1] = S2[4].y;
          r_h[2] = S2[5].x;
          r_h[3] = S2[5].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base_1 + 64) = _v4;
          }
          r_h[0] = S2[6].x;
          r_h[1] = S2[6].y;
          r_h[2] = S2[7].x;
          r_h[3] = S2[7].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base_1 + 96) = _v4;
          }
          r_h[0] = S2[8].x;
          r_h[1] = S2[8].y;
          r_h[2] = S2[9].x;
          r_h[3] = S2[9].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base_1 + state_stride_p2) = _v4;
          }
          r_h[0] = S2[10].x;
          r_h[1] = S2[10].y;
          r_h[2] = S2[11].x;
          r_h[3] = S2[11].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base_1 + state_stride_p2 + 32) = _v4;
          }
          r_h[0] = S2[12].x;
          r_h[1] = S2[12].y;
          r_h[2] = S2[13].x;
          r_h[3] = S2[13].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base_1 + state_stride_p2 + 64) = _v4;
          }
          r_h[0] = S2[14].x;
          r_h[1] = S2[14].y;
          r_h[2] = S2[15].x;
          r_h[3] = S2[15].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base_1 + state_stride_p2 + 96) = _v4;
          }
          r_h[0] = S2[16].x;
          r_h[1] = S2[16].y;
          r_h[2] = S2[17].x;
          r_h[3] = S2[17].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base_1 + 2 * state_stride_p2) = _v4;
          }
          r_h[0] = S2[18].x;
          r_h[1] = S2[18].y;
          r_h[2] = S2[19].x;
          r_h[3] = S2[19].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base_1 + 2 * state_stride_p2 + 32) = _v4;
          }
          r_h[0] = S2[20].x;
          r_h[1] = S2[20].y;
          r_h[2] = S2[21].x;
          r_h[3] = S2[21].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base_1 + 2 * state_stride_p2 + 64) = _v4;
          }
          r_h[0] = S2[22].x;
          r_h[1] = S2[22].y;
          r_h[2] = S2[23].x;
          r_h[3] = S2[23].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base_1 + 2 * state_stride_p2 + 96) = _v4;
          }
          r_h[0] = S2[24].x;
          r_h[1] = S2[24].y;
          r_h[2] = S2[25].x;
          r_h[3] = S2[25].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base_1 + 3 * state_stride_p2) = _v4;
          }
          r_h[0] = S2[26].x;
          r_h[1] = S2[26].y;
          r_h[2] = S2[27].x;
          r_h[3] = S2[27].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base_1 + 3 * state_stride_p2 + 32) = _v4;
          }
          r_h[0] = S2[28].x;
          r_h[1] = S2[28].y;
          r_h[2] = S2[29].x;
          r_h[3] = S2[29].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base_1 + 3 * state_stride_p2 + 64) = _v4;
          }
          r_h[0] = S2[30].x;
          r_h[1] = S2[30].y;
          r_h[2] = S2[31].x;
          r_h[3] = S2[31].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + snap_base_1 + 3 * state_stride_p2 + 96) = _v4;
          }
        }
      }
    }
    if (writeB_valid != 0) {
      if (disable_state_update == 0) {
        if (use_ssm_state_indices == 0) {
          r_h[0] = S2[0].x;
          r_h[1] = S2[0].y;
          r_h[2] = S2[1].x;
          r_h[3] = S2[1].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeB_base) = _v4;
          }
          r_h[0] = S2[2].x;
          r_h[1] = S2[2].y;
          r_h[2] = S2[3].x;
          r_h[3] = S2[3].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeB_base + 32) = _v4;
          }
          r_h[0] = S2[4].x;
          r_h[1] = S2[4].y;
          r_h[2] = S2[5].x;
          r_h[3] = S2[5].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeB_base + 64) = _v4;
          }
          r_h[0] = S2[6].x;
          r_h[1] = S2[6].y;
          r_h[2] = S2[7].x;
          r_h[3] = S2[7].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeB_base + 96) = _v4;
          }
          r_h[0] = S2[8].x;
          r_h[1] = S2[8].y;
          r_h[2] = S2[9].x;
          r_h[3] = S2[9].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeB_base + state_stride_p2) = _v4;
          }
          r_h[0] = S2[10].x;
          r_h[1] = S2[10].y;
          r_h[2] = S2[11].x;
          r_h[3] = S2[11].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeB_base + state_stride_p2 + 32) = _v4;
          }
          r_h[0] = S2[12].x;
          r_h[1] = S2[12].y;
          r_h[2] = S2[13].x;
          r_h[3] = S2[13].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeB_base + state_stride_p2 + 64) = _v4;
          }
          r_h[0] = S2[14].x;
          r_h[1] = S2[14].y;
          r_h[2] = S2[15].x;
          r_h[3] = S2[15].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeB_base + state_stride_p2 + 96) = _v4;
          }
          r_h[0] = S2[16].x;
          r_h[1] = S2[16].y;
          r_h[2] = S2[17].x;
          r_h[3] = S2[17].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeB_base + 2 * state_stride_p2) = _v4;
          }
          r_h[0] = S2[18].x;
          r_h[1] = S2[18].y;
          r_h[2] = S2[19].x;
          r_h[3] = S2[19].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeB_base + 2 * state_stride_p2 + 32) = _v4;
          }
          r_h[0] = S2[20].x;
          r_h[1] = S2[20].y;
          r_h[2] = S2[21].x;
          r_h[3] = S2[21].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeB_base + 2 * state_stride_p2 + 64) = _v4;
          }
          r_h[0] = S2[22].x;
          r_h[1] = S2[22].y;
          r_h[2] = S2[23].x;
          r_h[3] = S2[23].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeB_base + 2 * state_stride_p2 + 96) = _v4;
          }
          r_h[0] = S2[24].x;
          r_h[1] = S2[24].y;
          r_h[2] = S2[25].x;
          r_h[3] = S2[25].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeB_base + 3 * state_stride_p2) = _v4;
          }
          r_h[0] = S2[26].x;
          r_h[1] = S2[26].y;
          r_h[2] = S2[27].x;
          r_h[3] = S2[27].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeB_base + 3 * state_stride_p2 + 32) = _v4;
          }
          r_h[0] = S2[28].x;
          r_h[1] = S2[28].y;
          r_h[2] = S2[29].x;
          r_h[3] = S2[29].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeB_base + 3 * state_stride_p2 + 64) = _v4;
          }
          r_h[0] = S2[30].x;
          r_h[1] = S2[30].y;
          r_h[2] = S2[31].x;
          r_h[3] = S2[31].y;
          {
            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<float4*>(state + writeB_base + 3 * state_stride_p2 + 96) = _v4;
          }
        }
      }
    }
  }
}

}  // extern "C"
