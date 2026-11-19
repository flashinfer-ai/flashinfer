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

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
struct CakeFastDivmod { int32_t divisor; uint32_t multiplier; uint32_t shift_right; };
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


#if __CUDA_ARCH__ == 900
__device__ __forceinline__ float mul_f32_rn_ftz(float a, float b) {
    float r;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
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
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
#endif
    return r;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdp_450bb0db3b79acafa419(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, int* __restrict__ ssm_state_indices, CakeFastDivmod heads, int num_tokens, float scale, int disable_state_update, int use_ssm_state_indices, long long state_size_p0, long long state_stride_p1, long long state_stride_p2, long long state_stride_p0, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long ssm_state_indices_stride_p0, long long ssm_state_indices_stride_p1, long long out_stride_p0, long long out_stride_p1, long long out_stride_p2)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 900
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#else
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
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
    uint32_t _fast_div_q_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)blockIdx.x), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)blockIdx.x);
    uint32_t _fast_div_r_0 = (uint32_t)((unsigned int)blockIdx.x) - _fast_div_q_0 * (uint32_t)(heads.divisor);
    int n = (int)_fast_div_q_0;
    int h = (int)_fast_div_r_0;
    int tid_0 = tid;
    int lane_local = lane;
    int warp_local = warp;
    int q_warp_local = warp_local;
    int u = tid_0 - tid_0 / 8 * 8;
    int g = tid_0 / 8;
    int row0 = g * 2;
    unsigned int lane_byte = (unsigned int)(u * 16);
    unsigned int group_byte = (unsigned int)(g * 8);
    long long lane_elem = (long long)(u * 4);
    long long state_pool_size = state_size_p0;
    int read_state_slot_raw = initial_state_indices[n];
    int read_state_slot_valid = ((read_state_slot_raw >= 0 && state_pool_size > (long long)read_state_slot_raw) ? 1 : 0);
    int write_state_slot_raw = output_state_indices[n];
    int write_state_slot_valid = ((write_state_slot_raw >= 0 && state_pool_size > (long long)write_state_slot_raw) ? 1 : 0);
    long long head_rows_off = (long long)h * state_stride_p1 + (long long)row0 * state_stride_p2 + lane_elem;
    long long read_rows_base = (long long)read_state_slot_raw * state_stride_p0 + head_rows_off;
    long long write_rows_base = (long long)write_state_slot_raw * state_stride_p0 + head_rows_off;
    int k_col0 = lane_local * 4;
    int steps = 48;
    float r_q[4];
    float r_k[4];
#if __CUDA_ARCH__ == 900
    float bk[16];
    float bv[8];
    float bb[4];
    int bm0_0 = warp_local;
    int bm_0 = bm0_0;
    long long bki_0 = (long long)n * k_stride_p0 + (long long)bm_0 * k_stride_p1 + (long long)h * k_stride_p2;
    {
        uint2 _vld_0;
        _vld_0 = *reinterpret_cast<const uint2*>(k + bki_0 + (long long)k_col0);
        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&bk[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
            (&bk[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
        }
    }
    long long bvi_0 = (long long)n * v_stride_p0 + (long long)bm_0 * v_stride_p1 + (long long)h * v_stride_p2;
    int bvc_0 = lane_local * 2;
    bv[0] = (float)v[bvi_0 + (long long)bvc_0];
    bv[1] = (float)v[bvi_0 + (long long)bvc_0 + 1ll];
    long long bbi_0 = (long long)n * b_stride_p0 + (long long)bm_0 * b_stride_p1 + (long long)h * b_stride_p2;
    bb[0] = (float)b[bbi_0];
    int bm0_1 = 8 + warp_local;
    int bm_1 = bm0_1;
    long long bki_1 = (long long)n * k_stride_p0 + (long long)bm_1 * k_stride_p1 + (long long)h * k_stride_p2;
    {
        uint2 _vld_1;
        _vld_1 = *reinterpret_cast<const uint2*>(k + bki_1 + (long long)k_col0);
        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&bk[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
            (&bk[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
        }
    }
    long long bvi_1 = (long long)n * v_stride_p0 + (long long)bm_1 * v_stride_p1 + (long long)h * v_stride_p2;
    int bvc_1 = lane_local * 2;
    bv[2] = (float)v[bvi_1 + (long long)bvc_1];
    bv[3] = (float)v[bvi_1 + (long long)bvc_1 + 1ll];
    long long bbi_1 = (long long)n * b_stride_p0 + (long long)bm_1 * b_stride_p1 + (long long)h * b_stride_p2;
    bb[1] = (float)b[bbi_1];
    int bm0_2 = 16 + warp_local;
    int bm_2 = bm0_2;
    long long bki_2 = (long long)n * k_stride_p0 + (long long)bm_2 * k_stride_p1 + (long long)h * k_stride_p2;
    {
        uint2 _vld_2;
        _vld_2 = *reinterpret_cast<const uint2*>(k + bki_2 + (long long)k_col0);
        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&bk[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
            (&bk[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
        }
    }
    long long bvi_2 = (long long)n * v_stride_p0 + (long long)bm_2 * v_stride_p1 + (long long)h * v_stride_p2;
    int bvc_2 = lane_local * 2;
    bv[4] = (float)v[bvi_2 + (long long)bvc_2];
    bv[5] = (float)v[bvi_2 + (long long)bvc_2 + 1ll];
    long long bbi_2 = (long long)n * b_stride_p0 + (long long)bm_2 * b_stride_p1 + (long long)h * b_stride_p2;
    bb[2] = (float)b[bbi_2];
    int bm0_3 = 24 + warp_local;
    int bm_3 = bm0_3;
    long long bki_3 = (long long)n * k_stride_p0 + (long long)bm_3 * k_stride_p1 + (long long)h * k_stride_p2;
    {
        uint2 _vld_3;
        _vld_3 = *reinterpret_cast<const uint2*>(k + bki_3 + (long long)k_col0);
        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&bk[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
            (&bk[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
        }
    }
    long long bvi_3 = (long long)n * v_stride_p0 + (long long)bm_3 * v_stride_p1 + (long long)h * v_stride_p2;
    int bvc_3 = lane_local * 2;
    bv[6] = (float)v[bvi_3 + (long long)bvc_3];
    bv[7] = (float)v[bvi_3 + (long long)bvc_3 + 1ll];
    long long bbi_3 = (long long)n * b_stride_p0 + (long long)bm_3 * b_stride_p1 + (long long)h * b_stride_p2;
    bb[3] = (float)b[bbi_3];
#else
    unsigned int w_k[2];
    unsigned int w_v[2];
    unsigned int w_b[1];
    unsigned int w_q[2];
    unsigned int w_a[1];
    float r_a[3];
    float r_v[2];
    float r_b[1];
#endif
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
        float r_h[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base);
            S[0 + 0] = _v4.x;
            S[0 + 1] = _v4.y;
            S[0 + 2] = _v4.z;
            S[0 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 32ll);
            S[4 + 0] = _v4.x;
            S[4 + 1] = _v4.y;
            S[4 + 2] = _v4.z;
            S[4 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 64ll);
            S[8 + 0] = _v4.x;
            S[8 + 1] = _v4.y;
            S[8 + 2] = _v4.z;
            S[8 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 96ll);
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
            float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + state_stride_p2 + 32ll);
            S[20 + 0] = _v4.x;
            S[20 + 1] = _v4.y;
            S[20 + 2] = _v4.z;
            S[20 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + state_stride_p2 + 64ll);
            S[24 + 0] = _v4.x;
            S[24 + 1] = _v4.y;
            S[24 + 2] = _v4.z;
            S[24 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + state_stride_p2 + 96ll);
            S[28 + 0] = _v4.x;
            S[28 + 1] = _v4.y;
            S[28 + 2] = _v4.z;
            S[28 + 3] = _v4.w;
        }
#if __CUDA_ARCH__ == 900
        int fm_0 = warp_local;
        float x0_x = bk[0];
        float x0_y = bk[1];
        float x1_x = bk[2];
        float x1_y = bk[3];
        float sum_x;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x) : "f"(x0_x), "f"(x0_x), "f"(0.0f));
        float sum_y;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y) : "f"(x0_y), "f"(x0_y), "f"(0.0f));
        float _fma_12;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(x1_x), "f"(x1_x), "f"(sum_x));
        sum_x = _fma_12;
        float _fma_13;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(x1_y), "f"(x1_y), "f"(sum_y));
        sum_y = _fma_13;
        sum_x = sum_x + sum_y;
        float _warp_reduce_14 = sum_x;
#endif
        #pragma unroll
#if __CUDA_ARCH__ == 900
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_14 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_14, offset);
        float norm = rsqrtf(_warp_reduce_14 + 1e-06f);
        x0_x = mul_f32_rn_ftz(x0_x, norm);
        x0_y = mul_f32_rn_ftz(x0_y, norm);
        x1_x = mul_f32_rn_ftz(x1_x, norm);
        x1_y = mul_f32_rn_ftz(x1_y, norm);
        sK[fm_0 * 128 + k_col0] = x0_x;
        sK[fm_0 * 128 + k_col0 + 1] = x0_y;
        sK[fm_0 * 128 + k_col0 + 2] = x1_x;
        sK[fm_0 * 128 + k_col0 + 3] = x1_y;
        int v_col = lane_local * 2;
        sV[fm_0 * 64 + v_col] = bv[0];
        sV[fm_0 * 64 + v_col + 1] = bv[1];
        if (lane_local == 0) {
            float _exp_16 = __expf(-bb[0]);
            float _rcp_15 = approx_rcp(1.0f + _exp_16);
            sBeta[fm_0] = _rcp_15;
        }
        int fm_1 = 8 + warp_local;
        float x0_x_0 = bk[4];
        float x0_y_0 = bk[5];
        float x1_x_0 = bk[6];
        float x1_y_0 = bk[7];
        float sum_x_0;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_0) : "f"(x0_x_0), "f"(x0_x_0), "f"(0.0f));
        float sum_y_0;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_0) : "f"(x0_y_0), "f"(x0_y_0), "f"(0.0f));
        float _fma_17;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(x1_x_0), "f"(x1_x_0), "f"(sum_x_0));
        sum_x_0 = _fma_17;
        float _fma_18;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(x1_y_0), "f"(x1_y_0), "f"(sum_y_0));
        sum_y_0 = _fma_18;
        sum_x_0 = sum_x_0 + sum_y_0;
        float _warp_reduce_19 = sum_x_0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_19 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_19, offset);
        float norm_0 = rsqrtf(_warp_reduce_19 + 1e-06f);
        x0_x_0 = mul_f32_rn_ftz(x0_x_0, norm_0);
        x0_y_0 = mul_f32_rn_ftz(x0_y_0, norm_0);
        x1_x_0 = mul_f32_rn_ftz(x1_x_0, norm_0);
        x1_y_0 = mul_f32_rn_ftz(x1_y_0, norm_0);
        sK[fm_1 * 128 + k_col0] = x0_x_0;
        sK[fm_1 * 128 + k_col0 + 1] = x0_y_0;
        sK[fm_1 * 128 + k_col0 + 2] = x1_x_0;
        sK[fm_1 * 128 + k_col0 + 3] = x1_y_0;
        int v_col_0 = lane_local * 2;
        sV[fm_1 * 64 + v_col_0] = bv[2];
        sV[fm_1 * 64 + v_col_0 + 1] = bv[3];
        if (lane_local == 0) {
            float _exp_21 = __expf(-bb[1]);
            float _rcp_20 = approx_rcp(1.0f + _exp_21);
            sBeta[fm_1] = _rcp_20;
        }
        int fm_2 = 16 + warp_local;
        float x0_x_1 = bk[8];
        float x0_y_1 = bk[9];
        float x1_x_1 = bk[10];
        float x1_y_1 = bk[11];
        float sum_x_1;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_1) : "f"(x0_x_1), "f"(x0_x_1), "f"(0.0f));
        float sum_y_1;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_1) : "f"(x0_y_1), "f"(x0_y_1), "f"(0.0f));
        float _fma_22;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(x1_x_1), "f"(x1_x_1), "f"(sum_x_1));
        sum_x_1 = _fma_22;
        float _fma_23;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(x1_y_1), "f"(x1_y_1), "f"(sum_y_1));
        sum_y_1 = _fma_23;
        sum_x_1 = sum_x_1 + sum_y_1;
        float _warp_reduce_24 = sum_x_1;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_24 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_24, offset);
        float norm_1 = rsqrtf(_warp_reduce_24 + 1e-06f);
        x0_x_1 = mul_f32_rn_ftz(x0_x_1, norm_1);
        x0_y_1 = mul_f32_rn_ftz(x0_y_1, norm_1);
        x1_x_1 = mul_f32_rn_ftz(x1_x_1, norm_1);
        x1_y_1 = mul_f32_rn_ftz(x1_y_1, norm_1);
        sK[fm_2 * 128 + k_col0] = x0_x_1;
        sK[fm_2 * 128 + k_col0 + 1] = x0_y_1;
        sK[fm_2 * 128 + k_col0 + 2] = x1_x_1;
        sK[fm_2 * 128 + k_col0 + 3] = x1_y_1;
        int v_col_1 = lane_local * 2;
        sV[fm_2 * 64 + v_col_1] = bv[4];
        sV[fm_2 * 64 + v_col_1 + 1] = bv[5];
        if (lane_local == 0) {
            float _exp_26 = __expf(-bb[2]);
            float _rcp_25 = approx_rcp(1.0f + _exp_26);
            sBeta[fm_2] = _rcp_25;
        }
        int fm_3 = 24 + warp_local;
        float x0_x_2 = bk[12];
        float x0_y_2 = bk[13];
        float x1_x_2 = bk[14];
        float x1_y_2 = bk[15];
        float sum_x_2;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_2) : "f"(x0_x_2), "f"(x0_x_2), "f"(0.0f));
        float sum_y_2;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_2) : "f"(x0_y_2), "f"(x0_y_2), "f"(0.0f));
        float _fma_27;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(x1_x_2), "f"(x1_x_2), "f"(sum_x_2));
        sum_x_2 = _fma_27;
        float _fma_28;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(x1_y_2), "f"(x1_y_2), "f"(sum_y_2));
        sum_y_2 = _fma_28;
        sum_x_2 = sum_x_2 + sum_y_2;
        float _warp_reduce_29 = sum_x_2;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_29 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_29, offset);
        float norm_2 = rsqrtf(_warp_reduce_29 + 1e-06f);
        x0_x_2 = mul_f32_rn_ftz(x0_x_2, norm_2);
        x0_y_2 = mul_f32_rn_ftz(x0_y_2, norm_2);
        x1_x_2 = mul_f32_rn_ftz(x1_x_2, norm_2);
        x1_y_2 = mul_f32_rn_ftz(x1_y_2, norm_2);
        sK[fm_3 * 128 + k_col0] = x0_x_2;
        sK[fm_3 * 128 + k_col0 + 1] = x0_y_2;
        sK[fm_3 * 128 + k_col0 + 2] = x1_x_2;
        sK[fm_3 * 128 + k_col0 + 3] = x1_y_2;
        int v_col_2 = lane_local * 2;
        sV[fm_3 * 64 + v_col_2] = bv[6];
        sV[fm_3 * 64 + v_col_2 + 1] = bv[7];
        if (lane_local == 0) {
            float _exp_31 = __expf(-bb[3]);
            float _rcp_30 = approx_rcp(1.0f + _exp_31);
            sBeta[fm_3] = _rcp_30;
        }
        float bk_0[8];
        float bv_0[4];
        float bb_0[2];
        int bm0_4 = 32 + warp_local;
        int bm_4 = bm0_4;
        long long bki_4 = (long long)n * k_stride_p0 + (long long)bm_4 * k_stride_p1 + (long long)h * k_stride_p2;
        {
            uint2 _vld_32;
            _vld_32 = *reinterpret_cast<const uint2*>(k + bki_4 + (long long)k_col0);
            uint32_t* _vpairs_32 = reinterpret_cast<uint32_t*>(&_vld_32);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&bk_0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) << 16);
                (&bk_0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) & 0xffff0000u);
#else
        for (int mm = 0; mm < 6; mm++) {
            int m = mm * 8 + warp_local;
            if ((mm + 1) * 8 <= 48) {
                long long dk_index = (long long)n * k_stride_p0 + (long long)m * k_stride_p1 + (long long)h * k_stride_p2;
                {
                    uint2 _vld_8;
                    _vld_8 = *reinterpret_cast<const uint2*>(k + dk_index + (long long)k_col0);
                    uint2* _vdst_8 = reinterpret_cast<uint2*>(&w_k[0]);
                    *_vdst_8 = _vld_8;
                }
                long long dv_index = (long long)n * v_stride_p0 + (long long)m * v_stride_p1 + (long long)h * v_stride_p2;
                int dv_col = lane_local * 2;
                {
                    uint32_t _bf16_bits_9;
                    _bf16_bits_9 = *reinterpret_cast<const uint16_t*>(v + dv_index + (long long)dv_col);
                    w_v[0] = _bf16_bits_9;
                }
                {
                    uint32_t _bf16_bits_10;
                    _bf16_bits_10 = *reinterpret_cast<const uint16_t*>(v + dv_index + (long long)dv_col + 1ll);
                    w_v[1] = _bf16_bits_10;
                }
                long long db_index = (long long)n * b_stride_p0 + (long long)m * b_stride_p1 + (long long)h * b_stride_p2;
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
                float2 x0 = make_float2(r_k[0], r_k[1]);
                float2 x1 = make_float2(r_k[2], r_k[3]);
                float2 sum_pair = fma_f32x2_rn_ftz(x0, x0, make_float2(0.0f, 0.0f));
                sum_pair = fma_f32x2_rn_ftz(x1, x1, sum_pair);
                float sum_x = sum_pair.x + sum_pair.y;
                float _warp_reduce_12 = sum_x;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_12 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_12, offset);
                float norm = rsqrtf(_warp_reduce_12 + 1e-06f);
                float2 norm2 = make_float2(norm, norm);
                x0 = mul_f32x2(x0, norm2);
                x1 = mul_f32x2(x1, norm2);
                sK[m * 128 + k_col0] = x0.x;
                sK[m * 128 + k_col0 + 1] = x0.y;
                sK[m * 128 + k_col0 + 2] = x1.x;
                sK[m * 128 + k_col0 + 3] = x1.y;
                int v_col = lane_local * 2;
                sV[m * 64 + v_col] = r_v[0];
                sV[m * 64 + v_col + 1] = r_v[1];
                if (lane_local == 0) {
                    float _exp_14 = __expf(-r_b[0]);
                    float _rcp_13 = approx_rcp(1.0f + _exp_14);
                    sBeta[m] = _rcp_13;
                }
            }
            if ((mm + 1) * 8 > 48) {
                if (m < steps) {
                    long long dk_index_1 = (long long)n * k_stride_p0 + (long long)m * k_stride_p1 + (long long)h * k_stride_p2;
                    {
                        uint2 _vld_15;
                        _vld_15 = *reinterpret_cast<const uint2*>(k + dk_index_1 + (long long)k_col0);
                        uint2* _vdst_15 = reinterpret_cast<uint2*>(&w_k[0]);
                        *_vdst_15 = _vld_15;
                    }
                    long long dv_index_1 = (long long)n * v_stride_p0 + (long long)m * v_stride_p1 + (long long)h * v_stride_p2;
                    int dv_col_1 = lane_local * 2;
                    {
                        uint32_t _bf16_bits_16;
                        _bf16_bits_16 = *reinterpret_cast<const uint16_t*>(v + dv_index_1 + (long long)dv_col_1);
                        w_v[0] = _bf16_bits_16;
                    }
                    {
                        uint32_t _bf16_bits_17;
                        _bf16_bits_17 = *reinterpret_cast<const uint16_t*>(v + dv_index_1 + (long long)dv_col_1 + 1ll);
                        w_v[1] = _bf16_bits_17;
                    }
                    long long db_index_1 = (long long)n * b_stride_p0 + (long long)m * b_stride_p1 + (long long)h * b_stride_p2;
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
                    float2 x0_1 = make_float2(r_k[0], r_k[1]);
                    float2 x1_1 = make_float2(r_k[2], r_k[3]);
                    float2 sum_pair_1 = fma_f32x2_rn_ftz(x0_1, x0_1, make_float2(0.0f, 0.0f));
                    sum_pair_1 = fma_f32x2_rn_ftz(x1_1, x1_1, sum_pair_1);
                    float sum_x_1 = sum_pair_1.x + sum_pair_1.y;
                    float _warp_reduce_19 = sum_x_1;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_19 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_19, offset);
                    float norm_1 = rsqrtf(_warp_reduce_19 + 1e-06f);
                    float2 norm2_1 = make_float2(norm_1, norm_1);
                    x0_1 = mul_f32x2(x0_1, norm2_1);
                    x1_1 = mul_f32x2(x1_1, norm2_1);
                    sK[m * 128 + k_col0] = x0_1.x;
                    sK[m * 128 + k_col0 + 1] = x0_1.y;
                    sK[m * 128 + k_col0 + 2] = x1_1.x;
                    sK[m * 128 + k_col0 + 3] = x1_1.y;
                    int v_col_1 = lane_local * 2;
                    sV[m * 64 + v_col_1] = r_v[0];
                    sV[m * 64 + v_col_1 + 1] = r_v[1];
                    if (lane_local == 0) {
                        float _exp_21 = __expf(-r_b[0]);
                        float _rcp_20 = approx_rcp(1.0f + _exp_21);
                        sBeta[m] = _rcp_20;
                    }
                }
#endif
            }
        }
#if __CUDA_ARCH__ == 900
        long long bvi_4 = (long long)n * v_stride_p0 + (long long)bm_4 * v_stride_p1 + (long long)h * v_stride_p2;
        int bvc_4 = lane_local * 2;
        bv_0[0] = (float)v[bvi_4 + (long long)bvc_4];
        bv_0[1] = (float)v[bvi_4 + (long long)bvc_4 + 1ll];
        long long bbi_4 = (long long)n * b_stride_p0 + (long long)bm_4 * b_stride_p1 + (long long)h * b_stride_p2;
        bb_0[0] = (float)b[bbi_4];
        int bm0_5 = 40 + warp_local;
        int bm_5 = bm0_5;
        long long bki_5 = (long long)n * k_stride_p0 + (long long)bm_5 * k_stride_p1 + (long long)h * k_stride_p2;
        {
            uint2 _vld_33;
            _vld_33 = *reinterpret_cast<const uint2*>(k + bki_5 + (long long)k_col0);
            uint32_t* _vpairs_33 = reinterpret_cast<uint32_t*>(&_vld_33);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&bk_0[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) << 16);
                (&bk_0[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) & 0xffff0000u);
            }
        }
        long long bvi_5 = (long long)n * v_stride_p0 + (long long)bm_5 * v_stride_p1 + (long long)h * v_stride_p2;
        int bvc_5 = lane_local * 2;
        bv_0[2] = (float)v[bvi_5 + (long long)bvc_5];
        bv_0[3] = (float)v[bvi_5 + (long long)bvc_5 + 1ll];
        long long bbi_5 = (long long)n * b_stride_p0 + (long long)bm_5 * b_stride_p1 + (long long)h * b_stride_p2;
        bb_0[1] = (float)b[bbi_5];
        float bq[8];
        float ba[6];
        int bt0_0 = warp_local;
        int bt_0 = bt0_0;
        long long bqi_0 = (long long)n * q_stride_p0 + (long long)bt_0 * q_stride_p1 + (long long)h * q_stride_p2;
        {
            uint2 _vld_34;
            _vld_34 = *reinterpret_cast<const uint2*>(q + bqi_0 + (long long)k_col0);
            uint32_t* _vpairs_34 = reinterpret_cast<uint32_t*>(&_vld_34);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&bq[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) << 16);
                (&bq[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) & 0xffff0000u);
            }
        }
        long long bgi_0 = (long long)n * a_stride_p0 + (long long)bt_0 * a_stride_p1 + (long long)h * a_stride_p2;
        ba[0] = (float)a[bgi_0];
        ba[1] = dt_bias[h];
        ba[2] = A_log[h];
        int bt0_1 = 8 + warp_local;
        int bt_1 = bt0_1;
        long long bqi_1 = (long long)n * q_stride_p0 + (long long)bt_1 * q_stride_p1 + (long long)h * q_stride_p2;
        {
            uint2 _vld_35;
            _vld_35 = *reinterpret_cast<const uint2*>(q + bqi_1 + (long long)k_col0);
            uint32_t* _vpairs_35 = reinterpret_cast<uint32_t*>(&_vld_35);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&bq[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_35[_pair]) << 16);
                (&bq[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_35[_pair]) & 0xffff0000u);
            }
        }
        long long bgi_1 = (long long)n * a_stride_p0 + (long long)bt_1 * a_stride_p1 + (long long)h * a_stride_p2;
        ba[3] = (float)a[bgi_1];
        ba[4] = dt_bias[h];
        ba[5] = A_log[h];
        int fm_4 = 32 + warp_local;
        float x0_x_3 = bk_0[0];
        float x0_y_3 = bk_0[1];
        float x1_x_3 = bk_0[2];
        float x1_y_3 = bk_0[3];
        float sum_x_3;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_3) : "f"(x0_x_3), "f"(x0_x_3), "f"(0.0f));
        float sum_y_3;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_3) : "f"(x0_y_3), "f"(x0_y_3), "f"(0.0f));
        float _fma_36;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(x1_x_3), "f"(x1_x_3), "f"(sum_x_3));
        sum_x_3 = _fma_36;
        float _fma_37;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(x1_y_3), "f"(x1_y_3), "f"(sum_y_3));
        sum_y_3 = _fma_37;
        sum_x_3 = sum_x_3 + sum_y_3;
        float _warp_reduce_38 = sum_x_3;
#endif
        #pragma unroll
#if __CUDA_ARCH__ == 900
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_38 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_38, offset);
        float norm_3 = rsqrtf(_warp_reduce_38 + 1e-06f);
        x0_x_3 = mul_f32_rn_ftz(x0_x_3, norm_3);
        x0_y_3 = mul_f32_rn_ftz(x0_y_3, norm_3);
        x1_x_3 = mul_f32_rn_ftz(x1_x_3, norm_3);
        x1_y_3 = mul_f32_rn_ftz(x1_y_3, norm_3);
        sK[fm_4 * 128 + k_col0] = x0_x_3;
        sK[fm_4 * 128 + k_col0 + 1] = x0_y_3;
        sK[fm_4 * 128 + k_col0 + 2] = x1_x_3;
        sK[fm_4 * 128 + k_col0 + 3] = x1_y_3;
        int v_col_3 = lane_local * 2;
        sV[fm_4 * 64 + v_col_3] = bv_0[0];
        sV[fm_4 * 64 + v_col_3 + 1] = bv_0[1];
        if (lane_local == 0) {
            float _exp_40 = __expf(-bb_0[0]);
            float _rcp_39 = approx_rcp(1.0f + _exp_40);
            sBeta[fm_4] = _rcp_39;
        }
        int fm_5 = 40 + warp_local;
        float x0_x_4 = bk_0[4];
        float x0_y_4 = bk_0[5];
        float x1_x_4 = bk_0[6];
        float x1_y_4 = bk_0[7];
        float sum_x_4;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_4) : "f"(x0_x_4), "f"(x0_x_4), "f"(0.0f));
        float sum_y_4;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_4) : "f"(x0_y_4), "f"(x0_y_4), "f"(0.0f));
        float _fma_41;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(x1_x_4), "f"(x1_x_4), "f"(sum_x_4));
        sum_x_4 = _fma_41;
        float _fma_42;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(x1_y_4), "f"(x1_y_4), "f"(sum_y_4));
        sum_y_4 = _fma_42;
        sum_x_4 = sum_x_4 + sum_y_4;
        float _warp_reduce_43 = sum_x_4;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_43 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_43, offset);
        float norm_4 = rsqrtf(_warp_reduce_43 + 1e-06f);
        x0_x_4 = mul_f32_rn_ftz(x0_x_4, norm_4);
        x0_y_4 = mul_f32_rn_ftz(x0_y_4, norm_4);
        x1_x_4 = mul_f32_rn_ftz(x1_x_4, norm_4);
        x1_y_4 = mul_f32_rn_ftz(x1_y_4, norm_4);
        sK[fm_5 * 128 + k_col0] = x0_x_4;
        sK[fm_5 * 128 + k_col0 + 1] = x0_y_4;
        sK[fm_5 * 128 + k_col0 + 2] = x1_x_4;
        sK[fm_5 * 128 + k_col0 + 3] = x1_y_4;
        int v_col_4 = lane_local * 2;
        sV[fm_5 * 64 + v_col_4] = bv_0[2];
        sV[fm_5 * 64 + v_col_4 + 1] = bv_0[3];
        if (lane_local == 0) {
            float _exp_45 = __expf(-bb_0[1]);
            float _rcp_44 = approx_rcp(1.0f + _exp_45);
            sBeta[fm_5] = _rcp_44;
        }
        int ft_0 = warp_local;
        float x0_x_5 = bq[0];
        float x0_y_5 = bq[1];
        float x1_x_5 = bq[2];
        float x1_y_5 = bq[3];
        float sum_x_5;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_5) : "f"(x0_x_5), "f"(x0_x_5), "f"(0.0f));
        float sum_y_5;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_5) : "f"(x0_y_5), "f"(x0_y_5), "f"(0.0f));
        float _fma_46;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(x1_x_5), "f"(x1_x_5), "f"(sum_x_5));
        sum_x_5 = _fma_46;
        float _fma_47;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(x1_y_5), "f"(x1_y_5), "f"(sum_y_5));
        sum_y_5 = _fma_47;
        sum_x_5 = sum_x_5 + sum_y_5;
        float _warp_reduce_48 = sum_x_5;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_48 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_48, offset);
        float norm_5 = rsqrtf(_warp_reduce_48 + 1e-06f);
        norm_5 = norm_5 * scale;
        x0_x_5 = mul_f32_rn_ftz(x0_x_5, norm_5);
        x0_y_5 = mul_f32_rn_ftz(x0_y_5, norm_5);
        x1_x_5 = mul_f32_rn_ftz(x1_x_5, norm_5);
        x1_y_5 = mul_f32_rn_ftz(x1_y_5, norm_5);
        sQ[ft_0 * 128 + k_col0] = x0_x_5;
        sQ[ft_0 * 128 + k_col0 + 1] = x0_y_5;
        sQ[ft_0 * 128 + k_col0 + 2] = x1_x_5;
        sQ[ft_0 * 128 + k_col0 + 3] = x1_y_5;
        if (lane_local == 0) {
            float xv = ba[0] + ba[1];
            float softplus_x = xv;
            if (xv <= 20.0f) {
                float _exp_50 = __expf(xv);
                float _log2_49;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_49) : "f"(1.0f + _exp_50));
                softplus_x = _log2_49 * 0.6931472f;
#else
        for (int tt = 0; tt < 2; tt++) {
            int t = tt * 8 + q_warp_local;
            if ((tt + 1) * 8 <= 16) {
                long long dq_index = (long long)n * q_stride_p0 + (long long)t * q_stride_p1 + (long long)h * q_stride_p2;
                {
                    uint2 _vld_22;
                    _vld_22 = *reinterpret_cast<const uint2*>(q + dq_index + (long long)k_col0);
                    uint2* _vdst_22 = reinterpret_cast<uint2*>(&w_q[0]);
                    *_vdst_22 = _vld_22;
                }
                long long dg_index = (long long)n * a_stride_p0 + (long long)t * a_stride_p1 + (long long)h * a_stride_p2;
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
                float2 x0_2 = make_float2(r_q[0], r_q[1]);
                float2 x1_2 = make_float2(r_q[2], r_q[3]);
                float2 sum_pair_2 = fma_f32x2_rn_ftz(x0_2, x0_2, make_float2(0.0f, 0.0f));
                sum_pair_2 = fma_f32x2_rn_ftz(x1_2, x1_2, sum_pair_2);
                float sum_x_2 = sum_pair_2.x + sum_pair_2.y;
                float _warp_reduce_24 = sum_x_2;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_24 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_24, offset);
                float norm_2 = rsqrtf(_warp_reduce_24 + 1e-06f);
                norm_2 = norm_2 * scale;
                float2 norm2_2 = make_float2(norm_2, norm_2);
                x0_2 = mul_f32x2(x0_2, norm2_2);
                x1_2 = mul_f32x2(x1_2, norm2_2);
                sQ[t * 128 + k_col0] = x0_2.x;
                sQ[t * 128 + k_col0 + 1] = x0_2.y;
                sQ[t * 128 + k_col0 + 2] = x1_2.x;
                sQ[t * 128 + k_col0 + 3] = x1_2.y;
                float a_log_v = r_a[2];
                asm("" : "+f"(a_log_v));
                float xv = r_a[0] + r_a[1];
                float softplus_x = xv;
                if (xv <= 20.0f) {
                    float _exp_26 = __expf(xv);
                    float _log2_25;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_25) : "f"(1.0f + _exp_26));
                    softplus_x = _log2_25 * 0.6931472f;
                }
                float _exp_27 = __expf(a_log_v);
                float g_log = -_exp_27 * softplus_x;
                float alpha = __expf(g_log);
                if (lane_local == 0) {
                    sAlpha[t] = alpha;
                    if (use_ssm_state_indices != 0) {
                        sSsm[t] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)t * ssm_state_indices_stride_p1];
                    }
                    if (use_ssm_state_indices == 0) {
                        sSsm[t] = -1;
                    }
                }
#endif
            }
#if __CUDA_ARCH__ == 900
            float _exp_51 = __expf(ba[2]);
            float g_log = -_exp_51 * softplus_x;
            float _exp_52 = __expf(g_log);
            sAlpha[ft_0] = _exp_52;
            if (use_ssm_state_indices != 0) {
                sSsm[ft_0] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)ft_0 * ssm_state_indices_stride_p1];
            }
            if (use_ssm_state_indices == 0) {
                sSsm[ft_0] = -1;
            }
        }
        int ft_1 = 8 + warp_local;
        float x0_x_6 = bq[4];
        float x0_y_6 = bq[5];
        float x1_x_6 = bq[6];
        float x1_y_6 = bq[7];
        float sum_x_6;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_6) : "f"(x0_x_6), "f"(x0_x_6), "f"(0.0f));
        float sum_y_6;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_6) : "f"(x0_y_6), "f"(x0_y_6), "f"(0.0f));
        float _fma_53;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(x1_x_6), "f"(x1_x_6), "f"(sum_x_6));
        sum_x_6 = _fma_53;
        float _fma_54;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(x1_y_6), "f"(x1_y_6), "f"(sum_y_6));
        sum_y_6 = _fma_54;
        sum_x_6 = sum_x_6 + sum_y_6;
        float _warp_reduce_55 = sum_x_6;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_55 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_55, offset);
        float norm_6 = rsqrtf(_warp_reduce_55 + 1e-06f);
        norm_6 = norm_6 * scale;
        x0_x_6 = mul_f32_rn_ftz(x0_x_6, norm_6);
        x0_y_6 = mul_f32_rn_ftz(x0_y_6, norm_6);
        x1_x_6 = mul_f32_rn_ftz(x1_x_6, norm_6);
        x1_y_6 = mul_f32_rn_ftz(x1_y_6, norm_6);
        sQ[ft_1 * 128 + k_col0] = x0_x_6;
        sQ[ft_1 * 128 + k_col0 + 1] = x0_y_6;
        sQ[ft_1 * 128 + k_col0 + 2] = x1_x_6;
        sQ[ft_1 * 128 + k_col0 + 3] = x1_y_6;
        if (lane_local == 0) {
            float xv_1 = ba[3] + ba[4];
            float softplus_x_1 = xv_1;
            if (xv_1 <= 20.0f) {
                float _exp_57 = __expf(xv_1);
                float _log2_56;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_56) : "f"(1.0f + _exp_57));
                softplus_x_1 = _log2_56 * 0.6931472f;
            }
            float _exp_58 = __expf(ba[5]);
            float g_log_1 = -_exp_58 * softplus_x_1;
            float _exp_59 = __expf(g_log_1);
            sAlpha[ft_1] = _exp_59;
            if (use_ssm_state_indices != 0) {
                sSsm[ft_1] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)ft_1 * ssm_state_indices_stride_p1];
            }
            if (use_ssm_state_indices == 0) {
                sSsm[ft_1] = -1;
#else
            if ((tt + 1) * 8 > 16) {
                if (t < num_tokens) {
                    long long dq_index_1 = (long long)n * q_stride_p0 + (long long)t * q_stride_p1 + (long long)h * q_stride_p2;
                    {
                        uint2 _vld_28;
                        _vld_28 = *reinterpret_cast<const uint2*>(q + dq_index_1 + (long long)k_col0);
                        uint2* _vdst_28 = reinterpret_cast<uint2*>(&w_q[0]);
                        *_vdst_28 = _vld_28;
                    }
                    long long dg_index_1 = (long long)n * a_stride_p0 + (long long)t * a_stride_p1 + (long long)h * a_stride_p2;
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
                    float2 x0_3 = make_float2(r_q[0], r_q[1]);
                    float2 x1_3 = make_float2(r_q[2], r_q[3]);
                    float2 sum_pair_3 = fma_f32x2_rn_ftz(x0_3, x0_3, make_float2(0.0f, 0.0f));
                    sum_pair_3 = fma_f32x2_rn_ftz(x1_3, x1_3, sum_pair_3);
                    float sum_x_3 = sum_pair_3.x + sum_pair_3.y;
                    float _warp_reduce_30 = sum_x_3;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_30 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_30, offset);
                    float norm_3 = rsqrtf(_warp_reduce_30 + 1e-06f);
                    norm_3 = norm_3 * scale;
                    float2 norm2_3 = make_float2(norm_3, norm_3);
                    x0_3 = mul_f32x2(x0_3, norm2_3);
                    x1_3 = mul_f32x2(x1_3, norm2_3);
                    sQ[t * 128 + k_col0] = x0_3.x;
                    sQ[t * 128 + k_col0 + 1] = x0_3.y;
                    sQ[t * 128 + k_col0 + 2] = x1_3.x;
                    sQ[t * 128 + k_col0 + 3] = x1_3.y;
                    float a_log_v_1 = r_a[2];
                    asm("" : "+f"(a_log_v_1));
                    float xv_1 = r_a[0] + r_a[1];
                    float softplus_x_1 = xv_1;
                    if (xv_1 <= 20.0f) {
                        float _exp_32 = __expf(xv_1);
                        float _log2_31;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_31) : "f"(1.0f + _exp_32));
                        softplus_x_1 = _log2_31 * 0.6931472f;
                    }
                    float _exp_33 = __expf(a_log_v_1);
                    float g_log_1 = -_exp_33 * softplus_x_1;
                    float alpha_1 = __expf(g_log_1);
                    if (lane_local == 0) {
                        sAlpha[t] = alpha_1;
                        if (use_ssm_state_indices != 0) {
                            sSsm[t] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)t * ssm_state_indices_stride_p1];
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
#if __CUDA_ARCH__ == 900
            #pragma unroll 2
            for (int t = 0; t < 16; t++) {
                float alpha = sAlpha[t];
                S[0] = mul_f32_rn_ftz(S[0], alpha);
                S[1] = mul_f32_rn_ftz(S[1], alpha);
                S[2] = mul_f32_rn_ftz(S[2], alpha);
                S[3] = mul_f32_rn_ftz(S[3], alpha);
                S[4] = mul_f32_rn_ftz(S[4], alpha);
                S[5] = mul_f32_rn_ftz(S[5], alpha);
                S[6] = mul_f32_rn_ftz(S[6], alpha);
                S[7] = mul_f32_rn_ftz(S[7], alpha);
                S[8] = mul_f32_rn_ftz(S[8], alpha);
                S[9] = mul_f32_rn_ftz(S[9], alpha);
                S[10] = mul_f32_rn_ftz(S[10], alpha);
                S[11] = mul_f32_rn_ftz(S[11], alpha);
                S[12] = mul_f32_rn_ftz(S[12], alpha);
                S[13] = mul_f32_rn_ftz(S[13], alpha);
                S[14] = mul_f32_rn_ftz(S[14], alpha);
                S[15] = mul_f32_rn_ftz(S[15], alpha);
                S[16] = mul_f32_rn_ftz(S[16], alpha);
                S[17] = mul_f32_rn_ftz(S[17], alpha);
                S[18] = mul_f32_rn_ftz(S[18], alpha);
                S[19] = mul_f32_rn_ftz(S[19], alpha);
                S[20] = mul_f32_rn_ftz(S[20], alpha);
                S[21] = mul_f32_rn_ftz(S[21], alpha);
                S[22] = mul_f32_rn_ftz(S[22], alpha);
                S[23] = mul_f32_rn_ftz(S[23], alpha);
                S[24] = mul_f32_rn_ftz(S[24], alpha);
                S[25] = mul_f32_rn_ftz(S[25], alpha);
                S[26] = mul_f32_rn_ftz(S[26], alpha);
                S[27] = mul_f32_rn_ftz(S[27], alpha);
                S[28] = mul_f32_rn_ftz(S[28], alpha);
                S[29] = mul_f32_rn_ftz(S[29], alpha);
                S[30] = mul_f32_rn_ftz(S[30], alpha);
                S[31] = mul_f32_rn_ftz(S[31], alpha);
                int m_0 = t * 3;
#else
            #pragma unroll 1
            for (int t_1 = 0; t_1 < 16; t_1++) {
                float alpha_2 = sAlpha[t_1];
                float2 alpha2 = make_float2(alpha_2, alpha_2);
                S2[0] = mul_f32x2(S2[0], alpha2);
                S2[1] = mul_f32x2(S2[1], alpha2);
                S2[2] = mul_f32x2(S2[2], alpha2);
                S2[3] = mul_f32x2(S2[3], alpha2);
                S2[4] = mul_f32x2(S2[4], alpha2);
                S2[5] = mul_f32x2(S2[5], alpha2);
                S2[6] = mul_f32x2(S2[6], alpha2);
                S2[7] = mul_f32x2(S2[7], alpha2);
                S2[8] = mul_f32x2(S2[8], alpha2);
                S2[9] = mul_f32x2(S2[9], alpha2);
                S2[10] = mul_f32x2(S2[10], alpha2);
                S2[11] = mul_f32x2(S2[11], alpha2);
                S2[12] = mul_f32x2(S2[12], alpha2);
                S2[13] = mul_f32x2(S2[13], alpha2);
                S2[14] = mul_f32x2(S2[14], alpha2);
                S2[15] = mul_f32x2(S2[15], alpha2);
                int m_0 = t_1 * 3;
#endif
                unsigned int kbase_0 = sK_addr + (unsigned int)(m_0 * 512) + lane_byte;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(kbase_0));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(kbase_0 + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(kbase_0 + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(kbase_0 + 384));
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
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_0 * 256) + group_byte));
                float beta_0 = sBeta[m_0];
#if __CUDA_ARCH__ == 900
                float acc0_x = 0.0f;
                float acc0_y = 0.0f;
                float acc1_x = 0.0f;
                float acc1_y = 0.0f;
                float _fma_60;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x));
                acc0_x = _fma_60;
                float _fma_61;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y));
                acc0_y = _fma_61;
                float _fma_62;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x));
                acc1_x = _fma_62;
                float _fma_63;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y));
                acc1_y = _fma_63;
                float _fma_64;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x));
                acc0_x = _fma_64;
                float _fma_65;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y));
                acc0_y = _fma_65;
                float _fma_66;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x));
                acc1_x = _fma_66;
                float _fma_67;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y));
                acc1_y = _fma_67;
                float _fma_68;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x));
                acc0_x = _fma_68;
                float _fma_69;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y));
                acc0_y = _fma_69;
                float _fma_70;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x));
                acc1_x = _fma_70;
                float _fma_71;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y));
                acc1_y = _fma_71;
                float _fma_72;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x));
                acc0_x = _fma_72;
                float _fma_73;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y));
                acc0_y = _fma_73;
                float _fma_74;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x));
                acc1_x = _fma_74;
                float _fma_75;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y));
                acc1_y = _fma_75;
                float _add_76;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(acc0_x), "f"(acc1_x));
                float _add_77;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(acc0_y), "f"(acc1_y));
                float dot = _add_76 + _add_77;
                float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
                dot = dot + _shfl_xor_sync_78;
                float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
                dot = dot + _shfl_xor_sync_79;
                float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
                dot = dot + _shfl_xor_sync_80;
#else
                float2 acc0 = make_float2(0.0f, 0.0f);
                acc0 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0);
                float dot = acc0.x + acc0.y;
                float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
                dot = dot + _shfl_xor_sync_34;
                float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
                dot = dot + _shfl_xor_sync_35;
                float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
                dot = dot + _shfl_xor_sync_36;
#endif
                float vn_0 = (v_regs[0] - dot) * beta_0;
#if __CUDA_ARCH__ == 900
                float _fma_81;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(k_regs[0]), "f"(vn_0), "f"(S[0]));
                S[0] = _fma_81;
                float _fma_82;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(k_regs[1]), "f"(vn_0), "f"(S[1]));
                S[1] = _fma_82;
                float _fma_83;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(k_regs[2]), "f"(vn_0), "f"(S[2]));
                S[2] = _fma_83;
                float _fma_84;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(k_regs[3]), "f"(vn_0), "f"(S[3]));
                S[3] = _fma_84;
                float _fma_85;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(k_regs[4]), "f"(vn_0), "f"(S[4]));
                S[4] = _fma_85;
                float _fma_86;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(k_regs[5]), "f"(vn_0), "f"(S[5]));
                S[5] = _fma_86;
                float _fma_87;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(k_regs[6]), "f"(vn_0), "f"(S[6]));
                S[6] = _fma_87;
                float _fma_88;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(k_regs[7]), "f"(vn_0), "f"(S[7]));
                S[7] = _fma_88;
                float _fma_89;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(k_regs[8]), "f"(vn_0), "f"(S[8]));
                S[8] = _fma_89;
                float _fma_90;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(k_regs[9]), "f"(vn_0), "f"(S[9]));
                S[9] = _fma_90;
                float _fma_91;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(k_regs[10]), "f"(vn_0), "f"(S[10]));
                S[10] = _fma_91;
                float _fma_92;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(k_regs[11]), "f"(vn_0), "f"(S[11]));
                S[11] = _fma_92;
                float _fma_93;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(k_regs[12]), "f"(vn_0), "f"(S[12]));
                S[12] = _fma_93;
                float _fma_94;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(k_regs[13]), "f"(vn_0), "f"(S[13]));
                S[13] = _fma_94;
                float _fma_95;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(k_regs[14]), "f"(vn_0), "f"(S[14]));
                S[14] = _fma_95;
                float _fma_96;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(k_regs[15]), "f"(vn_0), "f"(S[15]));
                S[15] = _fma_96;
                float acc0_x_0 = 0.0f;
                float acc0_y_0 = 0.0f;
                float acc1_x_0 = 0.0f;
                float acc1_y_0 = 0.0f;
                float _fma_97;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_0));
                acc0_x_0 = _fma_97;
                float _fma_98;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_0));
                acc0_y_0 = _fma_98;
                float _fma_99;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_0));
                acc1_x_0 = _fma_99;
                float _fma_100;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_0));
                acc1_y_0 = _fma_100;
                float _fma_101;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_0));
                acc0_x_0 = _fma_101;
                float _fma_102;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_0));
                acc0_y_0 = _fma_102;
                float _fma_103;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_0));
                acc1_x_0 = _fma_103;
                float _fma_104;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_0));
                acc1_y_0 = _fma_104;
                float _fma_105;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_0));
                acc0_x_0 = _fma_105;
                float _fma_106;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_0));
                acc0_y_0 = _fma_106;
                float _fma_107;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_0));
                acc1_x_0 = _fma_107;
                float _fma_108;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_0));
                acc1_y_0 = _fma_108;
                float _fma_109;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_0));
                acc0_x_0 = _fma_109;
                float _fma_110;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_0));
                acc0_y_0 = _fma_110;
                float _fma_111;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_111) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_0));
                acc1_x_0 = _fma_111;
                float _fma_112;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_112) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_0));
                acc1_y_0 = _fma_112;
                float _add_113;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(acc0_x_0), "f"(acc1_x_0));
                float _add_114;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_114) : "f"(acc0_y_0), "f"(acc1_y_0));
                float dot_0 = _add_113 + _add_114;
                float _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 1);
                dot_0 = dot_0 + _shfl_xor_sync_115;
                float _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 2);
                dot_0 = dot_0 + _shfl_xor_sync_116;
                float _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 4);
                dot_0 = dot_0 + _shfl_xor_sync_117;
#else
                float2 vn2_0 = make_float2(vn_0, vn_0);
                S2[0] = fma_f32x2_rn_ftz(k2[0], vn2_0, S2[0]);
                S2[1] = fma_f32x2_rn_ftz(k2[1], vn2_0, S2[1]);
                S2[2] = fma_f32x2_rn_ftz(k2[2], vn2_0, S2[2]);
                S2[3] = fma_f32x2_rn_ftz(k2[3], vn2_0, S2[3]);
                S2[4] = fma_f32x2_rn_ftz(k2[4], vn2_0, S2[4]);
                S2[5] = fma_f32x2_rn_ftz(k2[5], vn2_0, S2[5]);
                S2[6] = fma_f32x2_rn_ftz(k2[6], vn2_0, S2[6]);
                S2[7] = fma_f32x2_rn_ftz(k2[7], vn2_0, S2[7]);
                float2 acc0_0 = make_float2(0.0f, 0.0f);
                acc0_0 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_0);
                float dot_0 = acc0_0.x + acc0_0.y;
                float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 1);
                dot_0 = dot_0 + _shfl_xor_sync_37;
                float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 2);
                dot_0 = dot_0 + _shfl_xor_sync_38;
                float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 4);
                dot_0 = dot_0 + _shfl_xor_sync_39;
#endif
                float vn_1 = (v_regs[1] - dot_0) * beta_0;
#if __CUDA_ARCH__ == 900
                float _fma_118;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_118) : "f"(k_regs[0]), "f"(vn_1), "f"(S[16]));
                S[16] = _fma_118;
                float _fma_119;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(k_regs[1]), "f"(vn_1), "f"(S[17]));
                S[17] = _fma_119;
                float _fma_120;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(k_regs[2]), "f"(vn_1), "f"(S[18]));
                S[18] = _fma_120;
                float _fma_121;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(k_regs[3]), "f"(vn_1), "f"(S[19]));
                S[19] = _fma_121;
                float _fma_122;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(k_regs[4]), "f"(vn_1), "f"(S[20]));
                S[20] = _fma_122;
                float _fma_123;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_123) : "f"(k_regs[5]), "f"(vn_1), "f"(S[21]));
                S[21] = _fma_123;
                float _fma_124;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(k_regs[6]), "f"(vn_1), "f"(S[22]));
                S[22] = _fma_124;
                float _fma_125;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(k_regs[7]), "f"(vn_1), "f"(S[23]));
                S[23] = _fma_125;
                float _fma_126;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(k_regs[8]), "f"(vn_1), "f"(S[24]));
                S[24] = _fma_126;
                float _fma_127;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(k_regs[9]), "f"(vn_1), "f"(S[25]));
                S[25] = _fma_127;
                float _fma_128;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_128) : "f"(k_regs[10]), "f"(vn_1), "f"(S[26]));
                S[26] = _fma_128;
                float _fma_129;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_129) : "f"(k_regs[11]), "f"(vn_1), "f"(S[27]));
                S[27] = _fma_129;
                float _fma_130;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_130) : "f"(k_regs[12]), "f"(vn_1), "f"(S[28]));
                S[28] = _fma_130;
                float _fma_131;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(k_regs[13]), "f"(vn_1), "f"(S[29]));
                S[29] = _fma_131;
                float _fma_132;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(k_regs[14]), "f"(vn_1), "f"(S[30]));
                S[30] = _fma_132;
                float _fma_133;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(k_regs[15]), "f"(vn_1), "f"(S[31]));
                S[31] = _fma_133;
                int m_1 = t * 3 + 1;
#else
                float2 vn2_1 = make_float2(vn_1, vn_1);
                S2[8] = fma_f32x2_rn_ftz(k2[0], vn2_1, S2[8]);
                S2[9] = fma_f32x2_rn_ftz(k2[1], vn2_1, S2[9]);
                S2[10] = fma_f32x2_rn_ftz(k2[2], vn2_1, S2[10]);
                S2[11] = fma_f32x2_rn_ftz(k2[3], vn2_1, S2[11]);
                S2[12] = fma_f32x2_rn_ftz(k2[4], vn2_1, S2[12]);
                S2[13] = fma_f32x2_rn_ftz(k2[5], vn2_1, S2[13]);
                S2[14] = fma_f32x2_rn_ftz(k2[6], vn2_1, S2[14]);
                S2[15] = fma_f32x2_rn_ftz(k2[7], vn2_1, S2[15]);
                int m_1 = t_1 * 3 + 1;
#endif
                unsigned int kbase_1 = sK_addr + (unsigned int)(m_1 * 512) + lane_byte;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(kbase_1));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(kbase_1 + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(kbase_1 + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(kbase_1 + 384));
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
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_1 * 256) + group_byte));
                float beta_1 = sBeta[m_1];
#if __CUDA_ARCH__ == 900
                float acc0_x_1 = 0.0f;
                float acc0_y_1 = 0.0f;
                float acc1_x_1 = 0.0f;
                float acc1_y_1 = 0.0f;
                float _fma_134;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_134) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_1));
                acc0_x_1 = _fma_134;
                float _fma_135;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_135) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_1));
                acc0_y_1 = _fma_135;
                float _fma_136;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_136) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_1));
                acc1_x_1 = _fma_136;
                float _fma_137;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_137) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_1));
                acc1_y_1 = _fma_137;
                float _fma_138;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_138) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_1));
                acc0_x_1 = _fma_138;
                float _fma_139;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_139) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_1));
                acc0_y_1 = _fma_139;
                float _fma_140;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_140) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_1));
                acc1_x_1 = _fma_140;
                float _fma_141;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_141) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_1));
                acc1_y_1 = _fma_141;
                float _fma_142;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_1));
                acc0_x_1 = _fma_142;
                float _fma_143;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_143) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_1));
                acc0_y_1 = _fma_143;
                float _fma_144;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_144) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_1));
                acc1_x_1 = _fma_144;
                float _fma_145;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_145) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_1));
                acc1_y_1 = _fma_145;
                float _fma_146;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_146) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_1));
                acc0_x_1 = _fma_146;
                float _fma_147;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_147) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_1));
                acc0_y_1 = _fma_147;
                float _fma_148;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_148) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_1));
                acc1_x_1 = _fma_148;
                float _fma_149;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_149) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_1));
                acc1_y_1 = _fma_149;
                float _add_150;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_150) : "f"(acc0_x_1), "f"(acc1_x_1));
                float _add_151;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_151) : "f"(acc0_y_1), "f"(acc1_y_1));
                float dot_1 = _add_150 + _add_151;
                float _shfl_xor_sync_152 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_152;
                float _shfl_xor_sync_153 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_153;
                float _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
                dot_1 = dot_1 + _shfl_xor_sync_154;
#else
                float2 acc0_1 = make_float2(0.0f, 0.0f);
                acc0_1 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_1);
                float dot_1 = acc0_1.x + acc0_1.y;
                float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_40;
                float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_41;
                float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
                dot_1 = dot_1 + _shfl_xor_sync_42;
#endif
                float vn_2 = (v_regs[0] - dot_1) * beta_1;
#if __CUDA_ARCH__ == 900
                float _fma_155;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(k_regs[0]), "f"(vn_2), "f"(S[0]));
                S[0] = _fma_155;
                float _fma_156;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(k_regs[1]), "f"(vn_2), "f"(S[1]));
                S[1] = _fma_156;
                float _fma_157;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(k_regs[2]), "f"(vn_2), "f"(S[2]));
                S[2] = _fma_157;
                float _fma_158;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_158) : "f"(k_regs[3]), "f"(vn_2), "f"(S[3]));
                S[3] = _fma_158;
                float _fma_159;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_159) : "f"(k_regs[4]), "f"(vn_2), "f"(S[4]));
                S[4] = _fma_159;
                float _fma_160;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_160) : "f"(k_regs[5]), "f"(vn_2), "f"(S[5]));
                S[5] = _fma_160;
                float _fma_161;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_161) : "f"(k_regs[6]), "f"(vn_2), "f"(S[6]));
                S[6] = _fma_161;
                float _fma_162;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_162) : "f"(k_regs[7]), "f"(vn_2), "f"(S[7]));
                S[7] = _fma_162;
                float _fma_163;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_163) : "f"(k_regs[8]), "f"(vn_2), "f"(S[8]));
                S[8] = _fma_163;
                float _fma_164;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_164) : "f"(k_regs[9]), "f"(vn_2), "f"(S[9]));
                S[9] = _fma_164;
                float _fma_165;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_165) : "f"(k_regs[10]), "f"(vn_2), "f"(S[10]));
                S[10] = _fma_165;
                float _fma_166;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_166) : "f"(k_regs[11]), "f"(vn_2), "f"(S[11]));
                S[11] = _fma_166;
                float _fma_167;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_167) : "f"(k_regs[12]), "f"(vn_2), "f"(S[12]));
                S[12] = _fma_167;
                float _fma_168;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_168) : "f"(k_regs[13]), "f"(vn_2), "f"(S[13]));
                S[13] = _fma_168;
                float _fma_169;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_169) : "f"(k_regs[14]), "f"(vn_2), "f"(S[14]));
                S[14] = _fma_169;
                float _fma_170;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_170) : "f"(k_regs[15]), "f"(vn_2), "f"(S[15]));
                S[15] = _fma_170;
                float acc0_x_2 = 0.0f;
                float acc0_y_2 = 0.0f;
                float acc1_x_2 = 0.0f;
                float acc1_y_2 = 0.0f;
                float _fma_171;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_171) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_2));
                acc0_x_2 = _fma_171;
                float _fma_172;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_172) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_2));
                acc0_y_2 = _fma_172;
                float _fma_173;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_173) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_2));
                acc1_x_2 = _fma_173;
                float _fma_174;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_174) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_2));
                acc1_y_2 = _fma_174;
                float _fma_175;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_175) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_2));
                acc0_x_2 = _fma_175;
                float _fma_176;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_176) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_2));
                acc0_y_2 = _fma_176;
                float _fma_177;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_177) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_2));
                acc1_x_2 = _fma_177;
                float _fma_178;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_178) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_2));
                acc1_y_2 = _fma_178;
                float _fma_179;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_179) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_2));
                acc0_x_2 = _fma_179;
                float _fma_180;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_180) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_2));
                acc0_y_2 = _fma_180;
                float _fma_181;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_181) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_2));
                acc1_x_2 = _fma_181;
                float _fma_182;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_182) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_2));
                acc1_y_2 = _fma_182;
                float _fma_183;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_183) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_2));
                acc0_x_2 = _fma_183;
                float _fma_184;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_184) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_2));
                acc0_y_2 = _fma_184;
                float _fma_185;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_185) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_2));
                acc1_x_2 = _fma_185;
                float _fma_186;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_186) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_2));
                acc1_y_2 = _fma_186;
                float _add_187;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_187) : "f"(acc0_x_2), "f"(acc1_x_2));
                float _add_188;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_188) : "f"(acc0_y_2), "f"(acc1_y_2));
                float dot_2 = _add_187 + _add_188;
                float _shfl_xor_sync_189 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 1);
                dot_2 = dot_2 + _shfl_xor_sync_189;
                float _shfl_xor_sync_190 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 2);
                dot_2 = dot_2 + _shfl_xor_sync_190;
                float _shfl_xor_sync_191 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 4);
                dot_2 = dot_2 + _shfl_xor_sync_191;
#else
                float2 vn2_2 = make_float2(vn_2, vn_2);
                S2[0] = fma_f32x2_rn_ftz(k2[0], vn2_2, S2[0]);
                S2[1] = fma_f32x2_rn_ftz(k2[1], vn2_2, S2[1]);
                S2[2] = fma_f32x2_rn_ftz(k2[2], vn2_2, S2[2]);
                S2[3] = fma_f32x2_rn_ftz(k2[3], vn2_2, S2[3]);
                S2[4] = fma_f32x2_rn_ftz(k2[4], vn2_2, S2[4]);
                S2[5] = fma_f32x2_rn_ftz(k2[5], vn2_2, S2[5]);
                S2[6] = fma_f32x2_rn_ftz(k2[6], vn2_2, S2[6]);
                S2[7] = fma_f32x2_rn_ftz(k2[7], vn2_2, S2[7]);
                float2 acc0_2 = make_float2(0.0f, 0.0f);
                acc0_2 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_2);
                float dot_2 = acc0_2.x + acc0_2.y;
                float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 1);
                dot_2 = dot_2 + _shfl_xor_sync_43;
                float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 2);
                dot_2 = dot_2 + _shfl_xor_sync_44;
                float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 4);
                dot_2 = dot_2 + _shfl_xor_sync_45;
#endif
                float vn_3 = (v_regs[1] - dot_2) * beta_1;
#if __CUDA_ARCH__ == 900
                float _fma_192;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_192) : "f"(k_regs[0]), "f"(vn_3), "f"(S[16]));
                S[16] = _fma_192;
                float _fma_193;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_193) : "f"(k_regs[1]), "f"(vn_3), "f"(S[17]));
                S[17] = _fma_193;
                float _fma_194;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_194) : "f"(k_regs[2]), "f"(vn_3), "f"(S[18]));
                S[18] = _fma_194;
                float _fma_195;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_195) : "f"(k_regs[3]), "f"(vn_3), "f"(S[19]));
                S[19] = _fma_195;
                float _fma_196;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_196) : "f"(k_regs[4]), "f"(vn_3), "f"(S[20]));
                S[20] = _fma_196;
                float _fma_197;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_197) : "f"(k_regs[5]), "f"(vn_3), "f"(S[21]));
                S[21] = _fma_197;
                float _fma_198;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_198) : "f"(k_regs[6]), "f"(vn_3), "f"(S[22]));
                S[22] = _fma_198;
                float _fma_199;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_199) : "f"(k_regs[7]), "f"(vn_3), "f"(S[23]));
                S[23] = _fma_199;
                float _fma_200;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_200) : "f"(k_regs[8]), "f"(vn_3), "f"(S[24]));
                S[24] = _fma_200;
                float _fma_201;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_201) : "f"(k_regs[9]), "f"(vn_3), "f"(S[25]));
                S[25] = _fma_201;
                float _fma_202;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_202) : "f"(k_regs[10]), "f"(vn_3), "f"(S[26]));
                S[26] = _fma_202;
                float _fma_203;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_203) : "f"(k_regs[11]), "f"(vn_3), "f"(S[27]));
                S[27] = _fma_203;
                float _fma_204;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_204) : "f"(k_regs[12]), "f"(vn_3), "f"(S[28]));
                S[28] = _fma_204;
                float _fma_205;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_205) : "f"(k_regs[13]), "f"(vn_3), "f"(S[29]));
                S[29] = _fma_205;
                float _fma_206;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_206) : "f"(k_regs[14]), "f"(vn_3), "f"(S[30]));
                S[30] = _fma_206;
                float _fma_207;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_207) : "f"(k_regs[15]), "f"(vn_3), "f"(S[31]));
                S[31] = _fma_207;
                int m_2 = t * 3 + 2;
#else
                float2 vn2_3 = make_float2(vn_3, vn_3);
                S2[8] = fma_f32x2_rn_ftz(k2[0], vn2_3, S2[8]);
                S2[9] = fma_f32x2_rn_ftz(k2[1], vn2_3, S2[9]);
                S2[10] = fma_f32x2_rn_ftz(k2[2], vn2_3, S2[10]);
                S2[11] = fma_f32x2_rn_ftz(k2[3], vn2_3, S2[11]);
                S2[12] = fma_f32x2_rn_ftz(k2[4], vn2_3, S2[12]);
                S2[13] = fma_f32x2_rn_ftz(k2[5], vn2_3, S2[13]);
                S2[14] = fma_f32x2_rn_ftz(k2[6], vn2_3, S2[14]);
                S2[15] = fma_f32x2_rn_ftz(k2[7], vn2_3, S2[15]);
                int m_2 = t_1 * 3 + 2;
#endif
                unsigned int kbase_2 = sK_addr + (unsigned int)(m_2 * 512) + lane_byte;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(kbase_2));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(kbase_2 + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(kbase_2 + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(kbase_2 + 384));
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
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_2 * 256) + group_byte));
                float beta_2 = sBeta[m_2];
#if __CUDA_ARCH__ == 900
                float acc0_x_3 = 0.0f;
                float acc0_y_3 = 0.0f;
                float acc1_x_3 = 0.0f;
                float acc1_y_3 = 0.0f;
                float _fma_208;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_208) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_3));
                acc0_x_3 = _fma_208;
                float _fma_209;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_209) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_3));
                acc0_y_3 = _fma_209;
                float _fma_210;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_210) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_3));
                acc1_x_3 = _fma_210;
                float _fma_211;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_211) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_3));
                acc1_y_3 = _fma_211;
                float _fma_212;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_212) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_3));
                acc0_x_3 = _fma_212;
                float _fma_213;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_213) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_3));
                acc0_y_3 = _fma_213;
                float _fma_214;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_214) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_3));
                acc1_x_3 = _fma_214;
                float _fma_215;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_215) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_3));
                acc1_y_3 = _fma_215;
                float _fma_216;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_216) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_3));
                acc0_x_3 = _fma_216;
                float _fma_217;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_217) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_3));
                acc0_y_3 = _fma_217;
                float _fma_218;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_218) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_3));
                acc1_x_3 = _fma_218;
                float _fma_219;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_219) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_3));
                acc1_y_3 = _fma_219;
                float _fma_220;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_220) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_3));
                acc0_x_3 = _fma_220;
                float _fma_221;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_221) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_3));
                acc0_y_3 = _fma_221;
                float _fma_222;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_222) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_3));
                acc1_x_3 = _fma_222;
                float _fma_223;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_223) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_3));
                acc1_y_3 = _fma_223;
                float _add_224;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_224) : "f"(acc0_x_3), "f"(acc1_x_3));
                float _add_225;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_225) : "f"(acc0_y_3), "f"(acc1_y_3));
                float dot_3 = _add_224 + _add_225;
                float _shfl_xor_sync_226 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 1);
                dot_3 = dot_3 + _shfl_xor_sync_226;
                float _shfl_xor_sync_227 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 2);
                dot_3 = dot_3 + _shfl_xor_sync_227;
                float _shfl_xor_sync_228 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 4);
                dot_3 = dot_3 + _shfl_xor_sync_228;
#else
                float2 acc0_3 = make_float2(0.0f, 0.0f);
                acc0_3 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_3);
                float dot_3 = acc0_3.x + acc0_3.y;
                float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 1);
                dot_3 = dot_3 + _shfl_xor_sync_46;
                float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 2);
                dot_3 = dot_3 + _shfl_xor_sync_47;
                float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 4);
                dot_3 = dot_3 + _shfl_xor_sync_48;
#endif
                float vn_4 = (v_regs[0] - dot_3) * beta_2;
#if __CUDA_ARCH__ == 900
                float _fma_229;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_229) : "f"(k_regs[0]), "f"(vn_4), "f"(S[0]));
                S[0] = _fma_229;
                float _fma_230;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_230) : "f"(k_regs[1]), "f"(vn_4), "f"(S[1]));
                S[1] = _fma_230;
                float _fma_231;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_231) : "f"(k_regs[2]), "f"(vn_4), "f"(S[2]));
                S[2] = _fma_231;
                float _fma_232;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_232) : "f"(k_regs[3]), "f"(vn_4), "f"(S[3]));
                S[3] = _fma_232;
                float _fma_233;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_233) : "f"(k_regs[4]), "f"(vn_4), "f"(S[4]));
                S[4] = _fma_233;
                float _fma_234;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_234) : "f"(k_regs[5]), "f"(vn_4), "f"(S[5]));
                S[5] = _fma_234;
                float _fma_235;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_235) : "f"(k_regs[6]), "f"(vn_4), "f"(S[6]));
                S[6] = _fma_235;
                float _fma_236;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_236) : "f"(k_regs[7]), "f"(vn_4), "f"(S[7]));
                S[7] = _fma_236;
                float _fma_237;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_237) : "f"(k_regs[8]), "f"(vn_4), "f"(S[8]));
                S[8] = _fma_237;
                float _fma_238;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_238) : "f"(k_regs[9]), "f"(vn_4), "f"(S[9]));
                S[9] = _fma_238;
                float _fma_239;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_239) : "f"(k_regs[10]), "f"(vn_4), "f"(S[10]));
                S[10] = _fma_239;
                float _fma_240;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_240) : "f"(k_regs[11]), "f"(vn_4), "f"(S[11]));
                S[11] = _fma_240;
                float _fma_241;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_241) : "f"(k_regs[12]), "f"(vn_4), "f"(S[12]));
                S[12] = _fma_241;
                float _fma_242;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_242) : "f"(k_regs[13]), "f"(vn_4), "f"(S[13]));
                S[13] = _fma_242;
                float _fma_243;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_243) : "f"(k_regs[14]), "f"(vn_4), "f"(S[14]));
                S[14] = _fma_243;
                float _fma_244;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_244) : "f"(k_regs[15]), "f"(vn_4), "f"(S[15]));
                S[15] = _fma_244;
                float acc0_x_4 = 0.0f;
                float acc0_y_4 = 0.0f;
                float acc1_x_4 = 0.0f;
                float acc1_y_4 = 0.0f;
                float _fma_245;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_245) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_4));
                acc0_x_4 = _fma_245;
                float _fma_246;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_246) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_4));
                acc0_y_4 = _fma_246;
                float _fma_247;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_247) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_4));
                acc1_x_4 = _fma_247;
                float _fma_248;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_248) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_4));
                acc1_y_4 = _fma_248;
                float _fma_249;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_249) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_4));
                acc0_x_4 = _fma_249;
                float _fma_250;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_250) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_4));
                acc0_y_4 = _fma_250;
                float _fma_251;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_251) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_4));
                acc1_x_4 = _fma_251;
                float _fma_252;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_252) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_4));
                acc1_y_4 = _fma_252;
                float _fma_253;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_253) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_4));
                acc0_x_4 = _fma_253;
                float _fma_254;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_254) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_4));
                acc0_y_4 = _fma_254;
                float _fma_255;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_255) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_4));
                acc1_x_4 = _fma_255;
                float _fma_256;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_256) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_4));
                acc1_y_4 = _fma_256;
                float _fma_257;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_257) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_4));
                acc0_x_4 = _fma_257;
                float _fma_258;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_258) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_4));
                acc0_y_4 = _fma_258;
                float _fma_259;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_259) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_4));
                acc1_x_4 = _fma_259;
                float _fma_260;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_260) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_4));
                acc1_y_4 = _fma_260;
                float _add_261;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_261) : "f"(acc0_x_4), "f"(acc1_x_4));
                float _add_262;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_262) : "f"(acc0_y_4), "f"(acc1_y_4));
                float dot_4 = _add_261 + _add_262;
                float _shfl_xor_sync_263 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 1);
                dot_4 = dot_4 + _shfl_xor_sync_263;
                float _shfl_xor_sync_264 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 2);
                dot_4 = dot_4 + _shfl_xor_sync_264;
                float _shfl_xor_sync_265 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 4);
                dot_4 = dot_4 + _shfl_xor_sync_265;
#else
                float2 vn2_4 = make_float2(vn_4, vn_4);
                S2[0] = fma_f32x2_rn_ftz(k2[0], vn2_4, S2[0]);
                S2[1] = fma_f32x2_rn_ftz(k2[1], vn2_4, S2[1]);
                S2[2] = fma_f32x2_rn_ftz(k2[2], vn2_4, S2[2]);
                S2[3] = fma_f32x2_rn_ftz(k2[3], vn2_4, S2[3]);
                S2[4] = fma_f32x2_rn_ftz(k2[4], vn2_4, S2[4]);
                S2[5] = fma_f32x2_rn_ftz(k2[5], vn2_4, S2[5]);
                S2[6] = fma_f32x2_rn_ftz(k2[6], vn2_4, S2[6]);
                S2[7] = fma_f32x2_rn_ftz(k2[7], vn2_4, S2[7]);
                float2 acc0_4 = make_float2(0.0f, 0.0f);
                acc0_4 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_4);
                float dot_4 = acc0_4.x + acc0_4.y;
                float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 1);
                dot_4 = dot_4 + _shfl_xor_sync_49;
                float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 2);
                dot_4 = dot_4 + _shfl_xor_sync_50;
                float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 4);
                dot_4 = dot_4 + _shfl_xor_sync_51;
#endif
                float vn_5 = (v_regs[1] - dot_4) * beta_2;
#if __CUDA_ARCH__ == 900
                float _fma_266;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_266) : "f"(k_regs[0]), "f"(vn_5), "f"(S[16]));
                S[16] = _fma_266;
                float _fma_267;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_267) : "f"(k_regs[1]), "f"(vn_5), "f"(S[17]));
                S[17] = _fma_267;
                float _fma_268;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_268) : "f"(k_regs[2]), "f"(vn_5), "f"(S[18]));
                S[18] = _fma_268;
                float _fma_269;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_269) : "f"(k_regs[3]), "f"(vn_5), "f"(S[19]));
                S[19] = _fma_269;
                float _fma_270;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_270) : "f"(k_regs[4]), "f"(vn_5), "f"(S[20]));
                S[20] = _fma_270;
                float _fma_271;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_271) : "f"(k_regs[5]), "f"(vn_5), "f"(S[21]));
                S[21] = _fma_271;
                float _fma_272;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_272) : "f"(k_regs[6]), "f"(vn_5), "f"(S[22]));
                S[22] = _fma_272;
                float _fma_273;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_273) : "f"(k_regs[7]), "f"(vn_5), "f"(S[23]));
                S[23] = _fma_273;
                float _fma_274;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_274) : "f"(k_regs[8]), "f"(vn_5), "f"(S[24]));
                S[24] = _fma_274;
                float _fma_275;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_275) : "f"(k_regs[9]), "f"(vn_5), "f"(S[25]));
                S[25] = _fma_275;
                float _fma_276;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_276) : "f"(k_regs[10]), "f"(vn_5), "f"(S[26]));
                S[26] = _fma_276;
                float _fma_277;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_277) : "f"(k_regs[11]), "f"(vn_5), "f"(S[27]));
                S[27] = _fma_277;
                float _fma_278;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_278) : "f"(k_regs[12]), "f"(vn_5), "f"(S[28]));
                S[28] = _fma_278;
                float _fma_279;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_279) : "f"(k_regs[13]), "f"(vn_5), "f"(S[29]));
                S[29] = _fma_279;
                float _fma_280;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_280) : "f"(k_regs[14]), "f"(vn_5), "f"(S[30]));
                S[30] = _fma_280;
                float _fma_281;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_281) : "f"(k_regs[15]), "f"(vn_5), "f"(S[31]));
                S[31] = _fma_281;
                unsigned int qbase = sQ_addr + (unsigned int)(t * 512) + lane_byte;
#else
                float2 vn2_5 = make_float2(vn_5, vn_5);
                S2[8] = fma_f32x2_rn_ftz(k2[0], vn2_5, S2[8]);
                S2[9] = fma_f32x2_rn_ftz(k2[1], vn2_5, S2[9]);
                S2[10] = fma_f32x2_rn_ftz(k2[2], vn2_5, S2[10]);
                S2[11] = fma_f32x2_rn_ftz(k2[3], vn2_5, S2[11]);
                S2[12] = fma_f32x2_rn_ftz(k2[4], vn2_5, S2[12]);
                S2[13] = fma_f32x2_rn_ftz(k2[5], vn2_5, S2[13]);
                S2[14] = fma_f32x2_rn_ftz(k2[6], vn2_5, S2[14]);
                S2[15] = fma_f32x2_rn_ftz(k2[7], vn2_5, S2[15]);
                unsigned int qbase = sQ_addr + (unsigned int)(t_1 * 512) + lane_byte;
#endif
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(qbase));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(qbase + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(qbase + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(qbase + 384));
#if __CUDA_ARCH__ == 900
                float acc0_x_5 = 0.0f;
                float acc0_y_5 = 0.0f;
                float acc1_x_5 = 0.0f;
                float acc1_y_5 = 0.0f;
                float _fma_282;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_282) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_5));
                acc0_x_5 = _fma_282;
                float _fma_283;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_283) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_5));
                acc0_y_5 = _fma_283;
                float _fma_284;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_284) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_5));
                acc1_x_5 = _fma_284;
                float _fma_285;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_285) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_5));
                acc1_y_5 = _fma_285;
                float _fma_286;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_286) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_5));
                acc0_x_5 = _fma_286;
                float _fma_287;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_287) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_5));
                acc0_y_5 = _fma_287;
                float _fma_288;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_288) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_5));
                acc1_x_5 = _fma_288;
                float _fma_289;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_289) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_5));
                acc1_y_5 = _fma_289;
                float _fma_290;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_290) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_5));
                acc0_x_5 = _fma_290;
                float _fma_291;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_291) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_5));
                acc0_y_5 = _fma_291;
                float _fma_292;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_292) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_5));
                acc1_x_5 = _fma_292;
                float _fma_293;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_293) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_5));
                acc1_y_5 = _fma_293;
                float _fma_294;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_294) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_5));
                acc0_x_5 = _fma_294;
                float _fma_295;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_295) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_5));
                acc0_y_5 = _fma_295;
                float _fma_296;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_296) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_5));
                acc1_x_5 = _fma_296;
                float _fma_297;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_297) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_5));
                acc1_y_5 = _fma_297;
                float _add_298;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_298) : "f"(acc0_x_5), "f"(acc1_x_5));
                float _add_299;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_299) : "f"(acc0_y_5), "f"(acc1_y_5));
                float dot_5 = _add_298 + _add_299;
                float _shfl_xor_sync_300 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 1);
                dot_5 = dot_5 + _shfl_xor_sync_300;
                float _shfl_xor_sync_301 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 2);
                dot_5 = dot_5 + _shfl_xor_sync_301;
                float _shfl_xor_sync_302 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 4);
                dot_5 = dot_5 + _shfl_xor_sync_302;
#else
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                float2 acc0_5 = make_float2(0.0f, 0.0f);
                acc0_5 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_5);
                float dot_5 = acc0_5.x + acc0_5.y;
                float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 1);
                dot_5 = dot_5 + _shfl_xor_sync_52;
                float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 2);
                dot_5 = dot_5 + _shfl_xor_sync_53;
                float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 4);
                dot_5 = dot_5 + _shfl_xor_sync_54;
#endif
                o4[0] = dot_5;
#if __CUDA_ARCH__ == 900
                float acc0_x_6 = 0.0f;
                float acc0_y_6 = 0.0f;
                float acc1_x_6 = 0.0f;
                float acc1_y_6 = 0.0f;
                float _fma_303;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_303) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_6));
                acc0_x_6 = _fma_303;
                float _fma_304;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_304) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_6));
                acc0_y_6 = _fma_304;
                float _fma_305;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_305) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_6));
                acc1_x_6 = _fma_305;
                float _fma_306;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_306) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_6));
                acc1_y_6 = _fma_306;
                float _fma_307;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_307) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_6));
                acc0_x_6 = _fma_307;
                float _fma_308;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_308) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_6));
                acc0_y_6 = _fma_308;
                float _fma_309;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_309) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_6));
                acc1_x_6 = _fma_309;
                float _fma_310;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_310) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_6));
                acc1_y_6 = _fma_310;
                float _fma_311;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_311) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_6));
                acc0_x_6 = _fma_311;
                float _fma_312;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_312) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_6));
                acc0_y_6 = _fma_312;
                float _fma_313;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_313) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_6));
                acc1_x_6 = _fma_313;
                float _fma_314;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_314) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_6));
                acc1_y_6 = _fma_314;
                float _fma_315;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_315) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_6));
                acc0_x_6 = _fma_315;
                float _fma_316;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_316) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_6));
                acc0_y_6 = _fma_316;
                float _fma_317;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_317) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_6));
                acc1_x_6 = _fma_317;
                float _fma_318;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_318) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_6));
                acc1_y_6 = _fma_318;
                float _add_319;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_319) : "f"(acc0_x_6), "f"(acc1_x_6));
                float _add_320;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_320) : "f"(acc0_y_6), "f"(acc1_y_6));
                float dot_6 = _add_319 + _add_320;
                float _shfl_xor_sync_321 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 1);
                dot_6 = dot_6 + _shfl_xor_sync_321;
                float _shfl_xor_sync_322 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 2);
                dot_6 = dot_6 + _shfl_xor_sync_322;
                float _shfl_xor_sync_323 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 4);
                dot_6 = dot_6 + _shfl_xor_sync_323;
#else
                float2 acc0_6 = make_float2(0.0f, 0.0f);
                acc0_6 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_6);
                float dot_6 = acc0_6.x + acc0_6.y;
                float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 1);
                dot_6 = dot_6 + _shfl_xor_sync_55;
                float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 2);
                dot_6 = dot_6 + _shfl_xor_sync_56;
                float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 4);
                dot_6 = dot_6 + _shfl_xor_sync_57;
#endif
                o4[1] = dot_6;
                if (u == 0) {
#if __CUDA_ARCH__ == 900
                    long long out_index = (long long)n * out_stride_p0 + (long long)t * out_stride_p1 + (long long)h * out_stride_p2;
#else
                    long long out_index = (long long)n * out_stride_p0 + (long long)t_1 * out_stride_p1 + (long long)h * out_stride_p2;
#endif
                    unsigned int packed[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o4[_lp*2 + 0], o4[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(out + (out_index + (long long)row0))[0] = reinterpret_cast<unsigned*>(packed)[0];
                }
#if __CUDA_ARCH__ == 900
                int ssm_slot = sSsm[t];
#else
                int ssm_slot = sSsm[t_1];
#endif
                int snap_valid = ((use_ssm_state_indices != 0 && ssm_slot >= 0 && state_pool_size > (long long)ssm_slot) ? 1 : 0);
                int safe_slot = ((snap_valid != 0) ? ssm_slot : 0);
                long long snap_base = (long long)safe_slot * state_stride_p0 + head_rows_off;
#if __CUDA_ARCH__ == 900
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base])), "f"(S[0]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 1ll])), "f"(S[1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll])), "f"(S[2]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll + 1ll])), "f"(S[3]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 32ll])), "f"(S[4]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 32ll + 1ll])), "f"(S[5]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 34ll])), "f"(S[6]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 34ll + 1ll])), "f"(S[7]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 64ll])), "f"(S[8]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 64ll + 1ll])), "f"(S[9]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 66ll])), "f"(S[10]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 66ll + 1ll])), "f"(S[11]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 96ll])), "f"(S[12]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 96ll + 1ll])), "f"(S[13]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 98ll])), "f"(S[14]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 98ll + 1ll])), "f"(S[15]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2])), "f"(S[16]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 1ll])), "f"(S[17]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 2ll])), "f"(S[18]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 2ll + 1ll])), "f"(S[19]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 32ll])), "f"(S[20]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 32ll + 1ll])), "f"(S[21]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 34ll])), "f"(S[22]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 34ll + 1ll])), "f"(S[23]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 64ll])), "f"(S[24]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 64ll + 1ll])), "f"(S[25]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 66ll])), "f"(S[26]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 66ll + 1ll])), "f"(S[27]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 96ll])), "f"(S[28]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 96ll + 1ll])), "f"(S[29]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 98ll])), "f"(S[30]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 98ll + 1ll])), "f"(S[31]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#else
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base])), "f"(S2[0].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 1ll])), "f"(S2[0].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll])), "f"(S2[1].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll + 1ll])), "f"(S2[1].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 32ll])), "f"(S2[2].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 32ll + 1ll])), "f"(S2[2].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 34ll])), "f"(S2[3].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 34ll + 1ll])), "f"(S2[3].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 64ll])), "f"(S2[4].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 64ll + 1ll])), "f"(S2[4].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 66ll])), "f"(S2[5].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 66ll + 1ll])), "f"(S2[5].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 96ll])), "f"(S2[6].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 96ll + 1ll])), "f"(S2[6].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 98ll])), "f"(S2[7].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 98ll + 1ll])), "f"(S2[7].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2])), "f"(S2[8].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 1ll])), "f"(S2[8].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 2ll])), "f"(S2[9].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 2ll + 1ll])), "f"(S2[9].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 32ll])), "f"(S2[10].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 32ll + 1ll])), "f"(S2[10].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 34ll])), "f"(S2[11].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 34ll + 1ll])), "f"(S2[11].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 64ll])), "f"(S2[12].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 64ll + 1ll])), "f"(S2[12].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 66ll])), "f"(S2[13].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 66ll + 1ll])), "f"(S2[13].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 96ll])), "f"(S2[14].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 96ll + 1ll])), "f"(S2[14].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 98ll])), "f"(S2[15].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + state_stride_p2 + 98ll + 1ll])), "f"(S2[15].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
#endif
            }
        }
        if (use_ssm_state_indices == 0) {
#if __CUDA_ARCH__ == 900
            #pragma unroll 2
            for (int t_1 = 0; t_1 < 16; t_1++) {
                float alpha_1 = sAlpha[t_1];
                S[0] = mul_f32_rn_ftz(S[0], alpha_1);
                S[1] = mul_f32_rn_ftz(S[1], alpha_1);
                S[2] = mul_f32_rn_ftz(S[2], alpha_1);
                S[3] = mul_f32_rn_ftz(S[3], alpha_1);
                S[4] = mul_f32_rn_ftz(S[4], alpha_1);
                S[5] = mul_f32_rn_ftz(S[5], alpha_1);
                S[6] = mul_f32_rn_ftz(S[6], alpha_1);
                S[7] = mul_f32_rn_ftz(S[7], alpha_1);
                S[8] = mul_f32_rn_ftz(S[8], alpha_1);
                S[9] = mul_f32_rn_ftz(S[9], alpha_1);
                S[10] = mul_f32_rn_ftz(S[10], alpha_1);
                S[11] = mul_f32_rn_ftz(S[11], alpha_1);
                S[12] = mul_f32_rn_ftz(S[12], alpha_1);
                S[13] = mul_f32_rn_ftz(S[13], alpha_1);
                S[14] = mul_f32_rn_ftz(S[14], alpha_1);
                S[15] = mul_f32_rn_ftz(S[15], alpha_1);
                S[16] = mul_f32_rn_ftz(S[16], alpha_1);
                S[17] = mul_f32_rn_ftz(S[17], alpha_1);
                S[18] = mul_f32_rn_ftz(S[18], alpha_1);
                S[19] = mul_f32_rn_ftz(S[19], alpha_1);
                S[20] = mul_f32_rn_ftz(S[20], alpha_1);
                S[21] = mul_f32_rn_ftz(S[21], alpha_1);
                S[22] = mul_f32_rn_ftz(S[22], alpha_1);
                S[23] = mul_f32_rn_ftz(S[23], alpha_1);
                S[24] = mul_f32_rn_ftz(S[24], alpha_1);
                S[25] = mul_f32_rn_ftz(S[25], alpha_1);
                S[26] = mul_f32_rn_ftz(S[26], alpha_1);
                S[27] = mul_f32_rn_ftz(S[27], alpha_1);
                S[28] = mul_f32_rn_ftz(S[28], alpha_1);
                S[29] = mul_f32_rn_ftz(S[29], alpha_1);
                S[30] = mul_f32_rn_ftz(S[30], alpha_1);
                S[31] = mul_f32_rn_ftz(S[31], alpha_1);
                int m_3 = t_1 * 3;
#else
            #pragma unroll 1
            for (int t_2 = 0; t_2 < 16; t_2++) {
                float alpha_3 = sAlpha[t_2];
                float2 alpha2_1 = make_float2(alpha_3, alpha_3);
                S2[0] = mul_f32x2(S2[0], alpha2_1);
                S2[1] = mul_f32x2(S2[1], alpha2_1);
                S2[2] = mul_f32x2(S2[2], alpha2_1);
                S2[3] = mul_f32x2(S2[3], alpha2_1);
                S2[4] = mul_f32x2(S2[4], alpha2_1);
                S2[5] = mul_f32x2(S2[5], alpha2_1);
                S2[6] = mul_f32x2(S2[6], alpha2_1);
                S2[7] = mul_f32x2(S2[7], alpha2_1);
                S2[8] = mul_f32x2(S2[8], alpha2_1);
                S2[9] = mul_f32x2(S2[9], alpha2_1);
                S2[10] = mul_f32x2(S2[10], alpha2_1);
                S2[11] = mul_f32x2(S2[11], alpha2_1);
                S2[12] = mul_f32x2(S2[12], alpha2_1);
                S2[13] = mul_f32x2(S2[13], alpha2_1);
                S2[14] = mul_f32x2(S2[14], alpha2_1);
                S2[15] = mul_f32x2(S2[15], alpha2_1);
                int m_3 = t_2 * 3;
#endif
                unsigned int kbase_3 = sK_addr + (unsigned int)(m_3 * 512) + lane_byte;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(kbase_3));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(kbase_3 + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(kbase_3 + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(kbase_3 + 384));
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
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_3 * 256) + group_byte));
                float beta_3 = sBeta[m_3];
#if __CUDA_ARCH__ == 900
                float acc0_x_7 = 0.0f;
                float acc0_y_7 = 0.0f;
                float acc1_x_7 = 0.0f;
                float acc1_y_7 = 0.0f;
                float _fma_324;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_324) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_7));
                acc0_x_7 = _fma_324;
                float _fma_325;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_325) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_7));
                acc0_y_7 = _fma_325;
                float _fma_326;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_326) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_7));
                acc1_x_7 = _fma_326;
                float _fma_327;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_327) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_7));
                acc1_y_7 = _fma_327;
                float _fma_328;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_328) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_7));
                acc0_x_7 = _fma_328;
                float _fma_329;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_329) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_7));
                acc0_y_7 = _fma_329;
                float _fma_330;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_330) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_7));
                acc1_x_7 = _fma_330;
                float _fma_331;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_331) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_7));
                acc1_y_7 = _fma_331;
                float _fma_332;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_332) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_7));
                acc0_x_7 = _fma_332;
                float _fma_333;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_333) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_7));
                acc0_y_7 = _fma_333;
                float _fma_334;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_334) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_7));
                acc1_x_7 = _fma_334;
                float _fma_335;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_335) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_7));
                acc1_y_7 = _fma_335;
                float _fma_336;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_336) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_7));
                acc0_x_7 = _fma_336;
                float _fma_337;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_337) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_7));
                acc0_y_7 = _fma_337;
                float _fma_338;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_338) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_7));
                acc1_x_7 = _fma_338;
                float _fma_339;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_339) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_7));
                acc1_y_7 = _fma_339;
                float _add_340;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_340) : "f"(acc0_x_7), "f"(acc1_x_7));
                float _add_341;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_341) : "f"(acc0_y_7), "f"(acc1_y_7));
                float dot_7 = _add_340 + _add_341;
                float _shfl_xor_sync_342 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 1);
                dot_7 = dot_7 + _shfl_xor_sync_342;
                float _shfl_xor_sync_343 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 2);
                dot_7 = dot_7 + _shfl_xor_sync_343;
                float _shfl_xor_sync_344 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 4);
                dot_7 = dot_7 + _shfl_xor_sync_344;
#else
                float2 acc0_7 = make_float2(0.0f, 0.0f);
                acc0_7 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_7);
                float dot_7 = acc0_7.x + acc0_7.y;
                float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 1);
                dot_7 = dot_7 + _shfl_xor_sync_58;
                float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 2);
                dot_7 = dot_7 + _shfl_xor_sync_59;
                float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 4);
                dot_7 = dot_7 + _shfl_xor_sync_60;
#endif
                float vn_6 = (v_regs[0] - dot_7) * beta_3;
#if __CUDA_ARCH__ == 900
                float _fma_345;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_345) : "f"(k_regs[0]), "f"(vn_6), "f"(S[0]));
                S[0] = _fma_345;
                float _fma_346;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_346) : "f"(k_regs[1]), "f"(vn_6), "f"(S[1]));
                S[1] = _fma_346;
                float _fma_347;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_347) : "f"(k_regs[2]), "f"(vn_6), "f"(S[2]));
                S[2] = _fma_347;
                float _fma_348;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_348) : "f"(k_regs[3]), "f"(vn_6), "f"(S[3]));
                S[3] = _fma_348;
                float _fma_349;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_349) : "f"(k_regs[4]), "f"(vn_6), "f"(S[4]));
                S[4] = _fma_349;
                float _fma_350;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_350) : "f"(k_regs[5]), "f"(vn_6), "f"(S[5]));
                S[5] = _fma_350;
                float _fma_351;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_351) : "f"(k_regs[6]), "f"(vn_6), "f"(S[6]));
                S[6] = _fma_351;
                float _fma_352;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_352) : "f"(k_regs[7]), "f"(vn_6), "f"(S[7]));
                S[7] = _fma_352;
                float _fma_353;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_353) : "f"(k_regs[8]), "f"(vn_6), "f"(S[8]));
                S[8] = _fma_353;
                float _fma_354;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_354) : "f"(k_regs[9]), "f"(vn_6), "f"(S[9]));
                S[9] = _fma_354;
                float _fma_355;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_355) : "f"(k_regs[10]), "f"(vn_6), "f"(S[10]));
                S[10] = _fma_355;
                float _fma_356;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_356) : "f"(k_regs[11]), "f"(vn_6), "f"(S[11]));
                S[11] = _fma_356;
                float _fma_357;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_357) : "f"(k_regs[12]), "f"(vn_6), "f"(S[12]));
                S[12] = _fma_357;
                float _fma_358;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_358) : "f"(k_regs[13]), "f"(vn_6), "f"(S[13]));
                S[13] = _fma_358;
                float _fma_359;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_359) : "f"(k_regs[14]), "f"(vn_6), "f"(S[14]));
                S[14] = _fma_359;
                float _fma_360;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_360) : "f"(k_regs[15]), "f"(vn_6), "f"(S[15]));
                S[15] = _fma_360;
                float acc0_x_7_1 = 0.0f;
                float acc0_y_7_1 = 0.0f;
                float acc1_x_7_1 = 0.0f;
                float acc1_y_7_1 = 0.0f;
                float _fma_361;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_361) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_361;
                float _fma_362;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_362) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_362;
                float _fma_363;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_363) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_363;
                float _fma_364;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_364) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_364;
                float _fma_365;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_365) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_365;
                float _fma_366;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_366) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_366;
                float _fma_367;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_367) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_367;
                float _fma_368;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_368) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_368;
                float _fma_369;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_369) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_369;
                float _fma_370;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_370) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_370;
                float _fma_371;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_371) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_371;
                float _fma_372;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_372) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_372;
                float _fma_373;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_373) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_373;
                float _fma_374;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_374) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_374;
                float _fma_375;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_375) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_375;
                float _fma_376;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_376) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_376;
                float _add_377;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_377) : "f"(acc0_x_7_1), "f"(acc1_x_7_1));
                float _add_378;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_378) : "f"(acc0_y_7_1), "f"(acc1_y_7_1));
                float dot_7_1 = _add_377 + _add_378;
                float _shfl_xor_sync_379 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 1);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_379;
                float _shfl_xor_sync_380 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 2);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_380;
                float _shfl_xor_sync_381 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 4);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_381;
#else
                float2 vn2_6 = make_float2(vn_6, vn_6);
                S2[0] = fma_f32x2_rn_ftz(k2[0], vn2_6, S2[0]);
                S2[1] = fma_f32x2_rn_ftz(k2[1], vn2_6, S2[1]);
                S2[2] = fma_f32x2_rn_ftz(k2[2], vn2_6, S2[2]);
                S2[3] = fma_f32x2_rn_ftz(k2[3], vn2_6, S2[3]);
                S2[4] = fma_f32x2_rn_ftz(k2[4], vn2_6, S2[4]);
                S2[5] = fma_f32x2_rn_ftz(k2[5], vn2_6, S2[5]);
                S2[6] = fma_f32x2_rn_ftz(k2[6], vn2_6, S2[6]);
                S2[7] = fma_f32x2_rn_ftz(k2[7], vn2_6, S2[7]);
                float2 acc0_7_1 = make_float2(0.0f, 0.0f);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_7_1);
                float dot_7_1 = acc0_7_1.x + acc0_7_1.y;
                float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 1);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_61;
                float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 2);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_62;
                float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 4);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_63;
#endif
                float vn_7 = (v_regs[1] - dot_7_1) * beta_3;
#if __CUDA_ARCH__ == 900
                float _fma_382;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_382) : "f"(k_regs[0]), "f"(vn_7), "f"(S[16]));
                S[16] = _fma_382;
                float _fma_383;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_383) : "f"(k_regs[1]), "f"(vn_7), "f"(S[17]));
                S[17] = _fma_383;
                float _fma_384;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_384) : "f"(k_regs[2]), "f"(vn_7), "f"(S[18]));
                S[18] = _fma_384;
                float _fma_385;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_385) : "f"(k_regs[3]), "f"(vn_7), "f"(S[19]));
                S[19] = _fma_385;
                float _fma_386;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_386) : "f"(k_regs[4]), "f"(vn_7), "f"(S[20]));
                S[20] = _fma_386;
                float _fma_387;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_387) : "f"(k_regs[5]), "f"(vn_7), "f"(S[21]));
                S[21] = _fma_387;
                float _fma_388;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_388) : "f"(k_regs[6]), "f"(vn_7), "f"(S[22]));
                S[22] = _fma_388;
                float _fma_389;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_389) : "f"(k_regs[7]), "f"(vn_7), "f"(S[23]));
                S[23] = _fma_389;
                float _fma_390;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_390) : "f"(k_regs[8]), "f"(vn_7), "f"(S[24]));
                S[24] = _fma_390;
                float _fma_391;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_391) : "f"(k_regs[9]), "f"(vn_7), "f"(S[25]));
                S[25] = _fma_391;
                float _fma_392;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_392) : "f"(k_regs[10]), "f"(vn_7), "f"(S[26]));
                S[26] = _fma_392;
                float _fma_393;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_393) : "f"(k_regs[11]), "f"(vn_7), "f"(S[27]));
                S[27] = _fma_393;
                float _fma_394;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_394) : "f"(k_regs[12]), "f"(vn_7), "f"(S[28]));
                S[28] = _fma_394;
                float _fma_395;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_395) : "f"(k_regs[13]), "f"(vn_7), "f"(S[29]));
                S[29] = _fma_395;
                float _fma_396;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_396) : "f"(k_regs[14]), "f"(vn_7), "f"(S[30]));
                S[30] = _fma_396;
                float _fma_397;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_397) : "f"(k_regs[15]), "f"(vn_7), "f"(S[31]));
                S[31] = _fma_397;
                int m_4 = t_1 * 3 + 1;
#else
                float2 vn2_7 = make_float2(vn_7, vn_7);
                S2[8] = fma_f32x2_rn_ftz(k2[0], vn2_7, S2[8]);
                S2[9] = fma_f32x2_rn_ftz(k2[1], vn2_7, S2[9]);
                S2[10] = fma_f32x2_rn_ftz(k2[2], vn2_7, S2[10]);
                S2[11] = fma_f32x2_rn_ftz(k2[3], vn2_7, S2[11]);
                S2[12] = fma_f32x2_rn_ftz(k2[4], vn2_7, S2[12]);
                S2[13] = fma_f32x2_rn_ftz(k2[5], vn2_7, S2[13]);
                S2[14] = fma_f32x2_rn_ftz(k2[6], vn2_7, S2[14]);
                S2[15] = fma_f32x2_rn_ftz(k2[7], vn2_7, S2[15]);
                int m_4 = t_2 * 3 + 1;
#endif
                unsigned int kbase_4 = sK_addr + (unsigned int)(m_4 * 512) + lane_byte;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(kbase_4));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(kbase_4 + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(kbase_4 + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(kbase_4 + 384));
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
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_4 * 256) + group_byte));
                float beta_4 = sBeta[m_4];
#if __CUDA_ARCH__ == 900
                float acc0_x_8 = 0.0f;
                float acc0_y_8 = 0.0f;
                float acc1_x_8 = 0.0f;
                float acc1_y_8 = 0.0f;
                float _fma_398;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_398) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_8));
                acc0_x_8 = _fma_398;
                float _fma_399;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_399) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_8));
                acc0_y_8 = _fma_399;
                float _fma_400;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_400) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_8));
                acc1_x_8 = _fma_400;
                float _fma_401;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_401) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_8));
                acc1_y_8 = _fma_401;
                float _fma_402;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_402) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_8));
                acc0_x_8 = _fma_402;
                float _fma_403;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_403) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_8));
                acc0_y_8 = _fma_403;
                float _fma_404;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_404) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_8));
                acc1_x_8 = _fma_404;
                float _fma_405;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_405) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_8));
                acc1_y_8 = _fma_405;
                float _fma_406;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_406) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_8));
                acc0_x_8 = _fma_406;
                float _fma_407;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_407) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_8));
                acc0_y_8 = _fma_407;
                float _fma_408;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_408) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_8));
                acc1_x_8 = _fma_408;
                float _fma_409;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_409) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_8));
                acc1_y_8 = _fma_409;
                float _fma_410;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_410) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_8));
                acc0_x_8 = _fma_410;
                float _fma_411;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_411) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_8));
                acc0_y_8 = _fma_411;
                float _fma_412;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_412) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_8));
                acc1_x_8 = _fma_412;
                float _fma_413;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_413) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_8));
                acc1_y_8 = _fma_413;
                float _add_414;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_414) : "f"(acc0_x_8), "f"(acc1_x_8));
                float _add_415;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_415) : "f"(acc0_y_8), "f"(acc1_y_8));
                float dot_8 = _add_414 + _add_415;
                float _shfl_xor_sync_416 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 1);
                dot_8 = dot_8 + _shfl_xor_sync_416;
                float _shfl_xor_sync_417 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 2);
                dot_8 = dot_8 + _shfl_xor_sync_417;
                float _shfl_xor_sync_418 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 4);
                dot_8 = dot_8 + _shfl_xor_sync_418;
#else
                float2 acc0_8 = make_float2(0.0f, 0.0f);
                acc0_8 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_8);
                float dot_8 = acc0_8.x + acc0_8.y;
                float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 1);
                dot_8 = dot_8 + _shfl_xor_sync_64;
                float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 2);
                dot_8 = dot_8 + _shfl_xor_sync_65;
                float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 4);
                dot_8 = dot_8 + _shfl_xor_sync_66;
#endif
                float vn_8 = (v_regs[0] - dot_8) * beta_4;
#if __CUDA_ARCH__ == 900
                float _fma_419;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_419) : "f"(k_regs[0]), "f"(vn_8), "f"(S[0]));
                S[0] = _fma_419;
                float _fma_420;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_420) : "f"(k_regs[1]), "f"(vn_8), "f"(S[1]));
                S[1] = _fma_420;
                float _fma_421;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_421) : "f"(k_regs[2]), "f"(vn_8), "f"(S[2]));
                S[2] = _fma_421;
                float _fma_422;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_422) : "f"(k_regs[3]), "f"(vn_8), "f"(S[3]));
                S[3] = _fma_422;
                float _fma_423;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_423) : "f"(k_regs[4]), "f"(vn_8), "f"(S[4]));
                S[4] = _fma_423;
                float _fma_424;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_424) : "f"(k_regs[5]), "f"(vn_8), "f"(S[5]));
                S[5] = _fma_424;
                float _fma_425;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_425) : "f"(k_regs[6]), "f"(vn_8), "f"(S[6]));
                S[6] = _fma_425;
                float _fma_426;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_426) : "f"(k_regs[7]), "f"(vn_8), "f"(S[7]));
                S[7] = _fma_426;
                float _fma_427;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_427) : "f"(k_regs[8]), "f"(vn_8), "f"(S[8]));
                S[8] = _fma_427;
                float _fma_428;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_428) : "f"(k_regs[9]), "f"(vn_8), "f"(S[9]));
                S[9] = _fma_428;
                float _fma_429;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_429) : "f"(k_regs[10]), "f"(vn_8), "f"(S[10]));
                S[10] = _fma_429;
                float _fma_430;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_430) : "f"(k_regs[11]), "f"(vn_8), "f"(S[11]));
                S[11] = _fma_430;
                float _fma_431;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_431) : "f"(k_regs[12]), "f"(vn_8), "f"(S[12]));
                S[12] = _fma_431;
                float _fma_432;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_432) : "f"(k_regs[13]), "f"(vn_8), "f"(S[13]));
                S[13] = _fma_432;
                float _fma_433;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_433) : "f"(k_regs[14]), "f"(vn_8), "f"(S[14]));
                S[14] = _fma_433;
                float _fma_434;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_434) : "f"(k_regs[15]), "f"(vn_8), "f"(S[15]));
                S[15] = _fma_434;
                float acc0_x_9 = 0.0f;
                float acc0_y_9 = 0.0f;
                float acc1_x_9 = 0.0f;
                float acc1_y_9 = 0.0f;
                float _fma_435;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_435) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_9));
                acc0_x_9 = _fma_435;
                float _fma_436;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_436) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_9));
                acc0_y_9 = _fma_436;
                float _fma_437;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_437) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_9));
                acc1_x_9 = _fma_437;
                float _fma_438;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_438) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_9));
                acc1_y_9 = _fma_438;
                float _fma_439;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_439) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_9));
                acc0_x_9 = _fma_439;
                float _fma_440;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_440) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_9));
                acc0_y_9 = _fma_440;
                float _fma_441;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_441) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_9));
                acc1_x_9 = _fma_441;
                float _fma_442;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_442) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_9));
                acc1_y_9 = _fma_442;
                float _fma_443;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_443) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_9));
                acc0_x_9 = _fma_443;
                float _fma_444;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_444) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_9));
                acc0_y_9 = _fma_444;
                float _fma_445;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_445) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_9));
                acc1_x_9 = _fma_445;
                float _fma_446;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_446) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_9));
                acc1_y_9 = _fma_446;
                float _fma_447;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_447) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_9));
                acc0_x_9 = _fma_447;
                float _fma_448;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_448) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_9));
                acc0_y_9 = _fma_448;
                float _fma_449;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_449) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_9));
                acc1_x_9 = _fma_449;
                float _fma_450;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_450) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_9));
                acc1_y_9 = _fma_450;
                float _add_451;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_451) : "f"(acc0_x_9), "f"(acc1_x_9));
                float _add_452;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_452) : "f"(acc0_y_9), "f"(acc1_y_9));
                float dot_9 = _add_451 + _add_452;
                float _shfl_xor_sync_453 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 1);
                dot_9 = dot_9 + _shfl_xor_sync_453;
                float _shfl_xor_sync_454 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 2);
                dot_9 = dot_9 + _shfl_xor_sync_454;
                float _shfl_xor_sync_455 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 4);
                dot_9 = dot_9 + _shfl_xor_sync_455;
#else
                float2 vn2_8 = make_float2(vn_8, vn_8);
                S2[0] = fma_f32x2_rn_ftz(k2[0], vn2_8, S2[0]);
                S2[1] = fma_f32x2_rn_ftz(k2[1], vn2_8, S2[1]);
                S2[2] = fma_f32x2_rn_ftz(k2[2], vn2_8, S2[2]);
                S2[3] = fma_f32x2_rn_ftz(k2[3], vn2_8, S2[3]);
                S2[4] = fma_f32x2_rn_ftz(k2[4], vn2_8, S2[4]);
                S2[5] = fma_f32x2_rn_ftz(k2[5], vn2_8, S2[5]);
                S2[6] = fma_f32x2_rn_ftz(k2[6], vn2_8, S2[6]);
                S2[7] = fma_f32x2_rn_ftz(k2[7], vn2_8, S2[7]);
                float2 acc0_9 = make_float2(0.0f, 0.0f);
                acc0_9 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_9);
                float dot_9 = acc0_9.x + acc0_9.y;
                float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 1);
                dot_9 = dot_9 + _shfl_xor_sync_67;
                float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 2);
                dot_9 = dot_9 + _shfl_xor_sync_68;
                float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 4);
                dot_9 = dot_9 + _shfl_xor_sync_69;
#endif
                float vn_9 = (v_regs[1] - dot_9) * beta_4;
#if __CUDA_ARCH__ == 900
                float _fma_456;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_456) : "f"(k_regs[0]), "f"(vn_9), "f"(S[16]));
                S[16] = _fma_456;
                float _fma_457;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_457) : "f"(k_regs[1]), "f"(vn_9), "f"(S[17]));
                S[17] = _fma_457;
                float _fma_458;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_458) : "f"(k_regs[2]), "f"(vn_9), "f"(S[18]));
                S[18] = _fma_458;
                float _fma_459;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_459) : "f"(k_regs[3]), "f"(vn_9), "f"(S[19]));
                S[19] = _fma_459;
                float _fma_460;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_460) : "f"(k_regs[4]), "f"(vn_9), "f"(S[20]));
                S[20] = _fma_460;
                float _fma_461;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_461) : "f"(k_regs[5]), "f"(vn_9), "f"(S[21]));
                S[21] = _fma_461;
                float _fma_462;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_462) : "f"(k_regs[6]), "f"(vn_9), "f"(S[22]));
                S[22] = _fma_462;
                float _fma_463;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_463) : "f"(k_regs[7]), "f"(vn_9), "f"(S[23]));
                S[23] = _fma_463;
                float _fma_464;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_464) : "f"(k_regs[8]), "f"(vn_9), "f"(S[24]));
                S[24] = _fma_464;
                float _fma_465;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_465) : "f"(k_regs[9]), "f"(vn_9), "f"(S[25]));
                S[25] = _fma_465;
                float _fma_466;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_466) : "f"(k_regs[10]), "f"(vn_9), "f"(S[26]));
                S[26] = _fma_466;
                float _fma_467;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_467) : "f"(k_regs[11]), "f"(vn_9), "f"(S[27]));
                S[27] = _fma_467;
                float _fma_468;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_468) : "f"(k_regs[12]), "f"(vn_9), "f"(S[28]));
                S[28] = _fma_468;
                float _fma_469;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_469) : "f"(k_regs[13]), "f"(vn_9), "f"(S[29]));
                S[29] = _fma_469;
                float _fma_470;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_470) : "f"(k_regs[14]), "f"(vn_9), "f"(S[30]));
                S[30] = _fma_470;
                float _fma_471;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_471) : "f"(k_regs[15]), "f"(vn_9), "f"(S[31]));
                S[31] = _fma_471;
                int m_5 = t_1 * 3 + 2;
#else
                float2 vn2_9 = make_float2(vn_9, vn_9);
                S2[8] = fma_f32x2_rn_ftz(k2[0], vn2_9, S2[8]);
                S2[9] = fma_f32x2_rn_ftz(k2[1], vn2_9, S2[9]);
                S2[10] = fma_f32x2_rn_ftz(k2[2], vn2_9, S2[10]);
                S2[11] = fma_f32x2_rn_ftz(k2[3], vn2_9, S2[11]);
                S2[12] = fma_f32x2_rn_ftz(k2[4], vn2_9, S2[12]);
                S2[13] = fma_f32x2_rn_ftz(k2[5], vn2_9, S2[13]);
                S2[14] = fma_f32x2_rn_ftz(k2[6], vn2_9, S2[14]);
                S2[15] = fma_f32x2_rn_ftz(k2[7], vn2_9, S2[15]);
                int m_5 = t_2 * 3 + 2;
#endif
                unsigned int kbase_5 = sK_addr + (unsigned int)(m_5 * 512) + lane_byte;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(kbase_5));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(kbase_5 + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(kbase_5 + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(kbase_5 + 384));
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
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_5 * 256) + group_byte));
                float beta_5 = sBeta[m_5];
#if __CUDA_ARCH__ == 900
                float acc0_x_10 = 0.0f;
                float acc0_y_10 = 0.0f;
                float acc1_x_10 = 0.0f;
                float acc1_y_10 = 0.0f;
                float _fma_472;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_472) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_10));
                acc0_x_10 = _fma_472;
                float _fma_473;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_473) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_10));
                acc0_y_10 = _fma_473;
                float _fma_474;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_474) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_10));
                acc1_x_10 = _fma_474;
                float _fma_475;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_475) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_10));
                acc1_y_10 = _fma_475;
                float _fma_476;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_476) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_10));
                acc0_x_10 = _fma_476;
                float _fma_477;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_477) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_10));
                acc0_y_10 = _fma_477;
                float _fma_478;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_478) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_10));
                acc1_x_10 = _fma_478;
                float _fma_479;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_479) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_10));
                acc1_y_10 = _fma_479;
                float _fma_480;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_480) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_10));
                acc0_x_10 = _fma_480;
                float _fma_481;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_481) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_10));
                acc0_y_10 = _fma_481;
                float _fma_482;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_482) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_10));
                acc1_x_10 = _fma_482;
                float _fma_483;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_483) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_10));
                acc1_y_10 = _fma_483;
                float _fma_484;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_484) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_10));
                acc0_x_10 = _fma_484;
                float _fma_485;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_485) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_10));
                acc0_y_10 = _fma_485;
                float _fma_486;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_486) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_10));
                acc1_x_10 = _fma_486;
                float _fma_487;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_487) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_10));
                acc1_y_10 = _fma_487;
                float _add_488;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_488) : "f"(acc0_x_10), "f"(acc1_x_10));
                float _add_489;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_489) : "f"(acc0_y_10), "f"(acc1_y_10));
                float dot_10 = _add_488 + _add_489;
                float _shfl_xor_sync_490 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 1);
                dot_10 = dot_10 + _shfl_xor_sync_490;
                float _shfl_xor_sync_491 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 2);
                dot_10 = dot_10 + _shfl_xor_sync_491;
                float _shfl_xor_sync_492 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 4);
                dot_10 = dot_10 + _shfl_xor_sync_492;
#else
                float2 acc0_10 = make_float2(0.0f, 0.0f);
                acc0_10 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_10);
                float dot_10 = acc0_10.x + acc0_10.y;
                float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 1);
                dot_10 = dot_10 + _shfl_xor_sync_70;
                float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 2);
                dot_10 = dot_10 + _shfl_xor_sync_71;
                float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 4);
                dot_10 = dot_10 + _shfl_xor_sync_72;
#endif
                float vn_10 = (v_regs[0] - dot_10) * beta_5;
#if __CUDA_ARCH__ == 900
                float _fma_493;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_493) : "f"(k_regs[0]), "f"(vn_10), "f"(S[0]));
                S[0] = _fma_493;
                float _fma_494;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_494) : "f"(k_regs[1]), "f"(vn_10), "f"(S[1]));
                S[1] = _fma_494;
                float _fma_495;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_495) : "f"(k_regs[2]), "f"(vn_10), "f"(S[2]));
                S[2] = _fma_495;
                float _fma_496;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_496) : "f"(k_regs[3]), "f"(vn_10), "f"(S[3]));
                S[3] = _fma_496;
                float _fma_497;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_497) : "f"(k_regs[4]), "f"(vn_10), "f"(S[4]));
                S[4] = _fma_497;
                float _fma_498;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_498) : "f"(k_regs[5]), "f"(vn_10), "f"(S[5]));
                S[5] = _fma_498;
                float _fma_499;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_499) : "f"(k_regs[6]), "f"(vn_10), "f"(S[6]));
                S[6] = _fma_499;
                float _fma_500;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_500) : "f"(k_regs[7]), "f"(vn_10), "f"(S[7]));
                S[7] = _fma_500;
                float _fma_501;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_501) : "f"(k_regs[8]), "f"(vn_10), "f"(S[8]));
                S[8] = _fma_501;
                float _fma_502;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_502) : "f"(k_regs[9]), "f"(vn_10), "f"(S[9]));
                S[9] = _fma_502;
                float _fma_503;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_503) : "f"(k_regs[10]), "f"(vn_10), "f"(S[10]));
                S[10] = _fma_503;
                float _fma_504;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_504) : "f"(k_regs[11]), "f"(vn_10), "f"(S[11]));
                S[11] = _fma_504;
                float _fma_505;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_505) : "f"(k_regs[12]), "f"(vn_10), "f"(S[12]));
                S[12] = _fma_505;
                float _fma_506;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_506) : "f"(k_regs[13]), "f"(vn_10), "f"(S[13]));
                S[13] = _fma_506;
                float _fma_507;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_507) : "f"(k_regs[14]), "f"(vn_10), "f"(S[14]));
                S[14] = _fma_507;
                float _fma_508;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_508) : "f"(k_regs[15]), "f"(vn_10), "f"(S[15]));
                S[15] = _fma_508;
                float acc0_x_11 = 0.0f;
                float acc0_y_11 = 0.0f;
                float acc1_x_11 = 0.0f;
                float acc1_y_11 = 0.0f;
                float _fma_509;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_509) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_11));
                acc0_x_11 = _fma_509;
                float _fma_510;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_510) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_11));
                acc0_y_11 = _fma_510;
                float _fma_511;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_511) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_11));
                acc1_x_11 = _fma_511;
                float _fma_512;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_512) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_11));
                acc1_y_11 = _fma_512;
                float _fma_513;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_513) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_11));
                acc0_x_11 = _fma_513;
                float _fma_514;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_514) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_11));
                acc0_y_11 = _fma_514;
                float _fma_515;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_515) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_11));
                acc1_x_11 = _fma_515;
                float _fma_516;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_516) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_11));
                acc1_y_11 = _fma_516;
                float _fma_517;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_517) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_11));
                acc0_x_11 = _fma_517;
                float _fma_518;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_518) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_11));
                acc0_y_11 = _fma_518;
                float _fma_519;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_519) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_11));
                acc1_x_11 = _fma_519;
                float _fma_520;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_520) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_11));
                acc1_y_11 = _fma_520;
                float _fma_521;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_521) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_11));
                acc0_x_11 = _fma_521;
                float _fma_522;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_522) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_11));
                acc0_y_11 = _fma_522;
                float _fma_523;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_523) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_11));
                acc1_x_11 = _fma_523;
                float _fma_524;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_524) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_11));
                acc1_y_11 = _fma_524;
                float _add_525;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_525) : "f"(acc0_x_11), "f"(acc1_x_11));
                float _add_526;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_526) : "f"(acc0_y_11), "f"(acc1_y_11));
                float dot_11 = _add_525 + _add_526;
                float _shfl_xor_sync_527 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 1);
                dot_11 = dot_11 + _shfl_xor_sync_527;
                float _shfl_xor_sync_528 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 2);
                dot_11 = dot_11 + _shfl_xor_sync_528;
                float _shfl_xor_sync_529 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 4);
                dot_11 = dot_11 + _shfl_xor_sync_529;
#else
                float2 vn2_10 = make_float2(vn_10, vn_10);
                S2[0] = fma_f32x2_rn_ftz(k2[0], vn2_10, S2[0]);
                S2[1] = fma_f32x2_rn_ftz(k2[1], vn2_10, S2[1]);
                S2[2] = fma_f32x2_rn_ftz(k2[2], vn2_10, S2[2]);
                S2[3] = fma_f32x2_rn_ftz(k2[3], vn2_10, S2[3]);
                S2[4] = fma_f32x2_rn_ftz(k2[4], vn2_10, S2[4]);
                S2[5] = fma_f32x2_rn_ftz(k2[5], vn2_10, S2[5]);
                S2[6] = fma_f32x2_rn_ftz(k2[6], vn2_10, S2[6]);
                S2[7] = fma_f32x2_rn_ftz(k2[7], vn2_10, S2[7]);
                float2 acc0_11 = make_float2(0.0f, 0.0f);
                acc0_11 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_11);
                float dot_11 = acc0_11.x + acc0_11.y;
                float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 1);
                dot_11 = dot_11 + _shfl_xor_sync_73;
                float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 2);
                dot_11 = dot_11 + _shfl_xor_sync_74;
                float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 4);
                dot_11 = dot_11 + _shfl_xor_sync_75;
#endif
                float vn_11 = (v_regs[1] - dot_11) * beta_5;
#if __CUDA_ARCH__ == 900
                float _fma_530;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_530) : "f"(k_regs[0]), "f"(vn_11), "f"(S[16]));
                S[16] = _fma_530;
                float _fma_531;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_531) : "f"(k_regs[1]), "f"(vn_11), "f"(S[17]));
                S[17] = _fma_531;
                float _fma_532;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_532) : "f"(k_regs[2]), "f"(vn_11), "f"(S[18]));
                S[18] = _fma_532;
                float _fma_533;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_533) : "f"(k_regs[3]), "f"(vn_11), "f"(S[19]));
                S[19] = _fma_533;
                float _fma_534;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_534) : "f"(k_regs[4]), "f"(vn_11), "f"(S[20]));
                S[20] = _fma_534;
                float _fma_535;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_535) : "f"(k_regs[5]), "f"(vn_11), "f"(S[21]));
                S[21] = _fma_535;
                float _fma_536;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_536) : "f"(k_regs[6]), "f"(vn_11), "f"(S[22]));
                S[22] = _fma_536;
                float _fma_537;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_537) : "f"(k_regs[7]), "f"(vn_11), "f"(S[23]));
                S[23] = _fma_537;
                float _fma_538;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_538) : "f"(k_regs[8]), "f"(vn_11), "f"(S[24]));
                S[24] = _fma_538;
                float _fma_539;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_539) : "f"(k_regs[9]), "f"(vn_11), "f"(S[25]));
                S[25] = _fma_539;
                float _fma_540;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_540) : "f"(k_regs[10]), "f"(vn_11), "f"(S[26]));
                S[26] = _fma_540;
                float _fma_541;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_541) : "f"(k_regs[11]), "f"(vn_11), "f"(S[27]));
                S[27] = _fma_541;
                float _fma_542;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_542) : "f"(k_regs[12]), "f"(vn_11), "f"(S[28]));
                S[28] = _fma_542;
                float _fma_543;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_543) : "f"(k_regs[13]), "f"(vn_11), "f"(S[29]));
                S[29] = _fma_543;
                float _fma_544;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_544) : "f"(k_regs[14]), "f"(vn_11), "f"(S[30]));
                S[30] = _fma_544;
                float _fma_545;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_545) : "f"(k_regs[15]), "f"(vn_11), "f"(S[31]));
                S[31] = _fma_545;
                unsigned int qbase_1 = sQ_addr + (unsigned int)(t_1 * 512) + lane_byte;
#else
                float2 vn2_11 = make_float2(vn_11, vn_11);
                S2[8] = fma_f32x2_rn_ftz(k2[0], vn2_11, S2[8]);
                S2[9] = fma_f32x2_rn_ftz(k2[1], vn2_11, S2[9]);
                S2[10] = fma_f32x2_rn_ftz(k2[2], vn2_11, S2[10]);
                S2[11] = fma_f32x2_rn_ftz(k2[3], vn2_11, S2[11]);
                S2[12] = fma_f32x2_rn_ftz(k2[4], vn2_11, S2[12]);
                S2[13] = fma_f32x2_rn_ftz(k2[5], vn2_11, S2[13]);
                S2[14] = fma_f32x2_rn_ftz(k2[6], vn2_11, S2[14]);
                S2[15] = fma_f32x2_rn_ftz(k2[7], vn2_11, S2[15]);
                unsigned int qbase_1 = sQ_addr + (unsigned int)(t_2 * 512) + lane_byte;
#endif
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(qbase_1));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(qbase_1 + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(qbase_1 + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(qbase_1 + 384));
#if __CUDA_ARCH__ == 900
                float acc0_x_12 = 0.0f;
                float acc0_y_12 = 0.0f;
                float acc1_x_12 = 0.0f;
                float acc1_y_12 = 0.0f;
                float _fma_546;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_546) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_12));
                acc0_x_12 = _fma_546;
                float _fma_547;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_547) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_12));
                acc0_y_12 = _fma_547;
                float _fma_548;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_548) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_12));
                acc1_x_12 = _fma_548;
                float _fma_549;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_549) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_12));
                acc1_y_12 = _fma_549;
                float _fma_550;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_550) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_12));
                acc0_x_12 = _fma_550;
                float _fma_551;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_551) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_12));
                acc0_y_12 = _fma_551;
                float _fma_552;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_552) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_12));
                acc1_x_12 = _fma_552;
                float _fma_553;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_553) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_12));
                acc1_y_12 = _fma_553;
                float _fma_554;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_554) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_12));
                acc0_x_12 = _fma_554;
                float _fma_555;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_555) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_12));
                acc0_y_12 = _fma_555;
                float _fma_556;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_556) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_12));
                acc1_x_12 = _fma_556;
                float _fma_557;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_557) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_12));
                acc1_y_12 = _fma_557;
                float _fma_558;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_558) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_12));
                acc0_x_12 = _fma_558;
                float _fma_559;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_559) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_12));
                acc0_y_12 = _fma_559;
                float _fma_560;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_560) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_12));
                acc1_x_12 = _fma_560;
                float _fma_561;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_561) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_12));
                acc1_y_12 = _fma_561;
                float _add_562;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_562) : "f"(acc0_x_12), "f"(acc1_x_12));
                float _add_563;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_563) : "f"(acc0_y_12), "f"(acc1_y_12));
                float dot_12 = _add_562 + _add_563;
                float _shfl_xor_sync_564 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 1);
                dot_12 = dot_12 + _shfl_xor_sync_564;
                float _shfl_xor_sync_565 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 2);
                dot_12 = dot_12 + _shfl_xor_sync_565;
                float _shfl_xor_sync_566 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 4);
                dot_12 = dot_12 + _shfl_xor_sync_566;
#else
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                float2 acc0_12 = make_float2(0.0f, 0.0f);
                acc0_12 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_12);
                float dot_12 = acc0_12.x + acc0_12.y;
                float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 1);
                dot_12 = dot_12 + _shfl_xor_sync_76;
                float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 2);
                dot_12 = dot_12 + _shfl_xor_sync_77;
                float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 4);
                dot_12 = dot_12 + _shfl_xor_sync_78;
#endif
                o4[0] = dot_12;
#if __CUDA_ARCH__ == 900
                float acc0_x_13 = 0.0f;
                float acc0_y_13 = 0.0f;
                float acc1_x_13 = 0.0f;
                float acc1_y_13 = 0.0f;
                float _fma_567;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_567) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_13));
                acc0_x_13 = _fma_567;
                float _fma_568;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_568) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_13));
                acc0_y_13 = _fma_568;
                float _fma_569;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_569) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_13));
                acc1_x_13 = _fma_569;
                float _fma_570;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_570) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_13));
                acc1_y_13 = _fma_570;
                float _fma_571;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_571) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_13));
                acc0_x_13 = _fma_571;
                float _fma_572;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_572) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_13));
                acc0_y_13 = _fma_572;
                float _fma_573;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_573) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_13));
                acc1_x_13 = _fma_573;
                float _fma_574;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_574) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_13));
                acc1_y_13 = _fma_574;
                float _fma_575;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_575) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_13));
                acc0_x_13 = _fma_575;
                float _fma_576;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_576) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_13));
                acc0_y_13 = _fma_576;
                float _fma_577;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_577) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_13));
                acc1_x_13 = _fma_577;
                float _fma_578;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_578) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_13));
                acc1_y_13 = _fma_578;
                float _fma_579;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_579) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_13));
                acc0_x_13 = _fma_579;
                float _fma_580;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_580) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_13));
                acc0_y_13 = _fma_580;
                float _fma_581;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_581) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_13));
                acc1_x_13 = _fma_581;
                float _fma_582;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_582) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_13));
                acc1_y_13 = _fma_582;
                float _add_583;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_583) : "f"(acc0_x_13), "f"(acc1_x_13));
                float _add_584;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_584) : "f"(acc0_y_13), "f"(acc1_y_13));
                float dot_13 = _add_583 + _add_584;
                float _shfl_xor_sync_585 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 1);
                dot_13 = dot_13 + _shfl_xor_sync_585;
                float _shfl_xor_sync_586 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 2);
                dot_13 = dot_13 + _shfl_xor_sync_586;
                float _shfl_xor_sync_587 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 4);
                dot_13 = dot_13 + _shfl_xor_sync_587;
#else
                float2 acc0_13 = make_float2(0.0f, 0.0f);
                acc0_13 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_13);
                float dot_13 = acc0_13.x + acc0_13.y;
                float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 1);
                dot_13 = dot_13 + _shfl_xor_sync_79;
                float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 2);
                dot_13 = dot_13 + _shfl_xor_sync_80;
                float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 4);
                dot_13 = dot_13 + _shfl_xor_sync_81;
#endif
                o4[1] = dot_13;
                if (u == 0) {
#if __CUDA_ARCH__ == 900
                    long long out_index_1 = (long long)n * out_stride_p0 + (long long)t_1 * out_stride_p1 + (long long)h * out_stride_p2;
#else
                    long long out_index_1 = (long long)n * out_stride_p0 + (long long)t_2 * out_stride_p1 + (long long)h * out_stride_p2;
#endif
                    unsigned int packed_1[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o4[_lp*2 + 0], o4[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(out + (out_index_1 + (long long)row0))[0] = reinterpret_cast<unsigned*>(packed_1)[0];
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
                        *reinterpret_cast<float4*>(state + write_rows_base + 32ll) = _v4;
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
                        *reinterpret_cast<float4*>(state + write_rows_base + 64ll) = _v4;
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
                        *reinterpret_cast<float4*>(state + write_rows_base + 96ll) = _v4;
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
                        *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 32ll) = _v4;
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
                        *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 64ll) = _v4;
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
                        *reinterpret_cast<float4*>(state + write_rows_base + state_stride_p2 + 96ll) = _v4;
                    }
                }
            }
        }
    }
}

} // extern "C"
