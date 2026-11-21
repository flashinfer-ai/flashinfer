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

#define LAUNCH_MIN_BLOCKS 4

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

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdp_1a3d01a5826508d64a06(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, int* __restrict__ ssm_state_indices, CakeFastDivmod heads, int num_tokens, float scale, int disable_state_update, int use_ssm_state_indices, long long state_size_p0, long long state_stride_p1, long long state_stride_p2, long long state_stride_p0, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long ssm_state_indices_stride_p0, long long ssm_state_indices_stride_p1, long long out_stride_p0, long long out_stride_p1, long long out_stride_p2)
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
    float* sQ = reinterpret_cast<float*>(smem_raw + 10752);
    const int sQ_addr = smem + 10752;
    float* sV = reinterpret_cast<float*>(smem_raw + 14336);
    const int sV_addr = smem + 14336;
    float* sAlpha = reinterpret_cast<float*>(smem_raw + 19712);
    const int sAlpha_addr = smem + 19712;
    float* sBeta = reinterpret_cast<float*>(smem_raw + 19740);
    const int sBeta_addr = smem + 19740;
    int* sSsm = reinterpret_cast<int*>(smem_raw + 19824);
    const int sSsm_addr = smem + 19824;

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
    int row0 = g * 4;
    unsigned int lane_byte = (unsigned int)(u * 16);
    unsigned int group_byte = (unsigned int)(g * 16);
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
    int steps = 21;
    float r_q[4];
    float r_k[4];
#if __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1030
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
    int bm0_1 = 4 + warp_local;
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
    int bm0_2 = 8 + warp_local;
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
    int bm0_3 = 12 + warp_local;
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
#endif
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
        {
            float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 2ll * state_stride_p2);
            S[32 + 0] = _v4.x;
            S[32 + 1] = _v4.y;
            S[32 + 2] = _v4.z;
            S[32 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 2ll * state_stride_p2 + 32ll);
            S[36 + 0] = _v4.x;
            S[36 + 1] = _v4.y;
            S[36 + 2] = _v4.z;
            S[36 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 2ll * state_stride_p2 + 64ll);
            S[40 + 0] = _v4.x;
            S[40 + 1] = _v4.y;
            S[40 + 2] = _v4.z;
            S[40 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 2ll * state_stride_p2 + 96ll);
            S[44 + 0] = _v4.x;
            S[44 + 1] = _v4.y;
            S[44 + 2] = _v4.z;
            S[44 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 3ll * state_stride_p2);
            S[48 + 0] = _v4.x;
            S[48 + 1] = _v4.y;
            S[48 + 2] = _v4.z;
            S[48 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 3ll * state_stride_p2 + 32ll);
            S[52 + 0] = _v4.x;
            S[52 + 1] = _v4.y;
            S[52 + 2] = _v4.z;
            S[52 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 3ll * state_stride_p2 + 64ll);
            S[56 + 0] = _v4.x;
            S[56 + 1] = _v4.y;
            S[56 + 2] = _v4.z;
            S[56 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(state + read_rows_base + 3ll * state_stride_p2 + 96ll);
            S[60 + 0] = _v4.x;
            S[60 + 1] = _v4.y;
            S[60 + 2] = _v4.z;
            S[60 + 3] = _v4.w;
        }
#if __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1030
#else
        float bk[16];
        float bv[8];
        float bb[4];
        int bm0_0 = warp_local;
        int bm_0 = bm0_0;
        long long bki_0 = (long long)n * k_stride_p0 + (long long)bm_0 * k_stride_p1 + (long long)h * k_stride_p2;
        {
            uint2 _vld_16;
            _vld_16 = *reinterpret_cast<const uint2*>(k + bki_0 + (long long)k_col0);
            uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&bk[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
                (&bk[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
            }
        }
        long long bvi_0 = (long long)n * v_stride_p0 + (long long)bm_0 * v_stride_p1 + (long long)h * v_stride_p2;
        int bvc_0 = lane_local * 2;
        bv[0] = (float)v[bvi_0 + (long long)bvc_0];
        bv[1] = (float)v[bvi_0 + (long long)bvc_0 + 1ll];
        long long bbi_0 = (long long)n * b_stride_p0 + (long long)bm_0 * b_stride_p1 + (long long)h * b_stride_p2;
        bb[0] = (float)b[bbi_0];
        int bm0_1 = 4 + warp_local;
        int bm_1 = bm0_1;
        long long bki_1 = (long long)n * k_stride_p0 + (long long)bm_1 * k_stride_p1 + (long long)h * k_stride_p2;
        {
            uint2 _vld_17;
            _vld_17 = *reinterpret_cast<const uint2*>(k + bki_1 + (long long)k_col0);
            uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&bk[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) << 16);
                (&bk[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) & 0xffff0000u);
            }
        }
        long long bvi_1 = (long long)n * v_stride_p0 + (long long)bm_1 * v_stride_p1 + (long long)h * v_stride_p2;
        int bvc_1 = lane_local * 2;
        bv[2] = (float)v[bvi_1 + (long long)bvc_1];
        bv[3] = (float)v[bvi_1 + (long long)bvc_1 + 1ll];
        long long bbi_1 = (long long)n * b_stride_p0 + (long long)bm_1 * b_stride_p1 + (long long)h * b_stride_p2;
        bb[1] = (float)b[bbi_1];
        int bm0_2 = 8 + warp_local;
        int bm_2 = bm0_2;
        long long bki_2 = (long long)n * k_stride_p0 + (long long)bm_2 * k_stride_p1 + (long long)h * k_stride_p2;
        {
            uint2 _vld_18;
            _vld_18 = *reinterpret_cast<const uint2*>(k + bki_2 + (long long)k_col0);
            uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&bk[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
                (&bk[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
            }
        }
        long long bvi_2 = (long long)n * v_stride_p0 + (long long)bm_2 * v_stride_p1 + (long long)h * v_stride_p2;
        int bvc_2 = lane_local * 2;
        bv[4] = (float)v[bvi_2 + (long long)bvc_2];
        bv[5] = (float)v[bvi_2 + (long long)bvc_2 + 1ll];
        long long bbi_2 = (long long)n * b_stride_p0 + (long long)bm_2 * b_stride_p1 + (long long)h * b_stride_p2;
        bb[2] = (float)b[bbi_2];
        int bm0_3 = 12 + warp_local;
        int bm_3 = bm0_3;
        long long bki_3 = (long long)n * k_stride_p0 + (long long)bm_3 * k_stride_p1 + (long long)h * k_stride_p2;
        {
            uint2 _vld_19;
            _vld_19 = *reinterpret_cast<const uint2*>(k + bki_3 + (long long)k_col0);
            uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&bk[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) << 16);
                (&bk[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) & 0xffff0000u);
            }
        }
        long long bvi_3 = (long long)n * v_stride_p0 + (long long)bm_3 * v_stride_p1 + (long long)h * v_stride_p2;
        int bvc_3 = lane_local * 2;
        bv[6] = (float)v[bvi_3 + (long long)bvc_3];
        bv[7] = (float)v[bvi_3 + (long long)bvc_3 + 1ll];
        long long bbi_3 = (long long)n * b_stride_p0 + (long long)bm_3 * b_stride_p1 + (long long)h * b_stride_p2;
        bb[3] = (float)b[bbi_3];
#endif
        int fm_0 = warp_local;
#if __CUDA_ARCH__ == 900
        float x0_x = bk[0];
        float x0_y = bk[1];
        float x1_x = bk[2];
        float x1_y = bk[3];
        float sum_x;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x) : "f"(x0_x), "f"(x0_x), "f"(0.0f));
        float sum_y;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y) : "f"(x0_y), "f"(x0_y), "f"(0.0f));
        float _fma_20;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(x1_x), "f"(x1_x), "f"(sum_x));
        sum_x = _fma_20;
        float _fma_21;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(x1_y), "f"(x1_y), "f"(sum_y));
        sum_y = _fma_21;
        sum_x = sum_x + sum_y;
        float _warp_reduce_22 = sum_x;
#else
        float2 x0 = make_float2(bk[0], bk[1]);
        float2 x1 = make_float2(bk[2], bk[3]);
        float2 sum_pair = fma_f32x2_rn_ftz(x0, x0, make_float2(0.0f, 0.0f));
        sum_pair = fma_f32x2_rn_ftz(x1, x1, sum_pair);
        float sum_x = sum_pair.x + sum_pair.y;
        float _warp_reduce_20 = sum_x;
#endif
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
            _warp_reduce_22 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_22, offset);
        float norm = rsqrtf(_warp_reduce_22 + 1e-06f);
        x0_x = mul_f32_rn_ftz(x0_x, norm);
        x0_y = mul_f32_rn_ftz(x0_y, norm);
        x1_x = mul_f32_rn_ftz(x1_x, norm);
        x1_y = mul_f32_rn_ftz(x1_y, norm);
        sK[fm_0 * 128 + k_col0] = x0_x;
        sK[fm_0 * 128 + k_col0 + 1] = x0_y;
        sK[fm_0 * 128 + k_col0 + 2] = x1_x;
        sK[fm_0 * 128 + k_col0 + 3] = x1_y;
#else
            _warp_reduce_20 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_20, offset);
        float norm = rsqrtf(_warp_reduce_20 + 1e-06f);
        float2 norm2 = make_float2(norm, norm);
        x0 = mul_f32x2(x0, norm2);
        x1 = mul_f32x2(x1, norm2);
        sK[fm_0 * 128 + k_col0] = x0.x;
        sK[fm_0 * 128 + k_col0 + 1] = x0.y;
        sK[fm_0 * 128 + k_col0 + 2] = x1.x;
        sK[fm_0 * 128 + k_col0 + 3] = x1.y;
#endif
        int v_col = lane_local * 2;
        sV[fm_0 * 64 + v_col] = bv[0];
        sV[fm_0 * 64 + v_col + 1] = bv[1];
        if (lane_local == 0) {
#if __CUDA_ARCH__ == 900
            float _exp_24 = __expf(-bb[0]);
            float _rcp_23 = approx_rcp(1.0f + _exp_24);
            sBeta[fm_0] = _rcp_23;
#else
            float _exp_22 = __expf(-bb[0]);
            float _rcp_21 = approx_rcp(1.0f + _exp_22);
            sBeta[fm_0] = _rcp_21;
#endif
        }
        int fm_1 = 4 + warp_local;
#if __CUDA_ARCH__ == 900
        float x0_x_0 = bk[4];
        float x0_y_0 = bk[5];
        float x1_x_0 = bk[6];
        float x1_y_0 = bk[7];
        float sum_x_0;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_0) : "f"(x0_x_0), "f"(x0_x_0), "f"(0.0f));
        float sum_y_0;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_0) : "f"(x0_y_0), "f"(x0_y_0), "f"(0.0f));
        float _fma_25;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(x1_x_0), "f"(x1_x_0), "f"(sum_x_0));
        sum_x_0 = _fma_25;
        float _fma_26;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(x1_y_0), "f"(x1_y_0), "f"(sum_y_0));
        sum_y_0 = _fma_26;
        sum_x_0 = sum_x_0 + sum_y_0;
        float _warp_reduce_27 = sum_x_0;
#else
        float2 x0_0 = make_float2(bk[4], bk[5]);
        float2 x1_0 = make_float2(bk[6], bk[7]);
        float2 sum_pair_0 = fma_f32x2_rn_ftz(x0_0, x0_0, make_float2(0.0f, 0.0f));
        sum_pair_0 = fma_f32x2_rn_ftz(x1_0, x1_0, sum_pair_0);
        float sum_x_0 = sum_pair_0.x + sum_pair_0.y;
        float _warp_reduce_23 = sum_x_0;
#endif
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
            _warp_reduce_27 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_27, offset);
        float norm_0 = rsqrtf(_warp_reduce_27 + 1e-06f);
        x0_x_0 = mul_f32_rn_ftz(x0_x_0, norm_0);
        x0_y_0 = mul_f32_rn_ftz(x0_y_0, norm_0);
        x1_x_0 = mul_f32_rn_ftz(x1_x_0, norm_0);
        x1_y_0 = mul_f32_rn_ftz(x1_y_0, norm_0);
        sK[fm_1 * 128 + k_col0] = x0_x_0;
        sK[fm_1 * 128 + k_col0 + 1] = x0_y_0;
        sK[fm_1 * 128 + k_col0 + 2] = x1_x_0;
        sK[fm_1 * 128 + k_col0 + 3] = x1_y_0;
#else
            _warp_reduce_23 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_23, offset);
        float norm_0 = rsqrtf(_warp_reduce_23 + 1e-06f);
        float2 norm2_0 = make_float2(norm_0, norm_0);
        x0_0 = mul_f32x2(x0_0, norm2_0);
        x1_0 = mul_f32x2(x1_0, norm2_0);
        sK[fm_1 * 128 + k_col0] = x0_0.x;
        sK[fm_1 * 128 + k_col0 + 1] = x0_0.y;
        sK[fm_1 * 128 + k_col0 + 2] = x1_0.x;
        sK[fm_1 * 128 + k_col0 + 3] = x1_0.y;
#endif
        int v_col_0 = lane_local * 2;
        sV[fm_1 * 64 + v_col_0] = bv[2];
        sV[fm_1 * 64 + v_col_0 + 1] = bv[3];
        if (lane_local == 0) {
#if __CUDA_ARCH__ == 900
            float _exp_29 = __expf(-bb[1]);
            float _rcp_28 = approx_rcp(1.0f + _exp_29);
            sBeta[fm_1] = _rcp_28;
#else
            float _exp_25 = __expf(-bb[1]);
            float _rcp_24 = approx_rcp(1.0f + _exp_25);
            sBeta[fm_1] = _rcp_24;
#endif
        }
        int fm_2 = 8 + warp_local;
#if __CUDA_ARCH__ == 900
        float x0_x_1 = bk[8];
        float x0_y_1 = bk[9];
        float x1_x_1 = bk[10];
        float x1_y_1 = bk[11];
        float sum_x_1;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_1) : "f"(x0_x_1), "f"(x0_x_1), "f"(0.0f));
        float sum_y_1;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_1) : "f"(x0_y_1), "f"(x0_y_1), "f"(0.0f));
        float _fma_30;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(x1_x_1), "f"(x1_x_1), "f"(sum_x_1));
        sum_x_1 = _fma_30;
        float _fma_31;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(x1_y_1), "f"(x1_y_1), "f"(sum_y_1));
        sum_y_1 = _fma_31;
        sum_x_1 = sum_x_1 + sum_y_1;
        float _warp_reduce_32 = sum_x_1;
#else
        float2 x0_1 = make_float2(bk[8], bk[9]);
        float2 x1_1 = make_float2(bk[10], bk[11]);
        float2 sum_pair_1 = fma_f32x2_rn_ftz(x0_1, x0_1, make_float2(0.0f, 0.0f));
        sum_pair_1 = fma_f32x2_rn_ftz(x1_1, x1_1, sum_pair_1);
        float sum_x_1 = sum_pair_1.x + sum_pair_1.y;
        float _warp_reduce_26 = sum_x_1;
#endif
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
            _warp_reduce_32 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_32, offset);
        float norm_1 = rsqrtf(_warp_reduce_32 + 1e-06f);
        x0_x_1 = mul_f32_rn_ftz(x0_x_1, norm_1);
        x0_y_1 = mul_f32_rn_ftz(x0_y_1, norm_1);
        x1_x_1 = mul_f32_rn_ftz(x1_x_1, norm_1);
        x1_y_1 = mul_f32_rn_ftz(x1_y_1, norm_1);
        sK[fm_2 * 128 + k_col0] = x0_x_1;
        sK[fm_2 * 128 + k_col0 + 1] = x0_y_1;
        sK[fm_2 * 128 + k_col0 + 2] = x1_x_1;
        sK[fm_2 * 128 + k_col0 + 3] = x1_y_1;
#else
            _warp_reduce_26 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_26, offset);
        float norm_1 = rsqrtf(_warp_reduce_26 + 1e-06f);
        float2 norm2_1 = make_float2(norm_1, norm_1);
        x0_1 = mul_f32x2(x0_1, norm2_1);
        x1_1 = mul_f32x2(x1_1, norm2_1);
        sK[fm_2 * 128 + k_col0] = x0_1.x;
        sK[fm_2 * 128 + k_col0 + 1] = x0_1.y;
        sK[fm_2 * 128 + k_col0 + 2] = x1_1.x;
        sK[fm_2 * 128 + k_col0 + 3] = x1_1.y;
#endif
        int v_col_1 = lane_local * 2;
        sV[fm_2 * 64 + v_col_1] = bv[4];
        sV[fm_2 * 64 + v_col_1 + 1] = bv[5];
        if (lane_local == 0) {
#if __CUDA_ARCH__ == 900
            float _exp_34 = __expf(-bb[2]);
            float _rcp_33 = approx_rcp(1.0f + _exp_34);
            sBeta[fm_2] = _rcp_33;
#else
            float _exp_28 = __expf(-bb[2]);
            float _rcp_27 = approx_rcp(1.0f + _exp_28);
            sBeta[fm_2] = _rcp_27;
#endif
        }
        int fm_3 = 12 + warp_local;
#if __CUDA_ARCH__ == 900
        float x0_x_2 = bk[12];
        float x0_y_2 = bk[13];
        float x1_x_2 = bk[14];
        float x1_y_2 = bk[15];
        float sum_x_2;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_2) : "f"(x0_x_2), "f"(x0_x_2), "f"(0.0f));
        float sum_y_2;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_2) : "f"(x0_y_2), "f"(x0_y_2), "f"(0.0f));
        float _fma_35;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(x1_x_2), "f"(x1_x_2), "f"(sum_x_2));
        sum_x_2 = _fma_35;
        float _fma_36;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(x1_y_2), "f"(x1_y_2), "f"(sum_y_2));
        sum_y_2 = _fma_36;
        sum_x_2 = sum_x_2 + sum_y_2;
        float _warp_reduce_37 = sum_x_2;
#else
        float2 x0_2 = make_float2(bk[12], bk[13]);
        float2 x1_2 = make_float2(bk[14], bk[15]);
        float2 sum_pair_2 = fma_f32x2_rn_ftz(x0_2, x0_2, make_float2(0.0f, 0.0f));
        sum_pair_2 = fma_f32x2_rn_ftz(x1_2, x1_2, sum_pair_2);
        float sum_x_2 = sum_pair_2.x + sum_pair_2.y;
        float _warp_reduce_29 = sum_x_2;
#endif
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
            _warp_reduce_37 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_37, offset);
        float norm_2 = rsqrtf(_warp_reduce_37 + 1e-06f);
        x0_x_2 = mul_f32_rn_ftz(x0_x_2, norm_2);
        x0_y_2 = mul_f32_rn_ftz(x0_y_2, norm_2);
        x1_x_2 = mul_f32_rn_ftz(x1_x_2, norm_2);
        x1_y_2 = mul_f32_rn_ftz(x1_y_2, norm_2);
        sK[fm_3 * 128 + k_col0] = x0_x_2;
        sK[fm_3 * 128 + k_col0 + 1] = x0_y_2;
        sK[fm_3 * 128 + k_col0 + 2] = x1_x_2;
        sK[fm_3 * 128 + k_col0 + 3] = x1_y_2;
#else
            _warp_reduce_29 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_29, offset);
        float norm_2 = rsqrtf(_warp_reduce_29 + 1e-06f);
        float2 norm2_2 = make_float2(norm_2, norm_2);
        x0_2 = mul_f32x2(x0_2, norm2_2);
        x1_2 = mul_f32x2(x1_2, norm2_2);
        sK[fm_3 * 128 + k_col0] = x0_2.x;
        sK[fm_3 * 128 + k_col0 + 1] = x0_2.y;
        sK[fm_3 * 128 + k_col0 + 2] = x1_2.x;
        sK[fm_3 * 128 + k_col0 + 3] = x1_2.y;
#endif
        int v_col_2 = lane_local * 2;
        sV[fm_3 * 64 + v_col_2] = bv[6];
        sV[fm_3 * 64 + v_col_2 + 1] = bv[7];
        if (lane_local == 0) {
#if __CUDA_ARCH__ == 900
            float _exp_39 = __expf(-bb[3]);
            float _rcp_38 = approx_rcp(1.0f + _exp_39);
            sBeta[fm_3] = _rcp_38;
#else
            float _exp_31 = __expf(-bb[3]);
            float _rcp_30 = approx_rcp(1.0f + _exp_31);
            sBeta[fm_3] = _rcp_30;
#endif
        }
        float bk_0[8];
        float bv_0[4];
        float bb_0[2];
        int bm0_4 = 16 + warp_local;
        int bm_4 = bm0_4;
        long long bki_4 = (long long)n * k_stride_p0 + (long long)bm_4 * k_stride_p1 + (long long)h * k_stride_p2;
        {
#if __CUDA_ARCH__ == 900
            uint2 _vld_40;
            _vld_40 = *reinterpret_cast<const uint2*>(k + bki_4 + (long long)k_col0);
            uint32_t* _vpairs_40 = reinterpret_cast<uint32_t*>(&_vld_40);
#else
            uint2 _vld_32;
            _vld_32 = *reinterpret_cast<const uint2*>(k + bki_4 + (long long)k_col0);
            uint32_t* _vpairs_32 = reinterpret_cast<uint32_t*>(&_vld_32);
#endif
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
#if __CUDA_ARCH__ == 900
                (&bk_0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_40[_pair]) << 16);
                (&bk_0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_40[_pair]) & 0xffff0000u);
#else
                (&bk_0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) << 16);
                (&bk_0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) & 0xffff0000u);
#endif
            }
        }
        long long bvi_4 = (long long)n * v_stride_p0 + (long long)bm_4 * v_stride_p1 + (long long)h * v_stride_p2;
        int bvc_4 = lane_local * 2;
        bv_0[0] = (float)v[bvi_4 + (long long)bvc_4];
        bv_0[1] = (float)v[bvi_4 + (long long)bvc_4 + 1ll];
        long long bbi_4 = (long long)n * b_stride_p0 + (long long)bm_4 * b_stride_p1 + (long long)h * b_stride_p2;
        bb_0[0] = (float)b[bbi_4];
        int bm0_5 = 20 + warp_local;
        int bm_5 = ((bm0_5) < (steps - 1) ? (bm0_5) : (steps - 1));
        long long bki_5 = (long long)n * k_stride_p0 + (long long)bm_5 * k_stride_p1 + (long long)h * k_stride_p2;
        {
#if __CUDA_ARCH__ == 900
            uint2 _vld_41;
            _vld_41 = *reinterpret_cast<const uint2*>(k + bki_5 + (long long)k_col0);
            uint32_t* _vpairs_41 = reinterpret_cast<uint32_t*>(&_vld_41);
#else
            uint2 _vld_33;
            _vld_33 = *reinterpret_cast<const uint2*>(k + bki_5 + (long long)k_col0);
            uint32_t* _vpairs_33 = reinterpret_cast<uint32_t*>(&_vld_33);
#endif
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
#if __CUDA_ARCH__ == 900
                (&bk_0[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) << 16);
                (&bk_0[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) & 0xffff0000u);
#else
                (&bk_0[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) << 16);
                (&bk_0[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_33[_pair]) & 0xffff0000u);
#endif
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
#if __CUDA_ARCH__ == 900
            uint2 _vld_42;
            _vld_42 = *reinterpret_cast<const uint2*>(q + bqi_0 + (long long)k_col0);
            uint32_t* _vpairs_42 = reinterpret_cast<uint32_t*>(&_vld_42);
#else
            uint2 _vld_34;
            _vld_34 = *reinterpret_cast<const uint2*>(q + bqi_0 + (long long)k_col0);
            uint32_t* _vpairs_34 = reinterpret_cast<uint32_t*>(&_vld_34);
#endif
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
#if __CUDA_ARCH__ == 900
                (&bq[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) << 16);
                (&bq[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) & 0xffff0000u);
#else
                (&bq[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) << 16);
                (&bq[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) & 0xffff0000u);
#endif
            }
        }
        long long bgi_0 = (long long)n * a_stride_p0 + (long long)bt_0 * a_stride_p1 + (long long)h * a_stride_p2;
        ba[0] = (float)a[bgi_0];
        ba[1] = dt_bias[h];
        ba[2] = A_log[h];
        int bt0_1 = 4 + warp_local;
        int bt_1 = ((bt0_1) < (num_tokens - 1) ? (bt0_1) : (num_tokens - 1));
        long long bqi_1 = (long long)n * q_stride_p0 + (long long)bt_1 * q_stride_p1 + (long long)h * q_stride_p2;
        {
#if __CUDA_ARCH__ == 900
            uint2 _vld_43;
            _vld_43 = *reinterpret_cast<const uint2*>(q + bqi_1 + (long long)k_col0);
            uint32_t* _vpairs_43 = reinterpret_cast<uint32_t*>(&_vld_43);
#else
            uint2 _vld_35;
            _vld_35 = *reinterpret_cast<const uint2*>(q + bqi_1 + (long long)k_col0);
            uint32_t* _vpairs_35 = reinterpret_cast<uint32_t*>(&_vld_35);
#endif
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
#if __CUDA_ARCH__ == 900
                (&bq[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) << 16);
                (&bq[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) & 0xffff0000u);
#else
                (&bq[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_35[_pair]) << 16);
                (&bq[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_35[_pair]) & 0xffff0000u);
#endif
            }
        }
        long long bgi_1 = (long long)n * a_stride_p0 + (long long)bt_1 * a_stride_p1 + (long long)h * a_stride_p2;
        ba[3] = (float)a[bgi_1];
        ba[4] = dt_bias[h];
        ba[5] = A_log[h];
        int fm_4 = 16 + warp_local;
#if __CUDA_ARCH__ == 900
        float x0_x_3 = bk_0[0];
        float x0_y_3 = bk_0[1];
        float x1_x_3 = bk_0[2];
        float x1_y_3 = bk_0[3];
        float sum_x_3;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_3) : "f"(x0_x_3), "f"(x0_x_3), "f"(0.0f));
        float sum_y_3;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_3) : "f"(x0_y_3), "f"(x0_y_3), "f"(0.0f));
        float _fma_44;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(x1_x_3), "f"(x1_x_3), "f"(sum_x_3));
        sum_x_3 = _fma_44;
        float _fma_45;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(x1_y_3), "f"(x1_y_3), "f"(sum_y_3));
        sum_y_3 = _fma_45;
        sum_x_3 = sum_x_3 + sum_y_3;
        float _warp_reduce_46 = sum_x_3;
#else
        float2 x0_3 = make_float2(bk_0[0], bk_0[1]);
        float2 x1_3 = make_float2(bk_0[2], bk_0[3]);
        float2 sum_pair_3 = fma_f32x2_rn_ftz(x0_3, x0_3, make_float2(0.0f, 0.0f));
        sum_pair_3 = fma_f32x2_rn_ftz(x1_3, x1_3, sum_pair_3);
        float sum_x_3 = sum_pair_3.x + sum_pair_3.y;
        float _warp_reduce_36 = sum_x_3;
#endif
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
            _warp_reduce_46 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_46, offset);
        float norm_3 = rsqrtf(_warp_reduce_46 + 1e-06f);
        x0_x_3 = mul_f32_rn_ftz(x0_x_3, norm_3);
        x0_y_3 = mul_f32_rn_ftz(x0_y_3, norm_3);
        x1_x_3 = mul_f32_rn_ftz(x1_x_3, norm_3);
        x1_y_3 = mul_f32_rn_ftz(x1_y_3, norm_3);
        sK[fm_4 * 128 + k_col0] = x0_x_3;
        sK[fm_4 * 128 + k_col0 + 1] = x0_y_3;
        sK[fm_4 * 128 + k_col0 + 2] = x1_x_3;
        sK[fm_4 * 128 + k_col0 + 3] = x1_y_3;
#else
            _warp_reduce_36 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_36, offset);
        float norm_3 = rsqrtf(_warp_reduce_36 + 1e-06f);
        float2 norm2_3 = make_float2(norm_3, norm_3);
        x0_3 = mul_f32x2(x0_3, norm2_3);
        x1_3 = mul_f32x2(x1_3, norm2_3);
        sK[fm_4 * 128 + k_col0] = x0_3.x;
        sK[fm_4 * 128 + k_col0 + 1] = x0_3.y;
        sK[fm_4 * 128 + k_col0 + 2] = x1_3.x;
        sK[fm_4 * 128 + k_col0 + 3] = x1_3.y;
#endif
        int v_col_3 = lane_local * 2;
        sV[fm_4 * 64 + v_col_3] = bv_0[0];
        sV[fm_4 * 64 + v_col_3 + 1] = bv_0[1];
        if (lane_local == 0) {
#if __CUDA_ARCH__ == 900
            float _exp_48 = __expf(-bb_0[0]);
            float _rcp_47 = approx_rcp(1.0f + _exp_48);
            sBeta[fm_4] = _rcp_47;
#else
            float _exp_38 = __expf(-bb_0[0]);
            float _rcp_37 = approx_rcp(1.0f + _exp_38);
            sBeta[fm_4] = _rcp_37;
#endif
        }
        int fm_5 = 20 + warp_local;
        if (fm_5 < steps) {
#if __CUDA_ARCH__ == 900
            float x0_x_4 = bk_0[4];
            float x0_y_4 = bk_0[5];
            float x1_x_4 = bk_0[6];
            float x1_y_4 = bk_0[7];
            float sum_x_4;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_4) : "f"(x0_x_4), "f"(x0_x_4), "f"(0.0f));
            float sum_y_4;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_4) : "f"(x0_y_4), "f"(x0_y_4), "f"(0.0f));
            float _fma_49;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(x1_x_4), "f"(x1_x_4), "f"(sum_x_4));
            sum_x_4 = _fma_49;
            float _fma_50;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(x1_y_4), "f"(x1_y_4), "f"(sum_y_4));
            sum_y_4 = _fma_50;
            sum_x_4 = sum_x_4 + sum_y_4;
            float _warp_reduce_51 = sum_x_4;
#else
            float2 x0_4 = make_float2(bk_0[4], bk_0[5]);
            float2 x1_4 = make_float2(bk_0[6], bk_0[7]);
            float2 sum_pair_4 = fma_f32x2_rn_ftz(x0_4, x0_4, make_float2(0.0f, 0.0f));
            sum_pair_4 = fma_f32x2_rn_ftz(x1_4, x1_4, sum_pair_4);
            float sum_x_4 = sum_pair_4.x + sum_pair_4.y;
            float _warp_reduce_39 = sum_x_4;
#endif
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
                _warp_reduce_51 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_51, offset);
            float norm_4 = rsqrtf(_warp_reduce_51 + 1e-06f);
            x0_x_4 = mul_f32_rn_ftz(x0_x_4, norm_4);
            x0_y_4 = mul_f32_rn_ftz(x0_y_4, norm_4);
            x1_x_4 = mul_f32_rn_ftz(x1_x_4, norm_4);
            x1_y_4 = mul_f32_rn_ftz(x1_y_4, norm_4);
            sK[fm_5 * 128 + k_col0] = x0_x_4;
            sK[fm_5 * 128 + k_col0 + 1] = x0_y_4;
            sK[fm_5 * 128 + k_col0 + 2] = x1_x_4;
            sK[fm_5 * 128 + k_col0 + 3] = x1_y_4;
#else
                _warp_reduce_39 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_39, offset);
            float norm_4 = rsqrtf(_warp_reduce_39 + 1e-06f);
            float2 norm2_4 = make_float2(norm_4, norm_4);
            x0_4 = mul_f32x2(x0_4, norm2_4);
            x1_4 = mul_f32x2(x1_4, norm2_4);
            sK[fm_5 * 128 + k_col0] = x0_4.x;
            sK[fm_5 * 128 + k_col0 + 1] = x0_4.y;
            sK[fm_5 * 128 + k_col0 + 2] = x1_4.x;
            sK[fm_5 * 128 + k_col0 + 3] = x1_4.y;
#endif
            int v_col_4 = lane_local * 2;
            sV[fm_5 * 64 + v_col_4] = bv_0[2];
            sV[fm_5 * 64 + v_col_4 + 1] = bv_0[3];
            if (lane_local == 0) {
#if __CUDA_ARCH__ == 900
                float _exp_53 = __expf(-bb_0[1]);
                float _rcp_52 = approx_rcp(1.0f + _exp_53);
                sBeta[fm_5] = _rcp_52;
#else
                float _exp_41 = __expf(-bb_0[1]);
                float _rcp_40 = approx_rcp(1.0f + _exp_41);
                sBeta[fm_5] = _rcp_40;
#endif
            }
        }
        int ft_0 = warp_local;
#if __CUDA_ARCH__ == 900
        float x0_x_5 = bq[0];
        float x0_y_5 = bq[1];
        float x1_x_5 = bq[2];
        float x1_y_5 = bq[3];
        float sum_x_5;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_5) : "f"(x0_x_5), "f"(x0_x_5), "f"(0.0f));
        float sum_y_5;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_5) : "f"(x0_y_5), "f"(x0_y_5), "f"(0.0f));
        float _fma_54;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(x1_x_5), "f"(x1_x_5), "f"(sum_x_5));
        sum_x_5 = _fma_54;
        float _fma_55;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(x1_y_5), "f"(x1_y_5), "f"(sum_y_5));
        sum_y_5 = _fma_55;
        sum_x_5 = sum_x_5 + sum_y_5;
        float _warp_reduce_56 = sum_x_5;
#else
        float2 x0_5 = make_float2(bq[0], bq[1]);
        float2 x1_5 = make_float2(bq[2], bq[3]);
        float2 sum_pair_5 = fma_f32x2_rn_ftz(x0_5, x0_5, make_float2(0.0f, 0.0f));
        sum_pair_5 = fma_f32x2_rn_ftz(x1_5, x1_5, sum_pair_5);
        float sum_x_5 = sum_pair_5.x + sum_pair_5.y;
        float _warp_reduce_42 = sum_x_5;
#endif
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
            _warp_reduce_56 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_56, offset);
        float norm_5 = rsqrtf(_warp_reduce_56 + 1e-06f);
#else
            _warp_reduce_42 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_42, offset);
        float norm_5 = rsqrtf(_warp_reduce_42 + 1e-06f);
#endif
        norm_5 = norm_5 * scale;
#if __CUDA_ARCH__ == 900
        x0_x_5 = mul_f32_rn_ftz(x0_x_5, norm_5);
        x0_y_5 = mul_f32_rn_ftz(x0_y_5, norm_5);
        x1_x_5 = mul_f32_rn_ftz(x1_x_5, norm_5);
        x1_y_5 = mul_f32_rn_ftz(x1_y_5, norm_5);
        sQ[ft_0 * 128 + k_col0] = x0_x_5;
        sQ[ft_0 * 128 + k_col0 + 1] = x0_y_5;
        sQ[ft_0 * 128 + k_col0 + 2] = x1_x_5;
        sQ[ft_0 * 128 + k_col0 + 3] = x1_y_5;
#else
        float2 norm2_5 = make_float2(norm_5, norm_5);
        x0_5 = mul_f32x2(x0_5, norm2_5);
        x1_5 = mul_f32x2(x1_5, norm2_5);
        sQ[ft_0 * 128 + k_col0] = x0_5.x;
        sQ[ft_0 * 128 + k_col0 + 1] = x0_5.y;
        sQ[ft_0 * 128 + k_col0 + 2] = x1_5.x;
        sQ[ft_0 * 128 + k_col0 + 3] = x1_5.y;
#endif
        if (lane_local == 0) {
            float xv = ba[0] + ba[1];
            float softplus_x = xv;
            if (xv <= 20.0f) {
#if __CUDA_ARCH__ == 900
                float _exp_58 = __expf(xv);
                float _log2_57;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_57) : "f"(1.0f + _exp_58));
                softplus_x = _log2_57 * 0.6931472f;
#else
                float _exp_44 = __expf(xv);
                float _log2_43;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_43) : "f"(1.0f + _exp_44));
                softplus_x = _log2_43 * 0.6931472f;
#endif
            }
#if __CUDA_ARCH__ == 900
            float _exp_59 = __expf(ba[2]);
            float g_log = -_exp_59 * softplus_x;
            float _exp_60 = __expf(g_log);
            sAlpha[ft_0] = _exp_60;
#else
            float _exp_45 = __expf(ba[2]);
            float g_log = -_exp_45 * softplus_x;
            float _exp_46 = __expf(g_log);
            sAlpha[ft_0] = _exp_46;
#endif
            if (use_ssm_state_indices != 0) {
                sSsm[ft_0] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)ft_0 * ssm_state_indices_stride_p1];
            }
            if (use_ssm_state_indices == 0) {
                sSsm[ft_0] = -1;
            }
        }
        int ft_1 = 4 + warp_local;
        if (ft_1 < num_tokens) {
#if __CUDA_ARCH__ == 900
            float x0_x_6 = bq[4];
            float x0_y_6 = bq[5];
            float x1_x_6 = bq[6];
            float x1_y_6 = bq[7];
            float sum_x_6;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_6) : "f"(x0_x_6), "f"(x0_x_6), "f"(0.0f));
            float sum_y_6;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_6) : "f"(x0_y_6), "f"(x0_y_6), "f"(0.0f));
            float _fma_61;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(x1_x_6), "f"(x1_x_6), "f"(sum_x_6));
            sum_x_6 = _fma_61;
            float _fma_62;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(x1_y_6), "f"(x1_y_6), "f"(sum_y_6));
            sum_y_6 = _fma_62;
            sum_x_6 = sum_x_6 + sum_y_6;
            float _warp_reduce_63 = sum_x_6;
#else
            float2 x0_6 = make_float2(bq[4], bq[5]);
            float2 x1_6 = make_float2(bq[6], bq[7]);
            float2 sum_pair_6 = fma_f32x2_rn_ftz(x0_6, x0_6, make_float2(0.0f, 0.0f));
            sum_pair_6 = fma_f32x2_rn_ftz(x1_6, x1_6, sum_pair_6);
            float sum_x_6 = sum_pair_6.x + sum_pair_6.y;
            float _warp_reduce_47 = sum_x_6;
#endif
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
                _warp_reduce_63 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_63, offset);
            float norm_6 = rsqrtf(_warp_reduce_63 + 1e-06f);
#else
                _warp_reduce_47 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_47, offset);
            float norm_6 = rsqrtf(_warp_reduce_47 + 1e-06f);
#endif
            norm_6 = norm_6 * scale;
#if __CUDA_ARCH__ == 900
            x0_x_6 = mul_f32_rn_ftz(x0_x_6, norm_6);
            x0_y_6 = mul_f32_rn_ftz(x0_y_6, norm_6);
            x1_x_6 = mul_f32_rn_ftz(x1_x_6, norm_6);
            x1_y_6 = mul_f32_rn_ftz(x1_y_6, norm_6);
            sQ[ft_1 * 128 + k_col0] = x0_x_6;
            sQ[ft_1 * 128 + k_col0 + 1] = x0_y_6;
            sQ[ft_1 * 128 + k_col0 + 2] = x1_x_6;
            sQ[ft_1 * 128 + k_col0 + 3] = x1_y_6;
#else
            float2 norm2_6 = make_float2(norm_6, norm_6);
            x0_6 = mul_f32x2(x0_6, norm2_6);
            x1_6 = mul_f32x2(x1_6, norm2_6);
            sQ[ft_1 * 128 + k_col0] = x0_6.x;
            sQ[ft_1 * 128 + k_col0 + 1] = x0_6.y;
            sQ[ft_1 * 128 + k_col0 + 2] = x1_6.x;
            sQ[ft_1 * 128 + k_col0 + 3] = x1_6.y;
#endif
            if (lane_local == 0) {
                float xv_1 = ba[3] + ba[4];
                float softplus_x_1 = xv_1;
                if (xv_1 <= 20.0f) {
#if __CUDA_ARCH__ == 900
                    float _exp_65 = __expf(xv_1);
                    float _log2_64;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_64) : "f"(1.0f + _exp_65));
                    softplus_x_1 = _log2_64 * 0.6931472f;
#else
                    float _exp_49 = __expf(xv_1);
                    float _log2_48;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_48) : "f"(1.0f + _exp_49));
                    softplus_x_1 = _log2_48 * 0.6931472f;
#endif
                }
#if __CUDA_ARCH__ == 900
                float _exp_66 = __expf(ba[5]);
                float g_log_1 = -_exp_66 * softplus_x_1;
                float _exp_67 = __expf(g_log_1);
                sAlpha[ft_1] = _exp_67;
#else
                float _exp_50 = __expf(ba[5]);
                float g_log_1 = -_exp_50 * softplus_x_1;
                float _exp_51 = __expf(g_log_1);
                sAlpha[ft_1] = _exp_51;
#endif
                if (use_ssm_state_indices != 0) {
                    sSsm[ft_1] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)ft_1 * ssm_state_indices_stride_p1];
                }
                if (use_ssm_state_indices == 0) {
                    sSsm[ft_1] = -1;
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
#endif
        if (use_ssm_state_indices != 0) {
            #pragma unroll 1
            for (int t = 0; t < 7; t++) {
                float alpha = sAlpha[t];
#if __CUDA_ARCH__ == 900
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
                S[32] = mul_f32_rn_ftz(S[32], alpha);
                S[33] = mul_f32_rn_ftz(S[33], alpha);
                S[34] = mul_f32_rn_ftz(S[34], alpha);
                S[35] = mul_f32_rn_ftz(S[35], alpha);
                S[36] = mul_f32_rn_ftz(S[36], alpha);
                S[37] = mul_f32_rn_ftz(S[37], alpha);
                S[38] = mul_f32_rn_ftz(S[38], alpha);
                S[39] = mul_f32_rn_ftz(S[39], alpha);
                S[40] = mul_f32_rn_ftz(S[40], alpha);
                S[41] = mul_f32_rn_ftz(S[41], alpha);
                S[42] = mul_f32_rn_ftz(S[42], alpha);
                S[43] = mul_f32_rn_ftz(S[43], alpha);
                S[44] = mul_f32_rn_ftz(S[44], alpha);
                S[45] = mul_f32_rn_ftz(S[45], alpha);
                S[46] = mul_f32_rn_ftz(S[46], alpha);
                S[47] = mul_f32_rn_ftz(S[47], alpha);
                S[48] = mul_f32_rn_ftz(S[48], alpha);
                S[49] = mul_f32_rn_ftz(S[49], alpha);
                S[50] = mul_f32_rn_ftz(S[50], alpha);
                S[51] = mul_f32_rn_ftz(S[51], alpha);
                S[52] = mul_f32_rn_ftz(S[52], alpha);
                S[53] = mul_f32_rn_ftz(S[53], alpha);
                S[54] = mul_f32_rn_ftz(S[54], alpha);
                S[55] = mul_f32_rn_ftz(S[55], alpha);
                S[56] = mul_f32_rn_ftz(S[56], alpha);
                S[57] = mul_f32_rn_ftz(S[57], alpha);
                S[58] = mul_f32_rn_ftz(S[58], alpha);
                S[59] = mul_f32_rn_ftz(S[59], alpha);
                S[60] = mul_f32_rn_ftz(S[60], alpha);
                S[61] = mul_f32_rn_ftz(S[61], alpha);
                S[62] = mul_f32_rn_ftz(S[62], alpha);
                S[63] = mul_f32_rn_ftz(S[63], alpha);
#else
                float2 alpha2 = make_float2(alpha, alpha);
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
                S2[16] = mul_f32x2(S2[16], alpha2);
                S2[17] = mul_f32x2(S2[17], alpha2);
                S2[18] = mul_f32x2(S2[18], alpha2);
                S2[19] = mul_f32x2(S2[19], alpha2);
                S2[20] = mul_f32x2(S2[20], alpha2);
                S2[21] = mul_f32x2(S2[21], alpha2);
                S2[22] = mul_f32x2(S2[22], alpha2);
                S2[23] = mul_f32x2(S2[23], alpha2);
                S2[24] = mul_f32x2(S2[24], alpha2);
                S2[25] = mul_f32x2(S2[25], alpha2);
                S2[26] = mul_f32x2(S2[26], alpha2);
                S2[27] = mul_f32x2(S2[27], alpha2);
                S2[28] = mul_f32x2(S2[28], alpha2);
                S2[29] = mul_f32x2(S2[29], alpha2);
                S2[30] = mul_f32x2(S2[30], alpha2);
                S2[31] = mul_f32x2(S2[31], alpha2);
#endif
                int m_0 = t * 3;
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
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                    : "r"(sV_addr + (unsigned int)(m_0 * 256) + group_byte));
                float beta_0 = sBeta[m_0];
#if __CUDA_ARCH__ == 900
                float acc0_x = 0.0f;
                float acc0_y = 0.0f;
                float _fma_68;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x));
                acc0_x = _fma_68;
                float _fma_69;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y));
                acc0_y = _fma_69;
                float _fma_70;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(S[2]), "f"(k_regs[2]), "f"(acc0_x));
                acc0_x = _fma_70;
                float _fma_71;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(S[3]), "f"(k_regs[3]), "f"(acc0_y));
                acc0_y = _fma_71;
                float _fma_72;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x));
                acc0_x = _fma_72;
                float _fma_73;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y));
                acc0_y = _fma_73;
                float _fma_74;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(S[6]), "f"(k_regs[6]), "f"(acc0_x));
                acc0_x = _fma_74;
                float _fma_75;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(S[7]), "f"(k_regs[7]), "f"(acc0_y));
                acc0_y = _fma_75;
                float _fma_76;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x));
                acc0_x = _fma_76;
                float _fma_77;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y));
                acc0_y = _fma_77;
                float _fma_78;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_78) : "f"(S[10]), "f"(k_regs[10]), "f"(acc0_x));
                acc0_x = _fma_78;
                float _fma_79;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(S[11]), "f"(k_regs[11]), "f"(acc0_y));
                acc0_y = _fma_79;
                float _fma_80;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x));
                acc0_x = _fma_80;
                float _fma_81;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y));
                acc0_y = _fma_81;
                float _fma_82;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(S[14]), "f"(k_regs[14]), "f"(acc0_x));
                acc0_x = _fma_82;
                float _fma_83;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(S[15]), "f"(k_regs[15]), "f"(acc0_y));
                acc0_y = _fma_83;
                float dot = acc0_x + acc0_y;
                float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
                dot = dot + _shfl_xor_sync_84;
                float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
                dot = dot + _shfl_xor_sync_85;
                float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
                dot = dot + _shfl_xor_sync_86;
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
                float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
                dot = dot + _shfl_xor_sync_52;
                float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
                dot = dot + _shfl_xor_sync_53;
                float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
                dot = dot + _shfl_xor_sync_54;
#endif
                float vn_0 = (v_regs[0] - dot) * beta_0;
#if __CUDA_ARCH__ == 900
                float _fma_87;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(k_regs[0]), "f"(vn_0), "f"(S[0]));
                S[0] = _fma_87;
                float _fma_88;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(k_regs[1]), "f"(vn_0), "f"(S[1]));
                S[1] = _fma_88;
                float _fma_89;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(k_regs[2]), "f"(vn_0), "f"(S[2]));
                S[2] = _fma_89;
                float _fma_90;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(k_regs[3]), "f"(vn_0), "f"(S[3]));
                S[3] = _fma_90;
                float _fma_91;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(k_regs[4]), "f"(vn_0), "f"(S[4]));
                S[4] = _fma_91;
                float _fma_92;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(k_regs[5]), "f"(vn_0), "f"(S[5]));
                S[5] = _fma_92;
                float _fma_93;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(k_regs[6]), "f"(vn_0), "f"(S[6]));
                S[6] = _fma_93;
                float _fma_94;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(k_regs[7]), "f"(vn_0), "f"(S[7]));
                S[7] = _fma_94;
                float _fma_95;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(k_regs[8]), "f"(vn_0), "f"(S[8]));
                S[8] = _fma_95;
                float _fma_96;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(k_regs[9]), "f"(vn_0), "f"(S[9]));
                S[9] = _fma_96;
                float _fma_97;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(k_regs[10]), "f"(vn_0), "f"(S[10]));
                S[10] = _fma_97;
                float _fma_98;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(k_regs[11]), "f"(vn_0), "f"(S[11]));
                S[11] = _fma_98;
                float _fma_99;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(k_regs[12]), "f"(vn_0), "f"(S[12]));
                S[12] = _fma_99;
                float _fma_100;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(k_regs[13]), "f"(vn_0), "f"(S[13]));
                S[13] = _fma_100;
                float _fma_101;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(k_regs[14]), "f"(vn_0), "f"(S[14]));
                S[14] = _fma_101;
                float _fma_102;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(k_regs[15]), "f"(vn_0), "f"(S[15]));
                S[15] = _fma_102;
                float acc0_x_0 = 0.0f;
                float acc0_y_0 = 0.0f;
                float _fma_103;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_0));
                acc0_x_0 = _fma_103;
                float _fma_104;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_0));
                acc0_y_0 = _fma_104;
                float _fma_105;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(S[18]), "f"(k_regs[2]), "f"(acc0_x_0));
                acc0_x_0 = _fma_105;
                float _fma_106;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(S[19]), "f"(k_regs[3]), "f"(acc0_y_0));
                acc0_y_0 = _fma_106;
                float _fma_107;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_0));
                acc0_x_0 = _fma_107;
                float _fma_108;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_0));
                acc0_y_0 = _fma_108;
                float _fma_109;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(S[22]), "f"(k_regs[6]), "f"(acc0_x_0));
                acc0_x_0 = _fma_109;
                float _fma_110;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(S[23]), "f"(k_regs[7]), "f"(acc0_y_0));
                acc0_y_0 = _fma_110;
                float _fma_111;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_111) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_0));
                acc0_x_0 = _fma_111;
                float _fma_112;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_112) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_0));
                acc0_y_0 = _fma_112;
                float _fma_113;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_113) : "f"(S[26]), "f"(k_regs[10]), "f"(acc0_x_0));
                acc0_x_0 = _fma_113;
                float _fma_114;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_114) : "f"(S[27]), "f"(k_regs[11]), "f"(acc0_y_0));
                acc0_y_0 = _fma_114;
                float _fma_115;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_0));
                acc0_x_0 = _fma_115;
                float _fma_116;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_0));
                acc0_y_0 = _fma_116;
                float _fma_117;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(S[30]), "f"(k_regs[14]), "f"(acc0_x_0));
                acc0_x_0 = _fma_117;
                float _fma_118;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_118) : "f"(S[31]), "f"(k_regs[15]), "f"(acc0_y_0));
                acc0_y_0 = _fma_118;
                float dot_0 = acc0_x_0 + acc0_y_0;
                float _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 1);
                dot_0 = dot_0 + _shfl_xor_sync_119;
                float _shfl_xor_sync_120 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 2);
                dot_0 = dot_0 + _shfl_xor_sync_120;
                float _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 4);
                dot_0 = dot_0 + _shfl_xor_sync_121;
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
                float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 1);
                dot_0 = dot_0 + _shfl_xor_sync_55;
                float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 2);
                dot_0 = dot_0 + _shfl_xor_sync_56;
                float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 4);
                dot_0 = dot_0 + _shfl_xor_sync_57;
#endif
                float vn_1 = (v_regs[1] - dot_0) * beta_0;
#if __CUDA_ARCH__ == 900
                float _fma_122;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(k_regs[0]), "f"(vn_1), "f"(S[16]));
                S[16] = _fma_122;
                float _fma_123;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_123) : "f"(k_regs[1]), "f"(vn_1), "f"(S[17]));
                S[17] = _fma_123;
                float _fma_124;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(k_regs[2]), "f"(vn_1), "f"(S[18]));
                S[18] = _fma_124;
                float _fma_125;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(k_regs[3]), "f"(vn_1), "f"(S[19]));
                S[19] = _fma_125;
                float _fma_126;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(k_regs[4]), "f"(vn_1), "f"(S[20]));
                S[20] = _fma_126;
                float _fma_127;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(k_regs[5]), "f"(vn_1), "f"(S[21]));
                S[21] = _fma_127;
                float _fma_128;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_128) : "f"(k_regs[6]), "f"(vn_1), "f"(S[22]));
                S[22] = _fma_128;
                float _fma_129;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_129) : "f"(k_regs[7]), "f"(vn_1), "f"(S[23]));
                S[23] = _fma_129;
                float _fma_130;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_130) : "f"(k_regs[8]), "f"(vn_1), "f"(S[24]));
                S[24] = _fma_130;
                float _fma_131;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(k_regs[9]), "f"(vn_1), "f"(S[25]));
                S[25] = _fma_131;
                float _fma_132;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(k_regs[10]), "f"(vn_1), "f"(S[26]));
                S[26] = _fma_132;
                float _fma_133;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(k_regs[11]), "f"(vn_1), "f"(S[27]));
                S[27] = _fma_133;
                float _fma_134;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_134) : "f"(k_regs[12]), "f"(vn_1), "f"(S[28]));
                S[28] = _fma_134;
                float _fma_135;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_135) : "f"(k_regs[13]), "f"(vn_1), "f"(S[29]));
                S[29] = _fma_135;
                float _fma_136;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_136) : "f"(k_regs[14]), "f"(vn_1), "f"(S[30]));
                S[30] = _fma_136;
                float _fma_137;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_137) : "f"(k_regs[15]), "f"(vn_1), "f"(S[31]));
                S[31] = _fma_137;
                float acc0_x_1 = 0.0f;
                float acc0_y_1 = 0.0f;
                float _fma_138;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_138) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_1));
                acc0_x_1 = _fma_138;
                float _fma_139;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_139) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_1));
                acc0_y_1 = _fma_139;
                float _fma_140;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_140) : "f"(S[34]), "f"(k_regs[2]), "f"(acc0_x_1));
                acc0_x_1 = _fma_140;
                float _fma_141;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_141) : "f"(S[35]), "f"(k_regs[3]), "f"(acc0_y_1));
                acc0_y_1 = _fma_141;
                float _fma_142;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_1));
                acc0_x_1 = _fma_142;
                float _fma_143;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_143) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_1));
                acc0_y_1 = _fma_143;
                float _fma_144;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_144) : "f"(S[38]), "f"(k_regs[6]), "f"(acc0_x_1));
                acc0_x_1 = _fma_144;
                float _fma_145;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_145) : "f"(S[39]), "f"(k_regs[7]), "f"(acc0_y_1));
                acc0_y_1 = _fma_145;
                float _fma_146;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_146) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_1));
                acc0_x_1 = _fma_146;
                float _fma_147;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_147) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_1));
                acc0_y_1 = _fma_147;
                float _fma_148;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_148) : "f"(S[42]), "f"(k_regs[10]), "f"(acc0_x_1));
                acc0_x_1 = _fma_148;
                float _fma_149;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_149) : "f"(S[43]), "f"(k_regs[11]), "f"(acc0_y_1));
                acc0_y_1 = _fma_149;
                float _fma_150;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_150) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_1));
                acc0_x_1 = _fma_150;
                float _fma_151;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_151) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_1));
                acc0_y_1 = _fma_151;
                float _fma_152;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_152) : "f"(S[46]), "f"(k_regs[14]), "f"(acc0_x_1));
                acc0_x_1 = _fma_152;
                float _fma_153;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_153) : "f"(S[47]), "f"(k_regs[15]), "f"(acc0_y_1));
                acc0_y_1 = _fma_153;
                float dot_1 = acc0_x_1 + acc0_y_1;
                float _shfl_xor_sync_154 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_154;
                float _shfl_xor_sync_155 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_155;
                float _shfl_xor_sync_156 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
                dot_1 = dot_1 + _shfl_xor_sync_156;
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
                float2 acc0_1 = make_float2(0.0f, 0.0f);
                acc0_1 = fma_f32x2_rn_ftz(S2[16], k2[0], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[17], k2[1], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[18], k2[2], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[19], k2[3], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[20], k2[4], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[21], k2[5], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[22], k2[6], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[23], k2[7], acc0_1);
                float dot_1 = acc0_1.x + acc0_1.y;
                float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_58;
                float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_59;
                float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
                dot_1 = dot_1 + _shfl_xor_sync_60;
#endif
                float vn_2 = (v_regs[2] - dot_1) * beta_0;
#if __CUDA_ARCH__ == 900
                float _fma_157;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(k_regs[0]), "f"(vn_2), "f"(S[32]));
                S[32] = _fma_157;
                float _fma_158;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_158) : "f"(k_regs[1]), "f"(vn_2), "f"(S[33]));
                S[33] = _fma_158;
                float _fma_159;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_159) : "f"(k_regs[2]), "f"(vn_2), "f"(S[34]));
                S[34] = _fma_159;
                float _fma_160;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_160) : "f"(k_regs[3]), "f"(vn_2), "f"(S[35]));
                S[35] = _fma_160;
                float _fma_161;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_161) : "f"(k_regs[4]), "f"(vn_2), "f"(S[36]));
                S[36] = _fma_161;
                float _fma_162;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_162) : "f"(k_regs[5]), "f"(vn_2), "f"(S[37]));
                S[37] = _fma_162;
                float _fma_163;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_163) : "f"(k_regs[6]), "f"(vn_2), "f"(S[38]));
                S[38] = _fma_163;
                float _fma_164;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_164) : "f"(k_regs[7]), "f"(vn_2), "f"(S[39]));
                S[39] = _fma_164;
                float _fma_165;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_165) : "f"(k_regs[8]), "f"(vn_2), "f"(S[40]));
                S[40] = _fma_165;
                float _fma_166;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_166) : "f"(k_regs[9]), "f"(vn_2), "f"(S[41]));
                S[41] = _fma_166;
                float _fma_167;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_167) : "f"(k_regs[10]), "f"(vn_2), "f"(S[42]));
                S[42] = _fma_167;
                float _fma_168;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_168) : "f"(k_regs[11]), "f"(vn_2), "f"(S[43]));
                S[43] = _fma_168;
                float _fma_169;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_169) : "f"(k_regs[12]), "f"(vn_2), "f"(S[44]));
                S[44] = _fma_169;
                float _fma_170;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_170) : "f"(k_regs[13]), "f"(vn_2), "f"(S[45]));
                S[45] = _fma_170;
                float _fma_171;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_171) : "f"(k_regs[14]), "f"(vn_2), "f"(S[46]));
                S[46] = _fma_171;
                float _fma_172;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_172) : "f"(k_regs[15]), "f"(vn_2), "f"(S[47]));
                S[47] = _fma_172;
                float acc0_x_2 = 0.0f;
                float acc0_y_2 = 0.0f;
                float _fma_173;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_173) : "f"(S[48]), "f"(k_regs[0]), "f"(acc0_x_2));
                acc0_x_2 = _fma_173;
                float _fma_174;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_174) : "f"(S[49]), "f"(k_regs[1]), "f"(acc0_y_2));
                acc0_y_2 = _fma_174;
                float _fma_175;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_175) : "f"(S[50]), "f"(k_regs[2]), "f"(acc0_x_2));
                acc0_x_2 = _fma_175;
                float _fma_176;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_176) : "f"(S[51]), "f"(k_regs[3]), "f"(acc0_y_2));
                acc0_y_2 = _fma_176;
                float _fma_177;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_177) : "f"(S[52]), "f"(k_regs[4]), "f"(acc0_x_2));
                acc0_x_2 = _fma_177;
                float _fma_178;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_178) : "f"(S[53]), "f"(k_regs[5]), "f"(acc0_y_2));
                acc0_y_2 = _fma_178;
                float _fma_179;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_179) : "f"(S[54]), "f"(k_regs[6]), "f"(acc0_x_2));
                acc0_x_2 = _fma_179;
                float _fma_180;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_180) : "f"(S[55]), "f"(k_regs[7]), "f"(acc0_y_2));
                acc0_y_2 = _fma_180;
                float _fma_181;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_181) : "f"(S[56]), "f"(k_regs[8]), "f"(acc0_x_2));
                acc0_x_2 = _fma_181;
                float _fma_182;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_182) : "f"(S[57]), "f"(k_regs[9]), "f"(acc0_y_2));
                acc0_y_2 = _fma_182;
                float _fma_183;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_183) : "f"(S[58]), "f"(k_regs[10]), "f"(acc0_x_2));
                acc0_x_2 = _fma_183;
                float _fma_184;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_184) : "f"(S[59]), "f"(k_regs[11]), "f"(acc0_y_2));
                acc0_y_2 = _fma_184;
                float _fma_185;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_185) : "f"(S[60]), "f"(k_regs[12]), "f"(acc0_x_2));
                acc0_x_2 = _fma_185;
                float _fma_186;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_186) : "f"(S[61]), "f"(k_regs[13]), "f"(acc0_y_2));
                acc0_y_2 = _fma_186;
                float _fma_187;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_187) : "f"(S[62]), "f"(k_regs[14]), "f"(acc0_x_2));
                acc0_x_2 = _fma_187;
                float _fma_188;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_188) : "f"(S[63]), "f"(k_regs[15]), "f"(acc0_y_2));
                acc0_y_2 = _fma_188;
                float dot_2 = acc0_x_2 + acc0_y_2;
                float _shfl_xor_sync_189 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 1);
                dot_2 = dot_2 + _shfl_xor_sync_189;
                float _shfl_xor_sync_190 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 2);
                dot_2 = dot_2 + _shfl_xor_sync_190;
                float _shfl_xor_sync_191 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 4);
                dot_2 = dot_2 + _shfl_xor_sync_191;
#else
                float2 vn2_2 = make_float2(vn_2, vn_2);
                S2[16] = fma_f32x2_rn_ftz(k2[0], vn2_2, S2[16]);
                S2[17] = fma_f32x2_rn_ftz(k2[1], vn2_2, S2[17]);
                S2[18] = fma_f32x2_rn_ftz(k2[2], vn2_2, S2[18]);
                S2[19] = fma_f32x2_rn_ftz(k2[3], vn2_2, S2[19]);
                S2[20] = fma_f32x2_rn_ftz(k2[4], vn2_2, S2[20]);
                S2[21] = fma_f32x2_rn_ftz(k2[5], vn2_2, S2[21]);
                S2[22] = fma_f32x2_rn_ftz(k2[6], vn2_2, S2[22]);
                S2[23] = fma_f32x2_rn_ftz(k2[7], vn2_2, S2[23]);
                float2 acc0_2 = make_float2(0.0f, 0.0f);
                acc0_2 = fma_f32x2_rn_ftz(S2[24], k2[0], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[25], k2[1], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[26], k2[2], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[27], k2[3], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[28], k2[4], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[29], k2[5], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[30], k2[6], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[31], k2[7], acc0_2);
                float dot_2 = acc0_2.x + acc0_2.y;
                float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 1);
                dot_2 = dot_2 + _shfl_xor_sync_61;
                float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 2);
                dot_2 = dot_2 + _shfl_xor_sync_62;
                float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 4);
                dot_2 = dot_2 + _shfl_xor_sync_63;
#endif
                float vn_3 = (v_regs[3] - dot_2) * beta_0;
#if __CUDA_ARCH__ == 900
                float _fma_192;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_192) : "f"(k_regs[0]), "f"(vn_3), "f"(S[48]));
                S[48] = _fma_192;
                float _fma_193;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_193) : "f"(k_regs[1]), "f"(vn_3), "f"(S[49]));
                S[49] = _fma_193;
                float _fma_194;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_194) : "f"(k_regs[2]), "f"(vn_3), "f"(S[50]));
                S[50] = _fma_194;
                float _fma_195;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_195) : "f"(k_regs[3]), "f"(vn_3), "f"(S[51]));
                S[51] = _fma_195;
                float _fma_196;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_196) : "f"(k_regs[4]), "f"(vn_3), "f"(S[52]));
                S[52] = _fma_196;
                float _fma_197;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_197) : "f"(k_regs[5]), "f"(vn_3), "f"(S[53]));
                S[53] = _fma_197;
                float _fma_198;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_198) : "f"(k_regs[6]), "f"(vn_3), "f"(S[54]));
                S[54] = _fma_198;
                float _fma_199;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_199) : "f"(k_regs[7]), "f"(vn_3), "f"(S[55]));
                S[55] = _fma_199;
                float _fma_200;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_200) : "f"(k_regs[8]), "f"(vn_3), "f"(S[56]));
                S[56] = _fma_200;
                float _fma_201;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_201) : "f"(k_regs[9]), "f"(vn_3), "f"(S[57]));
                S[57] = _fma_201;
                float _fma_202;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_202) : "f"(k_regs[10]), "f"(vn_3), "f"(S[58]));
                S[58] = _fma_202;
                float _fma_203;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_203) : "f"(k_regs[11]), "f"(vn_3), "f"(S[59]));
                S[59] = _fma_203;
                float _fma_204;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_204) : "f"(k_regs[12]), "f"(vn_3), "f"(S[60]));
                S[60] = _fma_204;
                float _fma_205;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_205) : "f"(k_regs[13]), "f"(vn_3), "f"(S[61]));
                S[61] = _fma_205;
                float _fma_206;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_206) : "f"(k_regs[14]), "f"(vn_3), "f"(S[62]));
                S[62] = _fma_206;
                float _fma_207;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_207) : "f"(k_regs[15]), "f"(vn_3), "f"(S[63]));
                S[63] = _fma_207;
#else
                float2 vn2_3 = make_float2(vn_3, vn_3);
                S2[24] = fma_f32x2_rn_ftz(k2[0], vn2_3, S2[24]);
                S2[25] = fma_f32x2_rn_ftz(k2[1], vn2_3, S2[25]);
                S2[26] = fma_f32x2_rn_ftz(k2[2], vn2_3, S2[26]);
                S2[27] = fma_f32x2_rn_ftz(k2[3], vn2_3, S2[27]);
                S2[28] = fma_f32x2_rn_ftz(k2[4], vn2_3, S2[28]);
                S2[29] = fma_f32x2_rn_ftz(k2[5], vn2_3, S2[29]);
                S2[30] = fma_f32x2_rn_ftz(k2[6], vn2_3, S2[30]);
                S2[31] = fma_f32x2_rn_ftz(k2[7], vn2_3, S2[31]);
#endif
                int m_1 = t * 3 + 1;
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
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                    : "r"(sV_addr + (unsigned int)(m_1 * 256) + group_byte));
                float beta_1 = sBeta[m_1];
#if __CUDA_ARCH__ == 900
                float acc0_x_3 = 0.0f;
                float acc0_y_3 = 0.0f;
                float _fma_208;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_208) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_3));
                acc0_x_3 = _fma_208;
                float _fma_209;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_209) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_3));
                acc0_y_3 = _fma_209;
                float _fma_210;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_210) : "f"(S[2]), "f"(k_regs[2]), "f"(acc0_x_3));
                acc0_x_3 = _fma_210;
                float _fma_211;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_211) : "f"(S[3]), "f"(k_regs[3]), "f"(acc0_y_3));
                acc0_y_3 = _fma_211;
                float _fma_212;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_212) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_3));
                acc0_x_3 = _fma_212;
                float _fma_213;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_213) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_3));
                acc0_y_3 = _fma_213;
                float _fma_214;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_214) : "f"(S[6]), "f"(k_regs[6]), "f"(acc0_x_3));
                acc0_x_3 = _fma_214;
                float _fma_215;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_215) : "f"(S[7]), "f"(k_regs[7]), "f"(acc0_y_3));
                acc0_y_3 = _fma_215;
                float _fma_216;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_216) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_3));
                acc0_x_3 = _fma_216;
                float _fma_217;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_217) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_3));
                acc0_y_3 = _fma_217;
                float _fma_218;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_218) : "f"(S[10]), "f"(k_regs[10]), "f"(acc0_x_3));
                acc0_x_3 = _fma_218;
                float _fma_219;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_219) : "f"(S[11]), "f"(k_regs[11]), "f"(acc0_y_3));
                acc0_y_3 = _fma_219;
                float _fma_220;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_220) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_3));
                acc0_x_3 = _fma_220;
                float _fma_221;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_221) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_3));
                acc0_y_3 = _fma_221;
                float _fma_222;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_222) : "f"(S[14]), "f"(k_regs[14]), "f"(acc0_x_3));
                acc0_x_3 = _fma_222;
                float _fma_223;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_223) : "f"(S[15]), "f"(k_regs[15]), "f"(acc0_y_3));
                acc0_y_3 = _fma_223;
                float dot_3 = acc0_x_3 + acc0_y_3;
                float _shfl_xor_sync_224 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 1);
                dot_3 = dot_3 + _shfl_xor_sync_224;
                float _shfl_xor_sync_225 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 2);
                dot_3 = dot_3 + _shfl_xor_sync_225;
                float _shfl_xor_sync_226 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 4);
                dot_3 = dot_3 + _shfl_xor_sync_226;
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
                float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 1);
                dot_3 = dot_3 + _shfl_xor_sync_64;
                float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 2);
                dot_3 = dot_3 + _shfl_xor_sync_65;
                float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 4);
                dot_3 = dot_3 + _shfl_xor_sync_66;
#endif
                float vn_4 = (v_regs[0] - dot_3) * beta_1;
#if __CUDA_ARCH__ == 900
                float _fma_227;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_227) : "f"(k_regs[0]), "f"(vn_4), "f"(S[0]));
                S[0] = _fma_227;
                float _fma_228;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_228) : "f"(k_regs[1]), "f"(vn_4), "f"(S[1]));
                S[1] = _fma_228;
                float _fma_229;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_229) : "f"(k_regs[2]), "f"(vn_4), "f"(S[2]));
                S[2] = _fma_229;
                float _fma_230;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_230) : "f"(k_regs[3]), "f"(vn_4), "f"(S[3]));
                S[3] = _fma_230;
                float _fma_231;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_231) : "f"(k_regs[4]), "f"(vn_4), "f"(S[4]));
                S[4] = _fma_231;
                float _fma_232;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_232) : "f"(k_regs[5]), "f"(vn_4), "f"(S[5]));
                S[5] = _fma_232;
                float _fma_233;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_233) : "f"(k_regs[6]), "f"(vn_4), "f"(S[6]));
                S[6] = _fma_233;
                float _fma_234;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_234) : "f"(k_regs[7]), "f"(vn_4), "f"(S[7]));
                S[7] = _fma_234;
                float _fma_235;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_235) : "f"(k_regs[8]), "f"(vn_4), "f"(S[8]));
                S[8] = _fma_235;
                float _fma_236;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_236) : "f"(k_regs[9]), "f"(vn_4), "f"(S[9]));
                S[9] = _fma_236;
                float _fma_237;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_237) : "f"(k_regs[10]), "f"(vn_4), "f"(S[10]));
                S[10] = _fma_237;
                float _fma_238;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_238) : "f"(k_regs[11]), "f"(vn_4), "f"(S[11]));
                S[11] = _fma_238;
                float _fma_239;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_239) : "f"(k_regs[12]), "f"(vn_4), "f"(S[12]));
                S[12] = _fma_239;
                float _fma_240;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_240) : "f"(k_regs[13]), "f"(vn_4), "f"(S[13]));
                S[13] = _fma_240;
                float _fma_241;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_241) : "f"(k_regs[14]), "f"(vn_4), "f"(S[14]));
                S[14] = _fma_241;
                float _fma_242;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_242) : "f"(k_regs[15]), "f"(vn_4), "f"(S[15]));
                S[15] = _fma_242;
                float acc0_x_4 = 0.0f;
                float acc0_y_4 = 0.0f;
                float _fma_243;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_243) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_4));
                acc0_x_4 = _fma_243;
                float _fma_244;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_244) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_4));
                acc0_y_4 = _fma_244;
                float _fma_245;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_245) : "f"(S[18]), "f"(k_regs[2]), "f"(acc0_x_4));
                acc0_x_4 = _fma_245;
                float _fma_246;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_246) : "f"(S[19]), "f"(k_regs[3]), "f"(acc0_y_4));
                acc0_y_4 = _fma_246;
                float _fma_247;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_247) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_4));
                acc0_x_4 = _fma_247;
                float _fma_248;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_248) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_4));
                acc0_y_4 = _fma_248;
                float _fma_249;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_249) : "f"(S[22]), "f"(k_regs[6]), "f"(acc0_x_4));
                acc0_x_4 = _fma_249;
                float _fma_250;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_250) : "f"(S[23]), "f"(k_regs[7]), "f"(acc0_y_4));
                acc0_y_4 = _fma_250;
                float _fma_251;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_251) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_4));
                acc0_x_4 = _fma_251;
                float _fma_252;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_252) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_4));
                acc0_y_4 = _fma_252;
                float _fma_253;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_253) : "f"(S[26]), "f"(k_regs[10]), "f"(acc0_x_4));
                acc0_x_4 = _fma_253;
                float _fma_254;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_254) : "f"(S[27]), "f"(k_regs[11]), "f"(acc0_y_4));
                acc0_y_4 = _fma_254;
                float _fma_255;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_255) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_4));
                acc0_x_4 = _fma_255;
                float _fma_256;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_256) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_4));
                acc0_y_4 = _fma_256;
                float _fma_257;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_257) : "f"(S[30]), "f"(k_regs[14]), "f"(acc0_x_4));
                acc0_x_4 = _fma_257;
                float _fma_258;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_258) : "f"(S[31]), "f"(k_regs[15]), "f"(acc0_y_4));
                acc0_y_4 = _fma_258;
                float dot_4 = acc0_x_4 + acc0_y_4;
                float _shfl_xor_sync_259 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 1);
                dot_4 = dot_4 + _shfl_xor_sync_259;
                float _shfl_xor_sync_260 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 2);
                dot_4 = dot_4 + _shfl_xor_sync_260;
                float _shfl_xor_sync_261 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 4);
                dot_4 = dot_4 + _shfl_xor_sync_261;
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
                float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 1);
                dot_4 = dot_4 + _shfl_xor_sync_67;
                float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 2);
                dot_4 = dot_4 + _shfl_xor_sync_68;
                float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 4);
                dot_4 = dot_4 + _shfl_xor_sync_69;
#endif
                float vn_5 = (v_regs[1] - dot_4) * beta_1;
#if __CUDA_ARCH__ == 900
                float _fma_262;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_262) : "f"(k_regs[0]), "f"(vn_5), "f"(S[16]));
                S[16] = _fma_262;
                float _fma_263;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_263) : "f"(k_regs[1]), "f"(vn_5), "f"(S[17]));
                S[17] = _fma_263;
                float _fma_264;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_264) : "f"(k_regs[2]), "f"(vn_5), "f"(S[18]));
                S[18] = _fma_264;
                float _fma_265;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_265) : "f"(k_regs[3]), "f"(vn_5), "f"(S[19]));
                S[19] = _fma_265;
                float _fma_266;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_266) : "f"(k_regs[4]), "f"(vn_5), "f"(S[20]));
                S[20] = _fma_266;
                float _fma_267;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_267) : "f"(k_regs[5]), "f"(vn_5), "f"(S[21]));
                S[21] = _fma_267;
                float _fma_268;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_268) : "f"(k_regs[6]), "f"(vn_5), "f"(S[22]));
                S[22] = _fma_268;
                float _fma_269;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_269) : "f"(k_regs[7]), "f"(vn_5), "f"(S[23]));
                S[23] = _fma_269;
                float _fma_270;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_270) : "f"(k_regs[8]), "f"(vn_5), "f"(S[24]));
                S[24] = _fma_270;
                float _fma_271;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_271) : "f"(k_regs[9]), "f"(vn_5), "f"(S[25]));
                S[25] = _fma_271;
                float _fma_272;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_272) : "f"(k_regs[10]), "f"(vn_5), "f"(S[26]));
                S[26] = _fma_272;
                float _fma_273;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_273) : "f"(k_regs[11]), "f"(vn_5), "f"(S[27]));
                S[27] = _fma_273;
                float _fma_274;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_274) : "f"(k_regs[12]), "f"(vn_5), "f"(S[28]));
                S[28] = _fma_274;
                float _fma_275;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_275) : "f"(k_regs[13]), "f"(vn_5), "f"(S[29]));
                S[29] = _fma_275;
                float _fma_276;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_276) : "f"(k_regs[14]), "f"(vn_5), "f"(S[30]));
                S[30] = _fma_276;
                float _fma_277;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_277) : "f"(k_regs[15]), "f"(vn_5), "f"(S[31]));
                S[31] = _fma_277;
                float acc0_x_5 = 0.0f;
                float acc0_y_5 = 0.0f;
                float _fma_278;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_278) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_5));
                acc0_x_5 = _fma_278;
                float _fma_279;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_279) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_5));
                acc0_y_5 = _fma_279;
                float _fma_280;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_280) : "f"(S[34]), "f"(k_regs[2]), "f"(acc0_x_5));
                acc0_x_5 = _fma_280;
                float _fma_281;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_281) : "f"(S[35]), "f"(k_regs[3]), "f"(acc0_y_5));
                acc0_y_5 = _fma_281;
                float _fma_282;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_282) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_5));
                acc0_x_5 = _fma_282;
                float _fma_283;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_283) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_5));
                acc0_y_5 = _fma_283;
                float _fma_284;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_284) : "f"(S[38]), "f"(k_regs[6]), "f"(acc0_x_5));
                acc0_x_5 = _fma_284;
                float _fma_285;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_285) : "f"(S[39]), "f"(k_regs[7]), "f"(acc0_y_5));
                acc0_y_5 = _fma_285;
                float _fma_286;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_286) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_5));
                acc0_x_5 = _fma_286;
                float _fma_287;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_287) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_5));
                acc0_y_5 = _fma_287;
                float _fma_288;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_288) : "f"(S[42]), "f"(k_regs[10]), "f"(acc0_x_5));
                acc0_x_5 = _fma_288;
                float _fma_289;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_289) : "f"(S[43]), "f"(k_regs[11]), "f"(acc0_y_5));
                acc0_y_5 = _fma_289;
                float _fma_290;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_290) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_5));
                acc0_x_5 = _fma_290;
                float _fma_291;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_291) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_5));
                acc0_y_5 = _fma_291;
                float _fma_292;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_292) : "f"(S[46]), "f"(k_regs[14]), "f"(acc0_x_5));
                acc0_x_5 = _fma_292;
                float _fma_293;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_293) : "f"(S[47]), "f"(k_regs[15]), "f"(acc0_y_5));
                acc0_y_5 = _fma_293;
                float dot_5 = acc0_x_5 + acc0_y_5;
                float _shfl_xor_sync_294 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 1);
                dot_5 = dot_5 + _shfl_xor_sync_294;
                float _shfl_xor_sync_295 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 2);
                dot_5 = dot_5 + _shfl_xor_sync_295;
                float _shfl_xor_sync_296 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 4);
                dot_5 = dot_5 + _shfl_xor_sync_296;
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
                float2 acc0_5 = make_float2(0.0f, 0.0f);
                acc0_5 = fma_f32x2_rn_ftz(S2[16], k2[0], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[17], k2[1], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[18], k2[2], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[19], k2[3], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[20], k2[4], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[21], k2[5], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[22], k2[6], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[23], k2[7], acc0_5);
                float dot_5 = acc0_5.x + acc0_5.y;
                float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 1);
                dot_5 = dot_5 + _shfl_xor_sync_70;
                float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 2);
                dot_5 = dot_5 + _shfl_xor_sync_71;
                float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 4);
                dot_5 = dot_5 + _shfl_xor_sync_72;
#endif
                float vn_6 = (v_regs[2] - dot_5) * beta_1;
#if __CUDA_ARCH__ == 900
                float _fma_297;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_297) : "f"(k_regs[0]), "f"(vn_6), "f"(S[32]));
                S[32] = _fma_297;
                float _fma_298;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_298) : "f"(k_regs[1]), "f"(vn_6), "f"(S[33]));
                S[33] = _fma_298;
                float _fma_299;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_299) : "f"(k_regs[2]), "f"(vn_6), "f"(S[34]));
                S[34] = _fma_299;
                float _fma_300;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_300) : "f"(k_regs[3]), "f"(vn_6), "f"(S[35]));
                S[35] = _fma_300;
                float _fma_301;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_301) : "f"(k_regs[4]), "f"(vn_6), "f"(S[36]));
                S[36] = _fma_301;
                float _fma_302;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_302) : "f"(k_regs[5]), "f"(vn_6), "f"(S[37]));
                S[37] = _fma_302;
                float _fma_303;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_303) : "f"(k_regs[6]), "f"(vn_6), "f"(S[38]));
                S[38] = _fma_303;
                float _fma_304;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_304) : "f"(k_regs[7]), "f"(vn_6), "f"(S[39]));
                S[39] = _fma_304;
                float _fma_305;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_305) : "f"(k_regs[8]), "f"(vn_6), "f"(S[40]));
                S[40] = _fma_305;
                float _fma_306;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_306) : "f"(k_regs[9]), "f"(vn_6), "f"(S[41]));
                S[41] = _fma_306;
                float _fma_307;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_307) : "f"(k_regs[10]), "f"(vn_6), "f"(S[42]));
                S[42] = _fma_307;
                float _fma_308;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_308) : "f"(k_regs[11]), "f"(vn_6), "f"(S[43]));
                S[43] = _fma_308;
                float _fma_309;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_309) : "f"(k_regs[12]), "f"(vn_6), "f"(S[44]));
                S[44] = _fma_309;
                float _fma_310;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_310) : "f"(k_regs[13]), "f"(vn_6), "f"(S[45]));
                S[45] = _fma_310;
                float _fma_311;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_311) : "f"(k_regs[14]), "f"(vn_6), "f"(S[46]));
                S[46] = _fma_311;
                float _fma_312;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_312) : "f"(k_regs[15]), "f"(vn_6), "f"(S[47]));
                S[47] = _fma_312;
                float acc0_x_6 = 0.0f;
                float acc0_y_6 = 0.0f;
                float _fma_313;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_313) : "f"(S[48]), "f"(k_regs[0]), "f"(acc0_x_6));
                acc0_x_6 = _fma_313;
                float _fma_314;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_314) : "f"(S[49]), "f"(k_regs[1]), "f"(acc0_y_6));
                acc0_y_6 = _fma_314;
                float _fma_315;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_315) : "f"(S[50]), "f"(k_regs[2]), "f"(acc0_x_6));
                acc0_x_6 = _fma_315;
                float _fma_316;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_316) : "f"(S[51]), "f"(k_regs[3]), "f"(acc0_y_6));
                acc0_y_6 = _fma_316;
                float _fma_317;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_317) : "f"(S[52]), "f"(k_regs[4]), "f"(acc0_x_6));
                acc0_x_6 = _fma_317;
                float _fma_318;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_318) : "f"(S[53]), "f"(k_regs[5]), "f"(acc0_y_6));
                acc0_y_6 = _fma_318;
                float _fma_319;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_319) : "f"(S[54]), "f"(k_regs[6]), "f"(acc0_x_6));
                acc0_x_6 = _fma_319;
                float _fma_320;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_320) : "f"(S[55]), "f"(k_regs[7]), "f"(acc0_y_6));
                acc0_y_6 = _fma_320;
                float _fma_321;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_321) : "f"(S[56]), "f"(k_regs[8]), "f"(acc0_x_6));
                acc0_x_6 = _fma_321;
                float _fma_322;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_322) : "f"(S[57]), "f"(k_regs[9]), "f"(acc0_y_6));
                acc0_y_6 = _fma_322;
                float _fma_323;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_323) : "f"(S[58]), "f"(k_regs[10]), "f"(acc0_x_6));
                acc0_x_6 = _fma_323;
                float _fma_324;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_324) : "f"(S[59]), "f"(k_regs[11]), "f"(acc0_y_6));
                acc0_y_6 = _fma_324;
                float _fma_325;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_325) : "f"(S[60]), "f"(k_regs[12]), "f"(acc0_x_6));
                acc0_x_6 = _fma_325;
                float _fma_326;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_326) : "f"(S[61]), "f"(k_regs[13]), "f"(acc0_y_6));
                acc0_y_6 = _fma_326;
                float _fma_327;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_327) : "f"(S[62]), "f"(k_regs[14]), "f"(acc0_x_6));
                acc0_x_6 = _fma_327;
                float _fma_328;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_328) : "f"(S[63]), "f"(k_regs[15]), "f"(acc0_y_6));
                acc0_y_6 = _fma_328;
                float dot_6 = acc0_x_6 + acc0_y_6;
                float _shfl_xor_sync_329 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 1);
                dot_6 = dot_6 + _shfl_xor_sync_329;
                float _shfl_xor_sync_330 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 2);
                dot_6 = dot_6 + _shfl_xor_sync_330;
                float _shfl_xor_sync_331 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 4);
                dot_6 = dot_6 + _shfl_xor_sync_331;
#else
                float2 vn2_6 = make_float2(vn_6, vn_6);
                S2[16] = fma_f32x2_rn_ftz(k2[0], vn2_6, S2[16]);
                S2[17] = fma_f32x2_rn_ftz(k2[1], vn2_6, S2[17]);
                S2[18] = fma_f32x2_rn_ftz(k2[2], vn2_6, S2[18]);
                S2[19] = fma_f32x2_rn_ftz(k2[3], vn2_6, S2[19]);
                S2[20] = fma_f32x2_rn_ftz(k2[4], vn2_6, S2[20]);
                S2[21] = fma_f32x2_rn_ftz(k2[5], vn2_6, S2[21]);
                S2[22] = fma_f32x2_rn_ftz(k2[6], vn2_6, S2[22]);
                S2[23] = fma_f32x2_rn_ftz(k2[7], vn2_6, S2[23]);
                float2 acc0_6 = make_float2(0.0f, 0.0f);
                acc0_6 = fma_f32x2_rn_ftz(S2[24], k2[0], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[25], k2[1], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[26], k2[2], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[27], k2[3], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[28], k2[4], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[29], k2[5], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[30], k2[6], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[31], k2[7], acc0_6);
                float dot_6 = acc0_6.x + acc0_6.y;
                float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 1);
                dot_6 = dot_6 + _shfl_xor_sync_73;
                float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 2);
                dot_6 = dot_6 + _shfl_xor_sync_74;
                float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 4);
                dot_6 = dot_6 + _shfl_xor_sync_75;
#endif
                float vn_7 = (v_regs[3] - dot_6) * beta_1;
#if __CUDA_ARCH__ == 900
                float _fma_332;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_332) : "f"(k_regs[0]), "f"(vn_7), "f"(S[48]));
                S[48] = _fma_332;
                float _fma_333;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_333) : "f"(k_regs[1]), "f"(vn_7), "f"(S[49]));
                S[49] = _fma_333;
                float _fma_334;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_334) : "f"(k_regs[2]), "f"(vn_7), "f"(S[50]));
                S[50] = _fma_334;
                float _fma_335;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_335) : "f"(k_regs[3]), "f"(vn_7), "f"(S[51]));
                S[51] = _fma_335;
                float _fma_336;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_336) : "f"(k_regs[4]), "f"(vn_7), "f"(S[52]));
                S[52] = _fma_336;
                float _fma_337;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_337) : "f"(k_regs[5]), "f"(vn_7), "f"(S[53]));
                S[53] = _fma_337;
                float _fma_338;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_338) : "f"(k_regs[6]), "f"(vn_7), "f"(S[54]));
                S[54] = _fma_338;
                float _fma_339;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_339) : "f"(k_regs[7]), "f"(vn_7), "f"(S[55]));
                S[55] = _fma_339;
                float _fma_340;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_340) : "f"(k_regs[8]), "f"(vn_7), "f"(S[56]));
                S[56] = _fma_340;
                float _fma_341;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_341) : "f"(k_regs[9]), "f"(vn_7), "f"(S[57]));
                S[57] = _fma_341;
                float _fma_342;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_342) : "f"(k_regs[10]), "f"(vn_7), "f"(S[58]));
                S[58] = _fma_342;
                float _fma_343;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_343) : "f"(k_regs[11]), "f"(vn_7), "f"(S[59]));
                S[59] = _fma_343;
                float _fma_344;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_344) : "f"(k_regs[12]), "f"(vn_7), "f"(S[60]));
                S[60] = _fma_344;
                float _fma_345;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_345) : "f"(k_regs[13]), "f"(vn_7), "f"(S[61]));
                S[61] = _fma_345;
                float _fma_346;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_346) : "f"(k_regs[14]), "f"(vn_7), "f"(S[62]));
                S[62] = _fma_346;
                float _fma_347;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_347) : "f"(k_regs[15]), "f"(vn_7), "f"(S[63]));
                S[63] = _fma_347;
#else
                float2 vn2_7 = make_float2(vn_7, vn_7);
                S2[24] = fma_f32x2_rn_ftz(k2[0], vn2_7, S2[24]);
                S2[25] = fma_f32x2_rn_ftz(k2[1], vn2_7, S2[25]);
                S2[26] = fma_f32x2_rn_ftz(k2[2], vn2_7, S2[26]);
                S2[27] = fma_f32x2_rn_ftz(k2[3], vn2_7, S2[27]);
                S2[28] = fma_f32x2_rn_ftz(k2[4], vn2_7, S2[28]);
                S2[29] = fma_f32x2_rn_ftz(k2[5], vn2_7, S2[29]);
                S2[30] = fma_f32x2_rn_ftz(k2[6], vn2_7, S2[30]);
                S2[31] = fma_f32x2_rn_ftz(k2[7], vn2_7, S2[31]);
#endif
                int m_2 = t * 3 + 2;
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
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                    : "r"(sV_addr + (unsigned int)(m_2 * 256) + group_byte));
                float beta_2 = sBeta[m_2];
#if __CUDA_ARCH__ == 900
                float acc0_x_7 = 0.0f;
                float acc0_y_7 = 0.0f;
                float _fma_348;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_348) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_7));
                acc0_x_7 = _fma_348;
                float _fma_349;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_349) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_7));
                acc0_y_7 = _fma_349;
                float _fma_350;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_350) : "f"(S[2]), "f"(k_regs[2]), "f"(acc0_x_7));
                acc0_x_7 = _fma_350;
                float _fma_351;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_351) : "f"(S[3]), "f"(k_regs[3]), "f"(acc0_y_7));
                acc0_y_7 = _fma_351;
                float _fma_352;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_352) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_7));
                acc0_x_7 = _fma_352;
                float _fma_353;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_353) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_7));
                acc0_y_7 = _fma_353;
                float _fma_354;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_354) : "f"(S[6]), "f"(k_regs[6]), "f"(acc0_x_7));
                acc0_x_7 = _fma_354;
                float _fma_355;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_355) : "f"(S[7]), "f"(k_regs[7]), "f"(acc0_y_7));
                acc0_y_7 = _fma_355;
                float _fma_356;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_356) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_7));
                acc0_x_7 = _fma_356;
                float _fma_357;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_357) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_7));
                acc0_y_7 = _fma_357;
                float _fma_358;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_358) : "f"(S[10]), "f"(k_regs[10]), "f"(acc0_x_7));
                acc0_x_7 = _fma_358;
                float _fma_359;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_359) : "f"(S[11]), "f"(k_regs[11]), "f"(acc0_y_7));
                acc0_y_7 = _fma_359;
                float _fma_360;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_360) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_7));
                acc0_x_7 = _fma_360;
                float _fma_361;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_361) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_7));
                acc0_y_7 = _fma_361;
                float _fma_362;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_362) : "f"(S[14]), "f"(k_regs[14]), "f"(acc0_x_7));
                acc0_x_7 = _fma_362;
                float _fma_363;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_363) : "f"(S[15]), "f"(k_regs[15]), "f"(acc0_y_7));
                acc0_y_7 = _fma_363;
                float dot_7 = acc0_x_7 + acc0_y_7;
                float _shfl_xor_sync_364 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 1);
                dot_7 = dot_7 + _shfl_xor_sync_364;
                float _shfl_xor_sync_365 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 2);
                dot_7 = dot_7 + _shfl_xor_sync_365;
                float _shfl_xor_sync_366 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 4);
                dot_7 = dot_7 + _shfl_xor_sync_366;
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
                float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 1);
                dot_7 = dot_7 + _shfl_xor_sync_76;
                float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 2);
                dot_7 = dot_7 + _shfl_xor_sync_77;
                float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 4);
                dot_7 = dot_7 + _shfl_xor_sync_78;
#endif
                float vn_8 = (v_regs[0] - dot_7) * beta_2;
#if __CUDA_ARCH__ == 900
                float _fma_367;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_367) : "f"(k_regs[0]), "f"(vn_8), "f"(S[0]));
                S[0] = _fma_367;
                float _fma_368;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_368) : "f"(k_regs[1]), "f"(vn_8), "f"(S[1]));
                S[1] = _fma_368;
                float _fma_369;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_369) : "f"(k_regs[2]), "f"(vn_8), "f"(S[2]));
                S[2] = _fma_369;
                float _fma_370;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_370) : "f"(k_regs[3]), "f"(vn_8), "f"(S[3]));
                S[3] = _fma_370;
                float _fma_371;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_371) : "f"(k_regs[4]), "f"(vn_8), "f"(S[4]));
                S[4] = _fma_371;
                float _fma_372;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_372) : "f"(k_regs[5]), "f"(vn_8), "f"(S[5]));
                S[5] = _fma_372;
                float _fma_373;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_373) : "f"(k_regs[6]), "f"(vn_8), "f"(S[6]));
                S[6] = _fma_373;
                float _fma_374;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_374) : "f"(k_regs[7]), "f"(vn_8), "f"(S[7]));
                S[7] = _fma_374;
                float _fma_375;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_375) : "f"(k_regs[8]), "f"(vn_8), "f"(S[8]));
                S[8] = _fma_375;
                float _fma_376;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_376) : "f"(k_regs[9]), "f"(vn_8), "f"(S[9]));
                S[9] = _fma_376;
                float _fma_377;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_377) : "f"(k_regs[10]), "f"(vn_8), "f"(S[10]));
                S[10] = _fma_377;
                float _fma_378;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_378) : "f"(k_regs[11]), "f"(vn_8), "f"(S[11]));
                S[11] = _fma_378;
                float _fma_379;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_379) : "f"(k_regs[12]), "f"(vn_8), "f"(S[12]));
                S[12] = _fma_379;
                float _fma_380;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_380) : "f"(k_regs[13]), "f"(vn_8), "f"(S[13]));
                S[13] = _fma_380;
                float _fma_381;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_381) : "f"(k_regs[14]), "f"(vn_8), "f"(S[14]));
                S[14] = _fma_381;
                float _fma_382;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_382) : "f"(k_regs[15]), "f"(vn_8), "f"(S[15]));
                S[15] = _fma_382;
                float acc0_x_8 = 0.0f;
                float acc0_y_8 = 0.0f;
                float _fma_383;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_383) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_8));
                acc0_x_8 = _fma_383;
                float _fma_384;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_384) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_8));
                acc0_y_8 = _fma_384;
                float _fma_385;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_385) : "f"(S[18]), "f"(k_regs[2]), "f"(acc0_x_8));
                acc0_x_8 = _fma_385;
                float _fma_386;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_386) : "f"(S[19]), "f"(k_regs[3]), "f"(acc0_y_8));
                acc0_y_8 = _fma_386;
                float _fma_387;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_387) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_8));
                acc0_x_8 = _fma_387;
                float _fma_388;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_388) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_8));
                acc0_y_8 = _fma_388;
                float _fma_389;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_389) : "f"(S[22]), "f"(k_regs[6]), "f"(acc0_x_8));
                acc0_x_8 = _fma_389;
                float _fma_390;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_390) : "f"(S[23]), "f"(k_regs[7]), "f"(acc0_y_8));
                acc0_y_8 = _fma_390;
                float _fma_391;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_391) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_8));
                acc0_x_8 = _fma_391;
                float _fma_392;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_392) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_8));
                acc0_y_8 = _fma_392;
                float _fma_393;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_393) : "f"(S[26]), "f"(k_regs[10]), "f"(acc0_x_8));
                acc0_x_8 = _fma_393;
                float _fma_394;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_394) : "f"(S[27]), "f"(k_regs[11]), "f"(acc0_y_8));
                acc0_y_8 = _fma_394;
                float _fma_395;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_395) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_8));
                acc0_x_8 = _fma_395;
                float _fma_396;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_396) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_8));
                acc0_y_8 = _fma_396;
                float _fma_397;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_397) : "f"(S[30]), "f"(k_regs[14]), "f"(acc0_x_8));
                acc0_x_8 = _fma_397;
                float _fma_398;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_398) : "f"(S[31]), "f"(k_regs[15]), "f"(acc0_y_8));
                acc0_y_8 = _fma_398;
                float dot_8 = acc0_x_8 + acc0_y_8;
                float _shfl_xor_sync_399 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 1);
                dot_8 = dot_8 + _shfl_xor_sync_399;
                float _shfl_xor_sync_400 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 2);
                dot_8 = dot_8 + _shfl_xor_sync_400;
                float _shfl_xor_sync_401 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 4);
                dot_8 = dot_8 + _shfl_xor_sync_401;
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
                float2 acc0_8 = make_float2(0.0f, 0.0f);
                acc0_8 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_8);
                float dot_8 = acc0_8.x + acc0_8.y;
                float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 1);
                dot_8 = dot_8 + _shfl_xor_sync_79;
                float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 2);
                dot_8 = dot_8 + _shfl_xor_sync_80;
                float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 4);
                dot_8 = dot_8 + _shfl_xor_sync_81;
#endif
                float vn_9 = (v_regs[1] - dot_8) * beta_2;
#if __CUDA_ARCH__ == 900
                float _fma_402;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_402) : "f"(k_regs[0]), "f"(vn_9), "f"(S[16]));
                S[16] = _fma_402;
                float _fma_403;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_403) : "f"(k_regs[1]), "f"(vn_9), "f"(S[17]));
                S[17] = _fma_403;
                float _fma_404;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_404) : "f"(k_regs[2]), "f"(vn_9), "f"(S[18]));
                S[18] = _fma_404;
                float _fma_405;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_405) : "f"(k_regs[3]), "f"(vn_9), "f"(S[19]));
                S[19] = _fma_405;
                float _fma_406;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_406) : "f"(k_regs[4]), "f"(vn_9), "f"(S[20]));
                S[20] = _fma_406;
                float _fma_407;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_407) : "f"(k_regs[5]), "f"(vn_9), "f"(S[21]));
                S[21] = _fma_407;
                float _fma_408;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_408) : "f"(k_regs[6]), "f"(vn_9), "f"(S[22]));
                S[22] = _fma_408;
                float _fma_409;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_409) : "f"(k_regs[7]), "f"(vn_9), "f"(S[23]));
                S[23] = _fma_409;
                float _fma_410;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_410) : "f"(k_regs[8]), "f"(vn_9), "f"(S[24]));
                S[24] = _fma_410;
                float _fma_411;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_411) : "f"(k_regs[9]), "f"(vn_9), "f"(S[25]));
                S[25] = _fma_411;
                float _fma_412;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_412) : "f"(k_regs[10]), "f"(vn_9), "f"(S[26]));
                S[26] = _fma_412;
                float _fma_413;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_413) : "f"(k_regs[11]), "f"(vn_9), "f"(S[27]));
                S[27] = _fma_413;
                float _fma_414;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_414) : "f"(k_regs[12]), "f"(vn_9), "f"(S[28]));
                S[28] = _fma_414;
                float _fma_415;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_415) : "f"(k_regs[13]), "f"(vn_9), "f"(S[29]));
                S[29] = _fma_415;
                float _fma_416;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_416) : "f"(k_regs[14]), "f"(vn_9), "f"(S[30]));
                S[30] = _fma_416;
                float _fma_417;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_417) : "f"(k_regs[15]), "f"(vn_9), "f"(S[31]));
                S[31] = _fma_417;
                float acc0_x_9 = 0.0f;
                float acc0_y_9 = 0.0f;
                float _fma_418;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_418) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_9));
                acc0_x_9 = _fma_418;
                float _fma_419;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_419) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_9));
                acc0_y_9 = _fma_419;
                float _fma_420;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_420) : "f"(S[34]), "f"(k_regs[2]), "f"(acc0_x_9));
                acc0_x_9 = _fma_420;
                float _fma_421;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_421) : "f"(S[35]), "f"(k_regs[3]), "f"(acc0_y_9));
                acc0_y_9 = _fma_421;
                float _fma_422;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_422) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_9));
                acc0_x_9 = _fma_422;
                float _fma_423;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_423) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_9));
                acc0_y_9 = _fma_423;
                float _fma_424;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_424) : "f"(S[38]), "f"(k_regs[6]), "f"(acc0_x_9));
                acc0_x_9 = _fma_424;
                float _fma_425;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_425) : "f"(S[39]), "f"(k_regs[7]), "f"(acc0_y_9));
                acc0_y_9 = _fma_425;
                float _fma_426;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_426) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_9));
                acc0_x_9 = _fma_426;
                float _fma_427;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_427) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_9));
                acc0_y_9 = _fma_427;
                float _fma_428;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_428) : "f"(S[42]), "f"(k_regs[10]), "f"(acc0_x_9));
                acc0_x_9 = _fma_428;
                float _fma_429;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_429) : "f"(S[43]), "f"(k_regs[11]), "f"(acc0_y_9));
                acc0_y_9 = _fma_429;
                float _fma_430;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_430) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_9));
                acc0_x_9 = _fma_430;
                float _fma_431;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_431) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_9));
                acc0_y_9 = _fma_431;
                float _fma_432;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_432) : "f"(S[46]), "f"(k_regs[14]), "f"(acc0_x_9));
                acc0_x_9 = _fma_432;
                float _fma_433;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_433) : "f"(S[47]), "f"(k_regs[15]), "f"(acc0_y_9));
                acc0_y_9 = _fma_433;
                float dot_9 = acc0_x_9 + acc0_y_9;
                float _shfl_xor_sync_434 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 1);
                dot_9 = dot_9 + _shfl_xor_sync_434;
                float _shfl_xor_sync_435 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 2);
                dot_9 = dot_9 + _shfl_xor_sync_435;
                float _shfl_xor_sync_436 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 4);
                dot_9 = dot_9 + _shfl_xor_sync_436;
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
                float2 acc0_9 = make_float2(0.0f, 0.0f);
                acc0_9 = fma_f32x2_rn_ftz(S2[16], k2[0], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[17], k2[1], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[18], k2[2], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[19], k2[3], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[20], k2[4], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[21], k2[5], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[22], k2[6], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[23], k2[7], acc0_9);
                float dot_9 = acc0_9.x + acc0_9.y;
                float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 1);
                dot_9 = dot_9 + _shfl_xor_sync_82;
                float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 2);
                dot_9 = dot_9 + _shfl_xor_sync_83;
                float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 4);
                dot_9 = dot_9 + _shfl_xor_sync_84;
#endif
                float vn_10 = (v_regs[2] - dot_9) * beta_2;
#if __CUDA_ARCH__ == 900
                float _fma_437;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_437) : "f"(k_regs[0]), "f"(vn_10), "f"(S[32]));
                S[32] = _fma_437;
                float _fma_438;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_438) : "f"(k_regs[1]), "f"(vn_10), "f"(S[33]));
                S[33] = _fma_438;
                float _fma_439;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_439) : "f"(k_regs[2]), "f"(vn_10), "f"(S[34]));
                S[34] = _fma_439;
                float _fma_440;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_440) : "f"(k_regs[3]), "f"(vn_10), "f"(S[35]));
                S[35] = _fma_440;
                float _fma_441;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_441) : "f"(k_regs[4]), "f"(vn_10), "f"(S[36]));
                S[36] = _fma_441;
                float _fma_442;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_442) : "f"(k_regs[5]), "f"(vn_10), "f"(S[37]));
                S[37] = _fma_442;
                float _fma_443;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_443) : "f"(k_regs[6]), "f"(vn_10), "f"(S[38]));
                S[38] = _fma_443;
                float _fma_444;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_444) : "f"(k_regs[7]), "f"(vn_10), "f"(S[39]));
                S[39] = _fma_444;
                float _fma_445;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_445) : "f"(k_regs[8]), "f"(vn_10), "f"(S[40]));
                S[40] = _fma_445;
                float _fma_446;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_446) : "f"(k_regs[9]), "f"(vn_10), "f"(S[41]));
                S[41] = _fma_446;
                float _fma_447;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_447) : "f"(k_regs[10]), "f"(vn_10), "f"(S[42]));
                S[42] = _fma_447;
                float _fma_448;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_448) : "f"(k_regs[11]), "f"(vn_10), "f"(S[43]));
                S[43] = _fma_448;
                float _fma_449;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_449) : "f"(k_regs[12]), "f"(vn_10), "f"(S[44]));
                S[44] = _fma_449;
                float _fma_450;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_450) : "f"(k_regs[13]), "f"(vn_10), "f"(S[45]));
                S[45] = _fma_450;
                float _fma_451;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_451) : "f"(k_regs[14]), "f"(vn_10), "f"(S[46]));
                S[46] = _fma_451;
                float _fma_452;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_452) : "f"(k_regs[15]), "f"(vn_10), "f"(S[47]));
                S[47] = _fma_452;
                float acc0_x_10 = 0.0f;
                float acc0_y_10 = 0.0f;
                float _fma_453;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_453) : "f"(S[48]), "f"(k_regs[0]), "f"(acc0_x_10));
                acc0_x_10 = _fma_453;
                float _fma_454;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_454) : "f"(S[49]), "f"(k_regs[1]), "f"(acc0_y_10));
                acc0_y_10 = _fma_454;
                float _fma_455;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_455) : "f"(S[50]), "f"(k_regs[2]), "f"(acc0_x_10));
                acc0_x_10 = _fma_455;
                float _fma_456;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_456) : "f"(S[51]), "f"(k_regs[3]), "f"(acc0_y_10));
                acc0_y_10 = _fma_456;
                float _fma_457;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_457) : "f"(S[52]), "f"(k_regs[4]), "f"(acc0_x_10));
                acc0_x_10 = _fma_457;
                float _fma_458;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_458) : "f"(S[53]), "f"(k_regs[5]), "f"(acc0_y_10));
                acc0_y_10 = _fma_458;
                float _fma_459;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_459) : "f"(S[54]), "f"(k_regs[6]), "f"(acc0_x_10));
                acc0_x_10 = _fma_459;
                float _fma_460;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_460) : "f"(S[55]), "f"(k_regs[7]), "f"(acc0_y_10));
                acc0_y_10 = _fma_460;
                float _fma_461;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_461) : "f"(S[56]), "f"(k_regs[8]), "f"(acc0_x_10));
                acc0_x_10 = _fma_461;
                float _fma_462;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_462) : "f"(S[57]), "f"(k_regs[9]), "f"(acc0_y_10));
                acc0_y_10 = _fma_462;
                float _fma_463;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_463) : "f"(S[58]), "f"(k_regs[10]), "f"(acc0_x_10));
                acc0_x_10 = _fma_463;
                float _fma_464;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_464) : "f"(S[59]), "f"(k_regs[11]), "f"(acc0_y_10));
                acc0_y_10 = _fma_464;
                float _fma_465;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_465) : "f"(S[60]), "f"(k_regs[12]), "f"(acc0_x_10));
                acc0_x_10 = _fma_465;
                float _fma_466;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_466) : "f"(S[61]), "f"(k_regs[13]), "f"(acc0_y_10));
                acc0_y_10 = _fma_466;
                float _fma_467;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_467) : "f"(S[62]), "f"(k_regs[14]), "f"(acc0_x_10));
                acc0_x_10 = _fma_467;
                float _fma_468;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_468) : "f"(S[63]), "f"(k_regs[15]), "f"(acc0_y_10));
                acc0_y_10 = _fma_468;
                float dot_10 = acc0_x_10 + acc0_y_10;
                float _shfl_xor_sync_469 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 1);
                dot_10 = dot_10 + _shfl_xor_sync_469;
                float _shfl_xor_sync_470 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 2);
                dot_10 = dot_10 + _shfl_xor_sync_470;
                float _shfl_xor_sync_471 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 4);
                dot_10 = dot_10 + _shfl_xor_sync_471;
#else
                float2 vn2_10 = make_float2(vn_10, vn_10);
                S2[16] = fma_f32x2_rn_ftz(k2[0], vn2_10, S2[16]);
                S2[17] = fma_f32x2_rn_ftz(k2[1], vn2_10, S2[17]);
                S2[18] = fma_f32x2_rn_ftz(k2[2], vn2_10, S2[18]);
                S2[19] = fma_f32x2_rn_ftz(k2[3], vn2_10, S2[19]);
                S2[20] = fma_f32x2_rn_ftz(k2[4], vn2_10, S2[20]);
                S2[21] = fma_f32x2_rn_ftz(k2[5], vn2_10, S2[21]);
                S2[22] = fma_f32x2_rn_ftz(k2[6], vn2_10, S2[22]);
                S2[23] = fma_f32x2_rn_ftz(k2[7], vn2_10, S2[23]);
                float2 acc0_10 = make_float2(0.0f, 0.0f);
                acc0_10 = fma_f32x2_rn_ftz(S2[24], k2[0], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[25], k2[1], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[26], k2[2], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[27], k2[3], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[28], k2[4], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[29], k2[5], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[30], k2[6], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[31], k2[7], acc0_10);
                float dot_10 = acc0_10.x + acc0_10.y;
                float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 1);
                dot_10 = dot_10 + _shfl_xor_sync_85;
                float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 2);
                dot_10 = dot_10 + _shfl_xor_sync_86;
                float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 4);
                dot_10 = dot_10 + _shfl_xor_sync_87;
#endif
                float vn_11 = (v_regs[3] - dot_10) * beta_2;
#if __CUDA_ARCH__ == 900
                float _fma_472;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_472) : "f"(k_regs[0]), "f"(vn_11), "f"(S[48]));
                S[48] = _fma_472;
                float _fma_473;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_473) : "f"(k_regs[1]), "f"(vn_11), "f"(S[49]));
                S[49] = _fma_473;
                float _fma_474;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_474) : "f"(k_regs[2]), "f"(vn_11), "f"(S[50]));
                S[50] = _fma_474;
                float _fma_475;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_475) : "f"(k_regs[3]), "f"(vn_11), "f"(S[51]));
                S[51] = _fma_475;
                float _fma_476;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_476) : "f"(k_regs[4]), "f"(vn_11), "f"(S[52]));
                S[52] = _fma_476;
                float _fma_477;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_477) : "f"(k_regs[5]), "f"(vn_11), "f"(S[53]));
                S[53] = _fma_477;
                float _fma_478;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_478) : "f"(k_regs[6]), "f"(vn_11), "f"(S[54]));
                S[54] = _fma_478;
                float _fma_479;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_479) : "f"(k_regs[7]), "f"(vn_11), "f"(S[55]));
                S[55] = _fma_479;
                float _fma_480;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_480) : "f"(k_regs[8]), "f"(vn_11), "f"(S[56]));
                S[56] = _fma_480;
                float _fma_481;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_481) : "f"(k_regs[9]), "f"(vn_11), "f"(S[57]));
                S[57] = _fma_481;
                float _fma_482;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_482) : "f"(k_regs[10]), "f"(vn_11), "f"(S[58]));
                S[58] = _fma_482;
                float _fma_483;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_483) : "f"(k_regs[11]), "f"(vn_11), "f"(S[59]));
                S[59] = _fma_483;
                float _fma_484;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_484) : "f"(k_regs[12]), "f"(vn_11), "f"(S[60]));
                S[60] = _fma_484;
                float _fma_485;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_485) : "f"(k_regs[13]), "f"(vn_11), "f"(S[61]));
                S[61] = _fma_485;
                float _fma_486;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_486) : "f"(k_regs[14]), "f"(vn_11), "f"(S[62]));
                S[62] = _fma_486;
                float _fma_487;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_487) : "f"(k_regs[15]), "f"(vn_11), "f"(S[63]));
                S[63] = _fma_487;
#else
                float2 vn2_11 = make_float2(vn_11, vn_11);
                S2[24] = fma_f32x2_rn_ftz(k2[0], vn2_11, S2[24]);
                S2[25] = fma_f32x2_rn_ftz(k2[1], vn2_11, S2[25]);
                S2[26] = fma_f32x2_rn_ftz(k2[2], vn2_11, S2[26]);
                S2[27] = fma_f32x2_rn_ftz(k2[3], vn2_11, S2[27]);
                S2[28] = fma_f32x2_rn_ftz(k2[4], vn2_11, S2[28]);
                S2[29] = fma_f32x2_rn_ftz(k2[5], vn2_11, S2[29]);
                S2[30] = fma_f32x2_rn_ftz(k2[6], vn2_11, S2[30]);
                S2[31] = fma_f32x2_rn_ftz(k2[7], vn2_11, S2[31]);
#endif
                unsigned int qbase = sQ_addr + (unsigned int)(t * 512) + lane_byte;
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
                float acc0_x_11 = 0.0f;
                float acc0_y_11 = 0.0f;
                float _fma_488;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_488) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_11));
                acc0_x_11 = _fma_488;
                float _fma_489;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_489) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_11));
                acc0_y_11 = _fma_489;
                float _fma_490;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_490) : "f"(S[2]), "f"(k_regs[2]), "f"(acc0_x_11));
                acc0_x_11 = _fma_490;
                float _fma_491;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_491) : "f"(S[3]), "f"(k_regs[3]), "f"(acc0_y_11));
                acc0_y_11 = _fma_491;
                float _fma_492;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_492) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_11));
                acc0_x_11 = _fma_492;
                float _fma_493;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_493) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_11));
                acc0_y_11 = _fma_493;
                float _fma_494;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_494) : "f"(S[6]), "f"(k_regs[6]), "f"(acc0_x_11));
                acc0_x_11 = _fma_494;
                float _fma_495;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_495) : "f"(S[7]), "f"(k_regs[7]), "f"(acc0_y_11));
                acc0_y_11 = _fma_495;
                float _fma_496;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_496) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_11));
                acc0_x_11 = _fma_496;
                float _fma_497;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_497) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_11));
                acc0_y_11 = _fma_497;
                float _fma_498;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_498) : "f"(S[10]), "f"(k_regs[10]), "f"(acc0_x_11));
                acc0_x_11 = _fma_498;
                float _fma_499;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_499) : "f"(S[11]), "f"(k_regs[11]), "f"(acc0_y_11));
                acc0_y_11 = _fma_499;
                float _fma_500;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_500) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_11));
                acc0_x_11 = _fma_500;
                float _fma_501;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_501) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_11));
                acc0_y_11 = _fma_501;
                float _fma_502;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_502) : "f"(S[14]), "f"(k_regs[14]), "f"(acc0_x_11));
                acc0_x_11 = _fma_502;
                float _fma_503;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_503) : "f"(S[15]), "f"(k_regs[15]), "f"(acc0_y_11));
                acc0_y_11 = _fma_503;
                float dot_11 = acc0_x_11 + acc0_y_11;
                float _shfl_xor_sync_504 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 1);
                dot_11 = dot_11 + _shfl_xor_sync_504;
                float _shfl_xor_sync_505 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 2);
                dot_11 = dot_11 + _shfl_xor_sync_505;
                float _shfl_xor_sync_506 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 4);
                dot_11 = dot_11 + _shfl_xor_sync_506;
#else
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                float2 acc0_11 = make_float2(0.0f, 0.0f);
                acc0_11 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_11);
                float dot_11 = acc0_11.x + acc0_11.y;
                float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 1);
                dot_11 = dot_11 + _shfl_xor_sync_88;
                float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 2);
                dot_11 = dot_11 + _shfl_xor_sync_89;
                float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 4);
                dot_11 = dot_11 + _shfl_xor_sync_90;
#endif
                o4[0] = dot_11;
#if __CUDA_ARCH__ == 900
                float acc0_x_12 = 0.0f;
                float acc0_y_12 = 0.0f;
                float _fma_507;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_507) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_12));
                acc0_x_12 = _fma_507;
                float _fma_508;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_508) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_12));
                acc0_y_12 = _fma_508;
                float _fma_509;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_509) : "f"(S[18]), "f"(k_regs[2]), "f"(acc0_x_12));
                acc0_x_12 = _fma_509;
                float _fma_510;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_510) : "f"(S[19]), "f"(k_regs[3]), "f"(acc0_y_12));
                acc0_y_12 = _fma_510;
                float _fma_511;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_511) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_12));
                acc0_x_12 = _fma_511;
                float _fma_512;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_512) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_12));
                acc0_y_12 = _fma_512;
                float _fma_513;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_513) : "f"(S[22]), "f"(k_regs[6]), "f"(acc0_x_12));
                acc0_x_12 = _fma_513;
                float _fma_514;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_514) : "f"(S[23]), "f"(k_regs[7]), "f"(acc0_y_12));
                acc0_y_12 = _fma_514;
                float _fma_515;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_515) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_12));
                acc0_x_12 = _fma_515;
                float _fma_516;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_516) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_12));
                acc0_y_12 = _fma_516;
                float _fma_517;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_517) : "f"(S[26]), "f"(k_regs[10]), "f"(acc0_x_12));
                acc0_x_12 = _fma_517;
                float _fma_518;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_518) : "f"(S[27]), "f"(k_regs[11]), "f"(acc0_y_12));
                acc0_y_12 = _fma_518;
                float _fma_519;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_519) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_12));
                acc0_x_12 = _fma_519;
                float _fma_520;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_520) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_12));
                acc0_y_12 = _fma_520;
                float _fma_521;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_521) : "f"(S[30]), "f"(k_regs[14]), "f"(acc0_x_12));
                acc0_x_12 = _fma_521;
                float _fma_522;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_522) : "f"(S[31]), "f"(k_regs[15]), "f"(acc0_y_12));
                acc0_y_12 = _fma_522;
                float dot_12 = acc0_x_12 + acc0_y_12;
                float _shfl_xor_sync_523 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 1);
                dot_12 = dot_12 + _shfl_xor_sync_523;
                float _shfl_xor_sync_524 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 2);
                dot_12 = dot_12 + _shfl_xor_sync_524;
                float _shfl_xor_sync_525 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 4);
                dot_12 = dot_12 + _shfl_xor_sync_525;
#else
                float2 acc0_12 = make_float2(0.0f, 0.0f);
                acc0_12 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_12);
                float dot_12 = acc0_12.x + acc0_12.y;
                float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 1);
                dot_12 = dot_12 + _shfl_xor_sync_91;
                float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 2);
                dot_12 = dot_12 + _shfl_xor_sync_92;
                float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 4);
                dot_12 = dot_12 + _shfl_xor_sync_93;
#endif
                o4[1] = dot_12;
#if __CUDA_ARCH__ == 900
                float acc0_x_13 = 0.0f;
                float acc0_y_13 = 0.0f;
                float _fma_526;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_526) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_13));
                acc0_x_13 = _fma_526;
                float _fma_527;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_527) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_13));
                acc0_y_13 = _fma_527;
                float _fma_528;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_528) : "f"(S[34]), "f"(k_regs[2]), "f"(acc0_x_13));
                acc0_x_13 = _fma_528;
                float _fma_529;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_529) : "f"(S[35]), "f"(k_regs[3]), "f"(acc0_y_13));
                acc0_y_13 = _fma_529;
                float _fma_530;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_530) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_13));
                acc0_x_13 = _fma_530;
                float _fma_531;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_531) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_13));
                acc0_y_13 = _fma_531;
                float _fma_532;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_532) : "f"(S[38]), "f"(k_regs[6]), "f"(acc0_x_13));
                acc0_x_13 = _fma_532;
                float _fma_533;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_533) : "f"(S[39]), "f"(k_regs[7]), "f"(acc0_y_13));
                acc0_y_13 = _fma_533;
                float _fma_534;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_534) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_13));
                acc0_x_13 = _fma_534;
                float _fma_535;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_535) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_13));
                acc0_y_13 = _fma_535;
                float _fma_536;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_536) : "f"(S[42]), "f"(k_regs[10]), "f"(acc0_x_13));
                acc0_x_13 = _fma_536;
                float _fma_537;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_537) : "f"(S[43]), "f"(k_regs[11]), "f"(acc0_y_13));
                acc0_y_13 = _fma_537;
                float _fma_538;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_538) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_13));
                acc0_x_13 = _fma_538;
                float _fma_539;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_539) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_13));
                acc0_y_13 = _fma_539;
                float _fma_540;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_540) : "f"(S[46]), "f"(k_regs[14]), "f"(acc0_x_13));
                acc0_x_13 = _fma_540;
                float _fma_541;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_541) : "f"(S[47]), "f"(k_regs[15]), "f"(acc0_y_13));
                acc0_y_13 = _fma_541;
                float dot_13 = acc0_x_13 + acc0_y_13;
                float _shfl_xor_sync_542 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 1);
                dot_13 = dot_13 + _shfl_xor_sync_542;
                float _shfl_xor_sync_543 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 2);
                dot_13 = dot_13 + _shfl_xor_sync_543;
                float _shfl_xor_sync_544 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 4);
                dot_13 = dot_13 + _shfl_xor_sync_544;
#else
                float2 acc0_13 = make_float2(0.0f, 0.0f);
                acc0_13 = fma_f32x2_rn_ftz(S2[16], k2[0], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[17], k2[1], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[18], k2[2], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[19], k2[3], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[20], k2[4], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[21], k2[5], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[22], k2[6], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[23], k2[7], acc0_13);
                float dot_13 = acc0_13.x + acc0_13.y;
                float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 1);
                dot_13 = dot_13 + _shfl_xor_sync_94;
                float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 2);
                dot_13 = dot_13 + _shfl_xor_sync_95;
                float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 4);
                dot_13 = dot_13 + _shfl_xor_sync_96;
#endif
                o4[2] = dot_13;
#if __CUDA_ARCH__ == 900
                float acc0_x_14 = 0.0f;
                float acc0_y_14 = 0.0f;
                float _fma_545;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_545) : "f"(S[48]), "f"(k_regs[0]), "f"(acc0_x_14));
                acc0_x_14 = _fma_545;
                float _fma_546;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_546) : "f"(S[49]), "f"(k_regs[1]), "f"(acc0_y_14));
                acc0_y_14 = _fma_546;
                float _fma_547;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_547) : "f"(S[50]), "f"(k_regs[2]), "f"(acc0_x_14));
                acc0_x_14 = _fma_547;
                float _fma_548;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_548) : "f"(S[51]), "f"(k_regs[3]), "f"(acc0_y_14));
                acc0_y_14 = _fma_548;
                float _fma_549;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_549) : "f"(S[52]), "f"(k_regs[4]), "f"(acc0_x_14));
                acc0_x_14 = _fma_549;
                float _fma_550;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_550) : "f"(S[53]), "f"(k_regs[5]), "f"(acc0_y_14));
                acc0_y_14 = _fma_550;
                float _fma_551;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_551) : "f"(S[54]), "f"(k_regs[6]), "f"(acc0_x_14));
                acc0_x_14 = _fma_551;
                float _fma_552;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_552) : "f"(S[55]), "f"(k_regs[7]), "f"(acc0_y_14));
                acc0_y_14 = _fma_552;
                float _fma_553;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_553) : "f"(S[56]), "f"(k_regs[8]), "f"(acc0_x_14));
                acc0_x_14 = _fma_553;
                float _fma_554;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_554) : "f"(S[57]), "f"(k_regs[9]), "f"(acc0_y_14));
                acc0_y_14 = _fma_554;
                float _fma_555;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_555) : "f"(S[58]), "f"(k_regs[10]), "f"(acc0_x_14));
                acc0_x_14 = _fma_555;
                float _fma_556;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_556) : "f"(S[59]), "f"(k_regs[11]), "f"(acc0_y_14));
                acc0_y_14 = _fma_556;
                float _fma_557;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_557) : "f"(S[60]), "f"(k_regs[12]), "f"(acc0_x_14));
                acc0_x_14 = _fma_557;
                float _fma_558;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_558) : "f"(S[61]), "f"(k_regs[13]), "f"(acc0_y_14));
                acc0_y_14 = _fma_558;
                float _fma_559;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_559) : "f"(S[62]), "f"(k_regs[14]), "f"(acc0_x_14));
                acc0_x_14 = _fma_559;
                float _fma_560;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_560) : "f"(S[63]), "f"(k_regs[15]), "f"(acc0_y_14));
                acc0_y_14 = _fma_560;
                float dot_14 = acc0_x_14 + acc0_y_14;
                float _shfl_xor_sync_561 = __shfl_xor_sync(0xFFFFFFFF, dot_14, 1);
                dot_14 = dot_14 + _shfl_xor_sync_561;
                float _shfl_xor_sync_562 = __shfl_xor_sync(0xFFFFFFFF, dot_14, 2);
                dot_14 = dot_14 + _shfl_xor_sync_562;
                float _shfl_xor_sync_563 = __shfl_xor_sync(0xFFFFFFFF, dot_14, 4);
                dot_14 = dot_14 + _shfl_xor_sync_563;
#else
                float2 acc0_14 = make_float2(0.0f, 0.0f);
                acc0_14 = fma_f32x2_rn_ftz(S2[24], k2[0], acc0_14);
                acc0_14 = fma_f32x2_rn_ftz(S2[25], k2[1], acc0_14);
                acc0_14 = fma_f32x2_rn_ftz(S2[26], k2[2], acc0_14);
                acc0_14 = fma_f32x2_rn_ftz(S2[27], k2[3], acc0_14);
                acc0_14 = fma_f32x2_rn_ftz(S2[28], k2[4], acc0_14);
                acc0_14 = fma_f32x2_rn_ftz(S2[29], k2[5], acc0_14);
                acc0_14 = fma_f32x2_rn_ftz(S2[30], k2[6], acc0_14);
                acc0_14 = fma_f32x2_rn_ftz(S2[31], k2[7], acc0_14);
                float dot_14 = acc0_14.x + acc0_14.y;
                float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, dot_14, 1);
                dot_14 = dot_14 + _shfl_xor_sync_97;
                float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, dot_14, 2);
                dot_14 = dot_14 + _shfl_xor_sync_98;
                float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, dot_14, 4);
                dot_14 = dot_14 + _shfl_xor_sync_99;
#endif
                o4[3] = dot_14;
                if (u == 0) {
                    long long out_index = (long long)n * out_stride_p0 + (long long)t * out_stride_p1 + (long long)h * out_stride_p2;
                    unsigned int packed[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o4[_lp*2 + 0], o4[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o4[_lp*2 + 2], o4[_lp*2+1 + 2]));
                        packed[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int2*>(out + (out_index + (long long)row0))[0] = reinterpret_cast<int2*>(packed)[0];
                }
                int ssm_slot = sSsm[t];
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
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2])), "f"(S[32]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 1ll])), "f"(S[33]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 2ll])), "f"(S[34]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 2ll + 1ll])), "f"(S[35]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 32ll])), "f"(S[36]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 32ll + 1ll])), "f"(S[37]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 34ll])), "f"(S[38]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 34ll + 1ll])), "f"(S[39]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 64ll])), "f"(S[40]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 64ll + 1ll])), "f"(S[41]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 66ll])), "f"(S[42]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 66ll + 1ll])), "f"(S[43]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 96ll])), "f"(S[44]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 96ll + 1ll])), "f"(S[45]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 98ll])), "f"(S[46]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 98ll + 1ll])), "f"(S[47]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2])), "f"(S[48]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 1ll])), "f"(S[49]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 2ll])), "f"(S[50]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 2ll + 1ll])), "f"(S[51]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 32ll])), "f"(S[52]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 32ll + 1ll])), "f"(S[53]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 34ll])), "f"(S[54]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 34ll + 1ll])), "f"(S[55]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 64ll])), "f"(S[56]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 64ll + 1ll])), "f"(S[57]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 66ll])), "f"(S[58]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 66ll + 1ll])), "f"(S[59]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 96ll])), "f"(S[60]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 96ll + 1ll])), "f"(S[61]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 98ll])), "f"(S[62]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 98ll + 1ll])), "f"(S[63]), "r"((unsigned int)(snap_valid != 0)) : "memory");
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
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2])), "f"(S2[16].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 1ll])), "f"(S2[16].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 2ll])), "f"(S2[17].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 2ll + 1ll])), "f"(S2[17].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 32ll])), "f"(S2[18].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 32ll + 1ll])), "f"(S2[18].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 34ll])), "f"(S2[19].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 34ll + 1ll])), "f"(S2[19].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 64ll])), "f"(S2[20].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 64ll + 1ll])), "f"(S2[20].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 66ll])), "f"(S2[21].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 66ll + 1ll])), "f"(S2[21].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 96ll])), "f"(S2[22].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 96ll + 1ll])), "f"(S2[22].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 98ll])), "f"(S2[23].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 2ll * state_stride_p2 + 98ll + 1ll])), "f"(S2[23].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2])), "f"(S2[24].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 1ll])), "f"(S2[24].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 2ll])), "f"(S2[25].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 2ll + 1ll])), "f"(S2[25].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 32ll])), "f"(S2[26].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 32ll + 1ll])), "f"(S2[26].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 34ll])), "f"(S2[27].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 34ll + 1ll])), "f"(S2[27].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 64ll])), "f"(S2[28].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 64ll + 1ll])), "f"(S2[28].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 66ll])), "f"(S2[29].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 66ll + 1ll])), "f"(S2[29].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 96ll])), "f"(S2[30].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 96ll + 1ll])), "f"(S2[30].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 98ll])), "f"(S2[31].x), "r"((unsigned int)(snap_valid != 0)) : "memory");
                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(&state[snap_base + 3ll * state_stride_p2 + 98ll + 1ll])), "f"(S2[31].y), "r"((unsigned int)(snap_valid != 0)) : "memory");
#endif
            }
        }
        if (use_ssm_state_indices == 0) {
            #pragma unroll 1
            for (int t_1 = 0; t_1 < 7; t_1++) {
                float alpha_1 = sAlpha[t_1];
#if __CUDA_ARCH__ == 900
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
                S[32] = mul_f32_rn_ftz(S[32], alpha_1);
                S[33] = mul_f32_rn_ftz(S[33], alpha_1);
                S[34] = mul_f32_rn_ftz(S[34], alpha_1);
                S[35] = mul_f32_rn_ftz(S[35], alpha_1);
                S[36] = mul_f32_rn_ftz(S[36], alpha_1);
                S[37] = mul_f32_rn_ftz(S[37], alpha_1);
                S[38] = mul_f32_rn_ftz(S[38], alpha_1);
                S[39] = mul_f32_rn_ftz(S[39], alpha_1);
                S[40] = mul_f32_rn_ftz(S[40], alpha_1);
                S[41] = mul_f32_rn_ftz(S[41], alpha_1);
                S[42] = mul_f32_rn_ftz(S[42], alpha_1);
                S[43] = mul_f32_rn_ftz(S[43], alpha_1);
                S[44] = mul_f32_rn_ftz(S[44], alpha_1);
                S[45] = mul_f32_rn_ftz(S[45], alpha_1);
                S[46] = mul_f32_rn_ftz(S[46], alpha_1);
                S[47] = mul_f32_rn_ftz(S[47], alpha_1);
                S[48] = mul_f32_rn_ftz(S[48], alpha_1);
                S[49] = mul_f32_rn_ftz(S[49], alpha_1);
                S[50] = mul_f32_rn_ftz(S[50], alpha_1);
                S[51] = mul_f32_rn_ftz(S[51], alpha_1);
                S[52] = mul_f32_rn_ftz(S[52], alpha_1);
                S[53] = mul_f32_rn_ftz(S[53], alpha_1);
                S[54] = mul_f32_rn_ftz(S[54], alpha_1);
                S[55] = mul_f32_rn_ftz(S[55], alpha_1);
                S[56] = mul_f32_rn_ftz(S[56], alpha_1);
                S[57] = mul_f32_rn_ftz(S[57], alpha_1);
                S[58] = mul_f32_rn_ftz(S[58], alpha_1);
                S[59] = mul_f32_rn_ftz(S[59], alpha_1);
                S[60] = mul_f32_rn_ftz(S[60], alpha_1);
                S[61] = mul_f32_rn_ftz(S[61], alpha_1);
                S[62] = mul_f32_rn_ftz(S[62], alpha_1);
                S[63] = mul_f32_rn_ftz(S[63], alpha_1);
#else
                float2 alpha2_1 = make_float2(alpha_1, alpha_1);
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
                S2[16] = mul_f32x2(S2[16], alpha2_1);
                S2[17] = mul_f32x2(S2[17], alpha2_1);
                S2[18] = mul_f32x2(S2[18], alpha2_1);
                S2[19] = mul_f32x2(S2[19], alpha2_1);
                S2[20] = mul_f32x2(S2[20], alpha2_1);
                S2[21] = mul_f32x2(S2[21], alpha2_1);
                S2[22] = mul_f32x2(S2[22], alpha2_1);
                S2[23] = mul_f32x2(S2[23], alpha2_1);
                S2[24] = mul_f32x2(S2[24], alpha2_1);
                S2[25] = mul_f32x2(S2[25], alpha2_1);
                S2[26] = mul_f32x2(S2[26], alpha2_1);
                S2[27] = mul_f32x2(S2[27], alpha2_1);
                S2[28] = mul_f32x2(S2[28], alpha2_1);
                S2[29] = mul_f32x2(S2[29], alpha2_1);
                S2[30] = mul_f32x2(S2[30], alpha2_1);
                S2[31] = mul_f32x2(S2[31], alpha2_1);
#endif
                int m_3 = t_1 * 3;
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
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                    : "r"(sV_addr + (unsigned int)(m_3 * 256) + group_byte));
                float beta_3 = sBeta[m_3];
#if __CUDA_ARCH__ == 900
                float acc0_x_15 = 0.0f;
                float acc0_y_15 = 0.0f;
                float _fma_564;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_564) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_15));
                acc0_x_15 = _fma_564;
                float _fma_565;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_565) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_15));
                acc0_y_15 = _fma_565;
                float _fma_566;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_566) : "f"(S[2]), "f"(k_regs[2]), "f"(acc0_x_15));
                acc0_x_15 = _fma_566;
                float _fma_567;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_567) : "f"(S[3]), "f"(k_regs[3]), "f"(acc0_y_15));
                acc0_y_15 = _fma_567;
                float _fma_568;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_568) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_15));
                acc0_x_15 = _fma_568;
                float _fma_569;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_569) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_15));
                acc0_y_15 = _fma_569;
                float _fma_570;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_570) : "f"(S[6]), "f"(k_regs[6]), "f"(acc0_x_15));
                acc0_x_15 = _fma_570;
                float _fma_571;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_571) : "f"(S[7]), "f"(k_regs[7]), "f"(acc0_y_15));
                acc0_y_15 = _fma_571;
                float _fma_572;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_572) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_15));
                acc0_x_15 = _fma_572;
                float _fma_573;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_573) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_15));
                acc0_y_15 = _fma_573;
                float _fma_574;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_574) : "f"(S[10]), "f"(k_regs[10]), "f"(acc0_x_15));
                acc0_x_15 = _fma_574;
                float _fma_575;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_575) : "f"(S[11]), "f"(k_regs[11]), "f"(acc0_y_15));
                acc0_y_15 = _fma_575;
                float _fma_576;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_576) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_15));
                acc0_x_15 = _fma_576;
                float _fma_577;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_577) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_15));
                acc0_y_15 = _fma_577;
                float _fma_578;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_578) : "f"(S[14]), "f"(k_regs[14]), "f"(acc0_x_15));
                acc0_x_15 = _fma_578;
                float _fma_579;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_579) : "f"(S[15]), "f"(k_regs[15]), "f"(acc0_y_15));
                acc0_y_15 = _fma_579;
                float dot_15 = acc0_x_15 + acc0_y_15;
                float _shfl_xor_sync_580 = __shfl_xor_sync(0xFFFFFFFF, dot_15, 1);
                dot_15 = dot_15 + _shfl_xor_sync_580;
                float _shfl_xor_sync_581 = __shfl_xor_sync(0xFFFFFFFF, dot_15, 2);
                dot_15 = dot_15 + _shfl_xor_sync_581;
                float _shfl_xor_sync_582 = __shfl_xor_sync(0xFFFFFFFF, dot_15, 4);
                dot_15 = dot_15 + _shfl_xor_sync_582;
#else
                float2 acc0_15 = make_float2(0.0f, 0.0f);
                acc0_15 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_15);
                acc0_15 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_15);
                acc0_15 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_15);
                acc0_15 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_15);
                acc0_15 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_15);
                acc0_15 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_15);
                acc0_15 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_15);
                acc0_15 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_15);
                float dot_15 = acc0_15.x + acc0_15.y;
                float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, dot_15, 1);
                dot_15 = dot_15 + _shfl_xor_sync_100;
                float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, dot_15, 2);
                dot_15 = dot_15 + _shfl_xor_sync_101;
                float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, dot_15, 4);
                dot_15 = dot_15 + _shfl_xor_sync_102;
#endif
                float vn_12 = (v_regs[0] - dot_15) * beta_3;
#if __CUDA_ARCH__ == 900
                float _fma_583;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_583) : "f"(k_regs[0]), "f"(vn_12), "f"(S[0]));
                S[0] = _fma_583;
                float _fma_584;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_584) : "f"(k_regs[1]), "f"(vn_12), "f"(S[1]));
                S[1] = _fma_584;
                float _fma_585;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_585) : "f"(k_regs[2]), "f"(vn_12), "f"(S[2]));
                S[2] = _fma_585;
                float _fma_586;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_586) : "f"(k_regs[3]), "f"(vn_12), "f"(S[3]));
                S[3] = _fma_586;
                float _fma_587;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_587) : "f"(k_regs[4]), "f"(vn_12), "f"(S[4]));
                S[4] = _fma_587;
                float _fma_588;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_588) : "f"(k_regs[5]), "f"(vn_12), "f"(S[5]));
                S[5] = _fma_588;
                float _fma_589;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_589) : "f"(k_regs[6]), "f"(vn_12), "f"(S[6]));
                S[6] = _fma_589;
                float _fma_590;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_590) : "f"(k_regs[7]), "f"(vn_12), "f"(S[7]));
                S[7] = _fma_590;
                float _fma_591;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_591) : "f"(k_regs[8]), "f"(vn_12), "f"(S[8]));
                S[8] = _fma_591;
                float _fma_592;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_592) : "f"(k_regs[9]), "f"(vn_12), "f"(S[9]));
                S[9] = _fma_592;
                float _fma_593;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_593) : "f"(k_regs[10]), "f"(vn_12), "f"(S[10]));
                S[10] = _fma_593;
                float _fma_594;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_594) : "f"(k_regs[11]), "f"(vn_12), "f"(S[11]));
                S[11] = _fma_594;
                float _fma_595;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_595) : "f"(k_regs[12]), "f"(vn_12), "f"(S[12]));
                S[12] = _fma_595;
                float _fma_596;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_596) : "f"(k_regs[13]), "f"(vn_12), "f"(S[13]));
                S[13] = _fma_596;
                float _fma_597;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_597) : "f"(k_regs[14]), "f"(vn_12), "f"(S[14]));
                S[14] = _fma_597;
                float _fma_598;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_598) : "f"(k_regs[15]), "f"(vn_12), "f"(S[15]));
                S[15] = _fma_598;
                float acc0_x_15_1 = 0.0f;
                float acc0_y_15_1 = 0.0f;
                float _fma_599;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_599) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_15_1));
                acc0_x_15_1 = _fma_599;
                float _fma_600;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_600) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_15_1));
                acc0_y_15_1 = _fma_600;
                float _fma_601;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_601) : "f"(S[18]), "f"(k_regs[2]), "f"(acc0_x_15_1));
                acc0_x_15_1 = _fma_601;
                float _fma_602;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_602) : "f"(S[19]), "f"(k_regs[3]), "f"(acc0_y_15_1));
                acc0_y_15_1 = _fma_602;
                float _fma_603;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_603) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_15_1));
                acc0_x_15_1 = _fma_603;
                float _fma_604;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_604) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_15_1));
                acc0_y_15_1 = _fma_604;
                float _fma_605;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_605) : "f"(S[22]), "f"(k_regs[6]), "f"(acc0_x_15_1));
                acc0_x_15_1 = _fma_605;
                float _fma_606;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_606) : "f"(S[23]), "f"(k_regs[7]), "f"(acc0_y_15_1));
                acc0_y_15_1 = _fma_606;
                float _fma_607;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_607) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_15_1));
                acc0_x_15_1 = _fma_607;
                float _fma_608;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_608) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_15_1));
                acc0_y_15_1 = _fma_608;
                float _fma_609;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_609) : "f"(S[26]), "f"(k_regs[10]), "f"(acc0_x_15_1));
                acc0_x_15_1 = _fma_609;
                float _fma_610;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_610) : "f"(S[27]), "f"(k_regs[11]), "f"(acc0_y_15_1));
                acc0_y_15_1 = _fma_610;
                float _fma_611;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_611) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_15_1));
                acc0_x_15_1 = _fma_611;
                float _fma_612;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_612) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_15_1));
                acc0_y_15_1 = _fma_612;
                float _fma_613;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_613) : "f"(S[30]), "f"(k_regs[14]), "f"(acc0_x_15_1));
                acc0_x_15_1 = _fma_613;
                float _fma_614;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_614) : "f"(S[31]), "f"(k_regs[15]), "f"(acc0_y_15_1));
                acc0_y_15_1 = _fma_614;
                float dot_15_1 = acc0_x_15_1 + acc0_y_15_1;
                float _shfl_xor_sync_615 = __shfl_xor_sync(0xFFFFFFFF, dot_15_1, 1);
                dot_15_1 = dot_15_1 + _shfl_xor_sync_615;
                float _shfl_xor_sync_616 = __shfl_xor_sync(0xFFFFFFFF, dot_15_1, 2);
                dot_15_1 = dot_15_1 + _shfl_xor_sync_616;
                float _shfl_xor_sync_617 = __shfl_xor_sync(0xFFFFFFFF, dot_15_1, 4);
                dot_15_1 = dot_15_1 + _shfl_xor_sync_617;
#else
                float2 vn2_12 = make_float2(vn_12, vn_12);
                S2[0] = fma_f32x2_rn_ftz(k2[0], vn2_12, S2[0]);
                S2[1] = fma_f32x2_rn_ftz(k2[1], vn2_12, S2[1]);
                S2[2] = fma_f32x2_rn_ftz(k2[2], vn2_12, S2[2]);
                S2[3] = fma_f32x2_rn_ftz(k2[3], vn2_12, S2[3]);
                S2[4] = fma_f32x2_rn_ftz(k2[4], vn2_12, S2[4]);
                S2[5] = fma_f32x2_rn_ftz(k2[5], vn2_12, S2[5]);
                S2[6] = fma_f32x2_rn_ftz(k2[6], vn2_12, S2[6]);
                S2[7] = fma_f32x2_rn_ftz(k2[7], vn2_12, S2[7]);
                float2 acc0_15_1 = make_float2(0.0f, 0.0f);
                acc0_15_1 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_15_1);
                acc0_15_1 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_15_1);
                acc0_15_1 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_15_1);
                acc0_15_1 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_15_1);
                acc0_15_1 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_15_1);
                acc0_15_1 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_15_1);
                acc0_15_1 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_15_1);
                acc0_15_1 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_15_1);
                float dot_15_1 = acc0_15_1.x + acc0_15_1.y;
                float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, dot_15_1, 1);
                dot_15_1 = dot_15_1 + _shfl_xor_sync_103;
                float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, dot_15_1, 2);
                dot_15_1 = dot_15_1 + _shfl_xor_sync_104;
                float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, dot_15_1, 4);
                dot_15_1 = dot_15_1 + _shfl_xor_sync_105;
#endif
                float vn_13 = (v_regs[1] - dot_15_1) * beta_3;
#if __CUDA_ARCH__ == 900
                float _fma_618;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_618) : "f"(k_regs[0]), "f"(vn_13), "f"(S[16]));
                S[16] = _fma_618;
                float _fma_619;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_619) : "f"(k_regs[1]), "f"(vn_13), "f"(S[17]));
                S[17] = _fma_619;
                float _fma_620;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_620) : "f"(k_regs[2]), "f"(vn_13), "f"(S[18]));
                S[18] = _fma_620;
                float _fma_621;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_621) : "f"(k_regs[3]), "f"(vn_13), "f"(S[19]));
                S[19] = _fma_621;
                float _fma_622;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_622) : "f"(k_regs[4]), "f"(vn_13), "f"(S[20]));
                S[20] = _fma_622;
                float _fma_623;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_623) : "f"(k_regs[5]), "f"(vn_13), "f"(S[21]));
                S[21] = _fma_623;
                float _fma_624;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_624) : "f"(k_regs[6]), "f"(vn_13), "f"(S[22]));
                S[22] = _fma_624;
                float _fma_625;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_625) : "f"(k_regs[7]), "f"(vn_13), "f"(S[23]));
                S[23] = _fma_625;
                float _fma_626;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_626) : "f"(k_regs[8]), "f"(vn_13), "f"(S[24]));
                S[24] = _fma_626;
                float _fma_627;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_627) : "f"(k_regs[9]), "f"(vn_13), "f"(S[25]));
                S[25] = _fma_627;
                float _fma_628;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_628) : "f"(k_regs[10]), "f"(vn_13), "f"(S[26]));
                S[26] = _fma_628;
                float _fma_629;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_629) : "f"(k_regs[11]), "f"(vn_13), "f"(S[27]));
                S[27] = _fma_629;
                float _fma_630;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_630) : "f"(k_regs[12]), "f"(vn_13), "f"(S[28]));
                S[28] = _fma_630;
                float _fma_631;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_631) : "f"(k_regs[13]), "f"(vn_13), "f"(S[29]));
                S[29] = _fma_631;
                float _fma_632;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_632) : "f"(k_regs[14]), "f"(vn_13), "f"(S[30]));
                S[30] = _fma_632;
                float _fma_633;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_633) : "f"(k_regs[15]), "f"(vn_13), "f"(S[31]));
                S[31] = _fma_633;
                float acc0_x_16 = 0.0f;
                float acc0_y_16 = 0.0f;
                float _fma_634;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_634) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_16));
                acc0_x_16 = _fma_634;
                float _fma_635;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_635) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_16));
                acc0_y_16 = _fma_635;
                float _fma_636;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_636) : "f"(S[34]), "f"(k_regs[2]), "f"(acc0_x_16));
                acc0_x_16 = _fma_636;
                float _fma_637;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_637) : "f"(S[35]), "f"(k_regs[3]), "f"(acc0_y_16));
                acc0_y_16 = _fma_637;
                float _fma_638;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_638) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_16));
                acc0_x_16 = _fma_638;
                float _fma_639;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_639) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_16));
                acc0_y_16 = _fma_639;
                float _fma_640;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_640) : "f"(S[38]), "f"(k_regs[6]), "f"(acc0_x_16));
                acc0_x_16 = _fma_640;
                float _fma_641;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_641) : "f"(S[39]), "f"(k_regs[7]), "f"(acc0_y_16));
                acc0_y_16 = _fma_641;
                float _fma_642;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_642) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_16));
                acc0_x_16 = _fma_642;
                float _fma_643;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_643) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_16));
                acc0_y_16 = _fma_643;
                float _fma_644;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_644) : "f"(S[42]), "f"(k_regs[10]), "f"(acc0_x_16));
                acc0_x_16 = _fma_644;
                float _fma_645;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_645) : "f"(S[43]), "f"(k_regs[11]), "f"(acc0_y_16));
                acc0_y_16 = _fma_645;
                float _fma_646;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_646) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_16));
                acc0_x_16 = _fma_646;
                float _fma_647;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_647) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_16));
                acc0_y_16 = _fma_647;
                float _fma_648;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_648) : "f"(S[46]), "f"(k_regs[14]), "f"(acc0_x_16));
                acc0_x_16 = _fma_648;
                float _fma_649;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_649) : "f"(S[47]), "f"(k_regs[15]), "f"(acc0_y_16));
                acc0_y_16 = _fma_649;
                float dot_16 = acc0_x_16 + acc0_y_16;
                float _shfl_xor_sync_650 = __shfl_xor_sync(0xFFFFFFFF, dot_16, 1);
                dot_16 = dot_16 + _shfl_xor_sync_650;
                float _shfl_xor_sync_651 = __shfl_xor_sync(0xFFFFFFFF, dot_16, 2);
                dot_16 = dot_16 + _shfl_xor_sync_651;
                float _shfl_xor_sync_652 = __shfl_xor_sync(0xFFFFFFFF, dot_16, 4);
                dot_16 = dot_16 + _shfl_xor_sync_652;
#else
                float2 vn2_13 = make_float2(vn_13, vn_13);
                S2[8] = fma_f32x2_rn_ftz(k2[0], vn2_13, S2[8]);
                S2[9] = fma_f32x2_rn_ftz(k2[1], vn2_13, S2[9]);
                S2[10] = fma_f32x2_rn_ftz(k2[2], vn2_13, S2[10]);
                S2[11] = fma_f32x2_rn_ftz(k2[3], vn2_13, S2[11]);
                S2[12] = fma_f32x2_rn_ftz(k2[4], vn2_13, S2[12]);
                S2[13] = fma_f32x2_rn_ftz(k2[5], vn2_13, S2[13]);
                S2[14] = fma_f32x2_rn_ftz(k2[6], vn2_13, S2[14]);
                S2[15] = fma_f32x2_rn_ftz(k2[7], vn2_13, S2[15]);
                float2 acc0_16 = make_float2(0.0f, 0.0f);
                acc0_16 = fma_f32x2_rn_ftz(S2[16], k2[0], acc0_16);
                acc0_16 = fma_f32x2_rn_ftz(S2[17], k2[1], acc0_16);
                acc0_16 = fma_f32x2_rn_ftz(S2[18], k2[2], acc0_16);
                acc0_16 = fma_f32x2_rn_ftz(S2[19], k2[3], acc0_16);
                acc0_16 = fma_f32x2_rn_ftz(S2[20], k2[4], acc0_16);
                acc0_16 = fma_f32x2_rn_ftz(S2[21], k2[5], acc0_16);
                acc0_16 = fma_f32x2_rn_ftz(S2[22], k2[6], acc0_16);
                acc0_16 = fma_f32x2_rn_ftz(S2[23], k2[7], acc0_16);
                float dot_16 = acc0_16.x + acc0_16.y;
                float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, dot_16, 1);
                dot_16 = dot_16 + _shfl_xor_sync_106;
                float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, dot_16, 2);
                dot_16 = dot_16 + _shfl_xor_sync_107;
                float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, dot_16, 4);
                dot_16 = dot_16 + _shfl_xor_sync_108;
#endif
                float vn_14 = (v_regs[2] - dot_16) * beta_3;
#if __CUDA_ARCH__ == 900
                float _fma_653;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_653) : "f"(k_regs[0]), "f"(vn_14), "f"(S[32]));
                S[32] = _fma_653;
                float _fma_654;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_654) : "f"(k_regs[1]), "f"(vn_14), "f"(S[33]));
                S[33] = _fma_654;
                float _fma_655;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_655) : "f"(k_regs[2]), "f"(vn_14), "f"(S[34]));
                S[34] = _fma_655;
                float _fma_656;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_656) : "f"(k_regs[3]), "f"(vn_14), "f"(S[35]));
                S[35] = _fma_656;
                float _fma_657;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_657) : "f"(k_regs[4]), "f"(vn_14), "f"(S[36]));
                S[36] = _fma_657;
                float _fma_658;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_658) : "f"(k_regs[5]), "f"(vn_14), "f"(S[37]));
                S[37] = _fma_658;
                float _fma_659;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_659) : "f"(k_regs[6]), "f"(vn_14), "f"(S[38]));
                S[38] = _fma_659;
                float _fma_660;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_660) : "f"(k_regs[7]), "f"(vn_14), "f"(S[39]));
                S[39] = _fma_660;
                float _fma_661;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_661) : "f"(k_regs[8]), "f"(vn_14), "f"(S[40]));
                S[40] = _fma_661;
                float _fma_662;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_662) : "f"(k_regs[9]), "f"(vn_14), "f"(S[41]));
                S[41] = _fma_662;
                float _fma_663;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_663) : "f"(k_regs[10]), "f"(vn_14), "f"(S[42]));
                S[42] = _fma_663;
                float _fma_664;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_664) : "f"(k_regs[11]), "f"(vn_14), "f"(S[43]));
                S[43] = _fma_664;
                float _fma_665;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_665) : "f"(k_regs[12]), "f"(vn_14), "f"(S[44]));
                S[44] = _fma_665;
                float _fma_666;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_666) : "f"(k_regs[13]), "f"(vn_14), "f"(S[45]));
                S[45] = _fma_666;
                float _fma_667;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_667) : "f"(k_regs[14]), "f"(vn_14), "f"(S[46]));
                S[46] = _fma_667;
                float _fma_668;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_668) : "f"(k_regs[15]), "f"(vn_14), "f"(S[47]));
                S[47] = _fma_668;
                float acc0_x_17 = 0.0f;
                float acc0_y_17 = 0.0f;
                float _fma_669;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_669) : "f"(S[48]), "f"(k_regs[0]), "f"(acc0_x_17));
                acc0_x_17 = _fma_669;
                float _fma_670;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_670) : "f"(S[49]), "f"(k_regs[1]), "f"(acc0_y_17));
                acc0_y_17 = _fma_670;
                float _fma_671;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_671) : "f"(S[50]), "f"(k_regs[2]), "f"(acc0_x_17));
                acc0_x_17 = _fma_671;
                float _fma_672;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_672) : "f"(S[51]), "f"(k_regs[3]), "f"(acc0_y_17));
                acc0_y_17 = _fma_672;
                float _fma_673;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_673) : "f"(S[52]), "f"(k_regs[4]), "f"(acc0_x_17));
                acc0_x_17 = _fma_673;
                float _fma_674;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_674) : "f"(S[53]), "f"(k_regs[5]), "f"(acc0_y_17));
                acc0_y_17 = _fma_674;
                float _fma_675;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_675) : "f"(S[54]), "f"(k_regs[6]), "f"(acc0_x_17));
                acc0_x_17 = _fma_675;
                float _fma_676;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_676) : "f"(S[55]), "f"(k_regs[7]), "f"(acc0_y_17));
                acc0_y_17 = _fma_676;
                float _fma_677;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_677) : "f"(S[56]), "f"(k_regs[8]), "f"(acc0_x_17));
                acc0_x_17 = _fma_677;
                float _fma_678;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_678) : "f"(S[57]), "f"(k_regs[9]), "f"(acc0_y_17));
                acc0_y_17 = _fma_678;
                float _fma_679;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_679) : "f"(S[58]), "f"(k_regs[10]), "f"(acc0_x_17));
                acc0_x_17 = _fma_679;
                float _fma_680;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_680) : "f"(S[59]), "f"(k_regs[11]), "f"(acc0_y_17));
                acc0_y_17 = _fma_680;
                float _fma_681;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_681) : "f"(S[60]), "f"(k_regs[12]), "f"(acc0_x_17));
                acc0_x_17 = _fma_681;
                float _fma_682;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_682) : "f"(S[61]), "f"(k_regs[13]), "f"(acc0_y_17));
                acc0_y_17 = _fma_682;
                float _fma_683;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_683) : "f"(S[62]), "f"(k_regs[14]), "f"(acc0_x_17));
                acc0_x_17 = _fma_683;
                float _fma_684;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_684) : "f"(S[63]), "f"(k_regs[15]), "f"(acc0_y_17));
                acc0_y_17 = _fma_684;
                float dot_17 = acc0_x_17 + acc0_y_17;
                float _shfl_xor_sync_685 = __shfl_xor_sync(0xFFFFFFFF, dot_17, 1);
                dot_17 = dot_17 + _shfl_xor_sync_685;
                float _shfl_xor_sync_686 = __shfl_xor_sync(0xFFFFFFFF, dot_17, 2);
                dot_17 = dot_17 + _shfl_xor_sync_686;
                float _shfl_xor_sync_687 = __shfl_xor_sync(0xFFFFFFFF, dot_17, 4);
                dot_17 = dot_17 + _shfl_xor_sync_687;
#else
                float2 vn2_14 = make_float2(vn_14, vn_14);
                S2[16] = fma_f32x2_rn_ftz(k2[0], vn2_14, S2[16]);
                S2[17] = fma_f32x2_rn_ftz(k2[1], vn2_14, S2[17]);
                S2[18] = fma_f32x2_rn_ftz(k2[2], vn2_14, S2[18]);
                S2[19] = fma_f32x2_rn_ftz(k2[3], vn2_14, S2[19]);
                S2[20] = fma_f32x2_rn_ftz(k2[4], vn2_14, S2[20]);
                S2[21] = fma_f32x2_rn_ftz(k2[5], vn2_14, S2[21]);
                S2[22] = fma_f32x2_rn_ftz(k2[6], vn2_14, S2[22]);
                S2[23] = fma_f32x2_rn_ftz(k2[7], vn2_14, S2[23]);
                float2 acc0_17 = make_float2(0.0f, 0.0f);
                acc0_17 = fma_f32x2_rn_ftz(S2[24], k2[0], acc0_17);
                acc0_17 = fma_f32x2_rn_ftz(S2[25], k2[1], acc0_17);
                acc0_17 = fma_f32x2_rn_ftz(S2[26], k2[2], acc0_17);
                acc0_17 = fma_f32x2_rn_ftz(S2[27], k2[3], acc0_17);
                acc0_17 = fma_f32x2_rn_ftz(S2[28], k2[4], acc0_17);
                acc0_17 = fma_f32x2_rn_ftz(S2[29], k2[5], acc0_17);
                acc0_17 = fma_f32x2_rn_ftz(S2[30], k2[6], acc0_17);
                acc0_17 = fma_f32x2_rn_ftz(S2[31], k2[7], acc0_17);
                float dot_17 = acc0_17.x + acc0_17.y;
                float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, dot_17, 1);
                dot_17 = dot_17 + _shfl_xor_sync_109;
                float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, dot_17, 2);
                dot_17 = dot_17 + _shfl_xor_sync_110;
                float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, dot_17, 4);
                dot_17 = dot_17 + _shfl_xor_sync_111;
#endif
                float vn_15 = (v_regs[3] - dot_17) * beta_3;
#if __CUDA_ARCH__ == 900
                float _fma_688;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_688) : "f"(k_regs[0]), "f"(vn_15), "f"(S[48]));
                S[48] = _fma_688;
                float _fma_689;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_689) : "f"(k_regs[1]), "f"(vn_15), "f"(S[49]));
                S[49] = _fma_689;
                float _fma_690;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_690) : "f"(k_regs[2]), "f"(vn_15), "f"(S[50]));
                S[50] = _fma_690;
                float _fma_691;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_691) : "f"(k_regs[3]), "f"(vn_15), "f"(S[51]));
                S[51] = _fma_691;
                float _fma_692;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_692) : "f"(k_regs[4]), "f"(vn_15), "f"(S[52]));
                S[52] = _fma_692;
                float _fma_693;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_693) : "f"(k_regs[5]), "f"(vn_15), "f"(S[53]));
                S[53] = _fma_693;
                float _fma_694;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_694) : "f"(k_regs[6]), "f"(vn_15), "f"(S[54]));
                S[54] = _fma_694;
                float _fma_695;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_695) : "f"(k_regs[7]), "f"(vn_15), "f"(S[55]));
                S[55] = _fma_695;
                float _fma_696;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_696) : "f"(k_regs[8]), "f"(vn_15), "f"(S[56]));
                S[56] = _fma_696;
                float _fma_697;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_697) : "f"(k_regs[9]), "f"(vn_15), "f"(S[57]));
                S[57] = _fma_697;
                float _fma_698;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_698) : "f"(k_regs[10]), "f"(vn_15), "f"(S[58]));
                S[58] = _fma_698;
                float _fma_699;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_699) : "f"(k_regs[11]), "f"(vn_15), "f"(S[59]));
                S[59] = _fma_699;
                float _fma_700;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_700) : "f"(k_regs[12]), "f"(vn_15), "f"(S[60]));
                S[60] = _fma_700;
                float _fma_701;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_701) : "f"(k_regs[13]), "f"(vn_15), "f"(S[61]));
                S[61] = _fma_701;
                float _fma_702;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_702) : "f"(k_regs[14]), "f"(vn_15), "f"(S[62]));
                S[62] = _fma_702;
                float _fma_703;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_703) : "f"(k_regs[15]), "f"(vn_15), "f"(S[63]));
                S[63] = _fma_703;
#else
                float2 vn2_15 = make_float2(vn_15, vn_15);
                S2[24] = fma_f32x2_rn_ftz(k2[0], vn2_15, S2[24]);
                S2[25] = fma_f32x2_rn_ftz(k2[1], vn2_15, S2[25]);
                S2[26] = fma_f32x2_rn_ftz(k2[2], vn2_15, S2[26]);
                S2[27] = fma_f32x2_rn_ftz(k2[3], vn2_15, S2[27]);
                S2[28] = fma_f32x2_rn_ftz(k2[4], vn2_15, S2[28]);
                S2[29] = fma_f32x2_rn_ftz(k2[5], vn2_15, S2[29]);
                S2[30] = fma_f32x2_rn_ftz(k2[6], vn2_15, S2[30]);
                S2[31] = fma_f32x2_rn_ftz(k2[7], vn2_15, S2[31]);
#endif
                int m_4 = t_1 * 3 + 1;
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
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                    : "r"(sV_addr + (unsigned int)(m_4 * 256) + group_byte));
                float beta_4 = sBeta[m_4];
#if __CUDA_ARCH__ == 900
                float acc0_x_18 = 0.0f;
                float acc0_y_18 = 0.0f;
                float _fma_704;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_704) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_18));
                acc0_x_18 = _fma_704;
                float _fma_705;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_705) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_18));
                acc0_y_18 = _fma_705;
                float _fma_706;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_706) : "f"(S[2]), "f"(k_regs[2]), "f"(acc0_x_18));
                acc0_x_18 = _fma_706;
                float _fma_707;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_707) : "f"(S[3]), "f"(k_regs[3]), "f"(acc0_y_18));
                acc0_y_18 = _fma_707;
                float _fma_708;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_708) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_18));
                acc0_x_18 = _fma_708;
                float _fma_709;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_709) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_18));
                acc0_y_18 = _fma_709;
                float _fma_710;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_710) : "f"(S[6]), "f"(k_regs[6]), "f"(acc0_x_18));
                acc0_x_18 = _fma_710;
                float _fma_711;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_711) : "f"(S[7]), "f"(k_regs[7]), "f"(acc0_y_18));
                acc0_y_18 = _fma_711;
                float _fma_712;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_712) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_18));
                acc0_x_18 = _fma_712;
                float _fma_713;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_713) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_18));
                acc0_y_18 = _fma_713;
                float _fma_714;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_714) : "f"(S[10]), "f"(k_regs[10]), "f"(acc0_x_18));
                acc0_x_18 = _fma_714;
                float _fma_715;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_715) : "f"(S[11]), "f"(k_regs[11]), "f"(acc0_y_18));
                acc0_y_18 = _fma_715;
                float _fma_716;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_716) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_18));
                acc0_x_18 = _fma_716;
                float _fma_717;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_717) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_18));
                acc0_y_18 = _fma_717;
                float _fma_718;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_718) : "f"(S[14]), "f"(k_regs[14]), "f"(acc0_x_18));
                acc0_x_18 = _fma_718;
                float _fma_719;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_719) : "f"(S[15]), "f"(k_regs[15]), "f"(acc0_y_18));
                acc0_y_18 = _fma_719;
                float dot_18 = acc0_x_18 + acc0_y_18;
                float _shfl_xor_sync_720 = __shfl_xor_sync(0xFFFFFFFF, dot_18, 1);
                dot_18 = dot_18 + _shfl_xor_sync_720;
                float _shfl_xor_sync_721 = __shfl_xor_sync(0xFFFFFFFF, dot_18, 2);
                dot_18 = dot_18 + _shfl_xor_sync_721;
                float _shfl_xor_sync_722 = __shfl_xor_sync(0xFFFFFFFF, dot_18, 4);
                dot_18 = dot_18 + _shfl_xor_sync_722;
#else
                float2 acc0_18 = make_float2(0.0f, 0.0f);
                acc0_18 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_18);
                acc0_18 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_18);
                acc0_18 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_18);
                acc0_18 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_18);
                acc0_18 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_18);
                acc0_18 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_18);
                acc0_18 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_18);
                acc0_18 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_18);
                float dot_18 = acc0_18.x + acc0_18.y;
                float _shfl_xor_sync_112 = __shfl_xor_sync(0xFFFFFFFF, dot_18, 1);
                dot_18 = dot_18 + _shfl_xor_sync_112;
                float _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, dot_18, 2);
                dot_18 = dot_18 + _shfl_xor_sync_113;
                float _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, dot_18, 4);
                dot_18 = dot_18 + _shfl_xor_sync_114;
#endif
                float vn_16 = (v_regs[0] - dot_18) * beta_4;
#if __CUDA_ARCH__ == 900
                float _fma_723;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_723) : "f"(k_regs[0]), "f"(vn_16), "f"(S[0]));
                S[0] = _fma_723;
                float _fma_724;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_724) : "f"(k_regs[1]), "f"(vn_16), "f"(S[1]));
                S[1] = _fma_724;
                float _fma_725;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_725) : "f"(k_regs[2]), "f"(vn_16), "f"(S[2]));
                S[2] = _fma_725;
                float _fma_726;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_726) : "f"(k_regs[3]), "f"(vn_16), "f"(S[3]));
                S[3] = _fma_726;
                float _fma_727;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_727) : "f"(k_regs[4]), "f"(vn_16), "f"(S[4]));
                S[4] = _fma_727;
                float _fma_728;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_728) : "f"(k_regs[5]), "f"(vn_16), "f"(S[5]));
                S[5] = _fma_728;
                float _fma_729;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_729) : "f"(k_regs[6]), "f"(vn_16), "f"(S[6]));
                S[6] = _fma_729;
                float _fma_730;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_730) : "f"(k_regs[7]), "f"(vn_16), "f"(S[7]));
                S[7] = _fma_730;
                float _fma_731;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_731) : "f"(k_regs[8]), "f"(vn_16), "f"(S[8]));
                S[8] = _fma_731;
                float _fma_732;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_732) : "f"(k_regs[9]), "f"(vn_16), "f"(S[9]));
                S[9] = _fma_732;
                float _fma_733;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_733) : "f"(k_regs[10]), "f"(vn_16), "f"(S[10]));
                S[10] = _fma_733;
                float _fma_734;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_734) : "f"(k_regs[11]), "f"(vn_16), "f"(S[11]));
                S[11] = _fma_734;
                float _fma_735;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_735) : "f"(k_regs[12]), "f"(vn_16), "f"(S[12]));
                S[12] = _fma_735;
                float _fma_736;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_736) : "f"(k_regs[13]), "f"(vn_16), "f"(S[13]));
                S[13] = _fma_736;
                float _fma_737;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_737) : "f"(k_regs[14]), "f"(vn_16), "f"(S[14]));
                S[14] = _fma_737;
                float _fma_738;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_738) : "f"(k_regs[15]), "f"(vn_16), "f"(S[15]));
                S[15] = _fma_738;
                float acc0_x_19 = 0.0f;
                float acc0_y_19 = 0.0f;
                float _fma_739;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_739) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_19));
                acc0_x_19 = _fma_739;
                float _fma_740;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_740) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_19));
                acc0_y_19 = _fma_740;
                float _fma_741;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_741) : "f"(S[18]), "f"(k_regs[2]), "f"(acc0_x_19));
                acc0_x_19 = _fma_741;
                float _fma_742;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_742) : "f"(S[19]), "f"(k_regs[3]), "f"(acc0_y_19));
                acc0_y_19 = _fma_742;
                float _fma_743;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_743) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_19));
                acc0_x_19 = _fma_743;
                float _fma_744;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_744) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_19));
                acc0_y_19 = _fma_744;
                float _fma_745;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_745) : "f"(S[22]), "f"(k_regs[6]), "f"(acc0_x_19));
                acc0_x_19 = _fma_745;
                float _fma_746;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_746) : "f"(S[23]), "f"(k_regs[7]), "f"(acc0_y_19));
                acc0_y_19 = _fma_746;
                float _fma_747;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_747) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_19));
                acc0_x_19 = _fma_747;
                float _fma_748;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_748) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_19));
                acc0_y_19 = _fma_748;
                float _fma_749;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_749) : "f"(S[26]), "f"(k_regs[10]), "f"(acc0_x_19));
                acc0_x_19 = _fma_749;
                float _fma_750;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_750) : "f"(S[27]), "f"(k_regs[11]), "f"(acc0_y_19));
                acc0_y_19 = _fma_750;
                float _fma_751;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_751) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_19));
                acc0_x_19 = _fma_751;
                float _fma_752;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_752) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_19));
                acc0_y_19 = _fma_752;
                float _fma_753;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_753) : "f"(S[30]), "f"(k_regs[14]), "f"(acc0_x_19));
                acc0_x_19 = _fma_753;
                float _fma_754;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_754) : "f"(S[31]), "f"(k_regs[15]), "f"(acc0_y_19));
                acc0_y_19 = _fma_754;
                float dot_19 = acc0_x_19 + acc0_y_19;
                float _shfl_xor_sync_755 = __shfl_xor_sync(0xFFFFFFFF, dot_19, 1);
                dot_19 = dot_19 + _shfl_xor_sync_755;
                float _shfl_xor_sync_756 = __shfl_xor_sync(0xFFFFFFFF, dot_19, 2);
                dot_19 = dot_19 + _shfl_xor_sync_756;
                float _shfl_xor_sync_757 = __shfl_xor_sync(0xFFFFFFFF, dot_19, 4);
                dot_19 = dot_19 + _shfl_xor_sync_757;
#else
                float2 vn2_16 = make_float2(vn_16, vn_16);
                S2[0] = fma_f32x2_rn_ftz(k2[0], vn2_16, S2[0]);
                S2[1] = fma_f32x2_rn_ftz(k2[1], vn2_16, S2[1]);
                S2[2] = fma_f32x2_rn_ftz(k2[2], vn2_16, S2[2]);
                S2[3] = fma_f32x2_rn_ftz(k2[3], vn2_16, S2[3]);
                S2[4] = fma_f32x2_rn_ftz(k2[4], vn2_16, S2[4]);
                S2[5] = fma_f32x2_rn_ftz(k2[5], vn2_16, S2[5]);
                S2[6] = fma_f32x2_rn_ftz(k2[6], vn2_16, S2[6]);
                S2[7] = fma_f32x2_rn_ftz(k2[7], vn2_16, S2[7]);
                float2 acc0_19 = make_float2(0.0f, 0.0f);
                acc0_19 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_19);
                acc0_19 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_19);
                acc0_19 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_19);
                acc0_19 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_19);
                acc0_19 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_19);
                acc0_19 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_19);
                acc0_19 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_19);
                acc0_19 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_19);
                float dot_19 = acc0_19.x + acc0_19.y;
                float _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, dot_19, 1);
                dot_19 = dot_19 + _shfl_xor_sync_115;
                float _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, dot_19, 2);
                dot_19 = dot_19 + _shfl_xor_sync_116;
                float _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, dot_19, 4);
                dot_19 = dot_19 + _shfl_xor_sync_117;
#endif
                float vn_17 = (v_regs[1] - dot_19) * beta_4;
#if __CUDA_ARCH__ == 900
                float _fma_758;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_758) : "f"(k_regs[0]), "f"(vn_17), "f"(S[16]));
                S[16] = _fma_758;
                float _fma_759;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_759) : "f"(k_regs[1]), "f"(vn_17), "f"(S[17]));
                S[17] = _fma_759;
                float _fma_760;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_760) : "f"(k_regs[2]), "f"(vn_17), "f"(S[18]));
                S[18] = _fma_760;
                float _fma_761;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_761) : "f"(k_regs[3]), "f"(vn_17), "f"(S[19]));
                S[19] = _fma_761;
                float _fma_762;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_762) : "f"(k_regs[4]), "f"(vn_17), "f"(S[20]));
                S[20] = _fma_762;
                float _fma_763;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_763) : "f"(k_regs[5]), "f"(vn_17), "f"(S[21]));
                S[21] = _fma_763;
                float _fma_764;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_764) : "f"(k_regs[6]), "f"(vn_17), "f"(S[22]));
                S[22] = _fma_764;
                float _fma_765;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_765) : "f"(k_regs[7]), "f"(vn_17), "f"(S[23]));
                S[23] = _fma_765;
                float _fma_766;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_766) : "f"(k_regs[8]), "f"(vn_17), "f"(S[24]));
                S[24] = _fma_766;
                float _fma_767;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_767) : "f"(k_regs[9]), "f"(vn_17), "f"(S[25]));
                S[25] = _fma_767;
                float _fma_768;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_768) : "f"(k_regs[10]), "f"(vn_17), "f"(S[26]));
                S[26] = _fma_768;
                float _fma_769;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_769) : "f"(k_regs[11]), "f"(vn_17), "f"(S[27]));
                S[27] = _fma_769;
                float _fma_770;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_770) : "f"(k_regs[12]), "f"(vn_17), "f"(S[28]));
                S[28] = _fma_770;
                float _fma_771;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_771) : "f"(k_regs[13]), "f"(vn_17), "f"(S[29]));
                S[29] = _fma_771;
                float _fma_772;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_772) : "f"(k_regs[14]), "f"(vn_17), "f"(S[30]));
                S[30] = _fma_772;
                float _fma_773;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_773) : "f"(k_regs[15]), "f"(vn_17), "f"(S[31]));
                S[31] = _fma_773;
                float acc0_x_20 = 0.0f;
                float acc0_y_20 = 0.0f;
                float _fma_774;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_774) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_20));
                acc0_x_20 = _fma_774;
                float _fma_775;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_775) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_20));
                acc0_y_20 = _fma_775;
                float _fma_776;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_776) : "f"(S[34]), "f"(k_regs[2]), "f"(acc0_x_20));
                acc0_x_20 = _fma_776;
                float _fma_777;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_777) : "f"(S[35]), "f"(k_regs[3]), "f"(acc0_y_20));
                acc0_y_20 = _fma_777;
                float _fma_778;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_778) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_20));
                acc0_x_20 = _fma_778;
                float _fma_779;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_779) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_20));
                acc0_y_20 = _fma_779;
                float _fma_780;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_780) : "f"(S[38]), "f"(k_regs[6]), "f"(acc0_x_20));
                acc0_x_20 = _fma_780;
                float _fma_781;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_781) : "f"(S[39]), "f"(k_regs[7]), "f"(acc0_y_20));
                acc0_y_20 = _fma_781;
                float _fma_782;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_782) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_20));
                acc0_x_20 = _fma_782;
                float _fma_783;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_783) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_20));
                acc0_y_20 = _fma_783;
                float _fma_784;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_784) : "f"(S[42]), "f"(k_regs[10]), "f"(acc0_x_20));
                acc0_x_20 = _fma_784;
                float _fma_785;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_785) : "f"(S[43]), "f"(k_regs[11]), "f"(acc0_y_20));
                acc0_y_20 = _fma_785;
                float _fma_786;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_786) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_20));
                acc0_x_20 = _fma_786;
                float _fma_787;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_787) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_20));
                acc0_y_20 = _fma_787;
                float _fma_788;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_788) : "f"(S[46]), "f"(k_regs[14]), "f"(acc0_x_20));
                acc0_x_20 = _fma_788;
                float _fma_789;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_789) : "f"(S[47]), "f"(k_regs[15]), "f"(acc0_y_20));
                acc0_y_20 = _fma_789;
                float dot_20 = acc0_x_20 + acc0_y_20;
                float _shfl_xor_sync_790 = __shfl_xor_sync(0xFFFFFFFF, dot_20, 1);
                dot_20 = dot_20 + _shfl_xor_sync_790;
                float _shfl_xor_sync_791 = __shfl_xor_sync(0xFFFFFFFF, dot_20, 2);
                dot_20 = dot_20 + _shfl_xor_sync_791;
                float _shfl_xor_sync_792 = __shfl_xor_sync(0xFFFFFFFF, dot_20, 4);
                dot_20 = dot_20 + _shfl_xor_sync_792;
#else
                float2 vn2_17 = make_float2(vn_17, vn_17);
                S2[8] = fma_f32x2_rn_ftz(k2[0], vn2_17, S2[8]);
                S2[9] = fma_f32x2_rn_ftz(k2[1], vn2_17, S2[9]);
                S2[10] = fma_f32x2_rn_ftz(k2[2], vn2_17, S2[10]);
                S2[11] = fma_f32x2_rn_ftz(k2[3], vn2_17, S2[11]);
                S2[12] = fma_f32x2_rn_ftz(k2[4], vn2_17, S2[12]);
                S2[13] = fma_f32x2_rn_ftz(k2[5], vn2_17, S2[13]);
                S2[14] = fma_f32x2_rn_ftz(k2[6], vn2_17, S2[14]);
                S2[15] = fma_f32x2_rn_ftz(k2[7], vn2_17, S2[15]);
                float2 acc0_20 = make_float2(0.0f, 0.0f);
                acc0_20 = fma_f32x2_rn_ftz(S2[16], k2[0], acc0_20);
                acc0_20 = fma_f32x2_rn_ftz(S2[17], k2[1], acc0_20);
                acc0_20 = fma_f32x2_rn_ftz(S2[18], k2[2], acc0_20);
                acc0_20 = fma_f32x2_rn_ftz(S2[19], k2[3], acc0_20);
                acc0_20 = fma_f32x2_rn_ftz(S2[20], k2[4], acc0_20);
                acc0_20 = fma_f32x2_rn_ftz(S2[21], k2[5], acc0_20);
                acc0_20 = fma_f32x2_rn_ftz(S2[22], k2[6], acc0_20);
                acc0_20 = fma_f32x2_rn_ftz(S2[23], k2[7], acc0_20);
                float dot_20 = acc0_20.x + acc0_20.y;
                float _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, dot_20, 1);
                dot_20 = dot_20 + _shfl_xor_sync_118;
                float _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, dot_20, 2);
                dot_20 = dot_20 + _shfl_xor_sync_119;
                float _shfl_xor_sync_120 = __shfl_xor_sync(0xFFFFFFFF, dot_20, 4);
                dot_20 = dot_20 + _shfl_xor_sync_120;
#endif
                float vn_18 = (v_regs[2] - dot_20) * beta_4;
#if __CUDA_ARCH__ == 900
                float _fma_793;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_793) : "f"(k_regs[0]), "f"(vn_18), "f"(S[32]));
                S[32] = _fma_793;
                float _fma_794;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_794) : "f"(k_regs[1]), "f"(vn_18), "f"(S[33]));
                S[33] = _fma_794;
                float _fma_795;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_795) : "f"(k_regs[2]), "f"(vn_18), "f"(S[34]));
                S[34] = _fma_795;
                float _fma_796;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_796) : "f"(k_regs[3]), "f"(vn_18), "f"(S[35]));
                S[35] = _fma_796;
                float _fma_797;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_797) : "f"(k_regs[4]), "f"(vn_18), "f"(S[36]));
                S[36] = _fma_797;
                float _fma_798;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_798) : "f"(k_regs[5]), "f"(vn_18), "f"(S[37]));
                S[37] = _fma_798;
                float _fma_799;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_799) : "f"(k_regs[6]), "f"(vn_18), "f"(S[38]));
                S[38] = _fma_799;
                float _fma_800;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_800) : "f"(k_regs[7]), "f"(vn_18), "f"(S[39]));
                S[39] = _fma_800;
                float _fma_801;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_801) : "f"(k_regs[8]), "f"(vn_18), "f"(S[40]));
                S[40] = _fma_801;
                float _fma_802;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_802) : "f"(k_regs[9]), "f"(vn_18), "f"(S[41]));
                S[41] = _fma_802;
                float _fma_803;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_803) : "f"(k_regs[10]), "f"(vn_18), "f"(S[42]));
                S[42] = _fma_803;
                float _fma_804;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_804) : "f"(k_regs[11]), "f"(vn_18), "f"(S[43]));
                S[43] = _fma_804;
                float _fma_805;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_805) : "f"(k_regs[12]), "f"(vn_18), "f"(S[44]));
                S[44] = _fma_805;
                float _fma_806;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_806) : "f"(k_regs[13]), "f"(vn_18), "f"(S[45]));
                S[45] = _fma_806;
                float _fma_807;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_807) : "f"(k_regs[14]), "f"(vn_18), "f"(S[46]));
                S[46] = _fma_807;
                float _fma_808;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_808) : "f"(k_regs[15]), "f"(vn_18), "f"(S[47]));
                S[47] = _fma_808;
                float acc0_x_21 = 0.0f;
                float acc0_y_21 = 0.0f;
                float _fma_809;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_809) : "f"(S[48]), "f"(k_regs[0]), "f"(acc0_x_21));
                acc0_x_21 = _fma_809;
                float _fma_810;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_810) : "f"(S[49]), "f"(k_regs[1]), "f"(acc0_y_21));
                acc0_y_21 = _fma_810;
                float _fma_811;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_811) : "f"(S[50]), "f"(k_regs[2]), "f"(acc0_x_21));
                acc0_x_21 = _fma_811;
                float _fma_812;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_812) : "f"(S[51]), "f"(k_regs[3]), "f"(acc0_y_21));
                acc0_y_21 = _fma_812;
                float _fma_813;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_813) : "f"(S[52]), "f"(k_regs[4]), "f"(acc0_x_21));
                acc0_x_21 = _fma_813;
                float _fma_814;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_814) : "f"(S[53]), "f"(k_regs[5]), "f"(acc0_y_21));
                acc0_y_21 = _fma_814;
                float _fma_815;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_815) : "f"(S[54]), "f"(k_regs[6]), "f"(acc0_x_21));
                acc0_x_21 = _fma_815;
                float _fma_816;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_816) : "f"(S[55]), "f"(k_regs[7]), "f"(acc0_y_21));
                acc0_y_21 = _fma_816;
                float _fma_817;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_817) : "f"(S[56]), "f"(k_regs[8]), "f"(acc0_x_21));
                acc0_x_21 = _fma_817;
                float _fma_818;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_818) : "f"(S[57]), "f"(k_regs[9]), "f"(acc0_y_21));
                acc0_y_21 = _fma_818;
                float _fma_819;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_819) : "f"(S[58]), "f"(k_regs[10]), "f"(acc0_x_21));
                acc0_x_21 = _fma_819;
                float _fma_820;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_820) : "f"(S[59]), "f"(k_regs[11]), "f"(acc0_y_21));
                acc0_y_21 = _fma_820;
                float _fma_821;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_821) : "f"(S[60]), "f"(k_regs[12]), "f"(acc0_x_21));
                acc0_x_21 = _fma_821;
                float _fma_822;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_822) : "f"(S[61]), "f"(k_regs[13]), "f"(acc0_y_21));
                acc0_y_21 = _fma_822;
                float _fma_823;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_823) : "f"(S[62]), "f"(k_regs[14]), "f"(acc0_x_21));
                acc0_x_21 = _fma_823;
                float _fma_824;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_824) : "f"(S[63]), "f"(k_regs[15]), "f"(acc0_y_21));
                acc0_y_21 = _fma_824;
                float dot_21 = acc0_x_21 + acc0_y_21;
                float _shfl_xor_sync_825 = __shfl_xor_sync(0xFFFFFFFF, dot_21, 1);
                dot_21 = dot_21 + _shfl_xor_sync_825;
                float _shfl_xor_sync_826 = __shfl_xor_sync(0xFFFFFFFF, dot_21, 2);
                dot_21 = dot_21 + _shfl_xor_sync_826;
                float _shfl_xor_sync_827 = __shfl_xor_sync(0xFFFFFFFF, dot_21, 4);
                dot_21 = dot_21 + _shfl_xor_sync_827;
#else
                float2 vn2_18 = make_float2(vn_18, vn_18);
                S2[16] = fma_f32x2_rn_ftz(k2[0], vn2_18, S2[16]);
                S2[17] = fma_f32x2_rn_ftz(k2[1], vn2_18, S2[17]);
                S2[18] = fma_f32x2_rn_ftz(k2[2], vn2_18, S2[18]);
                S2[19] = fma_f32x2_rn_ftz(k2[3], vn2_18, S2[19]);
                S2[20] = fma_f32x2_rn_ftz(k2[4], vn2_18, S2[20]);
                S2[21] = fma_f32x2_rn_ftz(k2[5], vn2_18, S2[21]);
                S2[22] = fma_f32x2_rn_ftz(k2[6], vn2_18, S2[22]);
                S2[23] = fma_f32x2_rn_ftz(k2[7], vn2_18, S2[23]);
                float2 acc0_21 = make_float2(0.0f, 0.0f);
                acc0_21 = fma_f32x2_rn_ftz(S2[24], k2[0], acc0_21);
                acc0_21 = fma_f32x2_rn_ftz(S2[25], k2[1], acc0_21);
                acc0_21 = fma_f32x2_rn_ftz(S2[26], k2[2], acc0_21);
                acc0_21 = fma_f32x2_rn_ftz(S2[27], k2[3], acc0_21);
                acc0_21 = fma_f32x2_rn_ftz(S2[28], k2[4], acc0_21);
                acc0_21 = fma_f32x2_rn_ftz(S2[29], k2[5], acc0_21);
                acc0_21 = fma_f32x2_rn_ftz(S2[30], k2[6], acc0_21);
                acc0_21 = fma_f32x2_rn_ftz(S2[31], k2[7], acc0_21);
                float dot_21 = acc0_21.x + acc0_21.y;
                float _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, dot_21, 1);
                dot_21 = dot_21 + _shfl_xor_sync_121;
                float _shfl_xor_sync_122 = __shfl_xor_sync(0xFFFFFFFF, dot_21, 2);
                dot_21 = dot_21 + _shfl_xor_sync_122;
                float _shfl_xor_sync_123 = __shfl_xor_sync(0xFFFFFFFF, dot_21, 4);
                dot_21 = dot_21 + _shfl_xor_sync_123;
#endif
                float vn_19 = (v_regs[3] - dot_21) * beta_4;
#if __CUDA_ARCH__ == 900
                float _fma_828;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_828) : "f"(k_regs[0]), "f"(vn_19), "f"(S[48]));
                S[48] = _fma_828;
                float _fma_829;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_829) : "f"(k_regs[1]), "f"(vn_19), "f"(S[49]));
                S[49] = _fma_829;
                float _fma_830;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_830) : "f"(k_regs[2]), "f"(vn_19), "f"(S[50]));
                S[50] = _fma_830;
                float _fma_831;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_831) : "f"(k_regs[3]), "f"(vn_19), "f"(S[51]));
                S[51] = _fma_831;
                float _fma_832;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_832) : "f"(k_regs[4]), "f"(vn_19), "f"(S[52]));
                S[52] = _fma_832;
                float _fma_833;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_833) : "f"(k_regs[5]), "f"(vn_19), "f"(S[53]));
                S[53] = _fma_833;
                float _fma_834;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_834) : "f"(k_regs[6]), "f"(vn_19), "f"(S[54]));
                S[54] = _fma_834;
                float _fma_835;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_835) : "f"(k_regs[7]), "f"(vn_19), "f"(S[55]));
                S[55] = _fma_835;
                float _fma_836;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_836) : "f"(k_regs[8]), "f"(vn_19), "f"(S[56]));
                S[56] = _fma_836;
                float _fma_837;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_837) : "f"(k_regs[9]), "f"(vn_19), "f"(S[57]));
                S[57] = _fma_837;
                float _fma_838;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_838) : "f"(k_regs[10]), "f"(vn_19), "f"(S[58]));
                S[58] = _fma_838;
                float _fma_839;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_839) : "f"(k_regs[11]), "f"(vn_19), "f"(S[59]));
                S[59] = _fma_839;
                float _fma_840;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_840) : "f"(k_regs[12]), "f"(vn_19), "f"(S[60]));
                S[60] = _fma_840;
                float _fma_841;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_841) : "f"(k_regs[13]), "f"(vn_19), "f"(S[61]));
                S[61] = _fma_841;
                float _fma_842;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_842) : "f"(k_regs[14]), "f"(vn_19), "f"(S[62]));
                S[62] = _fma_842;
                float _fma_843;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_843) : "f"(k_regs[15]), "f"(vn_19), "f"(S[63]));
                S[63] = _fma_843;
#else
                float2 vn2_19 = make_float2(vn_19, vn_19);
                S2[24] = fma_f32x2_rn_ftz(k2[0], vn2_19, S2[24]);
                S2[25] = fma_f32x2_rn_ftz(k2[1], vn2_19, S2[25]);
                S2[26] = fma_f32x2_rn_ftz(k2[2], vn2_19, S2[26]);
                S2[27] = fma_f32x2_rn_ftz(k2[3], vn2_19, S2[27]);
                S2[28] = fma_f32x2_rn_ftz(k2[4], vn2_19, S2[28]);
                S2[29] = fma_f32x2_rn_ftz(k2[5], vn2_19, S2[29]);
                S2[30] = fma_f32x2_rn_ftz(k2[6], vn2_19, S2[30]);
                S2[31] = fma_f32x2_rn_ftz(k2[7], vn2_19, S2[31]);
#endif
                int m_5 = t_1 * 3 + 2;
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
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                    : "r"(sV_addr + (unsigned int)(m_5 * 256) + group_byte));
                float beta_5 = sBeta[m_5];
#if __CUDA_ARCH__ == 900
                float acc0_x_22 = 0.0f;
                float acc0_y_22 = 0.0f;
                float _fma_844;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_844) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_22));
                acc0_x_22 = _fma_844;
                float _fma_845;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_845) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_22));
                acc0_y_22 = _fma_845;
                float _fma_846;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_846) : "f"(S[2]), "f"(k_regs[2]), "f"(acc0_x_22));
                acc0_x_22 = _fma_846;
                float _fma_847;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_847) : "f"(S[3]), "f"(k_regs[3]), "f"(acc0_y_22));
                acc0_y_22 = _fma_847;
                float _fma_848;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_848) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_22));
                acc0_x_22 = _fma_848;
                float _fma_849;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_849) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_22));
                acc0_y_22 = _fma_849;
                float _fma_850;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_850) : "f"(S[6]), "f"(k_regs[6]), "f"(acc0_x_22));
                acc0_x_22 = _fma_850;
                float _fma_851;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_851) : "f"(S[7]), "f"(k_regs[7]), "f"(acc0_y_22));
                acc0_y_22 = _fma_851;
                float _fma_852;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_852) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_22));
                acc0_x_22 = _fma_852;
                float _fma_853;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_853) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_22));
                acc0_y_22 = _fma_853;
                float _fma_854;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_854) : "f"(S[10]), "f"(k_regs[10]), "f"(acc0_x_22));
                acc0_x_22 = _fma_854;
                float _fma_855;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_855) : "f"(S[11]), "f"(k_regs[11]), "f"(acc0_y_22));
                acc0_y_22 = _fma_855;
                float _fma_856;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_856) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_22));
                acc0_x_22 = _fma_856;
                float _fma_857;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_857) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_22));
                acc0_y_22 = _fma_857;
                float _fma_858;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_858) : "f"(S[14]), "f"(k_regs[14]), "f"(acc0_x_22));
                acc0_x_22 = _fma_858;
                float _fma_859;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_859) : "f"(S[15]), "f"(k_regs[15]), "f"(acc0_y_22));
                acc0_y_22 = _fma_859;
                float dot_22 = acc0_x_22 + acc0_y_22;
                float _shfl_xor_sync_860 = __shfl_xor_sync(0xFFFFFFFF, dot_22, 1);
                dot_22 = dot_22 + _shfl_xor_sync_860;
                float _shfl_xor_sync_861 = __shfl_xor_sync(0xFFFFFFFF, dot_22, 2);
                dot_22 = dot_22 + _shfl_xor_sync_861;
                float _shfl_xor_sync_862 = __shfl_xor_sync(0xFFFFFFFF, dot_22, 4);
                dot_22 = dot_22 + _shfl_xor_sync_862;
#else
                float2 acc0_22 = make_float2(0.0f, 0.0f);
                acc0_22 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_22);
                acc0_22 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_22);
                acc0_22 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_22);
                acc0_22 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_22);
                acc0_22 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_22);
                acc0_22 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_22);
                acc0_22 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_22);
                acc0_22 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_22);
                float dot_22 = acc0_22.x + acc0_22.y;
                float _shfl_xor_sync_124 = __shfl_xor_sync(0xFFFFFFFF, dot_22, 1);
                dot_22 = dot_22 + _shfl_xor_sync_124;
                float _shfl_xor_sync_125 = __shfl_xor_sync(0xFFFFFFFF, dot_22, 2);
                dot_22 = dot_22 + _shfl_xor_sync_125;
                float _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, dot_22, 4);
                dot_22 = dot_22 + _shfl_xor_sync_126;
#endif
                float vn_20 = (v_regs[0] - dot_22) * beta_5;
#if __CUDA_ARCH__ == 900
                float _fma_863;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_863) : "f"(k_regs[0]), "f"(vn_20), "f"(S[0]));
                S[0] = _fma_863;
                float _fma_864;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_864) : "f"(k_regs[1]), "f"(vn_20), "f"(S[1]));
                S[1] = _fma_864;
                float _fma_865;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_865) : "f"(k_regs[2]), "f"(vn_20), "f"(S[2]));
                S[2] = _fma_865;
                float _fma_866;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_866) : "f"(k_regs[3]), "f"(vn_20), "f"(S[3]));
                S[3] = _fma_866;
                float _fma_867;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_867) : "f"(k_regs[4]), "f"(vn_20), "f"(S[4]));
                S[4] = _fma_867;
                float _fma_868;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_868) : "f"(k_regs[5]), "f"(vn_20), "f"(S[5]));
                S[5] = _fma_868;
                float _fma_869;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_869) : "f"(k_regs[6]), "f"(vn_20), "f"(S[6]));
                S[6] = _fma_869;
                float _fma_870;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_870) : "f"(k_regs[7]), "f"(vn_20), "f"(S[7]));
                S[7] = _fma_870;
                float _fma_871;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_871) : "f"(k_regs[8]), "f"(vn_20), "f"(S[8]));
                S[8] = _fma_871;
                float _fma_872;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_872) : "f"(k_regs[9]), "f"(vn_20), "f"(S[9]));
                S[9] = _fma_872;
                float _fma_873;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_873) : "f"(k_regs[10]), "f"(vn_20), "f"(S[10]));
                S[10] = _fma_873;
                float _fma_874;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_874) : "f"(k_regs[11]), "f"(vn_20), "f"(S[11]));
                S[11] = _fma_874;
                float _fma_875;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_875) : "f"(k_regs[12]), "f"(vn_20), "f"(S[12]));
                S[12] = _fma_875;
                float _fma_876;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_876) : "f"(k_regs[13]), "f"(vn_20), "f"(S[13]));
                S[13] = _fma_876;
                float _fma_877;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_877) : "f"(k_regs[14]), "f"(vn_20), "f"(S[14]));
                S[14] = _fma_877;
                float _fma_878;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_878) : "f"(k_regs[15]), "f"(vn_20), "f"(S[15]));
                S[15] = _fma_878;
                float acc0_x_23 = 0.0f;
                float acc0_y_23 = 0.0f;
                float _fma_879;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_879) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_23));
                acc0_x_23 = _fma_879;
                float _fma_880;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_880) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_23));
                acc0_y_23 = _fma_880;
                float _fma_881;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_881) : "f"(S[18]), "f"(k_regs[2]), "f"(acc0_x_23));
                acc0_x_23 = _fma_881;
                float _fma_882;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_882) : "f"(S[19]), "f"(k_regs[3]), "f"(acc0_y_23));
                acc0_y_23 = _fma_882;
                float _fma_883;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_883) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_23));
                acc0_x_23 = _fma_883;
                float _fma_884;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_884) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_23));
                acc0_y_23 = _fma_884;
                float _fma_885;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_885) : "f"(S[22]), "f"(k_regs[6]), "f"(acc0_x_23));
                acc0_x_23 = _fma_885;
                float _fma_886;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_886) : "f"(S[23]), "f"(k_regs[7]), "f"(acc0_y_23));
                acc0_y_23 = _fma_886;
                float _fma_887;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_887) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_23));
                acc0_x_23 = _fma_887;
                float _fma_888;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_888) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_23));
                acc0_y_23 = _fma_888;
                float _fma_889;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_889) : "f"(S[26]), "f"(k_regs[10]), "f"(acc0_x_23));
                acc0_x_23 = _fma_889;
                float _fma_890;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_890) : "f"(S[27]), "f"(k_regs[11]), "f"(acc0_y_23));
                acc0_y_23 = _fma_890;
                float _fma_891;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_891) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_23));
                acc0_x_23 = _fma_891;
                float _fma_892;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_892) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_23));
                acc0_y_23 = _fma_892;
                float _fma_893;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_893) : "f"(S[30]), "f"(k_regs[14]), "f"(acc0_x_23));
                acc0_x_23 = _fma_893;
                float _fma_894;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_894) : "f"(S[31]), "f"(k_regs[15]), "f"(acc0_y_23));
                acc0_y_23 = _fma_894;
                float dot_23 = acc0_x_23 + acc0_y_23;
                float _shfl_xor_sync_895 = __shfl_xor_sync(0xFFFFFFFF, dot_23, 1);
                dot_23 = dot_23 + _shfl_xor_sync_895;
                float _shfl_xor_sync_896 = __shfl_xor_sync(0xFFFFFFFF, dot_23, 2);
                dot_23 = dot_23 + _shfl_xor_sync_896;
                float _shfl_xor_sync_897 = __shfl_xor_sync(0xFFFFFFFF, dot_23, 4);
                dot_23 = dot_23 + _shfl_xor_sync_897;
#else
                float2 vn2_20 = make_float2(vn_20, vn_20);
                S2[0] = fma_f32x2_rn_ftz(k2[0], vn2_20, S2[0]);
                S2[1] = fma_f32x2_rn_ftz(k2[1], vn2_20, S2[1]);
                S2[2] = fma_f32x2_rn_ftz(k2[2], vn2_20, S2[2]);
                S2[3] = fma_f32x2_rn_ftz(k2[3], vn2_20, S2[3]);
                S2[4] = fma_f32x2_rn_ftz(k2[4], vn2_20, S2[4]);
                S2[5] = fma_f32x2_rn_ftz(k2[5], vn2_20, S2[5]);
                S2[6] = fma_f32x2_rn_ftz(k2[6], vn2_20, S2[6]);
                S2[7] = fma_f32x2_rn_ftz(k2[7], vn2_20, S2[7]);
                float2 acc0_23 = make_float2(0.0f, 0.0f);
                acc0_23 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_23);
                acc0_23 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_23);
                acc0_23 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_23);
                acc0_23 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_23);
                acc0_23 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_23);
                acc0_23 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_23);
                acc0_23 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_23);
                acc0_23 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_23);
                float dot_23 = acc0_23.x + acc0_23.y;
                float _shfl_xor_sync_127 = __shfl_xor_sync(0xFFFFFFFF, dot_23, 1);
                dot_23 = dot_23 + _shfl_xor_sync_127;
                float _shfl_xor_sync_128 = __shfl_xor_sync(0xFFFFFFFF, dot_23, 2);
                dot_23 = dot_23 + _shfl_xor_sync_128;
                float _shfl_xor_sync_129 = __shfl_xor_sync(0xFFFFFFFF, dot_23, 4);
                dot_23 = dot_23 + _shfl_xor_sync_129;
#endif
                float vn_21 = (v_regs[1] - dot_23) * beta_5;
#if __CUDA_ARCH__ == 900
                float _fma_898;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_898) : "f"(k_regs[0]), "f"(vn_21), "f"(S[16]));
                S[16] = _fma_898;
                float _fma_899;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_899) : "f"(k_regs[1]), "f"(vn_21), "f"(S[17]));
                S[17] = _fma_899;
                float _fma_900;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_900) : "f"(k_regs[2]), "f"(vn_21), "f"(S[18]));
                S[18] = _fma_900;
                float _fma_901;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_901) : "f"(k_regs[3]), "f"(vn_21), "f"(S[19]));
                S[19] = _fma_901;
                float _fma_902;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_902) : "f"(k_regs[4]), "f"(vn_21), "f"(S[20]));
                S[20] = _fma_902;
                float _fma_903;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_903) : "f"(k_regs[5]), "f"(vn_21), "f"(S[21]));
                S[21] = _fma_903;
                float _fma_904;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_904) : "f"(k_regs[6]), "f"(vn_21), "f"(S[22]));
                S[22] = _fma_904;
                float _fma_905;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_905) : "f"(k_regs[7]), "f"(vn_21), "f"(S[23]));
                S[23] = _fma_905;
                float _fma_906;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_906) : "f"(k_regs[8]), "f"(vn_21), "f"(S[24]));
                S[24] = _fma_906;
                float _fma_907;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_907) : "f"(k_regs[9]), "f"(vn_21), "f"(S[25]));
                S[25] = _fma_907;
                float _fma_908;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_908) : "f"(k_regs[10]), "f"(vn_21), "f"(S[26]));
                S[26] = _fma_908;
                float _fma_909;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_909) : "f"(k_regs[11]), "f"(vn_21), "f"(S[27]));
                S[27] = _fma_909;
                float _fma_910;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_910) : "f"(k_regs[12]), "f"(vn_21), "f"(S[28]));
                S[28] = _fma_910;
                float _fma_911;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_911) : "f"(k_regs[13]), "f"(vn_21), "f"(S[29]));
                S[29] = _fma_911;
                float _fma_912;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_912) : "f"(k_regs[14]), "f"(vn_21), "f"(S[30]));
                S[30] = _fma_912;
                float _fma_913;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_913) : "f"(k_regs[15]), "f"(vn_21), "f"(S[31]));
                S[31] = _fma_913;
                float acc0_x_24 = 0.0f;
                float acc0_y_24 = 0.0f;
                float _fma_914;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_914) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_24));
                acc0_x_24 = _fma_914;
                float _fma_915;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_915) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_24));
                acc0_y_24 = _fma_915;
                float _fma_916;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_916) : "f"(S[34]), "f"(k_regs[2]), "f"(acc0_x_24));
                acc0_x_24 = _fma_916;
                float _fma_917;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_917) : "f"(S[35]), "f"(k_regs[3]), "f"(acc0_y_24));
                acc0_y_24 = _fma_917;
                float _fma_918;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_918) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_24));
                acc0_x_24 = _fma_918;
                float _fma_919;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_919) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_24));
                acc0_y_24 = _fma_919;
                float _fma_920;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_920) : "f"(S[38]), "f"(k_regs[6]), "f"(acc0_x_24));
                acc0_x_24 = _fma_920;
                float _fma_921;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_921) : "f"(S[39]), "f"(k_regs[7]), "f"(acc0_y_24));
                acc0_y_24 = _fma_921;
                float _fma_922;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_922) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_24));
                acc0_x_24 = _fma_922;
                float _fma_923;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_923) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_24));
                acc0_y_24 = _fma_923;
                float _fma_924;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_924) : "f"(S[42]), "f"(k_regs[10]), "f"(acc0_x_24));
                acc0_x_24 = _fma_924;
                float _fma_925;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_925) : "f"(S[43]), "f"(k_regs[11]), "f"(acc0_y_24));
                acc0_y_24 = _fma_925;
                float _fma_926;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_926) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_24));
                acc0_x_24 = _fma_926;
                float _fma_927;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_927) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_24));
                acc0_y_24 = _fma_927;
                float _fma_928;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_928) : "f"(S[46]), "f"(k_regs[14]), "f"(acc0_x_24));
                acc0_x_24 = _fma_928;
                float _fma_929;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_929) : "f"(S[47]), "f"(k_regs[15]), "f"(acc0_y_24));
                acc0_y_24 = _fma_929;
                float dot_24 = acc0_x_24 + acc0_y_24;
                float _shfl_xor_sync_930 = __shfl_xor_sync(0xFFFFFFFF, dot_24, 1);
                dot_24 = dot_24 + _shfl_xor_sync_930;
                float _shfl_xor_sync_931 = __shfl_xor_sync(0xFFFFFFFF, dot_24, 2);
                dot_24 = dot_24 + _shfl_xor_sync_931;
                float _shfl_xor_sync_932 = __shfl_xor_sync(0xFFFFFFFF, dot_24, 4);
                dot_24 = dot_24 + _shfl_xor_sync_932;
#else
                float2 vn2_21 = make_float2(vn_21, vn_21);
                S2[8] = fma_f32x2_rn_ftz(k2[0], vn2_21, S2[8]);
                S2[9] = fma_f32x2_rn_ftz(k2[1], vn2_21, S2[9]);
                S2[10] = fma_f32x2_rn_ftz(k2[2], vn2_21, S2[10]);
                S2[11] = fma_f32x2_rn_ftz(k2[3], vn2_21, S2[11]);
                S2[12] = fma_f32x2_rn_ftz(k2[4], vn2_21, S2[12]);
                S2[13] = fma_f32x2_rn_ftz(k2[5], vn2_21, S2[13]);
                S2[14] = fma_f32x2_rn_ftz(k2[6], vn2_21, S2[14]);
                S2[15] = fma_f32x2_rn_ftz(k2[7], vn2_21, S2[15]);
                float2 acc0_24 = make_float2(0.0f, 0.0f);
                acc0_24 = fma_f32x2_rn_ftz(S2[16], k2[0], acc0_24);
                acc0_24 = fma_f32x2_rn_ftz(S2[17], k2[1], acc0_24);
                acc0_24 = fma_f32x2_rn_ftz(S2[18], k2[2], acc0_24);
                acc0_24 = fma_f32x2_rn_ftz(S2[19], k2[3], acc0_24);
                acc0_24 = fma_f32x2_rn_ftz(S2[20], k2[4], acc0_24);
                acc0_24 = fma_f32x2_rn_ftz(S2[21], k2[5], acc0_24);
                acc0_24 = fma_f32x2_rn_ftz(S2[22], k2[6], acc0_24);
                acc0_24 = fma_f32x2_rn_ftz(S2[23], k2[7], acc0_24);
                float dot_24 = acc0_24.x + acc0_24.y;
                float _shfl_xor_sync_130 = __shfl_xor_sync(0xFFFFFFFF, dot_24, 1);
                dot_24 = dot_24 + _shfl_xor_sync_130;
                float _shfl_xor_sync_131 = __shfl_xor_sync(0xFFFFFFFF, dot_24, 2);
                dot_24 = dot_24 + _shfl_xor_sync_131;
                float _shfl_xor_sync_132 = __shfl_xor_sync(0xFFFFFFFF, dot_24, 4);
                dot_24 = dot_24 + _shfl_xor_sync_132;
#endif
                float vn_22 = (v_regs[2] - dot_24) * beta_5;
#if __CUDA_ARCH__ == 900
                float _fma_933;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_933) : "f"(k_regs[0]), "f"(vn_22), "f"(S[32]));
                S[32] = _fma_933;
                float _fma_934;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_934) : "f"(k_regs[1]), "f"(vn_22), "f"(S[33]));
                S[33] = _fma_934;
                float _fma_935;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_935) : "f"(k_regs[2]), "f"(vn_22), "f"(S[34]));
                S[34] = _fma_935;
                float _fma_936;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_936) : "f"(k_regs[3]), "f"(vn_22), "f"(S[35]));
                S[35] = _fma_936;
                float _fma_937;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_937) : "f"(k_regs[4]), "f"(vn_22), "f"(S[36]));
                S[36] = _fma_937;
                float _fma_938;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_938) : "f"(k_regs[5]), "f"(vn_22), "f"(S[37]));
                S[37] = _fma_938;
                float _fma_939;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_939) : "f"(k_regs[6]), "f"(vn_22), "f"(S[38]));
                S[38] = _fma_939;
                float _fma_940;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_940) : "f"(k_regs[7]), "f"(vn_22), "f"(S[39]));
                S[39] = _fma_940;
                float _fma_941;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_941) : "f"(k_regs[8]), "f"(vn_22), "f"(S[40]));
                S[40] = _fma_941;
                float _fma_942;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_942) : "f"(k_regs[9]), "f"(vn_22), "f"(S[41]));
                S[41] = _fma_942;
                float _fma_943;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_943) : "f"(k_regs[10]), "f"(vn_22), "f"(S[42]));
                S[42] = _fma_943;
                float _fma_944;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_944) : "f"(k_regs[11]), "f"(vn_22), "f"(S[43]));
                S[43] = _fma_944;
                float _fma_945;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_945) : "f"(k_regs[12]), "f"(vn_22), "f"(S[44]));
                S[44] = _fma_945;
                float _fma_946;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_946) : "f"(k_regs[13]), "f"(vn_22), "f"(S[45]));
                S[45] = _fma_946;
                float _fma_947;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_947) : "f"(k_regs[14]), "f"(vn_22), "f"(S[46]));
                S[46] = _fma_947;
                float _fma_948;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_948) : "f"(k_regs[15]), "f"(vn_22), "f"(S[47]));
                S[47] = _fma_948;
                float acc0_x_25 = 0.0f;
                float acc0_y_25 = 0.0f;
                float _fma_949;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_949) : "f"(S[48]), "f"(k_regs[0]), "f"(acc0_x_25));
                acc0_x_25 = _fma_949;
                float _fma_950;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_950) : "f"(S[49]), "f"(k_regs[1]), "f"(acc0_y_25));
                acc0_y_25 = _fma_950;
                float _fma_951;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_951) : "f"(S[50]), "f"(k_regs[2]), "f"(acc0_x_25));
                acc0_x_25 = _fma_951;
                float _fma_952;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_952) : "f"(S[51]), "f"(k_regs[3]), "f"(acc0_y_25));
                acc0_y_25 = _fma_952;
                float _fma_953;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_953) : "f"(S[52]), "f"(k_regs[4]), "f"(acc0_x_25));
                acc0_x_25 = _fma_953;
                float _fma_954;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_954) : "f"(S[53]), "f"(k_regs[5]), "f"(acc0_y_25));
                acc0_y_25 = _fma_954;
                float _fma_955;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_955) : "f"(S[54]), "f"(k_regs[6]), "f"(acc0_x_25));
                acc0_x_25 = _fma_955;
                float _fma_956;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_956) : "f"(S[55]), "f"(k_regs[7]), "f"(acc0_y_25));
                acc0_y_25 = _fma_956;
                float _fma_957;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_957) : "f"(S[56]), "f"(k_regs[8]), "f"(acc0_x_25));
                acc0_x_25 = _fma_957;
                float _fma_958;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_958) : "f"(S[57]), "f"(k_regs[9]), "f"(acc0_y_25));
                acc0_y_25 = _fma_958;
                float _fma_959;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_959) : "f"(S[58]), "f"(k_regs[10]), "f"(acc0_x_25));
                acc0_x_25 = _fma_959;
                float _fma_960;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_960) : "f"(S[59]), "f"(k_regs[11]), "f"(acc0_y_25));
                acc0_y_25 = _fma_960;
                float _fma_961;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_961) : "f"(S[60]), "f"(k_regs[12]), "f"(acc0_x_25));
                acc0_x_25 = _fma_961;
                float _fma_962;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_962) : "f"(S[61]), "f"(k_regs[13]), "f"(acc0_y_25));
                acc0_y_25 = _fma_962;
                float _fma_963;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_963) : "f"(S[62]), "f"(k_regs[14]), "f"(acc0_x_25));
                acc0_x_25 = _fma_963;
                float _fma_964;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_964) : "f"(S[63]), "f"(k_regs[15]), "f"(acc0_y_25));
                acc0_y_25 = _fma_964;
                float dot_25 = acc0_x_25 + acc0_y_25;
                float _shfl_xor_sync_965 = __shfl_xor_sync(0xFFFFFFFF, dot_25, 1);
                dot_25 = dot_25 + _shfl_xor_sync_965;
                float _shfl_xor_sync_966 = __shfl_xor_sync(0xFFFFFFFF, dot_25, 2);
                dot_25 = dot_25 + _shfl_xor_sync_966;
                float _shfl_xor_sync_967 = __shfl_xor_sync(0xFFFFFFFF, dot_25, 4);
                dot_25 = dot_25 + _shfl_xor_sync_967;
#else
                float2 vn2_22 = make_float2(vn_22, vn_22);
                S2[16] = fma_f32x2_rn_ftz(k2[0], vn2_22, S2[16]);
                S2[17] = fma_f32x2_rn_ftz(k2[1], vn2_22, S2[17]);
                S2[18] = fma_f32x2_rn_ftz(k2[2], vn2_22, S2[18]);
                S2[19] = fma_f32x2_rn_ftz(k2[3], vn2_22, S2[19]);
                S2[20] = fma_f32x2_rn_ftz(k2[4], vn2_22, S2[20]);
                S2[21] = fma_f32x2_rn_ftz(k2[5], vn2_22, S2[21]);
                S2[22] = fma_f32x2_rn_ftz(k2[6], vn2_22, S2[22]);
                S2[23] = fma_f32x2_rn_ftz(k2[7], vn2_22, S2[23]);
                float2 acc0_25 = make_float2(0.0f, 0.0f);
                acc0_25 = fma_f32x2_rn_ftz(S2[24], k2[0], acc0_25);
                acc0_25 = fma_f32x2_rn_ftz(S2[25], k2[1], acc0_25);
                acc0_25 = fma_f32x2_rn_ftz(S2[26], k2[2], acc0_25);
                acc0_25 = fma_f32x2_rn_ftz(S2[27], k2[3], acc0_25);
                acc0_25 = fma_f32x2_rn_ftz(S2[28], k2[4], acc0_25);
                acc0_25 = fma_f32x2_rn_ftz(S2[29], k2[5], acc0_25);
                acc0_25 = fma_f32x2_rn_ftz(S2[30], k2[6], acc0_25);
                acc0_25 = fma_f32x2_rn_ftz(S2[31], k2[7], acc0_25);
                float dot_25 = acc0_25.x + acc0_25.y;
                float _shfl_xor_sync_133 = __shfl_xor_sync(0xFFFFFFFF, dot_25, 1);
                dot_25 = dot_25 + _shfl_xor_sync_133;
                float _shfl_xor_sync_134 = __shfl_xor_sync(0xFFFFFFFF, dot_25, 2);
                dot_25 = dot_25 + _shfl_xor_sync_134;
                float _shfl_xor_sync_135 = __shfl_xor_sync(0xFFFFFFFF, dot_25, 4);
                dot_25 = dot_25 + _shfl_xor_sync_135;
#endif
                float vn_23 = (v_regs[3] - dot_25) * beta_5;
#if __CUDA_ARCH__ == 900
                float _fma_968;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_968) : "f"(k_regs[0]), "f"(vn_23), "f"(S[48]));
                S[48] = _fma_968;
                float _fma_969;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_969) : "f"(k_regs[1]), "f"(vn_23), "f"(S[49]));
                S[49] = _fma_969;
                float _fma_970;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_970) : "f"(k_regs[2]), "f"(vn_23), "f"(S[50]));
                S[50] = _fma_970;
                float _fma_971;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_971) : "f"(k_regs[3]), "f"(vn_23), "f"(S[51]));
                S[51] = _fma_971;
                float _fma_972;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_972) : "f"(k_regs[4]), "f"(vn_23), "f"(S[52]));
                S[52] = _fma_972;
                float _fma_973;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_973) : "f"(k_regs[5]), "f"(vn_23), "f"(S[53]));
                S[53] = _fma_973;
                float _fma_974;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_974) : "f"(k_regs[6]), "f"(vn_23), "f"(S[54]));
                S[54] = _fma_974;
                float _fma_975;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_975) : "f"(k_regs[7]), "f"(vn_23), "f"(S[55]));
                S[55] = _fma_975;
                float _fma_976;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_976) : "f"(k_regs[8]), "f"(vn_23), "f"(S[56]));
                S[56] = _fma_976;
                float _fma_977;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_977) : "f"(k_regs[9]), "f"(vn_23), "f"(S[57]));
                S[57] = _fma_977;
                float _fma_978;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_978) : "f"(k_regs[10]), "f"(vn_23), "f"(S[58]));
                S[58] = _fma_978;
                float _fma_979;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_979) : "f"(k_regs[11]), "f"(vn_23), "f"(S[59]));
                S[59] = _fma_979;
                float _fma_980;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_980) : "f"(k_regs[12]), "f"(vn_23), "f"(S[60]));
                S[60] = _fma_980;
                float _fma_981;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_981) : "f"(k_regs[13]), "f"(vn_23), "f"(S[61]));
                S[61] = _fma_981;
                float _fma_982;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_982) : "f"(k_regs[14]), "f"(vn_23), "f"(S[62]));
                S[62] = _fma_982;
                float _fma_983;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_983) : "f"(k_regs[15]), "f"(vn_23), "f"(S[63]));
                S[63] = _fma_983;
#else
                float2 vn2_23 = make_float2(vn_23, vn_23);
                S2[24] = fma_f32x2_rn_ftz(k2[0], vn2_23, S2[24]);
                S2[25] = fma_f32x2_rn_ftz(k2[1], vn2_23, S2[25]);
                S2[26] = fma_f32x2_rn_ftz(k2[2], vn2_23, S2[26]);
                S2[27] = fma_f32x2_rn_ftz(k2[3], vn2_23, S2[27]);
                S2[28] = fma_f32x2_rn_ftz(k2[4], vn2_23, S2[28]);
                S2[29] = fma_f32x2_rn_ftz(k2[5], vn2_23, S2[29]);
                S2[30] = fma_f32x2_rn_ftz(k2[6], vn2_23, S2[30]);
                S2[31] = fma_f32x2_rn_ftz(k2[7], vn2_23, S2[31]);
#endif
                unsigned int qbase_1 = sQ_addr + (unsigned int)(t_1 * 512) + lane_byte;
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
                float acc0_x_26 = 0.0f;
                float acc0_y_26 = 0.0f;
                float _fma_984;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_984) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_26));
                acc0_x_26 = _fma_984;
                float _fma_985;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_985) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_26));
                acc0_y_26 = _fma_985;
                float _fma_986;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_986) : "f"(S[2]), "f"(k_regs[2]), "f"(acc0_x_26));
                acc0_x_26 = _fma_986;
                float _fma_987;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_987) : "f"(S[3]), "f"(k_regs[3]), "f"(acc0_y_26));
                acc0_y_26 = _fma_987;
                float _fma_988;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_988) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_26));
                acc0_x_26 = _fma_988;
                float _fma_989;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_989) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_26));
                acc0_y_26 = _fma_989;
                float _fma_990;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_990) : "f"(S[6]), "f"(k_regs[6]), "f"(acc0_x_26));
                acc0_x_26 = _fma_990;
                float _fma_991;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_991) : "f"(S[7]), "f"(k_regs[7]), "f"(acc0_y_26));
                acc0_y_26 = _fma_991;
                float _fma_992;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_992) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_26));
                acc0_x_26 = _fma_992;
                float _fma_993;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_993) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_26));
                acc0_y_26 = _fma_993;
                float _fma_994;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_994) : "f"(S[10]), "f"(k_regs[10]), "f"(acc0_x_26));
                acc0_x_26 = _fma_994;
                float _fma_995;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_995) : "f"(S[11]), "f"(k_regs[11]), "f"(acc0_y_26));
                acc0_y_26 = _fma_995;
                float _fma_996;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_996) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_26));
                acc0_x_26 = _fma_996;
                float _fma_997;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_997) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_26));
                acc0_y_26 = _fma_997;
                float _fma_998;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_998) : "f"(S[14]), "f"(k_regs[14]), "f"(acc0_x_26));
                acc0_x_26 = _fma_998;
                float _fma_999;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_999) : "f"(S[15]), "f"(k_regs[15]), "f"(acc0_y_26));
                acc0_y_26 = _fma_999;
                float dot_26 = acc0_x_26 + acc0_y_26;
                float _shfl_xor_sync_1000 = __shfl_xor_sync(0xFFFFFFFF, dot_26, 1);
                dot_26 = dot_26 + _shfl_xor_sync_1000;
                float _shfl_xor_sync_1001 = __shfl_xor_sync(0xFFFFFFFF, dot_26, 2);
                dot_26 = dot_26 + _shfl_xor_sync_1001;
                float _shfl_xor_sync_1002 = __shfl_xor_sync(0xFFFFFFFF, dot_26, 4);
                dot_26 = dot_26 + _shfl_xor_sync_1002;
#else
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                float2 acc0_26 = make_float2(0.0f, 0.0f);
                acc0_26 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_26);
                acc0_26 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_26);
                acc0_26 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_26);
                acc0_26 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_26);
                acc0_26 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_26);
                acc0_26 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_26);
                acc0_26 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_26);
                acc0_26 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_26);
                float dot_26 = acc0_26.x + acc0_26.y;
                float _shfl_xor_sync_136 = __shfl_xor_sync(0xFFFFFFFF, dot_26, 1);
                dot_26 = dot_26 + _shfl_xor_sync_136;
                float _shfl_xor_sync_137 = __shfl_xor_sync(0xFFFFFFFF, dot_26, 2);
                dot_26 = dot_26 + _shfl_xor_sync_137;
                float _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, dot_26, 4);
                dot_26 = dot_26 + _shfl_xor_sync_138;
#endif
                o4[0] = dot_26;
#if __CUDA_ARCH__ == 900
                float acc0_x_27 = 0.0f;
                float acc0_y_27 = 0.0f;
                float _fma_1003;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1003) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_27));
                acc0_x_27 = _fma_1003;
                float _fma_1004;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1004) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_27));
                acc0_y_27 = _fma_1004;
                float _fma_1005;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1005) : "f"(S[18]), "f"(k_regs[2]), "f"(acc0_x_27));
                acc0_x_27 = _fma_1005;
                float _fma_1006;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1006) : "f"(S[19]), "f"(k_regs[3]), "f"(acc0_y_27));
                acc0_y_27 = _fma_1006;
                float _fma_1007;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1007) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_27));
                acc0_x_27 = _fma_1007;
                float _fma_1008;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1008) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_27));
                acc0_y_27 = _fma_1008;
                float _fma_1009;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1009) : "f"(S[22]), "f"(k_regs[6]), "f"(acc0_x_27));
                acc0_x_27 = _fma_1009;
                float _fma_1010;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1010) : "f"(S[23]), "f"(k_regs[7]), "f"(acc0_y_27));
                acc0_y_27 = _fma_1010;
                float _fma_1011;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1011) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_27));
                acc0_x_27 = _fma_1011;
                float _fma_1012;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1012) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_27));
                acc0_y_27 = _fma_1012;
                float _fma_1013;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1013) : "f"(S[26]), "f"(k_regs[10]), "f"(acc0_x_27));
                acc0_x_27 = _fma_1013;
                float _fma_1014;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1014) : "f"(S[27]), "f"(k_regs[11]), "f"(acc0_y_27));
                acc0_y_27 = _fma_1014;
                float _fma_1015;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1015) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_27));
                acc0_x_27 = _fma_1015;
                float _fma_1016;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1016) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_27));
                acc0_y_27 = _fma_1016;
                float _fma_1017;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1017) : "f"(S[30]), "f"(k_regs[14]), "f"(acc0_x_27));
                acc0_x_27 = _fma_1017;
                float _fma_1018;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1018) : "f"(S[31]), "f"(k_regs[15]), "f"(acc0_y_27));
                acc0_y_27 = _fma_1018;
                float dot_27 = acc0_x_27 + acc0_y_27;
                float _shfl_xor_sync_1019 = __shfl_xor_sync(0xFFFFFFFF, dot_27, 1);
                dot_27 = dot_27 + _shfl_xor_sync_1019;
                float _shfl_xor_sync_1020 = __shfl_xor_sync(0xFFFFFFFF, dot_27, 2);
                dot_27 = dot_27 + _shfl_xor_sync_1020;
                float _shfl_xor_sync_1021 = __shfl_xor_sync(0xFFFFFFFF, dot_27, 4);
                dot_27 = dot_27 + _shfl_xor_sync_1021;
#else
                float2 acc0_27 = make_float2(0.0f, 0.0f);
                acc0_27 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_27);
                acc0_27 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_27);
                acc0_27 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_27);
                acc0_27 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_27);
                acc0_27 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_27);
                acc0_27 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_27);
                acc0_27 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_27);
                acc0_27 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_27);
                float dot_27 = acc0_27.x + acc0_27.y;
                float _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, dot_27, 1);
                dot_27 = dot_27 + _shfl_xor_sync_139;
                float _shfl_xor_sync_140 = __shfl_xor_sync(0xFFFFFFFF, dot_27, 2);
                dot_27 = dot_27 + _shfl_xor_sync_140;
                float _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, dot_27, 4);
                dot_27 = dot_27 + _shfl_xor_sync_141;
#endif
                o4[1] = dot_27;
#if __CUDA_ARCH__ == 900
                float acc0_x_28 = 0.0f;
                float acc0_y_28 = 0.0f;
                float _fma_1022;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1022) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_28));
                acc0_x_28 = _fma_1022;
                float _fma_1023;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1023) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_28));
                acc0_y_28 = _fma_1023;
                float _fma_1024;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1024) : "f"(S[34]), "f"(k_regs[2]), "f"(acc0_x_28));
                acc0_x_28 = _fma_1024;
                float _fma_1025;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1025) : "f"(S[35]), "f"(k_regs[3]), "f"(acc0_y_28));
                acc0_y_28 = _fma_1025;
                float _fma_1026;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1026) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_28));
                acc0_x_28 = _fma_1026;
                float _fma_1027;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1027) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_28));
                acc0_y_28 = _fma_1027;
                float _fma_1028;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1028) : "f"(S[38]), "f"(k_regs[6]), "f"(acc0_x_28));
                acc0_x_28 = _fma_1028;
                float _fma_1029;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1029) : "f"(S[39]), "f"(k_regs[7]), "f"(acc0_y_28));
                acc0_y_28 = _fma_1029;
                float _fma_1030;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1030) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_28));
                acc0_x_28 = _fma_1030;
                float _fma_1031;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1031) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_28));
                acc0_y_28 = _fma_1031;
                float _fma_1032;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1032) : "f"(S[42]), "f"(k_regs[10]), "f"(acc0_x_28));
                acc0_x_28 = _fma_1032;
                float _fma_1033;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1033) : "f"(S[43]), "f"(k_regs[11]), "f"(acc0_y_28));
                acc0_y_28 = _fma_1033;
                float _fma_1034;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1034) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_28));
                acc0_x_28 = _fma_1034;
                float _fma_1035;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1035) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_28));
                acc0_y_28 = _fma_1035;
                float _fma_1036;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1036) : "f"(S[46]), "f"(k_regs[14]), "f"(acc0_x_28));
                acc0_x_28 = _fma_1036;
                float _fma_1037;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1037) : "f"(S[47]), "f"(k_regs[15]), "f"(acc0_y_28));
                acc0_y_28 = _fma_1037;
                float dot_28 = acc0_x_28 + acc0_y_28;
                float _shfl_xor_sync_1038 = __shfl_xor_sync(0xFFFFFFFF, dot_28, 1);
                dot_28 = dot_28 + _shfl_xor_sync_1038;
                float _shfl_xor_sync_1039 = __shfl_xor_sync(0xFFFFFFFF, dot_28, 2);
                dot_28 = dot_28 + _shfl_xor_sync_1039;
                float _shfl_xor_sync_1040 = __shfl_xor_sync(0xFFFFFFFF, dot_28, 4);
                dot_28 = dot_28 + _shfl_xor_sync_1040;
#else
                float2 acc0_28 = make_float2(0.0f, 0.0f);
                acc0_28 = fma_f32x2_rn_ftz(S2[16], k2[0], acc0_28);
                acc0_28 = fma_f32x2_rn_ftz(S2[17], k2[1], acc0_28);
                acc0_28 = fma_f32x2_rn_ftz(S2[18], k2[2], acc0_28);
                acc0_28 = fma_f32x2_rn_ftz(S2[19], k2[3], acc0_28);
                acc0_28 = fma_f32x2_rn_ftz(S2[20], k2[4], acc0_28);
                acc0_28 = fma_f32x2_rn_ftz(S2[21], k2[5], acc0_28);
                acc0_28 = fma_f32x2_rn_ftz(S2[22], k2[6], acc0_28);
                acc0_28 = fma_f32x2_rn_ftz(S2[23], k2[7], acc0_28);
                float dot_28 = acc0_28.x + acc0_28.y;
                float _shfl_xor_sync_142 = __shfl_xor_sync(0xFFFFFFFF, dot_28, 1);
                dot_28 = dot_28 + _shfl_xor_sync_142;
                float _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, dot_28, 2);
                dot_28 = dot_28 + _shfl_xor_sync_143;
                float _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, dot_28, 4);
                dot_28 = dot_28 + _shfl_xor_sync_144;
#endif
                o4[2] = dot_28;
#if __CUDA_ARCH__ == 900
                float acc0_x_29 = 0.0f;
                float acc0_y_29 = 0.0f;
                float _fma_1041;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1041) : "f"(S[48]), "f"(k_regs[0]), "f"(acc0_x_29));
                acc0_x_29 = _fma_1041;
                float _fma_1042;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1042) : "f"(S[49]), "f"(k_regs[1]), "f"(acc0_y_29));
                acc0_y_29 = _fma_1042;
                float _fma_1043;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1043) : "f"(S[50]), "f"(k_regs[2]), "f"(acc0_x_29));
                acc0_x_29 = _fma_1043;
                float _fma_1044;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1044) : "f"(S[51]), "f"(k_regs[3]), "f"(acc0_y_29));
                acc0_y_29 = _fma_1044;
                float _fma_1045;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1045) : "f"(S[52]), "f"(k_regs[4]), "f"(acc0_x_29));
                acc0_x_29 = _fma_1045;
                float _fma_1046;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1046) : "f"(S[53]), "f"(k_regs[5]), "f"(acc0_y_29));
                acc0_y_29 = _fma_1046;
                float _fma_1047;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1047) : "f"(S[54]), "f"(k_regs[6]), "f"(acc0_x_29));
                acc0_x_29 = _fma_1047;
                float _fma_1048;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1048) : "f"(S[55]), "f"(k_regs[7]), "f"(acc0_y_29));
                acc0_y_29 = _fma_1048;
                float _fma_1049;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1049) : "f"(S[56]), "f"(k_regs[8]), "f"(acc0_x_29));
                acc0_x_29 = _fma_1049;
                float _fma_1050;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1050) : "f"(S[57]), "f"(k_regs[9]), "f"(acc0_y_29));
                acc0_y_29 = _fma_1050;
                float _fma_1051;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1051) : "f"(S[58]), "f"(k_regs[10]), "f"(acc0_x_29));
                acc0_x_29 = _fma_1051;
                float _fma_1052;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1052) : "f"(S[59]), "f"(k_regs[11]), "f"(acc0_y_29));
                acc0_y_29 = _fma_1052;
                float _fma_1053;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1053) : "f"(S[60]), "f"(k_regs[12]), "f"(acc0_x_29));
                acc0_x_29 = _fma_1053;
                float _fma_1054;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1054) : "f"(S[61]), "f"(k_regs[13]), "f"(acc0_y_29));
                acc0_y_29 = _fma_1054;
                float _fma_1055;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1055) : "f"(S[62]), "f"(k_regs[14]), "f"(acc0_x_29));
                acc0_x_29 = _fma_1055;
                float _fma_1056;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1056) : "f"(S[63]), "f"(k_regs[15]), "f"(acc0_y_29));
                acc0_y_29 = _fma_1056;
                float dot_29 = acc0_x_29 + acc0_y_29;
                float _shfl_xor_sync_1057 = __shfl_xor_sync(0xFFFFFFFF, dot_29, 1);
                dot_29 = dot_29 + _shfl_xor_sync_1057;
                float _shfl_xor_sync_1058 = __shfl_xor_sync(0xFFFFFFFF, dot_29, 2);
                dot_29 = dot_29 + _shfl_xor_sync_1058;
                float _shfl_xor_sync_1059 = __shfl_xor_sync(0xFFFFFFFF, dot_29, 4);
                dot_29 = dot_29 + _shfl_xor_sync_1059;
#else
                float2 acc0_29 = make_float2(0.0f, 0.0f);
                acc0_29 = fma_f32x2_rn_ftz(S2[24], k2[0], acc0_29);
                acc0_29 = fma_f32x2_rn_ftz(S2[25], k2[1], acc0_29);
                acc0_29 = fma_f32x2_rn_ftz(S2[26], k2[2], acc0_29);
                acc0_29 = fma_f32x2_rn_ftz(S2[27], k2[3], acc0_29);
                acc0_29 = fma_f32x2_rn_ftz(S2[28], k2[4], acc0_29);
                acc0_29 = fma_f32x2_rn_ftz(S2[29], k2[5], acc0_29);
                acc0_29 = fma_f32x2_rn_ftz(S2[30], k2[6], acc0_29);
                acc0_29 = fma_f32x2_rn_ftz(S2[31], k2[7], acc0_29);
                float dot_29 = acc0_29.x + acc0_29.y;
                float _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, dot_29, 1);
                dot_29 = dot_29 + _shfl_xor_sync_145;
                float _shfl_xor_sync_146 = __shfl_xor_sync(0xFFFFFFFF, dot_29, 2);
                dot_29 = dot_29 + _shfl_xor_sync_146;
                float _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, dot_29, 4);
                dot_29 = dot_29 + _shfl_xor_sync_147;
#endif
                o4[3] = dot_29;
                if (u == 0) {
                    long long out_index_1 = (long long)n * out_stride_p0 + (long long)t_1 * out_stride_p1 + (long long)h * out_stride_p2;
                    unsigned int packed_1[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o4[_lp*2 + 0], o4[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o4[_lp*2 + 2], o4[_lp*2+1 + 2]));
                        packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int2*>(out + (out_index_1 + (long long)row0))[0] = reinterpret_cast<int2*>(packed_1)[0];
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
                        *reinterpret_cast<float4*>(state + write_rows_base + 2ll * state_stride_p2) = _v4;
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
                        *reinterpret_cast<float4*>(state + write_rows_base + 2ll * state_stride_p2 + 32ll) = _v4;
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
                        *reinterpret_cast<float4*>(state + write_rows_base + 2ll * state_stride_p2 + 64ll) = _v4;
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
                        *reinterpret_cast<float4*>(state + write_rows_base + 2ll * state_stride_p2 + 96ll) = _v4;
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
                        *reinterpret_cast<float4*>(state + write_rows_base + 3ll * state_stride_p2) = _v4;
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
                        *reinterpret_cast<float4*>(state + write_rows_base + 3ll * state_stride_p2 + 32ll) = _v4;
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
                        *reinterpret_cast<float4*>(state + write_rows_base + 3ll * state_stride_p2 + 64ll) = _v4;
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
                        *reinterpret_cast<float4*>(state + write_rows_base + 3ll * state_stride_p2 + 96ll) = _v4;
                    }
                }
            }
        }
    }
}

} // extern "C"
