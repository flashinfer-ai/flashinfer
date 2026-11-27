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
#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
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
__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

__device__ __forceinline__ float mul_f32_rn_ftz(float a, float b) {
    float r;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
#else
__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
#endif
    return r;
}

#if __CUDA_ARCH__ == 900
__device__ __forceinline__ float mul_f32_rz_noftz(float a, float b) {
    float r;
    asm("mul.rz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
#else

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
kernel_cake_gdp_72b6872c98ed9abf0bdf(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __half* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, int* __restrict__ ssm_state_indices, long long* __restrict__ rand_seed, CakeFastDivmod heads, int num_tokens, float scale, int disable_state_update, int use_ssm_state_indices, int philox_rounds, long long state_size_p0, long long state_stride_p1, long long state_stride_p2, long long state_stride_p0, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long ssm_state_indices_stride_p0, long long ssm_state_indices_stride_p1, long long out_stride_p0, long long out_stride_p1, long long out_stride_p2)
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
    unsigned int* sNoise = reinterpret_cast<unsigned int*>(smem_raw + 19856);
    const int sNoise_addr = smem + 19856;

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
    int steps = 21;
    unsigned int seed_lo = 0;
    unsigned int seed_hi = 0;
    if (philox_rounds != 0) {
        long long seed_value = *rand_seed;
        seed_lo = (unsigned int)seed_value;
        seed_hi = (unsigned int)(seed_value >> 32ll);
    }
    long long read_slot_head = (long long)read_state_slot_raw * state_stride_p0 + (long long)h * state_stride_p1;
    float r_q[4];
    float r_k[4];
    float bk[12];
    float bv[6];
    float bb[3];
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
    int bm_2 = ((bm0_2) < (steps - 1) ? (bm0_2) : (steps - 1));
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
    float bq[4];
    float ba[3];
    int bt0_0 = warp_local;
    int bt_0 = ((bt0_0) < (num_tokens - 1) ? (bt0_0) : (num_tokens - 1));
    long long bqi_0 = (long long)n * q_stride_p0 + (long long)bt_0 * q_stride_p1 + (long long)h * q_stride_p2;
    {
        uint2 _vld_3;
        _vld_3 = *reinterpret_cast<const uint2*>(q + bqi_0 + (long long)k_col0);
        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&bq[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
            (&bq[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
        }
    }
    long long bgi_0 = (long long)n * a_stride_p0 + (long long)bt_0 * a_stride_p1 + (long long)h * a_stride_p2;
    ba[0] = (float)a[bgi_0];
    ba[1] = dt_bias[h];
    ba[2] = A_log[h];
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
        unsigned int h_words[2];
        unsigned int h_noise[16];
        {
            uint2 _vld_4;
            _vld_4 = *reinterpret_cast<const uint2*>(state + read_rows_base);
            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[0 + _pair * 2]))
                    : "r"(_vpairs_4[_pair]));
            }
        }
        {
            uint2 _vld_5;
            _vld_5 = *reinterpret_cast<const uint2*>(state + read_rows_base + 32ll);
            uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[4 + _pair * 2]))
                    : "r"(_vpairs_5[_pair]));
            }
        }
        {
            uint2 _vld_6;
            _vld_6 = *reinterpret_cast<const uint2*>(state + read_rows_base + 64ll);
            uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[8 + _pair * 2]))
                    : "r"(_vpairs_6[_pair]));
            }
        }
        {
            uint2 _vld_7;
            _vld_7 = *reinterpret_cast<const uint2*>(state + read_rows_base + 96ll);
            uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[12 + _pair * 2]))
                    : "r"(_vpairs_7[_pair]));
            }
        }
        {
            uint2 _vld_8;
            _vld_8 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2);
            uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[16 + _pair * 2]))
                    : "r"(_vpairs_8[_pair]));
            }
        }
        {
            uint2 _vld_9;
            _vld_9 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 32ll);
            uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[20 + _pair * 2]))
                    : "r"(_vpairs_9[_pair]));
            }
        }
        {
            uint2 _vld_10;
            _vld_10 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 64ll);
            uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[24 + _pair * 2]))
                    : "r"(_vpairs_10[_pair]));
            }
        }
        {
            uint2 _vld_11;
            _vld_11 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 96ll);
            uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[28 + _pair * 2]))
                    : "r"(_vpairs_11[_pair]));
            }
        }
        unsigned int noise_lane = sNoise_addr + (unsigned int)(tid_0 * 16);
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
        float _fma_12;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(x1_x), "f"(x1_x), "f"(sum_x));
        sum_x = _fma_12;
        float _fma_13;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(x1_y), "f"(x1_y), "f"(sum_y));
        sum_y = _fma_13;
        sum_x = sum_x + sum_y;
        float _warp_reduce_14 = sum_x;
#else
        float2 x0 = make_float2(bk[0], bk[1]);
        float2 x1 = make_float2(bk[2], bk[3]);
        float2 sum_pair = fma_f32x2_rn_ftz(x0, x0, make_float2(0.0f, 0.0f));
        sum_pair = fma_f32x2_rn_ftz(x1, x1, sum_pair);
        float sum_x = sum_pair.x + sum_pair.y;
        float _warp_reduce_12 = sum_x;
#endif
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
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
#else
            _warp_reduce_12 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_12, offset);
        float norm = rsqrtf(_warp_reduce_12 + 1e-06f);
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
            float _exp_16 = __expf(-bb[0]);
            float _rcp_15 = approx_rcp(1.0f + _exp_16);
            sBeta[fm_0] = _rcp_15;
#else
            float _exp_14 = __expf(-bb[0]);
            float _rcp_13 = approx_rcp(1.0f + _exp_14);
            sBeta[fm_0] = _rcp_13;
#endif
        }
        int fm_1 = 8 + warp_local;
#if __CUDA_ARCH__ == 900
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
#else
        float2 x0_0 = make_float2(bk[4], bk[5]);
        float2 x1_0 = make_float2(bk[6], bk[7]);
        float2 sum_pair_0 = fma_f32x2_rn_ftz(x0_0, x0_0, make_float2(0.0f, 0.0f));
        sum_pair_0 = fma_f32x2_rn_ftz(x1_0, x1_0, sum_pair_0);
        float sum_x_0 = sum_pair_0.x + sum_pair_0.y;
        float _warp_reduce_15 = sum_x_0;
#endif
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
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
#else
            _warp_reduce_15 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_15, offset);
        float norm_0 = rsqrtf(_warp_reduce_15 + 1e-06f);
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
            float _exp_21 = __expf(-bb[1]);
            float _rcp_20 = approx_rcp(1.0f + _exp_21);
            sBeta[fm_1] = _rcp_20;
#else
            float _exp_17 = __expf(-bb[1]);
            float _rcp_16 = approx_rcp(1.0f + _exp_17);
            sBeta[fm_1] = _rcp_16;
#endif
        }
        int fm_2 = 16 + warp_local;
        if (fm_2 < steps) {
#if __CUDA_ARCH__ == 900
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
#else
            float2 x0_1 = make_float2(bk[8], bk[9]);
            float2 x1_1 = make_float2(bk[10], bk[11]);
            float2 sum_pair_1 = fma_f32x2_rn_ftz(x0_1, x0_1, make_float2(0.0f, 0.0f));
            sum_pair_1 = fma_f32x2_rn_ftz(x1_1, x1_1, sum_pair_1);
            float sum_x_1 = sum_pair_1.x + sum_pair_1.y;
            float _warp_reduce_18 = sum_x_1;
#endif
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
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
#else
                _warp_reduce_18 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_18, offset);
            float norm_1 = rsqrtf(_warp_reduce_18 + 1e-06f);
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
                float _exp_26 = __expf(-bb[2]);
                float _rcp_25 = approx_rcp(1.0f + _exp_26);
                sBeta[fm_2] = _rcp_25;
#else
                float _exp_20 = __expf(-bb[2]);
                float _rcp_19 = approx_rcp(1.0f + _exp_20);
                sBeta[fm_2] = _rcp_19;
#endif
            }
        }
        int ft_0 = warp_local;
        if (ft_0 < num_tokens) {
#if __CUDA_ARCH__ == 900
            float x0_x_2 = bq[0];
            float x0_y_2 = bq[1];
            float x1_x_2 = bq[2];
            float x1_y_2 = bq[3];
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
#else
            float2 x0_2 = make_float2(bq[0], bq[1]);
            float2 x1_2 = make_float2(bq[2], bq[3]);
            float2 sum_pair_2 = fma_f32x2_rn_ftz(x0_2, x0_2, make_float2(0.0f, 0.0f));
            sum_pair_2 = fma_f32x2_rn_ftz(x1_2, x1_2, sum_pair_2);
            float sum_x_2 = sum_pair_2.x + sum_pair_2.y;
            float _warp_reduce_21 = sum_x_2;
#endif
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
                _warp_reduce_29 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_29, offset);
            float norm_2 = rsqrtf(_warp_reduce_29 + 1e-06f);
#else
                _warp_reduce_21 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_21, offset);
            float norm_2 = rsqrtf(_warp_reduce_21 + 1e-06f);
#endif
            norm_2 = norm_2 * scale;
#if __CUDA_ARCH__ == 900
            x0_x_2 = mul_f32_rn_ftz(x0_x_2, norm_2);
            x0_y_2 = mul_f32_rn_ftz(x0_y_2, norm_2);
            x1_x_2 = mul_f32_rn_ftz(x1_x_2, norm_2);
            x1_y_2 = mul_f32_rn_ftz(x1_y_2, norm_2);
            sQ[ft_0 * 128 + k_col0] = x0_x_2;
            sQ[ft_0 * 128 + k_col0 + 1] = x0_y_2;
            sQ[ft_0 * 128 + k_col0 + 2] = x1_x_2;
            sQ[ft_0 * 128 + k_col0 + 3] = x1_y_2;
#else
            float2 norm2_2 = make_float2(norm_2, norm_2);
            x0_2 = mul_f32x2(x0_2, norm2_2);
            x1_2 = mul_f32x2(x1_2, norm2_2);
            sQ[ft_0 * 128 + k_col0] = x0_2.x;
            sQ[ft_0 * 128 + k_col0 + 1] = x0_2.y;
            sQ[ft_0 * 128 + k_col0 + 2] = x1_2.x;
            sQ[ft_0 * 128 + k_col0 + 3] = x1_2.y;
#endif
            if (lane_local == 0) {
                float xv = ba[0] + ba[1];
                float softplus_x = xv;
                if (xv <= 20.0f) {
#if __CUDA_ARCH__ == 900
                    float _exp_31 = __expf(xv);
                    float _log2_30;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_30) : "f"(1.0f + _exp_31));
                    softplus_x = _log2_30 * 0.6931472f;
#else
                    float _exp_23 = __expf(xv);
                    float _log2_22;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_22) : "f"(1.0f + _exp_23));
                    softplus_x = _log2_22 * 0.6931472f;
#endif
                }
#if __CUDA_ARCH__ == 900
                float _exp_32 = __expf(ba[2]);
                float g_log = -_exp_32 * softplus_x;
                float _exp_33 = __expf(g_log);
                sAlpha[ft_0] = _exp_33;
#else
                float _exp_24 = __expf(ba[2]);
                float g_log = -_exp_24 * softplus_x;
                float _exp_25 = __expf(g_log);
                sAlpha[ft_0] = _exp_25;
#endif
                if (use_ssm_state_indices != 0) {
                    sSsm[ft_0] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)ft_0 * ssm_state_indices_stride_p1];
                }
                if (use_ssm_state_indices == 0) {
                    sSsm[ft_0] = -1;
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
#endif
        if (use_ssm_state_indices != 0) {
            if (philox_rounds != 0) {
                unsigned int key0 = seed_lo;
                unsigned int key1 = seed_hi;
                long long o_0 = read_rows_base;
                unsigned long long p0_0 = (unsigned long long)(unsigned int)o_0 * 3528531795ull;
                h_noise[0] = (unsigned int)(o_0 >> 32ll) ^ key0;
                h_noise[2] = (unsigned int)(p0_0 >> 32ull) ^ key1;
                h_noise[3] = (unsigned int)p0_0;
                long long o_1 = read_rows_base + 32ll;
                unsigned long long p0_1 = (unsigned long long)(unsigned int)o_1 * 3528531795ull;
                h_noise[4] = (unsigned int)(o_1 >> 32ll) ^ key0;
                h_noise[6] = (unsigned int)(p0_1 >> 32ull) ^ key1;
                h_noise[7] = (unsigned int)p0_1;
                long long o_2 = read_rows_base + state_stride_p2;
                unsigned long long p0_2 = (unsigned long long)(unsigned int)o_2 * 3528531795ull;
                h_noise[8] = (unsigned int)(o_2 >> 32ll) ^ key0;
                h_noise[10] = (unsigned int)(p0_2 >> 32ull) ^ key1;
                h_noise[11] = (unsigned int)p0_2;
                long long o_3 = read_rows_base + state_stride_p2 + 32ll;
                unsigned long long p0_3 = (unsigned long long)(unsigned int)o_3 * 3528531795ull;
                h_noise[12] = (unsigned int)(o_3 >> 32ll) ^ key0;
                h_noise[14] = (unsigned int)(p0_3 >> 32ull) ^ key1;
                h_noise[15] = (unsigned int)p0_3;
                key0 = key0 + 2654435769u;
                key1 = key1 + 3144134277u;
                unsigned long long q0_0 = (unsigned long long)h_noise[0] * 3528531795ull;
                unsigned long long q1_0 = (unsigned long long)h_noise[2] * 3449720151ull;
                h_noise[0] = (unsigned int)(q1_0 >> 32ull) ^ key0;
                h_noise[1] = (unsigned int)q1_0;
                h_noise[2] = (unsigned int)(q0_0 >> 32ull) ^ h_noise[3] ^ key1;
                h_noise[3] = (unsigned int)q0_0;
                unsigned long long q0_1 = (unsigned long long)h_noise[4] * 3528531795ull;
                unsigned long long q1_1 = (unsigned long long)h_noise[6] * 3449720151ull;
                h_noise[4] = (unsigned int)(q1_1 >> 32ull) ^ key0;
                h_noise[5] = (unsigned int)q1_1;
                h_noise[6] = (unsigned int)(q0_1 >> 32ull) ^ h_noise[7] ^ key1;
                h_noise[7] = (unsigned int)q0_1;
                unsigned long long q0_2 = (unsigned long long)h_noise[8] * 3528531795ull;
                unsigned long long q1_2 = (unsigned long long)h_noise[10] * 3449720151ull;
                h_noise[8] = (unsigned int)(q1_2 >> 32ull) ^ key0;
                h_noise[9] = (unsigned int)q1_2;
                h_noise[10] = (unsigned int)(q0_2 >> 32ull) ^ h_noise[11] ^ key1;
                h_noise[11] = (unsigned int)q0_2;
                unsigned long long q0_3 = (unsigned long long)h_noise[12] * 3528531795ull;
                unsigned long long q1_3 = (unsigned long long)h_noise[14] * 3449720151ull;
                h_noise[12] = (unsigned int)(q1_3 >> 32ull) ^ key0;
                h_noise[13] = (unsigned int)q1_3;
                h_noise[14] = (unsigned int)(q0_3 >> 32ull) ^ h_noise[15] ^ key1;
                h_noise[15] = (unsigned int)q0_3;
                key0 = key0 + 2654435769u;
                key1 = key1 + 3144134277u;
                unsigned long long s0_0 = (unsigned long long)h_noise[0] * 3528531795ull;
                unsigned long long s1_0 = (unsigned long long)h_noise[2] * 3449720151ull;
                h_noise[0] = (unsigned int)(s1_0 >> 32ull) ^ h_noise[1] ^ key0;
                h_noise[1] = (unsigned int)s1_0;
                h_noise[2] = (unsigned int)(s0_0 >> 32ull) ^ h_noise[3] ^ key1;
                h_noise[3] = (unsigned int)s0_0;
                unsigned long long s0_1 = (unsigned long long)h_noise[4] * 3528531795ull;
                unsigned long long s1_1 = (unsigned long long)h_noise[6] * 3449720151ull;
                h_noise[4] = (unsigned int)(s1_1 >> 32ull) ^ h_noise[5] ^ key0;
                h_noise[5] = (unsigned int)s1_1;
                h_noise[6] = (unsigned int)(s0_1 >> 32ull) ^ h_noise[7] ^ key1;
                h_noise[7] = (unsigned int)s0_1;
                unsigned long long s0_2 = (unsigned long long)h_noise[8] * 3528531795ull;
                unsigned long long s1_2 = (unsigned long long)h_noise[10] * 3449720151ull;
                h_noise[8] = (unsigned int)(s1_2 >> 32ull) ^ h_noise[9] ^ key0;
                h_noise[9] = (unsigned int)s1_2;
                h_noise[10] = (unsigned int)(s0_2 >> 32ull) ^ h_noise[11] ^ key1;
                h_noise[11] = (unsigned int)s0_2;
                unsigned long long s0_3 = (unsigned long long)h_noise[12] * 3528531795ull;
                unsigned long long s1_3 = (unsigned long long)h_noise[14] * 3449720151ull;
                h_noise[12] = (unsigned int)(s1_3 >> 32ull) ^ h_noise[13] ^ key0;
                h_noise[13] = (unsigned int)s1_3;
                h_noise[14] = (unsigned int)(s0_3 >> 32ull) ^ h_noise[15] ^ key1;
                h_noise[15] = (unsigned int)s0_3;
                key0 = key0 + 2654435769u;
                key1 = key1 + 3144134277u;
                unsigned long long s0_4 = (unsigned long long)h_noise[0] * 3528531795ull;
                unsigned long long s1_4 = (unsigned long long)h_noise[2] * 3449720151ull;
                h_noise[0] = (unsigned int)(s1_4 >> 32ull) ^ h_noise[1] ^ key0;
                h_noise[1] = (unsigned int)s1_4;
                h_noise[2] = (unsigned int)(s0_4 >> 32ull) ^ h_noise[3] ^ key1;
                h_noise[3] = (unsigned int)s0_4;
                unsigned long long s0_5 = (unsigned long long)h_noise[4] * 3528531795ull;
                unsigned long long s1_5 = (unsigned long long)h_noise[6] * 3449720151ull;
                h_noise[4] = (unsigned int)(s1_5 >> 32ull) ^ h_noise[5] ^ key0;
                h_noise[5] = (unsigned int)s1_5;
                h_noise[6] = (unsigned int)(s0_5 >> 32ull) ^ h_noise[7] ^ key1;
                h_noise[7] = (unsigned int)s0_5;
                unsigned long long s0_6 = (unsigned long long)h_noise[8] * 3528531795ull;
                unsigned long long s1_6 = (unsigned long long)h_noise[10] * 3449720151ull;
                h_noise[8] = (unsigned int)(s1_6 >> 32ull) ^ h_noise[9] ^ key0;
                h_noise[9] = (unsigned int)s1_6;
                h_noise[10] = (unsigned int)(s0_6 >> 32ull) ^ h_noise[11] ^ key1;
                h_noise[11] = (unsigned int)s0_6;
                unsigned long long s0_7 = (unsigned long long)h_noise[12] * 3528531795ull;
                unsigned long long s1_7 = (unsigned long long)h_noise[14] * 3449720151ull;
                h_noise[12] = (unsigned int)(s1_7 >> 32ull) ^ h_noise[13] ^ key0;
                h_noise[13] = (unsigned int)s1_7;
                h_noise[14] = (unsigned int)(s0_7 >> 32ull) ^ h_noise[15] ^ key1;
                h_noise[15] = (unsigned int)s0_7;
                key0 = key0 + 2654435769u;
                key1 = key1 + 3144134277u;
                unsigned long long s0_8 = (unsigned long long)h_noise[0] * 3528531795ull;
                unsigned long long s1_8 = (unsigned long long)h_noise[2] * 3449720151ull;
                h_noise[0] = (unsigned int)(s1_8 >> 32ull) ^ h_noise[1] ^ key0;
                h_noise[1] = (unsigned int)s1_8;
                h_noise[2] = (unsigned int)(s0_8 >> 32ull) ^ h_noise[3] ^ key1;
                h_noise[3] = (unsigned int)s0_8;
                unsigned long long s0_9 = (unsigned long long)h_noise[4] * 3528531795ull;
                unsigned long long s1_9 = (unsigned long long)h_noise[6] * 3449720151ull;
                h_noise[4] = (unsigned int)(s1_9 >> 32ull) ^ h_noise[5] ^ key0;
                h_noise[5] = (unsigned int)s1_9;
                h_noise[6] = (unsigned int)(s0_9 >> 32ull) ^ h_noise[7] ^ key1;
                h_noise[7] = (unsigned int)s0_9;
                unsigned long long s0_10 = (unsigned long long)h_noise[8] * 3528531795ull;
                unsigned long long s1_10 = (unsigned long long)h_noise[10] * 3449720151ull;
                h_noise[8] = (unsigned int)(s1_10 >> 32ull) ^ h_noise[9] ^ key0;
                h_noise[9] = (unsigned int)s1_10;
                h_noise[10] = (unsigned int)(s0_10 >> 32ull) ^ h_noise[11] ^ key1;
                h_noise[11] = (unsigned int)s0_10;
                unsigned long long s0_11 = (unsigned long long)h_noise[12] * 3528531795ull;
                unsigned long long s1_11 = (unsigned long long)h_noise[14] * 3449720151ull;
                h_noise[12] = (unsigned int)(s1_11 >> 32ull) ^ h_noise[13] ^ key0;
                h_noise[13] = (unsigned int)s1_11;
                h_noise[14] = (unsigned int)(s0_11 >> 32ull) ^ h_noise[15] ^ key1;
                h_noise[15] = (unsigned int)s0_11;
                key0 = key0 + 2654435769u;
                key1 = key1 + 3144134277u;
                #pragma unroll 1
                for (int _round = 5; _round < philox_rounds; _round++) {
                    unsigned long long t0_0 = (unsigned long long)h_noise[0] * 3528531795ull;
                    unsigned long long t1_0 = (unsigned long long)h_noise[2] * 3449720151ull;
                    h_noise[0] = (unsigned int)(t1_0 >> 32ull) ^ h_noise[1] ^ key0;
                    h_noise[1] = (unsigned int)t1_0;
                    h_noise[2] = (unsigned int)(t0_0 >> 32ull) ^ h_noise[3] ^ key1;
                    h_noise[3] = (unsigned int)t0_0;
                    unsigned long long t0_1 = (unsigned long long)h_noise[4] * 3528531795ull;
                    unsigned long long t1_1 = (unsigned long long)h_noise[6] * 3449720151ull;
                    h_noise[4] = (unsigned int)(t1_1 >> 32ull) ^ h_noise[5] ^ key0;
                    h_noise[5] = (unsigned int)t1_1;
                    h_noise[6] = (unsigned int)(t0_1 >> 32ull) ^ h_noise[7] ^ key1;
                    h_noise[7] = (unsigned int)t0_1;
                    unsigned long long t0_2 = (unsigned long long)h_noise[8] * 3528531795ull;
                    unsigned long long t1_2 = (unsigned long long)h_noise[10] * 3449720151ull;
                    h_noise[8] = (unsigned int)(t1_2 >> 32ull) ^ h_noise[9] ^ key0;
                    h_noise[9] = (unsigned int)t1_2;
                    h_noise[10] = (unsigned int)(t0_2 >> 32ull) ^ h_noise[11] ^ key1;
                    h_noise[11] = (unsigned int)t0_2;
                    unsigned long long t0_3 = (unsigned long long)h_noise[12] * 3528531795ull;
                    unsigned long long t1_3 = (unsigned long long)h_noise[14] * 3449720151ull;
                    h_noise[12] = (unsigned int)(t1_3 >> 32ull) ^ h_noise[13] ^ key0;
                    h_noise[13] = (unsigned int)t1_3;
                    h_noise[14] = (unsigned int)(t0_3 >> 32ull) ^ h_noise[15] ^ key1;
                    h_noise[15] = (unsigned int)t0_3;
                    key0 = key0 + 2654435769u;
                    key1 = key1 + 3144134277u;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 0), "r"(h_noise[0] & 536813567), "r"(h_noise[1] & 536813567), "r"(h_noise[2] & 536813567), "r"(h_noise[3] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 4096), "r"(h_noise[4] & 536813567), "r"(h_noise[5] & 536813567), "r"(h_noise[6] & 536813567), "r"(h_noise[7] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 8192), "r"(h_noise[8] & 536813567), "r"(h_noise[9] & 536813567), "r"(h_noise[10] & 536813567), "r"(h_noise[11] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 12288), "r"(h_noise[12] & 536813567), "r"(h_noise[13] & 536813567), "r"(h_noise[14] & 536813567), "r"(h_noise[15] & 536813567) : "memory");
            }
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
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_0 * 256) + group_byte));
                float beta_0 = sBeta[m_0];
#if __CUDA_ARCH__ == 900
                float acc0_x = 0.0f;
                float acc0_y = 0.0f;
                float acc1_x = 0.0f;
                float acc1_y = 0.0f;
                float _fma_34;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x));
                acc0_x = _fma_34;
                float _fma_35;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y));
                acc0_y = _fma_35;
                float _fma_36;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x));
                acc1_x = _fma_36;
                float _fma_37;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y));
                acc1_y = _fma_37;
                float _fma_38;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x));
                acc0_x = _fma_38;
                float _fma_39;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y));
                acc0_y = _fma_39;
                float _fma_40;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x));
                acc1_x = _fma_40;
                float _fma_41;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y));
                acc1_y = _fma_41;
                float _fma_42;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x));
                acc0_x = _fma_42;
                float _fma_43;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y));
                acc0_y = _fma_43;
                float _fma_44;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x));
                acc1_x = _fma_44;
                float _fma_45;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y));
                acc1_y = _fma_45;
                float _fma_46;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x));
                acc0_x = _fma_46;
                float _fma_47;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y));
                acc0_y = _fma_47;
                float _fma_48;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x));
                acc1_x = _fma_48;
                float _fma_49;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y));
                acc1_y = _fma_49;
                float _add_50;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(acc0_x), "f"(acc1_x));
                float _add_51;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(acc0_y), "f"(acc1_y));
                float dot = _add_50 + _add_51;
                float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
                dot = dot + _shfl_xor_sync_52;
                float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
                dot = dot + _shfl_xor_sync_53;
                float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
                dot = dot + _shfl_xor_sync_54;
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
                float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
                dot = dot + _shfl_xor_sync_26;
                float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
                dot = dot + _shfl_xor_sync_27;
                float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
                dot = dot + _shfl_xor_sync_28;
#endif
                float vn_0 = (v_regs[0] - dot) * beta_0;
#if __CUDA_ARCH__ == 900
                float _fma_55;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(k_regs[0]), "f"(vn_0), "f"(S[0]));
                S[0] = _fma_55;
                float _fma_56;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(k_regs[1]), "f"(vn_0), "f"(S[1]));
                S[1] = _fma_56;
                float _fma_57;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(k_regs[2]), "f"(vn_0), "f"(S[2]));
                S[2] = _fma_57;
                float _fma_58;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(k_regs[3]), "f"(vn_0), "f"(S[3]));
                S[3] = _fma_58;
                float _fma_59;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(k_regs[4]), "f"(vn_0), "f"(S[4]));
                S[4] = _fma_59;
                float _fma_60;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(k_regs[5]), "f"(vn_0), "f"(S[5]));
                S[5] = _fma_60;
                float _fma_61;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(k_regs[6]), "f"(vn_0), "f"(S[6]));
                S[6] = _fma_61;
                float _fma_62;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(k_regs[7]), "f"(vn_0), "f"(S[7]));
                S[7] = _fma_62;
                float _fma_63;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(k_regs[8]), "f"(vn_0), "f"(S[8]));
                S[8] = _fma_63;
                float _fma_64;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(k_regs[9]), "f"(vn_0), "f"(S[9]));
                S[9] = _fma_64;
                float _fma_65;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(k_regs[10]), "f"(vn_0), "f"(S[10]));
                S[10] = _fma_65;
                float _fma_66;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(k_regs[11]), "f"(vn_0), "f"(S[11]));
                S[11] = _fma_66;
                float _fma_67;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(k_regs[12]), "f"(vn_0), "f"(S[12]));
                S[12] = _fma_67;
                float _fma_68;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(k_regs[13]), "f"(vn_0), "f"(S[13]));
                S[13] = _fma_68;
                float _fma_69;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(k_regs[14]), "f"(vn_0), "f"(S[14]));
                S[14] = _fma_69;
                float _fma_70;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(k_regs[15]), "f"(vn_0), "f"(S[15]));
                S[15] = _fma_70;
                float acc0_x_0 = 0.0f;
                float acc0_y_0 = 0.0f;
                float acc1_x_0 = 0.0f;
                float acc1_y_0 = 0.0f;
                float _fma_71;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_0));
                acc0_x_0 = _fma_71;
                float _fma_72;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_0));
                acc0_y_0 = _fma_72;
                float _fma_73;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_0));
                acc1_x_0 = _fma_73;
                float _fma_74;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_0));
                acc1_y_0 = _fma_74;
                float _fma_75;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_0));
                acc0_x_0 = _fma_75;
                float _fma_76;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_0));
                acc0_y_0 = _fma_76;
                float _fma_77;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_0));
                acc1_x_0 = _fma_77;
                float _fma_78;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_78) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_0));
                acc1_y_0 = _fma_78;
                float _fma_79;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_0));
                acc0_x_0 = _fma_79;
                float _fma_80;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_0));
                acc0_y_0 = _fma_80;
                float _fma_81;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_0));
                acc1_x_0 = _fma_81;
                float _fma_82;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_0));
                acc1_y_0 = _fma_82;
                float _fma_83;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_0));
                acc0_x_0 = _fma_83;
                float _fma_84;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_0));
                acc0_y_0 = _fma_84;
                float _fma_85;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_0));
                acc1_x_0 = _fma_85;
                float _fma_86;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_0));
                acc1_y_0 = _fma_86;
                float _add_87;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(acc0_x_0), "f"(acc1_x_0));
                float _add_88;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(acc0_y_0), "f"(acc1_y_0));
                float dot_0 = _add_87 + _add_88;
                float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 1);
                dot_0 = dot_0 + _shfl_xor_sync_89;
                float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 2);
                dot_0 = dot_0 + _shfl_xor_sync_90;
                float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 4);
                dot_0 = dot_0 + _shfl_xor_sync_91;
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
                float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 1);
                dot_0 = dot_0 + _shfl_xor_sync_29;
                float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 2);
                dot_0 = dot_0 + _shfl_xor_sync_30;
                float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 4);
                dot_0 = dot_0 + _shfl_xor_sync_31;
#endif
                float vn_1 = (v_regs[1] - dot_0) * beta_0;
#if __CUDA_ARCH__ == 900
                float _fma_92;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(k_regs[0]), "f"(vn_1), "f"(S[16]));
                S[16] = _fma_92;
                float _fma_93;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(k_regs[1]), "f"(vn_1), "f"(S[17]));
                S[17] = _fma_93;
                float _fma_94;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(k_regs[2]), "f"(vn_1), "f"(S[18]));
                S[18] = _fma_94;
                float _fma_95;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(k_regs[3]), "f"(vn_1), "f"(S[19]));
                S[19] = _fma_95;
                float _fma_96;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(k_regs[4]), "f"(vn_1), "f"(S[20]));
                S[20] = _fma_96;
                float _fma_97;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(k_regs[5]), "f"(vn_1), "f"(S[21]));
                S[21] = _fma_97;
                float _fma_98;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(k_regs[6]), "f"(vn_1), "f"(S[22]));
                S[22] = _fma_98;
                float _fma_99;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(k_regs[7]), "f"(vn_1), "f"(S[23]));
                S[23] = _fma_99;
                float _fma_100;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(k_regs[8]), "f"(vn_1), "f"(S[24]));
                S[24] = _fma_100;
                float _fma_101;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(k_regs[9]), "f"(vn_1), "f"(S[25]));
                S[25] = _fma_101;
                float _fma_102;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(k_regs[10]), "f"(vn_1), "f"(S[26]));
                S[26] = _fma_102;
                float _fma_103;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(k_regs[11]), "f"(vn_1), "f"(S[27]));
                S[27] = _fma_103;
                float _fma_104;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(k_regs[12]), "f"(vn_1), "f"(S[28]));
                S[28] = _fma_104;
                float _fma_105;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(k_regs[13]), "f"(vn_1), "f"(S[29]));
                S[29] = _fma_105;
                float _fma_106;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(k_regs[14]), "f"(vn_1), "f"(S[30]));
                S[30] = _fma_106;
                float _fma_107;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(k_regs[15]), "f"(vn_1), "f"(S[31]));
                S[31] = _fma_107;
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
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_1 * 256) + group_byte));
                float beta_1 = sBeta[m_1];
#if __CUDA_ARCH__ == 900
                float acc0_x_1 = 0.0f;
                float acc0_y_1 = 0.0f;
                float acc1_x_1 = 0.0f;
                float acc1_y_1 = 0.0f;
                float _fma_108;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_1));
                acc0_x_1 = _fma_108;
                float _fma_109;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_1));
                acc0_y_1 = _fma_109;
                float _fma_110;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_1));
                acc1_x_1 = _fma_110;
                float _fma_111;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_111) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_1));
                acc1_y_1 = _fma_111;
                float _fma_112;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_112) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_1));
                acc0_x_1 = _fma_112;
                float _fma_113;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_113) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_1));
                acc0_y_1 = _fma_113;
                float _fma_114;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_114) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_1));
                acc1_x_1 = _fma_114;
                float _fma_115;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_1));
                acc1_y_1 = _fma_115;
                float _fma_116;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_1));
                acc0_x_1 = _fma_116;
                float _fma_117;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_1));
                acc0_y_1 = _fma_117;
                float _fma_118;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_118) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_1));
                acc1_x_1 = _fma_118;
                float _fma_119;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_1));
                acc1_y_1 = _fma_119;
                float _fma_120;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_1));
                acc0_x_1 = _fma_120;
                float _fma_121;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_1));
                acc0_y_1 = _fma_121;
                float _fma_122;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_1));
                acc1_x_1 = _fma_122;
                float _fma_123;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_123) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_1));
                acc1_y_1 = _fma_123;
                float _add_124;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_124) : "f"(acc0_x_1), "f"(acc1_x_1));
                float _add_125;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_125) : "f"(acc0_y_1), "f"(acc1_y_1));
                float dot_1 = _add_124 + _add_125;
                float _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_126;
                float _shfl_xor_sync_127 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_127;
                float _shfl_xor_sync_128 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
                dot_1 = dot_1 + _shfl_xor_sync_128;
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
                float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_32;
                float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_33;
                float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
                dot_1 = dot_1 + _shfl_xor_sync_34;
#endif
                float vn_2 = (v_regs[0] - dot_1) * beta_1;
#if __CUDA_ARCH__ == 900
                float _fma_129;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_129) : "f"(k_regs[0]), "f"(vn_2), "f"(S[0]));
                S[0] = _fma_129;
                float _fma_130;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_130) : "f"(k_regs[1]), "f"(vn_2), "f"(S[1]));
                S[1] = _fma_130;
                float _fma_131;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(k_regs[2]), "f"(vn_2), "f"(S[2]));
                S[2] = _fma_131;
                float _fma_132;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(k_regs[3]), "f"(vn_2), "f"(S[3]));
                S[3] = _fma_132;
                float _fma_133;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(k_regs[4]), "f"(vn_2), "f"(S[4]));
                S[4] = _fma_133;
                float _fma_134;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_134) : "f"(k_regs[5]), "f"(vn_2), "f"(S[5]));
                S[5] = _fma_134;
                float _fma_135;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_135) : "f"(k_regs[6]), "f"(vn_2), "f"(S[6]));
                S[6] = _fma_135;
                float _fma_136;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_136) : "f"(k_regs[7]), "f"(vn_2), "f"(S[7]));
                S[7] = _fma_136;
                float _fma_137;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_137) : "f"(k_regs[8]), "f"(vn_2), "f"(S[8]));
                S[8] = _fma_137;
                float _fma_138;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_138) : "f"(k_regs[9]), "f"(vn_2), "f"(S[9]));
                S[9] = _fma_138;
                float _fma_139;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_139) : "f"(k_regs[10]), "f"(vn_2), "f"(S[10]));
                S[10] = _fma_139;
                float _fma_140;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_140) : "f"(k_regs[11]), "f"(vn_2), "f"(S[11]));
                S[11] = _fma_140;
                float _fma_141;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_141) : "f"(k_regs[12]), "f"(vn_2), "f"(S[12]));
                S[12] = _fma_141;
                float _fma_142;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(k_regs[13]), "f"(vn_2), "f"(S[13]));
                S[13] = _fma_142;
                float _fma_143;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_143) : "f"(k_regs[14]), "f"(vn_2), "f"(S[14]));
                S[14] = _fma_143;
                float _fma_144;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_144) : "f"(k_regs[15]), "f"(vn_2), "f"(S[15]));
                S[15] = _fma_144;
                float acc0_x_2 = 0.0f;
                float acc0_y_2 = 0.0f;
                float acc1_x_2 = 0.0f;
                float acc1_y_2 = 0.0f;
                float _fma_145;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_145) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_2));
                acc0_x_2 = _fma_145;
                float _fma_146;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_146) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_2));
                acc0_y_2 = _fma_146;
                float _fma_147;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_147) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_2));
                acc1_x_2 = _fma_147;
                float _fma_148;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_148) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_2));
                acc1_y_2 = _fma_148;
                float _fma_149;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_149) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_2));
                acc0_x_2 = _fma_149;
                float _fma_150;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_150) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_2));
                acc0_y_2 = _fma_150;
                float _fma_151;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_151) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_2));
                acc1_x_2 = _fma_151;
                float _fma_152;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_152) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_2));
                acc1_y_2 = _fma_152;
                float _fma_153;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_153) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_2));
                acc0_x_2 = _fma_153;
                float _fma_154;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_154) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_2));
                acc0_y_2 = _fma_154;
                float _fma_155;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_2));
                acc1_x_2 = _fma_155;
                float _fma_156;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_2));
                acc1_y_2 = _fma_156;
                float _fma_157;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_2));
                acc0_x_2 = _fma_157;
                float _fma_158;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_158) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_2));
                acc0_y_2 = _fma_158;
                float _fma_159;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_159) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_2));
                acc1_x_2 = _fma_159;
                float _fma_160;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_160) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_2));
                acc1_y_2 = _fma_160;
                float _add_161;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_161) : "f"(acc0_x_2), "f"(acc1_x_2));
                float _add_162;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_162) : "f"(acc0_y_2), "f"(acc1_y_2));
                float dot_2 = _add_161 + _add_162;
                float _shfl_xor_sync_163 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 1);
                dot_2 = dot_2 + _shfl_xor_sync_163;
                float _shfl_xor_sync_164 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 2);
                dot_2 = dot_2 + _shfl_xor_sync_164;
                float _shfl_xor_sync_165 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 4);
                dot_2 = dot_2 + _shfl_xor_sync_165;
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
                float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 1);
                dot_2 = dot_2 + _shfl_xor_sync_35;
                float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 2);
                dot_2 = dot_2 + _shfl_xor_sync_36;
                float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 4);
                dot_2 = dot_2 + _shfl_xor_sync_37;
#endif
                float vn_3 = (v_regs[1] - dot_2) * beta_1;
#if __CUDA_ARCH__ == 900
                float _fma_166;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_166) : "f"(k_regs[0]), "f"(vn_3), "f"(S[16]));
                S[16] = _fma_166;
                float _fma_167;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_167) : "f"(k_regs[1]), "f"(vn_3), "f"(S[17]));
                S[17] = _fma_167;
                float _fma_168;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_168) : "f"(k_regs[2]), "f"(vn_3), "f"(S[18]));
                S[18] = _fma_168;
                float _fma_169;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_169) : "f"(k_regs[3]), "f"(vn_3), "f"(S[19]));
                S[19] = _fma_169;
                float _fma_170;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_170) : "f"(k_regs[4]), "f"(vn_3), "f"(S[20]));
                S[20] = _fma_170;
                float _fma_171;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_171) : "f"(k_regs[5]), "f"(vn_3), "f"(S[21]));
                S[21] = _fma_171;
                float _fma_172;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_172) : "f"(k_regs[6]), "f"(vn_3), "f"(S[22]));
                S[22] = _fma_172;
                float _fma_173;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_173) : "f"(k_regs[7]), "f"(vn_3), "f"(S[23]));
                S[23] = _fma_173;
                float _fma_174;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_174) : "f"(k_regs[8]), "f"(vn_3), "f"(S[24]));
                S[24] = _fma_174;
                float _fma_175;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_175) : "f"(k_regs[9]), "f"(vn_3), "f"(S[25]));
                S[25] = _fma_175;
                float _fma_176;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_176) : "f"(k_regs[10]), "f"(vn_3), "f"(S[26]));
                S[26] = _fma_176;
                float _fma_177;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_177) : "f"(k_regs[11]), "f"(vn_3), "f"(S[27]));
                S[27] = _fma_177;
                float _fma_178;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_178) : "f"(k_regs[12]), "f"(vn_3), "f"(S[28]));
                S[28] = _fma_178;
                float _fma_179;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_179) : "f"(k_regs[13]), "f"(vn_3), "f"(S[29]));
                S[29] = _fma_179;
                float _fma_180;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_180) : "f"(k_regs[14]), "f"(vn_3), "f"(S[30]));
                S[30] = _fma_180;
                float _fma_181;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_181) : "f"(k_regs[15]), "f"(vn_3), "f"(S[31]));
                S[31] = _fma_181;
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
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_2 * 256) + group_byte));
                float beta_2 = sBeta[m_2];
#if __CUDA_ARCH__ == 900
                float acc0_x_3 = 0.0f;
                float acc0_y_3 = 0.0f;
                float acc1_x_3 = 0.0f;
                float acc1_y_3 = 0.0f;
                float _fma_182;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_182) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_3));
                acc0_x_3 = _fma_182;
                float _fma_183;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_183) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_3));
                acc0_y_3 = _fma_183;
                float _fma_184;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_184) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_3));
                acc1_x_3 = _fma_184;
                float _fma_185;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_185) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_3));
                acc1_y_3 = _fma_185;
                float _fma_186;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_186) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_3));
                acc0_x_3 = _fma_186;
                float _fma_187;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_187) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_3));
                acc0_y_3 = _fma_187;
                float _fma_188;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_188) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_3));
                acc1_x_3 = _fma_188;
                float _fma_189;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_189) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_3));
                acc1_y_3 = _fma_189;
                float _fma_190;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_190) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_3));
                acc0_x_3 = _fma_190;
                float _fma_191;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_191) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_3));
                acc0_y_3 = _fma_191;
                float _fma_192;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_192) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_3));
                acc1_x_3 = _fma_192;
                float _fma_193;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_193) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_3));
                acc1_y_3 = _fma_193;
                float _fma_194;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_194) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_3));
                acc0_x_3 = _fma_194;
                float _fma_195;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_195) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_3));
                acc0_y_3 = _fma_195;
                float _fma_196;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_196) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_3));
                acc1_x_3 = _fma_196;
                float _fma_197;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_197) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_3));
                acc1_y_3 = _fma_197;
                float _add_198;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_198) : "f"(acc0_x_3), "f"(acc1_x_3));
                float _add_199;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_199) : "f"(acc0_y_3), "f"(acc1_y_3));
                float dot_3 = _add_198 + _add_199;
                float _shfl_xor_sync_200 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 1);
                dot_3 = dot_3 + _shfl_xor_sync_200;
                float _shfl_xor_sync_201 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 2);
                dot_3 = dot_3 + _shfl_xor_sync_201;
                float _shfl_xor_sync_202 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 4);
                dot_3 = dot_3 + _shfl_xor_sync_202;
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
                float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 1);
                dot_3 = dot_3 + _shfl_xor_sync_38;
                float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 2);
                dot_3 = dot_3 + _shfl_xor_sync_39;
                float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 4);
                dot_3 = dot_3 + _shfl_xor_sync_40;
#endif
                float vn_4 = (v_regs[0] - dot_3) * beta_2;
#if __CUDA_ARCH__ == 900
                float _fma_203;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_203) : "f"(k_regs[0]), "f"(vn_4), "f"(S[0]));
                S[0] = _fma_203;
                float _fma_204;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_204) : "f"(k_regs[1]), "f"(vn_4), "f"(S[1]));
                S[1] = _fma_204;
                float _fma_205;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_205) : "f"(k_regs[2]), "f"(vn_4), "f"(S[2]));
                S[2] = _fma_205;
                float _fma_206;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_206) : "f"(k_regs[3]), "f"(vn_4), "f"(S[3]));
                S[3] = _fma_206;
                float _fma_207;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_207) : "f"(k_regs[4]), "f"(vn_4), "f"(S[4]));
                S[4] = _fma_207;
                float _fma_208;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_208) : "f"(k_regs[5]), "f"(vn_4), "f"(S[5]));
                S[5] = _fma_208;
                float _fma_209;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_209) : "f"(k_regs[6]), "f"(vn_4), "f"(S[6]));
                S[6] = _fma_209;
                float _fma_210;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_210) : "f"(k_regs[7]), "f"(vn_4), "f"(S[7]));
                S[7] = _fma_210;
                float _fma_211;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_211) : "f"(k_regs[8]), "f"(vn_4), "f"(S[8]));
                S[8] = _fma_211;
                float _fma_212;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_212) : "f"(k_regs[9]), "f"(vn_4), "f"(S[9]));
                S[9] = _fma_212;
                float _fma_213;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_213) : "f"(k_regs[10]), "f"(vn_4), "f"(S[10]));
                S[10] = _fma_213;
                float _fma_214;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_214) : "f"(k_regs[11]), "f"(vn_4), "f"(S[11]));
                S[11] = _fma_214;
                float _fma_215;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_215) : "f"(k_regs[12]), "f"(vn_4), "f"(S[12]));
                S[12] = _fma_215;
                float _fma_216;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_216) : "f"(k_regs[13]), "f"(vn_4), "f"(S[13]));
                S[13] = _fma_216;
                float _fma_217;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_217) : "f"(k_regs[14]), "f"(vn_4), "f"(S[14]));
                S[14] = _fma_217;
                float _fma_218;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_218) : "f"(k_regs[15]), "f"(vn_4), "f"(S[15]));
                S[15] = _fma_218;
                float acc0_x_4 = 0.0f;
                float acc0_y_4 = 0.0f;
                float acc1_x_4 = 0.0f;
                float acc1_y_4 = 0.0f;
                float _fma_219;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_219) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_4));
                acc0_x_4 = _fma_219;
                float _fma_220;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_220) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_4));
                acc0_y_4 = _fma_220;
                float _fma_221;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_221) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_4));
                acc1_x_4 = _fma_221;
                float _fma_222;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_222) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_4));
                acc1_y_4 = _fma_222;
                float _fma_223;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_223) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_4));
                acc0_x_4 = _fma_223;
                float _fma_224;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_224) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_4));
                acc0_y_4 = _fma_224;
                float _fma_225;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_225) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_4));
                acc1_x_4 = _fma_225;
                float _fma_226;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_226) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_4));
                acc1_y_4 = _fma_226;
                float _fma_227;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_227) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_4));
                acc0_x_4 = _fma_227;
                float _fma_228;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_228) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_4));
                acc0_y_4 = _fma_228;
                float _fma_229;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_229) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_4));
                acc1_x_4 = _fma_229;
                float _fma_230;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_230) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_4));
                acc1_y_4 = _fma_230;
                float _fma_231;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_231) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_4));
                acc0_x_4 = _fma_231;
                float _fma_232;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_232) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_4));
                acc0_y_4 = _fma_232;
                float _fma_233;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_233) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_4));
                acc1_x_4 = _fma_233;
                float _fma_234;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_234) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_4));
                acc1_y_4 = _fma_234;
                float _add_235;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_235) : "f"(acc0_x_4), "f"(acc1_x_4));
                float _add_236;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_236) : "f"(acc0_y_4), "f"(acc1_y_4));
                float dot_4 = _add_235 + _add_236;
                float _shfl_xor_sync_237 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 1);
                dot_4 = dot_4 + _shfl_xor_sync_237;
                float _shfl_xor_sync_238 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 2);
                dot_4 = dot_4 + _shfl_xor_sync_238;
                float _shfl_xor_sync_239 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 4);
                dot_4 = dot_4 + _shfl_xor_sync_239;
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
                float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 1);
                dot_4 = dot_4 + _shfl_xor_sync_41;
                float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 2);
                dot_4 = dot_4 + _shfl_xor_sync_42;
                float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 4);
                dot_4 = dot_4 + _shfl_xor_sync_43;
#endif
                float vn_5 = (v_regs[1] - dot_4) * beta_2;
#if __CUDA_ARCH__ == 900
                float _fma_240;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_240) : "f"(k_regs[0]), "f"(vn_5), "f"(S[16]));
                S[16] = _fma_240;
                float _fma_241;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_241) : "f"(k_regs[1]), "f"(vn_5), "f"(S[17]));
                S[17] = _fma_241;
                float _fma_242;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_242) : "f"(k_regs[2]), "f"(vn_5), "f"(S[18]));
                S[18] = _fma_242;
                float _fma_243;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_243) : "f"(k_regs[3]), "f"(vn_5), "f"(S[19]));
                S[19] = _fma_243;
                float _fma_244;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_244) : "f"(k_regs[4]), "f"(vn_5), "f"(S[20]));
                S[20] = _fma_244;
                float _fma_245;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_245) : "f"(k_regs[5]), "f"(vn_5), "f"(S[21]));
                S[21] = _fma_245;
                float _fma_246;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_246) : "f"(k_regs[6]), "f"(vn_5), "f"(S[22]));
                S[22] = _fma_246;
                float _fma_247;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_247) : "f"(k_regs[7]), "f"(vn_5), "f"(S[23]));
                S[23] = _fma_247;
                float _fma_248;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_248) : "f"(k_regs[8]), "f"(vn_5), "f"(S[24]));
                S[24] = _fma_248;
                float _fma_249;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_249) : "f"(k_regs[9]), "f"(vn_5), "f"(S[25]));
                S[25] = _fma_249;
                float _fma_250;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_250) : "f"(k_regs[10]), "f"(vn_5), "f"(S[26]));
                S[26] = _fma_250;
                float _fma_251;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_251) : "f"(k_regs[11]), "f"(vn_5), "f"(S[27]));
                S[27] = _fma_251;
                float _fma_252;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_252) : "f"(k_regs[12]), "f"(vn_5), "f"(S[28]));
                S[28] = _fma_252;
                float _fma_253;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_253) : "f"(k_regs[13]), "f"(vn_5), "f"(S[29]));
                S[29] = _fma_253;
                float _fma_254;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_254) : "f"(k_regs[14]), "f"(vn_5), "f"(S[30]));
                S[30] = _fma_254;
                float _fma_255;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_255) : "f"(k_regs[15]), "f"(vn_5), "f"(S[31]));
                S[31] = _fma_255;
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
                float acc0_x_5 = 0.0f;
                float acc0_y_5 = 0.0f;
                float acc1_x_5 = 0.0f;
                float acc1_y_5 = 0.0f;
                float _fma_256;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_256) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_5));
                acc0_x_5 = _fma_256;
                float _fma_257;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_257) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_5));
                acc0_y_5 = _fma_257;
                float _fma_258;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_258) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_5));
                acc1_x_5 = _fma_258;
                float _fma_259;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_259) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_5));
                acc1_y_5 = _fma_259;
                float _fma_260;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_260) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_5));
                acc0_x_5 = _fma_260;
                float _fma_261;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_261) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_5));
                acc0_y_5 = _fma_261;
                float _fma_262;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_262) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_5));
                acc1_x_5 = _fma_262;
                float _fma_263;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_263) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_5));
                acc1_y_5 = _fma_263;
                float _fma_264;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_264) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_5));
                acc0_x_5 = _fma_264;
                float _fma_265;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_265) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_5));
                acc0_y_5 = _fma_265;
                float _fma_266;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_266) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_5));
                acc1_x_5 = _fma_266;
                float _fma_267;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_267) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_5));
                acc1_y_5 = _fma_267;
                float _fma_268;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_268) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_5));
                acc0_x_5 = _fma_268;
                float _fma_269;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_269) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_5));
                acc0_y_5 = _fma_269;
                float _fma_270;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_270) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_5));
                acc1_x_5 = _fma_270;
                float _fma_271;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_271) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_5));
                acc1_y_5 = _fma_271;
                float _add_272;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_272) : "f"(acc0_x_5), "f"(acc1_x_5));
                float _add_273;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_273) : "f"(acc0_y_5), "f"(acc1_y_5));
                float dot_5 = _add_272 + _add_273;
                float _shfl_xor_sync_274 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 1);
                dot_5 = dot_5 + _shfl_xor_sync_274;
                float _shfl_xor_sync_275 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 2);
                dot_5 = dot_5 + _shfl_xor_sync_275;
                float _shfl_xor_sync_276 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 4);
                dot_5 = dot_5 + _shfl_xor_sync_276;
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
                float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 1);
                dot_5 = dot_5 + _shfl_xor_sync_44;
                float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 2);
                dot_5 = dot_5 + _shfl_xor_sync_45;
                float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 4);
                dot_5 = dot_5 + _shfl_xor_sync_46;
#endif
                o4[0] = dot_5;
#if __CUDA_ARCH__ == 900
                float acc0_x_6 = 0.0f;
                float acc0_y_6 = 0.0f;
                float acc1_x_6 = 0.0f;
                float acc1_y_6 = 0.0f;
                float _fma_277;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_277) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_6));
                acc0_x_6 = _fma_277;
                float _fma_278;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_278) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_6));
                acc0_y_6 = _fma_278;
                float _fma_279;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_279) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_6));
                acc1_x_6 = _fma_279;
                float _fma_280;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_280) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_6));
                acc1_y_6 = _fma_280;
                float _fma_281;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_281) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_6));
                acc0_x_6 = _fma_281;
                float _fma_282;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_282) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_6));
                acc0_y_6 = _fma_282;
                float _fma_283;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_283) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_6));
                acc1_x_6 = _fma_283;
                float _fma_284;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_284) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_6));
                acc1_y_6 = _fma_284;
                float _fma_285;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_285) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_6));
                acc0_x_6 = _fma_285;
                float _fma_286;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_286) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_6));
                acc0_y_6 = _fma_286;
                float _fma_287;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_287) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_6));
                acc1_x_6 = _fma_287;
                float _fma_288;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_288) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_6));
                acc1_y_6 = _fma_288;
                float _fma_289;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_289) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_6));
                acc0_x_6 = _fma_289;
                float _fma_290;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_290) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_6));
                acc0_y_6 = _fma_290;
                float _fma_291;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_291) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_6));
                acc1_x_6 = _fma_291;
                float _fma_292;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_292) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_6));
                acc1_y_6 = _fma_292;
                float _add_293;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_293) : "f"(acc0_x_6), "f"(acc1_x_6));
                float _add_294;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_294) : "f"(acc0_y_6), "f"(acc1_y_6));
                float dot_6 = _add_293 + _add_294;
                float _shfl_xor_sync_295 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 1);
                dot_6 = dot_6 + _shfl_xor_sync_295;
                float _shfl_xor_sync_296 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 2);
                dot_6 = dot_6 + _shfl_xor_sync_296;
                float _shfl_xor_sync_297 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 4);
                dot_6 = dot_6 + _shfl_xor_sync_297;
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
                float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 1);
                dot_6 = dot_6 + _shfl_xor_sync_47;
                float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 2);
                dot_6 = dot_6 + _shfl_xor_sync_48;
                float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 4);
                dot_6 = dot_6 + _shfl_xor_sync_49;
#endif
                o4[1] = dot_6;
                if (u == 0) {
                    long long out_index = (long long)n * out_stride_p0 + (long long)t * out_stride_p1 + (long long)h * out_stride_p2;
                    unsigned int packed[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o4[_lp*2 + 0], o4[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(out + (out_index + (long long)row0))[0] = reinterpret_cast<unsigned*>(packed)[0];
                }
                int ssm_slot = sSsm[t];
                int snap_valid = ((use_ssm_state_indices != 0 && ssm_slot >= 0 && state_pool_size > (long long)ssm_slot) ? 1 : 0);
                int safe_slot = ((snap_valid != 0) ? ssm_slot : 0);
                long long snap_base = (long long)safe_slot * state_stride_p0 + head_rows_off;
                if (philox_rounds != 0) {
#if __CUDA_ARCH__ == 900
                    bool nan = 0;
                    uint32_t _setp_298_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_298_bits) : "f"(S[0]), "f"(S[1]));
                    bool _setp_298 = _setp_298_bits != 0;
                    nan = nan || _setp_298;
                    uint32_t _setp_299_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_299_bits) : "f"(S[2]), "f"(S[3]));
                    bool _setp_299 = _setp_299_bits != 0;
                    nan = nan || _setp_299;
                    uint32_t _setp_300_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_300_bits) : "f"(S[4]), "f"(S[5]));
                    bool _setp_300 = _setp_300_bits != 0;
                    nan = nan || _setp_300;
                    uint32_t _setp_301_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_301_bits) : "f"(S[6]), "f"(S[7]));
                    bool _setp_301 = _setp_301_bits != 0;
                    nan = nan || _setp_301;
                    uint32_t _setp_302_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_302_bits) : "f"(S[8]), "f"(S[9]));
                    bool _setp_302 = _setp_302_bits != 0;
                    nan = nan || _setp_302;
                    uint32_t _setp_303_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_303_bits) : "f"(S[10]), "f"(S[11]));
                    bool _setp_303 = _setp_303_bits != 0;
                    nan = nan || _setp_303;
                    uint32_t _setp_304_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_304_bits) : "f"(S[12]), "f"(S[13]));
                    bool _setp_304 = _setp_304_bits != 0;
                    nan = nan || _setp_304;
                    uint32_t _setp_305_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_305_bits) : "f"(S[14]), "f"(S[15]));
                    bool _setp_305 = _setp_305_bits != 0;
                    nan = nan || _setp_305;
                    uint32_t _setp_306_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_306_bits) : "f"(S[16]), "f"(S[17]));
                    bool _setp_306 = _setp_306_bits != 0;
                    nan = nan || _setp_306;
                    uint32_t _setp_307_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_307_bits) : "f"(S[18]), "f"(S[19]));
                    bool _setp_307 = _setp_307_bits != 0;
                    nan = nan || _setp_307;
                    uint32_t _setp_308_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_308_bits) : "f"(S[20]), "f"(S[21]));
                    bool _setp_308 = _setp_308_bits != 0;
                    nan = nan || _setp_308;
                    uint32_t _setp_309_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_309_bits) : "f"(S[22]), "f"(S[23]));
                    bool _setp_309 = _setp_309_bits != 0;
                    nan = nan || _setp_309;
                    uint32_t _setp_310_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_310_bits) : "f"(S[24]), "f"(S[25]));
                    bool _setp_310 = _setp_310_bits != 0;
                    nan = nan || _setp_310;
                    uint32_t _setp_311_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_311_bits) : "f"(S[26]), "f"(S[27]));
                    bool _setp_311 = _setp_311_bits != 0;
                    nan = nan || _setp_311;
                    uint32_t _setp_312_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_312_bits) : "f"(S[28]), "f"(S[29]));
                    bool _setp_312 = _setp_312_bits != 0;
                    nan = nan || _setp_312;
                    uint32_t _setp_313_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_313_bits) : "f"(S[30]), "f"(S[31]));
                    bool _setp_313 = _setp_313_bits != 0;
                    nan = nan || _setp_313;
                    int _warp_vote_314 = __any_sync(__activemask(), nan);
                    if (__builtin_expect(_warp_vote_314 == 0, 1)) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                            : "r"(noise_lane + 0));
                        unsigned int slo = __as_u32(mul_f32_rz_noftz(S[0], __uint_as_float(125829120)));
                        unsigned int shi = __as_u32(mul_f32_rz_noftz(S[1], __uint_as_float(125829120)));
                        unsigned int tlo = slo + (h_noise[0] & 8191) & 4294959104u;
                        unsigned int thi = shi + ((h_noise[0] & 536805376) >> 16) & 4294959104u;
                        float flo;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo) : "f"(__uint_as_float(tlo)), "f"(__uint_as_float(2004877312)));
                        float fhi;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi) : "f"(__uint_as_float(thi)), "f"(__uint_as_float(2004877312)));
                        uint32_t word;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word) : "f"(fhi), "f"(flo));
                        h_words[0] = word;
                        unsigned int slo_0 = __as_u32(mul_f32_rz_noftz(S[2], __uint_as_float(125829120)));
                        unsigned int shi_0 = __as_u32(mul_f32_rz_noftz(S[3], __uint_as_float(125829120)));
                        unsigned int tlo_0 = slo_0 + (h_noise[1] & 8191) & 4294959104u;
                        unsigned int thi_0 = shi_0 + ((h_noise[1] & 536805376) >> 16) & 4294959104u;
                        float flo_0;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_0) : "f"(__uint_as_float(tlo_0)), "f"(__uint_as_float(2004877312)));
                        float fhi_0;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_0) : "f"(__uint_as_float(thi_0)), "f"(__uint_as_float(2004877312)));
                        uint32_t word_0;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_0) : "f"(fhi_0), "f"(flo_0));
                        h_words[1] = word_0;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        unsigned int slo_1 = __as_u32(mul_f32_rz_noftz(S[8], __uint_as_float(125829120)));
                        unsigned int shi_1 = __as_u32(mul_f32_rz_noftz(S[9], __uint_as_float(125829120)));
                        unsigned int tlo_1 = slo_1 + (h_noise[2] & 8191) & 4294959104u;
                        unsigned int thi_1 = shi_1 + ((h_noise[2] & 536805376) >> 16) & 4294959104u;
                        float flo_1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_1) : "f"(__uint_as_float(tlo_1)), "f"(__uint_as_float(2004877312)));
                        float fhi_1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_1) : "f"(__uint_as_float(thi_1)), "f"(__uint_as_float(2004877312)));
                        uint32_t word_1;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_1) : "f"(fhi_1), "f"(flo_1));
                        h_words[0] = word_1;
                        unsigned int slo_2 = __as_u32(mul_f32_rz_noftz(S[10], __uint_as_float(125829120)));
                        unsigned int shi_2 = __as_u32(mul_f32_rz_noftz(S[11], __uint_as_float(125829120)));
                        unsigned int tlo_2 = slo_2 + (h_noise[3] & 8191) & 4294959104u;
                        unsigned int thi_2 = shi_2 + ((h_noise[3] & 536805376) >> 16) & 4294959104u;
                        float flo_2;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_2) : "f"(__uint_as_float(tlo_2)), "f"(__uint_as_float(2004877312)));
                        float fhi_2;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_2) : "f"(__uint_as_float(thi_2)), "f"(__uint_as_float(2004877312)));
                        uint32_t word_2;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_2) : "f"(fhi_2), "f"(flo_2));
                        h_words[1] = word_2;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                            : "r"(noise_lane + 4096));
                        unsigned int slo_3 = __as_u32(mul_f32_rz_noftz(S[4], __uint_as_float(125829120)));
                        unsigned int shi_3 = __as_u32(mul_f32_rz_noftz(S[5], __uint_as_float(125829120)));
                        unsigned int tlo_3 = slo_3 + (h_noise[0] & 8191) & 4294959104u;
                        unsigned int thi_3 = shi_3 + ((h_noise[0] & 536805376) >> 16) & 4294959104u;
                        float flo_3;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_3) : "f"(__uint_as_float(tlo_3)), "f"(__uint_as_float(2004877312)));
                        float fhi_3;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_3) : "f"(__uint_as_float(thi_3)), "f"(__uint_as_float(2004877312)));
                        uint32_t word_3;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_3) : "f"(fhi_3), "f"(flo_3));
                        h_words[0] = word_3;
                        unsigned int slo_4 = __as_u32(mul_f32_rz_noftz(S[6], __uint_as_float(125829120)));
                        unsigned int shi_4 = __as_u32(mul_f32_rz_noftz(S[7], __uint_as_float(125829120)));
                        unsigned int tlo_4 = slo_4 + (h_noise[1] & 8191) & 4294959104u;
                        unsigned int thi_4 = shi_4 + ((h_noise[1] & 536805376) >> 16) & 4294959104u;
                        float flo_4;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_4) : "f"(__uint_as_float(tlo_4)), "f"(__uint_as_float(2004877312)));
                        float fhi_4;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_4) : "f"(__uint_as_float(thi_4)), "f"(__uint_as_float(2004877312)));
                        uint32_t word_4;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_4) : "f"(fhi_4), "f"(flo_4));
                        h_words[1] = word_4;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        unsigned int slo_5 = __as_u32(mul_f32_rz_noftz(S[12], __uint_as_float(125829120)));
                        unsigned int shi_5 = __as_u32(mul_f32_rz_noftz(S[13], __uint_as_float(125829120)));
                        unsigned int tlo_5 = slo_5 + (h_noise[2] & 8191) & 4294959104u;
                        unsigned int thi_5 = shi_5 + ((h_noise[2] & 536805376) >> 16) & 4294959104u;
                        float flo_5;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_5) : "f"(__uint_as_float(tlo_5)), "f"(__uint_as_float(2004877312)));
                        float fhi_5;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_5) : "f"(__uint_as_float(thi_5)), "f"(__uint_as_float(2004877312)));
                        uint32_t word_5;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_5) : "f"(fhi_5), "f"(flo_5));
                        h_words[0] = word_5;
                        unsigned int slo_6 = __as_u32(mul_f32_rz_noftz(S[14], __uint_as_float(125829120)));
                        unsigned int shi_6 = __as_u32(mul_f32_rz_noftz(S[15], __uint_as_float(125829120)));
                        unsigned int tlo_6 = slo_6 + (h_noise[3] & 8191) & 4294959104u;
                        unsigned int thi_6 = shi_6 + ((h_noise[3] & 536805376) >> 16) & 4294959104u;
                        float flo_6;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_6) : "f"(__uint_as_float(tlo_6)), "f"(__uint_as_float(2004877312)));
                        float fhi_6;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_6) : "f"(__uint_as_float(thi_6)), "f"(__uint_as_float(2004877312)));
                        uint32_t word_6;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_6) : "f"(fhi_6), "f"(flo_6));
                        h_words[1] = word_6;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                            : "r"(noise_lane + 8192));
                        unsigned int slo_7 = __as_u32(mul_f32_rz_noftz(S[16], __uint_as_float(125829120)));
                        unsigned int shi_7 = __as_u32(mul_f32_rz_noftz(S[17], __uint_as_float(125829120)));
                        unsigned int tlo_7 = slo_7 + (h_noise[0] & 8191) & 4294959104u;
                        unsigned int thi_7 = shi_7 + ((h_noise[0] & 536805376) >> 16) & 4294959104u;
                        float flo_7;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_7) : "f"(__uint_as_float(tlo_7)), "f"(__uint_as_float(2004877312)));
                        float fhi_7;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_7) : "f"(__uint_as_float(thi_7)), "f"(__uint_as_float(2004877312)));
                        uint32_t word_7;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_7) : "f"(fhi_7), "f"(flo_7));
                        h_words[0] = word_7;
                        unsigned int slo_8 = __as_u32(mul_f32_rz_noftz(S[18], __uint_as_float(125829120)));
                        unsigned int shi_8 = __as_u32(mul_f32_rz_noftz(S[19], __uint_as_float(125829120)));
                        unsigned int tlo_8 = slo_8 + (h_noise[1] & 8191) & 4294959104u;
                        unsigned int thi_8 = shi_8 + ((h_noise[1] & 536805376) >> 16) & 4294959104u;
                        float flo_8;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_8) : "f"(__uint_as_float(tlo_8)), "f"(__uint_as_float(2004877312)));
                        float fhi_8;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_8) : "f"(__uint_as_float(thi_8)), "f"(__uint_as_float(2004877312)));
                        uint32_t word_8;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_8) : "f"(fhi_8), "f"(flo_8));
                        h_words[1] = word_8;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        unsigned int slo_9 = __as_u32(mul_f32_rz_noftz(S[24], __uint_as_float(125829120)));
                        unsigned int shi_9 = __as_u32(mul_f32_rz_noftz(S[25], __uint_as_float(125829120)));
                        unsigned int tlo_9 = slo_9 + (h_noise[2] & 8191) & 4294959104u;
                        unsigned int thi_9 = shi_9 + ((h_noise[2] & 536805376) >> 16) & 4294959104u;
                        float flo_9;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_9) : "f"(__uint_as_float(tlo_9)), "f"(__uint_as_float(2004877312)));
                        float fhi_9;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_9) : "f"(__uint_as_float(thi_9)), "f"(__uint_as_float(2004877312)));
                        uint32_t word_9;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_9) : "f"(fhi_9), "f"(flo_9));
                        h_words[0] = word_9;
                        unsigned int slo_10 = __as_u32(mul_f32_rz_noftz(S[26], __uint_as_float(125829120)));
                        unsigned int shi_10 = __as_u32(mul_f32_rz_noftz(S[27], __uint_as_float(125829120)));
                        unsigned int tlo_10 = slo_10 + (h_noise[3] & 8191) & 4294959104u;
                        unsigned int thi_10 = shi_10 + ((h_noise[3] & 536805376) >> 16) & 4294959104u;
                        float flo_10;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_10) : "f"(__uint_as_float(tlo_10)), "f"(__uint_as_float(2004877312)));
                        float fhi_10;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_10) : "f"(__uint_as_float(thi_10)), "f"(__uint_as_float(2004877312)));
                        uint32_t word_10;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_10) : "f"(fhi_10), "f"(flo_10));
                        h_words[1] = word_10;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                            : "r"(noise_lane + 12288));
                        unsigned int slo_11 = __as_u32(mul_f32_rz_noftz(S[20], __uint_as_float(125829120)));
                        unsigned int shi_11 = __as_u32(mul_f32_rz_noftz(S[21], __uint_as_float(125829120)));
                        unsigned int tlo_11 = slo_11 + (h_noise[0] & 8191) & 4294959104u;
                        unsigned int thi_11 = shi_11 + ((h_noise[0] & 536805376) >> 16) & 4294959104u;
                        float flo_11;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_11) : "f"(__uint_as_float(tlo_11)), "f"(__uint_as_float(2004877312)));
                        float fhi_11;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_11) : "f"(__uint_as_float(thi_11)), "f"(__uint_as_float(2004877312)));
                        uint32_t word_11;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_11) : "f"(fhi_11), "f"(flo_11));
                        h_words[0] = word_11;
                        unsigned int slo_12 = __as_u32(mul_f32_rz_noftz(S[22], __uint_as_float(125829120)));
                        unsigned int shi_12 = __as_u32(mul_f32_rz_noftz(S[23], __uint_as_float(125829120)));
                        unsigned int tlo_12 = slo_12 + (h_noise[1] & 8191) & 4294959104u;
                        unsigned int thi_12 = shi_12 + ((h_noise[1] & 536805376) >> 16) & 4294959104u;
                        float flo_12;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_12) : "f"(__uint_as_float(tlo_12)), "f"(__uint_as_float(2004877312)));
                        float fhi_12;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_12) : "f"(__uint_as_float(thi_12)), "f"(__uint_as_float(2004877312)));
                        uint32_t word_12;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_12) : "f"(fhi_12), "f"(flo_12));
                        h_words[1] = word_12;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        unsigned int slo_13 = __as_u32(mul_f32_rz_noftz(S[28], __uint_as_float(125829120)));
                        unsigned int shi_13 = __as_u32(mul_f32_rz_noftz(S[29], __uint_as_float(125829120)));
                        unsigned int tlo_13 = slo_13 + (h_noise[2] & 8191) & 4294959104u;
                        unsigned int thi_13 = shi_13 + ((h_noise[2] & 536805376) >> 16) & 4294959104u;
                        float flo_13;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_13) : "f"(__uint_as_float(tlo_13)), "f"(__uint_as_float(2004877312)));
                        float fhi_13;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_13) : "f"(__uint_as_float(thi_13)), "f"(__uint_as_float(2004877312)));
                        uint32_t word_13;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_13) : "f"(fhi_13), "f"(flo_13));
                        h_words[0] = word_13;
                        unsigned int slo_14 = __as_u32(mul_f32_rz_noftz(S[30], __uint_as_float(125829120)));
                        unsigned int shi_14 = __as_u32(mul_f32_rz_noftz(S[31], __uint_as_float(125829120)));
                        unsigned int tlo_14 = slo_14 + (h_noise[3] & 8191) & 4294959104u;
                        unsigned int thi_14 = shi_14 + ((h_noise[3] & 536805376) >> 16) & 4294959104u;
                        float flo_14;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_14) : "f"(__uint_as_float(tlo_14)), "f"(__uint_as_float(2004877312)));
                        float fhi_14;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_14) : "f"(__uint_as_float(thi_14)), "f"(__uint_as_float(2004877312)));
                        uint32_t word_14;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_14) : "f"(fhi_14), "f"(flo_14));
                        h_words[1] = word_14;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    } else {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                            : "r"(noise_lane + 0));
                        float _abs_315;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_315) : "f"(S[0]));
                        unsigned int s = __as_u32(mul_f32_rz_noftz(_abs_315, __uint_as_float(125829120)));
                        unsigned int _min_316 = ((s) < (260046848) ? (s) : (260046848));
                        unsigned int h_0 = _min_316 + (h_noise[0] & 8191) >> 13;
                        unsigned int mag = ((s <= 2139095040) ? h_0 : 32256);
                        float _abs_317;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_317) : "f"(S[1]));
                        unsigned int s_0 = __as_u32(mul_f32_rz_noftz(_abs_317, __uint_as_float(125829120)));
                        unsigned int _min_318 = ((s_0) < (260046848) ? (s_0) : (260046848));
                        unsigned int h_1 = _min_318 + ((h_noise[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_0 = ((s_0 <= 2139095040) ? h_1 : 32256);
                        uint32_t mags;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags) : "r"(mag), "r"(mag_0));
                        uint32_t signs;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs) : "r"(__as_u32(S[0])), "r"(__as_u32(S[1])));
                        unsigned int word_15 = signs & 2147516416u | mags;
                        h_words[0] = word_15;
                        float _abs_319;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_319) : "f"(S[2]));
                        unsigned int s_1 = __as_u32(mul_f32_rz_noftz(_abs_319, __uint_as_float(125829120)));
                        unsigned int _min_320 = ((s_1) < (260046848) ? (s_1) : (260046848));
                        unsigned int h_2 = _min_320 + (h_noise[1] & 8191) >> 13;
                        unsigned int mag_1 = ((s_1 <= 2139095040) ? h_2 : 32256);
                        float _abs_321;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_321) : "f"(S[3]));
                        unsigned int s_2 = __as_u32(mul_f32_rz_noftz(_abs_321, __uint_as_float(125829120)));
                        unsigned int _min_322 = ((s_2) < (260046848) ? (s_2) : (260046848));
                        unsigned int h_3 = _min_322 + ((h_noise[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_2 = ((s_2 <= 2139095040) ? h_3 : 32256);
                        uint32_t mags_0;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_0) : "r"(mag_1), "r"(mag_2));
                        uint32_t signs_0;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_0) : "r"(__as_u32(S[2])), "r"(__as_u32(S[3])));
                        unsigned int word_15_1 = signs_0 & 2147516416u | mags_0;
                        h_words[1] = word_15_1;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        float _abs_323;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_323) : "f"(S[8]));
                        unsigned int s_3 = __as_u32(mul_f32_rz_noftz(_abs_323, __uint_as_float(125829120)));
                        unsigned int _min_324 = ((s_3) < (260046848) ? (s_3) : (260046848));
                        unsigned int h_4 = _min_324 + (h_noise[2] & 8191) >> 13;
                        unsigned int mag_3 = ((s_3 <= 2139095040) ? h_4 : 32256);
                        float _abs_325;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_325) : "f"(S[9]));
                        unsigned int s_4 = __as_u32(mul_f32_rz_noftz(_abs_325, __uint_as_float(125829120)));
                        unsigned int _min_326 = ((s_4) < (260046848) ? (s_4) : (260046848));
                        unsigned int h_5 = _min_326 + ((h_noise[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_4 = ((s_4 <= 2139095040) ? h_5 : 32256);
                        uint32_t mags_1;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_1) : "r"(mag_3), "r"(mag_4));
                        uint32_t signs_1;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_1) : "r"(__as_u32(S[8])), "r"(__as_u32(S[9])));
                        unsigned int word_16 = signs_1 & 2147516416u | mags_1;
                        h_words[0] = word_16;
                        float _abs_327;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_327) : "f"(S[10]));
                        unsigned int s_5 = __as_u32(mul_f32_rz_noftz(_abs_327, __uint_as_float(125829120)));
                        unsigned int _min_328 = ((s_5) < (260046848) ? (s_5) : (260046848));
                        unsigned int h_6 = _min_328 + (h_noise[3] & 8191) >> 13;
                        unsigned int mag_5 = ((s_5 <= 2139095040) ? h_6 : 32256);
                        float _abs_329;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_329) : "f"(S[11]));
                        unsigned int s_6 = __as_u32(mul_f32_rz_noftz(_abs_329, __uint_as_float(125829120)));
                        unsigned int _min_330 = ((s_6) < (260046848) ? (s_6) : (260046848));
                        unsigned int h_7 = _min_330 + ((h_noise[3] & 536805376) >> 16) >> 13;
                        unsigned int mag_6 = ((s_6 <= 2139095040) ? h_7 : 32256);
                        uint32_t mags_2;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_2) : "r"(mag_5), "r"(mag_6));
                        uint32_t signs_2;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_2) : "r"(__as_u32(S[10])), "r"(__as_u32(S[11])));
                        unsigned int word_17 = signs_2 & 2147516416u | mags_2;
                        h_words[1] = word_17;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                            : "r"(noise_lane + 4096));
                        float _abs_331;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_331) : "f"(S[4]));
                        unsigned int s_7 = __as_u32(mul_f32_rz_noftz(_abs_331, __uint_as_float(125829120)));
                        unsigned int _min_332 = ((s_7) < (260046848) ? (s_7) : (260046848));
                        unsigned int h_8 = _min_332 + (h_noise[0] & 8191) >> 13;
                        unsigned int mag_7 = ((s_7 <= 2139095040) ? h_8 : 32256);
                        float _abs_333;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_333) : "f"(S[5]));
                        unsigned int s_8 = __as_u32(mul_f32_rz_noftz(_abs_333, __uint_as_float(125829120)));
                        unsigned int _min_334 = ((s_8) < (260046848) ? (s_8) : (260046848));
                        unsigned int h_9 = _min_334 + ((h_noise[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_8 = ((s_8 <= 2139095040) ? h_9 : 32256);
                        uint32_t mags_3;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_3) : "r"(mag_7), "r"(mag_8));
                        uint32_t signs_3;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_3) : "r"(__as_u32(S[4])), "r"(__as_u32(S[5])));
                        unsigned int word_18 = signs_3 & 2147516416u | mags_3;
                        h_words[0] = word_18;
                        float _abs_335;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_335) : "f"(S[6]));
                        unsigned int s_9 = __as_u32(mul_f32_rz_noftz(_abs_335, __uint_as_float(125829120)));
                        unsigned int _min_336 = ((s_9) < (260046848) ? (s_9) : (260046848));
                        unsigned int h_10 = _min_336 + (h_noise[1] & 8191) >> 13;
                        unsigned int mag_9 = ((s_9 <= 2139095040) ? h_10 : 32256);
                        float _abs_337;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_337) : "f"(S[7]));
                        unsigned int s_10 = __as_u32(mul_f32_rz_noftz(_abs_337, __uint_as_float(125829120)));
                        unsigned int _min_338 = ((s_10) < (260046848) ? (s_10) : (260046848));
                        unsigned int h_11 = _min_338 + ((h_noise[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_10 = ((s_10 <= 2139095040) ? h_11 : 32256);
                        uint32_t mags_4;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_4) : "r"(mag_9), "r"(mag_10));
                        uint32_t signs_4;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_4) : "r"(__as_u32(S[6])), "r"(__as_u32(S[7])));
                        unsigned int word_19 = signs_4 & 2147516416u | mags_4;
                        h_words[1] = word_19;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        float _abs_339;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_339) : "f"(S[12]));
                        unsigned int s_11 = __as_u32(mul_f32_rz_noftz(_abs_339, __uint_as_float(125829120)));
                        unsigned int _min_340 = ((s_11) < (260046848) ? (s_11) : (260046848));
                        unsigned int h_12 = _min_340 + (h_noise[2] & 8191) >> 13;
                        unsigned int mag_11 = ((s_11 <= 2139095040) ? h_12 : 32256);
                        float _abs_341;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_341) : "f"(S[13]));
                        unsigned int s_12 = __as_u32(mul_f32_rz_noftz(_abs_341, __uint_as_float(125829120)));
                        unsigned int _min_342 = ((s_12) < (260046848) ? (s_12) : (260046848));
                        unsigned int h_13 = _min_342 + ((h_noise[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_12 = ((s_12 <= 2139095040) ? h_13 : 32256);
                        uint32_t mags_5;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_5) : "r"(mag_11), "r"(mag_12));
                        uint32_t signs_5;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_5) : "r"(__as_u32(S[12])), "r"(__as_u32(S[13])));
                        unsigned int word_20 = signs_5 & 2147516416u | mags_5;
                        h_words[0] = word_20;
                        float _abs_343;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_343) : "f"(S[14]));
                        unsigned int s_13 = __as_u32(mul_f32_rz_noftz(_abs_343, __uint_as_float(125829120)));
                        unsigned int _min_344 = ((s_13) < (260046848) ? (s_13) : (260046848));
                        unsigned int h_14 = _min_344 + (h_noise[3] & 8191) >> 13;
                        unsigned int mag_13 = ((s_13 <= 2139095040) ? h_14 : 32256);
                        float _abs_345;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_345) : "f"(S[15]));
                        unsigned int s_14 = __as_u32(mul_f32_rz_noftz(_abs_345, __uint_as_float(125829120)));
                        unsigned int _min_346 = ((s_14) < (260046848) ? (s_14) : (260046848));
                        unsigned int h_15 = _min_346 + ((h_noise[3] & 536805376) >> 16) >> 13;
                        unsigned int mag_14 = ((s_14 <= 2139095040) ? h_15 : 32256);
                        uint32_t mags_6;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_6) : "r"(mag_13), "r"(mag_14));
                        uint32_t signs_6;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_6) : "r"(__as_u32(S[14])), "r"(__as_u32(S[15])));
                        unsigned int word_21 = signs_6 & 2147516416u | mags_6;
                        h_words[1] = word_21;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                            : "r"(noise_lane + 8192));
                        float _abs_347;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_347) : "f"(S[16]));
                        unsigned int s_15 = __as_u32(mul_f32_rz_noftz(_abs_347, __uint_as_float(125829120)));
                        unsigned int _min_348 = ((s_15) < (260046848) ? (s_15) : (260046848));
                        unsigned int h_16 = _min_348 + (h_noise[0] & 8191) >> 13;
                        unsigned int mag_15 = ((s_15 <= 2139095040) ? h_16 : 32256);
                        float _abs_349;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_349) : "f"(S[17]));
                        unsigned int s_16 = __as_u32(mul_f32_rz_noftz(_abs_349, __uint_as_float(125829120)));
                        unsigned int _min_350 = ((s_16) < (260046848) ? (s_16) : (260046848));
                        unsigned int h_17 = _min_350 + ((h_noise[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_16 = ((s_16 <= 2139095040) ? h_17 : 32256);
                        uint32_t mags_7;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_7) : "r"(mag_15), "r"(mag_16));
                        uint32_t signs_7;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_7) : "r"(__as_u32(S[16])), "r"(__as_u32(S[17])));
                        unsigned int word_22 = signs_7 & 2147516416u | mags_7;
                        h_words[0] = word_22;
                        float _abs_351;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_351) : "f"(S[18]));
                        unsigned int s_17 = __as_u32(mul_f32_rz_noftz(_abs_351, __uint_as_float(125829120)));
                        unsigned int _min_352 = ((s_17) < (260046848) ? (s_17) : (260046848));
                        unsigned int h_18 = _min_352 + (h_noise[1] & 8191) >> 13;
                        unsigned int mag_17 = ((s_17 <= 2139095040) ? h_18 : 32256);
                        float _abs_353;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_353) : "f"(S[19]));
                        unsigned int s_18 = __as_u32(mul_f32_rz_noftz(_abs_353, __uint_as_float(125829120)));
                        unsigned int _min_354 = ((s_18) < (260046848) ? (s_18) : (260046848));
                        unsigned int h_19 = _min_354 + ((h_noise[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_18 = ((s_18 <= 2139095040) ? h_19 : 32256);
                        uint32_t mags_8;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_8) : "r"(mag_17), "r"(mag_18));
                        uint32_t signs_8;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_8) : "r"(__as_u32(S[18])), "r"(__as_u32(S[19])));
                        unsigned int word_23 = signs_8 & 2147516416u | mags_8;
                        h_words[1] = word_23;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        float _abs_355;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_355) : "f"(S[24]));
                        unsigned int s_19 = __as_u32(mul_f32_rz_noftz(_abs_355, __uint_as_float(125829120)));
                        unsigned int _min_356 = ((s_19) < (260046848) ? (s_19) : (260046848));
                        unsigned int h_20 = _min_356 + (h_noise[2] & 8191) >> 13;
                        unsigned int mag_19 = ((s_19 <= 2139095040) ? h_20 : 32256);
                        float _abs_357;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_357) : "f"(S[25]));
                        unsigned int s_20 = __as_u32(mul_f32_rz_noftz(_abs_357, __uint_as_float(125829120)));
                        unsigned int _min_358 = ((s_20) < (260046848) ? (s_20) : (260046848));
                        unsigned int h_21 = _min_358 + ((h_noise[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_20 = ((s_20 <= 2139095040) ? h_21 : 32256);
                        uint32_t mags_9;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_9) : "r"(mag_19), "r"(mag_20));
                        uint32_t signs_9;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_9) : "r"(__as_u32(S[24])), "r"(__as_u32(S[25])));
                        unsigned int word_24 = signs_9 & 2147516416u | mags_9;
                        h_words[0] = word_24;
                        float _abs_359;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_359) : "f"(S[26]));
                        unsigned int s_21 = __as_u32(mul_f32_rz_noftz(_abs_359, __uint_as_float(125829120)));
                        unsigned int _min_360 = ((s_21) < (260046848) ? (s_21) : (260046848));
                        unsigned int h_22 = _min_360 + (h_noise[3] & 8191) >> 13;
                        unsigned int mag_21 = ((s_21 <= 2139095040) ? h_22 : 32256);
                        float _abs_361;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_361) : "f"(S[27]));
                        unsigned int s_22 = __as_u32(mul_f32_rz_noftz(_abs_361, __uint_as_float(125829120)));
                        unsigned int _min_362 = ((s_22) < (260046848) ? (s_22) : (260046848));
                        unsigned int h_23 = _min_362 + ((h_noise[3] & 536805376) >> 16) >> 13;
                        unsigned int mag_22 = ((s_22 <= 2139095040) ? h_23 : 32256);
                        uint32_t mags_10;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_10) : "r"(mag_21), "r"(mag_22));
                        uint32_t signs_10;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_10) : "r"(__as_u32(S[26])), "r"(__as_u32(S[27])));
                        unsigned int word_25 = signs_10 & 2147516416u | mags_10;
                        h_words[1] = word_25;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                            : "r"(noise_lane + 12288));
                        float _abs_363;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_363) : "f"(S[20]));
                        unsigned int s_23 = __as_u32(mul_f32_rz_noftz(_abs_363, __uint_as_float(125829120)));
                        unsigned int _min_364 = ((s_23) < (260046848) ? (s_23) : (260046848));
                        unsigned int h_24 = _min_364 + (h_noise[0] & 8191) >> 13;
                        unsigned int mag_23 = ((s_23 <= 2139095040) ? h_24 : 32256);
                        float _abs_365;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_365) : "f"(S[21]));
                        unsigned int s_24 = __as_u32(mul_f32_rz_noftz(_abs_365, __uint_as_float(125829120)));
                        unsigned int _min_366 = ((s_24) < (260046848) ? (s_24) : (260046848));
                        unsigned int h_25 = _min_366 + ((h_noise[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_24 = ((s_24 <= 2139095040) ? h_25 : 32256);
                        uint32_t mags_11;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_11) : "r"(mag_23), "r"(mag_24));
                        uint32_t signs_11;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_11) : "r"(__as_u32(S[20])), "r"(__as_u32(S[21])));
                        unsigned int word_26 = signs_11 & 2147516416u | mags_11;
                        h_words[0] = word_26;
                        float _abs_367;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_367) : "f"(S[22]));
                        unsigned int s_25 = __as_u32(mul_f32_rz_noftz(_abs_367, __uint_as_float(125829120)));
                        unsigned int _min_368 = ((s_25) < (260046848) ? (s_25) : (260046848));
                        unsigned int h_26 = _min_368 + (h_noise[1] & 8191) >> 13;
                        unsigned int mag_25 = ((s_25 <= 2139095040) ? h_26 : 32256);
                        float _abs_369;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_369) : "f"(S[23]));
                        unsigned int s_26 = __as_u32(mul_f32_rz_noftz(_abs_369, __uint_as_float(125829120)));
                        unsigned int _min_370 = ((s_26) < (260046848) ? (s_26) : (260046848));
                        unsigned int h_27 = _min_370 + ((h_noise[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_26 = ((s_26 <= 2139095040) ? h_27 : 32256);
                        uint32_t mags_12;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_12) : "r"(mag_25), "r"(mag_26));
                        uint32_t signs_12;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_12) : "r"(__as_u32(S[22])), "r"(__as_u32(S[23])));
                        unsigned int word_27 = signs_12 & 2147516416u | mags_12;
                        h_words[1] = word_27;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        float _abs_371;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_371) : "f"(S[28]));
                        unsigned int s_27 = __as_u32(mul_f32_rz_noftz(_abs_371, __uint_as_float(125829120)));
                        unsigned int _min_372 = ((s_27) < (260046848) ? (s_27) : (260046848));
                        unsigned int h_28 = _min_372 + (h_noise[2] & 8191) >> 13;
                        unsigned int mag_27 = ((s_27 <= 2139095040) ? h_28 : 32256);
                        float _abs_373;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_373) : "f"(S[29]));
                        unsigned int s_28 = __as_u32(mul_f32_rz_noftz(_abs_373, __uint_as_float(125829120)));
                        unsigned int _min_374 = ((s_28) < (260046848) ? (s_28) : (260046848));
                        unsigned int h_29 = _min_374 + ((h_noise[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_28 = ((s_28 <= 2139095040) ? h_29 : 32256);
                        uint32_t mags_13;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_13) : "r"(mag_27), "r"(mag_28));
                        uint32_t signs_13;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_13) : "r"(__as_u32(S[28])), "r"(__as_u32(S[29])));
                        unsigned int word_28 = signs_13 & 2147516416u | mags_13;
                        h_words[0] = word_28;
                        float _abs_375;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_375) : "f"(S[30]));
                        unsigned int s_29 = __as_u32(mul_f32_rz_noftz(_abs_375, __uint_as_float(125829120)));
                        unsigned int _min_376 = ((s_29) < (260046848) ? (s_29) : (260046848));
                        unsigned int h_30 = _min_376 + (h_noise[3] & 8191) >> 13;
                        unsigned int mag_29 = ((s_29 <= 2139095040) ? h_30 : 32256);
                        float _abs_377;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_377) : "f"(S[31]));
                        unsigned int s_30 = __as_u32(mul_f32_rz_noftz(_abs_377, __uint_as_float(125829120)));
                        unsigned int _min_378 = ((s_30) < (260046848) ? (s_30) : (260046848));
                        unsigned int h_31 = _min_378 + ((h_noise[3] & 536805376) >> 16) >> 13;
                        unsigned int mag_30 = ((s_30 <= 2139095040) ? h_31 : 32256);
                        uint32_t mags_14;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_14) : "r"(mag_29), "r"(mag_30));
                        uint32_t signs_14;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_14) : "r"(__as_u32(S[30])), "r"(__as_u32(S[31])));
                        unsigned int word_29 = signs_14 & 2147516416u | mags_14;
                        h_words[1] = word_29;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    }
#else
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 0));
                    uint32_t _cvt_rs_x2_f32_50;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_50) : "f"(S2[0].y), "f"(S2[0].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_50;
                    uint32_t _cvt_rs_x2_f32_51;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_51) : "f"(S2[1].y), "f"(S2[1].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_51;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_52;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_52) : "f"(S2[4].y), "f"(S2[4].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_52;
                    uint32_t _cvt_rs_x2_f32_53;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_53) : "f"(S2[5].y), "f"(S2[5].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_53;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 4096));
                    uint32_t _cvt_rs_x2_f32_54;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_54) : "f"(S2[2].y), "f"(S2[2].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_54;
                    uint32_t _cvt_rs_x2_f32_55;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_55) : "f"(S2[3].y), "f"(S2[3].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_55;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_56;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_56) : "f"(S2[6].y), "f"(S2[6].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_56;
                    uint32_t _cvt_rs_x2_f32_57;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_57) : "f"(S2[7].y), "f"(S2[7].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_57;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 8192));
                    uint32_t _cvt_rs_x2_f32_58;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_58) : "f"(S2[8].y), "f"(S2[8].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_58;
                    uint32_t _cvt_rs_x2_f32_59;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_59) : "f"(S2[9].y), "f"(S2[9].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_59;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_60;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_60) : "f"(S2[12].y), "f"(S2[12].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_60;
                    uint32_t _cvt_rs_x2_f32_61;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_61) : "f"(S2[13].y), "f"(S2[13].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_61;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 12288));
                    uint32_t _cvt_rs_x2_f32_62;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_62) : "f"(S2[10].y), "f"(S2[10].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_62;
                    uint32_t _cvt_rs_x2_f32_63;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_63) : "f"(S2[11].y), "f"(S2[11].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_63;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_64;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_64) : "f"(S2[14].y), "f"(S2[14].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_64;
                    uint32_t _cvt_rs_x2_f32_65;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_65) : "f"(S2[15].y), "f"(S2[15].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_65;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#endif
                }
                if (philox_rounds == 0) {
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_379;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_379) : "f"(S[1]), "f"(S[0]));
                    h_words[0] = _cvt_rn_f16x2_f32_379;
                    uint32_t _cvt_rn_f16x2_f32_380;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_380) : "f"(S[3]), "f"(S[2]));
                    h_words[1] = _cvt_rn_f16x2_f32_380;
#else
                    uint32_t _cvt_rn_f16x2_f32_66;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_66) : "f"(S2[0].y), "f"(S2[0].x));
                    h_words[0] = _cvt_rn_f16x2_f32_66;
                    uint32_t _cvt_rn_f16x2_f32_67;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_67) : "f"(S2[1].y), "f"(S2[1].x));
                    h_words[1] = _cvt_rn_f16x2_f32_67;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_381;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_381) : "f"(S[5]), "f"(S[4]));
                    h_words[0] = _cvt_rn_f16x2_f32_381;
                    uint32_t _cvt_rn_f16x2_f32_382;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_382) : "f"(S[7]), "f"(S[6]));
                    h_words[1] = _cvt_rn_f16x2_f32_382;
#else
                    uint32_t _cvt_rn_f16x2_f32_68;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_68) : "f"(S2[2].y), "f"(S2[2].x));
                    h_words[0] = _cvt_rn_f16x2_f32_68;
                    uint32_t _cvt_rn_f16x2_f32_69;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_69) : "f"(S2[3].y), "f"(S2[3].x));
                    h_words[1] = _cvt_rn_f16x2_f32_69;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_383;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_383) : "f"(S[9]), "f"(S[8]));
                    h_words[0] = _cvt_rn_f16x2_f32_383;
                    uint32_t _cvt_rn_f16x2_f32_384;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_384) : "f"(S[11]), "f"(S[10]));
                    h_words[1] = _cvt_rn_f16x2_f32_384;
#else
                    uint32_t _cvt_rn_f16x2_f32_70;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_70) : "f"(S2[4].y), "f"(S2[4].x));
                    h_words[0] = _cvt_rn_f16x2_f32_70;
                    uint32_t _cvt_rn_f16x2_f32_71;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_71) : "f"(S2[5].y), "f"(S2[5].x));
                    h_words[1] = _cvt_rn_f16x2_f32_71;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_385;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_385) : "f"(S[13]), "f"(S[12]));
                    h_words[0] = _cvt_rn_f16x2_f32_385;
                    uint32_t _cvt_rn_f16x2_f32_386;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_386) : "f"(S[15]), "f"(S[14]));
                    h_words[1] = _cvt_rn_f16x2_f32_386;
#else
                    uint32_t _cvt_rn_f16x2_f32_72;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_72) : "f"(S2[6].y), "f"(S2[6].x));
                    h_words[0] = _cvt_rn_f16x2_f32_72;
                    uint32_t _cvt_rn_f16x2_f32_73;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_73) : "f"(S2[7].y), "f"(S2[7].x));
                    h_words[1] = _cvt_rn_f16x2_f32_73;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_387;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_387) : "f"(S[17]), "f"(S[16]));
                    h_words[0] = _cvt_rn_f16x2_f32_387;
                    uint32_t _cvt_rn_f16x2_f32_388;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_388) : "f"(S[19]), "f"(S[18]));
                    h_words[1] = _cvt_rn_f16x2_f32_388;
#else
                    uint32_t _cvt_rn_f16x2_f32_74;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_74) : "f"(S2[8].y), "f"(S2[8].x));
                    h_words[0] = _cvt_rn_f16x2_f32_74;
                    uint32_t _cvt_rn_f16x2_f32_75;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_75) : "f"(S2[9].y), "f"(S2[9].x));
                    h_words[1] = _cvt_rn_f16x2_f32_75;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_389;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_389) : "f"(S[21]), "f"(S[20]));
                    h_words[0] = _cvt_rn_f16x2_f32_389;
                    uint32_t _cvt_rn_f16x2_f32_390;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_390) : "f"(S[23]), "f"(S[22]));
                    h_words[1] = _cvt_rn_f16x2_f32_390;
#else
                    uint32_t _cvt_rn_f16x2_f32_76;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_76) : "f"(S2[10].y), "f"(S2[10].x));
                    h_words[0] = _cvt_rn_f16x2_f32_76;
                    uint32_t _cvt_rn_f16x2_f32_77;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_77) : "f"(S2[11].y), "f"(S2[11].x));
                    h_words[1] = _cvt_rn_f16x2_f32_77;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_391;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_391) : "f"(S[25]), "f"(S[24]));
                    h_words[0] = _cvt_rn_f16x2_f32_391;
                    uint32_t _cvt_rn_f16x2_f32_392;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_392) : "f"(S[27]), "f"(S[26]));
                    h_words[1] = _cvt_rn_f16x2_f32_392;
#else
                    uint32_t _cvt_rn_f16x2_f32_78;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_78) : "f"(S2[12].y), "f"(S2[12].x));
                    h_words[0] = _cvt_rn_f16x2_f32_78;
                    uint32_t _cvt_rn_f16x2_f32_79;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_79) : "f"(S2[13].y), "f"(S2[13].x));
                    h_words[1] = _cvt_rn_f16x2_f32_79;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_393;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_393) : "f"(S[29]), "f"(S[28]));
                    h_words[0] = _cvt_rn_f16x2_f32_393;
                    uint32_t _cvt_rn_f16x2_f32_394;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_394) : "f"(S[31]), "f"(S[30]));
                    h_words[1] = _cvt_rn_f16x2_f32_394;
#else
                    uint32_t _cvt_rn_f16x2_f32_80;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_80) : "f"(S2[14].y), "f"(S2[14].x));
                    h_words[0] = _cvt_rn_f16x2_f32_80;
                    uint32_t _cvt_rn_f16x2_f32_81;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_81) : "f"(S2[15].y), "f"(S2[15].x));
                    h_words[1] = _cvt_rn_f16x2_f32_81;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                }
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
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_3 * 256) + group_byte));
                float beta_3 = sBeta[m_3];
#if __CUDA_ARCH__ == 900
                float acc0_x_7 = 0.0f;
                float acc0_y_7 = 0.0f;
                float acc1_x_7 = 0.0f;
                float acc1_y_7 = 0.0f;
                float _fma_395;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_395) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_7));
                acc0_x_7 = _fma_395;
                float _fma_396;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_396) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_7));
                acc0_y_7 = _fma_396;
                float _fma_397;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_397) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_7));
                acc1_x_7 = _fma_397;
                float _fma_398;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_398) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_7));
                acc1_y_7 = _fma_398;
                float _fma_399;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_399) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_7));
                acc0_x_7 = _fma_399;
                float _fma_400;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_400) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_7));
                acc0_y_7 = _fma_400;
                float _fma_401;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_401) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_7));
                acc1_x_7 = _fma_401;
                float _fma_402;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_402) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_7));
                acc1_y_7 = _fma_402;
                float _fma_403;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_403) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_7));
                acc0_x_7 = _fma_403;
                float _fma_404;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_404) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_7));
                acc0_y_7 = _fma_404;
                float _fma_405;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_405) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_7));
                acc1_x_7 = _fma_405;
                float _fma_406;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_406) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_7));
                acc1_y_7 = _fma_406;
                float _fma_407;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_407) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_7));
                acc0_x_7 = _fma_407;
                float _fma_408;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_408) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_7));
                acc0_y_7 = _fma_408;
                float _fma_409;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_409) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_7));
                acc1_x_7 = _fma_409;
                float _fma_410;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_410) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_7));
                acc1_y_7 = _fma_410;
                float _add_411;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_411) : "f"(acc0_x_7), "f"(acc1_x_7));
                float _add_412;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_412) : "f"(acc0_y_7), "f"(acc1_y_7));
                float dot_7 = _add_411 + _add_412;
                float _shfl_xor_sync_413 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 1);
                dot_7 = dot_7 + _shfl_xor_sync_413;
                float _shfl_xor_sync_414 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 2);
                dot_7 = dot_7 + _shfl_xor_sync_414;
                float _shfl_xor_sync_415 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 4);
                dot_7 = dot_7 + _shfl_xor_sync_415;
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
                float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 1);
                dot_7 = dot_7 + _shfl_xor_sync_82;
                float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 2);
                dot_7 = dot_7 + _shfl_xor_sync_83;
                float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 4);
                dot_7 = dot_7 + _shfl_xor_sync_84;
#endif
                float vn_6 = (v_regs[0] - dot_7) * beta_3;
#if __CUDA_ARCH__ == 900
                float _fma_416;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_416) : "f"(k_regs[0]), "f"(vn_6), "f"(S[0]));
                S[0] = _fma_416;
                float _fma_417;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_417) : "f"(k_regs[1]), "f"(vn_6), "f"(S[1]));
                S[1] = _fma_417;
                float _fma_418;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_418) : "f"(k_regs[2]), "f"(vn_6), "f"(S[2]));
                S[2] = _fma_418;
                float _fma_419;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_419) : "f"(k_regs[3]), "f"(vn_6), "f"(S[3]));
                S[3] = _fma_419;
                float _fma_420;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_420) : "f"(k_regs[4]), "f"(vn_6), "f"(S[4]));
                S[4] = _fma_420;
                float _fma_421;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_421) : "f"(k_regs[5]), "f"(vn_6), "f"(S[5]));
                S[5] = _fma_421;
                float _fma_422;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_422) : "f"(k_regs[6]), "f"(vn_6), "f"(S[6]));
                S[6] = _fma_422;
                float _fma_423;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_423) : "f"(k_regs[7]), "f"(vn_6), "f"(S[7]));
                S[7] = _fma_423;
                float _fma_424;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_424) : "f"(k_regs[8]), "f"(vn_6), "f"(S[8]));
                S[8] = _fma_424;
                float _fma_425;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_425) : "f"(k_regs[9]), "f"(vn_6), "f"(S[9]));
                S[9] = _fma_425;
                float _fma_426;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_426) : "f"(k_regs[10]), "f"(vn_6), "f"(S[10]));
                S[10] = _fma_426;
                float _fma_427;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_427) : "f"(k_regs[11]), "f"(vn_6), "f"(S[11]));
                S[11] = _fma_427;
                float _fma_428;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_428) : "f"(k_regs[12]), "f"(vn_6), "f"(S[12]));
                S[12] = _fma_428;
                float _fma_429;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_429) : "f"(k_regs[13]), "f"(vn_6), "f"(S[13]));
                S[13] = _fma_429;
                float _fma_430;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_430) : "f"(k_regs[14]), "f"(vn_6), "f"(S[14]));
                S[14] = _fma_430;
                float _fma_431;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_431) : "f"(k_regs[15]), "f"(vn_6), "f"(S[15]));
                S[15] = _fma_431;
                float acc0_x_7_1 = 0.0f;
                float acc0_y_7_1 = 0.0f;
                float acc1_x_7_1 = 0.0f;
                float acc1_y_7_1 = 0.0f;
                float _fma_432;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_432) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_432;
                float _fma_433;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_433) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_433;
                float _fma_434;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_434) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_434;
                float _fma_435;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_435) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_435;
                float _fma_436;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_436) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_436;
                float _fma_437;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_437) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_437;
                float _fma_438;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_438) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_438;
                float _fma_439;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_439) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_439;
                float _fma_440;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_440) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_440;
                float _fma_441;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_441) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_441;
                float _fma_442;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_442) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_442;
                float _fma_443;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_443) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_443;
                float _fma_444;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_444) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_444;
                float _fma_445;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_445) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_445;
                float _fma_446;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_446) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_446;
                float _fma_447;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_447) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_447;
                float _add_448;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_448) : "f"(acc0_x_7_1), "f"(acc1_x_7_1));
                float _add_449;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_449) : "f"(acc0_y_7_1), "f"(acc1_y_7_1));
                float dot_7_1 = _add_448 + _add_449;
                float _shfl_xor_sync_450 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 1);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_450;
                float _shfl_xor_sync_451 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 2);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_451;
                float _shfl_xor_sync_452 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 4);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_452;
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
                float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 1);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_85;
                float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 2);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_86;
                float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 4);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_87;
#endif
                float vn_7 = (v_regs[1] - dot_7_1) * beta_3;
#if __CUDA_ARCH__ == 900
                float _fma_453;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_453) : "f"(k_regs[0]), "f"(vn_7), "f"(S[16]));
                S[16] = _fma_453;
                float _fma_454;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_454) : "f"(k_regs[1]), "f"(vn_7), "f"(S[17]));
                S[17] = _fma_454;
                float _fma_455;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_455) : "f"(k_regs[2]), "f"(vn_7), "f"(S[18]));
                S[18] = _fma_455;
                float _fma_456;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_456) : "f"(k_regs[3]), "f"(vn_7), "f"(S[19]));
                S[19] = _fma_456;
                float _fma_457;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_457) : "f"(k_regs[4]), "f"(vn_7), "f"(S[20]));
                S[20] = _fma_457;
                float _fma_458;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_458) : "f"(k_regs[5]), "f"(vn_7), "f"(S[21]));
                S[21] = _fma_458;
                float _fma_459;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_459) : "f"(k_regs[6]), "f"(vn_7), "f"(S[22]));
                S[22] = _fma_459;
                float _fma_460;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_460) : "f"(k_regs[7]), "f"(vn_7), "f"(S[23]));
                S[23] = _fma_460;
                float _fma_461;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_461) : "f"(k_regs[8]), "f"(vn_7), "f"(S[24]));
                S[24] = _fma_461;
                float _fma_462;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_462) : "f"(k_regs[9]), "f"(vn_7), "f"(S[25]));
                S[25] = _fma_462;
                float _fma_463;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_463) : "f"(k_regs[10]), "f"(vn_7), "f"(S[26]));
                S[26] = _fma_463;
                float _fma_464;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_464) : "f"(k_regs[11]), "f"(vn_7), "f"(S[27]));
                S[27] = _fma_464;
                float _fma_465;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_465) : "f"(k_regs[12]), "f"(vn_7), "f"(S[28]));
                S[28] = _fma_465;
                float _fma_466;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_466) : "f"(k_regs[13]), "f"(vn_7), "f"(S[29]));
                S[29] = _fma_466;
                float _fma_467;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_467) : "f"(k_regs[14]), "f"(vn_7), "f"(S[30]));
                S[30] = _fma_467;
                float _fma_468;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_468) : "f"(k_regs[15]), "f"(vn_7), "f"(S[31]));
                S[31] = _fma_468;
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
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_4 * 256) + group_byte));
                float beta_4 = sBeta[m_4];
#if __CUDA_ARCH__ == 900
                float acc0_x_8 = 0.0f;
                float acc0_y_8 = 0.0f;
                float acc1_x_8 = 0.0f;
                float acc1_y_8 = 0.0f;
                float _fma_469;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_469) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_8));
                acc0_x_8 = _fma_469;
                float _fma_470;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_470) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_8));
                acc0_y_8 = _fma_470;
                float _fma_471;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_471) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_8));
                acc1_x_8 = _fma_471;
                float _fma_472;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_472) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_8));
                acc1_y_8 = _fma_472;
                float _fma_473;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_473) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_8));
                acc0_x_8 = _fma_473;
                float _fma_474;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_474) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_8));
                acc0_y_8 = _fma_474;
                float _fma_475;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_475) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_8));
                acc1_x_8 = _fma_475;
                float _fma_476;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_476) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_8));
                acc1_y_8 = _fma_476;
                float _fma_477;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_477) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_8));
                acc0_x_8 = _fma_477;
                float _fma_478;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_478) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_8));
                acc0_y_8 = _fma_478;
                float _fma_479;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_479) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_8));
                acc1_x_8 = _fma_479;
                float _fma_480;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_480) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_8));
                acc1_y_8 = _fma_480;
                float _fma_481;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_481) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_8));
                acc0_x_8 = _fma_481;
                float _fma_482;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_482) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_8));
                acc0_y_8 = _fma_482;
                float _fma_483;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_483) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_8));
                acc1_x_8 = _fma_483;
                float _fma_484;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_484) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_8));
                acc1_y_8 = _fma_484;
                float _add_485;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_485) : "f"(acc0_x_8), "f"(acc1_x_8));
                float _add_486;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_486) : "f"(acc0_y_8), "f"(acc1_y_8));
                float dot_8 = _add_485 + _add_486;
                float _shfl_xor_sync_487 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 1);
                dot_8 = dot_8 + _shfl_xor_sync_487;
                float _shfl_xor_sync_488 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 2);
                dot_8 = dot_8 + _shfl_xor_sync_488;
                float _shfl_xor_sync_489 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 4);
                dot_8 = dot_8 + _shfl_xor_sync_489;
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
                float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 1);
                dot_8 = dot_8 + _shfl_xor_sync_88;
                float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 2);
                dot_8 = dot_8 + _shfl_xor_sync_89;
                float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 4);
                dot_8 = dot_8 + _shfl_xor_sync_90;
#endif
                float vn_8 = (v_regs[0] - dot_8) * beta_4;
#if __CUDA_ARCH__ == 900
                float _fma_490;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_490) : "f"(k_regs[0]), "f"(vn_8), "f"(S[0]));
                S[0] = _fma_490;
                float _fma_491;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_491) : "f"(k_regs[1]), "f"(vn_8), "f"(S[1]));
                S[1] = _fma_491;
                float _fma_492;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_492) : "f"(k_regs[2]), "f"(vn_8), "f"(S[2]));
                S[2] = _fma_492;
                float _fma_493;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_493) : "f"(k_regs[3]), "f"(vn_8), "f"(S[3]));
                S[3] = _fma_493;
                float _fma_494;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_494) : "f"(k_regs[4]), "f"(vn_8), "f"(S[4]));
                S[4] = _fma_494;
                float _fma_495;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_495) : "f"(k_regs[5]), "f"(vn_8), "f"(S[5]));
                S[5] = _fma_495;
                float _fma_496;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_496) : "f"(k_regs[6]), "f"(vn_8), "f"(S[6]));
                S[6] = _fma_496;
                float _fma_497;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_497) : "f"(k_regs[7]), "f"(vn_8), "f"(S[7]));
                S[7] = _fma_497;
                float _fma_498;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_498) : "f"(k_regs[8]), "f"(vn_8), "f"(S[8]));
                S[8] = _fma_498;
                float _fma_499;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_499) : "f"(k_regs[9]), "f"(vn_8), "f"(S[9]));
                S[9] = _fma_499;
                float _fma_500;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_500) : "f"(k_regs[10]), "f"(vn_8), "f"(S[10]));
                S[10] = _fma_500;
                float _fma_501;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_501) : "f"(k_regs[11]), "f"(vn_8), "f"(S[11]));
                S[11] = _fma_501;
                float _fma_502;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_502) : "f"(k_regs[12]), "f"(vn_8), "f"(S[12]));
                S[12] = _fma_502;
                float _fma_503;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_503) : "f"(k_regs[13]), "f"(vn_8), "f"(S[13]));
                S[13] = _fma_503;
                float _fma_504;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_504) : "f"(k_regs[14]), "f"(vn_8), "f"(S[14]));
                S[14] = _fma_504;
                float _fma_505;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_505) : "f"(k_regs[15]), "f"(vn_8), "f"(S[15]));
                S[15] = _fma_505;
                float acc0_x_9 = 0.0f;
                float acc0_y_9 = 0.0f;
                float acc1_x_9 = 0.0f;
                float acc1_y_9 = 0.0f;
                float _fma_506;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_506) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_9));
                acc0_x_9 = _fma_506;
                float _fma_507;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_507) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_9));
                acc0_y_9 = _fma_507;
                float _fma_508;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_508) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_9));
                acc1_x_9 = _fma_508;
                float _fma_509;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_509) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_9));
                acc1_y_9 = _fma_509;
                float _fma_510;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_510) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_9));
                acc0_x_9 = _fma_510;
                float _fma_511;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_511) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_9));
                acc0_y_9 = _fma_511;
                float _fma_512;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_512) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_9));
                acc1_x_9 = _fma_512;
                float _fma_513;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_513) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_9));
                acc1_y_9 = _fma_513;
                float _fma_514;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_514) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_9));
                acc0_x_9 = _fma_514;
                float _fma_515;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_515) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_9));
                acc0_y_9 = _fma_515;
                float _fma_516;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_516) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_9));
                acc1_x_9 = _fma_516;
                float _fma_517;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_517) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_9));
                acc1_y_9 = _fma_517;
                float _fma_518;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_518) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_9));
                acc0_x_9 = _fma_518;
                float _fma_519;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_519) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_9));
                acc0_y_9 = _fma_519;
                float _fma_520;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_520) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_9));
                acc1_x_9 = _fma_520;
                float _fma_521;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_521) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_9));
                acc1_y_9 = _fma_521;
                float _add_522;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_522) : "f"(acc0_x_9), "f"(acc1_x_9));
                float _add_523;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_523) : "f"(acc0_y_9), "f"(acc1_y_9));
                float dot_9 = _add_522 + _add_523;
                float _shfl_xor_sync_524 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 1);
                dot_9 = dot_9 + _shfl_xor_sync_524;
                float _shfl_xor_sync_525 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 2);
                dot_9 = dot_9 + _shfl_xor_sync_525;
                float _shfl_xor_sync_526 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 4);
                dot_9 = dot_9 + _shfl_xor_sync_526;
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
                float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 1);
                dot_9 = dot_9 + _shfl_xor_sync_91;
                float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 2);
                dot_9 = dot_9 + _shfl_xor_sync_92;
                float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 4);
                dot_9 = dot_9 + _shfl_xor_sync_93;
#endif
                float vn_9 = (v_regs[1] - dot_9) * beta_4;
#if __CUDA_ARCH__ == 900
                float _fma_527;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_527) : "f"(k_regs[0]), "f"(vn_9), "f"(S[16]));
                S[16] = _fma_527;
                float _fma_528;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_528) : "f"(k_regs[1]), "f"(vn_9), "f"(S[17]));
                S[17] = _fma_528;
                float _fma_529;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_529) : "f"(k_regs[2]), "f"(vn_9), "f"(S[18]));
                S[18] = _fma_529;
                float _fma_530;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_530) : "f"(k_regs[3]), "f"(vn_9), "f"(S[19]));
                S[19] = _fma_530;
                float _fma_531;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_531) : "f"(k_regs[4]), "f"(vn_9), "f"(S[20]));
                S[20] = _fma_531;
                float _fma_532;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_532) : "f"(k_regs[5]), "f"(vn_9), "f"(S[21]));
                S[21] = _fma_532;
                float _fma_533;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_533) : "f"(k_regs[6]), "f"(vn_9), "f"(S[22]));
                S[22] = _fma_533;
                float _fma_534;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_534) : "f"(k_regs[7]), "f"(vn_9), "f"(S[23]));
                S[23] = _fma_534;
                float _fma_535;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_535) : "f"(k_regs[8]), "f"(vn_9), "f"(S[24]));
                S[24] = _fma_535;
                float _fma_536;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_536) : "f"(k_regs[9]), "f"(vn_9), "f"(S[25]));
                S[25] = _fma_536;
                float _fma_537;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_537) : "f"(k_regs[10]), "f"(vn_9), "f"(S[26]));
                S[26] = _fma_537;
                float _fma_538;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_538) : "f"(k_regs[11]), "f"(vn_9), "f"(S[27]));
                S[27] = _fma_538;
                float _fma_539;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_539) : "f"(k_regs[12]), "f"(vn_9), "f"(S[28]));
                S[28] = _fma_539;
                float _fma_540;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_540) : "f"(k_regs[13]), "f"(vn_9), "f"(S[29]));
                S[29] = _fma_540;
                float _fma_541;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_541) : "f"(k_regs[14]), "f"(vn_9), "f"(S[30]));
                S[30] = _fma_541;
                float _fma_542;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_542) : "f"(k_regs[15]), "f"(vn_9), "f"(S[31]));
                S[31] = _fma_542;
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
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_5 * 256) + group_byte));
                float beta_5 = sBeta[m_5];
#if __CUDA_ARCH__ == 900
                float acc0_x_10 = 0.0f;
                float acc0_y_10 = 0.0f;
                float acc1_x_10 = 0.0f;
                float acc1_y_10 = 0.0f;
                float _fma_543;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_543) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_10));
                acc0_x_10 = _fma_543;
                float _fma_544;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_544) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_10));
                acc0_y_10 = _fma_544;
                float _fma_545;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_545) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_10));
                acc1_x_10 = _fma_545;
                float _fma_546;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_546) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_10));
                acc1_y_10 = _fma_546;
                float _fma_547;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_547) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_10));
                acc0_x_10 = _fma_547;
                float _fma_548;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_548) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_10));
                acc0_y_10 = _fma_548;
                float _fma_549;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_549) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_10));
                acc1_x_10 = _fma_549;
                float _fma_550;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_550) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_10));
                acc1_y_10 = _fma_550;
                float _fma_551;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_551) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_10));
                acc0_x_10 = _fma_551;
                float _fma_552;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_552) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_10));
                acc0_y_10 = _fma_552;
                float _fma_553;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_553) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_10));
                acc1_x_10 = _fma_553;
                float _fma_554;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_554) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_10));
                acc1_y_10 = _fma_554;
                float _fma_555;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_555) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_10));
                acc0_x_10 = _fma_555;
                float _fma_556;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_556) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_10));
                acc0_y_10 = _fma_556;
                float _fma_557;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_557) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_10));
                acc1_x_10 = _fma_557;
                float _fma_558;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_558) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_10));
                acc1_y_10 = _fma_558;
                float _add_559;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_559) : "f"(acc0_x_10), "f"(acc1_x_10));
                float _add_560;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_560) : "f"(acc0_y_10), "f"(acc1_y_10));
                float dot_10 = _add_559 + _add_560;
                float _shfl_xor_sync_561 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 1);
                dot_10 = dot_10 + _shfl_xor_sync_561;
                float _shfl_xor_sync_562 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 2);
                dot_10 = dot_10 + _shfl_xor_sync_562;
                float _shfl_xor_sync_563 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 4);
                dot_10 = dot_10 + _shfl_xor_sync_563;
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
                float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 1);
                dot_10 = dot_10 + _shfl_xor_sync_94;
                float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 2);
                dot_10 = dot_10 + _shfl_xor_sync_95;
                float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 4);
                dot_10 = dot_10 + _shfl_xor_sync_96;
#endif
                float vn_10 = (v_regs[0] - dot_10) * beta_5;
#if __CUDA_ARCH__ == 900
                float _fma_564;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_564) : "f"(k_regs[0]), "f"(vn_10), "f"(S[0]));
                S[0] = _fma_564;
                float _fma_565;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_565) : "f"(k_regs[1]), "f"(vn_10), "f"(S[1]));
                S[1] = _fma_565;
                float _fma_566;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_566) : "f"(k_regs[2]), "f"(vn_10), "f"(S[2]));
                S[2] = _fma_566;
                float _fma_567;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_567) : "f"(k_regs[3]), "f"(vn_10), "f"(S[3]));
                S[3] = _fma_567;
                float _fma_568;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_568) : "f"(k_regs[4]), "f"(vn_10), "f"(S[4]));
                S[4] = _fma_568;
                float _fma_569;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_569) : "f"(k_regs[5]), "f"(vn_10), "f"(S[5]));
                S[5] = _fma_569;
                float _fma_570;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_570) : "f"(k_regs[6]), "f"(vn_10), "f"(S[6]));
                S[6] = _fma_570;
                float _fma_571;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_571) : "f"(k_regs[7]), "f"(vn_10), "f"(S[7]));
                S[7] = _fma_571;
                float _fma_572;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_572) : "f"(k_regs[8]), "f"(vn_10), "f"(S[8]));
                S[8] = _fma_572;
                float _fma_573;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_573) : "f"(k_regs[9]), "f"(vn_10), "f"(S[9]));
                S[9] = _fma_573;
                float _fma_574;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_574) : "f"(k_regs[10]), "f"(vn_10), "f"(S[10]));
                S[10] = _fma_574;
                float _fma_575;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_575) : "f"(k_regs[11]), "f"(vn_10), "f"(S[11]));
                S[11] = _fma_575;
                float _fma_576;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_576) : "f"(k_regs[12]), "f"(vn_10), "f"(S[12]));
                S[12] = _fma_576;
                float _fma_577;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_577) : "f"(k_regs[13]), "f"(vn_10), "f"(S[13]));
                S[13] = _fma_577;
                float _fma_578;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_578) : "f"(k_regs[14]), "f"(vn_10), "f"(S[14]));
                S[14] = _fma_578;
                float _fma_579;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_579) : "f"(k_regs[15]), "f"(vn_10), "f"(S[15]));
                S[15] = _fma_579;
                float acc0_x_11 = 0.0f;
                float acc0_y_11 = 0.0f;
                float acc1_x_11 = 0.0f;
                float acc1_y_11 = 0.0f;
                float _fma_580;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_580) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_11));
                acc0_x_11 = _fma_580;
                float _fma_581;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_581) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_11));
                acc0_y_11 = _fma_581;
                float _fma_582;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_582) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_11));
                acc1_x_11 = _fma_582;
                float _fma_583;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_583) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_11));
                acc1_y_11 = _fma_583;
                float _fma_584;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_584) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_11));
                acc0_x_11 = _fma_584;
                float _fma_585;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_585) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_11));
                acc0_y_11 = _fma_585;
                float _fma_586;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_586) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_11));
                acc1_x_11 = _fma_586;
                float _fma_587;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_587) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_11));
                acc1_y_11 = _fma_587;
                float _fma_588;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_588) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_11));
                acc0_x_11 = _fma_588;
                float _fma_589;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_589) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_11));
                acc0_y_11 = _fma_589;
                float _fma_590;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_590) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_11));
                acc1_x_11 = _fma_590;
                float _fma_591;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_591) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_11));
                acc1_y_11 = _fma_591;
                float _fma_592;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_592) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_11));
                acc0_x_11 = _fma_592;
                float _fma_593;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_593) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_11));
                acc0_y_11 = _fma_593;
                float _fma_594;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_594) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_11));
                acc1_x_11 = _fma_594;
                float _fma_595;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_595) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_11));
                acc1_y_11 = _fma_595;
                float _add_596;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_596) : "f"(acc0_x_11), "f"(acc1_x_11));
                float _add_597;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_597) : "f"(acc0_y_11), "f"(acc1_y_11));
                float dot_11 = _add_596 + _add_597;
                float _shfl_xor_sync_598 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 1);
                dot_11 = dot_11 + _shfl_xor_sync_598;
                float _shfl_xor_sync_599 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 2);
                dot_11 = dot_11 + _shfl_xor_sync_599;
                float _shfl_xor_sync_600 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 4);
                dot_11 = dot_11 + _shfl_xor_sync_600;
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
                float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 1);
                dot_11 = dot_11 + _shfl_xor_sync_97;
                float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 2);
                dot_11 = dot_11 + _shfl_xor_sync_98;
                float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 4);
                dot_11 = dot_11 + _shfl_xor_sync_99;
#endif
                float vn_11 = (v_regs[1] - dot_11) * beta_5;
#if __CUDA_ARCH__ == 900
                float _fma_601;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_601) : "f"(k_regs[0]), "f"(vn_11), "f"(S[16]));
                S[16] = _fma_601;
                float _fma_602;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_602) : "f"(k_regs[1]), "f"(vn_11), "f"(S[17]));
                S[17] = _fma_602;
                float _fma_603;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_603) : "f"(k_regs[2]), "f"(vn_11), "f"(S[18]));
                S[18] = _fma_603;
                float _fma_604;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_604) : "f"(k_regs[3]), "f"(vn_11), "f"(S[19]));
                S[19] = _fma_604;
                float _fma_605;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_605) : "f"(k_regs[4]), "f"(vn_11), "f"(S[20]));
                S[20] = _fma_605;
                float _fma_606;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_606) : "f"(k_regs[5]), "f"(vn_11), "f"(S[21]));
                S[21] = _fma_606;
                float _fma_607;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_607) : "f"(k_regs[6]), "f"(vn_11), "f"(S[22]));
                S[22] = _fma_607;
                float _fma_608;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_608) : "f"(k_regs[7]), "f"(vn_11), "f"(S[23]));
                S[23] = _fma_608;
                float _fma_609;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_609) : "f"(k_regs[8]), "f"(vn_11), "f"(S[24]));
                S[24] = _fma_609;
                float _fma_610;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_610) : "f"(k_regs[9]), "f"(vn_11), "f"(S[25]));
                S[25] = _fma_610;
                float _fma_611;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_611) : "f"(k_regs[10]), "f"(vn_11), "f"(S[26]));
                S[26] = _fma_611;
                float _fma_612;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_612) : "f"(k_regs[11]), "f"(vn_11), "f"(S[27]));
                S[27] = _fma_612;
                float _fma_613;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_613) : "f"(k_regs[12]), "f"(vn_11), "f"(S[28]));
                S[28] = _fma_613;
                float _fma_614;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_614) : "f"(k_regs[13]), "f"(vn_11), "f"(S[29]));
                S[29] = _fma_614;
                float _fma_615;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_615) : "f"(k_regs[14]), "f"(vn_11), "f"(S[30]));
                S[30] = _fma_615;
                float _fma_616;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_616) : "f"(k_regs[15]), "f"(vn_11), "f"(S[31]));
                S[31] = _fma_616;
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
                float acc0_x_12 = 0.0f;
                float acc0_y_12 = 0.0f;
                float acc1_x_12 = 0.0f;
                float acc1_y_12 = 0.0f;
                float _fma_617;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_617) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_12));
                acc0_x_12 = _fma_617;
                float _fma_618;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_618) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_12));
                acc0_y_12 = _fma_618;
                float _fma_619;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_619) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_12));
                acc1_x_12 = _fma_619;
                float _fma_620;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_620) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_12));
                acc1_y_12 = _fma_620;
                float _fma_621;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_621) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_12));
                acc0_x_12 = _fma_621;
                float _fma_622;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_622) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_12));
                acc0_y_12 = _fma_622;
                float _fma_623;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_623) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_12));
                acc1_x_12 = _fma_623;
                float _fma_624;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_624) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_12));
                acc1_y_12 = _fma_624;
                float _fma_625;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_625) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_12));
                acc0_x_12 = _fma_625;
                float _fma_626;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_626) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_12));
                acc0_y_12 = _fma_626;
                float _fma_627;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_627) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_12));
                acc1_x_12 = _fma_627;
                float _fma_628;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_628) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_12));
                acc1_y_12 = _fma_628;
                float _fma_629;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_629) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_12));
                acc0_x_12 = _fma_629;
                float _fma_630;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_630) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_12));
                acc0_y_12 = _fma_630;
                float _fma_631;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_631) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_12));
                acc1_x_12 = _fma_631;
                float _fma_632;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_632) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_12));
                acc1_y_12 = _fma_632;
                float _add_633;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_633) : "f"(acc0_x_12), "f"(acc1_x_12));
                float _add_634;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_634) : "f"(acc0_y_12), "f"(acc1_y_12));
                float dot_12 = _add_633 + _add_634;
                float _shfl_xor_sync_635 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 1);
                dot_12 = dot_12 + _shfl_xor_sync_635;
                float _shfl_xor_sync_636 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 2);
                dot_12 = dot_12 + _shfl_xor_sync_636;
                float _shfl_xor_sync_637 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 4);
                dot_12 = dot_12 + _shfl_xor_sync_637;
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
                float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 1);
                dot_12 = dot_12 + _shfl_xor_sync_100;
                float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 2);
                dot_12 = dot_12 + _shfl_xor_sync_101;
                float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 4);
                dot_12 = dot_12 + _shfl_xor_sync_102;
#endif
                o4[0] = dot_12;
#if __CUDA_ARCH__ == 900
                float acc0_x_13 = 0.0f;
                float acc0_y_13 = 0.0f;
                float acc1_x_13 = 0.0f;
                float acc1_y_13 = 0.0f;
                float _fma_638;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_638) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_13));
                acc0_x_13 = _fma_638;
                float _fma_639;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_639) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_13));
                acc0_y_13 = _fma_639;
                float _fma_640;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_640) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_13));
                acc1_x_13 = _fma_640;
                float _fma_641;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_641) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_13));
                acc1_y_13 = _fma_641;
                float _fma_642;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_642) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_13));
                acc0_x_13 = _fma_642;
                float _fma_643;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_643) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_13));
                acc0_y_13 = _fma_643;
                float _fma_644;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_644) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_13));
                acc1_x_13 = _fma_644;
                float _fma_645;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_645) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_13));
                acc1_y_13 = _fma_645;
                float _fma_646;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_646) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_13));
                acc0_x_13 = _fma_646;
                float _fma_647;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_647) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_13));
                acc0_y_13 = _fma_647;
                float _fma_648;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_648) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_13));
                acc1_x_13 = _fma_648;
                float _fma_649;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_649) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_13));
                acc1_y_13 = _fma_649;
                float _fma_650;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_650) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_13));
                acc0_x_13 = _fma_650;
                float _fma_651;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_651) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_13));
                acc0_y_13 = _fma_651;
                float _fma_652;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_652) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_13));
                acc1_x_13 = _fma_652;
                float _fma_653;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_653) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_13));
                acc1_y_13 = _fma_653;
                float _add_654;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_654) : "f"(acc0_x_13), "f"(acc1_x_13));
                float _add_655;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_655) : "f"(acc0_y_13), "f"(acc1_y_13));
                float dot_13 = _add_654 + _add_655;
                float _shfl_xor_sync_656 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 1);
                dot_13 = dot_13 + _shfl_xor_sync_656;
                float _shfl_xor_sync_657 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 2);
                dot_13 = dot_13 + _shfl_xor_sync_657;
                float _shfl_xor_sync_658 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 4);
                dot_13 = dot_13 + _shfl_xor_sync_658;
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
                float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 1);
                dot_13 = dot_13 + _shfl_xor_sync_103;
                float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 2);
                dot_13 = dot_13 + _shfl_xor_sync_104;
                float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 4);
                dot_13 = dot_13 + _shfl_xor_sync_105;
#endif
                o4[1] = dot_13;
                if (u == 0) {
                    long long out_index_1 = (long long)n * out_stride_p0 + (long long)t_1 * out_stride_p1 + (long long)h * out_stride_p2;
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
                    uint32_t _read_special_reg_659;
                    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(_read_special_reg_659));
                    int tid_r = (int)_read_special_reg_659;
#else
                    uint32_t _read_special_reg_106;
                    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(_read_special_reg_106));
                    int tid_r = (int)_read_special_reg_106;
#endif
                    long long noise_at = read_slot_head + (long long)(tid_r / 8 * 2) * state_stride_p2 + (long long)((tid_r - tid_r / 8 * 8) * 4);
                    if (philox_rounds != 0) {
#if __CUDA_ARCH__ == 900
                        bool nan_1 = 0;
                        uint32_t _setp_660_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_660_bits) : "f"(S[0]), "f"(S[1]));
                        bool _setp_660 = _setp_660_bits != 0;
                        nan_1 = nan_1 || _setp_660;
                        uint32_t _setp_661_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_661_bits) : "f"(S[2]), "f"(S[3]));
                        bool _setp_661 = _setp_661_bits != 0;
                        nan_1 = nan_1 || _setp_661;
                        uint32_t _setp_662_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_662_bits) : "f"(S[4]), "f"(S[5]));
                        bool _setp_662 = _setp_662_bits != 0;
                        nan_1 = nan_1 || _setp_662;
                        uint32_t _setp_663_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_663_bits) : "f"(S[6]), "f"(S[7]));
                        bool _setp_663 = _setp_663_bits != 0;
                        nan_1 = nan_1 || _setp_663;
                        uint32_t _setp_664_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_664_bits) : "f"(S[8]), "f"(S[9]));
                        bool _setp_664 = _setp_664_bits != 0;
                        nan_1 = nan_1 || _setp_664;
                        uint32_t _setp_665_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_665_bits) : "f"(S[10]), "f"(S[11]));
                        bool _setp_665 = _setp_665_bits != 0;
                        nan_1 = nan_1 || _setp_665;
                        uint32_t _setp_666_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_666_bits) : "f"(S[12]), "f"(S[13]));
                        bool _setp_666 = _setp_666_bits != 0;
                        nan_1 = nan_1 || _setp_666;
                        uint32_t _setp_667_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_667_bits) : "f"(S[14]), "f"(S[15]));
                        bool _setp_667 = _setp_667_bits != 0;
                        nan_1 = nan_1 || _setp_667;
                        uint32_t _setp_668_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_668_bits) : "f"(S[16]), "f"(S[17]));
                        bool _setp_668 = _setp_668_bits != 0;
                        nan_1 = nan_1 || _setp_668;
                        uint32_t _setp_669_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_669_bits) : "f"(S[18]), "f"(S[19]));
                        bool _setp_669 = _setp_669_bits != 0;
                        nan_1 = nan_1 || _setp_669;
                        uint32_t _setp_670_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_670_bits) : "f"(S[20]), "f"(S[21]));
                        bool _setp_670 = _setp_670_bits != 0;
                        nan_1 = nan_1 || _setp_670;
                        uint32_t _setp_671_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_671_bits) : "f"(S[22]), "f"(S[23]));
                        bool _setp_671 = _setp_671_bits != 0;
                        nan_1 = nan_1 || _setp_671;
                        uint32_t _setp_672_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_672_bits) : "f"(S[24]), "f"(S[25]));
                        bool _setp_672 = _setp_672_bits != 0;
                        nan_1 = nan_1 || _setp_672;
                        uint32_t _setp_673_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_673_bits) : "f"(S[26]), "f"(S[27]));
                        bool _setp_673 = _setp_673_bits != 0;
                        nan_1 = nan_1 || _setp_673;
                        uint32_t _setp_674_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_674_bits) : "f"(S[28]), "f"(S[29]));
                        bool _setp_674 = _setp_674_bits != 0;
                        nan_1 = nan_1 || _setp_674;
                        uint32_t _setp_675_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_675_bits) : "f"(S[30]), "f"(S[31]));
                        bool _setp_675 = _setp_675_bits != 0;
                        nan_1 = nan_1 || _setp_675;
                        int _warp_vote_676 = __any_sync(__activemask(), nan_1);
                        if (__builtin_expect(_warp_vote_676 == 0, 1)) {
                            unsigned int key0_1 = seed_lo;
                            unsigned int key1_1 = seed_hi;
                            long long o_4 = noise_at;
                            unsigned long long p0_4 = (unsigned long long)(unsigned int)o_4 * 3528531795ull;
                            h_noise[0] = (unsigned int)(o_4 >> 32ll) ^ key0_1;
                            h_noise[2] = (unsigned int)(p0_4 >> 32ull) ^ key1_1;
                            h_noise[3] = (unsigned int)p0_4;
                            long long o_5 = noise_at + 32ll;
                            unsigned long long p0_5 = (unsigned long long)(unsigned int)o_5 * 3528531795ull;
                            h_noise[4] = (unsigned int)(o_5 >> 32ll) ^ key0_1;
                            h_noise[6] = (unsigned int)(p0_5 >> 32ull) ^ key1_1;
                            h_noise[7] = (unsigned int)p0_5;
                            long long o_6 = noise_at + state_stride_p2;
                            unsigned long long p0_6 = (unsigned long long)(unsigned int)o_6 * 3528531795ull;
                            h_noise[8] = (unsigned int)(o_6 >> 32ll) ^ key0_1;
                            h_noise[10] = (unsigned int)(p0_6 >> 32ull) ^ key1_1;
                            h_noise[11] = (unsigned int)p0_6;
                            long long o_7 = noise_at + state_stride_p2 + 32ll;
                            unsigned long long p0_7 = (unsigned long long)(unsigned int)o_7 * 3528531795ull;
                            h_noise[12] = (unsigned int)(o_7 >> 32ll) ^ key0_1;
                            h_noise[14] = (unsigned int)(p0_7 >> 32ull) ^ key1_1;
                            h_noise[15] = (unsigned int)p0_7;
#else
                        unsigned int key0_1 = seed_lo;
                        unsigned int key1_1 = seed_hi;
                        long long o_4 = noise_at;
                        unsigned long long p0_4 = (unsigned long long)(unsigned int)o_4 * 3528531795ull;
                        h_noise[0] = (unsigned int)(o_4 >> 32ll) ^ key0_1;
                        h_noise[2] = (unsigned int)(p0_4 >> 32ull) ^ key1_1;
                        h_noise[3] = (unsigned int)p0_4;
                        long long o_5 = noise_at + 32ll;
                        unsigned long long p0_5 = (unsigned long long)(unsigned int)o_5 * 3528531795ull;
                        h_noise[4] = (unsigned int)(o_5 >> 32ll) ^ key0_1;
                        h_noise[6] = (unsigned int)(p0_5 >> 32ull) ^ key1_1;
                        h_noise[7] = (unsigned int)p0_5;
                        long long o_6 = noise_at + state_stride_p2;
                        unsigned long long p0_6 = (unsigned long long)(unsigned int)o_6 * 3528531795ull;
                        h_noise[8] = (unsigned int)(o_6 >> 32ll) ^ key0_1;
                        h_noise[10] = (unsigned int)(p0_6 >> 32ull) ^ key1_1;
                        h_noise[11] = (unsigned int)p0_6;
                        long long o_7 = noise_at + state_stride_p2 + 32ll;
                        unsigned long long p0_7 = (unsigned long long)(unsigned int)o_7 * 3528531795ull;
                        h_noise[12] = (unsigned int)(o_7 >> 32ll) ^ key0_1;
                        h_noise[14] = (unsigned int)(p0_7 >> 32ull) ^ key1_1;
                        h_noise[15] = (unsigned int)p0_7;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long q0_4 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long q1_4 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(q1_4 >> 32ull) ^ key0_1;
                        h_noise[1] = (unsigned int)q1_4;
                        h_noise[2] = (unsigned int)(q0_4 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)q0_4;
                        unsigned long long q0_5 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long q1_5 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(q1_5 >> 32ull) ^ key0_1;
                        h_noise[5] = (unsigned int)q1_5;
                        h_noise[6] = (unsigned int)(q0_5 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)q0_5;
                        unsigned long long q0_6 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long q1_6 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(q1_6 >> 32ull) ^ key0_1;
                        h_noise[9] = (unsigned int)q1_6;
                        h_noise[10] = (unsigned int)(q0_6 >> 32ull) ^ h_noise[11] ^ key1_1;
                        h_noise[11] = (unsigned int)q0_6;
                        unsigned long long q0_7 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long q1_7 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(q1_7 >> 32ull) ^ key0_1;
                        h_noise[13] = (unsigned int)q1_7;
                        h_noise[14] = (unsigned int)(q0_7 >> 32ull) ^ h_noise[15] ^ key1_1;
                        h_noise[15] = (unsigned int)q0_7;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long s0_12 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_12 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_12 >> 32ull) ^ h_noise[1] ^ key0_1;
                        h_noise[1] = (unsigned int)s1_12;
                        h_noise[2] = (unsigned int)(s0_12 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)s0_12;
                        unsigned long long s0_13 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_13 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_13 >> 32ull) ^ h_noise[5] ^ key0_1;
                        h_noise[5] = (unsigned int)s1_13;
                        h_noise[6] = (unsigned int)(s0_13 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)s0_13;
                        unsigned long long s0_14 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_14 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_14 >> 32ull) ^ h_noise[9] ^ key0_1;
                        h_noise[9] = (unsigned int)s1_14;
                        h_noise[10] = (unsigned int)(s0_14 >> 32ull) ^ h_noise[11] ^ key1_1;
                        h_noise[11] = (unsigned int)s0_14;
                        unsigned long long s0_15 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_15 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_15 >> 32ull) ^ h_noise[13] ^ key0_1;
                        h_noise[13] = (unsigned int)s1_15;
                        h_noise[14] = (unsigned int)(s0_15 >> 32ull) ^ h_noise[15] ^ key1_1;
                        h_noise[15] = (unsigned int)s0_15;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long s0_16 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_16 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_16 >> 32ull) ^ h_noise[1] ^ key0_1;
                        h_noise[1] = (unsigned int)s1_16;
                        h_noise[2] = (unsigned int)(s0_16 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)s0_16;
                        unsigned long long s0_17 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_17 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_17 >> 32ull) ^ h_noise[5] ^ key0_1;
                        h_noise[5] = (unsigned int)s1_17;
                        h_noise[6] = (unsigned int)(s0_17 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)s0_17;
                        unsigned long long s0_18 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_18 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_18 >> 32ull) ^ h_noise[9] ^ key0_1;
                        h_noise[9] = (unsigned int)s1_18;
                        h_noise[10] = (unsigned int)(s0_18 >> 32ull) ^ h_noise[11] ^ key1_1;
                        h_noise[11] = (unsigned int)s0_18;
                        unsigned long long s0_19 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_19 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_19 >> 32ull) ^ h_noise[13] ^ key0_1;
                        h_noise[13] = (unsigned int)s1_19;
                        h_noise[14] = (unsigned int)(s0_19 >> 32ull) ^ h_noise[15] ^ key1_1;
                        h_noise[15] = (unsigned int)s0_19;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long s0_20 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_20 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_20 >> 32ull) ^ h_noise[1] ^ key0_1;
                        h_noise[1] = (unsigned int)s1_20;
                        h_noise[2] = (unsigned int)(s0_20 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)s0_20;
                        unsigned long long s0_21 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_21 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_21 >> 32ull) ^ h_noise[5] ^ key0_1;
                        h_noise[5] = (unsigned int)s1_21;
                        h_noise[6] = (unsigned int)(s0_21 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)s0_21;
                        unsigned long long s0_22 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_22 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_22 >> 32ull) ^ h_noise[9] ^ key0_1;
                        h_noise[9] = (unsigned int)s1_22;
                        h_noise[10] = (unsigned int)(s0_22 >> 32ull) ^ h_noise[11] ^ key1_1;
                        h_noise[11] = (unsigned int)s0_22;
                        unsigned long long s0_23 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_23 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_23 >> 32ull) ^ h_noise[13] ^ key0_1;
                        h_noise[13] = (unsigned int)s1_23;
                        h_noise[14] = (unsigned int)(s0_23 >> 32ull) ^ h_noise[15] ^ key1_1;
                        h_noise[15] = (unsigned int)s0_23;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        #pragma unroll 1
                        for (int _round_1 = 5; _round_1 < philox_rounds; _round_1++) {
                            unsigned long long t0_4 = (unsigned long long)h_noise[0] * 3528531795ull;
                            unsigned long long t1_4 = (unsigned long long)h_noise[2] * 3449720151ull;
                            h_noise[0] = (unsigned int)(t1_4 >> 32ull) ^ h_noise[1] ^ key0_1;
                            h_noise[1] = (unsigned int)t1_4;
                            h_noise[2] = (unsigned int)(t0_4 >> 32ull) ^ h_noise[3] ^ key1_1;
                            h_noise[3] = (unsigned int)t0_4;
                            unsigned long long t0_5 = (unsigned long long)h_noise[4] * 3528531795ull;
                            unsigned long long t1_5 = (unsigned long long)h_noise[6] * 3449720151ull;
                            h_noise[4] = (unsigned int)(t1_5 >> 32ull) ^ h_noise[5] ^ key0_1;
                            h_noise[5] = (unsigned int)t1_5;
                            h_noise[6] = (unsigned int)(t0_5 >> 32ull) ^ h_noise[7] ^ key1_1;
                            h_noise[7] = (unsigned int)t0_5;
                            unsigned long long t0_6 = (unsigned long long)h_noise[8] * 3528531795ull;
                            unsigned long long t1_6 = (unsigned long long)h_noise[10] * 3449720151ull;
                            h_noise[8] = (unsigned int)(t1_6 >> 32ull) ^ h_noise[9] ^ key0_1;
                            h_noise[9] = (unsigned int)t1_6;
                            h_noise[10] = (unsigned int)(t0_6 >> 32ull) ^ h_noise[11] ^ key1_1;
                            h_noise[11] = (unsigned int)t0_6;
                            unsigned long long t0_7 = (unsigned long long)h_noise[12] * 3528531795ull;
                            unsigned long long t1_7 = (unsigned long long)h_noise[14] * 3449720151ull;
                            h_noise[12] = (unsigned int)(t1_7 >> 32ull) ^ h_noise[13] ^ key0_1;
                            h_noise[13] = (unsigned int)t1_7;
                            h_noise[14] = (unsigned int)(t0_7 >> 32ull) ^ h_noise[15] ^ key1_1;
                            h_noise[15] = (unsigned int)t0_7;
#endif
                            key0_1 = key0_1 + 2654435769u;
                            key1_1 = key1_1 + 3144134277u;
#if __CUDA_ARCH__ == 900
                            unsigned long long q0_4 = (unsigned long long)h_noise[0] * 3528531795ull;
                            unsigned long long q1_4 = (unsigned long long)h_noise[2] * 3449720151ull;
                            h_noise[0] = (unsigned int)(q1_4 >> 32ull) ^ key0_1;
                            h_noise[1] = (unsigned int)q1_4;
                            h_noise[2] = (unsigned int)(q0_4 >> 32ull) ^ h_noise[3] ^ key1_1;
                            h_noise[3] = (unsigned int)q0_4;
                            unsigned long long q0_5 = (unsigned long long)h_noise[4] * 3528531795ull;
                            unsigned long long q1_5 = (unsigned long long)h_noise[6] * 3449720151ull;
                            h_noise[4] = (unsigned int)(q1_5 >> 32ull) ^ key0_1;
                            h_noise[5] = (unsigned int)q1_5;
                            h_noise[6] = (unsigned int)(q0_5 >> 32ull) ^ h_noise[7] ^ key1_1;
                            h_noise[7] = (unsigned int)q0_5;
                            unsigned long long q0_6 = (unsigned long long)h_noise[8] * 3528531795ull;
                            unsigned long long q1_6 = (unsigned long long)h_noise[10] * 3449720151ull;
                            h_noise[8] = (unsigned int)(q1_6 >> 32ull) ^ key0_1;
                            h_noise[9] = (unsigned int)q1_6;
                            h_noise[10] = (unsigned int)(q0_6 >> 32ull) ^ h_noise[11] ^ key1_1;
                            h_noise[11] = (unsigned int)q0_6;
                            unsigned long long q0_7 = (unsigned long long)h_noise[12] * 3528531795ull;
                            unsigned long long q1_7 = (unsigned long long)h_noise[14] * 3449720151ull;
                            h_noise[12] = (unsigned int)(q1_7 >> 32ull) ^ key0_1;
                            h_noise[13] = (unsigned int)q1_7;
                            h_noise[14] = (unsigned int)(q0_7 >> 32ull) ^ h_noise[15] ^ key1_1;
                            h_noise[15] = (unsigned int)q0_7;
                            key0_1 = key0_1 + 2654435769u;
                            key1_1 = key1_1 + 3144134277u;
                            unsigned long long s0_12 = (unsigned long long)h_noise[0] * 3528531795ull;
                            unsigned long long s1_12 = (unsigned long long)h_noise[2] * 3449720151ull;
                            h_noise[0] = (unsigned int)(s1_12 >> 32ull) ^ h_noise[1] ^ key0_1;
                            h_noise[1] = (unsigned int)s1_12;
                            h_noise[2] = (unsigned int)(s0_12 >> 32ull) ^ h_noise[3] ^ key1_1;
                            h_noise[3] = (unsigned int)s0_12;
                            unsigned long long s0_13 = (unsigned long long)h_noise[4] * 3528531795ull;
                            unsigned long long s1_13 = (unsigned long long)h_noise[6] * 3449720151ull;
                            h_noise[4] = (unsigned int)(s1_13 >> 32ull) ^ h_noise[5] ^ key0_1;
                            h_noise[5] = (unsigned int)s1_13;
                            h_noise[6] = (unsigned int)(s0_13 >> 32ull) ^ h_noise[7] ^ key1_1;
                            h_noise[7] = (unsigned int)s0_13;
                            unsigned long long s0_14 = (unsigned long long)h_noise[8] * 3528531795ull;
                            unsigned long long s1_14 = (unsigned long long)h_noise[10] * 3449720151ull;
                            h_noise[8] = (unsigned int)(s1_14 >> 32ull) ^ h_noise[9] ^ key0_1;
                            h_noise[9] = (unsigned int)s1_14;
                            h_noise[10] = (unsigned int)(s0_14 >> 32ull) ^ h_noise[11] ^ key1_1;
                            h_noise[11] = (unsigned int)s0_14;
                            unsigned long long s0_15 = (unsigned long long)h_noise[12] * 3528531795ull;
                            unsigned long long s1_15 = (unsigned long long)h_noise[14] * 3449720151ull;
                            h_noise[12] = (unsigned int)(s1_15 >> 32ull) ^ h_noise[13] ^ key0_1;
                            h_noise[13] = (unsigned int)s1_15;
                            h_noise[14] = (unsigned int)(s0_15 >> 32ull) ^ h_noise[15] ^ key1_1;
                            h_noise[15] = (unsigned int)s0_15;
                            key0_1 = key0_1 + 2654435769u;
                            key1_1 = key1_1 + 3144134277u;
                            unsigned long long s0_16 = (unsigned long long)h_noise[0] * 3528531795ull;
                            unsigned long long s1_16 = (unsigned long long)h_noise[2] * 3449720151ull;
                            h_noise[0] = (unsigned int)(s1_16 >> 32ull) ^ h_noise[1] ^ key0_1;
                            h_noise[1] = (unsigned int)s1_16;
                            h_noise[2] = (unsigned int)(s0_16 >> 32ull) ^ h_noise[3] ^ key1_1;
                            h_noise[3] = (unsigned int)s0_16;
                            unsigned long long s0_17 = (unsigned long long)h_noise[4] * 3528531795ull;
                            unsigned long long s1_17 = (unsigned long long)h_noise[6] * 3449720151ull;
                            h_noise[4] = (unsigned int)(s1_17 >> 32ull) ^ h_noise[5] ^ key0_1;
                            h_noise[5] = (unsigned int)s1_17;
                            h_noise[6] = (unsigned int)(s0_17 >> 32ull) ^ h_noise[7] ^ key1_1;
                            h_noise[7] = (unsigned int)s0_17;
                            unsigned long long s0_18 = (unsigned long long)h_noise[8] * 3528531795ull;
                            unsigned long long s1_18 = (unsigned long long)h_noise[10] * 3449720151ull;
                            h_noise[8] = (unsigned int)(s1_18 >> 32ull) ^ h_noise[9] ^ key0_1;
                            h_noise[9] = (unsigned int)s1_18;
                            h_noise[10] = (unsigned int)(s0_18 >> 32ull) ^ h_noise[11] ^ key1_1;
                            h_noise[11] = (unsigned int)s0_18;
                            unsigned long long s0_19 = (unsigned long long)h_noise[12] * 3528531795ull;
                            unsigned long long s1_19 = (unsigned long long)h_noise[14] * 3449720151ull;
                            h_noise[12] = (unsigned int)(s1_19 >> 32ull) ^ h_noise[13] ^ key0_1;
                            h_noise[13] = (unsigned int)s1_19;
                            h_noise[14] = (unsigned int)(s0_19 >> 32ull) ^ h_noise[15] ^ key1_1;
                            h_noise[15] = (unsigned int)s0_19;
                            key0_1 = key0_1 + 2654435769u;
                            key1_1 = key1_1 + 3144134277u;
                            unsigned long long s0_20 = (unsigned long long)h_noise[0] * 3528531795ull;
                            unsigned long long s1_20 = (unsigned long long)h_noise[2] * 3449720151ull;
                            h_noise[0] = (unsigned int)(s1_20 >> 32ull) ^ h_noise[1] ^ key0_1;
                            h_noise[1] = (unsigned int)s1_20;
                            h_noise[2] = (unsigned int)(s0_20 >> 32ull) ^ h_noise[3] ^ key1_1;
                            h_noise[3] = (unsigned int)s0_20;
                            unsigned long long s0_21 = (unsigned long long)h_noise[4] * 3528531795ull;
                            unsigned long long s1_21 = (unsigned long long)h_noise[6] * 3449720151ull;
                            h_noise[4] = (unsigned int)(s1_21 >> 32ull) ^ h_noise[5] ^ key0_1;
                            h_noise[5] = (unsigned int)s1_21;
                            h_noise[6] = (unsigned int)(s0_21 >> 32ull) ^ h_noise[7] ^ key1_1;
                            h_noise[7] = (unsigned int)s0_21;
                            unsigned long long s0_22 = (unsigned long long)h_noise[8] * 3528531795ull;
                            unsigned long long s1_22 = (unsigned long long)h_noise[10] * 3449720151ull;
                            h_noise[8] = (unsigned int)(s1_22 >> 32ull) ^ h_noise[9] ^ key0_1;
                            h_noise[9] = (unsigned int)s1_22;
                            h_noise[10] = (unsigned int)(s0_22 >> 32ull) ^ h_noise[11] ^ key1_1;
                            h_noise[11] = (unsigned int)s0_22;
                            unsigned long long s0_23 = (unsigned long long)h_noise[12] * 3528531795ull;
                            unsigned long long s1_23 = (unsigned long long)h_noise[14] * 3449720151ull;
                            h_noise[12] = (unsigned int)(s1_23 >> 32ull) ^ h_noise[13] ^ key0_1;
                            h_noise[13] = (unsigned int)s1_23;
                            h_noise[14] = (unsigned int)(s0_23 >> 32ull) ^ h_noise[15] ^ key1_1;
                            h_noise[15] = (unsigned int)s0_23;
                            key0_1 = key0_1 + 2654435769u;
                            key1_1 = key1_1 + 3144134277u;
                            #pragma unroll 1
                            for (int _round_1 = 5; _round_1 < philox_rounds; _round_1++) {
                                unsigned long long t0_4 = (unsigned long long)h_noise[0] * 3528531795ull;
                                unsigned long long t1_4 = (unsigned long long)h_noise[2] * 3449720151ull;
                                h_noise[0] = (unsigned int)(t1_4 >> 32ull) ^ h_noise[1] ^ key0_1;
                                h_noise[1] = (unsigned int)t1_4;
                                h_noise[2] = (unsigned int)(t0_4 >> 32ull) ^ h_noise[3] ^ key1_1;
                                h_noise[3] = (unsigned int)t0_4;
                                unsigned long long t0_5 = (unsigned long long)h_noise[4] * 3528531795ull;
                                unsigned long long t1_5 = (unsigned long long)h_noise[6] * 3449720151ull;
                                h_noise[4] = (unsigned int)(t1_5 >> 32ull) ^ h_noise[5] ^ key0_1;
                                h_noise[5] = (unsigned int)t1_5;
                                h_noise[6] = (unsigned int)(t0_5 >> 32ull) ^ h_noise[7] ^ key1_1;
                                h_noise[7] = (unsigned int)t0_5;
                                unsigned long long t0_6 = (unsigned long long)h_noise[8] * 3528531795ull;
                                unsigned long long t1_6 = (unsigned long long)h_noise[10] * 3449720151ull;
                                h_noise[8] = (unsigned int)(t1_6 >> 32ull) ^ h_noise[9] ^ key0_1;
                                h_noise[9] = (unsigned int)t1_6;
                                h_noise[10] = (unsigned int)(t0_6 >> 32ull) ^ h_noise[11] ^ key1_1;
                                h_noise[11] = (unsigned int)t0_6;
                                unsigned long long t0_7 = (unsigned long long)h_noise[12] * 3528531795ull;
                                unsigned long long t1_7 = (unsigned long long)h_noise[14] * 3449720151ull;
                                h_noise[12] = (unsigned int)(t1_7 >> 32ull) ^ h_noise[13] ^ key0_1;
                                h_noise[13] = (unsigned int)t1_7;
                                h_noise[14] = (unsigned int)(t0_7 >> 32ull) ^ h_noise[15] ^ key1_1;
                                h_noise[15] = (unsigned int)t0_7;
                                key0_1 = key0_1 + 2654435769u;
                                key1_1 = key1_1 + 3144134277u;
                            }
                            unsigned int slo_15 = __as_u32(mul_f32_rz_noftz(S[0], __uint_as_float(125829120)));
                            unsigned int shi_15 = __as_u32(mul_f32_rz_noftz(S[1], __uint_as_float(125829120)));
                            unsigned int tlo_15 = slo_15 + (h_noise[0] & 8191) & 4294959104u;
                            unsigned int thi_15 = shi_15 + ((h_noise[0] & 536805376) >> 16) & 4294959104u;
                            float flo_15;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_15) : "f"(__uint_as_float(tlo_15)), "f"(__uint_as_float(2004877312)));
                            float fhi_15;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_15) : "f"(__uint_as_float(thi_15)), "f"(__uint_as_float(2004877312)));
                            uint32_t word_30;
                            asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_30) : "f"(fhi_15), "f"(flo_15));
                            h_words[0] = word_30;
                            unsigned int slo_15_1 = __as_u32(mul_f32_rz_noftz(S[2], __uint_as_float(125829120)));
                            unsigned int shi_15_1 = __as_u32(mul_f32_rz_noftz(S[3], __uint_as_float(125829120)));
                            unsigned int tlo_15_1 = slo_15_1 + (h_noise[1] & 8191) & 4294959104u;
                            unsigned int thi_15_1 = shi_15_1 + ((h_noise[1] & 536805376) >> 16) & 4294959104u;
                            float flo_15_1;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_15_1) : "f"(__uint_as_float(tlo_15_1)), "f"(__uint_as_float(2004877312)));
                            float fhi_15_1;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_15_1) : "f"(__uint_as_float(thi_15_1)), "f"(__uint_as_float(2004877312)));
                            uint32_t word_30_1;
                            asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_30_1) : "f"(fhi_15_1), "f"(flo_15_1));
                            h_words[1] = word_30_1;
                            reinterpret_cast<int2*>(state + (write_rows_base))[0] = reinterpret_cast<int2*>(h_words)[0];
                            unsigned int slo_16 = __as_u32(mul_f32_rz_noftz(S[8], __uint_as_float(125829120)));
                            unsigned int shi_16 = __as_u32(mul_f32_rz_noftz(S[9], __uint_as_float(125829120)));
                            unsigned int tlo_16 = slo_16 + (h_noise[2] & 8191) & 4294959104u;
                            unsigned int thi_16 = shi_16 + ((h_noise[2] & 536805376) >> 16) & 4294959104u;
                            float flo_16;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_16) : "f"(__uint_as_float(tlo_16)), "f"(__uint_as_float(2004877312)));
                            float fhi_16;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_16) : "f"(__uint_as_float(thi_16)), "f"(__uint_as_float(2004877312)));
                            uint32_t word_31;
                            asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_31) : "f"(fhi_16), "f"(flo_16));
                            h_words[0] = word_31;
                            unsigned int slo_17 = __as_u32(mul_f32_rz_noftz(S[10], __uint_as_float(125829120)));
                            unsigned int shi_17 = __as_u32(mul_f32_rz_noftz(S[11], __uint_as_float(125829120)));
                            unsigned int tlo_17 = slo_17 + (h_noise[3] & 8191) & 4294959104u;
                            unsigned int thi_17 = shi_17 + ((h_noise[3] & 536805376) >> 16) & 4294959104u;
                            float flo_17;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_17) : "f"(__uint_as_float(tlo_17)), "f"(__uint_as_float(2004877312)));
                            float fhi_17;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_17) : "f"(__uint_as_float(thi_17)), "f"(__uint_as_float(2004877312)));
                            uint32_t word_32;
                            asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_32) : "f"(fhi_17), "f"(flo_17));
                            h_words[1] = word_32;
                            reinterpret_cast<int2*>(state + (write_rows_base + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            unsigned int slo_18 = __as_u32(mul_f32_rz_noftz(S[4], __uint_as_float(125829120)));
                            unsigned int shi_18 = __as_u32(mul_f32_rz_noftz(S[5], __uint_as_float(125829120)));
                            unsigned int tlo_18 = slo_18 + (h_noise[4] & 8191) & 4294959104u;
                            unsigned int thi_18 = shi_18 + ((h_noise[4] & 536805376) >> 16) & 4294959104u;
                            float flo_18;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_18) : "f"(__uint_as_float(tlo_18)), "f"(__uint_as_float(2004877312)));
                            float fhi_18;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_18) : "f"(__uint_as_float(thi_18)), "f"(__uint_as_float(2004877312)));
                            uint32_t word_33;
                            asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_33) : "f"(fhi_18), "f"(flo_18));
                            h_words[0] = word_33;
                            unsigned int slo_19 = __as_u32(mul_f32_rz_noftz(S[6], __uint_as_float(125829120)));
                            unsigned int shi_19 = __as_u32(mul_f32_rz_noftz(S[7], __uint_as_float(125829120)));
                            unsigned int tlo_19 = slo_19 + (h_noise[5] & 8191) & 4294959104u;
                            unsigned int thi_19 = shi_19 + ((h_noise[5] & 536805376) >> 16) & 4294959104u;
                            float flo_19;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_19) : "f"(__uint_as_float(tlo_19)), "f"(__uint_as_float(2004877312)));
                            float fhi_19;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_19) : "f"(__uint_as_float(thi_19)), "f"(__uint_as_float(2004877312)));
                            uint32_t word_34;
                            asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_34) : "f"(fhi_19), "f"(flo_19));
                            h_words[1] = word_34;
                            reinterpret_cast<int2*>(state + (write_rows_base + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            unsigned int slo_20 = __as_u32(mul_f32_rz_noftz(S[12], __uint_as_float(125829120)));
                            unsigned int shi_20 = __as_u32(mul_f32_rz_noftz(S[13], __uint_as_float(125829120)));
                            unsigned int tlo_20 = slo_20 + (h_noise[6] & 8191) & 4294959104u;
                            unsigned int thi_20 = shi_20 + ((h_noise[6] & 536805376) >> 16) & 4294959104u;
                            float flo_20;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_20) : "f"(__uint_as_float(tlo_20)), "f"(__uint_as_float(2004877312)));
                            float fhi_20;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_20) : "f"(__uint_as_float(thi_20)), "f"(__uint_as_float(2004877312)));
                            uint32_t word_35;
                            asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_35) : "f"(fhi_20), "f"(flo_20));
                            h_words[0] = word_35;
                            unsigned int slo_21 = __as_u32(mul_f32_rz_noftz(S[14], __uint_as_float(125829120)));
                            unsigned int shi_21 = __as_u32(mul_f32_rz_noftz(S[15], __uint_as_float(125829120)));
                            unsigned int tlo_21 = slo_21 + (h_noise[7] & 8191) & 4294959104u;
                            unsigned int thi_21 = shi_21 + ((h_noise[7] & 536805376) >> 16) & 4294959104u;
                            float flo_21;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_21) : "f"(__uint_as_float(tlo_21)), "f"(__uint_as_float(2004877312)));
                            float fhi_21;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_21) : "f"(__uint_as_float(thi_21)), "f"(__uint_as_float(2004877312)));
                            uint32_t word_36;
                            asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_36) : "f"(fhi_21), "f"(flo_21));
                            h_words[1] = word_36;
                            reinterpret_cast<int2*>(state + (write_rows_base + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            unsigned int slo_22 = __as_u32(mul_f32_rz_noftz(S[16], __uint_as_float(125829120)));
                            unsigned int shi_22 = __as_u32(mul_f32_rz_noftz(S[17], __uint_as_float(125829120)));
                            unsigned int tlo_22 = slo_22 + (h_noise[8] & 8191) & 4294959104u;
                            unsigned int thi_22 = shi_22 + ((h_noise[8] & 536805376) >> 16) & 4294959104u;
                            float flo_22;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_22) : "f"(__uint_as_float(tlo_22)), "f"(__uint_as_float(2004877312)));
                            float fhi_22;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_22) : "f"(__uint_as_float(thi_22)), "f"(__uint_as_float(2004877312)));
                            uint32_t word_37;
                            asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_37) : "f"(fhi_22), "f"(flo_22));
                            h_words[0] = word_37;
                            unsigned int slo_23 = __as_u32(mul_f32_rz_noftz(S[18], __uint_as_float(125829120)));
                            unsigned int shi_23 = __as_u32(mul_f32_rz_noftz(S[19], __uint_as_float(125829120)));
                            unsigned int tlo_23 = slo_23 + (h_noise[9] & 8191) & 4294959104u;
                            unsigned int thi_23 = shi_23 + ((h_noise[9] & 536805376) >> 16) & 4294959104u;
                            float flo_23;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_23) : "f"(__uint_as_float(tlo_23)), "f"(__uint_as_float(2004877312)));
                            float fhi_23;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_23) : "f"(__uint_as_float(thi_23)), "f"(__uint_as_float(2004877312)));
                            uint32_t word_38;
                            asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_38) : "f"(fhi_23), "f"(flo_23));
                            h_words[1] = word_38;
                            reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                            unsigned int slo_24 = __as_u32(mul_f32_rz_noftz(S[24], __uint_as_float(125829120)));
                            unsigned int shi_24 = __as_u32(mul_f32_rz_noftz(S[25], __uint_as_float(125829120)));
                            unsigned int tlo_24 = slo_24 + (h_noise[10] & 8191) & 4294959104u;
                            unsigned int thi_24 = shi_24 + ((h_noise[10] & 536805376) >> 16) & 4294959104u;
                            float flo_24;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_24) : "f"(__uint_as_float(tlo_24)), "f"(__uint_as_float(2004877312)));
                            float fhi_24;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_24) : "f"(__uint_as_float(thi_24)), "f"(__uint_as_float(2004877312)));
                            uint32_t word_39;
                            asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_39) : "f"(fhi_24), "f"(flo_24));
                            h_words[0] = word_39;
                            unsigned int slo_25 = __as_u32(mul_f32_rz_noftz(S[26], __uint_as_float(125829120)));
                            unsigned int shi_25 = __as_u32(mul_f32_rz_noftz(S[27], __uint_as_float(125829120)));
                            unsigned int tlo_25 = slo_25 + (h_noise[11] & 8191) & 4294959104u;
                            unsigned int thi_25 = shi_25 + ((h_noise[11] & 536805376) >> 16) & 4294959104u;
                            float flo_25;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_25) : "f"(__uint_as_float(tlo_25)), "f"(__uint_as_float(2004877312)));
                            float fhi_25;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_25) : "f"(__uint_as_float(thi_25)), "f"(__uint_as_float(2004877312)));
                            uint32_t word_40;
                            asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_40) : "f"(fhi_25), "f"(flo_25));
                            h_words[1] = word_40;
                            reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            unsigned int slo_26 = __as_u32(mul_f32_rz_noftz(S[20], __uint_as_float(125829120)));
                            unsigned int shi_26 = __as_u32(mul_f32_rz_noftz(S[21], __uint_as_float(125829120)));
                            unsigned int tlo_26 = slo_26 + (h_noise[12] & 8191) & 4294959104u;
                            unsigned int thi_26 = shi_26 + ((h_noise[12] & 536805376) >> 16) & 4294959104u;
                            float flo_26;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_26) : "f"(__uint_as_float(tlo_26)), "f"(__uint_as_float(2004877312)));
                            float fhi_26;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_26) : "f"(__uint_as_float(thi_26)), "f"(__uint_as_float(2004877312)));
                            uint32_t word_41;
                            asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_41) : "f"(fhi_26), "f"(flo_26));
                            h_words[0] = word_41;
                            unsigned int slo_27 = __as_u32(mul_f32_rz_noftz(S[22], __uint_as_float(125829120)));
                            unsigned int shi_27 = __as_u32(mul_f32_rz_noftz(S[23], __uint_as_float(125829120)));
                            unsigned int tlo_27 = slo_27 + (h_noise[13] & 8191) & 4294959104u;
                            unsigned int thi_27 = shi_27 + ((h_noise[13] & 536805376) >> 16) & 4294959104u;
                            float flo_27;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_27) : "f"(__uint_as_float(tlo_27)), "f"(__uint_as_float(2004877312)));
                            float fhi_27;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_27) : "f"(__uint_as_float(thi_27)), "f"(__uint_as_float(2004877312)));
                            uint32_t word_42;
                            asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_42) : "f"(fhi_27), "f"(flo_27));
                            h_words[1] = word_42;
                            reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            unsigned int slo_28 = __as_u32(mul_f32_rz_noftz(S[28], __uint_as_float(125829120)));
                            unsigned int shi_28 = __as_u32(mul_f32_rz_noftz(S[29], __uint_as_float(125829120)));
                            unsigned int tlo_28 = slo_28 + (h_noise[14] & 8191) & 4294959104u;
                            unsigned int thi_28 = shi_28 + ((h_noise[14] & 536805376) >> 16) & 4294959104u;
                            float flo_28;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_28) : "f"(__uint_as_float(tlo_28)), "f"(__uint_as_float(2004877312)));
                            float fhi_28;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_28) : "f"(__uint_as_float(thi_28)), "f"(__uint_as_float(2004877312)));
                            uint32_t word_43;
                            asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_43) : "f"(fhi_28), "f"(flo_28));
                            h_words[0] = word_43;
                            unsigned int slo_29 = __as_u32(mul_f32_rz_noftz(S[30], __uint_as_float(125829120)));
                            unsigned int shi_29 = __as_u32(mul_f32_rz_noftz(S[31], __uint_as_float(125829120)));
                            unsigned int tlo_29 = slo_29 + (h_noise[15] & 8191) & 4294959104u;
                            unsigned int thi_29 = shi_29 + ((h_noise[15] & 536805376) >> 16) & 4294959104u;
                            float flo_29;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(flo_29) : "f"(__uint_as_float(tlo_29)), "f"(__uint_as_float(2004877312)));
                            float fhi_29;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(fhi_29) : "f"(__uint_as_float(thi_29)), "f"(__uint_as_float(2004877312)));
                            uint32_t word_44;
                            asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(word_44) : "f"(fhi_29), "f"(flo_29));
                            h_words[1] = word_44;
                            reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        } else {
                            unsigned int key0_2 = seed_lo;
                            unsigned int key1_2 = seed_hi;
                            long long o_8 = noise_at;
                            unsigned long long p0_8 = (unsigned long long)(unsigned int)o_8 * 3528531795ull;
                            h_noise[0] = (unsigned int)(o_8 >> 32ll) ^ key0_2;
                            h_noise[2] = (unsigned int)(p0_8 >> 32ull) ^ key1_2;
                            h_noise[3] = (unsigned int)p0_8;
                            long long o_9 = noise_at + 32ll;
                            unsigned long long p0_9 = (unsigned long long)(unsigned int)o_9 * 3528531795ull;
                            h_noise[4] = (unsigned int)(o_9 >> 32ll) ^ key0_2;
                            h_noise[6] = (unsigned int)(p0_9 >> 32ull) ^ key1_2;
                            h_noise[7] = (unsigned int)p0_9;
                            long long o_10 = noise_at + state_stride_p2;
                            unsigned long long p0_10 = (unsigned long long)(unsigned int)o_10 * 3528531795ull;
                            h_noise[8] = (unsigned int)(o_10 >> 32ll) ^ key0_2;
                            h_noise[10] = (unsigned int)(p0_10 >> 32ull) ^ key1_2;
                            h_noise[11] = (unsigned int)p0_10;
                            long long o_11 = noise_at + state_stride_p2 + 32ll;
                            unsigned long long p0_11 = (unsigned long long)(unsigned int)o_11 * 3528531795ull;
                            h_noise[12] = (unsigned int)(o_11 >> 32ll) ^ key0_2;
                            h_noise[14] = (unsigned int)(p0_11 >> 32ull) ^ key1_2;
                            h_noise[15] = (unsigned int)p0_11;
                            key0_2 = key0_2 + 2654435769u;
                            key1_2 = key1_2 + 3144134277u;
                            unsigned long long q0_8 = (unsigned long long)h_noise[0] * 3528531795ull;
                            unsigned long long q1_8 = (unsigned long long)h_noise[2] * 3449720151ull;
                            h_noise[0] = (unsigned int)(q1_8 >> 32ull) ^ key0_2;
                            h_noise[1] = (unsigned int)q1_8;
                            h_noise[2] = (unsigned int)(q0_8 >> 32ull) ^ h_noise[3] ^ key1_2;
                            h_noise[3] = (unsigned int)q0_8;
                            unsigned long long q0_9 = (unsigned long long)h_noise[4] * 3528531795ull;
                            unsigned long long q1_9 = (unsigned long long)h_noise[6] * 3449720151ull;
                            h_noise[4] = (unsigned int)(q1_9 >> 32ull) ^ key0_2;
                            h_noise[5] = (unsigned int)q1_9;
                            h_noise[6] = (unsigned int)(q0_9 >> 32ull) ^ h_noise[7] ^ key1_2;
                            h_noise[7] = (unsigned int)q0_9;
                            unsigned long long q0_10 = (unsigned long long)h_noise[8] * 3528531795ull;
                            unsigned long long q1_10 = (unsigned long long)h_noise[10] * 3449720151ull;
                            h_noise[8] = (unsigned int)(q1_10 >> 32ull) ^ key0_2;
                            h_noise[9] = (unsigned int)q1_10;
                            h_noise[10] = (unsigned int)(q0_10 >> 32ull) ^ h_noise[11] ^ key1_2;
                            h_noise[11] = (unsigned int)q0_10;
                            unsigned long long q0_11 = (unsigned long long)h_noise[12] * 3528531795ull;
                            unsigned long long q1_11 = (unsigned long long)h_noise[14] * 3449720151ull;
                            h_noise[12] = (unsigned int)(q1_11 >> 32ull) ^ key0_2;
                            h_noise[13] = (unsigned int)q1_11;
                            h_noise[14] = (unsigned int)(q0_11 >> 32ull) ^ h_noise[15] ^ key1_2;
                            h_noise[15] = (unsigned int)q0_11;
                            key0_2 = key0_2 + 2654435769u;
                            key1_2 = key1_2 + 3144134277u;
                            unsigned long long s0_24 = (unsigned long long)h_noise[0] * 3528531795ull;
                            unsigned long long s1_24 = (unsigned long long)h_noise[2] * 3449720151ull;
                            h_noise[0] = (unsigned int)(s1_24 >> 32ull) ^ h_noise[1] ^ key0_2;
                            h_noise[1] = (unsigned int)s1_24;
                            h_noise[2] = (unsigned int)(s0_24 >> 32ull) ^ h_noise[3] ^ key1_2;
                            h_noise[3] = (unsigned int)s0_24;
                            unsigned long long s0_25 = (unsigned long long)h_noise[4] * 3528531795ull;
                            unsigned long long s1_25 = (unsigned long long)h_noise[6] * 3449720151ull;
                            h_noise[4] = (unsigned int)(s1_25 >> 32ull) ^ h_noise[5] ^ key0_2;
                            h_noise[5] = (unsigned int)s1_25;
                            h_noise[6] = (unsigned int)(s0_25 >> 32ull) ^ h_noise[7] ^ key1_2;
                            h_noise[7] = (unsigned int)s0_25;
                            unsigned long long s0_26 = (unsigned long long)h_noise[8] * 3528531795ull;
                            unsigned long long s1_26 = (unsigned long long)h_noise[10] * 3449720151ull;
                            h_noise[8] = (unsigned int)(s1_26 >> 32ull) ^ h_noise[9] ^ key0_2;
                            h_noise[9] = (unsigned int)s1_26;
                            h_noise[10] = (unsigned int)(s0_26 >> 32ull) ^ h_noise[11] ^ key1_2;
                            h_noise[11] = (unsigned int)s0_26;
                            unsigned long long s0_27 = (unsigned long long)h_noise[12] * 3528531795ull;
                            unsigned long long s1_27 = (unsigned long long)h_noise[14] * 3449720151ull;
                            h_noise[12] = (unsigned int)(s1_27 >> 32ull) ^ h_noise[13] ^ key0_2;
                            h_noise[13] = (unsigned int)s1_27;
                            h_noise[14] = (unsigned int)(s0_27 >> 32ull) ^ h_noise[15] ^ key1_2;
                            h_noise[15] = (unsigned int)s0_27;
                            key0_2 = key0_2 + 2654435769u;
                            key1_2 = key1_2 + 3144134277u;
                            unsigned long long s0_28 = (unsigned long long)h_noise[0] * 3528531795ull;
                            unsigned long long s1_28 = (unsigned long long)h_noise[2] * 3449720151ull;
                            h_noise[0] = (unsigned int)(s1_28 >> 32ull) ^ h_noise[1] ^ key0_2;
                            h_noise[1] = (unsigned int)s1_28;
                            h_noise[2] = (unsigned int)(s0_28 >> 32ull) ^ h_noise[3] ^ key1_2;
                            h_noise[3] = (unsigned int)s0_28;
                            unsigned long long s0_29 = (unsigned long long)h_noise[4] * 3528531795ull;
                            unsigned long long s1_29 = (unsigned long long)h_noise[6] * 3449720151ull;
                            h_noise[4] = (unsigned int)(s1_29 >> 32ull) ^ h_noise[5] ^ key0_2;
                            h_noise[5] = (unsigned int)s1_29;
                            h_noise[6] = (unsigned int)(s0_29 >> 32ull) ^ h_noise[7] ^ key1_2;
                            h_noise[7] = (unsigned int)s0_29;
                            unsigned long long s0_30 = (unsigned long long)h_noise[8] * 3528531795ull;
                            unsigned long long s1_30 = (unsigned long long)h_noise[10] * 3449720151ull;
                            h_noise[8] = (unsigned int)(s1_30 >> 32ull) ^ h_noise[9] ^ key0_2;
                            h_noise[9] = (unsigned int)s1_30;
                            h_noise[10] = (unsigned int)(s0_30 >> 32ull) ^ h_noise[11] ^ key1_2;
                            h_noise[11] = (unsigned int)s0_30;
                            unsigned long long s0_31 = (unsigned long long)h_noise[12] * 3528531795ull;
                            unsigned long long s1_31 = (unsigned long long)h_noise[14] * 3449720151ull;
                            h_noise[12] = (unsigned int)(s1_31 >> 32ull) ^ h_noise[13] ^ key0_2;
                            h_noise[13] = (unsigned int)s1_31;
                            h_noise[14] = (unsigned int)(s0_31 >> 32ull) ^ h_noise[15] ^ key1_2;
                            h_noise[15] = (unsigned int)s0_31;
                            key0_2 = key0_2 + 2654435769u;
                            key1_2 = key1_2 + 3144134277u;
                            unsigned long long s0_32 = (unsigned long long)h_noise[0] * 3528531795ull;
                            unsigned long long s1_32 = (unsigned long long)h_noise[2] * 3449720151ull;
                            h_noise[0] = (unsigned int)(s1_32 >> 32ull) ^ h_noise[1] ^ key0_2;
                            h_noise[1] = (unsigned int)s1_32;
                            h_noise[2] = (unsigned int)(s0_32 >> 32ull) ^ h_noise[3] ^ key1_2;
                            h_noise[3] = (unsigned int)s0_32;
                            unsigned long long s0_33 = (unsigned long long)h_noise[4] * 3528531795ull;
                            unsigned long long s1_33 = (unsigned long long)h_noise[6] * 3449720151ull;
                            h_noise[4] = (unsigned int)(s1_33 >> 32ull) ^ h_noise[5] ^ key0_2;
                            h_noise[5] = (unsigned int)s1_33;
                            h_noise[6] = (unsigned int)(s0_33 >> 32ull) ^ h_noise[7] ^ key1_2;
                            h_noise[7] = (unsigned int)s0_33;
                            unsigned long long s0_34 = (unsigned long long)h_noise[8] * 3528531795ull;
                            unsigned long long s1_34 = (unsigned long long)h_noise[10] * 3449720151ull;
                            h_noise[8] = (unsigned int)(s1_34 >> 32ull) ^ h_noise[9] ^ key0_2;
                            h_noise[9] = (unsigned int)s1_34;
                            h_noise[10] = (unsigned int)(s0_34 >> 32ull) ^ h_noise[11] ^ key1_2;
                            h_noise[11] = (unsigned int)s0_34;
                            unsigned long long s0_35 = (unsigned long long)h_noise[12] * 3528531795ull;
                            unsigned long long s1_35 = (unsigned long long)h_noise[14] * 3449720151ull;
                            h_noise[12] = (unsigned int)(s1_35 >> 32ull) ^ h_noise[13] ^ key0_2;
                            h_noise[13] = (unsigned int)s1_35;
                            h_noise[14] = (unsigned int)(s0_35 >> 32ull) ^ h_noise[15] ^ key1_2;
                            h_noise[15] = (unsigned int)s0_35;
                            key0_2 = key0_2 + 2654435769u;
                            key1_2 = key1_2 + 3144134277u;
                            #pragma unroll 1
                            for (int _round_2 = 5; _round_2 < philox_rounds; _round_2++) {
                                unsigned long long t0_8 = (unsigned long long)h_noise[0] * 3528531795ull;
                                unsigned long long t1_8 = (unsigned long long)h_noise[2] * 3449720151ull;
                                h_noise[0] = (unsigned int)(t1_8 >> 32ull) ^ h_noise[1] ^ key0_2;
                                h_noise[1] = (unsigned int)t1_8;
                                h_noise[2] = (unsigned int)(t0_8 >> 32ull) ^ h_noise[3] ^ key1_2;
                                h_noise[3] = (unsigned int)t0_8;
                                unsigned long long t0_9 = (unsigned long long)h_noise[4] * 3528531795ull;
                                unsigned long long t1_9 = (unsigned long long)h_noise[6] * 3449720151ull;
                                h_noise[4] = (unsigned int)(t1_9 >> 32ull) ^ h_noise[5] ^ key0_2;
                                h_noise[5] = (unsigned int)t1_9;
                                h_noise[6] = (unsigned int)(t0_9 >> 32ull) ^ h_noise[7] ^ key1_2;
                                h_noise[7] = (unsigned int)t0_9;
                                unsigned long long t0_10 = (unsigned long long)h_noise[8] * 3528531795ull;
                                unsigned long long t1_10 = (unsigned long long)h_noise[10] * 3449720151ull;
                                h_noise[8] = (unsigned int)(t1_10 >> 32ull) ^ h_noise[9] ^ key0_2;
                                h_noise[9] = (unsigned int)t1_10;
                                h_noise[10] = (unsigned int)(t0_10 >> 32ull) ^ h_noise[11] ^ key1_2;
                                h_noise[11] = (unsigned int)t0_10;
                                unsigned long long t0_11 = (unsigned long long)h_noise[12] * 3528531795ull;
                                unsigned long long t1_11 = (unsigned long long)h_noise[14] * 3449720151ull;
                                h_noise[12] = (unsigned int)(t1_11 >> 32ull) ^ h_noise[13] ^ key0_2;
                                h_noise[13] = (unsigned int)t1_11;
                                h_noise[14] = (unsigned int)(t0_11 >> 32ull) ^ h_noise[15] ^ key1_2;
                                h_noise[15] = (unsigned int)t0_11;
                                key0_2 = key0_2 + 2654435769u;
                                key1_2 = key1_2 + 3144134277u;
                            }
                            float _abs_677;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_677) : "f"(S[0]));
                            unsigned int s_31 = __as_u32(mul_f32_rz_noftz(_abs_677, __uint_as_float(125829120)));
                            unsigned int _min_678 = ((s_31) < (260046848) ? (s_31) : (260046848));
                            unsigned int h_32 = _min_678 + (h_noise[0] & 8191) >> 13;
                            unsigned int mag_31 = ((s_31 <= 2139095040) ? h_32 : 32256);
                            float _abs_679;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_679) : "f"(S[1]));
                            unsigned int s_31_1 = __as_u32(mul_f32_rz_noftz(_abs_679, __uint_as_float(125829120)));
                            unsigned int _min_680 = ((s_31_1) < (260046848) ? (s_31_1) : (260046848));
                            unsigned int h_33 = _min_680 + ((h_noise[0] & 536805376) >> 16) >> 13;
                            unsigned int mag_31_1 = ((s_31_1 <= 2139095040) ? h_33 : 32256);
                            uint32_t mags_15;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_15) : "r"(mag_31), "r"(mag_31_1));
                            uint32_t signs_15;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_15) : "r"(__as_u32(S[0])), "r"(__as_u32(S[1])));
                            unsigned int word_45 = signs_15 & 2147516416u | mags_15;
                            h_words[0] = word_45;
                            float _abs_681;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_681) : "f"(S[2]));
                            unsigned int s_32 = __as_u32(mul_f32_rz_noftz(_abs_681, __uint_as_float(125829120)));
                            unsigned int _min_682 = ((s_32) < (260046848) ? (s_32) : (260046848));
                            unsigned int h_34 = _min_682 + (h_noise[1] & 8191) >> 13;
                            unsigned int mag_32 = ((s_32 <= 2139095040) ? h_34 : 32256);
                            float _abs_683;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_683) : "f"(S[3]));
                            unsigned int s_33 = __as_u32(mul_f32_rz_noftz(_abs_683, __uint_as_float(125829120)));
                            unsigned int _min_684 = ((s_33) < (260046848) ? (s_33) : (260046848));
                            unsigned int h_35 = _min_684 + ((h_noise[1] & 536805376) >> 16) >> 13;
                            unsigned int mag_33 = ((s_33 <= 2139095040) ? h_35 : 32256);
                            uint32_t mags_15_1;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_15_1) : "r"(mag_32), "r"(mag_33));
                            uint32_t signs_15_1;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_15_1) : "r"(__as_u32(S[2])), "r"(__as_u32(S[3])));
                            unsigned int word_45_1 = signs_15_1 & 2147516416u | mags_15_1;
                            h_words[1] = word_45_1;
                            reinterpret_cast<int2*>(state + (write_rows_base))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_685;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_685) : "f"(S[8]));
                            unsigned int s_34 = __as_u32(mul_f32_rz_noftz(_abs_685, __uint_as_float(125829120)));
                            unsigned int _min_686 = ((s_34) < (260046848) ? (s_34) : (260046848));
                            unsigned int h_36 = _min_686 + (h_noise[2] & 8191) >> 13;
                            unsigned int mag_34 = ((s_34 <= 2139095040) ? h_36 : 32256);
                            float _abs_687;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_687) : "f"(S[9]));
                            unsigned int s_35 = __as_u32(mul_f32_rz_noftz(_abs_687, __uint_as_float(125829120)));
                            unsigned int _min_688 = ((s_35) < (260046848) ? (s_35) : (260046848));
                            unsigned int h_37 = _min_688 + ((h_noise[2] & 536805376) >> 16) >> 13;
                            unsigned int mag_35 = ((s_35 <= 2139095040) ? h_37 : 32256);
                            uint32_t mags_16;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_16) : "r"(mag_34), "r"(mag_35));
                            uint32_t signs_16;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_16) : "r"(__as_u32(S[8])), "r"(__as_u32(S[9])));
                            unsigned int word_46 = signs_16 & 2147516416u | mags_16;
                            h_words[0] = word_46;
                            float _abs_689;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_689) : "f"(S[10]));
                            unsigned int s_36 = __as_u32(mul_f32_rz_noftz(_abs_689, __uint_as_float(125829120)));
                            unsigned int _min_690 = ((s_36) < (260046848) ? (s_36) : (260046848));
                            unsigned int h_38 = _min_690 + (h_noise[3] & 8191) >> 13;
                            unsigned int mag_36 = ((s_36 <= 2139095040) ? h_38 : 32256);
                            float _abs_691;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_691) : "f"(S[11]));
                            unsigned int s_37 = __as_u32(mul_f32_rz_noftz(_abs_691, __uint_as_float(125829120)));
                            unsigned int _min_692 = ((s_37) < (260046848) ? (s_37) : (260046848));
                            unsigned int h_39 = _min_692 + ((h_noise[3] & 536805376) >> 16) >> 13;
                            unsigned int mag_37 = ((s_37 <= 2139095040) ? h_39 : 32256);
                            uint32_t mags_17;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_17) : "r"(mag_36), "r"(mag_37));
                            uint32_t signs_17;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_17) : "r"(__as_u32(S[10])), "r"(__as_u32(S[11])));
                            unsigned int word_47 = signs_17 & 2147516416u | mags_17;
                            h_words[1] = word_47;
                            reinterpret_cast<int2*>(state + (write_rows_base + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_693;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_693) : "f"(S[4]));
                            unsigned int s_38 = __as_u32(mul_f32_rz_noftz(_abs_693, __uint_as_float(125829120)));
                            unsigned int _min_694 = ((s_38) < (260046848) ? (s_38) : (260046848));
                            unsigned int h_40 = _min_694 + (h_noise[4] & 8191) >> 13;
                            unsigned int mag_38 = ((s_38 <= 2139095040) ? h_40 : 32256);
                            float _abs_695;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_695) : "f"(S[5]));
                            unsigned int s_39 = __as_u32(mul_f32_rz_noftz(_abs_695, __uint_as_float(125829120)));
                            unsigned int _min_696 = ((s_39) < (260046848) ? (s_39) : (260046848));
                            unsigned int h_41 = _min_696 + ((h_noise[4] & 536805376) >> 16) >> 13;
                            unsigned int mag_39 = ((s_39 <= 2139095040) ? h_41 : 32256);
                            uint32_t mags_18;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_18) : "r"(mag_38), "r"(mag_39));
                            uint32_t signs_18;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_18) : "r"(__as_u32(S[4])), "r"(__as_u32(S[5])));
                            unsigned int word_48 = signs_18 & 2147516416u | mags_18;
                            h_words[0] = word_48;
                            float _abs_697;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_697) : "f"(S[6]));
                            unsigned int s_40 = __as_u32(mul_f32_rz_noftz(_abs_697, __uint_as_float(125829120)));
                            unsigned int _min_698 = ((s_40) < (260046848) ? (s_40) : (260046848));
                            unsigned int h_42 = _min_698 + (h_noise[5] & 8191) >> 13;
                            unsigned int mag_40 = ((s_40 <= 2139095040) ? h_42 : 32256);
                            float _abs_699;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_699) : "f"(S[7]));
                            unsigned int s_41 = __as_u32(mul_f32_rz_noftz(_abs_699, __uint_as_float(125829120)));
                            unsigned int _min_700 = ((s_41) < (260046848) ? (s_41) : (260046848));
                            unsigned int h_43 = _min_700 + ((h_noise[5] & 536805376) >> 16) >> 13;
                            unsigned int mag_41 = ((s_41 <= 2139095040) ? h_43 : 32256);
                            uint32_t mags_19;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_19) : "r"(mag_40), "r"(mag_41));
                            uint32_t signs_19;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_19) : "r"(__as_u32(S[6])), "r"(__as_u32(S[7])));
                            unsigned int word_49 = signs_19 & 2147516416u | mags_19;
                            h_words[1] = word_49;
                            reinterpret_cast<int2*>(state + (write_rows_base + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_701;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_701) : "f"(S[12]));
                            unsigned int s_42 = __as_u32(mul_f32_rz_noftz(_abs_701, __uint_as_float(125829120)));
                            unsigned int _min_702 = ((s_42) < (260046848) ? (s_42) : (260046848));
                            unsigned int h_44 = _min_702 + (h_noise[6] & 8191) >> 13;
                            unsigned int mag_42 = ((s_42 <= 2139095040) ? h_44 : 32256);
                            float _abs_703;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_703) : "f"(S[13]));
                            unsigned int s_43 = __as_u32(mul_f32_rz_noftz(_abs_703, __uint_as_float(125829120)));
                            unsigned int _min_704 = ((s_43) < (260046848) ? (s_43) : (260046848));
                            unsigned int h_45 = _min_704 + ((h_noise[6] & 536805376) >> 16) >> 13;
                            unsigned int mag_43 = ((s_43 <= 2139095040) ? h_45 : 32256);
                            uint32_t mags_20;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_20) : "r"(mag_42), "r"(mag_43));
                            uint32_t signs_20;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_20) : "r"(__as_u32(S[12])), "r"(__as_u32(S[13])));
                            unsigned int word_50 = signs_20 & 2147516416u | mags_20;
                            h_words[0] = word_50;
                            float _abs_705;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_705) : "f"(S[14]));
                            unsigned int s_44 = __as_u32(mul_f32_rz_noftz(_abs_705, __uint_as_float(125829120)));
                            unsigned int _min_706 = ((s_44) < (260046848) ? (s_44) : (260046848));
                            unsigned int h_46 = _min_706 + (h_noise[7] & 8191) >> 13;
                            unsigned int mag_44 = ((s_44 <= 2139095040) ? h_46 : 32256);
                            float _abs_707;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_707) : "f"(S[15]));
                            unsigned int s_45 = __as_u32(mul_f32_rz_noftz(_abs_707, __uint_as_float(125829120)));
                            unsigned int _min_708 = ((s_45) < (260046848) ? (s_45) : (260046848));
                            unsigned int h_47 = _min_708 + ((h_noise[7] & 536805376) >> 16) >> 13;
                            unsigned int mag_45 = ((s_45 <= 2139095040) ? h_47 : 32256);
                            uint32_t mags_21;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_21) : "r"(mag_44), "r"(mag_45));
                            uint32_t signs_21;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_21) : "r"(__as_u32(S[14])), "r"(__as_u32(S[15])));
                            unsigned int word_51 = signs_21 & 2147516416u | mags_21;
                            h_words[1] = word_51;
                            reinterpret_cast<int2*>(state + (write_rows_base + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_709;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_709) : "f"(S[16]));
                            unsigned int s_46 = __as_u32(mul_f32_rz_noftz(_abs_709, __uint_as_float(125829120)));
                            unsigned int _min_710 = ((s_46) < (260046848) ? (s_46) : (260046848));
                            unsigned int h_48 = _min_710 + (h_noise[8] & 8191) >> 13;
                            unsigned int mag_46 = ((s_46 <= 2139095040) ? h_48 : 32256);
                            float _abs_711;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_711) : "f"(S[17]));
                            unsigned int s_47 = __as_u32(mul_f32_rz_noftz(_abs_711, __uint_as_float(125829120)));
                            unsigned int _min_712 = ((s_47) < (260046848) ? (s_47) : (260046848));
                            unsigned int h_49 = _min_712 + ((h_noise[8] & 536805376) >> 16) >> 13;
                            unsigned int mag_47 = ((s_47 <= 2139095040) ? h_49 : 32256);
                            uint32_t mags_22;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_22) : "r"(mag_46), "r"(mag_47));
                            uint32_t signs_22;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_22) : "r"(__as_u32(S[16])), "r"(__as_u32(S[17])));
                            unsigned int word_52 = signs_22 & 2147516416u | mags_22;
                            h_words[0] = word_52;
                            float _abs_713;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_713) : "f"(S[18]));
                            unsigned int s_48 = __as_u32(mul_f32_rz_noftz(_abs_713, __uint_as_float(125829120)));
                            unsigned int _min_714 = ((s_48) < (260046848) ? (s_48) : (260046848));
                            unsigned int h_50 = _min_714 + (h_noise[9] & 8191) >> 13;
                            unsigned int mag_48 = ((s_48 <= 2139095040) ? h_50 : 32256);
                            float _abs_715;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_715) : "f"(S[19]));
                            unsigned int s_49 = __as_u32(mul_f32_rz_noftz(_abs_715, __uint_as_float(125829120)));
                            unsigned int _min_716 = ((s_49) < (260046848) ? (s_49) : (260046848));
                            unsigned int h_51 = _min_716 + ((h_noise[9] & 536805376) >> 16) >> 13;
                            unsigned int mag_49 = ((s_49 <= 2139095040) ? h_51 : 32256);
                            uint32_t mags_23;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_23) : "r"(mag_48), "r"(mag_49));
                            uint32_t signs_23;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_23) : "r"(__as_u32(S[18])), "r"(__as_u32(S[19])));
                            unsigned int word_53 = signs_23 & 2147516416u | mags_23;
                            h_words[1] = word_53;
                            reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_717;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_717) : "f"(S[24]));
                            unsigned int s_50 = __as_u32(mul_f32_rz_noftz(_abs_717, __uint_as_float(125829120)));
                            unsigned int _min_718 = ((s_50) < (260046848) ? (s_50) : (260046848));
                            unsigned int h_52 = _min_718 + (h_noise[10] & 8191) >> 13;
                            unsigned int mag_50 = ((s_50 <= 2139095040) ? h_52 : 32256);
                            float _abs_719;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_719) : "f"(S[25]));
                            unsigned int s_51 = __as_u32(mul_f32_rz_noftz(_abs_719, __uint_as_float(125829120)));
                            unsigned int _min_720 = ((s_51) < (260046848) ? (s_51) : (260046848));
                            unsigned int h_53 = _min_720 + ((h_noise[10] & 536805376) >> 16) >> 13;
                            unsigned int mag_51 = ((s_51 <= 2139095040) ? h_53 : 32256);
                            uint32_t mags_24;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_24) : "r"(mag_50), "r"(mag_51));
                            uint32_t signs_24;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_24) : "r"(__as_u32(S[24])), "r"(__as_u32(S[25])));
                            unsigned int word_54 = signs_24 & 2147516416u | mags_24;
                            h_words[0] = word_54;
                            float _abs_721;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_721) : "f"(S[26]));
                            unsigned int s_52 = __as_u32(mul_f32_rz_noftz(_abs_721, __uint_as_float(125829120)));
                            unsigned int _min_722 = ((s_52) < (260046848) ? (s_52) : (260046848));
                            unsigned int h_54 = _min_722 + (h_noise[11] & 8191) >> 13;
                            unsigned int mag_52 = ((s_52 <= 2139095040) ? h_54 : 32256);
                            float _abs_723;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_723) : "f"(S[27]));
                            unsigned int s_53 = __as_u32(mul_f32_rz_noftz(_abs_723, __uint_as_float(125829120)));
                            unsigned int _min_724 = ((s_53) < (260046848) ? (s_53) : (260046848));
                            unsigned int h_55 = _min_724 + ((h_noise[11] & 536805376) >> 16) >> 13;
                            unsigned int mag_53 = ((s_53 <= 2139095040) ? h_55 : 32256);
                            uint32_t mags_25;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_25) : "r"(mag_52), "r"(mag_53));
                            uint32_t signs_25;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_25) : "r"(__as_u32(S[26])), "r"(__as_u32(S[27])));
                            unsigned int word_55 = signs_25 & 2147516416u | mags_25;
                            h_words[1] = word_55;
                            reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_725;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_725) : "f"(S[20]));
                            unsigned int s_54 = __as_u32(mul_f32_rz_noftz(_abs_725, __uint_as_float(125829120)));
                            unsigned int _min_726 = ((s_54) < (260046848) ? (s_54) : (260046848));
                            unsigned int h_56 = _min_726 + (h_noise[12] & 8191) >> 13;
                            unsigned int mag_54 = ((s_54 <= 2139095040) ? h_56 : 32256);
                            float _abs_727;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_727) : "f"(S[21]));
                            unsigned int s_55 = __as_u32(mul_f32_rz_noftz(_abs_727, __uint_as_float(125829120)));
                            unsigned int _min_728 = ((s_55) < (260046848) ? (s_55) : (260046848));
                            unsigned int h_57 = _min_728 + ((h_noise[12] & 536805376) >> 16) >> 13;
                            unsigned int mag_55 = ((s_55 <= 2139095040) ? h_57 : 32256);
                            uint32_t mags_26;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_26) : "r"(mag_54), "r"(mag_55));
                            uint32_t signs_26;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_26) : "r"(__as_u32(S[20])), "r"(__as_u32(S[21])));
                            unsigned int word_56 = signs_26 & 2147516416u | mags_26;
                            h_words[0] = word_56;
                            float _abs_729;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_729) : "f"(S[22]));
                            unsigned int s_56 = __as_u32(mul_f32_rz_noftz(_abs_729, __uint_as_float(125829120)));
                            unsigned int _min_730 = ((s_56) < (260046848) ? (s_56) : (260046848));
                            unsigned int h_58 = _min_730 + (h_noise[13] & 8191) >> 13;
                            unsigned int mag_56 = ((s_56 <= 2139095040) ? h_58 : 32256);
                            float _abs_731;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_731) : "f"(S[23]));
                            unsigned int s_57 = __as_u32(mul_f32_rz_noftz(_abs_731, __uint_as_float(125829120)));
                            unsigned int _min_732 = ((s_57) < (260046848) ? (s_57) : (260046848));
                            unsigned int h_59 = _min_732 + ((h_noise[13] & 536805376) >> 16) >> 13;
                            unsigned int mag_57 = ((s_57 <= 2139095040) ? h_59 : 32256);
                            uint32_t mags_27;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_27) : "r"(mag_56), "r"(mag_57));
                            uint32_t signs_27;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_27) : "r"(__as_u32(S[22])), "r"(__as_u32(S[23])));
                            unsigned int word_57 = signs_27 & 2147516416u | mags_27;
                            h_words[1] = word_57;
                            reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_733;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_733) : "f"(S[28]));
                            unsigned int s_58 = __as_u32(mul_f32_rz_noftz(_abs_733, __uint_as_float(125829120)));
                            unsigned int _min_734 = ((s_58) < (260046848) ? (s_58) : (260046848));
                            unsigned int h_60 = _min_734 + (h_noise[14] & 8191) >> 13;
                            unsigned int mag_58 = ((s_58 <= 2139095040) ? h_60 : 32256);
                            float _abs_735;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_735) : "f"(S[29]));
                            unsigned int s_59 = __as_u32(mul_f32_rz_noftz(_abs_735, __uint_as_float(125829120)));
                            unsigned int _min_736 = ((s_59) < (260046848) ? (s_59) : (260046848));
                            unsigned int h_61 = _min_736 + ((h_noise[14] & 536805376) >> 16) >> 13;
                            unsigned int mag_59 = ((s_59 <= 2139095040) ? h_61 : 32256);
                            uint32_t mags_28;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_28) : "r"(mag_58), "r"(mag_59));
                            uint32_t signs_28;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_28) : "r"(__as_u32(S[28])), "r"(__as_u32(S[29])));
                            unsigned int word_58 = signs_28 & 2147516416u | mags_28;
                            h_words[0] = word_58;
                            float _abs_737;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_737) : "f"(S[30]));
                            unsigned int s_60 = __as_u32(mul_f32_rz_noftz(_abs_737, __uint_as_float(125829120)));
                            unsigned int _min_738 = ((s_60) < (260046848) ? (s_60) : (260046848));
                            unsigned int h_62 = _min_738 + (h_noise[15] & 8191) >> 13;
                            unsigned int mag_60 = ((s_60 <= 2139095040) ? h_62 : 32256);
                            float _abs_739;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_739) : "f"(S[31]));
                            unsigned int s_61 = __as_u32(mul_f32_rz_noftz(_abs_739, __uint_as_float(125829120)));
                            unsigned int _min_740 = ((s_61) < (260046848) ? (s_61) : (260046848));
                            unsigned int h_63 = _min_740 + ((h_noise[15] & 536805376) >> 16) >> 13;
                            unsigned int mag_61 = ((s_61 <= 2139095040) ? h_63 : 32256);
                            uint32_t mags_29;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_29) : "r"(mag_60), "r"(mag_61));
                            uint32_t signs_29;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_29) : "r"(__as_u32(S[30])), "r"(__as_u32(S[31])));
                            unsigned int word_59 = signs_29 & 2147516416u | mags_29;
                            h_words[1] = word_59;
                            reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#endif
                        }
#if __CUDA_ARCH__ == 900
#else
                        uint32_t _cvt_rs_x2_f32_107;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_107) : "f"(S2[0].y), "f"(S2[0].x), "r"(h_noise[0] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_107;
                        uint32_t _cvt_rs_x2_f32_108;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_108) : "f"(S2[1].y), "f"(S2[1].x), "r"(h_noise[1] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_108;
                        reinterpret_cast<int2*>(state + (write_rows_base))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_109;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_109) : "f"(S2[4].y), "f"(S2[4].x), "r"(h_noise[2] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_109;
                        uint32_t _cvt_rs_x2_f32_110;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_110) : "f"(S2[5].y), "f"(S2[5].x), "r"(h_noise[3] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_110;
                        reinterpret_cast<int2*>(state + (write_rows_base + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_111;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_111) : "f"(S2[2].y), "f"(S2[2].x), "r"(h_noise[4] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_111;
                        uint32_t _cvt_rs_x2_f32_112;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_112) : "f"(S2[3].y), "f"(S2[3].x), "r"(h_noise[5] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_112;
                        reinterpret_cast<int2*>(state + (write_rows_base + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_113;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_113) : "f"(S2[6].y), "f"(S2[6].x), "r"(h_noise[6] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_113;
                        uint32_t _cvt_rs_x2_f32_114;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_114) : "f"(S2[7].y), "f"(S2[7].x), "r"(h_noise[7] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_114;
                        reinterpret_cast<int2*>(state + (write_rows_base + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_115;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_115) : "f"(S2[8].y), "f"(S2[8].x), "r"(h_noise[8] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_115;
                        uint32_t _cvt_rs_x2_f32_116;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_116) : "f"(S2[9].y), "f"(S2[9].x), "r"(h_noise[9] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_116;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_117;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_117) : "f"(S2[12].y), "f"(S2[12].x), "r"(h_noise[10] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_117;
                        uint32_t _cvt_rs_x2_f32_118;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_118) : "f"(S2[13].y), "f"(S2[13].x), "r"(h_noise[11] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_118;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_119;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_119) : "f"(S2[10].y), "f"(S2[10].x), "r"(h_noise[12] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_119;
                        uint32_t _cvt_rs_x2_f32_120;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_120) : "f"(S2[11].y), "f"(S2[11].x), "r"(h_noise[13] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_120;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_121;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_121) : "f"(S2[14].y), "f"(S2[14].x), "r"(h_noise[14] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_121;
                        uint32_t _cvt_rs_x2_f32_122;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_122) : "f"(S2[15].y), "f"(S2[15].x), "r"(h_noise[15] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_122;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#endif
                    }
                    if (philox_rounds == 0) {
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_741;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_741) : "f"(S[1]), "f"(S[0]));
                        h_words[0] = _cvt_rn_f16x2_f32_741;
                        uint32_t _cvt_rn_f16x2_f32_742;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_742) : "f"(S[3]), "f"(S[2]));
                        h_words[1] = _cvt_rn_f16x2_f32_742;
#else
                        uint32_t _cvt_rn_f16x2_f32_123;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_123) : "f"(S2[0].y), "f"(S2[0].x));
                        h_words[0] = _cvt_rn_f16x2_f32_123;
                        uint32_t _cvt_rn_f16x2_f32_124;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_124) : "f"(S2[1].y), "f"(S2[1].x));
                        h_words[1] = _cvt_rn_f16x2_f32_124;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_743;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_743) : "f"(S[5]), "f"(S[4]));
                        h_words[0] = _cvt_rn_f16x2_f32_743;
                        uint32_t _cvt_rn_f16x2_f32_744;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_744) : "f"(S[7]), "f"(S[6]));
                        h_words[1] = _cvt_rn_f16x2_f32_744;
#else
                        uint32_t _cvt_rn_f16x2_f32_125;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_125) : "f"(S2[2].y), "f"(S2[2].x));
                        h_words[0] = _cvt_rn_f16x2_f32_125;
                        uint32_t _cvt_rn_f16x2_f32_126;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_126) : "f"(S2[3].y), "f"(S2[3].x));
                        h_words[1] = _cvt_rn_f16x2_f32_126;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_745;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_745) : "f"(S[9]), "f"(S[8]));
                        h_words[0] = _cvt_rn_f16x2_f32_745;
                        uint32_t _cvt_rn_f16x2_f32_746;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_746) : "f"(S[11]), "f"(S[10]));
                        h_words[1] = _cvt_rn_f16x2_f32_746;
#else
                        uint32_t _cvt_rn_f16x2_f32_127;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_127) : "f"(S2[4].y), "f"(S2[4].x));
                        h_words[0] = _cvt_rn_f16x2_f32_127;
                        uint32_t _cvt_rn_f16x2_f32_128;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_128) : "f"(S2[5].y), "f"(S2[5].x));
                        h_words[1] = _cvt_rn_f16x2_f32_128;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_747;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_747) : "f"(S[13]), "f"(S[12]));
                        h_words[0] = _cvt_rn_f16x2_f32_747;
                        uint32_t _cvt_rn_f16x2_f32_748;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_748) : "f"(S[15]), "f"(S[14]));
                        h_words[1] = _cvt_rn_f16x2_f32_748;
#else
                        uint32_t _cvt_rn_f16x2_f32_129;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_129) : "f"(S2[6].y), "f"(S2[6].x));
                        h_words[0] = _cvt_rn_f16x2_f32_129;
                        uint32_t _cvt_rn_f16x2_f32_130;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_130) : "f"(S2[7].y), "f"(S2[7].x));
                        h_words[1] = _cvt_rn_f16x2_f32_130;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_749;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_749) : "f"(S[17]), "f"(S[16]));
                        h_words[0] = _cvt_rn_f16x2_f32_749;
                        uint32_t _cvt_rn_f16x2_f32_750;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_750) : "f"(S[19]), "f"(S[18]));
                        h_words[1] = _cvt_rn_f16x2_f32_750;
#else
                        uint32_t _cvt_rn_f16x2_f32_131;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_131) : "f"(S2[8].y), "f"(S2[8].x));
                        h_words[0] = _cvt_rn_f16x2_f32_131;
                        uint32_t _cvt_rn_f16x2_f32_132;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_132) : "f"(S2[9].y), "f"(S2[9].x));
                        h_words[1] = _cvt_rn_f16x2_f32_132;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_751;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_751) : "f"(S[21]), "f"(S[20]));
                        h_words[0] = _cvt_rn_f16x2_f32_751;
                        uint32_t _cvt_rn_f16x2_f32_752;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_752) : "f"(S[23]), "f"(S[22]));
                        h_words[1] = _cvt_rn_f16x2_f32_752;
#else
                        uint32_t _cvt_rn_f16x2_f32_133;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_133) : "f"(S2[10].y), "f"(S2[10].x));
                        h_words[0] = _cvt_rn_f16x2_f32_133;
                        uint32_t _cvt_rn_f16x2_f32_134;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_134) : "f"(S2[11].y), "f"(S2[11].x));
                        h_words[1] = _cvt_rn_f16x2_f32_134;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_753;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_753) : "f"(S[25]), "f"(S[24]));
                        h_words[0] = _cvt_rn_f16x2_f32_753;
                        uint32_t _cvt_rn_f16x2_f32_754;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_754) : "f"(S[27]), "f"(S[26]));
                        h_words[1] = _cvt_rn_f16x2_f32_754;
#else
                        uint32_t _cvt_rn_f16x2_f32_135;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_135) : "f"(S2[12].y), "f"(S2[12].x));
                        h_words[0] = _cvt_rn_f16x2_f32_135;
                        uint32_t _cvt_rn_f16x2_f32_136;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_136) : "f"(S2[13].y), "f"(S2[13].x));
                        h_words[1] = _cvt_rn_f16x2_f32_136;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_755;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_755) : "f"(S[29]), "f"(S[28]));
                        h_words[0] = _cvt_rn_f16x2_f32_755;
                        uint32_t _cvt_rn_f16x2_f32_756;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_756) : "f"(S[31]), "f"(S[30]));
                        h_words[1] = _cvt_rn_f16x2_f32_756;
#else
                        uint32_t _cvt_rn_f16x2_f32_137;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_137) : "f"(S2[14].y), "f"(S2[14].x));
                        h_words[0] = _cvt_rn_f16x2_f32_137;
                        uint32_t _cvt_rn_f16x2_f32_138;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_138) : "f"(S2[15].y), "f"(S2[15].x));
                        h_words[1] = _cvt_rn_f16x2_f32_138;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                    }
                }
            }
        }
    }
}

} // extern "C"
