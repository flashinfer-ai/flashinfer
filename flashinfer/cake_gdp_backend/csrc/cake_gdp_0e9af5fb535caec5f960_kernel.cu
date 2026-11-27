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

#define LAUNCH_MIN_BLOCKS 3

#include <math_constants.h>

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


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
    return r;
}

__device__ __forceinline__ float mul_f32_rz_noftz(float a, float b) {
    float r;
    asm("mul.rz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdp_0e9af5fb535caec5f960(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __half* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, int* __restrict__ ssm_state_indices, long long* __restrict__ rand_seed, CakeFastDivmod heads, int num_tokens, float scale, int disable_state_update, int use_ssm_state_indices, int philox_rounds, long long state_size_p0, long long state_stride_p1, long long state_stride_p2, long long state_stride_p0, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long ssm_state_indices_stride_p0, long long ssm_state_indices_stride_p1, long long out_stride_p0, long long out_stride_p1, long long out_stride_p2)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

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
    unsigned int* sNoise = reinterpret_cast<unsigned int*>(smem_raw + 45376);
    const int sNoise_addr = smem + 45376;

    // === Task calls (declaration order) ===
    uint32_t _fast_div_q_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)blockIdx.x / 2), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)blockIdx.x / 2);
    uint32_t _fast_div_r_0 = (uint32_t)((unsigned int)blockIdx.x / 2) - _fast_div_q_0 * (uint32_t)(heads.divisor);
    int n = (int)_fast_div_q_0;
    int h = (int)_fast_div_r_0;
    int tid_0 = tid;
    int lane_local = lane;
    int warp_local = warp;
    int q_warp_local = warp_local;
    int u = tid_0 - tid_0 / 8 * 8;
    int part_groups = (int)((unsigned int)blockIdx.x % 2) * 16;
    int g = tid_0 / 8 + part_groups;
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
    unsigned int seed_lo = 0;
    unsigned int seed_hi = 0;
    if (philox_rounds != 0) {
        long long seed_value = *rand_seed;
        seed_lo = (unsigned int)seed_value;
        seed_hi = (unsigned int)(seed_value >> 32ll);
    }
    long long read_slot_head = (long long)read_state_slot_raw * state_stride_p0 + (long long)h * state_stride_p1 + (long long)(part_groups * 2) * state_stride_p2;
    float r_q[4];
    float r_k[4];
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
    if (read_state_slot_valid != 0) {
        float S[32];
        float k_regs[16];
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
        #pragma unroll
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
        int fm_1 = 4 + warp_local;
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
        int fm_2 = 8 + warp_local;
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
        int fm_3 = 12 + warp_local;
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
        float bk_0[16];
        float bv_0[8];
        float bb_0[4];
        int bm0_4 = 16 + warp_local;
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
            }
        }
        long long bvi_4 = (long long)n * v_stride_p0 + (long long)bm_4 * v_stride_p1 + (long long)h * v_stride_p2;
        int bvc_4 = lane_local * 2;
        bv_0[0] = (float)v[bvi_4 + (long long)bvc_4];
        bv_0[1] = (float)v[bvi_4 + (long long)bvc_4 + 1ll];
        long long bbi_4 = (long long)n * b_stride_p0 + (long long)bm_4 * b_stride_p1 + (long long)h * b_stride_p2;
        bb_0[0] = (float)b[bbi_4];
        int bm0_5 = 20 + warp_local;
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
        int bm0_6 = 24 + warp_local;
        int bm_6 = bm0_6;
        long long bki_6 = (long long)n * k_stride_p0 + (long long)bm_6 * k_stride_p1 + (long long)h * k_stride_p2;
        {
            uint2 _vld_34;
            _vld_34 = *reinterpret_cast<const uint2*>(k + bki_6 + (long long)k_col0);
            uint32_t* _vpairs_34 = reinterpret_cast<uint32_t*>(&_vld_34);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&bk_0[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) << 16);
                (&bk_0[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) & 0xffff0000u);
            }
        }
        long long bvi_6 = (long long)n * v_stride_p0 + (long long)bm_6 * v_stride_p1 + (long long)h * v_stride_p2;
        int bvc_6 = lane_local * 2;
        bv_0[4] = (float)v[bvi_6 + (long long)bvc_6];
        bv_0[5] = (float)v[bvi_6 + (long long)bvc_6 + 1ll];
        long long bbi_6 = (long long)n * b_stride_p0 + (long long)bm_6 * b_stride_p1 + (long long)h * b_stride_p2;
        bb_0[2] = (float)b[bbi_6];
        int bm0_7 = 28 + warp_local;
        int bm_7 = bm0_7;
        long long bki_7 = (long long)n * k_stride_p0 + (long long)bm_7 * k_stride_p1 + (long long)h * k_stride_p2;
        {
            uint2 _vld_35;
            _vld_35 = *reinterpret_cast<const uint2*>(k + bki_7 + (long long)k_col0);
            uint32_t* _vpairs_35 = reinterpret_cast<uint32_t*>(&_vld_35);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&bk_0[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_35[_pair]) << 16);
                (&bk_0[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_35[_pair]) & 0xffff0000u);
            }
        }
        long long bvi_7 = (long long)n * v_stride_p0 + (long long)bm_7 * v_stride_p1 + (long long)h * v_stride_p2;
        int bvc_7 = lane_local * 2;
        bv_0[6] = (float)v[bvi_7 + (long long)bvc_7];
        bv_0[7] = (float)v[bvi_7 + (long long)bvc_7 + 1ll];
        long long bbi_7 = (long long)n * b_stride_p0 + (long long)bm_7 * b_stride_p1 + (long long)h * b_stride_p2;
        bb_0[3] = (float)b[bbi_7];
        int fm_4 = 16 + warp_local;
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
        #pragma unroll
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
        int fm_5 = 20 + warp_local;
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
        int fm_6 = 24 + warp_local;
        float x0_x_5 = bk_0[8];
        float x0_y_5 = bk_0[9];
        float x1_x_5 = bk_0[10];
        float x1_y_5 = bk_0[11];
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
        x0_x_5 = mul_f32_rn_ftz(x0_x_5, norm_5);
        x0_y_5 = mul_f32_rn_ftz(x0_y_5, norm_5);
        x1_x_5 = mul_f32_rn_ftz(x1_x_5, norm_5);
        x1_y_5 = mul_f32_rn_ftz(x1_y_5, norm_5);
        sK[fm_6 * 128 + k_col0] = x0_x_5;
        sK[fm_6 * 128 + k_col0 + 1] = x0_y_5;
        sK[fm_6 * 128 + k_col0 + 2] = x1_x_5;
        sK[fm_6 * 128 + k_col0 + 3] = x1_y_5;
        int v_col_5 = lane_local * 2;
        sV[fm_6 * 64 + v_col_5] = bv_0[4];
        sV[fm_6 * 64 + v_col_5 + 1] = bv_0[5];
        if (lane_local == 0) {
            float _exp_50 = __expf(-bb_0[2]);
            float _rcp_49 = approx_rcp(1.0f + _exp_50);
            sBeta[fm_6] = _rcp_49;
        }
        int fm_7 = 28 + warp_local;
        float x0_x_6 = bk_0[12];
        float x0_y_6 = bk_0[13];
        float x1_x_6 = bk_0[14];
        float x1_y_6 = bk_0[15];
        float sum_x_6;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_6) : "f"(x0_x_6), "f"(x0_x_6), "f"(0.0f));
        float sum_y_6;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_6) : "f"(x0_y_6), "f"(x0_y_6), "f"(0.0f));
        float _fma_51;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(x1_x_6), "f"(x1_x_6), "f"(sum_x_6));
        sum_x_6 = _fma_51;
        float _fma_52;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(x1_y_6), "f"(x1_y_6), "f"(sum_y_6));
        sum_y_6 = _fma_52;
        sum_x_6 = sum_x_6 + sum_y_6;
        float _warp_reduce_53 = sum_x_6;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_53 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_53, offset);
        float norm_6 = rsqrtf(_warp_reduce_53 + 1e-06f);
        x0_x_6 = mul_f32_rn_ftz(x0_x_6, norm_6);
        x0_y_6 = mul_f32_rn_ftz(x0_y_6, norm_6);
        x1_x_6 = mul_f32_rn_ftz(x1_x_6, norm_6);
        x1_y_6 = mul_f32_rn_ftz(x1_y_6, norm_6);
        sK[fm_7 * 128 + k_col0] = x0_x_6;
        sK[fm_7 * 128 + k_col0 + 1] = x0_y_6;
        sK[fm_7 * 128 + k_col0 + 2] = x1_x_6;
        sK[fm_7 * 128 + k_col0 + 3] = x1_y_6;
        int v_col_6 = lane_local * 2;
        sV[fm_7 * 64 + v_col_6] = bv_0[6];
        sV[fm_7 * 64 + v_col_6 + 1] = bv_0[7];
        if (lane_local == 0) {
            float _exp_55 = __expf(-bb_0[3]);
            float _rcp_54 = approx_rcp(1.0f + _exp_55);
            sBeta[fm_7] = _rcp_54;
        }
        float bk_1[16];
        float bv_1[8];
        float bb_1[4];
        int bm0_8 = 32 + warp_local;
        int bm_8 = bm0_8;
        long long bki_8 = (long long)n * k_stride_p0 + (long long)bm_8 * k_stride_p1 + (long long)h * k_stride_p2;
        {
            uint2 _vld_56;
            _vld_56 = *reinterpret_cast<const uint2*>(k + bki_8 + (long long)k_col0);
            uint32_t* _vpairs_56 = reinterpret_cast<uint32_t*>(&_vld_56);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&bk_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_56[_pair]) << 16);
                (&bk_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_56[_pair]) & 0xffff0000u);
            }
        }
        long long bvi_8 = (long long)n * v_stride_p0 + (long long)bm_8 * v_stride_p1 + (long long)h * v_stride_p2;
        int bvc_8 = lane_local * 2;
        bv_1[0] = (float)v[bvi_8 + (long long)bvc_8];
        bv_1[1] = (float)v[bvi_8 + (long long)bvc_8 + 1ll];
        long long bbi_8 = (long long)n * b_stride_p0 + (long long)bm_8 * b_stride_p1 + (long long)h * b_stride_p2;
        bb_1[0] = (float)b[bbi_8];
        int bm0_9 = 36 + warp_local;
        int bm_9 = bm0_9;
        long long bki_9 = (long long)n * k_stride_p0 + (long long)bm_9 * k_stride_p1 + (long long)h * k_stride_p2;
        {
            uint2 _vld_57;
            _vld_57 = *reinterpret_cast<const uint2*>(k + bki_9 + (long long)k_col0);
            uint32_t* _vpairs_57 = reinterpret_cast<uint32_t*>(&_vld_57);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&bk_1[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_57[_pair]) << 16);
                (&bk_1[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_57[_pair]) & 0xffff0000u);
            }
        }
        long long bvi_9 = (long long)n * v_stride_p0 + (long long)bm_9 * v_stride_p1 + (long long)h * v_stride_p2;
        int bvc_9 = lane_local * 2;
        bv_1[2] = (float)v[bvi_9 + (long long)bvc_9];
        bv_1[3] = (float)v[bvi_9 + (long long)bvc_9 + 1ll];
        long long bbi_9 = (long long)n * b_stride_p0 + (long long)bm_9 * b_stride_p1 + (long long)h * b_stride_p2;
        bb_1[1] = (float)b[bbi_9];
        int bm0_10 = 40 + warp_local;
        int bm_10 = bm0_10;
        long long bki_10 = (long long)n * k_stride_p0 + (long long)bm_10 * k_stride_p1 + (long long)h * k_stride_p2;
        {
            uint2 _vld_58;
            _vld_58 = *reinterpret_cast<const uint2*>(k + bki_10 + (long long)k_col0);
            uint32_t* _vpairs_58 = reinterpret_cast<uint32_t*>(&_vld_58);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&bk_1[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_58[_pair]) << 16);
                (&bk_1[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_58[_pair]) & 0xffff0000u);
            }
        }
        long long bvi_10 = (long long)n * v_stride_p0 + (long long)bm_10 * v_stride_p1 + (long long)h * v_stride_p2;
        int bvc_10 = lane_local * 2;
        bv_1[4] = (float)v[bvi_10 + (long long)bvc_10];
        bv_1[5] = (float)v[bvi_10 + (long long)bvc_10 + 1ll];
        long long bbi_10 = (long long)n * b_stride_p0 + (long long)bm_10 * b_stride_p1 + (long long)h * b_stride_p2;
        bb_1[2] = (float)b[bbi_10];
        int bm0_11 = 44 + warp_local;
        int bm_11 = bm0_11;
        long long bki_11 = (long long)n * k_stride_p0 + (long long)bm_11 * k_stride_p1 + (long long)h * k_stride_p2;
        {
            uint2 _vld_59;
            _vld_59 = *reinterpret_cast<const uint2*>(k + bki_11 + (long long)k_col0);
            uint32_t* _vpairs_59 = reinterpret_cast<uint32_t*>(&_vld_59);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&bk_1[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_59[_pair]) << 16);
                (&bk_1[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_59[_pair]) & 0xffff0000u);
            }
        }
        long long bvi_11 = (long long)n * v_stride_p0 + (long long)bm_11 * v_stride_p1 + (long long)h * v_stride_p2;
        int bvc_11 = lane_local * 2;
        bv_1[6] = (float)v[bvi_11 + (long long)bvc_11];
        bv_1[7] = (float)v[bvi_11 + (long long)bvc_11 + 1ll];
        long long bbi_11 = (long long)n * b_stride_p0 + (long long)bm_11 * b_stride_p1 + (long long)h * b_stride_p2;
        bb_1[3] = (float)b[bbi_11];
        int fm_8 = 32 + warp_local;
        float x0_x_7 = bk_1[0];
        float x0_y_7 = bk_1[1];
        float x1_x_7 = bk_1[2];
        float x1_y_7 = bk_1[3];
        float sum_x_7;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_7) : "f"(x0_x_7), "f"(x0_x_7), "f"(0.0f));
        float sum_y_7;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_7) : "f"(x0_y_7), "f"(x0_y_7), "f"(0.0f));
        float _fma_60;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(x1_x_7), "f"(x1_x_7), "f"(sum_x_7));
        sum_x_7 = _fma_60;
        float _fma_61;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(x1_y_7), "f"(x1_y_7), "f"(sum_y_7));
        sum_y_7 = _fma_61;
        sum_x_7 = sum_x_7 + sum_y_7;
        float _warp_reduce_62 = sum_x_7;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_62 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_62, offset);
        float norm_7 = rsqrtf(_warp_reduce_62 + 1e-06f);
        x0_x_7 = mul_f32_rn_ftz(x0_x_7, norm_7);
        x0_y_7 = mul_f32_rn_ftz(x0_y_7, norm_7);
        x1_x_7 = mul_f32_rn_ftz(x1_x_7, norm_7);
        x1_y_7 = mul_f32_rn_ftz(x1_y_7, norm_7);
        sK[fm_8 * 128 + k_col0] = x0_x_7;
        sK[fm_8 * 128 + k_col0 + 1] = x0_y_7;
        sK[fm_8 * 128 + k_col0 + 2] = x1_x_7;
        sK[fm_8 * 128 + k_col0 + 3] = x1_y_7;
        int v_col_7 = lane_local * 2;
        sV[fm_8 * 64 + v_col_7] = bv_1[0];
        sV[fm_8 * 64 + v_col_7 + 1] = bv_1[1];
        if (lane_local == 0) {
            float _exp_64 = __expf(-bb_1[0]);
            float _rcp_63 = approx_rcp(1.0f + _exp_64);
            sBeta[fm_8] = _rcp_63;
        }
        int fm_9 = 36 + warp_local;
        float x0_x_8 = bk_1[4];
        float x0_y_8 = bk_1[5];
        float x1_x_8 = bk_1[6];
        float x1_y_8 = bk_1[7];
        float sum_x_8;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_8) : "f"(x0_x_8), "f"(x0_x_8), "f"(0.0f));
        float sum_y_8;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_8) : "f"(x0_y_8), "f"(x0_y_8), "f"(0.0f));
        float _fma_65;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(x1_x_8), "f"(x1_x_8), "f"(sum_x_8));
        sum_x_8 = _fma_65;
        float _fma_66;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(x1_y_8), "f"(x1_y_8), "f"(sum_y_8));
        sum_y_8 = _fma_66;
        sum_x_8 = sum_x_8 + sum_y_8;
        float _warp_reduce_67 = sum_x_8;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_67 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_67, offset);
        float norm_8 = rsqrtf(_warp_reduce_67 + 1e-06f);
        x0_x_8 = mul_f32_rn_ftz(x0_x_8, norm_8);
        x0_y_8 = mul_f32_rn_ftz(x0_y_8, norm_8);
        x1_x_8 = mul_f32_rn_ftz(x1_x_8, norm_8);
        x1_y_8 = mul_f32_rn_ftz(x1_y_8, norm_8);
        sK[fm_9 * 128 + k_col0] = x0_x_8;
        sK[fm_9 * 128 + k_col0 + 1] = x0_y_8;
        sK[fm_9 * 128 + k_col0 + 2] = x1_x_8;
        sK[fm_9 * 128 + k_col0 + 3] = x1_y_8;
        int v_col_8 = lane_local * 2;
        sV[fm_9 * 64 + v_col_8] = bv_1[2];
        sV[fm_9 * 64 + v_col_8 + 1] = bv_1[3];
        if (lane_local == 0) {
            float _exp_69 = __expf(-bb_1[1]);
            float _rcp_68 = approx_rcp(1.0f + _exp_69);
            sBeta[fm_9] = _rcp_68;
        }
        int fm_10 = 40 + warp_local;
        float x0_x_9 = bk_1[8];
        float x0_y_9 = bk_1[9];
        float x1_x_9 = bk_1[10];
        float x1_y_9 = bk_1[11];
        float sum_x_9;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_9) : "f"(x0_x_9), "f"(x0_x_9), "f"(0.0f));
        float sum_y_9;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_9) : "f"(x0_y_9), "f"(x0_y_9), "f"(0.0f));
        float _fma_70;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(x1_x_9), "f"(x1_x_9), "f"(sum_x_9));
        sum_x_9 = _fma_70;
        float _fma_71;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(x1_y_9), "f"(x1_y_9), "f"(sum_y_9));
        sum_y_9 = _fma_71;
        sum_x_9 = sum_x_9 + sum_y_9;
        float _warp_reduce_72 = sum_x_9;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_72 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_72, offset);
        float norm_9 = rsqrtf(_warp_reduce_72 + 1e-06f);
        x0_x_9 = mul_f32_rn_ftz(x0_x_9, norm_9);
        x0_y_9 = mul_f32_rn_ftz(x0_y_9, norm_9);
        x1_x_9 = mul_f32_rn_ftz(x1_x_9, norm_9);
        x1_y_9 = mul_f32_rn_ftz(x1_y_9, norm_9);
        sK[fm_10 * 128 + k_col0] = x0_x_9;
        sK[fm_10 * 128 + k_col0 + 1] = x0_y_9;
        sK[fm_10 * 128 + k_col0 + 2] = x1_x_9;
        sK[fm_10 * 128 + k_col0 + 3] = x1_y_9;
        int v_col_9 = lane_local * 2;
        sV[fm_10 * 64 + v_col_9] = bv_1[4];
        sV[fm_10 * 64 + v_col_9 + 1] = bv_1[5];
        if (lane_local == 0) {
            float _exp_74 = __expf(-bb_1[2]);
            float _rcp_73 = approx_rcp(1.0f + _exp_74);
            sBeta[fm_10] = _rcp_73;
        }
        int fm_11 = 44 + warp_local;
        float x0_x_10 = bk_1[12];
        float x0_y_10 = bk_1[13];
        float x1_x_10 = bk_1[14];
        float x1_y_10 = bk_1[15];
        float sum_x_10;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_10) : "f"(x0_x_10), "f"(x0_x_10), "f"(0.0f));
        float sum_y_10;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_10) : "f"(x0_y_10), "f"(x0_y_10), "f"(0.0f));
        float _fma_75;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(x1_x_10), "f"(x1_x_10), "f"(sum_x_10));
        sum_x_10 = _fma_75;
        float _fma_76;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(x1_y_10), "f"(x1_y_10), "f"(sum_y_10));
        sum_y_10 = _fma_76;
        sum_x_10 = sum_x_10 + sum_y_10;
        float _warp_reduce_77 = sum_x_10;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_77 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_77, offset);
        float norm_10 = rsqrtf(_warp_reduce_77 + 1e-06f);
        x0_x_10 = mul_f32_rn_ftz(x0_x_10, norm_10);
        x0_y_10 = mul_f32_rn_ftz(x0_y_10, norm_10);
        x1_x_10 = mul_f32_rn_ftz(x1_x_10, norm_10);
        x1_y_10 = mul_f32_rn_ftz(x1_y_10, norm_10);
        sK[fm_11 * 128 + k_col0] = x0_x_10;
        sK[fm_11 * 128 + k_col0 + 1] = x0_y_10;
        sK[fm_11 * 128 + k_col0 + 2] = x1_x_10;
        sK[fm_11 * 128 + k_col0 + 3] = x1_y_10;
        int v_col_10 = lane_local * 2;
        sV[fm_11 * 64 + v_col_10] = bv_1[6];
        sV[fm_11 * 64 + v_col_10 + 1] = bv_1[7];
        if (lane_local == 0) {
            float _exp_79 = __expf(-bb_1[3]);
            float _rcp_78 = approx_rcp(1.0f + _exp_79);
            sBeta[fm_11] = _rcp_78;
        }
        float bq[16];
        float ba[12];
        int bt0_0 = warp_local;
        int bt_0 = bt0_0;
        long long bqi_0 = (long long)n * q_stride_p0 + (long long)bt_0 * q_stride_p1 + (long long)h * q_stride_p2;
        {
            uint2 _vld_80;
            _vld_80 = *reinterpret_cast<const uint2*>(q + bqi_0 + (long long)k_col0);
            uint32_t* _vpairs_80 = reinterpret_cast<uint32_t*>(&_vld_80);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&bq[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_80[_pair]) << 16);
                (&bq[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_80[_pair]) & 0xffff0000u);
            }
        }
        long long bgi_0 = (long long)n * a_stride_p0 + (long long)bt_0 * a_stride_p1 + (long long)h * a_stride_p2;
        ba[0] = (float)a[bgi_0];
        ba[1] = dt_bias[h];
        ba[2] = A_log[h];
        int bt0_1 = 4 + warp_local;
        int bt_1 = bt0_1;
        long long bqi_1 = (long long)n * q_stride_p0 + (long long)bt_1 * q_stride_p1 + (long long)h * q_stride_p2;
        {
            uint2 _vld_81;
            _vld_81 = *reinterpret_cast<const uint2*>(q + bqi_1 + (long long)k_col0);
            uint32_t* _vpairs_81 = reinterpret_cast<uint32_t*>(&_vld_81);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&bq[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_81[_pair]) << 16);
                (&bq[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_81[_pair]) & 0xffff0000u);
            }
        }
        long long bgi_1 = (long long)n * a_stride_p0 + (long long)bt_1 * a_stride_p1 + (long long)h * a_stride_p2;
        ba[3] = (float)a[bgi_1];
        ba[4] = dt_bias[h];
        ba[5] = A_log[h];
        int bt0_2 = 8 + warp_local;
        int bt_2 = bt0_2;
        long long bqi_2 = (long long)n * q_stride_p0 + (long long)bt_2 * q_stride_p1 + (long long)h * q_stride_p2;
        {
            uint2 _vld_82;
            _vld_82 = *reinterpret_cast<const uint2*>(q + bqi_2 + (long long)k_col0);
            uint32_t* _vpairs_82 = reinterpret_cast<uint32_t*>(&_vld_82);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&bq[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_82[_pair]) << 16);
                (&bq[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_82[_pair]) & 0xffff0000u);
            }
        }
        long long bgi_2 = (long long)n * a_stride_p0 + (long long)bt_2 * a_stride_p1 + (long long)h * a_stride_p2;
        ba[6] = (float)a[bgi_2];
        ba[7] = dt_bias[h];
        ba[8] = A_log[h];
        int bt0_3 = 12 + warp_local;
        int bt_3 = bt0_3;
        long long bqi_3 = (long long)n * q_stride_p0 + (long long)bt_3 * q_stride_p1 + (long long)h * q_stride_p2;
        {
            uint2 _vld_83;
            _vld_83 = *reinterpret_cast<const uint2*>(q + bqi_3 + (long long)k_col0);
            uint32_t* _vpairs_83 = reinterpret_cast<uint32_t*>(&_vld_83);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&bq[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_83[_pair]) << 16);
                (&bq[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_83[_pair]) & 0xffff0000u);
            }
        }
        long long bgi_3 = (long long)n * a_stride_p0 + (long long)bt_3 * a_stride_p1 + (long long)h * a_stride_p2;
        ba[9] = (float)a[bgi_3];
        ba[10] = dt_bias[h];
        ba[11] = A_log[h];
        int ft_0 = warp_local;
        float x0_x_11 = bq[0];
        float x0_y_11 = bq[1];
        float x1_x_11 = bq[2];
        float x1_y_11 = bq[3];
        float sum_x_11;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_11) : "f"(x0_x_11), "f"(x0_x_11), "f"(0.0f));
        float sum_y_11;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_11) : "f"(x0_y_11), "f"(x0_y_11), "f"(0.0f));
        float _fma_84;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(x1_x_11), "f"(x1_x_11), "f"(sum_x_11));
        sum_x_11 = _fma_84;
        float _fma_85;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(x1_y_11), "f"(x1_y_11), "f"(sum_y_11));
        sum_y_11 = _fma_85;
        sum_x_11 = sum_x_11 + sum_y_11;
        float _warp_reduce_86 = sum_x_11;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_86 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_86, offset);
        float norm_11 = rsqrtf(_warp_reduce_86 + 1e-06f);
        norm_11 = norm_11 * scale;
        x0_x_11 = mul_f32_rn_ftz(x0_x_11, norm_11);
        x0_y_11 = mul_f32_rn_ftz(x0_y_11, norm_11);
        x1_x_11 = mul_f32_rn_ftz(x1_x_11, norm_11);
        x1_y_11 = mul_f32_rn_ftz(x1_y_11, norm_11);
        sQ[ft_0 * 128 + k_col0] = x0_x_11;
        sQ[ft_0 * 128 + k_col0 + 1] = x0_y_11;
        sQ[ft_0 * 128 + k_col0 + 2] = x1_x_11;
        sQ[ft_0 * 128 + k_col0 + 3] = x1_y_11;
        if (lane_local == 0) {
            float xv = ba[0] + ba[1];
            float softplus_x = xv;
            if (xv <= 20.0f) {
                float _exp_88 = __expf(xv);
                float _log2_87;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_87) : "f"(1.0f + _exp_88));
                softplus_x = _log2_87 * 0.6931472f;
            }
            float _exp_89 = __expf(ba[2]);
            float g_log = -_exp_89 * softplus_x;
            float _exp_90 = __expf(g_log);
            sAlpha[ft_0] = _exp_90;
            if (use_ssm_state_indices != 0) {
                sSsm[ft_0] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)ft_0 * ssm_state_indices_stride_p1];
            }
            if (use_ssm_state_indices == 0) {
                sSsm[ft_0] = -1;
            }
        }
        int ft_1 = 4 + warp_local;
        float x0_x_12 = bq[4];
        float x0_y_12 = bq[5];
        float x1_x_12 = bq[6];
        float x1_y_12 = bq[7];
        float sum_x_12;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_12) : "f"(x0_x_12), "f"(x0_x_12), "f"(0.0f));
        float sum_y_12;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_12) : "f"(x0_y_12), "f"(x0_y_12), "f"(0.0f));
        float _fma_91;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(x1_x_12), "f"(x1_x_12), "f"(sum_x_12));
        sum_x_12 = _fma_91;
        float _fma_92;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(x1_y_12), "f"(x1_y_12), "f"(sum_y_12));
        sum_y_12 = _fma_92;
        sum_x_12 = sum_x_12 + sum_y_12;
        float _warp_reduce_93 = sum_x_12;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_93 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_93, offset);
        float norm_12 = rsqrtf(_warp_reduce_93 + 1e-06f);
        norm_12 = norm_12 * scale;
        x0_x_12 = mul_f32_rn_ftz(x0_x_12, norm_12);
        x0_y_12 = mul_f32_rn_ftz(x0_y_12, norm_12);
        x1_x_12 = mul_f32_rn_ftz(x1_x_12, norm_12);
        x1_y_12 = mul_f32_rn_ftz(x1_y_12, norm_12);
        sQ[ft_1 * 128 + k_col0] = x0_x_12;
        sQ[ft_1 * 128 + k_col0 + 1] = x0_y_12;
        sQ[ft_1 * 128 + k_col0 + 2] = x1_x_12;
        sQ[ft_1 * 128 + k_col0 + 3] = x1_y_12;
        if (lane_local == 0) {
            float xv_1 = ba[3] + ba[4];
            float softplus_x_1 = xv_1;
            if (xv_1 <= 20.0f) {
                float _exp_95 = __expf(xv_1);
                float _log2_94;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_94) : "f"(1.0f + _exp_95));
                softplus_x_1 = _log2_94 * 0.6931472f;
            }
            float _exp_96 = __expf(ba[5]);
            float g_log_1 = -_exp_96 * softplus_x_1;
            float _exp_97 = __expf(g_log_1);
            sAlpha[ft_1] = _exp_97;
            if (use_ssm_state_indices != 0) {
                sSsm[ft_1] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)ft_1 * ssm_state_indices_stride_p1];
            }
            if (use_ssm_state_indices == 0) {
                sSsm[ft_1] = -1;
            }
        }
        int ft_2 = 8 + warp_local;
        float x0_x_13 = bq[8];
        float x0_y_13 = bq[9];
        float x1_x_13 = bq[10];
        float x1_y_13 = bq[11];
        float sum_x_13;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_13) : "f"(x0_x_13), "f"(x0_x_13), "f"(0.0f));
        float sum_y_13;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_13) : "f"(x0_y_13), "f"(x0_y_13), "f"(0.0f));
        float _fma_98;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(x1_x_13), "f"(x1_x_13), "f"(sum_x_13));
        sum_x_13 = _fma_98;
        float _fma_99;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(x1_y_13), "f"(x1_y_13), "f"(sum_y_13));
        sum_y_13 = _fma_99;
        sum_x_13 = sum_x_13 + sum_y_13;
        float _warp_reduce_100 = sum_x_13;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_100 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_100, offset);
        float norm_13 = rsqrtf(_warp_reduce_100 + 1e-06f);
        norm_13 = norm_13 * scale;
        x0_x_13 = mul_f32_rn_ftz(x0_x_13, norm_13);
        x0_y_13 = mul_f32_rn_ftz(x0_y_13, norm_13);
        x1_x_13 = mul_f32_rn_ftz(x1_x_13, norm_13);
        x1_y_13 = mul_f32_rn_ftz(x1_y_13, norm_13);
        sQ[ft_2 * 128 + k_col0] = x0_x_13;
        sQ[ft_2 * 128 + k_col0 + 1] = x0_y_13;
        sQ[ft_2 * 128 + k_col0 + 2] = x1_x_13;
        sQ[ft_2 * 128 + k_col0 + 3] = x1_y_13;
        if (lane_local == 0) {
            float xv_2 = ba[6] + ba[7];
            float softplus_x_2 = xv_2;
            if (xv_2 <= 20.0f) {
                float _exp_102 = __expf(xv_2);
                float _log2_101;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_101) : "f"(1.0f + _exp_102));
                softplus_x_2 = _log2_101 * 0.6931472f;
            }
            float _exp_103 = __expf(ba[8]);
            float g_log_2 = -_exp_103 * softplus_x_2;
            float _exp_104 = __expf(g_log_2);
            sAlpha[ft_2] = _exp_104;
            if (use_ssm_state_indices != 0) {
                sSsm[ft_2] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)ft_2 * ssm_state_indices_stride_p1];
            }
            if (use_ssm_state_indices == 0) {
                sSsm[ft_2] = -1;
            }
        }
        int ft_3 = 12 + warp_local;
        float x0_x_14 = bq[12];
        float x0_y_14 = bq[13];
        float x1_x_14 = bq[14];
        float x1_y_14 = bq[15];
        float sum_x_14;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_14) : "f"(x0_x_14), "f"(x0_x_14), "f"(0.0f));
        float sum_y_14;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_14) : "f"(x0_y_14), "f"(x0_y_14), "f"(0.0f));
        float _fma_105;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(x1_x_14), "f"(x1_x_14), "f"(sum_x_14));
        sum_x_14 = _fma_105;
        float _fma_106;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(x1_y_14), "f"(x1_y_14), "f"(sum_y_14));
        sum_y_14 = _fma_106;
        sum_x_14 = sum_x_14 + sum_y_14;
        float _warp_reduce_107 = sum_x_14;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_107 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_107, offset);
        float norm_14 = rsqrtf(_warp_reduce_107 + 1e-06f);
        norm_14 = norm_14 * scale;
        x0_x_14 = mul_f32_rn_ftz(x0_x_14, norm_14);
        x0_y_14 = mul_f32_rn_ftz(x0_y_14, norm_14);
        x1_x_14 = mul_f32_rn_ftz(x1_x_14, norm_14);
        x1_y_14 = mul_f32_rn_ftz(x1_y_14, norm_14);
        sQ[ft_3 * 128 + k_col0] = x0_x_14;
        sQ[ft_3 * 128 + k_col0 + 1] = x0_y_14;
        sQ[ft_3 * 128 + k_col0 + 2] = x1_x_14;
        sQ[ft_3 * 128 + k_col0 + 3] = x1_y_14;
        if (lane_local == 0) {
            float xv_3 = ba[9] + ba[10];
            float softplus_x_3 = xv_3;
            if (xv_3 <= 20.0f) {
                float _exp_109 = __expf(xv_3);
                float _log2_108;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_108) : "f"(1.0f + _exp_109));
                softplus_x_3 = _log2_108 * 0.6931472f;
            }
            float _exp_110 = __expf(ba[11]);
            float g_log_3 = -_exp_110 * softplus_x_3;
            float _exp_111 = __expf(g_log_3);
            sAlpha[ft_3] = _exp_111;
            if (use_ssm_state_indices != 0) {
                sSsm[ft_3] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)ft_3 * ssm_state_indices_stride_p1];
            }
            if (use_ssm_state_indices == 0) {
                sSsm[ft_3] = -1;
            }
        }
        __syncthreads();
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
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 2048), "r"(h_noise[4] & 536813567), "r"(h_noise[5] & 536813567), "r"(h_noise[6] & 536813567), "r"(h_noise[7] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 4096), "r"(h_noise[8] & 536813567), "r"(h_noise[9] & 536813567), "r"(h_noise[10] & 536813567), "r"(h_noise[11] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 6144), "r"(h_noise[12] & 536813567), "r"(h_noise[13] & 536813567), "r"(h_noise[14] & 536813567), "r"(h_noise[15] & 536813567) : "memory");
            }
            #pragma unroll 1
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
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_0 * 256) + group_byte));
                float beta_0 = sBeta[m_0];
                float acc0_x = 0.0f;
                float acc0_y = 0.0f;
                float acc1_x = 0.0f;
                float acc1_y = 0.0f;
                float _fma_112;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_112) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x));
                acc0_x = _fma_112;
                float _fma_113;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_113) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y));
                acc0_y = _fma_113;
                float _fma_114;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_114) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x));
                acc1_x = _fma_114;
                float _fma_115;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y));
                acc1_y = _fma_115;
                float _fma_116;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x));
                acc0_x = _fma_116;
                float _fma_117;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y));
                acc0_y = _fma_117;
                float _fma_118;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_118) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x));
                acc1_x = _fma_118;
                float _fma_119;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y));
                acc1_y = _fma_119;
                float _fma_120;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x));
                acc0_x = _fma_120;
                float _fma_121;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y));
                acc0_y = _fma_121;
                float _fma_122;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x));
                acc1_x = _fma_122;
                float _fma_123;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_123) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y));
                acc1_y = _fma_123;
                float _fma_124;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x));
                acc0_x = _fma_124;
                float _fma_125;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y));
                acc0_y = _fma_125;
                float _fma_126;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x));
                acc1_x = _fma_126;
                float _fma_127;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y));
                acc1_y = _fma_127;
                float _add_128;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(acc0_x), "f"(acc1_x));
                float _add_129;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_129) : "f"(acc0_y), "f"(acc1_y));
                float dot = _add_128 + _add_129;
                float _shfl_xor_sync_130 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
                dot = dot + _shfl_xor_sync_130;
                float _shfl_xor_sync_131 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
                dot = dot + _shfl_xor_sync_131;
                float _shfl_xor_sync_132 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
                dot = dot + _shfl_xor_sync_132;
                float vn_0 = (v_regs[0] - dot) * beta_0;
                float _fma_133;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(k_regs[0]), "f"(vn_0), "f"(S[0]));
                S[0] = _fma_133;
                float _fma_134;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_134) : "f"(k_regs[1]), "f"(vn_0), "f"(S[1]));
                S[1] = _fma_134;
                float _fma_135;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_135) : "f"(k_regs[2]), "f"(vn_0), "f"(S[2]));
                S[2] = _fma_135;
                float _fma_136;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_136) : "f"(k_regs[3]), "f"(vn_0), "f"(S[3]));
                S[3] = _fma_136;
                float _fma_137;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_137) : "f"(k_regs[4]), "f"(vn_0), "f"(S[4]));
                S[4] = _fma_137;
                float _fma_138;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_138) : "f"(k_regs[5]), "f"(vn_0), "f"(S[5]));
                S[5] = _fma_138;
                float _fma_139;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_139) : "f"(k_regs[6]), "f"(vn_0), "f"(S[6]));
                S[6] = _fma_139;
                float _fma_140;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_140) : "f"(k_regs[7]), "f"(vn_0), "f"(S[7]));
                S[7] = _fma_140;
                float _fma_141;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_141) : "f"(k_regs[8]), "f"(vn_0), "f"(S[8]));
                S[8] = _fma_141;
                float _fma_142;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(k_regs[9]), "f"(vn_0), "f"(S[9]));
                S[9] = _fma_142;
                float _fma_143;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_143) : "f"(k_regs[10]), "f"(vn_0), "f"(S[10]));
                S[10] = _fma_143;
                float _fma_144;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_144) : "f"(k_regs[11]), "f"(vn_0), "f"(S[11]));
                S[11] = _fma_144;
                float _fma_145;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_145) : "f"(k_regs[12]), "f"(vn_0), "f"(S[12]));
                S[12] = _fma_145;
                float _fma_146;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_146) : "f"(k_regs[13]), "f"(vn_0), "f"(S[13]));
                S[13] = _fma_146;
                float _fma_147;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_147) : "f"(k_regs[14]), "f"(vn_0), "f"(S[14]));
                S[14] = _fma_147;
                float _fma_148;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_148) : "f"(k_regs[15]), "f"(vn_0), "f"(S[15]));
                S[15] = _fma_148;
                float acc0_x_0 = 0.0f;
                float acc0_y_0 = 0.0f;
                float acc1_x_0 = 0.0f;
                float acc1_y_0 = 0.0f;
                float _fma_149;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_149) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_0));
                acc0_x_0 = _fma_149;
                float _fma_150;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_150) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_0));
                acc0_y_0 = _fma_150;
                float _fma_151;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_151) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_0));
                acc1_x_0 = _fma_151;
                float _fma_152;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_152) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_0));
                acc1_y_0 = _fma_152;
                float _fma_153;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_153) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_0));
                acc0_x_0 = _fma_153;
                float _fma_154;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_154) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_0));
                acc0_y_0 = _fma_154;
                float _fma_155;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_0));
                acc1_x_0 = _fma_155;
                float _fma_156;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_0));
                acc1_y_0 = _fma_156;
                float _fma_157;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_0));
                acc0_x_0 = _fma_157;
                float _fma_158;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_158) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_0));
                acc0_y_0 = _fma_158;
                float _fma_159;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_159) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_0));
                acc1_x_0 = _fma_159;
                float _fma_160;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_160) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_0));
                acc1_y_0 = _fma_160;
                float _fma_161;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_161) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_0));
                acc0_x_0 = _fma_161;
                float _fma_162;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_162) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_0));
                acc0_y_0 = _fma_162;
                float _fma_163;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_163) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_0));
                acc1_x_0 = _fma_163;
                float _fma_164;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_164) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_0));
                acc1_y_0 = _fma_164;
                float _add_165;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_165) : "f"(acc0_x_0), "f"(acc1_x_0));
                float _add_166;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_166) : "f"(acc0_y_0), "f"(acc1_y_0));
                float dot_0 = _add_165 + _add_166;
                float _shfl_xor_sync_167 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 1);
                dot_0 = dot_0 + _shfl_xor_sync_167;
                float _shfl_xor_sync_168 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 2);
                dot_0 = dot_0 + _shfl_xor_sync_168;
                float _shfl_xor_sync_169 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 4);
                dot_0 = dot_0 + _shfl_xor_sync_169;
                float vn_1 = (v_regs[1] - dot_0) * beta_0;
                float _fma_170;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_170) : "f"(k_regs[0]), "f"(vn_1), "f"(S[16]));
                S[16] = _fma_170;
                float _fma_171;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_171) : "f"(k_regs[1]), "f"(vn_1), "f"(S[17]));
                S[17] = _fma_171;
                float _fma_172;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_172) : "f"(k_regs[2]), "f"(vn_1), "f"(S[18]));
                S[18] = _fma_172;
                float _fma_173;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_173) : "f"(k_regs[3]), "f"(vn_1), "f"(S[19]));
                S[19] = _fma_173;
                float _fma_174;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_174) : "f"(k_regs[4]), "f"(vn_1), "f"(S[20]));
                S[20] = _fma_174;
                float _fma_175;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_175) : "f"(k_regs[5]), "f"(vn_1), "f"(S[21]));
                S[21] = _fma_175;
                float _fma_176;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_176) : "f"(k_regs[6]), "f"(vn_1), "f"(S[22]));
                S[22] = _fma_176;
                float _fma_177;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_177) : "f"(k_regs[7]), "f"(vn_1), "f"(S[23]));
                S[23] = _fma_177;
                float _fma_178;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_178) : "f"(k_regs[8]), "f"(vn_1), "f"(S[24]));
                S[24] = _fma_178;
                float _fma_179;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_179) : "f"(k_regs[9]), "f"(vn_1), "f"(S[25]));
                S[25] = _fma_179;
                float _fma_180;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_180) : "f"(k_regs[10]), "f"(vn_1), "f"(S[26]));
                S[26] = _fma_180;
                float _fma_181;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_181) : "f"(k_regs[11]), "f"(vn_1), "f"(S[27]));
                S[27] = _fma_181;
                float _fma_182;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_182) : "f"(k_regs[12]), "f"(vn_1), "f"(S[28]));
                S[28] = _fma_182;
                float _fma_183;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_183) : "f"(k_regs[13]), "f"(vn_1), "f"(S[29]));
                S[29] = _fma_183;
                float _fma_184;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_184) : "f"(k_regs[14]), "f"(vn_1), "f"(S[30]));
                S[30] = _fma_184;
                float _fma_185;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_185) : "f"(k_regs[15]), "f"(vn_1), "f"(S[31]));
                S[31] = _fma_185;
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
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_1 * 256) + group_byte));
                float beta_1 = sBeta[m_1];
                float acc0_x_1 = 0.0f;
                float acc0_y_1 = 0.0f;
                float acc1_x_1 = 0.0f;
                float acc1_y_1 = 0.0f;
                float _fma_186;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_186) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_1));
                acc0_x_1 = _fma_186;
                float _fma_187;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_187) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_1));
                acc0_y_1 = _fma_187;
                float _fma_188;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_188) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_1));
                acc1_x_1 = _fma_188;
                float _fma_189;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_189) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_1));
                acc1_y_1 = _fma_189;
                float _fma_190;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_190) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_1));
                acc0_x_1 = _fma_190;
                float _fma_191;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_191) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_1));
                acc0_y_1 = _fma_191;
                float _fma_192;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_192) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_1));
                acc1_x_1 = _fma_192;
                float _fma_193;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_193) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_1));
                acc1_y_1 = _fma_193;
                float _fma_194;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_194) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_1));
                acc0_x_1 = _fma_194;
                float _fma_195;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_195) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_1));
                acc0_y_1 = _fma_195;
                float _fma_196;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_196) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_1));
                acc1_x_1 = _fma_196;
                float _fma_197;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_197) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_1));
                acc1_y_1 = _fma_197;
                float _fma_198;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_198) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_1));
                acc0_x_1 = _fma_198;
                float _fma_199;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_199) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_1));
                acc0_y_1 = _fma_199;
                float _fma_200;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_200) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_1));
                acc1_x_1 = _fma_200;
                float _fma_201;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_201) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_1));
                acc1_y_1 = _fma_201;
                float _add_202;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_202) : "f"(acc0_x_1), "f"(acc1_x_1));
                float _add_203;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_203) : "f"(acc0_y_1), "f"(acc1_y_1));
                float dot_1 = _add_202 + _add_203;
                float _shfl_xor_sync_204 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_204;
                float _shfl_xor_sync_205 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_205;
                float _shfl_xor_sync_206 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
                dot_1 = dot_1 + _shfl_xor_sync_206;
                float vn_2 = (v_regs[0] - dot_1) * beta_1;
                float _fma_207;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_207) : "f"(k_regs[0]), "f"(vn_2), "f"(S[0]));
                S[0] = _fma_207;
                float _fma_208;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_208) : "f"(k_regs[1]), "f"(vn_2), "f"(S[1]));
                S[1] = _fma_208;
                float _fma_209;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_209) : "f"(k_regs[2]), "f"(vn_2), "f"(S[2]));
                S[2] = _fma_209;
                float _fma_210;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_210) : "f"(k_regs[3]), "f"(vn_2), "f"(S[3]));
                S[3] = _fma_210;
                float _fma_211;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_211) : "f"(k_regs[4]), "f"(vn_2), "f"(S[4]));
                S[4] = _fma_211;
                float _fma_212;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_212) : "f"(k_regs[5]), "f"(vn_2), "f"(S[5]));
                S[5] = _fma_212;
                float _fma_213;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_213) : "f"(k_regs[6]), "f"(vn_2), "f"(S[6]));
                S[6] = _fma_213;
                float _fma_214;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_214) : "f"(k_regs[7]), "f"(vn_2), "f"(S[7]));
                S[7] = _fma_214;
                float _fma_215;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_215) : "f"(k_regs[8]), "f"(vn_2), "f"(S[8]));
                S[8] = _fma_215;
                float _fma_216;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_216) : "f"(k_regs[9]), "f"(vn_2), "f"(S[9]));
                S[9] = _fma_216;
                float _fma_217;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_217) : "f"(k_regs[10]), "f"(vn_2), "f"(S[10]));
                S[10] = _fma_217;
                float _fma_218;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_218) : "f"(k_regs[11]), "f"(vn_2), "f"(S[11]));
                S[11] = _fma_218;
                float _fma_219;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_219) : "f"(k_regs[12]), "f"(vn_2), "f"(S[12]));
                S[12] = _fma_219;
                float _fma_220;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_220) : "f"(k_regs[13]), "f"(vn_2), "f"(S[13]));
                S[13] = _fma_220;
                float _fma_221;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_221) : "f"(k_regs[14]), "f"(vn_2), "f"(S[14]));
                S[14] = _fma_221;
                float _fma_222;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_222) : "f"(k_regs[15]), "f"(vn_2), "f"(S[15]));
                S[15] = _fma_222;
                float acc0_x_2 = 0.0f;
                float acc0_y_2 = 0.0f;
                float acc1_x_2 = 0.0f;
                float acc1_y_2 = 0.0f;
                float _fma_223;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_223) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_2));
                acc0_x_2 = _fma_223;
                float _fma_224;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_224) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_2));
                acc0_y_2 = _fma_224;
                float _fma_225;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_225) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_2));
                acc1_x_2 = _fma_225;
                float _fma_226;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_226) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_2));
                acc1_y_2 = _fma_226;
                float _fma_227;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_227) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_2));
                acc0_x_2 = _fma_227;
                float _fma_228;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_228) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_2));
                acc0_y_2 = _fma_228;
                float _fma_229;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_229) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_2));
                acc1_x_2 = _fma_229;
                float _fma_230;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_230) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_2));
                acc1_y_2 = _fma_230;
                float _fma_231;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_231) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_2));
                acc0_x_2 = _fma_231;
                float _fma_232;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_232) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_2));
                acc0_y_2 = _fma_232;
                float _fma_233;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_233) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_2));
                acc1_x_2 = _fma_233;
                float _fma_234;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_234) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_2));
                acc1_y_2 = _fma_234;
                float _fma_235;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_235) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_2));
                acc0_x_2 = _fma_235;
                float _fma_236;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_236) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_2));
                acc0_y_2 = _fma_236;
                float _fma_237;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_237) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_2));
                acc1_x_2 = _fma_237;
                float _fma_238;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_238) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_2));
                acc1_y_2 = _fma_238;
                float _add_239;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_239) : "f"(acc0_x_2), "f"(acc1_x_2));
                float _add_240;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_240) : "f"(acc0_y_2), "f"(acc1_y_2));
                float dot_2 = _add_239 + _add_240;
                float _shfl_xor_sync_241 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 1);
                dot_2 = dot_2 + _shfl_xor_sync_241;
                float _shfl_xor_sync_242 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 2);
                dot_2 = dot_2 + _shfl_xor_sync_242;
                float _shfl_xor_sync_243 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 4);
                dot_2 = dot_2 + _shfl_xor_sync_243;
                float vn_3 = (v_regs[1] - dot_2) * beta_1;
                float _fma_244;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_244) : "f"(k_regs[0]), "f"(vn_3), "f"(S[16]));
                S[16] = _fma_244;
                float _fma_245;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_245) : "f"(k_regs[1]), "f"(vn_3), "f"(S[17]));
                S[17] = _fma_245;
                float _fma_246;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_246) : "f"(k_regs[2]), "f"(vn_3), "f"(S[18]));
                S[18] = _fma_246;
                float _fma_247;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_247) : "f"(k_regs[3]), "f"(vn_3), "f"(S[19]));
                S[19] = _fma_247;
                float _fma_248;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_248) : "f"(k_regs[4]), "f"(vn_3), "f"(S[20]));
                S[20] = _fma_248;
                float _fma_249;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_249) : "f"(k_regs[5]), "f"(vn_3), "f"(S[21]));
                S[21] = _fma_249;
                float _fma_250;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_250) : "f"(k_regs[6]), "f"(vn_3), "f"(S[22]));
                S[22] = _fma_250;
                float _fma_251;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_251) : "f"(k_regs[7]), "f"(vn_3), "f"(S[23]));
                S[23] = _fma_251;
                float _fma_252;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_252) : "f"(k_regs[8]), "f"(vn_3), "f"(S[24]));
                S[24] = _fma_252;
                float _fma_253;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_253) : "f"(k_regs[9]), "f"(vn_3), "f"(S[25]));
                S[25] = _fma_253;
                float _fma_254;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_254) : "f"(k_regs[10]), "f"(vn_3), "f"(S[26]));
                S[26] = _fma_254;
                float _fma_255;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_255) : "f"(k_regs[11]), "f"(vn_3), "f"(S[27]));
                S[27] = _fma_255;
                float _fma_256;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_256) : "f"(k_regs[12]), "f"(vn_3), "f"(S[28]));
                S[28] = _fma_256;
                float _fma_257;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_257) : "f"(k_regs[13]), "f"(vn_3), "f"(S[29]));
                S[29] = _fma_257;
                float _fma_258;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_258) : "f"(k_regs[14]), "f"(vn_3), "f"(S[30]));
                S[30] = _fma_258;
                float _fma_259;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_259) : "f"(k_regs[15]), "f"(vn_3), "f"(S[31]));
                S[31] = _fma_259;
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
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_2 * 256) + group_byte));
                float beta_2 = sBeta[m_2];
                float acc0_x_3 = 0.0f;
                float acc0_y_3 = 0.0f;
                float acc1_x_3 = 0.0f;
                float acc1_y_3 = 0.0f;
                float _fma_260;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_260) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_3));
                acc0_x_3 = _fma_260;
                float _fma_261;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_261) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_3));
                acc0_y_3 = _fma_261;
                float _fma_262;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_262) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_3));
                acc1_x_3 = _fma_262;
                float _fma_263;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_263) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_3));
                acc1_y_3 = _fma_263;
                float _fma_264;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_264) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_3));
                acc0_x_3 = _fma_264;
                float _fma_265;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_265) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_3));
                acc0_y_3 = _fma_265;
                float _fma_266;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_266) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_3));
                acc1_x_3 = _fma_266;
                float _fma_267;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_267) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_3));
                acc1_y_3 = _fma_267;
                float _fma_268;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_268) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_3));
                acc0_x_3 = _fma_268;
                float _fma_269;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_269) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_3));
                acc0_y_3 = _fma_269;
                float _fma_270;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_270) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_3));
                acc1_x_3 = _fma_270;
                float _fma_271;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_271) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_3));
                acc1_y_3 = _fma_271;
                float _fma_272;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_272) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_3));
                acc0_x_3 = _fma_272;
                float _fma_273;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_273) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_3));
                acc0_y_3 = _fma_273;
                float _fma_274;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_274) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_3));
                acc1_x_3 = _fma_274;
                float _fma_275;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_275) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_3));
                acc1_y_3 = _fma_275;
                float _add_276;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_276) : "f"(acc0_x_3), "f"(acc1_x_3));
                float _add_277;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_277) : "f"(acc0_y_3), "f"(acc1_y_3));
                float dot_3 = _add_276 + _add_277;
                float _shfl_xor_sync_278 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 1);
                dot_3 = dot_3 + _shfl_xor_sync_278;
                float _shfl_xor_sync_279 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 2);
                dot_3 = dot_3 + _shfl_xor_sync_279;
                float _shfl_xor_sync_280 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 4);
                dot_3 = dot_3 + _shfl_xor_sync_280;
                float vn_4 = (v_regs[0] - dot_3) * beta_2;
                float _fma_281;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_281) : "f"(k_regs[0]), "f"(vn_4), "f"(S[0]));
                S[0] = _fma_281;
                float _fma_282;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_282) : "f"(k_regs[1]), "f"(vn_4), "f"(S[1]));
                S[1] = _fma_282;
                float _fma_283;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_283) : "f"(k_regs[2]), "f"(vn_4), "f"(S[2]));
                S[2] = _fma_283;
                float _fma_284;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_284) : "f"(k_regs[3]), "f"(vn_4), "f"(S[3]));
                S[3] = _fma_284;
                float _fma_285;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_285) : "f"(k_regs[4]), "f"(vn_4), "f"(S[4]));
                S[4] = _fma_285;
                float _fma_286;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_286) : "f"(k_regs[5]), "f"(vn_4), "f"(S[5]));
                S[5] = _fma_286;
                float _fma_287;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_287) : "f"(k_regs[6]), "f"(vn_4), "f"(S[6]));
                S[6] = _fma_287;
                float _fma_288;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_288) : "f"(k_regs[7]), "f"(vn_4), "f"(S[7]));
                S[7] = _fma_288;
                float _fma_289;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_289) : "f"(k_regs[8]), "f"(vn_4), "f"(S[8]));
                S[8] = _fma_289;
                float _fma_290;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_290) : "f"(k_regs[9]), "f"(vn_4), "f"(S[9]));
                S[9] = _fma_290;
                float _fma_291;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_291) : "f"(k_regs[10]), "f"(vn_4), "f"(S[10]));
                S[10] = _fma_291;
                float _fma_292;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_292) : "f"(k_regs[11]), "f"(vn_4), "f"(S[11]));
                S[11] = _fma_292;
                float _fma_293;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_293) : "f"(k_regs[12]), "f"(vn_4), "f"(S[12]));
                S[12] = _fma_293;
                float _fma_294;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_294) : "f"(k_regs[13]), "f"(vn_4), "f"(S[13]));
                S[13] = _fma_294;
                float _fma_295;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_295) : "f"(k_regs[14]), "f"(vn_4), "f"(S[14]));
                S[14] = _fma_295;
                float _fma_296;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_296) : "f"(k_regs[15]), "f"(vn_4), "f"(S[15]));
                S[15] = _fma_296;
                float acc0_x_4 = 0.0f;
                float acc0_y_4 = 0.0f;
                float acc1_x_4 = 0.0f;
                float acc1_y_4 = 0.0f;
                float _fma_297;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_297) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_4));
                acc0_x_4 = _fma_297;
                float _fma_298;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_298) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_4));
                acc0_y_4 = _fma_298;
                float _fma_299;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_299) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_4));
                acc1_x_4 = _fma_299;
                float _fma_300;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_300) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_4));
                acc1_y_4 = _fma_300;
                float _fma_301;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_301) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_4));
                acc0_x_4 = _fma_301;
                float _fma_302;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_302) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_4));
                acc0_y_4 = _fma_302;
                float _fma_303;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_303) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_4));
                acc1_x_4 = _fma_303;
                float _fma_304;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_304) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_4));
                acc1_y_4 = _fma_304;
                float _fma_305;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_305) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_4));
                acc0_x_4 = _fma_305;
                float _fma_306;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_306) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_4));
                acc0_y_4 = _fma_306;
                float _fma_307;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_307) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_4));
                acc1_x_4 = _fma_307;
                float _fma_308;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_308) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_4));
                acc1_y_4 = _fma_308;
                float _fma_309;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_309) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_4));
                acc0_x_4 = _fma_309;
                float _fma_310;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_310) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_4));
                acc0_y_4 = _fma_310;
                float _fma_311;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_311) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_4));
                acc1_x_4 = _fma_311;
                float _fma_312;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_312) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_4));
                acc1_y_4 = _fma_312;
                float _add_313;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_313) : "f"(acc0_x_4), "f"(acc1_x_4));
                float _add_314;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_314) : "f"(acc0_y_4), "f"(acc1_y_4));
                float dot_4 = _add_313 + _add_314;
                float _shfl_xor_sync_315 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 1);
                dot_4 = dot_4 + _shfl_xor_sync_315;
                float _shfl_xor_sync_316 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 2);
                dot_4 = dot_4 + _shfl_xor_sync_316;
                float _shfl_xor_sync_317 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 4);
                dot_4 = dot_4 + _shfl_xor_sync_317;
                float vn_5 = (v_regs[1] - dot_4) * beta_2;
                float _fma_318;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_318) : "f"(k_regs[0]), "f"(vn_5), "f"(S[16]));
                S[16] = _fma_318;
                float _fma_319;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_319) : "f"(k_regs[1]), "f"(vn_5), "f"(S[17]));
                S[17] = _fma_319;
                float _fma_320;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_320) : "f"(k_regs[2]), "f"(vn_5), "f"(S[18]));
                S[18] = _fma_320;
                float _fma_321;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_321) : "f"(k_regs[3]), "f"(vn_5), "f"(S[19]));
                S[19] = _fma_321;
                float _fma_322;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_322) : "f"(k_regs[4]), "f"(vn_5), "f"(S[20]));
                S[20] = _fma_322;
                float _fma_323;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_323) : "f"(k_regs[5]), "f"(vn_5), "f"(S[21]));
                S[21] = _fma_323;
                float _fma_324;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_324) : "f"(k_regs[6]), "f"(vn_5), "f"(S[22]));
                S[22] = _fma_324;
                float _fma_325;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_325) : "f"(k_regs[7]), "f"(vn_5), "f"(S[23]));
                S[23] = _fma_325;
                float _fma_326;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_326) : "f"(k_regs[8]), "f"(vn_5), "f"(S[24]));
                S[24] = _fma_326;
                float _fma_327;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_327) : "f"(k_regs[9]), "f"(vn_5), "f"(S[25]));
                S[25] = _fma_327;
                float _fma_328;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_328) : "f"(k_regs[10]), "f"(vn_5), "f"(S[26]));
                S[26] = _fma_328;
                float _fma_329;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_329) : "f"(k_regs[11]), "f"(vn_5), "f"(S[27]));
                S[27] = _fma_329;
                float _fma_330;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_330) : "f"(k_regs[12]), "f"(vn_5), "f"(S[28]));
                S[28] = _fma_330;
                float _fma_331;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_331) : "f"(k_regs[13]), "f"(vn_5), "f"(S[29]));
                S[29] = _fma_331;
                float _fma_332;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_332) : "f"(k_regs[14]), "f"(vn_5), "f"(S[30]));
                S[30] = _fma_332;
                float _fma_333;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_333) : "f"(k_regs[15]), "f"(vn_5), "f"(S[31]));
                S[31] = _fma_333;
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
                float acc0_x_5 = 0.0f;
                float acc0_y_5 = 0.0f;
                float acc1_x_5 = 0.0f;
                float acc1_y_5 = 0.0f;
                float _fma_334;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_334) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_5));
                acc0_x_5 = _fma_334;
                float _fma_335;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_335) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_5));
                acc0_y_5 = _fma_335;
                float _fma_336;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_336) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_5));
                acc1_x_5 = _fma_336;
                float _fma_337;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_337) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_5));
                acc1_y_5 = _fma_337;
                float _fma_338;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_338) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_5));
                acc0_x_5 = _fma_338;
                float _fma_339;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_339) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_5));
                acc0_y_5 = _fma_339;
                float _fma_340;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_340) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_5));
                acc1_x_5 = _fma_340;
                float _fma_341;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_341) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_5));
                acc1_y_5 = _fma_341;
                float _fma_342;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_342) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_5));
                acc0_x_5 = _fma_342;
                float _fma_343;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_343) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_5));
                acc0_y_5 = _fma_343;
                float _fma_344;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_344) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_5));
                acc1_x_5 = _fma_344;
                float _fma_345;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_345) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_5));
                acc1_y_5 = _fma_345;
                float _fma_346;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_346) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_5));
                acc0_x_5 = _fma_346;
                float _fma_347;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_347) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_5));
                acc0_y_5 = _fma_347;
                float _fma_348;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_348) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_5));
                acc1_x_5 = _fma_348;
                float _fma_349;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_349) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_5));
                acc1_y_5 = _fma_349;
                float _add_350;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_350) : "f"(acc0_x_5), "f"(acc1_x_5));
                float _add_351;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_351) : "f"(acc0_y_5), "f"(acc1_y_5));
                float dot_5 = _add_350 + _add_351;
                float _shfl_xor_sync_352 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 1);
                dot_5 = dot_5 + _shfl_xor_sync_352;
                float _shfl_xor_sync_353 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 2);
                dot_5 = dot_5 + _shfl_xor_sync_353;
                float _shfl_xor_sync_354 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 4);
                dot_5 = dot_5 + _shfl_xor_sync_354;
                o4[0] = dot_5;
                float acc0_x_6 = 0.0f;
                float acc0_y_6 = 0.0f;
                float acc1_x_6 = 0.0f;
                float acc1_y_6 = 0.0f;
                float _fma_355;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_355) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_6));
                acc0_x_6 = _fma_355;
                float _fma_356;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_356) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_6));
                acc0_y_6 = _fma_356;
                float _fma_357;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_357) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_6));
                acc1_x_6 = _fma_357;
                float _fma_358;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_358) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_6));
                acc1_y_6 = _fma_358;
                float _fma_359;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_359) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_6));
                acc0_x_6 = _fma_359;
                float _fma_360;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_360) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_6));
                acc0_y_6 = _fma_360;
                float _fma_361;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_361) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_6));
                acc1_x_6 = _fma_361;
                float _fma_362;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_362) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_6));
                acc1_y_6 = _fma_362;
                float _fma_363;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_363) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_6));
                acc0_x_6 = _fma_363;
                float _fma_364;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_364) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_6));
                acc0_y_6 = _fma_364;
                float _fma_365;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_365) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_6));
                acc1_x_6 = _fma_365;
                float _fma_366;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_366) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_6));
                acc1_y_6 = _fma_366;
                float _fma_367;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_367) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_6));
                acc0_x_6 = _fma_367;
                float _fma_368;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_368) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_6));
                acc0_y_6 = _fma_368;
                float _fma_369;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_369) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_6));
                acc1_x_6 = _fma_369;
                float _fma_370;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_370) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_6));
                acc1_y_6 = _fma_370;
                float _add_371;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_371) : "f"(acc0_x_6), "f"(acc1_x_6));
                float _add_372;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_372) : "f"(acc0_y_6), "f"(acc1_y_6));
                float dot_6 = _add_371 + _add_372;
                float _shfl_xor_sync_373 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 1);
                dot_6 = dot_6 + _shfl_xor_sync_373;
                float _shfl_xor_sync_374 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 2);
                dot_6 = dot_6 + _shfl_xor_sync_374;
                float _shfl_xor_sync_375 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 4);
                dot_6 = dot_6 + _shfl_xor_sync_375;
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
                    bool nan = 0;
                    uint32_t _setp_376_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_376_bits) : "f"(S[0]), "f"(S[1]));
                    bool _setp_376 = _setp_376_bits != 0;
                    nan = nan || _setp_376;
                    uint32_t _setp_377_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_377_bits) : "f"(S[2]), "f"(S[3]));
                    bool _setp_377 = _setp_377_bits != 0;
                    nan = nan || _setp_377;
                    uint32_t _setp_378_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_378_bits) : "f"(S[4]), "f"(S[5]));
                    bool _setp_378 = _setp_378_bits != 0;
                    nan = nan || _setp_378;
                    uint32_t _setp_379_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_379_bits) : "f"(S[6]), "f"(S[7]));
                    bool _setp_379 = _setp_379_bits != 0;
                    nan = nan || _setp_379;
                    uint32_t _setp_380_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_380_bits) : "f"(S[8]), "f"(S[9]));
                    bool _setp_380 = _setp_380_bits != 0;
                    nan = nan || _setp_380;
                    uint32_t _setp_381_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_381_bits) : "f"(S[10]), "f"(S[11]));
                    bool _setp_381 = _setp_381_bits != 0;
                    nan = nan || _setp_381;
                    uint32_t _setp_382_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_382_bits) : "f"(S[12]), "f"(S[13]));
                    bool _setp_382 = _setp_382_bits != 0;
                    nan = nan || _setp_382;
                    uint32_t _setp_383_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_383_bits) : "f"(S[14]), "f"(S[15]));
                    bool _setp_383 = _setp_383_bits != 0;
                    nan = nan || _setp_383;
                    uint32_t _setp_384_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_384_bits) : "f"(S[16]), "f"(S[17]));
                    bool _setp_384 = _setp_384_bits != 0;
                    nan = nan || _setp_384;
                    uint32_t _setp_385_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_385_bits) : "f"(S[18]), "f"(S[19]));
                    bool _setp_385 = _setp_385_bits != 0;
                    nan = nan || _setp_385;
                    uint32_t _setp_386_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_386_bits) : "f"(S[20]), "f"(S[21]));
                    bool _setp_386 = _setp_386_bits != 0;
                    nan = nan || _setp_386;
                    uint32_t _setp_387_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_387_bits) : "f"(S[22]), "f"(S[23]));
                    bool _setp_387 = _setp_387_bits != 0;
                    nan = nan || _setp_387;
                    uint32_t _setp_388_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_388_bits) : "f"(S[24]), "f"(S[25]));
                    bool _setp_388 = _setp_388_bits != 0;
                    nan = nan || _setp_388;
                    uint32_t _setp_389_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_389_bits) : "f"(S[26]), "f"(S[27]));
                    bool _setp_389 = _setp_389_bits != 0;
                    nan = nan || _setp_389;
                    uint32_t _setp_390_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_390_bits) : "f"(S[28]), "f"(S[29]));
                    bool _setp_390 = _setp_390_bits != 0;
                    nan = nan || _setp_390;
                    uint32_t _setp_391_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_391_bits) : "f"(S[30]), "f"(S[31]));
                    bool _setp_391 = _setp_391_bits != 0;
                    nan = nan || _setp_391;
                    int _warp_vote_392 = __any_sync(__activemask(), nan);
                    if (__builtin_expect(_warp_vote_392 == 0, 1)) {
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
                            : "r"(noise_lane + 2048));
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
                            : "r"(noise_lane + 4096));
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
                            : "r"(noise_lane + 6144));
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
                        float _abs_393;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_393) : "f"(S[0]));
                        unsigned int s = __as_u32(mul_f32_rz_noftz(_abs_393, __uint_as_float(125829120)));
                        unsigned int _min_394 = ((s) < (260046848) ? (s) : (260046848));
                        unsigned int h_0 = _min_394 + (h_noise[0] & 8191) >> 13;
                        unsigned int mag = ((s <= 2139095040) ? h_0 : 32256);
                        float _abs_395;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_395) : "f"(S[1]));
                        unsigned int s_0 = __as_u32(mul_f32_rz_noftz(_abs_395, __uint_as_float(125829120)));
                        unsigned int _min_396 = ((s_0) < (260046848) ? (s_0) : (260046848));
                        unsigned int h_1 = _min_396 + ((h_noise[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_0 = ((s_0 <= 2139095040) ? h_1 : 32256);
                        uint32_t mags;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags) : "r"(mag), "r"(mag_0));
                        uint32_t signs;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs) : "r"(__as_u32(S[0])), "r"(__as_u32(S[1])));
                        unsigned int word_15 = signs & 2147516416u | mags;
                        h_words[0] = word_15;
                        float _abs_397;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_397) : "f"(S[2]));
                        unsigned int s_1 = __as_u32(mul_f32_rz_noftz(_abs_397, __uint_as_float(125829120)));
                        unsigned int _min_398 = ((s_1) < (260046848) ? (s_1) : (260046848));
                        unsigned int h_2 = _min_398 + (h_noise[1] & 8191) >> 13;
                        unsigned int mag_1 = ((s_1 <= 2139095040) ? h_2 : 32256);
                        float _abs_399;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_399) : "f"(S[3]));
                        unsigned int s_2 = __as_u32(mul_f32_rz_noftz(_abs_399, __uint_as_float(125829120)));
                        unsigned int _min_400 = ((s_2) < (260046848) ? (s_2) : (260046848));
                        unsigned int h_3 = _min_400 + ((h_noise[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_2 = ((s_2 <= 2139095040) ? h_3 : 32256);
                        uint32_t mags_0;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_0) : "r"(mag_1), "r"(mag_2));
                        uint32_t signs_0;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_0) : "r"(__as_u32(S[2])), "r"(__as_u32(S[3])));
                        unsigned int word_15_1 = signs_0 & 2147516416u | mags_0;
                        h_words[1] = word_15_1;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        float _abs_401;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_401) : "f"(S[8]));
                        unsigned int s_3 = __as_u32(mul_f32_rz_noftz(_abs_401, __uint_as_float(125829120)));
                        unsigned int _min_402 = ((s_3) < (260046848) ? (s_3) : (260046848));
                        unsigned int h_4 = _min_402 + (h_noise[2] & 8191) >> 13;
                        unsigned int mag_3 = ((s_3 <= 2139095040) ? h_4 : 32256);
                        float _abs_403;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_403) : "f"(S[9]));
                        unsigned int s_4 = __as_u32(mul_f32_rz_noftz(_abs_403, __uint_as_float(125829120)));
                        unsigned int _min_404 = ((s_4) < (260046848) ? (s_4) : (260046848));
                        unsigned int h_5 = _min_404 + ((h_noise[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_4 = ((s_4 <= 2139095040) ? h_5 : 32256);
                        uint32_t mags_1;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_1) : "r"(mag_3), "r"(mag_4));
                        uint32_t signs_1;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_1) : "r"(__as_u32(S[8])), "r"(__as_u32(S[9])));
                        unsigned int word_16 = signs_1 & 2147516416u | mags_1;
                        h_words[0] = word_16;
                        float _abs_405;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_405) : "f"(S[10]));
                        unsigned int s_5 = __as_u32(mul_f32_rz_noftz(_abs_405, __uint_as_float(125829120)));
                        unsigned int _min_406 = ((s_5) < (260046848) ? (s_5) : (260046848));
                        unsigned int h_6 = _min_406 + (h_noise[3] & 8191) >> 13;
                        unsigned int mag_5 = ((s_5 <= 2139095040) ? h_6 : 32256);
                        float _abs_407;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_407) : "f"(S[11]));
                        unsigned int s_6 = __as_u32(mul_f32_rz_noftz(_abs_407, __uint_as_float(125829120)));
                        unsigned int _min_408 = ((s_6) < (260046848) ? (s_6) : (260046848));
                        unsigned int h_7 = _min_408 + ((h_noise[3] & 536805376) >> 16) >> 13;
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
                            : "r"(noise_lane + 2048));
                        float _abs_409;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_409) : "f"(S[4]));
                        unsigned int s_7 = __as_u32(mul_f32_rz_noftz(_abs_409, __uint_as_float(125829120)));
                        unsigned int _min_410 = ((s_7) < (260046848) ? (s_7) : (260046848));
                        unsigned int h_8 = _min_410 + (h_noise[0] & 8191) >> 13;
                        unsigned int mag_7 = ((s_7 <= 2139095040) ? h_8 : 32256);
                        float _abs_411;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_411) : "f"(S[5]));
                        unsigned int s_8 = __as_u32(mul_f32_rz_noftz(_abs_411, __uint_as_float(125829120)));
                        unsigned int _min_412 = ((s_8) < (260046848) ? (s_8) : (260046848));
                        unsigned int h_9 = _min_412 + ((h_noise[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_8 = ((s_8 <= 2139095040) ? h_9 : 32256);
                        uint32_t mags_3;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_3) : "r"(mag_7), "r"(mag_8));
                        uint32_t signs_3;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_3) : "r"(__as_u32(S[4])), "r"(__as_u32(S[5])));
                        unsigned int word_18 = signs_3 & 2147516416u | mags_3;
                        h_words[0] = word_18;
                        float _abs_413;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_413) : "f"(S[6]));
                        unsigned int s_9 = __as_u32(mul_f32_rz_noftz(_abs_413, __uint_as_float(125829120)));
                        unsigned int _min_414 = ((s_9) < (260046848) ? (s_9) : (260046848));
                        unsigned int h_10 = _min_414 + (h_noise[1] & 8191) >> 13;
                        unsigned int mag_9 = ((s_9 <= 2139095040) ? h_10 : 32256);
                        float _abs_415;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_415) : "f"(S[7]));
                        unsigned int s_10 = __as_u32(mul_f32_rz_noftz(_abs_415, __uint_as_float(125829120)));
                        unsigned int _min_416 = ((s_10) < (260046848) ? (s_10) : (260046848));
                        unsigned int h_11 = _min_416 + ((h_noise[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_10 = ((s_10 <= 2139095040) ? h_11 : 32256);
                        uint32_t mags_4;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_4) : "r"(mag_9), "r"(mag_10));
                        uint32_t signs_4;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_4) : "r"(__as_u32(S[6])), "r"(__as_u32(S[7])));
                        unsigned int word_19 = signs_4 & 2147516416u | mags_4;
                        h_words[1] = word_19;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        float _abs_417;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_417) : "f"(S[12]));
                        unsigned int s_11 = __as_u32(mul_f32_rz_noftz(_abs_417, __uint_as_float(125829120)));
                        unsigned int _min_418 = ((s_11) < (260046848) ? (s_11) : (260046848));
                        unsigned int h_12 = _min_418 + (h_noise[2] & 8191) >> 13;
                        unsigned int mag_11 = ((s_11 <= 2139095040) ? h_12 : 32256);
                        float _abs_419;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_419) : "f"(S[13]));
                        unsigned int s_12 = __as_u32(mul_f32_rz_noftz(_abs_419, __uint_as_float(125829120)));
                        unsigned int _min_420 = ((s_12) < (260046848) ? (s_12) : (260046848));
                        unsigned int h_13 = _min_420 + ((h_noise[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_12 = ((s_12 <= 2139095040) ? h_13 : 32256);
                        uint32_t mags_5;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_5) : "r"(mag_11), "r"(mag_12));
                        uint32_t signs_5;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_5) : "r"(__as_u32(S[12])), "r"(__as_u32(S[13])));
                        unsigned int word_20 = signs_5 & 2147516416u | mags_5;
                        h_words[0] = word_20;
                        float _abs_421;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_421) : "f"(S[14]));
                        unsigned int s_13 = __as_u32(mul_f32_rz_noftz(_abs_421, __uint_as_float(125829120)));
                        unsigned int _min_422 = ((s_13) < (260046848) ? (s_13) : (260046848));
                        unsigned int h_14 = _min_422 + (h_noise[3] & 8191) >> 13;
                        unsigned int mag_13 = ((s_13 <= 2139095040) ? h_14 : 32256);
                        float _abs_423;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_423) : "f"(S[15]));
                        unsigned int s_14 = __as_u32(mul_f32_rz_noftz(_abs_423, __uint_as_float(125829120)));
                        unsigned int _min_424 = ((s_14) < (260046848) ? (s_14) : (260046848));
                        unsigned int h_15 = _min_424 + ((h_noise[3] & 536805376) >> 16) >> 13;
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
                            : "r"(noise_lane + 4096));
                        float _abs_425;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_425) : "f"(S[16]));
                        unsigned int s_15 = __as_u32(mul_f32_rz_noftz(_abs_425, __uint_as_float(125829120)));
                        unsigned int _min_426 = ((s_15) < (260046848) ? (s_15) : (260046848));
                        unsigned int h_16 = _min_426 + (h_noise[0] & 8191) >> 13;
                        unsigned int mag_15 = ((s_15 <= 2139095040) ? h_16 : 32256);
                        float _abs_427;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_427) : "f"(S[17]));
                        unsigned int s_16 = __as_u32(mul_f32_rz_noftz(_abs_427, __uint_as_float(125829120)));
                        unsigned int _min_428 = ((s_16) < (260046848) ? (s_16) : (260046848));
                        unsigned int h_17 = _min_428 + ((h_noise[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_16 = ((s_16 <= 2139095040) ? h_17 : 32256);
                        uint32_t mags_7;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_7) : "r"(mag_15), "r"(mag_16));
                        uint32_t signs_7;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_7) : "r"(__as_u32(S[16])), "r"(__as_u32(S[17])));
                        unsigned int word_22 = signs_7 & 2147516416u | mags_7;
                        h_words[0] = word_22;
                        float _abs_429;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_429) : "f"(S[18]));
                        unsigned int s_17 = __as_u32(mul_f32_rz_noftz(_abs_429, __uint_as_float(125829120)));
                        unsigned int _min_430 = ((s_17) < (260046848) ? (s_17) : (260046848));
                        unsigned int h_18 = _min_430 + (h_noise[1] & 8191) >> 13;
                        unsigned int mag_17 = ((s_17 <= 2139095040) ? h_18 : 32256);
                        float _abs_431;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_431) : "f"(S[19]));
                        unsigned int s_18 = __as_u32(mul_f32_rz_noftz(_abs_431, __uint_as_float(125829120)));
                        unsigned int _min_432 = ((s_18) < (260046848) ? (s_18) : (260046848));
                        unsigned int h_19 = _min_432 + ((h_noise[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_18 = ((s_18 <= 2139095040) ? h_19 : 32256);
                        uint32_t mags_8;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_8) : "r"(mag_17), "r"(mag_18));
                        uint32_t signs_8;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_8) : "r"(__as_u32(S[18])), "r"(__as_u32(S[19])));
                        unsigned int word_23 = signs_8 & 2147516416u | mags_8;
                        h_words[1] = word_23;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        float _abs_433;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_433) : "f"(S[24]));
                        unsigned int s_19 = __as_u32(mul_f32_rz_noftz(_abs_433, __uint_as_float(125829120)));
                        unsigned int _min_434 = ((s_19) < (260046848) ? (s_19) : (260046848));
                        unsigned int h_20 = _min_434 + (h_noise[2] & 8191) >> 13;
                        unsigned int mag_19 = ((s_19 <= 2139095040) ? h_20 : 32256);
                        float _abs_435;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_435) : "f"(S[25]));
                        unsigned int s_20 = __as_u32(mul_f32_rz_noftz(_abs_435, __uint_as_float(125829120)));
                        unsigned int _min_436 = ((s_20) < (260046848) ? (s_20) : (260046848));
                        unsigned int h_21 = _min_436 + ((h_noise[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_20 = ((s_20 <= 2139095040) ? h_21 : 32256);
                        uint32_t mags_9;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_9) : "r"(mag_19), "r"(mag_20));
                        uint32_t signs_9;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_9) : "r"(__as_u32(S[24])), "r"(__as_u32(S[25])));
                        unsigned int word_24 = signs_9 & 2147516416u | mags_9;
                        h_words[0] = word_24;
                        float _abs_437;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_437) : "f"(S[26]));
                        unsigned int s_21 = __as_u32(mul_f32_rz_noftz(_abs_437, __uint_as_float(125829120)));
                        unsigned int _min_438 = ((s_21) < (260046848) ? (s_21) : (260046848));
                        unsigned int h_22 = _min_438 + (h_noise[3] & 8191) >> 13;
                        unsigned int mag_21 = ((s_21 <= 2139095040) ? h_22 : 32256);
                        float _abs_439;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_439) : "f"(S[27]));
                        unsigned int s_22 = __as_u32(mul_f32_rz_noftz(_abs_439, __uint_as_float(125829120)));
                        unsigned int _min_440 = ((s_22) < (260046848) ? (s_22) : (260046848));
                        unsigned int h_23 = _min_440 + ((h_noise[3] & 536805376) >> 16) >> 13;
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
                            : "r"(noise_lane + 6144));
                        float _abs_441;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_441) : "f"(S[20]));
                        unsigned int s_23 = __as_u32(mul_f32_rz_noftz(_abs_441, __uint_as_float(125829120)));
                        unsigned int _min_442 = ((s_23) < (260046848) ? (s_23) : (260046848));
                        unsigned int h_24 = _min_442 + (h_noise[0] & 8191) >> 13;
                        unsigned int mag_23 = ((s_23 <= 2139095040) ? h_24 : 32256);
                        float _abs_443;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_443) : "f"(S[21]));
                        unsigned int s_24 = __as_u32(mul_f32_rz_noftz(_abs_443, __uint_as_float(125829120)));
                        unsigned int _min_444 = ((s_24) < (260046848) ? (s_24) : (260046848));
                        unsigned int h_25 = _min_444 + ((h_noise[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_24 = ((s_24 <= 2139095040) ? h_25 : 32256);
                        uint32_t mags_11;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_11) : "r"(mag_23), "r"(mag_24));
                        uint32_t signs_11;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_11) : "r"(__as_u32(S[20])), "r"(__as_u32(S[21])));
                        unsigned int word_26 = signs_11 & 2147516416u | mags_11;
                        h_words[0] = word_26;
                        float _abs_445;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_445) : "f"(S[22]));
                        unsigned int s_25 = __as_u32(mul_f32_rz_noftz(_abs_445, __uint_as_float(125829120)));
                        unsigned int _min_446 = ((s_25) < (260046848) ? (s_25) : (260046848));
                        unsigned int h_26 = _min_446 + (h_noise[1] & 8191) >> 13;
                        unsigned int mag_25 = ((s_25 <= 2139095040) ? h_26 : 32256);
                        float _abs_447;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_447) : "f"(S[23]));
                        unsigned int s_26 = __as_u32(mul_f32_rz_noftz(_abs_447, __uint_as_float(125829120)));
                        unsigned int _min_448 = ((s_26) < (260046848) ? (s_26) : (260046848));
                        unsigned int h_27 = _min_448 + ((h_noise[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_26 = ((s_26 <= 2139095040) ? h_27 : 32256);
                        uint32_t mags_12;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_12) : "r"(mag_25), "r"(mag_26));
                        uint32_t signs_12;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_12) : "r"(__as_u32(S[22])), "r"(__as_u32(S[23])));
                        unsigned int word_27 = signs_12 & 2147516416u | mags_12;
                        h_words[1] = word_27;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        float _abs_449;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_449) : "f"(S[28]));
                        unsigned int s_27 = __as_u32(mul_f32_rz_noftz(_abs_449, __uint_as_float(125829120)));
                        unsigned int _min_450 = ((s_27) < (260046848) ? (s_27) : (260046848));
                        unsigned int h_28 = _min_450 + (h_noise[2] & 8191) >> 13;
                        unsigned int mag_27 = ((s_27 <= 2139095040) ? h_28 : 32256);
                        float _abs_451;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_451) : "f"(S[29]));
                        unsigned int s_28 = __as_u32(mul_f32_rz_noftz(_abs_451, __uint_as_float(125829120)));
                        unsigned int _min_452 = ((s_28) < (260046848) ? (s_28) : (260046848));
                        unsigned int h_29 = _min_452 + ((h_noise[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_28 = ((s_28 <= 2139095040) ? h_29 : 32256);
                        uint32_t mags_13;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_13) : "r"(mag_27), "r"(mag_28));
                        uint32_t signs_13;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_13) : "r"(__as_u32(S[28])), "r"(__as_u32(S[29])));
                        unsigned int word_28 = signs_13 & 2147516416u | mags_13;
                        h_words[0] = word_28;
                        float _abs_453;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_453) : "f"(S[30]));
                        unsigned int s_29 = __as_u32(mul_f32_rz_noftz(_abs_453, __uint_as_float(125829120)));
                        unsigned int _min_454 = ((s_29) < (260046848) ? (s_29) : (260046848));
                        unsigned int h_30 = _min_454 + (h_noise[3] & 8191) >> 13;
                        unsigned int mag_29 = ((s_29 <= 2139095040) ? h_30 : 32256);
                        float _abs_455;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_455) : "f"(S[31]));
                        unsigned int s_30 = __as_u32(mul_f32_rz_noftz(_abs_455, __uint_as_float(125829120)));
                        unsigned int _min_456 = ((s_30) < (260046848) ? (s_30) : (260046848));
                        unsigned int h_31 = _min_456 + ((h_noise[3] & 536805376) >> 16) >> 13;
                        unsigned int mag_30 = ((s_30 <= 2139095040) ? h_31 : 32256);
                        uint32_t mags_14;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_14) : "r"(mag_29), "r"(mag_30));
                        uint32_t signs_14;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_14) : "r"(__as_u32(S[30])), "r"(__as_u32(S[31])));
                        unsigned int word_29 = signs_14 & 2147516416u | mags_14;
                        h_words[1] = word_29;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    }
                }
                if (philox_rounds == 0) {
                    uint32_t _cvt_rn_f16x2_f32_457;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_457) : "f"(S[1]), "f"(S[0]));
                    h_words[0] = _cvt_rn_f16x2_f32_457;
                    uint32_t _cvt_rn_f16x2_f32_458;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_458) : "f"(S[3]), "f"(S[2]));
                    h_words[1] = _cvt_rn_f16x2_f32_458;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_459;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_459) : "f"(S[5]), "f"(S[4]));
                    h_words[0] = _cvt_rn_f16x2_f32_459;
                    uint32_t _cvt_rn_f16x2_f32_460;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_460) : "f"(S[7]), "f"(S[6]));
                    h_words[1] = _cvt_rn_f16x2_f32_460;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_461;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_461) : "f"(S[9]), "f"(S[8]));
                    h_words[0] = _cvt_rn_f16x2_f32_461;
                    uint32_t _cvt_rn_f16x2_f32_462;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_462) : "f"(S[11]), "f"(S[10]));
                    h_words[1] = _cvt_rn_f16x2_f32_462;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_463;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_463) : "f"(S[13]), "f"(S[12]));
                    h_words[0] = _cvt_rn_f16x2_f32_463;
                    uint32_t _cvt_rn_f16x2_f32_464;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_464) : "f"(S[15]), "f"(S[14]));
                    h_words[1] = _cvt_rn_f16x2_f32_464;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_465;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_465) : "f"(S[17]), "f"(S[16]));
                    h_words[0] = _cvt_rn_f16x2_f32_465;
                    uint32_t _cvt_rn_f16x2_f32_466;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_466) : "f"(S[19]), "f"(S[18]));
                    h_words[1] = _cvt_rn_f16x2_f32_466;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_467;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_467) : "f"(S[21]), "f"(S[20]));
                    h_words[0] = _cvt_rn_f16x2_f32_467;
                    uint32_t _cvt_rn_f16x2_f32_468;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_468) : "f"(S[23]), "f"(S[22]));
                    h_words[1] = _cvt_rn_f16x2_f32_468;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_469;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_469) : "f"(S[25]), "f"(S[24]));
                    h_words[0] = _cvt_rn_f16x2_f32_469;
                    uint32_t _cvt_rn_f16x2_f32_470;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_470) : "f"(S[27]), "f"(S[26]));
                    h_words[1] = _cvt_rn_f16x2_f32_470;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_471;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_471) : "f"(S[29]), "f"(S[28]));
                    h_words[0] = _cvt_rn_f16x2_f32_471;
                    uint32_t _cvt_rn_f16x2_f32_472;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_472) : "f"(S[31]), "f"(S[30]));
                    h_words[1] = _cvt_rn_f16x2_f32_472;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                }
            }
        }
        if (use_ssm_state_indices == 0) {
            #pragma unroll 1
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
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_3 * 256) + group_byte));
                float beta_3 = sBeta[m_3];
                float acc0_x_7 = 0.0f;
                float acc0_y_7 = 0.0f;
                float acc1_x_7 = 0.0f;
                float acc1_y_7 = 0.0f;
                float _fma_473;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_473) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_7));
                acc0_x_7 = _fma_473;
                float _fma_474;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_474) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_7));
                acc0_y_7 = _fma_474;
                float _fma_475;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_475) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_7));
                acc1_x_7 = _fma_475;
                float _fma_476;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_476) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_7));
                acc1_y_7 = _fma_476;
                float _fma_477;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_477) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_7));
                acc0_x_7 = _fma_477;
                float _fma_478;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_478) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_7));
                acc0_y_7 = _fma_478;
                float _fma_479;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_479) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_7));
                acc1_x_7 = _fma_479;
                float _fma_480;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_480) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_7));
                acc1_y_7 = _fma_480;
                float _fma_481;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_481) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_7));
                acc0_x_7 = _fma_481;
                float _fma_482;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_482) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_7));
                acc0_y_7 = _fma_482;
                float _fma_483;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_483) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_7));
                acc1_x_7 = _fma_483;
                float _fma_484;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_484) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_7));
                acc1_y_7 = _fma_484;
                float _fma_485;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_485) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_7));
                acc0_x_7 = _fma_485;
                float _fma_486;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_486) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_7));
                acc0_y_7 = _fma_486;
                float _fma_487;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_487) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_7));
                acc1_x_7 = _fma_487;
                float _fma_488;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_488) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_7));
                acc1_y_7 = _fma_488;
                float _add_489;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_489) : "f"(acc0_x_7), "f"(acc1_x_7));
                float _add_490;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_490) : "f"(acc0_y_7), "f"(acc1_y_7));
                float dot_7 = _add_489 + _add_490;
                float _shfl_xor_sync_491 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 1);
                dot_7 = dot_7 + _shfl_xor_sync_491;
                float _shfl_xor_sync_492 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 2);
                dot_7 = dot_7 + _shfl_xor_sync_492;
                float _shfl_xor_sync_493 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 4);
                dot_7 = dot_7 + _shfl_xor_sync_493;
                float vn_6 = (v_regs[0] - dot_7) * beta_3;
                float _fma_494;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_494) : "f"(k_regs[0]), "f"(vn_6), "f"(S[0]));
                S[0] = _fma_494;
                float _fma_495;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_495) : "f"(k_regs[1]), "f"(vn_6), "f"(S[1]));
                S[1] = _fma_495;
                float _fma_496;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_496) : "f"(k_regs[2]), "f"(vn_6), "f"(S[2]));
                S[2] = _fma_496;
                float _fma_497;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_497) : "f"(k_regs[3]), "f"(vn_6), "f"(S[3]));
                S[3] = _fma_497;
                float _fma_498;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_498) : "f"(k_regs[4]), "f"(vn_6), "f"(S[4]));
                S[4] = _fma_498;
                float _fma_499;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_499) : "f"(k_regs[5]), "f"(vn_6), "f"(S[5]));
                S[5] = _fma_499;
                float _fma_500;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_500) : "f"(k_regs[6]), "f"(vn_6), "f"(S[6]));
                S[6] = _fma_500;
                float _fma_501;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_501) : "f"(k_regs[7]), "f"(vn_6), "f"(S[7]));
                S[7] = _fma_501;
                float _fma_502;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_502) : "f"(k_regs[8]), "f"(vn_6), "f"(S[8]));
                S[8] = _fma_502;
                float _fma_503;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_503) : "f"(k_regs[9]), "f"(vn_6), "f"(S[9]));
                S[9] = _fma_503;
                float _fma_504;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_504) : "f"(k_regs[10]), "f"(vn_6), "f"(S[10]));
                S[10] = _fma_504;
                float _fma_505;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_505) : "f"(k_regs[11]), "f"(vn_6), "f"(S[11]));
                S[11] = _fma_505;
                float _fma_506;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_506) : "f"(k_regs[12]), "f"(vn_6), "f"(S[12]));
                S[12] = _fma_506;
                float _fma_507;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_507) : "f"(k_regs[13]), "f"(vn_6), "f"(S[13]));
                S[13] = _fma_507;
                float _fma_508;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_508) : "f"(k_regs[14]), "f"(vn_6), "f"(S[14]));
                S[14] = _fma_508;
                float _fma_509;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_509) : "f"(k_regs[15]), "f"(vn_6), "f"(S[15]));
                S[15] = _fma_509;
                float acc0_x_7_1 = 0.0f;
                float acc0_y_7_1 = 0.0f;
                float acc1_x_7_1 = 0.0f;
                float acc1_y_7_1 = 0.0f;
                float _fma_510;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_510) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_510;
                float _fma_511;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_511) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_511;
                float _fma_512;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_512) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_512;
                float _fma_513;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_513) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_513;
                float _fma_514;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_514) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_514;
                float _fma_515;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_515) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_515;
                float _fma_516;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_516) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_516;
                float _fma_517;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_517) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_517;
                float _fma_518;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_518) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_518;
                float _fma_519;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_519) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_519;
                float _fma_520;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_520) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_520;
                float _fma_521;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_521) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_521;
                float _fma_522;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_522) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_522;
                float _fma_523;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_523) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_523;
                float _fma_524;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_524) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_524;
                float _fma_525;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_525) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_525;
                float _add_526;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_526) : "f"(acc0_x_7_1), "f"(acc1_x_7_1));
                float _add_527;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_527) : "f"(acc0_y_7_1), "f"(acc1_y_7_1));
                float dot_7_1 = _add_526 + _add_527;
                float _shfl_xor_sync_528 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 1);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_528;
                float _shfl_xor_sync_529 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 2);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_529;
                float _shfl_xor_sync_530 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 4);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_530;
                float vn_7 = (v_regs[1] - dot_7_1) * beta_3;
                float _fma_531;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_531) : "f"(k_regs[0]), "f"(vn_7), "f"(S[16]));
                S[16] = _fma_531;
                float _fma_532;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_532) : "f"(k_regs[1]), "f"(vn_7), "f"(S[17]));
                S[17] = _fma_532;
                float _fma_533;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_533) : "f"(k_regs[2]), "f"(vn_7), "f"(S[18]));
                S[18] = _fma_533;
                float _fma_534;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_534) : "f"(k_regs[3]), "f"(vn_7), "f"(S[19]));
                S[19] = _fma_534;
                float _fma_535;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_535) : "f"(k_regs[4]), "f"(vn_7), "f"(S[20]));
                S[20] = _fma_535;
                float _fma_536;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_536) : "f"(k_regs[5]), "f"(vn_7), "f"(S[21]));
                S[21] = _fma_536;
                float _fma_537;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_537) : "f"(k_regs[6]), "f"(vn_7), "f"(S[22]));
                S[22] = _fma_537;
                float _fma_538;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_538) : "f"(k_regs[7]), "f"(vn_7), "f"(S[23]));
                S[23] = _fma_538;
                float _fma_539;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_539) : "f"(k_regs[8]), "f"(vn_7), "f"(S[24]));
                S[24] = _fma_539;
                float _fma_540;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_540) : "f"(k_regs[9]), "f"(vn_7), "f"(S[25]));
                S[25] = _fma_540;
                float _fma_541;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_541) : "f"(k_regs[10]), "f"(vn_7), "f"(S[26]));
                S[26] = _fma_541;
                float _fma_542;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_542) : "f"(k_regs[11]), "f"(vn_7), "f"(S[27]));
                S[27] = _fma_542;
                float _fma_543;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_543) : "f"(k_regs[12]), "f"(vn_7), "f"(S[28]));
                S[28] = _fma_543;
                float _fma_544;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_544) : "f"(k_regs[13]), "f"(vn_7), "f"(S[29]));
                S[29] = _fma_544;
                float _fma_545;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_545) : "f"(k_regs[14]), "f"(vn_7), "f"(S[30]));
                S[30] = _fma_545;
                float _fma_546;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_546) : "f"(k_regs[15]), "f"(vn_7), "f"(S[31]));
                S[31] = _fma_546;
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
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_4 * 256) + group_byte));
                float beta_4 = sBeta[m_4];
                float acc0_x_8 = 0.0f;
                float acc0_y_8 = 0.0f;
                float acc1_x_8 = 0.0f;
                float acc1_y_8 = 0.0f;
                float _fma_547;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_547) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_8));
                acc0_x_8 = _fma_547;
                float _fma_548;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_548) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_8));
                acc0_y_8 = _fma_548;
                float _fma_549;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_549) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_8));
                acc1_x_8 = _fma_549;
                float _fma_550;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_550) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_8));
                acc1_y_8 = _fma_550;
                float _fma_551;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_551) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_8));
                acc0_x_8 = _fma_551;
                float _fma_552;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_552) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_8));
                acc0_y_8 = _fma_552;
                float _fma_553;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_553) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_8));
                acc1_x_8 = _fma_553;
                float _fma_554;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_554) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_8));
                acc1_y_8 = _fma_554;
                float _fma_555;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_555) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_8));
                acc0_x_8 = _fma_555;
                float _fma_556;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_556) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_8));
                acc0_y_8 = _fma_556;
                float _fma_557;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_557) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_8));
                acc1_x_8 = _fma_557;
                float _fma_558;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_558) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_8));
                acc1_y_8 = _fma_558;
                float _fma_559;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_559) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_8));
                acc0_x_8 = _fma_559;
                float _fma_560;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_560) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_8));
                acc0_y_8 = _fma_560;
                float _fma_561;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_561) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_8));
                acc1_x_8 = _fma_561;
                float _fma_562;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_562) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_8));
                acc1_y_8 = _fma_562;
                float _add_563;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_563) : "f"(acc0_x_8), "f"(acc1_x_8));
                float _add_564;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_564) : "f"(acc0_y_8), "f"(acc1_y_8));
                float dot_8 = _add_563 + _add_564;
                float _shfl_xor_sync_565 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 1);
                dot_8 = dot_8 + _shfl_xor_sync_565;
                float _shfl_xor_sync_566 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 2);
                dot_8 = dot_8 + _shfl_xor_sync_566;
                float _shfl_xor_sync_567 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 4);
                dot_8 = dot_8 + _shfl_xor_sync_567;
                float vn_8 = (v_regs[0] - dot_8) * beta_4;
                float _fma_568;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_568) : "f"(k_regs[0]), "f"(vn_8), "f"(S[0]));
                S[0] = _fma_568;
                float _fma_569;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_569) : "f"(k_regs[1]), "f"(vn_8), "f"(S[1]));
                S[1] = _fma_569;
                float _fma_570;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_570) : "f"(k_regs[2]), "f"(vn_8), "f"(S[2]));
                S[2] = _fma_570;
                float _fma_571;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_571) : "f"(k_regs[3]), "f"(vn_8), "f"(S[3]));
                S[3] = _fma_571;
                float _fma_572;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_572) : "f"(k_regs[4]), "f"(vn_8), "f"(S[4]));
                S[4] = _fma_572;
                float _fma_573;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_573) : "f"(k_regs[5]), "f"(vn_8), "f"(S[5]));
                S[5] = _fma_573;
                float _fma_574;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_574) : "f"(k_regs[6]), "f"(vn_8), "f"(S[6]));
                S[6] = _fma_574;
                float _fma_575;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_575) : "f"(k_regs[7]), "f"(vn_8), "f"(S[7]));
                S[7] = _fma_575;
                float _fma_576;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_576) : "f"(k_regs[8]), "f"(vn_8), "f"(S[8]));
                S[8] = _fma_576;
                float _fma_577;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_577) : "f"(k_regs[9]), "f"(vn_8), "f"(S[9]));
                S[9] = _fma_577;
                float _fma_578;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_578) : "f"(k_regs[10]), "f"(vn_8), "f"(S[10]));
                S[10] = _fma_578;
                float _fma_579;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_579) : "f"(k_regs[11]), "f"(vn_8), "f"(S[11]));
                S[11] = _fma_579;
                float _fma_580;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_580) : "f"(k_regs[12]), "f"(vn_8), "f"(S[12]));
                S[12] = _fma_580;
                float _fma_581;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_581) : "f"(k_regs[13]), "f"(vn_8), "f"(S[13]));
                S[13] = _fma_581;
                float _fma_582;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_582) : "f"(k_regs[14]), "f"(vn_8), "f"(S[14]));
                S[14] = _fma_582;
                float _fma_583;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_583) : "f"(k_regs[15]), "f"(vn_8), "f"(S[15]));
                S[15] = _fma_583;
                float acc0_x_9 = 0.0f;
                float acc0_y_9 = 0.0f;
                float acc1_x_9 = 0.0f;
                float acc1_y_9 = 0.0f;
                float _fma_584;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_584) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_9));
                acc0_x_9 = _fma_584;
                float _fma_585;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_585) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_9));
                acc0_y_9 = _fma_585;
                float _fma_586;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_586) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_9));
                acc1_x_9 = _fma_586;
                float _fma_587;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_587) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_9));
                acc1_y_9 = _fma_587;
                float _fma_588;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_588) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_9));
                acc0_x_9 = _fma_588;
                float _fma_589;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_589) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_9));
                acc0_y_9 = _fma_589;
                float _fma_590;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_590) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_9));
                acc1_x_9 = _fma_590;
                float _fma_591;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_591) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_9));
                acc1_y_9 = _fma_591;
                float _fma_592;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_592) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_9));
                acc0_x_9 = _fma_592;
                float _fma_593;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_593) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_9));
                acc0_y_9 = _fma_593;
                float _fma_594;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_594) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_9));
                acc1_x_9 = _fma_594;
                float _fma_595;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_595) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_9));
                acc1_y_9 = _fma_595;
                float _fma_596;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_596) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_9));
                acc0_x_9 = _fma_596;
                float _fma_597;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_597) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_9));
                acc0_y_9 = _fma_597;
                float _fma_598;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_598) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_9));
                acc1_x_9 = _fma_598;
                float _fma_599;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_599) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_9));
                acc1_y_9 = _fma_599;
                float _add_600;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_600) : "f"(acc0_x_9), "f"(acc1_x_9));
                float _add_601;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_601) : "f"(acc0_y_9), "f"(acc1_y_9));
                float dot_9 = _add_600 + _add_601;
                float _shfl_xor_sync_602 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 1);
                dot_9 = dot_9 + _shfl_xor_sync_602;
                float _shfl_xor_sync_603 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 2);
                dot_9 = dot_9 + _shfl_xor_sync_603;
                float _shfl_xor_sync_604 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 4);
                dot_9 = dot_9 + _shfl_xor_sync_604;
                float vn_9 = (v_regs[1] - dot_9) * beta_4;
                float _fma_605;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_605) : "f"(k_regs[0]), "f"(vn_9), "f"(S[16]));
                S[16] = _fma_605;
                float _fma_606;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_606) : "f"(k_regs[1]), "f"(vn_9), "f"(S[17]));
                S[17] = _fma_606;
                float _fma_607;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_607) : "f"(k_regs[2]), "f"(vn_9), "f"(S[18]));
                S[18] = _fma_607;
                float _fma_608;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_608) : "f"(k_regs[3]), "f"(vn_9), "f"(S[19]));
                S[19] = _fma_608;
                float _fma_609;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_609) : "f"(k_regs[4]), "f"(vn_9), "f"(S[20]));
                S[20] = _fma_609;
                float _fma_610;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_610) : "f"(k_regs[5]), "f"(vn_9), "f"(S[21]));
                S[21] = _fma_610;
                float _fma_611;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_611) : "f"(k_regs[6]), "f"(vn_9), "f"(S[22]));
                S[22] = _fma_611;
                float _fma_612;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_612) : "f"(k_regs[7]), "f"(vn_9), "f"(S[23]));
                S[23] = _fma_612;
                float _fma_613;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_613) : "f"(k_regs[8]), "f"(vn_9), "f"(S[24]));
                S[24] = _fma_613;
                float _fma_614;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_614) : "f"(k_regs[9]), "f"(vn_9), "f"(S[25]));
                S[25] = _fma_614;
                float _fma_615;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_615) : "f"(k_regs[10]), "f"(vn_9), "f"(S[26]));
                S[26] = _fma_615;
                float _fma_616;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_616) : "f"(k_regs[11]), "f"(vn_9), "f"(S[27]));
                S[27] = _fma_616;
                float _fma_617;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_617) : "f"(k_regs[12]), "f"(vn_9), "f"(S[28]));
                S[28] = _fma_617;
                float _fma_618;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_618) : "f"(k_regs[13]), "f"(vn_9), "f"(S[29]));
                S[29] = _fma_618;
                float _fma_619;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_619) : "f"(k_regs[14]), "f"(vn_9), "f"(S[30]));
                S[30] = _fma_619;
                float _fma_620;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_620) : "f"(k_regs[15]), "f"(vn_9), "f"(S[31]));
                S[31] = _fma_620;
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
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_5 * 256) + group_byte));
                float beta_5 = sBeta[m_5];
                float acc0_x_10 = 0.0f;
                float acc0_y_10 = 0.0f;
                float acc1_x_10 = 0.0f;
                float acc1_y_10 = 0.0f;
                float _fma_621;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_621) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_10));
                acc0_x_10 = _fma_621;
                float _fma_622;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_622) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_10));
                acc0_y_10 = _fma_622;
                float _fma_623;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_623) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_10));
                acc1_x_10 = _fma_623;
                float _fma_624;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_624) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_10));
                acc1_y_10 = _fma_624;
                float _fma_625;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_625) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_10));
                acc0_x_10 = _fma_625;
                float _fma_626;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_626) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_10));
                acc0_y_10 = _fma_626;
                float _fma_627;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_627) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_10));
                acc1_x_10 = _fma_627;
                float _fma_628;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_628) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_10));
                acc1_y_10 = _fma_628;
                float _fma_629;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_629) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_10));
                acc0_x_10 = _fma_629;
                float _fma_630;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_630) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_10));
                acc0_y_10 = _fma_630;
                float _fma_631;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_631) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_10));
                acc1_x_10 = _fma_631;
                float _fma_632;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_632) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_10));
                acc1_y_10 = _fma_632;
                float _fma_633;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_633) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_10));
                acc0_x_10 = _fma_633;
                float _fma_634;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_634) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_10));
                acc0_y_10 = _fma_634;
                float _fma_635;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_635) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_10));
                acc1_x_10 = _fma_635;
                float _fma_636;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_636) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_10));
                acc1_y_10 = _fma_636;
                float _add_637;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_637) : "f"(acc0_x_10), "f"(acc1_x_10));
                float _add_638;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_638) : "f"(acc0_y_10), "f"(acc1_y_10));
                float dot_10 = _add_637 + _add_638;
                float _shfl_xor_sync_639 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 1);
                dot_10 = dot_10 + _shfl_xor_sync_639;
                float _shfl_xor_sync_640 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 2);
                dot_10 = dot_10 + _shfl_xor_sync_640;
                float _shfl_xor_sync_641 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 4);
                dot_10 = dot_10 + _shfl_xor_sync_641;
                float vn_10 = (v_regs[0] - dot_10) * beta_5;
                float _fma_642;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_642) : "f"(k_regs[0]), "f"(vn_10), "f"(S[0]));
                S[0] = _fma_642;
                float _fma_643;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_643) : "f"(k_regs[1]), "f"(vn_10), "f"(S[1]));
                S[1] = _fma_643;
                float _fma_644;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_644) : "f"(k_regs[2]), "f"(vn_10), "f"(S[2]));
                S[2] = _fma_644;
                float _fma_645;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_645) : "f"(k_regs[3]), "f"(vn_10), "f"(S[3]));
                S[3] = _fma_645;
                float _fma_646;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_646) : "f"(k_regs[4]), "f"(vn_10), "f"(S[4]));
                S[4] = _fma_646;
                float _fma_647;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_647) : "f"(k_regs[5]), "f"(vn_10), "f"(S[5]));
                S[5] = _fma_647;
                float _fma_648;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_648) : "f"(k_regs[6]), "f"(vn_10), "f"(S[6]));
                S[6] = _fma_648;
                float _fma_649;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_649) : "f"(k_regs[7]), "f"(vn_10), "f"(S[7]));
                S[7] = _fma_649;
                float _fma_650;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_650) : "f"(k_regs[8]), "f"(vn_10), "f"(S[8]));
                S[8] = _fma_650;
                float _fma_651;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_651) : "f"(k_regs[9]), "f"(vn_10), "f"(S[9]));
                S[9] = _fma_651;
                float _fma_652;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_652) : "f"(k_regs[10]), "f"(vn_10), "f"(S[10]));
                S[10] = _fma_652;
                float _fma_653;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_653) : "f"(k_regs[11]), "f"(vn_10), "f"(S[11]));
                S[11] = _fma_653;
                float _fma_654;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_654) : "f"(k_regs[12]), "f"(vn_10), "f"(S[12]));
                S[12] = _fma_654;
                float _fma_655;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_655) : "f"(k_regs[13]), "f"(vn_10), "f"(S[13]));
                S[13] = _fma_655;
                float _fma_656;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_656) : "f"(k_regs[14]), "f"(vn_10), "f"(S[14]));
                S[14] = _fma_656;
                float _fma_657;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_657) : "f"(k_regs[15]), "f"(vn_10), "f"(S[15]));
                S[15] = _fma_657;
                float acc0_x_11 = 0.0f;
                float acc0_y_11 = 0.0f;
                float acc1_x_11 = 0.0f;
                float acc1_y_11 = 0.0f;
                float _fma_658;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_658) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_11));
                acc0_x_11 = _fma_658;
                float _fma_659;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_659) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_11));
                acc0_y_11 = _fma_659;
                float _fma_660;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_660) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_11));
                acc1_x_11 = _fma_660;
                float _fma_661;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_661) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_11));
                acc1_y_11 = _fma_661;
                float _fma_662;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_662) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_11));
                acc0_x_11 = _fma_662;
                float _fma_663;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_663) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_11));
                acc0_y_11 = _fma_663;
                float _fma_664;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_664) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_11));
                acc1_x_11 = _fma_664;
                float _fma_665;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_665) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_11));
                acc1_y_11 = _fma_665;
                float _fma_666;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_666) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_11));
                acc0_x_11 = _fma_666;
                float _fma_667;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_667) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_11));
                acc0_y_11 = _fma_667;
                float _fma_668;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_668) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_11));
                acc1_x_11 = _fma_668;
                float _fma_669;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_669) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_11));
                acc1_y_11 = _fma_669;
                float _fma_670;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_670) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_11));
                acc0_x_11 = _fma_670;
                float _fma_671;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_671) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_11));
                acc0_y_11 = _fma_671;
                float _fma_672;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_672) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_11));
                acc1_x_11 = _fma_672;
                float _fma_673;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_673) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_11));
                acc1_y_11 = _fma_673;
                float _add_674;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_674) : "f"(acc0_x_11), "f"(acc1_x_11));
                float _add_675;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_675) : "f"(acc0_y_11), "f"(acc1_y_11));
                float dot_11 = _add_674 + _add_675;
                float _shfl_xor_sync_676 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 1);
                dot_11 = dot_11 + _shfl_xor_sync_676;
                float _shfl_xor_sync_677 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 2);
                dot_11 = dot_11 + _shfl_xor_sync_677;
                float _shfl_xor_sync_678 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 4);
                dot_11 = dot_11 + _shfl_xor_sync_678;
                float vn_11 = (v_regs[1] - dot_11) * beta_5;
                float _fma_679;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_679) : "f"(k_regs[0]), "f"(vn_11), "f"(S[16]));
                S[16] = _fma_679;
                float _fma_680;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_680) : "f"(k_regs[1]), "f"(vn_11), "f"(S[17]));
                S[17] = _fma_680;
                float _fma_681;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_681) : "f"(k_regs[2]), "f"(vn_11), "f"(S[18]));
                S[18] = _fma_681;
                float _fma_682;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_682) : "f"(k_regs[3]), "f"(vn_11), "f"(S[19]));
                S[19] = _fma_682;
                float _fma_683;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_683) : "f"(k_regs[4]), "f"(vn_11), "f"(S[20]));
                S[20] = _fma_683;
                float _fma_684;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_684) : "f"(k_regs[5]), "f"(vn_11), "f"(S[21]));
                S[21] = _fma_684;
                float _fma_685;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_685) : "f"(k_regs[6]), "f"(vn_11), "f"(S[22]));
                S[22] = _fma_685;
                float _fma_686;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_686) : "f"(k_regs[7]), "f"(vn_11), "f"(S[23]));
                S[23] = _fma_686;
                float _fma_687;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_687) : "f"(k_regs[8]), "f"(vn_11), "f"(S[24]));
                S[24] = _fma_687;
                float _fma_688;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_688) : "f"(k_regs[9]), "f"(vn_11), "f"(S[25]));
                S[25] = _fma_688;
                float _fma_689;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_689) : "f"(k_regs[10]), "f"(vn_11), "f"(S[26]));
                S[26] = _fma_689;
                float _fma_690;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_690) : "f"(k_regs[11]), "f"(vn_11), "f"(S[27]));
                S[27] = _fma_690;
                float _fma_691;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_691) : "f"(k_regs[12]), "f"(vn_11), "f"(S[28]));
                S[28] = _fma_691;
                float _fma_692;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_692) : "f"(k_regs[13]), "f"(vn_11), "f"(S[29]));
                S[29] = _fma_692;
                float _fma_693;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_693) : "f"(k_regs[14]), "f"(vn_11), "f"(S[30]));
                S[30] = _fma_693;
                float _fma_694;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_694) : "f"(k_regs[15]), "f"(vn_11), "f"(S[31]));
                S[31] = _fma_694;
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
                float acc0_x_12 = 0.0f;
                float acc0_y_12 = 0.0f;
                float acc1_x_12 = 0.0f;
                float acc1_y_12 = 0.0f;
                float _fma_695;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_695) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_12));
                acc0_x_12 = _fma_695;
                float _fma_696;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_696) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_12));
                acc0_y_12 = _fma_696;
                float _fma_697;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_697) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_12));
                acc1_x_12 = _fma_697;
                float _fma_698;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_698) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_12));
                acc1_y_12 = _fma_698;
                float _fma_699;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_699) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_12));
                acc0_x_12 = _fma_699;
                float _fma_700;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_700) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_12));
                acc0_y_12 = _fma_700;
                float _fma_701;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_701) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_12));
                acc1_x_12 = _fma_701;
                float _fma_702;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_702) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_12));
                acc1_y_12 = _fma_702;
                float _fma_703;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_703) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_12));
                acc0_x_12 = _fma_703;
                float _fma_704;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_704) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_12));
                acc0_y_12 = _fma_704;
                float _fma_705;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_705) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_12));
                acc1_x_12 = _fma_705;
                float _fma_706;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_706) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_12));
                acc1_y_12 = _fma_706;
                float _fma_707;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_707) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_12));
                acc0_x_12 = _fma_707;
                float _fma_708;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_708) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_12));
                acc0_y_12 = _fma_708;
                float _fma_709;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_709) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_12));
                acc1_x_12 = _fma_709;
                float _fma_710;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_710) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_12));
                acc1_y_12 = _fma_710;
                float _add_711;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_711) : "f"(acc0_x_12), "f"(acc1_x_12));
                float _add_712;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_712) : "f"(acc0_y_12), "f"(acc1_y_12));
                float dot_12 = _add_711 + _add_712;
                float _shfl_xor_sync_713 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 1);
                dot_12 = dot_12 + _shfl_xor_sync_713;
                float _shfl_xor_sync_714 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 2);
                dot_12 = dot_12 + _shfl_xor_sync_714;
                float _shfl_xor_sync_715 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 4);
                dot_12 = dot_12 + _shfl_xor_sync_715;
                o4[0] = dot_12;
                float acc0_x_13 = 0.0f;
                float acc0_y_13 = 0.0f;
                float acc1_x_13 = 0.0f;
                float acc1_y_13 = 0.0f;
                float _fma_716;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_716) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_13));
                acc0_x_13 = _fma_716;
                float _fma_717;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_717) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_13));
                acc0_y_13 = _fma_717;
                float _fma_718;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_718) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_13));
                acc1_x_13 = _fma_718;
                float _fma_719;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_719) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_13));
                acc1_y_13 = _fma_719;
                float _fma_720;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_720) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_13));
                acc0_x_13 = _fma_720;
                float _fma_721;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_721) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_13));
                acc0_y_13 = _fma_721;
                float _fma_722;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_722) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_13));
                acc1_x_13 = _fma_722;
                float _fma_723;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_723) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_13));
                acc1_y_13 = _fma_723;
                float _fma_724;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_724) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_13));
                acc0_x_13 = _fma_724;
                float _fma_725;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_725) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_13));
                acc0_y_13 = _fma_725;
                float _fma_726;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_726) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_13));
                acc1_x_13 = _fma_726;
                float _fma_727;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_727) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_13));
                acc1_y_13 = _fma_727;
                float _fma_728;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_728) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_13));
                acc0_x_13 = _fma_728;
                float _fma_729;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_729) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_13));
                acc0_y_13 = _fma_729;
                float _fma_730;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_730) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_13));
                acc1_x_13 = _fma_730;
                float _fma_731;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_731) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_13));
                acc1_y_13 = _fma_731;
                float _add_732;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_732) : "f"(acc0_x_13), "f"(acc1_x_13));
                float _add_733;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_733) : "f"(acc0_y_13), "f"(acc1_y_13));
                float dot_13 = _add_732 + _add_733;
                float _shfl_xor_sync_734 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 1);
                dot_13 = dot_13 + _shfl_xor_sync_734;
                float _shfl_xor_sync_735 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 2);
                dot_13 = dot_13 + _shfl_xor_sync_735;
                float _shfl_xor_sync_736 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 4);
                dot_13 = dot_13 + _shfl_xor_sync_736;
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
                    uint32_t _read_special_reg_737;
                    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(_read_special_reg_737));
                    int tid_r = (int)_read_special_reg_737;
                    long long noise_at = read_slot_head + (long long)(tid_r / 8 * 2) * state_stride_p2 + (long long)((tid_r - tid_r / 8 * 8) * 4);
                    if (philox_rounds != 0) {
                        bool nan_1 = 0;
                        uint32_t _setp_738_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_738_bits) : "f"(S[0]), "f"(S[1]));
                        bool _setp_738 = _setp_738_bits != 0;
                        nan_1 = nan_1 || _setp_738;
                        uint32_t _setp_739_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_739_bits) : "f"(S[2]), "f"(S[3]));
                        bool _setp_739 = _setp_739_bits != 0;
                        nan_1 = nan_1 || _setp_739;
                        uint32_t _setp_740_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_740_bits) : "f"(S[4]), "f"(S[5]));
                        bool _setp_740 = _setp_740_bits != 0;
                        nan_1 = nan_1 || _setp_740;
                        uint32_t _setp_741_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_741_bits) : "f"(S[6]), "f"(S[7]));
                        bool _setp_741 = _setp_741_bits != 0;
                        nan_1 = nan_1 || _setp_741;
                        uint32_t _setp_742_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_742_bits) : "f"(S[8]), "f"(S[9]));
                        bool _setp_742 = _setp_742_bits != 0;
                        nan_1 = nan_1 || _setp_742;
                        uint32_t _setp_743_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_743_bits) : "f"(S[10]), "f"(S[11]));
                        bool _setp_743 = _setp_743_bits != 0;
                        nan_1 = nan_1 || _setp_743;
                        uint32_t _setp_744_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_744_bits) : "f"(S[12]), "f"(S[13]));
                        bool _setp_744 = _setp_744_bits != 0;
                        nan_1 = nan_1 || _setp_744;
                        uint32_t _setp_745_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_745_bits) : "f"(S[14]), "f"(S[15]));
                        bool _setp_745 = _setp_745_bits != 0;
                        nan_1 = nan_1 || _setp_745;
                        uint32_t _setp_746_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_746_bits) : "f"(S[16]), "f"(S[17]));
                        bool _setp_746 = _setp_746_bits != 0;
                        nan_1 = nan_1 || _setp_746;
                        uint32_t _setp_747_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_747_bits) : "f"(S[18]), "f"(S[19]));
                        bool _setp_747 = _setp_747_bits != 0;
                        nan_1 = nan_1 || _setp_747;
                        uint32_t _setp_748_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_748_bits) : "f"(S[20]), "f"(S[21]));
                        bool _setp_748 = _setp_748_bits != 0;
                        nan_1 = nan_1 || _setp_748;
                        uint32_t _setp_749_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_749_bits) : "f"(S[22]), "f"(S[23]));
                        bool _setp_749 = _setp_749_bits != 0;
                        nan_1 = nan_1 || _setp_749;
                        uint32_t _setp_750_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_750_bits) : "f"(S[24]), "f"(S[25]));
                        bool _setp_750 = _setp_750_bits != 0;
                        nan_1 = nan_1 || _setp_750;
                        uint32_t _setp_751_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_751_bits) : "f"(S[26]), "f"(S[27]));
                        bool _setp_751 = _setp_751_bits != 0;
                        nan_1 = nan_1 || _setp_751;
                        uint32_t _setp_752_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_752_bits) : "f"(S[28]), "f"(S[29]));
                        bool _setp_752 = _setp_752_bits != 0;
                        nan_1 = nan_1 || _setp_752;
                        uint32_t _setp_753_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_753_bits) : "f"(S[30]), "f"(S[31]));
                        bool _setp_753 = _setp_753_bits != 0;
                        nan_1 = nan_1 || _setp_753;
                        int _warp_vote_754 = __any_sync(__activemask(), nan_1);
                        if (__builtin_expect(_warp_vote_754 == 0, 1)) {
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
                            float _abs_755;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_755) : "f"(S[0]));
                            unsigned int s_31 = __as_u32(mul_f32_rz_noftz(_abs_755, __uint_as_float(125829120)));
                            unsigned int _min_756 = ((s_31) < (260046848) ? (s_31) : (260046848));
                            unsigned int h_32 = _min_756 + (h_noise[0] & 8191) >> 13;
                            unsigned int mag_31 = ((s_31 <= 2139095040) ? h_32 : 32256);
                            float _abs_757;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_757) : "f"(S[1]));
                            unsigned int s_31_1 = __as_u32(mul_f32_rz_noftz(_abs_757, __uint_as_float(125829120)));
                            unsigned int _min_758 = ((s_31_1) < (260046848) ? (s_31_1) : (260046848));
                            unsigned int h_33 = _min_758 + ((h_noise[0] & 536805376) >> 16) >> 13;
                            unsigned int mag_31_1 = ((s_31_1 <= 2139095040) ? h_33 : 32256);
                            uint32_t mags_15;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_15) : "r"(mag_31), "r"(mag_31_1));
                            uint32_t signs_15;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_15) : "r"(__as_u32(S[0])), "r"(__as_u32(S[1])));
                            unsigned int word_45 = signs_15 & 2147516416u | mags_15;
                            h_words[0] = word_45;
                            float _abs_759;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_759) : "f"(S[2]));
                            unsigned int s_32 = __as_u32(mul_f32_rz_noftz(_abs_759, __uint_as_float(125829120)));
                            unsigned int _min_760 = ((s_32) < (260046848) ? (s_32) : (260046848));
                            unsigned int h_34 = _min_760 + (h_noise[1] & 8191) >> 13;
                            unsigned int mag_32 = ((s_32 <= 2139095040) ? h_34 : 32256);
                            float _abs_761;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_761) : "f"(S[3]));
                            unsigned int s_33 = __as_u32(mul_f32_rz_noftz(_abs_761, __uint_as_float(125829120)));
                            unsigned int _min_762 = ((s_33) < (260046848) ? (s_33) : (260046848));
                            unsigned int h_35 = _min_762 + ((h_noise[1] & 536805376) >> 16) >> 13;
                            unsigned int mag_33 = ((s_33 <= 2139095040) ? h_35 : 32256);
                            uint32_t mags_15_1;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_15_1) : "r"(mag_32), "r"(mag_33));
                            uint32_t signs_15_1;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_15_1) : "r"(__as_u32(S[2])), "r"(__as_u32(S[3])));
                            unsigned int word_45_1 = signs_15_1 & 2147516416u | mags_15_1;
                            h_words[1] = word_45_1;
                            reinterpret_cast<int2*>(state + (write_rows_base))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_763;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_763) : "f"(S[8]));
                            unsigned int s_34 = __as_u32(mul_f32_rz_noftz(_abs_763, __uint_as_float(125829120)));
                            unsigned int _min_764 = ((s_34) < (260046848) ? (s_34) : (260046848));
                            unsigned int h_36 = _min_764 + (h_noise[2] & 8191) >> 13;
                            unsigned int mag_34 = ((s_34 <= 2139095040) ? h_36 : 32256);
                            float _abs_765;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_765) : "f"(S[9]));
                            unsigned int s_35 = __as_u32(mul_f32_rz_noftz(_abs_765, __uint_as_float(125829120)));
                            unsigned int _min_766 = ((s_35) < (260046848) ? (s_35) : (260046848));
                            unsigned int h_37 = _min_766 + ((h_noise[2] & 536805376) >> 16) >> 13;
                            unsigned int mag_35 = ((s_35 <= 2139095040) ? h_37 : 32256);
                            uint32_t mags_16;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_16) : "r"(mag_34), "r"(mag_35));
                            uint32_t signs_16;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_16) : "r"(__as_u32(S[8])), "r"(__as_u32(S[9])));
                            unsigned int word_46 = signs_16 & 2147516416u | mags_16;
                            h_words[0] = word_46;
                            float _abs_767;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_767) : "f"(S[10]));
                            unsigned int s_36 = __as_u32(mul_f32_rz_noftz(_abs_767, __uint_as_float(125829120)));
                            unsigned int _min_768 = ((s_36) < (260046848) ? (s_36) : (260046848));
                            unsigned int h_38 = _min_768 + (h_noise[3] & 8191) >> 13;
                            unsigned int mag_36 = ((s_36 <= 2139095040) ? h_38 : 32256);
                            float _abs_769;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_769) : "f"(S[11]));
                            unsigned int s_37 = __as_u32(mul_f32_rz_noftz(_abs_769, __uint_as_float(125829120)));
                            unsigned int _min_770 = ((s_37) < (260046848) ? (s_37) : (260046848));
                            unsigned int h_39 = _min_770 + ((h_noise[3] & 536805376) >> 16) >> 13;
                            unsigned int mag_37 = ((s_37 <= 2139095040) ? h_39 : 32256);
                            uint32_t mags_17;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_17) : "r"(mag_36), "r"(mag_37));
                            uint32_t signs_17;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_17) : "r"(__as_u32(S[10])), "r"(__as_u32(S[11])));
                            unsigned int word_47 = signs_17 & 2147516416u | mags_17;
                            h_words[1] = word_47;
                            reinterpret_cast<int2*>(state + (write_rows_base + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_771;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_771) : "f"(S[4]));
                            unsigned int s_38 = __as_u32(mul_f32_rz_noftz(_abs_771, __uint_as_float(125829120)));
                            unsigned int _min_772 = ((s_38) < (260046848) ? (s_38) : (260046848));
                            unsigned int h_40 = _min_772 + (h_noise[4] & 8191) >> 13;
                            unsigned int mag_38 = ((s_38 <= 2139095040) ? h_40 : 32256);
                            float _abs_773;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_773) : "f"(S[5]));
                            unsigned int s_39 = __as_u32(mul_f32_rz_noftz(_abs_773, __uint_as_float(125829120)));
                            unsigned int _min_774 = ((s_39) < (260046848) ? (s_39) : (260046848));
                            unsigned int h_41 = _min_774 + ((h_noise[4] & 536805376) >> 16) >> 13;
                            unsigned int mag_39 = ((s_39 <= 2139095040) ? h_41 : 32256);
                            uint32_t mags_18;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_18) : "r"(mag_38), "r"(mag_39));
                            uint32_t signs_18;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_18) : "r"(__as_u32(S[4])), "r"(__as_u32(S[5])));
                            unsigned int word_48 = signs_18 & 2147516416u | mags_18;
                            h_words[0] = word_48;
                            float _abs_775;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_775) : "f"(S[6]));
                            unsigned int s_40 = __as_u32(mul_f32_rz_noftz(_abs_775, __uint_as_float(125829120)));
                            unsigned int _min_776 = ((s_40) < (260046848) ? (s_40) : (260046848));
                            unsigned int h_42 = _min_776 + (h_noise[5] & 8191) >> 13;
                            unsigned int mag_40 = ((s_40 <= 2139095040) ? h_42 : 32256);
                            float _abs_777;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_777) : "f"(S[7]));
                            unsigned int s_41 = __as_u32(mul_f32_rz_noftz(_abs_777, __uint_as_float(125829120)));
                            unsigned int _min_778 = ((s_41) < (260046848) ? (s_41) : (260046848));
                            unsigned int h_43 = _min_778 + ((h_noise[5] & 536805376) >> 16) >> 13;
                            unsigned int mag_41 = ((s_41 <= 2139095040) ? h_43 : 32256);
                            uint32_t mags_19;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_19) : "r"(mag_40), "r"(mag_41));
                            uint32_t signs_19;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_19) : "r"(__as_u32(S[6])), "r"(__as_u32(S[7])));
                            unsigned int word_49 = signs_19 & 2147516416u | mags_19;
                            h_words[1] = word_49;
                            reinterpret_cast<int2*>(state + (write_rows_base + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_779;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_779) : "f"(S[12]));
                            unsigned int s_42 = __as_u32(mul_f32_rz_noftz(_abs_779, __uint_as_float(125829120)));
                            unsigned int _min_780 = ((s_42) < (260046848) ? (s_42) : (260046848));
                            unsigned int h_44 = _min_780 + (h_noise[6] & 8191) >> 13;
                            unsigned int mag_42 = ((s_42 <= 2139095040) ? h_44 : 32256);
                            float _abs_781;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_781) : "f"(S[13]));
                            unsigned int s_43 = __as_u32(mul_f32_rz_noftz(_abs_781, __uint_as_float(125829120)));
                            unsigned int _min_782 = ((s_43) < (260046848) ? (s_43) : (260046848));
                            unsigned int h_45 = _min_782 + ((h_noise[6] & 536805376) >> 16) >> 13;
                            unsigned int mag_43 = ((s_43 <= 2139095040) ? h_45 : 32256);
                            uint32_t mags_20;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_20) : "r"(mag_42), "r"(mag_43));
                            uint32_t signs_20;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_20) : "r"(__as_u32(S[12])), "r"(__as_u32(S[13])));
                            unsigned int word_50 = signs_20 & 2147516416u | mags_20;
                            h_words[0] = word_50;
                            float _abs_783;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_783) : "f"(S[14]));
                            unsigned int s_44 = __as_u32(mul_f32_rz_noftz(_abs_783, __uint_as_float(125829120)));
                            unsigned int _min_784 = ((s_44) < (260046848) ? (s_44) : (260046848));
                            unsigned int h_46 = _min_784 + (h_noise[7] & 8191) >> 13;
                            unsigned int mag_44 = ((s_44 <= 2139095040) ? h_46 : 32256);
                            float _abs_785;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_785) : "f"(S[15]));
                            unsigned int s_45 = __as_u32(mul_f32_rz_noftz(_abs_785, __uint_as_float(125829120)));
                            unsigned int _min_786 = ((s_45) < (260046848) ? (s_45) : (260046848));
                            unsigned int h_47 = _min_786 + ((h_noise[7] & 536805376) >> 16) >> 13;
                            unsigned int mag_45 = ((s_45 <= 2139095040) ? h_47 : 32256);
                            uint32_t mags_21;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_21) : "r"(mag_44), "r"(mag_45));
                            uint32_t signs_21;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_21) : "r"(__as_u32(S[14])), "r"(__as_u32(S[15])));
                            unsigned int word_51 = signs_21 & 2147516416u | mags_21;
                            h_words[1] = word_51;
                            reinterpret_cast<int2*>(state + (write_rows_base + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_787;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_787) : "f"(S[16]));
                            unsigned int s_46 = __as_u32(mul_f32_rz_noftz(_abs_787, __uint_as_float(125829120)));
                            unsigned int _min_788 = ((s_46) < (260046848) ? (s_46) : (260046848));
                            unsigned int h_48 = _min_788 + (h_noise[8] & 8191) >> 13;
                            unsigned int mag_46 = ((s_46 <= 2139095040) ? h_48 : 32256);
                            float _abs_789;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_789) : "f"(S[17]));
                            unsigned int s_47 = __as_u32(mul_f32_rz_noftz(_abs_789, __uint_as_float(125829120)));
                            unsigned int _min_790 = ((s_47) < (260046848) ? (s_47) : (260046848));
                            unsigned int h_49 = _min_790 + ((h_noise[8] & 536805376) >> 16) >> 13;
                            unsigned int mag_47 = ((s_47 <= 2139095040) ? h_49 : 32256);
                            uint32_t mags_22;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_22) : "r"(mag_46), "r"(mag_47));
                            uint32_t signs_22;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_22) : "r"(__as_u32(S[16])), "r"(__as_u32(S[17])));
                            unsigned int word_52 = signs_22 & 2147516416u | mags_22;
                            h_words[0] = word_52;
                            float _abs_791;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_791) : "f"(S[18]));
                            unsigned int s_48 = __as_u32(mul_f32_rz_noftz(_abs_791, __uint_as_float(125829120)));
                            unsigned int _min_792 = ((s_48) < (260046848) ? (s_48) : (260046848));
                            unsigned int h_50 = _min_792 + (h_noise[9] & 8191) >> 13;
                            unsigned int mag_48 = ((s_48 <= 2139095040) ? h_50 : 32256);
                            float _abs_793;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_793) : "f"(S[19]));
                            unsigned int s_49 = __as_u32(mul_f32_rz_noftz(_abs_793, __uint_as_float(125829120)));
                            unsigned int _min_794 = ((s_49) < (260046848) ? (s_49) : (260046848));
                            unsigned int h_51 = _min_794 + ((h_noise[9] & 536805376) >> 16) >> 13;
                            unsigned int mag_49 = ((s_49 <= 2139095040) ? h_51 : 32256);
                            uint32_t mags_23;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_23) : "r"(mag_48), "r"(mag_49));
                            uint32_t signs_23;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_23) : "r"(__as_u32(S[18])), "r"(__as_u32(S[19])));
                            unsigned int word_53 = signs_23 & 2147516416u | mags_23;
                            h_words[1] = word_53;
                            reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_795;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_795) : "f"(S[24]));
                            unsigned int s_50 = __as_u32(mul_f32_rz_noftz(_abs_795, __uint_as_float(125829120)));
                            unsigned int _min_796 = ((s_50) < (260046848) ? (s_50) : (260046848));
                            unsigned int h_52 = _min_796 + (h_noise[10] & 8191) >> 13;
                            unsigned int mag_50 = ((s_50 <= 2139095040) ? h_52 : 32256);
                            float _abs_797;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_797) : "f"(S[25]));
                            unsigned int s_51 = __as_u32(mul_f32_rz_noftz(_abs_797, __uint_as_float(125829120)));
                            unsigned int _min_798 = ((s_51) < (260046848) ? (s_51) : (260046848));
                            unsigned int h_53 = _min_798 + ((h_noise[10] & 536805376) >> 16) >> 13;
                            unsigned int mag_51 = ((s_51 <= 2139095040) ? h_53 : 32256);
                            uint32_t mags_24;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_24) : "r"(mag_50), "r"(mag_51));
                            uint32_t signs_24;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_24) : "r"(__as_u32(S[24])), "r"(__as_u32(S[25])));
                            unsigned int word_54 = signs_24 & 2147516416u | mags_24;
                            h_words[0] = word_54;
                            float _abs_799;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_799) : "f"(S[26]));
                            unsigned int s_52 = __as_u32(mul_f32_rz_noftz(_abs_799, __uint_as_float(125829120)));
                            unsigned int _min_800 = ((s_52) < (260046848) ? (s_52) : (260046848));
                            unsigned int h_54 = _min_800 + (h_noise[11] & 8191) >> 13;
                            unsigned int mag_52 = ((s_52 <= 2139095040) ? h_54 : 32256);
                            float _abs_801;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_801) : "f"(S[27]));
                            unsigned int s_53 = __as_u32(mul_f32_rz_noftz(_abs_801, __uint_as_float(125829120)));
                            unsigned int _min_802 = ((s_53) < (260046848) ? (s_53) : (260046848));
                            unsigned int h_55 = _min_802 + ((h_noise[11] & 536805376) >> 16) >> 13;
                            unsigned int mag_53 = ((s_53 <= 2139095040) ? h_55 : 32256);
                            uint32_t mags_25;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_25) : "r"(mag_52), "r"(mag_53));
                            uint32_t signs_25;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_25) : "r"(__as_u32(S[26])), "r"(__as_u32(S[27])));
                            unsigned int word_55 = signs_25 & 2147516416u | mags_25;
                            h_words[1] = word_55;
                            reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_803;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_803) : "f"(S[20]));
                            unsigned int s_54 = __as_u32(mul_f32_rz_noftz(_abs_803, __uint_as_float(125829120)));
                            unsigned int _min_804 = ((s_54) < (260046848) ? (s_54) : (260046848));
                            unsigned int h_56 = _min_804 + (h_noise[12] & 8191) >> 13;
                            unsigned int mag_54 = ((s_54 <= 2139095040) ? h_56 : 32256);
                            float _abs_805;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_805) : "f"(S[21]));
                            unsigned int s_55 = __as_u32(mul_f32_rz_noftz(_abs_805, __uint_as_float(125829120)));
                            unsigned int _min_806 = ((s_55) < (260046848) ? (s_55) : (260046848));
                            unsigned int h_57 = _min_806 + ((h_noise[12] & 536805376) >> 16) >> 13;
                            unsigned int mag_55 = ((s_55 <= 2139095040) ? h_57 : 32256);
                            uint32_t mags_26;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_26) : "r"(mag_54), "r"(mag_55));
                            uint32_t signs_26;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_26) : "r"(__as_u32(S[20])), "r"(__as_u32(S[21])));
                            unsigned int word_56 = signs_26 & 2147516416u | mags_26;
                            h_words[0] = word_56;
                            float _abs_807;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_807) : "f"(S[22]));
                            unsigned int s_56 = __as_u32(mul_f32_rz_noftz(_abs_807, __uint_as_float(125829120)));
                            unsigned int _min_808 = ((s_56) < (260046848) ? (s_56) : (260046848));
                            unsigned int h_58 = _min_808 + (h_noise[13] & 8191) >> 13;
                            unsigned int mag_56 = ((s_56 <= 2139095040) ? h_58 : 32256);
                            float _abs_809;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_809) : "f"(S[23]));
                            unsigned int s_57 = __as_u32(mul_f32_rz_noftz(_abs_809, __uint_as_float(125829120)));
                            unsigned int _min_810 = ((s_57) < (260046848) ? (s_57) : (260046848));
                            unsigned int h_59 = _min_810 + ((h_noise[13] & 536805376) >> 16) >> 13;
                            unsigned int mag_57 = ((s_57 <= 2139095040) ? h_59 : 32256);
                            uint32_t mags_27;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_27) : "r"(mag_56), "r"(mag_57));
                            uint32_t signs_27;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_27) : "r"(__as_u32(S[22])), "r"(__as_u32(S[23])));
                            unsigned int word_57 = signs_27 & 2147516416u | mags_27;
                            h_words[1] = word_57;
                            reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_811;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_811) : "f"(S[28]));
                            unsigned int s_58 = __as_u32(mul_f32_rz_noftz(_abs_811, __uint_as_float(125829120)));
                            unsigned int _min_812 = ((s_58) < (260046848) ? (s_58) : (260046848));
                            unsigned int h_60 = _min_812 + (h_noise[14] & 8191) >> 13;
                            unsigned int mag_58 = ((s_58 <= 2139095040) ? h_60 : 32256);
                            float _abs_813;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_813) : "f"(S[29]));
                            unsigned int s_59 = __as_u32(mul_f32_rz_noftz(_abs_813, __uint_as_float(125829120)));
                            unsigned int _min_814 = ((s_59) < (260046848) ? (s_59) : (260046848));
                            unsigned int h_61 = _min_814 + ((h_noise[14] & 536805376) >> 16) >> 13;
                            unsigned int mag_59 = ((s_59 <= 2139095040) ? h_61 : 32256);
                            uint32_t mags_28;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_28) : "r"(mag_58), "r"(mag_59));
                            uint32_t signs_28;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_28) : "r"(__as_u32(S[28])), "r"(__as_u32(S[29])));
                            unsigned int word_58 = signs_28 & 2147516416u | mags_28;
                            h_words[0] = word_58;
                            float _abs_815;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_815) : "f"(S[30]));
                            unsigned int s_60 = __as_u32(mul_f32_rz_noftz(_abs_815, __uint_as_float(125829120)));
                            unsigned int _min_816 = ((s_60) < (260046848) ? (s_60) : (260046848));
                            unsigned int h_62 = _min_816 + (h_noise[15] & 8191) >> 13;
                            unsigned int mag_60 = ((s_60 <= 2139095040) ? h_62 : 32256);
                            float _abs_817;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_817) : "f"(S[31]));
                            unsigned int s_61 = __as_u32(mul_f32_rz_noftz(_abs_817, __uint_as_float(125829120)));
                            unsigned int _min_818 = ((s_61) < (260046848) ? (s_61) : (260046848));
                            unsigned int h_63 = _min_818 + ((h_noise[15] & 536805376) >> 16) >> 13;
                            unsigned int mag_61 = ((s_61 <= 2139095040) ? h_63 : 32256);
                            uint32_t mags_29;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_29) : "r"(mag_60), "r"(mag_61));
                            uint32_t signs_29;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_29) : "r"(__as_u32(S[30])), "r"(__as_u32(S[31])));
                            unsigned int word_59 = signs_29 & 2147516416u | mags_29;
                            h_words[1] = word_59;
                            reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        }
                    }
                    if (philox_rounds == 0) {
                        uint32_t _cvt_rn_f16x2_f32_819;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_819) : "f"(S[1]), "f"(S[0]));
                        h_words[0] = _cvt_rn_f16x2_f32_819;
                        uint32_t _cvt_rn_f16x2_f32_820;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_820) : "f"(S[3]), "f"(S[2]));
                        h_words[1] = _cvt_rn_f16x2_f32_820;
                        reinterpret_cast<int2*>(state + (write_rows_base))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_821;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_821) : "f"(S[5]), "f"(S[4]));
                        h_words[0] = _cvt_rn_f16x2_f32_821;
                        uint32_t _cvt_rn_f16x2_f32_822;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_822) : "f"(S[7]), "f"(S[6]));
                        h_words[1] = _cvt_rn_f16x2_f32_822;
                        reinterpret_cast<int2*>(state + (write_rows_base + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_823;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_823) : "f"(S[9]), "f"(S[8]));
                        h_words[0] = _cvt_rn_f16x2_f32_823;
                        uint32_t _cvt_rn_f16x2_f32_824;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_824) : "f"(S[11]), "f"(S[10]));
                        h_words[1] = _cvt_rn_f16x2_f32_824;
                        reinterpret_cast<int2*>(state + (write_rows_base + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_825;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_825) : "f"(S[13]), "f"(S[12]));
                        h_words[0] = _cvt_rn_f16x2_f32_825;
                        uint32_t _cvt_rn_f16x2_f32_826;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_826) : "f"(S[15]), "f"(S[14]));
                        h_words[1] = _cvt_rn_f16x2_f32_826;
                        reinterpret_cast<int2*>(state + (write_rows_base + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_827;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_827) : "f"(S[17]), "f"(S[16]));
                        h_words[0] = _cvt_rn_f16x2_f32_827;
                        uint32_t _cvt_rn_f16x2_f32_828;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_828) : "f"(S[19]), "f"(S[18]));
                        h_words[1] = _cvt_rn_f16x2_f32_828;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_829;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_829) : "f"(S[21]), "f"(S[20]));
                        h_words[0] = _cvt_rn_f16x2_f32_829;
                        uint32_t _cvt_rn_f16x2_f32_830;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_830) : "f"(S[23]), "f"(S[22]));
                        h_words[1] = _cvt_rn_f16x2_f32_830;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_831;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_831) : "f"(S[25]), "f"(S[24]));
                        h_words[0] = _cvt_rn_f16x2_f32_831;
                        uint32_t _cvt_rn_f16x2_f32_832;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_832) : "f"(S[27]), "f"(S[26]));
                        h_words[1] = _cvt_rn_f16x2_f32_832;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_833;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_833) : "f"(S[29]), "f"(S[28]));
                        h_words[0] = _cvt_rn_f16x2_f32_833;
                        uint32_t _cvt_rn_f16x2_f32_834;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_834) : "f"(S[31]), "f"(S[30]));
                        h_words[1] = _cvt_rn_f16x2_f32_834;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                    }
                }
            }
        }
    }
}

} // extern "C"
