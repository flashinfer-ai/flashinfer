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
    return r;
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdp_bc76e12e0c247b44a27d(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __half* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, int* __restrict__ ssm_state_indices, long long* __restrict__ rand_seed, CakeFastDivmod heads, int num_tokens, float scale, int disable_state_update, int use_ssm_state_indices, int philox_rounds, long long state_size_p0, long long state_stride_p1, long long state_stride_p2, long long state_stride_p0, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long ssm_state_indices_stride_p0, long long ssm_state_indices_stride_p1, long long out_stride_p0, long long out_stride_p1, long long out_stride_p2)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
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
#if __CUDA_ARCH__ == 1030
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
        float2 S2[32];
        float k_regs[16];
        float2 k2[8];
        float v_regs[4];
        float o4[4];
        unsigned int h_words[2];
        unsigned int h_noise[32];
        {
#if __CUDA_ARCH__ == 1030
            uint2 _vld_4;
            _vld_4 = *reinterpret_cast<const uint2*>(state + read_rows_base);
            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
#else
            uint2 _vld_0;
            _vld_0 = *reinterpret_cast<const uint2*>(state + read_rows_base);
            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
#endif
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
#if __CUDA_ARCH__ == 1030
                    : "r"(_vpairs_4[_pair]));
#else
                    : "r"(_vpairs_0[_pair]));
#endif
            }
        }
        {
#if __CUDA_ARCH__ == 1030
            uint2 _vld_5;
            _vld_5 = *reinterpret_cast<const uint2*>(state + read_rows_base + 32ll);
            uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5);
#else
            uint2 _vld_1;
            _vld_1 = *reinterpret_cast<const uint2*>(state + read_rows_base + 32ll);
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
#endif
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
#if __CUDA_ARCH__ == 1030
                    : "r"(_vpairs_5[_pair]));
#else
                    : "r"(_vpairs_1[_pair]));
#endif
            }
        }
        {
#if __CUDA_ARCH__ == 1030
            uint2 _vld_6;
            _vld_6 = *reinterpret_cast<const uint2*>(state + read_rows_base + 64ll);
            uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6);
#else
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(state + read_rows_base + 64ll);
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
#endif
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
#if __CUDA_ARCH__ == 1030
                    : "r"(_vpairs_6[_pair]));
#else
                    : "r"(_vpairs_2[_pair]));
#endif
            }
        }
        {
#if __CUDA_ARCH__ == 1030
            uint2 _vld_7;
            _vld_7 = *reinterpret_cast<const uint2*>(state + read_rows_base + 96ll);
            uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7);
#else
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(state + read_rows_base + 96ll);
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
#endif
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
#if __CUDA_ARCH__ == 1030
                    : "r"(_vpairs_7[_pair]));
#else
                    : "r"(_vpairs_3[_pair]));
#endif
            }
        }
        {
#if __CUDA_ARCH__ == 1030
            uint2 _vld_8;
            _vld_8 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2);
            uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8);
#else
            uint2 _vld_4;
            _vld_4 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2);
            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
#endif
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
#if __CUDA_ARCH__ == 1030
                    : "r"(_vpairs_8[_pair]));
#else
                    : "r"(_vpairs_4[_pair]));
#endif
            }
        }
        {
#if __CUDA_ARCH__ == 1030
            uint2 _vld_9;
            _vld_9 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 32ll);
            uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9);
#else
            uint2 _vld_5;
            _vld_5 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 32ll);
            uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5);
#endif
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
#if __CUDA_ARCH__ == 1030
                    : "r"(_vpairs_9[_pair]));
#else
                    : "r"(_vpairs_5[_pair]));
#endif
            }
        }
        {
#if __CUDA_ARCH__ == 1030
            uint2 _vld_10;
            _vld_10 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 64ll);
            uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10);
#else
            uint2 _vld_6;
            _vld_6 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 64ll);
            uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6);
#endif
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
#if __CUDA_ARCH__ == 1030
                    : "r"(_vpairs_10[_pair]));
#else
                    : "r"(_vpairs_6[_pair]));
#endif
            }
        }
        {
#if __CUDA_ARCH__ == 1030
            uint2 _vld_11;
            _vld_11 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 96ll);
            uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11);
#else
            uint2 _vld_7;
            _vld_7 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 96ll);
            uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7);
#endif
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
#if __CUDA_ARCH__ == 1030
                    : "r"(_vpairs_11[_pair]));
#else
                    : "r"(_vpairs_7[_pair]));
#endif
            }
        }
        {
#if __CUDA_ARCH__ == 1030
            uint2 _vld_12;
            _vld_12 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2);
            uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12);
#else
            uint2 _vld_8;
            _vld_8 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2);
            uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8);
#endif
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[32 + _pair * 2]))
#if __CUDA_ARCH__ == 1030
                    : "r"(_vpairs_12[_pair]));
#else
                    : "r"(_vpairs_8[_pair]));
#endif
            }
        }
        {
#if __CUDA_ARCH__ == 1030
            uint2 _vld_13;
            _vld_13 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2 + 32ll);
            uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13);
#else
            uint2 _vld_9;
            _vld_9 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2 + 32ll);
            uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9);
#endif
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[36 + _pair * 2]))
#if __CUDA_ARCH__ == 1030
                    : "r"(_vpairs_13[_pair]));
#else
                    : "r"(_vpairs_9[_pair]));
#endif
            }
        }
        {
#if __CUDA_ARCH__ == 1030
            uint2 _vld_14;
            _vld_14 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2 + 64ll);
            uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14);
#else
            uint2 _vld_10;
            _vld_10 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2 + 64ll);
            uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10);
#endif
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[40 + _pair * 2]))
#if __CUDA_ARCH__ == 1030
                    : "r"(_vpairs_14[_pair]));
#else
                    : "r"(_vpairs_10[_pair]));
#endif
            }
        }
        {
#if __CUDA_ARCH__ == 1030
            uint2 _vld_15;
            _vld_15 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2 + 96ll);
            uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15);
#else
            uint2 _vld_11;
            _vld_11 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2 + 96ll);
            uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11);
#endif
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[44 + _pair * 2]))
#if __CUDA_ARCH__ == 1030
                    : "r"(_vpairs_15[_pair]));
#else
                    : "r"(_vpairs_11[_pair]));
#endif
            }
        }
        {
#if __CUDA_ARCH__ == 1030
            uint2 _vld_16;
            _vld_16 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2);
            uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16);
#else
            uint2 _vld_12;
            _vld_12 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2);
            uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12);
#endif
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[48 + _pair * 2]))
#if __CUDA_ARCH__ == 1030
                    : "r"(_vpairs_16[_pair]));
#else
                    : "r"(_vpairs_12[_pair]));
#endif
            }
        }
        {
#if __CUDA_ARCH__ == 1030
            uint2 _vld_17;
            _vld_17 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2 + 32ll);
            uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17);
#else
            uint2 _vld_13;
            _vld_13 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2 + 32ll);
            uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13);
#endif
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[52 + _pair * 2]))
#if __CUDA_ARCH__ == 1030
                    : "r"(_vpairs_17[_pair]));
#else
                    : "r"(_vpairs_13[_pair]));
#endif
            }
        }
        {
#if __CUDA_ARCH__ == 1030
            uint2 _vld_18;
            _vld_18 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2 + 64ll);
            uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18);
#else
            uint2 _vld_14;
            _vld_14 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2 + 64ll);
            uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14);
#endif
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[56 + _pair * 2]))
#if __CUDA_ARCH__ == 1030
                    : "r"(_vpairs_18[_pair]));
#else
                    : "r"(_vpairs_14[_pair]));
#endif
            }
        }
        {
#if __CUDA_ARCH__ == 1030
            uint2 _vld_19;
            _vld_19 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2 + 96ll);
            uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19);
#else
            uint2 _vld_15;
            _vld_15 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2 + 96ll);
            uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15);
#endif
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[60 + _pair * 2]))
#if __CUDA_ARCH__ == 1030
                    : "r"(_vpairs_19[_pair]));
#else
                    : "r"(_vpairs_15[_pair]));
#endif
            }
        }
        unsigned int noise_lane = sNoise_addr + (unsigned int)(tid_0 * 16);
#if __CUDA_ARCH__ == 1030
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
        float2 x0 = make_float2(bk[0], bk[1]);
        float2 x1 = make_float2(bk[2], bk[3]);
        float2 sum_pair = fma_f32x2_rn_ftz(x0, x0, make_float2(0.0f, 0.0f));
        sum_pair = fma_f32x2_rn_ftz(x1, x1, sum_pair);
        float sum_x = sum_pair.x + sum_pair.y;
        float _warp_reduce_20 = sum_x;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_20 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_20, offset);
        float norm = rsqrtf(_warp_reduce_20 + 1e-06f);
        float2 norm2 = make_float2(norm, norm);
        x0 = mul_f32x2(x0, norm2);
        x1 = mul_f32x2(x1, norm2);
        sK[fm_0 * 128 + k_col0] = x0.x;
        sK[fm_0 * 128 + k_col0 + 1] = x0.y;
        sK[fm_0 * 128 + k_col0 + 2] = x1.x;
        sK[fm_0 * 128 + k_col0 + 3] = x1.y;
        int v_col = lane_local * 2;
        sV[fm_0 * 64 + v_col] = bv[0];
        sV[fm_0 * 64 + v_col + 1] = bv[1];
        if (lane_local == 0) {
            float _exp_22 = __expf(-bb[0]);
            float _rcp_21 = approx_rcp(1.0f + _exp_22);
            sBeta[fm_0] = _rcp_21;
        }
        int fm_1 = 4 + warp_local;
        float2 x0_0 = make_float2(bk[4], bk[5]);
        float2 x1_0 = make_float2(bk[6], bk[7]);
        float2 sum_pair_0 = fma_f32x2_rn_ftz(x0_0, x0_0, make_float2(0.0f, 0.0f));
        sum_pair_0 = fma_f32x2_rn_ftz(x1_0, x1_0, sum_pair_0);
        float sum_x_0 = sum_pair_0.x + sum_pair_0.y;
        float _warp_reduce_23 = sum_x_0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_23 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_23, offset);
        float norm_0 = rsqrtf(_warp_reduce_23 + 1e-06f);
        float2 norm2_0 = make_float2(norm_0, norm_0);
        x0_0 = mul_f32x2(x0_0, norm2_0);
        x1_0 = mul_f32x2(x1_0, norm2_0);
        sK[fm_1 * 128 + k_col0] = x0_0.x;
        sK[fm_1 * 128 + k_col0 + 1] = x0_0.y;
        sK[fm_1 * 128 + k_col0 + 2] = x1_0.x;
        sK[fm_1 * 128 + k_col0 + 3] = x1_0.y;
        int v_col_0 = lane_local * 2;
        sV[fm_1 * 64 + v_col_0] = bv[2];
        sV[fm_1 * 64 + v_col_0 + 1] = bv[3];
        if (lane_local == 0) {
            float _exp_25 = __expf(-bb[1]);
            float _rcp_24 = approx_rcp(1.0f + _exp_25);
            sBeta[fm_1] = _rcp_24;
        }
        int fm_2 = 8 + warp_local;
        float2 x0_1 = make_float2(bk[8], bk[9]);
        float2 x1_1 = make_float2(bk[10], bk[11]);
        float2 sum_pair_1 = fma_f32x2_rn_ftz(x0_1, x0_1, make_float2(0.0f, 0.0f));
        sum_pair_1 = fma_f32x2_rn_ftz(x1_1, x1_1, sum_pair_1);
        float sum_x_1 = sum_pair_1.x + sum_pair_1.y;
        float _warp_reduce_26 = sum_x_1;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_26 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_26, offset);
        float norm_1 = rsqrtf(_warp_reduce_26 + 1e-06f);
        float2 norm2_1 = make_float2(norm_1, norm_1);
        x0_1 = mul_f32x2(x0_1, norm2_1);
        x1_1 = mul_f32x2(x1_1, norm2_1);
        sK[fm_2 * 128 + k_col0] = x0_1.x;
        sK[fm_2 * 128 + k_col0 + 1] = x0_1.y;
        sK[fm_2 * 128 + k_col0 + 2] = x1_1.x;
        sK[fm_2 * 128 + k_col0 + 3] = x1_1.y;
        int v_col_1 = lane_local * 2;
        sV[fm_2 * 64 + v_col_1] = bv[4];
        sV[fm_2 * 64 + v_col_1 + 1] = bv[5];
        if (lane_local == 0) {
            float _exp_28 = __expf(-bb[2]);
            float _rcp_27 = approx_rcp(1.0f + _exp_28);
            sBeta[fm_2] = _rcp_27;
        }
        int fm_3 = 12 + warp_local;
        float2 x0_2 = make_float2(bk[12], bk[13]);
        float2 x1_2 = make_float2(bk[14], bk[15]);
        float2 sum_pair_2 = fma_f32x2_rn_ftz(x0_2, x0_2, make_float2(0.0f, 0.0f));
        sum_pair_2 = fma_f32x2_rn_ftz(x1_2, x1_2, sum_pair_2);
        float sum_x_2 = sum_pair_2.x + sum_pair_2.y;
        float _warp_reduce_29 = sum_x_2;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_29 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_29, offset);
        float norm_2 = rsqrtf(_warp_reduce_29 + 1e-06f);
        float2 norm2_2 = make_float2(norm_2, norm_2);
        x0_2 = mul_f32x2(x0_2, norm2_2);
        x1_2 = mul_f32x2(x1_2, norm2_2);
        sK[fm_3 * 128 + k_col0] = x0_2.x;
        sK[fm_3 * 128 + k_col0 + 1] = x0_2.y;
        sK[fm_3 * 128 + k_col0 + 2] = x1_2.x;
        sK[fm_3 * 128 + k_col0 + 3] = x1_2.y;
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
        int bm_5 = ((bm0_5) < (steps - 1) ? (bm0_5) : (steps - 1));
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
        int bt0_1 = 4 + warp_local;
        int bt_1 = ((bt0_1) < (num_tokens - 1) ? (bt0_1) : (num_tokens - 1));
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
        int fm_4 = 16 + warp_local;
        float2 x0_3 = make_float2(bk_0[0], bk_0[1]);
        float2 x1_3 = make_float2(bk_0[2], bk_0[3]);
        float2 sum_pair_3 = fma_f32x2_rn_ftz(x0_3, x0_3, make_float2(0.0f, 0.0f));
        sum_pair_3 = fma_f32x2_rn_ftz(x1_3, x1_3, sum_pair_3);
        float sum_x_3 = sum_pair_3.x + sum_pair_3.y;
        float _warp_reduce_36 = sum_x_3;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_36 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_36, offset);
        float norm_3 = rsqrtf(_warp_reduce_36 + 1e-06f);
        float2 norm2_3 = make_float2(norm_3, norm_3);
        x0_3 = mul_f32x2(x0_3, norm2_3);
        x1_3 = mul_f32x2(x1_3, norm2_3);
        sK[fm_4 * 128 + k_col0] = x0_3.x;
        sK[fm_4 * 128 + k_col0 + 1] = x0_3.y;
        sK[fm_4 * 128 + k_col0 + 2] = x1_3.x;
        sK[fm_4 * 128 + k_col0 + 3] = x1_3.y;
        int v_col_3 = lane_local * 2;
        sV[fm_4 * 64 + v_col_3] = bv_0[0];
        sV[fm_4 * 64 + v_col_3 + 1] = bv_0[1];
        if (lane_local == 0) {
            float _exp_38 = __expf(-bb_0[0]);
            float _rcp_37 = approx_rcp(1.0f + _exp_38);
            sBeta[fm_4] = _rcp_37;
        }
        int fm_5 = 20 + warp_local;
        if (fm_5 < steps) {
            float2 x0_4 = make_float2(bk_0[4], bk_0[5]);
            float2 x1_4 = make_float2(bk_0[6], bk_0[7]);
            float2 sum_pair_4 = fma_f32x2_rn_ftz(x0_4, x0_4, make_float2(0.0f, 0.0f));
            sum_pair_4 = fma_f32x2_rn_ftz(x1_4, x1_4, sum_pair_4);
            float sum_x_4 = sum_pair_4.x + sum_pair_4.y;
            float _warp_reduce_39 = sum_x_4;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_39 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_39, offset);
            float norm_4 = rsqrtf(_warp_reduce_39 + 1e-06f);
            float2 norm2_4 = make_float2(norm_4, norm_4);
            x0_4 = mul_f32x2(x0_4, norm2_4);
            x1_4 = mul_f32x2(x1_4, norm2_4);
            sK[fm_5 * 128 + k_col0] = x0_4.x;
            sK[fm_5 * 128 + k_col0 + 1] = x0_4.y;
            sK[fm_5 * 128 + k_col0 + 2] = x1_4.x;
            sK[fm_5 * 128 + k_col0 + 3] = x1_4.y;
            int v_col_4 = lane_local * 2;
            sV[fm_5 * 64 + v_col_4] = bv_0[2];
            sV[fm_5 * 64 + v_col_4 + 1] = bv_0[3];
            if (lane_local == 0) {
                float _exp_41 = __expf(-bb_0[1]);
                float _rcp_40 = approx_rcp(1.0f + _exp_41);
                sBeta[fm_5] = _rcp_40;
            }
        }
        int ft_0 = warp_local;
        float2 x0_5 = make_float2(bq[0], bq[1]);
        float2 x1_5 = make_float2(bq[2], bq[3]);
        float2 sum_pair_5 = fma_f32x2_rn_ftz(x0_5, x0_5, make_float2(0.0f, 0.0f));
        sum_pair_5 = fma_f32x2_rn_ftz(x1_5, x1_5, sum_pair_5);
        float sum_x_5 = sum_pair_5.x + sum_pair_5.y;
        float _warp_reduce_42 = sum_x_5;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_42 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_42, offset);
        float norm_5 = rsqrtf(_warp_reduce_42 + 1e-06f);
        norm_5 = norm_5 * scale;
        float2 norm2_5 = make_float2(norm_5, norm_5);
        x0_5 = mul_f32x2(x0_5, norm2_5);
        x1_5 = mul_f32x2(x1_5, norm2_5);
        sQ[ft_0 * 128 + k_col0] = x0_5.x;
        sQ[ft_0 * 128 + k_col0 + 1] = x0_5.y;
        sQ[ft_0 * 128 + k_col0 + 2] = x1_5.x;
        sQ[ft_0 * 128 + k_col0 + 3] = x1_5.y;
        if (lane_local == 0) {
            float xv = ba[0] + ba[1];
            float softplus_x = xv;
            if (xv <= 20.0f) {
                float _exp_44 = __expf(xv);
                float _log2_43;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_43) : "f"(1.0f + _exp_44));
                softplus_x = _log2_43 * 0.6931472f;
            }
            float _exp_45 = __expf(ba[2]);
            float g_log = -_exp_45 * softplus_x;
            float _exp_46 = __expf(g_log);
            sAlpha[ft_0] = _exp_46;
            if (use_ssm_state_indices != 0) {
                sSsm[ft_0] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)ft_0 * ssm_state_indices_stride_p1];
            }
            if (use_ssm_state_indices == 0) {
                sSsm[ft_0] = -1;
            }
        }
        int ft_1 = 4 + warp_local;
        if (ft_1 < num_tokens) {
            float2 x0_6 = make_float2(bq[4], bq[5]);
            float2 x1_6 = make_float2(bq[6], bq[7]);
            float2 sum_pair_6 = fma_f32x2_rn_ftz(x0_6, x0_6, make_float2(0.0f, 0.0f));
            sum_pair_6 = fma_f32x2_rn_ftz(x1_6, x1_6, sum_pair_6);
            float sum_x_6 = sum_pair_6.x + sum_pair_6.y;
            float _warp_reduce_47 = sum_x_6;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_47 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_47, offset);
            float norm_6 = rsqrtf(_warp_reduce_47 + 1e-06f);
            norm_6 = norm_6 * scale;
            float2 norm2_6 = make_float2(norm_6, norm_6);
            x0_6 = mul_f32x2(x0_6, norm2_6);
            x1_6 = mul_f32x2(x1_6, norm2_6);
            sQ[ft_1 * 128 + k_col0] = x0_6.x;
            sQ[ft_1 * 128 + k_col0 + 1] = x0_6.y;
            sQ[ft_1 * 128 + k_col0 + 2] = x1_6.x;
            sQ[ft_1 * 128 + k_col0 + 3] = x1_6.y;
            if (lane_local == 0) {
                float xv_1 = ba[3] + ba[4];
                float softplus_x_1 = xv_1;
                if (xv_1 <= 20.0f) {
                    float _exp_49 = __expf(xv_1);
                    float _log2_48;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_48) : "f"(1.0f + _exp_49));
                    softplus_x_1 = _log2_48 * 0.6931472f;
                }
                float _exp_50 = __expf(ba[5]);
                float g_log_1 = -_exp_50 * softplus_x_1;
                float _exp_51 = __expf(g_log_1);
                sAlpha[ft_1] = _exp_51;
                if (use_ssm_state_indices != 0) {
                    sSsm[ft_1] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)ft_1 * ssm_state_indices_stride_p1];
                }
                if (use_ssm_state_indices == 0) {
                    sSsm[ft_1] = -1;
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
                long long o_4 = read_rows_base + 2ll * state_stride_p2;
                unsigned long long p0_4 = (unsigned long long)(unsigned int)o_4 * 3528531795ull;
                h_noise[16] = (unsigned int)(o_4 >> 32ll) ^ key0;
                h_noise[18] = (unsigned int)(p0_4 >> 32ull) ^ key1;
                h_noise[19] = (unsigned int)p0_4;
                long long o_5 = read_rows_base + 2ll * state_stride_p2 + 32ll;
                unsigned long long p0_5 = (unsigned long long)(unsigned int)o_5 * 3528531795ull;
                h_noise[20] = (unsigned int)(o_5 >> 32ll) ^ key0;
                h_noise[22] = (unsigned int)(p0_5 >> 32ull) ^ key1;
                h_noise[23] = (unsigned int)p0_5;
                long long o_6 = read_rows_base + 3ll * state_stride_p2;
                unsigned long long p0_6 = (unsigned long long)(unsigned int)o_6 * 3528531795ull;
                h_noise[24] = (unsigned int)(o_6 >> 32ll) ^ key0;
                h_noise[26] = (unsigned int)(p0_6 >> 32ull) ^ key1;
                h_noise[27] = (unsigned int)p0_6;
                long long o_7 = read_rows_base + 3ll * state_stride_p2 + 32ll;
                unsigned long long p0_7 = (unsigned long long)(unsigned int)o_7 * 3528531795ull;
                h_noise[28] = (unsigned int)(o_7 >> 32ll) ^ key0;
                h_noise[30] = (unsigned int)(p0_7 >> 32ull) ^ key1;
                h_noise[31] = (unsigned int)p0_7;
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
                unsigned long long q0_4 = (unsigned long long)h_noise[16] * 3528531795ull;
                unsigned long long q1_4 = (unsigned long long)h_noise[18] * 3449720151ull;
                h_noise[16] = (unsigned int)(q1_4 >> 32ull) ^ key0;
                h_noise[17] = (unsigned int)q1_4;
                h_noise[18] = (unsigned int)(q0_4 >> 32ull) ^ h_noise[19] ^ key1;
                h_noise[19] = (unsigned int)q0_4;
                unsigned long long q0_5 = (unsigned long long)h_noise[20] * 3528531795ull;
                unsigned long long q1_5 = (unsigned long long)h_noise[22] * 3449720151ull;
                h_noise[20] = (unsigned int)(q1_5 >> 32ull) ^ key0;
                h_noise[21] = (unsigned int)q1_5;
                h_noise[22] = (unsigned int)(q0_5 >> 32ull) ^ h_noise[23] ^ key1;
                h_noise[23] = (unsigned int)q0_5;
                unsigned long long q0_6 = (unsigned long long)h_noise[24] * 3528531795ull;
                unsigned long long q1_6 = (unsigned long long)h_noise[26] * 3449720151ull;
                h_noise[24] = (unsigned int)(q1_6 >> 32ull) ^ key0;
                h_noise[25] = (unsigned int)q1_6;
                h_noise[26] = (unsigned int)(q0_6 >> 32ull) ^ h_noise[27] ^ key1;
                h_noise[27] = (unsigned int)q0_6;
                unsigned long long q0_7 = (unsigned long long)h_noise[28] * 3528531795ull;
                unsigned long long q1_7 = (unsigned long long)h_noise[30] * 3449720151ull;
                h_noise[28] = (unsigned int)(q1_7 >> 32ull) ^ key0;
                h_noise[29] = (unsigned int)q1_7;
                h_noise[30] = (unsigned int)(q0_7 >> 32ull) ^ h_noise[31] ^ key1;
                h_noise[31] = (unsigned int)q0_7;
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
                unsigned long long s0_4 = (unsigned long long)h_noise[16] * 3528531795ull;
                unsigned long long s1_4 = (unsigned long long)h_noise[18] * 3449720151ull;
                h_noise[16] = (unsigned int)(s1_4 >> 32ull) ^ h_noise[17] ^ key0;
                h_noise[17] = (unsigned int)s1_4;
                h_noise[18] = (unsigned int)(s0_4 >> 32ull) ^ h_noise[19] ^ key1;
                h_noise[19] = (unsigned int)s0_4;
                unsigned long long s0_5 = (unsigned long long)h_noise[20] * 3528531795ull;
                unsigned long long s1_5 = (unsigned long long)h_noise[22] * 3449720151ull;
                h_noise[20] = (unsigned int)(s1_5 >> 32ull) ^ h_noise[21] ^ key0;
                h_noise[21] = (unsigned int)s1_5;
                h_noise[22] = (unsigned int)(s0_5 >> 32ull) ^ h_noise[23] ^ key1;
                h_noise[23] = (unsigned int)s0_5;
                unsigned long long s0_6 = (unsigned long long)h_noise[24] * 3528531795ull;
                unsigned long long s1_6 = (unsigned long long)h_noise[26] * 3449720151ull;
                h_noise[24] = (unsigned int)(s1_6 >> 32ull) ^ h_noise[25] ^ key0;
                h_noise[25] = (unsigned int)s1_6;
                h_noise[26] = (unsigned int)(s0_6 >> 32ull) ^ h_noise[27] ^ key1;
                h_noise[27] = (unsigned int)s0_6;
                unsigned long long s0_7 = (unsigned long long)h_noise[28] * 3528531795ull;
                unsigned long long s1_7 = (unsigned long long)h_noise[30] * 3449720151ull;
                h_noise[28] = (unsigned int)(s1_7 >> 32ull) ^ h_noise[29] ^ key0;
                h_noise[29] = (unsigned int)s1_7;
                h_noise[30] = (unsigned int)(s0_7 >> 32ull) ^ h_noise[31] ^ key1;
                h_noise[31] = (unsigned int)s0_7;
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
                unsigned long long s0_12 = (unsigned long long)h_noise[16] * 3528531795ull;
                unsigned long long s1_12 = (unsigned long long)h_noise[18] * 3449720151ull;
                h_noise[16] = (unsigned int)(s1_12 >> 32ull) ^ h_noise[17] ^ key0;
                h_noise[17] = (unsigned int)s1_12;
                h_noise[18] = (unsigned int)(s0_12 >> 32ull) ^ h_noise[19] ^ key1;
                h_noise[19] = (unsigned int)s0_12;
                unsigned long long s0_13 = (unsigned long long)h_noise[20] * 3528531795ull;
                unsigned long long s1_13 = (unsigned long long)h_noise[22] * 3449720151ull;
                h_noise[20] = (unsigned int)(s1_13 >> 32ull) ^ h_noise[21] ^ key0;
                h_noise[21] = (unsigned int)s1_13;
                h_noise[22] = (unsigned int)(s0_13 >> 32ull) ^ h_noise[23] ^ key1;
                h_noise[23] = (unsigned int)s0_13;
                unsigned long long s0_14 = (unsigned long long)h_noise[24] * 3528531795ull;
                unsigned long long s1_14 = (unsigned long long)h_noise[26] * 3449720151ull;
                h_noise[24] = (unsigned int)(s1_14 >> 32ull) ^ h_noise[25] ^ key0;
                h_noise[25] = (unsigned int)s1_14;
                h_noise[26] = (unsigned int)(s0_14 >> 32ull) ^ h_noise[27] ^ key1;
                h_noise[27] = (unsigned int)s0_14;
                unsigned long long s0_15 = (unsigned long long)h_noise[28] * 3528531795ull;
                unsigned long long s1_15 = (unsigned long long)h_noise[30] * 3449720151ull;
                h_noise[28] = (unsigned int)(s1_15 >> 32ull) ^ h_noise[29] ^ key0;
                h_noise[29] = (unsigned int)s1_15;
                h_noise[30] = (unsigned int)(s0_15 >> 32ull) ^ h_noise[31] ^ key1;
                h_noise[31] = (unsigned int)s0_15;
                key0 = key0 + 2654435769u;
                key1 = key1 + 3144134277u;
                unsigned long long s0_16 = (unsigned long long)h_noise[0] * 3528531795ull;
                unsigned long long s1_16 = (unsigned long long)h_noise[2] * 3449720151ull;
                h_noise[0] = (unsigned int)(s1_16 >> 32ull) ^ h_noise[1] ^ key0;
                h_noise[1] = (unsigned int)s1_16;
                h_noise[2] = (unsigned int)(s0_16 >> 32ull) ^ h_noise[3] ^ key1;
                h_noise[3] = (unsigned int)s0_16;
                unsigned long long s0_17 = (unsigned long long)h_noise[4] * 3528531795ull;
                unsigned long long s1_17 = (unsigned long long)h_noise[6] * 3449720151ull;
                h_noise[4] = (unsigned int)(s1_17 >> 32ull) ^ h_noise[5] ^ key0;
                h_noise[5] = (unsigned int)s1_17;
                h_noise[6] = (unsigned int)(s0_17 >> 32ull) ^ h_noise[7] ^ key1;
                h_noise[7] = (unsigned int)s0_17;
                unsigned long long s0_18 = (unsigned long long)h_noise[8] * 3528531795ull;
                unsigned long long s1_18 = (unsigned long long)h_noise[10] * 3449720151ull;
                h_noise[8] = (unsigned int)(s1_18 >> 32ull) ^ h_noise[9] ^ key0;
                h_noise[9] = (unsigned int)s1_18;
                h_noise[10] = (unsigned int)(s0_18 >> 32ull) ^ h_noise[11] ^ key1;
                h_noise[11] = (unsigned int)s0_18;
                unsigned long long s0_19 = (unsigned long long)h_noise[12] * 3528531795ull;
                unsigned long long s1_19 = (unsigned long long)h_noise[14] * 3449720151ull;
                h_noise[12] = (unsigned int)(s1_19 >> 32ull) ^ h_noise[13] ^ key0;
                h_noise[13] = (unsigned int)s1_19;
                h_noise[14] = (unsigned int)(s0_19 >> 32ull) ^ h_noise[15] ^ key1;
                h_noise[15] = (unsigned int)s0_19;
                unsigned long long s0_20 = (unsigned long long)h_noise[16] * 3528531795ull;
                unsigned long long s1_20 = (unsigned long long)h_noise[18] * 3449720151ull;
                h_noise[16] = (unsigned int)(s1_20 >> 32ull) ^ h_noise[17] ^ key0;
                h_noise[17] = (unsigned int)s1_20;
                h_noise[18] = (unsigned int)(s0_20 >> 32ull) ^ h_noise[19] ^ key1;
                h_noise[19] = (unsigned int)s0_20;
                unsigned long long s0_21 = (unsigned long long)h_noise[20] * 3528531795ull;
                unsigned long long s1_21 = (unsigned long long)h_noise[22] * 3449720151ull;
                h_noise[20] = (unsigned int)(s1_21 >> 32ull) ^ h_noise[21] ^ key0;
                h_noise[21] = (unsigned int)s1_21;
                h_noise[22] = (unsigned int)(s0_21 >> 32ull) ^ h_noise[23] ^ key1;
                h_noise[23] = (unsigned int)s0_21;
                unsigned long long s0_22 = (unsigned long long)h_noise[24] * 3528531795ull;
                unsigned long long s1_22 = (unsigned long long)h_noise[26] * 3449720151ull;
                h_noise[24] = (unsigned int)(s1_22 >> 32ull) ^ h_noise[25] ^ key0;
                h_noise[25] = (unsigned int)s1_22;
                h_noise[26] = (unsigned int)(s0_22 >> 32ull) ^ h_noise[27] ^ key1;
                h_noise[27] = (unsigned int)s0_22;
                unsigned long long s0_23 = (unsigned long long)h_noise[28] * 3528531795ull;
                unsigned long long s1_23 = (unsigned long long)h_noise[30] * 3449720151ull;
                h_noise[28] = (unsigned int)(s1_23 >> 32ull) ^ h_noise[29] ^ key0;
                h_noise[29] = (unsigned int)s1_23;
                h_noise[30] = (unsigned int)(s0_23 >> 32ull) ^ h_noise[31] ^ key1;
                h_noise[31] = (unsigned int)s0_23;
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
                    unsigned long long t0_4 = (unsigned long long)h_noise[16] * 3528531795ull;
                    unsigned long long t1_4 = (unsigned long long)h_noise[18] * 3449720151ull;
                    h_noise[16] = (unsigned int)(t1_4 >> 32ull) ^ h_noise[17] ^ key0;
                    h_noise[17] = (unsigned int)t1_4;
                    h_noise[18] = (unsigned int)(t0_4 >> 32ull) ^ h_noise[19] ^ key1;
                    h_noise[19] = (unsigned int)t0_4;
                    unsigned long long t0_5 = (unsigned long long)h_noise[20] * 3528531795ull;
                    unsigned long long t1_5 = (unsigned long long)h_noise[22] * 3449720151ull;
                    h_noise[20] = (unsigned int)(t1_5 >> 32ull) ^ h_noise[21] ^ key0;
                    h_noise[21] = (unsigned int)t1_5;
                    h_noise[22] = (unsigned int)(t0_5 >> 32ull) ^ h_noise[23] ^ key1;
                    h_noise[23] = (unsigned int)t0_5;
                    unsigned long long t0_6 = (unsigned long long)h_noise[24] * 3528531795ull;
                    unsigned long long t1_6 = (unsigned long long)h_noise[26] * 3449720151ull;
                    h_noise[24] = (unsigned int)(t1_6 >> 32ull) ^ h_noise[25] ^ key0;
                    h_noise[25] = (unsigned int)t1_6;
                    h_noise[26] = (unsigned int)(t0_6 >> 32ull) ^ h_noise[27] ^ key1;
                    h_noise[27] = (unsigned int)t0_6;
                    unsigned long long t0_7 = (unsigned long long)h_noise[28] * 3528531795ull;
                    unsigned long long t1_7 = (unsigned long long)h_noise[30] * 3449720151ull;
                    h_noise[28] = (unsigned int)(t1_7 >> 32ull) ^ h_noise[29] ^ key0;
                    h_noise[29] = (unsigned int)t1_7;
                    h_noise[30] = (unsigned int)(t0_7 >> 32ull) ^ h_noise[31] ^ key1;
                    h_noise[31] = (unsigned int)t0_7;
                    key0 = key0 + 2654435769u;
                    key1 = key1 + 3144134277u;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 0), "r"(h_noise[0] & 536813567), "r"(h_noise[1] & 536813567), "r"(h_noise[2] & 536813567), "r"(h_noise[3] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 2048), "r"(h_noise[4] & 536813567), "r"(h_noise[5] & 536813567), "r"(h_noise[6] & 536813567), "r"(h_noise[7] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 4096), "r"(h_noise[8] & 536813567), "r"(h_noise[9] & 536813567), "r"(h_noise[10] & 536813567), "r"(h_noise[11] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 6144), "r"(h_noise[12] & 536813567), "r"(h_noise[13] & 536813567), "r"(h_noise[14] & 536813567), "r"(h_noise[15] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 8192), "r"(h_noise[16] & 536813567), "r"(h_noise[17] & 536813567), "r"(h_noise[18] & 536813567), "r"(h_noise[19] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 10240), "r"(h_noise[20] & 536813567), "r"(h_noise[21] & 536813567), "r"(h_noise[22] & 536813567), "r"(h_noise[23] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 12288), "r"(h_noise[24] & 536813567), "r"(h_noise[25] & 536813567), "r"(h_noise[26] & 536813567), "r"(h_noise[27] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 14336), "r"(h_noise[28] & 536813567), "r"(h_noise[29] & 536813567), "r"(h_noise[30] & 536813567), "r"(h_noise[31] & 536813567) : "memory");
            }
            #pragma unroll 1
            for (int t = 0; t < 7; t++) {
                float alpha = sAlpha[t];
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
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                    : "r"(sV_addr + (unsigned int)(m_0 * 256) + group_byte));
                float beta_0 = sBeta[m_0];
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
                float vn_0 = (v_regs[0] - dot) * beta_0;
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
                float vn_1 = (v_regs[1] - dot_0) * beta_0;
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
                float vn_2 = (v_regs[2] - dot_1) * beta_0;
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
                float vn_3 = (v_regs[3] - dot_2) * beta_0;
                float2 vn2_3 = make_float2(vn_3, vn_3);
                S2[24] = fma_f32x2_rn_ftz(k2[0], vn2_3, S2[24]);
                S2[25] = fma_f32x2_rn_ftz(k2[1], vn2_3, S2[25]);
                S2[26] = fma_f32x2_rn_ftz(k2[2], vn2_3, S2[26]);
                S2[27] = fma_f32x2_rn_ftz(k2[3], vn2_3, S2[27]);
                S2[28] = fma_f32x2_rn_ftz(k2[4], vn2_3, S2[28]);
                S2[29] = fma_f32x2_rn_ftz(k2[5], vn2_3, S2[29]);
                S2[30] = fma_f32x2_rn_ftz(k2[6], vn2_3, S2[30]);
                S2[31] = fma_f32x2_rn_ftz(k2[7], vn2_3, S2[31]);
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
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                    : "r"(sV_addr + (unsigned int)(m_1 * 256) + group_byte));
                float beta_1 = sBeta[m_1];
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
                float vn_4 = (v_regs[0] - dot_3) * beta_1;
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
                float vn_5 = (v_regs[1] - dot_4) * beta_1;
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
                float vn_6 = (v_regs[2] - dot_5) * beta_1;
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
                float vn_7 = (v_regs[3] - dot_6) * beta_1;
                float2 vn2_7 = make_float2(vn_7, vn_7);
                S2[24] = fma_f32x2_rn_ftz(k2[0], vn2_7, S2[24]);
                S2[25] = fma_f32x2_rn_ftz(k2[1], vn2_7, S2[25]);
                S2[26] = fma_f32x2_rn_ftz(k2[2], vn2_7, S2[26]);
                S2[27] = fma_f32x2_rn_ftz(k2[3], vn2_7, S2[27]);
                S2[28] = fma_f32x2_rn_ftz(k2[4], vn2_7, S2[28]);
                S2[29] = fma_f32x2_rn_ftz(k2[5], vn2_7, S2[29]);
                S2[30] = fma_f32x2_rn_ftz(k2[6], vn2_7, S2[30]);
                S2[31] = fma_f32x2_rn_ftz(k2[7], vn2_7, S2[31]);
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
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                    : "r"(sV_addr + (unsigned int)(m_2 * 256) + group_byte));
                float beta_2 = sBeta[m_2];
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
                float vn_8 = (v_regs[0] - dot_7) * beta_2;
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
                float vn_9 = (v_regs[1] - dot_8) * beta_2;
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
                float vn_10 = (v_regs[2] - dot_9) * beta_2;
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
                float vn_11 = (v_regs[3] - dot_10) * beta_2;
                float2 vn2_11 = make_float2(vn_11, vn_11);
                S2[24] = fma_f32x2_rn_ftz(k2[0], vn2_11, S2[24]);
                S2[25] = fma_f32x2_rn_ftz(k2[1], vn2_11, S2[25]);
                S2[26] = fma_f32x2_rn_ftz(k2[2], vn2_11, S2[26]);
                S2[27] = fma_f32x2_rn_ftz(k2[3], vn2_11, S2[27]);
                S2[28] = fma_f32x2_rn_ftz(k2[4], vn2_11, S2[28]);
                S2[29] = fma_f32x2_rn_ftz(k2[5], vn2_11, S2[29]);
                S2[30] = fma_f32x2_rn_ftz(k2[6], vn2_11, S2[30]);
                S2[31] = fma_f32x2_rn_ftz(k2[7], vn2_11, S2[31]);
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
                o4[0] = dot_11;
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
                o4[1] = dot_12;
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
                o4[2] = dot_13;
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
                if (philox_rounds != 0) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 0));
                    uint32_t _cvt_rs_x2_f32_100;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_100) : "f"(S2[0].y), "f"(S2[0].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_100;
                    uint32_t _cvt_rs_x2_f32_101;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_101) : "f"(S2[1].y), "f"(S2[1].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_101;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_102;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_102) : "f"(S2[4].y), "f"(S2[4].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_102;
                    uint32_t _cvt_rs_x2_f32_103;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_103) : "f"(S2[5].y), "f"(S2[5].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_103;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 2048));
                    uint32_t _cvt_rs_x2_f32_104;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_104) : "f"(S2[2].y), "f"(S2[2].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_104;
                    uint32_t _cvt_rs_x2_f32_105;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_105) : "f"(S2[3].y), "f"(S2[3].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_105;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_106;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_106) : "f"(S2[6].y), "f"(S2[6].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_106;
                    uint32_t _cvt_rs_x2_f32_107;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_107) : "f"(S2[7].y), "f"(S2[7].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_107;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 4096));
                    uint32_t _cvt_rs_x2_f32_108;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_108) : "f"(S2[8].y), "f"(S2[8].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_108;
                    uint32_t _cvt_rs_x2_f32_109;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_109) : "f"(S2[9].y), "f"(S2[9].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_109;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_110;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_110) : "f"(S2[12].y), "f"(S2[12].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_110;
                    uint32_t _cvt_rs_x2_f32_111;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_111) : "f"(S2[13].y), "f"(S2[13].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_111;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 6144));
                    uint32_t _cvt_rs_x2_f32_112;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_112) : "f"(S2[10].y), "f"(S2[10].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_112;
                    uint32_t _cvt_rs_x2_f32_113;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_113) : "f"(S2[11].y), "f"(S2[11].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_113;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_114;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_114) : "f"(S2[14].y), "f"(S2[14].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_114;
                    uint32_t _cvt_rs_x2_f32_115;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_115) : "f"(S2[15].y), "f"(S2[15].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_115;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 8192));
                    uint32_t _cvt_rs_x2_f32_116;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_116) : "f"(S2[16].y), "f"(S2[16].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_116;
                    uint32_t _cvt_rs_x2_f32_117;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_117) : "f"(S2[17].y), "f"(S2[17].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_117;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_118;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_118) : "f"(S2[20].y), "f"(S2[20].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_118;
                    uint32_t _cvt_rs_x2_f32_119;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_119) : "f"(S2[21].y), "f"(S2[21].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_119;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 10240));
                    uint32_t _cvt_rs_x2_f32_120;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_120) : "f"(S2[18].y), "f"(S2[18].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_120;
                    uint32_t _cvt_rs_x2_f32_121;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_121) : "f"(S2[19].y), "f"(S2[19].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_121;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_122;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_122) : "f"(S2[22].y), "f"(S2[22].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_122;
                    uint32_t _cvt_rs_x2_f32_123;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_123) : "f"(S2[23].y), "f"(S2[23].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_123;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 12288));
                    uint32_t _cvt_rs_x2_f32_124;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_124) : "f"(S2[24].y), "f"(S2[24].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_124;
                    uint32_t _cvt_rs_x2_f32_125;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_125) : "f"(S2[25].y), "f"(S2[25].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_125;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_126;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_126) : "f"(S2[28].y), "f"(S2[28].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_126;
                    uint32_t _cvt_rs_x2_f32_127;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_127) : "f"(S2[29].y), "f"(S2[29].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_127;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 14336));
                    uint32_t _cvt_rs_x2_f32_128;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_128) : "f"(S2[26].y), "f"(S2[26].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_128;
                    uint32_t _cvt_rs_x2_f32_129;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_129) : "f"(S2[27].y), "f"(S2[27].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_129;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_130;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_130) : "f"(S2[30].y), "f"(S2[30].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_130;
                    uint32_t _cvt_rs_x2_f32_131;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_131) : "f"(S2[31].y), "f"(S2[31].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_131;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                }
                if (philox_rounds == 0) {
                    uint32_t _cvt_rn_f16x2_f32_132;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_132) : "f"(S2[0].y), "f"(S2[0].x));
                    h_words[0] = _cvt_rn_f16x2_f32_132;
                    uint32_t _cvt_rn_f16x2_f32_133;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_133) : "f"(S2[1].y), "f"(S2[1].x));
                    h_words[1] = _cvt_rn_f16x2_f32_133;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_134;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_134) : "f"(S2[2].y), "f"(S2[2].x));
                    h_words[0] = _cvt_rn_f16x2_f32_134;
                    uint32_t _cvt_rn_f16x2_f32_135;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_135) : "f"(S2[3].y), "f"(S2[3].x));
                    h_words[1] = _cvt_rn_f16x2_f32_135;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_136;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_136) : "f"(S2[4].y), "f"(S2[4].x));
                    h_words[0] = _cvt_rn_f16x2_f32_136;
                    uint32_t _cvt_rn_f16x2_f32_137;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_137) : "f"(S2[5].y), "f"(S2[5].x));
                    h_words[1] = _cvt_rn_f16x2_f32_137;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_138;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_138) : "f"(S2[6].y), "f"(S2[6].x));
                    h_words[0] = _cvt_rn_f16x2_f32_138;
                    uint32_t _cvt_rn_f16x2_f32_139;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_139) : "f"(S2[7].y), "f"(S2[7].x));
                    h_words[1] = _cvt_rn_f16x2_f32_139;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_140;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_140) : "f"(S2[8].y), "f"(S2[8].x));
                    h_words[0] = _cvt_rn_f16x2_f32_140;
                    uint32_t _cvt_rn_f16x2_f32_141;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_141) : "f"(S2[9].y), "f"(S2[9].x));
                    h_words[1] = _cvt_rn_f16x2_f32_141;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_142;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_142) : "f"(S2[10].y), "f"(S2[10].x));
                    h_words[0] = _cvt_rn_f16x2_f32_142;
                    uint32_t _cvt_rn_f16x2_f32_143;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_143) : "f"(S2[11].y), "f"(S2[11].x));
                    h_words[1] = _cvt_rn_f16x2_f32_143;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_144;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_144) : "f"(S2[12].y), "f"(S2[12].x));
                    h_words[0] = _cvt_rn_f16x2_f32_144;
                    uint32_t _cvt_rn_f16x2_f32_145;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_145) : "f"(S2[13].y), "f"(S2[13].x));
                    h_words[1] = _cvt_rn_f16x2_f32_145;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_146;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_146) : "f"(S2[14].y), "f"(S2[14].x));
                    h_words[0] = _cvt_rn_f16x2_f32_146;
                    uint32_t _cvt_rn_f16x2_f32_147;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_147) : "f"(S2[15].y), "f"(S2[15].x));
                    h_words[1] = _cvt_rn_f16x2_f32_147;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_148;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_148) : "f"(S2[16].y), "f"(S2[16].x));
                    h_words[0] = _cvt_rn_f16x2_f32_148;
                    uint32_t _cvt_rn_f16x2_f32_149;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_149) : "f"(S2[17].y), "f"(S2[17].x));
                    h_words[1] = _cvt_rn_f16x2_f32_149;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_150;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_150) : "f"(S2[18].y), "f"(S2[18].x));
                    h_words[0] = _cvt_rn_f16x2_f32_150;
                    uint32_t _cvt_rn_f16x2_f32_151;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_151) : "f"(S2[19].y), "f"(S2[19].x));
                    h_words[1] = _cvt_rn_f16x2_f32_151;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_152;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_152) : "f"(S2[20].y), "f"(S2[20].x));
                    h_words[0] = _cvt_rn_f16x2_f32_152;
                    uint32_t _cvt_rn_f16x2_f32_153;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_153) : "f"(S2[21].y), "f"(S2[21].x));
                    h_words[1] = _cvt_rn_f16x2_f32_153;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_154;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_154) : "f"(S2[22].y), "f"(S2[22].x));
                    h_words[0] = _cvt_rn_f16x2_f32_154;
                    uint32_t _cvt_rn_f16x2_f32_155;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_155) : "f"(S2[23].y), "f"(S2[23].x));
                    h_words[1] = _cvt_rn_f16x2_f32_155;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_156;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_156) : "f"(S2[24].y), "f"(S2[24].x));
                    h_words[0] = _cvt_rn_f16x2_f32_156;
                    uint32_t _cvt_rn_f16x2_f32_157;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_157) : "f"(S2[25].y), "f"(S2[25].x));
                    h_words[1] = _cvt_rn_f16x2_f32_157;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_158;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_158) : "f"(S2[26].y), "f"(S2[26].x));
                    h_words[0] = _cvt_rn_f16x2_f32_158;
                    uint32_t _cvt_rn_f16x2_f32_159;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_159) : "f"(S2[27].y), "f"(S2[27].x));
                    h_words[1] = _cvt_rn_f16x2_f32_159;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_160;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_160) : "f"(S2[28].y), "f"(S2[28].x));
                    h_words[0] = _cvt_rn_f16x2_f32_160;
                    uint32_t _cvt_rn_f16x2_f32_161;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_161) : "f"(S2[29].y), "f"(S2[29].x));
                    h_words[1] = _cvt_rn_f16x2_f32_161;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_162;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_162) : "f"(S2[30].y), "f"(S2[30].x));
                    h_words[0] = _cvt_rn_f16x2_f32_162;
                    uint32_t _cvt_rn_f16x2_f32_163;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_163) : "f"(S2[31].y), "f"(S2[31].x));
                    h_words[1] = _cvt_rn_f16x2_f32_163;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                }
            }
        }
        if (use_ssm_state_indices == 0) {
            #pragma unroll 1
            for (int t_1 = 0; t_1 < 7; t_1++) {
                float alpha_1 = sAlpha[t_1];
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
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                    : "r"(sV_addr + (unsigned int)(m_3 * 256) + group_byte));
                float beta_3 = sBeta[m_3];
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
                float _shfl_xor_sync_164 = __shfl_xor_sync(0xFFFFFFFF, dot_15, 1);
                dot_15 = dot_15 + _shfl_xor_sync_164;
                float _shfl_xor_sync_165 = __shfl_xor_sync(0xFFFFFFFF, dot_15, 2);
                dot_15 = dot_15 + _shfl_xor_sync_165;
                float _shfl_xor_sync_166 = __shfl_xor_sync(0xFFFFFFFF, dot_15, 4);
                dot_15 = dot_15 + _shfl_xor_sync_166;
                float vn_12 = (v_regs[0] - dot_15) * beta_3;
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
                float _shfl_xor_sync_167 = __shfl_xor_sync(0xFFFFFFFF, dot_15_1, 1);
                dot_15_1 = dot_15_1 + _shfl_xor_sync_167;
                float _shfl_xor_sync_168 = __shfl_xor_sync(0xFFFFFFFF, dot_15_1, 2);
                dot_15_1 = dot_15_1 + _shfl_xor_sync_168;
                float _shfl_xor_sync_169 = __shfl_xor_sync(0xFFFFFFFF, dot_15_1, 4);
                dot_15_1 = dot_15_1 + _shfl_xor_sync_169;
                float vn_13 = (v_regs[1] - dot_15_1) * beta_3;
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
                float _shfl_xor_sync_170 = __shfl_xor_sync(0xFFFFFFFF, dot_16, 1);
                dot_16 = dot_16 + _shfl_xor_sync_170;
                float _shfl_xor_sync_171 = __shfl_xor_sync(0xFFFFFFFF, dot_16, 2);
                dot_16 = dot_16 + _shfl_xor_sync_171;
                float _shfl_xor_sync_172 = __shfl_xor_sync(0xFFFFFFFF, dot_16, 4);
                dot_16 = dot_16 + _shfl_xor_sync_172;
                float vn_14 = (v_regs[2] - dot_16) * beta_3;
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
                float _shfl_xor_sync_173 = __shfl_xor_sync(0xFFFFFFFF, dot_17, 1);
                dot_17 = dot_17 + _shfl_xor_sync_173;
                float _shfl_xor_sync_174 = __shfl_xor_sync(0xFFFFFFFF, dot_17, 2);
                dot_17 = dot_17 + _shfl_xor_sync_174;
                float _shfl_xor_sync_175 = __shfl_xor_sync(0xFFFFFFFF, dot_17, 4);
                dot_17 = dot_17 + _shfl_xor_sync_175;
                float vn_15 = (v_regs[3] - dot_17) * beta_3;
                float2 vn2_15 = make_float2(vn_15, vn_15);
                S2[24] = fma_f32x2_rn_ftz(k2[0], vn2_15, S2[24]);
                S2[25] = fma_f32x2_rn_ftz(k2[1], vn2_15, S2[25]);
                S2[26] = fma_f32x2_rn_ftz(k2[2], vn2_15, S2[26]);
                S2[27] = fma_f32x2_rn_ftz(k2[3], vn2_15, S2[27]);
                S2[28] = fma_f32x2_rn_ftz(k2[4], vn2_15, S2[28]);
                S2[29] = fma_f32x2_rn_ftz(k2[5], vn2_15, S2[29]);
                S2[30] = fma_f32x2_rn_ftz(k2[6], vn2_15, S2[30]);
                S2[31] = fma_f32x2_rn_ftz(k2[7], vn2_15, S2[31]);
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
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                    : "r"(sV_addr + (unsigned int)(m_4 * 256) + group_byte));
                float beta_4 = sBeta[m_4];
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
                float _shfl_xor_sync_176 = __shfl_xor_sync(0xFFFFFFFF, dot_18, 1);
                dot_18 = dot_18 + _shfl_xor_sync_176;
                float _shfl_xor_sync_177 = __shfl_xor_sync(0xFFFFFFFF, dot_18, 2);
                dot_18 = dot_18 + _shfl_xor_sync_177;
                float _shfl_xor_sync_178 = __shfl_xor_sync(0xFFFFFFFF, dot_18, 4);
                dot_18 = dot_18 + _shfl_xor_sync_178;
                float vn_16 = (v_regs[0] - dot_18) * beta_4;
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
                float _shfl_xor_sync_179 = __shfl_xor_sync(0xFFFFFFFF, dot_19, 1);
                dot_19 = dot_19 + _shfl_xor_sync_179;
                float _shfl_xor_sync_180 = __shfl_xor_sync(0xFFFFFFFF, dot_19, 2);
                dot_19 = dot_19 + _shfl_xor_sync_180;
                float _shfl_xor_sync_181 = __shfl_xor_sync(0xFFFFFFFF, dot_19, 4);
                dot_19 = dot_19 + _shfl_xor_sync_181;
                float vn_17 = (v_regs[1] - dot_19) * beta_4;
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
                float _shfl_xor_sync_182 = __shfl_xor_sync(0xFFFFFFFF, dot_20, 1);
                dot_20 = dot_20 + _shfl_xor_sync_182;
                float _shfl_xor_sync_183 = __shfl_xor_sync(0xFFFFFFFF, dot_20, 2);
                dot_20 = dot_20 + _shfl_xor_sync_183;
                float _shfl_xor_sync_184 = __shfl_xor_sync(0xFFFFFFFF, dot_20, 4);
                dot_20 = dot_20 + _shfl_xor_sync_184;
                float vn_18 = (v_regs[2] - dot_20) * beta_4;
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
                float _shfl_xor_sync_185 = __shfl_xor_sync(0xFFFFFFFF, dot_21, 1);
                dot_21 = dot_21 + _shfl_xor_sync_185;
                float _shfl_xor_sync_186 = __shfl_xor_sync(0xFFFFFFFF, dot_21, 2);
                dot_21 = dot_21 + _shfl_xor_sync_186;
                float _shfl_xor_sync_187 = __shfl_xor_sync(0xFFFFFFFF, dot_21, 4);
                dot_21 = dot_21 + _shfl_xor_sync_187;
                float vn_19 = (v_regs[3] - dot_21) * beta_4;
                float2 vn2_19 = make_float2(vn_19, vn_19);
                S2[24] = fma_f32x2_rn_ftz(k2[0], vn2_19, S2[24]);
                S2[25] = fma_f32x2_rn_ftz(k2[1], vn2_19, S2[25]);
                S2[26] = fma_f32x2_rn_ftz(k2[2], vn2_19, S2[26]);
                S2[27] = fma_f32x2_rn_ftz(k2[3], vn2_19, S2[27]);
                S2[28] = fma_f32x2_rn_ftz(k2[4], vn2_19, S2[28]);
                S2[29] = fma_f32x2_rn_ftz(k2[5], vn2_19, S2[29]);
                S2[30] = fma_f32x2_rn_ftz(k2[6], vn2_19, S2[30]);
                S2[31] = fma_f32x2_rn_ftz(k2[7], vn2_19, S2[31]);
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
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                    : "r"(sV_addr + (unsigned int)(m_5 * 256) + group_byte));
                float beta_5 = sBeta[m_5];
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
                float _shfl_xor_sync_188 = __shfl_xor_sync(0xFFFFFFFF, dot_22, 1);
                dot_22 = dot_22 + _shfl_xor_sync_188;
                float _shfl_xor_sync_189 = __shfl_xor_sync(0xFFFFFFFF, dot_22, 2);
                dot_22 = dot_22 + _shfl_xor_sync_189;
                float _shfl_xor_sync_190 = __shfl_xor_sync(0xFFFFFFFF, dot_22, 4);
                dot_22 = dot_22 + _shfl_xor_sync_190;
                float vn_20 = (v_regs[0] - dot_22) * beta_5;
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
                float _shfl_xor_sync_191 = __shfl_xor_sync(0xFFFFFFFF, dot_23, 1);
                dot_23 = dot_23 + _shfl_xor_sync_191;
                float _shfl_xor_sync_192 = __shfl_xor_sync(0xFFFFFFFF, dot_23, 2);
                dot_23 = dot_23 + _shfl_xor_sync_192;
                float _shfl_xor_sync_193 = __shfl_xor_sync(0xFFFFFFFF, dot_23, 4);
                dot_23 = dot_23 + _shfl_xor_sync_193;
                float vn_21 = (v_regs[1] - dot_23) * beta_5;
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
                float _shfl_xor_sync_194 = __shfl_xor_sync(0xFFFFFFFF, dot_24, 1);
                dot_24 = dot_24 + _shfl_xor_sync_194;
                float _shfl_xor_sync_195 = __shfl_xor_sync(0xFFFFFFFF, dot_24, 2);
                dot_24 = dot_24 + _shfl_xor_sync_195;
                float _shfl_xor_sync_196 = __shfl_xor_sync(0xFFFFFFFF, dot_24, 4);
                dot_24 = dot_24 + _shfl_xor_sync_196;
                float vn_22 = (v_regs[2] - dot_24) * beta_5;
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
                float _shfl_xor_sync_197 = __shfl_xor_sync(0xFFFFFFFF, dot_25, 1);
                dot_25 = dot_25 + _shfl_xor_sync_197;
                float _shfl_xor_sync_198 = __shfl_xor_sync(0xFFFFFFFF, dot_25, 2);
                dot_25 = dot_25 + _shfl_xor_sync_198;
                float _shfl_xor_sync_199 = __shfl_xor_sync(0xFFFFFFFF, dot_25, 4);
                dot_25 = dot_25 + _shfl_xor_sync_199;
                float vn_23 = (v_regs[3] - dot_25) * beta_5;
                float2 vn2_23 = make_float2(vn_23, vn_23);
                S2[24] = fma_f32x2_rn_ftz(k2[0], vn2_23, S2[24]);
                S2[25] = fma_f32x2_rn_ftz(k2[1], vn2_23, S2[25]);
                S2[26] = fma_f32x2_rn_ftz(k2[2], vn2_23, S2[26]);
                S2[27] = fma_f32x2_rn_ftz(k2[3], vn2_23, S2[27]);
                S2[28] = fma_f32x2_rn_ftz(k2[4], vn2_23, S2[28]);
                S2[29] = fma_f32x2_rn_ftz(k2[5], vn2_23, S2[29]);
                S2[30] = fma_f32x2_rn_ftz(k2[6], vn2_23, S2[30]);
                S2[31] = fma_f32x2_rn_ftz(k2[7], vn2_23, S2[31]);
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
                float _shfl_xor_sync_200 = __shfl_xor_sync(0xFFFFFFFF, dot_26, 1);
                dot_26 = dot_26 + _shfl_xor_sync_200;
                float _shfl_xor_sync_201 = __shfl_xor_sync(0xFFFFFFFF, dot_26, 2);
                dot_26 = dot_26 + _shfl_xor_sync_201;
                float _shfl_xor_sync_202 = __shfl_xor_sync(0xFFFFFFFF, dot_26, 4);
                dot_26 = dot_26 + _shfl_xor_sync_202;
                o4[0] = dot_26;
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
                float _shfl_xor_sync_203 = __shfl_xor_sync(0xFFFFFFFF, dot_27, 1);
                dot_27 = dot_27 + _shfl_xor_sync_203;
                float _shfl_xor_sync_204 = __shfl_xor_sync(0xFFFFFFFF, dot_27, 2);
                dot_27 = dot_27 + _shfl_xor_sync_204;
                float _shfl_xor_sync_205 = __shfl_xor_sync(0xFFFFFFFF, dot_27, 4);
                dot_27 = dot_27 + _shfl_xor_sync_205;
                o4[1] = dot_27;
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
                float _shfl_xor_sync_206 = __shfl_xor_sync(0xFFFFFFFF, dot_28, 1);
                dot_28 = dot_28 + _shfl_xor_sync_206;
                float _shfl_xor_sync_207 = __shfl_xor_sync(0xFFFFFFFF, dot_28, 2);
                dot_28 = dot_28 + _shfl_xor_sync_207;
                float _shfl_xor_sync_208 = __shfl_xor_sync(0xFFFFFFFF, dot_28, 4);
                dot_28 = dot_28 + _shfl_xor_sync_208;
                o4[2] = dot_28;
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
                float _shfl_xor_sync_209 = __shfl_xor_sync(0xFFFFFFFF, dot_29, 1);
                dot_29 = dot_29 + _shfl_xor_sync_209;
                float _shfl_xor_sync_210 = __shfl_xor_sync(0xFFFFFFFF, dot_29, 2);
                dot_29 = dot_29 + _shfl_xor_sync_210;
                float _shfl_xor_sync_211 = __shfl_xor_sync(0xFFFFFFFF, dot_29, 4);
                dot_29 = dot_29 + _shfl_xor_sync_211;
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
                    uint32_t _read_special_reg_212;
                    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(_read_special_reg_212));
                    int tid_r = (int)_read_special_reg_212;
                    long long noise_at = read_slot_head + (long long)(tid_r / 8 * 4) * state_stride_p2 + (long long)((tid_r - tid_r / 8 * 8) * 4);
                    if (philox_rounds != 0) {
                        unsigned int key0_1 = seed_lo;
                        unsigned int key1_1 = seed_hi;
                        long long o_8 = noise_at;
                        unsigned long long p0_8 = (unsigned long long)(unsigned int)o_8 * 3528531795ull;
                        h_noise[0] = (unsigned int)(o_8 >> 32ll) ^ key0_1;
                        h_noise[2] = (unsigned int)(p0_8 >> 32ull) ^ key1_1;
                        h_noise[3] = (unsigned int)p0_8;
                        long long o_9 = noise_at + 32ll;
                        unsigned long long p0_9 = (unsigned long long)(unsigned int)o_9 * 3528531795ull;
                        h_noise[4] = (unsigned int)(o_9 >> 32ll) ^ key0_1;
                        h_noise[6] = (unsigned int)(p0_9 >> 32ull) ^ key1_1;
                        h_noise[7] = (unsigned int)p0_9;
                        long long o_10 = noise_at + state_stride_p2;
                        unsigned long long p0_10 = (unsigned long long)(unsigned int)o_10 * 3528531795ull;
                        h_noise[8] = (unsigned int)(o_10 >> 32ll) ^ key0_1;
                        h_noise[10] = (unsigned int)(p0_10 >> 32ull) ^ key1_1;
                        h_noise[11] = (unsigned int)p0_10;
                        long long o_11 = noise_at + state_stride_p2 + 32ll;
                        unsigned long long p0_11 = (unsigned long long)(unsigned int)o_11 * 3528531795ull;
                        h_noise[12] = (unsigned int)(o_11 >> 32ll) ^ key0_1;
                        h_noise[14] = (unsigned int)(p0_11 >> 32ull) ^ key1_1;
                        h_noise[15] = (unsigned int)p0_11;
                        long long o_12 = noise_at + 2ll * state_stride_p2;
                        unsigned long long p0_12 = (unsigned long long)(unsigned int)o_12 * 3528531795ull;
                        h_noise[16] = (unsigned int)(o_12 >> 32ll) ^ key0_1;
                        h_noise[18] = (unsigned int)(p0_12 >> 32ull) ^ key1_1;
                        h_noise[19] = (unsigned int)p0_12;
                        long long o_13 = noise_at + 2ll * state_stride_p2 + 32ll;
                        unsigned long long p0_13 = (unsigned long long)(unsigned int)o_13 * 3528531795ull;
                        h_noise[20] = (unsigned int)(o_13 >> 32ll) ^ key0_1;
                        h_noise[22] = (unsigned int)(p0_13 >> 32ull) ^ key1_1;
                        h_noise[23] = (unsigned int)p0_13;
                        long long o_14 = noise_at + 3ll * state_stride_p2;
                        unsigned long long p0_14 = (unsigned long long)(unsigned int)o_14 * 3528531795ull;
                        h_noise[24] = (unsigned int)(o_14 >> 32ll) ^ key0_1;
                        h_noise[26] = (unsigned int)(p0_14 >> 32ull) ^ key1_1;
                        h_noise[27] = (unsigned int)p0_14;
                        long long o_15 = noise_at + 3ll * state_stride_p2 + 32ll;
                        unsigned long long p0_15 = (unsigned long long)(unsigned int)o_15 * 3528531795ull;
                        h_noise[28] = (unsigned int)(o_15 >> 32ll) ^ key0_1;
                        h_noise[30] = (unsigned int)(p0_15 >> 32ull) ^ key1_1;
                        h_noise[31] = (unsigned int)p0_15;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long q0_8 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long q1_8 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(q1_8 >> 32ull) ^ key0_1;
                        h_noise[1] = (unsigned int)q1_8;
                        h_noise[2] = (unsigned int)(q0_8 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)q0_8;
                        unsigned long long q0_9 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long q1_9 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(q1_9 >> 32ull) ^ key0_1;
                        h_noise[5] = (unsigned int)q1_9;
                        h_noise[6] = (unsigned int)(q0_9 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)q0_9;
                        unsigned long long q0_10 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long q1_10 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(q1_10 >> 32ull) ^ key0_1;
                        h_noise[9] = (unsigned int)q1_10;
                        h_noise[10] = (unsigned int)(q0_10 >> 32ull) ^ h_noise[11] ^ key1_1;
                        h_noise[11] = (unsigned int)q0_10;
                        unsigned long long q0_11 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long q1_11 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(q1_11 >> 32ull) ^ key0_1;
                        h_noise[13] = (unsigned int)q1_11;
                        h_noise[14] = (unsigned int)(q0_11 >> 32ull) ^ h_noise[15] ^ key1_1;
                        h_noise[15] = (unsigned int)q0_11;
                        unsigned long long q0_12 = (unsigned long long)h_noise[16] * 3528531795ull;
                        unsigned long long q1_12 = (unsigned long long)h_noise[18] * 3449720151ull;
                        h_noise[16] = (unsigned int)(q1_12 >> 32ull) ^ key0_1;
                        h_noise[17] = (unsigned int)q1_12;
                        h_noise[18] = (unsigned int)(q0_12 >> 32ull) ^ h_noise[19] ^ key1_1;
                        h_noise[19] = (unsigned int)q0_12;
                        unsigned long long q0_13 = (unsigned long long)h_noise[20] * 3528531795ull;
                        unsigned long long q1_13 = (unsigned long long)h_noise[22] * 3449720151ull;
                        h_noise[20] = (unsigned int)(q1_13 >> 32ull) ^ key0_1;
                        h_noise[21] = (unsigned int)q1_13;
                        h_noise[22] = (unsigned int)(q0_13 >> 32ull) ^ h_noise[23] ^ key1_1;
                        h_noise[23] = (unsigned int)q0_13;
                        unsigned long long q0_14 = (unsigned long long)h_noise[24] * 3528531795ull;
                        unsigned long long q1_14 = (unsigned long long)h_noise[26] * 3449720151ull;
                        h_noise[24] = (unsigned int)(q1_14 >> 32ull) ^ key0_1;
                        h_noise[25] = (unsigned int)q1_14;
                        h_noise[26] = (unsigned int)(q0_14 >> 32ull) ^ h_noise[27] ^ key1_1;
                        h_noise[27] = (unsigned int)q0_14;
                        unsigned long long q0_15 = (unsigned long long)h_noise[28] * 3528531795ull;
                        unsigned long long q1_15 = (unsigned long long)h_noise[30] * 3449720151ull;
                        h_noise[28] = (unsigned int)(q1_15 >> 32ull) ^ key0_1;
                        h_noise[29] = (unsigned int)q1_15;
                        h_noise[30] = (unsigned int)(q0_15 >> 32ull) ^ h_noise[31] ^ key1_1;
                        h_noise[31] = (unsigned int)q0_15;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long s0_24 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_24 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_24 >> 32ull) ^ h_noise[1] ^ key0_1;
                        h_noise[1] = (unsigned int)s1_24;
                        h_noise[2] = (unsigned int)(s0_24 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)s0_24;
                        unsigned long long s0_25 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_25 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_25 >> 32ull) ^ h_noise[5] ^ key0_1;
                        h_noise[5] = (unsigned int)s1_25;
                        h_noise[6] = (unsigned int)(s0_25 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)s0_25;
                        unsigned long long s0_26 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_26 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_26 >> 32ull) ^ h_noise[9] ^ key0_1;
                        h_noise[9] = (unsigned int)s1_26;
                        h_noise[10] = (unsigned int)(s0_26 >> 32ull) ^ h_noise[11] ^ key1_1;
                        h_noise[11] = (unsigned int)s0_26;
                        unsigned long long s0_27 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_27 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_27 >> 32ull) ^ h_noise[13] ^ key0_1;
                        h_noise[13] = (unsigned int)s1_27;
                        h_noise[14] = (unsigned int)(s0_27 >> 32ull) ^ h_noise[15] ^ key1_1;
                        h_noise[15] = (unsigned int)s0_27;
                        unsigned long long s0_28 = (unsigned long long)h_noise[16] * 3528531795ull;
                        unsigned long long s1_28 = (unsigned long long)h_noise[18] * 3449720151ull;
                        h_noise[16] = (unsigned int)(s1_28 >> 32ull) ^ h_noise[17] ^ key0_1;
                        h_noise[17] = (unsigned int)s1_28;
                        h_noise[18] = (unsigned int)(s0_28 >> 32ull) ^ h_noise[19] ^ key1_1;
                        h_noise[19] = (unsigned int)s0_28;
                        unsigned long long s0_29 = (unsigned long long)h_noise[20] * 3528531795ull;
                        unsigned long long s1_29 = (unsigned long long)h_noise[22] * 3449720151ull;
                        h_noise[20] = (unsigned int)(s1_29 >> 32ull) ^ h_noise[21] ^ key0_1;
                        h_noise[21] = (unsigned int)s1_29;
                        h_noise[22] = (unsigned int)(s0_29 >> 32ull) ^ h_noise[23] ^ key1_1;
                        h_noise[23] = (unsigned int)s0_29;
                        unsigned long long s0_30 = (unsigned long long)h_noise[24] * 3528531795ull;
                        unsigned long long s1_30 = (unsigned long long)h_noise[26] * 3449720151ull;
                        h_noise[24] = (unsigned int)(s1_30 >> 32ull) ^ h_noise[25] ^ key0_1;
                        h_noise[25] = (unsigned int)s1_30;
                        h_noise[26] = (unsigned int)(s0_30 >> 32ull) ^ h_noise[27] ^ key1_1;
                        h_noise[27] = (unsigned int)s0_30;
                        unsigned long long s0_31 = (unsigned long long)h_noise[28] * 3528531795ull;
                        unsigned long long s1_31 = (unsigned long long)h_noise[30] * 3449720151ull;
                        h_noise[28] = (unsigned int)(s1_31 >> 32ull) ^ h_noise[29] ^ key0_1;
                        h_noise[29] = (unsigned int)s1_31;
                        h_noise[30] = (unsigned int)(s0_31 >> 32ull) ^ h_noise[31] ^ key1_1;
                        h_noise[31] = (unsigned int)s0_31;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long s0_32 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_32 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_32 >> 32ull) ^ h_noise[1] ^ key0_1;
                        h_noise[1] = (unsigned int)s1_32;
                        h_noise[2] = (unsigned int)(s0_32 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)s0_32;
                        unsigned long long s0_33 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_33 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_33 >> 32ull) ^ h_noise[5] ^ key0_1;
                        h_noise[5] = (unsigned int)s1_33;
                        h_noise[6] = (unsigned int)(s0_33 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)s0_33;
                        unsigned long long s0_34 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_34 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_34 >> 32ull) ^ h_noise[9] ^ key0_1;
                        h_noise[9] = (unsigned int)s1_34;
                        h_noise[10] = (unsigned int)(s0_34 >> 32ull) ^ h_noise[11] ^ key1_1;
                        h_noise[11] = (unsigned int)s0_34;
                        unsigned long long s0_35 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_35 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_35 >> 32ull) ^ h_noise[13] ^ key0_1;
                        h_noise[13] = (unsigned int)s1_35;
                        h_noise[14] = (unsigned int)(s0_35 >> 32ull) ^ h_noise[15] ^ key1_1;
                        h_noise[15] = (unsigned int)s0_35;
                        unsigned long long s0_36 = (unsigned long long)h_noise[16] * 3528531795ull;
                        unsigned long long s1_36 = (unsigned long long)h_noise[18] * 3449720151ull;
                        h_noise[16] = (unsigned int)(s1_36 >> 32ull) ^ h_noise[17] ^ key0_1;
                        h_noise[17] = (unsigned int)s1_36;
                        h_noise[18] = (unsigned int)(s0_36 >> 32ull) ^ h_noise[19] ^ key1_1;
                        h_noise[19] = (unsigned int)s0_36;
                        unsigned long long s0_37 = (unsigned long long)h_noise[20] * 3528531795ull;
                        unsigned long long s1_37 = (unsigned long long)h_noise[22] * 3449720151ull;
                        h_noise[20] = (unsigned int)(s1_37 >> 32ull) ^ h_noise[21] ^ key0_1;
                        h_noise[21] = (unsigned int)s1_37;
                        h_noise[22] = (unsigned int)(s0_37 >> 32ull) ^ h_noise[23] ^ key1_1;
                        h_noise[23] = (unsigned int)s0_37;
                        unsigned long long s0_38 = (unsigned long long)h_noise[24] * 3528531795ull;
                        unsigned long long s1_38 = (unsigned long long)h_noise[26] * 3449720151ull;
                        h_noise[24] = (unsigned int)(s1_38 >> 32ull) ^ h_noise[25] ^ key0_1;
                        h_noise[25] = (unsigned int)s1_38;
                        h_noise[26] = (unsigned int)(s0_38 >> 32ull) ^ h_noise[27] ^ key1_1;
                        h_noise[27] = (unsigned int)s0_38;
                        unsigned long long s0_39 = (unsigned long long)h_noise[28] * 3528531795ull;
                        unsigned long long s1_39 = (unsigned long long)h_noise[30] * 3449720151ull;
                        h_noise[28] = (unsigned int)(s1_39 >> 32ull) ^ h_noise[29] ^ key0_1;
                        h_noise[29] = (unsigned int)s1_39;
                        h_noise[30] = (unsigned int)(s0_39 >> 32ull) ^ h_noise[31] ^ key1_1;
                        h_noise[31] = (unsigned int)s0_39;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long s0_40 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_40 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_40 >> 32ull) ^ h_noise[1] ^ key0_1;
                        h_noise[1] = (unsigned int)s1_40;
                        h_noise[2] = (unsigned int)(s0_40 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)s0_40;
                        unsigned long long s0_41 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_41 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_41 >> 32ull) ^ h_noise[5] ^ key0_1;
                        h_noise[5] = (unsigned int)s1_41;
                        h_noise[6] = (unsigned int)(s0_41 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)s0_41;
                        unsigned long long s0_42 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_42 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_42 >> 32ull) ^ h_noise[9] ^ key0_1;
                        h_noise[9] = (unsigned int)s1_42;
                        h_noise[10] = (unsigned int)(s0_42 >> 32ull) ^ h_noise[11] ^ key1_1;
                        h_noise[11] = (unsigned int)s0_42;
                        unsigned long long s0_43 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_43 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_43 >> 32ull) ^ h_noise[13] ^ key0_1;
                        h_noise[13] = (unsigned int)s1_43;
                        h_noise[14] = (unsigned int)(s0_43 >> 32ull) ^ h_noise[15] ^ key1_1;
                        h_noise[15] = (unsigned int)s0_43;
                        unsigned long long s0_44 = (unsigned long long)h_noise[16] * 3528531795ull;
                        unsigned long long s1_44 = (unsigned long long)h_noise[18] * 3449720151ull;
                        h_noise[16] = (unsigned int)(s1_44 >> 32ull) ^ h_noise[17] ^ key0_1;
                        h_noise[17] = (unsigned int)s1_44;
                        h_noise[18] = (unsigned int)(s0_44 >> 32ull) ^ h_noise[19] ^ key1_1;
                        h_noise[19] = (unsigned int)s0_44;
                        unsigned long long s0_45 = (unsigned long long)h_noise[20] * 3528531795ull;
                        unsigned long long s1_45 = (unsigned long long)h_noise[22] * 3449720151ull;
                        h_noise[20] = (unsigned int)(s1_45 >> 32ull) ^ h_noise[21] ^ key0_1;
                        h_noise[21] = (unsigned int)s1_45;
                        h_noise[22] = (unsigned int)(s0_45 >> 32ull) ^ h_noise[23] ^ key1_1;
                        h_noise[23] = (unsigned int)s0_45;
                        unsigned long long s0_46 = (unsigned long long)h_noise[24] * 3528531795ull;
                        unsigned long long s1_46 = (unsigned long long)h_noise[26] * 3449720151ull;
                        h_noise[24] = (unsigned int)(s1_46 >> 32ull) ^ h_noise[25] ^ key0_1;
                        h_noise[25] = (unsigned int)s1_46;
                        h_noise[26] = (unsigned int)(s0_46 >> 32ull) ^ h_noise[27] ^ key1_1;
                        h_noise[27] = (unsigned int)s0_46;
                        unsigned long long s0_47 = (unsigned long long)h_noise[28] * 3528531795ull;
                        unsigned long long s1_47 = (unsigned long long)h_noise[30] * 3449720151ull;
                        h_noise[28] = (unsigned int)(s1_47 >> 32ull) ^ h_noise[29] ^ key0_1;
                        h_noise[29] = (unsigned int)s1_47;
                        h_noise[30] = (unsigned int)(s0_47 >> 32ull) ^ h_noise[31] ^ key1_1;
                        h_noise[31] = (unsigned int)s0_47;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        #pragma unroll 1
                        for (int _round_1 = 5; _round_1 < philox_rounds; _round_1++) {
                            unsigned long long t0_8 = (unsigned long long)h_noise[0] * 3528531795ull;
                            unsigned long long t1_8 = (unsigned long long)h_noise[2] * 3449720151ull;
                            h_noise[0] = (unsigned int)(t1_8 >> 32ull) ^ h_noise[1] ^ key0_1;
                            h_noise[1] = (unsigned int)t1_8;
                            h_noise[2] = (unsigned int)(t0_8 >> 32ull) ^ h_noise[3] ^ key1_1;
                            h_noise[3] = (unsigned int)t0_8;
                            unsigned long long t0_9 = (unsigned long long)h_noise[4] * 3528531795ull;
                            unsigned long long t1_9 = (unsigned long long)h_noise[6] * 3449720151ull;
                            h_noise[4] = (unsigned int)(t1_9 >> 32ull) ^ h_noise[5] ^ key0_1;
                            h_noise[5] = (unsigned int)t1_9;
                            h_noise[6] = (unsigned int)(t0_9 >> 32ull) ^ h_noise[7] ^ key1_1;
                            h_noise[7] = (unsigned int)t0_9;
                            unsigned long long t0_10 = (unsigned long long)h_noise[8] * 3528531795ull;
                            unsigned long long t1_10 = (unsigned long long)h_noise[10] * 3449720151ull;
                            h_noise[8] = (unsigned int)(t1_10 >> 32ull) ^ h_noise[9] ^ key0_1;
                            h_noise[9] = (unsigned int)t1_10;
                            h_noise[10] = (unsigned int)(t0_10 >> 32ull) ^ h_noise[11] ^ key1_1;
                            h_noise[11] = (unsigned int)t0_10;
                            unsigned long long t0_11 = (unsigned long long)h_noise[12] * 3528531795ull;
                            unsigned long long t1_11 = (unsigned long long)h_noise[14] * 3449720151ull;
                            h_noise[12] = (unsigned int)(t1_11 >> 32ull) ^ h_noise[13] ^ key0_1;
                            h_noise[13] = (unsigned int)t1_11;
                            h_noise[14] = (unsigned int)(t0_11 >> 32ull) ^ h_noise[15] ^ key1_1;
                            h_noise[15] = (unsigned int)t0_11;
                            unsigned long long t0_12 = (unsigned long long)h_noise[16] * 3528531795ull;
                            unsigned long long t1_12 = (unsigned long long)h_noise[18] * 3449720151ull;
                            h_noise[16] = (unsigned int)(t1_12 >> 32ull) ^ h_noise[17] ^ key0_1;
                            h_noise[17] = (unsigned int)t1_12;
                            h_noise[18] = (unsigned int)(t0_12 >> 32ull) ^ h_noise[19] ^ key1_1;
                            h_noise[19] = (unsigned int)t0_12;
                            unsigned long long t0_13 = (unsigned long long)h_noise[20] * 3528531795ull;
                            unsigned long long t1_13 = (unsigned long long)h_noise[22] * 3449720151ull;
                            h_noise[20] = (unsigned int)(t1_13 >> 32ull) ^ h_noise[21] ^ key0_1;
                            h_noise[21] = (unsigned int)t1_13;
                            h_noise[22] = (unsigned int)(t0_13 >> 32ull) ^ h_noise[23] ^ key1_1;
                            h_noise[23] = (unsigned int)t0_13;
                            unsigned long long t0_14 = (unsigned long long)h_noise[24] * 3528531795ull;
                            unsigned long long t1_14 = (unsigned long long)h_noise[26] * 3449720151ull;
                            h_noise[24] = (unsigned int)(t1_14 >> 32ull) ^ h_noise[25] ^ key0_1;
                            h_noise[25] = (unsigned int)t1_14;
                            h_noise[26] = (unsigned int)(t0_14 >> 32ull) ^ h_noise[27] ^ key1_1;
                            h_noise[27] = (unsigned int)t0_14;
                            unsigned long long t0_15 = (unsigned long long)h_noise[28] * 3528531795ull;
                            unsigned long long t1_15 = (unsigned long long)h_noise[30] * 3449720151ull;
                            h_noise[28] = (unsigned int)(t1_15 >> 32ull) ^ h_noise[29] ^ key0_1;
                            h_noise[29] = (unsigned int)t1_15;
                            h_noise[30] = (unsigned int)(t0_15 >> 32ull) ^ h_noise[31] ^ key1_1;
                            h_noise[31] = (unsigned int)t0_15;
                            key0_1 = key0_1 + 2654435769u;
                            key1_1 = key1_1 + 3144134277u;
                        }
                        uint32_t _cvt_rs_x2_f32_213;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_213) : "f"(S2[0].y), "f"(S2[0].x), "r"(h_noise[0] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_213;
                        uint32_t _cvt_rs_x2_f32_214;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_214) : "f"(S2[1].y), "f"(S2[1].x), "r"(h_noise[1] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_214;
                        reinterpret_cast<int2*>(state + (write_rows_base))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_215;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_215) : "f"(S2[4].y), "f"(S2[4].x), "r"(h_noise[2] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_215;
                        uint32_t _cvt_rs_x2_f32_216;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_216) : "f"(S2[5].y), "f"(S2[5].x), "r"(h_noise[3] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_216;
                        reinterpret_cast<int2*>(state + (write_rows_base + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_217;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_217) : "f"(S2[2].y), "f"(S2[2].x), "r"(h_noise[4] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_217;
                        uint32_t _cvt_rs_x2_f32_218;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_218) : "f"(S2[3].y), "f"(S2[3].x), "r"(h_noise[5] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_218;
                        reinterpret_cast<int2*>(state + (write_rows_base + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_219;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_219) : "f"(S2[6].y), "f"(S2[6].x), "r"(h_noise[6] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_219;
                        uint32_t _cvt_rs_x2_f32_220;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_220) : "f"(S2[7].y), "f"(S2[7].x), "r"(h_noise[7] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_220;
                        reinterpret_cast<int2*>(state + (write_rows_base + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_221;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_221) : "f"(S2[8].y), "f"(S2[8].x), "r"(h_noise[8] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_221;
                        uint32_t _cvt_rs_x2_f32_222;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_222) : "f"(S2[9].y), "f"(S2[9].x), "r"(h_noise[9] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_222;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_223;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_223) : "f"(S2[12].y), "f"(S2[12].x), "r"(h_noise[10] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_223;
                        uint32_t _cvt_rs_x2_f32_224;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_224) : "f"(S2[13].y), "f"(S2[13].x), "r"(h_noise[11] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_224;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_225;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_225) : "f"(S2[10].y), "f"(S2[10].x), "r"(h_noise[12] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_225;
                        uint32_t _cvt_rs_x2_f32_226;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_226) : "f"(S2[11].y), "f"(S2[11].x), "r"(h_noise[13] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_226;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_227;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_227) : "f"(S2[14].y), "f"(S2[14].x), "r"(h_noise[14] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_227;
                        uint32_t _cvt_rs_x2_f32_228;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_228) : "f"(S2[15].y), "f"(S2[15].x), "r"(h_noise[15] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_228;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_229;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_229) : "f"(S2[16].y), "f"(S2[16].x), "r"(h_noise[16] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_229;
                        uint32_t _cvt_rs_x2_f32_230;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_230) : "f"(S2[17].y), "f"(S2[17].x), "r"(h_noise[17] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_230;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_231;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_231) : "f"(S2[20].y), "f"(S2[20].x), "r"(h_noise[18] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_231;
                        uint32_t _cvt_rs_x2_f32_232;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_232) : "f"(S2[21].y), "f"(S2[21].x), "r"(h_noise[19] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_232;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_233;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_233) : "f"(S2[18].y), "f"(S2[18].x), "r"(h_noise[20] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_233;
                        uint32_t _cvt_rs_x2_f32_234;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_234) : "f"(S2[19].y), "f"(S2[19].x), "r"(h_noise[21] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_234;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_235;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_235) : "f"(S2[22].y), "f"(S2[22].x), "r"(h_noise[22] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_235;
                        uint32_t _cvt_rs_x2_f32_236;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_236) : "f"(S2[23].y), "f"(S2[23].x), "r"(h_noise[23] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_236;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_237;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_237) : "f"(S2[24].y), "f"(S2[24].x), "r"(h_noise[24] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_237;
                        uint32_t _cvt_rs_x2_f32_238;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_238) : "f"(S2[25].y), "f"(S2[25].x), "r"(h_noise[25] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_238;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_239;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_239) : "f"(S2[28].y), "f"(S2[28].x), "r"(h_noise[26] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_239;
                        uint32_t _cvt_rs_x2_f32_240;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_240) : "f"(S2[29].y), "f"(S2[29].x), "r"(h_noise[27] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_240;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_241;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_241) : "f"(S2[26].y), "f"(S2[26].x), "r"(h_noise[28] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_241;
                        uint32_t _cvt_rs_x2_f32_242;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_242) : "f"(S2[27].y), "f"(S2[27].x), "r"(h_noise[29] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_242;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_243;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_243) : "f"(S2[30].y), "f"(S2[30].x), "r"(h_noise[30] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_243;
                        uint32_t _cvt_rs_x2_f32_244;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_244) : "f"(S2[31].y), "f"(S2[31].x), "r"(h_noise[31] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_244;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                    }
                    if (philox_rounds == 0) {
                        uint32_t _cvt_rn_f16x2_f32_245;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_245) : "f"(S2[0].y), "f"(S2[0].x));
                        h_words[0] = _cvt_rn_f16x2_f32_245;
                        uint32_t _cvt_rn_f16x2_f32_246;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_246) : "f"(S2[1].y), "f"(S2[1].x));
                        h_words[1] = _cvt_rn_f16x2_f32_246;
                        reinterpret_cast<int2*>(state + (write_rows_base))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_247;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_247) : "f"(S2[2].y), "f"(S2[2].x));
                        h_words[0] = _cvt_rn_f16x2_f32_247;
                        uint32_t _cvt_rn_f16x2_f32_248;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_248) : "f"(S2[3].y), "f"(S2[3].x));
                        h_words[1] = _cvt_rn_f16x2_f32_248;
                        reinterpret_cast<int2*>(state + (write_rows_base + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_249;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_249) : "f"(S2[4].y), "f"(S2[4].x));
                        h_words[0] = _cvt_rn_f16x2_f32_249;
                        uint32_t _cvt_rn_f16x2_f32_250;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_250) : "f"(S2[5].y), "f"(S2[5].x));
                        h_words[1] = _cvt_rn_f16x2_f32_250;
                        reinterpret_cast<int2*>(state + (write_rows_base + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_251;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_251) : "f"(S2[6].y), "f"(S2[6].x));
                        h_words[0] = _cvt_rn_f16x2_f32_251;
                        uint32_t _cvt_rn_f16x2_f32_252;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_252) : "f"(S2[7].y), "f"(S2[7].x));
                        h_words[1] = _cvt_rn_f16x2_f32_252;
                        reinterpret_cast<int2*>(state + (write_rows_base + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_253;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_253) : "f"(S2[8].y), "f"(S2[8].x));
                        h_words[0] = _cvt_rn_f16x2_f32_253;
                        uint32_t _cvt_rn_f16x2_f32_254;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_254) : "f"(S2[9].y), "f"(S2[9].x));
                        h_words[1] = _cvt_rn_f16x2_f32_254;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_255;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_255) : "f"(S2[10].y), "f"(S2[10].x));
                        h_words[0] = _cvt_rn_f16x2_f32_255;
                        uint32_t _cvt_rn_f16x2_f32_256;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_256) : "f"(S2[11].y), "f"(S2[11].x));
                        h_words[1] = _cvt_rn_f16x2_f32_256;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_257;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_257) : "f"(S2[12].y), "f"(S2[12].x));
                        h_words[0] = _cvt_rn_f16x2_f32_257;
                        uint32_t _cvt_rn_f16x2_f32_258;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_258) : "f"(S2[13].y), "f"(S2[13].x));
                        h_words[1] = _cvt_rn_f16x2_f32_258;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_259;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_259) : "f"(S2[14].y), "f"(S2[14].x));
                        h_words[0] = _cvt_rn_f16x2_f32_259;
                        uint32_t _cvt_rn_f16x2_f32_260;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_260) : "f"(S2[15].y), "f"(S2[15].x));
                        h_words[1] = _cvt_rn_f16x2_f32_260;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_261;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_261) : "f"(S2[16].y), "f"(S2[16].x));
                        h_words[0] = _cvt_rn_f16x2_f32_261;
                        uint32_t _cvt_rn_f16x2_f32_262;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_262) : "f"(S2[17].y), "f"(S2[17].x));
                        h_words[1] = _cvt_rn_f16x2_f32_262;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_263;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_263) : "f"(S2[18].y), "f"(S2[18].x));
                        h_words[0] = _cvt_rn_f16x2_f32_263;
                        uint32_t _cvt_rn_f16x2_f32_264;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_264) : "f"(S2[19].y), "f"(S2[19].x));
                        h_words[1] = _cvt_rn_f16x2_f32_264;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_265;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_265) : "f"(S2[20].y), "f"(S2[20].x));
                        h_words[0] = _cvt_rn_f16x2_f32_265;
                        uint32_t _cvt_rn_f16x2_f32_266;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_266) : "f"(S2[21].y), "f"(S2[21].x));
                        h_words[1] = _cvt_rn_f16x2_f32_266;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_267;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_267) : "f"(S2[22].y), "f"(S2[22].x));
                        h_words[0] = _cvt_rn_f16x2_f32_267;
                        uint32_t _cvt_rn_f16x2_f32_268;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_268) : "f"(S2[23].y), "f"(S2[23].x));
                        h_words[1] = _cvt_rn_f16x2_f32_268;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_269;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_269) : "f"(S2[24].y), "f"(S2[24].x));
                        h_words[0] = _cvt_rn_f16x2_f32_269;
                        uint32_t _cvt_rn_f16x2_f32_270;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_270) : "f"(S2[25].y), "f"(S2[25].x));
                        h_words[1] = _cvt_rn_f16x2_f32_270;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_271;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_271) : "f"(S2[26].y), "f"(S2[26].x));
                        h_words[0] = _cvt_rn_f16x2_f32_271;
                        uint32_t _cvt_rn_f16x2_f32_272;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_272) : "f"(S2[27].y), "f"(S2[27].x));
                        h_words[1] = _cvt_rn_f16x2_f32_272;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_273;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_273) : "f"(S2[28].y), "f"(S2[28].x));
                        h_words[0] = _cvt_rn_f16x2_f32_273;
                        uint32_t _cvt_rn_f16x2_f32_274;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_274) : "f"(S2[29].y), "f"(S2[29].x));
                        h_words[1] = _cvt_rn_f16x2_f32_274;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_275;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_275) : "f"(S2[30].y), "f"(S2[30].x));
                        h_words[0] = _cvt_rn_f16x2_f32_275;
                        uint32_t _cvt_rn_f16x2_f32_276;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_276) : "f"(S2[31].y), "f"(S2[31].x));
                        h_words[1] = _cvt_rn_f16x2_f32_276;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                    }
                }
            }
        }
    }
}

} // extern "C"
