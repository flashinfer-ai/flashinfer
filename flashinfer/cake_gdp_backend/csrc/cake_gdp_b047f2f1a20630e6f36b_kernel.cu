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

#define LAUNCH_MIN_BLOCKS 4

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

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdp_b047f2f1a20630e6f36b(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __half* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, int* __restrict__ ssm_state_indices, long long* __restrict__ rand_seed, CakeFastDivmod heads, int num_tokens, float scale, int disable_state_update, int use_ssm_state_indices, int philox_rounds, long long state_size_p0, long long state_stride_p1, long long state_stride_p2, long long state_stride_p0, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long ssm_state_indices_stride_p0, long long ssm_state_indices_stride_p1, long long out_stride_p0, long long out_stride_p1, long long out_stride_p2)
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
    unsigned int* sNoise = reinterpret_cast<unsigned int*>(smem_raw + 45376);
    const int sNoise_addr = smem + 45376;

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
    int steps = 48;
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
        unsigned int h_words[2];
#if __CUDA_ARCH__ == 900
        unsigned int h_noise[16];
#else
        unsigned int h_noise[32];
#endif
        {
            uint2 _vld_0;
            _vld_0 = *reinterpret_cast<const uint2*>(state + read_rows_base);
            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
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
                    : "r"(_vpairs_0[_pair]));
            }
        }
        {
            uint2 _vld_1;
            _vld_1 = *reinterpret_cast<const uint2*>(state + read_rows_base + 32ll);
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
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
                    : "r"(_vpairs_1[_pair]));
            }
        }
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(state + read_rows_base + 64ll);
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
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
                    : "r"(_vpairs_2[_pair]));
            }
        }
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(state + read_rows_base + 96ll);
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
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
                    : "r"(_vpairs_3[_pair]));
            }
        }
        {
            uint2 _vld_4;
            _vld_4 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[16 + _pair * 2]))
                    : "r"(_vpairs_4[_pair]));
            }
        }
        {
            uint2 _vld_5;
            _vld_5 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 32ll);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[20 + _pair * 2]))
                    : "r"(_vpairs_5[_pair]));
            }
        }
        {
            uint2 _vld_6;
            _vld_6 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 64ll);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[24 + _pair * 2]))
                    : "r"(_vpairs_6[_pair]));
            }
        }
        {
            uint2 _vld_7;
            _vld_7 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 96ll);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[28 + _pair * 2]))
                    : "r"(_vpairs_7[_pair]));
            }
        }
        {
            uint2 _vld_8;
            _vld_8 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[32 + _pair * 2]))
                    : "r"(_vpairs_8[_pair]));
            }
        }
        {
            uint2 _vld_9;
            _vld_9 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2 + 32ll);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[36 + _pair * 2]))
                    : "r"(_vpairs_9[_pair]));
            }
        }
        {
            uint2 _vld_10;
            _vld_10 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2 + 64ll);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[40 + _pair * 2]))
                    : "r"(_vpairs_10[_pair]));
            }
        }
        {
            uint2 _vld_11;
            _vld_11 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2 + 96ll);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[44 + _pair * 2]))
                    : "r"(_vpairs_11[_pair]));
            }
        }
        {
            uint2 _vld_12;
            _vld_12 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2);
            uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12);
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
                    : "r"(_vpairs_12[_pair]));
            }
        }
        {
            uint2 _vld_13;
            _vld_13 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2 + 32ll);
            uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13);
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
                    : "r"(_vpairs_13[_pair]));
            }
        }
        {
            uint2 _vld_14;
            _vld_14 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2 + 64ll);
            uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14);
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
                    : "r"(_vpairs_14[_pair]));
            }
        }
        {
            uint2 _vld_15;
            _vld_15 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2 + 96ll);
            uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15);
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
                    : "r"(_vpairs_15[_pair]));
            }
        }
        unsigned int noise_lane = sNoise_addr + (unsigned int)(tid_0 * 16);
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
        float bk_0[16];
        float bv_0[8];
        float bb_0[4];
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
        int bm_5 = bm0_5;
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
        int bm0_6 = 24 + warp_local;
        int bm_6 = bm0_6;
        long long bki_6 = (long long)n * k_stride_p0 + (long long)bm_6 * k_stride_p1 + (long long)h * k_stride_p2;
        {
#if __CUDA_ARCH__ == 900
            uint2 _vld_42;
            _vld_42 = *reinterpret_cast<const uint2*>(k + bki_6 + (long long)k_col0);
            uint32_t* _vpairs_42 = reinterpret_cast<uint32_t*>(&_vld_42);
#else
            uint2 _vld_34;
            _vld_34 = *reinterpret_cast<const uint2*>(k + bki_6 + (long long)k_col0);
            uint32_t* _vpairs_34 = reinterpret_cast<uint32_t*>(&_vld_34);
#endif
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
#if __CUDA_ARCH__ == 900
                (&bk_0[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) << 16);
                (&bk_0[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) & 0xffff0000u);
#else
                (&bk_0[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) << 16);
                (&bk_0[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_34[_pair]) & 0xffff0000u);
#endif
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
#if __CUDA_ARCH__ == 900
            uint2 _vld_43;
            _vld_43 = *reinterpret_cast<const uint2*>(k + bki_7 + (long long)k_col0);
            uint32_t* _vpairs_43 = reinterpret_cast<uint32_t*>(&_vld_43);
#else
            uint2 _vld_35;
            _vld_35 = *reinterpret_cast<const uint2*>(k + bki_7 + (long long)k_col0);
            uint32_t* _vpairs_35 = reinterpret_cast<uint32_t*>(&_vld_35);
#endif
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
#if __CUDA_ARCH__ == 900
                (&bk_0[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) << 16);
                (&bk_0[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) & 0xffff0000u);
#else
                (&bk_0[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_35[_pair]) << 16);
                (&bk_0[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_35[_pair]) & 0xffff0000u);
#endif
            }
        }
        long long bvi_7 = (long long)n * v_stride_p0 + (long long)bm_7 * v_stride_p1 + (long long)h * v_stride_p2;
        int bvc_7 = lane_local * 2;
        bv_0[6] = (float)v[bvi_7 + (long long)bvc_7];
        bv_0[7] = (float)v[bvi_7 + (long long)bvc_7 + 1ll];
        long long bbi_7 = (long long)n * b_stride_p0 + (long long)bm_7 * b_stride_p1 + (long long)h * b_stride_p2;
        bb_0[3] = (float)b[bbi_7];
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
        int fm_6 = 24 + warp_local;
#if __CUDA_ARCH__ == 900
        float x0_x_5 = bk_0[8];
        float x0_y_5 = bk_0[9];
        float x1_x_5 = bk_0[10];
        float x1_y_5 = bk_0[11];
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
        float2 x0_5 = make_float2(bk_0[8], bk_0[9]);
        float2 x1_5 = make_float2(bk_0[10], bk_0[11]);
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
        x0_x_5 = mul_f32_rn_ftz(x0_x_5, norm_5);
        x0_y_5 = mul_f32_rn_ftz(x0_y_5, norm_5);
        x1_x_5 = mul_f32_rn_ftz(x1_x_5, norm_5);
        x1_y_5 = mul_f32_rn_ftz(x1_y_5, norm_5);
        sK[fm_6 * 128 + k_col0] = x0_x_5;
        sK[fm_6 * 128 + k_col0 + 1] = x0_y_5;
        sK[fm_6 * 128 + k_col0 + 2] = x1_x_5;
        sK[fm_6 * 128 + k_col0 + 3] = x1_y_5;
#else
            _warp_reduce_42 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_42, offset);
        float norm_5 = rsqrtf(_warp_reduce_42 + 1e-06f);
        float2 norm2_5 = make_float2(norm_5, norm_5);
        x0_5 = mul_f32x2(x0_5, norm2_5);
        x1_5 = mul_f32x2(x1_5, norm2_5);
        sK[fm_6 * 128 + k_col0] = x0_5.x;
        sK[fm_6 * 128 + k_col0 + 1] = x0_5.y;
        sK[fm_6 * 128 + k_col0 + 2] = x1_5.x;
        sK[fm_6 * 128 + k_col0 + 3] = x1_5.y;
#endif
        int v_col_5 = lane_local * 2;
        sV[fm_6 * 64 + v_col_5] = bv_0[4];
        sV[fm_6 * 64 + v_col_5 + 1] = bv_0[5];
        if (lane_local == 0) {
#if __CUDA_ARCH__ == 900
            float _exp_58 = __expf(-bb_0[2]);
            float _rcp_57 = approx_rcp(1.0f + _exp_58);
            sBeta[fm_6] = _rcp_57;
#else
            float _exp_44 = __expf(-bb_0[2]);
            float _rcp_43 = approx_rcp(1.0f + _exp_44);
            sBeta[fm_6] = _rcp_43;
#endif
        }
        int fm_7 = 28 + warp_local;
#if __CUDA_ARCH__ == 900
        float x0_x_6 = bk_0[12];
        float x0_y_6 = bk_0[13];
        float x1_x_6 = bk_0[14];
        float x1_y_6 = bk_0[15];
        float sum_x_6;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_6) : "f"(x0_x_6), "f"(x0_x_6), "f"(0.0f));
        float sum_y_6;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_6) : "f"(x0_y_6), "f"(x0_y_6), "f"(0.0f));
        float _fma_59;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(x1_x_6), "f"(x1_x_6), "f"(sum_x_6));
        sum_x_6 = _fma_59;
        float _fma_60;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(x1_y_6), "f"(x1_y_6), "f"(sum_y_6));
        sum_y_6 = _fma_60;
        sum_x_6 = sum_x_6 + sum_y_6;
        float _warp_reduce_61 = sum_x_6;
#else
        float2 x0_6 = make_float2(bk_0[12], bk_0[13]);
        float2 x1_6 = make_float2(bk_0[14], bk_0[15]);
        float2 sum_pair_6 = fma_f32x2_rn_ftz(x0_6, x0_6, make_float2(0.0f, 0.0f));
        sum_pair_6 = fma_f32x2_rn_ftz(x1_6, x1_6, sum_pair_6);
        float sum_x_6 = sum_pair_6.x + sum_pair_6.y;
        float _warp_reduce_45 = sum_x_6;
#endif
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
            _warp_reduce_61 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_61, offset);
        float norm_6 = rsqrtf(_warp_reduce_61 + 1e-06f);
        x0_x_6 = mul_f32_rn_ftz(x0_x_6, norm_6);
        x0_y_6 = mul_f32_rn_ftz(x0_y_6, norm_6);
        x1_x_6 = mul_f32_rn_ftz(x1_x_6, norm_6);
        x1_y_6 = mul_f32_rn_ftz(x1_y_6, norm_6);
        sK[fm_7 * 128 + k_col0] = x0_x_6;
        sK[fm_7 * 128 + k_col0 + 1] = x0_y_6;
        sK[fm_7 * 128 + k_col0 + 2] = x1_x_6;
        sK[fm_7 * 128 + k_col0 + 3] = x1_y_6;
#else
            _warp_reduce_45 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_45, offset);
        float norm_6 = rsqrtf(_warp_reduce_45 + 1e-06f);
        float2 norm2_6 = make_float2(norm_6, norm_6);
        x0_6 = mul_f32x2(x0_6, norm2_6);
        x1_6 = mul_f32x2(x1_6, norm2_6);
        sK[fm_7 * 128 + k_col0] = x0_6.x;
        sK[fm_7 * 128 + k_col0 + 1] = x0_6.y;
        sK[fm_7 * 128 + k_col0 + 2] = x1_6.x;
        sK[fm_7 * 128 + k_col0 + 3] = x1_6.y;
#endif
        int v_col_6 = lane_local * 2;
        sV[fm_7 * 64 + v_col_6] = bv_0[6];
        sV[fm_7 * 64 + v_col_6 + 1] = bv_0[7];
        if (lane_local == 0) {
#if __CUDA_ARCH__ == 900
            float _exp_63 = __expf(-bb_0[3]);
            float _rcp_62 = approx_rcp(1.0f + _exp_63);
            sBeta[fm_7] = _rcp_62;
#else
            float _exp_47 = __expf(-bb_0[3]);
            float _rcp_46 = approx_rcp(1.0f + _exp_47);
            sBeta[fm_7] = _rcp_46;
#endif
        }
        float bk_1[16];
        float bv_1[8];
        float bb_1[4];
        int bm0_8 = 32 + warp_local;
        int bm_8 = bm0_8;
        long long bki_8 = (long long)n * k_stride_p0 + (long long)bm_8 * k_stride_p1 + (long long)h * k_stride_p2;
        {
#if __CUDA_ARCH__ == 900
            uint2 _vld_64;
            _vld_64 = *reinterpret_cast<const uint2*>(k + bki_8 + (long long)k_col0);
            uint32_t* _vpairs_64 = reinterpret_cast<uint32_t*>(&_vld_64);
#else
            uint2 _vld_48;
            _vld_48 = *reinterpret_cast<const uint2*>(k + bki_8 + (long long)k_col0);
            uint32_t* _vpairs_48 = reinterpret_cast<uint32_t*>(&_vld_48);
#endif
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
#if __CUDA_ARCH__ == 900
                (&bk_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) << 16);
                (&bk_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) & 0xffff0000u);
#else
                (&bk_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_48[_pair]) << 16);
                (&bk_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_48[_pair]) & 0xffff0000u);
#endif
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
#if __CUDA_ARCH__ == 900
            uint2 _vld_65;
            _vld_65 = *reinterpret_cast<const uint2*>(k + bki_9 + (long long)k_col0);
            uint32_t* _vpairs_65 = reinterpret_cast<uint32_t*>(&_vld_65);
#else
            uint2 _vld_49;
            _vld_49 = *reinterpret_cast<const uint2*>(k + bki_9 + (long long)k_col0);
            uint32_t* _vpairs_49 = reinterpret_cast<uint32_t*>(&_vld_49);
#endif
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
#if __CUDA_ARCH__ == 900
                (&bk_1[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_65[_pair]) << 16);
                (&bk_1[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_65[_pair]) & 0xffff0000u);
#else
                (&bk_1[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_49[_pair]) << 16);
                (&bk_1[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_49[_pair]) & 0xffff0000u);
#endif
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
#if __CUDA_ARCH__ == 900
            uint2 _vld_66;
            _vld_66 = *reinterpret_cast<const uint2*>(k + bki_10 + (long long)k_col0);
            uint32_t* _vpairs_66 = reinterpret_cast<uint32_t*>(&_vld_66);
#else
            uint2 _vld_50;
            _vld_50 = *reinterpret_cast<const uint2*>(k + bki_10 + (long long)k_col0);
            uint32_t* _vpairs_50 = reinterpret_cast<uint32_t*>(&_vld_50);
#endif
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
#if __CUDA_ARCH__ == 900
                (&bk_1[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_66[_pair]) << 16);
                (&bk_1[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_66[_pair]) & 0xffff0000u);
#else
                (&bk_1[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_50[_pair]) << 16);
                (&bk_1[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_50[_pair]) & 0xffff0000u);
#endif
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
#if __CUDA_ARCH__ == 900
            uint2 _vld_67;
            _vld_67 = *reinterpret_cast<const uint2*>(k + bki_11 + (long long)k_col0);
            uint32_t* _vpairs_67 = reinterpret_cast<uint32_t*>(&_vld_67);
#else
            uint2 _vld_51;
            _vld_51 = *reinterpret_cast<const uint2*>(k + bki_11 + (long long)k_col0);
            uint32_t* _vpairs_51 = reinterpret_cast<uint32_t*>(&_vld_51);
#endif
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
#if __CUDA_ARCH__ == 900
                (&bk_1[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_67[_pair]) << 16);
                (&bk_1[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_67[_pair]) & 0xffff0000u);
#else
                (&bk_1[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_51[_pair]) << 16);
                (&bk_1[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_51[_pair]) & 0xffff0000u);
#endif
            }
        }
        long long bvi_11 = (long long)n * v_stride_p0 + (long long)bm_11 * v_stride_p1 + (long long)h * v_stride_p2;
        int bvc_11 = lane_local * 2;
        bv_1[6] = (float)v[bvi_11 + (long long)bvc_11];
        bv_1[7] = (float)v[bvi_11 + (long long)bvc_11 + 1ll];
        long long bbi_11 = (long long)n * b_stride_p0 + (long long)bm_11 * b_stride_p1 + (long long)h * b_stride_p2;
        bb_1[3] = (float)b[bbi_11];
        int fm_8 = 32 + warp_local;
#if __CUDA_ARCH__ == 900
        float x0_x_7 = bk_1[0];
        float x0_y_7 = bk_1[1];
        float x1_x_7 = bk_1[2];
        float x1_y_7 = bk_1[3];
        float sum_x_7;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_7) : "f"(x0_x_7), "f"(x0_x_7), "f"(0.0f));
        float sum_y_7;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_7) : "f"(x0_y_7), "f"(x0_y_7), "f"(0.0f));
        float _fma_68;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(x1_x_7), "f"(x1_x_7), "f"(sum_x_7));
        sum_x_7 = _fma_68;
        float _fma_69;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(x1_y_7), "f"(x1_y_7), "f"(sum_y_7));
        sum_y_7 = _fma_69;
        sum_x_7 = sum_x_7 + sum_y_7;
        float _warp_reduce_70 = sum_x_7;
#else
        float2 x0_7 = make_float2(bk_1[0], bk_1[1]);
        float2 x1_7 = make_float2(bk_1[2], bk_1[3]);
        float2 sum_pair_7 = fma_f32x2_rn_ftz(x0_7, x0_7, make_float2(0.0f, 0.0f));
        sum_pair_7 = fma_f32x2_rn_ftz(x1_7, x1_7, sum_pair_7);
        float sum_x_7 = sum_pair_7.x + sum_pair_7.y;
        float _warp_reduce_52 = sum_x_7;
#endif
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
            _warp_reduce_70 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_70, offset);
        float norm_7 = rsqrtf(_warp_reduce_70 + 1e-06f);
        x0_x_7 = mul_f32_rn_ftz(x0_x_7, norm_7);
        x0_y_7 = mul_f32_rn_ftz(x0_y_7, norm_7);
        x1_x_7 = mul_f32_rn_ftz(x1_x_7, norm_7);
        x1_y_7 = mul_f32_rn_ftz(x1_y_7, norm_7);
        sK[fm_8 * 128 + k_col0] = x0_x_7;
        sK[fm_8 * 128 + k_col0 + 1] = x0_y_7;
        sK[fm_8 * 128 + k_col0 + 2] = x1_x_7;
        sK[fm_8 * 128 + k_col0 + 3] = x1_y_7;
#else
            _warp_reduce_52 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_52, offset);
        float norm_7 = rsqrtf(_warp_reduce_52 + 1e-06f);
        float2 norm2_7 = make_float2(norm_7, norm_7);
        x0_7 = mul_f32x2(x0_7, norm2_7);
        x1_7 = mul_f32x2(x1_7, norm2_7);
        sK[fm_8 * 128 + k_col0] = x0_7.x;
        sK[fm_8 * 128 + k_col0 + 1] = x0_7.y;
        sK[fm_8 * 128 + k_col0 + 2] = x1_7.x;
        sK[fm_8 * 128 + k_col0 + 3] = x1_7.y;
#endif
        int v_col_7 = lane_local * 2;
        sV[fm_8 * 64 + v_col_7] = bv_1[0];
        sV[fm_8 * 64 + v_col_7 + 1] = bv_1[1];
        if (lane_local == 0) {
#if __CUDA_ARCH__ == 900
            float _exp_72 = __expf(-bb_1[0]);
            float _rcp_71 = approx_rcp(1.0f + _exp_72);
            sBeta[fm_8] = _rcp_71;
#else
            float _exp_54 = __expf(-bb_1[0]);
            float _rcp_53 = approx_rcp(1.0f + _exp_54);
            sBeta[fm_8] = _rcp_53;
#endif
        }
        int fm_9 = 36 + warp_local;
#if __CUDA_ARCH__ == 900
        float x0_x_8 = bk_1[4];
        float x0_y_8 = bk_1[5];
        float x1_x_8 = bk_1[6];
        float x1_y_8 = bk_1[7];
        float sum_x_8;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_8) : "f"(x0_x_8), "f"(x0_x_8), "f"(0.0f));
        float sum_y_8;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_8) : "f"(x0_y_8), "f"(x0_y_8), "f"(0.0f));
        float _fma_73;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(x1_x_8), "f"(x1_x_8), "f"(sum_x_8));
        sum_x_8 = _fma_73;
        float _fma_74;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(x1_y_8), "f"(x1_y_8), "f"(sum_y_8));
        sum_y_8 = _fma_74;
        sum_x_8 = sum_x_8 + sum_y_8;
        float _warp_reduce_75 = sum_x_8;
#else
        float2 x0_8 = make_float2(bk_1[4], bk_1[5]);
        float2 x1_8 = make_float2(bk_1[6], bk_1[7]);
        float2 sum_pair_8 = fma_f32x2_rn_ftz(x0_8, x0_8, make_float2(0.0f, 0.0f));
        sum_pair_8 = fma_f32x2_rn_ftz(x1_8, x1_8, sum_pair_8);
        float sum_x_8 = sum_pair_8.x + sum_pair_8.y;
        float _warp_reduce_55 = sum_x_8;
#endif
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
            _warp_reduce_75 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_75, offset);
        float norm_8 = rsqrtf(_warp_reduce_75 + 1e-06f);
        x0_x_8 = mul_f32_rn_ftz(x0_x_8, norm_8);
        x0_y_8 = mul_f32_rn_ftz(x0_y_8, norm_8);
        x1_x_8 = mul_f32_rn_ftz(x1_x_8, norm_8);
        x1_y_8 = mul_f32_rn_ftz(x1_y_8, norm_8);
        sK[fm_9 * 128 + k_col0] = x0_x_8;
        sK[fm_9 * 128 + k_col0 + 1] = x0_y_8;
        sK[fm_9 * 128 + k_col0 + 2] = x1_x_8;
        sK[fm_9 * 128 + k_col0 + 3] = x1_y_8;
#else
            _warp_reduce_55 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_55, offset);
        float norm_8 = rsqrtf(_warp_reduce_55 + 1e-06f);
        float2 norm2_8 = make_float2(norm_8, norm_8);
        x0_8 = mul_f32x2(x0_8, norm2_8);
        x1_8 = mul_f32x2(x1_8, norm2_8);
        sK[fm_9 * 128 + k_col0] = x0_8.x;
        sK[fm_9 * 128 + k_col0 + 1] = x0_8.y;
        sK[fm_9 * 128 + k_col0 + 2] = x1_8.x;
        sK[fm_9 * 128 + k_col0 + 3] = x1_8.y;
#endif
        int v_col_8 = lane_local * 2;
        sV[fm_9 * 64 + v_col_8] = bv_1[2];
        sV[fm_9 * 64 + v_col_8 + 1] = bv_1[3];
        if (lane_local == 0) {
#if __CUDA_ARCH__ == 900
            float _exp_77 = __expf(-bb_1[1]);
            float _rcp_76 = approx_rcp(1.0f + _exp_77);
            sBeta[fm_9] = _rcp_76;
#else
            float _exp_57 = __expf(-bb_1[1]);
            float _rcp_56 = approx_rcp(1.0f + _exp_57);
            sBeta[fm_9] = _rcp_56;
#endif
        }
        int fm_10 = 40 + warp_local;
#if __CUDA_ARCH__ == 900
        float x0_x_9 = bk_1[8];
        float x0_y_9 = bk_1[9];
        float x1_x_9 = bk_1[10];
        float x1_y_9 = bk_1[11];
        float sum_x_9;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_9) : "f"(x0_x_9), "f"(x0_x_9), "f"(0.0f));
        float sum_y_9;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_9) : "f"(x0_y_9), "f"(x0_y_9), "f"(0.0f));
        float _fma_78;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_78) : "f"(x1_x_9), "f"(x1_x_9), "f"(sum_x_9));
        sum_x_9 = _fma_78;
        float _fma_79;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(x1_y_9), "f"(x1_y_9), "f"(sum_y_9));
        sum_y_9 = _fma_79;
        sum_x_9 = sum_x_9 + sum_y_9;
        float _warp_reduce_80 = sum_x_9;
#else
        float2 x0_9 = make_float2(bk_1[8], bk_1[9]);
        float2 x1_9 = make_float2(bk_1[10], bk_1[11]);
        float2 sum_pair_9 = fma_f32x2_rn_ftz(x0_9, x0_9, make_float2(0.0f, 0.0f));
        sum_pair_9 = fma_f32x2_rn_ftz(x1_9, x1_9, sum_pair_9);
        float sum_x_9 = sum_pair_9.x + sum_pair_9.y;
        float _warp_reduce_58 = sum_x_9;
#endif
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
            _warp_reduce_80 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_80, offset);
        float norm_9 = rsqrtf(_warp_reduce_80 + 1e-06f);
        x0_x_9 = mul_f32_rn_ftz(x0_x_9, norm_9);
        x0_y_9 = mul_f32_rn_ftz(x0_y_9, norm_9);
        x1_x_9 = mul_f32_rn_ftz(x1_x_9, norm_9);
        x1_y_9 = mul_f32_rn_ftz(x1_y_9, norm_9);
        sK[fm_10 * 128 + k_col0] = x0_x_9;
        sK[fm_10 * 128 + k_col0 + 1] = x0_y_9;
        sK[fm_10 * 128 + k_col0 + 2] = x1_x_9;
        sK[fm_10 * 128 + k_col0 + 3] = x1_y_9;
#else
            _warp_reduce_58 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_58, offset);
        float norm_9 = rsqrtf(_warp_reduce_58 + 1e-06f);
        float2 norm2_9 = make_float2(norm_9, norm_9);
        x0_9 = mul_f32x2(x0_9, norm2_9);
        x1_9 = mul_f32x2(x1_9, norm2_9);
        sK[fm_10 * 128 + k_col0] = x0_9.x;
        sK[fm_10 * 128 + k_col0 + 1] = x0_9.y;
        sK[fm_10 * 128 + k_col0 + 2] = x1_9.x;
        sK[fm_10 * 128 + k_col0 + 3] = x1_9.y;
#endif
        int v_col_9 = lane_local * 2;
        sV[fm_10 * 64 + v_col_9] = bv_1[4];
        sV[fm_10 * 64 + v_col_9 + 1] = bv_1[5];
        if (lane_local == 0) {
#if __CUDA_ARCH__ == 900
            float _exp_82 = __expf(-bb_1[2]);
            float _rcp_81 = approx_rcp(1.0f + _exp_82);
            sBeta[fm_10] = _rcp_81;
#else
            float _exp_60 = __expf(-bb_1[2]);
            float _rcp_59 = approx_rcp(1.0f + _exp_60);
            sBeta[fm_10] = _rcp_59;
#endif
        }
        int fm_11 = 44 + warp_local;
#if __CUDA_ARCH__ == 900
        float x0_x_10 = bk_1[12];
        float x0_y_10 = bk_1[13];
        float x1_x_10 = bk_1[14];
        float x1_y_10 = bk_1[15];
        float sum_x_10;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_10) : "f"(x0_x_10), "f"(x0_x_10), "f"(0.0f));
        float sum_y_10;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_10) : "f"(x0_y_10), "f"(x0_y_10), "f"(0.0f));
        float _fma_83;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(x1_x_10), "f"(x1_x_10), "f"(sum_x_10));
        sum_x_10 = _fma_83;
        float _fma_84;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(x1_y_10), "f"(x1_y_10), "f"(sum_y_10));
        sum_y_10 = _fma_84;
        sum_x_10 = sum_x_10 + sum_y_10;
        float _warp_reduce_85 = sum_x_10;
#else
        float2 x0_10 = make_float2(bk_1[12], bk_1[13]);
        float2 x1_10 = make_float2(bk_1[14], bk_1[15]);
        float2 sum_pair_10 = fma_f32x2_rn_ftz(x0_10, x0_10, make_float2(0.0f, 0.0f));
        sum_pair_10 = fma_f32x2_rn_ftz(x1_10, x1_10, sum_pair_10);
        float sum_x_10 = sum_pair_10.x + sum_pair_10.y;
        float _warp_reduce_61 = sum_x_10;
#endif
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
            _warp_reduce_85 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_85, offset);
        float norm_10 = rsqrtf(_warp_reduce_85 + 1e-06f);
        x0_x_10 = mul_f32_rn_ftz(x0_x_10, norm_10);
        x0_y_10 = mul_f32_rn_ftz(x0_y_10, norm_10);
        x1_x_10 = mul_f32_rn_ftz(x1_x_10, norm_10);
        x1_y_10 = mul_f32_rn_ftz(x1_y_10, norm_10);
        sK[fm_11 * 128 + k_col0] = x0_x_10;
        sK[fm_11 * 128 + k_col0 + 1] = x0_y_10;
        sK[fm_11 * 128 + k_col0 + 2] = x1_x_10;
        sK[fm_11 * 128 + k_col0 + 3] = x1_y_10;
#else
            _warp_reduce_61 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_61, offset);
        float norm_10 = rsqrtf(_warp_reduce_61 + 1e-06f);
        float2 norm2_10 = make_float2(norm_10, norm_10);
        x0_10 = mul_f32x2(x0_10, norm2_10);
        x1_10 = mul_f32x2(x1_10, norm2_10);
        sK[fm_11 * 128 + k_col0] = x0_10.x;
        sK[fm_11 * 128 + k_col0 + 1] = x0_10.y;
        sK[fm_11 * 128 + k_col0 + 2] = x1_10.x;
        sK[fm_11 * 128 + k_col0 + 3] = x1_10.y;
#endif
        int v_col_10 = lane_local * 2;
        sV[fm_11 * 64 + v_col_10] = bv_1[6];
        sV[fm_11 * 64 + v_col_10 + 1] = bv_1[7];
        if (lane_local == 0) {
#if __CUDA_ARCH__ == 900
            float _exp_87 = __expf(-bb_1[3]);
            float _rcp_86 = approx_rcp(1.0f + _exp_87);
            sBeta[fm_11] = _rcp_86;
#else
            float _exp_63 = __expf(-bb_1[3]);
            float _rcp_62 = approx_rcp(1.0f + _exp_63);
            sBeta[fm_11] = _rcp_62;
#endif
        }
        float bq[16];
        float ba[12];
        int bt0_0 = warp_local;
        int bt_0 = bt0_0;
        long long bqi_0 = (long long)n * q_stride_p0 + (long long)bt_0 * q_stride_p1 + (long long)h * q_stride_p2;
        {
#if __CUDA_ARCH__ == 900
            uint2 _vld_88;
            _vld_88 = *reinterpret_cast<const uint2*>(q + bqi_0 + (long long)k_col0);
            uint32_t* _vpairs_88 = reinterpret_cast<uint32_t*>(&_vld_88);
#else
            uint2 _vld_64;
            _vld_64 = *reinterpret_cast<const uint2*>(q + bqi_0 + (long long)k_col0);
            uint32_t* _vpairs_64 = reinterpret_cast<uint32_t*>(&_vld_64);
#endif
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
#if __CUDA_ARCH__ == 900
                (&bq[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_88[_pair]) << 16);
                (&bq[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_88[_pair]) & 0xffff0000u);
#else
                (&bq[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) << 16);
                (&bq[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) & 0xffff0000u);
#endif
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
#if __CUDA_ARCH__ == 900
            uint2 _vld_89;
            _vld_89 = *reinterpret_cast<const uint2*>(q + bqi_1 + (long long)k_col0);
            uint32_t* _vpairs_89 = reinterpret_cast<uint32_t*>(&_vld_89);
#else
            uint2 _vld_65;
            _vld_65 = *reinterpret_cast<const uint2*>(q + bqi_1 + (long long)k_col0);
            uint32_t* _vpairs_65 = reinterpret_cast<uint32_t*>(&_vld_65);
#endif
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
#if __CUDA_ARCH__ == 900
                (&bq[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_89[_pair]) << 16);
                (&bq[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_89[_pair]) & 0xffff0000u);
#else
                (&bq[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_65[_pair]) << 16);
                (&bq[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_65[_pair]) & 0xffff0000u);
#endif
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
#if __CUDA_ARCH__ == 900
            uint2 _vld_90;
            _vld_90 = *reinterpret_cast<const uint2*>(q + bqi_2 + (long long)k_col0);
            uint32_t* _vpairs_90 = reinterpret_cast<uint32_t*>(&_vld_90);
#else
            uint2 _vld_66;
            _vld_66 = *reinterpret_cast<const uint2*>(q + bqi_2 + (long long)k_col0);
            uint32_t* _vpairs_66 = reinterpret_cast<uint32_t*>(&_vld_66);
#endif
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
#if __CUDA_ARCH__ == 900
                (&bq[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_90[_pair]) << 16);
                (&bq[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_90[_pair]) & 0xffff0000u);
#else
                (&bq[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_66[_pair]) << 16);
                (&bq[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_66[_pair]) & 0xffff0000u);
#endif
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
#if __CUDA_ARCH__ == 900
            uint2 _vld_91;
            _vld_91 = *reinterpret_cast<const uint2*>(q + bqi_3 + (long long)k_col0);
            uint32_t* _vpairs_91 = reinterpret_cast<uint32_t*>(&_vld_91);
#else
            uint2 _vld_67;
            _vld_67 = *reinterpret_cast<const uint2*>(q + bqi_3 + (long long)k_col0);
            uint32_t* _vpairs_67 = reinterpret_cast<uint32_t*>(&_vld_67);
#endif
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
#if __CUDA_ARCH__ == 900
                (&bq[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_91[_pair]) << 16);
                (&bq[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_91[_pair]) & 0xffff0000u);
#else
                (&bq[12 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_67[_pair]) << 16);
                (&bq[12 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_67[_pair]) & 0xffff0000u);
#endif
            }
        }
        long long bgi_3 = (long long)n * a_stride_p0 + (long long)bt_3 * a_stride_p1 + (long long)h * a_stride_p2;
        ba[9] = (float)a[bgi_3];
        ba[10] = dt_bias[h];
        ba[11] = A_log[h];
        int ft_0 = warp_local;
#if __CUDA_ARCH__ == 900
        float x0_x_11 = bq[0];
        float x0_y_11 = bq[1];
        float x1_x_11 = bq[2];
        float x1_y_11 = bq[3];
        float sum_x_11;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_11) : "f"(x0_x_11), "f"(x0_x_11), "f"(0.0f));
        float sum_y_11;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_11) : "f"(x0_y_11), "f"(x0_y_11), "f"(0.0f));
        float _fma_92;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(x1_x_11), "f"(x1_x_11), "f"(sum_x_11));
        sum_x_11 = _fma_92;
        float _fma_93;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(x1_y_11), "f"(x1_y_11), "f"(sum_y_11));
        sum_y_11 = _fma_93;
        sum_x_11 = sum_x_11 + sum_y_11;
        float _warp_reduce_94 = sum_x_11;
#else
        float2 x0_11 = make_float2(bq[0], bq[1]);
        float2 x1_11 = make_float2(bq[2], bq[3]);
        float2 sum_pair_11 = fma_f32x2_rn_ftz(x0_11, x0_11, make_float2(0.0f, 0.0f));
        sum_pair_11 = fma_f32x2_rn_ftz(x1_11, x1_11, sum_pair_11);
        float sum_x_11 = sum_pair_11.x + sum_pair_11.y;
        float _warp_reduce_68 = sum_x_11;
#endif
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
            _warp_reduce_94 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_94, offset);
        float norm_11 = rsqrtf(_warp_reduce_94 + 1e-06f);
#else
            _warp_reduce_68 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_68, offset);
        float norm_11 = rsqrtf(_warp_reduce_68 + 1e-06f);
#endif
        norm_11 = norm_11 * scale;
#if __CUDA_ARCH__ == 900
        x0_x_11 = mul_f32_rn_ftz(x0_x_11, norm_11);
        x0_y_11 = mul_f32_rn_ftz(x0_y_11, norm_11);
        x1_x_11 = mul_f32_rn_ftz(x1_x_11, norm_11);
        x1_y_11 = mul_f32_rn_ftz(x1_y_11, norm_11);
        sQ[ft_0 * 128 + k_col0] = x0_x_11;
        sQ[ft_0 * 128 + k_col0 + 1] = x0_y_11;
        sQ[ft_0 * 128 + k_col0 + 2] = x1_x_11;
        sQ[ft_0 * 128 + k_col0 + 3] = x1_y_11;
#else
        float2 norm2_11 = make_float2(norm_11, norm_11);
        x0_11 = mul_f32x2(x0_11, norm2_11);
        x1_11 = mul_f32x2(x1_11, norm2_11);
        sQ[ft_0 * 128 + k_col0] = x0_11.x;
        sQ[ft_0 * 128 + k_col0 + 1] = x0_11.y;
        sQ[ft_0 * 128 + k_col0 + 2] = x1_11.x;
        sQ[ft_0 * 128 + k_col0 + 3] = x1_11.y;
#endif
        if (lane_local == 0) {
            float xv = ba[0] + ba[1];
            float softplus_x = xv;
            if (xv <= 20.0f) {
#if __CUDA_ARCH__ == 900
                float _exp_96 = __expf(xv);
                float _log2_95;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_95) : "f"(1.0f + _exp_96));
                softplus_x = _log2_95 * 0.6931472f;
#else
                float _exp_70 = __expf(xv);
                float _log2_69;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_69) : "f"(1.0f + _exp_70));
                softplus_x = _log2_69 * 0.6931472f;
#endif
            }
#if __CUDA_ARCH__ == 900
            float _exp_97 = __expf(ba[2]);
            float g_log = -_exp_97 * softplus_x;
            float _exp_98 = __expf(g_log);
            sAlpha[ft_0] = _exp_98;
#else
            float _exp_71 = __expf(ba[2]);
            float g_log = -_exp_71 * softplus_x;
            float _exp_72 = __expf(g_log);
            sAlpha[ft_0] = _exp_72;
#endif
            if (use_ssm_state_indices != 0) {
                sSsm[ft_0] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)ft_0 * ssm_state_indices_stride_p1];
            }
            if (use_ssm_state_indices == 0) {
                sSsm[ft_0] = -1;
            }
        }
        int ft_1 = 4 + warp_local;
#if __CUDA_ARCH__ == 900
        float x0_x_12 = bq[4];
        float x0_y_12 = bq[5];
        float x1_x_12 = bq[6];
        float x1_y_12 = bq[7];
        float sum_x_12;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_12) : "f"(x0_x_12), "f"(x0_x_12), "f"(0.0f));
        float sum_y_12;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_12) : "f"(x0_y_12), "f"(x0_y_12), "f"(0.0f));
        float _fma_99;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(x1_x_12), "f"(x1_x_12), "f"(sum_x_12));
        sum_x_12 = _fma_99;
        float _fma_100;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(x1_y_12), "f"(x1_y_12), "f"(sum_y_12));
        sum_y_12 = _fma_100;
        sum_x_12 = sum_x_12 + sum_y_12;
        float _warp_reduce_101 = sum_x_12;
#else
        float2 x0_12 = make_float2(bq[4], bq[5]);
        float2 x1_12 = make_float2(bq[6], bq[7]);
        float2 sum_pair_12 = fma_f32x2_rn_ftz(x0_12, x0_12, make_float2(0.0f, 0.0f));
        sum_pair_12 = fma_f32x2_rn_ftz(x1_12, x1_12, sum_pair_12);
        float sum_x_12 = sum_pair_12.x + sum_pair_12.y;
        float _warp_reduce_73 = sum_x_12;
#endif
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
            _warp_reduce_101 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_101, offset);
        float norm_12 = rsqrtf(_warp_reduce_101 + 1e-06f);
#else
            _warp_reduce_73 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_73, offset);
        float norm_12 = rsqrtf(_warp_reduce_73 + 1e-06f);
#endif
        norm_12 = norm_12 * scale;
#if __CUDA_ARCH__ == 900
        x0_x_12 = mul_f32_rn_ftz(x0_x_12, norm_12);
        x0_y_12 = mul_f32_rn_ftz(x0_y_12, norm_12);
        x1_x_12 = mul_f32_rn_ftz(x1_x_12, norm_12);
        x1_y_12 = mul_f32_rn_ftz(x1_y_12, norm_12);
        sQ[ft_1 * 128 + k_col0] = x0_x_12;
        sQ[ft_1 * 128 + k_col0 + 1] = x0_y_12;
        sQ[ft_1 * 128 + k_col0 + 2] = x1_x_12;
        sQ[ft_1 * 128 + k_col0 + 3] = x1_y_12;
#else
        float2 norm2_12 = make_float2(norm_12, norm_12);
        x0_12 = mul_f32x2(x0_12, norm2_12);
        x1_12 = mul_f32x2(x1_12, norm2_12);
        sQ[ft_1 * 128 + k_col0] = x0_12.x;
        sQ[ft_1 * 128 + k_col0 + 1] = x0_12.y;
        sQ[ft_1 * 128 + k_col0 + 2] = x1_12.x;
        sQ[ft_1 * 128 + k_col0 + 3] = x1_12.y;
#endif
        if (lane_local == 0) {
            float xv_1 = ba[3] + ba[4];
            float softplus_x_1 = xv_1;
            if (xv_1 <= 20.0f) {
#if __CUDA_ARCH__ == 900
                float _exp_103 = __expf(xv_1);
                float _log2_102;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_102) : "f"(1.0f + _exp_103));
                softplus_x_1 = _log2_102 * 0.6931472f;
#else
                float _exp_75 = __expf(xv_1);
                float _log2_74;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_74) : "f"(1.0f + _exp_75));
                softplus_x_1 = _log2_74 * 0.6931472f;
#endif
            }
#if __CUDA_ARCH__ == 900
            float _exp_104 = __expf(ba[5]);
            float g_log_1 = -_exp_104 * softplus_x_1;
            float _exp_105 = __expf(g_log_1);
            sAlpha[ft_1] = _exp_105;
#else
            float _exp_76 = __expf(ba[5]);
            float g_log_1 = -_exp_76 * softplus_x_1;
            float _exp_77 = __expf(g_log_1);
            sAlpha[ft_1] = _exp_77;
#endif
            if (use_ssm_state_indices != 0) {
                sSsm[ft_1] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)ft_1 * ssm_state_indices_stride_p1];
            }
            if (use_ssm_state_indices == 0) {
                sSsm[ft_1] = -1;
            }
        }
        int ft_2 = 8 + warp_local;
#if __CUDA_ARCH__ == 900
        float x0_x_13 = bq[8];
        float x0_y_13 = bq[9];
        float x1_x_13 = bq[10];
        float x1_y_13 = bq[11];
        float sum_x_13;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_13) : "f"(x0_x_13), "f"(x0_x_13), "f"(0.0f));
        float sum_y_13;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_13) : "f"(x0_y_13), "f"(x0_y_13), "f"(0.0f));
        float _fma_106;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(x1_x_13), "f"(x1_x_13), "f"(sum_x_13));
        sum_x_13 = _fma_106;
        float _fma_107;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(x1_y_13), "f"(x1_y_13), "f"(sum_y_13));
        sum_y_13 = _fma_107;
        sum_x_13 = sum_x_13 + sum_y_13;
        float _warp_reduce_108 = sum_x_13;
#else
        float2 x0_13 = make_float2(bq[8], bq[9]);
        float2 x1_13 = make_float2(bq[10], bq[11]);
        float2 sum_pair_13 = fma_f32x2_rn_ftz(x0_13, x0_13, make_float2(0.0f, 0.0f));
        sum_pair_13 = fma_f32x2_rn_ftz(x1_13, x1_13, sum_pair_13);
        float sum_x_13 = sum_pair_13.x + sum_pair_13.y;
        float _warp_reduce_78 = sum_x_13;
#endif
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
            _warp_reduce_108 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_108, offset);
        float norm_13 = rsqrtf(_warp_reduce_108 + 1e-06f);
#else
            _warp_reduce_78 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_78, offset);
        float norm_13 = rsqrtf(_warp_reduce_78 + 1e-06f);
#endif
        norm_13 = norm_13 * scale;
#if __CUDA_ARCH__ == 900
        x0_x_13 = mul_f32_rn_ftz(x0_x_13, norm_13);
        x0_y_13 = mul_f32_rn_ftz(x0_y_13, norm_13);
        x1_x_13 = mul_f32_rn_ftz(x1_x_13, norm_13);
        x1_y_13 = mul_f32_rn_ftz(x1_y_13, norm_13);
        sQ[ft_2 * 128 + k_col0] = x0_x_13;
        sQ[ft_2 * 128 + k_col0 + 1] = x0_y_13;
        sQ[ft_2 * 128 + k_col0 + 2] = x1_x_13;
        sQ[ft_2 * 128 + k_col0 + 3] = x1_y_13;
#else
        float2 norm2_13 = make_float2(norm_13, norm_13);
        x0_13 = mul_f32x2(x0_13, norm2_13);
        x1_13 = mul_f32x2(x1_13, norm2_13);
        sQ[ft_2 * 128 + k_col0] = x0_13.x;
        sQ[ft_2 * 128 + k_col0 + 1] = x0_13.y;
        sQ[ft_2 * 128 + k_col0 + 2] = x1_13.x;
        sQ[ft_2 * 128 + k_col0 + 3] = x1_13.y;
#endif
        if (lane_local == 0) {
            float xv_2 = ba[6] + ba[7];
            float softplus_x_2 = xv_2;
            if (xv_2 <= 20.0f) {
#if __CUDA_ARCH__ == 900
                float _exp_110 = __expf(xv_2);
                float _log2_109;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_109) : "f"(1.0f + _exp_110));
                softplus_x_2 = _log2_109 * 0.6931472f;
#else
                float _exp_80 = __expf(xv_2);
                float _log2_79;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_79) : "f"(1.0f + _exp_80));
                softplus_x_2 = _log2_79 * 0.6931472f;
#endif
            }
#if __CUDA_ARCH__ == 900
            float _exp_111 = __expf(ba[8]);
            float g_log_2 = -_exp_111 * softplus_x_2;
            float _exp_112 = __expf(g_log_2);
            sAlpha[ft_2] = _exp_112;
#else
            float _exp_81 = __expf(ba[8]);
            float g_log_2 = -_exp_81 * softplus_x_2;
            float _exp_82 = __expf(g_log_2);
            sAlpha[ft_2] = _exp_82;
#endif
            if (use_ssm_state_indices != 0) {
                sSsm[ft_2] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)ft_2 * ssm_state_indices_stride_p1];
            }
            if (use_ssm_state_indices == 0) {
                sSsm[ft_2] = -1;
            }
        }
        int ft_3 = 12 + warp_local;
#if __CUDA_ARCH__ == 900
        float x0_x_14 = bq[12];
        float x0_y_14 = bq[13];
        float x1_x_14 = bq[14];
        float x1_y_14 = bq[15];
        float sum_x_14;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_14) : "f"(x0_x_14), "f"(x0_x_14), "f"(0.0f));
        float sum_y_14;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_14) : "f"(x0_y_14), "f"(x0_y_14), "f"(0.0f));
        float _fma_113;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_113) : "f"(x1_x_14), "f"(x1_x_14), "f"(sum_x_14));
        sum_x_14 = _fma_113;
        float _fma_114;
        asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_114) : "f"(x1_y_14), "f"(x1_y_14), "f"(sum_y_14));
        sum_y_14 = _fma_114;
        sum_x_14 = sum_x_14 + sum_y_14;
        float _warp_reduce_115 = sum_x_14;
#else
        float2 x0_14 = make_float2(bq[12], bq[13]);
        float2 x1_14 = make_float2(bq[14], bq[15]);
        float2 sum_pair_14 = fma_f32x2_rn_ftz(x0_14, x0_14, make_float2(0.0f, 0.0f));
        sum_pair_14 = fma_f32x2_rn_ftz(x1_14, x1_14, sum_pair_14);
        float sum_x_14 = sum_pair_14.x + sum_pair_14.y;
        float _warp_reduce_83 = sum_x_14;
#endif
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
            _warp_reduce_115 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_115, offset);
        float norm_14 = rsqrtf(_warp_reduce_115 + 1e-06f);
#else
            _warp_reduce_83 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_83, offset);
        float norm_14 = rsqrtf(_warp_reduce_83 + 1e-06f);
#endif
        norm_14 = norm_14 * scale;
#if __CUDA_ARCH__ == 900
        x0_x_14 = mul_f32_rn_ftz(x0_x_14, norm_14);
        x0_y_14 = mul_f32_rn_ftz(x0_y_14, norm_14);
        x1_x_14 = mul_f32_rn_ftz(x1_x_14, norm_14);
        x1_y_14 = mul_f32_rn_ftz(x1_y_14, norm_14);
        sQ[ft_3 * 128 + k_col0] = x0_x_14;
        sQ[ft_3 * 128 + k_col0 + 1] = x0_y_14;
        sQ[ft_3 * 128 + k_col0 + 2] = x1_x_14;
        sQ[ft_3 * 128 + k_col0 + 3] = x1_y_14;
#else
        float2 norm2_14 = make_float2(norm_14, norm_14);
        x0_14 = mul_f32x2(x0_14, norm2_14);
        x1_14 = mul_f32x2(x1_14, norm2_14);
        sQ[ft_3 * 128 + k_col0] = x0_14.x;
        sQ[ft_3 * 128 + k_col0 + 1] = x0_14.y;
        sQ[ft_3 * 128 + k_col0 + 2] = x1_14.x;
        sQ[ft_3 * 128 + k_col0 + 3] = x1_14.y;
#endif
        if (lane_local == 0) {
            float xv_3 = ba[9] + ba[10];
            float softplus_x_3 = xv_3;
            if (xv_3 <= 20.0f) {
#if __CUDA_ARCH__ == 900
                float _exp_117 = __expf(xv_3);
                float _log2_116;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_116) : "f"(1.0f + _exp_117));
                softplus_x_3 = _log2_116 * 0.6931472f;
#else
                float _exp_85 = __expf(xv_3);
                float _log2_84;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_84) : "f"(1.0f + _exp_85));
                softplus_x_3 = _log2_84 * 0.6931472f;
#endif
            }
#if __CUDA_ARCH__ == 900
            float _exp_118 = __expf(ba[11]);
            float g_log_3 = -_exp_118 * softplus_x_3;
            float _exp_119 = __expf(g_log_3);
            sAlpha[ft_3] = _exp_119;
#else
            float _exp_86 = __expf(ba[11]);
            float g_log_3 = -_exp_86 * softplus_x_3;
            float _exp_87 = __expf(g_log_3);
            sAlpha[ft_3] = _exp_87;
#endif
            if (use_ssm_state_indices != 0) {
                sSsm[ft_3] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)ft_3 * ssm_state_indices_stride_p1];
            }
            if (use_ssm_state_indices == 0) {
                sSsm[ft_3] = -1;
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
#if __CUDA_ARCH__ == 900
#else
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
#endif
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
#if __CUDA_ARCH__ == 900
#else
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
#endif
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
#if __CUDA_ARCH__ == 900
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
#else
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
#endif
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
#if __CUDA_ARCH__ == 900
#else
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
#endif
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
#if __CUDA_ARCH__ == 900
#else
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
#endif
                    key0 = key0 + 2654435769u;
                    key1 = key1 + 3144134277u;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 0), "r"(h_noise[0] & 536813567), "r"(h_noise[1] & 536813567), "r"(h_noise[2] & 536813567), "r"(h_noise[3] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 2048), "r"(h_noise[4] & 536813567), "r"(h_noise[5] & 536813567), "r"(h_noise[6] & 536813567), "r"(h_noise[7] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 4096), "r"(h_noise[8] & 536813567), "r"(h_noise[9] & 536813567), "r"(h_noise[10] & 536813567), "r"(h_noise[11] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 6144), "r"(h_noise[12] & 536813567), "r"(h_noise[13] & 536813567), "r"(h_noise[14] & 536813567), "r"(h_noise[15] & 536813567) : "memory");
#if __CUDA_ARCH__ == 900
                unsigned int key0_0 = seed_lo;
                unsigned int key1_0 = seed_hi;
                long long o_4 = read_rows_base + 2ll * state_stride_p2;
                unsigned long long p0_4 = (unsigned long long)(unsigned int)o_4 * 3528531795ull;
                h_noise[0] = (unsigned int)(o_4 >> 32ll) ^ key0_0;
                h_noise[2] = (unsigned int)(p0_4 >> 32ull) ^ key1_0;
                h_noise[3] = (unsigned int)p0_4;
                long long o_5 = read_rows_base + 2ll * state_stride_p2 + 32ll;
                unsigned long long p0_5 = (unsigned long long)(unsigned int)o_5 * 3528531795ull;
                h_noise[4] = (unsigned int)(o_5 >> 32ll) ^ key0_0;
                h_noise[6] = (unsigned int)(p0_5 >> 32ull) ^ key1_0;
                h_noise[7] = (unsigned int)p0_5;
                long long o_6 = read_rows_base + 3ll * state_stride_p2;
                unsigned long long p0_6 = (unsigned long long)(unsigned int)o_6 * 3528531795ull;
                h_noise[8] = (unsigned int)(o_6 >> 32ll) ^ key0_0;
                h_noise[10] = (unsigned int)(p0_6 >> 32ull) ^ key1_0;
                h_noise[11] = (unsigned int)p0_6;
                long long o_7 = read_rows_base + 3ll * state_stride_p2 + 32ll;
                unsigned long long p0_7 = (unsigned long long)(unsigned int)o_7 * 3528531795ull;
                h_noise[12] = (unsigned int)(o_7 >> 32ll) ^ key0_0;
                h_noise[14] = (unsigned int)(p0_7 >> 32ull) ^ key1_0;
                h_noise[15] = (unsigned int)p0_7;
                key0_0 = key0_0 + 2654435769u;
                key1_0 = key1_0 + 3144134277u;
                unsigned long long q0_4 = (unsigned long long)h_noise[0] * 3528531795ull;
                unsigned long long q1_4 = (unsigned long long)h_noise[2] * 3449720151ull;
                h_noise[0] = (unsigned int)(q1_4 >> 32ull) ^ key0_0;
                h_noise[1] = (unsigned int)q1_4;
                h_noise[2] = (unsigned int)(q0_4 >> 32ull) ^ h_noise[3] ^ key1_0;
                h_noise[3] = (unsigned int)q0_4;
                unsigned long long q0_5 = (unsigned long long)h_noise[4] * 3528531795ull;
                unsigned long long q1_5 = (unsigned long long)h_noise[6] * 3449720151ull;
                h_noise[4] = (unsigned int)(q1_5 >> 32ull) ^ key0_0;
                h_noise[5] = (unsigned int)q1_5;
                h_noise[6] = (unsigned int)(q0_5 >> 32ull) ^ h_noise[7] ^ key1_0;
                h_noise[7] = (unsigned int)q0_5;
                unsigned long long q0_6 = (unsigned long long)h_noise[8] * 3528531795ull;
                unsigned long long q1_6 = (unsigned long long)h_noise[10] * 3449720151ull;
                h_noise[8] = (unsigned int)(q1_6 >> 32ull) ^ key0_0;
                h_noise[9] = (unsigned int)q1_6;
                h_noise[10] = (unsigned int)(q0_6 >> 32ull) ^ h_noise[11] ^ key1_0;
                h_noise[11] = (unsigned int)q0_6;
                unsigned long long q0_7 = (unsigned long long)h_noise[12] * 3528531795ull;
                unsigned long long q1_7 = (unsigned long long)h_noise[14] * 3449720151ull;
                h_noise[12] = (unsigned int)(q1_7 >> 32ull) ^ key0_0;
                h_noise[13] = (unsigned int)q1_7;
                h_noise[14] = (unsigned int)(q0_7 >> 32ull) ^ h_noise[15] ^ key1_0;
                h_noise[15] = (unsigned int)q0_7;
                key0_0 = key0_0 + 2654435769u;
                key1_0 = key1_0 + 3144134277u;
                unsigned long long s0_12 = (unsigned long long)h_noise[0] * 3528531795ull;
                unsigned long long s1_12 = (unsigned long long)h_noise[2] * 3449720151ull;
                h_noise[0] = (unsigned int)(s1_12 >> 32ull) ^ h_noise[1] ^ key0_0;
                h_noise[1] = (unsigned int)s1_12;
                h_noise[2] = (unsigned int)(s0_12 >> 32ull) ^ h_noise[3] ^ key1_0;
                h_noise[3] = (unsigned int)s0_12;
                unsigned long long s0_13 = (unsigned long long)h_noise[4] * 3528531795ull;
                unsigned long long s1_13 = (unsigned long long)h_noise[6] * 3449720151ull;
                h_noise[4] = (unsigned int)(s1_13 >> 32ull) ^ h_noise[5] ^ key0_0;
                h_noise[5] = (unsigned int)s1_13;
                h_noise[6] = (unsigned int)(s0_13 >> 32ull) ^ h_noise[7] ^ key1_0;
                h_noise[7] = (unsigned int)s0_13;
                unsigned long long s0_14 = (unsigned long long)h_noise[8] * 3528531795ull;
                unsigned long long s1_14 = (unsigned long long)h_noise[10] * 3449720151ull;
                h_noise[8] = (unsigned int)(s1_14 >> 32ull) ^ h_noise[9] ^ key0_0;
                h_noise[9] = (unsigned int)s1_14;
                h_noise[10] = (unsigned int)(s0_14 >> 32ull) ^ h_noise[11] ^ key1_0;
                h_noise[11] = (unsigned int)s0_14;
                unsigned long long s0_15 = (unsigned long long)h_noise[12] * 3528531795ull;
                unsigned long long s1_15 = (unsigned long long)h_noise[14] * 3449720151ull;
                h_noise[12] = (unsigned int)(s1_15 >> 32ull) ^ h_noise[13] ^ key0_0;
                h_noise[13] = (unsigned int)s1_15;
                h_noise[14] = (unsigned int)(s0_15 >> 32ull) ^ h_noise[15] ^ key1_0;
                h_noise[15] = (unsigned int)s0_15;
                key0_0 = key0_0 + 2654435769u;
                key1_0 = key1_0 + 3144134277u;
                unsigned long long s0_16 = (unsigned long long)h_noise[0] * 3528531795ull;
                unsigned long long s1_16 = (unsigned long long)h_noise[2] * 3449720151ull;
                h_noise[0] = (unsigned int)(s1_16 >> 32ull) ^ h_noise[1] ^ key0_0;
                h_noise[1] = (unsigned int)s1_16;
                h_noise[2] = (unsigned int)(s0_16 >> 32ull) ^ h_noise[3] ^ key1_0;
                h_noise[3] = (unsigned int)s0_16;
                unsigned long long s0_17 = (unsigned long long)h_noise[4] * 3528531795ull;
                unsigned long long s1_17 = (unsigned long long)h_noise[6] * 3449720151ull;
                h_noise[4] = (unsigned int)(s1_17 >> 32ull) ^ h_noise[5] ^ key0_0;
                h_noise[5] = (unsigned int)s1_17;
                h_noise[6] = (unsigned int)(s0_17 >> 32ull) ^ h_noise[7] ^ key1_0;
                h_noise[7] = (unsigned int)s0_17;
                unsigned long long s0_18 = (unsigned long long)h_noise[8] * 3528531795ull;
                unsigned long long s1_18 = (unsigned long long)h_noise[10] * 3449720151ull;
                h_noise[8] = (unsigned int)(s1_18 >> 32ull) ^ h_noise[9] ^ key0_0;
                h_noise[9] = (unsigned int)s1_18;
                h_noise[10] = (unsigned int)(s0_18 >> 32ull) ^ h_noise[11] ^ key1_0;
                h_noise[11] = (unsigned int)s0_18;
                unsigned long long s0_19 = (unsigned long long)h_noise[12] * 3528531795ull;
                unsigned long long s1_19 = (unsigned long long)h_noise[14] * 3449720151ull;
                h_noise[12] = (unsigned int)(s1_19 >> 32ull) ^ h_noise[13] ^ key0_0;
                h_noise[13] = (unsigned int)s1_19;
                h_noise[14] = (unsigned int)(s0_19 >> 32ull) ^ h_noise[15] ^ key1_0;
                h_noise[15] = (unsigned int)s0_19;
                key0_0 = key0_0 + 2654435769u;
                key1_0 = key1_0 + 3144134277u;
                unsigned long long s0_20 = (unsigned long long)h_noise[0] * 3528531795ull;
                unsigned long long s1_20 = (unsigned long long)h_noise[2] * 3449720151ull;
                h_noise[0] = (unsigned int)(s1_20 >> 32ull) ^ h_noise[1] ^ key0_0;
                h_noise[1] = (unsigned int)s1_20;
                h_noise[2] = (unsigned int)(s0_20 >> 32ull) ^ h_noise[3] ^ key1_0;
                h_noise[3] = (unsigned int)s0_20;
                unsigned long long s0_21 = (unsigned long long)h_noise[4] * 3528531795ull;
                unsigned long long s1_21 = (unsigned long long)h_noise[6] * 3449720151ull;
                h_noise[4] = (unsigned int)(s1_21 >> 32ull) ^ h_noise[5] ^ key0_0;
                h_noise[5] = (unsigned int)s1_21;
                h_noise[6] = (unsigned int)(s0_21 >> 32ull) ^ h_noise[7] ^ key1_0;
                h_noise[7] = (unsigned int)s0_21;
                unsigned long long s0_22 = (unsigned long long)h_noise[8] * 3528531795ull;
                unsigned long long s1_22 = (unsigned long long)h_noise[10] * 3449720151ull;
                h_noise[8] = (unsigned int)(s1_22 >> 32ull) ^ h_noise[9] ^ key0_0;
                h_noise[9] = (unsigned int)s1_22;
                h_noise[10] = (unsigned int)(s0_22 >> 32ull) ^ h_noise[11] ^ key1_0;
                h_noise[11] = (unsigned int)s0_22;
                unsigned long long s0_23 = (unsigned long long)h_noise[12] * 3528531795ull;
                unsigned long long s1_23 = (unsigned long long)h_noise[14] * 3449720151ull;
                h_noise[12] = (unsigned int)(s1_23 >> 32ull) ^ h_noise[13] ^ key0_0;
                h_noise[13] = (unsigned int)s1_23;
                h_noise[14] = (unsigned int)(s0_23 >> 32ull) ^ h_noise[15] ^ key1_0;
                h_noise[15] = (unsigned int)s0_23;
                key0_0 = key0_0 + 2654435769u;
                key1_0 = key1_0 + 3144134277u;
                #pragma unroll 1
                for (int _round_1 = 5; _round_1 < philox_rounds; _round_1++) {
                    unsigned long long t0_4 = (unsigned long long)h_noise[0] * 3528531795ull;
                    unsigned long long t1_4 = (unsigned long long)h_noise[2] * 3449720151ull;
                    h_noise[0] = (unsigned int)(t1_4 >> 32ull) ^ h_noise[1] ^ key0_0;
                    h_noise[1] = (unsigned int)t1_4;
                    h_noise[2] = (unsigned int)(t0_4 >> 32ull) ^ h_noise[3] ^ key1_0;
                    h_noise[3] = (unsigned int)t0_4;
                    unsigned long long t0_5 = (unsigned long long)h_noise[4] * 3528531795ull;
                    unsigned long long t1_5 = (unsigned long long)h_noise[6] * 3449720151ull;
                    h_noise[4] = (unsigned int)(t1_5 >> 32ull) ^ h_noise[5] ^ key0_0;
                    h_noise[5] = (unsigned int)t1_5;
                    h_noise[6] = (unsigned int)(t0_5 >> 32ull) ^ h_noise[7] ^ key1_0;
                    h_noise[7] = (unsigned int)t0_5;
                    unsigned long long t0_6 = (unsigned long long)h_noise[8] * 3528531795ull;
                    unsigned long long t1_6 = (unsigned long long)h_noise[10] * 3449720151ull;
                    h_noise[8] = (unsigned int)(t1_6 >> 32ull) ^ h_noise[9] ^ key0_0;
                    h_noise[9] = (unsigned int)t1_6;
                    h_noise[10] = (unsigned int)(t0_6 >> 32ull) ^ h_noise[11] ^ key1_0;
                    h_noise[11] = (unsigned int)t0_6;
                    unsigned long long t0_7 = (unsigned long long)h_noise[12] * 3528531795ull;
                    unsigned long long t1_7 = (unsigned long long)h_noise[14] * 3449720151ull;
                    h_noise[12] = (unsigned int)(t1_7 >> 32ull) ^ h_noise[13] ^ key0_0;
                    h_noise[13] = (unsigned int)t1_7;
                    h_noise[14] = (unsigned int)(t0_7 >> 32ull) ^ h_noise[15] ^ key1_0;
                    h_noise[15] = (unsigned int)t0_7;
                    key0_0 = key0_0 + 2654435769u;
                    key1_0 = key1_0 + 3144134277u;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 8192), "r"(h_noise[0] & 536813567), "r"(h_noise[1] & 536813567), "r"(h_noise[2] & 536813567), "r"(h_noise[3] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 10240), "r"(h_noise[4] & 536813567), "r"(h_noise[5] & 536813567), "r"(h_noise[6] & 536813567), "r"(h_noise[7] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 12288), "r"(h_noise[8] & 536813567), "r"(h_noise[9] & 536813567), "r"(h_noise[10] & 536813567), "r"(h_noise[11] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 14336), "r"(h_noise[12] & 536813567), "r"(h_noise[13] & 536813567), "r"(h_noise[14] & 536813567), "r"(h_noise[15] & 536813567) : "memory");
#else
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 8192), "r"(h_noise[16] & 536813567), "r"(h_noise[17] & 536813567), "r"(h_noise[18] & 536813567), "r"(h_noise[19] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 10240), "r"(h_noise[20] & 536813567), "r"(h_noise[21] & 536813567), "r"(h_noise[22] & 536813567), "r"(h_noise[23] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 12288), "r"(h_noise[24] & 536813567), "r"(h_noise[25] & 536813567), "r"(h_noise[26] & 536813567), "r"(h_noise[27] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 14336), "r"(h_noise[28] & 536813567), "r"(h_noise[29] & 536813567), "r"(h_noise[30] & 536813567), "r"(h_noise[31] & 536813567) : "memory");
#endif
            }
            #pragma unroll 1
            for (int t = 0; t < 16; t++) {
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
                float _fma_120;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x));
                acc0_x = _fma_120;
                float _fma_121;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y));
                acc0_y = _fma_121;
                float _fma_122;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(S[2]), "f"(k_regs[2]), "f"(acc0_x));
                acc0_x = _fma_122;
                float _fma_123;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_123) : "f"(S[3]), "f"(k_regs[3]), "f"(acc0_y));
                acc0_y = _fma_123;
                float _fma_124;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x));
                acc0_x = _fma_124;
                float _fma_125;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y));
                acc0_y = _fma_125;
                float _fma_126;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(S[6]), "f"(k_regs[6]), "f"(acc0_x));
                acc0_x = _fma_126;
                float _fma_127;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(S[7]), "f"(k_regs[7]), "f"(acc0_y));
                acc0_y = _fma_127;
                float _fma_128;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_128) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x));
                acc0_x = _fma_128;
                float _fma_129;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_129) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y));
                acc0_y = _fma_129;
                float _fma_130;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_130) : "f"(S[10]), "f"(k_regs[10]), "f"(acc0_x));
                acc0_x = _fma_130;
                float _fma_131;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(S[11]), "f"(k_regs[11]), "f"(acc0_y));
                acc0_y = _fma_131;
                float _fma_132;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x));
                acc0_x = _fma_132;
                float _fma_133;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y));
                acc0_y = _fma_133;
                float _fma_134;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_134) : "f"(S[14]), "f"(k_regs[14]), "f"(acc0_x));
                acc0_x = _fma_134;
                float _fma_135;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_135) : "f"(S[15]), "f"(k_regs[15]), "f"(acc0_y));
                acc0_y = _fma_135;
                float dot = acc0_x + acc0_y;
                float _shfl_xor_sync_136 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
                dot = dot + _shfl_xor_sync_136;
                float _shfl_xor_sync_137 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
                dot = dot + _shfl_xor_sync_137;
                float _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
                dot = dot + _shfl_xor_sync_138;
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
                float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
                dot = dot + _shfl_xor_sync_88;
                float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
                dot = dot + _shfl_xor_sync_89;
                float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
                dot = dot + _shfl_xor_sync_90;
#endif
                float vn_0 = (v_regs[0] - dot) * beta_0;
#if __CUDA_ARCH__ == 900
                float _fma_139;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_139) : "f"(k_regs[0]), "f"(vn_0), "f"(S[0]));
                S[0] = _fma_139;
                float _fma_140;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_140) : "f"(k_regs[1]), "f"(vn_0), "f"(S[1]));
                S[1] = _fma_140;
                float _fma_141;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_141) : "f"(k_regs[2]), "f"(vn_0), "f"(S[2]));
                S[2] = _fma_141;
                float _fma_142;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(k_regs[3]), "f"(vn_0), "f"(S[3]));
                S[3] = _fma_142;
                float _fma_143;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_143) : "f"(k_regs[4]), "f"(vn_0), "f"(S[4]));
                S[4] = _fma_143;
                float _fma_144;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_144) : "f"(k_regs[5]), "f"(vn_0), "f"(S[5]));
                S[5] = _fma_144;
                float _fma_145;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_145) : "f"(k_regs[6]), "f"(vn_0), "f"(S[6]));
                S[6] = _fma_145;
                float _fma_146;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_146) : "f"(k_regs[7]), "f"(vn_0), "f"(S[7]));
                S[7] = _fma_146;
                float _fma_147;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_147) : "f"(k_regs[8]), "f"(vn_0), "f"(S[8]));
                S[8] = _fma_147;
                float _fma_148;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_148) : "f"(k_regs[9]), "f"(vn_0), "f"(S[9]));
                S[9] = _fma_148;
                float _fma_149;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_149) : "f"(k_regs[10]), "f"(vn_0), "f"(S[10]));
                S[10] = _fma_149;
                float _fma_150;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_150) : "f"(k_regs[11]), "f"(vn_0), "f"(S[11]));
                S[11] = _fma_150;
                float _fma_151;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_151) : "f"(k_regs[12]), "f"(vn_0), "f"(S[12]));
                S[12] = _fma_151;
                float _fma_152;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_152) : "f"(k_regs[13]), "f"(vn_0), "f"(S[13]));
                S[13] = _fma_152;
                float _fma_153;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_153) : "f"(k_regs[14]), "f"(vn_0), "f"(S[14]));
                S[14] = _fma_153;
                float _fma_154;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_154) : "f"(k_regs[15]), "f"(vn_0), "f"(S[15]));
                S[15] = _fma_154;
                float acc0_x_0 = 0.0f;
                float acc0_y_0 = 0.0f;
                float _fma_155;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_0));
                acc0_x_0 = _fma_155;
                float _fma_156;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_0));
                acc0_y_0 = _fma_156;
                float _fma_157;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(S[18]), "f"(k_regs[2]), "f"(acc0_x_0));
                acc0_x_0 = _fma_157;
                float _fma_158;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_158) : "f"(S[19]), "f"(k_regs[3]), "f"(acc0_y_0));
                acc0_y_0 = _fma_158;
                float _fma_159;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_159) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_0));
                acc0_x_0 = _fma_159;
                float _fma_160;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_160) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_0));
                acc0_y_0 = _fma_160;
                float _fma_161;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_161) : "f"(S[22]), "f"(k_regs[6]), "f"(acc0_x_0));
                acc0_x_0 = _fma_161;
                float _fma_162;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_162) : "f"(S[23]), "f"(k_regs[7]), "f"(acc0_y_0));
                acc0_y_0 = _fma_162;
                float _fma_163;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_163) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_0));
                acc0_x_0 = _fma_163;
                float _fma_164;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_164) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_0));
                acc0_y_0 = _fma_164;
                float _fma_165;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_165) : "f"(S[26]), "f"(k_regs[10]), "f"(acc0_x_0));
                acc0_x_0 = _fma_165;
                float _fma_166;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_166) : "f"(S[27]), "f"(k_regs[11]), "f"(acc0_y_0));
                acc0_y_0 = _fma_166;
                float _fma_167;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_167) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_0));
                acc0_x_0 = _fma_167;
                float _fma_168;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_168) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_0));
                acc0_y_0 = _fma_168;
                float _fma_169;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_169) : "f"(S[30]), "f"(k_regs[14]), "f"(acc0_x_0));
                acc0_x_0 = _fma_169;
                float _fma_170;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_170) : "f"(S[31]), "f"(k_regs[15]), "f"(acc0_y_0));
                acc0_y_0 = _fma_170;
                float dot_0 = acc0_x_0 + acc0_y_0;
                float _shfl_xor_sync_171 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 1);
                dot_0 = dot_0 + _shfl_xor_sync_171;
                float _shfl_xor_sync_172 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 2);
                dot_0 = dot_0 + _shfl_xor_sync_172;
                float _shfl_xor_sync_173 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 4);
                dot_0 = dot_0 + _shfl_xor_sync_173;
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
                float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 1);
                dot_0 = dot_0 + _shfl_xor_sync_91;
                float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 2);
                dot_0 = dot_0 + _shfl_xor_sync_92;
                float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 4);
                dot_0 = dot_0 + _shfl_xor_sync_93;
#endif
                float vn_1 = (v_regs[1] - dot_0) * beta_0;
#if __CUDA_ARCH__ == 900
                float _fma_174;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_174) : "f"(k_regs[0]), "f"(vn_1), "f"(S[16]));
                S[16] = _fma_174;
                float _fma_175;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_175) : "f"(k_regs[1]), "f"(vn_1), "f"(S[17]));
                S[17] = _fma_175;
                float _fma_176;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_176) : "f"(k_regs[2]), "f"(vn_1), "f"(S[18]));
                S[18] = _fma_176;
                float _fma_177;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_177) : "f"(k_regs[3]), "f"(vn_1), "f"(S[19]));
                S[19] = _fma_177;
                float _fma_178;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_178) : "f"(k_regs[4]), "f"(vn_1), "f"(S[20]));
                S[20] = _fma_178;
                float _fma_179;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_179) : "f"(k_regs[5]), "f"(vn_1), "f"(S[21]));
                S[21] = _fma_179;
                float _fma_180;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_180) : "f"(k_regs[6]), "f"(vn_1), "f"(S[22]));
                S[22] = _fma_180;
                float _fma_181;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_181) : "f"(k_regs[7]), "f"(vn_1), "f"(S[23]));
                S[23] = _fma_181;
                float _fma_182;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_182) : "f"(k_regs[8]), "f"(vn_1), "f"(S[24]));
                S[24] = _fma_182;
                float _fma_183;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_183) : "f"(k_regs[9]), "f"(vn_1), "f"(S[25]));
                S[25] = _fma_183;
                float _fma_184;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_184) : "f"(k_regs[10]), "f"(vn_1), "f"(S[26]));
                S[26] = _fma_184;
                float _fma_185;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_185) : "f"(k_regs[11]), "f"(vn_1), "f"(S[27]));
                S[27] = _fma_185;
                float _fma_186;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_186) : "f"(k_regs[12]), "f"(vn_1), "f"(S[28]));
                S[28] = _fma_186;
                float _fma_187;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_187) : "f"(k_regs[13]), "f"(vn_1), "f"(S[29]));
                S[29] = _fma_187;
                float _fma_188;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_188) : "f"(k_regs[14]), "f"(vn_1), "f"(S[30]));
                S[30] = _fma_188;
                float _fma_189;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_189) : "f"(k_regs[15]), "f"(vn_1), "f"(S[31]));
                S[31] = _fma_189;
                float acc0_x_1 = 0.0f;
                float acc0_y_1 = 0.0f;
                float _fma_190;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_190) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_1));
                acc0_x_1 = _fma_190;
                float _fma_191;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_191) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_1));
                acc0_y_1 = _fma_191;
                float _fma_192;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_192) : "f"(S[34]), "f"(k_regs[2]), "f"(acc0_x_1));
                acc0_x_1 = _fma_192;
                float _fma_193;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_193) : "f"(S[35]), "f"(k_regs[3]), "f"(acc0_y_1));
                acc0_y_1 = _fma_193;
                float _fma_194;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_194) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_1));
                acc0_x_1 = _fma_194;
                float _fma_195;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_195) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_1));
                acc0_y_1 = _fma_195;
                float _fma_196;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_196) : "f"(S[38]), "f"(k_regs[6]), "f"(acc0_x_1));
                acc0_x_1 = _fma_196;
                float _fma_197;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_197) : "f"(S[39]), "f"(k_regs[7]), "f"(acc0_y_1));
                acc0_y_1 = _fma_197;
                float _fma_198;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_198) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_1));
                acc0_x_1 = _fma_198;
                float _fma_199;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_199) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_1));
                acc0_y_1 = _fma_199;
                float _fma_200;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_200) : "f"(S[42]), "f"(k_regs[10]), "f"(acc0_x_1));
                acc0_x_1 = _fma_200;
                float _fma_201;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_201) : "f"(S[43]), "f"(k_regs[11]), "f"(acc0_y_1));
                acc0_y_1 = _fma_201;
                float _fma_202;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_202) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_1));
                acc0_x_1 = _fma_202;
                float _fma_203;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_203) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_1));
                acc0_y_1 = _fma_203;
                float _fma_204;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_204) : "f"(S[46]), "f"(k_regs[14]), "f"(acc0_x_1));
                acc0_x_1 = _fma_204;
                float _fma_205;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_205) : "f"(S[47]), "f"(k_regs[15]), "f"(acc0_y_1));
                acc0_y_1 = _fma_205;
                float dot_1 = acc0_x_1 + acc0_y_1;
                float _shfl_xor_sync_206 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_206;
                float _shfl_xor_sync_207 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_207;
                float _shfl_xor_sync_208 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
                dot_1 = dot_1 + _shfl_xor_sync_208;
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
                float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_94;
                float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_95;
                float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
                dot_1 = dot_1 + _shfl_xor_sync_96;
#endif
                float vn_2 = (v_regs[2] - dot_1) * beta_0;
#if __CUDA_ARCH__ == 900
                float _fma_209;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_209) : "f"(k_regs[0]), "f"(vn_2), "f"(S[32]));
                S[32] = _fma_209;
                float _fma_210;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_210) : "f"(k_regs[1]), "f"(vn_2), "f"(S[33]));
                S[33] = _fma_210;
                float _fma_211;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_211) : "f"(k_regs[2]), "f"(vn_2), "f"(S[34]));
                S[34] = _fma_211;
                float _fma_212;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_212) : "f"(k_regs[3]), "f"(vn_2), "f"(S[35]));
                S[35] = _fma_212;
                float _fma_213;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_213) : "f"(k_regs[4]), "f"(vn_2), "f"(S[36]));
                S[36] = _fma_213;
                float _fma_214;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_214) : "f"(k_regs[5]), "f"(vn_2), "f"(S[37]));
                S[37] = _fma_214;
                float _fma_215;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_215) : "f"(k_regs[6]), "f"(vn_2), "f"(S[38]));
                S[38] = _fma_215;
                float _fma_216;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_216) : "f"(k_regs[7]), "f"(vn_2), "f"(S[39]));
                S[39] = _fma_216;
                float _fma_217;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_217) : "f"(k_regs[8]), "f"(vn_2), "f"(S[40]));
                S[40] = _fma_217;
                float _fma_218;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_218) : "f"(k_regs[9]), "f"(vn_2), "f"(S[41]));
                S[41] = _fma_218;
                float _fma_219;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_219) : "f"(k_regs[10]), "f"(vn_2), "f"(S[42]));
                S[42] = _fma_219;
                float _fma_220;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_220) : "f"(k_regs[11]), "f"(vn_2), "f"(S[43]));
                S[43] = _fma_220;
                float _fma_221;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_221) : "f"(k_regs[12]), "f"(vn_2), "f"(S[44]));
                S[44] = _fma_221;
                float _fma_222;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_222) : "f"(k_regs[13]), "f"(vn_2), "f"(S[45]));
                S[45] = _fma_222;
                float _fma_223;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_223) : "f"(k_regs[14]), "f"(vn_2), "f"(S[46]));
                S[46] = _fma_223;
                float _fma_224;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_224) : "f"(k_regs[15]), "f"(vn_2), "f"(S[47]));
                S[47] = _fma_224;
                float acc0_x_2 = 0.0f;
                float acc0_y_2 = 0.0f;
                float _fma_225;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_225) : "f"(S[48]), "f"(k_regs[0]), "f"(acc0_x_2));
                acc0_x_2 = _fma_225;
                float _fma_226;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_226) : "f"(S[49]), "f"(k_regs[1]), "f"(acc0_y_2));
                acc0_y_2 = _fma_226;
                float _fma_227;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_227) : "f"(S[50]), "f"(k_regs[2]), "f"(acc0_x_2));
                acc0_x_2 = _fma_227;
                float _fma_228;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_228) : "f"(S[51]), "f"(k_regs[3]), "f"(acc0_y_2));
                acc0_y_2 = _fma_228;
                float _fma_229;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_229) : "f"(S[52]), "f"(k_regs[4]), "f"(acc0_x_2));
                acc0_x_2 = _fma_229;
                float _fma_230;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_230) : "f"(S[53]), "f"(k_regs[5]), "f"(acc0_y_2));
                acc0_y_2 = _fma_230;
                float _fma_231;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_231) : "f"(S[54]), "f"(k_regs[6]), "f"(acc0_x_2));
                acc0_x_2 = _fma_231;
                float _fma_232;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_232) : "f"(S[55]), "f"(k_regs[7]), "f"(acc0_y_2));
                acc0_y_2 = _fma_232;
                float _fma_233;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_233) : "f"(S[56]), "f"(k_regs[8]), "f"(acc0_x_2));
                acc0_x_2 = _fma_233;
                float _fma_234;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_234) : "f"(S[57]), "f"(k_regs[9]), "f"(acc0_y_2));
                acc0_y_2 = _fma_234;
                float _fma_235;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_235) : "f"(S[58]), "f"(k_regs[10]), "f"(acc0_x_2));
                acc0_x_2 = _fma_235;
                float _fma_236;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_236) : "f"(S[59]), "f"(k_regs[11]), "f"(acc0_y_2));
                acc0_y_2 = _fma_236;
                float _fma_237;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_237) : "f"(S[60]), "f"(k_regs[12]), "f"(acc0_x_2));
                acc0_x_2 = _fma_237;
                float _fma_238;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_238) : "f"(S[61]), "f"(k_regs[13]), "f"(acc0_y_2));
                acc0_y_2 = _fma_238;
                float _fma_239;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_239) : "f"(S[62]), "f"(k_regs[14]), "f"(acc0_x_2));
                acc0_x_2 = _fma_239;
                float _fma_240;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_240) : "f"(S[63]), "f"(k_regs[15]), "f"(acc0_y_2));
                acc0_y_2 = _fma_240;
                float dot_2 = acc0_x_2 + acc0_y_2;
                float _shfl_xor_sync_241 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 1);
                dot_2 = dot_2 + _shfl_xor_sync_241;
                float _shfl_xor_sync_242 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 2);
                dot_2 = dot_2 + _shfl_xor_sync_242;
                float _shfl_xor_sync_243 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 4);
                dot_2 = dot_2 + _shfl_xor_sync_243;
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
                float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 1);
                dot_2 = dot_2 + _shfl_xor_sync_97;
                float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 2);
                dot_2 = dot_2 + _shfl_xor_sync_98;
                float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 4);
                dot_2 = dot_2 + _shfl_xor_sync_99;
#endif
                float vn_3 = (v_regs[3] - dot_2) * beta_0;
#if __CUDA_ARCH__ == 900
                float _fma_244;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_244) : "f"(k_regs[0]), "f"(vn_3), "f"(S[48]));
                S[48] = _fma_244;
                float _fma_245;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_245) : "f"(k_regs[1]), "f"(vn_3), "f"(S[49]));
                S[49] = _fma_245;
                float _fma_246;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_246) : "f"(k_regs[2]), "f"(vn_3), "f"(S[50]));
                S[50] = _fma_246;
                float _fma_247;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_247) : "f"(k_regs[3]), "f"(vn_3), "f"(S[51]));
                S[51] = _fma_247;
                float _fma_248;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_248) : "f"(k_regs[4]), "f"(vn_3), "f"(S[52]));
                S[52] = _fma_248;
                float _fma_249;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_249) : "f"(k_regs[5]), "f"(vn_3), "f"(S[53]));
                S[53] = _fma_249;
                float _fma_250;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_250) : "f"(k_regs[6]), "f"(vn_3), "f"(S[54]));
                S[54] = _fma_250;
                float _fma_251;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_251) : "f"(k_regs[7]), "f"(vn_3), "f"(S[55]));
                S[55] = _fma_251;
                float _fma_252;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_252) : "f"(k_regs[8]), "f"(vn_3), "f"(S[56]));
                S[56] = _fma_252;
                float _fma_253;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_253) : "f"(k_regs[9]), "f"(vn_3), "f"(S[57]));
                S[57] = _fma_253;
                float _fma_254;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_254) : "f"(k_regs[10]), "f"(vn_3), "f"(S[58]));
                S[58] = _fma_254;
                float _fma_255;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_255) : "f"(k_regs[11]), "f"(vn_3), "f"(S[59]));
                S[59] = _fma_255;
                float _fma_256;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_256) : "f"(k_regs[12]), "f"(vn_3), "f"(S[60]));
                S[60] = _fma_256;
                float _fma_257;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_257) : "f"(k_regs[13]), "f"(vn_3), "f"(S[61]));
                S[61] = _fma_257;
                float _fma_258;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_258) : "f"(k_regs[14]), "f"(vn_3), "f"(S[62]));
                S[62] = _fma_258;
                float _fma_259;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_259) : "f"(k_regs[15]), "f"(vn_3), "f"(S[63]));
                S[63] = _fma_259;
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
                float _fma_260;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_260) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_3));
                acc0_x_3 = _fma_260;
                float _fma_261;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_261) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_3));
                acc0_y_3 = _fma_261;
                float _fma_262;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_262) : "f"(S[2]), "f"(k_regs[2]), "f"(acc0_x_3));
                acc0_x_3 = _fma_262;
                float _fma_263;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_263) : "f"(S[3]), "f"(k_regs[3]), "f"(acc0_y_3));
                acc0_y_3 = _fma_263;
                float _fma_264;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_264) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_3));
                acc0_x_3 = _fma_264;
                float _fma_265;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_265) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_3));
                acc0_y_3 = _fma_265;
                float _fma_266;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_266) : "f"(S[6]), "f"(k_regs[6]), "f"(acc0_x_3));
                acc0_x_3 = _fma_266;
                float _fma_267;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_267) : "f"(S[7]), "f"(k_regs[7]), "f"(acc0_y_3));
                acc0_y_3 = _fma_267;
                float _fma_268;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_268) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_3));
                acc0_x_3 = _fma_268;
                float _fma_269;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_269) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_3));
                acc0_y_3 = _fma_269;
                float _fma_270;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_270) : "f"(S[10]), "f"(k_regs[10]), "f"(acc0_x_3));
                acc0_x_3 = _fma_270;
                float _fma_271;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_271) : "f"(S[11]), "f"(k_regs[11]), "f"(acc0_y_3));
                acc0_y_3 = _fma_271;
                float _fma_272;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_272) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_3));
                acc0_x_3 = _fma_272;
                float _fma_273;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_273) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_3));
                acc0_y_3 = _fma_273;
                float _fma_274;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_274) : "f"(S[14]), "f"(k_regs[14]), "f"(acc0_x_3));
                acc0_x_3 = _fma_274;
                float _fma_275;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_275) : "f"(S[15]), "f"(k_regs[15]), "f"(acc0_y_3));
                acc0_y_3 = _fma_275;
                float dot_3 = acc0_x_3 + acc0_y_3;
                float _shfl_xor_sync_276 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 1);
                dot_3 = dot_3 + _shfl_xor_sync_276;
                float _shfl_xor_sync_277 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 2);
                dot_3 = dot_3 + _shfl_xor_sync_277;
                float _shfl_xor_sync_278 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 4);
                dot_3 = dot_3 + _shfl_xor_sync_278;
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
                float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 1);
                dot_3 = dot_3 + _shfl_xor_sync_100;
                float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 2);
                dot_3 = dot_3 + _shfl_xor_sync_101;
                float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 4);
                dot_3 = dot_3 + _shfl_xor_sync_102;
#endif
                float vn_4 = (v_regs[0] - dot_3) * beta_1;
#if __CUDA_ARCH__ == 900
                float _fma_279;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_279) : "f"(k_regs[0]), "f"(vn_4), "f"(S[0]));
                S[0] = _fma_279;
                float _fma_280;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_280) : "f"(k_regs[1]), "f"(vn_4), "f"(S[1]));
                S[1] = _fma_280;
                float _fma_281;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_281) : "f"(k_regs[2]), "f"(vn_4), "f"(S[2]));
                S[2] = _fma_281;
                float _fma_282;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_282) : "f"(k_regs[3]), "f"(vn_4), "f"(S[3]));
                S[3] = _fma_282;
                float _fma_283;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_283) : "f"(k_regs[4]), "f"(vn_4), "f"(S[4]));
                S[4] = _fma_283;
                float _fma_284;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_284) : "f"(k_regs[5]), "f"(vn_4), "f"(S[5]));
                S[5] = _fma_284;
                float _fma_285;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_285) : "f"(k_regs[6]), "f"(vn_4), "f"(S[6]));
                S[6] = _fma_285;
                float _fma_286;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_286) : "f"(k_regs[7]), "f"(vn_4), "f"(S[7]));
                S[7] = _fma_286;
                float _fma_287;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_287) : "f"(k_regs[8]), "f"(vn_4), "f"(S[8]));
                S[8] = _fma_287;
                float _fma_288;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_288) : "f"(k_regs[9]), "f"(vn_4), "f"(S[9]));
                S[9] = _fma_288;
                float _fma_289;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_289) : "f"(k_regs[10]), "f"(vn_4), "f"(S[10]));
                S[10] = _fma_289;
                float _fma_290;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_290) : "f"(k_regs[11]), "f"(vn_4), "f"(S[11]));
                S[11] = _fma_290;
                float _fma_291;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_291) : "f"(k_regs[12]), "f"(vn_4), "f"(S[12]));
                S[12] = _fma_291;
                float _fma_292;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_292) : "f"(k_regs[13]), "f"(vn_4), "f"(S[13]));
                S[13] = _fma_292;
                float _fma_293;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_293) : "f"(k_regs[14]), "f"(vn_4), "f"(S[14]));
                S[14] = _fma_293;
                float _fma_294;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_294) : "f"(k_regs[15]), "f"(vn_4), "f"(S[15]));
                S[15] = _fma_294;
                float acc0_x_4 = 0.0f;
                float acc0_y_4 = 0.0f;
                float _fma_295;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_295) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_4));
                acc0_x_4 = _fma_295;
                float _fma_296;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_296) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_4));
                acc0_y_4 = _fma_296;
                float _fma_297;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_297) : "f"(S[18]), "f"(k_regs[2]), "f"(acc0_x_4));
                acc0_x_4 = _fma_297;
                float _fma_298;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_298) : "f"(S[19]), "f"(k_regs[3]), "f"(acc0_y_4));
                acc0_y_4 = _fma_298;
                float _fma_299;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_299) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_4));
                acc0_x_4 = _fma_299;
                float _fma_300;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_300) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_4));
                acc0_y_4 = _fma_300;
                float _fma_301;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_301) : "f"(S[22]), "f"(k_regs[6]), "f"(acc0_x_4));
                acc0_x_4 = _fma_301;
                float _fma_302;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_302) : "f"(S[23]), "f"(k_regs[7]), "f"(acc0_y_4));
                acc0_y_4 = _fma_302;
                float _fma_303;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_303) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_4));
                acc0_x_4 = _fma_303;
                float _fma_304;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_304) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_4));
                acc0_y_4 = _fma_304;
                float _fma_305;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_305) : "f"(S[26]), "f"(k_regs[10]), "f"(acc0_x_4));
                acc0_x_4 = _fma_305;
                float _fma_306;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_306) : "f"(S[27]), "f"(k_regs[11]), "f"(acc0_y_4));
                acc0_y_4 = _fma_306;
                float _fma_307;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_307) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_4));
                acc0_x_4 = _fma_307;
                float _fma_308;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_308) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_4));
                acc0_y_4 = _fma_308;
                float _fma_309;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_309) : "f"(S[30]), "f"(k_regs[14]), "f"(acc0_x_4));
                acc0_x_4 = _fma_309;
                float _fma_310;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_310) : "f"(S[31]), "f"(k_regs[15]), "f"(acc0_y_4));
                acc0_y_4 = _fma_310;
                float dot_4 = acc0_x_4 + acc0_y_4;
                float _shfl_xor_sync_311 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 1);
                dot_4 = dot_4 + _shfl_xor_sync_311;
                float _shfl_xor_sync_312 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 2);
                dot_4 = dot_4 + _shfl_xor_sync_312;
                float _shfl_xor_sync_313 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 4);
                dot_4 = dot_4 + _shfl_xor_sync_313;
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
                float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 1);
                dot_4 = dot_4 + _shfl_xor_sync_103;
                float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 2);
                dot_4 = dot_4 + _shfl_xor_sync_104;
                float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 4);
                dot_4 = dot_4 + _shfl_xor_sync_105;
#endif
                float vn_5 = (v_regs[1] - dot_4) * beta_1;
#if __CUDA_ARCH__ == 900
                float _fma_314;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_314) : "f"(k_regs[0]), "f"(vn_5), "f"(S[16]));
                S[16] = _fma_314;
                float _fma_315;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_315) : "f"(k_regs[1]), "f"(vn_5), "f"(S[17]));
                S[17] = _fma_315;
                float _fma_316;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_316) : "f"(k_regs[2]), "f"(vn_5), "f"(S[18]));
                S[18] = _fma_316;
                float _fma_317;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_317) : "f"(k_regs[3]), "f"(vn_5), "f"(S[19]));
                S[19] = _fma_317;
                float _fma_318;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_318) : "f"(k_regs[4]), "f"(vn_5), "f"(S[20]));
                S[20] = _fma_318;
                float _fma_319;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_319) : "f"(k_regs[5]), "f"(vn_5), "f"(S[21]));
                S[21] = _fma_319;
                float _fma_320;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_320) : "f"(k_regs[6]), "f"(vn_5), "f"(S[22]));
                S[22] = _fma_320;
                float _fma_321;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_321) : "f"(k_regs[7]), "f"(vn_5), "f"(S[23]));
                S[23] = _fma_321;
                float _fma_322;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_322) : "f"(k_regs[8]), "f"(vn_5), "f"(S[24]));
                S[24] = _fma_322;
                float _fma_323;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_323) : "f"(k_regs[9]), "f"(vn_5), "f"(S[25]));
                S[25] = _fma_323;
                float _fma_324;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_324) : "f"(k_regs[10]), "f"(vn_5), "f"(S[26]));
                S[26] = _fma_324;
                float _fma_325;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_325) : "f"(k_regs[11]), "f"(vn_5), "f"(S[27]));
                S[27] = _fma_325;
                float _fma_326;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_326) : "f"(k_regs[12]), "f"(vn_5), "f"(S[28]));
                S[28] = _fma_326;
                float _fma_327;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_327) : "f"(k_regs[13]), "f"(vn_5), "f"(S[29]));
                S[29] = _fma_327;
                float _fma_328;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_328) : "f"(k_regs[14]), "f"(vn_5), "f"(S[30]));
                S[30] = _fma_328;
                float _fma_329;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_329) : "f"(k_regs[15]), "f"(vn_5), "f"(S[31]));
                S[31] = _fma_329;
                float acc0_x_5 = 0.0f;
                float acc0_y_5 = 0.0f;
                float _fma_330;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_330) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_5));
                acc0_x_5 = _fma_330;
                float _fma_331;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_331) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_5));
                acc0_y_5 = _fma_331;
                float _fma_332;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_332) : "f"(S[34]), "f"(k_regs[2]), "f"(acc0_x_5));
                acc0_x_5 = _fma_332;
                float _fma_333;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_333) : "f"(S[35]), "f"(k_regs[3]), "f"(acc0_y_5));
                acc0_y_5 = _fma_333;
                float _fma_334;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_334) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_5));
                acc0_x_5 = _fma_334;
                float _fma_335;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_335) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_5));
                acc0_y_5 = _fma_335;
                float _fma_336;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_336) : "f"(S[38]), "f"(k_regs[6]), "f"(acc0_x_5));
                acc0_x_5 = _fma_336;
                float _fma_337;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_337) : "f"(S[39]), "f"(k_regs[7]), "f"(acc0_y_5));
                acc0_y_5 = _fma_337;
                float _fma_338;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_338) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_5));
                acc0_x_5 = _fma_338;
                float _fma_339;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_339) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_5));
                acc0_y_5 = _fma_339;
                float _fma_340;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_340) : "f"(S[42]), "f"(k_regs[10]), "f"(acc0_x_5));
                acc0_x_5 = _fma_340;
                float _fma_341;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_341) : "f"(S[43]), "f"(k_regs[11]), "f"(acc0_y_5));
                acc0_y_5 = _fma_341;
                float _fma_342;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_342) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_5));
                acc0_x_5 = _fma_342;
                float _fma_343;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_343) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_5));
                acc0_y_5 = _fma_343;
                float _fma_344;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_344) : "f"(S[46]), "f"(k_regs[14]), "f"(acc0_x_5));
                acc0_x_5 = _fma_344;
                float _fma_345;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_345) : "f"(S[47]), "f"(k_regs[15]), "f"(acc0_y_5));
                acc0_y_5 = _fma_345;
                float dot_5 = acc0_x_5 + acc0_y_5;
                float _shfl_xor_sync_346 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 1);
                dot_5 = dot_5 + _shfl_xor_sync_346;
                float _shfl_xor_sync_347 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 2);
                dot_5 = dot_5 + _shfl_xor_sync_347;
                float _shfl_xor_sync_348 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 4);
                dot_5 = dot_5 + _shfl_xor_sync_348;
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
                float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 1);
                dot_5 = dot_5 + _shfl_xor_sync_106;
                float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 2);
                dot_5 = dot_5 + _shfl_xor_sync_107;
                float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 4);
                dot_5 = dot_5 + _shfl_xor_sync_108;
#endif
                float vn_6 = (v_regs[2] - dot_5) * beta_1;
#if __CUDA_ARCH__ == 900
                float _fma_349;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_349) : "f"(k_regs[0]), "f"(vn_6), "f"(S[32]));
                S[32] = _fma_349;
                float _fma_350;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_350) : "f"(k_regs[1]), "f"(vn_6), "f"(S[33]));
                S[33] = _fma_350;
                float _fma_351;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_351) : "f"(k_regs[2]), "f"(vn_6), "f"(S[34]));
                S[34] = _fma_351;
                float _fma_352;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_352) : "f"(k_regs[3]), "f"(vn_6), "f"(S[35]));
                S[35] = _fma_352;
                float _fma_353;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_353) : "f"(k_regs[4]), "f"(vn_6), "f"(S[36]));
                S[36] = _fma_353;
                float _fma_354;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_354) : "f"(k_regs[5]), "f"(vn_6), "f"(S[37]));
                S[37] = _fma_354;
                float _fma_355;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_355) : "f"(k_regs[6]), "f"(vn_6), "f"(S[38]));
                S[38] = _fma_355;
                float _fma_356;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_356) : "f"(k_regs[7]), "f"(vn_6), "f"(S[39]));
                S[39] = _fma_356;
                float _fma_357;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_357) : "f"(k_regs[8]), "f"(vn_6), "f"(S[40]));
                S[40] = _fma_357;
                float _fma_358;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_358) : "f"(k_regs[9]), "f"(vn_6), "f"(S[41]));
                S[41] = _fma_358;
                float _fma_359;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_359) : "f"(k_regs[10]), "f"(vn_6), "f"(S[42]));
                S[42] = _fma_359;
                float _fma_360;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_360) : "f"(k_regs[11]), "f"(vn_6), "f"(S[43]));
                S[43] = _fma_360;
                float _fma_361;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_361) : "f"(k_regs[12]), "f"(vn_6), "f"(S[44]));
                S[44] = _fma_361;
                float _fma_362;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_362) : "f"(k_regs[13]), "f"(vn_6), "f"(S[45]));
                S[45] = _fma_362;
                float _fma_363;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_363) : "f"(k_regs[14]), "f"(vn_6), "f"(S[46]));
                S[46] = _fma_363;
                float _fma_364;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_364) : "f"(k_regs[15]), "f"(vn_6), "f"(S[47]));
                S[47] = _fma_364;
                float acc0_x_6 = 0.0f;
                float acc0_y_6 = 0.0f;
                float _fma_365;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_365) : "f"(S[48]), "f"(k_regs[0]), "f"(acc0_x_6));
                acc0_x_6 = _fma_365;
                float _fma_366;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_366) : "f"(S[49]), "f"(k_regs[1]), "f"(acc0_y_6));
                acc0_y_6 = _fma_366;
                float _fma_367;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_367) : "f"(S[50]), "f"(k_regs[2]), "f"(acc0_x_6));
                acc0_x_6 = _fma_367;
                float _fma_368;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_368) : "f"(S[51]), "f"(k_regs[3]), "f"(acc0_y_6));
                acc0_y_6 = _fma_368;
                float _fma_369;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_369) : "f"(S[52]), "f"(k_regs[4]), "f"(acc0_x_6));
                acc0_x_6 = _fma_369;
                float _fma_370;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_370) : "f"(S[53]), "f"(k_regs[5]), "f"(acc0_y_6));
                acc0_y_6 = _fma_370;
                float _fma_371;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_371) : "f"(S[54]), "f"(k_regs[6]), "f"(acc0_x_6));
                acc0_x_6 = _fma_371;
                float _fma_372;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_372) : "f"(S[55]), "f"(k_regs[7]), "f"(acc0_y_6));
                acc0_y_6 = _fma_372;
                float _fma_373;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_373) : "f"(S[56]), "f"(k_regs[8]), "f"(acc0_x_6));
                acc0_x_6 = _fma_373;
                float _fma_374;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_374) : "f"(S[57]), "f"(k_regs[9]), "f"(acc0_y_6));
                acc0_y_6 = _fma_374;
                float _fma_375;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_375) : "f"(S[58]), "f"(k_regs[10]), "f"(acc0_x_6));
                acc0_x_6 = _fma_375;
                float _fma_376;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_376) : "f"(S[59]), "f"(k_regs[11]), "f"(acc0_y_6));
                acc0_y_6 = _fma_376;
                float _fma_377;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_377) : "f"(S[60]), "f"(k_regs[12]), "f"(acc0_x_6));
                acc0_x_6 = _fma_377;
                float _fma_378;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_378) : "f"(S[61]), "f"(k_regs[13]), "f"(acc0_y_6));
                acc0_y_6 = _fma_378;
                float _fma_379;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_379) : "f"(S[62]), "f"(k_regs[14]), "f"(acc0_x_6));
                acc0_x_6 = _fma_379;
                float _fma_380;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_380) : "f"(S[63]), "f"(k_regs[15]), "f"(acc0_y_6));
                acc0_y_6 = _fma_380;
                float dot_6 = acc0_x_6 + acc0_y_6;
                float _shfl_xor_sync_381 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 1);
                dot_6 = dot_6 + _shfl_xor_sync_381;
                float _shfl_xor_sync_382 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 2);
                dot_6 = dot_6 + _shfl_xor_sync_382;
                float _shfl_xor_sync_383 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 4);
                dot_6 = dot_6 + _shfl_xor_sync_383;
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
                float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 1);
                dot_6 = dot_6 + _shfl_xor_sync_109;
                float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 2);
                dot_6 = dot_6 + _shfl_xor_sync_110;
                float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 4);
                dot_6 = dot_6 + _shfl_xor_sync_111;
#endif
                float vn_7 = (v_regs[3] - dot_6) * beta_1;
#if __CUDA_ARCH__ == 900
                float _fma_384;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_384) : "f"(k_regs[0]), "f"(vn_7), "f"(S[48]));
                S[48] = _fma_384;
                float _fma_385;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_385) : "f"(k_regs[1]), "f"(vn_7), "f"(S[49]));
                S[49] = _fma_385;
                float _fma_386;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_386) : "f"(k_regs[2]), "f"(vn_7), "f"(S[50]));
                S[50] = _fma_386;
                float _fma_387;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_387) : "f"(k_regs[3]), "f"(vn_7), "f"(S[51]));
                S[51] = _fma_387;
                float _fma_388;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_388) : "f"(k_regs[4]), "f"(vn_7), "f"(S[52]));
                S[52] = _fma_388;
                float _fma_389;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_389) : "f"(k_regs[5]), "f"(vn_7), "f"(S[53]));
                S[53] = _fma_389;
                float _fma_390;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_390) : "f"(k_regs[6]), "f"(vn_7), "f"(S[54]));
                S[54] = _fma_390;
                float _fma_391;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_391) : "f"(k_regs[7]), "f"(vn_7), "f"(S[55]));
                S[55] = _fma_391;
                float _fma_392;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_392) : "f"(k_regs[8]), "f"(vn_7), "f"(S[56]));
                S[56] = _fma_392;
                float _fma_393;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_393) : "f"(k_regs[9]), "f"(vn_7), "f"(S[57]));
                S[57] = _fma_393;
                float _fma_394;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_394) : "f"(k_regs[10]), "f"(vn_7), "f"(S[58]));
                S[58] = _fma_394;
                float _fma_395;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_395) : "f"(k_regs[11]), "f"(vn_7), "f"(S[59]));
                S[59] = _fma_395;
                float _fma_396;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_396) : "f"(k_regs[12]), "f"(vn_7), "f"(S[60]));
                S[60] = _fma_396;
                float _fma_397;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_397) : "f"(k_regs[13]), "f"(vn_7), "f"(S[61]));
                S[61] = _fma_397;
                float _fma_398;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_398) : "f"(k_regs[14]), "f"(vn_7), "f"(S[62]));
                S[62] = _fma_398;
                float _fma_399;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_399) : "f"(k_regs[15]), "f"(vn_7), "f"(S[63]));
                S[63] = _fma_399;
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
                float _fma_400;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_400) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_7));
                acc0_x_7 = _fma_400;
                float _fma_401;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_401) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_7));
                acc0_y_7 = _fma_401;
                float _fma_402;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_402) : "f"(S[2]), "f"(k_regs[2]), "f"(acc0_x_7));
                acc0_x_7 = _fma_402;
                float _fma_403;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_403) : "f"(S[3]), "f"(k_regs[3]), "f"(acc0_y_7));
                acc0_y_7 = _fma_403;
                float _fma_404;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_404) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_7));
                acc0_x_7 = _fma_404;
                float _fma_405;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_405) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_7));
                acc0_y_7 = _fma_405;
                float _fma_406;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_406) : "f"(S[6]), "f"(k_regs[6]), "f"(acc0_x_7));
                acc0_x_7 = _fma_406;
                float _fma_407;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_407) : "f"(S[7]), "f"(k_regs[7]), "f"(acc0_y_7));
                acc0_y_7 = _fma_407;
                float _fma_408;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_408) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_7));
                acc0_x_7 = _fma_408;
                float _fma_409;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_409) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_7));
                acc0_y_7 = _fma_409;
                float _fma_410;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_410) : "f"(S[10]), "f"(k_regs[10]), "f"(acc0_x_7));
                acc0_x_7 = _fma_410;
                float _fma_411;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_411) : "f"(S[11]), "f"(k_regs[11]), "f"(acc0_y_7));
                acc0_y_7 = _fma_411;
                float _fma_412;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_412) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_7));
                acc0_x_7 = _fma_412;
                float _fma_413;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_413) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_7));
                acc0_y_7 = _fma_413;
                float _fma_414;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_414) : "f"(S[14]), "f"(k_regs[14]), "f"(acc0_x_7));
                acc0_x_7 = _fma_414;
                float _fma_415;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_415) : "f"(S[15]), "f"(k_regs[15]), "f"(acc0_y_7));
                acc0_y_7 = _fma_415;
                float dot_7 = acc0_x_7 + acc0_y_7;
                float _shfl_xor_sync_416 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 1);
                dot_7 = dot_7 + _shfl_xor_sync_416;
                float _shfl_xor_sync_417 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 2);
                dot_7 = dot_7 + _shfl_xor_sync_417;
                float _shfl_xor_sync_418 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 4);
                dot_7 = dot_7 + _shfl_xor_sync_418;
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
                float _shfl_xor_sync_112 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 1);
                dot_7 = dot_7 + _shfl_xor_sync_112;
                float _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 2);
                dot_7 = dot_7 + _shfl_xor_sync_113;
                float _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 4);
                dot_7 = dot_7 + _shfl_xor_sync_114;
#endif
                float vn_8 = (v_regs[0] - dot_7) * beta_2;
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
                float acc0_x_8 = 0.0f;
                float acc0_y_8 = 0.0f;
                float _fma_435;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_435) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_8));
                acc0_x_8 = _fma_435;
                float _fma_436;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_436) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_8));
                acc0_y_8 = _fma_436;
                float _fma_437;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_437) : "f"(S[18]), "f"(k_regs[2]), "f"(acc0_x_8));
                acc0_x_8 = _fma_437;
                float _fma_438;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_438) : "f"(S[19]), "f"(k_regs[3]), "f"(acc0_y_8));
                acc0_y_8 = _fma_438;
                float _fma_439;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_439) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_8));
                acc0_x_8 = _fma_439;
                float _fma_440;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_440) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_8));
                acc0_y_8 = _fma_440;
                float _fma_441;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_441) : "f"(S[22]), "f"(k_regs[6]), "f"(acc0_x_8));
                acc0_x_8 = _fma_441;
                float _fma_442;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_442) : "f"(S[23]), "f"(k_regs[7]), "f"(acc0_y_8));
                acc0_y_8 = _fma_442;
                float _fma_443;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_443) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_8));
                acc0_x_8 = _fma_443;
                float _fma_444;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_444) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_8));
                acc0_y_8 = _fma_444;
                float _fma_445;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_445) : "f"(S[26]), "f"(k_regs[10]), "f"(acc0_x_8));
                acc0_x_8 = _fma_445;
                float _fma_446;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_446) : "f"(S[27]), "f"(k_regs[11]), "f"(acc0_y_8));
                acc0_y_8 = _fma_446;
                float _fma_447;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_447) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_8));
                acc0_x_8 = _fma_447;
                float _fma_448;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_448) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_8));
                acc0_y_8 = _fma_448;
                float _fma_449;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_449) : "f"(S[30]), "f"(k_regs[14]), "f"(acc0_x_8));
                acc0_x_8 = _fma_449;
                float _fma_450;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_450) : "f"(S[31]), "f"(k_regs[15]), "f"(acc0_y_8));
                acc0_y_8 = _fma_450;
                float dot_8 = acc0_x_8 + acc0_y_8;
                float _shfl_xor_sync_451 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 1);
                dot_8 = dot_8 + _shfl_xor_sync_451;
                float _shfl_xor_sync_452 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 2);
                dot_8 = dot_8 + _shfl_xor_sync_452;
                float _shfl_xor_sync_453 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 4);
                dot_8 = dot_8 + _shfl_xor_sync_453;
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
                float _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 1);
                dot_8 = dot_8 + _shfl_xor_sync_115;
                float _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 2);
                dot_8 = dot_8 + _shfl_xor_sync_116;
                float _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 4);
                dot_8 = dot_8 + _shfl_xor_sync_117;
#endif
                float vn_9 = (v_regs[1] - dot_8) * beta_2;
#if __CUDA_ARCH__ == 900
                float _fma_454;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_454) : "f"(k_regs[0]), "f"(vn_9), "f"(S[16]));
                S[16] = _fma_454;
                float _fma_455;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_455) : "f"(k_regs[1]), "f"(vn_9), "f"(S[17]));
                S[17] = _fma_455;
                float _fma_456;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_456) : "f"(k_regs[2]), "f"(vn_9), "f"(S[18]));
                S[18] = _fma_456;
                float _fma_457;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_457) : "f"(k_regs[3]), "f"(vn_9), "f"(S[19]));
                S[19] = _fma_457;
                float _fma_458;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_458) : "f"(k_regs[4]), "f"(vn_9), "f"(S[20]));
                S[20] = _fma_458;
                float _fma_459;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_459) : "f"(k_regs[5]), "f"(vn_9), "f"(S[21]));
                S[21] = _fma_459;
                float _fma_460;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_460) : "f"(k_regs[6]), "f"(vn_9), "f"(S[22]));
                S[22] = _fma_460;
                float _fma_461;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_461) : "f"(k_regs[7]), "f"(vn_9), "f"(S[23]));
                S[23] = _fma_461;
                float _fma_462;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_462) : "f"(k_regs[8]), "f"(vn_9), "f"(S[24]));
                S[24] = _fma_462;
                float _fma_463;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_463) : "f"(k_regs[9]), "f"(vn_9), "f"(S[25]));
                S[25] = _fma_463;
                float _fma_464;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_464) : "f"(k_regs[10]), "f"(vn_9), "f"(S[26]));
                S[26] = _fma_464;
                float _fma_465;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_465) : "f"(k_regs[11]), "f"(vn_9), "f"(S[27]));
                S[27] = _fma_465;
                float _fma_466;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_466) : "f"(k_regs[12]), "f"(vn_9), "f"(S[28]));
                S[28] = _fma_466;
                float _fma_467;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_467) : "f"(k_regs[13]), "f"(vn_9), "f"(S[29]));
                S[29] = _fma_467;
                float _fma_468;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_468) : "f"(k_regs[14]), "f"(vn_9), "f"(S[30]));
                S[30] = _fma_468;
                float _fma_469;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_469) : "f"(k_regs[15]), "f"(vn_9), "f"(S[31]));
                S[31] = _fma_469;
                float acc0_x_9 = 0.0f;
                float acc0_y_9 = 0.0f;
                float _fma_470;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_470) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_9));
                acc0_x_9 = _fma_470;
                float _fma_471;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_471) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_9));
                acc0_y_9 = _fma_471;
                float _fma_472;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_472) : "f"(S[34]), "f"(k_regs[2]), "f"(acc0_x_9));
                acc0_x_9 = _fma_472;
                float _fma_473;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_473) : "f"(S[35]), "f"(k_regs[3]), "f"(acc0_y_9));
                acc0_y_9 = _fma_473;
                float _fma_474;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_474) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_9));
                acc0_x_9 = _fma_474;
                float _fma_475;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_475) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_9));
                acc0_y_9 = _fma_475;
                float _fma_476;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_476) : "f"(S[38]), "f"(k_regs[6]), "f"(acc0_x_9));
                acc0_x_9 = _fma_476;
                float _fma_477;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_477) : "f"(S[39]), "f"(k_regs[7]), "f"(acc0_y_9));
                acc0_y_9 = _fma_477;
                float _fma_478;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_478) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_9));
                acc0_x_9 = _fma_478;
                float _fma_479;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_479) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_9));
                acc0_y_9 = _fma_479;
                float _fma_480;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_480) : "f"(S[42]), "f"(k_regs[10]), "f"(acc0_x_9));
                acc0_x_9 = _fma_480;
                float _fma_481;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_481) : "f"(S[43]), "f"(k_regs[11]), "f"(acc0_y_9));
                acc0_y_9 = _fma_481;
                float _fma_482;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_482) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_9));
                acc0_x_9 = _fma_482;
                float _fma_483;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_483) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_9));
                acc0_y_9 = _fma_483;
                float _fma_484;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_484) : "f"(S[46]), "f"(k_regs[14]), "f"(acc0_x_9));
                acc0_x_9 = _fma_484;
                float _fma_485;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_485) : "f"(S[47]), "f"(k_regs[15]), "f"(acc0_y_9));
                acc0_y_9 = _fma_485;
                float dot_9 = acc0_x_9 + acc0_y_9;
                float _shfl_xor_sync_486 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 1);
                dot_9 = dot_9 + _shfl_xor_sync_486;
                float _shfl_xor_sync_487 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 2);
                dot_9 = dot_9 + _shfl_xor_sync_487;
                float _shfl_xor_sync_488 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 4);
                dot_9 = dot_9 + _shfl_xor_sync_488;
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
                float _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 1);
                dot_9 = dot_9 + _shfl_xor_sync_118;
                float _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 2);
                dot_9 = dot_9 + _shfl_xor_sync_119;
                float _shfl_xor_sync_120 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 4);
                dot_9 = dot_9 + _shfl_xor_sync_120;
#endif
                float vn_10 = (v_regs[2] - dot_9) * beta_2;
#if __CUDA_ARCH__ == 900
                float _fma_489;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_489) : "f"(k_regs[0]), "f"(vn_10), "f"(S[32]));
                S[32] = _fma_489;
                float _fma_490;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_490) : "f"(k_regs[1]), "f"(vn_10), "f"(S[33]));
                S[33] = _fma_490;
                float _fma_491;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_491) : "f"(k_regs[2]), "f"(vn_10), "f"(S[34]));
                S[34] = _fma_491;
                float _fma_492;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_492) : "f"(k_regs[3]), "f"(vn_10), "f"(S[35]));
                S[35] = _fma_492;
                float _fma_493;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_493) : "f"(k_regs[4]), "f"(vn_10), "f"(S[36]));
                S[36] = _fma_493;
                float _fma_494;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_494) : "f"(k_regs[5]), "f"(vn_10), "f"(S[37]));
                S[37] = _fma_494;
                float _fma_495;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_495) : "f"(k_regs[6]), "f"(vn_10), "f"(S[38]));
                S[38] = _fma_495;
                float _fma_496;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_496) : "f"(k_regs[7]), "f"(vn_10), "f"(S[39]));
                S[39] = _fma_496;
                float _fma_497;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_497) : "f"(k_regs[8]), "f"(vn_10), "f"(S[40]));
                S[40] = _fma_497;
                float _fma_498;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_498) : "f"(k_regs[9]), "f"(vn_10), "f"(S[41]));
                S[41] = _fma_498;
                float _fma_499;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_499) : "f"(k_regs[10]), "f"(vn_10), "f"(S[42]));
                S[42] = _fma_499;
                float _fma_500;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_500) : "f"(k_regs[11]), "f"(vn_10), "f"(S[43]));
                S[43] = _fma_500;
                float _fma_501;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_501) : "f"(k_regs[12]), "f"(vn_10), "f"(S[44]));
                S[44] = _fma_501;
                float _fma_502;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_502) : "f"(k_regs[13]), "f"(vn_10), "f"(S[45]));
                S[45] = _fma_502;
                float _fma_503;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_503) : "f"(k_regs[14]), "f"(vn_10), "f"(S[46]));
                S[46] = _fma_503;
                float _fma_504;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_504) : "f"(k_regs[15]), "f"(vn_10), "f"(S[47]));
                S[47] = _fma_504;
                float acc0_x_10 = 0.0f;
                float acc0_y_10 = 0.0f;
                float _fma_505;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_505) : "f"(S[48]), "f"(k_regs[0]), "f"(acc0_x_10));
                acc0_x_10 = _fma_505;
                float _fma_506;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_506) : "f"(S[49]), "f"(k_regs[1]), "f"(acc0_y_10));
                acc0_y_10 = _fma_506;
                float _fma_507;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_507) : "f"(S[50]), "f"(k_regs[2]), "f"(acc0_x_10));
                acc0_x_10 = _fma_507;
                float _fma_508;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_508) : "f"(S[51]), "f"(k_regs[3]), "f"(acc0_y_10));
                acc0_y_10 = _fma_508;
                float _fma_509;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_509) : "f"(S[52]), "f"(k_regs[4]), "f"(acc0_x_10));
                acc0_x_10 = _fma_509;
                float _fma_510;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_510) : "f"(S[53]), "f"(k_regs[5]), "f"(acc0_y_10));
                acc0_y_10 = _fma_510;
                float _fma_511;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_511) : "f"(S[54]), "f"(k_regs[6]), "f"(acc0_x_10));
                acc0_x_10 = _fma_511;
                float _fma_512;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_512) : "f"(S[55]), "f"(k_regs[7]), "f"(acc0_y_10));
                acc0_y_10 = _fma_512;
                float _fma_513;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_513) : "f"(S[56]), "f"(k_regs[8]), "f"(acc0_x_10));
                acc0_x_10 = _fma_513;
                float _fma_514;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_514) : "f"(S[57]), "f"(k_regs[9]), "f"(acc0_y_10));
                acc0_y_10 = _fma_514;
                float _fma_515;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_515) : "f"(S[58]), "f"(k_regs[10]), "f"(acc0_x_10));
                acc0_x_10 = _fma_515;
                float _fma_516;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_516) : "f"(S[59]), "f"(k_regs[11]), "f"(acc0_y_10));
                acc0_y_10 = _fma_516;
                float _fma_517;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_517) : "f"(S[60]), "f"(k_regs[12]), "f"(acc0_x_10));
                acc0_x_10 = _fma_517;
                float _fma_518;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_518) : "f"(S[61]), "f"(k_regs[13]), "f"(acc0_y_10));
                acc0_y_10 = _fma_518;
                float _fma_519;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_519) : "f"(S[62]), "f"(k_regs[14]), "f"(acc0_x_10));
                acc0_x_10 = _fma_519;
                float _fma_520;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_520) : "f"(S[63]), "f"(k_regs[15]), "f"(acc0_y_10));
                acc0_y_10 = _fma_520;
                float dot_10 = acc0_x_10 + acc0_y_10;
                float _shfl_xor_sync_521 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 1);
                dot_10 = dot_10 + _shfl_xor_sync_521;
                float _shfl_xor_sync_522 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 2);
                dot_10 = dot_10 + _shfl_xor_sync_522;
                float _shfl_xor_sync_523 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 4);
                dot_10 = dot_10 + _shfl_xor_sync_523;
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
                float _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 1);
                dot_10 = dot_10 + _shfl_xor_sync_121;
                float _shfl_xor_sync_122 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 2);
                dot_10 = dot_10 + _shfl_xor_sync_122;
                float _shfl_xor_sync_123 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 4);
                dot_10 = dot_10 + _shfl_xor_sync_123;
#endif
                float vn_11 = (v_regs[3] - dot_10) * beta_2;
#if __CUDA_ARCH__ == 900
                float _fma_524;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_524) : "f"(k_regs[0]), "f"(vn_11), "f"(S[48]));
                S[48] = _fma_524;
                float _fma_525;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_525) : "f"(k_regs[1]), "f"(vn_11), "f"(S[49]));
                S[49] = _fma_525;
                float _fma_526;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_526) : "f"(k_regs[2]), "f"(vn_11), "f"(S[50]));
                S[50] = _fma_526;
                float _fma_527;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_527) : "f"(k_regs[3]), "f"(vn_11), "f"(S[51]));
                S[51] = _fma_527;
                float _fma_528;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_528) : "f"(k_regs[4]), "f"(vn_11), "f"(S[52]));
                S[52] = _fma_528;
                float _fma_529;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_529) : "f"(k_regs[5]), "f"(vn_11), "f"(S[53]));
                S[53] = _fma_529;
                float _fma_530;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_530) : "f"(k_regs[6]), "f"(vn_11), "f"(S[54]));
                S[54] = _fma_530;
                float _fma_531;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_531) : "f"(k_regs[7]), "f"(vn_11), "f"(S[55]));
                S[55] = _fma_531;
                float _fma_532;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_532) : "f"(k_regs[8]), "f"(vn_11), "f"(S[56]));
                S[56] = _fma_532;
                float _fma_533;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_533) : "f"(k_regs[9]), "f"(vn_11), "f"(S[57]));
                S[57] = _fma_533;
                float _fma_534;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_534) : "f"(k_regs[10]), "f"(vn_11), "f"(S[58]));
                S[58] = _fma_534;
                float _fma_535;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_535) : "f"(k_regs[11]), "f"(vn_11), "f"(S[59]));
                S[59] = _fma_535;
                float _fma_536;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_536) : "f"(k_regs[12]), "f"(vn_11), "f"(S[60]));
                S[60] = _fma_536;
                float _fma_537;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_537) : "f"(k_regs[13]), "f"(vn_11), "f"(S[61]));
                S[61] = _fma_537;
                float _fma_538;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_538) : "f"(k_regs[14]), "f"(vn_11), "f"(S[62]));
                S[62] = _fma_538;
                float _fma_539;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_539) : "f"(k_regs[15]), "f"(vn_11), "f"(S[63]));
                S[63] = _fma_539;
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
                float _fma_540;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_540) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_11));
                acc0_x_11 = _fma_540;
                float _fma_541;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_541) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_11));
                acc0_y_11 = _fma_541;
                float _fma_542;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_542) : "f"(S[2]), "f"(k_regs[2]), "f"(acc0_x_11));
                acc0_x_11 = _fma_542;
                float _fma_543;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_543) : "f"(S[3]), "f"(k_regs[3]), "f"(acc0_y_11));
                acc0_y_11 = _fma_543;
                float _fma_544;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_544) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_11));
                acc0_x_11 = _fma_544;
                float _fma_545;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_545) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_11));
                acc0_y_11 = _fma_545;
                float _fma_546;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_546) : "f"(S[6]), "f"(k_regs[6]), "f"(acc0_x_11));
                acc0_x_11 = _fma_546;
                float _fma_547;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_547) : "f"(S[7]), "f"(k_regs[7]), "f"(acc0_y_11));
                acc0_y_11 = _fma_547;
                float _fma_548;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_548) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_11));
                acc0_x_11 = _fma_548;
                float _fma_549;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_549) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_11));
                acc0_y_11 = _fma_549;
                float _fma_550;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_550) : "f"(S[10]), "f"(k_regs[10]), "f"(acc0_x_11));
                acc0_x_11 = _fma_550;
                float _fma_551;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_551) : "f"(S[11]), "f"(k_regs[11]), "f"(acc0_y_11));
                acc0_y_11 = _fma_551;
                float _fma_552;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_552) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_11));
                acc0_x_11 = _fma_552;
                float _fma_553;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_553) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_11));
                acc0_y_11 = _fma_553;
                float _fma_554;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_554) : "f"(S[14]), "f"(k_regs[14]), "f"(acc0_x_11));
                acc0_x_11 = _fma_554;
                float _fma_555;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_555) : "f"(S[15]), "f"(k_regs[15]), "f"(acc0_y_11));
                acc0_y_11 = _fma_555;
                float dot_11 = acc0_x_11 + acc0_y_11;
                float _shfl_xor_sync_556 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 1);
                dot_11 = dot_11 + _shfl_xor_sync_556;
                float _shfl_xor_sync_557 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 2);
                dot_11 = dot_11 + _shfl_xor_sync_557;
                float _shfl_xor_sync_558 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 4);
                dot_11 = dot_11 + _shfl_xor_sync_558;
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
                float _shfl_xor_sync_124 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 1);
                dot_11 = dot_11 + _shfl_xor_sync_124;
                float _shfl_xor_sync_125 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 2);
                dot_11 = dot_11 + _shfl_xor_sync_125;
                float _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 4);
                dot_11 = dot_11 + _shfl_xor_sync_126;
#endif
                o4[0] = dot_11;
#if __CUDA_ARCH__ == 900
                float acc0_x_12 = 0.0f;
                float acc0_y_12 = 0.0f;
                float _fma_559;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_559) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_12));
                acc0_x_12 = _fma_559;
                float _fma_560;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_560) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_12));
                acc0_y_12 = _fma_560;
                float _fma_561;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_561) : "f"(S[18]), "f"(k_regs[2]), "f"(acc0_x_12));
                acc0_x_12 = _fma_561;
                float _fma_562;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_562) : "f"(S[19]), "f"(k_regs[3]), "f"(acc0_y_12));
                acc0_y_12 = _fma_562;
                float _fma_563;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_563) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_12));
                acc0_x_12 = _fma_563;
                float _fma_564;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_564) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_12));
                acc0_y_12 = _fma_564;
                float _fma_565;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_565) : "f"(S[22]), "f"(k_regs[6]), "f"(acc0_x_12));
                acc0_x_12 = _fma_565;
                float _fma_566;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_566) : "f"(S[23]), "f"(k_regs[7]), "f"(acc0_y_12));
                acc0_y_12 = _fma_566;
                float _fma_567;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_567) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_12));
                acc0_x_12 = _fma_567;
                float _fma_568;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_568) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_12));
                acc0_y_12 = _fma_568;
                float _fma_569;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_569) : "f"(S[26]), "f"(k_regs[10]), "f"(acc0_x_12));
                acc0_x_12 = _fma_569;
                float _fma_570;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_570) : "f"(S[27]), "f"(k_regs[11]), "f"(acc0_y_12));
                acc0_y_12 = _fma_570;
                float _fma_571;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_571) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_12));
                acc0_x_12 = _fma_571;
                float _fma_572;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_572) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_12));
                acc0_y_12 = _fma_572;
                float _fma_573;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_573) : "f"(S[30]), "f"(k_regs[14]), "f"(acc0_x_12));
                acc0_x_12 = _fma_573;
                float _fma_574;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_574) : "f"(S[31]), "f"(k_regs[15]), "f"(acc0_y_12));
                acc0_y_12 = _fma_574;
                float dot_12 = acc0_x_12 + acc0_y_12;
                float _shfl_xor_sync_575 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 1);
                dot_12 = dot_12 + _shfl_xor_sync_575;
                float _shfl_xor_sync_576 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 2);
                dot_12 = dot_12 + _shfl_xor_sync_576;
                float _shfl_xor_sync_577 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 4);
                dot_12 = dot_12 + _shfl_xor_sync_577;
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
                float _shfl_xor_sync_127 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 1);
                dot_12 = dot_12 + _shfl_xor_sync_127;
                float _shfl_xor_sync_128 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 2);
                dot_12 = dot_12 + _shfl_xor_sync_128;
                float _shfl_xor_sync_129 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 4);
                dot_12 = dot_12 + _shfl_xor_sync_129;
#endif
                o4[1] = dot_12;
#if __CUDA_ARCH__ == 900
                float acc0_x_13 = 0.0f;
                float acc0_y_13 = 0.0f;
                float _fma_578;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_578) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_13));
                acc0_x_13 = _fma_578;
                float _fma_579;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_579) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_13));
                acc0_y_13 = _fma_579;
                float _fma_580;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_580) : "f"(S[34]), "f"(k_regs[2]), "f"(acc0_x_13));
                acc0_x_13 = _fma_580;
                float _fma_581;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_581) : "f"(S[35]), "f"(k_regs[3]), "f"(acc0_y_13));
                acc0_y_13 = _fma_581;
                float _fma_582;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_582) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_13));
                acc0_x_13 = _fma_582;
                float _fma_583;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_583) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_13));
                acc0_y_13 = _fma_583;
                float _fma_584;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_584) : "f"(S[38]), "f"(k_regs[6]), "f"(acc0_x_13));
                acc0_x_13 = _fma_584;
                float _fma_585;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_585) : "f"(S[39]), "f"(k_regs[7]), "f"(acc0_y_13));
                acc0_y_13 = _fma_585;
                float _fma_586;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_586) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_13));
                acc0_x_13 = _fma_586;
                float _fma_587;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_587) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_13));
                acc0_y_13 = _fma_587;
                float _fma_588;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_588) : "f"(S[42]), "f"(k_regs[10]), "f"(acc0_x_13));
                acc0_x_13 = _fma_588;
                float _fma_589;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_589) : "f"(S[43]), "f"(k_regs[11]), "f"(acc0_y_13));
                acc0_y_13 = _fma_589;
                float _fma_590;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_590) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_13));
                acc0_x_13 = _fma_590;
                float _fma_591;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_591) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_13));
                acc0_y_13 = _fma_591;
                float _fma_592;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_592) : "f"(S[46]), "f"(k_regs[14]), "f"(acc0_x_13));
                acc0_x_13 = _fma_592;
                float _fma_593;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_593) : "f"(S[47]), "f"(k_regs[15]), "f"(acc0_y_13));
                acc0_y_13 = _fma_593;
                float dot_13 = acc0_x_13 + acc0_y_13;
                float _shfl_xor_sync_594 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 1);
                dot_13 = dot_13 + _shfl_xor_sync_594;
                float _shfl_xor_sync_595 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 2);
                dot_13 = dot_13 + _shfl_xor_sync_595;
                float _shfl_xor_sync_596 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 4);
                dot_13 = dot_13 + _shfl_xor_sync_596;
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
                float _shfl_xor_sync_130 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 1);
                dot_13 = dot_13 + _shfl_xor_sync_130;
                float _shfl_xor_sync_131 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 2);
                dot_13 = dot_13 + _shfl_xor_sync_131;
                float _shfl_xor_sync_132 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 4);
                dot_13 = dot_13 + _shfl_xor_sync_132;
#endif
                o4[2] = dot_13;
#if __CUDA_ARCH__ == 900
                float acc0_x_14 = 0.0f;
                float acc0_y_14 = 0.0f;
                float _fma_597;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_597) : "f"(S[48]), "f"(k_regs[0]), "f"(acc0_x_14));
                acc0_x_14 = _fma_597;
                float _fma_598;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_598) : "f"(S[49]), "f"(k_regs[1]), "f"(acc0_y_14));
                acc0_y_14 = _fma_598;
                float _fma_599;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_599) : "f"(S[50]), "f"(k_regs[2]), "f"(acc0_x_14));
                acc0_x_14 = _fma_599;
                float _fma_600;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_600) : "f"(S[51]), "f"(k_regs[3]), "f"(acc0_y_14));
                acc0_y_14 = _fma_600;
                float _fma_601;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_601) : "f"(S[52]), "f"(k_regs[4]), "f"(acc0_x_14));
                acc0_x_14 = _fma_601;
                float _fma_602;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_602) : "f"(S[53]), "f"(k_regs[5]), "f"(acc0_y_14));
                acc0_y_14 = _fma_602;
                float _fma_603;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_603) : "f"(S[54]), "f"(k_regs[6]), "f"(acc0_x_14));
                acc0_x_14 = _fma_603;
                float _fma_604;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_604) : "f"(S[55]), "f"(k_regs[7]), "f"(acc0_y_14));
                acc0_y_14 = _fma_604;
                float _fma_605;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_605) : "f"(S[56]), "f"(k_regs[8]), "f"(acc0_x_14));
                acc0_x_14 = _fma_605;
                float _fma_606;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_606) : "f"(S[57]), "f"(k_regs[9]), "f"(acc0_y_14));
                acc0_y_14 = _fma_606;
                float _fma_607;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_607) : "f"(S[58]), "f"(k_regs[10]), "f"(acc0_x_14));
                acc0_x_14 = _fma_607;
                float _fma_608;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_608) : "f"(S[59]), "f"(k_regs[11]), "f"(acc0_y_14));
                acc0_y_14 = _fma_608;
                float _fma_609;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_609) : "f"(S[60]), "f"(k_regs[12]), "f"(acc0_x_14));
                acc0_x_14 = _fma_609;
                float _fma_610;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_610) : "f"(S[61]), "f"(k_regs[13]), "f"(acc0_y_14));
                acc0_y_14 = _fma_610;
                float _fma_611;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_611) : "f"(S[62]), "f"(k_regs[14]), "f"(acc0_x_14));
                acc0_x_14 = _fma_611;
                float _fma_612;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_612) : "f"(S[63]), "f"(k_regs[15]), "f"(acc0_y_14));
                acc0_y_14 = _fma_612;
                float dot_14 = acc0_x_14 + acc0_y_14;
                float _shfl_xor_sync_613 = __shfl_xor_sync(0xFFFFFFFF, dot_14, 1);
                dot_14 = dot_14 + _shfl_xor_sync_613;
                float _shfl_xor_sync_614 = __shfl_xor_sync(0xFFFFFFFF, dot_14, 2);
                dot_14 = dot_14 + _shfl_xor_sync_614;
                float _shfl_xor_sync_615 = __shfl_xor_sync(0xFFFFFFFF, dot_14, 4);
                dot_14 = dot_14 + _shfl_xor_sync_615;
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
                float _shfl_xor_sync_133 = __shfl_xor_sync(0xFFFFFFFF, dot_14, 1);
                dot_14 = dot_14 + _shfl_xor_sync_133;
                float _shfl_xor_sync_134 = __shfl_xor_sync(0xFFFFFFFF, dot_14, 2);
                dot_14 = dot_14 + _shfl_xor_sync_134;
                float _shfl_xor_sync_135 = __shfl_xor_sync(0xFFFFFFFF, dot_14, 4);
                dot_14 = dot_14 + _shfl_xor_sync_135;
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
                if (philox_rounds != 0) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 0));
#if __CUDA_ARCH__ == 900
                    float _abs_616;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_616) : "f"(S[0]));
                    unsigned int s = __as_u32(mul_f32_rz_noftz(_abs_616, __uint_as_float(125829120)));
                    unsigned int _min_617 = ((s) < (260046848) ? (s) : (260046848));
                    unsigned int h_0 = _min_617 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag = ((s <= 2139095040) ? h_0 : 32256);
                    float _abs_618;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_618) : "f"(S[1]));
                    unsigned int s_0 = __as_u32(mul_f32_rz_noftz(_abs_618, __uint_as_float(125829120)));
                    unsigned int _min_619 = ((s_0) < (260046848) ? (s_0) : (260046848));
                    unsigned int h_1 = _min_619 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_0 = ((s_0 <= 2139095040) ? h_1 : 32256);
                    uint32_t mags;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags) : "r"(mag), "r"(mag_0));
                    uint32_t signs;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs) : "r"(__as_u32(S[0])), "r"(__as_u32(S[1])));
                    unsigned int word = signs & 2147516416u | mags;
                    h_words[0] = word;
                    float _abs_620;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_620) : "f"(S[2]));
                    unsigned int s_1 = __as_u32(mul_f32_rz_noftz(_abs_620, __uint_as_float(125829120)));
                    unsigned int _min_621 = ((s_1) < (260046848) ? (s_1) : (260046848));
                    unsigned int h_2 = _min_621 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_1 = ((s_1 <= 2139095040) ? h_2 : 32256);
                    float _abs_622;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_622) : "f"(S[3]));
                    unsigned int s_2 = __as_u32(mul_f32_rz_noftz(_abs_622, __uint_as_float(125829120)));
                    unsigned int _min_623 = ((s_2) < (260046848) ? (s_2) : (260046848));
                    unsigned int h_3 = _min_623 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_2 = ((s_2 <= 2139095040) ? h_3 : 32256);
                    uint32_t mags_0;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_0) : "r"(mag_1), "r"(mag_2));
                    uint32_t signs_0;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_0) : "r"(__as_u32(S[2])), "r"(__as_u32(S[3])));
                    unsigned int word_0 = signs_0 & 2147516416u | mags_0;
                    h_words[1] = word_0;
#else
                    uint32_t _cvt_rs_x2_f32_136;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_136) : "f"(S2[0].y), "f"(S2[0].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_136;
                    uint32_t _cvt_rs_x2_f32_137;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_137) : "f"(S2[1].y), "f"(S2[1].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_137;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_624;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_624) : "f"(S[8]));
                    unsigned int s_3 = __as_u32(mul_f32_rz_noftz(_abs_624, __uint_as_float(125829120)));
                    unsigned int _min_625 = ((s_3) < (260046848) ? (s_3) : (260046848));
                    unsigned int h_4 = _min_625 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_3 = ((s_3 <= 2139095040) ? h_4 : 32256);
                    float _abs_626;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_626) : "f"(S[9]));
                    unsigned int s_4 = __as_u32(mul_f32_rz_noftz(_abs_626, __uint_as_float(125829120)));
                    unsigned int _min_627 = ((s_4) < (260046848) ? (s_4) : (260046848));
                    unsigned int h_5 = _min_627 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_4 = ((s_4 <= 2139095040) ? h_5 : 32256);
                    uint32_t mags_1;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_1) : "r"(mag_3), "r"(mag_4));
                    uint32_t signs_1;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_1) : "r"(__as_u32(S[8])), "r"(__as_u32(S[9])));
                    unsigned int word_1 = signs_1 & 2147516416u | mags_1;
                    h_words[0] = word_1;
                    float _abs_628;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_628) : "f"(S[10]));
                    unsigned int s_5 = __as_u32(mul_f32_rz_noftz(_abs_628, __uint_as_float(125829120)));
                    unsigned int _min_629 = ((s_5) < (260046848) ? (s_5) : (260046848));
                    unsigned int h_6 = _min_629 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_5 = ((s_5 <= 2139095040) ? h_6 : 32256);
                    float _abs_630;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_630) : "f"(S[11]));
                    unsigned int s_6 = __as_u32(mul_f32_rz_noftz(_abs_630, __uint_as_float(125829120)));
                    unsigned int _min_631 = ((s_6) < (260046848) ? (s_6) : (260046848));
                    unsigned int h_7 = _min_631 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_6 = ((s_6 <= 2139095040) ? h_7 : 32256);
                    uint32_t mags_2;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_2) : "r"(mag_5), "r"(mag_6));
                    uint32_t signs_2;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_2) : "r"(__as_u32(S[10])), "r"(__as_u32(S[11])));
                    unsigned int word_2 = signs_2 & 2147516416u | mags_2;
                    h_words[1] = word_2;
#else
                    uint32_t _cvt_rs_x2_f32_138;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_138) : "f"(S2[4].y), "f"(S2[4].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_138;
                    uint32_t _cvt_rs_x2_f32_139;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_139) : "f"(S2[5].y), "f"(S2[5].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_139;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 2048));
#if __CUDA_ARCH__ == 900
                    float _abs_632;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_632) : "f"(S[4]));
                    unsigned int s_7 = __as_u32(mul_f32_rz_noftz(_abs_632, __uint_as_float(125829120)));
                    unsigned int _min_633 = ((s_7) < (260046848) ? (s_7) : (260046848));
                    unsigned int h_8 = _min_633 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_7 = ((s_7 <= 2139095040) ? h_8 : 32256);
                    float _abs_634;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_634) : "f"(S[5]));
                    unsigned int s_8 = __as_u32(mul_f32_rz_noftz(_abs_634, __uint_as_float(125829120)));
                    unsigned int _min_635 = ((s_8) < (260046848) ? (s_8) : (260046848));
                    unsigned int h_9 = _min_635 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_8 = ((s_8 <= 2139095040) ? h_9 : 32256);
                    uint32_t mags_3;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_3) : "r"(mag_7), "r"(mag_8));
                    uint32_t signs_3;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_3) : "r"(__as_u32(S[4])), "r"(__as_u32(S[5])));
                    unsigned int word_3 = signs_3 & 2147516416u | mags_3;
                    h_words[0] = word_3;
                    float _abs_636;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_636) : "f"(S[6]));
                    unsigned int s_9 = __as_u32(mul_f32_rz_noftz(_abs_636, __uint_as_float(125829120)));
                    unsigned int _min_637 = ((s_9) < (260046848) ? (s_9) : (260046848));
                    unsigned int h_10 = _min_637 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_9 = ((s_9 <= 2139095040) ? h_10 : 32256);
                    float _abs_638;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_638) : "f"(S[7]));
                    unsigned int s_10 = __as_u32(mul_f32_rz_noftz(_abs_638, __uint_as_float(125829120)));
                    unsigned int _min_639 = ((s_10) < (260046848) ? (s_10) : (260046848));
                    unsigned int h_11 = _min_639 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_10 = ((s_10 <= 2139095040) ? h_11 : 32256);
                    uint32_t mags_4;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_4) : "r"(mag_9), "r"(mag_10));
                    uint32_t signs_4;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_4) : "r"(__as_u32(S[6])), "r"(__as_u32(S[7])));
                    unsigned int word_4 = signs_4 & 2147516416u | mags_4;
                    h_words[1] = word_4;
#else
                    uint32_t _cvt_rs_x2_f32_140;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_140) : "f"(S2[2].y), "f"(S2[2].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_140;
                    uint32_t _cvt_rs_x2_f32_141;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_141) : "f"(S2[3].y), "f"(S2[3].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_141;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_640;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_640) : "f"(S[12]));
                    unsigned int s_11 = __as_u32(mul_f32_rz_noftz(_abs_640, __uint_as_float(125829120)));
                    unsigned int _min_641 = ((s_11) < (260046848) ? (s_11) : (260046848));
                    unsigned int h_12 = _min_641 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_11 = ((s_11 <= 2139095040) ? h_12 : 32256);
                    float _abs_642;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_642) : "f"(S[13]));
                    unsigned int s_12 = __as_u32(mul_f32_rz_noftz(_abs_642, __uint_as_float(125829120)));
                    unsigned int _min_643 = ((s_12) < (260046848) ? (s_12) : (260046848));
                    unsigned int h_13 = _min_643 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_12 = ((s_12 <= 2139095040) ? h_13 : 32256);
                    uint32_t mags_5;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_5) : "r"(mag_11), "r"(mag_12));
                    uint32_t signs_5;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_5) : "r"(__as_u32(S[12])), "r"(__as_u32(S[13])));
                    unsigned int word_5 = signs_5 & 2147516416u | mags_5;
                    h_words[0] = word_5;
                    float _abs_644;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_644) : "f"(S[14]));
                    unsigned int s_13 = __as_u32(mul_f32_rz_noftz(_abs_644, __uint_as_float(125829120)));
                    unsigned int _min_645 = ((s_13) < (260046848) ? (s_13) : (260046848));
                    unsigned int h_14 = _min_645 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_13 = ((s_13 <= 2139095040) ? h_14 : 32256);
                    float _abs_646;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_646) : "f"(S[15]));
                    unsigned int s_14 = __as_u32(mul_f32_rz_noftz(_abs_646, __uint_as_float(125829120)));
                    unsigned int _min_647 = ((s_14) < (260046848) ? (s_14) : (260046848));
                    unsigned int h_15 = _min_647 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_14 = ((s_14 <= 2139095040) ? h_15 : 32256);
                    uint32_t mags_6;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_6) : "r"(mag_13), "r"(mag_14));
                    uint32_t signs_6;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_6) : "r"(__as_u32(S[14])), "r"(__as_u32(S[15])));
                    unsigned int word_6 = signs_6 & 2147516416u | mags_6;
                    h_words[1] = word_6;
#else
                    uint32_t _cvt_rs_x2_f32_142;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_142) : "f"(S2[6].y), "f"(S2[6].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_142;
                    uint32_t _cvt_rs_x2_f32_143;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_143) : "f"(S2[7].y), "f"(S2[7].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_143;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 4096));
#if __CUDA_ARCH__ == 900
                    float _abs_648;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_648) : "f"(S[16]));
                    unsigned int s_15 = __as_u32(mul_f32_rz_noftz(_abs_648, __uint_as_float(125829120)));
                    unsigned int _min_649 = ((s_15) < (260046848) ? (s_15) : (260046848));
                    unsigned int h_16 = _min_649 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_15 = ((s_15 <= 2139095040) ? h_16 : 32256);
                    float _abs_650;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_650) : "f"(S[17]));
                    unsigned int s_16 = __as_u32(mul_f32_rz_noftz(_abs_650, __uint_as_float(125829120)));
                    unsigned int _min_651 = ((s_16) < (260046848) ? (s_16) : (260046848));
                    unsigned int h_17 = _min_651 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_16 = ((s_16 <= 2139095040) ? h_17 : 32256);
                    uint32_t mags_7;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_7) : "r"(mag_15), "r"(mag_16));
                    uint32_t signs_7;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_7) : "r"(__as_u32(S[16])), "r"(__as_u32(S[17])));
                    unsigned int word_7 = signs_7 & 2147516416u | mags_7;
                    h_words[0] = word_7;
                    float _abs_652;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_652) : "f"(S[18]));
                    unsigned int s_17 = __as_u32(mul_f32_rz_noftz(_abs_652, __uint_as_float(125829120)));
                    unsigned int _min_653 = ((s_17) < (260046848) ? (s_17) : (260046848));
                    unsigned int h_18 = _min_653 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_17 = ((s_17 <= 2139095040) ? h_18 : 32256);
                    float _abs_654;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_654) : "f"(S[19]));
                    unsigned int s_18 = __as_u32(mul_f32_rz_noftz(_abs_654, __uint_as_float(125829120)));
                    unsigned int _min_655 = ((s_18) < (260046848) ? (s_18) : (260046848));
                    unsigned int h_19 = _min_655 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_18 = ((s_18 <= 2139095040) ? h_19 : 32256);
                    uint32_t mags_8;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_8) : "r"(mag_17), "r"(mag_18));
                    uint32_t signs_8;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_8) : "r"(__as_u32(S[18])), "r"(__as_u32(S[19])));
                    unsigned int word_8 = signs_8 & 2147516416u | mags_8;
                    h_words[1] = word_8;
#else
                    uint32_t _cvt_rs_x2_f32_144;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_144) : "f"(S2[8].y), "f"(S2[8].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_144;
                    uint32_t _cvt_rs_x2_f32_145;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_145) : "f"(S2[9].y), "f"(S2[9].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_145;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_656;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_656) : "f"(S[24]));
                    unsigned int s_19 = __as_u32(mul_f32_rz_noftz(_abs_656, __uint_as_float(125829120)));
                    unsigned int _min_657 = ((s_19) < (260046848) ? (s_19) : (260046848));
                    unsigned int h_20 = _min_657 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_19 = ((s_19 <= 2139095040) ? h_20 : 32256);
                    float _abs_658;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_658) : "f"(S[25]));
                    unsigned int s_20 = __as_u32(mul_f32_rz_noftz(_abs_658, __uint_as_float(125829120)));
                    unsigned int _min_659 = ((s_20) < (260046848) ? (s_20) : (260046848));
                    unsigned int h_21 = _min_659 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_20 = ((s_20 <= 2139095040) ? h_21 : 32256);
                    uint32_t mags_9;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_9) : "r"(mag_19), "r"(mag_20));
                    uint32_t signs_9;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_9) : "r"(__as_u32(S[24])), "r"(__as_u32(S[25])));
                    unsigned int word_9 = signs_9 & 2147516416u | mags_9;
                    h_words[0] = word_9;
                    float _abs_660;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_660) : "f"(S[26]));
                    unsigned int s_21 = __as_u32(mul_f32_rz_noftz(_abs_660, __uint_as_float(125829120)));
                    unsigned int _min_661 = ((s_21) < (260046848) ? (s_21) : (260046848));
                    unsigned int h_22 = _min_661 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_21 = ((s_21 <= 2139095040) ? h_22 : 32256);
                    float _abs_662;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_662) : "f"(S[27]));
                    unsigned int s_22 = __as_u32(mul_f32_rz_noftz(_abs_662, __uint_as_float(125829120)));
                    unsigned int _min_663 = ((s_22) < (260046848) ? (s_22) : (260046848));
                    unsigned int h_23 = _min_663 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_22 = ((s_22 <= 2139095040) ? h_23 : 32256);
                    uint32_t mags_10;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_10) : "r"(mag_21), "r"(mag_22));
                    uint32_t signs_10;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_10) : "r"(__as_u32(S[26])), "r"(__as_u32(S[27])));
                    unsigned int word_10 = signs_10 & 2147516416u | mags_10;
                    h_words[1] = word_10;
#else
                    uint32_t _cvt_rs_x2_f32_146;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_146) : "f"(S2[12].y), "f"(S2[12].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_146;
                    uint32_t _cvt_rs_x2_f32_147;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_147) : "f"(S2[13].y), "f"(S2[13].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_147;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 6144));
#if __CUDA_ARCH__ == 900
                    float _abs_664;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_664) : "f"(S[20]));
                    unsigned int s_23 = __as_u32(mul_f32_rz_noftz(_abs_664, __uint_as_float(125829120)));
                    unsigned int _min_665 = ((s_23) < (260046848) ? (s_23) : (260046848));
                    unsigned int h_24 = _min_665 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_23 = ((s_23 <= 2139095040) ? h_24 : 32256);
                    float _abs_666;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_666) : "f"(S[21]));
                    unsigned int s_24 = __as_u32(mul_f32_rz_noftz(_abs_666, __uint_as_float(125829120)));
                    unsigned int _min_667 = ((s_24) < (260046848) ? (s_24) : (260046848));
                    unsigned int h_25 = _min_667 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_24 = ((s_24 <= 2139095040) ? h_25 : 32256);
                    uint32_t mags_11;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_11) : "r"(mag_23), "r"(mag_24));
                    uint32_t signs_11;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_11) : "r"(__as_u32(S[20])), "r"(__as_u32(S[21])));
                    unsigned int word_11 = signs_11 & 2147516416u | mags_11;
                    h_words[0] = word_11;
                    float _abs_668;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_668) : "f"(S[22]));
                    unsigned int s_25 = __as_u32(mul_f32_rz_noftz(_abs_668, __uint_as_float(125829120)));
                    unsigned int _min_669 = ((s_25) < (260046848) ? (s_25) : (260046848));
                    unsigned int h_26 = _min_669 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_25 = ((s_25 <= 2139095040) ? h_26 : 32256);
                    float _abs_670;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_670) : "f"(S[23]));
                    unsigned int s_26 = __as_u32(mul_f32_rz_noftz(_abs_670, __uint_as_float(125829120)));
                    unsigned int _min_671 = ((s_26) < (260046848) ? (s_26) : (260046848));
                    unsigned int h_27 = _min_671 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_26 = ((s_26 <= 2139095040) ? h_27 : 32256);
                    uint32_t mags_12;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_12) : "r"(mag_25), "r"(mag_26));
                    uint32_t signs_12;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_12) : "r"(__as_u32(S[22])), "r"(__as_u32(S[23])));
                    unsigned int word_12 = signs_12 & 2147516416u | mags_12;
                    h_words[1] = word_12;
#else
                    uint32_t _cvt_rs_x2_f32_148;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_148) : "f"(S2[10].y), "f"(S2[10].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_148;
                    uint32_t _cvt_rs_x2_f32_149;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_149) : "f"(S2[11].y), "f"(S2[11].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_149;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_672;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_672) : "f"(S[28]));
                    unsigned int s_27 = __as_u32(mul_f32_rz_noftz(_abs_672, __uint_as_float(125829120)));
                    unsigned int _min_673 = ((s_27) < (260046848) ? (s_27) : (260046848));
                    unsigned int h_28 = _min_673 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_27 = ((s_27 <= 2139095040) ? h_28 : 32256);
                    float _abs_674;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_674) : "f"(S[29]));
                    unsigned int s_28 = __as_u32(mul_f32_rz_noftz(_abs_674, __uint_as_float(125829120)));
                    unsigned int _min_675 = ((s_28) < (260046848) ? (s_28) : (260046848));
                    unsigned int h_29 = _min_675 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_28 = ((s_28 <= 2139095040) ? h_29 : 32256);
                    uint32_t mags_13;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_13) : "r"(mag_27), "r"(mag_28));
                    uint32_t signs_13;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_13) : "r"(__as_u32(S[28])), "r"(__as_u32(S[29])));
                    unsigned int word_13 = signs_13 & 2147516416u | mags_13;
                    h_words[0] = word_13;
                    float _abs_676;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_676) : "f"(S[30]));
                    unsigned int s_29 = __as_u32(mul_f32_rz_noftz(_abs_676, __uint_as_float(125829120)));
                    unsigned int _min_677 = ((s_29) < (260046848) ? (s_29) : (260046848));
                    unsigned int h_30 = _min_677 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_29 = ((s_29 <= 2139095040) ? h_30 : 32256);
                    float _abs_678;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_678) : "f"(S[31]));
                    unsigned int s_30 = __as_u32(mul_f32_rz_noftz(_abs_678, __uint_as_float(125829120)));
                    unsigned int _min_679 = ((s_30) < (260046848) ? (s_30) : (260046848));
                    unsigned int h_31 = _min_679 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_30 = ((s_30 <= 2139095040) ? h_31 : 32256);
                    uint32_t mags_14;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_14) : "r"(mag_29), "r"(mag_30));
                    uint32_t signs_14;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_14) : "r"(__as_u32(S[30])), "r"(__as_u32(S[31])));
                    unsigned int word_14 = signs_14 & 2147516416u | mags_14;
                    h_words[1] = word_14;
#else
                    uint32_t _cvt_rs_x2_f32_150;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_150) : "f"(S2[14].y), "f"(S2[14].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_150;
                    uint32_t _cvt_rs_x2_f32_151;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_151) : "f"(S2[15].y), "f"(S2[15].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_151;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 8192));
#if __CUDA_ARCH__ == 900
                    float _abs_680;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_680) : "f"(S[32]));
                    unsigned int s_31 = __as_u32(mul_f32_rz_noftz(_abs_680, __uint_as_float(125829120)));
                    unsigned int _min_681 = ((s_31) < (260046848) ? (s_31) : (260046848));
                    unsigned int h_32 = _min_681 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_31 = ((s_31 <= 2139095040) ? h_32 : 32256);
                    float _abs_682;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_682) : "f"(S[33]));
                    unsigned int s_32 = __as_u32(mul_f32_rz_noftz(_abs_682, __uint_as_float(125829120)));
                    unsigned int _min_683 = ((s_32) < (260046848) ? (s_32) : (260046848));
                    unsigned int h_33 = _min_683 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_32 = ((s_32 <= 2139095040) ? h_33 : 32256);
                    uint32_t mags_15;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_15) : "r"(mag_31), "r"(mag_32));
                    uint32_t signs_15;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_15) : "r"(__as_u32(S[32])), "r"(__as_u32(S[33])));
                    unsigned int word_15 = signs_15 & 2147516416u | mags_15;
                    h_words[0] = word_15;
                    float _abs_684;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_684) : "f"(S[34]));
                    unsigned int s_33 = __as_u32(mul_f32_rz_noftz(_abs_684, __uint_as_float(125829120)));
                    unsigned int _min_685 = ((s_33) < (260046848) ? (s_33) : (260046848));
                    unsigned int h_34 = _min_685 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_33 = ((s_33 <= 2139095040) ? h_34 : 32256);
                    float _abs_686;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_686) : "f"(S[35]));
                    unsigned int s_34 = __as_u32(mul_f32_rz_noftz(_abs_686, __uint_as_float(125829120)));
                    unsigned int _min_687 = ((s_34) < (260046848) ? (s_34) : (260046848));
                    unsigned int h_35 = _min_687 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_34 = ((s_34 <= 2139095040) ? h_35 : 32256);
                    uint32_t mags_16;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_16) : "r"(mag_33), "r"(mag_34));
                    uint32_t signs_16;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_16) : "r"(__as_u32(S[34])), "r"(__as_u32(S[35])));
                    unsigned int word_16 = signs_16 & 2147516416u | mags_16;
                    h_words[1] = word_16;
#else
                    uint32_t _cvt_rs_x2_f32_152;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_152) : "f"(S2[16].y), "f"(S2[16].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_152;
                    uint32_t _cvt_rs_x2_f32_153;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_153) : "f"(S2[17].y), "f"(S2[17].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_153;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_688;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_688) : "f"(S[40]));
                    unsigned int s_35 = __as_u32(mul_f32_rz_noftz(_abs_688, __uint_as_float(125829120)));
                    unsigned int _min_689 = ((s_35) < (260046848) ? (s_35) : (260046848));
                    unsigned int h_36 = _min_689 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_35 = ((s_35 <= 2139095040) ? h_36 : 32256);
                    float _abs_690;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_690) : "f"(S[41]));
                    unsigned int s_36 = __as_u32(mul_f32_rz_noftz(_abs_690, __uint_as_float(125829120)));
                    unsigned int _min_691 = ((s_36) < (260046848) ? (s_36) : (260046848));
                    unsigned int h_37 = _min_691 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_36 = ((s_36 <= 2139095040) ? h_37 : 32256);
                    uint32_t mags_17;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_17) : "r"(mag_35), "r"(mag_36));
                    uint32_t signs_17;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_17) : "r"(__as_u32(S[40])), "r"(__as_u32(S[41])));
                    unsigned int word_17 = signs_17 & 2147516416u | mags_17;
                    h_words[0] = word_17;
                    float _abs_692;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_692) : "f"(S[42]));
                    unsigned int s_37 = __as_u32(mul_f32_rz_noftz(_abs_692, __uint_as_float(125829120)));
                    unsigned int _min_693 = ((s_37) < (260046848) ? (s_37) : (260046848));
                    unsigned int h_38 = _min_693 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_37 = ((s_37 <= 2139095040) ? h_38 : 32256);
                    float _abs_694;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_694) : "f"(S[43]));
                    unsigned int s_38 = __as_u32(mul_f32_rz_noftz(_abs_694, __uint_as_float(125829120)));
                    unsigned int _min_695 = ((s_38) < (260046848) ? (s_38) : (260046848));
                    unsigned int h_39 = _min_695 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_38 = ((s_38 <= 2139095040) ? h_39 : 32256);
                    uint32_t mags_18;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_18) : "r"(mag_37), "r"(mag_38));
                    uint32_t signs_18;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_18) : "r"(__as_u32(S[42])), "r"(__as_u32(S[43])));
                    unsigned int word_18 = signs_18 & 2147516416u | mags_18;
                    h_words[1] = word_18;
#else
                    uint32_t _cvt_rs_x2_f32_154;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_154) : "f"(S2[20].y), "f"(S2[20].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_154;
                    uint32_t _cvt_rs_x2_f32_155;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_155) : "f"(S2[21].y), "f"(S2[21].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_155;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 10240));
#if __CUDA_ARCH__ == 900
                    float _abs_696;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_696) : "f"(S[36]));
                    unsigned int s_39 = __as_u32(mul_f32_rz_noftz(_abs_696, __uint_as_float(125829120)));
                    unsigned int _min_697 = ((s_39) < (260046848) ? (s_39) : (260046848));
                    unsigned int h_40 = _min_697 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_39 = ((s_39 <= 2139095040) ? h_40 : 32256);
                    float _abs_698;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_698) : "f"(S[37]));
                    unsigned int s_40 = __as_u32(mul_f32_rz_noftz(_abs_698, __uint_as_float(125829120)));
                    unsigned int _min_699 = ((s_40) < (260046848) ? (s_40) : (260046848));
                    unsigned int h_41 = _min_699 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_40 = ((s_40 <= 2139095040) ? h_41 : 32256);
                    uint32_t mags_19;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_19) : "r"(mag_39), "r"(mag_40));
                    uint32_t signs_19;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_19) : "r"(__as_u32(S[36])), "r"(__as_u32(S[37])));
                    unsigned int word_19 = signs_19 & 2147516416u | mags_19;
                    h_words[0] = word_19;
                    float _abs_700;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_700) : "f"(S[38]));
                    unsigned int s_41 = __as_u32(mul_f32_rz_noftz(_abs_700, __uint_as_float(125829120)));
                    unsigned int _min_701 = ((s_41) < (260046848) ? (s_41) : (260046848));
                    unsigned int h_42 = _min_701 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_41 = ((s_41 <= 2139095040) ? h_42 : 32256);
                    float _abs_702;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_702) : "f"(S[39]));
                    unsigned int s_42 = __as_u32(mul_f32_rz_noftz(_abs_702, __uint_as_float(125829120)));
                    unsigned int _min_703 = ((s_42) < (260046848) ? (s_42) : (260046848));
                    unsigned int h_43 = _min_703 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_42 = ((s_42 <= 2139095040) ? h_43 : 32256);
                    uint32_t mags_20;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_20) : "r"(mag_41), "r"(mag_42));
                    uint32_t signs_20;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_20) : "r"(__as_u32(S[38])), "r"(__as_u32(S[39])));
                    unsigned int word_20 = signs_20 & 2147516416u | mags_20;
                    h_words[1] = word_20;
#else
                    uint32_t _cvt_rs_x2_f32_156;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_156) : "f"(S2[18].y), "f"(S2[18].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_156;
                    uint32_t _cvt_rs_x2_f32_157;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_157) : "f"(S2[19].y), "f"(S2[19].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_157;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_704;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_704) : "f"(S[44]));
                    unsigned int s_43 = __as_u32(mul_f32_rz_noftz(_abs_704, __uint_as_float(125829120)));
                    unsigned int _min_705 = ((s_43) < (260046848) ? (s_43) : (260046848));
                    unsigned int h_44 = _min_705 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_43 = ((s_43 <= 2139095040) ? h_44 : 32256);
                    float _abs_706;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_706) : "f"(S[45]));
                    unsigned int s_44 = __as_u32(mul_f32_rz_noftz(_abs_706, __uint_as_float(125829120)));
                    unsigned int _min_707 = ((s_44) < (260046848) ? (s_44) : (260046848));
                    unsigned int h_45 = _min_707 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_44 = ((s_44 <= 2139095040) ? h_45 : 32256);
                    uint32_t mags_21;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_21) : "r"(mag_43), "r"(mag_44));
                    uint32_t signs_21;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_21) : "r"(__as_u32(S[44])), "r"(__as_u32(S[45])));
                    unsigned int word_21 = signs_21 & 2147516416u | mags_21;
                    h_words[0] = word_21;
                    float _abs_708;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_708) : "f"(S[46]));
                    unsigned int s_45 = __as_u32(mul_f32_rz_noftz(_abs_708, __uint_as_float(125829120)));
                    unsigned int _min_709 = ((s_45) < (260046848) ? (s_45) : (260046848));
                    unsigned int h_46 = _min_709 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_45 = ((s_45 <= 2139095040) ? h_46 : 32256);
                    float _abs_710;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_710) : "f"(S[47]));
                    unsigned int s_46 = __as_u32(mul_f32_rz_noftz(_abs_710, __uint_as_float(125829120)));
                    unsigned int _min_711 = ((s_46) < (260046848) ? (s_46) : (260046848));
                    unsigned int h_47 = _min_711 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_46 = ((s_46 <= 2139095040) ? h_47 : 32256);
                    uint32_t mags_22;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_22) : "r"(mag_45), "r"(mag_46));
                    uint32_t signs_22;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_22) : "r"(__as_u32(S[46])), "r"(__as_u32(S[47])));
                    unsigned int word_22 = signs_22 & 2147516416u | mags_22;
                    h_words[1] = word_22;
#else
                    uint32_t _cvt_rs_x2_f32_158;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_158) : "f"(S2[22].y), "f"(S2[22].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_158;
                    uint32_t _cvt_rs_x2_f32_159;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_159) : "f"(S2[23].y), "f"(S2[23].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_159;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 12288));
#if __CUDA_ARCH__ == 900
                    float _abs_712;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_712) : "f"(S[48]));
                    unsigned int s_47 = __as_u32(mul_f32_rz_noftz(_abs_712, __uint_as_float(125829120)));
                    unsigned int _min_713 = ((s_47) < (260046848) ? (s_47) : (260046848));
                    unsigned int h_48 = _min_713 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_47 = ((s_47 <= 2139095040) ? h_48 : 32256);
                    float _abs_714;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_714) : "f"(S[49]));
                    unsigned int s_48 = __as_u32(mul_f32_rz_noftz(_abs_714, __uint_as_float(125829120)));
                    unsigned int _min_715 = ((s_48) < (260046848) ? (s_48) : (260046848));
                    unsigned int h_49 = _min_715 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_48 = ((s_48 <= 2139095040) ? h_49 : 32256);
                    uint32_t mags_23;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_23) : "r"(mag_47), "r"(mag_48));
                    uint32_t signs_23;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_23) : "r"(__as_u32(S[48])), "r"(__as_u32(S[49])));
                    unsigned int word_23 = signs_23 & 2147516416u | mags_23;
                    h_words[0] = word_23;
                    float _abs_716;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_716) : "f"(S[50]));
                    unsigned int s_49 = __as_u32(mul_f32_rz_noftz(_abs_716, __uint_as_float(125829120)));
                    unsigned int _min_717 = ((s_49) < (260046848) ? (s_49) : (260046848));
                    unsigned int h_50 = _min_717 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_49 = ((s_49 <= 2139095040) ? h_50 : 32256);
                    float _abs_718;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_718) : "f"(S[51]));
                    unsigned int s_50 = __as_u32(mul_f32_rz_noftz(_abs_718, __uint_as_float(125829120)));
                    unsigned int _min_719 = ((s_50) < (260046848) ? (s_50) : (260046848));
                    unsigned int h_51 = _min_719 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_50 = ((s_50 <= 2139095040) ? h_51 : 32256);
                    uint32_t mags_24;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_24) : "r"(mag_49), "r"(mag_50));
                    uint32_t signs_24;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_24) : "r"(__as_u32(S[50])), "r"(__as_u32(S[51])));
                    unsigned int word_24 = signs_24 & 2147516416u | mags_24;
                    h_words[1] = word_24;
#else
                    uint32_t _cvt_rs_x2_f32_160;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_160) : "f"(S2[24].y), "f"(S2[24].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_160;
                    uint32_t _cvt_rs_x2_f32_161;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_161) : "f"(S2[25].y), "f"(S2[25].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_161;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_720;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_720) : "f"(S[56]));
                    unsigned int s_51 = __as_u32(mul_f32_rz_noftz(_abs_720, __uint_as_float(125829120)));
                    unsigned int _min_721 = ((s_51) < (260046848) ? (s_51) : (260046848));
                    unsigned int h_52 = _min_721 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_51 = ((s_51 <= 2139095040) ? h_52 : 32256);
                    float _abs_722;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_722) : "f"(S[57]));
                    unsigned int s_52 = __as_u32(mul_f32_rz_noftz(_abs_722, __uint_as_float(125829120)));
                    unsigned int _min_723 = ((s_52) < (260046848) ? (s_52) : (260046848));
                    unsigned int h_53 = _min_723 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_52 = ((s_52 <= 2139095040) ? h_53 : 32256);
                    uint32_t mags_25;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_25) : "r"(mag_51), "r"(mag_52));
                    uint32_t signs_25;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_25) : "r"(__as_u32(S[56])), "r"(__as_u32(S[57])));
                    unsigned int word_25 = signs_25 & 2147516416u | mags_25;
                    h_words[0] = word_25;
                    float _abs_724;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_724) : "f"(S[58]));
                    unsigned int s_53 = __as_u32(mul_f32_rz_noftz(_abs_724, __uint_as_float(125829120)));
                    unsigned int _min_725 = ((s_53) < (260046848) ? (s_53) : (260046848));
                    unsigned int h_54 = _min_725 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_53 = ((s_53 <= 2139095040) ? h_54 : 32256);
                    float _abs_726;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_726) : "f"(S[59]));
                    unsigned int s_54 = __as_u32(mul_f32_rz_noftz(_abs_726, __uint_as_float(125829120)));
                    unsigned int _min_727 = ((s_54) < (260046848) ? (s_54) : (260046848));
                    unsigned int h_55 = _min_727 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_54 = ((s_54 <= 2139095040) ? h_55 : 32256);
                    uint32_t mags_26;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_26) : "r"(mag_53), "r"(mag_54));
                    uint32_t signs_26;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_26) : "r"(__as_u32(S[58])), "r"(__as_u32(S[59])));
                    unsigned int word_26 = signs_26 & 2147516416u | mags_26;
                    h_words[1] = word_26;
#else
                    uint32_t _cvt_rs_x2_f32_162;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_162) : "f"(S2[28].y), "f"(S2[28].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_162;
                    uint32_t _cvt_rs_x2_f32_163;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_163) : "f"(S2[29].y), "f"(S2[29].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_163;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 14336));
#if __CUDA_ARCH__ == 900
                    float _abs_728;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_728) : "f"(S[52]));
                    unsigned int s_55 = __as_u32(mul_f32_rz_noftz(_abs_728, __uint_as_float(125829120)));
                    unsigned int _min_729 = ((s_55) < (260046848) ? (s_55) : (260046848));
                    unsigned int h_56 = _min_729 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_55 = ((s_55 <= 2139095040) ? h_56 : 32256);
                    float _abs_730;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_730) : "f"(S[53]));
                    unsigned int s_56 = __as_u32(mul_f32_rz_noftz(_abs_730, __uint_as_float(125829120)));
                    unsigned int _min_731 = ((s_56) < (260046848) ? (s_56) : (260046848));
                    unsigned int h_57 = _min_731 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_56 = ((s_56 <= 2139095040) ? h_57 : 32256);
                    uint32_t mags_27;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_27) : "r"(mag_55), "r"(mag_56));
                    uint32_t signs_27;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_27) : "r"(__as_u32(S[52])), "r"(__as_u32(S[53])));
                    unsigned int word_27 = signs_27 & 2147516416u | mags_27;
                    h_words[0] = word_27;
                    float _abs_732;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_732) : "f"(S[54]));
                    unsigned int s_57 = __as_u32(mul_f32_rz_noftz(_abs_732, __uint_as_float(125829120)));
                    unsigned int _min_733 = ((s_57) < (260046848) ? (s_57) : (260046848));
                    unsigned int h_58 = _min_733 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_57 = ((s_57 <= 2139095040) ? h_58 : 32256);
                    float _abs_734;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_734) : "f"(S[55]));
                    unsigned int s_58 = __as_u32(mul_f32_rz_noftz(_abs_734, __uint_as_float(125829120)));
                    unsigned int _min_735 = ((s_58) < (260046848) ? (s_58) : (260046848));
                    unsigned int h_59 = _min_735 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_58 = ((s_58 <= 2139095040) ? h_59 : 32256);
                    uint32_t mags_28;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_28) : "r"(mag_57), "r"(mag_58));
                    uint32_t signs_28;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_28) : "r"(__as_u32(S[54])), "r"(__as_u32(S[55])));
                    unsigned int word_28 = signs_28 & 2147516416u | mags_28;
                    h_words[1] = word_28;
#else
                    uint32_t _cvt_rs_x2_f32_164;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_164) : "f"(S2[26].y), "f"(S2[26].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_164;
                    uint32_t _cvt_rs_x2_f32_165;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_165) : "f"(S2[27].y), "f"(S2[27].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_165;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_736;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_736) : "f"(S[60]));
                    unsigned int s_59 = __as_u32(mul_f32_rz_noftz(_abs_736, __uint_as_float(125829120)));
                    unsigned int _min_737 = ((s_59) < (260046848) ? (s_59) : (260046848));
                    unsigned int h_60 = _min_737 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_59 = ((s_59 <= 2139095040) ? h_60 : 32256);
                    float _abs_738;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_738) : "f"(S[61]));
                    unsigned int s_60 = __as_u32(mul_f32_rz_noftz(_abs_738, __uint_as_float(125829120)));
                    unsigned int _min_739 = ((s_60) < (260046848) ? (s_60) : (260046848));
                    unsigned int h_61 = _min_739 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_60 = ((s_60 <= 2139095040) ? h_61 : 32256);
                    uint32_t mags_29;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_29) : "r"(mag_59), "r"(mag_60));
                    uint32_t signs_29;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_29) : "r"(__as_u32(S[60])), "r"(__as_u32(S[61])));
                    unsigned int word_29 = signs_29 & 2147516416u | mags_29;
                    h_words[0] = word_29;
                    float _abs_740;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_740) : "f"(S[62]));
                    unsigned int s_61 = __as_u32(mul_f32_rz_noftz(_abs_740, __uint_as_float(125829120)));
                    unsigned int _min_741 = ((s_61) < (260046848) ? (s_61) : (260046848));
                    unsigned int h_62 = _min_741 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_61 = ((s_61 <= 2139095040) ? h_62 : 32256);
                    float _abs_742;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_742) : "f"(S[63]));
                    unsigned int s_62 = __as_u32(mul_f32_rz_noftz(_abs_742, __uint_as_float(125829120)));
                    unsigned int _min_743 = ((s_62) < (260046848) ? (s_62) : (260046848));
                    unsigned int h_63 = _min_743 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_62 = ((s_62 <= 2139095040) ? h_63 : 32256);
                    uint32_t mags_30;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_30) : "r"(mag_61), "r"(mag_62));
                    uint32_t signs_30;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_30) : "r"(__as_u32(S[62])), "r"(__as_u32(S[63])));
                    unsigned int word_30 = signs_30 & 2147516416u | mags_30;
                    h_words[1] = word_30;
#else
                    uint32_t _cvt_rs_x2_f32_166;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_166) : "f"(S2[30].y), "f"(S2[30].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_166;
                    uint32_t _cvt_rs_x2_f32_167;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_167) : "f"(S2[31].y), "f"(S2[31].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_167;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                }
                if (philox_rounds == 0) {
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_744;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_744) : "f"(S[1]), "f"(S[0]));
                    h_words[0] = _cvt_rn_f16x2_f32_744;
                    uint32_t _cvt_rn_f16x2_f32_745;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_745) : "f"(S[3]), "f"(S[2]));
                    h_words[1] = _cvt_rn_f16x2_f32_745;
#else
                    uint32_t _cvt_rn_f16x2_f32_168;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_168) : "f"(S2[0].y), "f"(S2[0].x));
                    h_words[0] = _cvt_rn_f16x2_f32_168;
                    uint32_t _cvt_rn_f16x2_f32_169;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_169) : "f"(S2[1].y), "f"(S2[1].x));
                    h_words[1] = _cvt_rn_f16x2_f32_169;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_746;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_746) : "f"(S[5]), "f"(S[4]));
                    h_words[0] = _cvt_rn_f16x2_f32_746;
                    uint32_t _cvt_rn_f16x2_f32_747;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_747) : "f"(S[7]), "f"(S[6]));
                    h_words[1] = _cvt_rn_f16x2_f32_747;
#else
                    uint32_t _cvt_rn_f16x2_f32_170;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_170) : "f"(S2[2].y), "f"(S2[2].x));
                    h_words[0] = _cvt_rn_f16x2_f32_170;
                    uint32_t _cvt_rn_f16x2_f32_171;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_171) : "f"(S2[3].y), "f"(S2[3].x));
                    h_words[1] = _cvt_rn_f16x2_f32_171;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_748;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_748) : "f"(S[9]), "f"(S[8]));
                    h_words[0] = _cvt_rn_f16x2_f32_748;
                    uint32_t _cvt_rn_f16x2_f32_749;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_749) : "f"(S[11]), "f"(S[10]));
                    h_words[1] = _cvt_rn_f16x2_f32_749;
#else
                    uint32_t _cvt_rn_f16x2_f32_172;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_172) : "f"(S2[4].y), "f"(S2[4].x));
                    h_words[0] = _cvt_rn_f16x2_f32_172;
                    uint32_t _cvt_rn_f16x2_f32_173;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_173) : "f"(S2[5].y), "f"(S2[5].x));
                    h_words[1] = _cvt_rn_f16x2_f32_173;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_750;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_750) : "f"(S[13]), "f"(S[12]));
                    h_words[0] = _cvt_rn_f16x2_f32_750;
                    uint32_t _cvt_rn_f16x2_f32_751;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_751) : "f"(S[15]), "f"(S[14]));
                    h_words[1] = _cvt_rn_f16x2_f32_751;
#else
                    uint32_t _cvt_rn_f16x2_f32_174;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_174) : "f"(S2[6].y), "f"(S2[6].x));
                    h_words[0] = _cvt_rn_f16x2_f32_174;
                    uint32_t _cvt_rn_f16x2_f32_175;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_175) : "f"(S2[7].y), "f"(S2[7].x));
                    h_words[1] = _cvt_rn_f16x2_f32_175;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_752;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_752) : "f"(S[17]), "f"(S[16]));
                    h_words[0] = _cvt_rn_f16x2_f32_752;
                    uint32_t _cvt_rn_f16x2_f32_753;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_753) : "f"(S[19]), "f"(S[18]));
                    h_words[1] = _cvt_rn_f16x2_f32_753;
#else
                    uint32_t _cvt_rn_f16x2_f32_176;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_176) : "f"(S2[8].y), "f"(S2[8].x));
                    h_words[0] = _cvt_rn_f16x2_f32_176;
                    uint32_t _cvt_rn_f16x2_f32_177;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_177) : "f"(S2[9].y), "f"(S2[9].x));
                    h_words[1] = _cvt_rn_f16x2_f32_177;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_754;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_754) : "f"(S[21]), "f"(S[20]));
                    h_words[0] = _cvt_rn_f16x2_f32_754;
                    uint32_t _cvt_rn_f16x2_f32_755;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_755) : "f"(S[23]), "f"(S[22]));
                    h_words[1] = _cvt_rn_f16x2_f32_755;
#else
                    uint32_t _cvt_rn_f16x2_f32_178;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_178) : "f"(S2[10].y), "f"(S2[10].x));
                    h_words[0] = _cvt_rn_f16x2_f32_178;
                    uint32_t _cvt_rn_f16x2_f32_179;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_179) : "f"(S2[11].y), "f"(S2[11].x));
                    h_words[1] = _cvt_rn_f16x2_f32_179;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_756;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_756) : "f"(S[25]), "f"(S[24]));
                    h_words[0] = _cvt_rn_f16x2_f32_756;
                    uint32_t _cvt_rn_f16x2_f32_757;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_757) : "f"(S[27]), "f"(S[26]));
                    h_words[1] = _cvt_rn_f16x2_f32_757;
#else
                    uint32_t _cvt_rn_f16x2_f32_180;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_180) : "f"(S2[12].y), "f"(S2[12].x));
                    h_words[0] = _cvt_rn_f16x2_f32_180;
                    uint32_t _cvt_rn_f16x2_f32_181;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_181) : "f"(S2[13].y), "f"(S2[13].x));
                    h_words[1] = _cvt_rn_f16x2_f32_181;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_758;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_758) : "f"(S[29]), "f"(S[28]));
                    h_words[0] = _cvt_rn_f16x2_f32_758;
                    uint32_t _cvt_rn_f16x2_f32_759;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_759) : "f"(S[31]), "f"(S[30]));
                    h_words[1] = _cvt_rn_f16x2_f32_759;
#else
                    uint32_t _cvt_rn_f16x2_f32_182;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_182) : "f"(S2[14].y), "f"(S2[14].x));
                    h_words[0] = _cvt_rn_f16x2_f32_182;
                    uint32_t _cvt_rn_f16x2_f32_183;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_183) : "f"(S2[15].y), "f"(S2[15].x));
                    h_words[1] = _cvt_rn_f16x2_f32_183;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_760;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_760) : "f"(S[33]), "f"(S[32]));
                    h_words[0] = _cvt_rn_f16x2_f32_760;
                    uint32_t _cvt_rn_f16x2_f32_761;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_761) : "f"(S[35]), "f"(S[34]));
                    h_words[1] = _cvt_rn_f16x2_f32_761;
#else
                    uint32_t _cvt_rn_f16x2_f32_184;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_184) : "f"(S2[16].y), "f"(S2[16].x));
                    h_words[0] = _cvt_rn_f16x2_f32_184;
                    uint32_t _cvt_rn_f16x2_f32_185;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_185) : "f"(S2[17].y), "f"(S2[17].x));
                    h_words[1] = _cvt_rn_f16x2_f32_185;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_762;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_762) : "f"(S[37]), "f"(S[36]));
                    h_words[0] = _cvt_rn_f16x2_f32_762;
                    uint32_t _cvt_rn_f16x2_f32_763;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_763) : "f"(S[39]), "f"(S[38]));
                    h_words[1] = _cvt_rn_f16x2_f32_763;
#else
                    uint32_t _cvt_rn_f16x2_f32_186;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_186) : "f"(S2[18].y), "f"(S2[18].x));
                    h_words[0] = _cvt_rn_f16x2_f32_186;
                    uint32_t _cvt_rn_f16x2_f32_187;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_187) : "f"(S2[19].y), "f"(S2[19].x));
                    h_words[1] = _cvt_rn_f16x2_f32_187;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_764;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_764) : "f"(S[41]), "f"(S[40]));
                    h_words[0] = _cvt_rn_f16x2_f32_764;
                    uint32_t _cvt_rn_f16x2_f32_765;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_765) : "f"(S[43]), "f"(S[42]));
                    h_words[1] = _cvt_rn_f16x2_f32_765;
#else
                    uint32_t _cvt_rn_f16x2_f32_188;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_188) : "f"(S2[20].y), "f"(S2[20].x));
                    h_words[0] = _cvt_rn_f16x2_f32_188;
                    uint32_t _cvt_rn_f16x2_f32_189;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_189) : "f"(S2[21].y), "f"(S2[21].x));
                    h_words[1] = _cvt_rn_f16x2_f32_189;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_766;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_766) : "f"(S[45]), "f"(S[44]));
                    h_words[0] = _cvt_rn_f16x2_f32_766;
                    uint32_t _cvt_rn_f16x2_f32_767;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_767) : "f"(S[47]), "f"(S[46]));
                    h_words[1] = _cvt_rn_f16x2_f32_767;
#else
                    uint32_t _cvt_rn_f16x2_f32_190;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_190) : "f"(S2[22].y), "f"(S2[22].x));
                    h_words[0] = _cvt_rn_f16x2_f32_190;
                    uint32_t _cvt_rn_f16x2_f32_191;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_191) : "f"(S2[23].y), "f"(S2[23].x));
                    h_words[1] = _cvt_rn_f16x2_f32_191;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_768;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_768) : "f"(S[49]), "f"(S[48]));
                    h_words[0] = _cvt_rn_f16x2_f32_768;
                    uint32_t _cvt_rn_f16x2_f32_769;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_769) : "f"(S[51]), "f"(S[50]));
                    h_words[1] = _cvt_rn_f16x2_f32_769;
#else
                    uint32_t _cvt_rn_f16x2_f32_192;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_192) : "f"(S2[24].y), "f"(S2[24].x));
                    h_words[0] = _cvt_rn_f16x2_f32_192;
                    uint32_t _cvt_rn_f16x2_f32_193;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_193) : "f"(S2[25].y), "f"(S2[25].x));
                    h_words[1] = _cvt_rn_f16x2_f32_193;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_770;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_770) : "f"(S[53]), "f"(S[52]));
                    h_words[0] = _cvt_rn_f16x2_f32_770;
                    uint32_t _cvt_rn_f16x2_f32_771;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_771) : "f"(S[55]), "f"(S[54]));
                    h_words[1] = _cvt_rn_f16x2_f32_771;
#else
                    uint32_t _cvt_rn_f16x2_f32_194;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_194) : "f"(S2[26].y), "f"(S2[26].x));
                    h_words[0] = _cvt_rn_f16x2_f32_194;
                    uint32_t _cvt_rn_f16x2_f32_195;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_195) : "f"(S2[27].y), "f"(S2[27].x));
                    h_words[1] = _cvt_rn_f16x2_f32_195;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_772;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_772) : "f"(S[57]), "f"(S[56]));
                    h_words[0] = _cvt_rn_f16x2_f32_772;
                    uint32_t _cvt_rn_f16x2_f32_773;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_773) : "f"(S[59]), "f"(S[58]));
                    h_words[1] = _cvt_rn_f16x2_f32_773;
#else
                    uint32_t _cvt_rn_f16x2_f32_196;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_196) : "f"(S2[28].y), "f"(S2[28].x));
                    h_words[0] = _cvt_rn_f16x2_f32_196;
                    uint32_t _cvt_rn_f16x2_f32_197;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_197) : "f"(S2[29].y), "f"(S2[29].x));
                    h_words[1] = _cvt_rn_f16x2_f32_197;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_774;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_774) : "f"(S[61]), "f"(S[60]));
                    h_words[0] = _cvt_rn_f16x2_f32_774;
                    uint32_t _cvt_rn_f16x2_f32_775;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_775) : "f"(S[63]), "f"(S[62]));
                    h_words[1] = _cvt_rn_f16x2_f32_775;
#else
                    uint32_t _cvt_rn_f16x2_f32_198;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_198) : "f"(S2[30].y), "f"(S2[30].x));
                    h_words[0] = _cvt_rn_f16x2_f32_198;
                    uint32_t _cvt_rn_f16x2_f32_199;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_199) : "f"(S2[31].y), "f"(S2[31].x));
                    h_words[1] = _cvt_rn_f16x2_f32_199;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                }
            }
        }
        if (use_ssm_state_indices == 0) {
            #pragma unroll 1
            for (int t_1 = 0; t_1 < 16; t_1++) {
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
                float _fma_776;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_776) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_15));
                acc0_x_15 = _fma_776;
                float _fma_777;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_777) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_15));
                acc0_y_15 = _fma_777;
                float _fma_778;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_778) : "f"(S[2]), "f"(k_regs[2]), "f"(acc0_x_15));
                acc0_x_15 = _fma_778;
                float _fma_779;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_779) : "f"(S[3]), "f"(k_regs[3]), "f"(acc0_y_15));
                acc0_y_15 = _fma_779;
                float _fma_780;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_780) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_15));
                acc0_x_15 = _fma_780;
                float _fma_781;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_781) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_15));
                acc0_y_15 = _fma_781;
                float _fma_782;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_782) : "f"(S[6]), "f"(k_regs[6]), "f"(acc0_x_15));
                acc0_x_15 = _fma_782;
                float _fma_783;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_783) : "f"(S[7]), "f"(k_regs[7]), "f"(acc0_y_15));
                acc0_y_15 = _fma_783;
                float _fma_784;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_784) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_15));
                acc0_x_15 = _fma_784;
                float _fma_785;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_785) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_15));
                acc0_y_15 = _fma_785;
                float _fma_786;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_786) : "f"(S[10]), "f"(k_regs[10]), "f"(acc0_x_15));
                acc0_x_15 = _fma_786;
                float _fma_787;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_787) : "f"(S[11]), "f"(k_regs[11]), "f"(acc0_y_15));
                acc0_y_15 = _fma_787;
                float _fma_788;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_788) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_15));
                acc0_x_15 = _fma_788;
                float _fma_789;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_789) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_15));
                acc0_y_15 = _fma_789;
                float _fma_790;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_790) : "f"(S[14]), "f"(k_regs[14]), "f"(acc0_x_15));
                acc0_x_15 = _fma_790;
                float _fma_791;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_791) : "f"(S[15]), "f"(k_regs[15]), "f"(acc0_y_15));
                acc0_y_15 = _fma_791;
                float dot_15 = acc0_x_15 + acc0_y_15;
                float _shfl_xor_sync_792 = __shfl_xor_sync(0xFFFFFFFF, dot_15, 1);
                dot_15 = dot_15 + _shfl_xor_sync_792;
                float _shfl_xor_sync_793 = __shfl_xor_sync(0xFFFFFFFF, dot_15, 2);
                dot_15 = dot_15 + _shfl_xor_sync_793;
                float _shfl_xor_sync_794 = __shfl_xor_sync(0xFFFFFFFF, dot_15, 4);
                dot_15 = dot_15 + _shfl_xor_sync_794;
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
                float _shfl_xor_sync_200 = __shfl_xor_sync(0xFFFFFFFF, dot_15, 1);
                dot_15 = dot_15 + _shfl_xor_sync_200;
                float _shfl_xor_sync_201 = __shfl_xor_sync(0xFFFFFFFF, dot_15, 2);
                dot_15 = dot_15 + _shfl_xor_sync_201;
                float _shfl_xor_sync_202 = __shfl_xor_sync(0xFFFFFFFF, dot_15, 4);
                dot_15 = dot_15 + _shfl_xor_sync_202;
#endif
                float vn_12 = (v_regs[0] - dot_15) * beta_3;
#if __CUDA_ARCH__ == 900
                float _fma_795;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_795) : "f"(k_regs[0]), "f"(vn_12), "f"(S[0]));
                S[0] = _fma_795;
                float _fma_796;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_796) : "f"(k_regs[1]), "f"(vn_12), "f"(S[1]));
                S[1] = _fma_796;
                float _fma_797;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_797) : "f"(k_regs[2]), "f"(vn_12), "f"(S[2]));
                S[2] = _fma_797;
                float _fma_798;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_798) : "f"(k_regs[3]), "f"(vn_12), "f"(S[3]));
                S[3] = _fma_798;
                float _fma_799;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_799) : "f"(k_regs[4]), "f"(vn_12), "f"(S[4]));
                S[4] = _fma_799;
                float _fma_800;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_800) : "f"(k_regs[5]), "f"(vn_12), "f"(S[5]));
                S[5] = _fma_800;
                float _fma_801;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_801) : "f"(k_regs[6]), "f"(vn_12), "f"(S[6]));
                S[6] = _fma_801;
                float _fma_802;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_802) : "f"(k_regs[7]), "f"(vn_12), "f"(S[7]));
                S[7] = _fma_802;
                float _fma_803;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_803) : "f"(k_regs[8]), "f"(vn_12), "f"(S[8]));
                S[8] = _fma_803;
                float _fma_804;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_804) : "f"(k_regs[9]), "f"(vn_12), "f"(S[9]));
                S[9] = _fma_804;
                float _fma_805;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_805) : "f"(k_regs[10]), "f"(vn_12), "f"(S[10]));
                S[10] = _fma_805;
                float _fma_806;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_806) : "f"(k_regs[11]), "f"(vn_12), "f"(S[11]));
                S[11] = _fma_806;
                float _fma_807;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_807) : "f"(k_regs[12]), "f"(vn_12), "f"(S[12]));
                S[12] = _fma_807;
                float _fma_808;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_808) : "f"(k_regs[13]), "f"(vn_12), "f"(S[13]));
                S[13] = _fma_808;
                float _fma_809;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_809) : "f"(k_regs[14]), "f"(vn_12), "f"(S[14]));
                S[14] = _fma_809;
                float _fma_810;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_810) : "f"(k_regs[15]), "f"(vn_12), "f"(S[15]));
                S[15] = _fma_810;
                float acc0_x_15_1 = 0.0f;
                float acc0_y_15_1 = 0.0f;
                float _fma_811;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_811) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_15_1));
                acc0_x_15_1 = _fma_811;
                float _fma_812;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_812) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_15_1));
                acc0_y_15_1 = _fma_812;
                float _fma_813;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_813) : "f"(S[18]), "f"(k_regs[2]), "f"(acc0_x_15_1));
                acc0_x_15_1 = _fma_813;
                float _fma_814;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_814) : "f"(S[19]), "f"(k_regs[3]), "f"(acc0_y_15_1));
                acc0_y_15_1 = _fma_814;
                float _fma_815;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_815) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_15_1));
                acc0_x_15_1 = _fma_815;
                float _fma_816;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_816) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_15_1));
                acc0_y_15_1 = _fma_816;
                float _fma_817;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_817) : "f"(S[22]), "f"(k_regs[6]), "f"(acc0_x_15_1));
                acc0_x_15_1 = _fma_817;
                float _fma_818;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_818) : "f"(S[23]), "f"(k_regs[7]), "f"(acc0_y_15_1));
                acc0_y_15_1 = _fma_818;
                float _fma_819;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_819) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_15_1));
                acc0_x_15_1 = _fma_819;
                float _fma_820;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_820) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_15_1));
                acc0_y_15_1 = _fma_820;
                float _fma_821;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_821) : "f"(S[26]), "f"(k_regs[10]), "f"(acc0_x_15_1));
                acc0_x_15_1 = _fma_821;
                float _fma_822;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_822) : "f"(S[27]), "f"(k_regs[11]), "f"(acc0_y_15_1));
                acc0_y_15_1 = _fma_822;
                float _fma_823;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_823) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_15_1));
                acc0_x_15_1 = _fma_823;
                float _fma_824;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_824) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_15_1));
                acc0_y_15_1 = _fma_824;
                float _fma_825;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_825) : "f"(S[30]), "f"(k_regs[14]), "f"(acc0_x_15_1));
                acc0_x_15_1 = _fma_825;
                float _fma_826;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_826) : "f"(S[31]), "f"(k_regs[15]), "f"(acc0_y_15_1));
                acc0_y_15_1 = _fma_826;
                float dot_15_1 = acc0_x_15_1 + acc0_y_15_1;
                float _shfl_xor_sync_827 = __shfl_xor_sync(0xFFFFFFFF, dot_15_1, 1);
                dot_15_1 = dot_15_1 + _shfl_xor_sync_827;
                float _shfl_xor_sync_828 = __shfl_xor_sync(0xFFFFFFFF, dot_15_1, 2);
                dot_15_1 = dot_15_1 + _shfl_xor_sync_828;
                float _shfl_xor_sync_829 = __shfl_xor_sync(0xFFFFFFFF, dot_15_1, 4);
                dot_15_1 = dot_15_1 + _shfl_xor_sync_829;
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
                float _shfl_xor_sync_203 = __shfl_xor_sync(0xFFFFFFFF, dot_15_1, 1);
                dot_15_1 = dot_15_1 + _shfl_xor_sync_203;
                float _shfl_xor_sync_204 = __shfl_xor_sync(0xFFFFFFFF, dot_15_1, 2);
                dot_15_1 = dot_15_1 + _shfl_xor_sync_204;
                float _shfl_xor_sync_205 = __shfl_xor_sync(0xFFFFFFFF, dot_15_1, 4);
                dot_15_1 = dot_15_1 + _shfl_xor_sync_205;
#endif
                float vn_13 = (v_regs[1] - dot_15_1) * beta_3;
#if __CUDA_ARCH__ == 900
                float _fma_830;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_830) : "f"(k_regs[0]), "f"(vn_13), "f"(S[16]));
                S[16] = _fma_830;
                float _fma_831;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_831) : "f"(k_regs[1]), "f"(vn_13), "f"(S[17]));
                S[17] = _fma_831;
                float _fma_832;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_832) : "f"(k_regs[2]), "f"(vn_13), "f"(S[18]));
                S[18] = _fma_832;
                float _fma_833;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_833) : "f"(k_regs[3]), "f"(vn_13), "f"(S[19]));
                S[19] = _fma_833;
                float _fma_834;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_834) : "f"(k_regs[4]), "f"(vn_13), "f"(S[20]));
                S[20] = _fma_834;
                float _fma_835;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_835) : "f"(k_regs[5]), "f"(vn_13), "f"(S[21]));
                S[21] = _fma_835;
                float _fma_836;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_836) : "f"(k_regs[6]), "f"(vn_13), "f"(S[22]));
                S[22] = _fma_836;
                float _fma_837;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_837) : "f"(k_regs[7]), "f"(vn_13), "f"(S[23]));
                S[23] = _fma_837;
                float _fma_838;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_838) : "f"(k_regs[8]), "f"(vn_13), "f"(S[24]));
                S[24] = _fma_838;
                float _fma_839;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_839) : "f"(k_regs[9]), "f"(vn_13), "f"(S[25]));
                S[25] = _fma_839;
                float _fma_840;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_840) : "f"(k_regs[10]), "f"(vn_13), "f"(S[26]));
                S[26] = _fma_840;
                float _fma_841;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_841) : "f"(k_regs[11]), "f"(vn_13), "f"(S[27]));
                S[27] = _fma_841;
                float _fma_842;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_842) : "f"(k_regs[12]), "f"(vn_13), "f"(S[28]));
                S[28] = _fma_842;
                float _fma_843;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_843) : "f"(k_regs[13]), "f"(vn_13), "f"(S[29]));
                S[29] = _fma_843;
                float _fma_844;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_844) : "f"(k_regs[14]), "f"(vn_13), "f"(S[30]));
                S[30] = _fma_844;
                float _fma_845;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_845) : "f"(k_regs[15]), "f"(vn_13), "f"(S[31]));
                S[31] = _fma_845;
                float acc0_x_16 = 0.0f;
                float acc0_y_16 = 0.0f;
                float _fma_846;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_846) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_16));
                acc0_x_16 = _fma_846;
                float _fma_847;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_847) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_16));
                acc0_y_16 = _fma_847;
                float _fma_848;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_848) : "f"(S[34]), "f"(k_regs[2]), "f"(acc0_x_16));
                acc0_x_16 = _fma_848;
                float _fma_849;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_849) : "f"(S[35]), "f"(k_regs[3]), "f"(acc0_y_16));
                acc0_y_16 = _fma_849;
                float _fma_850;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_850) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_16));
                acc0_x_16 = _fma_850;
                float _fma_851;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_851) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_16));
                acc0_y_16 = _fma_851;
                float _fma_852;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_852) : "f"(S[38]), "f"(k_regs[6]), "f"(acc0_x_16));
                acc0_x_16 = _fma_852;
                float _fma_853;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_853) : "f"(S[39]), "f"(k_regs[7]), "f"(acc0_y_16));
                acc0_y_16 = _fma_853;
                float _fma_854;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_854) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_16));
                acc0_x_16 = _fma_854;
                float _fma_855;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_855) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_16));
                acc0_y_16 = _fma_855;
                float _fma_856;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_856) : "f"(S[42]), "f"(k_regs[10]), "f"(acc0_x_16));
                acc0_x_16 = _fma_856;
                float _fma_857;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_857) : "f"(S[43]), "f"(k_regs[11]), "f"(acc0_y_16));
                acc0_y_16 = _fma_857;
                float _fma_858;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_858) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_16));
                acc0_x_16 = _fma_858;
                float _fma_859;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_859) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_16));
                acc0_y_16 = _fma_859;
                float _fma_860;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_860) : "f"(S[46]), "f"(k_regs[14]), "f"(acc0_x_16));
                acc0_x_16 = _fma_860;
                float _fma_861;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_861) : "f"(S[47]), "f"(k_regs[15]), "f"(acc0_y_16));
                acc0_y_16 = _fma_861;
                float dot_16 = acc0_x_16 + acc0_y_16;
                float _shfl_xor_sync_862 = __shfl_xor_sync(0xFFFFFFFF, dot_16, 1);
                dot_16 = dot_16 + _shfl_xor_sync_862;
                float _shfl_xor_sync_863 = __shfl_xor_sync(0xFFFFFFFF, dot_16, 2);
                dot_16 = dot_16 + _shfl_xor_sync_863;
                float _shfl_xor_sync_864 = __shfl_xor_sync(0xFFFFFFFF, dot_16, 4);
                dot_16 = dot_16 + _shfl_xor_sync_864;
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
                float _shfl_xor_sync_206 = __shfl_xor_sync(0xFFFFFFFF, dot_16, 1);
                dot_16 = dot_16 + _shfl_xor_sync_206;
                float _shfl_xor_sync_207 = __shfl_xor_sync(0xFFFFFFFF, dot_16, 2);
                dot_16 = dot_16 + _shfl_xor_sync_207;
                float _shfl_xor_sync_208 = __shfl_xor_sync(0xFFFFFFFF, dot_16, 4);
                dot_16 = dot_16 + _shfl_xor_sync_208;
#endif
                float vn_14 = (v_regs[2] - dot_16) * beta_3;
#if __CUDA_ARCH__ == 900
                float _fma_865;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_865) : "f"(k_regs[0]), "f"(vn_14), "f"(S[32]));
                S[32] = _fma_865;
                float _fma_866;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_866) : "f"(k_regs[1]), "f"(vn_14), "f"(S[33]));
                S[33] = _fma_866;
                float _fma_867;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_867) : "f"(k_regs[2]), "f"(vn_14), "f"(S[34]));
                S[34] = _fma_867;
                float _fma_868;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_868) : "f"(k_regs[3]), "f"(vn_14), "f"(S[35]));
                S[35] = _fma_868;
                float _fma_869;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_869) : "f"(k_regs[4]), "f"(vn_14), "f"(S[36]));
                S[36] = _fma_869;
                float _fma_870;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_870) : "f"(k_regs[5]), "f"(vn_14), "f"(S[37]));
                S[37] = _fma_870;
                float _fma_871;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_871) : "f"(k_regs[6]), "f"(vn_14), "f"(S[38]));
                S[38] = _fma_871;
                float _fma_872;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_872) : "f"(k_regs[7]), "f"(vn_14), "f"(S[39]));
                S[39] = _fma_872;
                float _fma_873;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_873) : "f"(k_regs[8]), "f"(vn_14), "f"(S[40]));
                S[40] = _fma_873;
                float _fma_874;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_874) : "f"(k_regs[9]), "f"(vn_14), "f"(S[41]));
                S[41] = _fma_874;
                float _fma_875;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_875) : "f"(k_regs[10]), "f"(vn_14), "f"(S[42]));
                S[42] = _fma_875;
                float _fma_876;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_876) : "f"(k_regs[11]), "f"(vn_14), "f"(S[43]));
                S[43] = _fma_876;
                float _fma_877;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_877) : "f"(k_regs[12]), "f"(vn_14), "f"(S[44]));
                S[44] = _fma_877;
                float _fma_878;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_878) : "f"(k_regs[13]), "f"(vn_14), "f"(S[45]));
                S[45] = _fma_878;
                float _fma_879;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_879) : "f"(k_regs[14]), "f"(vn_14), "f"(S[46]));
                S[46] = _fma_879;
                float _fma_880;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_880) : "f"(k_regs[15]), "f"(vn_14), "f"(S[47]));
                S[47] = _fma_880;
                float acc0_x_17 = 0.0f;
                float acc0_y_17 = 0.0f;
                float _fma_881;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_881) : "f"(S[48]), "f"(k_regs[0]), "f"(acc0_x_17));
                acc0_x_17 = _fma_881;
                float _fma_882;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_882) : "f"(S[49]), "f"(k_regs[1]), "f"(acc0_y_17));
                acc0_y_17 = _fma_882;
                float _fma_883;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_883) : "f"(S[50]), "f"(k_regs[2]), "f"(acc0_x_17));
                acc0_x_17 = _fma_883;
                float _fma_884;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_884) : "f"(S[51]), "f"(k_regs[3]), "f"(acc0_y_17));
                acc0_y_17 = _fma_884;
                float _fma_885;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_885) : "f"(S[52]), "f"(k_regs[4]), "f"(acc0_x_17));
                acc0_x_17 = _fma_885;
                float _fma_886;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_886) : "f"(S[53]), "f"(k_regs[5]), "f"(acc0_y_17));
                acc0_y_17 = _fma_886;
                float _fma_887;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_887) : "f"(S[54]), "f"(k_regs[6]), "f"(acc0_x_17));
                acc0_x_17 = _fma_887;
                float _fma_888;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_888) : "f"(S[55]), "f"(k_regs[7]), "f"(acc0_y_17));
                acc0_y_17 = _fma_888;
                float _fma_889;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_889) : "f"(S[56]), "f"(k_regs[8]), "f"(acc0_x_17));
                acc0_x_17 = _fma_889;
                float _fma_890;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_890) : "f"(S[57]), "f"(k_regs[9]), "f"(acc0_y_17));
                acc0_y_17 = _fma_890;
                float _fma_891;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_891) : "f"(S[58]), "f"(k_regs[10]), "f"(acc0_x_17));
                acc0_x_17 = _fma_891;
                float _fma_892;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_892) : "f"(S[59]), "f"(k_regs[11]), "f"(acc0_y_17));
                acc0_y_17 = _fma_892;
                float _fma_893;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_893) : "f"(S[60]), "f"(k_regs[12]), "f"(acc0_x_17));
                acc0_x_17 = _fma_893;
                float _fma_894;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_894) : "f"(S[61]), "f"(k_regs[13]), "f"(acc0_y_17));
                acc0_y_17 = _fma_894;
                float _fma_895;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_895) : "f"(S[62]), "f"(k_regs[14]), "f"(acc0_x_17));
                acc0_x_17 = _fma_895;
                float _fma_896;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_896) : "f"(S[63]), "f"(k_regs[15]), "f"(acc0_y_17));
                acc0_y_17 = _fma_896;
                float dot_17 = acc0_x_17 + acc0_y_17;
                float _shfl_xor_sync_897 = __shfl_xor_sync(0xFFFFFFFF, dot_17, 1);
                dot_17 = dot_17 + _shfl_xor_sync_897;
                float _shfl_xor_sync_898 = __shfl_xor_sync(0xFFFFFFFF, dot_17, 2);
                dot_17 = dot_17 + _shfl_xor_sync_898;
                float _shfl_xor_sync_899 = __shfl_xor_sync(0xFFFFFFFF, dot_17, 4);
                dot_17 = dot_17 + _shfl_xor_sync_899;
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
                float _shfl_xor_sync_209 = __shfl_xor_sync(0xFFFFFFFF, dot_17, 1);
                dot_17 = dot_17 + _shfl_xor_sync_209;
                float _shfl_xor_sync_210 = __shfl_xor_sync(0xFFFFFFFF, dot_17, 2);
                dot_17 = dot_17 + _shfl_xor_sync_210;
                float _shfl_xor_sync_211 = __shfl_xor_sync(0xFFFFFFFF, dot_17, 4);
                dot_17 = dot_17 + _shfl_xor_sync_211;
#endif
                float vn_15 = (v_regs[3] - dot_17) * beta_3;
#if __CUDA_ARCH__ == 900
                float _fma_900;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_900) : "f"(k_regs[0]), "f"(vn_15), "f"(S[48]));
                S[48] = _fma_900;
                float _fma_901;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_901) : "f"(k_regs[1]), "f"(vn_15), "f"(S[49]));
                S[49] = _fma_901;
                float _fma_902;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_902) : "f"(k_regs[2]), "f"(vn_15), "f"(S[50]));
                S[50] = _fma_902;
                float _fma_903;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_903) : "f"(k_regs[3]), "f"(vn_15), "f"(S[51]));
                S[51] = _fma_903;
                float _fma_904;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_904) : "f"(k_regs[4]), "f"(vn_15), "f"(S[52]));
                S[52] = _fma_904;
                float _fma_905;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_905) : "f"(k_regs[5]), "f"(vn_15), "f"(S[53]));
                S[53] = _fma_905;
                float _fma_906;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_906) : "f"(k_regs[6]), "f"(vn_15), "f"(S[54]));
                S[54] = _fma_906;
                float _fma_907;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_907) : "f"(k_regs[7]), "f"(vn_15), "f"(S[55]));
                S[55] = _fma_907;
                float _fma_908;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_908) : "f"(k_regs[8]), "f"(vn_15), "f"(S[56]));
                S[56] = _fma_908;
                float _fma_909;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_909) : "f"(k_regs[9]), "f"(vn_15), "f"(S[57]));
                S[57] = _fma_909;
                float _fma_910;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_910) : "f"(k_regs[10]), "f"(vn_15), "f"(S[58]));
                S[58] = _fma_910;
                float _fma_911;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_911) : "f"(k_regs[11]), "f"(vn_15), "f"(S[59]));
                S[59] = _fma_911;
                float _fma_912;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_912) : "f"(k_regs[12]), "f"(vn_15), "f"(S[60]));
                S[60] = _fma_912;
                float _fma_913;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_913) : "f"(k_regs[13]), "f"(vn_15), "f"(S[61]));
                S[61] = _fma_913;
                float _fma_914;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_914) : "f"(k_regs[14]), "f"(vn_15), "f"(S[62]));
                S[62] = _fma_914;
                float _fma_915;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_915) : "f"(k_regs[15]), "f"(vn_15), "f"(S[63]));
                S[63] = _fma_915;
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
                float _fma_916;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_916) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_18));
                acc0_x_18 = _fma_916;
                float _fma_917;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_917) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_18));
                acc0_y_18 = _fma_917;
                float _fma_918;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_918) : "f"(S[2]), "f"(k_regs[2]), "f"(acc0_x_18));
                acc0_x_18 = _fma_918;
                float _fma_919;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_919) : "f"(S[3]), "f"(k_regs[3]), "f"(acc0_y_18));
                acc0_y_18 = _fma_919;
                float _fma_920;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_920) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_18));
                acc0_x_18 = _fma_920;
                float _fma_921;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_921) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_18));
                acc0_y_18 = _fma_921;
                float _fma_922;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_922) : "f"(S[6]), "f"(k_regs[6]), "f"(acc0_x_18));
                acc0_x_18 = _fma_922;
                float _fma_923;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_923) : "f"(S[7]), "f"(k_regs[7]), "f"(acc0_y_18));
                acc0_y_18 = _fma_923;
                float _fma_924;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_924) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_18));
                acc0_x_18 = _fma_924;
                float _fma_925;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_925) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_18));
                acc0_y_18 = _fma_925;
                float _fma_926;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_926) : "f"(S[10]), "f"(k_regs[10]), "f"(acc0_x_18));
                acc0_x_18 = _fma_926;
                float _fma_927;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_927) : "f"(S[11]), "f"(k_regs[11]), "f"(acc0_y_18));
                acc0_y_18 = _fma_927;
                float _fma_928;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_928) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_18));
                acc0_x_18 = _fma_928;
                float _fma_929;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_929) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_18));
                acc0_y_18 = _fma_929;
                float _fma_930;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_930) : "f"(S[14]), "f"(k_regs[14]), "f"(acc0_x_18));
                acc0_x_18 = _fma_930;
                float _fma_931;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_931) : "f"(S[15]), "f"(k_regs[15]), "f"(acc0_y_18));
                acc0_y_18 = _fma_931;
                float dot_18 = acc0_x_18 + acc0_y_18;
                float _shfl_xor_sync_932 = __shfl_xor_sync(0xFFFFFFFF, dot_18, 1);
                dot_18 = dot_18 + _shfl_xor_sync_932;
                float _shfl_xor_sync_933 = __shfl_xor_sync(0xFFFFFFFF, dot_18, 2);
                dot_18 = dot_18 + _shfl_xor_sync_933;
                float _shfl_xor_sync_934 = __shfl_xor_sync(0xFFFFFFFF, dot_18, 4);
                dot_18 = dot_18 + _shfl_xor_sync_934;
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
                float _shfl_xor_sync_212 = __shfl_xor_sync(0xFFFFFFFF, dot_18, 1);
                dot_18 = dot_18 + _shfl_xor_sync_212;
                float _shfl_xor_sync_213 = __shfl_xor_sync(0xFFFFFFFF, dot_18, 2);
                dot_18 = dot_18 + _shfl_xor_sync_213;
                float _shfl_xor_sync_214 = __shfl_xor_sync(0xFFFFFFFF, dot_18, 4);
                dot_18 = dot_18 + _shfl_xor_sync_214;
#endif
                float vn_16 = (v_regs[0] - dot_18) * beta_4;
#if __CUDA_ARCH__ == 900
                float _fma_935;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_935) : "f"(k_regs[0]), "f"(vn_16), "f"(S[0]));
                S[0] = _fma_935;
                float _fma_936;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_936) : "f"(k_regs[1]), "f"(vn_16), "f"(S[1]));
                S[1] = _fma_936;
                float _fma_937;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_937) : "f"(k_regs[2]), "f"(vn_16), "f"(S[2]));
                S[2] = _fma_937;
                float _fma_938;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_938) : "f"(k_regs[3]), "f"(vn_16), "f"(S[3]));
                S[3] = _fma_938;
                float _fma_939;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_939) : "f"(k_regs[4]), "f"(vn_16), "f"(S[4]));
                S[4] = _fma_939;
                float _fma_940;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_940) : "f"(k_regs[5]), "f"(vn_16), "f"(S[5]));
                S[5] = _fma_940;
                float _fma_941;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_941) : "f"(k_regs[6]), "f"(vn_16), "f"(S[6]));
                S[6] = _fma_941;
                float _fma_942;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_942) : "f"(k_regs[7]), "f"(vn_16), "f"(S[7]));
                S[7] = _fma_942;
                float _fma_943;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_943) : "f"(k_regs[8]), "f"(vn_16), "f"(S[8]));
                S[8] = _fma_943;
                float _fma_944;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_944) : "f"(k_regs[9]), "f"(vn_16), "f"(S[9]));
                S[9] = _fma_944;
                float _fma_945;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_945) : "f"(k_regs[10]), "f"(vn_16), "f"(S[10]));
                S[10] = _fma_945;
                float _fma_946;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_946) : "f"(k_regs[11]), "f"(vn_16), "f"(S[11]));
                S[11] = _fma_946;
                float _fma_947;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_947) : "f"(k_regs[12]), "f"(vn_16), "f"(S[12]));
                S[12] = _fma_947;
                float _fma_948;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_948) : "f"(k_regs[13]), "f"(vn_16), "f"(S[13]));
                S[13] = _fma_948;
                float _fma_949;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_949) : "f"(k_regs[14]), "f"(vn_16), "f"(S[14]));
                S[14] = _fma_949;
                float _fma_950;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_950) : "f"(k_regs[15]), "f"(vn_16), "f"(S[15]));
                S[15] = _fma_950;
                float acc0_x_19 = 0.0f;
                float acc0_y_19 = 0.0f;
                float _fma_951;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_951) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_19));
                acc0_x_19 = _fma_951;
                float _fma_952;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_952) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_19));
                acc0_y_19 = _fma_952;
                float _fma_953;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_953) : "f"(S[18]), "f"(k_regs[2]), "f"(acc0_x_19));
                acc0_x_19 = _fma_953;
                float _fma_954;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_954) : "f"(S[19]), "f"(k_regs[3]), "f"(acc0_y_19));
                acc0_y_19 = _fma_954;
                float _fma_955;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_955) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_19));
                acc0_x_19 = _fma_955;
                float _fma_956;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_956) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_19));
                acc0_y_19 = _fma_956;
                float _fma_957;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_957) : "f"(S[22]), "f"(k_regs[6]), "f"(acc0_x_19));
                acc0_x_19 = _fma_957;
                float _fma_958;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_958) : "f"(S[23]), "f"(k_regs[7]), "f"(acc0_y_19));
                acc0_y_19 = _fma_958;
                float _fma_959;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_959) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_19));
                acc0_x_19 = _fma_959;
                float _fma_960;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_960) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_19));
                acc0_y_19 = _fma_960;
                float _fma_961;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_961) : "f"(S[26]), "f"(k_regs[10]), "f"(acc0_x_19));
                acc0_x_19 = _fma_961;
                float _fma_962;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_962) : "f"(S[27]), "f"(k_regs[11]), "f"(acc0_y_19));
                acc0_y_19 = _fma_962;
                float _fma_963;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_963) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_19));
                acc0_x_19 = _fma_963;
                float _fma_964;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_964) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_19));
                acc0_y_19 = _fma_964;
                float _fma_965;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_965) : "f"(S[30]), "f"(k_regs[14]), "f"(acc0_x_19));
                acc0_x_19 = _fma_965;
                float _fma_966;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_966) : "f"(S[31]), "f"(k_regs[15]), "f"(acc0_y_19));
                acc0_y_19 = _fma_966;
                float dot_19 = acc0_x_19 + acc0_y_19;
                float _shfl_xor_sync_967 = __shfl_xor_sync(0xFFFFFFFF, dot_19, 1);
                dot_19 = dot_19 + _shfl_xor_sync_967;
                float _shfl_xor_sync_968 = __shfl_xor_sync(0xFFFFFFFF, dot_19, 2);
                dot_19 = dot_19 + _shfl_xor_sync_968;
                float _shfl_xor_sync_969 = __shfl_xor_sync(0xFFFFFFFF, dot_19, 4);
                dot_19 = dot_19 + _shfl_xor_sync_969;
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
                float _shfl_xor_sync_215 = __shfl_xor_sync(0xFFFFFFFF, dot_19, 1);
                dot_19 = dot_19 + _shfl_xor_sync_215;
                float _shfl_xor_sync_216 = __shfl_xor_sync(0xFFFFFFFF, dot_19, 2);
                dot_19 = dot_19 + _shfl_xor_sync_216;
                float _shfl_xor_sync_217 = __shfl_xor_sync(0xFFFFFFFF, dot_19, 4);
                dot_19 = dot_19 + _shfl_xor_sync_217;
#endif
                float vn_17 = (v_regs[1] - dot_19) * beta_4;
#if __CUDA_ARCH__ == 900
                float _fma_970;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_970) : "f"(k_regs[0]), "f"(vn_17), "f"(S[16]));
                S[16] = _fma_970;
                float _fma_971;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_971) : "f"(k_regs[1]), "f"(vn_17), "f"(S[17]));
                S[17] = _fma_971;
                float _fma_972;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_972) : "f"(k_regs[2]), "f"(vn_17), "f"(S[18]));
                S[18] = _fma_972;
                float _fma_973;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_973) : "f"(k_regs[3]), "f"(vn_17), "f"(S[19]));
                S[19] = _fma_973;
                float _fma_974;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_974) : "f"(k_regs[4]), "f"(vn_17), "f"(S[20]));
                S[20] = _fma_974;
                float _fma_975;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_975) : "f"(k_regs[5]), "f"(vn_17), "f"(S[21]));
                S[21] = _fma_975;
                float _fma_976;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_976) : "f"(k_regs[6]), "f"(vn_17), "f"(S[22]));
                S[22] = _fma_976;
                float _fma_977;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_977) : "f"(k_regs[7]), "f"(vn_17), "f"(S[23]));
                S[23] = _fma_977;
                float _fma_978;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_978) : "f"(k_regs[8]), "f"(vn_17), "f"(S[24]));
                S[24] = _fma_978;
                float _fma_979;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_979) : "f"(k_regs[9]), "f"(vn_17), "f"(S[25]));
                S[25] = _fma_979;
                float _fma_980;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_980) : "f"(k_regs[10]), "f"(vn_17), "f"(S[26]));
                S[26] = _fma_980;
                float _fma_981;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_981) : "f"(k_regs[11]), "f"(vn_17), "f"(S[27]));
                S[27] = _fma_981;
                float _fma_982;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_982) : "f"(k_regs[12]), "f"(vn_17), "f"(S[28]));
                S[28] = _fma_982;
                float _fma_983;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_983) : "f"(k_regs[13]), "f"(vn_17), "f"(S[29]));
                S[29] = _fma_983;
                float _fma_984;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_984) : "f"(k_regs[14]), "f"(vn_17), "f"(S[30]));
                S[30] = _fma_984;
                float _fma_985;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_985) : "f"(k_regs[15]), "f"(vn_17), "f"(S[31]));
                S[31] = _fma_985;
                float acc0_x_20 = 0.0f;
                float acc0_y_20 = 0.0f;
                float _fma_986;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_986) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_20));
                acc0_x_20 = _fma_986;
                float _fma_987;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_987) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_20));
                acc0_y_20 = _fma_987;
                float _fma_988;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_988) : "f"(S[34]), "f"(k_regs[2]), "f"(acc0_x_20));
                acc0_x_20 = _fma_988;
                float _fma_989;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_989) : "f"(S[35]), "f"(k_regs[3]), "f"(acc0_y_20));
                acc0_y_20 = _fma_989;
                float _fma_990;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_990) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_20));
                acc0_x_20 = _fma_990;
                float _fma_991;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_991) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_20));
                acc0_y_20 = _fma_991;
                float _fma_992;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_992) : "f"(S[38]), "f"(k_regs[6]), "f"(acc0_x_20));
                acc0_x_20 = _fma_992;
                float _fma_993;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_993) : "f"(S[39]), "f"(k_regs[7]), "f"(acc0_y_20));
                acc0_y_20 = _fma_993;
                float _fma_994;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_994) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_20));
                acc0_x_20 = _fma_994;
                float _fma_995;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_995) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_20));
                acc0_y_20 = _fma_995;
                float _fma_996;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_996) : "f"(S[42]), "f"(k_regs[10]), "f"(acc0_x_20));
                acc0_x_20 = _fma_996;
                float _fma_997;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_997) : "f"(S[43]), "f"(k_regs[11]), "f"(acc0_y_20));
                acc0_y_20 = _fma_997;
                float _fma_998;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_998) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_20));
                acc0_x_20 = _fma_998;
                float _fma_999;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_999) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_20));
                acc0_y_20 = _fma_999;
                float _fma_1000;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1000) : "f"(S[46]), "f"(k_regs[14]), "f"(acc0_x_20));
                acc0_x_20 = _fma_1000;
                float _fma_1001;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1001) : "f"(S[47]), "f"(k_regs[15]), "f"(acc0_y_20));
                acc0_y_20 = _fma_1001;
                float dot_20 = acc0_x_20 + acc0_y_20;
                float _shfl_xor_sync_1002 = __shfl_xor_sync(0xFFFFFFFF, dot_20, 1);
                dot_20 = dot_20 + _shfl_xor_sync_1002;
                float _shfl_xor_sync_1003 = __shfl_xor_sync(0xFFFFFFFF, dot_20, 2);
                dot_20 = dot_20 + _shfl_xor_sync_1003;
                float _shfl_xor_sync_1004 = __shfl_xor_sync(0xFFFFFFFF, dot_20, 4);
                dot_20 = dot_20 + _shfl_xor_sync_1004;
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
                float _shfl_xor_sync_218 = __shfl_xor_sync(0xFFFFFFFF, dot_20, 1);
                dot_20 = dot_20 + _shfl_xor_sync_218;
                float _shfl_xor_sync_219 = __shfl_xor_sync(0xFFFFFFFF, dot_20, 2);
                dot_20 = dot_20 + _shfl_xor_sync_219;
                float _shfl_xor_sync_220 = __shfl_xor_sync(0xFFFFFFFF, dot_20, 4);
                dot_20 = dot_20 + _shfl_xor_sync_220;
#endif
                float vn_18 = (v_regs[2] - dot_20) * beta_4;
#if __CUDA_ARCH__ == 900
                float _fma_1005;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1005) : "f"(k_regs[0]), "f"(vn_18), "f"(S[32]));
                S[32] = _fma_1005;
                float _fma_1006;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1006) : "f"(k_regs[1]), "f"(vn_18), "f"(S[33]));
                S[33] = _fma_1006;
                float _fma_1007;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1007) : "f"(k_regs[2]), "f"(vn_18), "f"(S[34]));
                S[34] = _fma_1007;
                float _fma_1008;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1008) : "f"(k_regs[3]), "f"(vn_18), "f"(S[35]));
                S[35] = _fma_1008;
                float _fma_1009;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1009) : "f"(k_regs[4]), "f"(vn_18), "f"(S[36]));
                S[36] = _fma_1009;
                float _fma_1010;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1010) : "f"(k_regs[5]), "f"(vn_18), "f"(S[37]));
                S[37] = _fma_1010;
                float _fma_1011;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1011) : "f"(k_regs[6]), "f"(vn_18), "f"(S[38]));
                S[38] = _fma_1011;
                float _fma_1012;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1012) : "f"(k_regs[7]), "f"(vn_18), "f"(S[39]));
                S[39] = _fma_1012;
                float _fma_1013;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1013) : "f"(k_regs[8]), "f"(vn_18), "f"(S[40]));
                S[40] = _fma_1013;
                float _fma_1014;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1014) : "f"(k_regs[9]), "f"(vn_18), "f"(S[41]));
                S[41] = _fma_1014;
                float _fma_1015;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1015) : "f"(k_regs[10]), "f"(vn_18), "f"(S[42]));
                S[42] = _fma_1015;
                float _fma_1016;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1016) : "f"(k_regs[11]), "f"(vn_18), "f"(S[43]));
                S[43] = _fma_1016;
                float _fma_1017;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1017) : "f"(k_regs[12]), "f"(vn_18), "f"(S[44]));
                S[44] = _fma_1017;
                float _fma_1018;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1018) : "f"(k_regs[13]), "f"(vn_18), "f"(S[45]));
                S[45] = _fma_1018;
                float _fma_1019;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1019) : "f"(k_regs[14]), "f"(vn_18), "f"(S[46]));
                S[46] = _fma_1019;
                float _fma_1020;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1020) : "f"(k_regs[15]), "f"(vn_18), "f"(S[47]));
                S[47] = _fma_1020;
                float acc0_x_21 = 0.0f;
                float acc0_y_21 = 0.0f;
                float _fma_1021;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1021) : "f"(S[48]), "f"(k_regs[0]), "f"(acc0_x_21));
                acc0_x_21 = _fma_1021;
                float _fma_1022;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1022) : "f"(S[49]), "f"(k_regs[1]), "f"(acc0_y_21));
                acc0_y_21 = _fma_1022;
                float _fma_1023;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1023) : "f"(S[50]), "f"(k_regs[2]), "f"(acc0_x_21));
                acc0_x_21 = _fma_1023;
                float _fma_1024;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1024) : "f"(S[51]), "f"(k_regs[3]), "f"(acc0_y_21));
                acc0_y_21 = _fma_1024;
                float _fma_1025;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1025) : "f"(S[52]), "f"(k_regs[4]), "f"(acc0_x_21));
                acc0_x_21 = _fma_1025;
                float _fma_1026;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1026) : "f"(S[53]), "f"(k_regs[5]), "f"(acc0_y_21));
                acc0_y_21 = _fma_1026;
                float _fma_1027;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1027) : "f"(S[54]), "f"(k_regs[6]), "f"(acc0_x_21));
                acc0_x_21 = _fma_1027;
                float _fma_1028;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1028) : "f"(S[55]), "f"(k_regs[7]), "f"(acc0_y_21));
                acc0_y_21 = _fma_1028;
                float _fma_1029;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1029) : "f"(S[56]), "f"(k_regs[8]), "f"(acc0_x_21));
                acc0_x_21 = _fma_1029;
                float _fma_1030;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1030) : "f"(S[57]), "f"(k_regs[9]), "f"(acc0_y_21));
                acc0_y_21 = _fma_1030;
                float _fma_1031;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1031) : "f"(S[58]), "f"(k_regs[10]), "f"(acc0_x_21));
                acc0_x_21 = _fma_1031;
                float _fma_1032;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1032) : "f"(S[59]), "f"(k_regs[11]), "f"(acc0_y_21));
                acc0_y_21 = _fma_1032;
                float _fma_1033;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1033) : "f"(S[60]), "f"(k_regs[12]), "f"(acc0_x_21));
                acc0_x_21 = _fma_1033;
                float _fma_1034;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1034) : "f"(S[61]), "f"(k_regs[13]), "f"(acc0_y_21));
                acc0_y_21 = _fma_1034;
                float _fma_1035;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1035) : "f"(S[62]), "f"(k_regs[14]), "f"(acc0_x_21));
                acc0_x_21 = _fma_1035;
                float _fma_1036;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1036) : "f"(S[63]), "f"(k_regs[15]), "f"(acc0_y_21));
                acc0_y_21 = _fma_1036;
                float dot_21 = acc0_x_21 + acc0_y_21;
                float _shfl_xor_sync_1037 = __shfl_xor_sync(0xFFFFFFFF, dot_21, 1);
                dot_21 = dot_21 + _shfl_xor_sync_1037;
                float _shfl_xor_sync_1038 = __shfl_xor_sync(0xFFFFFFFF, dot_21, 2);
                dot_21 = dot_21 + _shfl_xor_sync_1038;
                float _shfl_xor_sync_1039 = __shfl_xor_sync(0xFFFFFFFF, dot_21, 4);
                dot_21 = dot_21 + _shfl_xor_sync_1039;
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
                float _shfl_xor_sync_221 = __shfl_xor_sync(0xFFFFFFFF, dot_21, 1);
                dot_21 = dot_21 + _shfl_xor_sync_221;
                float _shfl_xor_sync_222 = __shfl_xor_sync(0xFFFFFFFF, dot_21, 2);
                dot_21 = dot_21 + _shfl_xor_sync_222;
                float _shfl_xor_sync_223 = __shfl_xor_sync(0xFFFFFFFF, dot_21, 4);
                dot_21 = dot_21 + _shfl_xor_sync_223;
#endif
                float vn_19 = (v_regs[3] - dot_21) * beta_4;
#if __CUDA_ARCH__ == 900
                float _fma_1040;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1040) : "f"(k_regs[0]), "f"(vn_19), "f"(S[48]));
                S[48] = _fma_1040;
                float _fma_1041;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1041) : "f"(k_regs[1]), "f"(vn_19), "f"(S[49]));
                S[49] = _fma_1041;
                float _fma_1042;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1042) : "f"(k_regs[2]), "f"(vn_19), "f"(S[50]));
                S[50] = _fma_1042;
                float _fma_1043;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1043) : "f"(k_regs[3]), "f"(vn_19), "f"(S[51]));
                S[51] = _fma_1043;
                float _fma_1044;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1044) : "f"(k_regs[4]), "f"(vn_19), "f"(S[52]));
                S[52] = _fma_1044;
                float _fma_1045;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1045) : "f"(k_regs[5]), "f"(vn_19), "f"(S[53]));
                S[53] = _fma_1045;
                float _fma_1046;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1046) : "f"(k_regs[6]), "f"(vn_19), "f"(S[54]));
                S[54] = _fma_1046;
                float _fma_1047;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1047) : "f"(k_regs[7]), "f"(vn_19), "f"(S[55]));
                S[55] = _fma_1047;
                float _fma_1048;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1048) : "f"(k_regs[8]), "f"(vn_19), "f"(S[56]));
                S[56] = _fma_1048;
                float _fma_1049;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1049) : "f"(k_regs[9]), "f"(vn_19), "f"(S[57]));
                S[57] = _fma_1049;
                float _fma_1050;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1050) : "f"(k_regs[10]), "f"(vn_19), "f"(S[58]));
                S[58] = _fma_1050;
                float _fma_1051;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1051) : "f"(k_regs[11]), "f"(vn_19), "f"(S[59]));
                S[59] = _fma_1051;
                float _fma_1052;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1052) : "f"(k_regs[12]), "f"(vn_19), "f"(S[60]));
                S[60] = _fma_1052;
                float _fma_1053;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1053) : "f"(k_regs[13]), "f"(vn_19), "f"(S[61]));
                S[61] = _fma_1053;
                float _fma_1054;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1054) : "f"(k_regs[14]), "f"(vn_19), "f"(S[62]));
                S[62] = _fma_1054;
                float _fma_1055;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1055) : "f"(k_regs[15]), "f"(vn_19), "f"(S[63]));
                S[63] = _fma_1055;
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
                float _fma_1056;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1056) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_22));
                acc0_x_22 = _fma_1056;
                float _fma_1057;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1057) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_22));
                acc0_y_22 = _fma_1057;
                float _fma_1058;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1058) : "f"(S[2]), "f"(k_regs[2]), "f"(acc0_x_22));
                acc0_x_22 = _fma_1058;
                float _fma_1059;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1059) : "f"(S[3]), "f"(k_regs[3]), "f"(acc0_y_22));
                acc0_y_22 = _fma_1059;
                float _fma_1060;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1060) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_22));
                acc0_x_22 = _fma_1060;
                float _fma_1061;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1061) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_22));
                acc0_y_22 = _fma_1061;
                float _fma_1062;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1062) : "f"(S[6]), "f"(k_regs[6]), "f"(acc0_x_22));
                acc0_x_22 = _fma_1062;
                float _fma_1063;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1063) : "f"(S[7]), "f"(k_regs[7]), "f"(acc0_y_22));
                acc0_y_22 = _fma_1063;
                float _fma_1064;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1064) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_22));
                acc0_x_22 = _fma_1064;
                float _fma_1065;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1065) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_22));
                acc0_y_22 = _fma_1065;
                float _fma_1066;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1066) : "f"(S[10]), "f"(k_regs[10]), "f"(acc0_x_22));
                acc0_x_22 = _fma_1066;
                float _fma_1067;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1067) : "f"(S[11]), "f"(k_regs[11]), "f"(acc0_y_22));
                acc0_y_22 = _fma_1067;
                float _fma_1068;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1068) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_22));
                acc0_x_22 = _fma_1068;
                float _fma_1069;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1069) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_22));
                acc0_y_22 = _fma_1069;
                float _fma_1070;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1070) : "f"(S[14]), "f"(k_regs[14]), "f"(acc0_x_22));
                acc0_x_22 = _fma_1070;
                float _fma_1071;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1071) : "f"(S[15]), "f"(k_regs[15]), "f"(acc0_y_22));
                acc0_y_22 = _fma_1071;
                float dot_22 = acc0_x_22 + acc0_y_22;
                float _shfl_xor_sync_1072 = __shfl_xor_sync(0xFFFFFFFF, dot_22, 1);
                dot_22 = dot_22 + _shfl_xor_sync_1072;
                float _shfl_xor_sync_1073 = __shfl_xor_sync(0xFFFFFFFF, dot_22, 2);
                dot_22 = dot_22 + _shfl_xor_sync_1073;
                float _shfl_xor_sync_1074 = __shfl_xor_sync(0xFFFFFFFF, dot_22, 4);
                dot_22 = dot_22 + _shfl_xor_sync_1074;
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
                float _shfl_xor_sync_224 = __shfl_xor_sync(0xFFFFFFFF, dot_22, 1);
                dot_22 = dot_22 + _shfl_xor_sync_224;
                float _shfl_xor_sync_225 = __shfl_xor_sync(0xFFFFFFFF, dot_22, 2);
                dot_22 = dot_22 + _shfl_xor_sync_225;
                float _shfl_xor_sync_226 = __shfl_xor_sync(0xFFFFFFFF, dot_22, 4);
                dot_22 = dot_22 + _shfl_xor_sync_226;
#endif
                float vn_20 = (v_regs[0] - dot_22) * beta_5;
#if __CUDA_ARCH__ == 900
                float _fma_1075;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1075) : "f"(k_regs[0]), "f"(vn_20), "f"(S[0]));
                S[0] = _fma_1075;
                float _fma_1076;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1076) : "f"(k_regs[1]), "f"(vn_20), "f"(S[1]));
                S[1] = _fma_1076;
                float _fma_1077;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1077) : "f"(k_regs[2]), "f"(vn_20), "f"(S[2]));
                S[2] = _fma_1077;
                float _fma_1078;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1078) : "f"(k_regs[3]), "f"(vn_20), "f"(S[3]));
                S[3] = _fma_1078;
                float _fma_1079;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1079) : "f"(k_regs[4]), "f"(vn_20), "f"(S[4]));
                S[4] = _fma_1079;
                float _fma_1080;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1080) : "f"(k_regs[5]), "f"(vn_20), "f"(S[5]));
                S[5] = _fma_1080;
                float _fma_1081;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1081) : "f"(k_regs[6]), "f"(vn_20), "f"(S[6]));
                S[6] = _fma_1081;
                float _fma_1082;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1082) : "f"(k_regs[7]), "f"(vn_20), "f"(S[7]));
                S[7] = _fma_1082;
                float _fma_1083;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1083) : "f"(k_regs[8]), "f"(vn_20), "f"(S[8]));
                S[8] = _fma_1083;
                float _fma_1084;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1084) : "f"(k_regs[9]), "f"(vn_20), "f"(S[9]));
                S[9] = _fma_1084;
                float _fma_1085;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1085) : "f"(k_regs[10]), "f"(vn_20), "f"(S[10]));
                S[10] = _fma_1085;
                float _fma_1086;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1086) : "f"(k_regs[11]), "f"(vn_20), "f"(S[11]));
                S[11] = _fma_1086;
                float _fma_1087;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1087) : "f"(k_regs[12]), "f"(vn_20), "f"(S[12]));
                S[12] = _fma_1087;
                float _fma_1088;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1088) : "f"(k_regs[13]), "f"(vn_20), "f"(S[13]));
                S[13] = _fma_1088;
                float _fma_1089;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1089) : "f"(k_regs[14]), "f"(vn_20), "f"(S[14]));
                S[14] = _fma_1089;
                float _fma_1090;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1090) : "f"(k_regs[15]), "f"(vn_20), "f"(S[15]));
                S[15] = _fma_1090;
                float acc0_x_23 = 0.0f;
                float acc0_y_23 = 0.0f;
                float _fma_1091;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1091) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_23));
                acc0_x_23 = _fma_1091;
                float _fma_1092;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1092) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_23));
                acc0_y_23 = _fma_1092;
                float _fma_1093;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1093) : "f"(S[18]), "f"(k_regs[2]), "f"(acc0_x_23));
                acc0_x_23 = _fma_1093;
                float _fma_1094;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1094) : "f"(S[19]), "f"(k_regs[3]), "f"(acc0_y_23));
                acc0_y_23 = _fma_1094;
                float _fma_1095;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1095) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_23));
                acc0_x_23 = _fma_1095;
                float _fma_1096;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1096) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_23));
                acc0_y_23 = _fma_1096;
                float _fma_1097;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1097) : "f"(S[22]), "f"(k_regs[6]), "f"(acc0_x_23));
                acc0_x_23 = _fma_1097;
                float _fma_1098;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1098) : "f"(S[23]), "f"(k_regs[7]), "f"(acc0_y_23));
                acc0_y_23 = _fma_1098;
                float _fma_1099;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1099) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_23));
                acc0_x_23 = _fma_1099;
                float _fma_1100;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1100) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_23));
                acc0_y_23 = _fma_1100;
                float _fma_1101;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1101) : "f"(S[26]), "f"(k_regs[10]), "f"(acc0_x_23));
                acc0_x_23 = _fma_1101;
                float _fma_1102;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1102) : "f"(S[27]), "f"(k_regs[11]), "f"(acc0_y_23));
                acc0_y_23 = _fma_1102;
                float _fma_1103;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1103) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_23));
                acc0_x_23 = _fma_1103;
                float _fma_1104;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1104) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_23));
                acc0_y_23 = _fma_1104;
                float _fma_1105;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1105) : "f"(S[30]), "f"(k_regs[14]), "f"(acc0_x_23));
                acc0_x_23 = _fma_1105;
                float _fma_1106;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1106) : "f"(S[31]), "f"(k_regs[15]), "f"(acc0_y_23));
                acc0_y_23 = _fma_1106;
                float dot_23 = acc0_x_23 + acc0_y_23;
                float _shfl_xor_sync_1107 = __shfl_xor_sync(0xFFFFFFFF, dot_23, 1);
                dot_23 = dot_23 + _shfl_xor_sync_1107;
                float _shfl_xor_sync_1108 = __shfl_xor_sync(0xFFFFFFFF, dot_23, 2);
                dot_23 = dot_23 + _shfl_xor_sync_1108;
                float _shfl_xor_sync_1109 = __shfl_xor_sync(0xFFFFFFFF, dot_23, 4);
                dot_23 = dot_23 + _shfl_xor_sync_1109;
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
                float _shfl_xor_sync_227 = __shfl_xor_sync(0xFFFFFFFF, dot_23, 1);
                dot_23 = dot_23 + _shfl_xor_sync_227;
                float _shfl_xor_sync_228 = __shfl_xor_sync(0xFFFFFFFF, dot_23, 2);
                dot_23 = dot_23 + _shfl_xor_sync_228;
                float _shfl_xor_sync_229 = __shfl_xor_sync(0xFFFFFFFF, dot_23, 4);
                dot_23 = dot_23 + _shfl_xor_sync_229;
#endif
                float vn_21 = (v_regs[1] - dot_23) * beta_5;
#if __CUDA_ARCH__ == 900
                float _fma_1110;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1110) : "f"(k_regs[0]), "f"(vn_21), "f"(S[16]));
                S[16] = _fma_1110;
                float _fma_1111;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1111) : "f"(k_regs[1]), "f"(vn_21), "f"(S[17]));
                S[17] = _fma_1111;
                float _fma_1112;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1112) : "f"(k_regs[2]), "f"(vn_21), "f"(S[18]));
                S[18] = _fma_1112;
                float _fma_1113;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1113) : "f"(k_regs[3]), "f"(vn_21), "f"(S[19]));
                S[19] = _fma_1113;
                float _fma_1114;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1114) : "f"(k_regs[4]), "f"(vn_21), "f"(S[20]));
                S[20] = _fma_1114;
                float _fma_1115;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1115) : "f"(k_regs[5]), "f"(vn_21), "f"(S[21]));
                S[21] = _fma_1115;
                float _fma_1116;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1116) : "f"(k_regs[6]), "f"(vn_21), "f"(S[22]));
                S[22] = _fma_1116;
                float _fma_1117;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1117) : "f"(k_regs[7]), "f"(vn_21), "f"(S[23]));
                S[23] = _fma_1117;
                float _fma_1118;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1118) : "f"(k_regs[8]), "f"(vn_21), "f"(S[24]));
                S[24] = _fma_1118;
                float _fma_1119;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1119) : "f"(k_regs[9]), "f"(vn_21), "f"(S[25]));
                S[25] = _fma_1119;
                float _fma_1120;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1120) : "f"(k_regs[10]), "f"(vn_21), "f"(S[26]));
                S[26] = _fma_1120;
                float _fma_1121;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1121) : "f"(k_regs[11]), "f"(vn_21), "f"(S[27]));
                S[27] = _fma_1121;
                float _fma_1122;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1122) : "f"(k_regs[12]), "f"(vn_21), "f"(S[28]));
                S[28] = _fma_1122;
                float _fma_1123;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1123) : "f"(k_regs[13]), "f"(vn_21), "f"(S[29]));
                S[29] = _fma_1123;
                float _fma_1124;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1124) : "f"(k_regs[14]), "f"(vn_21), "f"(S[30]));
                S[30] = _fma_1124;
                float _fma_1125;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1125) : "f"(k_regs[15]), "f"(vn_21), "f"(S[31]));
                S[31] = _fma_1125;
                float acc0_x_24 = 0.0f;
                float acc0_y_24 = 0.0f;
                float _fma_1126;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1126) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_24));
                acc0_x_24 = _fma_1126;
                float _fma_1127;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1127) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_24));
                acc0_y_24 = _fma_1127;
                float _fma_1128;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1128) : "f"(S[34]), "f"(k_regs[2]), "f"(acc0_x_24));
                acc0_x_24 = _fma_1128;
                float _fma_1129;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1129) : "f"(S[35]), "f"(k_regs[3]), "f"(acc0_y_24));
                acc0_y_24 = _fma_1129;
                float _fma_1130;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1130) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_24));
                acc0_x_24 = _fma_1130;
                float _fma_1131;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1131) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_24));
                acc0_y_24 = _fma_1131;
                float _fma_1132;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1132) : "f"(S[38]), "f"(k_regs[6]), "f"(acc0_x_24));
                acc0_x_24 = _fma_1132;
                float _fma_1133;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1133) : "f"(S[39]), "f"(k_regs[7]), "f"(acc0_y_24));
                acc0_y_24 = _fma_1133;
                float _fma_1134;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1134) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_24));
                acc0_x_24 = _fma_1134;
                float _fma_1135;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1135) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_24));
                acc0_y_24 = _fma_1135;
                float _fma_1136;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1136) : "f"(S[42]), "f"(k_regs[10]), "f"(acc0_x_24));
                acc0_x_24 = _fma_1136;
                float _fma_1137;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1137) : "f"(S[43]), "f"(k_regs[11]), "f"(acc0_y_24));
                acc0_y_24 = _fma_1137;
                float _fma_1138;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1138) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_24));
                acc0_x_24 = _fma_1138;
                float _fma_1139;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1139) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_24));
                acc0_y_24 = _fma_1139;
                float _fma_1140;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1140) : "f"(S[46]), "f"(k_regs[14]), "f"(acc0_x_24));
                acc0_x_24 = _fma_1140;
                float _fma_1141;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1141) : "f"(S[47]), "f"(k_regs[15]), "f"(acc0_y_24));
                acc0_y_24 = _fma_1141;
                float dot_24 = acc0_x_24 + acc0_y_24;
                float _shfl_xor_sync_1142 = __shfl_xor_sync(0xFFFFFFFF, dot_24, 1);
                dot_24 = dot_24 + _shfl_xor_sync_1142;
                float _shfl_xor_sync_1143 = __shfl_xor_sync(0xFFFFFFFF, dot_24, 2);
                dot_24 = dot_24 + _shfl_xor_sync_1143;
                float _shfl_xor_sync_1144 = __shfl_xor_sync(0xFFFFFFFF, dot_24, 4);
                dot_24 = dot_24 + _shfl_xor_sync_1144;
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
                float _shfl_xor_sync_230 = __shfl_xor_sync(0xFFFFFFFF, dot_24, 1);
                dot_24 = dot_24 + _shfl_xor_sync_230;
                float _shfl_xor_sync_231 = __shfl_xor_sync(0xFFFFFFFF, dot_24, 2);
                dot_24 = dot_24 + _shfl_xor_sync_231;
                float _shfl_xor_sync_232 = __shfl_xor_sync(0xFFFFFFFF, dot_24, 4);
                dot_24 = dot_24 + _shfl_xor_sync_232;
#endif
                float vn_22 = (v_regs[2] - dot_24) * beta_5;
#if __CUDA_ARCH__ == 900
                float _fma_1145;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1145) : "f"(k_regs[0]), "f"(vn_22), "f"(S[32]));
                S[32] = _fma_1145;
                float _fma_1146;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1146) : "f"(k_regs[1]), "f"(vn_22), "f"(S[33]));
                S[33] = _fma_1146;
                float _fma_1147;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1147) : "f"(k_regs[2]), "f"(vn_22), "f"(S[34]));
                S[34] = _fma_1147;
                float _fma_1148;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1148) : "f"(k_regs[3]), "f"(vn_22), "f"(S[35]));
                S[35] = _fma_1148;
                float _fma_1149;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1149) : "f"(k_regs[4]), "f"(vn_22), "f"(S[36]));
                S[36] = _fma_1149;
                float _fma_1150;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1150) : "f"(k_regs[5]), "f"(vn_22), "f"(S[37]));
                S[37] = _fma_1150;
                float _fma_1151;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1151) : "f"(k_regs[6]), "f"(vn_22), "f"(S[38]));
                S[38] = _fma_1151;
                float _fma_1152;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1152) : "f"(k_regs[7]), "f"(vn_22), "f"(S[39]));
                S[39] = _fma_1152;
                float _fma_1153;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1153) : "f"(k_regs[8]), "f"(vn_22), "f"(S[40]));
                S[40] = _fma_1153;
                float _fma_1154;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1154) : "f"(k_regs[9]), "f"(vn_22), "f"(S[41]));
                S[41] = _fma_1154;
                float _fma_1155;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1155) : "f"(k_regs[10]), "f"(vn_22), "f"(S[42]));
                S[42] = _fma_1155;
                float _fma_1156;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1156) : "f"(k_regs[11]), "f"(vn_22), "f"(S[43]));
                S[43] = _fma_1156;
                float _fma_1157;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1157) : "f"(k_regs[12]), "f"(vn_22), "f"(S[44]));
                S[44] = _fma_1157;
                float _fma_1158;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1158) : "f"(k_regs[13]), "f"(vn_22), "f"(S[45]));
                S[45] = _fma_1158;
                float _fma_1159;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1159) : "f"(k_regs[14]), "f"(vn_22), "f"(S[46]));
                S[46] = _fma_1159;
                float _fma_1160;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1160) : "f"(k_regs[15]), "f"(vn_22), "f"(S[47]));
                S[47] = _fma_1160;
                float acc0_x_25 = 0.0f;
                float acc0_y_25 = 0.0f;
                float _fma_1161;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1161) : "f"(S[48]), "f"(k_regs[0]), "f"(acc0_x_25));
                acc0_x_25 = _fma_1161;
                float _fma_1162;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1162) : "f"(S[49]), "f"(k_regs[1]), "f"(acc0_y_25));
                acc0_y_25 = _fma_1162;
                float _fma_1163;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1163) : "f"(S[50]), "f"(k_regs[2]), "f"(acc0_x_25));
                acc0_x_25 = _fma_1163;
                float _fma_1164;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1164) : "f"(S[51]), "f"(k_regs[3]), "f"(acc0_y_25));
                acc0_y_25 = _fma_1164;
                float _fma_1165;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1165) : "f"(S[52]), "f"(k_regs[4]), "f"(acc0_x_25));
                acc0_x_25 = _fma_1165;
                float _fma_1166;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1166) : "f"(S[53]), "f"(k_regs[5]), "f"(acc0_y_25));
                acc0_y_25 = _fma_1166;
                float _fma_1167;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1167) : "f"(S[54]), "f"(k_regs[6]), "f"(acc0_x_25));
                acc0_x_25 = _fma_1167;
                float _fma_1168;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1168) : "f"(S[55]), "f"(k_regs[7]), "f"(acc0_y_25));
                acc0_y_25 = _fma_1168;
                float _fma_1169;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1169) : "f"(S[56]), "f"(k_regs[8]), "f"(acc0_x_25));
                acc0_x_25 = _fma_1169;
                float _fma_1170;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1170) : "f"(S[57]), "f"(k_regs[9]), "f"(acc0_y_25));
                acc0_y_25 = _fma_1170;
                float _fma_1171;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1171) : "f"(S[58]), "f"(k_regs[10]), "f"(acc0_x_25));
                acc0_x_25 = _fma_1171;
                float _fma_1172;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1172) : "f"(S[59]), "f"(k_regs[11]), "f"(acc0_y_25));
                acc0_y_25 = _fma_1172;
                float _fma_1173;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1173) : "f"(S[60]), "f"(k_regs[12]), "f"(acc0_x_25));
                acc0_x_25 = _fma_1173;
                float _fma_1174;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1174) : "f"(S[61]), "f"(k_regs[13]), "f"(acc0_y_25));
                acc0_y_25 = _fma_1174;
                float _fma_1175;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1175) : "f"(S[62]), "f"(k_regs[14]), "f"(acc0_x_25));
                acc0_x_25 = _fma_1175;
                float _fma_1176;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1176) : "f"(S[63]), "f"(k_regs[15]), "f"(acc0_y_25));
                acc0_y_25 = _fma_1176;
                float dot_25 = acc0_x_25 + acc0_y_25;
                float _shfl_xor_sync_1177 = __shfl_xor_sync(0xFFFFFFFF, dot_25, 1);
                dot_25 = dot_25 + _shfl_xor_sync_1177;
                float _shfl_xor_sync_1178 = __shfl_xor_sync(0xFFFFFFFF, dot_25, 2);
                dot_25 = dot_25 + _shfl_xor_sync_1178;
                float _shfl_xor_sync_1179 = __shfl_xor_sync(0xFFFFFFFF, dot_25, 4);
                dot_25 = dot_25 + _shfl_xor_sync_1179;
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
                float _shfl_xor_sync_233 = __shfl_xor_sync(0xFFFFFFFF, dot_25, 1);
                dot_25 = dot_25 + _shfl_xor_sync_233;
                float _shfl_xor_sync_234 = __shfl_xor_sync(0xFFFFFFFF, dot_25, 2);
                dot_25 = dot_25 + _shfl_xor_sync_234;
                float _shfl_xor_sync_235 = __shfl_xor_sync(0xFFFFFFFF, dot_25, 4);
                dot_25 = dot_25 + _shfl_xor_sync_235;
#endif
                float vn_23 = (v_regs[3] - dot_25) * beta_5;
#if __CUDA_ARCH__ == 900
                float _fma_1180;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1180) : "f"(k_regs[0]), "f"(vn_23), "f"(S[48]));
                S[48] = _fma_1180;
                float _fma_1181;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1181) : "f"(k_regs[1]), "f"(vn_23), "f"(S[49]));
                S[49] = _fma_1181;
                float _fma_1182;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1182) : "f"(k_regs[2]), "f"(vn_23), "f"(S[50]));
                S[50] = _fma_1182;
                float _fma_1183;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1183) : "f"(k_regs[3]), "f"(vn_23), "f"(S[51]));
                S[51] = _fma_1183;
                float _fma_1184;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1184) : "f"(k_regs[4]), "f"(vn_23), "f"(S[52]));
                S[52] = _fma_1184;
                float _fma_1185;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1185) : "f"(k_regs[5]), "f"(vn_23), "f"(S[53]));
                S[53] = _fma_1185;
                float _fma_1186;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1186) : "f"(k_regs[6]), "f"(vn_23), "f"(S[54]));
                S[54] = _fma_1186;
                float _fma_1187;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1187) : "f"(k_regs[7]), "f"(vn_23), "f"(S[55]));
                S[55] = _fma_1187;
                float _fma_1188;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1188) : "f"(k_regs[8]), "f"(vn_23), "f"(S[56]));
                S[56] = _fma_1188;
                float _fma_1189;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1189) : "f"(k_regs[9]), "f"(vn_23), "f"(S[57]));
                S[57] = _fma_1189;
                float _fma_1190;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1190) : "f"(k_regs[10]), "f"(vn_23), "f"(S[58]));
                S[58] = _fma_1190;
                float _fma_1191;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1191) : "f"(k_regs[11]), "f"(vn_23), "f"(S[59]));
                S[59] = _fma_1191;
                float _fma_1192;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1192) : "f"(k_regs[12]), "f"(vn_23), "f"(S[60]));
                S[60] = _fma_1192;
                float _fma_1193;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1193) : "f"(k_regs[13]), "f"(vn_23), "f"(S[61]));
                S[61] = _fma_1193;
                float _fma_1194;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1194) : "f"(k_regs[14]), "f"(vn_23), "f"(S[62]));
                S[62] = _fma_1194;
                float _fma_1195;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1195) : "f"(k_regs[15]), "f"(vn_23), "f"(S[63]));
                S[63] = _fma_1195;
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
                float _fma_1196;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1196) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_26));
                acc0_x_26 = _fma_1196;
                float _fma_1197;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1197) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_26));
                acc0_y_26 = _fma_1197;
                float _fma_1198;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1198) : "f"(S[2]), "f"(k_regs[2]), "f"(acc0_x_26));
                acc0_x_26 = _fma_1198;
                float _fma_1199;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1199) : "f"(S[3]), "f"(k_regs[3]), "f"(acc0_y_26));
                acc0_y_26 = _fma_1199;
                float _fma_1200;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1200) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_26));
                acc0_x_26 = _fma_1200;
                float _fma_1201;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1201) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_26));
                acc0_y_26 = _fma_1201;
                float _fma_1202;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1202) : "f"(S[6]), "f"(k_regs[6]), "f"(acc0_x_26));
                acc0_x_26 = _fma_1202;
                float _fma_1203;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1203) : "f"(S[7]), "f"(k_regs[7]), "f"(acc0_y_26));
                acc0_y_26 = _fma_1203;
                float _fma_1204;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1204) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_26));
                acc0_x_26 = _fma_1204;
                float _fma_1205;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1205) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_26));
                acc0_y_26 = _fma_1205;
                float _fma_1206;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1206) : "f"(S[10]), "f"(k_regs[10]), "f"(acc0_x_26));
                acc0_x_26 = _fma_1206;
                float _fma_1207;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1207) : "f"(S[11]), "f"(k_regs[11]), "f"(acc0_y_26));
                acc0_y_26 = _fma_1207;
                float _fma_1208;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1208) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_26));
                acc0_x_26 = _fma_1208;
                float _fma_1209;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1209) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_26));
                acc0_y_26 = _fma_1209;
                float _fma_1210;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1210) : "f"(S[14]), "f"(k_regs[14]), "f"(acc0_x_26));
                acc0_x_26 = _fma_1210;
                float _fma_1211;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1211) : "f"(S[15]), "f"(k_regs[15]), "f"(acc0_y_26));
                acc0_y_26 = _fma_1211;
                float dot_26 = acc0_x_26 + acc0_y_26;
                float _shfl_xor_sync_1212 = __shfl_xor_sync(0xFFFFFFFF, dot_26, 1);
                dot_26 = dot_26 + _shfl_xor_sync_1212;
                float _shfl_xor_sync_1213 = __shfl_xor_sync(0xFFFFFFFF, dot_26, 2);
                dot_26 = dot_26 + _shfl_xor_sync_1213;
                float _shfl_xor_sync_1214 = __shfl_xor_sync(0xFFFFFFFF, dot_26, 4);
                dot_26 = dot_26 + _shfl_xor_sync_1214;
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
                float _shfl_xor_sync_236 = __shfl_xor_sync(0xFFFFFFFF, dot_26, 1);
                dot_26 = dot_26 + _shfl_xor_sync_236;
                float _shfl_xor_sync_237 = __shfl_xor_sync(0xFFFFFFFF, dot_26, 2);
                dot_26 = dot_26 + _shfl_xor_sync_237;
                float _shfl_xor_sync_238 = __shfl_xor_sync(0xFFFFFFFF, dot_26, 4);
                dot_26 = dot_26 + _shfl_xor_sync_238;
#endif
                o4[0] = dot_26;
#if __CUDA_ARCH__ == 900
                float acc0_x_27 = 0.0f;
                float acc0_y_27 = 0.0f;
                float _fma_1215;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1215) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_27));
                acc0_x_27 = _fma_1215;
                float _fma_1216;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1216) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_27));
                acc0_y_27 = _fma_1216;
                float _fma_1217;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1217) : "f"(S[18]), "f"(k_regs[2]), "f"(acc0_x_27));
                acc0_x_27 = _fma_1217;
                float _fma_1218;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1218) : "f"(S[19]), "f"(k_regs[3]), "f"(acc0_y_27));
                acc0_y_27 = _fma_1218;
                float _fma_1219;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1219) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_27));
                acc0_x_27 = _fma_1219;
                float _fma_1220;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1220) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_27));
                acc0_y_27 = _fma_1220;
                float _fma_1221;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1221) : "f"(S[22]), "f"(k_regs[6]), "f"(acc0_x_27));
                acc0_x_27 = _fma_1221;
                float _fma_1222;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1222) : "f"(S[23]), "f"(k_regs[7]), "f"(acc0_y_27));
                acc0_y_27 = _fma_1222;
                float _fma_1223;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1223) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_27));
                acc0_x_27 = _fma_1223;
                float _fma_1224;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1224) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_27));
                acc0_y_27 = _fma_1224;
                float _fma_1225;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1225) : "f"(S[26]), "f"(k_regs[10]), "f"(acc0_x_27));
                acc0_x_27 = _fma_1225;
                float _fma_1226;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1226) : "f"(S[27]), "f"(k_regs[11]), "f"(acc0_y_27));
                acc0_y_27 = _fma_1226;
                float _fma_1227;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1227) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_27));
                acc0_x_27 = _fma_1227;
                float _fma_1228;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1228) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_27));
                acc0_y_27 = _fma_1228;
                float _fma_1229;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1229) : "f"(S[30]), "f"(k_regs[14]), "f"(acc0_x_27));
                acc0_x_27 = _fma_1229;
                float _fma_1230;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1230) : "f"(S[31]), "f"(k_regs[15]), "f"(acc0_y_27));
                acc0_y_27 = _fma_1230;
                float dot_27 = acc0_x_27 + acc0_y_27;
                float _shfl_xor_sync_1231 = __shfl_xor_sync(0xFFFFFFFF, dot_27, 1);
                dot_27 = dot_27 + _shfl_xor_sync_1231;
                float _shfl_xor_sync_1232 = __shfl_xor_sync(0xFFFFFFFF, dot_27, 2);
                dot_27 = dot_27 + _shfl_xor_sync_1232;
                float _shfl_xor_sync_1233 = __shfl_xor_sync(0xFFFFFFFF, dot_27, 4);
                dot_27 = dot_27 + _shfl_xor_sync_1233;
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
                float _shfl_xor_sync_239 = __shfl_xor_sync(0xFFFFFFFF, dot_27, 1);
                dot_27 = dot_27 + _shfl_xor_sync_239;
                float _shfl_xor_sync_240 = __shfl_xor_sync(0xFFFFFFFF, dot_27, 2);
                dot_27 = dot_27 + _shfl_xor_sync_240;
                float _shfl_xor_sync_241 = __shfl_xor_sync(0xFFFFFFFF, dot_27, 4);
                dot_27 = dot_27 + _shfl_xor_sync_241;
#endif
                o4[1] = dot_27;
#if __CUDA_ARCH__ == 900
                float acc0_x_28 = 0.0f;
                float acc0_y_28 = 0.0f;
                float _fma_1234;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1234) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_28));
                acc0_x_28 = _fma_1234;
                float _fma_1235;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1235) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_28));
                acc0_y_28 = _fma_1235;
                float _fma_1236;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1236) : "f"(S[34]), "f"(k_regs[2]), "f"(acc0_x_28));
                acc0_x_28 = _fma_1236;
                float _fma_1237;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1237) : "f"(S[35]), "f"(k_regs[3]), "f"(acc0_y_28));
                acc0_y_28 = _fma_1237;
                float _fma_1238;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1238) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_28));
                acc0_x_28 = _fma_1238;
                float _fma_1239;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1239) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_28));
                acc0_y_28 = _fma_1239;
                float _fma_1240;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1240) : "f"(S[38]), "f"(k_regs[6]), "f"(acc0_x_28));
                acc0_x_28 = _fma_1240;
                float _fma_1241;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1241) : "f"(S[39]), "f"(k_regs[7]), "f"(acc0_y_28));
                acc0_y_28 = _fma_1241;
                float _fma_1242;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1242) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_28));
                acc0_x_28 = _fma_1242;
                float _fma_1243;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1243) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_28));
                acc0_y_28 = _fma_1243;
                float _fma_1244;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1244) : "f"(S[42]), "f"(k_regs[10]), "f"(acc0_x_28));
                acc0_x_28 = _fma_1244;
                float _fma_1245;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1245) : "f"(S[43]), "f"(k_regs[11]), "f"(acc0_y_28));
                acc0_y_28 = _fma_1245;
                float _fma_1246;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1246) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_28));
                acc0_x_28 = _fma_1246;
                float _fma_1247;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1247) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_28));
                acc0_y_28 = _fma_1247;
                float _fma_1248;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1248) : "f"(S[46]), "f"(k_regs[14]), "f"(acc0_x_28));
                acc0_x_28 = _fma_1248;
                float _fma_1249;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1249) : "f"(S[47]), "f"(k_regs[15]), "f"(acc0_y_28));
                acc0_y_28 = _fma_1249;
                float dot_28 = acc0_x_28 + acc0_y_28;
                float _shfl_xor_sync_1250 = __shfl_xor_sync(0xFFFFFFFF, dot_28, 1);
                dot_28 = dot_28 + _shfl_xor_sync_1250;
                float _shfl_xor_sync_1251 = __shfl_xor_sync(0xFFFFFFFF, dot_28, 2);
                dot_28 = dot_28 + _shfl_xor_sync_1251;
                float _shfl_xor_sync_1252 = __shfl_xor_sync(0xFFFFFFFF, dot_28, 4);
                dot_28 = dot_28 + _shfl_xor_sync_1252;
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
                float _shfl_xor_sync_242 = __shfl_xor_sync(0xFFFFFFFF, dot_28, 1);
                dot_28 = dot_28 + _shfl_xor_sync_242;
                float _shfl_xor_sync_243 = __shfl_xor_sync(0xFFFFFFFF, dot_28, 2);
                dot_28 = dot_28 + _shfl_xor_sync_243;
                float _shfl_xor_sync_244 = __shfl_xor_sync(0xFFFFFFFF, dot_28, 4);
                dot_28 = dot_28 + _shfl_xor_sync_244;
#endif
                o4[2] = dot_28;
#if __CUDA_ARCH__ == 900
                float acc0_x_29 = 0.0f;
                float acc0_y_29 = 0.0f;
                float _fma_1253;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1253) : "f"(S[48]), "f"(k_regs[0]), "f"(acc0_x_29));
                acc0_x_29 = _fma_1253;
                float _fma_1254;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1254) : "f"(S[49]), "f"(k_regs[1]), "f"(acc0_y_29));
                acc0_y_29 = _fma_1254;
                float _fma_1255;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1255) : "f"(S[50]), "f"(k_regs[2]), "f"(acc0_x_29));
                acc0_x_29 = _fma_1255;
                float _fma_1256;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1256) : "f"(S[51]), "f"(k_regs[3]), "f"(acc0_y_29));
                acc0_y_29 = _fma_1256;
                float _fma_1257;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1257) : "f"(S[52]), "f"(k_regs[4]), "f"(acc0_x_29));
                acc0_x_29 = _fma_1257;
                float _fma_1258;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1258) : "f"(S[53]), "f"(k_regs[5]), "f"(acc0_y_29));
                acc0_y_29 = _fma_1258;
                float _fma_1259;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1259) : "f"(S[54]), "f"(k_regs[6]), "f"(acc0_x_29));
                acc0_x_29 = _fma_1259;
                float _fma_1260;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1260) : "f"(S[55]), "f"(k_regs[7]), "f"(acc0_y_29));
                acc0_y_29 = _fma_1260;
                float _fma_1261;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1261) : "f"(S[56]), "f"(k_regs[8]), "f"(acc0_x_29));
                acc0_x_29 = _fma_1261;
                float _fma_1262;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1262) : "f"(S[57]), "f"(k_regs[9]), "f"(acc0_y_29));
                acc0_y_29 = _fma_1262;
                float _fma_1263;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1263) : "f"(S[58]), "f"(k_regs[10]), "f"(acc0_x_29));
                acc0_x_29 = _fma_1263;
                float _fma_1264;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1264) : "f"(S[59]), "f"(k_regs[11]), "f"(acc0_y_29));
                acc0_y_29 = _fma_1264;
                float _fma_1265;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1265) : "f"(S[60]), "f"(k_regs[12]), "f"(acc0_x_29));
                acc0_x_29 = _fma_1265;
                float _fma_1266;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1266) : "f"(S[61]), "f"(k_regs[13]), "f"(acc0_y_29));
                acc0_y_29 = _fma_1266;
                float _fma_1267;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1267) : "f"(S[62]), "f"(k_regs[14]), "f"(acc0_x_29));
                acc0_x_29 = _fma_1267;
                float _fma_1268;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1268) : "f"(S[63]), "f"(k_regs[15]), "f"(acc0_y_29));
                acc0_y_29 = _fma_1268;
                float dot_29 = acc0_x_29 + acc0_y_29;
                float _shfl_xor_sync_1269 = __shfl_xor_sync(0xFFFFFFFF, dot_29, 1);
                dot_29 = dot_29 + _shfl_xor_sync_1269;
                float _shfl_xor_sync_1270 = __shfl_xor_sync(0xFFFFFFFF, dot_29, 2);
                dot_29 = dot_29 + _shfl_xor_sync_1270;
                float _shfl_xor_sync_1271 = __shfl_xor_sync(0xFFFFFFFF, dot_29, 4);
                dot_29 = dot_29 + _shfl_xor_sync_1271;
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
                float _shfl_xor_sync_245 = __shfl_xor_sync(0xFFFFFFFF, dot_29, 1);
                dot_29 = dot_29 + _shfl_xor_sync_245;
                float _shfl_xor_sync_246 = __shfl_xor_sync(0xFFFFFFFF, dot_29, 2);
                dot_29 = dot_29 + _shfl_xor_sync_246;
                float _shfl_xor_sync_247 = __shfl_xor_sync(0xFFFFFFFF, dot_29, 4);
                dot_29 = dot_29 + _shfl_xor_sync_247;
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
                    uint32_t _read_special_reg_1272;
                    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(_read_special_reg_1272));
                    int tid_r = (int)_read_special_reg_1272;
#else
                    uint32_t _read_special_reg_248;
                    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(_read_special_reg_248));
                    int tid_r = (int)_read_special_reg_248;
#endif
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
#if __CUDA_ARCH__ == 900
#else
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
#endif
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
#if __CUDA_ARCH__ == 900
#else
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
#endif
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
#if __CUDA_ARCH__ == 900
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long s0_28 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_28 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_28 >> 32ull) ^ h_noise[1] ^ key0_1;
                        h_noise[1] = (unsigned int)s1_28;
                        h_noise[2] = (unsigned int)(s0_28 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)s0_28;
                        unsigned long long s0_29 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_29 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_29 >> 32ull) ^ h_noise[5] ^ key0_1;
                        h_noise[5] = (unsigned int)s1_29;
                        h_noise[6] = (unsigned int)(s0_29 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)s0_29;
                        unsigned long long s0_30 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_30 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_30 >> 32ull) ^ h_noise[9] ^ key0_1;
                        h_noise[9] = (unsigned int)s1_30;
                        h_noise[10] = (unsigned int)(s0_30 >> 32ull) ^ h_noise[11] ^ key1_1;
                        h_noise[11] = (unsigned int)s0_30;
                        unsigned long long s0_31 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_31 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_31 >> 32ull) ^ h_noise[13] ^ key0_1;
                        h_noise[13] = (unsigned int)s1_31;
                        h_noise[14] = (unsigned int)(s0_31 >> 32ull) ^ h_noise[15] ^ key1_1;
                        h_noise[15] = (unsigned int)s0_31;
#else
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
#endif
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
#if __CUDA_ARCH__ == 900
#else
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
#endif
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        #pragma unroll 1
#if __CUDA_ARCH__ == 900
                        for (int _round_2 = 5; _round_2 < philox_rounds; _round_2++) {
#else
                        for (int _round_1 = 5; _round_1 < philox_rounds; _round_1++) {
#endif
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
#if __CUDA_ARCH__ == 900
#else
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
#endif
                            key0_1 = key0_1 + 2654435769u;
                            key1_1 = key1_1 + 3144134277u;
                        }
#if __CUDA_ARCH__ == 900
                        float _abs_1273;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1273) : "f"(S[0]));
                        unsigned int s_63 = __as_u32(mul_f32_rz_noftz(_abs_1273, __uint_as_float(125829120)));
                        unsigned int _min_1274 = ((s_63) < (260046848) ? (s_63) : (260046848));
                        unsigned int h_64 = _min_1274 + (h_noise[0] & 8191) >> 13;
                        unsigned int mag_63 = ((s_63 <= 2139095040) ? h_64 : 32256);
                        float _abs_1275;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1275) : "f"(S[1]));
                        unsigned int s_63_1 = __as_u32(mul_f32_rz_noftz(_abs_1275, __uint_as_float(125829120)));
                        unsigned int _min_1276 = ((s_63_1) < (260046848) ? (s_63_1) : (260046848));
                        unsigned int h_65 = _min_1276 + ((h_noise[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_63_1 = ((s_63_1 <= 2139095040) ? h_65 : 32256);
                        uint32_t mags_31;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_31) : "r"(mag_63), "r"(mag_63_1));
                        uint32_t signs_31;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_31) : "r"(__as_u32(S[0])), "r"(__as_u32(S[1])));
                        unsigned int word_31 = signs_31 & 2147516416u | mags_31;
                        h_words[0] = word_31;
                        float _abs_1277;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1277) : "f"(S[2]));
                        unsigned int s_64 = __as_u32(mul_f32_rz_noftz(_abs_1277, __uint_as_float(125829120)));
                        unsigned int _min_1278 = ((s_64) < (260046848) ? (s_64) : (260046848));
                        unsigned int h_66 = _min_1278 + (h_noise[1] & 8191) >> 13;
                        unsigned int mag_64 = ((s_64 <= 2139095040) ? h_66 : 32256);
                        float _abs_1279;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1279) : "f"(S[3]));
                        unsigned int s_65 = __as_u32(mul_f32_rz_noftz(_abs_1279, __uint_as_float(125829120)));
                        unsigned int _min_1280 = ((s_65) < (260046848) ? (s_65) : (260046848));
                        unsigned int h_67 = _min_1280 + ((h_noise[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_65 = ((s_65 <= 2139095040) ? h_67 : 32256);
                        uint32_t mags_31_1;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_31_1) : "r"(mag_64), "r"(mag_65));
                        uint32_t signs_31_1;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_31_1) : "r"(__as_u32(S[2])), "r"(__as_u32(S[3])));
                        unsigned int word_31_1 = signs_31_1 & 2147516416u | mags_31_1;
                        h_words[1] = word_31_1;
#else
                        uint32_t _cvt_rs_x2_f32_249;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_249) : "f"(S2[0].y), "f"(S2[0].x), "r"(h_noise[0] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_249;
                        uint32_t _cvt_rs_x2_f32_250;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_250) : "f"(S2[1].y), "f"(S2[1].x), "r"(h_noise[1] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_250;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1281;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1281) : "f"(S[8]));
                        unsigned int s_66 = __as_u32(mul_f32_rz_noftz(_abs_1281, __uint_as_float(125829120)));
                        unsigned int _min_1282 = ((s_66) < (260046848) ? (s_66) : (260046848));
                        unsigned int h_68 = _min_1282 + (h_noise[2] & 8191) >> 13;
                        unsigned int mag_66 = ((s_66 <= 2139095040) ? h_68 : 32256);
                        float _abs_1283;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1283) : "f"(S[9]));
                        unsigned int s_67 = __as_u32(mul_f32_rz_noftz(_abs_1283, __uint_as_float(125829120)));
                        unsigned int _min_1284 = ((s_67) < (260046848) ? (s_67) : (260046848));
                        unsigned int h_69 = _min_1284 + ((h_noise[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_67 = ((s_67 <= 2139095040) ? h_69 : 32256);
                        uint32_t mags_32;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_32) : "r"(mag_66), "r"(mag_67));
                        uint32_t signs_32;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_32) : "r"(__as_u32(S[8])), "r"(__as_u32(S[9])));
                        unsigned int word_32 = signs_32 & 2147516416u | mags_32;
                        h_words[0] = word_32;
                        float _abs_1285;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1285) : "f"(S[10]));
                        unsigned int s_68 = __as_u32(mul_f32_rz_noftz(_abs_1285, __uint_as_float(125829120)));
                        unsigned int _min_1286 = ((s_68) < (260046848) ? (s_68) : (260046848));
                        unsigned int h_70 = _min_1286 + (h_noise[3] & 8191) >> 13;
                        unsigned int mag_68 = ((s_68 <= 2139095040) ? h_70 : 32256);
                        float _abs_1287;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1287) : "f"(S[11]));
                        unsigned int s_69 = __as_u32(mul_f32_rz_noftz(_abs_1287, __uint_as_float(125829120)));
                        unsigned int _min_1288 = ((s_69) < (260046848) ? (s_69) : (260046848));
                        unsigned int h_71 = _min_1288 + ((h_noise[3] & 536805376) >> 16) >> 13;
                        unsigned int mag_69 = ((s_69 <= 2139095040) ? h_71 : 32256);
                        uint32_t mags_33;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_33) : "r"(mag_68), "r"(mag_69));
                        uint32_t signs_33;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_33) : "r"(__as_u32(S[10])), "r"(__as_u32(S[11])));
                        unsigned int word_33 = signs_33 & 2147516416u | mags_33;
                        h_words[1] = word_33;
#else
                        uint32_t _cvt_rs_x2_f32_251;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_251) : "f"(S2[4].y), "f"(S2[4].x), "r"(h_noise[2] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_251;
                        uint32_t _cvt_rs_x2_f32_252;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_252) : "f"(S2[5].y), "f"(S2[5].x), "r"(h_noise[3] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_252;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1289;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1289) : "f"(S[4]));
                        unsigned int s_70 = __as_u32(mul_f32_rz_noftz(_abs_1289, __uint_as_float(125829120)));
                        unsigned int _min_1290 = ((s_70) < (260046848) ? (s_70) : (260046848));
                        unsigned int h_72 = _min_1290 + (h_noise[4] & 8191) >> 13;
                        unsigned int mag_70 = ((s_70 <= 2139095040) ? h_72 : 32256);
                        float _abs_1291;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1291) : "f"(S[5]));
                        unsigned int s_71 = __as_u32(mul_f32_rz_noftz(_abs_1291, __uint_as_float(125829120)));
                        unsigned int _min_1292 = ((s_71) < (260046848) ? (s_71) : (260046848));
                        unsigned int h_73 = _min_1292 + ((h_noise[4] & 536805376) >> 16) >> 13;
                        unsigned int mag_71 = ((s_71 <= 2139095040) ? h_73 : 32256);
                        uint32_t mags_34;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_34) : "r"(mag_70), "r"(mag_71));
                        uint32_t signs_34;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_34) : "r"(__as_u32(S[4])), "r"(__as_u32(S[5])));
                        unsigned int word_34 = signs_34 & 2147516416u | mags_34;
                        h_words[0] = word_34;
                        float _abs_1293;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1293) : "f"(S[6]));
                        unsigned int s_72 = __as_u32(mul_f32_rz_noftz(_abs_1293, __uint_as_float(125829120)));
                        unsigned int _min_1294 = ((s_72) < (260046848) ? (s_72) : (260046848));
                        unsigned int h_74 = _min_1294 + (h_noise[5] & 8191) >> 13;
                        unsigned int mag_72 = ((s_72 <= 2139095040) ? h_74 : 32256);
                        float _abs_1295;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1295) : "f"(S[7]));
                        unsigned int s_73 = __as_u32(mul_f32_rz_noftz(_abs_1295, __uint_as_float(125829120)));
                        unsigned int _min_1296 = ((s_73) < (260046848) ? (s_73) : (260046848));
                        unsigned int h_75 = _min_1296 + ((h_noise[5] & 536805376) >> 16) >> 13;
                        unsigned int mag_73 = ((s_73 <= 2139095040) ? h_75 : 32256);
                        uint32_t mags_35;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_35) : "r"(mag_72), "r"(mag_73));
                        uint32_t signs_35;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_35) : "r"(__as_u32(S[6])), "r"(__as_u32(S[7])));
                        unsigned int word_35 = signs_35 & 2147516416u | mags_35;
                        h_words[1] = word_35;
#else
                        uint32_t _cvt_rs_x2_f32_253;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_253) : "f"(S2[2].y), "f"(S2[2].x), "r"(h_noise[4] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_253;
                        uint32_t _cvt_rs_x2_f32_254;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_254) : "f"(S2[3].y), "f"(S2[3].x), "r"(h_noise[5] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_254;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1297;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1297) : "f"(S[12]));
                        unsigned int s_74 = __as_u32(mul_f32_rz_noftz(_abs_1297, __uint_as_float(125829120)));
                        unsigned int _min_1298 = ((s_74) < (260046848) ? (s_74) : (260046848));
                        unsigned int h_76 = _min_1298 + (h_noise[6] & 8191) >> 13;
                        unsigned int mag_74 = ((s_74 <= 2139095040) ? h_76 : 32256);
                        float _abs_1299;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1299) : "f"(S[13]));
                        unsigned int s_75 = __as_u32(mul_f32_rz_noftz(_abs_1299, __uint_as_float(125829120)));
                        unsigned int _min_1300 = ((s_75) < (260046848) ? (s_75) : (260046848));
                        unsigned int h_77 = _min_1300 + ((h_noise[6] & 536805376) >> 16) >> 13;
                        unsigned int mag_75 = ((s_75 <= 2139095040) ? h_77 : 32256);
                        uint32_t mags_36;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_36) : "r"(mag_74), "r"(mag_75));
                        uint32_t signs_36;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_36) : "r"(__as_u32(S[12])), "r"(__as_u32(S[13])));
                        unsigned int word_36 = signs_36 & 2147516416u | mags_36;
                        h_words[0] = word_36;
                        float _abs_1301;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1301) : "f"(S[14]));
                        unsigned int s_76 = __as_u32(mul_f32_rz_noftz(_abs_1301, __uint_as_float(125829120)));
                        unsigned int _min_1302 = ((s_76) < (260046848) ? (s_76) : (260046848));
                        unsigned int h_78 = _min_1302 + (h_noise[7] & 8191) >> 13;
                        unsigned int mag_76 = ((s_76 <= 2139095040) ? h_78 : 32256);
                        float _abs_1303;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1303) : "f"(S[15]));
                        unsigned int s_77 = __as_u32(mul_f32_rz_noftz(_abs_1303, __uint_as_float(125829120)));
                        unsigned int _min_1304 = ((s_77) < (260046848) ? (s_77) : (260046848));
                        unsigned int h_79 = _min_1304 + ((h_noise[7] & 536805376) >> 16) >> 13;
                        unsigned int mag_77 = ((s_77 <= 2139095040) ? h_79 : 32256);
                        uint32_t mags_37;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_37) : "r"(mag_76), "r"(mag_77));
                        uint32_t signs_37;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_37) : "r"(__as_u32(S[14])), "r"(__as_u32(S[15])));
                        unsigned int word_37 = signs_37 & 2147516416u | mags_37;
                        h_words[1] = word_37;
#else
                        uint32_t _cvt_rs_x2_f32_255;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_255) : "f"(S2[6].y), "f"(S2[6].x), "r"(h_noise[6] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_255;
                        uint32_t _cvt_rs_x2_f32_256;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_256) : "f"(S2[7].y), "f"(S2[7].x), "r"(h_noise[7] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_256;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1305;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1305) : "f"(S[16]));
                        unsigned int s_78 = __as_u32(mul_f32_rz_noftz(_abs_1305, __uint_as_float(125829120)));
                        unsigned int _min_1306 = ((s_78) < (260046848) ? (s_78) : (260046848));
                        unsigned int h_80 = _min_1306 + (h_noise[8] & 8191) >> 13;
                        unsigned int mag_78 = ((s_78 <= 2139095040) ? h_80 : 32256);
                        float _abs_1307;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1307) : "f"(S[17]));
                        unsigned int s_79 = __as_u32(mul_f32_rz_noftz(_abs_1307, __uint_as_float(125829120)));
                        unsigned int _min_1308 = ((s_79) < (260046848) ? (s_79) : (260046848));
                        unsigned int h_81 = _min_1308 + ((h_noise[8] & 536805376) >> 16) >> 13;
                        unsigned int mag_79 = ((s_79 <= 2139095040) ? h_81 : 32256);
                        uint32_t mags_38;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_38) : "r"(mag_78), "r"(mag_79));
                        uint32_t signs_38;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_38) : "r"(__as_u32(S[16])), "r"(__as_u32(S[17])));
                        unsigned int word_38 = signs_38 & 2147516416u | mags_38;
                        h_words[0] = word_38;
                        float _abs_1309;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1309) : "f"(S[18]));
                        unsigned int s_80 = __as_u32(mul_f32_rz_noftz(_abs_1309, __uint_as_float(125829120)));
                        unsigned int _min_1310 = ((s_80) < (260046848) ? (s_80) : (260046848));
                        unsigned int h_82 = _min_1310 + (h_noise[9] & 8191) >> 13;
                        unsigned int mag_80 = ((s_80 <= 2139095040) ? h_82 : 32256);
                        float _abs_1311;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1311) : "f"(S[19]));
                        unsigned int s_81 = __as_u32(mul_f32_rz_noftz(_abs_1311, __uint_as_float(125829120)));
                        unsigned int _min_1312 = ((s_81) < (260046848) ? (s_81) : (260046848));
                        unsigned int h_83 = _min_1312 + ((h_noise[9] & 536805376) >> 16) >> 13;
                        unsigned int mag_81 = ((s_81 <= 2139095040) ? h_83 : 32256);
                        uint32_t mags_39;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_39) : "r"(mag_80), "r"(mag_81));
                        uint32_t signs_39;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_39) : "r"(__as_u32(S[18])), "r"(__as_u32(S[19])));
                        unsigned int word_39 = signs_39 & 2147516416u | mags_39;
                        h_words[1] = word_39;
#else
                        uint32_t _cvt_rs_x2_f32_257;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_257) : "f"(S2[8].y), "f"(S2[8].x), "r"(h_noise[8] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_257;
                        uint32_t _cvt_rs_x2_f32_258;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_258) : "f"(S2[9].y), "f"(S2[9].x), "r"(h_noise[9] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_258;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1313;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1313) : "f"(S[24]));
                        unsigned int s_82 = __as_u32(mul_f32_rz_noftz(_abs_1313, __uint_as_float(125829120)));
                        unsigned int _min_1314 = ((s_82) < (260046848) ? (s_82) : (260046848));
                        unsigned int h_84 = _min_1314 + (h_noise[10] & 8191) >> 13;
                        unsigned int mag_82 = ((s_82 <= 2139095040) ? h_84 : 32256);
                        float _abs_1315;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1315) : "f"(S[25]));
                        unsigned int s_83 = __as_u32(mul_f32_rz_noftz(_abs_1315, __uint_as_float(125829120)));
                        unsigned int _min_1316 = ((s_83) < (260046848) ? (s_83) : (260046848));
                        unsigned int h_85 = _min_1316 + ((h_noise[10] & 536805376) >> 16) >> 13;
                        unsigned int mag_83 = ((s_83 <= 2139095040) ? h_85 : 32256);
                        uint32_t mags_40;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_40) : "r"(mag_82), "r"(mag_83));
                        uint32_t signs_40;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_40) : "r"(__as_u32(S[24])), "r"(__as_u32(S[25])));
                        unsigned int word_40 = signs_40 & 2147516416u | mags_40;
                        h_words[0] = word_40;
                        float _abs_1317;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1317) : "f"(S[26]));
                        unsigned int s_84 = __as_u32(mul_f32_rz_noftz(_abs_1317, __uint_as_float(125829120)));
                        unsigned int _min_1318 = ((s_84) < (260046848) ? (s_84) : (260046848));
                        unsigned int h_86 = _min_1318 + (h_noise[11] & 8191) >> 13;
                        unsigned int mag_84 = ((s_84 <= 2139095040) ? h_86 : 32256);
                        float _abs_1319;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1319) : "f"(S[27]));
                        unsigned int s_85 = __as_u32(mul_f32_rz_noftz(_abs_1319, __uint_as_float(125829120)));
                        unsigned int _min_1320 = ((s_85) < (260046848) ? (s_85) : (260046848));
                        unsigned int h_87 = _min_1320 + ((h_noise[11] & 536805376) >> 16) >> 13;
                        unsigned int mag_85 = ((s_85 <= 2139095040) ? h_87 : 32256);
                        uint32_t mags_41;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_41) : "r"(mag_84), "r"(mag_85));
                        uint32_t signs_41;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_41) : "r"(__as_u32(S[26])), "r"(__as_u32(S[27])));
                        unsigned int word_41 = signs_41 & 2147516416u | mags_41;
                        h_words[1] = word_41;
#else
                        uint32_t _cvt_rs_x2_f32_259;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_259) : "f"(S2[12].y), "f"(S2[12].x), "r"(h_noise[10] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_259;
                        uint32_t _cvt_rs_x2_f32_260;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_260) : "f"(S2[13].y), "f"(S2[13].x), "r"(h_noise[11] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_260;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1321;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1321) : "f"(S[20]));
                        unsigned int s_86 = __as_u32(mul_f32_rz_noftz(_abs_1321, __uint_as_float(125829120)));
                        unsigned int _min_1322 = ((s_86) < (260046848) ? (s_86) : (260046848));
                        unsigned int h_88 = _min_1322 + (h_noise[12] & 8191) >> 13;
                        unsigned int mag_86 = ((s_86 <= 2139095040) ? h_88 : 32256);
                        float _abs_1323;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1323) : "f"(S[21]));
                        unsigned int s_87 = __as_u32(mul_f32_rz_noftz(_abs_1323, __uint_as_float(125829120)));
                        unsigned int _min_1324 = ((s_87) < (260046848) ? (s_87) : (260046848));
                        unsigned int h_89 = _min_1324 + ((h_noise[12] & 536805376) >> 16) >> 13;
                        unsigned int mag_87 = ((s_87 <= 2139095040) ? h_89 : 32256);
                        uint32_t mags_42;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_42) : "r"(mag_86), "r"(mag_87));
                        uint32_t signs_42;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_42) : "r"(__as_u32(S[20])), "r"(__as_u32(S[21])));
                        unsigned int word_42 = signs_42 & 2147516416u | mags_42;
                        h_words[0] = word_42;
                        float _abs_1325;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1325) : "f"(S[22]));
                        unsigned int s_88 = __as_u32(mul_f32_rz_noftz(_abs_1325, __uint_as_float(125829120)));
                        unsigned int _min_1326 = ((s_88) < (260046848) ? (s_88) : (260046848));
                        unsigned int h_90 = _min_1326 + (h_noise[13] & 8191) >> 13;
                        unsigned int mag_88 = ((s_88 <= 2139095040) ? h_90 : 32256);
                        float _abs_1327;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1327) : "f"(S[23]));
                        unsigned int s_89 = __as_u32(mul_f32_rz_noftz(_abs_1327, __uint_as_float(125829120)));
                        unsigned int _min_1328 = ((s_89) < (260046848) ? (s_89) : (260046848));
                        unsigned int h_91 = _min_1328 + ((h_noise[13] & 536805376) >> 16) >> 13;
                        unsigned int mag_89 = ((s_89 <= 2139095040) ? h_91 : 32256);
                        uint32_t mags_43;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_43) : "r"(mag_88), "r"(mag_89));
                        uint32_t signs_43;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_43) : "r"(__as_u32(S[22])), "r"(__as_u32(S[23])));
                        unsigned int word_43 = signs_43 & 2147516416u | mags_43;
                        h_words[1] = word_43;
#else
                        uint32_t _cvt_rs_x2_f32_261;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_261) : "f"(S2[10].y), "f"(S2[10].x), "r"(h_noise[12] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_261;
                        uint32_t _cvt_rs_x2_f32_262;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_262) : "f"(S2[11].y), "f"(S2[11].x), "r"(h_noise[13] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_262;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1329;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1329) : "f"(S[28]));
                        unsigned int s_90 = __as_u32(mul_f32_rz_noftz(_abs_1329, __uint_as_float(125829120)));
                        unsigned int _min_1330 = ((s_90) < (260046848) ? (s_90) : (260046848));
                        unsigned int h_92 = _min_1330 + (h_noise[14] & 8191) >> 13;
                        unsigned int mag_90 = ((s_90 <= 2139095040) ? h_92 : 32256);
                        float _abs_1331;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1331) : "f"(S[29]));
                        unsigned int s_91 = __as_u32(mul_f32_rz_noftz(_abs_1331, __uint_as_float(125829120)));
                        unsigned int _min_1332 = ((s_91) < (260046848) ? (s_91) : (260046848));
                        unsigned int h_93 = _min_1332 + ((h_noise[14] & 536805376) >> 16) >> 13;
                        unsigned int mag_91 = ((s_91 <= 2139095040) ? h_93 : 32256);
                        uint32_t mags_44;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_44) : "r"(mag_90), "r"(mag_91));
                        uint32_t signs_44;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_44) : "r"(__as_u32(S[28])), "r"(__as_u32(S[29])));
                        unsigned int word_44 = signs_44 & 2147516416u | mags_44;
                        h_words[0] = word_44;
                        float _abs_1333;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1333) : "f"(S[30]));
                        unsigned int s_92 = __as_u32(mul_f32_rz_noftz(_abs_1333, __uint_as_float(125829120)));
                        unsigned int _min_1334 = ((s_92) < (260046848) ? (s_92) : (260046848));
                        unsigned int h_94 = _min_1334 + (h_noise[15] & 8191) >> 13;
                        unsigned int mag_92 = ((s_92 <= 2139095040) ? h_94 : 32256);
                        float _abs_1335;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1335) : "f"(S[31]));
                        unsigned int s_93 = __as_u32(mul_f32_rz_noftz(_abs_1335, __uint_as_float(125829120)));
                        unsigned int _min_1336 = ((s_93) < (260046848) ? (s_93) : (260046848));
                        unsigned int h_95 = _min_1336 + ((h_noise[15] & 536805376) >> 16) >> 13;
                        unsigned int mag_93 = ((s_93 <= 2139095040) ? h_95 : 32256);
                        uint32_t mags_45;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_45) : "r"(mag_92), "r"(mag_93));
                        uint32_t signs_45;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_45) : "r"(__as_u32(S[30])), "r"(__as_u32(S[31])));
                        unsigned int word_45 = signs_45 & 2147516416u | mags_45;
                        h_words[1] = word_45;
#else
                        uint32_t _cvt_rs_x2_f32_263;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_263) : "f"(S2[14].y), "f"(S2[14].x), "r"(h_noise[14] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_263;
                        uint32_t _cvt_rs_x2_f32_264;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_264) : "f"(S2[15].y), "f"(S2[15].x), "r"(h_noise[15] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_264;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        unsigned int key0_1_1 = seed_lo;
                        unsigned int key1_1_1 = seed_hi;
                        long long o_12 = noise_at + 2ll * state_stride_p2;
                        unsigned long long p0_12 = (unsigned long long)(unsigned int)o_12 * 3528531795ull;
                        h_noise[0] = (unsigned int)(o_12 >> 32ll) ^ key0_1_1;
                        h_noise[2] = (unsigned int)(p0_12 >> 32ull) ^ key1_1_1;
                        h_noise[3] = (unsigned int)p0_12;
                        long long o_13 = noise_at + 2ll * state_stride_p2 + 32ll;
                        unsigned long long p0_13 = (unsigned long long)(unsigned int)o_13 * 3528531795ull;
                        h_noise[4] = (unsigned int)(o_13 >> 32ll) ^ key0_1_1;
                        h_noise[6] = (unsigned int)(p0_13 >> 32ull) ^ key1_1_1;
                        h_noise[7] = (unsigned int)p0_13;
                        long long o_14 = noise_at + 3ll * state_stride_p2;
                        unsigned long long p0_14 = (unsigned long long)(unsigned int)o_14 * 3528531795ull;
                        h_noise[8] = (unsigned int)(o_14 >> 32ll) ^ key0_1_1;
                        h_noise[10] = (unsigned int)(p0_14 >> 32ull) ^ key1_1_1;
                        h_noise[11] = (unsigned int)p0_14;
                        long long o_15 = noise_at + 3ll * state_stride_p2 + 32ll;
                        unsigned long long p0_15 = (unsigned long long)(unsigned int)o_15 * 3528531795ull;
                        h_noise[12] = (unsigned int)(o_15 >> 32ll) ^ key0_1_1;
                        h_noise[14] = (unsigned int)(p0_15 >> 32ull) ^ key1_1_1;
                        h_noise[15] = (unsigned int)p0_15;
                        key0_1_1 = key0_1_1 + 2654435769u;
                        key1_1_1 = key1_1_1 + 3144134277u;
                        unsigned long long q0_12 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long q1_12 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(q1_12 >> 32ull) ^ key0_1_1;
                        h_noise[1] = (unsigned int)q1_12;
                        h_noise[2] = (unsigned int)(q0_12 >> 32ull) ^ h_noise[3] ^ key1_1_1;
                        h_noise[3] = (unsigned int)q0_12;
                        unsigned long long q0_13 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long q1_13 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(q1_13 >> 32ull) ^ key0_1_1;
                        h_noise[5] = (unsigned int)q1_13;
                        h_noise[6] = (unsigned int)(q0_13 >> 32ull) ^ h_noise[7] ^ key1_1_1;
                        h_noise[7] = (unsigned int)q0_13;
                        unsigned long long q0_14 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long q1_14 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(q1_14 >> 32ull) ^ key0_1_1;
                        h_noise[9] = (unsigned int)q1_14;
                        h_noise[10] = (unsigned int)(q0_14 >> 32ull) ^ h_noise[11] ^ key1_1_1;
                        h_noise[11] = (unsigned int)q0_14;
                        unsigned long long q0_15 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long q1_15 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(q1_15 >> 32ull) ^ key0_1_1;
                        h_noise[13] = (unsigned int)q1_15;
                        h_noise[14] = (unsigned int)(q0_15 >> 32ull) ^ h_noise[15] ^ key1_1_1;
                        h_noise[15] = (unsigned int)q0_15;
                        key0_1_1 = key0_1_1 + 2654435769u;
                        key1_1_1 = key1_1_1 + 3144134277u;
                        unsigned long long s0_36 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_36 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_36 >> 32ull) ^ h_noise[1] ^ key0_1_1;
                        h_noise[1] = (unsigned int)s1_36;
                        h_noise[2] = (unsigned int)(s0_36 >> 32ull) ^ h_noise[3] ^ key1_1_1;
                        h_noise[3] = (unsigned int)s0_36;
                        unsigned long long s0_37 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_37 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_37 >> 32ull) ^ h_noise[5] ^ key0_1_1;
                        h_noise[5] = (unsigned int)s1_37;
                        h_noise[6] = (unsigned int)(s0_37 >> 32ull) ^ h_noise[7] ^ key1_1_1;
                        h_noise[7] = (unsigned int)s0_37;
                        unsigned long long s0_38 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_38 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_38 >> 32ull) ^ h_noise[9] ^ key0_1_1;
                        h_noise[9] = (unsigned int)s1_38;
                        h_noise[10] = (unsigned int)(s0_38 >> 32ull) ^ h_noise[11] ^ key1_1_1;
                        h_noise[11] = (unsigned int)s0_38;
                        unsigned long long s0_39 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_39 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_39 >> 32ull) ^ h_noise[13] ^ key0_1_1;
                        h_noise[13] = (unsigned int)s1_39;
                        h_noise[14] = (unsigned int)(s0_39 >> 32ull) ^ h_noise[15] ^ key1_1_1;
                        h_noise[15] = (unsigned int)s0_39;
                        key0_1_1 = key0_1_1 + 2654435769u;
                        key1_1_1 = key1_1_1 + 3144134277u;
                        unsigned long long s0_40 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_40 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_40 >> 32ull) ^ h_noise[1] ^ key0_1_1;
                        h_noise[1] = (unsigned int)s1_40;
                        h_noise[2] = (unsigned int)(s0_40 >> 32ull) ^ h_noise[3] ^ key1_1_1;
                        h_noise[3] = (unsigned int)s0_40;
                        unsigned long long s0_41 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_41 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_41 >> 32ull) ^ h_noise[5] ^ key0_1_1;
                        h_noise[5] = (unsigned int)s1_41;
                        h_noise[6] = (unsigned int)(s0_41 >> 32ull) ^ h_noise[7] ^ key1_1_1;
                        h_noise[7] = (unsigned int)s0_41;
                        unsigned long long s0_42 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_42 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_42 >> 32ull) ^ h_noise[9] ^ key0_1_1;
                        h_noise[9] = (unsigned int)s1_42;
                        h_noise[10] = (unsigned int)(s0_42 >> 32ull) ^ h_noise[11] ^ key1_1_1;
                        h_noise[11] = (unsigned int)s0_42;
                        unsigned long long s0_43 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_43 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_43 >> 32ull) ^ h_noise[13] ^ key0_1_1;
                        h_noise[13] = (unsigned int)s1_43;
                        h_noise[14] = (unsigned int)(s0_43 >> 32ull) ^ h_noise[15] ^ key1_1_1;
                        h_noise[15] = (unsigned int)s0_43;
                        key0_1_1 = key0_1_1 + 2654435769u;
                        key1_1_1 = key1_1_1 + 3144134277u;
                        unsigned long long s0_44 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_44 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_44 >> 32ull) ^ h_noise[1] ^ key0_1_1;
                        h_noise[1] = (unsigned int)s1_44;
                        h_noise[2] = (unsigned int)(s0_44 >> 32ull) ^ h_noise[3] ^ key1_1_1;
                        h_noise[3] = (unsigned int)s0_44;
                        unsigned long long s0_45 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_45 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_45 >> 32ull) ^ h_noise[5] ^ key0_1_1;
                        h_noise[5] = (unsigned int)s1_45;
                        h_noise[6] = (unsigned int)(s0_45 >> 32ull) ^ h_noise[7] ^ key1_1_1;
                        h_noise[7] = (unsigned int)s0_45;
                        unsigned long long s0_46 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_46 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_46 >> 32ull) ^ h_noise[9] ^ key0_1_1;
                        h_noise[9] = (unsigned int)s1_46;
                        h_noise[10] = (unsigned int)(s0_46 >> 32ull) ^ h_noise[11] ^ key1_1_1;
                        h_noise[11] = (unsigned int)s0_46;
                        unsigned long long s0_47 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_47 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_47 >> 32ull) ^ h_noise[13] ^ key0_1_1;
                        h_noise[13] = (unsigned int)s1_47;
                        h_noise[14] = (unsigned int)(s0_47 >> 32ull) ^ h_noise[15] ^ key1_1_1;
                        h_noise[15] = (unsigned int)s0_47;
                        key0_1_1 = key0_1_1 + 2654435769u;
                        key1_1_1 = key1_1_1 + 3144134277u;
                        #pragma unroll 1
                        for (int _round_3 = 5; _round_3 < philox_rounds; _round_3++) {
                            unsigned long long t0_12 = (unsigned long long)h_noise[0] * 3528531795ull;
                            unsigned long long t1_12 = (unsigned long long)h_noise[2] * 3449720151ull;
                            h_noise[0] = (unsigned int)(t1_12 >> 32ull) ^ h_noise[1] ^ key0_1_1;
                            h_noise[1] = (unsigned int)t1_12;
                            h_noise[2] = (unsigned int)(t0_12 >> 32ull) ^ h_noise[3] ^ key1_1_1;
                            h_noise[3] = (unsigned int)t0_12;
                            unsigned long long t0_13 = (unsigned long long)h_noise[4] * 3528531795ull;
                            unsigned long long t1_13 = (unsigned long long)h_noise[6] * 3449720151ull;
                            h_noise[4] = (unsigned int)(t1_13 >> 32ull) ^ h_noise[5] ^ key0_1_1;
                            h_noise[5] = (unsigned int)t1_13;
                            h_noise[6] = (unsigned int)(t0_13 >> 32ull) ^ h_noise[7] ^ key1_1_1;
                            h_noise[7] = (unsigned int)t0_13;
                            unsigned long long t0_14 = (unsigned long long)h_noise[8] * 3528531795ull;
                            unsigned long long t1_14 = (unsigned long long)h_noise[10] * 3449720151ull;
                            h_noise[8] = (unsigned int)(t1_14 >> 32ull) ^ h_noise[9] ^ key0_1_1;
                            h_noise[9] = (unsigned int)t1_14;
                            h_noise[10] = (unsigned int)(t0_14 >> 32ull) ^ h_noise[11] ^ key1_1_1;
                            h_noise[11] = (unsigned int)t0_14;
                            unsigned long long t0_15 = (unsigned long long)h_noise[12] * 3528531795ull;
                            unsigned long long t1_15 = (unsigned long long)h_noise[14] * 3449720151ull;
                            h_noise[12] = (unsigned int)(t1_15 >> 32ull) ^ h_noise[13] ^ key0_1_1;
                            h_noise[13] = (unsigned int)t1_15;
                            h_noise[14] = (unsigned int)(t0_15 >> 32ull) ^ h_noise[15] ^ key1_1_1;
                            h_noise[15] = (unsigned int)t0_15;
                            key0_1_1 = key0_1_1 + 2654435769u;
                            key1_1_1 = key1_1_1 + 3144134277u;
                        }
                        float _abs_1337;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1337) : "f"(S[32]));
                        unsigned int s_94 = __as_u32(mul_f32_rz_noftz(_abs_1337, __uint_as_float(125829120)));
                        unsigned int _min_1338 = ((s_94) < (260046848) ? (s_94) : (260046848));
                        unsigned int h_96 = _min_1338 + (h_noise[0] & 8191) >> 13;
                        unsigned int mag_94 = ((s_94 <= 2139095040) ? h_96 : 32256);
                        float _abs_1339;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1339) : "f"(S[33]));
                        unsigned int s_95 = __as_u32(mul_f32_rz_noftz(_abs_1339, __uint_as_float(125829120)));
                        unsigned int _min_1340 = ((s_95) < (260046848) ? (s_95) : (260046848));
                        unsigned int h_97 = _min_1340 + ((h_noise[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_95 = ((s_95 <= 2139095040) ? h_97 : 32256);
                        uint32_t mags_46;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_46) : "r"(mag_94), "r"(mag_95));
                        uint32_t signs_46;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_46) : "r"(__as_u32(S[32])), "r"(__as_u32(S[33])));
                        unsigned int word_46 = signs_46 & 2147516416u | mags_46;
                        h_words[0] = word_46;
                        float _abs_1341;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1341) : "f"(S[34]));
                        unsigned int s_96 = __as_u32(mul_f32_rz_noftz(_abs_1341, __uint_as_float(125829120)));
                        unsigned int _min_1342 = ((s_96) < (260046848) ? (s_96) : (260046848));
                        unsigned int h_98 = _min_1342 + (h_noise[1] & 8191) >> 13;
                        unsigned int mag_96 = ((s_96 <= 2139095040) ? h_98 : 32256);
                        float _abs_1343;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1343) : "f"(S[35]));
                        unsigned int s_97 = __as_u32(mul_f32_rz_noftz(_abs_1343, __uint_as_float(125829120)));
                        unsigned int _min_1344 = ((s_97) < (260046848) ? (s_97) : (260046848));
                        unsigned int h_99 = _min_1344 + ((h_noise[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_97 = ((s_97 <= 2139095040) ? h_99 : 32256);
                        uint32_t mags_47;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_47) : "r"(mag_96), "r"(mag_97));
                        uint32_t signs_47;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_47) : "r"(__as_u32(S[34])), "r"(__as_u32(S[35])));
                        unsigned int word_47 = signs_47 & 2147516416u | mags_47;
                        h_words[1] = word_47;
#else
                        uint32_t _cvt_rs_x2_f32_265;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_265) : "f"(S2[16].y), "f"(S2[16].x), "r"(h_noise[16] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_265;
                        uint32_t _cvt_rs_x2_f32_266;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_266) : "f"(S2[17].y), "f"(S2[17].x), "r"(h_noise[17] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_266;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1345;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1345) : "f"(S[40]));
                        unsigned int s_98 = __as_u32(mul_f32_rz_noftz(_abs_1345, __uint_as_float(125829120)));
                        unsigned int _min_1346 = ((s_98) < (260046848) ? (s_98) : (260046848));
                        unsigned int h_100 = _min_1346 + (h_noise[2] & 8191) >> 13;
                        unsigned int mag_98 = ((s_98 <= 2139095040) ? h_100 : 32256);
                        float _abs_1347;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1347) : "f"(S[41]));
                        unsigned int s_99 = __as_u32(mul_f32_rz_noftz(_abs_1347, __uint_as_float(125829120)));
                        unsigned int _min_1348 = ((s_99) < (260046848) ? (s_99) : (260046848));
                        unsigned int h_101 = _min_1348 + ((h_noise[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_99 = ((s_99 <= 2139095040) ? h_101 : 32256);
                        uint32_t mags_48;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_48) : "r"(mag_98), "r"(mag_99));
                        uint32_t signs_48;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_48) : "r"(__as_u32(S[40])), "r"(__as_u32(S[41])));
                        unsigned int word_48 = signs_48 & 2147516416u | mags_48;
                        h_words[0] = word_48;
                        float _abs_1349;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1349) : "f"(S[42]));
                        unsigned int s_100 = __as_u32(mul_f32_rz_noftz(_abs_1349, __uint_as_float(125829120)));
                        unsigned int _min_1350 = ((s_100) < (260046848) ? (s_100) : (260046848));
                        unsigned int h_102 = _min_1350 + (h_noise[3] & 8191) >> 13;
                        unsigned int mag_100 = ((s_100 <= 2139095040) ? h_102 : 32256);
                        float _abs_1351;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1351) : "f"(S[43]));
                        unsigned int s_101 = __as_u32(mul_f32_rz_noftz(_abs_1351, __uint_as_float(125829120)));
                        unsigned int _min_1352 = ((s_101) < (260046848) ? (s_101) : (260046848));
                        unsigned int h_103 = _min_1352 + ((h_noise[3] & 536805376) >> 16) >> 13;
                        unsigned int mag_101 = ((s_101 <= 2139095040) ? h_103 : 32256);
                        uint32_t mags_49;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_49) : "r"(mag_100), "r"(mag_101));
                        uint32_t signs_49;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_49) : "r"(__as_u32(S[42])), "r"(__as_u32(S[43])));
                        unsigned int word_49 = signs_49 & 2147516416u | mags_49;
                        h_words[1] = word_49;
#else
                        uint32_t _cvt_rs_x2_f32_267;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_267) : "f"(S2[20].y), "f"(S2[20].x), "r"(h_noise[18] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_267;
                        uint32_t _cvt_rs_x2_f32_268;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_268) : "f"(S2[21].y), "f"(S2[21].x), "r"(h_noise[19] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_268;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1353;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1353) : "f"(S[36]));
                        unsigned int s_102 = __as_u32(mul_f32_rz_noftz(_abs_1353, __uint_as_float(125829120)));
                        unsigned int _min_1354 = ((s_102) < (260046848) ? (s_102) : (260046848));
                        unsigned int h_104 = _min_1354 + (h_noise[4] & 8191) >> 13;
                        unsigned int mag_102 = ((s_102 <= 2139095040) ? h_104 : 32256);
                        float _abs_1355;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1355) : "f"(S[37]));
                        unsigned int s_103 = __as_u32(mul_f32_rz_noftz(_abs_1355, __uint_as_float(125829120)));
                        unsigned int _min_1356 = ((s_103) < (260046848) ? (s_103) : (260046848));
                        unsigned int h_105 = _min_1356 + ((h_noise[4] & 536805376) >> 16) >> 13;
                        unsigned int mag_103 = ((s_103 <= 2139095040) ? h_105 : 32256);
                        uint32_t mags_50;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_50) : "r"(mag_102), "r"(mag_103));
                        uint32_t signs_50;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_50) : "r"(__as_u32(S[36])), "r"(__as_u32(S[37])));
                        unsigned int word_50 = signs_50 & 2147516416u | mags_50;
                        h_words[0] = word_50;
                        float _abs_1357;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1357) : "f"(S[38]));
                        unsigned int s_104 = __as_u32(mul_f32_rz_noftz(_abs_1357, __uint_as_float(125829120)));
                        unsigned int _min_1358 = ((s_104) < (260046848) ? (s_104) : (260046848));
                        unsigned int h_106 = _min_1358 + (h_noise[5] & 8191) >> 13;
                        unsigned int mag_104 = ((s_104 <= 2139095040) ? h_106 : 32256);
                        float _abs_1359;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1359) : "f"(S[39]));
                        unsigned int s_105 = __as_u32(mul_f32_rz_noftz(_abs_1359, __uint_as_float(125829120)));
                        unsigned int _min_1360 = ((s_105) < (260046848) ? (s_105) : (260046848));
                        unsigned int h_107 = _min_1360 + ((h_noise[5] & 536805376) >> 16) >> 13;
                        unsigned int mag_105 = ((s_105 <= 2139095040) ? h_107 : 32256);
                        uint32_t mags_51;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_51) : "r"(mag_104), "r"(mag_105));
                        uint32_t signs_51;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_51) : "r"(__as_u32(S[38])), "r"(__as_u32(S[39])));
                        unsigned int word_51 = signs_51 & 2147516416u | mags_51;
                        h_words[1] = word_51;
#else
                        uint32_t _cvt_rs_x2_f32_269;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_269) : "f"(S2[18].y), "f"(S2[18].x), "r"(h_noise[20] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_269;
                        uint32_t _cvt_rs_x2_f32_270;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_270) : "f"(S2[19].y), "f"(S2[19].x), "r"(h_noise[21] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_270;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1361;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1361) : "f"(S[44]));
                        unsigned int s_106 = __as_u32(mul_f32_rz_noftz(_abs_1361, __uint_as_float(125829120)));
                        unsigned int _min_1362 = ((s_106) < (260046848) ? (s_106) : (260046848));
                        unsigned int h_108 = _min_1362 + (h_noise[6] & 8191) >> 13;
                        unsigned int mag_106 = ((s_106 <= 2139095040) ? h_108 : 32256);
                        float _abs_1363;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1363) : "f"(S[45]));
                        unsigned int s_107 = __as_u32(mul_f32_rz_noftz(_abs_1363, __uint_as_float(125829120)));
                        unsigned int _min_1364 = ((s_107) < (260046848) ? (s_107) : (260046848));
                        unsigned int h_109 = _min_1364 + ((h_noise[6] & 536805376) >> 16) >> 13;
                        unsigned int mag_107 = ((s_107 <= 2139095040) ? h_109 : 32256);
                        uint32_t mags_52;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_52) : "r"(mag_106), "r"(mag_107));
                        uint32_t signs_52;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_52) : "r"(__as_u32(S[44])), "r"(__as_u32(S[45])));
                        unsigned int word_52 = signs_52 & 2147516416u | mags_52;
                        h_words[0] = word_52;
                        float _abs_1365;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1365) : "f"(S[46]));
                        unsigned int s_108 = __as_u32(mul_f32_rz_noftz(_abs_1365, __uint_as_float(125829120)));
                        unsigned int _min_1366 = ((s_108) < (260046848) ? (s_108) : (260046848));
                        unsigned int h_110 = _min_1366 + (h_noise[7] & 8191) >> 13;
                        unsigned int mag_108 = ((s_108 <= 2139095040) ? h_110 : 32256);
                        float _abs_1367;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1367) : "f"(S[47]));
                        unsigned int s_109 = __as_u32(mul_f32_rz_noftz(_abs_1367, __uint_as_float(125829120)));
                        unsigned int _min_1368 = ((s_109) < (260046848) ? (s_109) : (260046848));
                        unsigned int h_111 = _min_1368 + ((h_noise[7] & 536805376) >> 16) >> 13;
                        unsigned int mag_109 = ((s_109 <= 2139095040) ? h_111 : 32256);
                        uint32_t mags_53;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_53) : "r"(mag_108), "r"(mag_109));
                        uint32_t signs_53;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_53) : "r"(__as_u32(S[46])), "r"(__as_u32(S[47])));
                        unsigned int word_53 = signs_53 & 2147516416u | mags_53;
                        h_words[1] = word_53;
#else
                        uint32_t _cvt_rs_x2_f32_271;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_271) : "f"(S2[22].y), "f"(S2[22].x), "r"(h_noise[22] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_271;
                        uint32_t _cvt_rs_x2_f32_272;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_272) : "f"(S2[23].y), "f"(S2[23].x), "r"(h_noise[23] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_272;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1369;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1369) : "f"(S[48]));
                        unsigned int s_110 = __as_u32(mul_f32_rz_noftz(_abs_1369, __uint_as_float(125829120)));
                        unsigned int _min_1370 = ((s_110) < (260046848) ? (s_110) : (260046848));
                        unsigned int h_112 = _min_1370 + (h_noise[8] & 8191) >> 13;
                        unsigned int mag_110 = ((s_110 <= 2139095040) ? h_112 : 32256);
                        float _abs_1371;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1371) : "f"(S[49]));
                        unsigned int s_111 = __as_u32(mul_f32_rz_noftz(_abs_1371, __uint_as_float(125829120)));
                        unsigned int _min_1372 = ((s_111) < (260046848) ? (s_111) : (260046848));
                        unsigned int h_113 = _min_1372 + ((h_noise[8] & 536805376) >> 16) >> 13;
                        unsigned int mag_111 = ((s_111 <= 2139095040) ? h_113 : 32256);
                        uint32_t mags_54;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_54) : "r"(mag_110), "r"(mag_111));
                        uint32_t signs_54;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_54) : "r"(__as_u32(S[48])), "r"(__as_u32(S[49])));
                        unsigned int word_54 = signs_54 & 2147516416u | mags_54;
                        h_words[0] = word_54;
                        float _abs_1373;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1373) : "f"(S[50]));
                        unsigned int s_112 = __as_u32(mul_f32_rz_noftz(_abs_1373, __uint_as_float(125829120)));
                        unsigned int _min_1374 = ((s_112) < (260046848) ? (s_112) : (260046848));
                        unsigned int h_114 = _min_1374 + (h_noise[9] & 8191) >> 13;
                        unsigned int mag_112 = ((s_112 <= 2139095040) ? h_114 : 32256);
                        float _abs_1375;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1375) : "f"(S[51]));
                        unsigned int s_113 = __as_u32(mul_f32_rz_noftz(_abs_1375, __uint_as_float(125829120)));
                        unsigned int _min_1376 = ((s_113) < (260046848) ? (s_113) : (260046848));
                        unsigned int h_115 = _min_1376 + ((h_noise[9] & 536805376) >> 16) >> 13;
                        unsigned int mag_113 = ((s_113 <= 2139095040) ? h_115 : 32256);
                        uint32_t mags_55;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_55) : "r"(mag_112), "r"(mag_113));
                        uint32_t signs_55;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_55) : "r"(__as_u32(S[50])), "r"(__as_u32(S[51])));
                        unsigned int word_55 = signs_55 & 2147516416u | mags_55;
                        h_words[1] = word_55;
#else
                        uint32_t _cvt_rs_x2_f32_273;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_273) : "f"(S2[24].y), "f"(S2[24].x), "r"(h_noise[24] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_273;
                        uint32_t _cvt_rs_x2_f32_274;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_274) : "f"(S2[25].y), "f"(S2[25].x), "r"(h_noise[25] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_274;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1377;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1377) : "f"(S[56]));
                        unsigned int s_114 = __as_u32(mul_f32_rz_noftz(_abs_1377, __uint_as_float(125829120)));
                        unsigned int _min_1378 = ((s_114) < (260046848) ? (s_114) : (260046848));
                        unsigned int h_116 = _min_1378 + (h_noise[10] & 8191) >> 13;
                        unsigned int mag_114 = ((s_114 <= 2139095040) ? h_116 : 32256);
                        float _abs_1379;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1379) : "f"(S[57]));
                        unsigned int s_115 = __as_u32(mul_f32_rz_noftz(_abs_1379, __uint_as_float(125829120)));
                        unsigned int _min_1380 = ((s_115) < (260046848) ? (s_115) : (260046848));
                        unsigned int h_117 = _min_1380 + ((h_noise[10] & 536805376) >> 16) >> 13;
                        unsigned int mag_115 = ((s_115 <= 2139095040) ? h_117 : 32256);
                        uint32_t mags_56;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_56) : "r"(mag_114), "r"(mag_115));
                        uint32_t signs_56;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_56) : "r"(__as_u32(S[56])), "r"(__as_u32(S[57])));
                        unsigned int word_56 = signs_56 & 2147516416u | mags_56;
                        h_words[0] = word_56;
                        float _abs_1381;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1381) : "f"(S[58]));
                        unsigned int s_116 = __as_u32(mul_f32_rz_noftz(_abs_1381, __uint_as_float(125829120)));
                        unsigned int _min_1382 = ((s_116) < (260046848) ? (s_116) : (260046848));
                        unsigned int h_118 = _min_1382 + (h_noise[11] & 8191) >> 13;
                        unsigned int mag_116 = ((s_116 <= 2139095040) ? h_118 : 32256);
                        float _abs_1383;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1383) : "f"(S[59]));
                        unsigned int s_117 = __as_u32(mul_f32_rz_noftz(_abs_1383, __uint_as_float(125829120)));
                        unsigned int _min_1384 = ((s_117) < (260046848) ? (s_117) : (260046848));
                        unsigned int h_119 = _min_1384 + ((h_noise[11] & 536805376) >> 16) >> 13;
                        unsigned int mag_117 = ((s_117 <= 2139095040) ? h_119 : 32256);
                        uint32_t mags_57;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_57) : "r"(mag_116), "r"(mag_117));
                        uint32_t signs_57;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_57) : "r"(__as_u32(S[58])), "r"(__as_u32(S[59])));
                        unsigned int word_57 = signs_57 & 2147516416u | mags_57;
                        h_words[1] = word_57;
#else
                        uint32_t _cvt_rs_x2_f32_275;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_275) : "f"(S2[28].y), "f"(S2[28].x), "r"(h_noise[26] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_275;
                        uint32_t _cvt_rs_x2_f32_276;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_276) : "f"(S2[29].y), "f"(S2[29].x), "r"(h_noise[27] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_276;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1385;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1385) : "f"(S[52]));
                        unsigned int s_118 = __as_u32(mul_f32_rz_noftz(_abs_1385, __uint_as_float(125829120)));
                        unsigned int _min_1386 = ((s_118) < (260046848) ? (s_118) : (260046848));
                        unsigned int h_120 = _min_1386 + (h_noise[12] & 8191) >> 13;
                        unsigned int mag_118 = ((s_118 <= 2139095040) ? h_120 : 32256);
                        float _abs_1387;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1387) : "f"(S[53]));
                        unsigned int s_119 = __as_u32(mul_f32_rz_noftz(_abs_1387, __uint_as_float(125829120)));
                        unsigned int _min_1388 = ((s_119) < (260046848) ? (s_119) : (260046848));
                        unsigned int h_121 = _min_1388 + ((h_noise[12] & 536805376) >> 16) >> 13;
                        unsigned int mag_119 = ((s_119 <= 2139095040) ? h_121 : 32256);
                        uint32_t mags_58;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_58) : "r"(mag_118), "r"(mag_119));
                        uint32_t signs_58;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_58) : "r"(__as_u32(S[52])), "r"(__as_u32(S[53])));
                        unsigned int word_58 = signs_58 & 2147516416u | mags_58;
                        h_words[0] = word_58;
                        float _abs_1389;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1389) : "f"(S[54]));
                        unsigned int s_120 = __as_u32(mul_f32_rz_noftz(_abs_1389, __uint_as_float(125829120)));
                        unsigned int _min_1390 = ((s_120) < (260046848) ? (s_120) : (260046848));
                        unsigned int h_122 = _min_1390 + (h_noise[13] & 8191) >> 13;
                        unsigned int mag_120 = ((s_120 <= 2139095040) ? h_122 : 32256);
                        float _abs_1391;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1391) : "f"(S[55]));
                        unsigned int s_121 = __as_u32(mul_f32_rz_noftz(_abs_1391, __uint_as_float(125829120)));
                        unsigned int _min_1392 = ((s_121) < (260046848) ? (s_121) : (260046848));
                        unsigned int h_123 = _min_1392 + ((h_noise[13] & 536805376) >> 16) >> 13;
                        unsigned int mag_121 = ((s_121 <= 2139095040) ? h_123 : 32256);
                        uint32_t mags_59;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_59) : "r"(mag_120), "r"(mag_121));
                        uint32_t signs_59;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_59) : "r"(__as_u32(S[54])), "r"(__as_u32(S[55])));
                        unsigned int word_59 = signs_59 & 2147516416u | mags_59;
                        h_words[1] = word_59;
#else
                        uint32_t _cvt_rs_x2_f32_277;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_277) : "f"(S2[26].y), "f"(S2[26].x), "r"(h_noise[28] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_277;
                        uint32_t _cvt_rs_x2_f32_278;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_278) : "f"(S2[27].y), "f"(S2[27].x), "r"(h_noise[29] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_278;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1393;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1393) : "f"(S[60]));
                        unsigned int s_122 = __as_u32(mul_f32_rz_noftz(_abs_1393, __uint_as_float(125829120)));
                        unsigned int _min_1394 = ((s_122) < (260046848) ? (s_122) : (260046848));
                        unsigned int h_124 = _min_1394 + (h_noise[14] & 8191) >> 13;
                        unsigned int mag_122 = ((s_122 <= 2139095040) ? h_124 : 32256);
                        float _abs_1395;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1395) : "f"(S[61]));
                        unsigned int s_123 = __as_u32(mul_f32_rz_noftz(_abs_1395, __uint_as_float(125829120)));
                        unsigned int _min_1396 = ((s_123) < (260046848) ? (s_123) : (260046848));
                        unsigned int h_125 = _min_1396 + ((h_noise[14] & 536805376) >> 16) >> 13;
                        unsigned int mag_123 = ((s_123 <= 2139095040) ? h_125 : 32256);
                        uint32_t mags_60;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_60) : "r"(mag_122), "r"(mag_123));
                        uint32_t signs_60;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_60) : "r"(__as_u32(S[60])), "r"(__as_u32(S[61])));
                        unsigned int word_60 = signs_60 & 2147516416u | mags_60;
                        h_words[0] = word_60;
                        float _abs_1397;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1397) : "f"(S[62]));
                        unsigned int s_124 = __as_u32(mul_f32_rz_noftz(_abs_1397, __uint_as_float(125829120)));
                        unsigned int _min_1398 = ((s_124) < (260046848) ? (s_124) : (260046848));
                        unsigned int h_126 = _min_1398 + (h_noise[15] & 8191) >> 13;
                        unsigned int mag_124 = ((s_124 <= 2139095040) ? h_126 : 32256);
                        float _abs_1399;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1399) : "f"(S[63]));
                        unsigned int s_125 = __as_u32(mul_f32_rz_noftz(_abs_1399, __uint_as_float(125829120)));
                        unsigned int _min_1400 = ((s_125) < (260046848) ? (s_125) : (260046848));
                        unsigned int h_127 = _min_1400 + ((h_noise[15] & 536805376) >> 16) >> 13;
                        unsigned int mag_125 = ((s_125 <= 2139095040) ? h_127 : 32256);
                        uint32_t mags_61;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_61) : "r"(mag_124), "r"(mag_125));
                        uint32_t signs_61;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_61) : "r"(__as_u32(S[62])), "r"(__as_u32(S[63])));
                        unsigned int word_61 = signs_61 & 2147516416u | mags_61;
                        h_words[1] = word_61;
#else
                        uint32_t _cvt_rs_x2_f32_279;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_279) : "f"(S2[30].y), "f"(S2[30].x), "r"(h_noise[30] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_279;
                        uint32_t _cvt_rs_x2_f32_280;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_280) : "f"(S2[31].y), "f"(S2[31].x), "r"(h_noise[31] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_280;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                    }
                    if (philox_rounds == 0) {
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1401;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1401) : "f"(S[1]), "f"(S[0]));
                        h_words[0] = _cvt_rn_f16x2_f32_1401;
                        uint32_t _cvt_rn_f16x2_f32_1402;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1402) : "f"(S[3]), "f"(S[2]));
                        h_words[1] = _cvt_rn_f16x2_f32_1402;
#else
                        uint32_t _cvt_rn_f16x2_f32_281;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_281) : "f"(S2[0].y), "f"(S2[0].x));
                        h_words[0] = _cvt_rn_f16x2_f32_281;
                        uint32_t _cvt_rn_f16x2_f32_282;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_282) : "f"(S2[1].y), "f"(S2[1].x));
                        h_words[1] = _cvt_rn_f16x2_f32_282;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1403;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1403) : "f"(S[5]), "f"(S[4]));
                        h_words[0] = _cvt_rn_f16x2_f32_1403;
                        uint32_t _cvt_rn_f16x2_f32_1404;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1404) : "f"(S[7]), "f"(S[6]));
                        h_words[1] = _cvt_rn_f16x2_f32_1404;
#else
                        uint32_t _cvt_rn_f16x2_f32_283;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_283) : "f"(S2[2].y), "f"(S2[2].x));
                        h_words[0] = _cvt_rn_f16x2_f32_283;
                        uint32_t _cvt_rn_f16x2_f32_284;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_284) : "f"(S2[3].y), "f"(S2[3].x));
                        h_words[1] = _cvt_rn_f16x2_f32_284;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1405;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1405) : "f"(S[9]), "f"(S[8]));
                        h_words[0] = _cvt_rn_f16x2_f32_1405;
                        uint32_t _cvt_rn_f16x2_f32_1406;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1406) : "f"(S[11]), "f"(S[10]));
                        h_words[1] = _cvt_rn_f16x2_f32_1406;
#else
                        uint32_t _cvt_rn_f16x2_f32_285;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_285) : "f"(S2[4].y), "f"(S2[4].x));
                        h_words[0] = _cvt_rn_f16x2_f32_285;
                        uint32_t _cvt_rn_f16x2_f32_286;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_286) : "f"(S2[5].y), "f"(S2[5].x));
                        h_words[1] = _cvt_rn_f16x2_f32_286;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1407;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1407) : "f"(S[13]), "f"(S[12]));
                        h_words[0] = _cvt_rn_f16x2_f32_1407;
                        uint32_t _cvt_rn_f16x2_f32_1408;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1408) : "f"(S[15]), "f"(S[14]));
                        h_words[1] = _cvt_rn_f16x2_f32_1408;
#else
                        uint32_t _cvt_rn_f16x2_f32_287;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_287) : "f"(S2[6].y), "f"(S2[6].x));
                        h_words[0] = _cvt_rn_f16x2_f32_287;
                        uint32_t _cvt_rn_f16x2_f32_288;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_288) : "f"(S2[7].y), "f"(S2[7].x));
                        h_words[1] = _cvt_rn_f16x2_f32_288;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1409;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1409) : "f"(S[17]), "f"(S[16]));
                        h_words[0] = _cvt_rn_f16x2_f32_1409;
                        uint32_t _cvt_rn_f16x2_f32_1410;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1410) : "f"(S[19]), "f"(S[18]));
                        h_words[1] = _cvt_rn_f16x2_f32_1410;
#else
                        uint32_t _cvt_rn_f16x2_f32_289;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_289) : "f"(S2[8].y), "f"(S2[8].x));
                        h_words[0] = _cvt_rn_f16x2_f32_289;
                        uint32_t _cvt_rn_f16x2_f32_290;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_290) : "f"(S2[9].y), "f"(S2[9].x));
                        h_words[1] = _cvt_rn_f16x2_f32_290;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1411;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1411) : "f"(S[21]), "f"(S[20]));
                        h_words[0] = _cvt_rn_f16x2_f32_1411;
                        uint32_t _cvt_rn_f16x2_f32_1412;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1412) : "f"(S[23]), "f"(S[22]));
                        h_words[1] = _cvt_rn_f16x2_f32_1412;
#else
                        uint32_t _cvt_rn_f16x2_f32_291;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_291) : "f"(S2[10].y), "f"(S2[10].x));
                        h_words[0] = _cvt_rn_f16x2_f32_291;
                        uint32_t _cvt_rn_f16x2_f32_292;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_292) : "f"(S2[11].y), "f"(S2[11].x));
                        h_words[1] = _cvt_rn_f16x2_f32_292;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1413;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1413) : "f"(S[25]), "f"(S[24]));
                        h_words[0] = _cvt_rn_f16x2_f32_1413;
                        uint32_t _cvt_rn_f16x2_f32_1414;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1414) : "f"(S[27]), "f"(S[26]));
                        h_words[1] = _cvt_rn_f16x2_f32_1414;
#else
                        uint32_t _cvt_rn_f16x2_f32_293;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_293) : "f"(S2[12].y), "f"(S2[12].x));
                        h_words[0] = _cvt_rn_f16x2_f32_293;
                        uint32_t _cvt_rn_f16x2_f32_294;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_294) : "f"(S2[13].y), "f"(S2[13].x));
                        h_words[1] = _cvt_rn_f16x2_f32_294;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1415;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1415) : "f"(S[29]), "f"(S[28]));
                        h_words[0] = _cvt_rn_f16x2_f32_1415;
                        uint32_t _cvt_rn_f16x2_f32_1416;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1416) : "f"(S[31]), "f"(S[30]));
                        h_words[1] = _cvt_rn_f16x2_f32_1416;
#else
                        uint32_t _cvt_rn_f16x2_f32_295;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_295) : "f"(S2[14].y), "f"(S2[14].x));
                        h_words[0] = _cvt_rn_f16x2_f32_295;
                        uint32_t _cvt_rn_f16x2_f32_296;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_296) : "f"(S2[15].y), "f"(S2[15].x));
                        h_words[1] = _cvt_rn_f16x2_f32_296;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1417;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1417) : "f"(S[33]), "f"(S[32]));
                        h_words[0] = _cvt_rn_f16x2_f32_1417;
                        uint32_t _cvt_rn_f16x2_f32_1418;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1418) : "f"(S[35]), "f"(S[34]));
                        h_words[1] = _cvt_rn_f16x2_f32_1418;
#else
                        uint32_t _cvt_rn_f16x2_f32_297;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_297) : "f"(S2[16].y), "f"(S2[16].x));
                        h_words[0] = _cvt_rn_f16x2_f32_297;
                        uint32_t _cvt_rn_f16x2_f32_298;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_298) : "f"(S2[17].y), "f"(S2[17].x));
                        h_words[1] = _cvt_rn_f16x2_f32_298;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1419;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1419) : "f"(S[37]), "f"(S[36]));
                        h_words[0] = _cvt_rn_f16x2_f32_1419;
                        uint32_t _cvt_rn_f16x2_f32_1420;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1420) : "f"(S[39]), "f"(S[38]));
                        h_words[1] = _cvt_rn_f16x2_f32_1420;
#else
                        uint32_t _cvt_rn_f16x2_f32_299;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_299) : "f"(S2[18].y), "f"(S2[18].x));
                        h_words[0] = _cvt_rn_f16x2_f32_299;
                        uint32_t _cvt_rn_f16x2_f32_300;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_300) : "f"(S2[19].y), "f"(S2[19].x));
                        h_words[1] = _cvt_rn_f16x2_f32_300;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1421;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1421) : "f"(S[41]), "f"(S[40]));
                        h_words[0] = _cvt_rn_f16x2_f32_1421;
                        uint32_t _cvt_rn_f16x2_f32_1422;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1422) : "f"(S[43]), "f"(S[42]));
                        h_words[1] = _cvt_rn_f16x2_f32_1422;
#else
                        uint32_t _cvt_rn_f16x2_f32_301;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_301) : "f"(S2[20].y), "f"(S2[20].x));
                        h_words[0] = _cvt_rn_f16x2_f32_301;
                        uint32_t _cvt_rn_f16x2_f32_302;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_302) : "f"(S2[21].y), "f"(S2[21].x));
                        h_words[1] = _cvt_rn_f16x2_f32_302;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1423;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1423) : "f"(S[45]), "f"(S[44]));
                        h_words[0] = _cvt_rn_f16x2_f32_1423;
                        uint32_t _cvt_rn_f16x2_f32_1424;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1424) : "f"(S[47]), "f"(S[46]));
                        h_words[1] = _cvt_rn_f16x2_f32_1424;
#else
                        uint32_t _cvt_rn_f16x2_f32_303;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_303) : "f"(S2[22].y), "f"(S2[22].x));
                        h_words[0] = _cvt_rn_f16x2_f32_303;
                        uint32_t _cvt_rn_f16x2_f32_304;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_304) : "f"(S2[23].y), "f"(S2[23].x));
                        h_words[1] = _cvt_rn_f16x2_f32_304;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1425;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1425) : "f"(S[49]), "f"(S[48]));
                        h_words[0] = _cvt_rn_f16x2_f32_1425;
                        uint32_t _cvt_rn_f16x2_f32_1426;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1426) : "f"(S[51]), "f"(S[50]));
                        h_words[1] = _cvt_rn_f16x2_f32_1426;
#else
                        uint32_t _cvt_rn_f16x2_f32_305;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_305) : "f"(S2[24].y), "f"(S2[24].x));
                        h_words[0] = _cvt_rn_f16x2_f32_305;
                        uint32_t _cvt_rn_f16x2_f32_306;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_306) : "f"(S2[25].y), "f"(S2[25].x));
                        h_words[1] = _cvt_rn_f16x2_f32_306;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1427;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1427) : "f"(S[53]), "f"(S[52]));
                        h_words[0] = _cvt_rn_f16x2_f32_1427;
                        uint32_t _cvt_rn_f16x2_f32_1428;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1428) : "f"(S[55]), "f"(S[54]));
                        h_words[1] = _cvt_rn_f16x2_f32_1428;
#else
                        uint32_t _cvt_rn_f16x2_f32_307;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_307) : "f"(S2[26].y), "f"(S2[26].x));
                        h_words[0] = _cvt_rn_f16x2_f32_307;
                        uint32_t _cvt_rn_f16x2_f32_308;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_308) : "f"(S2[27].y), "f"(S2[27].x));
                        h_words[1] = _cvt_rn_f16x2_f32_308;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1429;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1429) : "f"(S[57]), "f"(S[56]));
                        h_words[0] = _cvt_rn_f16x2_f32_1429;
                        uint32_t _cvt_rn_f16x2_f32_1430;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1430) : "f"(S[59]), "f"(S[58]));
                        h_words[1] = _cvt_rn_f16x2_f32_1430;
#else
                        uint32_t _cvt_rn_f16x2_f32_309;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_309) : "f"(S2[28].y), "f"(S2[28].x));
                        h_words[0] = _cvt_rn_f16x2_f32_309;
                        uint32_t _cvt_rn_f16x2_f32_310;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_310) : "f"(S2[29].y), "f"(S2[29].x));
                        h_words[1] = _cvt_rn_f16x2_f32_310;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1431;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1431) : "f"(S[61]), "f"(S[60]));
                        h_words[0] = _cvt_rn_f16x2_f32_1431;
                        uint32_t _cvt_rn_f16x2_f32_1432;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1432) : "f"(S[63]), "f"(S[62]));
                        h_words[1] = _cvt_rn_f16x2_f32_1432;
#else
                        uint32_t _cvt_rn_f16x2_f32_311;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_311) : "f"(S2[30].y), "f"(S2[30].x));
                        h_words[0] = _cvt_rn_f16x2_f32_311;
                        uint32_t _cvt_rn_f16x2_f32_312;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_312) : "f"(S2[31].y), "f"(S2[31].x));
                        h_words[1] = _cvt_rn_f16x2_f32_312;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                    }
                }
            }
        }
    }
}

} // extern "C"
