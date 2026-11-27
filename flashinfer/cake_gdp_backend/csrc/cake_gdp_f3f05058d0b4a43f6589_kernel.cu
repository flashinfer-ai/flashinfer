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

__global__ __launch_bounds__(64, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdp_f3f05058d0b4a43f6589(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __half* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, int* __restrict__ ssm_state_indices, long long* __restrict__ rand_seed, CakeFastDivmod heads, int num_tokens, float scale, int disable_state_update, int use_ssm_state_indices, int philox_rounds, long long state_size_p0, long long state_stride_p1, long long state_stride_p2, long long state_stride_p0, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long ssm_state_indices_stride_p0, long long ssm_state_indices_stride_p1, long long out_stride_p0, long long out_stride_p1, long long out_stride_p2)
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
    float* sQ = reinterpret_cast<float*>(smem_raw + 8192);
    const int sQ_addr = smem + 8192;
    float* sV = reinterpret_cast<float*>(smem_raw + 16384);
    const int sV_addr = smem + 16384;
    float* sAlpha = reinterpret_cast<float*>(smem_raw + 20480);
    const int sAlpha_addr = smem + 20480;
    float* sBeta = reinterpret_cast<float*>(smem_raw + 20544);
    const int sBeta_addr = smem + 20544;
    int* sSsm = reinterpret_cast<int*>(smem_raw + 20608);
    const int sSsm_addr = smem + 20608;
    unsigned int* sNoise = reinterpret_cast<unsigned int*>(smem_raw + 20672);
    const int sNoise_addr = smem + 20672;

    // === Task calls (declaration order) ===
    uint32_t _fast_div_q_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)blockIdx.x), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)blockIdx.x);
    uint32_t _fast_div_r_0 = (uint32_t)((unsigned int)blockIdx.x) - _fast_div_q_0 * (uint32_t)(heads.divisor);
    int n = (int)_fast_div_q_0;
    int h = (int)_fast_div_r_0;
    int tid_0 = tid;
    int lane_local = lane;
    int warp_local = warp;
    int q_warp_local = warp_local;
    int u = tid_0 - tid_0 / 4 * 4;
    int g = tid_0 / 4;
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
    int steps = num_tokens;
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
        float S[128];
#if __CUDA_ARCH__ == 900
#else
        float2 S2[64];
#endif
        float k_regs[32];
#if __CUDA_ARCH__ == 900
#else
        float2 k2[16];
#endif
        float v_regs[4];
        float o4[4];
        unsigned int h_words[2];
#if __CUDA_ARCH__ == 900
        unsigned int h_noise[16];
#else
        unsigned int h_noise[64];
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
            _vld_1 = *reinterpret_cast<const uint2*>(state + read_rows_base + 16ll);
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
            _vld_2 = *reinterpret_cast<const uint2*>(state + read_rows_base + 32ll);
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
            _vld_3 = *reinterpret_cast<const uint2*>(state + read_rows_base + 48ll);
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
            _vld_4 = *reinterpret_cast<const uint2*>(state + read_rows_base + 64ll);
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
            _vld_5 = *reinterpret_cast<const uint2*>(state + read_rows_base + 80ll);
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
            _vld_6 = *reinterpret_cast<const uint2*>(state + read_rows_base + 96ll);
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
            _vld_7 = *reinterpret_cast<const uint2*>(state + read_rows_base + 112ll);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[32 + _pair * 2]))
                    : "r"(_vpairs_8[_pair]));
            }
        }
        {
            uint2 _vld_9;
            _vld_9 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 16ll);
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
            _vld_10 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 32ll);
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
            _vld_11 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 48ll);
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
            _vld_12 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 64ll);
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
            _vld_13 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 80ll);
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
            _vld_14 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 96ll);
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
            _vld_15 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 112ll);
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
        {
            uint2 _vld_16;
            _vld_16 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2);
            uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[64 + _pair * 2]))
                    : "r"(_vpairs_16[_pair]));
            }
        }
        {
            uint2 _vld_17;
            _vld_17 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2 + 16ll);
            uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[68 + _pair * 2]))
                    : "r"(_vpairs_17[_pair]));
            }
        }
        {
            uint2 _vld_18;
            _vld_18 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2 + 32ll);
            uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[72 + _pair * 2]))
                    : "r"(_vpairs_18[_pair]));
            }
        }
        {
            uint2 _vld_19;
            _vld_19 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2 + 48ll);
            uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[76 + _pair * 2]))
                    : "r"(_vpairs_19[_pair]));
            }
        }
        {
            uint2 _vld_20;
            _vld_20 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2 + 64ll);
            uint32_t* _vpairs_20 = reinterpret_cast<uint32_t*>(&_vld_20);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[80 + _pair * 2]))
                    : "r"(_vpairs_20[_pair]));
            }
        }
        {
            uint2 _vld_21;
            _vld_21 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2 + 80ll);
            uint32_t* _vpairs_21 = reinterpret_cast<uint32_t*>(&_vld_21);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[84 + _pair * 2]))
                    : "r"(_vpairs_21[_pair]));
            }
        }
        {
            uint2 _vld_22;
            _vld_22 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2 + 96ll);
            uint32_t* _vpairs_22 = reinterpret_cast<uint32_t*>(&_vld_22);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[88 + _pair * 2]))
                    : "r"(_vpairs_22[_pair]));
            }
        }
        {
            uint2 _vld_23;
            _vld_23 = *reinterpret_cast<const uint2*>(state + read_rows_base + 2ll * state_stride_p2 + 112ll);
            uint32_t* _vpairs_23 = reinterpret_cast<uint32_t*>(&_vld_23);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[92 + _pair * 2]))
                    : "r"(_vpairs_23[_pair]));
            }
        }
        {
            uint2 _vld_24;
            _vld_24 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2);
            uint32_t* _vpairs_24 = reinterpret_cast<uint32_t*>(&_vld_24);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[96 + _pair * 2]))
                    : "r"(_vpairs_24[_pair]));
            }
        }
        {
            uint2 _vld_25;
            _vld_25 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2 + 16ll);
            uint32_t* _vpairs_25 = reinterpret_cast<uint32_t*>(&_vld_25);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[100 + _pair * 2]))
                    : "r"(_vpairs_25[_pair]));
            }
        }
        {
            uint2 _vld_26;
            _vld_26 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2 + 32ll);
            uint32_t* _vpairs_26 = reinterpret_cast<uint32_t*>(&_vld_26);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[104 + _pair * 2]))
                    : "r"(_vpairs_26[_pair]));
            }
        }
        {
            uint2 _vld_27;
            _vld_27 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2 + 48ll);
            uint32_t* _vpairs_27 = reinterpret_cast<uint32_t*>(&_vld_27);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[108 + _pair * 2]))
                    : "r"(_vpairs_27[_pair]));
            }
        }
        {
            uint2 _vld_28;
            _vld_28 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2 + 64ll);
            uint32_t* _vpairs_28 = reinterpret_cast<uint32_t*>(&_vld_28);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[112 + _pair * 2]))
                    : "r"(_vpairs_28[_pair]));
            }
        }
        {
            uint2 _vld_29;
            _vld_29 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2 + 80ll);
            uint32_t* _vpairs_29 = reinterpret_cast<uint32_t*>(&_vld_29);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[116 + _pair * 2]))
                    : "r"(_vpairs_29[_pair]));
            }
        }
        {
            uint2 _vld_30;
            _vld_30 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2 + 96ll);
            uint32_t* _vpairs_30 = reinterpret_cast<uint32_t*>(&_vld_30);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[120 + _pair * 2]))
                    : "r"(_vpairs_30[_pair]));
            }
        }
        {
            uint2 _vld_31;
            _vld_31 = *reinterpret_cast<const uint2*>(state + read_rows_base + 3ll * state_stride_p2 + 112ll);
            uint32_t* _vpairs_31 = reinterpret_cast<uint32_t*>(&_vld_31);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[124 + _pair * 2]))
                    : "r"(_vpairs_31[_pair]));
            }
        }
        unsigned int noise_lane = sNoise_addr + (unsigned int)(tid_0 * 16);
        #pragma unroll 2
        for (int mm = 0; mm < 8; mm++) {
            int m = mm * 2 + warp_local;
            if (m < steps) {
                long long k_index = (long long)n * k_stride_p0 + (long long)m * k_stride_p1 + (long long)h * k_stride_p2;
                {
                    uint2 _vld_32;
                    _vld_32 = *reinterpret_cast<const uint2*>(k + k_index + (long long)k_col0);
                    uint32_t* _vpairs_32 = reinterpret_cast<uint32_t*>(&_vld_32);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&r_k[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) << 16);
                        (&r_k[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_32[_pair]) & 0xffff0000u);
                    }
                }
#if __CUDA_ARCH__ == 900
                float x0_x = r_k[0];
                float x0_y = r_k[1];
                float x1_x = r_k[2];
                float x1_y = r_k[3];
                float sum_x;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x) : "f"(x0_x), "f"(x0_x), "f"(0.0f));
                float sum_y;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y) : "f"(x0_y), "f"(x0_y), "f"(0.0f));
                float _fma_33;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(x1_x), "f"(x1_x), "f"(sum_x));
                sum_x = _fma_33;
                float _fma_34;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(x1_y), "f"(x1_y), "f"(sum_y));
                sum_y = _fma_34;
                sum_x = sum_x + sum_y;
                float _warp_reduce_35 = sum_x;
#else
                float2 x0 = make_float2(r_k[0], r_k[1]);
                float2 x1 = make_float2(r_k[2], r_k[3]);
                float2 sum_pair = fma_f32x2_rn_ftz(x0, x0, make_float2(0.0f, 0.0f));
                sum_pair = fma_f32x2_rn_ftz(x1, x1, sum_pair);
                float sum_x = sum_pair.x + sum_pair.y;
                float _warp_reduce_33 = sum_x;
#endif
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
                    _warp_reduce_35 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_35, offset);
                float norm = rsqrtf(_warp_reduce_35 + 1e-06f);
                x0_x = mul_f32_rn_ftz(x0_x, norm);
                x0_y = mul_f32_rn_ftz(x0_y, norm);
                x1_x = mul_f32_rn_ftz(x1_x, norm);
                x1_y = mul_f32_rn_ftz(x1_y, norm);
                sK[m * 128 + k_col0] = x0_x;
                sK[m * 128 + k_col0 + 1] = x0_y;
                sK[m * 128 + k_col0 + 2] = x1_x;
                sK[m * 128 + k_col0 + 3] = x1_y;
#else
                    _warp_reduce_33 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_33, offset);
                float norm = rsqrtf(_warp_reduce_33 + 1e-06f);
                float2 norm2 = make_float2(norm, norm);
                x0 = mul_f32x2(x0, norm2);
                x1 = mul_f32x2(x1, norm2);
                sK[m * 128 + k_col0] = x0.x;
                sK[m * 128 + k_col0 + 1] = x0.y;
                sK[m * 128 + k_col0 + 2] = x1.x;
                sK[m * 128 + k_col0 + 3] = x1.y;
#endif
                long long v_index = (long long)n * v_stride_p0 + (long long)m * v_stride_p1 + (long long)h * v_stride_p2;
                int v_col = lane_local * 2;
                sV[m * 64 + v_col] = (float)v[v_index + (long long)v_col];
                sV[m * 64 + v_col + 1] = (float)v[v_index + (long long)v_col + 1ll];
                if (lane_local == 0) {
                    long long b_index = (long long)n * b_stride_p0 + (long long)m * b_stride_p1 + (long long)h * b_stride_p2;
#if __CUDA_ARCH__ == 900
                    float _exp_37 = __expf(-(float)b[b_index]);
                    float _rcp_36 = approx_rcp(1.0f + _exp_37);
                    sBeta[m] = _rcp_36;
#else
                    float _exp_35 = __expf(-(float)b[b_index]);
                    float _rcp_34 = approx_rcp(1.0f + _exp_35);
                    sBeta[m] = _rcp_34;
#endif
                }
            }
        }
        #pragma unroll 2
        for (int tt = 0; tt < 8; tt++) {
            int t = tt * 2 + warp_local;
            if (t < num_tokens) {
                long long q_index = (long long)n * q_stride_p0 + (long long)t * q_stride_p1 + (long long)h * q_stride_p2;
                {
#if __CUDA_ARCH__ == 900
                    uint2 _vld_38;
                    _vld_38 = *reinterpret_cast<const uint2*>(q + q_index + (long long)k_col0);
                    uint32_t* _vpairs_38 = reinterpret_cast<uint32_t*>(&_vld_38);
#else
                    uint2 _vld_36;
                    _vld_36 = *reinterpret_cast<const uint2*>(q + q_index + (long long)k_col0);
                    uint32_t* _vpairs_36 = reinterpret_cast<uint32_t*>(&_vld_36);
#endif
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
#if __CUDA_ARCH__ == 900
                        (&r_q[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_38[_pair]) << 16);
                        (&r_q[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_38[_pair]) & 0xffff0000u);
#else
                        (&r_q[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) << 16);
                        (&r_q[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) & 0xffff0000u);
#endif
                    }
                }
#if __CUDA_ARCH__ == 900
                float x0_x_1 = r_q[0];
                float x0_y_1 = r_q[1];
                float x1_x_1 = r_q[2];
                float x1_y_1 = r_q[3];
                float sum_x_1;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_1) : "f"(x0_x_1), "f"(x0_x_1), "f"(0.0f));
                float sum_y_1;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_1) : "f"(x0_y_1), "f"(x0_y_1), "f"(0.0f));
                float _fma_39;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(x1_x_1), "f"(x1_x_1), "f"(sum_x_1));
                sum_x_1 = _fma_39;
                float _fma_40;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(x1_y_1), "f"(x1_y_1), "f"(sum_y_1));
                sum_y_1 = _fma_40;
                sum_x_1 = sum_x_1 + sum_y_1;
                float _warp_reduce_41 = sum_x_1;
#else
                float2 x0_1 = make_float2(r_q[0], r_q[1]);
                float2 x1_1 = make_float2(r_q[2], r_q[3]);
                float2 sum_pair_1 = fma_f32x2_rn_ftz(x0_1, x0_1, make_float2(0.0f, 0.0f));
                sum_pair_1 = fma_f32x2_rn_ftz(x1_1, x1_1, sum_pair_1);
                float sum_x_1 = sum_pair_1.x + sum_pair_1.y;
                float _warp_reduce_37 = sum_x_1;
#endif
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
#if __CUDA_ARCH__ == 900
                    _warp_reduce_41 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_41, offset);
                float norm_1 = rsqrtf(_warp_reduce_41 + 1e-06f);
#else
                    _warp_reduce_37 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_37, offset);
                float norm_1 = rsqrtf(_warp_reduce_37 + 1e-06f);
#endif
                norm_1 = norm_1 * scale;
#if __CUDA_ARCH__ == 900
                x0_x_1 = mul_f32_rn_ftz(x0_x_1, norm_1);
                x0_y_1 = mul_f32_rn_ftz(x0_y_1, norm_1);
                x1_x_1 = mul_f32_rn_ftz(x1_x_1, norm_1);
                x1_y_1 = mul_f32_rn_ftz(x1_y_1, norm_1);
                sQ[t * 128 + k_col0] = x0_x_1;
                sQ[t * 128 + k_col0 + 1] = x0_y_1;
                sQ[t * 128 + k_col0 + 2] = x1_x_1;
                sQ[t * 128 + k_col0 + 3] = x1_y_1;
#else
                float2 norm2_1 = make_float2(norm_1, norm_1);
                x0_1 = mul_f32x2(x0_1, norm2_1);
                x1_1 = mul_f32x2(x1_1, norm2_1);
                sQ[t * 128 + k_col0] = x0_1.x;
                sQ[t * 128 + k_col0 + 1] = x0_1.y;
                sQ[t * 128 + k_col0 + 2] = x1_1.x;
                sQ[t * 128 + k_col0 + 3] = x1_1.y;
#endif
                if (lane_local == 0) {
                    long long gate_index = (long long)n * a_stride_p0 + (long long)t * a_stride_p1 + (long long)h * a_stride_p2;
                    float xv = (float)a[gate_index] + dt_bias[h];
                    float softplus_x = xv;
                    if (xv <= 20.0f) {
#if __CUDA_ARCH__ == 900
                        float _exp_43 = __expf(xv);
                        float _log2_42;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_42) : "f"(1.0f + _exp_43));
                        softplus_x = _log2_42 * 0.6931472f;
#else
                        float _exp_39 = __expf(xv);
                        float _log2_38;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_38) : "f"(1.0f + _exp_39));
                        softplus_x = _log2_38 * 0.6931472f;
#endif
                    }
#if __CUDA_ARCH__ == 900
                    float _exp_44 = __expf(A_log[h]);
                    float g_log = -_exp_44 * softplus_x;
                    float _exp_45 = __expf(g_log);
                    sAlpha[t] = _exp_45;
#else
                    float _exp_40 = __expf(A_log[h]);
                    float g_log = -_exp_40 * softplus_x;
                    float _exp_41 = __expf(g_log);
                    sAlpha[t] = _exp_41;
#endif
                    if (use_ssm_state_indices != 0) {
                        sSsm[t] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)t * ssm_state_indices_stride_p1];
                    }
                    if (use_ssm_state_indices == 0) {
                        sSsm[t] = -1;
                    }
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
        S2[32] = make_float2(S[64], S[65]);
        S2[33] = make_float2(S[66], S[67]);
        S2[34] = make_float2(S[68], S[69]);
        S2[35] = make_float2(S[70], S[71]);
        S2[36] = make_float2(S[72], S[73]);
        S2[37] = make_float2(S[74], S[75]);
        S2[38] = make_float2(S[76], S[77]);
        S2[39] = make_float2(S[78], S[79]);
        S2[40] = make_float2(S[80], S[81]);
        S2[41] = make_float2(S[82], S[83]);
        S2[42] = make_float2(S[84], S[85]);
        S2[43] = make_float2(S[86], S[87]);
        S2[44] = make_float2(S[88], S[89]);
        S2[45] = make_float2(S[90], S[91]);
        S2[46] = make_float2(S[92], S[93]);
        S2[47] = make_float2(S[94], S[95]);
        S2[48] = make_float2(S[96], S[97]);
        S2[49] = make_float2(S[98], S[99]);
        S2[50] = make_float2(S[100], S[101]);
        S2[51] = make_float2(S[102], S[103]);
        S2[52] = make_float2(S[104], S[105]);
        S2[53] = make_float2(S[106], S[107]);
        S2[54] = make_float2(S[108], S[109]);
        S2[55] = make_float2(S[110], S[111]);
        S2[56] = make_float2(S[112], S[113]);
        S2[57] = make_float2(S[114], S[115]);
        S2[58] = make_float2(S[116], S[117]);
        S2[59] = make_float2(S[118], S[119]);
        S2[60] = make_float2(S[120], S[121]);
        S2[61] = make_float2(S[122], S[123]);
        S2[62] = make_float2(S[124], S[125]);
        S2[63] = make_float2(S[126], S[127]);
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
                long long o_1 = read_rows_base + 16ll;
                unsigned long long p0_1 = (unsigned long long)(unsigned int)o_1 * 3528531795ull;
                h_noise[4] = (unsigned int)(o_1 >> 32ll) ^ key0;
                h_noise[6] = (unsigned int)(p0_1 >> 32ull) ^ key1;
                h_noise[7] = (unsigned int)p0_1;
                long long o_2 = read_rows_base + 32ll;
                unsigned long long p0_2 = (unsigned long long)(unsigned int)o_2 * 3528531795ull;
                h_noise[8] = (unsigned int)(o_2 >> 32ll) ^ key0;
                h_noise[10] = (unsigned int)(p0_2 >> 32ull) ^ key1;
                h_noise[11] = (unsigned int)p0_2;
                long long o_3 = read_rows_base + 48ll;
                unsigned long long p0_3 = (unsigned long long)(unsigned int)o_3 * 3528531795ull;
                h_noise[12] = (unsigned int)(o_3 >> 32ll) ^ key0;
                h_noise[14] = (unsigned int)(p0_3 >> 32ull) ^ key1;
                h_noise[15] = (unsigned int)p0_3;
#if __CUDA_ARCH__ == 900
#else
                long long o_4 = read_rows_base + state_stride_p2;
                unsigned long long p0_4 = (unsigned long long)(unsigned int)o_4 * 3528531795ull;
                h_noise[16] = (unsigned int)(o_4 >> 32ll) ^ key0;
                h_noise[18] = (unsigned int)(p0_4 >> 32ull) ^ key1;
                h_noise[19] = (unsigned int)p0_4;
                long long o_5 = read_rows_base + state_stride_p2 + 16ll;
                unsigned long long p0_5 = (unsigned long long)(unsigned int)o_5 * 3528531795ull;
                h_noise[20] = (unsigned int)(o_5 >> 32ll) ^ key0;
                h_noise[22] = (unsigned int)(p0_5 >> 32ull) ^ key1;
                h_noise[23] = (unsigned int)p0_5;
                long long o_6 = read_rows_base + state_stride_p2 + 32ll;
                unsigned long long p0_6 = (unsigned long long)(unsigned int)o_6 * 3528531795ull;
                h_noise[24] = (unsigned int)(o_6 >> 32ll) ^ key0;
                h_noise[26] = (unsigned int)(p0_6 >> 32ull) ^ key1;
                h_noise[27] = (unsigned int)p0_6;
                long long o_7 = read_rows_base + state_stride_p2 + 48ll;
                unsigned long long p0_7 = (unsigned long long)(unsigned int)o_7 * 3528531795ull;
                h_noise[28] = (unsigned int)(o_7 >> 32ll) ^ key0;
                h_noise[30] = (unsigned int)(p0_7 >> 32ull) ^ key1;
                h_noise[31] = (unsigned int)p0_7;
                long long o_8 = read_rows_base + 2ll * state_stride_p2;
                unsigned long long p0_8 = (unsigned long long)(unsigned int)o_8 * 3528531795ull;
                h_noise[32] = (unsigned int)(o_8 >> 32ll) ^ key0;
                h_noise[34] = (unsigned int)(p0_8 >> 32ull) ^ key1;
                h_noise[35] = (unsigned int)p0_8;
                long long o_9 = read_rows_base + 2ll * state_stride_p2 + 16ll;
                unsigned long long p0_9 = (unsigned long long)(unsigned int)o_9 * 3528531795ull;
                h_noise[36] = (unsigned int)(o_9 >> 32ll) ^ key0;
                h_noise[38] = (unsigned int)(p0_9 >> 32ull) ^ key1;
                h_noise[39] = (unsigned int)p0_9;
                long long o_10 = read_rows_base + 2ll * state_stride_p2 + 32ll;
                unsigned long long p0_10 = (unsigned long long)(unsigned int)o_10 * 3528531795ull;
                h_noise[40] = (unsigned int)(o_10 >> 32ll) ^ key0;
                h_noise[42] = (unsigned int)(p0_10 >> 32ull) ^ key1;
                h_noise[43] = (unsigned int)p0_10;
                long long o_11 = read_rows_base + 2ll * state_stride_p2 + 48ll;
                unsigned long long p0_11 = (unsigned long long)(unsigned int)o_11 * 3528531795ull;
                h_noise[44] = (unsigned int)(o_11 >> 32ll) ^ key0;
                h_noise[46] = (unsigned int)(p0_11 >> 32ull) ^ key1;
                h_noise[47] = (unsigned int)p0_11;
                long long o_12 = read_rows_base + 3ll * state_stride_p2;
                unsigned long long p0_12 = (unsigned long long)(unsigned int)o_12 * 3528531795ull;
                h_noise[48] = (unsigned int)(o_12 >> 32ll) ^ key0;
                h_noise[50] = (unsigned int)(p0_12 >> 32ull) ^ key1;
                h_noise[51] = (unsigned int)p0_12;
                long long o_13 = read_rows_base + 3ll * state_stride_p2 + 16ll;
                unsigned long long p0_13 = (unsigned long long)(unsigned int)o_13 * 3528531795ull;
                h_noise[52] = (unsigned int)(o_13 >> 32ll) ^ key0;
                h_noise[54] = (unsigned int)(p0_13 >> 32ull) ^ key1;
                h_noise[55] = (unsigned int)p0_13;
                long long o_14 = read_rows_base + 3ll * state_stride_p2 + 32ll;
                unsigned long long p0_14 = (unsigned long long)(unsigned int)o_14 * 3528531795ull;
                h_noise[56] = (unsigned int)(o_14 >> 32ll) ^ key0;
                h_noise[58] = (unsigned int)(p0_14 >> 32ull) ^ key1;
                h_noise[59] = (unsigned int)p0_14;
                long long o_15 = read_rows_base + 3ll * state_stride_p2 + 48ll;
                unsigned long long p0_15 = (unsigned long long)(unsigned int)o_15 * 3528531795ull;
                h_noise[60] = (unsigned int)(o_15 >> 32ll) ^ key0;
                h_noise[62] = (unsigned int)(p0_15 >> 32ull) ^ key1;
                h_noise[63] = (unsigned int)p0_15;
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
                unsigned long long q0_8 = (unsigned long long)h_noise[32] * 3528531795ull;
                unsigned long long q1_8 = (unsigned long long)h_noise[34] * 3449720151ull;
                h_noise[32] = (unsigned int)(q1_8 >> 32ull) ^ key0;
                h_noise[33] = (unsigned int)q1_8;
                h_noise[34] = (unsigned int)(q0_8 >> 32ull) ^ h_noise[35] ^ key1;
                h_noise[35] = (unsigned int)q0_8;
                unsigned long long q0_9 = (unsigned long long)h_noise[36] * 3528531795ull;
                unsigned long long q1_9 = (unsigned long long)h_noise[38] * 3449720151ull;
                h_noise[36] = (unsigned int)(q1_9 >> 32ull) ^ key0;
                h_noise[37] = (unsigned int)q1_9;
                h_noise[38] = (unsigned int)(q0_9 >> 32ull) ^ h_noise[39] ^ key1;
                h_noise[39] = (unsigned int)q0_9;
                unsigned long long q0_10 = (unsigned long long)h_noise[40] * 3528531795ull;
                unsigned long long q1_10 = (unsigned long long)h_noise[42] * 3449720151ull;
                h_noise[40] = (unsigned int)(q1_10 >> 32ull) ^ key0;
                h_noise[41] = (unsigned int)q1_10;
                h_noise[42] = (unsigned int)(q0_10 >> 32ull) ^ h_noise[43] ^ key1;
                h_noise[43] = (unsigned int)q0_10;
                unsigned long long q0_11 = (unsigned long long)h_noise[44] * 3528531795ull;
                unsigned long long q1_11 = (unsigned long long)h_noise[46] * 3449720151ull;
                h_noise[44] = (unsigned int)(q1_11 >> 32ull) ^ key0;
                h_noise[45] = (unsigned int)q1_11;
                h_noise[46] = (unsigned int)(q0_11 >> 32ull) ^ h_noise[47] ^ key1;
                h_noise[47] = (unsigned int)q0_11;
                unsigned long long q0_12 = (unsigned long long)h_noise[48] * 3528531795ull;
                unsigned long long q1_12 = (unsigned long long)h_noise[50] * 3449720151ull;
                h_noise[48] = (unsigned int)(q1_12 >> 32ull) ^ key0;
                h_noise[49] = (unsigned int)q1_12;
                h_noise[50] = (unsigned int)(q0_12 >> 32ull) ^ h_noise[51] ^ key1;
                h_noise[51] = (unsigned int)q0_12;
                unsigned long long q0_13 = (unsigned long long)h_noise[52] * 3528531795ull;
                unsigned long long q1_13 = (unsigned long long)h_noise[54] * 3449720151ull;
                h_noise[52] = (unsigned int)(q1_13 >> 32ull) ^ key0;
                h_noise[53] = (unsigned int)q1_13;
                h_noise[54] = (unsigned int)(q0_13 >> 32ull) ^ h_noise[55] ^ key1;
                h_noise[55] = (unsigned int)q0_13;
                unsigned long long q0_14 = (unsigned long long)h_noise[56] * 3528531795ull;
                unsigned long long q1_14 = (unsigned long long)h_noise[58] * 3449720151ull;
                h_noise[56] = (unsigned int)(q1_14 >> 32ull) ^ key0;
                h_noise[57] = (unsigned int)q1_14;
                h_noise[58] = (unsigned int)(q0_14 >> 32ull) ^ h_noise[59] ^ key1;
                h_noise[59] = (unsigned int)q0_14;
                unsigned long long q0_15 = (unsigned long long)h_noise[60] * 3528531795ull;
                unsigned long long q1_15 = (unsigned long long)h_noise[62] * 3449720151ull;
                h_noise[60] = (unsigned int)(q1_15 >> 32ull) ^ key0;
                h_noise[61] = (unsigned int)q1_15;
                h_noise[62] = (unsigned int)(q0_15 >> 32ull) ^ h_noise[63] ^ key1;
                h_noise[63] = (unsigned int)q0_15;
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
                unsigned long long s0_8 = (unsigned long long)h_noise[32] * 3528531795ull;
                unsigned long long s1_8 = (unsigned long long)h_noise[34] * 3449720151ull;
                h_noise[32] = (unsigned int)(s1_8 >> 32ull) ^ h_noise[33] ^ key0;
                h_noise[33] = (unsigned int)s1_8;
                h_noise[34] = (unsigned int)(s0_8 >> 32ull) ^ h_noise[35] ^ key1;
                h_noise[35] = (unsigned int)s0_8;
                unsigned long long s0_9 = (unsigned long long)h_noise[36] * 3528531795ull;
                unsigned long long s1_9 = (unsigned long long)h_noise[38] * 3449720151ull;
                h_noise[36] = (unsigned int)(s1_9 >> 32ull) ^ h_noise[37] ^ key0;
                h_noise[37] = (unsigned int)s1_9;
                h_noise[38] = (unsigned int)(s0_9 >> 32ull) ^ h_noise[39] ^ key1;
                h_noise[39] = (unsigned int)s0_9;
                unsigned long long s0_10 = (unsigned long long)h_noise[40] * 3528531795ull;
                unsigned long long s1_10 = (unsigned long long)h_noise[42] * 3449720151ull;
                h_noise[40] = (unsigned int)(s1_10 >> 32ull) ^ h_noise[41] ^ key0;
                h_noise[41] = (unsigned int)s1_10;
                h_noise[42] = (unsigned int)(s0_10 >> 32ull) ^ h_noise[43] ^ key1;
                h_noise[43] = (unsigned int)s0_10;
                unsigned long long s0_11 = (unsigned long long)h_noise[44] * 3528531795ull;
                unsigned long long s1_11 = (unsigned long long)h_noise[46] * 3449720151ull;
                h_noise[44] = (unsigned int)(s1_11 >> 32ull) ^ h_noise[45] ^ key0;
                h_noise[45] = (unsigned int)s1_11;
                h_noise[46] = (unsigned int)(s0_11 >> 32ull) ^ h_noise[47] ^ key1;
                h_noise[47] = (unsigned int)s0_11;
                unsigned long long s0_12 = (unsigned long long)h_noise[48] * 3528531795ull;
                unsigned long long s1_12 = (unsigned long long)h_noise[50] * 3449720151ull;
                h_noise[48] = (unsigned int)(s1_12 >> 32ull) ^ h_noise[49] ^ key0;
                h_noise[49] = (unsigned int)s1_12;
                h_noise[50] = (unsigned int)(s0_12 >> 32ull) ^ h_noise[51] ^ key1;
                h_noise[51] = (unsigned int)s0_12;
                unsigned long long s0_13 = (unsigned long long)h_noise[52] * 3528531795ull;
                unsigned long long s1_13 = (unsigned long long)h_noise[54] * 3449720151ull;
                h_noise[52] = (unsigned int)(s1_13 >> 32ull) ^ h_noise[53] ^ key0;
                h_noise[53] = (unsigned int)s1_13;
                h_noise[54] = (unsigned int)(s0_13 >> 32ull) ^ h_noise[55] ^ key1;
                h_noise[55] = (unsigned int)s0_13;
                unsigned long long s0_14 = (unsigned long long)h_noise[56] * 3528531795ull;
                unsigned long long s1_14 = (unsigned long long)h_noise[58] * 3449720151ull;
                h_noise[56] = (unsigned int)(s1_14 >> 32ull) ^ h_noise[57] ^ key0;
                h_noise[57] = (unsigned int)s1_14;
                h_noise[58] = (unsigned int)(s0_14 >> 32ull) ^ h_noise[59] ^ key1;
                h_noise[59] = (unsigned int)s0_14;
                unsigned long long s0_15 = (unsigned long long)h_noise[60] * 3528531795ull;
                unsigned long long s1_15 = (unsigned long long)h_noise[62] * 3449720151ull;
                h_noise[60] = (unsigned int)(s1_15 >> 32ull) ^ h_noise[61] ^ key0;
                h_noise[61] = (unsigned int)s1_15;
                h_noise[62] = (unsigned int)(s0_15 >> 32ull) ^ h_noise[63] ^ key1;
                h_noise[63] = (unsigned int)s0_15;
#endif
                key0 = key0 + 2654435769u;
                key1 = key1 + 3144134277u;
#if __CUDA_ARCH__ == 900
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
                unsigned long long s0_24 = (unsigned long long)h_noise[32] * 3528531795ull;
                unsigned long long s1_24 = (unsigned long long)h_noise[34] * 3449720151ull;
                h_noise[32] = (unsigned int)(s1_24 >> 32ull) ^ h_noise[33] ^ key0;
                h_noise[33] = (unsigned int)s1_24;
                h_noise[34] = (unsigned int)(s0_24 >> 32ull) ^ h_noise[35] ^ key1;
                h_noise[35] = (unsigned int)s0_24;
                unsigned long long s0_25 = (unsigned long long)h_noise[36] * 3528531795ull;
                unsigned long long s1_25 = (unsigned long long)h_noise[38] * 3449720151ull;
                h_noise[36] = (unsigned int)(s1_25 >> 32ull) ^ h_noise[37] ^ key0;
                h_noise[37] = (unsigned int)s1_25;
                h_noise[38] = (unsigned int)(s0_25 >> 32ull) ^ h_noise[39] ^ key1;
                h_noise[39] = (unsigned int)s0_25;
                unsigned long long s0_26 = (unsigned long long)h_noise[40] * 3528531795ull;
                unsigned long long s1_26 = (unsigned long long)h_noise[42] * 3449720151ull;
                h_noise[40] = (unsigned int)(s1_26 >> 32ull) ^ h_noise[41] ^ key0;
                h_noise[41] = (unsigned int)s1_26;
                h_noise[42] = (unsigned int)(s0_26 >> 32ull) ^ h_noise[43] ^ key1;
                h_noise[43] = (unsigned int)s0_26;
                unsigned long long s0_27 = (unsigned long long)h_noise[44] * 3528531795ull;
                unsigned long long s1_27 = (unsigned long long)h_noise[46] * 3449720151ull;
                h_noise[44] = (unsigned int)(s1_27 >> 32ull) ^ h_noise[45] ^ key0;
                h_noise[45] = (unsigned int)s1_27;
                h_noise[46] = (unsigned int)(s0_27 >> 32ull) ^ h_noise[47] ^ key1;
                h_noise[47] = (unsigned int)s0_27;
                unsigned long long s0_28 = (unsigned long long)h_noise[48] * 3528531795ull;
                unsigned long long s1_28 = (unsigned long long)h_noise[50] * 3449720151ull;
                h_noise[48] = (unsigned int)(s1_28 >> 32ull) ^ h_noise[49] ^ key0;
                h_noise[49] = (unsigned int)s1_28;
                h_noise[50] = (unsigned int)(s0_28 >> 32ull) ^ h_noise[51] ^ key1;
                h_noise[51] = (unsigned int)s0_28;
                unsigned long long s0_29 = (unsigned long long)h_noise[52] * 3528531795ull;
                unsigned long long s1_29 = (unsigned long long)h_noise[54] * 3449720151ull;
                h_noise[52] = (unsigned int)(s1_29 >> 32ull) ^ h_noise[53] ^ key0;
                h_noise[53] = (unsigned int)s1_29;
                h_noise[54] = (unsigned int)(s0_29 >> 32ull) ^ h_noise[55] ^ key1;
                h_noise[55] = (unsigned int)s0_29;
                unsigned long long s0_30 = (unsigned long long)h_noise[56] * 3528531795ull;
                unsigned long long s1_30 = (unsigned long long)h_noise[58] * 3449720151ull;
                h_noise[56] = (unsigned int)(s1_30 >> 32ull) ^ h_noise[57] ^ key0;
                h_noise[57] = (unsigned int)s1_30;
                h_noise[58] = (unsigned int)(s0_30 >> 32ull) ^ h_noise[59] ^ key1;
                h_noise[59] = (unsigned int)s0_30;
                unsigned long long s0_31 = (unsigned long long)h_noise[60] * 3528531795ull;
                unsigned long long s1_31 = (unsigned long long)h_noise[62] * 3449720151ull;
                h_noise[60] = (unsigned int)(s1_31 >> 32ull) ^ h_noise[61] ^ key0;
                h_noise[61] = (unsigned int)s1_31;
                h_noise[62] = (unsigned int)(s0_31 >> 32ull) ^ h_noise[63] ^ key1;
                h_noise[63] = (unsigned int)s0_31;
#endif
                key0 = key0 + 2654435769u;
                key1 = key1 + 3144134277u;
#if __CUDA_ARCH__ == 900
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
#else
                unsigned long long s0_32 = (unsigned long long)h_noise[0] * 3528531795ull;
                unsigned long long s1_32 = (unsigned long long)h_noise[2] * 3449720151ull;
                h_noise[0] = (unsigned int)(s1_32 >> 32ull) ^ h_noise[1] ^ key0;
                h_noise[1] = (unsigned int)s1_32;
                h_noise[2] = (unsigned int)(s0_32 >> 32ull) ^ h_noise[3] ^ key1;
                h_noise[3] = (unsigned int)s0_32;
                unsigned long long s0_33 = (unsigned long long)h_noise[4] * 3528531795ull;
                unsigned long long s1_33 = (unsigned long long)h_noise[6] * 3449720151ull;
                h_noise[4] = (unsigned int)(s1_33 >> 32ull) ^ h_noise[5] ^ key0;
                h_noise[5] = (unsigned int)s1_33;
                h_noise[6] = (unsigned int)(s0_33 >> 32ull) ^ h_noise[7] ^ key1;
                h_noise[7] = (unsigned int)s0_33;
                unsigned long long s0_34 = (unsigned long long)h_noise[8] * 3528531795ull;
                unsigned long long s1_34 = (unsigned long long)h_noise[10] * 3449720151ull;
                h_noise[8] = (unsigned int)(s1_34 >> 32ull) ^ h_noise[9] ^ key0;
                h_noise[9] = (unsigned int)s1_34;
                h_noise[10] = (unsigned int)(s0_34 >> 32ull) ^ h_noise[11] ^ key1;
                h_noise[11] = (unsigned int)s0_34;
                unsigned long long s0_35 = (unsigned long long)h_noise[12] * 3528531795ull;
                unsigned long long s1_35 = (unsigned long long)h_noise[14] * 3449720151ull;
                h_noise[12] = (unsigned int)(s1_35 >> 32ull) ^ h_noise[13] ^ key0;
                h_noise[13] = (unsigned int)s1_35;
                h_noise[14] = (unsigned int)(s0_35 >> 32ull) ^ h_noise[15] ^ key1;
                h_noise[15] = (unsigned int)s0_35;
                unsigned long long s0_36 = (unsigned long long)h_noise[16] * 3528531795ull;
                unsigned long long s1_36 = (unsigned long long)h_noise[18] * 3449720151ull;
                h_noise[16] = (unsigned int)(s1_36 >> 32ull) ^ h_noise[17] ^ key0;
                h_noise[17] = (unsigned int)s1_36;
                h_noise[18] = (unsigned int)(s0_36 >> 32ull) ^ h_noise[19] ^ key1;
                h_noise[19] = (unsigned int)s0_36;
                unsigned long long s0_37 = (unsigned long long)h_noise[20] * 3528531795ull;
                unsigned long long s1_37 = (unsigned long long)h_noise[22] * 3449720151ull;
                h_noise[20] = (unsigned int)(s1_37 >> 32ull) ^ h_noise[21] ^ key0;
                h_noise[21] = (unsigned int)s1_37;
                h_noise[22] = (unsigned int)(s0_37 >> 32ull) ^ h_noise[23] ^ key1;
                h_noise[23] = (unsigned int)s0_37;
                unsigned long long s0_38 = (unsigned long long)h_noise[24] * 3528531795ull;
                unsigned long long s1_38 = (unsigned long long)h_noise[26] * 3449720151ull;
                h_noise[24] = (unsigned int)(s1_38 >> 32ull) ^ h_noise[25] ^ key0;
                h_noise[25] = (unsigned int)s1_38;
                h_noise[26] = (unsigned int)(s0_38 >> 32ull) ^ h_noise[27] ^ key1;
                h_noise[27] = (unsigned int)s0_38;
                unsigned long long s0_39 = (unsigned long long)h_noise[28] * 3528531795ull;
                unsigned long long s1_39 = (unsigned long long)h_noise[30] * 3449720151ull;
                h_noise[28] = (unsigned int)(s1_39 >> 32ull) ^ h_noise[29] ^ key0;
                h_noise[29] = (unsigned int)s1_39;
                h_noise[30] = (unsigned int)(s0_39 >> 32ull) ^ h_noise[31] ^ key1;
                h_noise[31] = (unsigned int)s0_39;
                unsigned long long s0_40 = (unsigned long long)h_noise[32] * 3528531795ull;
                unsigned long long s1_40 = (unsigned long long)h_noise[34] * 3449720151ull;
                h_noise[32] = (unsigned int)(s1_40 >> 32ull) ^ h_noise[33] ^ key0;
                h_noise[33] = (unsigned int)s1_40;
                h_noise[34] = (unsigned int)(s0_40 >> 32ull) ^ h_noise[35] ^ key1;
                h_noise[35] = (unsigned int)s0_40;
                unsigned long long s0_41 = (unsigned long long)h_noise[36] * 3528531795ull;
                unsigned long long s1_41 = (unsigned long long)h_noise[38] * 3449720151ull;
                h_noise[36] = (unsigned int)(s1_41 >> 32ull) ^ h_noise[37] ^ key0;
                h_noise[37] = (unsigned int)s1_41;
                h_noise[38] = (unsigned int)(s0_41 >> 32ull) ^ h_noise[39] ^ key1;
                h_noise[39] = (unsigned int)s0_41;
                unsigned long long s0_42 = (unsigned long long)h_noise[40] * 3528531795ull;
                unsigned long long s1_42 = (unsigned long long)h_noise[42] * 3449720151ull;
                h_noise[40] = (unsigned int)(s1_42 >> 32ull) ^ h_noise[41] ^ key0;
                h_noise[41] = (unsigned int)s1_42;
                h_noise[42] = (unsigned int)(s0_42 >> 32ull) ^ h_noise[43] ^ key1;
                h_noise[43] = (unsigned int)s0_42;
                unsigned long long s0_43 = (unsigned long long)h_noise[44] * 3528531795ull;
                unsigned long long s1_43 = (unsigned long long)h_noise[46] * 3449720151ull;
                h_noise[44] = (unsigned int)(s1_43 >> 32ull) ^ h_noise[45] ^ key0;
                h_noise[45] = (unsigned int)s1_43;
                h_noise[46] = (unsigned int)(s0_43 >> 32ull) ^ h_noise[47] ^ key1;
                h_noise[47] = (unsigned int)s0_43;
                unsigned long long s0_44 = (unsigned long long)h_noise[48] * 3528531795ull;
                unsigned long long s1_44 = (unsigned long long)h_noise[50] * 3449720151ull;
                h_noise[48] = (unsigned int)(s1_44 >> 32ull) ^ h_noise[49] ^ key0;
                h_noise[49] = (unsigned int)s1_44;
                h_noise[50] = (unsigned int)(s0_44 >> 32ull) ^ h_noise[51] ^ key1;
                h_noise[51] = (unsigned int)s0_44;
                unsigned long long s0_45 = (unsigned long long)h_noise[52] * 3528531795ull;
                unsigned long long s1_45 = (unsigned long long)h_noise[54] * 3449720151ull;
                h_noise[52] = (unsigned int)(s1_45 >> 32ull) ^ h_noise[53] ^ key0;
                h_noise[53] = (unsigned int)s1_45;
                h_noise[54] = (unsigned int)(s0_45 >> 32ull) ^ h_noise[55] ^ key1;
                h_noise[55] = (unsigned int)s0_45;
                unsigned long long s0_46 = (unsigned long long)h_noise[56] * 3528531795ull;
                unsigned long long s1_46 = (unsigned long long)h_noise[58] * 3449720151ull;
                h_noise[56] = (unsigned int)(s1_46 >> 32ull) ^ h_noise[57] ^ key0;
                h_noise[57] = (unsigned int)s1_46;
                h_noise[58] = (unsigned int)(s0_46 >> 32ull) ^ h_noise[59] ^ key1;
                h_noise[59] = (unsigned int)s0_46;
                unsigned long long s0_47 = (unsigned long long)h_noise[60] * 3528531795ull;
                unsigned long long s1_47 = (unsigned long long)h_noise[62] * 3449720151ull;
                h_noise[60] = (unsigned int)(s1_47 >> 32ull) ^ h_noise[61] ^ key0;
                h_noise[61] = (unsigned int)s1_47;
                h_noise[62] = (unsigned int)(s0_47 >> 32ull) ^ h_noise[63] ^ key1;
                h_noise[63] = (unsigned int)s0_47;
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
                    unsigned long long t0_8 = (unsigned long long)h_noise[32] * 3528531795ull;
                    unsigned long long t1_8 = (unsigned long long)h_noise[34] * 3449720151ull;
                    h_noise[32] = (unsigned int)(t1_8 >> 32ull) ^ h_noise[33] ^ key0;
                    h_noise[33] = (unsigned int)t1_8;
                    h_noise[34] = (unsigned int)(t0_8 >> 32ull) ^ h_noise[35] ^ key1;
                    h_noise[35] = (unsigned int)t0_8;
                    unsigned long long t0_9 = (unsigned long long)h_noise[36] * 3528531795ull;
                    unsigned long long t1_9 = (unsigned long long)h_noise[38] * 3449720151ull;
                    h_noise[36] = (unsigned int)(t1_9 >> 32ull) ^ h_noise[37] ^ key0;
                    h_noise[37] = (unsigned int)t1_9;
                    h_noise[38] = (unsigned int)(t0_9 >> 32ull) ^ h_noise[39] ^ key1;
                    h_noise[39] = (unsigned int)t0_9;
                    unsigned long long t0_10 = (unsigned long long)h_noise[40] * 3528531795ull;
                    unsigned long long t1_10 = (unsigned long long)h_noise[42] * 3449720151ull;
                    h_noise[40] = (unsigned int)(t1_10 >> 32ull) ^ h_noise[41] ^ key0;
                    h_noise[41] = (unsigned int)t1_10;
                    h_noise[42] = (unsigned int)(t0_10 >> 32ull) ^ h_noise[43] ^ key1;
                    h_noise[43] = (unsigned int)t0_10;
                    unsigned long long t0_11 = (unsigned long long)h_noise[44] * 3528531795ull;
                    unsigned long long t1_11 = (unsigned long long)h_noise[46] * 3449720151ull;
                    h_noise[44] = (unsigned int)(t1_11 >> 32ull) ^ h_noise[45] ^ key0;
                    h_noise[45] = (unsigned int)t1_11;
                    h_noise[46] = (unsigned int)(t0_11 >> 32ull) ^ h_noise[47] ^ key1;
                    h_noise[47] = (unsigned int)t0_11;
                    unsigned long long t0_12 = (unsigned long long)h_noise[48] * 3528531795ull;
                    unsigned long long t1_12 = (unsigned long long)h_noise[50] * 3449720151ull;
                    h_noise[48] = (unsigned int)(t1_12 >> 32ull) ^ h_noise[49] ^ key0;
                    h_noise[49] = (unsigned int)t1_12;
                    h_noise[50] = (unsigned int)(t0_12 >> 32ull) ^ h_noise[51] ^ key1;
                    h_noise[51] = (unsigned int)t0_12;
                    unsigned long long t0_13 = (unsigned long long)h_noise[52] * 3528531795ull;
                    unsigned long long t1_13 = (unsigned long long)h_noise[54] * 3449720151ull;
                    h_noise[52] = (unsigned int)(t1_13 >> 32ull) ^ h_noise[53] ^ key0;
                    h_noise[53] = (unsigned int)t1_13;
                    h_noise[54] = (unsigned int)(t0_13 >> 32ull) ^ h_noise[55] ^ key1;
                    h_noise[55] = (unsigned int)t0_13;
                    unsigned long long t0_14 = (unsigned long long)h_noise[56] * 3528531795ull;
                    unsigned long long t1_14 = (unsigned long long)h_noise[58] * 3449720151ull;
                    h_noise[56] = (unsigned int)(t1_14 >> 32ull) ^ h_noise[57] ^ key0;
                    h_noise[57] = (unsigned int)t1_14;
                    h_noise[58] = (unsigned int)(t0_14 >> 32ull) ^ h_noise[59] ^ key1;
                    h_noise[59] = (unsigned int)t0_14;
                    unsigned long long t0_15 = (unsigned long long)h_noise[60] * 3528531795ull;
                    unsigned long long t1_15 = (unsigned long long)h_noise[62] * 3449720151ull;
                    h_noise[60] = (unsigned int)(t1_15 >> 32ull) ^ h_noise[61] ^ key0;
                    h_noise[61] = (unsigned int)t1_15;
                    h_noise[62] = (unsigned int)(t0_15 >> 32ull) ^ h_noise[63] ^ key1;
                    h_noise[63] = (unsigned int)t0_15;
#endif
                    key0 = key0 + 2654435769u;
                    key1 = key1 + 3144134277u;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 0), "r"(h_noise[0] & 536813567), "r"(h_noise[1] & 536813567), "r"(h_noise[2] & 536813567), "r"(h_noise[3] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 1024), "r"(h_noise[4] & 536813567), "r"(h_noise[5] & 536813567), "r"(h_noise[6] & 536813567), "r"(h_noise[7] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 2048), "r"(h_noise[8] & 536813567), "r"(h_noise[9] & 536813567), "r"(h_noise[10] & 536813567), "r"(h_noise[11] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 3072), "r"(h_noise[12] & 536813567), "r"(h_noise[13] & 536813567), "r"(h_noise[14] & 536813567), "r"(h_noise[15] & 536813567) : "memory");
#if __CUDA_ARCH__ == 900
                unsigned int key0_0 = seed_lo;
                unsigned int key1_0 = seed_hi;
                long long o_4 = read_rows_base + state_stride_p2;
                unsigned long long p0_4 = (unsigned long long)(unsigned int)o_4 * 3528531795ull;
                h_noise[0] = (unsigned int)(o_4 >> 32ll) ^ key0_0;
                h_noise[2] = (unsigned int)(p0_4 >> 32ull) ^ key1_0;
                h_noise[3] = (unsigned int)p0_4;
                long long o_5 = read_rows_base + state_stride_p2 + 16ll;
                unsigned long long p0_5 = (unsigned long long)(unsigned int)o_5 * 3528531795ull;
                h_noise[4] = (unsigned int)(o_5 >> 32ll) ^ key0_0;
                h_noise[6] = (unsigned int)(p0_5 >> 32ull) ^ key1_0;
                h_noise[7] = (unsigned int)p0_5;
                long long o_6 = read_rows_base + state_stride_p2 + 32ll;
                unsigned long long p0_6 = (unsigned long long)(unsigned int)o_6 * 3528531795ull;
                h_noise[8] = (unsigned int)(o_6 >> 32ll) ^ key0_0;
                h_noise[10] = (unsigned int)(p0_6 >> 32ull) ^ key1_0;
                h_noise[11] = (unsigned int)p0_6;
                long long o_7 = read_rows_base + state_stride_p2 + 48ll;
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
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 4096), "r"(h_noise[0] & 536813567), "r"(h_noise[1] & 536813567), "r"(h_noise[2] & 536813567), "r"(h_noise[3] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 5120), "r"(h_noise[4] & 536813567), "r"(h_noise[5] & 536813567), "r"(h_noise[6] & 536813567), "r"(h_noise[7] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 6144), "r"(h_noise[8] & 536813567), "r"(h_noise[9] & 536813567), "r"(h_noise[10] & 536813567), "r"(h_noise[11] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 7168), "r"(h_noise[12] & 536813567), "r"(h_noise[13] & 536813567), "r"(h_noise[14] & 536813567), "r"(h_noise[15] & 536813567) : "memory");
                unsigned int key0_1 = seed_lo;
                unsigned int key1_1 = seed_hi;
                long long o_8 = read_rows_base + 2ll * state_stride_p2;
                unsigned long long p0_8 = (unsigned long long)(unsigned int)o_8 * 3528531795ull;
                h_noise[0] = (unsigned int)(o_8 >> 32ll) ^ key0_1;
                h_noise[2] = (unsigned int)(p0_8 >> 32ull) ^ key1_1;
                h_noise[3] = (unsigned int)p0_8;
                long long o_9 = read_rows_base + 2ll * state_stride_p2 + 16ll;
                unsigned long long p0_9 = (unsigned long long)(unsigned int)o_9 * 3528531795ull;
                h_noise[4] = (unsigned int)(o_9 >> 32ll) ^ key0_1;
                h_noise[6] = (unsigned int)(p0_9 >> 32ull) ^ key1_1;
                h_noise[7] = (unsigned int)p0_9;
                long long o_10 = read_rows_base + 2ll * state_stride_p2 + 32ll;
                unsigned long long p0_10 = (unsigned long long)(unsigned int)o_10 * 3528531795ull;
                h_noise[8] = (unsigned int)(o_10 >> 32ll) ^ key0_1;
                h_noise[10] = (unsigned int)(p0_10 >> 32ull) ^ key1_1;
                h_noise[11] = (unsigned int)p0_10;
                long long o_11 = read_rows_base + 2ll * state_stride_p2 + 48ll;
                unsigned long long p0_11 = (unsigned long long)(unsigned int)o_11 * 3528531795ull;
                h_noise[12] = (unsigned int)(o_11 >> 32ll) ^ key0_1;
                h_noise[14] = (unsigned int)(p0_11 >> 32ull) ^ key1_1;
                h_noise[15] = (unsigned int)p0_11;
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
                key0_1 = key0_1 + 2654435769u;
                key1_1 = key1_1 + 3144134277u;
                #pragma unroll 1
                for (int _round_2 = 5; _round_2 < philox_rounds; _round_2++) {
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
                    key0_1 = key0_1 + 2654435769u;
                    key1_1 = key1_1 + 3144134277u;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 8192), "r"(h_noise[0] & 536813567), "r"(h_noise[1] & 536813567), "r"(h_noise[2] & 536813567), "r"(h_noise[3] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 9216), "r"(h_noise[4] & 536813567), "r"(h_noise[5] & 536813567), "r"(h_noise[6] & 536813567), "r"(h_noise[7] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 10240), "r"(h_noise[8] & 536813567), "r"(h_noise[9] & 536813567), "r"(h_noise[10] & 536813567), "r"(h_noise[11] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 11264), "r"(h_noise[12] & 536813567), "r"(h_noise[13] & 536813567), "r"(h_noise[14] & 536813567), "r"(h_noise[15] & 536813567) : "memory");
                unsigned int key0_2 = seed_lo;
                unsigned int key1_2 = seed_hi;
                long long o_12 = read_rows_base + 3ll * state_stride_p2;
                unsigned long long p0_12 = (unsigned long long)(unsigned int)o_12 * 3528531795ull;
                h_noise[0] = (unsigned int)(o_12 >> 32ll) ^ key0_2;
                h_noise[2] = (unsigned int)(p0_12 >> 32ull) ^ key1_2;
                h_noise[3] = (unsigned int)p0_12;
                long long o_13 = read_rows_base + 3ll * state_stride_p2 + 16ll;
                unsigned long long p0_13 = (unsigned long long)(unsigned int)o_13 * 3528531795ull;
                h_noise[4] = (unsigned int)(o_13 >> 32ll) ^ key0_2;
                h_noise[6] = (unsigned int)(p0_13 >> 32ull) ^ key1_2;
                h_noise[7] = (unsigned int)p0_13;
                long long o_14 = read_rows_base + 3ll * state_stride_p2 + 32ll;
                unsigned long long p0_14 = (unsigned long long)(unsigned int)o_14 * 3528531795ull;
                h_noise[8] = (unsigned int)(o_14 >> 32ll) ^ key0_2;
                h_noise[10] = (unsigned int)(p0_14 >> 32ull) ^ key1_2;
                h_noise[11] = (unsigned int)p0_14;
                long long o_15 = read_rows_base + 3ll * state_stride_p2 + 48ll;
                unsigned long long p0_15 = (unsigned long long)(unsigned int)o_15 * 3528531795ull;
                h_noise[12] = (unsigned int)(o_15 >> 32ll) ^ key0_2;
                h_noise[14] = (unsigned int)(p0_15 >> 32ull) ^ key1_2;
                h_noise[15] = (unsigned int)p0_15;
                key0_2 = key0_2 + 2654435769u;
                key1_2 = key1_2 + 3144134277u;
                unsigned long long q0_12 = (unsigned long long)h_noise[0] * 3528531795ull;
                unsigned long long q1_12 = (unsigned long long)h_noise[2] * 3449720151ull;
                h_noise[0] = (unsigned int)(q1_12 >> 32ull) ^ key0_2;
                h_noise[1] = (unsigned int)q1_12;
                h_noise[2] = (unsigned int)(q0_12 >> 32ull) ^ h_noise[3] ^ key1_2;
                h_noise[3] = (unsigned int)q0_12;
                unsigned long long q0_13 = (unsigned long long)h_noise[4] * 3528531795ull;
                unsigned long long q1_13 = (unsigned long long)h_noise[6] * 3449720151ull;
                h_noise[4] = (unsigned int)(q1_13 >> 32ull) ^ key0_2;
                h_noise[5] = (unsigned int)q1_13;
                h_noise[6] = (unsigned int)(q0_13 >> 32ull) ^ h_noise[7] ^ key1_2;
                h_noise[7] = (unsigned int)q0_13;
                unsigned long long q0_14 = (unsigned long long)h_noise[8] * 3528531795ull;
                unsigned long long q1_14 = (unsigned long long)h_noise[10] * 3449720151ull;
                h_noise[8] = (unsigned int)(q1_14 >> 32ull) ^ key0_2;
                h_noise[9] = (unsigned int)q1_14;
                h_noise[10] = (unsigned int)(q0_14 >> 32ull) ^ h_noise[11] ^ key1_2;
                h_noise[11] = (unsigned int)q0_14;
                unsigned long long q0_15 = (unsigned long long)h_noise[12] * 3528531795ull;
                unsigned long long q1_15 = (unsigned long long)h_noise[14] * 3449720151ull;
                h_noise[12] = (unsigned int)(q1_15 >> 32ull) ^ key0_2;
                h_noise[13] = (unsigned int)q1_15;
                h_noise[14] = (unsigned int)(q0_15 >> 32ull) ^ h_noise[15] ^ key1_2;
                h_noise[15] = (unsigned int)q0_15;
                key0_2 = key0_2 + 2654435769u;
                key1_2 = key1_2 + 3144134277u;
                unsigned long long s0_36 = (unsigned long long)h_noise[0] * 3528531795ull;
                unsigned long long s1_36 = (unsigned long long)h_noise[2] * 3449720151ull;
                h_noise[0] = (unsigned int)(s1_36 >> 32ull) ^ h_noise[1] ^ key0_2;
                h_noise[1] = (unsigned int)s1_36;
                h_noise[2] = (unsigned int)(s0_36 >> 32ull) ^ h_noise[3] ^ key1_2;
                h_noise[3] = (unsigned int)s0_36;
                unsigned long long s0_37 = (unsigned long long)h_noise[4] * 3528531795ull;
                unsigned long long s1_37 = (unsigned long long)h_noise[6] * 3449720151ull;
                h_noise[4] = (unsigned int)(s1_37 >> 32ull) ^ h_noise[5] ^ key0_2;
                h_noise[5] = (unsigned int)s1_37;
                h_noise[6] = (unsigned int)(s0_37 >> 32ull) ^ h_noise[7] ^ key1_2;
                h_noise[7] = (unsigned int)s0_37;
                unsigned long long s0_38 = (unsigned long long)h_noise[8] * 3528531795ull;
                unsigned long long s1_38 = (unsigned long long)h_noise[10] * 3449720151ull;
                h_noise[8] = (unsigned int)(s1_38 >> 32ull) ^ h_noise[9] ^ key0_2;
                h_noise[9] = (unsigned int)s1_38;
                h_noise[10] = (unsigned int)(s0_38 >> 32ull) ^ h_noise[11] ^ key1_2;
                h_noise[11] = (unsigned int)s0_38;
                unsigned long long s0_39 = (unsigned long long)h_noise[12] * 3528531795ull;
                unsigned long long s1_39 = (unsigned long long)h_noise[14] * 3449720151ull;
                h_noise[12] = (unsigned int)(s1_39 >> 32ull) ^ h_noise[13] ^ key0_2;
                h_noise[13] = (unsigned int)s1_39;
                h_noise[14] = (unsigned int)(s0_39 >> 32ull) ^ h_noise[15] ^ key1_2;
                h_noise[15] = (unsigned int)s0_39;
                key0_2 = key0_2 + 2654435769u;
                key1_2 = key1_2 + 3144134277u;
                unsigned long long s0_40 = (unsigned long long)h_noise[0] * 3528531795ull;
                unsigned long long s1_40 = (unsigned long long)h_noise[2] * 3449720151ull;
                h_noise[0] = (unsigned int)(s1_40 >> 32ull) ^ h_noise[1] ^ key0_2;
                h_noise[1] = (unsigned int)s1_40;
                h_noise[2] = (unsigned int)(s0_40 >> 32ull) ^ h_noise[3] ^ key1_2;
                h_noise[3] = (unsigned int)s0_40;
                unsigned long long s0_41 = (unsigned long long)h_noise[4] * 3528531795ull;
                unsigned long long s1_41 = (unsigned long long)h_noise[6] * 3449720151ull;
                h_noise[4] = (unsigned int)(s1_41 >> 32ull) ^ h_noise[5] ^ key0_2;
                h_noise[5] = (unsigned int)s1_41;
                h_noise[6] = (unsigned int)(s0_41 >> 32ull) ^ h_noise[7] ^ key1_2;
                h_noise[7] = (unsigned int)s0_41;
                unsigned long long s0_42 = (unsigned long long)h_noise[8] * 3528531795ull;
                unsigned long long s1_42 = (unsigned long long)h_noise[10] * 3449720151ull;
                h_noise[8] = (unsigned int)(s1_42 >> 32ull) ^ h_noise[9] ^ key0_2;
                h_noise[9] = (unsigned int)s1_42;
                h_noise[10] = (unsigned int)(s0_42 >> 32ull) ^ h_noise[11] ^ key1_2;
                h_noise[11] = (unsigned int)s0_42;
                unsigned long long s0_43 = (unsigned long long)h_noise[12] * 3528531795ull;
                unsigned long long s1_43 = (unsigned long long)h_noise[14] * 3449720151ull;
                h_noise[12] = (unsigned int)(s1_43 >> 32ull) ^ h_noise[13] ^ key0_2;
                h_noise[13] = (unsigned int)s1_43;
                h_noise[14] = (unsigned int)(s0_43 >> 32ull) ^ h_noise[15] ^ key1_2;
                h_noise[15] = (unsigned int)s0_43;
                key0_2 = key0_2 + 2654435769u;
                key1_2 = key1_2 + 3144134277u;
                unsigned long long s0_44 = (unsigned long long)h_noise[0] * 3528531795ull;
                unsigned long long s1_44 = (unsigned long long)h_noise[2] * 3449720151ull;
                h_noise[0] = (unsigned int)(s1_44 >> 32ull) ^ h_noise[1] ^ key0_2;
                h_noise[1] = (unsigned int)s1_44;
                h_noise[2] = (unsigned int)(s0_44 >> 32ull) ^ h_noise[3] ^ key1_2;
                h_noise[3] = (unsigned int)s0_44;
                unsigned long long s0_45 = (unsigned long long)h_noise[4] * 3528531795ull;
                unsigned long long s1_45 = (unsigned long long)h_noise[6] * 3449720151ull;
                h_noise[4] = (unsigned int)(s1_45 >> 32ull) ^ h_noise[5] ^ key0_2;
                h_noise[5] = (unsigned int)s1_45;
                h_noise[6] = (unsigned int)(s0_45 >> 32ull) ^ h_noise[7] ^ key1_2;
                h_noise[7] = (unsigned int)s0_45;
                unsigned long long s0_46 = (unsigned long long)h_noise[8] * 3528531795ull;
                unsigned long long s1_46 = (unsigned long long)h_noise[10] * 3449720151ull;
                h_noise[8] = (unsigned int)(s1_46 >> 32ull) ^ h_noise[9] ^ key0_2;
                h_noise[9] = (unsigned int)s1_46;
                h_noise[10] = (unsigned int)(s0_46 >> 32ull) ^ h_noise[11] ^ key1_2;
                h_noise[11] = (unsigned int)s0_46;
                unsigned long long s0_47 = (unsigned long long)h_noise[12] * 3528531795ull;
                unsigned long long s1_47 = (unsigned long long)h_noise[14] * 3449720151ull;
                h_noise[12] = (unsigned int)(s1_47 >> 32ull) ^ h_noise[13] ^ key0_2;
                h_noise[13] = (unsigned int)s1_47;
                h_noise[14] = (unsigned int)(s0_47 >> 32ull) ^ h_noise[15] ^ key1_2;
                h_noise[15] = (unsigned int)s0_47;
                key0_2 = key0_2 + 2654435769u;
                key1_2 = key1_2 + 3144134277u;
                #pragma unroll 1
                for (int _round_3 = 5; _round_3 < philox_rounds; _round_3++) {
                    unsigned long long t0_12 = (unsigned long long)h_noise[0] * 3528531795ull;
                    unsigned long long t1_12 = (unsigned long long)h_noise[2] * 3449720151ull;
                    h_noise[0] = (unsigned int)(t1_12 >> 32ull) ^ h_noise[1] ^ key0_2;
                    h_noise[1] = (unsigned int)t1_12;
                    h_noise[2] = (unsigned int)(t0_12 >> 32ull) ^ h_noise[3] ^ key1_2;
                    h_noise[3] = (unsigned int)t0_12;
                    unsigned long long t0_13 = (unsigned long long)h_noise[4] * 3528531795ull;
                    unsigned long long t1_13 = (unsigned long long)h_noise[6] * 3449720151ull;
                    h_noise[4] = (unsigned int)(t1_13 >> 32ull) ^ h_noise[5] ^ key0_2;
                    h_noise[5] = (unsigned int)t1_13;
                    h_noise[6] = (unsigned int)(t0_13 >> 32ull) ^ h_noise[7] ^ key1_2;
                    h_noise[7] = (unsigned int)t0_13;
                    unsigned long long t0_14 = (unsigned long long)h_noise[8] * 3528531795ull;
                    unsigned long long t1_14 = (unsigned long long)h_noise[10] * 3449720151ull;
                    h_noise[8] = (unsigned int)(t1_14 >> 32ull) ^ h_noise[9] ^ key0_2;
                    h_noise[9] = (unsigned int)t1_14;
                    h_noise[10] = (unsigned int)(t0_14 >> 32ull) ^ h_noise[11] ^ key1_2;
                    h_noise[11] = (unsigned int)t0_14;
                    unsigned long long t0_15 = (unsigned long long)h_noise[12] * 3528531795ull;
                    unsigned long long t1_15 = (unsigned long long)h_noise[14] * 3449720151ull;
                    h_noise[12] = (unsigned int)(t1_15 >> 32ull) ^ h_noise[13] ^ key0_2;
                    h_noise[13] = (unsigned int)t1_15;
                    h_noise[14] = (unsigned int)(t0_15 >> 32ull) ^ h_noise[15] ^ key1_2;
                    h_noise[15] = (unsigned int)t0_15;
                    key0_2 = key0_2 + 2654435769u;
                    key1_2 = key1_2 + 3144134277u;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 12288), "r"(h_noise[0] & 536813567), "r"(h_noise[1] & 536813567), "r"(h_noise[2] & 536813567), "r"(h_noise[3] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 13312), "r"(h_noise[4] & 536813567), "r"(h_noise[5] & 536813567), "r"(h_noise[6] & 536813567), "r"(h_noise[7] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 14336), "r"(h_noise[8] & 536813567), "r"(h_noise[9] & 536813567), "r"(h_noise[10] & 536813567), "r"(h_noise[11] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 15360), "r"(h_noise[12] & 536813567), "r"(h_noise[13] & 536813567), "r"(h_noise[14] & 536813567), "r"(h_noise[15] & 536813567) : "memory");
#else
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 4096), "r"(h_noise[16] & 536813567), "r"(h_noise[17] & 536813567), "r"(h_noise[18] & 536813567), "r"(h_noise[19] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 5120), "r"(h_noise[20] & 536813567), "r"(h_noise[21] & 536813567), "r"(h_noise[22] & 536813567), "r"(h_noise[23] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 6144), "r"(h_noise[24] & 536813567), "r"(h_noise[25] & 536813567), "r"(h_noise[26] & 536813567), "r"(h_noise[27] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 7168), "r"(h_noise[28] & 536813567), "r"(h_noise[29] & 536813567), "r"(h_noise[30] & 536813567), "r"(h_noise[31] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 8192), "r"(h_noise[32] & 536813567), "r"(h_noise[33] & 536813567), "r"(h_noise[34] & 536813567), "r"(h_noise[35] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 9216), "r"(h_noise[36] & 536813567), "r"(h_noise[37] & 536813567), "r"(h_noise[38] & 536813567), "r"(h_noise[39] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 10240), "r"(h_noise[40] & 536813567), "r"(h_noise[41] & 536813567), "r"(h_noise[42] & 536813567), "r"(h_noise[43] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 11264), "r"(h_noise[44] & 536813567), "r"(h_noise[45] & 536813567), "r"(h_noise[46] & 536813567), "r"(h_noise[47] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 12288), "r"(h_noise[48] & 536813567), "r"(h_noise[49] & 536813567), "r"(h_noise[50] & 536813567), "r"(h_noise[51] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 13312), "r"(h_noise[52] & 536813567), "r"(h_noise[53] & 536813567), "r"(h_noise[54] & 536813567), "r"(h_noise[55] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 14336), "r"(h_noise[56] & 536813567), "r"(h_noise[57] & 536813567), "r"(h_noise[58] & 536813567), "r"(h_noise[59] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 15360), "r"(h_noise[60] & 536813567), "r"(h_noise[61] & 536813567), "r"(h_noise[62] & 536813567), "r"(h_noise[63] & 536813567) : "memory");
#endif
            }
            #pragma unroll 1
            for (int t_1 = 0; t_1 < num_tokens; t_1++) {
                float alpha = sAlpha[t_1];
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
                S[64] = mul_f32_rn_ftz(S[64], alpha);
                S[65] = mul_f32_rn_ftz(S[65], alpha);
                S[66] = mul_f32_rn_ftz(S[66], alpha);
                S[67] = mul_f32_rn_ftz(S[67], alpha);
                S[68] = mul_f32_rn_ftz(S[68], alpha);
                S[69] = mul_f32_rn_ftz(S[69], alpha);
                S[70] = mul_f32_rn_ftz(S[70], alpha);
                S[71] = mul_f32_rn_ftz(S[71], alpha);
                S[72] = mul_f32_rn_ftz(S[72], alpha);
                S[73] = mul_f32_rn_ftz(S[73], alpha);
                S[74] = mul_f32_rn_ftz(S[74], alpha);
                S[75] = mul_f32_rn_ftz(S[75], alpha);
                S[76] = mul_f32_rn_ftz(S[76], alpha);
                S[77] = mul_f32_rn_ftz(S[77], alpha);
                S[78] = mul_f32_rn_ftz(S[78], alpha);
                S[79] = mul_f32_rn_ftz(S[79], alpha);
                S[80] = mul_f32_rn_ftz(S[80], alpha);
                S[81] = mul_f32_rn_ftz(S[81], alpha);
                S[82] = mul_f32_rn_ftz(S[82], alpha);
                S[83] = mul_f32_rn_ftz(S[83], alpha);
                S[84] = mul_f32_rn_ftz(S[84], alpha);
                S[85] = mul_f32_rn_ftz(S[85], alpha);
                S[86] = mul_f32_rn_ftz(S[86], alpha);
                S[87] = mul_f32_rn_ftz(S[87], alpha);
                S[88] = mul_f32_rn_ftz(S[88], alpha);
                S[89] = mul_f32_rn_ftz(S[89], alpha);
                S[90] = mul_f32_rn_ftz(S[90], alpha);
                S[91] = mul_f32_rn_ftz(S[91], alpha);
                S[92] = mul_f32_rn_ftz(S[92], alpha);
                S[93] = mul_f32_rn_ftz(S[93], alpha);
                S[94] = mul_f32_rn_ftz(S[94], alpha);
                S[95] = mul_f32_rn_ftz(S[95], alpha);
                S[96] = mul_f32_rn_ftz(S[96], alpha);
                S[97] = mul_f32_rn_ftz(S[97], alpha);
                S[98] = mul_f32_rn_ftz(S[98], alpha);
                S[99] = mul_f32_rn_ftz(S[99], alpha);
                S[100] = mul_f32_rn_ftz(S[100], alpha);
                S[101] = mul_f32_rn_ftz(S[101], alpha);
                S[102] = mul_f32_rn_ftz(S[102], alpha);
                S[103] = mul_f32_rn_ftz(S[103], alpha);
                S[104] = mul_f32_rn_ftz(S[104], alpha);
                S[105] = mul_f32_rn_ftz(S[105], alpha);
                S[106] = mul_f32_rn_ftz(S[106], alpha);
                S[107] = mul_f32_rn_ftz(S[107], alpha);
                S[108] = mul_f32_rn_ftz(S[108], alpha);
                S[109] = mul_f32_rn_ftz(S[109], alpha);
                S[110] = mul_f32_rn_ftz(S[110], alpha);
                S[111] = mul_f32_rn_ftz(S[111], alpha);
                S[112] = mul_f32_rn_ftz(S[112], alpha);
                S[113] = mul_f32_rn_ftz(S[113], alpha);
                S[114] = mul_f32_rn_ftz(S[114], alpha);
                S[115] = mul_f32_rn_ftz(S[115], alpha);
                S[116] = mul_f32_rn_ftz(S[116], alpha);
                S[117] = mul_f32_rn_ftz(S[117], alpha);
                S[118] = mul_f32_rn_ftz(S[118], alpha);
                S[119] = mul_f32_rn_ftz(S[119], alpha);
                S[120] = mul_f32_rn_ftz(S[120], alpha);
                S[121] = mul_f32_rn_ftz(S[121], alpha);
                S[122] = mul_f32_rn_ftz(S[122], alpha);
                S[123] = mul_f32_rn_ftz(S[123], alpha);
                S[124] = mul_f32_rn_ftz(S[124], alpha);
                S[125] = mul_f32_rn_ftz(S[125], alpha);
                S[126] = mul_f32_rn_ftz(S[126], alpha);
                S[127] = mul_f32_rn_ftz(S[127], alpha);
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
                S2[32] = mul_f32x2(S2[32], alpha2);
                S2[33] = mul_f32x2(S2[33], alpha2);
                S2[34] = mul_f32x2(S2[34], alpha2);
                S2[35] = mul_f32x2(S2[35], alpha2);
                S2[36] = mul_f32x2(S2[36], alpha2);
                S2[37] = mul_f32x2(S2[37], alpha2);
                S2[38] = mul_f32x2(S2[38], alpha2);
                S2[39] = mul_f32x2(S2[39], alpha2);
                S2[40] = mul_f32x2(S2[40], alpha2);
                S2[41] = mul_f32x2(S2[41], alpha2);
                S2[42] = mul_f32x2(S2[42], alpha2);
                S2[43] = mul_f32x2(S2[43], alpha2);
                S2[44] = mul_f32x2(S2[44], alpha2);
                S2[45] = mul_f32x2(S2[45], alpha2);
                S2[46] = mul_f32x2(S2[46], alpha2);
                S2[47] = mul_f32x2(S2[47], alpha2);
                S2[48] = mul_f32x2(S2[48], alpha2);
                S2[49] = mul_f32x2(S2[49], alpha2);
                S2[50] = mul_f32x2(S2[50], alpha2);
                S2[51] = mul_f32x2(S2[51], alpha2);
                S2[52] = mul_f32x2(S2[52], alpha2);
                S2[53] = mul_f32x2(S2[53], alpha2);
                S2[54] = mul_f32x2(S2[54], alpha2);
                S2[55] = mul_f32x2(S2[55], alpha2);
                S2[56] = mul_f32x2(S2[56], alpha2);
                S2[57] = mul_f32x2(S2[57], alpha2);
                S2[58] = mul_f32x2(S2[58], alpha2);
                S2[59] = mul_f32x2(S2[59], alpha2);
                S2[60] = mul_f32x2(S2[60], alpha2);
                S2[61] = mul_f32x2(S2[61], alpha2);
                S2[62] = mul_f32x2(S2[62], alpha2);
                S2[63] = mul_f32x2(S2[63], alpha2);
#endif
                int m_0 = t_1;
                unsigned int kbase_0 = sK_addr + (unsigned int)(m_0 * 512) + lane_byte;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(kbase_0));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(kbase_0 + 64));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(kbase_0 + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(kbase_0 + 192));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[16])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 3]))
                    : "r"(kbase_0 + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[20])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 3]))
                    : "r"(kbase_0 + 320));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[24])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 3]))
                    : "r"(kbase_0 + 384));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[28])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 3]))
                    : "r"(kbase_0 + 448));
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
                k2[8] = make_float2(k_regs[16], k_regs[17]);
                k2[9] = make_float2(k_regs[18], k_regs[19]);
                k2[10] = make_float2(k_regs[20], k_regs[21]);
                k2[11] = make_float2(k_regs[22], k_regs[23]);
                k2[12] = make_float2(k_regs[24], k_regs[25]);
                k2[13] = make_float2(k_regs[26], k_regs[27]);
                k2[14] = make_float2(k_regs[28], k_regs[29]);
                k2[15] = make_float2(k_regs[30], k_regs[31]);
#endif
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                    : "r"(sV_addr + (unsigned int)(m_0 * 256) + group_byte));
                float beta_0 = sBeta[m_0];
#if __CUDA_ARCH__ == 900
                float acc0_x = 0.0f;
                float acc0_y = 0.0f;
                float acc1_x = 0.0f;
                float acc1_y = 0.0f;
                float _fma_46;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x));
                acc0_x = _fma_46;
                float _fma_47;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y));
                acc0_y = _fma_47;
                float _fma_48;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x));
                acc1_x = _fma_48;
                float _fma_49;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y));
                acc1_y = _fma_49;
                float _fma_50;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x));
                acc0_x = _fma_50;
                float _fma_51;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y));
                acc0_y = _fma_51;
                float _fma_52;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x));
                acc1_x = _fma_52;
                float _fma_53;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y));
                acc1_y = _fma_53;
                float _fma_54;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x));
                acc0_x = _fma_54;
                float _fma_55;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y));
                acc0_y = _fma_55;
                float _fma_56;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x));
                acc1_x = _fma_56;
                float _fma_57;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y));
                acc1_y = _fma_57;
                float _fma_58;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x));
                acc0_x = _fma_58;
                float _fma_59;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y));
                acc0_y = _fma_59;
                float _fma_60;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x));
                acc1_x = _fma_60;
                float _fma_61;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y));
                acc1_y = _fma_61;
                float _fma_62;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(S[16]), "f"(k_regs[16]), "f"(acc0_x));
                acc0_x = _fma_62;
                float _fma_63;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(S[17]), "f"(k_regs[17]), "f"(acc0_y));
                acc0_y = _fma_63;
                float _fma_64;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(S[18]), "f"(k_regs[18]), "f"(acc1_x));
                acc1_x = _fma_64;
                float _fma_65;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(S[19]), "f"(k_regs[19]), "f"(acc1_y));
                acc1_y = _fma_65;
                float _fma_66;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(S[20]), "f"(k_regs[20]), "f"(acc0_x));
                acc0_x = _fma_66;
                float _fma_67;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(S[21]), "f"(k_regs[21]), "f"(acc0_y));
                acc0_y = _fma_67;
                float _fma_68;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(S[22]), "f"(k_regs[22]), "f"(acc1_x));
                acc1_x = _fma_68;
                float _fma_69;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(S[23]), "f"(k_regs[23]), "f"(acc1_y));
                acc1_y = _fma_69;
                float _fma_70;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(S[24]), "f"(k_regs[24]), "f"(acc0_x));
                acc0_x = _fma_70;
                float _fma_71;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(S[25]), "f"(k_regs[25]), "f"(acc0_y));
                acc0_y = _fma_71;
                float _fma_72;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(S[26]), "f"(k_regs[26]), "f"(acc1_x));
                acc1_x = _fma_72;
                float _fma_73;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(S[27]), "f"(k_regs[27]), "f"(acc1_y));
                acc1_y = _fma_73;
                float _fma_74;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(S[28]), "f"(k_regs[28]), "f"(acc0_x));
                acc0_x = _fma_74;
                float _fma_75;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(S[29]), "f"(k_regs[29]), "f"(acc0_y));
                acc0_y = _fma_75;
                float _fma_76;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(S[30]), "f"(k_regs[30]), "f"(acc1_x));
                acc1_x = _fma_76;
                float _fma_77;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(S[31]), "f"(k_regs[31]), "f"(acc1_y));
                acc1_y = _fma_77;
                float _add_78;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(acc0_x), "f"(acc1_x));
                float _add_79;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(acc0_y), "f"(acc1_y));
                float dot = _add_78 + _add_79;
                float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
                dot = dot + _shfl_xor_sync_80;
                float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
                dot = dot + _shfl_xor_sync_81;
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
                acc0 = fma_f32x2_rn_ftz(S2[8], k2[8], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[9], k2[9], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[10], k2[10], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[11], k2[11], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[12], k2[12], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[13], k2[13], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[14], k2[14], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[15], k2[15], acc0);
                float dot = acc0.x + acc0.y;
                float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
                dot = dot + _shfl_xor_sync_42;
                float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
                dot = dot + _shfl_xor_sync_43;
#endif
                float vn_0 = (v_regs[0] - dot) * beta_0;
#if __CUDA_ARCH__ == 900
                float _fma_82;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(k_regs[0]), "f"(vn_0), "f"(S[0]));
                S[0] = _fma_82;
                float _fma_83;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(k_regs[1]), "f"(vn_0), "f"(S[1]));
                S[1] = _fma_83;
                float _fma_84;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(k_regs[2]), "f"(vn_0), "f"(S[2]));
                S[2] = _fma_84;
                float _fma_85;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(k_regs[3]), "f"(vn_0), "f"(S[3]));
                S[3] = _fma_85;
                float _fma_86;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(k_regs[4]), "f"(vn_0), "f"(S[4]));
                S[4] = _fma_86;
                float _fma_87;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(k_regs[5]), "f"(vn_0), "f"(S[5]));
                S[5] = _fma_87;
                float _fma_88;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(k_regs[6]), "f"(vn_0), "f"(S[6]));
                S[6] = _fma_88;
                float _fma_89;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(k_regs[7]), "f"(vn_0), "f"(S[7]));
                S[7] = _fma_89;
                float _fma_90;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(k_regs[8]), "f"(vn_0), "f"(S[8]));
                S[8] = _fma_90;
                float _fma_91;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(k_regs[9]), "f"(vn_0), "f"(S[9]));
                S[9] = _fma_91;
                float _fma_92;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(k_regs[10]), "f"(vn_0), "f"(S[10]));
                S[10] = _fma_92;
                float _fma_93;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(k_regs[11]), "f"(vn_0), "f"(S[11]));
                S[11] = _fma_93;
                float _fma_94;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(k_regs[12]), "f"(vn_0), "f"(S[12]));
                S[12] = _fma_94;
                float _fma_95;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(k_regs[13]), "f"(vn_0), "f"(S[13]));
                S[13] = _fma_95;
                float _fma_96;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(k_regs[14]), "f"(vn_0), "f"(S[14]));
                S[14] = _fma_96;
                float _fma_97;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(k_regs[15]), "f"(vn_0), "f"(S[15]));
                S[15] = _fma_97;
                float _fma_98;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(k_regs[16]), "f"(vn_0), "f"(S[16]));
                S[16] = _fma_98;
                float _fma_99;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(k_regs[17]), "f"(vn_0), "f"(S[17]));
                S[17] = _fma_99;
                float _fma_100;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(k_regs[18]), "f"(vn_0), "f"(S[18]));
                S[18] = _fma_100;
                float _fma_101;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(k_regs[19]), "f"(vn_0), "f"(S[19]));
                S[19] = _fma_101;
                float _fma_102;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(k_regs[20]), "f"(vn_0), "f"(S[20]));
                S[20] = _fma_102;
                float _fma_103;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(k_regs[21]), "f"(vn_0), "f"(S[21]));
                S[21] = _fma_103;
                float _fma_104;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(k_regs[22]), "f"(vn_0), "f"(S[22]));
                S[22] = _fma_104;
                float _fma_105;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(k_regs[23]), "f"(vn_0), "f"(S[23]));
                S[23] = _fma_105;
                float _fma_106;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(k_regs[24]), "f"(vn_0), "f"(S[24]));
                S[24] = _fma_106;
                float _fma_107;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(k_regs[25]), "f"(vn_0), "f"(S[25]));
                S[25] = _fma_107;
                float _fma_108;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(k_regs[26]), "f"(vn_0), "f"(S[26]));
                S[26] = _fma_108;
                float _fma_109;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(k_regs[27]), "f"(vn_0), "f"(S[27]));
                S[27] = _fma_109;
                float _fma_110;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(k_regs[28]), "f"(vn_0), "f"(S[28]));
                S[28] = _fma_110;
                float _fma_111;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_111) : "f"(k_regs[29]), "f"(vn_0), "f"(S[29]));
                S[29] = _fma_111;
                float _fma_112;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_112) : "f"(k_regs[30]), "f"(vn_0), "f"(S[30]));
                S[30] = _fma_112;
                float _fma_113;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_113) : "f"(k_regs[31]), "f"(vn_0), "f"(S[31]));
                S[31] = _fma_113;
                float acc0_x_0 = 0.0f;
                float acc0_y_0 = 0.0f;
                float acc1_x_0 = 0.0f;
                float acc1_y_0 = 0.0f;
                float _fma_114;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_114) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_0));
                acc0_x_0 = _fma_114;
                float _fma_115;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_0));
                acc0_y_0 = _fma_115;
                float _fma_116;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(S[34]), "f"(k_regs[2]), "f"(acc1_x_0));
                acc1_x_0 = _fma_116;
                float _fma_117;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(S[35]), "f"(k_regs[3]), "f"(acc1_y_0));
                acc1_y_0 = _fma_117;
                float _fma_118;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_118) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_0));
                acc0_x_0 = _fma_118;
                float _fma_119;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_0));
                acc0_y_0 = _fma_119;
                float _fma_120;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(S[38]), "f"(k_regs[6]), "f"(acc1_x_0));
                acc1_x_0 = _fma_120;
                float _fma_121;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(S[39]), "f"(k_regs[7]), "f"(acc1_y_0));
                acc1_y_0 = _fma_121;
                float _fma_122;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_0));
                acc0_x_0 = _fma_122;
                float _fma_123;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_123) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_0));
                acc0_y_0 = _fma_123;
                float _fma_124;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(S[42]), "f"(k_regs[10]), "f"(acc1_x_0));
                acc1_x_0 = _fma_124;
                float _fma_125;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(S[43]), "f"(k_regs[11]), "f"(acc1_y_0));
                acc1_y_0 = _fma_125;
                float _fma_126;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_0));
                acc0_x_0 = _fma_126;
                float _fma_127;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_0));
                acc0_y_0 = _fma_127;
                float _fma_128;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_128) : "f"(S[46]), "f"(k_regs[14]), "f"(acc1_x_0));
                acc1_x_0 = _fma_128;
                float _fma_129;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_129) : "f"(S[47]), "f"(k_regs[15]), "f"(acc1_y_0));
                acc1_y_0 = _fma_129;
                float _fma_130;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_130) : "f"(S[48]), "f"(k_regs[16]), "f"(acc0_x_0));
                acc0_x_0 = _fma_130;
                float _fma_131;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(S[49]), "f"(k_regs[17]), "f"(acc0_y_0));
                acc0_y_0 = _fma_131;
                float _fma_132;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(S[50]), "f"(k_regs[18]), "f"(acc1_x_0));
                acc1_x_0 = _fma_132;
                float _fma_133;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(S[51]), "f"(k_regs[19]), "f"(acc1_y_0));
                acc1_y_0 = _fma_133;
                float _fma_134;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_134) : "f"(S[52]), "f"(k_regs[20]), "f"(acc0_x_0));
                acc0_x_0 = _fma_134;
                float _fma_135;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_135) : "f"(S[53]), "f"(k_regs[21]), "f"(acc0_y_0));
                acc0_y_0 = _fma_135;
                float _fma_136;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_136) : "f"(S[54]), "f"(k_regs[22]), "f"(acc1_x_0));
                acc1_x_0 = _fma_136;
                float _fma_137;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_137) : "f"(S[55]), "f"(k_regs[23]), "f"(acc1_y_0));
                acc1_y_0 = _fma_137;
                float _fma_138;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_138) : "f"(S[56]), "f"(k_regs[24]), "f"(acc0_x_0));
                acc0_x_0 = _fma_138;
                float _fma_139;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_139) : "f"(S[57]), "f"(k_regs[25]), "f"(acc0_y_0));
                acc0_y_0 = _fma_139;
                float _fma_140;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_140) : "f"(S[58]), "f"(k_regs[26]), "f"(acc1_x_0));
                acc1_x_0 = _fma_140;
                float _fma_141;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_141) : "f"(S[59]), "f"(k_regs[27]), "f"(acc1_y_0));
                acc1_y_0 = _fma_141;
                float _fma_142;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(S[60]), "f"(k_regs[28]), "f"(acc0_x_0));
                acc0_x_0 = _fma_142;
                float _fma_143;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_143) : "f"(S[61]), "f"(k_regs[29]), "f"(acc0_y_0));
                acc0_y_0 = _fma_143;
                float _fma_144;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_144) : "f"(S[62]), "f"(k_regs[30]), "f"(acc1_x_0));
                acc1_x_0 = _fma_144;
                float _fma_145;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_145) : "f"(S[63]), "f"(k_regs[31]), "f"(acc1_y_0));
                acc1_y_0 = _fma_145;
                float _add_146;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_146) : "f"(acc0_x_0), "f"(acc1_x_0));
                float _add_147;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_147) : "f"(acc0_y_0), "f"(acc1_y_0));
                float dot_0 = _add_146 + _add_147;
                float _shfl_xor_sync_148 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 1);
                dot_0 = dot_0 + _shfl_xor_sync_148;
                float _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 2);
                dot_0 = dot_0 + _shfl_xor_sync_149;
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
                S2[8] = fma_f32x2_rn_ftz(k2[8], vn2_0, S2[8]);
                S2[9] = fma_f32x2_rn_ftz(k2[9], vn2_0, S2[9]);
                S2[10] = fma_f32x2_rn_ftz(k2[10], vn2_0, S2[10]);
                S2[11] = fma_f32x2_rn_ftz(k2[11], vn2_0, S2[11]);
                S2[12] = fma_f32x2_rn_ftz(k2[12], vn2_0, S2[12]);
                S2[13] = fma_f32x2_rn_ftz(k2[13], vn2_0, S2[13]);
                S2[14] = fma_f32x2_rn_ftz(k2[14], vn2_0, S2[14]);
                S2[15] = fma_f32x2_rn_ftz(k2[15], vn2_0, S2[15]);
                float2 acc0_0 = make_float2(0.0f, 0.0f);
                acc0_0 = fma_f32x2_rn_ftz(S2[16], k2[0], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[17], k2[1], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[18], k2[2], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[19], k2[3], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[20], k2[4], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[21], k2[5], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[22], k2[6], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[23], k2[7], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[24], k2[8], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[25], k2[9], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[26], k2[10], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[27], k2[11], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[28], k2[12], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[29], k2[13], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[30], k2[14], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[31], k2[15], acc0_0);
                float dot_0 = acc0_0.x + acc0_0.y;
                float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 1);
                dot_0 = dot_0 + _shfl_xor_sync_44;
                float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 2);
                dot_0 = dot_0 + _shfl_xor_sync_45;
#endif
                float vn_1 = (v_regs[1] - dot_0) * beta_0;
#if __CUDA_ARCH__ == 900
                float _fma_150;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_150) : "f"(k_regs[0]), "f"(vn_1), "f"(S[32]));
                S[32] = _fma_150;
                float _fma_151;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_151) : "f"(k_regs[1]), "f"(vn_1), "f"(S[33]));
                S[33] = _fma_151;
                float _fma_152;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_152) : "f"(k_regs[2]), "f"(vn_1), "f"(S[34]));
                S[34] = _fma_152;
                float _fma_153;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_153) : "f"(k_regs[3]), "f"(vn_1), "f"(S[35]));
                S[35] = _fma_153;
                float _fma_154;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_154) : "f"(k_regs[4]), "f"(vn_1), "f"(S[36]));
                S[36] = _fma_154;
                float _fma_155;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(k_regs[5]), "f"(vn_1), "f"(S[37]));
                S[37] = _fma_155;
                float _fma_156;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(k_regs[6]), "f"(vn_1), "f"(S[38]));
                S[38] = _fma_156;
                float _fma_157;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(k_regs[7]), "f"(vn_1), "f"(S[39]));
                S[39] = _fma_157;
                float _fma_158;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_158) : "f"(k_regs[8]), "f"(vn_1), "f"(S[40]));
                S[40] = _fma_158;
                float _fma_159;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_159) : "f"(k_regs[9]), "f"(vn_1), "f"(S[41]));
                S[41] = _fma_159;
                float _fma_160;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_160) : "f"(k_regs[10]), "f"(vn_1), "f"(S[42]));
                S[42] = _fma_160;
                float _fma_161;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_161) : "f"(k_regs[11]), "f"(vn_1), "f"(S[43]));
                S[43] = _fma_161;
                float _fma_162;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_162) : "f"(k_regs[12]), "f"(vn_1), "f"(S[44]));
                S[44] = _fma_162;
                float _fma_163;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_163) : "f"(k_regs[13]), "f"(vn_1), "f"(S[45]));
                S[45] = _fma_163;
                float _fma_164;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_164) : "f"(k_regs[14]), "f"(vn_1), "f"(S[46]));
                S[46] = _fma_164;
                float _fma_165;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_165) : "f"(k_regs[15]), "f"(vn_1), "f"(S[47]));
                S[47] = _fma_165;
                float _fma_166;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_166) : "f"(k_regs[16]), "f"(vn_1), "f"(S[48]));
                S[48] = _fma_166;
                float _fma_167;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_167) : "f"(k_regs[17]), "f"(vn_1), "f"(S[49]));
                S[49] = _fma_167;
                float _fma_168;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_168) : "f"(k_regs[18]), "f"(vn_1), "f"(S[50]));
                S[50] = _fma_168;
                float _fma_169;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_169) : "f"(k_regs[19]), "f"(vn_1), "f"(S[51]));
                S[51] = _fma_169;
                float _fma_170;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_170) : "f"(k_regs[20]), "f"(vn_1), "f"(S[52]));
                S[52] = _fma_170;
                float _fma_171;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_171) : "f"(k_regs[21]), "f"(vn_1), "f"(S[53]));
                S[53] = _fma_171;
                float _fma_172;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_172) : "f"(k_regs[22]), "f"(vn_1), "f"(S[54]));
                S[54] = _fma_172;
                float _fma_173;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_173) : "f"(k_regs[23]), "f"(vn_1), "f"(S[55]));
                S[55] = _fma_173;
                float _fma_174;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_174) : "f"(k_regs[24]), "f"(vn_1), "f"(S[56]));
                S[56] = _fma_174;
                float _fma_175;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_175) : "f"(k_regs[25]), "f"(vn_1), "f"(S[57]));
                S[57] = _fma_175;
                float _fma_176;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_176) : "f"(k_regs[26]), "f"(vn_1), "f"(S[58]));
                S[58] = _fma_176;
                float _fma_177;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_177) : "f"(k_regs[27]), "f"(vn_1), "f"(S[59]));
                S[59] = _fma_177;
                float _fma_178;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_178) : "f"(k_regs[28]), "f"(vn_1), "f"(S[60]));
                S[60] = _fma_178;
                float _fma_179;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_179) : "f"(k_regs[29]), "f"(vn_1), "f"(S[61]));
                S[61] = _fma_179;
                float _fma_180;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_180) : "f"(k_regs[30]), "f"(vn_1), "f"(S[62]));
                S[62] = _fma_180;
                float _fma_181;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_181) : "f"(k_regs[31]), "f"(vn_1), "f"(S[63]));
                S[63] = _fma_181;
                float acc0_x_1 = 0.0f;
                float acc0_y_1 = 0.0f;
                float acc1_x_1 = 0.0f;
                float acc1_y_1 = 0.0f;
                float _fma_182;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_182) : "f"(S[64]), "f"(k_regs[0]), "f"(acc0_x_1));
                acc0_x_1 = _fma_182;
                float _fma_183;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_183) : "f"(S[65]), "f"(k_regs[1]), "f"(acc0_y_1));
                acc0_y_1 = _fma_183;
                float _fma_184;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_184) : "f"(S[66]), "f"(k_regs[2]), "f"(acc1_x_1));
                acc1_x_1 = _fma_184;
                float _fma_185;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_185) : "f"(S[67]), "f"(k_regs[3]), "f"(acc1_y_1));
                acc1_y_1 = _fma_185;
                float _fma_186;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_186) : "f"(S[68]), "f"(k_regs[4]), "f"(acc0_x_1));
                acc0_x_1 = _fma_186;
                float _fma_187;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_187) : "f"(S[69]), "f"(k_regs[5]), "f"(acc0_y_1));
                acc0_y_1 = _fma_187;
                float _fma_188;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_188) : "f"(S[70]), "f"(k_regs[6]), "f"(acc1_x_1));
                acc1_x_1 = _fma_188;
                float _fma_189;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_189) : "f"(S[71]), "f"(k_regs[7]), "f"(acc1_y_1));
                acc1_y_1 = _fma_189;
                float _fma_190;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_190) : "f"(S[72]), "f"(k_regs[8]), "f"(acc0_x_1));
                acc0_x_1 = _fma_190;
                float _fma_191;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_191) : "f"(S[73]), "f"(k_regs[9]), "f"(acc0_y_1));
                acc0_y_1 = _fma_191;
                float _fma_192;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_192) : "f"(S[74]), "f"(k_regs[10]), "f"(acc1_x_1));
                acc1_x_1 = _fma_192;
                float _fma_193;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_193) : "f"(S[75]), "f"(k_regs[11]), "f"(acc1_y_1));
                acc1_y_1 = _fma_193;
                float _fma_194;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_194) : "f"(S[76]), "f"(k_regs[12]), "f"(acc0_x_1));
                acc0_x_1 = _fma_194;
                float _fma_195;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_195) : "f"(S[77]), "f"(k_regs[13]), "f"(acc0_y_1));
                acc0_y_1 = _fma_195;
                float _fma_196;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_196) : "f"(S[78]), "f"(k_regs[14]), "f"(acc1_x_1));
                acc1_x_1 = _fma_196;
                float _fma_197;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_197) : "f"(S[79]), "f"(k_regs[15]), "f"(acc1_y_1));
                acc1_y_1 = _fma_197;
                float _fma_198;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_198) : "f"(S[80]), "f"(k_regs[16]), "f"(acc0_x_1));
                acc0_x_1 = _fma_198;
                float _fma_199;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_199) : "f"(S[81]), "f"(k_regs[17]), "f"(acc0_y_1));
                acc0_y_1 = _fma_199;
                float _fma_200;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_200) : "f"(S[82]), "f"(k_regs[18]), "f"(acc1_x_1));
                acc1_x_1 = _fma_200;
                float _fma_201;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_201) : "f"(S[83]), "f"(k_regs[19]), "f"(acc1_y_1));
                acc1_y_1 = _fma_201;
                float _fma_202;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_202) : "f"(S[84]), "f"(k_regs[20]), "f"(acc0_x_1));
                acc0_x_1 = _fma_202;
                float _fma_203;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_203) : "f"(S[85]), "f"(k_regs[21]), "f"(acc0_y_1));
                acc0_y_1 = _fma_203;
                float _fma_204;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_204) : "f"(S[86]), "f"(k_regs[22]), "f"(acc1_x_1));
                acc1_x_1 = _fma_204;
                float _fma_205;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_205) : "f"(S[87]), "f"(k_regs[23]), "f"(acc1_y_1));
                acc1_y_1 = _fma_205;
                float _fma_206;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_206) : "f"(S[88]), "f"(k_regs[24]), "f"(acc0_x_1));
                acc0_x_1 = _fma_206;
                float _fma_207;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_207) : "f"(S[89]), "f"(k_regs[25]), "f"(acc0_y_1));
                acc0_y_1 = _fma_207;
                float _fma_208;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_208) : "f"(S[90]), "f"(k_regs[26]), "f"(acc1_x_1));
                acc1_x_1 = _fma_208;
                float _fma_209;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_209) : "f"(S[91]), "f"(k_regs[27]), "f"(acc1_y_1));
                acc1_y_1 = _fma_209;
                float _fma_210;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_210) : "f"(S[92]), "f"(k_regs[28]), "f"(acc0_x_1));
                acc0_x_1 = _fma_210;
                float _fma_211;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_211) : "f"(S[93]), "f"(k_regs[29]), "f"(acc0_y_1));
                acc0_y_1 = _fma_211;
                float _fma_212;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_212) : "f"(S[94]), "f"(k_regs[30]), "f"(acc1_x_1));
                acc1_x_1 = _fma_212;
                float _fma_213;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_213) : "f"(S[95]), "f"(k_regs[31]), "f"(acc1_y_1));
                acc1_y_1 = _fma_213;
                float _add_214;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_214) : "f"(acc0_x_1), "f"(acc1_x_1));
                float _add_215;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_215) : "f"(acc0_y_1), "f"(acc1_y_1));
                float dot_1 = _add_214 + _add_215;
                float _shfl_xor_sync_216 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_216;
                float _shfl_xor_sync_217 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_217;
#else
                float2 vn2_1 = make_float2(vn_1, vn_1);
                S2[16] = fma_f32x2_rn_ftz(k2[0], vn2_1, S2[16]);
                S2[17] = fma_f32x2_rn_ftz(k2[1], vn2_1, S2[17]);
                S2[18] = fma_f32x2_rn_ftz(k2[2], vn2_1, S2[18]);
                S2[19] = fma_f32x2_rn_ftz(k2[3], vn2_1, S2[19]);
                S2[20] = fma_f32x2_rn_ftz(k2[4], vn2_1, S2[20]);
                S2[21] = fma_f32x2_rn_ftz(k2[5], vn2_1, S2[21]);
                S2[22] = fma_f32x2_rn_ftz(k2[6], vn2_1, S2[22]);
                S2[23] = fma_f32x2_rn_ftz(k2[7], vn2_1, S2[23]);
                S2[24] = fma_f32x2_rn_ftz(k2[8], vn2_1, S2[24]);
                S2[25] = fma_f32x2_rn_ftz(k2[9], vn2_1, S2[25]);
                S2[26] = fma_f32x2_rn_ftz(k2[10], vn2_1, S2[26]);
                S2[27] = fma_f32x2_rn_ftz(k2[11], vn2_1, S2[27]);
                S2[28] = fma_f32x2_rn_ftz(k2[12], vn2_1, S2[28]);
                S2[29] = fma_f32x2_rn_ftz(k2[13], vn2_1, S2[29]);
                S2[30] = fma_f32x2_rn_ftz(k2[14], vn2_1, S2[30]);
                S2[31] = fma_f32x2_rn_ftz(k2[15], vn2_1, S2[31]);
                float2 acc0_1 = make_float2(0.0f, 0.0f);
                acc0_1 = fma_f32x2_rn_ftz(S2[32], k2[0], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[33], k2[1], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[34], k2[2], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[35], k2[3], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[36], k2[4], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[37], k2[5], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[38], k2[6], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[39], k2[7], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[40], k2[8], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[41], k2[9], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[42], k2[10], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[43], k2[11], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[44], k2[12], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[45], k2[13], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[46], k2[14], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[47], k2[15], acc0_1);
                float dot_1 = acc0_1.x + acc0_1.y;
                float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_46;
                float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_47;
#endif
                float vn_2 = (v_regs[2] - dot_1) * beta_0;
#if __CUDA_ARCH__ == 900
                float _fma_218;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_218) : "f"(k_regs[0]), "f"(vn_2), "f"(S[64]));
                S[64] = _fma_218;
                float _fma_219;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_219) : "f"(k_regs[1]), "f"(vn_2), "f"(S[65]));
                S[65] = _fma_219;
                float _fma_220;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_220) : "f"(k_regs[2]), "f"(vn_2), "f"(S[66]));
                S[66] = _fma_220;
                float _fma_221;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_221) : "f"(k_regs[3]), "f"(vn_2), "f"(S[67]));
                S[67] = _fma_221;
                float _fma_222;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_222) : "f"(k_regs[4]), "f"(vn_2), "f"(S[68]));
                S[68] = _fma_222;
                float _fma_223;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_223) : "f"(k_regs[5]), "f"(vn_2), "f"(S[69]));
                S[69] = _fma_223;
                float _fma_224;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_224) : "f"(k_regs[6]), "f"(vn_2), "f"(S[70]));
                S[70] = _fma_224;
                float _fma_225;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_225) : "f"(k_regs[7]), "f"(vn_2), "f"(S[71]));
                S[71] = _fma_225;
                float _fma_226;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_226) : "f"(k_regs[8]), "f"(vn_2), "f"(S[72]));
                S[72] = _fma_226;
                float _fma_227;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_227) : "f"(k_regs[9]), "f"(vn_2), "f"(S[73]));
                S[73] = _fma_227;
                float _fma_228;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_228) : "f"(k_regs[10]), "f"(vn_2), "f"(S[74]));
                S[74] = _fma_228;
                float _fma_229;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_229) : "f"(k_regs[11]), "f"(vn_2), "f"(S[75]));
                S[75] = _fma_229;
                float _fma_230;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_230) : "f"(k_regs[12]), "f"(vn_2), "f"(S[76]));
                S[76] = _fma_230;
                float _fma_231;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_231) : "f"(k_regs[13]), "f"(vn_2), "f"(S[77]));
                S[77] = _fma_231;
                float _fma_232;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_232) : "f"(k_regs[14]), "f"(vn_2), "f"(S[78]));
                S[78] = _fma_232;
                float _fma_233;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_233) : "f"(k_regs[15]), "f"(vn_2), "f"(S[79]));
                S[79] = _fma_233;
                float _fma_234;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_234) : "f"(k_regs[16]), "f"(vn_2), "f"(S[80]));
                S[80] = _fma_234;
                float _fma_235;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_235) : "f"(k_regs[17]), "f"(vn_2), "f"(S[81]));
                S[81] = _fma_235;
                float _fma_236;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_236) : "f"(k_regs[18]), "f"(vn_2), "f"(S[82]));
                S[82] = _fma_236;
                float _fma_237;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_237) : "f"(k_regs[19]), "f"(vn_2), "f"(S[83]));
                S[83] = _fma_237;
                float _fma_238;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_238) : "f"(k_regs[20]), "f"(vn_2), "f"(S[84]));
                S[84] = _fma_238;
                float _fma_239;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_239) : "f"(k_regs[21]), "f"(vn_2), "f"(S[85]));
                S[85] = _fma_239;
                float _fma_240;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_240) : "f"(k_regs[22]), "f"(vn_2), "f"(S[86]));
                S[86] = _fma_240;
                float _fma_241;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_241) : "f"(k_regs[23]), "f"(vn_2), "f"(S[87]));
                S[87] = _fma_241;
                float _fma_242;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_242) : "f"(k_regs[24]), "f"(vn_2), "f"(S[88]));
                S[88] = _fma_242;
                float _fma_243;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_243) : "f"(k_regs[25]), "f"(vn_2), "f"(S[89]));
                S[89] = _fma_243;
                float _fma_244;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_244) : "f"(k_regs[26]), "f"(vn_2), "f"(S[90]));
                S[90] = _fma_244;
                float _fma_245;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_245) : "f"(k_regs[27]), "f"(vn_2), "f"(S[91]));
                S[91] = _fma_245;
                float _fma_246;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_246) : "f"(k_regs[28]), "f"(vn_2), "f"(S[92]));
                S[92] = _fma_246;
                float _fma_247;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_247) : "f"(k_regs[29]), "f"(vn_2), "f"(S[93]));
                S[93] = _fma_247;
                float _fma_248;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_248) : "f"(k_regs[30]), "f"(vn_2), "f"(S[94]));
                S[94] = _fma_248;
                float _fma_249;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_249) : "f"(k_regs[31]), "f"(vn_2), "f"(S[95]));
                S[95] = _fma_249;
                float acc0_x_2 = 0.0f;
                float acc0_y_2 = 0.0f;
                float acc1_x_2 = 0.0f;
                float acc1_y_2 = 0.0f;
                float _fma_250;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_250) : "f"(S[96]), "f"(k_regs[0]), "f"(acc0_x_2));
                acc0_x_2 = _fma_250;
                float _fma_251;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_251) : "f"(S[97]), "f"(k_regs[1]), "f"(acc0_y_2));
                acc0_y_2 = _fma_251;
                float _fma_252;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_252) : "f"(S[98]), "f"(k_regs[2]), "f"(acc1_x_2));
                acc1_x_2 = _fma_252;
                float _fma_253;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_253) : "f"(S[99]), "f"(k_regs[3]), "f"(acc1_y_2));
                acc1_y_2 = _fma_253;
                float _fma_254;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_254) : "f"(S[100]), "f"(k_regs[4]), "f"(acc0_x_2));
                acc0_x_2 = _fma_254;
                float _fma_255;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_255) : "f"(S[101]), "f"(k_regs[5]), "f"(acc0_y_2));
                acc0_y_2 = _fma_255;
                float _fma_256;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_256) : "f"(S[102]), "f"(k_regs[6]), "f"(acc1_x_2));
                acc1_x_2 = _fma_256;
                float _fma_257;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_257) : "f"(S[103]), "f"(k_regs[7]), "f"(acc1_y_2));
                acc1_y_2 = _fma_257;
                float _fma_258;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_258) : "f"(S[104]), "f"(k_regs[8]), "f"(acc0_x_2));
                acc0_x_2 = _fma_258;
                float _fma_259;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_259) : "f"(S[105]), "f"(k_regs[9]), "f"(acc0_y_2));
                acc0_y_2 = _fma_259;
                float _fma_260;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_260) : "f"(S[106]), "f"(k_regs[10]), "f"(acc1_x_2));
                acc1_x_2 = _fma_260;
                float _fma_261;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_261) : "f"(S[107]), "f"(k_regs[11]), "f"(acc1_y_2));
                acc1_y_2 = _fma_261;
                float _fma_262;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_262) : "f"(S[108]), "f"(k_regs[12]), "f"(acc0_x_2));
                acc0_x_2 = _fma_262;
                float _fma_263;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_263) : "f"(S[109]), "f"(k_regs[13]), "f"(acc0_y_2));
                acc0_y_2 = _fma_263;
                float _fma_264;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_264) : "f"(S[110]), "f"(k_regs[14]), "f"(acc1_x_2));
                acc1_x_2 = _fma_264;
                float _fma_265;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_265) : "f"(S[111]), "f"(k_regs[15]), "f"(acc1_y_2));
                acc1_y_2 = _fma_265;
                float _fma_266;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_266) : "f"(S[112]), "f"(k_regs[16]), "f"(acc0_x_2));
                acc0_x_2 = _fma_266;
                float _fma_267;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_267) : "f"(S[113]), "f"(k_regs[17]), "f"(acc0_y_2));
                acc0_y_2 = _fma_267;
                float _fma_268;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_268) : "f"(S[114]), "f"(k_regs[18]), "f"(acc1_x_2));
                acc1_x_2 = _fma_268;
                float _fma_269;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_269) : "f"(S[115]), "f"(k_regs[19]), "f"(acc1_y_2));
                acc1_y_2 = _fma_269;
                float _fma_270;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_270) : "f"(S[116]), "f"(k_regs[20]), "f"(acc0_x_2));
                acc0_x_2 = _fma_270;
                float _fma_271;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_271) : "f"(S[117]), "f"(k_regs[21]), "f"(acc0_y_2));
                acc0_y_2 = _fma_271;
                float _fma_272;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_272) : "f"(S[118]), "f"(k_regs[22]), "f"(acc1_x_2));
                acc1_x_2 = _fma_272;
                float _fma_273;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_273) : "f"(S[119]), "f"(k_regs[23]), "f"(acc1_y_2));
                acc1_y_2 = _fma_273;
                float _fma_274;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_274) : "f"(S[120]), "f"(k_regs[24]), "f"(acc0_x_2));
                acc0_x_2 = _fma_274;
                float _fma_275;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_275) : "f"(S[121]), "f"(k_regs[25]), "f"(acc0_y_2));
                acc0_y_2 = _fma_275;
                float _fma_276;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_276) : "f"(S[122]), "f"(k_regs[26]), "f"(acc1_x_2));
                acc1_x_2 = _fma_276;
                float _fma_277;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_277) : "f"(S[123]), "f"(k_regs[27]), "f"(acc1_y_2));
                acc1_y_2 = _fma_277;
                float _fma_278;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_278) : "f"(S[124]), "f"(k_regs[28]), "f"(acc0_x_2));
                acc0_x_2 = _fma_278;
                float _fma_279;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_279) : "f"(S[125]), "f"(k_regs[29]), "f"(acc0_y_2));
                acc0_y_2 = _fma_279;
                float _fma_280;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_280) : "f"(S[126]), "f"(k_regs[30]), "f"(acc1_x_2));
                acc1_x_2 = _fma_280;
                float _fma_281;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_281) : "f"(S[127]), "f"(k_regs[31]), "f"(acc1_y_2));
                acc1_y_2 = _fma_281;
                float _add_282;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_282) : "f"(acc0_x_2), "f"(acc1_x_2));
                float _add_283;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_283) : "f"(acc0_y_2), "f"(acc1_y_2));
                float dot_2 = _add_282 + _add_283;
                float _shfl_xor_sync_284 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 1);
                dot_2 = dot_2 + _shfl_xor_sync_284;
                float _shfl_xor_sync_285 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 2);
                dot_2 = dot_2 + _shfl_xor_sync_285;
#else
                float2 vn2_2 = make_float2(vn_2, vn_2);
                S2[32] = fma_f32x2_rn_ftz(k2[0], vn2_2, S2[32]);
                S2[33] = fma_f32x2_rn_ftz(k2[1], vn2_2, S2[33]);
                S2[34] = fma_f32x2_rn_ftz(k2[2], vn2_2, S2[34]);
                S2[35] = fma_f32x2_rn_ftz(k2[3], vn2_2, S2[35]);
                S2[36] = fma_f32x2_rn_ftz(k2[4], vn2_2, S2[36]);
                S2[37] = fma_f32x2_rn_ftz(k2[5], vn2_2, S2[37]);
                S2[38] = fma_f32x2_rn_ftz(k2[6], vn2_2, S2[38]);
                S2[39] = fma_f32x2_rn_ftz(k2[7], vn2_2, S2[39]);
                S2[40] = fma_f32x2_rn_ftz(k2[8], vn2_2, S2[40]);
                S2[41] = fma_f32x2_rn_ftz(k2[9], vn2_2, S2[41]);
                S2[42] = fma_f32x2_rn_ftz(k2[10], vn2_2, S2[42]);
                S2[43] = fma_f32x2_rn_ftz(k2[11], vn2_2, S2[43]);
                S2[44] = fma_f32x2_rn_ftz(k2[12], vn2_2, S2[44]);
                S2[45] = fma_f32x2_rn_ftz(k2[13], vn2_2, S2[45]);
                S2[46] = fma_f32x2_rn_ftz(k2[14], vn2_2, S2[46]);
                S2[47] = fma_f32x2_rn_ftz(k2[15], vn2_2, S2[47]);
                float2 acc0_2 = make_float2(0.0f, 0.0f);
                acc0_2 = fma_f32x2_rn_ftz(S2[48], k2[0], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[49], k2[1], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[50], k2[2], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[51], k2[3], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[52], k2[4], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[53], k2[5], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[54], k2[6], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[55], k2[7], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[56], k2[8], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[57], k2[9], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[58], k2[10], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[59], k2[11], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[60], k2[12], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[61], k2[13], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[62], k2[14], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[63], k2[15], acc0_2);
                float dot_2 = acc0_2.x + acc0_2.y;
                float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 1);
                dot_2 = dot_2 + _shfl_xor_sync_48;
                float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 2);
                dot_2 = dot_2 + _shfl_xor_sync_49;
#endif
                float vn_3 = (v_regs[3] - dot_2) * beta_0;
#if __CUDA_ARCH__ == 900
                float _fma_286;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_286) : "f"(k_regs[0]), "f"(vn_3), "f"(S[96]));
                S[96] = _fma_286;
                float _fma_287;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_287) : "f"(k_regs[1]), "f"(vn_3), "f"(S[97]));
                S[97] = _fma_287;
                float _fma_288;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_288) : "f"(k_regs[2]), "f"(vn_3), "f"(S[98]));
                S[98] = _fma_288;
                float _fma_289;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_289) : "f"(k_regs[3]), "f"(vn_3), "f"(S[99]));
                S[99] = _fma_289;
                float _fma_290;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_290) : "f"(k_regs[4]), "f"(vn_3), "f"(S[100]));
                S[100] = _fma_290;
                float _fma_291;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_291) : "f"(k_regs[5]), "f"(vn_3), "f"(S[101]));
                S[101] = _fma_291;
                float _fma_292;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_292) : "f"(k_regs[6]), "f"(vn_3), "f"(S[102]));
                S[102] = _fma_292;
                float _fma_293;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_293) : "f"(k_regs[7]), "f"(vn_3), "f"(S[103]));
                S[103] = _fma_293;
                float _fma_294;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_294) : "f"(k_regs[8]), "f"(vn_3), "f"(S[104]));
                S[104] = _fma_294;
                float _fma_295;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_295) : "f"(k_regs[9]), "f"(vn_3), "f"(S[105]));
                S[105] = _fma_295;
                float _fma_296;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_296) : "f"(k_regs[10]), "f"(vn_3), "f"(S[106]));
                S[106] = _fma_296;
                float _fma_297;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_297) : "f"(k_regs[11]), "f"(vn_3), "f"(S[107]));
                S[107] = _fma_297;
                float _fma_298;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_298) : "f"(k_regs[12]), "f"(vn_3), "f"(S[108]));
                S[108] = _fma_298;
                float _fma_299;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_299) : "f"(k_regs[13]), "f"(vn_3), "f"(S[109]));
                S[109] = _fma_299;
                float _fma_300;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_300) : "f"(k_regs[14]), "f"(vn_3), "f"(S[110]));
                S[110] = _fma_300;
                float _fma_301;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_301) : "f"(k_regs[15]), "f"(vn_3), "f"(S[111]));
                S[111] = _fma_301;
                float _fma_302;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_302) : "f"(k_regs[16]), "f"(vn_3), "f"(S[112]));
                S[112] = _fma_302;
                float _fma_303;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_303) : "f"(k_regs[17]), "f"(vn_3), "f"(S[113]));
                S[113] = _fma_303;
                float _fma_304;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_304) : "f"(k_regs[18]), "f"(vn_3), "f"(S[114]));
                S[114] = _fma_304;
                float _fma_305;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_305) : "f"(k_regs[19]), "f"(vn_3), "f"(S[115]));
                S[115] = _fma_305;
                float _fma_306;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_306) : "f"(k_regs[20]), "f"(vn_3), "f"(S[116]));
                S[116] = _fma_306;
                float _fma_307;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_307) : "f"(k_regs[21]), "f"(vn_3), "f"(S[117]));
                S[117] = _fma_307;
                float _fma_308;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_308) : "f"(k_regs[22]), "f"(vn_3), "f"(S[118]));
                S[118] = _fma_308;
                float _fma_309;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_309) : "f"(k_regs[23]), "f"(vn_3), "f"(S[119]));
                S[119] = _fma_309;
                float _fma_310;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_310) : "f"(k_regs[24]), "f"(vn_3), "f"(S[120]));
                S[120] = _fma_310;
                float _fma_311;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_311) : "f"(k_regs[25]), "f"(vn_3), "f"(S[121]));
                S[121] = _fma_311;
                float _fma_312;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_312) : "f"(k_regs[26]), "f"(vn_3), "f"(S[122]));
                S[122] = _fma_312;
                float _fma_313;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_313) : "f"(k_regs[27]), "f"(vn_3), "f"(S[123]));
                S[123] = _fma_313;
                float _fma_314;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_314) : "f"(k_regs[28]), "f"(vn_3), "f"(S[124]));
                S[124] = _fma_314;
                float _fma_315;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_315) : "f"(k_regs[29]), "f"(vn_3), "f"(S[125]));
                S[125] = _fma_315;
                float _fma_316;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_316) : "f"(k_regs[30]), "f"(vn_3), "f"(S[126]));
                S[126] = _fma_316;
                float _fma_317;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_317) : "f"(k_regs[31]), "f"(vn_3), "f"(S[127]));
                S[127] = _fma_317;
#else
                float2 vn2_3 = make_float2(vn_3, vn_3);
                S2[48] = fma_f32x2_rn_ftz(k2[0], vn2_3, S2[48]);
                S2[49] = fma_f32x2_rn_ftz(k2[1], vn2_3, S2[49]);
                S2[50] = fma_f32x2_rn_ftz(k2[2], vn2_3, S2[50]);
                S2[51] = fma_f32x2_rn_ftz(k2[3], vn2_3, S2[51]);
                S2[52] = fma_f32x2_rn_ftz(k2[4], vn2_3, S2[52]);
                S2[53] = fma_f32x2_rn_ftz(k2[5], vn2_3, S2[53]);
                S2[54] = fma_f32x2_rn_ftz(k2[6], vn2_3, S2[54]);
                S2[55] = fma_f32x2_rn_ftz(k2[7], vn2_3, S2[55]);
                S2[56] = fma_f32x2_rn_ftz(k2[8], vn2_3, S2[56]);
                S2[57] = fma_f32x2_rn_ftz(k2[9], vn2_3, S2[57]);
                S2[58] = fma_f32x2_rn_ftz(k2[10], vn2_3, S2[58]);
                S2[59] = fma_f32x2_rn_ftz(k2[11], vn2_3, S2[59]);
                S2[60] = fma_f32x2_rn_ftz(k2[12], vn2_3, S2[60]);
                S2[61] = fma_f32x2_rn_ftz(k2[13], vn2_3, S2[61]);
                S2[62] = fma_f32x2_rn_ftz(k2[14], vn2_3, S2[62]);
                S2[63] = fma_f32x2_rn_ftz(k2[15], vn2_3, S2[63]);
#endif
                unsigned int qbase = sQ_addr + (unsigned int)(t_1 * 512) + lane_byte;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(qbase));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(qbase + 64));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(qbase + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(qbase + 192));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[16])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 3]))
                    : "r"(qbase + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[20])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 3]))
                    : "r"(qbase + 320));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[24])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 3]))
                    : "r"(qbase + 384));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[28])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 3]))
                    : "r"(qbase + 448));
#if __CUDA_ARCH__ == 900
                float acc0_x_3 = 0.0f;
                float acc0_y_3 = 0.0f;
                float acc1_x_3 = 0.0f;
                float acc1_y_3 = 0.0f;
                float _fma_318;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_318) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_3));
                acc0_x_3 = _fma_318;
                float _fma_319;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_319) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_3));
                acc0_y_3 = _fma_319;
                float _fma_320;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_320) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_3));
                acc1_x_3 = _fma_320;
                float _fma_321;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_321) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_3));
                acc1_y_3 = _fma_321;
                float _fma_322;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_322) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_3));
                acc0_x_3 = _fma_322;
                float _fma_323;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_323) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_3));
                acc0_y_3 = _fma_323;
                float _fma_324;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_324) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_3));
                acc1_x_3 = _fma_324;
                float _fma_325;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_325) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_3));
                acc1_y_3 = _fma_325;
                float _fma_326;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_326) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_3));
                acc0_x_3 = _fma_326;
                float _fma_327;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_327) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_3));
                acc0_y_3 = _fma_327;
                float _fma_328;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_328) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_3));
                acc1_x_3 = _fma_328;
                float _fma_329;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_329) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_3));
                acc1_y_3 = _fma_329;
                float _fma_330;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_330) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_3));
                acc0_x_3 = _fma_330;
                float _fma_331;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_331) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_3));
                acc0_y_3 = _fma_331;
                float _fma_332;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_332) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_3));
                acc1_x_3 = _fma_332;
                float _fma_333;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_333) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_3));
                acc1_y_3 = _fma_333;
                float _fma_334;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_334) : "f"(S[16]), "f"(k_regs[16]), "f"(acc0_x_3));
                acc0_x_3 = _fma_334;
                float _fma_335;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_335) : "f"(S[17]), "f"(k_regs[17]), "f"(acc0_y_3));
                acc0_y_3 = _fma_335;
                float _fma_336;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_336) : "f"(S[18]), "f"(k_regs[18]), "f"(acc1_x_3));
                acc1_x_3 = _fma_336;
                float _fma_337;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_337) : "f"(S[19]), "f"(k_regs[19]), "f"(acc1_y_3));
                acc1_y_3 = _fma_337;
                float _fma_338;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_338) : "f"(S[20]), "f"(k_regs[20]), "f"(acc0_x_3));
                acc0_x_3 = _fma_338;
                float _fma_339;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_339) : "f"(S[21]), "f"(k_regs[21]), "f"(acc0_y_3));
                acc0_y_3 = _fma_339;
                float _fma_340;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_340) : "f"(S[22]), "f"(k_regs[22]), "f"(acc1_x_3));
                acc1_x_3 = _fma_340;
                float _fma_341;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_341) : "f"(S[23]), "f"(k_regs[23]), "f"(acc1_y_3));
                acc1_y_3 = _fma_341;
                float _fma_342;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_342) : "f"(S[24]), "f"(k_regs[24]), "f"(acc0_x_3));
                acc0_x_3 = _fma_342;
                float _fma_343;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_343) : "f"(S[25]), "f"(k_regs[25]), "f"(acc0_y_3));
                acc0_y_3 = _fma_343;
                float _fma_344;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_344) : "f"(S[26]), "f"(k_regs[26]), "f"(acc1_x_3));
                acc1_x_3 = _fma_344;
                float _fma_345;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_345) : "f"(S[27]), "f"(k_regs[27]), "f"(acc1_y_3));
                acc1_y_3 = _fma_345;
                float _fma_346;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_346) : "f"(S[28]), "f"(k_regs[28]), "f"(acc0_x_3));
                acc0_x_3 = _fma_346;
                float _fma_347;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_347) : "f"(S[29]), "f"(k_regs[29]), "f"(acc0_y_3));
                acc0_y_3 = _fma_347;
                float _fma_348;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_348) : "f"(S[30]), "f"(k_regs[30]), "f"(acc1_x_3));
                acc1_x_3 = _fma_348;
                float _fma_349;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_349) : "f"(S[31]), "f"(k_regs[31]), "f"(acc1_y_3));
                acc1_y_3 = _fma_349;
                float _add_350;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_350) : "f"(acc0_x_3), "f"(acc1_x_3));
                float _add_351;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_351) : "f"(acc0_y_3), "f"(acc1_y_3));
                float dot_3 = _add_350 + _add_351;
                float _shfl_xor_sync_352 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 1);
                dot_3 = dot_3 + _shfl_xor_sync_352;
                float _shfl_xor_sync_353 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 2);
                dot_3 = dot_3 + _shfl_xor_sync_353;
#else
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                k2[8] = make_float2(k_regs[16], k_regs[17]);
                k2[9] = make_float2(k_regs[18], k_regs[19]);
                k2[10] = make_float2(k_regs[20], k_regs[21]);
                k2[11] = make_float2(k_regs[22], k_regs[23]);
                k2[12] = make_float2(k_regs[24], k_regs[25]);
                k2[13] = make_float2(k_regs[26], k_regs[27]);
                k2[14] = make_float2(k_regs[28], k_regs[29]);
                k2[15] = make_float2(k_regs[30], k_regs[31]);
                float2 acc0_3 = make_float2(0.0f, 0.0f);
                acc0_3 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[8], k2[8], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[9], k2[9], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[10], k2[10], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[11], k2[11], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[12], k2[12], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[13], k2[13], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[14], k2[14], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[15], k2[15], acc0_3);
                float dot_3 = acc0_3.x + acc0_3.y;
                float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 1);
                dot_3 = dot_3 + _shfl_xor_sync_50;
                float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 2);
                dot_3 = dot_3 + _shfl_xor_sync_51;
#endif
                o4[0] = dot_3;
#if __CUDA_ARCH__ == 900
                float acc0_x_4 = 0.0f;
                float acc0_y_4 = 0.0f;
                float acc1_x_4 = 0.0f;
                float acc1_y_4 = 0.0f;
                float _fma_354;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_354) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_4));
                acc0_x_4 = _fma_354;
                float _fma_355;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_355) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_4));
                acc0_y_4 = _fma_355;
                float _fma_356;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_356) : "f"(S[34]), "f"(k_regs[2]), "f"(acc1_x_4));
                acc1_x_4 = _fma_356;
                float _fma_357;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_357) : "f"(S[35]), "f"(k_regs[3]), "f"(acc1_y_4));
                acc1_y_4 = _fma_357;
                float _fma_358;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_358) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_4));
                acc0_x_4 = _fma_358;
                float _fma_359;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_359) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_4));
                acc0_y_4 = _fma_359;
                float _fma_360;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_360) : "f"(S[38]), "f"(k_regs[6]), "f"(acc1_x_4));
                acc1_x_4 = _fma_360;
                float _fma_361;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_361) : "f"(S[39]), "f"(k_regs[7]), "f"(acc1_y_4));
                acc1_y_4 = _fma_361;
                float _fma_362;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_362) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_4));
                acc0_x_4 = _fma_362;
                float _fma_363;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_363) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_4));
                acc0_y_4 = _fma_363;
                float _fma_364;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_364) : "f"(S[42]), "f"(k_regs[10]), "f"(acc1_x_4));
                acc1_x_4 = _fma_364;
                float _fma_365;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_365) : "f"(S[43]), "f"(k_regs[11]), "f"(acc1_y_4));
                acc1_y_4 = _fma_365;
                float _fma_366;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_366) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_4));
                acc0_x_4 = _fma_366;
                float _fma_367;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_367) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_4));
                acc0_y_4 = _fma_367;
                float _fma_368;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_368) : "f"(S[46]), "f"(k_regs[14]), "f"(acc1_x_4));
                acc1_x_4 = _fma_368;
                float _fma_369;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_369) : "f"(S[47]), "f"(k_regs[15]), "f"(acc1_y_4));
                acc1_y_4 = _fma_369;
                float _fma_370;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_370) : "f"(S[48]), "f"(k_regs[16]), "f"(acc0_x_4));
                acc0_x_4 = _fma_370;
                float _fma_371;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_371) : "f"(S[49]), "f"(k_regs[17]), "f"(acc0_y_4));
                acc0_y_4 = _fma_371;
                float _fma_372;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_372) : "f"(S[50]), "f"(k_regs[18]), "f"(acc1_x_4));
                acc1_x_4 = _fma_372;
                float _fma_373;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_373) : "f"(S[51]), "f"(k_regs[19]), "f"(acc1_y_4));
                acc1_y_4 = _fma_373;
                float _fma_374;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_374) : "f"(S[52]), "f"(k_regs[20]), "f"(acc0_x_4));
                acc0_x_4 = _fma_374;
                float _fma_375;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_375) : "f"(S[53]), "f"(k_regs[21]), "f"(acc0_y_4));
                acc0_y_4 = _fma_375;
                float _fma_376;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_376) : "f"(S[54]), "f"(k_regs[22]), "f"(acc1_x_4));
                acc1_x_4 = _fma_376;
                float _fma_377;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_377) : "f"(S[55]), "f"(k_regs[23]), "f"(acc1_y_4));
                acc1_y_4 = _fma_377;
                float _fma_378;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_378) : "f"(S[56]), "f"(k_regs[24]), "f"(acc0_x_4));
                acc0_x_4 = _fma_378;
                float _fma_379;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_379) : "f"(S[57]), "f"(k_regs[25]), "f"(acc0_y_4));
                acc0_y_4 = _fma_379;
                float _fma_380;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_380) : "f"(S[58]), "f"(k_regs[26]), "f"(acc1_x_4));
                acc1_x_4 = _fma_380;
                float _fma_381;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_381) : "f"(S[59]), "f"(k_regs[27]), "f"(acc1_y_4));
                acc1_y_4 = _fma_381;
                float _fma_382;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_382) : "f"(S[60]), "f"(k_regs[28]), "f"(acc0_x_4));
                acc0_x_4 = _fma_382;
                float _fma_383;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_383) : "f"(S[61]), "f"(k_regs[29]), "f"(acc0_y_4));
                acc0_y_4 = _fma_383;
                float _fma_384;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_384) : "f"(S[62]), "f"(k_regs[30]), "f"(acc1_x_4));
                acc1_x_4 = _fma_384;
                float _fma_385;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_385) : "f"(S[63]), "f"(k_regs[31]), "f"(acc1_y_4));
                acc1_y_4 = _fma_385;
                float _add_386;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_386) : "f"(acc0_x_4), "f"(acc1_x_4));
                float _add_387;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_387) : "f"(acc0_y_4), "f"(acc1_y_4));
                float dot_4 = _add_386 + _add_387;
                float _shfl_xor_sync_388 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 1);
                dot_4 = dot_4 + _shfl_xor_sync_388;
                float _shfl_xor_sync_389 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 2);
                dot_4 = dot_4 + _shfl_xor_sync_389;
#else
                float2 acc0_4 = make_float2(0.0f, 0.0f);
                acc0_4 = fma_f32x2_rn_ftz(S2[16], k2[0], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[17], k2[1], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[18], k2[2], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[19], k2[3], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[20], k2[4], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[21], k2[5], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[22], k2[6], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[23], k2[7], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[24], k2[8], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[25], k2[9], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[26], k2[10], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[27], k2[11], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[28], k2[12], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[29], k2[13], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[30], k2[14], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[31], k2[15], acc0_4);
                float dot_4 = acc0_4.x + acc0_4.y;
                float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 1);
                dot_4 = dot_4 + _shfl_xor_sync_52;
                float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 2);
                dot_4 = dot_4 + _shfl_xor_sync_53;
#endif
                o4[1] = dot_4;
#if __CUDA_ARCH__ == 900
                float acc0_x_5 = 0.0f;
                float acc0_y_5 = 0.0f;
                float acc1_x_5 = 0.0f;
                float acc1_y_5 = 0.0f;
                float _fma_390;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_390) : "f"(S[64]), "f"(k_regs[0]), "f"(acc0_x_5));
                acc0_x_5 = _fma_390;
                float _fma_391;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_391) : "f"(S[65]), "f"(k_regs[1]), "f"(acc0_y_5));
                acc0_y_5 = _fma_391;
                float _fma_392;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_392) : "f"(S[66]), "f"(k_regs[2]), "f"(acc1_x_5));
                acc1_x_5 = _fma_392;
                float _fma_393;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_393) : "f"(S[67]), "f"(k_regs[3]), "f"(acc1_y_5));
                acc1_y_5 = _fma_393;
                float _fma_394;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_394) : "f"(S[68]), "f"(k_regs[4]), "f"(acc0_x_5));
                acc0_x_5 = _fma_394;
                float _fma_395;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_395) : "f"(S[69]), "f"(k_regs[5]), "f"(acc0_y_5));
                acc0_y_5 = _fma_395;
                float _fma_396;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_396) : "f"(S[70]), "f"(k_regs[6]), "f"(acc1_x_5));
                acc1_x_5 = _fma_396;
                float _fma_397;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_397) : "f"(S[71]), "f"(k_regs[7]), "f"(acc1_y_5));
                acc1_y_5 = _fma_397;
                float _fma_398;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_398) : "f"(S[72]), "f"(k_regs[8]), "f"(acc0_x_5));
                acc0_x_5 = _fma_398;
                float _fma_399;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_399) : "f"(S[73]), "f"(k_regs[9]), "f"(acc0_y_5));
                acc0_y_5 = _fma_399;
                float _fma_400;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_400) : "f"(S[74]), "f"(k_regs[10]), "f"(acc1_x_5));
                acc1_x_5 = _fma_400;
                float _fma_401;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_401) : "f"(S[75]), "f"(k_regs[11]), "f"(acc1_y_5));
                acc1_y_5 = _fma_401;
                float _fma_402;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_402) : "f"(S[76]), "f"(k_regs[12]), "f"(acc0_x_5));
                acc0_x_5 = _fma_402;
                float _fma_403;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_403) : "f"(S[77]), "f"(k_regs[13]), "f"(acc0_y_5));
                acc0_y_5 = _fma_403;
                float _fma_404;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_404) : "f"(S[78]), "f"(k_regs[14]), "f"(acc1_x_5));
                acc1_x_5 = _fma_404;
                float _fma_405;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_405) : "f"(S[79]), "f"(k_regs[15]), "f"(acc1_y_5));
                acc1_y_5 = _fma_405;
                float _fma_406;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_406) : "f"(S[80]), "f"(k_regs[16]), "f"(acc0_x_5));
                acc0_x_5 = _fma_406;
                float _fma_407;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_407) : "f"(S[81]), "f"(k_regs[17]), "f"(acc0_y_5));
                acc0_y_5 = _fma_407;
                float _fma_408;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_408) : "f"(S[82]), "f"(k_regs[18]), "f"(acc1_x_5));
                acc1_x_5 = _fma_408;
                float _fma_409;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_409) : "f"(S[83]), "f"(k_regs[19]), "f"(acc1_y_5));
                acc1_y_5 = _fma_409;
                float _fma_410;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_410) : "f"(S[84]), "f"(k_regs[20]), "f"(acc0_x_5));
                acc0_x_5 = _fma_410;
                float _fma_411;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_411) : "f"(S[85]), "f"(k_regs[21]), "f"(acc0_y_5));
                acc0_y_5 = _fma_411;
                float _fma_412;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_412) : "f"(S[86]), "f"(k_regs[22]), "f"(acc1_x_5));
                acc1_x_5 = _fma_412;
                float _fma_413;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_413) : "f"(S[87]), "f"(k_regs[23]), "f"(acc1_y_5));
                acc1_y_5 = _fma_413;
                float _fma_414;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_414) : "f"(S[88]), "f"(k_regs[24]), "f"(acc0_x_5));
                acc0_x_5 = _fma_414;
                float _fma_415;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_415) : "f"(S[89]), "f"(k_regs[25]), "f"(acc0_y_5));
                acc0_y_5 = _fma_415;
                float _fma_416;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_416) : "f"(S[90]), "f"(k_regs[26]), "f"(acc1_x_5));
                acc1_x_5 = _fma_416;
                float _fma_417;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_417) : "f"(S[91]), "f"(k_regs[27]), "f"(acc1_y_5));
                acc1_y_5 = _fma_417;
                float _fma_418;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_418) : "f"(S[92]), "f"(k_regs[28]), "f"(acc0_x_5));
                acc0_x_5 = _fma_418;
                float _fma_419;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_419) : "f"(S[93]), "f"(k_regs[29]), "f"(acc0_y_5));
                acc0_y_5 = _fma_419;
                float _fma_420;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_420) : "f"(S[94]), "f"(k_regs[30]), "f"(acc1_x_5));
                acc1_x_5 = _fma_420;
                float _fma_421;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_421) : "f"(S[95]), "f"(k_regs[31]), "f"(acc1_y_5));
                acc1_y_5 = _fma_421;
                float _add_422;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_422) : "f"(acc0_x_5), "f"(acc1_x_5));
                float _add_423;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_423) : "f"(acc0_y_5), "f"(acc1_y_5));
                float dot_5 = _add_422 + _add_423;
                float _shfl_xor_sync_424 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 1);
                dot_5 = dot_5 + _shfl_xor_sync_424;
                float _shfl_xor_sync_425 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 2);
                dot_5 = dot_5 + _shfl_xor_sync_425;
#else
                float2 acc0_5 = make_float2(0.0f, 0.0f);
                acc0_5 = fma_f32x2_rn_ftz(S2[32], k2[0], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[33], k2[1], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[34], k2[2], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[35], k2[3], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[36], k2[4], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[37], k2[5], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[38], k2[6], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[39], k2[7], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[40], k2[8], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[41], k2[9], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[42], k2[10], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[43], k2[11], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[44], k2[12], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[45], k2[13], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[46], k2[14], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[47], k2[15], acc0_5);
                float dot_5 = acc0_5.x + acc0_5.y;
                float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 1);
                dot_5 = dot_5 + _shfl_xor_sync_54;
                float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 2);
                dot_5 = dot_5 + _shfl_xor_sync_55;
#endif
                o4[2] = dot_5;
#if __CUDA_ARCH__ == 900
                float acc0_x_6 = 0.0f;
                float acc0_y_6 = 0.0f;
                float acc1_x_6 = 0.0f;
                float acc1_y_6 = 0.0f;
                float _fma_426;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_426) : "f"(S[96]), "f"(k_regs[0]), "f"(acc0_x_6));
                acc0_x_6 = _fma_426;
                float _fma_427;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_427) : "f"(S[97]), "f"(k_regs[1]), "f"(acc0_y_6));
                acc0_y_6 = _fma_427;
                float _fma_428;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_428) : "f"(S[98]), "f"(k_regs[2]), "f"(acc1_x_6));
                acc1_x_6 = _fma_428;
                float _fma_429;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_429) : "f"(S[99]), "f"(k_regs[3]), "f"(acc1_y_6));
                acc1_y_6 = _fma_429;
                float _fma_430;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_430) : "f"(S[100]), "f"(k_regs[4]), "f"(acc0_x_6));
                acc0_x_6 = _fma_430;
                float _fma_431;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_431) : "f"(S[101]), "f"(k_regs[5]), "f"(acc0_y_6));
                acc0_y_6 = _fma_431;
                float _fma_432;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_432) : "f"(S[102]), "f"(k_regs[6]), "f"(acc1_x_6));
                acc1_x_6 = _fma_432;
                float _fma_433;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_433) : "f"(S[103]), "f"(k_regs[7]), "f"(acc1_y_6));
                acc1_y_6 = _fma_433;
                float _fma_434;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_434) : "f"(S[104]), "f"(k_regs[8]), "f"(acc0_x_6));
                acc0_x_6 = _fma_434;
                float _fma_435;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_435) : "f"(S[105]), "f"(k_regs[9]), "f"(acc0_y_6));
                acc0_y_6 = _fma_435;
                float _fma_436;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_436) : "f"(S[106]), "f"(k_regs[10]), "f"(acc1_x_6));
                acc1_x_6 = _fma_436;
                float _fma_437;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_437) : "f"(S[107]), "f"(k_regs[11]), "f"(acc1_y_6));
                acc1_y_6 = _fma_437;
                float _fma_438;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_438) : "f"(S[108]), "f"(k_regs[12]), "f"(acc0_x_6));
                acc0_x_6 = _fma_438;
                float _fma_439;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_439) : "f"(S[109]), "f"(k_regs[13]), "f"(acc0_y_6));
                acc0_y_6 = _fma_439;
                float _fma_440;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_440) : "f"(S[110]), "f"(k_regs[14]), "f"(acc1_x_6));
                acc1_x_6 = _fma_440;
                float _fma_441;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_441) : "f"(S[111]), "f"(k_regs[15]), "f"(acc1_y_6));
                acc1_y_6 = _fma_441;
                float _fma_442;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_442) : "f"(S[112]), "f"(k_regs[16]), "f"(acc0_x_6));
                acc0_x_6 = _fma_442;
                float _fma_443;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_443) : "f"(S[113]), "f"(k_regs[17]), "f"(acc0_y_6));
                acc0_y_6 = _fma_443;
                float _fma_444;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_444) : "f"(S[114]), "f"(k_regs[18]), "f"(acc1_x_6));
                acc1_x_6 = _fma_444;
                float _fma_445;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_445) : "f"(S[115]), "f"(k_regs[19]), "f"(acc1_y_6));
                acc1_y_6 = _fma_445;
                float _fma_446;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_446) : "f"(S[116]), "f"(k_regs[20]), "f"(acc0_x_6));
                acc0_x_6 = _fma_446;
                float _fma_447;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_447) : "f"(S[117]), "f"(k_regs[21]), "f"(acc0_y_6));
                acc0_y_6 = _fma_447;
                float _fma_448;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_448) : "f"(S[118]), "f"(k_regs[22]), "f"(acc1_x_6));
                acc1_x_6 = _fma_448;
                float _fma_449;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_449) : "f"(S[119]), "f"(k_regs[23]), "f"(acc1_y_6));
                acc1_y_6 = _fma_449;
                float _fma_450;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_450) : "f"(S[120]), "f"(k_regs[24]), "f"(acc0_x_6));
                acc0_x_6 = _fma_450;
                float _fma_451;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_451) : "f"(S[121]), "f"(k_regs[25]), "f"(acc0_y_6));
                acc0_y_6 = _fma_451;
                float _fma_452;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_452) : "f"(S[122]), "f"(k_regs[26]), "f"(acc1_x_6));
                acc1_x_6 = _fma_452;
                float _fma_453;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_453) : "f"(S[123]), "f"(k_regs[27]), "f"(acc1_y_6));
                acc1_y_6 = _fma_453;
                float _fma_454;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_454) : "f"(S[124]), "f"(k_regs[28]), "f"(acc0_x_6));
                acc0_x_6 = _fma_454;
                float _fma_455;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_455) : "f"(S[125]), "f"(k_regs[29]), "f"(acc0_y_6));
                acc0_y_6 = _fma_455;
                float _fma_456;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_456) : "f"(S[126]), "f"(k_regs[30]), "f"(acc1_x_6));
                acc1_x_6 = _fma_456;
                float _fma_457;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_457) : "f"(S[127]), "f"(k_regs[31]), "f"(acc1_y_6));
                acc1_y_6 = _fma_457;
                float _add_458;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_458) : "f"(acc0_x_6), "f"(acc1_x_6));
                float _add_459;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_459) : "f"(acc0_y_6), "f"(acc1_y_6));
                float dot_6 = _add_458 + _add_459;
                float _shfl_xor_sync_460 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 1);
                dot_6 = dot_6 + _shfl_xor_sync_460;
                float _shfl_xor_sync_461 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 2);
                dot_6 = dot_6 + _shfl_xor_sync_461;
#else
                float2 acc0_6 = make_float2(0.0f, 0.0f);
                acc0_6 = fma_f32x2_rn_ftz(S2[48], k2[0], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[49], k2[1], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[50], k2[2], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[51], k2[3], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[52], k2[4], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[53], k2[5], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[54], k2[6], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[55], k2[7], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[56], k2[8], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[57], k2[9], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[58], k2[10], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[59], k2[11], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[60], k2[12], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[61], k2[13], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[62], k2[14], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[63], k2[15], acc0_6);
                float dot_6 = acc0_6.x + acc0_6.y;
                float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 1);
                dot_6 = dot_6 + _shfl_xor_sync_56;
                float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 2);
                dot_6 = dot_6 + _shfl_xor_sync_57;
#endif
                o4[3] = dot_6;
                if (u == 0) {
                    long long out_index = (long long)n * out_stride_p0 + (long long)t_1 * out_stride_p1 + (long long)h * out_stride_p2;
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
                int ssm_slot = sSsm[t_1];
                int snap_valid = ((use_ssm_state_indices != 0 && ssm_slot >= 0 && state_pool_size > (long long)ssm_slot) ? 1 : 0);
                int safe_slot = ((snap_valid != 0) ? ssm_slot : 0);
                long long snap_base = (long long)safe_slot * state_stride_p0 + head_rows_off;
                if (philox_rounds != 0) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 0));
#if __CUDA_ARCH__ == 900
                    float _abs_462;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_462) : "f"(S[0]));
                    unsigned int s = __as_u32(mul_f32_rz_noftz(_abs_462, __uint_as_float(125829120)));
                    unsigned int _min_463 = ((s) < (260046848) ? (s) : (260046848));
                    unsigned int h_0 = _min_463 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag = ((s <= 2139095040) ? h_0 : 32256);
                    float _abs_464;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_464) : "f"(S[1]));
                    unsigned int s_0 = __as_u32(mul_f32_rz_noftz(_abs_464, __uint_as_float(125829120)));
                    unsigned int _min_465 = ((s_0) < (260046848) ? (s_0) : (260046848));
                    unsigned int h_1 = _min_465 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_0 = ((s_0 <= 2139095040) ? h_1 : 32256);
                    uint32_t mags;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags) : "r"(mag), "r"(mag_0));
                    uint32_t signs;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs) : "r"(__as_u32(S[0])), "r"(__as_u32(S[1])));
                    unsigned int word = signs & 2147516416u | mags;
                    h_words[0] = word;
                    float _abs_466;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_466) : "f"(S[2]));
                    unsigned int s_1 = __as_u32(mul_f32_rz_noftz(_abs_466, __uint_as_float(125829120)));
                    unsigned int _min_467 = ((s_1) < (260046848) ? (s_1) : (260046848));
                    unsigned int h_2 = _min_467 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_1 = ((s_1 <= 2139095040) ? h_2 : 32256);
                    float _abs_468;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_468) : "f"(S[3]));
                    unsigned int s_2 = __as_u32(mul_f32_rz_noftz(_abs_468, __uint_as_float(125829120)));
                    unsigned int _min_469 = ((s_2) < (260046848) ? (s_2) : (260046848));
                    unsigned int h_3 = _min_469 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_2 = ((s_2 <= 2139095040) ? h_3 : 32256);
                    uint32_t mags_0;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_0) : "r"(mag_1), "r"(mag_2));
                    uint32_t signs_0;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_0) : "r"(__as_u32(S[2])), "r"(__as_u32(S[3])));
                    unsigned int word_0 = signs_0 & 2147516416u | mags_0;
                    h_words[1] = word_0;
#else
                    uint32_t _cvt_rs_x2_f32_58;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_58) : "f"(S2[0].y), "f"(S2[0].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_58;
                    uint32_t _cvt_rs_x2_f32_59;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_59) : "f"(S2[1].y), "f"(S2[1].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_59;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_470;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_470) : "f"(S[16]));
                    unsigned int s_3 = __as_u32(mul_f32_rz_noftz(_abs_470, __uint_as_float(125829120)));
                    unsigned int _min_471 = ((s_3) < (260046848) ? (s_3) : (260046848));
                    unsigned int h_4 = _min_471 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_3 = ((s_3 <= 2139095040) ? h_4 : 32256);
                    float _abs_472;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_472) : "f"(S[17]));
                    unsigned int s_4 = __as_u32(mul_f32_rz_noftz(_abs_472, __uint_as_float(125829120)));
                    unsigned int _min_473 = ((s_4) < (260046848) ? (s_4) : (260046848));
                    unsigned int h_5 = _min_473 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_4 = ((s_4 <= 2139095040) ? h_5 : 32256);
                    uint32_t mags_1;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_1) : "r"(mag_3), "r"(mag_4));
                    uint32_t signs_1;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_1) : "r"(__as_u32(S[16])), "r"(__as_u32(S[17])));
                    unsigned int word_1 = signs_1 & 2147516416u | mags_1;
                    h_words[0] = word_1;
                    float _abs_474;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_474) : "f"(S[18]));
                    unsigned int s_5 = __as_u32(mul_f32_rz_noftz(_abs_474, __uint_as_float(125829120)));
                    unsigned int _min_475 = ((s_5) < (260046848) ? (s_5) : (260046848));
                    unsigned int h_6 = _min_475 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_5 = ((s_5 <= 2139095040) ? h_6 : 32256);
                    float _abs_476;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_476) : "f"(S[19]));
                    unsigned int s_6 = __as_u32(mul_f32_rz_noftz(_abs_476, __uint_as_float(125829120)));
                    unsigned int _min_477 = ((s_6) < (260046848) ? (s_6) : (260046848));
                    unsigned int h_7 = _min_477 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_6 = ((s_6 <= 2139095040) ? h_7 : 32256);
                    uint32_t mags_2;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_2) : "r"(mag_5), "r"(mag_6));
                    uint32_t signs_2;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_2) : "r"(__as_u32(S[18])), "r"(__as_u32(S[19])));
                    unsigned int word_2 = signs_2 & 2147516416u | mags_2;
                    h_words[1] = word_2;
#else
                    uint32_t _cvt_rs_x2_f32_60;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_60) : "f"(S2[8].y), "f"(S2[8].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_60;
                    uint32_t _cvt_rs_x2_f32_61;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_61) : "f"(S2[9].y), "f"(S2[9].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_61;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 1024));
#if __CUDA_ARCH__ == 900
                    float _abs_478;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_478) : "f"(S[4]));
                    unsigned int s_7 = __as_u32(mul_f32_rz_noftz(_abs_478, __uint_as_float(125829120)));
                    unsigned int _min_479 = ((s_7) < (260046848) ? (s_7) : (260046848));
                    unsigned int h_8 = _min_479 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_7 = ((s_7 <= 2139095040) ? h_8 : 32256);
                    float _abs_480;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_480) : "f"(S[5]));
                    unsigned int s_8 = __as_u32(mul_f32_rz_noftz(_abs_480, __uint_as_float(125829120)));
                    unsigned int _min_481 = ((s_8) < (260046848) ? (s_8) : (260046848));
                    unsigned int h_9 = _min_481 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_8 = ((s_8 <= 2139095040) ? h_9 : 32256);
                    uint32_t mags_3;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_3) : "r"(mag_7), "r"(mag_8));
                    uint32_t signs_3;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_3) : "r"(__as_u32(S[4])), "r"(__as_u32(S[5])));
                    unsigned int word_3 = signs_3 & 2147516416u | mags_3;
                    h_words[0] = word_3;
                    float _abs_482;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_482) : "f"(S[6]));
                    unsigned int s_9 = __as_u32(mul_f32_rz_noftz(_abs_482, __uint_as_float(125829120)));
                    unsigned int _min_483 = ((s_9) < (260046848) ? (s_9) : (260046848));
                    unsigned int h_10 = _min_483 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_9 = ((s_9 <= 2139095040) ? h_10 : 32256);
                    float _abs_484;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_484) : "f"(S[7]));
                    unsigned int s_10 = __as_u32(mul_f32_rz_noftz(_abs_484, __uint_as_float(125829120)));
                    unsigned int _min_485 = ((s_10) < (260046848) ? (s_10) : (260046848));
                    unsigned int h_11 = _min_485 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_10 = ((s_10 <= 2139095040) ? h_11 : 32256);
                    uint32_t mags_4;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_4) : "r"(mag_9), "r"(mag_10));
                    uint32_t signs_4;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_4) : "r"(__as_u32(S[6])), "r"(__as_u32(S[7])));
                    unsigned int word_4 = signs_4 & 2147516416u | mags_4;
                    h_words[1] = word_4;
#else
                    uint32_t _cvt_rs_x2_f32_62;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_62) : "f"(S2[2].y), "f"(S2[2].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_62;
                    uint32_t _cvt_rs_x2_f32_63;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_63) : "f"(S2[3].y), "f"(S2[3].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_63;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 16ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_486;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_486) : "f"(S[20]));
                    unsigned int s_11 = __as_u32(mul_f32_rz_noftz(_abs_486, __uint_as_float(125829120)));
                    unsigned int _min_487 = ((s_11) < (260046848) ? (s_11) : (260046848));
                    unsigned int h_12 = _min_487 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_11 = ((s_11 <= 2139095040) ? h_12 : 32256);
                    float _abs_488;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_488) : "f"(S[21]));
                    unsigned int s_12 = __as_u32(mul_f32_rz_noftz(_abs_488, __uint_as_float(125829120)));
                    unsigned int _min_489 = ((s_12) < (260046848) ? (s_12) : (260046848));
                    unsigned int h_13 = _min_489 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_12 = ((s_12 <= 2139095040) ? h_13 : 32256);
                    uint32_t mags_5;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_5) : "r"(mag_11), "r"(mag_12));
                    uint32_t signs_5;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_5) : "r"(__as_u32(S[20])), "r"(__as_u32(S[21])));
                    unsigned int word_5 = signs_5 & 2147516416u | mags_5;
                    h_words[0] = word_5;
                    float _abs_490;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_490) : "f"(S[22]));
                    unsigned int s_13 = __as_u32(mul_f32_rz_noftz(_abs_490, __uint_as_float(125829120)));
                    unsigned int _min_491 = ((s_13) < (260046848) ? (s_13) : (260046848));
                    unsigned int h_14 = _min_491 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_13 = ((s_13 <= 2139095040) ? h_14 : 32256);
                    float _abs_492;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_492) : "f"(S[23]));
                    unsigned int s_14 = __as_u32(mul_f32_rz_noftz(_abs_492, __uint_as_float(125829120)));
                    unsigned int _min_493 = ((s_14) < (260046848) ? (s_14) : (260046848));
                    unsigned int h_15 = _min_493 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_14 = ((s_14 <= 2139095040) ? h_15 : 32256);
                    uint32_t mags_6;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_6) : "r"(mag_13), "r"(mag_14));
                    uint32_t signs_6;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_6) : "r"(__as_u32(S[22])), "r"(__as_u32(S[23])));
                    unsigned int word_6 = signs_6 & 2147516416u | mags_6;
                    h_words[1] = word_6;
#else
                    uint32_t _cvt_rs_x2_f32_64;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_64) : "f"(S2[10].y), "f"(S2[10].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_64;
                    uint32_t _cvt_rs_x2_f32_65;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_65) : "f"(S2[11].y), "f"(S2[11].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_65;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 80ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 2048));
#if __CUDA_ARCH__ == 900
                    float _abs_494;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_494) : "f"(S[8]));
                    unsigned int s_15 = __as_u32(mul_f32_rz_noftz(_abs_494, __uint_as_float(125829120)));
                    unsigned int _min_495 = ((s_15) < (260046848) ? (s_15) : (260046848));
                    unsigned int h_16 = _min_495 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_15 = ((s_15 <= 2139095040) ? h_16 : 32256);
                    float _abs_496;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_496) : "f"(S[9]));
                    unsigned int s_16 = __as_u32(mul_f32_rz_noftz(_abs_496, __uint_as_float(125829120)));
                    unsigned int _min_497 = ((s_16) < (260046848) ? (s_16) : (260046848));
                    unsigned int h_17 = _min_497 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_16 = ((s_16 <= 2139095040) ? h_17 : 32256);
                    uint32_t mags_7;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_7) : "r"(mag_15), "r"(mag_16));
                    uint32_t signs_7;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_7) : "r"(__as_u32(S[8])), "r"(__as_u32(S[9])));
                    unsigned int word_7 = signs_7 & 2147516416u | mags_7;
                    h_words[0] = word_7;
                    float _abs_498;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_498) : "f"(S[10]));
                    unsigned int s_17 = __as_u32(mul_f32_rz_noftz(_abs_498, __uint_as_float(125829120)));
                    unsigned int _min_499 = ((s_17) < (260046848) ? (s_17) : (260046848));
                    unsigned int h_18 = _min_499 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_17 = ((s_17 <= 2139095040) ? h_18 : 32256);
                    float _abs_500;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_500) : "f"(S[11]));
                    unsigned int s_18 = __as_u32(mul_f32_rz_noftz(_abs_500, __uint_as_float(125829120)));
                    unsigned int _min_501 = ((s_18) < (260046848) ? (s_18) : (260046848));
                    unsigned int h_19 = _min_501 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_18 = ((s_18 <= 2139095040) ? h_19 : 32256);
                    uint32_t mags_8;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_8) : "r"(mag_17), "r"(mag_18));
                    uint32_t signs_8;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_8) : "r"(__as_u32(S[10])), "r"(__as_u32(S[11])));
                    unsigned int word_8 = signs_8 & 2147516416u | mags_8;
                    h_words[1] = word_8;
#else
                    uint32_t _cvt_rs_x2_f32_66;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_66) : "f"(S2[4].y), "f"(S2[4].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_66;
                    uint32_t _cvt_rs_x2_f32_67;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_67) : "f"(S2[5].y), "f"(S2[5].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_67;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_502;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_502) : "f"(S[24]));
                    unsigned int s_19 = __as_u32(mul_f32_rz_noftz(_abs_502, __uint_as_float(125829120)));
                    unsigned int _min_503 = ((s_19) < (260046848) ? (s_19) : (260046848));
                    unsigned int h_20 = _min_503 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_19 = ((s_19 <= 2139095040) ? h_20 : 32256);
                    float _abs_504;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_504) : "f"(S[25]));
                    unsigned int s_20 = __as_u32(mul_f32_rz_noftz(_abs_504, __uint_as_float(125829120)));
                    unsigned int _min_505 = ((s_20) < (260046848) ? (s_20) : (260046848));
                    unsigned int h_21 = _min_505 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_20 = ((s_20 <= 2139095040) ? h_21 : 32256);
                    uint32_t mags_9;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_9) : "r"(mag_19), "r"(mag_20));
                    uint32_t signs_9;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_9) : "r"(__as_u32(S[24])), "r"(__as_u32(S[25])));
                    unsigned int word_9 = signs_9 & 2147516416u | mags_9;
                    h_words[0] = word_9;
                    float _abs_506;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_506) : "f"(S[26]));
                    unsigned int s_21 = __as_u32(mul_f32_rz_noftz(_abs_506, __uint_as_float(125829120)));
                    unsigned int _min_507 = ((s_21) < (260046848) ? (s_21) : (260046848));
                    unsigned int h_22 = _min_507 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_21 = ((s_21 <= 2139095040) ? h_22 : 32256);
                    float _abs_508;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_508) : "f"(S[27]));
                    unsigned int s_22 = __as_u32(mul_f32_rz_noftz(_abs_508, __uint_as_float(125829120)));
                    unsigned int _min_509 = ((s_22) < (260046848) ? (s_22) : (260046848));
                    unsigned int h_23 = _min_509 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_22 = ((s_22 <= 2139095040) ? h_23 : 32256);
                    uint32_t mags_10;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_10) : "r"(mag_21), "r"(mag_22));
                    uint32_t signs_10;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_10) : "r"(__as_u32(S[26])), "r"(__as_u32(S[27])));
                    unsigned int word_10 = signs_10 & 2147516416u | mags_10;
                    h_words[1] = word_10;
#else
                    uint32_t _cvt_rs_x2_f32_68;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_68) : "f"(S2[12].y), "f"(S2[12].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_68;
                    uint32_t _cvt_rs_x2_f32_69;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_69) : "f"(S2[13].y), "f"(S2[13].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_69;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 3072));
#if __CUDA_ARCH__ == 900
                    float _abs_510;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_510) : "f"(S[12]));
                    unsigned int s_23 = __as_u32(mul_f32_rz_noftz(_abs_510, __uint_as_float(125829120)));
                    unsigned int _min_511 = ((s_23) < (260046848) ? (s_23) : (260046848));
                    unsigned int h_24 = _min_511 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_23 = ((s_23 <= 2139095040) ? h_24 : 32256);
                    float _abs_512;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_512) : "f"(S[13]));
                    unsigned int s_24 = __as_u32(mul_f32_rz_noftz(_abs_512, __uint_as_float(125829120)));
                    unsigned int _min_513 = ((s_24) < (260046848) ? (s_24) : (260046848));
                    unsigned int h_25 = _min_513 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_24 = ((s_24 <= 2139095040) ? h_25 : 32256);
                    uint32_t mags_11;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_11) : "r"(mag_23), "r"(mag_24));
                    uint32_t signs_11;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_11) : "r"(__as_u32(S[12])), "r"(__as_u32(S[13])));
                    unsigned int word_11 = signs_11 & 2147516416u | mags_11;
                    h_words[0] = word_11;
                    float _abs_514;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_514) : "f"(S[14]));
                    unsigned int s_25 = __as_u32(mul_f32_rz_noftz(_abs_514, __uint_as_float(125829120)));
                    unsigned int _min_515 = ((s_25) < (260046848) ? (s_25) : (260046848));
                    unsigned int h_26 = _min_515 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_25 = ((s_25 <= 2139095040) ? h_26 : 32256);
                    float _abs_516;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_516) : "f"(S[15]));
                    unsigned int s_26 = __as_u32(mul_f32_rz_noftz(_abs_516, __uint_as_float(125829120)));
                    unsigned int _min_517 = ((s_26) < (260046848) ? (s_26) : (260046848));
                    unsigned int h_27 = _min_517 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_26 = ((s_26 <= 2139095040) ? h_27 : 32256);
                    uint32_t mags_12;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_12) : "r"(mag_25), "r"(mag_26));
                    uint32_t signs_12;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_12) : "r"(__as_u32(S[14])), "r"(__as_u32(S[15])));
                    unsigned int word_12 = signs_12 & 2147516416u | mags_12;
                    h_words[1] = word_12;
#else
                    uint32_t _cvt_rs_x2_f32_70;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_70) : "f"(S2[6].y), "f"(S2[6].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_70;
                    uint32_t _cvt_rs_x2_f32_71;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_71) : "f"(S2[7].y), "f"(S2[7].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_71;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 48ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_518;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_518) : "f"(S[28]));
                    unsigned int s_27 = __as_u32(mul_f32_rz_noftz(_abs_518, __uint_as_float(125829120)));
                    unsigned int _min_519 = ((s_27) < (260046848) ? (s_27) : (260046848));
                    unsigned int h_28 = _min_519 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_27 = ((s_27 <= 2139095040) ? h_28 : 32256);
                    float _abs_520;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_520) : "f"(S[29]));
                    unsigned int s_28 = __as_u32(mul_f32_rz_noftz(_abs_520, __uint_as_float(125829120)));
                    unsigned int _min_521 = ((s_28) < (260046848) ? (s_28) : (260046848));
                    unsigned int h_29 = _min_521 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_28 = ((s_28 <= 2139095040) ? h_29 : 32256);
                    uint32_t mags_13;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_13) : "r"(mag_27), "r"(mag_28));
                    uint32_t signs_13;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_13) : "r"(__as_u32(S[28])), "r"(__as_u32(S[29])));
                    unsigned int word_13 = signs_13 & 2147516416u | mags_13;
                    h_words[0] = word_13;
                    float _abs_522;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_522) : "f"(S[30]));
                    unsigned int s_29 = __as_u32(mul_f32_rz_noftz(_abs_522, __uint_as_float(125829120)));
                    unsigned int _min_523 = ((s_29) < (260046848) ? (s_29) : (260046848));
                    unsigned int h_30 = _min_523 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_29 = ((s_29 <= 2139095040) ? h_30 : 32256);
                    float _abs_524;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_524) : "f"(S[31]));
                    unsigned int s_30 = __as_u32(mul_f32_rz_noftz(_abs_524, __uint_as_float(125829120)));
                    unsigned int _min_525 = ((s_30) < (260046848) ? (s_30) : (260046848));
                    unsigned int h_31 = _min_525 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_30 = ((s_30 <= 2139095040) ? h_31 : 32256);
                    uint32_t mags_14;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_14) : "r"(mag_29), "r"(mag_30));
                    uint32_t signs_14;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_14) : "r"(__as_u32(S[30])), "r"(__as_u32(S[31])));
                    unsigned int word_14 = signs_14 & 2147516416u | mags_14;
                    h_words[1] = word_14;
#else
                    uint32_t _cvt_rs_x2_f32_72;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_72) : "f"(S2[14].y), "f"(S2[14].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_72;
                    uint32_t _cvt_rs_x2_f32_73;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_73) : "f"(S2[15].y), "f"(S2[15].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_73;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 112ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 4096));
#if __CUDA_ARCH__ == 900
                    float _abs_526;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_526) : "f"(S[32]));
                    unsigned int s_31 = __as_u32(mul_f32_rz_noftz(_abs_526, __uint_as_float(125829120)));
                    unsigned int _min_527 = ((s_31) < (260046848) ? (s_31) : (260046848));
                    unsigned int h_32 = _min_527 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_31 = ((s_31 <= 2139095040) ? h_32 : 32256);
                    float _abs_528;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_528) : "f"(S[33]));
                    unsigned int s_32 = __as_u32(mul_f32_rz_noftz(_abs_528, __uint_as_float(125829120)));
                    unsigned int _min_529 = ((s_32) < (260046848) ? (s_32) : (260046848));
                    unsigned int h_33 = _min_529 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_32 = ((s_32 <= 2139095040) ? h_33 : 32256);
                    uint32_t mags_15;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_15) : "r"(mag_31), "r"(mag_32));
                    uint32_t signs_15;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_15) : "r"(__as_u32(S[32])), "r"(__as_u32(S[33])));
                    unsigned int word_15 = signs_15 & 2147516416u | mags_15;
                    h_words[0] = word_15;
                    float _abs_530;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_530) : "f"(S[34]));
                    unsigned int s_33 = __as_u32(mul_f32_rz_noftz(_abs_530, __uint_as_float(125829120)));
                    unsigned int _min_531 = ((s_33) < (260046848) ? (s_33) : (260046848));
                    unsigned int h_34 = _min_531 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_33 = ((s_33 <= 2139095040) ? h_34 : 32256);
                    float _abs_532;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_532) : "f"(S[35]));
                    unsigned int s_34 = __as_u32(mul_f32_rz_noftz(_abs_532, __uint_as_float(125829120)));
                    unsigned int _min_533 = ((s_34) < (260046848) ? (s_34) : (260046848));
                    unsigned int h_35 = _min_533 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_34 = ((s_34 <= 2139095040) ? h_35 : 32256);
                    uint32_t mags_16;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_16) : "r"(mag_33), "r"(mag_34));
                    uint32_t signs_16;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_16) : "r"(__as_u32(S[34])), "r"(__as_u32(S[35])));
                    unsigned int word_16 = signs_16 & 2147516416u | mags_16;
                    h_words[1] = word_16;
#else
                    uint32_t _cvt_rs_x2_f32_74;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_74) : "f"(S2[16].y), "f"(S2[16].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_74;
                    uint32_t _cvt_rs_x2_f32_75;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_75) : "f"(S2[17].y), "f"(S2[17].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_75;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_534;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_534) : "f"(S[48]));
                    unsigned int s_35 = __as_u32(mul_f32_rz_noftz(_abs_534, __uint_as_float(125829120)));
                    unsigned int _min_535 = ((s_35) < (260046848) ? (s_35) : (260046848));
                    unsigned int h_36 = _min_535 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_35 = ((s_35 <= 2139095040) ? h_36 : 32256);
                    float _abs_536;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_536) : "f"(S[49]));
                    unsigned int s_36 = __as_u32(mul_f32_rz_noftz(_abs_536, __uint_as_float(125829120)));
                    unsigned int _min_537 = ((s_36) < (260046848) ? (s_36) : (260046848));
                    unsigned int h_37 = _min_537 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_36 = ((s_36 <= 2139095040) ? h_37 : 32256);
                    uint32_t mags_17;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_17) : "r"(mag_35), "r"(mag_36));
                    uint32_t signs_17;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_17) : "r"(__as_u32(S[48])), "r"(__as_u32(S[49])));
                    unsigned int word_17 = signs_17 & 2147516416u | mags_17;
                    h_words[0] = word_17;
                    float _abs_538;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_538) : "f"(S[50]));
                    unsigned int s_37 = __as_u32(mul_f32_rz_noftz(_abs_538, __uint_as_float(125829120)));
                    unsigned int _min_539 = ((s_37) < (260046848) ? (s_37) : (260046848));
                    unsigned int h_38 = _min_539 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_37 = ((s_37 <= 2139095040) ? h_38 : 32256);
                    float _abs_540;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_540) : "f"(S[51]));
                    unsigned int s_38 = __as_u32(mul_f32_rz_noftz(_abs_540, __uint_as_float(125829120)));
                    unsigned int _min_541 = ((s_38) < (260046848) ? (s_38) : (260046848));
                    unsigned int h_39 = _min_541 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_38 = ((s_38 <= 2139095040) ? h_39 : 32256);
                    uint32_t mags_18;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_18) : "r"(mag_37), "r"(mag_38));
                    uint32_t signs_18;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_18) : "r"(__as_u32(S[50])), "r"(__as_u32(S[51])));
                    unsigned int word_18 = signs_18 & 2147516416u | mags_18;
                    h_words[1] = word_18;
#else
                    uint32_t _cvt_rs_x2_f32_76;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_76) : "f"(S2[24].y), "f"(S2[24].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_76;
                    uint32_t _cvt_rs_x2_f32_77;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_77) : "f"(S2[25].y), "f"(S2[25].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_77;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 5120));
#if __CUDA_ARCH__ == 900
                    float _abs_542;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_542) : "f"(S[36]));
                    unsigned int s_39 = __as_u32(mul_f32_rz_noftz(_abs_542, __uint_as_float(125829120)));
                    unsigned int _min_543 = ((s_39) < (260046848) ? (s_39) : (260046848));
                    unsigned int h_40 = _min_543 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_39 = ((s_39 <= 2139095040) ? h_40 : 32256);
                    float _abs_544;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_544) : "f"(S[37]));
                    unsigned int s_40 = __as_u32(mul_f32_rz_noftz(_abs_544, __uint_as_float(125829120)));
                    unsigned int _min_545 = ((s_40) < (260046848) ? (s_40) : (260046848));
                    unsigned int h_41 = _min_545 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_40 = ((s_40 <= 2139095040) ? h_41 : 32256);
                    uint32_t mags_19;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_19) : "r"(mag_39), "r"(mag_40));
                    uint32_t signs_19;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_19) : "r"(__as_u32(S[36])), "r"(__as_u32(S[37])));
                    unsigned int word_19 = signs_19 & 2147516416u | mags_19;
                    h_words[0] = word_19;
                    float _abs_546;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_546) : "f"(S[38]));
                    unsigned int s_41 = __as_u32(mul_f32_rz_noftz(_abs_546, __uint_as_float(125829120)));
                    unsigned int _min_547 = ((s_41) < (260046848) ? (s_41) : (260046848));
                    unsigned int h_42 = _min_547 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_41 = ((s_41 <= 2139095040) ? h_42 : 32256);
                    float _abs_548;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_548) : "f"(S[39]));
                    unsigned int s_42 = __as_u32(mul_f32_rz_noftz(_abs_548, __uint_as_float(125829120)));
                    unsigned int _min_549 = ((s_42) < (260046848) ? (s_42) : (260046848));
                    unsigned int h_43 = _min_549 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_42 = ((s_42 <= 2139095040) ? h_43 : 32256);
                    uint32_t mags_20;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_20) : "r"(mag_41), "r"(mag_42));
                    uint32_t signs_20;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_20) : "r"(__as_u32(S[38])), "r"(__as_u32(S[39])));
                    unsigned int word_20 = signs_20 & 2147516416u | mags_20;
                    h_words[1] = word_20;
#else
                    uint32_t _cvt_rs_x2_f32_78;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_78) : "f"(S2[18].y), "f"(S2[18].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_78;
                    uint32_t _cvt_rs_x2_f32_79;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_79) : "f"(S2[19].y), "f"(S2[19].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_79;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 16ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_550;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_550) : "f"(S[52]));
                    unsigned int s_43 = __as_u32(mul_f32_rz_noftz(_abs_550, __uint_as_float(125829120)));
                    unsigned int _min_551 = ((s_43) < (260046848) ? (s_43) : (260046848));
                    unsigned int h_44 = _min_551 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_43 = ((s_43 <= 2139095040) ? h_44 : 32256);
                    float _abs_552;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_552) : "f"(S[53]));
                    unsigned int s_44 = __as_u32(mul_f32_rz_noftz(_abs_552, __uint_as_float(125829120)));
                    unsigned int _min_553 = ((s_44) < (260046848) ? (s_44) : (260046848));
                    unsigned int h_45 = _min_553 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_44 = ((s_44 <= 2139095040) ? h_45 : 32256);
                    uint32_t mags_21;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_21) : "r"(mag_43), "r"(mag_44));
                    uint32_t signs_21;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_21) : "r"(__as_u32(S[52])), "r"(__as_u32(S[53])));
                    unsigned int word_21 = signs_21 & 2147516416u | mags_21;
                    h_words[0] = word_21;
                    float _abs_554;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_554) : "f"(S[54]));
                    unsigned int s_45 = __as_u32(mul_f32_rz_noftz(_abs_554, __uint_as_float(125829120)));
                    unsigned int _min_555 = ((s_45) < (260046848) ? (s_45) : (260046848));
                    unsigned int h_46 = _min_555 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_45 = ((s_45 <= 2139095040) ? h_46 : 32256);
                    float _abs_556;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_556) : "f"(S[55]));
                    unsigned int s_46 = __as_u32(mul_f32_rz_noftz(_abs_556, __uint_as_float(125829120)));
                    unsigned int _min_557 = ((s_46) < (260046848) ? (s_46) : (260046848));
                    unsigned int h_47 = _min_557 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_46 = ((s_46 <= 2139095040) ? h_47 : 32256);
                    uint32_t mags_22;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_22) : "r"(mag_45), "r"(mag_46));
                    uint32_t signs_22;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_22) : "r"(__as_u32(S[54])), "r"(__as_u32(S[55])));
                    unsigned int word_22 = signs_22 & 2147516416u | mags_22;
                    h_words[1] = word_22;
#else
                    uint32_t _cvt_rs_x2_f32_80;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_80) : "f"(S2[26].y), "f"(S2[26].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_80;
                    uint32_t _cvt_rs_x2_f32_81;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_81) : "f"(S2[27].y), "f"(S2[27].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_81;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 80ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 6144));
#if __CUDA_ARCH__ == 900
                    float _abs_558;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_558) : "f"(S[40]));
                    unsigned int s_47 = __as_u32(mul_f32_rz_noftz(_abs_558, __uint_as_float(125829120)));
                    unsigned int _min_559 = ((s_47) < (260046848) ? (s_47) : (260046848));
                    unsigned int h_48 = _min_559 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_47 = ((s_47 <= 2139095040) ? h_48 : 32256);
                    float _abs_560;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_560) : "f"(S[41]));
                    unsigned int s_48 = __as_u32(mul_f32_rz_noftz(_abs_560, __uint_as_float(125829120)));
                    unsigned int _min_561 = ((s_48) < (260046848) ? (s_48) : (260046848));
                    unsigned int h_49 = _min_561 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_48 = ((s_48 <= 2139095040) ? h_49 : 32256);
                    uint32_t mags_23;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_23) : "r"(mag_47), "r"(mag_48));
                    uint32_t signs_23;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_23) : "r"(__as_u32(S[40])), "r"(__as_u32(S[41])));
                    unsigned int word_23 = signs_23 & 2147516416u | mags_23;
                    h_words[0] = word_23;
                    float _abs_562;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_562) : "f"(S[42]));
                    unsigned int s_49 = __as_u32(mul_f32_rz_noftz(_abs_562, __uint_as_float(125829120)));
                    unsigned int _min_563 = ((s_49) < (260046848) ? (s_49) : (260046848));
                    unsigned int h_50 = _min_563 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_49 = ((s_49 <= 2139095040) ? h_50 : 32256);
                    float _abs_564;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_564) : "f"(S[43]));
                    unsigned int s_50 = __as_u32(mul_f32_rz_noftz(_abs_564, __uint_as_float(125829120)));
                    unsigned int _min_565 = ((s_50) < (260046848) ? (s_50) : (260046848));
                    unsigned int h_51 = _min_565 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_50 = ((s_50 <= 2139095040) ? h_51 : 32256);
                    uint32_t mags_24;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_24) : "r"(mag_49), "r"(mag_50));
                    uint32_t signs_24;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_24) : "r"(__as_u32(S[42])), "r"(__as_u32(S[43])));
                    unsigned int word_24 = signs_24 & 2147516416u | mags_24;
                    h_words[1] = word_24;
#else
                    uint32_t _cvt_rs_x2_f32_82;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_82) : "f"(S2[20].y), "f"(S2[20].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_82;
                    uint32_t _cvt_rs_x2_f32_83;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_83) : "f"(S2[21].y), "f"(S2[21].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_83;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_566;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_566) : "f"(S[56]));
                    unsigned int s_51 = __as_u32(mul_f32_rz_noftz(_abs_566, __uint_as_float(125829120)));
                    unsigned int _min_567 = ((s_51) < (260046848) ? (s_51) : (260046848));
                    unsigned int h_52 = _min_567 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_51 = ((s_51 <= 2139095040) ? h_52 : 32256);
                    float _abs_568;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_568) : "f"(S[57]));
                    unsigned int s_52 = __as_u32(mul_f32_rz_noftz(_abs_568, __uint_as_float(125829120)));
                    unsigned int _min_569 = ((s_52) < (260046848) ? (s_52) : (260046848));
                    unsigned int h_53 = _min_569 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_52 = ((s_52 <= 2139095040) ? h_53 : 32256);
                    uint32_t mags_25;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_25) : "r"(mag_51), "r"(mag_52));
                    uint32_t signs_25;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_25) : "r"(__as_u32(S[56])), "r"(__as_u32(S[57])));
                    unsigned int word_25 = signs_25 & 2147516416u | mags_25;
                    h_words[0] = word_25;
                    float _abs_570;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_570) : "f"(S[58]));
                    unsigned int s_53 = __as_u32(mul_f32_rz_noftz(_abs_570, __uint_as_float(125829120)));
                    unsigned int _min_571 = ((s_53) < (260046848) ? (s_53) : (260046848));
                    unsigned int h_54 = _min_571 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_53 = ((s_53 <= 2139095040) ? h_54 : 32256);
                    float _abs_572;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_572) : "f"(S[59]));
                    unsigned int s_54 = __as_u32(mul_f32_rz_noftz(_abs_572, __uint_as_float(125829120)));
                    unsigned int _min_573 = ((s_54) < (260046848) ? (s_54) : (260046848));
                    unsigned int h_55 = _min_573 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_54 = ((s_54 <= 2139095040) ? h_55 : 32256);
                    uint32_t mags_26;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_26) : "r"(mag_53), "r"(mag_54));
                    uint32_t signs_26;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_26) : "r"(__as_u32(S[58])), "r"(__as_u32(S[59])));
                    unsigned int word_26 = signs_26 & 2147516416u | mags_26;
                    h_words[1] = word_26;
#else
                    uint32_t _cvt_rs_x2_f32_84;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_84) : "f"(S2[28].y), "f"(S2[28].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_84;
                    uint32_t _cvt_rs_x2_f32_85;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_85) : "f"(S2[29].y), "f"(S2[29].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_85;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 7168));
#if __CUDA_ARCH__ == 900
                    float _abs_574;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_574) : "f"(S[44]));
                    unsigned int s_55 = __as_u32(mul_f32_rz_noftz(_abs_574, __uint_as_float(125829120)));
                    unsigned int _min_575 = ((s_55) < (260046848) ? (s_55) : (260046848));
                    unsigned int h_56 = _min_575 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_55 = ((s_55 <= 2139095040) ? h_56 : 32256);
                    float _abs_576;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_576) : "f"(S[45]));
                    unsigned int s_56 = __as_u32(mul_f32_rz_noftz(_abs_576, __uint_as_float(125829120)));
                    unsigned int _min_577 = ((s_56) < (260046848) ? (s_56) : (260046848));
                    unsigned int h_57 = _min_577 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_56 = ((s_56 <= 2139095040) ? h_57 : 32256);
                    uint32_t mags_27;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_27) : "r"(mag_55), "r"(mag_56));
                    uint32_t signs_27;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_27) : "r"(__as_u32(S[44])), "r"(__as_u32(S[45])));
                    unsigned int word_27 = signs_27 & 2147516416u | mags_27;
                    h_words[0] = word_27;
                    float _abs_578;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_578) : "f"(S[46]));
                    unsigned int s_57 = __as_u32(mul_f32_rz_noftz(_abs_578, __uint_as_float(125829120)));
                    unsigned int _min_579 = ((s_57) < (260046848) ? (s_57) : (260046848));
                    unsigned int h_58 = _min_579 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_57 = ((s_57 <= 2139095040) ? h_58 : 32256);
                    float _abs_580;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_580) : "f"(S[47]));
                    unsigned int s_58 = __as_u32(mul_f32_rz_noftz(_abs_580, __uint_as_float(125829120)));
                    unsigned int _min_581 = ((s_58) < (260046848) ? (s_58) : (260046848));
                    unsigned int h_59 = _min_581 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_58 = ((s_58 <= 2139095040) ? h_59 : 32256);
                    uint32_t mags_28;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_28) : "r"(mag_57), "r"(mag_58));
                    uint32_t signs_28;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_28) : "r"(__as_u32(S[46])), "r"(__as_u32(S[47])));
                    unsigned int word_28 = signs_28 & 2147516416u | mags_28;
                    h_words[1] = word_28;
#else
                    uint32_t _cvt_rs_x2_f32_86;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_86) : "f"(S2[22].y), "f"(S2[22].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_86;
                    uint32_t _cvt_rs_x2_f32_87;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_87) : "f"(S2[23].y), "f"(S2[23].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_87;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 48ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_582;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_582) : "f"(S[60]));
                    unsigned int s_59 = __as_u32(mul_f32_rz_noftz(_abs_582, __uint_as_float(125829120)));
                    unsigned int _min_583 = ((s_59) < (260046848) ? (s_59) : (260046848));
                    unsigned int h_60 = _min_583 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_59 = ((s_59 <= 2139095040) ? h_60 : 32256);
                    float _abs_584;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_584) : "f"(S[61]));
                    unsigned int s_60 = __as_u32(mul_f32_rz_noftz(_abs_584, __uint_as_float(125829120)));
                    unsigned int _min_585 = ((s_60) < (260046848) ? (s_60) : (260046848));
                    unsigned int h_61 = _min_585 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_60 = ((s_60 <= 2139095040) ? h_61 : 32256);
                    uint32_t mags_29;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_29) : "r"(mag_59), "r"(mag_60));
                    uint32_t signs_29;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_29) : "r"(__as_u32(S[60])), "r"(__as_u32(S[61])));
                    unsigned int word_29 = signs_29 & 2147516416u | mags_29;
                    h_words[0] = word_29;
                    float _abs_586;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_586) : "f"(S[62]));
                    unsigned int s_61 = __as_u32(mul_f32_rz_noftz(_abs_586, __uint_as_float(125829120)));
                    unsigned int _min_587 = ((s_61) < (260046848) ? (s_61) : (260046848));
                    unsigned int h_62 = _min_587 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_61 = ((s_61 <= 2139095040) ? h_62 : 32256);
                    float _abs_588;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_588) : "f"(S[63]));
                    unsigned int s_62 = __as_u32(mul_f32_rz_noftz(_abs_588, __uint_as_float(125829120)));
                    unsigned int _min_589 = ((s_62) < (260046848) ? (s_62) : (260046848));
                    unsigned int h_63 = _min_589 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_62 = ((s_62 <= 2139095040) ? h_63 : 32256);
                    uint32_t mags_30;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_30) : "r"(mag_61), "r"(mag_62));
                    uint32_t signs_30;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_30) : "r"(__as_u32(S[62])), "r"(__as_u32(S[63])));
                    unsigned int word_30 = signs_30 & 2147516416u | mags_30;
                    h_words[1] = word_30;
#else
                    uint32_t _cvt_rs_x2_f32_88;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_88) : "f"(S2[30].y), "f"(S2[30].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_88;
                    uint32_t _cvt_rs_x2_f32_89;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_89) : "f"(S2[31].y), "f"(S2[31].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_89;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 112ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 8192));
#if __CUDA_ARCH__ == 900
                    float _abs_590;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_590) : "f"(S[64]));
                    unsigned int s_63 = __as_u32(mul_f32_rz_noftz(_abs_590, __uint_as_float(125829120)));
                    unsigned int _min_591 = ((s_63) < (260046848) ? (s_63) : (260046848));
                    unsigned int h_64 = _min_591 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_63 = ((s_63 <= 2139095040) ? h_64 : 32256);
                    float _abs_592;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_592) : "f"(S[65]));
                    unsigned int s_64 = __as_u32(mul_f32_rz_noftz(_abs_592, __uint_as_float(125829120)));
                    unsigned int _min_593 = ((s_64) < (260046848) ? (s_64) : (260046848));
                    unsigned int h_65 = _min_593 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_64 = ((s_64 <= 2139095040) ? h_65 : 32256);
                    uint32_t mags_31;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_31) : "r"(mag_63), "r"(mag_64));
                    uint32_t signs_31;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_31) : "r"(__as_u32(S[64])), "r"(__as_u32(S[65])));
                    unsigned int word_31 = signs_31 & 2147516416u | mags_31;
                    h_words[0] = word_31;
                    float _abs_594;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_594) : "f"(S[66]));
                    unsigned int s_65 = __as_u32(mul_f32_rz_noftz(_abs_594, __uint_as_float(125829120)));
                    unsigned int _min_595 = ((s_65) < (260046848) ? (s_65) : (260046848));
                    unsigned int h_66 = _min_595 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_65 = ((s_65 <= 2139095040) ? h_66 : 32256);
                    float _abs_596;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_596) : "f"(S[67]));
                    unsigned int s_66 = __as_u32(mul_f32_rz_noftz(_abs_596, __uint_as_float(125829120)));
                    unsigned int _min_597 = ((s_66) < (260046848) ? (s_66) : (260046848));
                    unsigned int h_67 = _min_597 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_66 = ((s_66 <= 2139095040) ? h_67 : 32256);
                    uint32_t mags_32;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_32) : "r"(mag_65), "r"(mag_66));
                    uint32_t signs_32;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_32) : "r"(__as_u32(S[66])), "r"(__as_u32(S[67])));
                    unsigned int word_32 = signs_32 & 2147516416u | mags_32;
                    h_words[1] = word_32;
#else
                    uint32_t _cvt_rs_x2_f32_90;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_90) : "f"(S2[32].y), "f"(S2[32].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_90;
                    uint32_t _cvt_rs_x2_f32_91;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_91) : "f"(S2[33].y), "f"(S2[33].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_91;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_598;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_598) : "f"(S[80]));
                    unsigned int s_67 = __as_u32(mul_f32_rz_noftz(_abs_598, __uint_as_float(125829120)));
                    unsigned int _min_599 = ((s_67) < (260046848) ? (s_67) : (260046848));
                    unsigned int h_68 = _min_599 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_67 = ((s_67 <= 2139095040) ? h_68 : 32256);
                    float _abs_600;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_600) : "f"(S[81]));
                    unsigned int s_68 = __as_u32(mul_f32_rz_noftz(_abs_600, __uint_as_float(125829120)));
                    unsigned int _min_601 = ((s_68) < (260046848) ? (s_68) : (260046848));
                    unsigned int h_69 = _min_601 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_68 = ((s_68 <= 2139095040) ? h_69 : 32256);
                    uint32_t mags_33;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_33) : "r"(mag_67), "r"(mag_68));
                    uint32_t signs_33;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_33) : "r"(__as_u32(S[80])), "r"(__as_u32(S[81])));
                    unsigned int word_33 = signs_33 & 2147516416u | mags_33;
                    h_words[0] = word_33;
                    float _abs_602;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_602) : "f"(S[82]));
                    unsigned int s_69 = __as_u32(mul_f32_rz_noftz(_abs_602, __uint_as_float(125829120)));
                    unsigned int _min_603 = ((s_69) < (260046848) ? (s_69) : (260046848));
                    unsigned int h_70 = _min_603 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_69 = ((s_69 <= 2139095040) ? h_70 : 32256);
                    float _abs_604;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_604) : "f"(S[83]));
                    unsigned int s_70 = __as_u32(mul_f32_rz_noftz(_abs_604, __uint_as_float(125829120)));
                    unsigned int _min_605 = ((s_70) < (260046848) ? (s_70) : (260046848));
                    unsigned int h_71 = _min_605 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_70 = ((s_70 <= 2139095040) ? h_71 : 32256);
                    uint32_t mags_34;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_34) : "r"(mag_69), "r"(mag_70));
                    uint32_t signs_34;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_34) : "r"(__as_u32(S[82])), "r"(__as_u32(S[83])));
                    unsigned int word_34 = signs_34 & 2147516416u | mags_34;
                    h_words[1] = word_34;
#else
                    uint32_t _cvt_rs_x2_f32_92;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_92) : "f"(S2[40].y), "f"(S2[40].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_92;
                    uint32_t _cvt_rs_x2_f32_93;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_93) : "f"(S2[41].y), "f"(S2[41].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_93;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 9216));
#if __CUDA_ARCH__ == 900
                    float _abs_606;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_606) : "f"(S[68]));
                    unsigned int s_71 = __as_u32(mul_f32_rz_noftz(_abs_606, __uint_as_float(125829120)));
                    unsigned int _min_607 = ((s_71) < (260046848) ? (s_71) : (260046848));
                    unsigned int h_72 = _min_607 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_71 = ((s_71 <= 2139095040) ? h_72 : 32256);
                    float _abs_608;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_608) : "f"(S[69]));
                    unsigned int s_72 = __as_u32(mul_f32_rz_noftz(_abs_608, __uint_as_float(125829120)));
                    unsigned int _min_609 = ((s_72) < (260046848) ? (s_72) : (260046848));
                    unsigned int h_73 = _min_609 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_72 = ((s_72 <= 2139095040) ? h_73 : 32256);
                    uint32_t mags_35;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_35) : "r"(mag_71), "r"(mag_72));
                    uint32_t signs_35;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_35) : "r"(__as_u32(S[68])), "r"(__as_u32(S[69])));
                    unsigned int word_35 = signs_35 & 2147516416u | mags_35;
                    h_words[0] = word_35;
                    float _abs_610;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_610) : "f"(S[70]));
                    unsigned int s_73 = __as_u32(mul_f32_rz_noftz(_abs_610, __uint_as_float(125829120)));
                    unsigned int _min_611 = ((s_73) < (260046848) ? (s_73) : (260046848));
                    unsigned int h_74 = _min_611 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_73 = ((s_73 <= 2139095040) ? h_74 : 32256);
                    float _abs_612;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_612) : "f"(S[71]));
                    unsigned int s_74 = __as_u32(mul_f32_rz_noftz(_abs_612, __uint_as_float(125829120)));
                    unsigned int _min_613 = ((s_74) < (260046848) ? (s_74) : (260046848));
                    unsigned int h_75 = _min_613 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_74 = ((s_74 <= 2139095040) ? h_75 : 32256);
                    uint32_t mags_36;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_36) : "r"(mag_73), "r"(mag_74));
                    uint32_t signs_36;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_36) : "r"(__as_u32(S[70])), "r"(__as_u32(S[71])));
                    unsigned int word_36 = signs_36 & 2147516416u | mags_36;
                    h_words[1] = word_36;
#else
                    uint32_t _cvt_rs_x2_f32_94;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_94) : "f"(S2[34].y), "f"(S2[34].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_94;
                    uint32_t _cvt_rs_x2_f32_95;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_95) : "f"(S2[35].y), "f"(S2[35].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_95;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 16ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_614;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_614) : "f"(S[84]));
                    unsigned int s_75 = __as_u32(mul_f32_rz_noftz(_abs_614, __uint_as_float(125829120)));
                    unsigned int _min_615 = ((s_75) < (260046848) ? (s_75) : (260046848));
                    unsigned int h_76 = _min_615 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_75 = ((s_75 <= 2139095040) ? h_76 : 32256);
                    float _abs_616;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_616) : "f"(S[85]));
                    unsigned int s_76 = __as_u32(mul_f32_rz_noftz(_abs_616, __uint_as_float(125829120)));
                    unsigned int _min_617 = ((s_76) < (260046848) ? (s_76) : (260046848));
                    unsigned int h_77 = _min_617 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_76 = ((s_76 <= 2139095040) ? h_77 : 32256);
                    uint32_t mags_37;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_37) : "r"(mag_75), "r"(mag_76));
                    uint32_t signs_37;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_37) : "r"(__as_u32(S[84])), "r"(__as_u32(S[85])));
                    unsigned int word_37 = signs_37 & 2147516416u | mags_37;
                    h_words[0] = word_37;
                    float _abs_618;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_618) : "f"(S[86]));
                    unsigned int s_77 = __as_u32(mul_f32_rz_noftz(_abs_618, __uint_as_float(125829120)));
                    unsigned int _min_619 = ((s_77) < (260046848) ? (s_77) : (260046848));
                    unsigned int h_78 = _min_619 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_77 = ((s_77 <= 2139095040) ? h_78 : 32256);
                    float _abs_620;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_620) : "f"(S[87]));
                    unsigned int s_78 = __as_u32(mul_f32_rz_noftz(_abs_620, __uint_as_float(125829120)));
                    unsigned int _min_621 = ((s_78) < (260046848) ? (s_78) : (260046848));
                    unsigned int h_79 = _min_621 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_78 = ((s_78 <= 2139095040) ? h_79 : 32256);
                    uint32_t mags_38;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_38) : "r"(mag_77), "r"(mag_78));
                    uint32_t signs_38;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_38) : "r"(__as_u32(S[86])), "r"(__as_u32(S[87])));
                    unsigned int word_38 = signs_38 & 2147516416u | mags_38;
                    h_words[1] = word_38;
#else
                    uint32_t _cvt_rs_x2_f32_96;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_96) : "f"(S2[42].y), "f"(S2[42].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_96;
                    uint32_t _cvt_rs_x2_f32_97;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_97) : "f"(S2[43].y), "f"(S2[43].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_97;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 80ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 10240));
#if __CUDA_ARCH__ == 900
                    float _abs_622;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_622) : "f"(S[72]));
                    unsigned int s_79 = __as_u32(mul_f32_rz_noftz(_abs_622, __uint_as_float(125829120)));
                    unsigned int _min_623 = ((s_79) < (260046848) ? (s_79) : (260046848));
                    unsigned int h_80 = _min_623 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_79 = ((s_79 <= 2139095040) ? h_80 : 32256);
                    float _abs_624;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_624) : "f"(S[73]));
                    unsigned int s_80 = __as_u32(mul_f32_rz_noftz(_abs_624, __uint_as_float(125829120)));
                    unsigned int _min_625 = ((s_80) < (260046848) ? (s_80) : (260046848));
                    unsigned int h_81 = _min_625 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_80 = ((s_80 <= 2139095040) ? h_81 : 32256);
                    uint32_t mags_39;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_39) : "r"(mag_79), "r"(mag_80));
                    uint32_t signs_39;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_39) : "r"(__as_u32(S[72])), "r"(__as_u32(S[73])));
                    unsigned int word_39 = signs_39 & 2147516416u | mags_39;
                    h_words[0] = word_39;
                    float _abs_626;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_626) : "f"(S[74]));
                    unsigned int s_81 = __as_u32(mul_f32_rz_noftz(_abs_626, __uint_as_float(125829120)));
                    unsigned int _min_627 = ((s_81) < (260046848) ? (s_81) : (260046848));
                    unsigned int h_82 = _min_627 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_81 = ((s_81 <= 2139095040) ? h_82 : 32256);
                    float _abs_628;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_628) : "f"(S[75]));
                    unsigned int s_82 = __as_u32(mul_f32_rz_noftz(_abs_628, __uint_as_float(125829120)));
                    unsigned int _min_629 = ((s_82) < (260046848) ? (s_82) : (260046848));
                    unsigned int h_83 = _min_629 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_82 = ((s_82 <= 2139095040) ? h_83 : 32256);
                    uint32_t mags_40;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_40) : "r"(mag_81), "r"(mag_82));
                    uint32_t signs_40;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_40) : "r"(__as_u32(S[74])), "r"(__as_u32(S[75])));
                    unsigned int word_40 = signs_40 & 2147516416u | mags_40;
                    h_words[1] = word_40;
#else
                    uint32_t _cvt_rs_x2_f32_98;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_98) : "f"(S2[36].y), "f"(S2[36].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_98;
                    uint32_t _cvt_rs_x2_f32_99;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_99) : "f"(S2[37].y), "f"(S2[37].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_99;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_630;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_630) : "f"(S[88]));
                    unsigned int s_83 = __as_u32(mul_f32_rz_noftz(_abs_630, __uint_as_float(125829120)));
                    unsigned int _min_631 = ((s_83) < (260046848) ? (s_83) : (260046848));
                    unsigned int h_84 = _min_631 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_83 = ((s_83 <= 2139095040) ? h_84 : 32256);
                    float _abs_632;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_632) : "f"(S[89]));
                    unsigned int s_84 = __as_u32(mul_f32_rz_noftz(_abs_632, __uint_as_float(125829120)));
                    unsigned int _min_633 = ((s_84) < (260046848) ? (s_84) : (260046848));
                    unsigned int h_85 = _min_633 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_84 = ((s_84 <= 2139095040) ? h_85 : 32256);
                    uint32_t mags_41;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_41) : "r"(mag_83), "r"(mag_84));
                    uint32_t signs_41;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_41) : "r"(__as_u32(S[88])), "r"(__as_u32(S[89])));
                    unsigned int word_41 = signs_41 & 2147516416u | mags_41;
                    h_words[0] = word_41;
                    float _abs_634;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_634) : "f"(S[90]));
                    unsigned int s_85 = __as_u32(mul_f32_rz_noftz(_abs_634, __uint_as_float(125829120)));
                    unsigned int _min_635 = ((s_85) < (260046848) ? (s_85) : (260046848));
                    unsigned int h_86 = _min_635 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_85 = ((s_85 <= 2139095040) ? h_86 : 32256);
                    float _abs_636;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_636) : "f"(S[91]));
                    unsigned int s_86 = __as_u32(mul_f32_rz_noftz(_abs_636, __uint_as_float(125829120)));
                    unsigned int _min_637 = ((s_86) < (260046848) ? (s_86) : (260046848));
                    unsigned int h_87 = _min_637 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_86 = ((s_86 <= 2139095040) ? h_87 : 32256);
                    uint32_t mags_42;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_42) : "r"(mag_85), "r"(mag_86));
                    uint32_t signs_42;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_42) : "r"(__as_u32(S[90])), "r"(__as_u32(S[91])));
                    unsigned int word_42 = signs_42 & 2147516416u | mags_42;
                    h_words[1] = word_42;
#else
                    uint32_t _cvt_rs_x2_f32_100;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_100) : "f"(S2[44].y), "f"(S2[44].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_100;
                    uint32_t _cvt_rs_x2_f32_101;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_101) : "f"(S2[45].y), "f"(S2[45].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_101;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 11264));
#if __CUDA_ARCH__ == 900
                    float _abs_638;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_638) : "f"(S[76]));
                    unsigned int s_87 = __as_u32(mul_f32_rz_noftz(_abs_638, __uint_as_float(125829120)));
                    unsigned int _min_639 = ((s_87) < (260046848) ? (s_87) : (260046848));
                    unsigned int h_88 = _min_639 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_87 = ((s_87 <= 2139095040) ? h_88 : 32256);
                    float _abs_640;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_640) : "f"(S[77]));
                    unsigned int s_88 = __as_u32(mul_f32_rz_noftz(_abs_640, __uint_as_float(125829120)));
                    unsigned int _min_641 = ((s_88) < (260046848) ? (s_88) : (260046848));
                    unsigned int h_89 = _min_641 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_88 = ((s_88 <= 2139095040) ? h_89 : 32256);
                    uint32_t mags_43;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_43) : "r"(mag_87), "r"(mag_88));
                    uint32_t signs_43;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_43) : "r"(__as_u32(S[76])), "r"(__as_u32(S[77])));
                    unsigned int word_43 = signs_43 & 2147516416u | mags_43;
                    h_words[0] = word_43;
                    float _abs_642;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_642) : "f"(S[78]));
                    unsigned int s_89 = __as_u32(mul_f32_rz_noftz(_abs_642, __uint_as_float(125829120)));
                    unsigned int _min_643 = ((s_89) < (260046848) ? (s_89) : (260046848));
                    unsigned int h_90 = _min_643 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_89 = ((s_89 <= 2139095040) ? h_90 : 32256);
                    float _abs_644;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_644) : "f"(S[79]));
                    unsigned int s_90 = __as_u32(mul_f32_rz_noftz(_abs_644, __uint_as_float(125829120)));
                    unsigned int _min_645 = ((s_90) < (260046848) ? (s_90) : (260046848));
                    unsigned int h_91 = _min_645 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_90 = ((s_90 <= 2139095040) ? h_91 : 32256);
                    uint32_t mags_44;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_44) : "r"(mag_89), "r"(mag_90));
                    uint32_t signs_44;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_44) : "r"(__as_u32(S[78])), "r"(__as_u32(S[79])));
                    unsigned int word_44 = signs_44 & 2147516416u | mags_44;
                    h_words[1] = word_44;
#else
                    uint32_t _cvt_rs_x2_f32_102;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_102) : "f"(S2[38].y), "f"(S2[38].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_102;
                    uint32_t _cvt_rs_x2_f32_103;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_103) : "f"(S2[39].y), "f"(S2[39].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_103;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 48ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_646;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_646) : "f"(S[92]));
                    unsigned int s_91 = __as_u32(mul_f32_rz_noftz(_abs_646, __uint_as_float(125829120)));
                    unsigned int _min_647 = ((s_91) < (260046848) ? (s_91) : (260046848));
                    unsigned int h_92 = _min_647 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_91 = ((s_91 <= 2139095040) ? h_92 : 32256);
                    float _abs_648;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_648) : "f"(S[93]));
                    unsigned int s_92 = __as_u32(mul_f32_rz_noftz(_abs_648, __uint_as_float(125829120)));
                    unsigned int _min_649 = ((s_92) < (260046848) ? (s_92) : (260046848));
                    unsigned int h_93 = _min_649 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_92 = ((s_92 <= 2139095040) ? h_93 : 32256);
                    uint32_t mags_45;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_45) : "r"(mag_91), "r"(mag_92));
                    uint32_t signs_45;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_45) : "r"(__as_u32(S[92])), "r"(__as_u32(S[93])));
                    unsigned int word_45 = signs_45 & 2147516416u | mags_45;
                    h_words[0] = word_45;
                    float _abs_650;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_650) : "f"(S[94]));
                    unsigned int s_93 = __as_u32(mul_f32_rz_noftz(_abs_650, __uint_as_float(125829120)));
                    unsigned int _min_651 = ((s_93) < (260046848) ? (s_93) : (260046848));
                    unsigned int h_94 = _min_651 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_93 = ((s_93 <= 2139095040) ? h_94 : 32256);
                    float _abs_652;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_652) : "f"(S[95]));
                    unsigned int s_94 = __as_u32(mul_f32_rz_noftz(_abs_652, __uint_as_float(125829120)));
                    unsigned int _min_653 = ((s_94) < (260046848) ? (s_94) : (260046848));
                    unsigned int h_95 = _min_653 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_94 = ((s_94 <= 2139095040) ? h_95 : 32256);
                    uint32_t mags_46;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_46) : "r"(mag_93), "r"(mag_94));
                    uint32_t signs_46;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_46) : "r"(__as_u32(S[94])), "r"(__as_u32(S[95])));
                    unsigned int word_46 = signs_46 & 2147516416u | mags_46;
                    h_words[1] = word_46;
#else
                    uint32_t _cvt_rs_x2_f32_104;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_104) : "f"(S2[46].y), "f"(S2[46].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_104;
                    uint32_t _cvt_rs_x2_f32_105;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_105) : "f"(S2[47].y), "f"(S2[47].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_105;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 112ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 12288));
#if __CUDA_ARCH__ == 900
                    float _abs_654;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_654) : "f"(S[96]));
                    unsigned int s_95 = __as_u32(mul_f32_rz_noftz(_abs_654, __uint_as_float(125829120)));
                    unsigned int _min_655 = ((s_95) < (260046848) ? (s_95) : (260046848));
                    unsigned int h_96 = _min_655 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_95 = ((s_95 <= 2139095040) ? h_96 : 32256);
                    float _abs_656;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_656) : "f"(S[97]));
                    unsigned int s_96 = __as_u32(mul_f32_rz_noftz(_abs_656, __uint_as_float(125829120)));
                    unsigned int _min_657 = ((s_96) < (260046848) ? (s_96) : (260046848));
                    unsigned int h_97 = _min_657 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_96 = ((s_96 <= 2139095040) ? h_97 : 32256);
                    uint32_t mags_47;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_47) : "r"(mag_95), "r"(mag_96));
                    uint32_t signs_47;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_47) : "r"(__as_u32(S[96])), "r"(__as_u32(S[97])));
                    unsigned int word_47 = signs_47 & 2147516416u | mags_47;
                    h_words[0] = word_47;
                    float _abs_658;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_658) : "f"(S[98]));
                    unsigned int s_97 = __as_u32(mul_f32_rz_noftz(_abs_658, __uint_as_float(125829120)));
                    unsigned int _min_659 = ((s_97) < (260046848) ? (s_97) : (260046848));
                    unsigned int h_98 = _min_659 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_97 = ((s_97 <= 2139095040) ? h_98 : 32256);
                    float _abs_660;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_660) : "f"(S[99]));
                    unsigned int s_98 = __as_u32(mul_f32_rz_noftz(_abs_660, __uint_as_float(125829120)));
                    unsigned int _min_661 = ((s_98) < (260046848) ? (s_98) : (260046848));
                    unsigned int h_99 = _min_661 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_98 = ((s_98 <= 2139095040) ? h_99 : 32256);
                    uint32_t mags_48;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_48) : "r"(mag_97), "r"(mag_98));
                    uint32_t signs_48;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_48) : "r"(__as_u32(S[98])), "r"(__as_u32(S[99])));
                    unsigned int word_48 = signs_48 & 2147516416u | mags_48;
                    h_words[1] = word_48;
#else
                    uint32_t _cvt_rs_x2_f32_106;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_106) : "f"(S2[48].y), "f"(S2[48].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_106;
                    uint32_t _cvt_rs_x2_f32_107;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_107) : "f"(S2[49].y), "f"(S2[49].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_107;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_662;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_662) : "f"(S[112]));
                    unsigned int s_99 = __as_u32(mul_f32_rz_noftz(_abs_662, __uint_as_float(125829120)));
                    unsigned int _min_663 = ((s_99) < (260046848) ? (s_99) : (260046848));
                    unsigned int h_100 = _min_663 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_99 = ((s_99 <= 2139095040) ? h_100 : 32256);
                    float _abs_664;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_664) : "f"(S[113]));
                    unsigned int s_100 = __as_u32(mul_f32_rz_noftz(_abs_664, __uint_as_float(125829120)));
                    unsigned int _min_665 = ((s_100) < (260046848) ? (s_100) : (260046848));
                    unsigned int h_101 = _min_665 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_100 = ((s_100 <= 2139095040) ? h_101 : 32256);
                    uint32_t mags_49;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_49) : "r"(mag_99), "r"(mag_100));
                    uint32_t signs_49;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_49) : "r"(__as_u32(S[112])), "r"(__as_u32(S[113])));
                    unsigned int word_49 = signs_49 & 2147516416u | mags_49;
                    h_words[0] = word_49;
                    float _abs_666;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_666) : "f"(S[114]));
                    unsigned int s_101 = __as_u32(mul_f32_rz_noftz(_abs_666, __uint_as_float(125829120)));
                    unsigned int _min_667 = ((s_101) < (260046848) ? (s_101) : (260046848));
                    unsigned int h_102 = _min_667 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_101 = ((s_101 <= 2139095040) ? h_102 : 32256);
                    float _abs_668;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_668) : "f"(S[115]));
                    unsigned int s_102 = __as_u32(mul_f32_rz_noftz(_abs_668, __uint_as_float(125829120)));
                    unsigned int _min_669 = ((s_102) < (260046848) ? (s_102) : (260046848));
                    unsigned int h_103 = _min_669 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_102 = ((s_102 <= 2139095040) ? h_103 : 32256);
                    uint32_t mags_50;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_50) : "r"(mag_101), "r"(mag_102));
                    uint32_t signs_50;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_50) : "r"(__as_u32(S[114])), "r"(__as_u32(S[115])));
                    unsigned int word_50 = signs_50 & 2147516416u | mags_50;
                    h_words[1] = word_50;
#else
                    uint32_t _cvt_rs_x2_f32_108;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_108) : "f"(S2[56].y), "f"(S2[56].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_108;
                    uint32_t _cvt_rs_x2_f32_109;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_109) : "f"(S2[57].y), "f"(S2[57].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_109;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 13312));
#if __CUDA_ARCH__ == 900
                    float _abs_670;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_670) : "f"(S[100]));
                    unsigned int s_103 = __as_u32(mul_f32_rz_noftz(_abs_670, __uint_as_float(125829120)));
                    unsigned int _min_671 = ((s_103) < (260046848) ? (s_103) : (260046848));
                    unsigned int h_104 = _min_671 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_103 = ((s_103 <= 2139095040) ? h_104 : 32256);
                    float _abs_672;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_672) : "f"(S[101]));
                    unsigned int s_104 = __as_u32(mul_f32_rz_noftz(_abs_672, __uint_as_float(125829120)));
                    unsigned int _min_673 = ((s_104) < (260046848) ? (s_104) : (260046848));
                    unsigned int h_105 = _min_673 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_104 = ((s_104 <= 2139095040) ? h_105 : 32256);
                    uint32_t mags_51;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_51) : "r"(mag_103), "r"(mag_104));
                    uint32_t signs_51;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_51) : "r"(__as_u32(S[100])), "r"(__as_u32(S[101])));
                    unsigned int word_51 = signs_51 & 2147516416u | mags_51;
                    h_words[0] = word_51;
                    float _abs_674;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_674) : "f"(S[102]));
                    unsigned int s_105 = __as_u32(mul_f32_rz_noftz(_abs_674, __uint_as_float(125829120)));
                    unsigned int _min_675 = ((s_105) < (260046848) ? (s_105) : (260046848));
                    unsigned int h_106 = _min_675 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_105 = ((s_105 <= 2139095040) ? h_106 : 32256);
                    float _abs_676;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_676) : "f"(S[103]));
                    unsigned int s_106 = __as_u32(mul_f32_rz_noftz(_abs_676, __uint_as_float(125829120)));
                    unsigned int _min_677 = ((s_106) < (260046848) ? (s_106) : (260046848));
                    unsigned int h_107 = _min_677 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_106 = ((s_106 <= 2139095040) ? h_107 : 32256);
                    uint32_t mags_52;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_52) : "r"(mag_105), "r"(mag_106));
                    uint32_t signs_52;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_52) : "r"(__as_u32(S[102])), "r"(__as_u32(S[103])));
                    unsigned int word_52 = signs_52 & 2147516416u | mags_52;
                    h_words[1] = word_52;
#else
                    uint32_t _cvt_rs_x2_f32_110;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_110) : "f"(S2[50].y), "f"(S2[50].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_110;
                    uint32_t _cvt_rs_x2_f32_111;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_111) : "f"(S2[51].y), "f"(S2[51].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_111;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 16ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_678;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_678) : "f"(S[116]));
                    unsigned int s_107 = __as_u32(mul_f32_rz_noftz(_abs_678, __uint_as_float(125829120)));
                    unsigned int _min_679 = ((s_107) < (260046848) ? (s_107) : (260046848));
                    unsigned int h_108 = _min_679 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_107 = ((s_107 <= 2139095040) ? h_108 : 32256);
                    float _abs_680;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_680) : "f"(S[117]));
                    unsigned int s_108 = __as_u32(mul_f32_rz_noftz(_abs_680, __uint_as_float(125829120)));
                    unsigned int _min_681 = ((s_108) < (260046848) ? (s_108) : (260046848));
                    unsigned int h_109 = _min_681 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_108 = ((s_108 <= 2139095040) ? h_109 : 32256);
                    uint32_t mags_53;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_53) : "r"(mag_107), "r"(mag_108));
                    uint32_t signs_53;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_53) : "r"(__as_u32(S[116])), "r"(__as_u32(S[117])));
                    unsigned int word_53 = signs_53 & 2147516416u | mags_53;
                    h_words[0] = word_53;
                    float _abs_682;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_682) : "f"(S[118]));
                    unsigned int s_109 = __as_u32(mul_f32_rz_noftz(_abs_682, __uint_as_float(125829120)));
                    unsigned int _min_683 = ((s_109) < (260046848) ? (s_109) : (260046848));
                    unsigned int h_110 = _min_683 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_109 = ((s_109 <= 2139095040) ? h_110 : 32256);
                    float _abs_684;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_684) : "f"(S[119]));
                    unsigned int s_110 = __as_u32(mul_f32_rz_noftz(_abs_684, __uint_as_float(125829120)));
                    unsigned int _min_685 = ((s_110) < (260046848) ? (s_110) : (260046848));
                    unsigned int h_111 = _min_685 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_110 = ((s_110 <= 2139095040) ? h_111 : 32256);
                    uint32_t mags_54;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_54) : "r"(mag_109), "r"(mag_110));
                    uint32_t signs_54;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_54) : "r"(__as_u32(S[118])), "r"(__as_u32(S[119])));
                    unsigned int word_54 = signs_54 & 2147516416u | mags_54;
                    h_words[1] = word_54;
#else
                    uint32_t _cvt_rs_x2_f32_112;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_112) : "f"(S2[58].y), "f"(S2[58].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_112;
                    uint32_t _cvt_rs_x2_f32_113;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_113) : "f"(S2[59].y), "f"(S2[59].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_113;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 80ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 14336));
#if __CUDA_ARCH__ == 900
                    float _abs_686;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_686) : "f"(S[104]));
                    unsigned int s_111 = __as_u32(mul_f32_rz_noftz(_abs_686, __uint_as_float(125829120)));
                    unsigned int _min_687 = ((s_111) < (260046848) ? (s_111) : (260046848));
                    unsigned int h_112 = _min_687 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_111 = ((s_111 <= 2139095040) ? h_112 : 32256);
                    float _abs_688;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_688) : "f"(S[105]));
                    unsigned int s_112 = __as_u32(mul_f32_rz_noftz(_abs_688, __uint_as_float(125829120)));
                    unsigned int _min_689 = ((s_112) < (260046848) ? (s_112) : (260046848));
                    unsigned int h_113 = _min_689 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_112 = ((s_112 <= 2139095040) ? h_113 : 32256);
                    uint32_t mags_55;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_55) : "r"(mag_111), "r"(mag_112));
                    uint32_t signs_55;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_55) : "r"(__as_u32(S[104])), "r"(__as_u32(S[105])));
                    unsigned int word_55 = signs_55 & 2147516416u | mags_55;
                    h_words[0] = word_55;
                    float _abs_690;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_690) : "f"(S[106]));
                    unsigned int s_113 = __as_u32(mul_f32_rz_noftz(_abs_690, __uint_as_float(125829120)));
                    unsigned int _min_691 = ((s_113) < (260046848) ? (s_113) : (260046848));
                    unsigned int h_114 = _min_691 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_113 = ((s_113 <= 2139095040) ? h_114 : 32256);
                    float _abs_692;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_692) : "f"(S[107]));
                    unsigned int s_114 = __as_u32(mul_f32_rz_noftz(_abs_692, __uint_as_float(125829120)));
                    unsigned int _min_693 = ((s_114) < (260046848) ? (s_114) : (260046848));
                    unsigned int h_115 = _min_693 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_114 = ((s_114 <= 2139095040) ? h_115 : 32256);
                    uint32_t mags_56;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_56) : "r"(mag_113), "r"(mag_114));
                    uint32_t signs_56;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_56) : "r"(__as_u32(S[106])), "r"(__as_u32(S[107])));
                    unsigned int word_56 = signs_56 & 2147516416u | mags_56;
                    h_words[1] = word_56;
#else
                    uint32_t _cvt_rs_x2_f32_114;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_114) : "f"(S2[52].y), "f"(S2[52].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_114;
                    uint32_t _cvt_rs_x2_f32_115;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_115) : "f"(S2[53].y), "f"(S2[53].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_115;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_694;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_694) : "f"(S[120]));
                    unsigned int s_115 = __as_u32(mul_f32_rz_noftz(_abs_694, __uint_as_float(125829120)));
                    unsigned int _min_695 = ((s_115) < (260046848) ? (s_115) : (260046848));
                    unsigned int h_116 = _min_695 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_115 = ((s_115 <= 2139095040) ? h_116 : 32256);
                    float _abs_696;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_696) : "f"(S[121]));
                    unsigned int s_116 = __as_u32(mul_f32_rz_noftz(_abs_696, __uint_as_float(125829120)));
                    unsigned int _min_697 = ((s_116) < (260046848) ? (s_116) : (260046848));
                    unsigned int h_117 = _min_697 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_116 = ((s_116 <= 2139095040) ? h_117 : 32256);
                    uint32_t mags_57;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_57) : "r"(mag_115), "r"(mag_116));
                    uint32_t signs_57;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_57) : "r"(__as_u32(S[120])), "r"(__as_u32(S[121])));
                    unsigned int word_57 = signs_57 & 2147516416u | mags_57;
                    h_words[0] = word_57;
                    float _abs_698;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_698) : "f"(S[122]));
                    unsigned int s_117 = __as_u32(mul_f32_rz_noftz(_abs_698, __uint_as_float(125829120)));
                    unsigned int _min_699 = ((s_117) < (260046848) ? (s_117) : (260046848));
                    unsigned int h_118 = _min_699 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_117 = ((s_117 <= 2139095040) ? h_118 : 32256);
                    float _abs_700;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_700) : "f"(S[123]));
                    unsigned int s_118 = __as_u32(mul_f32_rz_noftz(_abs_700, __uint_as_float(125829120)));
                    unsigned int _min_701 = ((s_118) < (260046848) ? (s_118) : (260046848));
                    unsigned int h_119 = _min_701 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_118 = ((s_118 <= 2139095040) ? h_119 : 32256);
                    uint32_t mags_58;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_58) : "r"(mag_117), "r"(mag_118));
                    uint32_t signs_58;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_58) : "r"(__as_u32(S[122])), "r"(__as_u32(S[123])));
                    unsigned int word_58 = signs_58 & 2147516416u | mags_58;
                    h_words[1] = word_58;
#else
                    uint32_t _cvt_rs_x2_f32_116;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_116) : "f"(S2[60].y), "f"(S2[60].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_116;
                    uint32_t _cvt_rs_x2_f32_117;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_117) : "f"(S2[61].y), "f"(S2[61].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_117;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 15360));
#if __CUDA_ARCH__ == 900
                    float _abs_702;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_702) : "f"(S[108]));
                    unsigned int s_119 = __as_u32(mul_f32_rz_noftz(_abs_702, __uint_as_float(125829120)));
                    unsigned int _min_703 = ((s_119) < (260046848) ? (s_119) : (260046848));
                    unsigned int h_120 = _min_703 + (h_noise[0] & 8191) >> 13;
                    unsigned int mag_119 = ((s_119 <= 2139095040) ? h_120 : 32256);
                    float _abs_704;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_704) : "f"(S[109]));
                    unsigned int s_120 = __as_u32(mul_f32_rz_noftz(_abs_704, __uint_as_float(125829120)));
                    unsigned int _min_705 = ((s_120) < (260046848) ? (s_120) : (260046848));
                    unsigned int h_121 = _min_705 + ((h_noise[0] & 536805376) >> 16) >> 13;
                    unsigned int mag_120 = ((s_120 <= 2139095040) ? h_121 : 32256);
                    uint32_t mags_59;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_59) : "r"(mag_119), "r"(mag_120));
                    uint32_t signs_59;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_59) : "r"(__as_u32(S[108])), "r"(__as_u32(S[109])));
                    unsigned int word_59 = signs_59 & 2147516416u | mags_59;
                    h_words[0] = word_59;
                    float _abs_706;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_706) : "f"(S[110]));
                    unsigned int s_121 = __as_u32(mul_f32_rz_noftz(_abs_706, __uint_as_float(125829120)));
                    unsigned int _min_707 = ((s_121) < (260046848) ? (s_121) : (260046848));
                    unsigned int h_122 = _min_707 + (h_noise[1] & 8191) >> 13;
                    unsigned int mag_121 = ((s_121 <= 2139095040) ? h_122 : 32256);
                    float _abs_708;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_708) : "f"(S[111]));
                    unsigned int s_122 = __as_u32(mul_f32_rz_noftz(_abs_708, __uint_as_float(125829120)));
                    unsigned int _min_709 = ((s_122) < (260046848) ? (s_122) : (260046848));
                    unsigned int h_123 = _min_709 + ((h_noise[1] & 536805376) >> 16) >> 13;
                    unsigned int mag_122 = ((s_122 <= 2139095040) ? h_123 : 32256);
                    uint32_t mags_60;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_60) : "r"(mag_121), "r"(mag_122));
                    uint32_t signs_60;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_60) : "r"(__as_u32(S[110])), "r"(__as_u32(S[111])));
                    unsigned int word_60 = signs_60 & 2147516416u | mags_60;
                    h_words[1] = word_60;
#else
                    uint32_t _cvt_rs_x2_f32_118;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_118) : "f"(S2[54].y), "f"(S2[54].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_118;
                    uint32_t _cvt_rs_x2_f32_119;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_119) : "f"(S2[55].y), "f"(S2[55].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_119;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 48ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    float _abs_710;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_710) : "f"(S[124]));
                    unsigned int s_123 = __as_u32(mul_f32_rz_noftz(_abs_710, __uint_as_float(125829120)));
                    unsigned int _min_711 = ((s_123) < (260046848) ? (s_123) : (260046848));
                    unsigned int h_124 = _min_711 + (h_noise[2] & 8191) >> 13;
                    unsigned int mag_123 = ((s_123 <= 2139095040) ? h_124 : 32256);
                    float _abs_712;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_712) : "f"(S[125]));
                    unsigned int s_124 = __as_u32(mul_f32_rz_noftz(_abs_712, __uint_as_float(125829120)));
                    unsigned int _min_713 = ((s_124) < (260046848) ? (s_124) : (260046848));
                    unsigned int h_125 = _min_713 + ((h_noise[2] & 536805376) >> 16) >> 13;
                    unsigned int mag_124 = ((s_124 <= 2139095040) ? h_125 : 32256);
                    uint32_t mags_61;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_61) : "r"(mag_123), "r"(mag_124));
                    uint32_t signs_61;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_61) : "r"(__as_u32(S[124])), "r"(__as_u32(S[125])));
                    unsigned int word_61 = signs_61 & 2147516416u | mags_61;
                    h_words[0] = word_61;
                    float _abs_714;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_714) : "f"(S[126]));
                    unsigned int s_125 = __as_u32(mul_f32_rz_noftz(_abs_714, __uint_as_float(125829120)));
                    unsigned int _min_715 = ((s_125) < (260046848) ? (s_125) : (260046848));
                    unsigned int h_126 = _min_715 + (h_noise[3] & 8191) >> 13;
                    unsigned int mag_125 = ((s_125 <= 2139095040) ? h_126 : 32256);
                    float _abs_716;
                    asm("abs.f32 %0, %1;" : "=f"(_abs_716) : "f"(S[127]));
                    unsigned int s_126 = __as_u32(mul_f32_rz_noftz(_abs_716, __uint_as_float(125829120)));
                    unsigned int _min_717 = ((s_126) < (260046848) ? (s_126) : (260046848));
                    unsigned int h_127 = _min_717 + ((h_noise[3] & 536805376) >> 16) >> 13;
                    unsigned int mag_126 = ((s_126 <= 2139095040) ? h_127 : 32256);
                    uint32_t mags_62;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_62) : "r"(mag_125), "r"(mag_126));
                    uint32_t signs_62;
                    asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_62) : "r"(__as_u32(S[126])), "r"(__as_u32(S[127])));
                    unsigned int word_62 = signs_62 & 2147516416u | mags_62;
                    h_words[1] = word_62;
#else
                    uint32_t _cvt_rs_x2_f32_120;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_120) : "f"(S2[62].y), "f"(S2[62].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_120;
                    uint32_t _cvt_rs_x2_f32_121;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_121) : "f"(S2[63].y), "f"(S2[63].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_121;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 112ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                }
                if (philox_rounds == 0) {
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_718;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_718) : "f"(S[1]), "f"(S[0]));
                    h_words[0] = _cvt_rn_f16x2_f32_718;
                    uint32_t _cvt_rn_f16x2_f32_719;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_719) : "f"(S[3]), "f"(S[2]));
                    h_words[1] = _cvt_rn_f16x2_f32_719;
#else
                    uint32_t _cvt_rn_f16x2_f32_122;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_122) : "f"(S2[0].y), "f"(S2[0].x));
                    h_words[0] = _cvt_rn_f16x2_f32_122;
                    uint32_t _cvt_rn_f16x2_f32_123;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_123) : "f"(S2[1].y), "f"(S2[1].x));
                    h_words[1] = _cvt_rn_f16x2_f32_123;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_720;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_720) : "f"(S[5]), "f"(S[4]));
                    h_words[0] = _cvt_rn_f16x2_f32_720;
                    uint32_t _cvt_rn_f16x2_f32_721;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_721) : "f"(S[7]), "f"(S[6]));
                    h_words[1] = _cvt_rn_f16x2_f32_721;
#else
                    uint32_t _cvt_rn_f16x2_f32_124;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_124) : "f"(S2[2].y), "f"(S2[2].x));
                    h_words[0] = _cvt_rn_f16x2_f32_124;
                    uint32_t _cvt_rn_f16x2_f32_125;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_125) : "f"(S2[3].y), "f"(S2[3].x));
                    h_words[1] = _cvt_rn_f16x2_f32_125;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 16ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_722;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_722) : "f"(S[9]), "f"(S[8]));
                    h_words[0] = _cvt_rn_f16x2_f32_722;
                    uint32_t _cvt_rn_f16x2_f32_723;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_723) : "f"(S[11]), "f"(S[10]));
                    h_words[1] = _cvt_rn_f16x2_f32_723;
#else
                    uint32_t _cvt_rn_f16x2_f32_126;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_126) : "f"(S2[4].y), "f"(S2[4].x));
                    h_words[0] = _cvt_rn_f16x2_f32_126;
                    uint32_t _cvt_rn_f16x2_f32_127;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_127) : "f"(S2[5].y), "f"(S2[5].x));
                    h_words[1] = _cvt_rn_f16x2_f32_127;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_724;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_724) : "f"(S[13]), "f"(S[12]));
                    h_words[0] = _cvt_rn_f16x2_f32_724;
                    uint32_t _cvt_rn_f16x2_f32_725;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_725) : "f"(S[15]), "f"(S[14]));
                    h_words[1] = _cvt_rn_f16x2_f32_725;
#else
                    uint32_t _cvt_rn_f16x2_f32_128;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_128) : "f"(S2[6].y), "f"(S2[6].x));
                    h_words[0] = _cvt_rn_f16x2_f32_128;
                    uint32_t _cvt_rn_f16x2_f32_129;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_129) : "f"(S2[7].y), "f"(S2[7].x));
                    h_words[1] = _cvt_rn_f16x2_f32_129;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 48ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_726;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_726) : "f"(S[17]), "f"(S[16]));
                    h_words[0] = _cvt_rn_f16x2_f32_726;
                    uint32_t _cvt_rn_f16x2_f32_727;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_727) : "f"(S[19]), "f"(S[18]));
                    h_words[1] = _cvt_rn_f16x2_f32_727;
#else
                    uint32_t _cvt_rn_f16x2_f32_130;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_130) : "f"(S2[8].y), "f"(S2[8].x));
                    h_words[0] = _cvt_rn_f16x2_f32_130;
                    uint32_t _cvt_rn_f16x2_f32_131;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_131) : "f"(S2[9].y), "f"(S2[9].x));
                    h_words[1] = _cvt_rn_f16x2_f32_131;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_728;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_728) : "f"(S[21]), "f"(S[20]));
                    h_words[0] = _cvt_rn_f16x2_f32_728;
                    uint32_t _cvt_rn_f16x2_f32_729;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_729) : "f"(S[23]), "f"(S[22]));
                    h_words[1] = _cvt_rn_f16x2_f32_729;
#else
                    uint32_t _cvt_rn_f16x2_f32_132;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_132) : "f"(S2[10].y), "f"(S2[10].x));
                    h_words[0] = _cvt_rn_f16x2_f32_132;
                    uint32_t _cvt_rn_f16x2_f32_133;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_133) : "f"(S2[11].y), "f"(S2[11].x));
                    h_words[1] = _cvt_rn_f16x2_f32_133;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 80ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_730;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_730) : "f"(S[25]), "f"(S[24]));
                    h_words[0] = _cvt_rn_f16x2_f32_730;
                    uint32_t _cvt_rn_f16x2_f32_731;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_731) : "f"(S[27]), "f"(S[26]));
                    h_words[1] = _cvt_rn_f16x2_f32_731;
#else
                    uint32_t _cvt_rn_f16x2_f32_134;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_134) : "f"(S2[12].y), "f"(S2[12].x));
                    h_words[0] = _cvt_rn_f16x2_f32_134;
                    uint32_t _cvt_rn_f16x2_f32_135;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_135) : "f"(S2[13].y), "f"(S2[13].x));
                    h_words[1] = _cvt_rn_f16x2_f32_135;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_732;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_732) : "f"(S[29]), "f"(S[28]));
                    h_words[0] = _cvt_rn_f16x2_f32_732;
                    uint32_t _cvt_rn_f16x2_f32_733;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_733) : "f"(S[31]), "f"(S[30]));
                    h_words[1] = _cvt_rn_f16x2_f32_733;
#else
                    uint32_t _cvt_rn_f16x2_f32_136;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_136) : "f"(S2[14].y), "f"(S2[14].x));
                    h_words[0] = _cvt_rn_f16x2_f32_136;
                    uint32_t _cvt_rn_f16x2_f32_137;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_137) : "f"(S2[15].y), "f"(S2[15].x));
                    h_words[1] = _cvt_rn_f16x2_f32_137;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 112ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_734;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_734) : "f"(S[33]), "f"(S[32]));
                    h_words[0] = _cvt_rn_f16x2_f32_734;
                    uint32_t _cvt_rn_f16x2_f32_735;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_735) : "f"(S[35]), "f"(S[34]));
                    h_words[1] = _cvt_rn_f16x2_f32_735;
#else
                    uint32_t _cvt_rn_f16x2_f32_138;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_138) : "f"(S2[16].y), "f"(S2[16].x));
                    h_words[0] = _cvt_rn_f16x2_f32_138;
                    uint32_t _cvt_rn_f16x2_f32_139;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_139) : "f"(S2[17].y), "f"(S2[17].x));
                    h_words[1] = _cvt_rn_f16x2_f32_139;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_736;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_736) : "f"(S[37]), "f"(S[36]));
                    h_words[0] = _cvt_rn_f16x2_f32_736;
                    uint32_t _cvt_rn_f16x2_f32_737;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_737) : "f"(S[39]), "f"(S[38]));
                    h_words[1] = _cvt_rn_f16x2_f32_737;
#else
                    uint32_t _cvt_rn_f16x2_f32_140;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_140) : "f"(S2[18].y), "f"(S2[18].x));
                    h_words[0] = _cvt_rn_f16x2_f32_140;
                    uint32_t _cvt_rn_f16x2_f32_141;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_141) : "f"(S2[19].y), "f"(S2[19].x));
                    h_words[1] = _cvt_rn_f16x2_f32_141;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 16ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_738;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_738) : "f"(S[41]), "f"(S[40]));
                    h_words[0] = _cvt_rn_f16x2_f32_738;
                    uint32_t _cvt_rn_f16x2_f32_739;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_739) : "f"(S[43]), "f"(S[42]));
                    h_words[1] = _cvt_rn_f16x2_f32_739;
#else
                    uint32_t _cvt_rn_f16x2_f32_142;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_142) : "f"(S2[20].y), "f"(S2[20].x));
                    h_words[0] = _cvt_rn_f16x2_f32_142;
                    uint32_t _cvt_rn_f16x2_f32_143;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_143) : "f"(S2[21].y), "f"(S2[21].x));
                    h_words[1] = _cvt_rn_f16x2_f32_143;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_740;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_740) : "f"(S[45]), "f"(S[44]));
                    h_words[0] = _cvt_rn_f16x2_f32_740;
                    uint32_t _cvt_rn_f16x2_f32_741;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_741) : "f"(S[47]), "f"(S[46]));
                    h_words[1] = _cvt_rn_f16x2_f32_741;
#else
                    uint32_t _cvt_rn_f16x2_f32_144;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_144) : "f"(S2[22].y), "f"(S2[22].x));
                    h_words[0] = _cvt_rn_f16x2_f32_144;
                    uint32_t _cvt_rn_f16x2_f32_145;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_145) : "f"(S2[23].y), "f"(S2[23].x));
                    h_words[1] = _cvt_rn_f16x2_f32_145;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 48ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_742;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_742) : "f"(S[49]), "f"(S[48]));
                    h_words[0] = _cvt_rn_f16x2_f32_742;
                    uint32_t _cvt_rn_f16x2_f32_743;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_743) : "f"(S[51]), "f"(S[50]));
                    h_words[1] = _cvt_rn_f16x2_f32_743;
#else
                    uint32_t _cvt_rn_f16x2_f32_146;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_146) : "f"(S2[24].y), "f"(S2[24].x));
                    h_words[0] = _cvt_rn_f16x2_f32_146;
                    uint32_t _cvt_rn_f16x2_f32_147;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_147) : "f"(S2[25].y), "f"(S2[25].x));
                    h_words[1] = _cvt_rn_f16x2_f32_147;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_744;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_744) : "f"(S[53]), "f"(S[52]));
                    h_words[0] = _cvt_rn_f16x2_f32_744;
                    uint32_t _cvt_rn_f16x2_f32_745;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_745) : "f"(S[55]), "f"(S[54]));
                    h_words[1] = _cvt_rn_f16x2_f32_745;
#else
                    uint32_t _cvt_rn_f16x2_f32_148;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_148) : "f"(S2[26].y), "f"(S2[26].x));
                    h_words[0] = _cvt_rn_f16x2_f32_148;
                    uint32_t _cvt_rn_f16x2_f32_149;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_149) : "f"(S2[27].y), "f"(S2[27].x));
                    h_words[1] = _cvt_rn_f16x2_f32_149;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 80ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_746;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_746) : "f"(S[57]), "f"(S[56]));
                    h_words[0] = _cvt_rn_f16x2_f32_746;
                    uint32_t _cvt_rn_f16x2_f32_747;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_747) : "f"(S[59]), "f"(S[58]));
                    h_words[1] = _cvt_rn_f16x2_f32_747;
#else
                    uint32_t _cvt_rn_f16x2_f32_150;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_150) : "f"(S2[28].y), "f"(S2[28].x));
                    h_words[0] = _cvt_rn_f16x2_f32_150;
                    uint32_t _cvt_rn_f16x2_f32_151;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_151) : "f"(S2[29].y), "f"(S2[29].x));
                    h_words[1] = _cvt_rn_f16x2_f32_151;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_748;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_748) : "f"(S[61]), "f"(S[60]));
                    h_words[0] = _cvt_rn_f16x2_f32_748;
                    uint32_t _cvt_rn_f16x2_f32_749;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_749) : "f"(S[63]), "f"(S[62]));
                    h_words[1] = _cvt_rn_f16x2_f32_749;
#else
                    uint32_t _cvt_rn_f16x2_f32_152;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_152) : "f"(S2[30].y), "f"(S2[30].x));
                    h_words[0] = _cvt_rn_f16x2_f32_152;
                    uint32_t _cvt_rn_f16x2_f32_153;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_153) : "f"(S2[31].y), "f"(S2[31].x));
                    h_words[1] = _cvt_rn_f16x2_f32_153;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 112ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_750;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_750) : "f"(S[65]), "f"(S[64]));
                    h_words[0] = _cvt_rn_f16x2_f32_750;
                    uint32_t _cvt_rn_f16x2_f32_751;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_751) : "f"(S[67]), "f"(S[66]));
                    h_words[1] = _cvt_rn_f16x2_f32_751;
#else
                    uint32_t _cvt_rn_f16x2_f32_154;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_154) : "f"(S2[32].y), "f"(S2[32].x));
                    h_words[0] = _cvt_rn_f16x2_f32_154;
                    uint32_t _cvt_rn_f16x2_f32_155;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_155) : "f"(S2[33].y), "f"(S2[33].x));
                    h_words[1] = _cvt_rn_f16x2_f32_155;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_752;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_752) : "f"(S[69]), "f"(S[68]));
                    h_words[0] = _cvt_rn_f16x2_f32_752;
                    uint32_t _cvt_rn_f16x2_f32_753;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_753) : "f"(S[71]), "f"(S[70]));
                    h_words[1] = _cvt_rn_f16x2_f32_753;
#else
                    uint32_t _cvt_rn_f16x2_f32_156;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_156) : "f"(S2[34].y), "f"(S2[34].x));
                    h_words[0] = _cvt_rn_f16x2_f32_156;
                    uint32_t _cvt_rn_f16x2_f32_157;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_157) : "f"(S2[35].y), "f"(S2[35].x));
                    h_words[1] = _cvt_rn_f16x2_f32_157;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 16ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_754;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_754) : "f"(S[73]), "f"(S[72]));
                    h_words[0] = _cvt_rn_f16x2_f32_754;
                    uint32_t _cvt_rn_f16x2_f32_755;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_755) : "f"(S[75]), "f"(S[74]));
                    h_words[1] = _cvt_rn_f16x2_f32_755;
#else
                    uint32_t _cvt_rn_f16x2_f32_158;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_158) : "f"(S2[36].y), "f"(S2[36].x));
                    h_words[0] = _cvt_rn_f16x2_f32_158;
                    uint32_t _cvt_rn_f16x2_f32_159;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_159) : "f"(S2[37].y), "f"(S2[37].x));
                    h_words[1] = _cvt_rn_f16x2_f32_159;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_756;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_756) : "f"(S[77]), "f"(S[76]));
                    h_words[0] = _cvt_rn_f16x2_f32_756;
                    uint32_t _cvt_rn_f16x2_f32_757;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_757) : "f"(S[79]), "f"(S[78]));
                    h_words[1] = _cvt_rn_f16x2_f32_757;
#else
                    uint32_t _cvt_rn_f16x2_f32_160;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_160) : "f"(S2[38].y), "f"(S2[38].x));
                    h_words[0] = _cvt_rn_f16x2_f32_160;
                    uint32_t _cvt_rn_f16x2_f32_161;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_161) : "f"(S2[39].y), "f"(S2[39].x));
                    h_words[1] = _cvt_rn_f16x2_f32_161;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 48ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_758;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_758) : "f"(S[81]), "f"(S[80]));
                    h_words[0] = _cvt_rn_f16x2_f32_758;
                    uint32_t _cvt_rn_f16x2_f32_759;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_759) : "f"(S[83]), "f"(S[82]));
                    h_words[1] = _cvt_rn_f16x2_f32_759;
#else
                    uint32_t _cvt_rn_f16x2_f32_162;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_162) : "f"(S2[40].y), "f"(S2[40].x));
                    h_words[0] = _cvt_rn_f16x2_f32_162;
                    uint32_t _cvt_rn_f16x2_f32_163;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_163) : "f"(S2[41].y), "f"(S2[41].x));
                    h_words[1] = _cvt_rn_f16x2_f32_163;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_760;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_760) : "f"(S[85]), "f"(S[84]));
                    h_words[0] = _cvt_rn_f16x2_f32_760;
                    uint32_t _cvt_rn_f16x2_f32_761;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_761) : "f"(S[87]), "f"(S[86]));
                    h_words[1] = _cvt_rn_f16x2_f32_761;
#else
                    uint32_t _cvt_rn_f16x2_f32_164;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_164) : "f"(S2[42].y), "f"(S2[42].x));
                    h_words[0] = _cvt_rn_f16x2_f32_164;
                    uint32_t _cvt_rn_f16x2_f32_165;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_165) : "f"(S2[43].y), "f"(S2[43].x));
                    h_words[1] = _cvt_rn_f16x2_f32_165;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 80ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_762;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_762) : "f"(S[89]), "f"(S[88]));
                    h_words[0] = _cvt_rn_f16x2_f32_762;
                    uint32_t _cvt_rn_f16x2_f32_763;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_763) : "f"(S[91]), "f"(S[90]));
                    h_words[1] = _cvt_rn_f16x2_f32_763;
#else
                    uint32_t _cvt_rn_f16x2_f32_166;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_166) : "f"(S2[44].y), "f"(S2[44].x));
                    h_words[0] = _cvt_rn_f16x2_f32_166;
                    uint32_t _cvt_rn_f16x2_f32_167;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_167) : "f"(S2[45].y), "f"(S2[45].x));
                    h_words[1] = _cvt_rn_f16x2_f32_167;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_764;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_764) : "f"(S[93]), "f"(S[92]));
                    h_words[0] = _cvt_rn_f16x2_f32_764;
                    uint32_t _cvt_rn_f16x2_f32_765;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_765) : "f"(S[95]), "f"(S[94]));
                    h_words[1] = _cvt_rn_f16x2_f32_765;
#else
                    uint32_t _cvt_rn_f16x2_f32_168;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_168) : "f"(S2[46].y), "f"(S2[46].x));
                    h_words[0] = _cvt_rn_f16x2_f32_168;
                    uint32_t _cvt_rn_f16x2_f32_169;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_169) : "f"(S2[47].y), "f"(S2[47].x));
                    h_words[1] = _cvt_rn_f16x2_f32_169;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 112ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_766;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_766) : "f"(S[97]), "f"(S[96]));
                    h_words[0] = _cvt_rn_f16x2_f32_766;
                    uint32_t _cvt_rn_f16x2_f32_767;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_767) : "f"(S[99]), "f"(S[98]));
                    h_words[1] = _cvt_rn_f16x2_f32_767;
#else
                    uint32_t _cvt_rn_f16x2_f32_170;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_170) : "f"(S2[48].y), "f"(S2[48].x));
                    h_words[0] = _cvt_rn_f16x2_f32_170;
                    uint32_t _cvt_rn_f16x2_f32_171;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_171) : "f"(S2[49].y), "f"(S2[49].x));
                    h_words[1] = _cvt_rn_f16x2_f32_171;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_768;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_768) : "f"(S[101]), "f"(S[100]));
                    h_words[0] = _cvt_rn_f16x2_f32_768;
                    uint32_t _cvt_rn_f16x2_f32_769;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_769) : "f"(S[103]), "f"(S[102]));
                    h_words[1] = _cvt_rn_f16x2_f32_769;
#else
                    uint32_t _cvt_rn_f16x2_f32_172;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_172) : "f"(S2[50].y), "f"(S2[50].x));
                    h_words[0] = _cvt_rn_f16x2_f32_172;
                    uint32_t _cvt_rn_f16x2_f32_173;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_173) : "f"(S2[51].y), "f"(S2[51].x));
                    h_words[1] = _cvt_rn_f16x2_f32_173;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 16ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_770;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_770) : "f"(S[105]), "f"(S[104]));
                    h_words[0] = _cvt_rn_f16x2_f32_770;
                    uint32_t _cvt_rn_f16x2_f32_771;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_771) : "f"(S[107]), "f"(S[106]));
                    h_words[1] = _cvt_rn_f16x2_f32_771;
#else
                    uint32_t _cvt_rn_f16x2_f32_174;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_174) : "f"(S2[52].y), "f"(S2[52].x));
                    h_words[0] = _cvt_rn_f16x2_f32_174;
                    uint32_t _cvt_rn_f16x2_f32_175;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_175) : "f"(S2[53].y), "f"(S2[53].x));
                    h_words[1] = _cvt_rn_f16x2_f32_175;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_772;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_772) : "f"(S[109]), "f"(S[108]));
                    h_words[0] = _cvt_rn_f16x2_f32_772;
                    uint32_t _cvt_rn_f16x2_f32_773;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_773) : "f"(S[111]), "f"(S[110]));
                    h_words[1] = _cvt_rn_f16x2_f32_773;
#else
                    uint32_t _cvt_rn_f16x2_f32_176;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_176) : "f"(S2[54].y), "f"(S2[54].x));
                    h_words[0] = _cvt_rn_f16x2_f32_176;
                    uint32_t _cvt_rn_f16x2_f32_177;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_177) : "f"(S2[55].y), "f"(S2[55].x));
                    h_words[1] = _cvt_rn_f16x2_f32_177;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 48ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_774;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_774) : "f"(S[113]), "f"(S[112]));
                    h_words[0] = _cvt_rn_f16x2_f32_774;
                    uint32_t _cvt_rn_f16x2_f32_775;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_775) : "f"(S[115]), "f"(S[114]));
                    h_words[1] = _cvt_rn_f16x2_f32_775;
#else
                    uint32_t _cvt_rn_f16x2_f32_178;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_178) : "f"(S2[56].y), "f"(S2[56].x));
                    h_words[0] = _cvt_rn_f16x2_f32_178;
                    uint32_t _cvt_rn_f16x2_f32_179;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_179) : "f"(S2[57].y), "f"(S2[57].x));
                    h_words[1] = _cvt_rn_f16x2_f32_179;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_776;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_776) : "f"(S[117]), "f"(S[116]));
                    h_words[0] = _cvt_rn_f16x2_f32_776;
                    uint32_t _cvt_rn_f16x2_f32_777;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_777) : "f"(S[119]), "f"(S[118]));
                    h_words[1] = _cvt_rn_f16x2_f32_777;
#else
                    uint32_t _cvt_rn_f16x2_f32_180;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_180) : "f"(S2[58].y), "f"(S2[58].x));
                    h_words[0] = _cvt_rn_f16x2_f32_180;
                    uint32_t _cvt_rn_f16x2_f32_181;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_181) : "f"(S2[59].y), "f"(S2[59].x));
                    h_words[1] = _cvt_rn_f16x2_f32_181;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 80ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_778;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_778) : "f"(S[121]), "f"(S[120]));
                    h_words[0] = _cvt_rn_f16x2_f32_778;
                    uint32_t _cvt_rn_f16x2_f32_779;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_779) : "f"(S[123]), "f"(S[122]));
                    h_words[1] = _cvt_rn_f16x2_f32_779;
#else
                    uint32_t _cvt_rn_f16x2_f32_182;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_182) : "f"(S2[60].y), "f"(S2[60].x));
                    h_words[0] = _cvt_rn_f16x2_f32_182;
                    uint32_t _cvt_rn_f16x2_f32_183;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_183) : "f"(S2[61].y), "f"(S2[61].x));
                    h_words[1] = _cvt_rn_f16x2_f32_183;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
#if __CUDA_ARCH__ == 900
                    uint32_t _cvt_rn_f16x2_f32_780;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_780) : "f"(S[125]), "f"(S[124]));
                    h_words[0] = _cvt_rn_f16x2_f32_780;
                    uint32_t _cvt_rn_f16x2_f32_781;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_781) : "f"(S[127]), "f"(S[126]));
                    h_words[1] = _cvt_rn_f16x2_f32_781;
#else
                    uint32_t _cvt_rn_f16x2_f32_184;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_184) : "f"(S2[62].y), "f"(S2[62].x));
                    h_words[0] = _cvt_rn_f16x2_f32_184;
                    uint32_t _cvt_rn_f16x2_f32_185;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_185) : "f"(S2[63].y), "f"(S2[63].x));
                    h_words[1] = _cvt_rn_f16x2_f32_185;
#endif
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 112ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                }
            }
        }
        if (use_ssm_state_indices == 0) {
            #pragma unroll 1
            for (int t_2 = 0; t_2 < num_tokens; t_2++) {
                float alpha_1 = sAlpha[t_2];
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
                S[64] = mul_f32_rn_ftz(S[64], alpha_1);
                S[65] = mul_f32_rn_ftz(S[65], alpha_1);
                S[66] = mul_f32_rn_ftz(S[66], alpha_1);
                S[67] = mul_f32_rn_ftz(S[67], alpha_1);
                S[68] = mul_f32_rn_ftz(S[68], alpha_1);
                S[69] = mul_f32_rn_ftz(S[69], alpha_1);
                S[70] = mul_f32_rn_ftz(S[70], alpha_1);
                S[71] = mul_f32_rn_ftz(S[71], alpha_1);
                S[72] = mul_f32_rn_ftz(S[72], alpha_1);
                S[73] = mul_f32_rn_ftz(S[73], alpha_1);
                S[74] = mul_f32_rn_ftz(S[74], alpha_1);
                S[75] = mul_f32_rn_ftz(S[75], alpha_1);
                S[76] = mul_f32_rn_ftz(S[76], alpha_1);
                S[77] = mul_f32_rn_ftz(S[77], alpha_1);
                S[78] = mul_f32_rn_ftz(S[78], alpha_1);
                S[79] = mul_f32_rn_ftz(S[79], alpha_1);
                S[80] = mul_f32_rn_ftz(S[80], alpha_1);
                S[81] = mul_f32_rn_ftz(S[81], alpha_1);
                S[82] = mul_f32_rn_ftz(S[82], alpha_1);
                S[83] = mul_f32_rn_ftz(S[83], alpha_1);
                S[84] = mul_f32_rn_ftz(S[84], alpha_1);
                S[85] = mul_f32_rn_ftz(S[85], alpha_1);
                S[86] = mul_f32_rn_ftz(S[86], alpha_1);
                S[87] = mul_f32_rn_ftz(S[87], alpha_1);
                S[88] = mul_f32_rn_ftz(S[88], alpha_1);
                S[89] = mul_f32_rn_ftz(S[89], alpha_1);
                S[90] = mul_f32_rn_ftz(S[90], alpha_1);
                S[91] = mul_f32_rn_ftz(S[91], alpha_1);
                S[92] = mul_f32_rn_ftz(S[92], alpha_1);
                S[93] = mul_f32_rn_ftz(S[93], alpha_1);
                S[94] = mul_f32_rn_ftz(S[94], alpha_1);
                S[95] = mul_f32_rn_ftz(S[95], alpha_1);
                S[96] = mul_f32_rn_ftz(S[96], alpha_1);
                S[97] = mul_f32_rn_ftz(S[97], alpha_1);
                S[98] = mul_f32_rn_ftz(S[98], alpha_1);
                S[99] = mul_f32_rn_ftz(S[99], alpha_1);
                S[100] = mul_f32_rn_ftz(S[100], alpha_1);
                S[101] = mul_f32_rn_ftz(S[101], alpha_1);
                S[102] = mul_f32_rn_ftz(S[102], alpha_1);
                S[103] = mul_f32_rn_ftz(S[103], alpha_1);
                S[104] = mul_f32_rn_ftz(S[104], alpha_1);
                S[105] = mul_f32_rn_ftz(S[105], alpha_1);
                S[106] = mul_f32_rn_ftz(S[106], alpha_1);
                S[107] = mul_f32_rn_ftz(S[107], alpha_1);
                S[108] = mul_f32_rn_ftz(S[108], alpha_1);
                S[109] = mul_f32_rn_ftz(S[109], alpha_1);
                S[110] = mul_f32_rn_ftz(S[110], alpha_1);
                S[111] = mul_f32_rn_ftz(S[111], alpha_1);
                S[112] = mul_f32_rn_ftz(S[112], alpha_1);
                S[113] = mul_f32_rn_ftz(S[113], alpha_1);
                S[114] = mul_f32_rn_ftz(S[114], alpha_1);
                S[115] = mul_f32_rn_ftz(S[115], alpha_1);
                S[116] = mul_f32_rn_ftz(S[116], alpha_1);
                S[117] = mul_f32_rn_ftz(S[117], alpha_1);
                S[118] = mul_f32_rn_ftz(S[118], alpha_1);
                S[119] = mul_f32_rn_ftz(S[119], alpha_1);
                S[120] = mul_f32_rn_ftz(S[120], alpha_1);
                S[121] = mul_f32_rn_ftz(S[121], alpha_1);
                S[122] = mul_f32_rn_ftz(S[122], alpha_1);
                S[123] = mul_f32_rn_ftz(S[123], alpha_1);
                S[124] = mul_f32_rn_ftz(S[124], alpha_1);
                S[125] = mul_f32_rn_ftz(S[125], alpha_1);
                S[126] = mul_f32_rn_ftz(S[126], alpha_1);
                S[127] = mul_f32_rn_ftz(S[127], alpha_1);
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
                S2[32] = mul_f32x2(S2[32], alpha2_1);
                S2[33] = mul_f32x2(S2[33], alpha2_1);
                S2[34] = mul_f32x2(S2[34], alpha2_1);
                S2[35] = mul_f32x2(S2[35], alpha2_1);
                S2[36] = mul_f32x2(S2[36], alpha2_1);
                S2[37] = mul_f32x2(S2[37], alpha2_1);
                S2[38] = mul_f32x2(S2[38], alpha2_1);
                S2[39] = mul_f32x2(S2[39], alpha2_1);
                S2[40] = mul_f32x2(S2[40], alpha2_1);
                S2[41] = mul_f32x2(S2[41], alpha2_1);
                S2[42] = mul_f32x2(S2[42], alpha2_1);
                S2[43] = mul_f32x2(S2[43], alpha2_1);
                S2[44] = mul_f32x2(S2[44], alpha2_1);
                S2[45] = mul_f32x2(S2[45], alpha2_1);
                S2[46] = mul_f32x2(S2[46], alpha2_1);
                S2[47] = mul_f32x2(S2[47], alpha2_1);
                S2[48] = mul_f32x2(S2[48], alpha2_1);
                S2[49] = mul_f32x2(S2[49], alpha2_1);
                S2[50] = mul_f32x2(S2[50], alpha2_1);
                S2[51] = mul_f32x2(S2[51], alpha2_1);
                S2[52] = mul_f32x2(S2[52], alpha2_1);
                S2[53] = mul_f32x2(S2[53], alpha2_1);
                S2[54] = mul_f32x2(S2[54], alpha2_1);
                S2[55] = mul_f32x2(S2[55], alpha2_1);
                S2[56] = mul_f32x2(S2[56], alpha2_1);
                S2[57] = mul_f32x2(S2[57], alpha2_1);
                S2[58] = mul_f32x2(S2[58], alpha2_1);
                S2[59] = mul_f32x2(S2[59], alpha2_1);
                S2[60] = mul_f32x2(S2[60], alpha2_1);
                S2[61] = mul_f32x2(S2[61], alpha2_1);
                S2[62] = mul_f32x2(S2[62], alpha2_1);
                S2[63] = mul_f32x2(S2[63], alpha2_1);
#endif
                int m_1 = t_2;
                unsigned int kbase_1 = sK_addr + (unsigned int)(m_1 * 512) + lane_byte;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(kbase_1));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(kbase_1 + 64));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(kbase_1 + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(kbase_1 + 192));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[16])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 3]))
                    : "r"(kbase_1 + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[20])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 3]))
                    : "r"(kbase_1 + 320));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[24])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 3]))
                    : "r"(kbase_1 + 384));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[28])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 3]))
                    : "r"(kbase_1 + 448));
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
                k2[8] = make_float2(k_regs[16], k_regs[17]);
                k2[9] = make_float2(k_regs[18], k_regs[19]);
                k2[10] = make_float2(k_regs[20], k_regs[21]);
                k2[11] = make_float2(k_regs[22], k_regs[23]);
                k2[12] = make_float2(k_regs[24], k_regs[25]);
                k2[13] = make_float2(k_regs[26], k_regs[27]);
                k2[14] = make_float2(k_regs[28], k_regs[29]);
                k2[15] = make_float2(k_regs[30], k_regs[31]);
#endif
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                    : "r"(sV_addr + (unsigned int)(m_1 * 256) + group_byte));
                float beta_1 = sBeta[m_1];
#if __CUDA_ARCH__ == 900
                float acc0_x_7 = 0.0f;
                float acc0_y_7 = 0.0f;
                float acc1_x_7 = 0.0f;
                float acc1_y_7 = 0.0f;
                float _fma_782;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_782) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_7));
                acc0_x_7 = _fma_782;
                float _fma_783;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_783) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_7));
                acc0_y_7 = _fma_783;
                float _fma_784;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_784) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_7));
                acc1_x_7 = _fma_784;
                float _fma_785;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_785) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_7));
                acc1_y_7 = _fma_785;
                float _fma_786;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_786) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_7));
                acc0_x_7 = _fma_786;
                float _fma_787;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_787) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_7));
                acc0_y_7 = _fma_787;
                float _fma_788;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_788) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_7));
                acc1_x_7 = _fma_788;
                float _fma_789;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_789) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_7));
                acc1_y_7 = _fma_789;
                float _fma_790;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_790) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_7));
                acc0_x_7 = _fma_790;
                float _fma_791;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_791) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_7));
                acc0_y_7 = _fma_791;
                float _fma_792;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_792) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_7));
                acc1_x_7 = _fma_792;
                float _fma_793;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_793) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_7));
                acc1_y_7 = _fma_793;
                float _fma_794;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_794) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_7));
                acc0_x_7 = _fma_794;
                float _fma_795;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_795) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_7));
                acc0_y_7 = _fma_795;
                float _fma_796;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_796) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_7));
                acc1_x_7 = _fma_796;
                float _fma_797;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_797) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_7));
                acc1_y_7 = _fma_797;
                float _fma_798;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_798) : "f"(S[16]), "f"(k_regs[16]), "f"(acc0_x_7));
                acc0_x_7 = _fma_798;
                float _fma_799;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_799) : "f"(S[17]), "f"(k_regs[17]), "f"(acc0_y_7));
                acc0_y_7 = _fma_799;
                float _fma_800;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_800) : "f"(S[18]), "f"(k_regs[18]), "f"(acc1_x_7));
                acc1_x_7 = _fma_800;
                float _fma_801;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_801) : "f"(S[19]), "f"(k_regs[19]), "f"(acc1_y_7));
                acc1_y_7 = _fma_801;
                float _fma_802;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_802) : "f"(S[20]), "f"(k_regs[20]), "f"(acc0_x_7));
                acc0_x_7 = _fma_802;
                float _fma_803;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_803) : "f"(S[21]), "f"(k_regs[21]), "f"(acc0_y_7));
                acc0_y_7 = _fma_803;
                float _fma_804;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_804) : "f"(S[22]), "f"(k_regs[22]), "f"(acc1_x_7));
                acc1_x_7 = _fma_804;
                float _fma_805;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_805) : "f"(S[23]), "f"(k_regs[23]), "f"(acc1_y_7));
                acc1_y_7 = _fma_805;
                float _fma_806;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_806) : "f"(S[24]), "f"(k_regs[24]), "f"(acc0_x_7));
                acc0_x_7 = _fma_806;
                float _fma_807;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_807) : "f"(S[25]), "f"(k_regs[25]), "f"(acc0_y_7));
                acc0_y_7 = _fma_807;
                float _fma_808;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_808) : "f"(S[26]), "f"(k_regs[26]), "f"(acc1_x_7));
                acc1_x_7 = _fma_808;
                float _fma_809;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_809) : "f"(S[27]), "f"(k_regs[27]), "f"(acc1_y_7));
                acc1_y_7 = _fma_809;
                float _fma_810;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_810) : "f"(S[28]), "f"(k_regs[28]), "f"(acc0_x_7));
                acc0_x_7 = _fma_810;
                float _fma_811;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_811) : "f"(S[29]), "f"(k_regs[29]), "f"(acc0_y_7));
                acc0_y_7 = _fma_811;
                float _fma_812;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_812) : "f"(S[30]), "f"(k_regs[30]), "f"(acc1_x_7));
                acc1_x_7 = _fma_812;
                float _fma_813;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_813) : "f"(S[31]), "f"(k_regs[31]), "f"(acc1_y_7));
                acc1_y_7 = _fma_813;
                float _add_814;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_814) : "f"(acc0_x_7), "f"(acc1_x_7));
                float _add_815;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_815) : "f"(acc0_y_7), "f"(acc1_y_7));
                float dot_7 = _add_814 + _add_815;
                float _shfl_xor_sync_816 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 1);
                dot_7 = dot_7 + _shfl_xor_sync_816;
                float _shfl_xor_sync_817 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 2);
                dot_7 = dot_7 + _shfl_xor_sync_817;
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
                acc0_7 = fma_f32x2_rn_ftz(S2[8], k2[8], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[9], k2[9], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[10], k2[10], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[11], k2[11], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[12], k2[12], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[13], k2[13], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[14], k2[14], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[15], k2[15], acc0_7);
                float dot_7 = acc0_7.x + acc0_7.y;
                float _shfl_xor_sync_186 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 1);
                dot_7 = dot_7 + _shfl_xor_sync_186;
                float _shfl_xor_sync_187 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 2);
                dot_7 = dot_7 + _shfl_xor_sync_187;
#endif
                float vn_4 = (v_regs[0] - dot_7) * beta_1;
#if __CUDA_ARCH__ == 900
                float _fma_818;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_818) : "f"(k_regs[0]), "f"(vn_4), "f"(S[0]));
                S[0] = _fma_818;
                float _fma_819;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_819) : "f"(k_regs[1]), "f"(vn_4), "f"(S[1]));
                S[1] = _fma_819;
                float _fma_820;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_820) : "f"(k_regs[2]), "f"(vn_4), "f"(S[2]));
                S[2] = _fma_820;
                float _fma_821;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_821) : "f"(k_regs[3]), "f"(vn_4), "f"(S[3]));
                S[3] = _fma_821;
                float _fma_822;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_822) : "f"(k_regs[4]), "f"(vn_4), "f"(S[4]));
                S[4] = _fma_822;
                float _fma_823;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_823) : "f"(k_regs[5]), "f"(vn_4), "f"(S[5]));
                S[5] = _fma_823;
                float _fma_824;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_824) : "f"(k_regs[6]), "f"(vn_4), "f"(S[6]));
                S[6] = _fma_824;
                float _fma_825;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_825) : "f"(k_regs[7]), "f"(vn_4), "f"(S[7]));
                S[7] = _fma_825;
                float _fma_826;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_826) : "f"(k_regs[8]), "f"(vn_4), "f"(S[8]));
                S[8] = _fma_826;
                float _fma_827;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_827) : "f"(k_regs[9]), "f"(vn_4), "f"(S[9]));
                S[9] = _fma_827;
                float _fma_828;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_828) : "f"(k_regs[10]), "f"(vn_4), "f"(S[10]));
                S[10] = _fma_828;
                float _fma_829;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_829) : "f"(k_regs[11]), "f"(vn_4), "f"(S[11]));
                S[11] = _fma_829;
                float _fma_830;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_830) : "f"(k_regs[12]), "f"(vn_4), "f"(S[12]));
                S[12] = _fma_830;
                float _fma_831;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_831) : "f"(k_regs[13]), "f"(vn_4), "f"(S[13]));
                S[13] = _fma_831;
                float _fma_832;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_832) : "f"(k_regs[14]), "f"(vn_4), "f"(S[14]));
                S[14] = _fma_832;
                float _fma_833;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_833) : "f"(k_regs[15]), "f"(vn_4), "f"(S[15]));
                S[15] = _fma_833;
                float _fma_834;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_834) : "f"(k_regs[16]), "f"(vn_4), "f"(S[16]));
                S[16] = _fma_834;
                float _fma_835;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_835) : "f"(k_regs[17]), "f"(vn_4), "f"(S[17]));
                S[17] = _fma_835;
                float _fma_836;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_836) : "f"(k_regs[18]), "f"(vn_4), "f"(S[18]));
                S[18] = _fma_836;
                float _fma_837;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_837) : "f"(k_regs[19]), "f"(vn_4), "f"(S[19]));
                S[19] = _fma_837;
                float _fma_838;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_838) : "f"(k_regs[20]), "f"(vn_4), "f"(S[20]));
                S[20] = _fma_838;
                float _fma_839;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_839) : "f"(k_regs[21]), "f"(vn_4), "f"(S[21]));
                S[21] = _fma_839;
                float _fma_840;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_840) : "f"(k_regs[22]), "f"(vn_4), "f"(S[22]));
                S[22] = _fma_840;
                float _fma_841;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_841) : "f"(k_regs[23]), "f"(vn_4), "f"(S[23]));
                S[23] = _fma_841;
                float _fma_842;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_842) : "f"(k_regs[24]), "f"(vn_4), "f"(S[24]));
                S[24] = _fma_842;
                float _fma_843;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_843) : "f"(k_regs[25]), "f"(vn_4), "f"(S[25]));
                S[25] = _fma_843;
                float _fma_844;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_844) : "f"(k_regs[26]), "f"(vn_4), "f"(S[26]));
                S[26] = _fma_844;
                float _fma_845;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_845) : "f"(k_regs[27]), "f"(vn_4), "f"(S[27]));
                S[27] = _fma_845;
                float _fma_846;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_846) : "f"(k_regs[28]), "f"(vn_4), "f"(S[28]));
                S[28] = _fma_846;
                float _fma_847;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_847) : "f"(k_regs[29]), "f"(vn_4), "f"(S[29]));
                S[29] = _fma_847;
                float _fma_848;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_848) : "f"(k_regs[30]), "f"(vn_4), "f"(S[30]));
                S[30] = _fma_848;
                float _fma_849;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_849) : "f"(k_regs[31]), "f"(vn_4), "f"(S[31]));
                S[31] = _fma_849;
                float acc0_x_7_1 = 0.0f;
                float acc0_y_7_1 = 0.0f;
                float acc1_x_7_1 = 0.0f;
                float acc1_y_7_1 = 0.0f;
                float _fma_850;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_850) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_850;
                float _fma_851;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_851) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_851;
                float _fma_852;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_852) : "f"(S[34]), "f"(k_regs[2]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_852;
                float _fma_853;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_853) : "f"(S[35]), "f"(k_regs[3]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_853;
                float _fma_854;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_854) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_854;
                float _fma_855;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_855) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_855;
                float _fma_856;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_856) : "f"(S[38]), "f"(k_regs[6]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_856;
                float _fma_857;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_857) : "f"(S[39]), "f"(k_regs[7]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_857;
                float _fma_858;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_858) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_858;
                float _fma_859;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_859) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_859;
                float _fma_860;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_860) : "f"(S[42]), "f"(k_regs[10]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_860;
                float _fma_861;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_861) : "f"(S[43]), "f"(k_regs[11]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_861;
                float _fma_862;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_862) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_862;
                float _fma_863;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_863) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_863;
                float _fma_864;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_864) : "f"(S[46]), "f"(k_regs[14]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_864;
                float _fma_865;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_865) : "f"(S[47]), "f"(k_regs[15]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_865;
                float _fma_866;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_866) : "f"(S[48]), "f"(k_regs[16]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_866;
                float _fma_867;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_867) : "f"(S[49]), "f"(k_regs[17]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_867;
                float _fma_868;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_868) : "f"(S[50]), "f"(k_regs[18]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_868;
                float _fma_869;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_869) : "f"(S[51]), "f"(k_regs[19]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_869;
                float _fma_870;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_870) : "f"(S[52]), "f"(k_regs[20]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_870;
                float _fma_871;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_871) : "f"(S[53]), "f"(k_regs[21]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_871;
                float _fma_872;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_872) : "f"(S[54]), "f"(k_regs[22]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_872;
                float _fma_873;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_873) : "f"(S[55]), "f"(k_regs[23]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_873;
                float _fma_874;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_874) : "f"(S[56]), "f"(k_regs[24]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_874;
                float _fma_875;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_875) : "f"(S[57]), "f"(k_regs[25]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_875;
                float _fma_876;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_876) : "f"(S[58]), "f"(k_regs[26]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_876;
                float _fma_877;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_877) : "f"(S[59]), "f"(k_regs[27]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_877;
                float _fma_878;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_878) : "f"(S[60]), "f"(k_regs[28]), "f"(acc0_x_7_1));
                acc0_x_7_1 = _fma_878;
                float _fma_879;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_879) : "f"(S[61]), "f"(k_regs[29]), "f"(acc0_y_7_1));
                acc0_y_7_1 = _fma_879;
                float _fma_880;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_880) : "f"(S[62]), "f"(k_regs[30]), "f"(acc1_x_7_1));
                acc1_x_7_1 = _fma_880;
                float _fma_881;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_881) : "f"(S[63]), "f"(k_regs[31]), "f"(acc1_y_7_1));
                acc1_y_7_1 = _fma_881;
                float _add_882;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_882) : "f"(acc0_x_7_1), "f"(acc1_x_7_1));
                float _add_883;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_883) : "f"(acc0_y_7_1), "f"(acc1_y_7_1));
                float dot_7_1 = _add_882 + _add_883;
                float _shfl_xor_sync_884 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 1);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_884;
                float _shfl_xor_sync_885 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 2);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_885;
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
                S2[8] = fma_f32x2_rn_ftz(k2[8], vn2_4, S2[8]);
                S2[9] = fma_f32x2_rn_ftz(k2[9], vn2_4, S2[9]);
                S2[10] = fma_f32x2_rn_ftz(k2[10], vn2_4, S2[10]);
                S2[11] = fma_f32x2_rn_ftz(k2[11], vn2_4, S2[11]);
                S2[12] = fma_f32x2_rn_ftz(k2[12], vn2_4, S2[12]);
                S2[13] = fma_f32x2_rn_ftz(k2[13], vn2_4, S2[13]);
                S2[14] = fma_f32x2_rn_ftz(k2[14], vn2_4, S2[14]);
                S2[15] = fma_f32x2_rn_ftz(k2[15], vn2_4, S2[15]);
                float2 acc0_7_1 = make_float2(0.0f, 0.0f);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[16], k2[0], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[17], k2[1], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[18], k2[2], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[19], k2[3], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[20], k2[4], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[21], k2[5], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[22], k2[6], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[23], k2[7], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[24], k2[8], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[25], k2[9], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[26], k2[10], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[27], k2[11], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[28], k2[12], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[29], k2[13], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[30], k2[14], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[31], k2[15], acc0_7_1);
                float dot_7_1 = acc0_7_1.x + acc0_7_1.y;
                float _shfl_xor_sync_188 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 1);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_188;
                float _shfl_xor_sync_189 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 2);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_189;
#endif
                float vn_5 = (v_regs[1] - dot_7_1) * beta_1;
#if __CUDA_ARCH__ == 900
                float _fma_886;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_886) : "f"(k_regs[0]), "f"(vn_5), "f"(S[32]));
                S[32] = _fma_886;
                float _fma_887;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_887) : "f"(k_regs[1]), "f"(vn_5), "f"(S[33]));
                S[33] = _fma_887;
                float _fma_888;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_888) : "f"(k_regs[2]), "f"(vn_5), "f"(S[34]));
                S[34] = _fma_888;
                float _fma_889;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_889) : "f"(k_regs[3]), "f"(vn_5), "f"(S[35]));
                S[35] = _fma_889;
                float _fma_890;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_890) : "f"(k_regs[4]), "f"(vn_5), "f"(S[36]));
                S[36] = _fma_890;
                float _fma_891;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_891) : "f"(k_regs[5]), "f"(vn_5), "f"(S[37]));
                S[37] = _fma_891;
                float _fma_892;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_892) : "f"(k_regs[6]), "f"(vn_5), "f"(S[38]));
                S[38] = _fma_892;
                float _fma_893;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_893) : "f"(k_regs[7]), "f"(vn_5), "f"(S[39]));
                S[39] = _fma_893;
                float _fma_894;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_894) : "f"(k_regs[8]), "f"(vn_5), "f"(S[40]));
                S[40] = _fma_894;
                float _fma_895;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_895) : "f"(k_regs[9]), "f"(vn_5), "f"(S[41]));
                S[41] = _fma_895;
                float _fma_896;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_896) : "f"(k_regs[10]), "f"(vn_5), "f"(S[42]));
                S[42] = _fma_896;
                float _fma_897;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_897) : "f"(k_regs[11]), "f"(vn_5), "f"(S[43]));
                S[43] = _fma_897;
                float _fma_898;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_898) : "f"(k_regs[12]), "f"(vn_5), "f"(S[44]));
                S[44] = _fma_898;
                float _fma_899;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_899) : "f"(k_regs[13]), "f"(vn_5), "f"(S[45]));
                S[45] = _fma_899;
                float _fma_900;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_900) : "f"(k_regs[14]), "f"(vn_5), "f"(S[46]));
                S[46] = _fma_900;
                float _fma_901;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_901) : "f"(k_regs[15]), "f"(vn_5), "f"(S[47]));
                S[47] = _fma_901;
                float _fma_902;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_902) : "f"(k_regs[16]), "f"(vn_5), "f"(S[48]));
                S[48] = _fma_902;
                float _fma_903;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_903) : "f"(k_regs[17]), "f"(vn_5), "f"(S[49]));
                S[49] = _fma_903;
                float _fma_904;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_904) : "f"(k_regs[18]), "f"(vn_5), "f"(S[50]));
                S[50] = _fma_904;
                float _fma_905;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_905) : "f"(k_regs[19]), "f"(vn_5), "f"(S[51]));
                S[51] = _fma_905;
                float _fma_906;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_906) : "f"(k_regs[20]), "f"(vn_5), "f"(S[52]));
                S[52] = _fma_906;
                float _fma_907;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_907) : "f"(k_regs[21]), "f"(vn_5), "f"(S[53]));
                S[53] = _fma_907;
                float _fma_908;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_908) : "f"(k_regs[22]), "f"(vn_5), "f"(S[54]));
                S[54] = _fma_908;
                float _fma_909;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_909) : "f"(k_regs[23]), "f"(vn_5), "f"(S[55]));
                S[55] = _fma_909;
                float _fma_910;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_910) : "f"(k_regs[24]), "f"(vn_5), "f"(S[56]));
                S[56] = _fma_910;
                float _fma_911;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_911) : "f"(k_regs[25]), "f"(vn_5), "f"(S[57]));
                S[57] = _fma_911;
                float _fma_912;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_912) : "f"(k_regs[26]), "f"(vn_5), "f"(S[58]));
                S[58] = _fma_912;
                float _fma_913;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_913) : "f"(k_regs[27]), "f"(vn_5), "f"(S[59]));
                S[59] = _fma_913;
                float _fma_914;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_914) : "f"(k_regs[28]), "f"(vn_5), "f"(S[60]));
                S[60] = _fma_914;
                float _fma_915;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_915) : "f"(k_regs[29]), "f"(vn_5), "f"(S[61]));
                S[61] = _fma_915;
                float _fma_916;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_916) : "f"(k_regs[30]), "f"(vn_5), "f"(S[62]));
                S[62] = _fma_916;
                float _fma_917;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_917) : "f"(k_regs[31]), "f"(vn_5), "f"(S[63]));
                S[63] = _fma_917;
                float acc0_x_8 = 0.0f;
                float acc0_y_8 = 0.0f;
                float acc1_x_8 = 0.0f;
                float acc1_y_8 = 0.0f;
                float _fma_918;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_918) : "f"(S[64]), "f"(k_regs[0]), "f"(acc0_x_8));
                acc0_x_8 = _fma_918;
                float _fma_919;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_919) : "f"(S[65]), "f"(k_regs[1]), "f"(acc0_y_8));
                acc0_y_8 = _fma_919;
                float _fma_920;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_920) : "f"(S[66]), "f"(k_regs[2]), "f"(acc1_x_8));
                acc1_x_8 = _fma_920;
                float _fma_921;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_921) : "f"(S[67]), "f"(k_regs[3]), "f"(acc1_y_8));
                acc1_y_8 = _fma_921;
                float _fma_922;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_922) : "f"(S[68]), "f"(k_regs[4]), "f"(acc0_x_8));
                acc0_x_8 = _fma_922;
                float _fma_923;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_923) : "f"(S[69]), "f"(k_regs[5]), "f"(acc0_y_8));
                acc0_y_8 = _fma_923;
                float _fma_924;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_924) : "f"(S[70]), "f"(k_regs[6]), "f"(acc1_x_8));
                acc1_x_8 = _fma_924;
                float _fma_925;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_925) : "f"(S[71]), "f"(k_regs[7]), "f"(acc1_y_8));
                acc1_y_8 = _fma_925;
                float _fma_926;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_926) : "f"(S[72]), "f"(k_regs[8]), "f"(acc0_x_8));
                acc0_x_8 = _fma_926;
                float _fma_927;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_927) : "f"(S[73]), "f"(k_regs[9]), "f"(acc0_y_8));
                acc0_y_8 = _fma_927;
                float _fma_928;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_928) : "f"(S[74]), "f"(k_regs[10]), "f"(acc1_x_8));
                acc1_x_8 = _fma_928;
                float _fma_929;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_929) : "f"(S[75]), "f"(k_regs[11]), "f"(acc1_y_8));
                acc1_y_8 = _fma_929;
                float _fma_930;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_930) : "f"(S[76]), "f"(k_regs[12]), "f"(acc0_x_8));
                acc0_x_8 = _fma_930;
                float _fma_931;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_931) : "f"(S[77]), "f"(k_regs[13]), "f"(acc0_y_8));
                acc0_y_8 = _fma_931;
                float _fma_932;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_932) : "f"(S[78]), "f"(k_regs[14]), "f"(acc1_x_8));
                acc1_x_8 = _fma_932;
                float _fma_933;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_933) : "f"(S[79]), "f"(k_regs[15]), "f"(acc1_y_8));
                acc1_y_8 = _fma_933;
                float _fma_934;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_934) : "f"(S[80]), "f"(k_regs[16]), "f"(acc0_x_8));
                acc0_x_8 = _fma_934;
                float _fma_935;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_935) : "f"(S[81]), "f"(k_regs[17]), "f"(acc0_y_8));
                acc0_y_8 = _fma_935;
                float _fma_936;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_936) : "f"(S[82]), "f"(k_regs[18]), "f"(acc1_x_8));
                acc1_x_8 = _fma_936;
                float _fma_937;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_937) : "f"(S[83]), "f"(k_regs[19]), "f"(acc1_y_8));
                acc1_y_8 = _fma_937;
                float _fma_938;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_938) : "f"(S[84]), "f"(k_regs[20]), "f"(acc0_x_8));
                acc0_x_8 = _fma_938;
                float _fma_939;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_939) : "f"(S[85]), "f"(k_regs[21]), "f"(acc0_y_8));
                acc0_y_8 = _fma_939;
                float _fma_940;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_940) : "f"(S[86]), "f"(k_regs[22]), "f"(acc1_x_8));
                acc1_x_8 = _fma_940;
                float _fma_941;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_941) : "f"(S[87]), "f"(k_regs[23]), "f"(acc1_y_8));
                acc1_y_8 = _fma_941;
                float _fma_942;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_942) : "f"(S[88]), "f"(k_regs[24]), "f"(acc0_x_8));
                acc0_x_8 = _fma_942;
                float _fma_943;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_943) : "f"(S[89]), "f"(k_regs[25]), "f"(acc0_y_8));
                acc0_y_8 = _fma_943;
                float _fma_944;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_944) : "f"(S[90]), "f"(k_regs[26]), "f"(acc1_x_8));
                acc1_x_8 = _fma_944;
                float _fma_945;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_945) : "f"(S[91]), "f"(k_regs[27]), "f"(acc1_y_8));
                acc1_y_8 = _fma_945;
                float _fma_946;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_946) : "f"(S[92]), "f"(k_regs[28]), "f"(acc0_x_8));
                acc0_x_8 = _fma_946;
                float _fma_947;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_947) : "f"(S[93]), "f"(k_regs[29]), "f"(acc0_y_8));
                acc0_y_8 = _fma_947;
                float _fma_948;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_948) : "f"(S[94]), "f"(k_regs[30]), "f"(acc1_x_8));
                acc1_x_8 = _fma_948;
                float _fma_949;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_949) : "f"(S[95]), "f"(k_regs[31]), "f"(acc1_y_8));
                acc1_y_8 = _fma_949;
                float _add_950;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_950) : "f"(acc0_x_8), "f"(acc1_x_8));
                float _add_951;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_951) : "f"(acc0_y_8), "f"(acc1_y_8));
                float dot_8 = _add_950 + _add_951;
                float _shfl_xor_sync_952 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 1);
                dot_8 = dot_8 + _shfl_xor_sync_952;
                float _shfl_xor_sync_953 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 2);
                dot_8 = dot_8 + _shfl_xor_sync_953;
#else
                float2 vn2_5 = make_float2(vn_5, vn_5);
                S2[16] = fma_f32x2_rn_ftz(k2[0], vn2_5, S2[16]);
                S2[17] = fma_f32x2_rn_ftz(k2[1], vn2_5, S2[17]);
                S2[18] = fma_f32x2_rn_ftz(k2[2], vn2_5, S2[18]);
                S2[19] = fma_f32x2_rn_ftz(k2[3], vn2_5, S2[19]);
                S2[20] = fma_f32x2_rn_ftz(k2[4], vn2_5, S2[20]);
                S2[21] = fma_f32x2_rn_ftz(k2[5], vn2_5, S2[21]);
                S2[22] = fma_f32x2_rn_ftz(k2[6], vn2_5, S2[22]);
                S2[23] = fma_f32x2_rn_ftz(k2[7], vn2_5, S2[23]);
                S2[24] = fma_f32x2_rn_ftz(k2[8], vn2_5, S2[24]);
                S2[25] = fma_f32x2_rn_ftz(k2[9], vn2_5, S2[25]);
                S2[26] = fma_f32x2_rn_ftz(k2[10], vn2_5, S2[26]);
                S2[27] = fma_f32x2_rn_ftz(k2[11], vn2_5, S2[27]);
                S2[28] = fma_f32x2_rn_ftz(k2[12], vn2_5, S2[28]);
                S2[29] = fma_f32x2_rn_ftz(k2[13], vn2_5, S2[29]);
                S2[30] = fma_f32x2_rn_ftz(k2[14], vn2_5, S2[30]);
                S2[31] = fma_f32x2_rn_ftz(k2[15], vn2_5, S2[31]);
                float2 acc0_8 = make_float2(0.0f, 0.0f);
                acc0_8 = fma_f32x2_rn_ftz(S2[32], k2[0], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[33], k2[1], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[34], k2[2], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[35], k2[3], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[36], k2[4], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[37], k2[5], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[38], k2[6], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[39], k2[7], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[40], k2[8], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[41], k2[9], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[42], k2[10], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[43], k2[11], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[44], k2[12], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[45], k2[13], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[46], k2[14], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[47], k2[15], acc0_8);
                float dot_8 = acc0_8.x + acc0_8.y;
                float _shfl_xor_sync_190 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 1);
                dot_8 = dot_8 + _shfl_xor_sync_190;
                float _shfl_xor_sync_191 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 2);
                dot_8 = dot_8 + _shfl_xor_sync_191;
#endif
                float vn_6 = (v_regs[2] - dot_8) * beta_1;
#if __CUDA_ARCH__ == 900
                float _fma_954;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_954) : "f"(k_regs[0]), "f"(vn_6), "f"(S[64]));
                S[64] = _fma_954;
                float _fma_955;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_955) : "f"(k_regs[1]), "f"(vn_6), "f"(S[65]));
                S[65] = _fma_955;
                float _fma_956;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_956) : "f"(k_regs[2]), "f"(vn_6), "f"(S[66]));
                S[66] = _fma_956;
                float _fma_957;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_957) : "f"(k_regs[3]), "f"(vn_6), "f"(S[67]));
                S[67] = _fma_957;
                float _fma_958;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_958) : "f"(k_regs[4]), "f"(vn_6), "f"(S[68]));
                S[68] = _fma_958;
                float _fma_959;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_959) : "f"(k_regs[5]), "f"(vn_6), "f"(S[69]));
                S[69] = _fma_959;
                float _fma_960;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_960) : "f"(k_regs[6]), "f"(vn_6), "f"(S[70]));
                S[70] = _fma_960;
                float _fma_961;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_961) : "f"(k_regs[7]), "f"(vn_6), "f"(S[71]));
                S[71] = _fma_961;
                float _fma_962;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_962) : "f"(k_regs[8]), "f"(vn_6), "f"(S[72]));
                S[72] = _fma_962;
                float _fma_963;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_963) : "f"(k_regs[9]), "f"(vn_6), "f"(S[73]));
                S[73] = _fma_963;
                float _fma_964;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_964) : "f"(k_regs[10]), "f"(vn_6), "f"(S[74]));
                S[74] = _fma_964;
                float _fma_965;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_965) : "f"(k_regs[11]), "f"(vn_6), "f"(S[75]));
                S[75] = _fma_965;
                float _fma_966;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_966) : "f"(k_regs[12]), "f"(vn_6), "f"(S[76]));
                S[76] = _fma_966;
                float _fma_967;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_967) : "f"(k_regs[13]), "f"(vn_6), "f"(S[77]));
                S[77] = _fma_967;
                float _fma_968;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_968) : "f"(k_regs[14]), "f"(vn_6), "f"(S[78]));
                S[78] = _fma_968;
                float _fma_969;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_969) : "f"(k_regs[15]), "f"(vn_6), "f"(S[79]));
                S[79] = _fma_969;
                float _fma_970;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_970) : "f"(k_regs[16]), "f"(vn_6), "f"(S[80]));
                S[80] = _fma_970;
                float _fma_971;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_971) : "f"(k_regs[17]), "f"(vn_6), "f"(S[81]));
                S[81] = _fma_971;
                float _fma_972;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_972) : "f"(k_regs[18]), "f"(vn_6), "f"(S[82]));
                S[82] = _fma_972;
                float _fma_973;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_973) : "f"(k_regs[19]), "f"(vn_6), "f"(S[83]));
                S[83] = _fma_973;
                float _fma_974;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_974) : "f"(k_regs[20]), "f"(vn_6), "f"(S[84]));
                S[84] = _fma_974;
                float _fma_975;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_975) : "f"(k_regs[21]), "f"(vn_6), "f"(S[85]));
                S[85] = _fma_975;
                float _fma_976;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_976) : "f"(k_regs[22]), "f"(vn_6), "f"(S[86]));
                S[86] = _fma_976;
                float _fma_977;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_977) : "f"(k_regs[23]), "f"(vn_6), "f"(S[87]));
                S[87] = _fma_977;
                float _fma_978;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_978) : "f"(k_regs[24]), "f"(vn_6), "f"(S[88]));
                S[88] = _fma_978;
                float _fma_979;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_979) : "f"(k_regs[25]), "f"(vn_6), "f"(S[89]));
                S[89] = _fma_979;
                float _fma_980;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_980) : "f"(k_regs[26]), "f"(vn_6), "f"(S[90]));
                S[90] = _fma_980;
                float _fma_981;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_981) : "f"(k_regs[27]), "f"(vn_6), "f"(S[91]));
                S[91] = _fma_981;
                float _fma_982;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_982) : "f"(k_regs[28]), "f"(vn_6), "f"(S[92]));
                S[92] = _fma_982;
                float _fma_983;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_983) : "f"(k_regs[29]), "f"(vn_6), "f"(S[93]));
                S[93] = _fma_983;
                float _fma_984;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_984) : "f"(k_regs[30]), "f"(vn_6), "f"(S[94]));
                S[94] = _fma_984;
                float _fma_985;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_985) : "f"(k_regs[31]), "f"(vn_6), "f"(S[95]));
                S[95] = _fma_985;
                float acc0_x_9 = 0.0f;
                float acc0_y_9 = 0.0f;
                float acc1_x_9 = 0.0f;
                float acc1_y_9 = 0.0f;
                float _fma_986;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_986) : "f"(S[96]), "f"(k_regs[0]), "f"(acc0_x_9));
                acc0_x_9 = _fma_986;
                float _fma_987;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_987) : "f"(S[97]), "f"(k_regs[1]), "f"(acc0_y_9));
                acc0_y_9 = _fma_987;
                float _fma_988;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_988) : "f"(S[98]), "f"(k_regs[2]), "f"(acc1_x_9));
                acc1_x_9 = _fma_988;
                float _fma_989;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_989) : "f"(S[99]), "f"(k_regs[3]), "f"(acc1_y_9));
                acc1_y_9 = _fma_989;
                float _fma_990;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_990) : "f"(S[100]), "f"(k_regs[4]), "f"(acc0_x_9));
                acc0_x_9 = _fma_990;
                float _fma_991;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_991) : "f"(S[101]), "f"(k_regs[5]), "f"(acc0_y_9));
                acc0_y_9 = _fma_991;
                float _fma_992;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_992) : "f"(S[102]), "f"(k_regs[6]), "f"(acc1_x_9));
                acc1_x_9 = _fma_992;
                float _fma_993;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_993) : "f"(S[103]), "f"(k_regs[7]), "f"(acc1_y_9));
                acc1_y_9 = _fma_993;
                float _fma_994;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_994) : "f"(S[104]), "f"(k_regs[8]), "f"(acc0_x_9));
                acc0_x_9 = _fma_994;
                float _fma_995;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_995) : "f"(S[105]), "f"(k_regs[9]), "f"(acc0_y_9));
                acc0_y_9 = _fma_995;
                float _fma_996;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_996) : "f"(S[106]), "f"(k_regs[10]), "f"(acc1_x_9));
                acc1_x_9 = _fma_996;
                float _fma_997;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_997) : "f"(S[107]), "f"(k_regs[11]), "f"(acc1_y_9));
                acc1_y_9 = _fma_997;
                float _fma_998;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_998) : "f"(S[108]), "f"(k_regs[12]), "f"(acc0_x_9));
                acc0_x_9 = _fma_998;
                float _fma_999;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_999) : "f"(S[109]), "f"(k_regs[13]), "f"(acc0_y_9));
                acc0_y_9 = _fma_999;
                float _fma_1000;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1000) : "f"(S[110]), "f"(k_regs[14]), "f"(acc1_x_9));
                acc1_x_9 = _fma_1000;
                float _fma_1001;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1001) : "f"(S[111]), "f"(k_regs[15]), "f"(acc1_y_9));
                acc1_y_9 = _fma_1001;
                float _fma_1002;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1002) : "f"(S[112]), "f"(k_regs[16]), "f"(acc0_x_9));
                acc0_x_9 = _fma_1002;
                float _fma_1003;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1003) : "f"(S[113]), "f"(k_regs[17]), "f"(acc0_y_9));
                acc0_y_9 = _fma_1003;
                float _fma_1004;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1004) : "f"(S[114]), "f"(k_regs[18]), "f"(acc1_x_9));
                acc1_x_9 = _fma_1004;
                float _fma_1005;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1005) : "f"(S[115]), "f"(k_regs[19]), "f"(acc1_y_9));
                acc1_y_9 = _fma_1005;
                float _fma_1006;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1006) : "f"(S[116]), "f"(k_regs[20]), "f"(acc0_x_9));
                acc0_x_9 = _fma_1006;
                float _fma_1007;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1007) : "f"(S[117]), "f"(k_regs[21]), "f"(acc0_y_9));
                acc0_y_9 = _fma_1007;
                float _fma_1008;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1008) : "f"(S[118]), "f"(k_regs[22]), "f"(acc1_x_9));
                acc1_x_9 = _fma_1008;
                float _fma_1009;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1009) : "f"(S[119]), "f"(k_regs[23]), "f"(acc1_y_9));
                acc1_y_9 = _fma_1009;
                float _fma_1010;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1010) : "f"(S[120]), "f"(k_regs[24]), "f"(acc0_x_9));
                acc0_x_9 = _fma_1010;
                float _fma_1011;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1011) : "f"(S[121]), "f"(k_regs[25]), "f"(acc0_y_9));
                acc0_y_9 = _fma_1011;
                float _fma_1012;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1012) : "f"(S[122]), "f"(k_regs[26]), "f"(acc1_x_9));
                acc1_x_9 = _fma_1012;
                float _fma_1013;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1013) : "f"(S[123]), "f"(k_regs[27]), "f"(acc1_y_9));
                acc1_y_9 = _fma_1013;
                float _fma_1014;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1014) : "f"(S[124]), "f"(k_regs[28]), "f"(acc0_x_9));
                acc0_x_9 = _fma_1014;
                float _fma_1015;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1015) : "f"(S[125]), "f"(k_regs[29]), "f"(acc0_y_9));
                acc0_y_9 = _fma_1015;
                float _fma_1016;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1016) : "f"(S[126]), "f"(k_regs[30]), "f"(acc1_x_9));
                acc1_x_9 = _fma_1016;
                float _fma_1017;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1017) : "f"(S[127]), "f"(k_regs[31]), "f"(acc1_y_9));
                acc1_y_9 = _fma_1017;
                float _add_1018;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_1018) : "f"(acc0_x_9), "f"(acc1_x_9));
                float _add_1019;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_1019) : "f"(acc0_y_9), "f"(acc1_y_9));
                float dot_9 = _add_1018 + _add_1019;
                float _shfl_xor_sync_1020 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 1);
                dot_9 = dot_9 + _shfl_xor_sync_1020;
                float _shfl_xor_sync_1021 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 2);
                dot_9 = dot_9 + _shfl_xor_sync_1021;
#else
                float2 vn2_6 = make_float2(vn_6, vn_6);
                S2[32] = fma_f32x2_rn_ftz(k2[0], vn2_6, S2[32]);
                S2[33] = fma_f32x2_rn_ftz(k2[1], vn2_6, S2[33]);
                S2[34] = fma_f32x2_rn_ftz(k2[2], vn2_6, S2[34]);
                S2[35] = fma_f32x2_rn_ftz(k2[3], vn2_6, S2[35]);
                S2[36] = fma_f32x2_rn_ftz(k2[4], vn2_6, S2[36]);
                S2[37] = fma_f32x2_rn_ftz(k2[5], vn2_6, S2[37]);
                S2[38] = fma_f32x2_rn_ftz(k2[6], vn2_6, S2[38]);
                S2[39] = fma_f32x2_rn_ftz(k2[7], vn2_6, S2[39]);
                S2[40] = fma_f32x2_rn_ftz(k2[8], vn2_6, S2[40]);
                S2[41] = fma_f32x2_rn_ftz(k2[9], vn2_6, S2[41]);
                S2[42] = fma_f32x2_rn_ftz(k2[10], vn2_6, S2[42]);
                S2[43] = fma_f32x2_rn_ftz(k2[11], vn2_6, S2[43]);
                S2[44] = fma_f32x2_rn_ftz(k2[12], vn2_6, S2[44]);
                S2[45] = fma_f32x2_rn_ftz(k2[13], vn2_6, S2[45]);
                S2[46] = fma_f32x2_rn_ftz(k2[14], vn2_6, S2[46]);
                S2[47] = fma_f32x2_rn_ftz(k2[15], vn2_6, S2[47]);
                float2 acc0_9 = make_float2(0.0f, 0.0f);
                acc0_9 = fma_f32x2_rn_ftz(S2[48], k2[0], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[49], k2[1], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[50], k2[2], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[51], k2[3], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[52], k2[4], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[53], k2[5], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[54], k2[6], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[55], k2[7], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[56], k2[8], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[57], k2[9], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[58], k2[10], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[59], k2[11], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[60], k2[12], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[61], k2[13], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[62], k2[14], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[63], k2[15], acc0_9);
                float dot_9 = acc0_9.x + acc0_9.y;
                float _shfl_xor_sync_192 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 1);
                dot_9 = dot_9 + _shfl_xor_sync_192;
                float _shfl_xor_sync_193 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 2);
                dot_9 = dot_9 + _shfl_xor_sync_193;
#endif
                float vn_7 = (v_regs[3] - dot_9) * beta_1;
#if __CUDA_ARCH__ == 900
                float _fma_1022;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1022) : "f"(k_regs[0]), "f"(vn_7), "f"(S[96]));
                S[96] = _fma_1022;
                float _fma_1023;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1023) : "f"(k_regs[1]), "f"(vn_7), "f"(S[97]));
                S[97] = _fma_1023;
                float _fma_1024;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1024) : "f"(k_regs[2]), "f"(vn_7), "f"(S[98]));
                S[98] = _fma_1024;
                float _fma_1025;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1025) : "f"(k_regs[3]), "f"(vn_7), "f"(S[99]));
                S[99] = _fma_1025;
                float _fma_1026;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1026) : "f"(k_regs[4]), "f"(vn_7), "f"(S[100]));
                S[100] = _fma_1026;
                float _fma_1027;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1027) : "f"(k_regs[5]), "f"(vn_7), "f"(S[101]));
                S[101] = _fma_1027;
                float _fma_1028;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1028) : "f"(k_regs[6]), "f"(vn_7), "f"(S[102]));
                S[102] = _fma_1028;
                float _fma_1029;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1029) : "f"(k_regs[7]), "f"(vn_7), "f"(S[103]));
                S[103] = _fma_1029;
                float _fma_1030;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1030) : "f"(k_regs[8]), "f"(vn_7), "f"(S[104]));
                S[104] = _fma_1030;
                float _fma_1031;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1031) : "f"(k_regs[9]), "f"(vn_7), "f"(S[105]));
                S[105] = _fma_1031;
                float _fma_1032;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1032) : "f"(k_regs[10]), "f"(vn_7), "f"(S[106]));
                S[106] = _fma_1032;
                float _fma_1033;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1033) : "f"(k_regs[11]), "f"(vn_7), "f"(S[107]));
                S[107] = _fma_1033;
                float _fma_1034;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1034) : "f"(k_regs[12]), "f"(vn_7), "f"(S[108]));
                S[108] = _fma_1034;
                float _fma_1035;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1035) : "f"(k_regs[13]), "f"(vn_7), "f"(S[109]));
                S[109] = _fma_1035;
                float _fma_1036;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1036) : "f"(k_regs[14]), "f"(vn_7), "f"(S[110]));
                S[110] = _fma_1036;
                float _fma_1037;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1037) : "f"(k_regs[15]), "f"(vn_7), "f"(S[111]));
                S[111] = _fma_1037;
                float _fma_1038;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1038) : "f"(k_regs[16]), "f"(vn_7), "f"(S[112]));
                S[112] = _fma_1038;
                float _fma_1039;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1039) : "f"(k_regs[17]), "f"(vn_7), "f"(S[113]));
                S[113] = _fma_1039;
                float _fma_1040;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1040) : "f"(k_regs[18]), "f"(vn_7), "f"(S[114]));
                S[114] = _fma_1040;
                float _fma_1041;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1041) : "f"(k_regs[19]), "f"(vn_7), "f"(S[115]));
                S[115] = _fma_1041;
                float _fma_1042;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1042) : "f"(k_regs[20]), "f"(vn_7), "f"(S[116]));
                S[116] = _fma_1042;
                float _fma_1043;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1043) : "f"(k_regs[21]), "f"(vn_7), "f"(S[117]));
                S[117] = _fma_1043;
                float _fma_1044;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1044) : "f"(k_regs[22]), "f"(vn_7), "f"(S[118]));
                S[118] = _fma_1044;
                float _fma_1045;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1045) : "f"(k_regs[23]), "f"(vn_7), "f"(S[119]));
                S[119] = _fma_1045;
                float _fma_1046;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1046) : "f"(k_regs[24]), "f"(vn_7), "f"(S[120]));
                S[120] = _fma_1046;
                float _fma_1047;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1047) : "f"(k_regs[25]), "f"(vn_7), "f"(S[121]));
                S[121] = _fma_1047;
                float _fma_1048;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1048) : "f"(k_regs[26]), "f"(vn_7), "f"(S[122]));
                S[122] = _fma_1048;
                float _fma_1049;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1049) : "f"(k_regs[27]), "f"(vn_7), "f"(S[123]));
                S[123] = _fma_1049;
                float _fma_1050;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1050) : "f"(k_regs[28]), "f"(vn_7), "f"(S[124]));
                S[124] = _fma_1050;
                float _fma_1051;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1051) : "f"(k_regs[29]), "f"(vn_7), "f"(S[125]));
                S[125] = _fma_1051;
                float _fma_1052;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1052) : "f"(k_regs[30]), "f"(vn_7), "f"(S[126]));
                S[126] = _fma_1052;
                float _fma_1053;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1053) : "f"(k_regs[31]), "f"(vn_7), "f"(S[127]));
                S[127] = _fma_1053;
#else
                float2 vn2_7 = make_float2(vn_7, vn_7);
                S2[48] = fma_f32x2_rn_ftz(k2[0], vn2_7, S2[48]);
                S2[49] = fma_f32x2_rn_ftz(k2[1], vn2_7, S2[49]);
                S2[50] = fma_f32x2_rn_ftz(k2[2], vn2_7, S2[50]);
                S2[51] = fma_f32x2_rn_ftz(k2[3], vn2_7, S2[51]);
                S2[52] = fma_f32x2_rn_ftz(k2[4], vn2_7, S2[52]);
                S2[53] = fma_f32x2_rn_ftz(k2[5], vn2_7, S2[53]);
                S2[54] = fma_f32x2_rn_ftz(k2[6], vn2_7, S2[54]);
                S2[55] = fma_f32x2_rn_ftz(k2[7], vn2_7, S2[55]);
                S2[56] = fma_f32x2_rn_ftz(k2[8], vn2_7, S2[56]);
                S2[57] = fma_f32x2_rn_ftz(k2[9], vn2_7, S2[57]);
                S2[58] = fma_f32x2_rn_ftz(k2[10], vn2_7, S2[58]);
                S2[59] = fma_f32x2_rn_ftz(k2[11], vn2_7, S2[59]);
                S2[60] = fma_f32x2_rn_ftz(k2[12], vn2_7, S2[60]);
                S2[61] = fma_f32x2_rn_ftz(k2[13], vn2_7, S2[61]);
                S2[62] = fma_f32x2_rn_ftz(k2[14], vn2_7, S2[62]);
                S2[63] = fma_f32x2_rn_ftz(k2[15], vn2_7, S2[63]);
#endif
                unsigned int qbase_1 = sQ_addr + (unsigned int)(t_2 * 512) + lane_byte;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(qbase_1));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(qbase_1 + 64));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(qbase_1 + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(qbase_1 + 192));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[16])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(16) + 3]))
                    : "r"(qbase_1 + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[20])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(20) + 3]))
                    : "r"(qbase_1 + 320));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[24])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(24) + 3]))
                    : "r"(qbase_1 + 384));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[28])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(28) + 3]))
                    : "r"(qbase_1 + 448));
#if __CUDA_ARCH__ == 900
                float acc0_x_10 = 0.0f;
                float acc0_y_10 = 0.0f;
                float acc1_x_10 = 0.0f;
                float acc1_y_10 = 0.0f;
                float _fma_1054;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1054) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_10));
                acc0_x_10 = _fma_1054;
                float _fma_1055;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1055) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_10));
                acc0_y_10 = _fma_1055;
                float _fma_1056;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1056) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_10));
                acc1_x_10 = _fma_1056;
                float _fma_1057;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1057) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_10));
                acc1_y_10 = _fma_1057;
                float _fma_1058;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1058) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_10));
                acc0_x_10 = _fma_1058;
                float _fma_1059;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1059) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_10));
                acc0_y_10 = _fma_1059;
                float _fma_1060;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1060) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_10));
                acc1_x_10 = _fma_1060;
                float _fma_1061;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1061) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_10));
                acc1_y_10 = _fma_1061;
                float _fma_1062;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1062) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_10));
                acc0_x_10 = _fma_1062;
                float _fma_1063;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1063) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_10));
                acc0_y_10 = _fma_1063;
                float _fma_1064;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1064) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_10));
                acc1_x_10 = _fma_1064;
                float _fma_1065;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1065) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_10));
                acc1_y_10 = _fma_1065;
                float _fma_1066;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1066) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_10));
                acc0_x_10 = _fma_1066;
                float _fma_1067;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1067) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_10));
                acc0_y_10 = _fma_1067;
                float _fma_1068;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1068) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_10));
                acc1_x_10 = _fma_1068;
                float _fma_1069;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1069) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_10));
                acc1_y_10 = _fma_1069;
                float _fma_1070;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1070) : "f"(S[16]), "f"(k_regs[16]), "f"(acc0_x_10));
                acc0_x_10 = _fma_1070;
                float _fma_1071;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1071) : "f"(S[17]), "f"(k_regs[17]), "f"(acc0_y_10));
                acc0_y_10 = _fma_1071;
                float _fma_1072;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1072) : "f"(S[18]), "f"(k_regs[18]), "f"(acc1_x_10));
                acc1_x_10 = _fma_1072;
                float _fma_1073;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1073) : "f"(S[19]), "f"(k_regs[19]), "f"(acc1_y_10));
                acc1_y_10 = _fma_1073;
                float _fma_1074;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1074) : "f"(S[20]), "f"(k_regs[20]), "f"(acc0_x_10));
                acc0_x_10 = _fma_1074;
                float _fma_1075;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1075) : "f"(S[21]), "f"(k_regs[21]), "f"(acc0_y_10));
                acc0_y_10 = _fma_1075;
                float _fma_1076;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1076) : "f"(S[22]), "f"(k_regs[22]), "f"(acc1_x_10));
                acc1_x_10 = _fma_1076;
                float _fma_1077;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1077) : "f"(S[23]), "f"(k_regs[23]), "f"(acc1_y_10));
                acc1_y_10 = _fma_1077;
                float _fma_1078;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1078) : "f"(S[24]), "f"(k_regs[24]), "f"(acc0_x_10));
                acc0_x_10 = _fma_1078;
                float _fma_1079;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1079) : "f"(S[25]), "f"(k_regs[25]), "f"(acc0_y_10));
                acc0_y_10 = _fma_1079;
                float _fma_1080;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1080) : "f"(S[26]), "f"(k_regs[26]), "f"(acc1_x_10));
                acc1_x_10 = _fma_1080;
                float _fma_1081;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1081) : "f"(S[27]), "f"(k_regs[27]), "f"(acc1_y_10));
                acc1_y_10 = _fma_1081;
                float _fma_1082;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1082) : "f"(S[28]), "f"(k_regs[28]), "f"(acc0_x_10));
                acc0_x_10 = _fma_1082;
                float _fma_1083;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1083) : "f"(S[29]), "f"(k_regs[29]), "f"(acc0_y_10));
                acc0_y_10 = _fma_1083;
                float _fma_1084;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1084) : "f"(S[30]), "f"(k_regs[30]), "f"(acc1_x_10));
                acc1_x_10 = _fma_1084;
                float _fma_1085;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1085) : "f"(S[31]), "f"(k_regs[31]), "f"(acc1_y_10));
                acc1_y_10 = _fma_1085;
                float _add_1086;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_1086) : "f"(acc0_x_10), "f"(acc1_x_10));
                float _add_1087;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_1087) : "f"(acc0_y_10), "f"(acc1_y_10));
                float dot_10 = _add_1086 + _add_1087;
                float _shfl_xor_sync_1088 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 1);
                dot_10 = dot_10 + _shfl_xor_sync_1088;
                float _shfl_xor_sync_1089 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 2);
                dot_10 = dot_10 + _shfl_xor_sync_1089;
#else
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                k2[8] = make_float2(k_regs[16], k_regs[17]);
                k2[9] = make_float2(k_regs[18], k_regs[19]);
                k2[10] = make_float2(k_regs[20], k_regs[21]);
                k2[11] = make_float2(k_regs[22], k_regs[23]);
                k2[12] = make_float2(k_regs[24], k_regs[25]);
                k2[13] = make_float2(k_regs[26], k_regs[27]);
                k2[14] = make_float2(k_regs[28], k_regs[29]);
                k2[15] = make_float2(k_regs[30], k_regs[31]);
                float2 acc0_10 = make_float2(0.0f, 0.0f);
                acc0_10 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[8], k2[8], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[9], k2[9], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[10], k2[10], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[11], k2[11], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[12], k2[12], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[13], k2[13], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[14], k2[14], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[15], k2[15], acc0_10);
                float dot_10 = acc0_10.x + acc0_10.y;
                float _shfl_xor_sync_194 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 1);
                dot_10 = dot_10 + _shfl_xor_sync_194;
                float _shfl_xor_sync_195 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 2);
                dot_10 = dot_10 + _shfl_xor_sync_195;
#endif
                o4[0] = dot_10;
#if __CUDA_ARCH__ == 900
                float acc0_x_11 = 0.0f;
                float acc0_y_11 = 0.0f;
                float acc1_x_11 = 0.0f;
                float acc1_y_11 = 0.0f;
                float _fma_1090;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1090) : "f"(S[32]), "f"(k_regs[0]), "f"(acc0_x_11));
                acc0_x_11 = _fma_1090;
                float _fma_1091;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1091) : "f"(S[33]), "f"(k_regs[1]), "f"(acc0_y_11));
                acc0_y_11 = _fma_1091;
                float _fma_1092;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1092) : "f"(S[34]), "f"(k_regs[2]), "f"(acc1_x_11));
                acc1_x_11 = _fma_1092;
                float _fma_1093;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1093) : "f"(S[35]), "f"(k_regs[3]), "f"(acc1_y_11));
                acc1_y_11 = _fma_1093;
                float _fma_1094;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1094) : "f"(S[36]), "f"(k_regs[4]), "f"(acc0_x_11));
                acc0_x_11 = _fma_1094;
                float _fma_1095;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1095) : "f"(S[37]), "f"(k_regs[5]), "f"(acc0_y_11));
                acc0_y_11 = _fma_1095;
                float _fma_1096;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1096) : "f"(S[38]), "f"(k_regs[6]), "f"(acc1_x_11));
                acc1_x_11 = _fma_1096;
                float _fma_1097;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1097) : "f"(S[39]), "f"(k_regs[7]), "f"(acc1_y_11));
                acc1_y_11 = _fma_1097;
                float _fma_1098;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1098) : "f"(S[40]), "f"(k_regs[8]), "f"(acc0_x_11));
                acc0_x_11 = _fma_1098;
                float _fma_1099;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1099) : "f"(S[41]), "f"(k_regs[9]), "f"(acc0_y_11));
                acc0_y_11 = _fma_1099;
                float _fma_1100;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1100) : "f"(S[42]), "f"(k_regs[10]), "f"(acc1_x_11));
                acc1_x_11 = _fma_1100;
                float _fma_1101;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1101) : "f"(S[43]), "f"(k_regs[11]), "f"(acc1_y_11));
                acc1_y_11 = _fma_1101;
                float _fma_1102;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1102) : "f"(S[44]), "f"(k_regs[12]), "f"(acc0_x_11));
                acc0_x_11 = _fma_1102;
                float _fma_1103;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1103) : "f"(S[45]), "f"(k_regs[13]), "f"(acc0_y_11));
                acc0_y_11 = _fma_1103;
                float _fma_1104;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1104) : "f"(S[46]), "f"(k_regs[14]), "f"(acc1_x_11));
                acc1_x_11 = _fma_1104;
                float _fma_1105;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1105) : "f"(S[47]), "f"(k_regs[15]), "f"(acc1_y_11));
                acc1_y_11 = _fma_1105;
                float _fma_1106;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1106) : "f"(S[48]), "f"(k_regs[16]), "f"(acc0_x_11));
                acc0_x_11 = _fma_1106;
                float _fma_1107;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1107) : "f"(S[49]), "f"(k_regs[17]), "f"(acc0_y_11));
                acc0_y_11 = _fma_1107;
                float _fma_1108;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1108) : "f"(S[50]), "f"(k_regs[18]), "f"(acc1_x_11));
                acc1_x_11 = _fma_1108;
                float _fma_1109;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1109) : "f"(S[51]), "f"(k_regs[19]), "f"(acc1_y_11));
                acc1_y_11 = _fma_1109;
                float _fma_1110;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1110) : "f"(S[52]), "f"(k_regs[20]), "f"(acc0_x_11));
                acc0_x_11 = _fma_1110;
                float _fma_1111;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1111) : "f"(S[53]), "f"(k_regs[21]), "f"(acc0_y_11));
                acc0_y_11 = _fma_1111;
                float _fma_1112;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1112) : "f"(S[54]), "f"(k_regs[22]), "f"(acc1_x_11));
                acc1_x_11 = _fma_1112;
                float _fma_1113;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1113) : "f"(S[55]), "f"(k_regs[23]), "f"(acc1_y_11));
                acc1_y_11 = _fma_1113;
                float _fma_1114;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1114) : "f"(S[56]), "f"(k_regs[24]), "f"(acc0_x_11));
                acc0_x_11 = _fma_1114;
                float _fma_1115;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1115) : "f"(S[57]), "f"(k_regs[25]), "f"(acc0_y_11));
                acc0_y_11 = _fma_1115;
                float _fma_1116;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1116) : "f"(S[58]), "f"(k_regs[26]), "f"(acc1_x_11));
                acc1_x_11 = _fma_1116;
                float _fma_1117;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1117) : "f"(S[59]), "f"(k_regs[27]), "f"(acc1_y_11));
                acc1_y_11 = _fma_1117;
                float _fma_1118;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1118) : "f"(S[60]), "f"(k_regs[28]), "f"(acc0_x_11));
                acc0_x_11 = _fma_1118;
                float _fma_1119;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1119) : "f"(S[61]), "f"(k_regs[29]), "f"(acc0_y_11));
                acc0_y_11 = _fma_1119;
                float _fma_1120;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1120) : "f"(S[62]), "f"(k_regs[30]), "f"(acc1_x_11));
                acc1_x_11 = _fma_1120;
                float _fma_1121;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1121) : "f"(S[63]), "f"(k_regs[31]), "f"(acc1_y_11));
                acc1_y_11 = _fma_1121;
                float _add_1122;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_1122) : "f"(acc0_x_11), "f"(acc1_x_11));
                float _add_1123;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_1123) : "f"(acc0_y_11), "f"(acc1_y_11));
                float dot_11 = _add_1122 + _add_1123;
                float _shfl_xor_sync_1124 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 1);
                dot_11 = dot_11 + _shfl_xor_sync_1124;
                float _shfl_xor_sync_1125 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 2);
                dot_11 = dot_11 + _shfl_xor_sync_1125;
#else
                float2 acc0_11 = make_float2(0.0f, 0.0f);
                acc0_11 = fma_f32x2_rn_ftz(S2[16], k2[0], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[17], k2[1], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[18], k2[2], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[19], k2[3], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[20], k2[4], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[21], k2[5], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[22], k2[6], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[23], k2[7], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[24], k2[8], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[25], k2[9], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[26], k2[10], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[27], k2[11], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[28], k2[12], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[29], k2[13], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[30], k2[14], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[31], k2[15], acc0_11);
                float dot_11 = acc0_11.x + acc0_11.y;
                float _shfl_xor_sync_196 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 1);
                dot_11 = dot_11 + _shfl_xor_sync_196;
                float _shfl_xor_sync_197 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 2);
                dot_11 = dot_11 + _shfl_xor_sync_197;
#endif
                o4[1] = dot_11;
#if __CUDA_ARCH__ == 900
                float acc0_x_12 = 0.0f;
                float acc0_y_12 = 0.0f;
                float acc1_x_12 = 0.0f;
                float acc1_y_12 = 0.0f;
                float _fma_1126;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1126) : "f"(S[64]), "f"(k_regs[0]), "f"(acc0_x_12));
                acc0_x_12 = _fma_1126;
                float _fma_1127;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1127) : "f"(S[65]), "f"(k_regs[1]), "f"(acc0_y_12));
                acc0_y_12 = _fma_1127;
                float _fma_1128;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1128) : "f"(S[66]), "f"(k_regs[2]), "f"(acc1_x_12));
                acc1_x_12 = _fma_1128;
                float _fma_1129;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1129) : "f"(S[67]), "f"(k_regs[3]), "f"(acc1_y_12));
                acc1_y_12 = _fma_1129;
                float _fma_1130;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1130) : "f"(S[68]), "f"(k_regs[4]), "f"(acc0_x_12));
                acc0_x_12 = _fma_1130;
                float _fma_1131;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1131) : "f"(S[69]), "f"(k_regs[5]), "f"(acc0_y_12));
                acc0_y_12 = _fma_1131;
                float _fma_1132;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1132) : "f"(S[70]), "f"(k_regs[6]), "f"(acc1_x_12));
                acc1_x_12 = _fma_1132;
                float _fma_1133;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1133) : "f"(S[71]), "f"(k_regs[7]), "f"(acc1_y_12));
                acc1_y_12 = _fma_1133;
                float _fma_1134;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1134) : "f"(S[72]), "f"(k_regs[8]), "f"(acc0_x_12));
                acc0_x_12 = _fma_1134;
                float _fma_1135;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1135) : "f"(S[73]), "f"(k_regs[9]), "f"(acc0_y_12));
                acc0_y_12 = _fma_1135;
                float _fma_1136;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1136) : "f"(S[74]), "f"(k_regs[10]), "f"(acc1_x_12));
                acc1_x_12 = _fma_1136;
                float _fma_1137;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1137) : "f"(S[75]), "f"(k_regs[11]), "f"(acc1_y_12));
                acc1_y_12 = _fma_1137;
                float _fma_1138;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1138) : "f"(S[76]), "f"(k_regs[12]), "f"(acc0_x_12));
                acc0_x_12 = _fma_1138;
                float _fma_1139;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1139) : "f"(S[77]), "f"(k_regs[13]), "f"(acc0_y_12));
                acc0_y_12 = _fma_1139;
                float _fma_1140;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1140) : "f"(S[78]), "f"(k_regs[14]), "f"(acc1_x_12));
                acc1_x_12 = _fma_1140;
                float _fma_1141;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1141) : "f"(S[79]), "f"(k_regs[15]), "f"(acc1_y_12));
                acc1_y_12 = _fma_1141;
                float _fma_1142;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1142) : "f"(S[80]), "f"(k_regs[16]), "f"(acc0_x_12));
                acc0_x_12 = _fma_1142;
                float _fma_1143;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1143) : "f"(S[81]), "f"(k_regs[17]), "f"(acc0_y_12));
                acc0_y_12 = _fma_1143;
                float _fma_1144;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1144) : "f"(S[82]), "f"(k_regs[18]), "f"(acc1_x_12));
                acc1_x_12 = _fma_1144;
                float _fma_1145;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1145) : "f"(S[83]), "f"(k_regs[19]), "f"(acc1_y_12));
                acc1_y_12 = _fma_1145;
                float _fma_1146;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1146) : "f"(S[84]), "f"(k_regs[20]), "f"(acc0_x_12));
                acc0_x_12 = _fma_1146;
                float _fma_1147;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1147) : "f"(S[85]), "f"(k_regs[21]), "f"(acc0_y_12));
                acc0_y_12 = _fma_1147;
                float _fma_1148;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1148) : "f"(S[86]), "f"(k_regs[22]), "f"(acc1_x_12));
                acc1_x_12 = _fma_1148;
                float _fma_1149;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1149) : "f"(S[87]), "f"(k_regs[23]), "f"(acc1_y_12));
                acc1_y_12 = _fma_1149;
                float _fma_1150;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1150) : "f"(S[88]), "f"(k_regs[24]), "f"(acc0_x_12));
                acc0_x_12 = _fma_1150;
                float _fma_1151;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1151) : "f"(S[89]), "f"(k_regs[25]), "f"(acc0_y_12));
                acc0_y_12 = _fma_1151;
                float _fma_1152;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1152) : "f"(S[90]), "f"(k_regs[26]), "f"(acc1_x_12));
                acc1_x_12 = _fma_1152;
                float _fma_1153;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1153) : "f"(S[91]), "f"(k_regs[27]), "f"(acc1_y_12));
                acc1_y_12 = _fma_1153;
                float _fma_1154;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1154) : "f"(S[92]), "f"(k_regs[28]), "f"(acc0_x_12));
                acc0_x_12 = _fma_1154;
                float _fma_1155;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1155) : "f"(S[93]), "f"(k_regs[29]), "f"(acc0_y_12));
                acc0_y_12 = _fma_1155;
                float _fma_1156;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1156) : "f"(S[94]), "f"(k_regs[30]), "f"(acc1_x_12));
                acc1_x_12 = _fma_1156;
                float _fma_1157;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1157) : "f"(S[95]), "f"(k_regs[31]), "f"(acc1_y_12));
                acc1_y_12 = _fma_1157;
                float _add_1158;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_1158) : "f"(acc0_x_12), "f"(acc1_x_12));
                float _add_1159;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_1159) : "f"(acc0_y_12), "f"(acc1_y_12));
                float dot_12 = _add_1158 + _add_1159;
                float _shfl_xor_sync_1160 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 1);
                dot_12 = dot_12 + _shfl_xor_sync_1160;
                float _shfl_xor_sync_1161 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 2);
                dot_12 = dot_12 + _shfl_xor_sync_1161;
#else
                float2 acc0_12 = make_float2(0.0f, 0.0f);
                acc0_12 = fma_f32x2_rn_ftz(S2[32], k2[0], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[33], k2[1], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[34], k2[2], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[35], k2[3], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[36], k2[4], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[37], k2[5], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[38], k2[6], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[39], k2[7], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[40], k2[8], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[41], k2[9], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[42], k2[10], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[43], k2[11], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[44], k2[12], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[45], k2[13], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[46], k2[14], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[47], k2[15], acc0_12);
                float dot_12 = acc0_12.x + acc0_12.y;
                float _shfl_xor_sync_198 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 1);
                dot_12 = dot_12 + _shfl_xor_sync_198;
                float _shfl_xor_sync_199 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 2);
                dot_12 = dot_12 + _shfl_xor_sync_199;
#endif
                o4[2] = dot_12;
#if __CUDA_ARCH__ == 900
                float acc0_x_13 = 0.0f;
                float acc0_y_13 = 0.0f;
                float acc1_x_13 = 0.0f;
                float acc1_y_13 = 0.0f;
                float _fma_1162;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1162) : "f"(S[96]), "f"(k_regs[0]), "f"(acc0_x_13));
                acc0_x_13 = _fma_1162;
                float _fma_1163;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1163) : "f"(S[97]), "f"(k_regs[1]), "f"(acc0_y_13));
                acc0_y_13 = _fma_1163;
                float _fma_1164;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1164) : "f"(S[98]), "f"(k_regs[2]), "f"(acc1_x_13));
                acc1_x_13 = _fma_1164;
                float _fma_1165;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1165) : "f"(S[99]), "f"(k_regs[3]), "f"(acc1_y_13));
                acc1_y_13 = _fma_1165;
                float _fma_1166;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1166) : "f"(S[100]), "f"(k_regs[4]), "f"(acc0_x_13));
                acc0_x_13 = _fma_1166;
                float _fma_1167;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1167) : "f"(S[101]), "f"(k_regs[5]), "f"(acc0_y_13));
                acc0_y_13 = _fma_1167;
                float _fma_1168;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1168) : "f"(S[102]), "f"(k_regs[6]), "f"(acc1_x_13));
                acc1_x_13 = _fma_1168;
                float _fma_1169;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1169) : "f"(S[103]), "f"(k_regs[7]), "f"(acc1_y_13));
                acc1_y_13 = _fma_1169;
                float _fma_1170;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1170) : "f"(S[104]), "f"(k_regs[8]), "f"(acc0_x_13));
                acc0_x_13 = _fma_1170;
                float _fma_1171;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1171) : "f"(S[105]), "f"(k_regs[9]), "f"(acc0_y_13));
                acc0_y_13 = _fma_1171;
                float _fma_1172;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1172) : "f"(S[106]), "f"(k_regs[10]), "f"(acc1_x_13));
                acc1_x_13 = _fma_1172;
                float _fma_1173;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1173) : "f"(S[107]), "f"(k_regs[11]), "f"(acc1_y_13));
                acc1_y_13 = _fma_1173;
                float _fma_1174;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1174) : "f"(S[108]), "f"(k_regs[12]), "f"(acc0_x_13));
                acc0_x_13 = _fma_1174;
                float _fma_1175;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1175) : "f"(S[109]), "f"(k_regs[13]), "f"(acc0_y_13));
                acc0_y_13 = _fma_1175;
                float _fma_1176;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1176) : "f"(S[110]), "f"(k_regs[14]), "f"(acc1_x_13));
                acc1_x_13 = _fma_1176;
                float _fma_1177;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1177) : "f"(S[111]), "f"(k_regs[15]), "f"(acc1_y_13));
                acc1_y_13 = _fma_1177;
                float _fma_1178;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1178) : "f"(S[112]), "f"(k_regs[16]), "f"(acc0_x_13));
                acc0_x_13 = _fma_1178;
                float _fma_1179;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1179) : "f"(S[113]), "f"(k_regs[17]), "f"(acc0_y_13));
                acc0_y_13 = _fma_1179;
                float _fma_1180;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1180) : "f"(S[114]), "f"(k_regs[18]), "f"(acc1_x_13));
                acc1_x_13 = _fma_1180;
                float _fma_1181;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1181) : "f"(S[115]), "f"(k_regs[19]), "f"(acc1_y_13));
                acc1_y_13 = _fma_1181;
                float _fma_1182;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1182) : "f"(S[116]), "f"(k_regs[20]), "f"(acc0_x_13));
                acc0_x_13 = _fma_1182;
                float _fma_1183;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1183) : "f"(S[117]), "f"(k_regs[21]), "f"(acc0_y_13));
                acc0_y_13 = _fma_1183;
                float _fma_1184;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1184) : "f"(S[118]), "f"(k_regs[22]), "f"(acc1_x_13));
                acc1_x_13 = _fma_1184;
                float _fma_1185;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1185) : "f"(S[119]), "f"(k_regs[23]), "f"(acc1_y_13));
                acc1_y_13 = _fma_1185;
                float _fma_1186;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1186) : "f"(S[120]), "f"(k_regs[24]), "f"(acc0_x_13));
                acc0_x_13 = _fma_1186;
                float _fma_1187;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1187) : "f"(S[121]), "f"(k_regs[25]), "f"(acc0_y_13));
                acc0_y_13 = _fma_1187;
                float _fma_1188;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1188) : "f"(S[122]), "f"(k_regs[26]), "f"(acc1_x_13));
                acc1_x_13 = _fma_1188;
                float _fma_1189;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1189) : "f"(S[123]), "f"(k_regs[27]), "f"(acc1_y_13));
                acc1_y_13 = _fma_1189;
                float _fma_1190;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1190) : "f"(S[124]), "f"(k_regs[28]), "f"(acc0_x_13));
                acc0_x_13 = _fma_1190;
                float _fma_1191;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1191) : "f"(S[125]), "f"(k_regs[29]), "f"(acc0_y_13));
                acc0_y_13 = _fma_1191;
                float _fma_1192;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1192) : "f"(S[126]), "f"(k_regs[30]), "f"(acc1_x_13));
                acc1_x_13 = _fma_1192;
                float _fma_1193;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1193) : "f"(S[127]), "f"(k_regs[31]), "f"(acc1_y_13));
                acc1_y_13 = _fma_1193;
                float _add_1194;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_1194) : "f"(acc0_x_13), "f"(acc1_x_13));
                float _add_1195;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_1195) : "f"(acc0_y_13), "f"(acc1_y_13));
                float dot_13 = _add_1194 + _add_1195;
                float _shfl_xor_sync_1196 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 1);
                dot_13 = dot_13 + _shfl_xor_sync_1196;
                float _shfl_xor_sync_1197 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 2);
                dot_13 = dot_13 + _shfl_xor_sync_1197;
#else
                float2 acc0_13 = make_float2(0.0f, 0.0f);
                acc0_13 = fma_f32x2_rn_ftz(S2[48], k2[0], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[49], k2[1], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[50], k2[2], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[51], k2[3], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[52], k2[4], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[53], k2[5], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[54], k2[6], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[55], k2[7], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[56], k2[8], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[57], k2[9], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[58], k2[10], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[59], k2[11], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[60], k2[12], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[61], k2[13], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[62], k2[14], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[63], k2[15], acc0_13);
                float dot_13 = acc0_13.x + acc0_13.y;
                float _shfl_xor_sync_200 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 1);
                dot_13 = dot_13 + _shfl_xor_sync_200;
                float _shfl_xor_sync_201 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 2);
                dot_13 = dot_13 + _shfl_xor_sync_201;
#endif
                o4[3] = dot_13;
                if (u == 0) {
                    long long out_index_1 = (long long)n * out_stride_p0 + (long long)t_2 * out_stride_p1 + (long long)h * out_stride_p2;
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
                    uint32_t _read_special_reg_1198;
                    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(_read_special_reg_1198));
                    int tid_r = (int)_read_special_reg_1198;
#else
                    uint32_t _read_special_reg_202;
                    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(_read_special_reg_202));
                    int tid_r = (int)_read_special_reg_202;
#endif
                    long long noise_at = read_slot_head + (long long)(tid_r / 4 * 4) * state_stride_p2 + (long long)((tid_r - tid_r / 4 * 4) * 4);
                    if (philox_rounds != 0) {
#if __CUDA_ARCH__ == 900
                        unsigned int key0_3 = seed_lo;
                        unsigned int key1_3 = seed_hi;
#else
                        unsigned int key0_1 = seed_lo;
                        unsigned int key1_1 = seed_hi;
#endif
                        long long o_16 = noise_at;
                        unsigned long long p0_16 = (unsigned long long)(unsigned int)o_16 * 3528531795ull;
#if __CUDA_ARCH__ == 900
                        h_noise[0] = (unsigned int)(o_16 >> 32ll) ^ key0_3;
                        h_noise[2] = (unsigned int)(p0_16 >> 32ull) ^ key1_3;
#else
                        h_noise[0] = (unsigned int)(o_16 >> 32ll) ^ key0_1;
                        h_noise[2] = (unsigned int)(p0_16 >> 32ull) ^ key1_1;
#endif
                        h_noise[3] = (unsigned int)p0_16;
                        long long o_17 = noise_at + 16ll;
                        unsigned long long p0_17 = (unsigned long long)(unsigned int)o_17 * 3528531795ull;
#if __CUDA_ARCH__ == 900
                        h_noise[4] = (unsigned int)(o_17 >> 32ll) ^ key0_3;
                        h_noise[6] = (unsigned int)(p0_17 >> 32ull) ^ key1_3;
#else
                        h_noise[4] = (unsigned int)(o_17 >> 32ll) ^ key0_1;
                        h_noise[6] = (unsigned int)(p0_17 >> 32ull) ^ key1_1;
#endif
                        h_noise[7] = (unsigned int)p0_17;
                        long long o_18 = noise_at + 32ll;
                        unsigned long long p0_18 = (unsigned long long)(unsigned int)o_18 * 3528531795ull;
#if __CUDA_ARCH__ == 900
                        h_noise[8] = (unsigned int)(o_18 >> 32ll) ^ key0_3;
                        h_noise[10] = (unsigned int)(p0_18 >> 32ull) ^ key1_3;
#else
                        h_noise[8] = (unsigned int)(o_18 >> 32ll) ^ key0_1;
                        h_noise[10] = (unsigned int)(p0_18 >> 32ull) ^ key1_1;
#endif
                        h_noise[11] = (unsigned int)p0_18;
                        long long o_19 = noise_at + 48ll;
                        unsigned long long p0_19 = (unsigned long long)(unsigned int)o_19 * 3528531795ull;
#if __CUDA_ARCH__ == 900
                        h_noise[12] = (unsigned int)(o_19 >> 32ll) ^ key0_3;
                        h_noise[14] = (unsigned int)(p0_19 >> 32ull) ^ key1_3;
#else
                        h_noise[12] = (unsigned int)(o_19 >> 32ll) ^ key0_1;
                        h_noise[14] = (unsigned int)(p0_19 >> 32ull) ^ key1_1;
#endif
                        h_noise[15] = (unsigned int)p0_19;
#if __CUDA_ARCH__ == 900
                        key0_3 = key0_3 + 2654435769u;
                        key1_3 = key1_3 + 3144134277u;
#else
                        long long o_20 = noise_at + state_stride_p2;
                        unsigned long long p0_20 = (unsigned long long)(unsigned int)o_20 * 3528531795ull;
                        h_noise[16] = (unsigned int)(o_20 >> 32ll) ^ key0_1;
                        h_noise[18] = (unsigned int)(p0_20 >> 32ull) ^ key1_1;
                        h_noise[19] = (unsigned int)p0_20;
                        long long o_21 = noise_at + state_stride_p2 + 16ll;
                        unsigned long long p0_21 = (unsigned long long)(unsigned int)o_21 * 3528531795ull;
                        h_noise[20] = (unsigned int)(o_21 >> 32ll) ^ key0_1;
                        h_noise[22] = (unsigned int)(p0_21 >> 32ull) ^ key1_1;
                        h_noise[23] = (unsigned int)p0_21;
                        long long o_22 = noise_at + state_stride_p2 + 32ll;
                        unsigned long long p0_22 = (unsigned long long)(unsigned int)o_22 * 3528531795ull;
                        h_noise[24] = (unsigned int)(o_22 >> 32ll) ^ key0_1;
                        h_noise[26] = (unsigned int)(p0_22 >> 32ull) ^ key1_1;
                        h_noise[27] = (unsigned int)p0_22;
                        long long o_23 = noise_at + state_stride_p2 + 48ll;
                        unsigned long long p0_23 = (unsigned long long)(unsigned int)o_23 * 3528531795ull;
                        h_noise[28] = (unsigned int)(o_23 >> 32ll) ^ key0_1;
                        h_noise[30] = (unsigned int)(p0_23 >> 32ull) ^ key1_1;
                        h_noise[31] = (unsigned int)p0_23;
                        long long o_24 = noise_at + 2ll * state_stride_p2;
                        unsigned long long p0_24 = (unsigned long long)(unsigned int)o_24 * 3528531795ull;
                        h_noise[32] = (unsigned int)(o_24 >> 32ll) ^ key0_1;
                        h_noise[34] = (unsigned int)(p0_24 >> 32ull) ^ key1_1;
                        h_noise[35] = (unsigned int)p0_24;
                        long long o_25 = noise_at + 2ll * state_stride_p2 + 16ll;
                        unsigned long long p0_25 = (unsigned long long)(unsigned int)o_25 * 3528531795ull;
                        h_noise[36] = (unsigned int)(o_25 >> 32ll) ^ key0_1;
                        h_noise[38] = (unsigned int)(p0_25 >> 32ull) ^ key1_1;
                        h_noise[39] = (unsigned int)p0_25;
                        long long o_26 = noise_at + 2ll * state_stride_p2 + 32ll;
                        unsigned long long p0_26 = (unsigned long long)(unsigned int)o_26 * 3528531795ull;
                        h_noise[40] = (unsigned int)(o_26 >> 32ll) ^ key0_1;
                        h_noise[42] = (unsigned int)(p0_26 >> 32ull) ^ key1_1;
                        h_noise[43] = (unsigned int)p0_26;
                        long long o_27 = noise_at + 2ll * state_stride_p2 + 48ll;
                        unsigned long long p0_27 = (unsigned long long)(unsigned int)o_27 * 3528531795ull;
                        h_noise[44] = (unsigned int)(o_27 >> 32ll) ^ key0_1;
                        h_noise[46] = (unsigned int)(p0_27 >> 32ull) ^ key1_1;
                        h_noise[47] = (unsigned int)p0_27;
                        long long o_28 = noise_at + 3ll * state_stride_p2;
                        unsigned long long p0_28 = (unsigned long long)(unsigned int)o_28 * 3528531795ull;
                        h_noise[48] = (unsigned int)(o_28 >> 32ll) ^ key0_1;
                        h_noise[50] = (unsigned int)(p0_28 >> 32ull) ^ key1_1;
                        h_noise[51] = (unsigned int)p0_28;
                        long long o_29 = noise_at + 3ll * state_stride_p2 + 16ll;
                        unsigned long long p0_29 = (unsigned long long)(unsigned int)o_29 * 3528531795ull;
                        h_noise[52] = (unsigned int)(o_29 >> 32ll) ^ key0_1;
                        h_noise[54] = (unsigned int)(p0_29 >> 32ull) ^ key1_1;
                        h_noise[55] = (unsigned int)p0_29;
                        long long o_30 = noise_at + 3ll * state_stride_p2 + 32ll;
                        unsigned long long p0_30 = (unsigned long long)(unsigned int)o_30 * 3528531795ull;
                        h_noise[56] = (unsigned int)(o_30 >> 32ll) ^ key0_1;
                        h_noise[58] = (unsigned int)(p0_30 >> 32ull) ^ key1_1;
                        h_noise[59] = (unsigned int)p0_30;
                        long long o_31 = noise_at + 3ll * state_stride_p2 + 48ll;
                        unsigned long long p0_31 = (unsigned long long)(unsigned int)o_31 * 3528531795ull;
                        h_noise[60] = (unsigned int)(o_31 >> 32ll) ^ key0_1;
                        h_noise[62] = (unsigned int)(p0_31 >> 32ull) ^ key1_1;
                        h_noise[63] = (unsigned int)p0_31;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
#endif
                        unsigned long long q0_16 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long q1_16 = (unsigned long long)h_noise[2] * 3449720151ull;
#if __CUDA_ARCH__ == 900
                        h_noise[0] = (unsigned int)(q1_16 >> 32ull) ^ key0_3;
#else
                        h_noise[0] = (unsigned int)(q1_16 >> 32ull) ^ key0_1;
#endif
                        h_noise[1] = (unsigned int)q1_16;
#if __CUDA_ARCH__ == 900
                        h_noise[2] = (unsigned int)(q0_16 >> 32ull) ^ h_noise[3] ^ key1_3;
#else
                        h_noise[2] = (unsigned int)(q0_16 >> 32ull) ^ h_noise[3] ^ key1_1;
#endif
                        h_noise[3] = (unsigned int)q0_16;
                        unsigned long long q0_17 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long q1_17 = (unsigned long long)h_noise[6] * 3449720151ull;
#if __CUDA_ARCH__ == 900
                        h_noise[4] = (unsigned int)(q1_17 >> 32ull) ^ key0_3;
#else
                        h_noise[4] = (unsigned int)(q1_17 >> 32ull) ^ key0_1;
#endif
                        h_noise[5] = (unsigned int)q1_17;
#if __CUDA_ARCH__ == 900
                        h_noise[6] = (unsigned int)(q0_17 >> 32ull) ^ h_noise[7] ^ key1_3;
#else
                        h_noise[6] = (unsigned int)(q0_17 >> 32ull) ^ h_noise[7] ^ key1_1;
#endif
                        h_noise[7] = (unsigned int)q0_17;
                        unsigned long long q0_18 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long q1_18 = (unsigned long long)h_noise[10] * 3449720151ull;
#if __CUDA_ARCH__ == 900
                        h_noise[8] = (unsigned int)(q1_18 >> 32ull) ^ key0_3;
#else
                        h_noise[8] = (unsigned int)(q1_18 >> 32ull) ^ key0_1;
#endif
                        h_noise[9] = (unsigned int)q1_18;
#if __CUDA_ARCH__ == 900
                        h_noise[10] = (unsigned int)(q0_18 >> 32ull) ^ h_noise[11] ^ key1_3;
#else
                        h_noise[10] = (unsigned int)(q0_18 >> 32ull) ^ h_noise[11] ^ key1_1;
#endif
                        h_noise[11] = (unsigned int)q0_18;
                        unsigned long long q0_19 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long q1_19 = (unsigned long long)h_noise[14] * 3449720151ull;
#if __CUDA_ARCH__ == 900
                        h_noise[12] = (unsigned int)(q1_19 >> 32ull) ^ key0_3;
#else
                        h_noise[12] = (unsigned int)(q1_19 >> 32ull) ^ key0_1;
#endif
                        h_noise[13] = (unsigned int)q1_19;
#if __CUDA_ARCH__ == 900
                        h_noise[14] = (unsigned int)(q0_19 >> 32ull) ^ h_noise[15] ^ key1_3;
#else
                        h_noise[14] = (unsigned int)(q0_19 >> 32ull) ^ h_noise[15] ^ key1_1;
#endif
                        h_noise[15] = (unsigned int)q0_19;
#if __CUDA_ARCH__ == 900
                        key0_3 = key0_3 + 2654435769u;
                        key1_3 = key1_3 + 3144134277u;
#else
                        unsigned long long q0_20 = (unsigned long long)h_noise[16] * 3528531795ull;
                        unsigned long long q1_20 = (unsigned long long)h_noise[18] * 3449720151ull;
                        h_noise[16] = (unsigned int)(q1_20 >> 32ull) ^ key0_1;
                        h_noise[17] = (unsigned int)q1_20;
                        h_noise[18] = (unsigned int)(q0_20 >> 32ull) ^ h_noise[19] ^ key1_1;
                        h_noise[19] = (unsigned int)q0_20;
                        unsigned long long q0_21 = (unsigned long long)h_noise[20] * 3528531795ull;
                        unsigned long long q1_21 = (unsigned long long)h_noise[22] * 3449720151ull;
                        h_noise[20] = (unsigned int)(q1_21 >> 32ull) ^ key0_1;
                        h_noise[21] = (unsigned int)q1_21;
                        h_noise[22] = (unsigned int)(q0_21 >> 32ull) ^ h_noise[23] ^ key1_1;
                        h_noise[23] = (unsigned int)q0_21;
                        unsigned long long q0_22 = (unsigned long long)h_noise[24] * 3528531795ull;
                        unsigned long long q1_22 = (unsigned long long)h_noise[26] * 3449720151ull;
                        h_noise[24] = (unsigned int)(q1_22 >> 32ull) ^ key0_1;
                        h_noise[25] = (unsigned int)q1_22;
                        h_noise[26] = (unsigned int)(q0_22 >> 32ull) ^ h_noise[27] ^ key1_1;
                        h_noise[27] = (unsigned int)q0_22;
                        unsigned long long q0_23 = (unsigned long long)h_noise[28] * 3528531795ull;
                        unsigned long long q1_23 = (unsigned long long)h_noise[30] * 3449720151ull;
                        h_noise[28] = (unsigned int)(q1_23 >> 32ull) ^ key0_1;
                        h_noise[29] = (unsigned int)q1_23;
                        h_noise[30] = (unsigned int)(q0_23 >> 32ull) ^ h_noise[31] ^ key1_1;
                        h_noise[31] = (unsigned int)q0_23;
                        unsigned long long q0_24 = (unsigned long long)h_noise[32] * 3528531795ull;
                        unsigned long long q1_24 = (unsigned long long)h_noise[34] * 3449720151ull;
                        h_noise[32] = (unsigned int)(q1_24 >> 32ull) ^ key0_1;
                        h_noise[33] = (unsigned int)q1_24;
                        h_noise[34] = (unsigned int)(q0_24 >> 32ull) ^ h_noise[35] ^ key1_1;
                        h_noise[35] = (unsigned int)q0_24;
                        unsigned long long q0_25 = (unsigned long long)h_noise[36] * 3528531795ull;
                        unsigned long long q1_25 = (unsigned long long)h_noise[38] * 3449720151ull;
                        h_noise[36] = (unsigned int)(q1_25 >> 32ull) ^ key0_1;
                        h_noise[37] = (unsigned int)q1_25;
                        h_noise[38] = (unsigned int)(q0_25 >> 32ull) ^ h_noise[39] ^ key1_1;
                        h_noise[39] = (unsigned int)q0_25;
                        unsigned long long q0_26 = (unsigned long long)h_noise[40] * 3528531795ull;
                        unsigned long long q1_26 = (unsigned long long)h_noise[42] * 3449720151ull;
                        h_noise[40] = (unsigned int)(q1_26 >> 32ull) ^ key0_1;
                        h_noise[41] = (unsigned int)q1_26;
                        h_noise[42] = (unsigned int)(q0_26 >> 32ull) ^ h_noise[43] ^ key1_1;
                        h_noise[43] = (unsigned int)q0_26;
                        unsigned long long q0_27 = (unsigned long long)h_noise[44] * 3528531795ull;
                        unsigned long long q1_27 = (unsigned long long)h_noise[46] * 3449720151ull;
                        h_noise[44] = (unsigned int)(q1_27 >> 32ull) ^ key0_1;
                        h_noise[45] = (unsigned int)q1_27;
                        h_noise[46] = (unsigned int)(q0_27 >> 32ull) ^ h_noise[47] ^ key1_1;
                        h_noise[47] = (unsigned int)q0_27;
                        unsigned long long q0_28 = (unsigned long long)h_noise[48] * 3528531795ull;
                        unsigned long long q1_28 = (unsigned long long)h_noise[50] * 3449720151ull;
                        h_noise[48] = (unsigned int)(q1_28 >> 32ull) ^ key0_1;
                        h_noise[49] = (unsigned int)q1_28;
                        h_noise[50] = (unsigned int)(q0_28 >> 32ull) ^ h_noise[51] ^ key1_1;
                        h_noise[51] = (unsigned int)q0_28;
                        unsigned long long q0_29 = (unsigned long long)h_noise[52] * 3528531795ull;
                        unsigned long long q1_29 = (unsigned long long)h_noise[54] * 3449720151ull;
                        h_noise[52] = (unsigned int)(q1_29 >> 32ull) ^ key0_1;
                        h_noise[53] = (unsigned int)q1_29;
                        h_noise[54] = (unsigned int)(q0_29 >> 32ull) ^ h_noise[55] ^ key1_1;
                        h_noise[55] = (unsigned int)q0_29;
                        unsigned long long q0_30 = (unsigned long long)h_noise[56] * 3528531795ull;
                        unsigned long long q1_30 = (unsigned long long)h_noise[58] * 3449720151ull;
                        h_noise[56] = (unsigned int)(q1_30 >> 32ull) ^ key0_1;
                        h_noise[57] = (unsigned int)q1_30;
                        h_noise[58] = (unsigned int)(q0_30 >> 32ull) ^ h_noise[59] ^ key1_1;
                        h_noise[59] = (unsigned int)q0_30;
                        unsigned long long q0_31 = (unsigned long long)h_noise[60] * 3528531795ull;
                        unsigned long long q1_31 = (unsigned long long)h_noise[62] * 3449720151ull;
                        h_noise[60] = (unsigned int)(q1_31 >> 32ull) ^ key0_1;
                        h_noise[61] = (unsigned int)q1_31;
                        h_noise[62] = (unsigned int)(q0_31 >> 32ull) ^ h_noise[63] ^ key1_1;
                        h_noise[63] = (unsigned int)q0_31;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
#endif
                        unsigned long long s0_48 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_48 = (unsigned long long)h_noise[2] * 3449720151ull;
#if __CUDA_ARCH__ == 900
                        h_noise[0] = (unsigned int)(s1_48 >> 32ull) ^ h_noise[1] ^ key0_3;
#else
                        h_noise[0] = (unsigned int)(s1_48 >> 32ull) ^ h_noise[1] ^ key0_1;
#endif
                        h_noise[1] = (unsigned int)s1_48;
#if __CUDA_ARCH__ == 900
                        h_noise[2] = (unsigned int)(s0_48 >> 32ull) ^ h_noise[3] ^ key1_3;
#else
                        h_noise[2] = (unsigned int)(s0_48 >> 32ull) ^ h_noise[3] ^ key1_1;
#endif
                        h_noise[3] = (unsigned int)s0_48;
                        unsigned long long s0_49 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_49 = (unsigned long long)h_noise[6] * 3449720151ull;
#if __CUDA_ARCH__ == 900
                        h_noise[4] = (unsigned int)(s1_49 >> 32ull) ^ h_noise[5] ^ key0_3;
#else
                        h_noise[4] = (unsigned int)(s1_49 >> 32ull) ^ h_noise[5] ^ key0_1;
#endif
                        h_noise[5] = (unsigned int)s1_49;
#if __CUDA_ARCH__ == 900
                        h_noise[6] = (unsigned int)(s0_49 >> 32ull) ^ h_noise[7] ^ key1_3;
#else
                        h_noise[6] = (unsigned int)(s0_49 >> 32ull) ^ h_noise[7] ^ key1_1;
#endif
                        h_noise[7] = (unsigned int)s0_49;
                        unsigned long long s0_50 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_50 = (unsigned long long)h_noise[10] * 3449720151ull;
#if __CUDA_ARCH__ == 900
                        h_noise[8] = (unsigned int)(s1_50 >> 32ull) ^ h_noise[9] ^ key0_3;
#else
                        h_noise[8] = (unsigned int)(s1_50 >> 32ull) ^ h_noise[9] ^ key0_1;
#endif
                        h_noise[9] = (unsigned int)s1_50;
#if __CUDA_ARCH__ == 900
                        h_noise[10] = (unsigned int)(s0_50 >> 32ull) ^ h_noise[11] ^ key1_3;
#else
                        h_noise[10] = (unsigned int)(s0_50 >> 32ull) ^ h_noise[11] ^ key1_1;
#endif
                        h_noise[11] = (unsigned int)s0_50;
                        unsigned long long s0_51 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_51 = (unsigned long long)h_noise[14] * 3449720151ull;
#if __CUDA_ARCH__ == 900
                        h_noise[12] = (unsigned int)(s1_51 >> 32ull) ^ h_noise[13] ^ key0_3;
#else
                        h_noise[12] = (unsigned int)(s1_51 >> 32ull) ^ h_noise[13] ^ key0_1;
#endif
                        h_noise[13] = (unsigned int)s1_51;
#if __CUDA_ARCH__ == 900
                        h_noise[14] = (unsigned int)(s0_51 >> 32ull) ^ h_noise[15] ^ key1_3;
#else
                        h_noise[14] = (unsigned int)(s0_51 >> 32ull) ^ h_noise[15] ^ key1_1;
#endif
                        h_noise[15] = (unsigned int)s0_51;
#if __CUDA_ARCH__ == 900
                        key0_3 = key0_3 + 2654435769u;
                        key1_3 = key1_3 + 3144134277u;
                        unsigned long long s0_52 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_52 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_52 >> 32ull) ^ h_noise[1] ^ key0_3;
                        h_noise[1] = (unsigned int)s1_52;
                        h_noise[2] = (unsigned int)(s0_52 >> 32ull) ^ h_noise[3] ^ key1_3;
                        h_noise[3] = (unsigned int)s0_52;
                        unsigned long long s0_53 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_53 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_53 >> 32ull) ^ h_noise[5] ^ key0_3;
                        h_noise[5] = (unsigned int)s1_53;
                        h_noise[6] = (unsigned int)(s0_53 >> 32ull) ^ h_noise[7] ^ key1_3;
                        h_noise[7] = (unsigned int)s0_53;
                        unsigned long long s0_54 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_54 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_54 >> 32ull) ^ h_noise[9] ^ key0_3;
                        h_noise[9] = (unsigned int)s1_54;
                        h_noise[10] = (unsigned int)(s0_54 >> 32ull) ^ h_noise[11] ^ key1_3;
                        h_noise[11] = (unsigned int)s0_54;
                        unsigned long long s0_55 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_55 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_55 >> 32ull) ^ h_noise[13] ^ key0_3;
                        h_noise[13] = (unsigned int)s1_55;
                        h_noise[14] = (unsigned int)(s0_55 >> 32ull) ^ h_noise[15] ^ key1_3;
                        h_noise[15] = (unsigned int)s0_55;
                        key0_3 = key0_3 + 2654435769u;
                        key1_3 = key1_3 + 3144134277u;
                        unsigned long long s0_56 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_56 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_56 >> 32ull) ^ h_noise[1] ^ key0_3;
                        h_noise[1] = (unsigned int)s1_56;
                        h_noise[2] = (unsigned int)(s0_56 >> 32ull) ^ h_noise[3] ^ key1_3;
                        h_noise[3] = (unsigned int)s0_56;
                        unsigned long long s0_57 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_57 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_57 >> 32ull) ^ h_noise[5] ^ key0_3;
                        h_noise[5] = (unsigned int)s1_57;
                        h_noise[6] = (unsigned int)(s0_57 >> 32ull) ^ h_noise[7] ^ key1_3;
                        h_noise[7] = (unsigned int)s0_57;
                        unsigned long long s0_58 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_58 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_58 >> 32ull) ^ h_noise[9] ^ key0_3;
                        h_noise[9] = (unsigned int)s1_58;
                        h_noise[10] = (unsigned int)(s0_58 >> 32ull) ^ h_noise[11] ^ key1_3;
                        h_noise[11] = (unsigned int)s0_58;
                        unsigned long long s0_59 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_59 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_59 >> 32ull) ^ h_noise[13] ^ key0_3;
                        h_noise[13] = (unsigned int)s1_59;
                        h_noise[14] = (unsigned int)(s0_59 >> 32ull) ^ h_noise[15] ^ key1_3;
                        h_noise[15] = (unsigned int)s0_59;
                        key0_3 = key0_3 + 2654435769u;
                        key1_3 = key1_3 + 3144134277u;
                        #pragma unroll 1
                        for (int _round_4 = 5; _round_4 < philox_rounds; _round_4++) {
                            unsigned long long t0_16 = (unsigned long long)h_noise[0] * 3528531795ull;
                            unsigned long long t1_16 = (unsigned long long)h_noise[2] * 3449720151ull;
                            h_noise[0] = (unsigned int)(t1_16 >> 32ull) ^ h_noise[1] ^ key0_3;
                            h_noise[1] = (unsigned int)t1_16;
                            h_noise[2] = (unsigned int)(t0_16 >> 32ull) ^ h_noise[3] ^ key1_3;
                            h_noise[3] = (unsigned int)t0_16;
                            unsigned long long t0_17 = (unsigned long long)h_noise[4] * 3528531795ull;
                            unsigned long long t1_17 = (unsigned long long)h_noise[6] * 3449720151ull;
                            h_noise[4] = (unsigned int)(t1_17 >> 32ull) ^ h_noise[5] ^ key0_3;
                            h_noise[5] = (unsigned int)t1_17;
                            h_noise[6] = (unsigned int)(t0_17 >> 32ull) ^ h_noise[7] ^ key1_3;
                            h_noise[7] = (unsigned int)t0_17;
                            unsigned long long t0_18 = (unsigned long long)h_noise[8] * 3528531795ull;
                            unsigned long long t1_18 = (unsigned long long)h_noise[10] * 3449720151ull;
                            h_noise[8] = (unsigned int)(t1_18 >> 32ull) ^ h_noise[9] ^ key0_3;
                            h_noise[9] = (unsigned int)t1_18;
                            h_noise[10] = (unsigned int)(t0_18 >> 32ull) ^ h_noise[11] ^ key1_3;
                            h_noise[11] = (unsigned int)t0_18;
                            unsigned long long t0_19 = (unsigned long long)h_noise[12] * 3528531795ull;
                            unsigned long long t1_19 = (unsigned long long)h_noise[14] * 3449720151ull;
                            h_noise[12] = (unsigned int)(t1_19 >> 32ull) ^ h_noise[13] ^ key0_3;
                            h_noise[13] = (unsigned int)t1_19;
                            h_noise[14] = (unsigned int)(t0_19 >> 32ull) ^ h_noise[15] ^ key1_3;
                            h_noise[15] = (unsigned int)t0_19;
                            key0_3 = key0_3 + 2654435769u;
                            key1_3 = key1_3 + 3144134277u;
                        }
                        float _abs_1199;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1199) : "f"(S[0]));
                        unsigned int s_127 = __as_u32(mul_f32_rz_noftz(_abs_1199, __uint_as_float(125829120)));
                        unsigned int _min_1200 = ((s_127) < (260046848) ? (s_127) : (260046848));
                        unsigned int h_128 = _min_1200 + (h_noise[0] & 8191) >> 13;
                        unsigned int mag_127 = ((s_127 <= 2139095040) ? h_128 : 32256);
                        float _abs_1201;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1201) : "f"(S[1]));
                        unsigned int s_127_1 = __as_u32(mul_f32_rz_noftz(_abs_1201, __uint_as_float(125829120)));
                        unsigned int _min_1202 = ((s_127_1) < (260046848) ? (s_127_1) : (260046848));
                        unsigned int h_129 = _min_1202 + ((h_noise[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_127_1 = ((s_127_1 <= 2139095040) ? h_129 : 32256);
                        uint32_t mags_63;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_63) : "r"(mag_127), "r"(mag_127_1));
                        uint32_t signs_63;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_63) : "r"(__as_u32(S[0])), "r"(__as_u32(S[1])));
                        unsigned int word_63 = signs_63 & 2147516416u | mags_63;
                        h_words[0] = word_63;
                        float _abs_1203;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1203) : "f"(S[2]));
                        unsigned int s_128 = __as_u32(mul_f32_rz_noftz(_abs_1203, __uint_as_float(125829120)));
                        unsigned int _min_1204 = ((s_128) < (260046848) ? (s_128) : (260046848));
                        unsigned int h_130 = _min_1204 + (h_noise[1] & 8191) >> 13;
                        unsigned int mag_128 = ((s_128 <= 2139095040) ? h_130 : 32256);
                        float _abs_1205;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1205) : "f"(S[3]));
                        unsigned int s_129 = __as_u32(mul_f32_rz_noftz(_abs_1205, __uint_as_float(125829120)));
                        unsigned int _min_1206 = ((s_129) < (260046848) ? (s_129) : (260046848));
                        unsigned int h_131 = _min_1206 + ((h_noise[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_129 = ((s_129 <= 2139095040) ? h_131 : 32256);
                        uint32_t mags_63_1;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_63_1) : "r"(mag_128), "r"(mag_129));
                        uint32_t signs_63_1;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_63_1) : "r"(__as_u32(S[2])), "r"(__as_u32(S[3])));
                        unsigned int word_63_1 = signs_63_1 & 2147516416u | mags_63_1;
                        h_words[1] = word_63_1;
                        reinterpret_cast<int2*>(state + (write_rows_base))[0] = reinterpret_cast<int2*>(h_words)[0];
                        float _abs_1207;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1207) : "f"(S[16]));
                        unsigned int s_130 = __as_u32(mul_f32_rz_noftz(_abs_1207, __uint_as_float(125829120)));
                        unsigned int _min_1208 = ((s_130) < (260046848) ? (s_130) : (260046848));
                        unsigned int h_132 = _min_1208 + (h_noise[2] & 8191) >> 13;
                        unsigned int mag_130 = ((s_130 <= 2139095040) ? h_132 : 32256);
                        float _abs_1209;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1209) : "f"(S[17]));
                        unsigned int s_131 = __as_u32(mul_f32_rz_noftz(_abs_1209, __uint_as_float(125829120)));
                        unsigned int _min_1210 = ((s_131) < (260046848) ? (s_131) : (260046848));
                        unsigned int h_133 = _min_1210 + ((h_noise[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_131 = ((s_131 <= 2139095040) ? h_133 : 32256);
                        uint32_t mags_64;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_64) : "r"(mag_130), "r"(mag_131));
                        uint32_t signs_64;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_64) : "r"(__as_u32(S[16])), "r"(__as_u32(S[17])));
                        unsigned int word_64 = signs_64 & 2147516416u | mags_64;
                        h_words[0] = word_64;
                        float _abs_1211;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1211) : "f"(S[18]));
                        unsigned int s_132 = __as_u32(mul_f32_rz_noftz(_abs_1211, __uint_as_float(125829120)));
                        unsigned int _min_1212 = ((s_132) < (260046848) ? (s_132) : (260046848));
                        unsigned int h_134 = _min_1212 + (h_noise[3] & 8191) >> 13;
                        unsigned int mag_132 = ((s_132 <= 2139095040) ? h_134 : 32256);
                        float _abs_1213;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1213) : "f"(S[19]));
                        unsigned int s_133 = __as_u32(mul_f32_rz_noftz(_abs_1213, __uint_as_float(125829120)));
                        unsigned int _min_1214 = ((s_133) < (260046848) ? (s_133) : (260046848));
                        unsigned int h_135 = _min_1214 + ((h_noise[3] & 536805376) >> 16) >> 13;
                        unsigned int mag_133 = ((s_133 <= 2139095040) ? h_135 : 32256);
                        uint32_t mags_65;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_65) : "r"(mag_132), "r"(mag_133));
                        uint32_t signs_65;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_65) : "r"(__as_u32(S[18])), "r"(__as_u32(S[19])));
                        unsigned int word_65 = signs_65 & 2147516416u | mags_65;
                        h_words[1] = word_65;
                        reinterpret_cast<int2*>(state + (write_rows_base + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        float _abs_1215;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1215) : "f"(S[4]));
                        unsigned int s_134 = __as_u32(mul_f32_rz_noftz(_abs_1215, __uint_as_float(125829120)));
                        unsigned int _min_1216 = ((s_134) < (260046848) ? (s_134) : (260046848));
                        unsigned int h_136 = _min_1216 + (h_noise[4] & 8191) >> 13;
                        unsigned int mag_134 = ((s_134 <= 2139095040) ? h_136 : 32256);
                        float _abs_1217;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1217) : "f"(S[5]));
                        unsigned int s_135 = __as_u32(mul_f32_rz_noftz(_abs_1217, __uint_as_float(125829120)));
                        unsigned int _min_1218 = ((s_135) < (260046848) ? (s_135) : (260046848));
                        unsigned int h_137 = _min_1218 + ((h_noise[4] & 536805376) >> 16) >> 13;
                        unsigned int mag_135 = ((s_135 <= 2139095040) ? h_137 : 32256);
                        uint32_t mags_66;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_66) : "r"(mag_134), "r"(mag_135));
                        uint32_t signs_66;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_66) : "r"(__as_u32(S[4])), "r"(__as_u32(S[5])));
                        unsigned int word_66 = signs_66 & 2147516416u | mags_66;
                        h_words[0] = word_66;
                        float _abs_1219;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1219) : "f"(S[6]));
                        unsigned int s_136 = __as_u32(mul_f32_rz_noftz(_abs_1219, __uint_as_float(125829120)));
                        unsigned int _min_1220 = ((s_136) < (260046848) ? (s_136) : (260046848));
                        unsigned int h_138 = _min_1220 + (h_noise[5] & 8191) >> 13;
                        unsigned int mag_136 = ((s_136 <= 2139095040) ? h_138 : 32256);
                        float _abs_1221;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1221) : "f"(S[7]));
                        unsigned int s_137 = __as_u32(mul_f32_rz_noftz(_abs_1221, __uint_as_float(125829120)));
                        unsigned int _min_1222 = ((s_137) < (260046848) ? (s_137) : (260046848));
                        unsigned int h_139 = _min_1222 + ((h_noise[5] & 536805376) >> 16) >> 13;
                        unsigned int mag_137 = ((s_137 <= 2139095040) ? h_139 : 32256);
                        uint32_t mags_67;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_67) : "r"(mag_136), "r"(mag_137));
                        uint32_t signs_67;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_67) : "r"(__as_u32(S[6])), "r"(__as_u32(S[7])));
                        unsigned int word_67 = signs_67 & 2147516416u | mags_67;
                        h_words[1] = word_67;
                        reinterpret_cast<int2*>(state + (write_rows_base + 16ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        float _abs_1223;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1223) : "f"(S[20]));
                        unsigned int s_138 = __as_u32(mul_f32_rz_noftz(_abs_1223, __uint_as_float(125829120)));
                        unsigned int _min_1224 = ((s_138) < (260046848) ? (s_138) : (260046848));
                        unsigned int h_140 = _min_1224 + (h_noise[6] & 8191) >> 13;
                        unsigned int mag_138 = ((s_138 <= 2139095040) ? h_140 : 32256);
                        float _abs_1225;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1225) : "f"(S[21]));
                        unsigned int s_139 = __as_u32(mul_f32_rz_noftz(_abs_1225, __uint_as_float(125829120)));
                        unsigned int _min_1226 = ((s_139) < (260046848) ? (s_139) : (260046848));
                        unsigned int h_141 = _min_1226 + ((h_noise[6] & 536805376) >> 16) >> 13;
                        unsigned int mag_139 = ((s_139 <= 2139095040) ? h_141 : 32256);
                        uint32_t mags_68;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_68) : "r"(mag_138), "r"(mag_139));
                        uint32_t signs_68;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_68) : "r"(__as_u32(S[20])), "r"(__as_u32(S[21])));
                        unsigned int word_68 = signs_68 & 2147516416u | mags_68;
                        h_words[0] = word_68;
                        float _abs_1227;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1227) : "f"(S[22]));
                        unsigned int s_140 = __as_u32(mul_f32_rz_noftz(_abs_1227, __uint_as_float(125829120)));
                        unsigned int _min_1228 = ((s_140) < (260046848) ? (s_140) : (260046848));
                        unsigned int h_142 = _min_1228 + (h_noise[7] & 8191) >> 13;
                        unsigned int mag_140 = ((s_140 <= 2139095040) ? h_142 : 32256);
                        float _abs_1229;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1229) : "f"(S[23]));
                        unsigned int s_141 = __as_u32(mul_f32_rz_noftz(_abs_1229, __uint_as_float(125829120)));
                        unsigned int _min_1230 = ((s_141) < (260046848) ? (s_141) : (260046848));
                        unsigned int h_143 = _min_1230 + ((h_noise[7] & 536805376) >> 16) >> 13;
                        unsigned int mag_141 = ((s_141 <= 2139095040) ? h_143 : 32256);
                        uint32_t mags_69;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_69) : "r"(mag_140), "r"(mag_141));
                        uint32_t signs_69;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_69) : "r"(__as_u32(S[22])), "r"(__as_u32(S[23])));
                        unsigned int word_69 = signs_69 & 2147516416u | mags_69;
                        h_words[1] = word_69;
                        reinterpret_cast<int2*>(state + (write_rows_base + 80ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        float _abs_1231;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1231) : "f"(S[8]));
                        unsigned int s_142 = __as_u32(mul_f32_rz_noftz(_abs_1231, __uint_as_float(125829120)));
                        unsigned int _min_1232 = ((s_142) < (260046848) ? (s_142) : (260046848));
                        unsigned int h_144 = _min_1232 + (h_noise[8] & 8191) >> 13;
                        unsigned int mag_142 = ((s_142 <= 2139095040) ? h_144 : 32256);
                        float _abs_1233;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1233) : "f"(S[9]));
                        unsigned int s_143 = __as_u32(mul_f32_rz_noftz(_abs_1233, __uint_as_float(125829120)));
                        unsigned int _min_1234 = ((s_143) < (260046848) ? (s_143) : (260046848));
                        unsigned int h_145 = _min_1234 + ((h_noise[8] & 536805376) >> 16) >> 13;
                        unsigned int mag_143 = ((s_143 <= 2139095040) ? h_145 : 32256);
                        uint32_t mags_70;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_70) : "r"(mag_142), "r"(mag_143));
                        uint32_t signs_70;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_70) : "r"(__as_u32(S[8])), "r"(__as_u32(S[9])));
                        unsigned int word_70 = signs_70 & 2147516416u | mags_70;
                        h_words[0] = word_70;
                        float _abs_1235;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1235) : "f"(S[10]));
                        unsigned int s_144 = __as_u32(mul_f32_rz_noftz(_abs_1235, __uint_as_float(125829120)));
                        unsigned int _min_1236 = ((s_144) < (260046848) ? (s_144) : (260046848));
                        unsigned int h_146 = _min_1236 + (h_noise[9] & 8191) >> 13;
                        unsigned int mag_144 = ((s_144 <= 2139095040) ? h_146 : 32256);
                        float _abs_1237;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1237) : "f"(S[11]));
                        unsigned int s_145 = __as_u32(mul_f32_rz_noftz(_abs_1237, __uint_as_float(125829120)));
                        unsigned int _min_1238 = ((s_145) < (260046848) ? (s_145) : (260046848));
                        unsigned int h_147 = _min_1238 + ((h_noise[9] & 536805376) >> 16) >> 13;
                        unsigned int mag_145 = ((s_145 <= 2139095040) ? h_147 : 32256);
                        uint32_t mags_71;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_71) : "r"(mag_144), "r"(mag_145));
                        uint32_t signs_71;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_71) : "r"(__as_u32(S[10])), "r"(__as_u32(S[11])));
                        unsigned int word_71 = signs_71 & 2147516416u | mags_71;
                        h_words[1] = word_71;
                        reinterpret_cast<int2*>(state + (write_rows_base + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        float _abs_1239;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1239) : "f"(S[24]));
                        unsigned int s_146 = __as_u32(mul_f32_rz_noftz(_abs_1239, __uint_as_float(125829120)));
                        unsigned int _min_1240 = ((s_146) < (260046848) ? (s_146) : (260046848));
                        unsigned int h_148 = _min_1240 + (h_noise[10] & 8191) >> 13;
                        unsigned int mag_146 = ((s_146 <= 2139095040) ? h_148 : 32256);
                        float _abs_1241;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1241) : "f"(S[25]));
                        unsigned int s_147 = __as_u32(mul_f32_rz_noftz(_abs_1241, __uint_as_float(125829120)));
                        unsigned int _min_1242 = ((s_147) < (260046848) ? (s_147) : (260046848));
                        unsigned int h_149 = _min_1242 + ((h_noise[10] & 536805376) >> 16) >> 13;
                        unsigned int mag_147 = ((s_147 <= 2139095040) ? h_149 : 32256);
                        uint32_t mags_72;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_72) : "r"(mag_146), "r"(mag_147));
                        uint32_t signs_72;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_72) : "r"(__as_u32(S[24])), "r"(__as_u32(S[25])));
                        unsigned int word_72 = signs_72 & 2147516416u | mags_72;
                        h_words[0] = word_72;
                        float _abs_1243;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1243) : "f"(S[26]));
                        unsigned int s_148 = __as_u32(mul_f32_rz_noftz(_abs_1243, __uint_as_float(125829120)));
                        unsigned int _min_1244 = ((s_148) < (260046848) ? (s_148) : (260046848));
                        unsigned int h_150 = _min_1244 + (h_noise[11] & 8191) >> 13;
                        unsigned int mag_148 = ((s_148 <= 2139095040) ? h_150 : 32256);
                        float _abs_1245;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1245) : "f"(S[27]));
                        unsigned int s_149 = __as_u32(mul_f32_rz_noftz(_abs_1245, __uint_as_float(125829120)));
                        unsigned int _min_1246 = ((s_149) < (260046848) ? (s_149) : (260046848));
                        unsigned int h_151 = _min_1246 + ((h_noise[11] & 536805376) >> 16) >> 13;
                        unsigned int mag_149 = ((s_149 <= 2139095040) ? h_151 : 32256);
                        uint32_t mags_73;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_73) : "r"(mag_148), "r"(mag_149));
                        uint32_t signs_73;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_73) : "r"(__as_u32(S[26])), "r"(__as_u32(S[27])));
                        unsigned int word_73 = signs_73 & 2147516416u | mags_73;
                        h_words[1] = word_73;
                        reinterpret_cast<int2*>(state + (write_rows_base + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        float _abs_1247;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1247) : "f"(S[12]));
                        unsigned int s_150 = __as_u32(mul_f32_rz_noftz(_abs_1247, __uint_as_float(125829120)));
                        unsigned int _min_1248 = ((s_150) < (260046848) ? (s_150) : (260046848));
                        unsigned int h_152 = _min_1248 + (h_noise[12] & 8191) >> 13;
                        unsigned int mag_150 = ((s_150 <= 2139095040) ? h_152 : 32256);
                        float _abs_1249;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1249) : "f"(S[13]));
                        unsigned int s_151 = __as_u32(mul_f32_rz_noftz(_abs_1249, __uint_as_float(125829120)));
                        unsigned int _min_1250 = ((s_151) < (260046848) ? (s_151) : (260046848));
                        unsigned int h_153 = _min_1250 + ((h_noise[12] & 536805376) >> 16) >> 13;
                        unsigned int mag_151 = ((s_151 <= 2139095040) ? h_153 : 32256);
                        uint32_t mags_74;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_74) : "r"(mag_150), "r"(mag_151));
                        uint32_t signs_74;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_74) : "r"(__as_u32(S[12])), "r"(__as_u32(S[13])));
                        unsigned int word_74 = signs_74 & 2147516416u | mags_74;
                        h_words[0] = word_74;
                        float _abs_1251;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1251) : "f"(S[14]));
                        unsigned int s_152 = __as_u32(mul_f32_rz_noftz(_abs_1251, __uint_as_float(125829120)));
                        unsigned int _min_1252 = ((s_152) < (260046848) ? (s_152) : (260046848));
                        unsigned int h_154 = _min_1252 + (h_noise[13] & 8191) >> 13;
                        unsigned int mag_152 = ((s_152 <= 2139095040) ? h_154 : 32256);
                        float _abs_1253;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1253) : "f"(S[15]));
                        unsigned int s_153 = __as_u32(mul_f32_rz_noftz(_abs_1253, __uint_as_float(125829120)));
                        unsigned int _min_1254 = ((s_153) < (260046848) ? (s_153) : (260046848));
                        unsigned int h_155 = _min_1254 + ((h_noise[13] & 536805376) >> 16) >> 13;
                        unsigned int mag_153 = ((s_153 <= 2139095040) ? h_155 : 32256);
                        uint32_t mags_75;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_75) : "r"(mag_152), "r"(mag_153));
                        uint32_t signs_75;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_75) : "r"(__as_u32(S[14])), "r"(__as_u32(S[15])));
                        unsigned int word_75 = signs_75 & 2147516416u | mags_75;
                        h_words[1] = word_75;
                        reinterpret_cast<int2*>(state + (write_rows_base + 48ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        float _abs_1255;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1255) : "f"(S[28]));
                        unsigned int s_154 = __as_u32(mul_f32_rz_noftz(_abs_1255, __uint_as_float(125829120)));
                        unsigned int _min_1256 = ((s_154) < (260046848) ? (s_154) : (260046848));
                        unsigned int h_156 = _min_1256 + (h_noise[14] & 8191) >> 13;
                        unsigned int mag_154 = ((s_154 <= 2139095040) ? h_156 : 32256);
                        float _abs_1257;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1257) : "f"(S[29]));
                        unsigned int s_155 = __as_u32(mul_f32_rz_noftz(_abs_1257, __uint_as_float(125829120)));
                        unsigned int _min_1258 = ((s_155) < (260046848) ? (s_155) : (260046848));
                        unsigned int h_157 = _min_1258 + ((h_noise[14] & 536805376) >> 16) >> 13;
                        unsigned int mag_155 = ((s_155 <= 2139095040) ? h_157 : 32256);
                        uint32_t mags_76;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_76) : "r"(mag_154), "r"(mag_155));
                        uint32_t signs_76;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_76) : "r"(__as_u32(S[28])), "r"(__as_u32(S[29])));
                        unsigned int word_76 = signs_76 & 2147516416u | mags_76;
                        h_words[0] = word_76;
                        float _abs_1259;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1259) : "f"(S[30]));
                        unsigned int s_156 = __as_u32(mul_f32_rz_noftz(_abs_1259, __uint_as_float(125829120)));
                        unsigned int _min_1260 = ((s_156) < (260046848) ? (s_156) : (260046848));
                        unsigned int h_158 = _min_1260 + (h_noise[15] & 8191) >> 13;
                        unsigned int mag_156 = ((s_156 <= 2139095040) ? h_158 : 32256);
                        float _abs_1261;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1261) : "f"(S[31]));
                        unsigned int s_157 = __as_u32(mul_f32_rz_noftz(_abs_1261, __uint_as_float(125829120)));
                        unsigned int _min_1262 = ((s_157) < (260046848) ? (s_157) : (260046848));
                        unsigned int h_159 = _min_1262 + ((h_noise[15] & 536805376) >> 16) >> 13;
                        unsigned int mag_157 = ((s_157 <= 2139095040) ? h_159 : 32256);
                        uint32_t mags_77;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_77) : "r"(mag_156), "r"(mag_157));
                        uint32_t signs_77;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_77) : "r"(__as_u32(S[30])), "r"(__as_u32(S[31])));
                        unsigned int word_77 = signs_77 & 2147516416u | mags_77;
                        h_words[1] = word_77;
                        reinterpret_cast<int2*>(state + (write_rows_base + 112ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        unsigned int key0_3_1 = seed_lo;
                        unsigned int key1_3_1 = seed_hi;
                        long long o_20 = noise_at + state_stride_p2;
                        unsigned long long p0_20 = (unsigned long long)(unsigned int)o_20 * 3528531795ull;
                        h_noise[0] = (unsigned int)(o_20 >> 32ll) ^ key0_3_1;
                        h_noise[2] = (unsigned int)(p0_20 >> 32ull) ^ key1_3_1;
                        h_noise[3] = (unsigned int)p0_20;
                        long long o_21 = noise_at + state_stride_p2 + 16ll;
                        unsigned long long p0_21 = (unsigned long long)(unsigned int)o_21 * 3528531795ull;
                        h_noise[4] = (unsigned int)(o_21 >> 32ll) ^ key0_3_1;
                        h_noise[6] = (unsigned int)(p0_21 >> 32ull) ^ key1_3_1;
                        h_noise[7] = (unsigned int)p0_21;
                        long long o_22 = noise_at + state_stride_p2 + 32ll;
                        unsigned long long p0_22 = (unsigned long long)(unsigned int)o_22 * 3528531795ull;
                        h_noise[8] = (unsigned int)(o_22 >> 32ll) ^ key0_3_1;
                        h_noise[10] = (unsigned int)(p0_22 >> 32ull) ^ key1_3_1;
                        h_noise[11] = (unsigned int)p0_22;
                        long long o_23 = noise_at + state_stride_p2 + 48ll;
                        unsigned long long p0_23 = (unsigned long long)(unsigned int)o_23 * 3528531795ull;
                        h_noise[12] = (unsigned int)(o_23 >> 32ll) ^ key0_3_1;
                        h_noise[14] = (unsigned int)(p0_23 >> 32ull) ^ key1_3_1;
                        h_noise[15] = (unsigned int)p0_23;
                        key0_3_1 = key0_3_1 + 2654435769u;
                        key1_3_1 = key1_3_1 + 3144134277u;
                        unsigned long long q0_20 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long q1_20 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(q1_20 >> 32ull) ^ key0_3_1;
                        h_noise[1] = (unsigned int)q1_20;
                        h_noise[2] = (unsigned int)(q0_20 >> 32ull) ^ h_noise[3] ^ key1_3_1;
                        h_noise[3] = (unsigned int)q0_20;
                        unsigned long long q0_21 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long q1_21 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(q1_21 >> 32ull) ^ key0_3_1;
                        h_noise[5] = (unsigned int)q1_21;
                        h_noise[6] = (unsigned int)(q0_21 >> 32ull) ^ h_noise[7] ^ key1_3_1;
                        h_noise[7] = (unsigned int)q0_21;
                        unsigned long long q0_22 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long q1_22 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(q1_22 >> 32ull) ^ key0_3_1;
                        h_noise[9] = (unsigned int)q1_22;
                        h_noise[10] = (unsigned int)(q0_22 >> 32ull) ^ h_noise[11] ^ key1_3_1;
                        h_noise[11] = (unsigned int)q0_22;
                        unsigned long long q0_23 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long q1_23 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(q1_23 >> 32ull) ^ key0_3_1;
                        h_noise[13] = (unsigned int)q1_23;
                        h_noise[14] = (unsigned int)(q0_23 >> 32ull) ^ h_noise[15] ^ key1_3_1;
                        h_noise[15] = (unsigned int)q0_23;
                        key0_3_1 = key0_3_1 + 2654435769u;
                        key1_3_1 = key1_3_1 + 3144134277u;
                        unsigned long long s0_60 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_60 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_60 >> 32ull) ^ h_noise[1] ^ key0_3_1;
                        h_noise[1] = (unsigned int)s1_60;
                        h_noise[2] = (unsigned int)(s0_60 >> 32ull) ^ h_noise[3] ^ key1_3_1;
                        h_noise[3] = (unsigned int)s0_60;
                        unsigned long long s0_61 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_61 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_61 >> 32ull) ^ h_noise[5] ^ key0_3_1;
                        h_noise[5] = (unsigned int)s1_61;
                        h_noise[6] = (unsigned int)(s0_61 >> 32ull) ^ h_noise[7] ^ key1_3_1;
                        h_noise[7] = (unsigned int)s0_61;
                        unsigned long long s0_62 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_62 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_62 >> 32ull) ^ h_noise[9] ^ key0_3_1;
                        h_noise[9] = (unsigned int)s1_62;
                        h_noise[10] = (unsigned int)(s0_62 >> 32ull) ^ h_noise[11] ^ key1_3_1;
                        h_noise[11] = (unsigned int)s0_62;
                        unsigned long long s0_63 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_63 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_63 >> 32ull) ^ h_noise[13] ^ key0_3_1;
                        h_noise[13] = (unsigned int)s1_63;
                        h_noise[14] = (unsigned int)(s0_63 >> 32ull) ^ h_noise[15] ^ key1_3_1;
                        h_noise[15] = (unsigned int)s0_63;
                        key0_3_1 = key0_3_1 + 2654435769u;
                        key1_3_1 = key1_3_1 + 3144134277u;
#else
                        unsigned long long s0_52 = (unsigned long long)h_noise[16] * 3528531795ull;
                        unsigned long long s1_52 = (unsigned long long)h_noise[18] * 3449720151ull;
                        h_noise[16] = (unsigned int)(s1_52 >> 32ull) ^ h_noise[17] ^ key0_1;
                        h_noise[17] = (unsigned int)s1_52;
                        h_noise[18] = (unsigned int)(s0_52 >> 32ull) ^ h_noise[19] ^ key1_1;
                        h_noise[19] = (unsigned int)s0_52;
                        unsigned long long s0_53 = (unsigned long long)h_noise[20] * 3528531795ull;
                        unsigned long long s1_53 = (unsigned long long)h_noise[22] * 3449720151ull;
                        h_noise[20] = (unsigned int)(s1_53 >> 32ull) ^ h_noise[21] ^ key0_1;
                        h_noise[21] = (unsigned int)s1_53;
                        h_noise[22] = (unsigned int)(s0_53 >> 32ull) ^ h_noise[23] ^ key1_1;
                        h_noise[23] = (unsigned int)s0_53;
                        unsigned long long s0_54 = (unsigned long long)h_noise[24] * 3528531795ull;
                        unsigned long long s1_54 = (unsigned long long)h_noise[26] * 3449720151ull;
                        h_noise[24] = (unsigned int)(s1_54 >> 32ull) ^ h_noise[25] ^ key0_1;
                        h_noise[25] = (unsigned int)s1_54;
                        h_noise[26] = (unsigned int)(s0_54 >> 32ull) ^ h_noise[27] ^ key1_1;
                        h_noise[27] = (unsigned int)s0_54;
                        unsigned long long s0_55 = (unsigned long long)h_noise[28] * 3528531795ull;
                        unsigned long long s1_55 = (unsigned long long)h_noise[30] * 3449720151ull;
                        h_noise[28] = (unsigned int)(s1_55 >> 32ull) ^ h_noise[29] ^ key0_1;
                        h_noise[29] = (unsigned int)s1_55;
                        h_noise[30] = (unsigned int)(s0_55 >> 32ull) ^ h_noise[31] ^ key1_1;
                        h_noise[31] = (unsigned int)s0_55;
                        unsigned long long s0_56 = (unsigned long long)h_noise[32] * 3528531795ull;
                        unsigned long long s1_56 = (unsigned long long)h_noise[34] * 3449720151ull;
                        h_noise[32] = (unsigned int)(s1_56 >> 32ull) ^ h_noise[33] ^ key0_1;
                        h_noise[33] = (unsigned int)s1_56;
                        h_noise[34] = (unsigned int)(s0_56 >> 32ull) ^ h_noise[35] ^ key1_1;
                        h_noise[35] = (unsigned int)s0_56;
                        unsigned long long s0_57 = (unsigned long long)h_noise[36] * 3528531795ull;
                        unsigned long long s1_57 = (unsigned long long)h_noise[38] * 3449720151ull;
                        h_noise[36] = (unsigned int)(s1_57 >> 32ull) ^ h_noise[37] ^ key0_1;
                        h_noise[37] = (unsigned int)s1_57;
                        h_noise[38] = (unsigned int)(s0_57 >> 32ull) ^ h_noise[39] ^ key1_1;
                        h_noise[39] = (unsigned int)s0_57;
                        unsigned long long s0_58 = (unsigned long long)h_noise[40] * 3528531795ull;
                        unsigned long long s1_58 = (unsigned long long)h_noise[42] * 3449720151ull;
                        h_noise[40] = (unsigned int)(s1_58 >> 32ull) ^ h_noise[41] ^ key0_1;
                        h_noise[41] = (unsigned int)s1_58;
                        h_noise[42] = (unsigned int)(s0_58 >> 32ull) ^ h_noise[43] ^ key1_1;
                        h_noise[43] = (unsigned int)s0_58;
                        unsigned long long s0_59 = (unsigned long long)h_noise[44] * 3528531795ull;
                        unsigned long long s1_59 = (unsigned long long)h_noise[46] * 3449720151ull;
                        h_noise[44] = (unsigned int)(s1_59 >> 32ull) ^ h_noise[45] ^ key0_1;
                        h_noise[45] = (unsigned int)s1_59;
                        h_noise[46] = (unsigned int)(s0_59 >> 32ull) ^ h_noise[47] ^ key1_1;
                        h_noise[47] = (unsigned int)s0_59;
                        unsigned long long s0_60 = (unsigned long long)h_noise[48] * 3528531795ull;
                        unsigned long long s1_60 = (unsigned long long)h_noise[50] * 3449720151ull;
                        h_noise[48] = (unsigned int)(s1_60 >> 32ull) ^ h_noise[49] ^ key0_1;
                        h_noise[49] = (unsigned int)s1_60;
                        h_noise[50] = (unsigned int)(s0_60 >> 32ull) ^ h_noise[51] ^ key1_1;
                        h_noise[51] = (unsigned int)s0_60;
                        unsigned long long s0_61 = (unsigned long long)h_noise[52] * 3528531795ull;
                        unsigned long long s1_61 = (unsigned long long)h_noise[54] * 3449720151ull;
                        h_noise[52] = (unsigned int)(s1_61 >> 32ull) ^ h_noise[53] ^ key0_1;
                        h_noise[53] = (unsigned int)s1_61;
                        h_noise[54] = (unsigned int)(s0_61 >> 32ull) ^ h_noise[55] ^ key1_1;
                        h_noise[55] = (unsigned int)s0_61;
                        unsigned long long s0_62 = (unsigned long long)h_noise[56] * 3528531795ull;
                        unsigned long long s1_62 = (unsigned long long)h_noise[58] * 3449720151ull;
                        h_noise[56] = (unsigned int)(s1_62 >> 32ull) ^ h_noise[57] ^ key0_1;
                        h_noise[57] = (unsigned int)s1_62;
                        h_noise[58] = (unsigned int)(s0_62 >> 32ull) ^ h_noise[59] ^ key1_1;
                        h_noise[59] = (unsigned int)s0_62;
                        unsigned long long s0_63 = (unsigned long long)h_noise[60] * 3528531795ull;
                        unsigned long long s1_63 = (unsigned long long)h_noise[62] * 3449720151ull;
                        h_noise[60] = (unsigned int)(s1_63 >> 32ull) ^ h_noise[61] ^ key0_1;
                        h_noise[61] = (unsigned int)s1_63;
                        h_noise[62] = (unsigned int)(s0_63 >> 32ull) ^ h_noise[63] ^ key1_1;
                        h_noise[63] = (unsigned int)s0_63;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
#endif
                        unsigned long long s0_64 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_64 = (unsigned long long)h_noise[2] * 3449720151ull;
#if __CUDA_ARCH__ == 900
                        h_noise[0] = (unsigned int)(s1_64 >> 32ull) ^ h_noise[1] ^ key0_3_1;
#else
                        h_noise[0] = (unsigned int)(s1_64 >> 32ull) ^ h_noise[1] ^ key0_1;
#endif
                        h_noise[1] = (unsigned int)s1_64;
#if __CUDA_ARCH__ == 900
                        h_noise[2] = (unsigned int)(s0_64 >> 32ull) ^ h_noise[3] ^ key1_3_1;
#else
                        h_noise[2] = (unsigned int)(s0_64 >> 32ull) ^ h_noise[3] ^ key1_1;
#endif
                        h_noise[3] = (unsigned int)s0_64;
                        unsigned long long s0_65 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_65 = (unsigned long long)h_noise[6] * 3449720151ull;
#if __CUDA_ARCH__ == 900
                        h_noise[4] = (unsigned int)(s1_65 >> 32ull) ^ h_noise[5] ^ key0_3_1;
#else
                        h_noise[4] = (unsigned int)(s1_65 >> 32ull) ^ h_noise[5] ^ key0_1;
#endif
                        h_noise[5] = (unsigned int)s1_65;
#if __CUDA_ARCH__ == 900
                        h_noise[6] = (unsigned int)(s0_65 >> 32ull) ^ h_noise[7] ^ key1_3_1;
#else
                        h_noise[6] = (unsigned int)(s0_65 >> 32ull) ^ h_noise[7] ^ key1_1;
#endif
                        h_noise[7] = (unsigned int)s0_65;
                        unsigned long long s0_66 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_66 = (unsigned long long)h_noise[10] * 3449720151ull;
#if __CUDA_ARCH__ == 900
                        h_noise[8] = (unsigned int)(s1_66 >> 32ull) ^ h_noise[9] ^ key0_3_1;
#else
                        h_noise[8] = (unsigned int)(s1_66 >> 32ull) ^ h_noise[9] ^ key0_1;
#endif
                        h_noise[9] = (unsigned int)s1_66;
#if __CUDA_ARCH__ == 900
                        h_noise[10] = (unsigned int)(s0_66 >> 32ull) ^ h_noise[11] ^ key1_3_1;
#else
                        h_noise[10] = (unsigned int)(s0_66 >> 32ull) ^ h_noise[11] ^ key1_1;
#endif
                        h_noise[11] = (unsigned int)s0_66;
                        unsigned long long s0_67 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_67 = (unsigned long long)h_noise[14] * 3449720151ull;
#if __CUDA_ARCH__ == 900
                        h_noise[12] = (unsigned int)(s1_67 >> 32ull) ^ h_noise[13] ^ key0_3_1;
#else
                        h_noise[12] = (unsigned int)(s1_67 >> 32ull) ^ h_noise[13] ^ key0_1;
#endif
                        h_noise[13] = (unsigned int)s1_67;
#if __CUDA_ARCH__ == 900
                        h_noise[14] = (unsigned int)(s0_67 >> 32ull) ^ h_noise[15] ^ key1_3_1;
#else
                        h_noise[14] = (unsigned int)(s0_67 >> 32ull) ^ h_noise[15] ^ key1_1;
#endif
                        h_noise[15] = (unsigned int)s0_67;
#if __CUDA_ARCH__ == 900
                        key0_3_1 = key0_3_1 + 2654435769u;
                        key1_3_1 = key1_3_1 + 3144134277u;
                        unsigned long long s0_68 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_68 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_68 >> 32ull) ^ h_noise[1] ^ key0_3_1;
                        h_noise[1] = (unsigned int)s1_68;
                        h_noise[2] = (unsigned int)(s0_68 >> 32ull) ^ h_noise[3] ^ key1_3_1;
                        h_noise[3] = (unsigned int)s0_68;
                        unsigned long long s0_69 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_69 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_69 >> 32ull) ^ h_noise[5] ^ key0_3_1;
                        h_noise[5] = (unsigned int)s1_69;
                        h_noise[6] = (unsigned int)(s0_69 >> 32ull) ^ h_noise[7] ^ key1_3_1;
                        h_noise[7] = (unsigned int)s0_69;
                        unsigned long long s0_70 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_70 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_70 >> 32ull) ^ h_noise[9] ^ key0_3_1;
                        h_noise[9] = (unsigned int)s1_70;
                        h_noise[10] = (unsigned int)(s0_70 >> 32ull) ^ h_noise[11] ^ key1_3_1;
                        h_noise[11] = (unsigned int)s0_70;
                        unsigned long long s0_71 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_71 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_71 >> 32ull) ^ h_noise[13] ^ key0_3_1;
                        h_noise[13] = (unsigned int)s1_71;
                        h_noise[14] = (unsigned int)(s0_71 >> 32ull) ^ h_noise[15] ^ key1_3_1;
                        h_noise[15] = (unsigned int)s0_71;
                        key0_3_1 = key0_3_1 + 2654435769u;
                        key1_3_1 = key1_3_1 + 3144134277u;
                        #pragma unroll 1
                        for (int _round_5 = 5; _round_5 < philox_rounds; _round_5++) {
                            unsigned long long t0_20 = (unsigned long long)h_noise[0] * 3528531795ull;
                            unsigned long long t1_20 = (unsigned long long)h_noise[2] * 3449720151ull;
                            h_noise[0] = (unsigned int)(t1_20 >> 32ull) ^ h_noise[1] ^ key0_3_1;
                            h_noise[1] = (unsigned int)t1_20;
                            h_noise[2] = (unsigned int)(t0_20 >> 32ull) ^ h_noise[3] ^ key1_3_1;
                            h_noise[3] = (unsigned int)t0_20;
                            unsigned long long t0_21 = (unsigned long long)h_noise[4] * 3528531795ull;
                            unsigned long long t1_21 = (unsigned long long)h_noise[6] * 3449720151ull;
                            h_noise[4] = (unsigned int)(t1_21 >> 32ull) ^ h_noise[5] ^ key0_3_1;
                            h_noise[5] = (unsigned int)t1_21;
                            h_noise[6] = (unsigned int)(t0_21 >> 32ull) ^ h_noise[7] ^ key1_3_1;
                            h_noise[7] = (unsigned int)t0_21;
                            unsigned long long t0_22 = (unsigned long long)h_noise[8] * 3528531795ull;
                            unsigned long long t1_22 = (unsigned long long)h_noise[10] * 3449720151ull;
                            h_noise[8] = (unsigned int)(t1_22 >> 32ull) ^ h_noise[9] ^ key0_3_1;
                            h_noise[9] = (unsigned int)t1_22;
                            h_noise[10] = (unsigned int)(t0_22 >> 32ull) ^ h_noise[11] ^ key1_3_1;
                            h_noise[11] = (unsigned int)t0_22;
                            unsigned long long t0_23 = (unsigned long long)h_noise[12] * 3528531795ull;
                            unsigned long long t1_23 = (unsigned long long)h_noise[14] * 3449720151ull;
                            h_noise[12] = (unsigned int)(t1_23 >> 32ull) ^ h_noise[13] ^ key0_3_1;
                            h_noise[13] = (unsigned int)t1_23;
                            h_noise[14] = (unsigned int)(t0_23 >> 32ull) ^ h_noise[15] ^ key1_3_1;
                            h_noise[15] = (unsigned int)t0_23;
                            key0_3_1 = key0_3_1 + 2654435769u;
                            key1_3_1 = key1_3_1 + 3144134277u;
                        }
                        float _abs_1263;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1263) : "f"(S[32]));
                        unsigned int s_158 = __as_u32(mul_f32_rz_noftz(_abs_1263, __uint_as_float(125829120)));
                        unsigned int _min_1264 = ((s_158) < (260046848) ? (s_158) : (260046848));
                        unsigned int h_160 = _min_1264 + (h_noise[0] & 8191) >> 13;
                        unsigned int mag_158 = ((s_158 <= 2139095040) ? h_160 : 32256);
                        float _abs_1265;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1265) : "f"(S[33]));
                        unsigned int s_159 = __as_u32(mul_f32_rz_noftz(_abs_1265, __uint_as_float(125829120)));
                        unsigned int _min_1266 = ((s_159) < (260046848) ? (s_159) : (260046848));
                        unsigned int h_161 = _min_1266 + ((h_noise[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_159 = ((s_159 <= 2139095040) ? h_161 : 32256);
                        uint32_t mags_78;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_78) : "r"(mag_158), "r"(mag_159));
                        uint32_t signs_78;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_78) : "r"(__as_u32(S[32])), "r"(__as_u32(S[33])));
                        unsigned int word_78 = signs_78 & 2147516416u | mags_78;
                        h_words[0] = word_78;
                        float _abs_1267;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1267) : "f"(S[34]));
                        unsigned int s_160 = __as_u32(mul_f32_rz_noftz(_abs_1267, __uint_as_float(125829120)));
                        unsigned int _min_1268 = ((s_160) < (260046848) ? (s_160) : (260046848));
                        unsigned int h_162 = _min_1268 + (h_noise[1] & 8191) >> 13;
                        unsigned int mag_160 = ((s_160 <= 2139095040) ? h_162 : 32256);
                        float _abs_1269;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1269) : "f"(S[35]));
                        unsigned int s_161 = __as_u32(mul_f32_rz_noftz(_abs_1269, __uint_as_float(125829120)));
                        unsigned int _min_1270 = ((s_161) < (260046848) ? (s_161) : (260046848));
                        unsigned int h_163 = _min_1270 + ((h_noise[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_161 = ((s_161 <= 2139095040) ? h_163 : 32256);
                        uint32_t mags_79;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_79) : "r"(mag_160), "r"(mag_161));
                        uint32_t signs_79;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_79) : "r"(__as_u32(S[34])), "r"(__as_u32(S[35])));
                        unsigned int word_79 = signs_79 & 2147516416u | mags_79;
                        h_words[1] = word_79;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                        float _abs_1271;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1271) : "f"(S[48]));
                        unsigned int s_162 = __as_u32(mul_f32_rz_noftz(_abs_1271, __uint_as_float(125829120)));
                        unsigned int _min_1272 = ((s_162) < (260046848) ? (s_162) : (260046848));
                        unsigned int h_164 = _min_1272 + (h_noise[2] & 8191) >> 13;
                        unsigned int mag_162 = ((s_162 <= 2139095040) ? h_164 : 32256);
                        float _abs_1273;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1273) : "f"(S[49]));
                        unsigned int s_163 = __as_u32(mul_f32_rz_noftz(_abs_1273, __uint_as_float(125829120)));
                        unsigned int _min_1274 = ((s_163) < (260046848) ? (s_163) : (260046848));
                        unsigned int h_165 = _min_1274 + ((h_noise[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_163 = ((s_163 <= 2139095040) ? h_165 : 32256);
                        uint32_t mags_80;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_80) : "r"(mag_162), "r"(mag_163));
                        uint32_t signs_80;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_80) : "r"(__as_u32(S[48])), "r"(__as_u32(S[49])));
                        unsigned int word_80 = signs_80 & 2147516416u | mags_80;
                        h_words[0] = word_80;
                        float _abs_1275;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1275) : "f"(S[50]));
                        unsigned int s_164 = __as_u32(mul_f32_rz_noftz(_abs_1275, __uint_as_float(125829120)));
                        unsigned int _min_1276 = ((s_164) < (260046848) ? (s_164) : (260046848));
                        unsigned int h_166 = _min_1276 + (h_noise[3] & 8191) >> 13;
                        unsigned int mag_164 = ((s_164 <= 2139095040) ? h_166 : 32256);
                        float _abs_1277;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1277) : "f"(S[51]));
                        unsigned int s_165 = __as_u32(mul_f32_rz_noftz(_abs_1277, __uint_as_float(125829120)));
                        unsigned int _min_1278 = ((s_165) < (260046848) ? (s_165) : (260046848));
                        unsigned int h_167 = _min_1278 + ((h_noise[3] & 536805376) >> 16) >> 13;
                        unsigned int mag_165 = ((s_165 <= 2139095040) ? h_167 : 32256);
                        uint32_t mags_81;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_81) : "r"(mag_164), "r"(mag_165));
                        uint32_t signs_81;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_81) : "r"(__as_u32(S[50])), "r"(__as_u32(S[51])));
                        unsigned int word_81 = signs_81 & 2147516416u | mags_81;
                        h_words[1] = word_81;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        float _abs_1279;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1279) : "f"(S[36]));
                        unsigned int s_166 = __as_u32(mul_f32_rz_noftz(_abs_1279, __uint_as_float(125829120)));
                        unsigned int _min_1280 = ((s_166) < (260046848) ? (s_166) : (260046848));
                        unsigned int h_168 = _min_1280 + (h_noise[4] & 8191) >> 13;
                        unsigned int mag_166 = ((s_166 <= 2139095040) ? h_168 : 32256);
                        float _abs_1281;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1281) : "f"(S[37]));
                        unsigned int s_167 = __as_u32(mul_f32_rz_noftz(_abs_1281, __uint_as_float(125829120)));
                        unsigned int _min_1282 = ((s_167) < (260046848) ? (s_167) : (260046848));
                        unsigned int h_169 = _min_1282 + ((h_noise[4] & 536805376) >> 16) >> 13;
                        unsigned int mag_167 = ((s_167 <= 2139095040) ? h_169 : 32256);
                        uint32_t mags_82;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_82) : "r"(mag_166), "r"(mag_167));
                        uint32_t signs_82;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_82) : "r"(__as_u32(S[36])), "r"(__as_u32(S[37])));
                        unsigned int word_82 = signs_82 & 2147516416u | mags_82;
                        h_words[0] = word_82;
                        float _abs_1283;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1283) : "f"(S[38]));
                        unsigned int s_168 = __as_u32(mul_f32_rz_noftz(_abs_1283, __uint_as_float(125829120)));
                        unsigned int _min_1284 = ((s_168) < (260046848) ? (s_168) : (260046848));
                        unsigned int h_170 = _min_1284 + (h_noise[5] & 8191) >> 13;
                        unsigned int mag_168 = ((s_168 <= 2139095040) ? h_170 : 32256);
                        float _abs_1285;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1285) : "f"(S[39]));
                        unsigned int s_169 = __as_u32(mul_f32_rz_noftz(_abs_1285, __uint_as_float(125829120)));
                        unsigned int _min_1286 = ((s_169) < (260046848) ? (s_169) : (260046848));
                        unsigned int h_171 = _min_1286 + ((h_noise[5] & 536805376) >> 16) >> 13;
                        unsigned int mag_169 = ((s_169 <= 2139095040) ? h_171 : 32256);
                        uint32_t mags_83;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_83) : "r"(mag_168), "r"(mag_169));
                        uint32_t signs_83;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_83) : "r"(__as_u32(S[38])), "r"(__as_u32(S[39])));
                        unsigned int word_83 = signs_83 & 2147516416u | mags_83;
                        h_words[1] = word_83;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 16ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        float _abs_1287;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1287) : "f"(S[52]));
                        unsigned int s_170 = __as_u32(mul_f32_rz_noftz(_abs_1287, __uint_as_float(125829120)));
                        unsigned int _min_1288 = ((s_170) < (260046848) ? (s_170) : (260046848));
                        unsigned int h_172 = _min_1288 + (h_noise[6] & 8191) >> 13;
                        unsigned int mag_170 = ((s_170 <= 2139095040) ? h_172 : 32256);
                        float _abs_1289;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1289) : "f"(S[53]));
                        unsigned int s_171 = __as_u32(mul_f32_rz_noftz(_abs_1289, __uint_as_float(125829120)));
                        unsigned int _min_1290 = ((s_171) < (260046848) ? (s_171) : (260046848));
                        unsigned int h_173 = _min_1290 + ((h_noise[6] & 536805376) >> 16) >> 13;
                        unsigned int mag_171 = ((s_171 <= 2139095040) ? h_173 : 32256);
                        uint32_t mags_84;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_84) : "r"(mag_170), "r"(mag_171));
                        uint32_t signs_84;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_84) : "r"(__as_u32(S[52])), "r"(__as_u32(S[53])));
                        unsigned int word_84 = signs_84 & 2147516416u | mags_84;
                        h_words[0] = word_84;
                        float _abs_1291;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1291) : "f"(S[54]));
                        unsigned int s_172 = __as_u32(mul_f32_rz_noftz(_abs_1291, __uint_as_float(125829120)));
                        unsigned int _min_1292 = ((s_172) < (260046848) ? (s_172) : (260046848));
                        unsigned int h_174 = _min_1292 + (h_noise[7] & 8191) >> 13;
                        unsigned int mag_172 = ((s_172 <= 2139095040) ? h_174 : 32256);
                        float _abs_1293;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1293) : "f"(S[55]));
                        unsigned int s_173 = __as_u32(mul_f32_rz_noftz(_abs_1293, __uint_as_float(125829120)));
                        unsigned int _min_1294 = ((s_173) < (260046848) ? (s_173) : (260046848));
                        unsigned int h_175 = _min_1294 + ((h_noise[7] & 536805376) >> 16) >> 13;
                        unsigned int mag_173 = ((s_173 <= 2139095040) ? h_175 : 32256);
                        uint32_t mags_85;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_85) : "r"(mag_172), "r"(mag_173));
                        uint32_t signs_85;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_85) : "r"(__as_u32(S[54])), "r"(__as_u32(S[55])));
                        unsigned int word_85 = signs_85 & 2147516416u | mags_85;
                        h_words[1] = word_85;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 80ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        float _abs_1295;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1295) : "f"(S[40]));
                        unsigned int s_174 = __as_u32(mul_f32_rz_noftz(_abs_1295, __uint_as_float(125829120)));
                        unsigned int _min_1296 = ((s_174) < (260046848) ? (s_174) : (260046848));
                        unsigned int h_176 = _min_1296 + (h_noise[8] & 8191) >> 13;
                        unsigned int mag_174 = ((s_174 <= 2139095040) ? h_176 : 32256);
                        float _abs_1297;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1297) : "f"(S[41]));
                        unsigned int s_175 = __as_u32(mul_f32_rz_noftz(_abs_1297, __uint_as_float(125829120)));
                        unsigned int _min_1298 = ((s_175) < (260046848) ? (s_175) : (260046848));
                        unsigned int h_177 = _min_1298 + ((h_noise[8] & 536805376) >> 16) >> 13;
                        unsigned int mag_175 = ((s_175 <= 2139095040) ? h_177 : 32256);
                        uint32_t mags_86;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_86) : "r"(mag_174), "r"(mag_175));
                        uint32_t signs_86;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_86) : "r"(__as_u32(S[40])), "r"(__as_u32(S[41])));
                        unsigned int word_86 = signs_86 & 2147516416u | mags_86;
                        h_words[0] = word_86;
                        float _abs_1299;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1299) : "f"(S[42]));
                        unsigned int s_176 = __as_u32(mul_f32_rz_noftz(_abs_1299, __uint_as_float(125829120)));
                        unsigned int _min_1300 = ((s_176) < (260046848) ? (s_176) : (260046848));
                        unsigned int h_178 = _min_1300 + (h_noise[9] & 8191) >> 13;
                        unsigned int mag_176 = ((s_176 <= 2139095040) ? h_178 : 32256);
                        float _abs_1301;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1301) : "f"(S[43]));
                        unsigned int s_177 = __as_u32(mul_f32_rz_noftz(_abs_1301, __uint_as_float(125829120)));
                        unsigned int _min_1302 = ((s_177) < (260046848) ? (s_177) : (260046848));
                        unsigned int h_179 = _min_1302 + ((h_noise[9] & 536805376) >> 16) >> 13;
                        unsigned int mag_177 = ((s_177 <= 2139095040) ? h_179 : 32256);
                        uint32_t mags_87;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_87) : "r"(mag_176), "r"(mag_177));
                        uint32_t signs_87;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_87) : "r"(__as_u32(S[42])), "r"(__as_u32(S[43])));
                        unsigned int word_87 = signs_87 & 2147516416u | mags_87;
                        h_words[1] = word_87;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        float _abs_1303;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1303) : "f"(S[56]));
                        unsigned int s_178 = __as_u32(mul_f32_rz_noftz(_abs_1303, __uint_as_float(125829120)));
                        unsigned int _min_1304 = ((s_178) < (260046848) ? (s_178) : (260046848));
                        unsigned int h_180 = _min_1304 + (h_noise[10] & 8191) >> 13;
                        unsigned int mag_178 = ((s_178 <= 2139095040) ? h_180 : 32256);
                        float _abs_1305;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1305) : "f"(S[57]));
                        unsigned int s_179 = __as_u32(mul_f32_rz_noftz(_abs_1305, __uint_as_float(125829120)));
                        unsigned int _min_1306 = ((s_179) < (260046848) ? (s_179) : (260046848));
                        unsigned int h_181 = _min_1306 + ((h_noise[10] & 536805376) >> 16) >> 13;
                        unsigned int mag_179 = ((s_179 <= 2139095040) ? h_181 : 32256);
                        uint32_t mags_88;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_88) : "r"(mag_178), "r"(mag_179));
                        uint32_t signs_88;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_88) : "r"(__as_u32(S[56])), "r"(__as_u32(S[57])));
                        unsigned int word_88 = signs_88 & 2147516416u | mags_88;
                        h_words[0] = word_88;
                        float _abs_1307;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1307) : "f"(S[58]));
                        unsigned int s_180 = __as_u32(mul_f32_rz_noftz(_abs_1307, __uint_as_float(125829120)));
                        unsigned int _min_1308 = ((s_180) < (260046848) ? (s_180) : (260046848));
                        unsigned int h_182 = _min_1308 + (h_noise[11] & 8191) >> 13;
                        unsigned int mag_180 = ((s_180 <= 2139095040) ? h_182 : 32256);
                        float _abs_1309;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1309) : "f"(S[59]));
                        unsigned int s_181 = __as_u32(mul_f32_rz_noftz(_abs_1309, __uint_as_float(125829120)));
                        unsigned int _min_1310 = ((s_181) < (260046848) ? (s_181) : (260046848));
                        unsigned int h_183 = _min_1310 + ((h_noise[11] & 536805376) >> 16) >> 13;
                        unsigned int mag_181 = ((s_181 <= 2139095040) ? h_183 : 32256);
                        uint32_t mags_89;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_89) : "r"(mag_180), "r"(mag_181));
                        uint32_t signs_89;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_89) : "r"(__as_u32(S[58])), "r"(__as_u32(S[59])));
                        unsigned int word_89 = signs_89 & 2147516416u | mags_89;
                        h_words[1] = word_89;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        float _abs_1311;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1311) : "f"(S[44]));
                        unsigned int s_182 = __as_u32(mul_f32_rz_noftz(_abs_1311, __uint_as_float(125829120)));
                        unsigned int _min_1312 = ((s_182) < (260046848) ? (s_182) : (260046848));
                        unsigned int h_184 = _min_1312 + (h_noise[12] & 8191) >> 13;
                        unsigned int mag_182 = ((s_182 <= 2139095040) ? h_184 : 32256);
                        float _abs_1313;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1313) : "f"(S[45]));
                        unsigned int s_183 = __as_u32(mul_f32_rz_noftz(_abs_1313, __uint_as_float(125829120)));
                        unsigned int _min_1314 = ((s_183) < (260046848) ? (s_183) : (260046848));
                        unsigned int h_185 = _min_1314 + ((h_noise[12] & 536805376) >> 16) >> 13;
                        unsigned int mag_183 = ((s_183 <= 2139095040) ? h_185 : 32256);
                        uint32_t mags_90;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_90) : "r"(mag_182), "r"(mag_183));
                        uint32_t signs_90;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_90) : "r"(__as_u32(S[44])), "r"(__as_u32(S[45])));
                        unsigned int word_90 = signs_90 & 2147516416u | mags_90;
                        h_words[0] = word_90;
                        float _abs_1315;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1315) : "f"(S[46]));
                        unsigned int s_184 = __as_u32(mul_f32_rz_noftz(_abs_1315, __uint_as_float(125829120)));
                        unsigned int _min_1316 = ((s_184) < (260046848) ? (s_184) : (260046848));
                        unsigned int h_186 = _min_1316 + (h_noise[13] & 8191) >> 13;
                        unsigned int mag_184 = ((s_184 <= 2139095040) ? h_186 : 32256);
                        float _abs_1317;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1317) : "f"(S[47]));
                        unsigned int s_185 = __as_u32(mul_f32_rz_noftz(_abs_1317, __uint_as_float(125829120)));
                        unsigned int _min_1318 = ((s_185) < (260046848) ? (s_185) : (260046848));
                        unsigned int h_187 = _min_1318 + ((h_noise[13] & 536805376) >> 16) >> 13;
                        unsigned int mag_185 = ((s_185 <= 2139095040) ? h_187 : 32256);
                        uint32_t mags_91;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_91) : "r"(mag_184), "r"(mag_185));
                        uint32_t signs_91;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_91) : "r"(__as_u32(S[46])), "r"(__as_u32(S[47])));
                        unsigned int word_91 = signs_91 & 2147516416u | mags_91;
                        h_words[1] = word_91;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 48ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        float _abs_1319;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1319) : "f"(S[60]));
                        unsigned int s_186 = __as_u32(mul_f32_rz_noftz(_abs_1319, __uint_as_float(125829120)));
                        unsigned int _min_1320 = ((s_186) < (260046848) ? (s_186) : (260046848));
                        unsigned int h_188 = _min_1320 + (h_noise[14] & 8191) >> 13;
                        unsigned int mag_186 = ((s_186 <= 2139095040) ? h_188 : 32256);
                        float _abs_1321;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1321) : "f"(S[61]));
                        unsigned int s_187 = __as_u32(mul_f32_rz_noftz(_abs_1321, __uint_as_float(125829120)));
                        unsigned int _min_1322 = ((s_187) < (260046848) ? (s_187) : (260046848));
                        unsigned int h_189 = _min_1322 + ((h_noise[14] & 536805376) >> 16) >> 13;
                        unsigned int mag_187 = ((s_187 <= 2139095040) ? h_189 : 32256);
                        uint32_t mags_92;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_92) : "r"(mag_186), "r"(mag_187));
                        uint32_t signs_92;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_92) : "r"(__as_u32(S[60])), "r"(__as_u32(S[61])));
                        unsigned int word_92 = signs_92 & 2147516416u | mags_92;
                        h_words[0] = word_92;
                        float _abs_1323;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1323) : "f"(S[62]));
                        unsigned int s_188 = __as_u32(mul_f32_rz_noftz(_abs_1323, __uint_as_float(125829120)));
                        unsigned int _min_1324 = ((s_188) < (260046848) ? (s_188) : (260046848));
                        unsigned int h_190 = _min_1324 + (h_noise[15] & 8191) >> 13;
                        unsigned int mag_188 = ((s_188 <= 2139095040) ? h_190 : 32256);
                        float _abs_1325;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1325) : "f"(S[63]));
                        unsigned int s_189 = __as_u32(mul_f32_rz_noftz(_abs_1325, __uint_as_float(125829120)));
                        unsigned int _min_1326 = ((s_189) < (260046848) ? (s_189) : (260046848));
                        unsigned int h_191 = _min_1326 + ((h_noise[15] & 536805376) >> 16) >> 13;
                        unsigned int mag_189 = ((s_189 <= 2139095040) ? h_191 : 32256);
                        uint32_t mags_93;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_93) : "r"(mag_188), "r"(mag_189));
                        uint32_t signs_93;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_93) : "r"(__as_u32(S[62])), "r"(__as_u32(S[63])));
                        unsigned int word_93 = signs_93 & 2147516416u | mags_93;
                        h_words[1] = word_93;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 112ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        unsigned int key0_4 = seed_lo;
                        unsigned int key1_4 = seed_hi;
                        long long o_24 = noise_at + 2ll * state_stride_p2;
                        unsigned long long p0_24 = (unsigned long long)(unsigned int)o_24 * 3528531795ull;
                        h_noise[0] = (unsigned int)(o_24 >> 32ll) ^ key0_4;
                        h_noise[2] = (unsigned int)(p0_24 >> 32ull) ^ key1_4;
                        h_noise[3] = (unsigned int)p0_24;
                        long long o_25 = noise_at + 2ll * state_stride_p2 + 16ll;
                        unsigned long long p0_25 = (unsigned long long)(unsigned int)o_25 * 3528531795ull;
                        h_noise[4] = (unsigned int)(o_25 >> 32ll) ^ key0_4;
                        h_noise[6] = (unsigned int)(p0_25 >> 32ull) ^ key1_4;
                        h_noise[7] = (unsigned int)p0_25;
                        long long o_26 = noise_at + 2ll * state_stride_p2 + 32ll;
                        unsigned long long p0_26 = (unsigned long long)(unsigned int)o_26 * 3528531795ull;
                        h_noise[8] = (unsigned int)(o_26 >> 32ll) ^ key0_4;
                        h_noise[10] = (unsigned int)(p0_26 >> 32ull) ^ key1_4;
                        h_noise[11] = (unsigned int)p0_26;
                        long long o_27 = noise_at + 2ll * state_stride_p2 + 48ll;
                        unsigned long long p0_27 = (unsigned long long)(unsigned int)o_27 * 3528531795ull;
                        h_noise[12] = (unsigned int)(o_27 >> 32ll) ^ key0_4;
                        h_noise[14] = (unsigned int)(p0_27 >> 32ull) ^ key1_4;
                        h_noise[15] = (unsigned int)p0_27;
                        key0_4 = key0_4 + 2654435769u;
                        key1_4 = key1_4 + 3144134277u;
                        unsigned long long q0_24 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long q1_24 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(q1_24 >> 32ull) ^ key0_4;
                        h_noise[1] = (unsigned int)q1_24;
                        h_noise[2] = (unsigned int)(q0_24 >> 32ull) ^ h_noise[3] ^ key1_4;
                        h_noise[3] = (unsigned int)q0_24;
                        unsigned long long q0_25 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long q1_25 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(q1_25 >> 32ull) ^ key0_4;
                        h_noise[5] = (unsigned int)q1_25;
                        h_noise[6] = (unsigned int)(q0_25 >> 32ull) ^ h_noise[7] ^ key1_4;
                        h_noise[7] = (unsigned int)q0_25;
                        unsigned long long q0_26 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long q1_26 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(q1_26 >> 32ull) ^ key0_4;
                        h_noise[9] = (unsigned int)q1_26;
                        h_noise[10] = (unsigned int)(q0_26 >> 32ull) ^ h_noise[11] ^ key1_4;
                        h_noise[11] = (unsigned int)q0_26;
                        unsigned long long q0_27 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long q1_27 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(q1_27 >> 32ull) ^ key0_4;
                        h_noise[13] = (unsigned int)q1_27;
                        h_noise[14] = (unsigned int)(q0_27 >> 32ull) ^ h_noise[15] ^ key1_4;
                        h_noise[15] = (unsigned int)q0_27;
                        key0_4 = key0_4 + 2654435769u;
                        key1_4 = key1_4 + 3144134277u;
                        unsigned long long s0_72 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_72 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_72 >> 32ull) ^ h_noise[1] ^ key0_4;
                        h_noise[1] = (unsigned int)s1_72;
                        h_noise[2] = (unsigned int)(s0_72 >> 32ull) ^ h_noise[3] ^ key1_4;
                        h_noise[3] = (unsigned int)s0_72;
                        unsigned long long s0_73 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_73 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_73 >> 32ull) ^ h_noise[5] ^ key0_4;
                        h_noise[5] = (unsigned int)s1_73;
                        h_noise[6] = (unsigned int)(s0_73 >> 32ull) ^ h_noise[7] ^ key1_4;
                        h_noise[7] = (unsigned int)s0_73;
                        unsigned long long s0_74 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_74 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_74 >> 32ull) ^ h_noise[9] ^ key0_4;
                        h_noise[9] = (unsigned int)s1_74;
                        h_noise[10] = (unsigned int)(s0_74 >> 32ull) ^ h_noise[11] ^ key1_4;
                        h_noise[11] = (unsigned int)s0_74;
                        unsigned long long s0_75 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_75 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_75 >> 32ull) ^ h_noise[13] ^ key0_4;
                        h_noise[13] = (unsigned int)s1_75;
                        h_noise[14] = (unsigned int)(s0_75 >> 32ull) ^ h_noise[15] ^ key1_4;
                        h_noise[15] = (unsigned int)s0_75;
                        key0_4 = key0_4 + 2654435769u;
                        key1_4 = key1_4 + 3144134277u;
                        unsigned long long s0_76 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_76 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_76 >> 32ull) ^ h_noise[1] ^ key0_4;
                        h_noise[1] = (unsigned int)s1_76;
                        h_noise[2] = (unsigned int)(s0_76 >> 32ull) ^ h_noise[3] ^ key1_4;
                        h_noise[3] = (unsigned int)s0_76;
                        unsigned long long s0_77 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_77 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_77 >> 32ull) ^ h_noise[5] ^ key0_4;
                        h_noise[5] = (unsigned int)s1_77;
                        h_noise[6] = (unsigned int)(s0_77 >> 32ull) ^ h_noise[7] ^ key1_4;
                        h_noise[7] = (unsigned int)s0_77;
                        unsigned long long s0_78 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_78 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_78 >> 32ull) ^ h_noise[9] ^ key0_4;
                        h_noise[9] = (unsigned int)s1_78;
                        h_noise[10] = (unsigned int)(s0_78 >> 32ull) ^ h_noise[11] ^ key1_4;
                        h_noise[11] = (unsigned int)s0_78;
                        unsigned long long s0_79 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_79 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_79 >> 32ull) ^ h_noise[13] ^ key0_4;
                        h_noise[13] = (unsigned int)s1_79;
                        h_noise[14] = (unsigned int)(s0_79 >> 32ull) ^ h_noise[15] ^ key1_4;
                        h_noise[15] = (unsigned int)s0_79;
                        key0_4 = key0_4 + 2654435769u;
                        key1_4 = key1_4 + 3144134277u;
#else
                        unsigned long long s0_68 = (unsigned long long)h_noise[16] * 3528531795ull;
                        unsigned long long s1_68 = (unsigned long long)h_noise[18] * 3449720151ull;
                        h_noise[16] = (unsigned int)(s1_68 >> 32ull) ^ h_noise[17] ^ key0_1;
                        h_noise[17] = (unsigned int)s1_68;
                        h_noise[18] = (unsigned int)(s0_68 >> 32ull) ^ h_noise[19] ^ key1_1;
                        h_noise[19] = (unsigned int)s0_68;
                        unsigned long long s0_69 = (unsigned long long)h_noise[20] * 3528531795ull;
                        unsigned long long s1_69 = (unsigned long long)h_noise[22] * 3449720151ull;
                        h_noise[20] = (unsigned int)(s1_69 >> 32ull) ^ h_noise[21] ^ key0_1;
                        h_noise[21] = (unsigned int)s1_69;
                        h_noise[22] = (unsigned int)(s0_69 >> 32ull) ^ h_noise[23] ^ key1_1;
                        h_noise[23] = (unsigned int)s0_69;
                        unsigned long long s0_70 = (unsigned long long)h_noise[24] * 3528531795ull;
                        unsigned long long s1_70 = (unsigned long long)h_noise[26] * 3449720151ull;
                        h_noise[24] = (unsigned int)(s1_70 >> 32ull) ^ h_noise[25] ^ key0_1;
                        h_noise[25] = (unsigned int)s1_70;
                        h_noise[26] = (unsigned int)(s0_70 >> 32ull) ^ h_noise[27] ^ key1_1;
                        h_noise[27] = (unsigned int)s0_70;
                        unsigned long long s0_71 = (unsigned long long)h_noise[28] * 3528531795ull;
                        unsigned long long s1_71 = (unsigned long long)h_noise[30] * 3449720151ull;
                        h_noise[28] = (unsigned int)(s1_71 >> 32ull) ^ h_noise[29] ^ key0_1;
                        h_noise[29] = (unsigned int)s1_71;
                        h_noise[30] = (unsigned int)(s0_71 >> 32ull) ^ h_noise[31] ^ key1_1;
                        h_noise[31] = (unsigned int)s0_71;
                        unsigned long long s0_72 = (unsigned long long)h_noise[32] * 3528531795ull;
                        unsigned long long s1_72 = (unsigned long long)h_noise[34] * 3449720151ull;
                        h_noise[32] = (unsigned int)(s1_72 >> 32ull) ^ h_noise[33] ^ key0_1;
                        h_noise[33] = (unsigned int)s1_72;
                        h_noise[34] = (unsigned int)(s0_72 >> 32ull) ^ h_noise[35] ^ key1_1;
                        h_noise[35] = (unsigned int)s0_72;
                        unsigned long long s0_73 = (unsigned long long)h_noise[36] * 3528531795ull;
                        unsigned long long s1_73 = (unsigned long long)h_noise[38] * 3449720151ull;
                        h_noise[36] = (unsigned int)(s1_73 >> 32ull) ^ h_noise[37] ^ key0_1;
                        h_noise[37] = (unsigned int)s1_73;
                        h_noise[38] = (unsigned int)(s0_73 >> 32ull) ^ h_noise[39] ^ key1_1;
                        h_noise[39] = (unsigned int)s0_73;
                        unsigned long long s0_74 = (unsigned long long)h_noise[40] * 3528531795ull;
                        unsigned long long s1_74 = (unsigned long long)h_noise[42] * 3449720151ull;
                        h_noise[40] = (unsigned int)(s1_74 >> 32ull) ^ h_noise[41] ^ key0_1;
                        h_noise[41] = (unsigned int)s1_74;
                        h_noise[42] = (unsigned int)(s0_74 >> 32ull) ^ h_noise[43] ^ key1_1;
                        h_noise[43] = (unsigned int)s0_74;
                        unsigned long long s0_75 = (unsigned long long)h_noise[44] * 3528531795ull;
                        unsigned long long s1_75 = (unsigned long long)h_noise[46] * 3449720151ull;
                        h_noise[44] = (unsigned int)(s1_75 >> 32ull) ^ h_noise[45] ^ key0_1;
                        h_noise[45] = (unsigned int)s1_75;
                        h_noise[46] = (unsigned int)(s0_75 >> 32ull) ^ h_noise[47] ^ key1_1;
                        h_noise[47] = (unsigned int)s0_75;
                        unsigned long long s0_76 = (unsigned long long)h_noise[48] * 3528531795ull;
                        unsigned long long s1_76 = (unsigned long long)h_noise[50] * 3449720151ull;
                        h_noise[48] = (unsigned int)(s1_76 >> 32ull) ^ h_noise[49] ^ key0_1;
                        h_noise[49] = (unsigned int)s1_76;
                        h_noise[50] = (unsigned int)(s0_76 >> 32ull) ^ h_noise[51] ^ key1_1;
                        h_noise[51] = (unsigned int)s0_76;
                        unsigned long long s0_77 = (unsigned long long)h_noise[52] * 3528531795ull;
                        unsigned long long s1_77 = (unsigned long long)h_noise[54] * 3449720151ull;
                        h_noise[52] = (unsigned int)(s1_77 >> 32ull) ^ h_noise[53] ^ key0_1;
                        h_noise[53] = (unsigned int)s1_77;
                        h_noise[54] = (unsigned int)(s0_77 >> 32ull) ^ h_noise[55] ^ key1_1;
                        h_noise[55] = (unsigned int)s0_77;
                        unsigned long long s0_78 = (unsigned long long)h_noise[56] * 3528531795ull;
                        unsigned long long s1_78 = (unsigned long long)h_noise[58] * 3449720151ull;
                        h_noise[56] = (unsigned int)(s1_78 >> 32ull) ^ h_noise[57] ^ key0_1;
                        h_noise[57] = (unsigned int)s1_78;
                        h_noise[58] = (unsigned int)(s0_78 >> 32ull) ^ h_noise[59] ^ key1_1;
                        h_noise[59] = (unsigned int)s0_78;
                        unsigned long long s0_79 = (unsigned long long)h_noise[60] * 3528531795ull;
                        unsigned long long s1_79 = (unsigned long long)h_noise[62] * 3449720151ull;
                        h_noise[60] = (unsigned int)(s1_79 >> 32ull) ^ h_noise[61] ^ key0_1;
                        h_noise[61] = (unsigned int)s1_79;
                        h_noise[62] = (unsigned int)(s0_79 >> 32ull) ^ h_noise[63] ^ key1_1;
                        h_noise[63] = (unsigned int)s0_79;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
#endif
                        unsigned long long s0_80 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_80 = (unsigned long long)h_noise[2] * 3449720151ull;
#if __CUDA_ARCH__ == 900
                        h_noise[0] = (unsigned int)(s1_80 >> 32ull) ^ h_noise[1] ^ key0_4;
#else
                        h_noise[0] = (unsigned int)(s1_80 >> 32ull) ^ h_noise[1] ^ key0_1;
#endif
                        h_noise[1] = (unsigned int)s1_80;
#if __CUDA_ARCH__ == 900
                        h_noise[2] = (unsigned int)(s0_80 >> 32ull) ^ h_noise[3] ^ key1_4;
#else
                        h_noise[2] = (unsigned int)(s0_80 >> 32ull) ^ h_noise[3] ^ key1_1;
#endif
                        h_noise[3] = (unsigned int)s0_80;
                        unsigned long long s0_81 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_81 = (unsigned long long)h_noise[6] * 3449720151ull;
#if __CUDA_ARCH__ == 900
                        h_noise[4] = (unsigned int)(s1_81 >> 32ull) ^ h_noise[5] ^ key0_4;
#else
                        h_noise[4] = (unsigned int)(s1_81 >> 32ull) ^ h_noise[5] ^ key0_1;
#endif
                        h_noise[5] = (unsigned int)s1_81;
#if __CUDA_ARCH__ == 900
                        h_noise[6] = (unsigned int)(s0_81 >> 32ull) ^ h_noise[7] ^ key1_4;
#else
                        h_noise[6] = (unsigned int)(s0_81 >> 32ull) ^ h_noise[7] ^ key1_1;
#endif
                        h_noise[7] = (unsigned int)s0_81;
                        unsigned long long s0_82 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_82 = (unsigned long long)h_noise[10] * 3449720151ull;
#if __CUDA_ARCH__ == 900
                        h_noise[8] = (unsigned int)(s1_82 >> 32ull) ^ h_noise[9] ^ key0_4;
#else
                        h_noise[8] = (unsigned int)(s1_82 >> 32ull) ^ h_noise[9] ^ key0_1;
#endif
                        h_noise[9] = (unsigned int)s1_82;
#if __CUDA_ARCH__ == 900
                        h_noise[10] = (unsigned int)(s0_82 >> 32ull) ^ h_noise[11] ^ key1_4;
#else
                        h_noise[10] = (unsigned int)(s0_82 >> 32ull) ^ h_noise[11] ^ key1_1;
#endif
                        h_noise[11] = (unsigned int)s0_82;
                        unsigned long long s0_83 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_83 = (unsigned long long)h_noise[14] * 3449720151ull;
#if __CUDA_ARCH__ == 900
                        h_noise[12] = (unsigned int)(s1_83 >> 32ull) ^ h_noise[13] ^ key0_4;
#else
                        h_noise[12] = (unsigned int)(s1_83 >> 32ull) ^ h_noise[13] ^ key0_1;
#endif
                        h_noise[13] = (unsigned int)s1_83;
#if __CUDA_ARCH__ == 900
                        h_noise[14] = (unsigned int)(s0_83 >> 32ull) ^ h_noise[15] ^ key1_4;
#else
                        h_noise[14] = (unsigned int)(s0_83 >> 32ull) ^ h_noise[15] ^ key1_1;
#endif
                        h_noise[15] = (unsigned int)s0_83;
#if __CUDA_ARCH__ == 900
                        key0_4 = key0_4 + 2654435769u;
                        key1_4 = key1_4 + 3144134277u;
#else
                        unsigned long long s0_84 = (unsigned long long)h_noise[16] * 3528531795ull;
                        unsigned long long s1_84 = (unsigned long long)h_noise[18] * 3449720151ull;
                        h_noise[16] = (unsigned int)(s1_84 >> 32ull) ^ h_noise[17] ^ key0_1;
                        h_noise[17] = (unsigned int)s1_84;
                        h_noise[18] = (unsigned int)(s0_84 >> 32ull) ^ h_noise[19] ^ key1_1;
                        h_noise[19] = (unsigned int)s0_84;
                        unsigned long long s0_85 = (unsigned long long)h_noise[20] * 3528531795ull;
                        unsigned long long s1_85 = (unsigned long long)h_noise[22] * 3449720151ull;
                        h_noise[20] = (unsigned int)(s1_85 >> 32ull) ^ h_noise[21] ^ key0_1;
                        h_noise[21] = (unsigned int)s1_85;
                        h_noise[22] = (unsigned int)(s0_85 >> 32ull) ^ h_noise[23] ^ key1_1;
                        h_noise[23] = (unsigned int)s0_85;
                        unsigned long long s0_86 = (unsigned long long)h_noise[24] * 3528531795ull;
                        unsigned long long s1_86 = (unsigned long long)h_noise[26] * 3449720151ull;
                        h_noise[24] = (unsigned int)(s1_86 >> 32ull) ^ h_noise[25] ^ key0_1;
                        h_noise[25] = (unsigned int)s1_86;
                        h_noise[26] = (unsigned int)(s0_86 >> 32ull) ^ h_noise[27] ^ key1_1;
                        h_noise[27] = (unsigned int)s0_86;
                        unsigned long long s0_87 = (unsigned long long)h_noise[28] * 3528531795ull;
                        unsigned long long s1_87 = (unsigned long long)h_noise[30] * 3449720151ull;
                        h_noise[28] = (unsigned int)(s1_87 >> 32ull) ^ h_noise[29] ^ key0_1;
                        h_noise[29] = (unsigned int)s1_87;
                        h_noise[30] = (unsigned int)(s0_87 >> 32ull) ^ h_noise[31] ^ key1_1;
                        h_noise[31] = (unsigned int)s0_87;
                        unsigned long long s0_88 = (unsigned long long)h_noise[32] * 3528531795ull;
                        unsigned long long s1_88 = (unsigned long long)h_noise[34] * 3449720151ull;
                        h_noise[32] = (unsigned int)(s1_88 >> 32ull) ^ h_noise[33] ^ key0_1;
                        h_noise[33] = (unsigned int)s1_88;
                        h_noise[34] = (unsigned int)(s0_88 >> 32ull) ^ h_noise[35] ^ key1_1;
                        h_noise[35] = (unsigned int)s0_88;
                        unsigned long long s0_89 = (unsigned long long)h_noise[36] * 3528531795ull;
                        unsigned long long s1_89 = (unsigned long long)h_noise[38] * 3449720151ull;
                        h_noise[36] = (unsigned int)(s1_89 >> 32ull) ^ h_noise[37] ^ key0_1;
                        h_noise[37] = (unsigned int)s1_89;
                        h_noise[38] = (unsigned int)(s0_89 >> 32ull) ^ h_noise[39] ^ key1_1;
                        h_noise[39] = (unsigned int)s0_89;
                        unsigned long long s0_90 = (unsigned long long)h_noise[40] * 3528531795ull;
                        unsigned long long s1_90 = (unsigned long long)h_noise[42] * 3449720151ull;
                        h_noise[40] = (unsigned int)(s1_90 >> 32ull) ^ h_noise[41] ^ key0_1;
                        h_noise[41] = (unsigned int)s1_90;
                        h_noise[42] = (unsigned int)(s0_90 >> 32ull) ^ h_noise[43] ^ key1_1;
                        h_noise[43] = (unsigned int)s0_90;
                        unsigned long long s0_91 = (unsigned long long)h_noise[44] * 3528531795ull;
                        unsigned long long s1_91 = (unsigned long long)h_noise[46] * 3449720151ull;
                        h_noise[44] = (unsigned int)(s1_91 >> 32ull) ^ h_noise[45] ^ key0_1;
                        h_noise[45] = (unsigned int)s1_91;
                        h_noise[46] = (unsigned int)(s0_91 >> 32ull) ^ h_noise[47] ^ key1_1;
                        h_noise[47] = (unsigned int)s0_91;
                        unsigned long long s0_92 = (unsigned long long)h_noise[48] * 3528531795ull;
                        unsigned long long s1_92 = (unsigned long long)h_noise[50] * 3449720151ull;
                        h_noise[48] = (unsigned int)(s1_92 >> 32ull) ^ h_noise[49] ^ key0_1;
                        h_noise[49] = (unsigned int)s1_92;
                        h_noise[50] = (unsigned int)(s0_92 >> 32ull) ^ h_noise[51] ^ key1_1;
                        h_noise[51] = (unsigned int)s0_92;
                        unsigned long long s0_93 = (unsigned long long)h_noise[52] * 3528531795ull;
                        unsigned long long s1_93 = (unsigned long long)h_noise[54] * 3449720151ull;
                        h_noise[52] = (unsigned int)(s1_93 >> 32ull) ^ h_noise[53] ^ key0_1;
                        h_noise[53] = (unsigned int)s1_93;
                        h_noise[54] = (unsigned int)(s0_93 >> 32ull) ^ h_noise[55] ^ key1_1;
                        h_noise[55] = (unsigned int)s0_93;
                        unsigned long long s0_94 = (unsigned long long)h_noise[56] * 3528531795ull;
                        unsigned long long s1_94 = (unsigned long long)h_noise[58] * 3449720151ull;
                        h_noise[56] = (unsigned int)(s1_94 >> 32ull) ^ h_noise[57] ^ key0_1;
                        h_noise[57] = (unsigned int)s1_94;
                        h_noise[58] = (unsigned int)(s0_94 >> 32ull) ^ h_noise[59] ^ key1_1;
                        h_noise[59] = (unsigned int)s0_94;
                        unsigned long long s0_95 = (unsigned long long)h_noise[60] * 3528531795ull;
                        unsigned long long s1_95 = (unsigned long long)h_noise[62] * 3449720151ull;
                        h_noise[60] = (unsigned int)(s1_95 >> 32ull) ^ h_noise[61] ^ key0_1;
                        h_noise[61] = (unsigned int)s1_95;
                        h_noise[62] = (unsigned int)(s0_95 >> 32ull) ^ h_noise[63] ^ key1_1;
                        h_noise[63] = (unsigned int)s0_95;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
#endif
                        #pragma unroll 1
#if __CUDA_ARCH__ == 900
                        for (int _round_6 = 5; _round_6 < philox_rounds; _round_6++) {
                            unsigned long long t0_24 = (unsigned long long)h_noise[0] * 3528531795ull;
                            unsigned long long t1_24 = (unsigned long long)h_noise[2] * 3449720151ull;
                            h_noise[0] = (unsigned int)(t1_24 >> 32ull) ^ h_noise[1] ^ key0_4;
                            h_noise[1] = (unsigned int)t1_24;
                            h_noise[2] = (unsigned int)(t0_24 >> 32ull) ^ h_noise[3] ^ key1_4;
                            h_noise[3] = (unsigned int)t0_24;
                            unsigned long long t0_25 = (unsigned long long)h_noise[4] * 3528531795ull;
                            unsigned long long t1_25 = (unsigned long long)h_noise[6] * 3449720151ull;
                            h_noise[4] = (unsigned int)(t1_25 >> 32ull) ^ h_noise[5] ^ key0_4;
                            h_noise[5] = (unsigned int)t1_25;
                            h_noise[6] = (unsigned int)(t0_25 >> 32ull) ^ h_noise[7] ^ key1_4;
                            h_noise[7] = (unsigned int)t0_25;
                            unsigned long long t0_26 = (unsigned long long)h_noise[8] * 3528531795ull;
                            unsigned long long t1_26 = (unsigned long long)h_noise[10] * 3449720151ull;
                            h_noise[8] = (unsigned int)(t1_26 >> 32ull) ^ h_noise[9] ^ key0_4;
                            h_noise[9] = (unsigned int)t1_26;
                            h_noise[10] = (unsigned int)(t0_26 >> 32ull) ^ h_noise[11] ^ key1_4;
                            h_noise[11] = (unsigned int)t0_26;
                            unsigned long long t0_27 = (unsigned long long)h_noise[12] * 3528531795ull;
                            unsigned long long t1_27 = (unsigned long long)h_noise[14] * 3449720151ull;
                            h_noise[12] = (unsigned int)(t1_27 >> 32ull) ^ h_noise[13] ^ key0_4;
                            h_noise[13] = (unsigned int)t1_27;
                            h_noise[14] = (unsigned int)(t0_27 >> 32ull) ^ h_noise[15] ^ key1_4;
                            h_noise[15] = (unsigned int)t0_27;
                            key0_4 = key0_4 + 2654435769u;
                            key1_4 = key1_4 + 3144134277u;
#else
                        for (int _round_1 = 5; _round_1 < philox_rounds; _round_1++) {
                            unsigned long long t0_16 = (unsigned long long)h_noise[0] * 3528531795ull;
                            unsigned long long t1_16 = (unsigned long long)h_noise[2] * 3449720151ull;
                            h_noise[0] = (unsigned int)(t1_16 >> 32ull) ^ h_noise[1] ^ key0_1;
                            h_noise[1] = (unsigned int)t1_16;
                            h_noise[2] = (unsigned int)(t0_16 >> 32ull) ^ h_noise[3] ^ key1_1;
                            h_noise[3] = (unsigned int)t0_16;
                            unsigned long long t0_17 = (unsigned long long)h_noise[4] * 3528531795ull;
                            unsigned long long t1_17 = (unsigned long long)h_noise[6] * 3449720151ull;
                            h_noise[4] = (unsigned int)(t1_17 >> 32ull) ^ h_noise[5] ^ key0_1;
                            h_noise[5] = (unsigned int)t1_17;
                            h_noise[6] = (unsigned int)(t0_17 >> 32ull) ^ h_noise[7] ^ key1_1;
                            h_noise[7] = (unsigned int)t0_17;
                            unsigned long long t0_18 = (unsigned long long)h_noise[8] * 3528531795ull;
                            unsigned long long t1_18 = (unsigned long long)h_noise[10] * 3449720151ull;
                            h_noise[8] = (unsigned int)(t1_18 >> 32ull) ^ h_noise[9] ^ key0_1;
                            h_noise[9] = (unsigned int)t1_18;
                            h_noise[10] = (unsigned int)(t0_18 >> 32ull) ^ h_noise[11] ^ key1_1;
                            h_noise[11] = (unsigned int)t0_18;
                            unsigned long long t0_19 = (unsigned long long)h_noise[12] * 3528531795ull;
                            unsigned long long t1_19 = (unsigned long long)h_noise[14] * 3449720151ull;
                            h_noise[12] = (unsigned int)(t1_19 >> 32ull) ^ h_noise[13] ^ key0_1;
                            h_noise[13] = (unsigned int)t1_19;
                            h_noise[14] = (unsigned int)(t0_19 >> 32ull) ^ h_noise[15] ^ key1_1;
                            h_noise[15] = (unsigned int)t0_19;
                            unsigned long long t0_20 = (unsigned long long)h_noise[16] * 3528531795ull;
                            unsigned long long t1_20 = (unsigned long long)h_noise[18] * 3449720151ull;
                            h_noise[16] = (unsigned int)(t1_20 >> 32ull) ^ h_noise[17] ^ key0_1;
                            h_noise[17] = (unsigned int)t1_20;
                            h_noise[18] = (unsigned int)(t0_20 >> 32ull) ^ h_noise[19] ^ key1_1;
                            h_noise[19] = (unsigned int)t0_20;
                            unsigned long long t0_21 = (unsigned long long)h_noise[20] * 3528531795ull;
                            unsigned long long t1_21 = (unsigned long long)h_noise[22] * 3449720151ull;
                            h_noise[20] = (unsigned int)(t1_21 >> 32ull) ^ h_noise[21] ^ key0_1;
                            h_noise[21] = (unsigned int)t1_21;
                            h_noise[22] = (unsigned int)(t0_21 >> 32ull) ^ h_noise[23] ^ key1_1;
                            h_noise[23] = (unsigned int)t0_21;
                            unsigned long long t0_22 = (unsigned long long)h_noise[24] * 3528531795ull;
                            unsigned long long t1_22 = (unsigned long long)h_noise[26] * 3449720151ull;
                            h_noise[24] = (unsigned int)(t1_22 >> 32ull) ^ h_noise[25] ^ key0_1;
                            h_noise[25] = (unsigned int)t1_22;
                            h_noise[26] = (unsigned int)(t0_22 >> 32ull) ^ h_noise[27] ^ key1_1;
                            h_noise[27] = (unsigned int)t0_22;
                            unsigned long long t0_23 = (unsigned long long)h_noise[28] * 3528531795ull;
                            unsigned long long t1_23 = (unsigned long long)h_noise[30] * 3449720151ull;
                            h_noise[28] = (unsigned int)(t1_23 >> 32ull) ^ h_noise[29] ^ key0_1;
                            h_noise[29] = (unsigned int)t1_23;
                            h_noise[30] = (unsigned int)(t0_23 >> 32ull) ^ h_noise[31] ^ key1_1;
                            h_noise[31] = (unsigned int)t0_23;
                            unsigned long long t0_24 = (unsigned long long)h_noise[32] * 3528531795ull;
                            unsigned long long t1_24 = (unsigned long long)h_noise[34] * 3449720151ull;
                            h_noise[32] = (unsigned int)(t1_24 >> 32ull) ^ h_noise[33] ^ key0_1;
                            h_noise[33] = (unsigned int)t1_24;
                            h_noise[34] = (unsigned int)(t0_24 >> 32ull) ^ h_noise[35] ^ key1_1;
                            h_noise[35] = (unsigned int)t0_24;
                            unsigned long long t0_25 = (unsigned long long)h_noise[36] * 3528531795ull;
                            unsigned long long t1_25 = (unsigned long long)h_noise[38] * 3449720151ull;
                            h_noise[36] = (unsigned int)(t1_25 >> 32ull) ^ h_noise[37] ^ key0_1;
                            h_noise[37] = (unsigned int)t1_25;
                            h_noise[38] = (unsigned int)(t0_25 >> 32ull) ^ h_noise[39] ^ key1_1;
                            h_noise[39] = (unsigned int)t0_25;
                            unsigned long long t0_26 = (unsigned long long)h_noise[40] * 3528531795ull;
                            unsigned long long t1_26 = (unsigned long long)h_noise[42] * 3449720151ull;
                            h_noise[40] = (unsigned int)(t1_26 >> 32ull) ^ h_noise[41] ^ key0_1;
                            h_noise[41] = (unsigned int)t1_26;
                            h_noise[42] = (unsigned int)(t0_26 >> 32ull) ^ h_noise[43] ^ key1_1;
                            h_noise[43] = (unsigned int)t0_26;
                            unsigned long long t0_27 = (unsigned long long)h_noise[44] * 3528531795ull;
                            unsigned long long t1_27 = (unsigned long long)h_noise[46] * 3449720151ull;
                            h_noise[44] = (unsigned int)(t1_27 >> 32ull) ^ h_noise[45] ^ key0_1;
                            h_noise[45] = (unsigned int)t1_27;
                            h_noise[46] = (unsigned int)(t0_27 >> 32ull) ^ h_noise[47] ^ key1_1;
                            h_noise[47] = (unsigned int)t0_27;
                            unsigned long long t0_28 = (unsigned long long)h_noise[48] * 3528531795ull;
                            unsigned long long t1_28 = (unsigned long long)h_noise[50] * 3449720151ull;
                            h_noise[48] = (unsigned int)(t1_28 >> 32ull) ^ h_noise[49] ^ key0_1;
                            h_noise[49] = (unsigned int)t1_28;
                            h_noise[50] = (unsigned int)(t0_28 >> 32ull) ^ h_noise[51] ^ key1_1;
                            h_noise[51] = (unsigned int)t0_28;
                            unsigned long long t0_29 = (unsigned long long)h_noise[52] * 3528531795ull;
                            unsigned long long t1_29 = (unsigned long long)h_noise[54] * 3449720151ull;
                            h_noise[52] = (unsigned int)(t1_29 >> 32ull) ^ h_noise[53] ^ key0_1;
                            h_noise[53] = (unsigned int)t1_29;
                            h_noise[54] = (unsigned int)(t0_29 >> 32ull) ^ h_noise[55] ^ key1_1;
                            h_noise[55] = (unsigned int)t0_29;
                            unsigned long long t0_30 = (unsigned long long)h_noise[56] * 3528531795ull;
                            unsigned long long t1_30 = (unsigned long long)h_noise[58] * 3449720151ull;
                            h_noise[56] = (unsigned int)(t1_30 >> 32ull) ^ h_noise[57] ^ key0_1;
                            h_noise[57] = (unsigned int)t1_30;
                            h_noise[58] = (unsigned int)(t0_30 >> 32ull) ^ h_noise[59] ^ key1_1;
                            h_noise[59] = (unsigned int)t0_30;
                            unsigned long long t0_31 = (unsigned long long)h_noise[60] * 3528531795ull;
                            unsigned long long t1_31 = (unsigned long long)h_noise[62] * 3449720151ull;
                            h_noise[60] = (unsigned int)(t1_31 >> 32ull) ^ h_noise[61] ^ key0_1;
                            h_noise[61] = (unsigned int)t1_31;
                            h_noise[62] = (unsigned int)(t0_31 >> 32ull) ^ h_noise[63] ^ key1_1;
                            h_noise[63] = (unsigned int)t0_31;
                            key0_1 = key0_1 + 2654435769u;
                            key1_1 = key1_1 + 3144134277u;
#endif
                        }
#if __CUDA_ARCH__ == 900
                        float _abs_1327;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1327) : "f"(S[64]));
                        unsigned int s_190 = __as_u32(mul_f32_rz_noftz(_abs_1327, __uint_as_float(125829120)));
                        unsigned int _min_1328 = ((s_190) < (260046848) ? (s_190) : (260046848));
                        unsigned int h_192 = _min_1328 + (h_noise[0] & 8191) >> 13;
                        unsigned int mag_190 = ((s_190 <= 2139095040) ? h_192 : 32256);
                        float _abs_1329;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1329) : "f"(S[65]));
                        unsigned int s_191 = __as_u32(mul_f32_rz_noftz(_abs_1329, __uint_as_float(125829120)));
                        unsigned int _min_1330 = ((s_191) < (260046848) ? (s_191) : (260046848));
                        unsigned int h_193 = _min_1330 + ((h_noise[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_191 = ((s_191 <= 2139095040) ? h_193 : 32256);
                        uint32_t mags_94;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_94) : "r"(mag_190), "r"(mag_191));
                        uint32_t signs_94;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_94) : "r"(__as_u32(S[64])), "r"(__as_u32(S[65])));
                        unsigned int word_94 = signs_94 & 2147516416u | mags_94;
                        h_words[0] = word_94;
                        float _abs_1331;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1331) : "f"(S[66]));
                        unsigned int s_192 = __as_u32(mul_f32_rz_noftz(_abs_1331, __uint_as_float(125829120)));
                        unsigned int _min_1332 = ((s_192) < (260046848) ? (s_192) : (260046848));
                        unsigned int h_194 = _min_1332 + (h_noise[1] & 8191) >> 13;
                        unsigned int mag_192 = ((s_192 <= 2139095040) ? h_194 : 32256);
                        float _abs_1333;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1333) : "f"(S[67]));
                        unsigned int s_193 = __as_u32(mul_f32_rz_noftz(_abs_1333, __uint_as_float(125829120)));
                        unsigned int _min_1334 = ((s_193) < (260046848) ? (s_193) : (260046848));
                        unsigned int h_195 = _min_1334 + ((h_noise[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_193 = ((s_193 <= 2139095040) ? h_195 : 32256);
                        uint32_t mags_95;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_95) : "r"(mag_192), "r"(mag_193));
                        uint32_t signs_95;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_95) : "r"(__as_u32(S[66])), "r"(__as_u32(S[67])));
                        unsigned int word_95 = signs_95 & 2147516416u | mags_95;
                        h_words[1] = word_95;
#else
                        uint32_t _cvt_rs_x2_f32_203;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_203) : "f"(S2[0].y), "f"(S2[0].x), "r"(h_noise[0] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_203;
                        uint32_t _cvt_rs_x2_f32_204;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_204) : "f"(S2[1].y), "f"(S2[1].x), "r"(h_noise[1] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_204;
                        reinterpret_cast<int2*>(state + (write_rows_base))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_205;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_205) : "f"(S2[8].y), "f"(S2[8].x), "r"(h_noise[2] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_205;
                        uint32_t _cvt_rs_x2_f32_206;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_206) : "f"(S2[9].y), "f"(S2[9].x), "r"(h_noise[3] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_206;
                        reinterpret_cast<int2*>(state + (write_rows_base + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_207;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_207) : "f"(S2[2].y), "f"(S2[2].x), "r"(h_noise[4] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_207;
                        uint32_t _cvt_rs_x2_f32_208;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_208) : "f"(S2[3].y), "f"(S2[3].x), "r"(h_noise[5] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_208;
                        reinterpret_cast<int2*>(state + (write_rows_base + 16ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_209;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_209) : "f"(S2[10].y), "f"(S2[10].x), "r"(h_noise[6] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_209;
                        uint32_t _cvt_rs_x2_f32_210;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_210) : "f"(S2[11].y), "f"(S2[11].x), "r"(h_noise[7] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_210;
                        reinterpret_cast<int2*>(state + (write_rows_base + 80ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_211;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_211) : "f"(S2[4].y), "f"(S2[4].x), "r"(h_noise[8] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_211;
                        uint32_t _cvt_rs_x2_f32_212;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_212) : "f"(S2[5].y), "f"(S2[5].x), "r"(h_noise[9] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_212;
                        reinterpret_cast<int2*>(state + (write_rows_base + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_213;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_213) : "f"(S2[12].y), "f"(S2[12].x), "r"(h_noise[10] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_213;
                        uint32_t _cvt_rs_x2_f32_214;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_214) : "f"(S2[13].y), "f"(S2[13].x), "r"(h_noise[11] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_214;
                        reinterpret_cast<int2*>(state + (write_rows_base + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_215;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_215) : "f"(S2[6].y), "f"(S2[6].x), "r"(h_noise[12] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_215;
                        uint32_t _cvt_rs_x2_f32_216;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_216) : "f"(S2[7].y), "f"(S2[7].x), "r"(h_noise[13] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_216;
                        reinterpret_cast<int2*>(state + (write_rows_base + 48ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_217;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_217) : "f"(S2[14].y), "f"(S2[14].x), "r"(h_noise[14] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_217;
                        uint32_t _cvt_rs_x2_f32_218;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_218) : "f"(S2[15].y), "f"(S2[15].x), "r"(h_noise[15] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_218;
                        reinterpret_cast<int2*>(state + (write_rows_base + 112ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_219;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_219) : "f"(S2[16].y), "f"(S2[16].x), "r"(h_noise[16] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_219;
                        uint32_t _cvt_rs_x2_f32_220;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_220) : "f"(S2[17].y), "f"(S2[17].x), "r"(h_noise[17] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_220;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_221;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_221) : "f"(S2[24].y), "f"(S2[24].x), "r"(h_noise[18] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_221;
                        uint32_t _cvt_rs_x2_f32_222;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_222) : "f"(S2[25].y), "f"(S2[25].x), "r"(h_noise[19] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_222;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_223;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_223) : "f"(S2[18].y), "f"(S2[18].x), "r"(h_noise[20] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_223;
                        uint32_t _cvt_rs_x2_f32_224;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_224) : "f"(S2[19].y), "f"(S2[19].x), "r"(h_noise[21] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_224;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 16ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_225;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_225) : "f"(S2[26].y), "f"(S2[26].x), "r"(h_noise[22] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_225;
                        uint32_t _cvt_rs_x2_f32_226;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_226) : "f"(S2[27].y), "f"(S2[27].x), "r"(h_noise[23] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_226;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 80ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_227;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_227) : "f"(S2[20].y), "f"(S2[20].x), "r"(h_noise[24] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_227;
                        uint32_t _cvt_rs_x2_f32_228;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_228) : "f"(S2[21].y), "f"(S2[21].x), "r"(h_noise[25] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_228;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_229;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_229) : "f"(S2[28].y), "f"(S2[28].x), "r"(h_noise[26] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_229;
                        uint32_t _cvt_rs_x2_f32_230;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_230) : "f"(S2[29].y), "f"(S2[29].x), "r"(h_noise[27] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_230;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_231;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_231) : "f"(S2[22].y), "f"(S2[22].x), "r"(h_noise[28] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_231;
                        uint32_t _cvt_rs_x2_f32_232;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_232) : "f"(S2[23].y), "f"(S2[23].x), "r"(h_noise[29] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_232;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 48ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_233;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_233) : "f"(S2[30].y), "f"(S2[30].x), "r"(h_noise[30] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_233;
                        uint32_t _cvt_rs_x2_f32_234;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_234) : "f"(S2[31].y), "f"(S2[31].x), "r"(h_noise[31] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_234;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 112ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_235;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_235) : "f"(S2[32].y), "f"(S2[32].x), "r"(h_noise[32] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_235;
                        uint32_t _cvt_rs_x2_f32_236;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_236) : "f"(S2[33].y), "f"(S2[33].x), "r"(h_noise[33] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_236;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1335;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1335) : "f"(S[80]));
                        unsigned int s_194 = __as_u32(mul_f32_rz_noftz(_abs_1335, __uint_as_float(125829120)));
                        unsigned int _min_1336 = ((s_194) < (260046848) ? (s_194) : (260046848));
                        unsigned int h_196 = _min_1336 + (h_noise[2] & 8191) >> 13;
                        unsigned int mag_194 = ((s_194 <= 2139095040) ? h_196 : 32256);
                        float _abs_1337;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1337) : "f"(S[81]));
                        unsigned int s_195 = __as_u32(mul_f32_rz_noftz(_abs_1337, __uint_as_float(125829120)));
                        unsigned int _min_1338 = ((s_195) < (260046848) ? (s_195) : (260046848));
                        unsigned int h_197 = _min_1338 + ((h_noise[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_195 = ((s_195 <= 2139095040) ? h_197 : 32256);
                        uint32_t mags_96;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_96) : "r"(mag_194), "r"(mag_195));
                        uint32_t signs_96;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_96) : "r"(__as_u32(S[80])), "r"(__as_u32(S[81])));
                        unsigned int word_96 = signs_96 & 2147516416u | mags_96;
                        h_words[0] = word_96;
                        float _abs_1339;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1339) : "f"(S[82]));
                        unsigned int s_196 = __as_u32(mul_f32_rz_noftz(_abs_1339, __uint_as_float(125829120)));
                        unsigned int _min_1340 = ((s_196) < (260046848) ? (s_196) : (260046848));
                        unsigned int h_198 = _min_1340 + (h_noise[3] & 8191) >> 13;
                        unsigned int mag_196 = ((s_196 <= 2139095040) ? h_198 : 32256);
                        float _abs_1341;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1341) : "f"(S[83]));
                        unsigned int s_197 = __as_u32(mul_f32_rz_noftz(_abs_1341, __uint_as_float(125829120)));
                        unsigned int _min_1342 = ((s_197) < (260046848) ? (s_197) : (260046848));
                        unsigned int h_199 = _min_1342 + ((h_noise[3] & 536805376) >> 16) >> 13;
                        unsigned int mag_197 = ((s_197 <= 2139095040) ? h_199 : 32256);
                        uint32_t mags_97;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_97) : "r"(mag_196), "r"(mag_197));
                        uint32_t signs_97;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_97) : "r"(__as_u32(S[82])), "r"(__as_u32(S[83])));
                        unsigned int word_97 = signs_97 & 2147516416u | mags_97;
                        h_words[1] = word_97;
#else
                        uint32_t _cvt_rs_x2_f32_237;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_237) : "f"(S2[40].y), "f"(S2[40].x), "r"(h_noise[34] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_237;
                        uint32_t _cvt_rs_x2_f32_238;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_238) : "f"(S2[41].y), "f"(S2[41].x), "r"(h_noise[35] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_238;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1343;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1343) : "f"(S[68]));
                        unsigned int s_198 = __as_u32(mul_f32_rz_noftz(_abs_1343, __uint_as_float(125829120)));
                        unsigned int _min_1344 = ((s_198) < (260046848) ? (s_198) : (260046848));
                        unsigned int h_200 = _min_1344 + (h_noise[4] & 8191) >> 13;
                        unsigned int mag_198 = ((s_198 <= 2139095040) ? h_200 : 32256);
                        float _abs_1345;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1345) : "f"(S[69]));
                        unsigned int s_199 = __as_u32(mul_f32_rz_noftz(_abs_1345, __uint_as_float(125829120)));
                        unsigned int _min_1346 = ((s_199) < (260046848) ? (s_199) : (260046848));
                        unsigned int h_201 = _min_1346 + ((h_noise[4] & 536805376) >> 16) >> 13;
                        unsigned int mag_199 = ((s_199 <= 2139095040) ? h_201 : 32256);
                        uint32_t mags_98;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_98) : "r"(mag_198), "r"(mag_199));
                        uint32_t signs_98;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_98) : "r"(__as_u32(S[68])), "r"(__as_u32(S[69])));
                        unsigned int word_98 = signs_98 & 2147516416u | mags_98;
                        h_words[0] = word_98;
                        float _abs_1347;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1347) : "f"(S[70]));
                        unsigned int s_200 = __as_u32(mul_f32_rz_noftz(_abs_1347, __uint_as_float(125829120)));
                        unsigned int _min_1348 = ((s_200) < (260046848) ? (s_200) : (260046848));
                        unsigned int h_202 = _min_1348 + (h_noise[5] & 8191) >> 13;
                        unsigned int mag_200 = ((s_200 <= 2139095040) ? h_202 : 32256);
                        float _abs_1349;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1349) : "f"(S[71]));
                        unsigned int s_201 = __as_u32(mul_f32_rz_noftz(_abs_1349, __uint_as_float(125829120)));
                        unsigned int _min_1350 = ((s_201) < (260046848) ? (s_201) : (260046848));
                        unsigned int h_203 = _min_1350 + ((h_noise[5] & 536805376) >> 16) >> 13;
                        unsigned int mag_201 = ((s_201 <= 2139095040) ? h_203 : 32256);
                        uint32_t mags_99;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_99) : "r"(mag_200), "r"(mag_201));
                        uint32_t signs_99;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_99) : "r"(__as_u32(S[70])), "r"(__as_u32(S[71])));
                        unsigned int word_99 = signs_99 & 2147516416u | mags_99;
                        h_words[1] = word_99;
#else
                        uint32_t _cvt_rs_x2_f32_239;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_239) : "f"(S2[34].y), "f"(S2[34].x), "r"(h_noise[36] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_239;
                        uint32_t _cvt_rs_x2_f32_240;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_240) : "f"(S2[35].y), "f"(S2[35].x), "r"(h_noise[37] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_240;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 16ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1351;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1351) : "f"(S[84]));
                        unsigned int s_202 = __as_u32(mul_f32_rz_noftz(_abs_1351, __uint_as_float(125829120)));
                        unsigned int _min_1352 = ((s_202) < (260046848) ? (s_202) : (260046848));
                        unsigned int h_204 = _min_1352 + (h_noise[6] & 8191) >> 13;
                        unsigned int mag_202 = ((s_202 <= 2139095040) ? h_204 : 32256);
                        float _abs_1353;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1353) : "f"(S[85]));
                        unsigned int s_203 = __as_u32(mul_f32_rz_noftz(_abs_1353, __uint_as_float(125829120)));
                        unsigned int _min_1354 = ((s_203) < (260046848) ? (s_203) : (260046848));
                        unsigned int h_205 = _min_1354 + ((h_noise[6] & 536805376) >> 16) >> 13;
                        unsigned int mag_203 = ((s_203 <= 2139095040) ? h_205 : 32256);
                        uint32_t mags_100;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_100) : "r"(mag_202), "r"(mag_203));
                        uint32_t signs_100;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_100) : "r"(__as_u32(S[84])), "r"(__as_u32(S[85])));
                        unsigned int word_100 = signs_100 & 2147516416u | mags_100;
                        h_words[0] = word_100;
                        float _abs_1355;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1355) : "f"(S[86]));
                        unsigned int s_204 = __as_u32(mul_f32_rz_noftz(_abs_1355, __uint_as_float(125829120)));
                        unsigned int _min_1356 = ((s_204) < (260046848) ? (s_204) : (260046848));
                        unsigned int h_206 = _min_1356 + (h_noise[7] & 8191) >> 13;
                        unsigned int mag_204 = ((s_204 <= 2139095040) ? h_206 : 32256);
                        float _abs_1357;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1357) : "f"(S[87]));
                        unsigned int s_205 = __as_u32(mul_f32_rz_noftz(_abs_1357, __uint_as_float(125829120)));
                        unsigned int _min_1358 = ((s_205) < (260046848) ? (s_205) : (260046848));
                        unsigned int h_207 = _min_1358 + ((h_noise[7] & 536805376) >> 16) >> 13;
                        unsigned int mag_205 = ((s_205 <= 2139095040) ? h_207 : 32256);
                        uint32_t mags_101;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_101) : "r"(mag_204), "r"(mag_205));
                        uint32_t signs_101;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_101) : "r"(__as_u32(S[86])), "r"(__as_u32(S[87])));
                        unsigned int word_101 = signs_101 & 2147516416u | mags_101;
                        h_words[1] = word_101;
#else
                        uint32_t _cvt_rs_x2_f32_241;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_241) : "f"(S2[42].y), "f"(S2[42].x), "r"(h_noise[38] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_241;
                        uint32_t _cvt_rs_x2_f32_242;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_242) : "f"(S2[43].y), "f"(S2[43].x), "r"(h_noise[39] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_242;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 80ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1359;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1359) : "f"(S[72]));
                        unsigned int s_206 = __as_u32(mul_f32_rz_noftz(_abs_1359, __uint_as_float(125829120)));
                        unsigned int _min_1360 = ((s_206) < (260046848) ? (s_206) : (260046848));
                        unsigned int h_208 = _min_1360 + (h_noise[8] & 8191) >> 13;
                        unsigned int mag_206 = ((s_206 <= 2139095040) ? h_208 : 32256);
                        float _abs_1361;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1361) : "f"(S[73]));
                        unsigned int s_207 = __as_u32(mul_f32_rz_noftz(_abs_1361, __uint_as_float(125829120)));
                        unsigned int _min_1362 = ((s_207) < (260046848) ? (s_207) : (260046848));
                        unsigned int h_209 = _min_1362 + ((h_noise[8] & 536805376) >> 16) >> 13;
                        unsigned int mag_207 = ((s_207 <= 2139095040) ? h_209 : 32256);
                        uint32_t mags_102;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_102) : "r"(mag_206), "r"(mag_207));
                        uint32_t signs_102;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_102) : "r"(__as_u32(S[72])), "r"(__as_u32(S[73])));
                        unsigned int word_102 = signs_102 & 2147516416u | mags_102;
                        h_words[0] = word_102;
                        float _abs_1363;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1363) : "f"(S[74]));
                        unsigned int s_208 = __as_u32(mul_f32_rz_noftz(_abs_1363, __uint_as_float(125829120)));
                        unsigned int _min_1364 = ((s_208) < (260046848) ? (s_208) : (260046848));
                        unsigned int h_210 = _min_1364 + (h_noise[9] & 8191) >> 13;
                        unsigned int mag_208 = ((s_208 <= 2139095040) ? h_210 : 32256);
                        float _abs_1365;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1365) : "f"(S[75]));
                        unsigned int s_209 = __as_u32(mul_f32_rz_noftz(_abs_1365, __uint_as_float(125829120)));
                        unsigned int _min_1366 = ((s_209) < (260046848) ? (s_209) : (260046848));
                        unsigned int h_211 = _min_1366 + ((h_noise[9] & 536805376) >> 16) >> 13;
                        unsigned int mag_209 = ((s_209 <= 2139095040) ? h_211 : 32256);
                        uint32_t mags_103;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_103) : "r"(mag_208), "r"(mag_209));
                        uint32_t signs_103;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_103) : "r"(__as_u32(S[74])), "r"(__as_u32(S[75])));
                        unsigned int word_103 = signs_103 & 2147516416u | mags_103;
                        h_words[1] = word_103;
#else
                        uint32_t _cvt_rs_x2_f32_243;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_243) : "f"(S2[36].y), "f"(S2[36].x), "r"(h_noise[40] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_243;
                        uint32_t _cvt_rs_x2_f32_244;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_244) : "f"(S2[37].y), "f"(S2[37].x), "r"(h_noise[41] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_244;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1367;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1367) : "f"(S[88]));
                        unsigned int s_210 = __as_u32(mul_f32_rz_noftz(_abs_1367, __uint_as_float(125829120)));
                        unsigned int _min_1368 = ((s_210) < (260046848) ? (s_210) : (260046848));
                        unsigned int h_212 = _min_1368 + (h_noise[10] & 8191) >> 13;
                        unsigned int mag_210 = ((s_210 <= 2139095040) ? h_212 : 32256);
                        float _abs_1369;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1369) : "f"(S[89]));
                        unsigned int s_211 = __as_u32(mul_f32_rz_noftz(_abs_1369, __uint_as_float(125829120)));
                        unsigned int _min_1370 = ((s_211) < (260046848) ? (s_211) : (260046848));
                        unsigned int h_213 = _min_1370 + ((h_noise[10] & 536805376) >> 16) >> 13;
                        unsigned int mag_211 = ((s_211 <= 2139095040) ? h_213 : 32256);
                        uint32_t mags_104;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_104) : "r"(mag_210), "r"(mag_211));
                        uint32_t signs_104;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_104) : "r"(__as_u32(S[88])), "r"(__as_u32(S[89])));
                        unsigned int word_104 = signs_104 & 2147516416u | mags_104;
                        h_words[0] = word_104;
                        float _abs_1371;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1371) : "f"(S[90]));
                        unsigned int s_212 = __as_u32(mul_f32_rz_noftz(_abs_1371, __uint_as_float(125829120)));
                        unsigned int _min_1372 = ((s_212) < (260046848) ? (s_212) : (260046848));
                        unsigned int h_214 = _min_1372 + (h_noise[11] & 8191) >> 13;
                        unsigned int mag_212 = ((s_212 <= 2139095040) ? h_214 : 32256);
                        float _abs_1373;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1373) : "f"(S[91]));
                        unsigned int s_213 = __as_u32(mul_f32_rz_noftz(_abs_1373, __uint_as_float(125829120)));
                        unsigned int _min_1374 = ((s_213) < (260046848) ? (s_213) : (260046848));
                        unsigned int h_215 = _min_1374 + ((h_noise[11] & 536805376) >> 16) >> 13;
                        unsigned int mag_213 = ((s_213 <= 2139095040) ? h_215 : 32256);
                        uint32_t mags_105;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_105) : "r"(mag_212), "r"(mag_213));
                        uint32_t signs_105;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_105) : "r"(__as_u32(S[90])), "r"(__as_u32(S[91])));
                        unsigned int word_105 = signs_105 & 2147516416u | mags_105;
                        h_words[1] = word_105;
#else
                        uint32_t _cvt_rs_x2_f32_245;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_245) : "f"(S2[44].y), "f"(S2[44].x), "r"(h_noise[42] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_245;
                        uint32_t _cvt_rs_x2_f32_246;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_246) : "f"(S2[45].y), "f"(S2[45].x), "r"(h_noise[43] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_246;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1375;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1375) : "f"(S[76]));
                        unsigned int s_214 = __as_u32(mul_f32_rz_noftz(_abs_1375, __uint_as_float(125829120)));
                        unsigned int _min_1376 = ((s_214) < (260046848) ? (s_214) : (260046848));
                        unsigned int h_216 = _min_1376 + (h_noise[12] & 8191) >> 13;
                        unsigned int mag_214 = ((s_214 <= 2139095040) ? h_216 : 32256);
                        float _abs_1377;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1377) : "f"(S[77]));
                        unsigned int s_215 = __as_u32(mul_f32_rz_noftz(_abs_1377, __uint_as_float(125829120)));
                        unsigned int _min_1378 = ((s_215) < (260046848) ? (s_215) : (260046848));
                        unsigned int h_217 = _min_1378 + ((h_noise[12] & 536805376) >> 16) >> 13;
                        unsigned int mag_215 = ((s_215 <= 2139095040) ? h_217 : 32256);
                        uint32_t mags_106;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_106) : "r"(mag_214), "r"(mag_215));
                        uint32_t signs_106;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_106) : "r"(__as_u32(S[76])), "r"(__as_u32(S[77])));
                        unsigned int word_106 = signs_106 & 2147516416u | mags_106;
                        h_words[0] = word_106;
                        float _abs_1379;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1379) : "f"(S[78]));
                        unsigned int s_216 = __as_u32(mul_f32_rz_noftz(_abs_1379, __uint_as_float(125829120)));
                        unsigned int _min_1380 = ((s_216) < (260046848) ? (s_216) : (260046848));
                        unsigned int h_218 = _min_1380 + (h_noise[13] & 8191) >> 13;
                        unsigned int mag_216 = ((s_216 <= 2139095040) ? h_218 : 32256);
                        float _abs_1381;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1381) : "f"(S[79]));
                        unsigned int s_217 = __as_u32(mul_f32_rz_noftz(_abs_1381, __uint_as_float(125829120)));
                        unsigned int _min_1382 = ((s_217) < (260046848) ? (s_217) : (260046848));
                        unsigned int h_219 = _min_1382 + ((h_noise[13] & 536805376) >> 16) >> 13;
                        unsigned int mag_217 = ((s_217 <= 2139095040) ? h_219 : 32256);
                        uint32_t mags_107;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_107) : "r"(mag_216), "r"(mag_217));
                        uint32_t signs_107;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_107) : "r"(__as_u32(S[78])), "r"(__as_u32(S[79])));
                        unsigned int word_107 = signs_107 & 2147516416u | mags_107;
                        h_words[1] = word_107;
#else
                        uint32_t _cvt_rs_x2_f32_247;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_247) : "f"(S2[38].y), "f"(S2[38].x), "r"(h_noise[44] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_247;
                        uint32_t _cvt_rs_x2_f32_248;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_248) : "f"(S2[39].y), "f"(S2[39].x), "r"(h_noise[45] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_248;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 48ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1383;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1383) : "f"(S[92]));
                        unsigned int s_218 = __as_u32(mul_f32_rz_noftz(_abs_1383, __uint_as_float(125829120)));
                        unsigned int _min_1384 = ((s_218) < (260046848) ? (s_218) : (260046848));
                        unsigned int h_220 = _min_1384 + (h_noise[14] & 8191) >> 13;
                        unsigned int mag_218 = ((s_218 <= 2139095040) ? h_220 : 32256);
                        float _abs_1385;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1385) : "f"(S[93]));
                        unsigned int s_219 = __as_u32(mul_f32_rz_noftz(_abs_1385, __uint_as_float(125829120)));
                        unsigned int _min_1386 = ((s_219) < (260046848) ? (s_219) : (260046848));
                        unsigned int h_221 = _min_1386 + ((h_noise[14] & 536805376) >> 16) >> 13;
                        unsigned int mag_219 = ((s_219 <= 2139095040) ? h_221 : 32256);
                        uint32_t mags_108;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_108) : "r"(mag_218), "r"(mag_219));
                        uint32_t signs_108;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_108) : "r"(__as_u32(S[92])), "r"(__as_u32(S[93])));
                        unsigned int word_108 = signs_108 & 2147516416u | mags_108;
                        h_words[0] = word_108;
                        float _abs_1387;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1387) : "f"(S[94]));
                        unsigned int s_220 = __as_u32(mul_f32_rz_noftz(_abs_1387, __uint_as_float(125829120)));
                        unsigned int _min_1388 = ((s_220) < (260046848) ? (s_220) : (260046848));
                        unsigned int h_222 = _min_1388 + (h_noise[15] & 8191) >> 13;
                        unsigned int mag_220 = ((s_220 <= 2139095040) ? h_222 : 32256);
                        float _abs_1389;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1389) : "f"(S[95]));
                        unsigned int s_221 = __as_u32(mul_f32_rz_noftz(_abs_1389, __uint_as_float(125829120)));
                        unsigned int _min_1390 = ((s_221) < (260046848) ? (s_221) : (260046848));
                        unsigned int h_223 = _min_1390 + ((h_noise[15] & 536805376) >> 16) >> 13;
                        unsigned int mag_221 = ((s_221 <= 2139095040) ? h_223 : 32256);
                        uint32_t mags_109;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_109) : "r"(mag_220), "r"(mag_221));
                        uint32_t signs_109;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_109) : "r"(__as_u32(S[94])), "r"(__as_u32(S[95])));
                        unsigned int word_109 = signs_109 & 2147516416u | mags_109;
                        h_words[1] = word_109;
#else
                        uint32_t _cvt_rs_x2_f32_249;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_249) : "f"(S2[46].y), "f"(S2[46].x), "r"(h_noise[46] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_249;
                        uint32_t _cvt_rs_x2_f32_250;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_250) : "f"(S2[47].y), "f"(S2[47].x), "r"(h_noise[47] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_250;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 112ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        unsigned int key0_5 = seed_lo;
                        unsigned int key1_5 = seed_hi;
                        long long o_28 = noise_at + 3ll * state_stride_p2;
                        unsigned long long p0_28 = (unsigned long long)(unsigned int)o_28 * 3528531795ull;
                        h_noise[0] = (unsigned int)(o_28 >> 32ll) ^ key0_5;
                        h_noise[2] = (unsigned int)(p0_28 >> 32ull) ^ key1_5;
                        h_noise[3] = (unsigned int)p0_28;
                        long long o_29 = noise_at + 3ll * state_stride_p2 + 16ll;
                        unsigned long long p0_29 = (unsigned long long)(unsigned int)o_29 * 3528531795ull;
                        h_noise[4] = (unsigned int)(o_29 >> 32ll) ^ key0_5;
                        h_noise[6] = (unsigned int)(p0_29 >> 32ull) ^ key1_5;
                        h_noise[7] = (unsigned int)p0_29;
                        long long o_30 = noise_at + 3ll * state_stride_p2 + 32ll;
                        unsigned long long p0_30 = (unsigned long long)(unsigned int)o_30 * 3528531795ull;
                        h_noise[8] = (unsigned int)(o_30 >> 32ll) ^ key0_5;
                        h_noise[10] = (unsigned int)(p0_30 >> 32ull) ^ key1_5;
                        h_noise[11] = (unsigned int)p0_30;
                        long long o_31 = noise_at + 3ll * state_stride_p2 + 48ll;
                        unsigned long long p0_31 = (unsigned long long)(unsigned int)o_31 * 3528531795ull;
                        h_noise[12] = (unsigned int)(o_31 >> 32ll) ^ key0_5;
                        h_noise[14] = (unsigned int)(p0_31 >> 32ull) ^ key1_5;
                        h_noise[15] = (unsigned int)p0_31;
                        key0_5 = key0_5 + 2654435769u;
                        key1_5 = key1_5 + 3144134277u;
                        unsigned long long q0_28 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long q1_28 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(q1_28 >> 32ull) ^ key0_5;
                        h_noise[1] = (unsigned int)q1_28;
                        h_noise[2] = (unsigned int)(q0_28 >> 32ull) ^ h_noise[3] ^ key1_5;
                        h_noise[3] = (unsigned int)q0_28;
                        unsigned long long q0_29 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long q1_29 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(q1_29 >> 32ull) ^ key0_5;
                        h_noise[5] = (unsigned int)q1_29;
                        h_noise[6] = (unsigned int)(q0_29 >> 32ull) ^ h_noise[7] ^ key1_5;
                        h_noise[7] = (unsigned int)q0_29;
                        unsigned long long q0_30 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long q1_30 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(q1_30 >> 32ull) ^ key0_5;
                        h_noise[9] = (unsigned int)q1_30;
                        h_noise[10] = (unsigned int)(q0_30 >> 32ull) ^ h_noise[11] ^ key1_5;
                        h_noise[11] = (unsigned int)q0_30;
                        unsigned long long q0_31 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long q1_31 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(q1_31 >> 32ull) ^ key0_5;
                        h_noise[13] = (unsigned int)q1_31;
                        h_noise[14] = (unsigned int)(q0_31 >> 32ull) ^ h_noise[15] ^ key1_5;
                        h_noise[15] = (unsigned int)q0_31;
                        key0_5 = key0_5 + 2654435769u;
                        key1_5 = key1_5 + 3144134277u;
                        unsigned long long s0_84 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_84 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_84 >> 32ull) ^ h_noise[1] ^ key0_5;
                        h_noise[1] = (unsigned int)s1_84;
                        h_noise[2] = (unsigned int)(s0_84 >> 32ull) ^ h_noise[3] ^ key1_5;
                        h_noise[3] = (unsigned int)s0_84;
                        unsigned long long s0_85 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_85 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_85 >> 32ull) ^ h_noise[5] ^ key0_5;
                        h_noise[5] = (unsigned int)s1_85;
                        h_noise[6] = (unsigned int)(s0_85 >> 32ull) ^ h_noise[7] ^ key1_5;
                        h_noise[7] = (unsigned int)s0_85;
                        unsigned long long s0_86 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_86 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_86 >> 32ull) ^ h_noise[9] ^ key0_5;
                        h_noise[9] = (unsigned int)s1_86;
                        h_noise[10] = (unsigned int)(s0_86 >> 32ull) ^ h_noise[11] ^ key1_5;
                        h_noise[11] = (unsigned int)s0_86;
                        unsigned long long s0_87 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_87 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_87 >> 32ull) ^ h_noise[13] ^ key0_5;
                        h_noise[13] = (unsigned int)s1_87;
                        h_noise[14] = (unsigned int)(s0_87 >> 32ull) ^ h_noise[15] ^ key1_5;
                        h_noise[15] = (unsigned int)s0_87;
                        key0_5 = key0_5 + 2654435769u;
                        key1_5 = key1_5 + 3144134277u;
                        unsigned long long s0_88 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_88 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_88 >> 32ull) ^ h_noise[1] ^ key0_5;
                        h_noise[1] = (unsigned int)s1_88;
                        h_noise[2] = (unsigned int)(s0_88 >> 32ull) ^ h_noise[3] ^ key1_5;
                        h_noise[3] = (unsigned int)s0_88;
                        unsigned long long s0_89 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_89 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_89 >> 32ull) ^ h_noise[5] ^ key0_5;
                        h_noise[5] = (unsigned int)s1_89;
                        h_noise[6] = (unsigned int)(s0_89 >> 32ull) ^ h_noise[7] ^ key1_5;
                        h_noise[7] = (unsigned int)s0_89;
                        unsigned long long s0_90 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_90 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_90 >> 32ull) ^ h_noise[9] ^ key0_5;
                        h_noise[9] = (unsigned int)s1_90;
                        h_noise[10] = (unsigned int)(s0_90 >> 32ull) ^ h_noise[11] ^ key1_5;
                        h_noise[11] = (unsigned int)s0_90;
                        unsigned long long s0_91 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_91 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_91 >> 32ull) ^ h_noise[13] ^ key0_5;
                        h_noise[13] = (unsigned int)s1_91;
                        h_noise[14] = (unsigned int)(s0_91 >> 32ull) ^ h_noise[15] ^ key1_5;
                        h_noise[15] = (unsigned int)s0_91;
                        key0_5 = key0_5 + 2654435769u;
                        key1_5 = key1_5 + 3144134277u;
                        unsigned long long s0_92 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_92 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_92 >> 32ull) ^ h_noise[1] ^ key0_5;
                        h_noise[1] = (unsigned int)s1_92;
                        h_noise[2] = (unsigned int)(s0_92 >> 32ull) ^ h_noise[3] ^ key1_5;
                        h_noise[3] = (unsigned int)s0_92;
                        unsigned long long s0_93 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_93 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_93 >> 32ull) ^ h_noise[5] ^ key0_5;
                        h_noise[5] = (unsigned int)s1_93;
                        h_noise[6] = (unsigned int)(s0_93 >> 32ull) ^ h_noise[7] ^ key1_5;
                        h_noise[7] = (unsigned int)s0_93;
                        unsigned long long s0_94 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_94 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_94 >> 32ull) ^ h_noise[9] ^ key0_5;
                        h_noise[9] = (unsigned int)s1_94;
                        h_noise[10] = (unsigned int)(s0_94 >> 32ull) ^ h_noise[11] ^ key1_5;
                        h_noise[11] = (unsigned int)s0_94;
                        unsigned long long s0_95 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_95 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_95 >> 32ull) ^ h_noise[13] ^ key0_5;
                        h_noise[13] = (unsigned int)s1_95;
                        h_noise[14] = (unsigned int)(s0_95 >> 32ull) ^ h_noise[15] ^ key1_5;
                        h_noise[15] = (unsigned int)s0_95;
                        key0_5 = key0_5 + 2654435769u;
                        key1_5 = key1_5 + 3144134277u;
                        #pragma unroll 1
                        for (int _round_7 = 5; _round_7 < philox_rounds; _round_7++) {
                            unsigned long long t0_28 = (unsigned long long)h_noise[0] * 3528531795ull;
                            unsigned long long t1_28 = (unsigned long long)h_noise[2] * 3449720151ull;
                            h_noise[0] = (unsigned int)(t1_28 >> 32ull) ^ h_noise[1] ^ key0_5;
                            h_noise[1] = (unsigned int)t1_28;
                            h_noise[2] = (unsigned int)(t0_28 >> 32ull) ^ h_noise[3] ^ key1_5;
                            h_noise[3] = (unsigned int)t0_28;
                            unsigned long long t0_29 = (unsigned long long)h_noise[4] * 3528531795ull;
                            unsigned long long t1_29 = (unsigned long long)h_noise[6] * 3449720151ull;
                            h_noise[4] = (unsigned int)(t1_29 >> 32ull) ^ h_noise[5] ^ key0_5;
                            h_noise[5] = (unsigned int)t1_29;
                            h_noise[6] = (unsigned int)(t0_29 >> 32ull) ^ h_noise[7] ^ key1_5;
                            h_noise[7] = (unsigned int)t0_29;
                            unsigned long long t0_30 = (unsigned long long)h_noise[8] * 3528531795ull;
                            unsigned long long t1_30 = (unsigned long long)h_noise[10] * 3449720151ull;
                            h_noise[8] = (unsigned int)(t1_30 >> 32ull) ^ h_noise[9] ^ key0_5;
                            h_noise[9] = (unsigned int)t1_30;
                            h_noise[10] = (unsigned int)(t0_30 >> 32ull) ^ h_noise[11] ^ key1_5;
                            h_noise[11] = (unsigned int)t0_30;
                            unsigned long long t0_31 = (unsigned long long)h_noise[12] * 3528531795ull;
                            unsigned long long t1_31 = (unsigned long long)h_noise[14] * 3449720151ull;
                            h_noise[12] = (unsigned int)(t1_31 >> 32ull) ^ h_noise[13] ^ key0_5;
                            h_noise[13] = (unsigned int)t1_31;
                            h_noise[14] = (unsigned int)(t0_31 >> 32ull) ^ h_noise[15] ^ key1_5;
                            h_noise[15] = (unsigned int)t0_31;
                            key0_5 = key0_5 + 2654435769u;
                            key1_5 = key1_5 + 3144134277u;
                        }
                        float _abs_1391;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1391) : "f"(S[96]));
                        unsigned int s_222 = __as_u32(mul_f32_rz_noftz(_abs_1391, __uint_as_float(125829120)));
                        unsigned int _min_1392 = ((s_222) < (260046848) ? (s_222) : (260046848));
                        unsigned int h_224 = _min_1392 + (h_noise[0] & 8191) >> 13;
                        unsigned int mag_222 = ((s_222 <= 2139095040) ? h_224 : 32256);
                        float _abs_1393;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1393) : "f"(S[97]));
                        unsigned int s_223 = __as_u32(mul_f32_rz_noftz(_abs_1393, __uint_as_float(125829120)));
                        unsigned int _min_1394 = ((s_223) < (260046848) ? (s_223) : (260046848));
                        unsigned int h_225 = _min_1394 + ((h_noise[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_223 = ((s_223 <= 2139095040) ? h_225 : 32256);
                        uint32_t mags_110;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_110) : "r"(mag_222), "r"(mag_223));
                        uint32_t signs_110;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_110) : "r"(__as_u32(S[96])), "r"(__as_u32(S[97])));
                        unsigned int word_110 = signs_110 & 2147516416u | mags_110;
                        h_words[0] = word_110;
                        float _abs_1395;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1395) : "f"(S[98]));
                        unsigned int s_224 = __as_u32(mul_f32_rz_noftz(_abs_1395, __uint_as_float(125829120)));
                        unsigned int _min_1396 = ((s_224) < (260046848) ? (s_224) : (260046848));
                        unsigned int h_226 = _min_1396 + (h_noise[1] & 8191) >> 13;
                        unsigned int mag_224 = ((s_224 <= 2139095040) ? h_226 : 32256);
                        float _abs_1397;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1397) : "f"(S[99]));
                        unsigned int s_225 = __as_u32(mul_f32_rz_noftz(_abs_1397, __uint_as_float(125829120)));
                        unsigned int _min_1398 = ((s_225) < (260046848) ? (s_225) : (260046848));
                        unsigned int h_227 = _min_1398 + ((h_noise[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_225 = ((s_225 <= 2139095040) ? h_227 : 32256);
                        uint32_t mags_111;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_111) : "r"(mag_224), "r"(mag_225));
                        uint32_t signs_111;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_111) : "r"(__as_u32(S[98])), "r"(__as_u32(S[99])));
                        unsigned int word_111 = signs_111 & 2147516416u | mags_111;
                        h_words[1] = word_111;
#else
                        uint32_t _cvt_rs_x2_f32_251;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_251) : "f"(S2[48].y), "f"(S2[48].x), "r"(h_noise[48] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_251;
                        uint32_t _cvt_rs_x2_f32_252;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_252) : "f"(S2[49].y), "f"(S2[49].x), "r"(h_noise[49] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_252;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1399;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1399) : "f"(S[112]));
                        unsigned int s_226 = __as_u32(mul_f32_rz_noftz(_abs_1399, __uint_as_float(125829120)));
                        unsigned int _min_1400 = ((s_226) < (260046848) ? (s_226) : (260046848));
                        unsigned int h_228 = _min_1400 + (h_noise[2] & 8191) >> 13;
                        unsigned int mag_226 = ((s_226 <= 2139095040) ? h_228 : 32256);
                        float _abs_1401;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1401) : "f"(S[113]));
                        unsigned int s_227 = __as_u32(mul_f32_rz_noftz(_abs_1401, __uint_as_float(125829120)));
                        unsigned int _min_1402 = ((s_227) < (260046848) ? (s_227) : (260046848));
                        unsigned int h_229 = _min_1402 + ((h_noise[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_227 = ((s_227 <= 2139095040) ? h_229 : 32256);
                        uint32_t mags_112;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_112) : "r"(mag_226), "r"(mag_227));
                        uint32_t signs_112;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_112) : "r"(__as_u32(S[112])), "r"(__as_u32(S[113])));
                        unsigned int word_112 = signs_112 & 2147516416u | mags_112;
                        h_words[0] = word_112;
                        float _abs_1403;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1403) : "f"(S[114]));
                        unsigned int s_228 = __as_u32(mul_f32_rz_noftz(_abs_1403, __uint_as_float(125829120)));
                        unsigned int _min_1404 = ((s_228) < (260046848) ? (s_228) : (260046848));
                        unsigned int h_230 = _min_1404 + (h_noise[3] & 8191) >> 13;
                        unsigned int mag_228 = ((s_228 <= 2139095040) ? h_230 : 32256);
                        float _abs_1405;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1405) : "f"(S[115]));
                        unsigned int s_229 = __as_u32(mul_f32_rz_noftz(_abs_1405, __uint_as_float(125829120)));
                        unsigned int _min_1406 = ((s_229) < (260046848) ? (s_229) : (260046848));
                        unsigned int h_231 = _min_1406 + ((h_noise[3] & 536805376) >> 16) >> 13;
                        unsigned int mag_229 = ((s_229 <= 2139095040) ? h_231 : 32256);
                        uint32_t mags_113;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_113) : "r"(mag_228), "r"(mag_229));
                        uint32_t signs_113;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_113) : "r"(__as_u32(S[114])), "r"(__as_u32(S[115])));
                        unsigned int word_113 = signs_113 & 2147516416u | mags_113;
                        h_words[1] = word_113;
#else
                        uint32_t _cvt_rs_x2_f32_253;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_253) : "f"(S2[56].y), "f"(S2[56].x), "r"(h_noise[50] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_253;
                        uint32_t _cvt_rs_x2_f32_254;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_254) : "f"(S2[57].y), "f"(S2[57].x), "r"(h_noise[51] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_254;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1407;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1407) : "f"(S[100]));
                        unsigned int s_230 = __as_u32(mul_f32_rz_noftz(_abs_1407, __uint_as_float(125829120)));
                        unsigned int _min_1408 = ((s_230) < (260046848) ? (s_230) : (260046848));
                        unsigned int h_232 = _min_1408 + (h_noise[4] & 8191) >> 13;
                        unsigned int mag_230 = ((s_230 <= 2139095040) ? h_232 : 32256);
                        float _abs_1409;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1409) : "f"(S[101]));
                        unsigned int s_231 = __as_u32(mul_f32_rz_noftz(_abs_1409, __uint_as_float(125829120)));
                        unsigned int _min_1410 = ((s_231) < (260046848) ? (s_231) : (260046848));
                        unsigned int h_233 = _min_1410 + ((h_noise[4] & 536805376) >> 16) >> 13;
                        unsigned int mag_231 = ((s_231 <= 2139095040) ? h_233 : 32256);
                        uint32_t mags_114;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_114) : "r"(mag_230), "r"(mag_231));
                        uint32_t signs_114;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_114) : "r"(__as_u32(S[100])), "r"(__as_u32(S[101])));
                        unsigned int word_114 = signs_114 & 2147516416u | mags_114;
                        h_words[0] = word_114;
                        float _abs_1411;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1411) : "f"(S[102]));
                        unsigned int s_232 = __as_u32(mul_f32_rz_noftz(_abs_1411, __uint_as_float(125829120)));
                        unsigned int _min_1412 = ((s_232) < (260046848) ? (s_232) : (260046848));
                        unsigned int h_234 = _min_1412 + (h_noise[5] & 8191) >> 13;
                        unsigned int mag_232 = ((s_232 <= 2139095040) ? h_234 : 32256);
                        float _abs_1413;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1413) : "f"(S[103]));
                        unsigned int s_233 = __as_u32(mul_f32_rz_noftz(_abs_1413, __uint_as_float(125829120)));
                        unsigned int _min_1414 = ((s_233) < (260046848) ? (s_233) : (260046848));
                        unsigned int h_235 = _min_1414 + ((h_noise[5] & 536805376) >> 16) >> 13;
                        unsigned int mag_233 = ((s_233 <= 2139095040) ? h_235 : 32256);
                        uint32_t mags_115;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_115) : "r"(mag_232), "r"(mag_233));
                        uint32_t signs_115;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_115) : "r"(__as_u32(S[102])), "r"(__as_u32(S[103])));
                        unsigned int word_115 = signs_115 & 2147516416u | mags_115;
                        h_words[1] = word_115;
#else
                        uint32_t _cvt_rs_x2_f32_255;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_255) : "f"(S2[50].y), "f"(S2[50].x), "r"(h_noise[52] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_255;
                        uint32_t _cvt_rs_x2_f32_256;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_256) : "f"(S2[51].y), "f"(S2[51].x), "r"(h_noise[53] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_256;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 16ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1415;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1415) : "f"(S[116]));
                        unsigned int s_234 = __as_u32(mul_f32_rz_noftz(_abs_1415, __uint_as_float(125829120)));
                        unsigned int _min_1416 = ((s_234) < (260046848) ? (s_234) : (260046848));
                        unsigned int h_236 = _min_1416 + (h_noise[6] & 8191) >> 13;
                        unsigned int mag_234 = ((s_234 <= 2139095040) ? h_236 : 32256);
                        float _abs_1417;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1417) : "f"(S[117]));
                        unsigned int s_235 = __as_u32(mul_f32_rz_noftz(_abs_1417, __uint_as_float(125829120)));
                        unsigned int _min_1418 = ((s_235) < (260046848) ? (s_235) : (260046848));
                        unsigned int h_237 = _min_1418 + ((h_noise[6] & 536805376) >> 16) >> 13;
                        unsigned int mag_235 = ((s_235 <= 2139095040) ? h_237 : 32256);
                        uint32_t mags_116;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_116) : "r"(mag_234), "r"(mag_235));
                        uint32_t signs_116;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_116) : "r"(__as_u32(S[116])), "r"(__as_u32(S[117])));
                        unsigned int word_116 = signs_116 & 2147516416u | mags_116;
                        h_words[0] = word_116;
                        float _abs_1419;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1419) : "f"(S[118]));
                        unsigned int s_236 = __as_u32(mul_f32_rz_noftz(_abs_1419, __uint_as_float(125829120)));
                        unsigned int _min_1420 = ((s_236) < (260046848) ? (s_236) : (260046848));
                        unsigned int h_238 = _min_1420 + (h_noise[7] & 8191) >> 13;
                        unsigned int mag_236 = ((s_236 <= 2139095040) ? h_238 : 32256);
                        float _abs_1421;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1421) : "f"(S[119]));
                        unsigned int s_237 = __as_u32(mul_f32_rz_noftz(_abs_1421, __uint_as_float(125829120)));
                        unsigned int _min_1422 = ((s_237) < (260046848) ? (s_237) : (260046848));
                        unsigned int h_239 = _min_1422 + ((h_noise[7] & 536805376) >> 16) >> 13;
                        unsigned int mag_237 = ((s_237 <= 2139095040) ? h_239 : 32256);
                        uint32_t mags_117;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_117) : "r"(mag_236), "r"(mag_237));
                        uint32_t signs_117;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_117) : "r"(__as_u32(S[118])), "r"(__as_u32(S[119])));
                        unsigned int word_117 = signs_117 & 2147516416u | mags_117;
                        h_words[1] = word_117;
#else
                        uint32_t _cvt_rs_x2_f32_257;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_257) : "f"(S2[58].y), "f"(S2[58].x), "r"(h_noise[54] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_257;
                        uint32_t _cvt_rs_x2_f32_258;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_258) : "f"(S2[59].y), "f"(S2[59].x), "r"(h_noise[55] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_258;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 80ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1423;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1423) : "f"(S[104]));
                        unsigned int s_238 = __as_u32(mul_f32_rz_noftz(_abs_1423, __uint_as_float(125829120)));
                        unsigned int _min_1424 = ((s_238) < (260046848) ? (s_238) : (260046848));
                        unsigned int h_240 = _min_1424 + (h_noise[8] & 8191) >> 13;
                        unsigned int mag_238 = ((s_238 <= 2139095040) ? h_240 : 32256);
                        float _abs_1425;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1425) : "f"(S[105]));
                        unsigned int s_239 = __as_u32(mul_f32_rz_noftz(_abs_1425, __uint_as_float(125829120)));
                        unsigned int _min_1426 = ((s_239) < (260046848) ? (s_239) : (260046848));
                        unsigned int h_241 = _min_1426 + ((h_noise[8] & 536805376) >> 16) >> 13;
                        unsigned int mag_239 = ((s_239 <= 2139095040) ? h_241 : 32256);
                        uint32_t mags_118;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_118) : "r"(mag_238), "r"(mag_239));
                        uint32_t signs_118;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_118) : "r"(__as_u32(S[104])), "r"(__as_u32(S[105])));
                        unsigned int word_118 = signs_118 & 2147516416u | mags_118;
                        h_words[0] = word_118;
                        float _abs_1427;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1427) : "f"(S[106]));
                        unsigned int s_240 = __as_u32(mul_f32_rz_noftz(_abs_1427, __uint_as_float(125829120)));
                        unsigned int _min_1428 = ((s_240) < (260046848) ? (s_240) : (260046848));
                        unsigned int h_242 = _min_1428 + (h_noise[9] & 8191) >> 13;
                        unsigned int mag_240 = ((s_240 <= 2139095040) ? h_242 : 32256);
                        float _abs_1429;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1429) : "f"(S[107]));
                        unsigned int s_241 = __as_u32(mul_f32_rz_noftz(_abs_1429, __uint_as_float(125829120)));
                        unsigned int _min_1430 = ((s_241) < (260046848) ? (s_241) : (260046848));
                        unsigned int h_243 = _min_1430 + ((h_noise[9] & 536805376) >> 16) >> 13;
                        unsigned int mag_241 = ((s_241 <= 2139095040) ? h_243 : 32256);
                        uint32_t mags_119;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_119) : "r"(mag_240), "r"(mag_241));
                        uint32_t signs_119;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_119) : "r"(__as_u32(S[106])), "r"(__as_u32(S[107])));
                        unsigned int word_119 = signs_119 & 2147516416u | mags_119;
                        h_words[1] = word_119;
#else
                        uint32_t _cvt_rs_x2_f32_259;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_259) : "f"(S2[52].y), "f"(S2[52].x), "r"(h_noise[56] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_259;
                        uint32_t _cvt_rs_x2_f32_260;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_260) : "f"(S2[53].y), "f"(S2[53].x), "r"(h_noise[57] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_260;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1431;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1431) : "f"(S[120]));
                        unsigned int s_242 = __as_u32(mul_f32_rz_noftz(_abs_1431, __uint_as_float(125829120)));
                        unsigned int _min_1432 = ((s_242) < (260046848) ? (s_242) : (260046848));
                        unsigned int h_244 = _min_1432 + (h_noise[10] & 8191) >> 13;
                        unsigned int mag_242 = ((s_242 <= 2139095040) ? h_244 : 32256);
                        float _abs_1433;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1433) : "f"(S[121]));
                        unsigned int s_243 = __as_u32(mul_f32_rz_noftz(_abs_1433, __uint_as_float(125829120)));
                        unsigned int _min_1434 = ((s_243) < (260046848) ? (s_243) : (260046848));
                        unsigned int h_245 = _min_1434 + ((h_noise[10] & 536805376) >> 16) >> 13;
                        unsigned int mag_243 = ((s_243 <= 2139095040) ? h_245 : 32256);
                        uint32_t mags_120;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_120) : "r"(mag_242), "r"(mag_243));
                        uint32_t signs_120;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_120) : "r"(__as_u32(S[120])), "r"(__as_u32(S[121])));
                        unsigned int word_120 = signs_120 & 2147516416u | mags_120;
                        h_words[0] = word_120;
                        float _abs_1435;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1435) : "f"(S[122]));
                        unsigned int s_244 = __as_u32(mul_f32_rz_noftz(_abs_1435, __uint_as_float(125829120)));
                        unsigned int _min_1436 = ((s_244) < (260046848) ? (s_244) : (260046848));
                        unsigned int h_246 = _min_1436 + (h_noise[11] & 8191) >> 13;
                        unsigned int mag_244 = ((s_244 <= 2139095040) ? h_246 : 32256);
                        float _abs_1437;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1437) : "f"(S[123]));
                        unsigned int s_245 = __as_u32(mul_f32_rz_noftz(_abs_1437, __uint_as_float(125829120)));
                        unsigned int _min_1438 = ((s_245) < (260046848) ? (s_245) : (260046848));
                        unsigned int h_247 = _min_1438 + ((h_noise[11] & 536805376) >> 16) >> 13;
                        unsigned int mag_245 = ((s_245 <= 2139095040) ? h_247 : 32256);
                        uint32_t mags_121;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_121) : "r"(mag_244), "r"(mag_245));
                        uint32_t signs_121;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_121) : "r"(__as_u32(S[122])), "r"(__as_u32(S[123])));
                        unsigned int word_121 = signs_121 & 2147516416u | mags_121;
                        h_words[1] = word_121;
#else
                        uint32_t _cvt_rs_x2_f32_261;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_261) : "f"(S2[60].y), "f"(S2[60].x), "r"(h_noise[58] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_261;
                        uint32_t _cvt_rs_x2_f32_262;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_262) : "f"(S2[61].y), "f"(S2[61].x), "r"(h_noise[59] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_262;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1439;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1439) : "f"(S[108]));
                        unsigned int s_246 = __as_u32(mul_f32_rz_noftz(_abs_1439, __uint_as_float(125829120)));
                        unsigned int _min_1440 = ((s_246) < (260046848) ? (s_246) : (260046848));
                        unsigned int h_248 = _min_1440 + (h_noise[12] & 8191) >> 13;
                        unsigned int mag_246 = ((s_246 <= 2139095040) ? h_248 : 32256);
                        float _abs_1441;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1441) : "f"(S[109]));
                        unsigned int s_247 = __as_u32(mul_f32_rz_noftz(_abs_1441, __uint_as_float(125829120)));
                        unsigned int _min_1442 = ((s_247) < (260046848) ? (s_247) : (260046848));
                        unsigned int h_249 = _min_1442 + ((h_noise[12] & 536805376) >> 16) >> 13;
                        unsigned int mag_247 = ((s_247 <= 2139095040) ? h_249 : 32256);
                        uint32_t mags_122;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_122) : "r"(mag_246), "r"(mag_247));
                        uint32_t signs_122;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_122) : "r"(__as_u32(S[108])), "r"(__as_u32(S[109])));
                        unsigned int word_122 = signs_122 & 2147516416u | mags_122;
                        h_words[0] = word_122;
                        float _abs_1443;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1443) : "f"(S[110]));
                        unsigned int s_248 = __as_u32(mul_f32_rz_noftz(_abs_1443, __uint_as_float(125829120)));
                        unsigned int _min_1444 = ((s_248) < (260046848) ? (s_248) : (260046848));
                        unsigned int h_250 = _min_1444 + (h_noise[13] & 8191) >> 13;
                        unsigned int mag_248 = ((s_248 <= 2139095040) ? h_250 : 32256);
                        float _abs_1445;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1445) : "f"(S[111]));
                        unsigned int s_249 = __as_u32(mul_f32_rz_noftz(_abs_1445, __uint_as_float(125829120)));
                        unsigned int _min_1446 = ((s_249) < (260046848) ? (s_249) : (260046848));
                        unsigned int h_251 = _min_1446 + ((h_noise[13] & 536805376) >> 16) >> 13;
                        unsigned int mag_249 = ((s_249 <= 2139095040) ? h_251 : 32256);
                        uint32_t mags_123;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_123) : "r"(mag_248), "r"(mag_249));
                        uint32_t signs_123;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_123) : "r"(__as_u32(S[110])), "r"(__as_u32(S[111])));
                        unsigned int word_123 = signs_123 & 2147516416u | mags_123;
                        h_words[1] = word_123;
#else
                        uint32_t _cvt_rs_x2_f32_263;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_263) : "f"(S2[54].y), "f"(S2[54].x), "r"(h_noise[60] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_263;
                        uint32_t _cvt_rs_x2_f32_264;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_264) : "f"(S2[55].y), "f"(S2[55].x), "r"(h_noise[61] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_264;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 48ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        float _abs_1447;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1447) : "f"(S[124]));
                        unsigned int s_250 = __as_u32(mul_f32_rz_noftz(_abs_1447, __uint_as_float(125829120)));
                        unsigned int _min_1448 = ((s_250) < (260046848) ? (s_250) : (260046848));
                        unsigned int h_252 = _min_1448 + (h_noise[14] & 8191) >> 13;
                        unsigned int mag_250 = ((s_250 <= 2139095040) ? h_252 : 32256);
                        float _abs_1449;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1449) : "f"(S[125]));
                        unsigned int s_251 = __as_u32(mul_f32_rz_noftz(_abs_1449, __uint_as_float(125829120)));
                        unsigned int _min_1450 = ((s_251) < (260046848) ? (s_251) : (260046848));
                        unsigned int h_253 = _min_1450 + ((h_noise[14] & 536805376) >> 16) >> 13;
                        unsigned int mag_251 = ((s_251 <= 2139095040) ? h_253 : 32256);
                        uint32_t mags_124;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_124) : "r"(mag_250), "r"(mag_251));
                        uint32_t signs_124;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_124) : "r"(__as_u32(S[124])), "r"(__as_u32(S[125])));
                        unsigned int word_124 = signs_124 & 2147516416u | mags_124;
                        h_words[0] = word_124;
                        float _abs_1451;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1451) : "f"(S[126]));
                        unsigned int s_252 = __as_u32(mul_f32_rz_noftz(_abs_1451, __uint_as_float(125829120)));
                        unsigned int _min_1452 = ((s_252) < (260046848) ? (s_252) : (260046848));
                        unsigned int h_254 = _min_1452 + (h_noise[15] & 8191) >> 13;
                        unsigned int mag_252 = ((s_252 <= 2139095040) ? h_254 : 32256);
                        float _abs_1453;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_1453) : "f"(S[127]));
                        unsigned int s_253 = __as_u32(mul_f32_rz_noftz(_abs_1453, __uint_as_float(125829120)));
                        unsigned int _min_1454 = ((s_253) < (260046848) ? (s_253) : (260046848));
                        unsigned int h_255 = _min_1454 + ((h_noise[15] & 536805376) >> 16) >> 13;
                        unsigned int mag_253 = ((s_253 <= 2139095040) ? h_255 : 32256);
                        uint32_t mags_125;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_125) : "r"(mag_252), "r"(mag_253));
                        uint32_t signs_125;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_125) : "r"(__as_u32(S[126])), "r"(__as_u32(S[127])));
                        unsigned int word_125 = signs_125 & 2147516416u | mags_125;
                        h_words[1] = word_125;
#else
                        uint32_t _cvt_rs_x2_f32_265;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_265) : "f"(S2[62].y), "f"(S2[62].x), "r"(h_noise[62] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_265;
                        uint32_t _cvt_rs_x2_f32_266;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_266) : "f"(S2[63].y), "f"(S2[63].x), "r"(h_noise[63] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_266;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 112ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                    }
                    if (philox_rounds == 0) {
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1455;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1455) : "f"(S[1]), "f"(S[0]));
                        h_words[0] = _cvt_rn_f16x2_f32_1455;
                        uint32_t _cvt_rn_f16x2_f32_1456;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1456) : "f"(S[3]), "f"(S[2]));
                        h_words[1] = _cvt_rn_f16x2_f32_1456;
#else
                        uint32_t _cvt_rn_f16x2_f32_267;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_267) : "f"(S2[0].y), "f"(S2[0].x));
                        h_words[0] = _cvt_rn_f16x2_f32_267;
                        uint32_t _cvt_rn_f16x2_f32_268;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_268) : "f"(S2[1].y), "f"(S2[1].x));
                        h_words[1] = _cvt_rn_f16x2_f32_268;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1457;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1457) : "f"(S[5]), "f"(S[4]));
                        h_words[0] = _cvt_rn_f16x2_f32_1457;
                        uint32_t _cvt_rn_f16x2_f32_1458;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1458) : "f"(S[7]), "f"(S[6]));
                        h_words[1] = _cvt_rn_f16x2_f32_1458;
#else
                        uint32_t _cvt_rn_f16x2_f32_269;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_269) : "f"(S2[2].y), "f"(S2[2].x));
                        h_words[0] = _cvt_rn_f16x2_f32_269;
                        uint32_t _cvt_rn_f16x2_f32_270;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_270) : "f"(S2[3].y), "f"(S2[3].x));
                        h_words[1] = _cvt_rn_f16x2_f32_270;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 16ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1459;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1459) : "f"(S[9]), "f"(S[8]));
                        h_words[0] = _cvt_rn_f16x2_f32_1459;
                        uint32_t _cvt_rn_f16x2_f32_1460;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1460) : "f"(S[11]), "f"(S[10]));
                        h_words[1] = _cvt_rn_f16x2_f32_1460;
#else
                        uint32_t _cvt_rn_f16x2_f32_271;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_271) : "f"(S2[4].y), "f"(S2[4].x));
                        h_words[0] = _cvt_rn_f16x2_f32_271;
                        uint32_t _cvt_rn_f16x2_f32_272;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_272) : "f"(S2[5].y), "f"(S2[5].x));
                        h_words[1] = _cvt_rn_f16x2_f32_272;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1461;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1461) : "f"(S[13]), "f"(S[12]));
                        h_words[0] = _cvt_rn_f16x2_f32_1461;
                        uint32_t _cvt_rn_f16x2_f32_1462;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1462) : "f"(S[15]), "f"(S[14]));
                        h_words[1] = _cvt_rn_f16x2_f32_1462;
#else
                        uint32_t _cvt_rn_f16x2_f32_273;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_273) : "f"(S2[6].y), "f"(S2[6].x));
                        h_words[0] = _cvt_rn_f16x2_f32_273;
                        uint32_t _cvt_rn_f16x2_f32_274;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_274) : "f"(S2[7].y), "f"(S2[7].x));
                        h_words[1] = _cvt_rn_f16x2_f32_274;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 48ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1463;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1463) : "f"(S[17]), "f"(S[16]));
                        h_words[0] = _cvt_rn_f16x2_f32_1463;
                        uint32_t _cvt_rn_f16x2_f32_1464;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1464) : "f"(S[19]), "f"(S[18]));
                        h_words[1] = _cvt_rn_f16x2_f32_1464;
#else
                        uint32_t _cvt_rn_f16x2_f32_275;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_275) : "f"(S2[8].y), "f"(S2[8].x));
                        h_words[0] = _cvt_rn_f16x2_f32_275;
                        uint32_t _cvt_rn_f16x2_f32_276;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_276) : "f"(S2[9].y), "f"(S2[9].x));
                        h_words[1] = _cvt_rn_f16x2_f32_276;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1465;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1465) : "f"(S[21]), "f"(S[20]));
                        h_words[0] = _cvt_rn_f16x2_f32_1465;
                        uint32_t _cvt_rn_f16x2_f32_1466;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1466) : "f"(S[23]), "f"(S[22]));
                        h_words[1] = _cvt_rn_f16x2_f32_1466;
#else
                        uint32_t _cvt_rn_f16x2_f32_277;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_277) : "f"(S2[10].y), "f"(S2[10].x));
                        h_words[0] = _cvt_rn_f16x2_f32_277;
                        uint32_t _cvt_rn_f16x2_f32_278;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_278) : "f"(S2[11].y), "f"(S2[11].x));
                        h_words[1] = _cvt_rn_f16x2_f32_278;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 80ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1467;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1467) : "f"(S[25]), "f"(S[24]));
                        h_words[0] = _cvt_rn_f16x2_f32_1467;
                        uint32_t _cvt_rn_f16x2_f32_1468;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1468) : "f"(S[27]), "f"(S[26]));
                        h_words[1] = _cvt_rn_f16x2_f32_1468;
#else
                        uint32_t _cvt_rn_f16x2_f32_279;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_279) : "f"(S2[12].y), "f"(S2[12].x));
                        h_words[0] = _cvt_rn_f16x2_f32_279;
                        uint32_t _cvt_rn_f16x2_f32_280;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_280) : "f"(S2[13].y), "f"(S2[13].x));
                        h_words[1] = _cvt_rn_f16x2_f32_280;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1469;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1469) : "f"(S[29]), "f"(S[28]));
                        h_words[0] = _cvt_rn_f16x2_f32_1469;
                        uint32_t _cvt_rn_f16x2_f32_1470;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1470) : "f"(S[31]), "f"(S[30]));
                        h_words[1] = _cvt_rn_f16x2_f32_1470;
#else
                        uint32_t _cvt_rn_f16x2_f32_281;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_281) : "f"(S2[14].y), "f"(S2[14].x));
                        h_words[0] = _cvt_rn_f16x2_f32_281;
                        uint32_t _cvt_rn_f16x2_f32_282;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_282) : "f"(S2[15].y), "f"(S2[15].x));
                        h_words[1] = _cvt_rn_f16x2_f32_282;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 112ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1471;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1471) : "f"(S[33]), "f"(S[32]));
                        h_words[0] = _cvt_rn_f16x2_f32_1471;
                        uint32_t _cvt_rn_f16x2_f32_1472;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1472) : "f"(S[35]), "f"(S[34]));
                        h_words[1] = _cvt_rn_f16x2_f32_1472;
#else
                        uint32_t _cvt_rn_f16x2_f32_283;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_283) : "f"(S2[16].y), "f"(S2[16].x));
                        h_words[0] = _cvt_rn_f16x2_f32_283;
                        uint32_t _cvt_rn_f16x2_f32_284;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_284) : "f"(S2[17].y), "f"(S2[17].x));
                        h_words[1] = _cvt_rn_f16x2_f32_284;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1473;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1473) : "f"(S[37]), "f"(S[36]));
                        h_words[0] = _cvt_rn_f16x2_f32_1473;
                        uint32_t _cvt_rn_f16x2_f32_1474;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1474) : "f"(S[39]), "f"(S[38]));
                        h_words[1] = _cvt_rn_f16x2_f32_1474;
#else
                        uint32_t _cvt_rn_f16x2_f32_285;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_285) : "f"(S2[18].y), "f"(S2[18].x));
                        h_words[0] = _cvt_rn_f16x2_f32_285;
                        uint32_t _cvt_rn_f16x2_f32_286;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_286) : "f"(S2[19].y), "f"(S2[19].x));
                        h_words[1] = _cvt_rn_f16x2_f32_286;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 16ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1475;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1475) : "f"(S[41]), "f"(S[40]));
                        h_words[0] = _cvt_rn_f16x2_f32_1475;
                        uint32_t _cvt_rn_f16x2_f32_1476;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1476) : "f"(S[43]), "f"(S[42]));
                        h_words[1] = _cvt_rn_f16x2_f32_1476;
#else
                        uint32_t _cvt_rn_f16x2_f32_287;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_287) : "f"(S2[20].y), "f"(S2[20].x));
                        h_words[0] = _cvt_rn_f16x2_f32_287;
                        uint32_t _cvt_rn_f16x2_f32_288;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_288) : "f"(S2[21].y), "f"(S2[21].x));
                        h_words[1] = _cvt_rn_f16x2_f32_288;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1477;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1477) : "f"(S[45]), "f"(S[44]));
                        h_words[0] = _cvt_rn_f16x2_f32_1477;
                        uint32_t _cvt_rn_f16x2_f32_1478;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1478) : "f"(S[47]), "f"(S[46]));
                        h_words[1] = _cvt_rn_f16x2_f32_1478;
#else
                        uint32_t _cvt_rn_f16x2_f32_289;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_289) : "f"(S2[22].y), "f"(S2[22].x));
                        h_words[0] = _cvt_rn_f16x2_f32_289;
                        uint32_t _cvt_rn_f16x2_f32_290;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_290) : "f"(S2[23].y), "f"(S2[23].x));
                        h_words[1] = _cvt_rn_f16x2_f32_290;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 48ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1479;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1479) : "f"(S[49]), "f"(S[48]));
                        h_words[0] = _cvt_rn_f16x2_f32_1479;
                        uint32_t _cvt_rn_f16x2_f32_1480;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1480) : "f"(S[51]), "f"(S[50]));
                        h_words[1] = _cvt_rn_f16x2_f32_1480;
#else
                        uint32_t _cvt_rn_f16x2_f32_291;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_291) : "f"(S2[24].y), "f"(S2[24].x));
                        h_words[0] = _cvt_rn_f16x2_f32_291;
                        uint32_t _cvt_rn_f16x2_f32_292;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_292) : "f"(S2[25].y), "f"(S2[25].x));
                        h_words[1] = _cvt_rn_f16x2_f32_292;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1481;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1481) : "f"(S[53]), "f"(S[52]));
                        h_words[0] = _cvt_rn_f16x2_f32_1481;
                        uint32_t _cvt_rn_f16x2_f32_1482;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1482) : "f"(S[55]), "f"(S[54]));
                        h_words[1] = _cvt_rn_f16x2_f32_1482;
#else
                        uint32_t _cvt_rn_f16x2_f32_293;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_293) : "f"(S2[26].y), "f"(S2[26].x));
                        h_words[0] = _cvt_rn_f16x2_f32_293;
                        uint32_t _cvt_rn_f16x2_f32_294;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_294) : "f"(S2[27].y), "f"(S2[27].x));
                        h_words[1] = _cvt_rn_f16x2_f32_294;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 80ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1483;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1483) : "f"(S[57]), "f"(S[56]));
                        h_words[0] = _cvt_rn_f16x2_f32_1483;
                        uint32_t _cvt_rn_f16x2_f32_1484;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1484) : "f"(S[59]), "f"(S[58]));
                        h_words[1] = _cvt_rn_f16x2_f32_1484;
#else
                        uint32_t _cvt_rn_f16x2_f32_295;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_295) : "f"(S2[28].y), "f"(S2[28].x));
                        h_words[0] = _cvt_rn_f16x2_f32_295;
                        uint32_t _cvt_rn_f16x2_f32_296;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_296) : "f"(S2[29].y), "f"(S2[29].x));
                        h_words[1] = _cvt_rn_f16x2_f32_296;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1485;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1485) : "f"(S[61]), "f"(S[60]));
                        h_words[0] = _cvt_rn_f16x2_f32_1485;
                        uint32_t _cvt_rn_f16x2_f32_1486;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1486) : "f"(S[63]), "f"(S[62]));
                        h_words[1] = _cvt_rn_f16x2_f32_1486;
#else
                        uint32_t _cvt_rn_f16x2_f32_297;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_297) : "f"(S2[30].y), "f"(S2[30].x));
                        h_words[0] = _cvt_rn_f16x2_f32_297;
                        uint32_t _cvt_rn_f16x2_f32_298;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_298) : "f"(S2[31].y), "f"(S2[31].x));
                        h_words[1] = _cvt_rn_f16x2_f32_298;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 112ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1487;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1487) : "f"(S[65]), "f"(S[64]));
                        h_words[0] = _cvt_rn_f16x2_f32_1487;
                        uint32_t _cvt_rn_f16x2_f32_1488;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1488) : "f"(S[67]), "f"(S[66]));
                        h_words[1] = _cvt_rn_f16x2_f32_1488;
#else
                        uint32_t _cvt_rn_f16x2_f32_299;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_299) : "f"(S2[32].y), "f"(S2[32].x));
                        h_words[0] = _cvt_rn_f16x2_f32_299;
                        uint32_t _cvt_rn_f16x2_f32_300;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_300) : "f"(S2[33].y), "f"(S2[33].x));
                        h_words[1] = _cvt_rn_f16x2_f32_300;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1489;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1489) : "f"(S[69]), "f"(S[68]));
                        h_words[0] = _cvt_rn_f16x2_f32_1489;
                        uint32_t _cvt_rn_f16x2_f32_1490;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1490) : "f"(S[71]), "f"(S[70]));
                        h_words[1] = _cvt_rn_f16x2_f32_1490;
#else
                        uint32_t _cvt_rn_f16x2_f32_301;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_301) : "f"(S2[34].y), "f"(S2[34].x));
                        h_words[0] = _cvt_rn_f16x2_f32_301;
                        uint32_t _cvt_rn_f16x2_f32_302;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_302) : "f"(S2[35].y), "f"(S2[35].x));
                        h_words[1] = _cvt_rn_f16x2_f32_302;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 16ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1491;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1491) : "f"(S[73]), "f"(S[72]));
                        h_words[0] = _cvt_rn_f16x2_f32_1491;
                        uint32_t _cvt_rn_f16x2_f32_1492;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1492) : "f"(S[75]), "f"(S[74]));
                        h_words[1] = _cvt_rn_f16x2_f32_1492;
#else
                        uint32_t _cvt_rn_f16x2_f32_303;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_303) : "f"(S2[36].y), "f"(S2[36].x));
                        h_words[0] = _cvt_rn_f16x2_f32_303;
                        uint32_t _cvt_rn_f16x2_f32_304;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_304) : "f"(S2[37].y), "f"(S2[37].x));
                        h_words[1] = _cvt_rn_f16x2_f32_304;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1493;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1493) : "f"(S[77]), "f"(S[76]));
                        h_words[0] = _cvt_rn_f16x2_f32_1493;
                        uint32_t _cvt_rn_f16x2_f32_1494;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1494) : "f"(S[79]), "f"(S[78]));
                        h_words[1] = _cvt_rn_f16x2_f32_1494;
#else
                        uint32_t _cvt_rn_f16x2_f32_305;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_305) : "f"(S2[38].y), "f"(S2[38].x));
                        h_words[0] = _cvt_rn_f16x2_f32_305;
                        uint32_t _cvt_rn_f16x2_f32_306;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_306) : "f"(S2[39].y), "f"(S2[39].x));
                        h_words[1] = _cvt_rn_f16x2_f32_306;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 48ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1495;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1495) : "f"(S[81]), "f"(S[80]));
                        h_words[0] = _cvt_rn_f16x2_f32_1495;
                        uint32_t _cvt_rn_f16x2_f32_1496;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1496) : "f"(S[83]), "f"(S[82]));
                        h_words[1] = _cvt_rn_f16x2_f32_1496;
#else
                        uint32_t _cvt_rn_f16x2_f32_307;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_307) : "f"(S2[40].y), "f"(S2[40].x));
                        h_words[0] = _cvt_rn_f16x2_f32_307;
                        uint32_t _cvt_rn_f16x2_f32_308;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_308) : "f"(S2[41].y), "f"(S2[41].x));
                        h_words[1] = _cvt_rn_f16x2_f32_308;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1497;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1497) : "f"(S[85]), "f"(S[84]));
                        h_words[0] = _cvt_rn_f16x2_f32_1497;
                        uint32_t _cvt_rn_f16x2_f32_1498;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1498) : "f"(S[87]), "f"(S[86]));
                        h_words[1] = _cvt_rn_f16x2_f32_1498;
#else
                        uint32_t _cvt_rn_f16x2_f32_309;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_309) : "f"(S2[42].y), "f"(S2[42].x));
                        h_words[0] = _cvt_rn_f16x2_f32_309;
                        uint32_t _cvt_rn_f16x2_f32_310;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_310) : "f"(S2[43].y), "f"(S2[43].x));
                        h_words[1] = _cvt_rn_f16x2_f32_310;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 80ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1499;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1499) : "f"(S[89]), "f"(S[88]));
                        h_words[0] = _cvt_rn_f16x2_f32_1499;
                        uint32_t _cvt_rn_f16x2_f32_1500;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1500) : "f"(S[91]), "f"(S[90]));
                        h_words[1] = _cvt_rn_f16x2_f32_1500;
#else
                        uint32_t _cvt_rn_f16x2_f32_311;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_311) : "f"(S2[44].y), "f"(S2[44].x));
                        h_words[0] = _cvt_rn_f16x2_f32_311;
                        uint32_t _cvt_rn_f16x2_f32_312;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_312) : "f"(S2[45].y), "f"(S2[45].x));
                        h_words[1] = _cvt_rn_f16x2_f32_312;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1501;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1501) : "f"(S[93]), "f"(S[92]));
                        h_words[0] = _cvt_rn_f16x2_f32_1501;
                        uint32_t _cvt_rn_f16x2_f32_1502;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1502) : "f"(S[95]), "f"(S[94]));
                        h_words[1] = _cvt_rn_f16x2_f32_1502;
#else
                        uint32_t _cvt_rn_f16x2_f32_313;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_313) : "f"(S2[46].y), "f"(S2[46].x));
                        h_words[0] = _cvt_rn_f16x2_f32_313;
                        uint32_t _cvt_rn_f16x2_f32_314;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_314) : "f"(S2[47].y), "f"(S2[47].x));
                        h_words[1] = _cvt_rn_f16x2_f32_314;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 112ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1503;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1503) : "f"(S[97]), "f"(S[96]));
                        h_words[0] = _cvt_rn_f16x2_f32_1503;
                        uint32_t _cvt_rn_f16x2_f32_1504;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1504) : "f"(S[99]), "f"(S[98]));
                        h_words[1] = _cvt_rn_f16x2_f32_1504;
#else
                        uint32_t _cvt_rn_f16x2_f32_315;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_315) : "f"(S2[48].y), "f"(S2[48].x));
                        h_words[0] = _cvt_rn_f16x2_f32_315;
                        uint32_t _cvt_rn_f16x2_f32_316;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_316) : "f"(S2[49].y), "f"(S2[49].x));
                        h_words[1] = _cvt_rn_f16x2_f32_316;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1505;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1505) : "f"(S[101]), "f"(S[100]));
                        h_words[0] = _cvt_rn_f16x2_f32_1505;
                        uint32_t _cvt_rn_f16x2_f32_1506;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1506) : "f"(S[103]), "f"(S[102]));
                        h_words[1] = _cvt_rn_f16x2_f32_1506;
#else
                        uint32_t _cvt_rn_f16x2_f32_317;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_317) : "f"(S2[50].y), "f"(S2[50].x));
                        h_words[0] = _cvt_rn_f16x2_f32_317;
                        uint32_t _cvt_rn_f16x2_f32_318;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_318) : "f"(S2[51].y), "f"(S2[51].x));
                        h_words[1] = _cvt_rn_f16x2_f32_318;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 16ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1507;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1507) : "f"(S[105]), "f"(S[104]));
                        h_words[0] = _cvt_rn_f16x2_f32_1507;
                        uint32_t _cvt_rn_f16x2_f32_1508;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1508) : "f"(S[107]), "f"(S[106]));
                        h_words[1] = _cvt_rn_f16x2_f32_1508;
#else
                        uint32_t _cvt_rn_f16x2_f32_319;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_319) : "f"(S2[52].y), "f"(S2[52].x));
                        h_words[0] = _cvt_rn_f16x2_f32_319;
                        uint32_t _cvt_rn_f16x2_f32_320;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_320) : "f"(S2[53].y), "f"(S2[53].x));
                        h_words[1] = _cvt_rn_f16x2_f32_320;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1509;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1509) : "f"(S[109]), "f"(S[108]));
                        h_words[0] = _cvt_rn_f16x2_f32_1509;
                        uint32_t _cvt_rn_f16x2_f32_1510;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1510) : "f"(S[111]), "f"(S[110]));
                        h_words[1] = _cvt_rn_f16x2_f32_1510;
#else
                        uint32_t _cvt_rn_f16x2_f32_321;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_321) : "f"(S2[54].y), "f"(S2[54].x));
                        h_words[0] = _cvt_rn_f16x2_f32_321;
                        uint32_t _cvt_rn_f16x2_f32_322;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_322) : "f"(S2[55].y), "f"(S2[55].x));
                        h_words[1] = _cvt_rn_f16x2_f32_322;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 48ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1511;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1511) : "f"(S[113]), "f"(S[112]));
                        h_words[0] = _cvt_rn_f16x2_f32_1511;
                        uint32_t _cvt_rn_f16x2_f32_1512;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1512) : "f"(S[115]), "f"(S[114]));
                        h_words[1] = _cvt_rn_f16x2_f32_1512;
#else
                        uint32_t _cvt_rn_f16x2_f32_323;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_323) : "f"(S2[56].y), "f"(S2[56].x));
                        h_words[0] = _cvt_rn_f16x2_f32_323;
                        uint32_t _cvt_rn_f16x2_f32_324;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_324) : "f"(S2[57].y), "f"(S2[57].x));
                        h_words[1] = _cvt_rn_f16x2_f32_324;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1513;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1513) : "f"(S[117]), "f"(S[116]));
                        h_words[0] = _cvt_rn_f16x2_f32_1513;
                        uint32_t _cvt_rn_f16x2_f32_1514;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1514) : "f"(S[119]), "f"(S[118]));
                        h_words[1] = _cvt_rn_f16x2_f32_1514;
#else
                        uint32_t _cvt_rn_f16x2_f32_325;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_325) : "f"(S2[58].y), "f"(S2[58].x));
                        h_words[0] = _cvt_rn_f16x2_f32_325;
                        uint32_t _cvt_rn_f16x2_f32_326;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_326) : "f"(S2[59].y), "f"(S2[59].x));
                        h_words[1] = _cvt_rn_f16x2_f32_326;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 80ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1515;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1515) : "f"(S[121]), "f"(S[120]));
                        h_words[0] = _cvt_rn_f16x2_f32_1515;
                        uint32_t _cvt_rn_f16x2_f32_1516;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1516) : "f"(S[123]), "f"(S[122]));
                        h_words[1] = _cvt_rn_f16x2_f32_1516;
#else
                        uint32_t _cvt_rn_f16x2_f32_327;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_327) : "f"(S2[60].y), "f"(S2[60].x));
                        h_words[0] = _cvt_rn_f16x2_f32_327;
                        uint32_t _cvt_rn_f16x2_f32_328;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_328) : "f"(S2[61].y), "f"(S2[61].x));
                        h_words[1] = _cvt_rn_f16x2_f32_328;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
#if __CUDA_ARCH__ == 900
                        uint32_t _cvt_rn_f16x2_f32_1517;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1517) : "f"(S[125]), "f"(S[124]));
                        h_words[0] = _cvt_rn_f16x2_f32_1517;
                        uint32_t _cvt_rn_f16x2_f32_1518;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_1518) : "f"(S[127]), "f"(S[126]));
                        h_words[1] = _cvt_rn_f16x2_f32_1518;
#else
                        uint32_t _cvt_rn_f16x2_f32_329;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_329) : "f"(S2[62].y), "f"(S2[62].x));
                        h_words[0] = _cvt_rn_f16x2_f32_329;
                        uint32_t _cvt_rn_f16x2_f32_330;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_330) : "f"(S2[63].y), "f"(S2[63].x));
                        h_words[1] = _cvt_rn_f16x2_f32_330;
#endif
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 112ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                    }
                }
            }
        }
    }
}

} // extern "C"
