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

__global__ __launch_bounds__(64, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdp_f3f05058d0b4a43f6589(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __half* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, int* __restrict__ ssm_state_indices, long long* __restrict__ rand_seed, CakeFastDivmod heads, int num_tokens, float scale, int disable_state_update, int use_ssm_state_indices, int philox_rounds, long long state_size_p0, long long state_stride_p1, long long state_stride_p2, long long state_stride_p0, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long ssm_state_indices_stride_p0, long long ssm_state_indices_stride_p1, long long out_stride_p0, long long out_stride_p1, long long out_stride_p2)
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
        float2 S2[64];
        float k_regs[32];
        float2 k2[16];
        float v_regs[4];
        float o4[4];
        unsigned int h_words[2];
        unsigned int h_noise[64];
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
                float2 x0 = make_float2(r_k[0], r_k[1]);
                float2 x1 = make_float2(r_k[2], r_k[3]);
                float2 sum_pair = fma_f32x2_rn_ftz(x0, x0, make_float2(0.0f, 0.0f));
                sum_pair = fma_f32x2_rn_ftz(x1, x1, sum_pair);
                float sum_x = sum_pair.x + sum_pair.y;
                float _warp_reduce_33 = sum_x;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_33 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_33, offset);
                float norm = rsqrtf(_warp_reduce_33 + 1e-06f);
                float2 norm2 = make_float2(norm, norm);
                x0 = mul_f32x2(x0, norm2);
                x1 = mul_f32x2(x1, norm2);
                sK[m * 128 + k_col0] = x0.x;
                sK[m * 128 + k_col0 + 1] = x0.y;
                sK[m * 128 + k_col0 + 2] = x1.x;
                sK[m * 128 + k_col0 + 3] = x1.y;
                long long v_index = (long long)n * v_stride_p0 + (long long)m * v_stride_p1 + (long long)h * v_stride_p2;
                int v_col = lane_local * 2;
                sV[m * 64 + v_col] = (float)v[v_index + (long long)v_col];
                sV[m * 64 + v_col + 1] = (float)v[v_index + (long long)v_col + 1ll];
                if (lane_local == 0) {
                    long long b_index = (long long)n * b_stride_p0 + (long long)m * b_stride_p1 + (long long)h * b_stride_p2;
                    float _exp_35 = __expf(-(float)b[b_index]);
                    float _rcp_34 = approx_rcp(1.0f + _exp_35);
                    sBeta[m] = _rcp_34;
                }
            }
        }
        #pragma unroll 2
        for (int tt = 0; tt < 8; tt++) {
            int t = tt * 2 + warp_local;
            if (t < num_tokens) {
                long long q_index = (long long)n * q_stride_p0 + (long long)t * q_stride_p1 + (long long)h * q_stride_p2;
                {
                    uint2 _vld_36;
                    _vld_36 = *reinterpret_cast<const uint2*>(q + q_index + (long long)k_col0);
                    uint32_t* _vpairs_36 = reinterpret_cast<uint32_t*>(&_vld_36);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&r_q[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) << 16);
                        (&r_q[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) & 0xffff0000u);
                    }
                }
                float2 x0_1 = make_float2(r_q[0], r_q[1]);
                float2 x1_1 = make_float2(r_q[2], r_q[3]);
                float2 sum_pair_1 = fma_f32x2_rn_ftz(x0_1, x0_1, make_float2(0.0f, 0.0f));
                sum_pair_1 = fma_f32x2_rn_ftz(x1_1, x1_1, sum_pair_1);
                float sum_x_1 = sum_pair_1.x + sum_pair_1.y;
                float _warp_reduce_37 = sum_x_1;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_37 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_37, offset);
                float norm_1 = rsqrtf(_warp_reduce_37 + 1e-06f);
                norm_1 = norm_1 * scale;
                float2 norm2_1 = make_float2(norm_1, norm_1);
                x0_1 = mul_f32x2(x0_1, norm2_1);
                x1_1 = mul_f32x2(x1_1, norm2_1);
                sQ[t * 128 + k_col0] = x0_1.x;
                sQ[t * 128 + k_col0 + 1] = x0_1.y;
                sQ[t * 128 + k_col0 + 2] = x1_1.x;
                sQ[t * 128 + k_col0 + 3] = x1_1.y;
                if (lane_local == 0) {
                    long long gate_index = (long long)n * a_stride_p0 + (long long)t * a_stride_p1 + (long long)h * a_stride_p2;
                    float xv = (float)a[gate_index] + dt_bias[h];
                    float softplus_x = xv;
                    if (xv <= 20.0f) {
                        float _exp_39 = __expf(xv);
                        float _log2_38;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_38) : "f"(1.0f + _exp_39));
                        softplus_x = _log2_38 * 0.6931472f;
                    }
                    float _exp_40 = __expf(A_log[h]);
                    float g_log = -_exp_40 * softplus_x;
                    float _exp_41 = __expf(g_log);
                    sAlpha[t] = _exp_41;
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
                key0 = key0 + 2654435769u;
                key1 = key1 + 3144134277u;
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
                    key0 = key0 + 2654435769u;
                    key1 = key1 + 3144134277u;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 0), "r"(h_noise[0] & 536813567), "r"(h_noise[1] & 536813567), "r"(h_noise[2] & 536813567), "r"(h_noise[3] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 1024), "r"(h_noise[4] & 536813567), "r"(h_noise[5] & 536813567), "r"(h_noise[6] & 536813567), "r"(h_noise[7] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 2048), "r"(h_noise[8] & 536813567), "r"(h_noise[9] & 536813567), "r"(h_noise[10] & 536813567), "r"(h_noise[11] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 3072), "r"(h_noise[12] & 536813567), "r"(h_noise[13] & 536813567), "r"(h_noise[14] & 536813567), "r"(h_noise[15] & 536813567) : "memory");
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
            }
            #pragma unroll 1
            for (int t_1 = 0; t_1 < num_tokens; t_1++) {
                float alpha = sAlpha[t_1];
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
                float vn_1 = (v_regs[1] - dot_0) * beta_0;
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
                float vn_2 = (v_regs[2] - dot_1) * beta_0;
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
                float vn_3 = (v_regs[3] - dot_2) * beta_0;
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
                o4[0] = dot_3;
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
                o4[1] = dot_4;
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
                o4[2] = dot_5;
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
                    uint32_t _cvt_rs_x2_f32_58;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_58) : "f"(S2[0].y), "f"(S2[0].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_58;
                    uint32_t _cvt_rs_x2_f32_59;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_59) : "f"(S2[1].y), "f"(S2[1].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_59;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_60;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_60) : "f"(S2[8].y), "f"(S2[8].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_60;
                    uint32_t _cvt_rs_x2_f32_61;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_61) : "f"(S2[9].y), "f"(S2[9].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_61;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 1024));
                    uint32_t _cvt_rs_x2_f32_62;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_62) : "f"(S2[2].y), "f"(S2[2].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_62;
                    uint32_t _cvt_rs_x2_f32_63;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_63) : "f"(S2[3].y), "f"(S2[3].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_63;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 16ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_64;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_64) : "f"(S2[10].y), "f"(S2[10].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_64;
                    uint32_t _cvt_rs_x2_f32_65;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_65) : "f"(S2[11].y), "f"(S2[11].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_65;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 80ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 2048));
                    uint32_t _cvt_rs_x2_f32_66;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_66) : "f"(S2[4].y), "f"(S2[4].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_66;
                    uint32_t _cvt_rs_x2_f32_67;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_67) : "f"(S2[5].y), "f"(S2[5].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_67;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_68;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_68) : "f"(S2[12].y), "f"(S2[12].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_68;
                    uint32_t _cvt_rs_x2_f32_69;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_69) : "f"(S2[13].y), "f"(S2[13].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_69;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 3072));
                    uint32_t _cvt_rs_x2_f32_70;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_70) : "f"(S2[6].y), "f"(S2[6].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_70;
                    uint32_t _cvt_rs_x2_f32_71;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_71) : "f"(S2[7].y), "f"(S2[7].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_71;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 48ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_72;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_72) : "f"(S2[14].y), "f"(S2[14].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_72;
                    uint32_t _cvt_rs_x2_f32_73;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_73) : "f"(S2[15].y), "f"(S2[15].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_73;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 112ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 4096));
                    uint32_t _cvt_rs_x2_f32_74;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_74) : "f"(S2[16].y), "f"(S2[16].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_74;
                    uint32_t _cvt_rs_x2_f32_75;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_75) : "f"(S2[17].y), "f"(S2[17].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_75;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_76;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_76) : "f"(S2[24].y), "f"(S2[24].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_76;
                    uint32_t _cvt_rs_x2_f32_77;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_77) : "f"(S2[25].y), "f"(S2[25].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_77;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 5120));
                    uint32_t _cvt_rs_x2_f32_78;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_78) : "f"(S2[18].y), "f"(S2[18].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_78;
                    uint32_t _cvt_rs_x2_f32_79;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_79) : "f"(S2[19].y), "f"(S2[19].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_79;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 16ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_80;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_80) : "f"(S2[26].y), "f"(S2[26].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_80;
                    uint32_t _cvt_rs_x2_f32_81;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_81) : "f"(S2[27].y), "f"(S2[27].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_81;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 80ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 6144));
                    uint32_t _cvt_rs_x2_f32_82;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_82) : "f"(S2[20].y), "f"(S2[20].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_82;
                    uint32_t _cvt_rs_x2_f32_83;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_83) : "f"(S2[21].y), "f"(S2[21].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_83;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_84;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_84) : "f"(S2[28].y), "f"(S2[28].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_84;
                    uint32_t _cvt_rs_x2_f32_85;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_85) : "f"(S2[29].y), "f"(S2[29].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_85;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 7168));
                    uint32_t _cvt_rs_x2_f32_86;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_86) : "f"(S2[22].y), "f"(S2[22].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_86;
                    uint32_t _cvt_rs_x2_f32_87;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_87) : "f"(S2[23].y), "f"(S2[23].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_87;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 48ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_88;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_88) : "f"(S2[30].y), "f"(S2[30].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_88;
                    uint32_t _cvt_rs_x2_f32_89;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_89) : "f"(S2[31].y), "f"(S2[31].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_89;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 112ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 8192));
                    uint32_t _cvt_rs_x2_f32_90;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_90) : "f"(S2[32].y), "f"(S2[32].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_90;
                    uint32_t _cvt_rs_x2_f32_91;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_91) : "f"(S2[33].y), "f"(S2[33].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_91;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_92;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_92) : "f"(S2[40].y), "f"(S2[40].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_92;
                    uint32_t _cvt_rs_x2_f32_93;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_93) : "f"(S2[41].y), "f"(S2[41].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_93;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 9216));
                    uint32_t _cvt_rs_x2_f32_94;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_94) : "f"(S2[34].y), "f"(S2[34].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_94;
                    uint32_t _cvt_rs_x2_f32_95;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_95) : "f"(S2[35].y), "f"(S2[35].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_95;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 16ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_96;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_96) : "f"(S2[42].y), "f"(S2[42].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_96;
                    uint32_t _cvt_rs_x2_f32_97;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_97) : "f"(S2[43].y), "f"(S2[43].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_97;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 80ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 10240));
                    uint32_t _cvt_rs_x2_f32_98;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_98) : "f"(S2[36].y), "f"(S2[36].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_98;
                    uint32_t _cvt_rs_x2_f32_99;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_99) : "f"(S2[37].y), "f"(S2[37].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_99;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_100;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_100) : "f"(S2[44].y), "f"(S2[44].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_100;
                    uint32_t _cvt_rs_x2_f32_101;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_101) : "f"(S2[45].y), "f"(S2[45].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_101;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 11264));
                    uint32_t _cvt_rs_x2_f32_102;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_102) : "f"(S2[38].y), "f"(S2[38].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_102;
                    uint32_t _cvt_rs_x2_f32_103;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_103) : "f"(S2[39].y), "f"(S2[39].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_103;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 48ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_104;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_104) : "f"(S2[46].y), "f"(S2[46].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_104;
                    uint32_t _cvt_rs_x2_f32_105;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_105) : "f"(S2[47].y), "f"(S2[47].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_105;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 112ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 12288));
                    uint32_t _cvt_rs_x2_f32_106;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_106) : "f"(S2[48].y), "f"(S2[48].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_106;
                    uint32_t _cvt_rs_x2_f32_107;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_107) : "f"(S2[49].y), "f"(S2[49].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_107;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_108;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_108) : "f"(S2[56].y), "f"(S2[56].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_108;
                    uint32_t _cvt_rs_x2_f32_109;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_109) : "f"(S2[57].y), "f"(S2[57].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_109;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 13312));
                    uint32_t _cvt_rs_x2_f32_110;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_110) : "f"(S2[50].y), "f"(S2[50].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_110;
                    uint32_t _cvt_rs_x2_f32_111;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_111) : "f"(S2[51].y), "f"(S2[51].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_111;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 16ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_112;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_112) : "f"(S2[58].y), "f"(S2[58].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_112;
                    uint32_t _cvt_rs_x2_f32_113;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_113) : "f"(S2[59].y), "f"(S2[59].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_113;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 80ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 14336));
                    uint32_t _cvt_rs_x2_f32_114;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_114) : "f"(S2[52].y), "f"(S2[52].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_114;
                    uint32_t _cvt_rs_x2_f32_115;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_115) : "f"(S2[53].y), "f"(S2[53].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_115;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_116;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_116) : "f"(S2[60].y), "f"(S2[60].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_116;
                    uint32_t _cvt_rs_x2_f32_117;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_117) : "f"(S2[61].y), "f"(S2[61].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_117;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 15360));
                    uint32_t _cvt_rs_x2_f32_118;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_118) : "f"(S2[54].y), "f"(S2[54].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_118;
                    uint32_t _cvt_rs_x2_f32_119;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_119) : "f"(S2[55].y), "f"(S2[55].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_119;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 48ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_120;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_120) : "f"(S2[62].y), "f"(S2[62].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_120;
                    uint32_t _cvt_rs_x2_f32_121;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_121) : "f"(S2[63].y), "f"(S2[63].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_121;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 112ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                }
                if (philox_rounds == 0) {
                    uint32_t _cvt_rn_f16x2_f32_122;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_122) : "f"(S2[0].y), "f"(S2[0].x));
                    h_words[0] = _cvt_rn_f16x2_f32_122;
                    uint32_t _cvt_rn_f16x2_f32_123;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_123) : "f"(S2[1].y), "f"(S2[1].x));
                    h_words[1] = _cvt_rn_f16x2_f32_123;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_124;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_124) : "f"(S2[2].y), "f"(S2[2].x));
                    h_words[0] = _cvt_rn_f16x2_f32_124;
                    uint32_t _cvt_rn_f16x2_f32_125;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_125) : "f"(S2[3].y), "f"(S2[3].x));
                    h_words[1] = _cvt_rn_f16x2_f32_125;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 16ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_126;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_126) : "f"(S2[4].y), "f"(S2[4].x));
                    h_words[0] = _cvt_rn_f16x2_f32_126;
                    uint32_t _cvt_rn_f16x2_f32_127;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_127) : "f"(S2[5].y), "f"(S2[5].x));
                    h_words[1] = _cvt_rn_f16x2_f32_127;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_128;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_128) : "f"(S2[6].y), "f"(S2[6].x));
                    h_words[0] = _cvt_rn_f16x2_f32_128;
                    uint32_t _cvt_rn_f16x2_f32_129;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_129) : "f"(S2[7].y), "f"(S2[7].x));
                    h_words[1] = _cvt_rn_f16x2_f32_129;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 48ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_130;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_130) : "f"(S2[8].y), "f"(S2[8].x));
                    h_words[0] = _cvt_rn_f16x2_f32_130;
                    uint32_t _cvt_rn_f16x2_f32_131;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_131) : "f"(S2[9].y), "f"(S2[9].x));
                    h_words[1] = _cvt_rn_f16x2_f32_131;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_132;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_132) : "f"(S2[10].y), "f"(S2[10].x));
                    h_words[0] = _cvt_rn_f16x2_f32_132;
                    uint32_t _cvt_rn_f16x2_f32_133;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_133) : "f"(S2[11].y), "f"(S2[11].x));
                    h_words[1] = _cvt_rn_f16x2_f32_133;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 80ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_134;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_134) : "f"(S2[12].y), "f"(S2[12].x));
                    h_words[0] = _cvt_rn_f16x2_f32_134;
                    uint32_t _cvt_rn_f16x2_f32_135;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_135) : "f"(S2[13].y), "f"(S2[13].x));
                    h_words[1] = _cvt_rn_f16x2_f32_135;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_136;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_136) : "f"(S2[14].y), "f"(S2[14].x));
                    h_words[0] = _cvt_rn_f16x2_f32_136;
                    uint32_t _cvt_rn_f16x2_f32_137;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_137) : "f"(S2[15].y), "f"(S2[15].x));
                    h_words[1] = _cvt_rn_f16x2_f32_137;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 112ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_138;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_138) : "f"(S2[16].y), "f"(S2[16].x));
                    h_words[0] = _cvt_rn_f16x2_f32_138;
                    uint32_t _cvt_rn_f16x2_f32_139;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_139) : "f"(S2[17].y), "f"(S2[17].x));
                    h_words[1] = _cvt_rn_f16x2_f32_139;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_140;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_140) : "f"(S2[18].y), "f"(S2[18].x));
                    h_words[0] = _cvt_rn_f16x2_f32_140;
                    uint32_t _cvt_rn_f16x2_f32_141;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_141) : "f"(S2[19].y), "f"(S2[19].x));
                    h_words[1] = _cvt_rn_f16x2_f32_141;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 16ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_142;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_142) : "f"(S2[20].y), "f"(S2[20].x));
                    h_words[0] = _cvt_rn_f16x2_f32_142;
                    uint32_t _cvt_rn_f16x2_f32_143;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_143) : "f"(S2[21].y), "f"(S2[21].x));
                    h_words[1] = _cvt_rn_f16x2_f32_143;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_144;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_144) : "f"(S2[22].y), "f"(S2[22].x));
                    h_words[0] = _cvt_rn_f16x2_f32_144;
                    uint32_t _cvt_rn_f16x2_f32_145;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_145) : "f"(S2[23].y), "f"(S2[23].x));
                    h_words[1] = _cvt_rn_f16x2_f32_145;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 48ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_146;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_146) : "f"(S2[24].y), "f"(S2[24].x));
                    h_words[0] = _cvt_rn_f16x2_f32_146;
                    uint32_t _cvt_rn_f16x2_f32_147;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_147) : "f"(S2[25].y), "f"(S2[25].x));
                    h_words[1] = _cvt_rn_f16x2_f32_147;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_148;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_148) : "f"(S2[26].y), "f"(S2[26].x));
                    h_words[0] = _cvt_rn_f16x2_f32_148;
                    uint32_t _cvt_rn_f16x2_f32_149;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_149) : "f"(S2[27].y), "f"(S2[27].x));
                    h_words[1] = _cvt_rn_f16x2_f32_149;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 80ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_150;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_150) : "f"(S2[28].y), "f"(S2[28].x));
                    h_words[0] = _cvt_rn_f16x2_f32_150;
                    uint32_t _cvt_rn_f16x2_f32_151;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_151) : "f"(S2[29].y), "f"(S2[29].x));
                    h_words[1] = _cvt_rn_f16x2_f32_151;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_152;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_152) : "f"(S2[30].y), "f"(S2[30].x));
                    h_words[0] = _cvt_rn_f16x2_f32_152;
                    uint32_t _cvt_rn_f16x2_f32_153;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_153) : "f"(S2[31].y), "f"(S2[31].x));
                    h_words[1] = _cvt_rn_f16x2_f32_153;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 112ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_154;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_154) : "f"(S2[32].y), "f"(S2[32].x));
                    h_words[0] = _cvt_rn_f16x2_f32_154;
                    uint32_t _cvt_rn_f16x2_f32_155;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_155) : "f"(S2[33].y), "f"(S2[33].x));
                    h_words[1] = _cvt_rn_f16x2_f32_155;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_156;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_156) : "f"(S2[34].y), "f"(S2[34].x));
                    h_words[0] = _cvt_rn_f16x2_f32_156;
                    uint32_t _cvt_rn_f16x2_f32_157;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_157) : "f"(S2[35].y), "f"(S2[35].x));
                    h_words[1] = _cvt_rn_f16x2_f32_157;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 16ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_158;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_158) : "f"(S2[36].y), "f"(S2[36].x));
                    h_words[0] = _cvt_rn_f16x2_f32_158;
                    uint32_t _cvt_rn_f16x2_f32_159;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_159) : "f"(S2[37].y), "f"(S2[37].x));
                    h_words[1] = _cvt_rn_f16x2_f32_159;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_160;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_160) : "f"(S2[38].y), "f"(S2[38].x));
                    h_words[0] = _cvt_rn_f16x2_f32_160;
                    uint32_t _cvt_rn_f16x2_f32_161;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_161) : "f"(S2[39].y), "f"(S2[39].x));
                    h_words[1] = _cvt_rn_f16x2_f32_161;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 48ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_162;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_162) : "f"(S2[40].y), "f"(S2[40].x));
                    h_words[0] = _cvt_rn_f16x2_f32_162;
                    uint32_t _cvt_rn_f16x2_f32_163;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_163) : "f"(S2[41].y), "f"(S2[41].x));
                    h_words[1] = _cvt_rn_f16x2_f32_163;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_164;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_164) : "f"(S2[42].y), "f"(S2[42].x));
                    h_words[0] = _cvt_rn_f16x2_f32_164;
                    uint32_t _cvt_rn_f16x2_f32_165;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_165) : "f"(S2[43].y), "f"(S2[43].x));
                    h_words[1] = _cvt_rn_f16x2_f32_165;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 80ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_166;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_166) : "f"(S2[44].y), "f"(S2[44].x));
                    h_words[0] = _cvt_rn_f16x2_f32_166;
                    uint32_t _cvt_rn_f16x2_f32_167;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_167) : "f"(S2[45].y), "f"(S2[45].x));
                    h_words[1] = _cvt_rn_f16x2_f32_167;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_168;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_168) : "f"(S2[46].y), "f"(S2[46].x));
                    h_words[0] = _cvt_rn_f16x2_f32_168;
                    uint32_t _cvt_rn_f16x2_f32_169;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_169) : "f"(S2[47].y), "f"(S2[47].x));
                    h_words[1] = _cvt_rn_f16x2_f32_169;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 2ll * state_stride_p2 + 112ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_170;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_170) : "f"(S2[48].y), "f"(S2[48].x));
                    h_words[0] = _cvt_rn_f16x2_f32_170;
                    uint32_t _cvt_rn_f16x2_f32_171;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_171) : "f"(S2[49].y), "f"(S2[49].x));
                    h_words[1] = _cvt_rn_f16x2_f32_171;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_172;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_172) : "f"(S2[50].y), "f"(S2[50].x));
                    h_words[0] = _cvt_rn_f16x2_f32_172;
                    uint32_t _cvt_rn_f16x2_f32_173;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_173) : "f"(S2[51].y), "f"(S2[51].x));
                    h_words[1] = _cvt_rn_f16x2_f32_173;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 16ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_174;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_174) : "f"(S2[52].y), "f"(S2[52].x));
                    h_words[0] = _cvt_rn_f16x2_f32_174;
                    uint32_t _cvt_rn_f16x2_f32_175;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_175) : "f"(S2[53].y), "f"(S2[53].x));
                    h_words[1] = _cvt_rn_f16x2_f32_175;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_176;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_176) : "f"(S2[54].y), "f"(S2[54].x));
                    h_words[0] = _cvt_rn_f16x2_f32_176;
                    uint32_t _cvt_rn_f16x2_f32_177;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_177) : "f"(S2[55].y), "f"(S2[55].x));
                    h_words[1] = _cvt_rn_f16x2_f32_177;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 48ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_178;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_178) : "f"(S2[56].y), "f"(S2[56].x));
                    h_words[0] = _cvt_rn_f16x2_f32_178;
                    uint32_t _cvt_rn_f16x2_f32_179;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_179) : "f"(S2[57].y), "f"(S2[57].x));
                    h_words[1] = _cvt_rn_f16x2_f32_179;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_180;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_180) : "f"(S2[58].y), "f"(S2[58].x));
                    h_words[0] = _cvt_rn_f16x2_f32_180;
                    uint32_t _cvt_rn_f16x2_f32_181;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_181) : "f"(S2[59].y), "f"(S2[59].x));
                    h_words[1] = _cvt_rn_f16x2_f32_181;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 80ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_182;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_182) : "f"(S2[60].y), "f"(S2[60].x));
                    h_words[0] = _cvt_rn_f16x2_f32_182;
                    uint32_t _cvt_rn_f16x2_f32_183;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_183) : "f"(S2[61].y), "f"(S2[61].x));
                    h_words[1] = _cvt_rn_f16x2_f32_183;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_184;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_184) : "f"(S2[62].y), "f"(S2[62].x));
                    h_words[0] = _cvt_rn_f16x2_f32_184;
                    uint32_t _cvt_rn_f16x2_f32_185;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_185) : "f"(S2[63].y), "f"(S2[63].x));
                    h_words[1] = _cvt_rn_f16x2_f32_185;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 3ll * state_stride_p2 + 112ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                }
            }
        }
        if (use_ssm_state_indices == 0) {
            #pragma unroll 1
            for (int t_2 = 0; t_2 < num_tokens; t_2++) {
                float alpha_1 = sAlpha[t_2];
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
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 3]))
                    : "r"(sV_addr + (unsigned int)(m_1 * 256) + group_byte));
                float beta_1 = sBeta[m_1];
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
                float vn_4 = (v_regs[0] - dot_7) * beta_1;
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
                float vn_5 = (v_regs[1] - dot_7_1) * beta_1;
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
                float vn_6 = (v_regs[2] - dot_8) * beta_1;
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
                float vn_7 = (v_regs[3] - dot_9) * beta_1;
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
                o4[0] = dot_10;
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
                o4[1] = dot_11;
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
                o4[2] = dot_12;
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
                    uint32_t _read_special_reg_202;
                    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(_read_special_reg_202));
                    int tid_r = (int)_read_special_reg_202;
                    long long noise_at = read_slot_head + (long long)(tid_r / 4 * 4) * state_stride_p2 + (long long)((tid_r - tid_r / 4 * 4) * 4);
                    if (philox_rounds != 0) {
                        unsigned int key0_1 = seed_lo;
                        unsigned int key1_1 = seed_hi;
                        long long o_16 = noise_at;
                        unsigned long long p0_16 = (unsigned long long)(unsigned int)o_16 * 3528531795ull;
                        h_noise[0] = (unsigned int)(o_16 >> 32ll) ^ key0_1;
                        h_noise[2] = (unsigned int)(p0_16 >> 32ull) ^ key1_1;
                        h_noise[3] = (unsigned int)p0_16;
                        long long o_17 = noise_at + 16ll;
                        unsigned long long p0_17 = (unsigned long long)(unsigned int)o_17 * 3528531795ull;
                        h_noise[4] = (unsigned int)(o_17 >> 32ll) ^ key0_1;
                        h_noise[6] = (unsigned int)(p0_17 >> 32ull) ^ key1_1;
                        h_noise[7] = (unsigned int)p0_17;
                        long long o_18 = noise_at + 32ll;
                        unsigned long long p0_18 = (unsigned long long)(unsigned int)o_18 * 3528531795ull;
                        h_noise[8] = (unsigned int)(o_18 >> 32ll) ^ key0_1;
                        h_noise[10] = (unsigned int)(p0_18 >> 32ull) ^ key1_1;
                        h_noise[11] = (unsigned int)p0_18;
                        long long o_19 = noise_at + 48ll;
                        unsigned long long p0_19 = (unsigned long long)(unsigned int)o_19 * 3528531795ull;
                        h_noise[12] = (unsigned int)(o_19 >> 32ll) ^ key0_1;
                        h_noise[14] = (unsigned int)(p0_19 >> 32ull) ^ key1_1;
                        h_noise[15] = (unsigned int)p0_19;
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
                        unsigned long long q0_16 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long q1_16 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(q1_16 >> 32ull) ^ key0_1;
                        h_noise[1] = (unsigned int)q1_16;
                        h_noise[2] = (unsigned int)(q0_16 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)q0_16;
                        unsigned long long q0_17 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long q1_17 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(q1_17 >> 32ull) ^ key0_1;
                        h_noise[5] = (unsigned int)q1_17;
                        h_noise[6] = (unsigned int)(q0_17 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)q0_17;
                        unsigned long long q0_18 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long q1_18 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(q1_18 >> 32ull) ^ key0_1;
                        h_noise[9] = (unsigned int)q1_18;
                        h_noise[10] = (unsigned int)(q0_18 >> 32ull) ^ h_noise[11] ^ key1_1;
                        h_noise[11] = (unsigned int)q0_18;
                        unsigned long long q0_19 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long q1_19 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(q1_19 >> 32ull) ^ key0_1;
                        h_noise[13] = (unsigned int)q1_19;
                        h_noise[14] = (unsigned int)(q0_19 >> 32ull) ^ h_noise[15] ^ key1_1;
                        h_noise[15] = (unsigned int)q0_19;
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
                        unsigned long long s0_48 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_48 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_48 >> 32ull) ^ h_noise[1] ^ key0_1;
                        h_noise[1] = (unsigned int)s1_48;
                        h_noise[2] = (unsigned int)(s0_48 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)s0_48;
                        unsigned long long s0_49 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_49 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_49 >> 32ull) ^ h_noise[5] ^ key0_1;
                        h_noise[5] = (unsigned int)s1_49;
                        h_noise[6] = (unsigned int)(s0_49 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)s0_49;
                        unsigned long long s0_50 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_50 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_50 >> 32ull) ^ h_noise[9] ^ key0_1;
                        h_noise[9] = (unsigned int)s1_50;
                        h_noise[10] = (unsigned int)(s0_50 >> 32ull) ^ h_noise[11] ^ key1_1;
                        h_noise[11] = (unsigned int)s0_50;
                        unsigned long long s0_51 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_51 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_51 >> 32ull) ^ h_noise[13] ^ key0_1;
                        h_noise[13] = (unsigned int)s1_51;
                        h_noise[14] = (unsigned int)(s0_51 >> 32ull) ^ h_noise[15] ^ key1_1;
                        h_noise[15] = (unsigned int)s0_51;
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
                        unsigned long long s0_64 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_64 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_64 >> 32ull) ^ h_noise[1] ^ key0_1;
                        h_noise[1] = (unsigned int)s1_64;
                        h_noise[2] = (unsigned int)(s0_64 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)s0_64;
                        unsigned long long s0_65 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_65 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_65 >> 32ull) ^ h_noise[5] ^ key0_1;
                        h_noise[5] = (unsigned int)s1_65;
                        h_noise[6] = (unsigned int)(s0_65 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)s0_65;
                        unsigned long long s0_66 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_66 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_66 >> 32ull) ^ h_noise[9] ^ key0_1;
                        h_noise[9] = (unsigned int)s1_66;
                        h_noise[10] = (unsigned int)(s0_66 >> 32ull) ^ h_noise[11] ^ key1_1;
                        h_noise[11] = (unsigned int)s0_66;
                        unsigned long long s0_67 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_67 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_67 >> 32ull) ^ h_noise[13] ^ key0_1;
                        h_noise[13] = (unsigned int)s1_67;
                        h_noise[14] = (unsigned int)(s0_67 >> 32ull) ^ h_noise[15] ^ key1_1;
                        h_noise[15] = (unsigned int)s0_67;
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
                        unsigned long long s0_80 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_80 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_80 >> 32ull) ^ h_noise[1] ^ key0_1;
                        h_noise[1] = (unsigned int)s1_80;
                        h_noise[2] = (unsigned int)(s0_80 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)s0_80;
                        unsigned long long s0_81 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_81 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_81 >> 32ull) ^ h_noise[5] ^ key0_1;
                        h_noise[5] = (unsigned int)s1_81;
                        h_noise[6] = (unsigned int)(s0_81 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)s0_81;
                        unsigned long long s0_82 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_82 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_82 >> 32ull) ^ h_noise[9] ^ key0_1;
                        h_noise[9] = (unsigned int)s1_82;
                        h_noise[10] = (unsigned int)(s0_82 >> 32ull) ^ h_noise[11] ^ key1_1;
                        h_noise[11] = (unsigned int)s0_82;
                        unsigned long long s0_83 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_83 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_83 >> 32ull) ^ h_noise[13] ^ key0_1;
                        h_noise[13] = (unsigned int)s1_83;
                        h_noise[14] = (unsigned int)(s0_83 >> 32ull) ^ h_noise[15] ^ key1_1;
                        h_noise[15] = (unsigned int)s0_83;
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
                        #pragma unroll 1
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
                        }
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
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_237;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_237) : "f"(S2[40].y), "f"(S2[40].x), "r"(h_noise[34] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_237;
                        uint32_t _cvt_rs_x2_f32_238;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_238) : "f"(S2[41].y), "f"(S2[41].x), "r"(h_noise[35] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_238;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_239;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_239) : "f"(S2[34].y), "f"(S2[34].x), "r"(h_noise[36] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_239;
                        uint32_t _cvt_rs_x2_f32_240;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_240) : "f"(S2[35].y), "f"(S2[35].x), "r"(h_noise[37] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_240;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 16ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_241;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_241) : "f"(S2[42].y), "f"(S2[42].x), "r"(h_noise[38] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_241;
                        uint32_t _cvt_rs_x2_f32_242;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_242) : "f"(S2[43].y), "f"(S2[43].x), "r"(h_noise[39] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_242;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 80ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_243;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_243) : "f"(S2[36].y), "f"(S2[36].x), "r"(h_noise[40] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_243;
                        uint32_t _cvt_rs_x2_f32_244;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_244) : "f"(S2[37].y), "f"(S2[37].x), "r"(h_noise[41] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_244;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_245;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_245) : "f"(S2[44].y), "f"(S2[44].x), "r"(h_noise[42] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_245;
                        uint32_t _cvt_rs_x2_f32_246;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_246) : "f"(S2[45].y), "f"(S2[45].x), "r"(h_noise[43] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_246;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_247;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_247) : "f"(S2[38].y), "f"(S2[38].x), "r"(h_noise[44] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_247;
                        uint32_t _cvt_rs_x2_f32_248;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_248) : "f"(S2[39].y), "f"(S2[39].x), "r"(h_noise[45] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_248;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 48ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_249;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_249) : "f"(S2[46].y), "f"(S2[46].x), "r"(h_noise[46] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_249;
                        uint32_t _cvt_rs_x2_f32_250;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_250) : "f"(S2[47].y), "f"(S2[47].x), "r"(h_noise[47] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_250;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 112ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_251;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_251) : "f"(S2[48].y), "f"(S2[48].x), "r"(h_noise[48] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_251;
                        uint32_t _cvt_rs_x2_f32_252;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_252) : "f"(S2[49].y), "f"(S2[49].x), "r"(h_noise[49] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_252;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_253;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_253) : "f"(S2[56].y), "f"(S2[56].x), "r"(h_noise[50] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_253;
                        uint32_t _cvt_rs_x2_f32_254;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_254) : "f"(S2[57].y), "f"(S2[57].x), "r"(h_noise[51] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_254;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_255;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_255) : "f"(S2[50].y), "f"(S2[50].x), "r"(h_noise[52] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_255;
                        uint32_t _cvt_rs_x2_f32_256;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_256) : "f"(S2[51].y), "f"(S2[51].x), "r"(h_noise[53] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_256;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 16ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_257;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_257) : "f"(S2[58].y), "f"(S2[58].x), "r"(h_noise[54] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_257;
                        uint32_t _cvt_rs_x2_f32_258;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_258) : "f"(S2[59].y), "f"(S2[59].x), "r"(h_noise[55] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_258;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 80ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_259;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_259) : "f"(S2[52].y), "f"(S2[52].x), "r"(h_noise[56] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_259;
                        uint32_t _cvt_rs_x2_f32_260;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_260) : "f"(S2[53].y), "f"(S2[53].x), "r"(h_noise[57] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_260;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_261;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_261) : "f"(S2[60].y), "f"(S2[60].x), "r"(h_noise[58] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_261;
                        uint32_t _cvt_rs_x2_f32_262;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_262) : "f"(S2[61].y), "f"(S2[61].x), "r"(h_noise[59] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_262;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_263;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_263) : "f"(S2[54].y), "f"(S2[54].x), "r"(h_noise[60] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_263;
                        uint32_t _cvt_rs_x2_f32_264;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_264) : "f"(S2[55].y), "f"(S2[55].x), "r"(h_noise[61] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_264;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 48ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_265;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_265) : "f"(S2[62].y), "f"(S2[62].x), "r"(h_noise[62] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_265;
                        uint32_t _cvt_rs_x2_f32_266;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_266) : "f"(S2[63].y), "f"(S2[63].x), "r"(h_noise[63] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_266;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 112ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                    }
                    if (philox_rounds == 0) {
                        uint32_t _cvt_rn_f16x2_f32_267;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_267) : "f"(S2[0].y), "f"(S2[0].x));
                        h_words[0] = _cvt_rn_f16x2_f32_267;
                        uint32_t _cvt_rn_f16x2_f32_268;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_268) : "f"(S2[1].y), "f"(S2[1].x));
                        h_words[1] = _cvt_rn_f16x2_f32_268;
                        reinterpret_cast<int2*>(state + (write_rows_base))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_269;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_269) : "f"(S2[2].y), "f"(S2[2].x));
                        h_words[0] = _cvt_rn_f16x2_f32_269;
                        uint32_t _cvt_rn_f16x2_f32_270;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_270) : "f"(S2[3].y), "f"(S2[3].x));
                        h_words[1] = _cvt_rn_f16x2_f32_270;
                        reinterpret_cast<int2*>(state + (write_rows_base + 16ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_271;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_271) : "f"(S2[4].y), "f"(S2[4].x));
                        h_words[0] = _cvt_rn_f16x2_f32_271;
                        uint32_t _cvt_rn_f16x2_f32_272;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_272) : "f"(S2[5].y), "f"(S2[5].x));
                        h_words[1] = _cvt_rn_f16x2_f32_272;
                        reinterpret_cast<int2*>(state + (write_rows_base + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_273;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_273) : "f"(S2[6].y), "f"(S2[6].x));
                        h_words[0] = _cvt_rn_f16x2_f32_273;
                        uint32_t _cvt_rn_f16x2_f32_274;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_274) : "f"(S2[7].y), "f"(S2[7].x));
                        h_words[1] = _cvt_rn_f16x2_f32_274;
                        reinterpret_cast<int2*>(state + (write_rows_base + 48ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_275;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_275) : "f"(S2[8].y), "f"(S2[8].x));
                        h_words[0] = _cvt_rn_f16x2_f32_275;
                        uint32_t _cvt_rn_f16x2_f32_276;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_276) : "f"(S2[9].y), "f"(S2[9].x));
                        h_words[1] = _cvt_rn_f16x2_f32_276;
                        reinterpret_cast<int2*>(state + (write_rows_base + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_277;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_277) : "f"(S2[10].y), "f"(S2[10].x));
                        h_words[0] = _cvt_rn_f16x2_f32_277;
                        uint32_t _cvt_rn_f16x2_f32_278;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_278) : "f"(S2[11].y), "f"(S2[11].x));
                        h_words[1] = _cvt_rn_f16x2_f32_278;
                        reinterpret_cast<int2*>(state + (write_rows_base + 80ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_279;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_279) : "f"(S2[12].y), "f"(S2[12].x));
                        h_words[0] = _cvt_rn_f16x2_f32_279;
                        uint32_t _cvt_rn_f16x2_f32_280;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_280) : "f"(S2[13].y), "f"(S2[13].x));
                        h_words[1] = _cvt_rn_f16x2_f32_280;
                        reinterpret_cast<int2*>(state + (write_rows_base + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_281;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_281) : "f"(S2[14].y), "f"(S2[14].x));
                        h_words[0] = _cvt_rn_f16x2_f32_281;
                        uint32_t _cvt_rn_f16x2_f32_282;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_282) : "f"(S2[15].y), "f"(S2[15].x));
                        h_words[1] = _cvt_rn_f16x2_f32_282;
                        reinterpret_cast<int2*>(state + (write_rows_base + 112ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_283;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_283) : "f"(S2[16].y), "f"(S2[16].x));
                        h_words[0] = _cvt_rn_f16x2_f32_283;
                        uint32_t _cvt_rn_f16x2_f32_284;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_284) : "f"(S2[17].y), "f"(S2[17].x));
                        h_words[1] = _cvt_rn_f16x2_f32_284;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_285;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_285) : "f"(S2[18].y), "f"(S2[18].x));
                        h_words[0] = _cvt_rn_f16x2_f32_285;
                        uint32_t _cvt_rn_f16x2_f32_286;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_286) : "f"(S2[19].y), "f"(S2[19].x));
                        h_words[1] = _cvt_rn_f16x2_f32_286;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 16ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_287;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_287) : "f"(S2[20].y), "f"(S2[20].x));
                        h_words[0] = _cvt_rn_f16x2_f32_287;
                        uint32_t _cvt_rn_f16x2_f32_288;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_288) : "f"(S2[21].y), "f"(S2[21].x));
                        h_words[1] = _cvt_rn_f16x2_f32_288;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_289;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_289) : "f"(S2[22].y), "f"(S2[22].x));
                        h_words[0] = _cvt_rn_f16x2_f32_289;
                        uint32_t _cvt_rn_f16x2_f32_290;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_290) : "f"(S2[23].y), "f"(S2[23].x));
                        h_words[1] = _cvt_rn_f16x2_f32_290;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 48ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_291;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_291) : "f"(S2[24].y), "f"(S2[24].x));
                        h_words[0] = _cvt_rn_f16x2_f32_291;
                        uint32_t _cvt_rn_f16x2_f32_292;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_292) : "f"(S2[25].y), "f"(S2[25].x));
                        h_words[1] = _cvt_rn_f16x2_f32_292;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_293;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_293) : "f"(S2[26].y), "f"(S2[26].x));
                        h_words[0] = _cvt_rn_f16x2_f32_293;
                        uint32_t _cvt_rn_f16x2_f32_294;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_294) : "f"(S2[27].y), "f"(S2[27].x));
                        h_words[1] = _cvt_rn_f16x2_f32_294;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 80ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_295;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_295) : "f"(S2[28].y), "f"(S2[28].x));
                        h_words[0] = _cvt_rn_f16x2_f32_295;
                        uint32_t _cvt_rn_f16x2_f32_296;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_296) : "f"(S2[29].y), "f"(S2[29].x));
                        h_words[1] = _cvt_rn_f16x2_f32_296;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_297;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_297) : "f"(S2[30].y), "f"(S2[30].x));
                        h_words[0] = _cvt_rn_f16x2_f32_297;
                        uint32_t _cvt_rn_f16x2_f32_298;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_298) : "f"(S2[31].y), "f"(S2[31].x));
                        h_words[1] = _cvt_rn_f16x2_f32_298;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 112ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_299;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_299) : "f"(S2[32].y), "f"(S2[32].x));
                        h_words[0] = _cvt_rn_f16x2_f32_299;
                        uint32_t _cvt_rn_f16x2_f32_300;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_300) : "f"(S2[33].y), "f"(S2[33].x));
                        h_words[1] = _cvt_rn_f16x2_f32_300;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_301;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_301) : "f"(S2[34].y), "f"(S2[34].x));
                        h_words[0] = _cvt_rn_f16x2_f32_301;
                        uint32_t _cvt_rn_f16x2_f32_302;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_302) : "f"(S2[35].y), "f"(S2[35].x));
                        h_words[1] = _cvt_rn_f16x2_f32_302;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 16ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_303;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_303) : "f"(S2[36].y), "f"(S2[36].x));
                        h_words[0] = _cvt_rn_f16x2_f32_303;
                        uint32_t _cvt_rn_f16x2_f32_304;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_304) : "f"(S2[37].y), "f"(S2[37].x));
                        h_words[1] = _cvt_rn_f16x2_f32_304;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_305;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_305) : "f"(S2[38].y), "f"(S2[38].x));
                        h_words[0] = _cvt_rn_f16x2_f32_305;
                        uint32_t _cvt_rn_f16x2_f32_306;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_306) : "f"(S2[39].y), "f"(S2[39].x));
                        h_words[1] = _cvt_rn_f16x2_f32_306;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 48ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_307;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_307) : "f"(S2[40].y), "f"(S2[40].x));
                        h_words[0] = _cvt_rn_f16x2_f32_307;
                        uint32_t _cvt_rn_f16x2_f32_308;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_308) : "f"(S2[41].y), "f"(S2[41].x));
                        h_words[1] = _cvt_rn_f16x2_f32_308;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_309;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_309) : "f"(S2[42].y), "f"(S2[42].x));
                        h_words[0] = _cvt_rn_f16x2_f32_309;
                        uint32_t _cvt_rn_f16x2_f32_310;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_310) : "f"(S2[43].y), "f"(S2[43].x));
                        h_words[1] = _cvt_rn_f16x2_f32_310;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 80ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_311;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_311) : "f"(S2[44].y), "f"(S2[44].x));
                        h_words[0] = _cvt_rn_f16x2_f32_311;
                        uint32_t _cvt_rn_f16x2_f32_312;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_312) : "f"(S2[45].y), "f"(S2[45].x));
                        h_words[1] = _cvt_rn_f16x2_f32_312;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_313;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_313) : "f"(S2[46].y), "f"(S2[46].x));
                        h_words[0] = _cvt_rn_f16x2_f32_313;
                        uint32_t _cvt_rn_f16x2_f32_314;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_314) : "f"(S2[47].y), "f"(S2[47].x));
                        h_words[1] = _cvt_rn_f16x2_f32_314;
                        reinterpret_cast<int2*>(state + (write_rows_base + 2ll * state_stride_p2 + 112ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_315;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_315) : "f"(S2[48].y), "f"(S2[48].x));
                        h_words[0] = _cvt_rn_f16x2_f32_315;
                        uint32_t _cvt_rn_f16x2_f32_316;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_316) : "f"(S2[49].y), "f"(S2[49].x));
                        h_words[1] = _cvt_rn_f16x2_f32_316;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_317;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_317) : "f"(S2[50].y), "f"(S2[50].x));
                        h_words[0] = _cvt_rn_f16x2_f32_317;
                        uint32_t _cvt_rn_f16x2_f32_318;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_318) : "f"(S2[51].y), "f"(S2[51].x));
                        h_words[1] = _cvt_rn_f16x2_f32_318;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 16ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_319;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_319) : "f"(S2[52].y), "f"(S2[52].x));
                        h_words[0] = _cvt_rn_f16x2_f32_319;
                        uint32_t _cvt_rn_f16x2_f32_320;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_320) : "f"(S2[53].y), "f"(S2[53].x));
                        h_words[1] = _cvt_rn_f16x2_f32_320;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_321;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_321) : "f"(S2[54].y), "f"(S2[54].x));
                        h_words[0] = _cvt_rn_f16x2_f32_321;
                        uint32_t _cvt_rn_f16x2_f32_322;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_322) : "f"(S2[55].y), "f"(S2[55].x));
                        h_words[1] = _cvt_rn_f16x2_f32_322;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 48ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_323;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_323) : "f"(S2[56].y), "f"(S2[56].x));
                        h_words[0] = _cvt_rn_f16x2_f32_323;
                        uint32_t _cvt_rn_f16x2_f32_324;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_324) : "f"(S2[57].y), "f"(S2[57].x));
                        h_words[1] = _cvt_rn_f16x2_f32_324;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_325;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_325) : "f"(S2[58].y), "f"(S2[58].x));
                        h_words[0] = _cvt_rn_f16x2_f32_325;
                        uint32_t _cvt_rn_f16x2_f32_326;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_326) : "f"(S2[59].y), "f"(S2[59].x));
                        h_words[1] = _cvt_rn_f16x2_f32_326;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 80ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_327;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_327) : "f"(S2[60].y), "f"(S2[60].x));
                        h_words[0] = _cvt_rn_f16x2_f32_327;
                        uint32_t _cvt_rn_f16x2_f32_328;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_328) : "f"(S2[61].y), "f"(S2[61].x));
                        h_words[1] = _cvt_rn_f16x2_f32_328;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_329;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_329) : "f"(S2[62].y), "f"(S2[62].x));
                        h_words[0] = _cvt_rn_f16x2_f32_329;
                        uint32_t _cvt_rn_f16x2_f32_330;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_330) : "f"(S2[63].y), "f"(S2[63].x));
                        h_words[1] = _cvt_rn_f16x2_f32_330;
                        reinterpret_cast<int2*>(state + (write_rows_base + 3ll * state_stride_p2 + 112ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                    }
                }
            }
        }
    }
}

} // extern "C"
