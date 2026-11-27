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
kernel_cake_gdp_527260b8ec384f213efd(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __half* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, int* __restrict__ ssm_state_indices, long long* __restrict__ rand_seed, CakeFastDivmod heads, int num_tokens, float scale, int disable_state_update, int use_ssm_state_indices, int philox_rounds, long long state_size_p0, long long state_stride_p1, long long state_stride_p2, long long state_stride_p0, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long ssm_state_indices_stride_p0, long long ssm_state_indices_stride_p1, long long out_stride_p0, long long out_stride_p1, long long out_stride_p2)
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
    int steps = num_tokens;
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
    if (read_state_slot_valid != 0) {
        float S[32];
        float k_regs[16];
        float v_regs[4];
        float o4[4];
        unsigned int h_words[2];
        unsigned int h_noise[16];
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
        unsigned int noise_lane = sNoise_addr + (unsigned int)(tid_0 * 16);
        #pragma unroll 2
        for (int mm = 0; mm < 4; mm++) {
            int m = mm * 4 + warp_local;
            if (m < steps) {
                long long k_index = (long long)n * k_stride_p0 + (long long)m * k_stride_p1 + (long long)h * k_stride_p2;
                {
                    uint2 _vld_8;
                    _vld_8 = *reinterpret_cast<const uint2*>(k + k_index + (long long)k_col0);
                    uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&r_k[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                        (&r_k[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                    }
                }
                float x0_x = r_k[0];
                float x0_y = r_k[1];
                float x1_x = r_k[2];
                float x1_y = r_k[3];
                float sum_x;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x) : "f"(x0_x), "f"(x0_x), "f"(0.0f));
                float sum_y;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y) : "f"(x0_y), "f"(x0_y), "f"(0.0f));
                float _fma_9;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(x1_x), "f"(x1_x), "f"(sum_x));
                sum_x = _fma_9;
                float _fma_10;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(x1_y), "f"(x1_y), "f"(sum_y));
                sum_y = _fma_10;
                sum_x = sum_x + sum_y;
                float _warp_reduce_11 = sum_x;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_11 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_11, offset);
                float norm = rsqrtf(_warp_reduce_11 + 1e-06f);
                x0_x = mul_f32_rn_ftz(x0_x, norm);
                x0_y = mul_f32_rn_ftz(x0_y, norm);
                x1_x = mul_f32_rn_ftz(x1_x, norm);
                x1_y = mul_f32_rn_ftz(x1_y, norm);
                sK[m * 128 + k_col0] = x0_x;
                sK[m * 128 + k_col0 + 1] = x0_y;
                sK[m * 128 + k_col0 + 2] = x1_x;
                sK[m * 128 + k_col0 + 3] = x1_y;
                long long v_index = (long long)n * v_stride_p0 + (long long)m * v_stride_p1 + (long long)h * v_stride_p2;
                int v_col = lane_local * 2;
                sV[m * 64 + v_col] = (float)v[v_index + (long long)v_col];
                sV[m * 64 + v_col + 1] = (float)v[v_index + (long long)v_col + 1ll];
                if (lane_local == 0) {
                    long long b_index = (long long)n * b_stride_p0 + (long long)m * b_stride_p1 + (long long)h * b_stride_p2;
                    float _exp_13 = __expf(-(float)b[b_index]);
                    float _rcp_12 = approx_rcp(1.0f + _exp_13);
                    sBeta[m] = _rcp_12;
                }
            }
        }
        #pragma unroll 2
        for (int tt = 0; tt < 4; tt++) {
            int t = tt * 4 + warp_local;
            if (t < num_tokens) {
                long long q_index = (long long)n * q_stride_p0 + (long long)t * q_stride_p1 + (long long)h * q_stride_p2;
                {
                    uint2 _vld_14;
                    _vld_14 = *reinterpret_cast<const uint2*>(q + q_index + (long long)k_col0);
                    uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&r_q[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) << 16);
                        (&r_q[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) & 0xffff0000u);
                    }
                }
                float x0_x_1 = r_q[0];
                float x0_y_1 = r_q[1];
                float x1_x_1 = r_q[2];
                float x1_y_1 = r_q[3];
                float sum_x_1;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_x_1) : "f"(x0_x_1), "f"(x0_x_1), "f"(0.0f));
                float sum_y_1;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(sum_y_1) : "f"(x0_y_1), "f"(x0_y_1), "f"(0.0f));
                float _fma_15;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(x1_x_1), "f"(x1_x_1), "f"(sum_x_1));
                sum_x_1 = _fma_15;
                float _fma_16;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(x1_y_1), "f"(x1_y_1), "f"(sum_y_1));
                sum_y_1 = _fma_16;
                sum_x_1 = sum_x_1 + sum_y_1;
                float _warp_reduce_17 = sum_x_1;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_17 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_17, offset);
                float norm_1 = rsqrtf(_warp_reduce_17 + 1e-06f);
                norm_1 = norm_1 * scale;
                x0_x_1 = mul_f32_rn_ftz(x0_x_1, norm_1);
                x0_y_1 = mul_f32_rn_ftz(x0_y_1, norm_1);
                x1_x_1 = mul_f32_rn_ftz(x1_x_1, norm_1);
                x1_y_1 = mul_f32_rn_ftz(x1_y_1, norm_1);
                sQ[t * 128 + k_col0] = x0_x_1;
                sQ[t * 128 + k_col0 + 1] = x0_y_1;
                sQ[t * 128 + k_col0 + 2] = x1_x_1;
                sQ[t * 128 + k_col0 + 3] = x1_y_1;
                if (lane_local == 0) {
                    long long gate_index = (long long)n * a_stride_p0 + (long long)t * a_stride_p1 + (long long)h * a_stride_p2;
                    float xv = (float)a[gate_index] + dt_bias[h];
                    float softplus_x = xv;
                    if (xv <= 20.0f) {
                        float _exp_19 = __expf(xv);
                        float _log2_18;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_18) : "f"(1.0f + _exp_19));
                        softplus_x = _log2_18 * 0.6931472f;
                    }
                    float _exp_20 = __expf(A_log[h]);
                    float g_log = -_exp_20 * softplus_x;
                    float _exp_21 = __expf(g_log);
                    sAlpha[t] = _exp_21;
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
            for (int t_1 = 0; t_1 < num_tokens; t_1++) {
                float alpha = sAlpha[t_1];
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
                int m_0 = t_1;
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
                float _fma_22;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x));
                acc0_x = _fma_22;
                float _fma_23;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y));
                acc0_y = _fma_23;
                float _fma_24;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x));
                acc1_x = _fma_24;
                float _fma_25;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y));
                acc1_y = _fma_25;
                float _fma_26;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x));
                acc0_x = _fma_26;
                float _fma_27;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y));
                acc0_y = _fma_27;
                float _fma_28;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x));
                acc1_x = _fma_28;
                float _fma_29;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y));
                acc1_y = _fma_29;
                float _fma_30;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x));
                acc0_x = _fma_30;
                float _fma_31;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y));
                acc0_y = _fma_31;
                float _fma_32;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x));
                acc1_x = _fma_32;
                float _fma_33;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y));
                acc1_y = _fma_33;
                float _fma_34;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x));
                acc0_x = _fma_34;
                float _fma_35;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y));
                acc0_y = _fma_35;
                float _fma_36;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x));
                acc1_x = _fma_36;
                float _fma_37;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y));
                acc1_y = _fma_37;
                float _add_38;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(acc0_x), "f"(acc1_x));
                float _add_39;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(acc0_y), "f"(acc1_y));
                float dot = _add_38 + _add_39;
                float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
                dot = dot + _shfl_xor_sync_40;
                float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
                dot = dot + _shfl_xor_sync_41;
                float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
                dot = dot + _shfl_xor_sync_42;
                float vn_0 = (v_regs[0] - dot) * beta_0;
                float _fma_43;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(k_regs[0]), "f"(vn_0), "f"(S[0]));
                S[0] = _fma_43;
                float _fma_44;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(k_regs[1]), "f"(vn_0), "f"(S[1]));
                S[1] = _fma_44;
                float _fma_45;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(k_regs[2]), "f"(vn_0), "f"(S[2]));
                S[2] = _fma_45;
                float _fma_46;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(k_regs[3]), "f"(vn_0), "f"(S[3]));
                S[3] = _fma_46;
                float _fma_47;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(k_regs[4]), "f"(vn_0), "f"(S[4]));
                S[4] = _fma_47;
                float _fma_48;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(k_regs[5]), "f"(vn_0), "f"(S[5]));
                S[5] = _fma_48;
                float _fma_49;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(k_regs[6]), "f"(vn_0), "f"(S[6]));
                S[6] = _fma_49;
                float _fma_50;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(k_regs[7]), "f"(vn_0), "f"(S[7]));
                S[7] = _fma_50;
                float _fma_51;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(k_regs[8]), "f"(vn_0), "f"(S[8]));
                S[8] = _fma_51;
                float _fma_52;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(k_regs[9]), "f"(vn_0), "f"(S[9]));
                S[9] = _fma_52;
                float _fma_53;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(k_regs[10]), "f"(vn_0), "f"(S[10]));
                S[10] = _fma_53;
                float _fma_54;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(k_regs[11]), "f"(vn_0), "f"(S[11]));
                S[11] = _fma_54;
                float _fma_55;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(k_regs[12]), "f"(vn_0), "f"(S[12]));
                S[12] = _fma_55;
                float _fma_56;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(k_regs[13]), "f"(vn_0), "f"(S[13]));
                S[13] = _fma_56;
                float _fma_57;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(k_regs[14]), "f"(vn_0), "f"(S[14]));
                S[14] = _fma_57;
                float _fma_58;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(k_regs[15]), "f"(vn_0), "f"(S[15]));
                S[15] = _fma_58;
                float acc0_x_0 = 0.0f;
                float acc0_y_0 = 0.0f;
                float acc1_x_0 = 0.0f;
                float acc1_y_0 = 0.0f;
                float _fma_59;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_0));
                acc0_x_0 = _fma_59;
                float _fma_60;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_0));
                acc0_y_0 = _fma_60;
                float _fma_61;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_0));
                acc1_x_0 = _fma_61;
                float _fma_62;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_0));
                acc1_y_0 = _fma_62;
                float _fma_63;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_0));
                acc0_x_0 = _fma_63;
                float _fma_64;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_0));
                acc0_y_0 = _fma_64;
                float _fma_65;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_0));
                acc1_x_0 = _fma_65;
                float _fma_66;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_0));
                acc1_y_0 = _fma_66;
                float _fma_67;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_0));
                acc0_x_0 = _fma_67;
                float _fma_68;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_0));
                acc0_y_0 = _fma_68;
                float _fma_69;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_0));
                acc1_x_0 = _fma_69;
                float _fma_70;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_0));
                acc1_y_0 = _fma_70;
                float _fma_71;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_0));
                acc0_x_0 = _fma_71;
                float _fma_72;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_0));
                acc0_y_0 = _fma_72;
                float _fma_73;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_0));
                acc1_x_0 = _fma_73;
                float _fma_74;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_0));
                acc1_y_0 = _fma_74;
                float _add_75;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(acc0_x_0), "f"(acc1_x_0));
                float _add_76;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(acc0_y_0), "f"(acc1_y_0));
                float dot_0 = _add_75 + _add_76;
                float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 1);
                dot_0 = dot_0 + _shfl_xor_sync_77;
                float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 2);
                dot_0 = dot_0 + _shfl_xor_sync_78;
                float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 4);
                dot_0 = dot_0 + _shfl_xor_sync_79;
                float vn_1 = (v_regs[1] - dot_0) * beta_0;
                float _fma_80;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(k_regs[0]), "f"(vn_1), "f"(S[16]));
                S[16] = _fma_80;
                float _fma_81;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(k_regs[1]), "f"(vn_1), "f"(S[17]));
                S[17] = _fma_81;
                float _fma_82;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(k_regs[2]), "f"(vn_1), "f"(S[18]));
                S[18] = _fma_82;
                float _fma_83;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(k_regs[3]), "f"(vn_1), "f"(S[19]));
                S[19] = _fma_83;
                float _fma_84;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(k_regs[4]), "f"(vn_1), "f"(S[20]));
                S[20] = _fma_84;
                float _fma_85;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(k_regs[5]), "f"(vn_1), "f"(S[21]));
                S[21] = _fma_85;
                float _fma_86;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(k_regs[6]), "f"(vn_1), "f"(S[22]));
                S[22] = _fma_86;
                float _fma_87;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(k_regs[7]), "f"(vn_1), "f"(S[23]));
                S[23] = _fma_87;
                float _fma_88;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(k_regs[8]), "f"(vn_1), "f"(S[24]));
                S[24] = _fma_88;
                float _fma_89;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(k_regs[9]), "f"(vn_1), "f"(S[25]));
                S[25] = _fma_89;
                float _fma_90;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(k_regs[10]), "f"(vn_1), "f"(S[26]));
                S[26] = _fma_90;
                float _fma_91;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(k_regs[11]), "f"(vn_1), "f"(S[27]));
                S[27] = _fma_91;
                float _fma_92;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(k_regs[12]), "f"(vn_1), "f"(S[28]));
                S[28] = _fma_92;
                float _fma_93;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(k_regs[13]), "f"(vn_1), "f"(S[29]));
                S[29] = _fma_93;
                float _fma_94;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(k_regs[14]), "f"(vn_1), "f"(S[30]));
                S[30] = _fma_94;
                float _fma_95;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(k_regs[15]), "f"(vn_1), "f"(S[31]));
                S[31] = _fma_95;
                unsigned int qbase = sQ_addr + (unsigned int)(t_1 * 512) + lane_byte;
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
                float acc0_x_1 = 0.0f;
                float acc0_y_1 = 0.0f;
                float acc1_x_1 = 0.0f;
                float acc1_y_1 = 0.0f;
                float _fma_96;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_1));
                acc0_x_1 = _fma_96;
                float _fma_97;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_1));
                acc0_y_1 = _fma_97;
                float _fma_98;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_1));
                acc1_x_1 = _fma_98;
                float _fma_99;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_1));
                acc1_y_1 = _fma_99;
                float _fma_100;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_1));
                acc0_x_1 = _fma_100;
                float _fma_101;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_1));
                acc0_y_1 = _fma_101;
                float _fma_102;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_1));
                acc1_x_1 = _fma_102;
                float _fma_103;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_1));
                acc1_y_1 = _fma_103;
                float _fma_104;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_1));
                acc0_x_1 = _fma_104;
                float _fma_105;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_1));
                acc0_y_1 = _fma_105;
                float _fma_106;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_1));
                acc1_x_1 = _fma_106;
                float _fma_107;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_1));
                acc1_y_1 = _fma_107;
                float _fma_108;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_1));
                acc0_x_1 = _fma_108;
                float _fma_109;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_1));
                acc0_y_1 = _fma_109;
                float _fma_110;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_1));
                acc1_x_1 = _fma_110;
                float _fma_111;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_111) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_1));
                acc1_y_1 = _fma_111;
                float _add_112;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(acc0_x_1), "f"(acc1_x_1));
                float _add_113;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(acc0_y_1), "f"(acc1_y_1));
                float dot_1 = _add_112 + _add_113;
                float _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_114;
                float _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_115;
                float _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
                dot_1 = dot_1 + _shfl_xor_sync_116;
                o4[0] = dot_1;
                float acc0_x_2 = 0.0f;
                float acc0_y_2 = 0.0f;
                float acc1_x_2 = 0.0f;
                float acc1_y_2 = 0.0f;
                float _fma_117;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_2));
                acc0_x_2 = _fma_117;
                float _fma_118;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_118) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_2));
                acc0_y_2 = _fma_118;
                float _fma_119;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_2));
                acc1_x_2 = _fma_119;
                float _fma_120;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_2));
                acc1_y_2 = _fma_120;
                float _fma_121;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_2));
                acc0_x_2 = _fma_121;
                float _fma_122;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_2));
                acc0_y_2 = _fma_122;
                float _fma_123;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_123) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_2));
                acc1_x_2 = _fma_123;
                float _fma_124;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_2));
                acc1_y_2 = _fma_124;
                float _fma_125;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_2));
                acc0_x_2 = _fma_125;
                float _fma_126;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_2));
                acc0_y_2 = _fma_126;
                float _fma_127;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_2));
                acc1_x_2 = _fma_127;
                float _fma_128;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_128) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_2));
                acc1_y_2 = _fma_128;
                float _fma_129;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_129) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_2));
                acc0_x_2 = _fma_129;
                float _fma_130;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_130) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_2));
                acc0_y_2 = _fma_130;
                float _fma_131;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_2));
                acc1_x_2 = _fma_131;
                float _fma_132;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_2));
                acc1_y_2 = _fma_132;
                float _add_133;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_133) : "f"(acc0_x_2), "f"(acc1_x_2));
                float _add_134;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_134) : "f"(acc0_y_2), "f"(acc1_y_2));
                float dot_2 = _add_133 + _add_134;
                float _shfl_xor_sync_135 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 1);
                dot_2 = dot_2 + _shfl_xor_sync_135;
                float _shfl_xor_sync_136 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 2);
                dot_2 = dot_2 + _shfl_xor_sync_136;
                float _shfl_xor_sync_137 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 4);
                dot_2 = dot_2 + _shfl_xor_sync_137;
                o4[1] = dot_2;
                if (u == 0) {
                    long long out_index = (long long)n * out_stride_p0 + (long long)t_1 * out_stride_p1 + (long long)h * out_stride_p2;
                    unsigned int packed[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o4[_lp*2 + 0], o4[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(out + (out_index + (long long)row0))[0] = reinterpret_cast<unsigned*>(packed)[0];
                }
                int ssm_slot = sSsm[t_1];
                int snap_valid = ((use_ssm_state_indices != 0 && ssm_slot >= 0 && state_pool_size > (long long)ssm_slot) ? 1 : 0);
                int safe_slot = ((snap_valid != 0) ? ssm_slot : 0);
                long long snap_base = (long long)safe_slot * state_stride_p0 + head_rows_off;
                if (philox_rounds != 0) {
                    bool nan = 0;
                    uint32_t _setp_138_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_138_bits) : "f"(S[0]), "f"(S[1]));
                    bool _setp_138 = _setp_138_bits != 0;
                    nan = nan || _setp_138;
                    uint32_t _setp_139_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_139_bits) : "f"(S[2]), "f"(S[3]));
                    bool _setp_139 = _setp_139_bits != 0;
                    nan = nan || _setp_139;
                    uint32_t _setp_140_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_140_bits) : "f"(S[4]), "f"(S[5]));
                    bool _setp_140 = _setp_140_bits != 0;
                    nan = nan || _setp_140;
                    uint32_t _setp_141_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_141_bits) : "f"(S[6]), "f"(S[7]));
                    bool _setp_141 = _setp_141_bits != 0;
                    nan = nan || _setp_141;
                    uint32_t _setp_142_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_142_bits) : "f"(S[8]), "f"(S[9]));
                    bool _setp_142 = _setp_142_bits != 0;
                    nan = nan || _setp_142;
                    uint32_t _setp_143_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_143_bits) : "f"(S[10]), "f"(S[11]));
                    bool _setp_143 = _setp_143_bits != 0;
                    nan = nan || _setp_143;
                    uint32_t _setp_144_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_144_bits) : "f"(S[12]), "f"(S[13]));
                    bool _setp_144 = _setp_144_bits != 0;
                    nan = nan || _setp_144;
                    uint32_t _setp_145_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_145_bits) : "f"(S[14]), "f"(S[15]));
                    bool _setp_145 = _setp_145_bits != 0;
                    nan = nan || _setp_145;
                    uint32_t _setp_146_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_146_bits) : "f"(S[16]), "f"(S[17]));
                    bool _setp_146 = _setp_146_bits != 0;
                    nan = nan || _setp_146;
                    uint32_t _setp_147_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_147_bits) : "f"(S[18]), "f"(S[19]));
                    bool _setp_147 = _setp_147_bits != 0;
                    nan = nan || _setp_147;
                    uint32_t _setp_148_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_148_bits) : "f"(S[20]), "f"(S[21]));
                    bool _setp_148 = _setp_148_bits != 0;
                    nan = nan || _setp_148;
                    uint32_t _setp_149_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_149_bits) : "f"(S[22]), "f"(S[23]));
                    bool _setp_149 = _setp_149_bits != 0;
                    nan = nan || _setp_149;
                    uint32_t _setp_150_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_150_bits) : "f"(S[24]), "f"(S[25]));
                    bool _setp_150 = _setp_150_bits != 0;
                    nan = nan || _setp_150;
                    uint32_t _setp_151_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_151_bits) : "f"(S[26]), "f"(S[27]));
                    bool _setp_151 = _setp_151_bits != 0;
                    nan = nan || _setp_151;
                    uint32_t _setp_152_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_152_bits) : "f"(S[28]), "f"(S[29]));
                    bool _setp_152 = _setp_152_bits != 0;
                    nan = nan || _setp_152;
                    uint32_t _setp_153_bits;
                    asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_153_bits) : "f"(S[30]), "f"(S[31]));
                    bool _setp_153 = _setp_153_bits != 0;
                    nan = nan || _setp_153;
                    int _warp_vote_154 = __any_sync(__activemask(), nan);
                    if (__builtin_expect(_warp_vote_154 == 0, 1)) {
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
                        float _abs_155;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_155) : "f"(S[0]));
                        unsigned int s = __as_u32(mul_f32_rz_noftz(_abs_155, __uint_as_float(125829120)));
                        unsigned int _min_156 = ((s) < (260046848) ? (s) : (260046848));
                        unsigned int h_0 = _min_156 + (h_noise[0] & 8191) >> 13;
                        unsigned int mag = ((s <= 2139095040) ? h_0 : 32256);
                        float _abs_157;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_157) : "f"(S[1]));
                        unsigned int s_0 = __as_u32(mul_f32_rz_noftz(_abs_157, __uint_as_float(125829120)));
                        unsigned int _min_158 = ((s_0) < (260046848) ? (s_0) : (260046848));
                        unsigned int h_1 = _min_158 + ((h_noise[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_0 = ((s_0 <= 2139095040) ? h_1 : 32256);
                        uint32_t mags;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags) : "r"(mag), "r"(mag_0));
                        uint32_t signs;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs) : "r"(__as_u32(S[0])), "r"(__as_u32(S[1])));
                        unsigned int word_15 = signs & 2147516416u | mags;
                        h_words[0] = word_15;
                        float _abs_159;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_159) : "f"(S[2]));
                        unsigned int s_1 = __as_u32(mul_f32_rz_noftz(_abs_159, __uint_as_float(125829120)));
                        unsigned int _min_160 = ((s_1) < (260046848) ? (s_1) : (260046848));
                        unsigned int h_2 = _min_160 + (h_noise[1] & 8191) >> 13;
                        unsigned int mag_1 = ((s_1 <= 2139095040) ? h_2 : 32256);
                        float _abs_161;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_161) : "f"(S[3]));
                        unsigned int s_2 = __as_u32(mul_f32_rz_noftz(_abs_161, __uint_as_float(125829120)));
                        unsigned int _min_162 = ((s_2) < (260046848) ? (s_2) : (260046848));
                        unsigned int h_3 = _min_162 + ((h_noise[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_2 = ((s_2 <= 2139095040) ? h_3 : 32256);
                        uint32_t mags_0;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_0) : "r"(mag_1), "r"(mag_2));
                        uint32_t signs_0;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_0) : "r"(__as_u32(S[2])), "r"(__as_u32(S[3])));
                        unsigned int word_15_1 = signs_0 & 2147516416u | mags_0;
                        h_words[1] = word_15_1;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        float _abs_163;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_163) : "f"(S[8]));
                        unsigned int s_3 = __as_u32(mul_f32_rz_noftz(_abs_163, __uint_as_float(125829120)));
                        unsigned int _min_164 = ((s_3) < (260046848) ? (s_3) : (260046848));
                        unsigned int h_4 = _min_164 + (h_noise[2] & 8191) >> 13;
                        unsigned int mag_3 = ((s_3 <= 2139095040) ? h_4 : 32256);
                        float _abs_165;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_165) : "f"(S[9]));
                        unsigned int s_4 = __as_u32(mul_f32_rz_noftz(_abs_165, __uint_as_float(125829120)));
                        unsigned int _min_166 = ((s_4) < (260046848) ? (s_4) : (260046848));
                        unsigned int h_5 = _min_166 + ((h_noise[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_4 = ((s_4 <= 2139095040) ? h_5 : 32256);
                        uint32_t mags_1;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_1) : "r"(mag_3), "r"(mag_4));
                        uint32_t signs_1;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_1) : "r"(__as_u32(S[8])), "r"(__as_u32(S[9])));
                        unsigned int word_16 = signs_1 & 2147516416u | mags_1;
                        h_words[0] = word_16;
                        float _abs_167;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_167) : "f"(S[10]));
                        unsigned int s_5 = __as_u32(mul_f32_rz_noftz(_abs_167, __uint_as_float(125829120)));
                        unsigned int _min_168 = ((s_5) < (260046848) ? (s_5) : (260046848));
                        unsigned int h_6 = _min_168 + (h_noise[3] & 8191) >> 13;
                        unsigned int mag_5 = ((s_5 <= 2139095040) ? h_6 : 32256);
                        float _abs_169;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_169) : "f"(S[11]));
                        unsigned int s_6 = __as_u32(mul_f32_rz_noftz(_abs_169, __uint_as_float(125829120)));
                        unsigned int _min_170 = ((s_6) < (260046848) ? (s_6) : (260046848));
                        unsigned int h_7 = _min_170 + ((h_noise[3] & 536805376) >> 16) >> 13;
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
                        float _abs_171;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_171) : "f"(S[4]));
                        unsigned int s_7 = __as_u32(mul_f32_rz_noftz(_abs_171, __uint_as_float(125829120)));
                        unsigned int _min_172 = ((s_7) < (260046848) ? (s_7) : (260046848));
                        unsigned int h_8 = _min_172 + (h_noise[0] & 8191) >> 13;
                        unsigned int mag_7 = ((s_7 <= 2139095040) ? h_8 : 32256);
                        float _abs_173;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_173) : "f"(S[5]));
                        unsigned int s_8 = __as_u32(mul_f32_rz_noftz(_abs_173, __uint_as_float(125829120)));
                        unsigned int _min_174 = ((s_8) < (260046848) ? (s_8) : (260046848));
                        unsigned int h_9 = _min_174 + ((h_noise[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_8 = ((s_8 <= 2139095040) ? h_9 : 32256);
                        uint32_t mags_3;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_3) : "r"(mag_7), "r"(mag_8));
                        uint32_t signs_3;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_3) : "r"(__as_u32(S[4])), "r"(__as_u32(S[5])));
                        unsigned int word_18 = signs_3 & 2147516416u | mags_3;
                        h_words[0] = word_18;
                        float _abs_175;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_175) : "f"(S[6]));
                        unsigned int s_9 = __as_u32(mul_f32_rz_noftz(_abs_175, __uint_as_float(125829120)));
                        unsigned int _min_176 = ((s_9) < (260046848) ? (s_9) : (260046848));
                        unsigned int h_10 = _min_176 + (h_noise[1] & 8191) >> 13;
                        unsigned int mag_9 = ((s_9 <= 2139095040) ? h_10 : 32256);
                        float _abs_177;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_177) : "f"(S[7]));
                        unsigned int s_10 = __as_u32(mul_f32_rz_noftz(_abs_177, __uint_as_float(125829120)));
                        unsigned int _min_178 = ((s_10) < (260046848) ? (s_10) : (260046848));
                        unsigned int h_11 = _min_178 + ((h_noise[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_10 = ((s_10 <= 2139095040) ? h_11 : 32256);
                        uint32_t mags_4;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_4) : "r"(mag_9), "r"(mag_10));
                        uint32_t signs_4;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_4) : "r"(__as_u32(S[6])), "r"(__as_u32(S[7])));
                        unsigned int word_19 = signs_4 & 2147516416u | mags_4;
                        h_words[1] = word_19;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        float _abs_179;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_179) : "f"(S[12]));
                        unsigned int s_11 = __as_u32(mul_f32_rz_noftz(_abs_179, __uint_as_float(125829120)));
                        unsigned int _min_180 = ((s_11) < (260046848) ? (s_11) : (260046848));
                        unsigned int h_12 = _min_180 + (h_noise[2] & 8191) >> 13;
                        unsigned int mag_11 = ((s_11 <= 2139095040) ? h_12 : 32256);
                        float _abs_181;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_181) : "f"(S[13]));
                        unsigned int s_12 = __as_u32(mul_f32_rz_noftz(_abs_181, __uint_as_float(125829120)));
                        unsigned int _min_182 = ((s_12) < (260046848) ? (s_12) : (260046848));
                        unsigned int h_13 = _min_182 + ((h_noise[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_12 = ((s_12 <= 2139095040) ? h_13 : 32256);
                        uint32_t mags_5;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_5) : "r"(mag_11), "r"(mag_12));
                        uint32_t signs_5;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_5) : "r"(__as_u32(S[12])), "r"(__as_u32(S[13])));
                        unsigned int word_20 = signs_5 & 2147516416u | mags_5;
                        h_words[0] = word_20;
                        float _abs_183;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_183) : "f"(S[14]));
                        unsigned int s_13 = __as_u32(mul_f32_rz_noftz(_abs_183, __uint_as_float(125829120)));
                        unsigned int _min_184 = ((s_13) < (260046848) ? (s_13) : (260046848));
                        unsigned int h_14 = _min_184 + (h_noise[3] & 8191) >> 13;
                        unsigned int mag_13 = ((s_13 <= 2139095040) ? h_14 : 32256);
                        float _abs_185;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_185) : "f"(S[15]));
                        unsigned int s_14 = __as_u32(mul_f32_rz_noftz(_abs_185, __uint_as_float(125829120)));
                        unsigned int _min_186 = ((s_14) < (260046848) ? (s_14) : (260046848));
                        unsigned int h_15 = _min_186 + ((h_noise[3] & 536805376) >> 16) >> 13;
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
                        float _abs_187;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_187) : "f"(S[16]));
                        unsigned int s_15 = __as_u32(mul_f32_rz_noftz(_abs_187, __uint_as_float(125829120)));
                        unsigned int _min_188 = ((s_15) < (260046848) ? (s_15) : (260046848));
                        unsigned int h_16 = _min_188 + (h_noise[0] & 8191) >> 13;
                        unsigned int mag_15 = ((s_15 <= 2139095040) ? h_16 : 32256);
                        float _abs_189;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_189) : "f"(S[17]));
                        unsigned int s_16 = __as_u32(mul_f32_rz_noftz(_abs_189, __uint_as_float(125829120)));
                        unsigned int _min_190 = ((s_16) < (260046848) ? (s_16) : (260046848));
                        unsigned int h_17 = _min_190 + ((h_noise[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_16 = ((s_16 <= 2139095040) ? h_17 : 32256);
                        uint32_t mags_7;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_7) : "r"(mag_15), "r"(mag_16));
                        uint32_t signs_7;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_7) : "r"(__as_u32(S[16])), "r"(__as_u32(S[17])));
                        unsigned int word_22 = signs_7 & 2147516416u | mags_7;
                        h_words[0] = word_22;
                        float _abs_191;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_191) : "f"(S[18]));
                        unsigned int s_17 = __as_u32(mul_f32_rz_noftz(_abs_191, __uint_as_float(125829120)));
                        unsigned int _min_192 = ((s_17) < (260046848) ? (s_17) : (260046848));
                        unsigned int h_18 = _min_192 + (h_noise[1] & 8191) >> 13;
                        unsigned int mag_17 = ((s_17 <= 2139095040) ? h_18 : 32256);
                        float _abs_193;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_193) : "f"(S[19]));
                        unsigned int s_18 = __as_u32(mul_f32_rz_noftz(_abs_193, __uint_as_float(125829120)));
                        unsigned int _min_194 = ((s_18) < (260046848) ? (s_18) : (260046848));
                        unsigned int h_19 = _min_194 + ((h_noise[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_18 = ((s_18 <= 2139095040) ? h_19 : 32256);
                        uint32_t mags_8;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_8) : "r"(mag_17), "r"(mag_18));
                        uint32_t signs_8;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_8) : "r"(__as_u32(S[18])), "r"(__as_u32(S[19])));
                        unsigned int word_23 = signs_8 & 2147516416u | mags_8;
                        h_words[1] = word_23;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        float _abs_195;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_195) : "f"(S[24]));
                        unsigned int s_19 = __as_u32(mul_f32_rz_noftz(_abs_195, __uint_as_float(125829120)));
                        unsigned int _min_196 = ((s_19) < (260046848) ? (s_19) : (260046848));
                        unsigned int h_20 = _min_196 + (h_noise[2] & 8191) >> 13;
                        unsigned int mag_19 = ((s_19 <= 2139095040) ? h_20 : 32256);
                        float _abs_197;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_197) : "f"(S[25]));
                        unsigned int s_20 = __as_u32(mul_f32_rz_noftz(_abs_197, __uint_as_float(125829120)));
                        unsigned int _min_198 = ((s_20) < (260046848) ? (s_20) : (260046848));
                        unsigned int h_21 = _min_198 + ((h_noise[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_20 = ((s_20 <= 2139095040) ? h_21 : 32256);
                        uint32_t mags_9;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_9) : "r"(mag_19), "r"(mag_20));
                        uint32_t signs_9;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_9) : "r"(__as_u32(S[24])), "r"(__as_u32(S[25])));
                        unsigned int word_24 = signs_9 & 2147516416u | mags_9;
                        h_words[0] = word_24;
                        float _abs_199;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_199) : "f"(S[26]));
                        unsigned int s_21 = __as_u32(mul_f32_rz_noftz(_abs_199, __uint_as_float(125829120)));
                        unsigned int _min_200 = ((s_21) < (260046848) ? (s_21) : (260046848));
                        unsigned int h_22 = _min_200 + (h_noise[3] & 8191) >> 13;
                        unsigned int mag_21 = ((s_21 <= 2139095040) ? h_22 : 32256);
                        float _abs_201;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_201) : "f"(S[27]));
                        unsigned int s_22 = __as_u32(mul_f32_rz_noftz(_abs_201, __uint_as_float(125829120)));
                        unsigned int _min_202 = ((s_22) < (260046848) ? (s_22) : (260046848));
                        unsigned int h_23 = _min_202 + ((h_noise[3] & 536805376) >> 16) >> 13;
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
                        float _abs_203;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_203) : "f"(S[20]));
                        unsigned int s_23 = __as_u32(mul_f32_rz_noftz(_abs_203, __uint_as_float(125829120)));
                        unsigned int _min_204 = ((s_23) < (260046848) ? (s_23) : (260046848));
                        unsigned int h_24 = _min_204 + (h_noise[0] & 8191) >> 13;
                        unsigned int mag_23 = ((s_23 <= 2139095040) ? h_24 : 32256);
                        float _abs_205;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_205) : "f"(S[21]));
                        unsigned int s_24 = __as_u32(mul_f32_rz_noftz(_abs_205, __uint_as_float(125829120)));
                        unsigned int _min_206 = ((s_24) < (260046848) ? (s_24) : (260046848));
                        unsigned int h_25 = _min_206 + ((h_noise[0] & 536805376) >> 16) >> 13;
                        unsigned int mag_24 = ((s_24 <= 2139095040) ? h_25 : 32256);
                        uint32_t mags_11;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_11) : "r"(mag_23), "r"(mag_24));
                        uint32_t signs_11;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_11) : "r"(__as_u32(S[20])), "r"(__as_u32(S[21])));
                        unsigned int word_26 = signs_11 & 2147516416u | mags_11;
                        h_words[0] = word_26;
                        float _abs_207;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_207) : "f"(S[22]));
                        unsigned int s_25 = __as_u32(mul_f32_rz_noftz(_abs_207, __uint_as_float(125829120)));
                        unsigned int _min_208 = ((s_25) < (260046848) ? (s_25) : (260046848));
                        unsigned int h_26 = _min_208 + (h_noise[1] & 8191) >> 13;
                        unsigned int mag_25 = ((s_25 <= 2139095040) ? h_26 : 32256);
                        float _abs_209;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_209) : "f"(S[23]));
                        unsigned int s_26 = __as_u32(mul_f32_rz_noftz(_abs_209, __uint_as_float(125829120)));
                        unsigned int _min_210 = ((s_26) < (260046848) ? (s_26) : (260046848));
                        unsigned int h_27 = _min_210 + ((h_noise[1] & 536805376) >> 16) >> 13;
                        unsigned int mag_26 = ((s_26 <= 2139095040) ? h_27 : 32256);
                        uint32_t mags_12;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_12) : "r"(mag_25), "r"(mag_26));
                        uint32_t signs_12;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_12) : "r"(__as_u32(S[22])), "r"(__as_u32(S[23])));
                        unsigned int word_27 = signs_12 & 2147516416u | mags_12;
                        h_words[1] = word_27;
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                        float _abs_211;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_211) : "f"(S[28]));
                        unsigned int s_27 = __as_u32(mul_f32_rz_noftz(_abs_211, __uint_as_float(125829120)));
                        unsigned int _min_212 = ((s_27) < (260046848) ? (s_27) : (260046848));
                        unsigned int h_28 = _min_212 + (h_noise[2] & 8191) >> 13;
                        unsigned int mag_27 = ((s_27 <= 2139095040) ? h_28 : 32256);
                        float _abs_213;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_213) : "f"(S[29]));
                        unsigned int s_28 = __as_u32(mul_f32_rz_noftz(_abs_213, __uint_as_float(125829120)));
                        unsigned int _min_214 = ((s_28) < (260046848) ? (s_28) : (260046848));
                        unsigned int h_29 = _min_214 + ((h_noise[2] & 536805376) >> 16) >> 13;
                        unsigned int mag_28 = ((s_28 <= 2139095040) ? h_29 : 32256);
                        uint32_t mags_13;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_13) : "r"(mag_27), "r"(mag_28));
                        uint32_t signs_13;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_13) : "r"(__as_u32(S[28])), "r"(__as_u32(S[29])));
                        unsigned int word_28 = signs_13 & 2147516416u | mags_13;
                        h_words[0] = word_28;
                        float _abs_215;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_215) : "f"(S[30]));
                        unsigned int s_29 = __as_u32(mul_f32_rz_noftz(_abs_215, __uint_as_float(125829120)));
                        unsigned int _min_216 = ((s_29) < (260046848) ? (s_29) : (260046848));
                        unsigned int h_30 = _min_216 + (h_noise[3] & 8191) >> 13;
                        unsigned int mag_29 = ((s_29 <= 2139095040) ? h_30 : 32256);
                        float _abs_217;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_217) : "f"(S[31]));
                        unsigned int s_30 = __as_u32(mul_f32_rz_noftz(_abs_217, __uint_as_float(125829120)));
                        unsigned int _min_218 = ((s_30) < (260046848) ? (s_30) : (260046848));
                        unsigned int h_31 = _min_218 + ((h_noise[3] & 536805376) >> 16) >> 13;
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
                    uint32_t _cvt_rn_f16x2_f32_219;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_219) : "f"(S[1]), "f"(S[0]));
                    h_words[0] = _cvt_rn_f16x2_f32_219;
                    uint32_t _cvt_rn_f16x2_f32_220;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_220) : "f"(S[3]), "f"(S[2]));
                    h_words[1] = _cvt_rn_f16x2_f32_220;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_221;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_221) : "f"(S[5]), "f"(S[4]));
                    h_words[0] = _cvt_rn_f16x2_f32_221;
                    uint32_t _cvt_rn_f16x2_f32_222;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_222) : "f"(S[7]), "f"(S[6]));
                    h_words[1] = _cvt_rn_f16x2_f32_222;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_223;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_223) : "f"(S[9]), "f"(S[8]));
                    h_words[0] = _cvt_rn_f16x2_f32_223;
                    uint32_t _cvt_rn_f16x2_f32_224;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_224) : "f"(S[11]), "f"(S[10]));
                    h_words[1] = _cvt_rn_f16x2_f32_224;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_225;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_225) : "f"(S[13]), "f"(S[12]));
                    h_words[0] = _cvt_rn_f16x2_f32_225;
                    uint32_t _cvt_rn_f16x2_f32_226;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_226) : "f"(S[15]), "f"(S[14]));
                    h_words[1] = _cvt_rn_f16x2_f32_226;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_227;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_227) : "f"(S[17]), "f"(S[16]));
                    h_words[0] = _cvt_rn_f16x2_f32_227;
                    uint32_t _cvt_rn_f16x2_f32_228;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_228) : "f"(S[19]), "f"(S[18]));
                    h_words[1] = _cvt_rn_f16x2_f32_228;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_229;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_229) : "f"(S[21]), "f"(S[20]));
                    h_words[0] = _cvt_rn_f16x2_f32_229;
                    uint32_t _cvt_rn_f16x2_f32_230;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_230) : "f"(S[23]), "f"(S[22]));
                    h_words[1] = _cvt_rn_f16x2_f32_230;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_231;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_231) : "f"(S[25]), "f"(S[24]));
                    h_words[0] = _cvt_rn_f16x2_f32_231;
                    uint32_t _cvt_rn_f16x2_f32_232;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_232) : "f"(S[27]), "f"(S[26]));
                    h_words[1] = _cvt_rn_f16x2_f32_232;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_233;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_233) : "f"(S[29]), "f"(S[28]));
                    h_words[0] = _cvt_rn_f16x2_f32_233;
                    uint32_t _cvt_rn_f16x2_f32_234;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_234) : "f"(S[31]), "f"(S[30]));
                    h_words[1] = _cvt_rn_f16x2_f32_234;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                }
            }
        }
        if (use_ssm_state_indices == 0) {
            #pragma unroll 1
            for (int t_2 = 0; t_2 < num_tokens; t_2++) {
                float alpha_1 = sAlpha[t_2];
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
                int m_1 = t_2;
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
                float acc0_x_3 = 0.0f;
                float acc0_y_3 = 0.0f;
                float acc1_x_3 = 0.0f;
                float acc1_y_3 = 0.0f;
                float _fma_235;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_235) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_3));
                acc0_x_3 = _fma_235;
                float _fma_236;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_236) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_3));
                acc0_y_3 = _fma_236;
                float _fma_237;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_237) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_3));
                acc1_x_3 = _fma_237;
                float _fma_238;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_238) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_3));
                acc1_y_3 = _fma_238;
                float _fma_239;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_239) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_3));
                acc0_x_3 = _fma_239;
                float _fma_240;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_240) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_3));
                acc0_y_3 = _fma_240;
                float _fma_241;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_241) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_3));
                acc1_x_3 = _fma_241;
                float _fma_242;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_242) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_3));
                acc1_y_3 = _fma_242;
                float _fma_243;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_243) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_3));
                acc0_x_3 = _fma_243;
                float _fma_244;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_244) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_3));
                acc0_y_3 = _fma_244;
                float _fma_245;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_245) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_3));
                acc1_x_3 = _fma_245;
                float _fma_246;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_246) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_3));
                acc1_y_3 = _fma_246;
                float _fma_247;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_247) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_3));
                acc0_x_3 = _fma_247;
                float _fma_248;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_248) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_3));
                acc0_y_3 = _fma_248;
                float _fma_249;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_249) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_3));
                acc1_x_3 = _fma_249;
                float _fma_250;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_250) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_3));
                acc1_y_3 = _fma_250;
                float _add_251;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_251) : "f"(acc0_x_3), "f"(acc1_x_3));
                float _add_252;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_252) : "f"(acc0_y_3), "f"(acc1_y_3));
                float dot_3 = _add_251 + _add_252;
                float _shfl_xor_sync_253 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 1);
                dot_3 = dot_3 + _shfl_xor_sync_253;
                float _shfl_xor_sync_254 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 2);
                dot_3 = dot_3 + _shfl_xor_sync_254;
                float _shfl_xor_sync_255 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 4);
                dot_3 = dot_3 + _shfl_xor_sync_255;
                float vn_2 = (v_regs[0] - dot_3) * beta_1;
                float _fma_256;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_256) : "f"(k_regs[0]), "f"(vn_2), "f"(S[0]));
                S[0] = _fma_256;
                float _fma_257;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_257) : "f"(k_regs[1]), "f"(vn_2), "f"(S[1]));
                S[1] = _fma_257;
                float _fma_258;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_258) : "f"(k_regs[2]), "f"(vn_2), "f"(S[2]));
                S[2] = _fma_258;
                float _fma_259;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_259) : "f"(k_regs[3]), "f"(vn_2), "f"(S[3]));
                S[3] = _fma_259;
                float _fma_260;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_260) : "f"(k_regs[4]), "f"(vn_2), "f"(S[4]));
                S[4] = _fma_260;
                float _fma_261;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_261) : "f"(k_regs[5]), "f"(vn_2), "f"(S[5]));
                S[5] = _fma_261;
                float _fma_262;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_262) : "f"(k_regs[6]), "f"(vn_2), "f"(S[6]));
                S[6] = _fma_262;
                float _fma_263;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_263) : "f"(k_regs[7]), "f"(vn_2), "f"(S[7]));
                S[7] = _fma_263;
                float _fma_264;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_264) : "f"(k_regs[8]), "f"(vn_2), "f"(S[8]));
                S[8] = _fma_264;
                float _fma_265;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_265) : "f"(k_regs[9]), "f"(vn_2), "f"(S[9]));
                S[9] = _fma_265;
                float _fma_266;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_266) : "f"(k_regs[10]), "f"(vn_2), "f"(S[10]));
                S[10] = _fma_266;
                float _fma_267;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_267) : "f"(k_regs[11]), "f"(vn_2), "f"(S[11]));
                S[11] = _fma_267;
                float _fma_268;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_268) : "f"(k_regs[12]), "f"(vn_2), "f"(S[12]));
                S[12] = _fma_268;
                float _fma_269;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_269) : "f"(k_regs[13]), "f"(vn_2), "f"(S[13]));
                S[13] = _fma_269;
                float _fma_270;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_270) : "f"(k_regs[14]), "f"(vn_2), "f"(S[14]));
                S[14] = _fma_270;
                float _fma_271;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_271) : "f"(k_regs[15]), "f"(vn_2), "f"(S[15]));
                S[15] = _fma_271;
                float acc0_x_3_1 = 0.0f;
                float acc0_y_3_1 = 0.0f;
                float acc1_x_3_1 = 0.0f;
                float acc1_y_3_1 = 0.0f;
                float _fma_272;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_272) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_3_1));
                acc0_x_3_1 = _fma_272;
                float _fma_273;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_273) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_3_1));
                acc0_y_3_1 = _fma_273;
                float _fma_274;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_274) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_3_1));
                acc1_x_3_1 = _fma_274;
                float _fma_275;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_275) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_3_1));
                acc1_y_3_1 = _fma_275;
                float _fma_276;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_276) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_3_1));
                acc0_x_3_1 = _fma_276;
                float _fma_277;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_277) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_3_1));
                acc0_y_3_1 = _fma_277;
                float _fma_278;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_278) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_3_1));
                acc1_x_3_1 = _fma_278;
                float _fma_279;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_279) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_3_1));
                acc1_y_3_1 = _fma_279;
                float _fma_280;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_280) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_3_1));
                acc0_x_3_1 = _fma_280;
                float _fma_281;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_281) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_3_1));
                acc0_y_3_1 = _fma_281;
                float _fma_282;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_282) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_3_1));
                acc1_x_3_1 = _fma_282;
                float _fma_283;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_283) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_3_1));
                acc1_y_3_1 = _fma_283;
                float _fma_284;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_284) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_3_1));
                acc0_x_3_1 = _fma_284;
                float _fma_285;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_285) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_3_1));
                acc0_y_3_1 = _fma_285;
                float _fma_286;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_286) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_3_1));
                acc1_x_3_1 = _fma_286;
                float _fma_287;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_287) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_3_1));
                acc1_y_3_1 = _fma_287;
                float _add_288;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_288) : "f"(acc0_x_3_1), "f"(acc1_x_3_1));
                float _add_289;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_289) : "f"(acc0_y_3_1), "f"(acc1_y_3_1));
                float dot_3_1 = _add_288 + _add_289;
                float _shfl_xor_sync_290 = __shfl_xor_sync(0xFFFFFFFF, dot_3_1, 1);
                dot_3_1 = dot_3_1 + _shfl_xor_sync_290;
                float _shfl_xor_sync_291 = __shfl_xor_sync(0xFFFFFFFF, dot_3_1, 2);
                dot_3_1 = dot_3_1 + _shfl_xor_sync_291;
                float _shfl_xor_sync_292 = __shfl_xor_sync(0xFFFFFFFF, dot_3_1, 4);
                dot_3_1 = dot_3_1 + _shfl_xor_sync_292;
                float vn_3 = (v_regs[1] - dot_3_1) * beta_1;
                float _fma_293;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_293) : "f"(k_regs[0]), "f"(vn_3), "f"(S[16]));
                S[16] = _fma_293;
                float _fma_294;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_294) : "f"(k_regs[1]), "f"(vn_3), "f"(S[17]));
                S[17] = _fma_294;
                float _fma_295;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_295) : "f"(k_regs[2]), "f"(vn_3), "f"(S[18]));
                S[18] = _fma_295;
                float _fma_296;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_296) : "f"(k_regs[3]), "f"(vn_3), "f"(S[19]));
                S[19] = _fma_296;
                float _fma_297;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_297) : "f"(k_regs[4]), "f"(vn_3), "f"(S[20]));
                S[20] = _fma_297;
                float _fma_298;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_298) : "f"(k_regs[5]), "f"(vn_3), "f"(S[21]));
                S[21] = _fma_298;
                float _fma_299;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_299) : "f"(k_regs[6]), "f"(vn_3), "f"(S[22]));
                S[22] = _fma_299;
                float _fma_300;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_300) : "f"(k_regs[7]), "f"(vn_3), "f"(S[23]));
                S[23] = _fma_300;
                float _fma_301;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_301) : "f"(k_regs[8]), "f"(vn_3), "f"(S[24]));
                S[24] = _fma_301;
                float _fma_302;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_302) : "f"(k_regs[9]), "f"(vn_3), "f"(S[25]));
                S[25] = _fma_302;
                float _fma_303;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_303) : "f"(k_regs[10]), "f"(vn_3), "f"(S[26]));
                S[26] = _fma_303;
                float _fma_304;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_304) : "f"(k_regs[11]), "f"(vn_3), "f"(S[27]));
                S[27] = _fma_304;
                float _fma_305;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_305) : "f"(k_regs[12]), "f"(vn_3), "f"(S[28]));
                S[28] = _fma_305;
                float _fma_306;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_306) : "f"(k_regs[13]), "f"(vn_3), "f"(S[29]));
                S[29] = _fma_306;
                float _fma_307;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_307) : "f"(k_regs[14]), "f"(vn_3), "f"(S[30]));
                S[30] = _fma_307;
                float _fma_308;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_308) : "f"(k_regs[15]), "f"(vn_3), "f"(S[31]));
                S[31] = _fma_308;
                unsigned int qbase_1 = sQ_addr + (unsigned int)(t_2 * 512) + lane_byte;
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
                float acc0_x_4 = 0.0f;
                float acc0_y_4 = 0.0f;
                float acc1_x_4 = 0.0f;
                float acc1_y_4 = 0.0f;
                float _fma_309;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_309) : "f"(S[0]), "f"(k_regs[0]), "f"(acc0_x_4));
                acc0_x_4 = _fma_309;
                float _fma_310;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_310) : "f"(S[1]), "f"(k_regs[1]), "f"(acc0_y_4));
                acc0_y_4 = _fma_310;
                float _fma_311;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_311) : "f"(S[2]), "f"(k_regs[2]), "f"(acc1_x_4));
                acc1_x_4 = _fma_311;
                float _fma_312;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_312) : "f"(S[3]), "f"(k_regs[3]), "f"(acc1_y_4));
                acc1_y_4 = _fma_312;
                float _fma_313;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_313) : "f"(S[4]), "f"(k_regs[4]), "f"(acc0_x_4));
                acc0_x_4 = _fma_313;
                float _fma_314;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_314) : "f"(S[5]), "f"(k_regs[5]), "f"(acc0_y_4));
                acc0_y_4 = _fma_314;
                float _fma_315;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_315) : "f"(S[6]), "f"(k_regs[6]), "f"(acc1_x_4));
                acc1_x_4 = _fma_315;
                float _fma_316;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_316) : "f"(S[7]), "f"(k_regs[7]), "f"(acc1_y_4));
                acc1_y_4 = _fma_316;
                float _fma_317;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_317) : "f"(S[8]), "f"(k_regs[8]), "f"(acc0_x_4));
                acc0_x_4 = _fma_317;
                float _fma_318;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_318) : "f"(S[9]), "f"(k_regs[9]), "f"(acc0_y_4));
                acc0_y_4 = _fma_318;
                float _fma_319;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_319) : "f"(S[10]), "f"(k_regs[10]), "f"(acc1_x_4));
                acc1_x_4 = _fma_319;
                float _fma_320;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_320) : "f"(S[11]), "f"(k_regs[11]), "f"(acc1_y_4));
                acc1_y_4 = _fma_320;
                float _fma_321;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_321) : "f"(S[12]), "f"(k_regs[12]), "f"(acc0_x_4));
                acc0_x_4 = _fma_321;
                float _fma_322;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_322) : "f"(S[13]), "f"(k_regs[13]), "f"(acc0_y_4));
                acc0_y_4 = _fma_322;
                float _fma_323;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_323) : "f"(S[14]), "f"(k_regs[14]), "f"(acc1_x_4));
                acc1_x_4 = _fma_323;
                float _fma_324;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_324) : "f"(S[15]), "f"(k_regs[15]), "f"(acc1_y_4));
                acc1_y_4 = _fma_324;
                float _add_325;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_325) : "f"(acc0_x_4), "f"(acc1_x_4));
                float _add_326;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_326) : "f"(acc0_y_4), "f"(acc1_y_4));
                float dot_4 = _add_325 + _add_326;
                float _shfl_xor_sync_327 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 1);
                dot_4 = dot_4 + _shfl_xor_sync_327;
                float _shfl_xor_sync_328 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 2);
                dot_4 = dot_4 + _shfl_xor_sync_328;
                float _shfl_xor_sync_329 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 4);
                dot_4 = dot_4 + _shfl_xor_sync_329;
                o4[0] = dot_4;
                float acc0_x_5 = 0.0f;
                float acc0_y_5 = 0.0f;
                float acc1_x_5 = 0.0f;
                float acc1_y_5 = 0.0f;
                float _fma_330;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_330) : "f"(S[16]), "f"(k_regs[0]), "f"(acc0_x_5));
                acc0_x_5 = _fma_330;
                float _fma_331;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_331) : "f"(S[17]), "f"(k_regs[1]), "f"(acc0_y_5));
                acc0_y_5 = _fma_331;
                float _fma_332;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_332) : "f"(S[18]), "f"(k_regs[2]), "f"(acc1_x_5));
                acc1_x_5 = _fma_332;
                float _fma_333;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_333) : "f"(S[19]), "f"(k_regs[3]), "f"(acc1_y_5));
                acc1_y_5 = _fma_333;
                float _fma_334;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_334) : "f"(S[20]), "f"(k_regs[4]), "f"(acc0_x_5));
                acc0_x_5 = _fma_334;
                float _fma_335;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_335) : "f"(S[21]), "f"(k_regs[5]), "f"(acc0_y_5));
                acc0_y_5 = _fma_335;
                float _fma_336;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_336) : "f"(S[22]), "f"(k_regs[6]), "f"(acc1_x_5));
                acc1_x_5 = _fma_336;
                float _fma_337;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_337) : "f"(S[23]), "f"(k_regs[7]), "f"(acc1_y_5));
                acc1_y_5 = _fma_337;
                float _fma_338;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_338) : "f"(S[24]), "f"(k_regs[8]), "f"(acc0_x_5));
                acc0_x_5 = _fma_338;
                float _fma_339;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_339) : "f"(S[25]), "f"(k_regs[9]), "f"(acc0_y_5));
                acc0_y_5 = _fma_339;
                float _fma_340;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_340) : "f"(S[26]), "f"(k_regs[10]), "f"(acc1_x_5));
                acc1_x_5 = _fma_340;
                float _fma_341;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_341) : "f"(S[27]), "f"(k_regs[11]), "f"(acc1_y_5));
                acc1_y_5 = _fma_341;
                float _fma_342;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_342) : "f"(S[28]), "f"(k_regs[12]), "f"(acc0_x_5));
                acc0_x_5 = _fma_342;
                float _fma_343;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_343) : "f"(S[29]), "f"(k_regs[13]), "f"(acc0_y_5));
                acc0_y_5 = _fma_343;
                float _fma_344;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_344) : "f"(S[30]), "f"(k_regs[14]), "f"(acc1_x_5));
                acc1_x_5 = _fma_344;
                float _fma_345;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_345) : "f"(S[31]), "f"(k_regs[15]), "f"(acc1_y_5));
                acc1_y_5 = _fma_345;
                float _add_346;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_346) : "f"(acc0_x_5), "f"(acc1_x_5));
                float _add_347;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_347) : "f"(acc0_y_5), "f"(acc1_y_5));
                float dot_5 = _add_346 + _add_347;
                float _shfl_xor_sync_348 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 1);
                dot_5 = dot_5 + _shfl_xor_sync_348;
                float _shfl_xor_sync_349 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 2);
                dot_5 = dot_5 + _shfl_xor_sync_349;
                float _shfl_xor_sync_350 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 4);
                dot_5 = dot_5 + _shfl_xor_sync_350;
                o4[1] = dot_5;
                if (u == 0) {
                    long long out_index_1 = (long long)n * out_stride_p0 + (long long)t_2 * out_stride_p1 + (long long)h * out_stride_p2;
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
                    uint32_t _read_special_reg_351;
                    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(_read_special_reg_351));
                    int tid_r = (int)_read_special_reg_351;
                    long long noise_at = read_slot_head + (long long)(tid_r / 8 * 2) * state_stride_p2 + (long long)((tid_r - tid_r / 8 * 8) * 4);
                    if (philox_rounds != 0) {
                        bool nan_1 = 0;
                        uint32_t _setp_352_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_352_bits) : "f"(S[0]), "f"(S[1]));
                        bool _setp_352 = _setp_352_bits != 0;
                        nan_1 = nan_1 || _setp_352;
                        uint32_t _setp_353_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_353_bits) : "f"(S[2]), "f"(S[3]));
                        bool _setp_353 = _setp_353_bits != 0;
                        nan_1 = nan_1 || _setp_353;
                        uint32_t _setp_354_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_354_bits) : "f"(S[4]), "f"(S[5]));
                        bool _setp_354 = _setp_354_bits != 0;
                        nan_1 = nan_1 || _setp_354;
                        uint32_t _setp_355_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_355_bits) : "f"(S[6]), "f"(S[7]));
                        bool _setp_355 = _setp_355_bits != 0;
                        nan_1 = nan_1 || _setp_355;
                        uint32_t _setp_356_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_356_bits) : "f"(S[8]), "f"(S[9]));
                        bool _setp_356 = _setp_356_bits != 0;
                        nan_1 = nan_1 || _setp_356;
                        uint32_t _setp_357_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_357_bits) : "f"(S[10]), "f"(S[11]));
                        bool _setp_357 = _setp_357_bits != 0;
                        nan_1 = nan_1 || _setp_357;
                        uint32_t _setp_358_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_358_bits) : "f"(S[12]), "f"(S[13]));
                        bool _setp_358 = _setp_358_bits != 0;
                        nan_1 = nan_1 || _setp_358;
                        uint32_t _setp_359_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_359_bits) : "f"(S[14]), "f"(S[15]));
                        bool _setp_359 = _setp_359_bits != 0;
                        nan_1 = nan_1 || _setp_359;
                        uint32_t _setp_360_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_360_bits) : "f"(S[16]), "f"(S[17]));
                        bool _setp_360 = _setp_360_bits != 0;
                        nan_1 = nan_1 || _setp_360;
                        uint32_t _setp_361_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_361_bits) : "f"(S[18]), "f"(S[19]));
                        bool _setp_361 = _setp_361_bits != 0;
                        nan_1 = nan_1 || _setp_361;
                        uint32_t _setp_362_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_362_bits) : "f"(S[20]), "f"(S[21]));
                        bool _setp_362 = _setp_362_bits != 0;
                        nan_1 = nan_1 || _setp_362;
                        uint32_t _setp_363_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_363_bits) : "f"(S[22]), "f"(S[23]));
                        bool _setp_363 = _setp_363_bits != 0;
                        nan_1 = nan_1 || _setp_363;
                        uint32_t _setp_364_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_364_bits) : "f"(S[24]), "f"(S[25]));
                        bool _setp_364 = _setp_364_bits != 0;
                        nan_1 = nan_1 || _setp_364;
                        uint32_t _setp_365_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_365_bits) : "f"(S[26]), "f"(S[27]));
                        bool _setp_365 = _setp_365_bits != 0;
                        nan_1 = nan_1 || _setp_365;
                        uint32_t _setp_366_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_366_bits) : "f"(S[28]), "f"(S[29]));
                        bool _setp_366 = _setp_366_bits != 0;
                        nan_1 = nan_1 || _setp_366;
                        uint32_t _setp_367_bits;
                        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }" : "=r"(_setp_367_bits) : "f"(S[30]), "f"(S[31]));
                        bool _setp_367 = _setp_367_bits != 0;
                        nan_1 = nan_1 || _setp_367;
                        int _warp_vote_368 = __any_sync(__activemask(), nan_1);
                        if (__builtin_expect(_warp_vote_368 == 0, 1)) {
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
                            float _abs_369;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_369) : "f"(S[0]));
                            unsigned int s_31 = __as_u32(mul_f32_rz_noftz(_abs_369, __uint_as_float(125829120)));
                            unsigned int _min_370 = ((s_31) < (260046848) ? (s_31) : (260046848));
                            unsigned int h_32 = _min_370 + (h_noise[0] & 8191) >> 13;
                            unsigned int mag_31 = ((s_31 <= 2139095040) ? h_32 : 32256);
                            float _abs_371;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_371) : "f"(S[1]));
                            unsigned int s_31_1 = __as_u32(mul_f32_rz_noftz(_abs_371, __uint_as_float(125829120)));
                            unsigned int _min_372 = ((s_31_1) < (260046848) ? (s_31_1) : (260046848));
                            unsigned int h_33 = _min_372 + ((h_noise[0] & 536805376) >> 16) >> 13;
                            unsigned int mag_31_1 = ((s_31_1 <= 2139095040) ? h_33 : 32256);
                            uint32_t mags_15;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_15) : "r"(mag_31), "r"(mag_31_1));
                            uint32_t signs_15;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_15) : "r"(__as_u32(S[0])), "r"(__as_u32(S[1])));
                            unsigned int word_45 = signs_15 & 2147516416u | mags_15;
                            h_words[0] = word_45;
                            float _abs_373;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_373) : "f"(S[2]));
                            unsigned int s_32 = __as_u32(mul_f32_rz_noftz(_abs_373, __uint_as_float(125829120)));
                            unsigned int _min_374 = ((s_32) < (260046848) ? (s_32) : (260046848));
                            unsigned int h_34 = _min_374 + (h_noise[1] & 8191) >> 13;
                            unsigned int mag_32 = ((s_32 <= 2139095040) ? h_34 : 32256);
                            float _abs_375;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_375) : "f"(S[3]));
                            unsigned int s_33 = __as_u32(mul_f32_rz_noftz(_abs_375, __uint_as_float(125829120)));
                            unsigned int _min_376 = ((s_33) < (260046848) ? (s_33) : (260046848));
                            unsigned int h_35 = _min_376 + ((h_noise[1] & 536805376) >> 16) >> 13;
                            unsigned int mag_33 = ((s_33 <= 2139095040) ? h_35 : 32256);
                            uint32_t mags_15_1;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_15_1) : "r"(mag_32), "r"(mag_33));
                            uint32_t signs_15_1;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_15_1) : "r"(__as_u32(S[2])), "r"(__as_u32(S[3])));
                            unsigned int word_45_1 = signs_15_1 & 2147516416u | mags_15_1;
                            h_words[1] = word_45_1;
                            reinterpret_cast<int2*>(state + (write_rows_base))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_377;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_377) : "f"(S[8]));
                            unsigned int s_34 = __as_u32(mul_f32_rz_noftz(_abs_377, __uint_as_float(125829120)));
                            unsigned int _min_378 = ((s_34) < (260046848) ? (s_34) : (260046848));
                            unsigned int h_36 = _min_378 + (h_noise[2] & 8191) >> 13;
                            unsigned int mag_34 = ((s_34 <= 2139095040) ? h_36 : 32256);
                            float _abs_379;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_379) : "f"(S[9]));
                            unsigned int s_35 = __as_u32(mul_f32_rz_noftz(_abs_379, __uint_as_float(125829120)));
                            unsigned int _min_380 = ((s_35) < (260046848) ? (s_35) : (260046848));
                            unsigned int h_37 = _min_380 + ((h_noise[2] & 536805376) >> 16) >> 13;
                            unsigned int mag_35 = ((s_35 <= 2139095040) ? h_37 : 32256);
                            uint32_t mags_16;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_16) : "r"(mag_34), "r"(mag_35));
                            uint32_t signs_16;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_16) : "r"(__as_u32(S[8])), "r"(__as_u32(S[9])));
                            unsigned int word_46 = signs_16 & 2147516416u | mags_16;
                            h_words[0] = word_46;
                            float _abs_381;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_381) : "f"(S[10]));
                            unsigned int s_36 = __as_u32(mul_f32_rz_noftz(_abs_381, __uint_as_float(125829120)));
                            unsigned int _min_382 = ((s_36) < (260046848) ? (s_36) : (260046848));
                            unsigned int h_38 = _min_382 + (h_noise[3] & 8191) >> 13;
                            unsigned int mag_36 = ((s_36 <= 2139095040) ? h_38 : 32256);
                            float _abs_383;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_383) : "f"(S[11]));
                            unsigned int s_37 = __as_u32(mul_f32_rz_noftz(_abs_383, __uint_as_float(125829120)));
                            unsigned int _min_384 = ((s_37) < (260046848) ? (s_37) : (260046848));
                            unsigned int h_39 = _min_384 + ((h_noise[3] & 536805376) >> 16) >> 13;
                            unsigned int mag_37 = ((s_37 <= 2139095040) ? h_39 : 32256);
                            uint32_t mags_17;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_17) : "r"(mag_36), "r"(mag_37));
                            uint32_t signs_17;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_17) : "r"(__as_u32(S[10])), "r"(__as_u32(S[11])));
                            unsigned int word_47 = signs_17 & 2147516416u | mags_17;
                            h_words[1] = word_47;
                            reinterpret_cast<int2*>(state + (write_rows_base + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_385;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_385) : "f"(S[4]));
                            unsigned int s_38 = __as_u32(mul_f32_rz_noftz(_abs_385, __uint_as_float(125829120)));
                            unsigned int _min_386 = ((s_38) < (260046848) ? (s_38) : (260046848));
                            unsigned int h_40 = _min_386 + (h_noise[4] & 8191) >> 13;
                            unsigned int mag_38 = ((s_38 <= 2139095040) ? h_40 : 32256);
                            float _abs_387;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_387) : "f"(S[5]));
                            unsigned int s_39 = __as_u32(mul_f32_rz_noftz(_abs_387, __uint_as_float(125829120)));
                            unsigned int _min_388 = ((s_39) < (260046848) ? (s_39) : (260046848));
                            unsigned int h_41 = _min_388 + ((h_noise[4] & 536805376) >> 16) >> 13;
                            unsigned int mag_39 = ((s_39 <= 2139095040) ? h_41 : 32256);
                            uint32_t mags_18;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_18) : "r"(mag_38), "r"(mag_39));
                            uint32_t signs_18;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_18) : "r"(__as_u32(S[4])), "r"(__as_u32(S[5])));
                            unsigned int word_48 = signs_18 & 2147516416u | mags_18;
                            h_words[0] = word_48;
                            float _abs_389;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_389) : "f"(S[6]));
                            unsigned int s_40 = __as_u32(mul_f32_rz_noftz(_abs_389, __uint_as_float(125829120)));
                            unsigned int _min_390 = ((s_40) < (260046848) ? (s_40) : (260046848));
                            unsigned int h_42 = _min_390 + (h_noise[5] & 8191) >> 13;
                            unsigned int mag_40 = ((s_40 <= 2139095040) ? h_42 : 32256);
                            float _abs_391;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_391) : "f"(S[7]));
                            unsigned int s_41 = __as_u32(mul_f32_rz_noftz(_abs_391, __uint_as_float(125829120)));
                            unsigned int _min_392 = ((s_41) < (260046848) ? (s_41) : (260046848));
                            unsigned int h_43 = _min_392 + ((h_noise[5] & 536805376) >> 16) >> 13;
                            unsigned int mag_41 = ((s_41 <= 2139095040) ? h_43 : 32256);
                            uint32_t mags_19;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_19) : "r"(mag_40), "r"(mag_41));
                            uint32_t signs_19;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_19) : "r"(__as_u32(S[6])), "r"(__as_u32(S[7])));
                            unsigned int word_49 = signs_19 & 2147516416u | mags_19;
                            h_words[1] = word_49;
                            reinterpret_cast<int2*>(state + (write_rows_base + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_393;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_393) : "f"(S[12]));
                            unsigned int s_42 = __as_u32(mul_f32_rz_noftz(_abs_393, __uint_as_float(125829120)));
                            unsigned int _min_394 = ((s_42) < (260046848) ? (s_42) : (260046848));
                            unsigned int h_44 = _min_394 + (h_noise[6] & 8191) >> 13;
                            unsigned int mag_42 = ((s_42 <= 2139095040) ? h_44 : 32256);
                            float _abs_395;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_395) : "f"(S[13]));
                            unsigned int s_43 = __as_u32(mul_f32_rz_noftz(_abs_395, __uint_as_float(125829120)));
                            unsigned int _min_396 = ((s_43) < (260046848) ? (s_43) : (260046848));
                            unsigned int h_45 = _min_396 + ((h_noise[6] & 536805376) >> 16) >> 13;
                            unsigned int mag_43 = ((s_43 <= 2139095040) ? h_45 : 32256);
                            uint32_t mags_20;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_20) : "r"(mag_42), "r"(mag_43));
                            uint32_t signs_20;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_20) : "r"(__as_u32(S[12])), "r"(__as_u32(S[13])));
                            unsigned int word_50 = signs_20 & 2147516416u | mags_20;
                            h_words[0] = word_50;
                            float _abs_397;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_397) : "f"(S[14]));
                            unsigned int s_44 = __as_u32(mul_f32_rz_noftz(_abs_397, __uint_as_float(125829120)));
                            unsigned int _min_398 = ((s_44) < (260046848) ? (s_44) : (260046848));
                            unsigned int h_46 = _min_398 + (h_noise[7] & 8191) >> 13;
                            unsigned int mag_44 = ((s_44 <= 2139095040) ? h_46 : 32256);
                            float _abs_399;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_399) : "f"(S[15]));
                            unsigned int s_45 = __as_u32(mul_f32_rz_noftz(_abs_399, __uint_as_float(125829120)));
                            unsigned int _min_400 = ((s_45) < (260046848) ? (s_45) : (260046848));
                            unsigned int h_47 = _min_400 + ((h_noise[7] & 536805376) >> 16) >> 13;
                            unsigned int mag_45 = ((s_45 <= 2139095040) ? h_47 : 32256);
                            uint32_t mags_21;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_21) : "r"(mag_44), "r"(mag_45));
                            uint32_t signs_21;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_21) : "r"(__as_u32(S[14])), "r"(__as_u32(S[15])));
                            unsigned int word_51 = signs_21 & 2147516416u | mags_21;
                            h_words[1] = word_51;
                            reinterpret_cast<int2*>(state + (write_rows_base + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_401;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_401) : "f"(S[16]));
                            unsigned int s_46 = __as_u32(mul_f32_rz_noftz(_abs_401, __uint_as_float(125829120)));
                            unsigned int _min_402 = ((s_46) < (260046848) ? (s_46) : (260046848));
                            unsigned int h_48 = _min_402 + (h_noise[8] & 8191) >> 13;
                            unsigned int mag_46 = ((s_46 <= 2139095040) ? h_48 : 32256);
                            float _abs_403;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_403) : "f"(S[17]));
                            unsigned int s_47 = __as_u32(mul_f32_rz_noftz(_abs_403, __uint_as_float(125829120)));
                            unsigned int _min_404 = ((s_47) < (260046848) ? (s_47) : (260046848));
                            unsigned int h_49 = _min_404 + ((h_noise[8] & 536805376) >> 16) >> 13;
                            unsigned int mag_47 = ((s_47 <= 2139095040) ? h_49 : 32256);
                            uint32_t mags_22;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_22) : "r"(mag_46), "r"(mag_47));
                            uint32_t signs_22;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_22) : "r"(__as_u32(S[16])), "r"(__as_u32(S[17])));
                            unsigned int word_52 = signs_22 & 2147516416u | mags_22;
                            h_words[0] = word_52;
                            float _abs_405;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_405) : "f"(S[18]));
                            unsigned int s_48 = __as_u32(mul_f32_rz_noftz(_abs_405, __uint_as_float(125829120)));
                            unsigned int _min_406 = ((s_48) < (260046848) ? (s_48) : (260046848));
                            unsigned int h_50 = _min_406 + (h_noise[9] & 8191) >> 13;
                            unsigned int mag_48 = ((s_48 <= 2139095040) ? h_50 : 32256);
                            float _abs_407;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_407) : "f"(S[19]));
                            unsigned int s_49 = __as_u32(mul_f32_rz_noftz(_abs_407, __uint_as_float(125829120)));
                            unsigned int _min_408 = ((s_49) < (260046848) ? (s_49) : (260046848));
                            unsigned int h_51 = _min_408 + ((h_noise[9] & 536805376) >> 16) >> 13;
                            unsigned int mag_49 = ((s_49 <= 2139095040) ? h_51 : 32256);
                            uint32_t mags_23;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_23) : "r"(mag_48), "r"(mag_49));
                            uint32_t signs_23;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_23) : "r"(__as_u32(S[18])), "r"(__as_u32(S[19])));
                            unsigned int word_53 = signs_23 & 2147516416u | mags_23;
                            h_words[1] = word_53;
                            reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_409;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_409) : "f"(S[24]));
                            unsigned int s_50 = __as_u32(mul_f32_rz_noftz(_abs_409, __uint_as_float(125829120)));
                            unsigned int _min_410 = ((s_50) < (260046848) ? (s_50) : (260046848));
                            unsigned int h_52 = _min_410 + (h_noise[10] & 8191) >> 13;
                            unsigned int mag_50 = ((s_50 <= 2139095040) ? h_52 : 32256);
                            float _abs_411;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_411) : "f"(S[25]));
                            unsigned int s_51 = __as_u32(mul_f32_rz_noftz(_abs_411, __uint_as_float(125829120)));
                            unsigned int _min_412 = ((s_51) < (260046848) ? (s_51) : (260046848));
                            unsigned int h_53 = _min_412 + ((h_noise[10] & 536805376) >> 16) >> 13;
                            unsigned int mag_51 = ((s_51 <= 2139095040) ? h_53 : 32256);
                            uint32_t mags_24;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_24) : "r"(mag_50), "r"(mag_51));
                            uint32_t signs_24;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_24) : "r"(__as_u32(S[24])), "r"(__as_u32(S[25])));
                            unsigned int word_54 = signs_24 & 2147516416u | mags_24;
                            h_words[0] = word_54;
                            float _abs_413;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_413) : "f"(S[26]));
                            unsigned int s_52 = __as_u32(mul_f32_rz_noftz(_abs_413, __uint_as_float(125829120)));
                            unsigned int _min_414 = ((s_52) < (260046848) ? (s_52) : (260046848));
                            unsigned int h_54 = _min_414 + (h_noise[11] & 8191) >> 13;
                            unsigned int mag_52 = ((s_52 <= 2139095040) ? h_54 : 32256);
                            float _abs_415;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_415) : "f"(S[27]));
                            unsigned int s_53 = __as_u32(mul_f32_rz_noftz(_abs_415, __uint_as_float(125829120)));
                            unsigned int _min_416 = ((s_53) < (260046848) ? (s_53) : (260046848));
                            unsigned int h_55 = _min_416 + ((h_noise[11] & 536805376) >> 16) >> 13;
                            unsigned int mag_53 = ((s_53 <= 2139095040) ? h_55 : 32256);
                            uint32_t mags_25;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_25) : "r"(mag_52), "r"(mag_53));
                            uint32_t signs_25;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_25) : "r"(__as_u32(S[26])), "r"(__as_u32(S[27])));
                            unsigned int word_55 = signs_25 & 2147516416u | mags_25;
                            h_words[1] = word_55;
                            reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_417;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_417) : "f"(S[20]));
                            unsigned int s_54 = __as_u32(mul_f32_rz_noftz(_abs_417, __uint_as_float(125829120)));
                            unsigned int _min_418 = ((s_54) < (260046848) ? (s_54) : (260046848));
                            unsigned int h_56 = _min_418 + (h_noise[12] & 8191) >> 13;
                            unsigned int mag_54 = ((s_54 <= 2139095040) ? h_56 : 32256);
                            float _abs_419;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_419) : "f"(S[21]));
                            unsigned int s_55 = __as_u32(mul_f32_rz_noftz(_abs_419, __uint_as_float(125829120)));
                            unsigned int _min_420 = ((s_55) < (260046848) ? (s_55) : (260046848));
                            unsigned int h_57 = _min_420 + ((h_noise[12] & 536805376) >> 16) >> 13;
                            unsigned int mag_55 = ((s_55 <= 2139095040) ? h_57 : 32256);
                            uint32_t mags_26;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_26) : "r"(mag_54), "r"(mag_55));
                            uint32_t signs_26;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_26) : "r"(__as_u32(S[20])), "r"(__as_u32(S[21])));
                            unsigned int word_56 = signs_26 & 2147516416u | mags_26;
                            h_words[0] = word_56;
                            float _abs_421;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_421) : "f"(S[22]));
                            unsigned int s_56 = __as_u32(mul_f32_rz_noftz(_abs_421, __uint_as_float(125829120)));
                            unsigned int _min_422 = ((s_56) < (260046848) ? (s_56) : (260046848));
                            unsigned int h_58 = _min_422 + (h_noise[13] & 8191) >> 13;
                            unsigned int mag_56 = ((s_56 <= 2139095040) ? h_58 : 32256);
                            float _abs_423;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_423) : "f"(S[23]));
                            unsigned int s_57 = __as_u32(mul_f32_rz_noftz(_abs_423, __uint_as_float(125829120)));
                            unsigned int _min_424 = ((s_57) < (260046848) ? (s_57) : (260046848));
                            unsigned int h_59 = _min_424 + ((h_noise[13] & 536805376) >> 16) >> 13;
                            unsigned int mag_57 = ((s_57 <= 2139095040) ? h_59 : 32256);
                            uint32_t mags_27;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_27) : "r"(mag_56), "r"(mag_57));
                            uint32_t signs_27;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_27) : "r"(__as_u32(S[22])), "r"(__as_u32(S[23])));
                            unsigned int word_57 = signs_27 & 2147516416u | mags_27;
                            h_words[1] = word_57;
                            reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                            float _abs_425;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_425) : "f"(S[28]));
                            unsigned int s_58 = __as_u32(mul_f32_rz_noftz(_abs_425, __uint_as_float(125829120)));
                            unsigned int _min_426 = ((s_58) < (260046848) ? (s_58) : (260046848));
                            unsigned int h_60 = _min_426 + (h_noise[14] & 8191) >> 13;
                            unsigned int mag_58 = ((s_58 <= 2139095040) ? h_60 : 32256);
                            float _abs_427;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_427) : "f"(S[29]));
                            unsigned int s_59 = __as_u32(mul_f32_rz_noftz(_abs_427, __uint_as_float(125829120)));
                            unsigned int _min_428 = ((s_59) < (260046848) ? (s_59) : (260046848));
                            unsigned int h_61 = _min_428 + ((h_noise[14] & 536805376) >> 16) >> 13;
                            unsigned int mag_59 = ((s_59 <= 2139095040) ? h_61 : 32256);
                            uint32_t mags_28;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_28) : "r"(mag_58), "r"(mag_59));
                            uint32_t signs_28;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_28) : "r"(__as_u32(S[28])), "r"(__as_u32(S[29])));
                            unsigned int word_58 = signs_28 & 2147516416u | mags_28;
                            h_words[0] = word_58;
                            float _abs_429;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_429) : "f"(S[30]));
                            unsigned int s_60 = __as_u32(mul_f32_rz_noftz(_abs_429, __uint_as_float(125829120)));
                            unsigned int _min_430 = ((s_60) < (260046848) ? (s_60) : (260046848));
                            unsigned int h_62 = _min_430 + (h_noise[15] & 8191) >> 13;
                            unsigned int mag_60 = ((s_60 <= 2139095040) ? h_62 : 32256);
                            float _abs_431;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_431) : "f"(S[31]));
                            unsigned int s_61 = __as_u32(mul_f32_rz_noftz(_abs_431, __uint_as_float(125829120)));
                            unsigned int _min_432 = ((s_61) < (260046848) ? (s_61) : (260046848));
                            unsigned int h_63 = _min_432 + ((h_noise[15] & 536805376) >> 16) >> 13;
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
                        uint32_t _cvt_rn_f16x2_f32_433;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_433) : "f"(S[1]), "f"(S[0]));
                        h_words[0] = _cvt_rn_f16x2_f32_433;
                        uint32_t _cvt_rn_f16x2_f32_434;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_434) : "f"(S[3]), "f"(S[2]));
                        h_words[1] = _cvt_rn_f16x2_f32_434;
                        reinterpret_cast<int2*>(state + (write_rows_base))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_435;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_435) : "f"(S[5]), "f"(S[4]));
                        h_words[0] = _cvt_rn_f16x2_f32_435;
                        uint32_t _cvt_rn_f16x2_f32_436;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_436) : "f"(S[7]), "f"(S[6]));
                        h_words[1] = _cvt_rn_f16x2_f32_436;
                        reinterpret_cast<int2*>(state + (write_rows_base + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_437;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_437) : "f"(S[9]), "f"(S[8]));
                        h_words[0] = _cvt_rn_f16x2_f32_437;
                        uint32_t _cvt_rn_f16x2_f32_438;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_438) : "f"(S[11]), "f"(S[10]));
                        h_words[1] = _cvt_rn_f16x2_f32_438;
                        reinterpret_cast<int2*>(state + (write_rows_base + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_439;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_439) : "f"(S[13]), "f"(S[12]));
                        h_words[0] = _cvt_rn_f16x2_f32_439;
                        uint32_t _cvt_rn_f16x2_f32_440;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_440) : "f"(S[15]), "f"(S[14]));
                        h_words[1] = _cvt_rn_f16x2_f32_440;
                        reinterpret_cast<int2*>(state + (write_rows_base + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_441;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_441) : "f"(S[17]), "f"(S[16]));
                        h_words[0] = _cvt_rn_f16x2_f32_441;
                        uint32_t _cvt_rn_f16x2_f32_442;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_442) : "f"(S[19]), "f"(S[18]));
                        h_words[1] = _cvt_rn_f16x2_f32_442;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_443;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_443) : "f"(S[21]), "f"(S[20]));
                        h_words[0] = _cvt_rn_f16x2_f32_443;
                        uint32_t _cvt_rn_f16x2_f32_444;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_444) : "f"(S[23]), "f"(S[22]));
                        h_words[1] = _cvt_rn_f16x2_f32_444;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_445;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_445) : "f"(S[25]), "f"(S[24]));
                        h_words[0] = _cvt_rn_f16x2_f32_445;
                        uint32_t _cvt_rn_f16x2_f32_446;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_446) : "f"(S[27]), "f"(S[26]));
                        h_words[1] = _cvt_rn_f16x2_f32_446;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_447;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_447) : "f"(S[29]), "f"(S[28]));
                        h_words[0] = _cvt_rn_f16x2_f32_447;
                        uint32_t _cvt_rn_f16x2_f32_448;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_448) : "f"(S[31]), "f"(S[30]));
                        h_words[1] = _cvt_rn_f16x2_f32_448;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                    }
                }
            }
        }
    }
}

} // extern "C"
