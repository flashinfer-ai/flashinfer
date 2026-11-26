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

__device__ __forceinline__ float mul_f32_rz_noftz(float a, float b) {
    float r;
    asm("mul.rz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_gdp_8f8062d89b8b871f1e2f(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __half* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, int* __restrict__ ssm_state_indices, long long* __restrict__ rand_seed, CakeFastDivmod heads, int num_tokens, float scale, int disable_state_update, int use_ssm_state_indices, int philox_rounds, long long state_size_p0, long long state_stride_p0, long long state_stride_p1, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long state_stride_p2, long long out_stride_p0, long long out_stride_p1, long long out_stride_p2, long long ssm_state_indices_stride_p0, long long ssm_state_indices_stride_p1)
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
    float* sQ = reinterpret_cast<float*>(smem_raw + 0);
    const int sQ_addr = smem + 0;
    float* sK = reinterpret_cast<float*>(smem_raw + 512);
    const int sK_addr = smem + 512;
    float* sV = reinterpret_cast<float*>(smem_raw + 2048);
    const int sV_addr = smem + 2048;
    float* sAlpha = reinterpret_cast<float*>(smem_raw + 2144);
    const int sAlpha_addr = smem + 2144;
    float* sBeta = reinterpret_cast<float*>(smem_raw + 2148);
    const int sBeta_addr = smem + 2148;

    // === Task calls (declaration order) ===
    int linear_block = blockIdx.x;
    int state_head = linear_block / 8;
    int split = linear_block - state_head * 8;
    uint32_t _fast_div_q_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)state_head), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)state_head);
    uint32_t _fast_div_r_0 = (uint32_t)((unsigned int)state_head) - _fast_div_q_0 * (uint32_t)(heads.divisor);
    int n = (int)_fast_div_q_0;
    int h = (int)_fast_div_r_0;
    int lane_local = lane;
    int warp_local = warp;
    long long state_pool_size = state_size_p0;
    int read_state_slot_raw = initial_state_indices[n];
    int read_state_slot_valid = ((read_state_slot_raw >= 0 && state_pool_size > (long long)read_state_slot_raw) ? 1 : 0);
    int write_state_slot_raw = output_state_indices[n];
    int write_state_slot_valid = ((write_state_slot_raw >= 0 && state_pool_size > (long long)write_state_slot_raw) ? 1 : 0);
    long long read_state_head_base = (long long)read_state_slot_raw * state_stride_p0 + (long long)h * state_stride_p1;
    long long write_state_head_base = (long long)write_state_slot_raw * state_stride_p0 + (long long)h * state_stride_p1;
    int split_v_base = split * 8;
    int k_col0 = lane_local * 4;
    unsigned int seed_lo = 0;
    unsigned int seed_hi = 0;
    if (philox_rounds != 0) {
        long long seed_value = *rand_seed;
        seed_lo = (unsigned int)seed_value;
        seed_hi = (unsigned int)(seed_value >> 32ll);
    }
    int steps = 3;
    int pre_m = ((warp_local) < (steps - 1) ? (warp_local) : (steps - 1));
    int pre_q_shift = warp_local + 1;
    int pre_q_first = ((pre_q_shift < 4) ? pre_q_shift : pre_q_shift - 4);
    int pre_t = ((pre_q_first) < (num_tokens - 1) ? (pre_q_first) : (num_tokens - 1));
    float pre_k[4];
    float pre_q[4];
    float pre_s[5];
    long long pre_k_index = (long long)n * k_stride_p0 + (long long)pre_m * k_stride_p1 + (long long)h * k_stride_p2;
    {
        uint2 _vld_0;
        _vld_0 = *reinterpret_cast<const uint2*>(k + pre_k_index + (long long)k_col0);
        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&pre_k[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
            (&pre_k[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
        }
    }
    int pre_v_lane = ((lane_local < 8) ? lane_local : 7);
    long long pre_v_index = (long long)n * v_stride_p0 + (long long)pre_m * v_stride_p1 + (long long)h * v_stride_p2;
    pre_s[0] = (float)v[pre_v_index + (long long)split_v_base + (long long)pre_v_lane];
    long long pre_b_index = (long long)n * b_stride_p0 + (long long)pre_m * b_stride_p1 + (long long)h * b_stride_p2;
    pre_s[1] = (float)b[pre_b_index];
    long long pre_q_index = (long long)n * q_stride_p0 + (long long)pre_t * q_stride_p1 + (long long)h * q_stride_p2;
    {
        uint2 _vld_1;
        _vld_1 = *reinterpret_cast<const uint2*>(q + pre_q_index + (long long)k_col0);
        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&pre_q[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
            (&pre_q[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
        }
    }
    long long pre_gate_index = (long long)n * a_stride_p0 + (long long)pre_t * a_stride_p1 + (long long)h * a_stride_p2;
    pre_s[2] = (float)a[pre_gate_index];
    pre_s[3] = dt_bias[h];
    pre_s[4] = A_log[h];
    if (read_state_slot_valid != 0) {
        float r_q[4];
        float r_k[4];
        float r_h[4];
        unsigned int h_words[2];
        unsigned int h_noise[8];
        unsigned int raw_w[4];
        long long raw_row = read_state_head_base + (long long)(split_v_base + warp_local) * state_stride_p2;
        long long raw_row_4 = raw_row + 4ll * state_stride_p2;
        {
            raw_w[0] = *reinterpret_cast<const uint32_t*>(state + raw_row + (long long)k_col0);
        }
        {
            raw_w[1] = *reinterpret_cast<const uint32_t*>(state + raw_row + (long long)(k_col0 + 2));
        }
        {
            raw_w[2] = *reinterpret_cast<const uint32_t*>(state + raw_row_4 + (long long)k_col0);
        }
        {
            raw_w[3] = *reinterpret_cast<const uint32_t*>(state + raw_row_4 + (long long)(k_col0 + 2));
        }
        if (warp_local < steps) {
            float k_pair0_x = pre_k[0];
            float k_pair0_y = pre_k[1];
            float k_pair0_x_0 = pre_k[2];
            float k_pair0_y_0 = pre_k[3];
            float fma_1;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_1) : "f"(k_pair0_x), "f"(k_pair0_x), "f"(0.0f));
            float fma_2;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_2) : "f"(k_pair0_y), "f"(k_pair0_y), "f"(0.0f));
            float k_pair0_x_1 = fma_1;
            float k_pair0_y_1 = fma_2;
            float fma_1_0;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_1_0) : "f"(k_pair0_x_0), "f"(k_pair0_x_0), "f"(k_pair0_x_1));
            float fma_2_0;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_2_0) : "f"(k_pair0_y_0), "f"(k_pair0_y_0), "f"(k_pair0_y_1));
            k_pair0_x_1 = fma_1_0;
            k_pair0_y_1 = fma_2_0;
            float sum_k = k_pair0_x_1 + k_pair0_y_1;
            float _warp_reduce_6 = sum_k;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_6 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_6, offset);
            sum_k = _warp_reduce_6;
            float k_norm = rsqrtf(sum_k + 1e-06f);
            float k_pair0_x_2 = k_norm;
            float k_pair0_y_2 = k_norm;
            float mul_1;
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(mul_1) : "f"(k_pair0_x), "f"(k_pair0_x_2));
            float mul_2;
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(mul_2) : "f"(k_pair0_y), "f"(k_pair0_y_2));
            k_pair0_x = mul_1;
            k_pair0_y = mul_2;
            float mul_1_0;
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(mul_1_0) : "f"(k_pair0_x_0), "f"(k_pair0_x_2));
            float mul_2_0;
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(mul_2_0) : "f"(k_pair0_y_0), "f"(k_pair0_y_2));
            k_pair0_x_0 = mul_1_0;
            k_pair0_y_0 = mul_2_0;
            sK[pre_m * 128 + k_col0] = k_pair0_x;
            sK[pre_m * 128 + k_col0 + 1] = k_pair0_y;
            sK[pre_m * 128 + k_col0 + 2] = k_pair0_x_0;
            sK[pre_m * 128 + k_col0 + 3] = k_pair0_y_0;
            if (lane_local < 8) {
                sV[pre_m * 8 + lane_local] = pre_s[0];
            }
            if (lane_local == 0) {
                float _exp_8 = __expf(-pre_s[1]);
                float _rcp_7 = approx_rcp(1.0f + _exp_8);
                sBeta[pre_m] = _rcp_7;
            }
        }
        if (pre_q_first < num_tokens) {
            float k_pair0_x_3 = pre_q[0];
            float k_pair0_y_3 = pre_q[1];
            float k_pair0_x_3_1 = pre_q[2];
            float k_pair0_y_3_1 = pre_q[3];
            float q_norm = scale;
            float fma_1_1;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_1_1) : "f"(k_pair0_x_3), "f"(k_pair0_x_3), "f"(0.0f));
            float fma_2_1;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_2_1) : "f"(k_pair0_y_3), "f"(k_pair0_y_3), "f"(0.0f));
            float k_pair0_x_4 = fma_1_1;
            float k_pair0_y_4 = fma_2_1;
            float fma_1_1_1;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_1_1_1) : "f"(k_pair0_x_3_1), "f"(k_pair0_x_3_1), "f"(k_pair0_x_4));
            float fma_2_1_1;
            asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_2_1_1) : "f"(k_pair0_y_3_1), "f"(k_pair0_y_3_1), "f"(k_pair0_y_4));
            k_pair0_x_4 = fma_1_1_1;
            k_pair0_y_4 = fma_2_1_1;
            float sum_q = k_pair0_x_4 + k_pair0_y_4;
            float _warp_reduce_9 = sum_q;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_9 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_9, offset);
            sum_q = _warp_reduce_9;
            float _rsqrt_10 = rsqrtf(sum_q + 1e-06f);
            q_norm = _rsqrt_10 * scale;
            float k_pair0_x_5 = q_norm;
            float k_pair0_y_5 = q_norm;
            float mul_1_1;
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(mul_1_1) : "f"(k_pair0_x_3), "f"(k_pair0_x_5));
            float mul_2_1;
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(mul_2_1) : "f"(k_pair0_y_3), "f"(k_pair0_y_5));
            float k_pair0_x_6 = mul_1_1;
            float k_pair0_y_6 = mul_2_1;
            float mul_1_1_1;
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(mul_1_1_1) : "f"(k_pair0_x_3_1), "f"(k_pair0_x_5));
            float mul_2_1_1;
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(mul_2_1_1) : "f"(k_pair0_y_3_1), "f"(k_pair0_y_5));
            float k_pair0_x_7 = mul_1_1_1;
            float k_pair0_y_7 = mul_2_1_1;
            sQ[pre_t * 128 + k_col0] = k_pair0_x_6;
            sQ[pre_t * 128 + k_col0 + 1] = k_pair0_y_6;
            sQ[pre_t * 128 + k_col0 + 2] = k_pair0_x_7;
            sQ[pre_t * 128 + k_col0 + 3] = k_pair0_y_7;
            if (lane_local == 0) {
                float x = pre_s[2] + pre_s[3];
                float softplus_x = x;
                if (x <= 20.0f) {
                    float _exp_12 = __expf(x);
                    float _log2_11;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_11) : "f"(1.0f + _exp_12));
                    softplus_x = _log2_11 * 0.6931472f;
                }
                float _exp_13 = __expf(pre_s[4]);
                float g_log = -_exp_13 * softplus_x;
                float _exp_14 = __expf(g_log);
                sAlpha[pre_t] = _exp_14;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int row_group = 0; row_group < 8; row_group += 4) {
            int local_row = row_group + warp_local;
            int v_row = split_v_base + local_row;
            float h_pair0[2];
            float h_pair1[2];
            unsigned int h_row_w[2];
            h_row_w[0] = ((row_group != 0) ? raw_w[2] : raw_w[0]);
            h_row_w[1] = ((row_group != 0) ? raw_w[3] : raw_w[1]);
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&h_pair0[_pair * 2]))
                    : "r"(h_row_w[_pair]));
            }
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&h_pair1[_pair * 2]))
                    : "r"(h_row_w[1 + _pair]));
            }
            #pragma unroll
            for (int t = 0; t < 1; t++) {
                float decay_val = sAlpha[t];
                float k_pair0_x_8 = decay_val;
                float k_pair0_y_8 = decay_val;
                float mul_1_2;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(mul_1_2) : "f"(h_pair0[0]), "f"(k_pair0_x_8));
                float mul_2_2;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(mul_2_2) : "f"(h_pair0[1]), "f"(k_pair0_y_8));
                h_pair0[0] = mul_1_2;
                h_pair0[1] = mul_2_2;
                float mul_1_2_1;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(mul_1_2_1) : "f"(h_pair1[0]), "f"(k_pair0_x_8));
                float mul_2_2_1;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(mul_2_2_1) : "f"(h_pair1[1]), "f"(k_pair0_y_8));
                h_pair1[0] = mul_1_2_1;
                h_pair1[1] = mul_2_2_1;
                #pragma unroll
                for (int j = 0; j < 3; j++) {
                    int m = t * 3 + j;
                    float k_pair0_x_8_1 = sK[m * 128 + k_col0];
                    float k_pair0_y_8_1 = sK[m * 128 + k_col0 + 1];
                    float k_pair0_x_9 = sK[m * 128 + k_col0 + 2];
                    float k_pair0_y_9 = sK[m * 128 + k_col0 + 3];
                    float fma_1_2;
                    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_1_2) : "f"(h_pair0[0]), "f"(k_pair0_x_8_1), "f"(0.0f));
                    float fma_2_2;
                    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_2_2) : "f"(h_pair0[1]), "f"(k_pair0_y_8_1), "f"(0.0f));
                    float k_pair0_x_10 = fma_1_2;
                    float k_pair0_y_10 = fma_2_2;
                    float fma_1_2_1;
                    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_1_2_1) : "f"(h_pair1[0]), "f"(k_pair0_x_9), "f"(k_pair0_x_10));
                    float fma_2_2_1;
                    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_2_2_1) : "f"(h_pair1[1]), "f"(k_pair0_y_9), "f"(k_pair0_y_10));
                    k_pair0_x_10 = fma_1_2_1;
                    k_pair0_y_10 = fma_2_2_1;
                    float sum_hk = k_pair0_x_10 + k_pair0_y_10;
                    float _warp_reduce_15 = sum_hk;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_15 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_15, offset);
                    sum_hk = _warp_reduce_15;
                    float v_val = sV[m * 8 + local_row];
                    float beta_val = sBeta[m];
                    float v_new = (v_val - sum_hk) * beta_val;
                    float k_pair0_x_11 = v_new;
                    float k_pair0_y_11 = v_new;
                    float fma_1_3;
                    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_1_3) : "f"(k_pair0_x_8_1), "f"(k_pair0_x_11), "f"(h_pair0[0]));
                    float fma_2_3;
                    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_2_3) : "f"(k_pair0_y_8_1), "f"(k_pair0_y_11), "f"(h_pair0[1]));
                    h_pair0[0] = fma_1_3;
                    h_pair0[1] = fma_2_3;
                    float fma_1_4;
                    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_1_4) : "f"(k_pair0_x_9), "f"(k_pair0_x_11), "f"(h_pair1[0]));
                    float fma_2_4;
                    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_2_4) : "f"(k_pair0_y_9), "f"(k_pair0_y_11), "f"(h_pair1[1]));
                    h_pair1[0] = fma_1_4;
                    h_pair1[1] = fma_2_4;
                }
                float k_pair0_x_12 = sQ[t * 128 + k_col0];
                float k_pair0_y_12 = sQ[t * 128 + k_col0 + 1];
                float k_pair0_x_13 = sQ[t * 128 + k_col0 + 2];
                float k_pair0_y_13 = sQ[t * 128 + k_col0 + 3];
                float fma_1_5;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_1_5) : "f"(h_pair0[0]), "f"(k_pair0_x_12), "f"(0.0f));
                float fma_2_5;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_2_5) : "f"(h_pair0[1]), "f"(k_pair0_y_12), "f"(0.0f));
                float k_pair0_x_14 = fma_1_5;
                float k_pair0_y_14 = fma_2_5;
                float fma_1_5_1;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_1_5_1) : "f"(h_pair1[0]), "f"(k_pair0_x_13), "f"(k_pair0_x_14));
                float fma_2_5_1;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_2_5_1) : "f"(h_pair1[1]), "f"(k_pair0_y_13), "f"(k_pair0_y_14));
                k_pair0_x_14 = fma_1_5_1;
                k_pair0_y_14 = fma_2_5_1;
                float sum_hq = k_pair0_x_14 + k_pair0_y_14;
                float _warp_reduce_16 = sum_hq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_16 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_16, offset);
                sum_hq = _warp_reduce_16;
                if (lane_local == 0) {
                    long long out_index = (long long)n * out_stride_p0 + (long long)t * out_stride_p1 + (long long)h * out_stride_p2;
                    out[out_index + (long long)v_row] = sum_hq;
                }
            }
            if (row_group == 0) {
                if (philox_rounds != 0) {
                    int upper = lane_local >> 4;
                    unsigned int key0 = seed_lo;
                    unsigned int key1 = seed_hi;
                    long long o_0 = read_state_head_base + (long long)(split_v_base + warp_local + upper * 4) * state_stride_p2 + (long long)((lane_local & 15) * 4);
                    unsigned long long p0_0 = (unsigned long long)(unsigned int)o_0 * 3528531795ull;
                    h_noise[0] = (unsigned int)(o_0 >> 32ll) ^ key0;
                    h_noise[2] = (unsigned int)(p0_0 >> 32ull) ^ key1;
                    h_noise[3] = (unsigned int)p0_0;
                    key0 = key0 + 2654435769u;
                    key1 = key1 + 3144134277u;
                    unsigned long long q0_0 = (unsigned long long)h_noise[0] * 3528531795ull;
                    unsigned long long q1_0 = (unsigned long long)h_noise[2] * 3449720151ull;
                    h_noise[0] = (unsigned int)(q1_0 >> 32ull) ^ key0;
                    h_noise[1] = (unsigned int)q1_0;
                    h_noise[2] = (unsigned int)(q0_0 >> 32ull) ^ h_noise[3] ^ key1;
                    h_noise[3] = (unsigned int)q0_0;
                    key0 = key0 + 2654435769u;
                    key1 = key1 + 3144134277u;
                    unsigned long long s0_0 = (unsigned long long)h_noise[0] * 3528531795ull;
                    unsigned long long s1_0 = (unsigned long long)h_noise[2] * 3449720151ull;
                    h_noise[0] = (unsigned int)(s1_0 >> 32ull) ^ h_noise[1] ^ key0;
                    h_noise[1] = (unsigned int)s1_0;
                    h_noise[2] = (unsigned int)(s0_0 >> 32ull) ^ h_noise[3] ^ key1;
                    h_noise[3] = (unsigned int)s0_0;
                    key0 = key0 + 2654435769u;
                    key1 = key1 + 3144134277u;
                    unsigned long long s0_1 = (unsigned long long)h_noise[0] * 3528531795ull;
                    unsigned long long s1_1 = (unsigned long long)h_noise[2] * 3449720151ull;
                    h_noise[0] = (unsigned int)(s1_1 >> 32ull) ^ h_noise[1] ^ key0;
                    h_noise[1] = (unsigned int)s1_1;
                    h_noise[2] = (unsigned int)(s0_1 >> 32ull) ^ h_noise[3] ^ key1;
                    h_noise[3] = (unsigned int)s0_1;
                    key0 = key0 + 2654435769u;
                    key1 = key1 + 3144134277u;
                    unsigned long long s0_2 = (unsigned long long)h_noise[0] * 3528531795ull;
                    unsigned long long s1_2 = (unsigned long long)h_noise[2] * 3449720151ull;
                    h_noise[0] = (unsigned int)(s1_2 >> 32ull) ^ h_noise[1] ^ key0;
                    h_noise[1] = (unsigned int)s1_2;
                    h_noise[2] = (unsigned int)(s0_2 >> 32ull) ^ h_noise[3] ^ key1;
                    h_noise[3] = (unsigned int)s0_2;
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
                        key0 = key0 + 2654435769u;
                        key1 = key1 + 3144134277u;
                    }
                    unsigned int own0 = ((upper != 0) ? h_noise[2] : h_noise[0]);
                    unsigned int own1 = ((upper != 0) ? h_noise[3] : h_noise[1]);
                    unsigned int give0 = ((upper != 0) ? h_noise[0] : h_noise[2]);
                    unsigned int give1 = ((upper != 0) ? h_noise[1] : h_noise[3]);
                    unsigned int got0 = __shfl_xor_sync(0xFFFFFFFF, give0, 16);
                    unsigned int got1 = __shfl_xor_sync(0xFFFFFFFF, give1, 16);
                    h_noise[4] = ((upper != 0) ? got0 : own0);
                    h_noise[5] = ((upper != 0) ? got1 : own1);
                    h_noise[6] = ((upper != 0) ? own0 : got0);
                    h_noise[7] = ((upper != 0) ? own1 : got1);
                }
            }
            if (use_ssm_state_indices != 0) {
                int slot0 = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0];
                if (slot0 >= 0 && state_pool_size > (long long)slot0) {
                    long long snap_head = (long long)slot0 * state_stride_p0 + (long long)h * state_stride_p1;
                    if (philox_rounds != 0) {
                        unsigned int w0 = ((row_group != 0) ? h_noise[6] : h_noise[4]);
                        unsigned int w1 = ((row_group != 0) ? h_noise[7] : h_noise[5]);
                        float _abs_17;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_17) : "f"(h_pair0[0]));
                        unsigned int s = __as_u32(mul_f32_rz_noftz(_abs_17, __uint_as_float(125829120)));
                        unsigned int _min_18 = ((s) < (260046848) ? (s) : (260046848));
                        unsigned int h_0 = _min_18 + (w0 & 8191) >> 13;
                        unsigned int mag = ((s <= 2139095040) ? h_0 : 32256);
                        float _abs_19;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_19) : "f"(h_pair0[1]));
                        unsigned int s_0 = __as_u32(mul_f32_rz_noftz(_abs_19, __uint_as_float(125829120)));
                        unsigned int _min_20 = ((s_0) < (260046848) ? (s_0) : (260046848));
                        unsigned int h_1 = _min_20 + ((w0 & 536805376) >> 16) >> 13;
                        unsigned int mag_0 = ((s_0 <= 2139095040) ? h_1 : 32256);
                        uint32_t mags;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags) : "r"(mag), "r"(mag_0));
                        uint32_t signs;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs) : "r"(__as_u32(h_pair0[0])), "r"(__as_u32(h_pair0[1])));
                        unsigned int word = signs & 2147516416u | mags;
                        h_words[0] = word;
                        float _abs_21;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_21) : "f"(h_pair1[0]));
                        unsigned int s_1 = __as_u32(mul_f32_rz_noftz(_abs_21, __uint_as_float(125829120)));
                        unsigned int _min_22 = ((s_1) < (260046848) ? (s_1) : (260046848));
                        unsigned int h_2 = _min_22 + (w1 & 8191) >> 13;
                        unsigned int mag_1 = ((s_1 <= 2139095040) ? h_2 : 32256);
                        float _abs_23;
                        asm("abs.f32 %0, %1;" : "=f"(_abs_23) : "f"(h_pair1[1]));
                        unsigned int s_2 = __as_u32(mul_f32_rz_noftz(_abs_23, __uint_as_float(125829120)));
                        unsigned int _min_24 = ((s_2) < (260046848) ? (s_2) : (260046848));
                        unsigned int h_3 = _min_24 + ((w1 & 536805376) >> 16) >> 13;
                        unsigned int mag_2 = ((s_2 <= 2139095040) ? h_3 : 32256);
                        uint32_t mags_0;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_0) : "r"(mag_1), "r"(mag_2));
                        uint32_t signs_0;
                        asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_0) : "r"(__as_u32(h_pair1[0])), "r"(__as_u32(h_pair1[1])));
                        unsigned int word_0 = signs_0 & 2147516416u | mags_0;
                        h_words[1] = word_0;
                    }
                    if (philox_rounds == 0) {
                        uint32_t _cvt_rn_f16x2_f32_25;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_25) : "f"(h_pair0[1]), "f"(h_pair0[0]));
                        h_words[0] = _cvt_rn_f16x2_f32_25;
                        uint32_t _cvt_rn_f16x2_f32_26;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_26) : "f"(h_pair1[1]), "f"(h_pair1[0]));
                        h_words[1] = _cvt_rn_f16x2_f32_26;
                    }
                    reinterpret_cast<int2*>(state + (snap_head + (long long)v_row * state_stride_p2 + (long long)k_col0))[0] = reinterpret_cast<int2*>(h_words)[0];
                }
            }
            if (write_state_slot_valid != 0) {
                if (disable_state_update == 0) {
                    if (use_ssm_state_indices == 0) {
                        if (philox_rounds != 0) {
                            unsigned int w0_1 = ((row_group != 0) ? h_noise[6] : h_noise[4]);
                            unsigned int w1_1 = ((row_group != 0) ? h_noise[7] : h_noise[5]);
                            float _abs_27;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_27) : "f"(h_pair0[0]));
                            unsigned int s_3 = __as_u32(mul_f32_rz_noftz(_abs_27, __uint_as_float(125829120)));
                            unsigned int _min_28 = ((s_3) < (260046848) ? (s_3) : (260046848));
                            unsigned int h_4 = _min_28 + (w0_1 & 8191) >> 13;
                            unsigned int mag_3 = ((s_3 <= 2139095040) ? h_4 : 32256);
                            float _abs_29;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_29) : "f"(h_pair0[1]));
                            unsigned int s_3_1 = __as_u32(mul_f32_rz_noftz(_abs_29, __uint_as_float(125829120)));
                            unsigned int _min_30 = ((s_3_1) < (260046848) ? (s_3_1) : (260046848));
                            unsigned int h_5 = _min_30 + ((w0_1 & 536805376) >> 16) >> 13;
                            unsigned int mag_3_1 = ((s_3_1 <= 2139095040) ? h_5 : 32256);
                            uint32_t mags_1;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_1) : "r"(mag_3), "r"(mag_3_1));
                            uint32_t signs_1;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_1) : "r"(__as_u32(h_pair0[0])), "r"(__as_u32(h_pair0[1])));
                            unsigned int word_1 = signs_1 & 2147516416u | mags_1;
                            h_words[0] = word_1;
                            float _abs_31;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_31) : "f"(h_pair1[0]));
                            unsigned int s_4 = __as_u32(mul_f32_rz_noftz(_abs_31, __uint_as_float(125829120)));
                            unsigned int _min_32 = ((s_4) < (260046848) ? (s_4) : (260046848));
                            unsigned int h_6 = _min_32 + (w1_1 & 8191) >> 13;
                            unsigned int mag_4 = ((s_4 <= 2139095040) ? h_6 : 32256);
                            float _abs_33;
                            asm("abs.f32 %0, %1;" : "=f"(_abs_33) : "f"(h_pair1[1]));
                            unsigned int s_5 = __as_u32(mul_f32_rz_noftz(_abs_33, __uint_as_float(125829120)));
                            unsigned int _min_34 = ((s_5) < (260046848) ? (s_5) : (260046848));
                            unsigned int h_7 = _min_34 + ((w1_1 & 536805376) >> 16) >> 13;
                            unsigned int mag_5 = ((s_5 <= 2139095040) ? h_7 : 32256);
                            uint32_t mags_1_1;
                            asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(mags_1_1) : "r"(mag_4), "r"(mag_5));
                            uint32_t signs_1_1;
                            asm("prmt.b32 %0, %1, %2, 0x7531;" : "=r"(signs_1_1) : "r"(__as_u32(h_pair1[0])), "r"(__as_u32(h_pair1[1])));
                            unsigned int word_1_1 = signs_1_1 & 2147516416u | mags_1_1;
                            h_words[1] = word_1_1;
                        }
                        if (philox_rounds == 0) {
                            uint32_t _cvt_rn_f16x2_f32_35;
                            asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_35) : "f"(h_pair0[1]), "f"(h_pair0[0]));
                            h_words[0] = _cvt_rn_f16x2_f32_35;
                            uint32_t _cvt_rn_f16x2_f32_36;
                            asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_36) : "f"(h_pair1[1]), "f"(h_pair1[0]));
                            h_words[1] = _cvt_rn_f16x2_f32_36;
                        }
                        reinterpret_cast<int2*>(state + (write_state_head_base + (long long)v_row * state_stride_p2 + (long long)k_col0))[0] = reinterpret_cast<int2*>(h_words)[0];
                    }
                }
            }
        }
    }
}

} // extern "C"
