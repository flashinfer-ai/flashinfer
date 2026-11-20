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

__global__ __launch_bounds__(128) void
kernel_cake_gdp_dbd09ba202e904a29148(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __half* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, int* __restrict__ ssm_state_indices, long long* __restrict__ rand_seed, CakeFastDivmod heads, int num_tokens, float scale, int disable_state_update, int use_ssm_state_indices, int philox_rounds, long long state_size_p0, long long state_stride_p0, long long state_stride_p1, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long state_stride_p2, long long out_stride_p0, long long out_stride_p1, long long out_stride_p2, long long ssm_state_indices_stride_p0, long long ssm_state_indices_stride_p1)
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
        int row_own = lane_local >> 4 & 1;
        int row_a = warp_local + row_own * 4;
        int row_b = warp_local + 4 - row_own * 4;
        int v_row_a = split_v_base + row_a;
        int v_row_b = split_v_base + row_b;
        float raw_a[4];
        float raw_b[4];
        long long raw_row_a = read_state_head_base + (long long)v_row_a * state_stride_p2;
        long long raw_row_b = read_state_head_base + (long long)v_row_b * state_stride_p2;
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(state + raw_row_a + (long long)k_col0);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&raw_a[0 + _pair * 2]))
                    : "r"(_vpairs_2[_pair]));
            }
        }
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(state + raw_row_b + (long long)k_col0);
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&raw_b[0 + _pair * 2]))
                    : "r"(_vpairs_3[_pair]));
            }
        }
        if (warp_local < steps) {
            float2 k_pair0 = make_float2(pre_k[0], pre_k[1]);
            float2 k_pair0_0 = make_float2(pre_k[2], pre_k[3]);
            float2 sum_k_pair = fma_f32x2_rn_ftz(k_pair0, k_pair0, make_float2(0.0f, 0.0f));
            sum_k_pair = fma_f32x2_rn_ftz(k_pair0_0, k_pair0_0, sum_k_pair);
            float sum_k = sum_k_pair.x + sum_k_pair.y;
            float _warp_reduce_4 = sum_k;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_4 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_4, offset);
            sum_k = _warp_reduce_4;
            float k_norm = rsqrtf(sum_k + 1e-06f);
            float2 k_norm_pair = make_float2(k_norm, k_norm);
            k_pair0 = mul_f32x2(k_pair0, k_norm_pair);
            k_pair0_0 = mul_f32x2(k_pair0_0, k_norm_pair);
            sK[pre_m * 128 + k_col0] = k_pair0.x;
            sK[pre_m * 128 + k_col0 + 1] = k_pair0.y;
            sK[pre_m * 128 + k_col0 + 2] = k_pair0_0.x;
            sK[pre_m * 128 + k_col0 + 3] = k_pair0_0.y;
            if (lane_local < 8) {
                sV[pre_m * 8 + lane_local] = pre_s[0];
            }
            if (lane_local == 0) {
                float _exp_6 = __expf(-pre_s[1]);
                float _rcp_5 = approx_rcp(1.0f + _exp_6);
                sBeta[pre_m] = _rcp_5;
            }
        }
        if (pre_q_first < num_tokens) {
            float2 k_pair0_1 = make_float2(pre_q[0], pre_q[1]);
            float2 k_pair0_1_1 = make_float2(pre_q[2], pre_q[3]);
            float q_norm = scale;
            float2 sum_k_pair_1 = fma_f32x2_rn_ftz(k_pair0_1, k_pair0_1, make_float2(0.0f, 0.0f));
            sum_k_pair_1 = fma_f32x2_rn_ftz(k_pair0_1_1, k_pair0_1_1, sum_k_pair_1);
            float sum_q = sum_k_pair_1.x + sum_k_pair_1.y;
            float _warp_reduce_7 = sum_q;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_7 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_7, offset);
            sum_q = _warp_reduce_7;
            float _rsqrt_8 = rsqrtf(sum_q + 1e-06f);
            q_norm = _rsqrt_8 * scale;
            float2 k_pair0_2 = make_float2(q_norm, q_norm);
            float2 q_pair0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair0) : "l"(*(const unsigned long long*)&k_pair0_1), "l"(*(const unsigned long long*)&k_pair0_2));
            float2 q_pair0_0;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q_pair0_0) : "l"(*(const unsigned long long*)&k_pair0_1_1), "l"(*(const unsigned long long*)&k_pair0_2));
            sQ[pre_t * 128 + k_col0] = q_pair0.x;
            sQ[pre_t * 128 + k_col0 + 1] = q_pair0.y;
            sQ[pre_t * 128 + k_col0 + 2] = q_pair0_0.x;
            sQ[pre_t * 128 + k_col0 + 3] = q_pair0_0.y;
            if (lane_local == 0) {
                float x = pre_s[2] + pre_s[3];
                float softplus_x = x;
                if (x <= 20.0f) {
                    float _exp_10 = __expf(x);
                    float _log2_9;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_9) : "f"(1.0f + _exp_10));
                    softplus_x = _log2_9 * 0.6931472f;
                }
                float _exp_11 = __expf(pre_s[4]);
                float g_log = -_exp_11 * softplus_x;
                float _exp_12 = __expf(g_log);
                sAlpha[pre_t] = _exp_12;
            }
        }
        __syncthreads();
        float2 ha_pair0 = make_float2(raw_a[0], raw_a[1]);
        float2 ha_pair0_0 = make_float2(raw_a[2], raw_a[3]);
        float2 ha_pair0_1 = make_float2(raw_b[0], raw_b[1]);
        float2 ha_pair0_2 = make_float2(raw_b[2], raw_b[3]);
        #pragma unroll
        for (int t = 0; t < 1; t++) {
            float decay_val = sAlpha[t];
            float2 decay_pair = make_float2(decay_val, decay_val);
            ha_pair0 = mul_f32x2(ha_pair0, decay_pair);
            ha_pair0_0 = mul_f32x2(ha_pair0_0, decay_pair);
            ha_pair0_1 = mul_f32x2(ha_pair0_1, decay_pair);
            ha_pair0_2 = mul_f32x2(ha_pair0_2, decay_pair);
            #pragma unroll
            for (int j = 0; j < 3; j++) {
                int m = t * 3 + j;
                float2 k_pair0_3 = make_float2(sK[m * 128 + k_col0], sK[m * 128 + k_col0 + 1]);
                float2 k_pair0_3_1 = make_float2(sK[m * 128 + k_col0 + 2], sK[m * 128 + k_col0 + 3]);
                float2 sum_k_pair_2 = fma_f32x2_rn_ftz(ha_pair0, k_pair0_3, make_float2(0.0f, 0.0f));
                sum_k_pair_2 = fma_f32x2_rn_ftz(ha_pair0_0, k_pair0_3_1, sum_k_pair_2);
                float2 sum_k_pair_0 = fma_f32x2_rn_ftz(ha_pair0_1, k_pair0_3, make_float2(0.0f, 0.0f));
                sum_k_pair_0 = fma_f32x2_rn_ftz(ha_pair0_2, k_pair0_3_1, sum_k_pair_0);
                float sum_a = sum_k_pair_2.x + sum_k_pair_2.y;
                float sum_b = sum_k_pair_0.x + sum_k_pair_0.y;
                float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, sum_b, 16);
                sum_a = sum_a + _shfl_xor_sync_13;
                float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, sum_a, 8);
                sum_a = sum_a + _shfl_xor_sync_14;
                float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, sum_a, 4);
                sum_a = sum_a + _shfl_xor_sync_15;
                float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, sum_a, 2);
                sum_a = sum_a + _shfl_xor_sync_16;
                float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, sum_a, 1);
                sum_a = sum_a + _shfl_xor_sync_17;
                sum_a = sum_a;
                float v_a = sV[m * 8 + row_a];
                float beta_val = sBeta[m];
                float vn_a = (v_a - sum_a) * beta_val;
                float vn_b = __shfl_xor_sync(0xFFFFFFFF, vn_a, 16);
                float2 k_pair0_4 = make_float2(vn_a, vn_a);
                float2 vn_b_pair = make_float2(vn_b, vn_b);
                ha_pair0 = fma_f32x2_rn_ftz(k_pair0_3, k_pair0_4, ha_pair0);
                ha_pair0_0 = fma_f32x2_rn_ftz(k_pair0_3_1, k_pair0_4, ha_pair0_0);
                ha_pair0_1 = fma_f32x2_rn_ftz(k_pair0_3, vn_b_pair, ha_pair0_1);
                ha_pair0_2 = fma_f32x2_rn_ftz(k_pair0_3_1, vn_b_pair, ha_pair0_2);
            }
            float2 k_pair0_5 = make_float2(sQ[t * 128 + k_col0], sQ[t * 128 + k_col0 + 1]);
            float2 k_pair0_5_1 = make_float2(sQ[t * 128 + k_col0 + 2], sQ[t * 128 + k_col0 + 3]);
            float2 sum_k_pair_3 = fma_f32x2_rn_ftz(ha_pair0, k_pair0_5, make_float2(0.0f, 0.0f));
            sum_k_pair_3 = fma_f32x2_rn_ftz(ha_pair0_0, k_pair0_5_1, sum_k_pair_3);
            float2 sum_k_pair_1_1 = fma_f32x2_rn_ftz(ha_pair0_1, k_pair0_5, make_float2(0.0f, 0.0f));
            sum_k_pair_1_1 = fma_f32x2_rn_ftz(ha_pair0_2, k_pair0_5_1, sum_k_pair_1_1);
            float sum_hq_a = sum_k_pair_3.x + sum_k_pair_3.y;
            float sum_hq_b = sum_k_pair_1_1.x + sum_k_pair_1_1.y;
            float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_b, 16);
            sum_hq_a = sum_hq_a + _shfl_xor_sync_18;
            float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_a, 8);
            sum_hq_a = sum_hq_a + _shfl_xor_sync_19;
            float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_a, 4);
            sum_hq_a = sum_hq_a + _shfl_xor_sync_20;
            float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_a, 2);
            sum_hq_a = sum_hq_a + _shfl_xor_sync_21;
            float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_a, 1);
            sum_hq_a = sum_hq_a + _shfl_xor_sync_22;
            sum_hq_a = sum_hq_a;
            if ((lane_local & 15) == 0) {
                long long out_index = (long long)n * out_stride_p0 + (long long)t * out_stride_p1 + (long long)h * out_stride_p2;
                out[out_index + (long long)v_row_a] = sum_hq_a;
            }
        }
        if (use_ssm_state_indices != 0) {
            int ssm_slot0 = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0];
            if (ssm_slot0 >= 0 && state_pool_size > (long long)ssm_slot0) {
                long long snap_head = (long long)ssm_slot0 * state_stride_p0 + (long long)h * state_stride_p1;
                int high = lane_local >> 4;
                if (philox_rounds != 0) {
                    unsigned int key0 = seed_lo;
                    unsigned int key1 = seed_hi;
                    long long o_0 = read_state_head_base + (long long)v_row_a * state_stride_p2 + (long long)((lane_local & 15) * 4);
                    unsigned long long p0_0 = (unsigned long long)(unsigned int)o_0 * 3528531795ull;
                    h_noise[0] = (unsigned int)(o_0 >> 32ll) ^ key0;
                    h_noise[2] = (unsigned int)(p0_0 >> 32ull) ^ key1;
                    h_noise[3] = (unsigned int)p0_0;
                    long long o_1 = read_state_head_base + (long long)v_row_a * state_stride_p2 + (long long)((lane_local & 15) * 4) + (long long)(v_row_b - v_row_a) * state_stride_p2;
                    unsigned long long p0_1 = (unsigned long long)(unsigned int)o_1 * 3528531795ull;
                    h_noise[4] = (unsigned int)(o_1 >> 32ll) ^ key0;
                    h_noise[6] = (unsigned int)(p0_1 >> 32ull) ^ key1;
                    h_noise[7] = (unsigned int)p0_1;
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
                    key0 = key0 + 2654435769u;
                    key1 = key1 + 3144134277u;
                    unsigned long long s0_2 = (unsigned long long)h_noise[0] * 3528531795ull;
                    unsigned long long s1_2 = (unsigned long long)h_noise[2] * 3449720151ull;
                    h_noise[0] = (unsigned int)(s1_2 >> 32ull) ^ h_noise[1] ^ key0;
                    h_noise[1] = (unsigned int)s1_2;
                    h_noise[2] = (unsigned int)(s0_2 >> 32ull) ^ h_noise[3] ^ key1;
                    h_noise[3] = (unsigned int)s0_2;
                    unsigned long long s0_3 = (unsigned long long)h_noise[4] * 3528531795ull;
                    unsigned long long s1_3 = (unsigned long long)h_noise[6] * 3449720151ull;
                    h_noise[4] = (unsigned int)(s1_3 >> 32ull) ^ h_noise[5] ^ key0;
                    h_noise[5] = (unsigned int)s1_3;
                    h_noise[6] = (unsigned int)(s0_3 >> 32ull) ^ h_noise[7] ^ key1;
                    h_noise[7] = (unsigned int)s0_3;
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
                        key0 = key0 + 2654435769u;
                        key1 = key1 + 3144134277u;
                    }
                }
                if (philox_rounds != 0) {
                    unsigned int w0 = ((high != 0) ? h_noise[2] : h_noise[0]);
                    unsigned int w1 = ((high != 0) ? h_noise[3] : h_noise[1]);
                    uint32_t _cvt_rs_x2_f32_23;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_23) : "f"(ha_pair0.y), "f"(ha_pair0.x), "r"(w0 & 536813567));
                    h_words[0] = _cvt_rs_x2_f32_23;
                    uint32_t _cvt_rs_x2_f32_24;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_24) : "f"(ha_pair0_0.y), "f"(ha_pair0_0.x), "r"(w1 & 536813567));
                    h_words[1] = _cvt_rs_x2_f32_24;
                }
                if (philox_rounds == 0) {
                    uint32_t _cvt_rn_f16x2_f32_25;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_25) : "f"(ha_pair0.y), "f"(ha_pair0.x));
                    h_words[0] = _cvt_rn_f16x2_f32_25;
                    uint32_t _cvt_rn_f16x2_f32_26;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_26) : "f"(ha_pair0_0.y), "f"(ha_pair0_0.x));
                    h_words[1] = _cvt_rn_f16x2_f32_26;
                }
                reinterpret_cast<int2*>(state + (snap_head + (long long)v_row_a * state_stride_p2 + (long long)k_col0))[0] = reinterpret_cast<int2*>(h_words)[0];
                if (philox_rounds != 0) {
                    unsigned int w0_1 = ((high != 0) ? h_noise[6] : h_noise[4]);
                    unsigned int w1_1 = ((high != 0) ? h_noise[7] : h_noise[5]);
                    uint32_t _cvt_rs_x2_f32_27;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_27) : "f"(ha_pair0_1.y), "f"(ha_pair0_1.x), "r"(w0_1 & 536813567));
                    h_words[0] = _cvt_rs_x2_f32_27;
                    uint32_t _cvt_rs_x2_f32_28;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_28) : "f"(ha_pair0_2.y), "f"(ha_pair0_2.x), "r"(w1_1 & 536813567));
                    h_words[1] = _cvt_rs_x2_f32_28;
                }
                if (philox_rounds == 0) {
                    uint32_t _cvt_rn_f16x2_f32_29;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_29) : "f"(ha_pair0_1.y), "f"(ha_pair0_1.x));
                    h_words[0] = _cvt_rn_f16x2_f32_29;
                    uint32_t _cvt_rn_f16x2_f32_30;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_30) : "f"(ha_pair0_2.y), "f"(ha_pair0_2.x));
                    h_words[1] = _cvt_rn_f16x2_f32_30;
                }
                reinterpret_cast<int2*>(state + (snap_head + (long long)v_row_b * state_stride_p2 + (long long)k_col0))[0] = reinterpret_cast<int2*>(h_words)[0];
            }
        }
        if (write_state_slot_valid != 0) {
            if (disable_state_update == 0) {
                if (use_ssm_state_indices == 0) {
                    int high_1 = lane_local >> 4;
                    if (philox_rounds != 0) {
                        unsigned int key0_1 = seed_lo;
                        unsigned int key1_1 = seed_hi;
                        long long o_2 = read_state_head_base + (long long)v_row_a * state_stride_p2 + (long long)((lane_local & 15) * 4);
                        unsigned long long p0_2 = (unsigned long long)(unsigned int)o_2 * 3528531795ull;
                        h_noise[0] = (unsigned int)(o_2 >> 32ll) ^ key0_1;
                        h_noise[2] = (unsigned int)(p0_2 >> 32ull) ^ key1_1;
                        h_noise[3] = (unsigned int)p0_2;
                        long long o_3 = read_state_head_base + (long long)v_row_a * state_stride_p2 + (long long)((lane_local & 15) * 4) + (long long)(v_row_b - v_row_a) * state_stride_p2;
                        unsigned long long p0_3 = (unsigned long long)(unsigned int)o_3 * 3528531795ull;
                        h_noise[4] = (unsigned int)(o_3 >> 32ll) ^ key0_1;
                        h_noise[6] = (unsigned int)(p0_3 >> 32ull) ^ key1_1;
                        h_noise[7] = (unsigned int)p0_3;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long q0_2 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long q1_2 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(q1_2 >> 32ull) ^ key0_1;
                        h_noise[1] = (unsigned int)q1_2;
                        h_noise[2] = (unsigned int)(q0_2 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)q0_2;
                        unsigned long long q0_3 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long q1_3 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(q1_3 >> 32ull) ^ key0_1;
                        h_noise[5] = (unsigned int)q1_3;
                        h_noise[6] = (unsigned int)(q0_3 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)q0_3;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long s0_6 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_6 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_6 >> 32ull) ^ h_noise[1] ^ key0_1;
                        h_noise[1] = (unsigned int)s1_6;
                        h_noise[2] = (unsigned int)(s0_6 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)s0_6;
                        unsigned long long s0_7 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_7 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_7 >> 32ull) ^ h_noise[5] ^ key0_1;
                        h_noise[5] = (unsigned int)s1_7;
                        h_noise[6] = (unsigned int)(s0_7 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)s0_7;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long s0_8 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_8 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_8 >> 32ull) ^ h_noise[1] ^ key0_1;
                        h_noise[1] = (unsigned int)s1_8;
                        h_noise[2] = (unsigned int)(s0_8 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)s0_8;
                        unsigned long long s0_9 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_9 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_9 >> 32ull) ^ h_noise[5] ^ key0_1;
                        h_noise[5] = (unsigned int)s1_9;
                        h_noise[6] = (unsigned int)(s0_9 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)s0_9;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long s0_10 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_10 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_10 >> 32ull) ^ h_noise[1] ^ key0_1;
                        h_noise[1] = (unsigned int)s1_10;
                        h_noise[2] = (unsigned int)(s0_10 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)s0_10;
                        unsigned long long s0_11 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_11 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_11 >> 32ull) ^ h_noise[5] ^ key0_1;
                        h_noise[5] = (unsigned int)s1_11;
                        h_noise[6] = (unsigned int)(s0_11 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)s0_11;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        #pragma unroll 1
                        for (int _round_1 = 5; _round_1 < philox_rounds; _round_1++) {
                            unsigned long long t0_2 = (unsigned long long)h_noise[0] * 3528531795ull;
                            unsigned long long t1_2 = (unsigned long long)h_noise[2] * 3449720151ull;
                            h_noise[0] = (unsigned int)(t1_2 >> 32ull) ^ h_noise[1] ^ key0_1;
                            h_noise[1] = (unsigned int)t1_2;
                            h_noise[2] = (unsigned int)(t0_2 >> 32ull) ^ h_noise[3] ^ key1_1;
                            h_noise[3] = (unsigned int)t0_2;
                            unsigned long long t0_3 = (unsigned long long)h_noise[4] * 3528531795ull;
                            unsigned long long t1_3 = (unsigned long long)h_noise[6] * 3449720151ull;
                            h_noise[4] = (unsigned int)(t1_3 >> 32ull) ^ h_noise[5] ^ key0_1;
                            h_noise[5] = (unsigned int)t1_3;
                            h_noise[6] = (unsigned int)(t0_3 >> 32ull) ^ h_noise[7] ^ key1_1;
                            h_noise[7] = (unsigned int)t0_3;
                            key0_1 = key0_1 + 2654435769u;
                            key1_1 = key1_1 + 3144134277u;
                        }
                    }
                    if (philox_rounds != 0) {
                        unsigned int w0_2 = ((high_1 != 0) ? h_noise[2] : h_noise[0]);
                        unsigned int w1_2 = ((high_1 != 0) ? h_noise[3] : h_noise[1]);
                        uint32_t _cvt_rs_x2_f32_31;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_31) : "f"(ha_pair0.y), "f"(ha_pair0.x), "r"(w0_2 & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_31;
                        uint32_t _cvt_rs_x2_f32_32;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_32) : "f"(ha_pair0_0.y), "f"(ha_pair0_0.x), "r"(w1_2 & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_32;
                    }
                    if (philox_rounds == 0) {
                        uint32_t _cvt_rn_f16x2_f32_33;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_33) : "f"(ha_pair0.y), "f"(ha_pair0.x));
                        h_words[0] = _cvt_rn_f16x2_f32_33;
                        uint32_t _cvt_rn_f16x2_f32_34;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_34) : "f"(ha_pair0_0.y), "f"(ha_pair0_0.x));
                        h_words[1] = _cvt_rn_f16x2_f32_34;
                    }
                    reinterpret_cast<int2*>(state + (write_state_head_base + (long long)v_row_a * state_stride_p2 + (long long)k_col0))[0] = reinterpret_cast<int2*>(h_words)[0];
                    if (philox_rounds != 0) {
                        unsigned int w0_3 = ((high_1 != 0) ? h_noise[6] : h_noise[4]);
                        unsigned int w1_3 = ((high_1 != 0) ? h_noise[7] : h_noise[5]);
                        uint32_t _cvt_rs_x2_f32_35;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_35) : "f"(ha_pair0_1.y), "f"(ha_pair0_1.x), "r"(w0_3 & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_35;
                        uint32_t _cvt_rs_x2_f32_36;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_36) : "f"(ha_pair0_2.y), "f"(ha_pair0_2.x), "r"(w1_3 & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_36;
                    }
                    if (philox_rounds == 0) {
                        uint32_t _cvt_rn_f16x2_f32_37;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_37) : "f"(ha_pair0_1.y), "f"(ha_pair0_1.x));
                        h_words[0] = _cvt_rn_f16x2_f32_37;
                        uint32_t _cvt_rn_f16x2_f32_38;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_38) : "f"(ha_pair0_2.y), "f"(ha_pair0_2.x));
                        h_words[1] = _cvt_rn_f16x2_f32_38;
                    }
                    reinterpret_cast<int2*>(state + (write_state_head_base + (long long)v_row_b * state_stride_p2 + (long long)k_col0))[0] = reinterpret_cast<int2*>(h_words)[0];
                }
            }
        }
    }
}

} // extern "C"
