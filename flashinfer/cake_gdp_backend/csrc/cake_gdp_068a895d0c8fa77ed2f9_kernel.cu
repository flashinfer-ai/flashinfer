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

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_gdp_068a895d0c8fa77ed2f9(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, int* __restrict__ ssm_state_indices, CakeFastDivmod heads, int num_tokens, float scale, int disable_state_update, int use_ssm_state_indices, long long state_size_p0, long long state_stride_p0, long long state_stride_p1, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long state_stride_p2, long long out_stride_p0, long long out_stride_p1, long long out_stride_p2, long long ssm_state_indices_stride_p0, long long ssm_state_indices_stride_p1)
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
    float* smem_v0 = reinterpret_cast<float*>(smem_raw + 0);
    const int smem_v0_addr = smem + 0;
    float* sQ = reinterpret_cast<float*>(smem_raw + 4096);
    const int sQ_addr = smem + 4096;
    float* sK = reinterpret_cast<float*>(smem_raw + 4608);
    const int sK_addr = smem + 4608;
    float* sV = reinterpret_cast<float*>(smem_raw + 6144);
    const int sV_addr = smem + 6144;
    float* sAlpha = reinterpret_cast<float*>(smem_raw + 6240);
    const int sAlpha_addr = smem + 6240;
    float* sBeta = reinterpret_cast<float*>(smem_raw + 6244);
    const int sBeta_addr = smem + 6244;

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
        #pragma unroll
        for (int copy_iter = 0; copy_iter < 2; copy_iter++) {
            int copy_seg = copy_iter * 128 + tid;
            int copy_row = copy_seg / 32;
            int copy_k_vec = copy_seg - copy_row * 32;
            int copy_v_row = split_v_base + copy_row;
            int copy_k_base = copy_k_vec * 4;
            int copy_dst = smem_v0_addr + (unsigned int)((copy_row * 128 + copy_k_base) * 4);
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(copy_dst), "l"(state + (read_state_head_base + (long long)copy_v_row * state_stride_p2 + (long long)copy_k_base)));
        }
        asm volatile("cp.async.commit_group;");
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
            float _warp_reduce_2 = sum_k;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
            sum_k = _warp_reduce_2;
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
                float _exp_4 = __expf(-pre_s[1]);
                float _rcp_3 = approx_rcp(1.0f + _exp_4);
                sBeta[pre_m] = _rcp_3;
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
            float _warp_reduce_5 = sum_q;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_5 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_5, offset);
            sum_q = _warp_reduce_5;
            float _rsqrt_6 = rsqrtf(sum_q + 1e-06f);
            q_norm = _rsqrt_6 * scale;
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
                    float _exp_8 = __expf(x);
                    float _log2_7;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_7) : "f"(1.0f + _exp_8));
                    softplus_x = _log2_7 * 0.6931472f;
                }
                float _exp_9 = __expf(pre_s[4]);
                float g_log = -_exp_9 * softplus_x;
                float _exp_10 = __expf(g_log);
                sAlpha[pre_t] = _exp_10;
            }
        }
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        #pragma unroll
        for (int row_group = 0; row_group < 8; row_group += 4) {
            int local_row = row_group + warp_local;
            int v_row = split_v_base + local_row;
            float h_pair0[2];
            h_pair0[0] = smem_v0[local_row * 128 + k_col0];
            h_pair0[1] = smem_v0[local_row * 128 + k_col0 + 1];
            float h_pair0_0[2];
            h_pair0_0[0] = smem_v0[local_row * 128 + k_col0 + 2];
            h_pair0_0[1] = smem_v0[local_row * 128 + k_col0 + 3];
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
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(mul_1_2_1) : "f"(h_pair0_0[0]), "f"(k_pair0_x_8));
                float mul_2_2_1;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(mul_2_2_1) : "f"(h_pair0_0[1]), "f"(k_pair0_y_8));
                h_pair0_0[0] = mul_1_2_1;
                h_pair0_0[1] = mul_2_2_1;
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
                    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_1_2_1) : "f"(h_pair0_0[0]), "f"(k_pair0_x_9), "f"(k_pair0_x_10));
                    float fma_2_2_1;
                    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_2_2_1) : "f"(h_pair0_0[1]), "f"(k_pair0_y_9), "f"(k_pair0_y_10));
                    k_pair0_x_10 = fma_1_2_1;
                    k_pair0_y_10 = fma_2_2_1;
                    float sum_hk = k_pair0_x_10 + k_pair0_y_10;
                    float _warp_reduce_11 = sum_hk;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_11 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_11, offset);
                    sum_hk = _warp_reduce_11;
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
                    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_1_4) : "f"(k_pair0_x_9), "f"(k_pair0_x_11), "f"(h_pair0_0[0]));
                    float fma_2_4;
                    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_2_4) : "f"(k_pair0_y_9), "f"(k_pair0_y_11), "f"(h_pair0_0[1]));
                    h_pair0_0[0] = fma_1_4;
                    h_pair0_0[1] = fma_2_4;
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
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_1_5_1) : "f"(h_pair0_0[0]), "f"(k_pair0_x_13), "f"(k_pair0_x_14));
                float fma_2_5_1;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(fma_2_5_1) : "f"(h_pair0_0[1]), "f"(k_pair0_y_13), "f"(k_pair0_y_14));
                k_pair0_x_14 = fma_1_5_1;
                k_pair0_y_14 = fma_2_5_1;
                float sum_hq = k_pair0_x_14 + k_pair0_y_14;
                float _warp_reduce_12 = sum_hq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_12 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_12, offset);
                sum_hq = _warp_reduce_12;
                if (lane_local == 0) {
                    long long out_index = (long long)n * out_stride_p0 + (long long)t * out_stride_p1 + (long long)h * out_stride_p2;
                    out[out_index + (long long)v_row] = sum_hq;
                }
                if (use_ssm_state_indices != 0) {
                    int ssm_slot = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)t * ssm_state_indices_stride_p1];
                    if (ssm_slot >= 0 && state_pool_size > (long long)ssm_slot) {
                        r_h[0] = h_pair0[0];
                        r_h[1] = h_pair0[1];
                        r_h[2] = h_pair0_0[0];
                        r_h[3] = h_pair0_0[1];
                        {
                            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
                            *reinterpret_cast<float4*>(state + (long long)ssm_slot * state_stride_p0 + (long long)h * state_stride_p1 + (long long)v_row * state_stride_p2 + (long long)k_col0) = _v4;
                        }
                    }
                }
            }
            if (write_state_slot_valid != 0) {
                if (disable_state_update == 0) {
                    if (use_ssm_state_indices == 0) {
                        r_h[0] = h_pair0[0];
                        r_h[1] = h_pair0[1];
                        r_h[2] = h_pair0_0[0];
                        r_h[3] = h_pair0_0[1];
                        {
                            float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
                            *reinterpret_cast<float4*>(state + write_state_head_base + (long long)v_row * state_stride_p2 + (long long)k_col0) = _v4;
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"
