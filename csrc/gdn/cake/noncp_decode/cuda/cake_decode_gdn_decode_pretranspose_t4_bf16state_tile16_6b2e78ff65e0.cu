// Copyright (c) 2026 by FlashInfer team.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// clang-format off
#include "cake_common.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SQ_OFF 0
#define SMEM_SQ_STAGE_BYTES 2176
#define SMEM_SQ_STRIDE 2176
#define SMEM_SK_OFF 2176
#define SMEM_SK_STAGE_BYTES 2176
#define SMEM_SK_STRIDE 2176
#define SMEM_SSCALAR_OFF 4352
#define SMEM_SSCALAR_STAGE_BYTES 32
#define SMEM_SSCALAR_STRIDE 32
#define SMEM_TOTAL 4480
#define THREADS 128
#define H 4
#define HV 8
#define INTERMEDIATE_BATCH_STRIDE 524288
#define INTERMEDIATE_TOKEN_STRIDE 131072
#define STRIDED_INPUTS 1
#define SCALE 0.08838834764831845



extern "C" {

__global__ __launch_bounds__(128) void
kernel_gdn_decode_pretranspose_t4_bf16state_tile16(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ intermediate_state, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, long long state_stride_p0, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* sQ = reinterpret_cast<float*>(smem_raw + 0);
    const int sQ_addr = smem + 0;
    float* sK = reinterpret_cast<float*>(smem_raw + 2176);
    const int sK_addr = smem + 2176;
    float* sScalar = reinterpret_cast<float*>(smem_raw + 4352);
    const int sScalar_addr = smem + 4352;

    // === Task calls (dependency order) ===
    int linear_block = blockIdx.x;
    int state_head = linear_block / 8;
    int i_v_tile = linear_block - state_head * 8;
    int v_tile_base = i_v_tile * 16;
    int n = state_head / HV;
    int h = state_head - n * HV;
    int lane_local = lane;
    int warp_local = warp;
    int qk_h = h / (HV / H);
    int read_state_slot_raw = initial_state_indices[n];
    int read_state_slot = ((read_state_slot_raw >= 0) ? read_state_slot_raw : 0);
    long long state_slot_stride = state_stride_p0;
    long long read_state_head_base = (long long)read_state_slot * state_slot_stride + (long long)h * 16384;
    int k_start = lane_local * 4;
    int qk_smem_col = k_start;
    int v_row_a = v_tile_base + warp_local * 4;
    int v_row_b = v_row_a + 1;
    int v_row_c = v_row_a + 2;
    int v_row_d = v_row_a + 3;
    float r_q[4];
    float r_k[4];
    float r_h_a[4];
    float r_h_b[4];
    float r_h_c[4];
    float r_h_d[4];
    float r_h[4];
    float r_v[4];
    float r_o[4];
    int t_pre = warp_local;
    {
        long long q_base = (long long)n * q_stride_p0 + (long long)t_pre * q_stride_p1 + (long long)qk_h * q_stride_p2;
        long long k_base = (long long)n * k_stride_p0 + (long long)t_pre * k_stride_p1 + (long long)qk_h * k_stride_p2;
        {
            uint2 _vld_0;
            _vld_0 = *reinterpret_cast<const uint2*>(q + q_base + (long long)k_start);
            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&r_q[0 + _pair * 2])[0]), "=f"((&r_q[0 + _pair * 2])[1])
                    : "r"(_vpairs_0[_pair]));
            }
        }
        {
            uint2 _vld_1;
            _vld_1 = *reinterpret_cast<const uint2*>(k + k_base + (long long)k_start);
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&r_k[0 + _pair * 2])[0]), "=f"((&r_k[0 + _pair * 2])[1])
                    : "r"(_vpairs_1[_pair]));
            }
        }
    }
    float2 _f2_0 = make_float2(r_q[0], r_q[1]);
    float2 q_raw_pair0 = _f2_0;
    float2 _f2_1 = make_float2(r_q[2], r_q[3]);
    float2 q_raw_pair1 = _f2_1;
    float2 _f2_2 = make_float2(r_k[0], r_k[1]);
    float2 k_raw_pair0 = _f2_2;
    float2 _f2_3 = make_float2(r_k[2], r_k[3]);
    float2 k_raw_pair1 = _f2_3;
    float2 _f2_4 = make_float2(0.0f, 0.0f);
    float2 sum_q_pair = fma_f32x2(q_raw_pair0, q_raw_pair0, _f2_4);
    sum_q_pair = fma_f32x2(q_raw_pair1, q_raw_pair1, sum_q_pair);
    float2 _f2_5 = make_float2(0.0f, 0.0f);
    float2 sum_k_pair = fma_f32x2(k_raw_pair0, k_raw_pair0, _f2_5);
    sum_k_pair = fma_f32x2(k_raw_pair1, k_raw_pair1, sum_k_pair);
    float _warp_reduce_0 = sum_q_pair.x + sum_q_pair.y;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    float sum_q = _warp_reduce_0;
    float _warp_reduce_1 = sum_k_pair.x + sum_k_pair.y;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
    float sum_k = _warp_reduce_1;
    float _rsqrt_0 = rsqrtf(sum_q + 1e-06f);
    float q_norm = _rsqrt_0 * SCALE;
    float _rsqrt_1 = rsqrtf(sum_k + 1e-06f);
    float k_norm = _rsqrt_1;
    float2 _f2_6 = make_float2(q_norm, q_norm);
    float2 q_pair0 = mul_f32x2(q_raw_pair0, _f2_6);
    float2 _f2_7 = make_float2(q_norm, q_norm);
    float2 q_pair1 = mul_f32x2(q_raw_pair1, _f2_7);
    float2 _f2_8 = make_float2(k_norm, k_norm);
    float2 k_pair0 = mul_f32x2(k_raw_pair0, _f2_8);
    float2 _f2_9 = make_float2(k_norm, k_norm);
    float2 k_pair1 = mul_f32x2(k_raw_pair1, _f2_9);
    sQ[t_pre * 136 + qk_smem_col] = q_pair0.x;
    sQ[t_pre * 136 + qk_smem_col + 1] = q_pair0.y;
    sQ[t_pre * 136 + qk_smem_col + 2] = q_pair1.x;
    sQ[t_pre * 136 + qk_smem_col + 3] = q_pair1.y;
    sK[t_pre * 136 + qk_smem_col] = k_pair0.x;
    sK[t_pre * 136 + qk_smem_col + 1] = k_pair0.y;
    sK[t_pre * 136 + qk_smem_col + 2] = k_pair1.x;
    sK[t_pre * 136 + qk_smem_col + 3] = k_pair1.y;
    float x = 0.0f;
    float beta_scalar = 0.0f;
    {
        long long a_index = (long long)n * a_stride_p0 + (long long)t_pre * a_stride_p1 + (long long)h * a_stride_p2;
        long long b_index = (long long)n * b_stride_p0 + (long long)t_pre * b_stride_p1 + (long long)h * b_stride_p2;
        x = (float)a[a_index] + dt_bias[h];
        float _expf_0 = __expf(-(float)b[b_index]);
        float _rcp_0 = approx_rcp(1.0f + _expf_0);
        beta_scalar = _rcp_0;
    }
    float _expf_2 = __expf(x);
    float _log2_0;
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(1.0f + _expf_2));
    float softplus_val = _log2_0 * 0.6931471805599453f;
    float softplus_x = ((x <= 20.0f) ? softplus_val : x);
    float _expf_3 = __expf(A_log[h]);
    float _expf_4 = __expf((-_expf_3) * softplus_x);
    float decay_scalar = _expf_4;
    if (lane_local == 0) {
        sScalar[t_pre * 2] = decay_scalar;
        sScalar[t_pre * 2 + 1] = beta_scalar;
    }
    __syncthreads();
    {
        uint2 _vld_2;
        _vld_2 = *reinterpret_cast<const uint2*>(state + read_state_head_base + (long long)(v_row_a * 128) + (long long)k_start);
        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&r_h_a[0 + _pair * 2])[0]), "=f"((&r_h_a[0 + _pair * 2])[1])
                : "r"(_vpairs_2[_pair]));
        }
    }
    {
        uint2 _vld_3;
        _vld_3 = *reinterpret_cast<const uint2*>(state + read_state_head_base + (long long)(v_row_b * 128) + (long long)k_start);
        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&r_h_b[0 + _pair * 2])[0]), "=f"((&r_h_b[0 + _pair * 2])[1])
                : "r"(_vpairs_3[_pair]));
        }
    }
    {
        uint2 _vld_4;
        _vld_4 = *reinterpret_cast<const uint2*>(state + read_state_head_base + (long long)(v_row_c * 128) + (long long)k_start);
        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&r_h_c[0 + _pair * 2])[0]), "=f"((&r_h_c[0 + _pair * 2])[1])
                : "r"(_vpairs_4[_pair]));
        }
    }
    {
        uint2 _vld_5;
        _vld_5 = *reinterpret_cast<const uint2*>(state + read_state_head_base + (long long)(v_row_d * 128) + (long long)k_start);
        uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&r_h_d[0 + _pair * 2])[0]), "=f"((&r_h_d[0 + _pair * 2])[1])
                : "r"(_vpairs_5[_pair]));
        }
    }
    float2 _f2_10 = make_float2(r_h_a[0], r_h_a[1]);
    float2 h_a_pair0 = _f2_10;
    float2 _f2_11 = make_float2(r_h_a[2], r_h_a[3]);
    float2 h_a_pair1 = _f2_11;
    float2 _f2_12 = make_float2(r_h_b[0], r_h_b[1]);
    float2 h_b_pair0 = _f2_12;
    float2 _f2_13 = make_float2(r_h_b[2], r_h_b[3]);
    float2 h_b_pair1 = _f2_13;
    float2 _f2_14 = make_float2(r_h_c[0], r_h_c[1]);
    float2 h_c_pair0 = _f2_14;
    float2 _f2_15 = make_float2(r_h_c[2], r_h_c[3]);
    float2 h_c_pair1 = _f2_15;
    float2 _f2_16 = make_float2(r_h_d[0], r_h_d[1]);
    float2 h_d_pair0 = _f2_16;
    float2 _f2_17 = make_float2(r_h_d[2], r_h_d[3]);
    float2 h_d_pair1 = _f2_17;
    #pragma unroll 1
    for (int t = 0; t < 4; t++) {
        int q_smem_addr = sQ_addr + (unsigned int)((t * 136 + qk_smem_col) * 4);
        int k_smem_addr = sK_addr + (unsigned int)((t * 136 + qk_smem_col) * 4);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&r_q[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_q[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_q[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_q[(0) + 3]))
            : "r"(q_smem_addr));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&r_k[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_k[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_k[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_k[(0) + 3]))
            : "r"(k_smem_addr));
        float2 _f2_18 = make_float2(r_q[0], r_q[1]);
        q_pair0 = _f2_18;
        float2 _f2_19 = make_float2(r_q[2], r_q[3]);
        q_pair1 = _f2_19;
        float2 _f2_20 = make_float2(r_k[0], r_k[1]);
        k_pair0 = _f2_20;
        float2 _f2_21 = make_float2(r_k[2], r_k[3]);
        k_pair1 = _f2_21;
        float decay_val = sScalar[t * 2];
        float beta_val = sScalar[t * 2 + 1];
        float2 _f2_22 = make_float2(decay_val, decay_val);
        float2 decay_pair = _f2_22;
        h_a_pair0 = mul_f32x2(h_a_pair0, decay_pair);
        h_a_pair1 = mul_f32x2(h_a_pair1, decay_pair);
        h_b_pair0 = mul_f32x2(h_b_pair0, decay_pair);
        h_b_pair1 = mul_f32x2(h_b_pair1, decay_pair);
        h_c_pair0 = mul_f32x2(h_c_pair0, decay_pair);
        h_c_pair1 = mul_f32x2(h_c_pair1, decay_pair);
        h_d_pair0 = mul_f32x2(h_d_pair0, decay_pair);
        h_d_pair1 = mul_f32x2(h_d_pair1, decay_pair);
        float2 _f2_23 = make_float2(0.0f, 0.0f);
        float2 sum_hk_a_pair = fma_f32x2(h_a_pair0, k_pair0, _f2_23);
        sum_hk_a_pair = fma_f32x2(h_a_pair1, k_pair1, sum_hk_a_pair);
        float2 _f2_24 = make_float2(0.0f, 0.0f);
        float2 sum_hk_b_pair = fma_f32x2(h_b_pair0, k_pair0, _f2_24);
        sum_hk_b_pair = fma_f32x2(h_b_pair1, k_pair1, sum_hk_b_pair);
        float2 _f2_25 = make_float2(0.0f, 0.0f);
        float2 sum_hk_c_pair = fma_f32x2(h_c_pair0, k_pair0, _f2_25);
        sum_hk_c_pair = fma_f32x2(h_c_pair1, k_pair1, sum_hk_c_pair);
        float2 _f2_26 = make_float2(0.0f, 0.0f);
        float2 sum_hk_d_pair = fma_f32x2(h_d_pair0, k_pair0, _f2_26);
        sum_hk_d_pair = fma_f32x2(h_d_pair1, k_pair1, sum_hk_d_pair);
        float _warp_reduce_2 = sum_hk_a_pair.x + sum_hk_a_pair.y;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
        float sum_hk_a = _warp_reduce_2;
        float _warp_reduce_3 = sum_hk_b_pair.x + sum_hk_b_pair.y;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
        float sum_hk_b = _warp_reduce_3;
        float _warp_reduce_4 = sum_hk_c_pair.x + sum_hk_c_pair.y;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_4 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_4, offset);
        float sum_hk_c = _warp_reduce_4;
        float _warp_reduce_5 = sum_hk_d_pair.x + sum_hk_d_pair.y;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_5 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_5, offset);
        float sum_hk_d = _warp_reduce_5;
        {
            long long v_base = (long long)n * v_stride_p0 + (long long)t * v_stride_p1 + (long long)h * v_stride_p2;
            {
                uint2 _vld_6;
                _vld_6 = *reinterpret_cast<const uint2*>(v + v_base + (long long)v_row_a);
                uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&r_v[0 + _pair * 2])[0]), "=f"((&r_v[0 + _pair * 2])[1])
                        : "r"(_vpairs_6[_pair]));
                }
            }
        }
        float v_new_a = (r_v[0] - sum_hk_a) * beta_val;
        float v_new_b = (r_v[1] - sum_hk_b) * beta_val;
        float v_new_c = (r_v[2] - sum_hk_c) * beta_val;
        float v_new_d = (r_v[3] - sum_hk_d) * beta_val;
        float2 _f2_27 = make_float2(v_new_a, v_new_a);
        h_a_pair0 = fma_f32x2(k_pair0, _f2_27, h_a_pair0);
        float2 _f2_28 = make_float2(v_new_a, v_new_a);
        h_a_pair1 = fma_f32x2(k_pair1, _f2_28, h_a_pair1);
        float2 _f2_29 = make_float2(v_new_b, v_new_b);
        h_b_pair0 = fma_f32x2(k_pair0, _f2_29, h_b_pair0);
        float2 _f2_30 = make_float2(v_new_b, v_new_b);
        h_b_pair1 = fma_f32x2(k_pair1, _f2_30, h_b_pair1);
        float2 _f2_31 = make_float2(v_new_c, v_new_c);
        h_c_pair0 = fma_f32x2(k_pair0, _f2_31, h_c_pair0);
        float2 _f2_32 = make_float2(v_new_c, v_new_c);
        h_c_pair1 = fma_f32x2(k_pair1, _f2_32, h_c_pair1);
        float2 _f2_33 = make_float2(v_new_d, v_new_d);
        h_d_pair0 = fma_f32x2(k_pair0, _f2_33, h_d_pair0);
        float2 _f2_34 = make_float2(v_new_d, v_new_d);
        h_d_pair1 = fma_f32x2(k_pair1, _f2_34, h_d_pair1);
        long long cache_head_base = (long long)n * (long long)INTERMEDIATE_BATCH_STRIDE + (long long)t * (long long)INTERMEDIATE_TOKEN_STRIDE + (long long)h * 16384;
        r_h[0] = h_a_pair0.x;
        r_h[1] = h_a_pair0.y;
        r_h[2] = h_a_pair1.x;
        r_h[3] = h_a_pair1.y;
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(r_h[0 + 0], r_h[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(intermediate_state))[cache_head_base + (long long)(v_row_a * 128) + (long long)k_start]) = _pk2;
        }
        r_h[0] = h_b_pair0.x;
        r_h[1] = h_b_pair0.y;
        r_h[2] = h_b_pair1.x;
        r_h[3] = h_b_pair1.y;
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(r_h[0 + 0], r_h[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(intermediate_state))[cache_head_base + (long long)(v_row_b * 128) + (long long)k_start]) = _pk2;
        }
        r_h[0] = h_c_pair0.x;
        r_h[1] = h_c_pair0.y;
        r_h[2] = h_c_pair1.x;
        r_h[3] = h_c_pair1.y;
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(r_h[0 + 0], r_h[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(intermediate_state))[cache_head_base + (long long)(v_row_c * 128) + (long long)k_start]) = _pk2;
        }
        r_h[0] = h_d_pair0.x;
        r_h[1] = h_d_pair0.y;
        r_h[2] = h_d_pair1.x;
        r_h[3] = h_d_pair1.y;
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(r_h[0 + 0], r_h[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(r_h[0 + 2], r_h[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(intermediate_state))[cache_head_base + (long long)(v_row_d * 128) + (long long)k_start]) = _pk2;
        }
        float2 _f2_35 = make_float2(0.0f, 0.0f);
        float2 sum_hq_a_pair = fma_f32x2(h_a_pair0, q_pair0, _f2_35);
        sum_hq_a_pair = fma_f32x2(h_a_pair1, q_pair1, sum_hq_a_pair);
        float2 _f2_36 = make_float2(0.0f, 0.0f);
        float2 sum_hq_b_pair = fma_f32x2(h_b_pair0, q_pair0, _f2_36);
        sum_hq_b_pair = fma_f32x2(h_b_pair1, q_pair1, sum_hq_b_pair);
        float2 _f2_37 = make_float2(0.0f, 0.0f);
        float2 sum_hq_c_pair = fma_f32x2(h_c_pair0, q_pair0, _f2_37);
        sum_hq_c_pair = fma_f32x2(h_c_pair1, q_pair1, sum_hq_c_pair);
        float2 _f2_38 = make_float2(0.0f, 0.0f);
        float2 sum_hq_d_pair = fma_f32x2(h_d_pair0, q_pair0, _f2_38);
        sum_hq_d_pair = fma_f32x2(h_d_pair1, q_pair1, sum_hq_d_pair);
        float _warp_reduce_6 = sum_hq_a_pair.x + sum_hq_a_pair.y;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_6 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_6, offset);
        float sum_hq_a = _warp_reduce_6;
        float _warp_reduce_7 = sum_hq_b_pair.x + sum_hq_b_pair.y;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_7 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_7, offset);
        float sum_hq_b = _warp_reduce_7;
        float _warp_reduce_8 = sum_hq_c_pair.x + sum_hq_c_pair.y;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_8 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_8, offset);
        float sum_hq_c = _warp_reduce_8;
        float _warp_reduce_9 = sum_hq_d_pair.x + sum_hq_d_pair.y;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_9 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_9, offset);
        float sum_hq_d = _warp_reduce_9;
        if (lane_local == 0) {
            r_o[0] = sum_hq_a;
            r_o[1] = sum_hq_b;
            r_o[2] = sum_hq_c;
            r_o[3] = sum_hq_d;
            {
                uint2 _pk2;
                __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                _pk[0] = __floats2bfloat162_rn(r_o[0 + 0], r_o[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(r_o[0 + 2], r_o[0 + 3]);
                *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[((n * 4 + t) * HV + h) * 128 + v_row_a]) = _pk2;
            }
        }
    }
}

} // extern "C"

#undef H
#undef HV
#undef INTERMEDIATE_BATCH_STRIDE
#undef INTERMEDIATE_TOKEN_STRIDE
#undef CAKE_INF
#undef NUM_MAIN_STAGES
#undef SCALE
#undef SMEM_SK_OFF
#undef SMEM_SK_STAGE_BYTES
#undef SMEM_SK_STRIDE
#undef SMEM_SQ_OFF
#undef SMEM_SQ_STAGE_BYTES
#undef SMEM_SQ_STRIDE
#undef SMEM_SSCALAR_OFF
#undef SMEM_SSCALAR_STAGE_BYTES
#undef SMEM_SSCALAR_STRIDE
#undef SMEM_TOTAL
#undef STRIDED_INPUTS
#undef THREADS
#undef sK_addr
#undef sQ_addr
#undef sScalar_addr
// clang-format on
