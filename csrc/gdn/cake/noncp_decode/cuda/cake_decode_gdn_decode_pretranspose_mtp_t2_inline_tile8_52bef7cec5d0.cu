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
#define THREADS 128
#define H 16
#define HV 32
#define INTERMEDIATE_BATCH_STRIDE 1048576
#define INTERMEDIATE_TOKEN_STRIDE 524288
#define STRIDED_INPUTS 1
#define SCALE 0.08838834764831845



extern "C" {

__global__ __launch_bounds__(128) void
kernel_gdn_decode_pretranspose_mtp_t2_inline_tile8(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, float* __restrict__ intermediate_state, int* __restrict__ initial_state_indices, long long state_stride_p0, long long state_stride_p1, long long state_stride_p2, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int linear_block = blockIdx.x;
    int state_head = linear_block / 16;
    int v_tile = linear_block - state_head * 16;
    int n = state_head / HV;
    int h = state_head - n * HV;
    int lane_local = lane;
    int warp_local = warp;
    int qk_h = h / (HV / H);
    int read_state_slot_raw = initial_state_indices[n];
    int read_state_slot = ((read_state_slot_raw >= 0) ? read_state_slot_raw : 0);
    long long read_state_head_base = (long long)read_state_slot * state_stride_p0 + (long long)h * state_stride_p1;
    int v_row_a = v_tile * 8 + warp_local * 2;
    int v_row_b = v_row_a + 1;
    int k_start = lane_local * 4;
    float r_q[4];
    float r_k[4];
    float r_h_a[4];
    float r_h_b[4];
    {
        float4 _v4 = *reinterpret_cast<const float4*>(state + read_state_head_base + (long long)v_row_a * state_stride_p2 + (long long)k_start);
        r_h_a[0 + 0] = _v4.x;
        r_h_a[0 + 1] = _v4.y;
        r_h_a[0 + 2] = _v4.z;
        r_h_a[0 + 3] = _v4.w;
    }
    {
        float4 _v4 = *reinterpret_cast<const float4*>(state + read_state_head_base + (long long)v_row_b * state_stride_p2 + (long long)k_start);
        r_h_b[0 + 0] = _v4.x;
        r_h_b[0 + 1] = _v4.y;
        r_h_b[0 + 2] = _v4.z;
        r_h_b[0 + 3] = _v4.w;
    }
    float2 _f2_0 = make_float2(r_h_a[0], r_h_a[1]);
    float2 h_a_pair0 = _f2_0;
    float2 _f2_1 = make_float2(r_h_a[2], r_h_a[3]);
    float2 h_a_pair1 = _f2_1;
    float2 _f2_2 = make_float2(r_h_b[0], r_h_b[1]);
    float2 h_b_pair0 = _f2_2;
    float2 _f2_3 = make_float2(r_h_b[2], r_h_b[3]);
    float2 h_b_pair1 = _f2_3;
    float r_A_log = A_log[h];
    float r_dt_bias = dt_bias[h];
    #pragma unroll
    for (int t = 0; t < 2; t++) {
        long long qk_base = (long long)((n * 2 + t) * H + qk_h) * 128;
        long long vh_base = (long long)((n * 2 + t) * HV + h) * 128;
        long long gate_index = (long long)((n * 2 + t) * HV + h);
        long long b_index = gate_index;
        {
            qk_base = (long long)n * q_stride_p0 + (long long)t * q_stride_p1 + (long long)qk_h * q_stride_p2;
            long long k_base = (long long)n * k_stride_p0 + (long long)t * k_stride_p1 + (long long)qk_h * k_stride_p2;
            vh_base = (long long)n * v_stride_p0 + (long long)t * v_stride_p1 + (long long)h * v_stride_p2;
            gate_index = (long long)n * a_stride_p0 + (long long)t * a_stride_p1 + (long long)h * a_stride_p2;
            b_index = (long long)n * b_stride_p0 + (long long)t * b_stride_p1 + (long long)h * b_stride_p2;
            {
                uint2 _vld_2;
                _vld_2 = *reinterpret_cast<const uint2*>(k + k_base + (long long)k_start);
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&r_k[0 + _pair * 2])[0]), "=f"((&r_k[0 + _pair * 2])[1])
                        : "r"(_vpairs_2[_pair]));
                }
            }
        }
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(q + qk_base + (long long)k_start);
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&r_q[0 + _pair * 2])[0]), "=f"((&r_q[0 + _pair * 2])[1])
                    : "r"(_vpairs_3[_pair]));
            }
        }
        float2 _f2_4 = make_float2(r_q[0], r_q[1]);
        float2 q_pair0 = _f2_4;
        float2 _f2_5 = make_float2(r_q[2], r_q[3]);
        float2 q_pair1 = _f2_5;
        float2 _f2_6 = make_float2(r_k[0], r_k[1]);
        float2 k_pair0 = _f2_6;
        float2 _f2_7 = make_float2(r_k[2], r_k[3]);
        float2 k_pair1 = _f2_7;
        float2 _f2_8 = make_float2(0.0f, 0.0f);
        float2 sum_q_pair = fma_f32x2(q_pair0, q_pair0, _f2_8);
        sum_q_pair = fma_f32x2(q_pair1, q_pair1, sum_q_pair);
        float2 _f2_9 = make_float2(0.0f, 0.0f);
        float2 sum_k_pair = fma_f32x2(k_pair0, k_pair0, _f2_9);
        sum_k_pair = fma_f32x2(k_pair1, k_pair1, sum_k_pair);
        float sum_q = sum_q_pair.x + sum_q_pair.y;
        float sum_k = sum_k_pair.x + sum_k_pair.y;
        float _warp_reduce_0 = sum_q;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        sum_q = _warp_reduce_0;
        float _warp_reduce_1 = sum_k;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        sum_k = _warp_reduce_1;
        float _rsqrt_0 = rsqrtf(sum_q + 1e-06f);
        float inv_q = _rsqrt_0 * SCALE;
        float _rsqrt_1 = rsqrtf(sum_k + 1e-06f);
        float inv_k = _rsqrt_1;
        float x = (float)a[gate_index] + r_dt_bias;
        float softplus_x = x;
        if (x <= 20.0f) {
            float _expf_0 = __expf(x);
            float _log2_0;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(1.0f + _expf_0));
            softplus_x = _log2_0 * 0.6931471805599453f;
        }
        float _expf_1 = __expf(-(float)b[b_index]);
        float _rcp_0 = approx_rcp(1.0f + _expf_1);
        float beta_val = _rcp_0;
        float _expf_2 = __expf(r_A_log);
        float _expf_3 = __expf((-_expf_2) * softplus_x);
        float decay_val = _expf_3;
        float2 _f2_10 = make_float2(decay_val, decay_val);
        float2 decay_pair = _f2_10;
        h_a_pair0 = mul_f32x2(h_a_pair0, decay_pair);
        h_a_pair1 = mul_f32x2(h_a_pair1, decay_pair);
        h_b_pair0 = mul_f32x2(h_b_pair0, decay_pair);
        h_b_pair1 = mul_f32x2(h_b_pair1, decay_pair);
        float2 _f2_11 = make_float2(0.0f, 0.0f);
        float2 sum_hk_a_pair = fma_f32x2(h_a_pair0, k_pair0, _f2_11);
        sum_hk_a_pair = fma_f32x2(h_a_pair1, k_pair1, sum_hk_a_pair);
        float2 _f2_12 = make_float2(0.0f, 0.0f);
        float2 sum_hk_b_pair = fma_f32x2(h_b_pair0, k_pair0, _f2_12);
        sum_hk_b_pair = fma_f32x2(h_b_pair1, k_pair1, sum_hk_b_pair);
        float sum_hk_a = sum_hk_a_pair.x + sum_hk_a_pair.y;
        float sum_hk_b = sum_hk_b_pair.x + sum_hk_b_pair.y;
        float _warp_reduce_2 = sum_hk_a;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
        sum_hk_a = _warp_reduce_2 * inv_k;
        float _warp_reduce_3 = sum_hk_b;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
        sum_hk_b = _warp_reduce_3 * inv_k;
        float v_new_a = ((float)v[vh_base + (long long)v_row_a] - sum_hk_a) * beta_val;
        float v_new_b = ((float)v[vh_base + (long long)v_row_b] - sum_hk_b) * beta_val;
        float2 _f2_13 = make_float2(inv_k * v_new_a, inv_k * v_new_a);
        float2 k_scale_a = _f2_13;
        float2 _f2_14 = make_float2(inv_k * v_new_b, inv_k * v_new_b);
        float2 k_scale_b = _f2_14;
        h_a_pair0 = fma_f32x2(k_pair0, k_scale_a, h_a_pair0);
        h_a_pair1 = fma_f32x2(k_pair1, k_scale_a, h_a_pair1);
        h_b_pair0 = fma_f32x2(k_pair0, k_scale_b, h_b_pair0);
        h_b_pair1 = fma_f32x2(k_pair1, k_scale_b, h_b_pair1);
        if (read_state_slot_raw >= 0) {
            long long cache_head_base = (long long)n * (long long)INTERMEDIATE_BATCH_STRIDE + (long long)t * (long long)INTERMEDIATE_TOKEN_STRIDE + (long long)h * 16384;
            r_h_a[0] = h_a_pair0.x;
            r_h_a[1] = h_a_pair0.y;
            r_h_a[2] = h_a_pair1.x;
            r_h_a[3] = h_a_pair1.y;
            {
                float4 _v4 = make_float4(r_h_a[0 + 0], r_h_a[0 + 1], r_h_a[0 + 2], r_h_a[0 + 3]);
                *reinterpret_cast<float4*>(intermediate_state + cache_head_base + (long long)(v_row_a * 128) + (long long)k_start) = _v4;
            }
            r_h_b[0] = h_b_pair0.x;
            r_h_b[1] = h_b_pair0.y;
            r_h_b[2] = h_b_pair1.x;
            r_h_b[3] = h_b_pair1.y;
            {
                float4 _v4 = make_float4(r_h_b[0 + 0], r_h_b[0 + 1], r_h_b[0 + 2], r_h_b[0 + 3]);
                *reinterpret_cast<float4*>(intermediate_state + cache_head_base + (long long)(v_row_b * 128) + (long long)k_start) = _v4;
            }
        }
        float2 _f2_15 = make_float2(0.0f, 0.0f);
        float2 sum_hq_a_pair = fma_f32x2(h_a_pair0, q_pair0, _f2_15);
        sum_hq_a_pair = fma_f32x2(h_a_pair1, q_pair1, sum_hq_a_pair);
        float2 _f2_16 = make_float2(0.0f, 0.0f);
        float2 sum_hq_b_pair = fma_f32x2(h_b_pair0, q_pair0, _f2_16);
        sum_hq_b_pair = fma_f32x2(h_b_pair1, q_pair1, sum_hq_b_pair);
        float sum_hq_a = sum_hq_a_pair.x + sum_hq_a_pair.y;
        float sum_hq_b = sum_hq_b_pair.x + sum_hq_b_pair.y;
        float _warp_reduce_4 = sum_hq_a;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_4 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_4, offset);
        sum_hq_a = _warp_reduce_4 * inv_q;
        float _warp_reduce_5 = sum_hq_b;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_5 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_5, offset);
        sum_hq_b = _warp_reduce_5 * inv_q;
        if (lane_local == 0) {
            if (read_state_slot_raw >= 0) {
                out[((n * 2 + t) * HV + h) * 128 + v_row_a] = sum_hq_a;
                out[((n * 2 + t) * HV + h) * 128 + v_row_b] = sum_hq_b;
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
#undef STRIDED_INPUTS
#undef THREADS
// clang-format on
