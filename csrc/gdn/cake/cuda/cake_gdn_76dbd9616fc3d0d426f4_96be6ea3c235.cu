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
// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_gdn_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128
#ifndef H
#error "H is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef HV
#error "HV is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef SCALE
#error "SCALE is a downstream specialization of this program; define it on the compile line"
#endif

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_gdn_76dbd9616fc3d0d426f4(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, long long state_size_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int linear_block = blockIdx.x;
    int state_head = linear_block / 8;
    int split = linear_block - state_head * 8;
    int n = state_head / HV;
    int h = state_head - n * HV;
    int lane_local = lane;
    int warp_local = warp;
    int qk_h = h / (HV / H);
    int qk_base = (n * H + qk_h) * 128;
    int vh_base = (n * HV + h) * 128;
    long long state_pool_size = state_size_p0;
    int read_state_slot = initial_state_indices[n];
    int read_state_slot_valid = ((read_state_slot >= 0 && state_pool_size > (long long)read_state_slot) ? 1 : 0);
    int safe_read_state_slot = ((read_state_slot_valid != 0) ? read_state_slot : 0);
    int write_state_slot = output_state_indices[n];
    int write_state_slot_valid = ((write_state_slot >= 0 && state_pool_size > (long long)write_state_slot) ? 1 : 0);
    int safe_write_state_slot = ((write_state_slot_valid != 0) ? write_state_slot : 0);
    long long read_state_head_base = ((long long)safe_read_state_slot * (long long)HV + (long long)h) * 16384;
    long long write_state_head_base = ((long long)safe_write_state_slot * (long long)HV + (long long)h) * 16384;
    int split_v_base = split * 16;
    int k_start = lane_local * 4;
    float r_q[4];
    float r_k[4];
    float r_h[4];
    float r_hs[16];
    float r_vrow[4];
    float2 r_q_pair0;
    float2 r_q_pair1;
    float2 r_k_pair0;
    float2 r_k_pair1;
    float r_A_log = A_log[h];
    float r_a = (float)a[n * HV + h];
    float r_dt_bias = dt_bias[h];
    float r_b = (float)b[n * HV + h];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        r_hs[i] = 0.0f;
    }
    if (read_state_slot_valid != 0) {
        #pragma unroll
        for (int rg = 0; rg < 4; rg++) {
            int load_v_row = split_v_base + rg * 4 + warp_local;
            {
                float4 _v4 = *reinterpret_cast<const float4*>(state + read_state_head_base + (long long)load_v_row * 128 + (long long)k_start);
                r_hs[rg * 4 + 0] = _v4.x;
                r_hs[rg * 4 + 1] = _v4.y;
                r_hs[rg * 4 + 2] = _v4.z;
                r_hs[rg * 4 + 3] = _v4.w;
            }
        }
    }
    #pragma unroll
    for (int rg_1 = 0; rg_1 < 4; rg_1++) {
        r_vrow[rg_1] = (float)v[vh_base + split_v_base + rg_1 * 4 + warp_local];
    }
    {
        uint2 _vld_1;
        _vld_1 = *reinterpret_cast<const uint2*>(q + qk_base + k_start);
        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&r_q[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
            (&r_q[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
        }
    }
    {
        uint2 _vld_2;
        _vld_2 = *reinterpret_cast<const uint2*>(k + qk_base + k_start);
        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&r_k[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
            (&r_k[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
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
    float2 sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair0, q_raw_pair0, _f2_4);
    sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair1, q_raw_pair1, sum_q_pair);
    float2 _f2_5 = make_float2(0.0f, 0.0f);
    float2 sum_k_pair = fma_f32x2_rn_ftz(k_raw_pair0, k_raw_pair0, _f2_5);
    sum_k_pair = fma_f32x2_rn_ftz(k_raw_pair1, k_raw_pair1, sum_k_pair);
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
    float q_norm = cake_f32_mul_f64const_rn(_rsqrt_0, SCALE);
    float _rsqrt_1 = rsqrtf(sum_k + 1e-06f);
    float k_norm = _rsqrt_1;
    float2 _f2_6 = make_float2(q_norm, q_norm);
    float2 q_norm_pair = _f2_6;
    float2 _f2_7 = make_float2(k_norm, k_norm);
    float2 k_norm_pair = _f2_7;
    float2 _mul_f32x2_0;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&q_raw_pair0), "l"(*(const unsigned long long*)&q_norm_pair));
    r_q_pair0 = _mul_f32x2_0;
    float2 _mul_f32x2_1;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&q_raw_pair1), "l"(*(const unsigned long long*)&q_norm_pair));
    r_q_pair1 = _mul_f32x2_1;
    float2 _mul_f32x2_2;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&k_raw_pair0), "l"(*(const unsigned long long*)&k_norm_pair));
    r_k_pair0 = _mul_f32x2_2;
    float2 _mul_f32x2_3;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&k_raw_pair1), "l"(*(const unsigned long long*)&k_norm_pair));
    r_k_pair1 = _mul_f32x2_3;
    float decay_lane = 0.0f;
    float beta_lane = 0.0f;
    if (lane_local == 0) {
        float x = r_a + r_dt_bias;
        float softplus_x = x;
        if (x <= 20.0f) {
            float _expf_0 = __expf(x);
            float _log2_0;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(1.0f + _expf_0));
            softplus_x = _log2_0 * 0.6931471805599453f;
        }
        float _expf_1 = __expf(-r_b);
        float _rcp_0 = approx_rcp(1.0f + _expf_1);
        beta_lane = _rcp_0;
        float _expf_2 = __expf(r_A_log);
        float g_log = (-_expf_2) * softplus_x;
        float _expf_3 = __expf(g_log);
        decay_lane = _expf_3;
    }
    float _shfl_0 = __shfl_sync(0xFFFFFFFF, decay_lane, 0);
    float decay_val = _shfl_0;
    float _shfl_1 = __shfl_sync(0xFFFFFFFF, beta_lane, 0);
    float beta_val = _shfl_1;
    float2 _f2_8 = make_float2(decay_val, decay_val);
    float2 decay_pair = _f2_8;
    #pragma unroll
    for (int rg_2 = 0; rg_2 < 4; rg_2++) {
        int v_row = split_v_base + rg_2 * 4 + warp_local;
        float2 _f2_9 = make_float2(r_hs[rg_2 * 4], r_hs[rg_2 * 4 + 1]);
        float2 h_src_pair0 = _f2_9;
        float2 _f2_10 = make_float2(r_hs[rg_2 * 4 + 2], r_hs[rg_2 * 4 + 3]);
        float2 h_src_pair1 = _f2_10;
        float2 _mul_f32x2_4;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&h_src_pair0), "l"(*(const unsigned long long*)&decay_pair));
        float2 h_pair0 = _mul_f32x2_4;
        float2 _mul_f32x2_5;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&h_src_pair1), "l"(*(const unsigned long long*)&decay_pair));
        float2 h_pair1 = _mul_f32x2_5;
        float2 _f2_11 = make_float2(0.0f, 0.0f);
        float2 sum_hk_pair = fma_f32x2_rn_ftz(h_pair0, r_k_pair0, _f2_11);
        sum_hk_pair = fma_f32x2_rn_ftz(h_pair1, r_k_pair1, sum_hk_pair);
        float sum_hk = sum_hk_pair.x + sum_hk_pair.y;
        float _warp_reduce_2 = sum_hk;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
        sum_hk = _warp_reduce_2;
        float v_val = r_vrow[rg_2];
        float v_new = (v_val - sum_hk) * beta_val;
        float2 _f2_12 = make_float2(v_new, v_new);
        float2 v_new_pair = _f2_12;
        float2 h_new_pair0 = fma_f32x2_rn_ftz(r_k_pair0, v_new_pair, h_pair0);
        float2 h_new_pair1 = fma_f32x2_rn_ftz(r_k_pair1, v_new_pair, h_pair1);
        r_h[0] = h_new_pair0.x;
        r_h[1] = h_new_pair0.y;
        r_h[2] = h_new_pair1.x;
        r_h[3] = h_new_pair1.y;
        float2 _f2_13 = make_float2(0.0f, 0.0f);
        float2 sum_hq_pair = fma_f32x2_rn_ftz(h_new_pair0, r_q_pair0, _f2_13);
        sum_hq_pair = fma_f32x2_rn_ftz(h_new_pair1, r_q_pair1, sum_hq_pair);
        float sum_hq = sum_hq_pair.x + sum_hq_pair.y;
        if (read_state_slot_valid != 0) {
            if (write_state_slot_valid != 0) {
                {
                    float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
                    *reinterpret_cast<float4*>((state + (write_state_head_base + (long long)(v_row * 128))) + k_start) = _v4;
                }
            }
        }
        float _warp_reduce_3 = sum_hq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
        sum_hq = _warp_reduce_3;
        if (lane_local == 0) {
            out[vh_base + v_row] = ((read_state_slot_valid != 0) ? sum_hq : 0.0f);
        }
    }
}

} // extern "C"
// clang-format on
