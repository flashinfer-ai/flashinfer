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
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
__device__ __forceinline__ int make_warp_uniform_local(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define THREADS 128
#ifndef H
#error "H is a downstream specialization of this program; define it on the compile line"
#elif H != 16
#error "H takes 16 in this program; another value renders other code"
#endif
#ifndef HV
#error "HV is a downstream specialization of this program; define it on the compile line"
#elif HV != 32
#error "HV takes 32 in this program; another value renders other code"
#endif
#ifndef INTERMEDIATE_BATCH_STRIDE
#error "INTERMEDIATE_BATCH_STRIDE is a downstream specialization of this program; define it on the compile line"
#elif INTERMEDIATE_BATCH_STRIDE != 1048576
#error "INTERMEDIATE_BATCH_STRIDE takes 1048576 in this program; another value renders other code"
#endif
#ifndef INTERMEDIATE_TOKEN_STRIDE
#error "INTERMEDIATE_TOKEN_STRIDE is a downstream specialization of this program; define it on the compile line"
#elif INTERMEDIATE_TOKEN_STRIDE != 524288
#error "INTERMEDIATE_TOKEN_STRIDE takes 524288 in this program; another value renders other code"
#endif
#ifndef STRIDED_INPUTS
#error "STRIDED_INPUTS is a downstream specialization of this program; define it on the compile line"
#elif STRIDED_INPUTS != 1
#error "STRIDED_INPUTS takes 1 in this program; another value renders other code"
#endif
#ifndef SCALE
#error "SCALE is a downstream specialization of this program; define it on the compile line"
#endif

__device__ __forceinline__ float2 mul_f32x2_local(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_gdn_decode_pretranspose_mtp_t2_inline_tile8_bb_tok_loads_hoist(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, float* __restrict__ intermediate_state, int* __restrict__ initial_state_indices, long long state_size_p0, long long state_stride_p0, long long state_stride_p1, long long state_stride_p2, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform_local(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int linear_block = blockIdx.x;
    int state_head = linear_block / 16;
    int v_tile = linear_block - state_head * 16;
    int n = state_head / HV;
    int h = state_head - n * HV;
    int lane_local = lane;
    int warp_local = warp;
    int qk_h = h / (HV / H);
    long long state_pool_size = state_size_p0;
    int read_state_slot_raw = initial_state_indices[n];
    int read_state_slot_valid = ((read_state_slot_raw >= 0 && state_pool_size > (long long)read_state_slot_raw) ? 1 : 0);
    int read_state_slot = ((read_state_slot_valid != 0) ? read_state_slot_raw : 0);
    long long read_state_head_base = (long long)read_state_slot * state_stride_p0 + (long long)h * state_stride_p1;
    int v_row_a = v_tile * 8 + warp_local * 2;
    int v_row_b = v_row_a + 1;
    int k_start = lane_local * 4;
    float r_q[4];
    float r_k[4];
    float r_h_a[4];
    float r_h_b[4];
    float r_qh[8];
    float r_kh[8];
    float r_abh[4];
    float r_vh[4];
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
    float2 h_a_pair0 = make_float2(r_h_a[0], r_h_a[1]);
    float2 h_a_pair1 = make_float2(r_h_a[2], r_h_a[3]);
    float2 h_b_pair0 = make_float2(r_h_b[0], r_h_b[1]);
    float2 h_b_pair1 = make_float2(r_h_b[2], r_h_b[3]);
    float r_A_log = A_log[h];
    float r_dt_bias = dt_bias[h];
    #pragma unroll
    for (int t_h = 0; t_h < 2; t_h++) {
        long long qk_base_h = (long long)((n * 2 + t_h) * H + qk_h) * 128;
        long long k_base_h = qk_base_h;
        long long vh_base_h = (long long)((n * 2 + t_h) * HV + h) * 128;
        long long gate_index_h = (long long)((n * 2 + t_h) * HV + h);
        long long b_index_h = gate_index_h;
        {
            qk_base_h = (long long)n * q_stride_p0 + (long long)t_h * q_stride_p1 + (long long)qk_h * q_stride_p2;
            k_base_h = (long long)n * k_stride_p0 + (long long)t_h * k_stride_p1 + (long long)qk_h * k_stride_p2;
            vh_base_h = (long long)n * v_stride_p0 + (long long)t_h * v_stride_p1 + (long long)h * v_stride_p2;
            gate_index_h = (long long)n * a_stride_p0 + (long long)t_h * a_stride_p1 + (long long)h * a_stride_p2;
            b_index_h = (long long)n * b_stride_p0 + (long long)t_h * b_stride_p1 + (long long)h * b_stride_p2;
        }
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(k + k_base_h + (long long)k_start);
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&r_kh[t_h * 4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                (&r_kh[t_h * 4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
            }
        }
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(q + qk_base_h + (long long)k_start);
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&r_qh[t_h * 4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                (&r_qh[t_h * 4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
            }
        }
        r_abh[t_h * 2] = (float)a[gate_index_h];
        r_abh[t_h * 2 + 1] = (float)b[b_index_h];
        r_vh[t_h * 2] = (float)v[vh_base_h + (long long)v_row_a];
        r_vh[t_h * 2 + 1] = (float)v[vh_base_h + (long long)v_row_b];
    }
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
        }
        r_k[0] = r_kh[t * 4];
        r_q[0] = r_qh[t * 4];
        r_k[1] = r_kh[t * 4 + 1];
        r_q[1] = r_qh[t * 4 + 1];
        r_k[2] = r_kh[t * 4 + 2];
        r_q[2] = r_qh[t * 4 + 2];
        r_k[3] = r_kh[t * 4 + 3];
        r_q[3] = r_qh[t * 4 + 3];
        float2 q_pair0 = make_float2(r_q[0], r_q[1]);
        float2 q_pair1 = make_float2(r_q[2], r_q[3]);
        float2 k_pair0 = make_float2(r_k[0], r_k[1]);
        float2 k_pair1 = make_float2(r_k[2], r_k[3]);
        float2 sum_q_pair = fma_f32x2_rn_ftz(q_pair0, q_pair0, make_float2(0.0f, 0.0f));
        sum_q_pair = fma_f32x2_rn_ftz(q_pair1, q_pair1, sum_q_pair);
        float2 sum_k_pair = fma_f32x2_rn_ftz(k_pair0, k_pair0, make_float2(0.0f, 0.0f));
        sum_k_pair = fma_f32x2_rn_ftz(k_pair1, k_pair1, sum_k_pair);
        float sum_q = sum_q_pair.x + sum_q_pair.y;
        float sum_k = sum_k_pair.x + sum_k_pair.y;
        float _warp_reduce_4 = sum_q;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_4 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_4, offset);
        sum_q = _warp_reduce_4;
        float _warp_reduce_5 = sum_k;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_5 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_5, offset);
        sum_k = _warp_reduce_5;
        float _rsqrt_6 = rsqrtf(sum_q + 1e-06f);
        float inv_q = cake_f32_mul_f64const_rn(_rsqrt_6, SCALE);
        float inv_k = rsqrtf(sum_k + 1e-06f);
        float x = r_abh[t * 2] + r_dt_bias;
        float _exp_8 = __expf(x);
        float _log2_7;
        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_7) : "f"(1.0f + _exp_8));
        float softplus_val = _log2_7 * 0.6931472f;
        float softplus_x = ((x <= 20.0f) ? softplus_val : x);
        float _exp_9 = __expf(-r_abh[t * 2 + 1]);
        float beta_val = approx_rcp(1.0f + _exp_9);
        float _exp_10 = __expf(r_A_log);
        float decay_val = __expf((-_exp_10) * softplus_x);
        float2 decay_pair = make_float2(decay_val, decay_val);
        h_a_pair0 = mul_f32x2_local(h_a_pair0, decay_pair);
        h_a_pair1 = mul_f32x2_local(h_a_pair1, decay_pair);
        h_b_pair0 = mul_f32x2_local(h_b_pair0, decay_pair);
        h_b_pair1 = mul_f32x2_local(h_b_pair1, decay_pair);
        float2 sum_hk_a_pair = fma_f32x2_rn_ftz(h_a_pair0, k_pair0, make_float2(0.0f, 0.0f));
        sum_hk_a_pair = fma_f32x2_rn_ftz(h_a_pair1, k_pair1, sum_hk_a_pair);
        float2 sum_hk_b_pair = fma_f32x2_rn_ftz(h_b_pair0, k_pair0, make_float2(0.0f, 0.0f));
        sum_hk_b_pair = fma_f32x2_rn_ftz(h_b_pair1, k_pair1, sum_hk_b_pair);
        float sum_hk_a = sum_hk_a_pair.x + sum_hk_a_pair.y;
        float sum_hk_b = sum_hk_b_pair.x + sum_hk_b_pair.y;
        float _warp_reduce_11 = sum_hk_a;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_11 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_11, offset);
        sum_hk_a = _warp_reduce_11 * inv_k;
        float _warp_reduce_12 = sum_hk_b;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_12 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_12, offset);
        sum_hk_b = _warp_reduce_12 * inv_k;
        float v_new_a = (r_vh[t * 2] - sum_hk_a) * beta_val;
        float v_new_b = (r_vh[t * 2 + 1] - sum_hk_b) * beta_val;
        float2 k_scale_a = make_float2(inv_k * v_new_a, inv_k * v_new_a);
        float2 k_scale_b = make_float2(inv_k * v_new_b, inv_k * v_new_b);
        h_a_pair0 = fma_f32x2_rn_ftz(k_pair0, k_scale_a, h_a_pair0);
        h_a_pair1 = fma_f32x2_rn_ftz(k_pair1, k_scale_a, h_a_pair1);
        h_b_pair0 = fma_f32x2_rn_ftz(k_pair0, k_scale_b, h_b_pair0);
        h_b_pair1 = fma_f32x2_rn_ftz(k_pair1, k_scale_b, h_b_pair1);
        r_h_a[0] = h_a_pair0.x;
        r_h_a[1] = h_a_pair0.y;
        r_h_a[2] = h_a_pair1.x;
        r_h_a[3] = h_a_pair1.y;
        r_h_b[0] = h_b_pair0.x;
        r_h_b[1] = h_b_pair0.y;
        r_h_b[2] = h_b_pair1.x;
        r_h_b[3] = h_b_pair1.y;
        if (read_state_slot_valid != 0) {
            long long cache_head_base = (long long)n * (long long)INTERMEDIATE_BATCH_STRIDE + (long long)t * (long long)INTERMEDIATE_TOKEN_STRIDE + (long long)h * 16384;
            {
                float4 _v4 = make_float4(r_h_a[0 + 0], r_h_a[0 + 1], r_h_a[0 + 2], r_h_a[0 + 3]);
                *reinterpret_cast<float4*>(intermediate_state + cache_head_base + (long long)(v_row_a * 128) + (long long)k_start) = _v4;
            }
            {
                float4 _v4 = make_float4(r_h_b[0 + 0], r_h_b[0 + 1], r_h_b[0 + 2], r_h_b[0 + 3]);
                *reinterpret_cast<float4*>(intermediate_state + cache_head_base + (long long)(v_row_b * 128) + (long long)k_start) = _v4;
            }
        }
        float2 sum_hq_a_pair = fma_f32x2_rn_ftz(h_a_pair0, q_pair0, make_float2(0.0f, 0.0f));
        sum_hq_a_pair = fma_f32x2_rn_ftz(h_a_pair1, q_pair1, sum_hq_a_pair);
        float2 sum_hq_b_pair = fma_f32x2_rn_ftz(h_b_pair0, q_pair0, make_float2(0.0f, 0.0f));
        sum_hq_b_pair = fma_f32x2_rn_ftz(h_b_pair1, q_pair1, sum_hq_b_pair);
        float sum_hq_a = sum_hq_a_pair.x + sum_hq_a_pair.y;
        float sum_hq_b = sum_hq_b_pair.x + sum_hq_b_pair.y;
        float _warp_reduce_13 = sum_hq_a;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_13 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_13, offset);
        sum_hq_a = _warp_reduce_13 * inv_q;
        float _warp_reduce_14 = sum_hq_b;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_14 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_14, offset);
        sum_hq_b = _warp_reduce_14 * inv_q;
        if (lane_local == 0 && read_state_slot_valid != 0) {
            out[((n * 2 + t) * HV + h) * 128 + v_row_a] = sum_hq_a;
            out[((n * 2 + t) * HV + h) * 128 + v_row_b] = sum_hq_b;
        }
    }
}

} // extern "C"
// clang-format on
