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
#define SMEM_SSTATE_OFF 0
#define SMEM_SSTATE_STAGE_BYTES 8192
#define SMEM_SSTATE_STRIDE 8192
#define SMEM_SQ_OFF 8192
#define SMEM_SQ_STAGE_BYTES 2048
#define SMEM_SQ_STRIDE 2048
#define SMEM_SK_OFF 10240
#define SMEM_SK_STAGE_BYTES 2048
#define SMEM_SK_STRIDE 2048
#define SMEM_SV_OFF 12288
#define SMEM_SV_STAGE_BYTES 2048
#define SMEM_SV_STRIDE 2048
#define SMEM_SSCALAR_OFF 14336
#define SMEM_SSCALAR_STAGE_BYTES 32
#define SMEM_SSCALAR_STRIDE 32
#define SMEM_TOTAL 14464
#define THREADS 128
#ifndef H
#error "H is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef HV
#error "HV is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef T_STEPS
#error "T_STEPS is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef UPDATE_STATE
#error "UPDATE_STATE is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef CACHE_INTERMEDIATE_STATES
#error "CACHE_INTERMEDIATE_STATES is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef INTERMEDIATE_BATCH_STRIDE
#error "INTERMEDIATE_BATCH_STRIDE is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef INTERMEDIATE_TOKEN_STRIDE
#error "INTERMEDIATE_TOKEN_STRIDE is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef STRIDED_INPUTS
#error "STRIDED_INPUTS is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef SCALE
#error "SCALE is a downstream specialization of this program; define it on the compile line"
#endif

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_gdn_0581947bd7b00d694574(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, float* __restrict__ intermediate_state, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, long long state_size_p0, long long state_stride_p0, long long state_stride_p1, long long state_stride_p2, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2)
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

    // Kernel setup ops
    float* sState = reinterpret_cast<float*>(smem_raw + SMEM_SSTATE_OFF);
    const int sState_addr = smem + SMEM_SSTATE_OFF;
    float* sQ = reinterpret_cast<float*>(smem_raw + SMEM_SQ_OFF);
    const int sQ_addr = smem + SMEM_SQ_OFF;
    float* sK = reinterpret_cast<float*>(smem_raw + SMEM_SK_OFF);
    const int sK_addr = smem + SMEM_SK_OFF;
    float* sV = reinterpret_cast<float*>(smem_raw + SMEM_SV_OFF);
    const int sV_addr = smem + SMEM_SV_OFF;
    float* sScalar = reinterpret_cast<float*>(smem_raw + SMEM_SSCALAR_OFF);
    const int sScalar_addr = smem + SMEM_SSCALAR_OFF;

    // === Task calls (dependency order) ===
    int linear_block = blockIdx.x;
    int state_head = linear_block / 8;
    int split = linear_block - state_head * 8;
    int n = state_head / HV;
    int h = state_head - n * HV;
    int lane_local = lane;
    int warp_local = warp;
    int qk_h = h / (HV / H);
    long long state_pool_size = state_size_p0;
    int read_state_slot_raw = initial_state_indices[n];
    int read_state_slot_valid = ((read_state_slot_raw >= 0 && state_pool_size > (long long)read_state_slot_raw) ? 1 : 0);
    int read_state_slot = ((read_state_slot_valid != 0) ? read_state_slot_raw : 0);
    int write_state_slot_raw = output_state_indices[n];
    int write_state_slot_valid = ((write_state_slot_raw >= 0 && state_pool_size > (long long)write_state_slot_raw) ? 1 : 0);
    int write_state_slot = ((write_state_slot_valid != 0) ? write_state_slot_raw : 0);
    long long read_state_head_base = (long long)read_state_slot * state_stride_p0 + (long long)h * state_stride_p1;
    long long write_state_head_base = (long long)write_state_slot * state_stride_p0 + (long long)h * state_stride_p1;
    int split_v_base = split * 16;
    int k_start = lane_local * 4;
    unsigned int r_qw[2];
    unsigned int r_kw[2];
    float r_h[4];
    #pragma unroll
    for (int copy_iter = 0; copy_iter < 4; copy_iter++) {
        int copy_seg = copy_iter * 128 + tid;
        int copy_row = copy_seg / 32;
        int copy_k_vec = copy_seg - copy_row * 32;
        int copy_v_row = split_v_base + copy_row;
        int copy_k_base = copy_k_vec * 4;
        int copy_dst = sState_addr + (unsigned int)((copy_row * 128 + copy_k_base) * 4);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(copy_dst), "l"(state + (read_state_head_base + (long long)copy_v_row * state_stride_p2 + (long long)copy_k_base)));
    }
    asm volatile("cp.async.commit_group;");
    if (warp_local < T_STEPS) {
        int t_w = warp_local;
        long long qk_base = (long long)((n * T_STEPS + t_w) * H + qk_h) * 128;
        long long k_base = qk_base;
        long long vh_base = (long long)((n * T_STEPS + t_w) * HV + h) * 128;
        long long gate_index = (long long)((n * T_STEPS + t_w) * HV + h);
        long long b_index = gate_index;
        {
            qk_base = (long long)n * q_stride_p0 + (long long)t_w * q_stride_p1 + (long long)qk_h * q_stride_p2;
            k_base = (long long)n * k_stride_p0 + (long long)t_w * k_stride_p1 + (long long)qk_h * k_stride_p2;
            vh_base = (long long)n * v_stride_p0 + (long long)t_w * v_stride_p1 + (long long)h * v_stride_p2;
            gate_index = (long long)n * a_stride_p0 + (long long)t_w * a_stride_p1 + (long long)h * a_stride_p2;
            b_index = (long long)n * b_stride_p0 + (long long)t_w * b_stride_p1 + (long long)h * b_stride_p2;
        }
        {
            uint2 _vld_0;
            _vld_0 = *reinterpret_cast<const uint2*>(q + qk_base + (long long)k_start);
            uint2* _vdst_0 = reinterpret_cast<uint2*>(&r_qw[0]);
            *_vdst_0 = _vld_0;
        }
        {
            uint2 _vld_1;
            _vld_1 = *reinterpret_cast<const uint2*>(k + k_base + (long long)k_start);
            uint2* _vdst_1 = reinterpret_cast<uint2*>(&r_kw[0]);
            *_vdst_1 = _vld_1;
        }
        int v_lane = ((lane_local < 16) ? lane_local : lane_local - 16);
        float v_val_w = (float)v[vh_base + (long long)split_v_base + (long long)v_lane];
        float a_val = (float)a[gate_index];
        float b_val = (float)b[b_index];
        float a_log_h = A_log[h];
        float dt_bias_h = dt_bias[h];
        float2 _f2_0 = make_float2(__uint_as_float(r_qw[0] << 16), __uint_as_float(r_qw[0] & 4294901760u));
        float2 q_raw_pair0 = _f2_0;
        float2 _f2_1 = make_float2(__uint_as_float(r_qw[1] << 16), __uint_as_float(r_qw[1] & 4294901760u));
        float2 q_raw_pair1 = _f2_1;
        float2 _f2_2 = make_float2(__uint_as_float(r_kw[0] << 16), __uint_as_float(r_kw[0] & 4294901760u));
        float2 k_raw_pair0 = _f2_2;
        float2 _f2_3 = make_float2(__uint_as_float(r_kw[1] << 16), __uint_as_float(r_kw[1] & 4294901760u));
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
        float q_norm = _rsqrt_0 * SCALE;
        float _rsqrt_1 = rsqrtf(sum_k + 1e-06f);
        float k_norm = _rsqrt_1;
        float2 _f2_6 = make_float2(q_norm, q_norm);
        float2 q_norm_pair = _f2_6;
        float2 _f2_7 = make_float2(k_norm, k_norm);
        float2 k_norm_pair = _f2_7;
        float2 _mul_f32x2_0;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&q_raw_pair0), "l"(*(const unsigned long long*)&q_norm_pair));
        float2 q_pair0 = _mul_f32x2_0;
        float2 _mul_f32x2_1;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&q_raw_pair1), "l"(*(const unsigned long long*)&q_norm_pair));
        float2 q_pair1 = _mul_f32x2_1;
        float2 _mul_f32x2_2;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&k_raw_pair0), "l"(*(const unsigned long long*)&k_norm_pair));
        float2 k_pair0 = _mul_f32x2_2;
        float2 _mul_f32x2_3;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&k_raw_pair1), "l"(*(const unsigned long long*)&k_norm_pair));
        float2 k_pair1 = _mul_f32x2_3;
        sQ[t_w * 128 + k_start] = q_pair0.x;
        sQ[t_w * 128 + k_start + 1] = q_pair0.y;
        sQ[t_w * 128 + k_start + 2] = q_pair1.x;
        sQ[t_w * 128 + k_start + 3] = q_pair1.y;
        sK[t_w * 128 + k_start] = k_pair0.x;
        sK[t_w * 128 + k_start + 1] = k_pair0.y;
        sK[t_w * 128 + k_start + 2] = k_pair1.x;
        sK[t_w * 128 + k_start + 3] = k_pair1.y;
        if (lane_local < 16) {
            sV[t_w * 16 + lane_local] = v_val_w;
        }
        if (lane_local == 0) {
            float x = a_val + dt_bias_h;
            float softplus_x = x;
            if (x <= 20.0f) {
                float _expf_0 = __expf(x);
                float _log2_0;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(1.0f + _expf_0));
                softplus_x = _log2_0 * 0.6931471805599453f;
            }
            float _expf_1 = __expf(-b_val);
            float _rcp_0 = approx_rcp(1.0f + _expf_1);
            float beta_val = _rcp_0;
            float _expf_2 = __expf(a_log_h);
            float g_log = (-_expf_2) * softplus_x;
            float _expf_3 = __expf(g_log);
            sScalar[t_w * 2] = _expf_3;
            sScalar[t_w * 2 + 1] = beta_val;
        }
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    #pragma unroll
    for (int row_group = 0; row_group < 16; row_group += 4) {
        int v_row = split_v_base + row_group + warp_local;
        int local_row = row_group + warp_local;
        float2 _f2_8 = make_float2(sState[local_row * 128 + k_start], sState[local_row * 128 + k_start + 1]);
        float2 h_pair0 = _f2_8;
        float2 _f2_9 = make_float2(sState[local_row * 128 + k_start + 2], sState[local_row * 128 + k_start + 3]);
        float2 h_pair1 = _f2_9;
        #pragma unroll
        for (int t = 0; t < T_STEPS; t++) {
            float2 _f2_10 = make_float2(sQ[t * 128 + k_start], sQ[t * 128 + k_start + 1]);
            float2 q_pair0_1 = _f2_10;
            float2 _f2_11 = make_float2(sQ[t * 128 + k_start + 2], sQ[t * 128 + k_start + 3]);
            float2 q_pair1_1 = _f2_11;
            float2 _f2_12 = make_float2(sK[t * 128 + k_start], sK[t * 128 + k_start + 1]);
            float2 k_pair0_1 = _f2_12;
            float2 _f2_13 = make_float2(sK[t * 128 + k_start + 2], sK[t * 128 + k_start + 3]);
            float2 k_pair1_1 = _f2_13;
            float decay_val = sScalar[t * 2];
            float beta_val_1 = sScalar[t * 2 + 1];
            float2 _f2_14 = make_float2(decay_val, decay_val);
            float2 decay_pair = _f2_14;
            float2 _mul_f32x2_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&h_pair0), "l"(*(const unsigned long long*)&decay_pair));
            h_pair0 = _mul_f32x2_4;
            float2 _mul_f32x2_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&h_pair1), "l"(*(const unsigned long long*)&decay_pair));
            h_pair1 = _mul_f32x2_5;
            float2 _f2_15 = make_float2(0.0f, 0.0f);
            float2 sum_hk_pair = fma_f32x2_rn_ftz(h_pair0, k_pair0_1, _f2_15);
            sum_hk_pair = fma_f32x2_rn_ftz(h_pair1, k_pair1_1, sum_hk_pair);
            float sum_hk = sum_hk_pair.x + sum_hk_pair.y;
            float _warp_reduce_2 = sum_hk;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
            sum_hk = _warp_reduce_2;
            float v_val = sV[t * 16 + local_row];
            float v_new = (v_val - sum_hk) * beta_val_1;
            float2 _f2_16 = make_float2(v_new, v_new);
            float2 v_new_pair = _f2_16;
            h_pair0 = fma_f32x2_rn_ftz(k_pair0_1, v_new_pair, h_pair0);
            h_pair1 = fma_f32x2_rn_ftz(k_pair1_1, v_new_pair, h_pair1);
            if (read_state_slot_valid != 0) {
                long long cache_head_base = (long long)n * (long long)INTERMEDIATE_BATCH_STRIDE + (long long)t * (long long)INTERMEDIATE_TOKEN_STRIDE + (long long)h * 16384;
                r_h[0] = h_pair0.x;
                r_h[1] = h_pair0.y;
                r_h[2] = h_pair1.x;
                r_h[3] = h_pair1.y;
                {
                    float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
                    *reinterpret_cast<float4*>(intermediate_state + cache_head_base + (long long)(v_row * 128) + (long long)k_start) = _v4;
                }
            }
            float2 _f2_17 = make_float2(0.0f, 0.0f);
            float2 sum_hq_pair = fma_f32x2_rn_ftz(h_pair0, q_pair0_1, _f2_17);
            sum_hq_pair = fma_f32x2_rn_ftz(h_pair1, q_pair1_1, sum_hq_pair);
            float sum_hq = sum_hq_pair.x + sum_hq_pair.y;
            float _warp_reduce_3 = sum_hq;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
            sum_hq = _warp_reduce_3;
            if (lane_local == 0) {
                if (read_state_slot_valid != 0) {
                    out[((n * T_STEPS + t) * HV + h) * 128 + v_row] = sum_hq;
                }
            }
        }
        if (read_state_slot_valid != 0) {
            if (write_state_slot_valid != 0) {
                r_h[0] = h_pair0.x;
                r_h[1] = h_pair0.y;
                r_h[2] = h_pair1.x;
                r_h[3] = h_pair1.y;
                {
                    float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
                    *reinterpret_cast<float4*>(state + write_state_head_base + (long long)v_row * state_stride_p2 + (long long)k_start) = _v4;
                }
            }
        }
    }
}

} // extern "C"
// clang-format on
