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
#define H 16
#define HV 32
#define T_STEPS 4
#define UPDATE_STATE 1
#define CACHE_INTERMEDIATE_STATES 1
#define INTERMEDIATE_BATCH_STRIDE 2097152
#define INTERMEDIATE_TOKEN_STRIDE 524288
#define STRIDED_INPUTS 1
#define SCALE 0.08838834764831845



extern "C" {

__global__ __launch_bounds__(128) void
kernel_gdn_decode_pretranspose_mtp_t4_splitv8(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, float* __restrict__ intermediate_state, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, long long state_stride_p0, long long state_stride_p1, long long state_stride_p2, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2)
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
    float* sState = reinterpret_cast<float*>(smem_raw + 0);
    const int sState_addr = smem + 0;
    float* sQ = reinterpret_cast<float*>(smem_raw + 8192);
    const int sQ_addr = smem + 8192;
    float* sK = reinterpret_cast<float*>(smem_raw + 10240);
    const int sK_addr = smem + 10240;
    float* sV = reinterpret_cast<float*>(smem_raw + 12288);
    const int sV_addr = smem + 12288;
    float* sScalar = reinterpret_cast<float*>(smem_raw + 14336);
    const int sScalar_addr = smem + 14336;

    // === Task calls (dependency order) ===
    int linear_block = blockIdx.x;
    int state_head = linear_block / 8;
    int split = linear_block - state_head * 8;
    int n = state_head / HV;
    int h = state_head - n * HV;
    int lane_local = lane;
    int warp_local = warp;
    int qk_h = h / (HV / H);
    int read_state_slot_raw = initial_state_indices[n];
    int read_state_slot = ((read_state_slot_raw >= 0) ? read_state_slot_raw : 0);
    int write_state_slot_raw = output_state_indices[n];
    int write_state_slot = ((write_state_slot_raw >= 0) ? write_state_slot_raw : 0);
    long long read_state_head_base = (long long)read_state_slot * state_stride_p0 + (long long)h * state_stride_p1;
    long long write_state_head_base = (long long)write_state_slot * state_stride_p0 + (long long)h * state_stride_p1;
    int split_v_base = split * 16;
    int k_start = lane_local * 4;
    float r_q[4];
    float r_k[4];
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
    if (warp_local == 0) {
        #pragma unroll
        for (int t = 0; t < T_STEPS; t++) {
            long long qk_base = (long long)((n * T_STEPS + t) * H + qk_h) * 128;
            long long vh_base = (long long)((n * T_STEPS + t) * HV + h) * 128;
            {
                qk_base = (long long)n * q_stride_p0 + (long long)t * q_stride_p1 + (long long)qk_h * q_stride_p2;
                vh_base = (long long)n * v_stride_p0 + (long long)t * v_stride_p1 + (long long)h * v_stride_p2;
            }
            {
                uint2 _vld_0;
                _vld_0 = *reinterpret_cast<const uint2*>(q + qk_base + (long long)k_start);
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
                long long k_base = (long long)n * k_stride_p0 + (long long)t * k_stride_p1 + (long long)qk_h * k_stride_p2;
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
            float2 q_pair0 = mul_f32x2(q_raw_pair0, q_norm_pair);
            float2 q_pair1 = mul_f32x2(q_raw_pair1, q_norm_pair);
            float2 k_pair0 = mul_f32x2(k_raw_pair0, k_norm_pair);
            float2 k_pair1 = mul_f32x2(k_raw_pair1, k_norm_pair);
            sQ[t * 128 + k_start] = q_pair0.x;
            sQ[t * 128 + k_start + 1] = q_pair0.y;
            sQ[t * 128 + k_start + 2] = q_pair1.x;
            sQ[t * 128 + k_start + 3] = q_pair1.y;
            sK[t * 128 + k_start] = k_pair0.x;
            sK[t * 128 + k_start + 1] = k_pair0.y;
            sK[t * 128 + k_start + 2] = k_pair1.x;
            sK[t * 128 + k_start + 3] = k_pair1.y;
            if (lane_local < 16) {
                sV[t * 16 + lane_local] = (float)v[vh_base + (long long)split_v_base + (long long)lane_local];
            }
            if (lane_local == 0) {
                long long gate_index = (long long)((n * T_STEPS + t) * HV + h);
                long long b_index = gate_index;
                {
                    gate_index = (long long)n * a_stride_p0 + (long long)t * a_stride_p1 + (long long)h * a_stride_p2;
                    b_index = (long long)n * b_stride_p0 + (long long)t * b_stride_p1 + (long long)h * b_stride_p2;
                }
                float x = (float)a[gate_index] + dt_bias[h];
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
                float _expf_2 = __expf(A_log[h]);
                float g_log = (-_expf_2) * softplus_x;
                float _expf_3 = __expf(g_log);
                sScalar[t * 2] = _expf_3;
                sScalar[t * 2 + 1] = beta_val;
            }
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
        for (int t_1 = 0; t_1 < T_STEPS; t_1++) {
            float2 _f2_10 = make_float2(sQ[t_1 * 128 + k_start], sQ[t_1 * 128 + k_start + 1]);
            float2 q_pair0_1 = _f2_10;
            float2 _f2_11 = make_float2(sQ[t_1 * 128 + k_start + 2], sQ[t_1 * 128 + k_start + 3]);
            float2 q_pair1_1 = _f2_11;
            float2 _f2_12 = make_float2(sK[t_1 * 128 + k_start], sK[t_1 * 128 + k_start + 1]);
            float2 k_pair0_1 = _f2_12;
            float2 _f2_13 = make_float2(sK[t_1 * 128 + k_start + 2], sK[t_1 * 128 + k_start + 3]);
            float2 k_pair1_1 = _f2_13;
            float decay_val = sScalar[t_1 * 2];
            float beta_val_1 = sScalar[t_1 * 2 + 1];
            float2 _f2_14 = make_float2(decay_val, decay_val);
            float2 decay_pair = _f2_14;
            h_pair0 = mul_f32x2(h_pair0, decay_pair);
            h_pair1 = mul_f32x2(h_pair1, decay_pair);
            float2 _f2_15 = make_float2(0.0f, 0.0f);
            float2 sum_hk_pair = fma_f32x2(h_pair0, k_pair0_1, _f2_15);
            sum_hk_pair = fma_f32x2(h_pair1, k_pair1_1, sum_hk_pair);
            float sum_hk = sum_hk_pair.x + sum_hk_pair.y;
            float _warp_reduce_2 = sum_hk;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
            sum_hk = _warp_reduce_2;
            float v_val = sV[t_1 * 16 + local_row];
            float v_new = (v_val - sum_hk) * beta_val_1;
            float2 _f2_16 = make_float2(v_new, v_new);
            float2 v_new_pair = _f2_16;
            h_pair0 = fma_f32x2(k_pair0_1, v_new_pair, h_pair0);
            h_pair1 = fma_f32x2(k_pair1_1, v_new_pair, h_pair1);
            {
                if (read_state_slot_raw >= 0) {
                    long long cache_head_base = (long long)n * (long long)INTERMEDIATE_BATCH_STRIDE + (long long)t_1 * (long long)INTERMEDIATE_TOKEN_STRIDE + (long long)h * 16384;
                    r_h[0] = h_pair0.x;
                    r_h[1] = h_pair0.y;
                    r_h[2] = h_pair1.x;
                    r_h[3] = h_pair1.y;
                    {
                        float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
                        *reinterpret_cast<float4*>(intermediate_state + cache_head_base + (long long)(v_row * 128) + (long long)k_start) = _v4;
                    }
                }
            }
            float2 _f2_17 = make_float2(0.0f, 0.0f);
            float2 sum_hq_pair = fma_f32x2(h_pair0, q_pair0_1, _f2_17);
            sum_hq_pair = fma_f32x2(h_pair1, q_pair1_1, sum_hq_pair);
            float sum_hq = sum_hq_pair.x + sum_hq_pair.y;
            float _warp_reduce_3 = sum_hq;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
            sum_hq = _warp_reduce_3;
            if (lane_local == 0) {
                if (read_state_slot_raw >= 0) {
                    out[((n * T_STEPS + t_1) * HV + h) * 128 + v_row] = sum_hq;
                }
            }
        }
        {
            if (read_state_slot_raw >= 0) {
                if (write_state_slot_raw >= 0) {
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
}

} // extern "C"

#undef CACHE_INTERMEDIATE_STATES
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
#undef SMEM_SSTATE_OFF
#undef SMEM_SSTATE_STAGE_BYTES
#undef SMEM_SSTATE_STRIDE
#undef SMEM_SV_OFF
#undef SMEM_SV_STAGE_BYTES
#undef SMEM_SV_STRIDE
#undef SMEM_TOTAL
#undef STRIDED_INPUTS
#undef THREADS
#undef T_STEPS
#undef UPDATE_STATE
#undef sK_addr
#undef sQ_addr
#undef sScalar_addr
#undef sState_addr
#undef sV_addr
// clang-format on
