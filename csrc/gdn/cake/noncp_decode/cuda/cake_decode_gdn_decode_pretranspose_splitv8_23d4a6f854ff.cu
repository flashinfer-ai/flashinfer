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

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SSTATE_OFF 0
#define SMEM_SSTATE_STAGE_BYTES 8192
#define SMEM_SSTATE_STRIDE 8192
#define SMEM_SV_OFF 8192
#define SMEM_SV_STAGE_BYTES 512
#define SMEM_SV_STRIDE 512
#define SMEM_SOUTPUT_OFF 8704
#define SMEM_SOUTPUT_STAGE_BYTES 256
#define SMEM_SOUTPUT_STRIDE 256
#define SMEM_TOTAL 9088
#define THREADS 128
#define H 16
#define HV 32
#define SCALE 0.08838834764831845



extern "C" {

__global__ __launch_bounds__(128) void
kernel_gdn_decode_pretranspose_splitv8(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices)
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
    float* sV = reinterpret_cast<float*>(smem_raw + 8192);
    const int sV_addr = smem + 8192;
    __nv_bfloat16* sOutput = reinterpret_cast<__nv_bfloat16*>(smem_raw + 8704);
    const int sOutput_addr = smem + 8704;

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
    int read_state_slot = initial_state_indices[n];
    int safe_read_state_slot = ((read_state_slot < 0) ? 0 : read_state_slot);
    int write_state_slot = output_state_indices[n];
    long long read_state_head_base = ((long long)safe_read_state_slot * (long long)HV + (long long)h) * 16384;
    long long write_state_head_base = ((long long)write_state_slot * (long long)HV + (long long)h) * 16384;
    int split_v_base = split * 16;
    int k_start = lane_local * 4;
    float r_q[4];
    float r_k[4];
    float r_v[4];
    float r_h[4];
    float2 r_q_pair0;
    float2 r_q_pair1;
    float2 r_k_pair0;
    float2 r_k_pair1;
    float r_A_log = A_log[h];
    float r_a = (float)a[n * HV + h];
    float r_dt_bias = dt_bias[h];
    float r_b = (float)b[n * HV + h];
    #pragma unroll
    for (int copy_iter = 0; copy_iter < 4; copy_iter++) {
        int copy_seg = copy_iter * 128 + tid;
        int copy_row = copy_seg / 32;
        int copy_k_vec = copy_seg - copy_row * 32;
        int copy_v_row = split_v_base + copy_row;
        int copy_k_base = copy_k_vec * 4;
        int copy_dst = sState_addr + (unsigned int)((copy_row * 128 + copy_k_base) * 4);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(copy_dst), "l"(state + (read_state_head_base + (long long)(copy_v_row * 128) + (long long)copy_k_base)), "r"((read_state_slot >= 0) ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;");
    {
        uint2 _vld_0 = *reinterpret_cast<const uint2*>(q + qk_base + k_start);
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
        uint2 _vld_1 = *reinterpret_cast<const uint2*>(k + qk_base + k_start);
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
    {
        uint2 _vld_2 = *reinterpret_cast<const uint2*>(v + vh_base + k_start);
        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&r_v[0 + _pair * 2])[0]), "=f"((&r_v[0 + _pair * 2])[1])
                : "r"(_vpairs_2[_pair]));
        }
    }
    if (warp_local == 0) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            sV[k_start + i] = r_v[i];
        }
    }
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
    r_q_pair0 = mul_f32x2(q_raw_pair0, q_norm_pair);
    r_q_pair1 = mul_f32x2(q_raw_pair1, q_norm_pair);
    r_k_pair0 = mul_f32x2(k_raw_pair0, k_norm_pair);
    r_k_pair1 = mul_f32x2(k_raw_pair1, k_norm_pair);
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
    __syncthreads();
    float2 _f2_8 = make_float2(decay_val, decay_val);
    float2 decay_pair = _f2_8;
    #pragma unroll
    for (int tile = 0; tile < 1; tile++) {
        int stage = 0;
        int tile_v_base = split_v_base + tile * 16;
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        if (tile + 1 < 1) {
            int next_tile_v_base = split_v_base + (tile + 1) * 16;
            int next_stage = 0;
            #pragma unroll
            for (int copy_iter_1 = 0; copy_iter_1 < 4; copy_iter_1++) {
                int copy_seg_1 = copy_iter_1 * 128 + tid;
                int copy_row_1 = copy_seg_1 / 32;
                int copy_k_vec_1 = copy_seg_1 - copy_row_1 * 32;
                int copy_v_row_1 = next_tile_v_base + copy_row_1;
                int copy_k_base_1 = copy_k_vec_1 * 4;
                int copy_dst_1 = sState_addr + (unsigned int)(next_stage * 8192) + (unsigned int)((copy_row_1 * 128 + copy_k_base_1) * 4);
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(copy_dst_1), "l"(state + (read_state_head_base + (long long)(copy_v_row_1 * 128) + (long long)copy_k_base_1)), "r"((read_state_slot >= 0) ? 16 : 0));
            }
            asm volatile("cp.async.commit_group;");
        }
        #pragma unroll
        for (int row_group = 0; row_group < 16; row_group += 4) {
            int v_row = tile_v_base + row_group + warp_local;
            int local_row = row_group + warp_local;
            float2 _f2_9 = make_float2(sState[stage * 2048 + local_row * 128 + k_start], sState[stage * 2048 + local_row * 128 + k_start + 1]);
            float2 h_src_pair0 = _f2_9;
            float2 _f2_10 = make_float2(sState[stage * 2048 + local_row * 128 + k_start + 2], sState[stage * 2048 + local_row * 128 + k_start + 3]);
            float2 h_src_pair1 = _f2_10;
            float2 h_pair0 = mul_f32x2(h_src_pair0, decay_pair);
            float2 h_pair1 = mul_f32x2(h_src_pair1, decay_pair);
            float2 _f2_11 = make_float2(0.0f, 0.0f);
            float2 sum_hk_pair = fma_f32x2(h_pair0, r_k_pair0, _f2_11);
            sum_hk_pair = fma_f32x2(h_pair1, r_k_pair1, sum_hk_pair);
            float sum_hk = sum_hk_pair.x + sum_hk_pair.y;
            float _warp_reduce_2 = sum_hk;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
            sum_hk = _warp_reduce_2;
            float v_val = sV[v_row];
            float v_new = (v_val - sum_hk) * beta_val;
            float2 _f2_12 = make_float2(v_new, v_new);
            float2 v_new_pair = _f2_12;
            float2 h_new_pair0 = fma_f32x2(r_k_pair0, v_new_pair, h_pair0);
            float2 h_new_pair1 = fma_f32x2(r_k_pair1, v_new_pair, h_pair1);
            r_h[0] = h_new_pair0.x;
            r_h[1] = h_new_pair0.y;
            r_h[2] = h_new_pair1.x;
            r_h[3] = h_new_pair1.y;
            float2 _f2_13 = make_float2(0.0f, 0.0f);
            float2 sum_hq_pair = fma_f32x2(h_new_pair0, r_q_pair0, _f2_13);
            sum_hq_pair = fma_f32x2(h_new_pair1, r_q_pair1, sum_hq_pair);
            float sum_hq = sum_hq_pair.x + sum_hq_pair.y;
            if (read_state_slot >= 0) {
                if (write_state_slot >= 0) {
                    {
                        float4 _v4 = make_float4(r_h[0 + 0], r_h[0 + 1], r_h[0 + 2], r_h[0 + 3]);
                        *reinterpret_cast<float4*>(state + (write_state_head_base + (long long)(v_row * 128)) + k_start) = _v4;
                    }
                }
            }
            float _warp_reduce_3 = sum_hq;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
            sum_hq = _warp_reduce_3;
            if (lane_local == 0) {
                out[vh_base + v_row] = ((read_state_slot >= 0) ? sum_hq : 0.0f);
            }
        }
    }
}

} // extern "C"

#undef H
#undef HV
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SCALE
#undef SMEM_SOUTPUT_OFF
#undef SMEM_SOUTPUT_STAGE_BYTES
#undef SMEM_SOUTPUT_STRIDE
#undef SMEM_SSTATE_OFF
#undef SMEM_SSTATE_STAGE_BYTES
#undef SMEM_SSTATE_STRIDE
#undef SMEM_SV_OFF
#undef SMEM_SV_STAGE_BYTES
#undef SMEM_SV_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef sOutput_addr
#undef sState_addr
#undef sV_addr
// clang-format on
