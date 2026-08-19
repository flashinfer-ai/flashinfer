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
#define SMEM_SSTATE_STAGE_BYTES 36864
#define SMEM_SSTATE_STRIDE 36864
#define SMEM_SQ_OFF 36864
#define SMEM_SQ_STAGE_BYTES 512
#define SMEM_SQ_STRIDE 512
#define SMEM_SK_OFF 37376
#define SMEM_SK_STAGE_BYTES 512
#define SMEM_SK_STRIDE 512
#define SMEM_TOTAL 38016
#define THREADS 256
#define H 16
#define HV 32
#define SCALE 0.08838834764831845



extern "C" {

__global__ __launch_bounds__(256) void
kernel_gdn_decode_nontranspose_fp32_t1(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, long long q_stride_p0, long long k_stride_p0, long long v_stride_p0, long long a_stride_p0, long long b_stride_p0)
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
    float* sQ = reinterpret_cast<float*>(smem_raw + 36864);
    const int sQ_addr = smem + 36864;
    float* sK = reinterpret_cast<float*>(smem_raw + 37376);
    const int sK_addr = smem + 37376;

    // === Task calls (dependency order) ===
    int state_head = blockIdx.x;
    int n = state_head / HV;
    int h = state_head - n * HV;
    int qk_h = h / (HV / H);
    long long q_base = (long long)n * q_stride_p0 + (long long)qk_h * 128;
    long long k_base = (long long)n * k_stride_p0 + (long long)qk_h * 128;
    long long v_base = (long long)n * v_stride_p0 + (long long)h * 128;
    int out_base = (n * HV + h) * 128;
    long long state_head_base = (long long)state_head * 128 * 128;
    int lane_local = lane;
    int warp_local = warp;
    int k_local = lane_local / 4;
    int v_local = lane_local - k_local * 4;
    int tile_v_local = warp_local * 4 + v_local;
    float r_q[4];
    float r_k[4];
    float r_h[16];
    float r_store[4];
    #pragma unroll
    for (int copy_iter = 0; copy_iter < 4; copy_iter++) {
        int copy_seg = copy_iter * 256 + tid;
        int copy_k = copy_seg / 8;
        int copy_v_vec = copy_seg - copy_k * 8;
        int copy_v_base = copy_v_vec * 4;
        int copy_dst = sState_addr + (unsigned int)((copy_k * 36 + copy_v_base) * 4);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(copy_dst), "l"(state + (state_head_base + (long long)(copy_k * 128) + (long long)copy_v_base)));
    }
    asm volatile("cp.async.commit_group;");
    if (warp_local == 0) {
        int qk_lane_base = lane_local * 4;
        {
            uint2 _vld_0 = *reinterpret_cast<const uint2*>(q + q_base + (long long)qk_lane_base);
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
            uint2 _vld_1 = *reinterpret_cast<const uint2*>(k + k_base + (long long)qk_lane_base);
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
        float sum_q = 0.0f;
        float sum_k = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            sum_q += r_q[i] * r_q[i];
            sum_k += r_k[i] * r_k[i];
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
        #pragma unroll
        for (int i_1 = 0; i_1 < 4; i_1++) {
            sQ[qk_lane_base + i_1] = r_q[i_1] * q_norm;
            sK[qk_lane_base + i_1] = r_k[i_1] * k_norm;
        }
    }
    float decay_lane = 0.0f;
    float beta_lane = 0.0f;
    if (lane_local == 0) {
        float gate_x = (float)a[(long long)n * a_stride_p0 + (long long)h] + dt_bias[h];
        float softplus_x = gate_x;
        if (gate_x <= 20.0f) {
            float _expf_0 = __expf(gate_x);
            float _log2_0;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(1.0f + _expf_0));
            softplus_x = _log2_0 * 0.6931471805599453f;
        }
        float _expf_1 = __expf(A_log[h]);
        float _expf_2 = __expf((-_expf_1) * softplus_x);
        decay_lane = _expf_2;
        float _expf_3 = __expf(-(float)b[(long long)n * b_stride_p0 + (long long)h]);
        float _rcp_0 = approx_rcp(1.0f + _expf_3);
        beta_lane = _rcp_0;
    }
    float _shfl_0 = __shfl_sync(0xFFFFFFFF, decay_lane, 0);
    float decay_val = _shfl_0;
    float _shfl_1 = __shfl_sync(0xFFFFFFFF, beta_lane, 0);
    float beta_val = _shfl_1;
    #pragma unroll
    for (int tile = 0; tile < 4; tile++) {
        int stage = tile % 2;
        int tile_v_base = tile * 32;
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        if (tile + 1 < 4) {
            int next_tile = tile + 1;
            int next_stage = next_tile % 2;
            int next_v_base = next_tile * 32;
            #pragma unroll
            for (int copy_iter_1 = 0; copy_iter_1 < 4; copy_iter_1++) {
                int copy_seg_1 = copy_iter_1 * 256 + tid;
                int copy_k_1 = copy_seg_1 / 8;
                int copy_v_vec_1 = copy_seg_1 - copy_k_1 * 8;
                int copy_v_base_1 = copy_v_vec_1 * 4;
                int copy_dst_1 = sState_addr + (unsigned int)((next_stage * 4608 + copy_k_1 * 36 + copy_v_base_1) * 4);
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(copy_dst_1), "l"(state + (state_head_base + (long long)(copy_k_1 * 128) + (long long)next_v_base + (long long)copy_v_base_1)));
            }
            asm volatile("cp.async.commit_group;");
        }
        float sum_hk = 0.0f;
        #pragma unroll
        for (int k_iter = 0; k_iter < 16; k_iter++) {
            int k_idx = k_iter * 8 + k_local;
            float h_val = sState[stage * 4608 + k_idx * 36 + tile_v_local] * decay_val;
            r_h[k_iter] = h_val;
            sum_hk += h_val * sK[k_idx];
        }
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, sum_hk, 16);
        sum_hk += _shfl_xor_0;
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, sum_hk, 8);
        sum_hk += _shfl_xor_1;
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, sum_hk, 4);
        sum_hk += _shfl_xor_2;
        float value_lane = 0.0f;
        if (k_local == 0) {
            value_lane = (float)v[v_base + (long long)tile_v_base + (long long)tile_v_local];
        }
        float _shfl_2 = __shfl_sync(0xFFFFFFFF, value_lane, v_local);
        float value_val = _shfl_2;
        float delta = (value_val - sum_hk) * beta_val;
        float sum_hq = 0.0f;
        #pragma unroll
        for (int k_iter_1 = 0; k_iter_1 < 16; k_iter_1++) {
            int k_idx_1 = k_iter_1 * 8 + k_local;
            float h_new = r_h[k_iter_1] + sK[k_idx_1] * delta;
            r_h[k_iter_1] = h_new;
            sState[stage * 4608 + k_idx_1 * 36 + tile_v_local] = h_new;
            sum_hq += h_new * sQ[k_idx_1];
        }
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, sum_hq, 16);
        sum_hq += _shfl_xor_3;
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, sum_hq, 8);
        sum_hq += _shfl_xor_4;
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, sum_hq, 4);
        sum_hq += _shfl_xor_5;
        if (k_local == 0) {
            out[out_base + tile_v_base + tile_v_local] = sum_hq;
        }
        __syncthreads();
        #pragma unroll
        for (int copy_iter_2 = 0; copy_iter_2 < 4; copy_iter_2++) {
            int copy_seg_2 = copy_iter_2 * 256 + tid;
            int copy_k_2 = copy_seg_2 / 8;
            int copy_v_vec_2 = copy_seg_2 - copy_k_2 * 8;
            int copy_v_base_2 = copy_v_vec_2 * 4;
            #pragma unroll
            for (int i_2 = 0; i_2 < 4; i_2++) {
                r_store[i_2] = sState[stage * 4608 + copy_k_2 * 36 + copy_v_base_2 + i_2];
            }
            {
                float4 _v4 = make_float4(r_store[0 + 0], r_store[0 + 1], r_store[0 + 2], r_store[0 + 3]);
                *reinterpret_cast<float4*>(state + state_head_base + (long long)(copy_k_2 * 128) + (long long)tile_v_base + (long long)copy_v_base_2) = _v4;
            }
        }
        __syncthreads();
    }
}

} // extern "C"

#undef H
#undef HV
#undef CAKE_INF
#undef NUM_MAIN_STAGES
#undef SCALE
#undef SMEM_SK_OFF
#undef SMEM_SK_STAGE_BYTES
#undef SMEM_SK_STRIDE
#undef SMEM_SQ_OFF
#undef SMEM_SQ_STAGE_BYTES
#undef SMEM_SQ_STRIDE
#undef SMEM_SSTATE_OFF
#undef SMEM_SSTATE_STAGE_BYTES
#undef SMEM_SSTATE_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef sK_addr
#undef sQ_addr
#undef sState_addr
// clang-format on
