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
#include "gdn_cp_common.cuh"

#define GDN_CP_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SHARED_K_OFF 0
#define SMEM_SHARED_K_STAGE_BYTES 512
#define SMEM_SHARED_K_STRIDE 512
#define SMEM_REDUCE_SCRATCH_OFF 512
#define SMEM_REDUCE_SCRATCH_STAGE_BYTES 16
#define SMEM_REDUCE_SCRATCH_STRIDE 16
#define SMEM_TOTAL 640
#define THREADS 128



extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_flashinfer_blackwell_gdn_cp_prefill_final_state_bf16_v1(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ alpha, float* __restrict__ beta, long long* __restrict__ cu_seqlens, float* __restrict__ initial_state, float* __restrict__ final_state, __nv_bfloat16* __restrict__ output, float scale, int normalize_qk, int write_output, int write_final_state, int num_q_heads, int num_k_heads, int num_v_heads, int num_state_heads)
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
    float* shared_k = reinterpret_cast<float*>(smem_raw + 0);
    const int shared_k_addr = smem + 0;
    float* reduce_scratch = reinterpret_cast<float*>(smem_raw + 512);
    const int reduce_scratch_addr = smem + 512;

    // === Task calls (dependency order) ===
    int state_head = blockIdx.x;
    int sequence = blockIdx.y;
    int value_dim = tid;
    int query_head = state_head * num_q_heads / num_state_heads;
    int key_head = state_head * num_k_heads / num_state_heads;
    int value_head = state_head * num_v_heads / num_state_heads;
    int sequence_start = (int)cu_seqlens[sequence];
    int sequence_end = (int)cu_seqlens[sequence + 1];
    long long state_base = (((long long)sequence * (long long)num_state_heads + (long long)state_head) * 128 + (long long)value_dim) * 128;
    float state_values[128];
    #pragma unroll
    for (int key_dim_init = 0; key_dim_init < 128; key_dim_init++) {
        state_values[key_dim_init] = initial_state[state_base + (long long)key_dim_init];
    }
    #pragma unroll 1
    for (int token = sequence_start; token < sequence_end; token++) {
        long long k_base = ((long long)token * (long long)num_k_heads + (long long)key_head) * 128;
        float k_value = (float)k[k_base + (long long)value_dim];
        if (normalize_qk != 0) {
            float _warp_reduce_0 = k_value * k_value;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
            float warp_norm_sq = _warp_reduce_0;
            if (lane == 0) {
                reduce_scratch[warp] = warp_norm_sq;
            }
            __syncthreads();
            float norm_sq = 0.0f;
            if (lane < 4) {
                norm_sq = reduce_scratch[lane];
            }
            float _warp_reduce_1 = norm_sq;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
            norm_sq = _warp_reduce_1;
            if (warp == 0) {
                if (elect_sync()) {
                    reduce_scratch[0] = norm_sq;
                }
            }
            __syncthreads();
            norm_sq = reduce_scratch[0];
            float _rsqrt_0 = rsqrtf(norm_sq + 1e-06f);
            shared_k[value_dim] = k_value * _rsqrt_0;
        } else {
            shared_k[value_dim] = k_value;
        }
        __syncthreads();
        long long gate_index = (long long)token * (long long)num_state_heads + (long long)state_head;
        float alpha_value = alpha[gate_index];
        float beta_value = beta[gate_index];
        float old_value = 0.0f;
        #pragma unroll
        for (int key_dim_dot = 0; key_dim_dot < 128; key_dim_dot++) {
            state_values[key_dim_dot] = state_values[key_dim_dot] * alpha_value;
            float _fma_0 = __fmaf_rn(shared_k[key_dim_dot], state_values[key_dim_dot], old_value);
            old_value = _fma_0;
        }
        long long v_base = ((long long)token * (long long)num_v_heads + (long long)value_head) * 128;
        float input_value = (float)v[v_base + (long long)value_dim];
        float residual = (input_value - old_value) * beta_value;
        {
            __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(residual);
            float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
            residual = _cvt_f32_0;
        }
        #pragma unroll
        for (int key_dim_update = 0; key_dim_update < 128; key_dim_update++) {
            float _fma_1 = __fmaf_rn(shared_k[key_dim_update], residual, state_values[key_dim_update]);
            state_values[key_dim_update] = _fma_1;
        }
        __syncthreads();
        if (write_output != 0) {
            long long q_base = ((long long)token * (long long)num_q_heads + (long long)query_head) * 128;
            float q_value = (float)q[q_base + (long long)value_dim];
            if (normalize_qk != 0) {
                float _warp_reduce_2 = q_value * q_value;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
                float warp_q_norm_sq = _warp_reduce_2;
                if (lane == 0) {
                    reduce_scratch[warp] = warp_q_norm_sq;
                }
                __syncthreads();
                float q_norm_sq = 0.0f;
                if (lane < 4) {
                    q_norm_sq = reduce_scratch[lane];
                }
                float _warp_reduce_3 = q_norm_sq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
                q_norm_sq = _warp_reduce_3;
                if (warp == 0) {
                    if (elect_sync()) {
                        reduce_scratch[0] = q_norm_sq;
                    }
                }
                __syncthreads();
                q_norm_sq = reduce_scratch[0];
                float _rsqrt_1 = rsqrtf(q_norm_sq + 1e-06f);
                q_value *= _rsqrt_1;
            }
            shared_k[value_dim] = q_value;
            __syncthreads();
            float output_value = 0.0f;
            #pragma unroll
            for (int key_dim_output = 0; key_dim_output < 128; key_dim_output++) {
                float _fma_2 = __fmaf_rn(shared_k[key_dim_output], state_values[key_dim_output], output_value);
                output_value = _fma_2;
            }
            long long output_base = ((long long)token * (long long)num_state_heads + (long long)state_head) * 128;
            output[output_base + (long long)value_dim] = output_value * scale;
            __syncthreads();
        }
    }
    if (write_final_state != 0) {
        #pragma unroll
        for (int key_dim_store = 0; key_dim_store < 128; key_dim_store++) {
            final_state[state_base + (long long)key_dim_store] = state_values[key_dim_store];
        }
    }
}

} // extern "C"

#undef GDN_CP_INF
#undef NUM_MAIN_STAGES
#undef SMEM_REDUCE_SCRATCH_OFF
#undef SMEM_REDUCE_SCRATCH_STAGE_BYTES
#undef SMEM_REDUCE_SCRATCH_STRIDE
#undef SMEM_SHARED_K_OFF
#undef SMEM_SHARED_K_STAGE_BYTES
#undef SMEM_SHARED_K_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef reduce_scratch_addr
#undef shared_k_addr
// clang-format on
