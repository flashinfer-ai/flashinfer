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
#define SMEM_SHARED_K_BLOCK_OFF 0
#define SMEM_SHARED_K_BLOCK_STAGE_BYTES 16384
#define SMEM_SHARED_K_BLOCK_STRIDE 16384
#define SMEM_SHARED_SYSTEM_OFF 16384
#define SMEM_SHARED_SYSTEM_STAGE_BYTES 16384
#define SMEM_SHARED_SYSTEM_STRIDE 16384
#define SMEM_SHARED_A_OFF 16384
#define SMEM_SHARED_A_STAGE_BYTES 8192
#define SMEM_SHARED_A_STRIDE 8192
#define SMEM_SHARED_INVERSE_OFF 32768
#define SMEM_SHARED_INVERSE_STAGE_BYTES 16384
#define SMEM_SHARED_INVERSE_STRIDE 16384
#define SMEM_SHARED_W_OFF 49152
#define SMEM_SHARED_W_STAGE_BYTES 16384
#define SMEM_SHARED_W_STRIDE 16384
#define SMEM_SHARED_U_OR_R_OFF 65536
#define SMEM_SHARED_U_OR_R_STAGE_BYTES 16384
#define SMEM_SHARED_U_OR_R_STRIDE 16384
#define SMEM_SHARED_GATE_OFF 81920
#define SMEM_SHARED_GATE_STAGE_BYTES 256
#define SMEM_SHARED_GATE_STRIDE 256
#define SMEM_SHARED_BETA_OFF 82176
#define SMEM_SHARED_BETA_STAGE_BYTES 256
#define SMEM_SHARED_BETA_STRIDE 256
#define SMEM_REDUCE_SCRATCH_OFF 82432
#define SMEM_REDUCE_SCRATCH_STAGE_BYTES 16
#define SMEM_REDUCE_SCRATCH_STRIDE 16
#define SMEM_LEGACY_SHARED_K_OFF 0
#define SMEM_LEGACY_SHARED_K_STAGE_BYTES 512
#define SMEM_LEGACY_SHARED_K_STRIDE 512
#define SMEM_TOTAL 82560
#define THREADS 128



extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_flashinfer_blackwell_gdn_cp_prefill_final_state_bf16_v1(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, float* __restrict__ alpha, float* __restrict__ beta, long long* __restrict__ cu_seqlens, float* __restrict__ initial_state, float* __restrict__ final_state, __nv_bfloat16* __restrict__ output, float scale, int normalize_qk, int write_output, int write_final_state, int use_block64_final_state, int num_q_heads, int num_k_heads, int num_v_heads, int num_state_heads)
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
    __nv_bfloat16* shared_k_block = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int shared_k_block_addr = smem + 0;
    float* shared_system = reinterpret_cast<float*>(smem_raw + 16384);
    const int shared_system_addr = smem + 16384;
    __nv_bfloat16* shared_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int shared_a_addr = smem + 16384;
    float* shared_inverse = reinterpret_cast<float*>(smem_raw + 32768);
    const int shared_inverse_addr = smem + 32768;
    __nv_bfloat16* shared_w = reinterpret_cast<__nv_bfloat16*>(smem_raw + 49152);
    const int shared_w_addr = smem + 49152;
    __nv_bfloat16* shared_u_or_r = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int shared_u_or_r_addr = smem + 65536;
    float* shared_gate = reinterpret_cast<float*>(smem_raw + 81920);
    const int shared_gate_addr = smem + 81920;
    float* shared_beta = reinterpret_cast<float*>(smem_raw + 82176);
    const int shared_beta_addr = smem + 82176;
    float* reduce_scratch = reinterpret_cast<float*>(smem_raw + 82432);
    const int reduce_scratch_addr = smem + 82432;
    float* legacy_shared_k = reinterpret_cast<float*>(smem_raw + 0);
    const int legacy_shared_k_addr = smem + 0;

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
    if (use_block64_final_state != 0) {
        float state_values_exact[128];
        #pragma unroll
        for (int key_dim_init = 0; key_dim_init < 128; key_dim_init++) {
            state_values_exact[key_dim_init] = initial_state[state_base + (long long)key_dim_init];
        }
        int sequence_length = sequence_end - sequence_start;
        int num_blocks = (sequence_length + 64 - 1) / 64;
        #pragma unroll 1
        for (int block_index = 0; block_index < num_blocks; block_index++) {
            int block_start = sequence_start + block_index * 64;
            int remaining = sequence_end - block_start;
            int valid_len = ((remaining < 64) ? remaining : 64);
            if (warp == 0) {
                int row0 = lane;
                int row1 = row0 + 32;
                float gate_delta0 = 0.0f;
                float gate_delta1 = 0.0f;
                float beta0 = 0.0f;
                float beta1 = 0.0f;
                if (row0 < valid_len) {
                    long long gate_index0 = (long long)(block_start + row0) * (long long)num_state_heads + (long long)state_head;
                    float _log_0 = logf(alpha[gate_index0]);
                    gate_delta0 = _log_0;
                    beta0 = beta[gate_index0];
                }
                if (row1 < valid_len) {
                    long long gate_index1 = (long long)(block_start + row1) * (long long)num_state_heads + (long long)state_head;
                    float _log_1 = logf(alpha[gate_index1]);
                    gate_delta1 = _log_1;
                    beta1 = beta[gate_index1];
                }
                float _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, gate_delta0, 1, 32);
                float prior0 = _shfl_up_0;
                float _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, gate_delta1, 1, 32);
                float prior1 = _shfl_up_1;
                if (lane >= 1) {
                    gate_delta0 = gate_delta0 + prior0;
                    gate_delta1 = gate_delta1 + prior1;
                }
                float _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, gate_delta0, 2, 32);
                float prior0_0 = _shfl_up_2;
                float _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, gate_delta1, 2, 32);
                float prior1_1 = _shfl_up_3;
                if (lane >= 2) {
                    gate_delta0 = gate_delta0 + prior0_0;
                    gate_delta1 = gate_delta1 + prior1_1;
                }
                float _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, gate_delta0, 4, 32);
                float prior0_2 = _shfl_up_4;
                float _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, gate_delta1, 4, 32);
                float prior1_3 = _shfl_up_5;
                if (lane >= 4) {
                    gate_delta0 = gate_delta0 + prior0_2;
                    gate_delta1 = gate_delta1 + prior1_3;
                }
                float _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, gate_delta0, 8, 32);
                float prior0_4 = _shfl_up_6;
                float _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, gate_delta1, 8, 32);
                float prior1_5 = _shfl_up_7;
                if (lane >= 8) {
                    gate_delta0 = gate_delta0 + prior0_4;
                    gate_delta1 = gate_delta1 + prior1_5;
                }
                float _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, gate_delta0, 16, 32);
                float prior0_6 = _shfl_up_8;
                float _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, gate_delta1, 16, 32);
                float prior1_7 = _shfl_up_9;
                if (lane >= 16) {
                    gate_delta0 = gate_delta0 + prior0_6;
                    gate_delta1 = gate_delta1 + prior1_7;
                }
                float _shfl_0 = __shfl_sync(0xFFFFFFFF, gate_delta0, 31);
                gate_delta1 = gate_delta1 + _shfl_0;
                shared_gate[row0] = gate_delta0;
                shared_gate[row1] = gate_delta1;
                shared_beta[row0] = beta0;
                shared_beta[row1] = beta1;
            }
            __syncthreads();
            #pragma unroll 1
            for (int block_row = 0; block_row < 64; block_row++) {
                float k_value = 0.0f;
                if (valid_len > block_row) {
                    long long k_base = ((long long)(block_start + block_row) * (long long)num_k_heads + (long long)key_head) * 128;
                    k_value = (float)k[k_base + (long long)value_dim];
                }
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
                    if (valid_len > block_row) {
                        float _rsqrt_0 = rsqrtf(norm_sq + 1e-06f);
                        k_value = k_value * _rsqrt_0;
                    }
                }
                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(k_value);
                shared_k_block[block_row * 128 + value_dim] = _cvt_bf16_0;
                __syncthreads();
            }
            #pragma unroll 1
            for (int system_elem = tid; system_elem < 4096; system_elem += 128) {
                int system_row = system_elem / 64;
                int system_col = system_elem % 64;
                float system_value = 0.0f;
                if (system_row < valid_len && system_col < valid_len && system_col < system_row) {
                    float gram_value = 0.0f;
                    #pragma unroll 1
                    for (int gram_dim = 0; gram_dim < 128; gram_dim++) {
                        float _cvt_f32_0 = __bfloat162float(shared_k_block[system_row * 128 + gram_dim]);
                        float _cvt_f32_1 = __bfloat162float(shared_k_block[system_col * 128 + gram_dim]);
                        float _fma_0 = __fmaf_rn(_cvt_f32_0, _cvt_f32_1, gram_value);
                        gram_value = _fma_0;
                    }
                    float gate_difference = shared_gate[system_row] - shared_gate[system_col];
                    float gate_ratio = 0.0f;
                    if (gate_difference <= 0.0f) {
                        float _exp_0 = expf(gate_difference);
                        gate_ratio = _exp_0;
                    }
                    system_value = gram_value * gate_ratio * shared_beta[system_row];
                }
                shared_system[system_elem] = system_value;
                shared_inverse[system_elem] = 0.0f;
            }
            __syncthreads();
            #pragma unroll 1
            for (int solve_row = 0; solve_row < 64; solve_row++) {
                if (value_dim < 64) {
                    int solve_col = value_dim;
                    float inverse_value = 0.0f;
                    if (valid_len > solve_row) {
                        if (solve_col == solve_row) {
                            inverse_value = 1.0f;
                        } else if (solve_col < solve_row) {
                            inverse_value = -shared_system[solve_row * 64 + solve_col];
                            #pragma unroll 1
                            for (int solve_inner = 0; solve_inner < 64; solve_inner++) {
                                if (solve_col < solve_inner && solve_inner < solve_row) {
                                    float _fma_1 = __fmaf_rn(-shared_system[solve_row * 64 + solve_inner], shared_inverse[solve_inner * 64 + solve_col], inverse_value);
                                    inverse_value = _fma_1;
                                }
                            }
                        }
                    } else if (solve_col == solve_row) {
                        inverse_value = 1.0f;
                    }
                    shared_inverse[solve_row * 64 + solve_col] = inverse_value;
                }
                __syncthreads();
            }
            #pragma unroll 1
            for (int a_elem = tid; a_elem < 4096; a_elem += 128) {
                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(shared_inverse[a_elem]);
                shared_a[a_elem] = _cvt_bf16_1;
            }
            __syncthreads();
            #pragma unroll 1
            for (int output_row = 0; output_row < 64; output_row++) {
                float u_value = 0.0f;
                float w_value = 0.0f;
                if (valid_len > output_row) {
                    #pragma unroll 1
                    for (int source_row = 0; source_row < 64; source_row++) {
                        if (valid_len > source_row) {
                            float _cvt_f32_2 = __bfloat162float(shared_a[output_row * 64 + source_row]);
                            float a_value = _cvt_f32_2;
                            int source_token = block_start + source_row;
                            long long v_base = ((long long)source_token * (long long)num_v_heads + (long long)value_head) * 128;
                            __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16((float)v[v_base + (long long)value_dim] * shared_beta[source_row]);
                            float _cvt_f32_3 = __bfloat162float(_cvt_bf16_2);
                            float vb_io = _cvt_f32_3;
                            float _cvt_f32_4 = __bfloat162float(shared_k_block[source_row * 128 + value_dim]);
                            float _exp_1 = expf(shared_gate[source_row]);
                            __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(_cvt_f32_4 * shared_beta[source_row] * _exp_1);
                            float _cvt_f32_5 = __bfloat162float(_cvt_bf16_3);
                            float kb_io = _cvt_f32_5;
                            float _fma_2 = __fmaf_rn(a_value, vb_io, u_value);
                            u_value = _fma_2;
                            float _fma_3 = __fmaf_rn(a_value, kb_io, w_value);
                            w_value = _fma_3;
                        }
                    }
                }
                __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(u_value);
                shared_u_or_r[output_row * 128 + value_dim] = _cvt_bf16_4;
                __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(w_value);
                shared_w[output_row * 128 + value_dim] = _cvt_bf16_5;
            }
            __syncthreads();
            float gate_last = shared_gate[valid_len - 1];
            #pragma unroll 1
            for (int residual_row = 0; residual_row < 64; residual_row++) {
                float residual_value = 0.0f;
                if (valid_len > residual_row) {
                    float projection = 0.0f;
                    #pragma unroll 1
                    for (int projection_dim = 0; projection_dim < 128; projection_dim++) {
                        float _cvt_f32_6 = __bfloat162float(shared_w[residual_row * 128 + projection_dim]);
                        __nv_bfloat16 _cvt_bf16_6 = __float2bfloat16(state_values_exact[projection_dim]);
                        float _cvt_f32_7 = __bfloat162float(_cvt_bf16_6);
                        float _fma_4 = __fmaf_rn(_cvt_f32_6, _cvt_f32_7, projection);
                        projection = _fma_4;
                    }
                    float _cvt_f32_8 = __bfloat162float(shared_u_or_r[residual_row * 128 + value_dim]);
                    residual_value = _cvt_f32_8 - projection;
                    float residual_gate_difference = gate_last - shared_gate[residual_row];
                    float residual_gate = 0.0f;
                    if (residual_gate_difference <= 0.0f) {
                        float _exp_2 = expf(residual_gate_difference);
                        residual_gate = _exp_2;
                    }
                    residual_value = residual_value * residual_gate;
                }
                __nv_bfloat16 _cvt_bf16_7 = __float2bfloat16(residual_value);
                shared_u_or_r[residual_row * 128 + value_dim] = _cvt_bf16_7;
            }
            __syncthreads();
            float _exp_3 = expf(gate_last);
            float state_decay = _exp_3;
            #pragma unroll
            for (int update_dim = 0; update_dim < 128; update_dim++) {
                float update_value = 0.0f;
                #pragma unroll 1
                for (int update_row = 0; update_row < 64; update_row++) {
                    if (valid_len > update_row) {
                        float _cvt_f32_9 = __bfloat162float(shared_k_block[update_row * 128 + update_dim]);
                        float _cvt_f32_10 = __bfloat162float(shared_u_or_r[update_row * 128 + value_dim]);
                        float _fma_5 = __fmaf_rn(_cvt_f32_9, _cvt_f32_10, update_value);
                        update_value = _fma_5;
                    }
                }
                state_values_exact[update_dim] = state_values_exact[update_dim] * state_decay;
                state_values_exact[update_dim] = state_values_exact[update_dim] + update_value;
            }
            __syncthreads();
        }
        #pragma unroll
        for (int key_dim_store = 0; key_dim_store < 128; key_dim_store++) {
            final_state[state_base + (long long)key_dim_store] = state_values_exact[key_dim_store];
        }
    } else {
        float state_values_legacy[128];
        #pragma unroll
        for (int key_dim_init_1 = 0; key_dim_init_1 < 128; key_dim_init_1++) {
            state_values_legacy[key_dim_init_1] = initial_state[state_base + (long long)key_dim_init_1];
        }
        #pragma unroll 1
        for (int token = sequence_start; token < sequence_end; token++) {
            long long k_base_1 = ((long long)token * (long long)num_k_heads + (long long)key_head) * 128;
            float k_value_1 = (float)k[k_base_1 + (long long)value_dim];
            if (normalize_qk != 0) {
                float _warp_reduce_2 = k_value_1 * k_value_1;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
                float warp_norm_sq_1 = _warp_reduce_2;
                if (lane == 0) {
                    reduce_scratch[warp] = warp_norm_sq_1;
                }
                __syncthreads();
                float norm_sq_1 = 0.0f;
                if (lane < 4) {
                    norm_sq_1 = reduce_scratch[lane];
                }
                float _warp_reduce_3 = norm_sq_1;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
                norm_sq_1 = _warp_reduce_3;
                if (warp == 0) {
                    if (elect_sync()) {
                        reduce_scratch[0] = norm_sq_1;
                    }
                }
                __syncthreads();
                norm_sq_1 = reduce_scratch[0];
                float _rsqrt_1 = rsqrtf(norm_sq_1 + 1e-06f);
                legacy_shared_k[value_dim] = k_value_1 * _rsqrt_1;
            } else {
                legacy_shared_k[value_dim] = k_value_1;
            }
            __syncthreads();
            long long gate_index = (long long)token * (long long)num_state_heads + (long long)state_head;
            float alpha_value = alpha[gate_index];
            float beta_value = beta[gate_index];
            float old_value = 0.0f;
            #pragma unroll
            for (int key_dim_dot = 0; key_dim_dot < 128; key_dim_dot++) {
                state_values_legacy[key_dim_dot] = state_values_legacy[key_dim_dot] * alpha_value;
                float _fma_6 = __fmaf_rn(legacy_shared_k[key_dim_dot], state_values_legacy[key_dim_dot], old_value);
                old_value = _fma_6;
            }
            long long v_base_1 = ((long long)token * (long long)num_v_heads + (long long)value_head) * 128;
            float input_value = (float)v[v_base_1 + (long long)value_dim];
            float residual = (input_value - old_value) * beta_value;
            __nv_bfloat16 _cvt_bf16_8 = __float2bfloat16(residual);
            float _cvt_f32_11 = __bfloat162float(_cvt_bf16_8);
            residual = _cvt_f32_11;
            #pragma unroll
            for (int key_dim_update = 0; key_dim_update < 128; key_dim_update++) {
                float _fma_7 = __fmaf_rn(legacy_shared_k[key_dim_update], residual, state_values_legacy[key_dim_update]);
                state_values_legacy[key_dim_update] = _fma_7;
            }
            __syncthreads();
            if (write_output != 0) {
                long long q_base = ((long long)token * (long long)num_q_heads + (long long)query_head) * 128;
                float q_value = (float)q[q_base + (long long)value_dim];
                if (normalize_qk != 0) {
                    float _warp_reduce_4 = q_value * q_value;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_4 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_4, offset);
                    float warp_q_norm_sq = _warp_reduce_4;
                    if (lane == 0) {
                        reduce_scratch[warp] = warp_q_norm_sq;
                    }
                    __syncthreads();
                    float q_norm_sq = 0.0f;
                    if (lane < 4) {
                        q_norm_sq = reduce_scratch[lane];
                    }
                    float _warp_reduce_5 = q_norm_sq;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_5 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_5, offset);
                    q_norm_sq = _warp_reduce_5;
                    if (warp == 0) {
                        if (elect_sync()) {
                            reduce_scratch[0] = q_norm_sq;
                        }
                    }
                    __syncthreads();
                    q_norm_sq = reduce_scratch[0];
                    float _rsqrt_2 = rsqrtf(q_norm_sq + 1e-06f);
                    q_value *= _rsqrt_2;
                }
                legacy_shared_k[value_dim] = q_value;
                __syncthreads();
                float output_value = 0.0f;
                #pragma unroll
                for (int key_dim_output = 0; key_dim_output < 128; key_dim_output++) {
                    float _fma_8 = __fmaf_rn(legacy_shared_k[key_dim_output], state_values_legacy[key_dim_output], output_value);
                    output_value = _fma_8;
                }
                long long output_base = ((long long)token * (long long)num_state_heads + (long long)state_head) * 128;
                output[output_base + (long long)value_dim] = output_value * scale;
                __syncthreads();
            }
        }
        if (write_final_state != 0) {
            #pragma unroll
            for (int key_dim_store_1 = 0; key_dim_store_1 < 128; key_dim_store_1++) {
                final_state[state_base + (long long)key_dim_store_1] = state_values_legacy[key_dim_store_1];
            }
        }
    }
}

} // extern "C"

#undef GDN_CP_INF
#undef NUM_MAIN_STAGES
#undef SMEM_LEGACY_SHARED_K_OFF
#undef SMEM_LEGACY_SHARED_K_STAGE_BYTES
#undef SMEM_LEGACY_SHARED_K_STRIDE
#undef SMEM_REDUCE_SCRATCH_OFF
#undef SMEM_REDUCE_SCRATCH_STAGE_BYTES
#undef SMEM_REDUCE_SCRATCH_STRIDE
#undef SMEM_SHARED_A_OFF
#undef SMEM_SHARED_A_STAGE_BYTES
#undef SMEM_SHARED_A_STRIDE
#undef SMEM_SHARED_BETA_OFF
#undef SMEM_SHARED_BETA_STAGE_BYTES
#undef SMEM_SHARED_BETA_STRIDE
#undef SMEM_SHARED_GATE_OFF
#undef SMEM_SHARED_GATE_STAGE_BYTES
#undef SMEM_SHARED_GATE_STRIDE
#undef SMEM_SHARED_INVERSE_OFF
#undef SMEM_SHARED_INVERSE_STAGE_BYTES
#undef SMEM_SHARED_INVERSE_STRIDE
#undef SMEM_SHARED_K_BLOCK_OFF
#undef SMEM_SHARED_K_BLOCK_STAGE_BYTES
#undef SMEM_SHARED_K_BLOCK_STRIDE
#undef SMEM_SHARED_SYSTEM_OFF
#undef SMEM_SHARED_SYSTEM_STAGE_BYTES
#undef SMEM_SHARED_SYSTEM_STRIDE
#undef SMEM_SHARED_U_OR_R_OFF
#undef SMEM_SHARED_U_OR_R_STAGE_BYTES
#undef SMEM_SHARED_U_OR_R_STRIDE
#undef SMEM_SHARED_W_OFF
#undef SMEM_SHARED_W_STAGE_BYTES
#undef SMEM_SHARED_W_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef legacy_shared_k_addr
#undef reduce_scratch_addr
#undef shared_a_addr
#undef shared_beta_addr
#undef shared_gate_addr
#undef shared_inverse_addr
#undef shared_k_block_addr
#undef shared_system_addr
#undef shared_u_or_r_addr
#undef shared_w_addr
// clang-format on
