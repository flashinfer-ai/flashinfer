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
#define SMEM_SQ_STAGE_BYTES 3072
#define SMEM_SQ_STRIDE 3072
#define SMEM_SK_OFF 3072
#define SMEM_SK_STAGE_BYTES 3072
#define SMEM_SK_STRIDE 3072
#define SMEM_SSCALAR_OFF 6144
#define SMEM_SSCALAR_STAGE_BYTES 32
#define SMEM_SSCALAR_STRIDE 32
#define SMEM_TOTAL 6272
#define THREADS 128
#define H 16
#define HV 64
#define T_STEPS 4
#define TILE_V_WIDE 64
#define UPDATE_STATE 0
#define CACHE_INTERMEDIATE_STATES 1
#define INTERMEDIATE_BATCH_STRIDE 5242880
#define INTERMEDIATE_TOKEN_STRIDE 1048576
#define STRIDED_INPUTS 1
#define SCALE 0.08838834764831845
#define num_v_tiles (128 / TILE_V_WIDE)
#define rows_per_group (TILE_V_WIDE / 8)
#define iters_per_group (rows_per_group / 4)



extern "C" {

__global__ __launch_bounds__(128) void
kernel_gdn_decode_pretranspose_mtp_t4_bf16state_wide128(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ intermediate_state, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, long long state_stride_p0, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2)
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
    float* sK = reinterpret_cast<float*>(smem_raw + 3072);
    const int sK_addr = smem + 3072;
    float* sScalar = reinterpret_cast<float*>(smem_raw + 6144);
    const int sScalar_addr = smem + 6144;

    // === Task calls (dependency order) ===
    int linear_block = blockIdx.x;
    int state_head = linear_block / num_v_tiles;
    int i_v_tile = linear_block - state_head * num_v_tiles;
    int v_tile_base = i_v_tile * TILE_V_WIDE;
    int n = state_head / HV;
    int h = state_head - n * HV;
    int group_idx = tid / 16;
    int lane_in_group = tid - group_idx * 16;
    int lane_local = lane;
    int warp_local = warp;
    int qk_h = h / (HV / H);
    int read_state_slot_raw = initial_state_indices[n];
    int read_state_slot = ((read_state_slot_raw >= 0) ? read_state_slot_raw : 0);
    int write_state_slot_raw = output_state_indices[n];
    int write_state_slot = ((write_state_slot_raw >= 0) ? write_state_slot_raw : 0);
    long long state_slot_stride = state_stride_p0;
    long long read_state_head_base = (long long)read_state_slot * state_slot_stride + (long long)h * 16384;
    long long write_state_head_base = (long long)write_state_slot * state_slot_stride + (long long)h * 16384;
    int k_start = lane_in_group * 8;
    int qk_smem_col = lane_in_group * 12;
    float r_q[8];
    float r_k[8];
    float r_h[8];
    float r_o[4];
    int t_pre_raw = warp_local;
    int t_pre = t_pre_raw;
    {
        long long q_base = (long long)n * q_stride_p0 + (long long)t_pre * q_stride_p1 + (long long)qk_h * q_stride_p2;
        long long k_base = (long long)n * k_stride_p0 + (long long)t_pre * k_stride_p1 + (long long)qk_h * k_stride_p2;
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(q + q_base + (long long)k_start);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&r_q[0 + _blk * 8 + _pair * 2])[0]), "=f"((&r_q[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_0[_pair]));
                }
            }
        }
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(k + k_base + (long long)k_start);
            uint4 _vld_1[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_1[_blk] = _vptr_1[_blk];
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&r_k[0 + _blk * 8 + _pair * 2])[0]), "=f"((&r_k[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_1[_pair]));
                }
            }
        }
    }
    float2 _f2_0 = make_float2(r_q[0], r_q[1]);
    float2 q_raw_pair0 = _f2_0;
    float2 _f2_1 = make_float2(r_q[2], r_q[3]);
    float2 q_raw_pair1 = _f2_1;
    float2 _f2_2 = make_float2(r_q[4], r_q[5]);
    float2 q_raw_pair2 = _f2_2;
    float2 _f2_3 = make_float2(r_q[6], r_q[7]);
    float2 q_raw_pair3 = _f2_3;
    float2 _f2_4 = make_float2(r_k[0], r_k[1]);
    float2 k_raw_pair0 = _f2_4;
    float2 _f2_5 = make_float2(r_k[2], r_k[3]);
    float2 k_raw_pair1 = _f2_5;
    float2 _f2_6 = make_float2(r_k[4], r_k[5]);
    float2 k_raw_pair2 = _f2_6;
    float2 _f2_7 = make_float2(r_k[6], r_k[7]);
    float2 k_raw_pair3 = _f2_7;
    float2 _f2_8 = make_float2(0.0f, 0.0f);
    float2 sum_q_pair = fma_f32x2(q_raw_pair0, q_raw_pair0, _f2_8);
    sum_q_pair = fma_f32x2(q_raw_pair1, q_raw_pair1, sum_q_pair);
    sum_q_pair = fma_f32x2(q_raw_pair2, q_raw_pair2, sum_q_pair);
    sum_q_pair = fma_f32x2(q_raw_pair3, q_raw_pair3, sum_q_pair);
    float2 _f2_9 = make_float2(0.0f, 0.0f);
    float2 sum_k_pair = fma_f32x2(k_raw_pair0, k_raw_pair0, _f2_9);
    sum_k_pair = fma_f32x2(k_raw_pair1, k_raw_pair1, sum_k_pair);
    sum_k_pair = fma_f32x2(k_raw_pair2, k_raw_pair2, sum_k_pair);
    sum_k_pair = fma_f32x2(k_raw_pair3, k_raw_pair3, sum_k_pair);
    float sum_q = sum_q_pair.x + sum_q_pair.y;
    float sum_k = sum_k_pair.x + sum_k_pair.y;
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, sum_q, 8);
    sum_q = sum_q + _shfl_xor_0;
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, sum_k, 8);
    sum_k = sum_k + _shfl_xor_1;
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, sum_q, 4);
    sum_q = sum_q + _shfl_xor_2;
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, sum_k, 4);
    sum_k = sum_k + _shfl_xor_3;
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, sum_q, 2);
    sum_q = sum_q + _shfl_xor_4;
    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, sum_k, 2);
    sum_k = sum_k + _shfl_xor_5;
    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, sum_q, 1);
    sum_q = sum_q + _shfl_xor_6;
    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, sum_k, 1);
    sum_k = sum_k + _shfl_xor_7;
    float _rsqrt_0 = rsqrtf(sum_q + 1e-06f);
    float q_norm = _rsqrt_0 * SCALE;
    float _rsqrt_1 = rsqrtf(sum_k + 1e-06f);
    float k_norm = _rsqrt_1;
    float2 _f2_10 = make_float2(q_norm, q_norm);
    float2 q_norm_pair = _f2_10;
    float2 _f2_11 = make_float2(k_norm, k_norm);
    float2 k_norm_pair = _f2_11;
    float2 q_pair0 = mul_f32x2(q_raw_pair0, q_norm_pair);
    float2 q_pair1 = mul_f32x2(q_raw_pair1, q_norm_pair);
    float2 q_pair2 = mul_f32x2(q_raw_pair2, q_norm_pair);
    float2 q_pair3 = mul_f32x2(q_raw_pair3, q_norm_pair);
    float2 k_pair0 = mul_f32x2(k_raw_pair0, k_norm_pair);
    float2 k_pair1 = mul_f32x2(k_raw_pair1, k_norm_pair);
    float2 k_pair2 = mul_f32x2(k_raw_pair2, k_norm_pair);
    float2 k_pair3 = mul_f32x2(k_raw_pair3, k_norm_pair);
    {
        if (lane_local < 16) {
            sQ[t_pre_raw * 192 + qk_smem_col] = q_pair0.x;
            sQ[t_pre_raw * 192 + qk_smem_col + 1] = q_pair0.y;
            sQ[t_pre_raw * 192 + qk_smem_col + 2] = q_pair1.x;
            sQ[t_pre_raw * 192 + qk_smem_col + 3] = q_pair1.y;
            sQ[t_pre_raw * 192 + qk_smem_col + 4] = q_pair2.x;
            sQ[t_pre_raw * 192 + qk_smem_col + 5] = q_pair2.y;
            sQ[t_pre_raw * 192 + qk_smem_col + 6] = q_pair3.x;
            sQ[t_pre_raw * 192 + qk_smem_col + 7] = q_pair3.y;
            sK[t_pre_raw * 192 + qk_smem_col] = k_pair0.x;
            sK[t_pre_raw * 192 + qk_smem_col + 1] = k_pair0.y;
            sK[t_pre_raw * 192 + qk_smem_col + 2] = k_pair1.x;
            sK[t_pre_raw * 192 + qk_smem_col + 3] = k_pair1.y;
            sK[t_pre_raw * 192 + qk_smem_col + 4] = k_pair2.x;
            sK[t_pre_raw * 192 + qk_smem_col + 5] = k_pair2.y;
            sK[t_pre_raw * 192 + qk_smem_col + 6] = k_pair3.x;
            sK[t_pre_raw * 192 + qk_smem_col + 7] = k_pair3.y;
        }
    }
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
    float g_log = (-_expf_3) * softplus_x;
    float _expf_4 = __expf(g_log);
    float decay_scalar = _expf_4;
    {
        if (lane_local == 0) {
            sScalar[t_pre_raw * 2] = decay_scalar;
            sScalar[t_pre_raw * 2 + 1] = beta_scalar;
        }
        __syncthreads();
    }
    #pragma unroll 1
    for (int iter_idx = 0; iter_idx < iters_per_group; iter_idx++) {
        int v_row_a = v_tile_base + group_idx * rows_per_group + iter_idx * 4;
        int v_row_b = v_row_a + 1;
        int v_row_c = v_row_a + 2;
        int v_row_d = v_row_a + 3;
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)(v_row_a * 128) + (long long)k_start);
            uint4 _vld_2[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_2[_blk] = _vptr_2[_blk];
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&r_h[0 + _blk * 8 + _pair * 2])[0]), "=f"((&r_h[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_2[_pair]));
                }
            }
        }
        float2 _f2_12 = make_float2(r_h[0], r_h[1]);
        float2 h_a_pair0 = _f2_12;
        float2 _f2_13 = make_float2(r_h[2], r_h[3]);
        float2 h_a_pair1 = _f2_13;
        float2 _f2_14 = make_float2(r_h[4], r_h[5]);
        float2 h_a_pair2 = _f2_14;
        float2 _f2_15 = make_float2(r_h[6], r_h[7]);
        float2 h_a_pair3 = _f2_15;
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)(v_row_b * 128) + (long long)k_start);
            uint4 _vld_3[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_3[_blk] = _vptr_3[_blk];
                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&r_h[0 + _blk * 8 + _pair * 2])[0]), "=f"((&r_h[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_3[_pair]));
                }
            }
        }
        float2 _f2_16 = make_float2(r_h[0], r_h[1]);
        float2 h_b_pair0 = _f2_16;
        float2 _f2_17 = make_float2(r_h[2], r_h[3]);
        float2 h_b_pair1 = _f2_17;
        float2 _f2_18 = make_float2(r_h[4], r_h[5]);
        float2 h_b_pair2 = _f2_18;
        float2 _f2_19 = make_float2(r_h[6], r_h[7]);
        float2 h_b_pair3 = _f2_19;
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)(v_row_c * 128) + (long long)k_start);
            uint4 _vld_4[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_4[_blk] = _vptr_4[_blk];
                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&r_h[0 + _blk * 8 + _pair * 2])[0]), "=f"((&r_h[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_4[_pair]));
                }
            }
        }
        float2 _f2_20 = make_float2(r_h[0], r_h[1]);
        float2 h_c_pair0 = _f2_20;
        float2 _f2_21 = make_float2(r_h[2], r_h[3]);
        float2 h_c_pair1 = _f2_21;
        float2 _f2_22 = make_float2(r_h[4], r_h[5]);
        float2 h_c_pair2 = _f2_22;
        float2 _f2_23 = make_float2(r_h[6], r_h[7]);
        float2 h_c_pair3 = _f2_23;
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)(v_row_d * 128) + (long long)k_start);
            uint4 _vld_5[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_5[_blk] = _vptr_5[_blk];
                uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&r_h[0 + _blk * 8 + _pair * 2])[0]), "=f"((&r_h[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_5[_pair]));
                }
            }
        }
        float2 _f2_24 = make_float2(r_h[0], r_h[1]);
        float2 h_d_pair0 = _f2_24;
        float2 _f2_25 = make_float2(r_h[2], r_h[3]);
        float2 h_d_pair1 = _f2_25;
        float2 _f2_26 = make_float2(r_h[4], r_h[5]);
        float2 h_d_pair2 = _f2_26;
        float2 _f2_27 = make_float2(r_h[6], r_h[7]);
        float2 h_d_pair3 = _f2_27;
        #pragma unroll 1
        for (int t = 0; t < T_STEPS; t++) {
            {
                int q_smem_addr = sQ_addr + (unsigned int)((t * 192 + qk_smem_col) * 4);
                int k_smem_addr = sK_addr + (unsigned int)((t * 192 + qk_smem_col) * 4);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&r_q[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_q[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_q[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_q[(0) + 3]))
                    : "r"(q_smem_addr));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&r_q[4])), "=r"(*reinterpret_cast<uint32_t*>(&r_q[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_q[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_q[(4) + 3]))
                    : "r"(q_smem_addr + 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&r_k[0])), "=r"(*reinterpret_cast<uint32_t*>(&r_k[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_k[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_k[(0) + 3]))
                    : "r"(k_smem_addr));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&r_k[4])), "=r"(*reinterpret_cast<uint32_t*>(&r_k[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&r_k[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&r_k[(4) + 3]))
                    : "r"(k_smem_addr + 16));
            }
            float2 _f2_28 = make_float2(r_q[0], r_q[1]);
            float2 q_pair0_0 = _f2_28;
            float2 _f2_29 = make_float2(r_q[2], r_q[3]);
            float2 q_pair1_1 = _f2_29;
            float2 _f2_30 = make_float2(r_q[4], r_q[5]);
            float2 q_pair2_2 = _f2_30;
            float2 _f2_31 = make_float2(r_q[6], r_q[7]);
            float2 q_pair3_3 = _f2_31;
            float2 _f2_32 = make_float2(r_k[0], r_k[1]);
            float2 k_pair0_4 = _f2_32;
            float2 _f2_33 = make_float2(r_k[2], r_k[3]);
            float2 k_pair1_5 = _f2_33;
            float2 _f2_34 = make_float2(r_k[4], r_k[5]);
            float2 k_pair2_6 = _f2_34;
            float2 _f2_35 = make_float2(r_k[6], r_k[7]);
            float2 k_pair3_7 = _f2_35;
            float decay_val = decay_scalar;
            {
                decay_val = sScalar[t * 2];
            }
            float2 _f2_36 = make_float2(decay_val, decay_val);
            float2 decay_pair = _f2_36;
            h_a_pair0 = mul_f32x2(h_a_pair0, decay_pair);
            h_a_pair1 = mul_f32x2(h_a_pair1, decay_pair);
            h_a_pair2 = mul_f32x2(h_a_pair2, decay_pair);
            h_a_pair3 = mul_f32x2(h_a_pair3, decay_pair);
            h_b_pair0 = mul_f32x2(h_b_pair0, decay_pair);
            h_b_pair1 = mul_f32x2(h_b_pair1, decay_pair);
            h_b_pair2 = mul_f32x2(h_b_pair2, decay_pair);
            h_b_pair3 = mul_f32x2(h_b_pair3, decay_pair);
            h_c_pair0 = mul_f32x2(h_c_pair0, decay_pair);
            h_c_pair1 = mul_f32x2(h_c_pair1, decay_pair);
            h_c_pair2 = mul_f32x2(h_c_pair2, decay_pair);
            h_c_pair3 = mul_f32x2(h_c_pair3, decay_pair);
            h_d_pair0 = mul_f32x2(h_d_pair0, decay_pair);
            h_d_pair1 = mul_f32x2(h_d_pair1, decay_pair);
            h_d_pair2 = mul_f32x2(h_d_pair2, decay_pair);
            h_d_pair3 = mul_f32x2(h_d_pair3, decay_pair);
            float2 _f2_37 = make_float2(0.0f, 0.0f);
            float2 sum_hk_a_pair = fma_f32x2(h_a_pair0, k_pair0_4, _f2_37);
            sum_hk_a_pair = fma_f32x2(h_a_pair1, k_pair1_5, sum_hk_a_pair);
            sum_hk_a_pair = fma_f32x2(h_a_pair2, k_pair2_6, sum_hk_a_pair);
            sum_hk_a_pair = fma_f32x2(h_a_pair3, k_pair3_7, sum_hk_a_pair);
            float2 _f2_38 = make_float2(0.0f, 0.0f);
            float2 sum_hk_b_pair = fma_f32x2(h_b_pair0, k_pair0_4, _f2_38);
            sum_hk_b_pair = fma_f32x2(h_b_pair1, k_pair1_5, sum_hk_b_pair);
            sum_hk_b_pair = fma_f32x2(h_b_pair2, k_pair2_6, sum_hk_b_pair);
            sum_hk_b_pair = fma_f32x2(h_b_pair3, k_pair3_7, sum_hk_b_pair);
            float2 _f2_39 = make_float2(0.0f, 0.0f);
            float2 sum_hk_c_pair = fma_f32x2(h_c_pair0, k_pair0_4, _f2_39);
            sum_hk_c_pair = fma_f32x2(h_c_pair1, k_pair1_5, sum_hk_c_pair);
            sum_hk_c_pair = fma_f32x2(h_c_pair2, k_pair2_6, sum_hk_c_pair);
            sum_hk_c_pair = fma_f32x2(h_c_pair3, k_pair3_7, sum_hk_c_pair);
            float2 _f2_40 = make_float2(0.0f, 0.0f);
            float2 sum_hk_d_pair = fma_f32x2(h_d_pair0, k_pair0_4, _f2_40);
            sum_hk_d_pair = fma_f32x2(h_d_pair1, k_pair1_5, sum_hk_d_pair);
            sum_hk_d_pair = fma_f32x2(h_d_pair2, k_pair2_6, sum_hk_d_pair);
            sum_hk_d_pair = fma_f32x2(h_d_pair3, k_pair3_7, sum_hk_d_pair);
            float sum_hk_a = sum_hk_a_pair.x + sum_hk_a_pair.y;
            float sum_hk_b = sum_hk_b_pair.x + sum_hk_b_pair.y;
            float sum_hk_c = sum_hk_c_pair.x + sum_hk_c_pair.y;
            float sum_hk_d = sum_hk_d_pair.x + sum_hk_d_pair.y;
            float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_a, 8);
            sum_hk_a = sum_hk_a + _shfl_xor_8;
            float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_b, 8);
            sum_hk_b = sum_hk_b + _shfl_xor_9;
            float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_c, 8);
            sum_hk_c = sum_hk_c + _shfl_xor_10;
            float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_d, 8);
            sum_hk_d = sum_hk_d + _shfl_xor_11;
            float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_a, 4);
            sum_hk_a = sum_hk_a + _shfl_xor_12;
            float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_b, 4);
            sum_hk_b = sum_hk_b + _shfl_xor_13;
            float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_c, 4);
            sum_hk_c = sum_hk_c + _shfl_xor_14;
            float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_d, 4);
            sum_hk_d = sum_hk_d + _shfl_xor_15;
            float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_a, 2);
            sum_hk_a = sum_hk_a + _shfl_xor_16;
            float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_b, 2);
            sum_hk_b = sum_hk_b + _shfl_xor_17;
            float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_c, 2);
            sum_hk_c = sum_hk_c + _shfl_xor_18;
            float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_d, 2);
            sum_hk_d = sum_hk_d + _shfl_xor_19;
            float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_a, 1);
            sum_hk_a = sum_hk_a + _shfl_xor_20;
            float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_b, 1);
            sum_hk_b = sum_hk_b + _shfl_xor_21;
            float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_c, 1);
            sum_hk_c = sum_hk_c + _shfl_xor_22;
            float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_d, 1);
            sum_hk_d = sum_hk_d + _shfl_xor_23;
            float beta_val = beta_scalar;
            {
                beta_val = sScalar[t * 2 + 1];
            }
            float v_new_a = 0.0f;
            float v_new_b = 0.0f;
            float v_new_c = 0.0f;
            float v_new_d = 0.0f;
            {
                long long v_base = (long long)n * v_stride_p0 + (long long)t * v_stride_p1 + (long long)h * v_stride_p2;
                v_new_a = ((float)v[v_base + (long long)v_row_a] - sum_hk_a) * beta_val;
                v_new_b = ((float)v[v_base + (long long)v_row_b] - sum_hk_b) * beta_val;
                v_new_c = ((float)v[v_base + (long long)v_row_c] - sum_hk_c) * beta_val;
                v_new_d = ((float)v[v_base + (long long)v_row_d] - sum_hk_d) * beta_val;
            }
            int out_base = ((n * T_STEPS + t) * HV + h) * 128;
            float2 _f2_41 = make_float2(v_new_a, v_new_a);
            float2 v_new_a_pair = _f2_41;
            float2 _f2_42 = make_float2(v_new_b, v_new_b);
            float2 v_new_b_pair = _f2_42;
            float2 _f2_43 = make_float2(v_new_c, v_new_c);
            float2 v_new_c_pair = _f2_43;
            float2 _f2_44 = make_float2(v_new_d, v_new_d);
            float2 v_new_d_pair = _f2_44;
            h_a_pair0 = fma_f32x2(k_pair0_4, v_new_a_pair, h_a_pair0);
            h_a_pair1 = fma_f32x2(k_pair1_5, v_new_a_pair, h_a_pair1);
            h_a_pair2 = fma_f32x2(k_pair2_6, v_new_a_pair, h_a_pair2);
            h_a_pair3 = fma_f32x2(k_pair3_7, v_new_a_pair, h_a_pair3);
            h_b_pair0 = fma_f32x2(k_pair0_4, v_new_b_pair, h_b_pair0);
            h_b_pair1 = fma_f32x2(k_pair1_5, v_new_b_pair, h_b_pair1);
            h_b_pair2 = fma_f32x2(k_pair2_6, v_new_b_pair, h_b_pair2);
            h_b_pair3 = fma_f32x2(k_pair3_7, v_new_b_pair, h_b_pair3);
            h_c_pair0 = fma_f32x2(k_pair0_4, v_new_c_pair, h_c_pair0);
            h_c_pair1 = fma_f32x2(k_pair1_5, v_new_c_pair, h_c_pair1);
            h_c_pair2 = fma_f32x2(k_pair2_6, v_new_c_pair, h_c_pair2);
            h_c_pair3 = fma_f32x2(k_pair3_7, v_new_c_pair, h_c_pair3);
            h_d_pair0 = fma_f32x2(k_pair0_4, v_new_d_pair, h_d_pair0);
            h_d_pair1 = fma_f32x2(k_pair1_5, v_new_d_pair, h_d_pair1);
            h_d_pair2 = fma_f32x2(k_pair2_6, v_new_d_pair, h_d_pair2);
            h_d_pair3 = fma_f32x2(k_pair3_7, v_new_d_pair, h_d_pair3);
            {
                long long cache_head_base = (long long)n * (long long)INTERMEDIATE_BATCH_STRIDE + (long long)t * (long long)INTERMEDIATE_TOKEN_STRIDE + (long long)h * 16384;
                r_h[0] = h_a_pair0.x;
                r_h[1] = h_a_pair0.y;
                r_h[2] = h_a_pair1.x;
                r_h[3] = h_a_pair1.y;
                r_h[4] = h_a_pair2.x;
                r_h[5] = h_a_pair2.y;
                r_h[6] = h_a_pair3.x;
                r_h[7] = h_a_pair3.y;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(r_h[0 + 0], r_h[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(r_h[0 + 2], r_h[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(r_h[0 + 4], r_h[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(r_h[0 + 6], r_h[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(intermediate_state))[cache_head_base + (long long)(v_row_a * 128) + (long long)k_start + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
                r_h[0] = h_b_pair0.x;
                r_h[1] = h_b_pair0.y;
                r_h[2] = h_b_pair1.x;
                r_h[3] = h_b_pair1.y;
                r_h[4] = h_b_pair2.x;
                r_h[5] = h_b_pair2.y;
                r_h[6] = h_b_pair3.x;
                r_h[7] = h_b_pair3.y;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(r_h[0 + 0], r_h[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(r_h[0 + 2], r_h[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(r_h[0 + 4], r_h[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(r_h[0 + 6], r_h[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(intermediate_state))[cache_head_base + (long long)(v_row_b * 128) + (long long)k_start + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
                r_h[0] = h_c_pair0.x;
                r_h[1] = h_c_pair0.y;
                r_h[2] = h_c_pair1.x;
                r_h[3] = h_c_pair1.y;
                r_h[4] = h_c_pair2.x;
                r_h[5] = h_c_pair2.y;
                r_h[6] = h_c_pair3.x;
                r_h[7] = h_c_pair3.y;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(r_h[0 + 0], r_h[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(r_h[0 + 2], r_h[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(r_h[0 + 4], r_h[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(r_h[0 + 6], r_h[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(intermediate_state))[cache_head_base + (long long)(v_row_c * 128) + (long long)k_start + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
                r_h[0] = h_d_pair0.x;
                r_h[1] = h_d_pair0.y;
                r_h[2] = h_d_pair1.x;
                r_h[3] = h_d_pair1.y;
                r_h[4] = h_d_pair2.x;
                r_h[5] = h_d_pair2.y;
                r_h[6] = h_d_pair3.x;
                r_h[7] = h_d_pair3.y;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(r_h[0 + 0], r_h[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(r_h[0 + 2], r_h[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(r_h[0 + 4], r_h[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(r_h[0 + 6], r_h[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(intermediate_state))[cache_head_base + (long long)(v_row_d * 128) + (long long)k_start + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
            float2 _f2_45 = make_float2(0.0f, 0.0f);
            float2 sum_hq_a_pair = fma_f32x2(h_a_pair0, q_pair0_0, _f2_45);
            sum_hq_a_pair = fma_f32x2(h_a_pair1, q_pair1_1, sum_hq_a_pair);
            sum_hq_a_pair = fma_f32x2(h_a_pair2, q_pair2_2, sum_hq_a_pair);
            sum_hq_a_pair = fma_f32x2(h_a_pair3, q_pair3_3, sum_hq_a_pair);
            float2 _f2_46 = make_float2(0.0f, 0.0f);
            float2 sum_hq_b_pair = fma_f32x2(h_b_pair0, q_pair0_0, _f2_46);
            sum_hq_b_pair = fma_f32x2(h_b_pair1, q_pair1_1, sum_hq_b_pair);
            sum_hq_b_pair = fma_f32x2(h_b_pair2, q_pair2_2, sum_hq_b_pair);
            sum_hq_b_pair = fma_f32x2(h_b_pair3, q_pair3_3, sum_hq_b_pair);
            float2 _f2_47 = make_float2(0.0f, 0.0f);
            float2 sum_hq_c_pair = fma_f32x2(h_c_pair0, q_pair0_0, _f2_47);
            sum_hq_c_pair = fma_f32x2(h_c_pair1, q_pair1_1, sum_hq_c_pair);
            sum_hq_c_pair = fma_f32x2(h_c_pair2, q_pair2_2, sum_hq_c_pair);
            sum_hq_c_pair = fma_f32x2(h_c_pair3, q_pair3_3, sum_hq_c_pair);
            float2 _f2_48 = make_float2(0.0f, 0.0f);
            float2 sum_hq_d_pair = fma_f32x2(h_d_pair0, q_pair0_0, _f2_48);
            sum_hq_d_pair = fma_f32x2(h_d_pair1, q_pair1_1, sum_hq_d_pair);
            sum_hq_d_pair = fma_f32x2(h_d_pair2, q_pair2_2, sum_hq_d_pair);
            sum_hq_d_pair = fma_f32x2(h_d_pair3, q_pair3_3, sum_hq_d_pair);
            float sum_hq_a = sum_hq_a_pair.x + sum_hq_a_pair.y;
            float sum_hq_b = sum_hq_b_pair.x + sum_hq_b_pair.y;
            float sum_hq_c = sum_hq_c_pair.x + sum_hq_c_pair.y;
            float sum_hq_d = sum_hq_d_pair.x + sum_hq_d_pair.y;
            float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_a, 8);
            sum_hq_a = sum_hq_a + _shfl_xor_24;
            float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_b, 8);
            sum_hq_b = sum_hq_b + _shfl_xor_25;
            float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_c, 8);
            sum_hq_c = sum_hq_c + _shfl_xor_26;
            float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_d, 8);
            sum_hq_d = sum_hq_d + _shfl_xor_27;
            float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_a, 4);
            sum_hq_a = sum_hq_a + _shfl_xor_28;
            float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_b, 4);
            sum_hq_b = sum_hq_b + _shfl_xor_29;
            float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_c, 4);
            sum_hq_c = sum_hq_c + _shfl_xor_30;
            float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_d, 4);
            sum_hq_d = sum_hq_d + _shfl_xor_31;
            float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_a, 2);
            sum_hq_a = sum_hq_a + _shfl_xor_32;
            float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_b, 2);
            sum_hq_b = sum_hq_b + _shfl_xor_33;
            float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_c, 2);
            sum_hq_c = sum_hq_c + _shfl_xor_34;
            float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_d, 2);
            sum_hq_d = sum_hq_d + _shfl_xor_35;
            float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_a, 1);
            sum_hq_a = sum_hq_a + _shfl_xor_36;
            float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_b, 1);
            sum_hq_b = sum_hq_b + _shfl_xor_37;
            float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_c, 1);
            sum_hq_c = sum_hq_c + _shfl_xor_38;
            float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_d, 1);
            sum_hq_d = sum_hq_d + _shfl_xor_39;
            if (lane_in_group == 0) {
                r_o[0] = sum_hq_a;
                r_o[1] = sum_hq_b;
                r_o[2] = sum_hq_c;
                r_o[3] = sum_hq_d;
                {
                    uint2 _pk2;
                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                    _pk[0] = __floats2bfloat162_rn(r_o[0 + 0], r_o[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(r_o[0 + 2], r_o[0 + 3]);
                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[out_base + v_row_a]) = _pk2;
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
#undef SMEM_TOTAL
#undef STRIDED_INPUTS
#undef THREADS
#undef TILE_V_WIDE
#undef T_STEPS
#undef UPDATE_STATE
#undef iters_per_group
#undef num_v_tiles
#undef rows_per_group
#undef sK_addr
#undef sQ_addr
#undef sScalar_addr
// clang-format on
