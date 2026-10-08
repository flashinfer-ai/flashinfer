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
#ifndef TILE_V
#error "TILE_V is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef STRIDED_INPUTS
#error "STRIDED_INPUTS is a downstream specialization of this program; define it on the compile line"
#endif
#ifndef SCALE
#error "SCALE is a downstream specialization of this program; define it on the compile line"
#endif
#define num_v_tiles (128 / TILE_V)
#define rows_per_group (TILE_V / 8)
#define num_blocks ((TILE_V + 31) / 32)
#define block_rows (rows_per_group / num_blocks)
#define LAUNCH_MIN_BLOCKS 9

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_d547d751c50040665077(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ intermediate_state, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, long long state_size_p0, long long state_stride_p0, long long v_stride_p0, long long v_stride_p2, long long q_stride_p0, long long q_stride_p2, long long k_stride_p0, long long k_stride_p2, long long a_stride_p0, long long a_stride_p2, long long b_stride_p0, long long b_stride_p2)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int linear_block = blockIdx.x;
    int state_head = linear_block / num_v_tiles;
    int i_v_tile = linear_block - state_head * num_v_tiles;
    int n = state_head / HV;
    int h = state_head - n * HV;
    int group_idx = tid / 16;
    int lane_in_group = tid - group_idx * 16;
    int qk_h = h / (HV / H);
    long long state_pool_size = state_size_p0;
    int read_state_slot_raw = initial_state_indices[n];
    int read_state_slot_valid = ((read_state_slot_raw >= 0 && state_pool_size > (long long)read_state_slot_raw) ? 1 : 0);
    int read_state_slot = ((read_state_slot_valid != 0) ? read_state_slot_raw : 0);
    int write_state_slot_raw = output_state_indices[n];
    int write_state_slot_valid = ((write_state_slot_raw >= 0 && state_pool_size > (long long)write_state_slot_raw) ? 1 : 0);
    int write_state_slot = ((write_state_slot_valid != 0) ? write_state_slot_raw : 0);
    long long state_slot_stride = state_stride_p0;
    long long read_state_head_base = (long long)read_state_slot * state_slot_stride + (long long)h * 16384;
    long long write_state_head_base = (long long)write_state_slot * state_slot_stride + (long long)h * 16384;
    int k_start = lane_in_group * 8;
    int group_row_base = i_v_tile * TILE_V + group_idx * rows_per_group;
    int out_base = (n * HV + h) * 128;
    long long v_base = (long long)n * (long long)(HV * 128) + (long long)h * 128;
    {
        v_base = (long long)n * v_stride_p0 + (long long)h * v_stride_p2;
    }
    float r_q[8];
    float r_k[8];
    float r_h_a[8];
    float r_h_b[8];
    float r_h_c[8];
    float r_h_d[8];
    float r_h[8];
    float r_o[4];
    int row_a = group_row_base;
    int row_b = row_a + 1;
    int row_c = row_a + 2;
    int row_d = row_a + 3;
    {
        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)(row_a * 128) + (long long)k_start);
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
                    : "=f"((&r_h_a[0 + _blk * 8 + _pair * 2])[0]), "=f"((&r_h_a[0 + _blk * 8 + _pair * 2])[1])
                    : "r"(_vpairs_0[_pair]));
            }
        }
    }
    {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)(row_b * 128) + (long long)k_start);
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
                    : "=f"((&r_h_b[0 + _blk * 8 + _pair * 2])[0]), "=f"((&r_h_b[0 + _blk * 8 + _pair * 2])[1])
                    : "r"(_vpairs_1[_pair]));
            }
        }
    }
    if (block_rows == 4) {
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)(row_c * 128) + (long long)k_start);
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
                        : "=f"((&r_h_c[0 + _blk * 8 + _pair * 2])[0]), "=f"((&r_h_c[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_2[_pair]));
                }
            }
        }
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)(row_d * 128) + (long long)k_start);
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
                        : "=f"((&r_h_d[0 + _blk * 8 + _pair * 2])[0]), "=f"((&r_h_d[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_3[_pair]));
                }
            }
        }
    }
    {
        long long q_base = (long long)n * q_stride_p0 + (long long)qk_h * q_stride_p2;
        long long k_base = (long long)n * k_stride_p0 + (long long)qk_h * k_stride_p2;
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(q + q_base + (long long)k_start);
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
                        : "=f"((&r_q[0 + _blk * 8 + _pair * 2])[0]), "=f"((&r_q[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_4[_pair]));
                }
            }
        }
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(k + k_base + (long long)k_start);
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
                        : "=f"((&r_k[0 + _blk * 8 + _pair * 2])[0]), "=f"((&r_k[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_5[_pair]));
                }
            }
        }
    }
    float x = 0.0f;
    float b_raw = 0.0f;
    {
        long long a_index = (long long)n * a_stride_p0 + (long long)h * a_stride_p2;
        long long b_index = (long long)n * b_stride_p0 + (long long)h * b_stride_p2;
        x = (float)a[a_index] + dt_bias[h];
        b_raw = (float)b[b_index];
    }
    float a_log_h = A_log[h];
    float v_a = (float)v[v_base + (long long)row_a];
    float v_b = (float)v[v_base + (long long)row_b];
    float v_c = 0.0f;
    float v_d = 0.0f;
    if (block_rows == 4) {
        v_c = (float)v[v_base + (long long)row_c];
        v_d = (float)v[v_base + (long long)row_d];
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
    float2 sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair0, q_raw_pair0, _f2_8);
    sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair1, q_raw_pair1, sum_q_pair);
    sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair2, q_raw_pair2, sum_q_pair);
    sum_q_pair = fma_f32x2_rn_ftz(q_raw_pair3, q_raw_pair3, sum_q_pair);
    float2 _f2_9 = make_float2(0.0f, 0.0f);
    float2 sum_k_pair = fma_f32x2_rn_ftz(k_raw_pair0, k_raw_pair0, _f2_9);
    sum_k_pair = fma_f32x2_rn_ftz(k_raw_pair1, k_raw_pair1, sum_k_pair);
    sum_k_pair = fma_f32x2_rn_ftz(k_raw_pair2, k_raw_pair2, sum_k_pair);
    sum_k_pair = fma_f32x2_rn_ftz(k_raw_pair3, k_raw_pair3, sum_k_pair);
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
    float scale_f32 = SCALE;
    float _rsqrt_0 = rsqrtf(sum_q + 1e-06f);
    float q_norm = _rsqrt_0 * scale_f32;
    float _rsqrt_1 = rsqrtf(sum_k + 1e-06f);
    float k_norm = _rsqrt_1;
    float2 _f2_10 = make_float2(q_norm, q_norm);
    float2 q_norm_pair = _f2_10;
    float2 _f2_11 = make_float2(k_norm, k_norm);
    float2 k_norm_pair = _f2_11;
    float2 _mul_f32x2_0;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&q_raw_pair0), "l"(*(const unsigned long long*)&q_norm_pair));
    float2 q_pair0 = _mul_f32x2_0;
    float2 _mul_f32x2_1;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&q_raw_pair1), "l"(*(const unsigned long long*)&q_norm_pair));
    float2 q_pair1 = _mul_f32x2_1;
    float2 _mul_f32x2_2;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&q_raw_pair2), "l"(*(const unsigned long long*)&q_norm_pair));
    float2 q_pair2 = _mul_f32x2_2;
    float2 _mul_f32x2_3;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&q_raw_pair3), "l"(*(const unsigned long long*)&q_norm_pair));
    float2 q_pair3 = _mul_f32x2_3;
    float2 _mul_f32x2_4;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&k_raw_pair0), "l"(*(const unsigned long long*)&k_norm_pair));
    float2 k_pair0 = _mul_f32x2_4;
    float2 _mul_f32x2_5;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&k_raw_pair1), "l"(*(const unsigned long long*)&k_norm_pair));
    float2 k_pair1 = _mul_f32x2_5;
    float2 _mul_f32x2_6;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_6) : "l"(*(const unsigned long long*)&k_raw_pair2), "l"(*(const unsigned long long*)&k_norm_pair));
    float2 k_pair2 = _mul_f32x2_6;
    float2 _mul_f32x2_7;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_7) : "l"(*(const unsigned long long*)&k_raw_pair3), "l"(*(const unsigned long long*)&k_norm_pair));
    float2 k_pair3 = _mul_f32x2_7;
    float _expf_0 = __expf(-b_raw);
    float _rcp_0 = approx_rcp(1.0f + _expf_0);
    float beta_val = _rcp_0;
    float _expf_1 = __expf(x);
    float _log2_0;
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(1.0f + _expf_1));
    float softplus_val = _log2_0 * 0.6931471805599453f;
    float softplus_x = ((x <= 20.0f) ? softplus_val : x);
    float _expf_2 = __expf(a_log_h);
    float _expf_3 = __expf((-_expf_2) * softplus_x);
    float decay_val = _expf_3;
    float2 _f2_12 = make_float2(decay_val, decay_val);
    float2 decay_pair = _f2_12;
    #pragma unroll 1
    for (int blk = 0; blk < num_blocks; blk++) {
        int cur_a = group_row_base + blk * block_rows;
        int cur_b = cur_a + 1;
        int cur_c = cur_a + 2;
        int cur_d = cur_a + 3;
        float2 _f2_13 = make_float2(r_h_a[0], r_h_a[1]);
        float2 _mul_f32x2_8;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_8) : "l"(*(const unsigned long long*)&_f2_13), "l"(*(const unsigned long long*)&decay_pair));
        float2 h_a_pair0 = _mul_f32x2_8;
        float2 _f2_14 = make_float2(r_h_a[2], r_h_a[3]);
        float2 _mul_f32x2_9;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_9) : "l"(*(const unsigned long long*)&_f2_14), "l"(*(const unsigned long long*)&decay_pair));
        float2 h_a_pair1 = _mul_f32x2_9;
        float2 _f2_15 = make_float2(r_h_a[4], r_h_a[5]);
        float2 _mul_f32x2_10;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_10) : "l"(*(const unsigned long long*)&_f2_15), "l"(*(const unsigned long long*)&decay_pair));
        float2 h_a_pair2 = _mul_f32x2_10;
        float2 _f2_16 = make_float2(r_h_a[6], r_h_a[7]);
        float2 _mul_f32x2_11;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_11) : "l"(*(const unsigned long long*)&_f2_16), "l"(*(const unsigned long long*)&decay_pair));
        float2 h_a_pair3 = _mul_f32x2_11;
        float2 _f2_17 = make_float2(0.0f, 0.0f);
        float2 sum_hk_a_pair = fma_f32x2_rn_ftz(h_a_pair0, k_pair0, _f2_17);
        sum_hk_a_pair = fma_f32x2_rn_ftz(h_a_pair1, k_pair1, sum_hk_a_pair);
        sum_hk_a_pair = fma_f32x2_rn_ftz(h_a_pair2, k_pair2, sum_hk_a_pair);
        sum_hk_a_pair = fma_f32x2_rn_ftz(h_a_pair3, k_pair3, sum_hk_a_pair);
        float sum_hk_a = sum_hk_a_pair.x + sum_hk_a_pair.y;
        float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_a, 8);
        sum_hk_a = sum_hk_a + _shfl_xor_8;
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_a, 4);
        sum_hk_a = sum_hk_a + _shfl_xor_9;
        float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_a, 2);
        sum_hk_a = sum_hk_a + _shfl_xor_10;
        float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_a, 1);
        sum_hk_a = sum_hk_a + _shfl_xor_11;
        float v_new_a = (v_a - sum_hk_a) * beta_val;
        float2 _f2_18 = make_float2(v_new_a, v_new_a);
        float2 v_new_a_pair = _f2_18;
        h_a_pair0 = fma_f32x2_rn_ftz(k_pair0, v_new_a_pair, h_a_pair0);
        h_a_pair1 = fma_f32x2_rn_ftz(k_pair1, v_new_a_pair, h_a_pair1);
        h_a_pair2 = fma_f32x2_rn_ftz(k_pair2, v_new_a_pair, h_a_pair2);
        h_a_pair3 = fma_f32x2_rn_ftz(k_pair3, v_new_a_pair, h_a_pair3);
        float2 _f2_19 = make_float2(0.0f, 0.0f);
        float2 sum_hq_a_pair = fma_f32x2_rn_ftz(h_a_pair0, q_pair0, _f2_19);
        sum_hq_a_pair = fma_f32x2_rn_ftz(h_a_pair1, q_pair1, sum_hq_a_pair);
        sum_hq_a_pair = fma_f32x2_rn_ftz(h_a_pair2, q_pair2, sum_hq_a_pair);
        sum_hq_a_pair = fma_f32x2_rn_ftz(h_a_pair3, q_pair3, sum_hq_a_pair);
        float sum_hq_a = sum_hq_a_pair.x + sum_hq_a_pair.y;
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_a, 8);
        sum_hq_a = sum_hq_a + _shfl_xor_12;
        float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_a, 4);
        sum_hq_a = sum_hq_a + _shfl_xor_13;
        float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_a, 2);
        sum_hq_a = sum_hq_a + _shfl_xor_14;
        float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_a, 1);
        sum_hq_a = sum_hq_a + _shfl_xor_15;
        float2 _f2_20 = make_float2(r_h_b[0], r_h_b[1]);
        float2 _mul_f32x2_12;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_12) : "l"(*(const unsigned long long*)&_f2_20), "l"(*(const unsigned long long*)&decay_pair));
        float2 h_b_pair0 = _mul_f32x2_12;
        float2 _f2_21 = make_float2(r_h_b[2], r_h_b[3]);
        float2 _mul_f32x2_13;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_13) : "l"(*(const unsigned long long*)&_f2_21), "l"(*(const unsigned long long*)&decay_pair));
        float2 h_b_pair1 = _mul_f32x2_13;
        float2 _f2_22 = make_float2(r_h_b[4], r_h_b[5]);
        float2 _mul_f32x2_14;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_14) : "l"(*(const unsigned long long*)&_f2_22), "l"(*(const unsigned long long*)&decay_pair));
        float2 h_b_pair2 = _mul_f32x2_14;
        float2 _f2_23 = make_float2(r_h_b[6], r_h_b[7]);
        float2 _mul_f32x2_15;
        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_15) : "l"(*(const unsigned long long*)&_f2_23), "l"(*(const unsigned long long*)&decay_pair));
        float2 h_b_pair3 = _mul_f32x2_15;
        float2 _f2_24 = make_float2(0.0f, 0.0f);
        float2 sum_hk_b_pair = fma_f32x2_rn_ftz(h_b_pair0, k_pair0, _f2_24);
        sum_hk_b_pair = fma_f32x2_rn_ftz(h_b_pair1, k_pair1, sum_hk_b_pair);
        sum_hk_b_pair = fma_f32x2_rn_ftz(h_b_pair2, k_pair2, sum_hk_b_pair);
        sum_hk_b_pair = fma_f32x2_rn_ftz(h_b_pair3, k_pair3, sum_hk_b_pair);
        float sum_hk_b = sum_hk_b_pair.x + sum_hk_b_pair.y;
        float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_b, 8);
        sum_hk_b = sum_hk_b + _shfl_xor_16;
        float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_b, 4);
        sum_hk_b = sum_hk_b + _shfl_xor_17;
        float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_b, 2);
        sum_hk_b = sum_hk_b + _shfl_xor_18;
        float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_b, 1);
        sum_hk_b = sum_hk_b + _shfl_xor_19;
        float v_new_b = (v_b - sum_hk_b) * beta_val;
        float2 _f2_25 = make_float2(v_new_b, v_new_b);
        float2 v_new_b_pair = _f2_25;
        h_b_pair0 = fma_f32x2_rn_ftz(k_pair0, v_new_b_pair, h_b_pair0);
        h_b_pair1 = fma_f32x2_rn_ftz(k_pair1, v_new_b_pair, h_b_pair1);
        h_b_pair2 = fma_f32x2_rn_ftz(k_pair2, v_new_b_pair, h_b_pair2);
        h_b_pair3 = fma_f32x2_rn_ftz(k_pair3, v_new_b_pair, h_b_pair3);
        float2 _f2_26 = make_float2(0.0f, 0.0f);
        float2 sum_hq_b_pair = fma_f32x2_rn_ftz(h_b_pair0, q_pair0, _f2_26);
        sum_hq_b_pair = fma_f32x2_rn_ftz(h_b_pair1, q_pair1, sum_hq_b_pair);
        sum_hq_b_pair = fma_f32x2_rn_ftz(h_b_pair2, q_pair2, sum_hq_b_pair);
        sum_hq_b_pair = fma_f32x2_rn_ftz(h_b_pair3, q_pair3, sum_hq_b_pair);
        float sum_hq_b = sum_hq_b_pair.x + sum_hq_b_pair.y;
        float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_b, 8);
        sum_hq_b = sum_hq_b + _shfl_xor_20;
        float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_b, 4);
        sum_hq_b = sum_hq_b + _shfl_xor_21;
        float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_b, 2);
        sum_hq_b = sum_hq_b + _shfl_xor_22;
        float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_b, 1);
        sum_hq_b = sum_hq_b + _shfl_xor_23;
        float2 _f2_27 = make_float2(0.0f, 0.0f);
        float2 h_c_pair0 = _f2_27;
        float2 _f2_28 = make_float2(0.0f, 0.0f);
        float2 h_c_pair1 = _f2_28;
        float2 _f2_29 = make_float2(0.0f, 0.0f);
        float2 h_c_pair2 = _f2_29;
        float2 _f2_30 = make_float2(0.0f, 0.0f);
        float2 h_c_pair3 = _f2_30;
        float2 _f2_31 = make_float2(0.0f, 0.0f);
        float2 sum_hk_c_pair = _f2_31;
        float sum_hk_c = 0.0f;
        float v_new_c = 0.0f;
        float2 _f2_32 = make_float2(0.0f, 0.0f);
        float2 v_new_c_pair = _f2_32;
        float2 _f2_33 = make_float2(0.0f, 0.0f);
        float2 sum_hq_c_pair = _f2_33;
        float sum_hq_c = 0.0f;
        float2 _f2_34 = make_float2(0.0f, 0.0f);
        float2 h_d_pair0 = _f2_34;
        float2 _f2_35 = make_float2(0.0f, 0.0f);
        float2 h_d_pair1 = _f2_35;
        float2 _f2_36 = make_float2(0.0f, 0.0f);
        float2 h_d_pair2 = _f2_36;
        float2 _f2_37 = make_float2(0.0f, 0.0f);
        float2 h_d_pair3 = _f2_37;
        float2 _f2_38 = make_float2(0.0f, 0.0f);
        float2 sum_hk_d_pair = _f2_38;
        float sum_hk_d = 0.0f;
        float v_new_d = 0.0f;
        float2 _f2_39 = make_float2(0.0f, 0.0f);
        float2 v_new_d_pair = _f2_39;
        float2 _f2_40 = make_float2(0.0f, 0.0f);
        float2 sum_hq_d_pair = _f2_40;
        float sum_hq_d = 0.0f;
        if (block_rows == 4) {
            float2 _f2_41 = make_float2(r_h_c[0], r_h_c[1]);
            float2 _mul_f32x2_16;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_16) : "l"(*(const unsigned long long*)&_f2_41), "l"(*(const unsigned long long*)&decay_pair));
            h_c_pair0 = _mul_f32x2_16;
            float2 _f2_42 = make_float2(r_h_c[2], r_h_c[3]);
            float2 _mul_f32x2_17;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_17) : "l"(*(const unsigned long long*)&_f2_42), "l"(*(const unsigned long long*)&decay_pair));
            h_c_pair1 = _mul_f32x2_17;
            float2 _f2_43 = make_float2(r_h_c[4], r_h_c[5]);
            float2 _mul_f32x2_18;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_18) : "l"(*(const unsigned long long*)&_f2_43), "l"(*(const unsigned long long*)&decay_pair));
            h_c_pair2 = _mul_f32x2_18;
            float2 _f2_44 = make_float2(r_h_c[6], r_h_c[7]);
            float2 _mul_f32x2_19;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_19) : "l"(*(const unsigned long long*)&_f2_44), "l"(*(const unsigned long long*)&decay_pair));
            h_c_pair3 = _mul_f32x2_19;
            float2 _f2_45 = make_float2(0.0f, 0.0f);
            sum_hk_c_pair = fma_f32x2_rn_ftz(h_c_pair0, k_pair0, _f2_45);
            sum_hk_c_pair = fma_f32x2_rn_ftz(h_c_pair1, k_pair1, sum_hk_c_pair);
            sum_hk_c_pair = fma_f32x2_rn_ftz(h_c_pair2, k_pair2, sum_hk_c_pair);
            sum_hk_c_pair = fma_f32x2_rn_ftz(h_c_pair3, k_pair3, sum_hk_c_pair);
            sum_hk_c = sum_hk_c_pair.x + sum_hk_c_pair.y;
            float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_c, 8);
            sum_hk_c = sum_hk_c + _shfl_xor_24;
            float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_c, 4);
            sum_hk_c = sum_hk_c + _shfl_xor_25;
            float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_c, 2);
            sum_hk_c = sum_hk_c + _shfl_xor_26;
            float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_c, 1);
            sum_hk_c = sum_hk_c + _shfl_xor_27;
            v_new_c = (v_c - sum_hk_c) * beta_val;
            float2 _f2_46 = make_float2(v_new_c, v_new_c);
            v_new_c_pair = _f2_46;
            h_c_pair0 = fma_f32x2_rn_ftz(k_pair0, v_new_c_pair, h_c_pair0);
            h_c_pair1 = fma_f32x2_rn_ftz(k_pair1, v_new_c_pair, h_c_pair1);
            h_c_pair2 = fma_f32x2_rn_ftz(k_pair2, v_new_c_pair, h_c_pair2);
            h_c_pair3 = fma_f32x2_rn_ftz(k_pair3, v_new_c_pair, h_c_pair3);
            float2 _f2_47 = make_float2(0.0f, 0.0f);
            sum_hq_c_pair = fma_f32x2_rn_ftz(h_c_pair0, q_pair0, _f2_47);
            sum_hq_c_pair = fma_f32x2_rn_ftz(h_c_pair1, q_pair1, sum_hq_c_pair);
            sum_hq_c_pair = fma_f32x2_rn_ftz(h_c_pair2, q_pair2, sum_hq_c_pair);
            sum_hq_c_pair = fma_f32x2_rn_ftz(h_c_pair3, q_pair3, sum_hq_c_pair);
            sum_hq_c = sum_hq_c_pair.x + sum_hq_c_pair.y;
            float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_c, 8);
            sum_hq_c = sum_hq_c + _shfl_xor_28;
            float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_c, 4);
            sum_hq_c = sum_hq_c + _shfl_xor_29;
            float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_c, 2);
            sum_hq_c = sum_hq_c + _shfl_xor_30;
            float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_c, 1);
            sum_hq_c = sum_hq_c + _shfl_xor_31;
            float2 _f2_48 = make_float2(r_h_d[0], r_h_d[1]);
            float2 _mul_f32x2_20;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_20) : "l"(*(const unsigned long long*)&_f2_48), "l"(*(const unsigned long long*)&decay_pair));
            h_d_pair0 = _mul_f32x2_20;
            float2 _f2_49 = make_float2(r_h_d[2], r_h_d[3]);
            float2 _mul_f32x2_21;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_21) : "l"(*(const unsigned long long*)&_f2_49), "l"(*(const unsigned long long*)&decay_pair));
            h_d_pair1 = _mul_f32x2_21;
            float2 _f2_50 = make_float2(r_h_d[4], r_h_d[5]);
            float2 _mul_f32x2_22;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_22) : "l"(*(const unsigned long long*)&_f2_50), "l"(*(const unsigned long long*)&decay_pair));
            h_d_pair2 = _mul_f32x2_22;
            float2 _f2_51 = make_float2(r_h_d[6], r_h_d[7]);
            float2 _mul_f32x2_23;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_23) : "l"(*(const unsigned long long*)&_f2_51), "l"(*(const unsigned long long*)&decay_pair));
            h_d_pair3 = _mul_f32x2_23;
            float2 _f2_52 = make_float2(0.0f, 0.0f);
            sum_hk_d_pair = fma_f32x2_rn_ftz(h_d_pair0, k_pair0, _f2_52);
            sum_hk_d_pair = fma_f32x2_rn_ftz(h_d_pair1, k_pair1, sum_hk_d_pair);
            sum_hk_d_pair = fma_f32x2_rn_ftz(h_d_pair2, k_pair2, sum_hk_d_pair);
            sum_hk_d_pair = fma_f32x2_rn_ftz(h_d_pair3, k_pair3, sum_hk_d_pair);
            sum_hk_d = sum_hk_d_pair.x + sum_hk_d_pair.y;
            float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_d, 8);
            sum_hk_d = sum_hk_d + _shfl_xor_32;
            float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_d, 4);
            sum_hk_d = sum_hk_d + _shfl_xor_33;
            float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_d, 2);
            sum_hk_d = sum_hk_d + _shfl_xor_34;
            float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, sum_hk_d, 1);
            sum_hk_d = sum_hk_d + _shfl_xor_35;
            v_new_d = (v_d - sum_hk_d) * beta_val;
            float2 _f2_53 = make_float2(v_new_d, v_new_d);
            v_new_d_pair = _f2_53;
            h_d_pair0 = fma_f32x2_rn_ftz(k_pair0, v_new_d_pair, h_d_pair0);
            h_d_pair1 = fma_f32x2_rn_ftz(k_pair1, v_new_d_pair, h_d_pair1);
            h_d_pair2 = fma_f32x2_rn_ftz(k_pair2, v_new_d_pair, h_d_pair2);
            h_d_pair3 = fma_f32x2_rn_ftz(k_pair3, v_new_d_pair, h_d_pair3);
            float2 _f2_54 = make_float2(0.0f, 0.0f);
            sum_hq_d_pair = fma_f32x2_rn_ftz(h_d_pair0, q_pair0, _f2_54);
            sum_hq_d_pair = fma_f32x2_rn_ftz(h_d_pair1, q_pair1, sum_hq_d_pair);
            sum_hq_d_pair = fma_f32x2_rn_ftz(h_d_pair2, q_pair2, sum_hq_d_pair);
            sum_hq_d_pair = fma_f32x2_rn_ftz(h_d_pair3, q_pair3, sum_hq_d_pair);
            sum_hq_d = sum_hq_d_pair.x + sum_hq_d_pair.y;
            float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_d, 8);
            sum_hq_d = sum_hq_d + _shfl_xor_36;
            float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_d, 4);
            sum_hq_d = sum_hq_d + _shfl_xor_37;
            float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_d, 2);
            sum_hq_d = sum_hq_d + _shfl_xor_38;
            float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, sum_hq_d, 1);
            sum_hq_d = sum_hq_d + _shfl_xor_39;
        }
        if (lane_in_group == 0) {
            if (read_state_slot_valid != 0) {
                r_o[0] = sum_hq_a;
                r_o[1] = sum_hq_b;
                if (block_rows == 4) {
                    r_o[2] = sum_hq_c;
                    r_o[3] = sum_hq_d;
                    {
                        uint2 _pk2;
                        __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                        _pk[0] = __floats2bfloat162_rn(r_o[0 + 0], r_o[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(r_o[0 + 2], r_o[0 + 3]);
                        *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[out_base + cur_a]) = _pk2;
                    }
                } else {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(r_o[0 + 0], r_o[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out))[out_base + cur_a]) = _pk;
                    }
                }
            }
        }
        r_h[0] = h_a_pair0.x;
        r_h[1] = h_a_pair0.y;
        r_h[2] = h_a_pair1.x;
        r_h[3] = h_a_pair1.y;
        r_h[4] = h_a_pair2.x;
        r_h[5] = h_a_pair2.y;
        r_h[6] = h_a_pair3.x;
        r_h[7] = h_a_pair3.y;
        if (read_state_slot_valid != 0 && write_state_slot_valid != 0) {
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(r_h[0 + 0], r_h[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(r_h[0 + 2], r_h[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(r_h[0 + 4], r_h[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(r_h[0 + 6], r_h[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[write_state_head_base + (long long)(cur_a * 128) + (long long)k_start + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
        r_h[0] = h_b_pair0.x;
        r_h[1] = h_b_pair0.y;
        r_h[2] = h_b_pair1.x;
        r_h[3] = h_b_pair1.y;
        r_h[4] = h_b_pair2.x;
        r_h[5] = h_b_pair2.y;
        r_h[6] = h_b_pair3.x;
        r_h[7] = h_b_pair3.y;
        if (read_state_slot_valid != 0 && write_state_slot_valid != 0) {
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(r_h[0 + 0], r_h[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(r_h[0 + 2], r_h[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(r_h[0 + 4], r_h[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(r_h[0 + 6], r_h[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[write_state_head_base + (long long)(cur_b * 128) + (long long)k_start + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
        if (block_rows == 4) {
            r_h[0] = h_c_pair0.x;
            r_h[1] = h_c_pair0.y;
            r_h[2] = h_c_pair1.x;
            r_h[3] = h_c_pair1.y;
            r_h[4] = h_c_pair2.x;
            r_h[5] = h_c_pair2.y;
            r_h[6] = h_c_pair3.x;
            r_h[7] = h_c_pair3.y;
            if (read_state_slot_valid != 0 && write_state_slot_valid != 0) {
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(r_h[0 + 0], r_h[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(r_h[0 + 2], r_h[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(r_h[0 + 4], r_h[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(r_h[0 + 6], r_h[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[write_state_head_base + (long long)(cur_c * 128) + (long long)k_start + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
            r_h[0] = h_d_pair0.x;
            r_h[1] = h_d_pair0.y;
            r_h[2] = h_d_pair1.x;
            r_h[3] = h_d_pair1.y;
            r_h[4] = h_d_pair2.x;
            r_h[5] = h_d_pair2.y;
            r_h[6] = h_d_pair3.x;
            r_h[7] = h_d_pair3.y;
            if (read_state_slot_valid != 0 && write_state_slot_valid != 0) {
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(r_h[0 + 0], r_h[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(r_h[0 + 2], r_h[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(r_h[0 + 4], r_h[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(r_h[0 + 6], r_h[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[write_state_head_base + (long long)(cur_d * 128) + (long long)k_start + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
        }
        if (blk + 1 < num_blocks) {
            int next_a = cur_a + block_rows;
            int next_b = next_a + 1;
            int next_c = next_a + 2;
            int next_d = next_a + 3;
            {
                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)(next_a * 128) + (long long)k_start);
                uint4 _vld_6[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_6[_blk] = _vptr_6[_blk];
                    uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&r_h_a[0 + _blk * 8 + _pair * 2])[0]), "=f"((&r_h_a[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_6[_pair]));
                    }
                }
            }
            v_a = (float)v[v_base + (long long)next_a];
            {
                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)(next_b * 128) + (long long)k_start);
                uint4 _vld_7[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_7[_blk] = _vptr_7[_blk];
                    uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&r_h_b[0 + _blk * 8 + _pair * 2])[0]), "=f"((&r_h_b[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_7[_pair]));
                    }
                }
            }
            v_b = (float)v[v_base + (long long)next_b];
            if (block_rows == 4) {
                {
                    const uint4* _vptr_8 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)(next_c * 128) + (long long)k_start);
                    uint4 _vld_8[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_8[_blk] = _vptr_8[_blk];
                        uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&r_h_c[0 + _blk * 8 + _pair * 2])[0]), "=f"((&r_h_c[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_8[_pair]));
                        }
                    }
                }
                v_c = (float)v[v_base + (long long)next_c];
                {
                    const uint4* _vptr_9 = reinterpret_cast<const uint4*>(state + read_state_head_base + (long long)(next_d * 128) + (long long)k_start);
                    uint4 _vld_9[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_9[_blk] = _vptr_9[_blk];
                        uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&r_h_d[0 + _blk * 8 + _pair * 2])[0]), "=f"((&r_h_d[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_9[_pair]));
                        }
                    }
                }
                v_d = (float)v[v_base + (long long)next_d];
            }
        }
    }
}

} // extern "C"
// clang-format on
