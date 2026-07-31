/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// clang-format off
// Generated from a recurrent-KDA Loom schedule.
// Raw generated body SHA256: 9119163b3b5cb6a8760b6a17a7ce01788a0e1c0078f8c812255225f27e5989e5
// Normalized generated SHA256: 9119163b3b5cb6a8760b6a17a7ce01788a0e1c0078f8c812255225f27e5989e5
// BEGIN FROZEN GENERATED BODY
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) LoomTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) LoomTensorMapPack { LoomTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(32) void
kernel_flashinfer_recurrent_kda_t1_direct(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ beta, __nv_bfloat16* __restrict__ state, __nv_bfloat16* __restrict__ out, int* __restrict__ cu_seqlens, int* __restrict__ ssm_state_indices, int* __restrict__ num_accepted_tokens, float scale, int g_stride_token, int state_stride_slot, int H, int HV, int head_ratio)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int work = blockIdx.x;
    const int value_splits = 16;
    int value_tile = work & value_splits - 1;
    int hv = work / value_splits;
    int n = blockIdx.y;
    int query_head = hv / head_ratio;
    int lane_0 = lane;
    int k_lane = lane_0 & 15;
    int v_lane = lane_0 / 16;
    int raw_token_pos = cu_seqlens[n];
    int seq_len = cu_seqlens[n + 1] - raw_token_pos;
    bool has_token = raw_token_pos >= 0 && raw_token_pos < gridDim.y && seq_len > 0;
    int token_pos = ((has_token) ? raw_token_pos : 0);
    int raw_slot = ssm_state_indices[n];
    int initial_slot = ((raw_slot < 0) ? 0 : raw_slot);
    bool active = raw_slot >= 0 && has_token;
    int tile_row_base = value_tile * 8;
    int elem_start = lane_0 * 4;
    float q_src[4];
    float k_src[4];
    float gate_src[4];
    float q_reg[8];
    float k_reg[8];
    float gate_reg[8];
    float state_rows[32];
    int q_base = (token_pos * H + query_head) * 128 + elem_start;
    int vg_base = (token_pos * HV + hv) * 128;
    int gate_base = token_pos * g_stride_token + hv * 128 + elem_start;
    {
        uint2 _vld_0 = *reinterpret_cast<const uint2*>(q + q_base);
        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&q_src[0 + _pair * 2])[0]), "=f"((&q_src[0 + _pair * 2])[1])
                : "r"(_vpairs_0[_pair]));
        }
    }
    {
        uint2 _vld_1 = *reinterpret_cast<const uint2*>(k + q_base);
        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&k_src[0 + _pair * 2])[0]), "=f"((&k_src[0 + _pair * 2])[1])
                : "r"(_vpairs_1[_pair]));
        }
    }
    {
        uint2 _vld_2 = *reinterpret_cast<const uint2*>(g + gate_base);
        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&gate_src[0 + _pair * 2])[0]), "=f"((&gate_src[0 + _pair * 2])[1])
                : "r"(_vpairs_2[_pair]));
        }
    }
    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    {
        q_sum_sq = q_src[0] * q_src[0] + q_src[1] * q_src[1] + (q_src[2] * q_src[2] + q_src[3] * q_src[3]);
        k_sum_sq = k_src[0] * k_src[0] + k_src[1] * k_src[1] + (k_src[2] * k_src[2] + k_src[3] * k_src[3]);
    }
    float _warp_reduce_0 = q_sum_sq;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    q_sum_sq = _warp_reduce_0;
    float _warp_reduce_1 = k_sum_sq;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
    k_sum_sq = _warp_reduce_1;
    float _rsqrt_0 = rsqrtf(q_sum_sq + 1e-06f);
    float q_scale = _rsqrt_0 * scale;
    float _rsqrt_1 = rsqrtf(k_sum_sq + 1e-06f);
    float k_scale = _rsqrt_1;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int source_lane = 2 * k_lane + i / 4;
        const int source_value = i & 3;
        float _shfl_0 = __shfl_sync(0xFFFFFFFF, q_src[source_value], source_lane);
        q_reg[i] = _shfl_0 * q_scale;
        float _shfl_1 = __shfl_sync(0xFFFFFFFF, k_src[source_value], source_lane);
        k_reg[i] = _shfl_1 * k_scale;
        float _shfl_2 = __shfl_sync(0xFFFFFFFF, gate_src[source_value], source_lane);
        float _expf_0 = __expf(_shfl_2);
        gate_reg[i] = _expf_0;
    }
    float k_dot_q = 0.0f;
    {
        k_dot_q = k_reg[0] * q_reg[0] + k_reg[1] * q_reg[1] + (k_reg[2] * q_reg[2] + k_reg[3] * q_reg[3]) + (k_reg[4] * q_reg[4] + k_reg[5] * q_reg[5] + (k_reg[6] * q_reg[6] + k_reg[7] * q_reg[7]));
    }
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, k_dot_q, 8);
    k_dot_q += _shfl_xor_0;
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, k_dot_q, 4);
    k_dot_q += _shfl_xor_1;
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, k_dot_q, 2);
    k_dot_q += _shfl_xor_2;
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, k_dot_q, 1);
    k_dot_q += _shfl_xor_3;
    float beta_value = (float)beta[token_pos * HV + hv];
    int initial_head_base = initial_slot * state_stride_slot + hv * 128 * 128;
    int output_head_base = (token_pos * HV + hv) * 128;
    int output_state_head_base = initial_slot * state_stride_slot + hv * 128 * 128;
    #pragma unroll 1
    for (int row_block = 0; row_block < 1; row_block++) {
        #pragma unroll
        for (int row_local = 0; row_local < 4; row_local++) {
            int row_in_tile = v_lane + 2 * (row_block * 4 + row_local);
            int value_row = tile_row_base + row_in_tile;
            int state_base = initial_head_base + value_row * 128 + k_lane * 8;
            {
                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(state + state_base);
                uint4 _vld_3[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_vld_3[_blk].x), "=r"(_vld_3[_blk].y), "=r"(_vld_3[_blk].z), "=r"(_vld_3[_blk].w)
                        : "l"(_vptr_3 + _blk) : "memory");
                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&state_rows[row_local * 8 + _blk * 8 + _pair * 2])[0]), "=f"((&state_rows[row_local * 8 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_3[_pair]));
                    }
                }
            }
        }
        #pragma unroll
        for (int row_local_1 = 0; row_local_1 < 4; row_local_1++) {
            int row_in_tile_1 = v_lane + 2 * (row_block * 4 + row_local_1);
            int value_row_1 = tile_row_base + row_in_tile_1;
            float pred = 0.0f;
            float base = 0.0f;
            {
                #pragma unroll
                for (int i_1 = 0; i_1 < 8; i_1++) {
                    float h_decay = state_rows[row_local_1 * 8 + i_1] * gate_reg[i_1];
                    pred += h_decay * k_reg[i_1];
                    base += h_decay * q_reg[i_1];
                }
            }
            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, pred, 8);
            pred += _shfl_xor_4;
            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, pred, 4);
            pred += _shfl_xor_5;
            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, pred, 2);
            pred += _shfl_xor_6;
            float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, pred, 1);
            pred += _shfl_xor_7;
            float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, base, 8);
            base += _shfl_xor_8;
            float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, base, 4);
            base += _shfl_xor_9;
            float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, base, 2);
            base += _shfl_xor_10;
            float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, base, 1);
            base += _shfl_xor_11;
            float v_value = 0.0f;
            if (k_lane == 0) {
                v_value = (float)v[vg_base + value_row_1];
            }
            float _shfl_3 = __shfl_sync(0xFFFFFFFF, v_value, v_lane * 16);
            v_value = _shfl_3;
            float delta = (v_value - pred) * beta_value;
            #pragma unroll
            for (int i_2 = 0; i_2 < 8; i_2++) {
                state_rows[row_local_1 * 8 + i_2] = state_rows[row_local_1 * 8 + i_2] * gate_reg[i_2] + delta * k_reg[i_2];
            }
            if (active) {
                int state_base_1 = output_state_head_base + value_row_1 * 128 + k_lane * 8;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(state_rows[row_local_1 * 8 + 0], state_rows[row_local_1 * 8 + 1]);
                    _pk[1] = __floats2bfloat162_rn(state_rows[row_local_1 * 8 + 2], state_rows[row_local_1 * 8 + 3]);
                    _pk[2] = __floats2bfloat162_rn(state_rows[row_local_1 * 8 + 4], state_rows[row_local_1 * 8 + 5]);
                    _pk[3] = __floats2bfloat162_rn(state_rows[row_local_1 * 8 + 6], state_rows[row_local_1 * 8 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[state_base_1 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
                if (k_lane == 0) {
                    out[output_head_base + value_row_1] = base + delta * k_dot_q;
                }
            } else if (has_token && k_lane == 0) {
                out[output_head_base + value_row_1] = 0.0f;
            }
        }
    }
}

} // extern "C"
// END FROZEN GENERATED BODY
// clang-format on
