/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_kimi_k3_vision_tower_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_kimi_k3_vision_tower_e0b196f00aea91ddbf42(float* __restrict__ partial_O, float* __restrict__ partial_ML, int* __restrict__ merge_table, int* __restrict__ seg_begin, int* __restrict__ seg_len, __nv_bfloat16* __restrict__ O, int num_heads, int part_rows, float softmax_scale_log2)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int ctas_per_unit = part_rows / 4;
    int unit = blockIdx.x / ctas_per_unit;
    int unit_row = blockIdx.x % ctas_per_unit * 4 + warp;
    int rec = unit * 4;
    int seg = merge_table[rec];
    int packed = merge_table[rec + 1];
    int first_slot = merge_table[rec + 2];
    int parts = merge_table[rec + 3];
    int head = packed >> 16;
    int c = packed & 65535;
    int doc_begin = seg_begin[seg];
    int doc_len = seg_len[seg];
    int local_row = c * part_rows + unit_row;
    if (local_row < doc_len) {
        int d_base = lane * 4;
        float part_m[8];
        float part_l[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            part_m[i] = -CAKE_INF;
            part_l[i] = 0.0f;
            if (parts > i) {
                int stat_row = (first_slot + i) * part_rows + unit_row;
                part_m[i] = partial_ML[stat_row * 2];
                part_l[i] = partial_ML[stat_row * 2 + 1];
            }
        }
        float m_ref = part_m[0];
        #pragma unroll
        for (int i_1 = 1; i_1 < 8; i_1++) {
            float _max_0 = max_noftz(m_ref, part_m[i_1]);
            m_ref = _max_0;
        }
        float safe_ref = ((m_ref == -CAKE_INF) ? 0.0f : m_ref);
        float ref_scaled = safe_ref * softmax_scale_log2;
        float acc[4];
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            acc[e] = 0.0f;
        }
        float l_tot = 0.0f;
        #pragma unroll
        for (int i_2 = 0; i_2 < 8; i_2++) {
            if (parts > i_2) {
                float a_i = 0.0f;
                if (part_m[i_2] > -CAKE_INF) {
                    float _fma_0 = __fmaf_rn(part_m[i_2], softmax_scale_log2, -ref_scaled);
                    float _exp2_0 = approx_exp2(_fma_0);
                    a_i = _exp2_0;
                }
                l_tot = l_tot + part_l[i_2] * a_i;
                float vals[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial_O + ((first_slot + i_2) * part_rows + unit_row) * 128 + d_base);
                    vals[0 + 0] = _v4.x;
                    vals[0 + 1] = _v4.y;
                    vals[0 + 2] = _v4.z;
                    vals[0 + 3] = _v4.w;
                }
                #pragma unroll
                for (int e_1 = 0; e_1 < 4; e_1++) {
                    float _fma_1 = __fmaf_rn(vals[e_1], a_i, acc[e_1]);
                    acc[e_1] = _fma_1;
                }
            }
        }
        float inv;
        if (l_tot != 0.0f && l_tot == l_tot) {
            float _rcp_0 = approx_rcp(l_tot);
            inv = _rcp_0;
        } else {
            inv = 0.0f;
        }
        int out_row = (doc_begin + local_row) * num_heads + head;
        {
            const float2 _prescale2_1 = {inv, inv};
            #if __CUDA_ARCH__ >= 1000
            #pragma unroll
            for (int _ps = 0; _ps < 2; _ps++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(&acc[0])[_ps], _prescale2_1);
            #else
            #pragma unroll
            for (int _ps = 0; _ps < 4; _ps++)
                acc[0 + _ps] *= inv;
            #endif
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(acc[0 + 0], acc[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(acc[0 + 2], acc[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(O + (out_row * 128 + d_base)))[0]) = _pk2;
        }
    }
}

} // extern "C"
