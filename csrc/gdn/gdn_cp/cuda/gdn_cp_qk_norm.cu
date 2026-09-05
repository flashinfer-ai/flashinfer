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
#define THREADS 32



extern "C" {

__global__ __launch_bounds__(32) void
kernel_flashinfer_blackwell_gdn_cp_prefill_qk_norm_f16_v1(__half* __restrict__ q, __half* __restrict__ k, __half* __restrict__ q_normalized, __half* __restrict__ k_normalized, int num_q_heads, int num_k_heads)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int token = blockIdx.x;
    int head = blockIdx.y;
    int elem = lane * 4;
    if (head < num_q_heads) {
        long long q_base = ((long long)token * (long long)num_q_heads + (long long)head) * 128;
        float q_values[4];
        {
            uint2 _vld_0;
            _vld_0 = *reinterpret_cast<const uint2*>(q + q_base + (long long)elem);
            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&q_values[0 + _pair * 2]))
                    : "r"(_vpairs_0[_pair]));
            }
        }
        float q_sq = 0.0f;
        #pragma unroll
        for (int q_item = 0; q_item < 4; q_item++) {
            float _fma_0 = __fmaf_rn(q_values[q_item], q_values[q_item], q_sq);
            q_sq = _fma_0;
        }
        float _warp_reduce_0 = q_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        q_sq = _warp_reduce_0;
        float _rsqrt_0 = rsqrtf(q_sq + 1e-06f);
        float q_inv = _rsqrt_0;
        #pragma unroll
        for (int q_item_out = 0; q_item_out < 4; q_item_out++) {
            q_values[q_item_out] = q_values[q_item_out] * q_inv;
        }
        #pragma unroll
        for (int q_item_store = 0; q_item_store < 4; q_item_store++) {
            *(reinterpret_cast<__half*>(q_normalized + (q_base + (long long)elem + (long long)q_item_store)) + (0)) = __float2half_rn(q_values[q_item_store]);
        }
    }
    if (head < num_k_heads) {
        long long k_base = ((long long)token * (long long)num_k_heads + (long long)head) * 128;
        float k_values[4];
        {
            uint2 _vld_1;
            _vld_1 = *reinterpret_cast<const uint2*>(k + k_base + (long long)elem);
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&k_values[0 + _pair * 2]))
                    : "r"(_vpairs_1[_pair]));
            }
        }
        float k_sq = 0.0f;
        #pragma unroll
        for (int k_item = 0; k_item < 4; k_item++) {
            float _fma_1 = __fmaf_rn(k_values[k_item], k_values[k_item], k_sq);
            k_sq = _fma_1;
        }
        float _warp_reduce_1 = k_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        k_sq = _warp_reduce_1;
        float _rsqrt_1 = rsqrtf(k_sq + 1e-06f);
        float k_inv = _rsqrt_1;
        #pragma unroll
        for (int k_item_out = 0; k_item_out < 4; k_item_out++) {
            k_values[k_item_out] = k_values[k_item_out] * k_inv;
        }
        #pragma unroll
        for (int k_item_store = 0; k_item_store < 4; k_item_store++) {
            *(reinterpret_cast<__half*>(k_normalized + (k_base + (long long)elem + (long long)k_item_store)) + (0)) = __float2half_rn(k_values[k_item_store]);
        }
    }
}

} // extern "C"

#undef GDN_CP_INF
#undef NUM_MAIN_STAGES
#undef THREADS
// clang-format on
