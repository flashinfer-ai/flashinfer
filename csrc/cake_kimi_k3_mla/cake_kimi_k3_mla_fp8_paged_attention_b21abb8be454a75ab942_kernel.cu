/*
 * Copyright (c) 2026 by FlashInfer team.
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

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_kimi_k3_mla_fp8_paged_attention_b21abb8be454a75ab942(__nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, __nv_bfloat16* __restrict__ O, int* __restrict__ cum_seq_lens_q, int batch, int num_heads, int num_split, float bmm2_scale, int m_tiles, int n_full_items, int tile_rows)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int part = warp % 4;
    int sr_idx = blockIdx.x * 2 + warp / 4;
    int row = sr_idx;
    int row_ok = 0;
    if (n_full_items == 0) {
        int rows_total = cum_seq_lens_q[batch] * num_heads;
        row_ok = ((sr_idx < rows_total) ? 1 : 0);
    } else {
        int sr_t = sr_idx / tile_rows;
        int sr_r = sr_idx - sr_t * tile_rows;
        int sr_item = n_full_items + sr_t;
        int sr_b = sr_item / m_tiles;
        int sr_m_tile = m_tiles - 1 - (sr_item - sr_b * m_tiles);
        int sr_b_ld = ((sr_b < batch) ? sr_b : 0);
        int sr_q_start = cum_seq_lens_q[sr_b_ld];
        int sr_rows_b = (cum_seq_lens_q[sr_b_ld + 1] - sr_q_start) * num_heads;
        int sr_local = sr_m_tile * tile_rows + sr_r;
        row = sr_q_start * num_heads + sr_local;
        int sr_in_rows = ((sr_local < sr_rows_b) ? 1 : 0);
        row_ok = ((sr_b < batch) ? sr_in_rows : 0);
    }
    if (row_ok != 0) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        int stat_base = row * num_split;
        int last_split = num_split - 1;
        int d0 = part * 128 + lane * 4;
        float pf[32];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int src_j = (stat_base + j) * 512 + d0;
            if (last_split >= j) {
                {
                    uint2 _vld_0;
                    _vld_0 = *reinterpret_cast<const uint2*>(partial_O + src_j);
                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&pf[j * 4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                        (&pf[j * 4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                    }
                }
            }
        }
        int s_ld = ((last_split < lane) ? last_split : lane);
        float m_raw = partial_max[stat_base + s_ld];
        float sum_raw = partial_sum[stat_base + s_ld];
        float m_s = ((last_split < lane) ? -CUDART_INF_F : m_raw);
        float max_m = m_s;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            max_m = max_noftz(max_m, __shfl_xor_sync(0xFFFFFFFF, max_m, offset));
        float w_s = 0.0f;
        if (m_s > -CUDART_INF_F) {
            float _exp2_1 = approx_exp2(m_s - max_m);
            w_s = _exp2_1 * sum_raw;
        }
        float sum_w = w_s;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_w += __shfl_xor_sync(0xFFFFFFFF, sum_w, offset);
        float inv_sum = 0.0f;
        if (sum_w > 0.0f) {
            float _rcp_2 = approx_rcp(sum_w);
            inv_sum = _rcp_2 * bmm2_scale;
        }
        float acc[4];
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            acc[e] = 0.0f;
        }
        #pragma unroll
        for (int j_1 = 0; j_1 < 8; j_1++) {
            float w_j = __shfl_sync(0xFFFFFFFF, w_s, j_1);
            #pragma unroll
            for (int e_1 = 0; e_1 < 4; e_1++) {
                float c_j = w_j * pf[j_1 * 4 + e_1];
                float safe_j = ((w_j > 0.0f) ? c_j : 0.0f);
                acc[e_1] = acc[e_1] + safe_j;
            }
        }
        #pragma unroll 4
        for (int k = 8; k < num_split; k++) {
            float w_k = __shfl_sync(0xFFFFFFFF, w_s, k);
            int src = (stat_base + k) * 512 + d0;
            float _vec_load_0[4];
            {
                uint2 _vld_3;
                _vld_3 = *reinterpret_cast<const uint2*>(partial_O + src);
                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_vec_load_0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                    (&_vec_load_0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                }
            }
            #pragma unroll
            for (int e_2 = 0; e_2 < 4; e_2++) {
                float c_e = w_k * _vec_load_0[e_2];
                float safe_e = ((w_k > 0.0f) ? c_e : 0.0f);
                acc[e_2] = acc[e_2] + safe_e;
            }
        }
        #pragma unroll
        for (int e_3 = 0; e_3 < 4; e_3++) {
            acc[e_3] = acc[e_3] * inv_sum;
        }
        unsigned int _packed_0[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
            _packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (row * 512 + d0))[0] = reinterpret_cast<unsigned*>(_packed_0)[0];
        unsigned int _packed_1[1];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 2], acc[_lp*2+1 + 2]));
            _packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<unsigned*>(O + (row * 512 + d0 + 2))[0] = reinterpret_cast<unsigned*>(_packed_1)[0];
    }
}

} // extern "C"
