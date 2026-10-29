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
#define THREADS 128
#define USE_PDL 0

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

__global__ __launch_bounds__(128) void
kernel_cake_dsv4_e880e7c12a67fe035a37(__nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, int num_q_heads, int num_split)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    const int wg_dummy = 0;
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y * 4 + warp;
    int stat_base = (batch_idx * num_q_heads + head_idx) * 5;
    int po_head_base = stat_base * 512;
    int d_base = lane * 16;
    float vals[80];
    #pragma unroll
    for (int s = 0; s < 5; s++) {
        float _vec_load_0[16];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(partial_O + (po_head_base + s * 512 + d_base) + 0);
            uint4 _vld_0[2];
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
        }
        #pragma unroll
        for (int e = 0; e < 16; e++) {
            vals[s * 16 + e] = _vec_load_0[e];
        }
    }
    float lane_lse[5];
    #pragma unroll
    for (int s_1 = 0; s_1 < 5; s_1++) {
        lane_lse[s_1] = partial_lse[stat_base + s_1];
    }
    float global_max = -CUDART_INF_F;
    #pragma unroll
    for (int s_2 = 0; s_2 < 5; s_2++) {
        global_max = max_noftz(global_max, lane_lse[s_2]);
    }
    float lane_w[5];
    float lane_sum[5];
    #pragma unroll
    for (int s_3 = 0; s_3 < 5; s_3++) {
        float _exp2_1 = approx_exp2(lane_lse[s_3] - global_max);
        lane_w[s_3] = ((lane_lse[s_3] == -CUDART_INF_F) ? 0.0f : _exp2_1);
        lane_sum[s_3] = lane_w[s_3];
    }
    lane_sum[0] = lane_sum[0] + lane_sum[4];
    lane_sum[0] = lane_sum[0] + lane_sum[2];
    lane_sum[1] = lane_sum[1] + lane_sum[3];
    lane_sum[0] = lane_sum[0] + lane_sum[1];
    float global_sum = lane_sum[0];
    float _rcp_2 = approx_rcp(global_sum);
    float inv_sum = ((global_sum > 0.0f) ? _rcp_2 : 0.0f);
    #pragma unroll
    for (int s_4 = 0; s_4 < 5; s_4++) {
        lane_w[s_4] = lane_w[s_4] * inv_sum;
    }
    int o_head_base = (batch_idx * num_q_heads + head_idx) * 512;
    float acc[16];
    #pragma unroll
    for (int e_1 = 0; e_1 < 16; e_1++) {
        acc[e_1] = 0.0f;
    }
    #pragma unroll
    for (int s_5 = 0; s_5 < 5; s_5++) {
        float split_weight = lane_w[s_5];
        #pragma unroll
        for (int e_2 = 0; e_2 < 16; e_2++) {
            acc[e_2] = __fmaf_rn(split_weight, vals[s_5 * 16 + e_2], acc[e_2]);
        }
    }
    unsigned int _packed_0[8];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
        _packed_0[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 2], acc[_lp*2+1 + 2]));
        _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 4], acc[_lp*2+1 + 4]));
        _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 6], acc[_lp*2+1 + 6]));
        _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 8], acc[_lp*2+1 + 8]));
        _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 10], acc[_lp*2+1 + 10]));
        _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 12], acc[_lp*2+1 + 12]));
        _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 14], acc[_lp*2+1 + 14]));
        _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int4*>(O + (o_head_base + d_base))[0] = reinterpret_cast<int4*>(_packed_0)[0];
    reinterpret_cast<int4*>(O + (o_head_base + d_base + 8))[0] = reinterpret_cast<int4*>(_packed_0 + (4))[0];
}

} // extern "C"
