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
kernel_cake_dsv4_50a2c6ca93259d30db6b(__nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, int num_heads, int num_splits)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int stat_base = (query_idx * num_heads + head_idx) * num_splits;
    int partial_base = stat_base * 512;
    int output_base = (query_idx * num_heads + head_idx) * 512;
    int d_base = tid * 4;
    float acc[4];
    #pragma unroll
    for (int elem = 0; elem < 4; elem++) {
        acc[elem] = 0.0f;
    }
    if (num_splits == 4) {
        float _vec_load_0[4];
        {
            uint2 _vld_0;
            _vld_0 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + d_base) + 0);
            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                (&_vec_load_0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
            }
        }
        float _vec_load_1[4];
        {
            uint2 _vld_1;
            _vld_1 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 512 + d_base) + 0);
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                (&_vec_load_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
            }
        }
        float _vec_load_2[4];
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 1024 + d_base) + 0);
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                (&_vec_load_2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
            }
        }
        float _vec_load_3[4];
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 1536 + d_base) + 0);
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_3[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                (&_vec_load_3[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
            }
        }
        float lse0 = partial_lse[stat_base];
        float lse1 = partial_lse[stat_base + 1];
        float lse2 = partial_lse[stat_base + 2];
        float lse3 = partial_lse[stat_base + 3];
        float global_max;
        asm("max.f32 %0, %1, %2;" : "=f"(global_max) : "f"(max_noftz(max_noftz(lse0, lse1), lse2)), "f"(lse3));
        float _exp2_4 = approx_exp2(lse0 - global_max);
        float w0 = ((lse0 == -CUDART_INF_F) ? 0.0f : _exp2_4);
        float _exp2_5 = approx_exp2(lse1 - global_max);
        float w1 = ((lse1 == -CUDART_INF_F) ? 0.0f : _exp2_5);
        float _exp2_6 = approx_exp2(lse2 - global_max);
        float w2 = ((lse2 == -CUDART_INF_F) ? 0.0f : _exp2_6);
        float _exp2_7 = approx_exp2(lse3 - global_max);
        float w3 = ((lse3 == -CUDART_INF_F) ? 0.0f : _exp2_7);
        float global_sum = w0 + w2 + (w1 + w3);
        float inv_sum = approx_rcp(global_sum);
        float weight0 = ((global_sum > 0.0f) ? w0 * inv_sum : 0.0f);
        float weight1 = ((global_sum > 0.0f) ? w1 * inv_sum : 0.0f);
        float weight2 = ((global_sum > 0.0f) ? w2 * inv_sum : 0.0f);
        float weight3 = ((global_sum > 0.0f) ? w3 * inv_sum : 0.0f);
        #pragma unroll
        for (int elem_1 = 0; elem_1 < 4; elem_1++) {
            acc[elem_1] = __fmaf_rn(weight0, _vec_load_0[elem_1], acc[elem_1]);
            acc[elem_1] = __fmaf_rn(weight1, _vec_load_1[elem_1], acc[elem_1]);
            acc[elem_1] = __fmaf_rn(weight2, _vec_load_2[elem_1], acc[elem_1]);
            acc[elem_1] = __fmaf_rn(weight3, _vec_load_3[elem_1], acc[elem_1]);
        }
    } else if (num_splits == 3) {
        float _vec_load_4[4];
        {
            uint2 _vld_8;
            _vld_8 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + d_base) + 0);
            uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_4[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                (&_vec_load_4[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
            }
        }
        float _vec_load_5[4];
        {
            uint2 _vld_9;
            _vld_9 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 512 + d_base) + 0);
            uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_5[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                (&_vec_load_5[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
            }
        }
        float _vec_load_6[4];
        {
            uint2 _vld_10;
            _vld_10 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 1024 + d_base) + 0);
            uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_6[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                (&_vec_load_6[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
            }
        }
        float lse0_1 = partial_lse[stat_base];
        float lse1_1 = partial_lse[stat_base + 1];
        float lse2_1 = partial_lse[stat_base + 2];
        float global_max_1;
        asm("max.f32 %0, %1, %2;" : "=f"(global_max_1) : "f"(max_noftz(lse0_1, lse1_1)), "f"(lse2_1));
        float _exp2_11 = approx_exp2(lse0_1 - global_max_1);
        float w0_1 = ((lse0_1 == -CUDART_INF_F) ? 0.0f : _exp2_11);
        float _exp2_12 = approx_exp2(lse1_1 - global_max_1);
        float w1_1 = ((lse1_1 == -CUDART_INF_F) ? 0.0f : _exp2_12);
        float _exp2_13 = approx_exp2(lse2_1 - global_max_1);
        float w2_1 = ((lse2_1 == -CUDART_INF_F) ? 0.0f : _exp2_13);
        float global_sum_1 = w0_1 + w2_1 + w1_1;
        float inv_sum_1 = approx_rcp(global_sum_1);
        float weight0_1 = ((global_sum_1 > 0.0f) ? w0_1 * inv_sum_1 : 0.0f);
        float weight1_1 = ((global_sum_1 > 0.0f) ? w1_1 * inv_sum_1 : 0.0f);
        float weight2_1 = ((global_sum_1 > 0.0f) ? w2_1 * inv_sum_1 : 0.0f);
        #pragma unroll
        for (int elem_2 = 0; elem_2 < 4; elem_2++) {
            acc[elem_2] = __fmaf_rn(weight0_1, _vec_load_4[elem_2], acc[elem_2]);
            acc[elem_2] = __fmaf_rn(weight1_1, _vec_load_5[elem_2], acc[elem_2]);
            acc[elem_2] = __fmaf_rn(weight2_1, _vec_load_6[elem_2], acc[elem_2]);
        }
    } else {
        if (num_splits == 5) {
            float _vec_load_7[4];
            {
                uint2 _vld_14;
                _vld_14 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + d_base) + 0);
                uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_vec_load_7[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) << 16);
                    (&_vec_load_7[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) & 0xffff0000u);
                }
            }
            float _vec_load_8[4];
            {
                uint2 _vld_15;
                _vld_15 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 512 + d_base) + 0);
                uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_vec_load_8[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) << 16);
                    (&_vec_load_8[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) & 0xffff0000u);
                }
            }
            float _vec_load_9[4];
            {
                uint2 _vld_16;
                _vld_16 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 1024 + d_base) + 0);
                uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_vec_load_9[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
                    (&_vec_load_9[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
                }
            }
            float _vec_load_10[4];
            {
                uint2 _vld_17;
                _vld_17 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 1536 + d_base) + 0);
                uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_vec_load_10[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) << 16);
                    (&_vec_load_10[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) & 0xffff0000u);
                }
            }
            float _vec_load_11[4];
            {
                uint2 _vld_18;
                _vld_18 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 2048 + d_base) + 0);
                uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_vec_load_11[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
                    (&_vec_load_11[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
                }
            }
            float lse0_2 = partial_lse[stat_base];
            float lse1_2 = partial_lse[stat_base + 1];
            float lse2_2 = partial_lse[stat_base + 2];
            float lse3_1 = partial_lse[stat_base + 3];
            float lse4 = partial_lse[stat_base + 4];
            float global_max_2;
            asm("max.f32 %0, %1, %2;" : "=f"(global_max_2) : "f"(max_noftz(max_noftz(max_noftz(lse0_2, lse1_2), lse2_2), lse3_1)), "f"(lse4));
            float _exp2_19 = approx_exp2(lse0_2 - global_max_2);
            float w0_2 = ((lse0_2 == -CUDART_INF_F) ? 0.0f : _exp2_19);
            float _exp2_20 = approx_exp2(lse1_2 - global_max_2);
            float w1_2 = ((lse1_2 == -CUDART_INF_F) ? 0.0f : _exp2_20);
            float _exp2_21 = approx_exp2(lse2_2 - global_max_2);
            float w2_2 = ((lse2_2 == -CUDART_INF_F) ? 0.0f : _exp2_21);
            float _exp2_22 = approx_exp2(lse3_1 - global_max_2);
            float w3_1 = ((lse3_1 == -CUDART_INF_F) ? 0.0f : _exp2_22);
            float _exp2_23 = approx_exp2(lse4 - global_max_2);
            float w4 = ((lse4 == -CUDART_INF_F) ? 0.0f : _exp2_23);
            float global_sum_2 = w0_2 + w4 + w2_2 + (w1_2 + w3_1);
            float inv_sum_2 = approx_rcp(global_sum_2);
            float weight0_2 = ((global_sum_2 > 0.0f) ? w0_2 * inv_sum_2 : 0.0f);
            float weight1_2 = ((global_sum_2 > 0.0f) ? w1_2 * inv_sum_2 : 0.0f);
            float weight2_2 = ((global_sum_2 > 0.0f) ? w2_2 * inv_sum_2 : 0.0f);
            float weight3_1 = ((global_sum_2 > 0.0f) ? w3_1 * inv_sum_2 : 0.0f);
            float weight4 = ((global_sum_2 > 0.0f) ? w4 * inv_sum_2 : 0.0f);
            #pragma unroll
            for (int elem_3 = 0; elem_3 < 4; elem_3++) {
                acc[elem_3] = __fmaf_rn(weight0_2, _vec_load_7[elem_3], acc[elem_3]);
                acc[elem_3] = __fmaf_rn(weight1_2, _vec_load_8[elem_3], acc[elem_3]);
                acc[elem_3] = __fmaf_rn(weight2_2, _vec_load_9[elem_3], acc[elem_3]);
                acc[elem_3] = __fmaf_rn(weight3_1, _vec_load_10[elem_3], acc[elem_3]);
                acc[elem_3] = __fmaf_rn(weight4, _vec_load_11[elem_3], acc[elem_3]);
            }
        } else {
            float local_lse = -CUDART_INF_F;
            if (lane < num_splits) {
                local_lse = partial_lse[stat_base + lane];
            }
            float global_max_3 = local_lse;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                global_max_3 = max_noftz(global_max_3, __shfl_xor_sync(0xFFFFFFFF, global_max_3, offset));
            float local_weight = 0.0f;
            if (lane < num_splits) {
                float _exp2_24 = approx_exp2(local_lse - global_max_3);
                local_weight = ((local_lse == -CUDART_INF_F) ? 0.0f : _exp2_24);
            }
            float global_sum_3 = local_weight;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                global_sum_3 += __shfl_xor_sync(0xFFFFFFFF, global_sum_3, offset);
            float _rcp_25 = approx_rcp(global_sum_3);
            float normalized_weight = ((global_sum_3 > 0.0f) ? local_weight * _rcp_25 : 0.0f);
            #pragma unroll 5
            for (int split = 0; split < num_splits; split++) {
                float _vec_load_12[4];
                {
                    uint2 _vld_26;
                    _vld_26 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + split * 512 + d_base) + 0);
                    uint32_t* _vpairs_26 = reinterpret_cast<uint32_t*>(&_vld_26);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&_vec_load_12[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_26[_pair]) << 16);
                        (&_vec_load_12[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_26[_pair]) & 0xffff0000u);
                    }
                }
                float weight = __shfl_sync(0xFFFFFFFF, normalized_weight, split);
                #pragma unroll
                for (int elem_4 = 0; elem_4 < 4; elem_4++) {
                    acc[elem_4] = __fmaf_rn(weight, _vec_load_12[elem_4], acc[elem_4]);
                }
            }
        }
    }
    unsigned int _packed_0[2];
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
    reinterpret_cast<int2*>(O + (output_base + d_base))[0] = reinterpret_cast<int2*>(_packed_0)[0];
}

} // extern "C"
