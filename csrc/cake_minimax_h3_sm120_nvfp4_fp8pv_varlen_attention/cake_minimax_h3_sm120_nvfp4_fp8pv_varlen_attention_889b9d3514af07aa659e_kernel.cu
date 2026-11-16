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
#define SMEM_RED_SUM_OFF 0
#define SMEM_RED_SUM_STAGE_BYTES 8192
#define SMEM_RED_SUM_STRIDE 8192
#define SMEM_RED_MAX_OFF 8192
#define SMEM_RED_MAX_STAGE_BYTES 8192
#define SMEM_RED_MAX_STRIDE 8192
#define SMEM_RED_NORM_OFF 16384
#define SMEM_RED_NORM_STAGE_BYTES 64
#define SMEM_RED_NORM_STRIDE 64
#define SMEM_RED_OFF 16448
#define SMEM_RED_STAGE_BYTES 32
#define SMEM_RED_STRIDE 32
#define SMEM_FLAG_OFF 16480
#define SMEM_FLAG_STAGE_BYTES 16
#define SMEM_FLAG_STRIDE 16
#define SMEM_TOTAL 16512
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_889b9d3514af07aa659e(__nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V, int* __restrict__ tile_table, float* __restrict__ partials, float* __restrict__ knorm_part, unsigned int* __restrict__ counters, float* __restrict__ mean_k, float* __restrict__ v_scale, float* __restrict__ k_scale, int num_tiles, int num_heads)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* red_sum = reinterpret_cast<float*>(smem_raw + 0);
    const int red_sum_addr = smem + 0;
    float* red_max = reinterpret_cast<float*>(smem_raw + 8192);
    const int red_max_addr = smem + 8192;
    float* red_norm = reinterpret_cast<float*>(smem_raw + 16384);
    const int red_norm_addr = smem + 16384;
    float* red = reinterpret_cast<float*>(smem_raw + 16448);
    const int red_addr = smem + 16448;
    int* flag = reinterpret_cast<int*>(smem_raw + 16480);
    const int flag_addr = smem + 16480;

    // === Task calls (dependency order) ===
    int col = (tid & 15) * 8;
    int rgrp = tid >> 4;
    int chan = tid & 127;
    int is_max = tid >> 7;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll 1
    for (int work = bid; work < num_tiles * num_heads; work += num_bids) {
        int tile = work / num_heads;
        int head = work % num_heads;
        int _vec_load_0[4];
        {
            const int4* _ivptr_0 = reinterpret_cast<const int4*>(tile_table + (8 * tile) + 0);
            int4 _ivld_0;
            _ivld_0 = *_ivptr_0;
            _vec_load_0[0 + 0] = _ivld_0.x;
            _vec_load_0[0 + 1] = _ivld_0.y;
            _vec_load_0[0 + 2] = _ivld_0.z;
            _vec_load_0[0 + 3] = _ivld_0.w;
        }
        int _vec_load_1[4];
        {
            const int4* _ivptr_1 = reinterpret_cast<const int4*>(tile_table + (8 * tile + 4) + 0);
            int4 _ivld_1;
            _ivld_1 = *_ivptr_1;
            _vec_load_1[0 + 0] = _ivld_1.x;
            _vec_load_1[0 + 1] = _ivld_1.y;
            _vec_load_1[0 + 2] = _ivld_1.z;
            _vec_load_1[0 + 3] = _ivld_1.w;
        }
        int seg = _vec_load_0[0];
        int row_begin = _vec_load_0[1];
        int row_end = _vec_load_0[2];
        int t0 = _vec_load_0[3];
        int t1 = _vec_load_1[0];
        int length = _vec_load_1[1];
        float ksum[8];
        float vmax[8];
        for (int e = 0; e < 8; e++) {
            ksum[e] = 0.0f;
            vmax[e] = 0.0f;
        }
        float kf8[8];
        float vf8[8];
        float ksq_a[1];
        float knmax_a[1];
        knmax_a[0] = 0.0f;
        #pragma unroll 1
        for (int batch = 0; batch < (row_end - row_begin + 128 - 1) / 128; batch++) {
            float _vec_load_2[8];
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(K + ((((row_end > row_begin + batch * 128 + rgrp) ? row_begin + batch * 128 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_2[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_2[_blk] = _vptr_2[_blk];
                    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                        (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_3[8];
            {
                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(V + ((((row_end > row_begin + batch * 128 + rgrp) ? row_begin + batch * 128 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_3[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_3[_blk] = _vptr_3[_blk];
                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                        (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_4[8];
            {
                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(K + ((((row_end > row_begin + batch * 128 + 16 + rgrp) ? row_begin + batch * 128 + 16 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_4[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_4[_blk] = _vptr_4[_blk];
                    uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_5[8];
            {
                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(V + ((((row_end > row_begin + batch * 128 + 16 + rgrp) ? row_begin + batch * 128 + 16 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_5[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_5[_blk] = _vptr_5[_blk];
                    uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                        (&_vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_6[8];
            {
                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(K + ((((row_end > row_begin + batch * 128 + 32 + rgrp) ? row_begin + batch * 128 + 32 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_6[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_6[_blk] = _vptr_6[_blk];
                    uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                        (&_vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_7[8];
            {
                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(V + ((((row_end > row_begin + batch * 128 + 32 + rgrp) ? row_begin + batch * 128 + 32 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_7[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_7[_blk] = _vptr_7[_blk];
                    uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                        (&_vec_load_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_8[8];
            {
                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(K + ((((row_end > row_begin + batch * 128 + 48 + rgrp) ? row_begin + batch * 128 + 48 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_8[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_8[_blk] = _vptr_8[_blk];
                    uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                        (&_vec_load_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_9[8];
            {
                const uint4* _vptr_9 = reinterpret_cast<const uint4*>(V + ((((row_end > row_begin + batch * 128 + 48 + rgrp) ? row_begin + batch * 128 + 48 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_9[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_9[_blk] = _vptr_9[_blk];
                    uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                        (&_vec_load_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_10[8];
            {
                const uint4* _vptr_10 = reinterpret_cast<const uint4*>(K + ((((row_end > row_begin + batch * 128 + 64 + rgrp) ? row_begin + batch * 128 + 64 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_10[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_10[_blk] = _vptr_10[_blk];
                    uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                        (&_vec_load_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_11[8];
            {
                const uint4* _vptr_11 = reinterpret_cast<const uint4*>(V + ((((row_end > row_begin + batch * 128 + 64 + rgrp) ? row_begin + batch * 128 + 64 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_11[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_11[_blk] = _vptr_11[_blk];
                    uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) << 16);
                        (&_vec_load_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_12[8];
            {
                const uint4* _vptr_12 = reinterpret_cast<const uint4*>(K + ((((row_end > row_begin + batch * 128 + 80 + rgrp) ? row_begin + batch * 128 + 80 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_12[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_12[_blk] = _vptr_12[_blk];
                    uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_12[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
                        (&_vec_load_12[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_13[8];
            {
                const uint4* _vptr_13 = reinterpret_cast<const uint4*>(V + ((((row_end > row_begin + batch * 128 + 80 + rgrp) ? row_begin + batch * 128 + 80 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_13[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_13[_blk] = _vptr_13[_blk];
                    uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_13[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) << 16);
                        (&_vec_load_13[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_14[8];
            {
                const uint4* _vptr_14 = reinterpret_cast<const uint4*>(K + ((((row_end > row_begin + batch * 128 + 96 + rgrp) ? row_begin + batch * 128 + 96 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_14[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_14[_blk] = _vptr_14[_blk];
                    uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_14[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) << 16);
                        (&_vec_load_14[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_15[8];
            {
                const uint4* _vptr_15 = reinterpret_cast<const uint4*>(V + ((((row_end > row_begin + batch * 128 + 96 + rgrp) ? row_begin + batch * 128 + 96 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_15[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_15[_blk] = _vptr_15[_blk];
                    uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_15[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) << 16);
                        (&_vec_load_15[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_16[8];
            {
                const uint4* _vptr_16 = reinterpret_cast<const uint4*>(K + ((((row_end > row_begin + batch * 128 + 112 + rgrp) ? row_begin + batch * 128 + 112 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_16[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_16[_blk] = _vptr_16[_blk];
                    uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_16[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
                        (&_vec_load_16[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_17[8];
            {
                const uint4* _vptr_17 = reinterpret_cast<const uint4*>(V + ((((row_end > row_begin + batch * 128 + 112 + rgrp) ? row_begin + batch * 128 + 112 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_17[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_17[_blk] = _vptr_17[_blk];
                    uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_17[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) << 16);
                        (&_vec_load_17[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) & 0xffff0000u);
                    }
                }
            }
            kf8[0] = _vec_load_2[0];
            vf8[0] = _vec_load_3[0];
            kf8[1] = _vec_load_2[1];
            vf8[1] = _vec_load_3[1];
            kf8[2] = _vec_load_2[2];
            vf8[2] = _vec_load_3[2];
            kf8[3] = _vec_load_2[3];
            vf8[3] = _vec_load_3[3];
            kf8[4] = _vec_load_2[4];
            vf8[4] = _vec_load_3[4];
            kf8[5] = _vec_load_2[5];
            vf8[5] = _vec_load_3[5];
            kf8[6] = _vec_load_2[6];
            vf8[6] = _vec_load_3[6];
            kf8[7] = _vec_load_2[7];
            vf8[7] = _vec_load_3[7];
            ksq_a[0] = 0.0f;
            ksum[0] = ksum[0] + ((row_end > row_begin + batch * 128 + rgrp) ? kf8[0] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + rgrp) ? kf8[0] : 0.0f), ((row_end > row_begin + batch * 128 + rgrp) ? kf8[0] : 0.0f), ksq_a[0]);
            vmax[0] = fmaxf(vmax[0], fabsf(((row_end > row_begin + batch * 128 + rgrp) ? vf8[0] : 0.0f)));
            ksum[1] = ksum[1] + ((row_end > row_begin + batch * 128 + rgrp) ? kf8[1] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + rgrp) ? kf8[1] : 0.0f), ((row_end > row_begin + batch * 128 + rgrp) ? kf8[1] : 0.0f), ksq_a[0]);
            vmax[1] = fmaxf(vmax[1], fabsf(((row_end > row_begin + batch * 128 + rgrp) ? vf8[1] : 0.0f)));
            ksum[2] = ksum[2] + ((row_end > row_begin + batch * 128 + rgrp) ? kf8[2] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + rgrp) ? kf8[2] : 0.0f), ((row_end > row_begin + batch * 128 + rgrp) ? kf8[2] : 0.0f), ksq_a[0]);
            vmax[2] = fmaxf(vmax[2], fabsf(((row_end > row_begin + batch * 128 + rgrp) ? vf8[2] : 0.0f)));
            ksum[3] = ksum[3] + ((row_end > row_begin + batch * 128 + rgrp) ? kf8[3] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + rgrp) ? kf8[3] : 0.0f), ((row_end > row_begin + batch * 128 + rgrp) ? kf8[3] : 0.0f), ksq_a[0]);
            vmax[3] = fmaxf(vmax[3], fabsf(((row_end > row_begin + batch * 128 + rgrp) ? vf8[3] : 0.0f)));
            ksum[4] = ksum[4] + ((row_end > row_begin + batch * 128 + rgrp) ? kf8[4] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + rgrp) ? kf8[4] : 0.0f), ((row_end > row_begin + batch * 128 + rgrp) ? kf8[4] : 0.0f), ksq_a[0]);
            vmax[4] = fmaxf(vmax[4], fabsf(((row_end > row_begin + batch * 128 + rgrp) ? vf8[4] : 0.0f)));
            ksum[5] = ksum[5] + ((row_end > row_begin + batch * 128 + rgrp) ? kf8[5] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + rgrp) ? kf8[5] : 0.0f), ((row_end > row_begin + batch * 128 + rgrp) ? kf8[5] : 0.0f), ksq_a[0]);
            vmax[5] = fmaxf(vmax[5], fabsf(((row_end > row_begin + batch * 128 + rgrp) ? vf8[5] : 0.0f)));
            ksum[6] = ksum[6] + ((row_end > row_begin + batch * 128 + rgrp) ? kf8[6] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + rgrp) ? kf8[6] : 0.0f), ((row_end > row_begin + batch * 128 + rgrp) ? kf8[6] : 0.0f), ksq_a[0]);
            vmax[6] = fmaxf(vmax[6], fabsf(((row_end > row_begin + batch * 128 + rgrp) ? vf8[6] : 0.0f)));
            ksum[7] = ksum[7] + ((row_end > row_begin + batch * 128 + rgrp) ? kf8[7] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + rgrp) ? kf8[7] : 0.0f), ((row_end > row_begin + batch * 128 + rgrp) ? kf8[7] : 0.0f), ksq_a[0]);
            vmax[7] = fmaxf(vmax[7], fabsf(((row_end > row_begin + batch * 128 + rgrp) ? vf8[7] : 0.0f)));
            float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 8);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_18;
            float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 4);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_19;
            float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 2);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_20;
            float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 1);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_21;
            knmax_a[0] = fmaxf(knmax_a[0], ksq_a[0]);
            kf8[0] = _vec_load_4[0];
            vf8[0] = _vec_load_5[0];
            kf8[1] = _vec_load_4[1];
            vf8[1] = _vec_load_5[1];
            kf8[2] = _vec_load_4[2];
            vf8[2] = _vec_load_5[2];
            kf8[3] = _vec_load_4[3];
            vf8[3] = _vec_load_5[3];
            kf8[4] = _vec_load_4[4];
            vf8[4] = _vec_load_5[4];
            kf8[5] = _vec_load_4[5];
            vf8[5] = _vec_load_5[5];
            kf8[6] = _vec_load_4[6];
            vf8[6] = _vec_load_5[6];
            kf8[7] = _vec_load_4[7];
            vf8[7] = _vec_load_5[7];
            ksq_a[0] = 0.0f;
            ksum[0] = ksum[0] + ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[0] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[0] : 0.0f), ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[0] : 0.0f), ksq_a[0]);
            vmax[0] = fmaxf(vmax[0], fabsf(((row_end > row_begin + batch * 128 + 16 + rgrp) ? vf8[0] : 0.0f)));
            ksum[1] = ksum[1] + ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[1] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[1] : 0.0f), ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[1] : 0.0f), ksq_a[0]);
            vmax[1] = fmaxf(vmax[1], fabsf(((row_end > row_begin + batch * 128 + 16 + rgrp) ? vf8[1] : 0.0f)));
            ksum[2] = ksum[2] + ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[2] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[2] : 0.0f), ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[2] : 0.0f), ksq_a[0]);
            vmax[2] = fmaxf(vmax[2], fabsf(((row_end > row_begin + batch * 128 + 16 + rgrp) ? vf8[2] : 0.0f)));
            ksum[3] = ksum[3] + ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[3] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[3] : 0.0f), ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[3] : 0.0f), ksq_a[0]);
            vmax[3] = fmaxf(vmax[3], fabsf(((row_end > row_begin + batch * 128 + 16 + rgrp) ? vf8[3] : 0.0f)));
            ksum[4] = ksum[4] + ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[4] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[4] : 0.0f), ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[4] : 0.0f), ksq_a[0]);
            vmax[4] = fmaxf(vmax[4], fabsf(((row_end > row_begin + batch * 128 + 16 + rgrp) ? vf8[4] : 0.0f)));
            ksum[5] = ksum[5] + ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[5] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[5] : 0.0f), ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[5] : 0.0f), ksq_a[0]);
            vmax[5] = fmaxf(vmax[5], fabsf(((row_end > row_begin + batch * 128 + 16 + rgrp) ? vf8[5] : 0.0f)));
            ksum[6] = ksum[6] + ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[6] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[6] : 0.0f), ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[6] : 0.0f), ksq_a[0]);
            vmax[6] = fmaxf(vmax[6], fabsf(((row_end > row_begin + batch * 128 + 16 + rgrp) ? vf8[6] : 0.0f)));
            ksum[7] = ksum[7] + ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[7] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[7] : 0.0f), ((row_end > row_begin + batch * 128 + 16 + rgrp) ? kf8[7] : 0.0f), ksq_a[0]);
            vmax[7] = fmaxf(vmax[7], fabsf(((row_end > row_begin + batch * 128 + 16 + rgrp) ? vf8[7] : 0.0f)));
            float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 8);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_22;
            float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 4);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_23;
            float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 2);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_24;
            float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 1);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_25;
            knmax_a[0] = fmaxf(knmax_a[0], ksq_a[0]);
            kf8[0] = _vec_load_6[0];
            vf8[0] = _vec_load_7[0];
            kf8[1] = _vec_load_6[1];
            vf8[1] = _vec_load_7[1];
            kf8[2] = _vec_load_6[2];
            vf8[2] = _vec_load_7[2];
            kf8[3] = _vec_load_6[3];
            vf8[3] = _vec_load_7[3];
            kf8[4] = _vec_load_6[4];
            vf8[4] = _vec_load_7[4];
            kf8[5] = _vec_load_6[5];
            vf8[5] = _vec_load_7[5];
            kf8[6] = _vec_load_6[6];
            vf8[6] = _vec_load_7[6];
            kf8[7] = _vec_load_6[7];
            vf8[7] = _vec_load_7[7];
            ksq_a[0] = 0.0f;
            ksum[0] = ksum[0] + ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[0] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[0] : 0.0f), ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[0] : 0.0f), ksq_a[0]);
            vmax[0] = fmaxf(vmax[0], fabsf(((row_end > row_begin + batch * 128 + 32 + rgrp) ? vf8[0] : 0.0f)));
            ksum[1] = ksum[1] + ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[1] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[1] : 0.0f), ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[1] : 0.0f), ksq_a[0]);
            vmax[1] = fmaxf(vmax[1], fabsf(((row_end > row_begin + batch * 128 + 32 + rgrp) ? vf8[1] : 0.0f)));
            ksum[2] = ksum[2] + ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[2] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[2] : 0.0f), ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[2] : 0.0f), ksq_a[0]);
            vmax[2] = fmaxf(vmax[2], fabsf(((row_end > row_begin + batch * 128 + 32 + rgrp) ? vf8[2] : 0.0f)));
            ksum[3] = ksum[3] + ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[3] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[3] : 0.0f), ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[3] : 0.0f), ksq_a[0]);
            vmax[3] = fmaxf(vmax[3], fabsf(((row_end > row_begin + batch * 128 + 32 + rgrp) ? vf8[3] : 0.0f)));
            ksum[4] = ksum[4] + ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[4] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[4] : 0.0f), ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[4] : 0.0f), ksq_a[0]);
            vmax[4] = fmaxf(vmax[4], fabsf(((row_end > row_begin + batch * 128 + 32 + rgrp) ? vf8[4] : 0.0f)));
            ksum[5] = ksum[5] + ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[5] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[5] : 0.0f), ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[5] : 0.0f), ksq_a[0]);
            vmax[5] = fmaxf(vmax[5], fabsf(((row_end > row_begin + batch * 128 + 32 + rgrp) ? vf8[5] : 0.0f)));
            ksum[6] = ksum[6] + ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[6] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[6] : 0.0f), ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[6] : 0.0f), ksq_a[0]);
            vmax[6] = fmaxf(vmax[6], fabsf(((row_end > row_begin + batch * 128 + 32 + rgrp) ? vf8[6] : 0.0f)));
            ksum[7] = ksum[7] + ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[7] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[7] : 0.0f), ((row_end > row_begin + batch * 128 + 32 + rgrp) ? kf8[7] : 0.0f), ksq_a[0]);
            vmax[7] = fmaxf(vmax[7], fabsf(((row_end > row_begin + batch * 128 + 32 + rgrp) ? vf8[7] : 0.0f)));
            float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 8);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_26;
            float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 4);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_27;
            float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 2);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_28;
            float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 1);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_29;
            knmax_a[0] = fmaxf(knmax_a[0], ksq_a[0]);
            kf8[0] = _vec_load_8[0];
            vf8[0] = _vec_load_9[0];
            kf8[1] = _vec_load_8[1];
            vf8[1] = _vec_load_9[1];
            kf8[2] = _vec_load_8[2];
            vf8[2] = _vec_load_9[2];
            kf8[3] = _vec_load_8[3];
            vf8[3] = _vec_load_9[3];
            kf8[4] = _vec_load_8[4];
            vf8[4] = _vec_load_9[4];
            kf8[5] = _vec_load_8[5];
            vf8[5] = _vec_load_9[5];
            kf8[6] = _vec_load_8[6];
            vf8[6] = _vec_load_9[6];
            kf8[7] = _vec_load_8[7];
            vf8[7] = _vec_load_9[7];
            ksq_a[0] = 0.0f;
            ksum[0] = ksum[0] + ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[0] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[0] : 0.0f), ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[0] : 0.0f), ksq_a[0]);
            vmax[0] = fmaxf(vmax[0], fabsf(((row_end > row_begin + batch * 128 + 48 + rgrp) ? vf8[0] : 0.0f)));
            ksum[1] = ksum[1] + ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[1] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[1] : 0.0f), ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[1] : 0.0f), ksq_a[0]);
            vmax[1] = fmaxf(vmax[1], fabsf(((row_end > row_begin + batch * 128 + 48 + rgrp) ? vf8[1] : 0.0f)));
            ksum[2] = ksum[2] + ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[2] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[2] : 0.0f), ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[2] : 0.0f), ksq_a[0]);
            vmax[2] = fmaxf(vmax[2], fabsf(((row_end > row_begin + batch * 128 + 48 + rgrp) ? vf8[2] : 0.0f)));
            ksum[3] = ksum[3] + ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[3] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[3] : 0.0f), ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[3] : 0.0f), ksq_a[0]);
            vmax[3] = fmaxf(vmax[3], fabsf(((row_end > row_begin + batch * 128 + 48 + rgrp) ? vf8[3] : 0.0f)));
            ksum[4] = ksum[4] + ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[4] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[4] : 0.0f), ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[4] : 0.0f), ksq_a[0]);
            vmax[4] = fmaxf(vmax[4], fabsf(((row_end > row_begin + batch * 128 + 48 + rgrp) ? vf8[4] : 0.0f)));
            ksum[5] = ksum[5] + ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[5] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[5] : 0.0f), ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[5] : 0.0f), ksq_a[0]);
            vmax[5] = fmaxf(vmax[5], fabsf(((row_end > row_begin + batch * 128 + 48 + rgrp) ? vf8[5] : 0.0f)));
            ksum[6] = ksum[6] + ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[6] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[6] : 0.0f), ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[6] : 0.0f), ksq_a[0]);
            vmax[6] = fmaxf(vmax[6], fabsf(((row_end > row_begin + batch * 128 + 48 + rgrp) ? vf8[6] : 0.0f)));
            ksum[7] = ksum[7] + ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[7] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[7] : 0.0f), ((row_end > row_begin + batch * 128 + 48 + rgrp) ? kf8[7] : 0.0f), ksq_a[0]);
            vmax[7] = fmaxf(vmax[7], fabsf(((row_end > row_begin + batch * 128 + 48 + rgrp) ? vf8[7] : 0.0f)));
            float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 8);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_30;
            float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 4);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_31;
            float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 2);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_32;
            float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 1);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_33;
            knmax_a[0] = fmaxf(knmax_a[0], ksq_a[0]);
            kf8[0] = _vec_load_10[0];
            vf8[0] = _vec_load_11[0];
            kf8[1] = _vec_load_10[1];
            vf8[1] = _vec_load_11[1];
            kf8[2] = _vec_load_10[2];
            vf8[2] = _vec_load_11[2];
            kf8[3] = _vec_load_10[3];
            vf8[3] = _vec_load_11[3];
            kf8[4] = _vec_load_10[4];
            vf8[4] = _vec_load_11[4];
            kf8[5] = _vec_load_10[5];
            vf8[5] = _vec_load_11[5];
            kf8[6] = _vec_load_10[6];
            vf8[6] = _vec_load_11[6];
            kf8[7] = _vec_load_10[7];
            vf8[7] = _vec_load_11[7];
            ksq_a[0] = 0.0f;
            ksum[0] = ksum[0] + ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[0] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[0] : 0.0f), ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[0] : 0.0f), ksq_a[0]);
            vmax[0] = fmaxf(vmax[0], fabsf(((row_end > row_begin + batch * 128 + 64 + rgrp) ? vf8[0] : 0.0f)));
            ksum[1] = ksum[1] + ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[1] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[1] : 0.0f), ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[1] : 0.0f), ksq_a[0]);
            vmax[1] = fmaxf(vmax[1], fabsf(((row_end > row_begin + batch * 128 + 64 + rgrp) ? vf8[1] : 0.0f)));
            ksum[2] = ksum[2] + ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[2] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[2] : 0.0f), ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[2] : 0.0f), ksq_a[0]);
            vmax[2] = fmaxf(vmax[2], fabsf(((row_end > row_begin + batch * 128 + 64 + rgrp) ? vf8[2] : 0.0f)));
            ksum[3] = ksum[3] + ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[3] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[3] : 0.0f), ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[3] : 0.0f), ksq_a[0]);
            vmax[3] = fmaxf(vmax[3], fabsf(((row_end > row_begin + batch * 128 + 64 + rgrp) ? vf8[3] : 0.0f)));
            ksum[4] = ksum[4] + ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[4] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[4] : 0.0f), ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[4] : 0.0f), ksq_a[0]);
            vmax[4] = fmaxf(vmax[4], fabsf(((row_end > row_begin + batch * 128 + 64 + rgrp) ? vf8[4] : 0.0f)));
            ksum[5] = ksum[5] + ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[5] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[5] : 0.0f), ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[5] : 0.0f), ksq_a[0]);
            vmax[5] = fmaxf(vmax[5], fabsf(((row_end > row_begin + batch * 128 + 64 + rgrp) ? vf8[5] : 0.0f)));
            ksum[6] = ksum[6] + ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[6] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[6] : 0.0f), ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[6] : 0.0f), ksq_a[0]);
            vmax[6] = fmaxf(vmax[6], fabsf(((row_end > row_begin + batch * 128 + 64 + rgrp) ? vf8[6] : 0.0f)));
            ksum[7] = ksum[7] + ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[7] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[7] : 0.0f), ((row_end > row_begin + batch * 128 + 64 + rgrp) ? kf8[7] : 0.0f), ksq_a[0]);
            vmax[7] = fmaxf(vmax[7], fabsf(((row_end > row_begin + batch * 128 + 64 + rgrp) ? vf8[7] : 0.0f)));
            float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 8);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_34;
            float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 4);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_35;
            float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 2);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_36;
            float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 1);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_37;
            knmax_a[0] = fmaxf(knmax_a[0], ksq_a[0]);
            kf8[0] = _vec_load_12[0];
            vf8[0] = _vec_load_13[0];
            kf8[1] = _vec_load_12[1];
            vf8[1] = _vec_load_13[1];
            kf8[2] = _vec_load_12[2];
            vf8[2] = _vec_load_13[2];
            kf8[3] = _vec_load_12[3];
            vf8[3] = _vec_load_13[3];
            kf8[4] = _vec_load_12[4];
            vf8[4] = _vec_load_13[4];
            kf8[5] = _vec_load_12[5];
            vf8[5] = _vec_load_13[5];
            kf8[6] = _vec_load_12[6];
            vf8[6] = _vec_load_13[6];
            kf8[7] = _vec_load_12[7];
            vf8[7] = _vec_load_13[7];
            ksq_a[0] = 0.0f;
            ksum[0] = ksum[0] + ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[0] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[0] : 0.0f), ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[0] : 0.0f), ksq_a[0]);
            vmax[0] = fmaxf(vmax[0], fabsf(((row_end > row_begin + batch * 128 + 80 + rgrp) ? vf8[0] : 0.0f)));
            ksum[1] = ksum[1] + ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[1] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[1] : 0.0f), ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[1] : 0.0f), ksq_a[0]);
            vmax[1] = fmaxf(vmax[1], fabsf(((row_end > row_begin + batch * 128 + 80 + rgrp) ? vf8[1] : 0.0f)));
            ksum[2] = ksum[2] + ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[2] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[2] : 0.0f), ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[2] : 0.0f), ksq_a[0]);
            vmax[2] = fmaxf(vmax[2], fabsf(((row_end > row_begin + batch * 128 + 80 + rgrp) ? vf8[2] : 0.0f)));
            ksum[3] = ksum[3] + ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[3] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[3] : 0.0f), ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[3] : 0.0f), ksq_a[0]);
            vmax[3] = fmaxf(vmax[3], fabsf(((row_end > row_begin + batch * 128 + 80 + rgrp) ? vf8[3] : 0.0f)));
            ksum[4] = ksum[4] + ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[4] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[4] : 0.0f), ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[4] : 0.0f), ksq_a[0]);
            vmax[4] = fmaxf(vmax[4], fabsf(((row_end > row_begin + batch * 128 + 80 + rgrp) ? vf8[4] : 0.0f)));
            ksum[5] = ksum[5] + ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[5] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[5] : 0.0f), ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[5] : 0.0f), ksq_a[0]);
            vmax[5] = fmaxf(vmax[5], fabsf(((row_end > row_begin + batch * 128 + 80 + rgrp) ? vf8[5] : 0.0f)));
            ksum[6] = ksum[6] + ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[6] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[6] : 0.0f), ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[6] : 0.0f), ksq_a[0]);
            vmax[6] = fmaxf(vmax[6], fabsf(((row_end > row_begin + batch * 128 + 80 + rgrp) ? vf8[6] : 0.0f)));
            ksum[7] = ksum[7] + ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[7] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[7] : 0.0f), ((row_end > row_begin + batch * 128 + 80 + rgrp) ? kf8[7] : 0.0f), ksq_a[0]);
            vmax[7] = fmaxf(vmax[7], fabsf(((row_end > row_begin + batch * 128 + 80 + rgrp) ? vf8[7] : 0.0f)));
            float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 8);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_38;
            float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 4);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_39;
            float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 2);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_40;
            float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 1);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_41;
            knmax_a[0] = fmaxf(knmax_a[0], ksq_a[0]);
            kf8[0] = _vec_load_14[0];
            vf8[0] = _vec_load_15[0];
            kf8[1] = _vec_load_14[1];
            vf8[1] = _vec_load_15[1];
            kf8[2] = _vec_load_14[2];
            vf8[2] = _vec_load_15[2];
            kf8[3] = _vec_load_14[3];
            vf8[3] = _vec_load_15[3];
            kf8[4] = _vec_load_14[4];
            vf8[4] = _vec_load_15[4];
            kf8[5] = _vec_load_14[5];
            vf8[5] = _vec_load_15[5];
            kf8[6] = _vec_load_14[6];
            vf8[6] = _vec_load_15[6];
            kf8[7] = _vec_load_14[7];
            vf8[7] = _vec_load_15[7];
            ksq_a[0] = 0.0f;
            ksum[0] = ksum[0] + ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[0] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[0] : 0.0f), ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[0] : 0.0f), ksq_a[0]);
            vmax[0] = fmaxf(vmax[0], fabsf(((row_end > row_begin + batch * 128 + 96 + rgrp) ? vf8[0] : 0.0f)));
            ksum[1] = ksum[1] + ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[1] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[1] : 0.0f), ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[1] : 0.0f), ksq_a[0]);
            vmax[1] = fmaxf(vmax[1], fabsf(((row_end > row_begin + batch * 128 + 96 + rgrp) ? vf8[1] : 0.0f)));
            ksum[2] = ksum[2] + ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[2] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[2] : 0.0f), ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[2] : 0.0f), ksq_a[0]);
            vmax[2] = fmaxf(vmax[2], fabsf(((row_end > row_begin + batch * 128 + 96 + rgrp) ? vf8[2] : 0.0f)));
            ksum[3] = ksum[3] + ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[3] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[3] : 0.0f), ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[3] : 0.0f), ksq_a[0]);
            vmax[3] = fmaxf(vmax[3], fabsf(((row_end > row_begin + batch * 128 + 96 + rgrp) ? vf8[3] : 0.0f)));
            ksum[4] = ksum[4] + ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[4] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[4] : 0.0f), ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[4] : 0.0f), ksq_a[0]);
            vmax[4] = fmaxf(vmax[4], fabsf(((row_end > row_begin + batch * 128 + 96 + rgrp) ? vf8[4] : 0.0f)));
            ksum[5] = ksum[5] + ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[5] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[5] : 0.0f), ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[5] : 0.0f), ksq_a[0]);
            vmax[5] = fmaxf(vmax[5], fabsf(((row_end > row_begin + batch * 128 + 96 + rgrp) ? vf8[5] : 0.0f)));
            ksum[6] = ksum[6] + ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[6] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[6] : 0.0f), ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[6] : 0.0f), ksq_a[0]);
            vmax[6] = fmaxf(vmax[6], fabsf(((row_end > row_begin + batch * 128 + 96 + rgrp) ? vf8[6] : 0.0f)));
            ksum[7] = ksum[7] + ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[7] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[7] : 0.0f), ((row_end > row_begin + batch * 128 + 96 + rgrp) ? kf8[7] : 0.0f), ksq_a[0]);
            vmax[7] = fmaxf(vmax[7], fabsf(((row_end > row_begin + batch * 128 + 96 + rgrp) ? vf8[7] : 0.0f)));
            float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 8);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_42;
            float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 4);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_43;
            float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 2);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_44;
            float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 1);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_45;
            knmax_a[0] = fmaxf(knmax_a[0], ksq_a[0]);
            kf8[0] = _vec_load_16[0];
            vf8[0] = _vec_load_17[0];
            kf8[1] = _vec_load_16[1];
            vf8[1] = _vec_load_17[1];
            kf8[2] = _vec_load_16[2];
            vf8[2] = _vec_load_17[2];
            kf8[3] = _vec_load_16[3];
            vf8[3] = _vec_load_17[3];
            kf8[4] = _vec_load_16[4];
            vf8[4] = _vec_load_17[4];
            kf8[5] = _vec_load_16[5];
            vf8[5] = _vec_load_17[5];
            kf8[6] = _vec_load_16[6];
            vf8[6] = _vec_load_17[6];
            kf8[7] = _vec_load_16[7];
            vf8[7] = _vec_load_17[7];
            ksq_a[0] = 0.0f;
            ksum[0] = ksum[0] + ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[0] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[0] : 0.0f), ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[0] : 0.0f), ksq_a[0]);
            vmax[0] = fmaxf(vmax[0], fabsf(((row_end > row_begin + batch * 128 + 112 + rgrp) ? vf8[0] : 0.0f)));
            ksum[1] = ksum[1] + ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[1] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[1] : 0.0f), ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[1] : 0.0f), ksq_a[0]);
            vmax[1] = fmaxf(vmax[1], fabsf(((row_end > row_begin + batch * 128 + 112 + rgrp) ? vf8[1] : 0.0f)));
            ksum[2] = ksum[2] + ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[2] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[2] : 0.0f), ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[2] : 0.0f), ksq_a[0]);
            vmax[2] = fmaxf(vmax[2], fabsf(((row_end > row_begin + batch * 128 + 112 + rgrp) ? vf8[2] : 0.0f)));
            ksum[3] = ksum[3] + ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[3] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[3] : 0.0f), ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[3] : 0.0f), ksq_a[0]);
            vmax[3] = fmaxf(vmax[3], fabsf(((row_end > row_begin + batch * 128 + 112 + rgrp) ? vf8[3] : 0.0f)));
            ksum[4] = ksum[4] + ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[4] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[4] : 0.0f), ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[4] : 0.0f), ksq_a[0]);
            vmax[4] = fmaxf(vmax[4], fabsf(((row_end > row_begin + batch * 128 + 112 + rgrp) ? vf8[4] : 0.0f)));
            ksum[5] = ksum[5] + ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[5] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[5] : 0.0f), ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[5] : 0.0f), ksq_a[0]);
            vmax[5] = fmaxf(vmax[5], fabsf(((row_end > row_begin + batch * 128 + 112 + rgrp) ? vf8[5] : 0.0f)));
            ksum[6] = ksum[6] + ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[6] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[6] : 0.0f), ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[6] : 0.0f), ksq_a[0]);
            vmax[6] = fmaxf(vmax[6], fabsf(((row_end > row_begin + batch * 128 + 112 + rgrp) ? vf8[6] : 0.0f)));
            ksum[7] = ksum[7] + ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[7] : 0.0f);
            ksq_a[0] = __fmaf_rn(((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[7] : 0.0f), ((row_end > row_begin + batch * 128 + 112 + rgrp) ? kf8[7] : 0.0f), ksq_a[0]);
            vmax[7] = fmaxf(vmax[7], fabsf(((row_end > row_begin + batch * 128 + 112 + rgrp) ? vf8[7] : 0.0f)));
            float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 8);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_46;
            float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 4);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_47;
            float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 2);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_48;
            float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, ksq_a[0], 1);
            ksq_a[0] = ksq_a[0] + _shfl_xor_sync_49;
            knmax_a[0] = fmaxf(knmax_a[0], ksq_a[0]);
        }
        for (int e_1 = 0; e_1 < 8; e_1++) {
            red_sum[rgrp * 128 + col + e_1] = ksum[e_1];
            red_max[rgrp * 128 + col + e_1] = vmax[e_1];
        }
        if ((tid & 15) == 0) {
            red_norm[rgrp] = knmax_a[0];
        }
        __syncthreads();
        int direct = ((t1 - t0 == 1) ? 1 : 0);
        if (direct == 0) {
            if (tid < 128) {
                float total = 0.0f;
                float vmx = 0.0f;
                for (int g = 0; g < 16; g++) {
                    total = total + red_sum[g * 128 + tid];
                    vmx = fmaxf(vmx, red_max[g * 128 + tid]);
                }
                *(reinterpret_cast<float*>(partials + (work * 256 + tid)) + (0)) = total;
                *(reinterpret_cast<float*>(partials + (work * 256 + 128 + tid)) + (0)) = vmx;
            }
            if (tid == 0) {
                float kn = 0.0f;
                for (int g_1 = 0; g_1 < 16; g_1++) {
                    kn = fmaxf(kn, red_norm[g_1]);
                }
                *(reinterpret_cast<float*>(knorm_part + work) + (0)) = kn;
            }
            __syncthreads();
            if (tid == 0) {
                unsigned int old;
                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(old) : "l"(&counters[seg * num_heads + head]), "r"(static_cast<uint32_t>(1)) : "memory");
                int last = (((int)old == t1 - t0 - 1) ? 1 : 0);
                flag[0] = last;
                if (last == 1) {
                    *(reinterpret_cast<unsigned int*>(counters + (seg * num_heads + head)) + (0)) = (unsigned int)0;
                }
            }
            __syncthreads();
        }
        int is_last = ((direct == 0) ? flag[0] : 1);
        if (is_last == 1) {
            float acc_a[1];
            float knf_a[1];
            float fold_v[4];
            float fold_k[4];
            acc_a[0] = 0.0f;
            knf_a[0] = 0.0f;
            if (direct == 1) {
                for (int g_2 = 0; g_2 < 16; g_2++) {
                    float value = ((is_max == 0) ? red_sum[g_2 * 128 + chan] : red_max[g_2 * 128 + chan]);
                    acc_a[0] = ((is_max == 0) ? acc_a[0] + value : fmaxf(acc_a[0], value));
                    knf_a[0] = fmaxf(knf_a[0], red_norm[g_2]);
                }
            } else {
                #pragma unroll 1
                for (int tt = t0; tt < t1; tt += 4) {
                    fold_v[0] = partials[(((t1 > tt) ? tt : t1 - 1) * num_heads + head) * 256 + is_max * 128 + chan];
                    fold_k[0] = knorm_part[((t1 > tt) ? tt : t1 - 1) * num_heads + head];
                    fold_v[1] = partials[(((t1 > tt + 1) ? tt + 1 : t1 - 1) * num_heads + head) * 256 + is_max * 128 + chan];
                    fold_k[1] = knorm_part[((t1 > tt + 1) ? tt + 1 : t1 - 1) * num_heads + head];
                    fold_v[2] = partials[(((t1 > tt + 2) ? tt + 2 : t1 - 1) * num_heads + head) * 256 + is_max * 128 + chan];
                    fold_k[2] = knorm_part[((t1 > tt + 2) ? tt + 2 : t1 - 1) * num_heads + head];
                    fold_v[3] = partials[(((t1 > tt + 3) ? tt + 3 : t1 - 1) * num_heads + head) * 256 + is_max * 128 + chan];
                    fold_k[3] = knorm_part[((t1 > tt + 3) ? tt + 3 : t1 - 1) * num_heads + head];
                    acc_a[0] = ((is_max == 0) ? acc_a[0] + ((t1 > tt) ? fold_v[0] : 0.0f) : fmaxf(acc_a[0], ((t1 > tt) ? fold_v[0] : 0.0f)));
                    knf_a[0] = fmaxf(knf_a[0], ((t1 > tt) ? fold_k[0] : 0.0f));
                    acc_a[0] = ((is_max == 0) ? acc_a[0] + ((t1 > tt + 1) ? fold_v[1] : 0.0f) : fmaxf(acc_a[0], ((t1 > tt + 1) ? fold_v[1] : 0.0f)));
                    knf_a[0] = fmaxf(knf_a[0], ((t1 > tt + 1) ? fold_k[1] : 0.0f));
                    acc_a[0] = ((is_max == 0) ? acc_a[0] + ((t1 > tt + 2) ? fold_v[2] : 0.0f) : fmaxf(acc_a[0], ((t1 > tt + 2) ? fold_v[2] : 0.0f)));
                    knf_a[0] = fmaxf(knf_a[0], ((t1 > tt + 2) ? fold_k[2] : 0.0f));
                    acc_a[0] = ((is_max == 0) ? acc_a[0] + ((t1 > tt + 3) ? fold_v[3] : 0.0f) : fmaxf(acc_a[0], ((t1 > tt + 3) ? fold_v[3] : 0.0f)));
                    knf_a[0] = fmaxf(knf_a[0], ((t1 > tt + 3) ? fold_k[3] : 0.0f));
                }
            }
            float acc = acc_a[0];
            float knf = knf_a[0];
            float lenf = (float)(((length > 0) ? length : 1));
            float mean_v = __fdiv_rn(acc, lenf);
            float msq = ((is_max == 0) ? mean_v * mean_v : 0.0f);
            for (int stage = 0; stage < 5; stage++) {
                float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, msq, 16 >> stage);
                msq = msq + _shfl_xor_sync_50;
            }
            if (lane == 0) {
                red[warp] = msq;
            }
            __syncthreads();
            if (length > 0) {
                int sh = seg * num_heads + head;
                if (is_max == 0) {
                    *(reinterpret_cast<float*>(mean_k + (sh * 128 + chan)) + (0)) = mean_v;
                } else {
                    float _fdiv_51 = __fdiv_rn(fmaxf(acc, 1e-12f), 448.0f);
                    *(reinterpret_cast<float*>(v_scale + (sh * 128 + chan)) + (0)) = _fdiv_51;
                }
                if (tid == 0) {
                    float mnorm2 = red[0] + red[1] + (red[2] + red[3]);
                    float _sqrt_52;
                    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_52) : "f"(knf));
                    float _sqrt_53;
                    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_53) : "f"(mnorm2));
                    float bound = _sqrt_52 + _sqrt_53;
                    float _fdiv_54 = __fdiv_rn(fmaxf(bound, 1e-12f), 2688.0f);
                    *(reinterpret_cast<float*>(k_scale + sh) + (0)) = _fdiv_54;
                }
            }
        }
        __syncthreads();
    }
}

} // extern "C"
