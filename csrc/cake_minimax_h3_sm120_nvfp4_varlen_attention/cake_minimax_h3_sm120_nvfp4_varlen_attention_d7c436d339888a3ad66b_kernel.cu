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
#define NUM_MAIN_1_STAGES 1
#define SMEM_RED_SUM_OFF 0
#define SMEM_RED_SUM_STAGE_BYTES 8192
#define SMEM_RED_SUM_STRIDE 8192
#define SMEM_RED_Q_OFF 8192
#define SMEM_RED_Q_STAGE_BYTES 8192
#define SMEM_RED_Q_STRIDE 8192
#define SMEM_FLAG_OFF 16384
#define SMEM_FLAG_STAGE_BYTES 16
#define SMEM_FLAG_STRIDE 16
#define SMEM_QM_S_OFF 16512
#define SMEM_QM_S_STAGE_BYTES 512
#define SMEM_QM_S_STRIDE 512
#define SMEM_TOTAL 17024
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_minimax_h3_sm120_nvfp4_varlen_attention_d7c436d339888a3ad66b(__nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K, int* __restrict__ tile_table, int* __restrict__ seg_tile_begin, float* __restrict__ partials, unsigned int* __restrict__ counters, float* __restrict__ mean_k, float* __restrict__ q_mean, unsigned int* __restrict__ q4, uint8_t* __restrict__ q_sf, unsigned int* __restrict__ qm4, uint8_t* __restrict__ qm_sf, int num_tiles, int num_heads, int q_slots)
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
    float* red_q = reinterpret_cast<float*>(smem_raw + 8192);
    const int red_q_addr = smem + 8192;
    int* flag = reinterpret_cast<int*>(smem_raw + 16384);
    const int flag_addr = smem + 16384;
    float* qm_s = reinterpret_cast<float*>(smem_raw + 16512);
    const int qm_s_addr = smem + 16512;

    // === Task calls (declaration order) ===
    int col = (tid & 15) * 8;
    int rgrp = tid >> 4;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int per_tile = (1 + q_slots) * num_heads;
    #pragma unroll 1
    for (int work = bid; work < num_tiles * per_tile; work += num_bids) {
        int tile = work / per_tile;
        int hk = work - tile * per_tile;
        if (hk < num_heads) {
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
            for (int e = 0; e < 8; e++) {
                ksum[e] = 0.0f;
            }
            float kf8[8];
            #pragma unroll 1
            for (int batch = 0; batch < (row_end - row_begin + 128 - 1) / 128; batch++) {
                int brow = row_begin + batch * 128;
                float vec_load[8];
                {
                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(K + ((((row_end > brow + rgrp) ? brow + rgrp : row_end - 1) * num_heads + hk) * 128 + col) + 0);
                    uint4 _vld_2[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_2[_blk] = _vptr_2[_blk];
                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vec_load[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                            (&vec_load[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float vec_load_1[8];
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(K + ((((row_end > brow + 16 + rgrp) ? brow + 16 + rgrp : row_end - 1) * num_heads + hk) * 128 + col) + 0);
                    uint4 _vld_3[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_3[_blk] = _vptr_3[_blk];
                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                            (&vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float vec_load_2[8];
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(K + ((((row_end > brow + 32 + rgrp) ? brow + 32 + rgrp : row_end - 1) * num_heads + hk) * 128 + col) + 0);
                    uint4 _vld_4[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_4[_blk] = _vptr_4[_blk];
                        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                            (&vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float vec_load_3[8];
                {
                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(K + ((((row_end > brow + 48 + rgrp) ? brow + 48 + rgrp : row_end - 1) * num_heads + hk) * 128 + col) + 0);
                    uint4 _vld_5[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_5[_blk] = _vptr_5[_blk];
                        uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                            (&vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float vec_load_4[8];
                {
                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(K + ((((row_end > brow + 64 + rgrp) ? brow + 64 + rgrp : row_end - 1) * num_heads + hk) * 128 + col) + 0);
                    uint4 _vld_6[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_6[_blk] = _vptr_6[_blk];
                        uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                            (&vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float vec_load_5[8];
                {
                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(K + ((((row_end > brow + 80 + rgrp) ? brow + 80 + rgrp : row_end - 1) * num_heads + hk) * 128 + col) + 0);
                    uint4 _vld_7[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_7[_blk] = _vptr_7[_blk];
                        uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                            (&vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float vec_load_6[8];
                {
                    const uint4* _vptr_8 = reinterpret_cast<const uint4*>(K + ((((row_end > brow + 96 + rgrp) ? brow + 96 + rgrp : row_end - 1) * num_heads + hk) * 128 + col) + 0);
                    uint4 _vld_8[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_8[_blk] = _vptr_8[_blk];
                        uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vec_load_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                            (&vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float vec_load_7[8];
                {
                    const uint4* _vptr_9 = reinterpret_cast<const uint4*>(K + ((((row_end > brow + 112 + rgrp) ? brow + 112 + rgrp : row_end - 1) * num_heads + hk) * 128 + col) + 0);
                    uint4 _vld_9[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_9[_blk] = _vptr_9[_blk];
                        uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vec_load_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                            (&vec_load_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                        }
                    }
                }
                kf8[0] = vec_load[0];
                kf8[1] = vec_load[1];
                kf8[2] = vec_load[2];
                kf8[3] = vec_load[3];
                kf8[4] = vec_load[4];
                kf8[5] = vec_load[5];
                kf8[6] = vec_load[6];
                kf8[7] = vec_load[7];
                ksum[0] = ksum[0] + ((row_end > brow + rgrp) ? kf8[0] : 0.0f);
                ksum[1] = ksum[1] + ((row_end > brow + rgrp) ? kf8[1] : 0.0f);
                ksum[2] = ksum[2] + ((row_end > brow + rgrp) ? kf8[2] : 0.0f);
                ksum[3] = ksum[3] + ((row_end > brow + rgrp) ? kf8[3] : 0.0f);
                ksum[4] = ksum[4] + ((row_end > brow + rgrp) ? kf8[4] : 0.0f);
                ksum[5] = ksum[5] + ((row_end > brow + rgrp) ? kf8[5] : 0.0f);
                ksum[6] = ksum[6] + ((row_end > brow + rgrp) ? kf8[6] : 0.0f);
                ksum[7] = ksum[7] + ((row_end > brow + rgrp) ? kf8[7] : 0.0f);
                kf8[0] = vec_load_1[0];
                kf8[1] = vec_load_1[1];
                kf8[2] = vec_load_1[2];
                kf8[3] = vec_load_1[3];
                kf8[4] = vec_load_1[4];
                kf8[5] = vec_load_1[5];
                kf8[6] = vec_load_1[6];
                kf8[7] = vec_load_1[7];
                ksum[0] = ksum[0] + ((row_end > brow + 16 + rgrp) ? kf8[0] : 0.0f);
                ksum[1] = ksum[1] + ((row_end > brow + 16 + rgrp) ? kf8[1] : 0.0f);
                ksum[2] = ksum[2] + ((row_end > brow + 16 + rgrp) ? kf8[2] : 0.0f);
                ksum[3] = ksum[3] + ((row_end > brow + 16 + rgrp) ? kf8[3] : 0.0f);
                ksum[4] = ksum[4] + ((row_end > brow + 16 + rgrp) ? kf8[4] : 0.0f);
                ksum[5] = ksum[5] + ((row_end > brow + 16 + rgrp) ? kf8[5] : 0.0f);
                ksum[6] = ksum[6] + ((row_end > brow + 16 + rgrp) ? kf8[6] : 0.0f);
                ksum[7] = ksum[7] + ((row_end > brow + 16 + rgrp) ? kf8[7] : 0.0f);
                kf8[0] = vec_load_2[0];
                kf8[1] = vec_load_2[1];
                kf8[2] = vec_load_2[2];
                kf8[3] = vec_load_2[3];
                kf8[4] = vec_load_2[4];
                kf8[5] = vec_load_2[5];
                kf8[6] = vec_load_2[6];
                kf8[7] = vec_load_2[7];
                ksum[0] = ksum[0] + ((row_end > brow + 32 + rgrp) ? kf8[0] : 0.0f);
                ksum[1] = ksum[1] + ((row_end > brow + 32 + rgrp) ? kf8[1] : 0.0f);
                ksum[2] = ksum[2] + ((row_end > brow + 32 + rgrp) ? kf8[2] : 0.0f);
                ksum[3] = ksum[3] + ((row_end > brow + 32 + rgrp) ? kf8[3] : 0.0f);
                ksum[4] = ksum[4] + ((row_end > brow + 32 + rgrp) ? kf8[4] : 0.0f);
                ksum[5] = ksum[5] + ((row_end > brow + 32 + rgrp) ? kf8[5] : 0.0f);
                ksum[6] = ksum[6] + ((row_end > brow + 32 + rgrp) ? kf8[6] : 0.0f);
                ksum[7] = ksum[7] + ((row_end > brow + 32 + rgrp) ? kf8[7] : 0.0f);
                kf8[0] = vec_load_3[0];
                kf8[1] = vec_load_3[1];
                kf8[2] = vec_load_3[2];
                kf8[3] = vec_load_3[3];
                kf8[4] = vec_load_3[4];
                kf8[5] = vec_load_3[5];
                kf8[6] = vec_load_3[6];
                kf8[7] = vec_load_3[7];
                ksum[0] = ksum[0] + ((row_end > brow + 48 + rgrp) ? kf8[0] : 0.0f);
                ksum[1] = ksum[1] + ((row_end > brow + 48 + rgrp) ? kf8[1] : 0.0f);
                ksum[2] = ksum[2] + ((row_end > brow + 48 + rgrp) ? kf8[2] : 0.0f);
                ksum[3] = ksum[3] + ((row_end > brow + 48 + rgrp) ? kf8[3] : 0.0f);
                ksum[4] = ksum[4] + ((row_end > brow + 48 + rgrp) ? kf8[4] : 0.0f);
                ksum[5] = ksum[5] + ((row_end > brow + 48 + rgrp) ? kf8[5] : 0.0f);
                ksum[6] = ksum[6] + ((row_end > brow + 48 + rgrp) ? kf8[6] : 0.0f);
                ksum[7] = ksum[7] + ((row_end > brow + 48 + rgrp) ? kf8[7] : 0.0f);
                kf8[0] = vec_load_4[0];
                kf8[1] = vec_load_4[1];
                kf8[2] = vec_load_4[2];
                kf8[3] = vec_load_4[3];
                kf8[4] = vec_load_4[4];
                kf8[5] = vec_load_4[5];
                kf8[6] = vec_load_4[6];
                kf8[7] = vec_load_4[7];
                ksum[0] = ksum[0] + ((row_end > brow + 64 + rgrp) ? kf8[0] : 0.0f);
                ksum[1] = ksum[1] + ((row_end > brow + 64 + rgrp) ? kf8[1] : 0.0f);
                ksum[2] = ksum[2] + ((row_end > brow + 64 + rgrp) ? kf8[2] : 0.0f);
                ksum[3] = ksum[3] + ((row_end > brow + 64 + rgrp) ? kf8[3] : 0.0f);
                ksum[4] = ksum[4] + ((row_end > brow + 64 + rgrp) ? kf8[4] : 0.0f);
                ksum[5] = ksum[5] + ((row_end > brow + 64 + rgrp) ? kf8[5] : 0.0f);
                ksum[6] = ksum[6] + ((row_end > brow + 64 + rgrp) ? kf8[6] : 0.0f);
                ksum[7] = ksum[7] + ((row_end > brow + 64 + rgrp) ? kf8[7] : 0.0f);
                kf8[0] = vec_load_5[0];
                kf8[1] = vec_load_5[1];
                kf8[2] = vec_load_5[2];
                kf8[3] = vec_load_5[3];
                kf8[4] = vec_load_5[4];
                kf8[5] = vec_load_5[5];
                kf8[6] = vec_load_5[6];
                kf8[7] = vec_load_5[7];
                ksum[0] = ksum[0] + ((row_end > brow + 80 + rgrp) ? kf8[0] : 0.0f);
                ksum[1] = ksum[1] + ((row_end > brow + 80 + rgrp) ? kf8[1] : 0.0f);
                ksum[2] = ksum[2] + ((row_end > brow + 80 + rgrp) ? kf8[2] : 0.0f);
                ksum[3] = ksum[3] + ((row_end > brow + 80 + rgrp) ? kf8[3] : 0.0f);
                ksum[4] = ksum[4] + ((row_end > brow + 80 + rgrp) ? kf8[4] : 0.0f);
                ksum[5] = ksum[5] + ((row_end > brow + 80 + rgrp) ? kf8[5] : 0.0f);
                ksum[6] = ksum[6] + ((row_end > brow + 80 + rgrp) ? kf8[6] : 0.0f);
                ksum[7] = ksum[7] + ((row_end > brow + 80 + rgrp) ? kf8[7] : 0.0f);
                kf8[0] = vec_load_6[0];
                kf8[1] = vec_load_6[1];
                kf8[2] = vec_load_6[2];
                kf8[3] = vec_load_6[3];
                kf8[4] = vec_load_6[4];
                kf8[5] = vec_load_6[5];
                kf8[6] = vec_load_6[6];
                kf8[7] = vec_load_6[7];
                ksum[0] = ksum[0] + ((row_end > brow + 96 + rgrp) ? kf8[0] : 0.0f);
                ksum[1] = ksum[1] + ((row_end > brow + 96 + rgrp) ? kf8[1] : 0.0f);
                ksum[2] = ksum[2] + ((row_end > brow + 96 + rgrp) ? kf8[2] : 0.0f);
                ksum[3] = ksum[3] + ((row_end > brow + 96 + rgrp) ? kf8[3] : 0.0f);
                ksum[4] = ksum[4] + ((row_end > brow + 96 + rgrp) ? kf8[4] : 0.0f);
                ksum[5] = ksum[5] + ((row_end > brow + 96 + rgrp) ? kf8[5] : 0.0f);
                ksum[6] = ksum[6] + ((row_end > brow + 96 + rgrp) ? kf8[6] : 0.0f);
                ksum[7] = ksum[7] + ((row_end > brow + 96 + rgrp) ? kf8[7] : 0.0f);
                kf8[0] = vec_load_7[0];
                kf8[1] = vec_load_7[1];
                kf8[2] = vec_load_7[2];
                kf8[3] = vec_load_7[3];
                kf8[4] = vec_load_7[4];
                kf8[5] = vec_load_7[5];
                kf8[6] = vec_load_7[6];
                kf8[7] = vec_load_7[7];
                ksum[0] = ksum[0] + ((row_end > brow + 112 + rgrp) ? kf8[0] : 0.0f);
                ksum[1] = ksum[1] + ((row_end > brow + 112 + rgrp) ? kf8[1] : 0.0f);
                ksum[2] = ksum[2] + ((row_end > brow + 112 + rgrp) ? kf8[2] : 0.0f);
                ksum[3] = ksum[3] + ((row_end > brow + 112 + rgrp) ? kf8[3] : 0.0f);
                ksum[4] = ksum[4] + ((row_end > brow + 112 + rgrp) ? kf8[4] : 0.0f);
                ksum[5] = ksum[5] + ((row_end > brow + 112 + rgrp) ? kf8[5] : 0.0f);
                ksum[6] = ksum[6] + ((row_end > brow + 112 + rgrp) ? kf8[6] : 0.0f);
                ksum[7] = ksum[7] + ((row_end > brow + 112 + rgrp) ? kf8[7] : 0.0f);
            }
            for (int e_1 = 0; e_1 < 8; e_1++) {
                red_sum[rgrp * 128 + col + e_1] = ksum[e_1];
            }
            __syncthreads();
            int direct = ((t1 - t0 == 1) ? 1 : 0);
            if (direct == 0) {
                if (tid < 128) {
                    float total = 0.0f;
                    for (int g = 0; g < 16; g++) {
                        total = total + red_sum[g * 128 + tid];
                    }
                    *(reinterpret_cast<float*>(partials + ((tile * num_heads + hk) * 128 + tid)) + (0)) = total;
                }
                __syncthreads();
                if (tid == 0) {
                    unsigned int old;
                    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(old) : "l"(&counters[seg * num_heads + hk]), "r"(static_cast<uint32_t>(1)) : "memory");
                    int last = (((int)old == t1 - t0 - 1) ? 1 : 0);
                    flag[0] = last;
                    if (last == 1) {
                        *(reinterpret_cast<unsigned int*>(counters + (seg * num_heads + hk)) + (0)) = (unsigned int)0;
                    }
                }
                __syncthreads();
            }
            int is_last = ((direct == 0) ? flag[0] : 1);
            if (is_last == 1) {
                if (tid < 128) {
                    float acc_a[1];
                    float fold_v[4];
                    acc_a[0] = 0.0f;
                    if (direct == 1) {
                        for (int g_1 = 0; g_1 < 16; g_1++) {
                            acc_a[0] = acc_a[0] + red_sum[g_1 * 128 + tid];
                        }
                    } else {
                        #pragma unroll 1
                        for (int tt = t0; tt < t1; tt += 4) {
                            for (int j = 0; j < 4; j++) {
                                int ttj = ((t1 > tt + j) ? tt + j : t1 - 1);
                                fold_v[j] = partials[(ttj * num_heads + hk) * 128 + tid];
                            }
                            for (int j_1 = 0; j_1 < 4; j_1++) {
                                float fv = ((t1 > tt + j_1) ? fold_v[j_1] : 0.0f);
                                acc_a[0] = acc_a[0] + fv;
                            }
                        }
                    }
                    if (length > 0) {
                        float lenf = (float)length;
                        float _fdiv_10 = __fdiv_rn(acc_a[0], lenf);
                        *(reinterpret_cast<float*>(mean_k + ((seg * num_heads + hk) * 128 + tid)) + (0)) = _fdiv_10;
                    }
                }
            }
            __syncthreads();
        } else {
            int slot = (hk - num_heads) / num_heads;
            int _vec_load_10[4];
            {
                const int4* _ivptr_11 = reinterpret_cast<const int4*>(tile_table + (8 * tile) + 0);
                int4 _ivld_11;
                _ivld_11 = *_ivptr_11;
                _vec_load_10[0 + 0] = _ivld_11.x;
                _vec_load_10[0 + 1] = _ivld_11.y;
                _vec_load_10[0 + 2] = _ivld_11.z;
                _vec_load_10[0 + 3] = _ivld_11.w;
            }
            int seg_1 = _vec_load_10[0];
            int row_begin_1 = _vec_load_10[1];
            int row_end_1 = _vec_load_10[2];
            int t0_1 = _vec_load_10[3];
            int brow_1 = row_begin_1 + slot * 128;
            if (brow_1 < row_end_1) {
                int qt = seg_tile_begin[seg_1] + (tile - t0_1) + slot;
                float qsum[8];
                float qf8[8];
                for (int e_2 = 0; e_2 < 8; e_2++) {
                    qsum[e_2] = 0.0f;
                }
                float vec_load_8[8];
                {
                    const uint4* _vptr_12 = reinterpret_cast<const uint4*>(Q + ((((row_end_1 > brow_1 + rgrp) ? brow_1 + rgrp : row_end_1 - 1) * num_heads + (hk - num_heads - slot * num_heads)) * 128 + col) + 0);
                    uint4 _vld_12[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_12[_blk] = _vptr_12[_blk];
                        uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vec_load_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
                            (&vec_load_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float vec_load_1_1[8];
                {
                    const uint4* _vptr_13 = reinterpret_cast<const uint4*>(Q + ((((row_end_1 > brow_1 + 16 + rgrp) ? brow_1 + 16 + rgrp : row_end_1 - 1) * num_heads + (hk - num_heads - slot * num_heads)) * 128 + col) + 0);
                    uint4 _vld_13[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_13[_blk] = _vptr_13[_blk];
                        uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vec_load_1_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) << 16);
                            (&vec_load_1_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float vec_load_2_1[8];
                {
                    const uint4* _vptr_14 = reinterpret_cast<const uint4*>(Q + ((((row_end_1 > brow_1 + 32 + rgrp) ? brow_1 + 32 + rgrp : row_end_1 - 1) * num_heads + (hk - num_heads - slot * num_heads)) * 128 + col) + 0);
                    uint4 _vld_14[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_14[_blk] = _vptr_14[_blk];
                        uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vec_load_2_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) << 16);
                            (&vec_load_2_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float vec_load_3_1[8];
                {
                    const uint4* _vptr_15 = reinterpret_cast<const uint4*>(Q + ((((row_end_1 > brow_1 + 48 + rgrp) ? brow_1 + 48 + rgrp : row_end_1 - 1) * num_heads + (hk - num_heads - slot * num_heads)) * 128 + col) + 0);
                    uint4 _vld_15[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_15[_blk] = _vptr_15[_blk];
                        uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vec_load_3_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) << 16);
                            (&vec_load_3_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float vec_load_4_1[8];
                {
                    const uint4* _vptr_16 = reinterpret_cast<const uint4*>(Q + ((((row_end_1 > brow_1 + 64 + rgrp) ? brow_1 + 64 + rgrp : row_end_1 - 1) * num_heads + (hk - num_heads - slot * num_heads)) * 128 + col) + 0);
                    uint4 _vld_16[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_16[_blk] = _vptr_16[_blk];
                        uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vec_load_4_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
                            (&vec_load_4_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float vec_load_5_1[8];
                {
                    const uint4* _vptr_17 = reinterpret_cast<const uint4*>(Q + ((((row_end_1 > brow_1 + 80 + rgrp) ? brow_1 + 80 + rgrp : row_end_1 - 1) * num_heads + (hk - num_heads - slot * num_heads)) * 128 + col) + 0);
                    uint4 _vld_17[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_17[_blk] = _vptr_17[_blk];
                        uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vec_load_5_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) << 16);
                            (&vec_load_5_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float vec_load_6_1[8];
                {
                    const uint4* _vptr_18 = reinterpret_cast<const uint4*>(Q + ((((row_end_1 > brow_1 + 96 + rgrp) ? brow_1 + 96 + rgrp : row_end_1 - 1) * num_heads + (hk - num_heads - slot * num_heads)) * 128 + col) + 0);
                    uint4 _vld_18[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_18[_blk] = _vptr_18[_blk];
                        uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vec_load_6_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
                            (&vec_load_6_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
                        }
                    }
                }
                float vec_load_7_1[8];
                {
                    const uint4* _vptr_19 = reinterpret_cast<const uint4*>(Q + ((((row_end_1 > brow_1 + 112 + rgrp) ? brow_1 + 112 + rgrp : row_end_1 - 1) * num_heads + (hk - num_heads - slot * num_heads)) * 128 + col) + 0);
                    uint4 _vld_19[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_19[_blk] = _vptr_19[_blk];
                        uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vec_load_7_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) << 16);
                            (&vec_load_7_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) & 0xffff0000u);
                        }
                    }
                }
                qf8[0] = vec_load_8[0];
                qf8[1] = vec_load_8[1];
                qf8[2] = vec_load_8[2];
                qf8[3] = vec_load_8[3];
                qf8[4] = vec_load_8[4];
                qf8[5] = vec_load_8[5];
                qf8[6] = vec_load_8[6];
                qf8[7] = vec_load_8[7];
                qsum[0] = qsum[0] + ((row_end_1 > brow_1 + rgrp) ? qf8[0] : 0.0f);
                qsum[1] = qsum[1] + ((row_end_1 > brow_1 + rgrp) ? qf8[1] : 0.0f);
                qsum[2] = qsum[2] + ((row_end_1 > brow_1 + rgrp) ? qf8[2] : 0.0f);
                qsum[3] = qsum[3] + ((row_end_1 > brow_1 + rgrp) ? qf8[3] : 0.0f);
                qsum[4] = qsum[4] + ((row_end_1 > brow_1 + rgrp) ? qf8[4] : 0.0f);
                qsum[5] = qsum[5] + ((row_end_1 > brow_1 + rgrp) ? qf8[5] : 0.0f);
                qsum[6] = qsum[6] + ((row_end_1 > brow_1 + rgrp) ? qf8[6] : 0.0f);
                qsum[7] = qsum[7] + ((row_end_1 > brow_1 + rgrp) ? qf8[7] : 0.0f);
                qf8[0] = vec_load_1_1[0];
                qf8[1] = vec_load_1_1[1];
                qf8[2] = vec_load_1_1[2];
                qf8[3] = vec_load_1_1[3];
                qf8[4] = vec_load_1_1[4];
                qf8[5] = vec_load_1_1[5];
                qf8[6] = vec_load_1_1[6];
                qf8[7] = vec_load_1_1[7];
                qsum[0] = qsum[0] + ((row_end_1 > brow_1 + 16 + rgrp) ? qf8[0] : 0.0f);
                qsum[1] = qsum[1] + ((row_end_1 > brow_1 + 16 + rgrp) ? qf8[1] : 0.0f);
                qsum[2] = qsum[2] + ((row_end_1 > brow_1 + 16 + rgrp) ? qf8[2] : 0.0f);
                qsum[3] = qsum[3] + ((row_end_1 > brow_1 + 16 + rgrp) ? qf8[3] : 0.0f);
                qsum[4] = qsum[4] + ((row_end_1 > brow_1 + 16 + rgrp) ? qf8[4] : 0.0f);
                qsum[5] = qsum[5] + ((row_end_1 > brow_1 + 16 + rgrp) ? qf8[5] : 0.0f);
                qsum[6] = qsum[6] + ((row_end_1 > brow_1 + 16 + rgrp) ? qf8[6] : 0.0f);
                qsum[7] = qsum[7] + ((row_end_1 > brow_1 + 16 + rgrp) ? qf8[7] : 0.0f);
                qf8[0] = vec_load_2_1[0];
                qf8[1] = vec_load_2_1[1];
                qf8[2] = vec_load_2_1[2];
                qf8[3] = vec_load_2_1[3];
                qf8[4] = vec_load_2_1[4];
                qf8[5] = vec_load_2_1[5];
                qf8[6] = vec_load_2_1[6];
                qf8[7] = vec_load_2_1[7];
                qsum[0] = qsum[0] + ((row_end_1 > brow_1 + 32 + rgrp) ? qf8[0] : 0.0f);
                qsum[1] = qsum[1] + ((row_end_1 > brow_1 + 32 + rgrp) ? qf8[1] : 0.0f);
                qsum[2] = qsum[2] + ((row_end_1 > brow_1 + 32 + rgrp) ? qf8[2] : 0.0f);
                qsum[3] = qsum[3] + ((row_end_1 > brow_1 + 32 + rgrp) ? qf8[3] : 0.0f);
                qsum[4] = qsum[4] + ((row_end_1 > brow_1 + 32 + rgrp) ? qf8[4] : 0.0f);
                qsum[5] = qsum[5] + ((row_end_1 > brow_1 + 32 + rgrp) ? qf8[5] : 0.0f);
                qsum[6] = qsum[6] + ((row_end_1 > brow_1 + 32 + rgrp) ? qf8[6] : 0.0f);
                qsum[7] = qsum[7] + ((row_end_1 > brow_1 + 32 + rgrp) ? qf8[7] : 0.0f);
                qf8[0] = vec_load_3_1[0];
                qf8[1] = vec_load_3_1[1];
                qf8[2] = vec_load_3_1[2];
                qf8[3] = vec_load_3_1[3];
                qf8[4] = vec_load_3_1[4];
                qf8[5] = vec_load_3_1[5];
                qf8[6] = vec_load_3_1[6];
                qf8[7] = vec_load_3_1[7];
                qsum[0] = qsum[0] + ((row_end_1 > brow_1 + 48 + rgrp) ? qf8[0] : 0.0f);
                qsum[1] = qsum[1] + ((row_end_1 > brow_1 + 48 + rgrp) ? qf8[1] : 0.0f);
                qsum[2] = qsum[2] + ((row_end_1 > brow_1 + 48 + rgrp) ? qf8[2] : 0.0f);
                qsum[3] = qsum[3] + ((row_end_1 > brow_1 + 48 + rgrp) ? qf8[3] : 0.0f);
                qsum[4] = qsum[4] + ((row_end_1 > brow_1 + 48 + rgrp) ? qf8[4] : 0.0f);
                qsum[5] = qsum[5] + ((row_end_1 > brow_1 + 48 + rgrp) ? qf8[5] : 0.0f);
                qsum[6] = qsum[6] + ((row_end_1 > brow_1 + 48 + rgrp) ? qf8[6] : 0.0f);
                qsum[7] = qsum[7] + ((row_end_1 > brow_1 + 48 + rgrp) ? qf8[7] : 0.0f);
                qf8[0] = vec_load_4_1[0];
                qf8[1] = vec_load_4_1[1];
                qf8[2] = vec_load_4_1[2];
                qf8[3] = vec_load_4_1[3];
                qf8[4] = vec_load_4_1[4];
                qf8[5] = vec_load_4_1[5];
                qf8[6] = vec_load_4_1[6];
                qf8[7] = vec_load_4_1[7];
                qsum[0] = qsum[0] + ((row_end_1 > brow_1 + 64 + rgrp) ? qf8[0] : 0.0f);
                qsum[1] = qsum[1] + ((row_end_1 > brow_1 + 64 + rgrp) ? qf8[1] : 0.0f);
                qsum[2] = qsum[2] + ((row_end_1 > brow_1 + 64 + rgrp) ? qf8[2] : 0.0f);
                qsum[3] = qsum[3] + ((row_end_1 > brow_1 + 64 + rgrp) ? qf8[3] : 0.0f);
                qsum[4] = qsum[4] + ((row_end_1 > brow_1 + 64 + rgrp) ? qf8[4] : 0.0f);
                qsum[5] = qsum[5] + ((row_end_1 > brow_1 + 64 + rgrp) ? qf8[5] : 0.0f);
                qsum[6] = qsum[6] + ((row_end_1 > brow_1 + 64 + rgrp) ? qf8[6] : 0.0f);
                qsum[7] = qsum[7] + ((row_end_1 > brow_1 + 64 + rgrp) ? qf8[7] : 0.0f);
                qf8[0] = vec_load_5_1[0];
                qf8[1] = vec_load_5_1[1];
                qf8[2] = vec_load_5_1[2];
                qf8[3] = vec_load_5_1[3];
                qf8[4] = vec_load_5_1[4];
                qf8[5] = vec_load_5_1[5];
                qf8[6] = vec_load_5_1[6];
                qf8[7] = vec_load_5_1[7];
                qsum[0] = qsum[0] + ((row_end_1 > brow_1 + 80 + rgrp) ? qf8[0] : 0.0f);
                qsum[1] = qsum[1] + ((row_end_1 > brow_1 + 80 + rgrp) ? qf8[1] : 0.0f);
                qsum[2] = qsum[2] + ((row_end_1 > brow_1 + 80 + rgrp) ? qf8[2] : 0.0f);
                qsum[3] = qsum[3] + ((row_end_1 > brow_1 + 80 + rgrp) ? qf8[3] : 0.0f);
                qsum[4] = qsum[4] + ((row_end_1 > brow_1 + 80 + rgrp) ? qf8[4] : 0.0f);
                qsum[5] = qsum[5] + ((row_end_1 > brow_1 + 80 + rgrp) ? qf8[5] : 0.0f);
                qsum[6] = qsum[6] + ((row_end_1 > brow_1 + 80 + rgrp) ? qf8[6] : 0.0f);
                qsum[7] = qsum[7] + ((row_end_1 > brow_1 + 80 + rgrp) ? qf8[7] : 0.0f);
                qf8[0] = vec_load_6_1[0];
                qf8[1] = vec_load_6_1[1];
                qf8[2] = vec_load_6_1[2];
                qf8[3] = vec_load_6_1[3];
                qf8[4] = vec_load_6_1[4];
                qf8[5] = vec_load_6_1[5];
                qf8[6] = vec_load_6_1[6];
                qf8[7] = vec_load_6_1[7];
                qsum[0] = qsum[0] + ((row_end_1 > brow_1 + 96 + rgrp) ? qf8[0] : 0.0f);
                qsum[1] = qsum[1] + ((row_end_1 > brow_1 + 96 + rgrp) ? qf8[1] : 0.0f);
                qsum[2] = qsum[2] + ((row_end_1 > brow_1 + 96 + rgrp) ? qf8[2] : 0.0f);
                qsum[3] = qsum[3] + ((row_end_1 > brow_1 + 96 + rgrp) ? qf8[3] : 0.0f);
                qsum[4] = qsum[4] + ((row_end_1 > brow_1 + 96 + rgrp) ? qf8[4] : 0.0f);
                qsum[5] = qsum[5] + ((row_end_1 > brow_1 + 96 + rgrp) ? qf8[5] : 0.0f);
                qsum[6] = qsum[6] + ((row_end_1 > brow_1 + 96 + rgrp) ? qf8[6] : 0.0f);
                qsum[7] = qsum[7] + ((row_end_1 > brow_1 + 96 + rgrp) ? qf8[7] : 0.0f);
                qf8[0] = vec_load_7_1[0];
                qf8[1] = vec_load_7_1[1];
                qf8[2] = vec_load_7_1[2];
                qf8[3] = vec_load_7_1[3];
                qf8[4] = vec_load_7_1[4];
                qf8[5] = vec_load_7_1[5];
                qf8[6] = vec_load_7_1[6];
                qf8[7] = vec_load_7_1[7];
                qsum[0] = qsum[0] + ((row_end_1 > brow_1 + 112 + rgrp) ? qf8[0] : 0.0f);
                qsum[1] = qsum[1] + ((row_end_1 > brow_1 + 112 + rgrp) ? qf8[1] : 0.0f);
                qsum[2] = qsum[2] + ((row_end_1 > brow_1 + 112 + rgrp) ? qf8[2] : 0.0f);
                qsum[3] = qsum[3] + ((row_end_1 > brow_1 + 112 + rgrp) ? qf8[3] : 0.0f);
                qsum[4] = qsum[4] + ((row_end_1 > brow_1 + 112 + rgrp) ? qf8[4] : 0.0f);
                qsum[5] = qsum[5] + ((row_end_1 > brow_1 + 112 + rgrp) ? qf8[5] : 0.0f);
                qsum[6] = qsum[6] + ((row_end_1 > brow_1 + 112 + rgrp) ? qf8[6] : 0.0f);
                qsum[7] = qsum[7] + ((row_end_1 > brow_1 + 112 + rgrp) ? qf8[7] : 0.0f);
                for (int e_3 = 0; e_3 < 8; e_3++) {
                    red_q[rgrp * 128 + col + e_3] = qsum[e_3];
                }
                __syncthreads();
                if (tid < 128) {
                    float qtot = 0.0f;
                    for (int g_2 = 0; g_2 < 16; g_2++) {
                        qtot = qtot + red_q[g_2 * 128 + tid];
                    }
                    int brows = ((row_end_1 - brow_1 < 128) ? row_end_1 - brow_1 : 128);
                    float qmean = __fdiv_rn(qtot, (float)brows);
                    *(reinterpret_cast<float*>(q_mean + ((qt * num_heads + (hk - num_heads - slot * num_heads)) * 128 + tid)) + (0)) = qmean;
                    qm_s[tid] = qmean;
                }
                __syncthreads();
                int lane16 = tid & 15;
                float qmv[8];
                for (int e_4 = 0; e_4 < 8; e_4++) {
                    qmv[e_4] = qm_s[col + e_4];
                }
                float qc[8];
                unsigned int qw[1];
                float qsf_a[1];
                for (int e_5 = 0; e_5 < 8; e_5++) {
                    float qf = vec_load_8[e_5];
                    qc[e_5] = ((row_end_1 > brow_1 + rgrp) ? qf - qmv[e_5] : 0.0f);
                }
                float max_1 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(qc[0]), fabsf(qc[1])), fabsf(qc[2])), fabsf(qc[3])), fabsf(qc[4])), fabsf(qc[5])), fabsf(qc[6])), fabsf(qc[7]));
                float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, max_1, 1);
                float max_2 = fmaxf(max_1, _shfl_xor_sync_20);
                uint16_t _cvt_satfinite_e4m3x2_f32_21;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_21) : "f"(0.0f), "f"(max_2 * 0.16666667f));
                uint16_t _e4m3x2_decode_22 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_21 & 0xFFu);
                uint32_t _f16x2_decode_22;
                float fp8_decode;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_22) : "h"(_e4m3x2_decode_22));
                uint16_t _f16_decode_22 = (uint16_t)_f16x2_decode_22;
                asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode) : "h"(_f16_decode_22));
                float fdiv_rn = __fdiv_rn(1.0f, fp8_decode);
                qc[0] = qc[0] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[1] = qc[1] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[2] = qc[2] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[3] = qc[3] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[4] = qc[4] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[5] = qc[5] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[6] = qc[6] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[7] = qc[7] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw[0]) : "f"(qc[0]), "f"(qc[1]), "f"(qc[2]), "f"(qc[3]), "f"(qc[4]), "f"(qc[5]), "f"(qc[6]), "f"(qc[7]));
                qsf_a[0] = max_2 * 0.16666667f;
                if (row_end_1 > brow_1 + rgrp) {
                    int qrow = (brow_1 + rgrp) * num_heads + (hk - num_heads - slot * num_heads);
                    *(reinterpret_cast<unsigned int*>(q4 + (qrow * 16 + lane16)) + (0)) = qw[0];
                    if ((lane16 & 1) == 0) {
                        uint16_t _cvt_satfinite_e4m3x2_f32_23;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_23) : "f"(0.0f), "f"(qsf_a[0]));
                        *reinterpret_cast<unsigned char*>(q_sf + (qrow * 8 + (lane16 >> 1))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_23 & 127));
                    }
                }
                for (int e_6 = 0; e_6 < 8; e_6++) {
                    float qf_1 = vec_load_1_1[e_6];
                    qc[e_6] = ((row_end_1 > brow_1 + 16 + rgrp) ? qf_1 - qmv[e_6] : 0.0f);
                }
                float max_1_0 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(qc[0]), fabsf(qc[1])), fabsf(qc[2])), fabsf(qc[3])), fabsf(qc[4])), fabsf(qc[5])), fabsf(qc[6])), fabsf(qc[7]));
                float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, max_1_0, 1);
                float max_2_0 = fmaxf(max_1_0, _shfl_xor_sync_24);
                uint16_t _cvt_satfinite_e4m3x2_f32_25;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_25) : "f"(0.0f), "f"(max_2_0 * 0.16666667f));
                uint16_t _e4m3x2_decode_26 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_25 & 0xFFu);
                uint32_t _f16x2_decode_26;
                float fp8_decode_0;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_26) : "h"(_e4m3x2_decode_26));
                uint16_t _f16_decode_26 = (uint16_t)_f16x2_decode_26;
                asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode_0) : "h"(_f16_decode_26));
                float fdiv_rn_0 = __fdiv_rn(1.0f, fp8_decode_0);
                qc[0] = qc[0] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                qc[1] = qc[1] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                qc[2] = qc[2] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                qc[3] = qc[3] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                qc[4] = qc[4] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                qc[5] = qc[5] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                qc[6] = qc[6] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                qc[7] = qc[7] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw[0]) : "f"(qc[0]), "f"(qc[1]), "f"(qc[2]), "f"(qc[3]), "f"(qc[4]), "f"(qc[5]), "f"(qc[6]), "f"(qc[7]));
                qsf_a[0] = max_2_0 * 0.16666667f;
                if (row_end_1 > brow_1 + 16 + rgrp) {
                    int qrow_1 = (brow_1 + 16 + rgrp) * num_heads + (hk - num_heads - slot * num_heads);
                    *(reinterpret_cast<unsigned int*>(q4 + (qrow_1 * 16 + lane16)) + (0)) = qw[0];
                    if ((lane16 & 1) == 0) {
                        uint16_t _cvt_satfinite_e4m3x2_f32_27;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_27) : "f"(0.0f), "f"(qsf_a[0]));
                        *reinterpret_cast<unsigned char*>(q_sf + (qrow_1 * 8 + (lane16 >> 1))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_27 & 127));
                    }
                }
                for (int e_7 = 0; e_7 < 8; e_7++) {
                    float qf_2 = vec_load_2_1[e_7];
                    qc[e_7] = ((row_end_1 > brow_1 + 32 + rgrp) ? qf_2 - qmv[e_7] : 0.0f);
                }
                float max_1_1 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(qc[0]), fabsf(qc[1])), fabsf(qc[2])), fabsf(qc[3])), fabsf(qc[4])), fabsf(qc[5])), fabsf(qc[6])), fabsf(qc[7]));
                float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, max_1_1, 1);
                float max_2_1 = fmaxf(max_1_1, _shfl_xor_sync_28);
                uint16_t _cvt_satfinite_e4m3x2_f32_29;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_29) : "f"(0.0f), "f"(max_2_1 * 0.16666667f));
                uint16_t _e4m3x2_decode_30 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_29 & 0xFFu);
                uint32_t _f16x2_decode_30;
                float fp8_decode_1;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_30) : "h"(_e4m3x2_decode_30));
                uint16_t _f16_decode_30 = (uint16_t)_f16x2_decode_30;
                asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode_1) : "h"(_f16_decode_30));
                float fdiv_rn_1 = __fdiv_rn(1.0f, fp8_decode_1);
                qc[0] = qc[0] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
                qc[1] = qc[1] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
                qc[2] = qc[2] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
                qc[3] = qc[3] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
                qc[4] = qc[4] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
                qc[5] = qc[5] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
                qc[6] = qc[6] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
                qc[7] = qc[7] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw[0]) : "f"(qc[0]), "f"(qc[1]), "f"(qc[2]), "f"(qc[3]), "f"(qc[4]), "f"(qc[5]), "f"(qc[6]), "f"(qc[7]));
                qsf_a[0] = max_2_1 * 0.16666667f;
                if (row_end_1 > brow_1 + 32 + rgrp) {
                    int qrow_2 = (brow_1 + 32 + rgrp) * num_heads + (hk - num_heads - slot * num_heads);
                    *(reinterpret_cast<unsigned int*>(q4 + (qrow_2 * 16 + lane16)) + (0)) = qw[0];
                    if ((lane16 & 1) == 0) {
                        uint16_t _cvt_satfinite_e4m3x2_f32_31;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_31) : "f"(0.0f), "f"(qsf_a[0]));
                        *reinterpret_cast<unsigned char*>(q_sf + (qrow_2 * 8 + (lane16 >> 1))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_31 & 127));
                    }
                }
                for (int e_8 = 0; e_8 < 8; e_8++) {
                    float qf_3 = vec_load_3_1[e_8];
                    qc[e_8] = ((row_end_1 > brow_1 + 48 + rgrp) ? qf_3 - qmv[e_8] : 0.0f);
                }
                float max_1_2 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(qc[0]), fabsf(qc[1])), fabsf(qc[2])), fabsf(qc[3])), fabsf(qc[4])), fabsf(qc[5])), fabsf(qc[6])), fabsf(qc[7]));
                float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, max_1_2, 1);
                float max_2_2 = fmaxf(max_1_2, _shfl_xor_sync_32);
                uint16_t _cvt_satfinite_e4m3x2_f32_33;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_33) : "f"(0.0f), "f"(max_2_2 * 0.16666667f));
                uint16_t _e4m3x2_decode_34 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_33 & 0xFFu);
                uint32_t _f16x2_decode_34;
                float fp8_decode_2;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_34) : "h"(_e4m3x2_decode_34));
                uint16_t _f16_decode_34 = (uint16_t)_f16x2_decode_34;
                asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode_2) : "h"(_f16_decode_34));
                float fdiv_rn_2 = __fdiv_rn(1.0f, fp8_decode_2);
                qc[0] = qc[0] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                qc[1] = qc[1] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                qc[2] = qc[2] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                qc[3] = qc[3] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                qc[4] = qc[4] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                qc[5] = qc[5] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                qc[6] = qc[6] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                qc[7] = qc[7] * ((fp8_decode_2 != 0.0f) ? fdiv_rn_2 : 0.0f);
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw[0]) : "f"(qc[0]), "f"(qc[1]), "f"(qc[2]), "f"(qc[3]), "f"(qc[4]), "f"(qc[5]), "f"(qc[6]), "f"(qc[7]));
                qsf_a[0] = max_2_2 * 0.16666667f;
                if (row_end_1 > brow_1 + 48 + rgrp) {
                    int qrow_3 = (brow_1 + 48 + rgrp) * num_heads + (hk - num_heads - slot * num_heads);
                    *(reinterpret_cast<unsigned int*>(q4 + (qrow_3 * 16 + lane16)) + (0)) = qw[0];
                    if ((lane16 & 1) == 0) {
                        uint16_t _cvt_satfinite_e4m3x2_f32_35;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_35) : "f"(0.0f), "f"(qsf_a[0]));
                        *reinterpret_cast<unsigned char*>(q_sf + (qrow_3 * 8 + (lane16 >> 1))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_35 & 127));
                    }
                }
                for (int e_9 = 0; e_9 < 8; e_9++) {
                    float qf_4 = vec_load_4_1[e_9];
                    qc[e_9] = ((row_end_1 > brow_1 + 64 + rgrp) ? qf_4 - qmv[e_9] : 0.0f);
                }
                float max_1_3 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(qc[0]), fabsf(qc[1])), fabsf(qc[2])), fabsf(qc[3])), fabsf(qc[4])), fabsf(qc[5])), fabsf(qc[6])), fabsf(qc[7]));
                float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, max_1_3, 1);
                float max_2_3 = fmaxf(max_1_3, _shfl_xor_sync_36);
                uint16_t _cvt_satfinite_e4m3x2_f32_37;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_37) : "f"(0.0f), "f"(max_2_3 * 0.16666667f));
                uint16_t _e4m3x2_decode_38 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_37 & 0xFFu);
                uint32_t _f16x2_decode_38;
                float fp8_decode_3;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_38) : "h"(_e4m3x2_decode_38));
                uint16_t _f16_decode_38 = (uint16_t)_f16x2_decode_38;
                asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode_3) : "h"(_f16_decode_38));
                float fdiv_rn_3 = __fdiv_rn(1.0f, fp8_decode_3);
                qc[0] = qc[0] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                qc[1] = qc[1] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                qc[2] = qc[2] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                qc[3] = qc[3] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                qc[4] = qc[4] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                qc[5] = qc[5] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                qc[6] = qc[6] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                qc[7] = qc[7] * ((fp8_decode_3 != 0.0f) ? fdiv_rn_3 : 0.0f);
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw[0]) : "f"(qc[0]), "f"(qc[1]), "f"(qc[2]), "f"(qc[3]), "f"(qc[4]), "f"(qc[5]), "f"(qc[6]), "f"(qc[7]));
                qsf_a[0] = max_2_3 * 0.16666667f;
                if (row_end_1 > brow_1 + 64 + rgrp) {
                    int qrow_4 = (brow_1 + 64 + rgrp) * num_heads + (hk - num_heads - slot * num_heads);
                    *(reinterpret_cast<unsigned int*>(q4 + (qrow_4 * 16 + lane16)) + (0)) = qw[0];
                    if ((lane16 & 1) == 0) {
                        uint16_t _cvt_satfinite_e4m3x2_f32_39;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_39) : "f"(0.0f), "f"(qsf_a[0]));
                        *reinterpret_cast<unsigned char*>(q_sf + (qrow_4 * 8 + (lane16 >> 1))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_39 & 127));
                    }
                }
                for (int e_10 = 0; e_10 < 8; e_10++) {
                    float qf_5 = vec_load_5_1[e_10];
                    qc[e_10] = ((row_end_1 > brow_1 + 80 + rgrp) ? qf_5 - qmv[e_10] : 0.0f);
                }
                float max_1_4 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(qc[0]), fabsf(qc[1])), fabsf(qc[2])), fabsf(qc[3])), fabsf(qc[4])), fabsf(qc[5])), fabsf(qc[6])), fabsf(qc[7]));
                float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, max_1_4, 1);
                float max_2_4 = fmaxf(max_1_4, _shfl_xor_sync_40);
                uint16_t _cvt_satfinite_e4m3x2_f32_41;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_41) : "f"(0.0f), "f"(max_2_4 * 0.16666667f));
                uint16_t _e4m3x2_decode_42 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_41 & 0xFFu);
                uint32_t _f16x2_decode_42;
                float fp8_decode_4;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_42) : "h"(_e4m3x2_decode_42));
                uint16_t _f16_decode_42 = (uint16_t)_f16x2_decode_42;
                asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode_4) : "h"(_f16_decode_42));
                float fdiv_rn_4 = __fdiv_rn(1.0f, fp8_decode_4);
                qc[0] = qc[0] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                qc[1] = qc[1] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                qc[2] = qc[2] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                qc[3] = qc[3] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                qc[4] = qc[4] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                qc[5] = qc[5] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                qc[6] = qc[6] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                qc[7] = qc[7] * ((fp8_decode_4 != 0.0f) ? fdiv_rn_4 : 0.0f);
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw[0]) : "f"(qc[0]), "f"(qc[1]), "f"(qc[2]), "f"(qc[3]), "f"(qc[4]), "f"(qc[5]), "f"(qc[6]), "f"(qc[7]));
                qsf_a[0] = max_2_4 * 0.16666667f;
                if (row_end_1 > brow_1 + 80 + rgrp) {
                    int qrow_5 = (brow_1 + 80 + rgrp) * num_heads + (hk - num_heads - slot * num_heads);
                    *(reinterpret_cast<unsigned int*>(q4 + (qrow_5 * 16 + lane16)) + (0)) = qw[0];
                    if ((lane16 & 1) == 0) {
                        uint16_t _cvt_satfinite_e4m3x2_f32_43;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_43) : "f"(0.0f), "f"(qsf_a[0]));
                        *reinterpret_cast<unsigned char*>(q_sf + (qrow_5 * 8 + (lane16 >> 1))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_43 & 127));
                    }
                }
                for (int e_11 = 0; e_11 < 8; e_11++) {
                    float qf_6 = vec_load_6_1[e_11];
                    qc[e_11] = ((row_end_1 > brow_1 + 96 + rgrp) ? qf_6 - qmv[e_11] : 0.0f);
                }
                float max_1_5 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(qc[0]), fabsf(qc[1])), fabsf(qc[2])), fabsf(qc[3])), fabsf(qc[4])), fabsf(qc[5])), fabsf(qc[6])), fabsf(qc[7]));
                float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, max_1_5, 1);
                float max_2_5 = fmaxf(max_1_5, _shfl_xor_sync_44);
                uint16_t _cvt_satfinite_e4m3x2_f32_45;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_45) : "f"(0.0f), "f"(max_2_5 * 0.16666667f));
                uint16_t _e4m3x2_decode_46 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_45 & 0xFFu);
                uint32_t _f16x2_decode_46;
                float fp8_decode_5;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_46) : "h"(_e4m3x2_decode_46));
                uint16_t _f16_decode_46 = (uint16_t)_f16x2_decode_46;
                asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode_5) : "h"(_f16_decode_46));
                float fdiv_rn_5 = __fdiv_rn(1.0f, fp8_decode_5);
                qc[0] = qc[0] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                qc[1] = qc[1] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                qc[2] = qc[2] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                qc[3] = qc[3] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                qc[4] = qc[4] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                qc[5] = qc[5] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                qc[6] = qc[6] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                qc[7] = qc[7] * ((fp8_decode_5 != 0.0f) ? fdiv_rn_5 : 0.0f);
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw[0]) : "f"(qc[0]), "f"(qc[1]), "f"(qc[2]), "f"(qc[3]), "f"(qc[4]), "f"(qc[5]), "f"(qc[6]), "f"(qc[7]));
                qsf_a[0] = max_2_5 * 0.16666667f;
                if (row_end_1 > brow_1 + 96 + rgrp) {
                    int qrow_6 = (brow_1 + 96 + rgrp) * num_heads + (hk - num_heads - slot * num_heads);
                    *(reinterpret_cast<unsigned int*>(q4 + (qrow_6 * 16 + lane16)) + (0)) = qw[0];
                    if ((lane16 & 1) == 0) {
                        uint16_t _cvt_satfinite_e4m3x2_f32_47;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_47) : "f"(0.0f), "f"(qsf_a[0]));
                        *reinterpret_cast<unsigned char*>(q_sf + (qrow_6 * 8 + (lane16 >> 1))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_47 & 127));
                    }
                }
                for (int e_12 = 0; e_12 < 8; e_12++) {
                    float qf_7 = vec_load_7_1[e_12];
                    qc[e_12] = ((row_end_1 > brow_1 + 112 + rgrp) ? qf_7 - qmv[e_12] : 0.0f);
                }
                float _max_14 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(qc[0]), fabsf(qc[1])), fabsf(qc[2])), fabsf(qc[3])), fabsf(qc[4])), fabsf(qc[5])), fabsf(qc[6])), fabsf(qc[7]));
                float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, _max_14, 1);
                float _max_15 = fmaxf(_max_14, _shfl_xor_sync_48);
                uint16_t _cvt_satfinite_e4m3x2_f32_49;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_49) : "f"(0.0f), "f"(_max_15 * 0.16666667f));
                uint16_t _e4m3x2_decode_50 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_49 & 0xFFu);
                uint32_t _f16x2_decode_50;
                float _fp8_decode_7;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_50) : "h"(_e4m3x2_decode_50));
                uint16_t _f16_decode_50 = (uint16_t)_f16x2_decode_50;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_7) : "h"(_f16_decode_50));
                float _fdiv_rn_7 = __fdiv_rn(1.0f, _fp8_decode_7);
                qc[0] = qc[0] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                qc[1] = qc[1] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                qc[2] = qc[2] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                qc[3] = qc[3] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                qc[4] = qc[4] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                qc[5] = qc[5] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                qc[6] = qc[6] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                qc[7] = qc[7] * ((_fp8_decode_7 != 0.0f) ? _fdiv_rn_7 : 0.0f);
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw[0]) : "f"(qc[0]), "f"(qc[1]), "f"(qc[2]), "f"(qc[3]), "f"(qc[4]), "f"(qc[5]), "f"(qc[6]), "f"(qc[7]));
                qsf_a[0] = _max_15 * 0.16666667f;
                if (row_end_1 > brow_1 + 112 + rgrp) {
                    int qrow_7 = (brow_1 + 112 + rgrp) * num_heads + (hk - num_heads - slot * num_heads);
                    *(reinterpret_cast<unsigned int*>(q4 + (qrow_7 * 16 + lane16)) + (0)) = qw[0];
                    if ((lane16 & 1) == 0) {
                        uint16_t _cvt_satfinite_e4m3x2_f32_51;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_51) : "f"(0.0f), "f"(qsf_a[0]));
                        *reinterpret_cast<unsigned char*>(q_sf + (qrow_7 * 8 + (lane16 >> 1))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_51 & 127));
                    }
                }
                if (tid < 32) {
                    for (int e_13 = 0; e_13 < 8; e_13++) {
                        qc[e_13] = qmv[e_13];
                    }
                    float _max_16 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(qc[0]), fabsf(qc[1])), fabsf(qc[2])), fabsf(qc[3])), fabsf(qc[4])), fabsf(qc[5])), fabsf(qc[6])), fabsf(qc[7]));
                    float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, _max_16, 1);
                    float _max_17 = fmaxf(_max_16, _shfl_xor_sync_52);
                    uint16_t _cvt_satfinite_e4m3x2_f32_53;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_53) : "f"(0.0f), "f"(_max_17 * 0.16666667f));
                    uint16_t _e4m3x2_decode_54 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_53 & 0xFFu);
                    uint32_t _f16x2_decode_54;
                    float _fp8_decode_8;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_54) : "h"(_e4m3x2_decode_54));
                    uint16_t _f16_decode_54 = (uint16_t)_f16x2_decode_54;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_decode_8) : "h"(_f16_decode_54));
                    float _fdiv_rn_8 = __fdiv_rn(1.0f, _fp8_decode_8);
                    qc[0] = qc[0] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                    qc[1] = qc[1] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                    qc[2] = qc[2] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                    qc[3] = qc[3] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                    qc[4] = qc[4] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                    qc[5] = qc[5] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                    qc[6] = qc[6] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                    qc[7] = qc[7] * ((_fp8_decode_8 != 0.0f) ? _fdiv_rn_8 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw[0]) : "f"(qc[0]), "f"(qc[1]), "f"(qc[2]), "f"(qc[3]), "f"(qc[4]), "f"(qc[5]), "f"(qc[6]), "f"(qc[7]));
                    qsf_a[0] = _max_17 * 0.16666667f;
                    if (tid < 16) {
                        int qmrow = qt * num_heads + (hk - num_heads - slot * num_heads);
                        *(reinterpret_cast<unsigned int*>(qm4 + (qmrow * 16 + lane16)) + (0)) = qw[0];
                        if ((lane16 & 1) == 0) {
                            uint16_t _cvt_satfinite_e4m3x2_f32_55;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_55) : "f"(0.0f), "f"(qsf_a[0]));
                            *reinterpret_cast<unsigned char*>(qm_sf + (qmrow * 8 + (lane16 >> 1))) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_55 & 127));
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"
