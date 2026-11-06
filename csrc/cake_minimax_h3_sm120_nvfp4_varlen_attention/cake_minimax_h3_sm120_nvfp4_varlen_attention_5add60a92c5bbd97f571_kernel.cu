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
#define SMEM_TOTAL 16512
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 3

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_minimax_h3_sm120_nvfp4_varlen_attention_5add60a92c5bbd97f571(__nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V, int* __restrict__ tile_table, int* __restrict__ seg_tile_begin, int* __restrict__ tok_seg, int* __restrict__ cu_seqlens, float* __restrict__ partials, unsigned int* __restrict__ counters, float* __restrict__ mean_k, float* __restrict__ q_mean, unsigned int* __restrict__ q4, unsigned int* __restrict__ k4, uint8_t* __restrict__ q_sf, uint8_t* __restrict__ k_sf, unsigned int* __restrict__ qm4, uint8_t* __restrict__ qm_sf, unsigned int* __restrict__ vt4, uint8_t* __restrict__ v_sf, unsigned int* __restrict__ barrier, int num_tiles, int total_tokens, int total_padded, int num_heads, int num_kblocks)
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

    // === Task calls (declaration order) ===
    int col = (tid & 15) * 8;
    int rgrp = tid >> 4;
    int sub = tid & 7;
    int trow = tid >> 3;
    int col16 = sub * 16;
    int num_tblk = total_padded / 32;
    int vwarp = tid >> 5;
    int vlane = tid & 31;
    int vblk = vwarp & 1;
    int vdp = (vwarp >> 1) * 16 + (vlane & 15);
    int vhalf = vlane >> 4;
    asm volatile("griddepcontrol.wait;" ::: "memory");
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
        int seg = _vec_load_0[0];
        int row_begin = _vec_load_0[1];
        int row_end = _vec_load_0[2];
        int t0 = _vec_load_0[3];
        int qt_base = seg_tile_begin[seg] + (tile - t0);
        float qsum[8];
        float qf8[8];
        #pragma unroll 1
        for (int batch = 0; batch < (row_end - row_begin + 128 - 1) / 128; batch++) {
            for (int e = 0; e < 8; e++) {
                qsum[e] = 0.0f;
            }
            int brow = row_begin + batch * 128;
            float vec_load[8];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + rgrp) ? brow + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_1[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                        (&vec_load[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                    }
                }
            }
            float vec_load_1[8];
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 16 + rgrp) ? brow + 16 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_2[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_2[_blk] = _vptr_2[_blk];
                    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                        (&vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                    }
                }
            }
            float vec_load_2[8];
            {
                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 32 + rgrp) ? brow + 32 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_3[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_3[_blk] = _vptr_3[_blk];
                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                        (&vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                    }
                }
            }
            float vec_load_3[8];
            {
                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 48 + rgrp) ? brow + 48 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_4[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_4[_blk] = _vptr_4[_blk];
                    uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                        (&vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                    }
                }
            }
            float vec_load_4[8];
            {
                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 64 + rgrp) ? brow + 64 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_5[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_5[_blk] = _vptr_5[_blk];
                    uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                        (&vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                    }
                }
            }
            float vec_load_5[8];
            {
                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 80 + rgrp) ? brow + 80 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_6[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_6[_blk] = _vptr_6[_blk];
                    uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                        (&vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                    }
                }
            }
            float vec_load_6[8];
            {
                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 96 + rgrp) ? brow + 96 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_7[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_7[_blk] = _vptr_7[_blk];
                    uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                        (&vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                    }
                }
            }
            float vec_load_7[8];
            {
                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(Q + ((((row_end > brow + 112 + rgrp) ? brow + 112 + rgrp : row_end - 1) * num_heads + head) * 128 + col) + 0);
                uint4 _vld_8[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_8[_blk] = _vptr_8[_blk];
                    uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                        (&vec_load_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                    }
                }
            }
            qf8[0] = vec_load[0];
            qf8[1] = vec_load[1];
            qf8[2] = vec_load[2];
            qf8[3] = vec_load[3];
            qf8[4] = vec_load[4];
            qf8[5] = vec_load[5];
            qf8[6] = vec_load[6];
            qf8[7] = vec_load[7];
            qsum[0] = qsum[0] + ((row_end > brow + rgrp) ? qf8[0] : 0.0f);
            qsum[1] = qsum[1] + ((row_end > brow + rgrp) ? qf8[1] : 0.0f);
            qsum[2] = qsum[2] + ((row_end > brow + rgrp) ? qf8[2] : 0.0f);
            qsum[3] = qsum[3] + ((row_end > brow + rgrp) ? qf8[3] : 0.0f);
            qsum[4] = qsum[4] + ((row_end > brow + rgrp) ? qf8[4] : 0.0f);
            qsum[5] = qsum[5] + ((row_end > brow + rgrp) ? qf8[5] : 0.0f);
            qsum[6] = qsum[6] + ((row_end > brow + rgrp) ? qf8[6] : 0.0f);
            qsum[7] = qsum[7] + ((row_end > brow + rgrp) ? qf8[7] : 0.0f);
            qf8[0] = vec_load_1[0];
            qf8[1] = vec_load_1[1];
            qf8[2] = vec_load_1[2];
            qf8[3] = vec_load_1[3];
            qf8[4] = vec_load_1[4];
            qf8[5] = vec_load_1[5];
            qf8[6] = vec_load_1[6];
            qf8[7] = vec_load_1[7];
            qsum[0] = qsum[0] + ((row_end > brow + 16 + rgrp) ? qf8[0] : 0.0f);
            qsum[1] = qsum[1] + ((row_end > brow + 16 + rgrp) ? qf8[1] : 0.0f);
            qsum[2] = qsum[2] + ((row_end > brow + 16 + rgrp) ? qf8[2] : 0.0f);
            qsum[3] = qsum[3] + ((row_end > brow + 16 + rgrp) ? qf8[3] : 0.0f);
            qsum[4] = qsum[4] + ((row_end > brow + 16 + rgrp) ? qf8[4] : 0.0f);
            qsum[5] = qsum[5] + ((row_end > brow + 16 + rgrp) ? qf8[5] : 0.0f);
            qsum[6] = qsum[6] + ((row_end > brow + 16 + rgrp) ? qf8[6] : 0.0f);
            qsum[7] = qsum[7] + ((row_end > brow + 16 + rgrp) ? qf8[7] : 0.0f);
            qf8[0] = vec_load_2[0];
            qf8[1] = vec_load_2[1];
            qf8[2] = vec_load_2[2];
            qf8[3] = vec_load_2[3];
            qf8[4] = vec_load_2[4];
            qf8[5] = vec_load_2[5];
            qf8[6] = vec_load_2[6];
            qf8[7] = vec_load_2[7];
            qsum[0] = qsum[0] + ((row_end > brow + 32 + rgrp) ? qf8[0] : 0.0f);
            qsum[1] = qsum[1] + ((row_end > brow + 32 + rgrp) ? qf8[1] : 0.0f);
            qsum[2] = qsum[2] + ((row_end > brow + 32 + rgrp) ? qf8[2] : 0.0f);
            qsum[3] = qsum[3] + ((row_end > brow + 32 + rgrp) ? qf8[3] : 0.0f);
            qsum[4] = qsum[4] + ((row_end > brow + 32 + rgrp) ? qf8[4] : 0.0f);
            qsum[5] = qsum[5] + ((row_end > brow + 32 + rgrp) ? qf8[5] : 0.0f);
            qsum[6] = qsum[6] + ((row_end > brow + 32 + rgrp) ? qf8[6] : 0.0f);
            qsum[7] = qsum[7] + ((row_end > brow + 32 + rgrp) ? qf8[7] : 0.0f);
            qf8[0] = vec_load_3[0];
            qf8[1] = vec_load_3[1];
            qf8[2] = vec_load_3[2];
            qf8[3] = vec_load_3[3];
            qf8[4] = vec_load_3[4];
            qf8[5] = vec_load_3[5];
            qf8[6] = vec_load_3[6];
            qf8[7] = vec_load_3[7];
            qsum[0] = qsum[0] + ((row_end > brow + 48 + rgrp) ? qf8[0] : 0.0f);
            qsum[1] = qsum[1] + ((row_end > brow + 48 + rgrp) ? qf8[1] : 0.0f);
            qsum[2] = qsum[2] + ((row_end > brow + 48 + rgrp) ? qf8[2] : 0.0f);
            qsum[3] = qsum[3] + ((row_end > brow + 48 + rgrp) ? qf8[3] : 0.0f);
            qsum[4] = qsum[4] + ((row_end > brow + 48 + rgrp) ? qf8[4] : 0.0f);
            qsum[5] = qsum[5] + ((row_end > brow + 48 + rgrp) ? qf8[5] : 0.0f);
            qsum[6] = qsum[6] + ((row_end > brow + 48 + rgrp) ? qf8[6] : 0.0f);
            qsum[7] = qsum[7] + ((row_end > brow + 48 + rgrp) ? qf8[7] : 0.0f);
            qf8[0] = vec_load_4[0];
            qf8[1] = vec_load_4[1];
            qf8[2] = vec_load_4[2];
            qf8[3] = vec_load_4[3];
            qf8[4] = vec_load_4[4];
            qf8[5] = vec_load_4[5];
            qf8[6] = vec_load_4[6];
            qf8[7] = vec_load_4[7];
            qsum[0] = qsum[0] + ((row_end > brow + 64 + rgrp) ? qf8[0] : 0.0f);
            qsum[1] = qsum[1] + ((row_end > brow + 64 + rgrp) ? qf8[1] : 0.0f);
            qsum[2] = qsum[2] + ((row_end > brow + 64 + rgrp) ? qf8[2] : 0.0f);
            qsum[3] = qsum[3] + ((row_end > brow + 64 + rgrp) ? qf8[3] : 0.0f);
            qsum[4] = qsum[4] + ((row_end > brow + 64 + rgrp) ? qf8[4] : 0.0f);
            qsum[5] = qsum[5] + ((row_end > brow + 64 + rgrp) ? qf8[5] : 0.0f);
            qsum[6] = qsum[6] + ((row_end > brow + 64 + rgrp) ? qf8[6] : 0.0f);
            qsum[7] = qsum[7] + ((row_end > brow + 64 + rgrp) ? qf8[7] : 0.0f);
            qf8[0] = vec_load_5[0];
            qf8[1] = vec_load_5[1];
            qf8[2] = vec_load_5[2];
            qf8[3] = vec_load_5[3];
            qf8[4] = vec_load_5[4];
            qf8[5] = vec_load_5[5];
            qf8[6] = vec_load_5[6];
            qf8[7] = vec_load_5[7];
            qsum[0] = qsum[0] + ((row_end > brow + 80 + rgrp) ? qf8[0] : 0.0f);
            qsum[1] = qsum[1] + ((row_end > brow + 80 + rgrp) ? qf8[1] : 0.0f);
            qsum[2] = qsum[2] + ((row_end > brow + 80 + rgrp) ? qf8[2] : 0.0f);
            qsum[3] = qsum[3] + ((row_end > brow + 80 + rgrp) ? qf8[3] : 0.0f);
            qsum[4] = qsum[4] + ((row_end > brow + 80 + rgrp) ? qf8[4] : 0.0f);
            qsum[5] = qsum[5] + ((row_end > brow + 80 + rgrp) ? qf8[5] : 0.0f);
            qsum[6] = qsum[6] + ((row_end > brow + 80 + rgrp) ? qf8[6] : 0.0f);
            qsum[7] = qsum[7] + ((row_end > brow + 80 + rgrp) ? qf8[7] : 0.0f);
            qf8[0] = vec_load_6[0];
            qf8[1] = vec_load_6[1];
            qf8[2] = vec_load_6[2];
            qf8[3] = vec_load_6[3];
            qf8[4] = vec_load_6[4];
            qf8[5] = vec_load_6[5];
            qf8[6] = vec_load_6[6];
            qf8[7] = vec_load_6[7];
            qsum[0] = qsum[0] + ((row_end > brow + 96 + rgrp) ? qf8[0] : 0.0f);
            qsum[1] = qsum[1] + ((row_end > brow + 96 + rgrp) ? qf8[1] : 0.0f);
            qsum[2] = qsum[2] + ((row_end > brow + 96 + rgrp) ? qf8[2] : 0.0f);
            qsum[3] = qsum[3] + ((row_end > brow + 96 + rgrp) ? qf8[3] : 0.0f);
            qsum[4] = qsum[4] + ((row_end > brow + 96 + rgrp) ? qf8[4] : 0.0f);
            qsum[5] = qsum[5] + ((row_end > brow + 96 + rgrp) ? qf8[5] : 0.0f);
            qsum[6] = qsum[6] + ((row_end > brow + 96 + rgrp) ? qf8[6] : 0.0f);
            qsum[7] = qsum[7] + ((row_end > brow + 96 + rgrp) ? qf8[7] : 0.0f);
            qf8[0] = vec_load_7[0];
            qf8[1] = vec_load_7[1];
            qf8[2] = vec_load_7[2];
            qf8[3] = vec_load_7[3];
            qf8[4] = vec_load_7[4];
            qf8[5] = vec_load_7[5];
            qf8[6] = vec_load_7[6];
            qf8[7] = vec_load_7[7];
            qsum[0] = qsum[0] + ((row_end > brow + 112 + rgrp) ? qf8[0] : 0.0f);
            qsum[1] = qsum[1] + ((row_end > brow + 112 + rgrp) ? qf8[1] : 0.0f);
            qsum[2] = qsum[2] + ((row_end > brow + 112 + rgrp) ? qf8[2] : 0.0f);
            qsum[3] = qsum[3] + ((row_end > brow + 112 + rgrp) ? qf8[3] : 0.0f);
            qsum[4] = qsum[4] + ((row_end > brow + 112 + rgrp) ? qf8[4] : 0.0f);
            qsum[5] = qsum[5] + ((row_end > brow + 112 + rgrp) ? qf8[5] : 0.0f);
            qsum[6] = qsum[6] + ((row_end > brow + 112 + rgrp) ? qf8[6] : 0.0f);
            qsum[7] = qsum[7] + ((row_end > brow + 112 + rgrp) ? qf8[7] : 0.0f);
            for (int e_1 = 0; e_1 < 8; e_1++) {
                red_q[rgrp * 128 + col + e_1] = qsum[e_1];
            }
            __syncthreads();
            int qt = qt_base + batch;
            if (tid < 128) {
                float qtot = 0.0f;
                for (int g = 0; g < 16; g++) {
                    qtot = qtot + red_q[g * 128 + tid];
                }
                int brows = ((row_end - brow < 128) ? row_end - brow : 128);
                float _fdiv_9 = __fdiv_rn(qtot, (float)brows);
                *(reinterpret_cast<float*>(q_mean + ((qt * num_heads + head) * 128 + tid)) + (0)) = _fdiv_9;
            }
            __syncthreads();
            #pragma unroll 1
            for (int blk = 0; blk < 4; blk++) {
                int tok = brow + blk * 32 + trow;
                int valid = ((tok < row_end) ? 1 : 0);
                int tok_c = ((tok < row_end) ? tok : row_end - 1);
                int is_first = ((valid == 1 && tok == brow) ? 1 : 0);
                float vec_load_0[16];
                {
                    const uint4* _vptr_10 = reinterpret_cast<const uint4*>(Q + ((tok_c * num_heads + head) * 128 + col16) + 0);
                    uint4 _vld_10[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
                        _vld_10[_blk] = _vptr_10[_blk];
                        uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                            (&vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
                        }
                    }
                }
                int qmbase = (qt * num_heads + head) * 128 + col16;
                float qmv[16];
                float qc[16];
                for (int quad = 0; quad < 4; quad++) {
                    float _vec_load_10[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(q_mean + (qmbase + quad * 4) + 0);
                        _vec_load_10[0 + 0] = _v4.x;
                        _vec_load_10[0 + 1] = _v4.y;
                        _vec_load_10[0 + 2] = _v4.z;
                        _vec_load_10[0 + 3] = _v4.w;
                    }
                    for (int e_2 = 0; e_2 < 4; e_2++) {
                        qmv[quad * 4 + e_2] = _vec_load_10[e_2];
                        float qf = vec_load_0[quad * 4 + e_2];
                        qc[quad * 4 + e_2] = ((valid == 1) ? qf - _vec_load_10[e_2] : 0.0f);
                    }
                }
                unsigned int qw[2];
                float qsf_a[1];
                float max_1 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(qc[0]), fabsf(qc[1])), fabsf(qc[2])), fabsf(qc[3])), fabsf(qc[4])), fabsf(qc[5])), fabsf(qc[6])), fabsf(qc[7])), fabsf(qc[8])), fabsf(qc[9])), fabsf(qc[10])), fabsf(qc[11])), fabsf(qc[12])), fabsf(qc[13])), fabsf(qc[14])), fabsf(qc[15]));
                uint16_t _cvt_satfinite_e4m3x2_f32_12;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_12) : "f"(0.0f), "f"(max_1 * 0.16666667f));
                uint16_t _e4m3x2_decode_13 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_12 & 0xFFu);
                uint32_t _f16x2_decode_13;
                float fp8_decode;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_13) : "h"(_e4m3x2_decode_13));
                uint16_t _f16_decode_13 = (uint16_t)_f16x2_decode_13;
                asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode) : "h"(_f16_decode_13));
                float fdiv_rn = __fdiv_rn(1.0f, fp8_decode);
                qc[0] = qc[0] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[1] = qc[1] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[2] = qc[2] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[3] = qc[3] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[4] = qc[4] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[5] = qc[5] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[6] = qc[6] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[7] = qc[7] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[8] = qc[8] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[9] = qc[9] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[10] = qc[10] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[11] = qc[11] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[12] = qc[12] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[13] = qc[13] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[14] = qc[14] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                qc[15] = qc[15] * ((fp8_decode != 0.0f) ? fdiv_rn : 0.0f);
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw[0]) : "f"(qc[0]), "f"(qc[1]), "f"(qc[2]), "f"(qc[3]), "f"(qc[4]), "f"(qc[5]), "f"(qc[6]), "f"(qc[7]));
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw[1]) : "f"(qc[8]), "f"(qc[9]), "f"(qc[10]), "f"(qc[11]), "f"(qc[12]), "f"(qc[13]), "f"(qc[14]), "f"(qc[15]));
                qsf_a[0] = max_1 * 0.16666667f;
                unsigned int qstore[4];
                qstore[0] = qw[0];
                qstore[1] = qw[1];
                unsigned int _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, qw[0], 1);
                qstore[2] = _shfl_xor_sync_14;
                unsigned int _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, qw[1], 1);
                qstore[3] = _shfl_xor_sync_15;
                if (valid == 1) {
                    uint16_t _cvt_satfinite_e4m3x2_f32_16;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_16) : "f"(0.0f), "f"(qsf_a[0]));
                    *reinterpret_cast<unsigned char*>(q_sf + ((tok * num_heads + head) * 8 + sub)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_16 & 127));
                    if ((sub & 1) == 0) {
                        reinterpret_cast<int4*>(q4 + (((tok * num_heads + head) * 64 + sub * 8) / 4))[0] = reinterpret_cast<int4*>(qstore)[0];
                    }
                }
                if (is_first == 1) {
                    unsigned int mw[2];
                    float msf_a[1];
                    float max_1_0 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(qmv[0]), fabsf(qmv[1])), fabsf(qmv[2])), fabsf(qmv[3])), fabsf(qmv[4])), fabsf(qmv[5])), fabsf(qmv[6])), fabsf(qmv[7])), fabsf(qmv[8])), fabsf(qmv[9])), fabsf(qmv[10])), fabsf(qmv[11])), fabsf(qmv[12])), fabsf(qmv[13])), fabsf(qmv[14])), fabsf(qmv[15]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_17;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_17) : "f"(0.0f), "f"(max_1_0 * 0.16666667f));
                    uint16_t _e4m3x2_decode_18 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_17 & 0xFFu);
                    uint32_t _f16x2_decode_18;
                    float fp8_decode_0;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_18) : "h"(_e4m3x2_decode_18));
                    uint16_t _f16_decode_18 = (uint16_t)_f16x2_decode_18;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode_0) : "h"(_f16_decode_18));
                    float fdiv_rn_0 = __fdiv_rn(1.0f, fp8_decode_0);
                    qmv[0] = qmv[0] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                    qmv[1] = qmv[1] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                    qmv[2] = qmv[2] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                    qmv[3] = qmv[3] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                    qmv[4] = qmv[4] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                    qmv[5] = qmv[5] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                    qmv[6] = qmv[6] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                    qmv[7] = qmv[7] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                    qmv[8] = qmv[8] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                    qmv[9] = qmv[9] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                    qmv[10] = qmv[10] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                    qmv[11] = qmv[11] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                    qmv[12] = qmv[12] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                    qmv[13] = qmv[13] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                    qmv[14] = qmv[14] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                    qmv[15] = qmv[15] * ((fp8_decode_0 != 0.0f) ? fdiv_rn_0 : 0.0f);
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(mw[0]) : "f"(qmv[0]), "f"(qmv[1]), "f"(qmv[2]), "f"(qmv[3]), "f"(qmv[4]), "f"(qmv[5]), "f"(qmv[6]), "f"(qmv[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(mw[1]) : "f"(qmv[8]), "f"(qmv[9]), "f"(qmv[10]), "f"(qmv[11]), "f"(qmv[12]), "f"(qmv[13]), "f"(qmv[14]), "f"(qmv[15]));
                    msf_a[0] = max_1_0 * 0.16666667f;
                    int qmrow = qt * num_heads + head;
                    *(reinterpret_cast<unsigned int*>(qm4 + (qmrow * 16 + sub * 2)) + (0)) = mw[0];
                    *(reinterpret_cast<unsigned int*>(qm4 + (qmrow * 16 + sub * 2 + 1)) + (0)) = mw[1];
                    uint16_t _cvt_satfinite_e4m3x2_f32_19;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_19) : "f"(0.0f), "f"(msf_a[0]));
                    *reinterpret_cast<unsigned char*>(qm_sf + (qmrow * 8 + sub)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_19 & 127));
                }
            }
        }
    }
    int items_a = num_tiles * num_heads;
    int n_a = ((items_a < num_bids) ? items_a : num_bids);
    int b_first = bid;
    int b_step = num_bids;
    #pragma unroll 1
    for (int work_1 = b_first; work_1 < num_tblk * num_heads; work_1 += b_step) {
        int tblk = work_1 / num_heads;
        int head_1 = work_1 % num_heads;
        int vlive[8];
        vlive[0] = ((tblk * 32 + (vblk * 4 + vhalf * 2) < total_tokens) ? 1 : 0);
        float _vec_load_11[2];
        {
            uint32_t _bf16x2_bits_20;
            _bf16x2_bits_20 = *reinterpret_cast<const uint32_t*>(V + ((((tblk * 32 + (vblk * 4 + vhalf * 2) < total_tokens) ? tblk * 32 + (vblk * 4 + vhalf * 2) : total_tokens - 1) * num_heads + head_1) * 128 + vdp * 2) + 0);
            (&_vec_load_11[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_20) << 16);
            (&_vec_load_11[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_20) & 0xffff0000u);
        }
        vlive[1] = ((tblk * 32 + (vblk * 4 + vhalf * 2 + 1) < total_tokens) ? 1 : 0);
        float _vec_load_12[2];
        {
            uint32_t _bf16x2_bits_21;
            _bf16x2_bits_21 = *reinterpret_cast<const uint32_t*>(V + ((((tblk * 32 + (vblk * 4 + vhalf * 2 + 1) < total_tokens) ? tblk * 32 + (vblk * 4 + vhalf * 2 + 1) : total_tokens - 1) * num_heads + head_1) * 128 + vdp * 2) + 0);
            (&_vec_load_12[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_21) << 16);
            (&_vec_load_12[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_21) & 0xffff0000u);
        }
        vlive[2] = ((tblk * 32 + (8 + vblk * 4 + vhalf * 2) < total_tokens) ? 1 : 0);
        float _vec_load_13[2];
        {
            uint32_t _bf16x2_bits_22;
            _bf16x2_bits_22 = *reinterpret_cast<const uint32_t*>(V + ((((tblk * 32 + (8 + vblk * 4 + vhalf * 2) < total_tokens) ? tblk * 32 + (8 + vblk * 4 + vhalf * 2) : total_tokens - 1) * num_heads + head_1) * 128 + vdp * 2) + 0);
            (&_vec_load_13[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_22) << 16);
            (&_vec_load_13[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_22) & 0xffff0000u);
        }
        vlive[3] = ((tblk * 32 + (8 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? 1 : 0);
        float _vec_load_14[2];
        {
            uint32_t _bf16x2_bits_23;
            _bf16x2_bits_23 = *reinterpret_cast<const uint32_t*>(V + ((((tblk * 32 + (8 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? tblk * 32 + (8 + vblk * 4 + vhalf * 2 + 1) : total_tokens - 1) * num_heads + head_1) * 128 + vdp * 2) + 0);
            (&_vec_load_14[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_23) << 16);
            (&_vec_load_14[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_23) & 0xffff0000u);
        }
        vlive[4] = ((tblk * 32 + (16 + vblk * 4 + vhalf * 2) < total_tokens) ? 1 : 0);
        float _vec_load_15[2];
        {
            uint32_t _bf16x2_bits_24;
            _bf16x2_bits_24 = *reinterpret_cast<const uint32_t*>(V + ((((tblk * 32 + (16 + vblk * 4 + vhalf * 2) < total_tokens) ? tblk * 32 + (16 + vblk * 4 + vhalf * 2) : total_tokens - 1) * num_heads + head_1) * 128 + vdp * 2) + 0);
            (&_vec_load_15[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_24) << 16);
            (&_vec_load_15[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_24) & 0xffff0000u);
        }
        vlive[5] = ((tblk * 32 + (16 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? 1 : 0);
        float _vec_load_16[2];
        {
            uint32_t _bf16x2_bits_25;
            _bf16x2_bits_25 = *reinterpret_cast<const uint32_t*>(V + ((((tblk * 32 + (16 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? tblk * 32 + (16 + vblk * 4 + vhalf * 2 + 1) : total_tokens - 1) * num_heads + head_1) * 128 + vdp * 2) + 0);
            (&_vec_load_16[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_25) << 16);
            (&_vec_load_16[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_25) & 0xffff0000u);
        }
        vlive[6] = ((tblk * 32 + (24 + vblk * 4 + vhalf * 2) < total_tokens) ? 1 : 0);
        float _vec_load_17[2];
        {
            uint32_t _bf16x2_bits_26;
            _bf16x2_bits_26 = *reinterpret_cast<const uint32_t*>(V + ((((tblk * 32 + (24 + vblk * 4 + vhalf * 2) < total_tokens) ? tblk * 32 + (24 + vblk * 4 + vhalf * 2) : total_tokens - 1) * num_heads + head_1) * 128 + vdp * 2) + 0);
            (&_vec_load_17[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_26) << 16);
            (&_vec_load_17[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_26) & 0xffff0000u);
        }
        vlive[7] = ((tblk * 32 + (24 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? 1 : 0);
        float _vec_load_18[2];
        {
            uint32_t _bf16x2_bits_27;
            _bf16x2_bits_27 = *reinterpret_cast<const uint32_t*>(V + ((((tblk * 32 + (24 + vblk * 4 + vhalf * 2 + 1) < total_tokens) ? tblk * 32 + (24 + vblk * 4 + vhalf * 2 + 1) : total_tokens - 1) * num_heads + head_1) * 128 + vdp * 2) + 0);
            (&_vec_load_18[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_27) << 16);
            (&_vec_load_18[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_27) & 0xffff0000u);
        }
        float va[8];
        float vb[8];
        va[0] = _vec_load_11[0];
        vb[0] = _vec_load_11[1];
        va[1] = _vec_load_12[0];
        vb[1] = _vec_load_12[1];
        va[2] = _vec_load_13[0];
        vb[2] = _vec_load_13[1];
        va[3] = _vec_load_14[0];
        vb[3] = _vec_load_14[1];
        va[4] = _vec_load_15[0];
        vb[4] = _vec_load_15[1];
        va[5] = _vec_load_16[0];
        vb[5] = _vec_load_16[1];
        va[6] = _vec_load_17[0];
        vb[6] = _vec_load_17[1];
        va[7] = _vec_load_18[0];
        vb[7] = _vec_load_18[1];
        va[0] = ((vlive[0] == 1) ? va[0] : 0.0f);
        vb[0] = ((vlive[0] == 1) ? vb[0] : 0.0f);
        va[1] = ((vlive[1] == 1) ? va[1] : 0.0f);
        vb[1] = ((vlive[1] == 1) ? vb[1] : 0.0f);
        va[2] = ((vlive[2] == 1) ? va[2] : 0.0f);
        vb[2] = ((vlive[2] == 1) ? vb[2] : 0.0f);
        va[3] = ((vlive[3] == 1) ? va[3] : 0.0f);
        vb[3] = ((vlive[3] == 1) ? vb[3] : 0.0f);
        va[4] = ((vlive[4] == 1) ? va[4] : 0.0f);
        vb[4] = ((vlive[4] == 1) ? vb[4] : 0.0f);
        va[5] = ((vlive[5] == 1) ? va[5] : 0.0f);
        vb[5] = ((vlive[5] == 1) ? vb[5] : 0.0f);
        va[6] = ((vlive[6] == 1) ? va[6] : 0.0f);
        vb[6] = ((vlive[6] == 1) ? vb[6] : 0.0f);
        va[7] = ((vlive[7] == 1) ? va[7] : 0.0f);
        vb[7] = ((vlive[7] == 1) ? vb[7] : 0.0f);
        float amax_a = 0.0f;
        float amax_b = 0.0f;
        for (int ii = 0; ii < 8; ii++) {
            amax_a = fmaxf(amax_a, fabsf(va[ii]));
            amax_b = fmaxf(amax_b, fabsf(vb[ii]));
        }
        float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, amax_a, 16);
        amax_a = fmaxf(amax_a, _shfl_xor_sync_28);
        float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, amax_b, 16);
        amax_b = fmaxf(amax_b, _shfl_xor_sync_29);
        float vsf_a = amax_a * 0.16666667f;
        float vsf_b = amax_b * 0.16666667f;
        uint16_t _e4m3x2_f32_0;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(vsf_b), "f"(vsf_a));
        uint16_t _e4m3x2_decode_30 = (uint16_t)((unsigned int)_e4m3x2_f32_0 & 0xFFu);
        uint32_t _f16x2_decode_30;
        float vdec_a;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_30) : "h"(_e4m3x2_decode_30));
        uint16_t _f16_decode_30 = (uint16_t)_f16x2_decode_30;
        asm("cvt.f32.f16 %0, %1;" : "=f"(vdec_a) : "h"(_f16_decode_30));
        uint16_t _e4m3x2_decode_31 = (uint16_t)((unsigned int)_e4m3x2_f32_0 >> 8 & 0xFFu);
        uint32_t _f16x2_decode_31;
        float vdec_b;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_31) : "h"(_e4m3x2_decode_31));
        uint16_t _f16_decode_31 = (uint16_t)_f16x2_decode_31;
        asm("cvt.f32.f16 %0, %1;" : "=f"(vdec_b) : "h"(_f16_decode_31));
        float _fdiv_32 = __fdiv_rn(1.0f, vdec_a);
        float vinv_a = ((vdec_a != 0.0f) ? _fdiv_32 : 0.0f);
        float _fdiv_33 = __fdiv_rn(1.0f, vdec_b);
        float vinv_b = ((vdec_b != 0.0f) ? _fdiv_33 : 0.0f);
        for (int ii_1 = 0; ii_1 < 8; ii_1++) {
            va[ii_1] = va[ii_1] * vinv_a;
            vb[ii_1] = vb[ii_1] * vinv_b;
        }
        unsigned int vwa[1];
        unsigned int vwb[1];
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(vwa[0]) : "f"(va[0]), "f"(va[1]), "f"(va[2]), "f"(va[3]), "f"(va[4]), "f"(va[5]), "f"(va[6]), "f"(va[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(vwb[0]) : "f"(vb[0]), "f"(vb[1]), "f"(vb[2]), "f"(vb[3]), "f"(vb[4]), "f"(vb[5]), "f"(vb[6]), "f"(vb[7]));
        int row_words = total_padded >> 3;
        int vt_word = (head_1 * 128 + vdp * 2) * row_words + (tblk * 32 >> 3) + vblk * 2 + vhalf;
        *(reinterpret_cast<unsigned int*>(vt4 + vt_word) + (0)) = vwa[0];
        *(reinterpret_cast<unsigned int*>(vt4 + (vt_word + row_words)) + (0)) = vwb[0];
        if (vhalf == 0) {
            int vsf_base = ((head_1 * num_kblocks + (tblk * 32 >> 7)) * 128 + vdp * 2) * 8 + (((tblk * 32 & 127) >> 4) + vblk);
            uint16_t _cvt_satfinite_e4m3x2_f32_34;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_34) : "f"(0.0f), "f"(vsf_a));
            *reinterpret_cast<unsigned char*>(v_sf + vsf_base) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_34 & 127));
            uint16_t _cvt_satfinite_e4m3x2_f32_35;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_35) : "f"(0.0f), "f"(vsf_b));
            *reinterpret_cast<unsigned char*>(v_sf + (vsf_base + 8)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_35 & 127));
        }
    }
    #pragma unroll 1
    for (int work_2 = bid; work_2 < num_tiles * num_heads; work_2 += num_bids) {
        int tile_1 = work_2 / num_heads;
        int head_2 = work_2 % num_heads;
        int _vec_load_19[4];
        {
            const int4* _ivptr_36 = reinterpret_cast<const int4*>(tile_table + (8 * tile_1) + 0);
            int4 _ivld_36;
            _ivld_36 = *_ivptr_36;
            _vec_load_19[0 + 0] = _ivld_36.x;
            _vec_load_19[0 + 1] = _ivld_36.y;
            _vec_load_19[0 + 2] = _ivld_36.z;
            _vec_load_19[0 + 3] = _ivld_36.w;
        }
        int _vec_load_20[4];
        {
            const int4* _ivptr_37 = reinterpret_cast<const int4*>(tile_table + (8 * tile_1 + 4) + 0);
            int4 _ivld_37;
            _ivld_37 = *_ivptr_37;
            _vec_load_20[0 + 0] = _ivld_37.x;
            _vec_load_20[0 + 1] = _ivld_37.y;
            _vec_load_20[0 + 2] = _ivld_37.z;
            _vec_load_20[0 + 3] = _ivld_37.w;
        }
        int seg_1 = _vec_load_19[0];
        int row_begin_1 = _vec_load_19[1];
        int row_end_1 = _vec_load_19[2];
        int t0_1 = _vec_load_19[3];
        int t1 = _vec_load_20[0];
        int length = _vec_load_20[1];
        float ksum[8];
        for (int e_3 = 0; e_3 < 8; e_3++) {
            ksum[e_3] = 0.0f;
        }
        float kf8[8];
        #pragma unroll 1
        for (int batch_1 = 0; batch_1 < (row_end_1 - row_begin_1 + 128 - 1) / 128; batch_1++) {
            int brow_1 = row_begin_1 + batch_1 * 128;
            float vec_load_8[8];
            {
                const uint4* _vptr_38 = reinterpret_cast<const uint4*>(K + ((((row_end_1 > brow_1 + rgrp) ? brow_1 + rgrp : row_end_1 - 1) * num_heads + head_2) * 128 + col) + 0);
                uint4 _vld_38[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_38[_blk] = _vptr_38[_blk];
                    uint32_t* _vpairs_38 = reinterpret_cast<uint32_t*>(&_vld_38[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_38[_pair]) << 16);
                        (&vec_load_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_38[_pair]) & 0xffff0000u);
                    }
                }
            }
            float vec_load_1_1[8];
            {
                const uint4* _vptr_39 = reinterpret_cast<const uint4*>(K + ((((row_end_1 > brow_1 + 16 + rgrp) ? brow_1 + 16 + rgrp : row_end_1 - 1) * num_heads + head_2) * 128 + col) + 0);
                uint4 _vld_39[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_39[_blk] = _vptr_39[_blk];
                    uint32_t* _vpairs_39 = reinterpret_cast<uint32_t*>(&_vld_39[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_1_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_39[_pair]) << 16);
                        (&vec_load_1_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_39[_pair]) & 0xffff0000u);
                    }
                }
            }
            float vec_load_2_1[8];
            {
                const uint4* _vptr_40 = reinterpret_cast<const uint4*>(K + ((((row_end_1 > brow_1 + 32 + rgrp) ? brow_1 + 32 + rgrp : row_end_1 - 1) * num_heads + head_2) * 128 + col) + 0);
                uint4 _vld_40[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_40[_blk] = _vptr_40[_blk];
                    uint32_t* _vpairs_40 = reinterpret_cast<uint32_t*>(&_vld_40[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_2_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_40[_pair]) << 16);
                        (&vec_load_2_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_40[_pair]) & 0xffff0000u);
                    }
                }
            }
            float vec_load_3_1[8];
            {
                const uint4* _vptr_41 = reinterpret_cast<const uint4*>(K + ((((row_end_1 > brow_1 + 48 + rgrp) ? brow_1 + 48 + rgrp : row_end_1 - 1) * num_heads + head_2) * 128 + col) + 0);
                uint4 _vld_41[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_41[_blk] = _vptr_41[_blk];
                    uint32_t* _vpairs_41 = reinterpret_cast<uint32_t*>(&_vld_41[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_3_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) << 16);
                        (&vec_load_3_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) & 0xffff0000u);
                    }
                }
            }
            float vec_load_4_1[8];
            {
                const uint4* _vptr_42 = reinterpret_cast<const uint4*>(K + ((((row_end_1 > brow_1 + 64 + rgrp) ? brow_1 + 64 + rgrp : row_end_1 - 1) * num_heads + head_2) * 128 + col) + 0);
                uint4 _vld_42[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_42[_blk] = _vptr_42[_blk];
                    uint32_t* _vpairs_42 = reinterpret_cast<uint32_t*>(&_vld_42[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_4_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) << 16);
                        (&vec_load_4_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_42[_pair]) & 0xffff0000u);
                    }
                }
            }
            float vec_load_5_1[8];
            {
                const uint4* _vptr_43 = reinterpret_cast<const uint4*>(K + ((((row_end_1 > brow_1 + 80 + rgrp) ? brow_1 + 80 + rgrp : row_end_1 - 1) * num_heads + head_2) * 128 + col) + 0);
                uint4 _vld_43[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_43[_blk] = _vptr_43[_blk];
                    uint32_t* _vpairs_43 = reinterpret_cast<uint32_t*>(&_vld_43[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_5_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) << 16);
                        (&vec_load_5_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) & 0xffff0000u);
                    }
                }
            }
            float vec_load_6_1[8];
            {
                const uint4* _vptr_44 = reinterpret_cast<const uint4*>(K + ((((row_end_1 > brow_1 + 96 + rgrp) ? brow_1 + 96 + rgrp : row_end_1 - 1) * num_heads + head_2) * 128 + col) + 0);
                uint4 _vld_44[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_44[_blk] = _vptr_44[_blk];
                    uint32_t* _vpairs_44 = reinterpret_cast<uint32_t*>(&_vld_44[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_6_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_44[_pair]) << 16);
                        (&vec_load_6_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_44[_pair]) & 0xffff0000u);
                    }
                }
            }
            float vec_load_7_1[8];
            {
                const uint4* _vptr_45 = reinterpret_cast<const uint4*>(K + ((((row_end_1 > brow_1 + 112 + rgrp) ? brow_1 + 112 + rgrp : row_end_1 - 1) * num_heads + head_2) * 128 + col) + 0);
                uint4 _vld_45[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_45[_blk] = _vptr_45[_blk];
                    uint32_t* _vpairs_45 = reinterpret_cast<uint32_t*>(&_vld_45[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&vec_load_7_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_45[_pair]) << 16);
                        (&vec_load_7_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_45[_pair]) & 0xffff0000u);
                    }
                }
            }
            kf8[0] = vec_load_8[0];
            kf8[1] = vec_load_8[1];
            kf8[2] = vec_load_8[2];
            kf8[3] = vec_load_8[3];
            kf8[4] = vec_load_8[4];
            kf8[5] = vec_load_8[5];
            kf8[6] = vec_load_8[6];
            kf8[7] = vec_load_8[7];
            ksum[0] = ksum[0] + ((row_end_1 > brow_1 + rgrp) ? kf8[0] : 0.0f);
            ksum[1] = ksum[1] + ((row_end_1 > brow_1 + rgrp) ? kf8[1] : 0.0f);
            ksum[2] = ksum[2] + ((row_end_1 > brow_1 + rgrp) ? kf8[2] : 0.0f);
            ksum[3] = ksum[3] + ((row_end_1 > brow_1 + rgrp) ? kf8[3] : 0.0f);
            ksum[4] = ksum[4] + ((row_end_1 > brow_1 + rgrp) ? kf8[4] : 0.0f);
            ksum[5] = ksum[5] + ((row_end_1 > brow_1 + rgrp) ? kf8[5] : 0.0f);
            ksum[6] = ksum[6] + ((row_end_1 > brow_1 + rgrp) ? kf8[6] : 0.0f);
            ksum[7] = ksum[7] + ((row_end_1 > brow_1 + rgrp) ? kf8[7] : 0.0f);
            kf8[0] = vec_load_1_1[0];
            kf8[1] = vec_load_1_1[1];
            kf8[2] = vec_load_1_1[2];
            kf8[3] = vec_load_1_1[3];
            kf8[4] = vec_load_1_1[4];
            kf8[5] = vec_load_1_1[5];
            kf8[6] = vec_load_1_1[6];
            kf8[7] = vec_load_1_1[7];
            ksum[0] = ksum[0] + ((row_end_1 > brow_1 + 16 + rgrp) ? kf8[0] : 0.0f);
            ksum[1] = ksum[1] + ((row_end_1 > brow_1 + 16 + rgrp) ? kf8[1] : 0.0f);
            ksum[2] = ksum[2] + ((row_end_1 > brow_1 + 16 + rgrp) ? kf8[2] : 0.0f);
            ksum[3] = ksum[3] + ((row_end_1 > brow_1 + 16 + rgrp) ? kf8[3] : 0.0f);
            ksum[4] = ksum[4] + ((row_end_1 > brow_1 + 16 + rgrp) ? kf8[4] : 0.0f);
            ksum[5] = ksum[5] + ((row_end_1 > brow_1 + 16 + rgrp) ? kf8[5] : 0.0f);
            ksum[6] = ksum[6] + ((row_end_1 > brow_1 + 16 + rgrp) ? kf8[6] : 0.0f);
            ksum[7] = ksum[7] + ((row_end_1 > brow_1 + 16 + rgrp) ? kf8[7] : 0.0f);
            kf8[0] = vec_load_2_1[0];
            kf8[1] = vec_load_2_1[1];
            kf8[2] = vec_load_2_1[2];
            kf8[3] = vec_load_2_1[3];
            kf8[4] = vec_load_2_1[4];
            kf8[5] = vec_load_2_1[5];
            kf8[6] = vec_load_2_1[6];
            kf8[7] = vec_load_2_1[7];
            ksum[0] = ksum[0] + ((row_end_1 > brow_1 + 32 + rgrp) ? kf8[0] : 0.0f);
            ksum[1] = ksum[1] + ((row_end_1 > brow_1 + 32 + rgrp) ? kf8[1] : 0.0f);
            ksum[2] = ksum[2] + ((row_end_1 > brow_1 + 32 + rgrp) ? kf8[2] : 0.0f);
            ksum[3] = ksum[3] + ((row_end_1 > brow_1 + 32 + rgrp) ? kf8[3] : 0.0f);
            ksum[4] = ksum[4] + ((row_end_1 > brow_1 + 32 + rgrp) ? kf8[4] : 0.0f);
            ksum[5] = ksum[5] + ((row_end_1 > brow_1 + 32 + rgrp) ? kf8[5] : 0.0f);
            ksum[6] = ksum[6] + ((row_end_1 > brow_1 + 32 + rgrp) ? kf8[6] : 0.0f);
            ksum[7] = ksum[7] + ((row_end_1 > brow_1 + 32 + rgrp) ? kf8[7] : 0.0f);
            kf8[0] = vec_load_3_1[0];
            kf8[1] = vec_load_3_1[1];
            kf8[2] = vec_load_3_1[2];
            kf8[3] = vec_load_3_1[3];
            kf8[4] = vec_load_3_1[4];
            kf8[5] = vec_load_3_1[5];
            kf8[6] = vec_load_3_1[6];
            kf8[7] = vec_load_3_1[7];
            ksum[0] = ksum[0] + ((row_end_1 > brow_1 + 48 + rgrp) ? kf8[0] : 0.0f);
            ksum[1] = ksum[1] + ((row_end_1 > brow_1 + 48 + rgrp) ? kf8[1] : 0.0f);
            ksum[2] = ksum[2] + ((row_end_1 > brow_1 + 48 + rgrp) ? kf8[2] : 0.0f);
            ksum[3] = ksum[3] + ((row_end_1 > brow_1 + 48 + rgrp) ? kf8[3] : 0.0f);
            ksum[4] = ksum[4] + ((row_end_1 > brow_1 + 48 + rgrp) ? kf8[4] : 0.0f);
            ksum[5] = ksum[5] + ((row_end_1 > brow_1 + 48 + rgrp) ? kf8[5] : 0.0f);
            ksum[6] = ksum[6] + ((row_end_1 > brow_1 + 48 + rgrp) ? kf8[6] : 0.0f);
            ksum[7] = ksum[7] + ((row_end_1 > brow_1 + 48 + rgrp) ? kf8[7] : 0.0f);
            kf8[0] = vec_load_4_1[0];
            kf8[1] = vec_load_4_1[1];
            kf8[2] = vec_load_4_1[2];
            kf8[3] = vec_load_4_1[3];
            kf8[4] = vec_load_4_1[4];
            kf8[5] = vec_load_4_1[5];
            kf8[6] = vec_load_4_1[6];
            kf8[7] = vec_load_4_1[7];
            ksum[0] = ksum[0] + ((row_end_1 > brow_1 + 64 + rgrp) ? kf8[0] : 0.0f);
            ksum[1] = ksum[1] + ((row_end_1 > brow_1 + 64 + rgrp) ? kf8[1] : 0.0f);
            ksum[2] = ksum[2] + ((row_end_1 > brow_1 + 64 + rgrp) ? kf8[2] : 0.0f);
            ksum[3] = ksum[3] + ((row_end_1 > brow_1 + 64 + rgrp) ? kf8[3] : 0.0f);
            ksum[4] = ksum[4] + ((row_end_1 > brow_1 + 64 + rgrp) ? kf8[4] : 0.0f);
            ksum[5] = ksum[5] + ((row_end_1 > brow_1 + 64 + rgrp) ? kf8[5] : 0.0f);
            ksum[6] = ksum[6] + ((row_end_1 > brow_1 + 64 + rgrp) ? kf8[6] : 0.0f);
            ksum[7] = ksum[7] + ((row_end_1 > brow_1 + 64 + rgrp) ? kf8[7] : 0.0f);
            kf8[0] = vec_load_5_1[0];
            kf8[1] = vec_load_5_1[1];
            kf8[2] = vec_load_5_1[2];
            kf8[3] = vec_load_5_1[3];
            kf8[4] = vec_load_5_1[4];
            kf8[5] = vec_load_5_1[5];
            kf8[6] = vec_load_5_1[6];
            kf8[7] = vec_load_5_1[7];
            ksum[0] = ksum[0] + ((row_end_1 > brow_1 + 80 + rgrp) ? kf8[0] : 0.0f);
            ksum[1] = ksum[1] + ((row_end_1 > brow_1 + 80 + rgrp) ? kf8[1] : 0.0f);
            ksum[2] = ksum[2] + ((row_end_1 > brow_1 + 80 + rgrp) ? kf8[2] : 0.0f);
            ksum[3] = ksum[3] + ((row_end_1 > brow_1 + 80 + rgrp) ? kf8[3] : 0.0f);
            ksum[4] = ksum[4] + ((row_end_1 > brow_1 + 80 + rgrp) ? kf8[4] : 0.0f);
            ksum[5] = ksum[5] + ((row_end_1 > brow_1 + 80 + rgrp) ? kf8[5] : 0.0f);
            ksum[6] = ksum[6] + ((row_end_1 > brow_1 + 80 + rgrp) ? kf8[6] : 0.0f);
            ksum[7] = ksum[7] + ((row_end_1 > brow_1 + 80 + rgrp) ? kf8[7] : 0.0f);
            kf8[0] = vec_load_6_1[0];
            kf8[1] = vec_load_6_1[1];
            kf8[2] = vec_load_6_1[2];
            kf8[3] = vec_load_6_1[3];
            kf8[4] = vec_load_6_1[4];
            kf8[5] = vec_load_6_1[5];
            kf8[6] = vec_load_6_1[6];
            kf8[7] = vec_load_6_1[7];
            ksum[0] = ksum[0] + ((row_end_1 > brow_1 + 96 + rgrp) ? kf8[0] : 0.0f);
            ksum[1] = ksum[1] + ((row_end_1 > brow_1 + 96 + rgrp) ? kf8[1] : 0.0f);
            ksum[2] = ksum[2] + ((row_end_1 > brow_1 + 96 + rgrp) ? kf8[2] : 0.0f);
            ksum[3] = ksum[3] + ((row_end_1 > brow_1 + 96 + rgrp) ? kf8[3] : 0.0f);
            ksum[4] = ksum[4] + ((row_end_1 > brow_1 + 96 + rgrp) ? kf8[4] : 0.0f);
            ksum[5] = ksum[5] + ((row_end_1 > brow_1 + 96 + rgrp) ? kf8[5] : 0.0f);
            ksum[6] = ksum[6] + ((row_end_1 > brow_1 + 96 + rgrp) ? kf8[6] : 0.0f);
            ksum[7] = ksum[7] + ((row_end_1 > brow_1 + 96 + rgrp) ? kf8[7] : 0.0f);
            kf8[0] = vec_load_7_1[0];
            kf8[1] = vec_load_7_1[1];
            kf8[2] = vec_load_7_1[2];
            kf8[3] = vec_load_7_1[3];
            kf8[4] = vec_load_7_1[4];
            kf8[5] = vec_load_7_1[5];
            kf8[6] = vec_load_7_1[6];
            kf8[7] = vec_load_7_1[7];
            ksum[0] = ksum[0] + ((row_end_1 > brow_1 + 112 + rgrp) ? kf8[0] : 0.0f);
            ksum[1] = ksum[1] + ((row_end_1 > brow_1 + 112 + rgrp) ? kf8[1] : 0.0f);
            ksum[2] = ksum[2] + ((row_end_1 > brow_1 + 112 + rgrp) ? kf8[2] : 0.0f);
            ksum[3] = ksum[3] + ((row_end_1 > brow_1 + 112 + rgrp) ? kf8[3] : 0.0f);
            ksum[4] = ksum[4] + ((row_end_1 > brow_1 + 112 + rgrp) ? kf8[4] : 0.0f);
            ksum[5] = ksum[5] + ((row_end_1 > brow_1 + 112 + rgrp) ? kf8[5] : 0.0f);
            ksum[6] = ksum[6] + ((row_end_1 > brow_1 + 112 + rgrp) ? kf8[6] : 0.0f);
            ksum[7] = ksum[7] + ((row_end_1 > brow_1 + 112 + rgrp) ? kf8[7] : 0.0f);
        }
        for (int e_4 = 0; e_4 < 8; e_4++) {
            red_sum[rgrp * 128 + col + e_4] = ksum[e_4];
        }
        __syncthreads();
        int direct = ((t1 - t0_1 == 1) ? 1 : 0);
        if (direct == 0) {
            if (tid < 128) {
                float total = 0.0f;
                for (int g_1 = 0; g_1 < 16; g_1++) {
                    total = total + red_sum[g_1 * 128 + tid];
                }
                *(reinterpret_cast<float*>(partials + (work_2 * 128 + tid)) + (0)) = total;
            }
            __syncthreads();
            if (tid == 0) {
                unsigned int old;
                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(old) : "l"(&counters[seg_1 * num_heads + head_2]), "r"(static_cast<uint32_t>(1)) : "memory");
                int last = (((int)old == t1 - t0_1 - 1) ? 1 : 0);
                flag[0] = last;
                if (last == 1) {
                    *(reinterpret_cast<unsigned int*>(counters + (seg_1 * num_heads + head_2)) + (0)) = (unsigned int)0;
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
                    for (int g_2 = 0; g_2 < 16; g_2++) {
                        acc_a[0] = acc_a[0] + red_sum[g_2 * 128 + tid];
                    }
                } else {
                    #pragma unroll 1
                    for (int tt = t0_1; tt < t1; tt += 4) {
                        for (int j = 0; j < 4; j++) {
                            int ttj = ((t1 > tt + j) ? tt + j : t1 - 1);
                            fold_v[j] = partials[(ttj * num_heads + head_2) * 128 + tid];
                        }
                        for (int j_1 = 0; j_1 < 4; j_1++) {
                            float fv = ((t1 > tt + j_1) ? fold_v[j_1] : 0.0f);
                            acc_a[0] = acc_a[0] + fv;
                        }
                    }
                }
                if (length > 0) {
                    float lenf = (float)length;
                    float _fdiv_46 = __fdiv_rn(acc_a[0], lenf);
                    *(reinterpret_cast<float*>(mean_k + ((seg_1 * num_heads + head_2) * 128 + tid)) + (0)) = _fdiv_46;
                }
            }
        }
        __syncthreads();
    }
    int items_d = num_tblk * num_heads;
    int d_first = bid;
    __syncthreads();
    if (tid == 0) {
        unsigned int gridu = (unsigned int)num_bids;
        unsigned int increment = 1;
        if (bid == 0) {
            increment = 2147483649u - gridu;
        }
        unsigned int old_1;
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
            : "=r"(old_1) : "l"(&barrier[0]), "r"(static_cast<uint32_t>(increment)) : "memory");
        unsigned int _wait_acquire_mask_0;
        do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_wait_acquire_mask_0) : "l"((reinterpret_cast<unsigned int*>(barrier) + (0))) : "memory");
        } while (((_wait_acquire_mask_0 ^ static_cast<unsigned int>((old_1 ^ 2147483648u) & 2147483648u)) & static_cast<unsigned int>(2147483648)) != 0);
    }
    __syncthreads();
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll 1
    for (int work_3 = d_first; work_3 < items_d; work_3 += num_bids) {
        int tblk_1 = work_3 / num_heads;
        int head_3 = work_3 % num_heads;
        int tok_1 = tblk_1 * 32 + trow;
        int valid_1 = ((tok_1 < total_tokens) ? 1 : 0);
        int tok_c_1 = ((tok_1 < total_tokens) ? tok_1 : total_tokens - 1);
        float vec_load_9[16];
        {
            const uint4* _vptr_47 = reinterpret_cast<const uint4*>(K + ((tok_c_1 * num_heads + head_3) * 128 + col16) + 0);
            uint4 _vld_47[2];
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
                _vld_47[_blk] = _vptr_47[_blk];
                uint32_t* _vpairs_47 = reinterpret_cast<uint32_t*>(&_vld_47[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&vec_load_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_47[_pair]) << 16);
                    (&vec_load_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_47[_pair]) & 0xffff0000u);
                }
            }
        }
        int seg_2 = tok_seg[tok_c_1];
        int mbase = (seg_2 * num_heads + head_3) * 128 + col16;
        float kc[16];
        for (int quad_1 = 0; quad_1 < 4; quad_1++) {
            float vec_load_1_2[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(mean_k + (mbase + quad_1 * 4) + 0);
                vec_load_1_2[0 + 0] = _v4.x;
                vec_load_1_2[0 + 1] = _v4.y;
                vec_load_1_2[0 + 2] = _v4.z;
                vec_load_1_2[0 + 3] = _v4.w;
            }
            for (int e_5 = 0; e_5 < 4; e_5++) {
                float kf = vec_load_9[quad_1 * 4 + e_5];
                kc[quad_1 * 4 + e_5] = ((valid_1 == 1) ? kf - vec_load_1_2[e_5] : 0.0f);
            }
        }
        unsigned int kw[2];
        float ksf_a[1];
        float max_1_1 = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fabsf(kc[0]), fabsf(kc[1])), fabsf(kc[2])), fabsf(kc[3])), fabsf(kc[4])), fabsf(kc[5])), fabsf(kc[6])), fabsf(kc[7])), fabsf(kc[8])), fabsf(kc[9])), fabsf(kc[10])), fabsf(kc[11])), fabsf(kc[12])), fabsf(kc[13])), fabsf(kc[14])), fabsf(kc[15]));
        uint16_t _cvt_satfinite_e4m3x2_f32_49;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_49) : "f"(0.0f), "f"(max_1_1 * 0.16666667f));
        uint16_t _e4m3x2_decode_50 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_49 & 0xFFu);
        uint32_t _f16x2_decode_50;
        float fp8_decode_1;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_50) : "h"(_e4m3x2_decode_50));
        uint16_t _f16_decode_50 = (uint16_t)_f16x2_decode_50;
        asm("cvt.f32.f16 %0, %1;" : "=f"(fp8_decode_1) : "h"(_f16_decode_50));
        float fdiv_rn_1 = __fdiv_rn(1.0f, fp8_decode_1);
        kc[0] = kc[0] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
        kc[1] = kc[1] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
        kc[2] = kc[2] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
        kc[3] = kc[3] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
        kc[4] = kc[4] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
        kc[5] = kc[5] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
        kc[6] = kc[6] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
        kc[7] = kc[7] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
        kc[8] = kc[8] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
        kc[9] = kc[9] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
        kc[10] = kc[10] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
        kc[11] = kc[11] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
        kc[12] = kc[12] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
        kc[13] = kc[13] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
        kc[14] = kc[14] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
        kc[15] = kc[15] * ((fp8_decode_1 != 0.0f) ? fdiv_rn_1 : 0.0f);
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(kw[0]) : "f"(kc[0]), "f"(kc[1]), "f"(kc[2]), "f"(kc[3]), "f"(kc[4]), "f"(kc[5]), "f"(kc[6]), "f"(kc[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(kw[1]) : "f"(kc[8]), "f"(kc[9]), "f"(kc[10]), "f"(kc[11]), "f"(kc[12]), "f"(kc[13]), "f"(kc[14]), "f"(kc[15]));
        ksf_a[0] = max_1_1 * 0.16666667f;
        unsigned int kstore[4];
        unsigned int _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, kw[0], 1);
        kstore[0] = _shfl_xor_sync_51;
        unsigned int _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, kw[1], 1);
        kstore[1] = _shfl_xor_sync_52;
        kstore[2] = kw[0];
        kstore[3] = kw[1];
        uint16_t _cvt_satfinite_e4m3x2_f32_53;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_53) : "f"(0.0f), "f"(ksf_a[0]));
        *reinterpret_cast<unsigned char*>(k_sf + (((head_3 * num_kblocks + (tok_1 >> 7)) * 128 + (tok_1 & 127)) * 8 + sub)) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_53 & 127));
        if (valid_1 == 1) {
            if ((sub & 1) == 1) {
                reinterpret_cast<int4*>(k4 + (((tok_1 * num_heads + head_3) * 64 + (sub - 1) * 8) / 4))[0] = reinterpret_cast<int4*>(kstore)[0];
            }
        }
    }
}

} // extern "C"
